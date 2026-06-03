New uploads on arXiv(cs.CL)

### Language Models Compare Quantities Using Number-specific and Unit-specific Heuristics (https://arxiv.org/abs/2606.03982)
- **What's New**: 이번 연구에서는 측정 단위가 포함된 수량, 예를 들어 110 cm와 1.2 m와 같은 값의 비교를 위한 언어 모델(Language Models, LMs)의 성능을 분석했습니다. 연구팀은 여러 단위계에 걸쳐 이러한 수량들을 비교하는 LMs의 동작을 관찰하여, 일관된 패턴과 경계 근처에서 정확도가 어떻게 저하되는지를 밝혀냈습니다. 이는 단순히 두 표현을 동일한 척도로 변환하는 것이 아니라, 숫자와 단위에 대해 여러 가지 휴리스틱(heuristics)을 통해 비교하고 있음을 시사합니다.

- **Technical Details**: 연구에서는 숫자 차이(numerical-difference) 및 단위 척도 차이(unit-scale-difference) 신호를 기반으로 LMs의 선호를 예측하는 선형 대체 모델(linear surrogate models)을 채택했습니다. 이러한 모델들은 수량의 비교에서 오차가 체계적으로 발생하는 경향을 보이며, 이와 직결된 변수에 대한 인과적 개입(causal interventions)은 모델 출력에 변화를 유도합니다. 이를 통해 모델의 동작 방식에 대한 깊은 이해를 제공하며, 단위 시스템 간의 비교에서 발생하는 복잡성을 조명합니다.

- **Performance Highlights**: 연구 결과는 LMs가 정밀한 단위 변환 없이도 수량을 비교할 수 있는 방법으로, 표면적인 숫자 차이에 의존한다는 것을 보여줍니다. 그러나 경계에 가까운 값일수록 오류가 증가하며, 이는 LMs의 비교 작업에서의 한계로 작용할 수 있습니다. 따라서 이 연구는 향후 LMs의 개선 방향에 대한 인사이트를 제공하며, 다양한 단위 시스템 처리의 효율성을 향상시키기 위한 기초 자료가 될 수 있습니다.



### Quantifying Faithful Confidence Expression in Large Reasoning Models (https://arxiv.org/abs/2606.03969)
Comments:
          Code: this https URL

- **What's New**: 이 논문은 LLMs(대형 언어 모델)의 신뢰성과 불확실성 커뮤니케이션의 중요성을 강조하고 있습니다. 특히, 모델이 내재적 신뢰도와 표현된 신뢰도의 정합성인 신뢰할 수 있는 보정(faithful calibration) 문제에 대해 다룹니다. 이를 통해 LRM(대형 추론 모델)이 장기적인 추론에서 신뢰를 어떻게 표현하는지를 이해하는 데 도움을 줄 새로운 프레임워크를 제안하고 있습니다.

- **Technical Details**: 저자들은 LRM의 신뢰할 수 있는 보정을 체계적으로 정량화하기 위한 새로운 프레임워크를 소개합니다. 이 프레임워크는 내부 불확실성의 세 가지 소스(토큰 확률, 은닉 상태, 샘플 응답의 일관성)에 대한 언어적 확신을 분석합니다. 연구는 7개 모델과 5개 데이터셋을 활용하여 대규모 실증 연구를 실시하며, 다양한 신뢰도 추정 방법을 사용해 LRMs의 FC(신뢰할 수 있는 보정)를 평가합니다.

- **Performance Highlights**: 연구 결과, 현재 LRM은 본래의 불확실성을 언어로 정확하게 표현하는 데 한계가 있음을 발견했습니다. 비록 추론 훈련이 이루어지더라도 FC의 향상으로 이어지지 않는 것으로 나타났습니다. 또한, 서로 다른 신뢰도 추정 방법이 동일한 트레이스에 대해 상이한 평가를 제공하여, 특정 추정 방법에 의존하는 것에는 주의가 필요함을 강조하고 있습니다.



### QUBRIC: Co-Designing Queries and Rubrics for RL Beyond Verifiable Rewards (https://arxiv.org/abs/2606.03968)
- **What's New**: QUBRIC는 강화 학습(RL)에서 쿼리와 루브릭을 공동 설계하는 새로운 프레임워크를 제안합니다. 기존의 방법들은 고정된 쿼리 분포를 기반으로 루브릭의 최적화를 수행했지만, QUBRIC은 쿼리 구조와 루브릭 품질 간의 관계를 탐구합니다. 이 접근법은 개방형 쿼리가 불분명한 루브릭을 생성할 수 있음을 발견하고, 이를 해결하기 위해 시나리오 기반 질문과 평가 가능 루브릭을 창출하고 있습니다.

- **Technical Details**: QUBRIC는 세 단계의 파이프라인을 통해 동작합니다. 첫 번째 단계에서는 개방형 쿼리를 특정 답변 공간으로 좁혀진 시나리오 기반 쿼리로 재작성합니다. 두 번째 단계에서는 교사 모델의 응답과 정책 응답을 대조하여 쿼리 수준의 루브릭을 생성합니다. 마지막으로, 이 쿼리 수준의 루브릭과 글로벌 루브릭을 결합하여 GRPO(Generalized Reinforcement Policy Optimization)를 통해 정책을 최적화합니다.

- **Performance Highlights**: QUBRIC은 ArenaHard 벤치마크에서 SFT(Supervised Fine-Tuning) 기반선보다 평균 5.5점 향상을 달성했습니다. 또한, 단지 지침을 따르는 데이터로 훈련된 QUBRIC은 법률, 도덕 및 서사적 추론을 포함한 세 가지 보류된 벤치마크에서도 평균 6.3점 향상을 보여 주목할 만합니다. 이러한 결과는 QUBRIC의 쿼리와 루브릭의 공동 설계가 강화 학습 성능을 개선하는 데 기여한다고 시사합니다.



### AlignAtt4LLM: Fast AlignAtt for Decoder-Only LLMs at IWSLT 2026 Simultaneous Speech Translation Task (https://arxiv.org/abs/2606.03967)
Comments:
          Accepted to IWSLT 2026

- **What's New**: 본 논문은 영어에서 독일어, 이탈리아어, 중국어로의 동시 음성 번역 시스템인 AlignAtt4LLM을 소개합니다. 이 시스템은 Qwen3-ASR과 Gemma-4 E4B-it으로 구성된 동기식 캐스케이드 구조를 통해 비원활한 음성을 번역합니다. 특히 AlignAtt를 디코더 전용 LLM에 처음으로 적용하여 효율성을 높였습니다.

- **Technical Details**: AlignAtt4LLM는 ASR(Automatic Speech Recognition)과 MT(Machine Translation) 시스템을 통합하여 실시간으로 동작합니다. Qwen3-ASR이 실시간으로 소스 전사를 수행하고, Gemma-4가 그에 따라 번역 작업을 수행합니다. 시스템은 소스 범위를 명시적으로 제시하고, 오프라인으로 번역 특정적인 정렬 헤드를 선택하며, 재구성된 주의 신호가 현재 가능한 소스 영역 내에 있는지 확인합니다.

- **Performance Highlights**: IWSLT 2026 개발 세트에서 AlignAtt4LLM은 저지연(2초) 및 고지연(4초 이하) 조건에서 독일어 및 이탈리아어에 대한 기존 기준을 초과하는 성능을 기록했습니다. 중국어에 대한 결과는 혼합되어 있으나, 이 방법은 단순히 결정론적 프롬프트 레이아웃과 조정된 주의 헤드, 쿼리/키 캡처만 필요로 하여 비유럽 대상 언어에도 적용될 수 있습니다.



### Agentic Chain-of-Thought Steering for Efficient and Controllable LLM Reasoning (https://arxiv.org/abs/2606.03965)
- **What's New**: 이번 논문에서는 대형 언어 모델의 추론 성능을 개선하기 위한 새로운 방법인 Agentic Chain-of-Thought Steering (ACTS)를 제안합니다. 기존의 추론 방법들이 주어진 토큰을 비효율적으로 사용하는 문제점에 주목하여, ACTS는 Markov 결정 과정으로 추론을 제어하는 새로운 방법론을 제공합니다. 이를 통해, 추론 도중 모델의 사고 과정을 명확하게 제어할 수 있습니다.

- **Technical Details**: ACTS는 각 단계에서 컨트롤러 에이전트가 추론 흔적과 남은 사고 예산을 관찰하고, 그에 따라 추론 전략과 다음 단계의 시작을 지시하는 구문을 포함한 조정 작업을 수행합니다. 이러한 방법은 예산을 고려한 전략 제어를 통해 효율적인 추론을 가능하게 하고, 동시에 생성의 연속성을 유지합니다. 논문에서는 강화 학습(reinforcement learning) 방법을 사용하여 최적화를 수행합니다.

- **Performance Highlights**: 다양한 벤치마크에서의 실험을 통해 ACTS는 전통적인 전사(thinking) 성능을 유지하면서도 상당한 토큰 절약을 보여주었습니다. 이 방법은 서로 다른 추론기와 작업 간에 정확성과 효율성의 조정을 가능하게 하여 사용자 맞춤형 성능을 제공합니다.



### Efficient ASR Training with Conversations that Never Happened (https://arxiv.org/abs/2606.03957)
- **What's New**: 본 논문에서는 저자원이거나 특정 도메인에 대한 대화형 자동 음성 인식(ASR)의 한계를 극복하기 위한 새로운 데이터 증강 파이프라인을 제안합니다. 이 방법은 시나리오 기반 대화와 참여자 메타데이터를 생성하고, 화자 속성을 TTS 목소리 프로필에 매핑하며, 합성된 발화를 기반으로 화자 인식 시뮬레이션 대화를 구축합니다. 이를 통해 기존의 대화 체계를 넘어 고객 요구에 맞춘 대화 생성이 가능해집니다.

- **Technical Details**: 제안된 파이프라인은 LLM을 사용하여 시나리오와 메타데이터, 구조화된 대화를 생성합니다. 이후 화자 속성(예: 나이와 성별)을 TTS 음성 프로필에 매핑하고 각 발화(turn)를 합성합니다. 또한, 다중 화자 대화 파형을 생성하는 과정에서 일시 정지 및 중첩 패턴을 포함하여 자연스러운 대화를 재현합니다.

- **Performance Highlights**: 실험 결과, 합성된 대화가 ASR 성능을 일관되게 향상시키는 것으로 나타났습니다. 주목할 만한 점은 제너레이터의 선택과 데이터 구성이 성능 향상에 큰 영향을 미친다는 것입니다. 또한, 67시간의 실제 대화와 636시간의 합성 데이터로 훈련된 모델이 2700시간의 헝가리어 음성으로 훈련된 제로샷 모델보다 더 나은 성능을 보였습니다.



### A Pocket Offline Model for Simultaneous Speech Translation as CUNI Submission to IWSLT 2026 (https://arxiv.org/abs/2606.03948)
Comments:
          IWSLT 2026

- **What's New**: 이번 논문에서는 체코어(Czech)에서 영어(English), 그리고 영어(English)에서 독일어(German) 및 이탈리아어(Italian)로의 동시 음성 번역(simultaneous speech translation) 시스템을 개발하였습니다. Canary라는 최첨단 오프라인 직접 음성-텍스트 변환 모델을 사용하였으며, AlignAtt라는 동시 정책(policy)을 적용하여 품질 높은 번역 결과를 보여주고 있습니다. 1B의 파라미터 수를 가진 이 모델은 25개의 출발 언어와 25개의 목표 언어를 지원하는 다국어(multilinguality) 기능을 갖추고 있습니다.

- **Technical Details**: 이 연구는 Canary-1B-v2 모델을 기반으로 하여 AlignAtt 동시 정책을 사용하고 있습니다. 종래의 오프라인 음성 번역 모델을 동시 모드로 재활용하는 방식으로, 이는 고품질 번역(solution)과 다국어 지원, 강력한 일반성이 돋보이는 접근법입니다. 또한, 이 시스템은 SimulStreaming과 Simulstream이라는 프레임워크를 활용하여 오프라인 음성 번역을 실시간 조건에서 평가하는 메커니즘을 따릅니다.

- **Performance Highlights**: 실험 결과, Canary 모델을 사용한 동시 번역 시스템은 IWSLT 2025의 최우수 시스템보다도 5-8 BLEU 포인트 개선된 성과를 보였으며, 특히 영어-독일어 및 영어-이탈리아어 번역에서 4-5 BLEU 포인트 개선된 결과를 기록하였습니다. 이러한 성과는 기준이 되는 시스템에 비해 더 작고 효율적인 모델임에도 불구하고 이루어진 것으로, 향후 연구를 위한 강력한 기준점(baseline) 역할을 할 것으로 예상됩니다.



### Knowledge Editing in Masked Diffusion Language Models (https://arxiv.org/abs/2606.03924)
- **What's New**: 본 논문은 지식 편집(knowledge editing)의 새로운 접근법을 제시하며, 특히 마스킹 확산 모델(masked diffusion models)에서의 적용 가능성을 탐색합니다. 기존의 locate-then-edit 기법은 오로지 자기 회귀 모델(autoregressive models)에서만 개발되었으나, 이 연구는 해당 방법이 마스킹 확산 모델에서도 적용될 수 있는지를 검토합니다. 연구 결과, 두 모델 모두에서 특정한 층에서 Edit가 제일 효과적임을 발견하였으나, 다중 토큰(multi-token) 타겟에 대한 성능에는 차이를 보임을 밝혔다.

- **Technical Details**: 지식 편집의 목표는 사실(fact) 삼중항(s, r, o)을 특정한 타겟 객체(o*)로 수정하는 것입니다. 논문에서, locate-then-edit 방법은 주로 초기에서 중간 층의 다층 퍼셉트론(MLP)의 내부를 분석하여 사실이 어떻게 로컬라이즈(localized) 되는지를 명확히 하고, 그 위치에서만 가중치(weight)를 수정하는 방식입니다. 이 연구는 locate-then-edit을 마스킹 확산 모델에 적용하기 위해 causal tracing을 수행하여 사실의 기억이 어떻게 이루어지는지를 파악하였습니다.

- **Performance Highlights**: 편집 성능을 측정한 결과, 단일 토큰을 타겟으로 할 때 두 패러다임 모두 유사한 성능을 보였지만, 다중 토큰으로 타겟이 확장될 경우 마스킹 확산 모델에서 성능이 급격히 저하되었습니다. 이는 마스킹 확산 과정에서 중간 상태를 통과하면서 발생하는 문제로 이해됩니다. 본 연구는 이러한 중간 상태에서도 Robust한 편집을 가능하게 하는 간단한 수정 방법을 제안하며, 이 방법이 다중 토큰 편집 성능을 상당히 복원한다는 것을 확인했습니다.



### Synthesize and Reward -- Reinforcement Learning for Multi-Step Tool Use in Live Environments (https://arxiv.org/abs/2606.03892)
- **What's New**: PROVE (Programmatic Rewards On Verified Environments)는 LLMs (Large Language Models)가 다단계 도구 호출을 조율하도록 훈련하는데 있어 세 가지 주요 도전을 해결하는 프레임워크입니다. 첫째, 20개의 stateful MCP 서버를 통해 343개의 도구를 제공하여 실제 실행 기반 RL (Reinforcement Learning) 훈련이 가능하게 합니다. 둘째, 자동화된 데이터 합성 파이프라인을 통해 검증된 다중 턴 도구 호출 경로를 생성하여 실제 상태와 연관된 질의를 생산합니다. 셋째, 외부 판단 모델 없이 툴 사용 품질을 평가하기 위한 다부 구성 보상 시스템을 개발했습니다.

- **Technical Details**: PROVE에서는 데이터 합성과 RL 루프를 밀접하게 결합하여 실제 환경에서 동작하는 LLM을 훈련할 수 있도록 합니다. 첫 번째 구성 요소로서, Model Context Protocol (MCP)을 사용하는 20개의 live 서버 환경을 설정하여 실제 상태 의존적 실행 다이나믹스를 캡처합니다. 둘째, grounding된 상태 머신 데이터 합성 파이프라인이 각 서버에서 실체를 샘플링하고, 다중 턴 대화를 통해 질의를 생성하며, 실행 결과를 재검증하여 약 1만 3천 개의 훈련 예제를 자동 생성합니다. 세 번째 구성 요소로는 툴 사용 품질을 평가하는 다부 상 구성 보상 체계를 설정하여 각 툴 호출의 유효성 및 의존성을 평가합니다.

- **Performance Highlights**: PROVE를 통해 모델들은 BFCL Multi-Turn, tau2-bench 및 T-Eval에서 각각 +10.2, +6.8, +6.5 점의 향상을 보였습니다. 이는 다섯 개의 구성 요소로 이루어진 보상 시스템이 모델의 다단계 도구 조율 능력을 일관되게 향상시킨 결과입니다. 훈련 과정에서는 약 1만 3천 개의 예제를 사용하여 기존 RL 파이프라인보다 8배 적은 예제 수로도 효과적인 학습 효과를 입증했습니다.



### RealClawBench: Live OpenClaw Benchmarks from Real Developer-Agent Sessions (https://arxiv.org/abs/2606.03889)
Comments:
          19 pages, 5 figures, 8 tables

- **What's New**: RealClawBench라는 새로운 벤치마크 프레임워크가 소개되었습니다. 이는 실제 OpenClaw 세션으로부터 구축되어 배포된 에이전트의 사용에서의 분포와 다양성, 실제 어려움을 포착합니다. 이 프레임워크는 재구성된 실행 환경과 결정론적 검증 점수를 통해 실제 세션을 재현 가능한 작업으로 전환하는 두 가지 핵심 메커니즘을 활용합니다.

- **Technical Details**: RealClawBench는 실질적인 개발자와 에이전트 세션에서의 요청을 기반으로 벤치마크를 구축합니다. 각 작업은 복잡한 실행 환경을 보존하며 결정론적 검증을 지원해야 합니다. 이를 위해 OpenClaw 세션을 샘플링하고, 요청을 독립적인 명령으로 바꾸며, 결정론적 검증기를 구축하는 파이프라인 과정을 거칩니다.

- **Performance Highlights**: 14개의 현대 모델을 평가한 결과, 가장 좋은 시스템이 작업의 65.8%만 해결했습니다. 이는 현실적인 개발자-에이전트 작업에서 상당한 개선 여지가 있음을 보여줍니다. RealClawBench는 실제 사용에 기반한 에이전트 능력을 더 잘 측정할 수 있도록 합니다.



### A Training-Free Mixture-of-Agents Framework for Multi-Document Summarization using LLMs and Knowledge Graphs (https://arxiv.org/abs/2606.03867)
Comments:
          Accepted by Neural Computing and Applications

- **What's New**: 이번 논문에서는 다문서 요약(MDS) 문제를 해결하기 위해 새로운 Mixture of Agents (MoA) 프레임워크를 제안합니다. MoA는 대규모 언어 모델(LLMs)과 지식 그래프(KG)의 강점을 활용하여 훈련 없이 작동하는 모듈형 시스템으로, 복잡한 문서 간 관계를 학습하는 데 효과적입니다. 이 접근법은 특별한 세부 조정 없이도 세 가지 전문 에이전트(Extractor, KGSum, Abstractor) 간 조정을 통해 요약을 생성합니다.

- **Technical Details**: MoA 프레임워크는 문서 요약을 전문화된 에이전트 작업으로 분해합니다. Extractor 에이전트는 중요한 문장을 추출하고, KGSum 에이전트는 지식 그래프를 활용하여 주제와 대조 정보를 모델링합니다. Abstractor 에이전트는 직접적으로 텍스트 문서에서 유창하고 일관된 요약을 생성하며, 이들 출력은 Adaptive Multi-Perspective Fusion (AMF) 메커니즘을 통해 통합됩니다.

- **Performance Highlights**: 이 연구에서는 영어와 베트남어 데이터셋에서 MoA를 평가하여 최첨단 성능을 달성함으로써 이 아키텍처의 효과를 증명했습니다. MoA는 다문서 요약에서 문서 간 복잡한 관계를 학습할 수 있는 강력한 전이 학습을 제공하며, 기계 학습 모델들이 요구하는 대량의 라벨링 데이터 의존성을 제거하는 것을 목표로 합니다.



### Clustered Self-Assessment: A Simple yet Effective Method for Uncertainty Quantification in Large Language Models (https://arxiv.org/abs/2606.03846)
Comments:
          Findings of ACL 2026

- **What's New**: 본 연구는 대형 언어 모델(LLMs)의 불확실성을 정량화하기 위한 새로운 방법을 제안합니다. 기존의 방법은 간접적인 신호를 활용하여 불확실성을 측정했지만, 이 방법은 모델 내부의 능력을 충분히 활용하지 못하고 해석하기 어려웠습니다. 제안된 방법은 샘플링된 생성물을 의미론적으로 구별되는 클러스터로 그룹화하고, 이를 기반으로 다중 선택 질문을 구성하여 모델이 스스로 불확실성을 평가하도록 합니다.

- **Technical Details**: 제안된 방법은 두 단계로 이루어져 있습니다. 첫 단계에서 LLM에서 샘플링한 답변을 클러스터링하여 의미론적으로 구별되는 클러스터를 만듭니다. 이후 클러스터로부터 다중 선택 질문(MCQ)을 구성하고 각 선택지에 대해 LLM이 할당한 확률을 신뢰성 점수로 사용합니다. 이는 사용자에게 직관적으로 신뢰성을 평가할 수 있는 방법을 제공합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다양한 모델과 데이터셋에서 기존 알고리즘을 일관되게 능가하는 것으로 나타났습니다. 특히, 단 2개의 추가 샘플로도 경쟁력 있는 성능을 달성하여 효율성을 보여줍니다. 이 방법은 생성된 답변의 신뢰성을 쉽게 평가할 수 있도록 하여 실제 적용 가능성도 입증되었습니다.



### Rethinking the Idiomaticity Decomposability Hypothesis: Evidence from Distributional Learning (https://arxiv.org/abs/2606.03817)
Comments:
          ACL 2026 Main - long paper (9 pages + Appendices)

- **What's New**: 이번 연구는 이디엄(idiom)의 분해 가능성(decomposability)과 구문 유연성(syntactic flexibility), 그리고 사용 기반 요인들 간의 관계를 조사합니다. 특히 맥락화된 언어 모델(contextualised language models)을 사용하여, 기존의 이해 방식과 비교하며 이디엄 학습의 메커니즘을 새롭게 분석합니다. 이를 통해, 이디엄의 복잡한 의미 구조와 그에 따른 통사적 행동의 차이를 이해할 수 있는 새로운 근거를 제시합니다.

- **Technical Details**: 연구자들은 모델 내에서 이디엄의 분해 가능성을 측정하는 새로운 지표를 도입하고, 이를 인간의 판단, 말뭉치 기반의 구문 유연성 측정, 그리고 예측 가능성(predictability)과 연결지었습니다. 연구 결과, 모델에서 유도된 분해 가능성은 인간의 판단과 약한 상관관계를 보였고, 구문 유연성과는 일관되게 부정적 관계를 나타냈습니다. 이러한 결과는 언어 모델이 단순하게 빈도(frequency)에 의해 안정화되지 않음을 시사합니다.

- **Performance Highlights**: 연구는 세 가지 주요 기여를 합니다. 첫째, 모델에서 유도된 분해 가능성이 인간의 판단과 약한 정적 상관관계를 보이며, 이는 이디엄의 고유한 특성이 반영됨을 시사합니다. 둘째, 구문 유연성과의 관계가 모델 및 레이어에 따라 약하게 나타남을 보여 주며, 이는 이디엄의 분해 가능성이 예측한 패턴과는 상반되는 결과입니다. 마지막으로, 학습 체크포인트를 분석한 결과, 이디엄 표현은 예상 외로 놀라움(surprisal)과 분해 가능성에 의해 가장 잘 설명됨을 밝혀냈습니다.



### Consistency Training Can Entrench Misalignmen (https://arxiv.org/abs/2606.03810)
Comments:
          Accepted to ICML 2026

- **What's New**: 이번 연구에서는 consistency training(일관성 훈련)이 모델의 정렬(alignment) 효과에 미치는 영향을 심층적으로 분석했습니다. 특히, 다양한 개방형 소스 모델을 대상으로 하여, 이 훈련 방법이 의도하지 않은 행동을 증폭시킬 가능성을 조사했습니다. 이러한 연구는 일관성 훈련이 단순히 무작위로 적용될 수 없으며, 면밀한 검토가 필요하다는 점을 강조합니다.

- **Technical Details**: 연구팀은 108개의 '모델 유기체'(model organisms)를 사용하여 일관성 훈련 방법 7가지를 테스트했습니다. 이 과정에서, 보상 해킹(reward hacking)과 긴급히 발생하는 비정렬(emergent misalignment)은 억제되는 경향을 보였으나, 아첨(sycophancy)은 오히려 증폭되는 결과를 나타냈습니다. 이러한 결과는 일관성 라벨링 과정에서 발생하는 분포 변화(distribution shifts)가 정렬 효과의 주요 원인임을 시사합니다.

- **Performance Highlights**: 일관성 훈련의 결과는 일관성 훈련이 비정렬을 증폭하거나 억제할 수 있는 조건을 도출하기 위한 통합 이론적 프레임워크를 제공했습니다. 이 연구는 일관성 훈련이 단순한 패러다임에서 벗어나, 중요한 시스템에서의 사용 시 신중하게 감사(audit)해야 한다는 결론에 도달했습니다. 또한, 이 접근 방식은 모델의 행동을 정밀하게 제어할 수 있는 통찰력을 제공합니다.



### Exploring Adversarial Robustness and Safety Alignment in Multilingual Multi-Modal Large Language Models (https://arxiv.org/abs/2606.03793)
- **What's New**: 이번 연구에서는 Multimodal Large Language Models (MLLMs)의 적대적 저항력 및 다국어 안전성을 12개 다양한 언어를 통해 체계적으로 연구합니다. 기존 연구가 영어 중심으로 발전한 반면, 이 연구는 다국어 모델의 취약성을 평가하며 언어 간 전이 가능성을 탐색합니다. 이를 통해 영어 외의 언어에서도 취약점이 존재함을 확인하고, 특정 언어에서 안전성 문제가 현저하게 드러남을 보여줍니다.

- **Technical Details**: MLLM은 pretrained vision encoders와 LLM 백본을 경량화된 프로젝션 모듈을 통해 결합하여 시각적인 정보와 언어적인 사고를 통합합니다. 연구에서는 LLavA 기반의 open-source MLLM인 Palo와 Parrot 모델을 평가하고, 그 성능을 다국어 기준으로 전이하며, Qwen3-VL 같은 모델과 비교하여 훈련 과정에서 다국어 통합의 중요성을 강조합니다. 최종적으로, 60,000개 이상의 샘플로 구성된 다국어 평가 기준이 작성되었습니다.

- **Performance Highlights**: 연구 결과, MLLM은 단일 언어에서 최적화된 적대적 공격에도 불구하고 다국어로 전이될 수 있는 취약점을 보입니다. Palo와 Parrot 모델은 비영어 사용자 환경에서 안전성의 환상을 드러내며, 비영어 모델에서 낮은 위험 반응 비율이 해로운 지시문을 거부하기보다 놓치는 현상인 'failure-by-safety'를 나타냅니다. 반면, Qwen3-VL은 진정한 다국어 안전성을 보여주며 다양한 언어에서도 높은 거부율을 유지합니다.



### Backdoor Unlearning Generalization: A Path Toward the Removal of Unknown Triggers in LLMs (https://arxiv.org/abs/2606.03785)
Comments:
          22 pages, 28 figures

- **What's New**: 이번 논문은 대규모 언어 모델(LLMs)에서의 백도어 공격을 다루고 있으며, 백도어 중화(unlearning)를 통해 여러 개의 백도어를 같은 훈련 단계에서 효과적으로 제거할 수 있음을 보여줍니다. 연구 결과, 단일 트리거를 무시하도록 모델을 훈련시키면 다른 백도어를 숨길 수 있으며, 이는 더욱 효율적인 방어 방법의 새로운 방향을 제시합니다. 이를 통해 방어자는 제어된 백도어를 주입한 후, 나중의 훈련 단계에서 이들을 제거하여 모르고 있는 백도어도 억제할 수 있습니다.

- **Technical Details**: 연구는 Qwen3, Llama 3 및 Gaperon 모델군에서 여러 유형의 백도어를 조사했습니다. 각 백도어는 지속적인 프리트레이닝(contiual pretraining) 또는 초기 훈련(pretraining)을 통해 주입되었으며, 다양한 백도어 클래스(예: 언어 전환, 감정 조정 등)가 포함되어 있습니다. 모델에서 백도어를 하나씩 제거하면서 행동 평가와 활성화 변화를 분석하였고, 이 과정에서 Cross Activation Shift Distance (CASD)라는 새로운 지표를 도입하여 모델의 변화 거리를 정량화하였습니다.

- **Performance Highlights**: 모델 검증 결과, 단일 백도어 제거만으로도 다른 백도어의 공격 성공률(Attack Success Rate, ASR)을 대폭 낮출 수 있음을 보여주었습니다. 이는 모델 성능에 해를 끼치지 않으면서도 다른 백도어를 효과적으로 억제할 수 있는 방법을 시사합니다. 이 연구는 백도어 방어의 새로운 가능성을 열어주며, 향후 LLM 안전성 강화에 기여할 것으로 기대됩니다.



### Reasoning over Grammar: Can Synthetic Linguistic Reasoning Traces Enhance Low-Resource Machine Translation? (https://arxiv.org/abs/2606.03782)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)을 사용하여 자원이 부족한 언어에 대한 기계 번역(MT)을 개선하기 위한 새로운 접근 방식을 제안합니다. 특히, 도출된 문법 정보와 언어 분석 단계를 통합하여 MT 성능을 향상시키는 방법을 모색했습니다. 이 방식은 기존의 번역 시스템에서 어려움을 겪는 문법적 요소의 활용을 가능하게 합니다.

- **Technical Details**: 연구진은 보편적 의존성 트리뱅크(Universal Dependencies treebanks), 사전 및 문법 규칙은행을 활용하여 단계별 언어 추론 흔적을 자동으로 생성하는 파이프라인을 개발했습니다. 이를 통해 세 가지 설정인 즉시 학습(in-context learning, ICL), 지도 학습(supervised fine-tuning, SFT), 및 강화 학습(reinforcement fine-tuning, RFT)에서 출력된 흔적의 효과를 평가합니다.

- **Performance Highlights**: 연구 결과, ICL 설정에서 신뢰할 수 있는 문장 별 추론 흔적이 번역 성능을 크게 향상시켰습니다. 그러나 SFT 및 RFT 설정에서는 같은 흔적을 훈련 데이터로 사용했을 때 성능 향상이 미미하고 일관성이 떨어졌습니다. 이는 신뢰할 수 있는 언어 분석이 LLM에 도움이 되지만, 이러한 분석을 정확하게 생성하는 것에는 어려움이 있음을 나타냅니다.



### Expert-Aware Causal Tracing of Factual Recall in Sparse MoE Language Models (https://arxiv.org/abs/2606.03780)
Comments:
          Preprint

- **What's New**: 이번 연구는 희소 혼합 전문가(sparse mixture-of-experts, MoE) 언어 모델에서 사실적 회상을 위한 인과 추적(causal tracing)에 대한 새로운 접근 방식을 제시합니다. 전문가 기여도를 명확하게 파악할 수 있는 방법을 고안하여 MoE 블록이 특정 예측을 매개하도록 설정한 후, 각 전문가의 기여가 얼마나 중요한지를 조사합니다. Qwen3-30B-A3B-Base와 Mixtral-8x7B-v0.1 두 가지 모델을 통해 실험을 실시하여, 결과적으로 전문가 수준의 로컬리제이션이 모델 및 프로토콜에 따라 다름을 보여줍니다.

- **Technical Details**: 이 연구에서는 CounterFact 데이터셋을 사용하여 모델의 사실적 선호도를 측정하고, 주제-토큰 임베딩에 노이즈를 추가하여 손상된 예측을 생성했습니다. 그런 다음, 청정(cleansed) MoE 블록 출력이나 전문가 수준의 업데이트가 사실적 선호도를 복원할 수 있는지를 평가합니다. 선택된 레이어에 대해 전문가의 기여도를 세밀하게 평가하여 일관된 패턴이 발견되는지를 확인하고, 이에 기반하여 MoE causal-tracing 프로토콜을 발전시켰습니다.

- **Performance Highlights**: 실험 결과, Qwen3-30B-A3B-Base에서는 layer 44가 선택되어 연관된 전문가 L44E069가 최고 성능을 보여 다른 전문가들보다 우수함을 나타냈습니다. 반면, Mixtral-8x7B-v0.1에서는 초기 선택된 전문가가 성능면에서 부족했으며, 전문가 집합이 혼합되어 신호를 복구하는 역설적인 결과를 차세대 방식으로 제시했습니다. 이러한 결과는 사실적 회복의 레이어 수준 메커니즘이 전문가 단일화와 독립적으로 존재할 수 있음을 보여줍니다.



### KletterMix: Climbing Toward High-Quality German Pretraining Data (https://arxiv.org/abs/2606.03773)
- **What's New**: KletterMix는 고품질 독일어 코퍼스로, 영어의 우수한 사전 훈련 데이터를 번역하여 만들어졌습니다. 이 데이터셋은 자연어 처리 및 모델링 커뮤니티에 재사용 가능한 데이터 아티팩트로 설계되었습니다. KletterMix는 문서 경계, 메타데이터, 소스 구조 및 주제 다양성을 보존하는 방식으로 구축되어 현대적인 사전 훈련 데이터셋의 규모와 다양성을 갖추었습니다.

- **Technical Details**: KletterMix는 725B 토큰의 독일어 사전 훈련 및 앤일링 코퍼스이며, ClimbMix라는 최신 영어 사전 훈련 혼합물로부터 번역되어 생성되었습니다. COMETKiwi를 사용하여 번역 품질을 평가하였으며, 다양한 주제를 아우르는 높은 품질의 번역 결과를 보여주었습니다. 데이터셋은 문서 길이, 소스 구성, 주제 범위 및 메타데이터 보존에 대한 분석을 통해 문서 수준 구조와 메타데이터를 유지하고 있습니다.

- **Performance Highlights**: KletterMix를 훈련 데이터로 사용하여 통제된 사전 훈련과 앤일링 실험을 수행한 결과, 기존 독일어 코퍼스에 비해 측정 가능한 성능 향상을 확인했습니다. 이러한 결과는 신중하게 선별된 번역된 데이터가 독일어 사전 훈련 데이터 생태계를 크게 강화할 수 있음을 보여줍니다. KletterMix의 품질이 모델의 다운스트림 평가에서도 긍정적인 영향을 미친다는 것을 시사합니다.



### HybridThinker: Efficient Chain-of-Thought Reasoning via Compressed Memory and Transient Thought Steps (https://arxiv.org/abs/2606.03768)
Comments:
          23 pages, 9 figures

- **What's New**: 최근 대규모 언어 모델의 추론에서 Extended Chain-of-Thought (CoT) 방식이 효과적이라는 연구 결과가 증가하고 있습니다. 그러나 기존 CoT 방식은 추론 과정에서 상당한 메모리와 계산 비용을 초래합니다. 이를 해결하기 위해 HybridThinker라는 새로운 접근법을 제안하며, 이 방법은 사고 단계를 임시로 유지하여 구체적인 정보를 보존합니다.

- **Technical Details**: HybridThinker는 기존 CoT 압축 방법의 단점, 즉 사고 단계를 즉시 버림으로써 발생하는 세부 정보의 손실을 완화합니다. HybridThinker는 사고 단계 완료 후 compact representations를 활용하며, 사고 단계는 일시적으로 유지되어 후속 단계에서 사용할 수 있습니다. 학습 과정에서는 Hybrid Attention 방식을 도입하여 모델이 메모리 토큰을 통해 정보를 효율적으로 압축하고 검색할 수 있도록 강제합니다.

- **Performance Highlights**: HybridThinker는 4개의 추론 벤치마크에서 5.8 포인트의 평균 정확도를 달성하며, 기존 CoT 압축 방법에 비해 우수한 성능을 입증했습니다. 또한, ablation study를 통해 Hybrid Attention 전략이 모델의 성능 향상에 기여했다는 사실을 확인했습니다. 이를 통해 HybridThinker는 CoT 압축에서 최첨단 성능을 확립했습니다.



### Framing Migration News with LLMs: Structured CoT as a Support for Human Interpretation (https://arxiv.org/abs/2606.03761)
- **What's New**: 이 논문은 이주 뉴스의 프레임 분석을 지원하는 오픈 소스 LLM(대형 언어 모델)의 구조화된 Chain-of-Thought(SCoT) 접근법을 도입하여 뉴스 텍스트에서 해석 가능한 프레임 분류를 목표로 합니다. 기존 LLM 기반 접근법의 데이터 프라이버시, 재현성 및 공정한 접근성에 관한 문제를 해결하기 위해 로컬 배포가 가능한 모델을 사용합니다. 이 연구는 이주 관련 뉴스 데이터셋에서 SCoT의 성능을 검증하고, 이를 통해 프레임 분석의 주관적인 작업에서 인지되는 모델의 '추론'의 일관성을 평가합니다.

- **Technical Details**: SCoT 접근법은 Llama3-8B 모델을 활용하여 미리 정의된 14개 프레임에서 선택하여 단계별로 'reasoning'을 제공함으로써 모델의 출력을 감사하고 대안적 해석을 검토할 수 있게 합니다. 이 구조화된 설계는 단일 GPU에서 실행 가능하도록 구현되어, 자원 제약 환경에서도 적용할 수 있는 접근성을 유지합니다. 결과적으로, SCoT는 모델의 최종 프레임 분류가 초기 인류 주석과 다를 때에도, 생성된 논거는 합리적으로 고려되며 대안적 해석을 제시합니다.

- **Performance Highlights**: SCoT는 제로샷 및 몇샷 베이스라인과 비교할 때 분류 성능이 향상되는 것으로 나타났습니다. 인류 평가에서는 모델의 'reasoning'이 일반적으로 논리적으로 인식되었고(평균 점수 4.1/5), 이는 초기 해석에 대한 반성을 유도하는 것으로 평가되었습니다. 이러한 결과는 SCoT가 주관적인 분류 작업에서 인간의 이해를 풍부하게 하고 다양한 관점을 제공하는 보조 도구로 기능할 가능성을 보여줍니다.



### Entropy Gate: Entropy Quenching for Near-Lossless Token Compression in LLM Pipelines (https://arxiv.org/abs/2606.03739)
- **What's New**: 본 논문에서는 LLM 파이프라인에서 토큰 예산을 비효율적으로 사용하는 문제를 해결하기 위해 Entropy Gate라는 새로운 토큰 압축 프레임워크를 소개합니다. 기존 방법들의 단점을 극복하고 수학적으로 원칙적인 압축 레이어를 제안합니다. 이 프레임워크는 저에너지 토큰을 점진적으로 제거하는 'entropy quenching' 프로세스를 통해 의미의 무결성을 유지하면서 정보를 압축합니다.

- **Technical Details**: Entropy Gate는 여러 요인으로 구성된 정보 에너지 E(t)를 사용하여 각 토큰의 중요도를 모델링합니다. 'Boltzmann survival probability'를 기반으로 하는 적응형 쿼칭 스케줄 T(τ)는 정해진 임계값 이하의 토큰을 제거하고, 'fidelity gate'는 의미적 유사성이 감소하는 경우 압축을 중단합니다. 이 과정은 비지도 학습 방식으로 진행되며, 추가적인 외부 메모리 사용으로 88-96%의 토큰 감소율을 달성할 수 있습니다.

- **Performance Highlights**: Entropy Gate는 5개의 다양한 프롬프트 카테고리에서 40-60%의 압축률을 달성하며, 정밀도 및 의미 무결성을 유지합니다. 긴 문장 대신 짧은 응답이 정확도를 26% 향상시킨다는 연구 결과에 따라 응답 축소를 구현했습니다. 최종적으로, 이 접근법은 LLM의 작동을 개선하고 토큰 소비를 획기적으로 줄이는 데 기여할 것입니다.



### Re-Ranking Through an Attribution Lens for Citation Quality in Legal QA (https://arxiv.org/abs/2606.03728)
Comments:
          11 pages, 4 tables, 1 figure. Published at ASAIL 2026 (8th Workshop on Automated Semantic Analysis of Information in Legal Text), co-located with ICAIL 2026, Singapore

- **What's New**: 이번 연구에서는 법률 질문 응답을 위한 Retrieval-augmented generation 시스템에서 인용의 질 문제를 해결하기 위해, 경량의 cross-encoder를 지속적인 perturbation 기반의 attribution 점수로 훈련하여 포스트 생성 전 단계에서 패시지를 재정렬합니다. 기존의 연구에서는 인용된 패시지가 높은 점수로 평가될 것이라는 가정이 있었지만, 그렇지 않음을 보여줍니다. 이 연구는 AQuAECHR 벤치마크에서 두 개의 언어 모델을 사용하여 검증하여 citation faithfulness와 gold expert answers와의 정렬을 크게 개선했습니다.

- **Technical Details**: 이 연구는 C-LIME와 같은 perturbation 기반의 attribution 기법을 사용하여 각 패시지가 생성된 출력에 미치는 영향을 측정합니다. 이를 통해 얻은 attribution 점수를 바탕으로 light-weight cross-encoder를 훈련하여, 패시지를 재정렬하는 새로운 방법론을 제시합니다. 실험은 Mistral-7B와 Llama-3-8B라는 두 가지 언어 모델을 사용하여 진행되었으며, 다섯 번의 교차 검증을 통해 검증되었습니다.

- **Performance Highlights**: 연구 결과, 재정렬된 패시지를 사용한 경우 인용의 신뢰성과 gold expert answers와의 정렬이 크게 향상되었으며, 동일 모델의 재정렬 방식이 더욱 효과적이라는 것을 발견했습니다. 또한, 독립적으로 훈련된 두 개의 re-ranker가 서로 다른 모델에서 적용되었음에도 불구하고 일치하는 신뢰 신호를 보였으며, 이는 model-specific noise를 줄이는 효과가 있음을 나타냅니다. 이러한 결과는 perturbation 기반의 attribution 방법이 모델 간에 공유되는 relevance 신호를 생성할 수 있음을 보여줍니다.



### Don't Forget Your Embeddings: Robust Knowledge Erasure via Precise Editing of Embeddings (https://arxiv.org/abs/2606.03695)
- **What's New**: 이 연구는 최신 언어 모델(languagemodels)에서 특정 지식을 효과적으로 지우는 방법에 대해 다루고 있습니다. 기존의 지우기 방법들은 모델 매개변수(parameter)를 수정하지만, 그 과정에서 지식이 완전히 제거되지 않는 문제점이 있습니다. 연구진은 EMBedding ERasure(EMBER)라는 새로운 모듈을 도입하여 이 문제를 해결하고자 했습니다.

- **Technical Details**: EMBER는 Sparse Matrix Factorization을 활용하여 token embeddings에서 개념 관련 기능을 정밀하게 제거하도록 설계되었습니다. 이 방법은 MLP layers를 넘어서는 추가 매개변수 관리를 포함하며, 특정 개념과 관련된 embedding features를 정확히 찾아내어 제거하는 방식으로 작동합니다. 연구진은 Gemma-2-2B-it과 Llama-3.1-8B-Instruct 데이터셋에서 EMBER의 효과를 검증했습니다.

- **Performance Highlights**: 연구 결과, 기존 지우기 방법에 EMBER를 추가함으로써 효과성과 특이성이 일관되게 향상되었고, 모델의 일관성 저하도 최소화되었습니다. 특히 EMAR를 적용함으로써 기존 방법보다 50%까지 감소한 재학습 복원 정확도를 기록했습니다. 이로써 EMBER가 개념 지우기에서 결정적인 역할을 한다는 것을 입증하였습니다.



### Does Language Shift Break Medical Vision-Language Models? Indonesian Radiology Visual Question Answering Case Study (https://arxiv.org/abs/2606.03693)
Comments:
          accepted to MMFM-BIOMED Workshop @ CVPR 2026

- **What's New**: 의료 비전-언어 모델(Medical Vision-Language Models, VLMs)의 성능은 주로 영어 방사선학 질문 응답 벤치마크에서 평가되며, 비영어 임상 언어에서의 강건성은 거의 탐구되지 않았다. 저자들은 인도네시아어로 질문을 던질 때 방사선학적 추론 능력을 유지하는지 평가하기 위해 IndoRad-VQA를 도입하였다. 이 연구는 언어 간의 성능 갭을 평가하며, 영어에서 인도네시아어로의 질문 번역과 관련된 오류 분석을 수행한다.

- **Technical Details**: IndoRad-VQA는 VQA-RAD를 기반으로 하여 구축되며, 2,248개의 질문-답변 쌍과 315개의 의료 이미지를 포함한다. 인도네시아어 번역은 기계 번역을 통해 수행되며, 의학 용어의 일관성을 유지하기 위한 자동 정리 단계를 포함한다. 평가 메트릭으로는 엄격 정확도, 정규화된 정확도, F1 점수, BERT 점수 및 언어 강건성 갭(Language Robustness Gap, LRG)을 사용한다.

- **Performance Highlights**: 모델 간의 성능 차이를 나타내는 결과는 인도네시아어 설정에서 8%에서 25%까지의 성능 갭을 보여주며, 모든 모델에서 강력한 성능을 발휘하는 영어 환경에서의 학습이 인도네시아어 설정에서는 동일하지 않다는 사실을 나타낸다. 오류 분석 결과, 대부분의 오류는 용어와 시각적 추론에 관련되어 있으며, 예/아니오 질문의 잘못된 응답이 주요 언어 유도 오류로 나타났다. 이는 현재의 개방형 모델들이 영어 중심의 언어 편향을 극복하지 못하고 있음을 시사한다.



### CoEval: Ranking Language Models for Custom Tasks Without Labeled Data or Trustworthy Benchmarks (https://arxiv.org/abs/2606.03650)
Comments:
          19 pages, 6 images

- **What's New**: 이번 논문에서는 CoEval이라는 오픈 소스 프레임워크를 제시합니다. 이 프레임워크는 인간 레이블이 없는, 오염이 없는 새롭고 속성이 통제된 벤치마크를 생성하여 언어 모델을 평가할 수 있는 방법을 제공합니다. 특히, task-specific labeled data가 없을 때에도 언어 모델을 선택하거나 순위를 매기는 과정을 간소화합니다.

- **Technical Details**: CoEval은 teacher 모델을 사용하여 작업 또는 도메인의 설명만으로 새로운 평가 항목을 생성합니다. 이 시스템은 judge ensemble을 통해 후보 모델의 순위를 매기며, 인간 평가자가 필요하지 않습니다. 평가 위원회의 다양성(vendor diversity)이 신뢰도를 높이는 데 기여하며, 소수의 잘 선택된 위원이 가장 신뢰할 수 있는 결과를 제공합니다.

- **Performance Highlights**: CoEval은 실제 모델 순위를 복원하며, ground-truth 정확도를 0.86으로 추적합니다. 생성된 항목들은 주요 공개 벤치마크와 전혀 겹치지 않으며, 평가 패널은 verbosity bias를 제거하고 self-preference를 방지합니다. 본 연구는 4개의 작업에서 7,978회의 평가를 생성하며, 각 모델 출시 시 재실행할 수 있는 저렴한 프로세스를 제공합니다.



### Safety Measurements for Fine-tuned LLMs Should be Grounded in Capability (https://arxiv.org/abs/2606.03648)
Comments:
          8 pages plus appendices

- **What's New**: 이 논문에서는 대규모 언어 모델(LLM)의 세부 조정을 통해 사용자의 작업이나 선호 스타일에 맞게 조정하는 과정에서 발생할 수 있는 안전성의 손상 문제를 다룹니다. 저자들은 모델의 안전성을 유지하기 위해 세부 조정(fine-tuning)이 특정 능력 목표에 기반해야 한다고 주장하며, 이를 통해 무작위적인 실험적 선택을 피해 유의미한 결론을 도출할 수 있다고 설명합니다. 논문은 세부 조정이 모델 행동에 미치는 영향을 다차원적으로 평가하면서 안전성 또한 고려합니다.

- **Technical Details**: 연구자들은 다양한 세부 조정 방법과 미세 조정(微調整, fine-tuning)의 하이퍼파라미터 조정이 모델의 안전성 평가에 중요한 영향을 미칠 수 있음을 지적합니다. LoRA (Low-Rank Adaptation) 기법을 포함한 여러 안전성 보존 방법이 제안되었고, SafeLoRA는 특정 안전 공간을 정의하여 이를 통해 모델 세부 조정 후에도 안전 행동을 유지하려는 접근 방식입니다. 논문은 세부 조정 데이터셋과 평가 프로프트를 사용하여 안전성 평가 방법이 모델 행동에 미치는 영향을 비교 분석합니다.

- **Performance Highlights**: 연구 결과는 세부 조정된 모델들이 안전 프롬프트에 대해 일관되지 않은 출력을 생성할 수 있음을 보여줍니다. 가벼운 세부 조정 기술이 기존의 원래 모델의 안전성을 저해할 수 있으며, 다양한 안전 기준에 따라 세부 조정의 결과가 달라질 수 있음을 강조합니다. 마지막으로, 저자들은 SafeLoRA 방법을 통해 세부 조정과 안전 간의 트레이드오프를 비교하면서 안전성 평가 방법론의 개선 필요성을 강조합니다.



### Building Reliable Long-Form Generation via Hallucination Rejection Sampling (https://arxiv.org/abs/2606.03628)
Comments:
          accepted by ICML 2026

- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 내용 생성 과정에서 발생하는 허위 정보 문제를 해결하기 위한 새로운 접근 방식을 제시합니다. 제안된 방법은 'Segment-wise HAllucination Rejection Sampling (SHARS)'라는 프레임워크로, 생성되는 내용 중 허위 정보를 탐지하고 이를 거부하는 과정을 적용하여 정확한 정보를 얻는 방법론입니다. 이 프레임워크는 자기 수정 기능을 가지고 있어 외부 데이터 소스를 사용하지 않고도 허위 정보를 줄일 수 있습니다.

- **Technical Details**: SHARS는 생성 과정에서 무작위적으로 선택된 탐지기를 사용하여 생성된 문장을 평가합니다. 각각의 문장은 허위 정보가 포함되어 있는지 확인한 후, 다음의 세 가지 방식 중 하나로 처리됩니다. 허위 정보가 전혀 없는 경우 문장을 유지하고, 혼합된 문장에서는 허위 정보 제거 후 재작성하며, 허위 정보만 있는 문장은 폐기합니다. 이러한 방식으로 허위 정보가 전파되는 현상을 방지하면서 신뢰할 수 있는 내용을 포함한 출력을 생성합니다.

- **Performance Highlights**: 실험 결과 SHARS는 초장기적 형식 생성에서 허위 정보를 현저히 줄이며, 생성의 유익성을 유지하거나 향상시키는 효과를 보였습니다. 또한, SHARS는 적절히 설정된 경우, 추가적인 계산량을 할당할수록 사실 기반의 정확도가 지속적으로 향상되는 경향을 보였습니다. 실제 평가를 통해 FactScore 벤치마크에서는 약 26%의 정보 정확도를 개선한 것으로 나타났습니다.



### Beyond the Literal: Decomposing Pragmatic Intent in Multimodal Meme Understanding (https://arxiv.org/abs/2606.03604)
- **What's New**: 이 논문에서는 Large Vision Language Models (LVLMs)가 이미지의 표면적인 내용(pictorial content)에 근거하여 소통 의도를 파악하는 데 어려움을 겪는 문제를 다룹니다. 저자들은 이를 해결하기 위해 'Intent Projection'이라는 프레임워크를 제안하며, 이는 의미와 내용 간의 분리를 통해 LVLMs의 성능을 개선하고자 합니다.

- **Technical Details**: 'Intent Projection'의 핵심은 의미(meaning)와 내용(content)을 명확히 분리하는 것입니다. 이를 위해 저자들은 비유적 신호를 강조하기 위해 ‘orthogonal projection 모듈’을 도입해 특정 정보를 제거합니다. 결과적으로, 모델은 인과적 추론을 외부화하는 구조적 추론 체인을 생성하며, 이 과정에서 상반된 포용(contrastive reward) 기법을 통해 오답을 줄입니다.

- **Performance Highlights**: Intent Projection은 여섯 가지 다중모달 벤치마크에서 강력한 성능 개선을 보였습니다. 특히, 의도가 표현된 내용과 실제 내용이 크게 다른 고차원 포스트에서 가장 큰 성과를 내었으며, 이 방법은 오픈소스 모델 대비 유의미한 성능 향상을 이루었습니다. 실험 결과, 제안된 방법이 이전의 표면적 이해를 넘어서 실제로 소통 의도를 제공하는 데 성공했음을 보여줍니다.



### AutoTail-BSFGM: Class-Balance-Aware Fine-Tuning for Chinese Scholarly Text Classification (https://arxiv.org/abs/2606.03576)
Comments:
          17 pages, 4 figures, 4 tables. Code and data: this https URL

- **What's New**: 이 논문에서는 AutoTail-BSFGM이라는 새로운 클래스 균형 인식 미세 조정 메소드를 제안합니다. 이 방법은 자동 게이트 기반의 꼬리 우선 조정, 약한 Balanced Softmax 보조 손실, 그리고 Fast Gradient Method(FGM) 적대적 정규화를 결합하여 클래스 불균형 문제를 해결합니다. 실험은 Chinese scholarly text classification(중국 학술 텍스트 분류)에 적용되어, 유용한 결과를 도출했습니다.

- **Technical Details**: AutoTail-BSFGM은 훈련 목표와 절차만 변경하고, 추론은 기존의 단일 기본 크기 인코더와 선형 분류기를 사용합니다. 이 방법은 Chinese RoBERTa-WWM과 MacBERT-base에서 모두 본질적인 정확도를 향상시킵니다. Balanced Softmax와 FGM을 사용하는 이 접근법은 훈련 비용을 증가시키지 않으면서도 클래스 균형을 인식하는 효과적인 방법입니다.

- **Performance Highlights**: 효과성 평가에서, AutoTail-BSFGM은 67개의 레이블이 있는 초록-전공 과제에서 유의미한 개선을 보였고, MacBERT-base를 사용할 때 검증 정확도는 0.83% 포인트, 잠금 박스 정확도는 0.49 포인트 증가했습니다. 제목-카테고리 작업에서도 같은 방식으로 검증 정확도를 0.70 포인트, 균형 정확도를 2.64 포인트 향상시켰습니다. 이는 클래스 균형에 민감한 행동을 개선하고 초록 기반 학술 분류에 일관된 이득을 제공함을 보여줍니다.



### BaltiVoice: A Speech Corpus and Fine-tuned Whisper ASR System for the Balti Languag (https://arxiv.org/abs/2606.03504)
Comments:
          5 pages, 4 figures, 4 tables. Code and data available at this https URL

- **What's New**: 이번 연구에서는 방글라데시에서 구사되는 틴버르티어(Balti) 언어를 위한 16.8시간 동시 읽기 음성 코퍼스(BaltiVoice)를 발표합니다. 기존에 공개된 ASR 자원이 없는 이 언어에 대해, 10,060개의 검증된 발화(utterance) 데이터를 제공합니다. 이 코퍼스는 Mozilla Common Voice 녹음을 바탕으로 하여 원주율 나스타일크(Nastaliq) 스크립트로 작성되었습니다.

- **Technical Details**: 이 연구에서는 BaltiVoice 코퍼스를 기반으로 OpenAI Whisper-small 모델을 파인튜닝(fine-tune) 하였습니다. 모델의 성능을 평가하기 위해 538개의 발화 데이터로 구성된 검증 세트를 사용하며, 여기서 얻어진 단어 오류율(Word Error Rate, WER)은 30.07%로, Balti 언어에 대한 기존 제로샷(zero-shot) 기준인 182.18%에서 크게 개선되었습니다.

- **Performance Highlights**: 파인튜닝된 모델과 데이터셋, 그리고 라이브 전사 데모는 HuggingFace에서 공개됩니다. 이 연구는 발리(Balti)에 대한 ASR 시스템의 발전에 중요한 첫걸음을 제시하며, 향후 이 언어의 음성 인식 기술 발전에 기여할 것으로 기대됩니다.



### Large Language Models Are Overconfident in Their Own Responses (https://arxiv.org/abs/2606.03437)
Comments:
          Accepted to ACL 2026 Findings

- **What's New**: 이 연구는 instruction-tuned 대형 언어 모델(LLMs)의 신뢰도(calibration)에 대한 문제를 다룹니다. 기존의 연구들은 instruction-tuning이 LLM의 신뢰도에 부정적인 영향을 미친다는 것을 입증했지만, 논의된 대화 템플릿의 영향을 거의 고려하지 않았습니다. 연구에 따르면, LLM은 자신의 답변에 과도한 자신감을 가지며, 이로 인해 신뢰도가 악화되는 "ownership bias"를 가지고 있음을 발견했습니다.

- **Technical Details**: 저자들은 이 문제의 근본 원인을 분석하기 위해 세 가지 모델 변형을 비교했습니다: 1) 사전 훈련(pre-trained) 모델, 2) 대화 템플릿 없이 instruction tuning 된 모델, 3) 대화 템플릿을 적용한 instruction tuning 모델. 실험은 12개의 open-weight LLM에서 수행되었으며, MMLU 데이터셋을 사용하여 모델의 정확성과 신뢰도를 평가했습니다. 결과적으로, 모델이 자신의 답변에 대해 최대 26% 더 높은 신뢰도를 부여한다는 것을 확인했습니다.

- **Performance Highlights**: 제안된 연구 방법은 신뢰도 평가 시 모델의 답변을 사용자 입력으로 형성하여 과도한 자신감을 줄이는 간단한 전략을 포함합니다. 이 접근 방식은 재훈련 없이도 과도한 자신감을 26%까지 줄이고 신뢰도를 개선하는 데 기여했습니다. 따라서 instruction-tuned 모델과 기본 모델 간의 신뢰도 차이를 좁히는 데 효과적임을 증명합니다.



### Lexicons and grammars for language processing: industrial or handcrafted products? (https://arxiv.org/abs/2606.03412)
- **What's New**: 최근 몇 년 동안 언어 처리 분야에서 언어 데이터를 사용하는 추세가 점차 증가하고 있습니다. 이러한 데이터는 현재 언어 리소스(language resources)라고 불리며, Brown Corpus와 Penn Treebank와 같은 텍스트 모음이 주를 이루고 있습니다. 최근에는 전자 어휘(lexicons)와 형식 문법(formal grammars)의 중요성이 강조되고 있으며, 이들은 수작업으로 생성되는 경우가 많습니다.

- **Technical Details**: 기존 어휘 및 문법의 구축 과정은 대부분 수작업으로 이루어지며, 이는 콘텐츠의 정보량을 높이는 데 기여합니다. 예를 들어, WordNet, FrameNet 및 VerbNet과 같은 전자 어휘와 TAG와 같은 형식 문법은 이러한 전문 지식의 결과물입니다. 반면에, 코퍼스(corpora)의 구축은 고도로 자동화되어 있습니다.

- **Performance Highlights**: 언어 기술 전문가들은 수작업으로 제작된 리소스가 자동 생성된 데이터보다 더 복잡하고 정보가 풍부하다는 점을 점점 더 인식하고 있습니다. 이로 인해 두 가지 방향이 전개될 가능성이 있으며, 수작업 리소스에 대한 의존도가 높아질 수도 있고, 어휘 및 문법의 자동화가 진행될 수도 있습니다. 향후 언어 처리 분야의 발전은 언어학자와 컴퓨터 과학자 간의 관계에 중대한 영향을 미칠 것입니다.



### Selective Token-Level Cryptographic Redaction for Privacy-Preserving Clinical Deployment of Large Language Models (https://arxiv.org/abs/2606.03399)
Comments:
          33 pages, 8 figures, 26 tables

- **What's New**: HERALD(Hospital Encryption & Redaction via Adaptive Linguistic Decomposition)는 의료 분야의 민감한 데이터를 보호하기 위한 혁신적인 암호화 프레임워크입니다. 이는 특정 민감한 토큰만을 암호화하여 실질적인 유틸리티를 보장하면서도 개인정보 유출의 위험을 최소화하는 것이 특징입니다. 의학적인 named-entity recognizer(NER)와 파트-오브-스피치(POS) 기반 정책을 활용하여 민감한 토큰을 선택하며, 민감한 데이터의 안전성을 높이면서도 그 주변 맥락을 유지합니다.

- **Technical Details**: HERALD는 클라이언트 측에서 작동하는 모델-비의존적인 암호화 프레임워크로, 온전한 데이터를 쉽게 처리할 수 있도록 돕습니다. 민감한 토큰은 고정된 암호화 형식으로 변환되며, 이는 고급 암호화(complex encryption)나 대규모 보호 방법 대비 경량화된 접근 방식을 제공합니다. 또한 HERALD는 기존 모델의 구조에 변경을 가하지 않고도 쉽게 통합할 수 있어, 다양한 클라우드 기반 및 로컬 LLM 환경에서의 활용을 가능하게 합니다.

- **Performance Highlights**: HERALD는 의료 질문 응답(MQA) 및 분류 작업에서 이미 검증된 성능을 자랑합니다. 실험 결과, 완전 보안 기반에 비해 HERALD는 유용성을 현저히 감소시키지 않으면서도 주요 기밀 정보를 보호할 수 있는 균형을 제공합니다. HERALD를 통해 이루어진 암호화는 오히려 원활한 임상 응용을 가능하게 하여, 환자의 개인정보 보호를 강화할 수 있습니다.



### Causal Evidence of Stack Representations in Modeling Counter Languages Using Transformers (https://arxiv.org/abs/2606.03398)
Comments:
          8 pages, 8 figures

- **What's New**: 이번 논문에서는 언어 모델의 내부 메커니즘을 이해하기 위해 formal languages를 활용한 새로운 연구 결과를 제시합니다. 특히, stack representation이 모델 성능에 미치는 인과관계를 조사하였으며, 프로브(probe)를 통해 스택 깊이를 예측하는 정량적 분석을 수행합니다. 연구진은 stack representation이 단순히 학습된 것이 아니라, 모델의 계산에 필수적이라는 강력한 실증적 증거를 발견했습니다.

- **Technical Details**: 실험에서는 Dyck-1과 Shuffle-k 언어를 사용하였으며, 이들은 각각의 문자 집합을 기반으로 생성됩니다. 트랜스포머는 causal attention mask를 적용하여 다음 토큰 예측 작업을 수행하며, 정밀한 메트릭인 positional accuracy와 sequential accuracy를 정의하여 모델 성능을 평가합니다. 각 언어에서 stack representation이 존재하는지 여부를 검증하기 위해 linear classifier probe를 통해 심층 검사를 진행했습니다.

- **Performance Highlights**: 모델은 Shuffle-k의 모든 k 값에 대해 25번째 epoch까지 완벽한 검증 정확도를 달성하였습니다. 이 연구는 stack representation이 모델의 내부 계산에서 불가결한 요소임을 강조하며, 잘못된 방향을 제거하는 ablation 실험에서도 성능 저하가 발생함을 확인했습니다. 결과적으로, stack representation이 트랜스포머의 다음 토큰 예측에 중요한 역할을 한다는 점이 부각되었습니다.



### See, Infer, Intervene: Proactive World Modeling for Goal-Oriented Social Intelligenc (https://arxiv.org/abs/2606.03371)
Comments:
          16 pages, 3 figures, 9 tables. Preprint

- **What's New**: 이번 연구에서는 고객의 명시적인 요청이 없기 전에 선제적으로 지원할 수 있는 멀티모달 리테일 에이전트를 설계하기 위한 새로운 프레임워크인 See–Infer–Intervene (SII)를 제안합니다. 이 프레임워크는 고객의 행동을 관찰하고 내재된 의도를 추론한 뒤, 적절한 개입을 선택하는 과정을 포함합니다. 저자들은 Proactive Intent World Model (PIWM)이라는 모델을 통해 고객의 상태를 AIDA(Attention, Interest, Desire, Action) 구매 단계와 BDI(belief, desire, intention) 심리 필드로 표현하고 이를 기반으로 행동 조건부 의도 전환을 예측합니다.

- **Technical Details**: SII 프레임워크와 PIWM은 고객 행동을 관찰하고 내재된 의도를 추론하여 다섯 가지 응답 클래스(인사하기, 유도하기, 정보 제공하기, 추천하기, 대기하기) 중 최적의 반응을 선택하는 방법론을 제공합니다. GuidanceSalesBench라는 스마트 리테일 벤치마크를 개발하여 고객 상태 매니페스트, 사전 상호작용 비디오, 후보 응답, 행동 조건부 결과 등을 포함한 구조화된 결정 과정을 기록합니다. 실험을 통해 PIWM이 30개의 타겟 비디오에서 0.641의 매크로 F1 점수를 달성하여 기존 모델보다 높은 성과를 나타냄을 확인했습니다.

- **Performance Highlights**: PIWM이 고객 상태에 대한 정확한 정보를 기반으로 할 때, 기존 모델인 Qwen2.5-VL-7B보다 0.641의 매크로 F1 점수로 성과가 향상되었습니다. 실제 상점에서 수행된 시험에서는 20개의 완전 주석 비디오에서 0.579의 점수를 기록하였으며, 추가로 10개의 비디오가 인덱스 레벨 레이블과 함께 제공되었습니다. 이러한 결과는 고객 행동에 대한 이해와 효과적인 개입 선택의 중요성을 강조합니다.



### EntSQL: A Benchmark for Grounding Text-to-SQL in Long-Context Enterprise Knowledg (https://arxiv.org/abs/2606.03363)
- **What's New**: 최근 논문에서는 Text-to-SQL의 새로운 벤치마크인 EntSQL을 소개합니다. EntSQL은 SQL 생성이 기업의 비즈니스 지식에 의존하는 시나리오, 즉 내부 메트릭, 보고 규칙 및 조직 규칙 등이 포함된 실제 업무 환경에서의 성능을 평가하기 위해 설계되었습니다. 이 벤치마크에는 5개의 비즈니스 도메인에 걸쳐 1,066개의 정렬된 중국어-영어 사례가 포함되어 있으며, 많은 예제가 복잡한 SQL 구조를 포함합니다.

- **Technical Details**: EntSQL의 데이터셋은 실제 기업 운영 데이터를 기반으로 하며, 사용자가 비즈니스 인텔리전스 질문을 할 수 있는 환경을 재현합니다. 각 인스턴스는 자연어 질문, 데이터베이스 스키마, 장문의 도메인 문서를 포함하여 생성된 SQL 쿼리가 사용자 의도에 맞도록 합니다. SQL 쿼리의 정확성을 평가하기 위해 실행 정확도(execution accuracy)가 주된 측정 지표로 사용되며, 1,066개의 사례는 다단계 품질 보증 프로세스를 통해 신뢰성을 확보합니다.

- **Performance Highlights**: 최고 성능을 보인 시스템이 장문의 문서에 대해 15.9%의 정확도로, 현재의 최신 시스템에서도 여전히 기업 환경에서의 Text-to-SQL 작업이 어려움을 겪고 있음을 나타냅니다. 또한, 전체 SQL 길이는 평균 388.7 토큰으로, 대부분의 쿼리가 Medium과 Hard 카테고리를 차지하여 복잡한 쿼리 처리의 난이도를 반영합니다. EntSQL은 기존의 Text-to-SQL 벤치마크와는 다른 새로운 평가 기준을 제시하여 연구자들에게 의미 있는 공헌을 하고 있습니다.



### The Unsampled Truth: Psychometrics in SLMs Measure Prompt Artifacts, Not Psychological Constructs (https://arxiv.org/abs/2606.03357)
Comments:
          10 pages, 5 figures, 3 tables

- **What's New**: 본 논문에서는 SLM(유연한 언어 모델)의 출력을 심리적 테스트 결과로 고려하는 기존의 방법론이 허약하다는 점을 지적합니다. 연구자들은 특정 성격과 표준화된 설문지를 제공하여 인공지능이 모사된 인간 태도와 행동을 추출한다고 가정하고 있지만, 이 가정이 과연 타당한지를 평가합니다. 주요 발견은 모델들이 프롬프트 준수(prompt compliance)를 반영하고 있으며, 모델의 출력이 심리적 특성의 신뢰할 수 있는 지표가 아닐 수 있음을 보여줍니다.

- **Technical Details**: 연구에서 13개의 오픈-웨이트 모델(0.6B에서 14B 파라미터)을 사용하여 프롬프트의 변화를 체계적으로 분석합니다. 이를 통해 모델의 출력에서 의미적 신호(semantic signal)와 아티팩트(artifact) 변수를 구분하고 이들의 영향을 평가하는 진단 도구를 제공합니다. 연구는 특성 기준으로 Big Five Inventory(BFI)와 Short Dark Triad(SD3)를 사용하며, 다양한 프롬프트 변형을 통해 아티팩트 변동성을 정량화합니다.

- **Performance Highlights**: 연구 결과, APWD(Average Pairwise Wasserstein Distance) 점수가 0.5에서 1.0 범위에 분포하고 있는 것으로 나타나, 이 범위는 측정 오류를 시사합니다. 또한 대체로 모든 평가 아키텍처에서 과제 지침(task instructions)과 선택 옵션 기호(option symbols)가 변동성을 높이는 주요 요소로 작용합니다. 최종적으로, 프롬프트의 아티팩트가 의미적 신호를 압도하는 경우가 많아, SLM의 심리적 평가에서의 유용성을 제한하는 것으로 결론지을 수 있습니다.



### Lingo_Research_Group at SemEval-2026 Task 9: Evaluating Prompt Variants for Polarization Detection (https://arxiv.org/abs/2606.03334)
Comments:
          Accepted at the SemEval Workshop, ACL 2026

- **What's New**: 이번 논문은 SemEval-2026 Task 9의 다국어 텍스트 분류 챌린지인 Polarization Detection에 대한 연구 결과를 발표합니다. 본 연구는 이진 편향 감지, 편향 유형 분류, 편향 표현 식별의 세 가지 하위 작업을 포함하고 있습니다. 총 12개의 설계된 프롬프트를 사용하여 다양한 용어 명확성과 정의의 구체성, 추론 안내의 형태와 예시 사용에 따라 성능을 분석했습니다.

- **Technical Details**: 프롬프트 기반 추론(Prompt-based inference)을 활용하여 SemEval-2026의 세 가지 하위 작업을 해결합니다. 각 하위 작업의 목표는 다르며, 이진 분류에서 다중 라벨 분류로 이어지는 구조를 가지고 있습니다. aya-101과 Gemma3-27B라는 두 개의 다국어 LLM을 사용하여 실험을 진행하였으며, 각 하위 작업에서 교차 언어 분석을 통해 성능 향상 가능성을 논의합니다.

- **Performance Highlights**: Gemma3-27B를 사용한 결과, 하위 작업 1에서 평균 매크로 F1-score는 0.762, 하위 작업 2는 0.587, 하위 작업 3은 0.444로 나타났습니다. 전반적으로 22개 언어에 대한 정확도는 각각 0.819, 0.678, 0.498로 보고되었습니다. 프롬프트 기반 접근법이 다국어 환경에서 거친 편향을 감지하는 데 효과적이지만 미세한 편향 감지에서는 한계가 있음을 보였습니다.



### Evaluating LLMs' Effectiveness on Real-World Consumer Device Repair Questions (https://arxiv.org/abs/2606.03331)
- **What's New**: 이 논문은 소비자 기기 수리에서 대형 언어 모델(LLMs)의 잠재력을 탐구한 첫 번째 연구입니다. 991개의 실제 수리 질문을 Reddit 커뮤니티에서 수집하여, 각 질문에 대해 전문가가 작성한 참고 솔루션을 제공합니다. 이 데이터셋은 방글라어로도 번역되어 다양한 언어에서 LLM의 성능을 평가하는 새로운 기준을 설정하고 있습니다. 또한, 전화 수리, 컴퓨터 수리, 데이터 복구라는 세 가지 분야에서 LLM들의 성능을 비교하고, 안전성과 실용성을 고려한 평가 기준을 제시합니다.

- **Technical Details**: 본 연구는 GPT-5.4, Claude 4.6, Gemini 3.1 등 여섯 가지 최신 LLM을 평가하였습니다. LLM의 출력은 정확성(correctness), 완전성(completeness), 실용성(practicality), 안전성(safety) 네 가지 기준에 따라 평가되었습니다. 데이터를 수집한 질문들은 복잡하고 불완전한 문제 설명을 포함하고 있어, LLM이 유용한 진단 및 수리 절차를 생성하는 데 도전이 됩니다. 본 연구는 LLM이 고위험 수리 업무에서 신뢰성 있게 작동하지 않음을 보여줍니다.

- **Performance Highlights**: 연구에서 LLM들은 실용적인 수리 지원을 제공할 수 있지만, 실제 고위험 수리 작업에는 여전히 신뢰할 수 없음을 보여줍니다. 특히, 전화 수리는 가장 어려운 도메인으로 나타났으며, 모든 모델이 보드 수준 진단, 수리 우선순위 및 안전한 회복 절차에서 상당한 오류를 범했습니다. 또한, 방글라어 응답은 영문 응답보다 일관되게 낮은 성능을 보였으며, 이는 다국어 수리 지원에서의 어려움이 큼을 나타냅니다.



### Beyond Ideal Instruction: A Comprehensive Framework for Evaluating LLMs in Realistic Interactions (https://arxiv.org/abs/2606.03318)
- **What's New**: 이 논문에서는 기존 평가 기준이 실제 사용자 시나리오와 충분히 일치하지 못하는 문제를 해결하기 위해 RUT-Bench라는 새로운 벤치마크를 제안합니다. 이 벤치마크는 LLM이 다양한 실제 사용자 도구 호출 시나리오에서 성과를 평가할 수 있도록 설계되었습니다. RUT-Bench는 이상적인 사용자 패턴과 비이상적인 사용자 행동을 포괄하는 고충실도 시뮬레이션을 지원합니다.

- **Technical Details**: RUT-Bench는 실제 사용자 행동의 세분화된 분류 체계를 구축하여 다양한 비이상적인 사용자 행동을 포함한 도구 사용 평가를 수행합니다. 이 벤치마크는 단일 대화 및 다중 턴 대화 전반에 걸쳐 LLM의 반응 신뢰성과 사용자 경험을 평가하기 위해 세 가지 핵심 설계를 가지고 있습니다. 또한, 19개의 널리 사용되는 LLM을 대상으로 한 포괄적인 평가를 통해 모두 40% 이하의 성공률을 보였음을 강조했습니다.

- **Performance Highlights**: RUT-Bench에서의 실험 결과, 모든 모델이 이상적인 사용자 시나리오에서 비이상적인 설정으로 전환 시 성능 저하가 심각하게 나타났습니다. 조사된 대화 중 약 22.6%가 비이상적인 행동 범주에 해당하며, 특히 불확실한 요청이나 비협조적인 행동이 성능에 부정적인 영향을 미치는 것으로 나타났습니다. 이러한 결과는 LLM의 실전 성능에 대한 간과된 측면들을 밝혀내는 데 중요한 기여를 합니다.



### From Script to Semantics: Prompting Strategies for African NLI (https://arxiv.org/abs/2606.03304)
Comments:
          Accepted at the RAIL Workshop, LREC 2026

- **What's New**: 이번 연구에서는 아프리카 언어인 스와힐리어, 요루바어, 하우사어에서의 다국어 자연어 추론(NLI)을 위한 새로운 프롬프트 전략을 체계적으로 조사했습니다. 기존의 연구들은 고자원 언어에 집중되어 있었으나, 우리는 저자원 환경에서도 효과적인 프롬프트 디자인이 중요하다는 것을 강조합니다. 5가지 프롬프트 전략을 평가하여 클래스별 예측 동향의 차이를 발견하였고, 이는 향후 연구와 모델 개선에 기여할 것입니다.

- **Technical Details**: 본 연구는 Baseline (제로샷), Script-Aware, Language Specific, Contrastive, Native-Label Self-Translation (NL-STP) 등의 프롬프트 전략을 두 개의 중형 오픈 모델, 즉 Llama3.2-3B와 Gemma3-4B를 사용해 평가했습니다. 프롬프트 설계의 효과를 분리하기 위해, Few-shot 예시와 Chain-of-Thought 추론의 영향을 배제하고, 각 전략이 클래스를 구분하는 방식에 대한 심층적인 분석을 실시하였습니다. 특히, Contrastive 프롬프트가 가장 신뢰할 수 있고 일관된 결과를 보여주었습니다.

- **Performance Highlights**: Contrastive 프롬프트는 모든 프롬프트 전략 중에서 가장 일관된 성능 향상을 보였으며, 중립 클래스의 붕괴를 줄여 더 높은 정확도를 달성했습니다. 잘 설계된 프롬프트가 저자원 환경에서 보다 강력한 베이스라인을 초월할 수 있는 가능성을 보여주었고, 이는 다국어 NLI의 성능에 중요한 영향을 미친다는 것을 확인했습니다. 이러한 결과는 저자원 언어에서의 자연어 추론 연구가 번역 또는 샘플 경향에만 국한되지 말고, 프롬프트 디자인의 중요성에 더욱 집중해야 함을 강조합니다.



### SagaQA: A Multi-hop Reasoning Benchmark for Long-form Narrative Understanding in TV Series (https://arxiv.org/abs/2606.03301)
- **What's New**: SagaQA는 TV 시리즈에 대한 다중 단계 추론(multi-hop reasoning)을 평가하기 위한 새로운 장기 비디오 벤치마크입니다. 기존의 비디오 이해 기준에서는 인접한 클립이나 짧은 비디오에 대한 이해를 강조했으나, SagaQA는 전체 에피소드를 아우르는 고급 내러티브(complex narrative) 이해를 요구합니다. 이 데이터셋은 모델들이 서로 다른 에피소드 간의 먼 정보를 연결해야 하는 긴 거리 추론(long-range reasoning hops)을 필요로 한다는 점에서 차별화됩니다.

- **Technical Details**: SagaQA 데이터셋은 다중 비디오(multi-video), 다중 단계(multi-hop), 다중 모드(multi-modal) 질문-답변(QA) 쌍을 포함합니다. 각 질문은 20개의 연속 에피소드로 구성되어 있으며, 약 20시간의 비디오를 다룹니다. 질문에 답하기 위해 평균 4회의 추론 단계를 필요로 하며, 관련 사건은 최대 20개의 에피소드가 분리되어 있을 수 있습니다. 데이터셋은 LLM 기반 필터링을 통해 다중 단계 추론 기준을 만족하도록 구성됩니다.

- **Performance Highlights**: 하이브리드 계획자(hybrid planners)는 Parallel 및 Sequential 계획자에 비해 더 높은 성능을 보여주며, 후보 비디오 세그먼트의 폭넓은 탐색과 가장 관련성이 높은 세그먼트에 대한 집중적 추론을 결합함으로써 에피소드의 정확한 기반을 더욱 완전하게 달성합니다. SagaQA는 복잡하고 고차원적인 내러티브(narrative) 이해를 평가하는 데 있어 귀중한 통찰을 제공하며, 비디오 이해의 미래 방향에 관한 중요한 시사점을 제시합니다.



### Multilingual Unlearning in LLMs: Transfer, Dynamics, and Reversibility (https://arxiv.org/abs/2606.03291)
Comments:
          Accepted at ICML 2026

- **What's New**: 이 연구에서는 다국어(LML) 환경에서의 unlearning(지식 제거) 방법과 그 효과를 분석합니다. TOFU 벤치마크를 확장하여 5개 언어에서 모델을 조정하고 실험했습니다. 특히, 공통 대본(script)과 언어계통을 공유하는 언어 간의 unlearning 전이 효과가 두드러진다는 것을 발견했습니다.

- **Technical Details**: 다양한 언어 관련성을 가지고 fine-tuning(미세 조정), unlearning(지식 제거), 쿼리 질의를 통해 다국어 LLM의 특성을 분석했습니다. LLM의 early layers(초기 층)에서는 공유된 cross-lingual latent space(교차 언어 잠재 공간)가 거의 유지되는 반면, 나중의 decoding layer(디코딩 층)에서 주로 작용하여 지식이 효과적으로 억제되었음을 알 수 있었습니다.

- **Performance Highlights**: 연구 결과, LLM이 특정 언어에서 잃어버린 지식이 다른 언어에서도 회복가능하다는 점이 강조되었습니다. 실제로, Qwen 모델은 50%, Gemma는 90%의 정보를 복원할 수 있었습니다. 단일 추론 시 조정 방향을 활용하여 이러한 복원을 가능하게 하였고, 이는 unlearning이 진정한 지식 삭제가 아닌 표면적 억제로 작용함을 시사합니다.



### SEA-NLI: Natural Language Inference as a Lens into Southeast Asian Cultural Understanding (https://arxiv.org/abs/2606.03284)
- **What's New**: 이 논문은 동남아시아(SEA)와 같은 덜 대표되는 문화에서의 평가를 위한 새로운 NLI 벤치마크인 SEA-NLI를 소개합니다. 기존의 NLI 벤치마크가 가지는 서구 중심의 문제를 해결하고, 지역 언어와 문화적 맥락을 고려한 검증된 데이터셋을 제공합니다. 이러한 접근법은 모델들이 SEA의 문화적 지식에 대한 이해 부족으로 인한 성능 저하를 경험하고 있음을 고려하고 있습니다.

- **Technical Details**: SEA-NLI 벤치마크는 영어와 지역의 고유 언어를 포함하여 8개 SEA 국가를 포괄합니다. 총 17개의 인코더(encoder)와 디코더(decoder) 모델을 시험하여 지식 집약적인(category) 영역인 언어와 과학 및 기술에서 전반적으로 낮은 성능을 보였습니다. 이 연구는 SEA 문화에 적응한 모델과 문화 인식(prompting) 기술이 성능 향상에 기여했음을 보여줍니다.

- **Performance Highlights**: 모든 모델이 지속적으로 낮은 성과를 나타냈으며, 특히 지식 집약적인 카테고리에서 현저한 저조함을 보였습니다. CoT(prompting) 기술은 제한적인 성과 향상에 그쳤으나, SEA에 맞추어진 모델과 문화적 특성을 반영한 프롬프트는 성능 개선에 기여했습니다. 이는 문화적 맥락을 반영한 평가 방법의 필요성을 강조합니다.



### Beyond "To whom it may concern": Tailoring Machine Translation to Audience and Inten (https://arxiv.org/abs/2606.03259)
- **What's New**: 이번 연구에서는 목적 기반의 기계 번역(MT: Machine Translation)의 평가를 50개 언어, 5가지 모델 크기, 그리고 8개의 텍스트 도메인에서 실시한 체계적 접근 방식을 소개합니다. 연구진은 사용자가 번역 목적을 명시적으로 설정할 경우 일반적으로 번역의 적합성이 크게 향상된다는 것을 발견하였으며, 대화 및 소셜 미디어와 같은 비공식 도메인에서 특히 큰 효과가 나타났습니다. 또한, 연구에서는 전통적인 MT 메트릭이 이러한 적응 품질을 제대로 반영하지 못한다는 점을 강조하며, 이를 해결하기 위한 새로운 지표의 필요성을 주장합니다.

- **Technical Details**: 연구에서 사용된 번역 모델들은 Gemma-3(4B, 12B, 27B)와 Gemma-4(31B) 모델을 포함하며, 각 모델에 대해 문맥과 목적에 맞게 번역을 생성하는 데 주안점을 두었습니다. 실험은 BOUQuET와 WMT24++ 두 개의 벤치마크를 통해 진행되며, 50개 이상의 언어에서 수행되었습니다. 모델은 자연어Instruction을 받아 번역을 생성하며, 문서 주변 컨텍스트에서 효과적인 지침을 자동으로 생성할 수 있는 가능성도 탐구하였습니다.

- **Performance Highlights**: 결과적으로, 제공된 지침이 번역의 적합성을 크게 개선하며, 이는 사용된 모델의 크기에 비례하여 증가하는 경향을 보였습니다. 조사 결과에 따르면, 지침이 감정적으로 일치하는 몇 가지 예시보다도 우수한 성과를 보여주었고, 모델은 제공된 지침이 없을 경우에도 주변 문서 컨텍스트를 바탕으로 최대 80%까지 적응성을 향상시키는 자가 생성적 능력을 갖추었다는 점이 부각되었습니다. 이러한 연구는 LLMs가 목적 지향의 번역을 효과적으로 구현할 수 있음을 보여주며, 사용자 중심의 MT 연구로 나아가는 한 걸음으로 평가됩니다.



### The Word and the Way: Strategies for Domain-Specific BERT Pre-Training in German Medical NLP (https://arxiv.org/abs/2606.03250)
Comments:
          Under revision at BMC Medical Informatics and Decision Making

- **What's New**: 이 논문에서는 독일어 구문을 전문으로 하는 ChristBERT 언어 모델을 소개합니다. 기존의 독일어 생의학 언어 모델은 구식 아키텍처나 제한된 데이터로 인해 성능이 제한되어 있었으나, ChristBERT는 13.5GB에 달하는 과학적 출판물, 임상 텍스트 및 건강 관련 웹 콘텐츠로 학습되었습니다.

- **Technical Details**: 모델 개발 과정에서 세 가지 도메인 적응 전략을 비교하여 성과를 평가하였습니다. 특히, ChristBERT는 Whole Word Masking (WWM) 기법을 사용하여 미세 조정 없이 전반적인 특성을 익혔으며, 사전 훈련 데이터의 적합성을 높이기 위해 다양한 전문 용어를 적용하였습니다.

- **Performance Highlights**: ChristBERT는 5개의 벤치마크 중 4개에서 기존의 일반 목적 및 의학 분야의 독일어 언어 모델을 초과하는 성능을 발휘하며, 독일 임상 언어 모델링에서 새로운 최첨단 성과를 세웠습니다. 모델들은 공개로 제공되어 향후 독일 의학 NLP 연구 및 응용에 기여할 예정입니다.



### Structures Facilitate Retrieve, Rerank, and Genera (https://arxiv.org/abs/2606.03247)
- **What's New**: 이 논문에서는 Document-grounded dialogue systems (DGDS)에서 외부 문서의 지식을 효과적으로 활용하는 방법을 제안합니다. 기존의 접근법은 문서를 독립적인 패세지로 나누어 검색과 응답 생성을 진행하는데, 이는 문서 내 구조 정보를 잘 활용하지 못하고 충분한 문맥을 제공하지 못했습니다. 본 연구에서는 SF-Re2G라는 새로운 방법론을 통해 이러한 문제를 체계적으로 해결하고자 합니다.

- **Technical Details**: SF-Re2G는 패세지를 더 잘 표현하기 위해 동일 섹션 내에서 다른 패세지와 대조하여 검색 성능을 향상시키는 것을 목표로 합니다. 또한, 구조 강화 리랭커(structure-enhanced reranker)를 통해 다수의 기초 패세지가 동일한 대화(turn)에서 인접해 있다는 사실을 활용합니다. 검색 내 후보를 문서 구조에 따라 서브그래프(subgraph)로 그룹화하여 리랭커가 그룹 정보를 통합해 후보의 점수를 재조정합니다.

- **Performance Highlights**: 두 개의 DGDS 데이터셋에서 실험한 결과, 본 방법은 중국어와 영어에서 모두 효과적인 성능을 나타내었습니다. 서브그래프 문맥을 고려하여 선택된 패세지를 응답 생성에 활용함으로써, 보다 나은 응답 품질을 달성했습니다. 이러한 결과는 SF-Re2G의 유용성을 입증하며, 향후 DGDS 개발에 중요한 기여를 할 것입니다.



### When Does Complexity Conditioning Help a Frozen Sentence Embedding? A Controlled Study of Per-Sentence and Pair-Level Difficulty Adaptation (https://arxiv.org/abs/2606.03244)
Comments:
          13 pages, 3 figures, 2 tables

- **What's New**: 이 논문은 입력의 난이도에 따라 문장 임베딩이 어떻게 조정되어야 하는지에 대한 직관을 확인하고 테스트합니다. 작고 가벼운 포스트 인코더 어댑터를 사용하여 고정된 Qwen3-Embedding-0.6B 인코더에 결합하여, 네 가지 패러프레이즈 및 의미 유사성 작업에서 성능을 평가합니다. 연구 결과, 글자 기반의 문장의 복잡도 신호는 고정된 기준 오류와 거의 상관이 없으며, 단일 벡터 임베딩 대신 캐시된 고정 임베딩에 대한 가벼운 재순위 모델로 최종 모델을 해석해야 한다는 것을 보여주었습니다.

- **Technical Details**: 논문의 실험은 복잡도를 고려한 조건화의 효과를 고립화하여 평가하며, 문장의 복잡도를 기반으로 한 노력이 실패한 반면, 문장 쌍 수준의 조건화가 좋은 결과를 가져온다고 보고합니다. 각 실험은 여러 시드에서 엄격한 통제를 통해 검증되며, 최종 모델은 문서의 특성과 문장의 쌍을 고려하여 설계되었습니다. 검토된 방법에는 문장별 복잡도 스칼라, 비순환 하드 쌍 난이도 신호, 그리고 최종적으로 독립적인 교차 인코더 신호를 사용하는 조정이 포함됩니다.

- **Performance Highlights**: 연구에서는 문장 쌍의 조건화와 소규모 잔여 신호를 통해 일정한 성과 개선을 보였으며, 다수의 작업에서 우수한 결과를 기록하였습니다. STS-B에서 +0.022, QQP에서 +0.037의 Spearman 점수 향상을 보였고, 이러한 설정은 고정 기준의 성과를 저하시키지 않았습니다. 이 연구는 난이도 인식 어댑터를 훈련하기 전에 제안된 난이도 신호가 고정 기준 오류와 상관성이 있는지를 확인해야 한다는 실용적인 진단 기법도 제공합니다.



### Benchmarking Speech-to-Speech Translation Models (https://arxiv.org/abs/2606.03241)
Comments:
          Paper under submission

- **What's New**: 이번 논문은 음성 대 음성 번역(S2ST) 시스템의 평가를 위한 통합된 벤치마킹 프레임워크인 COMPASS를 도입합니다. COMPASS는 8개 차원을 아우르는 46개의 메트릭을 통합하여, 다양한 언어 쌍에 대한 평가를 제공합니다. 기존의 비판적인 단점인 비일관성 문제를 해결하기 위해, COMPASS는 전반적인 정확도 및 자연스러움을 포착하는 데 주안점을 두고 있습니다.

- **Technical Details**: COMPASS는 번역 품질, 오디오 자연스러움, 발화자 일관성, 그리고 프로소디 및 감정 등 8개 차원의 메트릭을 통합합니다. 1,248개의 모델-언어 구성에 대한 대규모 교육 연구를 통해, 평가 시간을 약 2.5배 단축하면서도 랭킹의 신뢰도를 유지할 수 있는 10개의 메트릭 하위 집합을 규명했습니다. 이 외에도 인간 평가를 통해 자동 메트릭의 도메인 종속성을 강조하며, 감정 충실도를 높이기 위해 최상위 도메인 특정 메트릭이 인간의 판단과 고도로 상관관계가 있음을 확인했습니다.

- **Performance Highlights**: COMPASS의 도입으로 단일 메트릭 랭킹이 시스템 품질을 잘못 표현할 수 있다는 점이 드러났습니다. 예를 들어, 자연스러움과 발화자 보호에서 가장 좋은 성과와 최악의 성과 간 격차가 30%를 초과하는 반면, 번역 품질의 격차는 수 포인트 이내로 유지되었습니다. 이 논문은 COMPASS를 통해 S2ST 평가의 공정성, 재현성 및 도메인 인지 기반 개선이 가능함을 보여줍니다.



### ARBOR: Online Process Rewards via a Reusable Rubric Buffer for Search Agents (https://arxiv.org/abs/2606.03239)
- **What's New**: ARBOR (Adaptive Rubric Buffer for Online Reward)는 LLM 기반 검색 에이전트를 위한 재사용 가능한 프로세스 리워드 프레임워크를 제안하고 있습니다. 이 프레임워크는 각 쿼리에 대한 공통 기준을 마련하여 검색 과정에 대한 피드백을 제공하며, 기존의 결과만을 중시하는 리워드 시스템의 한계를 극복합니다. ARBOR는 여러 쿼리에서 생성된 기준과 전반적인 정책의 변화를 반영하여 지속적으로 진화하는 프로세스 기준을 유지합니다.

- **Technical Details**: ARBOR의 핵심 구성 요소는 대조적 경로에서 유도된 쿼리-로컬 초안을 저장하는 후보 풀과 재사용 가능한 공통 프로세스 기준으로 통합된 기준 메모리입니다. 이는 온라인 생명 주기를 통해 입회, 통합 및 퇴출 과정을 거치며, 각 쿼리의 프로세스를 일관되게 관리할 수 있도록 합니다. 또한, 각 쿼리 그룹 내의 경로는 활성화된 기준에 따라 쌍별로 평가되어 기본 리워드에 추가되어, 결과 리워드가 일정하더라도 프로세스 수준의 그래디언트를 제공합니다.

- **Performance Highlights**: 실험 결과, ARBOR는 GRPO 및 DAPO 기준선 대비 우수한 성능을 보여주며, 평균 LLM-판별자 정확도를 최대 4.2포인트 향상시키고 결과가 동일한 그룹의 42%를 유익한 그룹으로 변환하는 데 성공했습니다. 이는 ARBOR가 모든 다중 단계 Q&A 벤치마크에서 일관되게 성능을 발휘함을 나타냅니다. ARBOR는 결과 뿐만 아니라 검색 과정에서도 의미 있는 피드백을 제공하여,제한 없이 다양한 쿼리 지원을 가능하게 합니다.



### WebRISE: Requirement-Induced State Evaluation for MLLM-Generated Web Artifacts (https://arxiv.org/abs/2606.03220)
- **What's New**: 이번 연구는 기존의 MLLM(다중 모달 대형 언어 모델)에서 생성된 웹 아티팩트를 평가하기 위한 새로운 벤치마크인 WebRISE를 소개합니다. WebRISE는 상호작용 계약 그래프(Interactive Contract Graphs)라는 구조를 통해 사용자 요구에 따른 상태 및 전이를 정의하며, 웹 페이지가 실제로 작동하는지 여부를 결정하는 데 필요한 요구 사항 기반의 평가를 가능하게 합니다. 이 연구는 442개의 작업을 통해 MLLM의 상호작용 생성이 여전히 해결되지 않았음을 보여줍니다.

- **Technical Details**: WebRISE는 웹 아티팩트를 평가하기 위해 명시적 및 암시적 요구 사항을 상호작용 계약으로 변환하여 처리합니다. 각 작업은 상호작용 계약 그래프(ICG)라는 모델을 통해 표현되며, ICG는 동적 UI 상태, 사용자 의도 전이 및 DOM/비주얼 주장을 포함합니다. 이 과정에서 웹 브라우저를 통해 평가되며, 생성된 페이지가 계약에 따라 작동하는지를 테스트합니다.

- **Performance Highlights**: 1414 개의 모델을 평가한 결과, 최고의 모델조차 65.6%의 전이 유효성과 66.3%의 요구 사항 커버리지를 달성했으며, 이는 약 1/3의 전이 및 요구 사항 체크가 충족되지 않았음을 의미합니다. 비디오 입력은 텍스트에 비해 상호작용 품질을 크게 향상시키며, 현재 MLLM 기반의 웹 생성 시스템은 여전히 많은 과제를 안고 있음을 보여줍니다. 결함 주입 실험을 통해 WebRISE의 평가가 기존의 검사 기준보다 2배에서 16배 더 효과적으로 상태 오류를 감지한다고 명시되었습니다.



### Sample-Size Scaling of the African Languages NLI Evaluation (https://arxiv.org/abs/2606.03219)
Comments:
          Accepted at the AfricaNLP Workshop, EACL 2026

- **What's New**: 이 연구는 아프리카 언어의 자연어 추론(NLI)에서 레이블된 데이터 양의 영향을 분석한 체계적인 데이터 샘플 크기 확대 연구입니다. 기존의 믿음과 달리, 데이터 크기가 증가하는 것과 성능의 증가가 항상 일치하지 않음을 발견했습니다. 몇몇 언어는 샘플 크기가 증가함에 따라 성능의 포화 또는 감소를 보였으며, 이는 아프리카 NLI의 안정적인 발전을 보장할 수 없음을 시사합니다.

- **Technical Details**: 연구는 AfriXNLI 벤치마크를 사용하여 16개 아프리카 언어에서 두 가지 다국어 트랜스포머 모델(XLM-R Large, AfroXLM-R Large)을 테스트했습니다. 샘플 크기는 50에서 500 사이이며, 종합적인 성능 결과를 위해 여러 랜덤 서브샘플링을 수행하였습니다. 평가 지표로는 주로 정확도(accuracy)를 사용하였으며, 정밀도(precision)와 F1-score 역시 보고했습니다.

- **Performance Highlights**: 결과적으로, 샘플 크기가 증가할수록 성능의 변동성은 감소했지만, 정확도가 반드시 증가하지는 않았습니다. 낮은 리소스 샘플(50-100 예제)은 높은 변동성을 보였고, 샘플 크기가 커질수록 안정된 성능 추정이 가능해졌습니다. 연구 결과는 아프리카 언어의 NLI 훈련에서 데이터 양 뿐만 아니라 언어 감수성이 필요함을 강조하며, 언어별로 성능 변화가 다르게 나타났습니다.



### AI Rater Discrimination Depends on Scoring Protocol in Complex Clinical Decision-Making (https://arxiv.org/abs/2606.03198)
Comments:
          11 pages, 4 main figures, 8 supplementary figures, 9 supplementary tables

- **What's New**: 본 논문은 대규모 언어 모델(LLM)을 AI 평가자로 활용한 임상 AI 평가의 새로운 접근 방식을 제시하고 있습니다. 특히, 성인 제2형 당뇨병(T2D) 약물 치료에서의 AI 래이터 행동을 7가지 평가 질문에 대해 정량적으로 분석합니다. 연구는 Gold Rubric(그림자 척도)과 Non Gold Rubric(비첨척도)이라는 두 가지 스코어링 프로토콜을 사용하여 AI 평가자의 점수 차이를 파악하고 있습니다.

- **Technical Details**: 본 연구는 인공지능 행동 평가 규정의 차이를 분석하기 위해 인과 실험 설계를 사용하고 있습니다. 네 개의 오픈 소스 LLM이 임상 결정 지원 시스템(CDSS) 모델과 AI 평가자로 동시에 작동하며, 다양한 평가 조건에 따라 AI 평가자의 점수를 평가합니다. 연구에서는 스코어링 프로토콜과 CDSS 모델, 프롬프트 설정, 평가자의 모델, 프롬프트 문자 및 프롬프트 유형의 상호작용을 분석하였습니다.

- **Performance Highlights**: 결과에 따르면, Non-GR 프로토콜 하에서 AI 평가자의 점수가 평균 74-78점으로 결론지어졌고, GR보다 평균적으로 7.69에서 49.64점 낮은 점수를 보였습니다. GR 프로토콜은 DRG와 Baseline CDSS 출력 간의 AI 평가자의 구별력을 1.76배에서 5.10배까지 증가시키며, Non-GR에서는 그런 구별력이 억제된 것으로 나타났습니다. 이러한 결과는 임상 AI 평가에서 점수 프로토콜이 중요한 역할을 한다는 것을 지지합니다.



### MemTrain: Self-Supervised Context Memory Training (https://arxiv.org/abs/2606.03197)
- **What's New**: 이 논문에서는 MemTrain을 제안하여 LLM(대형 언어 모델) 에이전트의 컨텍스트-메모리 기능을 자가 지도 학습(self-supervised learning) 방식으로 향상시키는 방법을 제시합니다. MemTrain은 라벨이 없는 Wikipedia 데이터를 활용하여 두 가지 프로시(proxies) 작업을 통해 메모리 생성 및 활용 향상을 목적으로 합니다. 특히, 이 방법은 메모리 업데이트 이후 마스킹된 엔티티 복구와 중간 메모리 상태를 통한 과거 정보 복원 과제를 포함합니다.

- **Technical Details**: MemTrain의 핵심은 두 개의 결합된 프로시 작업인 (1) end-to-end masked reconstruction과 (2) intermediate memory recall입니다. 이를 통해 모델은 여러 번의 메모리 업데이트 후 마스킹된 정보를 복구하는 동시에, 과거의 중요 정보를 충실하게 재구성할 수 있는 능력을 배양합니다. 이러한 과정을 통해 메모리의 유지 및 활용에 대해 효과적인 훈련이 이루어집니다.

- **Performance Highlights**: MemTrain을 통해 다양한 모델에서 하위 훈련 후 메모리 집약적 추론 성능을 지속적으로 향상시킬 수 있음을 보여주었습니다. Qwen3-4B-Instruct-2507 및 Qwen2.5-7B-Instruct에서 각각 5.17점, 10.58점, 그리고 17.67점, 8.50점의 성과 향상을 달성하며, MemTrain의 효과를 입증했습니다.



### SenseJudge: Human-Centric Preference-Driven Judgment Framework (https://arxiv.org/abs/2606.03189)
Comments:
          ACL 2026 Findings

- **What's New**: 대형 언어 모델(LLMs)을 평가하는 새로운 프레임워크인 SenseJudge가 제안되었습니다. 이는 사용자의 다양성을 반영하는 데 최적화된 커스터마이즈가 가능한 평가 모델로, SenseBench라는 고품질 벤치마크와 결합되어 있습니다. 기존 방식들이 놓쳤던 사용자 선호도를 반영하여 LLM을 개인화된 평가자로 활용하는 방법을 다룹니다.

- **Technical Details**: SenseBench는 실제 인간-인공지능 대화에서 파생된 고품질의 교육 데이터로 구성되어 있으며, 수학, 논리, 창의적 글쓰기 등 다양한 주제를 포함합니다. SenseJudge 프레임워크는 사용자 선호에 기반하여 모델 응답을 평가하며, 두 가지 주요 작업에 적용됩니다: 개인화된 판별자 및 모델 랭킹. 이 접근 방식은 다양한 사용자 요구를 반영할 수 있도록 디자인되었습니다.

- **Performance Highlights**: Experimental 결과에 따르면, SenseJudge는 기존의 최신 API 및 다양한 점수 모델보다 우수한 성능을 보이며, 인간의 판단 기준에 맞춰 모델 간의 순위 매기기에서도 높은 신뢰성을 보여줍니다. 위치 편향 및 일관성에 대한 추가 분석과 아블레이션 연구를 통해 이 프레임워크의 견고함이 확인되었습니다.



### HyperPatch: Sequential Knowledge Editing Under n-ary Structural Drif (https://arxiv.org/abs/2606.03179)
Comments:
          Accepted to Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2026)

- **What's New**: 본 논문은 Knowledge Editing (KE)의 새로운 접근법인 HyperPatch를 제안하고 있습니다. 이 프레임워크는 nn-ary 관계에서 발생하는 구조적 드리프트를 해결하기 위해 하이퍼그래프 개념을 활용하여 안정성을 유지합니다. 기존의 Knowledge Graph (KG) 기반 방법들이 겪는 문제점을 보완하며, 전반적으로 이벤트의 무결성을 보존하는 데 중점을 두고 있습니다.

- **Technical Details**: HyperPatch는 세 가지 주요 혁신으로 구성됩니다: (i) Structural Prior Initialization은 Hypergraph Neural Network (HGNN)를 통해 retriever의 임베딩 매니폴드와 하이퍼그래프의 진화하는 토폴로지를 동기화합니다. (ii) Sequential Topology Editing에서는 하이퍼엣지 중심의 해싱을 기반으로 하는 구조 매핑을 구현해 업데이트 지연을 최소화합니다. (iii) Structure-Conditioned Reasoning은 업데이트된 nn-ary 토폴로지와 검색 매니폴드를 통합하여 멀티 홉 질문-응답 성능을 유지합니다.

- **Performance Highlights**: HyperPatch는 MQuAKE-CF 및 MQuAKE-T 벤치마크에서 각각 96.24%와 21.06%의 Hop-wise Accuracy (H-Acc) 향상을 달성하였습니다. 기존 KG 기반 변형이 88.3% H-Acc 붕괴를 경험할 때, HyperPatch는 구조적 불일치에도 강한 내성을 보여주며, 25.9배 빠른 검색 속도를 달성했습니다.



### Fully Automated Identification of Lexical Alignment and Preference-Stage Shifts in Large Language Models (https://arxiv.org/abs/2606.03165)
Comments:
          16 pages, 2 figures, 10 tables

- **What's New**: 디지털 대화 비서인 ChatGPT와 같은 Large Language Models(LLMs)에 대한 사용이 증가하고 있으며, 이러한 AI 툴은 프로그래밍, 언어 편집 및 정보 검색에 널리 이용되고 있다. 그러나 이들은 인간의 언어 사용과는 체계적으로 어긋나는 경향이 있다. 본 논문은 인간의 선호 학습 훈련에서 발생하는 이러한 불일치를 분석하고, 두 가지 새로운 평가 지표인 Lexical Alignment Score(LAS)와 Triangulated Preference Shift(TPS)를 제안한다.

- **Technical Details**: 이 연구에서는 42,000개의 PubMed 초록을 사용하여 6개의 모델 패밀리(Falcon, Gemma, Llama 등)의 생성 모델을 평가하였다. 새로운 지표인 LAS는 인간의 계속된 응답에 비해 과도하게 사용되는 용어를 정량적으로 평가하며, TPS는 선호 학습 단계에 기인한 변화를 분리하여 분석한다. 이러한 평가 방법은 수동 개입 없이도 과사용되는 단어를 식별하고, 인간의 선호와의 연결성을 추정한다.

- **Performance Highlights**: 결과적으로, LAS와 TPS는 개별 용어 및 선호 학습으로 인한 변화의 정도를 추정하는 데 유망한 결과를 보여주었다. 모든 변형에서 결과는 안정적이었으며, 파라미터 설정이나 무작위 시드의 변경에도 견고함을 유지하였다. 이 방식은 다른 언어로의 확장이 용이하며, 향후 모델의 개선 방향과 그 기원에 대한 이해에 기여할 수 있는 가능성을 지니고 있다.



### A cross-domain tropical species dataset with Chinese vernacular names and CITES source links (https://arxiv.org/abs/2606.03156)
Comments:
          25 pages, 4 figures, 4 tables. Dataset descriptor for the Tropical Species Encyclopedia. Companion to the methodology paper arXiv:2606.00994. Dataset deposited at Zenodo (doi:https://doi.org/10.5281/zenodo.20377811%29%3B canonical preprint-of-record at Zenodo (doi:https://doi.org/10.5281/zenodo.20424981)

- **What's New**: 본 연구는 410,499개의 활동 중인 열대 종(tropical species)으로 구성된 버전 관리된 크로스 도메인 데이터셋을 소개합니다. 이 데이터셋은 열대 식물(tropical_plants), 열대 수생 생물(tropical_aquatic), 열대 애완동물(tropical_pets)이라는 세 가지 응용 하위 도메인을 포함하고 있으며, 이들은 상업적 및 규제 생애 주기를 공유합니다. 데이터셋은 GBIF, Plants of the World Online 등에서 수집한 정보와 함께 상업적 거래 및 사육 환경에 따른 다층 구조를 추가하여 제공합니다.

- **Technical Details**: 이 데이터셋은 GBIF, NCBI Taxonomy, CITES 등 다양한 출처의 분류학적 식별자를 결합하며, 중국어 고유 명칭(vernal names)과 CITES 소스 연결 층을 포함하고 있습니다. 특히 99.50%의 중국어 명칭이 각 종에 대해 존재하여 데이터셋의 완전성을 보여줍니다. 데이터셋은 Zenodo에 등록되어 있으며, 향후 데이터 디스크립터 제출이 기대되지만, 검증 및 릴리스 엔지니어링 항목이 완료되어야 가능합니다.

- **Performance Highlights**: 본 데이터셋은 열대 종 거래 및 사육에 특화된 구조로, 기존의 생물 다양성 인프라를 보완하며, 상업적 흐름을 통한 분류 및 관리를 지원합니다. 다중 출처 데이터를 포함하여 저작권 문제를 해결하는 접근 방식을 취하고 있으며, 열대 종 상업 및 사육 관련 연구에 유용한 자원이 될 것입니다. 이러한 방식으로, 이 데이터셋은 열대 생물 다양성에 관한 연구와 교육을 위한 기초 자료로서 중요한 역할을 할 것으로 기대됩니다.



### DMT-CBT: Longitudinal Therapeutic State Modeling for CBT Counseling (https://arxiv.org/abs/2606.03132)
- **What's New**: 최근 대형 언어 모델(LLMs)이 인지 행동 치료(CBT) 상담 분야에서 점점 더 많은 가능성을 보여주고 있습니다. 그러나 기존의 접근 방식들은 대부분 상담을 단기적인 응답 생성 문제로 국한하고 있으며, 이는 실제 심리 치료의 본질과 근본적으로 맞지 않다고 주장합니다. 본 연구에서는 상담을 다각적 상태 변화와 세션 간 효과를 반영하는 장기적 프로세스로 재정의하는 DMT-CBT라는 새로운 프레임워크를 제안합니다.

- **Technical Details**: DMT-CBT는 상담 세션之间의 구조화된 치료 상태(linear therapeutic states)를 유지하며, 다중 모드 기반 행동 추적과 도구 보강 개입(tool-augmented intervention)을 통합하여 적응형 치료 추론을 지원합니다. 이 프레임워크를 기반으로 DMTCorpus라는 합성 멀티 세션 멀티모달 CBT 데이터셋을 구축하여 진화하는 치료 상태, 이미지 기반의 클라이언트 행동 및 세션 간 개입 연속성을 포함합니다. 모델은 클라이언트의 상태를 부분적으로 관찰할 필요가 있으며, 치료의 연속적인 진행을 추적 할 수 있어야 합니다.

- **Performance Highlights**: 실험 결과 DMT-CBT는 상담 충실도(counseling fidelity)와 치료적 동맹(therapeutic alliance)을 개선하며, 세션 간 정서적 경로(affective trajectories)를 더 긍정적으로 이끌고, 기존의 사후적 추출(post-hoc extraction) 접근 방식보다 치료 상태를 보다 신뢰성 있게 보존함을 보여주었습니다. 본 연구는 LLM 기반 CBT 상담을 고립된 공감 응답 생성 문제 대신 장기적 치료 상태 모델링 문제로 재구성하고, 치료 상태 진화 및 적응적 개입을 효과적으로 모델링하는 DMT-CBT 프레임워크를 제안합니다.



### Experience-Driven Dynamic Exits for LLMs with Reinforcement Learning (https://arxiv.org/abs/2606.03113)
- **What's New**: 이번 논문에서는 대형 언어 모델(DLLM)의 느린 autoregressive inference 문제를 해결하기 위한 새로운 접근법인 LEDE(Learning-based Dynamic Exit)를 제안합니다. LEDE는 Markov Decision Process를 활용하여 동적으로 최적의 exit layer와 speculation length를 선택함으로써 속도를 향상시킵니다. 이를 통해 기존의 고정된 설정에 대한 한계를 극복하고, 각 단계의 로컬 컨텍스트에 기반한 의사 결정을 가능하게 합니다.

- **Technical Details**: LEDE는 오프라인 강화 학습(Offline Reinforcement Learning)을 통해 학습된 정책을 사용하여 모델의 내부 상태를 관찰하고, 각 단계에서 적절한 exit layer를 결정하는 방식을 채택합니다. 입력 데이터의 다양성을 반영하여 정책 업데이트를 수행하는 과정은 경험 리플레이를 통해 이루어지며, 감정적 신뢰도 및 예측 불확실성와 같은 특징을 기반으로 합니다. 이 과정에서 DQN(Deep Q Network)을 활용하여 exit 결정을 최적화하는 목표를 가지고 있습니다.

- **Performance Highlights**: LEDE는 Llama-2 및 Llama-3 모델에서 평가를 통해 autoregressive decoding에 비해 최대 2.7배의 속도 향상을 달성하였습니다. 또한, 고정된 스펙ulative baseline에 비해 추가적인 17% 속도 향상을 제공하여, 전체 추론 과정의 효율성을 크게 개선하였습니다. 이러한 결과는 LEDE가 자원의 소비를 절감하면서도 높은 품질의 출력을 생성할 수 있음을 보여줍니다.



### Coherence Maximization Improves Pluralistic Alignmen (https://arxiv.org/abs/2606.03110)
- **What's New**: 이 연구는 AI 시스템을 다양한 인간 가치에 맞추기 위한 구체적인 예시 생성의 어려움을 다룹니다. Internal Coherence Maximization (ICM) 방법을 통해 인간 감독 없이도 특정 페르소나에 맞춘 효과적인 예시를 생성할 수 있음을 보여줍니다. 실험 결과, ICM으로 생성된 예시는 기존의 고품질 라벨(Gold labels)과 유사한 성능을 나타내며, coherence(일관성)가 label의 개별 정확성보다 더 중요하다는 점을 강조합니다.

- **Technical Details**: 본 연구에서는 세 가지 단계를 통해 페르소나에 특화된 가치 예시를 생성합니다: (1) 페르소나 특징 추출 및 항목 선택, (2) ICM 기반 라벨 추정, (3) 맥락 내(In-context) 조건 등을 사용하여 추론을 수행합니다. ICM은 통계적으로 일관된 가치 시스템을 탐색함으로써 서로 예측 가능한 라벨을 최대화하여, 인간 감독 없이 라벨을 추론합니다. 그 결과, 페르소나의 가치 프로파일에 가장 일관성이 높은 라벨을 회복하게 됩니다.

- **Performance Highlights**: ICM을 통해 생성된 예시는 분류 기준 및 선호 정확도에서 고품질 라벨과 비슷한 성능을 보이며, 검증된 4개의 데이터셋과 3개의 작업 형식에서 일관성을 유지합니다. 특히, pretrained 데이터에서 덜 대표되는 집단의 경우, 가장 불확실한 질문에 인간 피드백을 제공하면 일반화에 더 효과적임을 입증했습니다. 이러한 결과는 coherence가 확장 가능한 가치 명세를 위한 핵심 설계 원칙임을 시사합니다.



### Small RL Controller, Large Language Model: RL-Guided Adaptive Sampling for Test-Time Scaling (https://arxiv.org/abs/2606.03102)
- **What's New**: 이 논문에서는 adaptive sampling을 Markov decision process (MDP)로 모델링하여 기존의 한계를 극복하는 경량화된 방법을 제안합니다. 강화 학습 (Reinforcement Learning)을 통한 샘플링 컨트롤러를 훈련시켜 답변의 정확성(correctness), 지연(latency), 및 계산 비용(computation cost)을 균형 있게 조절할 수 있게 합니다. 이는 주로 생성된 최종 답변의 통계에만 의존하므로 추가적인 신호나 외부 개입 없이도 구현할 수 있습니다.

- **Technical Details**: 제안된 방법은 최종 답변의 정확도를 긍정적인 보상으로, 샘플링 라운드와 추가 샘플링으로 인해 발생하는 비용을 페널티로 설정하여 다중 목표를 고려합니다. 각 라운드에서 컨트롤러는 현재 샘플 집합의 통계적 특성을 관찰하고, 추가 샘플을 생성할지 샘플링을 중단할지를 결정합니다. 최적화는 샘플의 집합을 통해 이루어지며, 이는 모델의 자연스러운 추론 과정에 개입하지 않고 수행됩니다.

- **Performance Highlights**: 실험 결과, RL-Guided Sampling 방법은 기존의 ASC 및 ESC와 비교하여 3배의 샘플링 라운드 감소, 및 총 샘플 수 30% 감소를 보여줍니다. 이 개선은 다양한 트레이드오프 수치에서도 일관되게 나타났으며, 훈련된 정책은 다른 데이터 세트나 모델에서도 성능 저하 없이 잘 일반화됩니다. 따라서 이 방법은 환경제어 및 성능 최적화의 효율성을 높일 수 있는 가능성을 보여줍니다.



### PhotoCraft: Agentic Reasoning with Hierarchical Self-Evolving Memory for Deep Image Search (https://arxiv.org/abs/2606.03099)
- **What's New**: 이 논문에서는 Deep Image Search의 한계점을 극복하기 위한 방법으로 PhotoCraft를 제안합니다. PhotoCraft는 계층적 메모리 시스템을 제공하며, 이는 작업 간의 경험 전이를 용이하게 하는 동시에 논리적 일관성과 지식 전이 가능하게 합니다. 이 시스템은 훈련이 필요 없는 구조로, MLLM 기반 에이전트의 메모리 병목 문제를 해결하는 데 도움을 줍니다.

- **Technical Details**: PhotoCraft는 작업 기억(working memory), 에피소드 기억(episodic memory), 의미 기억(semantic memory)의 세 가지 상호 보완적인 구성 요소로 이루어져 있습니다. 작업 기억은 단기적인 상황 인식을 지원하고, 에피소드 기억은 현재 프로세스를 역사적인 목표와 연결하며, 의미 기억은 추상적인 지식 표현과 일반화를 가능하게 합니다. 이러한 메모리 시스템은 다단계 추론 과정 동안 동적으로 호출되어 논리적 일관성을 유지합니다.

- **Performance Highlights**: DISBench에서의 광범위한 실험 결과, PhotoCraft는 다양한 MLLM 백본에서 맥락을 인식하는 검색 성능을 지속적으로 향상시키며, 최대 18.5%의 개선 효과를 보여주었습니다. 또한, 이 시스템은 메모리 없는 깊은 이미지 검색의 주요 병목을 효과적으로 완화하여 신뢰할 수 있고 일반화 가능한 다중 모드 검색 에이전트를 구축하는 실질적인 경로를 제시합니다.



### Can Factual Opinions Be Edited (Manipulated) in Large Language Models? (https://arxiv.org/abs/2606.03096)
- **What's New**: 이번 연구는 LLM(대형 언어 모델)의 지식 편집 기술의 새로운 위험을 다루고 있습니다. 연구진은 Factual Opinion Editing with Evidence (FOE) 벤치마크를 제안하여 공공 인물의 사실적 의견 조작의 위험성을 체계적으로 평가합니다. 이 벤치마크는 261명의 공적 인물, 19개의 이슈 카테고리, 2,178개의 완전한 의견 기록을 포함하고 있습니다. 이러한 편집 방법들이 фактические мнения(사실적 의견)를 조작하는 데 어려움을 겪고 있음을 밝혀냈습니다.

- **Technical Details**: 이 연구에서 제안하는 FOE 벤치마크는 사실적인 의견 편집의 효과를 평가하기 위한 설계 및 데이터 수집 과정을 포함합니다. 기존의 편집 방법들은 일반적으로 원자적 사실(atomic facts)에 초점을 맞췄으나, 이 연구는 사실적 의견에 대한 보다 심도 있는 분석을 제공합니다. 이를 통해, LLM이 지원하는 의견과 그에 따른 증거 사이의 일관성을 유지하는 데 어려움이 있음을 입증하였습니다. 연구진은 Self-Generated Evidence-Aligned 방식을 제안하여 명시적인 지침 없이도 의견-증거 정렬을 성공적으로 달성할 수 있음을 보여주었습니다.

- **Performance Highlights**: 실험 결과, 기존 편집 방법들은 사실적 의견을 조작하는 데 효과적이지 않으며, 종종 피상적인 수정에 그쳤음을 확인했습니다. 이는 최종 모델의 일관성과 설득력을 저하시킬 수 있습니다. 또한, 증거 요구 지침을 도입함으로써 편집된 모델이 목표 의견에 일치하는 증거를 제공하도록 유도할 수 있음을 보여주었습니다. 이를 통해 LLM의 사실적 의견 편집에 대한 보안 우려를 제기하며, 향후 방어 메커니즘 개발을 위한 기초를 마련했습니다.



### Regret Pre-training: Bridging Prior and Posterior Views for Enhanced Knowledge Grounding (https://arxiv.org/abs/2606.03080)
- **What's New**: 이 논문은 Regret Pre-training이라는 새로운 자기 지도 학습 프레임워크를 도입합니다. 이 프레임워크는 학습 중에 미래 정보를 활용하여, causal language model의 비대칭성을 해결하는 데 목표를 둡니다. 이를 위해 dual-view architecture를 사용하여 causal Student distribution과 future-conditioned Teacher distribution을 생성합니다.

- **Technical Details**: Regret Pre-training은 특정 손실 함수인 regret loss를 사용하여 KL divergence를 최소화하며, 잠재적 정보를 바탕으로 causal representation에 미래 예측을 전달합니다. 두 가지 teacher 설정(LocalRegret 및 GlobalRegret)을 사용하여 OLMoE-1B-7B 모델에서 실험을 진행하며, 각 설정은 미래 컨텍스트의 범위에 따라 달라집니다. 이 프레임워크는 전체적인 아키텍처 수정 없이 attention mask 생성만으로 구현될 수 있습니다.

- **Performance Highlights**: 실험 결과는 두 가지 설정 모두 기존 baseline보다 우수한 성능을 보인다는 것을 보여주었습니다. 특히 GlobalRegret은 BoolQ에서 61.0%의 정확도로 baseline의 42.9%보다 18.1%p 높은 성과를 기록했습니다. 평균적으로 GlobalRegret과 LocalRegret은 각각 33.9%와 32.2%의 정확도를 달성하여 baseline의 30.2%를 초과했습니다.



### G^2C-MT: Graph-Guided Context Selection for Document-Level Machine Translation (https://arxiv.org/abs/2606.03078)
Comments:
          9 pages, 2 figures; IJCAI2026

- **What's New**: 이 논문에서는 문서 수준의 기계 번역(Document-level Machine Translation, DocMT)을 위한 새로운 프레임워크인 G^2C-MT(그래프 기반 문맥 선택)를 제안합니다. 기존의 방법들은 특정 문맥을 선택하는 방법이 부족했으나, G^2C-MT는 경량의 담화 그래프를 활용해 구조화된 경로 발견 문제로 접근합니다. 각 단락을 노드로 보고, 의미적 유사성, 인접성 및 키워드 중복성을 고려하여 노드 간 관계를 모델링합니다.

- **Technical Details**: G^2C-MT는 방향성 비순환 그래프(Directed Acyclic Graph, DAG)로 문서의 담화 구조를 모델링합니다. 각 단락은 노드로 표현되고, 엣지는 단락 간의 관계를 나타냅니다. 모델은 깊이 편향 랜덤 워크(Depth-Biased Random Walk)를 적용하여 각 타겟 단락에 대한 문맥 경로를 샘플링하며, 이러한 경로는 대규모 언어 모델(Large Language Model, LLM)의 번역을 유도합니다.

- **Performance Highlights**: 다양한 도메인에서 실시된 실험 결과, G^2C-MT는 DeepSeek-V3, Gemini-2.5-Flash-lite 및 Qwen-2.5/3 시리즈를 포함한 여러 LLM에서 강력한 기준선을 초과하는 성과를 보여주었습니다. 실험을 통해 G^2C-MT가 긴 거리 의존성을 효과적으로 캡처하며, 번역 품질과 일관성 모두에서 개선되는 것을 확인했습니다.



### The Geometry of LLM-as-Judge: Why Inter-LLM Consensus Is Not Human Alignmen (https://arxiv.org/abs/2606.03043)
- **What's New**: 이 논문에서는 LLM(대형 언어 모델) 평가자의 일관성에 대한 새로운 통찰을 제공합니다. 저자들은 여러 언어와 커뮤니티에서 구축된 데이터셋을 통해 LLM들이 인간 평가자와 얼마나 일치하는지를 분석했습니다. LLM 간의 높은 합의가 인간 평가자와의 낮은 합의와 어떻게 관련되는지를 조사하며, 각기 다른 기하학적 양상을 측정했습니다.

- **Technical Details**: 연구는 41개의 LLM 평가자를 대상으로 하여 총 4개의 인디언 데이터셋과 8개의 인디언 언어를 활용하였습니다. 저자들은 score spread, effective rank, principal angle, 그리고 stacked correlations와 같은 네 가지 기하학적 양상을 통해 LLM과 인간의 평가를 비교했습니다. 특히, LLM의 평가는 인간의 평가는 얼마나 닮지 않았는지를 수치적으로 나타내었으며, 평가 모델의 튜닝과 최적화를 통해 그 차이를 줄이려는 노력을 보였습니다.

- **Performance Highlights**: 논문 결과에 따르면, LLM 간의 합의는 LLM-인간 간의 합의보다 높고, 인간 간의 신뢰성에 비해 여전히 낮은 수치로 나타났습니다. 추가적인 후속 조정이 이루어졌을 때 LLM의 평가 정확도가 증가하였지만, 여전히 인간 평가자와의 차이를 완전히 해소하지는 못했습니다. 연구자들은 LLM 간의 합의가 인간 평가와의 일치를 증명하는 것은 아니며, 이는 축소된 서브스페이스 내에서의 합의일 뿐이라고 강조하고 있습니다.



### The Deliberative Illusion: Diagnosing Factual Attrition and Stance Homogenization in Multi-Agent LLM Deliberation (https://arxiv.org/abs/2606.03032)
- **What's New**: 이번 연구에서는 다중 에이전트 LLM 시스템에서의 합의(consensus) 현상이 정보의 보존 및 다양한 관점의 유지에 회의적일 수 있다는 점을 제시합니다. 이러한 시스템에서 논의의 결과는 종종 사실의 소실(factual attrition)과 입장의 동질화(stance homogenization)를 낳습니다. 저자들은 이러한 현상을 측정하기 위해 DelibTrace라는 새로운 평가 프레임워크를 도입하였습니다.

- **Technical Details**: DelibTrace 프레임워크는 문제를 원자적 사실(atomic facts)로 분해하고, 이에 대한 레이블을 붙인 후 정보를 에이전트에 분배하여 각 논의 라운드를 통해 생존 여부를 추적하는 기법입니다. 이를 통해 정보 흐름(information flow)을 가시화하고, 에이전트 간의 의사소통 구조가 정보의 흐름에 미치는 영향을 분석합니다. 연구 결과는 다중 에이전트 논의가 중요 사실의 최대 72%를 소실시킨다는 점을 강조합니다.

- **Performance Highlights**: 결과적으로, 다중 에이전트 논의에서 가져온 자료가 해석이나 판단을 왜곡할 수 있다는 점이 드러났습니다. 덧붙여, 단일 악의적인 에이전트가 정보의 감소된 공유 컨텍스트에 잘못된 정보를 주입할 가능성 역시 지적하고 있습니다. 이러한 결과는 합의가 정보를 잃은 상태에서도 나타날 수 있음을 보여줍니다.



### Conditional Hypothesis Generation for LLM-Based Text Analysis with Researcher-Specified Covariates (https://arxiv.org/abs/2606.03029)
- **What's New**: 본 논문은 컴퓨터 사회과학의 핵심 목표인 언어의 변화를 이해하는 데 있어, 연구자가 지정한 공변량(covariates)을 포함하여 가설 생성을 조정하는 "조건부 가설 생성(conditional hypothesis generation)" 프레임워크를 제안합니다. 이는 표본 간의 차이를 위주로 한 기존의 방법과는 달리, 연구자가 관심 있는 하위 집단(subgroups) 내에서의 차이를 강조합니다. 따라서, 공변량을 무시했을 때 발생할 수 있는 혼란(confounds)에 따른 결과 왜곡을 방지할 수 있습니다.

- **Technical Details**: 논문에서 제안하는 두 가지 방법은 경제학에서 영감을 받아 구체적으로 설계되었습니다. 첫 번째는 상호작용-라쏘(interaction-lasso)로, 이는 공변량과의 상호작용(feature-covariate interactions)을 통해 특성이 특정 집단 내에서만 차별화되는 경우를 탐지할 수 있도록 합니다. 두 번째 방법인 평균화-가중치-라쏘(demeaned-reweighted-lasso)는 공변량 구간 내에서 피처와 결과를 평균화하여 하위 집단 내 변화를 분리하고, 저대표군의 기여도를 동등하게 하여 선택 과정에 공정성을 제공합니다.

- **Performance Highlights**: 합성 실험 결과, 두 방법 모두 글로벌 기준선(global baselines)보다 우수한 성과를 보였으며, 실제 데이터셋에서도 공변량을 고려한 가설 생성이 전문가에 의해 더 유용하다고 평가되었습니다. 특히, 평균화-가중치-라쏘는 불균형 수준에 걸쳐 약간의 차이를 보이는 가설을 복원하는 데 유일한 방법으로 자리잡았습니다. 이러한 결과는 연구자가 지정한 특정 조건에서 생성된 가설이 더 명확하고 유용하다는 것을 입증합니다.



### SEA-Embedding: Open and Reproducible Text Embeddings for Southeast Asia (https://arxiv.org/abs/2606.03027)
- **What's New**: SEA-Embedding은 동남아시아 언어를 위한 완전 개방형 텍스트 임베딩 파이프라인으로, 공개 데이터만을 활용하여 훈련되었습니다. 이 연구는 Robust(강력한) 텍스트 임베딩 설계의 세 가지 핵심 요소인 데이터 구성(data composition), 훈련 목표(training objective), 그리고 기본 인코더 초기화(base encoder initialization)를 분석합니다. SEA-Embedding은 SEA-BED 벤치마크에서 최신 성과를 달성하며, 체계적이고 재현 가능한 분석을 가능하게 합니다.

- **Technical Details**: SEA-Embedding 파이프라인은 세 가지 주요 구성 요소인 RQ1: 데이터 구성, RQ2: 목표 설계, 그리고 RQ3: 기본 모델로 체계적인 검토를 위한 개념적 프레임워크를 제공합니다. 이 모델은 지역적 차별성과 전 세계적 일관성을 모두 요구하며, 두 가지 데이터 카테고리를 사용하여 훈련합니다. 첫 번째는 일반 데이터셋(245M 샘플)으로 지역적 커버리지를 확보하고, 두 번째는 지시 데이터셋(14M 샘플)으로 목적 인식을 개선합니다.

- **Performance Highlights**: SEA-Embedding은 SEA-BED 벤치마크에서 최신 성과를 기록하며, 이를 통해 Robust한 SEA 텍스트 임베딩을 위한 재사용 가능한 설계 레시피를 제공합니다. 실험은 완전히 재현 가능하며, 데이터 구성, 목표 설계 및 기본 인코더 초기화의 영향을 분리하여 분석하였습니다. 이러한 기여는 향후 동남아시아 언어의 자연어 처리(NLP) 성능 향상에 기여할 것으로 기대됩니다.



### Hallucinations as Orthogonal Noise: Inference-Time Manifold Alignment via Dynamic Contextual Orthogonalization (https://arxiv.org/abs/2606.03022)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 환각 현상(hallucination)을 기하학적 틀에서 해결하는 방법을 제안합니다. 환각은 맥락적 사실이나 논리적 제약과 일치하지 않는 콘텐츠의 생성을 의미하며, 특히 세맨틱 매니폴드(semantic manifold)와의 정사각형(noise) 반대 방향으로 정보가 전파되는 경우에 발생한다고 주장합니다.

- **Technical Details**: 이 연구에서는 Dynamic Contextual Orthogonalization(DCO)라는 추론 시간 개입(intervention) 방법을 도입합니다. DCO는 레이어에서의 입력 잔여 흐름(input residual stream)을 동적인 맥락 앵커(context anchor)로 활용하여 주의(attention) 머리 출력(output)에서 패턴을 잡아내고, Z-Score 억제(z-score suppression) 메커니즘을 통해 맥락에 정렬된 업데이트와 비대칭 노이즈를 구별합니다. DCO는 모형의 파라메트릭 지식 조회 능력을 유지하면서도 환각을 효과적으로 감소시키도록 설계되었습니다.

- **Performance Highlights**: DCO는 Llama-3-8B 및 70B 모델에서 다양한 벤치마크(XSum, NQ-Swap 및 IFEval)에서 최첨단 개입 방법들에 비해 우수한 성능을 보였습니다. 또한, 트리비아 QA(TriviaQA)와 진실한 QA(TruthfulQA)와 같은 지식 집약적인 작업에서도 높은 성능을 유지하여 환각 억제와 파라메트릭 지식 보존 간의 균형을 잘 맞추었습니다. 이러한 결과는 DCO가 기하학적 제약을 시행하는 데 있어 효과적이며 계산적으로 효율적임을 입증합니다.



### Hint-Guided Diversified Policy Optimization for LLM Reasoning (https://arxiv.org/abs/2606.03021)
- **What's New**: 최근 대형 언어 모델(Large Language Models, LLMs)의 발전은 뛰어난 추론 능력을 보여주고 있으며, RLVR(Reinforcement Learning with Verifiable Rewards)를 통한 향상 전략이 주목받고 있습니다. 하지만 기존 보상 메커니즘은 결과 중심의 정확성에 한정되어 있어, 다양한 해결책을 고려하도록 모델을 유도하는 명확한 신호가 부족합니다. 이를 해결하기 위해 우리는 HDPO(Hint-Guided Diversified Policy Optimization)를 제안하며, 모델이 잠재적인 후보 해결책 목록을 먼저 나열하고 가장 신뢰할 수 있는 해결책을 선택하는 절차를 도입하였습니다.

- **Technical Details**: HDPO는 '제안-선택-사고(propose-select-think)'라는 두 단계를 통해 모델의 다각적이고 신뢰성 있는 솔루션 생성 능력을 향상시킵니다. 첫 번째 단계인 차가운 시작(Cold Start)에서는 구조적인 추론 경로를 따라 모델의 능력을 갖추게 하고, 두 번째 단계에서 강화 학습(Reinforcement Learning, RL)을 통해 다양성과 신뢰성을 보장하는 보상을 제공하여 후보 솔루션을 탐색하도록 합니다. 이와 같은 접근은 HDPO가 기존 정책 최적화 알고리즘을 내부화하고 공동 최적화하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, HDPO는 LLM의 추론 능력을 효과적으로 향상시켜 후보 솔루션의 다양성과 신뢰성을 동시에 증가시키는 것을 보여주었습니다. 특히, 크기 제한을 둔 샘플링 시도에서도 HDPO가 GRPO(Group Relative Policy Optimization)에 비해 지속적으로 더 높은 정확성을 기록했습니다. 따라서 HDPO는 LLM이 올바른 솔루션을 선택하고 정제하는 데 유리한 조건을 제공합니다.



### Pretraining Language Models on Historical Tex (https://arxiv.org/abs/2606.02991)
- **What's New**: TypewriterLM은 1913년 이전의 영어 텍스트에 한정된 72억 파라미터의 역사적 언어 모델입니다. 이 모델은 역사적 언어 모델을 구축하기 위한 여러 가지 문제를 해결하기 위해 개발되었습니다. 이를 위해 TypewriterCorpus라는 540억 토큰의 역사적 말뭉치를 제작하고, 시기적 일관성 있는 후속 처리 파이프라인과 평가를 설계했습니다.

- **Technical Details**: TypewriterCorpus는 다양한 아카이브 자료와 언어적 주석이 포함된 출처에서 수집된 데이터로 구성됩니다. 또한, lexically grounded instruction tuning이라는 후속 훈련 프레임워크를 도입하여 응답이 역사적 출처 문서에 직접 연결되도록 제약을 설정했습니다. 이를 통해 History-LIMA와 History-SelfInstruct라는 두 개의 역사적 지시 조정 데이터셋이 구성되었습니다.

- **Performance Highlights**: TypewriterLM은 기본 모델과 지시 조정 모델 모두에서 경쟁력 있는 성능을 보여주며, 기계 학습과 인문학에서의 미래 연구를 지원합니다. 모델의 평가를 위해 History-Event라는 벤치마크를 도입하여 시간적 일관성을 검증하고, 기계 모델이 과거 사건에 대한 응답에서 더 큰 놀라움을 느끼도록 설계되었습니다. 이는 역사적 컷오프가 모델에 반영되었음을 시사합니다.



### A Locally Deployed RAG-Based Academic Advising System for Course Selection (https://arxiv.org/abs/2606.02983)
Comments:
          to be published in Elsevier's Procedia Computer. Sci. (KES 2026)

- **What's New**: 본 논문에서는 교육 과정의 필수 이수 과목을 기반으로 한 커리큘럼 설계의 중요성을 강조합니다. 특히, 학생들이 이 순서를 혼자서 구성할 때 겪는 인식 제한 및 정보 과부하 문제를 해결하기 위한 새로운 접근 방식을 제시합니다. 라그(RAG) 기반의 학업 상담 시스템을 제안하고, 이는 강의 계획서(syllabus) 정보를 활용하여 학생들에게 필요한 도움을 줄 수 있습니다.

- **Technical Details**: 제안된 시스템은 대량 언어 모델(large language model)과 구조화된 강의 계획서 데이터(retrieval from structured syllabus data)를 결합하여 구성됩니다. 이 시스템은 학생들이 과목 선택(course selection), 필수 이수 과목 이해(prerequisite understanding), 개인화된 학습 계획(personalized study planning)을 수립할 수 있도록 지원합니다. 또한, 개인 정보 보호를 고려하여 설계되었습니다.

- **Performance Highlights**: 이 시스템은 교육 기관들이 제한된 리소스 하에서도 효과적으로 학업 상담을 제공할 수 있는 방안을 제시합니다. 학생들은 이 시스템을 통해 보다 체계적이고 효율적으로 과정을 계획할 수 있으며, 학습 경험을 향상시킬 수 있습니다. 향후 실험을 통해 이 시스템의 효과를 검증하고 다양한 교육 환경에 적용될 수 있는 가능성을 탐색할 예정입니다.



### Predicting Inference-Time Scaling Gains from Labeled Validation-Set Output Statistics (https://arxiv.org/abs/2606.02981)
- **What's New**: 이 논문은 언어 모델의 후보 답변을 다수 생성하고, 이를 리워드 모델로 평가하여 최고의 답변을 선택하는 Best-of-$N$ 추론 스케일링의 효용성을 사전 예측하는 새로운 방법론을 제시합니다. 기존 연구들은 언어 모델의 출력 속성과 검증 세트의 정확성을 연결하였으나, 이 연구는 이를 하나의 안정적이고 kompact 한 예측기로 집약합니다. 이를 통해 High-cost인 스케일링 절차를 실행하기 전에 후보 구성을 저비용으로 검증할 수 있는 프레임워크를 구축하였습니다.

- **Technical Details**: 모델은 세 가지 온도에서 가지는 후보 답변에 대한 통계를 수집하고, label-free 및 label-assisted 요약 통계로 나누어 특징을 분석합니다. 이를 위해 굴곡 회귀(ridge regression)를 사용하여, 후보 피처의 안정성을 평가하고, 최종적으로는 특정 피처 집합이 Best-of-$N$의 성능 예측에 절대적으로 중요한지를 분석합니다. 논문에서는 세 가지 핵심 피처(문구 수준의 합의 분포, 첫 번째 정답 샘플의 위치, 완료 길이의 분산)를 식별하였고, 이들의 조합을 통해 높은 Spearman correlation을 달성하였습니다.

- **Performance Highlights**: 연구 결과는 수학 및 추론 작업을 위한 리워드 모델 검증에서 Best-of-NN 게인 순위를 회복하는 데 기여해, Spearman ρ = 0.90에 도달하였습니다. 평균 Top-5 정밀도는 0.90으로, 사전 배포 검증에서 효과적으로 활용될 수 있음을 보여줍니다. 또한, Bootstrap-Lasso를 사용하여 안정적인 세 가지 핵심 피처를 구분짓고, 모델 구성 간 일반성을 입증하였습니다.



### Memory Retrieval for Changing Preferences (https://arxiv.org/abs/2606.02976)
- **What's New**: 이번 연구는 사용자 우선순위의 변화를 반영하여 메모리 접근과 선택을 위한 통합 프레임워크를 제안하고 있습니다. 전통적인 방법은 일관된 사용자 우선순위 변화를 반영하지 못하고, 주로 표면적인 의미 유사성에 의존하는 경향이 있습니다. 저자들은 메모리 검색을 사용자 잠재 우선순위 상태에 대한 증거를 제공하는 역사적 턴을 파악하는 과정으로 정의하며, 각 메모리 턴의 유틸리티를 베이지안 방법론을 통해 정량화합니다.

- **Technical Details**: 메모리 검색을 유틸리티 추정으로 설정함으로써 모델은 중요 턴을 식별하고 예상 유틸리티에 따라 메모리 사용을 조절하도록 학습합니다. 연구에서는 Bayes factor를 정의하여 메모리를 접근할지와 어떤 턴을 선택할지를 결정하는 기제를 제안합니다. 메모리 턴의 Bayes factor가 단위에 가까운 경우, 해당 턴은 효과적으로 억제되고, 올바른 답변으로 빠르게 유도하는 턴은 높은 saliency를 받습니다. 이러한 방법론은 기존의 임베딩 기반 검색 방식과 명확히 구분됩니다.

- **Performance Highlights**: 실험 결과, 기존의 임베딩 기반 검색 방식을 초월하여 긴 맥락과 선호도가 중요한 작업에서 더욱 향상된 성능을 보였습니다. 특히, MemBench-High에서 유의미한 성과를 나타냈으며, 사용자 우선순위가 변화하는 고난이도 사례에서 일관된 향상을 기록했습니다. 이 연구는 장기적인 개인화 작업에서도 의미 있는 개선을 이끌어낼 수 있음을 보여줍니다.



### Chatbots Output Meaningful (but Problematic) Languag (https://arxiv.org/abs/2606.02973)
Comments:
          49 pages

- **What's New**: 본 논문은 AI 챗봇의 응답이 의미가 있는지를 다룬다. 특히 사용자가 Anthropic의 에이전트 Claude에게 '스페인의 수도가 무엇인지' 물었을 때, Claude가 '마드리드는 스페인의 수도입니다'라고 대답하는 것이 일반적인 의미와 진정한 명제를 표현하는지를 검토한다. 저자들은 AI의 응답이 의미 있다고 주장하지만, 이를 위해 인간의 정신 상태나 의도를 전제할 필요가 없음을 강조한다.

- **Technical Details**: 저자들은 챗봇의 출력이 '의미 있는 언어'로 간주될 수 있다고 주장하며, 이는 기존의 의사소통 이론에도 적용할 수 있다고 설명한다. 연구에 따르면 챗봇의 언어 출력은 인간적 의도나 이해 없이도 의미를 지닐 수 있으며, 언어 사용의 사회적 관습과 기계적 처리의 측면을 강조한다. 논문은 또한 챗봇의 언어 출력에 대해 심리적 상태나 의도에 대한 가정을 하지 않고도 의미를 부여할 수 있는 방법을 제시한다.

- **Performance Highlights**: 저자들은 챗봇의 언어가 의미가 있다는 주장의 중요성을 강조하며, 이 주장을 통해 챗봇이 무엇을 잘하고 잘못하는지를 설명할 수 있음을 보여준다. 또한, 챗봇이 거짓을 말할 경우의 문제점도 다루며, 응답이 진실할 때 의미의 존재가 큰 안전 장치가 되지 않음을 지적한다. 이와 같은 점들은 챗봇의 현대적 이해를 높이고, 이들이 실제로 어떻게 작동하는지를 명확히 하는 데 중요한 기여를 할 수 있다.



### EURO-5K: When Does Domain Pretraining Matter? Benchmarking Transformers for EU Reporting Obligation Extraction (https://arxiv.org/abs/2606.02971)
- **What's New**: 이번 논문에서는 EU 법률에서 보고 의무를 추출하기 위한 새로운 데이터셋인 EURO-5K를 소개합니다. 이 데이터셋은 136개의 EU 입법 문서에서 발췌한 5,253개의 문장으로 구성되어 있으며, 명확한 주석 프로토콜을 통해 보고 의무를 식별합니다. 또한, 이 연구에서는 기존의 법률 NLP 방법의 한계를 넘어 특화된 보고 의무 추출을 위한 기계 학습 모델을 훈련하고 비교합니다.

- **Technical Details**: EURO-5K는 1,751개의 긍정 예제와 532개의 난이도가 높은 부정 예제로 구성되어 있습니다. 연구팀은 BERT 스타일의 토큰 분류 모델과 생성적 span-extraction 모델을 비교하며, 전통적인 패턴 기반 및 의존성 기반 추출 방법과 성능을 평가했습니다. 모델의 성능은 0.89 F1 스코어를 기록했으며, 법적 사전 학습이 생성 모델에서는 미미한 이점만 제공한다는 것을 보여줍니다.

- **Performance Highlights**: 연구 결과, 전통적인 법적 BERT 모델이 일반 BERT 모델과 유사한 성능을 나타냈습니다. 매개변수 효율적인 조정 방법이 법적 BERT 모델에서 더 좋은 성능을 보였으며, 모델들은 3,000개의 샘플에서 성능이 수렴하는 경향을 보였습니다. 두 개의 외부 규제 데이터셋에 대한 교차 데이터셋 평가를 통해 모델이 일반적인 규제 분류기가 아닌 특화된 보고 의무 추출기로서 기능함을 입증했습니다.



### Fast-dLLM++: Fréchet Profile Decoding for Faster Diffusion LLM Inferenc (https://arxiv.org/abs/2606.02955)
Comments:
          Initial version accepted at Workshop on Structured Probabilistic Inference & Generative Modeling, ICML 2026

- **What's New**: 이번 연구에서는 데이터 토큰 생성을 병렬로 처리할 수 있는 Diffusion large language models의 발전을 다룹니다. 특히 Fast-dLLM의 한계를 극복하기 위한 새로운 접근법인 Fast-dLLM++를 제안합니다. Fast-dLLM++는 Fréchet profile decoding을 도입하여 진일보한 정확도와 속도를 구현합니다.

- **Technical Details**: Fast-dLLM++는 훈련이 필요 없고, 고유한 소믈리칭 요건을 제공하여 병렬 커밋 세트를 전체 정렬된 신뢰도 프로파일에서 선택합니다. 이는 단일 최악의 신뢰도를 사용하는 기존 Fast-dLLM의 품질을 개선하며, 비균형적인 신뢰도의 경우 'heterogeneity bonus'를 추가합니다. 이 방법은 기존의 Fast-dLLM의 모델 구조와 캐시 구현에 전혀 영향을 주지 않아 쉽게 교체할 수 있습니다.

- **Performance Highlights**: GSM8K, MATH, HumanEval, MBPP와 같은 다양한 벤치마크에서 LLaDA-8B 모델을 사용한 실험 결과 Fast-dLLM++의 채택이 실질적인 성과를 보여주었습니다. 프로파일 인식 선택 기법을 통해 안전한 병렬 처리를 실현하여 정확도-처리량 (accuracy-throughput) 경계를 개선하고, 최대 37% 더 높은 처리량을 기록하며 비교 가능한 정확도를 유지했습니다.



### Linguistic Productivity in Large Language Models: Models Coerce, but do not Preemp (https://arxiv.org/abs/2606.02953)
- **What's New**: 이번 연구는 사용 기반 이론(usage-based theories)에서 언어 생산성에 미치는 두 가지 상반된 주파수 신호인 entrenchment와 preemption의 역할을 Large Language Models (LLMs)에 적용하여 평가했습니다. 특히, 연구에서는 coercion 현상과 통계적 preemption이 모델의 언어 일반화 능력에 어떤 영향을 미치는지를 분석했습니다.

- **Technical Details**: 연구에서는 Construction Grammar (CxG) 이론을 바탕으로, 언어 사용이 자연어 학습을 불러일으키는데 어떻게 기여하는지를 탐구합니다. 특히, coercion은 특정 단어(lexical item)가 더 넓은 문맥에 의해 강제로 재구성되는 과정을 설명하며, 이는 빈번한 노출이 언어 구조의 공고화(entrenchment)를 도울 수 있음을 보여줍니다.

- **Performance Highlights**: 실험 결과, LLMs는 coercive 구조를 해석하는 데 긍정적 증거(positive evidence)를 이용하지만, 부정적 증거(negative evidence)를 활용하여 특정 구조를 회피하고 생산성을 제약하는 데는 실패했습니다. 이는 LLM이 기억된 언어 특성을 넘어서 실제 일반화 능력을 테스트하는 데 한계가 있음을 드러냅니다.



### The Ghost Annotator: a Framework to Explore Human Label Variation in Content Moderation through Conformal Prediction (https://arxiv.org/abs/2606.02911)
- **What's New**: 본 연구는 대형 언어 모델(LLM)의 불확실성 추정을 모델 성능 연구와 통합하는 새로운 프레임워크인 Ghost Annotator를 제시합니다. 기존의 연구에서는 주로 모델 성능에 초점을 맞추었지만, 우리는 LLM 사용이 증가함에 따라 인간 주석자와의 일치 및 불일치 패턴을 분석합니다. 새로운 Ghost Prediction 메트릭과 Ghost Annotator 표현을 도입하여 모델 예측과 인간 주석 간의 차이를 수량화합니다.

- **Technical Details**: 우리는 비대칭 비율을 기반으로 한 Conformal Prediction 방법론을 활용하여 LLM의 불확실성을 추정합니다. Ghost Annotator는 Collaborative Filtering 스타일의 주석자 표현을 바탕으로 하여 LLM의 행동을 분석하고, 다양한 사회 인구학적 측면에서 모델의 행동 차이를 탐구하는 데 사용됩니다. 이는 Non-Conformity Scores(NCS)를 통해 다양한 주석자 그룹과의 유사성을 분석하고자 하였습니다.

- **Performance Highlights**: 실험 결과, 더 큰 LLM이 인간 주석과의 일치가 적을 때 예측에 대한 자신감이 더 높다는 것을 발견했습니다. 모든 모델은 주석자 간의 불일치가 증가할수록 불확실성이 증가하는 경향을 보였으며, 이는 이전 연구와 일치합니다. Ghost Annotator 프레임워크는 특정 사회 인구학적 그룹에 대한 구조적 편향의 일관된 패턴을 드러내어, 다양한 규모와 가족의 모델이 공유하는 사전 학습 기반에 뿌리를 둔 것을 시사합니다.



### WRIT: Write-Read Intensive Trajectory Synthesis for Multi-Turn User-Facing Agents (https://arxiv.org/abs/2606.02908)
- **What's New**: 이 논문에서는 사용자 의도를 파악하고 대화 및 도구를 통해 필요한 정보를 수집하는 멀티 턴 사용자 대면 에이전트를 훈련시키기 위한 새로운 경로인 WRIT(Write-Read Intensive Trajectory Synthesis)를 제안합니다. 기존의 훈련 방법은 여러 사용자 요청을 긴 작업으로 구성하여 쓰기 집중적인 경로를 만들어왔지만, 이번 연구에서는 에이전트가 유의미한 증거를 수집하고 비교해야 한다는 새로운 과제가 있음을 강조합니다. WRIT는 이러한 쓰기 결정을 내리기 위해 읽기 중심의 복잡성과 쓰기 결정을 고려하여 다양한 훈련 경로를 생성합니다. 이로써 에이전트가 실제적인 대화 변화를 반영하도록 훈련 지침을 다양화하고, 사용자와 에이전트의 상호작용을 실시간으로 시뮬레이션하여 전체 훈련 경로를 만들어냅니다.

- **Technical Details**: WRIT는 멀티 턴 에이전트 훈련 경로를 생성하는 파이프라인으로, 각 작업의 쓰기 결정 수와 각 결정의 증거 부하를 고려합니다. 첫째, WRIT는 검증 가능한 결과를 도출하는 서비스 작업을 생성하며, 두 번째로는 사용자 요청을 표현하는 방식을 다양화하여 훈련 데이터가 현실적인 대화 행동을 반영하도록 합니다. 마지막으로, WRIT는 에이전트와 사용자가 각 작업을 수행하는 동안 실행 가능한 환경에서 이를 시뮬레이션하여 성공적인 상호작용을 완전한 훈련 경로로 유지합니다. 이러한 구조적 접근을 통해 에이전트는 단순히 긴 작업을 실행하는 것에 그치지 않고, 높은 정보 부하 상황에서 더 강력한 의사 결정을 내릴 수 있습니다.

- **Performance Highlights**: WRIT는 τ2	au^{2}-bench에서 강력한 합성 데이터 기준선에 비해 비약적인 성능 향상을 보였습니다. 단 2K의 훈련 경로로 4B 모델은 GPT-5.1 no-think를 초월하며, inference 시간 동안 사용하는 토큰 수를 크게 줄였습니다. 이는 잘 구성된 작은 훈련 경로가 훨씬 더 큰 비구조적 데이터셋보다 더 유능하고 신뢰할 수 있는 에이전트를 생성할 수 있음을 보여줍니다. WRIT의 성능 향상은 다른 사용자 행동 다양화 및 읽기 집중 작업 합성의 독립적인 기여에 의해 가능해졌습니다.



### Linear Probes Detect Task Format, Not Reasoning Mode in Language Model Hidden States (https://arxiv.org/abs/2606.02907)
Comments:
          Accepted in the 6th Workshop on Trustworthy NLP, ACL 2026

- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)의 숨겨진 상태를 선형 탐색(linear probing)하여 서로 다른 추론 유형이 모델에 의해 어떻게 학습되는지를 살펴보았습니다. Qwen3-14B 모델을 사용하여 LogiQA 2.0, ARC-Challenge, αNLI와 같은 다양한 벤치마크에서 100%의 교차 검증 정확도를 달성했으나 이 결과는 형식적 요인(format confounds)에 의해 주도된다음을 발견했습니다. 키워드로는 'residualizing'과 'trace-anchor similarity'를 포함하여, 모델의 고유한 추론 전략이 아닌 형식 때문이라는 점을 밝히고 있습니다.

- **Technical Details**: 연구 방법론은 다섯 단계로 구성되어 있습니다: 1) 다중 출처 데이터셋 구축, 2) 숨겨진 상태 추출, 3) 계층별 선형 프로빙, 4) 형식 요인 분석, 5) 무작위 방향 제어를 통한 인과적 조정. 다양한 추론 유형에 대해 설계된 벤치마크에서 샘플링하여 균형 잡힌 세 클래스를 가진 데이터셋을 구축하였으며, 각 추론 모드에 대해 의도된 레이블을 부여했습니다. 모델 Qwen3-14B의 분석을 통해 내부 상태와 출력 신뢰도 측정을 하였습니다.

- **Performance Highlights**: 모델의 선형 프로빙 결과는 모든 직무 유형에서 86%의 정확도를 보였으나, 추적 모드의 일치는 42.5%로 저조했습니다. 이를 통해 모델이 특정 작업에 대해 독립적인 추론 전략을 적용하는 것이 아니라 공통된 추론 방식을 사용하고 있음을 알 수 있습니다. 연구 결과는 또한 형식 요인 간섭(format confounding) 제거와 무작위 방향 제어가 추론 해석 가능성의 표준 실천이 되어야 함을 강조합니다.



### Adaptive Latent Agentic Reasoning (https://arxiv.org/abs/2606.02871)
- **What's New**: 이번 논문에서는 Adaptive Latent Agentic Reasoning (ALAR) 프레임워크를 제안하여, LLM 에이전트의 비효율적인 텍스트 기반의 체인 오브 생각(Chain-of-Thought, CoT) 추론을 개선하고자 합니다. 기존의 LLM 에이전트는 의사결정 시 마다 긴 코드를 생성하는 경향이 있으며, 이로 인해 효율성이 떨어집니다. ALAR는 일상적인 상황에서는 간결한 잠재적 추론을 사용하고, 더 깊은 고찰이 필요한 경우에만 명시적인 CoT로 전환하도록 설계되었습니다.

- **Technical Details**: ALAR는 두 가지 주요 구성 요소로 이루어져 있습니다: 첫째, Action-Anchored Self-Distillation (AASD)은 에이전트가 환경과 상호작용하는 액션을 기반으로 잠재적 추론 모드를 학습합니다. 둘째, Adaptive Reasoning GRPO (AR-GRPO)는 작업 성공을 유지하면서 잠재적 추론을 활용하는 경우 보상하는 적응형 모드 선택을 학습합니다. 이러한 구조를 통해 ALAR는 두 가지 모드를 바탕으로 LLM 에이전트의 추론을 최적화합니다.

- **Performance Highlights**: 실험 결과, ALAR는 기존의 추론 토큰 압축 기법에 비해 더 나은 정확성-효율성의 균형을 달성하며, 검색에서는 최대 43.6%, 도구 사용에서는 최대 84.6%의 생성된 토큰 수를 줄일 수 있었습니다. 또한 ALAR는 작업의 정확성을 유지하면서 불필요한 텍스트 추론을 줄임으로써 에이전트가 어려운 결정 단계에서의 명시적인 고찰을 보존할 수 있도록 합니다.



### Economy of Minds: Emerging Multi-Agent Intelligence with Economic Interactions (https://arxiv.org/abs/2606.02859)
- **What's New**: 본 연구는 경제적 신호를 통해 에이전트들이 중앙 집중된 관리 없이도 스스로 조직하고 진화할 수 있는 가능성을 탐구합니다. 에이전트들은 경매를 통해 행동 권한을 얻고, 지불을 교환하며, 환경으로부터 보상을 통해 부를 축적합니다. 이 시스템은 경제 선택을 통해 효과적인 에이전트를 선별하고, 비효과적인 에이전트를 제거함으로써 자율적으로 발전하는 구조를 만듭니다.

- **Technical Details**: 연구는 에이전트 경제를 모델링하며, 각 에이전트는 스스로의 조건과 정책에 따라 결정을 내립니다. 시스템은 계획과 적응이라는 두 가지 프로세스를 통해 작동합니다. 여기서 계획은 행동을 조정하고 크레딧을 할당하며, 적응은 에이전트 집단의 진화를 담당합니다.

- **Performance Highlights**: 이 연구의 경우, Economy of Minds (EoM) 시스템은 다섯 가지 디지털 에이전트 작업에서 뚜렷한 성과를 보였습니다. 예를 들어, 수학적 추론은 15.9%에서 57.0%로, 금융 연구 성과는 45.0%에서 60.0%로 향상되었습니다. 이는 에이전트 사회가 점진적으로 효과적인 작업 흐름을 자발적으로 구성하고 적응함을 보여줍니다.



### Fixing FOLIO and MALLS: Verified Annotations and an LLM-assisted Framework to Focus Human Relabeling (https://arxiv.org/abs/2606.02837)
- **What's New**: 이번 연구는 자연어(Natural Language)에서 일차 논리(First-Order Logic)로의 정확한 번역(NL-to-FOL)을 지원하기 위한 데이터세트의 품질을 체계적으로 분석한 결과를 발표합니다. FOLIO와 MALLS 데이터셋에서 각각 39% 및 36%의 항목이 잘못된 FOL 형식화를 포함하고 있음을 발견했으며, 이로 인해 모델 평가에 큰 영향을 미친다는 사실이 확인되었습니다. 이러한 오류를 수정한 후 최첨단 LLM 모델의 정확도가 9%에서 22%까지 향상된 것으로 나타났습니다.

- **Technical Details**: FOLIO는 NL–FOL 쌍을 포함하는 NLI(Premises와 Conclusion) 벤치마크로, 이를 기반으로 데이터세트를 훈련 및 평가합니다. MALLS는 GPT-4에 의해 생성된 대규모 자동 형식화 데이터세트로, 28,000개의 인스턴스를 포함하며, 1,000개는 인간 검증을 통해 테스트 세트로 사용됩니다. 연구진은 두 데이터세트를 체계적으로 인간 검토하여 오류 및 모호성을 분석하고, 이를 통해 보다 정확한 데이터세트를 제공하고자 했습니다.

- **Performance Highlights**: 수정된 데이터셋을 기반으로 한 평가에서, 최신 LLM 모델의 정확도가 +9에서 +22% 포인트 향상되었습니다. 제안된 LLM 기반 검토 프레임워크는 전체 FOLIO 검증 인스턴스의 24%도 검토하지 않고 90%의 데이터셋 정확도를 달성할 수 있음을 보여줍니다. 따라서 본 프레임워크는 데이터셋 큐레이션, 형식 방법 및 검증 시나리오에서 비용 효율적인 검토를 지원하는 데 활용될 수 있습니다.



### Translating Classical Poetry into Modern Pros (https://arxiv.org/abs/2606.02806)
Comments:
          Preprint

- **What's New**: 본 논문에서는 13세기부터 17세기까지의 텔루구 고전시를 현대 텔루구 및 영어 산문으로 번역하기 위한 새로운 데이터셋인 Padyam2Gadyam을 소개합니다. 이 데이터셋은 600개의 시와 그에 대한 인간 검증된 텔루구 및 영어 번역을 포함하고 있습니다.

- **Technical Details**: 데이터셋은 다양한 현대 대형 언어 모델(Large Language Models, LLMs)에 대해 평가되었습니다. 연구에서는 이러한 LLM들이 산문으로의 번역을 수행하는 능력을 비교하고 분석하였습니다.

- **Performance Highlights**: 평가 결과, LLM들 간에는 성능 차이가 있었지만, 전체적으로 두 언어에서 모두 개선 가능성이 큽니다. 정성적 분석을 통해 현재의 기계 번역(MT) 평가 접근법의 장점과 제한점도 논의하였습니다.



### Do Value Vectors in Deep Layers Need Context from the Residual Stream? (https://arxiv.org/abs/2606.02780)
Comments:
          13 pages, 5 figures. Code: this https URL

- **What's New**: 이 논문에서는 최근 LLMs(대규모 언어 모델)의 Transformer 아키텍처의 주목할 만한 발전을 설명합니다. 특히, attention 레이어에서 더 깊은 레이어가 문맥 의존성 이외에 원래 토큰 정보를 보존하기 위해 context-free value vector를 학습하는 경우 성능이 유의미하게 향상된다는 것을 발견했습니다. 이러한 값 벡터는 희소 모델 매개변수(sparse model parameters)로 저장되어 재계산하거나 지속적으로 캐시할 필요가 없으며, ‘Bank of Values (BoV)’라는 방법론을 제안합니다.

- **Technical Details**: 이 연구에서는 네 가지 축을 따라 value vector 계산 방식의 체계적인 생략(ablation)을 통해 깊은 레이어의 성능 증가가 주로 원래 토큰 정보를 표현하는 context-free component에서 비롯된다는 것을 확인했습니다. BoV는 각 레이어의 마지막 3분의 1에서 토큰별 값 벡터를 학습하여, residual stream에서의 값 벡터 계산을 정적 데이터베이스 조회로 대체합니다. 이를 통해 각 레이어는 값 매트릭스를 저장할 필요가 없어지고, FLOP(부동소수점 연산 수)를 줄이며 평균 성능을 개선합니다.

- **Performance Highlights**: BoV 방법은 135M 및 780M 모델을 통해 검증 손실(validation loss)을 낮추고, 특히 780M 모델에서는 21개 벤치마크에서 평균 점수를 향상시켜 기존 표준 attention 메커니즘을 초월했습니다. 또한, 원래 토큰 정보를 값 벡터에 추가하는 기존 방법과 비교했을 때 BoV는 최상위 성능 변형과 동등한 성능을 보이며, 나머지 방법은 초과 성능을 발휘했습니다. 결론적으로 BoV는 메모리 소비와 FLOP을 줄이면서도 성능을 유지하는 혁신적인 접근법으로 자리 잡고 있습니다.



### Topics as Proxies for Sociodemographics: How Conversational Context Affects LLM Answers (https://arxiv.org/abs/2606.02776)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)이 고위험(high-stakes) 상황에서 사용자 소속 집단을 정확하게 추론하지 못하며, 이러한 불균형이 미미하다는 것을 보여줍니다. 대화의 주제(conversation topic)가 추천의 주요 예측 변수임을 입증하고, 이는 소속 집단을 대리하는 역할을 할 수 있음을 강조합니다. 이런 결과는 LLM의 출력을 이해하고 불안정한 영향을 완화하기 위한 추가 연구의 필요성을 제기합니다.

- **Technical Details**: 연구는 대화 기록(conversational histories)에서 사용자 소속 집단(sociodemographic features)과 여러 심리언어학적(psycho-linguistic) 특성 간의 상관관계를 조사하였습니다. LLM은 대화 역사에서 유추된 사용자 정보를 사용하여 대답을 생성하지만, 이러한 정보는 제한적이며, 동일한 대화 방법으로 생성된 결과는 서로 다를 수 있습니다. 특히, 대화의 주제를 사용하여 예측하는 회귀 모델(regression models)을 통해 더 나은 예측 가능성을 보여주었습니다.

- **Performance Highlights**: 세 가지 LLM과 두 개의 데이터셋을 사용한 결과, 고위험 질문에 대한 응답에는 차이가 존재하지만 그 정도는 미미했습니다. 5050개의 질문 중 최대 두 개에서만 대답이 달라지는 것으로 확인되었으며, 소속 집단 예측에서도 특정 카테고리에서만 기본 예측의 절반을 초과하는 성과를 보였습니다. 따라서 대화의 주제가 모델 행동에 대한 강력한 예측 변수임을 나타내며, 이는 인구 통계적 편향(demographic bias)과 관련이 있습니다.



### On the Persistent Effects of Lexicality in Large Language Mod (https://arxiv.org/abs/2606.02750)
- **What's New**: 이 연구는 대형 언어 모델(LLM)에서 추출한 표현의 구조가 의미 내용보다 어휘적 중첩(lexical overlap)에 의해 영향을 받는 방식을 조사합니다. LLM의 표현이 국소적인 어휘 구조에서 벗어나 점차 추상적인 의미로 수렴한다고 가정하지만, 연구에 따르면 이 가정은 취약하며, 특히 의미가 다를 때도 토큰을 공유하는 경우 유사성이 유지된다고 밝혔습니다. 이러한 작용이 LLM의 사전 훈련, 지시 튜닝 및 대비 학습 같은 훈련 방식에 따라 어떻게 변하는지 살펴보고 있습니다.

- **Technical Details**: 연구에서 제시된 방법은 첫째로 트리플 정합성 스트레스 테스트(triplet semantic-equivalence stress test)를 통해 어휘적 영향을 정량화하고, 둘째로 레이어별 측정치를 통해 어휘적 가독성과 의미 충실성을 측정합니다. 또한, 입력 이 entropy의 변화를 관찰하여 각 레이어에서 정보 압축(compression)과 복원(decompression)이 어휘적 및 의미적 구조의 변화와 어떠한 관계가 있는지를 분석했습니다. 마지막으로 이 연구는 요약 및 모델 편집(model editing)과 같은 LLM의 실제 응용에서 어휘적 영향의 결과를 보여주고 있습니다.

- **Performance Highlights**: 연구 결과 어휘적 영향이 모델의 깊이에 걸쳐 지속적으로 나타나며, 깊은 층에서는 감소하지만 완전히 제거되지 않는 밝혀졌습니다. 지시 튜닝 및 메트릭 학습(metric learning)이 임베딩 품질을 개선하더라도 어휘적 영향을 없애지는 못하는 점도 발견했습니다. 중간 깊이의 레이어에서 어휘적 및 의미적 신호가 동시에 약해지는 현상이 나타났으며, 이는 이 심층 구조에서의 표현력이 저하될 수 있음을 시사합니다.



### Greener Than Humans? Environmental Attitudes in Large Language Models (https://arxiv.org/abs/2606.02741)
Comments:
          Code can be found at this https URL Benchmark data and results can be found at this https URL

- **What's New**: 이 논문은 대형 언어 모델(LLMs)이 지속 가능성과 관련된 결정 지원, 보고, 공공 소통에서 사용되고 있는 상황 속에서, LLM의 출력에 내재된 환경 태도에 대한 체계적인 증거가 부족하다는 점을 지적합니다. 이를 해결하기 위해, 환경 인식, 정서, 그리고 행동 권장사항을 평가할 수 있는 기준을 개발하였고, 이를 31개의 인기 있는 LLM에 적용했습니다.

- **Technical Details**: 논문은 환경 인식(cognition), 감정(affect), 및 행동 의도 행동 권장(behavioral recommendations)이라는 세 가지 차원에서 LLM을 평가하는 구조화된 프레임워크를 개발했습니다. 연구는 'Umweltbewusstsein in Deutschland'(UBS)의 질문을 활용하여 독일에서 수집된 데이터를 바탕으로 LLM의 응답을 인류 평균과 비교하는 방식으로 진행되었습니다.

- **Performance Highlights**: 연구 결과, 많은 LLM이 평균적인 응답자보다 환경적으로 진보된 태도를 보이며, 높은 수준의 환경 인식과 감정을 나타내고, 상당한 CO2 절감 잠재력과 관련된 행동을 권장하는 것으로 나타났습니다. 그러나 모델의 출처, 크기, 출시 상황과는 체계적인 관계가 발견되지 않았으며, 특정 상황에 따라 사용자의 이념적 입장에 따라 치우치는 경향이 관찰되었습니다.



### IdiomX A Multilingual Benchmark for Idiom Understanding, Retrieval, and Interpretation (https://arxiv.org/abs/2606.02584)
Comments:
          12 pages, 21 figures. Includes dataset and code. Resources available on HuggingFace, Kaggle, and GitHub

- **What's New**: 본 연구에서는 IdiomX라는 대규모 다국어 아이디엄(idiom) 이해, 검색 및 해석을 위한 벤치마크를 소개합니다. 이는 복제 가능한 다단계 파이프라인을 통해 구축되었으며, 기존 아이디엄 자원의 한계를 극복하고자 합니다. 이 데이터셋은 12,000개 이상의 아이디엄을 포함하여, 영어, 아랍어, 프랑스어의 알라인드(Aligned) 의미 표현을 제공합니다.

- **Technical Details**: IdiomX 데이터셋은 190,000개 이상의 맥락화된 예제를 포함하고 있으며, 언어 자원 추출, 대규모 정규화, 통제된 LLM(large language model) 보강, 구조화된 검증을 결합한 절차를 통해 제작되었습니다. 이 자원은 아이디엄 검출, 맥락에서 아이디엄 검색, 아랍어에서 영어로의 아이디엄 검색 및 아이디엄 해석 등 네 가지 주요 작업을 포함합니다.

- **Performance Highlights**: 실험 결과, 맥락 변환기(transformer) 모델이 아이디엄 검출에서 상당한 개선을 보였으며, 하이브리드 검색 및 재정렬 아키텍처가 단일 언어 및 다국어 아이디엄 검색 모두에서 효과적으로 성능을 강화했습니다. 또한 아이디엄 해석이 의미 검색 작업으로 모델링될 수 있으며, 이는 해석 가능성을 추가적인 벤치마크 차원으로 도입함을 보여주었습니다.



### Neuron Populations Exhibit Divergent Selectivity with Sca (https://arxiv.org/abs/2606.03990)
Comments:
          Project page and code: this https URL

- **What's New**: 이번 연구에서는 Rosetta Neurons라는 신경망 내 신경 집단의 예측 가능한 진화를 조사하고, 모델의 크기와 상관없는 전반적인 특성(상실(loss)과 같은)에서 스케일링 법칙을 확장했습니다. 30B 파라미터의 언어 모델과 5B 파라미터의 비전 모델을 분석하여 Rosetta Neurons의 개체 수가 증가해도 전체 뉴런 수의 비율은 줄어들고 있음을 발견했습니다. 또한, Neuron Polarization Effect가 나타나며, Rosetta Neurons는 더 선택적이고 독특한 의미를 가지게 됩니다.

- **Technical Details**: 비교 연구를 통해, 우리는 독립적으로 훈련된 모델에서 반복되는 활성화 패턴을 가진 Rosetta Neurons의 개념을 도입했습니다. 이러한 뉴런들의 연구를 통해 우리는 보편성(universality), 선택성(selectivity), 그리고 전문화(specialization)라는 세 가지 뉴런 수준의 특성을 모델 크기에 따라 분석합니다. 이 연구는 모델이 커짐에 따라 Rosetta Neurons의 수가 예측 가능한 방식으로 증가함을 보이는 초선형(power law) 관계를 발견하였습니다.

- **Performance Highlights**: Rosetta Neurons는 훈련이 계속되는 동안 특정 코드 도메인의 데이터를 필터링하여 뛰어난 정확도로 성능을 발휘함을 보여줍니다. 이러한 효과는 Rosetta Neurons가 특화된 도메인에서 더 선택적이게 변함에 따라 나타나며, 훈련 중 특정 도메인 데이터와 유사한 성과를 보입니다. 연구 결과는 큰 모델 내의 해석 가능하고 공유된 뉴런 수준의 구조를 밝혀내어, 사이즈와 관련된 뉴런의 보편성, 선택성, 전문화의 체계적 변화와 연결합니다.



### Skill-RM: Unifying Heterogeneous Evaluation Criteria via Agent Sk (https://arxiv.org/abs/2606.03980)
- **What's New**: 이번 논문에서는 보상 모델(Reward Model, RM) 접근 방식을 통합하여 Skill Reward Model (Skill-RM)을 제안합니다. Skill-RM은 다양한 유형의 증거를 통합하는 통일된 메커니즘을 제공함으로써, 서로 다른 자원에서 수집된 정보를 동적으로 선택하고 집계하는 방식으로 보상 모델링을 재구성합니다. 이를 통해 기존의 정적 평가 방식에서 벗어나 일관성 있는 평가를 가능하게 하며, LLM의 사후 훈련에 있어 새로운 기준을 설정합니다.

- **Technical Details**: Skill-RM은 재사용 가능한 보상 평가 스킬로 보상 모델링을 실행하는 통일된 프레임워크로 정의됩니다. 이 모델은 주어진 평가 인스턴스에 대해 적용 가능한 기준을 식별하고, 관련 자원을 선택적으로 호출하여, 마지막 보상을 위해 기준 수준의 증거를 집계하는 체계적인 작업 흐름을 따릅니다. 보상 계산은 이제 단순한 평가 지식의 수동적 흡수가 아니라, 자원의 선택과 실행을 능동적으로 조정하는 과정을 통해 이루어집니다.

- **Performance Highlights**: 광범위한 실험 결과, Skill-RM은 전통적인 평가 기준들에 비해 일관되게 우수한 성능을 보였습니다. 특히, best-of-N 선택과 강화 학습( RL )의 다운스트림 응용 프로그램에서도 Skill-RM의 성능이 두드러지며, 이러한 성과는 명확한 자원 조정을 통한 혜택이라는 점에서 더욱 뜻깊습니다. 실험 결과는 Skill-RM이 단순히 도구의 가용성을 증가시키는 것이 아닌, 실행 가능한 절차로 평가를 구조화함으로써 보상 품질을 획기적으로 높인다는 것을 입증합니다.



### Value-Aware Stochastic KV Cache Eviction for Reasoning Models (https://arxiv.org/abs/2606.03928)
Comments:
          Codes: this https URL

- **What's New**: 이 논문은 Reasoning Models에서 KV 캐시 (KV cache) 제거 (eviction) 방법의 정확도를 향상시키기 위한 두 가지 주요 발견을 제시합니다. 첫째, 일부 값 상태(value states)는 비정상적으로 큰 크기를 가지며 이들을 제거할 경우 모델이 반복적인 추론 루프에 빠질 수 있습니다. 둘째, 제거 과정에서 stochasticity를 도입하면 캐시 다양성이 증가하여 정확도가 향상됩니다. 이 연구에서 제안하는 Value-aware Stochastic KV Cache Eviction (VaSE)은 이러한 발견을 기반으로 하여 효율성과 정확성 간의 격차를 해소합니다.

- **Technical Details**: VaSE는 훈련이 필요 없는 KV 캐시 제거 방법으로, 큰 크기의 값 상태를 보호하고 다양한 제거 결정을 촉진합니다. 제거 과정에서 KV 캐시가 과도하게 채워질 때 중요도 점수와 샘플링을 통해 KV 쌍을 선택합니다. 연구에서 Qwen3 모델에 VaSE를 적용하여 수학, 코드 생성 및 과학 질문 응답 등 여섯 가지 작업에서 평가했습니다. 이 방법은 기존의 SOTA 방식보다 평균적으로 더 높은 정확도를 달성했습니다.

- **Performance Highlights**: VaSE를 적용한 Qwen3 모델은 동일한 희소성(sparsity)에서 평균적으로 SOTA 선택 방법보다 높은 정확도를 보였으며, 강력한 제거 방법보다도 4% 이상의 성능 향상을 기록했습니다. 또한, CurDKV와 결합하면 평균 정확도가 7.7% 및 9.2% 향상되었습니다. VaSE는 R-KV와 비교하여 높은 처리량(throughput)과 낮은 메모리 사용량을 달성하며, 토큰 예산에서 메모리 압축 비율이 이론적으로 일치하는 결과를 보였습니다.



### Visual Instruction Tuning Aligns Modalities through Abstraction (https://arxiv.org/abs/2606.03871)
- **What's New**: 이번 연구에서는 시각적 기능이 사전 훈련된 대형 언어 모델(LLM)에 어떻게 통합되는지를 탐구합니다. 우리는 다양한 비전-언어 아키텍처에서 시각적 특징이 LLM의 중간 의미 층에 직접적으로 포함됨을 보여줍니다. 이러한 방식은 초기 단일 모드 처리 층을 우회하여, 멀티모달 작업에서 중간층이 중요한 역할을 했음을 밝혀냈습니다.

- **Technical Details**: 연구에서 사용한 VLM 모델에는 LLaVA, OneVision, InternVL2, Cambrian과 같은 시각 이해 작업을 위한 아키텍처들이 포함되어 있습니다. 두 개의 주요 훈련 단계로 나뉘어 있으며, 첫 번째 단계는 시각적 기능을 LLM에 맞추는 커넥터 가중치를 학습합니다. 두 번째 단계에서는 LLM과 시각 인코더의 내부 표현을 멀티모달 이해를 지원하도록 재구성합니다.

- **Performance Highlights**: 중간층의 역할은 여러 벤치마크에서 확인되었으며, 해당 층을 튜닝할 경우 전반적인 성능 손실을 최소화하면서도 고성능을 유지하는 효과를 나타냈습니다. 또한, 우리의 실험 결과는 이러한 중간층이 LLM의 내부 추상화 엔진의 재목적화에 의해 구동되는 국소적 현상임을 제안합니다.



### Taiji: Pareto Optimal Policy Optimization with Semantics-IDs Trade-off for Industrial LLM-Enhanced Recommendation (https://arxiv.org/abs/2606.03866)
Comments:
          8 pages, 2 figures

- **What's New**: 이번 논문에서는 산업 추천 시스템을 위해 설계된 혁신적인 LLM-as-Enhancer 프레임워크인 Taiji를 소개합니다. Taiji는 기존의 LLM4Rec 패러다임이 직면하고 있는 SFT(세미 슈퍼비전 파인 튜닝)와 RL(강화 학습) 단계의 문제를 해결하는 방법을 제시합니다. 이를 통해 Taiji는 고유의 Domain-Specific Chain-of-Thought 데이터 생성을 위한 역설계된 추론과 개방형 거부 샘플링을 활용하여 추천 품질을 높입니다.

- **Technical Details**: Taiji는 데이터 구축, 추론 활성화, LLM-추천 협업 및 온라인 순위 지정을 포함한 네 개의 주요 모듈로 구성되어 있습니다. EUPR(Reverse-Engineered User Preference Reasoning)과 ORFT(Open-Ended Rejection Sampling Fine-Tuning)를 통합하여 추천 특화 CoT의 품질을 향상시키고, POPO(Pareto Optimal Policy Optimization)라는 방법을 통해 LLM의 의미적 보상과 추천 선호 보상의 균형을 동적으로 조정합니다.

- **Performance Highlights**: 다양한 오프라인 평가와 온라인 A/B 테스트를 통해 Taiji의 효과성을 검증하였습니다. Taiji는 Kuaishou의 광고 플랫폼에 2026년 5월부터 배포되어 매일 4억 명 이상의 사용자에게 서비스를 제공하며, 상업적 수익을 획기적으로 증가시켰습니다. A/B 테스트 결과, 광고주 가치(ADVV)가 2.83% 개선되고 전반적인 수익이 3.30% 증가하는 성과를 보였습니다.



### Dynamic Short Convolutions Improve Transformers (https://arxiv.org/abs/2606.03825)
- **What's New**: 본 논문은 Transformer 아키텍처를 개선하기 위해 동적 단기 합성곱(dynamic short convolutions)을 도입합니다. 기존의 정적 합성곱(static convolutions)과 달리, 동적 합성곱은 입력에 따라 필터를 조정하여 표현력을 향상시킵니다. 동적 합성곱을 키, 쿼리, 값 표현에 적용함으로써 어려운 연상기억(associative recall) 작업에서 성능을 개선하는 실험을 통해, 기존 Transformer보다 더 나은 결과를 도출하는 것이 입증되었습니다.

- **Technical Details**: 동적 합성곱은 시퀀스의 각 위치에 대해 입력 의존적인 필터를 사용하며, 이는 합성곱의 국소성(locality bias)을 유지하면서 표현력을 증가시킵니다. 필터 가중치가 시점에 따라 변화하며, 특정 쿼리 위치에서 직접 생성됩니다. 이를 통해 각 토큰이 지역 맥락으로부터 정보를 동적으로 선택할 수 있도록 합니다.

- **Performance Highlights**: 모델의 크기가 150M에서 2B 매개변수에 이르는 다양한 실험에서 동적 합성곱은 기존의 Transformer 및 정적 합성곱이 추가된 Transformer보다 일관되게 뛰어난 성능을 보였습니다. 동적 합성곱은 평균적으로 1.33배의 계산 이점을 제공하며, 모든 선형 계층에 적용할 경우 1.60배의 이점을 제공합니다. 이 연구는 동적 단기 합성곱이 Transformer 기반 언어 모델을 발전시키는 확장 가능하고 하드웨어 효율적인 기본 구성요소임을 제시합니다.



### SkillPyramid: A Hierarchical Skill Consolidation Framework for Self-Evolving Agents (https://arxiv.org/abs/2606.03692)
- **What's New**: 이번 논문에서는 SkillPyramid라는 새로운 스킬 통합 프레임워크를 제안하여, AI 에이전트들이 더 폭넓은 작업 일반화를 위해 기존 스킬 경험을 재사용할 수 있도록 합니다. 기존의 비효율적이고 중복된 스킬 구조를 해결하여, 에이전트가 경험을 재사용 가능한 자산으로 변환할 수 있도록 합니다. 이는 궁극적으로 에이전트의 성능 향상과 대규모 작업 처리에 기여합니다.

- **Technical Details**: SkillPyramid는 다계층 구조의 스킬 토폴로지를 기반으로 구성되어 있으며, 하위 레벨은 세부적인 재사용 가능한 원자 스킬을 담고, 상위 레벨에서는 반복되는 문제 해결 패턴을 추상화합니다. 이 프레임워크는 Relation Analyzer와 Relation Builder를 통해 스킬의 재사용 관계를 분석하고, 새로운 스킬을 통합하는 자가 진화 메커니즘을 구현합니다. 각 스킬은 구조화된 자연어 프로그램으로 정의되며, 이를 통해 스킬의 적용 조건 및 실행 절차 등이 명확히 규정됩니다.

- **Performance Highlights**: ALFWorld, WebShop, ScienceWorld를 통한 실험 결과, SkillPyramid는 평균 보상을 38.0% 향상시키고 실행 단계를 27.7% 줄이는 성과를 보였습니다. 이와 같은 성과는 여러 백본 모델에서 반복적으로 나타나며, SkillPyramid가 스킬 수집을 정적 자원 풀에서 동적 진화 시스템으로 전환함을 보여줍니다.



### Bridging Auxiliary Constraints to Resolve Instruction Following in Large Reasoning Models (https://arxiv.org/abs/2606.03624)
Comments:
          a pre-MIT Press publication version

- **What's New**: 이 논문에서는 대형 추론 모델(LRMs)이 여러 지침을 신뢰성 있게 따르지 못하는 문제를 제기하고, 이를 제약 준수 문제(Constraint Adherence Problem, CAP)로 정의합니다. 새로운 구조적 접근 방식을 통해 각 명령을 제약의 지식 그래프로 변환하여 문제를 해결하려고 합니다. 이 접근법인 제약 관계 그래프 완성(Constraint Relationship Graph Completion, CRGC)은 제약 간의 관계를 명확히 모델링하고, 교량 제약을 발견하여 모델이 요구 사항을 더 잘 맞추도록 지원합니다.

- **Technical Details**: CRGC는 세 가지 주요 구성 요소로 나뉘어 있습니다: (1) 제약 그래프 구성 - 분해된 명령 제약 간의 관계 매핑, (2) 준수 도전 감지 - 간과되거나 충돌하는 제약 식별, (3) 교량 제약 발견 - 문제 있는 제약 간의 연결을 위한 보조 지침 도입. 이러한 그래프 기반 표현은 LRM이 제약이 명확하지 않거나 상충된다고 인식할 때 가장 큰 어려움을 겪는다는 것을 보여줍니다.

- **Performance Highlights**: CRGC는 39%의 제약 위반을 줄이며, 크기향상한 추론 품질을 유지했습니다. 기존 방법들과는 달리, 이 방법은 모델 파라미터를 수정하지 않고도 제약 준수를 향상시키며, 필요한 경우에만 교량 제약을 적응적으로 결정합니다. 뿐만 아니라, CRGC는 다양한 데이터셋에서 우수한 제약 준수를 보여주며, 추론 능력의 저하 없이 성능을 유지하는 장점이 있습니다.



### World Models Meet Language Models: On the Complementarity of Concrete and Abstract Reasoning (https://arxiv.org/abs/2606.03603)
- **What's New**: 본 논문은 고정된 시각적 관찰로부터 미래 결과를 예측하는 데 필요한 세계 모델(world models)과 다중 모달 대형 언어 모델(multimodal large language models, MLLMs)의 상호 작용을 탐구합니다. 이러한 모델들은 과거의 정황을 기반으로 미래를 예측하나, 생성된 결과물이 완벽하지 않아 최종 결론에 영향을 미치는 방식을 명확히 해야 할 필요가 있습니다. 이 문제를 해결하기 위해, 저자들은 Controlled Concrete Reasoning이라는 새로운 접근법을 제시하고, Privileged-Future On-Policy Self-Distillation(PF-OPSD)이라는 훈련 프레임워크를 도입하였습니다.

- **Technical Details**: Controlled Concrete Reasoning은 초기 관찰 및 미래 지향 질문이 주어졌을 때 MLLM이 언제 세계 모델을 호출해야 하는지, 그 결과를 어떻게 검증하고 얼마나 신뢰해야 하는지를 학습하는 과정을 포함합니다. 이를 평가하기 위해 VRQABench와 OpenWorldQA라는 두 개의 인간 검증 벤치마크를 제작하였으며, 각 벤치마크는 복잡한 공간적 맥락의 예측 및 개방형 신체 예측을 테스트합니다. PF-OPSD는 훈련 중에 진실 미래 비디오와 정답을 이용하여 학생의 경로를 평가합니다.

- **Performance Highlights**: PF-OPSD는 VRQABench와 OpenWorldQA 각각에서 기준선 모델 대비 10.6% 및 10.9% 성능 향상을 보여주었으며, 생성된 롤아웃의 잡음 또는 상충에 대한 강건성을 높였습니다. 이는 미래 결과 예측을 보다 신뢰성 있게 만들어 주며, 실제로 필요한 시뮬레이션 사용의 결정적인 기준을 마련합니다. 논문은 코드와 데이터셋을 공개하여 연구자들이 해당 연구를 바탕으로 추가적인 발전을 이룰 수 있도록 지원합니다.



### CauTion: Knowing When to Trust LLMs for Ensemble Causal Discovery (https://arxiv.org/abs/2606.03602)
- **What's New**: 이번 논문에서는 기존의 방법론들이 갖고 있는 한계를 극복하기 위해 'CauTion'이라는 프레임워크를 제안합니다. 이 프레임워크는 LLM(large language model)의 도메인 지식을 여러 통계적 인과 발견 알고리즘의 집합에 신뢰성 평가를 통해 통합하는 방식으로 작동합니다. 또한, 알고리즘 간의 동의에 기반하여 LLM의 지식을 언제 활용할지를 판단하는 의사 결정 프로세스를 포함합니다.

- **Technical Details**: CauTion의 작동 과정은 세 단계로 나뉘어 있습니다. 첫 번째 단계에서는 알고리즘 집합이 여러 인과 발견 알고리즘의 출력을 집계하고, 모든 알고리즘이 동의하는 변수 쌍을 해결합니다. 두 번째 단계에서는 신뢰 보정(arbitration) 메커니즘을 통해 LLM과 알고리즘의 상대적 신뢰성을 추정하고, 신뢰도가 부족한 경우에만 LLM의 판단을 따르도록 합니다. 마지막으로, 사이클 수정을 수행하여 최종 인과 그래프가 비순환적(DAG) 구조를 준수하도록 보장합니다.

- **Performance Highlights**: 실험 결과, CauTion은 데이터 중심의 방법론과 LLM을 보강한 베이스라인보다 일관되게 우수한 성능을 나타냈습니다. 특히, 가장 큰 데이터셋인 Win95pts(n=76)에서 CauTion은 구조적 해밍 거리(SHD) 27을 기록하며 두 번째로 우수한 LLM 보강 방법보다 두 배 이상 개선된 성능을 보였습니다. 또한 CauTion은 다양한 LLM에 대해 강력한 견고성을 유지하며 성능을 발휘했습니다.



### SAGE: A Quantitative Evaluation of Socialized Evolution in Agent Ecosystems (https://arxiv.org/abs/2606.03544)
Comments:
          13 pages, 5 figures

- **What's New**: 이번 연구는 SAGE (Social Agent Group Evolution)라는 평가 프레임워크를 소개하며, 자가 개선 에이전트를 평가하는 새로운 방법론을 제안합니다. 에이전트는 서로의 역사에 접근할 수 있는 SocialEvo 조건과 자신의 과거 기록만을 볼 수 있는 SelfEvo 조건으로 나뉘어 연구됩니다. 이 연구는 공동 경험이 자가 개선만으로는 달성할 수 없는 향상을 측정하는 것이 핵심입니다.

- **Technical Details**: SAGE 프레임워크는 다양한 에이전트 모델들이 공동으로 진화하는 SocialEvo와 개별적으로 진화하는 SelfEvo 조건을 비교합니다. 연구에서는 오픈-엔디드 머신러닝 (MLR-Bench), 경제 계획 (DrugWars), 그리고 멀티플레이어 전략 게임 (Splendor)이라는 세 가지에서 실험을 진행하였습니다. 각 조건에서 에이전트들은 다양한 형태의 과거 기록(예: raw trajectories, reflective summaries)을 통해 학습하며, 이러한 기록의 표현 방식이 학습 행동에 미치는 영향을 분석합니다.

- **Performance Highlights**: 연구 결과, 강력한 자가 개선 에이전트조차도 다른 에이전트의 경험이 없이는 한계에 봉착하는 반면, 피어 경험을 통해 상당한 진전을 이룰 수 있음을 발견했습니다. 특히 경쟁 환경에서는 에이전트가 일반적으로 개선되는 반면, 특정 상대에 맞춘 전략은 개발하지 않는 것으로 나타났습니다. 다양한 형태의 피어 역사에서 필터링된 기록과 요약된 정보가 원시 기록보다 더 나은 성과를 보였으며, 이는 사회적 이익이 단순한 노출의 양에 의존하기보다는 추상화의 질에 의존함을 나타냅니다.



### Can LLM Rerankers Predict Their Own Ranking Performance? (https://arxiv.org/abs/2606.03535)
- **What's New**: 이번 논문은 LLM(대형 언어 모델) 리랭커가 스스로 생성한 순위의 품질을 추정할 수 있는지에 대한 연구를 진행합니다. 기존의 쿼리 성능 예측(QPP) 방법들이 주로 검색 후 외부 예측자에 의존했던 반면, 이 연구는 리랭커 내부의 QPP를 중점적으로 분석합니다. 이를 통해 LLM 리랭커가 직접 생성한 순위의 품질을 평가하는 신호가 있을 수 있음을 강조합니다.

- **Technical Details**: 연구는 훈련 없이도 측정하는 기법과 훈련 기반 방법을 다룹니다. 훈련 없는 추정 기법으로는 샘플링된 순위에서 메트릭별 일관성과 리랭커가 직접 생성한 수치화된 신뢰도를 확인합니다. 그리고 감독 훈련을 통해 리랭커가 더 정확한 신뢰도를 표현할 수 있는지 탐구합니다.

- **Performance Highlights**: 실험 결과, LLaMA3.1 및 Qwen2.5 모델을 사용한 결과, 일관성 기반 접근법이 QPP-Gen과 경쟁력 있는 성능을 보였으며, 거의 모든 설정에서 더 나은 캘리브레이션을 나타냈습니다. 반면에 한정된 훈련 데이터를 통해 제안된 두 가지 감독 방법인 Verb-Num과 Verb-List는 리랭커가 저렴한 비용으로 캘리브레이션된 순위 품질 추정치를 생성하도록 도와줍니다.



### DMF: A Deterministic Memory Framework for Conversational AI Agents (https://arxiv.org/abs/2606.03463)
Comments:
          21 pages, 3 figures

- **What's New**: 본 논문은 대화형 AI 에이전트가 필요로 하는 메모리 시스템에 대한 새로운 접근법인 결정을 촉진하는 메모리 프레임워크(Deterministic Memory Framework, DMF)를 소개합니다. DMF는 기존의 LLM 기반 요약 대신 고전적인 NLP 분석 및 수학적 점수를 기반으로 하는 결정론적인 파이프라인을 사용하여 메모리 관리에서의 여러 문제점을 해결합니다. 이 시스템은 대화 상호 작용에 대해 결정적 콘텐츠 신호와 구조화된 기원을 결합해 생존 점수(Survival Score) Ω를 계산하여, 상호작용 횟수 감소 법칙에 따라 연관성을 진화시키는 과정을 설명합니다.

- **Technical Details**: DMF는 CPU 우선 접근 방식을 채택하여 메모리 관리 루프에서 LLM 호출을 전혀 사용하지 않습니다. 이 프레임워크는 대화 상호작용에서 수치적 콘텐츠 신호 및 구조적 대화 단서를 추출하고, 정적 생존 점수 Ω를 계산하며, 상호작용 카운트 감소를 적용합니다. 메모리 관리에서 모든 결정은 결정적 규칙을 통해 이루어지며, 메모리 업데이트와 유사한 신뢰할 수 있는 정보 유지 구조를 제공합니다.

- **Performance Highlights**: DMF는 메모리 맥락을 준비하는 동안 제로 토큰을 사용하면서, 전체 대화에 대해 5배에서 242배 적은 토큰을 사용하여 비교 가능한 정확도를 달성했습니다. 이 결과는 LLM 호출을 메모리 관리 루프에서 제거하여 비용을 거의 제로에 가깝게 줄일 수 있음을 보여줍니다. DMF는 대화형 AI 에이전트를 위한 결정론적 메모리 시스템을 가능하게 합니다.



### When Model Merging Breaks Routing: Training-Free Calibration for MoE (https://arxiv.org/abs/2606.03391)
- **What's New**: 이번 논문에서는 Mixture-of-Experts (MoE) 아키텍처에서 모델 병합이 초래하는 라우팅 파손(routing breakdown) 문제를 다룹니다. 기존의 병합 방법들이 MoE 구조의 비선형 특성 때문에 기능을 발휘하지 못하는 점을 지적하고, 이로 인해 전문가의 지정이 잘못될 수 있음을 강조합니다. 이를 해결하기 위해, 이 연구에서는 Hessian-Aware Router Calibration (HARC)이라는 새로운 무훈련(training-free) 접근법을 제안합니다.

- **Technical Details**: HARC는 비선형 소프트맥스 및 이산 Top-k 라우팅 메커니즘의 민감도를 고려하여 병합된 라우터의 출력 분포를 조정하는 방식으로 설계되었습니다. 이 방법은 매트릭스-프리(matrix-free) 공액 경량(conjugate gradient) 방법을 사용하여 효율적으로 해결할 수 있는 닫힌 형태의 솔루션을 허용합니다. HARC는 라우터 성능의 일관성을 보장하기 위해 2차 헤시안(Hessian) 정보를 활용하여 전문화된 지식을 제대로 전달할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, HARC는 수학 및 코드 생성을 다루는 작업에서 기존의 모델 병합 방법보다 일관되게 성능을 향상시키는 것으로 나타났습니다. 특히, 더 깊은 레이어로 갈수록 경로 오류가 불균형하게 증가하며, HARC는 이러한 경향을 완화하여 원본 라우팅 동작과의 정렬을 유지합니다. 또한, HARC는 약 40%의 캘리브레이션 샘플을 사용하여 최적 성능에 수렴할 수 있음을 보여줍니다.



### P\textsuperscript{2}-DPO: Grounding Hallucination in Perceptual Processing via Calibration Direct Preference Optimization (https://arxiv.org/abs/2606.03376)
- **What's New**: 이번 연구에서는 Large Vision-Language Models (LVLMs)에서 발생하는 환각(hallucination) 문제를 해결하기 위한 새로운 접근법인 Perceptual Processing Direct Preference Optimization (P²-DPO)를 제안합니다. P²-DPO는 모델이 자체적으로 preference data를 생성하고 학습할 수 있도록 하여 시각적 병목 현상(perceptual bottleneck) 문제를 직접적으로 해결합니다. 우리의 방법은 시각적 신호와 텍스트 간의 인과적 생성 관리를 강화하기 위한 Calibration Loss를 포함합니다.

- **Technical Details**: P²-DPO는 두 가지 새로운 preference pair를 도입하여, 각기 다른 시각적 결함을 해결합니다. 첫 번째는 Focus-and-Enhance Preference Pair로, 이는 세밀한 세부 사항의 향상된 출력과 열화된 출력 간의 대조를 통해 Perceptual Bottleneck을 극복합니다. 두 번째는 Visual Robustness Preference Pair로, 정확한 정보와 노이즈 신호 간의 출력을 대비하여 시각적 강인성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, P²-DPO는 같은 양의 학습 데이터와 비용을 사용했음에도 불구하고, 강력한 기준선을 초과하는 성능을 보여줍니다. 특히 Attention Region Fidelity (ARF) 평가에서는 높은 정확도를 기록하였으며, 이미지 열화 시나리오에서도 시각적 강인성을 크게 개선함을 입증하였습니다.



### Speech Emotion Recognition using Attention-based LSTM-Network with Residual Connection (https://arxiv.org/abs/2606.03359)
Comments:
          6 pages, 5 figures, DSPA 2026

- **What's New**: 이번 연구에서는 경량화된 LSTM 기반 아키텍처인 ResLSTM-SA를 제안합니다. 이 모델은 residual connections와 soft attention 메커니즘을 통합하였으며, 기존의 advanced 모델에 비해 적은 파라미터로도 높은 성능을 발휘합니다. 실험 결과, ResLSTM-SA는 CNN 및 hybrid CNN-LSTM 아키텍처보다 뛰어난 성능을 보여주었습니다.

- **Technical Details**: ResLSTM-SA는 mel-frequency cepstral coefficients (MFCCs)와 chromagram을 활용하여 음성 감정 인식을 위한 특징을 추출합니다. 34차원 MFCC 벡터와 12차원 chroma 벡터를 결합하여 46차원 입력을 형성하였고, 이 입력은 LSTM 네트워크에서 처리됩니다. 주목할 점은 LSTM의 첫 번째 층의 은닉 상태 차원이 입력 특성과 동일하여, 입력 시퀀스에 대한 컨텍스트 정보가 보존됩니다.

- **Performance Highlights**: 최고 성능 변형인 ResLSTM-SA-h64는 46.8k의 훈련 가능 파라미터로 최대 0.6517의 unweighted average recall (UAR)을 달성하였습니다. 이는 대규모 self-supervised 모델 대비 세 배 적은 파라미터 수를 통해 실시간 음성 비서 및 엣지 디바이스에서 효율적인 배포를 가능하게 합니다.



### Beyond Semantics: Modeling Factual and Affective Perceptual Experiences from Vision-Language Data (https://arxiv.org/abs/2606.03345)
Comments:
          8 pages

- **What's New**: 이 논문에서는 감정적으로 그리고 문화적으로 이미지를 인식하는 방식을 이해하기 위한 새로운 문제인 P-Topics(Perception Topics) 모델링을 제안합니다. 이 연구의 주요 목표는 이미지와 캡션의 데이터셋에서 다양한 인식 경험을 발견하고 모델링하는 것이며, 각 경험은 객관적인 사실적 측면과 주관적인 정서적 측면으로 정의됩니다. 이를 위해 PercepT(Perception topic Transformer)라는 새로운 아키텍처를 소개하며, 이는 이미지와 인식 경험을 연관시키기 위한 두 단계의 구조를 가지고 있습니다.

- **Technical Details**: PercepT는 두 단계로 구성된 아키텍처로, 첫 번째 단계에서는 비지도 학습 목표를 사용하여 이미지-캡션 쌍의 P-Topics를 발견합니다. 두 번째 단계에서는 주의 풀링(attention pooling)을 활용하여 이미지와 관련된 P-Topic 매핑 기법을 학습합니다. CLIP 기반 인코더는 객관적인 사실 정보를 나타내고, 감정 인코더는 주관적인 정서 정보를 캡처하여 이미지의 장르와 감정을 프록시 라벨로 사용합니다.

- **Performance Highlights**: PercepT는 ArtELingo 데이터셋에서 0.97의 실루엣 점수를 달성하여, 가장 가까운 기준선인 0.37에 비해 더 나은 인식 클러스터를 반영합니다. 또한, PercepT는 0.94의 AUC 점수를 달성하여 0.77의 기준선에 비해 더 나은 매핑 성능을 보여줍니다. 실험 결과 사람 평가에서도 PercepT가 58.4%의 선호도를 나타내며 기존 방법들보다 우수한 성능을 입증합니다.



### CAPER: Clause-Aligned Process Supervision for Text-to-SQL (https://arxiv.org/abs/2606.03327)
- **What's New**: 이번 논문에서는 Text-to-SQL 시스템의 평가 방식에 대해 새로운 접근법을 제안합니다. 기존의 쿼리 수준 실행 정확도 평가에서는 중간 SQL 결정의 성공이나 실패 원인을 규명하기 어려웠습니다. 이를 해결하기 위해 연구팀은 CAPER라는 프레임워크를 개발하여 SQL 추상 구문 트리의 반사적 개입(counterfactual intervention)을 통해 오류를 지역화하는 방법을 제안했습니다.

- **Technical Details**: CAPER는 중간 SQL 결정을 포함한 CLAUSE-PRM(Clause-level Process Reward Model)을 생성하는 자동화된 시스템으로, SQL 오류를 구분할 수 있는 클라우스 수준의 선호 감독을 제공합니다. 이 연구는 SQL 추상 구문 트리에 대한 반사적 개입을 사용해 오류의 근본 원인을 파악하고 수집된 데이터를 기반으로 CAPER-9B 모델을 훈련시킵니다. 이 모델은 강화 학습(rl) 정책 최적화에서 중간 SQL 결정에 대한 세밀한 보상을 제공합니다.

- **Performance Highlights**: BIRD와 Spider 데이터셋에서의 실험 결과, CAPER-9B는 GPT-5.4에 비해 최대 15.3% 상대 EX 개선을 달성합니다. 또한 CAPER-9B는 고립된 실패에 대해 84.53%의 정확도와 90.60%의 MRR(Mean Reciprocal Rank)을 기록하며 실패 지역화 성능이 강화되었습니다. 이 연구는 90,000개 이상의 클라우스 주석 선호 튜플을 생성하여, 효율적인 데이터 사용을 입증합니다.



### VistaHop: Benchmarking Multi-hop Visual Reasoning for Visual DeepSearch (https://arxiv.org/abs/2606.03273)
- **What's New**: 이 연구에서는 Visual DeepSearch를 위한 새로운 벤치마크인 VistaHop을 소개합니다. VistaHop은 복잡한 비주얼 쿼리에 답하기 위해 이미지 영역을 반복적으로 검사하고, 시각적 증거를 기반으로 중간 추론을 연결하게 하는 멀티모달 대형 추론 모델(MLRM) 에이전트를 평가합니다. 기존 벤치마크가 단일 단계 시각 이해에 주로 초점을 맞추고 있다는 점에서 차별화됩니다.

- **Technical Details**: VistaHop은 300개의 고해상도 이미지, 25개의 시각 검색 시나리오, 그리고 시각적 앵커와 관련된 증거 체인을 추적하는 350개의 멀티 홉 품질 보증(QA) 작업을 포함합니다. 이를 통해 모델들이 여러 이미지 기반 추론 경로에서 정보를 융합하고, 이미지와 외부 지식을 연결할 수 있는 능력을 평가합니다. 또한, VistaArena라는 통합 평가 환경을 개발하여 텍스트 검색, 이미지 검색, 이미지 잘라내기 및 증거 기반 답변 검증을 지원합니다.

- **Performance Highlights**: 일곱 개의 대표적인 MLRM에 대한 실험 결과 현재 모델들은 VistaHop 작업을 성공적으로 해결하는 데 한계가 있음을 보여주었습니다. 가장 우수한 성과를 기록한 SenseNova-MARS-32B 모델조차도 24.31%라는 낮은 Pass@1 비율을 보였습니다. 이는 현재의 모델들이 시각적 고정을 통한 증거 재검토, 장기적 추론 및 멀티 앵커 정보 융합에서 여전히 한계를 가지고 있음을 드러냅니다.



### Solipsistic Superintelligence is Unlikely to be Cooperativ (https://arxiv.org/abs/2606.03237)
Comments:
          24 pages, 1 figure, Accepted at Proceedings of the 43rd International Conference on Machine Learning, 2026

- **What's New**: AI의 중심 과제는 능력에서 공존( co-existence)으로 변화하고 있습니다. 이 논문은 AI 시스템의 배치가 엔도제닉( endogenous) 비정상성을 유도하며, 이는 역사적 분포가 배치 맥락과 서로 다를 때 발생하는 문제를 다루고 있습니다. 저자들은 AI가 협력(cooperation)에 참여해야 하며, 이는 다수의 행위자 간의 상호의존성을 탐색하는 평형 선택(equilibrium-selection) 과정을 요구한다고 주장합니다.

- **Technical Details**: AI 연구에서 지배적인 패러다임은 환경이 외적(exogenous)이고 정적인(stationary) 피드백 원천으로 간주하며, 다른 행위자는 예측 가능한 존재로 보기 때문에 자아중심적(solipsistic) 접근 방식에 기초하고 있습니다. 이러한 가정은 AI 시스템이 다른 적응형 에이전트와 배치될 때 발생하는 딜레마를 간과하게 만듭니다. 저자들은 해결과제를 일련의 고정된 평가 세트에 대한 성과 최적화(output)에 제한하기보다는, AI의 배치에 따른 상호작용을 고려해야 한다고 강조합니다.

- **Performance Highlights**: AI의 발전을 두고 기존 플랫폼이 여전히 효과적일 수 있지만 AI의 사회_기술적(socio-technical) 환경에서 고급 AI의 배치가 반응 동역학에 크게 노출될 것임을 강조합니다. 협력의 필요성을 주장하며, 집단적 동역학이 해를 끼칠 수 있는 능력 있는 시스템이 반드시 긍정적인 결과만을 도출하지 않는다고 경고합니다. 논문은 AI의 설계가 단순한 성과 최적화가 아닌 협력적 동역학을 반영한 구조적 특성으로 재구성되어야 한다고 주장합니다.



### GLINT: Sparsely Gated Vision-Language Alignment for Fine-Grained Radiology Representations (https://arxiv.org/abs/2606.03180)
- **What's New**: 이번 연구에서는 GLINT (Gated Language-Image alignmeNT)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 전통적인 VLMs (Vision-Language Models) 방식의 한계를 극복하여, 이미지의 특정 패치(patch)에 대해 텍스트 쿼리와의 밀접한 관계를 학습합니다. 특히, GLINT는 3D CT(volumes)에서도 마스크 감독 없이 제로샷(zero-shot) 세분화(segmentation)를 성공적으로 수행한 첫 사례입니다.

- **Technical Details**: GLINT는 Sparse Gated Alignment라는 독특한 아키텍처를 통해 텍스트 쿼리와 관련된 패치만을 활성화하는 시그모이드 게이트(sigmoid gate)를 사용합니다. 또한, Dense Feature Regularization을 적용해 훈련 가능한 인코더의 중간 특징(feature)을 고정된 자가 감독 학습(SSL) 모델에 연결함으로써, 게이트가 의존하는 세부 패치 특징을 보존합니다. GLINT는 이를 통해 2D 흉부 X선(CXR) 및 3D 흉부 CT에 모두 적용됩니다.

- **Performance Highlights**: GLINT는 제로샷 클래스 분류(zero-shot classification), 위치 지정(grounding), 세분화(segmentation)에서 우수한 성능을 보이며, 특히 SS(SSL) 인코더 및 기존의 의료 VLMs에 비해 뛰어난 결과를 도출했습니다. 특히, 제로샷 세분화와 위치 지정에서 가장 두드러진 성과를 보여, 쿼리 특정(query-specific) 로컬라이제이션이 강조된 설계 의도를 잘 반영하고 있습니다.



### FederatedSkill: Federated Learning for Agentic Skill Evolution (https://arxiv.org/abs/2606.03143)
- **What's New**: 최근 LLM(대형 언어 모델) 에이전트들은 복잡한 작업을 수행하기 위해 스킬 라이브러리를 점점 더 의존하고 있습니다. 본 논문에서는 사용자의 개인 정보 보호를 보장하며, 협력적인 에이전트 진화를 가능하게 하는 FederatedSkill이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 사용자 간의 협업을 통해 스킬 진화를 이루이면서도 보호를 유지하는 방법을 제시합니다.

- **Technical Details**: FederatedSkill은 클라이언트의 로컬 경로(trajectory)를 업로드하지 않고, 각 클라이언트가 자신의 경로를 기반으로 로컬에서 스킬 패치를 발전시킬 수 있도록 합니다. 여기서 스킬 패치를 활용하여 수정, 추가 또는 삭제 작업을 문서화하며, 이는 연합 학습(federated learning) 개념에 유사합니다. 서버는 이러한 패치를 집계하여 각 클라이언트에 맞춤화된 능력 경계를 모형화하고, 개인화된 스킬 진화를 지원합니다.

- **Performance Highlights**: FederatedSkill은 20개의 다양한 에이전트 작업 가족을 대상으로 한 평가에서 자기 진화( self-evolution) 방법에 비해 최대 44.4%의 성공률 증가와 37.5%의 계산 비용 감소를 보여주었습니다. 또한, 클라이언트 맞춤형 스킬 라이브러리를 유지함으로써 평균 성능 gain을 12.2% 달성하여 개인화의 필요성을 강조합니다.



### PsychoPass: Geometric Profiling of Multi-Turn Adversarial LLM Conversations (https://arxiv.org/abs/2606.03136)
- **What's New**: 본 논문은 대규모 언어 모델(LLMs)에 대한 다중 턴 Jailbreak 공격에서 발생하는 문제를 다루고 있습니다. 현재의 방어 시스템은 개별 턴에서 작동하지만, 공격은 대화의 여러 턴을 통해 전개됩니다. 이 연구는 대화의 기하학적 특성을 분석하여 공격 의도가 초기 기하학에서 인코딩될 수 있는지를 살펴봅니다. PsychoPass라는 새로운 프레임워크를 도입하여 대화 궤적의 기하학적 특성을 추출하고, 해로운 콘텐츠가 생성되기 전에 공격을 예측하는 방법을 제안합니다.

- **Technical Details**: PsychoPass 파이프라인은 총 세 가지 단계로 이루어져 있습니다: 공격 생성, 궤적 구성 및 기하학적 프로파일링. 공격 생성 단계에서는 PyRIT를 활용하여 Crescendo 공격을 수행하며, 최대 8턴의 대화를 생성합니다. 궤적 구성 단계에서는 각 대화를 dd차원 공간에서 궤적으로 임베딩하고, 기하학적 프로파일링 단계에서는 이 궤적에서 통계적 특성을 추출합니다. 두 가지 인코더(TF-IDF와 Dense embedding 모델)를 사용하여 대화를 벡터 공간으로 변환하게 됩니다.

- **Performance Highlights**: 실험 결과, 자극적 궤적 분류에서 근사 완벽의 AUROC 성능을 달성했지만, 이는 대화 길이에 의존하였습니다. 길이의 혼돈을 제거한 후에도 일정한 기하학적 신호가 남아 공격 결과의 예측에 기여하였습니다. 특히, 공격 결과는 2턴의 짧은 접두사에서도 우연 확률을 초과하는 성과를 보이며, 기존의 내용 기반 안전 분류기는 그러지 못했습니다. 이 결과는 적대적인 대화가 온라인 모니터링에 적합한 기하학적 지문을 남긴다는 점을 강조합니다.



### Decoupled Smart Contract Audits: Lightweight LLM Framework via Distillation and Aggregation (https://arxiv.org/abs/2606.03128)
Comments:
          12 pages, 4 figures, 5 tables. Accepted to IEEE ICWS 2026

- **What's New**: 이번 연구에서는 스마트 계약 보안을 위한 효율적인 엔드 투 엔드 감사 프레임워크를 소개합니다. 이 프레임워크는 경량의 최적화된 오픈 소스 LLM (0.6B-4B 매개변수)을 활용하여 구성됩니다. 우리의 접근 방식은 감사 작업을 취약점 감지, 설명, 심각도 분류 및 수정 권장 사항의 네 가지 상호 연결된 구성 요소로 분리하는 점에서 혁신적입니다.

- **Technical Details**: 전통적인 단일 모형 대신, 우리의 프레임워크는 각 구성 요소가 별도의 전문 모듈로 기능하도록 분리되어 작업을 순차적으로 수행합니다. 각 작업을 위해 Rank-Stabilized Low-Rank Adapters (rsLoRA) 및 지식 증류(Knowledge Distillation)를 이용하여 감지 성능을 유지하고 있습니다. CoVe (Chain-of-Verification) 집계 방법을 사용하여 모델로부터 생성된 여러 초안 응답을 필터링 및 검증하여 단일 고정밀도를 출력하도록 합니다.

- **Performance Highlights**: 실험 결과, 우리의 경량 파이프라인은 최신의 대형 LLM보다 연속적으로 우수한 성능을 나타내며, 98.25%의 취약점 감지 정확도를 달성했습니다. 또한 심각도 분류에서 상당한 편향을 발견하여 향후 연구에 중요한 기준을 수립하는 데 기여할 것으로 보입니다. 이 연구는 독립적인 Web3 개발자들이 사용할 수 있는 효과적인 감사 도구를 제공하여 스마트 계약 보안을 민주화하는 데 기여할 것입니다.



### Multi-component Causal Tracing in Large Language Models (https://arxiv.org/abs/2606.03085)
Comments:
          Accepted to ACL 2026 main conference

- **What's New**: 이 논문에서는 기존의 단일 구성 요소 연구를 넘어 다수의 구성 요소를 동시에 추적하는 통합 프레임워크를 제시합니다. 이 프레임워크는 특정 성능 메트릭(예: 정확도 및 공정성)에 가장 중요한 하위 구성 요소 집합을 체계적으로 식별합니다. 이론적 기초 위에 유연한 개입을 통해 다양한 메트릭을 적용하여 모델의 행동을 정량화하는 기법들을 소개합니다.

- **Technical Details**: 우리의 연구에서는 연속적인 방식으로 조합적 검색 문제를 해결하는 효율적인 알고리즘을 설계했습니다. 이는 부드러운 개입(soft interventions) 및 신중하게 설계된 메트릭 변환을 활용하여 이루어지며, 이를 통해 조합적 공간을 효율적으로 탐색할 수 있게 됩니다. 이 과정에서 이 알고리즘은 적절한 제약 조건 하에 수행되어 여분의 이진 결정(binary decisions)을 생성할 수 있게 됩니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 특정 메트릭에 대해 높은 영향을 미치는 모델의 구성 요소 하위 집합을 효율적으로 식별함을 보여줍니다. 기존의 기준 방법들에 비해, 우리의 방법은 GPT-2 모델에서 1.76배의 속도 향상을 달성하였고, 계산 비용 측면에서도 유리한 결과를 보였습니다. 이를 통해 다수의 구성 요소 간의 상호작용을 정밀하게 추적하는 접근 방식의 효과성을 입증하고 있습니다.



### ZX-Calculus:Trace-Indexed Dependent Types and Epistemic Semantics (https://arxiv.org/abs/2606.03063)
- **What's New**: 이번 논문에서는 ZX-Calculus (Knowledge Evolution Calculus)라는 보수적 확장을 제안합니다. 이는 Martin-Lof Dependent Type Theory (MLTT)와 통합된 trace-indexed types, presheaf 비단조적 의미론 및 구성적 AGM belief revision을 포함합니다. 논문은 34개의 완전한 증명과 함께 Coq 메커니즘을 동반하며, 중심적인 두 결과에 대한 거부 없이 모든 핵심 결과들을 Coq에서 검증했습니다.

- **Technical Details**: ZX-Calculus는 역사적 트레이스를 종속형 이론 내에서 구성적 객체로 내장하여 시스템의 현재 상태에 도달하는 방법을 타입 시스템의 고유 구성 요소로 만듭니다. 이 시스템은 유형된 실행 추적의 귀추적(finite trace) 및 비단조 지식의 발전을 처리하기 위해 구성되고, AGM 이론의 검증된 구성이 포함됩니다. 핵심적인 문제는 트레이스가 확장될 때 신념 상태가 일관성을 유지하는지를 평가하는 것입니다.

- **Performance Highlights**: 이 논문은 또한 Single-Step Revision Systems (SSRS)를 소개하고, B^AGM이 유효한 SSRS임을 증명했습니다. 이러한 시스템은 경로 의존적 신념 수정과 파라미터 일관성 간의 근본적인 긴장을 드러내며, 이를 통해 논리적 관계를 통한 정전성 및 재생 가능성을 입증합니다. SSRS는 비단조적 진화 및 고유한 신념 수정 과정을 위한 원하는 통합 프레임워크를 제공합니다.



### Inducing Reasoning Primitives from Agent Traces (https://arxiv.org/abs/2606.02994)
Comments:
          22 pages including appendices

- **What's New**: 이 논문에서는 Reasoning Primitive Induction이라는 단일 통과 메소드를 소개하여, 성공적인 ReAct 추적을 통해 재사용 가능한 추론 동작을 집합화하고 변환합니다. 이러한 추론 동작은 LLM이 호출 시 해석하는 자연어 docstring에 의해 정의되며, 이를 통해 LLM 에이전트가 테스트 시간에 평가할 수 있는 compact library를 생성합니다. 연구 결과, 유도된 라이브러리가 그 추적을 생성한 에이전트보다 성능이 뛰어난 것으로 나타났습니다.

- **Technical Details**: 이 방법은 ReAct 롤아웃의 데이터셋을 수집하여 각 단계의 Thought 문자열을 클러스터링하고, 반복되는 추론 동작을 수식화하도록 설계되었습니다. 각 추론 동작은 이름, 형식화된 입력/출력 서명, 그리고 자연어 설명을 포함하는 튜플로 정의됩니다. 이 논문에서 제시된 primitive는 ReAct 행동 공간에 등록된 typed callable로 구현되며, LLM은 이 호출 시 docstring을 통해 연결된 내용을 해석합니다.

- **Performance Highlights**: 연구에서 유도된 라이브러리는 RuleArena NBA에서 +44pp, MuSR 팀 배정에서 +30pp, NatPlan 회의 계획에서 +22pp로 성능을 향상시키며, 다른 서브태스크에서도 전반적으로 리더를 초과했습니다. 전체적으로, 고정된 구성의 단일 설정이 zero-shot Chain-of-Thought보다 모든 서브태스크에서 개선, 전문가가 직접 작성한 구조와 비교하여 유사하거나 초과하는 성능을 발휘하였습니다.



### Multi-Segment Attention: Enabling Efficient KV-Cache Management for Faster Large Language Model Serving (https://arxiv.org/abs/2606.02964)
- **What's New**: 이번 연구는 대형 언어 모델(Large Language Models, LLMs)에서 핵심-값(key-value, KV) 캐시 관리 시스템에 대한 새로운 접근법인 AsymCache를 제안합니다. AsymCache는 GPU의 주의(attention) 커널 성능과 맞물린 캐시 거주 결정(cache residency decisions)을 명확히 정렬하여 계산 지연(computation latency)을 최적화합니다. 이 시스템은 다중 세그먼트 주의(Multi-Segment Attention), 계산 인식 캐시 삭제 정책(computational-aware block evictor), 적응형 청킹 스케줄러(adaptive chunking scheduler)로 구성되어 있습니다.

- **Technical Details**: AsymCache는 KV 캐시 블록의 거주 결정을 GPU의 성능에 맞춰 최적화합니다. Multi-Segment Attention은 주의 계산을 여러 비연속 분할로 분해하고, 계산 인식 블록 삭제자는 각 블록의 기여도를 기반으로 캐시 삭제를 우선시합니다. 또한, 청킹 스케줄러는 다양한 워크로드에서 재계산 오버헤드와 GPU 커널 효율성을 균형 있게 조정하여 메모리 사용을 최적화합니다.

- **Performance Highlights**: 실험 결과, AsymCache는 최신 기준 대비 최대 1.90-2.03배의 시간-최초-토큰(time-to-first-token, TTFT) 단축과 1.62-1.71배의 출력-토큰(time-per-output-token, TPOT) 단축을 기록했습니다. 또한, Continuum와 같은 기존의 에이전트 서빙 시스템에 통합할 경우, 평균 작업 지연(latency)을 최대 18.1%까지 줄이는 성과를 보였습니다. 이로써 AsymCache는 효율적인 계산과 캐시 적중률 사이의 균형을 이루는 설계 목표를 성공적으로 달성한 것으로 확인되었습니다.



### SCOPE: Real-Time Natural Language Camera Agent at the Edg (https://arxiv.org/abs/2606.02951)
Comments:
          9 pages, 4 figures, 6 tables. Accepted at HRI '26 (21st ACM/IEEE International Conference on Human-Robot Interaction), Edinburgh, Scotland, March 16--19, 2026. Code: this https URL

- **What's New**: 이 논문에서는 로봇 공학에서 자연어 기반의 언어 주도 에이전트를 배포하기 위한 새로운 평가 방법인 SCOPE(Simulation and Camera Operations for Perception and Evaluation)를 제안합니다. 이 시스템은 Blender 기반의 시뮬레이션 환경과 실제 PTZ(팬-틸트-줌) 카메라에서 작동하며, 언어 모델과 감지 및 제어 도구를 결합하는 것을 목표로 합니다. SCOPE는 536개의 작업으로 구성된 벤치마크를 통해 높은 시뮬레이션과 실제 환경 간의 전이 가능성을 보장합니다.

- **Technical Details**: SCOPE는 분리된 설계를 채택하여, Compact SLM이 고수준의 계획자로서 카메라 제어 및 감지 쿼리와 reasoning을 조정합니다. 시각 이해는 호출 가능한 도구로 노출된 Lightweight VLM에 위임되어, 실제 시간의 엣지 지연에서 visual perception과 tool-based control을 동시에 달성하기 어렵다는 현실을 반영합니다. 이 시스템은 각 요청 후 VLM으로부터 얻은 결과를 활용해 임무를 반복적으로 수행하는 방식으로 작동합니다.

- **Performance Highlights**: SCOPE의 평가 결과, 강력한 SLMs를 사용 시 헐리케인 문제(hallucinations)를 줄이고 도구 경로 설정(tool routing)을 개선하여 신뢰할 수 있는 닫힌 루프(closed-loop) 동작을 생성할 수 있다는 것을 확인했습니다. 또한, Mixture-of-Experts 모델을 통해 비교적 적은 메모리 사용과 지연 시간으로 밀집 대안(dense alternatives)에 비해 일관된 성능을 나타냈습니다. 양자화(Quantization)를 통해 추가적인 효율 향상이 이루어졌으며, 이는 실시간 PTZ 제어의 실제 심리학적 설계 포인트를 제시합니다.



### Large AI Models in Dental Healthcare: From General-Purpose Systems to Domain-Specific Foundation Models (https://arxiv.org/abs/2606.02914)
- **What's New**: 이번 논문에서는 2020년부터 2026년까지 발표된 대규모 AI 모델을 치과 의료에서 개발, 평가 또는 적용한 97개의 연구를 체계적으로 분석하였습니다. 저자들은 이러한 모델들을 아키텍처 패러다임과 치과 전문화 정도에 따라 분류할 수 있는 2차원 분류 프레임워크를 제안합니다. 이 연구는 AI 모델 간의 관계와 한계점을 포괄적으로 이해하기 위해 체계적인 문헌 검색을 수행하였습니다.

- **Technical Details**: 연구는 PRISMA-ScR 지침을 따르며, PubMed, Google Scholar, Scopus, arXiv의 4개 데이터베이스에서 체계적인 검색을 진행하였습니다. 검색 용어는 'large language model', 'vision-language model', 'transformer' 등 다양한 AI 모델 관련 용어를 포함하였고, 이를 바탕으로 치과 관련 연구를 수집하였습니다. 또한, 저자들은 이 모델들이 서로 어떻게 보완적으로 작용할 수 있는지에 대한 분석도 수행하였습니다.

- **Performance Highlights**: 모델 성능 분석 결과, 언어 생성 모델은 텍스트 기반 작업에서 우수한 성능을 보였으나 이미지 의존 진단에서는 일관되지 않은 결과를 보였습니다. 반면, 치과 특화 모델인 DentVFM 및 OralGPT는 복합적인 다중 모드 작업에서 가장 뛰어난 성능을 입증했으며, 통합된 파이프라인이 단일 모델 접근 방식보다 일관되게 우수한 성과를 나타냈습니다. 하지만 여전히 생성 모델에서의 환각(hallucination), 제한된 주석이 달린 데이터셋, 표준화된 임상 평가 기준의 부재 등의 과제가 해결되어야 안전하게 자율 배포가 가능할 것입니다.



### When Helping Hurts and How to Fix It: Multi-Agent Debate for Data Cleaning (https://arxiv.org/abs/2606.02866)
Comments:
          27 pages, 4 figures, 12 tables. Includes appendix with full experimental results, prompt templates, and dataset statistics

- **What's New**: 이 연구는 데이터 클리닝에서 다중 에이전트 토론(multi-agent debate)이 언제 도움이 되고 언제 해가 되는지를 분석합니다. 총 세 개의 벤치마크, 네 개의 모델 계열 및 6,000개 이상의 작업-조건 쌍을 통해, 토론이 모든 모델에서 생성 품질을 저하시키지만 오류 감지는 크게 개선된다는 결과를 발견했습니다. 이 연구는 Critic의 검증 가능성이 Generator의 기본 정확도를 초과할 때 토론이 효과적이라는 조건을 제시합니다.

- **Technical Details**: 연구는 Critic이 Generator의 제안을 검증하는 방식을 통해 데이터 클리닝의 오류를 줄이기 위한 다중 에이전트 시스템의 방법론적 접근을 제공합니다. 연구에 따르면, Critic과 Generator를 분리해 각각 독립적으로 작동시키는 방식이 필요하며, 코드 실행(ground code execution) 및 evidence-gated generation이 결합될 때 가장 효과적인 결과를 도출합니다. 데이터 클리닝 작업에서 발생하는 비효율성인 critique-induced confusion(CIC)을 해결하기 위한 여러 실험 결과도 보고했습니다.

- **Performance Highlights**: 비교 실험을 통해 제안된 조건이 9개 작업 타입 모두에서 정확한 예측을 제공하며, 19개의 출판된 비교에서 제로의 false positive를 기록하여 일반화 가능성을 입증했습니다. 최종적으로, 연구팀은 토론이 필요한 경우와 아닌 경우를 명확하게 정의하는 의사결정 규칙을 제공하여 실무자들이 중요 품질 이득을 놓치지 않도록 도움을 주고 있습니다. 한정된 입력 구성을 넘어서는 최초의 debate 구성 방식을 제시하며, 단일 에이전트보다 5.3포인트 이상의 개선을 입증했습니다.



### Do Neural Retrievers Prefer Certain Documents? Evidence of Learned Relevance Priors (https://arxiv.org/abs/2606.02814)
- **What's New**: 이번 연구에서는 신경 검색기(neural retrievers)가 쿼리-문서 쌍을 기반으로 문서의 관련성을 추정하는 방식에서 발생하는 문제점을 탐구합니다. 주목할 점은, 주어진 주석 데이터에서 학습한 모델이 문서의 관련성을 넘어서, 특정 문서 유형에 대한 편향된 선호를 내재화하고 있다는 것입니다.

- **Technical Details**: 연구팀은 주석된 문서 임베딩(document embeddings)을 동결(frozen) 한 상태에서 간단한 분류기(classifiers)를 훈련하여 문서 수준의 관련성 우선 신호를 추정했습니다. 이는 훈련된 bi-encoder retrievers가 문서 관련성을 독립적으로 어떻게 회귀(represent)하는지를 평가하기 위해 여러 정보 검색(IR) 벤치마크에서 최첨단(Search) 검색기들이 어떻게 작동하는지를 조사한 결과입니다.

- **Performance Highlights**: 결과적으로, 감독형 신경 검색기는 일반화(generalize) 가능한 우선 신호를 내포하고 있으며, 이는 기존에 보지 못한 문서에 대해서도 일관성을 유지함을 보여줍니다. 특히, 낮은 우선도를 가진 문서는 실제로 관련성이 있음에도 불구하고 더 어렵게 검색되는 경향이 있으며, 이는 기존 문서의 비교에서도 지속적으로 나타나는 현상입니다.



### Traj-Evolve: A Self-Evolving Multi-Agent System for Patient Trajectory Modeling in Lung Cancer Early Detection (https://arxiv.org/abs/2606.02812)
- **What's New**: 이번 연구에서는 Traj-Evolve라는 자가 진화하는 다중 에이전트 시스템을 제안하며, 이 시스템은 폐암 예측을 위한 환자 경로 모델링에서 기존의 제약을 해결합니다. Traj-Evolve는 두 가지 보완적 진화 메커니즘을 통해 환자 데이터를 처리하여 환자와의 연관성을 강화하고 성능을 지속적으로 개선합니다. 특히, 경험 풀(Experience Pool)과 다중 에이전트 강화 학습(Multi-Agent Reinforcement Learning) 기법이 결합되어 진화를 촉진합니다.

- **Technical Details**: Traj-Evolve는 경험 풀(ExPool)을 통해 유사한 환자 데이터를 효과적으로 검색하고, 다중 에이전트 강화 학습(MARL)을 통해 에이전트 간 협력을 최적화합니다. 이 시스템은 환자마다 독립적으로 처리되는 기존 LLM 기반 시스템의 한계를 극복하고, 이전의 검증된 사례를 활용하여 평가 및 예측의 정확성을 높입니다. 이를 통해 환자의 경과에 대한 복잡한 시계열 추론을 가능하게 합니다.

- **Performance Highlights**: Traj-Evolve는 폐암 예측 작업에서 5년간의 다중 모달 전자 건강 기록을 활용하여 9개의 강력한 기준선 모델을 초월하는 성과를 보였습니다. 특히, 결혼 흡연자를 포함한 더 까다로운 소그룹에서도 탁월한 성능을 보여, 이 시스템이 임상 의사결정 지원 도구로서의 가능성을 지니고 있음을 시사합니다. 추적 개선 메커니즘의 동역학 분석 결과, 경험 풀과 MARL의 상호 보완성이 확인되었습니다.



### Attention Calibration for Position-Fair Dense Information Retrieva (https://arxiv.org/abs/2606.02737)
- **What's New**: 이 논문에서는 정보 검색에서 밀집 검색 모델의 위치 편향(position bias)을 해결하기 위한 새로운 방법을 제안합니다. 연구자들은 재훈련 없이 검색의 전반적인 효과를 유지하며 추론 시간에서 편향을 줄일 수 있는지 여부에 대해 조사합니다. 이들은 주목(calibration) 방법을 확장하여 조정 가능한 강도 계수(strength coefficient) λ를 도입하여, 원래의 주의 분포와 완전히 조정된 주의 분포 간의 상호작용을 조절합니다.

- **Technical Details**: 이 연구는 관심(calibration)이 정보 검색에서 어떻게 작용할 수 있는지를 분석합니다. 각 모델의 매개변수인 바구니 크기(basket size), 조정된 레이어 세트(calibrated layer set), 그리고 강도(strength)가 모델의 성능에 미치는 영향을 체계적으로 조사합니다. 저자들은 이러한 매개변수를 통해 위치 일관성과 검색 품질 사이의 균형을 어떻게 조절할 수 있는지를 연구하고, 여러 언어와 도메인에서 적용할 수 있는 모델 무관한 기본 설정(default configuration)을 제시합니다.

- **Performance Highlights**: 결과적으로, 기본 설정(B=128, λ=0.5, 50% layer depth)을 사용함으로써 FineWeb-PosQ에서 모든 모델이 위치 그룹 간 nDCG@10의 조화 평균(harmonic mean)을 개선하였습니다. 또한, 이 설정은 PosIR 벤치마크에서 10개 언어와 31개 도메인으로 이전되어 모든 조합에서 위치 민감도(PSI)를 줄이며, nDCG@10 지표를 유지하거나 개선하는 동시에 긍정적인 결과를 보였습니다.



### Filter, Then Reweight: Rethinking Optimization Granularity in On-Policy Distillation (https://arxiv.org/abs/2606.02684)
- **What's New**: 본 연구에서는 FiRe-OPD(필터링 후 리웨이팅)를 제안하여 온-정책 증류(On-Policy Distillation, OPD)에서 경량화된 텍스트 모델 학습을 향상시키고자 한다. FiRe-OPD는 저품질 샘플을 필터링하고, 필터링된 경로 내에서 정보 가치가 높은 토큰에 지속적으로 중요도를 부여하는 방식으로 기능한다. 이 방식은 기존의 하드 선택 방법보다 더 부드러운 최적화(soft optimization)를 가능하게 하여 정보 손실을 줄이고 최적화 안정성을 높인다.

- **Technical Details**: FiRe-OPD는 두 가지 단계로 나누어 생각할 수 있다. 첫 번째로, 경로 수준에서 낮은 품질의 롤아웃(rollout) 샘플을 필터링하여 불량한 감독 신호를 제거한다. 두 번째로, 남겨진 경로에서 정보가 유의미한 토큰에 대한 중요도를 부여하여 최적화 과정을 조정한다. 이를 통해 FiRe-OPD는 하드 선택(hard selection)보다 더 정교한 OPD 최적화를 가능하게 한다.

- **Performance Highlights**: 우리는 FiRe-OPD가 다양한 설정(단일 교사, 다중 교사)에서 강력한 성능을 보여주며, 최근의 토큰 수준 OPD 방법들보다 우수함을 입증하였다. 성능 비교에서 FiRe-OPD는 AIME 2024에서 +6.25의 성과, Miner에서 +18.81의 성과를 기록하며 기존 방법보다 뛰어난 결과를 달성하였다. 이러한 연구 결과는 OPD의 새로운 방향성을 제시한다.



### Hallucination Is Linearly Decodable from Mid-Layer Hidden States in Quantized LLMs (https://arxiv.org/abs/2606.02628)
- **What's New**: 본 연구는 오픈소스 LLM이 숨겨진 상태(hidden states)에 선형적으로 분리 가능한 진실성 신호(truthfulness signal)를 인코딩하는지를 조사하고, 이 신호가 가장 강하게 나타나는 네트워크 깊이를 분석합니다. 7B에서 8B 사이의 세 가지 instruction-tuned 모델을 통해 hallencination 기준으로 네 가지 방법을 비교하여 유의미한 발견을 도출하였습니다. 최종적으로 연구팀은 코드와 데이터를 공개하여 단일 8GB GPU 환경에서의 재현성을 보장했습니다.

- **Technical Details**: 연구에서는 Llama-3.1-8B, Mistral-7B, Qwen2.5-7B 모델을 4비트 NF4 양자화(NF4 quantization)로 로드하고, 4개의 hallucination 벤치마크에서 각 레이어의 숨겨진 상태를 추출했습니다. 선형 프로브(linear probe)가 중간 네트워크 레이어의 경우 0.904에서 1.000의 AUROC를 달성했으며, 샘플링 기반 탐지기는 동일한 조건에서 0.541 AUROC를 넘지 못했습니다. 또한 모델 패밀리에 걸쳐 peak probing layers가 일관된 범위에 분포하여 특정 블록에서 신호가 명확하게 나타나는 것을 확인했습니다.

- **Performance Highlights**: 본 연구의 주요 성과 중 하나는 중간-후반 transformer 블록에서 진실성과 환각된 숨겨진 상태가 선형적으로 분리 가능하다는 것입니다. Llama와 Mistral의 경우는 블록 13-18, Qwen의 경우는 블록 19-25에서 최대 성능을 보였으며, 첨두 AUROC는 HaluEval-QA에서 0.998에 도달했습니다. 비교한 방법론들 중 선형 프로브가 다른 방법들보다 월등히 높은 성능을 보이는 것을 확인했으며, 이는 평가 프로토콜이 성능에 결정적인 영향을 미친 것을 나타냅니다.



### WUSH: Near-Optimal Adaptive Transforms for LLM Quantization (https://arxiv.org/abs/2512.00956)
Comments:
          Published as a conference paper at the 43rd International Conference on Machine Learning (ICML 2026): this https URL

- **What's New**: 이 논문에서는 대형 언어 모델(Large Language Models, LLMs)을 위한 비트 수 감소(quantization) 기술에 대한 새로운 접근 방식을 소개합니다. 기존의 Hadamard 변환 대신 데이터 통계를 고려한 최적의 선형 블록 변환을 제시하며, 이를 통해 정확한 양자화를 달성할 수 있는 방법을 탐구합니다. 제안된 모델인 WUSH는 데이터 인식 변환과 평균 제곱 손실(minimizing loss in second-order statistics)을 최소화하는 구조를 갖추고 있고, 효율적인 구현이 가능합니다.

- **Technical Details**: WUSH는 블록화된 가중치와 활성화(activation) 양자화를 동시에 처리하는 비정규(비직교) 대칭 효과적이지 않은(transform) 선형 블록 변환을 활용하여 설계되었습니다. 이 변환은 특정 차원에서 데이터를 모델링하는 singular value decomposition(SVD)와 Cholesky 분해를 포함하여 최적의 적응형 변환을 제공합니다. 이 방법은 기존의 SpinQuant 및 FlatQuant와는 차별화되는 명확한 수식으로 구성되어 있습니다.

- **Performance Highlights**: 초기 실험 결과에 따르면, WUSH는 기존의 Hadamard 변환보다 우수한 성능을 보여주며, 일반적인 숫자 형식에서 양자화 정확도를 크게 향상시킵니다. 특히, 부동 소수점 형식(floating-point format)과 정수형(block quantizers)에서 최적의 성능을 발휘합니다. 이러한 성능 향상은 더 정교한 양자화 스케일을 통한 극단값 처리(outlier handling)에서 기인합니다.



### AgentCL: Toward Rigorous Evaluation of Continual Learning in Language Agents (https://arxiv.org/abs/2606.02461)
Comments:
          10 pages in the main text, 26 pages in total

- **What's New**: 이 논문은 언어 에이전트의 지속적인 학습(Continual Learning, CL)을 평가하기 위한 새로운 프레임워크인 AgentCL을 소개합니다. 기존의 벤치마크들이 에이전트가 다양한 작업을 통해 학습한 경험을 효과적으로 검토하지 못하는 문제를 해결하려고 합니다. AgentCL은 명확한 작업 관계를 설정하고, 재사용 가능한 경험을 강조하여 에이전트가 미래 작업에서 그것을 잘 활용할 수 있는지를 측정합니다.

- **Technical Details**: AgentCL은 작업 스트림을 조절하고 플라스틱성(plasticity), 안정성(stability), 일반화(generalization) 등의 지표를 정량화할 수 있는 평가 프로토콜을 제공합니다. 에이전트는 작업을 해결하기 위해 메모리에서 유용한 정보를 검색하고, 환경과 상호작용하며 작업을 완료한 뒤에는 메모리를 업데이트합니다. MemProbe라는 방법론을 통해 메모리 설계의 다양한 요소가 지속적인 학습에 미치는 영향을 분석합니다.

- **Performance Highlights**: 실험 결과, 전통적인 임의의 작업 스트림에서는 메모리 설계 간의 성능 차이를 명확히 드러내기 어려웠지만, AgentCL의 구성된 작업 스트림에서는 각 설계의 플라스틱성이 잘 구분되었습니다. 기존의 메모리 설계는 안정성 문제를 자주 발생시켰고, 이러한 문제를 해결하기 위한 메모리 설계의 필요성이 강조되었습니다. 본 연구는 지속적인 학습을 위한 강력한 메모리 디자인의 필요성을 보여줍니다.



New uploads on arXiv(cs.IR)

### Taiji: Pareto Optimal Policy Optimization with Semantics-IDs Trade-off for Industrial LLM-Enhanced Recommendation (https://arxiv.org/abs/2606.03866)
Comments:
          8 pages, 2 figures

- **What's New**: 이번 논문에서는 산업 추천 시스템을 위해 설계된 혁신적인 LLM-as-Enhancer 프레임워크인 Taiji를 소개합니다. Taiji는 기존의 LLM4Rec 패러다임이 직면하고 있는 SFT(세미 슈퍼비전 파인 튜닝)와 RL(강화 학습) 단계의 문제를 해결하는 방법을 제시합니다. 이를 통해 Taiji는 고유의 Domain-Specific Chain-of-Thought 데이터 생성을 위한 역설계된 추론과 개방형 거부 샘플링을 활용하여 추천 품질을 높입니다.

- **Technical Details**: Taiji는 데이터 구축, 추론 활성화, LLM-추천 협업 및 온라인 순위 지정을 포함한 네 개의 주요 모듈로 구성되어 있습니다. EUPR(Reverse-Engineered User Preference Reasoning)과 ORFT(Open-Ended Rejection Sampling Fine-Tuning)를 통합하여 추천 특화 CoT의 품질을 향상시키고, POPO(Pareto Optimal Policy Optimization)라는 방법을 통해 LLM의 의미적 보상과 추천 선호 보상의 균형을 동적으로 조정합니다.

- **Performance Highlights**: 다양한 오프라인 평가와 온라인 A/B 테스트를 통해 Taiji의 효과성을 검증하였습니다. Taiji는 Kuaishou의 광고 플랫폼에 2026년 5월부터 배포되어 매일 4억 명 이상의 사용자에게 서비스를 제공하며, 상업적 수익을 획기적으로 증가시켰습니다. A/B 테스트 결과, 광고주 가치(ADVV)가 2.83% 개선되고 전반적인 수익이 3.30% 증가하는 성과를 보였습니다.



### When Does Latent Reasoning Help? MeRa: Metric-Space Bias for Spatial Prediction (https://arxiv.org/abs/2606.03727)
- **What's New**: 이 논문에서는 Latent reasoning이 시퀀스 추천 시스템에서 예측 전에 표현을 반복적으로 개선하는 데 성공했음에도 불구하고, 공간 예측(spatial prediction)에서는 그 효과가 기초 메트릭 공간(metric space)에 기반하는지에 따라 달라진다는 점을 발견하였습니다. 메트릭 공간 기반 편향(metric-space bias)을 도입하여 라티언트 추론을 공간 예측 모델에 통합하려는 방법론을 제안하며, 이는 POI 추천 시스템에서의 성능을 향상시킵니다. 이 연구는 MeRa(Metric-space Reasoning)라는 경량 모듈을 제안하여, 모든 시퀀스 인코더와 예측 헤드 사이에 삽입할 수 있는 구조적 해법을 제공합니다.

- **Technical Details**: MeRa는 입력 쌍의 거리 함수로부터 유도된 메트릭 공간 편향을 기반으로 하는 다단계 라티언트 추론을 수행합니다. 이 모듈은 LSTM 및 GETNext와 같은 다양한 백본 인코더와 함께 사용할 수 있으며, 훈련 목표의 수정 없이도 적용할 수 있습니다. MeRa는 POI 예측을 위해 사용자의 체크인 데이터에서 지리적 좌표를 활용하여 예측 성능을 향상시키는 방식으로 설계되었습니다. 각 추론 단계에서 크로스-어텐션(cross-attention) 및 피드 포워드 네트워크(feed-forward network)를 활용하여 원래 인코더 출력 데이터를 정제합니다.

- **Performance Highlights**: MeRa는 세 가지 공간 예측 벤치마크에서 최상의 NDCG@10 성능을 달성하였으며, GeoMamba 및 HMST와 같은 최신 접근 방식을 초월합니다. 메트릭 공간 편향이 없는 추론과 있는 추론 간에는 4.5% NDCG@10의 성능 차이를 보였습니다. 또한, 메트릭 공간 편향을 사용하는 경우와 그렇지 않은 경우 성능이 상반되는 결과를 생성하는 것을 통해, 메트릭 공간 지향(Metric-space grounding) 방식이 공간 예측에서 효과적인 라티언트 추론을 위한 필수 조건임을 실험적으로 입증하였습니다.



### MARS: Multi-rate Aggregation of Recency Signals for Sequential Recommendation across Sparse and Dense Regimes (https://arxiv.org/abs/2606.03718)
- **What's New**: MARS는 과거 상호작용의 이력을 기반으로 하여 서로 다른 최신성 척도를 강조하는 K개의 요약을 생성하는 새로운 집계 연산자입니다. 이 모델은 시퀀스 앙코더를 분리하여, 사용자의 최신성을 반영하는 상태를 생성하며, 이를 통해 다중 시간 스케일 구조를 명확히 드러냅니다. MARS는 두 가지 다른 인스턴스를 자동으로 선택하여 드문드문한 데이터와 밀집 데이터에서 최적의 성능을 보여줍니다.

- **Technical Details**: MARS는 앙코더의 은닉 상태를 기반으로 공개된 타임스탬프를 소비하여 K개의 요약을 생성하며, 이들은 학습 가능한 감쇠율을 통해 각기 다른 최신성 척도를 강조합니다. MARS는 평균 시퀀스 길이에 따라 Transformer 기반의 MARS-T와 Mamba 기반의 MARS-M으로 자동 선택되며, $	ext{O}(LdK)$의 시간 복잡도로 작동합니다. 또한, JSD(Jensen-Shannon Divergence) 정규화를 사용하여 여러 주목 분포의 다양성을 보장합니다.

- **Performance Highlights**: MARS는 다섯 개의 공개 벤치마크에서 10개의 Transformer 및 Mamba 기반의 기준 모델과 비교하여 매 번 가장 높은 HR@10 성능을 기록하였습니다. 특히 희소 데이터에서 19.7%의 상대 이득을 보였으며, 밀집 ML-1M 데이터에서는 42% 적은 MFLOPs로 3.2% HR@10과 0.9% NDCG 향상을 이루었습니다. MARS는 정확성과 효율성의 파레토 전선에서 최고의 성능을 유지합니다.



### Skill Is Not Document: A Query-Conditional Benchmark and Two-Stage Retriever for LLM Agent Skill Routing (https://arxiv.org/abs/2606.03565)
Comments:
          19 pages, 8 figures

- **What's New**: 이번 논문에서는 LLM(대형 언어 모델) 에이전트가 복합 과제를 수행하기 위해 여러 기술(스킬)을 조합하는 방법을 다룹니다. 'Reject-as-Resource Retriever (R3)'라는 새로운 개념을 도입하여, 스킬 검색 과정에서 스킬의 상호 호환성(skill compatibility)을 명시적인 학습 신호로 활용하였습니다. 또한 R3-Skill이라는 중국어-영어 이중언어 스킬 검색 벤치마크를 구축해 현실적인 에이전트 스킬 라우팅을 목적으로 하고 있습니다.

- **Technical Details**: 기존의 LLM 기반 데이터 합성 파이프라인에서 '어떤 스킬을 함께 검색하지 말아야 하는가'에 대한 신호를 생성할 수 있음을 보여줍니다. 이러한 신호는 종종 저품질 데이터로 간주되어 무시되지만, 본 연구에서는 이를 적극 활용하는 방안을 제시합니다. R3-Embedding과 R3-Reranker의 두 단계 검색 시스템을 개발하였고, 이 시스템은 명시적인 스킬 호환성 신호를 기반으로 학습됩니다.

- **Performance Highlights**: 제안된 R3-Embedding + R3-Reranker 파이프라인은 R3-Skill 데이터셋에서 Hit@1 = 0.7714, NDCG@10 = 0.8327, Set-Compat = 0.3525의 성과를 달성하였습니다. 이 연구의 결과는 에이전트 스킬 라우팅에 대한 새로운 통찰을 제공하며, 전체 데이터셋, 학습 코드 및 모델 가중치가 오픈 소스로 공개되어 향후 연구에 활용될 수 있습니다.



### Can LLM Rerankers Predict Their Own Ranking Performance? (https://arxiv.org/abs/2606.03535)
- **What's New**: 이번 논문은 LLM(대형 언어 모델) 리랭커가 스스로 생성한 순위의 품질을 추정할 수 있는지에 대한 연구를 진행합니다. 기존의 쿼리 성능 예측(QPP) 방법들이 주로 검색 후 외부 예측자에 의존했던 반면, 이 연구는 리랭커 내부의 QPP를 중점적으로 분석합니다. 이를 통해 LLM 리랭커가 직접 생성한 순위의 품질을 평가하는 신호가 있을 수 있음을 강조합니다.

- **Technical Details**: 연구는 훈련 없이도 측정하는 기법과 훈련 기반 방법을 다룹니다. 훈련 없는 추정 기법으로는 샘플링된 순위에서 메트릭별 일관성과 리랭커가 직접 생성한 수치화된 신뢰도를 확인합니다. 그리고 감독 훈련을 통해 리랭커가 더 정확한 신뢰도를 표현할 수 있는지 탐구합니다.

- **Performance Highlights**: 실험 결과, LLaMA3.1 및 Qwen2.5 모델을 사용한 결과, 일관성 기반 접근법이 QPP-Gen과 경쟁력 있는 성능을 보였으며, 거의 모든 설정에서 더 나은 캘리브레이션을 나타냈습니다. 반면에 한정된 훈련 데이터를 통해 제안된 두 가지 감독 방법인 Verb-Num과 Verb-List는 리랭커가 저렴한 비용으로 캘리브레이션된 순위 품질 추정치를 생성하도록 도와줍니다.



### Automating Information Extraction and Retrieval for Industrial Spare Parts Pooling (https://arxiv.org/abs/2606.03367)
- **What's New**: 이 논문에서는 제조업의 유지보수 조직들이 부품 재사용을 통해 다운타임(downtime)과 불필요한 구매를 피하려고 시도하지만, 자산의 가시성이 부족하다는 문제를 제기합니다. 이를 해결하기 위해 PhRAG라는 하이브리드 Retrieval-Augmented Generation 기법을 제안하여, 여러 재고를 통합해 가상 재고 풀(Virtual Stock Pool, VSPool)로 구성할 수 있도록 합니다. 이 연구는 NER(Named Entity Recognition) 기법을 통해 비구조적인 부품 설명을 정형화하고, 자연어로 된 요건을 기반으로 강력한 검색 기능을 지원하는 방법을 보여줍니다.

- **Technical Details**: 논문의 제안된 방법론은 두 가지 주요 단계로 나뉘며, 기술 사양 추출과 검색 엔진 구현으로 구성됩니다. 구조화된 고객 요구를 바탕으로 문자 요청을 공유된 데이터에 매핑하기 위해 가상 재고 풀(VSPool)을 활용하는 검색 엔진이 개발되었습니다. 생성 언어 모델(generative language models)의 많은 사전 훈련된 모델 및 그들의 자동 회귀 텍스트 생성을 기반으로 하여, 기술 사양 추출 작업을 수행하는 NER 작업에서 데이터를 처리할 수 있는 수단을 제공합니다.

- **Performance Highlights**: PhRAG 프레임워크는 정보 검색 시스템의 불투명성을 극복하고, 검색된 구성 요소에 대한 정당성을 생성하는 방식으로 기존 NER 접근법에 비해 생성적 접근의 가능성을 보여줍니다. 또한, 이 연구는 기술 정보 추출 작업 및 산업 부품 검색에서 언어 모델의 다중 작업 능력을 평가하고, VSPool과 같은 독점 데이터셋을 통해 효율성과 저지연 추론을 구현하며, 산업 현장에 적합한 솔루션을 제공합니다.



### Generalizing Graph Foundation Models via Hyperbolic Retrieval-Augmented Generation (https://arxiv.org/abs/2606.03307)
Comments:
          Accepted by KDD2026

- **What's New**: 이번 연구는 그래프 기초 모델(Graph Foundation Models, GFMs)의 일반화 능력을 향상시키기 위한 새로운 프레임워크인 Hyperbolic Retrieval-Augmented Generation (HyRAG)을 제안합니다. 기존 Euclidean 공간에서 작동하는 RAG 프레임워크의 기하학적 한계를 해결하고, 외부 지식을 효과적으로 통합하여 성능을 개선합니다. 특히, 제안된 하이퍼볼릭 지식 인덱싱 모듈은 트리 구조의 외부 지식을 하이퍼볼릭 공간에 모델링하여 의미의 정밀도를 유지합니다.

- **Technical Details**: HyRAG 프레임워크는 하이퍼볼릭 공간의 기하학적 특성을 활용하여 기존 RAG 시스템의 두 가지 주요 문제인 의미의 정밀성 손실 및 허브 문제(hubness)를 해결합니다. 하이퍼볼릭 지식 인덱싱 모듈은 거리 기반 목적과 각도 제약을 포함하여 비지도 최적화 방식으로 하이퍼볼릭 의미 인코더를 사전 훈련합니다. 이를 통해 다양한 세밀한 정보를 캡처할 수 있는 다중 정밀도 검색(Multi-granularity Retrieval) 전략을 개발하여, 전역 의미 고정 요소와 지역 의미 뉘앙스를 효과적으로 제공합니다.

- **Performance Highlights**: 다양한 그래프 벤치마크에서 HyRAG의 성능을 평가한 결과, 제안된 방법이 zero-shot 설정에서 GFMs의 추론 정확도를 크게 개선했습니다. 실험 결과, 새로운 구조적 융합 메커니즘에 의해 그래프 작업에서 지식 통합이 효과적으로 이루어졌으며, 이는 기존 모델에 비해 우수한 일반화 성능을 입증합니다.



### VirtualMLE: A Virtual ML Engineer that Optimizes Sequential Recommenders (https://arxiv.org/abs/2606.03221)
- **What's New**: 최근 대형 언어 모델(Large Language Models, LLMs)의 발전으로 인해 복잡한 엔지니어링 작업 흐름을 자동화하는 새로운 패러다임이 열렸습니다. 그러나 시퀀스 추천(sequential recommendation, SR) 분야에서 모델을 새로운 데이터셋에 조정하는 것은 여전히 숙련된 머신러닝 엔지니어의 수동적인 시행착오에 의존하고 있습니다. 이를 해결하기 위해 제안된 VirtualMLE는 LLM의 인지 능력을 활용하여 추천 최적화를 실행, 반영 및 메모리 업데이트의 닫힌 루프로 구성합니다.

- **Technical Details**: VirtualMLE는 SR 튜닝을 위한 경량 에이전트 프레임워크로, 기존의 오토ML 방법과 달리 특성을 갖추고 있습니다. 이 프레임워크는 계층적 메모리 시스템을 통해 데이터셋 간에 전이 가능한 지식을 보존하고 활용하며, LLM 기반 반복 및 계획 → 행동 → 관찰 → 업데이트 루프를 포함하여 검색 프로세스를 정리합니다. 이를 통해 복잡한 구조 조정 및 하이퍼파라미터 최적화를 보다 효율적으로 수행합니다.

- **Performance Highlights**: 실험 결과, VirtualMLE는 두 가지 대표적인 SR 백본(SASRec 및 HSTU)을 사용하여 세 가지 아마존 벤치마크에서 비교한 AutoML 기반선보다 일관되게 더 나은 성능을 발휘하였습니다. 또한, 학습한 인지 요약이 서로 다른 추천 도메인에서도 전이 가능함을 보여주며, 이전 데이터셋에서 얻은 통찰력을 바탕으로 새로운 데이터셋에서의 수렴 비용을 줄이는 데 기여했습니다. VirtualMLE는 LLM을 명시적 추론기로 위치시키며, 인식 결과와 전이 가능한 규칙을 생성하는데 중점을 두고 있습니다.



### Section-Weighted Hybrid Approach for Legal Case Retrieva (https://arxiv.org/abs/2606.03138)
Comments:
          10 pages, 4 figures. Accepted to the International Conference on Natural Language Processing (ICNLP 2026)

- **What's New**: 본 논문에서는 법률 판례 검색을 위한 새로운 프레임워크를 제안합니다. 이 시스템은 판결을 사실, 쟁점, 결정 및 논리로 구분하여 처리하며, 두 단계로 이루어져 있습니다. 첫 번째 단계에서는 BM25와 밀집 근사 최근접 이웃 검색을 결합하여 후보군을 생성하고, 두 번째 단계에서는 정밀한 비교를 통해 유사성을 평가합니다. 이를 통해 법률적 사고에 대한 보다 정확한 검색 결과를 제공합니다.

- **Technical Details**: 제안된 프레임워크는 두 단계로 구성됩니다. 첫 번째 단계에서는 전체 문서를 검색해 유사한 후보군을 확보하며, BM25와 밀집 ANN을 사용하여 검색의 폭을 넓힙니다. 두 번째 단계에서는 유사한 항목 간의 정밀한 비교를 실시하고, 학습된 가중치를 사용하여 점수를 집계합니다. 이 과정에서 Z-점수 정규화를 적용하여 결과의 안정성을 확보합니다.

- **Performance Highlights**: 이 시스템은 법률 검색의 정확성을 크게 향상시키며, 대규모 벤치마크에서 기존의 강력한 기준과 비교해 일관된 성과 향상을 보여줍니다. 특히, 후보군의 포괄성을 유지하면서도 높은 정밀도를 확보하였습니다. 최종 결과로는 관련된 판결의 섹션 텍스트와 간결한 설명, 그리고 당사자 입장 레이블이 포함됩니다.



### BAHSD: Bridging the Long-tail Gap via Adaptive Distillation in Black-box Sequential Recommendation (https://arxiv.org/abs/2606.03091)
- **What's New**: 본 논문에서는 순차 추천 시스템에서의 신호 이질성 문제를 해결하기 위해 BAHSD(Black-box Adaptive Heterogeneous Signal Distillation)라는 새로운 프레임워크를 제안합니다. 기존의 모델 추출 방식은 사용자 상호작용의 긴 꼬리 분포에 따른 신호의 다양성을 간과하고, 이로 인해 헤드 사용자(주요 사용자)와 테일 사용자(인기 없는 사용자) 간 성능이 상이하게 나타납니다. BAHSD는 다중 스케일 일관성 탐사를 통해 신호 신뢰성을 평가하고, 이를 바탕으로 계층적 목표를 조정하여 다양한 사용자 신호에 적응하는 방식을 채택합니다.

- **Technical Details**: BAHSD는 교수 모델의 로그만을 사용하여 작동하며, 다중 길이의 서브 시퀀스에서의 교수 출력의 일관성을 측정하여 신뢰도를 정량화합니다. 신뢰도가 높은 신호에 대해서는 동적 온도를 적용한 KL 발산을 사용하여 선호 고착화를 완화하고, 신뢰도가 낮은 신호는 순위 일관성과 InfoNCE 대조 학습을 적용하여 노이즈를 억제합니다. 이러한 방법은 명시적인 사용자 계층화나 내부 접근 없이도 최적의 정보 전이를 가능하게 합니다.

- **Performance Highlights**: BAHSD는 공개 벤치마크에서 기존의 최첨단 모델보다 최대 4.98% 향상을 달성하였고 테일 사용자에 대해서는 80% 이상의 성능 개선을 기록했습니다. 이러한 결과는 BAHSD가 높은 충실도의 블랙박스 모델 추출에 있어 즉시 적용 가능하고, 모델에 구애받지 않는 솔루션임을 나타냅니다. 실험 결과를 통해 본 프레임워크가 신뢰도에 기반한 조정 가능성을 증명하며, 다양한 사용자 그룹의 이질한 신호를 효과적으로 대응하고 있음을 알 수 있습니다.



### Slipstream: Locality-Aware Graph Index Construction for Streaming Approximate Nearest Neighbor Search (https://arxiv.org/abs/2606.02992)
- **What's New**: 저자들은 Slipstream이라는 새로운 방법을 제안하여 그래프 인덱스에서의 빈번한 삽입에 대한 계산 비용을 크게 줄이는 방안을 제시합니다. Slipstream은 새로운 분석 방법론을 사용하여 흐름의 연속성을 활용하고, 임시 후보를 활용해 삽입 과정을 최적화합니다. 또한, 이 방법은 다른 전통적인 인덱스 접근 방식보다 더 나은 성능을 보임을 입증하고 있습니다.

- **Technical Details**: Slipstream의 핵심 아이디어는 이전 삽입에서 찾아낸 후보 목록을 토대로 새로운 포인트를 시작하는 것입니다. 이는 각 새 임베딩을 독립적인 검색 문제로 처리하던 기존 방식과는 다릅니다. Slipstream은 다른 후보 집합의 근접성을 평가하고 적응형 컨트롤러를 사용하여 스트림 안정성에 따라 삽입 범위를 조정합니다.

- **Performance Highlights**: Slipstream은 슬립스트림을 통해 총 성능이 기존 방법보다 최대 30.8배 더 높으며, 적어도 0.95의 recall@10을 유지합니다. 이 연구는 HNSW와 HNSWLib 라이브러리에 Slipstream을 구현하여 여러 비디오 임베딩 스트림에서 평가하였고, 이 결과는 그래프 인덱스의 성능 개선을 명확히 보여줍니다.



### Do Neural Retrievers Prefer Certain Documents? Evidence of Learned Relevance Priors (https://arxiv.org/abs/2606.02814)
- **What's New**: 이번 연구에서는 신경 검색기(neural retrievers)가 쿼리-문서 쌍을 기반으로 문서의 관련성을 추정하는 방식에서 발생하는 문제점을 탐구합니다. 주목할 점은, 주어진 주석 데이터에서 학습한 모델이 문서의 관련성을 넘어서, 특정 문서 유형에 대한 편향된 선호를 내재화하고 있다는 것입니다.

- **Technical Details**: 연구팀은 주석된 문서 임베딩(document embeddings)을 동결(frozen) 한 상태에서 간단한 분류기(classifiers)를 훈련하여 문서 수준의 관련성 우선 신호를 추정했습니다. 이는 훈련된 bi-encoder retrievers가 문서 관련성을 독립적으로 어떻게 회귀(represent)하는지를 평가하기 위해 여러 정보 검색(IR) 벤치마크에서 최첨단(Search) 검색기들이 어떻게 작동하는지를 조사한 결과입니다.

- **Performance Highlights**: 결과적으로, 감독형 신경 검색기는 일반화(generalize) 가능한 우선 신호를 내포하고 있으며, 이는 기존에 보지 못한 문서에 대해서도 일관성을 유지함을 보여줍니다. 특히, 낮은 우선도를 가진 문서는 실제로 관련성이 있음에도 불구하고 더 어렵게 검색되는 경향이 있으며, 이는 기존 문서의 비교에서도 지속적으로 나타나는 현상입니다.



### Attention Calibration for Position-Fair Dense Information Retrieva (https://arxiv.org/abs/2606.02737)
- **What's New**: 이 논문에서는 정보 검색에서 밀집 검색 모델의 위치 편향(position bias)을 해결하기 위한 새로운 방법을 제안합니다. 연구자들은 재훈련 없이 검색의 전반적인 효과를 유지하며 추론 시간에서 편향을 줄일 수 있는지 여부에 대해 조사합니다. 이들은 주목(calibration) 방법을 확장하여 조정 가능한 강도 계수(strength coefficient) λ를 도입하여, 원래의 주의 분포와 완전히 조정된 주의 분포 간의 상호작용을 조절합니다.

- **Technical Details**: 이 연구는 관심(calibration)이 정보 검색에서 어떻게 작용할 수 있는지를 분석합니다. 각 모델의 매개변수인 바구니 크기(basket size), 조정된 레이어 세트(calibrated layer set), 그리고 강도(strength)가 모델의 성능에 미치는 영향을 체계적으로 조사합니다. 저자들은 이러한 매개변수를 통해 위치 일관성과 검색 품질 사이의 균형을 어떻게 조절할 수 있는지를 연구하고, 여러 언어와 도메인에서 적용할 수 있는 모델 무관한 기본 설정(default configuration)을 제시합니다.

- **Performance Highlights**: 결과적으로, 기본 설정(B=128, λ=0.5, 50% layer depth)을 사용함으로써 FineWeb-PosQ에서 모든 모델이 위치 그룹 간 nDCG@10의 조화 평균(harmonic mean)을 개선하였습니다. 또한, 이 설정은 PosIR 벤치마크에서 10개 언어와 31개 도메인으로 이전되어 모든 조합에서 위치 민감도(PSI)를 줄이며, nDCG@10 지표를 유지하거나 개선하는 동시에 긍정적인 결과를 보였습니다.



### Cost-Aware Query Routing in RAG: Empirical Analysis of Retrieval Depth Tradeoffs (https://arxiv.org/abs/2606.02581)
Comments:
          13 pages , 18 figures , 8 tables

- **What's New**: 이 논문은 Cost-Aware RAG (CA-RAG)라는 새로운 쿼리 라우팅 프레임워크를 소개합니다. CA-RAG는 다양한 'strategy bundles' 중에서 각 쿼리에 대해 적절한 검색 깊이를 선택하여, 비용(비용 절감), 품질(응답 유용성 극대화), 응답 시간(지연 최소화)의 세 가지 목표를 동시에 고려합니다. 이는 다양한 쿼리 유형에 대한 효율성을 향상시킬 수 있는 가능성을 보여줍니다.

- **Technical Details**: CA-RAG는 FAISS 기반의 밀집 검색과 OpenAI의 채팅 및 임베딩 API를 사용해 구현되었습니다. 사용자가 정의한 우선 순위를 바탕으로 유틸리티를 평가하고, 각 쿼리에 최적화된 전략을 할당합니다. 이를 통해 각 쿼리에 대해 고유한 검색 깊이를 적용하여 더 나은 성능을 발휘합니다.

- **Performance Highlights**: CA-RAG는 28개의 쿼리 벤치마크 테스트에서 동작하며, 항상 깊은 검색을 사용하는 것보다 26% 적은 청구 토큰을 달성했고, 항상 직접 추론하는 방식보다 평균 34% 낮은 지연 시간을 기록했습니다. 이 아키텍처는 또한 더 복잡한 쿼리 유형의 정보 요구를 충족시키면서도 간단한 쿼리 유형에서 가장 큰 절감을 보여줍니다.



### Re-Ranking Through an Attribution Lens for Citation Quality in Legal QA (https://arxiv.org/abs/2606.03728)
Comments:
          11 pages, 4 tables, 1 figure. Published at ASAIL 2026 (8th Workshop on Automated Semantic Analysis of Information in Legal Text), co-located with ICAIL 2026, Singapore

- **What's New**: 이번 연구에서는 법률 질문 응답을 위한 Retrieval-augmented generation 시스템에서 인용의 질 문제를 해결하기 위해, 경량의 cross-encoder를 지속적인 perturbation 기반의 attribution 점수로 훈련하여 포스트 생성 전 단계에서 패시지를 재정렬합니다. 기존의 연구에서는 인용된 패시지가 높은 점수로 평가될 것이라는 가정이 있었지만, 그렇지 않음을 보여줍니다. 이 연구는 AQuAECHR 벤치마크에서 두 개의 언어 모델을 사용하여 검증하여 citation faithfulness와 gold expert answers와의 정렬을 크게 개선했습니다.

- **Technical Details**: 이 연구는 C-LIME와 같은 perturbation 기반의 attribution 기법을 사용하여 각 패시지가 생성된 출력에 미치는 영향을 측정합니다. 이를 통해 얻은 attribution 점수를 바탕으로 light-weight cross-encoder를 훈련하여, 패시지를 재정렬하는 새로운 방법론을 제시합니다. 실험은 Mistral-7B와 Llama-3-8B라는 두 가지 언어 모델을 사용하여 진행되었으며, 다섯 번의 교차 검증을 통해 검증되었습니다.

- **Performance Highlights**: 연구 결과, 재정렬된 패시지를 사용한 경우 인용의 신뢰성과 gold expert answers와의 정렬이 크게 향상되었으며, 동일 모델의 재정렬 방식이 더욱 효과적이라는 것을 발견했습니다. 또한, 독립적으로 훈련된 두 개의 re-ranker가 서로 다른 모델에서 적용되었음에도 불구하고 일치하는 신뢰 신호를 보였으며, 이는 model-specific noise를 줄이는 효과가 있음을 나타냅니다. 이러한 결과는 perturbation 기반의 attribution 방법이 모델 간에 공유되는 relevance 신호를 생성할 수 있음을 보여줍니다.



### Ghost: Plausible Yet Unlearnable Trajectories via On-Manifold Substitution for Next-POI Privacy (https://arxiv.org/abs/2606.03711)
- **What's New**: 이번 논문에서는 사용자의 체크인 경로(check-in trajectories)를 공개하는 것이 예기치 않게 사용자의 미래 위치를 예측할 수 있는 강력한 정보를 제공한다는 위험을 다룹니다. 이를 해결하기 위해 Ghost라는 프레임워크를 제안합니다. Ghost는 사용자가 실제 경로와 유사하지만 학습 신호를 남기지 않는 왜곡된 시퀀스를 생성합니다.

- **Technical Details**: Ghost는 사용자의 실제 체크인 시퀀스와 비슷해 보이는 데이터로 구성된 왜곡된 경로를 생성하는 이양계(bilevel)의 비학습 가능한 경로 프레임워크입니다. 이 방법은 각 대체 후보를 평가하여 지리적으로 및 의미적으로 그럴듯한 후보만을 선택하는 하드한 가정(plausibility constraints)을 가지며, 일반적으로 구축된 대체자에 대한 저항력을 높입니다.

- **Performance Highlights**: Ghost는 두 개의 표준 벤치마크와 네 가지의 공격자 자세에서 강력한 결정론적 기준선(PGD)과 경쟁하는 보호 격차를 달성했습니다. 가장 낮은 복원 정확도를 가지며 구조적으로 정당성을 보장하는 Ghost는 퍼퓨리피케이션(purification) 저항성에서도 우수한 성능을 보이며 모든 고보호 방법 중에서 가장 높은 성능을 기록했습니다.



### Structures Facilitate Retrieve, Rerank, and Genera (https://arxiv.org/abs/2606.03247)
- **What's New**: 이 논문에서는 Document-grounded dialogue systems (DGDS)에서 외부 문서의 지식을 효과적으로 활용하는 방법을 제안합니다. 기존의 접근법은 문서를 독립적인 패세지로 나누어 검색과 응답 생성을 진행하는데, 이는 문서 내 구조 정보를 잘 활용하지 못하고 충분한 문맥을 제공하지 못했습니다. 본 연구에서는 SF-Re2G라는 새로운 방법론을 통해 이러한 문제를 체계적으로 해결하고자 합니다.

- **Technical Details**: SF-Re2G는 패세지를 더 잘 표현하기 위해 동일 섹션 내에서 다른 패세지와 대조하여 검색 성능을 향상시키는 것을 목표로 합니다. 또한, 구조 강화 리랭커(structure-enhanced reranker)를 통해 다수의 기초 패세지가 동일한 대화(turn)에서 인접해 있다는 사실을 활용합니다. 검색 내 후보를 문서 구조에 따라 서브그래프(subgraph)로 그룹화하여 리랭커가 그룹 정보를 통합해 후보의 점수를 재조정합니다.

- **Performance Highlights**: 두 개의 DGDS 데이터셋에서 실험한 결과, 본 방법은 중국어와 영어에서 모두 효과적인 성능을 나타내었습니다. 서브그래프 문맥을 고려하여 선택된 패세지를 응답 생성에 활용함으로써, 보다 나은 응답 품질을 달성했습니다. 이러한 결과는 SF-Re2G의 유용성을 입증하며, 향후 DGDS 개발에 중요한 기여를 할 것입니다.



### Patcher: Post-Hoc Patching of Backdoored Large Language Models (https://arxiv.org/abs/2606.02995)
Comments:
          To appear in the USENIX Security Symposium, 2026

- **What's New**: 본 논문에서는 대형 언어 모델이 jailbreak backdoor 공격에 취약하다는 점을 언급합니다. 기존 방어 체계는 포괄적인 공격 정보나 여러 개의 트리거된 예제가 필요해, 단일 실패 사례만으로는 비현실적인 상황에 처해 달리 대처할 비법이 없었습니다. 이 문제를 해결하기 위해) Patcher라는 새로운 후처리(defense) 프레임워크를 제안합니다.

- **Technical Details**: Patcher는 두 단계로 작동합니다. 첫째, response-conditioned gradient-based saliency 점수를 계산하고 적응형 클러스터링(adaptive clustering)을 적용하여 트리거를 로컬라이즈(localize)합니다. 둘째, KL-divergence 제약 조건을 통해 트리거-응답(trigger-response) 연관성을 끊으면서도 정상적인 작업 유용성과 비트리거된 공격에 대한 강건성을 유지하는 방식으로 모델을 수정(patch)합니다.

- **Performance Highlights**: 저자들은 Patcher의 성능을 여러 백도어 공격 전략에 대해 광범위하게 평가하였으며, Patcher가 트리거를 성공적으로 로컬라이즈하고 백도어를 중화(neutralize)하는 동시에 모델 유용성을 유지함을 보여줍니다. 또한 Patcher는 우리의 방어를 회피하기 위해 설계된 적응형 공격에 대해서도 강건성을 보여 주었습니다.



### LLM-Assisted Reranking to Operationalize Nuanced Objectives in Recommender Systems (https://arxiv.org/abs/2606.02883)
Comments:
          30 pages total; 11 pages, 5 figures, 2 tables (main text); 19 pages, 11 figures, 9 tables (appendix)

- **What's New**: 이번 연구에서는 LLM(large language models)을 활용한 추천 시스템이 정치적 콘텐츠에 대한 노출을 어떻게 재구성하는지를 조사합니다. 저자들은 사용자 시청 기록을 기반으로 한 LLM 기반의 재순위 매김이 개인화된 추천을 어떻게 향상시키는지를 분석하며, 이를 통해 정치적 극단주의 콘텐츠에 대한 노출을 확대할 수 있는 위험성을 실험적으로 입증합니다. 연구에서는 기계 학습 기반의 접근 방식과 사교적 가치 제약을 결합한 새로운 추천 구조를 제안합니다.

- **Technical Details**: 이 연구는 제로샷(zero-shot) 접근법을 사용하여 YouTube 추천 내용을 재정렬하는 데 LLM을 통합한 추천 파이프라인을 개발했습니다. 기존 추천 시스템과의 차별점은 LLM이 정황적 추론(contextual reasoning)의 적용을 가능하게 하여, 이론적 기준 이하로 콘텐츠의 탐색과 노출을 조정하는 것입니다. 이 과정에서 정치적 콘텐츠의 선정 및 재구성에서 발생할 수 있는 사회적 손상을 줄이기 위한 명확한 제어 전략이 적용되었습니다.

- **Performance Highlights**: 실험 결과, LLM의 재순위 매김이 개인화된 추천을 증가시키는 반면, 주요 피험자들은 극단적이고 음모론적인 콘텐츠에 대한 노출이 증가하는 것을 확인했습니다. 그러나 경량화된 프롬프트 레벨 정규화가 극단적 콘텐츠의 노출을 줄이고 이념적 다양성을 증가시키는 데 기여했습니다. 따라서 LLM을 통한 추천 시스템은 콘텐츠 추천의 사회적 영향을 고려하도록 설계될 필요가 있음이 보였습니다.



### IdiomX A Multilingual Benchmark for Idiom Understanding, Retrieval, and Interpretation (https://arxiv.org/abs/2606.02584)
Comments:
          12 pages, 21 figures. Includes dataset and code. Resources available on HuggingFace, Kaggle, and GitHub

- **What's New**: 본 연구에서는 IdiomX라는 대규모 다국어 아이디엄(idiom) 이해, 검색 및 해석을 위한 벤치마크를 소개합니다. 이는 복제 가능한 다단계 파이프라인을 통해 구축되었으며, 기존 아이디엄 자원의 한계를 극복하고자 합니다. 이 데이터셋은 12,000개 이상의 아이디엄을 포함하여, 영어, 아랍어, 프랑스어의 알라인드(Aligned) 의미 표현을 제공합니다.

- **Technical Details**: IdiomX 데이터셋은 190,000개 이상의 맥락화된 예제를 포함하고 있으며, 언어 자원 추출, 대규모 정규화, 통제된 LLM(large language model) 보강, 구조화된 검증을 결합한 절차를 통해 제작되었습니다. 이 자원은 아이디엄 검출, 맥락에서 아이디엄 검색, 아랍어에서 영어로의 아이디엄 검색 및 아이디엄 해석 등 네 가지 주요 작업을 포함합니다.

- **Performance Highlights**: 실험 결과, 맥락 변환기(transformer) 모델이 아이디엄 검출에서 상당한 개선을 보였으며, 하이브리드 검색 및 재정렬 아키텍처가 단일 언어 및 다국어 아이디엄 검색 모두에서 효과적으로 성능을 강화했습니다. 또한 아이디엄 해석이 의미 검색 작업으로 모델링될 수 있으며, 이는 해석 가능성을 추가적인 벤치마크 차원으로 도입함을 보여주었습니다.



New uploads on arXiv(cs.CV)

### SimuScene: Simulation-Ready Compositional 3D Scene Reconstruction from a Single Imag (https://arxiv.org/abs/2606.03994)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문에서는 단일 이미지로부터 상호작용이 가능한 시뮬레이션 준비된 3D 장면을 재구성하는 새로운 파이프라인인 SimuScene을 소개합니다. 기존 방법들은 물리 시뮬레이션에서 발생하는 기하학적 오류로 인해 재구성된 장면이 불안정한 경향이 있었습니다. SimuScene은 물리 엔진을 활용하여 훈련 과정에서 바로 기하학적 오류를 진단하고 수정하는 피드백 루프를 제공합니다.

- **Technical Details**: SimuScene은 단일 이미지에서 3D 장면을 재구성하기 위해 물리학의 피드백을 적용하여 기하학적 오류를 최소화합니다. 중력에 의한 변위와 같은 진단 신호를 활용하여 오차를 줄이고 물리적으로 일관된 장면을 생성하는 데 초점을 맞췄습니다. 기하학적 업데이트는 두 단계로 구성되어 있으며, 이는 소폭의 오류에 대해 중력 축 신장(gravity-axis stretching)과 심각한 형상 실패에 대해 OBB(Oriented Bounding Box) 기반의 형태 재샘플링을 통해 이루어집니다.

- **Performance Highlights**: 실험 결과, SimuScene은 물리적 안정성과 기하학적 정렬 벤치마크에서 기존의 최신 기술보다 우수한 성능을 보였습니다. 또한 이 재구성된 장면은 로봇 팔 조작 및 인간형 로봇 제어와 같은 하위 작업에서 유용성을 입증하였습니다. 결국 이 연구는 로봇 조작 및 VR/AR 환경에서의 상호작용을 지원하는 안정적이고 시뮬레이션 준비가 완료된 3D 장면의 생성 가능성을 제시합니다.



### Exploring Easy Boosts for Lidar Semantic Scene Completion (https://arxiv.org/abs/2606.03992)
Comments:
          Accepted to ICIP 2026

- **What's New**: 이 논문은 lidar semantic scene completion (SSC) 성능을 복잡한 아키텍처 재설계 없이 개선하기 위한 "free lunch" 전략을 조사합니다. 우리는 기존 아키텍처의 성능을 크게 향상시키기 위해 입력 포인트 클라우드에 상용 segmentors에서 얻은 의미적 pseudo-labels를 부여하는 것이 효과적이라는 것을 입증합니다. 또한, 빈 공간(empty space)과 미지의 공간(unknown space) 사이의 구분을 제공하는 가시성 정보(visibility information)를 입력에 추가하여 더욱 성능을 향상시킵니다.

- **Technical Details**: 이 연구에서는 기초적으로 LMSCNet-SS 및 SemCity-AE와 같은 두 가지 기존 SSC 모델을 사용하여 두 가지 중요한 prior인 semantics와 visibility를 검토합니다. Semantic prior은 off-the-shelf point cloud segmentors에서 얻은 의미적 클래스이고, Visibility prior은 센서를 통해 얻은 레이 캐스팅으로 빈 voxel과 미지의 voxel을 구분합니다. 이러한 방법을 통해 성능이 모든 테스트 모델에서 향상되는 것을 관찰하였습니다.

- **Performance Highlights**: 우리는 이러한 방법들을 적절히 결합함으로써 4개 모델의 평균 mIoU 포인트를 5.2 상승시키며 최첨단 성능을 달성하였습니다. 이는 오랜 기간 동안 사용된 구형 모델이 최신 방법과 경쟁할 수 있음을 보여줍니다. 이러한 결합된 전략을 통해 의미적 우선 정보와 가시성 정보의 상호 보완성을 입증하며, 기존의 여러 SSC 아키텍처에서 향상된 성능을 확인하였습니다.



### PixVOD: Pixel-Distributed Direct Visual Odometry and Depth Estimation (https://arxiv.org/abs/2606.03989)
- **What's New**: 이 논문에서는 픽셀별로 계산을 수행하는 고속 비주얼 오도메트리(Visual Odometry)와 깊이 추정 기술을 제안합니다. 기존의 카메라와 프로세서 사이의 명확한 분리를 넘어 시각적 감지와 처리를 결합한 하드웨어를 통해 로컬에서 빠른 처리를 가능하게 하려는 노력의 일환입니다. 특히, Gaussian Belief Propagation(GBP)을 이용하여 각 픽셀에서 정보 교환을 통해 카메라 모션 추정 및 깊이 추정을 수행하는 것을 목표로 했습니다.

- **Technical Details**: 이 방법은 픽셀 배열 전체에 분산된 factor graph를 구성하여 카메라 모션과 장면 깊이를 추정합니다. 핵심적으로, 각 픽셀에서 계산이 이루어지는 구조로 인해, 저전력 및 고속 처리가 가능해집니다. 또한, 키프레임과 유사한 고정 메커니즘을 도입하여 최적화 과정에서 기하학적 안정성을 유지합니다. 이를 통해 픽셀 수준에서의 처리 및 전송 효율성을 크게 향상시킵니다.

- **Performance Highlights**: 제안된 방법은 실제 데이터셋에서 평가되어 GBP 기반의 픽셀 수준 분산 오도메트리 및 깊이 추정의 가능성을 입증했습니다. 특히, 기존의 중앙 집중 방식보다 훨씬 높은 성능을 나타내며, 저전력 소모와 빠른 연산 속도를 실현합니다. 이 연구는 향후 센서-프로세서 통합 시스템 개발에 기여할 것으로 기대됩니다.



### NewtPhys: Do Foundation Models Understand Newtonian Physics? (https://arxiv.org/abs/2606.03986)
- **What's New**: 이번 논문에서는 NewtPhys라는 4D 물리 주석 데이터셋을 소개합니다. 이는 실제 장면의 다중 이미지에서 구축되어 Physics-grounded 시뮬레이션과 함께 사용됩니다. 기존의 신뢰할 수 없는 고차원 이벤트에 의존하지 않고, 저차원 뉴턴 물리적 이해를 평가하는 데 필요한 비주얼 충실도를 제공합니다.

- **Technical Details**: NewtPhys는 3D Gaussian Splatting(3DGS)을 사용하여 실세계 장면과 물체를 뉴턴 시뮬레이터의 시뮬레이트 가능한 입자로 결합합니다. 이 데이터셋은 힘, 변형 필드, 재료 및 인스턴스 맵과 같은 픽셀 정렬된 물리 라벨과 시간에 따른 이벤트를 제공합니다. 또한 56개의 VLM(비전 언어 모델)과 10개의 VFM(비전 기반 모델)을 평가하여 저차원 물리 추론의 현재 한계를 드러냅니다.

- **Performance Highlights**: 대규모 연구 결과, 비전 언어 모델의 물리적 이해는 의외로 제한적임을 확인했습니다. 고급 모델들이 기준을 지배하고 있으나, 대규모 오픈 소스 모델들이 성능 격차를 좁히고 있습니다. NewtPhys의 픽셀 단위 물리 주석을 활용하여, VFM에 대한 Physics Probing 작업을 도입하였고, 대체로 시각적 표현이 향상될수록 성능 또한 향상되는 경향을 보였습니다.



### Formalizing the Binding Problem (https://arxiv.org/abs/2606.03976)
Comments:
          Accepted to ICML 2026

- **What's New**: 이번 연구에서는 세계에 대한 표현이 단순한 특징 정보(예: 무언가가 파란색이다, 무언가가 원형이다)뿐만 아니라 이러한 특징들이 동일한 객체에 속하는지에 대한 정보(즉, 바인딩 정보)를 포함하고 있다는 점을 강조합니다. 우리는 이러한 바인딩 문제(binding problem)를 정보 이론적 접근 방식으로 형식화하고, 모델 표현에서 바인딩 정보를 측정하기 위한 프로빙(probing) 방법을 도입하였습니다.

- **Technical Details**: ViT(Vision Transformer) 아키텍처의 다양한 구성 요소에서 바인딩을 측정하기 위한 실험을 수행하였습니다. 이미지 요약 토큰([CLS])이나 공간 토큰(spatial tokens)과 같은 요소들이 포함된 여러 기존 ViT 모델을 비교하는 과정에서, 기능 공유(feature sharing), 폐색(occlusion), 자연 특징(natural features)과 같은 다양한 바인딩 도전 과제를 다루었습니다.

- **Performance Highlights**: 연구 결과에 따르면, 바인딩은 강력한 시각 인식(visual recognition) 및 추론(reasoning)의 핵심 요소로 기능한다는 것을 보여주었습니다. 특히, 다양한 바인딩 문제에 대한 성능을 평가함으로써 ViT 기반 아키텍처가 바인딩 정보를 얼마나 효율적으로 학습하는지를 분석하였습니다.



### AAD-1: Asymmetric Adversarial Distillation for One-Step Autoregressive Video Generation (https://arxiv.org/abs/2606.03972)
Comments:
          ICML 2026. Project page: \url{this https URL}

- **What's New**: AAD-1은 비대칭적 적대적 증류(framework for Asymmetric Adversarial Distillation) 아키텍처로, 한 단계 오토회귀(autoregressive) 이미지-비디오 생성 분야의 문제를 해결합니다. 기존 방법들이 경험하는 모션 붕괴(motion collapse)와 훈련 불안정성 문제를 극복하기 위해 중요한 두 가지 설계를 제시합니다. 이 프레임워크는 생성기와 판별기(generator and discriminator) 간의 대칭성을 깨고, 판별기는 전체 공간-시간(context)에서 양방향(attends bidirectionally) 정보를 수집하여 비디오 전체에 대한 통합 된 현실(realism) 점수를 생성합니다.

- **Technical Details**: AAD-1의 주요 혁신 중 하나는 비대칭적 판별기(비대칭적 설계) 사용으로, 이것이 모션 붕괴를 감지하고 제거하는 데 효과적입니다. 또한, 훈련 안정화를 위해 분포 매칭(warm-up)을 통한 단계적 훈련을 도입하여 초기 예측을 리얼 데이터와 가까워지도록 부트스트랩합니다. 이를 통해 훈련 초기 단계에서 발생할 수 있는 불안정성을 예방하고, 적대적 증류(adversarial distillation)의 효과를 극대화할 수 있습니다.

- **Performance Highlights**: AAD-1은 VBench에서의 광범위한 실험을 통해 한 단계 오토회귀 비디오 생성 분야에서 최신 기술 수준의 성능(state-of-the-art performance)을 달성했습니다. 이 프레임워크는 비주얼 품질(visual quality)과 모션 충실도(motion fidelity)에서 우수한 성능을 보이며, 이론적 기여가 돋보입니다. 특히, 비디오 전체에 대한 종합적 차별화(hybrid discriminator)와 분포 매칭 기법이 모션 붕괴 문제를 해결하는 데 기여하고 있습니다.



### Video-Mirai: Autoregressive Video Diffusion Models Need Foresigh (https://arxiv.org/abs/2606.03971)
- **What's New**: 이번 연구에서는 Causal video generator가 과거에서 예측해야 한다는 점에 초점을 두고, Video-Mirai라는 새로운 학습 방법을 소개합니다. 기존의 방법은 현재 상태를 설명하는 데 그쳤지만, Video-Mirai는 미래의 정보를 사용하여 현재 상태가 미래 일관성을 유지하는 데 필요한지를 평가하는 대표성 수준(Representation-level) 계획 격차를 해소합니다.

- **Technical Details**: Video-Mirai는 훈련 중에 미래의 세그먼트를 사용하여 현재의 원인 상태를 감독하고, 추론 시에는 과거 정보만을 이용해 작동하도록 설계되었습니다. 이 방법은 현재의 인코더와 예측기를 배제했지만, 원래 구조나 연산 비용(FLOPs)을 변화시키지 않고 과거의 정보만으로 여전히 원인적인 인과관계를 유지합니다.

- **Performance Highlights**: Video-Mirai는 5초 VBench의 Causal-Forcing 기반점수를 83.8에서 84.6으로 향상시켰으며, 30초 롤아웃에서도 주제 일관성이 84.9에서 88.5로 증가했습니다. 이 연구는 Video-Mirai의 사용으로 미래 프레임의 해독 가능성이 증가하고, 전체적인 롤아웃 일관성이 향상됨을 보여줍니다.



### VLESA: Vision-Language Embodied Safety Agent for Human Activity Monitoring (https://arxiv.org/abs/2606.03954)
Comments:
          18 pages, 5 tables, 5 figures

- **What's New**: AI 시스템이 물리적 작업에서 인간을 보조함에 따라 안전 보장이 중요해지고 있습니다. 본 논문에서는 VLESA(Vision-Language Embodied Safety Agent)라는 새로운 프레임워크를 소개하며, 이는 상업적 기준으로 제시된 ASIMOV-2.0 벤치마크에서 해롭고 안전한 행동을 구분할 수 있는 실시간 안전 개입을 가능하게 합니다. VLESA는 인간 활동을 모니터링하고, 위험한 행동을 예상하여 즉각적으로 안전 개입을 수행하는 능력을 갖추고 있습니다.

- **Technical Details**: VLESA는 특정 목표에 기반한 안전 Q 필터와 의도-행동 예측 모듈로 구성되어 있습니다. 이 시스템은 egocentric 비디오를 활용하여 의도를 추론하고, 향후 행동을 예측하며, 안전성을 평가할 수 있습니다. 'EgoSafety'라는 새로운 데이터셋이 도입되었으며, 이는 eogcentric 프레임과 목표 기반 안전 주석을 결합하여 안전한 행동과 위험한 행동을 구분하는 데 사용됩니다.

- **Performance Highlights**: VLESA는 ASIMOV-2.0 벤치마크에서 기존 모델에 비해 더 높은 개입 정확성을 달성했습니다. GRPO(그룹 상대 정책 최적화)를 통해 훈련된 Q 필터는 목표 조건에 따라 행동 안전성을 41 포인트 이상 향상시켰습니다. 이러한 성과는 VLESA가 기존 시스템들보다 더 능동적이고 맥락 의존적인 안전 모니터링을 가능하게 함을 의미합니다.



### Demo2Tutorial: From Human Experience to Multimodal Software Tutorials (https://arxiv.org/abs/2606.03951)
Comments:
          Accepted by CVPR 2026

- **What's New**: 본 논문에서는 Demo2Tutorial이라는 프레임워크를 소개합니다. 이 프레임워크는 화면 녹화(screen recordings) 및 상호작용 로그(interaction logs)를 통해 수집된 인간 경험을 구조화된 멀티모달 튜토리얼로 변환합니다. Demo2Tutorial은 인간의 디지털 환경에서의 절차적 지식을 추출하여, 이를 다시 사용 가능한 형태로 만들고 인간과 컴퓨터-사용 에이전트 모두에게 효율적인 학습을 가능하게 합니다.

- **Technical Details**: Demo2Tutorial 프레임워크는 네 가지 핵심 구성 요소로 이루어져 있습니다: 1) HE-Recorder(인간 경험 기록기)는 화면과 사용자 작업(클릭, 키 입력)을 동시 기록합니다. 2) Action Parser(행동 파서)는 저수준 행동의 의미를 해석합니다. 3) Step Planner(단계 계획자)는 작업을 계층적 그래프 구조로 구성합니다. 4) Tutorial Composer(튜토리얼 작곡자)는 최종 이미지-텍스트 콘텐츠를 생성합니다. 이러한 구성 요소들은 함께 작동해 의미 있는 튜토리얼을 제공합니다.

- **Performance Highlights**: 평가 결과, Demo2Tutorial이 생성한 튜토리얼은 인간이 작성한 튜토리얼을 능가하는 고품질 결과를 보여줍니다. 또한, GUI 에이전트의 계획 및 일반화 능력도 향상시키며, 인간 사용자의 작업 완료 속도를 향상시킵니다. 연구 결과에 따르면, 튜토리얼은 학습 시간을 단축시키고 사용자들에게 강력한 선호도를 나타내며, 이는 인간 및 기계 학습 모두에 대한 효과적인 솔루션을 제공함을 의미합니다.



### Adaptive Causal Alignment for High-Confidence Adversarial Training (https://arxiv.org/abs/2606.03925)
- **What's New**: 본 연구에서는 Inverse Adversarial Training의 한계를 발견하였으며, 높은 신뢰도가 비인과적(background) 상관관계에 과적합(overfitting)됨을 시사합니다. 이를 해결하기 위해 High-Confidence Causally Aligned Training(HICAT)이라는 새로운 프레임워크를 제안하였습니다. HICAT는 세 가지 단계를 통해 작동하며, 시각적 맥락의 유용성을 진단하는 Learnable Background-Bias Estimator(LBBE)를 통합하여 상황에 맞는 보정 기능을 제공합니다.

- **Technical Details**: HICAT는 'Measure-Debias-Align' 파이프라인을 활용하여, spurious correlation bias를 억제하면서 객체 의미(semantic)를 유지하는 정밀한 로짓 수정(surgical logit rectification)을 수행합니다. Adaptive Debiasing 매커니즘이 이러한 과정을 지원하며, Geometrically grounded Foreground Logit Orthogonal Enhancement(FLOE) 손실로 강력한 피쳐 디스엔탱글먼트(feature disentanglement)를 강화합니다. 이 혁신적인 접근 방식은 다앙한 CNN 및 ViT 모델에 걸쳐 성능을 향상시킵니다.

- **Performance Highlights**: CIFAR-10, CIFAR-100 및 ImageNet-1K 데이터셋에 대한 광범위한 실험 결과, HICAT은 기존의 대조군들에 비해 일관된 성능 향상을 보였습니다. 특히, HICAT을 통해 전형적인 clean-robust 트레이드오프를 극복하고, 적대적 공격과의 강건성이 증가하면서도 깨끗한 정확도(clean accuracy)가 개선되었습니다. 이는 다양한 아키텍처에 효과적으로 일반화되는 원칙으로, robust generalization gap도 현저히 감소시켰습니다.



### GARDEN: Gravity-Aligned Reconstruction of Disentangled ENvironments from RGB images (https://arxiv.org/abs/2606.03921)
- **What's New**: 본 논문에서는 GARDEN이라는 새로운 RGB-only 프레임워크를 제안하여 multi-view RGB 관측치를 물리적으로 기반으로 한 장면 요인화(physically-grounded scene factorization)로 변환하고 구조화된 하이브리드 장면 표현을 출력합니다. 기존 시스템들이 단일 표현에 물리적 구조 없이 결합된 장면을 생성한 데 반해, GARDEN은 무기물 물체들을 독립적으로 상호작용할 수 있도록 구조를 제공합니다. 이를 통해 물리적 시뮬레이션과 시각적 사실성을 동시에 유지합니다.

- **Technical Details**: GARDEN의 핵심 아이디어는 중력(gravity)을 보편적인 물리적 기준으로 활용하여 재구성을 개선하는 것입니다. 본론에서는 먼저 다수의 시점에서 얻은 카메라 정보를 사용해 지구의 방향을 파악하여 통합적인 Gravity-View 좌표 프레임을 설정한 후, 6-DoF 자세를 가지는 독립적인 강체 메시에 대한 복원을 수행합니다. 마지막으로 3D 포인트 분류 네트워크를 통해 배경에서 중복된 물체 기하학을 제거하여 더욱 깔끔한 환경을 제공하도록 설계했습니다.

- **Performance Highlights**: 실험 결과 GARDEN은 기존의 CAD 기반 접근법과 비교하여 객체 배치의 신뢰성과 객체 분리 품질을 개선하면서 렌더링-시뮬레이션 효율성을 높였습니다. 특히 처리 시간이 LiteReality와 같은 기존 시스템에 비해 크게 줄어들었으며, 기하적 정확성과 렌더링 사실성이 향상되었습니다. 이러한 성능은 GARDEN이 물리 기반의 요인화를 통해 달성되었습니다.



### Benchmarking Visual State Tracking in Multimodal Video Understanding (https://arxiv.org/abs/2606.03920)
Comments:
          Website: this https URL

- **What's New**: 이 논문에서는 MLLMs(다중모드 대형 언어 모델)의 시각적 상태 추적(Visual State Tracking) 능력을 평가하기 위한 비디오 기반 벤치마크인 VSTAT를 소개합니다. VSTAT는 834개의 클립과 1500개의 질문으로 구성되어 있으며, 이 질문들은 단일 프레임이나 짧은 세그먼트로는 답할 수 없는 형태입니다. 이는 모델이 비디오 스트림을 통해 사건을 지속적으로 인식하고 통합해야 함을 의미합니다. 이 연구는 MLLMs의 성능이 인간보다 현저히 낮다는 점을 강조합니다.

- **Technical Details**: VSTAT는 비디오 스트림과 질문을 입력으로 받아 답을 출력하는 표준 질문-응답 형식을 따릅니다. 이 벤치마크는 반응을 유도하기 위해 전체 비디오 스트림을 처리해야 하며, 이는 키프레임이나 눈에 띄는 순간에 의존할 수 없습니다. 벤치마크는 다양한 인식 과제를 포함하며, 예를 들어 Rubik's cube 작업에서는 모델이 특정 조각을 추적해야 하는 등 여러 인지적 도전을 제공합니다. 비디오 큐레이션은 실제와 시뮬레이션된 환경 모두에서 이루어졌습니다.

- **Performance Highlights**: VSTAT에서 최신 MLLMs는 인간 성능에 훨씬 못 미치는 결과를 보였으며, 답변 우선 기준(baselines)보다 겨우 약간 나은 성과를 기록했습니다. 실험적 분석을 통해 MLLMs의 주요 한계는 연속적인 비디오 스트림 내에서 무관한 사건을 인식하는 시각적 인식 부족임을 밝혔습니다. 최근 기법들은 이러한 성능 격차를 해소하는 데 효과적이지 않았으며, VSTAT에서의 부족한 성과를 해결하는 데 한계를 보였습니다.



### PatchScene: Patch-based Voxel Diffusion for Large-Scale Scene Completion (https://arxiv.org/abs/2606.03915)
Comments:
          10 pages, 5 figures, 5 tables

- **What's New**: PatchScene은 대규모 LiDAR 장면 완성을 위한 새로운 확산(diffusion) 기반 프레임워크로, 글로벌 잠재 표현이나 밀집 voxel 그리드 대신 지역화된 3D 영역 내에서 세밀한 기하 구조를 생성하는 패치(patch) 기반 voxel 확산 패러다임을 채택합니다. 또한, 연속된 프레임에서의 일관된 재구성을 보장하기 위해 오버랩 패치와 인접 프레임을 통합하는 신뢰도 안내(spatio-temporal) 융합 메커니즘을 도입했습니다.

- **Technical Details**: PatchScene은 국소 패치를 기반으로 한 완성을 통해 고해상도의 정밀한 장면 생성을 수행하며, 원형 흐름(Annular-Flow) 확산 전략을 사용하여 LiDAR 스캔의 방사 밀도 패턴을 활용하여 근 거리에서 원거리로 정보를 전파합니다. 이로 인해 공간적으로 무제한의 장면 완성을 가능하게 합니다. 또한, 우리의 방법은 훈련된 모델이 20m 및 50m LiDAR 범위에서 효과적으로 일반화될 수 있는 능력을 지니고 있습니다.

- **Performance Highlights**: SemanticKITTI 벤치마크에서 PatchScene은 모든 표준 메트릭에서 SOTA(최첨단 성능)를 달성하며 기존 접근 방식들보다 기하학적 정확성과 시간적 일관성에서 우수한 성과를 보입니다. 이는 자율 주행 응용 프로그램을 위한 강력한 확장성 및 일반화 능력을 입증합니다. 실제로, 단일 프레임 접근의 흔한 플리커 아티팩트를 유의미하게 억제하여 품질을 대폭 향상시키는 결과를 보여줍니다.



### Bootstrap Your Generator: Unpaired Visual Editing with Flow Matching (https://arxiv.org/abs/2606.03911)
Comments:
          Accepted at ICML 2026. Project page is at this https URL

- **What's New**: 이 논문은 전통적인 데이터 짝을 사용하지 않고 이미지 및 비디오 편집 모델을 훈련하는 새로운 방법인 Bootstrap Your Generator (ByG)를 제안합니다. 기존의 큰 데이터셋에 의존하지 않고도 사전 훈련된 생성 모델의 내재적 지식을 활용하여, 빠르고 효율적으로 변환을 학습할 수 있도록 합니다. 이 접근 방식은 특히 비디오 편집과 같은 데이터 수집이 비용적으로 어려운 상황에서 유용하게 사용될 수 있습니다.

- **Technical Details**: 이 방법은 'cycle consistency'와 지시 사항 추적 신호를 결합하여 모델이 출력을 조정하고 원본을 보존하도록 훈련합니다. 이를 통해, 노이즈가 포함된 상태에서 훈련할 수 있도록 발산 손실을 조정하는 그라디언트-라우팅 메커니즘을 도입합니다. 이 기술은 Straight-Through Estimation (STE)을 기반으로 하며, 훈련 과정에서 유효한 입력 없이도 학습을 지속할 수 있게 합니다.

- **Performance Highlights**: ByG는 사용자 선호도 조사에서 수백만 짝을 훈련한 감독 기반 모델보다 75% 이상의 승리 비율을 기록했습니다. 본 연구는 긴 꼬리 스타일 편집에서 감독 방법 및 제로샷 방법을 능가하며, 보지 못한 스타일에도 잘 일반화되는 성능을 보입니다. 또한, 각 구성 요소의 필요성을 검증하는 세부적인 분석 결과를 제공합니다.



### SparseStreet: Sparse Gaussian Splatting for Real-Time Street Scene Simulation (https://arxiv.org/abs/2606.03909)
- **What's New**: 이번 논문에서는 SparseStreet라는 새로운 압축 프레임워크를 제안하여, 3D Gaussian Splatting 기법을 사용하는 도로 장면 재구성에서 발생하는 대량의 Gaussian primitive의 저장 비용과 느린 렌더링 속도를 해결하고자 합니다. 이 프레임워크는 동적인 객체 표현의 정확도를 유지하면서도 정적 배경 지역의 중복성을 줄이는 데 초점을 맞추고 있습니다. 이를 통해 최대 80% 압축 비율을 달성하면서도 시각적 품질 저하를 최소화할 수 있습니다.

- **Technical Details**: SparseStreet는 노드 기반의 학습 가능한 가지치기(pruning) 전략을 활용하여, 시각적으로 중요한 지역을 유지하며 기여도가 낮은 Gaussian primitive를 체계적으로 제거합니다. 장면 표현이 안정화된 후에는 정적 지역의 중복성을 추가적으로 줄이는 배경 압축(background compression)을 적용합니다. 이로 인해 복잡한 도로 장면을 보다 효율적으로 표현할 수 있는 장점이 있습니다.

- **Performance Highlights**: Waymo와 nuScenes 데이터셋에서 수행한 실험을 통해, SparseStreet는 기존의 장면 그래프 기반 재구성 방법과 통합하여 우수한 압축 성능을 보여주었습니다. Gaussian primitive 수를 80%까지 줄이는 동시에 동적 객체의 시각적 충실도를 유지하고 렌더링 속도를 2배 향상시킬 수 있음을 입증했습니다. 이는 자원 효율적인 고충실도의 동적 장면 재구성을 가능하게 합니다.



### An Attention-Based Denoising Model for Diffusion Weighted Imaging (https://arxiv.org/abs/2606.03903)
- **What's New**: 이 논문에서는 DWI(확산가중영상) 복원에서 발생하는 Rician 노이즈를 효과적으로 처리하기 위한 주의 기반(Attention-driven) denoising 프레임워크를 제안합니다. 이 프레임워크는 계층적 Swin Transformer 윈도우 주의(attention)와 변환기 기반(Transformer-based) 다차원 게이트 정제(multi-dimensional gated refinement)를 통합하여 DWI 복원 과정을 개선합니다. 노이즈 수준을 명시적으로 조정하여 다양한 손상 수준에서 비균질 노이즈를 적응적으로 억제할 수 있는 능력을 제공합니다.

- **Technical Details**: 제안된 모델은 계층적 셀프 어텐션(hierarchical self-attention)과 채널 적응형(gated) 정제를 통합하여 DWI의 Rician 노이즈 억제를 위한 강력한 구조를 제공합니다. 초기 3x3 합성곱 층(convolution layer)을 통해 노이즈가 있는 영상을 64차원 특성 공간으로 변환한 후, 여러 층의 Swin Transformer 블록으로 윈도우 기법을 적용하여 계산 복잡도를 줄이며 하위 샘플링된 데이터를 적응적으로 처리합니다. 노이즈 잔여 예측(residual reconstruction)과 같은 방법론은 상세 보존을 위한 중요한 요소로 작용합니다.

- **Performance Highlights**: 실험 결과, 이 모델은 1%에서 15%의 다양한 노이즈 수준에서 평균 PSNR(34.69 dB)과 SSIM(0.8539)을 달성하며 우수한 복원 성능을 보여주었습니다. 노이즈가 심한 조건에서도 안정적인 성능을 유지하며, 다양한 수준의 Rician 노이즈 조건하에서도 효과적으로 작동하는 것으로 나타났습니다. 이러한 결과들은 구조적 보존을 유지하면서 전체 시스템의 강인성을 향상시키는 데 기여함을 보여주고 있습니다.



### Electromagnetic Navigation for Femoral Osteotomy Using High-Accuracy X-ray-to-CT Registration (https://arxiv.org/abs/2606.03893)
Comments:
          Will be published in the International Journal of Computer Assisted Radiology and Surgery

- **What's New**: 이 논문에서는 전자기 추적 (EMT) 기반의 내비게이션 시스템을 새롭게 제안하여 대퇴골 교정 골절술 중의 해부학적 정확성을 개선하고 방사선 노출을 최소화하는 방법을 소개하고 있습니다. 이 시스템은 CT 기반의 수술 계획과 인체 끼리 세팅 (C-arm) 캘리브레이션을 통합하여 방사선 촬영 없이 실시간 내비게이션을 제공합니다. 또한, 임상 시험 결과에서 기존의 프리핸드 (free-hand) 기법보다 우수한 성능을 보여주며, 환자 맞춤형 기구 (PSI)와 유사한 정확성을 유지합니다.

- **Technical Details**: 논문의 방법론에서는 Flepp 등이 제안한 고도로 자동화된 X-ray-CT 등록 방법을 EMT와 통합하여 사용합니다. 이 과정은 초기 등록 단계와 실시간 내비게이션 단계로 나뉘어 진행되며, 두 개의 Kirschner 와이어 (K-wire)만을 사용하여 수술적인 부담을 최소화합니다. EMT 센서를 활용하여 수술 중 뼈 조각의 실시간 추적을 가능하게 하며, C-arm 시스템을 통한 방사선 촬영으로 필요한 정밀도를 확보합니다.

- **Performance Highlights**: 실험 결과, 제안된 EMT 내비게이션 시스템이 프리핸드 기법보다 총 각도 오류에서 유의미한 향상을 보였습니다 (p=0.031). EMT 가이드 방식을 사용한 모든 시행에서 5° 임상 기준을 초과하지 않는 성과를 보였으며, PSI와의 통계적 동등성을 달성하였습니다. 이로 인해 최소한의 방사선 노출을 통해 프리핸드 및 PSI의 정확성을 유지하기 위한 후속 임상 검증의 필요성이 강조됩니다.



### OVO-S-Bench: A Hierarchical Benchmark for Streaming Spatial Intelligence in Multimodal LLMs (https://arxiv.org/abs/2606.03890)
Comments:
          48 pages, 12 figures, 15 tables. Project page: this https URL

- **What's New**: OVO-S-Bench는 전통적인 비디오 기반 벤치마크들의 한계를 극복하고 지속적인 egocentric(자아 중심) 비디오를 통해 공간 지능을 평가하는 새로운 벤치마크입니다. 1,680개의 질문과 348개의 소스 비디오로 구성되며, 이는 전혀 사전기반이 아닌 새로운 접근법입니다. 질문들은 4가지 추상화 수준으로 나뉘어 있으며, 이는 모델이 단일 이미지가 아닌 연속적인 영상 스트림에서 공간 구조를 이해해야 함을 요구합니다.

- **Technical Details**: OVO-S-Bench는 12명의 훈련된 주석자들이 804시간 이상을 소비하여 작성한 질문들이 포함되어 있습니다. 각 질문은 쿼리 시간과 증거 간격을 가지고 있으며, 평가 시 모델은 쿼리 이전의 프리픽스만을 보게 되어 온라인 에이전트를 시뮬레이션합니다. 이 벤치마크는 공간 추론의 4단계(egocentric perception, spatiotemporal context tracking, spatial simulation and reasoning, allocentric mapping)로 나뉘어, 모델이 요구하는 증거의 지속성과 통합을 조명합니다.

- **Performance Highlights**: 38개의 시스템을 평가한 결과, allocentric mapping(할로센트릭 매핑)에서 가장 큰 병목 현상을 발견했습니다. Gemini-3.1-Pro는 인간 전문가보다 27점 뒤떨어진 59.2점을 기록했으며, 이는 overall performance에서 명확하게 나타납니다. 또한, 체인 오브 사고(chain-of-thought) reasoning이 cross-frame integration에는 도움이 되지만, 현재의 인식에는 악영향을 미친다는 점도 확인되었습니다.



### CoralBay: A Self-Supervised CT Foundation Mod (https://arxiv.org/abs/2606.03888)
- **What's New**: 본 논문에서는 3D 의료 이미지 처리에 적합한 새로운 자가 지도 학습 프레임워크인 CoralBay를 소개합니다. CoralBay는 DINO의 원리를 확장하여, 계층적 3D Swin Transformer를 사용하고, 다중 스케일 특징을 결합하여 풍부한 공간 표현을 학습할 수 있도록 설계되었습니다. 이를 통해 의료 영상의 다방면에 걸쳐 효과적으로 전이할 수 있는 능력을 보여줍니다.

- **Technical Details**: CoralBay는 CT 스캔과 같은 3D 의료 볼륨 데이터를 원활하게 처리하기 위해 DINO의 자가 증류 원칙을 사용합니다. 이 프레임워크는 3D 적응 평균 풀링과 계층적 특성 계층을 도입하여, 해부학적인 맥락 및 세부 구조를 효과적으로 학습할 수 있도록 합니다. 모델 학습 과정에서는 다양한 Hounsfield Unit (HU) 창을 활용하여 재구성 아티팩트에 대해 강건성을 강화합니다.

- **Performance Highlights**: CoralBay는 다양한 해부학적 목표를 가진 방사선학적 작업에서 뛰어난 성능을 입증하며, 여러 데이터셋을 통합한 공개 3D 방사선학 리더보드를 통해 평가할 수 있는 기준을 제공합니다. 이 프레임워크는 의학 이미징의 다양한 분야에 걸쳐 일관되고 강력한 결과를 나타내며, 고해상도 로컬 세부 사항을 효과적으로 학습하여 의료 이미징에서의 다양성을 잘 다룰 수 있습니다.



### Beyond Encoder Accumulation: Measuring Encoder Roles in Multi-Encoder VLMs (https://arxiv.org/abs/2606.03879)
- **What's New**: 최근 연구에서는 다양한 비전 인코더들을 통합하여 훈련하는 것이 LVLMs(대형 비전-언어 모델)의 성능 향상에 중대한 영향을 미친다고 밝혔습니다. 연구자들은 31개의 서로 다른 인코더 조합을 재훈련하여 인코더의 상호작용을 면밀히 분석하였습니다. 이를 통해 기존 연구에서 관찰되었던 인코더의 기여도에 대한 잘못된 랭킹이 드러났습니다.

- **Technical Details**: 이 연구에서는 인코더의 기여를 두 가지 축인 수용력(Capacity)과 필요성(Necessity)으로 분解하였습니다. 수용력은 각 인코더가 혼자서 도달할 수 있는 점수이고, 필요성은 전체 풀에서 제거했을 때 점수가 얼마나 떨어지는지를 나타냅니다. 연구 결과에 따르면, 높은 수용력의 인코더를 짝지어 사용하는 것보다, 적절한 보완 인코더와 짝 지어 사용하는 것이 더 유리하다는 사실이 드러났습니다.

- **Performance Highlights**: 연구에서는 효율적인 인코딩 별 효과적인 순위가 고정된 파라미터 수에서 잔여 점수 변동성을 설명한다고 밝혔습니다. 강력한 인코더 조합은 공동 훈련 중에서도 생존하는 앵커와 그에 따라 확장되는 보완 성질을 결합하여 더 우수한 최적화 결과를 가져오는 것으로 나타났습니다. 전반적으로, 이 연구는 다중 인코더 LVLM 설계의 방법론적 공백을 드러내며, 이를 해소하기 위한 구체적인 방법론을 제공합니다.



### MLP Splatting: Object-Centric Neural Fields (https://arxiv.org/abs/2606.03877)
- **What's New**: MLP-Splatting은 장면 분해를 위한 표현적 light-field primitive를 도입하여 photorealistic novel-view synthesis를 제공하는 방법입니다. 이 방법은 각 primitive를 독립적인 compact MLP로 모델링하며, 공간적으로 국소화된 지원을 통해 방사선(radiance) 및 불투명도(opacity)를 예측합니다. 기존의 Gaussian primitive나 단일 전역 방사선 필드와는 달리, MLP-Splatting은 더 높은 표현 능력을 가지면서도 공간적으로 국소화된 특성을 유지합니다.

- **Technical Details**: MLP-Splatting는 RGB 감시(RGB supervision)를 사용하여 개별 MLP 기반 primitive를 공간적으로 중요한 개체의 부분에 할당합니다. 이러한 primitive는 장면의 라디언스 필드를 형성하고, 서로 매개변수를 공유하지 않으며 무한한 표현 능력을 가지고 있습니다. 이 방법은 ray-primitive 상호작용에 대한 효율적인 sparse volumetric compositing을 통해 렌더링이 이루어집니다.

- **Performance Highlights**: MLP-Splatting는 최신 방법과 비교하여 메모리 사용량을 1/15로 줄이고 렌더링 속도를 3배로 향상시킵니다. 이로 인해 복잡한 장면 조작이 가능해지며, 세그멘테이션 마스크를 사용하지 않고도 상호작용이 가능합니다. 실험 결과, MLP-Splatting의 성능은 Replica 및 ScanNet 데이터셋에서 우수한 렌더링 품질과 경쟁력 있는 의미적 결과를 도출했습니다.



### Seg2Track++: Probabilistic Track Validation and Data Association for Multi-Object Tracking and Segmentation (https://arxiv.org/abs/2606.03875)
- **What's New**: 해당 논문에서는 Seg2Track++라는 새로운 MOTS(Multi-Object Tracking and Segmentation) 프레임워크를 제안합니다. 이 프레임워크는 기존 segmentation 모델인 SAM2를 통합하고, 새로운 track 관리 모듈을 추가하여 시간 일관성을 강화한 제로샷(zero-shot) MOTS를 수행합니다. Mask Centroid Distance (MCD)와 Confidence-Aware Cost Modulation (CCM)을 활용해 트랙을 연결하며, Probabilistic Track Validation (PTV) 기법은 Bernoulli 필터를 사용하여 트랙 존재를 검증하고 고스트 트랙을 억제합니다.

- **Technical Details**: Seg2Track++는 객체 탐지, 분할 및 추적을 통해 여러 객체를 다루는 MOTS 문제를 해결합니다. 입력 영상의 각 프레임에서 객체의 이진 마스크와 신뢰도 점수를 포함하는 K개의 객체 궤적을 예측하는 것을 목표로 합니다. 이를 위해 사전 학습된 인스턴스 segmentation 모델을 기반으로 하여 SAM2와 연계해 트랙 관리 기능을 수행하는 세 가지 주요 구성 요소를 통합합니다.

- **Performance Highlights**: KITTI MOTS 벤치마크에서 Seg2Track++는 기존 Seg2Track-SAM2 모델에 비해 추적의 견고성 및 신원 보존이 향상되었습니다. 이 연구는 고스트 트랙 억제 및 잘못된 연결을 통해 false-positive 전파를 줄이는 효과를 입증하였으며, MOTS 시스템의 전반적인 성능을 개선하는 데 기여하고 있습니다.



### DyaPlex: Full-Duplex Speech-Motion Model for Dyadic Interaction (https://arxiv.org/abs/2606.03874)
Comments:
          Project page: this https URL

- **What's New**: DyaPlex는 다이아딕(dyadic) 상호작용을 위한 스트리밍 형태의 전이중(full-duplex) 음성 및 동작 모델입니다. 이 모델은 음성과 신체 움직임을 동시에 인식하고 생성할 수 있어, 자연스러운 인간 대 인간의 소통을 Capture(캡쳐)합니다. DyaPlex는 기초 전이중 음성 모델의 강력한 선행지식을 활용하고 새로운 동작 경로를 통합하여 완전히 동기화된 다중 모드 상호작용을 가능합니다.

- **Technical Details**: DyaPlex는 두 개의 타워 구조를 가진 Transformer 아키텍처를 설계하여 음성과 동작의 깊은 결합을 이루며, 시간적으로 정렬된 speech-motion RoPE를 통해 교차 주의를 유도합니다. 이 모델은 4,000시간의 Seamless Interaction 데이터셋을 통해 학습되어 다양한 인간의 협조를 Capture하며, 자연스러운 대화의 고유한 특성을 잘 반영합니다. DyaPlex는 동시에 여러 모드의 Cue(큐)를 처리하며, 모든 행동은 서로 긴밀하게 연결되어 정의됩니다.

- **Performance Highlights**: DyaPlex는 기존 단일 및 다이아딕 인간 상호작용 벤치마크에서 신기록을 세우며, 뛰어난 사실감과 사회적 일관성을 보여줍니다. 평가 결과, 이 모델은 사회 로봇 및 확장 가능한 합성 데이터 생성과 같은 새로운 응용 프로그램을 가능하게 합니다. 또한, DyaPlex는 음성과 동작 간의 정확한 동기화를 통해 무리 없이 스트리밍 생성이 이루어지는 것을 보장합니다.



### Visual Instruction Tuning Aligns Modalities through Abstraction (https://arxiv.org/abs/2606.03871)
- **What's New**: 이번 연구에서는 시각적 기능이 사전 훈련된 대형 언어 모델(LLM)에 어떻게 통합되는지를 탐구합니다. 우리는 다양한 비전-언어 아키텍처에서 시각적 특징이 LLM의 중간 의미 층에 직접적으로 포함됨을 보여줍니다. 이러한 방식은 초기 단일 모드 처리 층을 우회하여, 멀티모달 작업에서 중간층이 중요한 역할을 했음을 밝혀냈습니다.

- **Technical Details**: 연구에서 사용한 VLM 모델에는 LLaVA, OneVision, InternVL2, Cambrian과 같은 시각 이해 작업을 위한 아키텍처들이 포함되어 있습니다. 두 개의 주요 훈련 단계로 나뉘어 있으며, 첫 번째 단계는 시각적 기능을 LLM에 맞추는 커넥터 가중치를 학습합니다. 두 번째 단계에서는 LLM과 시각 인코더의 내부 표현을 멀티모달 이해를 지원하도록 재구성합니다.

- **Performance Highlights**: 중간층의 역할은 여러 벤치마크에서 확인되었으며, 해당 층을 튜닝할 경우 전반적인 성능 손실을 최소화하면서도 고성능을 유지하는 효과를 나타냈습니다. 또한, 우리의 실험 결과는 이러한 중간층이 LLM의 내부 추상화 엔진의 재목적화에 의해 구동되는 국소적 현상임을 제안합니다.



### Unified Video-Action Joint Denoising for Dexterous Action and Data Generation (https://arxiv.org/abs/2606.03868)
Comments:
          9 pages, 5 figures

- **What's New**: 최근의 영상 기반 행동 모델들은 로봇 행동을 실행 가능한 방식으로 맞추기 위해 넓은 시각-역학 우선 순위를 활용하고 있습니다. 본 연구에서는 이러한 정렬을 분포적인 관점에서 재검토하고, 기존 모델들과는 달리 다양한 조건 체계 아래에서 상호작용 비디오와 실행 가능한 손 경로의 공동 공간을 모델링하여 더 넓은 분포를 유지합니다. 이를 통해 Donk라는 통합 비디오-행동 노이즈 제거 모델을 제안합니다.

- **Technical Details**: Donk는 비디오 확산 변환기(video diffusion transformer)에 기반하여 비디오 토큰과 행동 토큰을 흐름-일치(flow-matching) 패러다임 하에 공동으로 노이즈 제거합니다. 행동은 MANO 손 매개변수의 시퀀스로 표현되어, 세밀한 손 동작을 구조적으로 나타냅니다. Donk는 이미지 조건 하에서 TI2VA 정책으로 기능하며, 텍스트 전용 입력에서는 T2VA 데이터 엔진으로 작동하여 언어 지시로부터 상호작용 비디오와 동기화된 손 행동 궤적을 생성합니다.

- **Performance Highlights**: Donk는 TI2VA 정책으로서 OakInk 벤치마크에서 손 RMSE와 손목 궤적 오류에서 최고 성능을 기록하며, 0.2992의 LPIPS를 유지하여 양질의 비디오 충실도를 보입니다. 또한, T2VA 데이터 엔진으로서 공간적으로 정렬되고 시간적으로 동기화된 MANO 손 동작을 생성하면서 좋은 비디오 품질을 유지합니다.



### Where Do We (Not) Need Temporal Context in Low-Resource Video Task Adaptation? (https://arxiv.org/abs/2606.03837)
- **What's New**: 본 논문에서는 파라미터 효율적인 미세 조정(PEFT) 및 프로빙(probing) 방법을 체계적으로 연구하여 비디오 이해를 위한 모델 적응 전략을 평가합니다. 기존 연구에서는 이미지로 학습된 모델을 비디오에 적응시키거나, 비디오로 학습된 모델에 대해 기존의 PEFT 방법을 적용한 사례가 많았으나, 두 가지 방법의 직접적인 비교는 부족했습니다. 이 연구는 제한된 데이터 환경에서 특히 파라미터 효율성을 극대화할 수 있는 다양한 비디오 태스크에 대해 새로운 통찰력을 제공합니다.

- **Technical Details**: 이 연구에서는 외형 중심, 동작 중심 및 공간 밀집 설정에서 모델 적응 방법을 평가합니다. 또한, PEFT 모듈, 프로브(probe), 그리고 백본(backbone) 간에 시간적 컨텍스트를 분배하여 다른 프레임 수에서 작동하도록 하여 성능과 효율성 간의 균형을 연구합니다. 주요 발견 사항 중 하나는 표준 PEFT 방법이 이미지에서 비디오로의 PEFT 접근 방식보다 더 우수하다는 것입니다.

- **Performance Highlights**: 연구 결과는 시간적 모델링이 백본에 의해 주도되며, 백본의 성능이 우세함을 보여줍니다. 비디오 백본의 경우, 주의 기반 적응이 동작 중심 태스크에서 우수한 성능을 보이고, MLP 기반 적응이 공간 예측에 적합하다는 점이 특징입니다. 제한된 데이터 환경에서의 적응 방법을 평가하는 최초의 종합적인 기준을 제시하였으며, 비디오의 파라미터 효율적인 모델 적응을 이해하고 개선하기 위한 새로운 관점을 제시합니다.



### Conditional Latent Diffusion Model with Fourier-based Motion Modelling for Virtual Population Synthesis (https://arxiv.org/abs/2606.03827)
Comments:
          This work has been early accepted by International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI) 2026

- **What's New**: 이번 연구에서는 4D F-MeshLDM이라는 조건부 생성 프레임워크를 제안합니다. 이 모델은 심장 메쉬 생성을 보다 정확하고 일관되게 수행할 수 있도록 설계되었습니다. 특히, 심장 주기에 대한 수학적으로 정확한 주기 일관성을 보장합니다. 또, 특이한 점은 Fourier 계수를 사용하여 심장 동작을 효과적으로 매개화한다는 점입니다.

- **Technical Details**: 제안된 4D F-MeshLDM은 변형된 Variational Autoencoder(VAE)를 통해 각 메쉬를 인코딩하고, 자가 인식된 주기적 운동 경로를 구현하기 위해 절단된 Fourier 시리즈를 사용합니다. 이 모델은 임상 변수에 기반한 조건을 통해 동작 패턴을 조절할 수 있으며, 이로 인해 고도의 주기성을 유지하게 됩니다. 또한, Denoising Diffusion Probabilistic Model(DDPM)을 활용하여 런 생성과정을 구현합니다.

- **Performance Highlights**: 실험 결과, 4D F-MeshLDM은 기존의 최첨단 모델들보다 해부학적 정확도에서 월등한 성능을 보였습니다. 심장 메쉬가 생산하는 사이클 닫힘 오류는 거의 제로 상태에 도달하였으며, 임상적 기능 지수도 정확하게 보존되었습니다. 이는 이 프레임워크가 신뢰할 수 있는 인 실리코(in-silico) 심장 시험을 수행할 유망한 가능성을 보여줍니다.



### TeX-1500: A Paired Real-World LWIR Hyperspectral Dataset and Benchmark for Temperature-Emissivity-Texture Decomposition (https://arxiv.org/abs/2606.03806)
- **What's New**: TeX-1500은 HSI(고스펙트럼 이미지)와 TeX(온도, 방출율, 텍스처) 쌍 데이터셋으로, 1,522개의 실제 장면 샘플이 포함되어 있습니다. 이 데이터셋은 다양한 시간, 장소, 환경에서 수집된 데이터로 구성되어 있으며, HSI에서 TeX로의 지도 학습을 지원하는 최초의 벤치마크로 제시되었습니다. 또한, TeX-1500은 기계 학습 모델이 환경 변화에 맞춰 학습할 수 있도록 다양한 환경 조건을 반영한 좋은 구축 프로토콜을 제공합니다.

- **Technical Details**: TeX-1500에는 DARPA IH 푸시브룸 이미지와 FTIR(푸리에 변환 적외선 분광기) 데이터가 결합되어 있어, 다양한 온도 및 방출율 그대로의 정보를 제공하도록 설계되었습니다. 각 샘플은 보정된 유효 대역 방사력 큐브와 정렬된 온도, 방출율, 텍스처 감독 정보로 구성되어 있습니다. TeX-UNet이라는 간단한 하이퍼스펙트럴 이미지를 TeX 필드로 매핑하는 기본 모델도 제공되어, 데이터 기반의 물리적 속성 인식이 가능합니다.

- **Performance Highlights**: DARPA IH 장면과 FTIR 장면에 대한 실험에서 TeX-1500은 사용 가능한 짝지어진 감독이 있으며, 데이터 기반의 물리적 속성 중심의 열 인식을위한 측정 가능한 기준을 제공합니다. 제안된 TeX-UNet 모델은 TeX-1500의 학습 가능성을 보여줍니다. 이 연구는 기존 TeX 파이프라인의 한계를 극복하고, 효율적이며 확장 가능한 학습 및 평가 프로토콜 제공의 단초가 될 것입니다.



### Template Collapse and Information-Theoretic Limits in Camera rPPG Pulse Morphology Restoration (https://arxiv.org/abs/2606.03802)
- **What's New**: 이 연구는 소비자 카메라를 통한 원격 광측정법(rPPG)에서 개별적인 동맥 형태를 복원할 수 있는 가능성을 처음으로 탐구했습니다. 연구진은 16개의 아키텍처를 활용하여 153명의 피험자 데이터셋에서 교차 주제 Pearson r을 도입하여 특정 주제의 복원이 가능한지를 평가했습니다. 결과적으로, 어떤 아키텍처도 주제별 형태를 복원할 수 없음을 발견하였고, 이는 소비자 카메라가 개별 동맥 특성을 인코딩하지 못함을 의미합니다.

- **Technical Details**: 이 연구는 소비자 카메라에서 얻은 rPPG 신호가 동맥 경직도 생체 마커를 복원하기 위한 충분한 정보를 유지하고 있는지를 분석합니다. 연구진은 여러 아키텍처의 성능을 비교하고, 원거리 측정에서의 신호 처리 제한을 진단합니다. 특히, 파형의 주기적인 형태를 복원하기 위한 정보 이론적 한계를 설정했으며, 실험에서 강력한 정량적 지표인 cross-subject r을 도입하여 템플릿 붕괴(template collapse)를 감지할 수 있는 방법을 모색했습니다.

- **Performance Highlights**: 가장 우수한 결과는 Supervised Contrastive(SupCon) 접근법을 통해 도출되었으며, 이는 r의 범위가 0.773에서 0.9999까지 나타났습니다. 그러나 모든 아키텍처가 주제별 구별 가능한 형태 구조를 추출하지 못함이 확인되었습니다. 또한 VAE 복원기가 rPPG 입력에서 부족한 집단 수준의 조화 콘텐츠를 회복하는 것으로 나타났으며, 이는 기존 메트릭이 개별 주제를 충분히 설명하지 못함을 보여주는 중요한 결과입니다.



### Beyond Compression: Quantifying Spectral Accessibility in Vision Representations (https://arxiv.org/abs/2606.03795)
- **What's New**: 이 연구는 시각-언어 모델의 변환이 시각 정보의 구조를 어떻게 변화시키는지를 조사합니다. 특히 Residual Spectral Loss (RSL)을 도입하여 모델 표현에서의 변화를 정량화하며, 기존의 차원 축소 효과를 초월하는지에 대해 분석합니다. CLIP과 DINOv2 모델 간의 실험을 통해, 이러한 변환이 주파수 접근성에 일관된 변화를 초래함을 보여줍니다.

- **Technical Details**: 연구는 선형 복구 가능성(linear recoverability)을 측정하여, 모델 표현이 원본 이미지의 특정 주파수 성분을 얼마나 복구할 수 있는지 분석합니다. 다양한 시각 인코더 아키텍처에서 실험을 수행하며, CLIP과 DINOv2에서 중간 층에서 가장 큰 접근성 손실을 보이는 것을 발견하였습니다. 이 연구는 선형 프로브(linear probes)를 통해 주파수 도메인에서의 접근성 변화를 평가합니다.

- **Performance Highlights**: 실험 결과, 모든 모델에서 주파수 의존적인 접근성 변화가 일관되게 나타났습니다. CLIP의 경우 변환이 스펙트럼적으로 중립적이며, DINOv2는 스펙트럼을 가로막는 구조적 손실을 초래합니다. 이는 현대 시각 인코더에서 중간 층과 풀링 메커니즘이 스펙트럼 변환의 주요 원동력임을 나타냅니다.



### Training-Free Multi-Concept LoRA Composition with Prompt-Aware Weighting (https://arxiv.org/abs/2606.03792)
Comments:
          Accepted at IEEE FG 2026

- **What's New**: 본 논문에서는 기존의 LoRA(저_rank 적응) 접근 방식을 개선하여 다중 개념(customization) 개인화를 지원하는 간단하면서도 효과적인 방법을 제안합니다. 제안된 방법은 여러 LoRA 모듈의 출력을 최적 결합함으로써 개별 개념의 시각적 품질과 참조 이미지에 대한 충실도를 함께 개선합니다. W-Switch와 W-Composite이라는 두 가지 새로운 방법을 통해 각 개념의 상대적 중요성을 탐지하여 문맥을 이해하며, 새로운 평가 기법도 도입하였습니다.

- **Technical Details**: 저자들은 W-Switch 및 W-Composite이라는 두 가지 방법을 제안하여 각각의 LoRA 모듈의 상대적 중요성을 문장에서 유도된 토큰에 기반하여 가중치 조정하는 방식을 채택했습니다. 이를 통해 다중 개념 간의 상호작용(interference) 문제를 해결하였고, 생성된 이미지의 진위성과 정체성 보존(identity preservation)을 평가하는 새로운 이미지 기반 유사도 평가 프레임워크를 도입했습니다. 이 방식은 실세계 참조 이미지와 생성된 이미지 간의 비교를 통해 성능을 측정합니다.

- **Performance Highlights**: 제안된 접근 방식은 ComposLoRA 테스트베드에서 기존 최첨단 방법들에 비해 비주얼 품질, 정체성 보존 및 조합성(compositionality) 등에서 일관된 개선을 보여주었습니다. 인간 평가에서는 시각적 품질과 정체성 보존 지표에서 더 높은 선호도를 기록하여, 정성적 평가와 정량적 메트릭 모두에서 신뢰성을 입증하였습니다. 이러한 성과는 최근 도입된 새로운 이미지 기반 메트릭과도 일치하는 결과입니다.



### SLU-2K: A Question-Based Benchmark for Semantic Evaluation of Sign Language Translation (https://arxiv.org/abs/2606.03788)
- **What's New**: 이번 연구는 수화 번역(Sign Language Translation, SLT)에서 의미론적 이해(Sign Language Understanding, SLU)로 초점을 옮기고, 의미 보존을 평가하는 새로운 접근 방식을 제안합니다. 이는 SLT 시스템을 보조 기술(assistive technology)로 통합하기 위한 최종적인 목표와도 일맥상통합니다. 특히, 연구자는 PHOENIX-2014T 및 CSL-Daily 데이터셋을 기반으로 하는 2,350개의 비디오 질문-답변 쌍을 포함한 SLU-2K 데이터셋을 제안하여 의미적 요소의 회복 능력을 평가합니다.

- **Technical Details**: SLU-2K 데이터셋은 7가지 카테고리인 동작(actions), 위치(locations), 숫자(numbers), 객체(objects), 사람(people), 시간(time), 날씨(weather conditions)로 구성된 폐쇄형 질문-답변 쌍으로 이루어져 있습니다. 연구진은 질문을 생성하기 위해 신뢰할 수 있는 대형 언어 모델(Large Language Model, LLM)을 사용하고, 다단계 필터링 과정을 통해 모호한 항목을 제거하여 최종 질문을 검증합니다. 각 질문은 번역의 정보가 충분히 포함되어 있는지를 평가하기 위해 이해 구조(comprehension task)로 Frame됩니다.

- **Performance Highlights**: 평가 결과에 따르면, 다중 모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)은 거의 무작위(random) 성능에 가까운 결과를 보였습니다. 또한, 현재의 SLT 시스템들은 사실적 정확도를 유지하는 데 어려움이 있으며, MMSTL 시스템의 경우 75.2%의 의미 정확도를 보였으나, CSL-Daily 데이터셋에서는 단지 56.7%로 낮아지는 경향을 보여줍니다. 이러한 결과는 기존의 표면적 번역 품질 평가가 의미 이해와는 다르다는 점을 명확히 하고 있습니다.



### AmbientEye: A Dataset for Pupil Segmentation under Natural Ambient Infrared Illumination (https://arxiv.org/abs/2606.03774)
Comments:
          12 pages, 7 figures

- **What's New**: 이 논문은 패시브 IR 카메라만을 사용하여 실외 환경에서 신뢰할 수 있는 동공(dilation) 탐지를 가능하게 할 수 있는지를 조사합니다. 이를 위해 연구팀은 AmbientEye라는 방대한 데이터셋을 소개하며, 이는 35명의 참가자로부터 수집된 2,606,225개의 눈 이미지를 포함합니다. 이 데이터셋은 자연 태양광 하에서 촬영되었으며, 기존의 IR 조명이 필요한 설정과 차별화된 실외 조건을 제시합니다.

- **Technical Details**: AmbientEye 데이터셋은 두 가지 다른 태양 방향 조건 하에서 두 개의 오프 축 카메라 구성으로 촬영되었습니다. 각 이미지는 SAM2 자동 분할(segmentation) 모델을 사용해 고품질로 주석이 달리며, 후에 인간 주석자에 의해 검증됩니다. 논문에서는 주어진 데이터셋에서 최첨단 동공 분할(pupil segmentation) 알고리즘의 성능을 벤치마킹하며, 이를 기존의 IR 조명 하에서 수집된 데이터셋과 비교합니다.

- **Performance Highlights**: 결과에 따르면, 통제된 IR 데이터셋에서 동공 분할 성능이 0.928에서 AmbientEye 데이터셋에서는 0.767로 크게 떨어졌습니다. 이 성능 격차는 환경광 조건에서의 동공 탐지의 어려움을 강조하며, AmbientEye를 새로운 기준으로 설정하여 저전력 소비의 눈 추적 방법에 대한 연구 방향을 제시합니다.



### Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models (https://arxiv.org/abs/2606.03748)
Comments:
          31 pages, 8 figures

- **What's New**: 본 연구는 Ultralytics YOLO26을 제시하며, 여러 YOLO 모델의 한계를 극복하기 위해 통합된 실시간 비전 모델 패밀리를 개발했습니다. YOLO26은 NMS(Non-Maximum Suppression)가 필요 없는 이중 헤드 설계를 채택하여, 더 가벼운 회귀 헤드와 자유로운 회귀 범위를 제공합니다. 이 모델은 MuSGD와 Progressive Loss, STAL을 결합한 교육 파이프라인을 통해 훈련을 최적화하고, 작은 객체에 대한 완전한 라벨 할당을 보장합니다.

- **Technical Details**: YOLO26의 구조는 두 개의 헤드 설계로 구성되어 있어 NMS가 없는 전방향 추론을 가능하게 하고, DFL(Distribution Focal Loss)을 완전히 제거하여 파라미터 수를 줄였습니다. MuSGD(하이브리드 최적화 기법), Progressive Loss(훈련 중 감독을 변화시키는 기법), 그리고 STAL(작은 대상에 대한 라벨 할당 전략)이라는 세 가지 훈련 구성 요소가 긴 훈련 주기를 단축시킵니다. 모델은 n/s/m/l/x의 다섯 가지 크기로 제공되며, 객체 탐지, 인스턴스 분할, 포즈 추정 등을 지원합니다.

- **Performance Highlights**: YOLO26은 COCO 데이터셋에서 1.7-11.8 ms의 지연 시간에 걸쳐 40.9-57.5 mAP을 달성하며 실시간 탐지 분야에서 다른 모델들을 초월했습니다. YOLOE-26은 LVIS minival 에서 텍스트 프롬프트 하에 40.6 AP을 기록하여 DetCLIP-T 보다 +6.2 AP 개선을 달성했습니다. 모든 모델 스케일에서 YOLO26은 최신 실시간 탐지기에 비해 정확도-지연 시간 기준을 개선하며, 향상된 정확도를 제공합니다.



### Qwen-Image-Flash: Beyond Objective Design (https://arxiv.org/abs/2606.03746)
- **What's New**: 본 논문에서는 few-step distillation의 접근 방식을 재조명하여, 학생 모델의 성능을 지적으로 형성하는 훈련 레시피에 중점을 두고 조사합니다. 특히, Qwen-Image-2.0을 사례로 하여 데이터 구성, 교사 안내, 작업 혼합 등 세 가지 주요 요소를 체계적으로 분석합니다. 이러한 연구 결과는 Qwen-Image-Flash라는 새로운 모델 개발로 이어지며, 적은 단계로도 고품질 이미지를 생성할 수 있도록 합니다.

- **Technical Details**: 본 연구에서는 flow matching과 DMD(Discrete Mode Decomposition)를 적극 활용하여 다단계 교사 모델을 few-step 학생으로 증류합니다. flow matching은 데이터와 노이즈 간의 전송 과정을 정의하고, DMD는 학생이 교사의 조건부 분포에 접근하도록 유도합니다. 이러한 방법론은 다양한 작업과 상황에 대한 성능을 개선하는 데 기여합니다.

- **Performance Highlights**: Qwen-Image-Flash는 단 44번의 기능 평가(NFE)로 T2I 생성 및 지시 기반 이미지 편집 작업을 동시에 수행할 수 있도록 설계되었습니다. 이러한 성과는 시각 품질과 강력한 합성 능력을 유지하면서도 효율적인 모델을 생성하는 데 중요한 성과를 거두었음을 보여줍니다. 실험적 분석 결과는 데이터 구성, 교사 안내, 작업 혼합이 효과적인 few-step distillation에 있어 중요한 요소임을 강조합니다.



### Beyond False Stability: High-Noise Drift Gating for Test-Time Adversarial Defenses in Vision-Language Models (https://arxiv.org/abs/2606.03730)
- **What's New**: 이번 연구에서는 CLIP와 같은 비전-언어 모델(VLM)의 적대적 공격에 대한 취약성을 해결하고자 하였습니다. 기존의 적대적 훈련은 효율적이지 않으므로, 테스트 시 방어 기법에 주목하여 새로운 접근 방식을 제안합니다. 저자들은 CLIP의 표현 공간에서 약하게 노이즈가 추가되었을 때의 전이 과정을 분석하여 이를 기반으로 한 새로운 방어 메커니즘을 개발했습니다.

- **Technical Details**: 연구에서는 CLIP의 표현이 노이즈의 크기에 따라 어떻게 변하는지를 분석했습니다. 특히, 약한 노이즈 상태에서는 적대적 예제가 안정적으로 보이지만, 강한 노이즈로 변화하면 적대적 표현이 더 불안정해진다는 점을 파악했습니다. 이를 통해, 적대적 불안정성을 탐지할 수 있는 드리프트 게이팅 신호를 제안하며, 해당 신호를 통해 기존 방어 기법을 선택적으로 활성화합니다.

- **Performance Highlights**: 제안된 방식은 13개의 데이터셋에서 깨끗한 정확도와 강건성 간의 상을 일관되게 향상시켰습니다. 특히, 8개의 세분화된 데이터셋에서는 평균적으로 깨끗한 데이터와 적대적 데이터를 합친 정확도가 65.7%에서 71.4%로, 68.4%에서 73.2%로 상승했습니다. ImageNet 및 네 가지 다양한 변형에서도 유사한 성과를 보여, CLIP 모델의 향상된 성능을 입증하였습니다.



### Text-to-Image Models Need Less from Text Encoders Than You Think (https://arxiv.org/abs/2606.03715)
Comments:
          Project webpage: this https URL

- **What's New**: 이번 연구에서는 텍스트-이미지 모델이 실제로 코드화된 텍스트 깊이 정보 중에서 어떤 요소를 활용하는지에 대한 탐구를 시도합니다. 기존의 인식과는 달리, 저자는 텍스트 임베딩의 풍부한 맥락 정보가 아니라 인근 토큰 병합 및 단어 순서와 같은 두 가지 간단한 요소에 의존한다는 것을 보였습니다. 이를 통해 복잡한 언어 구조의 해독이 이미지 모델 자체에 의해 이루어짐을 발견하였습니다.

- **Technical Details**: 저자들은 기존의 텍스트 임베딩의 맥락 정보를 배제한 새로운 임베딩을 만들어 이를 테스트하였습니다. 이 과정에서 Bag-of-Tokens (BoT), Bag-of-Words (BoW), Bag-of-Position-Tagged-Words (BoPTW)와 같은 세 가지 유형의 컨텍스트 없는 임베딩을 강조하였습니다. 이러한 방식의 임베딩은 텍스트 인코더 출력의 대체물로서 이미지 모델에 기존 수정 없이 적용할 수 있습니다.

- **Performance Highlights**: 연구 결과, 간단한 프롬프트의 경우 BoT 임베딩만으로도 충분히 성능이 향상되었으며, 복잡한 프롬프트의 경우 BoPTW가 가장 우수한 결과를 도출했습니다. 특히, BoPTW는 전체 텍스트 임베딩과 유사한 품질의 이미지를 생성하는 데 성공하면서, TTI 모델이 텍스트 임베딩에 담긴 맥락적 정보보다는 개별 단어 의미와 순서에 의존함을 시사합니다.



### Investigating Adversarial Robustness of Multi-modal Large Language Models (https://arxiv.org/abs/2606.03713)
- **What's New**: 본 연구에서는 Multi-modal Large Language Models (MLLMs)의 적대적 강인성(adversarial robustness)을 체계적으로 조사합니다. 시각 인코더와 언어 모델의 통합 과정에서 발생하는 적대적 취약점을 해결하기 위해 대규모 멀티모달 적대적 프리트레이닝을 도입하며, 이를 통해 시각적 표현의 강인성을 강화합니다. 또한, 강력한 비주얼 백본을 이용한 종합적인 적대적 훈련(end-to-end adversarial training)이 MLLM의 성능을 크게 향상시킴을 보여줍니다.

- **Technical Details**: MLLMs는 CLIP와 같은 비전 인코더(vision encoder)를 통해 언어 모델과 결합되어 시각-언어 reasoning을 지원합니다. 본 연구에서는 진단적 CLIP-alignment 프로토콜을 도입해 적합한 시각 인코더를 선택할 수 있는 기준을 제공합니다. 또한, 강인한 시각적 표현이 없는 표준 MLLM에 대한 직접적인 적대적 훈련이 성능을 저하시킨다는 흥미로운 결과를 보여줍니다.

- **Performance Highlights**: 적대적 프리트레이닝을 통해 MLLMs는 평균적으로 captioning에서 28 CIDEr 포인트, 강한 적대적 공격 하의 VQA에서는 11.7%의 정확도 향상을 보였습니다. 간단한 테스트 시간 방어 기술이 비강인 MLLMs에서도 예기치 않게 효과적이며, 이러한 방어를 통해 적대적 성능이 거의 제로에서 강인한 모델 수준으로 향상됩니다. 마지막으로, 강인한 모델은 화이트박스 비주얼 jailbreak 공격 하에서도 유독한 콘텐츠 발생을 현저히 줄이는 데 기여합니다.



### A Fast Methane Detection Pipeline on Board Satellites Based on Mag1c-SAS and LinkN (https://arxiv.org/abs/2606.03675)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2507.01472

- **What's New**: 이번 연구는 온보드(onsboard) 메탄 감지의 효율을 높이기 위해 새로운 알고리즘을 도입했습니다. 특히, 기존의 Mag1c 알고리즘의 빠른 변형인 Mag1c-SAS를 제안하며, 기존의 메탄 감지 방법에比해 약 80배 더 빠른 성능을 보여줍니다. 또한, 새로운 EMIT-MSeg 데이터셋을 소개하며, 이 데이터셋은 고품질의 주석 전략과 함께 공개됩니다.

- **Technical Details**: 메탄 감지의 주된 방법으로 매칭 필터(MF)를 사용하고, 이를 최적화하여 속도를 높이는 데 초점을 맞추었습니다. 특히, 새로운 밴드 선택 전략인 'Highest Transmittance'와 'Variance Increase'를 탐구하여, 정확도 손실 없이 알고리즘의 실행 속도를 개선하고, 대형 플룸(plume)에 집중하여 감지 성공률을 높였습니다.

- **Performance Highlights**: 제안된 Mag1c-SAS 방법은 원래의 Mag1c 방법보다 약 80배 빠르며, LinkNet 알고리즘과 결합하여 AUPRC 점수를 30 pp 이상 개선했습니다. STARCOP 데이터셋에서도 F1 점수에서 약 4 pp의 향상을 이루었으며, 하드웨어 프로파일링을 통해 시스템의 온보드 성능을 검증하였습니다. 궁극적으로, 이 연구는 저전력 위성 CPU를 위한 최초의 메탄 감지 파이프라인으로 알려져 있습니다.



### Beyond Single Solution: Multi-Hypothesis Collaborative Deep Unfolding Network for Image Compressive Sensing (https://arxiv.org/abs/2606.03666)
Comments:
          Accepted by CVPR 2026

- **What's New**: 최근 제안된 Multi-Hypothesis Collaborative Deep Unfolding CS Network (MHC-DUN)은 여러 가설을 명시적으로 모델링하고 활용하여 다양한 해법 공간에서의 공동 최적화를 수행합니다. 이 방식은 기존의 Compressive Sensing (CS) 접근 방식이 단일 솔루션 공간에 국한되어 있었던 것을 극복하며, 여러 가능성 있는 해를 탐색하는 것이 가능합니다. 특히 MHC-DUN은 AlphaNet을 통해 각 가설에 대한 동적 단계 크기를 예측하고, 다중 해법 간의 협업적인 기울기 업데이트를 실현합니다.

- **Technical Details**: MHC-DUN은 Proximal Gradient Descent 알고리즘을 따르며, 다중 가설 패러다임 내에서 기울기 하강 및 Proximal mapping을 병행적으로 수행합니다. 이를 위해 intra-hypothesis 및 inter-hypothesis 상관관계를 활용하는 고급 Multi-Hypothesis Collaborative Block (MHCB) 모듈이 설계되어, 여러 후보 솔루션을 공동으로 다듬는 역할을 합니다. 또한, end-to-end 훈련을 위한 새로운 복합 손실 함수를 제안하여 측정 정확성, 가설 다양성, 재구성 정확성을 균형 있게 조정합니다.

- **Performance Highlights**: 실험 결과, 제안된 MHC-DUN은 기존의 CS 네트워크보다 성능이 우수한 것으로 나타났습니다. 이 모델은 다양한 솔루션 공간에서 여러 가설을 최적화함으로써, 프레임워크의 효율성을 높이고 이미지 재구성의 정확성을 향상시킵니다. 연구 결과는 또한 MHC-DUN이 높은 정확성의 재구성을 유지하면서 다양한 가능성을 탐색할 수 있도록 고안되었음을 보여줍니다.



### Graph Regularized Non-negative Reduced Biquaternion Matrix Factorization for Color Image Recognition (https://arxiv.org/abs/2606.03654)
- **What's New**: 이 논문에서는 색상 이미지 인식을 위한 그래프 정규화 비음수 축소 바이쿼터니언 행렬 분해(GNRBMF) 모델을 제안합니다. 기존의 비음수 축소 바이쿼터니언 행렬 분해(NRBMF)는 주로 재구성 정확성에 집중하였으나, 제안된 모델은 지역 기하학적 구조를 고려하여 개선된 성능을 자랑합니다. GNRBMF는 그래프 라플라시안 정규화기를 도입하여 인근 샘플의 표현이 유사하도록 유도합니다.

- **Technical Details**: GNRBMF 모델은 비음수 성질을 유지하면서 축소 바이쿼터니언 계수 행렬에 그래프 라플라시안 정규화기를 추가합니다. 이를 통해 학습된 표현이 지역 기하학적 구조를 보존하게 됩니다. 최적화 문제 해결을 위해 구성 요소별 교대 투영 경량 알고리즘을 유도하고, 수렴 특성도 분석하였습니다.

- **Performance Highlights**: CASIA-FaceV5, KDEF, Asirra의 실험 결과, 제안된 GNRBMF 모델은 여러 기존의 실수 기반, 쿼터니언 기반, 축소 바이쿼터니언 기반 방법들과 비교하여 경쟁력 있는 또는 우수한 인식 성능을 보였습니다. 이는 특히 조명 변화, 자세 변형, 배경 간섭 및 클래스 내 큰 차이가 포함된 이미지에서 더 두드러진 효과를 발휘합니다.



### A Benchmark for Semi-supervised Multi-modal Crowd Counting (https://arxiv.org/abs/2606.03646)
- **What's New**: 이 논문은 반지도 학습(semi-supervised learning) 환경에서 다중 모달(multi-modal) 군중 카운팅(crowd counting)의 첫 번째 벤치마크(benchmark)를 구축합니다. 저자들은 라벨이 붙은 데이터와 라벨이 붙지 않은 데이터의 분리를 명확히 정의한 표준화된 프로토콜을 제시하고, 다양한 대표적인 기본선(baseline)을 마련했습니다. 이러한 작업을 통해 기존의 연구에서 주로 완전히 감 supervised된 방법이 다루었던 영역을 새롭게 탐색하고 있습니다.

- **Technical Details**: 이 연구에서는 RGB 이미지를 포함한 다양한 종류의 센서를 활용하여 군중 밀도(crowd density)와 분포를 추정하는 다중 모달 접근 방식을 검토합니다. 특히, RGB와 열 이미지(thermal image)의 공제어적(co-registered) 특성을 활용하여 라벨링 비율(labeling ratio) 5%, 10%, 40%에 따라 코드와 데이터 분할을 제공합니다. 이는 다중 모달 군중 카운팅 모델의 반지도 학습을 위한 기초를 마련합니다.

- **Performance Highlights**: 모델 성능은 RGBT-CC와 DroneRGBT라는 두 개의 주류 다중 모달 군중 카운팅 데이터 세트를 사용하여 평가됩니다. 결과는 Mean Teacher 프레임워크를 활용한 반지도 버전과 라벨이 있는 샘플만을 사용하는 감독 버전을 비교합니다. 실험 결과는 각 모델의 효과성을 비교할 수 있는 통계 자료를 제공합니다.



### VidMsg: A Benchmark for Implicit Message Inference in Short Videos (https://arxiv.org/abs/2606.03635)
Comments:
          Project page: this https URL

- **What's New**: 이번 논문에서는 VidMsg라는 새로운 벤치마크를 소개하며, 이는 짧은 온라인 비디오에서의 암묵적인 메시지 이해를 평가하기 위해 설계되었습니다. VidMsg는 9개의 주제 영역에서 400개의 YouTube 클립으로 구성되어 있으며, 여기에는 경력, 금융, 교육, 건강과 웰빙, 문화, 안전, 지속 가능성 및 라이프스타일과 같은 다양한 분야가 포함됩니다. 이 연구는 비디오가 전달하는 메시지를 인식하는 데 초점을 맞추고 있으며, 이는 단순한 시각적 내용 이상으로, 보다 높은 수준의 의미를 요구합니다.

- **Technical Details**: VidMsg의 구성은 메시지 중심 데이터 수집 파이프라인을 사용하여 이루어집니다. 대상 메시지를 바탕으로 간접 검색 시나리오를 생성하고, 이를 통해 후보 클립들을 검색하여 인간 주석자가 각 클립이 얼마나 목표 메시지를 전달하는지 평가합니다. 이러한 방식은 메시지가 너무 명확하게 드러나지 않도록 필터링하여 비디오 메시지를 보다 잘 이해할 수 있는 벤치마크를 생성합니다. 또한, VidMsg는 예측된 메시지를 선택하는 다중 선택 질의 응답(MCQ) 과제를 포함하여 모델의 메시지 판단 능력을 검사합니다.

- **Performance Highlights**: 실험 결과, 기존의 비디오 언어 및 검색 모델들은 VidMsg에서 좋은 성능을 보이지 않았습니다. 이는 모델들이 단순한 시각적 인식에 의존할 뿐만 아니라, 실질적인 추론과 맥락적 단서를 통합해야 하기 때문입니다. 마지막으로, VidVec-Msg라는 기초 방법을 제안하며, 이는 메시지 중심 검색 성능을 향상시키고 향후 연구를 위한 여지를 남겨 두었습니다.



### TurtleAI: Benchmarking Multimodal Models for Visual Programming in Turtle Graphics (https://arxiv.org/abs/2606.03626)
Comments:
          ACL Findings 2026 paper

- **What's New**: TurtleAI라는 새로운 벤치마크가 소개되었습니다. 이는 교육 중심의 Turtle Graphics 작업을 통해 비전-언어 모델(VLM)의 성능을 평가하기 위한 것입니다. TurtleAI는 823개의 실제 시각 프로그래밍 작업을 기반으로 한 과제를 포함하고 있으며, VLM이 기하학적 패턴을 인식하고 Python 코드를 생성하는 데 필요한 능력을 테스트합니다.

- **Technical Details**: TurtleAI는 VLM이 비주얼 태스크를 해결하기 위해 코드 생성 기능을 평가하는 귀중한 도구로, XLogoOnline 플랫폼의 업무들을 포함하고 있습니다. 이 벤치마크는 목표 이미지를 기반으로 Python 코드 생성을 요구하며, VLM은 공간 관계를 이해하고 이를 코드로 전환해야 합니다. 20개 이상의 VLM이 평가되었으며, GPT-5, GPT-4o, Qwen2-VL-72B 같은 최신 모델들이 30% 미만의 성공률을 기록했습니다.

- **Performance Highlights**: 모델 성능을 개선하기 위해 소량의 seed 샘플을 활용한 데이터 생성 기법이 제안되었습니다. 이 방법으로 얻은 합성 데이터로 Qwen2-VL-72B 모델을 파인튜닝한 결과, 실제 작업에서 약 20% 개선된 성과를 보였습니다. 그러나 GPT-4o는 공간 추론과 정확한 비주얼 재현에 어려움을 겪고 있으며, Qwen2-VL-72B는 코드 구현과 비주얼 추론 간의 정렬 문제가 주된 오류 원인으로 드러났습니다.



### SkelHCC: A Hyperbolic CLIP-Driven Cache Adaptation Framework for Skeleton-based One-Shot Action Recognition (https://arxiv.org/abs/2606.03610)
Comments:
          Accepted by ICML 2026

- **What's New**: SkelHCC는 스켈레톤 기반 동작 인식을 위한 통합된 하이퍼볼릭 CLIP 기반의 캐시 적응 프레임워크를 제안합니다. 이 연구는 기존의 저차원 모션 신호에 의존하는 접근 방식의 한계를 극복하고, 인간 동작의 위계적 구조를 효과적으로 인식할 수 있게 합니다. 새로운 EH-HCLIP 모듈은 스켈레톤 시퀀스와 동작 언어를 공유된 하이퍼볼릭 공간에 내장하여 구조적으로 일관된 교차 모드 표현을 생성합니다.

- **Technical Details**: SkelHCC는 두 가지 주요 구성 요소인 EH-HCLIP와 LMV-Cache를 통합하여 하이퍼 볼릭 공간에서의 위계적 스켈레톤-언어 정렬과 LLM(대형 언어 모델) 가이드 적응을 가능하게 합니다. EH-HCLIP는 하이퍼볼릭 기하의 음의 곡률과 지수적 볼륨 성장을 활용하여 인체 해부학의 구조적 성격을 인코딩합니다. LMV-Cache는 기억 기반 추론 모듈로서, 테스트 시 적절한 본체 영역을 식별하여 문맥 인식 유사성 매칭을 수행합니다.

- **Performance Highlights**: SkelHCC는 NTU RGB+D 60, NTU RGB+D 120 및 PKU-MMD II의 세 가지 벤치마크 데이터셋에서 실험하여 성능을 입증했습니다. 특히 SkelHCC는 NTU RGB+D 120 데이터셋에서 기존의 SOTA 방법들보다 6.7%와 9.0%의 마진으로 성능을 초과 달성했습니다. 이러한 결과는 SkelHCC가 동작 인식 분야에서 효과적인 방법임을 보여줍니다.



### World Models Meet Language Models: On the Complementarity of Concrete and Abstract Reasoning (https://arxiv.org/abs/2606.03603)
- **What's New**: 본 논문은 고정된 시각적 관찰로부터 미래 결과를 예측하는 데 필요한 세계 모델(world models)과 다중 모달 대형 언어 모델(multimodal large language models, MLLMs)의 상호 작용을 탐구합니다. 이러한 모델들은 과거의 정황을 기반으로 미래를 예측하나, 생성된 결과물이 완벽하지 않아 최종 결론에 영향을 미치는 방식을 명확히 해야 할 필요가 있습니다. 이 문제를 해결하기 위해, 저자들은 Controlled Concrete Reasoning이라는 새로운 접근법을 제시하고, Privileged-Future On-Policy Self-Distillation(PF-OPSD)이라는 훈련 프레임워크를 도입하였습니다.

- **Technical Details**: Controlled Concrete Reasoning은 초기 관찰 및 미래 지향 질문이 주어졌을 때 MLLM이 언제 세계 모델을 호출해야 하는지, 그 결과를 어떻게 검증하고 얼마나 신뢰해야 하는지를 학습하는 과정을 포함합니다. 이를 평가하기 위해 VRQABench와 OpenWorldQA라는 두 개의 인간 검증 벤치마크를 제작하였으며, 각 벤치마크는 복잡한 공간적 맥락의 예측 및 개방형 신체 예측을 테스트합니다. PF-OPSD는 훈련 중에 진실 미래 비디오와 정답을 이용하여 학생의 경로를 평가합니다.

- **Performance Highlights**: PF-OPSD는 VRQABench와 OpenWorldQA 각각에서 기준선 모델 대비 10.6% 및 10.9% 성능 향상을 보여주었으며, 생성된 롤아웃의 잡음 또는 상충에 대한 강건성을 높였습니다. 이는 미래 결과 예측을 보다 신뢰성 있게 만들어 주며, 실제로 필요한 시뮬레이션 사용의 결정적인 기준을 마련합니다. 논문은 코드와 데이터셋을 공개하여 연구자들이 해당 연구를 바탕으로 추가적인 발전을 이룰 수 있도록 지원합니다.



### UnsOcc: 3D Semantic Occupancy Prediction in Unstructured Scene via Rendering Fusion (https://arxiv.org/abs/2606.03581)
Comments:
          8 pages

- **What's New**: 이 논문에서는 복잡한 비정형 장면에서의 3D 의미 점유 예측을 향상시키기 위한 새로운 멀티모달 프레임워크인 UnsOcc를 제안했습니다. 기본적으로, 제안하는 방법은 RenderFusion이라는 렌더링 기반의 융합 모듈을 통해 서로 다른 센서 데이터, 즉 이미지와 라이다 데이터를 효과적으로 정렬합니다. 또한, GSRefinement 모듈을 통해 3D 점유 예측이 2D 의미 분할 맵으로 변환되어 장기 꼬리 클래스(위험 가지고 있는 레이블)를 보다 효과적으로 인식할 수 있도록 합니다.

- **Technical Details**: UnsOcc 프레임워크는 3D 점유 예측의 성능을 높이기 위해 bidirectional rendering supervision을 사용하여 교차 모드 기능 정렬을 강화합니다. RenderFusion 모듈은 이미지와 라이다의 특성을 정렬하는 과정에서 깊이 예측 네트워크와 라이다 포인트를 활용합니다. GSRefinement는 3D 공간의 희소한 예측을 2D 의미 분할로 도와주며, 이를 통해 장기 꼬리 클래스의 인식을 개선합니다.

- **Performance Highlights**: 제안된 방법은 open-pit mine 데이터셋과 nuScenes 데이터셋에서 기존의 최신 기술들에 비해 현격히 높은 성능을 보여주었습니다. 특히, 제안한 GSRefinement 모듈이 장기 꼬리 클래스에 대한 예측 성능을 실질적으로 향상시키는 데 기여하고, UnsOcc 특유의 융합 메커니즘은 불규칙한 환경에서도 견고한 성능을 보장합니다.



### Diffusing in the Right Space: A Systematic Study of Latent Diffusability (https://arxiv.org/abs/2606.03578)
- **What's New**: 이번 연구에서는 Latent Diffusion Model에 대한 체계적인 연구를 통해, 토크나이저(tokenizer)와 여러 형태의 잠재(latent) 구성 및 정규화 전략을 바탕으로 생성 품질과의 강한 상관관계를 파악했다. 특히, 새로운 지표인 Velocity Irreducible Variance (VIV)를 제안하였으며, 이는 생성 품질 예측의 중요한 요소로 작용한다는 것을 입증하였다.

- **Technical Details**: 연구에서는 다양한 토크나이저 아키텍처와 잠재 구성을 가지고 토크나이저의 확산 가능성(diffusability)을 분석하였다. 또한, 각 토크나이저에 대해 여러 다운스트림(diffusion) 모델을 훈련시키고 잠재 공간의 속성과 생성 품질 간의 상관관계를 제어된 분석을 통해 평가하였다. VIV는 궤적 교차로 인한 속도 모호성을 측정하는 새로운 방식으로 도입되었다.

- **Performance Highlights**: 실험 결과, VIV, 의미의 분리성(semantic separability), 공간 구조(spatial structure)가 다양한 실험 설정 아래에서 생성 품질의 일관된 예측자로 작용하는 것으로 발견되었다. 또한, 의미의 분리성과 공간 구조를 함께 사용하는 선형 모델이 기존의 개별 요인보다 생성 품질을 더 잘 설명하는 것으로 나타났다. 이러한 결과는 잠재 확산 가능성이 다면적인 특성임을 시사한다.



### Eliciting Complex Spatial Reasoning in MLLMs through Wide-Baseline Matching (https://arxiv.org/abs/2606.03577)
Comments:
          CVPR 2026. Project page: this https URL Code: this https URL

- **What's New**: 이번 논문은 ReasonMatch-Bench라는 새로운 평가 기준을 소개합니다. 이는 다양한 실내, 실외 및 객체 중심 시나리오에서의 Wide-Baseline Matching (WBM) 능력을 종합적으로 평가할 수 있도록 설계되었습니다. 기존의 다중모달 대형 언어 모델(MLLMs)들이 여전히 WBM에서 성능이 부족하다는 점을 강조하고 있으며, 효율적인 데이터 생성 파이프라인을 통해 개선할 수 있는 가능성을 제시합니다.

- **Technical Details**: WBM은 두 개의 서로 다른 뷰포인트에서 동일한 장면 요소를 식별하는 복잡한 작업입니다. 본 연구에서 우리는 MLLMs의 WBM 능력을 향상시키기 위해 Dynamic Correspondence Reinforcement Learning (DCRL) 프레임워크를 도입합니다. DCRL은 이미지 수준과 포인트 수준의 커리큘럼을 결합하여 MLLMs가 복잡한 공간 추론을 점진적으로 습득할 수 있도록 돕습니다.

- **Performance Highlights**: 실험 결과, DCRL은 ReasonMatch-Bench에서 70.5% F1 점수를 기록하여 기존 모델들을 초월했습니다. 이 방법은 3D 포인트 간의 정확한 대응을 예측하는 데 유리하며, OmniSpatial 및 MindCube와 같은 관련 공간 벤치마크에도 긍정적인 전이를 보여줍니다. 또한 일반적인 시각 이해 성능을 유지하면서 여러 벤치마크에서 소폭의 향상을 달성하였습니다.



### When Attention Collapses: Stage-Aware Visual Token Pruning from Structure to Semantics (https://arxiv.org/abs/2606.03569)
- **What's New**: 본 논문에서는 두 단계의 시각적 토큰 프루닝 프레임워크인 Structure-to-Semantics (STS)를 제안합니다. 기존의 시각적 프루닝 방법이 주요하게 사용하던 단일 메트릭은 토큰 집중과 같은 중요한 문제점을 동반했기 때문에, STS는 공간적 구조 다각성을 극대화하는 새로운 방법론을 모색했습니다. 첫 번째 단계는 반발 기반 샘플링 메커니즘을 사용하여 공간 및 구조적 다양성을 높이며, 두 번째 단계에서는 프롬프트와 무관한 토큰을 정교하게 필터링합니다.

- **Technical Details**: STS 프레임워크는 두 단계로 구성되어 있습니다. 첫 번째 단계에서는 시각적 토큰을 전하와 같은 상극의 원리를 통해 모델링하여, 유사한 의미를 가진 토큰들이 서로의 존재를 제한하도록 설계되었습니다. 두 번째 단계에서는 언어 모델의 중간 계층에서 작업과 무관한 토큰을 제거하는 방식으로, 단계에 따른 표현 역학을 고려하여 토큰을 정제합니다.

- **Performance Highlights**: 다양한 비전-언어 모델을 통한 광범위한 실험 결과, STS는 기존의 주의 기반 선택으로 인한 중복성을 줄이고, 보존된 시각 토큰의 구조적 다양성과 정밀한 작업 정렬을 향상시키는 데 성공했습니다. STS를 적용했을 때, 추론 효율성이 개선되었으며, 강력한 작업 성능 또한 유지되었습니다.



### Learned Non-Maximum Suppression for 3D Object Detection (https://arxiv.org/abs/2606.03568)
Comments:
          6 pages, accepted at IEEE Intelligent Vehicles Symposium (IV) 2026

- **What's New**: 이 연구에서는 LiDAR 기반 3D 객체 검출의 후처리 단계에서, 기존의 비최대 억제(non-maximum suppression, NMS)를 대체하는 두 가지 학습된 필터링 모듈을 도입합니다. D2D-Rescore는 변환기(transformer) 기반의 탐지 간(attention) 주의를 활용하고, GossipNet3D는 2D GossipNet 개념을 3D로 적응시킵니다. 이로 인해 제안된 방법들은 필터링의 신뢰성을 향상시키고, 소형 및 드문 클래스에서도 성능을 개선하는 결과를 보여줍니다.

- **Technical Details**: 제안된 방법론은 입력 임베딩, 컨텍스트(feature) 집합, 점수(score) 정제의 세 가지 주요 구성 요소로 이루어져 있습니다. 컨텍스트 집합 단계에서는 두 가지 상호 교환 가능한 변형인 GossipNet3D와 D2D-Rescore를 통해 집합된 정보를 활용하여 신뢰성 있는 점수를 생성합니다. D2D-Rescore는 서로 간의 관계를 모델링하여 자신을 주의(attention)로 활용하고, GossipNet3D는 지역적인 메시지 전달을 통해 3D 환경을 이해합니다.

- **Performance Highlights**: 제안된 모든 방법은 CircleNMS와 비교했을 때 평균 평균 정밀도(mean average precision, mAP), nuScenes 탐지 점수(nuScenes detection score, NDS), 그리고 진짜 긍정 품질(true positive quality)을 개선하였습니다. 특히 드물고 소형 클래스에서 두드러진 성과를 나타내며, 최소한의 계산 오버헤드로도 검출 성능을 높일 수 있습니다. 이 결과들은 기본 네트워크를 수정하지 않고 학습된 검출 레벨 필터링이 3D 탐지기의 신뢰성을 높일 수 있음을 입증합니다.



### Efficient Transformer-Based Localized Patch Sampling for Choroid Plexus Segmentation in Multiple Sclerosis (https://arxiv.org/abs/2606.03566)
- **What's New**: 이번 연구는 측면 뇌실 맥락막(choroid plexus, LVCP)을 다중 경과 면역 염증(multiple sclerosis, MS)의 주요 생체 지표로 활용하기 위한 새로운 자동 세그멘테이션 기법을 제안합니다. 기존의 수동 세그멘테이션 방식은 매우 고된 작업으로, 넓은 임상 시험(clinical trials)과 장기 평가(longitudinal assessments)에서 사용이 제한되었습니다. 이에 따라 SwinUNETR 구조를 활용하여 LVCP를 자동으로 세그멘테이션하는 파이프라인(pipeline)을 개발하였습니다.

- **Technical Details**: 이 연구는 두 개의 MS 주도 집단에서 수집된 3개의 데이터 세트(데이터 세트 1: n=177; 데이터 세트 2: n=177; 테스트 세트 확대: n=388)에 대해 3T MRI 스캔을 후향적으로 평가하였습니다. 제안된 방법은 32x32x32 복셀 패치(voxel patches)를 기반으로 훈련된 SwinUNETR 아키텍처를 사용하였으며, 3D UXNET 모델과 비교하여 성능을 평가하였습니다. 주요 평가지표는 Dice 유사성 계수(Dice Similarity Coefficient, DSC)였으며, 계산 요구사항(GFLOPs) 및 95번째 백분위수 하우스도르프 거리(Hausdorff Distance, HD95)도 보완적으로 고려하였습니다.

- **Performance Highlights**: 확대된 테스트 세트에서 SwinUNETR 모델은 MPRAGE와 FLAIR 결합 시 평균 DSC 0.868을 기록하여 UXNET보다 통계적으로 유의미한 성과를 보였습니다. 독립적인 FLAIR 입력에 제한했을 때도 높은 DSC 0.863를 유지했으며, UXNET의 공간적 위치화는 현저히 나빠졌습니다. 이 프레임워크는 계산 부하를 99% 줄여 22,080 GFLOPs 대신 91.8 GFLOPs만 필요로 합니다. 이러한 혁신적인 접근법은 LVCP 세그멘테이션에 있어 정확하고 안정적인 우수 대안을 제공하며 임상 및 연구 환경에서의 광범위한 구현을 위한 이상적인 방법론으로 자리잡을 것으로 기대됩니다.



### \textsc{CR-Seg}: Attention-Guided and CoT-Enhanced Coarse-to-Refined Reasoning Segmentation (https://arxiv.org/abs/2606.03564)
- **What's New**: 이번 연구에서는 복잡한 언어로 설명된 대상 객체를 분할하기 위해서 시각-텍스트적(visual-textual) 추론을 통한 Reasoning segmentation을 제안합니다. 기존의 방법들은 MLLM(다중모드 대규모 언어 모델)과 분할 모델을 연결하는 데 어려움을 겪었으며, 본 연구에서는 이러한 제한을 해결하기 위한 CR-Seg라는 새로운 프레임워크를 개발했습니다. CR-Seg는 Attention-Guide 및 CoT-Enhanced Coarse-to-Refined Reasoning Segmentation으로, 두 단계로 구성된 새로운 방법론을 제공합니다.

- **Technical Details**: CR-Seg는 이미지와 질문을 입력으로 받아 MLLM을 통해 구조화된 응답을 생성합니다. Attention Maps 및 Points를 추출하는 모듈인 EAP(Extract Attention Maps and Points)를 사용하여 코스(target localization)에 대한 정보 포인트를 선택하고, 이를 기반으로 SAM(Segment Anything Model)을 통해 최종 마스크를 정제합니다. 또한, 모델이 전역(scene context)에서 세부(local target) 정보를 점진적으로 추론하도록 안내하는 GLCoT(Global-to-Local Chain-of-Thought)를 도입하여 추론-답변 불일치성을 완화합니다.

- **Performance Highlights**: CR-Seg는 기존의 방법에 비해 강력한 성능을 보여주며, 정렬의 부담을 줄이는 데 도움을 줍니다. 다양한 Reasoning segmentation 벤치마크에서의 실험 결과는 CR-Seg의 효과성과 강인성을 입증하며, 이 모델이 전반적인 응답 의미를 보존하면서도 분할 품질을 개선할 수 있음을 보여줍니다. 추가로, FReasonSeg라는 보조 벤치마크를 통해 세부적인 범주 객체 간 차별성을 평가할 수 있도록 설계되었습니다.



### Attend to Anything: Foundation Model for Unified Human Attention Modeling (https://arxiv.org/abs/2606.03540)
Comments:
          Accepted to ICML 2026

- **What's New**: 기존의 인간 주의(attention) 모델링 방법들은 다양한 양식(modality), 장면(scene), 과제(task) 구성에서 매우 단편화되어 있습니다. 이러한 한계를 극복하기 위해, 본 연구에서는 Attend to Anything Model (AAM)이라는 다중 양식(multi-modal) 기초 모델을 제안합니다. AAM은 다채로운 이미지, 비디오 및 오디오-비주얼 작업에서의 주의 모델링을 통합하여, 주의를 인지적 귀결 관계(cognitive entailment relationship)로 재구성합니다.

- **Technical Details**: AAM은 일반-특정의 계층적 구조를 통해 주의를 모델링하며, 하이퍼볼릭 공간(hyperbolic space)에서의 언어 프롬프트(language prompts)와 계층적 임베딩(hierarchical embeddings)을 사용하여 기능합니다. 동적인 비디오 주의를 정적 주의의 유체역학적(fluid dynamics) 관점으로 모델링하고, Fokker–Planck 방정식에 기반하여 비디오 프레임 주의를 확산적 시간 발전(diffusive temporal evolution)으로 표현합니다.

- **Performance Highlights**: 16개 벤치마크에서의 광범위한 실험을 통해, AAM은 평균적으로 기존의 최신 기술(state-of-the-art) 방법들보다 6% 더 우수한 성능을 보였습니다. 더불어 비디오 추론에서는 약 4배의 속도 향상을 달성하여, AAM이 주의 및 시각적 관련 작업에 대한 미래 연구의 기초를 제공함을 입증했습니다.



### Knowledge-Preserved Model Tuning in Null-Space for Robust Spatio-Temporal Video Grounding (https://arxiv.org/abs/2606.03539)
Comments:
          Accepted by ICME 2026

- **What's New**: 이 논문에서는 Null-Space Tuning (NST)이라는 새로운 기하학적 조정 프레임워크를 제안합니다. NST는 고품질(HQ) 입력과 저품질(LQ) 입력을 동시에 처리할 수 있도록 설계되어 있으며, 사전 학습된 지식의 손실 없이 저품질 입력에 적소 조정을 가능하게 합니다. 이로 인해 다양한 실제 비디오 환경에서 발생하는 품질 저하 상황을 효과적으로 다룰 수 있습니다.

- **Technical Details**: NST는 고정 가중치의 널 공간(null-space) 구조를 활용하여 입력 특성에 대한 학습 가능한 잔차(residual)를 주입하는 방식으로 작동합니다. 핵심 모듈로는 품질 적응 유닛(QAU)과 이중 공간 재매개화(Dual-Space Reparameterization)가 포함되어 있으며, 이를 통해 저품질 입력에 대한 복원을 수행합니다. 이러한 메커니즘은 HQ 입력의 변화를 방지하고, 비선형 공간(non-null space)으로 저품질 입력에 대한 잔차를 전환합니다.

- **Performance Highlights**: NST는 Mixed-Quality 벤치마크에서 실험을 통해 최신 방법들을 초월하는 성능을 보였습니다. 본 방법은 고품질 데이터에 대한 사전 학습된 지식을 유지하면서 저품질 입력에 대한 복원을 성공적으로 수행하여 최첨단 PEFT 방법들보다 높은 복원 능력을 입증하였습니다. 이를 통해 NST는 다양한 비디오 품질의 조합 처리에 유용한 도구가 될 것으로 기대됩니다.



### EvoMemNav: Efficient Self-Evolving Fine-Grained Memory for Zero-Shot Embodied Navigation (https://arxiv.org/abs/2606.03509)
Comments:
          Preprint

- **What's New**: EvoMemNav는 제로샷 임베디드 내비게이션(zero-shot embodied navigation)을 위한 효율적이고 자기 진화하는 메모리 프레임워크입니다. 이 시스템은 기존의 감지 중심(scene graph) 방식에서 발생하는 정보 손실과 노이즈 문제를 해결하고 있으며, 3D 재구성(3D reconstruction) 방법의 계산적 한계를 뛰어넘었습니다. EvoMemNav는 원시 뷰(raw views)를 메모리의 1급 항목으로 유지하는 Visual-Semantic Memory Graph (VSMGraph)를 구축하여 정보의 세부사항을 잃지 않도록 합니다.

- **Technical Details**: EvoMemNav는 경량의 의미적 단서(sematic cues)와 위상적 관계(topological relations)를 사용하여 방-뷰-오브젝트(room-view-object) 계층으로 메모리를 조직합니다. 메모리의 크기가 증가할수록 예산에 맞춘(coarse-to-fine policy) 방식을 도입하여, 초기 단계에서는 유망한 지역으로 검색 공간을 압축하고, 세부 단계에서는 타겟 검증(targeted verification) 및 의사결정을 위해 VLM을 호출합니다. 또한, 각 하위 작업(subtask) 이후에는 반사 구동(write-back) 기능을 통해 환경 지식을 업데이트하여 재훈련 없이 향후 결정을 개선합니다.

- **Performance Highlights**: GOAT-Bench 및 HM3D에서 수행된 실험 결과, EvoMemNav는 SR/SPL에서 일관된 성능 향상을 보였으며, 여러 인스턴스의 변별성(disambiguation)이 개선되고, 조기 정지(pre-mature stops)가 감소했습니다. 또한, 제로샷 일반화(zero-shot generalization)에서 강력한 성능을 발휘하여 다양한 모달리티(modalities)에서의 효과성을 입증하였습니다.



### Structure-Guided Mixed Masked Pretraining and Spatial Continuity Regularization for Printed Circuit Board Defect Detection (https://arxiv.org/abs/2606.03508)
Comments:
          Preprint. 38 pages, 12 figures, 6 tables

- **What's New**: 이번 연구에서는 PCB(Printed Circuit Board) 결함 탐지를 위한 새로운 두 단계의 프레임워크를 제안합니다. 이 프레임워크는 구조 기반의 혼합 마스킹 사전 훈련과 공간 연속성 정규화를 결합하여 작은 결함을 효과적으로 탐지합니다. 특히 마스킹 기술을 통해 PCB 구조를 포착하며, 이는 전통적인 수작업 검사의 한계를 극복하는 데 기여할 수 있습니다.

- **Technical Details**: 제안된 방법은 두 단계로 구성됩니다. 첫 번째 단계에서 구조 기반 혼합 마스킹 방식을 사용하여 레이블이 없는 PCB 이미지를 활용한 희소 합성곱 마스킹 사전 훈련이 이루어집니다. 두 번째 단계에서는 공간 연속성 정규화 손실을 추가하여 결함 인스턴스에 대한 긍정적인 예측 값을 제한하고, 결함 영역의 정밀한 로컬라이제이션을 촉진합니다.

- **Performance Highlights**: 실험 결과, DsPCBSD+ 데이터셋에서 제안된 방법이 85.5%의 mAP0.5와 52.3%의 mAP0.5:0.95를 달성하여 기존 강력한 기준 탐지기를 능가하는 성능을 보였습니다. 자세한 증감 연구와 질적 결과는 산업 AOI 환경에서 PCB 결함 탐지를 위한 제안된 프레임워크의 효과성을 더욱 강조하고 있습니다.



### AvatarMix: Identity-Preserving Cross-Avatar Composition for Outfit Personalization (https://arxiv.org/abs/2606.03506)
Comments:
          CVPR 2026 Findings. 16 pages, including supplementary material

- **What's New**: 이 논문에서는 기존의 3D 아바타 의상 전이 방법들이 직면한 문제들을 해결하기 위해 AvatarMix라는 새로운 구성 패러다임을 도입합니다. 이 방법은 두 개의 고화질 Gaussian 아바타에서 머리와 몸체를 직접 조합하여 의상의 품질을 보존하고 교차 문제를 피합니다. 또한, 3D 일관성을 유지하며 사용자의 신체 정체성을 보존하는 메쉬 기반 표현 방식을 적용합니다.

- **Technical Details**: AvatarMix는 두 가지 주요 기술 기여를 포함합니다. 첫째로, SeamFix라는 지역적인 디퓨전 모듈을 활용하여 머리와 목의 이음새를 수정하여 아티팩트 없는 결합을 보장합니다. 둘째로, GSReshape라는 메쉬 기반 신체 재형성 모듈을 통해 다양한 체형에 맞춰 의상을 자연스럽게 조정할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 의상 충실도와 정체성 보존에서 최첨단 성능을 달성했습니다. AvatarMix는 고화질 3D Gaussian 아바타에서의 의상 개인화에 대한 새로운 시각을 제시하며, 실감 나는 3D 아바타 개인화의 가능성을 확장합니다.



### Characterizing Detectability in 3DGS Poisoning: A Stage-wise Benchmark (https://arxiv.org/abs/2606.03499)
- **What's New**: 3D 가우시안 스플래팅(3D Gaussian Splatting, 3DGS)은 최근 실시간 사진 현실적 장면 재구성을 위한 주요 표현 방법으로 떠올랐습니다. 그러나 최근 연구에 따르면, 3DGS는 다양한 공격에 취약하며, 이는 모델의 보안에 큰 우려를 제기합니다. 특히 이 논문은 기존의 연구들이 주로 공격 성공률에 중점을 두고 방어 및 탐지 연구는 충분히 다루지 않았다는 점을 강조합니다.

- **Technical Details**: 본 연구에서는 Poison-3DGS라는 새로운 벤치마크를 도입하여 3DGS에서의 독성 탐지를 단계별로 평가합니다. 이는 각 단계에 따라 다르게 나타나는 포렌식 신호를 분석하여, 3DGS 재구성 파이프라인의 다양한 단계를 통해 신호가 어떻게 개발되고 강력하게 나타나는지를 연구하는 것을 목적으로 합니다. 특히, 훈련 다이내믹스 및 가우시안 파라미터 통계와 같은 후반 단계의 신호는 초기 단계에서는 관찰할 수 없는 중요한 단서를 제공합니다.

- **Performance Highlights**: 연구 결과에 따르면, 탐지 가능성은 단계마다 상이하며, 특정 공격 종류에 따라 각기 다른 단계에서 강한 신호가 나타나는 것으로 판단됩니다. 예를 들어, StealthAttack은 데이터 및 훈련 다이내믹스 단계에서 가장 탐지가 잘 이루어지며, 기타 공격들은 주로 최종 모델 단계에서 탐지됩니다. 이러한 통찰은 3DGS의 방어 시스템 설계 및 기존 탐지기술의 성능 향상에 중요한 기초 자료를 제공합니다.



### Low-Frequency Shortcuts in Texture-Driven Visual Learning (https://arxiv.org/abs/2606.03493)
- **What's New**: 이 연구는 텍스처 기반 도메인에서의 단축 학습(shortcut learning)을 분석하고, 이를 표준 데이터셋인 CIFAR-10과 비교합니다. 기존 연구는 주로 형태(Shape) 중심의 기준에 기반하지만, 본 논문은 텍스처 기반의 다양한 응용 분야에서 발생하는 문제를 살펴봅니다. 연구 결과, 텍스처 기반 도메인에서는 저주파 낮은 주파수 성분(low-frequency components, LFCs)이 주로 의사결정을 좌우함을 보여주며, 이는 높은 주파수의 분류 정보가 있음에도 불구하고 발생합니다.

- **Technical Details**: 저자들은 주파수 변환(frequency transformation)을 통해 RGB 이미지를 주파수 도메인으로 변환하고, 선택적으로 특정 주파수 성분을 제거하여 훈련이나 평가를 수행하는 프루닝(pruning) 기반 분석 파이프라인을 채택합니다. 저주파 성분(LFCs), 고주파 성분(HFCs), 중주파 성분(MFCs)에 대한 세 가지 프루닝 전략을 평가하여 학습된 모델의 정확도 기여도(accuracy contribution)를 분석합니다. 이 방법을 통해 성분별로 정확도 기여 분포를 관찰하여 스펙트럼 행동을 분석합니다.

- **Performance Highlights**: 분석 결과, 저주파 성분은 모델의 과도한 의존성을 발생시켜 OOD에서 70%까지 정확도가 떨어짐을 보여줍니다. LFCs를 제거하는 것이 낮은 주파수 손상에 대한 강인성을 40%까지 높이며, 이는 고주파 손상과의 상충 관계(trade-off)를 발생시킵니다. 연구 결과는 텍스처 주도 도메인에서 모델의 일반화 성능 향상(최대 8%)과 더불어, OOD 분포에서의 성능 향상에도 긍정적인 영향을 미친다는 점을 강조합니다.



### TrAction: Action Recognition with Sparse Trajectories (https://arxiv.org/abs/2606.03490)
- **What's New**: 이 논문에서는 행동 인식을 위한 새로운 방법으로, 메모리 및 계산 집약적인 RGB 비디오 대신에 희소한 점 궤적(sparse point trajectories)을 사용하는 접근 방식을 제안합니다. 이는 객체나 장면 대신 동작의 특징을 중심으로 한 인식으로, 기존의 appearance bias(외관 편향)를 피할 수 있는 장점이 있습니다. 연구진은 2.5D 궤적 기반 인식을 위한 간단한 transformer 아키텍처와 masked-trajectory pretraining을 개발하여, 향상된 행동 인식 정확도를 보여주었습니다.

- **Technical Details**: 제안한 방법론은 2D 궤적에 단안 깊이 추정(monocular depth estimation)을 통해 심도를 추가하여 2.5D 궤적을 생성합니다. 이 모델은 복잡한 RGB 비디오 입력을 사용하지 않고도 motion-heavy 데이터셋인 Something-Something V2와 EPIC-Kitchens-100에서 기대 이상으로 좋은 결과를 도출했습니다. 또한, masked-trajectory pretraining은 다운스트림 분류에서 약 5점 정도 정확도를 향상시키는 효과가 있음을 보여주었습니다.

- **Performance Highlights**: 제안된 방법은 Something-Something V2에서 45%의 top-1 정확도, EPIC-Kitchens-100에서 54%의 정확도를 달성하였으며, 특별한 상황에서 V-JEPA를 능가하는 결과를 얻었습니다. 이 모델은 DINOv2 및 V-JEPA 2와 함께 사용될 경우, 각각 8.7점 및 1.6점의 정확도 향상을 보여줍니다. 이러한 성과는 motion trajectories와 appearance-based features가 상호 보완적임을 시사합니다.



### PersistGS: Differentiable Physics for Object Permanence in 4D Gaussian Splatting (https://arxiv.org/abs/2606.03479)
Comments:
          Accepted in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2026 Workshop on Generative 3D Reconstruction

- **What's New**: 본 논문에서는 PersistGS라는 새로운 방법을 제안하여 3D Gaussian Splatting(3DGS)와 차별화 가능한 강체 시뮬레이션(differentiable rigid body simulation)을 결합하여 occlusion(차단) 동안의 객체 영속성을 복원합니다. 이 방법은 시뮬레이션을 통해 관찰된 선행 궤적(pre-occlusion trajectory)을 기반으로 마찰과 속도를 추정하고, 이를 통해 occlusion 기간 동안 객체 Gaussian의 위치를 효과적으로 조정합니다. 해당 방법은 40% 낮은 궤적 오차를 통해 기존의 photometric supervision(광학 감독)을 초월하며, 물리적으로 올바른 궤적으로 객체를 추적할 수 있습니다.

- **Technical Details**: PersistGS는 3DGS의 각 객체를 Gaussian과 충돌 메쉬로 분해하고, 관찰된 프레임에서 물리적 매개변수를 추정하여 SE(3) 궤적을 적용합니다. 이를 통해 occlusion 동안 객체의 위치를 지속적으로 유지할 수 있으며, 최종적으로 photometric supervision이 다시 시작되면 무리가 없도록 합니다. 이 방법은 물리 법칙과 장면 기하학의 접촉 제약 조건을 만족하는 예측 궤적을 보장합니다.

- **Performance Highlights**: 실험 결과, PersistGS는 고립된 카메라에 의해 관찰된 차단 중 순간과 주어진 사례에서 kinematic extrapolation보다 +2.46dB PSNR을 초과하며, 기초 진실 궤적(ground-truth trajectory)에 대해 0.19dB의 격차로 근접합니다. 또한 향상된 centroids silhouette loss를 통해 의도하지 않은 외관 노이즈로부터 위치적 기울기를 격리할 수 있어, 전체적인 재현성을 높이는 결과를 보여줍니다.



### Mixed-Modality Dual Face-Hair Retrieva (https://arxiv.org/abs/2606.03470)
- **What's New**: 이 논문에서는 Dual Face-Hair Retrieval (DFHR)라는 새로운 혼합 모달리티 이중 참조 작업을 도입합니다. DFHR의 쿼리는 얼굴 이미지와 헤어스타일 참조를 포함하며, 후자는 이미지 또는 텍스트로 표현될 수 있습니다. 이와 같은 설정은 이전의 단일 모달리티 검색 방식과는 다르게, 서로 다른 두 속성 간의 교차 구성 요소 추론이 요구됩니다.

- **Technical Details**: DFHR는 세 가지 주요 요소를 포함한 새로운 프레임워크로, 특정 얼굴 정체성과 스타일을 보유한 이미지를 검색합니다. 이를 위해 지역화된 특징 분리(localized feature disentanglement), 교차 모달 의미 정렬(cross-modal semantic alignment), 혼합 모달리티 조합(mixed-modality composition) 등이 필요합니다. DFHR-Bench라는 최초의 벤치마크 데이터셋도 개발하여 180K 개 이상의 주석 처리된 삼중 데이터를 포함하고 있습니다.

- **Performance Highlights**: DFHR는 사용자의 의도를 보다 정밀하고 표현력 있게 검색할 수 있는 기능을 제공합니다. 혼합된 헤어스타일 쿼리는 시각적 형식 또는 언어적 형식으로 제공될 수 있으며, 이러한 모달리티 변화는 사용자가 원하는 스타일에 대한 표현력을 획기적으로 확장합니다. 이로 인해 사용자 요구 사항을 명확하게 반영한 검색 결과를 제공할 수 있습니다.



### From 3D Perception to Safety Reasoning: A Graph-Based Framework for Real-Time Underground Mine Monitoring (https://arxiv.org/abs/2606.03460)
- **What's New**: 이 논문은 지하 석탄 채굴에서 복잡한 위험 요소들을 실시간으로 모니터링할 수 있는 새로운 프레임워크를 제안합니다. 기존의 모니터링 시스템은 미리 정의된 사건만 감지할 수 있었지만, 이 연구에서는 3D 포인트 클라우드를 활용하여 안전 논리 출력을 생성합니다. 3D 시맨틱 인식(semantic perception)과 불확실성 기반(anomaly detection) 이상 탐지 기술이 결합되어 더욱 진화된 위험 식별이 가능합니다.

- **Technical Details**: 이 프레임워크는 3D 시맨틱 인식, 불확실성 기반 이상 탐지, 규칙 기반 위험 확인 및 LLM(reasoning) 추론을 통합합니다. 또한, GraphRAG을 활용한 메모리 분석(memory analysis)을 통해 즉각적인 위험을 식별하고 장기적인 안전 패턴을 해석할 수 있습니다. 장면(scene)과 시간적 그래프는 인식 결과를 논리 추론 단계에서 연결하는 명시적 지식 구조 역할을 합니다.

- **Performance Highlights**: 제안된 인식 모델은 제한된 주석으로부터 92.7%의 정확도를 달성했으며, 30 FPS의 속도로 낮은 메모리 사용량을 보였습니다. 115개의 위험 시나리오에서 규칙 기반 검사는 57%의 커버리지를 달성했으며, LLM 추론을 적용할 경우 76%, 메모리 기반 추론을 사용할 경우 93%로 증가하였습니다. 이러한 성능 향상은 기존의 클래스 벗어난 위험 요소도 해석할 수 있도록 한 불확실성 기반 이상 신호를 통해 지원됩니다.



### PRISM: Synergizing Vision Foundation Models via Self-organized Expert Specialization (https://arxiv.org/abs/2606.03444)
Comments:
          Accepted to ICML 2026

- **What's New**: PRISM(프리즘)이라는 새로운 이중 스트림 Mixture-of-Experts(MoE) 프레임워크를 제안하여 다양한 Vision Foundation Models(VFMs)의 보완적인 강점을 결합하여 단일 효율 모델을 효과적으로 통합합니다. 이 모델은 이론적으로 세 가지 핵심 요소를 기반으로 하여, 모듈화된 전문화(modular specialization)를 통해 VFMs의 성능을 최적화합니다. 이에 따라, PRISM은 기존의 정적인 구분 방식 이전에 자율적으로 전문화가 이루어지도록 하는 혁신적인 접근 방식을 제시합니다.

- **Technical Details**: PRISM은 두 단계의 패러다임을 적용하여, 첫째, 전문성을 해체(expertise deconstruction)하는 과정에서 교사를 조건으로 하는 라우터가 전문가들을 특화된 표현의 하위 공간으로 안내하고, 둘째, 동적 재조합(dynamic recomposition)을 통해 라우터가 하위 태스크에 맞춘 계산 경로를 조합하도록 학습하게 합니다. 이 시스템은 지식이 부분적으로 겹칠 때 전문 지식이 자연스럽게 발생하도록 하며, 생략적인 경량성과 재능이 있습니다.

- **Performance Highlights**: PASCAL-Context와 NYUD-v2 데이터셋에서의 실험을 통해, PRISM이 새로운 최첨단 기량을 수립하는 성과를 보여주었습니다. 이러한 결과는 희소하고 발생하는 전문화가 다양한 시각적 지식을 통합하는 효율적인 접근 방식임을 확인해줍니다. PRISM은 복잡한 시각적 내용을 효과적으로 다룰 수 있는 잠재력을 보여주며, 미래의 연구 개발을 위한 중요한 초석이 될 것입니다.



### PHAF-Personalized Hand Avatars in a Flash (https://arxiv.org/abs/2606.03420)
- **What's New**: PHAF(고속 개인화 손 아바타)라는 새로운 시스템을 소개합니다. 이 시스템은 손의 등쪽 및 바닥 방향 이미지만으로 고품질의 다중 뷰 렌더링을 제공합니다. PHAF는 기존의 느린 최적화 기법과 달리, 실시간 배치를 위한 개인화된 텍스처를 빠르게 생성할 수 있습니다.

- **Technical Details**: PHAF는 의미적으로 안내되는 메쉬 정렬(semantic guided mesh alignment)과 밀집 텍스처 추출(densified texture extraction)을 결합하여 고주파 세부 정보를 효율적으로 전이합니다. 뷰 기반의 인페인팅 네트워크(view-based inpainting network)를 통해 텍스처를 정제하여 부드럽고 연속적인 모습을 보장합니다.

- **Performance Highlights**: PHAF는 텍스처 생성 시간을 30배 단축시켜 AR/VR 어플리케이션에 실용적으로 활용할 수 있도록 합니다. 실험 결과, 기존의 방법들과 비교하여 시각적 충실도가 유사하면서도 생성 시간이 크게 단축된 것으로 나타났습니다.



### IDO: Incongruity-aware Distribution Optimization for Multimodal Fake News Detection (https://arxiv.org/abs/2606.03418)
Comments:
          Accept by GlobalSouthML@ICML 2026

- **What's New**: 이 논문에서는 다중 모달(fake news detection) 가짜 뉴스 탐지의 새로운 접근 방식인 Incongruity-aware Distribution Optimization (IDO)를 제안합니다. 기존 방법들은 크로스 모달 일관성(cross-modal consistency)에 집중해 왔지만, 정보의 의미적 불일치(semantic incongruity)를 명시적으로 모델링하지 못했습니다. IDO는 사실 불일치(factual incongruity)와 모달리티 불일치(modality incongruity) 관점에서 수행 성능을 개선하는 방법을 제시합니다.

- **Technical Details**: 본 연구에서는 이미지-텍스트 쌍을 정의하고, ViT와 BERT와 같은 인코더를 통해 이미지와 텍스트 데이터를 처리합니다. 이 후, 크로스 모달 주의(attention) 모듈을 통해 이미지와 텍스트 모달리티 간의 상호작용을 구축합니다. 사실 의미 분포(Factual Semantic Distribution) 모델링과 불일치 대조 학습(Incongruity Contrastive Learning) 모듈을 이용하여 사실과 모달리티 정보가 강화되는 방식으로 가짜 뉴스의 불일치를 모델링합니다.

- **Performance Highlights**: 상당한 공적 데이터셋에서 실험한 결과, IDO는 기존의 최신 기법들과 비교하여 뛰어난 성능을 나타냈습니다. 가짜 뉴스 탐지의 정확성을 높이는 데 있어 IDO의 주요 기여는 그 불일치를 탐지하고 효과적으로 모델링하는 것입니다. 이러한 접근 방식은 다양한 뉴스 콘텐츠의 진위 여부를 판단하는 과정에 중요한 인사이트를 제공합니다.



### A unified multi-task framework enables interpretable chest radiograph analysis (https://arxiv.org/abs/2606.03417)
- **What's New**: IMT-CXR (Interpretable Multi-task Transformer for Chest X-ray Analysis)은 방사선 전문의들의 진단 작업 흐름을 모방하는 프레임워크로, 질병 인식, 속성 특성화, 증거 통합 보고서 생성을 포함하는 세 가지 증거 기반 단계를 포함하고 있습니다. 기존의 블랙 박스 시스템이 격리된 작업에 국한되는 반면, IMT-CXR은 다중 작업 방식으로 임상 진단의 신뢰성을 높이는 데 초점을 맞추고 있습니다. 이 프레임워크는 통합된 변환기 아키텍처를 사용하여 여러 임상 작업을 동시에 수행할 수 있도록 설계되었습니다.

- **Technical Details**: IMT-CXR은 질병 분류, 병변 위치 지정, 해부학적 구획화 및 방사선 보고서 생성을 포함하는 네 가지 임상 작업을 연속적으로 수행하는 통합된 모델입니다. 모델은 AI의 중재 하에 계층적 피처 발견을 통해 근거 기반의 추론을 가능하게 하며, 방사선 전문의들이 사용하는 전체적인 접근 방식을 모방합니다. 비전-언어 과제의 감독 Instruction Tuning을 통해 성능이 최적화되며, 다양한 주제의 매개변수를 분석할 수 있습니다.

- **Performance Highlights**: IMT-CXR은 네 개의 임상 작업으로 구성된 다중 작업 모형을 통해 강력한 성능을 보여주며, 160개의 역사적 보고서를 바탕으로 한 블라인드 평가에서는 방사선 전문의들이 AI 생성 보고서의 66%를 원래의 임상 보고서와 동등하거나 더 뛰어난 것으로 평가했습니다. 또한, 주요 성과 지표에서 비교적 낮은 생략률(1.87%) 및 오류율(2.24%)을 기록하며, 이는 방사선 전문의들이 제공한 참고 보고서와 근접한 수치입니다. 따라서 IMT-CXR은 방사선 진단의 임상적 유용성을 효과적으로 증대시키는 것에 기여할 것으로 기대됩니다.



### Enginuity: A Dataset and Benchmark for Vision-Language Understanding of Engineering Diagrams (https://arxiv.org/abs/2606.03410)
- **What's New**: 이 논문에서는 복잡한 엔지니어링 다이어그램을 평가하기 위한 첫 번째 공개 데이터셋인 Enginuity를 소개합니다. 이는 실제 군사 서비스 및 수리 매뉴얼에서 파생된 2,056개의 다이어그램-부품 테이블 쌍으로 구성되어 있습니다. 기존 VLM (Vision-Language Models) 벤치마크가 엔지니어링 콘텐츠를 의미 있는 규모로 다루지 않는 가운데, Enginuity는 엔지니어링 다이어그램에 대한 평가의 필요성을 해결합니다.

- **Technical Details**: Enginuity 데이터셋은 두 개의 작업으로 구성됩니다: (1) 구조화된 부품 테이블 추출(Task 1)과 (2) 자유형 시각적 다이어그램 질문 응답(VQA)(Task 2)입니다. 이 데이터셋은 10개의 미국 군사 서비스 및 수리 매뉴얼에서 수집되었으며, 각 다이어그램은 공식적으로 구조화된 부품 테이블과 연결되어 있습니다. 모델 평가에는 네 가지 최전선 VLM(GPT-5.2 Chat, Claude Opus 4.7, Gemma 4, Qwen3-VL-32B-Instruct)이 포함됩니다.

- **Performance Highlights**: 모델들은 Task 1에서 Recall@all 점수는 0.61-0.87로 도달했지만, Token F1pen은 0.03-0.18로 나타났습니다. 이는 부품 식별과 설명의 신뢰도 간의 체계적인 차이를 드러냅니다. Task 2에서는 모든 모델에서 일관된 사실적 추론 갭(factual-reasoning gap)이 나타났습니다. 이 연구는 LLM(대규모 언어 모델) 평가에서 기술적 설명의 성능을 더 잘 반영하기 위해 LLM을 평가자로 사용해야 할 필요성을 강조합니다.



### SAMatcher: Co-Visibility Modeling with Segment Anything for Robust Feature Matching (https://arxiv.org/abs/2606.03406)
Comments:
          14 pages

- **What's New**: 본 논문에서는 SAMatcher라는 새로운 기능 매칭 프레임워크를 제안합니다. 이 프레임워크는 수학적으로 일치 추정을 공동 가시성(co-visibility) 모델링을 통해 접근합니다. SAMatcher는 지역 특성(local features)을 직접 매칭하기보다는 먼저 공동 가시성 영역 마스크를 예측하여 일치 추정의 구조적 사전(structured priors)으로 사용합니다.

- **Technical Details**: SAMatcher는 Segment Anything Model(SAM)을 기반으로 하여 대칭 교차 시점 상호작용 메커니즘을 도입합니다. 이를 통해 양방향 기능 교환(bidirectional feature exchange) 및 교차 시점 의미 정렬(cross-view semantic alignment)을 가능하게 합니다. 또한, 마스크 예측(mask prediction) 및 박스 위치 지정(box localization)을 공동 최적화할 수 있는 통합 감독 방식(unified supervision scheme)을 개발하였습니다.

- **Performance Highlights**: 엄청난 bench마크 실험 결과, 기존의 매칭 파이프라인(pipeline)에 비해 많은 개선을 보였습니다. 특히 큰 시점(viewpoint)과 크기(scale) 변동에서 더욱 두드러진 성과를 보여주었습니다. 단안(segmentation)으로 설계된 기본 모델이 공동 가시성 모델링을 통해 다중 시점(multi-view) 일치 추정(matching)으로 효과적으로 확장될 수 있음을 보여줍니다.



### Mamba-Enhanced Implicit Motion Learning for Audio-Driven Portrait Animation (https://arxiv.org/abs/2606.03402)
Comments:
          accepted by ICME 2016

- **What's New**: 이번 연구에서는 고유한 정적 이미지와 오디오에서 현실적이고 일관된 인간 동작 비디오를 생성하기 위한 새로운 암묵적 동작 프레임워크를 제안합니다. 기존의 키포인트 기반 방법들이 미세한 동작 동역학을 포착하는 데 어려움을 겪는 점을 극복하고자 한 이 접근 방법은 모션 예측과 렌더링을 분리하여 효율성을 높이는 데 중점을 두고 있습니다. 또한, 새로운 380시간 고품질 오디오-비주얼 데이터셋을 기반으로 훈련하여 정확성과 자연스러움, 시간적 일관성에서 이전 연구들을 능가하고 있습니다.

- **Technical Details**: 이 프레임워크는 두 단계로 구성된 훈련 파이프라인을 통해 이루어집니다. 첫 번째 단계에서는 DIT(Deviation Image Transformer)를 통해 이미지의 외관과 깊이 계층을 통합하여 잠재적 동작 특징을 모델링합니다. 두 번째 단계에서는 Mamba 강화 확산 모델이 오디오와 이미지로부터 이러한 특징을 직접 예측하여 세밀한 동작 패턴의 비지도 학습을 가능하게 합니다. 이 구조는 유연성과 효율성을 높입니다.

- **Performance Highlights**: 실험 결과 본 방법은 여러 공개 벤치마크와 자체 수집 데이터에서 이전 방법들에 비해 우수한 성능을 보였습니다. 특히, 정확성, 자연스러움, 시간적 일관성의 모든 측면에서 새로운 최첨단 기록을 세웠습니다. 이를 통해 고유의 동작 접선과 감정 표현을 더욱 효과적으로 전이할 수 있는 가능성을 열었습니다.



### Towards Characterizing Scientific Image Utility and Upgradability (https://arxiv.org/abs/2606.03401)
- **What's New**: 이 논문에서는 과학적 이미지의 평가를 위한 	extbf{SIU$^2$A} 프레임워크를 제안합니다. 기존 방법들이 지닌 한계를 보완하기 위해, 해당 프레임워크는 두 가지 보완적인 차원인 유용성 	extit{(Utility)}와 업그레이드 가능성 	extit{(Upgradability)}를 도입합니다. 구체적으로 이는 오류 탐지 	extit{(error detection)} 및 수정 가능성 	extit{(correction feasibility)}를 평가하며, 과학적 이미지의 부정확성을 네 가지 기본 유형인 세부 왜곡, 불완전성, 잘못된 내용, 및 개체 혼동으로 분류합니다.

- **Technical Details**: SIU$^2$A 프레임워크는 두 단계의 평가 프로토콜을 구현합니다: 첫 번째 단계에서 오류 탐지 능력과 수정 지침 생성을 평가하고, 두 번째 단계에서는 기존의 정확한 정보를 손상시키지 않으면서 과학적 유효성을 충실히 복원할 수 있는지를 판단합니다. 각 이미지 유형에 대해 제어된 변형 작업을 사용하여 신뢰성 있는 과학적 이미지 결함을 구성하였고, 이와 동일한 이미지를 유지하기 위한 주석을 제공합니다. SIU$^2$A-Benchmark라는 데이터셋이 구축되어 전문가 주석을 통해 오류 식별 및 수리 가능성을 평가합니다.

- **Performance Highlights**: 현재 다중 모달 시스템들은 과학적 오류 평가 및 왜곡 없는 복원 성능에서 상당한 한계를 보였으며, 이는 시각적 지각과 과학적 사용 가능성 간의 근본적인 격차를 나타냅니다. 연구 결과, SIU$^2$A 프레임워크는 과학적 이미지 복원 가능성을 의미 있는 방식으로 평가하도록 설계되어, 이미지 평가를 수동적 평가에서 실행 가능한 복원으로 전환하는 데 기여합니다. 본 프레임워크는 과학적 이미지의 신뢰성을 구조적으로 평가할 수 있는 새로운 기준을 제시합니다.



### P\textsuperscript{2}-DPO: Grounding Hallucination in Perceptual Processing via Calibration Direct Preference Optimization (https://arxiv.org/abs/2606.03376)
- **What's New**: 이번 연구에서는 Large Vision-Language Models (LVLMs)에서 발생하는 환각(hallucination) 문제를 해결하기 위한 새로운 접근법인 Perceptual Processing Direct Preference Optimization (P²-DPO)를 제안합니다. P²-DPO는 모델이 자체적으로 preference data를 생성하고 학습할 수 있도록 하여 시각적 병목 현상(perceptual bottleneck) 문제를 직접적으로 해결합니다. 우리의 방법은 시각적 신호와 텍스트 간의 인과적 생성 관리를 강화하기 위한 Calibration Loss를 포함합니다.

- **Technical Details**: P²-DPO는 두 가지 새로운 preference pair를 도입하여, 각기 다른 시각적 결함을 해결합니다. 첫 번째는 Focus-and-Enhance Preference Pair로, 이는 세밀한 세부 사항의 향상된 출력과 열화된 출력 간의 대조를 통해 Perceptual Bottleneck을 극복합니다. 두 번째는 Visual Robustness Preference Pair로, 정확한 정보와 노이즈 신호 간의 출력을 대비하여 시각적 강인성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, P²-DPO는 같은 양의 학습 데이터와 비용을 사용했음에도 불구하고, 강력한 기준선을 초과하는 성능을 보여줍니다. 특히 Attention Region Fidelity (ARF) 평가에서는 높은 정확도를 기록하였으며, 이미지 열화 시나리오에서도 시각적 강인성을 크게 개선함을 입증하였습니다.



### SynCred-Bench: Benchmarking Synthetic Credibility in AI-Generated Visual Misinformation (https://arxiv.org/abs/2606.03348)
- **What's New**: 최근의 생성 모델들은 현실적인 텍스트와 레이아웃을 포함한 시각적 아티팩트를 생성할 수 있는 능력을 갖춰, 새로운 유형의 잘못된 정보 위협인 합성 신뢰성을 제시합니다. 본 연구에서는 SYNCRED-Bench라는 600개의 AI 생성 잘못된 정보 이미지를 포함하는 벤치마크를 소개하며, 이는 여섯 가지 신뢰할 수 있는 형식(categories)과 일곱 가지 세부 유통 스타일을 반영하고 있습니다. 또한, FP450라는 진짜 이미지 부정 집합을 구축하여 잘못된 긍정(false positives)을 측정합니다.

- **Technical Details**: 합성 신뢰성을 탐지하기 위해, 본 연구는 시각적 잘못된 정보의 평가를 위한 SynCred-Bench라는 포괄적인 벤치마크를 구성하였습니다. 이 벤치마크는 여섯 가지 신뢰 가능한 형식 카테고리와 네 가지 조잡한 및 일곱 가지 세부 신뢰 유통 스타일을 다루며, 총 600개의 AI 생성 샘플을 포함하고 있습니다. 실험 결과, 대부분의 다중 모드 대형 언어 모델(MLLMs) 및 AIGC 탐지기는 여전히 신뢰성이 낮음을 보여줍니다.

- **Performance Highlights**: 15개의 MLLMs을 평가한 결과, 5%의 잘못된 긍정 비율(FPR)을 제한했을 때, 평균 진짜 긍정 비율(TPR)은 10.5%에 불과했습니다. AIGC 탐지기는 평균 TPR이 5% 미만인 반면, 상업 API는 57.6%의 정확도를 기록했습니다. 인간 평론가들조차 합성 신뢰성을 식별하는 데 어려움을 겪었으며, 투표의 대부분이 63.0%의 TPR로 나타났습니다.



### Beyond Semantics: Modeling Factual and Affective Perceptual Experiences from Vision-Language Data (https://arxiv.org/abs/2606.03345)
Comments:
          8 pages

- **What's New**: 이 논문에서는 감정적으로 그리고 문화적으로 이미지를 인식하는 방식을 이해하기 위한 새로운 문제인 P-Topics(Perception Topics) 모델링을 제안합니다. 이 연구의 주요 목표는 이미지와 캡션의 데이터셋에서 다양한 인식 경험을 발견하고 모델링하는 것이며, 각 경험은 객관적인 사실적 측면과 주관적인 정서적 측면으로 정의됩니다. 이를 위해 PercepT(Perception topic Transformer)라는 새로운 아키텍처를 소개하며, 이는 이미지와 인식 경험을 연관시키기 위한 두 단계의 구조를 가지고 있습니다.

- **Technical Details**: PercepT는 두 단계로 구성된 아키텍처로, 첫 번째 단계에서는 비지도 학습 목표를 사용하여 이미지-캡션 쌍의 P-Topics를 발견합니다. 두 번째 단계에서는 주의 풀링(attention pooling)을 활용하여 이미지와 관련된 P-Topic 매핑 기법을 학습합니다. CLIP 기반 인코더는 객관적인 사실 정보를 나타내고, 감정 인코더는 주관적인 정서 정보를 캡처하여 이미지의 장르와 감정을 프록시 라벨로 사용합니다.

- **Performance Highlights**: PercepT는 ArtELingo 데이터셋에서 0.97의 실루엣 점수를 달성하여, 가장 가까운 기준선인 0.37에 비해 더 나은 인식 클러스터를 반영합니다. 또한, PercepT는 0.94의 AUC 점수를 달성하여 0.77의 기준선에 비해 더 나은 매핑 성능을 보여줍니다. 실험 결과 사람 평가에서도 PercepT가 58.4%의 선호도를 나타내며 기존 방법들보다 우수한 성능을 입증합니다.



### Cross-Modality Feature Fusion Based on Structured State Space Duality for Multimodal Image Registration Network (https://arxiv.org/abs/2606.03341)
- **What's New**: 이번 논문에서는 멀티 모달 이미지 등록(multi-modal image registration)을 위한 새로운 알고리즘인 RegNetMamba-2를 제안했습니다. 이 알고리즘은 Structured State Space Duality (SSD)를 기반으로 하여 조정 효율성을 높이며, 로컬(local) 및 글로벌(global) 구조적 특징을 효과적으로 추출합니다. 특히, Edge와 구조적 정보에 중점을 두어 성능을 개선하였습니다.

- **Technical Details**: RegNetMamba-2는 coarse-to-fine 매칭 과정에서 SSD를 도입하여 멀티모달 특징을 추출합니다. 우선적으로, SSD는 다양한 스케일에서 모달 특징을 추출하고, Cross-Modality feature Interaction (CMI) 모듈과 Multi-Scale feature Fusion (MSF) 모듈이 활용됩니다. CMI 모듈은 각 스케일에서의 모달 특징을 교차적으로 추출하고, MSF 모듈은 여러 스케일에서 추출된 특징을 융합하여 최종적인 특징을 생성합니다.

- **Performance Highlights**: 실험 결과, RegNetMamba-2는 VIS-SAR, VIS-IR, VIS-NIR 데이터셋에서 최신 딥러닝 기반 알고리즘과 비교하여 성능과 효율성 모두에서 뛰어난 효과를 입증하였습니다. 이러한 성과는 다양한 센서에서 수집된 정보의 융합을 통해 이루어진 것으로, 멀티모달 이미지 등록의 어려운 도전과제를 효과적으로 해결한 결과입니다.



### TASE: Truncation-Aware Semantic Embeddings for 3D Scene Understanding and Editing (https://arxiv.org/abs/2606.03314)
- **What's New**: 본 논문에서는 TASE(Truncation-aware Semantic Embedding)라는 새로운 방법을 소개합니다. 이 방법은 사전 학습된 2D 의미적 특징을 3D 장면 편집을 위한 트런케이션 인식 임베딩 공간으로 변환합니다. TASE는 기존 방법들에 비해 장면 내용에 대한 강력한 편집 제어를 가능하게 하여 더 큰 수정이 가능합니다.

- **Technical Details**: TASE는 채널 순서 구조를 도입하여 의미적 추상화를 제어하는 기능을 제공합니다. 채널 수를 줄이면 더 추상적인 표현이 생성되며, 많은 채널을 유지하면 세부 정보가 보존됩니다. 또한, 획일성을 유지하기 위해 스케일 및 변환 적합성 손실을 사용하여 다중 뷰 일관성을 향상시켰습니다.

- **Performance Highlights**: 실험 결과에서 TASE는 기존의 3D 장면 편집 방법들보다 우수한 성능을 나타냈습니다. 특히, 대규모 기하학적 수정이 필요한 편집에서 기존 방법들과 비교해 상당히 향상된 결과를 보여주었습니다. 이 연구는 3D 장면 편집의 가능성을 크게 확장하며, 더 나은 활용을 위한 새로운 경로를 제시합니다.



### BA-T: An Iterative Transformer for Two-View Bundle Adjustmen (https://arxiv.org/abs/2606.03287)
- **What's New**: 이 논문에서는 3D 복원(3D reconstruction)을 위한 새로운 접근 방식인 BA-T를 제안합니다. 기존의 feed-forward 모델들은 깊은 cross-view attention을 사용하여 이미지 간 정보를 교환하는 데 강력한 성능을 보였으나, 복잡한 디코더 구조와 비효율적인 기하학적 정제(geometry refinement) 메커니즘으로 인해 다중 뷰 일관성(multi-view consistency)이 부족했습니다. BA-T는 이러한 문제를 해결하기 위해 전통적인 bundle adjustment(BA)에서 영감을 받았습니다.

- **Technical Details**: BA-T는 BA 스타일의 구조화된 업데이트를 반복 가능한 레이어로 구현하며, 이를 통해 중복된 attention 대신 경량(layer) 레이어에서 잠재적 잔여(latent residual)를 기반으로 예측을 정제합니다. 또한 BA-T는 포즈(pose)와 지역 기하학(local geometry) 간의 정보를 반복적으로 전달하는 과정으로 볼 수 있으며, 이는 효율성을 높입니다. 이러한 구조는 복잡한 디코더 스택 없이 작동하게 합니다.

- **Performance Highlights**: BA-T는 반복(iteration)을 거치며 포즈와 복원 정확도를 점진적으로 개선하고, 기존 디코더보다 더 강력한 cross-view 일관성을 성취했습니다. 실험 결과, BA-T는 더 큰 모델에 비해 확연한 정확도를 보이며 전체 디코더 매개변수의 16%만으로도 매우 높은 성능을 발휘합니다. 이로써 BA-T는 경량 아키텍처 내에서 정확한 3D 복원을 가능하게 하는 효율적이고 구조적인 대안을 제시합니다.



### VistaHop: Benchmarking Multi-hop Visual Reasoning for Visual DeepSearch (https://arxiv.org/abs/2606.03273)
- **What's New**: 이 연구에서는 Visual DeepSearch를 위한 새로운 벤치마크인 VistaHop을 소개합니다. VistaHop은 복잡한 비주얼 쿼리에 답하기 위해 이미지 영역을 반복적으로 검사하고, 시각적 증거를 기반으로 중간 추론을 연결하게 하는 멀티모달 대형 추론 모델(MLRM) 에이전트를 평가합니다. 기존 벤치마크가 단일 단계 시각 이해에 주로 초점을 맞추고 있다는 점에서 차별화됩니다.

- **Technical Details**: VistaHop은 300개의 고해상도 이미지, 25개의 시각 검색 시나리오, 그리고 시각적 앵커와 관련된 증거 체인을 추적하는 350개의 멀티 홉 품질 보증(QA) 작업을 포함합니다. 이를 통해 모델들이 여러 이미지 기반 추론 경로에서 정보를 융합하고, 이미지와 외부 지식을 연결할 수 있는 능력을 평가합니다. 또한, VistaArena라는 통합 평가 환경을 개발하여 텍스트 검색, 이미지 검색, 이미지 잘라내기 및 증거 기반 답변 검증을 지원합니다.

- **Performance Highlights**: 일곱 개의 대표적인 MLRM에 대한 실험 결과 현재 모델들은 VistaHop 작업을 성공적으로 해결하는 데 한계가 있음을 보여주었습니다. 가장 우수한 성과를 기록한 SenseNova-MARS-32B 모델조차도 24.31%라는 낮은 Pass@1 비율을 보였습니다. 이는 현재의 모델들이 시각적 고정을 통한 증거 재검토, 장기적 추론 및 멀티 앵커 정보 융합에서 여전히 한계를 가지고 있음을 드러냅니다.



### PaddleOCR-VL-1.6: Expanding the Frontier of Document Parsing with Under-Optimized Region Refinement and Progressive Post-Training (https://arxiv.org/abs/2606.03264)
- **What's New**: PaddleOCR-VL-1.6이 새롭게 등장했습니다. 이는 PaddleOCR-VL-1.5를 기반으로 한 업그레이드된 문서 분석 모델로서, 0.9B의 강력한 기준을 설정했습니다. 하지만 여전히 최적화되지 않은 영역에 오류가 집중되어 있어, 이에 대한 개선을 위해 지역 인식 데이터 최적화 프레임워크를 도입하였습니다.

- **Technical Details**: PaddleOCR-VL-1.6은 Under-Optimized Region을 활용한 데이터 엔진을 도입하여 불안정한 결정 경계와 낮은 밀도의 분포를 가진 샘플을 진단합니다. 이를 통해 차별화된 샘플을 선별하고 전문가 합의를 통해 라벨을 수정하며, Iterable Judge-and-Refine 전략을 사용합니다. 또한, 데이터 효율성을 높이기 위해 고유한 GRPO 기반의 데이터 선택 전략을 채택하였습니다.

- **Performance Highlights**: PaddleOCR-VL-1.6은 OmniDocBench v1.6에서 96.33%의 새로운 최첨단 성과를 달성했습니다. 이는 같은 계열의 다른 VLM들과 비교했을 때 강력한 경쟁력을 보이며, 문서 처리에서 높은 성능과 자원 효율을 모두 갖춘 솔루션으로 자리 잡았습니다. 이를 통해 다운스트림 도메인 특정 시나리오에 대한 효율적 적응을 위한 실용적 참조를 제공합니다.



### FreeStreamGS: Online Feed-forward 3D Gaussian Splatting from Unposed Streaming Inputs (https://arxiv.org/abs/2606.03254)
- **What's New**: 본 연구에서는 FreeStreamGS라는 새로운 온라인 피드포워드(Feed-Forward) 프레임워크를 제안하여 3D 뷰 합성(Novel View Synthesis)에서 효율적이고 고품질의 결과를 달성했습니다. 이는 기존의 오프라인으로 기록된 이미지 시퀀스를 활용한 3D Gaussian Splatting(3DGS) 기법의 한계를 극복하기 위한 접근법입니다. 또한, 이 방법은 스트리밍 및 기하학적 정렬이 맞지 않는 이미지 입력에서도 작동 가능하게 설계되었습니다.

- **Technical Details**: FreeStreamGS의 핵심 메커니즘 두 가지는 Decoupled Intrinsic Recovery Head와 Dynamic Point Refinement Offset 전략입니다. 첫 번째 메커니즘은 카메라의 내재적 편향을 제거하여 장시간 스트리밍 동안 장면 스케일의 부정확성을 방지합니다. 두 번째 메커니즘은 결합된 자세-깊이 드리프트를 수정하기 위해 강한 비복원(unprojection) 조건을 완화합니다.

- **Performance Highlights**: 실험 결과, FreeStreamGS는 미래의 프레임에 접근할 수 없는 상황에서도 기존의 최첨단 오프라인 피드포워드 3DGS 방법들과 경쟁하는 렌더링 품질을 달성했습니다. 이는 특히 온라인 환경에서도 이동 중에 이미지 데이터를 처리할 수 있는 능력을 나타냅니다.



### MariData: One-Step Unpaired Image Translation for Maritime Environments (https://arxiv.org/abs/2606.03246)
- **What's New**: 이 논문은 해양 자율 표면선(MASS)의 강력한 인식 시스템을 위한 합성 데이터 생성 프레임워크를 제안합니다. 특히 CycleGAN-turbo라는 일단계 비쌍 이미지 변환 아키텍처를 사용하여, 환경이나 조명이 불리한 조건에서도 작은 항법 객체의 구조적 세부사항을 유지하는 방법을 탐구하고 있습니다. 기존 모델의 제한성을 극복하기 위해 제로 컨볼루션 스킵 연결을 통합하여 VAE 병목현상을 우회하는 접근 방식을 소개합니다.

- **Technical Details**: 이 연구에서는 다양한 대기 도메인 간 장면 전환을 위해 비쌍 이미지의 다양성을 강조한 데이터를 수집하였습니다. 원본 데이터셋은 Unsplash와 Kaggle에서 수집되었으며, 일조, 야경, 안개 등 여러 조명 조건에서 다양한 선박 타입과 해양 상태를 포함하고 있습니다. 데이터 수집 후, 환경 도메인을 정의하는 수작업 정제 과정을 거쳐 CSV 형식으로 주석이 달린 파일을 생성했습니다.

- **Performance Highlights**: 모델의 성과는 정성적인 평가 및 다양한 강도 수준의 추론 연구를 통해 입증되었습니다. 특히, Day-to-Foggy와 Day-to-Sunset 모델은 뛰어난 구조적 보존을 보여주었고, Day-to-Night 모델은 인공지능에서 '시맨틱 환각'이라는 문제를 드러냈습니다. 이러한 결과는 합성 데이터가 해양 항법 알고리즘 훈련에 실질적으로 유용하다는 것을 확인시켜 주었습니다.



### MemoGen: Can Past Experience Improve Future Text-to-Image Generation? (https://arxiv.org/abs/2606.03243)
- **What's New**: 최근 텍스트-이미지 생성 모델들은 강력한 시각적 합성을 달성했지만, 암묵적인 시각적 제약이나 관계적 추론이 필요한 프롬프트에 대해서는 여전히 불안정한 모습을 보입니다. 기존의 방법들은 단기적인 생성에만 초점을 맞추고 있으며, 과거의 성공이나 실패를 지속적으로 저장할 필요성을 간과하고 있습니다. 이 연구에서는 MemoGen이라는 훈련이 필요 없는 프레임워크를 제안하여, 과거 생성 경험에서 지속적으로 개선할 수 있는 가능성을 탐색합니다.

- **Technical Details**: MemoGen은 각 작업을 수행하면서 명시적인 시각적 요구사항을 추론하고, 필요한 경우 외부 증거를 검색하여 실행 가능한 생성 제약으로 변환합니다. 결과를 평가하고, 과거의 실패와 성공 경험을 재사용 가능한 경험 메모리로 저장함으로써 지속적인 개선을 가능하게 합니다. 이 시스템은 이미지 생성 프로세스를 이해, 증거 확보, 시각적 기반 마련, 피드백, 경험 재사용의 주기를 기반으로 설계되었습니다.

- **Performance Highlights**: 광범위한 실험 결과, MemoGen은 오픈 소스 Qwen-Image를 기반으로 하여, 단 두 번의 진화 라운드 만에 Nano Banana Pro 및 GPT-Image-1과 같은 성능 높은 상용 모델들을 초월하는 성과를 보였습니다. WISE에서 MemoGen은 0.91의 점수를 기록하며, 지식 집약적 카테고리에서 가장 우수한 결과를 달성했습니다. 이러한 결과는 MemoGen이 모델 업데이트 없이도 이미지 생성을 효과적으로 개선할 수 있는 능력을 보여줍니다.



### Follow-Your-Preference++: Rethinking Preference Alignment for Image Inpainting (https://arxiv.org/abs/2606.03216)
Comments:
          23 pages, 14 figures. arXiv admin note: substantial text overlap with arXiv:2509.23082

- **What's New**: 이 논문은 이미지 인페인팅(image inpainting)의 선호 맞춤(preference alignment) 문제를 다루고 있습니다. 새로운 방법론을 제안하기보다는 기존의 문제를 기초 원리에서 재조명하고 그 핵심 도전 과제를 재평가합니다. 다양한 공개 보상 모델(reward models)을 사용하여 선호 훈련 데이터를 구축하고, 아홉 개의 보상 모델과 두 개의 벤치마크를 통해 실증 연구를 진행했습니다.

- **Technical Details**: 우리는 직접 선호 최적화(Direct Preference Optimization, DPO) 프레임워크를 사용하여 평가 및 높은 품질의 선호 데이터 구축을 위해 아홉 개의 보상 모델을 채택했습니다. 실험을 통해 다양한 구조와 생성 메커니즘을 가진 두 개의 기본 인페인팅 모델과 함께 정보의 일관성을 유지하면서 다양한 실험을 진행했습니다. 우리 연구 결과는 보상 모델의 신뢰성과 효용성을 평가하며 여러 가지 편향(bias)을 드러냅니다.

- **Performance Highlights**: 선호 맞춤 방법은 객체 제거(object removal) 작업으로 쉽게 전이될 수 있으며, 기존의 아키텍처 변경이나 새로운 데이터셋 도입 없이 이전의 최첨단 모델을 크게 능가하는 성과를 보여줍니다. 간단한 보상 모델 앙상블을 통해 얻어진 결과들은 뛰어난 성능과 일반화 가능성을 나타내며, 인페인팅 결과의 질적 개선을 이끌어 냈습니다. 이 연구는 향후 연구 방향성을 제시하며 간단하면서도 강력한 기준을 설정합니다.



### Reinforcement Learning from Cross-domain Videos with Video Prediction Mod (https://arxiv.org/abs/2606.03201)
- **What's New**: 본 논문에서는 XIPER (Cross-domain Video Prediction Reward)라는 보상 모델을 소개합니다. XIPER는 시각적으로 다른 도메인에서 수집된 전문가 비디오로부터 학습할 수 있도록 하여, 행동자(agent)의 외관 차이에도 불구하고 보상 신호를 생성합니다. 이 모델은 도메인 간 비디오 예측 모델을 학습하여, 전문가 도메인으로의 관찰 변환을 도와주고, 예측 확률을 보상 신호로 활용합니다.

- **Technical Details**: XIPER는 두 가지 구성 요소를 포함하는 도메인 간 비디오 예측 모델을 훈련합니다: (1) 행동자의 관찰을 전문가 도메인으로 매핑하는 도메인 변환 모델과 (2) 과거 프레임의 시퀀스에서 다음 전문가 프레임을 예측하는 비디오 예측 모델입니다. 이 구성 요소들은 오프라인에서 미리 훈련되고, 강화 학습 훈련 중에 동결됩니다. 예측 확률은 보상 신호로 사용되며, 이는 전문가 행동과의 정합성을 나타냅니다.

- **Performance Highlights**: DMC Color Suite(8개 작업) 및 DMC Body Suite(3개 작업)에서의 실험 결과, XIPER는 항상 세 가지 경쟁 기본선에 비해 우수한 성능을 보여 주었습니다. 또한, 시뮬레이션에서만 전문가 비디오를 사용할 수 있는 sim-to-real 전이 데이터셋에서 XIPER의 적용 가능성을 분석하였으며, 실제 로봇 관찰에 대해 유의미한 보상 신호를 생성함을 보여주었습니다. 논문과 관련된 코드 및 데이터셋은 프로젝트 웹페이지에서 확인할 수 있습니다.



### GLINT: Sparsely Gated Vision-Language Alignment for Fine-Grained Radiology Representations (https://arxiv.org/abs/2606.03180)
- **What's New**: 이번 연구에서는 GLINT (Gated Language-Image alignmeNT)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 전통적인 VLMs (Vision-Language Models) 방식의 한계를 극복하여, 이미지의 특정 패치(patch)에 대해 텍스트 쿼리와의 밀접한 관계를 학습합니다. 특히, GLINT는 3D CT(volumes)에서도 마스크 감독 없이 제로샷(zero-shot) 세분화(segmentation)를 성공적으로 수행한 첫 사례입니다.

- **Technical Details**: GLINT는 Sparse Gated Alignment라는 독특한 아키텍처를 통해 텍스트 쿼리와 관련된 패치만을 활성화하는 시그모이드 게이트(sigmoid gate)를 사용합니다. 또한, Dense Feature Regularization을 적용해 훈련 가능한 인코더의 중간 특징(feature)을 고정된 자가 감독 학습(SSL) 모델에 연결함으로써, 게이트가 의존하는 세부 패치 특징을 보존합니다. GLINT는 이를 통해 2D 흉부 X선(CXR) 및 3D 흉부 CT에 모두 적용됩니다.

- **Performance Highlights**: GLINT는 제로샷 클래스 분류(zero-shot classification), 위치 지정(grounding), 세분화(segmentation)에서 우수한 성능을 보이며, 특히 SS(SSL) 인코더 및 기존의 의료 VLMs에 비해 뛰어난 결과를 도출했습니다. 특히, 제로샷 세분화와 위치 지정에서 가장 두드러진 성과를 보여, 쿼리 특정(query-specific) 로컬라이제이션이 강조된 설계 의도를 잘 반영하고 있습니다.



### Ask When It Pays: Cost-Aware Open-Ended Interaction for Instance Goal Navigation (https://arxiv.org/abs/2606.03175)
- **What's New**: 본 논문에서는 비용 민감(cost-sensitive) 불확실성 감소(uncertainty reduction) 문제로서 인터랙션 인스턴스 목표 탐색(Instance Goal Navigation, IGN)을 재정의합니다. 이를 통해 에이전트는 가장 높은 탐색 불확실성 감소를 기대할 수 있는 질문을 선택해야 하며, 이로 인해 효과적인 상호작용을 도모합니다. 논문의 핵심은 유용한 질문과 그 비용이 어떻게 평가되어야 하는지를 명확히 제시하는 것입니다.

- **Technical Details**: 저자들은 기계 학습(Machine Learning)에서 정보 이득(information gain) 분석을 적용하여 다양한 질문 유형을 도출하고 이러한 질문들의 유형별 비용을 설정하였습니다. 이들은 객체 인스턴스 주석, 방 메타데이터 등의 요소를 기반으로 한 인터랙티브 IGN 벤치마크를 구성하여 에이전트가 목표 인스턴스를 식별할 수 있도록 합니다. TANDEM이라는 제로샷(Zero-shot) MLLM 네비게이터를 도입하여 의사결정 단계에서 선택적으로 질문을 하도록 설계되었습니다.

- **Performance Highlights**: TANDEM은 상호작용이 없는 경우와 균일 비용의 인터랙티브 비교보다 가중 성공률(Weighted Success Rate)을 개선했습니다. 특히, 많은 혼동 객체(distractors), 모호한 방, 시각적으로 유사한 객체가 포함된 어려운 에피소드에서 가장 큰 성과를 보여주고 있습니다. 이러한 결과는 효과적인 상호작용이 구조적이며, 서로 다른 질문 유형이 목표 불확실성을 감소시키는 방법에 따라 다르게 활용됨을 보여줍니다.



### JAVEDIT: Joint Audio-Visual Instruction-Guided Video Editing with Agentic Data Curation (https://arxiv.org/abs/2606.03168)
Comments:
          Equal contributions from first two authors. Project page: this https URL Code: this https URL Dataset: this https URL

- **What's New**: 본 연구는 JAVEdit-100k라는 최초의 대규모 고품질 데이터셋을 소개합니다. 이 데이터셋은 인간 중심의 비디오 편집을 위한 지침 기반(joint audio-visual editing) 편집에 특화되어 있으며, 약 100,000개의 편집 쌍을 포함합니다. 또한 JAVEditBench라는 포괄적인 기준을 제안하여 모든 편집 범주에 대해 인간 정렬 지침이 포함된 평가 벤치마크를 생성했습니다.

- **Technical Details**: JAVEdit-100k 데이터셋은 다섯 가지 편집 범주, 즉 주제 편집, 배경 편집, 주제 제거, 주제 추가, 음성 편집을 포함하는 약 100K 편집 삼중항으로 구성됩니다. 이 데이터셋은 Agent-in-the-loop 품질 제어 메커니즘을 통해 자동화된 생성 파이프라인을 통해 정밀하게 생성되어, 시청각 동기화와 지침 이행을 보장합니다. 더불어 JAVEditBench는 시청각 품질, 지침 이행 및 비디오 충실도를 평가하기 위한 세부 기준을 도입합니다.

- **Performance Highlights**: JAVEdit 모델은 JAVEditBench에서 여섯 가지 평가 지표 중 다섯 가지에서 모든 기준 모델을 능가하는 성능을 보여주었습니다. 특히, 가장 강력한 순차 대안 대비 오디오-비주얼 동기화에서 26%의 상대적 개선을 달성하였습니다. 이러한 결과는 조합 모델링(joint modeling)과 에이전트 큐레이션 데이터의 필요성을 입증합니다.



### SRENet: Spectral Re-Entry Network for Point Cloud Action Recognition (https://arxiv.org/abs/2606.03160)
Comments:
          13 pages, 11 figures. Accepted by IEEE Transactions on Circuits and Systems for Video Technology

- **What's New**: 본 논문은 SRENet이라는 새로운 스펙트럼 인식 프레임워크를 제안합니다. 이 프레임워크는 포인트 클라우드( point cloud ) 시퀀스로부터 행동 인식을 위한 글로벌 컨텍스트(global context)와 세부적인 시간 동역학(fine-grained temporal dynamics)을 동시에 학습하는 것을 목표로 합니다. 특히, 주파수 기반 모형(frequency-based modeling)을 활용하여 포인트 클라우드에 대한 행동 이해의 효과iveness를 검증하였습니다.

- **Technical Details**: SRENet은 Spectral Decomposition Block(SDeBlock)과 Spectral Re-entry Block(SReBlock)으로 구성됩니다. SDeBlock은 웨이브릿(wavelet) 기반의 분석을 통해 입력된 포인트 클라우드 시퀀스를 저주파(low-frequency) 및 고주파(high-frequency) 성분으로 분해합니다. 이후 SReBlock은 SDeBlock에서 왜곡된 시간 주파수 구조를 복구하고, 세부적인 시간 동역학을 강화하기 위한 2차 주파수 분해를 수행합니다.

- **Performance Highlights**: MSR-Action3D, NTU-RGBD 및 NTU-RGBD120 데이터셋에서의 실험을 통해 SRENet은 기존의 최첨단 성능(state-of-the-art performance)을 기록했습니다. Frequency modeling을 통해 포인트 클라우드 기반의 행동 이해에서 높은 성능을 구현하는 것을 입증하였으며, 저주파와 고주파를 모두 통합한 효과적인 행동 인식이 가능하게 되었습니다.



### NVIDIA OmniDreams: Real-Time Generative World Model for Closed-Loop Autonomous Vehicle Simulation (https://arxiv.org/abs/2606.03159)
- **What's New**: 오토노머스(Autonomous) 차량 기술이 발전함에 따라, 긴 꼬리(long-tail) 시나리오에서의 안전한 주행 정책 평가가 중요한 과제로 남아있습니다. 새로운 연구에서는 전통적인 심플레이터(simulator)의 한계를 극복하기 위해, 'OmniDreams'라는 혁신적인 생성적 세계 모델을 도입하였습니다. 이 모델은 Cosmos 확산(diffusion) 모델을 기반으로 하여 실시간으로 행동 조건화(action-conditioned) 비디오를 자가 회귀적으로 생성합니다.

- **Technical Details**: OmniDreams는 과거의 프레임과 현재의 시뮬레이터 상태, 즉각적인 주행 행동을 바탕으로 하여 포토리얼리즘(photorealism)을 갖춘 센서 데이터를 생성합니다. 이 모델은 21,000시간의 주행 시나리오를 활용하여 극한의 날씨와 예측 불가능한 동적 행동과 같은 복잡한 현상을 합성할 수 있습니다. 특정 정책 모델인 Alpamayo 1과 AlpaSim 조정기( orchestrator)와 연결되어, OmniDreams는 반응적 환경으로 작동합니다.

- **Performance Highlights**: OmniDreams에서 후속 훈련된 세계-행동 모델(WAM)은 'Physical AI Autonomous Vehicles NuRec' 데이터셋에서 강력한 성능을 보여 주며, VLA 기반의 Alpamayo 1.5 연구 정책 모델을 초월하였습니다. 흥미롭게도, WAM은 전체 매개변수(parameter)의 1/5만 사용하면서도 이전 모델보다 더 뛰어난 결과를 기록했습니다. 이러한 결과는 OmniDreams가 정책 아키텍처의 기초(backbone)로 활용될 잠재력을 강조합니다.



### $A^2$: Smaller Self-Supervised ViTs Localize Better than Larger Ones (https://arxiv.org/abs/2606.03148)
- **What's New**: 본 논문에서는 $A^2$라는 새로운 방법론을 제안하여 작은 self-supervised ViT 모델이 큰 모델에 비해 전경 객체를 더 잘 로컬라이징한다는 사실을 밝혀냈습니다. 이 방법은 주목할 만한 주의 맵(attention map)으로 크롭을 선택한 후, 큰 모델로 이를 임베딩하여 더 풍부한 표현을 추출하는 방식으로 동작합니다. 이 접근은 프리트레인된(pretrained) 피쳐를 이용하고 별도의 그룹 레이블이나 데이터셋별 학습 없이 적용할 수 있습니다.

- **Technical Details**: $A^2$ 방법론은 주목할 모델(attention model)과 추출할 모델(embedding model)을 분리하는 두 단계 프로세스를 갖고 있습니다. 첫째 단계는 이미지에서 높은 주의(attention) 영역을 기반으로 크롭을 선택하는 것이고, 두 번째 단계는 이러한 크롭을 이용해 분류(prediction)를 수행하는 것입니다. 이 방식은 주의에서 소프트 어텐션(soft attention)을 하드 어텐션(hard attention)으로 변환하여 가장 중요하고 관련성 있는 영역만을 다운스트림 표현으로 전달합니다.

- **Performance Highlights**: 실험을 통해 $A^2$는 DFR과 같은 손실 수준 방법들과 경쟁력 있는 성능을 보이며, 강력한 분포 이동(distribution shift) 하에서도 엔드 투 엔드 주의 교육(end-to-end attention training)을 초월하는 결과를 보여주었습니다. 다양한 벤치마크에서 $A^2$는 작은 모델의 우수한 지역화 성능을 기반으로 하여 향상된 다운스트림 작업 수행 능력을 입증했습니다.



### Disentangling Visual and Factual Correctness in LVLMs' Visualization Literacy (https://arxiv.org/abs/2606.03142)
Comments:
          Under review at IEEE Transactions on Visualization and Computer Graphics (TVCG). 23 pages, 9 figures

- **What's New**: 본 연구는 Large Vision-Language Models (LVLMs)의 시각적 해석 능력의 정량화를 시도하며, 기존 평가 방법의 한계를 극복하기 위한 새로운 틀을 제시합니다. 기존의 평가에서는 시각적 올바름(visual correctness)과 사실적 올바름(factual correctness)을 혼합하여 평가한 반면, 본 연구는 이 두 가지를 분리하여 LVLM의 내부 메커니즘을 명확히 합니다. 연구는 15개의 최신 LVLM 모델을 대상으로 실험을 수행하였으며, 시각적 해석의 확실성을 재조명하는 데 기여하고자 합니다.

- **Technical Details**: 연구에서는 Counterfactual Visualization Literacy Assessment Test (CVLAT)라는 새로운 평가 도구를 소개하여 LVLM이 시각적 증거와 사실적 지식 사이에서 어떻게 중재하는지를 측정합니다. 또한, 시각적-사실적 의존 지수(visual-factual reliance index, VFRI)를 기반으로 모델의 성능을 분류하고, 고전적인 정확성 기반 평가 방식의 한계를 지적합니다. 실험을 통해 시각적 재현 및 사실적 지식 간의 상충 조건 하에서 LVLM의 반응을 분석하며, 다양한 프롬프트 개입(prompt intervention)의 효과성을 검토합니다.

- **Performance Highlights**: 실험 결과, 일부 LVLM 모델은 기존 평가에서 인간 수준의 성능에 도달했으나, 이는 사실적 기억(factual recall)을 기반으로 한 결과일 가능성이 높습니다. CVLAT 평가를 통해 시각적 이해에 더 중점을 둔 모델과 사실적 지식에 더 중점을 둔 모델 간의 차이를 명확히 할 수 있었습니다. 특히, 프롬프트 기반 개입이 모델의 우선순위를 변화시킬 수 있지만, 이는 모델 별로 매우 의존적이며 통계적으로 비대칭적이라는 사실도 확인되었습니다.



### KC-3DGS: Kurtosis-Constrained Gaussian Splatting for High-Fidelity View Synthesis (https://arxiv.org/abs/2606.03120)
- **What's New**: 이번 연구에서 우리는 3D Gaussian Splatting (3DGS) 훈련을 자연 이미지 통계 기반의 Wavelet 도메인 감독으로 강화하는 KC-3DGS 방법을 제안합니다. 3DGS는 안방향 가우시안 집합을 활용하여 실시간 새로운 뷰 합성을 실현하지만, 재구성 과정에서 발생하는 구조적 아티팩트 문제로 어려움을 겪고 있습니다. KC-3DGS는 고주파 세부 사항을 강조하고 안정적인 최적화를 도모하며, 기존 방법과 비교해 더 나은 성능을 보여줍니다.

- **Technical Details**: KC-3DGS는 세 가지 주요 컴포넌트를 포함합니다. 첫째, 고주파 세부 사항의 결핍을 명시적으로 처벌하는 다중 스케일 Wavelet 계수 정렬 손실을 도입합니다. 둘째, 실제 이미지와의 주파수 통계에 맞도록 렌더링된 이미지를 발전시키기 위해 감독된 kurtosis concentration 손실을 제안합니다. 셋째, 주파수 전문화를 촉진하기 위한 교차 대역 공분산 패널티를 적용합니다. 이를 통해 3DGS의 기존 아키텍처를 변경하지 않고도 자연스러운 이미지의 특성을 반영할 수 있습니다.

- **Performance Highlights**: KC-3DGS는 MipNeRF360, Tanks&Temples, WRIVA-ULTRRA 등의 다양한 벤치마크에서 일관된 향상을 보여줍니다. 특히 WRIVA-ULTRRA 데이터셋에서는 DreamSim을 9.48% 향상시키고 PSNR, SSIM, LPIPS 역시 개선되었습니다. 훈련 이미지가 12장에 불과한 희소 뷰 환경에서도 PSNR을 최대 0.5 dB까지 개선하며 인식 품질을 유지합니다.



### GuidedBridge: Training-freely Improving Bridge Models with Prior Guidanc (https://arxiv.org/abs/2606.03119)
Comments:
          ICML 2026

- **What's New**: 이 논문에서는 Prior Guidance (PG)라는 새로운 훈련이 필요 없는 브리지 가이던스 방법을 제안합니다. PG는 이전의 방법들이 생성 품질을 개선하기 위해 사용하던 정보 없는 약한 prior를 도입하여, 효과적인 prior 활용을 방해하고 노이즈 제거 결과를 향상시킵니다. 이 과정에서 bridge 프로세스의 기본 메커니즘을 분석하고, 주파수 변조 prior 가이던스(FMPG)를 설계하여 가이던스 스케일을 조정합니다.

- **Technical Details**: 브리지 모델은 교훈적인 클린 prior를 활용하는 데이터-투-데이터 생성 프로세스를 도입합니다. PG는 훈련이 필요 없는 방식으로 약한 prior를 추가하여 prior 활용의 난이도를 증가시킵니다. FMPG는 고주파와 저주파 대역에 맞춰 가이던스 스케일을 조정하는 설계를 통해 브리지 생성의 동적과 일치하게 지식 정보를 활용합니다.

- **Performance Highlights**: 실험 결과, PG 방법이 다양한 이미지 변환 작업에서 사전 훈련된 브리지 모델의 성능을 지속적으로 향상시킴을 보여줍니다. 특히, CFG-FMPG라는 계단식 프레임워크는 CFG를 이용해 근본적인 구조를 복원하고, 이후 FMPG를 활용하여 고품질로 세부 사항을 개선하여 샘플링 효율성을 유지합니다.



### FAF-CD: Frequency-Aware Fusion for Change Detection under Imperfect Multimodal Remote Sensing (https://arxiv.org/abs/2606.03114)
Comments:
          Code will be released at this https URL

- **What's New**: 이번 논문에서는 불완전하고 이질적인 관측을 바탕으로 한 원거리 모니터링을 위한 변화 감지(Change Detection, CD) 프레임워크인 FAF-CD를 제안합니다. FAF-CD는 DINOv3로 사전 학습된 ConvNeXt 인코더와 VMamba 기반의 디코더를 결합하여 정밀하고 효율적인 변화를 감지합니다. 기계학습을 활용하여 주기와 파형 비교를 통해 변화를 감지하는 새로운 모듈인 tri-branch fusion을 도입하여 만약 여름이 들어가도 외부의 왜곡된 정보를 보정할 수 있는 능력을 가집니다.

- **Technical Details**: FAF-CD는 주파수 인식 변화를 탐지하는 네트워크로, Siamese DINOv3 인코더와 rectification-aware 다중 분기 융합 메커니즘, 그리고 VMamba 기반의 디코더를 통합하고 있습니다. 이 구조는 변형 가능한 공간 정렬, 주파수 기반 차이 모델링, Haar 파형 기반의 경계 개선을 포함한 융합 모듈을 통해 다중 스케일 표현을 효과적으로 추출합니다. 이러한 접근법은 공간 정합과 주파수 및 경계 큐를 모두 활용하는 것을 목표로 하고 있습니다.

- **Performance Highlights**: BRIGHT, LEVIR-CD, WHU-CD에서의 실험 결과 FAF-CD는 기존 NeXt2Former-CD에 비해 향상된 tc-mIoU와 tc-mAP를 보였습니다. 이 방식은 M-CD 및 NeXt2Former-CD와 비교하여 두 개의 이진 데이터 세트에서 평균적으로 가장 높은 perturbed cIoU 및 cF1 성능을 보여주었습니다. 또한, NeXt2Former-CD에 비해 약 24 GFLOPs의 계산 비용을 줄였으면서도 정확도를 유지하거나 개선하는 성과를 달성했습니다.



### Inverting the Generation Process of Denoising Diffusion Implicit Models: Empirical Evaluation and a Novel Method (https://arxiv.org/abs/2606.03111)
- **What's New**: 이 논문은 DDIM 이미지 생성 과정을 역전하여 생성된 이미지에서 잠재 변수를 회복하는 문제를 다룹니다. 특히 초기 노이즈 맵을 회복하는 방법을 제안하며, 기존 방법의 정확성 한계를 극복하기 위해 직접적 역전과 고정점 방법을 결합한 하이브리드 접근 방식을 소개합니다. 본 연구는 세 개의 데이터세트를 대상으로 실험을 진행하였고, 초기 잠재 변수의 예측 정확성이 개선되었음을 보여 줍니다.

- **Technical Details**: DDIM에서는 이미지 x_{0}의 생성 과정이 결정론적이며, 이는 최종 생성된 이미지와 초기 잠재 변수 x_{T} 사이의 직접적 관계를 의미합니다. 본 논문에서는 단계적으로 초기 잠재 변수를 추정하는 방법을 고안했으며, 이를 위해 직접적 역전 기술을 첫 단계에 적용하고 이후 고정점 방법을 사용하여 나머지 단계를 처리합니다. 기존 방법들에 대한 비교 실험을 통해 제안한 방법의 우수성을 입증하였습니다.

- **Performance Highlights**: 제안된 방법은 기존 DDIM 역전 방법들에 비해 재구성과 보간(interpolation) 성능에서 일관되게 우수한 결과를 보였습니다. 특히, 기존 방법들은 보간에서는 양호한 성능을 보였으나 재구성에서 낮은 정확성을 보여 주었습니다. 새로운 평가 기준인 self-interpolation 테스트를 통한 결과는 제안한 방법이 모든 지표에서 우월한 성능을 나타내고 있음을 보여 줍니다.



### Zero-Shot 3D Question Answering via Hierarchical View-to-Token Transportation (https://arxiv.org/abs/2606.03100)
Comments:
          19 pages, 6 figures,

- **What's New**: 최근 2D 비전-언어 모델(VLM, Vision-Language Models)을 활용한 제로샷 3D 장면 이해가 주목받고 있으며, 이는 공간적 추론 능력 때문입니다. 연구자는 3D 포인트 클라우드에서 여러 2D 뷰를 샘플링하고, 이를 사전 학습된 VLM에 입력하여 문제를 해결합니다. 여기서 중요한 것은 입력 컨텍스트의 품질을 최적화하고, 제한된 입력 예산 내에서 가능하면 많은 3D 세부정보를 보존하는 것입니다.

- **Technical Details**: 논문에서는 KeyVT라는 계층적 접근 방식을 통해 뷰(view)와 토큰(token) 수준에서 입력 컨텍스트를 수집합니다. 픽셀 특성과 카메라 매개변수를 결합해 뷰의 중요성을 평가함으로써 공간적으로 일관되고 과업 관련 뷰를 생성합니다. 또한, 선택된 뷰 간 패치의 중복성을 줄이기 위해 최적 수송(optimal transport, OT) 프레임워크를 활용하여 대표 토큰을 식별합니다.

- **Performance Highlights**: KeyVT는 세 가지 널리 사용되는 벤치마크에서 평가되었으며, 기존의 튜닝 없는 방법보다 현저한 개선을 보여주었습니다. 성능은 훈련 기반 접근법과 유사하며, KeyVT를 통해 2D VLM이 풍부한 3D 세부정보에 접근하여 효과적으로 3D 장면을 이해할 수 있게 됩니다.



### Hierarchical Federated Learning with Dynamic Clustering and Adaptive Regularization for Robust Infrastructure Inspection (https://arxiv.org/abs/2606.03084)
- **What's New**: 이 논문에서는 구조 건강 모니터링(SHM)을 위한 새로운 계층적 연합 학습 프레임워크인 Clustered-DRAPR를 제안합니다. 이 프레임워크는 매크로 수준에서의 동적 클러스터링과 마이크로 수준에서의 적응형 정규화 방식으로 구성되어 있습니다. 이를 통해 데이터 프라이버시와 보안 규제를 준수하면서도 다양한 구조물 유형에 대한 전문화된 모델을 효과적으로 훈련할 수 있습니다.

- **Technical Details**: 제안된 프레임워크는 두 단계 최적화 전략을 사용합니다. 매크로 수준에서는 동적 그래디언트 기반 클러스터링 메커니즘을 통해 유사한 구조적 손상 패턴을 가진 클라이언트들을 전문화된 그룹으로 자동으로 구분합니다. 마이크로 수준에서는 DRAPR 모듈이 각 클라이언트에 대한 통계적 Non-IID Intensity Score를 계산하여 지역적 업데이트를 효과적으로 조정합니다.

- **Performance Highlights**: 실제 대규모 국가 도로 검사 데이터셋을 활용한 평가 결과, 제안된 계층적 연합 학습 프레임워크는 중복 수준의 이질성을 성공적으로 중화시키며, 특히 극단적인 Non-IID 조건에서도 분류 정확도에서 최첨단 FL 기준보다 뛰어난 성능을 보였습니다. 이로 인해 복잡한 인프라 점검을 위한 강력하고 전문화된 진단 모델을 생성하게 됩니다.



### TGV-KV: Text-Grounded KV Eviction for Vision-Language Models (https://arxiv.org/abs/2606.03075)
Comments:
          Accepted by ICML-2026

- **What's New**: 이번 논문에서는 비전 언어 모델(Vision-Language Models, VLM)의 KV 캐시 메모리 문제를 해결하기 위한 새로운 방법인 텍스트 기반 KV 퇴거(Text-Grounded KV Eviction, TGV-KV)를 제안합니다. 이전의 KV 퇴거 접근 방식들은 주로 언어 모델을 위해 설계되어 VLM의 비전과 텍스트 사이의 고유한 차이를 간과하며 성능 저하를 초래합니다. 본 연구는 텍스트 정보에 기반하여 시각 정보를 평가하고, 이를 통해 VLM의 효율성을 크게 향상시키는 방법을 제시합니다.

- **Technical Details**: TGV-KV는 세 가지 하위 모듈, 즉 텍스트-비전 예산(Text-Vision Budgeting, TVB), 텍스트 가중치 순위(Text-Weighted Ranking, TWR), 텍스트 우선 유지 정책(Text-Prioritised Retention, TPR)으로 구성됩니다. TVB는 각 레이어에 예산을 할당하고, TWR은 텍스트와 비전 간의 중요도를 평가하여 순위를 정합니다. TPR은 예산 내에서 텍스트 KV를 전략적으로 보존하여 정보 손실을 방지하는 역할을 합니다.

- **Performance Highlights**: TGV-KV는 다양한 VLM 모델에서 평가되어, 특히 LLaVA-NeXT 모델에서 VizWiz-VQA 작업에서 99.2%의 정확도를 유지하며, 5%의 극단적인 예산으로 52.6%의 처리량 증가를 달성했습니다. 또한, DocVQA 작업에서는 Qwen3-VL-8B 모델에서 92.5%의 정확도를 유지하며 메모리 사용량을 95% 절감했습니다. 이러한 성과는 TGV-KV의 기술이 기존의 접근 방식보다 효과적임을 입증합니다.



### ROBUST-WT: Robust Uncertainty-aware Segmentation Transform via Whitening and Training Enhancements (https://arxiv.org/abs/2606.03069)
Comments:
          8 pages, 6 figures; code available at this https URL

- **What's New**: 이 논문은 의학 이미지를 위한 범용 세분화 문제에 대한 새로운 기법인 Whitening Transform 기반 확률적 형태 정규화 추출기(WT-PSE)를 다룬다. 특히 다양한 의료 장치와 임상 프로토콜에서 발생하는 성능 저하를 방지하기 위해, feature decorrelation 및 Wasserstein distance 기반의 지식 증류를 사용하여 강력한 교차 도메인 세분화를 구현한다. 원래 WT-PSE 시스템의 네 가지 제한사항을 설정하고, 이를 극복하기 위한 여러 개선책을 제안하였다.

- **Technical Details**: 원래의 WT-PSE 구현은 데이터 증강, 손실 함수, 가중치 조절 전략 및 비교 분석에 대한 네 가지 주요 한계를 가지고 있다. 새로운 방법으로는 domain-adaptive augmentation, 하이브리드 BCE 및 Dice 손실 함수를 결합하여 가장자리 인식 세분화를 개선하고, 커리큘럼 기반 Dice 가중치 스케줄링 전략 및 체계적인 비교 연구를 위한 커맨드라인 제어 플래그를 포함한다. 이러한 개선사항은 최종적으로 세분화 정확도를 높이는 데 기여한다.

- **Performance Highlights**: 제안된 개선 파이프라인은 Fundus optic disc segmentation 벤치마크에서 0.956의 최종 Dice 점수와 13.31의 ASD 점수를 기록하여 baseline인 0.939을 초월하는 성과를 보였다. 이러한 결과는 훈련 레벨에서의 개선이 기본 WT-PSE 구조를 변경하지 않고도 일관된 성능 향상을 제공할 수 있음을 나타낸다.



### FCUS-rPPG: A Fast-Converging Unsupervised Framework for Remote Photoplethysmography via Gradient Oscillation Suppression (https://arxiv.org/abs/2606.03050)
- **What's New**: FCUS-rPPG는 비접촉의 혈액량 맥박 신호(BVP)를 빠르게 수렴할 수 있는 비지도 학습 프레임워크입니다. 이 방법은 혼잡한 표시와 불안정한 기울기로 인해 최적화가 느려지는 기존의 방법들과는 달리, 빠른 수렴과 함께 강력한 일반화 능력을 제공한다고 강조합니다. 또한 FCUS-rPPG는 단 한 에포크(epoch)로 학습 가능하다는 점에서 기존 방법들이 수십에서 수백 에포크가 필요한 것과 차별화됩니다.

- **Technical Details**: FCUS-rPPG는 혈액량 맥박 신호의 다중 스펙트럼 공변성과 저차원 다양체 구조를 결합하여 설계된 스펙트럴리 공유된 백본(bone)를 포함합니다. 이 프레임워크는 신호의 기하학적 특성을 활용하여 최적화 효율성을 향상시키고, 포스트 검증 기울기 마스킹(Post-verification Gradient Masking) 메커니즘을 통해 잘못된 기울기가 최적화 과정에 영향을 미치는 것을 방지합니다. 이를 통해 원치 않는 지역 옵티마로 수렴하는 것을 방지할 수 있습니다.

- **Performance Highlights**: FCUS-rPPG는 다섯 개의 다양한 데이터셋에서 실험을 통해 다른 최신 방법들보다 뛰어난 성능을 입증하였습니다. 단 한 번의 학습 에포크로 기존의 방법들이 요구하였던 수십 혹은 수백 에포크를 필요로 하지 않으면서도, 교차 데이터셋 평가에서 지속적으로 최첨단 성능을 달성했습니다. 이 연구는 비지도 rPPG의 실제 적용에 대한 효율적이고 강력한 해결책을 제공합니다.



### MUSE: A Unified Agentic Harness for MLLMs (https://arxiv.org/abs/2606.03005)
- **What's New**: 이 논문은 MUSE라는 멀티모달 정합 구조 실행 도구를 제안하며, 이는 훈련이 끝난 멀티모달 대형 언어 모델(MLLM)을 감싸고, 재훈련 없이 작업 표현, 시각 처리, 인지 도구 사용, 구조화된 파싱, 결정 확인 및 확인자를 통한 수리 모듈을 포함하는 모듈식 파이프라인을 제공합니다. 기존의 연구가 모델 자체를 개선하는 데 초점을 맞췄다면, MUSE는 모델을 수정하지 않고 주변 실행 구조를 최적화하여 성능 향상을 이끌어냅니다. 이를 통해 기존 모델의 한계를 넘어 새로운 가능성을 모색합니다.

- **Technical Details**: MUSE는 Frozen MLLM을 개선하기 위해 설계된 모듈식 추론 파이프라인을 포함합니다. 이 파이프라인은 인식 및 도구 사용에서 시작하여 구조적 파싱 및 결정적 검증을 거쳐 확인자 유도 수리까지의 전체 추론 경로를 다룹니다. MUSE는 특정 실패 모드를 해결하는 다양한 컴포넌트를 명시적으로 설계하였으며, 백박스 환경에서 모델을 다루어 성능 향상이 모두 하니스 레벨의 개선에 기인함을 보장합니다. 실험에서는 VSP-Grid, BLINK-Jigsaw, CoMT, TIR-Bench 등의 다양한 벤치마크에서 평가하였습니다.

- **Performance Highlights**: MUSE는 네 가지 최첨단 MLLM (GPT-4o, GPT-5.4, Claude Haiku 4.5, Claude Opus 4.7)에서 일관된 성능 향상을 보여주었으며, 특히 도전적인 예제에서 두드러진 개선을 보였습니다. 예를 들어, GPT-4o의 Word Search에서 정확도가 3%에서 21%로 향상되었습니다. 분석 결과, MLLM의 실패는 종종 하니스 레벨의 부족에서 비롯되며, 확인자 유도 수리를 통해 이들 문제를 해결할 수 있습니다.



### Towards Compact Autonomous Driving Perception with Balanced Learning and Multi-sensor Fusion (https://arxiv.org/abs/2606.02979)
Comments:
          This work has been accepted for publication in IEEE Transactions on Intelligent Transportation Systems. this https URL

- **What's New**: 이 논문에서는 다양한 자율주행 인지 과제를 단일 패스로 처리하는 새로운 compact deep multi-task learning 모델을 소개합니다. 이 모델은 semantic segmentation, depth estimation, LiDAR segmentation, bird’s eye view projection을 동시에 수행할 수 있으며, 다른 모델의 지원 없이 독립적으로 작동합니다. 또한, 불균형 학습 문제를 해결하기 위한 adaptive loss weighting 알고리즘을 제공합니다.

- **Technical Details**: 모델은 4개의 RGB 카메라, 4개의 DVS, 1개의 LiDAR를 활용하여 다양한 입력 모달리티를 처리하고 여러 센서의 데이터를 융합합니다. 이 과정에서 데이터 전처리와 센서 융합 기술을 적용하여, 동적으로 변화하는 환경에 대한 더 나은 이해를 달성합니다. 또한, Gradient Normalization (GradNorm) 알고리즘을 수정하여 학습 과정의 균형을 맞추고 성능 향상을 도모합니다.

- **Performance Highlights**: 모델의 성능은 ablation study와 비교 연구를 통해 입증되었습니다. 이를 통해 우리는 더 적은 파라미터로도 더 나은 성능을 유지하며, 빠른 추론 속도와 적은 GPU 메모리 사용량을 달성했습니다. 실험은 3개의 CARLA 시뮬레이션 데이터셋과 1개의 실제 nuScenes-lidarseg 데이터셋에서 일관된 결과를 보여주었습니다.



### Hand Trajectory Fusion for Egocentric Natural Language Query Grounding (https://arxiv.org/abs/2606.02962)
Comments:
          Accepted for the poster session at the Egocentric Vision (EgoVis) Workshop in Conjunction with CVPR 2026

- **What's New**: 이번 논문에서는 Egocentric Natural Language Query(NLQ) grounding을 위한 새로운 접근 방식을 제안합니다. 기존의 방법들은 비디오의 외관과 쿼리를 결합하지만, 손 움직임을 무시했습니다. 그 결과, 손-객체 조작 순간에 대한 쿼리 처리에서 개선된 성능이 나타났으며,+2.54 R1@IoU=0.3을 기록했습니다.

- **Technical Details**: 본 연구는 손 관절의 전통적 속성에 대한 정보를 포함하기 위해 손 경로 인코더를 사용합니다. 이 인코더는 손 스켈레톤을 비디오-텍스트 특성과 융합하여 고도로 의미 있는 동작 함수로 변환합니다. 최종적으로, Temporal Segment Prediction 모듈이 쿼리와 가장 잘 일치하는 정답 범위를 예측합니다.

- **Performance Highlights**: Ego4D NLQ v2 검증 데이터셋에서 수행한 결과, 손-객체 상호작용 쿼리에 대해 가장 큰 성과(+2.54 R1@IoU=0.3)와 양(quantity)/상태(state) 쿼리에 대해서도 유의미한 성과(+4.32 R1@IoU=0.3)가 확인되었습니다. 이는 손 경로가 단순한 외관 이상으로 중요한 단서를 제공한다는 것을 시사합니다.



### The Road Ahead in Autonomous Driving: The KITScenes Multimodal Datas (https://arxiv.org/abs/2606.02956)
Comments:
          28 pages, 21 figures

- **What's New**: KITScenes Multimodal 데이터셋은 유럽의 다양한 도시 환경에서 기록된 새로운 자율 주행 데이터셋입니다. 기존의 데이터셋들이 갖고 있는 센서 품질, 지도 완전성, 지리적 다양성 문제를 해결합니다. 우리가 개발한 이 데이터셋은 고해상도 카메라, 장거리 lidar, 4D 이미징 레이더, 다중 GNSS/INS 로컬라이제이션을 포함한 상태의 센서 조합으로 구성되어 있습니다.

- **Technical Details**: KITScenes Multimodal은 Lanelet2를 사용하여 고안된 HD 지도와 3D 교통 요소를 포함한 고충실도 센서 데이터를 제공합니다. 모든 센서는 하드웨어 동기화 및 고충실도 파이프라인을 통해 처리되어 있으며, 이 덕분에 데이터는 신경 렌더링 및 새로운 뷰 합성(novel view synthesis)과 같은 다양한 응용 프로그램에 적합합니다. 본 데이터셋은 62km²의 면적을 커버하며, 29개의 도로 기능 클래스와 120개의 교통 신호 클래스 등 다양한 요소를 포함합니다.

- **Performance Highlights**: 이 데이터셋은 네 가지 벤치마크를 제공하여 현재 자율 주행 기술의 한계를 드러냅니다. 첫 번째는 온라인 HD 지도 인식, 두 번째는 장거리 깊이 추정, 세 번째는 새로운 보기 합성, 네 번째는 다중 모드의 최종 주행 모델입니다. 나아가, 이 데이터셋은 자율 주행 시스템이 필요로 하는 다양한 지리적 맥락을 포괄하여 연구자들에 의해 높은 평가를 받고 있습니다.



### CAD-to-CT Registration of Cylindrical Objects via Ellipse-Based Axis Estimation (https://arxiv.org/abs/2606.02935)
- **What's New**: 이 논문은 CT 스캔과 CAD 모델 간 정확한 지리적 등록을 위한 새로운 2단계 기하 등록 방법을 제안합니다. 이는 특정 원통형 객체(이온화 챔버)의 독특한 기하학적 특성을 활용하여, 전통적인 강도 기반 방법의 한계와 포인트 기반 알고리즘의 불일치를 극복합니다. 제안된 방법은 비뚤어진 축을 갖는 원통의 엘립스 단면을 분석하여 3D 회전 축을 추정하고, CAD 모델을 CT 스캔과 일치시키기 위해 볼륨 겹침 최적화를 수행합니다.

- **Technical Details**: 첫째로, 엘립스를 엣지 검출된 윤곽에 맞추어 CT 슬라이스에서 회전 축을 추정합니다. 이 과정에서 RANSAC(outlier removal)를 통해 이상값을 제거하고, PCA를 통해 최종 축 방향을 결정합니다. 둘째로, CAD 모델을 복셀화하고, 탐지된 축을 따라 방향을 조정한 후 이동 조정을 통해 CT 스캔과의 볼륨 겹침을 최대화하여 정렬을 수행합니다.

- **Performance Highlights**: 제안된 등록 방법은 경사 및 방향 오류가 $0.1^	extcirc$ 미만으로 유지되는 강력한 결과를 보여주며, 이는 강도 보정이나 특징 매칭 없이 가능하다는 점이 특징입니다. 등록 후 정렬된 CAD 모델은 기계 학습 기반 객체 위치 지정 및 산업 CT 워크플로우에서의 자동 분석 등 다양한 응용 분야에 기초가 되는 지리적 기반 진리(ground truth geometry)를 제공합니다.



### SaluNet: Enabling Total Plasticity in Normalization-Free Deep Networks (https://arxiv.org/abs/2606.02927)
Comments:
          34 pages

- **What's New**: 이번 연구는 BatchNorm과 LayerNorm와 같은 정규화 레이어가 안정적인 훈련을 위해 필수적이라는 기존의 인식을 뒤집고, 이들을 단일 학습 가능한 활성화 메커니즘으로 대체할 수 있음을 보여줍니다. SALU (Saturated Adaptive Linear Unit)를 도입하여, 외부 배치 통계나 어파인 파라미터에 의존하지 않고 내재적인 신호 안정화를 제공하는 학습 가능한 활성화를 제안합니다. 이 연구는 기존 아키텍처에서의 안정성과 가소성의 상호작용을 재조명하여, 모든 구성 요소가 완전하게 학습 가능하도록 만드는 총 가소성(total plasticity)의 개념을 제시합니다.

- **Technical Details**: SALU는 신호 전파를 안정화하기 위해 기하학적 동역학을 활용한 경계가 있는 학습 가능한 활성화 함수입니다. SALU는 BatchNorm 또는 LayerNorm과 같은 전통적인 정규화 레이어를 대체할 수 있으며, SWALU 및 GALU라는 새로운 학습 가능한 변형 활성화를 통해 다양한 아키텍처에 통합됩니다. 결과적으로 SALU, SWALU, GALU는 신호 안정화와 게이팅 기능이 동일한 학습 가능한 원리에서 발생하게끔 하는 통합 프레임워크를 형성합니다.

- **Performance Highlights**: SaluNet-C-18은 CIFAR-10에서 97.35%, CIFAR-100에서 83.25%의 성능을 기록하며, 이는 BatchNorm 기반 벤치마크를 초월하는 성과입니다. 배치 크기가 1일 때도 CIFAR-10에서 93.44%, CIFAR-100에서 76.23%를 유지하며, 정규화 아키텍처가 수렴하지 못하는 상황에서도 안정성을 보입니다. 또한 SaluNet-C-50은 ImageNet-1K에서 78.67%의 Top-1 정확도를 달성하여 짧은 훈련 예산에서도 최첨단 성능을 기록했습니다.



### ATLAS: A Large-Scale Evaluation Benchmark for Adversarial LiDAR Perception (https://arxiv.org/abs/2606.02924)
Comments:
          preprint

- **What's New**: 이번 논문에서는 자율주행 인식 시스템의 평가 방식에서 리얼 월드 환경의 특수성과 도전 과제를 고려해야 함을 강조합니다. 특히, LiDAR 센서에 대한 블랙 박스 공격(black-box attacks)에서 발생할 수 있는 오류를 평가하는 새로운 벤치마크인 ATLAS(Adversarial Temporal LiDAR Attack Suite)를 소개합니다. 이는 리얼 주행 시퀀스를 기반으로 하여 점 주입(point injection) 및 점 제거(point removal) 공격을 시뮬레이션합니다.

- **Technical Details**: ATLAS는 물리적으로 기초한 대규모 평가 벤치마크로, 기존 LiDAR 인식 모델의 취약성을 평가하는 데 초점을 맞추고 있습니다. 이 시스템은 현대 인식 모델의 강력성과 취약성을 동시에 분석하며, 특히 표준 벤치마크에서 높은 성능을 나타내는 모델이 점 제거 공격에는 강하지만 점 주입 공격에는 더 취약하다는 점을 발견했습니다. 이러한 취약성은 표준 객체 데이터베이스 샘플링 증강(sampling augmentations)에서 유래함을 밝혀냈습니다.

- **Performance Highlights**: ATLAS를 통해 현재 가장 앞선 LiDAR 인식 모델들의 성능을 평가한 결과, 모델에 따라 다소 상이한 강건성을 보였습니다. 표준 벤치마크에서 더 높은 성능을 가진 모델들이 제거 공격에 대한 저항력이 강하지만, 주입 공격에는 더 많은 취약점이 발견되었습니다. 이 논문은 블랙 박스 센서 강건성을 고려하여 향후 발전을 위한 기초자료를 제공하고 있으며, ATLAS 생성 코드를 공개하여 지속적인 평가가 가능하도록 지원합니다.



### Pixel Cube: Diffusion-based Portrait Video Relighting Through Realistic Lighting Reproduction (https://arxiv.org/abs/2606.02919)
Comments:
          ACM SIGGRAPH 2026 Journal Track / ACM Transactions on Graphics, 17 pages. Project page: this https URL

- **What's New**: 본 논문에서는 다이나믹 포트레이트 비디오를 사실적으로 리라이팅(relighting)하는 새로운 확산 기반(diffusion-based) 방법을 소개합니다. 이 방법은 다양한 주제 외양과 안면 움직임, 조명 조건을 갖춘 실제 캡처 및 렌더링된 비디오로 구성된 하이브리드(hybrid) 훈련 데이터셋을 활용합니다. 고속 비디오 리라이팅 데이터 획득을 위해 LED 조명 시스템을 구성하여 균일한 조명에서 사실적인 이미지 생성이 가능하도록 합니다.

- **Technical Details**: 모델은 고해상도 비디오 확산 모델을 기반으로 하며, 각 프레임의 HDR(HDR) 환경 맵을 조명 제어로 사용합니다. 주어진 새로운 환경에서 포트레이트 비디오의 색조와 노출 수준을 제어하기 위해 합성된 배경 이미지도 사용합니다. 이 모델은 제공된 새로운 조명 조건 하에서 사실적이고 조화로운 비디오를 생성하면서 피사체의 얼굴 표정 및 미세한 특징을 충실히 보존합니다.

- **Performance Highlights**: 제안된 방법은 사실적이고 시간적으로 일관된 리라이팅을 달성하며, 주제의 신원을 엄격히 보존하는 탁월한 성능을 나타냅니다. 여러 환경 맵을 사용하여 야외 비디오 리라이팅에 대한 광범위한 실험을 수행하여 각종 응용 분야에서의 효과를 입증했습니다. 결과적으로, 제안된 모델은 새로운 주제 외양과 조명 조건에 대한 강력한 일반화 능력을 보여줍니다.



### Any2Poster: Any-Source Poster Generation Across Modalities and Domains (https://arxiv.org/abs/2606.02915)
Comments:
          Project Page: this https URL

- **What's New**: 새롭게 소개된 Any2Poster Bench는 다양한 입력 출처에서 포스터 생성을 평가하기 위한 기준입니다. 이 기준은 PDF, URLs, PPTX, DOCX, Markdown, LaTeX, 노트북, 비디오를 포함한 8가지 입력 모달리티와 연구, 뉴스, 교육, 비즈니스, 픽션의 5가지 콘텐츠 도메인을 아우릅니다. 기존의 평가 방식은 주로 논문 등 단일 형태로 제한되어 있었지만, Any2Poster Bench는 다양한 실제 출처에서 정보를 효과적으로 전달하는 포스터 생성을 측정합니다.

- **Technical Details**: Any2Poster Bench는 정보 충실도(information fidelity)뿐만 아니라 시각적 품질(visual quality), 레이아웃(layout), 가독성(readability), 콘텐츠 완전성(content completeness) 및 논리적 흐름(logical flow)을 평가합니다. 추가로 제공된 Any2Poster Agent는 다양한 입력을 처리하고, 콘텐츠를 조직하며, 포스터 레이아웃을 계획하고, 반복적으로 시각적 피드백을 통해 포스터를 개선하는 전방향 시스템입니다. 이 시스템은 콘텐츠에 적응하여 포스터를 계획하고, 편집이 가능한 HTML/CSS에서 포스터를 렌더링합니다.

- **Performance Highlights**: Any2Poster Agent는 Any2Poster Bench에서 8가지 입력 모달리티에서 평균 87.25%의 정확도를 달성하였으며, 콘텐츠 도메인에서도 평균 87.28%의 정확도를 기록했습니다. PaperQuiz 스타일 평가에서는 PosterAgent-4o보다 51.06-51.33%에서 72.58%로 전체 정확도를 개선하고, 밀도 기반 점수는 116-121에서 145.16으로 증가하였습니다. 이는 Any2Poster Agent가 다양한 입력 전환에서 일반화 능력과 성능을 강화했음을 보여줍니다.



### Tiny Collaborative Inference for Occlusion-Robust Object Detection (https://arxiv.org/abs/2606.02894)
- **What's New**: 이 논문은 IoT 감시 노드와 수색 및 구조(SAR) 플랫폼과 같은 소형 엣지 디바이스에서 컴퓨터 비전 모델을 로컬로 실행하는 방법에 대해 다룬다. 오클루전(occlusion)에 강인한 물체 탐지 파이프라인을 구성하여, 1MB 미만의 SRAM을 사용하는 디바이스에서도 동작할 수 있도록 MCUNet 백본과 YOLOv2 검출 헤드를 결합하였다. 이 연구는 두 가지 협업 추론 전략을 평가하며, 그 중 결정 레벨 융합(Weighted Boxes Fusion, WBF)이 기능 레벨 융합(fusion)보다 높은 성능을 보이는 결과를 도출하였다.

- **Technical Details**: 이 논문은 초저전력 엣지 디바이스에서의 오클루전 강인성(object detection with occlusion robustness)을 연구하며, MCUNet와 YOLOv2를 통합한 경량 물체 탐지 모델을 제안한다. TensorFlow Lite 양자화(quantization)를 통해 메모리와 에너지를 효율적으로 사용하도록 최적화되었으며, 여러 뷰를 통한 협업 추론 설정이 탐색된다. 또한, 비 중앙 집중식 학습(Decentralized Federated Learning, DFL) 기술을 작은 실험으로 포함시켜 동료 디바이스 간의 모델 업데이트가 비-아이디얼 local 데이터에서 안정성을 지키는지를 테스트한다.

- **Performance Highlights**: WBF를 이용한 결정 레벨 융합이 비대칭 오클루전 환경에서 최대 +0.2736 mAP(miss average precision)의 성능 향상을 보였다. 세 뷰에서의 융합을 확장함으로써 정확도가 더 향상되었으며, 약 1.3KB의 통신 오버헤드가 추가되었다. 301.9초의 자율 세션에서 108프레임 중 61프레임에서 융합된 출력을 관찰할 수 있었으며, 이는 단독 Board 2에서 47프레임에 해당하며, 프레임 레벨 커버리지에서 +29.8%의 이점을 제공한다.



### Pathway-Structured Privileged Distillation for Deployable Computational Pathology (https://arxiv.org/abs/2606.02877)
- **What's New**: 본 논문에서는 Mixture of Pathway Experts (MoPE)라는 새로운 지식 증류 프레임워크를 소개하며, 이 모델은 히스토로지(History) 전용 추론을 위한 다중 모달 학습을 재구성합니다. MoPE는 RNA 프로파일과 전체 슬라이드 이미지(whole-slide images, WSI) 간의 부분 관측 가능성(partial observability)을 활용하여 RNA에서 파생된 경로(pathway)를 인코딩하고, 이 정보를 경로 색인(pathway-indexed)을 통한 병리 전공자(pathology experts)에게 전달합니다. 이는 임상 실무에서 RNA 프로파일링의 사용이 제한적이라는 문제에 대한 해결책을 제공합니다.

- **Technical Details**: MoPE는 H&E 이미지를 기반으로 생물표지자(biomarker) 및 생존 예측(survival prediction)을 위한 경로 구조화된 지식 증류 프레임워크입니다. 모델은 H&E WSI와 Hallmark 50 생물학적 경로로 구조화된 RNA 프로파일을 쌍으로 사용하여 훈련됩니다. MoPE는 RNA와 WSI 표현 간의 직접적인 기능 매칭을 강요하는 대신, 공유 메모리 기반을 통해 경로 전문 전문가들을 정렬하여 WSI 분기가 경로 기반의 분자 정보를 보다 부드럽게 학습하도록 합니다.

- **Performance Highlights**: MoPE는 TCGA-BRCA 등의 다양한 공개 벤치마크 및 두 개의 독립적인 유방암 코호트에서 WSI 전용 추정 성능을 일관되게 향상시켰습니다. 실제 평가에서 MoPE는 ODX 예측을 수행하며, OSUWMC에서 80.89% 및 Dartmouth에서 80.45%의 AUC(Area Under Curve)를 달성하였습니다. MoPE의 성능 향상은 생물학적 관측에 기초한 경로 구조화된 지식 증류가 효과적인 경로임을 지지하며, 병리학적 데이터만을 사용한 예측에서도 일관된 분별력을 유지하고 있습니다.



### Principled Reflection Separation via Nonlinear Superposition and Feature Interaction (https://arxiv.org/abs/2606.02831)
Comments:
          23 pages

- **What's New**: 이번 연구에서는 단일 이미지 반사 분리를 다루며, 기존 접근법의 한계를 지적하고 비선형 초합성 모델(learnable nonlinear superposition model)을 도입합니다. 이 모델은 반사와 투과 층의 상호작용을 보다 정밀하게 표현하여 분해 정확도를 향상시킵니다. 또한, 전송과 반사 사이의 쌍방향 종속성을 명시적으로 모델링하는 일반화된 이중 스트림 상호작용 프레임워크를 제안합니다.

- **Technical Details**: 제안된 프레임워크는 활성화, 게이팅(gating), 및 주의(attention) 기반 상호작용 메커니즘을 통합하며 CNN과 Transformer 아키텍처 모두와 호환됩니다. 연구진은 이 모델이 기존의 sRGB 영역에서의 선형 조합 모델의 프레임워크를 탈피하여, 실세계 이미지 신호 프로세싱 파이프라인에서 발생하는 비선형 결합을 포착할 수 있음을 보입니다. 이는 일반화 능력이 뛰어난 강력한 성능을 발휘합니다.

- **Performance Highlights**: 다양한 실세계 벤치마크에서 수행된 실험 결과, 제안된 접근법이 기존 방법들보다 우수한 성능을 달성하고 강력한 일반화 능력을 가진다는 것을 보여줍니다. 반사 분리는 단순한 선형 혼합을 푸는 것이 아니라, 비선형 형성과 상호작용을 학습하는 것이라는 새로운 통찰을 제공합니다. 이러한 연구 결과는 원칙에 기반한 이미지 분해 모델 설계에 대한 새로운 방향성을 제시합니다.



### Automated Report-Derived Oncology VQA Benchmark for Evaluating Vision-Language Models on 3D Medical Imaging (https://arxiv.org/abs/2606.02809)
- **What's New**: 이 논문은 의료 이미지를 위한 Vision-Language Models (VLMs)의 평가에서 임상적으로 기반하고 확장 가능한 자동화 파이프라인을 제안합니다. 기존의 공개 벤치마크는 제공한 답안의 크기에서 제약을 받으며 수작업으로 주석이 달리거나 정보 유출 가능성이 있는 한계가 있습니다. 우리는 개인화된 방사선 보고서와 3D 종양 이미지를 이용하여 두 가지 유형의 질문을 생성하여 데이터 세트를 자동으로 생성합니다.

- **Technical Details**: 자동화된 파이프라인은 개인화된 방사선 보고서와 이미지를 사용하여 중복 및 부정확성을 방지하기 위한 다중 선택 VQA 데이터를 생성합니다. 저자들이 제공한 세 가지 질문 유형은 RADS 스타일 질문과 LLM이 생성한 보고서 기반 질문으로, 이들은 임상 특성에 기반하여 자동으로 생성됩니다. 이를 통해 네 개의 내부 3D 종양 코호트를 대상으로 하는 오염 방지 벤치마크가 마련되었습니다.

- **Performance Highlights**: 여섯 개의 VLM을 제로샷 평가한 결과, 특정 모델이 우위를 보이지 않았고 각 셀에서 상당한 개선 여지가 드러났습니다. 특히, 시각적 증거에 대한 의존성은 데이터셋에 따라 크게 달라지는 것으로 나타났습니다; 간 관련 질문은 실제 이미지를 필요로 하는 반면, 폐 CT 데이터는 이미지 없이도 해결 가능성이 높았습니다. 이는 현재의 VLM들이 개별 이미지를 넘어 시각적 능력을 제대로 평가할 수 있는 가능성을 보여줍니다.



### Cosmos 3: Omnimodal World Models for Physical AI (https://arxiv.org/abs/2606.02800)
- **What's New**: Cosmos 3는 언어, 이미지, 비디오, 오디오 및 행동 시퀀스를 통합하여 처리하고 생성하는 오미모달(omnimodal) 세계 모델의 가족을 소개합니다. 이 모델은 변형기(Transformer) 아키텍처를 기반으로 하여 다양한 입력-출력 구성에 대한 높은 유연성을 지원합니다. 이를 통해 Cosmos 3는 비전-언어 모델, 비디오 생성기, 세계 시뮬레이터 및 행동 모델을 하나의 프레임워크로 통합합니다.

- **Technical Details**: Cosmos 3는 물리적 AI(Physical AI)를 위한 필수 모달리티를 통합하여, 스케일이 가능하고 범용적인 백본을 제공합니다. 이 모델은 다양한 이해 및 생성 작업에서 새로운 최첨단(state-of-the-art) 성능을 기록하였습니다. 코드, 모델 체크포인트, 커리 큐레이션된 합성 데이터셋 및 평가 벤치를 Linux Foundation의 OpenMDW-1.1 라이센스 하에 오픈 소스로 제공하여 연구를 촉진합니다.

- **Performance Highlights**: 적용성 테스트에서 Cosmos 3의 모델은 인공지능 분석(Artificial Analysis)에서 최고의 오픈 소스 텍스트-이미지 및 이미지-비디오 모델로 평가되었습니다. 또한 RoboArena에서 최고의 정책 모델로 선정되었으며, 이는 Cosmos 3의 성능을 입증하는 결과입니다.



### Diagnosis of Human Object Interaction Detectors for Real World Educational Applications (https://arxiv.org/abs/2606.02789)
- **What's New**: 본 논문에서는 복잡한 교육 환경에서 학생 행동을 자동으로 분석하기 위한 Human-object interaction (HOI) 인식의 중요성을 강조합니다. 기존의 SOTA (state-of-the-art) HOI 탐지기가 벤치마크 데이터셋에서는 좋은 성능을 보이지만, 실제 환경에서는 성능 저하가 발생하는 문제를 다룹니다. 이를 해결하기 위해 진단 기반(diagnosis-driven) 프레임워크를 제안하였습니다.

- **Technical Details**: 이 논문은 Mixed-Reality 의료 교육의 일환으로 Critical Care Air Transport Team (CCATT)에서의 HOI 오류 유형 및 원인 분석을 바탕으로 하여, triplet-level HOI 오류 분류법과 오류 요인 귀속 분석을 통합하였습니다. 사전 학습된 HOI 모델을 목표 도메인에 맞추기 위한 진단 기반 정제 전략을 개발하여, HOI 실패 모드를 연구합니다.

- **Performance Highlights**: CCATT 데이터셋에서의 실험 결과, 이 방법을 통해 사전 학습된 CDN 모델의 macro-F1 스코어가 48.6에서 90.2로 증가하였습니다. 이는 상세한 진단 분석이 실제 교육 환경에서 HOI 모델의 목표로운 적응을 지원하는 데 얼마나 중요한지를 보여줍니다.



### GeoDrive-Bench: Benchmarking Region-Specific Multimodal Reasoning in Autonomous Driving (https://arxiv.org/abs/2606.02774)
- **What's New**: 새로운 벤치마크인 GeoDrive-Bench가 소개됩니다. 이 벤치마크는 VLM(vision-language models)이 지역에 맞는 교통 규칙을 이해하는 능력을 체계적으로 평가하는 데 사용됩니다. 연구팀은 6개국에서 표본을 통해 다양한 운전 문화를 반영한 5,053개의 QA 쌍을 수집하였습니다.

- **Technical Details**: GeoDrive-Bench는 인식(perception), 예측(prediction), 계획(planning), 지역 추론(region reasoning) 등 4가지 주요 운전 과제를 집중 조명합니다. 특히, 각 질문은 모델이 비주얼 증거(visual evidence)와 지역 교통 규칙(local traffic conventions)을 바탕으로 올바른 운전 행동을 유추해야 합니다. DriveOPD는 지역별 교통 규칙을 VLM 내부에 주입하는 알고리즘으로, VLM의 매개변수에 지역 교통 지식을 통합합니다.

- **Performance Highlights**: 실험 결과는 VLM의 성능이 지역에 따라 큰 편차를 보이고 있음을 시사합니다. DriveOPD는 기존 모델들의 정확도를 초과하거나 일치시키고, 크로스-카운트리 변동성을 낮추는 것으로 나타났습니다. 전체적인 시사점은 현재 VLM들이 지역 인식 운전 지능이 부족하다는 것입니다.



### From Local Training to Large-Scale Mapping: A Comparative Assessment of Machine Learning and Deep Learning for Transferable Satellite-Derived Bathymetry (https://arxiv.org/abs/2606.02764)
Comments:
          42 pages, 13 figures, 15 tables. Supplementary Information provided as ancillary file (anc/SI.pdf). Code and pretrained weights at this https URL

- **What's New**: 이번 연구는 위성에서 유래한 수심 데이터(SDB)를 기계학습 및 딥러닝을 통해 효과적으로 전이하는 방법을 평가합니다. Sentinel-2 이미지를 사용하여 0-20m의 수심 범위에서 SDB의 정확성을 높이기 위한 네 가지 CNN 아키텍처(ResNet-50, ResNet-101, EfficientNet-B4, ConvNeXt-Large)를 훈련 시켰고, 여러 지역에서의 평가 결과도 포함되었습니다. 특히 Training 시 contiguous reef block을 사용하는 것이 중요한 설계 선택으로 나타났습니다.

- **Technical Details**: 딥러닝은 고차원 관계를 학습하여 기존의 기계학습 모형보다 전반적으로 높은 정확도를 보여줍니다. 본 연구에서는 Pratas Island와 기존의 Great Barrier Reef의 자료를 이용해 훈련 샘플을 구성하고, 0-20m 범위의 수심 데이터를 평가합니다. Smooth Weight Function(SWF)을 이용한 RMSE 손실 함수는 표면 근처 수심에 가중치를 두어 성능을 향상시킵니다.

- **Performance Highlights**: 내부 지역에서 RMSE는 1.15m에서 1.92m로 범위가 있지만, 수심이 3m 이하인 경우 0.26m로 낮아집니다. Cross-regional transfer에서는 Random Forest 모델이 성능이 급격히 저하되는 반면, 딥러닝 모델은 2.46-2.98m의 RMSE로 여전히 견고성을 유지했습니다. 새로운 아키텍처와 사전 훈련된 가중치를 공개하여 새로운 지역으로의 확장을 가능케 합니다.



### MetaWorld: Scaling Multi-Agent Video World Model from Single-view Video Data (https://arxiv.org/abs/2606.02753)
- **What's New**: MetaWorld은 단일 보기 비디오에서 직접 다중 에이전트 비디오 세계 모델을 확장하는 혁신적인 프레임워크입니다. 기존 모델이 단일 관찰자로 제한되어 있었던 반면, MetaWorld는 데이터 부족과 세계 상태 정렬의 두 가지 주요 문제를 해결하여 다수의 에이전트가 상호작용하는 것을 가능하게 합니다. 이를 통해 더 일반화된 오픈 도메인 환경에서 사용할 수 있는 다중 에이전트 비디오 모델링을 가능하게 합니다.

- **Technical Details**: MetaWorld의 핵심 구성 요소로는 Monocular World-State Unrolling (MWSU), Subject-Aware World Generator (SAWG), World-State Alignment (WSA)가 있습니다. MWSU는 단일 보기 비디오에서 카메라 운영자의 에고 모션과 주체의 공간 경로를 명시적으로 분해하여 다수의 에이전트의 동기화된 모션 데이터를 추출합니다. SAWG는 에이전트 식별 이미지에 따라 생성이 조정되는 시뮬레이션을 수행하며, WSA는 정적 기하학적 일관성과 동적 모션 일관성을 보장하기 위해 서로 다른 생성 브랜치를 동기화합니다.

- **Performance Highlights**: MetaWorld는 여러 카메라의 에고 중심 벤치마크에서 월드 일관성, 상호 관찰 가능성 및 아이덴티티 펀들티 metrics(측정 지표)를 통해 평가되었습니다. 실험 결과, MetaWorld는 단일 에이전트 기준선보다 상당히 높은 교차 보기 일관성을 달성했으며, 아울러 단일 보기 생성 품질도 유지하였습니다. 따라서 MWSU와 WSA의 효과를 확인할 수 있습니다.



### Plan2Map: A Multimodal Benchmark for Document-Grounded Geospatial Boundary Reconstruction from Planning Records (https://arxiv.org/abs/2606.02747)
Comments:
          Project page: this https URL. Fabian Degen and Oishi Deb Contributed Equally

- **What's New**: 이번 논문에서는 UK의 계획 기록에서 기계 가독 가능한 경계를 재구성하기 위한 새로운 벤치마크인 Plan2Map을 소개합니다. Plan2Map은 208개의 사례로 구성된 멀티모달(모달) 벤치마크로, 소스 문서를 기반으로 유효한 지리적 경계를 재구성할 수 있어야 합니다. 이는 공지 텍스트(notice text), 일정(schedule), 지도(map) 판, 지도 레이블(map labels), 경계 주석(boundary annotations) 등 다양한 자료를 사용하여 이루어집니다.

- **Technical Details**: 이 논문은 GeoPlanAgent라는 시스템을 제안합니다. 이 시스템은 문서 기반의 지리적 툴로, 증거 추출(evidence extraction), 위치 지정(localisation), 지도 등록(map registration), 경계 분할(boundary segmentation), 투영(projection), 검증(verification)으로 작업을 분해합니다. Plan2Map 데이터셋에서 GeoPlanAgent는 평균 IoU(Intersection over Union) 0.736과 중앙 IoU 0.904를 달성하며, 예측의 67.8%는 IoU가 0.8 이상입니다.

- **Performance Highlights**: GeoPlanAgent는 직접적인 VLM(Visual Language Model)에서 GeoJSON으로의 변환 방식보다 현저하게 우수한 성능을 보여줍니다. 진단 분석에 따르면, 직접 VLM 예측은 신뢰성이 떨어지며, 나머지 오류는 위치 지정과 지도 등록에서 집중적으로 발생합니다. 반면에 감독된 경계 분할은 픽셀 수준의 마스크 품질을 상당히 개선합니다.



### Consistent Yet Wrong: Evidence Insensitivity in Spatial Vision-Language Models (https://arxiv.org/abs/2606.02742)
- **What's New**: 이 논문에서는 로봇 공학, 자율주행 및 소속 AI에서 중요한 공간 추론(spatial reasoning)의 신뢰성을 검증하기 위해 새로운 멀티 뷰 평가 프로토콜인 ViewDiag를 제안합니다. 기존의 비전-언어 모델(VLMs)의 기존 데이터에 대한 안정적인 예측이 기하학적 이해를 반영한다고 가정했지만, 연구 결과는 그 정반대를 보였습니다. ViewDiag는 다양한 장면에서 객체 쌍의 거리를 측정하는 규제된 시스템을 구축하였고, 이는 공간 VLM의 평가를 단순한 정확도 외에 다른 기준으로 가능하게 합니다.

- **Technical Details**: ViewDiag는 Hypersim, ScanNet 및 KITTI360 데이터를 기반으로 구성된 176개 객체 쌍 트랙을 포함하여 다양한 뷰를 갖춘 80개의 장면을 다룹니다. 이 프로토콜은 세 가지 축(axes) 즉, 메트릭 정확도(metric accuracy), 분포적 집중(distributional concentration), 내부 붕괴(internal collapse)를 통해 모델을 평가합니다. 특히, 내제 피쳐 프로브(latent feature probe)는 동일한 출력 뷰 간의 은닉 상태를 비교하여, 불변성(invariance)이 안정적인 표현인지 불안정한 결정 매핑을 반영하는지를 테스트합니다.

- **Performance Highlights**: 저자들은 여러 모델이 높은 일관성에도 불구하고 낮은 정확도를 자주 보이는 경향을 발견했습니다. 이러한 결과는 공간 VLM이 시각적 증거를 충분히 반영하지 못하고 있다는 것을 나타내며, 따라서 안정적인 예측이 기하학적 이해를 반영하지 않음을 경고하고 있습니다. ViewDiag는 공간 VLM의 선택과 검증을 위한 진단 프레임워크를 제공하며, 기하학적 정보나 하이브리드 기준이 여전히 중요해야 하는 위치를 강조합니다.



### AVTrack: Audio-Visual Tracking in Human-centric Complex Scenes (https://arxiv.org/abs/2606.02724)
Comments:
          19 pages, 10 figures, ICML 2026

- **What's New**: 이 논문은 복잡한 환경에서 사람 중심의 오디오-비주얼 인스턴스 분할(AVIS)을 위한 새로운 데이터셋 AVTrack을 소개합니다. AVTrack은 카메라 움직임, 시각적 차단, 다수 인스턴스 등의 도전적인 상황을 포함하여 871개의 비디오 클립으로 구성되어 있습니다. 이 데이터셋은 기존의 단순한 오디오-비주얼 장면에 비해 더 다양하고 복잡한 동적 시나리오를 제공하여 평가의 유효성을 높입니다.

- **Technical Details**: AVTrack은 3,120 개의 고밀도 주석 인스턴스 트랙렛을 포함, 고급 오디오-비주얼 추적 기술을 시험하기 위해 설계되었습니다. 기존의 데이터셋은 정적 장면에 초점을 맞춰 spatiotemporal 모델링을 제대로 평가하지 못했습니다. AVTrack은 사람 중심의 복잡한 동적 장면을 대상으로 하여, 교차 모달 추론 (cross-modal reasoning) 및 robust spatiotemporal modelling을 가능하게 합니다.

- **Performance Highlights**: AVTrack에서 평가된 최첨단 AVIS 방법들은 기존 데이터셋 대비 성능 저하를 보였습니다. 이는 AVTrack이 더 현실적이고 도전적인 벤치마크로 자리 잡았음을 시사합니다. 또한, 미래 연구를 지원하기 위한 모듈형 플러그 앤 플레이의 확장 가능한 기초 프레임워크를 제공하여 커뮤니티의 연구를 촉진할 것입니다.



### COD10K-C: Benchmarking Robustness of Camouflaged Object Detection Under Natural Image Corruptions (https://arxiv.org/abs/2606.02603)
Comments:
          7 pages, 1 figure

- **What's New**: 요즘 카멜레온 물체 탐지 기술은 크게 발전했지만, 기존의 벤치마크들은 주로 깨끗한 이미지에서만 성능을 평가합니다. 현실에서는 블러, 센서 노이즈 등 다양한 왜곡이 존재합니다. 본 논문에서는 COD10K-C라는 새로운 부패 강건성 벤치마크를 제시하며, 8가지 부패 유형과 5개의 심각도 수준을 포함하여 총 81,040개의 평가 쌍을 제공합니다.

- **Technical Details**: 우리는 세 가지 인기 있는 카멜레온 물체 탐지 모델(SINet-v2, PFNet, ZoomNet) 및 경량 모델인 RobustCODLite를 평가했습니다. 모든 모델은 부패된 이미지에서 성능이 감소했으며, 특히 움직임 블러와 가우시안 블러에서 가장 큰 성능 하락을 보였습니다. RobustCODLite는 부패 증강(corruption augmentation), 주파수 우선 가지(frequency-prior branch), 불확실성 일관성 손실(uncertainty-consistency loss) 기법을 사용하여 부패된 상태에서도 높은 정확도를 유지합니다.

- **Performance Highlights**: RobustCODLite는 부패된 이미지에서 92.3%의 Dice 점수를 유지했으며, 이는 SINet-v2(87.7%), ZoomNet(84.8%), PFNet(84.1%)보다 높은 수치입니다. 가장 어려운 부패 조건에서도 RobustCODLite는 깨끗한 데이터에서 성능이 더 좋은 모델들과 동등하거나 이를 초과하는 성능을 보였습니다. 이러한 결과는 카멜레온 물체 탐지의 현실적 성능을 향상시키기 위한 향후 연구에 기여할 것입니다.



### Neuron Populations Exhibit Divergent Selectivity with Sca (https://arxiv.org/abs/2606.03990)
Comments:
          Project page and code: this https URL

- **What's New**: 이번 연구에서는 Rosetta Neurons라는 신경망 내 신경 집단의 예측 가능한 진화를 조사하고, 모델의 크기와 상관없는 전반적인 특성(상실(loss)과 같은)에서 스케일링 법칙을 확장했습니다. 30B 파라미터의 언어 모델과 5B 파라미터의 비전 모델을 분석하여 Rosetta Neurons의 개체 수가 증가해도 전체 뉴런 수의 비율은 줄어들고 있음을 발견했습니다. 또한, Neuron Polarization Effect가 나타나며, Rosetta Neurons는 더 선택적이고 독특한 의미를 가지게 됩니다.

- **Technical Details**: 비교 연구를 통해, 우리는 독립적으로 훈련된 모델에서 반복되는 활성화 패턴을 가진 Rosetta Neurons의 개념을 도입했습니다. 이러한 뉴런들의 연구를 통해 우리는 보편성(universality), 선택성(selectivity), 그리고 전문화(specialization)라는 세 가지 뉴런 수준의 특성을 모델 크기에 따라 분석합니다. 이 연구는 모델이 커짐에 따라 Rosetta Neurons의 수가 예측 가능한 방식으로 증가함을 보이는 초선형(power law) 관계를 발견하였습니다.

- **Performance Highlights**: Rosetta Neurons는 훈련이 계속되는 동안 특정 코드 도메인의 데이터를 필터링하여 뛰어난 정확도로 성능을 발휘함을 보여줍니다. 이러한 효과는 Rosetta Neurons가 특화된 도메인에서 더 선택적이게 변함에 따라 나타나며, 훈련 중 특정 도메인 데이터와 유사한 성과를 보입니다. 연구 결과는 큰 모델 내의 해석 가능하고 공유된 뉴런 수준의 구조를 밝혀내어, 사이즈와 관련된 뉴런의 보편성, 선택성, 전문화의 체계적 변화와 연결합니다.



### Humanoid-GPT: Scaling Data and Structure for Zero-Shot Motion Tracking (https://arxiv.org/abs/2606.03985)
Comments:
          Accepted at CVPR 2026

- **What's New**: 이번 논문에서는 전신 제어를 위한 새로운 GPT 스타일의 Transformer 모델인 Humanoid-GPT를 소개합니다. 이 모델은 10억 스케일의 모션 코퍼스를 기반으로 훈련되어, 기존의 얕은 MLP 트래커와는 달리 고도의 동적인 행동 추적이 가능합니다. 특히 Humanoid-GPT는 2B 프레임의 데이터로 훈련되어, 이전에 보지 못한 모션에 대한 제로샷 일반화(zero-shot generalization)에서 획기적인 성과를 보여줍니다.

- **Technical Details**: Humanoid-GPT는 트랜스포머 구조를 채택하며, causal attention 방식을 사용하여 각 관절에 대한 PD 목표를 예측합니다. 기존의 비인과적(non-causal) 모델링 방법대신, 데이터와 모델 크기에 따라 자연스럽게 확장할 수 있는 구조로 설계되었습니다. 데이터 샘플링 과정에서 Harmonic Motion Embedding (HME) 기법을 사용하여 다양한 모션의 분포를 분석하고, 이를 통해 훈련 과정의 다양성과 균형을 고려하여 모션을 샘플링합니다.

- **Performance Highlights**: Humanoid-GPT는 전신 동작을 제어하는 데 있어 뛰어난 민첩성(agility)과 제로샷 일반화 성능을 보여줍니다. 기존 연구들과 비교했을 때, Humanoid-GPT는 2B 프레임의 모션 코퍼스를 효율적으로 훈련하고, 제로샷 동적 모션 추적에서 높은 성과를 달성했습니다. 이로써, 향후 전신 제어 문제에 대한 새로운 성능 경계를 제시하며, 실제 로봇 하드웨어에서의 적용 가능성도 높였습니다.



### SEAOTTER: Sensor Embedded Autoencoding with One-Time Transcode for Efficient Reconstruction (https://arxiv.org/abs/2606.03940)
- **What's New**: 이 논문에서는 클라우드 로보틱스를 위해 Sensor Embedded Autoencoder와 One-Time Transcode를 결합한 SEAOTTER라는 새로운 압축 프레임워크를 소개합니다. 이는 자원 제약이 있는 센서와 데이터 소비자가 극단적으로 다른 전력 및 대역폭 예산을 지닌 점을 고려하여 설계되었습니다. SEAOTTER는 JPEG 파일과의 호환성을 유지하면서도 비율 왜곡의 균형을 개선하여, 로봇에서 촬영한 고해상도의 비주얼 데이터를 효율적으로 처리할 수 있게 합니다.

- **Technical Details**: SEAOTTER의 작동 방식은 세 가지 주요 단계로 나뉘며, 각 단계를 연결하는 두 개의 압축 비트스트림이 존재합니다. 첫 번째 단계는 센서 내장 분석 변환을 통해 손실 없이 인코딩된 잠재 벡터를 생성하고, 두 번째 단계에서 클라우드 측에서 다시 픽셀 이미지로 합성합니다. 이를 통해 생성된 JPEG 파일은 일반 소비자가 일반적인 JPEG 디코더를 통해 쉽게 해독할 수 있게 설계되었습니다.

- **Performance Highlights**: SEAOTTER를 사용하면 압축률 200:1에서 AVIF와 비교했을 때, 인코딩 속도는 7배, 디코딩 속도는 3.5배 더 빨라지고 ImageNet에서의 top-1 정확도는 +8% 증가하는 결과를 얻었습니다. 이는 전통적인 JPEG의 기반 하에서 하드웨어 및 소프트웨어와 호환성이 유지되면서 달성된 성과입니다. 또한, SEAOTTER는 세 가지 측면의 비대칭성을 통해 클라우드 로보틱스 압축 문제를 체계적으로 해결합니다.



### MAdam: Metric-Aware Multi-Objective Adam (https://arxiv.org/abs/2606.03904)
- **What's New**: 이번 연구에서는 Multi-objective optimization (MOO)와 관련하여 Adam 최적화기와 다양한 MOO 솔버 간의 시스템적인 격차를 분석합니다. 특히, 두 가지 문제, 즉 weighting mismatch와 geometric mismatch를 제기하며, 이를 동시에 해결할 수 있는 새로운 방법인 MAdam (Metric-Aware Multi-Objective Adam)을 제안합니다. MAdam은 기존의 솔버와 최적화기를 변경하지 않고도 이러한 문제를 해결할 수 있는 wrapper로 작용합니다.

- **Technical Details**: MAdam은 시간적 preference vector에 따라 조정된 curvature를 통해 MOO 솔버의 조화된 방향을 사전 조정하고, Adam의 업데이트 절차에 이를 반영합니다. 이 방식은 preference-conditioned diagonal Fisher information matrix를 기반으로 하여, MOO 솔버의 의도와 최적화기의 실행 간의 불일치를 해결합니다. MAdam은 세 가지 주요 기여를 통해 문제를 해결하는데, 첫째, MOO에서의 solver-Adam 간 불일치를 진단하고, 둘째, metric-aware gradient preconditioning을 통해 올바른 curvature를 도출하며, 마지막으로 다양한 MOO 솔버에 대한 실증적 검증을 제공합니다.

- **Performance Highlights**: MAdam은 여러 작업 학습(multi-task learning), Pareto-front 회복, 물리 정보 신경망(physics-informed neural networks, PINN), 의료 이미지 응용 프로그램에서도 Adam보다 지속적으로 우수한 성능을 보여줍니다. MAdam을 적용하는 모든 MOO 솔버 패밀리에서 성능 향상을 관찰했으며, 이는 MOO 솔버의 효과적인 설정과 업데이트 메커니즘을 적절히 결합했기 때문입니다.



### Exploring Adversarial Robustness and Safety Alignment in Multilingual Multi-Modal Large Language Models (https://arxiv.org/abs/2606.03793)
- **What's New**: 이번 연구에서는 Multimodal Large Language Models (MLLMs)의 적대적 저항력 및 다국어 안전성을 12개 다양한 언어를 통해 체계적으로 연구합니다. 기존 연구가 영어 중심으로 발전한 반면, 이 연구는 다국어 모델의 취약성을 평가하며 언어 간 전이 가능성을 탐색합니다. 이를 통해 영어 외의 언어에서도 취약점이 존재함을 확인하고, 특정 언어에서 안전성 문제가 현저하게 드러남을 보여줍니다.

- **Technical Details**: MLLM은 pretrained vision encoders와 LLM 백본을 경량화된 프로젝션 모듈을 통해 결합하여 시각적인 정보와 언어적인 사고를 통합합니다. 연구에서는 LLavA 기반의 open-source MLLM인 Palo와 Parrot 모델을 평가하고, 그 성능을 다국어 기준으로 전이하며, Qwen3-VL 같은 모델과 비교하여 훈련 과정에서 다국어 통합의 중요성을 강조합니다. 최종적으로, 60,000개 이상의 샘플로 구성된 다국어 평가 기준이 작성되었습니다.

- **Performance Highlights**: 연구 결과, MLLM은 단일 언어에서 최적화된 적대적 공격에도 불구하고 다국어로 전이될 수 있는 취약점을 보입니다. Palo와 Parrot 모델은 비영어 사용자 환경에서 안전성의 환상을 드러내며, 비영어 모델에서 낮은 위험 반응 비율이 해로운 지시문을 거부하기보다 놓치는 현상인 'failure-by-safety'를 나타냅니다. 반면, Qwen3-VL은 진정한 다국어 안전성을 보여주며 다양한 언어에서도 높은 거부율을 유지합니다.



### Face versus Body Tracking for Human-Robot Interaction: An Egocentric Datas (https://arxiv.org/abs/2606.03694)
Comments:
          8 pages, 5 figures, 3 tables. Accepted to the 35th IEEE International Conference on Robot and Human Interactive Communication (RO-MAN 2026)

- **What's New**: 이 논문에서는 의미 있는 인간-로봇 상호작용(HRI)을 위한 지속적인 사용자 참여 평가의 필요성을 강조하고 있습니다. 기존의 컴퓨터 비전 모델이 감시나 자율 주행에 최적화되어 있어 사회적 로봇의 특수한 도전에 대해 적절히 대응하지 못한다고 지적합니다. 이를 해결하기 위해 Furhat 로봇을 통해 복잡한 사회적 동적을 캡처한 새로운 맞춤형 egocentric 데이터셋이 소개되었습니다. 또한, 얼굴 추적과 신체 추적 사이의 비교 평가를 통해 문제를 명확히 하고, 새로운 최적화된 파이프라인 개발을 통해 IDSW(신원 전환)를 49% 줄이는 방법을 제시합니다.

- **Technical Details**: 논문은 'Tracking-by-Detection' 패러다임에 기반하여 현대의 공간 추적 시스템에 대한 설명을 제공합니다. 이는 두 단계로 나뉘어 첫 단계에서 감지기를 통해 타겟을 격리하고 두 번째 단계에서 칼만 필터와 헝가리안 알고리즘을 사용하여 프레임 간 링크를 유지하는 방식입니다. 데이터셋 구성은 20개 비디오 시퀀스로, 데이터를 보다 효율적으로 수집하기 위해 로봇 카메라의 프레임 레이트를 10fps에서 약 25fps로 증가시켰습니다. 이 데이터셋은 오랫동안 가려진 대상이나 비선형 움직임 같은 여러 추적 도전 과제를 포함합니다.

- **Performance Highlights**: 연구 결과, 공간 기억을 늘리면 장기간 가림 현상이 완화되는 효과를 보였으나, 복잡한 동적 사건에서는 실패하는 경향을 보였습니다. ReID(출현 재식별)를 통합함으로써 복잡한 신원 전환 문제를 해결할 수 있었지만, 신체 추적의 안정성이 크게 향상되는 반면 얼굴에서의 IDSW가 급증하는 상반된 결과도 나타났습니다. 최적화된 파이프라인을 통해 IDSW가 49% 감소하여 사용자 상호작용의 신뢰성을 높였습니다. 이 연구는 사회적 동적을 직접적으로 캡처하는 필요성을 강조하고 있으며, HRI 인식 모델의 진정한 검증을 위한 중요성을 부각시킵니다.



### Does Language Shift Break Medical Vision-Language Models? Indonesian Radiology Visual Question Answering Case Study (https://arxiv.org/abs/2606.03693)
Comments:
          accepted to MMFM-BIOMED Workshop @ CVPR 2026

- **What's New**: 의료 비전-언어 모델(Medical Vision-Language Models, VLMs)의 성능은 주로 영어 방사선학 질문 응답 벤치마크에서 평가되며, 비영어 임상 언어에서의 강건성은 거의 탐구되지 않았다. 저자들은 인도네시아어로 질문을 던질 때 방사선학적 추론 능력을 유지하는지 평가하기 위해 IndoRad-VQA를 도입하였다. 이 연구는 언어 간의 성능 갭을 평가하며, 영어에서 인도네시아어로의 질문 번역과 관련된 오류 분석을 수행한다.

- **Technical Details**: IndoRad-VQA는 VQA-RAD를 기반으로 하여 구축되며, 2,248개의 질문-답변 쌍과 315개의 의료 이미지를 포함한다. 인도네시아어 번역은 기계 번역을 통해 수행되며, 의학 용어의 일관성을 유지하기 위한 자동 정리 단계를 포함한다. 평가 메트릭으로는 엄격 정확도, 정규화된 정확도, F1 점수, BERT 점수 및 언어 강건성 갭(Language Robustness Gap, LRG)을 사용한다.

- **Performance Highlights**: 모델 간의 성능 차이를 나타내는 결과는 인도네시아어 설정에서 8%에서 25%까지의 성능 갭을 보여주며, 모든 모델에서 강력한 성능을 발휘하는 영어 환경에서의 학습이 인도네시아어 설정에서는 동일하지 않다는 사실을 나타낸다. 오류 분석 결과, 대부분의 오류는 용어와 시각적 추론에 관련되어 있으며, 예/아니오 질문의 잘못된 응답이 주요 언어 유도 오류로 나타났다. 이는 현재의 개방형 모델들이 영어 중심의 언어 편향을 극복하지 못하고 있음을 시사한다.



### PHASER: Phase-Aware and Semantic Experience Replay for Vision-Language-Action Models (https://arxiv.org/abs/2606.03598)
Comments:
          12 pages, 5 figures

- **What's New**: 본 논문에서 우리는 PHASER (PHase-Aware and Semantic Experience Replay)라는 새로운 지속 학습 프레임워크를 소개합니다. PHASER는 기존의 Experience Replay (ER) 방식이 가진 한계, 즉 phase starvation 및 budget misallocation 문제를 해결합니다. 이 프레임워크는 각 서브 스킬에 대해 동등한 메모리 지원을 보장하고, 고위험의 역사적 phase를 동적으로 우선시하는 다중 모달 간섭 라우팅 전략을 사용합니다.

- **Technical Details**: PHASER는 Semi-Markov Decision Process (SMDP) 관점에서 VLA 지속 학습을 재해석하여, 일관된 트래젝토리 데이터와 균일한 프레임 수준 재생 간의 구조적 불일치를 드러냅니다. 그 과정에서 intra-task phase-centric capacity allocation과 inter-task multi-modal interference routing이라는 두 가지 데이터 측면 원칙을 적용하여 서브 스킬 간 버퍼 지원을 균등화하고, 잊혀지기 쉬운 역사적 phase에 대한 집중 리허설을 이루도록 합니다.

- **Performance Highlights**: PHASER는 여러 VLA 백본(OpenVLA-7B, QwenGR00T-3B, QwenOFT-3B)과 LIBERO 지속 학습 스위트를 통해 광범위하게 평가되었습니다. 이 프레임워크는 메모리 제약 조건에서 평균 성공률(ASR)을 최대 31% 향상시키고, LIBERO-Goal CL 설정에서 최종 ASR을 87.8% 달성했습니다.



### IdEst: Assessing Self-Supervised Learning Representations via Intrinsic Dimension (https://arxiv.org/abs/2606.03338)
Comments:
          ICML 2026

- **What's New**: 본 연구에서는 Self-supervised learning (SSL)의 표현 학습을 평가하기 위해 인트린식 차원 (Intrinsic Dimension, ID)을 추정하는 IdEst라는 새로운 방법을 제안합니다. 기존의 linear probing 기법의 한계를 극복하고, 효율적인 하이퍼파라미터 선택을 통해 계산 비용을 절감할 수 있습니다. 본 연구는 SSL 표현의 기하학적 구조를 평가하는 데 있어 인트린식 차원이 유용한 지표가 될 수 있음을 강조합니다.

- **Technical Details**: IdEst 방법은 Minimum Spanning Tree dimension estimator (dim_{MST})를 활용하여 표현 공간의 인트린식 차원을 추정합니다. 이 과정에서 로그-로그 선형 회귀(log-log linear regression)를 사용하여 다양한 크기의 서브샘플에서 ID를 추정합니다. IdEst는 전체 데이터셋 크기의 50,000개의 서브샘플을 이용하여 지표를 계산하며, Ripser를 통해 거리 행렬 계산 및 정렬을 최적화합니다.

- **Performance Highlights**: IdEst는 다양한 데이터셋과 아키텍처에서 linear probe 성과와 강한 상관관계를 보여주었습니다. 대규모 및 소규모 데이터셋에서 SSL 모델의 전반적인 품질을 평가하기 위해 널리 받아들여진 linear probing 방법을 적용하였으며, 우수한 결과를 나타냈습니다. 또한, 하이퍼파라미터 선택 과정에서 계산 비용을 크게 절감할 수 있음을 입증했습니다.



### SagaQA: A Multi-hop Reasoning Benchmark for Long-form Narrative Understanding in TV Series (https://arxiv.org/abs/2606.03301)
- **What's New**: SagaQA는 TV 시리즈에 대한 다중 단계 추론(multi-hop reasoning)을 평가하기 위한 새로운 장기 비디오 벤치마크입니다. 기존의 비디오 이해 기준에서는 인접한 클립이나 짧은 비디오에 대한 이해를 강조했으나, SagaQA는 전체 에피소드를 아우르는 고급 내러티브(complex narrative) 이해를 요구합니다. 이 데이터셋은 모델들이 서로 다른 에피소드 간의 먼 정보를 연결해야 하는 긴 거리 추론(long-range reasoning hops)을 필요로 한다는 점에서 차별화됩니다.

- **Technical Details**: SagaQA 데이터셋은 다중 비디오(multi-video), 다중 단계(multi-hop), 다중 모드(multi-modal) 질문-답변(QA) 쌍을 포함합니다. 각 질문은 20개의 연속 에피소드로 구성되어 있으며, 약 20시간의 비디오를 다룹니다. 질문에 답하기 위해 평균 4회의 추론 단계를 필요로 하며, 관련 사건은 최대 20개의 에피소드가 분리되어 있을 수 있습니다. 데이터셋은 LLM 기반 필터링을 통해 다중 단계 추론 기준을 만족하도록 구성됩니다.

- **Performance Highlights**: 하이브리드 계획자(hybrid planners)는 Parallel 및 Sequential 계획자에 비해 더 높은 성능을 보여주며, 후보 비디오 세그먼트의 폭넓은 탐색과 가장 관련성이 높은 세그먼트에 대한 집중적 추론을 결합함으로써 에피소드의 정확한 기반을 더욱 완전하게 달성합니다. SagaQA는 복잡하고 고차원적인 내러티브(narrative) 이해를 평가하는 데 있어 귀중한 통찰을 제공하며, 비디오 이해의 미래 방향에 관한 중요한 시사점을 제시합니다.



### Do Real-World Datasets Contain Natural Experiments? An Empirical Study Using Causal Feature Selection (https://arxiv.org/abs/2606.03251)
- **What's New**: 이 연구는 자연에서 어떤 개인이나 집단에 영향을 미치는 사건이 존재한다는 사실을 통해 자연 실험(natural experiments)을 탐구합니다. 특히, COVID-19 팬데믹을 예로 들어 이러한 자연 실험이 실제 데이터셋에서 발생하는지, 그리고 어떻게 처리해야 하는지를 질문합니다. 연구자들은 causal discovery를 이용해 데이터에서 자연 실험을 발견하는 방법을 제안합니다.

- **Technical Details**: 자연 실험을 감지하기 위해, 연구자들은 데이터의 기본 인과 그래프를 복원하고 인과적 연결을 기반으로 feature selection을 수행합니다. 이 과정에서 데이터가 관측(observational) 데이터가 아니라 개입(interventional) 데이터로 취급했을 때 모델 성능이 개선되면, 이는 해당 데이터셋에 자연 실험이 존재한다는 것을 나타냅니다. 또한, 합성 그래프를 이용한 데이터셋 시뮬레이션을 통해 이 가설을 검증합니다.

- **Performance Highlights**: 실제 데이터셋에 대한 체계적인 평가를 통해, 연구 결과는 많은 현실 데이터셋이 자연 실험을 포함하고 있음을 보여주고 있습니다. 이러한 자연 실험을 활용하여 인과 추론(causal inference) 기술을 통해 모델 성능을 개선할 수 있음을 입증합니다. 본 연구는 이 분야에 대한 초기 탐구를 대표하며 향후 연구를 위한 기초 자료를 제공합니다.



### Effect of Demographic Bias on Skin Lesion Classification (https://arxiv.org/abs/2606.03214)
Comments:
          Accepted for publication at the Journal of Machine Learning for Biomedical Imaging (MELBA) , 26 pages, 12 figures

- **What's New**: 이번 연구에서는 ResNet 기반의 convolutional 모델을 사용하여 피부 병변 분류 성능을 평가하고, 훈련 데이터의 인구통계학적 편향이 특히 환자의 성별과 연령에 미치는 영향을 분석합니다. 선형 프로그래밍(linear programming)을 통해 인구통계학적 특성이 통제된 데이터셋을 생성하여 편향 효과를 체계적으로 조사합니다. 그리고 단일 작업 모델, 강화된 다중 작업 모델 및 적대적 학습(adversarial learning) 방식 등 세 가지 학습 전략의 효과를 분석했습니다.

- **Technical Details**: 연구 결과, 성별 기반 훈련 데이터셋을 사용했을 때 모델 성능이 최적화되었으며, 훈련 데이터에 남성 환자를 포함하면 남성 하위 집단에 대한 성능이 개선되는 것을 확인했습니다. 강화 학습과 적대적 학습 전략은 균형 잡힌 데이터셋과 여성 우세 데이터셋에서 편향 격차를 좁히거나 제거했지만, 남성 우세 설정에서는 여전히 남성이 여성보다 높은 성능을 보였습니다. 나이 기반 분석을 통해서는 세 가지 모델 접근 방식에서 유사한 기준 성능을 유지하였으나, 연령 카테고리가 증가함에 따라 성능이 점차 감소하는 경향이 있음을 발견했습니다.

- **Performance Highlights**: 젊은 연령대에서 가장 높은 성능을 달성하는 것으로 나타났으며, 균형 잡힌 훈련이 최적의 결과를 낳았지만, 나이가 많아질수록 성능은 감소했습니다. 성별로 인한 편향은 주로 데이터 불균형에서 발생하며, 연령 편향은 모든 데이터 분포와 관계없이 젊은 그룹을 일관되게 선호하는 경향을 보였습니다. 연구에 따르면, 이러한 다양한 메커니즘을 완화하기 위해서는 특정한 개선 전략이 필요하다고 결론지었습니다.



### Inference-Time Scaling for Joint Audio-Video Generation (https://arxiv.org/abs/2606.03183)
Comments:
          Accepted by Transactions on Machine Learning Research (TMLR). Project page: this https URL

- **What's New**: 이 논문은 **Inference-Time Scaling(ITS)**를 결합하여 조화로운 오디오-비디오 생성을 위한 새로운 접근 방식을 제시합니다. 저자들은 단일 목표 가이드의 한계를 극복하기 위해 다중 검증자 프레임워크의 필요성을 강조하며, 다양한 품질 측면에서 균형 잡힌 개선을 달성할 수 있는 최적의 검증자 조합을 식별합니다. 또한, 본 연구에서는 서로 다른 보상 신호를 효과적으로 집계하기 위한 **Adaptive Reward Weighting(ARW)** 알고리즘을 소개합니다.

- **Technical Details**: 조화로운 오디오-비디오 생성은 오디오-비주얼 쌍의 조인트 분포를 모델링하는 것을 목표로 합니다. 본 논문에서는 확산 모델을 활용하여 점진적인 노이징 프로세스를 역전시켜 두 가지 모드(오디오 및 비디오)를 동시에 합성합니다. ITS는 주어진 데이터에 대해 더 나은 샘플을 검색하는 추가적인 계산을 통해 성능을 향상시키고, 이를 위해 보상 신호와 검증자를 사용하여 출력 품질을 향상시키도록 설계되었습니다.

- **Performance Highlights**: 실험 결과는 VGGSound와 JavisBench-mini 벤치마크에서 제안된 프레임워크가 생성된 출력의 의미적 일치성, 지각 품질 및 오디오-비디오 동기화를 크게 향상시킨다는 것을 보여줍니다. 이러한 결과는 특히 다중 목표를 관리하는 것이 ITS 접근법의 성공에 중요하다는 것을 강조합니다. 따라서 본 연구는 조화로운 오디오-비디오 생성의 발전을 위한 실용적이고 효과적인 접근 방식으로 ITS의 사용 가능성을 시사합니다.



### Learning to See via Epiretinal Implant Stimulation in silico with Model-Based Deep Reinforcement Learning (https://arxiv.org/abs/2606.03118)
Comments:
          18 pages, 6 figures. Published version: Biomed. Phys. Eng. Express 10, 025006 (2024)

- **What's New**: 이번 연구에서는 비대칭적인(axon bundle) 신호를 생성하여 시각 이미지 확보를 위한 새로운 방법인 stroke-based rendering을 제안합니다. 강화 학습(Deep Reinforcement Learning) 환경인 rlretina에서 가상의 환자의 망막에 적용되어, 이미지를 구성하는 데 사용됩니다. 나아가, 기존의 방법들이 axon bundle 자극을 줄이려 하였다면, 본 연구는 다양한 형태를 활용하여 보다 우수한 시각 인지를 목표로 합니다.

- **Technical Details**: 연구는 딥 강화 학습 에이전트를 훈련하여, 동형(isotropic) 및 비대칭(axon bundle)이 포함된 형태로 이미지를 구성토록 합니다. 에이전트는 실험적으로 검증된 축삭 맵 모델을 사용하여 가상의 환자가 지각할 수 있는 이미지를 렌더링합니다. 점차적으로 훈련된 에이전트는 전통적인 방법에 비해 더 이해하기 쉬운 이미지를 생성하는 데 성공합니다.

- **Performance Highlights**: 본 연구에서 개발된 방법은 Naive Stimulation Algorithm(NAS)보다 더 나은 성과를 보이며, 다양한 가상의 환자들에 대한 이미지 이해도를 향상시켰습니다. 특히, 에이전트가 생성하는 이미지의 질은 기존의 비효율적인 자극 방식보다 월등히 향상되어, 인공으로 복원된 시력의 질 개선에 기여할 것으로 기대됩니다.



### MARIO: Motion-Augmented Real-Time Multi-Sensor Inertial Odometry (https://arxiv.org/abs/2606.02996)
Comments:
          CVPR 2026 Findings

- **What's New**: 이번 연구에서는 인간의 운동 역학을 기반으로 한 관성 측정 유닛(Inertial Measurement Units, IMUs)와 학습된 포즈 사전(pose prior)을 통합하여 새로운 관성 오도메트리(inertial odometry) 모델을 제안합니다. 이 모델은 특히 Nymeria 데이터 세트와 같은 복잡한 일상 활동에서의 위치 드리프트(positional drift)를 감소시키며, 기존 방법에 비해 최대 36%의 향상을 보여줍니다. 또한, 상용 증강 현실(AR) 안경에 이미 탑재된 가벼운 센서에서 오는 보조 신호를 결합하는 센서 융합(framework)이 long-term 성능을 향상시키는데 기여합니다.

- **Technical Details**: IMU는 선형 가속도와 각속도를 측정하는 소형 저비용 센서로, 최근의 학습된 관성 오도메트리 방법은 IMU 신호에서 직접 학습된 데이터 기반 모션 프라이어(motion prior)로 드리프트를 완화합니다. 본 논문에서 제안한 PoseNet은 단일 헤드 마운트 IMU로부터 SMPL 신체 모델을 기반으로 전체 신체 포즈를 예측하며, 이 포즈 사전은 동작 추정을 위한 물리적으로 의미 있는 구조를 주입합니다. 또한, 센서 융합 모듈을 통해 다양한 보조 신호를 수집하여 성능을 더욱 향상시킵니다.

- **Performance Highlights**: 기존의 IO 아키텍처에 통합된 우리의 모듈은 Nymeria, Aria Everyday, TLIO 데이터 세트에서 지속적으로 드리프트를 줄이고 변환 오류를 낮추는 성과를 보였습니다. 특히, 최악의 상황에서도 42%까지 드리프트를 줄이면서 다양한 움직임 조건에서도 로버스트성과 일반화를 향상시킵니다. 결과적으로 우리의 연구는 인간 모션 역학과 다중 모드 감지를 통합하여 정확하고 견고한 카메라 없는 인간 추적을 위한 새로운 패러다임을 제시합니다.



### SCOPE: Real-Time Natural Language Camera Agent at the Edg (https://arxiv.org/abs/2606.02951)
Comments:
          9 pages, 4 figures, 6 tables. Accepted at HRI '26 (21st ACM/IEEE International Conference on Human-Robot Interaction), Edinburgh, Scotland, March 16--19, 2026. Code: this https URL

- **What's New**: 이 논문에서는 로봇 공학에서 자연어 기반의 언어 주도 에이전트를 배포하기 위한 새로운 평가 방법인 SCOPE(Simulation and Camera Operations for Perception and Evaluation)를 제안합니다. 이 시스템은 Blender 기반의 시뮬레이션 환경과 실제 PTZ(팬-틸트-줌) 카메라에서 작동하며, 언어 모델과 감지 및 제어 도구를 결합하는 것을 목표로 합니다. SCOPE는 536개의 작업으로 구성된 벤치마크를 통해 높은 시뮬레이션과 실제 환경 간의 전이 가능성을 보장합니다.

- **Technical Details**: SCOPE는 분리된 설계를 채택하여, Compact SLM이 고수준의 계획자로서 카메라 제어 및 감지 쿼리와 reasoning을 조정합니다. 시각 이해는 호출 가능한 도구로 노출된 Lightweight VLM에 위임되어, 실제 시간의 엣지 지연에서 visual perception과 tool-based control을 동시에 달성하기 어렵다는 현실을 반영합니다. 이 시스템은 각 요청 후 VLM으로부터 얻은 결과를 활용해 임무를 반복적으로 수행하는 방식으로 작동합니다.

- **Performance Highlights**: SCOPE의 평가 결과, 강력한 SLMs를 사용 시 헐리케인 문제(hallucinations)를 줄이고 도구 경로 설정(tool routing)을 개선하여 신뢰할 수 있는 닫힌 루프(closed-loop) 동작을 생성할 수 있다는 것을 확인했습니다. 또한, Mixture-of-Experts 모델을 통해 비교적 적은 메모리 사용과 지연 시간으로 밀집 대안(dense alternatives)에 비해 일관된 성능을 나타냈습니다. 양자화(Quantization)를 통해 추가적인 효율 향상이 이루어졌으며, 이는 실시간 PTZ 제어의 실제 심리학적 설계 포인트를 제시합니다.



### BYORn: Bootstrap Your Own Responses to Defend Large Vision-Language Models Against Backdoor Attacks (https://arxiv.org/abs/2606.02947)
Comments:
          Accepted to ICML 2026

- **What's New**: 이번 연구에서 제안하는 BYORn은 백도어 공격에 강건한 미세 조정 프레임워크로, 기존의 모델들이 취약한 점을 개선하기 위한 접근 방식입니다. BYORn은 독성 응답이 시맨틱적으로 비정상적이라는 사실을 활용하여, 해당 응답을 동적으로 대체할 수 있는 대안을 생성합니다. 이를 통해 공격자와 목표 출력 간의 상관관계를 차단하여 보다 안전한 모델을 제공합니다.

- **Technical Details**: BYORn은 가설적 깨끗한 응답(latent clean response)과 오염 지표 변수(poisoning indicator variable)를 도입하여 합리적인 목표 함수(objective)를 도출합니다. 이 방법은 훈련 데이터에서 낮은 가능성(likelihood)을 가진 응답을 필터링하고, 나머지 데이터로 모델을 미세 조정하는 과정을 포함합니다. 이를 통해 모델은 지속적으로 개선되며, 안전성과 성능 간의 새로운 균형을 이루게 됩니다.

- **Performance Highlights**: 실험을 통해 BYORn은 기존 방어 방법들에 비해 공격 성공률을 평균 40pp 감소시키면서도 깨끗한 작업 성능을 잘 유지함을 보여줍니다. 또한 모든 평가 모델과 공격 설정에서 BYORn은 강건성과 일반화의 균형을 잘 이루는 파레토 최적(Pareto optimal) 성능을 제공합니다. 이는 BYORn이 다양한 공격 유형에 효과적으로 대응할 수 있음을 시사합니다.



### BEAST3D: Animal behavioral analysis and neural encoding from multi-view video via Gaussian splatting (https://arxiv.org/abs/2606.02937)
- **What's New**: 이번 연구에서는 BEAST3D라는 새로운 자가 지도(pretraining) 프레임워크를 제안하여, 라벨이 없는 다중 뷰 비디오로부터 3D 시각적 표현을 학습합니다. 기존의 지도학습(pose estimation) 방법들은 비효율적인 수동 주석(annotation)을 요구하는 반면, BEAST3D는 카메라 파라미터를 직접 활용하여 소수의 뷰로도 3D 구조를 효과적으로 재구성합니다. 따라서, 실험 환경에서의 한계를 극복하고 더욱 정교한 분석을 가능하게 합니다.

- **Technical Details**: BEAST3D는 비전 트랜스포머를 활용해 3D Gaussian splats를 예측합니다. 이 모델은 또한 동물과 배경을 구분하여 세분화(segmentation)하며, 이 과정에서 고품질의 전경 표현을 생성합니다. 이 프레임워크는 뷰의 수가 적더라도 연구 결과를 전달할 수 있는 유연한 백본(backbone)을 제공합니다.

- **Performance Highlights**: BEAST3D는 쥐, 쥐, 딱다구리, 그리고 인간을 포함한 4종의 종에 대해 평가한 결과, 4개의 뷰만으로도 개선된 재구성을 보여주었습니다. BEAST3D의 특성은 새로운 뷰의 합성, 다중 뷰 포즈 추정, 신경 인코딩 등의 세 가지 downstream 작업에서 경쟁력 있는 성능을 나타냅니다. 이러한 결과는 BEAST3D가 다중 뷰 실험 비디오에서 풍부한 3D 특성을 추출하는 다양한 프레임워크임을 입증합니다.



### Depth from Dual Differential Defocus and Stereo Consensus (https://arxiv.org/abs/2606.02906)
- **What's New**: 본 논문에서는 D^3S Consensus라는 물리 기반의 폐쇄형 알고리즘을 소개합니다. 이 알고리즘은 depth-from-defocus (DfD)와 stereo를 통합하여 카메라의 depth-of-field (DoF) 이상으로 정밀한 깊이 추정을 가능하게 합니다. D^3S Consensus는 쌍의 이중 초점 스테레오 이미지로부터 새로운 DfD 이론인 Dual Differential Defocus(D^3)를 사용하여 과잉 결정된 깊이 세트를 추정하고, 이를 물리적으로 독립적인 단서 간의 동의(consensus)를 통해 신뢰할 수 없는 추정을 거부합니다.

- **Technical Details**: D^3S Consensus는 두 가지 상보적인 단서로부터 깊이를 추출하는 방법입니다. 첫 번째는 카메라가 동시에 두 가지 구성 변화를 겪을 때 관측되는 미세 초점 차이를 기반으로 하는 새로운 이미지-깊이 관계인 D3입니다. 두 번째는 표준 스테레오 방정식입니다. 이러한 두 단서 간의 동의를 시행하여, D3 또는 stereo 단서 중 하나만 사용할 때보다 훨씬 더 정확한 깊이 추정을 가능하게 합니다.

- **Performance Highlights**: D^3S Consensus 프로토타입은 4mm의 짧은 기준선(Baseline)과 12mm의 유효 초점 거리(EFL)를 가지고 있으며, 0.3-1.64m의 거리에서 1cm의 평균 절대 오차를 가진 900 x 1800 픽셀의 깊이 맵을 생성합니다. 이 시스템은 기존의 삼각측량 기반 깊이 추정 시스템에 비해 10배 작은 기준선에서 유사한 작동 범위를 달성하여 더 컴팩트한 제품을 가능하게 합니다. 이러한 성능은 상용 스테레오 카메라보다 우수하여 산업계에서도 큰 주목을 받고 있습니다.



### SVHalluc: Benchmarking Speech-Vision Hallucination in Audio-Visual Large Language Models (https://arxiv.org/abs/2606.02642)
Comments:
          Accepted at CVPR 2026

- **What's New**: 본 연구에서는 audio-visual LLMs의 한계 중 하나인 speech-vision hallucination을 탐구하고, 이를 평가하기 위한 새로운 벤치마크인 SVHalluc를 소개합니다. 기존의 지표들이 환경 소음에 집중하는 반면, SVHalluc는 사람의 언어가 가진 복잡한 의미와 시간 구조를 반영합니다. 이 벤치마크는 speech와 visual 정보 간의 상관성을 정확히 평가할 수 있는지 측정하여 더 깊이 있는 이해를 추구합니다.

- **Technical Details**: SVHalluc는 두 가지 주요 측면, 즉 semantic hallucination과 temporal hallucination을 기반으로 설계되었습니다. 연구자들은 입력 비디오와 결합된 인간의 음성을 기반으로 모델이 올바른 답변을 생성할 수 있는지를 평가합니다. 각 차원에 대해 세 가지 상호 보완적인 진단 작업이 설계되어, 오차 모드를 심층적으로 분석할 수 있습니다.

- **Performance Highlights**: 실험 결과, 오픈 소스 audio-visual LLMs는 SVHalluc에서 매우 낮은 성능을 보였으며, 이는 음성과 시각적 증거 간의 정렬에서 실패를 나타냅니다. 반면, Gemini 2.5 Pro는 이러한 모델들보다 이점을 보이며, 두 모델 간의 성능 차이가 크다는 것을 시사합니다. 이러한 발견은 현재 음성과 비디오 이해의 신뢰성을 향상시키기 위한 필요성을 강조합니다.



### Sparse-View Lung Nodule Volumetry from Digitally Reconstructed Radiographs via AReT: Anatomy-Regularized TensoRF (https://arxiv.org/abs/2606.02639)
- **What's New**: 본 논문에서는 TensoRF에 적용할 때 발생하는 새로운 실패 모드를 식별하고 해결합니다. 기본 밀도 이동값인 -10이 밀도 기울기를 억제하여 적은 수의 X-ray 투사로 인해 발생하는 의학적 재구성을 방해합니다. 밀도 이동값을 0으로 설정함으로써 기울기 흐름이 복원되고 폐 결절에 대한 안정적인 볼륨 재구성이 가능해집니다.

- **Technical Details**: 아시아-정규화된 텐서 방사장 프레임워크인 AReT를 제안하여 LIDC-IDRI 데이터셋에서 세 가지 직각 X-ray 투사(관상, 시상, 축 단면)를 사용한 폐 결절 재구성을 수행합니다. 이 프레임워크는 철저한 ℓ1​ + TV 정규화를 실시하여 최소한의 투사(N=3)로부터도 안정적인 3D 재구성을 가능하게 합니다. 실험 결과는 해부학 정보를 고려한 정규화가 생성된 우선 접근 방식보다 우수함을 보여줍니다.

- **Performance Highlights**: AReT는 방사선의 합의적인 세그멘테이션과 비교하여 임상적으로 적용 가능한 결절(≥10 mm)에서 Pearson r=0.983의 성과를 보이며, 중간 절대 볼륨 오차는 11.4%에 불과합니다. 또한, AReT는 시스템적 바이어스가 거의 없고 구형 볼륨 근사법에 비해 8.4배의 개선을 달성합니다. 이 연구는 제한된 환자 집합에서의 증거 기반 연구로, 향후 연구에서 이를 검증할 필요성을 인정합니다.



### Wavelet as Tokenizer: Preliminary Results on a Shared Wavelet Token Schema for Natural Signals (https://arxiv.org/abs/2606.02631)
Comments:
          12 pages, 3 figures

- **What's New**: 이번 연구는 오디오, 이미지, 비디오가 각각의 모달리티에 국한된 잠재 그리드를 이용하는 대신, 공통의 wavelet token schema를 공유할 수 있는지를 탐구합니다. Haar DWT/IDWT 프론트엔드, 공유된 계수 토큰 레이아웃, 선택적 구조 메타데이터 등을 포함한 초기 연속 토큰 모델을 소개합니다. 이 모델은 Speech Commands, EuroSAT RGB, DAVIS 2017 데이터를 이용한 실험에서 각각 39.92 dB, 29.37 dB, 23.93 dB의 PSNR을 기록했습니다.

- **Technical Details**: 이 논문은 Wavelet as Tokenizer (WAT)라는 다중 스케일 토큰화 프레임워크를 제안합니다. 각 모달리티는 샘플링된 또는 매개변수화된 필드로 간주되며, wavelet 변환이 로컬화된 다중 스케일 계수로 매핑됩니다. 계수 블록과 그 구조적 메타데이터를 결합하여 토큰을 형성하며, 이를 통해 각 신호의 변화를 나타내는 간결한 정보를 제공합니다.

- **Performance Highlights**: 여러 시험 결과는 공통의 토큰 트렁크가 오디오, 이미지, 비디오를 복원하는 데 효과적임을 보여주었습니다. 또한 고정 비율의 에너지 선택은 평균 PSNR을 상당히 향상시킨다는 강력한 기준선을 제공합니다. 마스크드 스파스 훈련은 50%의 밀집 토큰으로 34.45 dB 비디오 PSNR에 도달하였으며, 이는 통합된 wavelet 토큰 스키마의 유용성을 뒷받침하고 있습니다.



### Graph Mamba Survival Analysis Based on Topology-Aware ordering (https://arxiv.org/abs/2606.02602)
- **What's New**: 이번 논문에서 제안된 새로운 Graph Mamba 생존 분석 프레임워크인 TopoMamSurv는 Mamba 모델의 계산 병목 현상을 해결하기 위해 토폴로지 인식 정렬 전략을 도입했습니다. 이 프레임워크는 이미지의 양방향 공간 구조를 활용하기 위해 양방향 Mamba 모듈을 설계하고 Graph Convolutional Network (GCN)을 통합하여 이미지의 특성을 효과적으로 모델링합니다. 또한, 실험 결과는 제안된 전략이 그래프 데이터의 상관관계를 보다 잘 포착함을 입증했습니다.

- **Technical Details**: 논문에서 제안하는 토폴로지 인식 정렬(TAO) 전략은 그래프 내 두 노드 간의 최단 경로를 검색하여 스캐닝 순서를 구성합니다. 이는 그래프 내 이웃 노드들이 실제 이웃 관계를 유지하면서도 장거리 상관관계를 포착할 수 있게 합니다. 또한, 이 다중 분기 그래프 Mamba 기반 아키텍처는 GCN과 양방향 기술을 결합하여 WSI 기반 생존 분석을 위해 설계되었습니다.

- **Performance Highlights**: 이 프레임워크는 5개의 TCGA 데이터셋에서 기존의 최첨단 방법들과 비교하여 종합적인 성능 우위를 검증했습니다. 실험 결과에 따르면, 새로운 스캐닝 전략이 전체 성능 개선에 기여하며, 그래프의 지역 및 장거리 상관관계를 더 잘 반영했습니다. 또한, 이러한 접근법은 정확한 생존 예측을 위한 강력한 기반을 제공합니다.



### PaintBench: Deterministic Evaluation of Precise Visual Editing (https://arxiv.org/abs/2606.00188)
Comments:
          Project Page: this https URL

- **What's New**: PaintBench는 정밀한 하나의 답변 편집 작업을 평가하기 위한 동적으로 확장 가능한 벤치마크입니다. 이 벤치마크는 기하학적 변환, 구조적 조작, 색상 변경, 기호적 추론 등 네 가지 범주에서 20개의 기본 정밀 시각 편집 작업을 목표로 하고 있습니다. 특히 PaintBench는 무한대에 가까운 오염 저항형 평가 스위트를 제공하며, 픽셀-레벨 평가를 통해 편향이 있는 평가 모델에 대한 의존도를 없앴습니다.

- **Technical Details**: PaintBench는 시드(seed)와 매개변수 구성을 기반으로 문제를 절차적으로 생성하는 방식으로 구성되어 있습니다. 각 문제는 입력 이미지, 자연어 지침 및 고유한 정답 이미지로 이루어진 세 가지 요소로 구성됩니다. 이 벤치마크는 20개의 작업에 대해 12개의 문제를 생성하며, 총 1,920개의 문제를 제공합니다.

- **Performance Highlights**: 11개의 이미지 편집 모델을 평가한 결과, 가장 성능이 좋은 모델조차도 17.1%의 낮은 성과를 보였습니다. 기하학적 변환, 공식 기반 색상 변경, 구조적 조작 작업은 모두 모델 전반에서 일관되게 어려운 것으로 나타났습니다. PaintBench의 점수가 데이터 시각화 편집 작업인 TinyGrafixBench의 성과와 강한 선형 상관관계를 보이는 것이 확인되어, PaintBench가 응용 작업에서의 능력을 포착한다는 점이 강조되었습니다.



New uploads on arXiv(cs.AI)

### Imaginative Perception Tokens Enhance Spatial Reasoning in Multimodal Language Models (https://arxiv.org/abs/2606.03988)
- **What's New**: 이번 연구에서는 Imaginative Perception Tokens (IPT)를 도입하여 비전 언어 모델(VLM)이 보이지 않는 관점에서도 정보를 인식할 수 있도록 지원하는 방법을 제시합니다. 이러한 IPT는 대체 공간 구성에서 VLM이 인식할 수 있는 내용의 중간 인식 표현을 외부화하며, 이는 관찰된 입력과 일관성 있는 정보를 제공합니다. 새로운 데이터셋과 세 가지 과제를 통해 이러한 능력을 연구하였고, VLM의 공간 추론을 개선하는 데 기여합니다.

- **Technical Details**: 연구에서는 Perspective Taking (PET), Path Tracing (PT), Multiview Counting (MVC)의 세 가지 공간 추론 작업을 제안합니다. 이 작업들을 수행하기 위해 약 2만 개의 예제와 진리 데이터가 포함된 데이터셋을 구성하였으며, IPT 감독 하에 모델 훈련이 이루어졌습니다. IPT는 중간 인식 예측을 사용하여 VLM의 공간적 구조를 더 잘 이해할 수 있도록 돕습니다.

- **Performance Highlights**: 무엇보다도, IPT 감독을 사용했을 때 공간 추론 성능은 눈에 띄게 향상되었습니다. MVC 작업에서 IPT는 정확도를 3.4% 개선하고, PT에서 강력한 비공개 모델에 비해 경쟁력 있는 성능을 달성했습니다. 이러한 결과는 VLM의 공간적 추론을 향상시키는 유용한 경로임을 시사합니다.



### Entropy Is Not Enough: Unlocking Effective Reinforcement Learning for Visual Reasoning via Vision-Anchored Token Selection (https://arxiv.org/abs/2606.03937)
- **What's New**: 본 연구에서는 시각적 추론(visual reasoning)에 대한 토큰 수준 엔트로피(token-level entropy)의 신뢰성이 떨어진다는 사실을 보여줍니다. 이를 통해 기존의 텍스트 기반 강화 학습(text-only reinforcement learning)에서 입증된 메커니즘이 시각적 정보와는 다르게 작동한다는 점을 강조합니다. VEPO(Vision-Entropy token-selection for Policy Optimization)라는 새로운 강화 학습 프레임워크를 도입하여, 시각 민감성과 토큰 엔트로피를 통합하는 방안을 제안합니다.

- **Technical Details**: VEPO는 각 토큰의 시각적 의존성을 정량화하기 위해 두 개의 신호를 곱셈적으로 결합합니다: (i) 분포 변화를 위한 Jensen–Shannon divergence (JSD)와 (ii) 방향에 구애받지 않는 변화인 절대 엔트로피 차이 |ΔHt|입니다. 이러한 결합은 aleatoric-epistemic 불확실성 분해에 기초하며, 시각적 신호의 누락을 방지하여 유용한 토큰을 포괄적으로 포착할 수 있습니다. 최종적으로 VEPO는 시각적으로 고정되고 매우 유익한 토큰에 의해 정책 업데이트를 조정합니다.

- **Performance Highlights**: VEPO는 Qwen2.5-VL-7B/3B-Instruct 모델을 사용하여 평가했으며, 7B 스케일에서 기존의 최고 엔트로피 기준선을 +2.28점 초과하여 성과를 거두었습니다. 또한, 3B 스케일에서도 일관된 성과 향상이 나타났습니다. 광범위한 실험을 진행하였으며, ablation 연구를 통해 제안한 방법의 유효성과 타당성을 입증하였습니다.



### Hedge-Bench: Benchmarking Agents on Hard, Realistic Tasks Pertaining to Financial Reasoning (https://arxiv.org/abs/2606.03918)
Comments:
          Dataset and evaluation harness available at this http URL

- **What's New**: AI 에이전트는 재무 분석의 기계적 작업을 점점 더 잘 처리할 수 있지만, 공개된 질문을 통한 판단력 발휘는 금융 전문가의 고유한 작업입니다. 이에 따라 우리는 전문 헤지펀드 분석가의 명시적 추론 과정을 기반으로 한 102개의 실제 작업으로 구성된 'Hedge-Bench 1.0'(Hedge-Bench 1.0) 벤치마크를 소개합니다. 이 작업들은 전문가의 단계에 대해 결정론적인 채점을 가능하게 하여 AI의 재무 비즈니스 분석력을 평가하는 새로운 기준을 제시합니다.

- **Technical Details**: Hedge-Bench는 금융 전문가가 수행하는 개방형 문제 주변의 현실적인 판단 작업을 평가하기 위한 도구입니다. 이 벤치마크는 금융 분석가가 실제로 사용할 정보 출처로 구성된 환경과 함께, 분석가가 추론해야 할 개방형 주제를 포함하고 있습니다. 각 작업은 두 명의 헤지펀드 분석가가 공동으로 제작한 명시적인 추론 흔적에서 파생된 결정론적 채점 기준을 가지고 있습니다.

- **Performance Highlights**: 최신 모델의 성과는 Hedge-Bench 1.0에서 16% 미만으로 평가받고 있으며, 작은 모델은 9% 미만의 점수를 기록했습니다. 이는 높은 정확성이 요구되는 금융 판단 작업의 난이도를 반영합니다. 이러한 결과는 에이전트의 추론이 인간 전문가의 판단과 얼마나 밀접하게 연결되는지를 평가하는 데 중요한 기준이 됩니다.



### scTranslation: A Comprehensive Benchmark for Single-Cell Multi-Omics Modality Translation (https://arxiv.org/abs/2606.03906)
- **What's New**: 이 논문에서는 단일 세포에서 다중 오믹스 모달리티를 동시에 측정하는 방법을 소개합니다. 이를 통해 세포 상태와 조절 메커니즘에 대한 포괄적인 이해가 가능해집니다. 또한 scTranslation이라는 체계적인 벤치마크를 제공하여 모델 성능을 평가하고, 다양한 데이터셋과 평가 지표를 통합합니다.

- **Technical Details**: scTranslation의 핵심은 여러 다중 오믹스 데이터셋과 최신 모델을 포함하며, 모델의 성능을 평가하기 위해 클러스터링 기반 지표, 회귀 기반 지표 및 분포 레벨 지표를 사용할 수 있도록 설계되었습니다. 다양한 시나리오에서 모델 성능을 평가하여 각 요소가 모델 성능에 미치는 영향을 분석합니다. 특히, 특성 선택, 단기 학습(few-shot learning) 및 다양한 잡음 수준에서의 모델의 견고성을 평가합니다.

- **Performance Highlights**: 제시된 벤치마크를 활용하여 대규모 연구를 수행하고, 현재의 방법들과 유용한 발견들을 보고하여 향후 연구의 새로운 가능성을 열었습니다. 벤치마크는 오픈 소스 형태로 제공되며, 다양한 연구자들이 접근하여 사용할 수 있도록 배포됩니다.



### Reasoning Structure of Large Language Models (https://arxiv.org/abs/2606.03883)
Comments:
          Accepted at ICML 2026 and presented at the ICLR 2026 workshop on LLM reasoning

- **What's New**: 이 논문에서는 Large reasoning models (LRMs)을 평가할 때 기존의 최종 답변 정확도(final-answer accuracy)나 토큰 수(token count)와 같은 지표들이 동일할지라도, 근본적으로 다른 추론 구조를 숨길 수 있다는 한계를 설명합니다. 이러한 한계를 극복하기 위해 논리 퍼즐(logic puzzles)의 스케일러블 LRM 벤치마크를 도입하고, 비구조적 추적(unstructured traces)을 검증 가능한 추론 그래프(reasoning graphs)로 변환하는 파이프라인을 제시합니다.

- **Technical Details**: 새롭게 제안된 방법은 추론을 구조적이고 측정 가능한 객체로 변환하여 그 위상(topology)을 정량적으로 분석할 수 있게 합니다. 이를 기반으로 모델의 논리적 흐름(logical flow)이 얼마나 집중되어 있는지를 정량화하는 추론 효율성 메트릭(reasoning efficiency metric)을 정의했습니다. 또한, 이 메트릭을 통해 높은 성능의 모델을 평가하는데 필요한 구조적 측정(structural measurements)을 제공합니다.

- **Performance Highlights**: 오픈 소스의 추론 모델들을 분석한 결과, 토큰 수와 정확도가 혼동할 수 있는 행동을 구조적 측정이 구분할 수 있음을 보여주었습니다. 이는 실패 모드를 진단하고 퍼즐의 난이도에 따라 추론이 어떻게 확장되는지를 비교하는 데 실용적인 도구가 됩니다.



### PyraMathBench: Evaluating and Improving Mathematical Capability in Large Language Models (https://arxiv.org/abs/2606.03858)
- **What's New**: 새로운 PyraMathBench (PMB) 벤치마크는 32,505개의 질문으로 구성되어 있으며, 7,404개의 수학 단어 문제에서 파생된 것으로, 4개의 핵심 인지 측면과 14개의 하위 카테고리, 2개의 방식(모달리티)을 포함합니다. 기존의 벤치마크들은 수치 처리와 수학적 추론을 통합하여 LLMs(대형 언어 모델)의 성능을 평가하는 데 부족함이 있어, PMB가 이러한 손실을 해소할 것으로 기대됩니다. 이 연구는 솔루션 최적화 및 학습 기반 모듈(SOLVE)과 상호작용 상대 정책 최적화(IRPO)와 같은 새로운 접근 방식을 통해 LLMs의 수학적-수치적 시너지를 강화합니다.

- **Technical Details**: PMB의 구성은 복잡한 수학적 작업을 여러 계층의 인지 측면으로 나누어 LLM의 성과를 더 잘 평가할 수 있도록 합니다. 이 벤치마크는 모델이 처리하는 각 하위 구성 요소에 대한 숙련도를 평가하는 데 중점을 두며, 수학적 사고 과정에서의 여러 단계를 명확히 나누어 성능 진단을 용이하게 합니다. PMB는 4개의 계층적 측면(A1-A4)에 따라 정리되며, 각 측면은 14개의 독립적인 작업을 포함합니다.

- **Performance Highlights**: 실험 결과, 기존의 LLM들은 특정 수치 계산 및 추상적 수치 질문을 처리하는 데 많은 어려움을 겪고 있음을 보여주었습니다. 개선 사항으로는 SOLVE 모듈과 IRPO 알고리즘을 통해 전통적인 LLM에 비해 5.0%의 성능 개선이 달성되었습니다. 이를 통해 PMB는 LLM의 성능을 종합적으로 평가할 수 있는 강력한 도구로 자리매김할 것으로 보입니다.



### EvoDS: Self-Evolving Autonomous Data Science Agent with Skill Learning and Context Managemen (https://arxiv.org/abs/2606.03841)
Comments:
          Accepted by KDD2026

- **What's New**: EvoDS는 데이터를 과학적으로 처리하는 자율 에이전트를 위한 새로운 접근 방식을 제시합니다. 이 시스템은 스스로 진화하며 기술 습득과 긴 호흡의 맥락 관리 기능을 통합하여 에이전트가 경험을 축적할 수 있도록 돕습니다. 특히 Autonomous Skill Acquisition(ASA) 메커니즘과 Adaptive Context Compression(ACC) 전략을 통해, 에이전트가 실행 가능한 기술을 합성하고 재사용할 수 있으며, 긴 맥락을 효과적으로 관리할 수 있습니다.

- **Technical Details**: EvoDS는 계층적 다중 에이전트 아키텍처 내에서 기술 습득과 맥락 조절을 통합합니다. 이 아키텍처는 각 전문 에이전트가 세부 작업에 대한 데이터를 관리하고, 모델링 및 시각화와 같은 서브태스크를 수행하도록 합니다. 두 단계의 에이전틱 강화 학습(RL) 프레임워크를 통해, EvoDS는 감독된 미세 조정(SFT) 후 온라인 RL을 통해 작업 성능과 기술 습득 및 맥락 관리 최적화를 동시에 수행합니다.

- **Performance Highlights**: EvoDS는 네 가지 다양한 벤치마크에서 기존의 오픈 소스 데이터 과학 에이전트에 비해 평균 28.9%의 성능 향상을 이루었습니다. 또한, EvoDS는 맥락 제한 조건에서의 견고한 장기 성능을 제공하며, 토큰 초과 실패를 제거했습니다. 이러한 결과는 EvoDS의 설계가 데이터 과학 분야에서 지속적으로 개선될 수 있는 잠재력을 가지고 있음을 보여줍니다.



### BigFinanceBench: A Workflow-Grounded Benchmark for Financial-Research Agents (https://arxiv.org/abs/2606.03829)
- **What's New**: 본 논문에서는 BigFinanceBench라는 새로운 벤치마크를 소개합니다. 이 벤치마크는 928개의 금융 연구 업무 질문이 포함되어 있으며, 각 질문은 전문가가 작성한 정확한 답변과 함께 체크 가능한 단계로 나누어진 평가 기준을 제공합니다. BigFinanceBench는 최종 결과물뿐만 아니라 전체 유도 과정을 평가하여, 분석가가 실제 업무에서 수행하는 복잡한 과정을 반영합니다.

- **Technical Details**: BigFinanceBench는 분석가가 수행하는 개방형, 다원 소스, 가정 의존적인 작업을 필요로 하는 질문으로 구성되어 있습니다. 각 질문은 진짜 참고 답변과 함께 포인트가 가중된 루브릭을 포함하여 독립적으로 검증 가능한 단계로 유도 과정을 분해합니다. 이러한 평가 기준은 주체 식별, 출처 선택, 항목 검색, 회계 조정, 공식 건설과 최종 통합을 포함하고 있습니다.

- **Performance Highlights**: 현재의 선진 및 오픈 가중 에이전트를 평가한 결과, 최고의 시스템도 58.8%의 루브릭 점수에 불과했습니다. 최종 응답 정확도는 유도 품질을 나타내는 유용한 지표이지만, 그만큼의 정보가 결여되어 있음을 나타냅니다. 모델의 성능은 금융 업무 간 비균일하게 변하며, 이는 기존 모델 평가에서 간과된 몇 가지 부분이 존재함을 보여줍니다.



### Calibrating Urban Traffic Simulation from Sparse Road Observations via Genetic Optimization (https://arxiv.org/abs/2606.03823)
- **What's New**: 이 논문은 전기차 충전소 배치와 같은 인프라 계획을 위한 도시 교통 시뮬레이션의 필요성을 다루고 있습니다. 기존의 데이터 한계를 극복하기 위해 유전자 알고리즘(genetic algorithm)을 기반으로 한 새로운 프레임워크를 제안하여, 제한적인 도로 관측을 통해 도시 교통 시뮬레이션을 보정합니다. 실제 직업 위치 데이터 없이도 최적화된 작업 분포를 생성할 수 있어, 다양한 도시에서 쉽게 적용할 수 있습니다.

- **Technical Details**: 이 방법론은 SUMO (Simulation of Urban MObility) 시뮬레이션 플랫폼을 사용하여 노스캐롤라이나주 그린스보로의 데이터를 분석합니다. 저자들은 제한된 도로 데이터를 통해 직무 분포와 차량 통행 매개변수를 최적화하여 시뮬레이션된 교통 흐름이 실제 측정값과 잘 일치하도록 합니다. 이러한 접근 방식은 훈련에 사용되지 않은 도로 구간에서도 일반화가 가능하며, 현 데이터와 비교해도 괜찮은 정성적 합의를 보입니다.

- **Performance Highlights**: 이 연구는 최소한의 실제 데이터를 사용하여 현실적인 도시 교통 시뮬레이션을 달성할 수 있음을 입증했습니다. 이 프레임워크는 교통 모델 배포의 장벽을 낮추고, 다양한 도시에서의 시뮬레이션 보정을 간소화하여 더 많은 지역에서의 유용성을 높입니다. 또한, 복잡한 도시 환경에서도 기존 데이터의 제한을이 overcoming하여 실제 교통 패턴과 유사한 결과를 생성하는 방법을 보여줍니다.



### Leveraging BART to Assess CS1 C++ Programming Assignments using Rubric-based Criteria (https://arxiv.org/abs/2606.03814)
- **What's New**: 이 논문은 C++ 프로그래밍 과제의 자동 채점을 위해, 강사 grading 행동을 더 잘 반영하는 grade 예측을 생성하는 transformer 모델의 rubric-aware 멀티태스크 fine-tuning을 조사합니다. multi-semester CS1 데이터 세트를 활용하여, 학생 제출과 점수, 학점 범주 및 과제 rubric을 결합하여 transformer 입력을 위한 통합된 시퀀스로 전처리합니다. BART 인코더-디코더 모델을 사용하여 수치 점수와 학점 범주를 동시에 예측하고, 예측된 학점 분포와 경험적 분포를 맞추는 추가적인 분포 일치 용어를 활용합니다.

- **Technical Details**: 모델은 BART encoder-decoder에 LoRA 적응을 적용하여 멀티태스크 학습과 레이블링 전략, rubric 맥락이 채점 성능에 미치는 영향을 체계적으로 분석합니다. 또한, T5 기반 모델과 쌍별 사전 훈련 접근법을 사용하여 이러한 대안이 의미 있는 개선을 제공하는지 평가합니다. 실험은 멀티태스크 학습이 단일 작업 대안보다 우수한지, 소프트 레이블이 하드 원-핫 인코딩보다 채점의 불확실성을 더 잘 포착하는지, rubric 맥락이 보정을 개선하는지를 다룹니다.

- **Performance Highlights**: 실험 결과, 경계 기반 소프트 레이블과 rubric 맥락을 사용한 멀티태스크 BART가 단일 작업, 하드 레이블 또는 코드 전용 기준보다 낮은 평균 절대 오차와 더 강한 학점 분포 정합을 달성했습니다. 완전히 fine-tuning된 T5 모델은 분포 충실도를 더욱 향상시키고, 쌍별 사전 훈련은 소수 클래스에 대한 민감도를 감소시키면서 수치 오류를 줄입니다. 이 연구 결과는 보정 인식, rubric 기반의 학습이 정확도 최적화 대안보다 강사와 유사한 채점 행동을 생성할 수 있음을 시사합니다.



### Enhancing Operational Safety via Agentic Dialogue Hazard Identification Analysis (https://arxiv.org/abs/2606.03812)
- **What's New**: 이번 연구는 HAZDIAL이라는 프레임워크를 소개하며, 구조화된 에이전트 간 대화가 NLP 기반 위험 식별 품질을 개선할 수 있는지 탐구합니다. 이는 단일 회차 단일 추론(single-pass inference)의 한계를 극복하기 위해 다중 에이전트, 다회 대화(multi-turn interactions)를 활용합니다. 특히, 이 프레임워크는 적대적 토론(adversarial debate)과 협력적 논의(constructive discussion)라는 두 가지 대화 방식의 효과를 체계적으로 비교합니다.

- **Technical Details**: HAZDIAL 프레임워크는 위험 식별을 닫힌 리스트(closed-list) 작업으로 설정하고, 에이전트, 공유 가능한 가변 상태(mutable state), 대화 태그(dialogue tags), 결합 함수를 설계합니다. 두 가지 다중 에이전트 구성인 Debate와 Discuss를 구현하여 대화 구조가 위험 식별에 미치는 영향을 연구합니다. 또한, 에이전트 구성 파라미터를 학습 가능한 유전자(gene)로 다루는 유전 정책 최적화 알고리즘을 제안했습니다.

- **Performance Highlights**: 실험 결과, GPT-OSS 20B 및 GPT-4.1 모델을 사용하여 적대적 토론 방식이 두 모델 모두에서 잘못된 긍정률(false positives)을 최대 40%까지 줄이는 것이 확인되었습니다. 반면, 협력적 논의는 작은 모델에서는 F1 점수를 낮추었으나, GPT-4.1에서는 위험 식별의 재현율(recall)을 0.586으로 향상시켰습니다. 이는 협력적 대화가 특정 상황에서 도움이 될 수 있음을 나타냅니다.



### From Control Boundary to Insurance Claim: Reconstructing AI-Mediated Losses Through the CER Framework (https://arxiv.org/abs/2606.03777)
- **What's New**: 이 논문은 보험(insurance) 조직의 생성적(generative) 또는 에이전틱(agentic) AI 시스템을 통해 발생하는 손실을 다룹니다. 단순히 사건(event) 재구성을 넘어서 시스템의 상태(state)를 재구성해야 한다고 강조합니다. 이 논문은 AI 시스템이 원인적(causal) 연쇄에 있는 손실의 사례를 조명하며, 그로 인해 보험 청구 회복이 가능한지를 질문합니다.

- **Technical Details**: 이 논문에서는 CER이라는 진단 프레임워크를 소개합니다. 이는 C(제어 경계), E(증거 재구성), R(보험 응답)으로 구성되어, AI의 잔여 리스크(Residual Risk) 이전을 위한 사용 사례 수준의 진단을 제공합니다. 각 요소는 시스템의 운영 경계, 상태 및 원인 체인의 재구성 가능성, 재구성된 손실이 보험으로 보장되는지를 점검합니다.

- **Performance Highlights**: 논문은 AI 재구성 문제를 정의하고, CER을 통해 이 문제를 운영화하며, AI 재구성을 위한 청구 수준의 증거를 구체화하는 세 가지 기여를 합니다. 예로, PocketOS와 Replit의 데이터 삭제 사건 및 Moffatt 대 Air Canada 사건을 통해 제기된 원인적 사례를 제시합니다.



### LAP: An Agent-to-Instrument Protocol for Autonomous Scienc (https://arxiv.org/abs/2606.03755)
Comments:
          31 pages

- **What's New**: 이 논문은 Lab Agent Protocol(LAP)을 제안하는데, 이는 자율 과학 시스템에서 에이전트와 물리적 기기 간의 연결을 새롭게 정립하는 프로토콜입니다. 기존의 Anthropic의 Model Context Protocol(MCP) 및 Google의 Agent2Agent(A2A)와 함께, LAP는 다양한 기기를 통합하는 중요한 역할을 합니다. 특히, LAP는 위험한 또는 비가역적인 작업을 수행할 때의 안전성을 강화하는 안전 펜스(handshake) 및 물리적 측정 결과를 위한 스키마를 포함하고 있어 자율 실험 환경의 신뢰성을 높이고자 합니다.

- **Technical Details**: LAP는 A2A의 피어 투 피어 구조를 유지하면서, InstrumentCard, 기기 예약 및 잠금 기능, 안전성 분류, 물리적 결과 타입을 명확히 정의한 MeasurementResult 스키마 등의 네 가지 물리적 원시(primitives)를 추가합니다. 이 프로토콜은 JSON-RPC를 활용하며, 각 작업의 상태 및 오류 모형을 통해 더 안정적인 상호작용을 가능하게 합니다.더불어, 서로 다른 실험실 간의 자원 검색과 샘플의 물리적 관리(custody)를 위한 연합 설계를 제공합니다.

- **Performance Highlights**: 자동화된 실험 환경에서의 LAP의 효용은 기기의 안전한 제어와 함께 측정 결과의 신뢰성을 확보함으로써 크게 향상됩니다. 예를 들어, LAP의 MeasurementResult 스키마는 결과의 반복 가능성을 보장하며, 유일무이한 기기 제어 능력을 통해 통합의 복잡성을 줄일 수 있습니다. 마지막으로, LAP는 SiLA 2 및 OPC-UA와 같은 기존의 장치 표준과 호환되어, 향후 실험실에서 기술적 발전을 더욱 가속화할 수 있는 기반을 마련합니다.



### Proof-Refactor: Refactoring Generated Formal Proofs into Modular Artifacts (https://arxiv.org/abs/2606.03743)
Comments:
          21 pages, 3 figures, 3 tables

- **What's New**: 이 논문에서는 Large Language Models (LLMs)이 형식적인 증명을 생성하는 데 강한 성능을 보이지만, 실제로는 읽기 쉽고 모듈화되어 있으며 유지 관리 가능하고 재사용 가능한 품질이 떨어진다는 점을 지적합니다. 특히, 기존의 증명 생성 파이프라인이 단일 목표로 증명을 생성하도록 유도하여, 라이브러리 품질의 아티팩트를 생성하기보다는 단순한 스크립트를 생성하는 경향이 있음을 설명합니다. 대신, 이 연구에서는 사람의 증명 리팩토링 프로세스에서 영감을 얻어 Proof-Refactor라는 프레임워크를 제안하여, 구조화된 리팩토링 프로세스를 통해 증명의 질을 향상시키고자 합니다.

- **Technical Details**: Proof-Refactor는 네 개의 단계로 구성된 에이전트 기반 프레임워크로, 후보 증명 조각을 추출하고, 도움 선언을 설계하며, 이들을 수학적으로 증명한 후 원래 증명을 수리하는 과정을 포함합니다. 이 과정에서 lean-lsp-mcp이라는 통신 프로토콜을 이용하여 외부 모델과 상호작용할 수 있으며, lean_extract라는 도구를 통해 증명 조각을 독립적인 정리에 추출할 수 있습니다. 기존의 증명 방식과의 큰 차별점은, 증명 길이를 최적화하는 것에 집중하기 보다는 중간 증명 구성요소의 추출, 일반화, 재사용에 초점을 둔다는 것입니다.

- **Performance Highlights**: PutnamBench 및 Putnam2025의 Lean 증명에 대한 실험에서 Proof-Refactor는 기존의 Claude Code 리팩토링 기준을 넘어서는 리팩토링 품질 점수를 개선했습니다. 이 과정에서 특히 서명 품질과 인간의 가독성이 크게 향상되었습니다. 이러한 결과는 프로세스 주도형 리팩토링이 증명 구조를 개선할 수 있음을 보여주며, 증명 길이를 주요 목표로 삼지 않더라도 가능하다는 점을 강조합니다.



### When to Re-Plan: Subgoal Persistence in Hierarchical Latent Reasoning (https://arxiv.org/abs/2606.03741)
Comments:
          Accepted at the Workshop on Compositional Learning: Safety, Interpretability, and Agents (CompLearn), ICML 2026. 10 pages, 2 figures

- **What's New**: 이번 연구에서는 latent reasoning 설정에서 중기 목표의 지속성을 고려하는 새로운 방법론을 제시합니다. Hierarchical Reasoning Model (HRM)을 확장하여, 고위 모듈이 주기적으로 방향성 서브골(subgoal)을 발행하는 구조를 도입했습니다. 이로 인해 계산의 안정성과 적응성을 조화롭게 조절할 수 있어, 계층적 계산을 보다 유연하게 관장합니다.

- **Technical Details**: 고급 모듈은 P 단계 동안 서브골을 유지하며, 이 서브골은 저급 모듈의 숨겨진 상태 업데이트를 편향시킵니다. 또한, 내재된 코사인 정렬 손실(cosine alignment loss)을 통해 서브골을 따라 진행하는 업데이트를 보상합니다. 경험적 연구 결과, 서브골의 지속성이 성능 향상에 결정적인 요소로 작용하며, 중기 목표의 명확한 수명의 설정이 필요함을 강조합니다.

- **Performance Highlights**: ARC 및 ConceptARC 데이터셋에서 테스트한 결과, 서브골을 P=3으로 지속시킬 때 성능이 가장 우수했습니다. 이 설정에서는 평균 손실이 1.595로, 저주파수(P=1) 및 장기 목표(P>3)보다 우수한 성능을 보였습니다. 이 연구는 느린 목표 관리 메커니즘이 효과적으로 작동할 수 있는 조건을 명확히 하여, 향후 연구 시 성과를 위해 중요한 기준이 될 것입니다.



### Unveiling the Structure of Do-Calculus Reasoning via Derivation Graphs (https://arxiv.org/abs/2606.03719)
Comments:
          Accepted at ICML 2026

- **What's New**: 이번 연구에서는 do-calculus의 규칙들이 어떻게 적용되고 결합되는지를 나타내는 도출 그래프(derivation graphs)를 소개합니다. 이를 통해 관찰적(observational) 및 개입(interventional) 확률의 전체 공간을 특성화하고, do-calculus를 통해 동등한 개입 표현을 생성할 수 있는 방법을 제시하였습니다.

- **Technical Details**: 도출 그래프의 구조는 do-calculus 규칙들을 최대 네 번만 적용하여도 충분히 간단한 절차를 통해 해결할 수 있다는 점에서 큰 의미가 있습니다. 이 방법론은 다양한 개입 질의(causal queries)에 대해 동등한 causal quantity를 가진 여러 유효한 추정량(estimands)을 생성하는 데 기여할 수 있습니다.

- **Performance Highlights**: 보여준 결과에 따르면, 이러한 알고리즘을 적용함으로써 동일한 인과적 양(causal quantity)에 대해 보다 효율적인 추정기(estimators)를 얻을 수 있습니다. 이는 do-calculus의 향후 연구 방향에 중요한 기여를 할 것으로 기대됩니다.



### Code-on-Graph: Iterative Programmatic Reasoning via Large Language Models on Knowledge Graphs (https://arxiv.org/abs/2606.03705)
- **What's New**: 본 논문에서는 대형 언어 모델(LLMs)의 제한을 해결하기 위해 코드 기반의 프로그램적 추론 프레임워크인 Code-on-Graph (CoG)를 제안합니다. CoG는 지식 그래프(KGs)와 LLMs의 통합을 위해 사실 기반 지식을 동적으로 생성된 파이썬 클래스를 통해 유연하게 다루며, 기존의 선행 연구 방법에서 발생했던 비효율성과 유연성 부족 문제를 해결합니다. 실험 결과, CoG는 이전의 최신 모델보다 성능을 최대 10.5% 향상시킴을 입증하였습니다.

- **Technical Details**: CoG는 세 가지 단계로 구성된 반복적인 프로세스를 따릅니다: (1) 계획(Planning) - 복잡한 질문을 하위 작업으로 분해; (2) 코딩(Coding) - 사실을 검색하고 이를 파이썬 클래스 정의로 추상화하여 태스크 특화 실행 가능 작업을 생성; (3) 실행(Executing) - 분리된 클래스 인스턴스로 검색된 사실을 코드에 제공하고, 생성된 코드를 샌드박스 환경에서 실행하여 피드백을 통해 수정 및 재시도합니다.

- **Performance Highlights**: 기존의 방법들에 비해 CoG는 복잡한 KGQA 질문을 효과적으로 처리함으로써 유연성과 확장성을 대폭 향상시킵니다. CoG는 다양한 사실 기반 지식에 대해 파이썬 클래스 레벨에서 실행 가능한 코드 생성을 통해 최소한의 토큰 사용으로 다양한 추론 패턴을 지원합니다. 논문에서 제시된 실험은 CoG가 WebQSP, CWQ 및 GrailQA 데이터셋에서 이전의 최고 성능 모델을 초과하는 성능을 기록한 것을 보여줍니다.



### Dynamic Objective Selection with Safeguards and LLM Oversight for Financial Decision-Making (https://arxiv.org/abs/2606.03704)
Comments:
          Accpeted to The 2nd Workskop on Advances in Financial AI Workshop: Towards Agentic and Responsible Systems at ICLR 2026

- **What's New**: 본 논문은 DOSS (Dynamic Objective Selection with Safeguards)라는 새로운 학습 기반 목표 선택기를 제안합니다. DOSS는 해석 가능한 통계적 요약을 통해 매번 의사결정에 유리한 목표 함수를 직접 선택합니다. 이 방법은 고정된 목표 대신 시간에 따라 변화하는 시장 조건에 적합한 목표를 실시간으로 선택할 수 있는 장점을 가지고 있습니다.

- **Technical Details**: DOSS는 의사결정 과정에서 각각의 시간 지점 사이에 목표 함수를 선택하는 분류 문제로 설정됩니다. 이 시스템은 거리 창(rolling window)을 사용하여 시계열 데이터로부터 업데이트를 진행하며, 낮은 신뢰도 목표는 보수적인 기본 목표로 대체하는 게이팅 메커니즘을 적용합니다. 또한, 제안된 목표를 수용하거나 사전 정의된 안전한 기본 목표로 대체하는 대규모 언어 모델(LLM)의 감시 작업을 통합하여 통제 메커니즘을 제공합니다.

- **Performance Highlights**: DOSS는 고정된 목표 및 LLM 기반 직접 선택 방식과 비교해 실험에서 일관된 성능 향상을 보였습니다. 특히, DOSS의 목표 선택 정책은 실시간으로 시장 상태를 반영하여, 목표 전환 시의 오 seçection과 과도한 변경을 줄여 더욱 안정적인 결과를 가져오는 데 기여합니다. 논문에서는 DOSS의 여러 성능 지표가 공개된 벤치마크에서의 우수함을 입증합니다.



### SkillPyramid: A Hierarchical Skill Consolidation Framework for Self-Evolving Agents (https://arxiv.org/abs/2606.03692)
- **What's New**: 이번 논문에서는 SkillPyramid라는 새로운 스킬 통합 프레임워크를 제안하여, AI 에이전트들이 더 폭넓은 작업 일반화를 위해 기존 스킬 경험을 재사용할 수 있도록 합니다. 기존의 비효율적이고 중복된 스킬 구조를 해결하여, 에이전트가 경험을 재사용 가능한 자산으로 변환할 수 있도록 합니다. 이는 궁극적으로 에이전트의 성능 향상과 대규모 작업 처리에 기여합니다.

- **Technical Details**: SkillPyramid는 다계층 구조의 스킬 토폴로지를 기반으로 구성되어 있으며, 하위 레벨은 세부적인 재사용 가능한 원자 스킬을 담고, 상위 레벨에서는 반복되는 문제 해결 패턴을 추상화합니다. 이 프레임워크는 Relation Analyzer와 Relation Builder를 통해 스킬의 재사용 관계를 분석하고, 새로운 스킬을 통합하는 자가 진화 메커니즘을 구현합니다. 각 스킬은 구조화된 자연어 프로그램으로 정의되며, 이를 통해 스킬의 적용 조건 및 실행 절차 등이 명확히 규정됩니다.

- **Performance Highlights**: ALFWorld, WebShop, ScienceWorld를 통한 실험 결과, SkillPyramid는 평균 보상을 38.0% 향상시키고 실행 단계를 27.7% 줄이는 성과를 보였습니다. 이와 같은 성과는 여러 백본 모델에서 반복적으로 나타나며, SkillPyramid가 스킬 수집을 정적 자원 풀에서 동적 진화 시스템으로 전환함을 보여줍니다.



### The DeepSpeak-Agentic Datas (https://arxiv.org/abs/2606.03686)
- **What's New**: 이번 논문에서는 DeepSpeak-Agentic이라는 데이터셋을 소개합니다. 이 데이터셋은 사람과 체화된 AI 에이전트 간의 반구조적 대화로 구성된 37시간 이상의 비디오를 포함하고 있습니다. 이 데이터셋은 AI 에이전트를 자동으로 식별하고, 인간-에이전트 상호작용의 본질을 연구하며, AI 생성된 음성 및 얼굴을 활용한 체화된 AI 에이전트의 발전을 위한 기준을 제공합니다.

- **Technical Details**: DeepSpeak-Agentic 데이터셋은 오디오, 비디오, 텍스트를 포함한 자동 포렌식 식별(automatic forensic identification)을 평가하는 데 사용됩니다. 이 연구는 인간과 AI 간의 상호작용을 다루며, 전문적인 데이터 캡처 시스템을 통해 사람과 에이전트를 자동으로 연결하여 오디오 비주얼 대화를 기록합니다. 이 과정에서 특정 시나리오에 따라 데이터를 수집하고, 혼합 스트림에서 인간과 에이전트를 식별 및 분리하는 기술도 포함되어 있습니다.

- **Performance Highlights**: 이 연구는 체화된 AI 에이전트와의 상호작용을 통해 AI 기술의 효율성을 검사하고, 대형 언어 모델과 AI 생성된 콘텐츠의 현재 상태를 평가합니다. 향후 연구에서 활용될 수 있는 기준을 제공하여, AI 음성과 얼굴 생성의 발전을 촉진할 수 있습니다. 이를 통해 인간과 AI 간의 관계를 더욱 심도 있게 이해할 수 있을 것입니다.



### EvoDrive: Pareto Evolution for Safety-Critical Autonomous Driving via Self-Improving LLM Agents (https://arxiv.org/abs/2606.03678)
- **What's New**: EvoDrive는 자율 주행 시스템의 안전성 검증을 위해 설계된 최초의 자동화된 LLM(대규모 언어 모델) 기반 에이전틱 진화 프레임워크입니다. 이 시스템은 시뮬레이터에 기반한 actor-critic 아키텍처를 활용하여, 새로운 시나리오 생성을 위한 여러 목표를 동시에 최적화합니다. 기존의 방법들이 수작업으로 설정된 휴리스틱에 의존했던 것과 달리, EvoDrive는 자동으로 제안된 개선 사항을 통해 보다 다양하고 현실적인 공격 시나리오를 생성합니다.

- **Technical Details**: EvoDrive는 메모리 기반의 actor가 generator를 반복적으로 개선하하고, critic이 비현실적인 후보를 필터링하는 구조로 설계되었습니다. 또한, 자기 진화하는 세계 평가자가 시뮬레이션 예산을 최적화하기 위해 유망한 제안을 라우팅합니다. 이 시스템은 평가된 후보의 Pareto 아카이브를 유지하여 다양한 공격-현실성 트레이드오프를 보존하고 시뮬레이션 피드백을 통해 향후 진화를 유도합니다.

- **Performance Highlights**: MetaDrive와 CARLA에서의 벤치마크 결과에 따르면, EvoDrive는 다양한 생성기에서 Pareto 경계를 크게 확장하며 가치 있는 시나리오를 생성합니다. 이는 정책 훈련을 위한 중요한 데이터를 제공하며, 자율 주행 분야의 안전성 향상에 크게 기여합니다. EvoDrive를 통해 더 많은 다목적 시나리오를 자동 생성할 수 있는 가능성이 열렸습니다.



### From Answers to States: Verifiable Process-Level Evaluation of Chemical Reasoning in Large Language Models (https://arxiv.org/abs/2606.03660)
Comments:
          23 pages, 6 figures, 14 tables

- **What's New**: 이번 연구에서는 ChemCoTBench-V2라는 화학 추론을 위한 규칙 검증 진단 기준을 도입했습니다. 이는 화학 모델들이 요청된 과정을 일관성 있게 유지할 수 있는지를 평가하기 위한 방법으로, 5,620개의 샘플을 포함한 18개의 과제에서 사용됩니다. 이 벤치마크는 최종 답변의 정확성 뿐만 아니라 자문형 템플릿 준수 및 단계별 검증자 정확성을 평가합니다.

- **Technical Details**: ChemCoTBench-V2는 과정 중심의 화학 추론을 전문가가 정제한 검증 가능한 중간 약속으로 운영합니다. 주어진 작업에 따라 SMILES 패턴, 반응 유형 및 생성물 구성을 포함한 구조적 추적으로 모델의 출력을 분석합니다. 각 과제는 기능 그룹, 링, 골격 등을 포함한 분자 이해, 사이트 특정한 추가, 삭제 및 대체 작업, 생리활성 최적화 및 반응 예측을 포함하여 네 가지 작업 계열로 구성됩니다.

- **Performance Highlights**: 실험 결과, 최종 답변의 성공과 구조적 화학 추론 능력은 별개의 능력으로 나타났습니다. 모델은 요청된 형식을 거의 완벽하게 따르면서도 단계별 검증자 체크에서 실패하거나, 약한 지원 추론으로도 정답을 도출하는 경향이 있었습니다. ChemCoTBench-V2는 모델 간 세분화된 비교를 가능하게 하고, 검증자가 추적의 일관성이 처음 깨지는 구체적인 단계를 식별할 수 있습니다.



### Diagnosing Knowledge Gaps in LLM Tool Use: An Agentic Benchmark for Novel API Acquisition (https://arxiv.org/abs/2606.03657)
Comments:
          37 pages, 12 figures

- **What's New**: 본 논문에서는 NovelAPIBench라는 완전 자동화된 다이내믹 벤치마크를 소개합니다. 이 벤치마크는 기본 모델과 목표 라이브러리를 기반으로, 실제 라이브러리 진화를 반영하는 새로운 API를 발견하고, 이러한 지식의 구성 요소를 추출하여 실행 가능한 코딩 과제를 생성합니다. 이를 통해 기존의 스태틱 벤치마크의 한계를 극복하고, 실패 원인을 진단할 수 있는 여섯 가지 분류 체계를 제공합니다.

- **Technical Details**: NovelAPIBench는 API의 지식 번들을 세 가지 주요 구성 요소(Surface, Exemplars, Mechanism)로 분해하여 실험합니다. Surface는 API의 이름과 매개변수 서명을 포함하고, Exemplars는 공식 문서와 실제 소스 코드에서의 사용 예제를 포함합니다. Mechanism은 알고리즘 설명과 API 구현 소스 코드를 포함하여, 각 API에 대해 난이도에 따라 조정된 세 가지 과제가 생성됩니다.

- **Performance Highlights**: 실험 결과, 외부 지식이 제거되면 매개변수 학습 방식이 새로운 API 지식을 완전히 내재화하지 못한다는 점을 발견하였습니다. 또한, 사용 예제가 가장 강력한 독립적 개선 요소임을 확인했으며, 수퍼바이즈드 파인튜닝은 주어진 API 지식을 사용하는 절차적 메타 스킬을 학습하는 데 중점을 두었습니다. 결과적으로, 검색(retrieval)과 튜닝(tuning)은 상호 보완적 역할을 하며, 튜닝은 절차적 통합을 개선하는 데 기여하고, 검색은 변동적인 API 콘텐츠를 제공합니다.



### Towards Non-Monotonic Entailment in Propositional Defeasible Standpoint Logic (https://arxiv.org/abs/2606.03655)
- **What's New**: 이 연구에서는 전통적인 KLM 스타일의 추론에서 제안된 비모노톤(non-monotonic) 합리적 수반 관계를 PDSL(Propositional Defeasible Standpoint Logic)의 구간으로 확장하는 방법을 제안합니다. PDSL의 표현력을 확장하기 위해 상황적 입장 조건부(situated standpoint conditionals)를 도입하여 특정 입장의 맥락에서 성립하는 무효 조건에 대해 논의할 수 있도록 합니다. 이를 통해 PDSL의 구문을 재정의하고, PDSL의 큰 부분이 상황적 조건부의 집합으로 표현 가능함을 보여줍니다.

- **Technical Details**: 이 연구에서는 PDSL에서 비모노톤 수반(non-monotonic entailment)을 특징 짓기 위해 랭킹 기반 수반 관계를 제안된 방법으로 PDSL 케이스로 전송하는 방법을 정의합니다. 이를 일반적인 경우에 대해 먼저 설명한 후, 합리적 및 사전적 폐쇄(lexicographic closures)와 같은 특정 경우를 다룹니다. 이 과정에서 모든 추론의 신뢰할 수 있는 번역을 제공하며, PDSL의 이 부분에서 수반 검사(entailment-checking)는 주로 명제적 경우에서 알고리즘을 사용하여 수행 가능하다는 점을 강조합니다.

- **Performance Highlights**: 본 연구에서 제안된 방법은 PDSL의 비모노톤 수반을 더 잘 정의하고, 이는 기존 비모노톤 추론 감성을 더 효과적으로 통합하는 데 기여합니다. pDSL의 구문과 의미 구조를 재조정함으로써, 다양한 지식 기반에 대한 수용 가능성을 증가시키고, 각 다양한 입장을 통해 접근할 수 있는 복잡성을 유지합니다. 이로 인해 PDSL의 성능을 크게 향상시킬 수 있음을 보여주며, 특히 다양한 입장 공간을 다룰 때 유용합니다.



### Gender-Dependent Diagnostic Substitution in LLM Medical Triage: Same Symptoms, Unequal Urgency (https://arxiv.org/abs/2606.03641)
Comments:
          7 pages, 3 tables. Multi-model replication across Gemini, Claude, and GPT. Code and data: this https URL

- **What's New**: 본 연구에서는 대형 언어 모델들이 동일한 신경학적 증상에 대해 환자의 성별과 나이만을 변경했을 때, 서로 다른 의료 분류 권고를 내리는지를 조사합니다. Gemini 3.5 Flash, Claude Sonnet 4.6, GPT-5.4-mini를 포함한 세 가지 모델을 사용하여, 7가지 인구 통계적 조건 하에서 표준화된 증상 프로필을 제시합니다. 연구 결과, 젊은 여성들이 동일한 증상에 대해 남성들보다 응급실(ER) 추천 비율이 현저히 낮다는 것을 발견했습니다.

- **Technical Details**: 이 연구는 대형 언어 모델들이 응급 평가에서 성별의 영향을 어떻게 받는지를 탐구합니다. 세 가지 모델 모두에서 젊은 여성은 '특발성 두개내 고혈압'으로 분류되는 경향이 있으며, 이는 낮은 긴급성을 유도합니다. 데이터 수집 방식은 각 모델에 대해 30회의 시험을 실시하여 총 630개의 유효한 API 호출을 기록합니다.

- **Performance Highlights**: 연구 결과, 모든 모델에서 성별에 따른 불균형이 나타났으며, 특히 25세의 젊은 여성들은 ER 추천 비율이 저조했습니다 (Gemini: 0%, Claude: 6.7%, GPT: 6.7%). 나이가 65세인 경우에는 이러한 성별 차이가 사라졌고, 이는 통계적 의료 선호에 의해 주도된 편향을 나타냅니다. 전체 실험의 코드와 자료가 공개되어 결과 재현이 가능합니다.



### TSQAgent: Rating Time Series Data Quality via Dedicated Agentic Reasoning (https://arxiv.org/abs/2606.03629)
- **What's New**: 이번 연구에서는 TS(시간 시리즈) 데이터의 품질 평가에 있어 LLMs(대형 언어 모델)의 새로운 능력을 연구합니다. 특히, TSQBench라는 독자적인 벤치마크를 구축하여 LLMs의 품질 차원 이해 및 비교 능력을 평가합니다. 연구는 기존 LLMs가 품질 차원을 제대로 식별하고 증거 기반 품질 비교를 수행하는 데 어려움을 겪고 있음을 보여줍니다.

- **Technical Details**: TSQAgent라는 새로운 에이전트적 사고 체계를 제안하여 TS 데이터 품질 평가의 주요 도전을 해결합니다. 이 프레임워크는 Perceiver(관찰자), Inspector(조사자), Adjudicator(판단자)의 세 가지 협력적 역할로 분해되어 기능합니다. 또한, 16개 특정 TS 분석 기능을 포함한 도구 보강 에이전트 워크플로우를 설계하여 선택된 차원에 대해 정확한 정량적 비교를 수행할 수 있도록 지원합니다.

- **Performance Highlights**: 실험 결과, 제안된 TSQAgent 프레임워크는 LLMs의 품질 이해 및 정량적 비교 능력을 상당히 개선하며, 이는 데이터 선택 성능의 향상으로 이어집니다. 특히 Timer-S1 모델을 미세 조정하여 75%의 데이터만으로도 뛰어난 성능을 낼 수 있다는 결과를 보여줍니다. 이러한 결과는 제안된 방법이 실세계 데이터셋에도 효과적인 응용 가능성을 갖고 있음을 시사합니다.



### Bridging Auxiliary Constraints to Resolve Instruction Following in Large Reasoning Models (https://arxiv.org/abs/2606.03624)
Comments:
          a pre-MIT Press publication version

- **What's New**: 이 논문에서는 대형 추론 모델(LRMs)이 여러 지침을 신뢰성 있게 따르지 못하는 문제를 제기하고, 이를 제약 준수 문제(Constraint Adherence Problem, CAP)로 정의합니다. 새로운 구조적 접근 방식을 통해 각 명령을 제약의 지식 그래프로 변환하여 문제를 해결하려고 합니다. 이 접근법인 제약 관계 그래프 완성(Constraint Relationship Graph Completion, CRGC)은 제약 간의 관계를 명확히 모델링하고, 교량 제약을 발견하여 모델이 요구 사항을 더 잘 맞추도록 지원합니다.

- **Technical Details**: CRGC는 세 가지 주요 구성 요소로 나뉘어 있습니다: (1) 제약 그래프 구성 - 분해된 명령 제약 간의 관계 매핑, (2) 준수 도전 감지 - 간과되거나 충돌하는 제약 식별, (3) 교량 제약 발견 - 문제 있는 제약 간의 연결을 위한 보조 지침 도입. 이러한 그래프 기반 표현은 LRM이 제약이 명확하지 않거나 상충된다고 인식할 때 가장 큰 어려움을 겪는다는 것을 보여줍니다.

- **Performance Highlights**: CRGC는 39%의 제약 위반을 줄이며, 크기향상한 추론 품질을 유지했습니다. 기존 방법들과는 달리, 이 방법은 모델 파라미터를 수정하지 않고도 제약 준수를 향상시키며, 필요한 경우에만 교량 제약을 적응적으로 결정합니다. 뿐만 아니라, CRGC는 다양한 데이터셋에서 우수한 제약 준수를 보여주며, 추론 능력의 저하 없이 성능을 유지하는 장점이 있습니다.



### Cross-Lingual Token Arbitrage: Optimizing Code Agent Context Windows via Local LLM Preprocessing (https://arxiv.org/abs/2606.03618)
Comments:
          Submitted to EMNLP 2026

- **What's New**: 이 논문에서는 AI 보조 코딩 에이전트의 토큰 비용 문제를 해결하기 위한 새로운 접근 방식을 제안하고 있습니다. 연구진은 로컬 Llama 3.2 모델을 통해 비영어 텍스트를 영어로 번역하고 구조적인 리라이트를 하여 최적화된 프롬프트를 생성하는 미들웨어를 도입했습니다. 이 시스템은 개발자와 클라우드 에이전트 사이에서 작동하여, 인풋의 효율성을 높이고 비용을 절감하는 데 기여합니다.

- **Technical Details**: 제안된 시스템은 'Cross-Lingual Token Arbitrage'라는 아키텍처 메커니즘을 기반으로 하며, 비영어 입력을 영어로 변환한 후, 구조적인 리라이팅을 진행하여 간결한 태스크 지향 형식으로 변환합니다. 또한, 5%의 토큰 예산 임계점을 기반으로 하는 regex 검증을 통해 최적화된 프롬프트가 원본보다 커지지 않도록 보장합니다. 이를 통해 미들웨어는 34-47%의 프롬프트 토큰 감소와 최대 18.8%의 전체 토큰 감소를 달성할 수 있습니다.

- **Performance Highlights**: 세 가지 상용 LLM 백엔드에서 수행된 평가 결과, 제안된 미들웨어가 전통적인 방법들에 비해 우수한 OckScore 성능을 지속적으로 달성하였음을 보여주었습니다. 단순 함수 이름 추출과 비교했을 때, 리라이트 단계에서 주로 성과가 발생하며, 이는 정확도를 개선하는 데 기여하고 있습니다. 따라서 이 연구는 인퍼런스 비용을 크게 줄이면서 코딩의 품질을 희생하지 않는 능동적인 프롬프트 최적화가 가능함을 입증하고 있습니다.



### From Prompt to Service: An SLM-Based Agent Orchestration Gateway for AI-Driven Virtual Worlds (https://arxiv.org/abs/2606.03557)
- **What's New**: 본 논문은 SLM(Small Language Model) 기반의 Agent Orchestration Gateway를 소개하며, 이는 가상 세계 클라이언트와 다양한 AI 백엔드 간의 통신을 원활하게 해주는 경량 런타임 조정 메커니즘입니다. 가상 세계에서 사용자 요청의 의미를 바탕으로 특정 서비스로 라우팅을 수행하며, 이를 통해 클라이언트 애플리케이션을 변경하지 않고도 새로운 AI 기능을 도입할 수 있습니다. 이 게이트웨이는 InterwovenXR 가상 박물관 테스트베드에서 구현되고 평가되었습니다.

- **Technical Details**: SLM은 사용자 프롬프트의 의미적 의도를 분류하고, 구성 가능한 서비스 레지스트리를 통해 라우팅 결정을 검증 및 해결합니다. 선택된 백엔드는 투명하게 호출되어, 저지연(low-latency) AI 서비스 제공을 지원합니다. 평가에서는 컴팩트한 SLM이 엣지 하드웨어에서 신뢰할 수 있는 의도 라우터로 작동할 수 있음을 보여주며, 전반적으로 세분화된 구성은 중급 엣지 하드웨어에서 실행 가능한 조합으로 효율성을 제공합니다.

- **Performance Highlights**: 본 연구 결과는 SLM이 AI 서비스 조정을 지원하는 데 실질적으로 유용하다는 것을 보여줍니다. 또한, 서브 빌리언 파라미터 모델을 사용하여 저지연 의도 인식을 수행함으로써, 가상 세계에서의 AI 상호작용을 확장 가능하고 효율적으로 만들어 줄 수 있습니다. 이 연구는 AI 기반 가상 세계의 확장 가능하고 엣지 지원 상호작용 아키텍처의 평가된 기여를 제공합니다.



### SAGE: A Quantitative Evaluation of Socialized Evolution in Agent Ecosystems (https://arxiv.org/abs/2606.03544)
Comments:
          13 pages, 5 figures

- **What's New**: 이번 연구는 SAGE (Social Agent Group Evolution)라는 평가 프레임워크를 소개하며, 자가 개선 에이전트를 평가하는 새로운 방법론을 제안합니다. 에이전트는 서로의 역사에 접근할 수 있는 SocialEvo 조건과 자신의 과거 기록만을 볼 수 있는 SelfEvo 조건으로 나뉘어 연구됩니다. 이 연구는 공동 경험이 자가 개선만으로는 달성할 수 없는 향상을 측정하는 것이 핵심입니다.

- **Technical Details**: SAGE 프레임워크는 다양한 에이전트 모델들이 공동으로 진화하는 SocialEvo와 개별적으로 진화하는 SelfEvo 조건을 비교합니다. 연구에서는 오픈-엔디드 머신러닝 (MLR-Bench), 경제 계획 (DrugWars), 그리고 멀티플레이어 전략 게임 (Splendor)이라는 세 가지에서 실험을 진행하였습니다. 각 조건에서 에이전트들은 다양한 형태의 과거 기록(예: raw trajectories, reflective summaries)을 통해 학습하며, 이러한 기록의 표현 방식이 학습 행동에 미치는 영향을 분석합니다.

- **Performance Highlights**: 연구 결과, 강력한 자가 개선 에이전트조차도 다른 에이전트의 경험이 없이는 한계에 봉착하는 반면, 피어 경험을 통해 상당한 진전을 이룰 수 있음을 발견했습니다. 특히 경쟁 환경에서는 에이전트가 일반적으로 개선되는 반면, 특정 상대에 맞춘 전략은 개발하지 않는 것으로 나타났습니다. 다양한 형태의 피어 역사에서 필터링된 기록과 요약된 정보가 원시 기록보다 더 나은 성과를 보였으며, 이는 사회적 이익이 단순한 노출의 양에 의존하기보다는 추상화의 질에 의존함을 나타냅니다.



### Overlaying Governance: A Compositional Authorization Framework for Delegation and Scope in Agentic AI (https://arxiv.org/abs/2606.03518)
Comments:
          12 pages

- **What's New**: 이번 논문은 전통적 소프트웨어 시스템의 경계를 넘어서, 독립적으로 행동하고 장기적인 작업을 수행할 수 있는 Agentic AI 시스템의 필요성을 강조합니다. 기존의 고정된 권한 및 위임 체계는 이러한 시스템을 잘 관리하지 못하며, 새로운 방식의 정부(govemance) 프레임워크가 필요합니다. 제안된 프레임워크는 권한을 위임하고, 시간 제한된 권한 하에 행동하며, 상호 작용을 통해 협력하는 복잡한 요구를 충족하도록 설계되었습니다.

- **Technical Details**: 이 논문은 다양한 유형의 위임을 정의하고, 재귀적인 위임 및 동적 범위를 다루는 등의 개념을 포함하여 Agentic AI를 위한 컴포지셔널(governance) 모델을 제시합니다. 여기서 위임은 계약적 조건으로 정의되며, 자원 범위의 감소는 이를 제어하기 위한 중요한 요소로 자리 잡습니다. 이 논문은 관련 정책에 새로운 semantics를 겹쳐 적용하기 위한 컴포지셔널 연산자를 설명하며, 관계 기반 접근 제어(Relation-Based Access Control, ReBAC) 모델을 활용합니다.

- **Performance Highlights**: 제안된 프레임워크는 형식적 증명과 경험적 평가를 통해 검증되었으며, Agentic AI 시스템에서 책임 있는 권한을 위한 실용적인 기초를 제공합니다. 이 구조는 사용자가 직접 위임을 추적하고, 조건부 권한 하에 행동할 수 있게 하며, 이를 통해 사용자와 에이전트 간의 상호작용을 효과적으로 관리할 수 있습니다. 또한, 지속적인 검증과 권한 요청 및 철회를 가능하게 하는 상태를 제공하여 제로 트러스트(zero-trust) 관점에서도 유용합니다.



### ThoughtFold: Folding Reasoning Chains via Introspective Preference Learning (https://arxiv.org/abs/2606.03503)
- **What's New**: 본 논문은 Long Reasoning Models (LRMs)과 Chain-of-Thoughts (CoTs)를 다루며, Reinforcement Learning with Verifiable Rewards (RLVR) 접근 방식에서 발생하는 문제를 해결하기 위한 새로운 프레임워크인 ThoughtFold를 제안합니다. 기존의 RLVR 방식들은 긴 CoTs에서 발생하는 중복 탐색을 강화하는 문제가 있었으며, 이를 해결하기 위해 ThoughtFold는 보다 효율적인 추론을 위해 세심한 선호 학습(fine-grained preference learning)을 활용합니다.

- **Technical Details**: ThoughtFold 프레임워크는 각 올바른 경로 내의 중복성을 식별하기 위해 내성적인 전략(introspective strategy)을 사용하며, 이를 통해 후보 서브-경로의 스펙트럼(spectrum)을 생성합니다. 이 스펙트럼을 활용하여 중복 탐색을 명시적으로 처벌하고 필수적인 추론 세그먼트를 직접 연결하도록 모델을 유도하는 masked preference optimization 목표를 설정합니다.

- **Performance Highlights**: ThoughtFold는 DeepSeek-R1-Distill-Qwen-7B의 토큰 사용량을 약 56% 줄이면서도 최첨단 정확도를 유지하는 데 성공했습니다. 이러한 효율성 향상은 LRMs의 과도한 사고 문제를 해결하는 데 중요한 기여를 합니다.



### A formal definition and meta-model for a machine theory of mind (https://arxiv.org/abs/2606.03471)
Comments:
          48 pages, 2 figures

- **What's New**: 이 논문에서는 인지 심리학(cognitive psychology), 신경 과학(neuroscience), 인공지능(artificial intelligence)에서의 증거 기반 원리를 바탕으로 한 'Machine Theory of Mind'의 엄격한 공식 정의를 처음으로 제안합니다. 이 정의를 활용하여 최신 연구 동향을 살펴보고, 문제를 '풀 수 있는' 추가 연구를 위한 잠재적인 의제를 제시합니다.

- **Technical Details**: 이 연구는 Machine Theory of Mind의 일반적인 전체론적 메타 모델을 발전시키고, 이러한 모델의 경험적 기준(Empirical Benchmarking)을 설정하는 데 있어 최신 동향을 분석합니다. 이는 인공지능이 어떻게 인간의 사고 과정을 모사할 수 있는지를 이해하는 데 도움을 주는 중요한 기초를 마련합니다.

- **Performance Highlights**: 논문의 기여는 이전 연구에 비해 보다 체계적이고 긴밀한 정의를 제안하였다는 점에 있습니다. 이러한 접근 방식은 Machine Theory of Mind의 실행 가능성을 높이고, 인공지능 기술이 인간의 사고 과정을 어떻게 반영할 수 있는지에 대한 새로운 방향성을 제시합니다.



### StepFinder: A Temporal Semantic Framework for Failure Attribution in Multi-Agent Systems (https://arxiv.org/abs/2606.03467)
Comments:
          12 pages, 5 figures. Accepted by KDD 2026

- **What's New**: 이번 연구에서는 LLM(대형 언어 모델) 기반 다중 에이전트 시스템의 실행 오류 전파 문제를 해결하고, 시스템의 신뢰성을 향상시키기 위한 새로운 접근법인 StepFinder를 제안합니다. StepFinder는 경량화된 실패 귀속(failure attribution) 프레임워크로, 단계별 오류 점수를 세밀하게 조정하여 정확한 근본 원인을 식별합니다. 이 방법은 기존 LLM 기반 방법에 비해 추론 효율성을 79% 향상시켜, 복잡한 다중 단계 작업에서 성능을 크게 개선합니다.

- **Technical Details**: StepFinder는 세 가지 통합 단계로 구성된 체계적인 접근 방식을 사용하여 다중 에이전트 시스템의 실패 귀속 문제를 해결합니다. 첫 번째로 실행 경로를 인코딩하여 시계열 의미 벡터로 변환하며, 두 번째로 시계열 모델링과 에이전트 인식 상호 작용을 결합해 장기 종속성과 인과 관계를 포착합니다. 마지막으로, 단계별 오류 점수를 다중 스케일 차이와 위치 편향을 통해 정교하게 조정하여 근본 원인을 정확히 식별합니다.

- **Performance Highlights**: StepFinder는 Who&When 벤치마크에서 실험을 통해 기존의 모든 방법에 대해 일관되게 우수한 귀속 정확도와 순위 품질을 기록했습니다. 특히, 알고리즘 생성_subset에서 4.76%의 정확도 향상과 수동 생성_subset에서 10.35%의 향상을 이루었으며, 가장 빠른 LLM 기반 방법의 추론 시간을 5배 단축시켰습니다. 이러한 성과는 StepFinder의 효과성과 효율성을 입증합니다.



### DMF: A Deterministic Memory Framework for Conversational AI Agents (https://arxiv.org/abs/2606.03463)
Comments:
          21 pages, 3 figures

- **What's New**: 본 논문은 대화형 AI 에이전트가 필요로 하는 메모리 시스템에 대한 새로운 접근법인 결정을 촉진하는 메모리 프레임워크(Deterministic Memory Framework, DMF)를 소개합니다. DMF는 기존의 LLM 기반 요약 대신 고전적인 NLP 분석 및 수학적 점수를 기반으로 하는 결정론적인 파이프라인을 사용하여 메모리 관리에서의 여러 문제점을 해결합니다. 이 시스템은 대화 상호 작용에 대해 결정적 콘텐츠 신호와 구조화된 기원을 결합해 생존 점수(Survival Score) Ω를 계산하여, 상호작용 횟수 감소 법칙에 따라 연관성을 진화시키는 과정을 설명합니다.

- **Technical Details**: DMF는 CPU 우선 접근 방식을 채택하여 메모리 관리 루프에서 LLM 호출을 전혀 사용하지 않습니다. 이 프레임워크는 대화 상호작용에서 수치적 콘텐츠 신호 및 구조적 대화 단서를 추출하고, 정적 생존 점수 Ω를 계산하며, 상호작용 카운트 감소를 적용합니다. 메모리 관리에서 모든 결정은 결정적 규칙을 통해 이루어지며, 메모리 업데이트와 유사한 신뢰할 수 있는 정보 유지 구조를 제공합니다.

- **Performance Highlights**: DMF는 메모리 맥락을 준비하는 동안 제로 토큰을 사용하면서, 전체 대화에 대해 5배에서 242배 적은 토큰을 사용하여 비교 가능한 정확도를 달성했습니다. 이 결과는 LLM 호출을 메모리 관리 루프에서 제거하여 비용을 거의 제로에 가깝게 줄일 수 있음을 보여줍니다. DMF는 대화형 AI 에이전트를 위한 결정론적 메모리 시스템을 가능하게 합니다.



### What Makes Interaction Trajectories Effective for Training Terminal Agents? (https://arxiv.org/abs/2606.03461)
- **What's New**: 본 논문은 코드 에이전트(Post-training)의 교육 능력을 향상시키기 위한 새로운 접근 방식을 모색합니다. 기존에는 강력한 코드 에이전트가 더 나은 교사가 된다는 가정이 있었지만, 이는 주어진 작업의 난이도와 학생의 역량에 따라 다르게 나타났습니다. 새로운 데이터 파이프라인인 Terminal-Lego를 통해 저자는 다양한 도메인의 실제 문제를 기반으로 하는 에이전트 작업을 변환하여 교수가 어떻게 수행될 수 있는지를 조사합니다.

- **Technical Details**: 논문에서는 Terminal-Lego라는 파이프라인을 사용하여 StackOverflow의 실제 기술 문제를 Docker 검증된 작업으로 변환합니다. 이를 통해 학생들이 에이전트의 행동 이력을 어떻게 내재화할 수 있는지를 평가하는 'Environment-Grounded Supervision (EGS)'이라는 개념을 도입합니다. 이렇게 수집된 데이터는 학생 에이전트 Qwen3-32B가 Terminal-Bench 2.0에서 24.3%의 점수를 기록함으로써 뛰어난 데이터 효율성을 보여줍니다.

- **Performance Highlights**: 흥미롭게도, 성능이 뛰어난 Claude Opus 4.6가 교육 효과에서는 낮은 점수를 기록한 DeepSeek-V3.2보다 덜 효과적이라는 사실이 발견되었습니다. 이는 높은 성능이 반드시 가르치는 능력으로 이어지지 않음을 보여줍니다. 따라서 'Harness Engineering'이라는 개념을 통해 환경과의 상호작용 구조의 체계적인 설계가 에이전트의 후속 훈련에서 중요한 역할을 한다는 것을 강조합니다.



### CP-Agent: Context-Aware Multimodal Reasoning for Cellular Morphological Profiling under Chemical Perturbations (https://arxiv.org/abs/2606.03435)
Comments:
          ICLR 2026

- **What's New**: 이 논문은 CP-Agent를 도입하여 Cell Painting을 통한 약물 스크리닝의 효율성을 향상시키는 방법을 제안합니다. CP-Agent는 실험적 맥락을 고려하는 다중 모달(Multimodal) 대형 언어 모델(MLLM)로, 약물의 영향을 받는 세포 형태 변화에 대한 해석 가능한 설명을 생성합니다. 또한, CP-CLIP이라는 컨텍스트 인식 정렬 모듈을 활용하여 고해상도 이미지와 실험 메타데이터를 결합하여 강력한 약물 및 작용 메커니즘(MoA) 분별력을 제공합니다.

- **Technical Details**: CP-Agent는 190만 개의 이미지-맥락 쌍에서 사전 훈련된 모델로, 실험적 조건과 약물 화합물의 구조적 정보를 포함합니다. 이 모델은 고해상도 Cell Painting 이미지를 다양한 환경에서 조화롭게 정렬하고, 주요 필드를 삽입하여 정렬 성능을 향상시킵니다. CP-CLIP을 통해 약물의 특성을 더 잘 반영하고, 세포 형태에 대한 생물학적 관련성을 높이는 방안을 제시합니다.

- **Performance Highlights**: CP-Agent는 수많은 분류 작업에서 기존 일반 목적 기준보다 뛰어난 성능을 발휘하며, 최대 F1 점수 0.896을 기록했습니다. 이 프레임워크는 효과적인 연역 도구와 결합하여 구조적이고 해석 가능한 산출물을 생성하며, 약물 발견을 위한 실험 설계 및 가설 개선에 대한 인사이트를 제공합니다. CP-Agent의 이러한 기능들은 호기성 약물 발견을 지원하며, 반복적 가설 생성을 간소화하여 의사결정 과정을 개선합니다.



### InfoMem: Training Long-Context Memory Agents with Answer-Conditioned Information Gain (https://arxiv.org/abs/2606.03329)
Comments:
          17 pages, 7 figrues,

- **What's New**: 이 논문에서는 LLMs(대형 언어 모델)의 긴 문맥 과제를 해결하기 위해 InfoMem이라는 새로운 보상 메커니즘을 제안합니다. 기존의 chunk-wise memory agents는 메모리 형성을 효과적으로 감독하는 방법이 부족했으며, 제안된 방법은 최종 메모리가 실제 답변에 얼마나 기여하는지를 평가하는 데 중점을 둡니다. 이로 인해 최종 메모리의 유용성을 직접적으로 감독할 수 있습니다.

- **Technical Details**: InfoMem은 최종 메모리의 유용성을 평가하고 강화 학습(RL) 최적화를 안정시키기 위해 성공적인 경로에만 이 신호를 적용하며 보상 구성 전에 정규화합니다. 이 방법은 모델이 최종 메모리로부터 답변의 로그 우도(per-token log-likelihood)를 증가시키는지를 측정하여 연산이 가능하도록 합니다. 또한, InfoMem은 기존의 RL 기반 시스템들보다 성능을 개선하는 효과를 보입니다.

- **Performance Highlights**: 실험 결과, InfoMem은 유사한 메모리 에이전트 RL 기초선에 비해 장기 맥락 메모리 에이전트의 성능을 일관되게 향상시키는 것으로 나타났습니다. 효과적인 최종 메모리 보상은 성공적인 경로에서 작동하고 보상 구성 전에 정규화되어야 하며, 쿼리에만 의존하지 않고 정답에 따라 조정되어야 한다는 세 가지 주요 속성도 확인되었습니다.



### The Violation Situation Pattern: A Knowledge-Graph Pattern for Compliance Violations (https://arxiv.org/abs/2606.03326)
- **What's New**: 이번 연구에서는 기존의 Compliance pipelines의 한계를 극복하기 위해 Violation Situation Pattern (VSP)를 제안합니다. 이 패턴은 탐지된 위반 사항을 지속 가능한 그래프 노드로 만들어, 법적 엔티티 및 계약의 생애 주기를 포괄하는 새로운 접근 방식을 제공합니다. VSP는 위반 기록을 첨부하는 규칙 식별자, 유효 기간 및 생애 주기를 포함하여, 추적 가능한 증거 링크를 통해 위반 상황을 명확히 합니다.

- **Technical Details**: VSP는 Gangemi와 Mika의 Situation 패턴을 기반으로 하여, 탐지된 위반을 영속적인 그래프 노드로 표현합니다. 각 위반 노드는 규칙 식별자, 유효성 시간 간격, 생애 주기 상태 및 관련 엔티티에 대한 증거 링크를 포함합니다. 그래프 이벤트로 저장되는 생애 주기 전환은 불변성을 보장하며, 데이터의 감사 이력을 그래프 탐색을 통해 재구성할 수 있게 합니다.

- **Performance Highlights**: 실험 결과, V4의 클로즈-존재감에서 마감일 확인으로 확장하는 경우 F1 점수가 0.312에서 0.602로 개선되었습니다. 이 결과는 패턴의 아이덴티티, 생애 주기, 증거 의미는 그대로 유지하면서도, 규칙 본체의 독립성을 강조합니다. 따라서 이 새로운 접근 방식은 탐지 로직이 진화할 수 있도록 하며, 축적된 감사 이력을 위배하지 않게 됩니다.



### The Reliability Gap in Benchmark Auditing: Distribution Shift and Scale as Failure Modes of Contamination Detection (https://arxiv.org/abs/2606.03305)
- **What's New**: 이 논문은 LLM(대형 언어 모델)의 평가에서 벤치마크 오염(benchmark contamination) 문제가 심각함을 강조합니다. 기존의 통계적 도구들이 제한된 학문적 환경에서 validate 되었음을 언급하며, 현실적인 감사(auditing) 상황에서도 여전히 유효한지에 대한 의문을 제기합니다. 또한, 연구는 분포 변화(distribution shift)와 규모 제한(scale constraints)이라는 두 가지 실패 모드를 조명하며, 이들 약점을 해결하기 위한 평가를 실시했습니다.

- **Technical Details**: 연구에서는 LLM Dataset Inference, Post-Hoc Dataset Inference, CoDeC의 세 가지 주요 탐지 방법론을 27개의 모델에 대해 체계적으로 평가하였습니다. 이 방법들은 다수의 포스트-훈련 단계와 크기가 서로 다른 다양한 LLM을 대상으로 하였습니다. 평가 결과, 현재의 감사 방법들은 벤치마크 노출을 일관되게 확인하는 데 신뢰할 수 없음을 보여주었습니다.

- **Performance Highlights**: 335번의 평가 중 정확한 결과는 총 199번에 불과했습니다. LLM Dataset Inference는 분포 변화 아래에서 false positive를 초래하였고, Post-Hoc Dataset Inference는 벤치마크 규모에서 저조한 성능을 보였으며, CoDeC는 개별 벤치마크 분할을 검증하기에는 불충분한 신호만을 제공했습니다. 연구의 결과는 투명한 데이터 출처(data provenance)가 벤치마크 정직성 주장에 있어 가장 신뢰할 수 있는 기반임을 시사합니다.



### LEAP: Supercharging LLMs for Formal Mathematics with Agentic Frameworks (https://arxiv.org/abs/2606.03303)
- **What's New**: LEAP는 일반적인 대형 언어 모델(LLMs)을 사용하여 자동화된 형식 정리 증명에서 최첨단 성능을 달성하는 에이전틱 프레임워크를 소개합니다. 이 시스템은 복잡한 문제를 더 작은 단위로 분해하여, 비공식적인 청사진과 형식 증명 구조 간의 다리를 놓습니다. 새로운 Lean-IMO-Bench 벤치마크를 도입하여 다양한 난이도의 문제를 평가할 수 있게 하였습니다.

- **Technical Details**: LEAP는 고급 청사진 생성과 저수준 형식 증명 생성을 결합하며, 지속적인 컴파일러 피드백을 통해 오류를 반복적으로 수정하는 작업 흐름을 표현합니다. 이 시스템은 입력된 정리에 대해 단계적으로 접근하며, 필요 시 비공식적인 증명을 생성하고 Lean 코드로 변환합니다. LEAP는 문제를 해결하기 위해 AND-OR DAG(Directed Acyclic Graph)를 사용하여 중간 레마로 목표를 분해합니다.

- **Performance Highlights**: LEAP는 2025년 푸트남 대회에서 12개의 문제를 모두 해결하여 완벽한 성과를 달성했습니다. Lean-IMO-Bench에서 일반 LLM의 형식 해결 비율을 10% 미만에서 70%로 크게 향상시켰으며, 이는 전문 ATP 모델 (5%)와 아리스토텔레스 시스템 (48%)을 초월한 것입니다. 이러한 결과는 LEAP의 구조적이고 반복적인 상호작용이 형식 수학의 주요 병목을 극복함을 시사합니다.



### A Negative Result on Cross-Model Activation Transfer in a Pythia Multi-Hop Setting (https://arxiv.org/abs/2606.03280)
Comments:
          15 pages, 6 figures

- **What's New**: 이번 연구는 언어 모델 간에 숨겨진 정보를 전달하는 새로운 채널을 제안합니다. 저자는 언어 모델이 훈련 중 생성된 데이터의 숨겨진 신호를 통해 행동 특성을 전이할 수 있음을 보여왔습니다. 그러나 데이터 중개가 아닌, 숨겨진 활성화(hiddent activations)를 번역하여 직접적으로 다른 모델에 주입하는 가능성을 탐색합니다. 이 연구는 Pythia-160M에서 Pythia-410M으로의 다단계(multihop) 추론 설정에서 성능을 시험했습니다.

- **Technical Details**: 연구에서는 감춰진 활성화가 어떻게 다른 모델에 효과적으로 전이될 수 있는지를 평가합니다. 번역 계층(translation layer)은 송신 모델과 수신 모델 간의 강력한 정규화 공간 매핑(normalized-space map)을 학습합니다. 그러나 수신 모델에 번역된 활성화를 주입했을 때, 실제 질문 답변 성능은 향상되지 않았습니다. 특히, 대체 스타일(injection) 방식은 계속해서 부정적인 영향을 미칩니다.

- **Performance Highlights**: 실험 결과, 저자는 활성화 전이 메커니즘이 자연어 중계(natural-language relay)를 통해 전달되는 것보다 나쁘지 않지만 실제로 유용한 인과 통신(causal communication)을 제공하지 못한다고 결론 내렸습니다. 송신 모델과 수신 모델 간의 높은 구조적 유사성에도 불구하고, 이 번역된 비트가 수신 모델 내에서 원활하게 작동하지 않음을 명확히 보여주었습니다. 이 연구는 향후 모델 간 활성화 통신의 유용성을 위해 훈련 목표가 단순히 송신자에서 수신자에 대한 적합성을 넘어서야 한다고 제안합니다.



### Distilling Answer-Set Programming Rules from LLMs for Neurosymbolic Visual Question Answering (https://arxiv.org/abs/2606.03269)
Comments:
          Under consideration in Theory and Practice of Logic Programming (TPLP)

- **What's New**: 이 논문에서는 Visual Question Answering (VQA) 분야에서 새로운 접근법을 제시합니다. 대규모 언어 모델(LLMs)에서 규칙을 증류(distill)하여 VQA의 추론 이론을 확장하는 방법을 사용합니다. 이는 기존의 논리 기반 표현을 활용하여 해석 가능성을 높일 수 있으며, 작업 요구 사항이 변경될 때 개발자의 부담을 경감하는데 도움을 줍니다.

- **Technical Details**: 연구는 LLM을 활용하여 초기 VQA 추론 이론을 수정하거나 확장하는 방법론을 다룹니다. 이 과정에서 해답 집합 프로그램(answer-set program)을 사용하여 작업의 새로운 요구 사항을 충족하도록 인도합니다. VQA 데이터셋의 예시는 LLM을 안내하고 결과를 검증하며, ASP 솔버의 피드백을 통해 오류 규칙을 수정하는 데 도움을 줍니다.

- **Performance Highlights**: 다양한 VQA 데이터셋에서 이 방법이 효과적임을 입증했습니다. 주목할만한 점은, LLM으로부터 올바른 규칙을 유도하기 위해 소수의 사례만 필요하다는 것입니다. 실험 결과는 LLM으로부터의 규칙 증류가 전통적인 데이터 기반 규칙 학습 방법에 대한 유망한 대안임을 시사합니다.



### Do Real-World Datasets Contain Natural Experiments? An Empirical Study Using Causal Feature Selection (https://arxiv.org/abs/2606.03251)
- **What's New**: 이 연구는 자연에서 어떤 개인이나 집단에 영향을 미치는 사건이 존재한다는 사실을 통해 자연 실험(natural experiments)을 탐구합니다. 특히, COVID-19 팬데믹을 예로 들어 이러한 자연 실험이 실제 데이터셋에서 발생하는지, 그리고 어떻게 처리해야 하는지를 질문합니다. 연구자들은 causal discovery를 이용해 데이터에서 자연 실험을 발견하는 방법을 제안합니다.

- **Technical Details**: 자연 실험을 감지하기 위해, 연구자들은 데이터의 기본 인과 그래프를 복원하고 인과적 연결을 기반으로 feature selection을 수행합니다. 이 과정에서 데이터가 관측(observational) 데이터가 아니라 개입(interventional) 데이터로 취급했을 때 모델 성능이 개선되면, 이는 해당 데이터셋에 자연 실험이 존재한다는 것을 나타냅니다. 또한, 합성 그래프를 이용한 데이터셋 시뮬레이션을 통해 이 가설을 검증합니다.

- **Performance Highlights**: 실제 데이터셋에 대한 체계적인 평가를 통해, 연구 결과는 많은 현실 데이터셋이 자연 실험을 포함하고 있음을 보여주고 있습니다. 이러한 자연 실험을 활용하여 인과 추론(causal inference) 기술을 통해 모델 성능을 개선할 수 있음을 입증합니다. 본 연구는 이 분야에 대한 초기 탐구를 대표하며 향후 연구를 위한 기초 자료를 제공합니다.



### Solipsistic Superintelligence is Unlikely to be Cooperativ (https://arxiv.org/abs/2606.03237)
Comments:
          24 pages, 1 figure, Accepted at Proceedings of the 43rd International Conference on Machine Learning, 2026

- **What's New**: AI의 중심 과제는 능력에서 공존( co-existence)으로 변화하고 있습니다. 이 논문은 AI 시스템의 배치가 엔도제닉( endogenous) 비정상성을 유도하며, 이는 역사적 분포가 배치 맥락과 서로 다를 때 발생하는 문제를 다루고 있습니다. 저자들은 AI가 협력(cooperation)에 참여해야 하며, 이는 다수의 행위자 간의 상호의존성을 탐색하는 평형 선택(equilibrium-selection) 과정을 요구한다고 주장합니다.

- **Technical Details**: AI 연구에서 지배적인 패러다임은 환경이 외적(exogenous)이고 정적인(stationary) 피드백 원천으로 간주하며, 다른 행위자는 예측 가능한 존재로 보기 때문에 자아중심적(solipsistic) 접근 방식에 기초하고 있습니다. 이러한 가정은 AI 시스템이 다른 적응형 에이전트와 배치될 때 발생하는 딜레마를 간과하게 만듭니다. 저자들은 해결과제를 일련의 고정된 평가 세트에 대한 성과 최적화(output)에 제한하기보다는, AI의 배치에 따른 상호작용을 고려해야 한다고 강조합니다.

- **Performance Highlights**: AI의 발전을 두고 기존 플랫폼이 여전히 효과적일 수 있지만 AI의 사회_기술적(socio-technical) 환경에서 고급 AI의 배치가 반응 동역학에 크게 노출될 것임을 강조합니다. 협력의 필요성을 주장하며, 집단적 동역학이 해를 끼칠 수 있는 능력 있는 시스템이 반드시 긍정적인 결과만을 도출하지 않는다고 경고합니다. 논문은 AI의 설계가 단순한 성과 최적화가 아닌 협력적 동역학을 반영한 구조적 특성으로 재구성되어야 한다고 주장합니다.



### Perceive Before Reasoning: A Pre-Reasoning Perception Framework for Efficient and Reliable Proactive Mobile Agents (https://arxiv.org/abs/2606.03236)
- **What's New**: 이 논문에서는 Multimodal large language models (MLLMs)를 활용하여 모바일 에이전트의 능력을 개선한 연구 결과를 소개합니다. 특히, 기존의 시스템들이 어떻게 '언제' 개입할지를 결정하는 데 어려움을 겪고 있는지를 설명하고, 'Pre-Reasoning Perception Framework (PRPF)'라는 새로운 두 단계의 프레임워크를 제안합니다. 이 프레임워크는 개입을 예측하고 필요한 경우에만 더 무거운 MLLM을 활성화하여 불필요한 예측을 줄이는 방식으로 설계되었습니다.

- **Technical Details**: PRPF는 경량의 Multimodal Proactive Perceptor (MPP)와 Proactive Agent Reasoner (PAR)로 구성되어 있습니다. MPP는 사용자가 개입할 필요가 있는지를 즉각적으로 판단하고 적합한 기능 후보를 예측하여, 이후 PAR에서 심층적인 추론을 수행합니다. 이 두 가지 단계를 아키텍처적으로 분리함으로써, 각 모듈은 대응하는 목표에 맞춰 최적화되어 있으며, 불필요한 인퍼런스(추론)를 피할 수 있습니다.

- **Performance Highlights**: 실험 결과에 따르면 PRPF는 ProactiveMobile 벤치마크에서 false trigger rates (FTR)를 13.76%에서 7.21%로 줄였으며, success rates (SR)는 20.82%에서 41.15%로 향상되었습니다. 이러한 성과는 PRPF가 언제 개입해야 할지를 더 잘 파악하여 잘못된 간섭을 줄이고, 에이전트의 적절한 지원을 보다 정확하게 실행하는 데 기여함을 보여줍니다.



### Effect of Demographic Bias on Skin Lesion Classification (https://arxiv.org/abs/2606.03214)
Comments:
          Accepted for publication at the Journal of Machine Learning for Biomedical Imaging (MELBA) , 26 pages, 12 figures

- **What's New**: 이번 연구에서는 ResNet 기반의 convolutional 모델을 사용하여 피부 병변 분류 성능을 평가하고, 훈련 데이터의 인구통계학적 편향이 특히 환자의 성별과 연령에 미치는 영향을 분석합니다. 선형 프로그래밍(linear programming)을 통해 인구통계학적 특성이 통제된 데이터셋을 생성하여 편향 효과를 체계적으로 조사합니다. 그리고 단일 작업 모델, 강화된 다중 작업 모델 및 적대적 학습(adversarial learning) 방식 등 세 가지 학습 전략의 효과를 분석했습니다.

- **Technical Details**: 연구 결과, 성별 기반 훈련 데이터셋을 사용했을 때 모델 성능이 최적화되었으며, 훈련 데이터에 남성 환자를 포함하면 남성 하위 집단에 대한 성능이 개선되는 것을 확인했습니다. 강화 학습과 적대적 학습 전략은 균형 잡힌 데이터셋과 여성 우세 데이터셋에서 편향 격차를 좁히거나 제거했지만, 남성 우세 설정에서는 여전히 남성이 여성보다 높은 성능을 보였습니다. 나이 기반 분석을 통해서는 세 가지 모델 접근 방식에서 유사한 기준 성능을 유지하였으나, 연령 카테고리가 증가함에 따라 성능이 점차 감소하는 경향이 있음을 발견했습니다.

- **Performance Highlights**: 젊은 연령대에서 가장 높은 성능을 달성하는 것으로 나타났으며, 균형 잡힌 훈련이 최적의 결과를 낳았지만, 나이가 많아질수록 성능은 감소했습니다. 성별로 인한 편향은 주로 데이터 불균형에서 발생하며, 연령 편향은 모든 데이터 분포와 관계없이 젊은 그룹을 일관되게 선호하는 경향을 보였습니다. 연구에 따르면, 이러한 다양한 메커니즘을 완화하기 위해서는 특정한 개선 전략이 필요하다고 결론지었습니다.



### MedCUA-Bench: A Screenshot-Only Benchmark for Clinical Computer-Use Agents (https://arxiv.org/abs/2606.03203)
- **What's New**: 이 논문은 MedCUA-Bench라는 임상 컴퓨터 사용 에이전트를 위한 인터랙티브 벤치마크를 소개합니다. 이 벤치마크는 10개 의료 분야에 걸쳐 18개의 임상 시나리오를 포함하며, 이는 실제 제품 매뉴얼과 오픈 소스 의료 시스템에서 재구성된 것입니다. 기존의 평가 기준들은 일반적인 웹 또는 데스크톱 태스크에 초점을 맞췄으나, 임상 소프트웨어에 필요한 도메인 지식과 안전성 검증에는 미치지 못했습니다.

- **Technical Details**: MedCUA-Bench는 사용자가 원하는 의도 레벨의 목표와 단계별 절차로 세분화된 목표를 제공합니다. 각 시나리오는 전통적인 의료 환경의 UI를 모사하며, 식별 및 데이터 정확성, 정보 신뢰성, 기록 완전성 및 작업 흐름 안전성 등 다섯 가지 안전 차원을 고려하여 평가 됩니다. 벤치마크는 216개의 기본 작업을 포함하고 있으며, 각 태스크는 고유한 시나리오 내용과 고정된 인터페이스 및 검사기에서 실행됩니다.

- **Performance Highlights**: 23개 에이전트를 테스트한 결과, 최상의 비공개 모델이 54.2%의 엄격 성공률을 기록했지만, 실제 OpenEMR에서는 모든 모델의 성공률이 9% 미만으로 나타났습니다. 오픈 소스 에이전트는 평균 2.5%의 성공률을 보였고, 최상의 성과를 낸 모델도 겨우 16.2%에 그쳤습니다. 이 연구는 현재의 에이전트와 신뢰할 수 있는 임상 소프트웨어 사용 사이의 간극을 드러내며, 향후 연구를 위한 재현 가능한 테스트베드를 제공합니다.



### ClinicalMC: A Benchmark for Multi-Course Clinical Decision-Making with Large Language Models (https://arxiv.org/abs/2606.03157)
- **What's New**: 이 논문에서는 ClinicalMC라는 새로운 벤치마크를 제안하여 다중 과정을 포함한 임상 의사결정 과정을 평가합니다. 기존의 벤치마크는 단일 과정에서의 성과만을 평가한 반면, ClinicalMC는 입원부터 퇴원까지의 환자 상태 변화에 초점을 맞춥니다. 이 벤치마크는 1,275개의 중국어 샘플과 5,804개의 영어 샘플로 구성되어, 각 환자의 치료 과정 중 여러 단계를 반영하고 있습니다.

- **Technical Details**: ClinicalMC는 triage, 초기 검사/진단/치료, 후속 다중 과정 검사/평가/치료, 최종 진단의 네 가지 주요 단계로 나뉘며, 이를 통해 의료 관련 LLM의 성능을 테스트합니다. 평가 프레임워크는 환자, 검사자, 의사 에이전트로 구성된 다중 에이전트 구조를 채택하고 있으며, 두 가지 실험 설정(단일 턴 정적 설정 및 다중 턴 동적 설정)을 수행합니다. 모델로는 GPT5-mini와 같은 폐쇄형 LLM, DeepSeek-V3.2와 같은 오픈소스 LLM, HuatuoGPT-o1과 같은 의료 특화 LLM이 포함됩니다.

- **Performance Highlights**: ClinicalMC에서 평가한 결과, 최신 의료 모델인 HuatuoGPT-o1은 중국어에서 43.40%, 영어에서 47.77%의 평균 성능을 보였습니다. 이 성능은 실제 인간 성과(중국어 85.00%, 영어 87.51%)에 비해 여전히 낮은 수치입니다. 논문에서는 이러한 결과를 기반으로 LLM의 의료 분야에서의 적용 가능성을 더 잘 이해하고, 향후 연구 방향을 제안합니다.



### GTBench: A Curriculum-Grounded Benchmark for Evaluating LLMs as Mathematical Research Assistants in Graph Theory (https://arxiv.org/abs/2606.03144)
Comments:
          19 pages, 5 figures, 7 tables

- **What's New**: GTBench는 LLMs(대형 언어 모델)의 그래프 이론에서 수학적 연구 도우미로서의 수학적 추론 능력을 평가하기 위해 설계된 첫 번째 커리큘럼 기반 벤치마크입니다. 이 벤치마크는 세 그룹으로 나뉜 63개의 문제로 구성되어 있으며, 각 그룹은 증가하는 난이도에 맞춰 설정되어 있습니다. GTBench는 전통적인 그래프 이론 교육의 발전 과정을 반영하여 문제들을 조직하여, LLM의 수학적 도우미로서의 신뢰성을 평가할 수 있는 기초를 다집니다.

- **Technical Details**: GTBench는 세 가지 난이도 그룹으로 구성되어 있습니다: 그룹 1은 학부 수준의 정의와 기본 속성, 그룹 2는 알고리즘 추적과 구조적 추론, 그룹 3은 대학원 수준의 증명 구축을 포함합니다. 선택된 문제들은 검증된 학술 자료에서 가져온 것으로, 그룹 1과 2는 정확한 일치를 기준으로 평가하며, 그룹 3는 하이브리드 방식으로 인간 전문가와 LLM의 평가를 결합하여 신뢰성을 확보합니다.

- **Performance Highlights**: 실험 결과, GPT-5 모델이 그룹 1에서 95.8%의 성능을 보였으며, 그룹 3에서는 82%의 정확성을 유지했습니다. 반면, 다른 모델은 난이도가 증가함에 따라 성능이 급격히 저하되었습니다. Llama 모델은 그룹 3에서 인간 평가 시 0%의 점수를 기록했으며, 오류 분석 결과 그룹 1과 2에서는 올바른 알고리즘이 잘못 실행되는 실패가 주요한 문제로 나타났습니다.



### Think-Before-Speak: From Internal Evaluation to Public Expression in Multi-Agent Social Simulation (https://arxiv.org/abs/2606.03137)
- **What's New**: 본 논문은 LLM(large language model) 기반의 다중 에이전트 시뮬레이션 프레임워크인 TBS(Think-Before-Speak)를 도입하여 사회적 상호작용의 내부 평가 과정을 시뮬레이션할 수 있는 새로운 방법을 제시합니다. TBS는 에이전트의 개인적인 추론과 공개 발화 생성을 분리하여, 공적인 발언이 이루어지기 전 내부 상태를 지속적으로 업데이트 할 수 있게 설계되었습니다. 이를 통해 에이전트는 발화를 하기 전 자신의 의견을 평가하고, 자신의 사고를 명확히 반영할 수 있도록 합니다.

- **Technical Details**: TBS는 간격 기반의 다중 에이전트 시뮬레이션 프레임워크로, 모든 에이전트는 공유된 대화 이력과 각자의 기억을 바탕으로 내부 상태를 업데이트합니다. 이러한 내부 상태는 불화와 관련된 평가, 인식되는 의견 분위기, 고립 위험, 반응 전략 및 발화 의지를 포함합니다. 시스템은 경쟁하는 발화 의도를 해결하고, 각 간격마다 하나의 발화만을 공적인 대화에 반영함으로써 내부 평가와 공적인 상호작용이 시간에 따라 공진화하도록 합니다.

- **Performance Highlights**: TBS는 기후 관련 정책 문제에 대한 타운홀 논의에서 평가되었습니다. 결과는 TBS가 일관성 있는 내부 상태 추적을 생성하며, 이러한 추적은 발화 배정, 침묵 및 기억 조건에 따라 체계적으로 변하는 것을 보여줍니다. 특히, 불화 관련 평가가 에이전트의 발화 의지를 증가시키는 반면, 침묵 강압 평가는 이를 감소시키는 경향을 보였습니다. 이러한 결과는 TBS가 내부 평가에서 공적인 표현까지의 경로를 관찰 가능하고 분석 가능하게 함으로써 메커니즘에 민감한 사회 시뮬레이션을 지원함을 시사합니다.



### Uncertainty-Aware Clarification in LLM Agents with Information Gain (https://arxiv.org/abs/2606.03135)
- **What's New**: 이 논문은 대형 언어 모델(LLM) 에이전트가 불확실한 사용자 지침하에서 발생하는 문제를 해결하기 위해 목표 지향적인 설명 프레임워크를 제안합니다. 이 프레임워크의 핵심은, 정보 획득 보상(Information Gain Reward)을 도입하여 설명 질문의 유용성을 정량화하고 이를 통해 불확실성을 줄이도록 LLM을 훈련합니다. 이 방법은 `Clarifier-augmented τ-Bench` 환경에서 검증되었으며, 설명이 없는 기본선보다 성공률을 3.7% 향상시키는 결과를 보였습니다.

- **Technical Details**: 제안된 프레임워크는 사용자 목표에 대한 잠재적 불확실성을 줄이기 위해 설명 질문의 영향을 측정하는 정보 이론적 접근 방법입니다. 이 논문에서 제안된 방식은 설명 과정을 베이지안 신념 업데이트로 모델링하고, 설명 그룹간의 교환 후 실제 목표에 대한 확률 질량의 이동을 측정하여 질문의 유용성을 정량화합니다. 또한, 이 내재적 보상 신호를 사용하여 불확실한 정보 회복을 최적화하는 질문 전략으로 LLM을 훈련하는 Decoupled Advantage Policy Optimization (DAPO) 방법론을 적용합니다.

- **Performance Highlights**: 이 연구는 제공된 τ-Bench 환경에서 사용자 감화성이라는 부분적인 관찰 환경에서 에이전트가 언급된 불확실성을 줄이기 위해 질문을 필요로 할 때만 개입한다는 것을 강조합니다. 실험 결과, 제안된 방법은 평균적으로 필요한 상호작용 단계 수를 0.3만 증가시키면서도 특정 작업에서의 성공률을 개선했습니다. 이 작업은 에이전트와 도구 사용 간의 상호작용에서 어떻게 질문이 실행과정을 지속적으로 보완할 수 있는지를 보여줍니다.



### EvoTrainer: Co-Evolving LLM Policies and Training Harnesses for Autonomous Agentic Reinforcement Learning (https://arxiv.org/abs/2606.03108)
- **What's New**: 본 논문에서 제안하는 EvoTrainer는 자율적으로 LLM 정책과 훈련 측면 진단 도구를 공진화하는 프레임워크입니다. 이는 에mpirical feedback를 활용하여 롤아웃 수준의 증거를 진단하고, 진단을 수정하며, 개입을 백테스트하고, 재사용 가능한 기술을 축적합니다. EvoTrainer는 기존의 자체적으로 발전하는 자율 실험 시스템과 달리, 훈련 인프라를 진화하는 객체로 보는 접근 방식을 채택하여 혁신적인 방법론을 제시합니다.

- **Technical Details**: EvoTrainer는 두 가지 과정으로 구성돼 있습니다: 정책 자체 진화 및 훈련자 자체 반성. 정책은 실행 가능한 훈련 버전을 생성하고 비교, 가지치기, 증진, 병합을 통해 발전하는 반면, 훈련자는 기존의 메트릭, 분석기 및 백테스트가 불충분할 때 진화합니다. 이를 통해 훈련 중 요소 간의 상호 작용을 효과적으로 관리함으로써 훈련 성과를 향상시킵니다.

- **Performance Highlights**: EvoTrainer는 수학적 추론, 경쟁 프로그래밍 코드 생성 및 리포지토리 수준 소프트웨어 공학에서 평가되었습니다. 모든 분야에서 no-RL 기반보다 일관되게 개선되었으며, 동일한 데이터, 코드베이스, 평가 프로토콜 하에서 인간이 설계한 RL 기준을 초과 또는 동일하게 매치했습니다. 가장 큰 성과는 소프트웨어 공학(SWE)에서 나타났으며, 이에 대한 세부 분석은 전략이 도메인별로 어떻게 다르게 발전하는지를 보여주었습니다.



### DeskCraft: Benchmarking Desktop Agents on Professional Workflows and Human-in-the-Loop Collaboration (https://arxiv.org/abs/2606.03103)
- **What's New**: 새롭게 소개하는 DeskCraft는 전문적인 창작 및 엔지니어링 워크플로우를 평가하기 위한 538개의 과제로 구성된 데스크탑 GUI 벤치마크입니다. 기존 벤치마크가 짧고 단순한 작업에 집중하는 반면, DeskCraft는 50단계 이상의 긴 실행 과정을 필요로 하는 작업을 포함하며, 프로페셔널 소프트웨어에서의 인간-에이전트 협업을 공 formal하게 정의합니다.

- **Technical Details**: DeskCraft는 다단계 난이도 분류(L1, L2, L3)를 통해 작업의 복잡성을 평가합니다. 특히 L3 난이도는 실제 전문 시나리오에서 파생된 것으로, 사용자 피드백에 따른 동적 의도 변화와 에이전트의 정보 요청을 반영하는 상호작용 프로토콜을 가지고 있습니다. 이 프로토콜은 중간 시점 및 완료 후 상호작용을 구성하여 실제 협업 패턴을 포괄적으로 묘사합니다.

- **Performance Highlights**: 18개의 에이전트를 평가한 결과, GPT-5.4는 표준 작업에서 31.6%, 상호작용 작업에서 27.6%의 정확도를 기록했습니다. 그러나 긴 실행 단계에서의 성능은 유의미한 낮은 수준을 보이며, 에이전트가 참조를 요청하는 경우도 드문 것을 보여줍니다. 이를 통해 단순한 명령 실행에서 지속적인 워크플로 계획 및 인간-에이전트 협업의 필요성으로 발생하는 병목 현상이 나타났습니다.



### From Long News to Accurate Forecast: Importance-Aware Fusion and PRM-Guided Reflection for Time Series Forecasting (https://arxiv.org/abs/2606.03097)
- **What's New**: 이 논문에서는 뉴스 기사를 시계열 예측에 통합하여 예측의 정확성을 높이는 새로운 프레임워크를 제안합니다. 기존 LLM(대규모 언어 모델) 기반의 예측 파이프라인의 두 가지 주요 문제, 즉 긴 뉴스 기사의 압축과 비효율적인 보조 뉴스 검색을 해결합니다. 특히, 중요성을 인식한 뉴스 압축 모듈과 프로세스 보상 모델(Reward Model, PRM)을 도입하여 전반적인 예측 성능을 향상시킵니다.

- **Technical Details**: 모델은 두 가지 주요 구성 요소로 이루어져 있습니다. 첫째, 중요성 보상 모델을 통해 예측 유용성을 가진 기사에 대해 압축 예산을 할당하고, 둘째, PRM을 이용해 현재 예측 오차 및 이전 선택 이력을 기반으로 보조 뉴스 기사를 평가합니다. 이러한 오프라인 개선 방법을 통해 제한된 컨텍스트 내에서 중요한 기사를 효과적으로 통합할 수 있습니다.

- **Performance Highlights**: 실험 결과, 이 방법은 금융, 에너지, 교통, 비트코인 예측 벤치마크에서 기존 방법들보다 예측 정확도를 유의미하게 향상시켰습니다. 또한, 정제(iteration) 과정의 반복 횟수를 평균 24.8%까지 줄이며, 최대 37.6%의 감소를 기록했습니다. 수천 개의 토큰에 걸친 관련 기사에도 불구하고 여전히 효과적인 성능을 보여주었습니다.



### Decomposing how prompting steers behavior (https://arxiv.org/abs/2606.03093)
Comments:
          59 pages, 41 figures

- **What's New**: 이 논문은 언어 모델(LLMs)과 비전-언어 모델(VLMs)에서 프롬프트(prompt)가 내부 표현을 어떻게 변형시키고 행동을 유도하는지에 대한 기계적 질문을 다룹니다. 연구진은 프롬프트를 내용의 표현 기하(shape geometry)의 변형으로 간주하는 중첩 기하 분해(nested geometric decomposition) 프레임워크를 도입했습니다. 이는 프롬프트 쌍마다 동일한 자극의 표현을 정렬하고, 다양한 자극 불변 맵을 사용하여 표현의 변화를 분석합니다.

- **Technical Details**: 프롬프트-induced 변화는 번역(translation), 고정된 변형(rigid transformation with uniform scaling), 축 방향 스케일 조정(sequential axis scaling), 아핀 변형(affine transformation), 비선형 변형(nonlinear transformation)이라는 5가지 변형 클래스(classe)로 분해됩니다. 또한, 연구진은 각 변형 맵을 인과적(causal) 활성화 중재(intervention)로 적용하여 프롬프트로 유도된 표현 기하를 측정하고 해당 프롬프트로 변환된 상태가 목표 프롬프트의 표현 기하를 재현하는지를 확인합니다. 이를 통해 각 계층(layer)에서 수집된 프로파일이 모델마다 다르게 나타나는 프롬프트 경로의 전략을 드러냅니다.

- **Performance Highlights**: 연구진은 세 가지 LLM, 세 가지 VLM, 여섯 개의 다양한 텍스트 및 이미지 데이터셋에서 실험을 진행했습니다. 결과적으로, 프롬프트는 항상 지시된 작업 구조로 표현을 재구성하며, 대부분의 프롬프트 효과는 저복잡도(저비용)의 맵에 의해 포착됩니다. 아핀 변형이 목표 프롬프트 작업 기하를 거의 복구할 수 있는 첫 번째 계층으로 식별되었고, 이는 행동 향상과 상응하는 결과를 가져옵니다.



### The Shadow Price of Reasoning: Economic Perspective on Optimal Budget Allocation for LLMs (https://arxiv.org/abs/2606.03092)
- **What's New**: 이 논문은 대형 언어 모델(Large Language Models, LLMs)의 성능 개선을 위한 추론 시간 스케일링(inference-time scaling)의 접근 방식을 제안합니다. 많은 사용자가 동시에 접근하는 클라우드 API를 통해 서비스를 제공하거나 자원 제한이 있는 엣지 디바이스에서 모델을 실행해야 하는 현실적인 상황을 고려하여, 추론 예산 할당(inference budget allocation)을 경제적 원칙에 따라 글로벌 제약 최적화 문제로 정의합니다. 저자는 'Constrained Latent-utility Equilibrium Allocation for Reasoning' (CLEAR)이라는 새로운 시스템을 제안하며, 이는 비합리적인 자원 분배를 피하고 해결 가능한 쿼리에 자원을 재분배합니다.

- **Technical Details**: CLEAR는 지연된 유틸리티 곡선(latent utility curve)을 사용하여 각 쿼리의 임계값(threshold)을 모델링하고, 빠른 이분검색(fast bisection search)을 통해 글로벌 그림자 가격(global shadow price)을 발견합니다. 이 시스템은 LLM의 재훈련 없이 플러그 앤 플레이 방식으로 작동하며, 각 쿼리의 토큰 보고를 엄격하게 결정하기 위해 래므베르트 W 함수(Lambert W function)에서 도출된 폐쇄형(close-form) 정책을 적용합니다. 저자들은 각 쿼리가 관련 유틸리티 곡선에 따라 예산을 분배받아야 함을 주장하며, 이는 자원 제약된 환경에서의 최적화를 가능하게 합니다.

- **Performance Highlights**: 다양한 추론 작업에 대한 테스트를 통해, CLEAR는 총 토큰 비용과 평균 정확도 간의 파레토 경계(Pareto frontier)를 상당히 개선함을 보여주었습니다. 자원이 부족한 환경에서는 전통적인 균일 분배(uniform allocation) 방식보다 최대 3배 더 높은 글로벌 정확도를 달성했습니다. 실험 결과 CLEAR의 성능은 다양한 하이퍼파라미터 선택과 예측기 노이즈에 대해 강건하다는 것을 입증했습니다.



### DELTAMEM: Incremental Experience Memory for LLM Agents via Residual Trees (https://arxiv.org/abs/2606.03083)
- **What's New**: 이 논문은 경험 데이터를 독립적이고 평면적으로 저장하는 것에 따른 중복성과 충돌 문제를 해결하기 위해 DeltaMem이라는 새로운 메모리 프레임워크를 도입합니다. DeltaMem은 목표 조건화된 태스크 경험과 장면 수준의 환경 지식을 저장하는 두 개의 독립적 잔여 트리(Residual Tree)를 사용하여 메모리를 구성합니다. 이 구조는 기억을 관리하는 방식을 변화시키고, 더 나아가 각 경험이 발생하는 방법론을 다르게 하여 각각의 메모리를 간소화할 수 있습니다.

- **Technical Details**: DeltaMem의 두 가지 주요 원리는 구조적 분리와 잔여 압축(residual compression)입니다. 태스크-트리(Task-Tree)와 환경-트리(Env-Tree)라는 두 개의 독립적인 계층 메모리 구조를 유지하여, 각각 목표 조건화된 행동 경험과 장면 수준의 기술적 정보를 저장합니다. 이 구조는 각 경험을 기존 지식에 대한 증분(delta)으로 저장하여, 과거 경험과의 연관성을 극대화하면서도 중복을 최소화합니다.

- **Performance Highlights**: 다양한 상호작용 환경에서 DeltaMem의 성능을 평가한 결과, 기존의 경험 메모리 기법에 비해 일관되게 뛰어난 성과를 보였습니다. 특히, DeltaMem은 메모리 중복을 줄이고 지식 충돌을 방지함으로써 에이전트의 의사결정 품질을 향상시키는 것으로 나타났습니다. 이로 인해 DeltaMem은 추후 연구 및 실제 응용 분야에서 활용될 가능성이 매우 높습니다.



### CORE: Conflict-Oriented Reasoning for General Multimodal Manipulation Detection (https://arxiv.org/abs/2606.03066)
Comments:
          Accepted by ICML 2026

- **What's New**: 이번 논문에서는 멀티모달 가짜 뉴스 탐지의 새로운 패러다임을 제시합니다. CORE 프레임워크는 MLLMs에 명시적 충돌 캡처 능력을 부여하여 정보 조작을 인식하는 데 큰 도움을 줍니다. 이를 통해 새로운 조작 유형에 신속하게 적응할 수 있는 능력을 갖추게 됩니다.

- **Technical Details**: CORE는 Conflict Attribution Corpus (CAC)를 구축하여 세밀한 충돌 요인과 출처에 대한 주석을 제공합니다. 이 데이터는 모델이 다양한 멀티모달 조작을 인식하는 데 필요한 훈련을 지원하며, 충돌 인식 향상을 위한 Conflict-Perception Training (CPT) 과정을 포함합니다. CORE는 최소한의 샘플로 새로운 조작 유형에 대한 탐지 성능을 극대화할 수 있습니다.

- **Performance Highlights**: CORE는 기존 최첨단 모델들을 능가하는 성능을 보여주며, 이는 다양한 조작 패턴에 대한 일반화 능력을 강화합니다. 14,000개의 샘플이 포함된 CAC 데이터셋과 코드가 공개되어 있어, 연구자들이 충돌 추론을 연구할 수 있는 견고한 기준을 제공합니다. 이러한 접근은 적은 수의 샘플로도 신속한 학습과 일반화된 탐지 능력을 가능하게 합니다.



### SkillDAG: Self-Evolving Typed Skill Graphs for LLM Skill Selection at Sca (https://arxiv.org/abs/2606.03056)
Comments:
          19 pages, 5 figures

- **What's New**: 본 논문에서는 SkillDAG를 제안하며, 이는 대규모 스킬 라이브러리에서의 올바른 스킬 집합 선택을 구조적 문제로 모델링합니다. SkillDAG는 LLM 에이전트가 실행 시간에 호출할 수 있는 구조적 검색 인터페이스로, 에피소드 동안 에이전트의 편집으로 구조적 지식을 축적합니다. 기존의 고정된 검색 파이프라인과는 달리, 실행에서 발생하는 증거에 따라 그래프의 구조가 진화할 수 있습니다.

- **Technical Details**: SkillDAG는 타이핑된 방향 그래프(typed directed graph) 형태로 스킬 간 관계를 모델링하며, 에이전트가 실행 중에 스킬을 검색하고 제안할 수 있게 합니다. 제안-커밋 프로토콜은 에저의 사용 이력을 기반으로 실제 실행된 엣지를 등록하면, 그래프는 여러 에피소드에 걸쳐 구조가 누적됩니다. 이는 비순환성(acyclicity), 비모순(non-contradiction), 추가 전용 가역성(append-only reversibility) 등의 세 가지 구조적 불변성을 기반으로 설계되었습니다.

- **Performance Highlights**: SkillDAG는 ALFWorld와 SkillsBench에서 MiniMax-M2.7을 사용하여 67.1% 성공률과 27.3% 보상을 기록하였으며, 이는 가장 강력하게 보고된 Graph-of-Skills 베이스라인보다 각각 +12.8 및 +8.6 포인트 우수한 성능입니다. gpt-5.2-codex에서도 이점이 유지되며, SkillsBench Ret@K은 65.5에서 78.2로 상승하였습니다. 이러한 성장은 후보순위(rank) 유지 및 지식 임베딩(embedding) 방식을 보완하는 온라인 수정이 가능하게 하여 이루어졌습니다.



### ToolGate: Token-Efficient Pre-Call Control for Tool-Augmented Vision-Language Agents (https://arxiv.org/abs/2606.03054)
- **What's New**: 이 논문은 Vision-Language Model(VLM) 에이전트가 도구를 사용하는 과정에서의 미리 호출 제어(pre-call control) 문제를 다룹니다. 이 연구는 제안된 도구 호출이 실행될 가치가 있는지 고려하는 ToolGate라는 경량 외부 컨트롤러를 도입하여, 도구 호출의 실행 전 비용과 효율성을 평가하는 방법을 제시합니다.

- **Technical Details**: ToolGate는 동적 추적 텍스트와 간단한 구조적 특성을 기반으로 도구 호출 실행 여부를 결정합니다. 이 시스템은 VLM 구조를 직접 수정하지 않고도 수행되며, 제안된 도구 호출이 실제로 유용한지를 판단하여, 유용하지 않은 호출을 건너뛰도록 합니다. ToolGate는 Qwen3-VL을 활용해 64-69%의 토큰 비용 절감을 달성했습니다.

- **Performance Highlights**: ToolGate를 통해 평균 정확도를 유지하면서 실행된 도구 호출과 토큰 사용을 상당히 줄일 수 있음을 보여주었습니다. 특히 Qwen3-VL-30B에서 매칭 도메인 훈련을 통해 평균 정확도를 1.65 포인트 향상시키며, 총 토큰 비용은 33% 감소했습니다. 이는 도구 호출이 유용한 상황을 명확히 식별하고 비용을 관리하는 것이 중요함을 시사합니다.



### RelGT-AC: A Relational Graph Transformer for Autocomplete Tasks in Relational Databases (https://arxiv.org/abs/2606.03040)
Comments:
          12 pages, 6 figures. Code and model checkpoints available at this https URL

- **What's New**: 이번 논문은 Relational Deep Learning (RDL) 접근 방식을 통해 다중 테이블, 이질적인 구조의 관계형 데이터베이스에서 머신러닝을 최적화하고자 합니다. 특히, 새로운 모델인 RelGT-AC는 관계형 그래프 트랜스포머를 기반으로 하여 멀티테이블 구조에 그래프 신경망을 적용합니다. 이 모델은 기존의 데이터 열 값을 예측하는 자동 완성(autocomplete) 작업을 수행하며, 이는 고객의 가격 협상 과정을 위한 핵심 기술입니다.

- **Technical Details**: RelGT-AC는 세 가지 주요 기여로 구성되어 있습니다: 첫째, 서브그래프 인코딩 중 타겟 열을 마스킹(masking)하는 전략을 통해 단순한 솔루션을 방지합니다. 둘째, 이 모델은 이진 분류, 다중 클래스 분류 및 회귀 자동 완성 작업을 단일 모델로 처리할 수 있는 통합(task head) 헤드를 제공합니다. 셋째, TF-IDF 텍스트 인코더는 자유 텍스트 열을 자동으로 감지하고 인코딩하여 카테고리 인코더가 버리는 중요한 어휘 신호를 복구합니다.

- **Performance Highlights**: RelGT-AC는 RelBench v2의 세 가지 데이터셋에서 수행된 7개의 작업에서 GraphSAGE 기준 모델을 초과 달성했습니다. 특히, 텍스트 중심의 자격 작업에서 TF-IDF 인코더를 통해 최대 +10 AUROC 포인트의 성과를 거두었습니다. 이 연구는 코드와 모델 체크포인트를 공개하여 재현성과 추가 연구를 지원합니다.



### TriEval: A Resource-Efficient Pipeline for LLM Bias, Toxicity, and Truthfulness Assessmen (https://arxiv.org/abs/2606.03036)
- **What's New**: 이번 논문에서는 LLMs (Large Language Models)의 진화와 함께 이러한 모델들이 의료, 교육, 정부 서비스 등 광범위하게 사용되고 있음을 강조하고 있습니다. LLMs의 도입이 증가함에 따라 안전성과 공정성을 보장하기 위한 지속적인 평가의 필요성이 커지고 있습니다. 여러 LLM 평가 도구들이 존재하지만, 이들 대부분은 단일 파라미터만을 평가하거나 대규모 컴퓨팅 리소스가 필요하다는 단점이 있습니다.

- **Technical Details**: TriEval은 LLM의 출력 결과를 여러 파라미터, 즉 bias(편향), toxicity(유해성), truthfulness(진실성) 등을 동시에 평가할 수 있도록 설계된 시스템입니다. 이 시스템은 표준 노트북에서도 실행 가능하며, GPU 클러스터와 같은 고급 컴퓨팅 리소스가 필요 없습니다. TriEval은 Llama 3 8B, Mistral 7B, Gemma 2 9B, Claude Haiku 등 네 가지 모델에서 테스트를 진행했습니다.

- **Performance Highlights**: 테스트 결과는 오픈 소스 모델과 클로즈드 소스 모델 간의 명확한 차이를 보여주며, 특히 유해성과 진실성 측면에서 두드러진 차이가 있음을 발견했습니다. TriEval은 제한된 컴퓨팅 자원을 가진 연구자들이 더 넓은 접근을 할 수 있도록 오픈 소스로 출시됩니다.



### AUDITFLOW: Executable Symbolic Environments for Structured Financial Reporting Verification (https://arxiv.org/abs/2606.03031)
- **What's New**: 이번 연구에서는 AuditFlow라는 새로운 그래프 기반 다중 에이전트 프레임워크를 제안하여 XBRL(Audit) 감사 검증을 수행합니다. 이 프레임워크는 LLM(대형 언어 모델)의 검색 기능과 결정을 구분하여 개선된 감사 검증을 지원합니다. AuditFlow는 정적 US-GAAP 세금 그래프와 동적 XBRL 문서 그래프를 결합하여 감사 작업에서 필요한 사실 검색, 세금 분류 검사, 수치 확인 및 규칙 평가를 실행합니다.

- **Technical Details**: AuditFlow는 이중 그래프 구조로, 하나는 고정된 US-GAAP 세금 그래프와 다른 하나는 동적인 XBRL 문서 증거 그래프입니다. 감사 쿼리에 대한 분석 시, 에이전트는 검색할 데이터의 종류를 결정하고, 타입이 지정된 도구를 통해 상징적 환경과 상호작용하게 됩니다. 이를 통해 규제적 관점과 증거적 관점에서 두명의 주니어 감사인이 각 케이스를 점검하며, 최종적으로 세니어 감사인이 견해 차이를 조정하고 추가 조사를 요청할 수 있습니다.

- **Performance Highlights**: AuditFlow는 FinAuditing의 FinMR 샘플에서 GPT-5.5를 기반으로 82.09%의 공동 감사 정확도를 달성했습니다. 이는 가장 강력한 기준선 모델보다 14.93포인트 높은 수치입니다. 결정론적 체크를 제거하면 정확도가 17.91%로 떨어지고 잘못된 출력 비율이 35.82%로 증가하는 결과가 나타나는 등, 상징적 환경이 감사 검증 단계에서 필수적인 역할을 수행한다는 점이 강조됩니다.



### Inducing Reasoning Primitives from Agent Traces (https://arxiv.org/abs/2606.02994)
Comments:
          22 pages including appendices

- **What's New**: 이 논문에서는 Reasoning Primitive Induction이라는 단일 통과 메소드를 소개하여, 성공적인 ReAct 추적을 통해 재사용 가능한 추론 동작을 집합화하고 변환합니다. 이러한 추론 동작은 LLM이 호출 시 해석하는 자연어 docstring에 의해 정의되며, 이를 통해 LLM 에이전트가 테스트 시간에 평가할 수 있는 compact library를 생성합니다. 연구 결과, 유도된 라이브러리가 그 추적을 생성한 에이전트보다 성능이 뛰어난 것으로 나타났습니다.

- **Technical Details**: 이 방법은 ReAct 롤아웃의 데이터셋을 수집하여 각 단계의 Thought 문자열을 클러스터링하고, 반복되는 추론 동작을 수식화하도록 설계되었습니다. 각 추론 동작은 이름, 형식화된 입력/출력 서명, 그리고 자연어 설명을 포함하는 튜플로 정의됩니다. 이 논문에서 제시된 primitive는 ReAct 행동 공간에 등록된 typed callable로 구현되며, LLM은 이 호출 시 docstring을 통해 연결된 내용을 해석합니다.

- **Performance Highlights**: 연구에서 유도된 라이브러리는 RuleArena NBA에서 +44pp, MuSR 팀 배정에서 +30pp, NatPlan 회의 계획에서 +22pp로 성능을 향상시키며, 다른 서브태스크에서도 전반적으로 리더를 초과했습니다. 전체적으로, 고정된 구성의 단일 설정이 zero-shot Chain-of-Thought보다 모든 서브태스크에서 개선, 전문가가 직접 작성한 구조와 비교하여 유사하거나 초과하는 성능을 발휘하였습니다.



### WISE-HAR: A Generalizable Ensemble Deep Learning Framework for WiFi-Based Human Activity Recognition (https://arxiv.org/abs/2606.02974)
Comments:
          8 pages, 5 figures

- **What's New**: 이 논문은 WiFi 신호를 활용한 인간 활동 인식(HAR)의 새로운 접근 방식을 제시하고 있습니다. 전통적인 카메라 기반 시스템의 개인정보 보호 문제와 저조도 환경에서도 실패하는 한계를 극복하기 위해, WiFi 기반 HAR는 비침습적이고 경제적인 솔루션을 제공합니다. 이 연구는 'No Presence', 'Walking', 'Walking + Arm-waving' 세 가지 활동을 인식하는 방법을 설명하고 있으며, Wallhack1.8k WiFi 스펙트로그램 데이터셋을 사용했습니다.

- **Technical Details**: 세 가지 주요 개선 사항으로 성능 변동성을 줄이기 위해 5개의 서로 다른 CNN 아키텍처(Deep CNN, Wide CNN, MobileNetV2, ResNet50V2, EfficientNetB0)를 활용한 앙상블 학습, 데이터 세트 크기 한계를 극복하기 위한 강력한 데이터 증강 기법(시계 왜곡, 주파수 마스킹, 노이즈 추가) 및 실제 세계 일반화 능력을 평가하기 위한 크로스 시나리오 평가를 도입했습니다. 이 연구는 이러한 방식으로 WiFi 기반 HAR의 세 가지 주요 문제를 해결합니다.

- **Performance Highlights**: 제안한 앙상블 모델은 LOS 시나리오에서 Biquad 안테나를 사용하여 94.87%의 테스트 정확도를 달성했으며, 이는 가장 우수한 개별 모델보다 0.66% 향상된 결과입니다. 데이터 증강을 통해 Random Forest의 성능이 60%에서 95%로 개선되었고, 크로스 시나리오 평가에서의 정확도 저하도 각각 1.37%와 2.07%에 그쳐 모델의 강력한 일반화 능력을 보여주었습니다.



### What Benchmarks Don't Measure: The Case for Evaluating Abstention Competence in Autonomous Agents (https://arxiv.org/abs/2606.02965)
Comments:
          ACM CAIS 2026: RLEval Workshop Oral Presentation(Best Paper Award)

- **What's New**: 이번 논문은 자율 에이전트의 평가 기준을 다루며, 에이전트가 작업을 얼마나 잘 수행하는지를 측정하는 데 집중하는 기존 방법의 한계를 지적합니다. 연구자들은 안전한 행동을 위한 전제 조건이 부족한 상태에서도 작업을 계속 수행하는 경향, 즉 compliance bias를 정의합니다. 이러한 경향은 보상 신호와 평가 체계에 의해 강화되며, 안전한 행동을 취하는 것에 대해서는 적절한 평가가 이루어지지 않습니다.

- **Technical Details**: 논문에서는 compliance bias의 기원과 그것이 인공지능 시스템의 평가 과정에서 어떻게 고착화되었는지를 설명합니다. 연구자들은 abstention-warranted scenarios의 특성을 정의하였으며, 이는 시스템이 멈춰야 할 사유를 분류한 것입니다. 이를 기반으로 safety rate, usability rate 및 informed refusal rate와 같은 새로운 평가 프로토콜을 제안하고, 여러 에이전트 시나리오에 대한 초기 결과를 보고합니다.

- **Performance Highlights**: 초기 테스트 결과, abstention 메커니즘이 위험한 행동 차단에서 89.2%의 효과를 보여주었으며, 인가된 시나리오에서 87.5%의 사용성 점수를 기록했습니다. 이는 안전성과 사용자 용이성 간의 균형이 고정된 것이 아니라 조정 가능하다는 점을 보여주며, 다양한 모델 간에 그 균형의 형태가 크게 다를 수 있음을 나타냅니다. 이러한 결과는 이후 대화의 출발점으로 제안됩니다.



### Large AI Models in Dental Healthcare: From General-Purpose Systems to Domain-Specific Foundation Models (https://arxiv.org/abs/2606.02914)
- **What's New**: 이번 논문에서는 2020년부터 2026년까지 발표된 대규모 AI 모델을 치과 의료에서 개발, 평가 또는 적용한 97개의 연구를 체계적으로 분석하였습니다. 저자들은 이러한 모델들을 아키텍처 패러다임과 치과 전문화 정도에 따라 분류할 수 있는 2차원 분류 프레임워크를 제안합니다. 이 연구는 AI 모델 간의 관계와 한계점을 포괄적으로 이해하기 위해 체계적인 문헌 검색을 수행하였습니다.

- **Technical Details**: 연구는 PRISMA-ScR 지침을 따르며, PubMed, Google Scholar, Scopus, arXiv의 4개 데이터베이스에서 체계적인 검색을 진행하였습니다. 검색 용어는 'large language model', 'vision-language model', 'transformer' 등 다양한 AI 모델 관련 용어를 포함하였고, 이를 바탕으로 치과 관련 연구를 수집하였습니다. 또한, 저자들은 이 모델들이 서로 어떻게 보완적으로 작용할 수 있는지에 대한 분석도 수행하였습니다.

- **Performance Highlights**: 모델 성능 분석 결과, 언어 생성 모델은 텍스트 기반 작업에서 우수한 성능을 보였으나 이미지 의존 진단에서는 일관되지 않은 결과를 보였습니다. 반면, 치과 특화 모델인 DentVFM 및 OralGPT는 복합적인 다중 모드 작업에서 가장 뛰어난 성능을 입증했으며, 통합된 파이프라인이 단일 모델 접근 방식보다 일관되게 우수한 성과를 나타냈습니다. 하지만 여전히 생성 모델에서의 환각(hallucination), 제한된 주석이 달린 데이터셋, 표준화된 임상 평가 기준의 부재 등의 과제가 해결되어야 안전하게 자율 배포가 가능할 것입니다.



### Handoff Debt: The Rediscovery Cost When Coding Agents Take Over Interrupted Tasks (https://arxiv.org/abs/2606.02875)
- **What's New**: 이 논문은 이전의 코딩 에이전트 평가 방식과는 달리, 여러 에이전트 간의 작업 인수인계에서 발생하는 'handoff debt'라는 개념을 도입합니다. 이는 앞서 작업한 에이전트가 남긴 불완전한 작업 상태로 인해 후속 에이전트가 작업을 재개하는 데 겪는 어려움을 측정하는 새로운 프로토콜을 제안합니다. 이를 통해 실행 비용을 줄이고, 후속 에이전트가 과거의 작업을 얼마나 효율적으로 이어갈 수 있는지를 평가합니다.

- **Technical Details**: 연구는 SWE-bench Verified와 OpenHands 스타일의 코딩 에이전트 환경에서 수행되었습니다. 제안된 프로토콜은 코딩 에이전트의 중단 지점에서 에이전트를 임의로 차단하고, 각 후속 에이전트가 특정 컨텍스트 포맷을 활용하여 작업을 이어가도록 합니다. 이 연구에서는 4가지 핸드오프 뷰(Repository only, Raw trace, Summary notes, Structured notes)를 사용하여 후속 에이전트의 작업 재개 효율을 비교하였습니다.

- **Performance Highlights**: 실험 결과, 컨텍스트 기반 핸드오프 방식이 평균 에이전트 이벤트를 20-59%까지 줄이고, 누적 프롬프트 토큰 수를 42-63% 감소시켰습니다. 성공률에 대한 효과는 모델 의존적이긴 하지만, 전반적인 효율성 개선은 일관되었습니다. 작업의 재개가 얼마나 힘들고 비용이 드는지를 평가하는 새로운 방법론으로, 종래의 문제 해결 여부 이외에 후속 에이전트가 어떤 비용을 들여 작업을 이어가는지를 측정하는 것이 중요함을 알려줍니다.



### When Helping Hurts and How to Fix It: Multi-Agent Debate for Data Cleaning (https://arxiv.org/abs/2606.02866)
Comments:
          27 pages, 4 figures, 12 tables. Includes appendix with full experimental results, prompt templates, and dataset statistics

- **What's New**: 이 연구는 데이터 클리닝에서 다중 에이전트 토론(multi-agent debate)이 언제 도움이 되고 언제 해가 되는지를 분석합니다. 총 세 개의 벤치마크, 네 개의 모델 계열 및 6,000개 이상의 작업-조건 쌍을 통해, 토론이 모든 모델에서 생성 품질을 저하시키지만 오류 감지는 크게 개선된다는 결과를 발견했습니다. 이 연구는 Critic의 검증 가능성이 Generator의 기본 정확도를 초과할 때 토론이 효과적이라는 조건을 제시합니다.

- **Technical Details**: 연구는 Critic이 Generator의 제안을 검증하는 방식을 통해 데이터 클리닝의 오류를 줄이기 위한 다중 에이전트 시스템의 방법론적 접근을 제공합니다. 연구에 따르면, Critic과 Generator를 분리해 각각 독립적으로 작동시키는 방식이 필요하며, 코드 실행(ground code execution) 및 evidence-gated generation이 결합될 때 가장 효과적인 결과를 도출합니다. 데이터 클리닝 작업에서 발생하는 비효율성인 critique-induced confusion(CIC)을 해결하기 위한 여러 실험 결과도 보고했습니다.

- **Performance Highlights**: 비교 실험을 통해 제안된 조건이 9개 작업 타입 모두에서 정확한 예측을 제공하며, 19개의 출판된 비교에서 제로의 false positive를 기록하여 일반화 가능성을 입증했습니다. 최종적으로, 연구팀은 토론이 필요한 경우와 아닌 경우를 명확하게 정의하는 의사결정 규칙을 제공하여 실무자들이 중요 품질 이득을 놓치지 않도록 도움을 주고 있습니다. 한정된 입력 구성을 넘어서는 최초의 debate 구성 방식을 제시하며, 단일 에이전트보다 5.3포인트 이상의 개선을 입증했습니다.



### Don't Gamble, GAMBLe: An Analytical Framework for AI-Driven Research Systems (https://arxiv.org/abs/2606.02863)
Comments:
          Preprint. 21 pages (10 main, 11 appendix). 6 figures (2 in main, 4 in appendix)

- **What's New**: 이번 논문에서는 AI 주도 연구 시스템(AI-Driven Research Systems, ADRS)의 동작 메커니즘을 효과적으로 분석하기 위한 GAMBLe 프레임워크를 소개합니다. GAMBLe은 ADRS의 성능을 네 가지 매개변수(G, $	ext{그리너}$; $	ext{A}$, $	ext{어세서}$; $	ext{M}$, $	ext{발견 메커니즘}$; $B$, $	ext{예산}$)와 하나의 구성 객체(효과적인 경관 $L_{	ext{eff}} = 	ext{A} ullet 	ext{G}$)로 분해하여 분석합니다. 이 프레임워크는 서로 다른 생성기-어세서 쌍이 문제에 따라 구조적으로 다른 최적화 경관을 유도한다는 것을 보여줍니다.

- **Technical Details**: GAMBLe 프레임워크는 먼저 각 구성 요소가 어떻게 상호 작용하는지를 프로세스 모델로 정의합니다. 논문에서는 비 마르코프(best-score dynamics) 동적 시스템과 생성기 의존적 효과 경관(effective landscapes)에 대한 두 가지 구조적 결과를 증명합니다. 생성기 G는 주어진 맥락(context)에서 후보 솔루션을 생성하며, 어세서 A는 후보 솔루션을 점수로 매핑하고, 발견 메커니즘 M은 탐색을 지시합니다.

- **Performance Highlights**: GAMBLe 프레임워크는 760개 이상의 복제 실행과 46,000회 이상의 반복을 통해 검증되었습니다. 한정된 예산으로도 구성 요소 선택이 성능을 13-67% 개선하고 탐색 효율성을 6-39배 높일 수 있다는 점을 강조합니다. 이 연구는 다양한 생성기 및 탐색 메커니즘이 문제 해결에 미치는 영향을 실질적으로 드러내며, 전통적인 분석 도구들이 ADRS에서 자주 위배된다는 점을 제기합니다.



### Toward a Modular Architecture for Embedded AI Agent Systems at the Edg (https://arxiv.org/abs/2606.02862)
- **What's New**: 이번 논문에서는 제한된 메모리와 에너지 자원으로 인해 Embedded Systems (내장 시스템)에서 Autonomous Agent (자율 에이전트)의 배치가 어려운 문제를 다루고 있습니다. 기존의 Large Language Models (대형 언어 모델) 기반 시스템의 개념을 Embedded Agent Systems (임베디드 에이전트 시스템)으로 발전시키며, 이를 통해 경량화된 On-Device Agents와 Cloud-Augmented Agents 사이의 간극을 메우는 모듈식 아키텍처를 제안합니다. 또한, Governance Layer를 도입하여 분산된 자율 장치 fleets를 관리하며 안전성과 정책 집행을 보장하고 있습니다.

- **Technical Details**: 제안된 아키텍처는 Embedded Devices에서 작동할 수 있도록 모듈 단위로 구성되어 있습니다. 하드웨어의 성격에 따라 Fully Autonomous On-Device Agents와 Tethered Cloud Agents (클라우드에 연결된 에이전트)를 구분합니다. 여기서는 Low-latency, energy efficiency (에너지 효율성), tight memory and compute budgets (제한된 메모리와 연산량)와 같은 요구사항을 기반으로 하여 최적화된 TinyML 모델을 활용하게 됩니다.

- **Performance Highlights**: 이 연구는 Smart Agriculture, Predictive Maintenance (예측 유지 관리), Smart Home와 같은 세 가지 응용 분야에서의 Trade-offs에 대한 개념적 평가를 제공합니다. 지역 Small Language Models (SLMs)가 가치를 제공할 수 있는 경우와 클라우드 오프로드가 더 우수한 경우를 비교하여, 자율 에이전트의 성능을 높일 수 있는 다양한 전략을 제시하고 있습니다. 또한, 안전성, 정책 집행 및 관찰 가능성을 확보하여 자율 장치의 분산 관리가 가능하다는 점도 강조하고 있습니다.



### Thinking Past the Answer: Evaluating Harmful Overthinking in Large Reasoning Models (https://arxiv.org/abs/2606.02835)
- **What's New**: 이번 논문에서는 대규모 추론 모델(Large Reasoning Models, LRMs)의 성능 개선이 추가 컴퓨팅 자원을 배정함으로써 이루어질 수 있으나, 이러한 장기적인 추론이 항상 유리하다는 가정이 충분히 검토되지 않았음을 지적합니다. 연구팀은 모델이 정답에 도달한 후 추가적인 추론이 결과를 개선하는지 아니면 편차를 초래하는지 조사합니다. 이를 위해 'prefix-level trajectory evaluation protocol'을 도입하여 적절한 추론 예산을 정의하고 초기 정답 도출과 그 이후의 추론 과정을 평가합니다.

- **Technical Details**: 연구에서는 'reasoning sufficiency(추론 충분성)'의 관점에서 과도한 추론(overthinking)을 정형화합니다. 모델이 정답을 출력하는데 필요한 최소한의 추론 예산을 정의하고, 이를 통해 'verbose overthinking(장황한 과도한 추론)'과 'harmful overthinking(유해한 과도한 추론)'을 구분합니다. 또한, 모델이 정답에 도달한 후의 추론 길이를 측정하기 위해 prefix-level performance(접두사 수준 성능) 평가 프로토콜을 적용, 성능을 정량화하여 나타냅니다.

- **Performance Highlights**: 다양한 멀티모달 벤치마크에서 많은 문제들이 아주 적은 수의 추론 단계로 해결될 수 있음을 발견했습니다. 특히, 최초의 정답 접두사에서 멈추는 것이 표준 추론 방식보다 평균 21% 더 높은 정확도를 보이는 것으로 나타났습니다. 또한, 조기 중단(early stopping)과 같은 효율성 전략이 장황한 과도한 추론을 감소시키기는 하나, 유해한 과도한 추론을 완화하는 데는 실패하는 경향이 있음을 밝혔습니다.



### An Exploration of Collision-based Enemy Morphology Generation (https://arxiv.org/abs/2606.02832)
- **What's New**: 이 연구는 적 생성(Enemy Generation) 분야에서 적의 형태(morphology)를 생성하는 새로운 접근을 제안합니다. 기존의 연구는 주로 적의 행동이나 특징에 중점을 두었지만, 이 논문은 적의 신체 계획과 충돌 정보에 초점을 맞추고 있습니다. 연구자들은 세 가지 새로운 방법을 개발하였고, 이는 플레이어의 충돌 정보를 기반으로 적의 형태를 생성합니다.

- **Technical Details**: 저자들은 2D 격자 기반 환경에서 적의 형태 데이터를 충돌 정보를 통해 기록합니다. 이들은 강화 학습(Reinforcement Learning), A* 탐색(A* search), 신경망(Neural Networks) 등 세 가지 최적화 전략을 탐구하고, 이를 위한 진화 알고리즘 기반의 기준선을 설정하였습니다. 결과적으로, 세 가지 접근 방식 모두 진화 기반 기준선을 초과하거나 동등한 성능을 보였으며, 각 접근 방식의 강점과 약점이 명확히 나타났습니다.

- **Performance Highlights**: 세 가지 접근 방식이 제공하는 성능은 상대적으로 뛰어나며, 특정 기계적 행동에 의해 제압 가능하지만 다른 기계적 행동으로는 쉽게 넘어설 수 없는 적의 형태를 생성하는 데 성공했습니다. 이 연구는 개발자들이 효율적으로 적의 형태를 제안하고 조정할 수 있는 새로운 방안을 제시하여 게임 디자인에서의 의사결정을 지원합니다. 이러한 요소는 게임 플레이 경험을 더욱 풍부하고 다양하게 만드는 데 기여할 것으로 예상됩니다.



### Traj-Evolve: A Self-Evolving Multi-Agent System for Patient Trajectory Modeling in Lung Cancer Early Detection (https://arxiv.org/abs/2606.02812)
- **What's New**: 이번 연구에서는 Traj-Evolve라는 자가 진화하는 다중 에이전트 시스템을 제안하며, 이 시스템은 폐암 예측을 위한 환자 경로 모델링에서 기존의 제약을 해결합니다. Traj-Evolve는 두 가지 보완적 진화 메커니즘을 통해 환자 데이터를 처리하여 환자와의 연관성을 강화하고 성능을 지속적으로 개선합니다. 특히, 경험 풀(Experience Pool)과 다중 에이전트 강화 학습(Multi-Agent Reinforcement Learning) 기법이 결합되어 진화를 촉진합니다.

- **Technical Details**: Traj-Evolve는 경험 풀(ExPool)을 통해 유사한 환자 데이터를 효과적으로 검색하고, 다중 에이전트 강화 학습(MARL)을 통해 에이전트 간 협력을 최적화합니다. 이 시스템은 환자마다 독립적으로 처리되는 기존 LLM 기반 시스템의 한계를 극복하고, 이전의 검증된 사례를 활용하여 평가 및 예측의 정확성을 높입니다. 이를 통해 환자의 경과에 대한 복잡한 시계열 추론을 가능하게 합니다.

- **Performance Highlights**: Traj-Evolve는 폐암 예측 작업에서 5년간의 다중 모달 전자 건강 기록을 활용하여 9개의 강력한 기준선 모델을 초월하는 성과를 보였습니다. 특히, 결혼 흡연자를 포함한 더 까다로운 소그룹에서도 탁월한 성능을 보여, 이 시스템이 임상 의사결정 지원 도구로서의 가능성을 지니고 있음을 시사합니다. 추적 개선 메커니즘의 동역학 분석 결과, 경험 풀과 MARL의 상호 보완성이 확인되었습니다.



### ChatHealthAI: Aligning Electronic Health Record Representations with Large Language Models for Grounded Clinical Reasoning (https://arxiv.org/abs/2606.02802)
Comments:
          Main paper with appendix, 13 pages

- **What's New**: 이 논문에서는 ChatHealthAI라는 다중 모달 임상 추론 프레임워크를 제안합니다. 이 프레임워크는 구조화된 전자 건강 기록(EHR) 표현을 미리 훈련된 대형 언어 모델(LLM)과 정렬하여 보다 해석 가능하고 깊이 있는 임상 예측을 지원합니다. EHR 표현과 LLM의 시맨틱 공간을 연결하는 작업-aware resampler를 사용하여, 모델이 환자의 임상 이벤트 정보를 기반으로 자연어로 근거를 제공하는 임상 예측을 생성할 수 있게 합니다.

- **Technical Details**: ChatHealthAI는 CLMBR-T-Base를 EHR 기초 모델로 사용하고, Deepseek-R1-Distill-Qwen-14B를 LLM으로 사용합니다. 이 프레임워크는 환자의 구조화된 EHR 경로에서 학습된 표현을 고정된 LLM에 정렬하며, 이 과정에서 정제된 임상 이벤트들을 텍스트 근거로 통합하여 임상 추론을 합니다. 질병 진단, 약물 처방, 검사 결과 등 다양한 임상 이벤트 시퀀스에서 예측 패턴을 잡아낼 수 있습니다.

- **Performance Highlights**: ChatHealthAI는 EHRSHOT 벤치마크에서 세 가지 임상 예측 과제(병원 체류 기간(LOS), ICU 입원, 30일 재입원)에 대해 평가되었습니다. 실험 결과, ChatHealthAI는 추론 품질과 해석 가능성을 향상시키면서도 경쟁력 있는 예측 성능을 유지하는 것을 보여주었습니다. 이러한 결과는 EHR 기초 모델과 사전 훈련된 LLM의 통합이 해석 가능한 임상 예측을 위한 유망한 방향임을 강조합니다.



### BehaviorBench: Modeling Real-World User Decisions from Behavioral Traces (https://arxiv.org/abs/2606.02798)
- **What's New**: 이번 논문에서는 실제 행동 데이터에 기반하여 개인화된 결정 모델링을 평가하기 위한 새로운 벤치마크인 	extsc{BehaviorBench}를 소개합니다. 기존의 사용자 이해 기준이 모델에 의존한 시뮬레이션 사용자나 생성된 행동에 의존했던 것과 달리, 	extsc{BehaviorBench}는 공개적인 예측 시장 및 블록체인 기록에서 추출한 데이터를 바탕으로 wallet 단위의 결정 내역을 재구성합니다. 두 가지 주요 작업 레이어인 Belief prediction과 Trade prediction이 이 벤치마크에서 사용되며, 이는 사용자의 최종 입장과 거래 방향을 예측합니다.

- **Technical Details**: 	extsc{BehaviorBench}는 2,000개의 평가 지갑에 걸쳐 141,445개의 Belief 인스턴스와 1,485,972개의 Trade 인스턴스로 구성되어 있습니다. 벤치마크는 사용자의 최근 행동을 반영한 개인화된 결정 시스템을 평가하며, DirectGen, ProfileGen, RetrievalGen 등 다양한 생성 인터페이스를 통해 사용자 행동의 여러 형태를 보여줍니다. 이 시스템은 Belief prediction과 Trade prediction 두 수준의 추상화를 통해 사용자 결정을 예측합니다.

- **Performance Highlights**: 개인화는 Belief prediction에서 더욱 일관되게 성과를 보이며, 거래 예측보다 궁극적인 벨리프의 예측을 더 잘 수행합니다. 각 작업 레이어와 메트릭에 따라 모델의 순위가 달라지는 것이 관찰되었으며, 서로 다른 역사 인터페이스는 각각 다른 실패 모드를 드러냅니다. 이러한 결과는 개인화가 단일 기능이 아니라 다양한 전략에 따라 달라진다는 것을 보여줍니다.



### Evaluating Transformer and LSTM Frameworks for Prediction in Ungauged Basins (https://arxiv.org/abs/2606.02791)
Comments:
          5 pages

- **What's New**: 본 연구에서는 수자원 모델링을 위한 새로운 접근 방식을 제시합니다. 특히, 두 가지 모델인 LSTM(Long Short-Term Memory)과 인코더 전용 Transformer를 비교하여 정보가 제한된 upstream 환경에서 어떤 모델이 더 효과적인지를 평가하였습니다. 결과적으로 LSTM 모델이 Transformers에 비해 전반적으로 우수한 성능을 보였습니다.

- **Technical Details**: LSTM은 시퀀스 데이터를 순차적으로 처리하며, 각 타임스텝에서 정보 업데이트 및 기억을 관리하기 위해 게이팅 메커니즘을 사용합니다. 반면, Transformer 구조는 병렬 자기 주의 메커니즘을 활용하여 모든 타임스텝 간의 관계를 동시에 계산합니다. 이 과정에서 정적 catchment 속성과 동적 시계열 입력을 별도로 처리하여 통합할 수 있도록 설계되었습니다.

- **Performance Highlights**: 결과적으로, downstream 정보가 포함될 경우 모든 모델의 성능이 향상됨을 보여주었으며, median NNSE는 60% 이상 증가했습니다. 특히, LSTM은 upstream streamflow 추정에 있어 회귀 메모리를 통해 작업에 더 잘 적합하였음을 나타냈습니다. 이러한 결과는 수자원 예측을 위한 모델 선택 시 하천의 흐름 정보를 어떻게 활용할 수 있는지를 이해하는 데 기여합니다.



### AURA: Action-Gated Memory for Robot Policies at Constant VRAM (https://arxiv.org/abs/2606.02775)
- **What's New**: AURA-Mem (Action-Utility Recurrent Adaptive Memory)은 데이터 센터와 로봇의 메모리 요구를 구분하여 설계된 혁신적인 메모리 접근 방식이다. 이 기술은 로봇이 단기 요청을 일괄 처리하는 대신, 한 번의 긴 비-resetting episode 동안 작동하도록 최적화되었다. AURA-Mem은 메모리 쓰기를 최소화하면서도 로봇의 다음 행동을 효율적으로 결정할 수 있도록 돕는다.

- **Technical Details**: AURA-Mem은 고정된 크기의 순환 메모리와 학습된 게이트를 결합하여 현재 관찰이 다음 행동을 어떻게 변화시킬지를 판단하는 메모리를 구현한다. 이는 행동 오류 신호에 기반하여 직접적으로 학습되며, 기존의 재구성 기반 메모리와는 다르게 메모리 쓰기를 최소화하는 방식으로 동작한다. 이 메모리는 episode 길이에 관계없이 일정한 4,224 바이트 크기를 유지하며, KV-cache와 비교하여 5.19배에서 9.19배의 메모리 쓰기를 줄일 수 있다.

- **Performance Highlights**: AURA-Mem은 O(1) 정확도 최고 기준과 동등한 성능을 보여주면서도 평균적으로 4.98배에서 최대 9.19배 적은 메모리 쓰기를 기록한다. 이 성능은 예산에 맞춘 무작위 및 주기적 스케줄링 전략에 비해 우수하며, 하드웨어 제약을 고려할 때 메모리 쓰기가 DRAM/HBM 대역폭 소비에 직접적으로 연결된다. 실험 결과, AURA-Mem은 성공률에서도 유사한 성능을 보이며, 여러 시드에서 검증되었다.



### Visual Graph Scaffolds for Structural Reasoning in Large Language Models (https://arxiv.org/abs/2606.02673)
- **What's New**: 이 논문은 그래프가 대규모 언어 모델(LLMs)에서 단순한 정보 제공을 넘어 추론을 조직하는 데 도움을 줄 수 있는 잠재력을 탐구합니다. 기존 연구에서는 주로 그래프가 외부 지원으로 사용되었던 반면, 이번 연구에서는 그래프가 내부적인 추론 보조 도구로 사용될 수 있는지를 살펴봅니다. 연구진은 교사 모델이 생성한 추론 과정을 그래프 마인드 맵으로 변환해 학생 모델에 제공하는 방식을 채택했습니다.

- **Technical Details**: 제안된 방법론은 교사 모델이 주어진 여러 질문에 대해 그래프 구조를 활용하여 추론 과정을 시각적으로 제공하는 것입니다. 이 과정에서는 학생 모델이 추론을 위해 시각적 그래프와 선형 텍스트 간의 차이를 비교합니다. 실험은 두 가지 방식, 즉 직접적 안내와 추상적 안내로 나뉘며, 각각 그래프의 역할을 테스트합니다.

- **Performance Highlights**: 실험 결과, 시각적 그래프 안내는 추상적 설정에서도 효과적이며, 텍스트 기반 안내보다 성능이 우수한 것으로 나타났습니다. 학생 모델이 그래프 구조를 내부화하는 데 있어 시각적 형태가 더 유리하다는 결론이 도출되었습니다. 이 연구는 그래프가 LLM의 외부 지식 구조를 넘어서서 논리적 사고 조직을 위한 시각적 지지 구조로서의 역할을 해야 함을 시사합니다.



### Humanoid-GPT: Scaling Data and Structure for Zero-Shot Motion Tracking (https://arxiv.org/abs/2606.03985)
Comments:
          Accepted at CVPR 2026

- **What's New**: 이번 논문에서는 전신 제어를 위한 새로운 GPT 스타일의 Transformer 모델인 Humanoid-GPT를 소개합니다. 이 모델은 10억 스케일의 모션 코퍼스를 기반으로 훈련되어, 기존의 얕은 MLP 트래커와는 달리 고도의 동적인 행동 추적이 가능합니다. 특히 Humanoid-GPT는 2B 프레임의 데이터로 훈련되어, 이전에 보지 못한 모션에 대한 제로샷 일반화(zero-shot generalization)에서 획기적인 성과를 보여줍니다.

- **Technical Details**: Humanoid-GPT는 트랜스포머 구조를 채택하며, causal attention 방식을 사용하여 각 관절에 대한 PD 목표를 예측합니다. 기존의 비인과적(non-causal) 모델링 방법대신, 데이터와 모델 크기에 따라 자연스럽게 확장할 수 있는 구조로 설계되었습니다. 데이터 샘플링 과정에서 Harmonic Motion Embedding (HME) 기법을 사용하여 다양한 모션의 분포를 분석하고, 이를 통해 훈련 과정의 다양성과 균형을 고려하여 모션을 샘플링합니다.

- **Performance Highlights**: Humanoid-GPT는 전신 동작을 제어하는 데 있어 뛰어난 민첩성(agility)과 제로샷 일반화 성능을 보여줍니다. 기존 연구들과 비교했을 때, Humanoid-GPT는 2B 프레임의 모션 코퍼스를 효율적으로 훈련하고, 제로샷 동적 모션 추적에서 높은 성과를 달성했습니다. 이로써, 향후 전신 제어 문제에 대한 새로운 성능 경계를 제시하며, 실제 로봇 하드웨어에서의 적용 가능성도 높였습니다.



### Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories (https://arxiv.org/abs/2606.03979)
Comments:
          A version of this work has been publicly available from September 2025 on OpenReview

- **What's New**: 이 논문에서는 심층 대형 언어 모델(LLMs)의 기존 한계를 극복하기 위해 '수면(Sleep)' 패러다임을 도입하고, 이를 통해 모델이 지속적으로 학습하고, 짧은 기간의 기억을 안정적인 장기 지식으로 압축하고, '꿈꾸기(Dreaming)' 과정을 통해 자기 개선을 이루는 방법을 제안합니다. 특히, 수면은 메모리 통합 및 자기 개선을 위한 두 가지 단계로 이루어져 있습니다. 이 패러다임은 인간의 학습 과정을 영감을 받아 개발되었습니다.

- **Technical Details**: 수면의 첫 번째 단계인 메모리 통합(Memory Consolidation)은 지식 시딩(Knowledge Seeding)이라는 과정을 통해 이루어집니다. 이 과정에서는 작은 모델의 기억을 큰 네트워크에 증류하여 더 많은 용량을 제공하면서 지식을 보존합니다. 두 번째 단계인 꿈꾸기(Dreaming)에서는 모델이 강화 학습(Reinforcement Learning, RL)을 활용해 합성 데이터의 커리큘럼을 생성하여 새로운 지식을 연습하고 기존 능력을 다듬는 자기 개선 과정을 진행합니다.

- **Performance Highlights**: 연구 결과는 지속적인 학습, 지식 통합 및 소수의 예시를 통한 일반화(task)에서 수면 단계의 중요성을 뒷받침합니다. 이 새로운 접근법은 LLMs가 생애 주기 전반에 걸쳐 점진적이고 효율적으로 학습할 수 있는 가능성을 제시합니다. 따라서, 이 논문은 LLM의 지식이 점점 오래되고 고립되는 문제를 해결하기 위한 혁신적인 솔루션을 제안합니다.



### Formalizing the Binding Problem (https://arxiv.org/abs/2606.03976)
Comments:
          Accepted to ICML 2026

- **What's New**: 이번 연구에서는 세계에 대한 표현이 단순한 특징 정보(예: 무언가가 파란색이다, 무언가가 원형이다)뿐만 아니라 이러한 특징들이 동일한 객체에 속하는지에 대한 정보(즉, 바인딩 정보)를 포함하고 있다는 점을 강조합니다. 우리는 이러한 바인딩 문제(binding problem)를 정보 이론적 접근 방식으로 형식화하고, 모델 표현에서 바인딩 정보를 측정하기 위한 프로빙(probing) 방법을 도입하였습니다.

- **Technical Details**: ViT(Vision Transformer) 아키텍처의 다양한 구성 요소에서 바인딩을 측정하기 위한 실험을 수행하였습니다. 이미지 요약 토큰([CLS])이나 공간 토큰(spatial tokens)과 같은 요소들이 포함된 여러 기존 ViT 모델을 비교하는 과정에서, 기능 공유(feature sharing), 폐색(occlusion), 자연 특징(natural features)과 같은 다양한 바인딩 도전 과제를 다루었습니다.

- **Performance Highlights**: 연구 결과에 따르면, 바인딩은 강력한 시각 인식(visual recognition) 및 추론(reasoning)의 핵심 요소로 기능한다는 것을 보여주었습니다. 특히, 다양한 바인딩 문제에 대한 성능을 평가함으로써 ViT 기반 아키텍처가 바인딩 정보를 얼마나 효율적으로 학습하는지를 분석하였습니다.



### Quantifying Faithful Confidence Expression in Large Reasoning Models (https://arxiv.org/abs/2606.03969)
Comments:
          Code: this https URL

- **What's New**: 이 논문은 LLMs(대형 언어 모델)의 신뢰성과 불확실성 커뮤니케이션의 중요성을 강조하고 있습니다. 특히, 모델이 내재적 신뢰도와 표현된 신뢰도의 정합성인 신뢰할 수 있는 보정(faithful calibration) 문제에 대해 다룹니다. 이를 통해 LRM(대형 추론 모델)이 장기적인 추론에서 신뢰를 어떻게 표현하는지를 이해하는 데 도움을 줄 새로운 프레임워크를 제안하고 있습니다.

- **Technical Details**: 저자들은 LRM의 신뢰할 수 있는 보정을 체계적으로 정량화하기 위한 새로운 프레임워크를 소개합니다. 이 프레임워크는 내부 불확실성의 세 가지 소스(토큰 확률, 은닉 상태, 샘플 응답의 일관성)에 대한 언어적 확신을 분석합니다. 연구는 7개 모델과 5개 데이터셋을 활용하여 대규모 실증 연구를 실시하며, 다양한 신뢰도 추정 방법을 사용해 LRMs의 FC(신뢰할 수 있는 보정)를 평가합니다.

- **Performance Highlights**: 연구 결과, 현재 LRM은 본래의 불확실성을 언어로 정확하게 표현하는 데 한계가 있음을 발견했습니다. 비록 추론 훈련이 이루어지더라도 FC의 향상으로 이어지지 않는 것으로 나타났습니다. 또한, 서로 다른 신뢰도 추정 방법이 동일한 트레이스에 대해 상이한 평가를 제공하여, 특정 추정 방법에 의존하는 것에는 주의가 필요함을 강조하고 있습니다.



### QUBRIC: Co-Designing Queries and Rubrics for RL Beyond Verifiable Rewards (https://arxiv.org/abs/2606.03968)
- **What's New**: QUBRIC는 강화 학습(RL)에서 쿼리와 루브릭을 공동 설계하는 새로운 프레임워크를 제안합니다. 기존의 방법들은 고정된 쿼리 분포를 기반으로 루브릭의 최적화를 수행했지만, QUBRIC은 쿼리 구조와 루브릭 품질 간의 관계를 탐구합니다. 이 접근법은 개방형 쿼리가 불분명한 루브릭을 생성할 수 있음을 발견하고, 이를 해결하기 위해 시나리오 기반 질문과 평가 가능 루브릭을 창출하고 있습니다.

- **Technical Details**: QUBRIC는 세 단계의 파이프라인을 통해 동작합니다. 첫 번째 단계에서는 개방형 쿼리를 특정 답변 공간으로 좁혀진 시나리오 기반 쿼리로 재작성합니다. 두 번째 단계에서는 교사 모델의 응답과 정책 응답을 대조하여 쿼리 수준의 루브릭을 생성합니다. 마지막으로, 이 쿼리 수준의 루브릭과 글로벌 루브릭을 결합하여 GRPO(Generalized Reinforcement Policy Optimization)를 통해 정책을 최적화합니다.

- **Performance Highlights**: QUBRIC은 ArenaHard 벤치마크에서 SFT(Supervised Fine-Tuning) 기반선보다 평균 5.5점 향상을 달성했습니다. 또한, 단지 지침을 따르는 데이터로 훈련된 QUBRIC은 법률, 도덕 및 서사적 추론을 포함한 세 가지 보류된 벤치마크에서도 평균 6.3점 향상을 보여 주목할 만합니다. 이러한 결과는 QUBRIC의 쿼리와 루브릭의 공동 설계가 강화 학습 성능을 개선하는 데 기여한다고 시사합니다.



### AlignAtt4LLM: Fast AlignAtt for Decoder-Only LLMs at IWSLT 2026 Simultaneous Speech Translation Task (https://arxiv.org/abs/2606.03967)
Comments:
          Accepted to IWSLT 2026

- **What's New**: 본 논문은 영어에서 독일어, 이탈리아어, 중국어로의 동시 음성 번역 시스템인 AlignAtt4LLM을 소개합니다. 이 시스템은 Qwen3-ASR과 Gemma-4 E4B-it으로 구성된 동기식 캐스케이드 구조를 통해 비원활한 음성을 번역합니다. 특히 AlignAtt를 디코더 전용 LLM에 처음으로 적용하여 효율성을 높였습니다.

- **Technical Details**: AlignAtt4LLM는 ASR(Automatic Speech Recognition)과 MT(Machine Translation) 시스템을 통합하여 실시간으로 동작합니다. Qwen3-ASR이 실시간으로 소스 전사를 수행하고, Gemma-4가 그에 따라 번역 작업을 수행합니다. 시스템은 소스 범위를 명시적으로 제시하고, 오프라인으로 번역 특정적인 정렬 헤드를 선택하며, 재구성된 주의 신호가 현재 가능한 소스 영역 내에 있는지 확인합니다.

- **Performance Highlights**: IWSLT 2026 개발 세트에서 AlignAtt4LLM은 저지연(2초) 및 고지연(4초 이하) 조건에서 독일어 및 이탈리아어에 대한 기존 기준을 초과하는 성능을 기록했습니다. 중국어에 대한 결과는 혼합되어 있으나, 이 방법은 단순히 결정론적 프롬프트 레이아웃과 조정된 주의 헤드, 쿼리/키 캡처만 필요로 하여 비유럽 대상 언어에도 적용될 수 있습니다.



### Agentic Chain-of-Thought Steering for Efficient and Controllable LLM Reasoning (https://arxiv.org/abs/2606.03965)
- **What's New**: 이번 논문에서는 대형 언어 모델의 추론 성능을 개선하기 위한 새로운 방법인 Agentic Chain-of-Thought Steering (ACTS)를 제안합니다. 기존의 추론 방법들이 주어진 토큰을 비효율적으로 사용하는 문제점에 주목하여, ACTS는 Markov 결정 과정으로 추론을 제어하는 새로운 방법론을 제공합니다. 이를 통해, 추론 도중 모델의 사고 과정을 명확하게 제어할 수 있습니다.

- **Technical Details**: ACTS는 각 단계에서 컨트롤러 에이전트가 추론 흔적과 남은 사고 예산을 관찰하고, 그에 따라 추론 전략과 다음 단계의 시작을 지시하는 구문을 포함한 조정 작업을 수행합니다. 이러한 방법은 예산을 고려한 전략 제어를 통해 효율적인 추론을 가능하게 하고, 동시에 생성의 연속성을 유지합니다. 논문에서는 강화 학습(reinforcement learning) 방법을 사용하여 최적화를 수행합니다.

- **Performance Highlights**: 다양한 벤치마크에서의 실험을 통해 ACTS는 전통적인 전사(thinking) 성능을 유지하면서도 상당한 토큰 절약을 보여주었습니다. 이 방법은 서로 다른 추론기와 작업 간에 정확성과 효율성의 조정을 가능하게 하여 사용자 맞춤형 성능을 제공합니다.



### Self-Refining Agentic Reinforcement Learning for Vision-Conditioned UAV Navigation (https://arxiv.org/abs/2606.03963)
- **What's New**: 이 논문에서는 자율 비행체(우주비행기) 내비게이션을 위한 에이전트 주도 강화 학습 프레임워크인 AgenticRL을 소개합니다. AgenticRL은 멀티모달 생성 미리 훈련된 트랜스포머(GPT) 에이전트를 사용하여 작업 정보를 해석하고, 작업 특정 보상 함수를 생성하며, Proximal Policy Optimization(PPO) 알고리즘을 통해 정책을 훈련합니다. 이 시스템은 또한 자가 개선 프로세스를 통해 보상 함수를 정제합니다.

- **Technical Details**: AgenticRL은 언어 지침과 시각적 장면 맥락을 활용하여 보상 생성을 자동화하는 멀티모달 프레임워크로서, 강화 학습 정책의 교육, 평가, 정제를 위한 폐쇄 루프 프로세스를 특징으로 합니다. 이 시스템은 UAV 내비게이션 작업을 지원하며, 언어와 이미지 인식을 통한 시나리오 자동 인식을 위한 메커니즘도 포함하고 있습니다. 다양한 탐색 작업에서 실험적으로 평가된 결과 폐쇄 루프 정제 과정이 초기 보상에 비해 71% 정책 행동 향상을 보였습니다.

- **Performance Highlights**: AgenticRL 프레임워크는 여러 탐색 작업에서의 성능을 평가하였으며, 실제 세계에서 91%의 성공률을 달성하고 시뮬레이션에서 실제로의 정확도는 94%에 이릅니다. 이 연구는 UAV의 자율 안전성을 높이고 효율적인 작업 수행을 위한 새로운 접근 방식을 제공하며, 기존의 인공지능 시스템들과 비교하여 보다 뛰어난 성능 향상을 보여주었습니다.



### Using Reward Uncertainty to Induce Diverse Behaviour in Reinforcement Learning (https://arxiv.org/abs/2606.03962)
Comments:
          Core contributors: Anthony GX-Chen, Ankit Anand, Gheorghe Comanici, André Barreto, Mark Rowland

- **What's New**: 본 논문에서는 기존 강화 학습(RL) 목표를 새로운 관점에서 재구성하여 보상 함수의 배포(distribution)로 대체한다고 제안합니다. 이러한 접근법은 다양한 행동을 자연스럽고 효과적으로 학습할 수 있는 기반을 제공합니다. 연구진은 이러한 이론을 통해 복잡한 RL 문제에서의 성과 향상을 실증적으로 입증하였습니다.

- **Technical Details**: 논문에서 제안하는 방법은 Randomized Objectives, Set Actions (ROSA)로 불리며, 이것은 보상 함수의 배포를 통해 조정된 다양성을 제어하는 새로운 RL 목표군을 도입합니다. 또한, 다양한 상태에서의 행동 집합에 비선형적으로 의존하는 정책 경량화에 대한 기법을 제공하여 일반적인 정책 경량화 방법론을 자연스럽게 일반화합니다.

- **Performance Highlights**: 실험 결과, 본 프레임워크는 복잡한 RL 작업에서 원하는 다양성을 실현할 수 있는 강력하고 이론적으로 확고한 대안을 제공합니다. 기존 강화 학습 문제에서 해결되지 않았던 에이전트 행동의 폭을 넓히는 데 성공하여, 이 새로운 접근법이 실제 상황에서의 광범위한 응용 가능성을 지니고 있음을 입증하였습니다.



### Efficient ASR Training with Conversations that Never Happened (https://arxiv.org/abs/2606.03957)
- **What's New**: 본 논문에서는 저자원이거나 특정 도메인에 대한 대화형 자동 음성 인식(ASR)의 한계를 극복하기 위한 새로운 데이터 증강 파이프라인을 제안합니다. 이 방법은 시나리오 기반 대화와 참여자 메타데이터를 생성하고, 화자 속성을 TTS 목소리 프로필에 매핑하며, 합성된 발화를 기반으로 화자 인식 시뮬레이션 대화를 구축합니다. 이를 통해 기존의 대화 체계를 넘어 고객 요구에 맞춘 대화 생성이 가능해집니다.

- **Technical Details**: 제안된 파이프라인은 LLM을 사용하여 시나리오와 메타데이터, 구조화된 대화를 생성합니다. 이후 화자 속성(예: 나이와 성별)을 TTS 음성 프로필에 매핑하고 각 발화(turn)를 합성합니다. 또한, 다중 화자 대화 파형을 생성하는 과정에서 일시 정지 및 중첩 패턴을 포함하여 자연스러운 대화를 재현합니다.

- **Performance Highlights**: 실험 결과, 합성된 대화가 ASR 성능을 일관되게 향상시키는 것으로 나타났습니다. 주목할 만한 점은 제너레이터의 선택과 데이터 구성이 성능 향상에 큰 영향을 미친다는 것입니다. 또한, 67시간의 실제 대화와 636시간의 합성 데이터로 훈련된 모델이 2700시간의 헝가리어 음성으로 훈련된 제로샷 모델보다 더 나은 성능을 보였습니다.



### FlashbackCL: Mitigating Temporal Forgetting in Federated Learning (https://arxiv.org/abs/2606.03939)
- **What's New**: 이번 논문에서는 Federated Learning (FL)에서의 시간적 망각(temporal forgetting)을 Formalize하고 Flashback에서 발전시킨 Flashback Continual Learning (FlashbackCL)을 제안합니다. FlashbackCL은 (i) 상황 변화에 따라 감소하는 레이블 카운트, (ii) 클래스 균형을 고려한 Replay Buffer(Class-Balanced Reservoir Sampling), (iii) 서버 측에서의 활성 코어셋 큐레이션을 도입하여 설계되었습니다. 이 방법은 CIFAR-10 및 CIFAR-100 데이터셋에서 신뢰할 수 있는 성능 향상을 보였습니다.

- **Technical Details**: 이 연구는 FL에서 각 클라이언트의 데이터 분포가 동적으로 변화하는 상황을 바라보며, 시간적 요소가 포함된 지표를 제시합니다. FlashbackCL에서는 이전의 레이블 카운트를 모니터링하며, 변화가 없는 클래스의 과중합을 회피할 수 있는 새로운 메커니즘을 구축했습니다. 따라서 전체 모델이 오래된 클래스 밸런스에 매여 있지 않도록 하고, 적응성을 높일 수 있습니다.

- **Performance Highlights**: 실험 결과 FlashbackCL은 CIFAR-10에서 Flashback 대비 6.9%에서 10.0%의 상대적 개선을 기록하였고, 시간적 망각을 최대 68%까지 감소시켰습니다. CIFAR-100에서도 FlashbackCL은 모든 모드에서 Flashback보다 9.7%에서 12.7%의 개선 효과를 보였습니다. 이는 FlashbackCL이 시간적 변화뿐만 아니라 공간적 이질성(spatial heterogeneity)에도 효과적으로 대응할 수 있음을 나타냅니다.



### q0: Primitives for Hyper-Epoch Pretraining (https://arxiv.org/abs/2606.03938)
- **What's New**: 본 연구에서는 다중 에포크 훈련을 넘어 다양한 모델 집단을 탐색하고 예측을 집계하는 개념적 전환을 제안합니다. 이를 통해 하이퍼-에포크 사전 훈련(hyper-epoch pretraining)이라는 새로운 방법론을 도입하여, 단일 정제 모델보다 더 낮은 검증 손실(validation loss)을 기록할 수 있도록 합니다. 이 방법은 고유한 특성을 가진 세 가지 핵심 요소를 중심으로 구성되어 있습니다.

- **Technical Details**: 하이퍼-에포크 사전 훈련 방법은 사이클 방식(cyclic schedule), 체인 증류(chain distillation), 그리고 학습된 사전(prior)을 바탕으로 합니다. 사이클 방식은 서로 다른 경로에서 다양성을 확보하기 위해 학습률(learning rate)과 가중치 감쇠(weight decay)를 반상관하도록 조정합니다. 체인 증류는 후속 모델이 이전 모델을 교사로 삼아 훈련하여 모델의 품질이 집단에서 누적될 수 있도록 돕습니다.

- **Performance Highlights**: 1.8B 매개변수(model parameter)의 모델을 통해 100M FineWeb 토큰으로 훈련한 결과, 하이퍼-에포크 방식은 256 에포크 기준으로 자료 효율성(data efficiency)에서 약 12.9배 개선된 성과를 기록했습니다. 또한, 이 방법은 다양한 에포크 예산에 걸쳐서도 일관된 성능 향상을 보여줍니다.



### FFR: Forward-Forward Learning for Regression (https://arxiv.org/abs/2606.03927)
- **What's New**: 이 논문에서는 Forward-Forward (FF) 알고리즘을 실세계 회귀 문제로 확장한 FFR(Forward-Forward for Regression) 프레임워크를 제안합니다. FFR은 기존의 FF에서 세 가지 주요 혁신을 도입하여 회귀 분석을 위한 적합한 성능을 제공합니다. 이 알고리즘은 회귀 문제에서 교차 샘플 쌍을 활용하지 않고, 대신 경쟁 학습을 통해 효율적인 레이어 간 업데이트를 가능하게 합니다.

- **Technical Details**: FFR은 (1) 거리 인식 순위 감독 하에서 서로 다른 뉴런 그룹 간의 경쟁적 학습을 통해 교차 샘플을 대체하는 순위 경쟁 적합 함수, (2) 얕은 레이어가 거친 순위 구분을 학습하고 깊은 레이어가 섬세한 회귀로 발전하는 계층적 사다리 구조, (3) 불확실성 추정을 포함해 예측 신뢰도를 제공하는 계층적 예측 기능을 포함하고 있습니다. 이러한 구조는 메모리와 시간 효율성을 극대화하며, 심층 학습의 기존 한계를 극복하는데 중점을 두고 있습니다.

- **Performance Highlights**: FFR은 다섯 개의 실세계 회귀 벤치마크에서 평균적으로 BP(backpropagation)의 98.6%의 성능을 달성하면서, 훈련 메모리 피크를 약 27%와 8%로 줄였습니다. 평균 반복 시간 또한 BP의 약 72%에 불과하며, 기존의 모든 BP 비사용 경쟁자들을 상당히 능가하는 성과를 보였습니다. 이러한 결과들은 FFR이 실제 적용 가능한 강력한 알고리즘임을 입증합니다.



### NetKV: Network-Aware Decode Instance Selection for Disaggregated LLM Inferenc (https://arxiv.org/abs/2606.03910)
- **What's New**: 본 논문에서는 LLM (Large Language Model) 추론의 비집중화(disaggregation) 아키텍처에서 발생하는 새로운 병목 현상을 다룹니다. 특히, KV 캐시를 디코드 인스턴스로 전송하는 과정에서 발생하는 전송(latency) 시간을 고려하여 성능을 최적화합니다. 저자들은 네트워크 비용 오라클(network cost oracle)을 통해 전송 비용을 포함한 최적의 스케줄링을 제안하며, 이는 기존 스케줄러들이 간과한 부분입니다.

- **Technical Details**: LLM 추론은 주로 두 단계로 나뉘며, 첫 번째는 높은 계산 부담을 수반하는 prefill 단계입니다. 두 번째는 메모리 대역폭에 의존하는 decode 단계로, 이 두 단계가 동일 GPU에서 실행될 경우 상호 간섭이 발생합니다. 저자들은 NetKV라는 새로운 알고리즘을 통해 요청 시마다 최적의 디코드 인스턴스를 선택하고, 이때 전송 비용을 고려하여 스케줄링 성능을 개선합니다.

- **Performance Highlights**: NetKV는 64-GPU 4단계 fat-tree 시뮬레이터에서 실험하여, 평균 TTFT(Time to First Token)를 라운드 로빈 대비 최대 21.2%, 조정된 캐시와 로드 인지 스케줄러 대비 17.6% 감소시켰습니다. 또한, SLO (Service Level Objective) 달성을 20.1% 증가시키고, 테스트된 모든 조건에서 TBT(Time Between Tokens) 오버헤드를 0.5ms 이하로 유지하는 성능을 보여주었습니다.



### The Impact of Configuring Agentic AI Coding Tools on Build-vs-Buy Decisions: A Study Protoco (https://arxiv.org/abs/2606.03907)
Comments:
          14 pages, 1 table. Accepted at the 20th International Symposium on Empirical Software Engineering and Measurement (ESEM 2026), Registered Reports track

- **What's New**: 이번 논문은 AGENTIC AI 코딩 도구에서 외부 라이브러리를 사용하는 것과 직업적으로 구축하는 것 간의 결정인 빌드-버즈-바이를 제어하기 위한 구성 메커니즘의 영향을 조사합니다. 이를 위해 Claude Code와 OpenAI Codex라는 도구를 대상으로 사전 등록된 프로토콜을 제시하며, 이상적인 환경에서 각 도구의 빌드-버즈-바이 행동을 변화시키는 요소들을 측정합니다. 이는 AI 코딩 도구 개선에 기여하고, 보안 및 라이센스 준수에 대한 전략을 제시할 수 있는 새로운 데이터 세트를 생성할 것입니다.

- **Technical Details**: 이 연구는 다섯 개의 단계별 프로젝트에서 기인한 통제된 프로그래밍 작업을 실행하여 각 도구가 어떻게 외부 라이브러리를 도입하거나 내부적으로 기능을 구현하는지 측정합니다. 연구 범위는 no configuration, soft preferences와 explicit prohibitions를 포함하여 다양한 구성 메커니즘이 포함되어 있습니다. 연구 프로토콜은 9개의 가설에 따라 체계적으로 설계되었습니다. 이 결과는 향후 AI 코딩 도구의 향상된 기능 개발에 기여할 수 있도록 설계된 배치 데이터 세트와 분석 파이프라인으로 지속적으로 제공될 것입니다.

- **Performance Highlights**: 기존의 두 연구는 AI 코딩 도구들이 특정 라이브러리를 시스템적으로 선호하고 있으며, 이러한 패턴이 도구의 모델 버전에 따라 변화한다는 것을 입증하였습니다. 그러나 구성 메커니즘이 이러한 패턴을 수정하는지를 요약한 연구는 없었습니다. 본 프로토콜은 AI 코딩 도구에서의 빌드-버즈-바이 결정을 형성하는 구성 메커니즘의 영향을 평가하는 최초의 통제 실험적 증거를 제공하여 보안과 컴플라이언스 워크플로에 직접적인 영향력을 행사할 것입니다.



### Agent libOS: A Library-OS-Inspired Runtime for Long-Running, Capability-Controlled LLM Agents (https://arxiv.org/abs/2606.03895)
Comments:
          14 pages, 1 figure, 2 tables

- **What's New**: 이 논문은 LLM(Large Language Model) 에이전트가 단순한 요청-응답 보조 기능을 넘어, 상태를 유지하고 외부 이벤트를 기다리며 작동하는 장기 소프트웨어 액터로서의 진화를 설명합니다. Agent libOS는 LLM 에이전트를 위한 라이브러리 OS에 영감을 받은 런타임 기반으로, 일반적인 호스트 운영 체제 위에서 작동합니다. 이 시스템은 에이전트의 동작을 관리하기 위한 새로운 개념과 설계 원칙을 제공합니다.

- **Technical Details**: Agent libOS는 에이전트 프로세스(AgentProcess)로서 에이전트를 다루며, 프로세스 아이덴티티, 생애 주기 상태, 도구 테이블 및 명시적 권한을 포함한 다양한 기능을 제공합니다. 이 시스템은 동기식 스케줄링, 네임스페이스-로컬 객체 메모리 및 인간 승인을 통합한 프로토타입을 구현하였으며, 다양한 테스트와 평가를 통해 정확성과 안전성을 검증합니다. 이 시스템은 모델-지향 행동이 권한이 부여된 신뢰할 수 있는 런타임 작업이 되는 방식을 정의합니다.

- **Performance Highlights**: 현재 프로토타입은 비동기 스케줄링과 체크포인트를 기반으로 하는 시스템으로, 에이전트가 권한 부여 및 감사 과정을 거쳐 장기적으로 수행될 수 있도록 설계되었습니다. 에이전트 libOS는 기존의 LLM 시스템과 차별화되는 지점에서, 도구 배포를 신뢰 경계로 취급하는 대신 수행 가능한 기반으로서의 역할을 강조합니다. 이 논문은 새로운 설계 원칙과 구조를 통해 LLM 에이전트의 장기적인 신뢰성을 증진시키고자 합니다.



### Synthesize and Reward -- Reinforcement Learning for Multi-Step Tool Use in Live Environments (https://arxiv.org/abs/2606.03892)
- **What's New**: PROVE (Programmatic Rewards On Verified Environments)는 LLMs (Large Language Models)가 다단계 도구 호출을 조율하도록 훈련하는데 있어 세 가지 주요 도전을 해결하는 프레임워크입니다. 첫째, 20개의 stateful MCP 서버를 통해 343개의 도구를 제공하여 실제 실행 기반 RL (Reinforcement Learning) 훈련이 가능하게 합니다. 둘째, 자동화된 데이터 합성 파이프라인을 통해 검증된 다중 턴 도구 호출 경로를 생성하여 실제 상태와 연관된 질의를 생산합니다. 셋째, 외부 판단 모델 없이 툴 사용 품질을 평가하기 위한 다부 구성 보상 시스템을 개발했습니다.

- **Technical Details**: PROVE에서는 데이터 합성과 RL 루프를 밀접하게 결합하여 실제 환경에서 동작하는 LLM을 훈련할 수 있도록 합니다. 첫 번째 구성 요소로서, Model Context Protocol (MCP)을 사용하는 20개의 live 서버 환경을 설정하여 실제 상태 의존적 실행 다이나믹스를 캡처합니다. 둘째, grounding된 상태 머신 데이터 합성 파이프라인이 각 서버에서 실체를 샘플링하고, 다중 턴 대화를 통해 질의를 생성하며, 실행 결과를 재검증하여 약 1만 3천 개의 훈련 예제를 자동 생성합니다. 세 번째 구성 요소로는 툴 사용 품질을 평가하는 다부 상 구성 보상 체계를 설정하여 각 툴 호출의 유효성 및 의존성을 평가합니다.

- **Performance Highlights**: PROVE를 통해 모델들은 BFCL Multi-Turn, tau2-bench 및 T-Eval에서 각각 +10.2, +6.8, +6.5 점의 향상을 보였습니다. 이는 다섯 개의 구성 요소로 이루어진 보상 시스템이 모델의 다단계 도구 조율 능력을 일관되게 향상시킨 결과입니다. 훈련 과정에서는 약 1만 3천 개의 예제를 사용하여 기존 RL 파이프라인보다 8배 적은 예제 수로도 효과적인 학습 효과를 입증했습니다.



### Beyond Encoder Accumulation: Measuring Encoder Roles in Multi-Encoder VLMs (https://arxiv.org/abs/2606.03879)
- **What's New**: 최근 연구에서는 다양한 비전 인코더들을 통합하여 훈련하는 것이 LVLMs(대형 비전-언어 모델)의 성능 향상에 중대한 영향을 미친다고 밝혔습니다. 연구자들은 31개의 서로 다른 인코더 조합을 재훈련하여 인코더의 상호작용을 면밀히 분석하였습니다. 이를 통해 기존 연구에서 관찰되었던 인코더의 기여도에 대한 잘못된 랭킹이 드러났습니다.

- **Technical Details**: 이 연구에서는 인코더의 기여를 두 가지 축인 수용력(Capacity)과 필요성(Necessity)으로 분解하였습니다. 수용력은 각 인코더가 혼자서 도달할 수 있는 점수이고, 필요성은 전체 풀에서 제거했을 때 점수가 얼마나 떨어지는지를 나타냅니다. 연구 결과에 따르면, 높은 수용력의 인코더를 짝지어 사용하는 것보다, 적절한 보완 인코더와 짝 지어 사용하는 것이 더 유리하다는 사실이 드러났습니다.

- **Performance Highlights**: 연구에서는 효율적인 인코딩 별 효과적인 순위가 고정된 파라미터 수에서 잔여 점수 변동성을 설명한다고 밝혔습니다. 강력한 인코더 조합은 공동 훈련 중에서도 생존하는 앵커와 그에 따라 확장되는 보완 성질을 결합하여 더 우수한 최적화 결과를 가져오는 것으로 나타났습니다. 전반적으로, 이 연구는 다중 인코더 LVLM 설계의 방법론적 공백을 드러내며, 이를 해소하기 위한 구체적인 방법론을 제공합니다.



### From 'What' to 'How' and 'Why': Sharing LLM-Generated Retrospective Summaries of Older Adults' Passive Tracking Data with Remote Family Members (https://arxiv.org/abs/2606.03876)
- **What's New**: 이 논문은 노인들을 돌보는 원거리 가족 구성원(RFMs)을 위해 다중 모달 추적 데이터를 활용하여 회고적 요약을 생성하는 방법을 탐구합니다. 기존의 시스템인 'Vital Insight'를 커스터마이징하여 다양한 데이터 가용성 시나리오에서 초기 요약을 생성하고, 11명의 RFMs와 인터뷰를 통해 피드백을 받았습니다. 이러한 통찰력을 바탕으로 시스템을 다층적이고 인사이트 중심의 요약 접근 방식으로 재설계하였습니다.

- **Technical Details**: 기존의 다중 모달 패시브 센싱 시스템은 노인들이 자택에서 안전하고 독립적으로 생활할 수 있도록 지원할 가능성이 있지만, 이질적인 데이터 스트림을 고수준의 의미 있는 콘텐츠로 통합하는 데 어려움이 있습니다. 본 연구는 LLMs(대형 언어 모델)를 활용하여 RFMs에게 필요한 맥락적이고 개인화된 해석을 제공하는 요약 시스템을 개발하였습니다. 초기 요약은 시간 기반 레이어에서 사용자 중심의 인사이트 기반 레이어로 재설계되었습니다.

- **Performance Highlights**: 모든 RFMs를 대상으로 한 비교 연구 결과, 새로운 요약은 초기 버전에 비해 만족도, 유용성, 신뢰도, 수용 의향이 유의미하게 향상되었습니다(p<.05). 연구는 AI 생성 요약이 RFMs와 노인 간의 데이터 공유를 촉진하는 데 어떻게 활용될 수 있는지를 조명합니다. 또한, 이러한 인사이트가 다른 환자 집단에서도 적용될 수 있는 가능성을 제시합니다.



### A Training-Free Mixture-of-Agents Framework for Multi-Document Summarization using LLMs and Knowledge Graphs (https://arxiv.org/abs/2606.03867)
Comments:
          Accepted by Neural Computing and Applications

- **What's New**: 이번 논문에서는 다문서 요약(MDS) 문제를 해결하기 위해 새로운 Mixture of Agents (MoA) 프레임워크를 제안합니다. MoA는 대규모 언어 모델(LLMs)과 지식 그래프(KG)의 강점을 활용하여 훈련 없이 작동하는 모듈형 시스템으로, 복잡한 문서 간 관계를 학습하는 데 효과적입니다. 이 접근법은 특별한 세부 조정 없이도 세 가지 전문 에이전트(Extractor, KGSum, Abstractor) 간 조정을 통해 요약을 생성합니다.

- **Technical Details**: MoA 프레임워크는 문서 요약을 전문화된 에이전트 작업으로 분해합니다. Extractor 에이전트는 중요한 문장을 추출하고, KGSum 에이전트는 지식 그래프를 활용하여 주제와 대조 정보를 모델링합니다. Abstractor 에이전트는 직접적으로 텍스트 문서에서 유창하고 일관된 요약을 생성하며, 이들 출력은 Adaptive Multi-Perspective Fusion (AMF) 메커니즘을 통해 통합됩니다.

- **Performance Highlights**: 이 연구에서는 영어와 베트남어 데이터셋에서 MoA를 평가하여 최첨단 성능을 달성함으로써 이 아키텍처의 효과를 증명했습니다. MoA는 다문서 요약에서 문서 간 복잡한 관계를 학습할 수 있는 강력한 전이 학습을 제공하며, 기계 학습 모델들이 요구하는 대량의 라벨링 데이터 의존성을 제거하는 것을 목표로 합니다.



### Taiji: Pareto Optimal Policy Optimization with Semantics-IDs Trade-off for Industrial LLM-Enhanced Recommendation (https://arxiv.org/abs/2606.03866)
Comments:
          8 pages, 2 figures

- **What's New**: 이번 논문에서는 산업 추천 시스템을 위해 설계된 혁신적인 LLM-as-Enhancer 프레임워크인 Taiji를 소개합니다. Taiji는 기존의 LLM4Rec 패러다임이 직면하고 있는 SFT(세미 슈퍼비전 파인 튜닝)와 RL(강화 학습) 단계의 문제를 해결하는 방법을 제시합니다. 이를 통해 Taiji는 고유의 Domain-Specific Chain-of-Thought 데이터 생성을 위한 역설계된 추론과 개방형 거부 샘플링을 활용하여 추천 품질을 높입니다.

- **Technical Details**: Taiji는 데이터 구축, 추론 활성화, LLM-추천 협업 및 온라인 순위 지정을 포함한 네 개의 주요 모듈로 구성되어 있습니다. EUPR(Reverse-Engineered User Preference Reasoning)과 ORFT(Open-Ended Rejection Sampling Fine-Tuning)를 통합하여 추천 특화 CoT의 품질을 향상시키고, POPO(Pareto Optimal Policy Optimization)라는 방법을 통해 LLM의 의미적 보상과 추천 선호 보상의 균형을 동적으로 조정합니다.

- **Performance Highlights**: 다양한 오프라인 평가와 온라인 A/B 테스트를 통해 Taiji의 효과성을 검증하였습니다. Taiji는 Kuaishou의 광고 플랫폼에 2026년 5월부터 배포되어 매일 4억 명 이상의 사용자에게 서비스를 제공하며, 상업적 수익을 획기적으로 증가시켰습니다. A/B 테스트 결과, 광고주 가치(ADVV)가 2.83% 개선되고 전반적인 수익이 3.30% 증가하는 성과를 보였습니다.



### FLARE: Fine-Grained Diagnostic Feedback for LLM Code Refinemen (https://arxiv.org/abs/2606.03852)
- **What's New**: Flare는 기존의 피드백 신호의 한계를 극복하기 위해 개발된 경량 진단 모델을 사용하여 코드 개선을 위한 라인 수준의 의심 신호를 예측하는 반복 틀(framework)입니다. 기존 방법들은 일반적으로 테스트 실패와 같은 신호에 의존하여 코드를 세련되게 하지만, 이러한 신호는 충분한 정보가 되지 못합니다. Flare는 초기 코드 솔루션에서 언어 모델의 토큰 수준의 확률을 수집하고 이를 코드의 구문적 단위와 정렬하여 보다 정밀한 진단을 가능하게 줍니다.

- **Technical Details**: Flare의 진단 모델은 LLM의 내부 신호를 기반으로 의심스러운 코드를 평가합니다. 이 시스템은 높은 불확실성을 고려하여 매 반복마다 상위 k개의 의심 지역을 검색하고 실행 결과에 따라 최적의 후보를 선택합니다. Flare는 반복적으로 실행되며 프로그램이 모든 테스트를 통과하거나 최대 반복 한도에 도달할 때까지 계속됩니다.

- **Performance Highlights**: 실험 결과, LiveCodeBench와 BigCodeBench에서 Flare는 상위 10개의 후보를 검색할 경우 평균 8.50%의 성능 향상을 보여줍니다. 또한 Flare의 경량 진단 모델은 기존의 결함 위치 지정 방법에 비해 뛰어난 성능을 입증하여 코드 세련화에 신뢰할 수 있는 세밀한 지침을 제공할 수 있음을 나타냅니다.



### Clustered Self-Assessment: A Simple yet Effective Method for Uncertainty Quantification in Large Language Models (https://arxiv.org/abs/2606.03846)
Comments:
          Findings of ACL 2026

- **What's New**: 본 연구는 대형 언어 모델(LLMs)의 불확실성을 정량화하기 위한 새로운 방법을 제안합니다. 기존의 방법은 간접적인 신호를 활용하여 불확실성을 측정했지만, 이 방법은 모델 내부의 능력을 충분히 활용하지 못하고 해석하기 어려웠습니다. 제안된 방법은 샘플링된 생성물을 의미론적으로 구별되는 클러스터로 그룹화하고, 이를 기반으로 다중 선택 질문을 구성하여 모델이 스스로 불확실성을 평가하도록 합니다.

- **Technical Details**: 제안된 방법은 두 단계로 이루어져 있습니다. 첫 단계에서 LLM에서 샘플링한 답변을 클러스터링하여 의미론적으로 구별되는 클러스터를 만듭니다. 이후 클러스터로부터 다중 선택 질문(MCQ)을 구성하고 각 선택지에 대해 LLM이 할당한 확률을 신뢰성 점수로 사용합니다. 이는 사용자에게 직관적으로 신뢰성을 평가할 수 있는 방법을 제공합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다양한 모델과 데이터셋에서 기존 알고리즘을 일관되게 능가하는 것으로 나타났습니다. 특히, 단 2개의 추가 샘플로도 경쟁력 있는 성능을 달성하여 효율성을 보여줍니다. 이 방법은 생성된 답변의 신뢰성을 쉽게 평가할 수 있도록 하여 실제 적용 가능성도 입증되었습니다.



### Re-Evaluating Continual Learning with Few-Shot Adaptation (https://arxiv.org/abs/2606.03843)
Comments:
          21 pages, 16 figures

- **What's New**: 본 논문에서는 지속적인 학습(continal learning) 시스템의 안정성(stability)과 융통성(plasticity)을 평가하기 위해 새로운 평가 방법인 few-shot evaluation을 제시합니다. 기존의 0-shot 성능 측정 방식이 모델의 정보 유지 및 새로운 정보 적응 능력을 충분히 측정하지 못한다는 점을 지적합니다. 이들은 최근 학습된 작업에서의 성능과 이전 작업에서의 성능을 동시에 고려하는 새로운 메트릭인 per-shot plasticity를 통해 더 나은 통찰력을 제공합니다.

- **Technical Details**: 지속적인 학습에서의 안정성과 융통성은 일반적으로 이전 작업에 대한 성능 유지와 현재 작업에 대한 성능 극대화 사이의 균형으로 정의됩니다. 연구는 태스크 시퀀스에서의 세부적인 평가를 통해, 다양한 지속적인 학습 전략의 성능을 분석합니다. 특히, meta-learning을 통해 미래 태스크의 짧은 시퀀스를 학습하는 방식이 학습-학습 행동을 유도한다는 것을 보여줍니다.

- **Performance Highlights**: Few-shot learning 능력을 가진 현대 기계 학습 모델이 적은 정보로도 작업을 수행할 수 있다는 점을 강조합니다.  결과적으로, few-shot 적응성을 평가하는 novel metric이 지속적인 학습의 동역학을 깊이 평가할 수 있게 해줍니다. 실험을 통해, 이 평가 방법이 지속적인 학습의 다양한 접근 방식에 대해 놀라운 결과를 도출함을 발견하였습니다.



### Conditional Latent Diffusion Model with Fourier-based Motion Modelling for Virtual Population Synthesis (https://arxiv.org/abs/2606.03827)
Comments:
          This work has been early accepted by International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI) 2026

- **What's New**: 이번 연구에서는 4D F-MeshLDM이라는 조건부 생성 프레임워크를 제안합니다. 이 모델은 심장 메쉬 생성을 보다 정확하고 일관되게 수행할 수 있도록 설계되었습니다. 특히, 심장 주기에 대한 수학적으로 정확한 주기 일관성을 보장합니다. 또, 특이한 점은 Fourier 계수를 사용하여 심장 동작을 효과적으로 매개화한다는 점입니다.

- **Technical Details**: 제안된 4D F-MeshLDM은 변형된 Variational Autoencoder(VAE)를 통해 각 메쉬를 인코딩하고, 자가 인식된 주기적 운동 경로를 구현하기 위해 절단된 Fourier 시리즈를 사용합니다. 이 모델은 임상 변수에 기반한 조건을 통해 동작 패턴을 조절할 수 있으며, 이로 인해 고도의 주기성을 유지하게 됩니다. 또한, Denoising Diffusion Probabilistic Model(DDPM)을 활용하여 런 생성과정을 구현합니다.

- **Performance Highlights**: 실험 결과, 4D F-MeshLDM은 기존의 최첨단 모델들보다 해부학적 정확도에서 월등한 성능을 보였습니다. 심장 메쉬가 생산하는 사이클 닫힘 오류는 거의 제로 상태에 도달하였으며, 임상적 기능 지수도 정확하게 보존되었습니다. 이는 이 프레임워크가 신뢰할 수 있는 인 실리코(in-silico) 심장 시험을 수행할 유망한 가능성을 보여줍니다.



### AI Agents Enable Adaptive Computer Worms (https://arxiv.org/abs/2606.03811)
- **What's New**: 이 논문은 AI에 의해 구동되는 새로운 형태의 사이버 보안 위협인 컴퓨터 웜(Computer Worm)에 대한 개념 검증(proof-of-concept)을 제시합니다. 전통적인 웜은 미리 정해진 취약점을 활용하여 전파되었지만, 이 새로운 웜은 각 목표에 맞춘 공격 전략을 실시간으로 생성합니다. 이러한 특성으로 인해 공격자는 새로운 감염을 생성하는 데 추가 비용이 발생하지 않으며, 이는 공격자와 방어자 사이의 경제적 비대칭을 초래합니다.

- **Technical Details**: 실험은 리눅스, 윈도우 및 IoT 기기를 포함한 네트워크 상에서 제한된 가상 환경 내에서 수행되었습니다. 이 웜은 훔친(computed) 자원을 통해 유지되는 대형 언어 모델(LLM)을 활용하여 공격 로직을 생성하고, 이를 통해 다양한 공격 전략을 구사합니다. 이전의 고정된 코드 대신, 이 웜은 기록된 정보에 기반하여 타겟에 적응하며 실시간으로 공격 논리를 합성합니다.

- **Performance Highlights**: 연구팀은 AI 구동의 자기 유지형 사이버 위협이 더 이상 이론적인 위험이 아님을 증명했습니다. 이는 인간 운영자 없이 전파될 수 있으며, 결과적으로 고정된 코드가 아닌, 타겟에 대한 추론 능력으로 정의될 수 있습니다. 저자들은 이 새로운 위협에 맞서기 위한 방어 체계의 설계를 위한 기초 자료를 제공하고 있고, 연구 결과는 사이버 보안 및 공공 정책 분야의 중요한 도전으로 평가되고 있습니다.



### Consistency Training Can Entrench Misalignmen (https://arxiv.org/abs/2606.03810)
Comments:
          Accepted to ICML 2026

- **What's New**: 이번 연구에서는 consistency training(일관성 훈련)이 모델의 정렬(alignment) 효과에 미치는 영향을 심층적으로 분석했습니다. 특히, 다양한 개방형 소스 모델을 대상으로 하여, 이 훈련 방법이 의도하지 않은 행동을 증폭시킬 가능성을 조사했습니다. 이러한 연구는 일관성 훈련이 단순히 무작위로 적용될 수 없으며, 면밀한 검토가 필요하다는 점을 강조합니다.

- **Technical Details**: 연구팀은 108개의 '모델 유기체'(model organisms)를 사용하여 일관성 훈련 방법 7가지를 테스트했습니다. 이 과정에서, 보상 해킹(reward hacking)과 긴급히 발생하는 비정렬(emergent misalignment)은 억제되는 경향을 보였으나, 아첨(sycophancy)은 오히려 증폭되는 결과를 나타냈습니다. 이러한 결과는 일관성 라벨링 과정에서 발생하는 분포 변화(distribution shifts)가 정렬 효과의 주요 원인임을 시사합니다.

- **Performance Highlights**: 일관성 훈련의 결과는 일관성 훈련이 비정렬을 증폭하거나 억제할 수 있는 조건을 도출하기 위한 통합 이론적 프레임워크를 제공했습니다. 이 연구는 일관성 훈련이 단순한 패러다임에서 벗어나, 중요한 시스템에서의 사용 시 신중하게 감사(audit)해야 한다는 결론에 도달했습니다. 또한, 이 접근 방식은 모델의 행동을 정밀하게 제어할 수 있는 통찰력을 제공합니다.



### PURGE: Projected Unlearning via Retain-Guided Erasur (https://arxiv.org/abs/2606.03808)
Comments:
          13 pages, 10 figures, 6 tables

- **What's New**: 본 논문에서는 PURGE라는 기계적 비학습(machine unlearning) 알고리즘을 제안합니다. PURGE는 지속적 학습(Continual Learning, CL)과 기계적 비학습이 상호 대칭적인 문제라는 관찰에서 비롯되었습니다. 이 알고리즘은 매 단계에서 유지 세트 손실을 증가시키지 않도록 제한하는 경량의 그래디언트 투영(gradient projection) 방식을 사용합니다.

- **Technical Details**: PURGE는 A-GEM을 기반으로 하여 forget-set의 활성화를 intermediate layer에서 제거하고, retain-distribution을 향해 밀어냅니다. 핵심 설계 선택 중 하나는 retain-confusion target으로, 이는 forget 분포를 단순히 uniform 분포로 밀어내는 대신, 모델의 자연스러운 혼동 패턴을 목표로 삼습니다. 알고리즘은 두 가지 자가 조정 중단 기준(retain-loss budget와 forget-accuracy target)을 가지고 있어 수동 조정 없이 스스로 중단 시점을 결정합니다.

- **Performance Highlights**: 다섯 개의 데이터셋(CIFAR-10, MNIST, SVHN, STL10, PathMNIST)에서의 실험에 따르면, PURGE는 retain 정확도를 96% 이상 유지하며 MIA AUROC는 0.5에 근접하여 우수한 결과를 나타냅니다. 이는 다른 기법인 gradient ascent, KL-uniform 등과 비교했을 때 개인정보 보호-유용성 경계에서 성능이 뛰어난 것으로 입증되었습니다.



### LiveBand: Live Accompaniment Generation in the Audio Domain (https://arxiv.org/abs/2606.03803)
- **What's New**: LiveBand는 실시간으로 라이브 오디오 입력에 대한 고충실도 음악 반주를 생성하는 시스템입니다. 이는 엄격한 인과적 제약 조건을 준수하면서, 후속 오디오 프레임에 대한 예측 없이 작동할 수 있도록 설계되었습니다. 본 논문에서는 기존 모델에서의 다음 단계 예측 감독을 대체하여 조건부 적대적 판별기를 통한 시퀀스 수준 감독을 사용하는 방법을 제안합니다.

- **Technical Details**: LiveBand의 훈련은 미리 훈련된 오디오 오토인코더의 연속 잠재 공간에서 인과적 변환기(generator) 사용하여 이루어지며, 판별자로부터 적대적 시퀀스 수준 감독을 포함합니다. 모델은 각 시간 스텝에서 오직 인과적으로 이용 가능한 혼합 맥락과 가우시안 노이즈만을 수신하여 다음 출력을 예측합니다. 이를 통해 학습의 설계를 통해 추론 시간 분포와 일치시키며, teacher forcing과 관련된 노출 편향을 제거합니다.

- **Performance Highlights**: 다중악기 음악 반주 벤치마크에서 LiveBand는 오디오 품질, 비트 정렬, 믹스 준수에 대한 객관적 측정에서 이전 작업보다 개선된 성과를 보였습니다. 또, 소비자 하드웨어에서 미래를 바라보지 않고 실시간 스트리밍 생성을 가능하게 합니다. 이러한 구조는 높은 음악적 일관성을 유지하면서도 레이턴시 제약을 고려할 수 있도록 설계되었습니다.



### Trading Human Curation for Synthetic Augmentation in RLVR (https://arxiv.org/abs/2606.03800)
Comments:
          21 pages, 5 main-text figures, 4 appendix figures. Preprint

- **What's New**: 본 논문은 강화학습에서 고품질 학습 과제가 부족한 문제를 다루며, 이러한 과제를 수작업으로 작성하는 것이 경제적으로 비효율적임을 지적합니다. 특히 RLVR(Verifiable rewards Reinforcement Learning)에서 에이전틱(Agentic) 언어 모델을 사용할 때, 수작업 대신 자동 생성된 과제가 인간 저자가 작성한 과제를 대체할 수 있는 가능성을 탐구하고 있습니다. 이 연구는 저비용 하에서 수작업 저자(base)에서 생성된 다양한 변형(variants)을 이용해 추가적인 인간 큐레이션을 줄이는 효과를 평가합니다.

- **Technical Details**: 연구에서는 10개의 수작업 과제를 기준으로 80개의 자동 생성된 변형을 활용하여 실험을 진행했습니다. 이 과정에서 다양한 조정 가능성을 고려하여 비율(ρ) 및 인건비 비율(c_human/c_aug)이 어떻게 증가하는지를 측정하였습니다. 특히, 자동 생성된 내용이 추가적인 수작업 과제를 대체하면서도 유의미한 일반화 능력을 유지할 수 있음을 보여주었습니다.

- **Performance Highlights**: 실험 결과, 자동 생성된 80개의 변형이 전통적인 수작업 97개 과제와 유사한 성능을 발휘했으며, 대체적으로 점수에서 약 +0.96% 상승을 보였습니다. 또한, 이 전략은 비용 효율성 측면에서 유의미한 결과를 보였으며, 비용 조정 거래율(ρ_cost)은 [1.4×, 11.6×] 범위에서 수치적으로 안정성을 보여주었습니다. 이러한 결과는 RLVR의 경제성을 개선하고, 데이터 큐레이션 비용을 낮추는 가능성을 제시합니다.



### Signed Spiking Neuron Enabled by an Orthogonal-Easy-Axis Magnetic Tunnel Junction (https://arxiv.org/abs/2606.03796)
- **What's New**: 이번 연구에서는 전통적인 스파이킹 뉴론보다 풍부한 정보를 전달할 수 있는 서명 스파이킹 뉴론(signed spiking neurons)에 대한 새로운 설계를 제안합니다. 이 연구는 서명 누적 통합 및 발화(signed leaky integrate-and-fire, LIF) 작업을 위한 컴팩트한 자기 터널 접합(magnetic tunnel junction, MTJ) 기반 뉴론을 소개합니다. 이 장치는 자유층과 고정층의 직각 쉬운 축(orthogonal easy axes)을 통해 양극 스파이크 생성(bipolar spike generation)을 가능하게 합니다.

- **Technical Details**: 제안한 장치는 자기 모멘트 동역학(magnetic-moment dynamics)을 서명 LIF 막 전위(evolution of signed LIF membrane-potential)로 매핑합니다. Landau-Lifshitz-Gilbert 시뮬레이션(Landau--Lifshitz--Gilbert simulations)에 따르면, 최적의 자유층 치수는 장치 응답이 서명 LIF 방정식을 따르게 할 수 있습니다. 대표적인 설계 크기는 10 nm x 45 nm x 50 nm이며, 약 2:9:10의 종횡비(aspect ratio)를 가지고 있습니다.

- **Performance Highlights**: 이 장치-뉴론 모델을 이용한 네트워크 평가 결과, CIFAR-10에서 91.06%, CIFAR10-DVS에서 77.40%의 정확도를 달성하며, 이상적인 서명 LIF 뉴론의 대부분의 정확도를 유지합니다. 이러한 성능은 새로운 MTJ 기반 뉴론이 높은 정보 전달 능력을 가지고 있음을 나타냅니다.



### E2LLM: Towards Efficient LLM Serving in Heterogeneous Edge/Fog Environments (https://arxiv.org/abs/2606.03770)
- **What's New**: 대형 언어 모델(LLMs)의 배포는 비용 효율성, 낮은 대기 시간 및 최적의 자원 활용성을 고려해야 합니다. 이 논문에서는 제한된 자원 환경에서 LLM을 효율적으로 배포하기 위한 새로운 프레임워크 E2LLM을 소개합니다. E2LLM은 단순히 모델을 분할하는 것이 아니라 여러 장치 그룹에 모델을 복제하고 각 복제본에서 모델 병렬성을 적용합니다.

- **Technical Details**: E2LLM은 각 복제본을 PREFILL 또는 DECODER 역할로 분리하여 입력 및 출력 토큰 처리의 효율성을 높입니다. 구체적으로, Genetic Algorithm을 사용하여 장치를 클러스터로 구성하고, 각 클러스터 내에서 Dynamic Programming을 이용하여 최적의 파티션 전략을 결정합니다. 이러한 구조는 Edge 및 Fog 환경에서의 자원 제약을 고려하여 모델 병렬 실행의 병목 현상을 최소화합니다.

- **Performance Highlights**: 실험 결과, E2LLM은 분할 방법(Splitwise)과 비교하여 높은 수요 조건에서 평균 대기 시간을 50% 이상 줄이고, 2배 높은 디코딩 처리량을 달성했습니다. 이는 LLM의 효율적인 배포가 가능함을 시사하며, 다양한 작업 부하에 효과적으로 적응할 수 있는 능력을 보여줍니다.



### Merit or networks? What decides where research is published (https://arxiv.org/abs/2606.03763)
- **What's New**: 이 논문은 과학 출판이 아이디어의 질을 보상하는지 아니면 학계의 연결 고리를 보상하는지를 다룹니다. 기존 연구가 출판 결과를 바탕으로 측정할 수밖에 없었던 한계를 극복하고, LLM evaluator를 통해 출판 전 텍스트에서 아이디어의 질을 직접 측정합니다. 경제학을 사례 연구로 사용하여 다섯 가지 입력 변수를 결합하여 저널 배정의 예측 모델을 개발했습니다.

- **Technical Details**: 연구에서는 논문의 아이디어 질을 직접 수치화하기 위해 경제학 관련 6,208개의 작업 논문을 분석했습니다. 제시된 모델은 execution quality, connection index, author-ability index, 및 언어 모델의 텍스트 점수를 포함하는 다섯 가지 입력 변수를 사용합니다. 결과적으로 이러한 입력은 서로 경쟁하는 것이 아니라 Prestige(위신)의 사다리에서 순서대로 배열되어 있음을 보여줍니다.

- **Performance Highlights**: 연구 결과, execution은 가장 중요한 요소로 나타났으며, 아이디어 질이 그 다음으로 중요합니다. 반면, 연결의 중요성은 주로 가장 선택적인 저널에서의 성공에 영향을 미칩니다. 결국, 논문은 과학 출판의 meritocracy(능력주의)와 network (네트워크) 관점을 모두 포함하는 결과를 제시합니다.



### Tool-Aware Optimization with Entropy Guidance for Efficient Agentic Reinforcement Learning (https://arxiv.org/abs/2606.03762)
- **What's New**: 이번 연구에서 제안하는 TAO-RL은 대규모 언어 모델(LLM)의 효율적인 정책 최적화를 위한 통합 프레임워크입니다. 이 프레임워크는 도구 인지가 있는 경로 필터링(tool-aware trajectory filtering)과 엔트로피 기반 탐색(entropy-guided exploration)을 결합하여, 외부 도구와의 상호작용이 훈련에 미치는 부정적인 영향을 줄이는 데 초점을 맞추고 있습니다. 기존의 방법들의 한계를 보완하기 위해 TAO-RL은 실패한 도구 호출을 필터링하고, 유용한 정보가 포함된 데이터만을 사용함으로써 안정적인 훈련 배포를 형성합니다.

- **Technical Details**: TAO-RL은 두 가지 기준을 통해 데이터를 필터링합니다. 첫째, 도구 호출이 실패한 경로는 제거되어, 훈련에 불필요한 잡음을 주입하지 않도록 합니다. 둘째, 모든 롤아웃이 정답이거나 전부 오답인 경우도 제거하여, 정책이 학습할 수 없는 사례를 피합니다. 알고리즘 측면에서 TAO-RL은 엔트로피 기반의 보너스를 도입하여, 훈련 분포에서 도구 호출 후 엔트로피가 높은 포인트에서 탐색에 대한 인센티브를 제공합니다.

- **Performance Highlights**: 7개의 복잡한 추론 벤치마크에서 실시된 광범위한 실험 결과, TAO-RL은 기존 방법들보다 우수한 성과를 보였으며, 정책의 안정성과 추론 성능에서 뛰어난 결과를 나타냈습니다. TAO-RL은 차별화된 답변 다양성을 유지하며, 도구와의 상호작용에서 더욱 강력한 추론 행동을 드러내는 것을 목표로 합니다.



### Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models (https://arxiv.org/abs/2606.03748)
Comments:
          31 pages, 8 figures

- **What's New**: 본 연구는 Ultralytics YOLO26을 제시하며, 여러 YOLO 모델의 한계를 극복하기 위해 통합된 실시간 비전 모델 패밀리를 개발했습니다. YOLO26은 NMS(Non-Maximum Suppression)가 필요 없는 이중 헤드 설계를 채택하여, 더 가벼운 회귀 헤드와 자유로운 회귀 범위를 제공합니다. 이 모델은 MuSGD와 Progressive Loss, STAL을 결합한 교육 파이프라인을 통해 훈련을 최적화하고, 작은 객체에 대한 완전한 라벨 할당을 보장합니다.

- **Technical Details**: YOLO26의 구조는 두 개의 헤드 설계로 구성되어 있어 NMS가 없는 전방향 추론을 가능하게 하고, DFL(Distribution Focal Loss)을 완전히 제거하여 파라미터 수를 줄였습니다. MuSGD(하이브리드 최적화 기법), Progressive Loss(훈련 중 감독을 변화시키는 기법), 그리고 STAL(작은 대상에 대한 라벨 할당 전략)이라는 세 가지 훈련 구성 요소가 긴 훈련 주기를 단축시킵니다. 모델은 n/s/m/l/x의 다섯 가지 크기로 제공되며, 객체 탐지, 인스턴스 분할, 포즈 추정 등을 지원합니다.

- **Performance Highlights**: YOLO26은 COCO 데이터셋에서 1.7-11.8 ms의 지연 시간에 걸쳐 40.9-57.5 mAP을 달성하며 실시간 탐지 분야에서 다른 모델들을 초월했습니다. YOLOE-26은 LVIS minival 에서 텍스트 프롬프트 하에 40.6 AP을 기록하여 DetCLIP-T 보다 +6.2 AP 개선을 달성했습니다. 모든 모델 스케일에서 YOLO26은 최신 실시간 탐지기에 비해 정확도-지연 시간 기준을 개선하며, 향상된 정확도를 제공합니다.



### Qwen-Image-Flash: Beyond Objective Design (https://arxiv.org/abs/2606.03746)
- **What's New**: 본 논문에서는 few-step distillation의 접근 방식을 재조명하여, 학생 모델의 성능을 지적으로 형성하는 훈련 레시피에 중점을 두고 조사합니다. 특히, Qwen-Image-2.0을 사례로 하여 데이터 구성, 교사 안내, 작업 혼합 등 세 가지 주요 요소를 체계적으로 분석합니다. 이러한 연구 결과는 Qwen-Image-Flash라는 새로운 모델 개발로 이어지며, 적은 단계로도 고품질 이미지를 생성할 수 있도록 합니다.

- **Technical Details**: 본 연구에서는 flow matching과 DMD(Discrete Mode Decomposition)를 적극 활용하여 다단계 교사 모델을 few-step 학생으로 증류합니다. flow matching은 데이터와 노이즈 간의 전송 과정을 정의하고, DMD는 학생이 교사의 조건부 분포에 접근하도록 유도합니다. 이러한 방법론은 다양한 작업과 상황에 대한 성능을 개선하는 데 기여합니다.

- **Performance Highlights**: Qwen-Image-Flash는 단 44번의 기능 평가(NFE)로 T2I 생성 및 지시 기반 이미지 편집 작업을 동시에 수행할 수 있도록 설계되었습니다. 이러한 성과는 시각 품질과 강력한 합성 능력을 유지하면서도 효율적인 모델을 생성하는 데 중요한 성과를 거두었음을 보여줍니다. 실험적 분석 결과는 데이터 구성, 교사 안내, 작업 혼합이 효과적인 few-step distillation에 있어 중요한 요소임을 강조합니다.



### Staying Alive: Uncensored Survival Analysis with Tabular Foundation Models (https://arxiv.org/abs/2606.03689)
- **What's New**: 본 연구에서는 Survival Analysis (SA)에서의 우측 검열 문제를 해결하기 위해 Training-Free 방법을 제안합니다. Tabular Foundation Models (TFM)을 활용하여 이벤트 발생 시간을 예측하고, 우측 검열된 데이터를 반복적으로 보완하는 방식을 도입하였습니다. 제안된 방법은 단일 스칼라 매개변수만으로 Accelerated Failure Time (AFT) 모델을 구축할 수 있는 기법으로, 기존 기법들과 비교해 훈련 과정이 필요 없습니다.

- **Technical Details**: 이 모델은 Buckley-James 추정기를 바탕으로 비모수적 in-context 추정기를 통해 우측 검열 데이터를 impute 합니다. AFT 모델의 수식을 사용하여 로그 가능성을 최대화하는 방식으로 단일 스칼라 파라미터를 조정하고, 초기화 시 Kaplan-Meier jackknife 추정기를 통해 데이터 기반의 warm start를 제공합니다. 이 방식은 데이터를 비모수적으로 다룰 수 있는 강점을 지니고 있습니다.

- **Performance Highlights**: 실험을 통해 제안된 방법이 Cox 회귀 및 모수적 AFT 모델을 포함한 여러 생존 회귀 모델들과 경쟁력 있는 성능을 보임을 확인하였습니다. 여러 생존 분석 벤치마크에서의 결과는 기존의 훈련이 필요한 모델들과 유사한 성능을 제공하며, 권장된 방식이 실제 SA 응용에 효과적임을 보여줍니다.



### A Close Look At World Model Recovery In Supervised Fine-Tuned LLM Planners (https://arxiv.org/abs/2606.03685)
Comments:
          17 pages. Under review at TMLR

- **What's New**: 이번 논문은 대형 언어 모델(LLM)에서 감독된 세분화(Supervised Fine-Tuning, SFT)가 클래식 계획 문제를 해결하는 데 어떻게 기여하는지를 탐구하고 있습니다. 일반적인 계획 문제의 복잡성과 LLM의 제너레이티브(generative) 기능으로 인해 문제 해결 과정에서 세계 모델(world model)의 회복이 이루어지는지에 대한 질문을 제기합니다. 연구 결과에 따르면, SFT가 LLM의 내부 표현을 포함하여 계획 문제에 대한 이해도를 높일 수 있음을 보여줍니다.

- **Technical Details**: 우선, 연구에서는 LLM이 동작의 유효성과 상태 프레디케이트(state predicates)를 선형적으로 인코딩하는 방법을 제시합니다. LLM들은 의사결정 과정에서 세계 모델을 활용하는 방식으로 학습할 수 있으며, 이는 전통적인 계획 문제 해결에서 중요한 역할을 할 수 있습니다. 또한, 다양한 데이터 분포를 사용하여 LLM의 SFT 과정이 세계 모델 회복에 미치는 영향을 분석하며, 행동의 유효성을 결정하는 과정에서 발생하는 도전과제들을 해결하려 합니다.

- **Performance Highlights**: 결과적으로, 유효한 행동 시퀀스에 직접적으로 SFT를 적용함으로써 LLM은 유효한 행동과 상태 프레디케이트의 진리 값을 효과적으로 학습할 수 있습니다. 계획 데이터에서 좋은 상태 공간 커버리지를 제공받은 LLM들은 일반적으로 더 정확한 세계 모델 회복을 보이며, 이는 프로젝트의 전반적인 성과를 향상하는 데 기여합니다. 이 연구는 LLM의 계획 능력이 과연 효과적으로 나타나는지를 검증하는 데 중요한 통찰을 제공합니다.



### AUGUSTE: Online-Learning dApp for Predictive URLLC Scheduling (https://arxiv.org/abs/2606.03664)
- **What's New**: 본 논문은 5G의 초신뢰성 및 저지연 통신(URLLC)을 위해 새로운 학습 기반의 MAC(Medium Access Control) 스케줄링 프레임워크인 AUGUSTE를 제안합니다. AUGUSTE는 온라인 머신 러닝(ML) 모델을 활용하여 패킷 도착 예측 및 자원 할당을 사전에 진행하는 방식을 적용합니다. 이를 통해 기존의 Scheduling Request(SR) 절차를 개선하고, 자원의 비효율적인 사용을 줄일 수 있는 방법을 제시합니다.

- **Technical Details**: AUGUSTE는 적응형 상태 기계(adaptive state machine)를 사용하여, unbiased한 도착 통계를 수집하는 학습 단계와 학습된 예측을 활용하여 트래픽이 예상될 때에만 스케줄링하는 확신 단계로 전환됩니다. 이는 URLLC 트래픽 패턴(request-response, ML edge inference, 주기적인 자율 보고) 세 가지에 대해 OpenAirInterface를 실행하는 실제 5G 테스트베드에서 평가되었습니다. AUGUSTE의 설계는 특히 SR 기반의 절차에서 발생하는 오버헤드를 크게 감소시킵니다.

- **Performance Highlights**: AUGUSTE는 항상 켜져 있는 스케줄링의 중앙값 RTT(Round Trip Time)와 일치하는 성과를 보여주며, 이는 약 10 ms로 SR 기반의 기준인 20 ms를 절반으로 줄인 것입니다. 또한, AUGUSTE는 자원 비용 측면에서도 약 1/10로 줄이며, 오버헤드가 7-10%에 불과하여 효율성을 높입니다. 이로 인해 URLLC의 원활한 운영을 통한 5G 기술의 발전에 기여할 수 있습니다.



### CoEval: Ranking Language Models for Custom Tasks Without Labeled Data or Trustworthy Benchmarks (https://arxiv.org/abs/2606.03650)
Comments:
          19 pages, 6 images

- **What's New**: 이번 논문에서는 CoEval이라는 오픈 소스 프레임워크를 제시합니다. 이 프레임워크는 인간 레이블이 없는, 오염이 없는 새롭고 속성이 통제된 벤치마크를 생성하여 언어 모델을 평가할 수 있는 방법을 제공합니다. 특히, task-specific labeled data가 없을 때에도 언어 모델을 선택하거나 순위를 매기는 과정을 간소화합니다.

- **Technical Details**: CoEval은 teacher 모델을 사용하여 작업 또는 도메인의 설명만으로 새로운 평가 항목을 생성합니다. 이 시스템은 judge ensemble을 통해 후보 모델의 순위를 매기며, 인간 평가자가 필요하지 않습니다. 평가 위원회의 다양성(vendor diversity)이 신뢰도를 높이는 데 기여하며, 소수의 잘 선택된 위원이 가장 신뢰할 수 있는 결과를 제공합니다.

- **Performance Highlights**: CoEval은 실제 모델 순위를 복원하며, ground-truth 정확도를 0.86으로 추적합니다. 생성된 항목들은 주요 공개 벤치마크와 전혀 겹치지 않으며, 평가 패널은 verbosity bias를 제거하고 self-preference를 방지합니다. 본 연구는 4개의 작업에서 7,978회의 평가를 생성하며, 각 모델 출시 시 재실행할 수 있는 저렴한 프로세스를 제공합니다.



### Safety Measurements for Fine-tuned LLMs Should be Grounded in Capability (https://arxiv.org/abs/2606.03648)
Comments:
          8 pages plus appendices

- **What's New**: 이 논문에서는 대규모 언어 모델(LLM)의 세부 조정을 통해 사용자의 작업이나 선호 스타일에 맞게 조정하는 과정에서 발생할 수 있는 안전성의 손상 문제를 다룹니다. 저자들은 모델의 안전성을 유지하기 위해 세부 조정(fine-tuning)이 특정 능력 목표에 기반해야 한다고 주장하며, 이를 통해 무작위적인 실험적 선택을 피해 유의미한 결론을 도출할 수 있다고 설명합니다. 논문은 세부 조정이 모델 행동에 미치는 영향을 다차원적으로 평가하면서 안전성 또한 고려합니다.

- **Technical Details**: 연구자들은 다양한 세부 조정 방법과 미세 조정(微調整, fine-tuning)의 하이퍼파라미터 조정이 모델의 안전성 평가에 중요한 영향을 미칠 수 있음을 지적합니다. LoRA (Low-Rank Adaptation) 기법을 포함한 여러 안전성 보존 방법이 제안되었고, SafeLoRA는 특정 안전 공간을 정의하여 이를 통해 모델 세부 조정 후에도 안전 행동을 유지하려는 접근 방식입니다. 논문은 세부 조정 데이터셋과 평가 프로프트를 사용하여 안전성 평가 방법이 모델 행동에 미치는 영향을 비교 분석합니다.

- **Performance Highlights**: 연구 결과는 세부 조정된 모델들이 안전 프롬프트에 대해 일관되지 않은 출력을 생성할 수 있음을 보여줍니다. 가벼운 세부 조정 기술이 기존의 원래 모델의 안전성을 저해할 수 있으며, 다양한 안전 기준에 따라 세부 조정의 결과가 달라질 수 있음을 강조합니다. 마지막으로, 저자들은 SafeLoRA 방법을 통해 세부 조정과 안전 간의 트레이드오프를 비교하면서 안전성 평가 방법론의 개선 필요성을 강조합니다.



### Black-box, Adaptive, Efficient, Transferable, Harmful, Applicable... Attacks Are All You Need to Break LLMs (https://arxiv.org/abs/2606.03647)
- **What's New**: 본 논문에서는 Indirect Harm Optimization(IHO)이라는 새로운 공격 방법론을 제안합니다. IHO는 복잡한 디펜스 파이프라인이나 폐쇄형 모델에 대한 공격의 전반적인 평가와 비교 가능성을 향상시키기 위해 설계되었습니다. 기존의 공격 방법이 충족하지 못했던 블랙박스(black-box) 접근성 요구사항을 충족하며, 자동화된 방식으로 해로운 공격을 수행하는 것이 가능합니다.

- **Technical Details**: IHO는 피해 판단(judge)과의 반복적 선호 최적화(iterative preference optimization)를 통해 훈련된 마스크된 확산 언어 모델(masked diffusion language model) 기반의 공격자입니다. 이 공격자는 단일 쿼리에 대한 적응형 공격을 수행할 수 있으며, 다양한 방어 모델과 미세 조정 없이도 적용할 수 있습니다. IHO는 모델의 복잡한 멀티 스테이지 파이프라인과 데이터 감지기를 넘어서서 신뢰성 있는 공격 성과를 얻는 것을 목표로 합니다.

- **Performance Highlights**: IHO는 이전의 최첨단 방법들에 비해 상당한 공격 성공률을 달성하며, 복층 방어(Circuit Breaker와 같은)와 결합된 모델에서도 효과적으로 작용합니다. IHO의 구현 비용은 상대적으로 낮아 효율성을 극대화할 수 있으며, 다양한 모델에 대해 적응할 수 있는 능력을 갖추고 있습니다. 이 연구 결과는 IHO가 표준화된 jailbreaking 평가가 신뢰성을 향상시킬 수 있는 실용적인 단계로 자리잡는 것을 보여줍니다.



### The Shape of Addition: Geometric Structures of Arithmetic in Large Language Models (https://arxiv.org/abs/2606.03645)
Comments:
          Accepted by ICML 2026

- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)의 내적 계산과 이산 출력 간의 불일치를 강조합니다. 논문은 다중 피연산자 덧셈 시의 잔여 스트림 기하학을 분석하여 Iso-Raw-Sum Trajectory (IRST)라는 새로운 기하학적 구조를 발견했습니다. 이 연구는 또한 Noisy Quantization Model을 제안하여 내부 신경 소음에 의해 발생하는 산술 오류의 메커니즘을 설명하고 있습니다.

- **Technical Details**: 연구는 모델이 세 자리 수 이상의 10자리 정수를 더하는 복잡한 산술 작업에서의 내부 표현을 조사합니다. 잔여 스트림 활성화의 분석을 통해 probe versatility라는 현상을 발견했으며, 이는 경량 프로브가 단일 활성화 벡터에서 다양한 신호를 동시에 해독할 수 있음을 보여줍니다. 이러한 신호의 공존은 LLM의 표현 공간이 고도로 구조화된 다양체임을 의미합니다.

- **Performance Highlights**: 이 논문은 이론적인 프레임워크를 통하여 수학적 진리를 지키면서 잘못된 토큰 선택이 발생하는 경우에도 올바른 잠재 신호가 보존됨을 검증합니다. 마지막으로, 이 연구는 이중 스트림 일관성 점검 방법을 도입하여 추론 시 성능을 크게 회복하는 데 기여합니다. 이를 통해 LLM의 내부 표현이 올바른 수학적 요소를 유지할 수 있음을 확인하였습니다.



### VidMsg: A Benchmark for Implicit Message Inference in Short Videos (https://arxiv.org/abs/2606.03635)
Comments:
          Project page: this https URL

- **What's New**: 이번 논문에서는 VidMsg라는 새로운 벤치마크를 소개하며, 이는 짧은 온라인 비디오에서의 암묵적인 메시지 이해를 평가하기 위해 설계되었습니다. VidMsg는 9개의 주제 영역에서 400개의 YouTube 클립으로 구성되어 있으며, 여기에는 경력, 금융, 교육, 건강과 웰빙, 문화, 안전, 지속 가능성 및 라이프스타일과 같은 다양한 분야가 포함됩니다. 이 연구는 비디오가 전달하는 메시지를 인식하는 데 초점을 맞추고 있으며, 이는 단순한 시각적 내용 이상으로, 보다 높은 수준의 의미를 요구합니다.

- **Technical Details**: VidMsg의 구성은 메시지 중심 데이터 수집 파이프라인을 사용하여 이루어집니다. 대상 메시지를 바탕으로 간접 검색 시나리오를 생성하고, 이를 통해 후보 클립들을 검색하여 인간 주석자가 각 클립이 얼마나 목표 메시지를 전달하는지 평가합니다. 이러한 방식은 메시지가 너무 명확하게 드러나지 않도록 필터링하여 비디오 메시지를 보다 잘 이해할 수 있는 벤치마크를 생성합니다. 또한, VidMsg는 예측된 메시지를 선택하는 다중 선택 질의 응답(MCQ) 과제를 포함하여 모델의 메시지 판단 능력을 검사합니다.

- **Performance Highlights**: 실험 결과, 기존의 비디오 언어 및 검색 모델들은 VidMsg에서 좋은 성능을 보이지 않았습니다. 이는 모델들이 단순한 시각적 인식에 의존할 뿐만 아니라, 실질적인 추론과 맥락적 단서를 통합해야 하기 때문입니다. 마지막으로, VidVec-Msg라는 기초 방법을 제안하며, 이는 메시지 중심 검색 성능을 향상시키고 향후 연구를 위한 여지를 남겨 두었습니다.



### AnchorMoE: Interpretable Time Series Classification via Anchor-Routed MoE (https://arxiv.org/abs/2606.03631)
Comments:
          Accepted by KDD 2026, 12 pages

- **What's New**: 이 논문은 AnchorMoE라는 해석 가능한 분류 프레임워크를 제안합니다. 이 모델은 Mixture-of-Experts (MoE) 아키텍처를 기반으로 하여, 로컬 패치의 다각적 표현을 포함하고 전문화된 전문가에게 라우팅하는 방식으로 구성됩니다. 최종 예측 결과는 입력 세그먼트의 정확한 가감합으로 나타나며, 이는 사전 접근 방식으로 투명성을 보장합니다.

- **Technical Details**: AnchorMoE는 여러 변수를 고려하여 각 패치를 시간, 스펙트럼, 맥락적 표현으로 확장하고 이를 전문화된 전문가에게 라우팅함으로써 최종 예측을 수행합니다. 또한, 전문가 간의 상호 직교성을 보장함으로써 다양한 예측 패턴에 특화된 전문가가 서로 다른 역할을 수행하도록 유도합니다. 불확실성을 감지하는 신뢰성 게이트를 도입하여 배경 노이스를 효과적으로 억제하며, 이를 통해 모델의 신뢰성을 유지합니다.

- **Performance Highlights**: 실제 및 합성 벤치마크를 활용한 광범위한 평가 결과, AnchorMoE는 경쟁력 있는 분류 성능을 달성합니다. 이 모델은 고급 분류기와 비교할 때 정확성을 유지하면서도 결정의 근거를 명확하게 제시할 수 있습니다. 즉, AnchorMoE는 해석 가능성과 예측 성능 간의 균형을 이룸으로써, 고위험 분야에서 안전한 배포를 가능하게 합니다.



### Building Reliable Long-Form Generation via Hallucination Rejection Sampling (https://arxiv.org/abs/2606.03628)
Comments:
          accepted by ICML 2026

- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 내용 생성 과정에서 발생하는 허위 정보 문제를 해결하기 위한 새로운 접근 방식을 제시합니다. 제안된 방법은 'Segment-wise HAllucination Rejection Sampling (SHARS)'라는 프레임워크로, 생성되는 내용 중 허위 정보를 탐지하고 이를 거부하는 과정을 적용하여 정확한 정보를 얻는 방법론입니다. 이 프레임워크는 자기 수정 기능을 가지고 있어 외부 데이터 소스를 사용하지 않고도 허위 정보를 줄일 수 있습니다.

- **Technical Details**: SHARS는 생성 과정에서 무작위적으로 선택된 탐지기를 사용하여 생성된 문장을 평가합니다. 각각의 문장은 허위 정보가 포함되어 있는지 확인한 후, 다음의 세 가지 방식 중 하나로 처리됩니다. 허위 정보가 전혀 없는 경우 문장을 유지하고, 혼합된 문장에서는 허위 정보 제거 후 재작성하며, 허위 정보만 있는 문장은 폐기합니다. 이러한 방식으로 허위 정보가 전파되는 현상을 방지하면서 신뢰할 수 있는 내용을 포함한 출력을 생성합니다.

- **Performance Highlights**: 실험 결과 SHARS는 초장기적 형식 생성에서 허위 정보를 현저히 줄이며, 생성의 유익성을 유지하거나 향상시키는 효과를 보였습니다. 또한, SHARS는 적절히 설정된 경우, 추가적인 계산량을 할당할수록 사실 기반의 정확도가 지속적으로 향상되는 경향을 보였습니다. 실제 평가를 통해 FactScore 벤치마크에서는 약 26%의 정보 정확도를 개선한 것으로 나타났습니다.



### TurtleAI: Benchmarking Multimodal Models for Visual Programming in Turtle Graphics (https://arxiv.org/abs/2606.03626)
Comments:
          ACL Findings 2026 paper

- **What's New**: TurtleAI라는 새로운 벤치마크가 소개되었습니다. 이는 교육 중심의 Turtle Graphics 작업을 통해 비전-언어 모델(VLM)의 성능을 평가하기 위한 것입니다. TurtleAI는 823개의 실제 시각 프로그래밍 작업을 기반으로 한 과제를 포함하고 있으며, VLM이 기하학적 패턴을 인식하고 Python 코드를 생성하는 데 필요한 능력을 테스트합니다.

- **Technical Details**: TurtleAI는 VLM이 비주얼 태스크를 해결하기 위해 코드 생성 기능을 평가하는 귀중한 도구로, XLogoOnline 플랫폼의 업무들을 포함하고 있습니다. 이 벤치마크는 목표 이미지를 기반으로 Python 코드 생성을 요구하며, VLM은 공간 관계를 이해하고 이를 코드로 전환해야 합니다. 20개 이상의 VLM이 평가되었으며, GPT-5, GPT-4o, Qwen2-VL-72B 같은 최신 모델들이 30% 미만의 성공률을 기록했습니다.

- **Performance Highlights**: 모델 성능을 개선하기 위해 소량의 seed 샘플을 활용한 데이터 생성 기법이 제안되었습니다. 이 방법으로 얻은 합성 데이터로 Qwen2-VL-72B 모델을 파인튜닝한 결과, 실제 작업에서 약 20% 개선된 성과를 보였습니다. 그러나 GPT-4o는 공간 추론과 정확한 비주얼 재현에 어려움을 겪고 있으며, Qwen2-VL-72B는 코드 구현과 비주얼 추론 간의 정렬 문제가 주된 오류 원인으로 드러났습니다.



### Physics-Guided Policy Optimization with Self-Distillation (https://arxiv.org/abs/2606.03620)
- **What's New**: 이번 연구에서는 물리학에서 영감을 받아 개발된 Physics-Guided Policy Optimization (PGPO) 방법론을 소개합니다. PGPO는 self-distilled policy optimization (SDPO)의 훈련 중 발생하는 불안정성을 해결하기 위한 새로운 접근 방식으로, 상호 정보량(mutual information) 추정치를 활용하여 업데이트의 신뢰도를 조정합니다. 이 과정은 훈련 시 생성되는 신호의 질에 따라 동적으로 보정 크기를 조정하여 학습 안정성을 높입니다.

- **Technical Details**: PGPO는 비점성 유체 역학(viscous fluid dynamics)에서 영감을 얻어 구성된 방법으로, 업데이트 단계의 크기를 정보적 신뢰성에 따라 조정하는 메커니즘을 채택합니다. 연구에서는 PGPO의 동역학이 확률 미분 방정식(stochastic differential equations) 수준에서의 약한 근사(order-1 weak approximation)을 보여주며, 각 업데이트 과정에서의 오버헤드가 미미하다고 설명합니다. 이를 통해 SDPO의 기본 credit assignment 문제를 극복하고 있습니다.

- **Performance Highlights**: PGPO는 Science-QA 데이터셋에서 실험이 진행되었으며, 4개 도메인 중 3개에서 SDPO를 초과하는 성능 향상을 보여주었습니다. 특히, PGPO는 최대 4.5 포인트 까지의 성능 개선을 기록했으며, SDPO가 훈련 마지막 단계에서 붕괴될 때에도 안정성을 유지하는 성능을 보였습니다.



### Exploiting Verification-Generation Gap: Test-Time Reinforcement Learning with Confidence-Conditioned Verification (https://arxiv.org/abs/2606.03608)
- **What's New**: 이번 연구에서는 label-free 환경에서의 Pass@k 성능 최적화의 중요성을 강조하며, 성능 향상 방식으로 TTRL-CoCoV 프레임워크를 제안합니다. TTRL(CoCoV)의 핵심은 적응형 신뢰도 기반 메커니즘을 통해 불확실한 샘플을 평가하고, 탐색의 다양성을 유지하는 것입니다. 이 접근법은 기존의 RLVR(Reinforcement Learning from Verifiable Rewards) 방법론과 차별화되는 부분입니다.

- **Technical Details**: TTRL-CoCoV는 신뢰도가 높은 샘플에 대해 검증기를 부트스트랩하여 무신뢰 샘플에서 잘못된 pseudo-label을 걸러내고, 중간 신뢰도를 가진 샘플은 검증 없이 처리하는 방식으로 동작합니다. 이러한 신뢰도 조건부 메커니즘을 통해 평균적인 Pass@1 성능을 9.8% 향상시키고, Pass@16 성능을 18.7% 증가시켰습니다. 이 연구에서는 세 가지 주요 문제를 분석하여 해결 방안을 제시합니다: 노이즈 탐색 신호, 길이 붕괴, 경험적 형식 실패입니다.

- **Performance Highlights**: TTRL-CoCoV는 6개의 복잡한 추론 벤치마크에서 뛰어난 성능을 입증하며, 무신뢰 환경에서의 Pass@k 성능 저하를 역전시킵니다. 또한, 다양한 백본 모델을 적용하여 평균적으로 Pass@1 성능이 5.0% 향상되는 결과를 보였습니다. 이러한 성과는 %를 기준으로 관리된 탐색적 생성 범위의 폭을 크게 확장시킴으로써 이루어진 것입니다.



### Testing LLM Arithmetic Reasoning Generalization with Automatic Numeric-Remapping Attacks (https://arxiv.org/abs/2606.03606)
- **What's New**: 이 논문에서는 대규모 언어 모델(Large Language Models, LLMs)이 산술 문제를 푸는 데 있어 발견한 취약성을 다루고 있습니다. 특히, 숫자 변환(numeric remapping) 공격을 통해 문제의 구성이 바뀌더라도 같은 추론 방법을 적용해야 하는 경우 모델의 성능이 얼마나 저하되는지를 조사합니다. 자동화된 알고리즘을 사용하여 문제별 기호 표현을 생성하고, 제한된 숫자 변환을 통해 새로운 문제를 만들어 모델 성능의 저하를 측정합니다.

- **Technical Details**: 제안된 구조화된 파이프라인은 원 문제-답 변환에서 기호 표현을 추출하고, 변환 가능한 값을 제안하며, 자연어로 문제를 다시 서술하고, 변경된 정답을 재계산하는 과정을 포함합니다. 각 단계마다 중간 결과물을 검증하여 무효한 변환을 줄이고 문제 스키마를 유지합니다. 이 연구에서 다룬 숫자 변환 공격은 여러 산술 문제 벤치마크에서 모델 성능을 평가하여 기존 답변을 바탕으로 한 변환이 모델의 정확도에 미치는 영향을 보여줍니다.

- **Performance Highlights**: 실험 결과, GSM8K 데이터셋에서는 모델의 정확도가 12.16%에서 25.82%까지 감소하는 것으로 나타났습니다. 반면에 MAWPS와 MultiArith 데이터셋에서는 대부분의 정확도가 98% 이상으로 유지되어, 데이터셋의 구조에 따라 모델의 내구성이 크게 달라질 수 있음을 보여줍니다. 이로 인해 원래의 벤치마크 정확도가 산술 추론의 안정성을 과대평가할 수 있다는 점이 강조됩니다.



### CauTion: Knowing When to Trust LLMs for Ensemble Causal Discovery (https://arxiv.org/abs/2606.03602)
- **What's New**: 이번 논문에서는 기존의 방법론들이 갖고 있는 한계를 극복하기 위해 'CauTion'이라는 프레임워크를 제안합니다. 이 프레임워크는 LLM(large language model)의 도메인 지식을 여러 통계적 인과 발견 알고리즘의 집합에 신뢰성 평가를 통해 통합하는 방식으로 작동합니다. 또한, 알고리즘 간의 동의에 기반하여 LLM의 지식을 언제 활용할지를 판단하는 의사 결정 프로세스를 포함합니다.

- **Technical Details**: CauTion의 작동 과정은 세 단계로 나뉘어 있습니다. 첫 번째 단계에서는 알고리즘 집합이 여러 인과 발견 알고리즘의 출력을 집계하고, 모든 알고리즘이 동의하는 변수 쌍을 해결합니다. 두 번째 단계에서는 신뢰 보정(arbitration) 메커니즘을 통해 LLM과 알고리즘의 상대적 신뢰성을 추정하고, 신뢰도가 부족한 경우에만 LLM의 판단을 따르도록 합니다. 마지막으로, 사이클 수정을 수행하여 최종 인과 그래프가 비순환적(DAG) 구조를 준수하도록 보장합니다.

- **Performance Highlights**: 실험 결과, CauTion은 데이터 중심의 방법론과 LLM을 보강한 베이스라인보다 일관되게 우수한 성능을 나타냈습니다. 특히, 가장 큰 데이터셋인 Win95pts(n=76)에서 CauTion은 구조적 해밍 거리(SHD) 27을 기록하며 두 번째로 우수한 LLM 보강 방법보다 두 배 이상 개선된 성능을 보였습니다. 또한 CauTion은 다양한 LLM에 대해 강력한 견고성을 유지하며 성능을 발휘했습니다.



### DDOR: Delta Debugging for Explainable Overrefusal Testing and Repair (https://arxiv.org/abs/2606.03601)
- **What's New**: 이 논문에서는 LLM(대형 언어 모델)의 과도한 거부(overrefusal) 문제를 해결하기 위한 자동화된 프레임워크인 DDOR(Delta Debugging for OverRefusal)를 제안합니다. DDOR은 사용자 입력과 출력만을 기반으로 모델의 위험 요소를 점검하고 개선하는 데 중점을 두며, 이를 통해 모델이 안전하다고 간주하는 요청을 무차별적으로 거부하는 문제를 해결하고자 합니다. DDOR는 최소 거부 유도 조각(mRTF)을 로컬라이징하여 과도한 거부를 줄이는 방법을 제공합니다.

- **Technical Details**: DDOR은 세 가지 주요 컴포넌트로 구성되어 있습니다: 결함 로컬라이제이션(fault localization), 테스트 케이스 생성(test case generation), 그리고 오라클 검증(oracle validation)입니다. 결함 로컬라이제이션 단계에서는 델타 디버깅을 활용하여 거부를 유도하는 최소 조각을 pinpoint합니다. 생성된 새로운 프롬프트는 다양한 맥락과 의도를 가진 자연스러운 형식으로 미세 조정되어, 실제 사용자가 의도한 요청을 반영하여 과도한 거부를 탐지하는 데 효과적입니다.

- **Performance Highlights**: DDOR의 성능 평가 결과는 DDOR이 평균 19.30% 더 높은 과도한 거부율(test generation)과 함께 11.15배 더 많은 유효 테스트 케이스를 생성하는 것으로 나타났습니다. 수리(repair)의 경우, mRTF 기반 접근 방식은 평균 69.19%의 과도한 거부 감소를 보였으며, 원본 의도를 보존하는 데에도 더 나은 성과를 나타냈습니다. 이러한 결과는 DDOR이 과도한 거부 문제를 해결하는 데 있어 더 효율적이고 실제로 유용한 방법임을 나타냅니다.



### PHASER: Phase-Aware and Semantic Experience Replay for Vision-Language-Action Models (https://arxiv.org/abs/2606.03598)
Comments:
          12 pages, 5 figures

- **What's New**: 본 논문에서 우리는 PHASER (PHase-Aware and Semantic Experience Replay)라는 새로운 지속 학습 프레임워크를 소개합니다. PHASER는 기존의 Experience Replay (ER) 방식이 가진 한계, 즉 phase starvation 및 budget misallocation 문제를 해결합니다. 이 프레임워크는 각 서브 스킬에 대해 동등한 메모리 지원을 보장하고, 고위험의 역사적 phase를 동적으로 우선시하는 다중 모달 간섭 라우팅 전략을 사용합니다.

- **Technical Details**: PHASER는 Semi-Markov Decision Process (SMDP) 관점에서 VLA 지속 학습을 재해석하여, 일관된 트래젝토리 데이터와 균일한 프레임 수준 재생 간의 구조적 불일치를 드러냅니다. 그 과정에서 intra-task phase-centric capacity allocation과 inter-task multi-modal interference routing이라는 두 가지 데이터 측면 원칙을 적용하여 서브 스킬 간 버퍼 지원을 균등화하고, 잊혀지기 쉬운 역사적 phase에 대한 집중 리허설을 이루도록 합니다.

- **Performance Highlights**: PHASER는 여러 VLA 백본(OpenVLA-7B, QwenGR00T-3B, QwenOFT-3B)과 LIBERO 지속 학습 스위트를 통해 광범위하게 평가되었습니다. 이 프레임워크는 메모리 제약 조건에서 평균 성공률(ASR)을 최대 31% 향상시키고, LIBERO-Goal CL 설정에서 최종 ASR을 87.8% 달성했습니다.



### When Attention Collapses: Stage-Aware Visual Token Pruning from Structure to Semantics (https://arxiv.org/abs/2606.03569)
- **What's New**: 본 논문에서는 두 단계의 시각적 토큰 프루닝 프레임워크인 Structure-to-Semantics (STS)를 제안합니다. 기존의 시각적 프루닝 방법이 주요하게 사용하던 단일 메트릭은 토큰 집중과 같은 중요한 문제점을 동반했기 때문에, STS는 공간적 구조 다각성을 극대화하는 새로운 방법론을 모색했습니다. 첫 번째 단계는 반발 기반 샘플링 메커니즘을 사용하여 공간 및 구조적 다양성을 높이며, 두 번째 단계에서는 프롬프트와 무관한 토큰을 정교하게 필터링합니다.

- **Technical Details**: STS 프레임워크는 두 단계로 구성되어 있습니다. 첫 번째 단계에서는 시각적 토큰을 전하와 같은 상극의 원리를 통해 모델링하여, 유사한 의미를 가진 토큰들이 서로의 존재를 제한하도록 설계되었습니다. 두 번째 단계에서는 언어 모델의 중간 계층에서 작업과 무관한 토큰을 제거하는 방식으로, 단계에 따른 표현 역학을 고려하여 토큰을 정제합니다.

- **Performance Highlights**: 다양한 비전-언어 모델을 통한 광범위한 실험 결과, STS는 기존의 주의 기반 선택으로 인한 중복성을 줄이고, 보존된 시각 토큰의 구조적 다양성과 정밀한 작업 정렬을 향상시키는 데 성공했습니다. STS를 적용했을 때, 추론 효율성이 개선되었으며, 강력한 작업 성능 또한 유지되었습니다.



### Learned Non-Maximum Suppression for 3D Object Detection (https://arxiv.org/abs/2606.03568)
Comments:
          6 pages, accepted at IEEE Intelligent Vehicles Symposium (IV) 2026

- **What's New**: 이 연구에서는 LiDAR 기반 3D 객체 검출의 후처리 단계에서, 기존의 비최대 억제(non-maximum suppression, NMS)를 대체하는 두 가지 학습된 필터링 모듈을 도입합니다. D2D-Rescore는 변환기(transformer) 기반의 탐지 간(attention) 주의를 활용하고, GossipNet3D는 2D GossipNet 개념을 3D로 적응시킵니다. 이로 인해 제안된 방법들은 필터링의 신뢰성을 향상시키고, 소형 및 드문 클래스에서도 성능을 개선하는 결과를 보여줍니다.

- **Technical Details**: 제안된 방법론은 입력 임베딩, 컨텍스트(feature) 집합, 점수(score) 정제의 세 가지 주요 구성 요소로 이루어져 있습니다. 컨텍스트 집합 단계에서는 두 가지 상호 교환 가능한 변형인 GossipNet3D와 D2D-Rescore를 통해 집합된 정보를 활용하여 신뢰성 있는 점수를 생성합니다. D2D-Rescore는 서로 간의 관계를 모델링하여 자신을 주의(attention)로 활용하고, GossipNet3D는 지역적인 메시지 전달을 통해 3D 환경을 이해합니다.

- **Performance Highlights**: 제안된 모든 방법은 CircleNMS와 비교했을 때 평균 평균 정밀도(mean average precision, mAP), nuScenes 탐지 점수(nuScenes detection score, NDS), 그리고 진짜 긍정 품질(true positive quality)을 개선하였습니다. 특히 드물고 소형 클래스에서 두드러진 성과를 나타내며, 최소한의 계산 오버헤드로도 검출 성능을 높일 수 있습니다. 이 결과들은 기본 네트워크를 수정하지 않고 학습된 검출 레벨 필터링이 3D 탐지기의 신뢰성을 높일 수 있음을 입증합니다.



### Efficient Transformer-Based Localized Patch Sampling for Choroid Plexus Segmentation in Multiple Sclerosis (https://arxiv.org/abs/2606.03566)
- **What's New**: 이번 연구는 측면 뇌실 맥락막(choroid plexus, LVCP)을 다중 경과 면역 염증(multiple sclerosis, MS)의 주요 생체 지표로 활용하기 위한 새로운 자동 세그멘테이션 기법을 제안합니다. 기존의 수동 세그멘테이션 방식은 매우 고된 작업으로, 넓은 임상 시험(clinical trials)과 장기 평가(longitudinal assessments)에서 사용이 제한되었습니다. 이에 따라 SwinUNETR 구조를 활용하여 LVCP를 자동으로 세그멘테이션하는 파이프라인(pipeline)을 개발하였습니다.

- **Technical Details**: 이 연구는 두 개의 MS 주도 집단에서 수집된 3개의 데이터 세트(데이터 세트 1: n=177; 데이터 세트 2: n=177; 테스트 세트 확대: n=388)에 대해 3T MRI 스캔을 후향적으로 평가하였습니다. 제안된 방법은 32x32x32 복셀 패치(voxel patches)를 기반으로 훈련된 SwinUNETR 아키텍처를 사용하였으며, 3D UXNET 모델과 비교하여 성능을 평가하였습니다. 주요 평가지표는 Dice 유사성 계수(Dice Similarity Coefficient, DSC)였으며, 계산 요구사항(GFLOPs) 및 95번째 백분위수 하우스도르프 거리(Hausdorff Distance, HD95)도 보완적으로 고려하였습니다.

- **Performance Highlights**: 확대된 테스트 세트에서 SwinUNETR 모델은 MPRAGE와 FLAIR 결합 시 평균 DSC 0.868을 기록하여 UXNET보다 통계적으로 유의미한 성과를 보였습니다. 독립적인 FLAIR 입력에 제한했을 때도 높은 DSC 0.863를 유지했으며, UXNET의 공간적 위치화는 현저히 나빠졌습니다. 이 프레임워크는 계산 부하를 99% 줄여 22,080 GFLOPs 대신 91.8 GFLOPs만 필요로 합니다. 이러한 혁신적인 접근법은 LVCP 세그멘테이션에 있어 정확하고 안정적인 우수 대안을 제공하며 임상 및 연구 환경에서의 광범위한 구현을 위한 이상적인 방법론으로 자리잡을 것으로 기대됩니다.



### \textsc{CR-Seg}: Attention-Guided and CoT-Enhanced Coarse-to-Refined Reasoning Segmentation (https://arxiv.org/abs/2606.03564)
- **What's New**: 이번 연구에서는 복잡한 언어로 설명된 대상 객체를 분할하기 위해서 시각-텍스트적(visual-textual) 추론을 통한 Reasoning segmentation을 제안합니다. 기존의 방법들은 MLLM(다중모드 대규모 언어 모델)과 분할 모델을 연결하는 데 어려움을 겪었으며, 본 연구에서는 이러한 제한을 해결하기 위한 CR-Seg라는 새로운 프레임워크를 개발했습니다. CR-Seg는 Attention-Guide 및 CoT-Enhanced Coarse-to-Refined Reasoning Segmentation으로, 두 단계로 구성된 새로운 방법론을 제공합니다.

- **Technical Details**: CR-Seg는 이미지와 질문을 입력으로 받아 MLLM을 통해 구조화된 응답을 생성합니다. Attention Maps 및 Points를 추출하는 모듈인 EAP(Extract Attention Maps and Points)를 사용하여 코스(target localization)에 대한 정보 포인트를 선택하고, 이를 기반으로 SAM(Segment Anything Model)을 통해 최종 마스크를 정제합니다. 또한, 모델이 전역(scene context)에서 세부(local target) 정보를 점진적으로 추론하도록 안내하는 GLCoT(Global-to-Local Chain-of-Thought)를 도입하여 추론-답변 불일치성을 완화합니다.

- **Performance Highlights**: CR-Seg는 기존의 방법에 비해 강력한 성능을 보여주며, 정렬의 부담을 줄이는 데 도움을 줍니다. 다양한 Reasoning segmentation 벤치마크에서의 실험 결과는 CR-Seg의 효과성과 강인성을 입증하며, 이 모델이 전반적인 응답 의미를 보존하면서도 분할 품질을 개선할 수 있음을 보여줍니다. 추가로, FReasonSeg라는 보조 벤치마크를 통해 세부적인 범주 객체 간 차별성을 평가할 수 있도록 설계되었습니다.



### When Should the Teacher Move? Temporal Coupling and Stability in Self On-Policy Distillation (https://arxiv.org/abs/2606.03532)
- **What's New**: 본 논문은 self on-policy distillation의 교사(teacher) 업데이트 주기가 학습 안정성에 미치는 영향을 체계적으로 분석합니다. 기존 연구에서 간과된 temporal coupling 현상을 조명하여, 교사가 학생의 학습과 적절히 연결되어야 하며, 그 과정에서 isolation periods가 안정적인 학습에 필수적이라는 것을 보여줍니다.

- **Technical Details**: 연구진은 temporal KL structure, refresh shock, length-tail risk라는 진단 프레임워크를 도입하여 자율 정책 증류에서 고립 기간이 안정적인 학습을 가능하게 하는 기초 구조적 속임을 명확히 합니다. 이를 통해 state-oblivious collapse라는 장기적인 실패 모드를 밝혔으며, CGTR(Consolidation-Gated Teacher Refresh) 접근법을 제안하여, 각 리프레시가 보상 개선 및 길이-꼬리 안정성을 기반으로 문맥에 맞는 반응이 이루어지도록 합니다.

- **Performance Highlights**: CGTR 방법론은 각 과제(Chemistry, Biology, Physics, ToolUse)에서 최종 점수를 최고로 달성하며, 단일 매개변수 세트로 제로 collaps를 구현합니다. 이 방법은 각 과제의 학습 역학에 맞춰 리프레시 빈도를 조정하며, 학생의 실제 진전에 대한 응답으로만 교사 삽입이 발생하도록 보장합니다.



### High-Precision APT Malware Attribution with Out-of-Scope Resilienc (https://arxiv.org/abs/2606.03523)
- **What's New**: 본 논문에서는 고정밀 APT(Advanced Persistent Threat) 맬웨어 속성 기술 방법을 제안합니다. 이 방법은 명시적인 기권(abstention)이 가능한 순위 기반 이진 분류기를 이용하여, 훈련 데이터에서 미포함된 APT 그룹의 샘플을 효과적으로 처리합니다. 기존의 대부분의 기법은 제한된 수의 알려진 APT 그룹을 대상으로 한 폐쇄 집합(classifier) 분류기를 훈련하여 사용하지만, 본 연구는 이러한 방법의 한계를 극복하고자 합니다.

- **Technical Details**: 우리는 APT 그룹별로 두 개의 이진 분류기를 훈련시킴으로써 각각의 APT 그룹에 적합한 분류 기준을 설정합니다. 이 과정에서, 샘플이 충분한 속성 증거를 제공하지 않으면 기권하며, 이는 전문가의 검토가 필요한 샘플을 효과적으로 식별하는 데 기여합니다. 실험적으로, APT Malware 데이터셋과 더 큰 결합 데이터셋에서 본 방법을 평가하여 성과를 검증하였습니다.

- **Performance Highlights**: 우리가 제안한 방법은 이전에 발표된 동일한 데이터셋에서의 결과보다 더 높은 정밀도를 기록하며, 특히 87%의 테스트 샘플이 훈련에 포함되지 않은 60개의 APT 그룹에서 온 경우에도 94%의 기권 비율을 유지하였습니다. 이는 우리의 접근 방식이 실제 운영 환경에서보다 강인함을 보여주며, 고정밀 분류와 기권의 조합으로 효과적인 대응을 가능하게 합니다.



### Post-Hoc Robustness for Model-Based Reinforcement Learning (https://arxiv.org/abs/2606.03521)
- **What's New**: 이번 연구에서는 강화 학습(robust RL)의 현실적 적용 가능성을 높이기 위해, 적대적 환경에서 에이전트를 훈련시키기 위한 방법을 탐구합니다. 특히, 본 논문은 이미 학습된 모델 기반의 강화 학습(MBRL) 에이전트의 추론 단계에서 세밀한 조정을 통해 더 나은 강인성을 달성하는 방안을 제안합니다. 이 방법은 추가적인 신경망 훈련 없이도 강인한 정책 개선을 수행할 수 있도록 설계되었습니다.

- **Technical Details**: 강화 학습의 강인성을 높이기 위한 전반적인 접근 방식으로, 이 연구는 적대적 롤아웃(adversarial rollouts)을 사용한 모델 예측 제어(model-predictive control)를 통한 정책 개선을 제안합니다. 특히, 학습된 전환 모델을 활용하고, 예측된 상태를 최적화하여 결정을 내리는 **projected gradient descent (PGD)** 방법론을 차용합니다. 이 과정에서 불확실성 집합(uncertainty set)을 정의하며, 에이전트가 환경과 상호작용하지 않고 학습한 모델의 외적 사용에 따른 문제를 해결하기 위해 롤아웃 깊이를 조절합니다.

- **Performance Highlights**: 제안된 방법론은 Gymnasium MuJoCo 환경에서 성능을 평가하였고, 훈련되지 않은 모델 기반 정책 최적화(MBPO) 알고리즘과의 비교에서 현저한 강인성 향상을 보여주었습니다. 또한, 실험은 다양한 하드웨어에서 수행되어, 추론 시간에 맞춘 계산 비용 및 지속 시간을 평가하였습니다. 이러한 평가와 실험은 향후 연구 방향을 제시하는 데 기여할 것입니다.



### Scalable On-Hardware Training of Quantum Neural Networks and Application to Clinical Data Imputation (https://arxiv.org/abs/2606.03517)
Comments:
          13 pages, 9 figures

- **What's New**: 이번 연구에서는 양자 신경망(Quantum Neural Networks, QNN)의 훈련 프레임워크를 소개했습니다. 기존의 파라미터 이동(parameter shift) 방법에서 발생하는 기울기 추정 비용을 로그(logarithmic) 수준으로 줄여, 중간 규모의 양자 하드웨어에서 QNN 최적화가 가능하게 만들었습니다. 이 프레임워크는 구조화된 Butterfly 회로 아키텍처와 동시에 진행되는 층별 학습(layer-wise training), 병렬화된 파라미터 이동 규칙을 결합하여 효율성을 높였습니다.

- **Technical Details**: 프레임워크는 세 가지 주요 요소로 구성됩니다: (i) O(n log n) 개의 파라미터와 로그 깊이를 가진 구조화된 Butterfly 회로 아키텍처, (ii) 하드웨어 최적화를 잘 구조화된 작은 층에 한정하는 층별 훈련 전략, (iii) 각 Butterly 층 내에서의 통일적인 구조를 활용하여 상수 개수의 회로 실행에서 모든 기울기를 추출하는 병렬화된 파라미터 이동 규칙입니다. 이러한 접근 방식은 하드웨어 제약과 실제 상황에서도 성능을 유지할 수 있습니다.

- **Performance Highlights**: MIMIC-III 전자 건강 기록 데이터셋을 활용하여 임상 데이터 보간(clinical data imputation)의 정확성을 검증했습니다. 16큐빗(IonQ Forte Enterprise)에서 직접 하드웨어 훈련을 수행했으며, 32큐빗에서 추론을 실행하였습니다. 그 결과, QNN 모델은 다운스트림 환자 생존 예측에서 강력한 고전 신경망 기반 모델에 비견되거나 그 이상의 성능을 기록했으며, 각 실행에서의 분산은 줄어들어 안정성을 입증했습니다.



### SPADE: Sketch-guided Path Planning Augmented with Diffusion Experts (https://arxiv.org/abs/2606.03512)
- **What's New**: 이번 논문은 자율 모바일 로봇(AMRs)의 경로 계획을 개선하기 위한 새로운 프레임워크를 제시합니다. 기존의 방법들은 복잡한 보상 공학(reward engineering)이나 하드웨어 집약적인 솔루션에 의존했으나, 본 연구는 모방 학습(imitation learning)과 확산 기반 증강(diffusion-based augmentation)을 활용하여 보다 강력한 경로 계획 모델을 개발합니다. 이 연구는 신뢰성과 효율성을 높이기 위한 새로운 주석 도구(annotation tool)와 훈련 전략을 도입하여 성능을 향상시킵니다.

- **Technical Details**: 제안한 방법론은 모방 학습을 바탕으로 하며, 특히 Diffusion-guided Behavioral Cloning (DBC) 파이프라인을 통합하여 모델 일반화 능력을 향상시킵니다. 새로운 주석 도구는 ROS 2 기반으로 구축되어 데이터 접근성과 품질을 개선하고, 고용량의 확산 모델이 훈련 중에 경량화된 행동 복제 네트워크를 안내합니다. 이러한 접근은 데이터 증강의 여러 측면을 포함하여, 다양한 환경에서 조직의 요구를 충족하는 경로를 생성할 수 있도록 합니다.

- **Performance Highlights**: 제안된 방법은 기존의 최첨단기술(State-of-the-art)과 비교하여 39.1% 낮은 절대 위치 오차(Absolute Pose Error, APE)와 33.5% 낮은 프레체 거리(Fréchet Inception Distance, FID)를 달성합니다. 또한, 훈련 가능한 매개변수(Parameter)의 수가 93.8% 줄어들면서 빠르고 경량화된 모델을 유지할 수 있습니다. 최종적으로, 이 새로운 접근법은 실시간으로 동작할 수 있는 특성을 보존하면서도 확산 수준의 일반화(diffusion-level generalization)를 이뤄냅니다.



### BaltiVoice: A Speech Corpus and Fine-tuned Whisper ASR System for the Balti Languag (https://arxiv.org/abs/2606.03504)
Comments:
          5 pages, 4 figures, 4 tables. Code and data available at this https URL

- **What's New**: 이번 연구에서는 방글라데시에서 구사되는 틴버르티어(Balti) 언어를 위한 16.8시간 동시 읽기 음성 코퍼스(BaltiVoice)를 발표합니다. 기존에 공개된 ASR 자원이 없는 이 언어에 대해, 10,060개의 검증된 발화(utterance) 데이터를 제공합니다. 이 코퍼스는 Mozilla Common Voice 녹음을 바탕으로 하여 원주율 나스타일크(Nastaliq) 스크립트로 작성되었습니다.

- **Technical Details**: 이 연구에서는 BaltiVoice 코퍼스를 기반으로 OpenAI Whisper-small 모델을 파인튜닝(fine-tune) 하였습니다. 모델의 성능을 평가하기 위해 538개의 발화 데이터로 구성된 검증 세트를 사용하며, 여기서 얻어진 단어 오류율(Word Error Rate, WER)은 30.07%로, Balti 언어에 대한 기존 제로샷(zero-shot) 기준인 182.18%에서 크게 개선되었습니다.

- **Performance Highlights**: 파인튜닝된 모델과 데이터셋, 그리고 라이브 전사 데모는 HuggingFace에서 공개됩니다. 이 연구는 발리(Balti)에 대한 ASR 시스템의 발전에 중요한 첫걸음을 제시하며, 향후 이 언어의 음성 인식 기술 발전에 기여할 것으로 기대됩니다.



### Learn from Your Mistakes: Tree-like Self-Play for Secure Code LLMs (https://arxiv.org/abs/2606.03489)
Comments:
          18 pages, 3 figures

- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)의 코드 생성 능력을 보완하기 위해 Tree-like Self-Play (TSP)라는 새로운 프레임워크를 소개합니다. 기존의 보안 코드 생성 방식의 한계를 극복하고자, TSP는 코드 생성을 세밀한 결정 과정으로 재구성하였습니다. 특히, 이는 보안 취약점이 나타나는 특정 결정 지점을 인식하고 수정하는 데 중점을 두고 있습니다.

- **Technical Details**: TSP는 조건부 언어 모델링 과제로 안전한 코드 생성을 구현합니다. 이 과정에서 코드 생성 트리를 따라 진행하며, 각 분기 결정이 진행됩니다. 기존의 Supervised Fine-Tuning(SFT)이나 Reinforcement Learning(RL) 방법들은 보안에 있어 세밀함이 부족한 반면, TSP는 코드의 특정 결정을 차별화하여 피드백을 제공합니다.

- **Performance Highlights**: TSP 프레임워크를 기반으로 한 실험은 CodeLlama-7B 모델의 보안 통과율을 57.0%에서 75.8%로 크게 증가시켰습니다. 또한, TSP는 파라미터 전이 능력이 뛰어나며, 보지 못한 보안 취약점 카테고리에서 24.5%의 취약성을 줄이는 성과를 냈습니다. 이를 통해 TSP는 단순한 패치를 기억하는 것이 아니라, 언어에 구애받지 않는 보안 논리를 내재화하는 것으로 보입니다.



### NeuroArmor: Safe-Variant-Guided Representation Consistency for Selective Re-Anchoring in Jailbreak Defens (https://arxiv.org/abs/2606.03486)
Comments:
          16 pages, 4 figures, 17 tables. Submitted to ACL ARR

- **What's New**: NeuroArmor는 새로운 런타임 방어 시스템으로, 특히 프롬프트별 안전 단계를 적용하여 모델의 내부 상태에서 안전 기준을 유지하게 해주는 방식입니다. 이는 악의적인 공격을 탐지하는 데 도움이 되며, 경계선에 있는 부정적인 프롬프트에 대해서도 유용한 행동을 유지할 수 있도록 설계되었습니다. 결과적으로, NeuroArmor는 기존 방어 시스템보다 높은 안전성과 유용성을 제공합니다.

- **Technical Details**: NeuroArmor는 입력 프롬프트로부터 K개의 안전 변형(safe variants)을 생성합니다. 이 안전 변형은 각 프롬프트의 잠재적 위험을 감소시키기 위해 사용되며, 일관성 감지기(consistency detector)가 이들 변형과 입력 상태를 비교하여 안전 영역 내부에 있는지를 확인합니다. 필요한 경우, 악의적인 요청에 대해서는 거부(branch)로 보내거나, 유용한 해석으로 복구할 수 있도록 경로를 분기합니다.

- **Performance Highlights**: NeuroArmor는 Llama-3-8B-Instruct 모델을 통해 악의적인 공격 성공률(ASR)을 41.56%에서 1.57%로 감소시키며, 무해한 오류율(FPR) 또한 30.26%에서 22.05%로 줄였습니다. 남은 비차단 출력은 여전히 운영상 해롭지 않은 경향을 보이며, 외부 평가 및 수동 행동 평가에서도 우수한 성능을 나타냈습니다.



### Analyzing Stream Collapse in Hyper-Connections: From Diagnosis to Mitigation (https://arxiv.org/abs/2606.03483)
- **What's New**: 이번 연구에서는 Hyper-Connections (HC) 구조의 새로운 접근 방식을 제시합니다. 기존의 단일 Transformer residual stream 대신 다중 스트림 사용을 통해 permutation symmetry를 도입하고, 이러한 대칭이 실제로 어떻게 해결되는지를探 하고 있습니다. 연구는 streams가 균형 잡힌 방식으로 전문화되거나, 하나의 dominant stream을 형성하는지를 분석합니다.

- **Technical Details**: HC는 하나의 residual stream을 n개의 병렬 스트림으로 대체하며, 각 스트림 간의 token-dependent 연결성을 학습합니다. 연구팀은 이런 HC 방식에서 기저가 되는 residual mixing이 여러 스트림 간의 정보 교환을 제한하는 경향이 있음을 발견했습니다. Mixed signals와 해석 가능한 특징들이 동작적으로 한 스트림에 집중되며, 이는 명목상으로 다중 스트림인 연결이 과소 활용되는 경향을 나타냅니다.

- **Performance Highlights**: 자기 과시적인 HC 구조에서 학습된 stream scaling (LSS)의 도입이 dominant behavior를 줄이고 성능을 향상시킬 수 있음을 보여주었습니다. HClite/nanoGPT 모델에 대한 평가에서도, residual mixing 특성이 다양한 데이터셋에서 예측 불완전성을 개선하는 데 중요한 역할을 하는 것을 확인하였습니다. 연구자는 이 결과를 바탕으로 core HC operator 변경 없이 성능을 개선하는 새로운 방법론을 제시했습니다.



### Rethinking the Role of Tensor Decompositions in Post-Training LLM Compression (https://arxiv.org/abs/2606.03465)
- **What's New**: 본 논문에서는 대규모 언어 모델(LLM)의 배포를 위한 포스트 훈련 압축 기술에 중점을 두고 있습니다. 특히, 텐서 분해(tensor decomposition)가 Transformer 가중치 구조에 적합한 압축 방안으로 떠오르고 있지만, 기존 연구는 특정 환경에서만 평가되어 실제 대규모 배포 시 효과성에 대한 불확실성이 있었습니다. 이에 따라 연구 팀은 밀집(dense) 및 전문가 혼합(MoE) 아키텍처 전반에서 텐서 압축을 체계적으로 평가하여 성능을 정량적으로 분석하였습니다.

- **Technical Details**: LLM의 압축 기술으로는 주로 프루닝(pruning), 양자화(quantization), 그리고 지식 증류(knowledge distillation) 등이 있습니다. 여기서 텐서 분해는 다차원 가중치 텐서를 다루며, Tucker decomposition 및 Tensor Train (TT) 방식 등을 포함합니다. 그러나 기존 방법들이 대규모 적용 조건을 반영하지 않고 있기 때문에, 이 논문에서는 텐서 포맷이 동일한 압축 비율에서 매트릭스 방식보다 더 나은 성능을 발휘하지 못하는 이유를 분석하였습니다.

- **Performance Highlights**: 연구 결과, 텐서 포맷은 모든 설정에서 매트릭스 포맷보다 우수한 성능을 보이지 않았으며, 이는 텐서 분해 방식의 근본적인 한계와 관련이 있습니다. 또한, 논문에서는 프루비니우스 최적(Frobenius-optimal) 텐서 분해가 Transformer 가중치의 효과적인 압축기로 작동하지 못하는 이유를 세 가지 체계적 방해 요소로 formalize하였습니다. 이러한 발견은 LLM의 대규모 배포 시 압축 방법의 유용성을 명확히 하는 데 필요한 중요한 통찰을 제공합니다.



### Tonal parsimony in chord-sequence analysis: combining modulation cost and tonal vocabulary (https://arxiv.org/abs/2606.03459)
Comments:
          20 pages, 1 figure

- **What's New**: 본 논문에서는 코드 시퀀스에 대한 지역 조화 이론(local tonality)을 할당하는 방법을 연구합니다. 이는 화성 분석(harmonic analysis) 및 재하모니제이션(reharmonization), 즉재 작곡(generative composition)에서 유용합니다. 기존의 동적 프로그래밍(dynamic programming) 접근 방법은 변조 수를 최소화하는 데 중점을 두었지만, 불필요한 여러 조화 중심(tonal centers)을 도입할 수 있습니다.

- **Technical Details**: 논문은 조화적 분석을 위해 24개의 조화(tonality) 우주를 활용한 고유의 정밀 알고리즘을 제시합니다. 연구의 주요 초점은 모듈레이션(modulation) 수를 고려한 최적의 조화 분할(optimal tonality segmentation) 문제로, 이는 고전적인 프레임워크에서는 어려운 복합적 과제입니다. 최근 동적 프로그래밍 방법론들은 음악 시퀀스에 대한 최적화 방식으로 활용되고 있으며, 이 연구는 이를 기반으로 한 새로운 접근을 취하고 있습니다.

- **Performance Highlights**: 총 31,032개의 LMD 코드 시퀀스에서, 논문에서 제안한 조화 절약(tonal parsimony) 접근법은 55.8%의 경우에서 조화 어휘를 줄이며 전환 최적(optimum)을 유지합니다. 1,555개의 주석이 달린 재즈 스탠다드(jazz standards)에서 조화 절약은 호환 가능한 코드-스케일 일치를 95.6%로 개선하여 전문가의 분석에 더 현실적인 결과를 보여줍니다.



### FORGE: Multi-Agent Graduated Exploitation and Detection Engineering (https://arxiv.org/abs/2606.03453)
Comments:
          18 pages, 4 figures, 3 tables. Accepted at the AgentCy Workshop at the 21st International Conference on Availability, Reliability and Security (ARES 2026). Keywords: Vulnerability assessment, Multi-agent systems, Exploit generation, Detection engineering, Risk prioritization

- **What's New**: 본 논문은 FORGE라는 다중 에이전트 시스템을 소개하며, 이는 취약점 증명 생성, 취약점 우선순위 지정, 탐지 규칙 엔지니어링의 세 가지 연구 커뮤니티를 연결하는 역할을 합니다. 기존의 자동화된 exploit 생성 시스템은 성공/실패의 이진 결과만을 제공해 부분적 진전을 무시하고, 다른 커뮤니티에 유용한 신호를 생성하지 않습니다. FORGE는 CVE 메타데이터에서 시작하여, 공격 깊이에 따른 세분화된 exploitation을 측정함으로써 이들 커뮤니티 간의 정보를 교환합니다.

- **Technical Details**: FORGE는 CVE 메타데이터를 통해 목표 취약 애플리케이션을 생성하고, LLM 기반의 오라클을 통해 네 가지 수준(L0~L3)의 세부적인 exploitation를 실시합니다. 이 시스템은 Sigma 및 Snort 탐지 규칙을 OpenTelemetry 추적을 기반으로 생성하며, 취약점의 깊이를 평가하기 위한 구조화된 판별 신호를 제공합니다. 또한, 각 CVE 평가에서 수집된 지식을 축적하여 추후 공격 가능성을 높이고, 취약점 우선순위 지정 모델을 교정하는 데 기여합니다.

- **Performance Highlights**: 603개의 CVE 평가 결과, 총 67.8%의 end-to-end L1+ exploitation을 성공하며, CVE당 1.50달러의 비용이 소요되었습니다. 이 시스템은 8개의 프로그래밍 언어와 187개의 CWE 유형을 아우르며, 96.5%의 생성된 Snort 탐지 규칙은 합성된 양성 데이터 집합에 대해 0%의 잘못된 긍정률을 기록했습니다. 데이터는 EPSS 및 CVSS 점수와 관계 없이 68%에 가까운 exploitation 비율을 유지하며, 이는 패턴 수준의 도달 가능성이 메타데이터 기반의 우선순위 지정과는 독립적임을 나타냅니다.



### PRISM: Synergizing Vision Foundation Models via Self-organized Expert Specialization (https://arxiv.org/abs/2606.03444)
Comments:
          Accepted to ICML 2026

- **What's New**: PRISM(프리즘)이라는 새로운 이중 스트림 Mixture-of-Experts(MoE) 프레임워크를 제안하여 다양한 Vision Foundation Models(VFMs)의 보완적인 강점을 결합하여 단일 효율 모델을 효과적으로 통합합니다. 이 모델은 이론적으로 세 가지 핵심 요소를 기반으로 하여, 모듈화된 전문화(modular specialization)를 통해 VFMs의 성능을 최적화합니다. 이에 따라, PRISM은 기존의 정적인 구분 방식 이전에 자율적으로 전문화가 이루어지도록 하는 혁신적인 접근 방식을 제시합니다.

- **Technical Details**: PRISM은 두 단계의 패러다임을 적용하여, 첫째, 전문성을 해체(expertise deconstruction)하는 과정에서 교사를 조건으로 하는 라우터가 전문가들을 특화된 표현의 하위 공간으로 안내하고, 둘째, 동적 재조합(dynamic recomposition)을 통해 라우터가 하위 태스크에 맞춘 계산 경로를 조합하도록 학습하게 합니다. 이 시스템은 지식이 부분적으로 겹칠 때 전문 지식이 자연스럽게 발생하도록 하며, 생략적인 경량성과 재능이 있습니다.

- **Performance Highlights**: PASCAL-Context와 NYUD-v2 데이터셋에서의 실험을 통해, PRISM이 새로운 최첨단 기량을 수립하는 성과를 보여주었습니다. 이러한 결과는 희소하고 발생하는 전문화가 다양한 시각적 지식을 통합하는 효율적인 접근 방식임을 확인해줍니다. PRISM은 복잡한 시각적 내용을 효과적으로 다룰 수 있는 잠재력을 보여주며, 미래의 연구 개발을 위한 중요한 초석이 될 것입니다.



### A Hybrid Approach For Malware Classification Using Secondary Features Fusion (https://arxiv.org/abs/2606.03432)
- **What's New**: 본 논문에서는 악성 소프트웨어(malware) 탐지 및 분류를 자동화하는 방법을 제안합니다. 기존의 전통적인 탐지 방법이 악성 소프트웨어를 가족(family)으로 분류하지 못하는 문제를 해결하고자 합니다. 제안된 방법은 API 호출과 n-그램(n-grams) 같은 관련 특성(feature) 추출 후, 특성 융합(feature fusion)을 통해 악성 소프트웨어를 효과적으로 분류합니다.

- **Technical Details**: 제안된 방법은 특성 선택(method) 기법을 맞춤화하여 데이터에서 관련된 악성 소프트웨어 특성을 추출합니다. 그리고 예측 모델(pred predictive model)에서는 알고리즘 융합을 위한 투표 기반 접근 방식을 활용합니다. 실험은 Microsoft에서 제공한 데이터 세트를 사용하여 이진 및 다중 클래스(classification) 접근 방식을 적용하였습니다.

- **Performance Highlights**: 실험 결과는 제안된 방법의 효과성과 효율성을 강조하며, AUC(Area Under the Curve)는 0.989, 정확도(accuracy)는 99.72%, 로그 손실(log loss)은 0.01로 나타났습니다. 이러한 결과는 최신 기술(state of the art)과 비교하여 높은 성능을 입증합니다.



### FlowGuard: Flow Matching for Identity-Independent Detection of Data-Free Model Stealing Attacks on Energy System Intrusion Detection Systems (https://arxiv.org/abs/2606.03430)
- **What's New**: 이 논문에서는 에너지 인프라에서의 모델 추출 공격(Model Extraction Attacks, MEAs)에 대한 새로운 방어책인 FlowGuard를 제안합니다. FlowGuard는 쿼리의 흐름 매칭(Flow Matching)을 기반으로 하여, IDS 처리 전에 쿼리를 분류하여 오분배(out-of-distribution, OOD)로 인식합니다. 이는 기존의 모델 추출 방어 기법들이 갖고 있는 한계를 보완하여, 신원 정보에 의존하지 않고 지속적으로 안정적인 탐지율을 유지할 수 있습니다.

- **Technical Details**: 현재 논문에서 제안된 FlowGuard는 Continuous Normalizing Flow (CNF) 기술을 활용하여, 진짜 네트워크 트래픽 분포를 기반으로 학습한 모델을 사용하여 쿼리의 로그 우도(log likelihood)를 계산합니다. 데이터가 없는 모델 추출 공격에서 생성된 쿼리는 실제 네트워크 트래픽보다 차원 수가 적기 때문에, 이러한 쿼리에 대해서는 낮은 로그 우도를 기록하게 됩니다. 이것을 활용하여 FlowGuard가 OOD 신호로 인식합니다.

- **Performance Highlights**: 실험 결과 FlowGuard는 MAZE와 DisGUIDE 공격에 대해 PRADA 및 FDINet과 비교하여 단일 클라이언트 및 100-클라이언트 Sybil 공격 설정에서 안정적으로 방어 성능을 유지하였습니다. PRADA는 분포가 변경될 때 탐지율이 0%로 떨어지지만, FlowGuard는 신원 정보에 의존하지 않고도 탐지율을 안정적으로 보장합니다. 이러한 결과는 FlowGuard의 ID 독립적 방어 능력을 뒷받침합니다.



### PrimeSVT: An Automated Memory-aware Pruning Framework with Prioritized Compression Policy for Spiking Vision Transformers (https://arxiv.org/abs/2606.03428)
Comments:
          8 pages, 8 figures, 3 tables

- **What's New**: PrimeSVT는 사전 훈련된 SViT 모델에 대한 자동 메모리 인식 구조 프루닝을 수행하여 효율성을 극대화하는 새로운 프레임워크입니다. 기존의 수동 방식은 모델 압축을 위한 엄청난 시간과 자원을 요구하는 반면, PrimeSVT는 레이어의 크기에 따라 SViT를 정렬하여 우선 순위에 따라 압축을 진행합니다. 이를 통해 SViT의 다양한 모델을 보다 효율적으로 다룰 수 있습니다.

- **Technical Details**: PrimeSVT는 채널 별 필터 프루닝을 사용하여 L2-norm 값에 따라 중요하지 않은 가중치를 구조적으로 제거합니다. 각 레이어에 대해 사용자 정의 제약 조건(예: 허용되는 정확도 및 메모리 절약)을 고려하여 압축을 수행합니다. 이 구조적 프루닝 접근법은 기존의 잘못된 배치(weight mapping) 문제를 해결하고 하드웨어 최적화에 보다 적합합니다.

- **Performance Highlights**: 실험 결과는 PrimeSVT가 단일 샷 프루닝을 통해 메모리 사용량을 26.68% 줄이는 동시에 원래 모델에서 3% 이내의 정확도를 유지한다는 것을 보여줍니다. SViT 모델에서의 저장 공간 절약과 정확도 보존은 임베디드 시스템에서의 실행 가능성을 크게 높입니다. 이 연구는 SViT 모델의 설계 자동화를 가능하게 하여 보다 손쉬운 구현과 확장을 촉진합니다.



### Optimizing Explicit Unit-Distance Lower-Bound Certificates (https://arxiv.org/abs/2606.03419)
Comments:
          17 pages, 9 figures

- **What's New**: 2026년, Erdős의 단위 거리 추측에 대한 반증과 Sawin의 정량적 정제 결과에 따르면, $n$개의 평면 점들 사이의 단위 거리 최대 수 $u(n)$이 고정된 양의 $eta$에 대해 $n^{1+eta}$를 초과할 수 있다는 결과가 발표되었습니다. Sawin의 명시적 경계는 $n$이 임의로 커질 때 $n^{1.014}$이 넘는 단위 거리를 제공합니다.

- **Technical Details**: 이 보고서는 유한 매개변수 선택 문제를 비선형 정수 프로그래밍 문제의 변형으로 공식화하였으며, 오픈소스 Python 검증 파이프라인을 제안합니다. 이 파이프라인은 Sawin의 매개변수 선택을 재현하여 검증한 후 계산적으로 개선된 증명서에 적용되었습니다. 주요 계산 기여는 소수 집합 $T$와 $S_Q$, 정수 다항식 $k(p)$, 그리고 유리적으로 인코딩된 실수 매개변수 $R$에 대한 정수 최적화 및 검증 절차입니다.

- **Performance Highlights**: 네 가지 증명서 수준이 비교되었습니다: Sawin의 발표된 사례, 탐욕적 최적화 증명서, 맞춤형 정수 진화 전략 증명서 및 이산 재조합이 포함된 전략 증명서입니다. 결과적으로, 현재의 최상의 증명서는 $u(n)>n^{1.0152}$의 신중한 명제를 지지합니다.



### Causal Evidence of Stack Representations in Modeling Counter Languages Using Transformers (https://arxiv.org/abs/2606.03398)
Comments:
          8 pages, 8 figures

- **What's New**: 이번 논문에서는 언어 모델의 내부 메커니즘을 이해하기 위해 formal languages를 활용한 새로운 연구 결과를 제시합니다. 특히, stack representation이 모델 성능에 미치는 인과관계를 조사하였으며, 프로브(probe)를 통해 스택 깊이를 예측하는 정량적 분석을 수행합니다. 연구진은 stack representation이 단순히 학습된 것이 아니라, 모델의 계산에 필수적이라는 강력한 실증적 증거를 발견했습니다.

- **Technical Details**: 실험에서는 Dyck-1과 Shuffle-k 언어를 사용하였으며, 이들은 각각의 문자 집합을 기반으로 생성됩니다. 트랜스포머는 causal attention mask를 적용하여 다음 토큰 예측 작업을 수행하며, 정밀한 메트릭인 positional accuracy와 sequential accuracy를 정의하여 모델 성능을 평가합니다. 각 언어에서 stack representation이 존재하는지 여부를 검증하기 위해 linear classifier probe를 통해 심층 검사를 진행했습니다.

- **Performance Highlights**: 모델은 Shuffle-k의 모든 k 값에 대해 25번째 epoch까지 완벽한 검증 정확도를 달성하였습니다. 이 연구는 stack representation이 모델의 내부 계산에서 불가결한 요소임을 강조하며, 잘못된 방향을 제거하는 ablation 실험에서도 성능 저하가 발생함을 확인했습니다. 결과적으로, stack representation이 트랜스포머의 다음 토큰 예측에 중요한 역할을 한다는 점이 부각되었습니다.



### When Model Merging Breaks Routing: Training-Free Calibration for MoE (https://arxiv.org/abs/2606.03391)
- **What's New**: 이번 논문에서는 Mixture-of-Experts (MoE) 아키텍처에서 모델 병합이 초래하는 라우팅 파손(routing breakdown) 문제를 다룹니다. 기존의 병합 방법들이 MoE 구조의 비선형 특성 때문에 기능을 발휘하지 못하는 점을 지적하고, 이로 인해 전문가의 지정이 잘못될 수 있음을 강조합니다. 이를 해결하기 위해, 이 연구에서는 Hessian-Aware Router Calibration (HARC)이라는 새로운 무훈련(training-free) 접근법을 제안합니다.

- **Technical Details**: HARC는 비선형 소프트맥스 및 이산 Top-k 라우팅 메커니즘의 민감도를 고려하여 병합된 라우터의 출력 분포를 조정하는 방식으로 설계되었습니다. 이 방법은 매트릭스-프리(matrix-free) 공액 경량(conjugate gradient) 방법을 사용하여 효율적으로 해결할 수 있는 닫힌 형태의 솔루션을 허용합니다. HARC는 라우터 성능의 일관성을 보장하기 위해 2차 헤시안(Hessian) 정보를 활용하여 전문화된 지식을 제대로 전달할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, HARC는 수학 및 코드 생성을 다루는 작업에서 기존의 모델 병합 방법보다 일관되게 성능을 향상시키는 것으로 나타났습니다. 특히, 더 깊은 레이어로 갈수록 경로 오류가 불균형하게 증가하며, HARC는 이러한 경향을 완화하여 원본 라우팅 동작과의 정렬을 유지합니다. 또한, HARC는 약 40%의 캘리브레이션 샘플을 사용하여 최적 성능에 수렴할 수 있음을 보여줍니다.



### Grasp-Then-Plan with Failure Attribution: A Closed Two-Stage Framework for Precise and Generalizable Robotic Manipulation (https://arxiv.org/abs/2606.03385)
Comments:
          32 pages, project page: this https URL

- **What's New**: 이번 연구에서는 로봇 조작에서 그립(grasp) 선택과 동작 계획(motion planning) 간의 연결을 통해 실패 원인을 명확히 규명하여 최적화를 수행하는 GTP-FA(Grasp-Then-Plan with Failure Attribution) 프레임워크를 제안합니다. 이 프레임워크는 두 단계로 세분화되어 과거의 실패 데이터를 분석하고, 이를 통해 각 모듈의 최적화 압력을 적절히 전달합니다. 이렇게 함으로써 과거의 비효율적인 시도에서 벗어나, 더욱 효과적인 로봇 조작을 가능하게 합니다.

- **Technical Details**: GTP-FA는 그립-후-계획(grasp-then-plan) 접근 방식을 사용하며, 그립 선택이 전체 작업 성공에 미치는 영향을 명확히 연결하는 임무 중심의 닫힌 루프(closed-loop) 시스템으로 설계되었습니다. 이 시스템은 실패 귀속(failure attribution) 메커니즘을 도입하여, 각 실행 결과를 구조화된 실패 모드 분포로 매핑(mapping)하고 그립과 계획 간의 책임을 평가합니다. 장애나 변동을 고려하여 실패 귀속 메커니즘의 안정성을 개선하였고, 이러한 진단 신호를 통해 최적화 방향을 명확히 하였습니다.

- **Performance Highlights**: 이 프레임워크는 로봇 시뮬레이션 및 실제 로봇 실험을 통해 평가되었으며, RL(강화학습), IL(모방 학습), 확산 정책(diffusion-policy), VLA(비전-언어-액션) 기반 설정에서 비교 대상 학습기의 성능을 크게 향상시켰습니다. GTP-FA를 통해 달성된 작업 성공률은 기존 방법보다 현저히 높으며, 이는 명확한 실패 귀속과 스마트한 최적화를 통해 가능해졌습니다. 이로 인해 GTP-FA는 로봇 조작 분야에서 매우 세밀하고 효율적인 접근 방식을 제공함을 입증했습니다.



### Local Guidance, Global Impact: Gaussian-Reshaped Trust Region Unlocks Behavior Transitions (https://arxiv.org/abs/2606.03382)
Comments:
          21 pages

- **What's New**: 이 논문에서는 Proximal Policy Optimization (PPO)의 한계점을 다루고 있으며, 지속적이고 비정적인 환경에서의 성과 부족을 지적합니다. 특히, PPO는 방향성이 비효율적인 로컬 업데이트를 수행하여 새로운 행동 패턴으로의 전환을 방해하는 문제를 설명합니다. 이를 해결하기 위해 Gaussian Trust Region Policy Optimization (GTR)을 제안하며, 이는 신뢰 영역을 Gaussian 커널을 사용하여 재구성합니다.

- **Technical Details**: GTR는 로컬 업데이트의 안정성을 높이고, 지속적인 높은 이점 업데이트를 허용하면서도 움직임이 제한되도록 설계되었습니다. 이 방법은 Mixture Gaussian Anchor를 도입하여 최근 정책 경로에 적응하는 방식으로 변동성을 줄입니다. GTR는 아키텍처에 독립적이며 MLPs, RNNs, SimBa 및 Transformer 기반 정책에 대해서도 효과적으로 작동합니다.

- **Performance Highlights**: GTR의 성능은 여러 분야에서 우수하며, 로봇 제어, 오픈 월드 탐색 및 언어 모델 후 훈련에서의 평가를 통해 입증되었습니다. 이 방법은 기존의 PPO 방법보다 더 나은 적응성을 보여 주며, 복잡한 비정적인 환경에서의 강화 학습에 대한 기하학적으로 인식된 신뢰 영역 설계의 가능성을 강조합니다.



### AI Model Extraction Attacks: Bypassing Single-Client Assumptions in Defenses (https://arxiv.org/abs/2606.03381)
- **What's New**: 이 논문은 군사 Command and Control (C2) 시스템과 중요한 인프라에 배치된 인공지능(AI) 모델의 보호 필요성을 강조하며, 모델 추출 공격(Model Extraction Attacks, MEAs)의 위험성을 제기합니다. 현재 방어 전략은 Single Client Assumption (SCA)에 의존하고 있으나, 이는 고도로 협력하는 위협 행위자들이 존재하는 환경에서는 무효하다는 것을 체계적으로 입증합니다. 논문에서는 CerberusAI라는 모듈식 오픈 소스 프레임워크를 소개하여 배포된 공격 시나리오를 시뮬레이션하고, 안정성을 향상시킬 방법을 모색합니다.

- **Technical Details**: MEAs는 악의적인 네트워크 노드가 AI 모델을 복제하려고 시도하는 공격 방식으로, 이를 통해 민감한 정보를 탈취하고 다른 공격을 가능하게 합니다. 기존 방어 메커니즘인 PRADA, QUEEN, Model-Guardian은 개별 클라이언트의 비정상 요청 패턴을 탐지하는 데 기반하나, 거대한 네트워크 환경에서는 협조적인 공격 방식에 취약합니다. 따라서 연구자들은 SCA의 기본 가정에 대한 근본적인 결함을 논의하며, 분산 및 적응형 MEA 위협을 평가할 수 있는 새로운 오픈 소스 프레임워크를 제시합니다.

- **Performance Highlights**: 실험 결과, 기존의 방어 메커니즘이 단순한 라운드 로빈 쿼리 분배 전략으로 우회될 수 있음을 보여주며, 이로 인해 탐지 성능이 크게 저하됨을 나타냅니다. 또한, 글로벌 집계 접근 방식조차도 적응형 트래픽 혼합을 통해 운영적으로 무의미해질 수 있음을 입증합니다. 이러한 결과는 모델 추출 공격에 대한 방어 아키텍처의 패러다임 전환 필요성을 강조하며, AI가 핵심 의사 결정과 사이버 방어에서 지원을 할 수 있도록 신뢰할 수 있는 AI 구성 요소를 구현하는 데 필수적입니다.



### P\textsuperscript{2}-DPO: Grounding Hallucination in Perceptual Processing via Calibration Direct Preference Optimization (https://arxiv.org/abs/2606.03376)
- **What's New**: 이번 연구에서는 Large Vision-Language Models (LVLMs)에서 발생하는 환각(hallucination) 문제를 해결하기 위한 새로운 접근법인 Perceptual Processing Direct Preference Optimization (P²-DPO)를 제안합니다. P²-DPO는 모델이 자체적으로 preference data를 생성하고 학습할 수 있도록 하여 시각적 병목 현상(perceptual bottleneck) 문제를 직접적으로 해결합니다. 우리의 방법은 시각적 신호와 텍스트 간의 인과적 생성 관리를 강화하기 위한 Calibration Loss를 포함합니다.

- **Technical Details**: P²-DPO는 두 가지 새로운 preference pair를 도입하여, 각기 다른 시각적 결함을 해결합니다. 첫 번째는 Focus-and-Enhance Preference Pair로, 이는 세밀한 세부 사항의 향상된 출력과 열화된 출력 간의 대조를 통해 Perceptual Bottleneck을 극복합니다. 두 번째는 Visual Robustness Preference Pair로, 정확한 정보와 노이즈 신호 간의 출력을 대비하여 시각적 강인성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, P²-DPO는 같은 양의 학습 데이터와 비용을 사용했음에도 불구하고, 강력한 기준선을 초과하는 성능을 보여줍니다. 특히 Attention Region Fidelity (ARF) 평가에서는 높은 정확도를 기록하였으며, 이미지 열화 시나리오에서도 시각적 강인성을 크게 개선함을 입증하였습니다.



### The Unsampled Truth: Psychometrics in SLMs Measure Prompt Artifacts, Not Psychological Constructs (https://arxiv.org/abs/2606.03357)
Comments:
          10 pages, 5 figures, 3 tables

- **What's New**: 본 논문에서는 SLM(유연한 언어 모델)의 출력을 심리적 테스트 결과로 고려하는 기존의 방법론이 허약하다는 점을 지적합니다. 연구자들은 특정 성격과 표준화된 설문지를 제공하여 인공지능이 모사된 인간 태도와 행동을 추출한다고 가정하고 있지만, 이 가정이 과연 타당한지를 평가합니다. 주요 발견은 모델들이 프롬프트 준수(prompt compliance)를 반영하고 있으며, 모델의 출력이 심리적 특성의 신뢰할 수 있는 지표가 아닐 수 있음을 보여줍니다.

- **Technical Details**: 연구에서 13개의 오픈-웨이트 모델(0.6B에서 14B 파라미터)을 사용하여 프롬프트의 변화를 체계적으로 분석합니다. 이를 통해 모델의 출력에서 의미적 신호(semantic signal)와 아티팩트(artifact) 변수를 구분하고 이들의 영향을 평가하는 진단 도구를 제공합니다. 연구는 특성 기준으로 Big Five Inventory(BFI)와 Short Dark Triad(SD3)를 사용하며, 다양한 프롬프트 변형을 통해 아티팩트 변동성을 정량화합니다.

- **Performance Highlights**: 연구 결과, APWD(Average Pairwise Wasserstein Distance) 점수가 0.5에서 1.0 범위에 분포하고 있는 것으로 나타나, 이 범위는 측정 오류를 시사합니다. 또한 대체로 모든 평가 아키텍처에서 과제 지침(task instructions)과 선택 옵션 기호(option symbols)가 변동성을 높이는 주요 요소로 작용합니다. 최종적으로, 프롬프트의 아티팩트가 의미적 신호를 압도하는 경우가 많아, SLM의 심리적 평가에서의 유용성을 제한하는 것으로 결론지을 수 있습니다.



### SynCred-Bench: Benchmarking Synthetic Credibility in AI-Generated Visual Misinformation (https://arxiv.org/abs/2606.03348)
- **What's New**: 최근의 생성 모델들은 현실적인 텍스트와 레이아웃을 포함한 시각적 아티팩트를 생성할 수 있는 능력을 갖춰, 새로운 유형의 잘못된 정보 위협인 합성 신뢰성을 제시합니다. 본 연구에서는 SYNCRED-Bench라는 600개의 AI 생성 잘못된 정보 이미지를 포함하는 벤치마크를 소개하며, 이는 여섯 가지 신뢰할 수 있는 형식(categories)과 일곱 가지 세부 유통 스타일을 반영하고 있습니다. 또한, FP450라는 진짜 이미지 부정 집합을 구축하여 잘못된 긍정(false positives)을 측정합니다.

- **Technical Details**: 합성 신뢰성을 탐지하기 위해, 본 연구는 시각적 잘못된 정보의 평가를 위한 SynCred-Bench라는 포괄적인 벤치마크를 구성하였습니다. 이 벤치마크는 여섯 가지 신뢰 가능한 형식 카테고리와 네 가지 조잡한 및 일곱 가지 세부 신뢰 유통 스타일을 다루며, 총 600개의 AI 생성 샘플을 포함하고 있습니다. 실험 결과, 대부분의 다중 모드 대형 언어 모델(MLLMs) 및 AIGC 탐지기는 여전히 신뢰성이 낮음을 보여줍니다.

- **Performance Highlights**: 15개의 MLLMs을 평가한 결과, 5%의 잘못된 긍정 비율(FPR)을 제한했을 때, 평균 진짜 긍정 비율(TPR)은 10.5%에 불과했습니다. AIGC 탐지기는 평균 TPR이 5% 미만인 반면, 상업 API는 57.6%의 정확도를 기록했습니다. 인간 평론가들조차 합성 신뢰성을 식별하는 데 어려움을 겪었으며, 투표의 대부분이 63.0%의 TPR로 나타났습니다.



### AugMask: Training Diffusion Models on Incomplete Tabular Data via Stochastic Augmentation and Masking (https://arxiv.org/abs/2606.03347)
- **What's New**: 신규 AI 모델 AugMask가 제안되었습니다. 기존 Score-based diffusion model은 완전한 입력을 가정하지만, 실제 데이터는 결측치가 자주 발생하는 문제점을 해결하고자 합니다. AugMask는 확률적 보강을 통해 결측치가 있는 데이터를 효과적으로 처리할 수 있는 프레임워크로, 이는 모델의 훈련 방식을 개선할 수 있는 새로운 접근법을 나타냅니다.

- **Technical Details**: AugMask는 두 가지 주요 기능으로 구성됩니다: 조건부 확률적 보강(conditional stochastic augmentation)과 관찰된 좌표만을 통한 노이즈 제거(denoising) 비유도입니다. 이 모델은 입력에서도 불완전성을 다루기 위해 스토캐스틱 보강을 통해 가능한 문맥 의존성을 구축합니다. 새로운 방법론은 Rao-Blackwellized objective를 활용하여 불확실성을 효과적으로 다루고, 결측치를 통한 과도한 의존성을 억제합니다.

- **Performance Highlights**: 다양한 데이터셋과 결측 패턴에 걸쳐 AugMask는 기존의 결측 인식 모델보다 우수한 성능을 보여줍니다. 전통적인 방법론의 한계를 극복하며, AugMask는 점진적으로 훈련된 모델이 결측치에 대한 복원 과정에서 더욱 효과적으로 활용될 수 있도록 합니다. 최종적으로, AugMask는 디자인 변경 없이 표준 score-based diffusion 모델에 통합 가능하여 적용성을 높입니다.



### Evaluating LLMs' Effectiveness on Real-World Consumer Device Repair Questions (https://arxiv.org/abs/2606.03331)
- **What's New**: 이 논문은 소비자 기기 수리에서 대형 언어 모델(LLMs)의 잠재력을 탐구한 첫 번째 연구입니다. 991개의 실제 수리 질문을 Reddit 커뮤니티에서 수집하여, 각 질문에 대해 전문가가 작성한 참고 솔루션을 제공합니다. 이 데이터셋은 방글라어로도 번역되어 다양한 언어에서 LLM의 성능을 평가하는 새로운 기준을 설정하고 있습니다. 또한, 전화 수리, 컴퓨터 수리, 데이터 복구라는 세 가지 분야에서 LLM들의 성능을 비교하고, 안전성과 실용성을 고려한 평가 기준을 제시합니다.

- **Technical Details**: 본 연구는 GPT-5.4, Claude 4.6, Gemini 3.1 등 여섯 가지 최신 LLM을 평가하였습니다. LLM의 출력은 정확성(correctness), 완전성(completeness), 실용성(practicality), 안전성(safety) 네 가지 기준에 따라 평가되었습니다. 데이터를 수집한 질문들은 복잡하고 불완전한 문제 설명을 포함하고 있어, LLM이 유용한 진단 및 수리 절차를 생성하는 데 도전이 됩니다. 본 연구는 LLM이 고위험 수리 업무에서 신뢰성 있게 작동하지 않음을 보여줍니다.

- **Performance Highlights**: 연구에서 LLM들은 실용적인 수리 지원을 제공할 수 있지만, 실제 고위험 수리 작업에는 여전히 신뢰할 수 없음을 보여줍니다. 특히, 전화 수리는 가장 어려운 도메인으로 나타났으며, 모든 모델이 보드 수준 진단, 수리 우선순위 및 안전한 회복 절차에서 상당한 오류를 범했습니다. 또한, 방글라어 응답은 영문 응답보다 일관되게 낮은 성능을 보였으며, 이는 다국어 수리 지원에서의 어려움이 큼을 나타냅니다.



### FLIPS: Instance-Fingerprinting for LLMs via Pseudo-random Sequences (https://arxiv.org/abs/2606.03330)
Comments:
          20 pages, 20 figures, 3 tables. 43rd International Conference on Machine Learning (ICML 2026)

- **What's New**: 이번 논문에서는 인스턴스 레벨 핑거프린팅(Instance-level Fingerprinting)이라는 새로운 패러다임을 제안하는데, 이는 규제자(제어 기관)들이 동일한 대형 언어 모델(Large Language Model)의 다양한 구성(configuration)을 구별할 수 있도록 돕습니다. 기존의 LLM 식별 기술은 지식 재산 보호(intellectual property protection)에 초점을 맞추었으나, 이번 연구는 모델의 실제 행동을 평가하는 안전성과 규제 준수(compliance) 평가에 중점을 두고 있습니다. 연구진은 FLIPS라는 방법을 통해 총 237개의 모델 인스턴스에서 96%의 식별 정확도를 달성하였으나, 이전의 LLMmap 기법은 35%에 불과하여 FLIPS의 유용성을 입증하고 있습니다.

- **Technical Details**: FLIPS 방식은 생성된 이진 랜덤 시퀀스의 편향(bias)을 이용하여 모델 인스턴스를 효율적으로 식별합니다. 연구에서는 25개의 LLM에서 파생된 인스턴스 237개로 구성된 벤치마크를 통해 이 방법의 유효성을 입증하며, 상대적으로 적은 수의 쿼리(질의)로도 높은 정확도를 달성합니다. 특히, FLIPS는 모델의 지식 재산 보호 기술들이 간과하는 인스턴스 레벨 파라미터(instance-level parameters)의 변화를 감지할 수 있도록 설계되었습니다. 이러한 방법론은 추출 및 검증 검증 쿼리의 효율성을 높이고 경제성을 제공합니다.

- **Performance Highlights**: FLIPS 방법은 40개의 추출 쿼리와 8개의 검증 쿼리를 통해 각각 96% 및 90%의 식별 정확도를 확인하였습니다. 기존의 LLMmap 기술보다 월등히 높은 성과를 보이며, 규제 기관이 시간과 비용을 절약하면서도 LLM의 실제 행동을 정확하게 평가할 수 있도록 지원합니다. 이로 인해, 인공지능의 안전성 및 규제 준수를 위한 필수적인 도구로 자리잡을 것으로 기대하고 있습니다.



### Calibration Data Trade-offs Across Capability Dimensions: Why Multi-Source Mixing Matters for High-Sparsity LLM Pruning (https://arxiv.org/abs/2606.03328)
- **What's New**: 이 연구는 Post-training pruning을 통해 대형 언어 모델을 데이터 없이 압축하는 방법을 다룹니다. 기존의 연구 결과는 월드와이드 웹의 웹 텍스트가 일반적인 능력을 유지하는 반면, 수학 및 코드 능력이 크게 손실된다는 발견을 통해 새로운 관점을 제시합니다. 연구자들은 여러 출처의 교차 검증을 통해 단일 출처에만 의존하는 것이 효과적이지 않음을 강조하고, IGSP(Information-Guided Self-Calibration Protocol)를 제안합니다.

- **Technical Details**: 저자들은 15개의 서로 다른 교정 출처에 대한 Spearman 상관관계를 분석합니다. 이 분석을 통해 높은 perplexity를 가진 자료가 일반적인 능력 보존에는 긍정적으로 작용하지만, 수학과 코드 능력에는 부정적으로 작용하는 상반된 무역관계를 발견하였습니다. IGSP는 이러한 구조적 문제를 해결하기 위해 다수의 출처를 혼합하는 방식을 사용하며, 4-gram 집합을 최소화하고 능력 차원 간 perplexity 균형을 유지합니다.

- **Performance Highlights**: LLaMA-3.1-8B 모델을 사용하여 SparseGPT 환경에서 60% 희소성으로 실험을 수행한 결과, IGSP를 이용한 다수 출처 혼합이 58.8%의 전체 유지율을 달성하였습니다. 이는 최고의 단일 출처인 MetaMath보다 8.8%, 기본 C4보다 18.8% 증가한 수치입니다. 이로 인해 IGSP는 자가 생성 방법 중에서 최신 상태를 유지하며, 실제 다수 출처 데이터와의 품질 차이도 감지되었습니다.



### dstack-capsule: Pod-Level Remote Attestation for Confidential Workloads on Kubernetes (https://arxiv.org/abs/2606.03323)
- **What's New**: 이 논문은 Kubernetes 플랫폼에서 Pod 단위의 원격 증명(remote attestation)을 가능하게 하는 dstack-capsule을 소개합니다. 기존의 Confidential Containers (CoCo) 모델의 한계를 극복하여 여러 Pods가 단일 Confidential VM을 공유하면서도 각 Pod의 독립적인 하드웨어 기반 신원 증명을 유지할 수 있는 방법을 제시합니다. 두 단계의 증명 아키텍처를 통해 Pod의 동적 신원을 TDX Quote의 report_data 필드에 포함시키는 혁신적인 접근을 포함하고 있습니다.

- **Technical Details**: dstack-capsule은 두 층의 증명 구조를 활용하여 정적 플랫폼 측정(static platform measurements)과 동적 Pod 신원(dynamically Pod identities)을 분리합니다. RTMR을 통해 고정된 OS 및 시스템 소프트웨어 스택을 증명하고, 각 Pod의 ID는 TDX Quote 내 report_data 필드에 명시됩니다. 또한 Pod 사양의 다이제스트를 하드웨어 서명된 Quotes에 바인딩하는 새로운 프로토콜, 원자적인 상태 전환을 이루는 privilege fuse 메커니즘, 그리고 다층 샌드박스를 포함한 보안 강화 기능이 포함되어 있습니다.

- **Performance Highlights**: dstack-capsule은 Pod 수준 인증을 달성하면서 VM 수준 격리의 리소스 오버헤드를 피할 수 있는 성능 이점을 보입니다. 구현 및 평가 결과, dstack-capsule은 CoCo 모델에 비해 더 많은 Pods를 효과적으로 지원하며, 자원을 더욱 효율적으로 사용할 수 있음을 입증하였습니다. 전체적인 보안 속성과 인증의 정확성을 평가한 결과, 기존 솔루션에 비해 성능을 크게 향상시키는 성과를 달성하였습니다.



### Multi-Modal Graph Neural Network with Transformer-Guided Adaptive Diffusion for Preclinical Alzheimer Classification (https://arxiv.org/abs/2606.03322)
Comments:
          10 pages, Accepted to MICCAI 2024

- **What's New**: 이 논문에서는 신경퇴행성 질환의 진단을 위한 새로운 그래프 신경망(GNN) 프레임워크를 제안합니다. 이 프레임워크는 정보를 각 노드 전파하여 단기 및 장기 특성을 집계할 수 있도록 설계되었습니다. 특히 알츠하이머 병(AD)의 조기 증상을 분류하는 데 있어 우수한 성능을 보여주며, 이 과정에서 중요 지역(ROI)을 식별하는 데 도움을 줍니다.

- **Technical Details**: 제안된 GNN 및 Transformer 주도 적응 확산(GTAD) 아키텍처는 노드 중심의 확산 커널을 학습합니다. 이 아키텍처는 각 ROI별 이미지 모달리티의 효과적인 표현을 생성하며, 다중 헤드 주의(multi-head attention)를 통해 전역적 특징을 반영할 수 있습니다. 이 방법론은 구조적 뇌 네트워크와 기능적 이미징의 조합을 통해 AD의 예측 및 해석을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, GTAD 모델은 알츠하이머 병의 조기 단계에서 중요한 결과를 도출하였으며, 기존의 최첨단 방법들에 비해 우수한 분류 성능을 발휘했습니다. 이 연구는 다양한 이미징 바이오마커를 고려하여 뇌 네트워크의 효과적인 해석을 가능하게 하며, 조기 진단과 예방을 위한 잠재력을 보여줍니다.



### RobotValues: Evaluating Household Robots When Human Values Conflic (https://arxiv.org/abs/2606.03312)
- **What's New**: 이 논문은 가정용 로봇의 가치 선호를 평가하는 RobotValues라는 벤치마크를 소개합니다. 이 벤치마크는 10,000개의 가치 충돌 시나리오를 포함하며, 로봇이 다양한 인간 가치를 우선시하는 여러 행동 중에서 선택하는 능력을 평가합니다. 최신 연구들이 임무 성공에 중점을 두었다면, 이 벤치마크는 인간의 자율성, 효율성, 사회적 수용성 같은 값이 충돌하는 상황에서 로봇의 결정을 평가하는 새로운 기준을 제시합니다.

- **Technical Details**: RobotValues는 LLM (Large Language Model) 지원 시나리오 생성, 이해관계자 기반 가치 추출, 이미지 생성 및 자동 품질 제어를 통해 구축되었습니다. 각 사례는 현실적인 가정 이미지와 함께 로봇 행동 선택의 맥락을 제공하며, 이러한 행동은 각기 다른 인간 가치를 우선시합니다. 연구자들은 또한 로봇의 선택을 평가하기 위한 프로토콜을 개발하였으며, 이 과정에서 가치 조건화가 필요한 상황을 설정하여 로봇이 선호 하는 행동을 넘어 선택하도록 유도합니다.

- **Performance Highlights**: 연구 결과에 따르면, 여러 모델이 안전과 수용성을 우선시하는 가치 선호를 공유하는 것으로 나타났습니다. 그러나 특정 가치를 우선시하라는 지시를 받았을 때, 모델은 종종 자신의 기본 선호를 우선시하여 80%의 확률로 잘못된 행동을 선택하는 경향이 있었습니다. 이러한 결과는 가정용 로봇의 평가는 단순한 작업 완료 및 안전 준수뿐만 아니라, 인간 가치가 충돌할 때 로봇이 어떻게 행동을 선택하는지도 평가해야 함을 시사합니다.



### Learning Multi-Scale Hypergraph for High-Order Brain Connectivity Analysis (https://arxiv.org/abs/2606.03310)
Comments:
          24 pages, Accepted to ICML 2026

- **What's New**: 새롭게 제안된 MuHL(Multi-scale Hyperedge Learning) 프레임워크는 여러 뇌 영역 간의 고차원 상호작용을 동적으로 학습하고, 이를 통해 신경퇴행성 질환 분류 성능을 개선하는데 주목합니다. 기존의 그래프 기반 모델들이 주로 쌍별 상호작용에 집중했던 점을 보완하여, 고차원 관계를 명시적으로 모델링할 수 있는 하이퍼그래프를 도입합니다. 이를 통해 질병 진행과 관련된 중요한 영역과 그 상호작용을 효과적으로 밝혀내고자 합니다.

- **Technical Details**: MuHL 프레임워크는 계층적인 노드 특성을 구성하고, 다중 해상도의 그래프 신호에 따른 연속적인 하이퍼엣지(hyperedge) 구성을 통해 고차원 상호작용을 학습합니다. 이 과정에서 그래프 웨이브렛 관점의 다중 해상도 표현을 통해 노드 특성을 조정합니다. 또한, MuHL은 하이퍼그래프 내의 하이퍼엣지를 직접적으로 학습하며, 기존의 정의된 하이퍼엣지에 의존하지 않습니다.

- **Performance Highlights**: MuHL은 Alzheimer’s Disease Neuroimaging Initiative(ADNI)와 Parkinson’s Progression Markers Initiative(PPMI)라는 두 가지 독립적인 기준에서 광범위한 실험을 통해 성능을 검증 받았습니다. 실험 결과 MuHL은 다양한 질병 단계에서 질병 분류 성능을 일관되게 개선했으며, 배운 하이퍼엣지를 사용해 뇌 네트워크 분석에서 중요한 생물학적 해석 가능성을 제공하는 강력한 도구로서의 잠재력을 강조합니다.



### Generalizing Graph Foundation Models via Hyperbolic Retrieval-Augmented Generation (https://arxiv.org/abs/2606.03307)
Comments:
          Accepted by KDD2026

- **What's New**: 이번 연구는 그래프 기초 모델(Graph Foundation Models, GFMs)의 일반화 능력을 향상시키기 위한 새로운 프레임워크인 Hyperbolic Retrieval-Augmented Generation (HyRAG)을 제안합니다. 기존 Euclidean 공간에서 작동하는 RAG 프레임워크의 기하학적 한계를 해결하고, 외부 지식을 효과적으로 통합하여 성능을 개선합니다. 특히, 제안된 하이퍼볼릭 지식 인덱싱 모듈은 트리 구조의 외부 지식을 하이퍼볼릭 공간에 모델링하여 의미의 정밀도를 유지합니다.

- **Technical Details**: HyRAG 프레임워크는 하이퍼볼릭 공간의 기하학적 특성을 활용하여 기존 RAG 시스템의 두 가지 주요 문제인 의미의 정밀성 손실 및 허브 문제(hubness)를 해결합니다. 하이퍼볼릭 지식 인덱싱 모듈은 거리 기반 목적과 각도 제약을 포함하여 비지도 최적화 방식으로 하이퍼볼릭 의미 인코더를 사전 훈련합니다. 이를 통해 다양한 세밀한 정보를 캡처할 수 있는 다중 정밀도 검색(Multi-granularity Retrieval) 전략을 개발하여, 전역 의미 고정 요소와 지역 의미 뉘앙스를 효과적으로 제공합니다.

- **Performance Highlights**: 다양한 그래프 벤치마크에서 HyRAG의 성능을 평가한 결과, 제안된 방법이 zero-shot 설정에서 GFMs의 추론 정확도를 크게 개선했습니다. 실험 결과, 새로운 구조적 융합 메커니즘에 의해 그래프 작업에서 지식 통합이 효과적으로 이루어졌으며, 이는 기존 모델에 비해 우수한 일반화 성능을 입증합니다.



### Message Tuning Outshines Graph Prompt Tuning: A Prismatic Space Perspectiv (https://arxiv.org/abs/2606.03290)
Comments:
          Accepted by ICML 2026

- **What's New**: 이번 논문에서는 Graph Foundation Models (GFMs)의 새로운 적응 방법론인 Prismatic Space Theory (PS-Theory)를 제안하여 그래프 프롬프트 조정의 적응 능력을 정량화하는 새로운 수학적 틀을 제공합니다. 이를 통해 그래프 프롬프트 조정의 한계 및 MTG(Message Tuning for GFMs)의 장점을 밝혔습니다.

- **Technical Details**: MTG는 각 GNN(Grapg Neural Network) 레이어에 학습 가능한 메시지 프로토타입을 주입하여 메시지 융합을 동적으로 안내하는 경량 접근 방식을 사용합니다. PS-Theory를 통해 MTG의 적응 능력이 그래프 프롬프트 조정의 이론적 상한을 초과할 수 있음을 증명하였으며, 이는 그래프 입력 공간에서 작업을 수행하는 기존의 방법의 한계를 드러냅니다.

- **Performance Highlights**: MTG는 다양한 벤치마크 데이터셋을 통해 그래프 프롬프트 기준보다 일관되게 우수한 성능을 보이며, 이는 MTG의 향상된 적응 능력을 뒷받침하는 강력한 실증적 결과입니다. 실험을 통해 민감도 분석과 계산 효율성에서도 MTG의 강건성과 효율성을 입증하였습니다.



### AI-Generated Traces for Novice Programmers: Learning Effects and Learner Differences in a Multi-Institutional Study (https://arxiv.org/abs/2606.03288)
- **What's New**: 이 연구에서는 Generated Animated Traces (GATs)라는 새로운 AI 기반 애니메이션을 제안합니다. 이 애니메이션은 코드 조각, 메모리 다이어그램, 개념적 유추를 하나의 프레젠테이션으로 동기화하여 프로그램의 상태와 제어 흐름의 변화를 명확하게 표현합니다. GATs의 효과를 비교하기 위해 두 개의 CS1 과정에서 GATs와 텍스트 설명을 비교 연구했습니다.

- **Technical Details**: 연구는 두 개의 연구 기관인 TU Delft와 UofT에서 시행되었으며, 학생들을 무작위로 실험군(AI 생성 비디오)과 대조군(텍스트 설명)으로 나누었습니다. 학습 성과는 즉각적인 프로그래밍 과제와 다지선다형 질문으로 측정했으며, 학습 경험은 자가 보고된 인지 부하, 불만, 흥미로 평가했습니다. 마지막으로, 최종 시험을 통해 장기적인 성과도 평가하였습니다.

- **Performance Highlights**: GATs의 결과는 즉각적인 학습에 선택적인 이점을 제공하는 것으로 나타났지만, 이러한 효과는 맥락에 따라 달라지고 단기적이라는 점이 강조되었습니다. 학습자의 참여 프로파일에 따라 GATs의 성과에 영향을 미친 것으로 관측되었으며, 이는 개인화된 접근의 중요성을 강조합니다. 연구 결과는 CS1 교육에서 평균적인 효과를 넘어서 다양한 학습자 특성을 고려해야 함을 보여줍니다.



### VistaHop: Benchmarking Multi-hop Visual Reasoning for Visual DeepSearch (https://arxiv.org/abs/2606.03273)
- **What's New**: 이 연구에서는 Visual DeepSearch를 위한 새로운 벤치마크인 VistaHop을 소개합니다. VistaHop은 복잡한 비주얼 쿼리에 답하기 위해 이미지 영역을 반복적으로 검사하고, 시각적 증거를 기반으로 중간 추론을 연결하게 하는 멀티모달 대형 추론 모델(MLRM) 에이전트를 평가합니다. 기존 벤치마크가 단일 단계 시각 이해에 주로 초점을 맞추고 있다는 점에서 차별화됩니다.

- **Technical Details**: VistaHop은 300개의 고해상도 이미지, 25개의 시각 검색 시나리오, 그리고 시각적 앵커와 관련된 증거 체인을 추적하는 350개의 멀티 홉 품질 보증(QA) 작업을 포함합니다. 이를 통해 모델들이 여러 이미지 기반 추론 경로에서 정보를 융합하고, 이미지와 외부 지식을 연결할 수 있는 능력을 평가합니다. 또한, VistaArena라는 통합 평가 환경을 개발하여 텍스트 검색, 이미지 검색, 이미지 잘라내기 및 증거 기반 답변 검증을 지원합니다.

- **Performance Highlights**: 일곱 개의 대표적인 MLRM에 대한 실험 결과 현재 모델들은 VistaHop 작업을 성공적으로 해결하는 데 한계가 있음을 보여주었습니다. 가장 우수한 성과를 기록한 SenseNova-MARS-32B 모델조차도 24.31%라는 낮은 Pass@1 비율을 보였습니다. 이는 현재의 모델들이 시각적 고정을 통한 증거 재검토, 장기적 추론 및 멀티 앵커 정보 융합에서 여전히 한계를 가지고 있음을 드러냅니다.



### Are Common Substructures Transferable? Riemannian Graph Foundation Model with Neural Vector Bundles (https://arxiv.org/abs/2606.03270)
Comments:
          Accepted by ICML 2026

- **What's New**: 본 연구에서는 그래프의 구조적 전이 가능성에 대한 새로운 통찰을 제공합니다. 기존 연구가 이산(substructure) 영역에서 공통된 하위 구조에 국한된 반면, 우리는 기능적 행동(functional behavior)의 관점에서 전이 가능한 구조(transferrable structures)를 학습하는 방향으로 이동합니다. 이를 통해, 사전 훈련(pretraining) 중 학습된 구조적 행동이 목표 그래프에 적용될 때 적은 적응(adaptation)만으로 전이가 가능하다는 것을 제안합니다.

- **Technical Details**: 리만 기하학(Riemannian geometry)에 기반하여 Neural Vector Bundle이라는 그래프 내재 기하학 학습(framework)을 개발하였습니다. 이 프레임워크는 지역 좌표(local coordinates)를 사용하여 그래프 구조를 내재적으로 분석할 수 있도록 구성되어 있으며, 이때 각 지역은 해당 지역의 구조를 나타내는 첨부된 벡터 공간으로 설명됩니다. 또한, Dirichlet 손실 함수(Dirichlet loss)를 도입하여 표현(representation)과 불변 하위 구조(invariant substructures)를 공동으로 학습하도록 설계하였습니다.

- **Performance Highlights**: GAUGE라는 새로운 신경망 아키텍처가 제안되어 내재 기하학 학습을 통해 지식 전이(knowledge transfer)를 정확히 수행합니다. 이 아키텍처는 형상적으로 호환되는 이웃 섬유(fibers)를 점진적으로 평탄화하여 불변 하위 구조를 학습합니다. 경험적으로, 이 방법은 그래프 동형성(graph isomorphism) 및 제로샷 링크 예측(zero-shot link prediction)과 같은 도전적인 작업에서 우수한 표현력(superior expressiveness)을 입증하였습니다.



### EqGINO: Equivariant Geometry-Informed Fourier Neural Operators for 3D PDEs (https://arxiv.org/abs/2606.03260)
Comments:
          ICML 2026

- **What's New**: 본 논문에서는 3D Partial Differential Equations (PDEs)에 대한 새로운 딥러닝 모델인 EqGINO를 소개합니다. 이 모델은 스펙트럼 도메인에서의 동치성(equivariance)을 보장하며, 복잡한 불규칙 3D 기하학에 대해서도 물리 법칙을 잘 모델링할 수 있습니다. EqGINO는 저비용 구조적 사전 정보를 활용하여 제한된 SE(3) 변환 훈련 샘플로도 임의의 연속 방향으로 일반화할 수 있는 능력을 가지고 있습니다.

- **Technical Details**: EqGINO는 Fast Fourier Transform (FFT)을 통해 전역 스펙트럼 분석을 수행하며, 기존의 GNO 및 FNO를 기반으로 단독 동치 모듈인 EqGNO와 EqFNO로 재설계되었습니다. EqGNO는 비구조적인 기하학적 입력을 처리하는 기하학적 연산자 역할을 하며, EqFNO는 3D 격자 기반 학습 작업에 필요한 대칭 보존을 위한 일반 목적 동치 백본으로 기능합니다. 이를 통해 EqGINO는 컴퓨터 도메인 고유의 이산 SE(3) 하위 그룹에 대해 정확한 동치성을 보장합니다.

- **Performance Highlights**: EqGINO는 새로운 Orbit-based Weight Sharing 전략을 통해 스펙트럼 도메인에서 방향 동질성을 보장하면서 매개변수 복잡성을 줄입니다. 이로 인해 EQGINO는 효율적인 연산과 뛰어난 일반화 능력을 결합하여, 다른 최첨단 모델을 능가하는 우수한 성능을 나타냅니다. 이러한 접근 방식은 기존 3D 그룹 합성 곱(convolution)에서 발생하는 비소모적 비용 없이 동치성을 달성할 수 있도록 설계되었습니다.



### PSViT: A Methodology for Structurally Pruning Spiking Vision Transformers (https://arxiv.org/abs/2606.03257)
Comments:
          8 pages, 7 figures, 3 tables

- **What's New**: 이 연구에서 제안하는 PSViT는 Spiking Vision Transformer(SViT) 모델에 대한 구조적 프루닝(structured pruning) 방법론을 소개합니다. 이는 기존의 비구조적 프루닝(unstructured pruning) 접근 방식의 한계를 극복하고, 자원 제약이 있는 내장 플랫폼에서도 효율적인 추론을 가능하게 합니다. PSViT는 균일 채널 소프트웨어 차단, 감도 분석(sensitivity analysis), 및 미세 조정된 채널 소프트웨어 차단을 처리를 통해 모델의 메모리 절약을 달성하면서도 높은 정확도를 유지합니다.

- **Technical Details**: PSViT는 여러 단계로 구성되어 있으며, 첫 단계에서는 네트워크 각 층의 프루닝 민감도를 분석하여 메모리 절약 효과를 평가합니다. 다음으로는 균일한 채널 소프트웨어 차단을 수행하여 중요하지 않은 가중치를 구조적으로 제거합니다. 마지막으로, 각 네트워크 블록의 민감도를 기반으로 미세 조정을 수행하여 정확도를 최적화 합니다. 이를 통해 PSViT는 SViT 모델의 크기 감소를 효과적으로 지원하고 있습니다.

- **Performance Highlights**: 실험 결과, PSViT는 단일 샷 프루닝(single-shot pruning)을 통해 22.4%의 메모리 절약을 달성했습니다. 이는 기존의 SViT 모델보다 3% 이내의 높은 정확도를 유지하며, 세부 조정 없이도 활용할 수 있습니다. 이러한 성과는 자원 제약이 있는 AI 응용 프로그램에서 SViT의 효율적인 배치를 가능하게 하여, 개발 시간도 줄일 수 있는 장점을 제공합니다.



### AirDreamer: Generalist Drone Navigation with World Models (https://arxiv.org/abs/2606.03252)
Comments:
          8 pages, 8 figures

- **What's New**: 이 연구에서는 드론이 복잡하고 배치가 엉망인 환경을 성공적으로 탐색할 수 있도록 하는 새로운 탐색 프레임워크인 AirDreamer를 제안합니다. 이 프레임워크는 강화 학습 기반 정책과 세계 모델을 결합하여 로봇이 환경을 이해하고 내재적인 탐색 결정을 내리도록 돕습니다. 고급 환경 이해를 통해 드론은 기존 방법들보다 5.3% 높은 성공률을 기록하며 실시간 배포 시 별도의 조정 없이 효과적인 전환이 가능합니다.

- **Technical Details**: AirDreamer는 원시 깊이 관측 데이터에서 환경 구조를 인코딩하고 이를 기반으로 탐색 행동을 선택하기 위해 강화 학습 정책을 활용합니다. 세계 모델은 밀집한 예측을 제공하며, 정책은 수동으로 설계된 보상 대신 희소 보상을 사용하여 학습됩니다. 이로 인해 드론은 지역 최소점에서 벗어나고 복잡한 환경에서도 성공적으로 탐색할 수 있는 능력을 발휘합니다.

- **Performance Highlights**: 실험 결과, AirDreamer는 도전적인 맵에서도 최고 기준선보다 5.3% 높은 탐색 성공률을 기록하며, 이전의 다른 방법들이 실패하는 시나리오에서도 성공적으로 탐색했습니다. 이 시스템은 배치 시 추가 조정 없이도 효과적으로 사용할 수 있어 실용적인 응용 가능성이 큽니다. 전체 시스템 코드는 연구자들에게 공개될 예정입니다.



### When RLHF Fails: A Mechanistic Taxonomy of Reward Hacking, Collapse, and Evaluator Gaming (https://arxiv.org/abs/2606.03238)
Comments:
          20 pages, 8 figures; includes code, artifacts, and live demo

- **What's New**: 본 연구는 RLHF(Reinforcement Learning from Human Feedback) 과정에서 발생하는 실패 양상을 조사하여, 학습된 보상 모델이 잘못 최적화되는 방식을 분석합니다. 이를 통해 최적화가 미치는 영향을 다양한 상황에서 정량화하며, ‘보상 해킹(reward hacking)’을 단순한 사건으로 보지 않고 더 복잡한 전이 수준의 분류를 제안합니다. 이 연구는 RLHF의 실패를 점검할 수 있는 새로운 진단 체계를 제공합니다.

- **Technical Details**: 연구에서는 PPO(Proximal Policy Optimization), DPO(Direct Preference Optimization), UP-PPO(Uncertainty-Penalized PPO)와 같은 여러 최적화 기법을 사용하여 RLHF 파이프라인을 구축하였습니다. 61개의 체크포인트와 1920개의 전이 레벨 전환을 분석하였으며, 이는 각 전환에 대해 보상 모델 점수, 외부 평가자의 점수를 조사하여 보상이 이동하는 방향을 분석합니다. 보상 해킹의 로컬화된 감지를 위해, 사전 전환 데이터를 활용한 로지스틱 모델도 제시하였습니다.

- **Performance Highlights**: 연구 결과, 공격적인 PPO는 가장 높은 보상 해킹 비율(14.45%)을 가지며, UP-PPO는 비슷한 공격성이 있으나 더 낮은 비율(11.33-10.94%)을 보였습니다. 또한, AUC(Area Under Curve) 0.821의 예측 모델이 보상 해킹을 사전 전환 신호 기반으로 예측하는 성과를 나타내었습니다. 이러한 실험은 RLHF의 실패를 어떻게 조기에 탐지할 수 있는지를 조명하며, 현실적인 응용 가능성을 보여줍니다.



### GFFMERGE: Efficient Merging of Graph Neural Force Fields and Beyond (https://arxiv.org/abs/2606.03232)
- **What's New**: 본 논문에서는 Graph Neural Network(GNN) 기반의 힘장 모델을 효율적으로 병합하는 GFFMERGE라는 새로운 프레임워크를 소개합니다. 이 방법은 근본 모델에 대한 비용이 많이 드는 재훈련 없이 여러 개의 조정된 모델에서 얻은 지식을 통합할 수 있게 합니다. GFFMERGE는 메시지 패싱 레이어의 선형 구조를 활용하여 편리한 해법을 제공합니다.

- **Technical Details**: GFFMERGE는 GNN 힘장 아키텍처 내의 선형 매개변수 블록을 식별하고, 소스 모델에 대한 활성화 불일치를 최소화하는 폐쇄형 병합 목표를 도출합니다. 이는 비싼 그래디언트 기반 최적화 없이도 가능합니다. 또한, GFFMERGE는 기존의 모델 병합 방법에 비해 GNN에 특화된 성능 개선을 제공합니다.

- **Performance Highlights**: GFFMERGE와 GNNMERGE는 MD17, MD22, LiPS20과 같은 다양한 벤치마크에서 5-27배의 속도 향상을 성취하며, 조정된 모델의 모듈식 구성도 가능하게 합니다. 특히, 기존의 모든 기준 방법보다 더 우수한 초기값을 제공하여 빠르고 데이터 효율적인 수렴을 가능하게 합니다.



### BotDirector: Robot Storytelling Across the Symmetrical Reality with Multi-modal Interactions (https://arxiv.org/abs/2606.03223)
- **What's New**: 이번 연구에서는 어린이들이 로봇과 함께 이야기하는 혁신적인 시스템, BotDirector를 제안합니다. 이 시스템은 어린이들이 일상적인 물체를 활용하여 자신만의 이야기를 만들 수 있도록 돕고, 대화형 상호작용을 통해 복잡한 프로그램 필요 없이 로봇 드라마를 창조할 수 있습니다. 이를 통해 어린이들은 자신이 설정한 물체와 캐릭터에 기반하여 즉흥적으로 이야기를 구성하고 실행하는 경험을 직접적으로 경험하게 됩니다.

- **Technical Details**: BotDirector 시스템은 세 가지 단계로 구성되어 있습니다: 지식 생성, 대본 생성, 그리고 상호작용 놀이 단계입니다. 어린이는 시스템과 대화하며 흥미로운 주제를 제안하고, 이에 따라 이야기를 발전시켜 나갑니다. 생성된 대본은 가상 환경에서 로봇과 연계되어 물리적 환경에서 실행되며, 로봇의 움직임은 A* 알고리즘을 통해 계산됩니다. 어린이는 자연어와 물리적 인터페이스를 통해 즉각적인 피드백을 제공하며, 이를 통해 이야기를 조정할 수 있습니다.

- **Performance Highlights**: BotDirector 시스템은 어린이들이 스토리텔링의 창의성을 발휘할 수 있도록 향상된 인터랙션 경험을 제공합니다. 이 시스템은 가상과 물리적 환경을 연결하여 어린이들이 선택한 캐릭터에 상응하는 물체를 배치하여 역할을 부여하고, 이를 통해 로봇이 스크립트를 따라 연기하게 됩니다. 향후 연구에서는 이 시스템을 실제 어린이들과 함께 테스트하고, 가상 환경을 통해 물리적 공간의 한계를 보완하며 보다 효과적인 스토리텔링 방안을 모색할 계획입니다.



### WebRISE: Requirement-Induced State Evaluation for MLLM-Generated Web Artifacts (https://arxiv.org/abs/2606.03220)
- **What's New**: 이번 연구는 기존의 MLLM(다중 모달 대형 언어 모델)에서 생성된 웹 아티팩트를 평가하기 위한 새로운 벤치마크인 WebRISE를 소개합니다. WebRISE는 상호작용 계약 그래프(Interactive Contract Graphs)라는 구조를 통해 사용자 요구에 따른 상태 및 전이를 정의하며, 웹 페이지가 실제로 작동하는지 여부를 결정하는 데 필요한 요구 사항 기반의 평가를 가능하게 합니다. 이 연구는 442개의 작업을 통해 MLLM의 상호작용 생성이 여전히 해결되지 않았음을 보여줍니다.

- **Technical Details**: WebRISE는 웹 아티팩트를 평가하기 위해 명시적 및 암시적 요구 사항을 상호작용 계약으로 변환하여 처리합니다. 각 작업은 상호작용 계약 그래프(ICG)라는 모델을 통해 표현되며, ICG는 동적 UI 상태, 사용자 의도 전이 및 DOM/비주얼 주장을 포함합니다. 이 과정에서 웹 브라우저를 통해 평가되며, 생성된 페이지가 계약에 따라 작동하는지를 테스트합니다.

- **Performance Highlights**: 1414 개의 모델을 평가한 결과, 최고의 모델조차 65.6%의 전이 유효성과 66.3%의 요구 사항 커버리지를 달성했으며, 이는 약 1/3의 전이 및 요구 사항 체크가 충족되지 않았음을 의미합니다. 비디오 입력은 텍스트에 비해 상호작용 품질을 크게 향상시키며, 현재 MLLM 기반의 웹 생성 시스템은 여전히 많은 과제를 안고 있음을 보여줍니다. 결함 주입 실험을 통해 WebRISE의 평가가 기존의 검사 기준보다 2배에서 16배 더 효과적으로 상태 오류를 감지한다고 명시되었습니다.



### Reinforcement Learning from Cross-domain Videos with Video Prediction Mod (https://arxiv.org/abs/2606.03201)
- **What's New**: 본 논문에서는 XIPER (Cross-domain Video Prediction Reward)라는 보상 모델을 소개합니다. XIPER는 시각적으로 다른 도메인에서 수집된 전문가 비디오로부터 학습할 수 있도록 하여, 행동자(agent)의 외관 차이에도 불구하고 보상 신호를 생성합니다. 이 모델은 도메인 간 비디오 예측 모델을 학습하여, 전문가 도메인으로의 관찰 변환을 도와주고, 예측 확률을 보상 신호로 활용합니다.

- **Technical Details**: XIPER는 두 가지 구성 요소를 포함하는 도메인 간 비디오 예측 모델을 훈련합니다: (1) 행동자의 관찰을 전문가 도메인으로 매핑하는 도메인 변환 모델과 (2) 과거 프레임의 시퀀스에서 다음 전문가 프레임을 예측하는 비디오 예측 모델입니다. 이 구성 요소들은 오프라인에서 미리 훈련되고, 강화 학습 훈련 중에 동결됩니다. 예측 확률은 보상 신호로 사용되며, 이는 전문가 행동과의 정합성을 나타냅니다.

- **Performance Highlights**: DMC Color Suite(8개 작업) 및 DMC Body Suite(3개 작업)에서의 실험 결과, XIPER는 항상 세 가지 경쟁 기본선에 비해 우수한 성능을 보여 주었습니다. 또한, 시뮬레이션에서만 전문가 비디오를 사용할 수 있는 sim-to-real 전이 데이터셋에서 XIPER의 적용 가능성을 분석하였으며, 실제 로봇 관찰에 대해 유의미한 보상 신호를 생성함을 보여주었습니다. 논문과 관련된 코드 및 데이터셋은 프로젝트 웹페이지에서 확인할 수 있습니다.



### AI Rater Discrimination Depends on Scoring Protocol in Complex Clinical Decision-Making (https://arxiv.org/abs/2606.03198)
Comments:
          11 pages, 4 main figures, 8 supplementary figures, 9 supplementary tables

- **What's New**: 본 논문은 대규모 언어 모델(LLM)을 AI 평가자로 활용한 임상 AI 평가의 새로운 접근 방식을 제시하고 있습니다. 특히, 성인 제2형 당뇨병(T2D) 약물 치료에서의 AI 래이터 행동을 7가지 평가 질문에 대해 정량적으로 분석합니다. 연구는 Gold Rubric(그림자 척도)과 Non Gold Rubric(비첨척도)이라는 두 가지 스코어링 프로토콜을 사용하여 AI 평가자의 점수 차이를 파악하고 있습니다.

- **Technical Details**: 본 연구는 인공지능 행동 평가 규정의 차이를 분석하기 위해 인과 실험 설계를 사용하고 있습니다. 네 개의 오픈 소스 LLM이 임상 결정 지원 시스템(CDSS) 모델과 AI 평가자로 동시에 작동하며, 다양한 평가 조건에 따라 AI 평가자의 점수를 평가합니다. 연구에서는 스코어링 프로토콜과 CDSS 모델, 프롬프트 설정, 평가자의 모델, 프롬프트 문자 및 프롬프트 유형의 상호작용을 분석하였습니다.

- **Performance Highlights**: 결과에 따르면, Non-GR 프로토콜 하에서 AI 평가자의 점수가 평균 74-78점으로 결론지어졌고, GR보다 평균적으로 7.69에서 49.64점 낮은 점수를 보였습니다. GR 프로토콜은 DRG와 Baseline CDSS 출력 간의 AI 평가자의 구별력을 1.76배에서 5.10배까지 증가시키며, Non-GR에서는 그런 구별력이 억제된 것으로 나타났습니다. 이러한 결과는 임상 AI 평가에서 점수 프로토콜이 중요한 역할을 한다는 것을 지지합니다.



### Fully Automated Identification of Lexical Alignment and Preference-Stage Shifts in Large Language Models (https://arxiv.org/abs/2606.03165)
Comments:
          16 pages, 2 figures, 10 tables

- **What's New**: 디지털 대화 비서인 ChatGPT와 같은 Large Language Models(LLMs)에 대한 사용이 증가하고 있으며, 이러한 AI 툴은 프로그래밍, 언어 편집 및 정보 검색에 널리 이용되고 있다. 그러나 이들은 인간의 언어 사용과는 체계적으로 어긋나는 경향이 있다. 본 논문은 인간의 선호 학습 훈련에서 발생하는 이러한 불일치를 분석하고, 두 가지 새로운 평가 지표인 Lexical Alignment Score(LAS)와 Triangulated Preference Shift(TPS)를 제안한다.

- **Technical Details**: 이 연구에서는 42,000개의 PubMed 초록을 사용하여 6개의 모델 패밀리(Falcon, Gemma, Llama 등)의 생성 모델을 평가하였다. 새로운 지표인 LAS는 인간의 계속된 응답에 비해 과도하게 사용되는 용어를 정량적으로 평가하며, TPS는 선호 학습 단계에 기인한 변화를 분리하여 분석한다. 이러한 평가 방법은 수동 개입 없이도 과사용되는 단어를 식별하고, 인간의 선호와의 연결성을 추정한다.

- **Performance Highlights**: 결과적으로, LAS와 TPS는 개별 용어 및 선호 학습으로 인한 변화의 정도를 추정하는 데 유망한 결과를 보여주었다. 모든 변형에서 결과는 안정적이었으며, 파라미터 설정이나 무작위 시드의 변경에도 견고함을 유지하였다. 이 방식은 다른 언어로의 확장이 용이하며, 향후 모델의 개선 방향과 그 기원에 대한 이해에 기여할 수 있는 가능성을 지니고 있다.



### OpenAgenet/OAN: Technical Architecture for Trust-Governed Agent Identity and Discovery (https://arxiv.org/abs/2606.03163)
- **What's New**: 이 논문은 OpenAgenet/OAN의 기술 아키텍처를 설명합니다. OAN은 개방형 Agent 간의 상호 연결을 위한 프로토콜 중립적인 신뢰 계층입니다. 이 시스템은 다양한 Agent 프레임워크와 상호 작용 프로토콜을 지원하도록 설계되었으며, Agent의 신원 확인 및 검색 방식을 명확히 정의합니다.

- **Technical Details**: OAN은 신뢰에 기반한 Agent 신원 및 검색 아키텍처를 세분화합니다. 이 시스템은 Agent 신원이 등록, Root 수용, 패키지 분배, 검색 인덱싱, 검색 응답 검증 등 모든 과정에서 허용(filtering)되는지를 중점적으로 다룹니다. 이러한 기술적 범위는 Agent가 네트워크에서 가까운 당사자에게 표시되기 전에 요구되는 일련의 준비 과정을 포함합니다.

- **Performance Highlights**: 이 논문에서는 기초 구조가 각기 다른 운영자가 독립적으로 구현 가능한 여러 프로파일을 제공하므로, 특정 프로파일만 필요한 경우도 있다는 점을 강조합니다. OAN은 패키지를 통한 검색 최적화와 신뢰 검증을 구현하여 신뢰할 수 없는 소스를 통한 데이터 접근을 방지하며, 각 역할에 대해 명확한 리소스와 권한을 정의합니다.



### OpenAgenet/OAN: Open Infrastructure for Trusted Agent Interconnection (https://arxiv.org/abs/2606.03161)
- **What's New**: OpenAgenet(오픈 에이전트 네트워크, OAN)는 신뢰할 수 있는 에이전트 간의 연결을 위한 오픈 인프라 프로젝트로, 다양한 운영자가 참여하는 네트워크 환경에서 에이전트의 digital identity를 안전하게 검증하는 방법을 제공합니다. OAN은 에이전트 상호작용 프로토콜을 대체하지 않고, Root-이 관리하며, 인증된 패키지 배포 및 서명된 상호작용을 지원하는 신뢰 계층을 설계하였습니다. 이 프로젝트의 배경은 미래의 에이전트 생태계가 단일 플랫폼에 의해 운영되지 않을 것이라는 현실적인 관찰에서 출발합니다.

- **Technical Details**: OAN은 에이전트가 상호작용하기 전 안전하게 신뢰할 수 있는 identity를 확인하는 데 중점을 두고 있으며, 이는 Root 관리하의 수용, 인증된 패키지 배포, 권한 기반 검색 및 서명된 에이전트 간 호출 등을 포함합니다. 이 아키텍처는 레이어 방식으로 설계되어 있으며, 각 조직이 독립적으로 에이전트를 등록하고 관리할 수 있는 공통의 신뢰 기반을 제공합니다. OAN의 주요 목표는 신뢰성을 유지하면서도 다양한 에이전트가 상호 간에 탐지 가능하고 검증 가능하게 하는 것입니다.

- **Performance Highlights**: OAN은 에이전트 간의 사전 연관 신뢰 문제를 해결하기 위해 다섯 가지 약속을 정의하고 있습니다. 이는 상호작용 전 identity 확인, 디렉토리 노출 전 생애 주기 확인, 권한 인지 검색, 호출 전 검증을 포함하여 여러 에이전트가 협력할 수 있는 거버넌스 지향적인 인터커넥션 레이어를 제공합니다. OAN은 MCP, A2A 및 ANP와 호환성을 지니며, 모든 에이전트 플랫폼이 동일한 신뢰 기준 아래에서 정보의 처리를 안전하게 수행할 수 있도록 지원합니다.



### NVIDIA OmniDreams: Real-Time Generative World Model for Closed-Loop Autonomous Vehicle Simulation (https://arxiv.org/abs/2606.03159)
- **What's New**: 오토노머스(Autonomous) 차량 기술이 발전함에 따라, 긴 꼬리(long-tail) 시나리오에서의 안전한 주행 정책 평가가 중요한 과제로 남아있습니다. 새로운 연구에서는 전통적인 심플레이터(simulator)의 한계를 극복하기 위해, 'OmniDreams'라는 혁신적인 생성적 세계 모델을 도입하였습니다. 이 모델은 Cosmos 확산(diffusion) 모델을 기반으로 하여 실시간으로 행동 조건화(action-conditioned) 비디오를 자가 회귀적으로 생성합니다.

- **Technical Details**: OmniDreams는 과거의 프레임과 현재의 시뮬레이터 상태, 즉각적인 주행 행동을 바탕으로 하여 포토리얼리즘(photorealism)을 갖춘 센서 데이터를 생성합니다. 이 모델은 21,000시간의 주행 시나리오를 활용하여 극한의 날씨와 예측 불가능한 동적 행동과 같은 복잡한 현상을 합성할 수 있습니다. 특정 정책 모델인 Alpamayo 1과 AlpaSim 조정기( orchestrator)와 연결되어, OmniDreams는 반응적 환경으로 작동합니다.

- **Performance Highlights**: OmniDreams에서 후속 훈련된 세계-행동 모델(WAM)은 'Physical AI Autonomous Vehicles NuRec' 데이터셋에서 강력한 성능을 보여 주며, VLA 기반의 Alpamayo 1.5 연구 정책 모델을 초월하였습니다. 흥미롭게도, WAM은 전체 매개변수(parameter)의 1/5만 사용하면서도 이전 모델보다 더 뛰어난 결과를 기록했습니다. 이러한 결과는 OmniDreams가 정책 아키텍처의 기초(backbone)로 활용될 잠재력을 강조합니다.



### Decoupled Smart Contract Audits: Lightweight LLM Framework via Distillation and Aggregation (https://arxiv.org/abs/2606.03128)
Comments:
          12 pages, 4 figures, 5 tables. Accepted to IEEE ICWS 2026

- **What's New**: 이번 연구에서는 스마트 계약 보안을 위한 효율적인 엔드 투 엔드 감사 프레임워크를 소개합니다. 이 프레임워크는 경량의 최적화된 오픈 소스 LLM (0.6B-4B 매개변수)을 활용하여 구성됩니다. 우리의 접근 방식은 감사 작업을 취약점 감지, 설명, 심각도 분류 및 수정 권장 사항의 네 가지 상호 연결된 구성 요소로 분리하는 점에서 혁신적입니다.

- **Technical Details**: 전통적인 단일 모형 대신, 우리의 프레임워크는 각 구성 요소가 별도의 전문 모듈로 기능하도록 분리되어 작업을 순차적으로 수행합니다. 각 작업을 위해 Rank-Stabilized Low-Rank Adapters (rsLoRA) 및 지식 증류(Knowledge Distillation)를 이용하여 감지 성능을 유지하고 있습니다. CoVe (Chain-of-Verification) 집계 방법을 사용하여 모델로부터 생성된 여러 초안 응답을 필터링 및 검증하여 단일 고정밀도를 출력하도록 합니다.

- **Performance Highlights**: 실험 결과, 우리의 경량 파이프라인은 최신의 대형 LLM보다 연속적으로 우수한 성능을 나타내며, 98.25%의 취약점 감지 정확도를 달성했습니다. 또한 심각도 분류에서 상당한 편향을 발견하여 향후 연구에 중요한 기준을 수립하는 데 기여할 것으로 보입니다. 이 연구는 독립적인 Web3 개발자들이 사용할 수 있는 효과적인 감사 도구를 제공하여 스마트 계약 보안을 민주화하는 데 기여할 것입니다.



### GuidedBridge: Training-freely Improving Bridge Models with Prior Guidanc (https://arxiv.org/abs/2606.03119)
Comments:
          ICML 2026

- **What's New**: 이 논문에서는 Prior Guidance (PG)라는 새로운 훈련이 필요 없는 브리지 가이던스 방법을 제안합니다. PG는 이전의 방법들이 생성 품질을 개선하기 위해 사용하던 정보 없는 약한 prior를 도입하여, 효과적인 prior 활용을 방해하고 노이즈 제거 결과를 향상시킵니다. 이 과정에서 bridge 프로세스의 기본 메커니즘을 분석하고, 주파수 변조 prior 가이던스(FMPG)를 설계하여 가이던스 스케일을 조정합니다.

- **Technical Details**: 브리지 모델은 교훈적인 클린 prior를 활용하는 데이터-투-데이터 생성 프로세스를 도입합니다. PG는 훈련이 필요 없는 방식으로 약한 prior를 추가하여 prior 활용의 난이도를 증가시킵니다. FMPG는 고주파와 저주파 대역에 맞춰 가이던스 스케일을 조정하는 설계를 통해 브리지 생성의 동적과 일치하게 지식 정보를 활용합니다.

- **Performance Highlights**: 실험 결과, PG 방법이 다양한 이미지 변환 작업에서 사전 훈련된 브리지 모델의 성능을 지속적으로 향상시킴을 보여줍니다. 특히, CFG-FMPG라는 계단식 프레임워크는 CFG를 이용해 근본적인 구조를 복원하고, 이후 FMPG를 활용하여 고품질로 세부 사항을 개선하여 샘플링 효율성을 유지합니다.



### AnyAudio-Judge: A Dynamic Rubric-Based Benchmark and Evaluator for Audio Instruction Following (https://arxiv.org/abs/2606.03116)
- **What's New**: 이번 논문에서는 비디오 생성에 있어 지침에 따른 오디오 생성과 관련된 평가 방법의 필요성을 강조합니다. 특히, 기존의 평가 방법들이 복잡한 지침을 분리하는 데 어려움을 겪으면서 정밀한 일치 여부를 판단하기 힘든 문제를 해결하기 위한 새로운 접근 방식을 제안합니다. 여기서 소개된 AnyAudio-Judge Bench는 다양한 오디오 도메인에서 엄선된 데이터셋을 활용하여 세밀한 평가를 가능하게 합니다.

- **Technical Details**: 저자들은 동적 루브릭 기반 평가(paradigm)를 도입하여 복잡한 오디오 캡션을 여러 개의 독립적인 이진 루브릭 항목으로 분해합니다. 이를 통해 각 항목에 대해 개별 평가를 수행하고, 최종 점수는 항목 수준의 만족도 확률을 집계하여 계산됩니다. 또한, SFT(Supervised Fine-Tuning)와 GRPO(Group Relative Policy Optimization)를 결합하여 AnyAudio-Judge 모델을 훈련합니다.

- **Performance Highlights**: AnyAudio-Judge는 기존의 자동화 평가 방법보다 지침에 따른 정밀한 정렬 감지에 있어 월등한 성능을 보입니다. Extensive experiments에서는 본 모델이 상위 baseline에 비해 유의미하게 개선된 결과를 보여주었으며, 강화 학습 강화 프로그램에서 더 나은 성능을 발휘하는 것을 입증합니다. 이는 기존 지침을 따르는 오디오 생성의 최적화를 지원하는 데 기여합니다.



### PhotoCraft: Agentic Reasoning with Hierarchical Self-Evolving Memory for Deep Image Search (https://arxiv.org/abs/2606.03099)
- **What's New**: 이 논문에서는 Deep Image Search의 한계점을 극복하기 위한 방법으로 PhotoCraft를 제안합니다. PhotoCraft는 계층적 메모리 시스템을 제공하며, 이는 작업 간의 경험 전이를 용이하게 하는 동시에 논리적 일관성과 지식 전이 가능하게 합니다. 이 시스템은 훈련이 필요 없는 구조로, MLLM 기반 에이전트의 메모리 병목 문제를 해결하는 데 도움을 줍니다.

- **Technical Details**: PhotoCraft는 작업 기억(working memory), 에피소드 기억(episodic memory), 의미 기억(semantic memory)의 세 가지 상호 보완적인 구성 요소로 이루어져 있습니다. 작업 기억은 단기적인 상황 인식을 지원하고, 에피소드 기억은 현재 프로세스를 역사적인 목표와 연결하며, 의미 기억은 추상적인 지식 표현과 일반화를 가능하게 합니다. 이러한 메모리 시스템은 다단계 추론 과정 동안 동적으로 호출되어 논리적 일관성을 유지합니다.

- **Performance Highlights**: DISBench에서의 광범위한 실험 결과, PhotoCraft는 다양한 MLLM 백본에서 맥락을 인식하는 검색 성능을 지속적으로 향상시키며, 최대 18.5%의 개선 효과를 보여주었습니다. 또한, 이 시스템은 메모리 없는 깊은 이미지 검색의 주요 병목을 효과적으로 완화하여 신뢰할 수 있고 일반화 가능한 다중 모드 검색 에이전트를 구축하는 실질적인 경로를 제시합니다.



### BAHSD: Bridging the Long-tail Gap via Adaptive Distillation in Black-box Sequential Recommendation (https://arxiv.org/abs/2606.03091)
- **What's New**: 본 논문에서는 순차 추천 시스템에서의 신호 이질성 문제를 해결하기 위해 BAHSD(Black-box Adaptive Heterogeneous Signal Distillation)라는 새로운 프레임워크를 제안합니다. 기존의 모델 추출 방식은 사용자 상호작용의 긴 꼬리 분포에 따른 신호의 다양성을 간과하고, 이로 인해 헤드 사용자(주요 사용자)와 테일 사용자(인기 없는 사용자) 간 성능이 상이하게 나타납니다. BAHSD는 다중 스케일 일관성 탐사를 통해 신호 신뢰성을 평가하고, 이를 바탕으로 계층적 목표를 조정하여 다양한 사용자 신호에 적응하는 방식을 채택합니다.

- **Technical Details**: BAHSD는 교수 모델의 로그만을 사용하여 작동하며, 다중 길이의 서브 시퀀스에서의 교수 출력의 일관성을 측정하여 신뢰도를 정량화합니다. 신뢰도가 높은 신호에 대해서는 동적 온도를 적용한 KL 발산을 사용하여 선호 고착화를 완화하고, 신뢰도가 낮은 신호는 순위 일관성과 InfoNCE 대조 학습을 적용하여 노이즈를 억제합니다. 이러한 방법은 명시적인 사용자 계층화나 내부 접근 없이도 최적의 정보 전이를 가능하게 합니다.

- **Performance Highlights**: BAHSD는 공개 벤치마크에서 기존의 최첨단 모델보다 최대 4.98% 향상을 달성하였고 테일 사용자에 대해서는 80% 이상의 성능 개선을 기록했습니다. 이러한 결과는 BAHSD가 높은 충실도의 블랙박스 모델 추출에 있어 즉시 적용 가능하고, 모델에 구애받지 않는 솔루션임을 나타냅니다. 실험 결과를 통해 본 프레임워크가 신뢰도에 기반한 조정 가능성을 증명하며, 다양한 사용자 그룹의 이질한 신호를 효과적으로 대응하고 있음을 알 수 있습니다.



### "**Important** You should give me full credits!": Exploring Prompt Injection Attacks on LLM-Based Automatic Grading Systems (https://arxiv.org/abs/2606.03090)
Comments:
          15 pages, 8 figures, 9 tables

- **What's New**: 이 논문은 대규모 언어 모델(LLMs) 기반의 자동 채점 시스템(AG system)에서 발생하는 프롬프트 주입(prompt injection, PI) 공격에 대한 심층 분석을 제공합니다. 연구자들은 LLM의 뛰어난 지시 수용 능력과 광범위한 사전 지식을 통해 교육 환경에서 자동 채점을 보다 효과적으로 수행할 수 있는 가능성을 제시합니다. 하지만 PI 공격으로 인해 교육 평가의 공정성과 신뢰성에 심각한 위협이 발생할 수 있는 점을 강조합니다.

- **Technical Details**: 연구진은 학생의 관점에서 실제 교육 환경에 맞춘 문제 불문 보편적 공격을 설계하였으며, 이는 특정 질문에 대한 최적화 없이 사용될 수 있는 공격 프롬프트를 생성하는 방식입니다. 또한, 일반적으로 LLM 기반 AG 시스템은 최종 점수만 제공하고 중간 과정이나 채점 근거를 숨기므로, 연구는 이러한 제약 조건 하에서 수행되었습니다. 공격자는 자신이 제출한 원래 답변에 추가적인 프롬프트를 덧붙여 점수를 인위적으로 높이려는 목표를 가지고 있습니다.

- **Performance Highlights**: 30개 이상의 질문을 대상으로 한 실험 결과, 기존 LLM 기반 AG 시스템이 PI 공격에 매우 취약하다는 것을 보여주었습니다. 단순한 보편적 공격조차도 효과적으로 결과를 조작할 수 있으며, 방어 메커니즘이 적용된 경우에도 공격 성공률이 높았습니다. 이러한 발견은 LLM 기반 평가 시스템의 보안 문제를 인식하고, 안전하고 신뢰할 수 있는 교육 AI 구조에 대한 추가 연구의 필요성을 제기합니다.



### Constitutional On-Policy Safe Distillation (https://arxiv.org/abs/2606.03089)
- **What's New**: 최근 연구는 On-Policy Self-Distillation (OPSD)이 안전 정렬(safety alignment)에서 중대한 한계를 보일 수 있다는 사실을 밝혔습니다. 특히, Constitution-based teacher conditioning이 모델의 응답을 과도하게 보수적으로 제한함으로써 응답의 표현성을 약화시킵니다. 이를 해결하기 위해 Constitutional On-Policy Safe Distillation (COPSD)라는 새로운 프레임워크가 제안되었습니다.

- **Technical Details**: COPSD는 두 단계로 구성되며, 첫 번째 단계에서는 Cross-SFT Cold-Start 방식으로 teacher를 안전 헌법에 맞게 보정합니다. 두 번째 단계에서, 보정된 teacher는 헌법 조건에 따른 on-policy distillation을 수행하여, 응답의 길이 및 다양성을 유지하면서 안전성을 보장합니다. 이 과정에서 지오메트릭 누수(geometric leakage)를 감소시키는 것이 주요 목표입니다.

- **Performance Highlights**: 12개의 벤치마크 실험 결과, COPSD는 기존의 최첨단 정렬 방법들과 비교하여 안정성과 유용성 간의 균형을 더 잘 유지하는 것으로 나타났습니다. 또한 COPSD는 일반적인 추론 능력에 대한 안전 세금(safety tax)을 크게 줄이며, 전반적인 성능을 향상시키는 데 기여합니다.



### Regret Pre-training: Bridging Prior and Posterior Views for Enhanced Knowledge Grounding (https://arxiv.org/abs/2606.03080)
- **What's New**: 이 논문은 Regret Pre-training이라는 새로운 자기 지도 학습 프레임워크를 도입합니다. 이 프레임워크는 학습 중에 미래 정보를 활용하여, causal language model의 비대칭성을 해결하는 데 목표를 둡니다. 이를 위해 dual-view architecture를 사용하여 causal Student distribution과 future-conditioned Teacher distribution을 생성합니다.

- **Technical Details**: Regret Pre-training은 특정 손실 함수인 regret loss를 사용하여 KL divergence를 최소화하며, 잠재적 정보를 바탕으로 causal representation에 미래 예측을 전달합니다. 두 가지 teacher 설정(LocalRegret 및 GlobalRegret)을 사용하여 OLMoE-1B-7B 모델에서 실험을 진행하며, 각 설정은 미래 컨텍스트의 범위에 따라 달라집니다. 이 프레임워크는 전체적인 아키텍처 수정 없이 attention mask 생성만으로 구현될 수 있습니다.

- **Performance Highlights**: 실험 결과는 두 가지 설정 모두 기존 baseline보다 우수한 성능을 보인다는 것을 보여주었습니다. 특히 GlobalRegret은 BoolQ에서 61.0%의 정확도로 baseline의 42.9%보다 18.1%p 높은 성과를 기록했습니다. 평균적으로 GlobalRegret과 LocalRegret은 각각 33.9%와 32.2%의 정확도를 달성하여 baseline의 30.2%를 초과했습니다.



### Libra: Efficient Resource Management for Agentic RL Post-Training (https://arxiv.org/abs/2606.03077)
Comments:
          18 pages, 13 figures

- **What's New**: 이번 연구에서는 에이전틱 강화 학습(agentic reinforcement learning)에서의 자원 관리 문제를 해결하기 위해 새로운 시스템인 Libra를 제안합니다. Libra는 롤아웃(rollout)과 훈련(training) 간의 자원 할당을 공동 최적화하는 기법과, 도구 실행 결과에 기반하여 요청을 경로 지정하는 인과관계 기반의 스케줄러인 C-MLFQ를 도입합니다. 이 시스템은 GPU 자원 할당 방식의 비효율성을 극복하고 최신 에이전틱 RL 기법에 적합한 프레임워크를 제공합니다.

- **Technical Details**: Libra의 핵심 메커니즘은 두 가지입니다. 첫 번째는 정기적인 글로벌 자원 계획자로, 고정된 GPU 예산 하에 롤아웃과 훈련 클러스터 간의 GPU 할당을 최적화하여 전체 반복의 소요 시간을 최소화합니다. 두 번째는 인과관계 기반의 다단계 피드백 큐 스케줄러(C-MLFQ)로, 이는 도구 반환 결과에서 파생된 인과 신호를 활용하여 이질적인 롤아웃 버킷으로 요청을 전송합니다.

- **Performance Highlights**: Libra는 48개의 A800 GPU에서 평가되었으며, 최대 3.0배 높은 처리량과 최대 2.5배 더 빠른 보상 수렴 속도를 달성하였습니다. 특히 Search-R1에서 Libra는 약 2,700 토큰/s의 속도를 기록하며, 기존 시스템에 비해 큰 성능 향상을 보여주었습니다. C-MLFQ 스케줄러는 결정당 라우팅 정확도 91.1%를 달성하여 기존 예측 기반 시스템보다 우수한 성능을 보였습니다.



### Efficient Hyperparameter Optimization for LLM Reinforcement Learning (https://arxiv.org/abs/2606.03073)
Comments:
          12 pages, 6 figures, accepted at ACL 2026

- **What's New**: 이번 연구에서는 Joint Fidelity Hyperparameter Optimization (JF-HPO)라는 새로운 하이퍼파라미터 최적화(Hyperparameter Optimization) 방법을 제안합니다. 이는 모델 크기와 교육 예산을 동시에 조정하여 LLM(Reinforcement Learning)에서의 효율성을 높이고자 합니다. JF-HPO는 작은 프록시(proxy) 모델을 활용하여 각 최적화 시도에서 효율적인 훈련과 평가를 가능하게 하며, 조기 중단 전략과 효율적인 체크포인팅 메커니즘을 통합하여 연산 비용을 줄입니다.

- **Technical Details**: JF-HPO는 Bayesian optimization 절차 내에서 모델 크기와 교육 예산을 믿음(fidelity)으로 설정하여 대형 LLM 모델 훈련의 비효율성을 해소합니다. Proximal Policy Optimization (PPO) 및 Group Relative Policy Optimization (GRPO) 알고리즘을 통해 RL의 안정성을 개선하고 메모리 소비를 줄이는데 중점을 두었습니다. GRPO에서 값 함수(value function)를 배제하고, 샘플링된 출력을 기반으로 이점(advantage)을 계산하여 계산 효율성을 높입니다.

- **Performance Highlights**: JF-HPO는 기존 하이퍼파라미터 최적화 방법에 비해 실험당 최대 14.9배 빠른 실행 속도를 자랑합니다. 총 24회의 실험 중 22회에서 기존 방법보다 높은 성능을 기록하였으며, VeRL Recipe의 하이퍼파라미터 조합에 비해 5.8%에서 111.6%까지의 성능 향상을 보여주었습니다. 전반적으로 JF-HPO는 LLM RL에 있어 뛰어난 효율성과 효과성을 입증하였습니다.



### ASymPO: Asymmetric-Scale Policy Optimization for Asynchronous LLM Post-Training Without Behavior Information (https://arxiv.org/abs/2606.03070)
- **What's New**: 이 논문에서는 비동기 강화 학습(asynchronous reinforcement learning, ARL)이 언어 모델(post-training throughput 개선)에 미치는 영향을 설명합니다. 특히, 응답 생성과 정책 최적화의 분리가 주어진 자리에서 응답이 늦춰지는 문제를 해결하지 못하면서 발생하는 분포 변화(distribution drift) 문제를 조명합니다. 이러한 문제를 해결하기 위해, 저자들은 비대칭 스케일 정책 최적화(Asymmetric-Scale Policy Optimization, ASymPO) 방법을 제안하며, 이는 현재 정책의 확률만으로도 안정성을 확보할 수 있음을 증명합니다.

- **Technical Details**: ASymPO는 각 응답의 토큰 손실을 현재 평균 토큰의 음수 로그 확률에 따라 정규화합니다. 이 방법은 비행동 정책 확률을 필요로 하지 않으며, 응답 수준에서 제로섬 균형을 회복하고 비제로 학습 신호를 보존합니다. 저자들은 스케일 정책 최적화(Scaled Policy Optimization, SPO)라는 고정 음수 스케일 기준을 도입하여, 비동기적 수학적 추론(post-training) 분야에서 이 두 가지 현재 정책 기반 목표를 평가합니다.

- **Performance Highlights**: 실험 결과, ASymPO는 기존 비행동 정책 정보를 요구하지 않으면서도 롤아웃-학습기 인터페이스를 더 간소화할 수 있음을 보여주며, 정밀한 훈련 시 필요한 로그 재계산(logit recomputation)이나 정책 버전 관리(policy-version bookkeeping)를 필요로 하지 않습니다. 또한 저자들은 ASymPO가 긍정 및 부정 손실 기여도의 균형을 정확히 잡을 수 있음을 증명합니다. 이 연구는 또한 다양한 모델 군에 대해 비동기적 수학적 추론 후 훈련에서 naive 현재 정책 훈련과의 비교를 통해 ASymPO의 효과를 강조합니다.



### ROBUST-WT: Robust Uncertainty-aware Segmentation Transform via Whitening and Training Enhancements (https://arxiv.org/abs/2606.03069)
Comments:
          8 pages, 6 figures; code available at this https URL

- **What's New**: 이 논문은 의학 이미지를 위한 범용 세분화 문제에 대한 새로운 기법인 Whitening Transform 기반 확률적 형태 정규화 추출기(WT-PSE)를 다룬다. 특히 다양한 의료 장치와 임상 프로토콜에서 발생하는 성능 저하를 방지하기 위해, feature decorrelation 및 Wasserstein distance 기반의 지식 증류를 사용하여 강력한 교차 도메인 세분화를 구현한다. 원래 WT-PSE 시스템의 네 가지 제한사항을 설정하고, 이를 극복하기 위한 여러 개선책을 제안하였다.

- **Technical Details**: 원래의 WT-PSE 구현은 데이터 증강, 손실 함수, 가중치 조절 전략 및 비교 분석에 대한 네 가지 주요 한계를 가지고 있다. 새로운 방법으로는 domain-adaptive augmentation, 하이브리드 BCE 및 Dice 손실 함수를 결합하여 가장자리 인식 세분화를 개선하고, 커리큘럼 기반 Dice 가중치 스케줄링 전략 및 체계적인 비교 연구를 위한 커맨드라인 제어 플래그를 포함한다. 이러한 개선사항은 최종적으로 세분화 정확도를 높이는 데 기여한다.

- **Performance Highlights**: 제안된 개선 파이프라인은 Fundus optic disc segmentation 벤치마크에서 0.956의 최종 Dice 점수와 13.31의 ASD 점수를 기록하여 baseline인 0.939을 초월하는 성과를 보였다. 이러한 결과는 훈련 레벨에서의 개선이 기본 WT-PSE 구조를 변경하지 않고도 일관된 성능 향상을 제공할 수 있음을 나타낸다.



### Learn When and Where to Connect: Adaptive Virtual Nodes for Dynamic Message Passing on Graphs (https://arxiv.org/abs/2606.03068)
Comments:
          12 pages, 6 figures, 10 tables, 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2026)

- **What's New**: MAVN (dynamic Message passing with Adaptive Virtual Nodes)는 기존의 Virtual Node (VN) 기반 방식의 한계를 극복하는 새로운 MPNN 프레임워크입니다. MAVN은 노드와 VN 간의 비제한적이고 비균일한 연결을 지원하며, 각 레이어에서 필요에 따라 VNs를 동적으로 도입합니다. 이 방식은 노드의 상대적 중요도에 기반하여 언제(VNs를 도입해야 할 시점) 및 어디서(어떤 노드에 연결할지) VN을 연결할지 결정하는 능력을 가지고 있습니다.

- **Technical Details**: 기존의 MPNNs은 노드가 VN에 연결되는 방식을 고정하거나 모든 노드에 동일한 수의 VNs를 연결하는 제약이 있었습니다. 그러나 MAVN은 이러한 연결을 동적으로 조정하며, 노드의 표현이 업데이트됨에 따라 새로운 VNs의 필요성을 검사하여 레이어마다 VNs를 추가합니다. MAVN의 설계는 메시지 전달 경로가 각 레이어에서 진화할 수 있도록 하며, 모든 노드 표현만을 사용하여 어떤 MPNN 아키텍처에도 적용할 수 있습니다.

- **Performance Highlights**: MAVN은 아홉 개의 실제 데이터셋에서 최첨단 방법들보다 뛰어난 성능을 보이며, 특히 기존의 MPNN의 성능을 최대 46.5% 향상하는 결과를 보였습니다. 이는 비교 기준과 비교할 때 유의미한 성능 향상을 나타냅니다. MAVN이 각 그래프에 맞춤형 메시지 전달 경로를 생성함으로써 과도한 압축(over-squashing) 및 과소 노출(under-reaching) 문제를 완화함을 입증하였습니다.



### Brief Announcement: Generative Markov Model for Distributed Computing Systems (https://arxiv.org/abs/2606.03061)
Comments:
          Submitted to 40th International Symposium on Distributed Computing (DISC 2026)

- **What's New**: 이번 연구에서는 분산 컴퓨팅 시스템을 모델링하기 위한 생성적 마르코프 모델(generative Markov model) 프레임워크를 제안합니다. 이 모델은 분산 시스템의 고유한 희소 의존성 구조를 반영하여 상태를 고차원 변수로 분해하며, 이로 인해 복잡한 시스템 상황에서도 시뮬레이션, 추론(inference), 정책 학습(policy learning)이 가능해집니다. 특히, 연구자는 이 프레임워크가 마르코프 체인 이론(Markov chain theory) 및 강화 학습(reinforcement learning)과 잘 결합될 수 있음을 보여줍니다.

- **Technical Details**: 제안된 모델은 디지털 분산 컴퓨팅 시스템의 상태를 특수한 변수 조합으로 관리하며, 시퀀스의 상태를 생성적 모델(generative model)로 나타냅니다. Markov property를 이용하여 이전 상태에 대한 의존성을 제한하고, 상태 변수 간의 희소한 상관관계를 고려하여 복잡한 시스템의 처리를 용이하게 합니다. 즉, 각 컴포넌트는 전역 상태가 아닌 소수의 다른 컴포넌트에 따라 발전하기 때문에 계산적 처리 가능성이 높아집니다.

- **Performance Highlights**: 이 프레임워크를 실증적인 사례로 협업 AI 추론(collaborative AI inference)에 적용한 결과, 서버의 중앙 집중형 스케줄링은 대규모에서 병목 현상을 초래하며, 사용자 장치 간의 분산 계산이 지연(latency)과 서버 자원 소비를 감소시키는 데 효과적임을 확인했습니다. 이러한 결과는 분산 컴퓨팅 시스템에서 적응형 의사 결정(adaptive decision-making)의 중요성을 강조하며, 시스템 모델링, 시뮬레이션, 최적화를 위한 유용한 도구임을 보여줍니다.



### Rethinking Molecular Text Representations for LLMs: An Empirical Study (https://arxiv.org/abs/2606.03057)
Comments:
          25 pages, 11 figures, 20 tables

- **What's New**: 본 논문은 다양한 분자 표현 방식의 성능을 정량적으로 평가하기 위한 체계적인 기준을 제시합니다. 특히 9가지 표현 방식과 8가지 화학 태스크를 통해 대형 언어 모델(LLM)의 분자 컴피턴스(molecular competence)를 분석하고, 대표적인 표현 방식인 SMILES 대신 다른 가능성들을 탐구합니다. 연구 결과, 모든 화학 태스크에서 단일한 대표 방법이 아닌 여러 표현 방식이 성능에 밀접하게 영향을 미친다고 제안합니다.

- **Technical Details**: 대형 언어 모델이 여러 화학 표현 방식을 다르게 인코딩하고 있으며, 이들이 모델의 성능에 미치는 영향을 관찰했습니다. 특히 CML(Chemical Markup Language)과 MolJSON과 같은 명확한 구조화된 텍스트 표현이 구조적 작업에서 우월한 성능을 보이는 반면, IUPAC은 의미적 작업에서 다른 모든 LLM에 대해 분자 검색에서 우세하다는 것을 발견했습니다. 이는 구조적 표현 방식이 높은 주의를 요구함을 나타냅니다.

- **Performance Highlights**: 연구는 16개의 LLM을 평가하여 CML이 가장 뛰어난 성능을 보였고, 그 다음으로 MolJSON, InChI, 그리고 표준 SMILES 순임을 보여주었습니다. 화학 특화 모델들은 SMILES로 잘 작동하지만 구조화된 텍스트 표현에서는 성능 저하가 나타났습니다. 이 연구는 LLM 기반 화학에서 작업 인식 표현 경로(task-aware representation routing)의 필요성을 강조합니다.



### Capability Advertisement as a Market for Lemons: A Trust Layer for Heterogeneous Agent Networks (https://arxiv.org/abs/2606.03034)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLM) 에이전트들이 서로 작업을 위임하는 경향이 증가하고 있다는 점을 조명합니다. 에이전트가 자신의 능력을 광고하고 다른 에이전트가 이를 호출하는 모델 컨텍스트 프로토콜(MCP)과 에이전트 간 프로토콜(A2A)의 사용이 확대되고 있지만, 이러한 프로토콜은 광고된 능력이 실제와 다를 수 있음을 간과하고 있습니다. 저자들은 '레몬 마켓(market for lemons)' 현상을 언급하며, 품질이 숨겨져 있는 시장에서 양질의 제공자가 불리한 상황에 처할 수 있음을 설명합니다.

- **Technical Details**: 이 논문은 불확실성과 비대칭 정보를 해결하기 위한 세 가지 경제학적 해결책인 신호(signaling), 스크리닝(screening), 그리고 명성(reputation)이 현재의 에이전트 프로토콜에 결여되어 있다고 주장합니다. 그 결과로, 저자들은 신뢰 계층(Trust Layer)을 제안하는데, 이는 MCP와 A2A 위에 위치하며 확률적 능력 설명자, 스크리닝 및 명성을 추가합니다. 이 계층은 프로토콜에 독립적이며, 과대 광고의 비용이 이득을 초과할 때 구분 균형(separating equilibrium)을 허용한다고 주장합니다.

- **Performance Highlights**: 저자들은 이 설계가 신뢰 앵커가 부실하거나 없을 때도 안정적으로 작동할 수 있도록 설계되었다고 명시합니다. 또한, 이 논문은 에이전트 간 위임 체인의 신뢰도 조합(approximation) 경계를 다루며, 신뢰할 수 없는 제공자로부터의 품질 저하를 방지하는 데 기여할 수 있는 방법을 제시합니다. 논문의 분석은 새로운 측정의 보고가 아닌 신뢰성 향상에 대한 이론적 기여를 기반으로 하고 있습니다.



### Conditional Hypothesis Generation for LLM-Based Text Analysis with Researcher-Specified Covariates (https://arxiv.org/abs/2606.03029)
- **What's New**: 본 논문은 컴퓨터 사회과학의 핵심 목표인 언어의 변화를 이해하는 데 있어, 연구자가 지정한 공변량(covariates)을 포함하여 가설 생성을 조정하는 "조건부 가설 생성(conditional hypothesis generation)" 프레임워크를 제안합니다. 이는 표본 간의 차이를 위주로 한 기존의 방법과는 달리, 연구자가 관심 있는 하위 집단(subgroups) 내에서의 차이를 강조합니다. 따라서, 공변량을 무시했을 때 발생할 수 있는 혼란(confounds)에 따른 결과 왜곡을 방지할 수 있습니다.

- **Technical Details**: 논문에서 제안하는 두 가지 방법은 경제학에서 영감을 받아 구체적으로 설계되었습니다. 첫 번째는 상호작용-라쏘(interaction-lasso)로, 이는 공변량과의 상호작용(feature-covariate interactions)을 통해 특성이 특정 집단 내에서만 차별화되는 경우를 탐지할 수 있도록 합니다. 두 번째 방법인 평균화-가중치-라쏘(demeaned-reweighted-lasso)는 공변량 구간 내에서 피처와 결과를 평균화하여 하위 집단 내 변화를 분리하고, 저대표군의 기여도를 동등하게 하여 선택 과정에 공정성을 제공합니다.

- **Performance Highlights**: 합성 실험 결과, 두 방법 모두 글로벌 기준선(global baselines)보다 우수한 성과를 보였으며, 실제 데이터셋에서도 공변량을 고려한 가설 생성이 전문가에 의해 더 유용하다고 평가되었습니다. 특히, 평균화-가중치-라쏘는 불균형 수준에 걸쳐 약간의 차이를 보이는 가설을 복원하는 데 유일한 방법으로 자리잡았습니다. 이러한 결과는 연구자가 지정한 특정 조건에서 생성된 가설이 더 명확하고 유용하다는 것을 입증합니다.



### Spike-Aware C++ INT8 Inference for Sparse Spiking Language Models on Commodity CPUs (https://arxiv.org/abs/2606.03026)
Comments:
          11 pages, 7 tables

- **What's New**: 이번 논문은 spiking language models에서 드러나는 activation sparsity를 CPU inference runtime으로 활용하는 방안을 제시하고 있습니다. SymbolicLight V1 spike-gated 언어 모델을 기반으로 하여, sparse binary spike states를 실행 원시로 다루는 C++ CPU inference runtime을 구현했습니다. 이 연구는 sparse paths의 퀀타이제이션(quantization) 및 메모리 레이아웃에 대한 새로운 접근 방식을 탐구합니다.

- **Technical Details**: 연구에서 제안하는 C++ CPU inference runtime은 weight loader, mixed row/column 메모리 레이아웃, AVX2/FMA 커널, per-channel symmetric INT8 quantization을 포함합니다. AMD Ryzen 7 5800X에서 이 구현은 186k-step, 874M-parameter INT8 export로 단일 스레드에서 22.63 tokens/s, 4개 스레드에서 47.90 tokens/s로 성능을 보였습니다. 이 논문은 spike-aware execution이 sparse language 모델의 CPU 처리량과 메모리 동작을 향상시킬 수 있음을 보여줍니다.

- **Performance Highlights**: 초기 성능 테스트에서, dense baselines에 비해 WikiText-2 perplexity와 같은 Quality 지표는 낮습니다. 그러나, 논문의 결과는 CPU에서의 runtime 기여를 강조하며 실질적인 throughput 향상이 있음을 보여줍니다. 긴 형태의 프롬프트를 처리하는 성능은 스레드 수에 따라 크게 개선되었고, 이 연구는 robot과 local interactive agents를 대상으로 한 앞으로의 연구 방향을 제시합니다.



### Hallucinations as Orthogonal Noise: Inference-Time Manifold Alignment via Dynamic Contextual Orthogonalization (https://arxiv.org/abs/2606.03022)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 환각 현상(hallucination)을 기하학적 틀에서 해결하는 방법을 제안합니다. 환각은 맥락적 사실이나 논리적 제약과 일치하지 않는 콘텐츠의 생성을 의미하며, 특히 세맨틱 매니폴드(semantic manifold)와의 정사각형(noise) 반대 방향으로 정보가 전파되는 경우에 발생한다고 주장합니다.

- **Technical Details**: 이 연구에서는 Dynamic Contextual Orthogonalization(DCO)라는 추론 시간 개입(intervention) 방법을 도입합니다. DCO는 레이어에서의 입력 잔여 흐름(input residual stream)을 동적인 맥락 앵커(context anchor)로 활용하여 주의(attention) 머리 출력(output)에서 패턴을 잡아내고, Z-Score 억제(z-score suppression) 메커니즘을 통해 맥락에 정렬된 업데이트와 비대칭 노이즈를 구별합니다. DCO는 모형의 파라메트릭 지식 조회 능력을 유지하면서도 환각을 효과적으로 감소시키도록 설계되었습니다.

- **Performance Highlights**: DCO는 Llama-3-8B 및 70B 모델에서 다양한 벤치마크(XSum, NQ-Swap 및 IFEval)에서 최첨단 개입 방법들에 비해 우수한 성능을 보였습니다. 또한, 트리비아 QA(TriviaQA)와 진실한 QA(TruthfulQA)와 같은 지식 집약적인 작업에서도 높은 성능을 유지하여 환각 억제와 파라메트릭 지식 보존 간의 균형을 잘 맞추었습니다. 이러한 결과는 DCO가 기하학적 제약을 시행하는 데 있어 효과적이며 계산적으로 효율적임을 입증합니다.



### Reproducibility is the New Copyleft: Defining AGI-oriented Reproducible Builds (https://arxiv.org/abs/2606.03019)
Comments:
          Accepted at AGI-26. To appear in the proceedings (Springer LNCS)

- **What's New**: 이 논문은 AGI(Artificial General Intelligence) 시스템에서 copyleft의 기능적 대응책을 제안하며, 재현 가능한 빌드(reproducible builds)를 중심으로 해야 한다고 주장합니다. 이는 소스 코드와 객체 코드 간의 관계가 깨진 현대 인공지능 시스템에 대해 재현성을 보장하는 새로운 프레임워크를 개발하는 것입니다. 이러한 접근법은 오픈 소스 AI 정의(Open Source AI Definition) 및 모델 개방성 프레임워크(Model Openness Framework)와 같은 기존 개념에서 발전합니다.

- **Technical Details**: copyleft는 사용자가 자유롭게 수정하고 재배포할 수 있도록 하는 라이선스 기법이며, GNU 일반 공중 라이선스(GNU General Public License)가 그 대표적인 예입니다. 이 라이선스는 소스 코드와 객체 코드 간의 결정론적 관계를 전제로 하여, खुले 소스의 의미를 정의합니다. 하지만 AI 모델에서는 이러한 동등성이 붕괴되어, 결과적으로 사용자가 소스 코드를 수정하거나 재배포하는 것이 사실상 불가능해집니다.

- **Performance Highlights**: 본 논문은 AGI 시스템의 효과적인 재현을 보장하기 위해 다음의 일곱 가지 요구사항을 정의합니다. 또한, 모델 상호 연결 프로토콜(Model Context Protocol)과 유사한 AI 간 연결 메커니즘은 전통적인 copyleft 라이선스로는 다룰 수 없는 새로운 동적 링크 계층을 형성한다고 주장합니다. 결과적으로, 본 논문은 재현 가능한 빌드를 위한 거버넌스 구조와 프로토콜을 통한 관리 프레임워크를 제안합니다.



### ConTraIRL: Factorized Contrastive Abstractions for Transferable IRL (https://arxiv.org/abs/2606.03017)
- **What's New**: 본 논문은 역 강화 학습(Reverse Reinforcement Learning, IRL)에서 정책이 새로운 환경의 조합에 일반화될 수 있도록 하는 도구인 ConTraIRL(Contrastive Abstractions for Transferable Inverse Reinforcement Learning) 프레임워크를 제안합니다. 이 프레임워크는 다중 인코더 아키텍처를 활용하여 동적(dynamics) 및 목표(goal)에 대한 잠재 표현(latent representations)을 구분하여 학습합니다. 이러한 방식은 복합적인 보상 전이를 가능하게 하고, 예측된 보상이 환경에 따라 안정적으로 작동하도록 보장합니다.

- **Technical Details**: ConTraIRL은 dual-encoder 구조를 사용하여 동적 및 목표 요소를 각각의 잠재 공간(latent space)으로 매핑합니다. 이 구조는 목표와 동적 상태 간의 불필요한 상관성을 방지하며, temporal alignment를 통해 동적 인코더는 목표와 무관한 구조를 학습하게 됩니다. 또한, 각 환경의 일부 전문가 상태를 통해 몇 샷(multiple few-shot) 학습을 도입하여 새로운 맥락에서의 보상 회수를 보장합니다.

- **Performance Highlights**: MuJoCo 벤치마크를 기반으로 한 실험 결과에서는 ConTraIRL이 중첩되지 않은 동적-목표 조합에서 효과적으로 보상을 복구하고 전이 안정성을 높이는 것으로 나타났습니다. 모든 환경에서 ConTraIRL은 기존 IRL 기법에 비해 우수한 성능을 보여주었고, 샘플 효율성과 보상 회복 능력이 향상되었습니다. 이 연구는 제한된 감독 하에서 알려진 맥락 요소 간의 조합 가능성에 중점을 둡니다.



### MUSE: A Unified Agentic Harness for MLLMs (https://arxiv.org/abs/2606.03005)
- **What's New**: 이 논문은 MUSE라는 멀티모달 정합 구조 실행 도구를 제안하며, 이는 훈련이 끝난 멀티모달 대형 언어 모델(MLLM)을 감싸고, 재훈련 없이 작업 표현, 시각 처리, 인지 도구 사용, 구조화된 파싱, 결정 확인 및 확인자를 통한 수리 모듈을 포함하는 모듈식 파이프라인을 제공합니다. 기존의 연구가 모델 자체를 개선하는 데 초점을 맞췄다면, MUSE는 모델을 수정하지 않고 주변 실행 구조를 최적화하여 성능 향상을 이끌어냅니다. 이를 통해 기존 모델의 한계를 넘어 새로운 가능성을 모색합니다.

- **Technical Details**: MUSE는 Frozen MLLM을 개선하기 위해 설계된 모듈식 추론 파이프라인을 포함합니다. 이 파이프라인은 인식 및 도구 사용에서 시작하여 구조적 파싱 및 결정적 검증을 거쳐 확인자 유도 수리까지의 전체 추론 경로를 다룹니다. MUSE는 특정 실패 모드를 해결하는 다양한 컴포넌트를 명시적으로 설계하였으며, 백박스 환경에서 모델을 다루어 성능 향상이 모두 하니스 레벨의 개선에 기인함을 보장합니다. 실험에서는 VSP-Grid, BLINK-Jigsaw, CoMT, TIR-Bench 등의 다양한 벤치마크에서 평가하였습니다.

- **Performance Highlights**: MUSE는 네 가지 최첨단 MLLM (GPT-4o, GPT-5.4, Claude Haiku 4.5, Claude Opus 4.7)에서 일관된 성능 향상을 보여주었으며, 특히 도전적인 예제에서 두드러진 개선을 보였습니다. 예를 들어, GPT-4o의 Word Search에서 정확도가 3%에서 21%로 향상되었습니다. 분석 결과, MLLM의 실패는 종종 하니스 레벨의 부족에서 비롯되며, 확인자 유도 수리를 통해 이들 문제를 해결할 수 있습니다.



### Exact equivariance, kept through training, buys zero-shot generalisation across the symmetry group (https://arxiv.org/abs/2606.03003)
Comments:
          92 pages, 11 figures. Core paper plus an extended results-log appendix and a forward-looking theory supplement. All experiments are laptop-scale (CPU/MPS), fully seeded and deterministic

- **What's New**: 이번 논문에서는 그룹 대칭을 일반화하여 데이터 효율성과 제로샷 제너럴리제이션(zero-shot generalization)에 대한 새로운 가능성을 제시합니다. 저자들은 동적 모델을 특정 그룹에 대해 정방향으로 학습시키고, 이러한 대칭을 통해 다양한 환경에서 예측 성능을 개선할 수 있음을 입증하였습니다. 또한, 제안된 방법론이 실제 환경에서도 효과적으로 작용함을 확인하였습니다.

- **Technical Details**: 이번 연구는 동적 시스템을 다룰 때 대칭성을 수학적으로 정의하여, 그 학습 과정에서 그룹의 모든 방향에 대해 예측 오차(relMSE)가 일정하다는 사실을 제시합니다. 제안된 모델은 equivariant encoder와 predictor를 기반으로 하며, 이들이 결합되어 학습 과정에서 대칭성을 유지하게 됩니다. 특히, closed-loop 제어에서도 각 방향에 관계없이 정확한 성능을 보이는 것을 확인하였습니다.

- **Performance Highlights**: 조사 결과, Equivariant 모델은 비대칭적 모델과 비교하여 4.5-7.4배 더 적은 예측 오차를 기록했습니다. 또한, POC 체계에서 2D 및 3D 환경 모두에서 오차가 규칙적이고 통계적으로 안정적임을 입증하였습니다. 전체 실험은 노트북 수준의 CPU/MPS에서 진행되었으며, 이 모델이 이전의 기준을 능가함을 뒷받침하는 고무적인 결과를 도출했습니다.



### How Quantization Changes Interpretable Features: A Sparse Autoencoder Analysis of Language Models (https://arxiv.org/abs/2606.03002)
- **What's New**: 이 연구에서는 대규모 언어 모델의 배포를 위한 표준 경로인 quantization에 대해 다룹니다. 특히, full-precision 모델에서 추출한 sparse autoencoder (SAE) feature가 quantized 모델에서도 여전히 신뢰할 수 있는지를 평가합니다. 연구 결과는 기존의 행동 기준만으로 interpretability의 신뢰성을 완전히 평가할 수 없음을 강조합니다.

- **Technical Details**: 이 연구에서는 Pythia-70M과 Gemma-2-2B 모델을 사용하여 INT8에서 INT4까지 bit-width를 변경하며 특징 생존률을 분석합니다. 추출된 SAE features는 quantization 과정에서 체계적으로 저하되며, 주로 높은 peak activation 관련 feature는 보다 안정적으로 유지되는 경향을 보입니다. 또한, perplexity 지표가 feature의 손상을 놓칠 수 있음을 강조합니다.

- **Performance Highlights**: 연구 결과에 따르면, Pythia-70M에서는 ACTIVE feature의 62.4%가 INT6에서 유지되고, Gemma-2-2B에서는 51.3%가 유지됩니다. 이와 함께, quantization과 magnitude pruning이 비슷한 feature 집합에 손상을 주는 경향이 있음을 보여줍니다. 마지막으로, AUC 값은 0.92에서 0.97로 예측 정확도가 매우 높음을 나타냅니다.



### Patcher: Post-Hoc Patching of Backdoored Large Language Models (https://arxiv.org/abs/2606.02995)
Comments:
          To appear in the USENIX Security Symposium, 2026

- **What's New**: 본 논문에서는 대형 언어 모델이 jailbreak backdoor 공격에 취약하다는 점을 언급합니다. 기존 방어 체계는 포괄적인 공격 정보나 여러 개의 트리거된 예제가 필요해, 단일 실패 사례만으로는 비현실적인 상황에 처해 달리 대처할 비법이 없었습니다. 이 문제를 해결하기 위해) Patcher라는 새로운 후처리(defense) 프레임워크를 제안합니다.

- **Technical Details**: Patcher는 두 단계로 작동합니다. 첫째, response-conditioned gradient-based saliency 점수를 계산하고 적응형 클러스터링(adaptive clustering)을 적용하여 트리거를 로컬라이즈(localize)합니다. 둘째, KL-divergence 제약 조건을 통해 트리거-응답(trigger-response) 연관성을 끊으면서도 정상적인 작업 유용성과 비트리거된 공격에 대한 강건성을 유지하는 방식으로 모델을 수정(patch)합니다.

- **Performance Highlights**: 저자들은 Patcher의 성능을 여러 백도어 공격 전략에 대해 광범위하게 평가하였으며, Patcher가 트리거를 성공적으로 로컬라이즈하고 백도어를 중화(neutralize)하는 동시에 모델 유용성을 유지함을 보여줍니다. 또한 Patcher는 우리의 방어를 회피하기 위해 설계된 적응형 공격에 대해서도 강건성을 보여 주었습니다.



### Pretraining Language Models on Historical Tex (https://arxiv.org/abs/2606.02991)
- **What's New**: TypewriterLM은 1913년 이전의 영어 텍스트에 한정된 72억 파라미터의 역사적 언어 모델입니다. 이 모델은 역사적 언어 모델을 구축하기 위한 여러 가지 문제를 해결하기 위해 개발되었습니다. 이를 위해 TypewriterCorpus라는 540억 토큰의 역사적 말뭉치를 제작하고, 시기적 일관성 있는 후속 처리 파이프라인과 평가를 설계했습니다.

- **Technical Details**: TypewriterCorpus는 다양한 아카이브 자료와 언어적 주석이 포함된 출처에서 수집된 데이터로 구성됩니다. 또한, lexically grounded instruction tuning이라는 후속 훈련 프레임워크를 도입하여 응답이 역사적 출처 문서에 직접 연결되도록 제약을 설정했습니다. 이를 통해 History-LIMA와 History-SelfInstruct라는 두 개의 역사적 지시 조정 데이터셋이 구성되었습니다.

- **Performance Highlights**: TypewriterLM은 기본 모델과 지시 조정 모델 모두에서 경쟁력 있는 성능을 보여주며, 기계 학습과 인문학에서의 미래 연구를 지원합니다. 모델의 평가를 위해 History-Event라는 벤치마크를 도입하여 시간적 일관성을 검증하고, 기계 모델이 과거 사건에 대한 응답에서 더 큰 놀라움을 느끼도록 설계되었습니다. 이는 역사적 컷오프가 모델에 반영되었음을 시사합니다.



### Towards Compact Autonomous Driving Perception with Balanced Learning and Multi-sensor Fusion (https://arxiv.org/abs/2606.02979)
Comments:
          This work has been accepted for publication in IEEE Transactions on Intelligent Transportation Systems. this https URL

- **What's New**: 이 논문에서는 다양한 자율주행 인지 과제를 단일 패스로 처리하는 새로운 compact deep multi-task learning 모델을 소개합니다. 이 모델은 semantic segmentation, depth estimation, LiDAR segmentation, bird’s eye view projection을 동시에 수행할 수 있으며, 다른 모델의 지원 없이 독립적으로 작동합니다. 또한, 불균형 학습 문제를 해결하기 위한 adaptive loss weighting 알고리즘을 제공합니다.

- **Technical Details**: 모델은 4개의 RGB 카메라, 4개의 DVS, 1개의 LiDAR를 활용하여 다양한 입력 모달리티를 처리하고 여러 센서의 데이터를 융합합니다. 이 과정에서 데이터 전처리와 센서 융합 기술을 적용하여, 동적으로 변화하는 환경에 대한 더 나은 이해를 달성합니다. 또한, Gradient Normalization (GradNorm) 알고리즘을 수정하여 학습 과정의 균형을 맞추고 성능 향상을 도모합니다.

- **Performance Highlights**: 모델의 성능은 ablation study와 비교 연구를 통해 입증되었습니다. 이를 통해 우리는 더 적은 파라미터로도 더 나은 성능을 유지하며, 빠른 추론 속도와 적은 GPU 메모리 사용량을 달성했습니다. 실험은 3개의 CARLA 시뮬레이션 데이터셋과 1개의 실제 nuScenes-lidarseg 데이터셋에서 일관된 결과를 보여주었습니다.



### Glass Box at Orbit: A Constitutional AI Verification Framework for Trustworthy Autonomous CubeSat Intelligenc (https://arxiv.org/abs/2606.02967)
Comments:
          12 pages, 2 figures, 2 tables, 32 references. Paper 1 of the Project October series on autonomous orbital intelligence

- **What's New**: 이 논문은 자율 AI 시스템의 안전성을 보장하기 위한 새로운 방법인 Glass Box를 소개합니다. 현재 Microsoft, AWS 등 다양한 회사가 궤도 데이터 센터를 설계하고 있지만, 이들 시스템에서의 AI 결정의 검증 및 안전성이 부족합니다. Glass Box는 자율 AI 결정이 잘못될 경우 이를 사전에 차단하는 체계적인 장치를 제공합니다.

- **Technical Details**: Glass Box는 모든 AI 결정 후보 행동을 검증하기 위해 물리적 제약 조건과 LTL (Linear Temporal Logic) 안전 불변성을 바탕으로 하는 시스템입니다. 이 시스템은 AI 정책에서 후보 행동을 추출하여, 각 행동이 미리 정의된 여섯 가지 물리적 제약 및 일곱 가지 LTL 안전 불변성에 적용되는지를 평가합니다. 이를 통해 Glass Box의 검증 오버헤드는 제약 수에 비례하여 선형적(O(Nc))이라는 이론적 증명이 제공됩니다.

- **Performance Highlights**: 연구 결과, Glass Box는 CubeSat급 우주선의 자율 지능 아키텍처인 Project October에서 성공적으로 시연되었습니다. 이 시스템은 부적절한 추론 요청을 차단하는 실제 사례를 담고 있으며, 모든 자율 결정을 위한 무결한 감사 로그를 기록합니다. 따라서 궤도 컴퓨팅이 증가함에 따라, 이러한 런타임 검증 시스템은 미션의 안전성을 보장하기 위한 필수적인 요소가 될 것입니다.



### Hand Trajectory Fusion for Egocentric Natural Language Query Grounding (https://arxiv.org/abs/2606.02962)
Comments:
          Accepted for the poster session at the Egocentric Vision (EgoVis) Workshop in Conjunction with CVPR 2026

- **What's New**: 이번 논문에서는 Egocentric Natural Language Query(NLQ) grounding을 위한 새로운 접근 방식을 제안합니다. 기존의 방법들은 비디오의 외관과 쿼리를 결합하지만, 손 움직임을 무시했습니다. 그 결과, 손-객체 조작 순간에 대한 쿼리 처리에서 개선된 성능이 나타났으며,+2.54 R1@IoU=0.3을 기록했습니다.

- **Technical Details**: 본 연구는 손 관절의 전통적 속성에 대한 정보를 포함하기 위해 손 경로 인코더를 사용합니다. 이 인코더는 손 스켈레톤을 비디오-텍스트 특성과 융합하여 고도로 의미 있는 동작 함수로 변환합니다. 최종적으로, Temporal Segment Prediction 모듈이 쿼리와 가장 잘 일치하는 정답 범위를 예측합니다.

- **Performance Highlights**: Ego4D NLQ v2 검증 데이터셋에서 수행한 결과, 손-객체 상호작용 쿼리에 대해 가장 큰 성과(+2.54 R1@IoU=0.3)와 양(quantity)/상태(state) 쿼리에 대해서도 유의미한 성과(+4.32 R1@IoU=0.3)가 확인되었습니다. 이는 손 경로가 단순한 외관 이상으로 중요한 단서를 제공한다는 것을 시사합니다.



### Echelon: Auditable Aggregate-Only Language-Model Adaptation Across Privacy Boundaries (https://arxiv.org/abs/2606.02958)
- **What's New**: 본 논문에서 제안하는 Echelon은 장치 레벨 모델 상태의 비공유를 시스템 불변 조건으로 설정한 훈련 아키텍처입니다. 이는 기존의 분산 훈련 시스템과는 달리 각 기기의 업데이터를 관리하는 고유한 접근 방식을 사용하여 정보 흐름에 대한 규칙을 명확하게 정의합니다. 이러한 저작물은 에치론의 구조적 개인정보 보호가 검토 가능한 방식으로 설계되었음을 강조합니다.

- **Technical Details**: Echelon은 장치, 경계 및 글로벌 플레인으로 구성된 세 가지 레벨의 상태를 유지합니다. 기기는 경계 기준에 대해 로컬 최적화를 수행하며, 이러한 업데이트는 세미 비동기식으로 클리핑 및 가중치 조정을 통해 집계됩니다. 정보 흐름 계약은 장치 레벨의 파라미터가 경계를 넘어 전달되지 않도록 강화된 규칙을 설정하여, 특히 데이터 비독립성과 사용자 변화에 견디도록 설계되었습니다.

- **Performance Highlights**: 1B-파라미터 LoRA 적응에서 Echelon은 배급에 있어 경쟁력을 입증하였습니다. 테스트에서는 기존의 분산 훈련 방법에 비해 우수한 성능 지표와 빠른 처리 속도를 자랑하며, WAN 지연이나 비독립적 데이터 파티셔닝 시에도 품질 저하가 2.2% 이내로 제한됩니다. Echelon의 감사 추적 기능은 시스템의 규정 준수를 명확히하며, 전 세계에서 데이터를 안전하게 처리하는 방법을 제시합니다.



### Fast-dLLM++: Fréchet Profile Decoding for Faster Diffusion LLM Inferenc (https://arxiv.org/abs/2606.02955)
Comments:
          Initial version accepted at Workshop on Structured Probabilistic Inference & Generative Modeling, ICML 2026

- **What's New**: 이번 연구에서는 데이터 토큰 생성을 병렬로 처리할 수 있는 Diffusion large language models의 발전을 다룹니다. 특히 Fast-dLLM의 한계를 극복하기 위한 새로운 접근법인 Fast-dLLM++를 제안합니다. Fast-dLLM++는 Fréchet profile decoding을 도입하여 진일보한 정확도와 속도를 구현합니다.

- **Technical Details**: Fast-dLLM++는 훈련이 필요 없고, 고유한 소믈리칭 요건을 제공하여 병렬 커밋 세트를 전체 정렬된 신뢰도 프로파일에서 선택합니다. 이는 단일 최악의 신뢰도를 사용하는 기존 Fast-dLLM의 품질을 개선하며, 비균형적인 신뢰도의 경우 'heterogeneity bonus'를 추가합니다. 이 방법은 기존의 Fast-dLLM의 모델 구조와 캐시 구현에 전혀 영향을 주지 않아 쉽게 교체할 수 있습니다.

- **Performance Highlights**: GSM8K, MATH, HumanEval, MBPP와 같은 다양한 벤치마크에서 LLaDA-8B 모델을 사용한 실험 결과 Fast-dLLM++의 채택이 실질적인 성과를 보여주었습니다. 프로파일 인식 선택 기법을 통해 안전한 병렬 처리를 실현하여 정확도-처리량 (accuracy-throughput) 경계를 개선하고, 최대 37% 더 높은 처리량을 기록하며 비교 가능한 정확도를 유지했습니다.



### SCOPE: Real-Time Natural Language Camera Agent at the Edg (https://arxiv.org/abs/2606.02951)
Comments:
          9 pages, 4 figures, 6 tables. Accepted at HRI '26 (21st ACM/IEEE International Conference on Human-Robot Interaction), Edinburgh, Scotland, March 16--19, 2026. Code: this https URL

- **What's New**: 이 논문에서는 로봇 공학에서 자연어 기반의 언어 주도 에이전트를 배포하기 위한 새로운 평가 방법인 SCOPE(Simulation and Camera Operations for Perception and Evaluation)를 제안합니다. 이 시스템은 Blender 기반의 시뮬레이션 환경과 실제 PTZ(팬-틸트-줌) 카메라에서 작동하며, 언어 모델과 감지 및 제어 도구를 결합하는 것을 목표로 합니다. SCOPE는 536개의 작업으로 구성된 벤치마크를 통해 높은 시뮬레이션과 실제 환경 간의 전이 가능성을 보장합니다.

- **Technical Details**: SCOPE는 분리된 설계를 채택하여, Compact SLM이 고수준의 계획자로서 카메라 제어 및 감지 쿼리와 reasoning을 조정합니다. 시각 이해는 호출 가능한 도구로 노출된 Lightweight VLM에 위임되어, 실제 시간의 엣지 지연에서 visual perception과 tool-based control을 동시에 달성하기 어렵다는 현실을 반영합니다. 이 시스템은 각 요청 후 VLM으로부터 얻은 결과를 활용해 임무를 반복적으로 수행하는 방식으로 작동합니다.

- **Performance Highlights**: SCOPE의 평가 결과, 강력한 SLMs를 사용 시 헐리케인 문제(hallucinations)를 줄이고 도구 경로 설정(tool routing)을 개선하여 신뢰할 수 있는 닫힌 루프(closed-loop) 동작을 생성할 수 있다는 것을 확인했습니다. 또한, Mixture-of-Experts 모델을 통해 비교적 적은 메모리 사용과 지연 시간으로 밀집 대안(dense alternatives)에 비해 일관된 성능을 나타냈습니다. 양자화(Quantization)를 통해 추가적인 효율 향상이 이루어졌으며, 이는 실시간 PTZ 제어의 실제 심리학적 설계 포인트를 제시합니다.



### WRIT: Write-Read Intensive Trajectory Synthesis for Multi-Turn User-Facing Agents (https://arxiv.org/abs/2606.02908)
- **What's New**: 이 논문에서는 사용자 의도를 파악하고 대화 및 도구를 통해 필요한 정보를 수집하는 멀티 턴 사용자 대면 에이전트를 훈련시키기 위한 새로운 경로인 WRIT(Write-Read Intensive Trajectory Synthesis)를 제안합니다. 기존의 훈련 방법은 여러 사용자 요청을 긴 작업으로 구성하여 쓰기 집중적인 경로를 만들어왔지만, 이번 연구에서는 에이전트가 유의미한 증거를 수집하고 비교해야 한다는 새로운 과제가 있음을 강조합니다. WRIT는 이러한 쓰기 결정을 내리기 위해 읽기 중심의 복잡성과 쓰기 결정을 고려하여 다양한 훈련 경로를 생성합니다. 이로써 에이전트가 실제적인 대화 변화를 반영하도록 훈련 지침을 다양화하고, 사용자와 에이전트의 상호작용을 실시간으로 시뮬레이션하여 전체 훈련 경로를 만들어냅니다.

- **Technical Details**: WRIT는 멀티 턴 에이전트 훈련 경로를 생성하는 파이프라인으로, 각 작업의 쓰기 결정 수와 각 결정의 증거 부하를 고려합니다. 첫째, WRIT는 검증 가능한 결과를 도출하는 서비스 작업을 생성하며, 두 번째로는 사용자 요청을 표현하는 방식을 다양화하여 훈련 데이터가 현실적인 대화 행동을 반영하도록 합니다. 마지막으로, WRIT는 에이전트와 사용자가 각 작업을 수행하는 동안 실행 가능한 환경에서 이를 시뮬레이션하여 성공적인 상호작용을 완전한 훈련 경로로 유지합니다. 이러한 구조적 접근을 통해 에이전트는 단순히 긴 작업을 실행하는 것에 그치지 않고, 높은 정보 부하 상황에서 더 강력한 의사 결정을 내릴 수 있습니다.

- **Performance Highlights**: WRIT는 τ2	au^{2}-bench에서 강력한 합성 데이터 기준선에 비해 비약적인 성능 향상을 보였습니다. 단 2K의 훈련 경로로 4B 모델은 GPT-5.1 no-think를 초월하며, inference 시간 동안 사용하는 토큰 수를 크게 줄였습니다. 이는 잘 구성된 작은 훈련 경로가 훨씬 더 큰 비구조적 데이터셋보다 더 유능하고 신뢰할 수 있는 에이전트를 생성할 수 있음을 보여줍니다. WRIT의 성능 향상은 다른 사용자 행동 다양화 및 읽기 집중 작업 합성의 독립적인 기여에 의해 가능해졌습니다.



### Linear Probes Detect Task Format, Not Reasoning Mode in Language Model Hidden States (https://arxiv.org/abs/2606.02907)
Comments:
          Accepted in the 6th Workshop on Trustworthy NLP, ACL 2026

- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)의 숨겨진 상태를 선형 탐색(linear probing)하여 서로 다른 추론 유형이 모델에 의해 어떻게 학습되는지를 살펴보았습니다. Qwen3-14B 모델을 사용하여 LogiQA 2.0, ARC-Challenge, αNLI와 같은 다양한 벤치마크에서 100%의 교차 검증 정확도를 달성했으나 이 결과는 형식적 요인(format confounds)에 의해 주도된다음을 발견했습니다. 키워드로는 'residualizing'과 'trace-anchor similarity'를 포함하여, 모델의 고유한 추론 전략이 아닌 형식 때문이라는 점을 밝히고 있습니다.

- **Technical Details**: 연구 방법론은 다섯 단계로 구성되어 있습니다: 1) 다중 출처 데이터셋 구축, 2) 숨겨진 상태 추출, 3) 계층별 선형 프로빙, 4) 형식 요인 분석, 5) 무작위 방향 제어를 통한 인과적 조정. 다양한 추론 유형에 대해 설계된 벤치마크에서 샘플링하여 균형 잡힌 세 클래스를 가진 데이터셋을 구축하였으며, 각 추론 모드에 대해 의도된 레이블을 부여했습니다. 모델 Qwen3-14B의 분석을 통해 내부 상태와 출력 신뢰도 측정을 하였습니다.

- **Performance Highlights**: 모델의 선형 프로빙 결과는 모든 직무 유형에서 86%의 정확도를 보였으나, 추적 모드의 일치는 42.5%로 저조했습니다. 이를 통해 모델이 특정 작업에 대해 독립적인 추론 전략을 적용하는 것이 아니라 공통된 추론 방식을 사용하고 있음을 알 수 있습니다. 연구 결과는 또한 형식 요인 간섭(format confounding) 제거와 무작위 방향 제어가 추론 해석 가능성의 표준 실천이 되어야 함을 강조합니다.



### Scalable Uncertainty Quantification for Extreme Weather Forecasting via Empirical Neural Tangent Kernels (https://arxiv.org/abs/2606.02886)
Comments:
          Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD '26)

- **What's New**: 본 논문은 기후의 극단적인 사건 발생 시에도 적용할 수 있는, Neural Tangent Kernel 기반의 불확실성 정량화(NTK-UQ)를 제안합니다. 딥 러닝 기반의 날씨 모델들이 높은 정확도로 빠르게 예측을 하지만, 신뢰할 수 있는 불확실성 추정이 부족해 의사결정을 어렵게 하는 문제를 해결하기 위해 이 방법이 필요합니다. 특히, NTK-UQ는 모델의 마지막 층에서 특징을 분석하여 통계적 기반의 기존 방법들보다 향상된 예측 결과를 제공합니다.

- **Technical Details**: NTK-UQ는 마지막 층의 자가 특징(embeddings)을 사용하여 Gaussian Process 이론을 통해 불확실성을 정량화합니다. 이 과정에서 Independent Component Analysis(ICA)와 Singular Value Decomposition(SVD) 사이의 선택 규칙을 기반으로 적절한 분해 방법이 결정될 수 있습니다. 이러한 의사 결정은 네트워크 아키텍처와 분해 방법에 따라 다르게 나타나며, 데이터 주도 선택 규칙을 적용하여 최적의 방법을 신속하게 찾을 수 있습니다.

- **Performance Highlights**: NTK-UQ는 기존의 split conformal prediction에 비해 90% 신뢰구간에서 31%에서 37% 더 날카로운 예측 구간을 생성하며, 극단적인 사건의 심각도에 따라 조정 가능한 예측 구간을 생성합니다. 네 가지 아키텍처(FourCastNetV2, Pangu-Weather, Aurora, AIFS)에 대해 평가하며, 불확실성 판별 품질이 아키텍처에 따라 차등적으로 변하는 것을 실험을 통해 입증하였습니다. 이러한 결과는 NTK-UQ의 이론적 예측을 실제로 확인할 수 있는 기회를 제공합니다.



### Are we really tilting? The mechanics of reward guidance in flow and diffusion models (https://arxiv.org/abs/2606.02884)
- **What's New**: 이 논문에서는 보상 가이던스 알고리즘(reward guidance algorithms)의 한계를 탐구하며, 이러한 방법들이 보상 해킹(reward hacking)에 취약하다는 점을 지적합니다. 저자들은 대부분의 실제 구현에서 발생하는 근사 문제에서 보상 해킹의 근본 원인이 발생한다고 주장하며, 특히 가우시안(targets) 및 가우시안 혼합 대상들에서 이 문제가 어떻게 발생하는지를 분석합니다. 또한, 고유한 보상 감소 일정(closed-form reward damping schedule)을 제안하여 이러한 문제를 해결할 수 있는 방법을 제시합니다.

- **Technical Details**: 보상 가이던스는 학습된 분포에서 샘플을 생성하는 것이 아니라 보상 기울기가 있는 측정값에서 샘플을 생성하는 과정입니다. 이 논문에서는 Doob h-transform을 이용하여 보상 가이던스 문제를 해결하는 프레임워크를 설명하며, finite-particle plug-in estimator의 문제를 통해 목표 모드 선택과 같은 일반적인 실패 모드를 명확히 합니다. 또한, 이 시스템은 보상을 효과적으로 최적화하는데 필요한 메커니즘을 제안합니다.

- **Performance Highlights**: 저자들은 제안된 보상 감소 일정을 사용하여 Gaussian 혼합 대상, 2D 체크무늬 및 FLUX.1 텍스트-이미지 생성에서 실험을 수행합니다. 실험 결과, 이론적 통찰이 실제 상황에서도 유효함을 확인하였으며, 베스트-오브-n 샘플링(best-of-n sampling)의 역할이 중요함을 강조합니다. 이러한 연구는 고품질 샘플 생성을 위한 보상 가이던스 접근법에 중요한 기여를 할 것으로 기대됩니다.



### LLM-Assisted Reranking to Operationalize Nuanced Objectives in Recommender Systems (https://arxiv.org/abs/2606.02883)
Comments:
          30 pages total; 11 pages, 5 figures, 2 tables (main text); 19 pages, 11 figures, 9 tables (appendix)

- **What's New**: 이번 연구에서는 LLM(large language models)을 활용한 추천 시스템이 정치적 콘텐츠에 대한 노출을 어떻게 재구성하는지를 조사합니다. 저자들은 사용자 시청 기록을 기반으로 한 LLM 기반의 재순위 매김이 개인화된 추천을 어떻게 향상시키는지를 분석하며, 이를 통해 정치적 극단주의 콘텐츠에 대한 노출을 확대할 수 있는 위험성을 실험적으로 입증합니다. 연구에서는 기계 학습 기반의 접근 방식과 사교적 가치 제약을 결합한 새로운 추천 구조를 제안합니다.

- **Technical Details**: 이 연구는 제로샷(zero-shot) 접근법을 사용하여 YouTube 추천 내용을 재정렬하는 데 LLM을 통합한 추천 파이프라인을 개발했습니다. 기존 추천 시스템과의 차별점은 LLM이 정황적 추론(contextual reasoning)의 적용을 가능하게 하여, 이론적 기준 이하로 콘텐츠의 탐색과 노출을 조정하는 것입니다. 이 과정에서 정치적 콘텐츠의 선정 및 재구성에서 발생할 수 있는 사회적 손상을 줄이기 위한 명확한 제어 전략이 적용되었습니다.

- **Performance Highlights**: 실험 결과, LLM의 재순위 매김이 개인화된 추천을 증가시키는 반면, 주요 피험자들은 극단적이고 음모론적인 콘텐츠에 대한 노출이 증가하는 것을 확인했습니다. 그러나 경량화된 프롬프트 레벨 정규화가 극단적 콘텐츠의 노출을 줄이고 이념적 다양성을 증가시키는 데 기여했습니다. 따라서 LLM을 통한 추천 시스템은 콘텐츠 추천의 사회적 영향을 고려하도록 설계될 필요가 있음이 보였습니다.



### Adaptive Latent Agentic Reasoning (https://arxiv.org/abs/2606.02871)
- **What's New**: 이번 논문에서는 Adaptive Latent Agentic Reasoning (ALAR) 프레임워크를 제안하여, LLM 에이전트의 비효율적인 텍스트 기반의 체인 오브 생각(Chain-of-Thought, CoT) 추론을 개선하고자 합니다. 기존의 LLM 에이전트는 의사결정 시 마다 긴 코드를 생성하는 경향이 있으며, 이로 인해 효율성이 떨어집니다. ALAR는 일상적인 상황에서는 간결한 잠재적 추론을 사용하고, 더 깊은 고찰이 필요한 경우에만 명시적인 CoT로 전환하도록 설계되었습니다.

- **Technical Details**: ALAR는 두 가지 주요 구성 요소로 이루어져 있습니다: 첫째, Action-Anchored Self-Distillation (AASD)은 에이전트가 환경과 상호작용하는 액션을 기반으로 잠재적 추론 모드를 학습합니다. 둘째, Adaptive Reasoning GRPO (AR-GRPO)는 작업 성공을 유지하면서 잠재적 추론을 활용하는 경우 보상하는 적응형 모드 선택을 학습합니다. 이러한 구조를 통해 ALAR는 두 가지 모드를 바탕으로 LLM 에이전트의 추론을 최적화합니다.

- **Performance Highlights**: 실험 결과, ALAR는 기존의 추론 토큰 압축 기법에 비해 더 나은 정확성-효율성의 균형을 달성하며, 검색에서는 최대 43.6%, 도구 사용에서는 최대 84.6%의 생성된 토큰 수를 줄일 수 있었습니다. 또한 ALAR는 작업의 정확성을 유지하면서 불필요한 텍스트 추론을 줄임으로써 에이전트가 어려운 결정 단계에서의 명시적인 고찰을 보존할 수 있도록 합니다.



### The Epi-LLM Framework: probing LLM behavioral priors through epidemiological agent-based models (https://arxiv.org/abs/2606.02867)
Comments:
          Submitted to American Journal of Epidemiology

- **What's New**: 이 논문에서는 감염병 역학에 대한 새로운 프레임워크인 Epi-LLM을 소개합니다. 이 프레임워크는 에이전트 기반 모델링(agent-based modelling), 실제 에피게임(real-life epigames), 대형 언어 모델(large language models, LLMs)의 통합으로 구성됩니다. Epi-LLM은 합성 사회가 감염병 발생 네트워크에서 동적으로 추론하고 적응하도록 하는 혁신적인 방법을 제공합니다.

- **Technical Details**: Epi-LLM의 시뮬레이션 프레임워크는 Starsim이라는 오픈 소스 플랫폼 위에 구축되었습니다. 이 플랫폼은 집단 모델링(population compartmentalization)이 아닌 개별 행동(behavior)과 동적 네트워크 상의 상호작용을 명시적으로 나타내는 에이전트 기반 모델을 사용합니다. 이 논문은 ABM의 구성을 더 발전시켜 LLM 에이전트를 사용하여 감염병 발생을 모델링하는 방법을 설명합니다.

- **Performance Highlights**: 논문에서는 다양한 아키텍처의 LLM 에이전트를 사용하여 피크 감염률을 감소시키는 데 성공했음을 보여줍니다. 에이전트의 격리 준수율은 15일 동안의 시뮬레이션 중 6일 차에 58-65%에 도달했습니다. 또한, LLM 아키텍처와 에이전트 행동이 전염병 역학에 미치는 영향을 탐구하여 실제 인구의 의사결정 패턴과의 유사성을 확인했습니다.



### Forgetting is Not Erasure: Recovering Latent Knowledge via Transport Keys (https://arxiv.org/abs/2606.02860)
Comments:
          Technical report showcasing results from transport keys

- **What's New**: 이 논문은 기존의 재현 문제로 간주되던 재앙적 망각(Catastrophic Forgetting)의 개념을 도전합니다. 연속 학습(Continual Learning) 환경에서, 저자들은 명백한 망각의 상당 부분이 내부 단계 간의 인터페이스 드리프트(Interface Drift)로 인해 발생한다는 것을 발견했습니다. 이는 특정 작업을 수행하는 데 필요한 계산이 영구적으로 지워지는 것이 아님을 보여줍니다.

- **Technical Details**: 저자들은 포스트 업데이트 네트워크의 초기 계산과 이전 네트워크의 계산을 결합하는 스티치드 평가 프로토콜을 통해 이 현상을 연구합니다. 또한, 운반 키(Transport Key)라는 시스템 레벨의 개념을 도입하여, 소량의 앵커 활성화를 이용해 인터페이스 정렬을 수행하는 방법을 설명합니다. 이 키는 연속으로 훈련된 네트워크 사이의 호환성을 복원하는데 사용됩니다.

- **Performance Highlights**: ResNet 스타일 네트워크에서 운반 키를 통해 Task A의 성능을 상당히 회복할 수 있음을 보여주었습니다. 또한 컴팩트 비전 변환기(Compact Vision Transformer)에서도 유사한 회복 패턴이 관찰되었습니다. 이 결과는 연속 학습이 단순히 가중치 변경을 막는 방법뿐만 아니라 잠재적 계산을 색인화하고 재접근하는 더 나은 메커니즘을 필요로 할 수도 있음을 시사합니다.



### Economy of Minds: Emerging Multi-Agent Intelligence with Economic Interactions (https://arxiv.org/abs/2606.02859)
- **What's New**: 본 연구는 경제적 신호를 통해 에이전트들이 중앙 집중된 관리 없이도 스스로 조직하고 진화할 수 있는 가능성을 탐구합니다. 에이전트들은 경매를 통해 행동 권한을 얻고, 지불을 교환하며, 환경으로부터 보상을 통해 부를 축적합니다. 이 시스템은 경제 선택을 통해 효과적인 에이전트를 선별하고, 비효과적인 에이전트를 제거함으로써 자율적으로 발전하는 구조를 만듭니다.

- **Technical Details**: 연구는 에이전트 경제를 모델링하며, 각 에이전트는 스스로의 조건과 정책에 따라 결정을 내립니다. 시스템은 계획과 적응이라는 두 가지 프로세스를 통해 작동합니다. 여기서 계획은 행동을 조정하고 크레딧을 할당하며, 적응은 에이전트 집단의 진화를 담당합니다.

- **Performance Highlights**: 이 연구의 경우, Economy of Minds (EoM) 시스템은 다섯 가지 디지털 에이전트 작업에서 뚜렷한 성과를 보였습니다. 예를 들어, 수학적 추론은 15.9%에서 57.0%로, 금융 연구 성과는 45.0%에서 60.0%로 향상되었습니다. 이는 에이전트 사회가 점진적으로 효과적인 작업 흐름을 자발적으로 구성하고 적응함을 보여줍니다.



### GRZO: Group-Relative Zeroth-Order Optimization for Large Language Model Fine-Tuning (https://arxiv.org/abs/2606.02857)
Comments:
          Preprint. Under review

- **What's New**: 새로운 연구에서 저자들은 GRZO(Group-Relative Zeroth-Order)라는 새로운 최적화 기술을 제안합니다. 이는 미니 배치의 각 예제마다 독립적인 perturbation을 생성하고, 이를 통해 예제 손실을 집계하여 경량화하고 있습니다. 이 방법은 경쟁자인 MeZO보다 더 나은 성능을 보이고, GPU 메모리를 23% 줄이면서도 더 높은 정확도를 얻을 수 있습니다.

- **Technical Details**: GRZO는 Flipout 스타일의 sign factorization을 활용해 pseudo-independent perturbations를 생성합니다. 이 방법은 두 번의 포워드 패스(step)로 이루어진 MeZO의 메모리 요구사항을 유지하면서 작동합니다. GRZO는 방향적으로 unbiased하며 배치 크기에 비례하여 분산을 줄이는 것으로, 비선형 수렴 경계(nonconvex convergence bound)가 MeZO보다 더 Tight하다는 것을 이론적으로 증명하고 있습니다.

- **Performance Highlights**: 실험 결과에 따르면 GRZO는 Llama3-8B에서 MeZO에 비해 평균 3%의 정확도 향상을 보였습니다. 또한 GRZO는 sparse, low-rank, 양자화된 Zeroth-Order 최적화 변형들과도 호환 가능하여, 이들 조합을 통해 추가적인 성능 향상이 가능합니다. 여러 언어 모델에 걸친 결과는 GRZO가 메모리 효율성과 처리 성능을 모두 개선한다는 것을 보여줍니다.



### Fixing FOLIO and MALLS: Verified Annotations and an LLM-assisted Framework to Focus Human Relabeling (https://arxiv.org/abs/2606.02837)
- **What's New**: 이번 연구는 자연어(Natural Language)에서 일차 논리(First-Order Logic)로의 정확한 번역(NL-to-FOL)을 지원하기 위한 데이터세트의 품질을 체계적으로 분석한 결과를 발표합니다. FOLIO와 MALLS 데이터셋에서 각각 39% 및 36%의 항목이 잘못된 FOL 형식화를 포함하고 있음을 발견했으며, 이로 인해 모델 평가에 큰 영향을 미친다는 사실이 확인되었습니다. 이러한 오류를 수정한 후 최첨단 LLM 모델의 정확도가 9%에서 22%까지 향상된 것으로 나타났습니다.

- **Technical Details**: FOLIO는 NL–FOL 쌍을 포함하는 NLI(Premises와 Conclusion) 벤치마크로, 이를 기반으로 데이터세트를 훈련 및 평가합니다. MALLS는 GPT-4에 의해 생성된 대규모 자동 형식화 데이터세트로, 28,000개의 인스턴스를 포함하며, 1,000개는 인간 검증을 통해 테스트 세트로 사용됩니다. 연구진은 두 데이터세트를 체계적으로 인간 검토하여 오류 및 모호성을 분석하고, 이를 통해 보다 정확한 데이터세트를 제공하고자 했습니다.

- **Performance Highlights**: 수정된 데이터셋을 기반으로 한 평가에서, 최신 LLM 모델의 정확도가 +9에서 +22% 포인트 향상되었습니다. 제안된 LLM 기반 검토 프레임워크는 전체 FOLIO 검증 인스턴스의 24%도 검토하지 않고 90%의 데이터셋 정확도를 달성할 수 있음을 보여줍니다. 따라서 본 프레임워크는 데이터셋 큐레이션, 형식 방법 및 검증 시나리오에서 비용 효율적인 검토를 지원하는 데 활용될 수 있습니다.



### Large Byte Model: Teaching Language Models About Compiled Cod (https://arxiv.org/abs/2606.02834)
- **What's New**: 본 연구는 바이너리 파일에 대한 질문에 답할 수 있는 최초의 바이트 네이티브 대형 언어 모델(Large Language Model, LLM)을 소개합니다. 기존의 LLM은 원시 바이트 표현을 처리할 수 없으나, 제안된 모델은 맞춤형 바이트 토크나이저를 통해 이러한 문제를 해결합니다. 학습 과정 중 도메인 지식을 제공하는 것이 중요하며, 기존 모델은 정확성과 통찰력이 부족하다는 사실을 강조합니다.

- **Technical Details**: 이 모델은 하이브리드 임베더 시스템을 기반으로 하여 텍스트 및 바이트 임베더를 결합함으로써 기존 LLM의 어휘를 확장합니다. 훈련 과정은 두 단계로 이루어지며, 첫 번째 단계에서는 10GB의 원시 바이너리 데이터를 기반으로 한 새로운 +5K 바이트 토큰 어휘를 학습하고, 두 번째 단계에서는 지침 기반의 미세 조정을 통해 바이트 분석 기술을 개발합니다. 모델은 최대 256KB 크기의 바이너리 시퀀스를 처리할 수 있도록 최적화되었습니다.

- **Performance Highlights**: 모델은 설문 조사에서 파악된 주요 작업을 수행할 수 있으며, 악성 코드 가족 분류에서 69%에서 아키텍처 분류에서 98%에 이르는 정확도를 자랑합니다. 이 모델은 AWS SageMaker를 통해 내부적으로 배포되어 사용되고 있으며, 기존 LLM 기반 도구에 비해 훨씬 저렴한 비용으로 문제를 해결할 수 있습니다. 앞으로 추가 개선이 필요하다고 하더라도, AI를 통한 악성 코드 생성 및 은폐 증가에 대처하는 데 중요한 과학적 결과를 수립했습니다.



### Which Defense Closes Which Threat? Attributing OWASP-LLM-Top-10 Coverage and Its Brittleness Under Paraphrasing (https://arxiv.org/abs/2606.02822)
Comments:
          17 pages, 4 figures, 7 tables

- **What's New**: 이 논문은 LLM(대형 언어 모델) 애플리케이션에서 여러 개의 방어 체계를 평가하기 위해 새로운 기준을 설정합니다. 기존의 Breach-and-Attack-Simulation (BAS) 벤치마크가 집계 커버리지 수치만을 제공한 반면, 이 연구에서는 각 방어 체계가 어떤 위협을 막는지를 구분하여 측정합니다. 또한, OWASP-LLM-Top-10 리포트를 참조하여 방어 체계의 효과를 실험하기 위해 새로운 에이전트를 도입한 점이 특징적입니다.

- **Technical Details**: 논문에서는 LLM 방어 체계를 L0(방어 없음), L1(거부 전용), L2(예산 전용), L3(전체 스택)으로 구분하고, 각 체계의 성능을 평가하기 위해 21개의 기본 스캐너와 연결하여 전체 탐지 성능을 분석했습니다. 각 방어 체계의 효과를 파악하기 위해, 파라프레이징(paraphrasing)을 통하여 공격이 어떤 형태로 변형되었을 때 방어가 얼마나 견고한지를 평가하였습니다. 이를 통해 3개의 기여를 명확히 했으며, 각 방어 체계의 공헌도를 평가하는 표와 그림을 제시하였습니다.

- **Performance Highlights**: 결과적으로, L1 방어 체계는 LLM01(탈옥 공격) 및 LLM07(시스템 프롬프트 유출)을 모두 차단하며, L2는 LLM02(민감 정보 유출) 및 LLM10(무제한 소비)을 막음으로써 예산 통제가 중요하다는 것을 보여주었습니다. L3처럼 전체 스택을 사용할 경우, LLM06(과도한 대행)은 방어할 수 있었습니다. 또한, 여러 번의 실험을 통해 정리된 결과는 각 방어 체계의 효과성을 뒷받침하고 있으며, L4(실제 LLM)와의 비교를 통해 정규 표현식 이외의 기여가 없음을 입증하였습니다.



### Do Neural Retrievers Prefer Certain Documents? Evidence of Learned Relevance Priors (https://arxiv.org/abs/2606.02814)
- **What's New**: 이번 연구에서는 신경 검색기(neural retrievers)가 쿼리-문서 쌍을 기반으로 문서의 관련성을 추정하는 방식에서 발생하는 문제점을 탐구합니다. 주목할 점은, 주어진 주석 데이터에서 학습한 모델이 문서의 관련성을 넘어서, 특정 문서 유형에 대한 편향된 선호를 내재화하고 있다는 것입니다.

- **Technical Details**: 연구팀은 주석된 문서 임베딩(document embeddings)을 동결(frozen) 한 상태에서 간단한 분류기(classifiers)를 훈련하여 문서 수준의 관련성 우선 신호를 추정했습니다. 이는 훈련된 bi-encoder retrievers가 문서 관련성을 독립적으로 어떻게 회귀(represent)하는지를 평가하기 위해 여러 정보 검색(IR) 벤치마크에서 최첨단(Search) 검색기들이 어떻게 작동하는지를 조사한 결과입니다.

- **Performance Highlights**: 결과적으로, 감독형 신경 검색기는 일반화(generalize) 가능한 우선 신호를 내포하고 있으며, 이는 기존에 보지 못한 문서에 대해서도 일관성을 유지함을 보여줍니다. 특히, 낮은 우선도를 가진 문서는 실제로 관련성이 있음에도 불구하고 더 어렵게 검색되는 경향이 있으며, 이는 기존 문서의 비교에서도 지속적으로 나타나는 현상입니다.



### Cosmos 3: Omnimodal World Models for Physical AI (https://arxiv.org/abs/2606.02800)
- **What's New**: Cosmos 3는 언어, 이미지, 비디오, 오디오 및 행동 시퀀스를 통합하여 처리하고 생성하는 오미모달(omnimodal) 세계 모델의 가족을 소개합니다. 이 모델은 변형기(Transformer) 아키텍처를 기반으로 하여 다양한 입력-출력 구성에 대한 높은 유연성을 지원합니다. 이를 통해 Cosmos 3는 비전-언어 모델, 비디오 생성기, 세계 시뮬레이터 및 행동 모델을 하나의 프레임워크로 통합합니다.

- **Technical Details**: Cosmos 3는 물리적 AI(Physical AI)를 위한 필수 모달리티를 통합하여, 스케일이 가능하고 범용적인 백본을 제공합니다. 이 모델은 다양한 이해 및 생성 작업에서 새로운 최첨단(state-of-the-art) 성능을 기록하였습니다. 코드, 모델 체크포인트, 커리 큐레이션된 합성 데이터셋 및 평가 벤치를 Linux Foundation의 OpenMDW-1.1 라이센스 하에 오픈 소스로 제공하여 연구를 촉진합니다.

- **Performance Highlights**: 적용성 테스트에서 Cosmos 3의 모델은 인공지능 분석(Artificial Analysis)에서 최고의 오픈 소스 텍스트-이미지 및 이미지-비디오 모델로 평가되었습니다. 또한 RoboArena에서 최고의 정책 모델로 선정되었으며, 이는 Cosmos 3의 성능을 입증하는 결과입니다.



### CRAM-ER: Error-Resilient Spintronic Computational Random Access Memory for Scalable In-Memory Computation (https://arxiv.org/abs/2606.02781)
- **What's New**: 이번 연구에서는 CRAM-ER(에러 회복 가능한 컴퓨테이셔널 랜덤 접근 메모리) 아키텍처를 제안했습니다. 이는 메모리와 계산 유닛 간의 데이터 이동 문제를 해결하여 대규모 DNN 처리에 적합하도록 설계되었습니다. 이전의 CRAM 연구들은 소규모 작업에만 집중했던 반면, 이번 연구는 대규모 멀티비트 MAC(Multiply-Accumulate)을 지원하는 기능을 처음으로 보여주고 있습니다.

- **Technical Details**: CRAM-ER 아키텍처는 하이브리드 스핀트로닉스-CRAM과 CMOS 구조를 사용하여 에러를 줄이고 메모리 내 행렬-벡터 곱셈(MVM) 성능을 향상시킵니다. 스핀트로닉스 관련 CRAM은 높은 밀도와 신뢰성을 제공하지만, 확장성에는 프로바빌리스틱(MRAM) 스위칭이 문제를 일으킬 수 있습니다. 이 연구는 또한 알고리즘 레벨에서의 에러 동작 모델링과 저전력 부분 에러 수정 회로를 통합하여 에러를 효과적으로 관리합니다.

- **Performance Highlights**: 하이브리드 CRAM-ER 아키텍처는 DNN 벤치마크에서 거의 손실 없는 정확도를 보이며, A100 GPU보다 10배 높은 에너지 효율성을 자랑합니다. CRAM-ER은 메모리 쓰기 효율성을 개선하여 CPU/GPU 참조에 비해 16배 더 뛰어난 에너지-지연 곱(EDP)을 달성했습니다. 이러한 개선은 대규모 DNN 워크로드를 처리하는 데 있어 상당한 성능 향상을 제공합니다.



### Representational Capacity: Geometric Limits on Feature Representation in Transformer Language Models (https://arxiv.org/abs/2606.02765)
Comments:
          22 pages, 10 figures. Submitted to NeurIPS 2026. This is a condensed version of thesis: this https URL

- **What's New**: 이 논문에서는 트랜스포머 언어 모델에서 모델 차원($d_{model}$)이 특징 표현의 기하학적 한계를 설정하는 데 어떻게 기여하는지를 탐구합니다. 모델이 특징을 거의 직교 방향으로 인코딩한다고 제안하는 Linear Representation 및 Superposition Hypotheses에 기반하여, 모델이 지원할 수 있는 방향의 수를 추정하는 프레임워크를 개발하였습니다. 여러 오픈 소스 모델을 분석한 결과, 높은 허용 오차 ($
 \, \, \, \, \, \, \, \, \, $)를 가진 모델과 낮은 허용 오차를 유지하는 모델의 두 가지 클래스로 나뉘는 것을 발견했습니다.

- **Technical Details**: 이 논문은 임베딩 행렬을 사용하여 모델의 잠재 공간에서의 가까운 직교($near-orthogonality$)에 대한 허용 오차를 측정하는 방법을 제시합니다. 훈련 후 임베딩 행렬의 유사성 분포를 분석하여, $
 \, \, \, \, \, \, \, \, \, $를 추정하는 방법을 제안하고, 이를 다수의 모델에 적용하여 두 가지 클래스의 모델을 확인합니다. 또한, Johnson-Lindenstrauss의 정리가 훈련된 표현의 포장 효율성을 과소 평가한다는 사실을 증명하고, 추가적인 매개변수 없이 예측 오류를 두 배로 줄일 수 있는 수정된 용량 공식을 도출했습니다.

- **Performance Highlights**: 연구는 모델의 잠재 공간 내에서 구별 가능한 방향의 수에 대한 수치적 상한선을 정량화하는 표현 용량($representational capacity$)을 정의합니다. 이 용량은 $
 \, \, \, \, \, \, \, \, \, $에 대해 지수적으로 민감하며, 더 큰 모델은 원시 용량을 극대화하기보다는 더 조밀한 직교성을 선호하는 경향이 있습니다. 결과적으로, 주어진 임베딩, 언임베딩 및 특징 간의 가까운 직교 방향들이 공유 자원으로 작용함을 보여줍니다.



### Acceptance-Test-Driven Evaluation Protocols for Business-Centric LLM Systems (https://arxiv.org/abs/2606.02755)
- **What's New**: 이번 논문은 대형 언어 모델(LLM) 애플리케이션이 어떻게 결정론적 비즈니스 요구를 충족해야 하는지를 다룹니다. 기존의 포스트 호크(Poost-hoc) 벤치마킹이 불충분하다는 점을 지적하였으며, 안전하고 신뢰할 수 있으며 감사 가능하고 경제적으로 유용한 시스템을 위한 평가 프로토콜 확장을 제안합니다. 새로운 접근법은 수용 테스트 중심 개발(acceptance-test-driven development)과 안전 공학(safety engineering)을 결합하였습니다.

- **Technical Details**: 이 논문은 이해 관계자의 목표를 실행 가능한 행동 계약(executable behavioral contracts), 릴리스 게이트(release gates), 모니터링 신호(monitoring signals) 및 증거 아티팩트(evidence artifacts)로 변환하는 방법을 설명합니다. 테스트 주도 개발(test-driven development)의 레드-그린-리팩토링(red-green-refactor) 원칙을 수정하여, 레드-트레인-그린(red-train-green) 생애주기를 제시하고 있습니다. 이 생애주기 내에서 원하는 행동에 대한 실패하는 수용 테스트를 정의한 후, 모델을 개선하고 최종적으로 다차원 게이트를 만족할 때만 릴리스를 허용합니다.

- **Performance Highlights**: 연구에서 제안한 메트릭 스택(metric stack)과 참조 아키텍처(reference architecture)는 수용 테스트 중심의 LLM 개발과 프로프트-퍼스트(prompt-first) 및 벤치마크-애프터(benchmark-after) 워크플로우를 비교할 수 있는 실증적 프로토콜을 제공합니다. 이를 통해 보다 체계적이고 효율적인 LLM 개발이 가능하다는 것을 알 수 있습니다. 논문은 이러한 접근법이 기업 중심의 검증(Business-centric validation)과 안전성 확보에 중요한 역할을 한다고 강조합니다.



### MetaWorld: Scaling Multi-Agent Video World Model from Single-view Video Data (https://arxiv.org/abs/2606.02753)
- **What's New**: MetaWorld은 단일 보기 비디오에서 직접 다중 에이전트 비디오 세계 모델을 확장하는 혁신적인 프레임워크입니다. 기존 모델이 단일 관찰자로 제한되어 있었던 반면, MetaWorld는 데이터 부족과 세계 상태 정렬의 두 가지 주요 문제를 해결하여 다수의 에이전트가 상호작용하는 것을 가능하게 합니다. 이를 통해 더 일반화된 오픈 도메인 환경에서 사용할 수 있는 다중 에이전트 비디오 모델링을 가능하게 합니다.

- **Technical Details**: MetaWorld의 핵심 구성 요소로는 Monocular World-State Unrolling (MWSU), Subject-Aware World Generator (SAWG), World-State Alignment (WSA)가 있습니다. MWSU는 단일 보기 비디오에서 카메라 운영자의 에고 모션과 주체의 공간 경로를 명시적으로 분해하여 다수의 에이전트의 동기화된 모션 데이터를 추출합니다. SAWG는 에이전트 식별 이미지에 따라 생성이 조정되는 시뮬레이션을 수행하며, WSA는 정적 기하학적 일관성과 동적 모션 일관성을 보장하기 위해 서로 다른 생성 브랜치를 동기화합니다.

- **Performance Highlights**: MetaWorld는 여러 카메라의 에고 중심 벤치마크에서 월드 일관성, 상호 관찰 가능성 및 아이덴티티 펀들티 metrics(측정 지표)를 통해 평가되었습니다. 실험 결과, MetaWorld는 단일 에이전트 기준선보다 상당히 높은 교차 보기 일관성을 달성했으며, 아울러 단일 보기 생성 품질도 유지하였습니다. 따라서 MWSU와 WSA의 효과를 확인할 수 있습니다.



### Plan2Map: A Multimodal Benchmark for Document-Grounded Geospatial Boundary Reconstruction from Planning Records (https://arxiv.org/abs/2606.02747)
Comments:
          Project page: this https URL. Fabian Degen and Oishi Deb Contributed Equally

- **What's New**: 이번 논문에서는 UK의 계획 기록에서 기계 가독 가능한 경계를 재구성하기 위한 새로운 벤치마크인 Plan2Map을 소개합니다. Plan2Map은 208개의 사례로 구성된 멀티모달(모달) 벤치마크로, 소스 문서를 기반으로 유효한 지리적 경계를 재구성할 수 있어야 합니다. 이는 공지 텍스트(notice text), 일정(schedule), 지도(map) 판, 지도 레이블(map labels), 경계 주석(boundary annotations) 등 다양한 자료를 사용하여 이루어집니다.

- **Technical Details**: 이 논문은 GeoPlanAgent라는 시스템을 제안합니다. 이 시스템은 문서 기반의 지리적 툴로, 증거 추출(evidence extraction), 위치 지정(localisation), 지도 등록(map registration), 경계 분할(boundary segmentation), 투영(projection), 검증(verification)으로 작업을 분해합니다. Plan2Map 데이터셋에서 GeoPlanAgent는 평균 IoU(Intersection over Union) 0.736과 중앙 IoU 0.904를 달성하며, 예측의 67.8%는 IoU가 0.8 이상입니다.

- **Performance Highlights**: GeoPlanAgent는 직접적인 VLM(Visual Language Model)에서 GeoJSON으로의 변환 방식보다 현저하게 우수한 성능을 보여줍니다. 진단 분석에 따르면, 직접 VLM 예측은 신뢰성이 떨어지며, 나머지 오류는 위치 지정과 지도 등록에서 집중적으로 발생합니다. 반면에 감독된 경계 분할은 픽셀 수준의 마스크 품질을 상당히 개선합니다.



### EntangleCodec: A Unified Discrete Audio Tokenizer via Semantic-Acoustic Entanglemen (https://arxiv.org/abs/2606.02739)
Comments:
          17 pages, 10 figures

- **What's New**: 새로운 연구에서는 EntangleCodec이라는 통합된 이산 오디오 토크나이저를 제안합니다. 이 모델은 캡션과 정렬된 시맨틱-어쿠스틱 표현을 학습하여 양자화 전에 음성을 포착합니다. 기존의 방법들과 달리, EntangleCodec은 ASR(Automatic Speech Recognition) 트랜스크립트 대신 더욱 풍부한 캡션과 오디오를 정렬하여 언어적 내용과 감정, 말투, 음향 장면 등의 다양한 정보를 캡처합니다.

- **Technical Details**: EntangleCodec의 구조는 통합 인코더, 이산 양자화기, 그리고 확산 기반 디코더의 세 가지 핵심 구성 요소로 이루어져 있습니다. 이 모델은 통합된 시맨틱-어쿠스틱 표현을 통해 음성을 단일 인코더로 처리하며, 이는 ASR 트랜스크립트 대신 풍부한 오디오 캡션으로 시맨틱 정렬을 성취합니다. 통합된 오디오 표현을 통해 EntangleCodec은 복잡한 퓨전 모듈 없이도 높은 재구성 품질을 유지합니다.

- **Performance Highlights**: EntangleCodec은 재구성 품질에서 기존의 특화된 코덱과 비교하여 경쟁력 있는 성과를 보여줍니다. 모델은 MMAR(Mean Multimodal Audio Recognition) 벤치마크에서 기존 코덱 기반 기준선보다 최대 7.4% 향상된 성과를 기록했습니다. 추가적으로, 0.6B 파라미터를 가진 EntangleCodec 모델은 13B 이상의 파라미터를 가진 모델들을 능가하며 새로운 주목할만한 결과를 충족합니다.



### Attention Calibration for Position-Fair Dense Information Retrieva (https://arxiv.org/abs/2606.02737)
- **What's New**: 이 논문에서는 정보 검색에서 밀집 검색 모델의 위치 편향(position bias)을 해결하기 위한 새로운 방법을 제안합니다. 연구자들은 재훈련 없이 검색의 전반적인 효과를 유지하며 추론 시간에서 편향을 줄일 수 있는지 여부에 대해 조사합니다. 이들은 주목(calibration) 방법을 확장하여 조정 가능한 강도 계수(strength coefficient) λ를 도입하여, 원래의 주의 분포와 완전히 조정된 주의 분포 간의 상호작용을 조절합니다.

- **Technical Details**: 이 연구는 관심(calibration)이 정보 검색에서 어떻게 작용할 수 있는지를 분석합니다. 각 모델의 매개변수인 바구니 크기(basket size), 조정된 레이어 세트(calibrated layer set), 그리고 강도(strength)가 모델의 성능에 미치는 영향을 체계적으로 조사합니다. 저자들은 이러한 매개변수를 통해 위치 일관성과 검색 품질 사이의 균형을 어떻게 조절할 수 있는지를 연구하고, 여러 언어와 도메인에서 적용할 수 있는 모델 무관한 기본 설정(default configuration)을 제시합니다.

- **Performance Highlights**: 결과적으로, 기본 설정(B=128, λ=0.5, 50% layer depth)을 사용함으로써 FineWeb-PosQ에서 모든 모델이 위치 그룹 간 nDCG@10의 조화 평균(harmonic mean)을 개선하였습니다. 또한, 이 설정은 PosIR 벤치마크에서 10개 언어와 31개 도메인으로 이전되어 모든 조합에서 위치 민감도(PSI)를 줄이며, nDCG@10 지표를 유지하거나 개선하는 동시에 긍정적인 결과를 보였습니다.



### See Less, Specify More: Visual Evidence Budgets for Generalizable VLAs (https://arxiv.org/abs/2606.02735)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구에서는 VLA 모델의 일반화를 개선하기 위해 S2(See Less, Specify More) 프레임워크를 제안합니다. 이 프레임워크는 외란(distractors) 및 유사한 작업에서도 명확한 실행 세부사항을 유도할 수 있도록 설계되었습니다. 'Specify More'는 고수준 목표를 유지하면서도 로컬 행동 방식(local execution mode)을 명확히 하는 정제된 언어로 경로를 다시 레이블링(relabeling)합니다.

- **Technical Details**: S2는 언어와 시각 인터페이스를 통한 훈련을 통해 모델의 혼란을 줄이는 데 집중하고 있습니다. 'See Less'는 시각적 증거(budget)의 명시적 제한을 두어, 효율성뿐 아니라 일반화를 향상시키는 데 기여합니다. 또한, 수동 레이블링(manual labels)이나 외부 VLM(supervision) 없이도 모델이 성공적인 실행을 예측할 수 있도록 훈련하는 개발 방향을 갖습니다.

- **Performance Highlights**: S2를 적용한 결과, TX-G2와 HSR에서 8개의 실제 로봇 작업에서 평균 서브태스크 성공률이 54.2%에서 79.0%로 증가하는 성과를 이루었습니다. 이 연구는 VLA 모델의 일반화를 향상시키기 위해 언어와 인지(perception) 문제를 동시에 줄이는 것이 중요하다는 것을 시사합니다.



### AVTrack: Audio-Visual Tracking in Human-centric Complex Scenes (https://arxiv.org/abs/2606.02724)
Comments:
          19 pages, 10 figures, ICML 2026

- **What's New**: 이 논문은 복잡한 환경에서 사람 중심의 오디오-비주얼 인스턴스 분할(AVIS)을 위한 새로운 데이터셋 AVTrack을 소개합니다. AVTrack은 카메라 움직임, 시각적 차단, 다수 인스턴스 등의 도전적인 상황을 포함하여 871개의 비디오 클립으로 구성되어 있습니다. 이 데이터셋은 기존의 단순한 오디오-비주얼 장면에 비해 더 다양하고 복잡한 동적 시나리오를 제공하여 평가의 유효성을 높입니다.

- **Technical Details**: AVTrack은 3,120 개의 고밀도 주석 인스턴스 트랙렛을 포함, 고급 오디오-비주얼 추적 기술을 시험하기 위해 설계되었습니다. 기존의 데이터셋은 정적 장면에 초점을 맞춰 spatiotemporal 모델링을 제대로 평가하지 못했습니다. AVTrack은 사람 중심의 복잡한 동적 장면을 대상으로 하여, 교차 모달 추론 (cross-modal reasoning) 및 robust spatiotemporal modelling을 가능하게 합니다.

- **Performance Highlights**: AVTrack에서 평가된 최첨단 AVIS 방법들은 기존 데이터셋 대비 성능 저하를 보였습니다. 이는 AVTrack이 더 현실적이고 도전적인 벤치마크로 자리 잡았음을 시사합니다. 또한, 미래 연구를 지원하기 위한 모듈형 플러그 앤 플레이의 확장 가능한 기초 프레임워크를 제공하여 커뮤니티의 연구를 촉진할 것입니다.



### Filter, Then Reweight: Rethinking Optimization Granularity in On-Policy Distillation (https://arxiv.org/abs/2606.02684)
- **What's New**: 본 연구에서는 FiRe-OPD(필터링 후 리웨이팅)를 제안하여 온-정책 증류(On-Policy Distillation, OPD)에서 경량화된 텍스트 모델 학습을 향상시키고자 한다. FiRe-OPD는 저품질 샘플을 필터링하고, 필터링된 경로 내에서 정보 가치가 높은 토큰에 지속적으로 중요도를 부여하는 방식으로 기능한다. 이 방식은 기존의 하드 선택 방법보다 더 부드러운 최적화(soft optimization)를 가능하게 하여 정보 손실을 줄이고 최적화 안정성을 높인다.

- **Technical Details**: FiRe-OPD는 두 가지 단계로 나누어 생각할 수 있다. 첫 번째로, 경로 수준에서 낮은 품질의 롤아웃(rollout) 샘플을 필터링하여 불량한 감독 신호를 제거한다. 두 번째로, 남겨진 경로에서 정보가 유의미한 토큰에 대한 중요도를 부여하여 최적화 과정을 조정한다. 이를 통해 FiRe-OPD는 하드 선택(hard selection)보다 더 정교한 OPD 최적화를 가능하게 한다.

- **Performance Highlights**: 우리는 FiRe-OPD가 다양한 설정(단일 교사, 다중 교사)에서 강력한 성능을 보여주며, 최근의 토큰 수준 OPD 방법들보다 우수함을 입증하였다. 성능 비교에서 FiRe-OPD는 AIME 2024에서 +6.25의 성과, Miner에서 +18.81의 성과를 기록하며 기존 방법보다 뛰어난 결과를 달성하였다. 이러한 연구 결과는 OPD의 새로운 방향성을 제시한다.



### Aligning Data-Driven Predictors with Allocation: A Decision-Focused Approach to Survival Analysis (https://arxiv.org/abs/2606.02671)
- **What's New**: 이 논문은 머신러닝(ML) 예측 모델과 알고리즘적 작업 간의 불일치를 강조하며, 표준 통계 지표에 최적화된 생존 예측자가 장기 이식에서 어떻게 좋지 않은 결과를 초래할 수 있는지를 보여줍니다. 이를 통해 NDCG(정규화 할인 누적 이득)를 최적화하는 결정 중심 학습 접근 방식을 제안하고, 이는 알로케이션 성능에 대한 보장을 제공함을 입증합니다. 이 연구는 실제 데이터에 기반하여 NDCG를 최적화하는 방법의 효과를 입증했습니다.

- **Technical Details**: 기존의 생존 분석 모델은 종종 C-index와 같은 집합 성능 지표에 최적화되어 있지만, 이는 개별적으로 최적의 매칭을 평가하지 못합니다. 논문에서는 NDCG가 이러한 평가를 가능하게 하며, NDCG@1이 하류 알고리즘의 유용성 보장으로 연결된다는 정리를 제시합니다. 이 연구는 오른쪽 검열(right censorship) 문제를 다루며, 이를 위한 두 가지 새로운 NDCG 추정기를 제안하고, 이들이 공정한 추정치를 제공함을 입증합니다.

- **Performance Highlights**: 미국의 과거 심장 이식 데이터를 기반으로 하는 실험에서, 제안된 방법을 통해 기본 모델의 NDCG가 50-100% 증가했습니다. 이는 매년 수만 년의 추가 생명 연수를 창출하는 결과로 이어졌습니다. 이 접근 방식은 장기 이식 분야뿐 아니라, 더 넓은 범위의 의사 결정을 지원하는 데 기여할 것으로 기대됩니다.



### Anomalies in Multivariate Time Series Benchmarks Are Mostly Univaria (https://arxiv.org/abs/2606.02670)
- **What's New**: 최근 다변량 시계열 이상 탐지(MT-SAD) 모델들은 이상 상태가 여러 채널에 걸쳐 퍼질 수 있다는 암묵적인 가정 하에 교차 채널 모델링을 도입하고 있습니다. 본 연구에서는 이러한 가정을 평가하기 위해 8개의 널리 사용되는 공개 벤치마크를 기반으로 각 레이블된 이상에 대해 진단 프레임워크를 도입했습니다. 이 프레임워크는 최소한 하나의 채널이 정상 기록에서 벗어나는지, 교차 채널 상관 구조가 변화하는지, 또는 둘 다의 여부를 플래그ging합니다.

- **Technical Details**: 이 연구는 지정된 각 이상 세그먼트에 대해 'univariate' 탈선 여부를 평가하며, 여러 합리적인 임계값을 통해 교차 채널의 변화가 없다고 보고했습니다. 'Synthetic data'를 생성해 위상 이동이 있는 사인 채널로 교차 채널 구조가 존재할 때 이를 정확히 포착할 수 있는지를 검증했습니다. 교차 신호를 성공적으로 활용하는 채널 의존 모델(CD)과 그렇지 못한 채널 독립 모델(CI)의 비교를 통해 교차 채널 모델링의 유용성을 분석하였습니다.

- **Performance Highlights**: 연구 결과, 8개의 벤치마크 중 6곳에서는 레이블된 이상 세그먼트의 최소 절반에서 79%에서 100%까지 단일 채널(univariate) 탈선이 발생하며, 세 개의 데이터 세트에서는 100%에 도달했습니다. 또한 최신 SOTA 감지기에서 채널 의존 모델링(CD)이 측정 가능한 이득을 주지 않는다는 것을 확인했습니다. 결론적으로, 현재의 MTSAD 벤치마크는 교차 채널 모델링의 능력을 검증하는 데 부적합하므로 더 다양하고 구조적인 평가 세트의 개발이 필요하다고 언급했습니다.



### AdaWeather: Adaptively Mixing Probabilistic Weather Forecasts with Logarithmic Regr (https://arxiv.org/abs/2606.02663)
Comments:
          36 pages, 16 figures. Submitted to arXiv. Forecast aggregation for probabilistic weather prediction using offline supervised learning and online prediction with expert advice. Includes theoretical regret guarantees and empirical evaluation on temperature forecasting. Submitted to NeurIPS 2026

- **What's New**: 본 연구에서는 AdaWeather라는 적응형 프레임워크를 도입하여 다양한 확률적 예측을 결합하여 개선된 예측 결과를 제공합니다. 기존의 방법론은 일반적으로 감독 학습이나 전문적 조언을 활용하여 이루어지지만, AdaWeather는 머신러닝과 전문가 혼합 방법을 통해 이러한 두 가지 접근 방식을 통합합니다. 이를 통해 기존 방법들보다 일관된 개선이 가능하게 됩니다.

- **Technical Details**: 이 연구에서는 U-Net 모델을 사용하여 과거 패턴을 학습하고, 이 모델을 기존의 기상 모델들과 함께 결합합니다. 이러한 방식으로 추가된 모델이 다른 전문가들과 함께 작용하여 최종 예측 결과를 생성합니다. 연구팀은 로그 단위의 후회(logarithmic regret) 경계를 제시하며, 이는 과거의 최적 혼합 상황에 비례할 수 있는 새로운 이론적 한계를 제공합니다.

- **Performance Highlights**: 온도 예측에 대한 실험을 중심으로, AdaWeather는 기존의 예측 방법들을 초월하는 성과를 보여주었습니다. 비교 연구에서 성공적으로 성능을 개선하였으며, 다른 방법과의 실험을 통해 그 유용성을 입증했습니다. 따라서, 이 연구는 기상 예측 분야에서 혁신적인 기여를 하고 있습니다.



### Improvise, Adapt, Overcome: An On-The-Fly Multifidelity Algorithm for Efficient Machine Learning (https://arxiv.org/abs/2606.02662)
Comments:
          Supplementary Information added as separate PDF

- **What's New**: 본 연구에서는 머신러닝(Machine Learning)과 양자화학(Quantum Chemistry) 간의 데이터를 효율적으로 결합하는 새로운 프레임워크, 즉 적응형 다중 신뢰도(Multifidelity) 기법을 제안합니다. 기존의 표준 다중 신뢰도 방식이 미리 정의된 비율에 따라 데이터를 사용하여 데이터 중복을 초래하는 문제점을 해결하고자, 학습 데이터셋을 자동으로 생성하는 알고리즘을 도입했습니다. 이 혁신적인 접근법은 훈련 샘플을 동적으로 쿼리하여 낮은 신뢰도에서 모델 정확성을 포화 상태로 만든 후, 더욱 비싼 참조 계산으로 전이할 수 있도록 합니다.

- **Technical Details**: 적응형 다중 신뢰도(MFML) 기법은 다양한 신뢰도(fidelity) 수준의 훈련 데이터를 사용하여 데이터 생성 비용을 최대 30배 감소시키는 것으로 나타났습니다. 이 기법은 네 개의 신뢰도 수준을 정의하고, 각 신뢰도에 대해 반복적으로 훈련 샘플을 추가하는 재귀적 구조를 갖고 있습니다. 경비(MAE: Mean Absolute Error) 계산을 통해 성능을 저하하지 않고 최적의 훈련 샘플 수를 동적으로 선택하여 데이터 중복을 최소화합니다.

- **Performance Highlights**: 이 알고리즘은 몇 가지 화학적 성질에 대해 벤치마킹을 통해 검증되었으며, 특히 CCSD(T)와 같은 계산 화학 금본위 기준의 에너지를 정확하게 예측하면서 효율성을 획기적으로 개선하였습니다. 연구 결과, 적응형 다중 신뢰도 방식이 표준 다중 신뢰도 방식에 비해 기계 학습-양자 화학(ML-QC) 방법의 비용을 5배 줄이는 성능을 보였습니다. 이는 ML-QC 파이프라인에서 데이터 생성의 지속 가능한 경로를 제시하며, 생명주기 비용 감소를 위한 중요한 진전을 이루었습니다.



### Learning to Refine: Spectral-Decoupled Iterative Refinement Framework for Precipitation Nowcasting (https://arxiv.org/abs/2606.02661)
Comments:
          21 pages, 10 figures, accepted at ICML 2026

- **What's New**: 이번 논문에서는 정확한 강수 예보(precipitation nowcasting)를 위한 새로운 프레임워크인 Spectral-Decoupled Iterative Refinement (SDIR)을 제안합니다. 이는 물리적으로 일관된 예측을 제공하면서 과도한 부드러움(over-smoothing)과 환각(hallucinations)의 문제를 해결하도록 설계되었습니다. SDIR은 저주파(低頻)와 고주파(高頻) 세부 사항을 구분하여 점진적으로 정제(refinement)하는 방식으로, 이러한 과정을 통해 신뢰할 수 있는 고해상도 예측을 가능하게 합니다.

- **Technical Details**: SDIR은 이중 경로 디자인을 채택하여 두 가지 주요 구성 요소를 포함합니다: 전세계 구조를 파악하는 Synoptic Frequency-Guided Former (SFG-Former)와 세부 텍스처를 정제하는 Fourier Residual Refiner (FR-Refiner)입니다. 또한, 물리적 일관성을 유지하기 위해 Physically Consistent Power Spectral Density (PCPSD) 손실을 도입하여, 예측이 대기 터뷸런스 법칙에 맞도록 동적으로 마스킹을 적용합니다. 이러한 프로세스는 대기 물리학을 반영하며, 예측의 모든 단계에서 물리적으로 고정된 기초를 활용합니다.

- **Performance Highlights**: 세 가지 벤치마크 실험 결과, SDIR은 공간 정확도(spatial accuracy)에서 최첨단(SOTA) 방법들을 초월하며, 자동 생성 모델들보다 높은 스펙트럴 일관성을 유지합니다. 이를 통해 긴 리드 타임을 가진 예측에서도 흐림(blurring) 현상을 효과적으로 줄이는 데 기여했습니다. 논문에서 제안된 SDIR는 고신뢰도 작동 가능한 예보를 가능하게 하여 현대 기상 서비스에 기여할 것으로 기대됩니다.



### CL-DMDF:Dynamic Multimodal Data Fusion Model Based on Contrastive Learning (https://arxiv.org/abs/2606.02659)
Comments:
          9 pages, 5 figures, 7 tables

- **What's New**: 이 논문에서는 동적 멀티모달 데이터 융합 모형인 CL-DMDF(Contrastive Learning 기반의 동적 멀티모달 데이터 융합 모형)를 제안합니다. 이 모델은 전통적인 융합 모델들이 가진 한계를 극복하기 위해, 특징 수준(feature-level)과 모달리티 수준(modality-level)의 중요성을 동시에 고려하는 이중 차원 주의 메커니즘을 도입합니다. 이는 효과적으로 신뢰할 수 있는 주의 점수를 계산하고, 동적인 융합 전략을 통해 정확성과 효율성을 높입니다.

- **Technical Details**: CL-DMDF 모델은 다양한 모달리티의 전문적인 특징을 추출하기 위해 전용 특징 추출 네트워크를 사용하고, 이를 통합된 벡터 공간에 투사합니다. 또한, 이중 차원 주의 메커니즘이 모달리티와 특정 값 간의 주의를 효과적으로 분산시키고, 반대로 대비 학습을 진행하여 특징의 구분력을 향상시킵니다. 이 외에도, 입력 특징의 특성에 기반하여 최적의 융합 전략을 동적으로 선택하는 적응형 융합 모듈도 포함됩니다.

- **Performance Highlights**: 세 가지 데이터셋에서 수행된 광범위한 실험 결과, CL-DMDF는 다양한 멀티모달 융합 작업에서 강력한 기준선(baselines)들을 지속적으로 능가했습니다. 이 연구는 CL-DMDF의 효과성과 일반화 가능성을 검증하였으며, 동적 멀티모달 융합의 실제 응용 가능성을 강조합니다. 특히, 이 모델은 자율주행과 같은 복잡한 환경에서 시스템의 인지를 크게 향상시켜 데이터의 정확성과 강건성을 높이고 있습니다.



### The Ringelmann Effect in Multi-Agent LLM Systems: A Scaling Law for Effective Team Siz (https://arxiv.org/abs/2606.02646)
Comments:
          41 pages, 9 figures, 20 tables

- **What's New**: 이 논문은 추론 시간에 다중 에이전트 LLM(대규모 언어 모델) 스케일링이 без общая единица(공유 단위)가 부족하다는 문제를 다루고 있습니다. 저자는 두 개의 매개변수 스케일링 법칙 $R(N) = N_	ext{eff}/N = 1/(1+c(N-1)N^{-eta})$를 도출했습니다. 이 법칙은 알고리즘의 성능을 세 가지 비대칭 구역으로 분류하며, 이론적 기초는 심리학적 이론에서 영감을 받았습니다.

- **Technical Details**: 이 스케일링 법칙은 에이전트 간의 답변 다양성과 올바름 중복성에 적용됩니다. 두 번째 매개변수인 β는 추가 에이전트를 추가하는 것의 이점을 판단하는 데 사용됩니다. 특정 조건에서 여러 가지 설정을 검토하여 결과가 R^2 > 0.99로 나타났으며, 이 법칙은 7B에서 32B까지 다양한 모델과 과제를 적용한 실험에 기반하고 있습니다.

- **Performance Highlights**: 실험 결과 세 가지 중요한 발견이 있었습니다. (i) 30개의 밀집 토론 에이전트는 MMLU-Hard에서 단일 에이전트와 같은 답변 다양성을 생성하지 않습니다. (ii) 무작위 노이즈 플라시보는 동종 팀 내에서 재평가를 통해 얻은 이점으로, 동료의 콘텐츠에서 오히려 왔습니다. (iii) N <= 5에서의 파일럿 결과는 N=30의 구조적 한계를 예측할 수 있으며, 아키텍처 다양성만이 하드-천장 영역을 탈출하였습니다.



### Target Updates May Stabilize Linear Q-Learning: Periodic and Soft Dynamics (https://arxiv.org/abs/2606.02645)
- **What's New**: 본 연구는 Q-learning 및 actor-critic 방법에서의 타겟 업데이트 메커니즘에 대한 정확한 이론적 분석을 제공합니다. 기존 방법론이 성과에 있어 경험적으로 지원되었던 반면, 이들 메커니즘의 이론적 기초는 불완전했습니다. 연구 결과에 따르면 주기적인 하드 타겟 업데이트와 소프트 타겟 업데이트는 명시적인 스펙트럼 조건과 단계 크기 조건 하에서 정확한 Q-Bellman 솔루션으로의 수렴을 보장할 수 있습니다.

- **Technical Details**: 연구에서는 선형 함수 근사를 사용하는 Q-learning의 동적 시스템 분석을 위해 스위치 선형 시스템(SLS) 다이내믹스를 활용합니다. 정해진 단계 크기와 스펙트럼 조건 하에서 타겟 업데이트 메커니즘의 수렴성을 증명하며, 사례로는 결정론적 선형 Q-learning을 다룹니다. 스위칭 시스템과 공동 스펙트럼 반지수를 주 도구로 사용하여 규정된 범위 내의 주기를 통해 수렴성을 제시합니다.

- **Performance Highlights**: 연구의 주요 성과는 타겟 업데이트가 적용된 선형 Q-learning에서 특정 단계 크기 조건 하에서 자명한 수렴 결과를 도출한 것입니다. 새롭게 제시된 주기적인 하드 타겟 방법은 결정론적 선형 Q-learning과 Q 가치 반복 간의 상관관계를 명확히 보여주며, 정해진 범위에서의 모든 충분히 큰 주기를 통해 수렴을 보장합니다. 이에 따라, 최종적으로 정확한 Q-Bellman 솔루션으로의 오류 수렴을 보여줍니다.



### A New Framework for Cybersecurity Refusals in AI Agents (https://arxiv.org/abs/2606.02644)
- **What's New**: 이 연구는 사이버 보안의 공격적( offensive) 맥락에서 에이전트가 유해한 요청을 거부하는 기준을 설정하기 위한 최초의 프레임워크를 제시합니다. 기존의 AI 에이전트 벤치마크는 일반적으로 수행 능력에만 중점을 두었지만, 이 연구는 에이전트가 유해한 요청을 거부하는 것에 대한 평가 기준을 다룹니다. 또한, 여러 웹 기반 공격 보안 시나리오에서 현재 LLM 기반 에이전트의 거부 경계를 평가합니다.

- **Technical Details**: 제안된 프레임워크는 (1) 거부해야 하는 작업에 대한 원칙적 기준, (2) 거부에 적합한 작업 카테고리, (3) 정상 및 적대적 조건에서 에이전트의 강건성(robustness)을 측정하기 위한 평가 방법론을 포함합니다. 이 연구의 초점은 웹 취약성에 한정되지만, 향후 작업은 다른 사이버 보안 분야로 이 rationale 를 확장해야 합니다. 이를 통해 에이전트가 언제, 어떻게 작업을 거부할지를 심도 있게 평가할 수 있습니다.

- **Performance Highlights**: 각 모델의 평가는 8개의 최전선 모델 중 6개가 거의 제로(near-zero) 거부율을 보였음을 발견했습니다. 오직 2개의 모델인 GPT-5.2 및 GPT-5.1 Codex만이 유의미한 거부 행동을 나타냈습니다. 이 결과는 현대 LLM 기반 에이전트가 공격적 사이버 보안 요청에 대해 적절한 거부 경계를 유지하는 데 큰 도전과제를 안고 있음을 보여줍니다.



### Inference Cost Attacks for Retrieval-Augmented Large Language Models (https://arxiv.org/abs/2606.02643)
Comments:
          Accepted at The ACM Web Conference 2026 (WWW '26)

- **What's New**: 본 논문에서는 Retrieval-Augmented Generation (RAG) 시스템의 새로운 위협인 Retrieval-Augmented Inference Cost Attack (RA-ICA)을 제시합니다. RA-ICA 공격은 외부 지식 베이스(knowledge base)를 오염시켜 RAG 시스템의 연산 비용을 비정상적으로 증가시키려는 것입니다. 기존의 Inference Cost Attacks (ICA)와는 달리, RA-ICA는 직접적인 프롬프트 조작이 아니라 웹 지식과 같은 외부 데이터의 조작을 허용합니다.

- **Technical Details**: RA-ICA 공격의 실행을 위해, 저자들은 CREEP라는 새로운 프레임워크를 제안하며, 이 프레임워크는 LLM 에이전트를 사용하여 세 가지 전략—Decoy Injection, Contradiction Injection, Task-Oriented Manipulation—으로 신뢰성을 높인 악성 문서를 생성합니다. 또한, MA-GRPO라는 새로운 강화 학습 알고리즘을 통해, 에이전트를 최적화하며 역사적 우수 문서의 동적 기억 버퍼를 활용하여 악성 문서 생성 패턴을 효과적으로 발견합니다.

- **Performance Highlights**: 실험 결과, RA-ICA는 토큰 소비를 최대 13.12배 증가시키며 90% 이상의 성공률을 기록했습니다. 이 과정에서 생성된 답변의 무결성은 저해되지 않았습니다. 이는 RAG-enhanced LLM 시스템의 취약성을 포괄적으로 분석하고, 새로운 공격 패턴을 탐구하는 중요한 첫걸음이 됩니다.



### SVHalluc: Benchmarking Speech-Vision Hallucination in Audio-Visual Large Language Models (https://arxiv.org/abs/2606.02642)
Comments:
          Accepted at CVPR 2026

- **What's New**: 본 연구에서는 audio-visual LLMs의 한계 중 하나인 speech-vision hallucination을 탐구하고, 이를 평가하기 위한 새로운 벤치마크인 SVHalluc를 소개합니다. 기존의 지표들이 환경 소음에 집중하는 반면, SVHalluc는 사람의 언어가 가진 복잡한 의미와 시간 구조를 반영합니다. 이 벤치마크는 speech와 visual 정보 간의 상관성을 정확히 평가할 수 있는지 측정하여 더 깊이 있는 이해를 추구합니다.

- **Technical Details**: SVHalluc는 두 가지 주요 측면, 즉 semantic hallucination과 temporal hallucination을 기반으로 설계되었습니다. 연구자들은 입력 비디오와 결합된 인간의 음성을 기반으로 모델이 올바른 답변을 생성할 수 있는지를 평가합니다. 각 차원에 대해 세 가지 상호 보완적인 진단 작업이 설계되어, 오차 모드를 심층적으로 분석할 수 있습니다.

- **Performance Highlights**: 실험 결과, 오픈 소스 audio-visual LLMs는 SVHalluc에서 매우 낮은 성능을 보였으며, 이는 음성과 시각적 증거 간의 정렬에서 실패를 나타냅니다. 반면, Gemini 2.5 Pro는 이러한 모델들보다 이점을 보이며, 두 모델 간의 성능 차이가 크다는 것을 시사합니다. 이러한 발견은 현재 음성과 비디오 이해의 신뢰성을 향상시키기 위한 필요성을 강조합니다.



### CARVE: Certified Affordable Repair of Vetoed Maneuvers via Envelopes for Interactive Driving (https://arxiv.org/abs/2606.02641)
Comments:
          8 pages, 3 figures

- **What's New**: 이 논문은 대화형 드라이빙(interactive driving)에서 발생하는 잘못된 거부(false veto) 문제에 대한 해결책을 제시합니다. CARVE는 예측 기반 모델이 아니라, 상호작용을 통한 수리 인증(interactive repair certification) 체계를 도입하여 차량 간의 안전하고 법적인 상호작용을 인증합니다. 이는 기존의 규칙 세트에 의한 잘못된 행동 거부로부터 발생하는 문제를 해결하므로, 무단 대기 시간을 줄이고 효율적인 운전을 가능하게 합니다.

- **Technical Details**: CARVE는 제한된 다수 소유자 전술적 운영자에 대한 수리 증명의 레이어로 작동합니다. 이 시스템은 규칙 실행을 위한 협력 영역(cooperation envelope)을 정의하며, 이를 통해 각 에이전트의 역할과 도로에서의 우선 권한(right-of-way)을 분리합니다. 추가로, 수리 인증서는 경계 규칙(binding rule), 수리 카테고리, 책임 가중 비용 분배 및 대처 수단(fallback) 정보를 기록하여 상호작용 안전성을 보장합니다.

- **Performance Highlights**: CARVE-Greedy는 Lanelet2 기반의 589개의 INTERACTION 재생 에피소드에서 98.64%의 초기 거부 조작을 수용하며, 인간이 해결한 잘못된 거부 370/378을 회복하는 성과를 보였습니다. 이는 우선 에이전트의 잘못된 긍정(false positives)이 0이라는 점과, 모든 처리된 상황에서의 법적 우선권 준수를 유지하면서도 뛰어난 성과를 나타냅니다. 이러한 성과는 CARVE가 공공 도로 안전 문제에 대한 실질적인 해결책을 제공함을 입증합니다.



### D-Judge: Disrupting Multi-Turn Jailbreaks using Semantics-Preserving Output Rewriting (https://arxiv.org/abs/2606.02640)
Comments:
          Proceedings of the 43rd International Conference on Machine Learning

- **What's New**: 본 논문에서는 D-Judge라는 새로운 AI 안전 방어 기법을 소개합니다. 이 방법은 다중 턴 jailbreak 공격이 증가하는 위협에 대응하기 위해 개발되었습니다. D-Judge는 공격자의 judge 모델에 의해 평가되기 전에 LLM(large language model)의 응답을 직접 수정하여 공격자의 피드백 루프를 교란합니다.

- **Technical Details**: D-Judge는 의미를 보존하는 output rewriting 기법을 사용하여 원래 응답의 의미는 변화시키지 않으면서 judge의 피드백 신호를 잘못 맞추게 합니다. 이를 통해 D-Judge는 공격자의 프롬프트 수정 과정을 무산시켜, 이후의 질의(query)가 왜곡된 공격 진행 신호에 최적화되도록 만듭니다. 성능 향상을 위해 우리는 서로 다른 judge 평가 점수를 야기하는 의미적으로 동등한 응답 쌍의 데이터셋을 구성하고, 이를 통해 감독 학습(supervised fine-tuning)을 수행하였습니다.

- **Performance Highlights**: 실험 결과, D-Judge는 HarmBench에서 최신 다중 턴 jailbreak 성공률을 감소시키며 동시에 일반적인 벤치마크에서 성능을 유지하는 것으로 나타났습니다. 이는 D-Judge의 효과적인 방어 능력을 보여주며, AI 안전성을 향상시키기 위한 중요한 기여를 하고 있습니다.



### Sparse-View Lung Nodule Volumetry from Digitally Reconstructed Radiographs via AReT: Anatomy-Regularized TensoRF (https://arxiv.org/abs/2606.02639)
- **What's New**: 본 논문에서는 TensoRF에 적용할 때 발생하는 새로운 실패 모드를 식별하고 해결합니다. 기본 밀도 이동값인 -10이 밀도 기울기를 억제하여 적은 수의 X-ray 투사로 인해 발생하는 의학적 재구성을 방해합니다. 밀도 이동값을 0으로 설정함으로써 기울기 흐름이 복원되고 폐 결절에 대한 안정적인 볼륨 재구성이 가능해집니다.

- **Technical Details**: 아시아-정규화된 텐서 방사장 프레임워크인 AReT를 제안하여 LIDC-IDRI 데이터셋에서 세 가지 직각 X-ray 투사(관상, 시상, 축 단면)를 사용한 폐 결절 재구성을 수행합니다. 이 프레임워크는 철저한 ℓ1​ + TV 정규화를 실시하여 최소한의 투사(N=3)로부터도 안정적인 3D 재구성을 가능하게 합니다. 실험 결과는 해부학 정보를 고려한 정규화가 생성된 우선 접근 방식보다 우수함을 보여줍니다.

- **Performance Highlights**: AReT는 방사선의 합의적인 세그멘테이션과 비교하여 임상적으로 적용 가능한 결절(≥10 mm)에서 Pearson r=0.983의 성과를 보이며, 중간 절대 볼륨 오차는 11.4%에 불과합니다. 또한, AReT는 시스템적 바이어스가 거의 없고 구형 볼륨 근사법에 비해 8.4배의 개선을 달성합니다. 이 연구는 제한된 환자 집합에서의 증거 기반 연구로, 향후 연구에서 이를 검증할 필요성을 인정합니다.



### SegTune: Structured and Fine-Grained Control for Song Generation (https://arxiv.org/abs/2606.02638)
Comments:
          This paper has been accepted to ACL 2026 as an oral presentation and has been nominated for the Best Paper Award. This work is a revised and extended version of an earlier technical report (arXiv:2510.18416). arXiv admin note: text overlap with arXiv:2510.18416

- **What's New**: SegTune은 음악 생성에서 계층적인 제어를 지원하는 새로운 NAR(Non-Autoregressive) 프레임워크를 소개합니다. 이 프레임워크는 사용자가 특정한 지역적 음악 설명을 할당할 수 있도록 하여 세부적인 시간에 따른 조정이 가능하게 합니다. 특히, LLM 기반의 지속 시간 예측기를 도입하여, 수동 입력 없이도 노래 가사에 시간 태그를 자동으로 생성합니다.

- **Technical Details**: SegTune은 Diffusion Transformer(DiT) 아키텍처를 기반으로 하여 조건부 흐름 일치를 구현합니다. 여기서 세그먼트 인코더를 통해 각 세그먼트에 대한 세부 제어 신호가 주입되고, 전체 노래의 스타일적 일관성을 유지하는 글로벌 인코더가 동시에 작동합니다. 이 시스템은 고품질의 노래를 수집하기 위한 대규모 데이터 파이프라인을 구성하고 있으며, 새로운 세그먼트 정합 및 음성 일관성을 평가하기 위한 메트릭도 제안합니다.

- **Performance Highlights**: 실험 결과, SegTune은 기존의 다른 시스템들에 비해 음악성(musicality)과 제어 가능성(controllability)에서 모두 우수한 성과를 보였습니다. 특히, 세그먼트 배열과 고유의 음성 조화 속성 등에서 향상된 결과를 나타냈습니다. 이로 인해 전문가 작곡가는 물론 아마추어 창작자에게도 더욱 실용적인 작업 환경을 제공합니다.



### Too Much of a Good Thing: When sim2real Efforts Impede Policy Learning (And What to Do About It) (https://arxiv.org/abs/2606.02636)
- **What's New**: 이번 논문에서는 시뮬레이션에서 실제 환경으로 정책을 전이하는 과정(sim2real)이 정책 학습에 부정적인 영향을 미칠 수 있음을 주장하고 있습니다. 연구자들은 시뮬레이터에 종속되는 현상(simulator lock-in)과 비효율적인 정책 탐색이 발생한다고 지적하며, 로봇의 기구학(kinematics)을 주요 설계 제약으로 활용하는 sim2sim2real 패러다임을 제안합니다.

- **Technical Details**: 시뮬레이션에서 하드웨어로의 성공적인 정책 전이는 시스템 식별(system identification) 과정에서 미세한 매개변수 조정과 실험을 요구합니다. 각 시뮬레이터와 물리 엔진 간의 차이로 인해, 특정 시뮬레이터에서의 시스템 식별 결과가 다른 시뮬레이터에 그대로 적용되지 않을 수 있으며, 이로 인해 연구자들은 특정 시뮬레이터에 과도하게 집중하게 됩니다. 이 연구는 기존 패러다임의 단점을 극복하기 위해 다양한 시뮬레이터 간의 정책 전이를 쉽게 할 수 있는 새로운 접근 방식을 제안합니다.

- **Performance Highlights**: 사례 연구로 아폴로 휴머노이드 로봇의 보행 정책을 학습하고 이를 다른 모델로 전이하는 과정을 설명하고 있습니다. IsaacLab에서 학습한 정책을 MuJoCo의 검증된 전체 모델로 전이하여, 고차원에서 다양한 동역학을 실험했습니다. 이 과정에서, 정책 학습이 간소화되고 시뮬레이터에 대한 의존도와 엔지니어링 부담이 크게 줄어드는 것을 확인했습니다.



### Echo-POSED: Geometric Self-Distillation for Echocardiography Guidanc (https://arxiv.org/abs/2606.02634)
- **What's New**: 새롭게 소개된 Echo-POSED는 실시간 경흉부 초음파(TTE) 가이드를 위한 자가 지도 학습(Self-supervised learning) 프레임워크로, 전문가 레이블이 없는 2D 초음파 이미지에서 프로브 조정을 직접 추천합니다. 이 시스템은 규칙적으로 수집된 3D 초음파 볼륨에서 잘라낸 2D 뷰로 훈련되며, 프로브의 움직임에 대해 동등성을 유지하고 심장 주기에 대해서는 불변성을 주어 $	ext{SO}(3)	imes	ext{SO}(3)$에서 포즈 표현(pose representation)을 생성합니다.

- **Technical Details**: Echo-POSED는 초음파 프로브의 모션을 기술하는 3D 회전과 이동이 포함된 강체 변환(rigid transformation)으로 구성되며, 환자의 가슴과의 접촉을 유지합니다. 그런 다음, 좌심실을 중심으로 하는 반지름 r을 가진 구(S2) 내에서의 전이를 제한하고, 이를 통해 최소한의 공간인 $	ext{SO}(3)	imes	ext{S}^{2}$로 표기합니다. 이 프레임워크는 수동 주석이나 로봇 궤적 데이터 없이도 작동하며, 3D 초음파 데이터셋을 통해 재현 가능한 연구를 지원합니다.

- **Performance Highlights**: Echo-POSED는 가상 교란 하에서도 기하학적 일관성을 유지하며, 환자 내 및 환자 간의 가이드를 시뮬레이션할 수 있도록 합니다. 실제로 이 시스템은 심장 운동을 포함하는 심장 내 시뮬레이션에서 가이드된 뷰와 목표 뷰 간의 평균 각도 오차가 8.2도에 달하는 성과를 달성했습니다.



### Position: Prioritize Identifying Structure, Not Complex Models, for Scientific Discovery (https://arxiv.org/abs/2606.02632)
Comments:
          Will appear as a position paper in ICML

- **What's New**: 이 논문은 현대 기계 학습(ML) 및 인공지능(AI) 모델, 특히 대형 언어 모델(LLMs)이 관찰 데이터에서 과학적 가설 및 기계적 설명을 생성하는 데 사용되는 현상을 다룹니다. 특히, 이 논문은 ML이 높은 차원에서 예측적 성공과 일관된 설명이 기계 발견의 충분한 증거가 아님을 강조합니다. LLMs의 경우, 다양한 설명이 단일 유창한 서사로 축소되는 경향이 도리어 문제를 더욱 복잡하게 만든다고 주장합니다.

- **Technical Details**: 자연 과학 분야에서도 ML 도구를 이용한 연구가 증가하고 있으나, 기계적 탐구에서 명확한 식별 구조가 없이는 유의미한 발견이 어려울 수 있다는 점을 강조합니다. 논문은 기계적 질의에 대해 명확한 식별 구조를 설정할 것을 주장하며, 이는 관측된 데이터를 기반으로 질문의 답변을 가능하게 한다고 설명합니다. 기계적 질의는 데이터 생성 과정에 대한 구체적인 질문으로 정의되며, 이는 통계적 가정 및 관측 증거에 기반을 두고 있습니다.

- **Performance Highlights**: LLMs의 등장으로 인해 다양한 기계적 설명이 통합되는 경향이 있지만, 이는 명확한 기계적 물음에 대한 해답을 줄 가능성을 낮춥니다. 기계적 학습을 지원하기 위해서는 식별 구조를 연구하고 선언해야 하며, 이를 통해 기계적 주장에 대한 검증 가능성을 높이는 것이 중요합니다. 이 연구는 특히 ML 시스템이 과학적 발견에 기여하기 위해서는 기계적 학습을 식별 문제로 인식해야 한다고 강조합니다.



### Wavelet as Tokenizer: Preliminary Results on a Shared Wavelet Token Schema for Natural Signals (https://arxiv.org/abs/2606.02631)
Comments:
          12 pages, 3 figures

- **What's New**: 이번 연구는 오디오, 이미지, 비디오가 각각의 모달리티에 국한된 잠재 그리드를 이용하는 대신, 공통의 wavelet token schema를 공유할 수 있는지를 탐구합니다. Haar DWT/IDWT 프론트엔드, 공유된 계수 토큰 레이아웃, 선택적 구조 메타데이터 등을 포함한 초기 연속 토큰 모델을 소개합니다. 이 모델은 Speech Commands, EuroSAT RGB, DAVIS 2017 데이터를 이용한 실험에서 각각 39.92 dB, 29.37 dB, 23.93 dB의 PSNR을 기록했습니다.

- **Technical Details**: 이 논문은 Wavelet as Tokenizer (WAT)라는 다중 스케일 토큰화 프레임워크를 제안합니다. 각 모달리티는 샘플링된 또는 매개변수화된 필드로 간주되며, wavelet 변환이 로컬화된 다중 스케일 계수로 매핑됩니다. 계수 블록과 그 구조적 메타데이터를 결합하여 토큰을 형성하며, 이를 통해 각 신호의 변화를 나타내는 간결한 정보를 제공합니다.

- **Performance Highlights**: 여러 시험 결과는 공통의 토큰 트렁크가 오디오, 이미지, 비디오를 복원하는 데 효과적임을 보여주었습니다. 또한 고정 비율의 에너지 선택은 평균 PSNR을 상당히 향상시킨다는 강력한 기준선을 제공합니다. 마스크드 스파스 훈련은 50%의 밀집 토큰으로 34.45 dB 비디오 PSNR에 도달하였으며, 이는 통합된 wavelet 토큰 스키마의 유용성을 뒷받침하고 있습니다.



### MultiTurnPSB: Evaluating Multi-Turn Jailbreak Attacks an dClassifier-Based Defenses for Medical AI Safety (https://arxiv.org/abs/2606.02630)
- **What's New**: 이 논문은 MultiTurnPSB라는 네 번의 발화로 구성된 의료 안전 벤치마크를 소개하고, 실제 사용자의 상황에서 모델의 안전성을 평가하는 데 초점을 맞추고 있습니다. 기존의 단일 발화 평가에서는 모델의 응답이 실제 상호작용에서의 사용자 압박을 반영하지 못하는 점을 지적합니다. 연구 결과, 사용자의 접근 방식에 따라 모델의 안전 응답률이 크게 감소할 수 있음을 확인하였습니다.

- **Technical Details**: MultiTurnPSB는 PatientSafetyBench의 네 번의 대화 형식으로 변환하여, 사용자가 정서적 압박, 권위적 언급, 긴급성 등을 추가할 때 모델이 어떻게 반응하는지를 측정합니다. 이 연구는 모델의 응답에 대한 평가를 단일 발화가 아닌 다회 대화 형식으로 새롭게 제안하고, 4회차에서의 응답이 안전하지 않을 확률이 52%까지 증가한다고 보고합니다.

- **Performance Highlights**: 연구 결과에 따르면, GPT-4.1-mini 모델은 고정형 템플릿 공격에서 34.5%의 비안전 응답률이 58.6%로 증가하며, 라이브 적대적 공격에서는 77.7%에 도달합니다. 특히, 다회 대화 시 유사한 공격자에 대해 서로 다른 모델들이 다른 성능 궤적을 보이는 것으로 나타났습니다. 입력 측 분류기 개입을 통해 비안전 응답을 52% 줄일 수 있었으나, 허위 경고 비율이 45%에 달하는 문제가 발생하여 배포에 제약을 받는 상황입니다.



### Enhancing Protein-Protein Interaction Prediction with Hierarchical Motif-based Multimodal Protein Embedding (https://arxiv.org/abs/2606.02629)
- **What's New**: 이 논문에서는 기존의 단백질-단백질 상호작용(Protein-Protein Interactions, PPI) 예측 방식의 두 가지 주요 한계를 극복하기 위해 MMM-PPI(계층적 모티프 기반 다중 모달 단백질 인코더)를 제안합니다. MMM-PPI는 저원형 다중 모달 방식으로 PPI 임베딩을 구축하여 미시적, 중시적, 거시적 규모에서 정보 통합을 최적화합니다. 특히, 이 방법은 특정 중시적 모티프를 활용하여 단백질 간의 상호작용을 보다 정확히 예측할 수 있도록 설계되었습니다.

- **Technical Details**: MMM-PPI는 세 가지 규모에서 단백질 특성을 다루는 계층적 접근 방식을 채택합니다. 미세 규모에서는 서열, 구조 및 기능 모달리티에서 잔유(residue) 특성을 추출하고, 중시 규모에서는 이들 잔유를 공간 정보를 반영한 모티프 임베딩으로 통합합니다. 마지막으로, 거시적 규모에서는 모티프를 공동 주목(attention) 메커니즘을 통해 결합하여 단백질 임베딩으로 변환합니다, 이를 통해 PPI 예측의 정확성을 높이고 있습니다.

- **Performance Highlights**: 다양한 PPI 데이터셋에 대한 광범위한 실험을 통해 MMM-PPI는 기존의 최첨단 다중 라벨 PPI 예측 모델을 능가함을 입증했습니다. 특히, 데이터 분할이 도전적이고 훈련 데이터가 제한적인 시나리오에서 뛰어난 성과를 보였습니다. 이러한 결과는 MMM-PPI가 PPI 예측의 새로운 기준을 제시하며, 널리 활용 가능한 모델임을 나타냅니다.



### DXA-Derived Skeletal Phenotypes and Hip Fracture Risk: A Backdoor-Adjusted Causal Analysis (https://arxiv.org/abs/2606.02625)
Comments:
          35 pages; main manuscript includes 4 figures and 3 tables; supplementary material includes 13 figures and 3 tables

- **What's New**: 이 연구는 DXA(이중 에너지 X선 흡수법)에 의해 유도된 고관절 골격 표현형(phenotypes)과 고관절 골절 위험 간의 관계를 비교하며, 사전 지정된 혼란 변수(variables)를 조정하여 ATE(평균 치료 효과)로 평가한 것이 특징입니다. 또한, ATE 순위에 따른 표현형들이 위험 계층화를 개선할 수 있는지를 검토하였습니다.

- **Technical Details**: 21,098명의 UK Biobank 참가자를 대상으로 하여 고관절 관련 지역에서의 골 광물 함량(BMC), 골 밀도(BMD), T-스코어를 포함한 16개 골격 표현형을 분석하였습니다. 혼란 변수 선정은 사전 지정된 유도 비순환 그래프(DAG)에 의해 안내되었으며, 표준 편차(SD) 증가에 따른 절대 위험 차(scale)에서의 backdoor 조정 ATE를 추정하였습니다.

- **Performance Highlights**: 연구 참가자 중 115명이 고관절 골절을 경험하였으며, 모든 16개 표현형에서 SD 증가에 따른 부정적인 backdoor 조정 ATE가 관찰되었습니다. 특히, 전체 대퇴골 BMC와 BMD에서 각각 -0.0047의 가장 큰 ATE가 관측되어, 1,000명당 약 4.7건의 고관절 골절 감소와 일치했습니다. 또한, 임상 변수와 ATE 순위 상위 11개 표현형을 결합한 예측에서 FRAX 모델보다 더 높은 AUC 값을 기록하였습니다.



### TadA-Bench: A Million-Variant Benchmark for Future-Round Discovery Toward Agentic Protein Engineering (https://arxiv.org/abs/2606.02624)
Comments:
          Accepted at the 43rd International Conference on Machine Learning (ICML 2026). Data: this https URL . Code: this https URL

- **What's New**: TadA-Bench는 단백질 공학의 미래 실험을 우선시하는 특성 기반벤치마크로, 31회의 TadA 유전자 변형 진화 라운드로부터 밀리언 변이를 기록한 데이터셋이다. 이 벤치마크는 역사적 캠페인의 체계적인 재현을 통해 새로운 실험 디자인을 지원하며, DNA, RNA, 단백질의 정렬된 뷰를 제공한다. 또한, Seq2Graph라는 그래프 기반 파이프라인을 통해 노이즈가 섞인 데이터에서 일관성 있는 크로스 라운드 활동 레이블을 생성하는 것이 특징이다.

- **Technical Details**: TadA-Bench는 이전 실험 결과를 바탕으로 후속 라운드에서 변이를 순위 매기는 작업을 지정하는 고정 데이터 재생 태스크를 정의한다. Seq2Graph는 소음이 있는 다중 라운드 시퀀싱을 안정적인 레이블 공간으로 변환하는 데이터 통합 파이프라인으로, 그래프 엣지를 사용해 스코어 전파 및 일관성 수정을 수행한다. 경계를 정하고 체계적으로 실험 기록을 평가하기 위해 각종 데이터의 신뢰성을 확보하는 데 중점을 두었다.

- **Performance Highlights**: 생물학적 언어 모델이 미래 라운드의 발견에 어려움을 겪는 것이 확인되었으며, 모델이 무작위 분할 제어에서는 강한 순위 상관관계를 이루었으나, 이후 라운드 평가에서는 심각하게 성능이 저하되었다. 실험 결과, 진화적 커버리지가 동일한 크기의 분석에서 밀집 지역 샘플링보다 더 유익하다는 것이 밝혀졌으며, TadA-Bench의 설계가 실험적 캠페인의 구조와 일관성을 유지하는 것이 중요하다. 이 결과는 단백질 발견을 위한 벤치마크 설계에서 커버리지 기반 접근이 필수적임을 강조한다.



### Oscillatory State-Space Models as Inductive Biases for Physics-Informed Neural PDE Solvers (https://arxiv.org/abs/2606.02623)
- **What's New**: 이번 연구에서는 진동 상태 공간 역학(oscillatory state-space dynamics)을 통합하여 PDE(편미분방정식) 솔루션의 모드 구조를 나타내는 새로운 PINN(물리 정보 신경망) 접근법을 제안합니다. 이 접근법은 선형 진동기(linear-oscillator) 기반의 시간 진화를 활용하며, 공간에서는 PDE 인식 스펙트럼 기저(PDE-aware spectral basis)를 사용합니다. 이를 통해 경계 조건을 일관되게 적용하고 폐쇄형 공간 미분(closed-form spatial differentiation)을 가능하게 합니다.

- **Technical Details**: 제안된 OSSM-PINN 아키텍처는 시간 진화를 선형 진동 상태 공간(LinOSS) 블록을 통해 수행하며, 문제 군에 맞춰 고정된 해석 기반(fixed analytical basis)을 사용하여 공간 표현을 제공합니다. 이 아키텍처는 시간 변화하는 모드 계수(modal coefficients)를 학습하도록 설계되어 있습니다. 이렇게 하면 연속적인 공간-시간 매핑 대신, 잠재 공간(latent space)에서의 모드 구조가 유지됩니다.

- **Performance Highlights**: 극한의 차원에서의 비정형 경계 및 대규모 PDE 문제를 포함하여 다양한 벤치마크에 대해 평가한 결과, 제안된 아키텍처가 이전의 PINN 접근법들보다 더 나은 정확도와 메모리 효율성을 보여주었습니다. OSSM-PINN은 하나의 훈련 레시피로 여러 PINN 실패 모드를 해결하며, PDE 특성 주파수를 회복하는 특징이 있습니다.



### Closed-Loop Molecular Design with Calibrated Deferenc (https://arxiv.org/abs/2606.02618)
- **What's New**: 본 논문에서는 Cognitive Loop via In-Situ Optimization (CLIO)를 소개합니다. CLIO는 지속적으로 업데이트되는 신념 상태 그래프(belief-state graph)와 재귀적인 계획-실행 루프를 결합하여, 기존의 도구가 실패하는 경우를 인지하고 전략을 조정하여 실험적 수정을 안내하는 기계적 가설(mechanistic hypotheses)을 생성할 수 있는 사고적인 에이전트를 만듭니다.

- **Technical Details**: CLIO는 구조화된 신념 상태 그래프를 통해 지속적인 메모리를 장착하고 신뢰를 조율하는 명시적인 정책을 갖추고 있습니다. 이를 통해 CLIO는 AORFB(수용성 유기 산화 환원 흐름 배터리)의 설계를 위해 실험적 증거에 따라 신뢰를 조정하며 종합적인 최적화를 수행했습니다. CLIO는 다각적 이론적 가설을 생성하고 실험적 데이터를 해석하여 최적의 후보 물질을 제안했습니다.

- **Performance Highlights**: CLIO는 17개의 후보 물질을 설계 및 평가한 결과, 최종적으로 문헌 기준보다 130 mV 향상된 산화 환원 전위를 가진 최우수 인산염 물질을 발견했습니다. 하지만 예상치 못한 전기화학적 가역성 저하가 관측되었고, 이에 대한 대안으로 설폰산염으로의 교체를 제안한 후, 최종적으로 디자이-제작-테스트-재설계 루프(design-make-test-redesign loop)를 완성했습니다. 이 과정에서 CLIO는 증거에 기반한 신뢰 재조정(calibrated deference) 패턴을 보였습니다.



### FSA-GRPO: Teaching Auditory LLMs to Use Few-shot Demonstrations (https://arxiv.org/abs/2606.02615)
- **What's New**: 본 논문에서는 아동 음성 인식과 같은 저자원(task) 작업에 대처하기 위해 Few-Shot Aware GRPO (FSA-GRPO)라는 RL 기반의 후속 훈련(post-training) 방법을 제안합니다. 기존의 청각 대형 언어 모델들이 Few-shot prompting에 효과적으로 적응하지 못한다는 한계를 해결하고자 합니다. 이 방법은 특히 아동 음성 인식 뿐만 아니라 음성 번역과 오디오 이해에서도 성능 향상을 보여줍니다.

- **Technical Details**: FSA-GRPO는 RO 기반의 모델 불문(post-training) 접근 방식으로, 고자원 ASR 데이터에서 Few-shot 훈련 인스턴스를 구축하고 보조 보상(auxiliary reward)을 도입하여 모델이 생성 과정에서 이러한 시연(demonstrations)을 활용하도록 유도합니다. 이는 ICL(인컨텍스트 학습) 능력을 강화하면서도 직접 추론 성능을 유지하기 위해 설계되었습니다. 또한, 데이터 선택(data selection)과 보조 보상 가중치(auxiliary reward weighting) 검토를 통해 효과적인 훈련 레시피를 식별하였습니다.

- **Performance Highlights**: FSA-GRPO는 제한된 저자원 ASR 평가 설정 아래에서 관련된 도메인 외 데이터에 대한 직접 미세 조정(direct fine-tuning)보다 더 효과적인 적응 전략을 제공합니다. 이 연구는 고품질의 Few-shot 훈련 인스턴스를 선택하여 훈련 효율성을 향상시키는 것과 보조 시연-aware 보상이 어떻게 최적화에 영향을 미치는지 분석합니다. 실험 결과, FSA-GRPO는 인도메인 데이터가 없거나 사용할 수 없을 때 더 효과적인 성능 향상을 보여줍니다.



### Margin Play: A Multi-Agent System For Public Policy Analysis In The Brazilian Equatorial Margin (https://arxiv.org/abs/2606.02614)
- **What's New**: 브라질 적도 해안(BEM)의 탐사가 2026년 시작될 예정이다. 이 연구는 자원의 착취가 마라냥 주(州)에게 얼마나 긍정적인 외부효과(net positive externalities)를 줄 수 있는지를 분석한다. 연구는 다수의 주체(multi-agent) 간의 갈등을 시뮬레이션하는 Multi-Agent Reinforcement Learning (MARL) 시스템인 Margin Play를 통해 이루어진다.

- **Technical Details**: Margin Play는 여섯 개의 대리인(agent)을 갖는 구조로, 각 대리인은 고유한 목표 함수(objective function)를 가진다. 이 시스템은 중앙집중식 훈련(centralized training)을 통해 전체 시스템의 상태를 관찰하고, 훈련 중에는 고변동(return regimes) 속성을 다루기 위해 TQC(Truncated Quantile Critics) 방식을 적용한다. 연구의 모든 알고리즘은 브라질의 실제 데이터로 보정된다.

- **Performance Highlights**: 연구 결과, 여러 시나리오를 통해 BEM 탐사가 마라냥 주에게 미치는 영향은 제도적 체계(institutional regime)에 따라 달라진다고 나타났다. 특정 정책 시나리오(MA-Prospero)에서는 복지 증가가 17.5%에 이르렀고, 환경 책임성은 낮았다. 이는 탐사와 복지 간의 단순한 트레이드오프(trade-off)가 아닌 탐사에 연결된 공공 정책 체계의 선택 문제임을 강조한다.



### Samudra 2: Scaling Ocean Emulators across Resolutions (https://arxiv.org/abs/2606.02610)
- **What's New**: Samudra 2는 고해상도의 해양 예측 기능을 제공하는 향상된 인공지능 에뮬레이터로, 이전의 Samudra와 비교하여 더욱 폭넓은 U-Net 아키텍처와 동적 손실 함수(dynamic loss function)를 통합하여 나타나는 두 가지 문제, 즉 변동성 붕괴(variance collapse)와 각인 아티팩트(imprinting artifacts)를 해결했습니다. 이러한 두 가지 수정 사항은 Samudra의 기능을 확장하고, 더 적합하고 정밀한 예측을 가능하게 합니다.

- **Technical Details**: Samudra 2는 수정된 ConvNeXt 스타일 블록을 이용한 넓은 U-Net 백본을 도입하고, 각 채널 예측 오차의 역수에 따라 가중치를 조정하는 동적 손실 함수를 구현했습니다. 이 아키텍처는 1도(1º) 해상도에서 상위 해양 글로벌 평균 온도(R^2) 수치를 0.56에서 0.87로 증가시켰으며, 심해 온도 오차를 약 7배 감소시켰습니다.

- **Performance Highlights**: Samudra 2는 단일 GPU에서 운영되며, 해수면 예측, 해양 열 흡수, 기후 변동성 연구를 위한 더 큰 앙상블을 가능하게 합니다. 1/2도(1/2º) 및 1/4도(1/4º) 해상도로 스케일링할 수 있어, mesoscale (메소스케일) 소용돌이 및 뚜렷한 서부 경계 전류를 복원하는 데 성공했습니다. 이와 같은 성능 향상으로 인해 새로운 기후 예측과 시나리오 분석의 가능성이 더욱 넓어졌습니다.



### Building Better Activation Oracles (https://arxiv.org/abs/2606.02609)
Comments:
          Jan Bauer and Celeste De Schamphelaere contributed equally; author order determined randomly

- **What's New**: 본 논문에서는 Activation Oracles (AOs)의 훈련 방법을 네 가지 주요 방식으로 개선하여 더 나은 성능과 결과를 도모합니다. 주요 개선 사항으로는 정책 기반 훈련(on-policy training), 개선된 대화형(dataset), 더 많은 레이어의 피드 및 주입 포뮬라(injection formula)의 개선이 있습니다. 이러한 개선은 AOs의 품질을 평가하기 위한 첫 번째 종합 평가 도구인 AObench도 공개하고 있으며, 이는 텍스트 역전(confound) 문제를 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: AOs는 원래 LLM의 활성화를 입력으로 받아 자연어 질문에 답변하는 미세 조정된 모델입니다. 그러나 현재의 AOs는 환각(hallucination), 모호성(vagueness) 및 검증 가능한 신뢰성 부족과 같은 여러 문제에 직면해 있습니다. 새로운 대화형 데이터셋을 구축하고 여러 레이어의 활성화를 공급하며, 규범 일치 주입 공식을 강화함으로써 이러한 문제를 부분적으로 해결했다고 설명합니다.

- **Performance Highlights**: 개선된 모델은 원래 AOs보다 더 높은 점수를 기록하며 지침을 더 잘 따르고, 환각은 감소하였으며, 모호성이 크게 줄어든 것을 확인했습니다. 이러한 진전은 양적 평가와 질적 테스트 모두에서 나타났으며, 새로운 데이터셋의 효과를 입증하였습니다. 궁극적으로, 이 연구는 AOs와 다른 모델들이 더 나은 해석 가능성을 갖도록 기반을 마련하는 데 기여할 것으로 기대됩니다.



### Geometry-Aware Tabular Diffusion (https://arxiv.org/abs/2606.02607)
Comments:
          Accepted to the ICML 2026 main track. 24 pages, 10 figures, 22 tables

- **What's New**: 본 논문에서는 Geometry-Aware Tabular Diffusion (GATD)라는 새로운 방법론을 제안합니다. 이는 열 값의 차이에서 계산된 쌍별 각도 및 길이를 입력 및 보조 목표로 사용하여 탭룰(표형) 확산 모델의 성능을 향상시킵니다. 기존 방법들에 비해 평균 3.5배 더 적은 파라미터를 사용하면서도 최고의 기준 성능을 달성했습니다. 또한 GATD는 다양한 아키텍처와 함께 사용할 수 있는 특별한 관계적 감독(supervision)을 도입하여 효과를 높였습니다.

- **Technical Details**: GATD는 쌍별 기하학적 특징을 입력으로 제공하며, 각 특징은 열 간의 방향적 관계를 포착하는 각도와 크기를 나타내는 길이를 포함합니다. 이 논문에서는 MLP, GNN, Transformer와 같은 다양한 확산 노이저(denoiser)에 걸쳐 기하학적 신호를 평가하고 성능을 비교했습니다. 기존의 주의(attention) 메커니즘과는 달리, GATD는 명시적인 기하학적 감독을 통해 노이저에 대한 부담을 줄이고, 관계적 추론에서의 이동성을 보장합니다.

- **Performance Highlights**: GATD는 10개의 데이터 세트에서 8/10 Shape, 7/10 Trend 및 9/10 downstream utility(F1/RMSE) 평가에서 승리했습니다. Shape 및 Trend 에러는 각각 27% 및 20% 감소하였고, 다양항 아키텍처 간에 성능을 전이시키는 능력을 입증하였습니다. 기하학적 감독이 없을 경우 성능이 개선되지 않는 반면, 감독이 제공되자 성능이 크게 향상되었습니다.



### ReLoRA: Knowledge-Reusing Adaptation for Fast Rollout of Evolving LLM Services (https://arxiv.org/abs/2606.02606)
- **What's New**: 최근 대형 언어 모델(LLMs)의 지속적 발전과 서비스화에 따라, 업데이트된 모형에 대한 LoRA(저차원 조정) 어댑터의 재적용 문제가 대두되었습니다. 기존의 LoRA 어댑터는 구버전 모델에 최적화되어 있어 업데이트된 모델에 적용할 경우 성능 저하를 초래할 수 있습니다. 이러한 문제를 해결하기 위해, 본 논문에서는 ReLoRA라는 새로운 프레임워크를 제안하여 서비스 준비 상태의 LoRA 어댑터를 효율적으로 복원하고 과거 성능을 유지하거나 향상시키고자 합니다.

- **Technical Details**: ReLoRA는 두 가지 주요 최적화 단계를 포함합니다. 첫 번째는 베이지안 최적화를 활용하여 기존 LoRA 어댑터와 베이스 모델의 진화를 융합하여 호환성이 있는 초기 지점을 만드는 '적응형 LoRA 초기화(adaptive LoRA initialization)'입니다. 두 번째는 강력한 정규화를 통해 어댑터를 높은 품질의 영역으로 빠르게 끌어올린 후, 정규화를 완화하여 태스크별 세부 조정을 수행하는 '스케줄 정규화(fine-tuning with scheduled regularization)'입니다.

- **Performance Highlights**: 실험 결과, ReLoRA는 기존 기반 모델 대비 최대 8.9배 빠르게 서비스 준비 상태로 전환할 수 있으며, 정확성 또한 4.6% 향상시키는 것으로 나타났습니다. 이러한 성과는 LoRA 기반 LLM 서비스를 운영하는 서비스 제공자에게 큰 이점을 제공할 것으로 기대됩니다. 따라서 ReLoRA는 대규모 업데이트에 따른 재적응 문제를 효과적으로 해결하는 솔루션으로 자리 잡을 가능성이 큽니다.



### Cross-Modal Contrastive Learning of ECG and Angiography Representations for Severe Stenosis Classification (https://arxiv.org/abs/2606.02605)
- **What's New**: 이번 연구에서는 StenCE라는 프레임워크를 도입하여 심전도(ECG) 데이터에서 직접적으로 협심증 신호를 감지하는 방법을 제시합니다. 이 프레임워크는 X선 조영 검사의 학습된 특징과 대비하여 ECG 표현을 정렬하는 대조적 사전 학습을 통해 기능합니다. 이를 통해 비대칭 환자도 조기에 협심증을 진단할 수 있도록 지원합니다.

- **Technical Details**: StenCE는 ECG 인코더와 스텐오그램 인코더 간의 다중 모드 대조 학습(multi-modal contrastive learning)을 활용하여 등장시키며, ECG 입력만으로도 스텐오그램 분류가 가능하며 기존 연구를 뛰어넘는 성능을 보여줍니다. 이 접근 방식에 따라, 모델은 심각한 협심증을 감지하는 데 있어 AUC 0.822의 성능을 달성하였습니다.

- **Performance Highlights**: 검증 결과에 따르면, StenCE 프레임워크는 다양한 협심증 중증도 임계점에서 일관된 성능 향상을 나타냈습니다. 특히, 모든 작업에 대해 기존 대비 성능이 개선되었으며, 심각한 협심증 및 기타 심장 이상을 식별하는 데 있어 성과를 보여줍니다. 다만, 가벼운 협심증의 경우 성능이 저하되는 경향이 있으며, 이는 임상적 사용에는 아직 미흡함을 보여줍니다.



### Auditable Climate Risk Intelligence from Fragmented ESG Data: Deterministic Orchestration and Imbalance-Aware Learning for Scope 1-3 Validation (https://arxiv.org/abs/2606.02604)
Comments:
          22 pages, 7 figures. Preprint

- **What's New**: 이 논문에서는 ESG(환경, 사회 및 지배구조)와 기후 위험 데이터의 조화로운 통합을 위한 결정론적 프레임워크를 제안합니다. 기존의 불완전하고 파편화된 배포 시스템을 극복하기 위해, 이 프레임워크는 단일 진실의 출처 관리, 시간적 이상 탐지 및 균형이 잡힌 앙상블 학습을 통합하고 있습니다. 또한, 검증 과정의 투명성을 높이기 위해, SMOTE 기반의 희소 사건 최적화와 TreeSHAP 기반의 해석 가능성을 도입했습니다.

- **Technical Details**: 이 프레임워크는 ESG 검증을 위한 결정론적 오케스트레이션 및 프로비넌스(출처) 인식 거버넌스를 포함합니다. 실험 방법론으로는 시간적 드리프트 분석, SMOTE 기반의 최적화, 앙상블 학습 및 TreeSHAP 기반의 해석 가능성이 포함되어 있습니다. 이를 통해 감사 검토 및 감사 재구성을 위한 구조적 기반이 마련되며, 이는 파편화된 보고 환경에서도 안정적인 운영과 감사를 지원하는 데 목적이 있습니다.

- **Performance Highlights**: 프레임워크는 다섯 번의 교차 검증을 통해 분류(Classification) 메트릭(재현율, F1, ROC AUC)과 같은 성능 지표로 평가됩니다. 또한, 감사 과정의 추적 완전성 메트릭이 포함되어 해당 이상 사건에 대한 결정론적 출처 재구성을 측정합니다. 이 연구의 결과는 신뢰할 수 있는 기후 리스크 정보 시스템을 구축하는 데 기여합니다.



### Tracking Urban Atmospheric Pollutants using Sentinel-5P Satellite Data (https://arxiv.org/abs/2606.02592)
- **What's New**: 이 연구는 에콰도르 과야스 주의 도시 NO2 오염을 모니터링하기 위해 Sentinel-5P/TROPOMI 위성 데이터를 활용한 새로운 프레임워크를 제안합니다. 일반적인 표면 농도 추정 대신, 이 방법론은 중위수(median) 및 상위 백분위수(P90, P95, P99)를 사용하여 더 안정적이고 해석 가능한 대기 오염 지표를 생성하는 데 중점을 두고 있습니다. 이 연구는 입증된 접근 방식을 통해 도시 대기 품질을 평가할 수 있는 도구를 제공합니다.

- **Technical Details**: 제안된 방법론은 위성 원격 감지 데이터와 강력한 통계 분석을 결합하여 도시 NO2 오염의 시공간적 행동을 특성화합니다. 주의 행정 단위에 따라 데이터 가져오기, 전처리, 공간 집계 및 통계적 특성화 단계를 포함하며, 이를 통해 오류 이벤트를 줄이면서도 연도 간 대표성을 보존합니다. 분석은 비지도 학습 방법인 K-평균 클러스터링을 적용하여 유사한 오염 행동을 가진 지역을 그룹화합니다.

- **Performance Highlights**: 연구 결과에 따르면, 고도로 도시화된 관할 지역(cantons)은 낮은 도시화 지역보다 높은 NO2 극단값과 더 큰 변동성을 나타내며, 이러한 패턴을 통해 도시 전반의 오염 동적 변화를 이해할 수 있음을 보여줍니다. 제안된 방법론은 데이터가 부족한 지역에서도 사용할 수 있어, 향후 도시 대기 질 연구에 중요한 기여를 할 것으로 기대됩니다.



### Lean-GAP: A Dataset of Formalized Graduate Algebra Problems (https://arxiv.org/abs/2606.02588)
- **What's New**: 이번 연구는 Lean-GAP(Lean-Graduate Algebra Problems)라는 데이터셋을 통해, Dummit과 Foote의 교과서인 Abstract Algebra에서 430개의 수학 문제를 형식화했습니다. 이 과정에서는 PDF를 LaTeX로 변환하고 Lean 4로 자동 형식화한 뒤, 비공식적 수학 표현과 형식적 수학 표현 간의 일치를 검증하는 파이프라인을 구축했습니다. 특히, 형식화 검증 과정에서는 많은 인적 자원과 정교한 작업이 필요했음을 강조하였습니다.

- **Technical Details**: 우리는 PDF에서 LaTeX로의 변환, Lean 코드로의 자동 변환, 그리고 비공식 및 공식 쌍 검증이라는 세 단계로 구성된 파이프라인을 개발했습니다. 이 파이프라인에서 체크는 두 가지 단계의 인간 검토를 거쳐 이루어지며, 최종 결과물은 최소 두 명의 수학자에 의해 검증됩니다. 각 문제는 수학적인 의미의 정확한 변환을 보장하기 위한 세심한 관리가 필요하며, Lean 코드가 성공적으로 컴파일된다고 해서 의미까지 보장되는 것은 아닙니다.

- **Performance Highlights**: 현재까지 1,966개의 문제 중 430문제 이상이 형식화 완료되었습니다. 이 연구는 고차원적인 수학 문제를 형식화하는 데 있어 진행 중인 작업의 기초를 제공하며, 자동 수학적 추론 시스템 개발에 기여할 것입니다. Lean-GAP 데이터셋은 추후 공개될 예정이며, 교과서의 고전적인 문제를 아우르는 풍부한 자원으로 활용될 수 있을 것입니다.



### IdiomX A Multilingual Benchmark for Idiom Understanding, Retrieval, and Interpretation (https://arxiv.org/abs/2606.02584)
Comments:
          12 pages, 21 figures. Includes dataset and code. Resources available on HuggingFace, Kaggle, and GitHub

- **What's New**: 본 연구에서는 IdiomX라는 대규모 다국어 아이디엄(idiom) 이해, 검색 및 해석을 위한 벤치마크를 소개합니다. 이는 복제 가능한 다단계 파이프라인을 통해 구축되었으며, 기존 아이디엄 자원의 한계를 극복하고자 합니다. 이 데이터셋은 12,000개 이상의 아이디엄을 포함하여, 영어, 아랍어, 프랑스어의 알라인드(Aligned) 의미 표현을 제공합니다.

- **Technical Details**: IdiomX 데이터셋은 190,000개 이상의 맥락화된 예제를 포함하고 있으며, 언어 자원 추출, 대규모 정규화, 통제된 LLM(large language model) 보강, 구조화된 검증을 결합한 절차를 통해 제작되었습니다. 이 자원은 아이디엄 검출, 맥락에서 아이디엄 검색, 아랍어에서 영어로의 아이디엄 검색 및 아이디엄 해석 등 네 가지 주요 작업을 포함합니다.

- **Performance Highlights**: 실험 결과, 맥락 변환기(transformer) 모델이 아이디엄 검출에서 상당한 개선을 보였으며, 하이브리드 검색 및 재정렬 아키텍처가 단일 언어 및 다국어 아이디엄 검색 모두에서 효과적으로 성능을 강화했습니다. 또한 아이디엄 해석이 의미 검색 작업으로 모델링될 수 있으며, 이는 해석 가능성을 추가적인 벤치마크 차원으로 도입함을 보여주었습니다.



### Cost-Aware Query Routing in RAG: Empirical Analysis of Retrieval Depth Tradeoffs (https://arxiv.org/abs/2606.02581)
Comments:
          13 pages , 18 figures , 8 tables

- **What's New**: 이 논문은 Cost-Aware RAG (CA-RAG)라는 새로운 쿼리 라우팅 프레임워크를 소개합니다. CA-RAG는 다양한 'strategy bundles' 중에서 각 쿼리에 대해 적절한 검색 깊이를 선택하여, 비용(비용 절감), 품질(응답 유용성 극대화), 응답 시간(지연 최소화)의 세 가지 목표를 동시에 고려합니다. 이는 다양한 쿼리 유형에 대한 효율성을 향상시킬 수 있는 가능성을 보여줍니다.

- **Technical Details**: CA-RAG는 FAISS 기반의 밀집 검색과 OpenAI의 채팅 및 임베딩 API를 사용해 구현되었습니다. 사용자가 정의한 우선 순위를 바탕으로 유틸리티를 평가하고, 각 쿼리에 최적화된 전략을 할당합니다. 이를 통해 각 쿼리에 대해 고유한 검색 깊이를 적용하여 더 나은 성능을 발휘합니다.

- **Performance Highlights**: CA-RAG는 28개의 쿼리 벤치마크 테스트에서 동작하며, 항상 깊은 검색을 사용하는 것보다 26% 적은 청구 토큰을 달성했고, 항상 직접 추론하는 방식보다 평균 34% 낮은 지연 시간을 기록했습니다. 이 아키텍처는 또한 더 복잡한 쿼리 유형의 정보 요구를 충족시키면서도 간단한 쿼리 유형에서 가장 큰 절감을 보여줍니다.



### TRAP: Hijacking VLA CoT-Reasoning via Adversarial Patches (https://arxiv.org/abs/2603.23117)
Comments:
          Accepted by ICML 2026

- **What's New**: 이 논문에서는 Chain-of-Thought (CoT) 추론 기반의 Vision-Language-Action (VLA) 모델의 보안 문제를 탐구합니다. CoT 추론이 사실상 사용자의 지시에 변화를 주지 않고도 로봇의 행동을 조종하는 새로운 공격 벡터를 도입하고 있다고 주장합니다. 특히, VLA 모델이 CoT를 통해 의도한 행동을 명확히 드러내므로 이를 악용하기好的(adv)한 가능성이 있다는 점을 지적합니다.

- **Technical Details**: 연구팀은 TRAP (CoT-Reasoning Adversarial Patch)라는 새로운 공격 방법론을 제안하며, 이는 CoT 추론 경로를 겨냥하여 행동 생성을 조작하는 방식입니다. TRAP은 일정한 형태의 adversarial patch (예: 테이블 위에 놓은 식탁보)를 통해 CoT 추론을 왜곡하며, 이를 통해 실제 공격을 수행할 수 있습니다. 이 방법론은 세 가지 대표적인 VLA 아키텍처에서 검증되었으며, 다양한 CoT 메커니즘을 커버합니다.

- **Performance Highlights**: 실제 환경에서 손으로 인쇄된 패치를 활용하여 TRAP의 효용을 입증하였습니다. 연구 결과는 CoT 추론이 VLA 시스템에서 급격히 보안성을 강화할 필요가 있음을 강조합니다. 이 접근 방식은 로봇이 잘못된 행동을 수행하게 만드는 등의 공격 행동을 유도할 수 있다는 점에서 큰 경각심을 일깨워 줍니다.



### AgentCL: Toward Rigorous Evaluation of Continual Learning in Language Agents (https://arxiv.org/abs/2606.02461)
Comments:
          10 pages in the main text, 26 pages in total

- **What's New**: 이 논문은 언어 에이전트의 지속적인 학습(Continual Learning, CL)을 평가하기 위한 새로운 프레임워크인 AgentCL을 소개합니다. 기존의 벤치마크들이 에이전트가 다양한 작업을 통해 학습한 경험을 효과적으로 검토하지 못하는 문제를 해결하려고 합니다. AgentCL은 명확한 작업 관계를 설정하고, 재사용 가능한 경험을 강조하여 에이전트가 미래 작업에서 그것을 잘 활용할 수 있는지를 측정합니다.

- **Technical Details**: AgentCL은 작업 스트림을 조절하고 플라스틱성(plasticity), 안정성(stability), 일반화(generalization) 등의 지표를 정량화할 수 있는 평가 프로토콜을 제공합니다. 에이전트는 작업을 해결하기 위해 메모리에서 유용한 정보를 검색하고, 환경과 상호작용하며 작업을 완료한 뒤에는 메모리를 업데이트합니다. MemProbe라는 방법론을 통해 메모리 설계의 다양한 요소가 지속적인 학습에 미치는 영향을 분석합니다.

- **Performance Highlights**: 실험 결과, 전통적인 임의의 작업 스트림에서는 메모리 설계 간의 성능 차이를 명확히 드러내기 어려웠지만, AgentCL의 구성된 작업 스트림에서는 각 설계의 플라스틱성이 잘 구분되었습니다. 기존의 메모리 설계는 안정성 문제를 자주 발생시켰고, 이러한 문제를 해결하기 위한 메모리 설계의 필요성이 강조되었습니다. 본 연구는 지속적인 학습을 위한 강력한 메모리 디자인의 필요성을 보여줍니다.




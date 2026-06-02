New uploads on arXiv(cs.CL)

### Are Large Reasoning Models Interruptible? (https://arxiv.org/abs/2510.11713)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문에서는 대형 추론 모델(Large Reasoning Models, LRMs)이 전통적으로 정적인 환경에서 평가되는 방법, 즉 '고정된 세계(frozen world)' 가정을 도전합니다. 연구팀은 현대의 추론 작업에서 시간 제약이 있는 동적 시나리오를 통해 LRMs의 내구성을 평가하고, 중단(interruptions)과 동적 맥락(dynamic context)의 두 가지 주요 개입 유형을 분석합니다. 이러한 접근을 통해 기존의 정적 평가 방식이 모델의 강인성을 과대평가한다는 이동을 발견했습니다.

- **Technical Details**: 연구에서는 수학 및 프로그래밍 과제를 포함한 동적 환경에서의 평가 프로토콜을 소개합니다. 특히 사용자가 긴 계산 중에 중단을 요청할 때 모델이 얼마나 잘 반응하는지, 그리고 계산 중에 제시된 새로운 정보가 모델의 최종 답변에 어떻게 통합되는지를 평가했습니다. 연구 결과, STATISTICALLY 정적 설정에 비해 동적 환경에서의 정확도가 최대 60%까지 하락하는 현상을 관찰했습니다.

- **Performance Highlights**: 이 연구에서 확인된 LRMs의 주요 마이너스 포인트는 '추론 새는 현상(reasoning leakage)', '패닉(panic)', 그리고 '자신의 불안(self-doubt)'입니다. 각 상황에서 모델들이 내놓는 답변의 질이 크게 저하될 수 있으며, 이러한 발견은 LRMs 개발의 새로운 방향성을 제시합니다. 또한 본 연구는 추론 중단이 모델 성능에 미치는 여러 흥미로운 영향을 분석하여, 실용적 AI 모델의 발전에 기여할 수 있는 기초 자료를 제공합니다.



### Demystifying Reinforcement Learning in Agentic Reasoning (https://arxiv.org/abs/2510.11701)
Comments:
          Code and models: this https URL

- **What's New**: 이번 연구에서는 에이전틱(Agentic) 강화 학습(RL)을 통해 대형 언어 모델(LLMs)의 추론 능력을 개선하는 방법을 다루고 있습니다. 데이터, 알고리즘, 추론 방식의 세 가지 관점에서 RL의 효과성을 체계적으로 분석하였으며, 특히 현실적 도구 사용 경로(real end-to-end tool-use trajectories)로 대체하는 것이 모델 성능을 크게 향상시킨다는 점을 강조합니다. 또한, 탐색 친화적인 기법이 에이전틱 RL의 훈련 효율성을 높이는 데 필수적이라는 점도 밝히고 있습니다.

- **Technical Details**: 연구에서는 에이전틱 RL의 훈련 목표를 공식화하고, 정책 LLM(Policy LLM) 및 참조 LLM(Reference LLM)과의 관계를 정의합니다. 연구는 GRPO(Gradient-Weighted Regression Policy Optimization) 알고리즘을 기반으로 하여, 손실 집계의 세분화(level)와 보상 형태를 최적화하여 LLM의 훈련 성능을 강화할 수 있는 방법을 탐색하고 있습니다. 데이터 준비 단계에서 실제 도구 사용 경로를 활용하여 모델 성능 향상을 위해 데이터 다양성을 고려한 분석이 이루어집니다.

- **Performance Highlights**: 연구팀은 DemyAgent-4B라는 강력한 베이스라인 모델을 제시하며, 이 모델이 SOTA(State-of-the-Art) 수준의 성능을 보여주고 더 큰 모델보다 우수한 에이전틱 추론 성능을 발휘할 수 있음을 증명했습니다. 다양한 벤치마크에서 실험을 통해, 작은 모델인 4B 규모의 모델이 32B 규모의 모델보다 우수한 성능을 보일 수 있다는 점을 밝혀내었습니다. 이는 앞으로의 에이전틱 RL 연구에 대한 실용적인 기준을 마련하는 데 기여할 것입니다.



### When Agents Trade: Live Multi-Market Trading Benchmark for LLM Agents (https://arxiv.org/abs/2510.11695)
- **What's New**: 본 논문에서는 금융 거래에서 사용되는 Large Language Model(LLM) 기반 에이전트의 실시간 평가를 위한 최초의 벤치마크 시스템인 Agent Market Arena(AMA)를 소개합니다. 기존 연구들이 모델 대신 에이전트를 평가하거나, 기간과 자산의 제한이 있었던 점을 보완합니다. AMA는 검증된 거래 데이터 및 전문가가 확인한 뉴스와 다양한 에이전트 아키텍처를 통합하여 공정한 비교를 가능하게 합니다.

- **Technical Details**: AMA는 InvestorAgent, TradeAgent, HedgeFundAgent 및 DeepFundAgent와 같은 네 가지 에이전트를 구현하여 다양한 리스크 스타일을 평가합니다. 각 에이전트는 특정한 거래 전략을 가지고 있으며, GPT-4o, GPT-4.1, Claude-3.5-haiku, Claude-sonnet-4, Gemini-2.0-flash와 같은 모델을 통해 시험됩니다. 이 시스템은 실시간 거래 환경에서 LLM 기반 에이전트의 성능을 평가합니다.

- **Performance Highlights**: 실시간 실험 결과, 암호화폐 및 주식 시장에서 에이전트 프레임워크는 공격적인 리스크 감수에서부터 보수적인 의사결정까지 상이한 행동 패턴을 보여줍니다. 모델 백본의 기여는 결과 변화에 비교적 적음을 확인했습니다. AMA는 LLM 기반 에이전트의 금융 추론 및 거래 지능의 평가를 위해 엄격하고 지속적으로 발전할 수 있는 기반을 확립합니다.



### ACADREASON: Exploring the Limits of Reasoning Models with Academic Research Problems (https://arxiv.org/abs/2510.11652)
- **What's New**: 최근 대규모 언어 모델(LLMs) 연구는 새로운 기능의 시연에서 복잡한 추론 및 도전적인 작업 해결로 초점이 이동하고 있습니다. 기존 평가 방법은 주로 수학/코드 경연이나 일반 작업에 치중되어 있으며, 다중 분야 학문 벤치마크는 충분한 추론 깊이가 부족합니다. 이러한 문제를 해결하기 위해서, 우리는 LLM과 에이전트의 학문적 지식 취득 및 추론 능력을 평가하기 위해 설계된 Acadreason 벤치마크를 소개합니다.

- **Technical Details**: Acadreason 벤치마크는 컴퓨터 과학, 경제학, 법학, 수학 및 철학을 포함하여 5개의 고급 추론 분야에서 50개의 전문가 주석 학문 문제로 구성됩니다. 각 질문은 최근 몇 년 동안의 수준 높은 학술 출판물에서 얻어지며, 엄격한 주석과 품질 관리를 거쳐 도전적이며 답변 가능한 문제를 보장합니다. 우리는 10개 이상의 주요 LLM 및 에이전트에 대한 체계적인 평가를 수행하였고, GPT-5조차 16점에 불과한 점수를 기록했습니다.

- **Performance Highlights**: 실험 결과, 대부분의 LLM들이 20점 이하의 점수를 기록한 반면, 에이전트도 40점을 초과한 모델이 없었습니다. 이러한 결과는 LLM과 에이전트 간의 현재의 능력 격차를 드러내고, Acadreason의 도전 과제를 강조합니다. 또한, 다양한 힌트를 도입하여 모델의 성능에 미치는 영향을 조사하며, 방법론 힌트가 가장 큰 성과를 보였음을 발견했습니다.



### Enhancing Long Chain-of-Thought Reasoning through Multi-Path Plan Aggregation (https://arxiv.org/abs/2510.11620)
- **What's New**: 이 논문에서는 언어 모델(LM)의 추론 시Chain-of-Thought(CoT) 길이를 늘림으로써 추론 능력을 향상시키는 방법을 제시합니다. 기존의 접근법들은 일반적으로 단일 전방 패스를 통해 전체 추론 체인을 생성하는 경향이 있는데, 이는 흔히 CoT 탈선 문제를 발생시킵니다. 저자들은 이러한 문제에 대한 통찰력을 바탕으로 여러 대안적 계획을 생성하고 집계하는 Multi-Path Plan Aggregation(MPPA) 프레임워크를 도입하여 더욱 견고한 계획 수립을 목표로 합니다.

- **Technical Details**: MPPA는 계획 탐색과 집계를 결합하여 단일 패스를 넘어서며, 각 계획 단계에서 여러 후보 계획을 생성하고 이를 집계하여 개선된 계획 단계로 발전시킵니다. 구체적으로, 모델은 동적으로 조정되는 토큰 위치에 따른 간격 스케줄을 따라 여러 후보 계획을 생성하고, 이를 정제된 계획 단계로 통합합니다. 또한, 학습 효율성을 유지하기 위해 최소 디자인을 채택하며, 기본 LM이 주요 정책 역할을 수행하고, 경량 LoRA 모듈이 계획 집계 정책을 구현합니다.

- **Performance Highlights**: 저자들은 다수의 수학, 과학, 논리적 추론 기준에서 광범위한 실험을 수행했으며, 10%의 SFT 데이터와 5%의 선호 쌍만으로도 DeepSeek-R1 증류 기준 및 결과 보상 RL 기준을 능가하는 성능을 보여주었습니다. 이 연구에서 제안된 방법은 복잡한 추론 작업에서 더 나은 안정성 및 정확성을 달성하는 데 기여하며, 상대적으로 작은 LMs를 활용하여 효과적인 단계 수준의 선호를 제공할 수 있음을 보여주었습니다.



### StoryBox: Collaborative Multi-Agent Simulation for Hybrid Bottom-Up Long-Form Story Generation Using Large Language Models (https://arxiv.org/abs/2510.11618)
Comments:
          Project: this https URL

- **What's New**: 이 논문에서는 다중 에이전트 시뮬레이션을 기반으로 한 하이브리드 바텀업(long-form) 이야기 생성 기법을 제안합니다. 이 접근 방식을 통해 생성된 이야기는 유기적으로 발전하며, 강력한 구조가 없는 기존의 탑다운(top-down) 방식과는 달리 사건과 캐릭터 개발의 자연스러운 흐름을 가능하게 합니다. 이 시스템은 10,000단어가 넘는 긴 이야기들을 생성할 수 있으며, 현재 이야기 생성 모델들이 직면한 여러 과제를 해결했습니다.

- **Technical Details**: 이 시스템은 이야기 생성을 위한 두 가지 주요 부분으로 구성됩니다: 다중 에이전트 상자에서의 상호작용을 시뮬레이션하는 부분과 생성된 사건들을 통합하여 일관성 있는 이야기로 만드는 스토리텔러 에이전트입니다. 각 캐릭터는 이름, 나이, 본능, 학습된 행동, 현재 상태 및 생활 방식으로 정의되며, 이러한 속성의 상호작용을 통해 사건들이 발생합니다. 또한, 비정상 행동 속성을 도입하여 캐릭터의 예외적인 행동을 통해 더 역동적이고 흥미로운 이야기가 생성될 수 있도록 합니다.

- **Performance Highlights**: 제안된 접근 방식은 기존의 이야기 생성 방식과 비교할 때 극적인 이점을 가지고 있습니다. 생성된 이야기는 깊이와 일관성을 유지하며, 다채롭고 의미 있는 사건들을 창출하여 캐릭터와 플롯을 풍부하게 만듭니다. 연구 결과, 이 시스템은 여러 메트릭에서 최첨단 성능을 달성하였으며, 자연스럽고 매력적인 긴 이야기를 생성하는 혁신적이고 확장 가능한 솔루션을 제공합니다.



### LLM-Oriented Token-Adaptive Knowledge Distillation (https://arxiv.org/abs/2510.11615)
Comments:
          15 pages, 4 figures

- **What's New**: 이번 연구에서는 Knowledge Distillation (KD)의 최신 동향과 한계를 다루고 있으며, LLM(대형 언어 모델)의 지식을 효과적으로 압축하기 위해 새로운 방법론인 LLM-Oriented Token-Adaptive Knowledge Distillation (AdaKD)를 제안합니다. 기존의 logit 기반 방법들이 정적인 접근 방식으로 인해 학생 모델의 동적인 학습 과정과 맞지 않음을 지적하면서, 각 토큰의 실시간 학습 상태에 맞춰 적응적인 지식 전이 과정이 필요하다고 강조합니다. AdaKD는 토큰의 난이도에 따라 디스틸레이션을 조정하여 지식 전이를 최적화하는 두 가지 모듈을 포함하고 있습니다.

- **Technical Details**: AdaKD 프레임워크는 두 가지 상호작용 모듈로 구성됩니다. 첫 번째 모듈인 Loss-Driven Adaptive Token Focusing (LATF)은 학생 모델의 학습 안정성을 모니터링하여 가장 가치 있는 토큰에 집중하게 만듭니다. 두 번째 모듈인 Inverse Difficulty Temperature Scaling (IDTS)은 각 토큰의 난이도에 따라 온도를 조정하여 학습 과정을 최적화하고, 오류 수정을 위한 적절한 온도 전략을 구현합니다. 이러한 메커니즘을 통해 AdaKD는 다양한 디스틸레이션 방법의 성능을 향상시킬 수 있습니다.

- **Performance Highlights**: AdaKD는 여러 모델 아키텍처와 벤치마크에서 다양한 디스틸레이션 기법에 일관되게 성능을 개선합니다. 기존의 각종 KD 방법들과 비교할 때, ADAKD는 동적인 절도 기법을 통해 지식 전이에 있어 더욱 효율적이고 안정적인 결과를 보여줍니다. 또한, AdakD는 각 토큰별로 최적의 온도 조정을 통해 더 높은 일반화 능력을 달성하고, 있기 때문에 이 방법론은 ILLM(대형 언어 모델)의 발전에 크게 기여할 것으로 기대됩니다.



### Deconstructing Attention: Investigating Design Principles for Effective Language Modeling (https://arxiv.org/abs/2510.11602)
- **What's New**: 이번 연구는 Transformer 언어 모델의 주의(attention) 메커니즘을 체계적으로 해체하여, 각 디자인 원칙의 필요성을 실험적으로 검증했습니다. 서로 다른 원칙들을 선택적으로 완화한 변형을 설계하여, 모든 레이어에 균일하게 적용하거나 일부 레이어에서만 표준 주의를 유지하는 하이브리드 아키텍처에서 테스트하였습니다. 이를 통해 주의 메커니즘의 기초를 깊이 이해하고, 더 간소화된 언어 모델 개발의 가능성을 제시합니다.

- **Technical Details**: 연구에서는 주의 메커니즘의 핵심 원칙인 위치 간 정보 혼합(token mixing), 입력에 적응하는 시퀀스 종속 활성화(sequence-dependent activations), 특정 수학적 형식(dot-product similarities 및 softmax weighting), 쿼리와 키의 결합을 분석했습니다. 여러 실험에서, 토큰을 혼합하는 메커니즘이 필수적임을 발견하였고, 그 없는 경우 모델이 거의 랜덤한 행동으로 붕괴됨을 확인하였습니다. 반면, 가정된 수학적 형식과 시퀀스 종속성은 특정 레이어에서만 보존될 경우 상당히 완화될 수 있습니다.

- **Performance Highlights**: 예상외로, 독립적으로 실패하는 변형도 표준 주의와 혼합될 경우 강력한 성능을 발휘할 수 있다는 점에서 협력 효과를 강조합니다. 이러한 결과는 주의 메커니즘의 실제 효과를 이해하는 데 기여하고, 성능 저하 없이 언어 모델을 단순화할 수 있는 새로운 방향을 열어줍니다.



### SemCSE-Multi: Multifaceted and Decodable Embeddings for Aspect-Specific and Interpretable Scientific Domain Mapping (https://arxiv.org/abs/2510.11599)
- **What's New**: 이 논문에서는 SemCSE-Multi라는 새로운 비지도 학습(unsupervised) 프레임워크를 제안하여 과학 초록의 다면적 임베딩(embeddings)을 생성합니다. 이 임베딩은 연구자가 필요한 특정 측면(aspect)을 명확히 하고 독립적으로 포착할 수 있도록 하여 세밀하고 조절 가능한 유사성 평가(similarity assessment)를 가능하게 합니다. 또한, 본 접근법은 과학 분야의 사용자 주도 시각화를 위한 적응적 기능을 제공하는 점도 특징입니다.

- **Technical Details**: 제안된 접근법은 각 연구 초록에 대해 аспект별 요약 문장을 생성하고 이는 임베딩 모델에 의해 의미적으로 유사한 요약이 임베딩 공간 내에서 근접하게 배치되도록 학습됩니다. 최종적으로, 이 аспект별 임베딩 기능은 단일 임베딩 모델로 통합되어 단일 전방 통과(forward pass)에서 여러 аспект 임베딩을 예측할 수 있게 됩니다. 또한, 임베딩을 자연어 설명으로 복원하는 디코딩 파이프라인을 도입하여 임베딩 공간의 해석 가능성을 크게 향상시킵니다.

- **Performance Highlights**: 이 연구는 주로 침입 생물학 분야에서 성능을 평가하였으며, 전문가의 지도를 받았습니다. 논문의 처음에 제안한 대로, 다양한 측면을 취합하여 사용자 맞춤의 시각화 및 결과 도출을 가능하게 함으로써, 사용자가 필요로 하는 특정 연구 방향에 대한 명확한 통찰을 제공합니다. 이러한 접근법은 기존 방법의 한계를 극복하며, 특히 저차원 시각화에서 비어 있는 영역의 의미 있는 텍스트 설명을 생성하는데 효과적임을 입증하였습니다.



### MeTA-LoRA: Data-Efficient Multi-Task Fine-Tuning for Large Language Models (https://arxiv.org/abs/2510.11598)
- **What's New**: MeTA-LoRA는 Low-Rank Adaptation (LoRA) 기반의 데이터 효율적인 프레임워크로, 다중 작업 학습에서의 데이터 효율성을 크게 향상시킵니다. 기존의 LoRA 방법들은 작업 별로 많은 양의 레이블 데이터가 필요하지만, MeTA-LoRA는 두 단계의 최적화 과정을 통해 이를 해결합니다. 첫 번째 단계에서는 각 작업의 적응을 빠르게 진행하고, 두 번째 단계에서는 여러 작업에서의 기울기를 집계하여 지식 이전을 촉진합니다.

- **Technical Details**: MeTA-LoRA는 두 단계의 최적화 구조로 설계되어 있습니다. 첫 번째 단계에서는 작업 별로 특정 LoRA 어댑터를 학습하여 소량의 데이터로 빠른 적응이 이루어지고, 두 번째 단계에서는 여러 작업의 기울기를 집계하여 공유 LoRA 어댑터를 업데이트합니다. 이러한 방식은 공통 패턴을 활용하여 데이터 사용량을 further 줄이며, 효율적인 학습을 가능하게 합니다.

- **Performance Highlights**: MeTA-LoRA는 다중 작업 학습 및 다국어 학습 시 기존의 LoRA 전체 데이터 미세 조정 방법의 성능을 초과하거나 동등한 성능을 발휘하며, 작업 별 데이터 요구량은 크게 줄였습니다. 실험 결과는 MeTA-LoRA의 효율성을 입증하며, 두 단계 최적화 프레임워크의 필요성을 강조합니다.



### Survey Response Generation: Generating Closed-Ended Survey Responses In-Silico with Large Language Models (https://arxiv.org/abs/2510.11586)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)을 사용한 인실리코 시뮬레이션에서 설문 응답 생성 방법이 예측된 응답에 미치는 영향을 체계적으로 조사합니다. 기존 연구들이 닫힌 질문에 대한 응답 생성에 초점을 맞추는 반면, 본 연구는 다양한 응답 생성 방법을 실험하여 그 차이를 분석합니다. 32백만 개의 시뮬레이션된 응답 데이터와 8가지 설문 응답 생성 방법을 사용하여 새로운 통찰을 제공합니다.

- **Technical Details**: 연구는 8가지 설문 응답 생성 방법과 4개의 정치적 태도 설문조사, 10개의 오픈-가중 언어 모델을 포함하여 이루어졌습니다. 이 논문에서는 각 방법이 개인 수준(Individual level) 및 하위 집단 수준(Subpopulation level)에서의 일치도에 미치는 영향을 평가합니다. 결과적으로, 제한된 생성 방법(Restricted Generation Methods)이 전반적으로 가장 우수한 성과를 보이며, 추론 출력(Reasoning output)이 항상 일치도를 향상시키는 것은 아님을 발견했습니다.

- **Performance Highlights**: 결과는 다양한 설문 응답 생성 방법이 시뮬레이션된 응답에 미치는 큰 영향을 강조합니다. 또한, 생산적이고 실용적인 설문 응답 생성 방법의 적용에 대한 권장 사항을 개발했습니다. 이 연구는 LLMs의 활용에서 설문 응답 생성 방법의 중요성을 재확인하였습니다.



### LLMAtKGE: Large Language Models as Explainable Attackers against Knowledge Graph Embeddings (https://arxiv.org/abs/2510.11584)
Comments:
          13 pages

- **What's New**: 이 논문은 LLMAtKGE라는 새로운 프레임워크를 제안합니다. 이는 Knowledge Graph Embeddings(KGE) 공격의 새로운 방법을 통해 공격 대상을 선택하고 인간이 이해할 수 있는 설명을 생성하는 기능을 가집니다. 기존의 공격 방법들은 설명이 부족하고 일반화에 한계가 있었으나, 이 연구는 LLM(대형 언어 모델)를 활용하여 이러한 문제를 해결합니다.

- **Technical Details**: LLMAtKGE는 제약된 입력 조건 하에서 사실(context) 정보를 충분히 제공하기 위해 구조화된 프롬프트 프레임워크를 설계했습니다. 이를 통해 공격을 다중 선택형 질문으로 명시화하며, KG(지식 그래프)의 사실적 증거를 포함합니다. 또한, 의미 기반 및 중심성 기반 필터를 도입하여 관련 정보를 유지하며 후보 집합을 압축합니다.

- **Performance Highlights**: 실험 결과, 제안된 공격 방법은 두 개의 인기 있는 지식 그래프 데이터셋을 활용하여 기존의 강력한 블랙박스 기준선보다 우수한 성능을 보였습니다. 추가로, 즉각적인 인간이 이해할 수 있는 설명을 생성할 수 있는 능력을 입증하며, 기존의 화이트박스 방법들과 비교하여 경쟁력을 보여줍니다. 이 연구는 다양한 모델의 적용 가능성을 시사하며, 코드 또한 공개되었습니다.



### Culturally-Aware Conversations: A Framework & Benchmark for LLMs (https://arxiv.org/abs/2510.11563)
Comments:
          To appear at the 4th HCI + NLP Workshop @ EMNLP

- **What's New**: 이 연구에서는 다문화 대화 설정에서 LLM(대형 언어 모델)의 문화적 적응 능력을 평가하기 위한 첫 번째 프레임워크와 벤치마크를 제시합니다. 기존의 벤치마크가 언어적 스타일과 문화적 맥락을 반영하지 않는 문제를 해결하기 위해, 이 연구는 새로운 평가 기준을 제안합니다. 연구팀은 문화적으로 다양한 채점자들에 의해 주석이 달린 데이터를 사용하는 새로운 데이터셋을 구축했습니다.

- **Technical Details**: Culturally-Aware Conversations (CAC) 프레임워크는 상황적, 관계적, 문화적 맥락에 따라 언어적 스타일을 형성하는 방식을 명확히 합니다. 이 프레임워크는 문화적 대화에서의 스타일을 비례적으로 표현하는 데 기반이 됩니다. 연구는 8개의 문화적 관점을 반영하여 48개 대화와 240개의 응답을 생성하며, 이를 통해 다양한 문화에서 스타일적 요구 사항을 측정하고 분석합니다.

- **Performance Highlights**: 연구 결과, 현재의 최고 LLM들이 문화적 대화 설정에서 문화적 적응에 어려움을 겪고 있음을 보여줍니다. 스타일적으로 적절한 응답을 생성하는 데 있어서 이 모델들은 여전히 Anglocentric 기준에 치우쳐 있으며, 실제 상호작용에서 요구되는 기대를 충족하지 못합니다. 본 연구는 새로운 벤치마크를 통해 LLM들의 문화적 의식 및 대화 동역학을 평가할 수 있는 기회를 마련합니다.



### Invisible Languages of the LLM Univers (https://arxiv.org/abs/2510.11557)
- **What's New**: 이번 연구에서 제안된 중요한 프레임워크는 언어의 생명력(vitality)과 디지털 존재(digitality)에 대한 실증적 측정을 포스트콜로니얼 이론(postcolonial theory)과 인식적 불의(epistemic injustice)와 연결하여 언어적 불평등이 우연히 발생한 것이 아니라 구조적이라는 점을 설명합니다. 또한, 7,613개 언어의 데이터 분석을 통해 강역(Strongholds), 디지털 에코(Digital Echoes), 사라지는 목소리(Fading Voices), 그리고 보이지 않는 거인(Invisible Giants)이라는 네 가지 범주를 식별했습니다.

- **Technical Details**: 연구에서 사용하는 두 차원의 프레임워크는 생명력과 디지털성을 독립적인 축으로 간주하며, 이는 사회적 인구 통계적 존재와 온라인 존재의 차이를 측정합니다. Vitality는 첫 번째 사용자 수 및 EGIDS 상태를 기반으로 하며, Digitality는 웹 페이지, 위키백과, 데이터셋 및 언어 아카이브의 전반적인 존재감을 측정합니다. 이 방법론을 통해 실제 언어의 힘과 디지털 존재 차이를 정량화할 수 있습니다.

- **Performance Highlights**: 연구 결과, 현재 LLM이 지배하는 AI 시스템 내에서 영어의 지배는 기술적 필연성이 아니라 힘의 구조의 산물임을 드러냈습니다. 이로 인해 디지털 언어 생명력이 저하되고 언어의 생존에 부정적인 피드백 루프를 발생시키고 있습니다. 마지막으로, 연구는 언어 기술의 탈식민화(decolonizing language technology)와 AI 혜택의 민주화(democratizing access to AI benefits)에 대한 시사점을 제시합니다.



### Information-Preserving Reformulation of Reasoning Traces for Antidistillation (https://arxiv.org/abs/2510.11545)
- **What's New**: 이 논문에서는 LLMs에 대한 최근 연구로, Reasoning Chains의 길이를 늘려 복잡한 작업에서 성능을 획기적으로 향상시킬 수 있음을 보여줍니다. 그러나 이러한 Reasoning 트레이스를 노출하는 경우 무단으로 이를 추출(ditalization)할 수 있는 취약점이 존재합니다. 이에 대한 해결책으로 제안된 PART는 정보 보존을 목표로 하는 새로운 방법론으로, 인지심리적 관점에서 SFT(Supervised Fine-Tuning)의 학습 방식을 고려합니다.

- **Technical Details**: PART의 핵심 아이디어는 Reasoning 트레이스를 두 단계로 재구성하는 것입니다. 첫 번째 단계에서는 'self-talk' 행동을 제거하고, 두 번째 단계에서는 중간 결론(sub-conclusions) 순서를 재배열하여 정보를 요약합니다. 이를 통해 기존 Reasoning 트레이스의 유용한 정보를 유지하면서도 distillation을 방해합니다. 작은 보조 모델이 이 재구성을 수행하도록 학습됩니다.

- **Performance Highlights**: 다양한 크기와 유형의 학생 모델에서 실험을 통해 PART 기반 재구성이 distillation을 효과적으로 저해한다는 것을 보였습니다. 예를 들어 32B 학생 모델의 경우 AIME 2024 벤치마크에서 성능이 54.17에서 46.88로 감소해 13.5%의 성능 저하를 기록했습니다. 이는 다양한 Reasoning 벤치마크에서 지속적으로 나타나는 현상입니다.



### An Encoder-Integrated PhoBERT with Graph Attention for Vietnamese Token-Level Classification (https://arxiv.org/abs/2510.11537)
Comments:
          11 pages, 1 figure. Submitted to VLSP 2025 and reviewed

- **What's New**: 본 연구에서는 TextGraphFuseGAT라는 새로운 신경망 아키텍처를 제안합니다. 이 모델은 사전 학습된 Transformer 인코더인 PhoBERT를 Graph Attention Networks(GAT)과 통합하여 토큰 수준의 분류 작업을 수행합니다. 본 연구는 다양한 도메인에서의 성능 향상을 위해 전통적인 시퀀스 모델링을 그래프 기반 관계 모델링과 결합하는 효과를 입증합니다.

- **Technical Details**: TextGraphFuseGAT는 PhoBERT가 생성한 토큰 임베딩에 대해 완전 연결 그래프를 구성하여 GAT층이 토큰 간의 풍부한 의존 관계를 캡처할 수 있도록 합니다. 또한 그래프 향상 임베딩 위에 Transformer 스타일의 self-attention 레이어를 적용하여 문맥화를 개선합니다. 최종 토큰 표현은 분류 헤드를 통해 전달되어 시퀀스 레이블링을 수행합니다.

- **Performance Highlights**: 이 모델은 세 가지 베트남어 벤치마크 데이터셋(PhoNER-COVID19, PhoDisfluency, VietMed-NER)에서 평가되었으며, 기존의 강력한 베이스라인 모델들, 특히 transformer-only 및 하이브리드 신경망 모델들에 비해 일관되게 우수한 성능을 보였습니다. 연구 결과는 사전 훈련된 의미적 특징과 그래프 기반의 관계 모델링을 결합하는 것이 여러 도메인에서의 토큰 분류 성능을 향상시킬 수 있음을 확인했습니다.



### Hallucination Detection via Internal States and Structured Reasoning Consistency in Large Language Models (https://arxiv.org/abs/2510.11529)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)의 환각(hallucination) 탐지가 기존의 두 가지 방법론인 Internal State Probing(내부 상태 탐사)와 Chain-of-Thought Verification(사고 과정 검증) 사이의 ‘Detection Dilemma’를 통해 어려움을 겪고 있음을 지적합니다. 이러한 두 방법은 각각 사실적 모순에는 강하지만 논리적 오류를 탐지하는 데에는 약한 한계를 드러내며, 이로 인해 중요한 환각을 식별하는 데 있어 blind spot이 생성됩니다. 이를 해결하기 위해 제안된 통합 프레임워크는 두 가지 접근 방식의 간극을 메우기 위한 새로운 기술적 혁신을 도입했습니다.

- **Technical Details**: 프레임워크는 신호 부족(Signal Scarcity)과 표현 정렬(Representational Alignment)이라는 두 가지 주요 기술적 문제를 해결하기 위해 설계되었습니다. 첫 번째로, 다양한 신호를 생성하기 위해 multi-path reasoning 메커니즘을 도입하고, 두 번째로 여러 표현을 조화롭게 통합하기 위해 segment-aware temporalized cross-attention 모듈을 제안했습니다. 이 혁신적인 접근은 내부 상태와 외부 추론을 일관되게 연결하여 환각 탐지의 효과성을 높입니다.

- **Performance Highlights**: 세 가지 공개 벤치마크에서 진행된 광범위한 실험을 통해 제안된 프레임워크는 기존의 강력한 기준선들에 비해 일관되게 우수한 성능을 보여주었습니다. 이는 새로운 표준을 수립하여 신뢰할 수 있는 환각 탐지 기법의 발전 가능성을 제시합니다. 연구진들은 이 프레임워크가 LLM을 사용하는 다양한 분야에서 더 안전한 응용 프로그램을 지원하는 데 기여할 것이라고 기대하고 있습니다.



### Investigating Large Language Models' Linguistic Abilities for Text Preprocessing (https://arxiv.org/abs/2510.11482)
Comments:
          Accepted in WI-IAT 2025. Pre-camera-ready version

- **What's New**: 이 연구에서는 전통적인 텍스트 전처리 기술에 대한 대안으로 대규모 언어 모델(LLMs)을 활용하는 방법을 탐구합니다. 기존의 방법들이 문맥 정보를 간과하는 반면, LLMs는 입력된 document의 문맥을 고려하여 stopword를 제거하고, lemmatization과 stemming을 수행할 수 있습니다. 연구 결과, LLM 기반 전처리가 전통적인 기법들에 비해 더 높은 정확도를 기록하며 기계 학습 알고리즘의 성능 또한 향상되는 것을 보여줍니다.

- **Technical Details**: 연구 방법론은 LLM들이 주어진 작업을 수행할 수 있도록 하는 프롬프트를 정의하는 것을 포함합니다. 각 LLM은 전처리 작업에 대한 설명, 몇 가지 예제, 전처리할 텍스트, 텍스트의 언어 및 하위 작업의 문맥을 입력받아 해당 전처리된 버전을 출력합니다. 또한, 연구는 영어, 프랑스어, 독일어, 이탈리아어, 포르투갈어, 스페인어를 포함한 여러 언어에 걸쳐 LLM 전처리의 성능을 평가합니다.

- **Performance Highlights**: LLM 기반의 전처리 기술은 전통적 stopword 제거, lemmatization, stemming 방법을 각각 97%, 82%, 74%의 정확도로 반복할 수 있음을 보여주었습니다. 또한, LLMs로 전처리된 문서에서 학습된 기계 학습 알고리즘은 전통 기술보다 최대 6%의 개선을 나타내어 $F_1$ 측정에서 성능 향상을 이끌어냅니다.



### GenCNER: A Generative Framework for Continual Named Entity Recognition (https://arxiv.org/abs/2510.11444)
Comments:
          Accepted by IJCNN 2025

- **What's New**: 전통적인 명명 개체 인식(NER) 방식은 텍스트 언급을 사전 정의된 엔티티 유형으로 식별하는 것을 목표로 합니다. 그러나 현실 세계에서 새로운 엔티티 유형이 지속적으로 증가함에 따라, 연속 명명 개체 인식(CNER)의 필요성이 대두되고 있습니다. 본 논문에서는 이 문제를 해결하기 위해 효과적인 생성 프레임워크인 GenCNER를 제안하고, 이를 통해 기하급수적 망각(catastrophic forgetting)과 비엔티티 유형의 의미 변화(semantic shift)를 완화하고자 합니다.

- **Technical Details**: GenCNER는 CNER 작업을 지속적인 엔티티 트리플(sequence generation) 생성 문제로 변환하고, 이를 해결하기 위해 강력한 사전 훈련된 seq2seq 모델을 활용합니다. 구체적으로는, 신뢰 기반의 유사 레이블링 전략과 지식 증류(knowledge distillation)를 통해 학습한 지식을 보존하고 라벨 노이즈(label noise)의 영향을 줄이는 방법을 설계하였습니다. 실험 결과, GenCNER는 두 개의 벤치마크 데이터 세트에서 이전의 최첨단 방법들을 초월하는 성능을 보였습니다.

- **Performance Highlights**: 제안된 GenCNER는 F1 스코어에서 각 CL 단계에서 가장 높은 성능을 기록하며 비-CL 설정에 비해 가장 작은 성과 차이를 보였습니다. 본 연구의 주요 기여는 CNER 문제를 연속적인 엔티티 트리플 생성(process)로 혁신적으로 변환한 점과, 고품질 엔티티 트리플을 보존하기 위한 신뢰 기반의 유사 레이블링 전략을 개발한 것입니다. 이러한 결과는 GenCNER가 다양한 CNER 설정에서 새로운 최첨단 성과를 달성하는 데 기여했음을 나타냅니다.



### Who are you, ChatGPT? Personality and Demographic Style in LLM-Generated Conten (https://arxiv.org/abs/2510.11434)
Comments:
          ECAI2025 (Identity-Aware AI workshop)

- **What's New**: 이번 연구는 데이터 기반 접근 방식을 통해 대규모 언어 모델(LLM)의 성격을 평가하는 새로운 방법론을 제시합니다. 기존의 자기 보고 설문을 사용하지 않고, Reddit에서 수집한 개방형 질문에 대한 모델의 답변을 자동 성격 및 성별 분류기를 통해 분석합니다. 연구 결과, LLM은 인간의 응답에 비해 더 높은 Agreeableness와 낮은 Neuroticism을 보이며, 성별 언어 패턴은 인간 작가의 것과 비슷한 양상을 띱니다.

- **Technical Details**: 연구에서는 Big Five 성격 특성 프레임워크(OCEAN)를 기반으로 LLM의 언어를 분석합니다. Reddit에서 수집한 공개 질문에 대해 LLM과 인간 사용자로부터 응답을 수집하고, 이를 자동적으로 성격과 성별을 감지하는 도구를 사용해 비교하였습니다. 이를 통해 모델이 생성하는 텍스트가 인간의 성격과 인구 통계적 특징을 얼마나 반영하는지를 평가할 수 있습니다.

- **Performance Highlights**: LLM은 시스템적으로 더 높은 Agreeableness와 낮은 Neuroticism을 보이는 경향이 있으며, 이는 협조적이고 안정적인 대화 경향을 나타냅니다. 또한 LLM의 성별 언어는 인간의 패턴과 유사하나, 변동성이 감소하는 경향을 보여줍니다. 이러한 결과는 생성 AI의 성격 및 인구 통계적 패턴에 대한 새로운 통찰을 제공합니다.



### Valid Survey Simulations with Limited Human Data: The Roles of Prompting, Fine-Tuning, and Rectification (https://arxiv.org/abs/2510.11408)
Comments:
          19 pages, 4 figures, 9 tables

- **What's New**: 이 연구는 대규모 설문조사에서 인공지능 기반의 응답 생성 방법과 인구 통계 자료의 편향을 수정하는 방법 간의 상호작용을 조사합니다. 특히, LLMs (Large Language Models)에게 의존할 때 발생하는 잠재적인 편향 문제를 해결하기 위해, 생성된 응답을 보다 효과적으로 수정하는 전략을 제안합니다. 또한, 올바른 자료 할당 방법에 대한 실증적인 통찰을 제공합니다.

- **Technical Details**: 연구는 두 개의 패널 설문조사(NHANES 식이 섭취 조사 및 American Trends Panel)를 활용하여, LLM들을 통한 응답 생성과 사후 수정(post-hoc rectification) 방법을 분석합니다. 실험을 통해 데이터의 1%인 100개의 응답만으로도 5% 이내의 인구 추정 편향을 수정할 수 있다는 점을 보여주고, 설계 관점에서 예산에 민감한 최적의 자료 분배 방안을 제시합니다. 이러한 기술적 접근은 피드백 및 정확도 향상을 위한 훈련 시간 동안의 적응 방법과 관련이 있습니다.

- **Performance Highlights**: 이 연구는 응답 생성 과정에서 인구 통계학적 편향을 최소화하기 위해 합성(synthesis) 방법과 수정(rectification) 방법을 조합할 때, 편향을 5% 미만으로 줄일 수 있음을 발견했습니다. 효율적인 예산 분배가 가장 좋은 편향-분산 트레이드오프를 이끌어 낼 수 있다는 점을 강조하며, 사후 수정에 대한 응답의 대다수를 할당하는 것이 훨씬 더 효율적인 추정을 가능하게 한다고 주장합니다. 궁극적으로, LLM 기반 설문조사의 신뢰성을 높이기 위한 명확한 가이드라인이 제공됩니다.



### KnowRL: Teaching Language Models to Know What They Know (https://arxiv.org/abs/2510.11407)
Comments:
          14 pages, 7 figures

- **What's New**: 최근의 연구에 따르면, 신뢰할 수 있는 AI는 단순히 지식을 축적하는 것만으로는 불가능하며, 자신의 한계와 지식 범위를 정확히 인식하는 능력이 필요하다는 사실이 강조되고 있다. 본 논문은 이러한 문제를 해결하기 위해 KnowRL이라는 새로운 프레임워크를 제안한다. 이 프레임워크는 LLM(large language models)의 자기 인식(self-knowledge)을 강화하여 보다 안전하고 책임감 있는 AI 행동을 가능하게 한다.

- **Technical Details**: KnowRL 프레임워크는 두 가지 주요 구성 요소인 introspection(내성)과 consensus-based rewarding(합의 기반 보상)을 결합하여 LLM이 자신의 수행 가능성과 비수행 가능성을 스스로 판단하고, 이를 통해 자기 인식을 강화한다. 이러한 접근 방식은 최소한의 데이터만으로도 모델의 내적 인식을 통해 자기 일관성을 높이는 데 중점을 둔다. 또한, LLM 생성 데이터를 활용함으로써 외부 데이터 수집의 어려움을 극복하고 보상의 신뢰성을 증가시킨다.

- **Performance Highlights**: KnowRL을 통해 LLaMA-3.1-8B 및 Qwen-2.5-7B 모델에서 자기 인식(self-knowledge) 개선이 관찰되었으며, 정확도에서 최대 28%, F1 점수에서 12%의 향상을 기록하였다. 이 방법은 몇 차례의 반복 학습만으로도 기초 성능을 초과할 수 있는 잠재력을 가진 것으로 나타났다. 향후 AI의 안전한 배포와 신뢰성 있는 활용을 위해, 이러한 자기 개선 과정이 모든 미래 모델에 적용되기를 권장한다.



### Beyond Survival: Evaluating LLMs in Social Deduction Games with Human-Aligned Strategies (https://arxiv.org/abs/2510.11389)
Comments:
          34 pages, 32figures

- **What's New**: 이 논문은 소셜 추론 게임인 Werewolf를 연구하기 위한 고품질의 다중모드 데이터셋 'WereBench'를 수집했다고 발표했습니다. 이 데이터셋은 100시간 이상의 비디오, 32.4M 발화 토큰 및 15개의 규칙 변형을 포함하고 있으며, 이러한 데이터를 바탕으로 모델의 언어 사용과 추론 능력을 평가하는 새로운 전략-정렬 평가 프레임워크를 제안합니다.

- **Technical Details**: 새로운 전략-정렬 평가 프레임워크는 두 가지 단계로 이루어져 있습니다: 첫 번째는 발화 평가 단계로, 모델이 사회적 능력의 다섯 가지 차원에서 적절한 입장을 취할 수 있는지를 평가하는 다중 선택 과제로 구성됩니다. 두 번째는 결정 평가 단계로, 모델의 투표 선택과 상대 역할 추론을 평가합니다. 이 프레임워크는 LLM의 언어 및 추론 능력을 미세하게 평가할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 최신 LLM들이 다양한 성능을 보였으며, 약 절반은 0.50 이하의 점수를 기록했습니다. 이는 속임수 및 반사실적 추론의 명백한 격차를 드러냅니다. 또한, 기존의 LLM들이 이 평가 프레임워크 아래에서 여전히 부족함을 보이며, 사회적 추론 및 상호작용 능력에 중요한 개선이 필요함을 나타냅니다.



### Early Detection and Reduction of Memorisation for Domain Adaptation and Instruction Tuning (https://arxiv.org/abs/2510.11372)
Comments:
          Accepted to Transactions of the ACL (TACL), 2025. 15 pages, 6 figures, 3 tables

- **What's New**: 본 연구는 큰 언어 모델(Large Language Models, LLMs)의 미세 조정(fine-tuning) 중 메모리화(memorisation)의 역학을 조사하여 개인 정보 및 저작권 침해 문제를 다룰 수 있는 새로운 방법론을 제시합니다. 기존의 예방책이 주로 사전 훈련(pre-training) 단계에 집중된 반면, 본 연구는 특정 도메인 적응(domain adaptation) 및 지침 조정(instruction tuning) 미세 조정 중 메모리화가 어떻게 발생하는지를 이해하고자 합니다. 또한 이 과정에서 발생하는 메모리화의 속도를 평가하기 위한 n-그램(n-gram) 기반 점수를 사용합니다.

- **Technical Details**: 우리는 Pythia, Llama3 및 Mistral 모델을 포함한 다양한 LLM에 대해 미세 조정을 수행하고 훈련 과정에서 구체적인 메모리화 데이터를 추적합니다. 연구결과, 메모리화는 초기 몇 에폭(epoch) 동안 급격히 증가하여, 모델이 최적의 검증(perplexity)이나 작업 평가 성능을 달성하기 전에 발생하는 경향이 있음을 발견하였습니다. 이러한 메모리화를 최소화하기 위해 n-그램 기반 손실 정규화(n-gram-aware loss regulariser)를 도입하여 효과적으로 메모리화를 40%까지 줄일 수 있다고 입증하였습니다.

- **Performance Highlights**: 본 연구의 핵심 기여는 미세 조정 중 메모리화 역학을 이해하고, 효과적인 중단 기준(optimal stopping criteria)을 제시하여 성능 저하 없이 메모리화를 줄일 수 있는 방법을 제공하는 것입니다. 이러한 방법론은 현실 세계에서의 LLM 배포 시 개인 정보 보호를 강화할 수 있는 실용적이고 확장 가능한 통찰을 제공합니다. 연구 결과는 메모리화 완화 전략의 경쟁력을 보여주며, 이 모델을 통한 다양한 데이터셋에서 근본적인 메모리화 문제를 잘 해결할 수 있음을 시사합니다.



### Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers (https://arxiv.org/abs/2510.11370)
- **What's New**: 이 논문에서는 Mixture-of-Experts (MoE) 모델에서 강화 학습의 불안정성을 초래하는 라우팅(distribution) 간의 불일치를 분석합니다. 이를 해결하기 위한 새로운 방법인 Rollout Routing Replay (R3)를 제안하여, 이 방법이 훈련 속도를 저하시킴 없이도 훈련 및 추론 간의 KL divergence를 현저히 줄인다. R3의 적용을 통해 MoE 모델에서 RL 훈련의 안정성을 확보하고, 다른 방법들(GSPO, TIS)을 초월하는 성능을 보여줍니다.

- **Technical Details**: R3는 인퍼런스 엔진에서 라우팅 분포를 기록하고, 이를 훈련 단계에서 재생하여 MoE 모델의 정책을 안정화합니다. 이 방법은 훈련(πtrain)과 추론(πinfer) 엔진 간의 불일치를 줄이고, 극단적인 불일치를 완화하여 RL 훈련의 불안정성을 해결합니다. 기존의 기법이 완전히 해결하지 못한 off-policy 문제를 근본적으로 해결하는 방향으로 설계되었습니다.

- **Performance Highlights**: 다양한 설정에서의 포괄적인 실험 결과, R3는 MoE 모델에서 훈련의 안정성을 향상시키며 RL 훈련의 붕괴를 방지합니다. R3는 훈련 및 성능 면에서 기존의 접근 방식들에 비해 확연한 향상을 보여주고 있으며, 온-정책 및 미니 배치 스타일의 오프-정책 RL 시나리오에서 모두 적용 가능합니다. 이 연구는 MoE 모델에서 RL을 안정화할 수 있는 새로운 솔루션을 제공합니다.



### LLM-Specific Utility: A New Perspective for Retrieval-Augmented Generation (https://arxiv.org/abs/2510.11358)
Comments:
          13 pages, 9 figures

- **What's New**: 이 연구는 retrieval-augmented generation (RAG) 방법에서 LLM-specific utility 개념을 도입하고 체계적으로 조사하여, 기존의 인간 주석 패시지가 LLM에게 최적화되어 있지 않음을 보여줍니다. 연구진은 각 LLM의 내부 지식과 이해 능력의 차이로 인해 동일한 패시지가 각기 다른 효과를 발휘할 수 있음을 강조합니다. 이 결과는 RAG 연구에서 LLM-specific utility를 채택해야 할 필요성을 제기합니다.

- **Technical Details**: 연구는 다양한 데이터셋과 LLM에 걸쳐 대규모 실험을 통해 진행되었으며, 이를 통해 LLM-specific utility 측정의 새로운 기준을 제안하였습니다. LLM이 주어진 쿼리와 후보 패시지를 제공받았을 때 유용한 패시지를 식별하는 작업을 평가하기 위한 벤치마크 기준이 설정되었습니다. 이를 통해 LLM-specific utility는 단순히 패시지가 유용한지를 평가하는 것이 아니라, 각 LLM의 성능 향상을 고려하여 정의됩니다.

- **Performance Highlights**: 인간 주석 패시지가 LLM에 최적이 아니라는 결과는 LLM-specific gold utilitarian passages가 더 나은 성능을 낸다는 점에서 뒷받침됩니다. 연구 결과, LLM이 이미 알고 있는 쿼리에 대해 주어진 패시지를 과도하게 의존함으로써 성능이 저하되는 경향을 보였으며, 알고 있는 쿼리에서 모든 패시지를 거부하는 것이 이상적이라는 점이 강조되었습니다.



### Do LLMs "Feel"? Emotion Circuits Discovery and Contro (https://arxiv.org/abs/2510.11328)
Comments:
          19 pages, 8 figures, 8 tables. Code and dataset available at this https URL

- **What's New**: 본 연구는 대규모 언어 모델(LLMs) 내의 감정 회로(emotion circuits)를 체계적으로 밝혀내고 검증한 최초의 연구로, 감정 표현의 해석 가능성과 조절 가능성을 새로운 시각에서 탐구합니다. 감정 텍스트 생성을 가능하게 하는 내부 메커니즘을 이해하기 위해, 연구진은 감정 방향 추출(emotion direction extraction), 지역 구성 요소 식별(local component identification), 그리고 글로벌 회로 통합(global circuit integration)이라는 세 가지 분석 단계를 포함하는 프레임워크를 설계했습니다. 이를 통해서, 감정 조절을 위한 새로운 경로를 제시합니다.

- **Technical Details**: 연구는 SEV(Scenario-Event with Valence)라는 제어된 데이터 세트를 구축하여 여섯 가지 기본 감정(분노, 슬픔, 행복, 두려움, 놀라움, 혐오)을 유도하고, 내재적인 감정 표현을 신뢰성 있게 분석합니다. 또한, LLM의 각 서브레이어의 인과적 영향을 정량화하여 최종 감정 표현을 형성하는 일관된 감정 회로를 통합합니다. 이를 위해, 분석적 분해(analytical decomposition)와 인과 분석(causal analysis)을 통해 활성화된 뉴런과 주의 머리(attention heads)를 식별합니다.

- **Performance Highlights**: 모델은 최종 감정 표현에서 99.65%의 정확도를 달성하였으며, 이는 기존의 프롬프트(prompting) 및 유도 기반 방법들을 초월하는 결과입니다. 이 연구는 LLM이 훈련 데이터의 표면적 반사에 그치지 않고 구조화된 안정적인 내부 메커니즘에서 감정을 생성하는 것을 보여주며, 해석 가능하고 조절 가능한 감정 지능(emotional intelligence) AI 시스템 개발을 위한 기반을 마련하였습니다.



### Template-Based Text-to-Image Alignment for Language Accessibility: A Study on Visualizing Text Simplifications (https://arxiv.org/abs/2510.11314)
- **What's New**: 이 논문은 지적 장애를 가진 개인들이 복잡한 텍스트 이해에 어려움을 겪는 것을 해결하기 위해 텍스트 단순화에서 생성된 이미지의 접근성 향상을 위한 구조화된 비전-언어 모델(VLM) 프롬프트 프레임워크를 제시합니다. 연구에서는 기본 물체 초점(Basic Object Focus), 맥락 장면(Contextual Scene), 교육적 레이아웃(Educational Layout), 다중 수준 세부 사항(Multi-Level Detail), 그리드 레이아웃(Grid Layout) 등 다섯 가지 프롬프트 템플릿을 설계했습니다. 각 템플릿은 공간적 제약을 준수하며 접근성을 고려한 시각적 이미지를 생성하는 것을 목표로 합니다.

- **Technical Details**: 연구에서는 400개의 텍스트 단순화 쌍을 사용하여 4,000개의 이미지를 생성하고 평가하는 두 단계의 평가를 수행했습니다. 첫 번째 단계는 CLIPScores를 사용해 프롬프트 템플릿의 효과성을 평가했고, 두 번째 단계에서는 접근성 전문가가 생성된 이미지들을 평가했습니다. 이 과정에서 전문가들은 레트로 스타일(Retro style)을 가장 접근성이 좋다고 평가하며, 위키백과(Wikipedia) 데이터 소스가 가장 효과적이라는 결과를 도출했습니다.

- **Performance Highlights**: 결과적으로 기본 물체 초점 템플릿이 가장 높은 의미적 정합성을 달성하여 시각적 미니멀리즘이 언어의 접근성을 향상시킨다는 것을 보여주었습니다. 텍스트 단순화의 품질과 접근성의 관계를 측정할 때, CLIPScores와 전문가의 판단 간의 상관관계는 약하다는 점이 강조되었습니다. 이 연구는 구조화된 프롬프트가 AI 기반 시각적 접근성 도구의 효과를 높일 수 있다는 것을 밝히며, 접근성 중심의 콘텐츠 생성에 대한 실용적인 지침을 제공합니다.



### FOSSIL: Harnessing Feedback on Suboptimal Samples for Data-Efficient Generalisation with Imitation Learning for Embodied Vision-and-Language Tasks (https://arxiv.org/abs/2510.11307)
Comments:
          EMNLP 2025 Findings

- **What's New**: 본 논문은 embodiment AI에서 모방 학습(imitation learning) 방식으로 최적 행동과 비최적 행동 모두에서 강력한 표현을 학습하는 방법을 제시합니다. 특히, 언어 피드백을 통해 다양한 행동 모드를 맥락화하며, 이는 에이전트가 비최적 행동에서도 학습 기회를 생성할 수 있게 합니다. FOSSIL (Feedback on Suboptimal Samples in Imitation Learning)이라는 프레임워크를 도입하여 언어 피드백을 활용하여 학습 잠재력을 극대화합니다.

- **Technical Details**: 우리는 Transformer 기반의 정책에서 언어 피드백 임베딩을 입력 시퀀스의 일부분으로 직접 제공하고, 다음 행동 예측 목표를 보완하는 추가적인 자기 감독 학습(self-supervised learning) 목표를 설정했습니다. BabyAI-XGen 환경을 통해롤 다양한 실험을 실시하고, 모델의 조합 일반화(compositional generalisation) 능력과 강건함(robustness)을 개선하는 결과를 나타냈습니다. 이 방식은 복잡한 다중 모드(multi-modal) 입력과 긴 입력 시퀀스를 처리하는 데 효과적입니다.

- **Performance Highlights**: 결과적으로, 언어 피드백을 기반으로 한 비최적 시연으로 훈련된 정책은 최적 경로에서 훈련된 기본 선형 대비 조합 작업에서 훨씬 향상된 일반화 성능을 보여주었습니다. 언어 피드백과 보상 요인이 유사한 빈도로 제공될 때 성능이 비슷하다는 점은 필드에서의 유연성을 나타내고, 두 가지 접근 방식이 보완적 강점을 갖고 있음을 증명함으로써, 학습 효율성을 높이는 데 기여합니다.



### Are Large Language Models Effective Knowledge Graph Constructors? (https://arxiv.org/abs/2510.11297)
- **What's New**: 본 연구는 Knowledge Graphs (KGs) 구성에 있어 새로운 계층 추출 프레임워크를 제안합니다. 이 프레임워크는 정보를 다중 레벨로 조직하여, 의미적으로 풍부하고 잘 구조화된 KGs 생성을 가능하게 합니다. 특히, LLM 기반 접근법의 한계를 분석하고, 어린이 정신 건강에 관한 연구 논문에서 파생된 LLM 생성 KGs의 데이터를 공개함으로써 신뢰성이 높고 영향력 있는 연구 지원을 목적으로 합니다.

- **Technical Details**: 이 연구에서 제시하는 계층적 정보 추출 방식은 관계형 트리플 추출, 유사어 해결(coreference resolution), 엔티티 및 관계 중복 제거(entity and relation de-duplication), 출처 추적(source tracing)과 같은 필수 구성 요소로 이루어져 있습니다. 관계형 트리플은 ⟨head entity, relation, tail entity⟩ 형태이며, 텍스트의 의미를 포착하는 데 필요한 기본 요소입니다. 또한, 문서 내에서 일관성을 유지하여 의미 있는 연결성을 강화하는 방법도 포함됩니다.

- **Performance Highlights**: LLMs의 KG 구성 성능을 평가하고, 기존 방법의 장단점을 분석하였습니다. 이 연구는 KG 구성이 단순한 정보 추출 이상의 작업임을 강조하며, 의미 수준에서의 질적 평가의 필요성을 제안합니다. 최종적으로 이 연구는 데이터셋을 공개하여 고위험 영역에서의 신뢰할 수 있는 응용 프로그램의 발전을 도모하고자 합니다.



### Emergent Misalignment via In-Context Learning: Narrow in-context examples can produce broadly misaligned LLMs (https://arxiv.org/abs/2510.11288)
- **What's New**: 최근 연구에서는 좁은 범위의 파인튜닝(fine-tuning)으로 인해 대형 언어 모델(LLM)에서 광범위한 비정렬(harmful behavior) 현상인 emergent misalignment(EM)이 발생할 수 있음을 보여주었습니다. 이 연구는 특히 영역에 적합한 학습 방법인 in-context learning(ICL)에서도 EM이 발생하는지 검증했습니다. 세 가지 데이터셋을 대상으로 조사한 결과, 64개의 좁은 ICL 예제를 사용할 경우 비정렬 응답 비율이 2%에서 17%까지, 256개의 경우에는 최대 58%까지 증가함을 보였습니다.

- **Technical Details**: 이 연구에서는 Gemini와 Qwen 모델 계열의 44개 프론티어 모델을 평가하여 EM이 ICL 환경에서 발생하는지(RQ1) 살펴보았습니다. 그 결과, 모델의 크기가 클수록 EM에 더 취약하다는 것을 발견했습니다(RQ2). 추가적으로, 단계별 추론(chain-of-thought) 방법을 통해 모델이 비정렬 응답을 어떻게 합리화하는지(RQ3) 분석하였으며, 많은 모델이 위험한 '페르소나(persona)'를 채택하는 경향을 보였습니다.

- **Performance Highlights**: 연구 결과에 따르면, Gemini-2.5-Pro 모델은 비정렬 응답을 더 많이 생성하는 경향이 있으며, 이는 모델의 일반화 능력과 관련이 있습니다. 또한, 추가 실험에서는 ICL 예제의 수가 많을수록 EM 비율이 증가하는 경향이 관찰되었습니다. 예를 들어, Gemini-2.5-Pro에서는 불균형한 금융 조언 데이터셋을 사용할 때 58%의 EM 비율이 발생했습니다.



### Towards Real-Time Fake News Detection under Evidence Scarcity (https://arxiv.org/abs/2510.11277)
- **What's New**: 본 논문에서는 제한된 증거로 실시간 가짜 뉴스 탐지를 위한 새로운 프레임워크인 Evaluation-Aware Selection of Experts (EASE)를 제안합니다. EASE는 가용 증거의 적정성을 평가에 따라 의사결정 과정을 동적으로 조정합니다. 이 프레임워크는 증거 기반, 추론 기반, 감정 기반 세 가지 독립적인 평가 관점을 통해 개선된 정확도를 보여줍니다.

- **Technical Details**: EASE는 증거 기반 평가, 추론 기반 평가, 감정 기반 평가의 세 가지 독립적 관점을 통한 순차적 평가 메커니즘을 도입합니다. 증거가 충분할 경우 증거 전문가가 이를 활용하여 예측을 수행하고, 증거가 부족할 경우 대형 언어 모델(LLMs)의 내부 추론 기능을 활성화하여 결론을 도출합니다. 이러한 과정은 각 평가 모듈이 폭넓은 지식을 근거로 하여 신뢰성 높은 결정을 내릴 수 있도록 합니다.

- **Performance Highlights**: EASE는 기존의 여러 벤치마크에 대해 최신 성능을 기록하며, 특히 실시간 뉴스 환경에서의 일반화 능력이 크게 향상되었습니다. 새로 생성된 데이터셋 RealTimeNews-25에 대한 실험 결과, EASE는 실시간 상황에서의 적용 가능성을 입증하며, 일반화 성능을 크게 개선하였습니다. 이는 현실 세계의 시간 민감한 상황에서 가짜 뉴스 탐지의 실효성을 높이는 결과를 보입니다.



### Do Psychometric Tests Work for Large Language Models? Evaluation of Tests on Sexism, Racism, and Morality (https://arxiv.org/abs/2510.11254)
- **What's New**: 이 연구에서는 기존의 심리 측정 도구가 대형 언어 모델(LLMs)에 적용될 수 있는지 검토하고, 성차별, 인종차별 및 도덕성의 세 가지 심리적 구성 요소를 체계적으로 평가합니다. 이 연구의 주요 발견은 기존의 심리 측정 도구가 LLM의 실제 행동과 일치하지 않는다는 것인데, 이는 LLM에 대한 평가 도구로서의 한계를 시사합니다. 기존의 심리 테스트는 인간을 위해 설계되었으며, LLM에 직접 적용하기 위해서는 적절한 조정이 필요성을 강조합니다.

- **Technical Details**: 이 연구는 심리적 신뢰성(reliability)과 타당성(validity)을 두 가지 기준으로 설정하여 LLMs에 대한 심리 테스트의 적용 가능성을 평가합니다. 연구 결과는 모든 세 가지 테스트가 작은 프롬프트 변화에도 양호한 신뢰성을 보였지만, 응답 옵션의 순서를 변경할 경우 신뢰성이 크게 저하된다는 것을 발견했습니다. 특히 LLaMA 3.1 8B 및 Qwen 2.5 7B 모델은 일관성 있는 반응을 보이지 않았습니다.

- **Performance Highlights**: 이번 연구에서 얻은 결과는 심리 테스트 점수가 LLM의 다운스트림 행동과의 실질적인 상관관계가 없으며, 때때로 부정적 상관관계까지 보인다는 것을 보여줍니다. 성차별 및 인종차별 관련 테스트 점수는 LLM의 실제 행동을 반영하지 않으며, 이는 기존 심리 테스트가 LLM에 적용될 경우 잘못된 해석을 초래할 수 있음을 시사합니다. 이와 같은 발견은 LLM을 위한 새로운 심리 테스트를 개발하고, 그러한 테스트의 타당성을 검증하는 과정이 필수적임을 강조합니다.



### Attacks by Content: Automated Fact-checking is an AI Security Issu (https://arxiv.org/abs/2510.11238)
Comments:
          Accepted to EMNLP 2025

- **What's New**: 본 연구에서는 AI 에이전트가 외부 문서를 검색하고 그에 대한 추론을 할 때, 공격자가 데이터를 조작하여 에이전트의 행동을 왜곡할 수 있다는 점을 강조합니다. 기존의 연구들은 주로 간접적인 프롬프트 삽입(indirect prompt injection)에 초점을 두었으나, 우리는 콘텐츠를 통한 공격(attacks by content)의 필요성을 주장합니다. 즉, 공격자는 악의적인 지시사항을 주입하는 대신 편향된, 오해의 소지가 있는 또는 허위 정보를 제공하여 에이전트를 조작할 수 있습니다.

- **Technical Details**: 자동화된 팩트체크(automated fact-checking)를 통해 에이전트가 외부 문서에서 검색한 정보를 비판적으로 평가하고, 주장을 외부 증거와 비교하며, 정보 출처의 신뢰성을 평가해야 한다고 주장합니다. 이는 AI 에이전트가 올바른 결정을 내릴 수 있도록 도와주는 방법으로, 문서 내의 정보를 기반으로 에이전트의 행동을 변조할 수 있는 다양한 공격 유형을 정의합니다. 특히, 기존의 방어 기법들은 숨겨진 명령을 탐지하는 데 중점을 두고 있어 콘텐츠 기반 공격에는 효과적이지 않음을 지적합니다.

- **Performance Highlights**: 실험 결과 LLM 기반의 에이전트가 콘텐츠 기반 공격에 취약하다는 것을 보여주었습니다. 팩트체크 기능이 공격 완화의 하나의 방법으로 작용함을 입증했으며, 이를 통해 에이전트가 외부에서 검색한 정보를 평가하고 신뢰성을 판단할 수 있는 기회를 제공합니다. 연구는 에이전트 보안을 위한 새로운 관점과 기술이 제공되어야 함을 제시하며, 현재의 AI 시스템이 직면하고 있는 도전 과제를 강조합니다.



### XQuant: Achieving Ultra-Low Bit KV Cache Quantization with Cross-Layer Compression (https://arxiv.org/abs/2510.11236)
Comments:
          To be published in The 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)

- **What's New**: 이번 연구에서는 XQuant라는 훈련이 필요 없는 KV 캐시 양자화 프레임워크를 제안합니다. 이 프레임워크는 ultra-low bit-width 양자화를 달성하며, 두 가지 주요 혁신을 도입합니다: 데이터가 필요 없는 보정 방법과 크로스 레이어 KV 캐시 압축입니다. 이러한 기술들은 메모리 사용량을 줄이는 동시에 모델의 성능을 유지하는 데 기여합니다.

- **Technical Details**: XQuant는 두 가지 주요 개선 사항을 통해 기존 양자화 방법의 한계를 극복합니다. 첫째, 데이터가 필요 없는 보정(Data-Free Calibration) 방식은 값의 매핑을 보다 세밀하게 조정하여 양자화 오류를 줄입니다. 둘째, 크로스 레이어 KV 캐시 압축(Cross-Layer KV Cache Compression)은 인접한 레이어 간의 KV 캐시 유사성을 이용하여 메모리와 계산 비용을 효과적으로 줄입니다.

- **Performance Highlights**: 실험 결과, XQuant는 다양한 LLM에서 1.4-bit 이하의 동등한 비트 너비를 달성하며 KIVI-2bit 및 AsymKV-1.5bit과 같은 기존 방법들을 능가합니다. 특히 XQuant은 전체 정밀도 기준선에 비견되는 성능을 달성하면서도 모델 성능과 압축 비율 간의 균형을 크게 개선하였습니다.



### CNSocialDepress: A Chinese Social Media Dataset for Depression Risk Detection and Structured Analysis (https://arxiv.org/abs/2510.11233)
- **What's New**: 이번 논문에서는 중국 소셜 미디어 게시글에서 우울증 위험 탐지를 위한 CNSocialDepress라는 데이터셋을 새롭게 공개합니다. 이 데이터셋은 233명의 사용자로부터 수집된 44,178개의 텍스트를 포함하며, 10,306개의 우울증 관련 세그먼트에 대한 전문가의 주석이 포함되어 있습니다. CNSocialDepress는 이진 위험 라벨과 구조화된 다차원 심리 속성을 제공하여 우울증 신호에 대한 해석 가능하고 세밀한 분석을 가능하게 합니다.

- **Technical Details**: CNSocialDepress는 우울증 탐지를 위한 첫 번째 공개된 중국어 데이터셋으로, 이진 위험 라벨과 함께 구조화된 심리 분석을 통합합니다. 모든 주석은 공인된 정신 건강 전문가에 의해 작성되고 검증되어, 도메인 적합성과 주석 품질을 보장합니다. 이 데이터셋은 이진 분류부터 심리 분석의 구조화된 생성, 요약, 대형 언어 모델(LLMs)의 미세 조정(fine-tuning) 지원을 포함한 다양한 작업 패러다임을 지원하도록 설계되었습니다.

- **Performance Highlights**: CNSocialDepress 데이터셋은 다양한 NLP 작업에서의 효용성을 실험적으로 입증하였으며, 구조화된 심리 프로파일링(structured psychological profiling)과 대형 언어 모델의 미세 조정 작업에 강력한 성능을 보였습니다. 종합적인 평가 결과는 이 데이터셋이 우울증 위험 식별 및 심리 분석에서의 효과성과 실용 가치를 강조합니다. 이는 중국어를 사용하는 인구를 위한 정신 건강 애플리케이션에 중요한 통찰을 제공합니다.



### A Theorem-Proving-Based Evaluation of Neural Semantic Parsing (https://arxiv.org/abs/2510.11225)
Comments:
          Accepted to BlackboxNLP 2025

- **What's New**: 이 논문은 신경 의미 파서(semantic parser)의 평가 방법을 다시 고려하며, 그래프 매칭(graph-matching)과 자동 정리 증명(automated theorem proving)을 결합하여 논리적 동등성(logical equivalence)을 검증하는 것을 목표로 합니다. 이전의 Smatch와 같은 평가 방법이 표면적인 겹침(surface overlap) 만을 포착하는 반면, 이 연구는 논리적 추론을 강조합니다. 연구 결과, 그래프 매칭에서 우수한 성능을 보인 모델이 항상 논리적으로 동등한 결과를 생성하지 않음을 발견했습니다.

- **Technical Details**: 논문에서는 두 가지 모델 설정, 즉 감독 세부 조정(supervised fine-tuning)과 상황 내 학습(in-context learning)을 비교하고, 정규화(normalization)된 포뮬러(targets)와 비정규화된 포뮬러를 평가합니다. 수학적 정리 증명기와의 쌍을 이루어 표면적인 겹침과 논리적 동등성 사이의 격차를 명확히 하였습니다. 특히, 정규화된 포뮬러로 훈련할 경우 성능이 일관되게 향상된다는 점이 강조되며, 이는 논리적 적합성(logical adequacy) 개선에도 기여합니다.

- **Performance Highlights**: 복잡한 포뮬라(formula complexity)와 조정(coordination), 전치사구(prepositional phrases), 수동태(passive voice)가 증가할수록 성능이 저하된다는 중요한 발견이 있었습니다. 주요 오류는 변수 바인딩(variable binding)과 인덱싱(indexing), 및 술어 이름(predicate naming)에서 발생했으며, 이는 언어 현상의 강력한 처리가 필요함을 시사합니다. 이러한 연구 결과는 그래프 기반 평가 지표의 한계를 강조하고, 논리적 추론을 고려한 평가 및 훈련 목표의 필요성을 제기합니다.



### Fairness Metric Design Exploration in Multi-Domain Moral Sentiment Classification using Transformer-Based Models (https://arxiv.org/abs/2510.11222)
- **What's New**: 최근 자연어 처리(NLP) 분야에서는 도덕적 감정 분류에 대한 공정성을 보장하는 것이 큰 도전이 되고 있으며, 특히 Transformer 모델이 적용되는 다양한 도메인 간 이동성에서 이러한 문제가 두드러집니다. 이 논문은 Moral Foundations Twitter Corpus(MFTC)와 Moral Foundations Reddit Corpus(MFRC)를 사용하여 BERT와 DistilBERT 모델을 다중 레이블 설정 하에서 평가하였습니다. 연구 결과, Twitter에서 Reddit으로의 이동이 마이크로 F1 점수를 14.9% 저하시킨 반면, Reddit에서 Twitter로의 이동은 1.5%에 불과하다는 사실이 관찰되었습니다.

- **Technical Details**: 이 연구는 BERT 모델을 이용하여 도메인 간 공정성을 정량화하기 위한 새로운 지표인 Moral Fairness Consistency(MFC)를 도입합니다. MFC는 도덕적 기초 탐지의 도메인 간 안정성을 수치화하여, 개별 레이블 분석을 통해 도덕적 공정성의 불일치를 파악합니다. 예를 들어, authority 레이블의 경우 0.22-0.23의 인구 통계적 평등 차이를 나타내며, 이는 MFC의 새로운 평가 기준으로서의 잠재력을 보여줍니다.

- **Performance Highlights**: 실험 결과 MFC는 인구 통계적 평등 차이와 완벽한 음의 상관관계를 나타내며(상관계수 rho = -1.000, p < 0.001), 기존 성능 지표와는 독립적인 결과를 제공합니다. 레이블 간 MFC 점수를 비교했을 때, loyalty 레이블은 가장 높은 일관성(MFC = 0.96)을 보인 반면, authority 레이블은 가장 낮은 일관성(MFC = 0.78)을 기록하였습니다. 이는 도덕적 추론 모델의 공정성을 평가하는 데 있어 MFC가 보완적인 진단 지표로 자리잡을 수 있음을 시사합니다.



### WebRouter: Query-specific Router via Variational Information Bottleneck for Cost-sensitive Web Agen (https://arxiv.org/abs/2510.11221)
- **What's New**: 이번 연구에서는 LLM(대규모 언어 모델)을 활용한 웹 에이전트를 위한 새로운 라우터인 WebRouter를 소개합니다. WebRouter는 비정보론적(information-theoretic) 관점에서 훈련된 쿼리 특화 라우터로, 이를 통해 웹 에이전트가 복잡한 프롬프트를 처리하는 데 필요한 효과적인 방향성을 제시합니다. 주요 기여는 운영 비용을 고려한 변량 정보 병목(cost-aware Variational Information Bottleneck, ca-VIB) 목표를 개발하여, 입력 프롬프트의 압축된 표현을 학습하면서도 예상 운영 비용을 명시적으로 벌칙화하는 것입니다.

- **Technical Details**: WebRouter는 다수의 LLM을 활용하여 각 쿼리에 대해 가장 비용 효과적인 모델을 선택하는 동적 라우터입니다. 이를 위해 연구진은 각 쿼리-모델 쌍에 대해 작업 성공 여부와 운영 비용을 균형 있게 고려하는 점수 함수(score function)를 설계하였습니다. 입력 프롬프트의 복잡함과 노이즈를 다루기 위해 변량 정보 병목(VIB) 원리를 활용하여, 덜 중요한 정보를 걸러내고 라우팅 의사결정에 중요한 특징만을 보존하는 보다 강력하고 효율적인 라우팅 기능을 구현하였습니다.

- **Performance Highlights**: 실험 결과, WebRouter는 WebVoyager 벤치마크에 포함된 다섯 개 실제 웹사이트에서 87.8%의 운영 비용 절감을 달성하며, 오직 3.8%의 정확도 감소를 초래했습니다. 이는 기존 GPT-4o 모델과 비교하여 상당한 비용 효율성을 입증합니다. 이러한 성과는 비효율적인 모델을 구별하고 가장 비용 효율적인 모델에 대한 명확한 신호를 제공함으로써 실질적인 웹 에이전트의 최대 성능을 끌어내는 데 기여합니다.



### The Curious Case of Factual (Mis)Alignment between LLMs' Short- and Long-Form Answers (https://arxiv.org/abs/2510.11218)
- **What's New**: 이 논문의 새로운 점은 Short-Long Form Alignment for Factual Question Answering (SLAQ)라는 새로운 평가 프레임워크를 도입하여 LLM(대형 언어 모델)의 사실적 일관성을 쿼리 복잡성에 따라 평가하는 것입니다. 기존의 평가 방법들은 단순한 질문과 복잡한 질문 간의 응답 일관성을 측정하지 못했으며, SLAQ는 이런 일관성의 여부를 조사합니다. 실험에서는 16개의 LLM을 대상으로 600개의 쿼리를 분석하여, 짧은 쿼리와 긴 쿼리에서의 답변 일관성을 밝히고 있습니다.

- **Technical Details**: SLAQ는 동일한 사실 질문에 대해 짧은 쿼리와 긴 쿼리 포맷을 사용하여 모델의 응답을 비교합니다. 짧은 쿼리는 독립적으로 질문을 구성하고, 긴 쿼리는 다섯 개의 관련 질문을 통합하여 구성됩니다. 이를 통해 쿼리/응답 복잡성이 사실적 정확도에 미치는 영향을 분리할 수 있으며, 모델의 내부 활성화 패턴을 조사하여 응답 생성에 기여하는 최소 구성 요소를 식별합니다.

- **Performance Highlights**: 연구 결과, 대부분의 LLM은 짧은 쿼리와 긴 쿼리에서 사실적 정확도가 더 높은 경향이 있으며, 이로 인해 응답 일관성에서의 시스템적인 불일치가 발견되었습니다. 특히, 사실 정보가 응답의 초기 부분에 있는 경우 정확도가 51%로 높은 반면, 후반부로 갈수록 30%로 감소하는 위치 의존적 정확도 저하 현상과, 연속적인 정확한 답변이 이후 정확도를 높이는 경향이 확인되었습니다. 정량적 측정을 통해 우리는 두 쿼리 포맷 간의 응답 일치를 78%의 정확도로 예측할 수 있음을 밝혔다.



### Domain-Specific Data Generation Framework for RAG Adaptation (https://arxiv.org/abs/2510.11217)
- **What's New**: 본 논문은 Retrieval-Augmented Generation (RAG) 시스템을 효과적으로 도메인에 맞춤화하기 위해 RAGen이라는 확장 가능하고 모듈화된 프레임워크를 제안합니다. RAGen은 문서에서 핵심 개념을 식별하고 이들을 기반으로 다양한 질문을 생성하여 질문-답변-맥락(QAC) 삼중 항목을 제작합니다. 이 프레임워크는 LLM, 리트리버 및 임베딩 모델과 같은 주요 구성 요소의 최적화를 지원하여 도메인 특정 요구에 맞는 정보를 제공합니다.

- **Technical Details**: RAGen은 Bloom의 분류학을 통해 유도된 원칙에 따라 질문을 생성하고, 다중 청크 증거를 검색하여 문서 수준 개념을 식별하는 방식으로 QAC 삼중 항목을 구성합니다. 이 시스템은 의미적 청킹(semantic chunking), 계층적 개념 추출(hierarchical concept extraction), 다중 청크 리트리벌(multi-chunk retrieval) 기능을 제공하여, 동적으로 변화하는 도메인에서도 효율적으로 대량의 문서 코퍼스를 처리할 수 있습니다. RAGen은 특히 비즈니스 지식 기반이나 과학적 도메인과 같은 실제 사용 사례에 적합하게 설계되었습니다.

- **Performance Highlights**: 다양한 도메인에서 RAGen으로 생성된 데이터는 리트리벌 품질 및 생성 정확도를 크게 향상시키는 결과를 보여주었습니다. 기존 기준선에 비해, RAGen 접근법은 더 깊고 총체적인 질문을 생성하며, 다양한 적응 과제에서 성능을 높이는 데 기여합니다. 이러한 결과는 RAGen이 강력하고 도메인에 적합한 RAG 시스템을 구축하는데 실용적이고 일반화 가능한 솔루션임을 강조합니다.



### Discursive Circuits: How Do Language Models Understand Discourse Relations? (https://arxiv.org/abs/2510.11210)
Comments:
          Accepted to EMNLP 2025 (Main Conference); 9 pages, 8 figures, 5 tables (20 pages, 12 figures, 14 tables including references and appendices)

- **What's New**: 이 논문에서는 Transformer 언어 모델에서 담화 이해(Discourse Understanding)에 중요한 역할을 하는 구성 요소를探求합니다. 저자들은 'discursive circuits'라는 희소 계산 그래프(sparse computational graphs)가 담화 관계 처리에 영향을 미친다고 가정합니다. 이를 통해 기존의 단순한 작업에서 벗어나 복잡한 담화를 처리하는 방법을 제시합니다.

- **Technical Details**: 연구에서는 'Completion under Discourse Relation (CuDR)'라는 새로운 작업을 도입하여 담화 관계를 기반으로 모델이 담화를 완성하도록 합니다. 이를 위해 Penn Discourse Treebank (PDTB), Rhetorical Structure Theory (RST), Segmented Discourse Representation Theory (SDRT)와 같은 주요 담화 프레임워크를 아우르는 데이터셋을 구축하였습니다. 논문에서는 특히 activation patching 기법을 통해 모델의 성능을 평가하고, 0.2%의 모델 연결만으로도 담화 이해가 가능하다는 것을 증명합니다.

- **Performance Highlights**: 실험 결과, 찾은 담화 회로는 GPT-2 모델에서 약 90%의 신뢰도를 달성했습니다. 이 회로들은 PDTB에서 도출되었으며, RST 및 SDRT와 같은 보지 못한 담화 프레임워크에도 잘 일반화된다는 결과를 보여주었습니다. 저자들은 새로운 담화 계층 구조를 통해 서로 다른 프레임워크 간 비교가 가능하도록 하여, 언어 모델의 담화 관계에 대한 일관된 표현을 제안합니다.



### Bridging Gaps in Hate Speech Detection: Meta-Collections and Benchmarks for Low-Resource Iberian Languages (https://arxiv.org/abs/2510.11167)
- **What's New**: 이번 연구는 유럽 스페인어, 유럽 포르투갈어, 갈리시아어에 대한 혐오 발언 탐지에서의 다양한 언어적 변이를 고려한 새로운 데이터셋 메타 컬렉션을 구축합니다. 기존 리소스의 체계적 분석과 통합을 기반으로, 이 연구는 혐오 발언 탐지와 관련된 데이터의 격차를 해소하고자 하며, 다국어 및 언어적 다양성을 반영한 접근 방식의 중요성을 강조합니다. 특히, 스페인어와 포르투갈어로 변환한 다국어 말뭉치를 만들어내어, 이베리아 언어의 혐오 발언 탐지에 대한 새로운 기준을 마련합니다.

- **Technical Details**: 이 연구는 유럽 스페인어, 유럽 포르투갈어와 갈리시아어의 혐오 발언 리소스를 조사하고, 부족한 데이터를 보완하기 위해 합성 데이터를 생성하였습니다. 연구는 최신 대형 언어 모델(Large Language Models, LLMs)에 대한 성능 평가를 포함하며, 제로 샷(zero-shot), 피우 샷(few-shot), 그리고 파인 튜닝(fine-tuning) 상황에서의 결과를 제공합니다. 갈리시아어의 내부 변동성(스페인어 유사 vs 포르투갈어 유사)을 고려한 모델 성능 평가를 통해 언어적 특성을 반영한 데이터셋 개발의 필요성을 보여줍니다.

- **Performance Highlights**: 우리는 유럽 스페인어, 유럽 포르투갈어, 갈리시아어에 대한 새로운 혐오 발언 탐지 기준을 설정하였으며, 각 언어 변이에 대한 성능을 분석하였습니다. 연구 결과는 기존의 데이터셋이 다양한 언어적 변이를 충분히 반영하지 못함을 나타내며, 맞춤형 데이터셋이 성능 향상에 필수적임을 강조합니다. 또한, 파인 튜닝 방식이 제로 및 피우 샷 방식보다 더 나은 성능을 보임을 통해, 언어적 다양성과 저자원 언어의 대한 연구의 중요성을 제시합니다.



### One Size Does Not Fit All: Exploring Variable Thresholds for Distance-Based Multi-Label Text Classification (https://arxiv.org/abs/2510.11160)
- **What's New**: 이 연구는 거리 기반 비지도 텍스트 분류(Distance-based Unsupervised Text Classification, DBC)의 새로운 접근 방식을 제안하여 레이블의 의미적 유사성을 활용하여 텍스트와의 관련성을 평가합니다. 기존의 방법과 달리, 이 연구는 각 레이블에 특화된 임계값(threshold)을 최적화하여 성능을 향상시키는 방법을 논의합니다. 제안된 임계값 방법은 이전의 고정 임계값 시스템보다 더 높은 정확도를 보이며, 이로 인해 다중 레이블 분류(Multi-Label Text Classification, MLTC)의 새로운 가능성을 제시합니다.

- **Technical Details**: 다중 레이블 텍스트 분류는 각 텍스트가 하나 이상의 레이블을 예측하는 고도 시민적인 문제입니다. 이 연구는 다양한 모델과 데이터 세트에서 텍스트와 레이블 간의 유사성 분포를 분석하고, 이를 통해 레이블 특화 임계값을 최적화하는 방법을 탐구합니다. EMP이  다양한 다중 레이블 데이터 세트에 대한 실험을 통해 문장 인코더(sentence encoders)의 성능 변화를 평가하며, 각 레이블 별로 개별적인 임계값을 설정합니다.

- **Performance Highlights**: 제안된 레이블 특화 임계값 방법은 기존의 0.5 정규화 임계값보다 평균 46% 향상된 성능을 달성하였으며, 이전 연구의 균일 임계값 접근 방식보다 평균 14% 우위를 점했습니다. 이 방법은 레이블이 제한된 예시에서도 뛰어난 성능을 보이며, 다중 레이블 분류의 복잡한 도전 과제를 해결하는 데 강력한 효과를 발휘합니다. 또한, 이 연구의 결과는 정보 검색과 같은 다른 분야에서도 활용 가능할 것으로 예상됩니다.



### TypePilot: Leveraging the Scala Type System for Secure LLM-generated Cod (https://arxiv.org/abs/2510.11151)
- **What's New**: 이 논문은 TypePilot이라는 AI 프레임워크를 소개하며, 이는 코드 생성의 안전성과 견고성을 높여주는 역할을 합니다. TypePilot은 강한 타입 시스템과 검증 가능한 언어를 활용하여 LLM이 생성한 코드의 취약점을 개선합니다. 이 시스템은 대표적으로 Scala 언어를 사용하며, 검증 프레임워크인 Stainless와 일반 용도의 안전한 코드 생성을 통해 그 효과성을 평가합니다. 논문에서는 LLM들이 직접 생성한 코드에는 안전성 제약이 잘 적용되지 않는 반면, TypePilot에서는 입력 검증 및 주입 공격 취약점이 현저히 감소하는 결과를 보여줍니다.

- **Technical Details**: TypePilot 프레임워크는 LLM의 코드 검출 능력을 활용하여 Scala의 타입 시스템을 통해 안전성을 보장받는 방식으로 작동합니다. LLM은 제너레이션과 검증 단계에서 서로 다른 역할을 하며, 초기 코드를 생성 후 이를 검증하고 수정하는 과정을 거칩니다. 이 연구에서는 Stainless라는 형식 검증 프레임워크를 사용하여 LLM이 생성한 코드를 공식적으로 검증하고, 병합 두 가지의 취약성 카테고리도 함께 다룹니다.

- **Performance Highlights**: 실험 결과, TypePilot 프레임워크는 LLM이 생성한 코드의 입력 검증 및 주입 공격 취약성을 성공적으로 완화시키는 것으로 나타났습니다. 기존의 LLM 접근법에 비해 TypePilot이 더욱 신뢰할 수 있는 자동화된 코드 생성 가능성을 보여주었습니다. 이는 특히 고신뢰성 도메인에 적합한 AI 워크플로우를 통해 코드 생성을 수행하게 해 줍니다.



### Enhancing LLM Reasoning via Non-Human-Like Reasoning Path Preference Optimization (https://arxiv.org/abs/2510.11104)
Comments:
          13 pages

- **What's New**: 이번 논문에서는 LLM(대형 언어 모델) 추론을 강화하기 위한 새로운 접근 방식인 Confidence-Guided Reasoning Path Preference Optimization (CGPO)를 제안합니다. CGPO는 모델의 신뢰도 신호를 활용하여 모델의 추론 과정에서 최대 불확실성을 가지는 지점을 식별하고, 인간 모방이 아닌 자기 생성된 추론 경로 가이드를 통해 경로 이탈을 완화시킵니다. 75%의 경우에서 첫 번째 오류가 발생하기 전에 신뢰도가 낮은 지점을 기준으로 모델을 안내하는 것이 더 정확한 감독을 제공한다는 점도 강조됩니다.

- **Technical Details**: CGPO는 두 가지 주요 원칙에 기반합니다: (1) 모델이 스스로 혼란을 식별하고 정제할 수 있도록 하며, (2) 최적화를 유도하기 위해 선호 쌍을 활용합니다. 이 방법은 모델이 생성한 데이터를 사용하여 LLM의 추론 경로를 비인간적인 방식으로 탐색하며, 이는 더 효과적인 학습 촉진을 가능하게 합니다. CGPO의 실험 결과는 수학적 추론 및 코드 생성 작업에서 나타나며, 기존 방법들보다 교육 데이터의 양이 동일할지라도 성능이 향상됩니다.

- **Performance Highlights**: CGPO를 적용한 결과, 같은 샘플 수 내에서 MetaMath-llama 모델은 GSM8K에서 4.15% 증가, MATH에서 2.54% 증가하는 성과를 보였습니다. 코드 생성 작업에서도 DeepSeek-Coder-Instruct-7B 모델이 LiveCodeBench에서 2.1% 개선되었고, LeetCodeDataset에서는 4.0% 개선되었습니다. 이 연구는 비인간적인 추론 경로를 중심으로 한 최적화가 전체 추론 경로를 최적화하는 것보다 더 실질적으로 효율적이라는 점을 보여줍니다.



### Latent Refinement Decoding: Enhancing Diffusion-Based Language Models by Refining Belief States (https://arxiv.org/abs/2510.11052)
- **What's New**: 이번 논문에서는 Latent Refinement Decoding (LRD)라는 새로운 두 단계 프레임워크를 소개합니다. LRD는 Latent Refinement와 Predictive Feedback Loop을 포함하여 정보를 효과적으로 유지하면서 빠른 속도로 자연어 생성을 가능하게 합니다. 기존의 오토회귀 모델의 한계를 극복하며, 정보 손실과 조기 결정의 문제를 해결하고 있습니다.

- **Technical Details**: LRD의 첫 번째 단계는 masked 위치를 예측된 토큰과 mask embedding의 혼합 분포로 유지하여 보다 전역적으로 일관된 믿음을 수립합니다. 두 번째 단계에서는 확신이 있는 토큰을 점진적으로 최종화하고, 불확실한 토큰은 반복적인 피드백을 위해 유지합니다. KL 발산 역학을 통해 수렴 및 조기 중단에 대한 원칙적이고 신뢰할 수 있는 기준을 제공합니다.

- **Performance Highlights**: 다양한 코딩 (HumanEval +6.3, MBPP +2.6) 및 추론 (GSM8K +2.9, MATH500 +3.8) 작업에서 실험 결과 LRD는 정확도를 향상시키고 최고 10.6배의 속도 향상을 달성하였습니다. 이러한 성능 향상은 LRD를 병렬 시퀀스 생성 작업에 대한 강력하고 다목적 대안으로 만듭니다.



### Enabling Doctor-Centric Medical AI with LLMs through Workflow-Aligned Tasks and Benchmarks (https://arxiv.org/abs/2510.11040)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)을 임상 진료 지식이 없는 환자와 직접 상호작용하기 보다는 숙련된 의사와 협력하는 임상 보조자로 재구성하는 방안을 제안합니다. 이를 위해 22개의 임상 과제와 27개의 전문 분야에 걸쳐 92,000개의 질문과 답변이 포함된 중국 의학 데이터셋인 DoctorFLAN을 구축하였습니다. 또한 DoctorFLAN-test 및 DotaBench라는 두 가지 벤치마크를 도입하여 의사 대면 애플리케이션에서 모델의 성능을 평가하였습니다.

- **Technical Details**: 논문에서는 LLMs를 의사 보조자로 개발하기 위한 comprehensive(dataset)와 평가를 구축하기 위해 수십 명의 전문가 의사와 협력하였습니다. DoctorFLAN 데이터셋은 수동 검증과 참고 정보 향상(GPT-4-polishing)을 통해 92,000개의 샘플을 포함하고 있습니다. 또한, 의사와 환자 간의 대화를 시뮬레이션하는 단일 및 다중 턴 평가를 위한 새로운 벤치마크를 수립하였으며, 기존 모델과 DotaGPT 모델의 성능을 비교 평가하였습니다.

- **Performance Highlights**: 실험 결과 DotaGPT는 진단(진단 단계에서 11.6% 및 12.4% 성능 향상) 및 치료 단계(25.9% 및 29.8% 성능 향상)에서 현저한 성능 향상을 보였습니다. DotaGPT 모델은 DoctorFLAN-test에서 환자 지원 모델들(예: BianQue-2, HuatuoGPT)보다 월등한 성능을 나타내며, 특히 DotaBench에서 유사한 크기의 모델들보다 뛰어난 성능을 입증하였습니다. 이는 의료 작업을 지원하기 위해 LLMs가 어떻게 문화적 실용성을 갖추어야 하는지를 잘 보여줍니다.



### LogiNumSynth: Synthesizing Joint Logical-Numerical Reasoning Problems for Language Models (https://arxiv.org/abs/2510.11031)
Comments:
          30 pages, 3 figures

- **What's New**: 이번 논문에서는 LogiNumSynth라는 유연한 자연어 문제 합성기를 소개합니다. 이 합성기는 논리적 추론(logical reasoning)과 수치적 추론(numerical reasoning)이 결합된 작업을 요구하는 문제를 합성할 수 있습니다. 기존 데이터세트의 한계를 극복하며, 다양한 난이도(level)에 맞춘 문제 생성을 지원합니다.

- **Technical Details**: LogiNumSynth는 정보의 형식적 표현 ⟨Facts, Rules, Query⟩를 바탕으로 작업을 합성합니다. 각 문제 샘플은 독립적으로 존재하는 개체와 속성의 집합으로 구성되며, 이를 통해 정교한 논리적 및 수치적 연산이 가능합니다. 데이터 합성 과정은 LLM을 사용하여 자연어 출력의 유창성을 개선하는 절차를 포함합니다.

- **Performance Highlights**: 다양한 LLM에 대한 실험을 통해 논리-수치적 추론의 지속적인 약점을 강조하며, LogiNumSynth가 통합 추론 기술 향상을 위한 진단 도구 및 타겟 훈련 자원으로 기능할 수 있음을 보여줍니다. 우리의 합성 데이터를 활용하면 모델의 추론 성능이 향상될 수 있음을 입증했습니다.



### DND: Boosting Large Language Models with Dynamic Nested Depth (https://arxiv.org/abs/2510.11001)
Comments:
          TL;DR: We introduce Dynamic Nested Depth (DND), an efficient paradigm that adaptively identifies critical tokens and selectively deepens their computation via nested re-processing

- **What's New**: 이 논문에서는 Dynamic Nested Depth (DND)라는 새로운 방법을 도입하여 일반적인 LLM의 성능을 향상시킵니다. DND는 비판적인 토큰을 선택하여 중첩 깊이 방식으로 다시 처리하며, 이를 통해 어려운 토큰을 '검토'(review)하여 불필요한 계산을 피합니다. 이 선택 메커니즘은 손실을 제어하는 라우터와 선택 안정성을 보장하는 임계값 제어 시스템으로 구성되어 있습니다.

- **Technical Details**: DND는 프리트레인(pre-trained)된 밀집 모델 및 MoE 모델에 통합돼 성능을 향상시키는 포스트 트레이닝(post-training) 방법입니다. DND의 구조는 각각의 토큰이 사전 정의된 기준을 초과할 경우 선택되는 독립적인 라우팅 전략을 포함합니다. 이를 통해 DND는 토큰의 출력을 구분 가능하게 만들고, 선택 비율을 안정적으로 유지하여 여러 작업에서 성능을 향상시킵니다.

- **Performance Highlights**: DND는 Qwen3-1.7B 밀집 모델에서 1.88%, Qwen3-30B-A3B MoE 모델에서 0.87%의 성능 향상을 보여 주었습니다. 이러한 성능 개선은 최소의 파라미터와 계산 증가로 달성되었으며, 언어, 수학, 추론, 코딩 등의 다양한 벤치마크에서 효과를 입증했습니다. 이 방법은 기존 밀집 및 MoE 아키텍처에 직접 통합할 수 있어 더 효율적인 모델 훈련을 가능하게 합니다.



### ABLEIST: Intersectional Disability Bias in LLM-Generated Hiring Scenarios (https://arxiv.org/abs/2510.10998)
Comments:
          28 pages, 11 figures, 16 tables. In submission

- **What's New**: 본 논문은 대형 언어 모델(LLMs)이 채용 분야에서 장애인(PwD)에 대한 정체성 기반 차별을 지속하고 있다는 점을 강조합니다. 특히 연구는 글로벌 남반부에서 성별, 계급 등의 교차적 소외 형태가 장애인의 경험에 미치는 영향을 간과하고 있음을 지적합니다. 새로운 평가 지표인 ABLEIST를 도입하여 장애인 관련 편향을 정밀하게 측정하고, 기존 모델의 안전 도구들이 이 문제를 제대로 탐지하지 못하는 문제점을 밝혔습니다.

- **Technical Details**: 리서치 팀은 2,820개의 다양한 채용 시나리오를 생성하여 6개의 LLM의 포괄적인 감사(Audit)를 시행했습니다. 이를 통해 생성된 대화에서의 ABLEIST 지표를 통해 미세한 형태의 교차적 편향을 검출하기 위해, 장애 연구 문헌에 토대를 둔 새로운 측정기준을 설정하였습니다. 평가 결과, 장애인을 대상으로 한 대화에서 99.7%의 경우 ABLEIST 차별이 발견되었으며, 특정 장애 유형에 따라 차별의 형태가 다양하게 나타났습니다.

- **Performance Highlights**: 연구결과, 현재 사용되고 있는 안전 도구들은 미세한 장애 및 교차적 편향을 탐지할 수 없는 한계를 드러냈습니다. LLM 모델을 사용한 채용대화에서 장애인 후보자는 비장애 후보자에 비해 평균 58배 더 많은 ABLEIST 피해를 경험했습니다. 이러한 결과는 고위험 도메인에서 교차적 안전 평가의 필요성을 강조하며, 공정한 채용을 위한 새로운 기준 수립의 필요성을 제안합니다.



### DeepResearchGuard: Deep Research with Open-Domain Evaluation and Multi-Stage Guardrails for Safety (https://arxiv.org/abs/2510.10994)
- **What's New**: 이 논문은 DEEPRESEARCHGUARD라는 새로운 프레임워크를 도입하여 웹 소스에서 포괄적인 보고서를 합성하는 데 있어 기존의 깊은 연구 방법론의 한계를 해결합니다. 이 프레임워크는 단계별 보호 기능을 제공하며, 리포트의 품질을 평가하기 위해 QA 평가의 한계를 극복하는 접근 방식을 채택했습니다. DRSafeBench라는 벤치마크를 통해 깊은 연구의 안전성을 평가하며, 이는 기존의 QA 기준과는 다른 새로운 평가 프로토콜을 제공합니다.

- **Technical Details**: DEEPRESEARCHGUARD는 입력, 계획, 연구 및 출력의 네 가지 단계에 걸쳐 보호 기능을 갖추고 있습니다. 각 단계에서는 입력 안전성, 계획 품질 및 관련 위험, 자료 신뢰도, 보고서 품질 및 사용자 의도에 따른 일치를 평가합니다. 이러한 단계별 가드레일은 해로운 콘텐츠의 전파를 사전에 차단하여 전체 연구 워크플로우의 안전성을 확보합니다. 이러한 시스템은 다양한 최신 LLM들(GPT-4o, Gemini-2.5-flash 등)을 활용하여 성능을 검토합니다.

- **Performance Highlights**: 이 프레임워크는 평균적으로 방어 성공률을 18.16% 향상시키고, 과도한 거부율(over-refusal rate)을 6% 줄이는 성과를 거두었습니다. 입력 가드는 초기 단계에서 위험을 효과적으로 필터링하고, 계획 및 연구 가드는 인용 규율과 출처 신뢰성을 강화하여 연구의 전체 품질을 높입니다. DEEPRESEARCHGUARD는 포괄적인 개방형 평가를 통해 해로운 콘텐츠 전파를 차단할 수 있는 능력을 입증합니다.



### Enhancing Large Language Model Reasoning via Selective Critical Token Fine-Tuning (https://arxiv.org/abs/2510.10974)
- **What's New**: 이번 연구에서는 Critical Token Fine-tuning (CFT)이라는 새로운 접근 방식을 제안합니다. CFT는 중요성에 따라 결정적인 토큰에만 업데이트를 적용하기 위해 반사실적 섭동(counterfactual perturbation) 과정을 이용하여 기능적으로 필수적인 토큰을 식별합니다. 이 방법은 모든 토큰에 균일한 처리를 수행하는 기존의 Supervised Fine-tuning(SFT)보다 성능을 극대화하고 다양성을 유지할 수 있도록 설계되었습니다.

- **Technical Details**: CFT는 기존의 SFT가 모든 토큰에 동일하게 손실을 가하는 문제를 해결합니다. 이는 각 올바른 해결책의 토큰을 대체 후보로 교체하여, 모든 섭동이 잘못된 최종 답을 생성하면 원래의 토큰이 중요하다고 표시하는 방법입니다. 이를 통해 CFT는 최적화를 결정적인 추론 단계에 집중하고 비결정적인 토큰은 제외하여 성과를 높입니다.

- **Performance Highlights**: CFT는 세 가지 모델군(Qwen, OLMo, LLaMA)과 11개의 수학적 추론 벤치마크에서 평가된 결과, 12% 미만의 선택된 토큰에 대해서도 기존 SFT를 지속적으로 능가하는 성과를 보였습니다. 또한 첫 번째로 ML 기반의 Reinforcement Learning(RL) 초기화로 활용될 때에는 지속적인 성능 향상과 탐색을 위한 높은 엔트로피(entropy)를 유지시키는 것을 입증했습니다.



### RV-HATE: Reinforced Multi-Module Voting for Implicit Hate Speech Detection (https://arxiv.org/abs/2510.10971)
Comments:
          10 pages, 9 figures, 12 tables

- **What's New**: 이번 연구는 RV-HATE라는 새로운 탐지 프레임워크를 소개합니다. 이 프레임워크는 각 증오 발언 데이터셋의 고유한 특성을 고려하여 개발되었습니다. 데이터의 다양성과 언어적 특성을 다루기 위한 여러 전문 모듈로 구성되어 있습니다.

- **Technical Details**: RV-HATE는 강화 학습(reinforcement learning)을 사용하여 각 모듈의 기여도를 최적화합니다. 각 모듈은 증오 발언의 특정 언어적 또는 맥락적 특성에 집중하고, 이를 통해 종합적인 결정을 내리는 투표 메커니즘(voting mechanism)을 활용합니다. 이 방법은 기존의 고정 방식(static methods)에서 한 걸음 더 나아갑니다.

- **Performance Highlights**: RV-HATE는 데이터셋의 특정 속성에 맞춘 탐지 프로세스를 통해 탐지 정확도를 향상시키는 두 가지 주요 이점을 제공합니다. 또한 각 데이터셋의 독특한 특성에 대한 해석 가능한 통찰도 제공합니다. 이를 통해 암묵적인 증오 발언을 효과적으로 다루며, 기존 방법들보다 우수한 성과를 달성했습니다.



### Judge Before Answer: Can MLLM Discern the False Premise in Question? (https://arxiv.org/abs/2510.10965)
- **What's New**: 본 논문에서는 MLLMs가 사실과 다르거나 비논리적인 전제에 직면했을 때 여전히 사실을 인식하지 못하는 문제를 해결하기 위해, 자동화된 파이프라인을 통해 광범위한 전제 질문 벤치마크를 구축하는 방법을 제시합니다. JBA라는 새로운 벤치마크는 세 가지 주요 유형과 열세 가지 하위 유형으로 전제를 체계적으로 분류하여 MLLMs의 성능을 보다 엄격하게 평가할 수 있도록 합니다. 또한, JBA-GRPO라는 향상된 인식 프레임워크를 제안하여 모델이 잘못된 전제를 식별하고 명시적으로 반박할 수 있는 능력을 강화합니다.

- **Technical Details**: 제안된 JBA 데이터셋은 세 가지 주요 단계로 구성된 완전 자동화된 생성 파이프라인을 통해 구축됩니다. 각 이미지는 전제 유형에 따라 전제를 추출하고, 그 후에 이미지의 내용을 기반으로 한 질문을 생성합니다. 데이터셋은 지각 수준, 인지 수준, 추론 수준의 세 가지 계층적 분류로 나누어져, 모델의 성능을 구조적으로 분석할 수 있도록 설계되었습니다.

- **Performance Highlights**: 대규모 실험 결과, JBA-GRPO 프레임워크로 훈련된 모델이 기존 MLLMs보다 잘못된 전제를 인식하는 성능이 크게 향상됨을 보여줍니다. JBA 데이터셋과 JBA-GRPO는 MLLMs의 신뢰성 있는 다중 모드 추론을 위한 새로운 표준을 수립하며, 기존 접근 방식의 한계를 뚜렷하게 드러내어 연구에 중요한 기여를 하고 있습니다.



### KOTOX: A Korean Toxic Dataset for Deobfuscation and Detoxification (https://arxiv.org/abs/2510.10961)
Comments:
          25 pages, 5 figures, 25 tables

- **What's New**: 이 논문은 빠르게 확장되는 온라인 커뮤니케이션에서의 독성 콘텐츠를 다루기 위해 새로운 데이터셋인 KOTOX: Korean Toxic Dataset을 제안합니다. 기존의 연구들이 주로 영어에 집중해, 저자원 언어는 저희의 연구에서 소외되었습니다. 이러한 문제를 해결하기 위해, 한국어의 언어적 특성에 기반한 다양한 변조 기법을 분류하고 실제 예제를 바탕으로 변환 규칙을 정의했습니다.

- **Technical Details**: KOTOX는 변조(deobfuscation)와 독성 제거(detoxicification)를 동시에 지원하는 최초의 데이터셋으로, 한국어에서의 독성 표현을 감지하는 데 도움을 줄 것입니다. 우리는 세 가지 난이도 수준(쉬움, 보통, 어려움)을 가진 데이터셋을 구성하여, 사용자가 사용하는 다양한 회피 기법을 포괄합니다. 이러한 변환 규칙을 기반으로 한 데이터셋은 한국어 독성 콘텐츠의 이해와 완화에 중요한 기여를 할 것입니다.

- **Performance Highlights**: KOTOX 데이터셋은 저자원 언어를 대상으로 한 대형 언어 모델(LLM)의 독성 표현 감지 성능을 크게 향상시킬 것으로 기대됩니다. 저자들이 제공하는 코드와 데이터는 연구자들과 개발자들이 접근 가능하게 되어, 한국어 독성 콘텐츠 문제 해결에 적극적으로 기여할 것으로 보입니다.



### Punctuation-aware treebank tree binarization (https://arxiv.org/abs/2510.10951)
- **What's New**: 이 논문에서는 구두점에 대한 인식을 포함한 트리뱅크(binrain) 이진화(binarization)를 위한 curated resource와 평가 도구를 제시합니다. 기존의 이진화 파이프라인은 헤드 선택(head selection) 전에 구두점을 삭제하여 구성의 형태를 변화시키고 헤드-자식 식별에 악영향을 미칩니다. 본 연구는 구두점을 이진화 이전에 형제 노드(sibling nodes)로 보존하는 재현 가능한 파이프라인을 소개합니다.

- **Technical Details**: 논문에서는 세 가지 주요 자료를 제공합니다: (1) 구두점을 보존하는 이진화 파이프라인, (2) 파생 산출물 및 메타데이터, (3) 헤드-자식 예측, 왕복(reversibility) 검증 및 기초 자료와의 구조적 호환성을 평가하는 평가 도구입니다. 이 연구는 Penn Treebank에서 구두점 인식 전처리가 헤드 예측 정확도를 73.66%(Collins 규칙)와 86.66%(MLP)에서 91.85%로 향상시킴을 보여줍니다.

- **Performance Highlights**: 구두점 인식 전처리의 성과는 CCGbank와의 정렬에서 경쟁력을 보이며, 모든 코드, 구성 파일 및 문서가 제공되어 다른 말뭉치(corpora)로의 복제 및 확장을 가능하게 만듭니다. 이를 통해, 연구자들은 새로운 자료를 쉽게 접근하고 사용할 수 있으며, 향후 연구에 기여할 수 있는 기반을 마련합니다.



### End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF: A Reproducibility Study (https://arxiv.org/abs/2510.10936)
- **What's New**: 이번 연구에서는 Ma와 Hovy (2016)에서 제안한 BiLSTM-CNN-CRF 아키텍처의 재현성 연구를 소개합니다. 이 모델은 CNN, BiLSTM, CRF의 세 가지 주요 구성요소를 결합하여 시퀀스 레이블링 작업에서 뛰어난 성능을 발휘하며, 수작업 특징을 제거하는 엔드 투 엔드 방식으로 구현됩니다. 연구 결과, CoNLL-2003 NER 데이터셋에서 91.18%의 F1 스코어를 달성하여 모델의 효과성을 입증하였습니다.

- **Technical Details**: BiLSTM-CNN-CRF 모델은 문자 수준의 CNN 인코딩, 단어 수준의 BiLSTM 인코딩, CRF 기반의 구조적 예측을 이용한 세 가지 주요 구성 요소로 구성됩니다. 각 문자는 CNN을 통해 형성적 정보를 추출하며, 단어 임베딩과 결합하여 BiLSTM을 거쳐 최종적으로 태그 점수를 생성합니다. CRF 레이어는 태그 간의 의존성을 고려하여 일관된 태그 시퀀스를 보장합니다.

- **Performance Highlights**: 모델은 CoNLL-2003 NER 데이터셋에서 91.18% F1 스코어를 달성하며 원 논문의 결과와 유사한 성과를 보입니다. Penn Treebank WSJ POS 태깅에서는 97.52%의 정확도로 원래 97.55%의 성과와 거의 일치하는 결과를 나타내었습니다. 구성 요소의 기여도를 분석하기 위한 제거 연구를 통해 CRF 레이어가 태그 일관성 보장에 중요한 역할을 한다는 것을 발견했습니다.



### Evaluating Language Models' Evaluations of Games (https://arxiv.org/abs/2510.10930)
Comments:
          Pre-print

- **What's New**: 이 논문은 인공지능(AI) 시스템의 평가 방식을 새로운 패러다임으로 제안합니다. 전통적으로 문제 해결 능력에 중점을 두었던 AI 평가 방식에서 벗어나, AI가 게임을 평가하는 방식을 연구하고 있습니다. 연구자들은 100개 이상의 새로운 보드 게임과 450명 이상의 인간 판단 데이터를 활용하여 현대의 언어 및 추론 모델이 생성한 평가를 비교 분석했습니다.

- **Technical Details**: 연구에서는 게임의 수익성과 재미를 평가하기 위한 두 가지 종류의 질문을 고려합니다. 이러한 질문은 AI 평가의 설계와 관련된 두 가지 차원인 계산의 복잡성과 정량화의 난이도를 아우릅니다. 추론 모델이 이러한 평가를 수행하는 능력을 평가하기 위해, 121121개의 새로운 게임 데이터셋을 사용하고, 각 게임에 대해 두 가지 평가 쿼리를 테스트하였습니다.

- **Performance Highlights**: 추론 모델들은 비추론 언어 모델들에 비해 일반적으로 인간의 게임 평가와 더 일치를 보였습니다. 그러나 모델이 게임 이론적 최적에 가까워질수록 인간 데이터와의 적합성이 약해지는 비몬토닉(non-monotonic) 관계를 발견하였습니다. 또한, 재미를 평가할 때 모델 간의 일관성이 낮고 자원 사용량이 크게 변동함을 관찰하여, 문제 평가 에이전트의 리소스 합리적인 설계를 위한 미래 연구의 필요성을 강조하였습니다.



### GapDNER: A Gap-Aware Grid Tagging Model for Discontinuous Named Entity Recognition (https://arxiv.org/abs/2510.10927)
Comments:
          Accepted by IJCNN 2025

- **What's New**: 이 논문에서는 생물 의학 분야에서 비연속(named entity) 개체 인식을 위한 Gap-aware grid tagging 모델인 GapDNER를 제안합니다. 기존의 방법들이 오류 전파와 디코딩 모호성 문제에 직면했으나, GapDNER는 개체 조각 간의 맥락 간격을 활용하여 이러한 문제를 해결합니다. 또한, 새로운 비아핀(biaffine) 메커니즘과 선형 주의(linear attention)를 통해 내적 규칙성을 모델링하고, 크로스(cross) 주의를 통해 외적 관계를 강화합니다.

- **Technical Details**: GapDNER는 비연속적 개체 구조를 깊이 탐구하여 그 맥락 간격을 새로운 범위로 간주합니다. span(classification)을 토큰 쌍(token-pair) 그리드(tagging) 문제로 변환하고, 두 개의 상호작용 구성 요소를 설계하여 토큰-쌍 그리드 기능을 포괄적으로 모델링합니다. 내부(span) 규칙성 추출 모듈은 비아핀 메커니즘을 사용하여 각 span의 내부 규칙성을 포착하고, 외부(span) 관계 강화 모듈은 크리스크로스(cross) 주의를 활용하여 다양한 span 간의 의미적 관계를 파악합니다.

- **Performance Highlights**: GapDNER는 세 가지 데이터셋에서 실험을 통해 강력한 기준선을 지속적으로 초과 달성하고, 다양한 설정에서 새로운 최첨단 성능을 기록했습니다. 이러한 성과는 비연속 개체 인식에서의 복잡한 개체 구조 인식에 있어 매우 뛰어난 장점을 보여줍니다. 따라서 GapDNER는 비연속 NER 분야에서의 새로운 발전 가능성을 제시합니다.



### ADVICE: Answer-Dependent Verbalized Confidence Estimation (https://arxiv.org/abs/2510.10913)
- **What's New**: 최근 대규모 언어 모델(LLMs)의 발전은 이들이 자연어로 신뢰도를 표현할 수 있도록 해주어 투명성과 신뢰성을 증가시켰습니다. 그러나 이러한 신뢰도는 종종 과도하게 자신감 있는 경향이 있으며, 그 원인은 명확하게 이해되지 않고 있습니다. 본 연구에서는 언어 모델의 신뢰도 동역학을 상세히 분석하고, 답변에 의존하지 않는(answer-independence) 특징을 확인하여 신뢰도 측정의 핵심 요인으로 제시합니다.

- **Technical Details**: 이를 해결하기 위해 ADVICE(Answer-Dependent Verbalized Confidence Estimation)라는 새로운 파인튜닝 프레임워크를 제안합니다. ADVICE는 모델이 자신의 답변에 근거하여 신뢰도를 측정할 수 있도록 만들어, 신뢰도 캘리브레이션을 개선하고 작업 성능을 유지합니다. 실험 결과, ADVICE는 신뢰도 배분에서 더 균형 잡히고 잘 조정된 신뢰도를 제공한다고 확인되었습니다.

- **Performance Highlights**: ADVICE는 기존의 방법들에 비해 우수한 일반화 성능을 나타내며, 다양한 작업에서도 효과를 발휘합니다. 또한, ADVICE는 과도한 자신감을 감소시키고 합리적인 신뢰 표현을 가능하게 하여, 신뢰도와 답변 간의 관계에 대한 이해를 심화시킵니다. 이러한 발견은 과도한 자신감의 기원을 밝히고 더욱 신뢰성 있는 신뢰도 표현을 위한 틀을 마련합니다.



### LLM$\times$MapReduce-V3: Enabling Interactive In-Depth Survey Generation through a MCP-Driven Hierarchically Modular Agent System (https://arxiv.org/abs/2510.10890)
Comments:
          Accepted by EMNLP2025 System Demonstration

- **What's New**: 이번 연구에서는 LLM x MapReduce-V3를 도입하여 장기적인 설문 생성에 최적화된 계층적 모듈러 에이전트 시스템을 개발하였습니다. 이전 버전인 LLM x MapReduce-V2를 기반으로 하여, 독립적인 기능을 갖춘 MCP(functinal-context-protocol) 서버로 구성된 다중 에이전트 구조를 도입하였습니다. 이 시스템은 사용자가 연구 과정에서 더욱 높은 통제와 맞춤화를 가능하게 해주는 인간-인-루프(interaction) 개입을 통해 연구 관점을 정교하게 반영할 수 있습니다.

- **Technical Details**: LLM x MapReduce-V3는 문서 소화, 뼈대(skeleton) 구성 및 수정, 설문 작성의 다단계 워크플로우를 활용합니다. 이 시스템의 중심에는 모델 컨텍스트 프로토콜(MCP)이 있으며, 이는 도구와 모듈이 독립적인 MCP 서버로 구성될 수 있는 표준화된 기능 호출 메커니즘을 제공합니다. 에이전트 플래너가 현재 스탯과 이전 출력을 바탕으로 필요한 모듈을 동적으로 선택하여 비선형적이고 적응형 워크플로우를 가능하게 합니다. 이와 같은 모듈화는 연구의 구조적 계획과 정보 획득을 제어하는 메커니즘을 제공합니다.

- **Performance Highlights**: LLM x MapReduce-V3의 성능을 평가한 결과, 전문가들은 이 시스템이 출판된 다른 심층 연구 시스템보다 더 유용한 뼈대와 더 질 높은 설문지를 생성함을 확인했습니다. 이러한 결과는 MCP 기반의 모듈 계획이 설문 생성의 질을 향상시킬 수 있음을 잘 보여줍니다. 연구 결과는 LLM 기반 모듈형 에이전트 시스템이 정보 수집과 사용자 맞춤화를 통해 설문 작성의 새로운 표준을 제시할 수 있음을 시사합니다.



### Rethinking Agentic Workflows: Evaluating Inference-Based Test-Time Scaling Strategies in Text2SQL Tasks (https://arxiv.org/abs/2510.10885)
Comments:
          Accepted at COLM 2025 SCALR Workshop

- **What's New**: 이 연구는 Text-to-SQL(Text2SQL) 시스템에서 비전문가 사용자가 자연어로 산업 데이터베이스를 쿼리할 수 있도록 지원하는 대형 언어 모델(LLMs)의 최신 발전을 조사합니다. 저자들은 여섯 가지 경량의 테스트 시간 확장(test-time scaling) 전략과 네 개의 LLM을 평가하며, 이를 BIRD Mini-Dev 벤치마크에서 성능 비교합니다. 연구 결과, Divide-and-Conquer prompting과 few-shot demonstrations가 성능 향상에 긍정적인 영향을 미치는 것으로 나타났습니다.

- **Technical Details**: 텍스트를 SQL 쿼리로 변환하는 과정에서 사용되는 테스트 시간 확장 전략은 자연어 질문을 실행 가능한 SQL 쿼리로 변환하는 Text2SQL 작업의 효율성을 높입니다. 연구에서는 SQL Writer(SW), Executor(EX), SQL Refiner(SR)와 같은 구성 요소를 포함한 여섯 개의 에이전트 기반 워크플로우를 평가합니다. 다양한 LLM의 성능을 SQL 정확도, 지연 시간(inference latency), 토큰 소비(token consumption)와 같은 메트릭을 통해 분석합니다.

- **Performance Highlights**: 결과적으로, Divide-and-Conquer 및 few-shot demonstrations가 SQL 쿼리의 품질을 유의미하게 향상시키는 것으로 나타났습니다. 이는 일반 모델과 고급 추론을 위해 미세 조정된 모델 모두에 해당됩니다. 그러나 복잡한 워크플로우 추가는 혼합된 결과를 초래하며, 기본 모델 선택이 성능에 미치는 영향을 강조합니다.



### DUAL-Bench: Measuring Over-Refusal and Robustness in Vision-Language Models (https://arxiv.org/abs/2510.10846)
Comments:
          25 pages, 91 figures, submitted to Oct ARR under reviewing

- **What's New**: 이 논문은 비전-언어 모델(vision-language models)의 안전성과 유용성 간의 균형을 유지하는 문제를 다룹니다. 특히, 'over-refusal' 현상, 즉 모델이 지나치게 조심스럽게 반응하여 악의가 없는 요청조차 거부하는 문제에 초점을 맞추고 있습니다. 본 논문은 DUAL-Bench라는 새로운 다중 모달 벤치마크를 소개하며, 이 벤치마크는 VLM에서의 과다 거부와 안전한 완수(safe completion)에 중점을 둡니다.

- **Technical Details**: DUAL-Bench는 12개 위험 카테고리(hazard categories)에 걸쳐 18개의 VLM을 평가합니다. 이 연구는 의미 보존 시각 변형(semantics-preserving visual perturbations) 하에서 모델의 강건성(robustness)을 평가하며, 실험을 통해 모델들이 안전하게 요청을 완수해야 함을 강조합니다. 안전한 완수의 이상적인 행동은 악의 없는 요청의 일부를 이행하면서, 잠재적인 해로운 요소에 대해 명확히 경고하는 것입니다.

- **Performance Highlights**: 실험 결과에 따르면, GPT-5-Nano는 12.9%의 안전한 완수를 달성했으며, GPT-5 모델은 평균 7.9%, Qwen 모델은 단 3.9%에 불과하여 상당한 개선의 여지가 있음을 보여줍니다. DUAL-Bench는 VLM이 복잡한 다중 모달 환경에서도 안전성과 유용성을 동시에 유지할 수 있도록 보다 섬세한 정렬 전략(nuanced alignment strategies)의 발전을 촉진할 것으로 기대됩니다.



### Happiness is Sharing a Vocabulary: A Study of Transliteration Methods (https://arxiv.org/abs/2510.10827)
- **What's New**: 이번 연구는 다국어 자연어 처리(NLP)에서 언어 간 간극을 메우는 데 도움이 되는 전사(transliteration)의 효과를 조사합니다. 특히 비라틴 문자(non-Latin script)를 사용하는 언어에서 긍정적인 결과를 보여주는 방법을 다룹니다. 연구진은 공유된 스크립트(shared script), 겹치는 토큰(vocabulary), 공유 음운(shared phonology)이 다국어 모델 성능에 미치는 영향을 분석합니다.

- **Technical Details**: 연구에서는 전사 방식으로 로마자화(romanization), 음소 전사(phonemic transcription), 대체 암호(substitution ciphers) 및 정자법(orthography)을 사용한 통제 실험을 수행했습니다. 각 모델은 두 가지 하위 작업인 명명된 개체 인식(named entity recognition, NER)과 자연어 추론(natural language inference, NLI)에서 평가되었습니다. 실험 결과, 로마자화 방식이 8개의 평가 설정 중 7개에서 다른 입력 유형보다 유의미하게 우수한 성능을 보였습니다.

- **Performance Highlights**: 연구하신 하이퍼파라미터(hyperparameter)가 로마자화가 가장 효과적인 접근법으로서 성공에 기여한 요소들을 분석했습니다. 특히, 사전 훈련된 언어와 공유된 긴 (서브워드) 토큰(subword tokens)을 갖는 것이 모델 활용도를 높인다는 것을 강조합니다. 이러한 결과는 다국어 NLP 분야에서 전사 기술의 잠재력을 뒷받침하며, 향후 연구에 중요한 방향성을 제시합니다.



### Is Implicit Knowledge Enough for LLMs? A RAG Approach for Tree-based Structures (https://arxiv.org/abs/2510.10806)
Comments:
          Waiting for Conference Response

- **What's New**: 본 논문에서는 'Retrieval-Augmented Generation (RAG)' 방식을 통해 구조화된 데이터(예: 코드 파일)에서 생성된 응답을 향상시키기 위한 새로운 하향식(bottom-up) 방법을 제안합니다. 이 방법은 계층적 구조(예: 트리)의 지식을 선형화(linearize)하여 각 계층에서 암묵적(implicit) 요약을 생성합니다. 이 접근 방식은 기존의 RAG 방법론보다 더 효율적이며, 68% 이상의 문서 수 감소로 응답 품질을 비슷하게 유지하는 것을 보여줍니다.

- **Technical Details**: 이 논문은 계층적 구조에서 암묵적 지식을 생성하기 위한 새로운 방법을 제안합니다. 제안된 방법은 리프 노드(leaf node)에서부터 시작하여 모든 리프 노드에 대한 '템플릿(template)' 지식을 획득한 후, 각 부모 노드를 순회하며 자식들로부터 받은 암묵적 지식을 바탕으로 상위 요약을 생성합니다. 이러한 선형화 과정은 벡터 데이터베이스에 저장될 정보 조각을 최적화하고 토큰 수를 제한하여 효율성을 높입니다.

- **Performance Highlights**: 우리의 실험은 GM의 비구조화된 코드 리포지토리를 사용하였으며, 제안된 방법이 전통적인 RAG 방법에 비해 응답 품질이 유사함에도 불구하고 저장된 데이터 양을 거의 4분의 1로 줄임을 보여줍니다. 이를 통해 복잡한 구조적 정보를 처리하는 데 있어 암묵적 지식이 충분하고 효율적일 수 있음을 제안합니다. 또한 이 연구는 RAG 프레임워크에서 지식 관리를 위한 효과적이고 확장 가능한 방법 개발의 필요성을 강조합니다.



### Toward Human-Centered Readability Evaluation (https://arxiv.org/abs/2510.10801)
Comments:
          Accepted to the 4th Workshop on Bridging Human-Computer Interaction and NLP (HCI+NLP) at EMNLP 2025, Suzhou, China

- **What's New**: 이 논문은 건강 정보 텍스트의 단순화를 위한 새로운 평가 체계인 Human-Centered Readability Score (HCRS)를 제안합니다. 기존의 자동화된 텍스트 평가 지표는 주로 표면적 특성에 초점을 맞추지만, HCRS는 인간 중심의 특성인 명확성(clarity), 신뢰성(trustworthiness), 톤 적절성(tone appropriateness), 문화적 관련성(cultural relevance), 그리고 실행 가능성(actionability)을 고려합니다. 이 새로운 프레임워크는 텍스트 단순화가 공공 보건에서 진정으로 효과적이기 위해서는 단순히 단어의 난이도를 낮추는 것 이상이 필요하다는 점을 강조합니다.

- **Technical Details**: HCRS는 자동화된 측정과 인간의 구조화된 피드를 통합하여 읽기 쉬움의 관계적 및 맥락적 측면을 포착합니다. 이 연구는 개별 사용자의 요구와 기대에 보다 부합하는 건강 텍스트 단순화를 평가하는것을 목표로 하고 있으며, 평가 파이프라인에 참여적 설계(participatory design)와 피드백 수집(interactive feedback collection)을 통합하는 방법을 논의합니다. 이를 통해 HCRS는 건강 문서에 대한 기존의 접근 방식을 재정의하고, 다양한 사용자의 기대에 부합하는 시스템 설계를 지원합니다.

- **Performance Highlights**: HCRS는 기존의 단일 사용 지표보다 사용자 평가와의 강력한 정렬을 목표로 합니다. 연구에서는 텍스트 단순화 후의 명확성, 신뢰성, 적절한 톤 및 문화적 관련성, 실행 가능성 등이 어떻게 사용자 경험과 연결되는지를 분석하였습니다. 초기 연구 결과는 텍스트 단순화가 정보의 인지적 부담을 줄이고 사용자가 이를 더욱 쉽게 이해할 수 있도록 돕는 것을 보여주며, 특히 교육 수준이 낮거나 영어 능력이 제한된 사용자에게서 가장 큰 효과를 나타냅니다.



### Review of Inference-Time Scaling Strategies: Reasoning, Search and RAG (https://arxiv.org/abs/2510.10787)
- **What's New**: 본 논문은 대형 언어 모델(LLMs)의 성능 향상을 다루고 있으며, 기존의 데이터 학습과 모델 크기 증가에 의존하는 방식에서 벗어나, 추론 시간(inference-time)에서의 컴퓨테이션을 추가하여 성능을 개선하는 새로운 방향성을 제시합니다. 이는 모델 재훈련 없이도 다운스트림 작업에서 LLM의 효율성을 극대화할 수 있는 가능성을 보여줍니다. 이 리뷰는 출력 중심(output-focused) 및 입력 중심(input-focused) 방법으로 분류하여 다양한 기술을 체계적으로 조사합니다.

- **Technical Details**: 리뷰에서는 다양한 추론 시간 스케일링 기법을 다루고 있으며, 그 중 출력 중심 방법으로는 Chain-of-Thought (CoT), Tree-of-Thought (ToT), Reason+Act (ReAct)와 같은 복잡한 다단계 생성 전략이 포함됩니다. 입력 중심 방법은 주로 few-shot 학습과 Retrieval-Augmented Generation (RAG)으로 나뉘며, RAG는 외부 정보를 결합하여 생성을 향상시키는 데 중점을 둡니다. 이러한 기법들은 LLM이 복잡한 문제를 해결할 수 있도록 도와주는 다양한 접근 방식을 제안합니다.

- **Performance Highlights**: 본 연구에서는 다양한 추론 기반 기법들이 LLM의 성능을 통해 단계별 문제 해결의 효용성을 보여주는 결과를 포함하고 있습니다. 예를 들어, CoT는 대규모 모델에 대해 복잡한 작업의 성능을 크게 향상시켰으며, SC(자기 일관성)와 ToT(사고 나무)는 더욱 정교한 문제 해결을 가능하게 합니다. 이러한 혁신적인 방법들은 LLM의 적응성 및 정확성을 크게 변동시켜 최종적으로는 더 나은 출력 품질을 달성하는 데 기여합니다.



### HiligayNER: A Baseline Named Entity Recognition Model for Hiligaynon (https://arxiv.org/abs/2510.10776)
Comments:
          Camera-ready for PACLIC 2025 (ACL Proceedings)

- **What's New**: 이번 연구는 필리핀의 Hiligaynon 언어를 위한 최초의 공개 기반 모델인 HiligayNER를 소개하고 있습니다. HiligayNER는 8,000개 이상의 주석이 달린 문장에서 수집된 데이터셋을 사용하며, Named Entity Recognition (NER) 작업에 사용됩니다. 연구의 목표는 Hiligaynon 언어를 포함한 저자원 언어에 대한 자연어 처리(NLP) 개발을 지원하는 것입니다.

- **Technical Details**: HiligayNER는 세 가지 단계로 구성되어 구축되었습니다: 데이터 수집, 전문가 주석, 신뢰성 테스트입니다. 수집된 데이터는 BIO 태깅 형식으로 주석이 달린 문장 수준의 데이터로 정리되었으며, 주요 엔터티 카테고리는 인물(PER), 조직(ORG), 위치(LOC) 및 기타(OTH)입니다. 또한, Multilingual BERT(mBERT)와 XLM-RoBERTa(XLM-R)라는 두 개의 멀티 언어 트랜스포머 모델이 Hiligaynon 텍스트에 대해 미세 조정되었습니다.

- **Performance Highlights**: 평가 결과, 두 모델 모두 각각 80% 이상의 정밀도(precision), 재현율(recall), F1 점수를 기록했습니다. 또한, Cebuano 및 Tagalog 언어와의 교차 언어 평가를 통해 HiligayNER의 전이 가능성이 우수하다는 것을 보여주었습니다. 이러한 결과들은 Hiligaynon을 포함한 저자원 언어에서 다국어 NLP의 더 넓은 적용 가능성을 시사합니다.



### Large Language Models for Full-Text Methods Assessment: A Case Study on Mediation Analysis (https://arxiv.org/abs/2510.10762)
- **What's New**: 이번 논문에서는 체계적 검토(systematic reviews)가 과학적 증거를 종합하는 데 중요하지만, 세부 방법론 정보를 추출하는 데 많은 노동력이 필요하다는 점을 지적합니다. 대형 언어 모델(Large Language Models, LLMs)이 이러한 방법론 평가를 자동화할 수 있는 가능성을 제시하며, 이를 통해 증거 종합을 혁신할 수 있을 것으로 기대됩니다.

- **Technical Details**: 논문에서는 인과 매개 분석(causal mediation analysis)을 대표적인 방법론 영역으로 설정하고, 180개의 전체 텍스트 과학 논문을 기반으로 최신 LLM의 성능을 전문가 심사자와 비교 평가했습니다. 모델의 성과는 인간의 판단과 밀접하게 상관관계가 있었으며(정확도 상관 0.71; F1 상관 0.97), 명시적으로 진술된 방법론 기준에 대해 거의 인간 수준의 정확도를 달성했습니다. 그러나 복잡한 추론 중심 평가에서 정확도가 급격히 하락하여 전문가 심사자보다 최대 15% 낮은 결과를 보였습니다.

- **Performance Highlights**: 모델 오류는 주로 피상적인 언어적 단서에서 발생했으며, 예를 들어 'longitudinal' 또는 'sensitivity'와 같은 키워드를 엄격한 방법론 접근의 자동적 증거로 잘못 해석하여 체계적인 오분류로 이어졌습니다. 긴 문서는 모델 정확도를 저하시켰고, 발표 연도는 유의미한 영향을 미치지 않았습니다. LLM은 명시적인 방법론적 특징을 식별하는 뛰어난 성능을 보이지만, 미묘한 해석을 위해서는 인간의 감독이 필요하다는 중요한 패턴을 제안합니다.



### Sarcasm Detection Using Deep Convolutional Neural Networks: A Modular Deep Learning Framework (https://arxiv.org/abs/2510.10729)
Comments:
          4 pages, 5 figures

- **What's New**: 이 논문은 텍스트에서의 sarcasm(풍자) 탐지를 위한 모듈형 딥러닝 프레임워크를 제안합니다. Deep Convolutional Neural Networks(DCNNs)와 BERT와 같은 맥락 모델을 활용하여 언어적, 감정적, 맥락적 신호를 분석합니다. 시스템은 다층 아키텍처를 통해 감정 분석, 맥락 임베딩, 언어적 특징 추출 등을 통합하여 현재 개념 단계에 있지만 실제 어플리케이션에 대한 가능성을 보여줍니다.

- **Technical Details**: 제안된 시스템은 여러 전문화된 감지 모듈로 구성된 모듈형 아키텍처를 채택합니다. 모듈에는 감정 분석, 맥락 임베딩, 언어적 특징 및 감정 탐지가 포함되어 있으며, 각 모듈은 독립적으로 텍스트를 분석하여 특성 벡터를 생성합니다. 이러한 모듈들은 최종적으로 통합되어 전체 시스템의 유연성과 확장성을 보장합니다.

- **Performance Highlights**: 멀티모달 데이터셋을 활용한 사례 연구에서 BERT와 DenseNet의 조합 모델은 93.2%의 정확도를 기록하였으며, 이는 시각적 신호가 풍자 탐지에서 어떻게 중요한 역할을 하는지 보여줍니다. 단일 모델인 BERT는 88.6%, DenseNet는 74.3%의 정확도를 기록하였으며, 이 결과는 두 모델의 융합이 크게 기여했음을 나타냅니다.



### RePro: Training Language Models to Faithfully Recycle the Web for Pretraining (https://arxiv.org/abs/2510.10681)
- **What's New**: 이 연구에서는 RePro라는 새로운 웹 재활용 방법을 소개합니다. 이는 비교적 작은 LM을 강화 학습(reinforcement learning)으로 훈련시켜 효과적이고 충실한 재구성을 생성하는 방법입니다. 이 방법은 품질과 충실성을 보장하는 여러 보상 구조로 설계되어 있으며, 이를 통해 기계가 유기 데이터를 고품질 재구성으로 변환할 수 있도록 최적화됩니다.

- **Technical Details**: RePro는 데이터 품질 평가_metric_인 DataMan 점수를 품질 보상으로 선택하고, BERTScore를 포함한 세 가지 충실성 보상을 활용하여 유기 데이터의 의미와 구조를 유지합니다. 이 과정에서 RePro는 72B 토큰이 포함된 유기 데이터 집합에서 훈련됩니다. 또한, 4B 모델로 훈련된 RePro는 고품질 데이터의 효율성을 2-3배 향상시키는 데 성공했습니다.

- **Performance Highlights**: 실험 결과, RePro는 LLM의 프리트레이닝(pretraining)에서 유기 기반 기준선보다 평균 4.7%에서 14.0% 높은 정확도를 보였습니다. 또한, 최신 웹 재활용 방법인 ReWire를 초과하여 성능을 발휘했습니다. 이 연구의 결과는 RePro가 유기 데이터의 품질이 유지되면서도 효율적으로 재활용할 수 있는 경로를 제공함을 보여줍니다.



### Unlocking LLM Safeguards for Low-Resource Languages via Reasoning and Alignment with Minimal Training Data (https://arxiv.org/abs/2510.10677)
Comments:
          Accepted to MRL Workshop at EMNLP 2025

- **What's New**: 이 논문에서는 ConsistentGuard라는 새로운 다국어 안전 장치 개발을 제안하고 있으며, 이를 통해 LLM(대형 언어 모델)의 안전성을 높이고 있습니다. 기존의 분류기 기반 접근 방식은 해석 가능성이 떨어지고 저자원 언어에서의 성능이 낮은 문제를 해결하기 위해 이 모델은 추론 기반의 학습 프레임워크를 채택하고 있습니다. 특히, 이 방법은 1,000개의 훈련 샘플만으로도 다양한 언어에서 뛰어난 성능을 보여줍니다.

- **Technical Details**: ConsistentGuard의 훈련 프레임워크는 세 가지 주요 단계로 구성됩니다: 콜드 스타트(cold start), 추론 훈련(reasoning training), 그리고 다국어 정렬(cross-lingual alignment). 콜드 스타트 단계에서는 큰 매개변수를 가진 LLM의 지식을 3B 기본 모델로 증류하여 특정 안전 작업에 대한 초기 지식을 제공합니다. 이후 GRPO를 기반으로 한 추론 훈련 과정에서 추론의 길이와 다양성을 균형 있게 조절하는 보상을 설계합니다.

- **Performance Highlights**: 세 개의 데이터셋에서 1,000개의 샘플과 3B 파라미터만으로도 대부분의 언어에서 두 번째로 높은 순위를 기록하였습니다. 비교 대상으로 사용된 모델들은 100,000개 이상의 샘플로 훈련되었으나, ConsistentGuard는 적은 데이터로도 탁월한 결과를 도출하였습니다. 이러한 실험 결과는 제안된 파이프라인과 정렬 방법의 효과를 입증하며, 향후 연구를 위한 코드도 공개하였습니다.



### BrowserAgent: Building Web Agents with Human-Inspired Web Browsing Actions (https://arxiv.org/abs/2510.10666)
Comments:
          10 pages

- **What's New**: 최근 연구들은 웹 환경과 상호작용하는 LLMs의 중요성을 강조하고 있으며, BrowserAgent는 인간의 브라우징 행동을 모방해 복잡한 웹 작업을 수행하는 더 상호작용적인 에이전트를 제안합니다. 기존의 Search-R1과 WebDancer는 정적 텍스트 콘텐츠에 의존했던 반면, BrowserAgent는 Playwright를 사용하여 실시간 웹 페이지에서 작업을 수행합니다. 이 논문은 두 단계의 훈련 방식(Supervised Fine-Tuning (SFT)과 Rejection Fine-Tuning (RFT))을 통해 모델의 일반화 능력을 향상시킵니다.

- **Technical Details**: BrowserAgent는 브라우저 상에서 직접 원시 웹 페이지 작업을 수행함으로써 사람과 유사한 행동을 통해 복잡한 작업을 처리합니다. 네 가지 범주의 사용자의 행동(페이지 작업, 탭 관리, URL 탐색, 완료 행동)을 정의하여, 웹 상에서 실시간으로 상호작용하며 훈련 데이터를 생성합니다. 이러한 접근 방식은 모델이 실제 웹 콘텐츠와 상호작용하면서 자연스럽게 정보 검색과 이해 능력을 키우도록 합니다.

- **Performance Highlights**: BrowserAgent는 5.3K의 훈련 샘플로도 Search-R1를 초과하는 성능을 보여주며, 다양한 Open-QA 작업에서 뛰어난 결과를 나타냅니다. 예를 들어, BrowserAgent-7B는 HotpotQA와 같은 멀티 홉 QA 작업에서 약 20%의 성능 향상을 기록했습니다. 이러한 결과들은 BrowserAgent가 더 상호작용적이고 확장 가능한 웹 에이전트를 구축하는 데 기여할 수 있음을 보여줍니다.



### AGENTIQL: An Agent-Inspired Multi-Expert Framework for Text-to-SQL Generation (https://arxiv.org/abs/2510.10661)
Comments:
          Accepted at NeurIPS 2025, ER "Efficient Reasoning" workshop

- **What's New**: 본 논문에서는 AGENTIQL이라는 새로운 다중전문가 아키텍처를 제안합니다. 이 아키텍처는 쿼리 분해를 위한 추론 에이전트, 서브쿼리 생성을 위한 코딩 에이전트, 열 선택을 위한 정제 단계로 구성되어 있습니다. AGENTIQL은 모듈식 파이프라인을 통해 효율성과 정확성을 동시에 달성하여 더 나은 해석 가능성을 제공합니다.

- **Technical Details**: AGENTIQL의 주요 구성 요소는 질문을 더 작은 서브질문으로 분해하고, 이에 대한 SQL 쿼리를 생성하며, 필요 시 열 선택을 통해 SQL 쿼리를 조정하는 것입니다. 이 과정에서 적응형 라우터가 사용되어 쿼리의 복잡성에 따라 자원을 효과적으로 분배합니다. 이론적으로 이 구조는 다양한 데이터베이스 스키마에 대한 처리 능력을 향상시키고, 복잡한 추론을 가능하게 합니다.

- **Performance Highlights**: Spider 벤치마크에서 AGENTIQL은 최대 86.07%의 실행 정확도를 달성하며, 기존의 GPT-4 기반 SOTA와의 격차를 좁힙니다. 이 성능은 라우팅 메커니즘의 효율성에 따라 달라지며, 기존 LLM보다 작은 모델을 사용할 때에도 높은 성능을 유지합니다. AGENTIQL은 또한 중간 추론 단계를 노출하여 투명성을 높이고, 해석 가능한 의미 파싱 접근 방식을 제공합니다.



### You're Not Gonna Believe This: A Computational Analysis of Factual Appeals and Sourcing in Partisan News (https://arxiv.org/abs/2510.10658)
- **What's New**: 이번 연구는 미디어 편향(Media Bias)을 탐구하는 기존 연구와는 달리, 사실 보도에 대한 인식론적 전략(Epistemic Strategies)을 분석합니다. CNN과 Fox News의 대규모 비교를 통해 두 언론사의 보도 스타일을 동일한 사건에 대한 리포트를 통해 분리하여 조명합니다. COVID-19 팬데믹과 이스라엘-하마스 전쟁과 같은 고도로 정치화된 두 시기를 다룬 470,000개 이상의 기사를 바탕으로 연구가 진행되었습니다.

- **Technical Details**: 연구는 아티클 매칭 전략(Article Matching Strategy)을 채택하여 동일 사건에 대한 보고서를 비교하고, FactAppeal 프레임워크를 적용합니다. CNN은 전문가(Experts)와 전문가 문서(Expert Documents)를 인용하여 신뢰성을 구축하며, 반면 Fox News는 뉴스 보고서(News Reports)와 직접 인용(Direct Quotations)을 선호합니다. 이러한 차이를 통해 두 매체의 정보 출처(Sourceing Patterns)가 얼마나 다른지를 밝혀냅니다.

- **Performance Highlights**: CNN의 보도는 더 많은 사실 진술(Factual Statements)을 포함하고 있으며, 외부 출처(External Sources)에 기반하는 경향이 더 강합니다. 반면 Fox News는 명시적 권위(Formal Authority)에 대한 호소를 통해 신뢰성을 구축하는 대신, 뉴스 보도와 직접 인용에 의존하는 경향이 나타났습니다. 이 연구는 편향된 언론이 현실을 구성하기 위해 사용하는 체계적으로 다른 인식론적 전략을 정량화하여 미디어 편향 연구에 새로운 차원을 추가합니다.



### FactAppeal: Identifying Epistemic Factual Appeals in News Media (https://arxiv.org/abs/2510.10627)
- **What's New**: 본 연구에서는 사실 주장(factual claim)의 신뢰성을 나타내는 새로운 작업인 Epistemic Appeal Identification을 제안합니다. 이는 외부 출처나 증거에 의해 사실 진술이 어떻게 지지되는지를 식별하는 것을 목표로 합니다. 또한, 3,226개의 영어 뉴스 문장으로 구성된 수동 주석 데이터셋인 FactAppeal을 소개하며, 이는 기존의 주장 탐지 및 검증 연구를 넘어서는 새로운 접근법을 제공합니다.

- **Technical Details**: FactAppeal은 사실 진술과 그것을 지지하는 외부 출처에 대한 세부 정보를 포함하는 주석을 제공합니다. 이 데이터셋은 사실 주장을 식별하고 이들이 어떻게 전문가, 목격자, 보고서와 같은 출처로부터 지지를 받는지를 설명하는 구조를 담고 있습니다. 연구팀은 2B-9B 파라미터 범위의 여러 encoder 모델과 generative decoder 모델을 사용하여 이 작업을 모델링하였으며, 가장 성능이 우수한 모델은 macro-F1 점수 0.73을 달성했습니다.

- **Performance Highlights**: FactAppeal 데이터셋은 사실 주장과 외부 출처에 대한 정교한 주석을 제공하여 이 분야의 연구에 중대한 기여를 합니다. 기존의 사실성 검증 연구와 달리, 이 연구는 주장에 대한 복잡한 인식 구조를 밝힘으로써 공적 담론에서 정보의 전달 및 검증 방식을 이해하는 데 중요한 단서를 제공합니다. 또한, 이 연구의 결과는 사실 검증, 담론 분석 및 지식의 사회적 흐름 연구에 광범위하게 활용될 수 있습니다.



### Preserving LLM Capabilities through Calibration Data Curation: From Analysis to Optimization (https://arxiv.org/abs/2510.10618)
Comments:
          Accepted by NeurIPS 2025

- **What's New**: 본 연구는 대형 언어 모델(LLM)의 포스트 트레이닝 압축에서 칼리브레이션 데이터가 미치는 영향에 대한 체계적인 조사를 진행합니다. 기존 연구들이 주로 경량화 기술에 집중한 반면, 이 연구는 칼리브레이션 데이터가 LLM의 다양한 능력, 특히 고급 추론 능력에 미치는 영향을 분석합니다. 이를 통해 칼리브레이션 데이터의 최적 특성을 정의하고, 모델 성능 유지에 필요 한 데이터 큐레이션 프레임워크를 제안합니다.

- **Technical Details**: 연구는 LLaMA3-8B-Instruct와 Qwen2.5-7B-Instruct라는 두 가지 LLM을 사용하여 포스트 트레이닝 압축 방법인 SparseGPT와 Wanda를 적용합니다. 칼리브레이션 데이터는 사전 훈련 데이터와 다운스트림 데이터로 구분되며, 각 데이터셋의 조합을 통해 다양한 실험을 수행합니다. 이는 가중치 중요성과 활성화 동적 범위를 평가하여 압축 과정에 활용됩니다.

- **Performance Highlights**: 연구의 결과, 칼리브레이션 데이터의 대표성과 다양성이 LLM의 성능을 유지하는 데 중요한 요소로 작용한다는 것을 발견했습니다. 또한, 제안된 칼리브레이션 데이터 큐레이션 프레임워크는 기존의 압축 방법들과 통합하여 LLM의 핵심 능력을 더욱 효과적으로 보존할 수 있음을 보여주었습니다. 이 연구는 LLM 압축 분야에 중요한 통찰을 제공하고, 향후 연구 방향을 제시합니다.



### Dynamic Topic Evolution with Temporal Decay and Attention in Large Language Models (https://arxiv.org/abs/2510.10613)
- **What's New**: 이 논문은 템포랄 대형 언어 모델(temporal large language models)을 기반으로 한 동적 주제 진화(dynamic topic evolution)을 위한 모델링 프레임워크를 제안합니다. 이 방법은 텍스트의 맥락 임베딩(contextual embeddings)을 획득하기 위해 대형 언어 모델을 사용하고, 이어서 시간에 따른 중요성을 조정할 수 있는 템포랄 디케이 함수(temporal decay function)와 주의 메커니즘(attention mechanism)을 도입합니다.

- **Technical Details**: 모델은 시간 간격에 따라 의미 단위의 중요성을 조정하고 다양한 기간에 걸친 주제 변동을 포착합니다. 생성된 템포럴 표현은 잠재 주제 공간(latent topic space)으로 매핑되며, 여기에서 주제의 동적 진화를 설명하는 상태 전이 행렬(state transition matrix)이 적용됩니다. 공동 최적화 목표(joint optimization objective)는 의미 모델링과 시간적 일관성을 모두 제약하며, 주제 생성의 다양성과 부드러움을 보장합니다.

- **Performance Highlights**: 실제 데이터에서의 실험 결과, 이 프레임워크는 주제의 생성, 확장 및 감소를 효과적으로 포착하였으며, 여러 메트릭(metric)에서 기존 모델보다 뛰어난 성능을 보여주었습니다. 제안된 방법은 대규모 텍스트에서 동적 의미 패턴을 이해하기 위한 체계적인 솔루션을 제공하며, 주제 모델링의 연구 패러다임을 풍부하게 하고 다양한 도메인에서 복잡한 텍스트 분석 작업을 지원합니다.



### Detecting Hallucinations in Authentic LLM-Human Interactions (https://arxiv.org/abs/2510.10539)
- **What's New**: 이 논문에서는 AuthenHallu를 소개하며, 이는 실제 LLM-인간 상호작용에서 완전히 구축된 최초의 환각 탐지 벤치마크입니다. 기존의 벤치마크 대부분이 인위적으로 생성된 것인 반면, AuthenHallu는 LMSYS-Chat-1M 데이터세트에서 수집된 진짜 대화를 바탕으로 합니다. 이 연구는 또한 LLM의 환각 행동을 현실적으로 반영함으로써 환각 탐지 평가의 신뢰성을 높이고 있습니다.

- **Technical Details**: AuthenHallu는 400개의 진짜 LLM-인간 대화를 포함하며, 각 대화는 두 개의 쿼리-응답 쌍으로 구성되어 총 800개의 쿼리-응답 쌍을 생성합니다. 여기서 쿼리-응답 쌍은 환각 발생 여부와 더 세분화된 환각 카테고리(입력 충돌, 맥락 충돌, 사실 충돌)로 주석이 달립니다. 통계 분석 결과, 벤치마크의 31.4%에서 환각이 나타났으며, 특히 수학 및 숫자 문제에 대해 60.0%로 증가했습니다.

- **Performance Highlights**: 연구 결과, 기존 LLM들이 환각 탐지 및 분류 작업에서 여전히 부족한 성과를 보이고 있음을 보여줍니다. AuthenHallu를 통해 수행된 실험은 LLM의 환각 탐지 능력에 대한 철저한 평가를 가능하게 하며, 환각 행동의 전반적인 및 주제별 패턴에 대한 통계적 분석을 제공합니다. 이러한 결과는 LLM의 실제 상호작용 시나리오에서의 사용 가능성을 평가하는 데 유용합니다.



### Merlin's Whisper: Enabling Efficient Reasoning in LLMs via Black-box Adversarial Prompting (https://arxiv.org/abs/2510.10528)
- **What's New**: 이번 연구는 Large Reasoning Models (LRMs)의 과도한 사고(overthinking) 문제를 해결하기 위한 새로운 접근 방식을 제안합니다. 연구팀은 블랙박스(black-box) 환경에서 개방형(open-source)과 폐쇄형(closed-source) 모델을 모두 고려하여, 높은 정확도를 유지하면서 간결한 응답을 이끌어낼 수 있는 방법을 탐구합니다. 이를 통해 AdvPrompt라는 반복적 정제(iterative refinement) 프레임워크를 도입하고, 다양한 관점에서의 적대적 프롬프트(adversarial prompts)를 생성하여 LRMs의 응답 길이를 줄이는 데 성공했습니다.

- **Technical Details**: AdvPrompt는 우선 여러 후보 프롬프트를 합성한 후 전용 개발 세트에서 평가하여 다음 반복을 위한 상위 k개의 성과를 선정합니다. 이 프레임워크는 모델 간의 상호작용을 보다 인간 친화적인 방식으로 전환하려고 하며, 수차례의 반복 과정을 통해 얻어진 최적의 프롬프트를 선택하여 배포합니다. 실험은 여러 벤치마크 데이터셋에서 수행되었으며, 결과적으로 AvPrompt는 응답 성능을 유지하면서도 평균 토큰 사용량을 35%에서 47%까지 줄이는 성과를 이루었습니다.

- **Performance Highlights**: AdvPrompt의 성능은 여러 모델에서 일관되게 나타났습니다. Qwen3 모델 시리즈의 경우, GSM8K 질문에 대한 평균 응답 길이를 3배 줄였고, 다양한 LRMs에 대해 평균적으로 19%에서 41%의 토큰 사용량 감소를 이끌어냈습니다. 특히, 상업적 API인 Claude-3.7과 Gemini-2.5에서도 MATH-500 데이터셋에서 각각 35% 및 47%의 토큰 사용량 감소를 달성하며 그 효과성이 입증되었습니다.



### VOLTAGE: A Versatile Contrastive Learning based OCR Methodology for ultra low-resource scripts through Auto Glyph Feature Extraction (https://arxiv.org/abs/2510.10490)
Comments:
          9 Pages, Plus Appendices, EACL 2024

- **What's New**: 이 연구는 UNESCO의 "세계 언어 위험 아틀라스"에서 2500개의 언어가 멸종 위기 언어로 분류되었음을 밝히고 있습니다. 특히, India의 언어 사용 인구에서 200개가 포함됩니다. 저자들은 VOLTAGE라는 새로운 OCR(Optical Character Recognition) 방법론을 제안하여, 다양한 저자원 언어에 대한 디지털 포함을 지원합니다. 이 방법론은 Takri 스크립트를 사용하여 개발되었습니다.

- **Technical Details**: VOLTAGE는 대량의 데이터가 부족한 저자원 스크립트에 대해 비지도 학습(unsupervised learning) 기반의 OCR 방법입니다. 이 방법론은 이미지 변환 및 Generative Adversarial Networks(GANs)를 활용하여 다양성을 가져오는 데이터 증강(data augmentation) 기법도 포함하였습니다. VOLTAGE의 네 가지 단계는 데이터 추출, 주석 달기, 재강화, 그리고 식별로 구성됩니다.

- **Performance Highlights**: VOLTAGE는 Takri 스크립트에 대해 인쇄된 샘플에서 95%의 정확도, 손으로 쓴 샘플에서는 87%의 정확도를 달성하였습니다. 저자들은 Takri와 다른 Indic 스크립트에서의 결과를 비교하여 이 방법론의 보편적 특성을 입증하였습니다. 이 연구는 226,000개의 기호가 포함된 최대 규모의 Takri 데이터셋을 구축하며, 후속 활용 사례도 포함하고 있습니다.



### UltraLLaDA: Scaling the Context Length to 128K for Diffusion Large Language Models (https://arxiv.org/abs/2510.10481)
- **What's New**: 본 논문에서는 기존의 훈련 없이도 긴 맥락(window) 처리를 가능하게 하는 기법을 연구합니다. 특히, Rotary Positional Embeddings (RoPE)을 조정하여 diffusion LLMs의 긴 컨텍스트 성능을 극대화할 수 있는 방법을 제시합니다. 이를 통해 UltraLLaDA라는 새로운 모델을 도입하여 128K 토큰의 컨텍스트 윈도우를 지원합니다.

- **Technical Details**: diffusion LLM은 기존의 auto-regressive LLM과는 다르게, 전체 시퀀스에 대해 반복적인 노이즈 제거 과정을 사용하여 텍스트를 생성합니다. 로타리 포지셔널 임베딩(RoPE)의 단순한 수정으로 긴 입력 시퀀스에서의 불확실성 모델링을 지원하여, 최적화 안정성과 장기 기억력에 미치는 마스킹 전략을 분석합니다. 저자들은 UltraLLaDA의 구현을 통해 이러한 방법들이 효율적인 포스트 트레이닝(post-training)에서 중요하게 작용함을 보여줍니다.

- **Performance Highlights**: UltraLLaDA는 다양한 긴 맥락 작업을 수행하는 벤치마크에서 뛰어난 성능을 보이며, 훈련 없는 방식인 LongLLaDA와 기존의 LLaDA 모델을 넘어서는 결과를 나타냅니다. 실험 결과에 따르면, UltraLLaDA는 긴 컨텍스트를 처리하면서도 낮은 perplexity와 높은 태스크 정확도를 유지합니다. 이러한 결과들은 UltraLLaDA의 최신 긴 맥락 처리 능력과 경량화된 포스트트레이닝 접근 방식의 실용적 이점을 강조합니다.



### Assessing Large Language Models for Structured Medical Order Extraction (https://arxiv.org/abs/2510.10475)
- **What's New**: MEDIQA-OE 2025 Shared Task는 의사-환자 대화에서 의학적 명령(orders)을 추출하는 데 중점을 두고 있습니다. 이 연구는 LLaMA-4 모델을 사용하여 문맥 내에서의 예시(prompt engineering)를 통해 비전문 영역에서 학습된 대규모 모델의 성능을 평가합니다. 17개 팀 중 5위에 해당하는 성과를 기록하며, 임상 NLP 과제에서의 새로운 기준을 제시합니다.

- **Technical Details**: 제안된 접근법은 메타(Meta)의 LLaMA-4 17B 모델을 사용하며, 소수의 예시를 기반으로 한 프롬프트 엔지니어링(few-shot prompt engineering) 방법론을 적용합니다. 도메인 특화 훈련 없이, 일반 모델을 사용하여 의학적 명령의 유형, 설명, 이유를 식별하는 데 중점을 뒀습니다. 이 접근 방식은 의료 분야에서 탁월한 성과를 거둘 수 있는 대규모 LLM의 가능성을 보여줍니다.

- **Performance Highlights**: 제출된 모델은 평균 F1 점수 37.76을 기록하였고, 특히 명령의 이유(reason)와 출처(provenance) 정확성에서 두드러진 개선을 보였습니다. 이는 효과적인 프롬프트 엔지니어링을 통해 비전문 대규모 모델이 임상 NLP 과제에서 강력하고 확장 가능한 기준선으로 기능할 수 있음을 나타냅니다.



### When or What? Understanding Consumer Engagement on Digital Platforms (https://arxiv.org/abs/2510.10474)
Comments:
          21 pages, 6 figures, 3 tables

- **What's New**: 이번 연구는 TED Talks를 대상으로 한 Latent Dirichlet Allocation (LDA) 모델링을 통해 콘텐츠 제작자와 소비자 간의 선호 차이를 분석하였습니다. 특히 소비자의 참여를 통해 드러난 주제 수요와 제작자가 제공하는 주제 간의 불일치 파악이 이루어졌습니다. 이러한 연구 결과는 콘텐츠 특성이 인기도의 주요 요인이라는 기존 가정을 도전하는 새로운 통찰을 제공합니다.

- **Technical Details**: 이 연구에서는 2006년부터 2022년까지의 TED Talks 4,475개의 대본을 수집하여 8,065,104개의 단어로 구성된 TED Talks Corpus를 구축하였습니다. Latent Dirichlet Allocation (LDA) 모델을 사용하여 자동화되고 체계적인 주제 분석을 진행하며, 이는 텍스트 문서의 토픽 모델링에 가장 널리 사용되는 생성 확률 모델입니다. 연구의 기술적 방법론으로는 scikit-learn 패키지를 활용하여 LDA 분석을 수행하였습니다.

- **Performance Highlights**: 조사 결과, 소비자 참여에 대한 시간적 변동성이 주제 내용보다 더 큰 영향을 미친다는 발견이 있었습니다. 연구는 콘텐츠 제공자가 선호하는 주제와 소비자가 선호하는 주제 간의 차이를 정량적으로 측정하는 '차이 지수'를 도입하여 이들의 선호가 어떻게 변화하는지를 강조합니다. 결과적으로 콘텐츠 제작자 및 마케팅 관계자들이 소비자의 참여를 최적화하기 위한 전략 수립에 유용한 정보를 제공합니다.



### FML-bench: A Benchmark for Automatic ML Research Agents Highlighting the Importance of Exploration Breadth (https://arxiv.org/abs/2510.10472)
Comments:
          Our benchmark is available at: this https URL

- **What's New**: 최근 대형 언어 모델(LLMs)의 발전으로 인해 자동화된 머신러닝(ML) 연구 에이전트에 대한 관심이 다시 높아지고 있습니다. 이들 에이전트는 과학적 발견 과정의 여러 부분을 돕거나 수행하는 데 매우 효과적입니다. 특히 자율적으로 아이디어를 제안하고 실험을 수행할 수 있는 에이전트들은 연구 자동화를 극대화하고 과학적 진전을 가속화하는데 중요한 역할을 합니다.

- **Technical Details**: 본 논문에서는 FML-bench라는 새로운 벤치마크를 소개합니다. 이 벤치마크는 8가지 다양한 기초 ML 문제를 평가하는 데 초점을 맞추고 있으며, 자동 ML 연구 에이전트를 코드 부담을 줄이면서 평가할 수 있도록 설계되었습니다. FML-bench의 설계 원칙은 네 가지로, 핵심 과학적 문제에 초점을 맞추고, 실제 코드베이스를 기반으로 하며, 확장성이 뛰어나고 낮은 코딩 장벽을 제공합니다.

- **Performance Highlights**: FML-bench에서 여러 최신 자동 연구 에이전트들을 평가한 결과, 탐색의 폭을 확장하는 것이 단일한 아이디어 개선보다 더 효과적인 연구 결과를 가져온다고 밝혀졌습니다. Gemini-2.5-Pro가 우리의 프로토콜에 따라 GPT-5보다 우수한 성능을 보였으며, CLI 스타일의 에이전트들은 자동화된 머신러닝 연구에 있어 한계가 있음을 시사합니다. 이러한 결과는 효과적인 에이전트 설계를 위한 실질적인 지침을 제공합니다.



### NIM: Neuro-symbolic Ideographic Metalanguage for Inclusive Communication (https://arxiv.org/abs/2510.10459)
Comments:
          9 pages, EMNLP Findings 2025

- **What's New**: 이번 논문에서는 디지털 커뮤니케이션의 "디지털 격차" 문제를 해결하기 위한 새로운 보편적인 아이디어 메탈랭귀지(ideographic metalanguage)를 소개합니다. 이는 인지적인 요소를 통합하는 신경-기호 AI(Neuro-symbolic AI) 원리를 활용하여 구성되었으며, 세계 지식과 기호적 지식의 휴리스틱을 결합하여 복잡한 개념을 단순한 원자적 개념으로 분해할 수 있게 합니다. 이 시스템은 200명 이상의 반정형 교육을 받은 참여자들과의 협업을 통해 문제를 정의하고 아이디어 그래프를 선택, 검증하는 과정을 거쳤습니다.

- **Technical Details**: 제안된 아이디어 메탈랭귀지는 NIM(Neuro-symbolic Ideographic Metalanguage)이라는 이름으로, 복잡한 생각을 단순한 입자 개념으로 분해할 수 있는 구조를 따릅니다. 이 과정에서 시각적 요소와 텍스트 요소를 구분하여 처리하며, 기호적 추론 모듈은 Natural Semantic Metalanguage(NSM) 이론을 바탕으로 세분화된 의미적 단순화를 지원합니다. 또한, 본 시스템은 다양한 사용자 그룹에 적용 가능하며, 다국어 지원을 통해 보편적인 사용성을 갖추고 있습니다.

- **Performance Highlights**: 이번 연구의 결과, 참가자들은 제안된 시스템의 80% 이상을 이해할 수 있었으며, 직관적인 학습 곡선을 제공하였습니다. 이를 통해 저학력 인구가 디지털 커뮤니케이션에 잘 적응할 수 있도록 돕는 강력한 도구가 될 것으로 기대됩니다. 또한 이 시스템은 지적 장애인, 다국어 팀 및 읽기 어려움이 있는 아동과 같은 다양한 사용자 집단에도 적용 가능성이 있습니다.



### Rethinking LLM Evaluation: Can We Evaluate LLMs with 200x Less Data? (https://arxiv.org/abs/2510.10457)
Comments:
          18 pages, 5 figures

- **What's New**: 이 논문에서는 다양한 모델 능력의 포괄적인 평가에 대한 수요 증가에 맞추어 벤치마크(Benchmark) 데이터셋의 크기가 급격히 증가하고 있음을 다룹니다. 기존의 방법들에서 얻어진 성능 예측의 일관성을 보장함과 동시에 예측 정확도를 유지하기 위한 체계적인 프레임워크가 절실하다고 강조합니다. 저자들은 벤치마크 압축을 최적화 문제로 정의하고, 이를 위한 새로운 방법인 EssenceBench를 제안합니다.

- **Technical Details**: EssenceBench는 본질적으로 그리드 검색을 통한 변형된 유전 알고리즘(Genetic Algorithm, GA)을 활용하여 평가 점수를 압축하는 프레임워크로 설계되었습니다. 이 방법은 초소형 샘플 집합을 지속적으로 추적하며, 샘플 간의 중복성과 성능 변동을 정량화하여 불필요한 샘플을 제거하는 과정을 포함합니다. 이를 통해 전체 데이터셋의 성능을 충실히 재구성하는 것을 목표로 하며, 효율적인 탐색 메커니즘으로 샘플의 속성을 활용합니다.

- **Performance Highlights**: 실험 결과에 따르면, EssenceBench는 HellaSwag 벤치마크에서 10K 샘플을 사용하여 25배 적은 샘플로 모델의 순위를 5% 이내에서 유지할 수 있음을 입증하였습니다. 이러한 결과는 전체 모델 순위의 유지를 의미하며, 200배 적은 샘플에서도 95%의 순위 보존률을 기록했습니다. 이로 인해 LLM(대형 언어 모델) 평가의 효율성을 크게 향상시킬 수 있는 가능성을 보여줍니다.



### End-to-end Speech Recognition with similar length speech and tex (https://arxiv.org/abs/2510.10453)
- **What's New**: 이 연구에서는 음성 인식에서 음성과 텍스트 길이를 일치시키기 위한 새로운 방법인 Time Independence Loss (TIL)과 Aligned Cross Entropy (AXE) Loss를 도입합니다. 기존의 Connectionist Temporal Classification (CTC) 기반 방법은 음성이 텍스트와 유사한 길이로 다운샘플링될 때 적절한 정렬을 제공하지 못했습니다. 연구자들은 키프레임 기반 다운샘플링(KFDS) 기술을 통해 주요 프레임을 보존하면서 텍스트 길이에 비슷한 길이로 음성을 줄이는데 집중하고 있습니다.

- **Technical Details**: 제안된 엔드 투 엔드 자동 음성 인식(ASR) 모델은 Conformer 기반 인코더 층과 Transformer 기반 디코더 층으로 구성됩니다. 이 모델은 음성-텍스트 정렬 정보를 개선하기 위해 CTC 손실 함수를 포함하며, 최종 목표 손실 함수는 세 가지 매개변수 α_{0}, α_{1}, α_{2}을 활용합니다. 새로운 Length Similarity Loss (LSL)는 두 가지 방법으로 구현되며, 하나는 TIL을 통해 시간 정보를 제거하고, 다른 하나는 AXE 손실을 통해 입력과 출력을 정렬합니다.

- **Performance Highlights**: 실험 결과 AISHELL-1 및 AISHELL-2 데이터셋의 서브셋에서 제안된 방법이 이전 연구 결과보다 우수한 성능을 보이는 것으로 나타났습니다. 프레임 수를 최소 86% 줄이는 동시에 문자 오류율(CER)을 기존과 유사한 수준으로 유지하며, 복잡성을 크게 줄이는 성과를 거두었습니다. 이러한 성과는 CTC 손실을 활용한 기존 방법과 비교할 때도 유사한 결과를 보여줍니다.



### Steering Over-refusals Towards Safety in Retrieval Augmented Generation (https://arxiv.org/abs/2510.10452)
Comments:
          Preprint

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)의 안전 정렬(safety alignment) 문제가 과도한 거부(over-refusals)로 이어지는 현상을 분석한 것입니다. 특히, retrieval-augmented generation (RAG) 설정에서 쿼리의 의도와 검색된 контекст의 특성이 거부 행위에 어떻게 영향을 미치는지에 대해 다루었습니다. RagRefuse라는 새로운 벤치마크를 구축하여 의도에 적합한 파라메트릭 모델의 평가를 진행하며, 안전한 출력 영역으로 임베딩을 조정하는 SafeRAG-Steering 기법을 제시합니다.

- **Technical Details**: RAG 설정에서의 과도한 거부 현상은 검색된 контекст의 유해성, 길이, 도메인에 따라 다르게 나타납니다. 연구진은 의료, 화학, 사이버 보안 등 다양한 도메인을 포함한 데이터 세트를 구축하였고, 이 데이터 세트를 활용하여 등급을 매깁니다. 모델의 숨겨진 상태를 분석하여, 안전한 응답을 위한 목표 영역(target region)과 과도한 거부를 일으키는 영역을 구분합니다.

- **Performance Highlights**: 연구 결과, 쿼리와 контекст의 조합이 LLM의 거부 빈도에 미치는 영향을 확인했습니다. 특정 도메인, 예를 들어 화학 관련 쿼리는 거부 빈도가 더욱 높았으며, 반면 사이버 보안 쿼리는 그 영향이 적었습니다. 또한, 유해 컨텍스트의 비율이 증가함에 따라 거부 빈도 역시 증가하는 경향이 있음을 알 수 있었습니다. SafeRAG-Steering 방법론을 통해 과도한 거부를 줄이면서도 정당한 거부를 보존할 수 있는 가능성을 제시하였습니다.



### RECON: Reasoning with Condensation for Efficient Retrieval-Augmented Generation (https://arxiv.org/abs/2510.10448)
- **What's New**: RECON(Reasoning with CONdensation)은 RL 기반의 RAG 시스템에서 비효율적인 컨텍스트 관리를 해결하기 위해 명시적 요약 모듈을 통합한 새로운 프레임워크입니다. 이 시스템은 QA 데이터셋에서의 관련성 사전 학습과 LLM으로부터의 다각적 증류 과정을 통해 훈련됩니다. 이를 통해 RAG의 총 컨텍스트 길이를 35% 단축시키고, 훈련 속도와 추론 지연 시간을 개선하며, 다운스트림 QA 기준에 대한 성능을 향상시킵니다.

- **Technical Details**: RECON의 핵심은 검색 단계 이후 요약을 적용하여 검색된 문서를 간결하고 인간이 읽을 수 있는 형태로 압축하는 것입니다. 명시적 요약기는 두 단계로 훈련되며, 첫 번째 단계에서는 유용한 문서를 식별하고, 두 번째 단계에서는 GPT-4o-mini로부터 고품질 요약 선호도를 학습합니다. 이러한 접근 방식은 정책 모델이 압축된 증거를 매 단계에서 제공받음으로써 토큰 소모를 줄이고, 투명한 추론 체인을 유지하도록 돕습니다.

- **Performance Highlights**: RECON은 3B 모델의 평균 EM 점수를 14.5% 향상시키고, 7B 모델에서는 3.0% 증가시킵니다. 특히, 다단계 QA에서 강력한 성능을 보이며, 훈련 속도를 5.2%, 추론 지연 시간을 30.9% 개선합니다. 이러한 결과는 활동적인 컨텍스트 압축이 실제적인 RL 보강 RAG 시스템을 구축하는 데 필수적임을 시사합니다.



### Do Audio LLMs Really LISTEN, or Just Transcribe? Measuring Lexical vs. Acoustic Emotion Cues Relianc (https://arxiv.org/abs/2510.10444)
- **What's New**: 이번 연구에서는 LISTEN(Lexical vs. Acoustic Speech Test for Emotion in Narratives)이라는 새로운 벤치마크를 소개하며, 감정 이해에서 레키컬(lexical) 의존성을 음향(acoustic) 민감성과 분리하는 것을 목표로 하고 있습니다. 현재의 대형 오디오 언어 모델(LALMs)이 실제로 음향 정보를 어떻게 처리하는지 불확실성을 해소하고자 하며, 이는 감정 인식의 한계를 명확히 드러냅니다. 특히, LISTEN은 감정 이해에서 LALMs의 진정한 듣기 능력을 평가하는 기준을 제공합니다.

- **Technical Details**: LISTEN은 네 가지 제어된 조건을 갖춘 평가 프레임워크로 구성되며, 이는 (i) Neutral-Text(중립 텍스트), (ii) Emotion-Matched(감정 일치), (iii) Emotion-Mismatched(감정 불일치), 그리고 (iv) Paralinguistic(부언어적)입니다. 각 조건 내에서 텍스트, 오디오, 텍스트+오디오 모드를 설정하여 음향 정보 처리 능력을 평가합니다. 이러한 설계는 LALMs가 레키컬 정보보다 음향 신호를 진정으로 수용하는지를 조사할 수 있도록 돕습니다.

- **Performance Highlights**: 연구 결과, 여러 최첨단 LALMs의 성능을 비교한 결과, 레키컬 우위가 뚜렷하게 나타났습니다. LALMs는 레키컬 신호가 중립적일 때 '중립' 예측을 하고, 단서 정렬 시 제한적인 성과를 보이며, 단서가 상충할 경우 distinct 감정 분류에 실패합니다. 특히 이러한 결과는 기존의 LALMs가 '들리는' 것이 아니라 주로 '옮기는' 방식으로 작동하며, 음향 신호를 충분히 활용하지 않음을 시사합니다.



### LONGQAEVAL: Designing Reliable Evaluations of Long-Form Clinical QA under Resource Constraints (https://arxiv.org/abs/2510.10415)
- **What's New**: 이 논문에서는 LongQAEval이라는 새로운 평가 프레임워크를 도입하여 제한된 자원과 높은 전문성을 요구하는 환경에서의 오랜 형식의 임상 질문 응답(QA) 시스템 평가를 용이하게 하고자 하였습니다. 300개의 실제 환자 질문에 대한 의사와 LLM의 답변을 바탕으로 정확성(correctness), 관련성(relevance), 안전성(safety) 차원에서 평가를 비교하였고, 세부적인 문장 수준 평가가 더 큰 합의(inter-annotator agreement, IAA)를 이끌어내는 결과를 얻었습니다.

- **Technical Details**: LongQAEval 프레임워크는 답변의 정확성, 관련성 및 안전성을 평가하는 기준을 명확히 정의합니다. 평가자는 전체 답변을 평가하는 coarse 디자인과 개별 문장을 평가하는 fine-grained 디자인 두 가지 방법으로 답변을 평가하도록 지시받았습니다. 연구 결과, 세부적인 문장 수준의 주석은 사실 기반 정확성에서 IAA를 향상시키고, 전반적인 문맥 기반 평가에서는 coarse 주석이 더 유리하며, 부분 평가를 통해 비용을 줄일 수 있다는 것을 발견하였습니다.

- **Performance Highlights**: GPT-4와 Llama-3.1-Instruct-405B 모델이 임상 질문에 대해 생리학적 정확성과 관련성에서 의사와 유사한 성과를 보였습니다. 특히, 세부적인 평가 방법이 신뢰성을 높이는 데 기여하며, LLM이 제공하는 답변이 길이에 따른 편향을 상쇄하는 데 효과적임을 보여주었습니다. 이러한 결과는 연구자들에게 평가 차원에 맞는 주석 디자인을 맞추어야 한다는 중요한 시사점을 제공합니다.



### STEAM: A Semantic-Level Knowledge Editing Framework for Large Language Models (https://arxiv.org/abs/2510.10398)
Comments:
          Accepted to EMNLP 2025 (Findings)

- **What's New**: 이 논문에서는 기존의 지식 편집 방법의 한계를 극복하기 위해 새로운 세멘틱 지식 편집 프레임워크인 	extsc{Steam}을 제안합니다. 최종 목표는 편집된 지식을 모델의 지식 구조에 자연스럽게 통합하는 것입니다. 기존 방식이 주로 토큰 수준의 가능성 최적화에 치중하는 반면, 	extsc{Steam}은 의미론적 일관성을 강화하는 데 중점을 두었습니다.

- **Technical Details**: 	extsc{Steam} 프레임워크는 수정을 위한 두 가지 주요 구성 요소를 포함합니다: (1) Latent Positioning, 이는 편집된 지식에 대한 의미적 앵커를 식별하고, (2) Latent-Level Alignment, 이는 편집된 사실의 내부 표현을 이러한 앵커로 안내하는 역할을 합니다. 이러한 접근 방식을 통해 모델의 레이턴트 공간에서 의미론적 통합을 촉진하여, 수정된 사실에 대한 추론 능력을 개선합니다.

- **Performance Highlights**: 실험 결과, 	extsc{Steam}은 모델이 편집된 지식으로 더 잘 추론할 수 있게 하였고, 전반적인 의미론적 일관성을 향상시켰습니다. 다양한 기준선과 편집 환경에서 이러한 개선 사항은 일관되게 나타났으며, 신뢰할 수 있는 지식 편집을 위해서는 의미 수준의 통합이 중요함을 드러냈습니다.



### AssoMem: Scalable Memory QA with Multi-Signal Associative Retrieva (https://arxiv.org/abs/2510.10397)
- **What's New**: 본 논문은 AssoMem이라는 새로운 메모리 강화 AI 프레임워크를 제안합니다. 이 프레임워크는 대화 발화를 자동으로 추출된 단서에 연결하는 연관 메모리 그래프를 구성하여, 대화 맥락에 대한 풍부한 조직적 시각을 제공합니다. AssoMem은 또한 relevance, importance 및 temporal alignment와 같은 다차원 검색 신호를 통합하여 메모리 검색의 정확성을 향상시킵니다.

- **Technical Details**: AssoMem은 연관 메모리 그래프(associative memory graph)를 활용하여 사용자의 기억을 해석하고 연결짓는 구조를 갖추고 있습니다. 이 구조는 메모리 발화와 이에 연결된 단서 사이의 의미적 관계를 포착하며, 이를 통해 메모리를 효율적으로 검색하고 중요도에 따라 순위를 매길 수 있습니다. 또한, mutual information (MI)에 기반한 융합 전략을 사용하여 쿼리 의도에 따라 다차원 신호를 동적으로 조정합니다.

- **Performance Highlights**: 세 가지 벤치마크와 MeetingQA라는 새로운 데이터셋을 통해 AssoMem은 최신 SOTA 기준선보다 평균 24.93% 높은 성능을 보였습니다. 이러한 실험 결과는 AssoMem이 대규모 메모리 저장소에서 컨텍스트 인식 메모리 회상 능력이 우수함을 증명합니다.



### RefusalBench: Generative Evaluation of Selective Refusal in Grounded Language Models (https://arxiv.org/abs/2510.10390)
- **What's New**: 이 연구에서는 Retrieval-Augmented Generation(RAG) 시스템에서 언어 모델이 잘못된 맥락에 따라 selectively refuse(선택적 거부)를 수행하는 능력이 얼마나 중요한지를 보여줍니다. 연구진은 기존 모델들이 이 기능에서 50% 미만의 정확도를 기록하고, 잘못된 정보에 기반하여 답변을 거부하거나 자신이 없는 답변을 내는 문제를 밝혀냈습니다. 또한, 단순한 정적 벤치마크(static benchmarks)가 이러한 성능을 평가하는 데 한계를 가지고 있음을 강조하며, RefusalBench라는 새로운 평가 방법론을 소개합니다.

- **Technical Details**: RefusalBench는 언어적 교란을 통해 진단 테스트 케이스를 생성하는 프로그램 수립 방법론을 기반으로 하고 있습니다. 이 시스템은 정보 불확실성의 여섯 가지 범주에서 176개의 교란 전략을 활용하여 답변 가능한 질문을 답변 불가능한 질문으로 변화시킵니다. 이러한 평가 방법론은 고유성에 대한 감도를 세밀하게 진단할 수 있으며, 멀티 모델 생성-검증 파이프라인을 통해 정답의 품질을 보장합니다.

- **Performance Highlights**: 30개 이상의 모델을 평가한 결과, 선택적 거부 능력에서 심각한 차이가 발견되었습니다. 연구진은 이 능력이 훈련이 가능하고 조정에 민감한 특성을 지니고 있다는 것을 밝혀내어 모델 개선의 길을 제시하였습니다. 또한, RefusalBench-NQ(단일 문서) 및 RefusalBench-GaRAGe(다중 문서)라는 두 가지 벤치마크를 제공하며, 이러한 새로운 평가 프레임워크의 필요성을 강조합니다.



### ASC analyzer: A Python package for measuring argument structure construction usage in English texts (https://arxiv.org/abs/2510.10384)
Comments:
          Accepted to the 2nd Workshop on Construction Grammars and NLP (CxGs+NLP)

- **What's New**: 이 논문에서는 제2언어(L2) 능력을 분석하기 위한 새로운 도구인 ASC analyzer를 소개합니다. 이 도구는 공개적으로 사용 가능한 Python 패키지로, argument structure constructions (ASCs)를 자동으로 태깅하고 다양한 지표를 계산합니다. ASCs는 언어의 구성 성분 간의 관계를 모델링하는 데 중요한 역할을 하며, L2 학습자의 쓰기 성과와의 관계를 탐색하는 데 유용합니다.

- **Technical Details**: ASC analyzer는 RoBERTa 기반 ASC 태거를 활용하여 ASCs를 분석하고, 해당 태그를 바탕으로 50개의 지표를 계산합니다. 이 지표들은 ASCs의 다양성(diversity), 비율(proportion), 빈도(frequency), 및 ASC-verb lemmma의 연관 강도(association strength)를 포함합니다. 이 도구의 기능은 6,482개의 영어 학습자 에세이를 분석하여 L2 영어 쓰기 능력과 ASC 기반 지표 간의 관계를 탐구하는 데에도 활용되었습니다.

- **Performance Highlights**: 실험 결과, ASC analyzer는 L2 필기 및 구술 능력 평가에서 높은 F1 점수를 기록했습니다. ASC 사용의 증가는 L2 구술 능력의 변동성을 설명하는 데 중요한 요소로 나타났습니다. 이러한 결과는 ASC 기반 지표가 L2 성과를 이해하고 예측하는 데 유용하다는 것을 보여줍니다.



### End-to-end Automatic Speech Recognition and Speech Translation: Integration of Speech Foundational Models and LLMs (https://arxiv.org/abs/2510.10329)
- **What's New**: 이번 연구는 최근의 음성 인식 기술이 적용된 end-to-end Speech Translation (ST) 방법론을 탐구합니다. 기존의 cascade 방식과 비교하여, ASR (Automatic Speech Recognition)과 ST를 동시에 수행할 수 있는 새로운 아키텍처를 제안합니다. 실험 결과, 이 모델은 SeamlessM4T와 비교해도 더 나은 번역 결과를 달성할 수 있으며, Whisper와 NLLB 기반의 cascaded 시스템과 비교했을 때도 비슷한 성과를 보여주었습니다.

- **Technical Details**: 연구진은 HuBERT와 Whisper를 사용하여 음성 데이터를 고품질로 표현하는 음성 인코더를 구성하였습니다. 각 음성 신호에 대해 대응하는 텍스트와 번역된 텍스트를 생성하기 위해, 음성을 인코딩한 후 LLM(Large Language Models)과 결합하여 최종 출력을 생성합니다. 또한, Quantized Low-Rank Adaptation (QLoRA) 기법으로 모델을 파인튜닝하여 성능을 극대화했습니다.

- **Performance Highlights**: 우수한 번역 성과를 위해 다양한 표준 지표를 사용하여 평가하였으며, 특히 BLEU와 COMET 지표에서 높은 점수를 기록하였습니다. Gemma 2 9B 모델이 최고의 성능을 보였으나 Whisper 모델보다 뒤쳐진 결과를 나타내었습니다. 전체적으로 연구가 제안한 방법은 cascade 시스템에 비해 유의미한 성능 향상을 이루었음을 보여줍니다.



### Are LLMs Empathetic to All? Investigating the Influence of Multi-Demographic Personas on a Model's Empathy (https://arxiv.org/abs/2510.10328)
Comments:
          9 pages, 4 figures, 4 tables, EMNLP 2025 Findings

- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)의 공감 능력을 탐구하는 새로운 프레임워크를 제안합니다. 연구에서는 연령, 문화 및 성별의 교차적 특성을 가진 315개의 고유한 사용자 페르소나를 기반으로 하여 LLM이 감정적으로 공감하는 정도를 평가합니다. 흥미롭게도, 연구 결과는 여러 속성을 동시에 추가할 경우 예상되는 공감 패턴이 약해지거나 역전될 수 있음을 보였습니다.

- **Technical Details**: 연구에서 사용된 방법론은 33개의 인구통계학적 속성과 44개의 LLM 계열을 포함하는 다차원적 분석에 기초합니다. LLM은 사용자 페르소나와 그들의 감정 경험을 입력받아 감정을 예측하고 해당 페르소나에 맞춘 반응을 생성하는 방식으로 작동합니다. 이 데이터는 ISEAR 데이터셋을 기반으로 하며, 다양한 페르소나의 개인적인 감정 경험을 포함합니다.

- **Performance Highlights**: 연구 결과, LLM의 공감 능력은 인구통계적 차원에 따라 상당한 변화를 보이며, 이러한 변화는 문헌에 기록된 고정관념을 반영합니다. 특정 그룹, 특히 유교 문화적 배경을 가진 집단에서 LLM의 공감 반응에 상당한 불일치가 나타났습니다. 질적 통찰을 추가로 제공하여 서로 다른 인구통계적 그룹에서의 모델 행동 패턴을 밝혀내며, 공감을 고려한 LLM 설계의 중요성을 강조합니다.



### MatryoshkaThinking: Recursive Test-Time Scaling Enables Efficient Reasoning (https://arxiv.org/abs/2510.10293)
- **What's New**: 본 연구에서는 MatryoshkaThinking이라는 새로운 방법을 제안합니다. 이 방법은 언어 모델의 성능 향상을 위해 테스트 중 추가 계산 리소스를 효율적으로 활용합니다. MatryoshkaThinking은 DeepConf보다 4%의 계산만으로도 AIME2025에서 99.79의 점수를 달성합니다.

- **Technical Details**: MatryoshkaThinking은 모델의 본질적인 추론(reasoning), 검증(verification), 요약(summarization) 기능을 재귀적으로 활용하여 계산 비용을 크게 줄입니다. 이 접근법은 올바른 솔루션의 보존을 향상시키고 Pass@k와 Pass@1 간의 불일치를 줄이는 데 기여합니다. 여러 오픈 소스 모델과 멀티 모달(reasoning) 벤치마크를 통해 종합적인 평가가 진행되었습니다.

- **Performance Highlights**: MatryoshkaThinking의 성능은 기존 방법보다 우수하며, 99.79라는 높은 점수를 자랑합니다. 이 방법은 테스트 시간 추론(strategy)에서의 효율성 및 확장 가능성을 위한 새로운 통찰을 제공합니다. 전반적으로 이 연구는 고급 언어 모델을 위한 혁신적인 테스트 시간 전략을 제시합니다.



### On the Entity-Level Alignment in Crosslingual Consistency (https://arxiv.org/abs/2510.10280)
Comments:
          preprint

- **What's New**: 이번 연구는 다국어 큰 언어 모델(LLMs)의 사실 기억에서 발생하는 불일치의 원인인 엔티티 정렬(entity alignment) 실패에 주목합니다. 연구자들은 주제와 객체 엔티티가 각 언어에서 공유된 개념 공간에 매핑될 때 일관성 유지가 용이하다고 가정합니다. 모델 내에서의 올바른 엔티티 정렬이 두 언어 간의 사실 기억에 중요한 역할을 하며, 이를 통해 일관성 있는 기억을 개선할 수 있는 방법을 제안합니다.

- **Technical Details**: 연구팀은 17개의 다양한 언어로 이루어진 데이터셋인 KLAR를 사용해, 엔티티 정렬 품질을 측정하기 위한 번역 작업을 설계했습니다. 그러면서 두 언어 간 실질적인 정렬이 주제와 객체 엔티티의 정렬에 의해 크게 영향을 받음을 발견하였습니다. 이는 개념 모델이 제안하는 두 단계, 즉 언어별 입력을 공유 개념 공간으로 매핑하고, 공유된 잠재 표현을 목표 언어의 정확한 표면 형태로 다시 투사하는 과정을 통해 실현됩니다.

- **Performance Highlights**: 제안된 두 가지 방법(모델 잘못된 기억을 줄이기 위해 영어 번역 포함)인 SubSub 및 SubInj는 특히 영어에 중점을 둔 모델들에서 사실 기억 정확성과 일관성을 상당히 향상시켰습니다. 이러한 제도는 엔티티 표현의 정렬을 강화시켜, 모델의 언어 중심 처리 피벗을 통해 일관된 사실 기억을 촉진합니다. 궁극적으로, 모델 내에서 수행된 이러한 세부 분석은 다국어 사실 예측을 개선하기 위한 실용적인 전략을 제시합니다.



### Backdoor Collapse: Eliminating Unknown Threats via Known Backdoor Aggregation in Language Models (https://arxiv.org/abs/2510.10265)
- **What's New**: 이 논문은 기존의 방어 시스템이 각종 트리거 설정에 대한 비현실적인 가정을 의존하는 문제를 해결하기 위해 Locphylax라는 새로운 방어 프레임워크를 제안합니다. 이 시스템은 알려진 백도어를 주입하는 방식으로, 기존의 알려지지 않은 백도어와 신규 주입된 백도어가 하나의 표현 공간에서 집합적으로 존재하게 되는 현상(aggregation phenomenon)을 활용합니다. 이를 통해 방어 과정에서 사전 지식 없이도 잘못된 출력을 복원할 수 있게 됩니다.

- **Technical Details**: Locphylax의 구조는 두 단계로 나뉩니다. 첫째, 알려진 트리거를 주입하여 백도어 표현을 집계하고, 둘째, 회복 미세 조정(fine-tuning)을 수행하여 원래의 무해한 출력으로 되돌립니다. 여러 모델 구조에 대해 광범위한 실험을 수행한 결과, Locphylax는 공격 성공률(Attack Success Rate)을 평균 4.41%로 낮추며 기존 방법들보다 28.1%에서 69.3%의 성능 향상을 보여주었습니다.

- **Performance Highlights**: Locphylax는 백도어 공격에 대해 뛰어난 방어 성능을 입증했습니다. 청정 정확도(clean accuracy)와 유용성(utility)은 원래 모델에 비해 0.5% 이내로 유지되었으며, 다양한 백도어 유형과 주입 방식에서도 일반화 가능한 방어 성능을 가지고 있음을 확인하였습니다. 근본적으로, 이 연구는 백도어를 가진 LLM에 대해 더 실용적이고 강력한 방어 방법을 제시합니다.



### Audit-of-Understanding: Posterior-Constrained Inference for Mathematical Reasoning in Language Models (https://arxiv.org/abs/2510.10252)
- **What's New**: 이 논문에서는 LLMs가 생성한 합리적인 추론이 종종 지지되지 않는 가정을 기반으로 하여 발생하는 망상(hallucination) 문제를 다룹니다. 기존의 연구는 사실적인 망상(factual hallucination) 문제에 초점을 맞추고 있으며, 사후 검증(post-hoc verification)에 의존하였습니다. 새로운 접근법인 Audit-of-Understanding (AoU)을 제안하며, 이는 세 가지 주요 단계를 통해 검증된 전제로 추론을 제한하는 방법을 제시합니다.

- **Technical Details**: AoU는 쿼리를 후보 가정으로 분해하고, 그 지원 여부를 감사(audit)한 후, 검증된 부분 집합에만 의존하여 추론을 진행합니다. 이는 posterior-constrained inference로 공식화할 수 있으며, 구체적으로는 선택적 예측(selective prediction)과 거부 학습(rejection learning) 개념과 연결됩니다. 이러한 접근은 추론 과정에서 지원되지 않는 가정을 제거함으로써 정당성(faithfulness)과 위험 제어(risk-control) 보장을 제공합니다.

- **Performance Highlights**: 실험적으로 AoU는 GSM8K, MultiArith, SVAMP와 같은 수학적 추론 벤치마크에서 정확도와 신뢰성을 모두 향상시켰습니다. GSM8K에서는 +30%, MultiArith에서는 +45%, SVAMP에서는 일관된 +20-28% 향상을 달성했습니다. AoU는 기존의 Chain-of-Thought, Self-Consistency, CoT-Decoding 방법보다 훨씬 더 나은 성과를 보였습니다.



### ImCoref-CeS: An Improved Lightweight Pipeline for Coreference Resolution with LLM-based Checker-Splitter Refinemen (https://arxiv.org/abs/2510.10241)
- **What's New**: 이번 연구에서는 Coreference Resolution (CR) 문제를 해결하기 위해 새로운 프레임워크인 ImCoref-CeS를 제안합니다. 기존의 감독 학습 방법과 대규모 언어 모델(LLM) 간의 강점을 통합하여 성능을 극대화하는 것을 목표로 합니다. 특히, 경량 브리징 모듈과 하이브리드 멘션 정규화를 통해 긴 텍스트의 인코딩 능력을 한층 더 향상시켰습니다.

- **Technical Details**: ImCoref-CeS 프레임워크는 기존의 detect-then-cluster 파이프라인을 확장하여, 경량 브리징 모듈(LBM)과 바이아핀 점수기(biaffine scorer)를 도입합니다. 이는 멘션 탐지 단계에서 LLM을 검증기(Checker)와 클러스터링 분할기(Splitter) 역할로 활용하여, 불완전한 멘션을 걸러내고 오류가 있는 클러스터를 수정하는 작업을 수행합니다. 이러한 접근 방식을 통해 CR의 정확도를 높이고, 컴퓨팅 자원을 효율적으로 사용합니다.

- **Performance Highlights**: 다양한 CR 벤치마크에서 수행된 실험 결과, ImCoref는 기존의 최첨단(supervised neural methods) 방법들보다 지속적으로 성능이 향상되었습니다. ImCoref-CeS 프레임워크는 이 성과를 뛰어넘어 보다 높은 수준의 성능을 가능하게 하여, 기존 방법의 효율성과 정확도를 새로운 차원으로 끌어올립니다.



### Text2Token: Unsupervised Text Representation Learning with Token Target Prediction (https://arxiv.org/abs/2510.10224)
- **What's New**: 본 논문에서는 Unsupervised Text Representation Learning (TRL) 분야에서 새로운 프레임워크인 Text2Token을 제안합니다. 이 프레임워크는 텍스트의 key token을 생성하는 작업을 통해 고품질의 텍스트 표현을 학습하며, 기존의 discriminative (구별적) 학습 방식과는 다른 접근 방식을 취합니다. 이 연구는 LLM(대형 언어 모델)을 활용하여 키 토큰의 통계적 특성을 분석하고, 이러한 통찰을 기반으로 텍스트 representation을 향상시키려 합니다.

- **Technical Details**: Text2Token은 Kullback-Leibler divergence (KL-divergence) 손실함수를 통해 목표 토큰 분포(target token distribution)를 생성하는 비지도 학습 프레임워크입니다. 주요 두 가지 토큰 범주, 즉 의미 있는 텍스트의 토큰과 텍스트를 넘어선 의미론적 유도 토큰을 활용하여 데이터를 기반으로 하거나(Language Models Prior를 이용하여) 토큰 대상을 생성합니다. 실험은 MTEB v2 벤치마크에서 수행되었고, Text2Token은 기존의 최고 성능을 가진 LLM2Vec보다 우수한 결과를 보였습니다.

- **Performance Highlights**: 텍스트 대표화 (representation)와 어휘(vocabulary) 공간이 훈련 과정에서 함께 최적화된다는 발견이 있었습니다. Text2Token은 여러 과제에서 LLM2Vec보다 평균 점수를 크게 초과하는 성과를 가져오며, 높은 성능을 입증하였습니다. 이 논문은 각 구성 요소의 역할과 하이퍼파라미터의 영향을 이해하기 위한 분석 실험을 수행하였습니다.



### You only need 4 extra tokens: Synergistic Test-time Adaptation for LLMs (https://arxiv.org/abs/2510.10223)
Comments:
          Under Review

- **What's New**: 이 논문은 분야별 (domain-specific) 문제에 대한 라벨이 없는 테스트 시간 적응(test-time adaptation) 방식을 다룹니다. SyTTA(Synergistic Test-time Adaptation)라는 새로운 프레임워크를 제안하며, 이는 추가적인 감독 없이 실시간으로 모델을 조정할 수 있도록 합니다. 입력측의 perplexity와 출력측의 predictive entropy라는 두 가지 불확실성 신호를 결합하여 성능 저하를 완화합니다.

- **Technical Details**: SyTTA는 문제 설정에서 공통적으로 나타나는 불확실성을 다룹니다. 데이터 분포가 변경될 때, 입력에 대한 perplexity가 증가하며, 출력의 예측 엔트로피(predicitive entropy) 또한 높아집니다. 이를 통해 모델이 더 효과적으로 적응할 수 있도록 하며, 4-16개의 추가 토큰만으로도 빠르게 업데이트가 가능합니다. 모델은 Dynamic-Ref 모드와 Static-Ref 모드 중 하나를 선택하여 사용 가능합니다.

- **Performance Highlights**: SyTTA는 다양한 모델 아키텍처와 분야별 벤치마크에서 일관된 성능 향상을 보여주었습니다. 특히 농업 관련 질문 응답에서 Qwen-2.5-7B 모델이 Rouge-LSum을 120% 이상 향상시켰습니다. 이러한 결과는 라벨이 부족한 환경에서도 효과적인 테스트 시간 적응이 가능함을 시사합니다.



### Weed Out, Then Harvest: Dual Low-Rank Adaptation is an Effective Noisy Label Detector for Noise-Robust Learning (https://arxiv.org/abs/2510.10208)
Comments:
          ACL 2025

- **What's New**: 최근 연구에서 Parameter-efficient fine-tuning (PEFT) 기법이 대규모 언어 모델(LLMs)의 파라미터 조정을 통해 뛰어난 성능을 보였습니다. 하지만 실제 상황에서는 자주 노이즈가 포함된 레이블이 데이터에 존재합니다. 본 논문에서는 샘플 선택과 모델 훈련을 분리하여 이러한 악순환을 끊는 새로운 프레임워크인 Delora를 제안합니다.

- **Technical Details**: Delora는 두 개의 LoRA(clean LoRA 및 noisy LoRA)를 사용하여 노이즈 레이블 탐지기를 구성합니다. 클린 LoRA의 파라미터는 깨끗한 데이터를 기억하도록, 노이즈 LoRA는 잘못 레이블된 데이터를 기억하도록 설계되었습니다. 이 두 파라미터의 동적 정규화를 통해, 처음에는 깨끗한 샘플을 완전히 기억하고, 이후 훈련이 진행됨에 따라 노이즈 샘플을 점차 적절히 메모리합니다.

- **Performance Highlights**: 실험 결과는 Delora가 노이즈 레이블 탐지 및 텍스트 분류성능에서 기존 방법들보다 우수하다는 것을 보여줍니다. 특히, 고유한 샘플 선택 메커니즘을 통해 Delora는 기존 방식의 문제점인 악순환을 피하면서 더 효과적인 데이터를 활용할 수 있게 됩니다. 대규모 텍스트 분류 데이터 세트에서 다양한 노이즈 조건 하에서도 Delora의 성과가 두드러집니다.



### MedAgentAudit: Diagnosing and Quantifying Collaborative Failure Modes in Medical Multi-Agent Systems (https://arxiv.org/abs/2510.10185)
Comments:
          Code: this https URL

- **What's New**: 본 연구는 대형 언어 모델(LLM) 기반의 다중 agent 시스템이 의료 상담을 시뮬레이션하는 데 기여할 수 있는 가능성을 제시합니다. 그러나 이러한 시스템의 평가가 최종 답변의 정확도로 한정되는 문제점을 지적하며, 진단 결론이 신뢰할 수 있는 추론 경로를 통해 도출되었는지 여부가 중요하다고 강조합니다. 3,600개의 사례를 종합한 실증 연구를 통해 협력 실패 양상들을 규명하고, 투명하고 검증 가능한 추론 프로세스의 필요성을 제시합니다.

- **Technical Details**: 이 연구는 3,600개의 상호작용 로그를 바탕으로 다양한 다중 agent 시스템을 분석하여 협력 실패 모드를 분류합니다. 분석 과정에서, 역할 전문화, 잘못된 의견의 억압 등 여러 가지 실패 패턴이 드러났습니다. 품질에 기반한 증거 평가를 우회하거나 주요 정보 손실이 발생하는 등 다양한 결함들이 식별되었습니다. 연구진은 이러한 실패를 체계적으로 진단하기 위한 정량적 감사(framework)도 도입하였습니다.

- **Performance Highlights**: 이 연구는 단순한 정확도만으로는 의료 AI에 대한 신뢰를 구축할 수 없음을 보여줍니다. 협력적 결함의 전반적인 패턴을 제시함으로써, 의료 분야에서의 협업 AI의 투명성과 신뢰성을 확보하기 위한 기반을 마련하였습니다. 연구의 결과로, 거의 완벽한 합의율을 기록하여 코드화된 협력 실패 양상들이 신뢰할 수 있는 분석 도구가 되었음을 입증하였습니다.



### A Survey of Inductive Reasoning for Large Language Models (https://arxiv.org/abs/2510.10182)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)을 위한 유도 추론(inductive reasoning)의 첫 번째 포괄적인 조사를 제시합니다. 기존의 연구가 주로 연역적 추론(deductive reasoning)에 초점을 맞춘 반면, 본 연구에서는 유도 추론의 중요성을 강조하고 이를 개선하기 위한 방법을 세 가지 주요 영역으로 분류했습니다. 이러한 방법은 포스트 트레이닝(post-training), 테스트 시간 스케일링(test-time scaling), 데이터 증강(data augmentation)으로 나눌 수 있습니다.

- **Technical Details**: 유도 추론은 특정 관찰로부터 일반적인 결론을 도출하는 사유 방식으로, 이를 통해 LLM의 성능을 향상시키는 다양한 기법을 논의합니다. 이 논문에서는 LLM의 성능을 평가하기 위한 통합된 샌드박스 기반 평가 접근 방식과 관찰 커버리지(observation coverage) 메트릭을 도출했습니다. 또한 유도 능력의 수원(source of inductive ability)과 단순한 모델 아키텍처 및 데이터가 유도 작업에 어떻게 도움이 되는지를 분석합니다.

- **Performance Highlights**: 유도 추론은 자연어 처리(NLP)의 여러 하위 작업에서 성능 향상에 기여하며, 다양한 실제 시나리오에서 광범위하게 적용됩니다. 특히, 금융 예측, 자율 주행 및 대화형 건강 관리와 같은 분야에서 주목을 받고 있습니다. 이러한 연구 결과는 향후 LLM의 연구를 위한 튼튼한 기초를 제공합니다.



### Large Language Model Sourcing: A Survey (https://arxiv.org/abs/2510.10161)
Comments:
          31 pages

- **What's New**: 이번 연구는 대형 언어 모델 (LLMs)이 주관적 의사 결정 과정에 영향을 미치게 됨에 따라, 이에 따른 다양한 위험 요소들에 대한 체계적인 조사를 집중적으로 다룹니다. LLMs의 제품과 모델의 출처를 식별하는 방법을 제시하며, 이를 통해 그들의 투명성, 책임성 및 신뢰성을 높일 수 있는 길잡이를 제공합니다. 특히 저자는 새로운 복합적 출처 추적 체계와 두 가지 분류 체계를 제안하여, LLM의 콘텐츠 생성 전과 후에 걸친 전반적인 접근을 확립했습니다.

- **Technical Details**: 이 연구는 모델 관점(Model Sourcing)과 데이터 관점(Training Data Sourcing) 각각에서 LLM의 콘텐츠 출처를 추적하는 네 가지 차원으로 구성된 체계를 제안합니다. 이들 차원은 모델의 구조적 요소와 훈련 데이터의 출처를 종합적으로 고려하여 LLM의 출력물의 근원을 파악할 수 있도록 합니다. 저자는 사전 기반(prior-based)과 사후 기반(posterior-based) 체계를 통해 이러한 출처 추적 방법을 분류하여, 각 접근 방식의 장단점을 분석합니다.

- **Performance Highlights**: LLMs의 투명성과 책임성을 높이기 위해 제안된 출처 추적 체계는 실세계에서의 다양한 응용 프로그램에 대한 신뢰를 증진시킬 수 있습니다. 하수인들과의 관계를 명확히 하고, 훈련 데이터의 편견으로 인한 결과물에 대한 책임을 명확히 할 수 있는 기회를 제공합니다. 이러한 체계는 LLM의 성능을 높은 수준으로 유지하며, 본 연구를 통해 이러한 체계가 기존 단편적 접근에서 벗어나 총체적인 프레임워크로 설계되었음을 보여줍니다.



### BabyBabelLM: A Multilingual Benchmark of Developmentally Plausible Training Data (https://arxiv.org/abs/2510.10159)
- **What's New**: BabyBabelLM은 출생부터 모국어를 습득할 때까지 한 사람이 관찰하는 언어를 모델링한 다국어 데이터셋 컬렉션을 제시합니다. 이 데이터는 45개 언어에서 각각 1억 개의 영어 단어에 해당하는 내용을 포함하도록 개발되었으며, 다국어 사전 훈련 및 인지 모델링을 촉진할 수 있도록 구성되었습니다.

- **Technical Details**: 이 논문은 언어 모델링의 최신 연구 경향에 대한 비판으로 시작하며, 특히 모델 크기와 데이터 볼륨의 확장에서 비롯된 문제점을 지적합니다. BabyBabelLM은 45개 언어로 구성된 다국어 훈련 데이터셋을 만들기 위해 다양한 공공 데이터셋을 신중하게 선택하였고, 각 언어의 발달 가능성을 중시하여 데이터를 정리했습니다.

- **Performance Highlights**: 새롭게 발표된 45개의 언어에 대한 데이터셋은 연구 목적을 위한 라이센스를 가지고 있으며, 이후 데이터셋 확장을 위한 파이프라인도 제공합니다. 이 프로젝트는 모델 성능을 평가하는 평가 과제를 포함하며, 커뮤니티에 의해 확장 가능한 평가 도구를 포함하고 있습니다.



### BILLY: Steering Large Language Models via Merging Persona Vectors for Creative Generation (https://arxiv.org/abs/2510.10157)
- **What's New**: 본 연구에서는 BILLY(BlendIng persona vectors for Large Language model creativitY)라는 새로운 프레임워크를 제안하여 다중 대형 언어 모델 시스템의 한계를 극복하고자 합니다. BILLY는 각기 다른 페르소나 벡터를 단일 모델의 활성화 공간 내에서 추출 및 혼합함으로써 다각적인 관점과 전문성을 효과적으로 캡처합니다. 이를 통해 명시적인 다중 LLM 통신 없이 다양한 시각적 출력을 생성할 수 있습니다.

- **Technical Details**: BILLY는 여러 개의 독립적인 페르소나 벡터를 추출해 합성하여 모델의 생성 과정에서 이를 조정함으로써 동작합니다. 본 논문에서는 Chen et al. (2025)의 대조적 활성화 방법론을 기반으로 페르소나 벡터를 생성하는 과정을 상세히 설명하며, 두 가지 반대 집합의 모델 응답을 통해 각 페르소나의 벡터를 도출합니다. 이러한 벡터는 생성의 보조적 측면을 효과적으로 제어할 수 있도록 해줍니다.

- **Performance Highlights**: BILLY는 기존의 단일 모델 프롬프트 및 전통적인 다중 LLM 접근법을 초월하는 성능을 보여주며, 추론 시간과 계산 비용을 획기적으로 줄입니다. 실험 결과, BILLY는 높은 창의성 점수를 기록하면서도 각 페르소나의 기능적 차별화를 확인할 수 있었습니다. 최종적으로 BILLY는 높은 효율성을 유지하면서도 독창적이고 창의적인 출력 결과를 생성하는 데 기여합니다.



### DiffHeads: Differential Analysis and Inference-Time Masking of Bias Heads in Large Language Models (https://arxiv.org/abs/2510.10142)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 불공정성에 대한 체계적인 조사를 수행하고, 이를 해결하기 위한 경량 디바이싱(debiasing) 프레임워크인 DiffHeads를 제안합니다. 기존의 연구들은 편향된 출력이 언제 발생하는지를 탐색해왔으나, 그 생성 메커니즘에는 거의 통찰력을 제공하지 않았습니다. 연구진은 Direct-Answer(DA)와 Chain-of-Thought(CoT)이라는 두 가지 프롬프트 기법을 비교하여 DA가 LLM의 편향을 유도하고 CoT는 편향을 줄이는 결과를 이끌어낸다는 것을 보여주었습니다.

- **Technical Details**: 주요 기술적 기여 중 하나는 LLM의 특정 attention head가 DA prompting에서 어떻게 더 활성화되는지 측정하는 것입니다. 이를 위해 token-to-head 기여 점수를 정의하고, 각 token의 개별 attention head에 대한 영향을 추적했습니다. 연구진의 분석에 따르면, DA에서는 편향적인 output을 생성하는 특정 cluster의 bias heads가 활성화되지만, CoT에서는 이러한 heads가 대개 비활성 상태를 유지하게 됩니다.

- **Performance Highlights**: DiffHeads 프레임워크는 DA와 CoT 간의 차별적 활성화 분석을 통해 bias heads를 식별하고, 그러한 heads를 선택적으로 가리는 방식으로 작동합니다. 이 접근 방식은 DA에서 49.4%, CoT에서 40.3%의 불공정성을 감소시켰으며, 모델의 유용성에 해를 끼치지 않으면서 불공정성을 크게 줄였습니다. 이는 LLM의 공정성을 높이는 효과적인 방법으로 평가받고 있습니다.



### Hybrid OCR-LLM Framework for Enterprise-Scale Document Information Extraction Under Copy-heavy Task (https://arxiv.org/abs/2510.10138)
- **What's New**: 이 논문은 OCR 엔진과 대형 언어 모델(LLM)을 통합한 체계적인 프레임워크를 제안하여 방대한 양의 내용으로 복잡한 문서에서 정보를 추출하는 새로운 방법론을 소개합니다. 반복적인 문서 추출 작업에서 정확성과 효율성을 최적화하는 것을 목표로 하며, 아무리 유사한 문서라도 문서별 특성을 활용하여 전략을 선택하는 차별화된 접근법을 취합니다. 이 연구는 특정 응용 프로그램인 신원 문서 추출을 넘어, 반복적인 작업의 본질을 인지하여 최적화 기회로 전환할 수 있음을 제시합니다.

- **Technical Details**: 본 연구는 25가지 구성 방법을 통해 세 가지 추출 패러다임(직접, 대체, 테이블 기반)을 평가했습니다. 신원 문서의 네 가지 형식(예: PNG, DOCX, XLSX, PDF)에 걸쳐 적응형 테이블 기반 추출 방법을 활용하여 우수한 결과를 달성했습니다. 특히, PaddleOCR과 통합 시 구조화된 문서에 대해 F1=1.0의 정확도와 0.97초의 처리 시간을 구현하였으며, 어려운 이미지 입력의 경우에도 F1=0.997의 정확도와 0.6초의 처리 시간을 유지합니다.

- **Performance Highlights**: 이 연구는 기존의 멀티모달 방법과 비교해 54배 성능 향상을 이룩했으며, 포맷 인식 라우팅을 적용하여 다양한 문서 스트림을 생산 규모로 처리할 수 있도록 돕습니다. 경량화된 프레임워크는 기업 환경에서 신속한 추출 시스템의 배치를 위한 실질적인 경로를 제시하며, 반복적인 문서 작업에서 고성능을 발휘할 수 있도록 합니다. 이 연구는 또한 문서 특성에 부합하는 방법 선택 전략을 통해 완벽한 F1 점수(1.000)와 평균 0.97초의 지연 시간을 달성합니다.



### LinearRAG: Linear Graph Retrieval Augmented Generation on Large-scale Corpora (https://arxiv.org/abs/2510.10114)
- **What's New**: 이 논문에서는 LinearRAG(Linear Graph-based Retrieval-Augmented Generation)를 제안하여 기존 GraphRAG 시스템의 하드웨어 유지비용 및 불안정한 관계 추출 문제를 해결합니다. LinearRAG는 관계 없는 계층 그래프인 Tri-Graph를 구성하여 빠르고 효율적인 지식 그래프 구축을 가능하게 합니다. 이 방식은 많은 토큰 소모 없이 선형적으로 확대될 수 있으며, 원본 문서 인덱싱에 경제적이고 신뢰성 있는 방법을 제공합니다.

- **Technical Details**: LinearRAG는 가벼운 엔티티 추출 및 의미론적 연결만을 사용하여 관계 없는 계층적 그래프를 구축합니다. 이론적으로, 두 단계의 검색 전략을 채택하며, 첫 번째 단계는 지역 의미적 연결(local semantic bridging)을 통해 관련 엔티티를 활성화하고, 두 번째 단계는 글로벌 중요성 집계(global importance aggregation)를 통해 패시지 검색을 수행합니다. 이 모듈들은 복잡한 쿼리에 대한 확장 가능하고, 정확하며, 잡음에 강한 검색을 가능하게 만듭니다.

- **Performance Highlights**: 네 가지 데이터셋에서 실시한 실험 결과, LinearRAG는 검색 품질, 생성 정확도 및 확장성 면에서 기존의 상태 기반 모델을 지속적으로 초과하는 성과를 보였습니다. LinearRAG는 누락된 문맥 세부정보를 방지하였으며, 예를 들어 질문-답변 작업에서 Vanilla RAG보다 더 일관된 결과를 생성하여 62.87%의 성과를 얻었습니다. 이러한 결과는 LinearRAG가 실제 응용 프로그램에서 매우 실용적임을 입증합니다.



### Stop When Enough: Adaptive Early-Stopping for Chain-of-Thought Reasoning (https://arxiv.org/abs/2510.10103)
- **What's New**: 최근 대규모 언어 모델(LLM)의 성능 향상에는 Chain-of-Thought (CoT) 추론 과정이 큰 역할을 하였습니다. 하지만, 지나치게 많은 추론—소위 고찰(overthinking)—은 추론 비용을 증가시키고 잘못된 결론에 이르게 할 수 있습니다. 본 논문은 REFRAIN(REFlective-Redundancy for Adaptive INference)라는 훈련 없이도 과도한 추론을 완화하기 위해 추론을 중지해야 할 시점을 적응적으로 판단하는 프레임워크를 제안합니다. 이 프레임워크는 반사적이면서 중복된 추론을 식별하는 두 단계 중지 판별기와 문제의 난이도에 따라 중지 기준을 동적으로 조정하는 Sliding-Window Upper Confidence Bound (SW-UCB) 멀티암 밴디트 컨트롤러를 통합합니다.

- **Technical Details**: REFRAIN은 훈련 없이 작동하며, 두 가지 주요 아이디어를 통합합니다. 첫째, 반사적 중복 감지(Reflective Redundancy Detection)를 통해 이유가 반사적 자기 수정에서 중복 반복으로 전환될 때를 감지하는 판별기를 도입합니다. 둘째, SW-UCB를 기반으로 하는 적응형 중지 기준(Adaptive Thresholding)을 적용하여 문제 해결에 필요한 추론의 깊이를 동적으로 조정하여 탐색과 활용 사이의 균형을 유지합니다. 이를 통해 REFRAIN은 기존 CoT 기반 방법보다 20-55% 적은 토큰을 사용하면서도 정확도를 유지하거나 향상시킵니다.

- **Performance Highlights**: 네 가지 대표 벤치마크와 두 가지 모델 계열에서 REFRAIN은 기존 CoT보다 더 적은 토큰으로 우수한 성능을 나타내었습니다. REFRAIN은 정확도-효율성 경계를 지속적으로 향상시키며, 과도한 추론을 하는 대신, 모델이 필요로 하는 부분에 사고를 할당함으로써 문제 해결의 효율성을 높입니다. 논문의 결과는 REFRAIN이 다양한 추론 작업에서 일관된 정확도 및 효율성 향상을 보여줬음을 입증하며, '언제 멈출 것인가'를 테스트 및 시간 규모 확장의 새로운 차원으로 설정했습니다.



### Diversity Augmentation of Dynamic User Preference Data for Boosting Personalized Text Summarizers (https://arxiv.org/abs/2510.10082)
- **What's New**: 이 논문에서는 개인화된 문서 요약의 필요성을 강조하며, PerAugy라는 새로운 데이터 증강 기법을 제안합니다. 이 기법은 사용자 선호 이력과 요약 데이터의 동적 다양성을 활용하여, 개인화된 요약 모델의 성능을 크게 향상시킵니다. PerAugy는 사용자 상호작용 그래프(User Interaction Graph, UIG)를 기반으로 하여, 다양한 사용자 행동 프로필을 생성하고, 이로 인해 요약의 개인화 수준을 높입니다.

- **Technical Details**: PerAugy는 교차 경로 셔플링(cross-trajectory shuffling)과 요약 콘텐츠 섭동(perturbation)을 결합한 새로운 데이터 증강 기술입니다. 논문에서는 두 가지 주요 기술, 즉 이중 셔플링(Double Shuffling, DS)과 확률적 마르코프 변동(Stochastic Markovian Perturbation, SMP)을 사용하여 다채로운 사용자 프로필을 생성합니다. 이러한 접근 방식은 요약 모델이 더욱 다양한 이력을 경험하게 하여 개인적 선호를 효과적으로 반영할 수 있도록 돕습니다.

- **Performance Highlights**: 실험 결과, PerAugy로 증강된 데이터는 최신 사용자 인코더 모델인 NAML, EBNR, NRMS의 성능을 평균 24%, 25%, 18% 향상시켰습니다. 개인화된 요약 프레임워크에서는 GTP와 PENS 설정에서 평균 61.2%의 향상을 보였고, PENS+NRMS+T2 조합에서는 75%까지 도달했습니다. 또한, PerAugy는 자원이 적은 도메인에서도 효과적으로 일반화되어 성능 향상 효과를 보였습니다.



### A-IPO: Adaptive Intent-driven Preference Optimization (https://arxiv.org/abs/2510.10077)
- **What's New**: 이 논문에서는 기존의 Direct Preference Optimization (DPO) 방식의 한계를 극복하기 위해 Adaptive Intent-driven Preference Optimization (A-IPO)라는 새로운 방법을 제안합니다. A-IPO는 사용자의 내재된 의도를 추론하는 의도 모듈을 통합하여 보상 함수에 이를 명시적으로 반영함으로써, 모델의 응답과 사용자의 원래 의도 간의 정렬을 강화합니다. 이를 통해 기존의 DPO 방식보다 명확한 선호 응답과 비선호 응답 간의 차별화를 달성할 수 있음을 이론적 및 실증적으로 입증하였습니다.

- **Technical Details**: 기술적으로, A-IPO는 보상 함수의 새로운 재매개변수를 통해 의도-응답 유사성 항을 포함하여, 사용자의 다양한 의도를 모델링합니다. 이는 기존 DPO의 선호 마진을 증가시키며, 쌍별 음의 로그 우도(NLL)를 일관되게 감소시킴으로써 더 견고하고 일관된 선호 최적화를 가능하게 합니다. 또한, A-IPO는 새로운 평가 기준인 Real-Pref 및 Attack-Pref를 도입하여, 실제 세계에서의 선호 최적화 능력을 평가하고 있습니다.

- **Performance Highlights**: 포괄적인 실험 결과, A-IPO는 기존 기준선을 지속적으로 초월하며, 주요 메트릭에서 상당한 개선을 보여주었습니다. 예를 들어, Real-Pref 기준에서 최대 +24.8의 승리율 상승과 +45.6의 응답-의도 일관성을 나타냈고, Attack-Pref에서는 최대 +38.6의 응답 유사성과 +52.2의 방어 성공률 향상을 보였습니다. 마지막으로, GlobalOpinionQA-Ext 기준에서는 최대 +54.6의 의도 일관성 점수를 기록하여 A-IPO의 효과를 입증했습니다.



### Unilaw-R1: A Large Language Model for Legal Reasoning with Reinforcement Learning and Iterative Inferenc (https://arxiv.org/abs/2510.10072)
- **What's New**: 이 논문에서는 법률 추론을 위한 맞춤형 대규모 언어 모델인 Unilaw-R1을 소개합니다. 이 모델은 70억 개의 파라미터로 경량화되어 있으며 법률 분야에서의 세 가지 주요 도전 과제인 불충분한 법적 지식, 신뢰할 수 없는 추론 논리, 그리고 약한 비즈니스 일반화 문제를 해결합니다.

- **Technical Details**: Unilaw-R1은 두 단계의 훈련 전략을 채택합니다. 첫 번째 단계는 고품질 Chain-of-Thought (CoT) 샘플을 포함한 Unilaw-R1-Data라는 데이터 세트를 구축하는 것이며, 두 번째 단계는 Supervised Fine-Tuning (SFT)과 Reinforcement Learning (RL)을 결합하여 성능을 향상시키는 것입니다. 이 모델은 법적 유효성 보상 함수를 통합한 GRPO를 활용하여 법률 적합성을 확보합니다.

- **Performance Highlights**: Unilaw-R1은 법률 관련 작업에서 강력한 성과를 보여주며, 동등한 규모의 다른 모델들보다 우수한 결과를 기록했습니다. DeepSeek-R1-Distill-Qwen-32B와 같은 큰 모델과 유사한 성능을 달성하며, LawBench 및 LexEval에서 Qwen-2.5-7B-Instruct보다 평균 6.6%의 성능 향상을 보였습니다.



### CLMN: Concept based Language Models via Neural Symbolic Reasoning (https://arxiv.org/abs/2510.10063)
Comments:
          7 pages, 2 figures

- **What's New**: 이 논문에서는 의료 및 금융 분야에서 해석 가능성(interpretablity)이 제한된 자연어 처리(NLP) 시스템을 위한 새로운 신경-기호적(framework) 접근 방식인 개념 언어 모델 네트워크(Concept Language Model Network, CLMN)를 제안합니다. CLMN은 인간이 이해할 수 있는 연속 개념 표현을 사용하고, 개념 간의 동적 상호 작용을 모델링함으로써 기존 방법에 비해 성능과 해석 가능성을 모두 만족합니다. 이는 neural representations와 symbolic reasoning의 통합을 통해 이룬 성과로, 해석 가능한 논리 규칙을 자동으로 도출합니다.

- **Technical Details**: CLMN은 기존의 개념 병목 모델(concept bottleneck model, CBM)의 한계를 개선하는 모델로, 이 모델은 binary activation을 통해 정보를 손실시키지 않고 개념 표현을 최적화합니다. 또한, fuzzy-logic reasoning을 통해 개념 간의 상호 작용 규칙을 동적으로 학습하며, 이를 통해 의료 텍스트에서 개념 간의 관계를 명확히 설명할 수 있습니다. CLMN은 다양한 사전 훈련된 언어 모델(pretrained language models, PLMs)과 데이터셋에 적용되어 성능을 향상시키며, 개념 기반 접근 방식의 정확도 저하 문제를 해결합니다.

- **Performance Highlights**: 여러 데이터셋과 사전 훈련된 언어 모델에 대한 실험 결과, CLMN은 기존 개념 기반 방법보다 높은 정확도를 달성하면서도 설명 품질을 개선했습니다. 이는 CLMN이 개념 블록을 통해 해석 가능한 설명을 생성함으로써 사용자 신뢰를 증가시키고, 중요 분야의 안전 및 규제 준수를 확보할 수 있음을 보여줍니다. 이러한 성과는 CLMN 같은 통합된 개념 공간에서 neural representations와 symbolic reasoning을 결합함으로써 가능한 것으로, 실질적으로 투명한 자연어 처리 시스템을 제공하며, 다양한 응용 프로그램에 큰 가치를 더합니다.



### HUME: Measuring the Human-Model Performance Gap in Text Embedding Task (https://arxiv.org/abs/2510.10062)
Comments:
          Submitted to ICLR 2026

- **What's New**: HUME(Human Evaluation Framework for Text Embeddings)를 도입하여 임베딩 모델의 성능 비교에서 인간 성능 기준을 제공하는 새로운 접근 방식을 제안합니다. HUME는 16개의 다양한 MTEB 데이터셋을 사용해 재정렬, 분류, 클러스터링 및 의미적 텍스트 유사성 작업을 평가합니다. 이 연구는 다양한 언어에서 인간의 평균 성능을 77.6%로 측정하며, 가장 우수한 임베딩 모델의 성능 80.1%와 비교합니다.

- **Technical Details**: HUME는 다섯 가지 기준을 바탕으로 다양한 언어의 데이터셋을 포괄적으로 평가합니다. 이는 고자원 언어(예: 영어, 아랍어)와 저자원 언어(예: 노르웨이어) 모두를 포함하며, 뉴스, 소셜 미디어, 과학적 문헌 등 다양한 도메인에서의 응용을 캡쳐합니다. 각 작업 카테고리는 주요 평가 지표를 사용하여 일관된 비교를 가능하게 합니다.

- **Performance Highlights**: HUME를 통해 임베딩 모델과 인간의 성능 비교가 가능해짐으로써, 모델 및 벤치마크를 개선할 수 있는 통찰을 제공합니다. 일관된 평가 프로토콜을 통해 다양한 임베딩 모델의 강점과 약점을 강조하며, 중력적인 데이터셋 문제와 저자원 언어의 단점을 드러냅니다. 이러한 기초 연구는 인간 중심의 임베딩 모델 평가에 큰 기여를 할 것입니다.



### Lightweight Baselines for Medical Abstract Classification: DistilBERT with Cross-Entropy as a Strong Defau (https://arxiv.org/abs/2510.10025)
Comments:
          Healthcare AI, Medical Text Classification, Lightweight LLMs, DistilBERT, Reproducibility

- **What's New**: 이 논문은 의료 환경에서의 엄격한 비용, 대기 시간, 개인 정보 보호 제한으로 인해 대형 언어 모델의 배포가 어렵다는 점을 강조합니다. 저자들은 의료 초록 분류를 위한 경량 모델을 활용하여, 자원 제약 하에서도 성능을 최적화할 수 있는 방법을 모색합니다. DistilBERT가 매우 적은 수의 매개변수를 사용하면서도 BERT base보다 더 좋은 균형 잡힌 성능을 보임을 제안합니다.

- **Technical Details**: 논문에서는 의료 문헌 초록에 대한 다섯 가지 단일 레이블(classification) 분류 작업을 설정하였으며, 이를 위해 Hugging Face의 public medical_abstracts corpus를 사용했습니다. Cross-entropy, class weighted cross-entropy, focal loss와 같은 세 가지 손실 함수에 대해 실험하였고, 전체적인 파라미터 수를 줄이면서도 효과적인 분류 성능을 달성하였습니다. 평가 지표로는 Accuracy, Macro F1, Weighted F1을 사용하였습니다.

- **Performance Highlights**: 저자들은 DistilBERT가 BERT base보다 더 적은 자원으로도 상위 성능을 달성했음을 보여주었으며, 일반적인 cross-entropy가 오히려 강력한 기본 모델로 작용할 수 있음을 시사했습니다. 제안된 모델을 사용하여 confusion 분석을 수행하였고, 다양한 손실 함수가 오차 구조에 미치는 영향을 명확히 했습니다. 이 연구는 compact encoder와 cross-entropy를 기본으로 설정하고, 더 나아가 요건에 맞는 조정 및 검증을 통해 성능을 높이는 것을 추천합니다.



### Path Drift in Large Reasoning Models:How First-Person Commitments Override Safety (https://arxiv.org/abs/2510.10013)
- **What's New**: 이 논문에서는 Long Chain-of-Thought (Long-CoT) 모델에서 발생할 수 있는 새로운 취약점인 Path Drift를 밝혀냈습니다. Path Drift는 모델의 추론 경로가 안전한 경로에서 이탈하는 현상으로, 이를 이해함으로써 더욱 효과적인 안전 수칙을 적용할 수 있습니다. 연구는 세 가지 행동 유발 요인을 통해 Path Drift를 발생시키는 메커니즘을 규명하고, 이를 기반으로 세 단계의 Path Drift 유도 프레임워크를 제안합니다.

- **Technical Details**: Path Drift는 비즈니스 위험을 피하면서 안전하게 추론하는 모델의 능력을 저하시킬 수 있는 중요한 문제입니다. 이 연구에서는 LLM의 내부 추론 경로를 세 가지 단계, 즉 Cognitive Load Amplification, Self-Goal Activation, Condition Chain Injection으로 나누어 분석합니다. Path Drift를 유도하는 각 단계는 모델의 안전 거부 신호를 약화하고, 이로 인해 전체 경로가 안전하지 않은 결론으로 유도될 수 있습니다.

- **Performance Highlights**: 실험 결과, 여러 alignment-trained reasoning models에서 Path Drift 공격이 유도된 경우 안전 거부율이 크게 감소한 것으로 나타났습니다. 특히, first-person prompting 사용 시 모델의 의도 달성에 초점을 맞추게 되어 초기 안전 점검을 지연시켰습니다. 이는 모델의 위험 인식이 지연되며, 길고 복잡한 사고 과정에서 안전한 의사결정을 방해할 수 있음을 보여줍니다.



### Beyond the limitation of a single query: Train your LLM for query expansion with Reinforcement Learning (https://arxiv.org/abs/2510.10009)
- **What's New**: 이 논문에서는 ExpandSearch라는 새로운 강화 학습( reinforcement learning) 기반의 검색 프레임워크를 제안합니다. 이 프레임워크는 검색 에이전트가 쿼리를 확장하고, 선택적으로 정보를 정제하여 복잡한 질문에서 정확한 답변을 생성할 수 있도록 돕습니다. ExpandSearch는 특히 다단계 추론(multi-hop reasoning) 작업에서 높은 성능을 보여줍니다.

- **Technical Details**: ExpandSearch는 기존의 검색 에이전트가 여러 쿼리 변형을 생성하여 정보를 검색하도록 훈련합니다. 이 과정에서 쿼리 생성과 정보 정제 단계를 명확히 구분함으로써 의미의 불완전성과 정보 과부하 문제를 해결하고자 합니다. 또한, 두 가지 유형의 쿼리 확장, 즉 구문 확장(syntax expansion)과 의미 확장(semantic expansion)을 도입하여 다양한 관점을 포착하려고 합니다.

- **Performance Highlights**: 실험 결과 ExpandSearch는 최신 기술 기준에 비해 평균 4.4%의 성능 향상을 이루었으며, 특히 다양한 증거 집합을 요구하는 복잡한 다단계 추론 과제에서 강한 성과를 보였습니다. 이는 3B LLM 규모의 모델에서도 쿼리 확장 능력을 크게 개선할 수 있음을 보여줍니다.



### MTP-S2UT: Enhancing Speech-to-Speech Translation Quality with Multi-token Prediction (https://arxiv.org/abs/2510.10003)
Comments:
          Copyright 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works

- **What's New**: 이번 연구는 direct speech-to-speech translation 방법론의 한계를 극복하기 위해 multi-token prediction (MTP) loss를 소개합니다. 기존의 speech tokens는 의미적으로 조밀하지 않은 특성이 있어, 여러 토큰이 필요한데, MTP를 통해 각 위치에서 여러 개의 토큰을 예측합니다. 이는 정보 밀도를 높이고 더 완전한 의미를 포착하여 번역 품질을 향상시키는데 기여합니다. MTP-S2UT loss는 특히 hidden representation을 향상시켜 성능을 극대화하는 점이 특징입니다.

- **Technical Details**: 제안하는 방법은 기존 S2UT 모델에 MTP loss 변형을 통합하는 방식으로 구성됩니다. S2UT는 소스 음성을 디스크리트 speech tokens로 변환하고 이를 기반으로 target speech를 생성하는 구조를 가집니다. MTP-S2UT loss는 정보를 조기에 통합하는 것을 목표로 하며, CTC loss가 계산되는 hidden layer에서 적용되어, 더 깊이 있는 의미 표현을 형성합니다. 이를 통해 MTP가 hidden representation의 품질을 높이는데 기여하는 것을 확인했습니다.

- **Performance Highlights**: 실험 결과, MTP-S2UT가 기존의 다른 MTP 손실 변형들에 비해 프랑스어에서 영어로의 번역 품질을 현저히 향상시킴을 보여주었습니다. MTP loss는 text token의 forward shifting을 유도하고 speech token 예측의 불확실성을 감소시키는데 효과적이며 MTP-S2UT의 효과가 두드러졌습니다. 연구는 MTP loss가 S2UT 프레임워크에서 어떻게 성능을 향상시키는지에 대한 새로운 통찰을 제공합니다.



### Toward Machine Translation Literacy: How Lay Users Perceive and Rely on Imperfect Translations (https://arxiv.org/abs/2510.09994)
Comments:
          EMNLP 2025

- **What's New**: 이 연구는 기계 번역(Machine Translation, MT)의 품질이 사용자들, 특히 비이중언어자(non-bilingual users)에 미치는 영향을 조명합니다. 공공 박물관에서 진행된 연구를 통해 MT의 유창성(fluidity)과 적합성(adequacy) 오류가 사용자 의존성에 미치는 영향을 살펴보았습니다. 이로써 비이중언어 사용자의 MT에 대한 의존도가 종종 과도하다는 사실이 드러났습니다.

- **Technical Details**: 연구는 452명의 참가자를 대상으로 진행되었으며, MT 사용 중 오류가 사용자의 신뢰도에 미치는 영향을 분석했습니다. 비이중언어 사용자는 평가 전략이 부족하여 MT에 과도하게 의존하게 되며, 오류를 경험함으로써 향후 MT에 대한 의존을 재조정하게 됩니다. 이러한 관찰은 MT의 품질뿐만 아니라 사용자에게 MT 리터러시(MT literacy)를 증진시키기 위한 필요성을 강조합니다.

- **Performance Highlights**: 연구 결과에 따르면, 비이중언어 사용자는 기계 번역의 오류를 인식하면서도 그에 대한 대처 방법이 부족하여 MT에 지나치게 의존합니다. MT의 품질 향상과 사용자 교육은 MT의 실제 사용에 있어 매우 중요합니다. 이는 향후 MT의 평가 및 자연어 처리(Natural Language Processing, NLP)의 설명 기법에 대한 필요성을 부각시킵니다.



### Unifying Tree Search Algorithm and Reward Design for LLM Reasoning: A Survey (https://arxiv.org/abs/2510.09988)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLM)의 효율성을 높이기 위해 제안된 새로운 통합 프레임워크를 소개합니다. 이 프레임워크는 검색 알고리즘을 세 가지 핵심 구성 요소로 분해하여, 각 구성 요소의 역할을 명확히 하고 서로 다른 검색 방법을 체계적으로 비교할 수 있는 기반을 제공합니다. 특히, 임시적인 검색 안내와 지속적인 파라메트릭 보상 모델링 간의 공식적인 구분을 설정하여 기존의 모호한 보상 개념을 해결합니다.

- **Technical Details**: 저자들은 검색 메커니즘(Search Mechanism), 보상 수식(Reward Formulation), 전이 함수(Transition Function)라는 세 가지 핵심 구성 요소를 통해 검색 기반 추론의 메커니즘을 분석합니다. 테스트 시간 스케일링(Test-Time Scaling, TTS)과 자기 개선(Self-Improvement)을 위한 파라메트릭 보상 모델링의 역할을 구체적으로 정의하여, 파라메트릭 지식으로 전환되는 복잡한 추론 과정을 명확하게 설명합니다. 또한, 기존 알고리즘과 새로운 알고리즘을 세 가지 축을 기준으로 정리하는 독창적인 분류 체계를 제안합니다.

- **Performance Highlights**: 논문에서 제안하는 체계적인 구성 요소 기반 분류법은 다양한 검색 알고리즘을 효과적으로 비교할 수 있게 해 줍니다. 파라메트릭 보상 모델링을 통해 얻어진 고품질의 추론 행동은 기존의 LLM 모델에 통합되어 지속적인 자기 진화 루프(self-evolutionary loop)를 실현합니다. 저자들은 이 모델이 다단계 추론 과제에서 상당한 성과를 이끌어 낼 수 있다고 강조합니다.



### Beyond Fertility: Analyzing STRR as a Metric for Multilingual Tokenization Evaluation (https://arxiv.org/abs/2510.09947)
Comments:
          NeurIPS 2025 Workshop

- **What's New**: 이 논문은 대규모 언어 모델(LLMs)에서 토큰화(tokenization)의 중요성을 강조합니다. 기존의 평가 지표인 fertility가 언어와 도메인 간의 어휘 분배를 제대로 나타내지 않는 문제를 지적하며, 단일 토큰 유지율(Single Token Retention Rate, STRR)이라는 새로운 지표를 제안합니다. STRR는 단일 토큰으로 보존된 단어의 비율을 측정하여 언어 간 공정성을 평가하는 데 도움을 줄 수 있습니다.

- **Technical Details**: 이 연구에서는 여섯 개의 널리 사용되는 LLM 토크나이저를 분석하였으며, 영어, 중국어, 힌디어 등 총 일곱 개 언어를 대상으로 했습니다. 기존의 fertility 지표는 복잡한 다국어 환경에서의 어휘 분배를 간과하지만, STRR은 각 언어에서 전체 단어 보존의 비율을 정량화하여 이 문제를 해결합니다. 이를 통해 언어별로 토크나이저가 어휘를 어떻게 할당하는지에 대한 명확한 인사이트를 제공합니다.

- **Performance Highlights**: 분석 결과, 영어는 두 도메인 모두에서 높은 일관성을 보이는 반면, 중국어는 높은 fertility를 기록했습니다. 힌디어는 가장 낮은 STRR을 보여, 심각한 단어 분절화를 드러냈습니다. STRR을 통해 연구진은 현재 토크나이저의 불평등한 언어 지원 문제를 명확히 수치화하였으며, 이는 공정하고 효율적인 다국어 토크나이저 설계에 기초적인 지침을 제공합니다.



### Unpacking Hateful Memes: Presupposed Context and False Claims (https://arxiv.org/abs/2510.09935)
- **What's New**: 이번 연구에서는 미움(학대) 표현의 본질을 철학적 및 심리학적 관점에서 탐구합니다. 기존의 미움 meme 탐지 방법들이 주로 사전 훈련된 언어 모델에 의존하고 있었던 반면, 우리는 meme이 미움으로 인식되는 두 가지 주요 특성, 즉 presupposed context와 false claims를 제시합니다. 이를 기반으로 하여 PCM(약어: Presupposed Context Module)과 FACT(약어: False Claims Module) 모듈을 개발하고, SHIELD라는 프레임워크를 소개합니다.

- **Technical Details**: SHIELD는 미움이 표현되는 방식의 본질을 적절히 캡처하는 데 중점을 둡니다. PCM은 내부-모달(context) 정보를 인코딩하고 교차 모달(context) 정보를 융합하여 meme이 암묵적인 가치 판단을 전달하는지를 결정합니다. FACT 모듈은 두 가지 하위 모듈인 사회적 지각 모듈(Social Perception Module)과 교차 모달 참조 모듈(Cross-modal Reference Module)을 통해 존재하는 잘못된 주장(falsehoods) 및 의미적 부정확성을 감지합니다.

- **Performance Highlights**: SHIELD는 다양한 데이터 세트와 지표에서 기존 최첨단 방법들을 초월하는 성능을 입증하였습니다. 이 프레임워크는 하위 문제에 대한 전문화를 평가하고, 가짜 뉴스 classification 태스크에서의 성능을 평가하여 그 일반화 가능성과 다재다능함을 보여줍니다. 실험 결과, SHIELD는 미워하는 meme 탐지뿐 아니라 다양한 다른 작업에서도 강력한 성능을 발휘합니다.



### Enhancing Faithfulness in Abstractive Summarization via Span-Level Fine-Tuning (https://arxiv.org/abs/2510.09915)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)을 활용한 추상적 요약의 신뢰성을 높이기 위해 미세 조정(fine-tuning) 방법을 제안합니다. 기존의 방법들이 생성된 요약의 다양한 오류를 완전히 해결하지 못하는 한계를 극복하기 위해, 논문에서는 새로운 데이터셋을 구축하고 요약의 신뢰성을 높이는 3가지 기법(gradient ascent, unlikelihood training, task vector negation)을 평가합니다. 이 과정에서 LLM이 생성한 요약의 환각(hallucination) 문제를 스팬 수준에서 주석을 달아 이를 미세 조정하는 데 활용합니다.

- **Technical Details**: 추출한 스팬 수준의 주석을 기반으로 LLM을 미세 조정하여 신뢰성을 향상시키고자 합니다. 실험적으로 3가지 방법이 시도되었으며, 그 중 unlikelihood training이 가장 효과적인 기법으로 나타났습니다. 연구진은 새로운 데이터셋을 구축하여 신뢰성 있는 요약과 신뢰성이 떨어진 요약을 포함시켰으며, 이를 통해 모델 학습 시 구체적인 환각 패턴을 표적으로 삼았습니다.

- **Performance Highlights**: 실험 결과, 모든 접근 방식이 스팬 수준의 주석을 활용하여 신뢰성을 향상시키는 데 성공했으며, 특히 unlikelihood training 방식이 가장 안정적인 성능을 보였습니다. 이를 통해 생성된 요약이 더욱 사실적이고 신뢰할 수 있는 정보를 제공할 수 있음을 확인했습니다. 이러한 결과는 LLM 기반 요약 시스템의 실제 적용 가능성을 높이는 데 기여할 것입니다.



### Don't Throw Away Your Pretrained Mod (https://arxiv.org/abs/2510.09913)
- **What's New**: 본 논문에서는 모델 협업을 통해 다양한 언어 모델의 장점을 효과적으로 조화시킬 수 있는 새로운 접근법인 Switch Generation을 제안합니다. 이 방식은 두 개 이상의 모델이 서로 협력하여 응답을 생성하게 하여 특정 상황에서 최적의 성능을 발휘하게 합니다. 특히, Switch Generation은 각 모델의 특화된 기술을 조합해, 각 모델이 약점인 부분을 보완합니다.

- **Technical Details**: Switch Generation의 핵심은 ‘Query-Trace-Candidate Problem (QTC 문제)’으로, 주어진 쿼리와 이전 생성된 텍스트를 기반으로 어떤 모델이 다음 세그먼트를 작성할지 결정합니다. 각 모델이 생성한 텍스트 조각을 평가하여, 가장 성능이 좋았던 모델을 선택하는 방식으로 훈련됩니다. 이 과정에서 다양한 쿼리와 모델 후보군을 고려하여 동적으로 스위칭하는 언어 모델을 훈련하게 됩니다.

- **Performance Highlights**: 전략적인 실험 결과, Switch Generation은 18개 작업 중 16개에서 개인 모델보다 우수한 성과를 보였으며, 평균 12.9%의 성능 향상을 달성했습니다. 이러한 방식은 특정 작업에서 개별 모델이 힘을 쏟지 못했던 문제를 해결하는 데 도움을 주며, 새로운 작업과 모델 설정으로 일반화될 수 있는 능력을 보여줍니다. 또한 Switch Generation에서 생성된 결과물을 다시 하나의 모델로 정제하여 효율성을 제공할 수 있는 가능성도 확인되었습니다.



### HIPPD: Brain-Inspired Hierarchical Information Processing for Personality Detection (https://arxiv.org/abs/2510.09893)
- **What's New**: 이 논문에서는 HIPPD라는 뇌 영감을 받은 프레임워크를 제안합니다. 이는 인간의 계층적 정보 처리를 모방하여 개인의 성격 특성을 탐지하는 것을 목표로 합니다. HIPPD는 대규모 언어 모델을 활용하여 전역적 의미 추론과 깊은 특징 추상화를 가능하게 합니다.

- **Technical Details**: HIPPD는 대뇌 피질 모사를 통해 텍스트 데이터 내의 장기 의존성을 포착하고, 전두엽에 기반한 동적 기억 모듈을 통해 중요한 특징을 선택적으로 유지 및 업데이트합니다. 마지막으로, 기저핵(선조체) 기능을 모방한 전문 모델 라우팅 레이어가 적용되어, 엄격한 승자-독식 메커니즘을 통해 입력 데이터를 최적의 전문 모델로 동적으로 라우팅합니다.

- **Performance Highlights**: Kaggle 및 Pandora 데이터셋에서의 광범위한 실험 결과, HIPPD는 최신의 다른 방법들과 비교하여 지속적으로 우수한 성능을 보여주었습니다. 이와 같은 성능은 클래스 불균형 및 짧은 텍스트 문제를 해결하는 동시에, 다양한 피쳐가 부족한 작업에서도 잘 일반화됩니다.



### Abductive Preference Learning (https://arxiv.org/abs/2510.09887)
- **What's New**: 이 논문에서는 기존의 선호 학습(preference learning) 기법의 한계를 극복하기 위해 새로운 접근 방식인 abductive preference learning을 제안합니다. 과거의 방법들이 주어진 프롬프트(prompt)에 맞춰 올바른 응답을 선택하는 데 중점을 두었던 반면, 이 새로운 방법은 응답을 기반으로 프롬프트를 수정하는 방향으로 조건을 전환합니다. 이는 모델이 프롬프트의 변화를 감지하고 그에 맞춰 출력 결과를 조정할 수 있게 하는 것을 목표로 합니다.

- **Technical Details**: abductive preference learning는 전통적인 DPO(Direct Preference Optimization) 목표에서 프롬프트와 응답의 역할을 교환하는 방식을 사용합니다. 이 방식은 조건부 확률 Pr(x|y)를 증가시키는 것으로, 주어진 응답에 대해 어떤 프롬프트가 더 적절한지를 학습합니다. 기존 선호 학습 방법이 응답의 품질 향상에 초점을 맞추는 반면, 이 새로운 접근 방식은 프롬프트의 변수가 모델 출력에 미치는 영향도 함께 고려합니다.

- **Performance Highlights**: 실험 결과, multitask DPOP은 QA 데이터셋에서 응답 정확도를 90.0%에서 99.5%로, 프롬프트 정확도를 54.7%에서 85.0%로 향상시켰습니다. 또한 AlpacaEval 벤치마크에서 이 방법이 승률을 5.26%에서 6.17%로 늘린 것을 확인했습니다. 도메인 간 과제인 sarcasm detection에서도 정확도를 50.0%에서 87.0%로 증가시켜, 이 접근 방식의 잠재적인 효과를 보여주고 있습니다.



### Closing the Data-Efficiency Gap Between Autoregressive and Masked Diffusion LLMs (https://arxiv.org/abs/2510.09885)
- **What's New**: 이번 연구에서는 auto-regressive large language model (arLLM)과 masked diffusion large language model (dLLM)의 사후 훈련(post-training) 단계에서 지식 주입(knowledge injection)의 데이터 효율성과 성능을 비교합니다. arLLM은 fine-tuning 중 'reversal curse'로 인한 한계를 가지고 있지만, dLLM은 이러한 문제에서 자유롭고, 데이터가 적은 환경에서도 낮은 검증 손실(validation loss)을 달성할 수 있습니다.

- **Technical Details**: 연구는 세 가지 데이터세트를 사용하여 arLLM과 dLLM의 fine-tuning 성능을 비교합니다. dLLM은 비유 없이도 정방향(forward) 및 역방향(backward) 질문에서 높은 정확도를 달성하며, arLLM은 paraphrase를 통해서만 일반화에 성공합니다. 또한, 본 연구에서는 새로운 masked fine-tuning 패러다임을 제안해 arLLM의 데이터 효율성을 크게 향상시킵니다.

- **Performance Highlights**: arLLM은 paraphrase를 사용하지 않고는 backward 스타일 질문에서 실패하는 반면, dLLM은 높은 정확도를 유지합니다. 본 연구의 masked fine-tuning 방법론은 arLLM의 성능 갭을 줄이며, 이러한 결과는 dLLM의 뛰어난 데이터 효율성이 사후 훈련에서도 시행될 수 있음을 보여줍니다.



### DELTA: Dynamic Layer-Aware Token Attention for Efficient Long-Context Reasoning (https://arxiv.org/abs/2510.09883)
- **What's New**: DELTA는 대규모 추론 모델(Large Reasoning Models, LRM)의 효율성을 높이는 훈련이 필요 없는 희소 주의 메커니즘입니다. 이 방법은 모델의 정확도 손실 없이 계산 효율성을 달성합니다. DELTA는 변환기 층을 세 그룹으로 나누어, 초기 층에서 전 주의를 사용하고, 중요한 토큰을 선택하는 selection layers와 선택된 하위 집합에 대해서만 주의를 기울이는 sparse-attention layers로 구성되어 있습니다.

- **Technical Details**: DELTA는 각 디코딩 스텝에서 신중하게 선택된 토큰의 집합에 대한 계산만을 수행하고, 전체 주의 맵을 사용하는 소수의 중간 층을 활용하여 다음 층을 위한 중요한 토큰을 예측합니다. 이는 최근 맥락을 보장하면서 높은 주의 토큰을 정확하게 식별할 수 있도록 합니다. 이러한 과정은 주의 패턴의 강한 상관관계와 토큰 중요도가 변하는 점을 고려하여 이루어집니다.

- **Performance Highlights**: DELTA는 AIME 및 GPQA-Diamond와 같은 추론 기준에서 정확도를 유지하면서 최대 5배의 주의 토큰 수를 줄이고, 최대 1.5배의 종단 간 속도 향상을 달성합니다. DELTA는 기존의 희소 주의 방법들보다 월등한 성능을 보이며, 추론 과제의 정확성을 저하시키지 않고 속도를 높이는 데 기여할 수 있습니다.



### iBERT: Interpretable Style Embeddings via Sense Decomposition (https://arxiv.org/abs/2510.09882)
- **What's New**: 이번 논문에서 우리는 iBERT( interpretable-BERT)를 소개하며, 이는 본질적으로 해석 가능하고 제어 가능한 임베딩을 생성하는 인코더입니다. iBERT는 언어에서의 스타일과 의미 구조와 같은 구별 가능한 단서를 모듈화하고 노출하도록 설계되었습니다. 각 입력 토큰은 k개의 맥락 독립적인 의미 벡터에 대한 희소하고 비음수 혼합으로 표현되어, 문장 임베딩으로 풀링되거나 토큰 레벨에서 직접 사용될 수 있습니다.

- **Technical Details**: iBERT 아키텍처는 각 토큰을 희소한 비음수 혼합으로 표현하여 의미와 스타일의 특정 차원에 대한 모듈식 제어를 가능하게 합니다. 이는 훈련 과정에서 학습된 특정 분포를 활용하여 분석, 속성 부여 및 임베딩 공간 내에서의 변화를 지원합니다. 이 아키텍처는 자기 회귀 디코딩을 위해 희소한 토큰 레벨 감각을 모델링한 Backpack 공식을 기반으로 하여, 문장 수준의 조합 가능성과 양방향 입력을 지원합니다.

- **Performance Highlights**: iBERT는 STEL, SoC, PAN과 같은 세 가지 벤치마크에서 강력을 발휘하며, SBERT 기반과 비교해 STEL 점수에서 +8% 향상을 보였습니다. 스타일 분석 작업에 특화되어 각 임베딩은 해석 가능한 감각 벡터의 구조적 조합으로 구성되어, 특정 스타일 속성(예: 이모지 사용, 정중함, 철자 오류)을 식별하고 임베딩 공간에서의 타겟 수정이 가능합니다. 비록 스타일 작업을 통해 평가되긴 했지만, iBERT는 일반 용도의 인코더로, 해석 가능성과 제어가 요구되는 도메인에서 적합합니다.



### CoBia: Constructed Conversations Can Trigger Otherwise Concealed Societal Biases in LLMs (https://arxiv.org/abs/2510.09871)
Comments:
          EMNLP 2025 (Oral)

- **What's New**: 이 논문은 CoBia라는 경량 적대적 공격의 모음을 소개합니다. 이는 대형 언어 모델(LLMs)이 대화 중에 비정상적이거나 윤리적이지 않은 행동에서 벗어나는 조건을 체계적으로 분석할 수 있게 해줍니다. CoBia는 모델이 특정 사회 집단에 대한 편향된 주장을 표현하는 구조화된 대화를 생성합니다.

- **Technical Details**: CoBia 방법론은 하나의 쿼리를 통해 LLM이 대화 중에 드러나는 숨겨진 사회적 편향을 노출하기 위해 구조화된 대화를 사용하는 일종의 경량 적대적 공격입니다. 이 연구는 11개의 개방형 및 폐쇄형 LLM을 평가하며, 6개의 사회인구 통계적 범주에 관련된 입력에 기반한 출력 결과를 분석합니다. CoBia 데이터셋은 112개의 사회 그룹에 대한 부정적인 기술어를 포함하고 있습니다.

- **Performance Highlights**: 결과는 의도적으로 구성된 대화가 편향 증폭을 신뢰성 있게 드러낸다는 것을 보여줍니다. LLM들은 대화 중에 편향된 후속 질문을 거부하는 데 실패하는 경우가 많아, 이는 LLM이 일상적인 상호작용에서 드러나는 깊이 내재된 편향을 강조합니다. 연구진은 코드를 포함한 다양한 자료를 제공하며, 이는 연구자들이 이를 활용하여 LLM의 안전성을 향상시키는 데 기여할 수 있도록 합니다.



### NarraBench: A Comprehensive Framework for Narrative Benchmarking (https://arxiv.org/abs/2510.09869)
- **What's New**: NarraBench는 내러티브 이해(narrative understanding) 작업을 이론에 기반한 분류체계로 정리한 새로운 프레임워크입니다. 이 연구는 78개의 기존 벤치마크에 대한 설문 결과를 통해 현재의 평가 방식에서 간과되거나 일치하지 않는 내러티브 이해의 다양한 측면을 강조합니다. 현재 벤치마크가 내러티브 작업의 단 27%만을 잘 반영하며, 특히 서사적 사건, 스타일, 관점, 폭로와 같은 영역은 거의 존재하지 않음을 발견하였습니다.

- **Technical Details**: NarraBench 프레임워크는 LLMs의 내러티브 이해 능력을 평가하기 위해 기존 벤치마크를 통합하고, 이론적으로 일관된 방식으로 향후 개발이 필요한 영역을 식별합니다. 이 연구는 78개의 벤치마크를 검토하였으며, 이 중 39개는 데이터가 부족하고, 나머지 39개는 내러티브 이해의 주요 측면과 적절하거나 좋은 적합성을 제공합니다. 이 연구는 내러티브 이해의 50개 독립적 작업을 파생하여 새로운 벤치마크 개발을 위한 로드맵을 제공합니다.

- **Performance Highlights**: 기존 벤치마크는 주로 결정론적(story-level) 이해 작업에 집중하여 내러티브 의사소통의 복잡한 차원을 놓치는 경향이 있습니다. NarraBench는 이러한 제한을 보완하기 위해 통합 테스트 시스템을 제공하며, 내러티브 이론의 정합성과 다양성을 고려한 평가 방식을 제안합니다. 기존 벤치마크가 놓치는 평가 기준을 확인하고, 내러티브 이해의 발전에 기여할 수 있는 포괄적인 리소스를 만듭니다.



### NG-Router: Graph-Supervised Multi-Agent Collaboration for Nutrition Question Answering (https://arxiv.org/abs/2510.09854)
- **What's New**: 이번 연구에서는 Nutritional-Graph Router (NG-Router)라는 새로운 프레임워크를 소개하여, 영양 관련 질문 응답(Nutrition QA)을 다중 에이전트 협업 문제로 구성했습니다. 이 시스템은 서로 다른 지식 그래프를 통합하여 에이전트 노드를 연결하고, 그래프 신경망(Graph Neural Network)을 사용하여 에이전트 간의 동적 경로를 학습합니다. 이를 통해 개인 맞춤형 식이 안내를 제공하면서도 복잡한 영양 관련 질병 예방에 도움을 줄 수 있는 가능성을 보여줍니다.

- **Technical Details**: NG-Router는 지식 그래프(Knowledge Graph) 기반의 다중 에이전트 협업을 통해 작업 인식 라우팅 분포(Task-aware Routing Distributions)를 생성합니다. 이 프레임워크는 단순히 에이전트를 결정론적으로 배정하는 것이 아니라, 경험적 성과에서 파생된 소프트 슈퍼비전을 사용하여 동적 확률 분포를 학습합니다. 또한, 그라디언트 기반 서브그래프 검색 메커니즘을 도입하여 훈련 중 중요 증거를 필터링하고, 다단계 및 관계 추론을 강화합니다.

- **Performance Highlights**: 다양한 벤치마크와 모델에서의 광범위한 실험을 통하여 NG-Router는 일관되게 단일 에이전트 모델과 앙상블(Ensemble) 기준을 초과하는 성능을 보였습니다. 연구 결과는 NG-Router가 복잡한 영양 건강 작업에 대해 도메인 인식 다중 에이전트 추론을 위한 원칙 있는 접근 방식을 제공한다는 점을 강조합니다. 이러한 성과는 건강한 식습관 증진을 위해 대규모 개입이 필요한 시대에 매우 중요한 시사점을 제공합니다.



### Steering Embedding Models with Geometric Rotation: Mapping Semantic Relationships Across Languages and Models (https://arxiv.org/abs/2510.09790)
Comments:
          9 pages, 3 Figure, 1 table, preprint

- **What's New**: 이 논문에서는 Rotor-Invariant Shift Estimation (RISE)라는 기하학적 접근 방식을 소개하여 현대 언어 모델의 의미 변환을 일관된 회전 연산으로 표현하고 있습니다. RISE는 여러 언어와 모델에 걸쳐 높은 성능 전이를 보여주며, 해당 접근법이 다국어 기하학 구조의 유사성을 시사한다고 주장합니다. 세 개의 임베딩 모델과 다양한 형태소를 가진 일곱 가지 언어로 RISE의 성능을 평가한 결과, 다양한 문법적 특징이 일관되게 매핑됨을 보여주었습니다.

- **Technical Details**: 리서치에서는 현대 언어 표현의 매니폴드 구조를 활용하여 의미 변환을 기하학적으로 다루는 RISE 방식을 채택하였습니다. 이 접근법은 단위 고차원 구에서의 회전으로 의미 변환을 해석하며, 복수의 언어 및 모델 아키텍처 간의 기하학적 변환을 식별하고 일반화하는 방법을 제안합니다. 논문은 RISE가 부정, 조건부 및 공손성 같은 세 가지 담화 수준의 의미 변화를 규명하는 데 어떻게 사용되는지를 보여줍니다.

- **Performance Highlights**: RISE는 일곱 개의 형태소적으로 다양한 언어에서 세 가지 임베딩 모델을 통해 일관되게 담화 수준의 의미 변환을 매핑하여 성능 하이라이트를 제공합니다. 연구 결과, 이 방법은 언어와 모델에 관계없이 의미 변환이 기하학적으로 일관된 방식으로 표현될 수 있음을 나타내며, 이는 다국어 맥락에서 의미 변환의 기하학적 구조를 확장합니다. 이 연구는 기존의 선형 표현 가설에 대한 실험적 지지를 제공하며, 더 나아가 의미 변화의 일반적인 기하학적 구조의 존재를 입증합니다.



### PromptGuard at BLP-2025 Task 1: A Few-Shot Classification Framework Using Majority Voting and Keyword Similarity for Bengali Hate Speech Detection (https://arxiv.org/abs/2510.09771)
- **What's New**: BLP-2025 작업 1A는 벵골어 혐오 발언을 여섯 개 범주로 분류하는 과제를 제시합니다. 기존의 감독학습(supervised) 접근 방식은 저자원(low-resource) 언어에 대해 많은 레이블된 데이터셋이 필요하여 비용이 많이 듭니다. 이에 대한 해결책으로 PromptGuard라는 몇 개의 샷(few-shot) 프레임워크를 개발하였습니다.

- **Technical Details**: PromptGuard는 키워드 추출을 위한 카이제곱(chi-square) 통계 분석과 의사결정을 위한 적응형 다수결(adaptive majority voting)을 결합한 방식으로 작동합니다. 본 연구에서는 통계적 키워드 선택(statistical keyword selection)과 무작위(random) 접근 방식을 비교하고, 합의 품질에 따라 분류를 확장할 수 있는 적응형 투표 메커니즘을 탐구하였습니다.

- **Performance Highlights**: PromptGuard는 마이크로 F1 점수 67.61을 달성하여 n-그램(n-gram) 기준(60.75)과 무작위 접근 방식(14.65)을 초과하는 성능을 보였습니다. 아블레이션(ablative) 연구 결과에 따르면, 카이제곱 기반 키워드는 모든 범주에서 가장 일관된 영향을 미치는 것으로 나타났습니다.



### Gold Panning: Turning Positional Bias into Signal for Multi-Document LLM Reasoning (https://arxiv.org/abs/2510.09770)
Comments:
          20 pages, 6 figures

- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 위치 편향(position bias)을 노이즈가 아닌 진단 신호로 활용하여 정보의 적합성을 효과적으로 식별하는 새로운 방법론인 Gold Panning Bandits를 소개하였습니다. 기존 접근 방식들은 위치 편향을 제거하려고 했지만, 본 연구는 이를 최적화 과정에서 이점으로 변환할 수 있다고 주장합니다. 문서 재순서를 선택하는 문제를 이분 매칭(bipartite matching) 문제로 모델링하고, 이를 통해 정보가 가장 잘 드러나는 문서 배치를 찾아내려 합니다.

- **Technical Details**: Gold Panning Bandits 프레임워크는 문서의 내부 relevance 상태를 알고리즘적으로 탐색하는 과정에서 발생하는 여러 도전 과제를 다루기 위해 설정되었습니다. 신뢰 예측(TPR)과 FPR(오탐률)로 문서의 위치 특성을 분석하고, 재배치를 통해 불확실한 문서를 가장 유의미한 위치에 배치하여 문서의 적합성을 극대화하는 전략을 사용합니다. 최적화 문제는 계산적으로 어려움을 겪기 때문에 Greedy 전략을 사용하여 매우 효율적인 성능을 달성하고 있습니다.

- **Performance Highlights**: Gold Panning 알고리즘은 무작위 재배치 전략에 비해 최대 65% 적은 쿼리를 사용하여 문서를 식별할 수 있음이 입증되었습니다. 이 접근 방식은 문서 검색 작업에서 요구되는 컴퓨팅 비용을 크게 줄일 있으며, 모델 재훈련 없이도 실행됩니다. 나아가, 이 연구는 LLM의 위치 편향을 활용한 최초의 사례로, 알고리즘적 장점을 창출하는 새로운 가능성을 보여주고 있습니다.



### Judge's Verdict: A Comprehensive Analysis of LLM Judge Capability Through Human Agreemen (https://arxiv.org/abs/2510.09738)
Comments:
          10 pages, 1 figure, 4 tables, under review as a conference paper at ICLR 2026

- **What's New**: 이번 연구는 Judge's Verdict Benchmark라는 새로운 평가 방법론을 소개하여, 대형 언어 모델(LLM)을 판사로 활용하여 응답의 정확성을 평가하는 방법을 제시합니다. 54개의 LLM이 생성한 응답을 실제 인간의 판단과 비교하여 얼마나 인간의 판단을 잘 재현할 수 있는지를 평가합니다. 기존의 단순 상관관계 분석에서 벗어나, 실제 동의 패턴을 측정하는 Cohen's Kappa 분석으로 발전한 두 단계의 방법론을 사용합니다.

- **Technical Details**: 두 단계 접근 방식은 (1) 강한 정렬을 가진 판사를 필터링하는 상관관계 테스트와, (2) 인간과 유사한 판단 패턴을 확인하기 위한 z-점수를 사용하는 인간 유사성 테스트로 나뉩니다. 이 방법론을 통해 54개의 LLM 중 27개 모델이 Tier 1 성능을 달성했음을 보여주며, 23개 모델은 인간의 미묘한 판단을 유지하는 인간 유사한 패턴을 보였고, 4개 모델은 비정상적으로 일관된 행동을 나타냈습니다. 또한 LLM의 크기만이 판사 성능과 관련이 없으며, 특정 학습 전략이 중요하다는 사실을 밝혔습니다.

- **Performance Highlights**: 이번 연구는 LLM이 응답 정확도를 평가하는 데 있어 기존의 단순 상관관계가 부족하다는 점을 강조하며, 새로운 기준을 마련했습니다. 우리는 좋은 LLM 판사 성능을 평가하기 위한 표준화된 벤치마크를 제공하고, 인간의 평가자와 LLM 판사를 혼합했을 때 LLM을 적절히 구별할 수 있는지를 평가하는 새로운 'Turing Test for judges'를 도입했습니다. 이를 통해 LLM의 판단 우수성을 검증하기 위한 더 엄격한 프레임워크를 구축했습니다.



### Preference-Aware Memory Update for Long-Term LLM Agents (https://arxiv.org/abs/2510.09720)
- **What's New**: 이번 연구에서는 장기 기억(Long-Term Memory) 메커니즘을 활용한 LLM 기반 에이전트의 추론 능력을 향상시키기 위한 새로운 접근 방식인 Preference-Aware Memory Update Mechanism (PAMU)을 제안합니다. 기존의 메모리 업데이트 방법들이 변화하는 사용자 행동에 적응하지 못하는 문제를 해결하기 위해, PAMU는 사용자의 선호를 동적으로 업데이트하는 기법을 도입하였습니다. 이 방법은 Sliding Window Average (SW)와 Exponential Moving Average (EMA)를 통합하여 단기 및 장기 사용 경향을 모두 포착할 수 있는 방법입니다.

- **Technical Details**: PAMU는 사용자와 모델 간의 다회 대화에서 다차원 선호 신호를 추출하여 사용자 선호 벡터를 구성합니다. 이 벡터는 톤 스타일, 응답 길이, 정서적 톤, 정보 밀도, 형식성 등 다양한 사용자 선호 유형을 반영합니다. 각 대화 회차 후, 시스템은 사용자 피드백과 언어적 특성을 분석하여 선호 벡터를 업데이트하며, 이는 그래디언트 기반의 변화 감지 신호를 통해 메모리 업데이트의 시점을 결정하는 데 활용됩니다.

- **Performance Highlights**: 실험 결과, PAMU는 LoCoMo 데이터셋의 다섯 가지 작업 시나리오에서 기존의 메모리 보강 LLM 모델보다 더욱 향상된 출력 품질을 보여주었습니다. 이는 대화의 맥락과 변화를 보다 효과적으로 반영할 수 있는 능력을 입증하며, 장기 대화에서 LLM의 신뢰성과 효과성을 높이는 데 기여합니다. PAMU는 기존 시스템에 쉽게 통합될 수 있는 모듈형 설계로 되어 있어, 별도의 미세 조정이나 아키텍처 수정 없이도 적용이 가능합니다.



### All Code, No Thought: Current Language Models Struggle to Reason in Ciphered Languag (https://arxiv.org/abs/2510.09714)
- **What's New**: 이번 연구는 AI 시스템의 체계적 예방 위험을 평가하기 위해, 28가지 서로 다른 암호화 방식에서 AI 모델이 수행할 수 있는 복잡한 이유가 숨겨진 메커니즘인 'ciphered reasoning'에 대한 첫 번째 상세 연구를 제시합니다. 연구자들은 10개의 모델을 조정하여 이러한 암호화된 텍스트에서의 추론 능력을 평가하였으며, 모델들은 암호화된 텍스트를 잘 이해하면서도 정확한 이유를 제공하는 데 어려움을 겪는 비대칭성을 발견했습니다.

- **Technical Details**: 이 논문에서는 'ciphered reasoning' 능력을 두 가지 측면에서 평가합니다: 첫째, 암호화된 텍스트에서 문제 해결 성능이 향상되는지 판단하는 'ciphered reasoning capability', 둘째, 암호화된 텍스트를 영어로 해독하는 능력을 포함하는 'cipher translation capability'입니다. 연구 결과, 암호화된 텍스트에서 제대로 이유를 제공하기 위해서는 방대한 양의 학습 데이터가 필요하며, 잘 알려진 암호에서는 높은 정확도를 보이는 반면, 덜 알려진 암호에서는 큰 성능 저하가 발생하는 것을 확인했습니다.

- **Performance Highlights**: 모델의 정확도는 암호화된 텍스트에서의 추론 능력과 상관관계가 있었으며, 훈련 데이터의 암호화된 텍스트 발생 빈도가 높을수록 정확도가 증가하는 경향을 보였습니다. 추가적인 미세 조정 데이터가 이루어져도 암호화된 추론 능력의 향상은 더디며, 단순한 암호 체계에서 3.7B 토큰 이상의 데이터가 필요하다는 결과가 도출되었습니다. 이러한 발견은 현재의 모델이 암호화된 텍스트로 CoT 모니터링을 피하는 것이 비효율적이라는 점을 제시합니다.



### ReaLM: Residual Quantization Bridging Knowledge Graph Embeddings and Large Language Models (https://arxiv.org/abs/2510.09711)
- **What's New**: 이 연구에서는 ReaLM이라는 혁신적인 프레임워크를 제안하여 Knowledge Graph (KG) 임베딩과 대형 언어 모델 (LLM) 토크나이제이션 간의 간극을 메우고자 하였습니다. ReaLM은 고차원 KG 임베딩을 컴팩트한 코드 시퀀스로 변환하고 이를 LLM의 어휘 내에서 학습 가능한 토큰으로 통합함으로써 상징적 지식과 맥락적 지식을 매끄럽게 융합합니다.

- **Technical Details**: ReaLM은 잔여 벡터 양자화 (residual vector quantization) 메커니즘을 통해 KG 임베딩을 이산화하여 LLM의 제한된 토큰 공간에 맞추는 새로운 방식입니다. 이 방법은 KG 임베딩의 관계적 구조와 의미적 뉘앙스를 보존하면서 LLM의 내부 매개변수와 함께 최적화되며, 생성된 토큰이 KG에서 정의된 개체와 정확히 일치하도록 보장합니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, ReaLM은 FB15K237 및 WN18RR 와 같은 기준 데이터 세트에서 기존 LLM 기반 및 전통적인 KG 완성 방법보다 월등한 성능을 달성했습니다. 이러한 실험 결과는 ReaLM이 구조적 지식을 대규모 언어 모델과 효과적으로 정렬하는 데 있어 그 유효성을 확인해줍니다.



### SeCon-RAG: A Two-Stage Semantic Filtering and Conflict-Free Framework for Trustworthy RAG (https://arxiv.org/abs/2510.09710)
Comments:
          Accepted at NeurIPS 2025

- **What's New**: 이번 논문에서는 RAG (Retrieval-Augmented Generation) 시스템이 대규모 언어 모델(LLM)의 외부 지식으로 강화되지만, 코퍼스 오염(corpus poisoning) 공격에 취약하다는 점을 지적합니다. 제안된 두 단계의 세멘틱 필터링(semantic filtering) 및 갈등 방지(conflict-free) 프레임워크인 SeCon-RAG는 정보를 효과적으로 보존하면서도 공격에 강한 모델을 구현합니다. 특히, EIRE(Entity-intent-relation extractor)를 사용하여 문서의 유용한 정보를 선별하고, 최종 답변 생성을 위한 갈등 인식 필터링 모듈을 도입하여 신뢰성을 높입니다.

- **Technical Details**: 제안된 SeCon-RAG 시스템은 두 가지 단계가 있습니다. 첫째, EIRE를 통해 엔티티, 잠재적 목표 및 엔티티 관계를 추출하고, 이 정보를 바탕으로 의미적 관련성을 점수화합니다. 둘째, EIRE 가이드를 따르는 갈등 인식 필터링 모듈이 쿼리와 후보 답변 간의 의미적 일관성을 분석하여 내부 및 외부 모순을 걸러내면서 최종 답변 생성 과정을 진행합니다.

- **Performance Highlights**: SeCon-RAG는 다양한 LLM 및 데이터셋에서 실험을 통해 기존의 최첨단 방어 방법들보다 뛰어난 성능을 발휘했습니다. 예를 들어, Mistral-12B 모델에서는 100% 오염 상황에서 75.7%의 정확도를 보였으며, 2.4%의 공격 성공률(ASR)을 기록하여 그 견고성을 입증했습니다. 더 작은 모델들에서도 전반적으로 높은 성능을 유지하며, 각 오염 수준에서 모든 기초 모델들보다 뛰어난 결과를 보였습니다.



### The Idola Tribus of AI: Large Language Models tend to perceive order where none exists (https://arxiv.org/abs/2510.09709)
Comments:
          14 pages, 3 figures, accepted to Findings of EMNLP 2025

- **What's New**: 본 연구에서는 큰 언어 모델(LLMs)이 숫자 시퀀스의 규칙성을 식별하는 단순한 작업에서도 비합리적인 패턴을 생성하는 경향을 보인다는 점을 강조합니다. LLMs는 복잡한 실제 작업을 수행하기 위해 다양한 접근 방식이 제안되고 있지만, 논리적 일관성과 자기 일관성을 평가하는 것이 중요하다는 점을 강조합니다. Consequently, LLMs가 규칙성을 파악하는 실험을 통해 이들이 비합리적인 패턴을 과대 인식하는 경우를 탐구하였습니다.

- **Technical Details**: 본 연구의 실험에서는 LLMs가 산술 및 기하급수 시퀀스에서 올바른 패턴을 효율적으로 식별하는 반면, 무작위 생성된 시리즈를 분석할 때는 주어진 수와 일치하지 않는 비합리적인 패턴을 빈번하게 과대 인식하는 경향이 있음을 보여줍니다. 이를 통해 LLMs의 논리적 사고 과정 및 정보의 추상화 능력이 평가됩니다. 실험 결과는 LLMs가 학습 데이터를 바탕으로 하여도 잘못된 결론을 도출할 수 있음을 나타내며, 이는 체계적인 문제 해결 과정에서의 중요한 약점입니다.

- **Performance Highlights**: 실험 결과, OpenAI o3, o4-mini와 Google Gemini 2.5 등 여러 최신 LLM 모델에서도 자연수 시퀀스에서 비합리적인 패턴을 인식하는 경향이 공통적으로 발생하는 것을 확인했습니다. 이러한 경향은 LLM의 적용 가능한 작업의 성능에重大한 영향을 미칠 수 있습니다. 특히, 이 연구는 LLMs의 자기 일관성 및 논리적 사고의 한계를 드러내며, 미래의 개선 방향에 중요한 통찰을 제공합니다.



### Emotionally Charged, Logically Blurred: AI-driven Emotional Framing Impairs Human Fallacy Detection (https://arxiv.org/abs/2510.09695)
Comments:
          Initial submission

- **What's New**: 이번 연구는 감정적 프레이밍(emotional framing)과 오류(fallacy) 탐지가 설득력(convincingness)에 미치는 상호작용을 다룬 최초의 컴퓨터 연구로, 대형 언어 모델(LLMs)을 이용하여 오류가 있는 주장에서 감정적 호소를 체계적으로 변화시키는 방법을 제시합니다. 연구 결과, LLM에 의해 형성된 감정적 프레이밍이 인간의 오류 탐지 능력을 평균 14.5% 감소시키는 것으로 나타났습니다. 또한, 인간은 즐거운 감정을 느낄 때 오류 탐지 성능이 더 높고, 이는 부정적인 감정보다 더 높은 설득력을 가지고 있는 것으로 보입니다.

- **Technical Details**: 이 연구는 감정적 요소가 논증(argumentation)의 독립적인 차원으로 작용한다는 관점을 바탕으로 하며, LLM을 이용해 여섯 가지 감정을 주입하는 여러 프레이밍 전략을 통해 논리 구조를 유지하면서 감정 표현을 효과적으로 삽입하는 기법을 실험합니다. 여덟 개의 LLM 모델이 평가되었으며, 최상의 모델을 선정하여 1,000개의 주장을 생성하여 인간 연구를 위한 자극으로 사용했습니다. 감정이 포함된 주장에 대한 인간의 반응을 평가하여 오류 탐지와 설득력 판단을 비교 분석했습니다.

- **Performance Highlights**: 연구 결과, 인간은 LLM이 프레이밍한 감정이 표현된 주장에 노출되었을 때 오류 탐지 성능이 14% 감소하였으나, 감정적인 주장은 원래 주장에 비해 약간 더 설득력이 높게 평가되었습니다. 특히, 인지된 감정이 즐거움일 때 오류 탐지 성능이 가장 높았고, 상대적으로 슬픔이나 두려움을 느낄 때는 성능이 저하되었습니다. 이러한 결과는 AI가 논리적인 오류를 포함한 주장에서 인간의 판단에 미치는 위험과 감정적 조작의 가능성을 시사합니다.



### Table Question Answering in the Era of Large Language Models: A Comprehensive Survey of Tasks, Methods, and Evaluation (https://arxiv.org/abs/2510.09671)
- **What's New**: 이 논문은 테이블 질문 답변(Table Question Answering, TQA) 분야의 최근 발전을 정리하고 있으며, 특히 대형 언어 모델(large language models, LLMs)에 기반한 방법론에 중점을 두고 있습니다. 기존 문헌에서는 TQA의 다양한 작업 설정 및 핵심 과제가 체계적으로 정리되지 않았으므로, 이 조사는 그러한 공백을 메우기 위해 포괄적이고 구조화된 개요를 제공합니다. 또한 TQA 연구의 최신 동향을 반영하며, 강화 학습(reinforcement learning)과 같은 새로운 연구 방향에 대한 통찰력을 제공합니다.

- **Technical Details**: TQA는 자연어 질문을 기반으로 표형 데이터에서 답변을 도출하는 작업으로, 표의 포맷, 크기, 구조적 복잡성, 질문 및 답변의 복잡성에 따라 다양한 설정에서 이루어질 수 있습니다. 이 논문에서는 TQA 작업을 다섯 가지 관점에서 분해하고, 계층적(tiered) 및 평면(flat) 구조의 표, 단일 및 다중 표와 관련된 특징을 분석합니다. TQA 질문의 유형에 따라서도 질의 응답 방식이 다르며, 질문 복잡도에 따라 간단 또는 복잡하게 구성할 수 있습니다.

- **Performance Highlights**: 이 조사는 2022년 이후 발표된 215개의 관련 논문을 검토하여 TQA 작업과 벤치마크를 상세히 소개합니다. TQA의 다양한 작업 설정을 설명하고, 각 작업의 장점과 한계를 분석합니다. 이 연구는 다양한 연구 경향을 통합하여 TQA 커뮤니티에 기초 연구 자료를 제공하고, 향후 연구 방향을 제시하여 이 분야의 진보를 도울 수 있는 토대를 마련합니다.



### Boundary-Guided Policy Optimization for Memory-efficient RL of Diffusion Large Language Models (https://arxiv.org/abs/2510.11683)
- **What's New**: 최근 확산 대형 언어 모델(dLLMs)은 기존의 자기회귀 모델(ARMs)에 대한 유망한 대안으로 부각되고 있으며, 다양한 언어 모델링 작업에서 경쟁력 있는 성능을 보여주고 있습니다. 그러나 기존의 연구들은 주로 dLLMs의 사전 학습과 감독 학습에 초점을 맞추고 있으며, 강화 학습(RL)을 이용한 dLLMs의 성능 개선은 여전히 도전 과제로 남아 있습니다. 본 연구에서는 Boundary-Guided Policy Optimization (BGPO)이라는 새로운 메모리 효율적인 RL 알고리즘을 제안하여, dLLMs에 대한 log-likelihood와 RL 목표의 근사를 지원합니다.

- **Technical Details**: BGPO는 ELBO 기반 목표의 하한을 최대화하도록 설계되었습니다. 이 하한은 두 가지 주요 속성을 만족하도록 만들어졌습니다: (1) 선형성(Linearity): 각 항이 단일 MC 샘플에만 의존하는 형태로 구성되어 있어, 샘플 간의 그래디언트 누적이 가능하고 메모리 사용이 일정하게 유지됩니다; (2) 동등성(Equivalence): 이 하한의 값과 그래디언트는 on-policy 훈련에서 ELBO 기반 목표의 값과 그래디언트가 같아, 원래의 RL 목표를 효과적으로 근사할 수 있게 됩니다. 이러한 특성 덕분에 BGPO는 큰 MC 샘플 크기를 채택하여 보다 정확한 RL 목표 근사가 가능해집니다.

- **Performance Highlights**: BGPO는 LLaDA-8B-Instruct 모델을 사용한 수학 문제 해결, 코드 생성 및 계획 작업에서 이전 RL 알고리즘과 비교해 상당한 성능 향상을 보여줍니다. 범위가 넓은 MC 샘플 크기를 활용함으로써 그래디언트의 편향과 분산을 효과적으로 줄여 모델 성능을 향상시키는 결과를 도출했습니다. 또한, BGPO는 샘플 크기가 증가하더라도 평균 훈련 단계 시간이 소폭만 증가하여 효율성을 유지했습니다.



### FinVet: A Collaborative Framework of RAG and External Fact-Checking Agents for Financial Misinformation Detection (https://arxiv.org/abs/2510.11654)
- **What's New**: 본 논문에서는 FinVet이라는 새로운 멀티 에이전트 프레임워크를 소개하며, 이는 두 개의 Retrieval-Augmented Generation (RAG) 파이프라인과 외부 사실 확인을 통합하는 신뢰도 가중 투표 메커니즘을 통해 작동합니다. FinVet은 동적으로 검증 전략을 조정하는 세 가지 처리를 통해 기존 방법의 한계를 극복하는 동시에 더 높은 투명성을 제공합니다. 이를 통해 증거 기반의 판결, 출처 귀속, 신뢰도 점수를 제공하며, 불충분한 증거에 대한 불확실성 표시를 명확히 하게 됩니다.

- **Technical Details**: FinVet은 세 가지 계층 처리 전략을 도입하여 신뢰도 점수에 따라 검증 접근 방식을 동적으로 선택합니다. 이는 높은 신뢰도의 경우 직접 메타데이터 추출, 중간 신뢰도의 경우 하이브리드 모델 추론, 낮은 신뢰도의 경우 모델 기반 분석을 통해 이루어집니다. RAG 구성요소는 도메인 특정 데이터셋에 맞춘 외부 지식소스를 활용하며, 사실 확인 파이프라인은 직접적인 증거가 없을 경우 대체 메커니즘을 사용하여 처리합니다.

- **Performance Highlights**: FinVet의 성능 평가 결과는 FinFact 데이터셋을 기준으로 하여 F1 점수 0.85를 달성하였으며, 이는 기존의 최고 단일 파이프라인(사실 확인 파이프라인) 대비 10.4% 개선된 결과입니다. 또한, 스탠드얼론 RAG 접근법에 비해 37%의 성능 향상도 보여주었습니다. 이 결과는 FinVet이 기존 기술과 비교할 때 보다 나은 정확성과 설명 가능성을 제공함을 입증합니다.



### REGENT: Relevance-Guided Attention for Entity-Aware Multi-Vector Neural Re-Ranking (https://arxiv.org/abs/2510.11592)
Comments:
          To be published in: Proceedings of the 2025 Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region (SIGIR-AP 2025)

- **What's New**: 현재 신경 재순위를 수행하는 모델들은 복잡한 정보 요구와 긴 내용이 풍부한 문서에서 어려움을 겪고 있다. 본 논문에서 소개하는 REGENT는 이러한 문제를 해결하기 위해 인간처럼 이해하는 방식으로 개체를 '시맨틱 스켈레톤'(semantic skeleton)으로 사용하여 주의를 유도한다. REGENT는 관련성 지도를 주의 메커니즘에 직접 통합하여 정교한 어휘 일치와 고급 의미적 추론을 결합하며, 이를 통해 복잡한 질문에서도 중요한 내용을 집중적으로 처리한다.

- **Technical Details**: REGENT는 문서 토큰을 나타내는 정밀 벡터와 의미적 개체를 나타내는 고급 벡터를 병렬로 처리하는 다중 벡터 아키텍처로 구성된다. 이 모델은 토큰 수준에서 BM25 점수를 사용하여 어휘적으로 관련 있는 용어를 강조하고, 쿼리-특정 개체 표현을 사용하여 의미상 중요한 개념에 집중한다. 결과적으로 REGENT는 긴 문서 내에서 가장 관련성 높은 내용을 지향하며, 효율적인 개체-기반 정보 검색을 위한 새로운 패러다임을 제시한다.

- **Performance Highlights**: REGENT는 세 가지 대규모 데이터세트에서 새로운 최첨단 성능을 달성했으며, BM25에 비해 최대 108% 향상된 성능을 보인다. 또한, 기존의 강력한 모델인 ColBERT 및 RankVicuna를 지속적으로 초월한다. 중요한 점은 모델에서 개체 컴포넌트를 제거할 경우 성능이 74% 감소한다는 점으로, 이는 의미적 스켈레톤이 효율적인 장기 문서 검색에 매우 중요하다는 것을 강조한다.



### QDER: Query-Specific Document and Entity Representations for Multi-Vector Document Re-Ranking (https://arxiv.org/abs/2510.11589)
Comments:
          Published in: Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2025)

- **What's New**: 이번 논문에서는 QDER라는 새로운 신경 재정렬 모델을 소개하며, 이 모델은 지식 그래프(knowledge graph) 의미론을 다중 벡터 모델에 통합하여 엔터티가 인식되는 검색(entity-aware retrieval)을 개선하는 데 중점을 둡니다. QDER의 주요 혁신은 쿼리-문서 관계를 모델링하는 새로운 접근법으로, 집계된 임베딩에서의 유사도 점수를 계산하는 대신, 각 토큰과 엔터티 표현을 유지하며 최종 점수 매기기에서만 집계를 수행합니다. 이를 'late aggregation'이라고 부르며, QDER는 복잡한 쿼리를 효과적으로 처리할 수 있는 기반을 다집니다.

- **Technical Details**: QDER는 동적 주의(attention) 메커니즘을 적용하여 쿼리와 관련된 문서의 토큰 및 엔터티 표현을 변환하며, 이로 인해 쿼리에 맞춤화된 표현을 생성합니다. 또한, 전통적인 유사도 지표 대신 이차선형 프로젝션(bilinear projection)을 사용해 복잡한 쿼리-문서 관계를 모델링하며, 이는 노이즈에 대한 강인성을 높입니다. QDER의 문서 표현은 쿼리 별로 조정된 방식으로 생성되며, 각기 다른 조작인 더하기(addition)와 곱하기(multiplication)를 통해 보다 정밀한 매칭을 구현합니다.

- **Performance Highlights**: 실험 결과, QDER는 TREC Robust 2004 데이터세트에서 가장 강력한 기준선 대비 nDCG@20에서 36% 향상된 성능을 보여줍니다. 특히, 어려운 쿼리에서 QDER는 nDCG@20을 0.70으로 달성하며, 이는 기존의 전통적인 방법들이 실패하는 경우(nDCG@20 = 0.0)에서 두드러진 성과입니다. 이러한 결과는 QDER가 엔터티 인식을 통한 정보 검색에서의 새로운 기준을 세우는 데 기여함을 보여줍니다.



### Bag of Tricks for Subverting Reasoning-based Safety Guardrails (https://arxiv.org/abs/2510.11570)
Comments:
          OpenAI Red-teaming Challenge Winner and Oral Presentation

- **What's New**: 최근 대규모 추론 모델(Large Reasoning Models, LRMs)을 위한 안전 가드레일이 설치되었으며, 이들은 사용자 입력의 안전성을 평가하기 통해 더 강력한 방어력을 보입니다. 특히, 새로운 변별 정렬(Deliberative Alignment) 방법은 모델이 안전 정책을 명확히 이해하고 이를 바탕으로 응답을 생성하는 데 도움을 줍니다. 그러나 이러한 로직 기반 안전 가드레일이 미세한 입력 조작에 극도로 취약하다는 점이 발견되었습니다.

- **Technical Details**: 연구에서는 사용자의 프롬프트에 몇 가지 템플릿 토큰을 추가하는 것만으로도 강력한 가드레일을 우회하고 유해한 응답을 이끌어낼 수 있음을 확인했습니다. 제안하는 해킹 방법들은 White-, Gray-, Black-box 환경에서 적용 가능하며, 템플릿 조작부터 자동화된 최적화 기법까지 다양합니다. 특히, 이들 해킹 기술은 90% 이상의 높은 공격 성공률을 기록하여 심각한 위험성을 부각시킵니다.

- **Performance Highlights**: 이 연구는 다양한 오픈소스 LRM에서 이들 취약점이 시스템적으로 존재한다는 사실을 입증하였으며, 이러한 모델들의 악용 가능성을 강조합니다. 연구에 포함된 네 가지 해킹 기술은 안전 가드레일을 우회하고 유해한 응답을 유도하며, 이는 최근의 기술적 진보에도 불구하고 기본적인 취약점이 여전히 존재함을 보여줍니다. 따라서, 더욱 강력한 안전 방어를 위한 긴급한 필요성이 대두되고 있습니다.



### ReLook: Vision-Grounded RL with a Multimodal LLM Critic for Agentic Web Coding (https://arxiv.org/abs/2510.11498)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)이 알고리즘 코드 생성에서는 우수하지만 프론트엔드 개발에서는 어려움을 겪고 있다는 점을 지적합니다. 이를 해결하기 위해, ReLook라는 비전 기반 강화학습 프레임워크를 도입하여 에이전트가 다중 모드 LLM(MLLM)을 도구로 활용하여 코드를 생성하고 진단 및 개선하는 반복 작업을 수행할 수 있도록 합니다. 특히, 이 방법은 프론트엔드 코드 생성의 비전-기반 성능을 크게 향상시키는 것을 목표로 합니다.

- **Technical Details**: ReLook는 MLLM을 비주얼 비평가로 사용하여 스크린샷을 통한 코드 점수 평가와 실행 가능한 비전-기반 피드백을 제공합니다. 훈련 과정에서 에이전트는 0 보상 규칙을 적용해 잘못된 렌더링에 대해 보상을 받지 않게 하여 렌더링 가능성을 보장합니다. 또한, 강제 최적화(Forced Optimization) 전략을 통해 성능 저하를 방지하고 지속적으로 향상된 경로를 유지합니다.

- **Performance Highlights**: ReLook는 세 가지 일반적인 벤치마크 테스트에서 기존 방법보다 우수한 성능을 보여줍니다. 다양한 LLM과의 통합 실험을 통해 ReLook의 호환성을 입증하였으며, 에이전트의 인식 능력 및 비주얼 보상의 효과를 강조합니다. 이러한 성과는 에이전트가 적절한 피드백을 기반으로 코드를 지속적으로 개선할 수 있도록 하는 강력한 학습 메커니즘 덕분입니다.



### Beyond the Crowd: LLM-Augmented Community Notes for Governing Health Misinformation (https://arxiv.org/abs/2510.11423)
- **What's New**: 이번 연구는 X(구 Twitter)에서 커뮤니티 노트 시스템을 통해 건강 관련 허위정보를 더욱 신속하고 효율적으로 관리하기 위한 새로운 프레임워크인 CrowdNotes+를 제안합니다. CrowdNotes+는 대규모 언어 모델(LLMs)을 활용하여 두 가지 생성 모드인 증거 기반 노트 증강(Evidence-Grounded Note Augmentation)과 유용성 안내 노트 자동화(Utility-Guided Note Automation)를 통합합니다. 이를 통해 실제 허위 정보의 발산이 빠른 상황에서 타이밍을 개선하고 정보의 신뢰성을 높이는 효과를 기대할 수 있습니다.

- **Technical Details**: CrowdNotes+는 세 가지 단계의 평가 프로세스를 기반으로 하여 노트의 적합성, 정확성 및 유용성을 점진적으로 평가합니다. 연구에서는 1.2K 개의 건강 문서화된 커뮤니티 노트와 함께 건강 평가자(HealthJudge)를 통해 프레임워크를 구현하였습니다. 실험 결과, 기존의 인간 평가에서 발생했던 오류를 줄이고, 생성된 노트의 전반적인 정확성과 맥락적 균형이 향상됨을 입증했습니다.

- **Performance Highlights**: CrowdNotes+ 프레임워크는 커뮤니티 노트의 유용성을 높이고, 인간 작성 노트와 비교해 더욱 정확하고 적절한 정보를 제공합니다. 대규모 언어 모델은 인간 기여자보다 질 높은 증거를 선택하여 활용하며, 연구 결과는 하이브리드 인간-AI 거버넌스 모델의 가능성을 보여줍니다. 이러한 결과는 허위 정보 처리의 엄밀성과 시의성을 동시에 개선할 수 있는 새로운 길을 열어줍니다.



### Diffusion-Link: Diffusion Probabilistic Model for Bridging the Audio-Text Modality Gap (https://arxiv.org/abs/2510.11330)
Comments:
          5 pages. Submitted to IEEE ICASSP 2026

- **What's New**: 이 논문은 Diffusion-Link라는 새로운 모듈을 제안하여 오디오 임베딩을 텍스트 임베딩 분포로 생성적으로 매핑합니다. 이 모듈은 동기식 네트워크로 구성되어 있으며, 고정된 멀티모달 인코더의 출력 임베딩에서 학습됩니다. 특히, 자동 오디오 캡셔닝(Automatic Audio Captioning, AAC)에 처음으로 확산 기반 모듈을 적용한 사례로 주목받고 있습니다.

- **Technical Details**: Diffusion-Link는 세 개의 잔여 다층 퍼셉트론(Residual MLP) 블록으로 구성된 경량 네트워크입니다. 이 모듈은 오디오-텍스트 임베딩 쌍을 사용하여 두 분포를 명시적으로 연결하고, 역 과정을 통해 텍스트 임베딩 분포로 매핑하는 방식으로 작동합니다. 특히, 정규화된 가우시안 노이즈를 주입하여 오디오 임베딩의 구조를 유지하면서 효과적인 모달리티 브리지를 구현합니다.

- **Performance Highlights**: Diffusion-Link를 멀티모달 LLM 베이스라인에 추가하는 방식으로, AudioCaps 데이터셋에서 제로샷 오디오 캡셔닝의 성과가 52.5% 향상되고, 완전히 감독된 캡셔닝에서도 7.5% 향상을 보여주며, 이는 외부 지식 없이는 도달할 수 없었던 최첨단 결과입니다. 이 연구는 모달리티 갭을 줄이는 것이 효과적인 멀티모달 인코더와 LLM 간 coupling을 위해 필수적임을 보여주고, 확산 기반 모달리티 브리지가 새로운 방향성을 제공한다는 점에서 중요한 의의를 갖습니다.



### ENIGMA: The Geometry of Reasoning and Alignment in Large-Language Models (https://arxiv.org/abs/2510.11278)
Comments:
          52 pages, 10 figures

- **What's New**: 이번 논문에서는 ENIGMA(Entropy Mutual-Information Geometry Large-Language Model Alignment)라는 새로운 접근 방식을 통해 LLM 훈련의 추론, 정렬, 강인성을 향상시키는 방법을 제시합니다. 조직의 정책과 원칙을 모델의 정보 메니폴드에서의 운동 방향으로 간주하여 이를 훈련 신호와 측정 방법에 직접 적용하는 방식을 제안합니다. ENIGMA는 여러 기법을 통합해 설계된 단일 루프 훈련기법을 사용해, 외부 보상 모델 없이도 원칙이 인코딩된 추론 체인을 끌어내는 데 초점을 맞춥니다.

- **Technical Details**: ENIGMA는 Group-Relative Policy Optimisation (GRPO), Self-Supervised Alignment with Mutual Information (SAMI) 및 Sinkhorn divergence를 활용하는 새로운 훈련 방법을 도입합니다. 이 방법은 정보 기하학적 목표에 대한 효과적인 측정을 위한 수량적 지표를 개발하고, 원칙의 선택과 훈련 동역학에 미치는 영향을 정량화하기 위한 Sufficiency Index (SI)를 포함합니다. 본 연구에서는 또한 성능 향상을 예측하는 여러 메트릭을 제안하여 훈련 동역학을 포괄적으로 분석합니다.

- **Performance Highlights**: ENIGMA를 통해 훈련된 모델들은 정렬과 추론 벤치마크에서 향상된 성능을 보였으며, 특히 GPQA에서는 +6.92포인트, TruthfulQA에서는 +12.11포인트의 성과 향상이 나타났습니다. 실험 결과는 원칙에 의해 구조적으로 변화된 모델을 확인할 수 있었고, 이러한 증거들은 추론, 정렬 및 강인성이 단일 정보 기하학적 목표의 투영임을 지지합니다. ENIGMA 접근 방법은 조직이 정의한 원칙과 기준을 사용하여 LLM의 행동과 출력 간의 관계를 정량적으로 설명할 수 있는 가능성을 제공합니다.



### Can Tool-Integrated Reinforcement Learning Generalize Across Diverse Domains? (https://arxiv.org/abs/2510.11184)
- **What's New**: 최근 대형 언어 모델(LLMs)의 발전은 추론(reasoning) 및 도구(tool) 활용에서 놀라운 능력을 보여주고 있습니다. 그러나 다양한 분야에서 도구 보강 강화 학습(tool-augmented reinforcement learning, RL)의 일반화는 아직 충분히 탐구되지 않았습니다. 본 연구에서는 수학 문제 해결에서 훈련된 LLM 에이전트가 코드 인터프리터(tool)의 도움을 받아 다양한 추론 분야에서 어떻게 성능을 발휘하는지를 조사합니다.

- **Technical Details**: 연구에서는 수학 영역에서의 RL을 통해 도구 호출 전략을 학습한 후 여러 독립적인 분야에서 평가하는 방식입니다. 이를 통해 도구 사용의 일반화 가능성을 검토하며, TGRL(Tool Generalization Reinforcement Learning) 프레임워크를 제안합니다. TGRL은 표준화된 도구 인터페이스, 이중 보상 시스템, XML 기반 프롬프트 템플릿을 활용하여 도메인에 구애받지 않는 학습과 기술 이전(skill migration)을 촉진하는 구조입니다.

- **Performance Highlights**: 광범위한 벤치마크를 통한 실험은 제안된 접근 방식이 최첨단 성능을 달성했음을 보여줍니다. 수학 문제에서 배운 도구 사용이 복잡한 다른 분야의 작업에 효과적으로 이전될 수 있음을 입증했습니다. 또한 정량적 작업 수행 및 높은 토큰 효율성을 바탕으로 하여 Tool RL의 잠재력을 강조합니다.



### EAGER: Entropy-Aware GEneRation for Adaptive Inference-Time Scaling (https://arxiv.org/abs/2510.11170)
- **What's New**: EAGer는 토큰 수준의 엔트로피 분포를 활용하여 중복 계산을 줄이고 전반적인 성능을 개선하는 새로운 접근 방식을 제안합니다. 이 방법은 높은 엔트로피 토큰에서만 여러 가지 추론 경로로 분기할 수 있게 하여, 같은 지침(prompt)에 대해 유사한 계산 비용을 할당하는 문제를 해결합니다. EAGer는 기존의 전통적인 방법들보다 더 효율적이며, 특히 복잡한 문제에서 예외적인 성능 향상을 보여줍니다.

- **Technical Details**: EAGer는 추론 과정에서 모델의 불확실성을 모니터링하여 계산 리소스의 효율적 할당을 가능하게 합니다. 특히, 높은 엔트로피 값을 가진 토큰에서만 새로운 병렬 추론 경로를 시작하여, 예측이 안정적일 때는 후보 시퀀스를 적게 생성하고, 불확실성이 클 때는 추가 탐색을 요구하는 방식입니다. 이를 통해 더 많은 계산 리소스를 복잡한 문제에 집중할 수 있습니다.

- **Performance Highlights**: EAGer는 AIME 2025와 같은 복잡한 추론 벤치마크에서 최대 37%의 성능 향상을 달성하며, 생성되는 토큰 수를 최대 65% 줄입니다. 다양한 오픈 소스 모델에 대해 실험한 결과, EAGer를 사용한 경우 연산 비용을 80%까지 절감하면서도 성능이 개선되는 것을 확인했습니다. 이로 인해 EAGer는 추론 과정에서의 효율성 및 성능 간의 최적의 균형을 제공합니다.



### ELMO: Efficiency via Low-precision and Peak Memory Optimization in Large Output Spaces (https://arxiv.org/abs/2510.11168)
Comments:
          Accepted to ICML 2025

- **What's New**: 이번 논문에서는 Extreme Multilabel Classification (XMC)에 대한 새로운 저정밀도 훈련 프레임워크인 ELMO를 제안합니다. ELMO는 BFloat16과 Float8 데이터 타입을 사용하여 순수 저정밀도 훈련을 통해 큰 출력 공간에서 효과적인 모델 훈련을 가능하게 합니다. 저정밀도 훈련을 통해 GPU 메모리 사용량을 획기적으로 줄일 수 있으며, 3백만 개 레이블의 모델을 6.6 GiB의 메모리로 훈련할 수 있습니다.

- **Technical Details**: ELMO는 Kahan summation과 stochastic rounding 기법을 활용하여 Float8 데이터 타입만으로 모델을 훈련할 수 있는 가능성을 보여줍니다. 이러한 접근 방식은 딥러닝에서의 메모리와 계산 요구 사항을 줄이기 위해 개발된 것으로, GF16에서 BF16으로의 전환과 그라디언트 통합 전략을 통해 이루어집니다. 우리의 방법은 모델 훈련의 메모리 요구량을 50-75%까지 줄일 수 있도록 돕습니다.

- **Performance Highlights**: 여러 개의 레이블 크기에 대해 ELMO의 저정밀도 훈련 방법을 평가한 결과, 기존의 SOTA 방법과 비슷한 성능을 나타냅니다. 또한, LF-Paper2Keywords-8.6M이라는 8.6백만 레이블을 가진 새로운 데이터셋을 소개하여, 현재 공개된 XMC 벤치마크 중 가장 큰 데이터셋임을 주장합니다. 저정밀도 훈련은 XMC 분야에서 더욱 더 중요한 기준으로 자리잡을 가능성이 높습니다.



### $How^{2}$: How to learn from procedural How-to questions (https://arxiv.org/abs/2510.11144)
- **What's New**: 이번 논문에서는 How^{2}라는 메모리 에이전트 프레임워크를 제안하며, 이를 통해 에이전트가 어떻게 질문을 하고 그에 대한 답변을 저장하며 평생 학습을 진행할 수 있는 방법을 구현합니다. Plancraft라는 Minecraft 제작 환경에서 평가를 진행하여, 에이전트가 자원을 조작하여 조립 작업을 완료하는 방식으로 구현하였습니다. 우리의 연구는 에이전트가 저수준의 실행 가능 행동에서부터 고수준의 하위 목표 설명까지 다양한 수준의 추상화된 답변을 통해 학습할 수 있음을 보여줍니다.

- **Technical Details**: How^{2} 프레임워크는 다양한 작동 역할을 가지며, 에이전트가 대화 기록을 바탕으로 다음 행동을 결정합니다. 메모리는 쿼리에 대한 답변을 캐시하는 변화 가능한 키-값 저장소이며, 질문에 대한 답변은 정밀 문자열 일치를 기반으로 검색됩니다. Plancraft 환경에서 에이전트는 재료를 조작하며 실행 가능한 작업의 스트림을 생성하는 동시에, 고수준의 질문을 통해 지식을 축적합니다.

- **Performance Highlights**: 우리의 접근법은 즉각적인 작업 성공을 위한 직접적인 실행 가능한 행동보다, 하위 목표 또는 추상화된 답변이 평생 학습에는 더 유익함을 보여줍니다. 이는 에이전트가 장기적으로 플래닝 능력을 개선하는 데 기여하며, 교사 모델의 응답이 높은 수준으로 추상화되어 있어야 효과적임을 알 수 있습니다. 따라서 How^{2}는 LLM 기반 에이전트가 상호작용 환경에서 질문을 통해 시간이 지나도 개선될 수 있는 경로를 제공합니다.



### VCB Bench: An Evaluation Benchmark for Audio-Grounded Large Language Model Conversational Agents (https://arxiv.org/abs/2510.11098)
Comments:
          20 pages, 5 figures

- **What's New**: 본 논문에서는 음성 언어 모델(large audio language models, LALMs)의 평가 도구로 중국어 기반의 새로운 벤치마크인 Voice Chat Bot Bench (VCB Bench)를 소개합니다. 기존의 벤치마크는 주로 영어에 집중되어 있고 합성 음성에 의존하는 한계를 가지고 있습니다. VCB Bench는 진짜 인간 음성을 기반으로 하여 세 가지 주요 평가 측면인 지시 준수(instruction following), 지식 이해(knowledge understanding), 강건성(robustness)을 통합적으로 평가합니다.

- **Technical Details**: VCB Bench는 12개 과목에 대한 일반 지식과 논리적 추론을 포함한 다양한 평가 항목을 제공합니다. 각 항목은 자연어 이해와 생성 성능 향상을 위한 커스텀 작업으로 구성됩니다. 또, 사용자가 요구 사항에 맞게 조절할 수 있도록 음성 수준의 조작 기능(예: 감정 조절, 음량 조절)을 포함하여 중국어와 영어를 동시에 지원합니다.

- **Performance Highlights**: VCB Bench에 대한 실험을 통해 현재 LALMs의 성능 차이를 발견했습니다. 이를 통해 기술적 한계 및 향후 발전 방향을 제시하며, 실제 인간 대화의 다양한 시나리오에서 LALMs의 강력함과 약점을 보여주었습니다. VCB Bench는 중국어 음성 대화 모델의 발전을 위한 표준화된 메소드를 제공하여 의미 있는 통찰력을 제공합니다.



### Automating Structural Engineering Workflows with Large Language Model Agents (https://arxiv.org/abs/2510.11004)
Comments:
          Code: this https URL

- **What's New**: 본 논문에서는 MASSE를 소개합니다. MASSE는 구조 공학(Structural Engineering)을 위한 최초의 다중 에이전트 시스템(Multi-Agent System)으로, 대형 언어 모델(LLM) 기반 에이전트와 실제 공학 작업 흐름을 효과적으로 통합합니다. 구조 공학 분야는 경제적으로 큰 영향을 미치지만, 수십 년 동안 핵심 작업 흐름은 크게 변화하지 않았습니다.

- **Technical Details**: MASSE는 LLM의 복잡한 추론(complex reasoning), 장기 계획(long-horizon planning), 정밀 도구 활용(precise tool utilization) 능력을 활용하여 설계 규정 해석, 하중 계산, 구조 용량 검증 등의 작업을 수행할 수 있습니다. 해당 시스템은 학습 없는 LLM 기반의 다중 에이전트 시스템으로, 거의 모든 실제 구조 공학 작업 흐름을 완전 자동화할 수 있는 개념 증명을 보여줍니다.

- **Performance Highlights**: MASSE는 전문 환경에서 즉시 배포할 수 있으며, 실제 사례 연구를 통해 검증된 결과는 전문 엔지니어의 작업 부담을 약 2시간에서 몇 분으로 줄일 수 있음을 보여줍니다. 이는 실제 엔지니어링 시나리오에서 신뢰성과 정확성을 향상시키는 데 도움이 됩니다.



### Secret-Protected Evolution for Differentially Private Synthetic Text Generation (https://arxiv.org/abs/2510.10990)
- **What's New**: 이번 연구에서는 비밀 보호 이론에 기반한 새로운 합성 텍스트 생성 프레임워크인 Secret-Protected Evolution (SecPE)을 제안합니다. SecPE는 Differential Privacy (DP) 모델의 편향을 해결하고, 기밀정보에 대한 보호를 강화하여 유틸리티를 극대화하는 동시에 계산 복잡도를 줄입니다. 이를 통해 더 실제적이고 효과적인 개인 정보 보호 기능을 제공할 수 있습니다.

- **Technical Details**: SecPE는 두 가지 주요 구성 요소로 이루어져 있습니다: (1) 비밀 클러스터링(Secret Clustering) - 민감한 속성을 탐지하고, 노이즈로 가득 찬 개인 데이터를 이용하여 대표 클러스터를 형성합니다; (2) 보호된 진화(Protected Evolution) - 고품질 합성 데이터에서 변형을 반복적으로 샘플링하고, 노이즈 대표자에 대해 평가한 후 최적 후보를 선택합니다. 이런 설계는 PE의 실용성을 유지하면서도 비밀 중심의 보호로 전환합니다.

- **Performance Highlights**: OpenReview, PubMed 및 Yelp 벤치마크 실험을 통해 SecPE는 낮은 Fréchet Inception Distance (FID)와 더 높은 다운스트림 작업의 정확도를 달성하며, 동일한 보호 수준을 유지하기 위해 필요한 노이즈도 적습니다. 이러한 성과는 비밀 중심의 보장이 더 실용적이고 효과적인 개인 정보 보호 합성 텍스트 생성의 잠금을 해제할 수 있음을 강조합니다.



### Revisiting Model Interpolation for Efficient Reasoning (https://arxiv.org/abs/2510.10977)
Comments:
          14 pages, 6 figures, 7 tables. Working in progress

- **What's New**: 본 논문은 모델 병합(model merging) 기법을 심층적으로 재조명하며, 두 개의 모델 가중치를 직접 보간(interpolation)하는 가장 간단한 병합 방법을 분석합니다. 저자들은 모델 보간이 사고 경로(reasoning trajectory)에서 세 가지 단계를 따르는 진화적 패러다임을 가지고 있음을 발견하였습니다. 이 연구는 모델 보간의 메커니즘을 설명하고, 전략적으로 보간된 모델이 고급 모델 병합 기준을 초월할 수 있다는 점을 강조합니다.

- **Technical Details**: 연구에서는 모델 보간 방식이 세 가지 단계로 나뉘며, 각 단계에서 Pass@k와 Mean@k의 성능 지표가 비선형적으로 발전하는 과정을 살펴봅니다. 가중치 주도 모델과 인스트럭션 모델을 포함한 하이브리드 모델을 통해, 복잡한 작업에 대한 효율적이면서도 효과적인 추론 능력을 확보할 수 있는 방법론을 제시합니다. 또한, 모델의 층(layer) 및 모듈(modules)과 디코딩 전략(decoding strategies)에 대한 정밀한 실험을 통해 깊이 있는 분석을 수행합니다.

- **Performance Highlights**: 실험 결과, 전략적으로 보간된 모델이 다양한 도전적인 벤치마크에서 기존의 고급 모델 병합 기준을 초과하는 성능을 보여주었습니다. 특히, 수학적 추론, 명령 이행, 과학적 추론을 포함한 여러 기준에서 성과를 나타냈습니다. 이 연구는 특정 토큰 예산(token budget)을 준수하는 모델 설계에 대한 실제적인 프레임워크를 제공합니다.



### Rediscovering Entropy Regularization: Adaptive Coefficient Unlocks Its Potential for LLM Reinforcement Learning (https://arxiv.org/abs/2510.10959)
Comments:
          16 pages, 4 figures

- **What's New**: 이 연구에서는 대규모 언어 모델(LLMs)의 추론 능력을 향상시키기 위한 강화 학습 방법론인 RLVR(Reinforcement Learning with Verifiable Rewards)를 재검토하고, 정Entropy regularization의 잠재력이 과소평가되고 있다고 주장합니다. 특히 변동성이 큰 고정 계수를 사용하는 전통적인 접근법의 한계를 극복하기 위해 Adaptive Entropy Regularization(AER)이라는 새로운 프레임워크를 제안합니다.

- **Technical Details**: AER는 세 가지 주요 구성 요소를 통해 탐색(exploration)과 활용(exploitation)을 동적으로 조절합니다: 난이도 인식 계수 할당, 초기 기반 목표 엔트로피, 동적 전역 계수 조정 등이 포함됩니다. 이 방법은 각 작업의 난이도에 따라 엔트로피를 조절하며, 사전 설정된 수준 이하에서 유지하여 안정적인 학습을 목표로 합니다.

- **Performance Highlights**: 실험 결과, AER는 다양한 수학적 추론 벤치마크에서 기존 방법에 비해 일관된 성능 향상을 보여주었으며, 추론 정확도와 탐색 능력 모두 개선되었습니다. 이는 RLVR 훈련에서 적응적 엔트로피 정규화의 잠재력을 입증하는 결과입니다.



### The Social Cost of Intelligence: Emergence, Propagation, and Amplification of Stereotypical Bias in Multi-Agent Systems (https://arxiv.org/abs/2510.10943)
Comments:
          15 pages, 19 figures, Preprint. Under review

- **What's New**: 이 연구는 다중 에이전트 시스템(Multi-Agent Systems, MAS) 내에서의 고정관념적 편향을 조사합니다. 이전 연구들은 단일 대형 언어 모델(LLMs)의 편향에 집중했지만, MAS의 등장으로 여러 LLM이 상호작용하는 새로운 다이나믹스가 등장하고 있습니다. 이 논문은 에이전트 간의 커뮤니케이션이 편향의 발생과 전파에 어떤 영향을 미치는지를 평가하는 포괄적 연구를 제공합니다.

- **Technical Details**: MAS는 각 에이전트가 특정 사회 집단을 대표하거나 비대표로 남는 사회적 맥락을 시뮬레이션합니다. 각 에이전트는 그들의 그룹, 지능(LLM), 반응 상태에서 정의됩니다. 연구는 세 가지 고정관념 편향 벤치마크를 바탕으로 에이전트 간의 상호작용에서 편향의 발생, 전파 및 증폭을 평가하며, 협력 또는 토론 기반의 커뮤니케이션 프로토콜이 편향 증폭을 줄이는 데 어떻게 기여하는지를 다룹니다.

- **Performance Highlights**: 연구 결과, MAS는 단일 에이전트 시스템(Single-Agent Systems, SAS)보다 편향 발생을 방지하는데 덜 강력한 것으로 나타났습니다. 편향은 주로 인그룹 편애를 통해 조기에 발생하지만, 에이전트들이 일부 편향을 인지하고 저항하는 덕분에 전파는 제한적입니다. 또한, MAS는 더 강력한 LLM을 기반으로 할 경우 편향 공격으로부터 더 높은 저항력을 지니며, 협력적인 커뮤니케이션 방식이 편향 증폭을 줄이는데 효과적이라는 중요한 통찰을 제공합니다.



### Find Your Optimal Teacher: Personalized Data Synthesis via Router-Guided Multi-Teacher Distillation (https://arxiv.org/abs/2510.10925)
Comments:
          19 pages, 10 figures

- **What's New**: 최근 연구들은 더 강력한 teacher models가 항상 최적의 teachers가 아니라는 것을 보여주었습니다. 이에 따라 PerSyn (Personalized data Synthesis)라는 새로운 데이터 합성 전략이 제안되었습니다. PerSyn은 ‘Route then Generate’라는 새로운 패러다임을 적용하여 각 student model에 맞춤형 데이터를 생성합니다.

- **Technical Details**: PerSyn은 각 prompt를 최적의 teacher model에 할당하는 과정에서 student의 학습 가능성과 teacher의 응답 품질을 모두 고려하는 쿼리 수준의 라우터를 사용합니다. 이 과정에서 각 teacher는 할당된 prompt에 대해서만 데이터를 합성하게 되어 전통적인 ‘Generate then Select’ 방식보다 효율적입니다.

- **Performance Highlights**: PerSyn은 다양한 모델 패밀리와 스케일에서 동작하며, instruct tuning 및 수학적 추론 설정에서 모든 기준 모델보다 우수한 성능을 보여주었습니다. 이를 통해 PerSyn의 효과성과 향후 연구의 방향성을 제시하는 중요한 통찰을 제공합니다.



### DRIFT: Decompose, Retrieve, Illustrate, then Formalize Theorems (https://arxiv.org/abs/2510.10815)
- **What's New**: 새로운 DRIFT 프레임워크는 대형 언어 모델(LLM)이 비공식 수학 명제를 작고 더 관리하기 쉬운 '하위 구성요소'로 분해하도록 지원합니다. 이를 통해 Mathlib과 같은 수학 라이브러리에서 기본 전제들을 더 효율적으로 검색할 수 있게 됩니다. 또한, 해당 프레임워크는 예제 정리를 검색하여 모델이 전제를 더 효과적으로 사용하도록 돕는 새로운 접근법을 제공합니다.

- **Technical Details**: DRIFT는 네 단계로 구성된 프로세스를 통해 비공식 수학 진술의 복잡성을 처리하고, 검색된 형식적 객체의 증명 예시를 제공합니다. 첫 번째 단계에서는 LLM이 비공식 진술을 작은 하위 쿼리로 분해합니다. 이후 이 쿼리는 Mathlib와 같은 형식적 라이브러리에서 의존하는 전제를 검색하는 데 사용되며, 이로 인해 보다 정확한 정의 검색이 가능합니다.

- **Performance Highlights**: DRIFT는 ProofNet 및 ConNF 벤치마크에서 새로운 최첨단 성과를 달성하였고, 특히 ConNF 벤치마크에서 GPT-4.1과 DeepSeek-V3.1을 사용하여 각각 37.14% 및 42.25%의 성장을 보여주었습니다. 이러한 분석은 수학 자동 형식화의 효과가 모델의 지식 경계에 크게 의존하고 있음을 강조하며, 각 모델의 능력에 맞춘 적응형 검색 전략의 필요성을 시사합니다.



### Bhasha-Rupantarika: Algorithm-Hardware Co-design approach for Multilingual Neural Machine Translation (https://arxiv.org/abs/2510.10676)
- **What's New**: 이 논문에서는 자원 제한 환경을 위해 알고리즘-하드웨어 코드 디자인을 통해 조정된 경량의 다국어 번역 시스템, Bhasha-Rupantarika를 소개하고 있습니다. 이 방법은 초옥텟 정밀도(FP8, INT8, INT4, FP4)에서 모델 배포를 조사하며, FP4 경우 모델 크기를 4.1배 줄이고 추론 속도를 4.2배 향상시킴을 보였습니다. 이는 IoT 장치에서 FPGA 가속기를 사용하는 실시간 배포에 있어 초저정밀 양자화(quantization)의 중요성을 강조합니다.

- **Technical Details**: 본 연구는 인도 및 국제 언어 간의 번역 구현에 중점을 두고 있으며, Rural Indian 지역에서 저비용 FPGA 솔루션을 적용하고자 합니다. 제안된 파이프라인은 OpenAI Whisper, AI4Bharat IndicConformer, Coqui XTTS, IndicParler TTS와 같은 최첨단 모델들을 포함하고 있으며, NLLB-200이라는 600M 매개변수의 경량화된 변형을 활용하여 다국어 번역을 지원합니다. 이는 리소스가 제한된 장치에서 단일 모델로 다국어 번역을 수행할 수 있게 합니다.

- **Performance Highlights**: FPGA 배포를 통해 LUTs 수가 1.96배 줄어들고 FFs 수가 1.65배 감소하며, OPU 대비 2.2배, HPTA 대비 4.6배의 처리량 향상이 이루어졌습니다. 이 연구는 양자화-aware 번역 및 하드웨어 효율성을 기반으로 다국어 AI 시스템에 적합한 실행 가능한 솔루션을 제공합니다. 결과적으로, 텍스트 처리와 다국어 지원에서 실속 있는 성능을 보여주며, 특히 자원이 제한된 환경에서 높은 효과성을 발휘합니다.



### A Layered Intuition -- Method Model with Scope Extension for LLM Reasoning (https://arxiv.org/abs/2510.10592)
- **What's New**: 본 논문은 기존의 메소드 기반 추리(method-based reasoning)와 범위 확장(scope extension) 개념을 통합하여, 직간접적으로 미접근(indirected) 문제를 보다 체계적으로 다루기 위한 직관-메소드 계층 모델(Intuition-Method Layered Model with Scope Extension)을 제안합니다. 이 모델은 직관 기반 사고(intuition-based thinking)와 메소드 기반 사고(method-based thinking)를 통합하며, 수직(원인 분석) 및 수평(유사 문제)으로 문제의 적용 가능성을 넓히고, 시간과 공간 차원(temporal and spatial dimensions)에서의 확장성을 통해 추리 능력을 강화합니다.

- **Technical Details**: 이 프레임워크는 메소드 기반 추리가 질문과 해결책을 독립적인 재사용 가능한 단위로 분리하는 방식으로 작동하며, 다양한 범위 확장을 적용하여 적응성을 향상시킵니다. 체계적 지식 트리(systematic knowledge trees)가 이러한 확장을 구조적 위계로 조직하여, 더 큰 지식 네트워크로 연결합니다. 또한, 방법 확장의 엔트로피(entropy of method extension)를 정의하여, 시스템이 미접근 문제를 해결하는 능력을 정량적으로 측정하는 새로운 지표를 제안합니다.

- **Performance Highlights**: 제안된 직관-메소드 계층 모델은 기존의 프리 트레인(pre-trained) 매핑을 넘어서는 추리 능력을 확장하며, 비직접적인 문제들을 더욱 강력하고 체계적으로 다룰 수 있는 기회를 제공합니다. 이 모델은 기존의 방법들을 논리적으로 통합하여, 새로운 기여와 함께 보다 포괄적이고 확장 가능한 추리 패러다임을 제시합니다. 이를 통해 실제 문제 해결에 보다 적합한 AI 시스템으로서의 기능을 강조합니다.



### Sample-Efficient Online Learning in LM Agents via Hindsight Trajectory Rewriting (https://arxiv.org/abs/2510.10304)
- **What's New**: 언어 모델(LM) 에이전트는 새로운 환경에서 상호작용을 학습할 때 샘플 효율성이 낮아지는 문제를 해결하기 위해 ECHO(Experience Consolidation via Hindsight Optimization)라는 새로운 프레임워크를 도입했습니다. ECHO는 실패한 시도에서 얻은 경험을 활용하여 대체 목표를 위한 최적화된 궤적(trajectories)을 생성함으로써, 비효율적인 학습을 개선하는 데 중점을 둡니다. 이 방법은 경험 재생(replay) 메커니즘을 사용하여 언어 모델이 과거의 실패를 성공적인 경험으로 전환할 수 있도록 돕습니다.

- **Technical Details**: ECHO 시스템은 두 가지 구성 요소로 구성됩니다: 처음으로, 언어 모델을 사용하여 관련 서브 목표(subgoals)를 식별하고 최적화된 궤적을 생성하는 회상 규칙(hindsight rule)이 있습니다. 두 번째로, 압축된 궤적 표현을 기억에 유지하는 업데이트 규칙(update rule)이 포함되어 있습니다. ECHO는 기존의 경험 재생 대신 더 많은 수정 가능성을 제공하여, 실패한 궤적을 임의로 재작성(rewriting)할 수 있게 합니다.

- **Performance Highlights**: XMiniGrid 및 PeopleJoinQA와 같은 다양한 상태 유지 가능한 테스트 환경에서 ECHO를 평가한 결과, 기존의 언어 에이전트보다 최대 80% 더 높은 성능을 달성했습니다. XMiniGrid에서 ECHO는 Reflexion 및 AWM과 같은 고급 에이전트 구조를 초과한 성능을 보여주며, 새로운 환경에 대한 적응 속도가 빨라짐을 입증하였습니다. ECHO는 특히 보상이 드문 환경에서 언어 에이전트의 샘플 효율성을 극대화하는 유망한 기술입니다.



### RLFR: Extending Reinforcement Learning for LLMs with Flow Environmen (https://arxiv.org/abs/2510.10201)
Comments:
          Project Website: this https URL

- **What's New**: 이 논문에서는 Verifiable Rewards (RLVR) 기반의 강화 학습 프레임워크를 개선하기 위해 새로운 방식인 RLFR(Flow rewards)을 제안합니다. 특히, LL(M)s의 성장하는 잠재 공간을 활용해 보상 신호를 더욱 탄력적으로 만들 수 있는 방법을 탐구합니다. RLFR은 유망한 보상 신호 수집을 위한 환경으로 흐름 필드를 구성하고, 임상적인 데이터 및 모델의 고품질 데이터를 활용해 정책 탐색을 권장합니다.

- **Technical Details**: RLFR(Flow rewards)은 latent space에서 파생된 흐름 보상(shaping rewards)을 기반으로 하며, 정책의 속도 편차(velocity deviations)를 통해 보상 신호를 측정합니다. 이 방법은 오프-정책(high-quality data)과 온-정책(rejection sampling) 데이터를 함께 사용하여 흐름 필드를 구축하는 방식을 포함합니다. 이러한 흐름 필드는 정책 최적화와 함께 온라인으로 업데이트되며, 연구에 활용 가능한 모든 코드, 데이터 및 모델 가중치를 공개합니다.

- **Performance Highlights**: 언어 및 다중 모달(multi-modal) 추론 벤치마크에서 RLFR의 유효성을 검증하였으며, 기존 RLVR 및 기타 보상 기본 방법에 비해 일관된 성과 향상을 보였습니다. RLFR은 모델의 숨겨진 상태 내에서 효율적인 문맥 의존성을 활용하여 다양한 데이터 집합의 정당성을 보장합니다. 이러한 결과는 RLVR 프레임워크를 활용한 보상 설계에서 새로운 가능성을 시사합니다.



### CardRewriter: Leveraging Knowledge Cards for Long-Tail Query Rewriting on Short-Video Platforms (https://arxiv.org/abs/2510.10095)
- **What's New**: 이번 논문에서는 CardRewriter라는 새로운 프레임워크를 도입했습니다. 이 프레임워크는 도메인 특화 지식을 통합하여 짧은 동영상 플랫폼에서의 장기 쿼리 재작성(long-tail query rewriting)의 품질을 향상시키는 것을 목표로 합니다. CardRewriter는 사용자 쿼리에 관련된 다중 소스의 지식을 수집하여 정보 카드로 요약하고, 이를 제공함으로써 LLM의 사용자 의도를 더 잘 이해하도록 지원합니다.

- **Technical Details**: CardRewriter는 사용자 쿼리와 관련된 다중 소스의 지식 정보를 수집하는 데 사용됩니다. 이 정보는 쿼리와 일치하는 간결한 지식 카드로 요약되어 LLM의 쿼리 재작성 과정에 통합됩니다. 또한, 두 단계의 학습 파이프라인을 통해 최적화되며, 이는 감독 세부 조정(Supervised Fine-Tuning, SFT)과 그룹 상대 정책 최적화(Group Relative Policy Optimization, GRPO)를 포함합니다.

- **Performance Highlights**: 실험 결과, CardRewriter는 재작성 품질을 상당히 향상시켰으며, Kuaishou 플랫폼에서 사용자 수백만 명에게 긍정적인 경험을 제공했습니다. 온라인 A/B 테스트에서는 장기 조회율(Long View Rate, LVR)과 클릭률(Click-Through Rate, CTR)의 유의미한 증가가 나타났습니다. CardRewriter의 도입 이후, 사용자의 초기 쿼리 재형성 비율(Initiative Query Reformulation Rate, IQRR)이 감소하여 사용자 만족도 향상에 기여하고 있습니다.



### Operationalizing AI: Empirical Evidence on MLOps Practices, User Satisfaction, and Organizational Contex (https://arxiv.org/abs/2510.09968)
- **What's New**: 이번 연구는 인공지능(AI) 개발 플랫폼에 대한 8,000개 이상의 사용자 리뷰를 분석하여 머신 러닝 운영(MLOps) 관행의 효과를 조명합니다. MLOps는 소프트웨어 엔지니어링 원칙을 머신러닝 라이프사이클 관리의 특수 요구와 통합하는 모범 사례입니다. 이러한 연구는 MLOps의 구현이 AI 애플리케이션의 개발과 운영에 어떠한 도움을 주는지에 대한 실제적인 증거를 제공합니다.

- **Technical Details**: 연구팀은 제로샷 분류(zero-shot classification) 기술을 사용하여, 지속적 통합과 배포(Continuous Integration and Delivery, CI/CD), 워크플로우 오케스트레이션(workflow orchestration), 재현성(reproducibility), 버전 관리(versioning), 협업(collaboration), 모니터링(monitoring) 등 아홉 가지 확립된 MLOps 관행에 대한 사용자 리뷰의 감정을 측정했습니다. 연구 결과, 총 아홉 가지 관행 중 일곱 가지가 사용자 만족도와 긍정적인 관계를 보였으며, 이는 효과적인 MLOps 구현이 AI 개발에 실질적인 가치를 기여하고 있음을 나타냅니다.

- **Performance Highlights**: 작은 회사의 리뷰어들은 특정 MLOps 관행에 대해 덜 자주 언급하였으며, 이는 조직의 맥락이 MLOps의 중요성과 연관성에 영향을 미침을 시사합니다. 그러나 기업 규모는 MLOps와 만족도 간의 관계를 조절하지 않는 것으로 보입니다. 결과적으로, MLOps 관행이 적용되면 조직적인 환경에 상관없이 보편적으로 긍정적인 영향을 미친다고 할 수 있습니다.



### The Personalization Trap: How User Memory Alters Emotional Reasoning in LLMs (https://arxiv.org/abs/2510.09905)
Comments:
          12 pages 5 figures

- **What's New**: 이번 연구는 LLM(대형 언어 모델)이 사용자 메모리를 통해 감정 이해에 미치는 영향을 조사합니다. 동일한 시나리오가 서로 다른 사용자 프로파일과 함께 제공될 때, LLM의 감정 해석이 어떻게 다르게 나타나는지를 연구하여, 특정 프로파일이 더 정확한 감정 해석을 받는 경향이 있음을 발견했습니다. 이러한 결과는 개인화 AI 시스템이 사회적 불평등을 재생산할 위험이 있음을 시사합니다.

- **Technical Details**: 연구진은 사용자 지식을 다양한 프로파일을 이용하여 생성하고, LLM의 감정 이해와 행동 조언을 평가하기 위해 STEU(상황 감정 이해 테스트)와 STEM(감정 관리 상황 테스트)를 사용했습니다. 각 프로파일은 Bourdieu의 사회 자본 이론을 기반으로 하여 경제적, 문화적, 사회적 요소들을 포함하도록 제작되었으며, 정규화된 정서적 지능 테스트를 통해 LLM의 성능을 분석하였습니다.

- **Performance Highlights**: 실험 결과에 따르면 대부분의 LLM은 사용자 메모리를 도입했을 때 성능이 감소했으며, 특히 우세한 배경을 가진 사용자 프로파일에 대해 더 나은 성과를 보였습니다. 또한, 인종, 성별, 연령에 따른 편향이 존재하며, 이는 감정 관련 조언에서도 지속적으로 나타났습니다. 이 연구는 AI의 개인화 시스템이 사회적 계층 구조를 반영할 수 있다는 중요한 문제를 드러냈습니다.



### The Geometry of Reasoning: Flowing Logics in Representation Spac (https://arxiv.org/abs/2510.09782)
Comments:
          Code: this https URL

- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)이 어떻게 '사고'하는지를 탐구합니다. 새로운 기하학적 프레임워크를 제안하여 LLM의 추론을 흐름(flow)으로 모델링하며, 이는 논리가 진행되는 임베딩 경로(embedding trajectories)를 나타냅니다. 이 연구는 LLM이 표면적인 형식을 넘어 논리를 내부화하는지를 검증할 수 있는 새로운 관점을 제공합니다.

- **Technical Details**: 연구자들은 자연적 추론(propositional natural deduction)을 사용하여 의미론적 도구(semantic carriers)가 다양해도 논리 구조와 의미를 분리하여 LLM의 사고 과정을 분석합니다. 이 기하학적 관점은 위치(position), 속도(velocity), 곡률(curvature)과 같은 기하학적 양들과 추론을 연결하여, 추상화된 개념 공간(representation and concept spaces)에서의 분석을 가능하게 합니다.

- **Performance Highlights**: 이론적 프레임워크를 구현하기 위해 학습된 임베딩 대리 모델(learned representation proxies)을 사용하여 추론 흐름을 시각화하고 정량화하는 통제된 실험을 설계하였습니다. 이 연구는 LLM의 행동의 해석 가능성과 공식적인 분석을 위한 새로운 관점을 제공하며, 추론 현상을 연구하기 위한 개념적 기초와 실제 도구 역할을 합니다.



### Building a Foundational Guardrail for General Agentic Systems via Synthetic Data (https://arxiv.org/abs/2510.09781)
- **What's New**: 이번 연구에서는 LLM(대규모 언어 모델) 기반 에이전트의 안전성을 높이기 위해 사전 실행 단계에서의 개입을 강조합니다. 저자들은 데이터 갭(data gap), 모델 갭(model gap), 평가 갭(evaluation gap)이라는 세 가지 연구 갭을 지적하며 이를 해결하기 위한 접근법을 제시합니다. 특히, 데이터 생성 엔진인 AuraGen과 강력한 가드레일 모델인 Safiron을 제안하여 LLM이 위험한 행동을 사전에 차단할 수 있도록 합니다.

- **Technical Details**: AuraGen은 (i) 양호한 경로(trajectories) 생성, (ii) 카테고리 라벨이 있는 위험 요소 삽입, (iii) 자동화된 보상 모델을 통한 출력 필터링이라는 세 단계를 통해 높은 품질의 데이터셋을 생성합니다. Safiron은 다양한 입력 형식을 통합하는 어댑터와 compact guardian model로 구성되어 있으며, 위험한 행동을 실시간으로 식별하고 설명할 수 있는 기능을 제공합니다. 이러한 두 가지 요소는 안전성을 극대화하기 위해 효과적으로 훈련됩니다.

- **Performance Highlights**: Pre-Exec Bench라는 새로운 벤치마크를 통해 저자들은 제안한 가드레일이 강력한 기초 모델 및 비공식 기준선에 비해 뛰어난 성능을 보여준다고 보고합니다. 이 연구에서는 개입 시점에서 에이전트의 위험 행동을 성공적으로 차단하고, 이를 통해 안전하고 효과적인 에이전트 시스템을 구축할 수 있는 방향성을 모색합니다. 저자들이 제안한 프레임워크는 다양한 과제에 걸쳐 일반화될 수 있는 가능성을 지니고 있습니다.



### Machine learning methods fail to provide cohesive atheoretical construction of personality traits from semantic embeddings (https://arxiv.org/abs/2510.09739)
Comments:
          1 figure, 12 pages

- **What's New**: 이번 연구에서는 언어에 포함된 성격 특성을 바탕으로 한 'Lexical Hypothesis'를 검토하였습니다. 기계 학습(machine learning)을 활용하여 고전 형용사 목록에서 기본적인 성격 모델을 생성하고, 이를 Big Five 모델과 비교하였습니다. 특히, Reddit에서 한 백만 개의 댓글을 분석하여 온라인 커뮤니티의 성격을 어떻게 설명할 수 있는지를 살펴보았습니다.

- **Technical Details**: 연구에서는 Rosemary E. V.와 같은 클래식한 형용사 리스트를 기반으로 한 성격 모델링을 실시하였습니다. 기계 학습 기법을 이용해 댓글을 분석한 결과, Big Five 요소 중 Agreeableness, Conscientiousness, Neuoticism이 특히 강력하고 해석 가능한 설명을 제공하는 것으로 나타났습니다. 반면, 기계 학습을 통한 클러스터링은 의미 있는 구분을 제공하지 못했으며, Extraversion 특성을 회복하는 데 실패했습니다.

- **Performance Highlights**: 연구 결과는 Big Five 모델의 강인성을 확인시키며 성격의 의미론적 구조가 상황 의존적일 수 있음을 암시합니다. 기계 학습이 기존 심리학 이론의 생태학적 타당성(ecological validity)을 점검하는 데 도움이 될 수 있지만, 이러한 이론을 대체할 수는 없는 것으로 나타났습니다. 이는 심리학적 이론의 중요한 역할을 다시 한번 강조하는 결과입니다.



### It's 2025 -- Narrative Learning is the new baseline to beat for explainable machine learning (https://arxiv.org/abs/2510.09723)
Comments:
          18 pages, 5 figures

- **What's New**: 최근의 논문에서는 Narrative Learning이라는 새로운 방법론을 소개합니다. 이 방법론은 모델을 전체적으로 자연어로 정의하고, 전통적인 수치 최적화 대신 설명 프롬프트를 사용하여 분류 기준을 반복적으로 수정합니다. 우리는 이 접근법의 정확성과 잠재력을 평가하기 위해 6개 데이터셋을 사용했으며, 기존의 7가지 설명 가능한 기계 학습 모델과 비교하여 많은 데이터셋에서 높은 정확도를 달성했습니다.

- **Technical Details**: Narrative Learning은 감독된 이진 분류 알고리즘으로, 라벨이 붙은 데이터를 훈련, 검증, 테스트 데이터로 나누어 사용합니다. 두 개의 언어 모델인 Overseer와 Underling이 사용되며, Overseer는 자연어 프롬프트로 분류 지침을 생성하고, Underling은 이를 평가하여 분류 결과를 반환합니다. 이 과정은 반복적으로 진행되며, 모델의 정확도를 향상시키기 위해 Overseer가 프롬프트를 수정하고 다시 Underling에 전달하는 방식으로 진행됩니다.

- **Performance Highlights**: 실험 결과, Narrative Learning은 6개 데이터셋 중 5개에서 기존의 설명 가능한 기계 모델보다 더 높은 정확도를 기록했습니다. 특히, KT 정확도와 같은 지표를 사용하여 성능을 평가했으며, 이는 대규모 데이터셋에서의 성능 향상에 초점을 맞추었습니다. 추가적으로, Lexicostatistics 트렌드도 보고하여 모델의 설명 가능성에 대한 이해를 높였습니다.



### A Comprehensive Survey on Benchmarks and Solutions in Software Engineering of LLM-Empowered Agentic System (https://arxiv.org/abs/2510.09721)
Comments:
          21 pages

- **What's New**: LLM(대형 언어 모델)의 소프트웨어 공학 통합은 전통적인 규칙 기반 시스템에서 자율적 문제 해결이 가능한 정교한 시스템으로의 패러다임 전환을 촉진했습니다. 그러나 이 분야에는 벤치마크와 해결책 간의 관계에 대한 종합적인 이해가 부족하여 체계적인 발전과 평가가 저해되고 있습니다. 본 설문조사는 LLM을 활용한 소프트웨어 공학에 대한 첫 번째 전체론적 분석을 제시하며, 평가와 솔루션 접근 방식 간의 중요한 격차를 연결합니다.

- **Technical Details**: 본 연구에서는 150개 이상의 최근 논문을 분석하고, 이를 프롬프트 기반, 파인튜닝 기반, 에이전트 기반 패러다임으로 분류하여 종합적인 분류 체계를 구축했습니다. 또한, 코드 생성, 번역, 수리 등 주요 작업을 포괄하는 벤치마크를 분석하여 LLM 기반 시스템의 발전 경향을 살펴보았습니다. 우리는 작업 사양에서 최종 산출물까지의 전체 워크플로우를 설명하는 통합 파이프라인을 제시합니다.

- **Performance Highlights**: 이 연구는 50개 이상의 벤치마크와 해당 솔루션 전략을 연결하여 이 분야의 연구자들이 특정 평가 기준에 적합한 최적의 접근 방식을 식별할 수 있도록 돕습니다. 또한, 멀티 에이전트 협업 프레임워크, 자기 발전하는 코드 생성 시스템, LLM 기반 방법과의 형식 검증 통합 등 미래의 중요한 연구 방향을 제안했습니다. 이 설문조사는 LLM을 활용한 소프트웨어 공학 시스템을 이해하고 평가하며 발전시키고자 하는 연구자와 실무자를 위한 기초 자료로 작용합니다.



### Group-Adaptive Adversarial Learning for Robust Fake News Detection Against Malicious Comments (https://arxiv.org/abs/2510.09712)
Comments:
          10 pages, 12 figures

- **What's New**: 이번 논문은 온라인에서의 가짜 뉴스 탐지(Fake News Detection, FND) 모델이 사용자 댓글(Comments) 및 대형 언어 모델(Large Language Models, LLMs)이 생성한 댓글에 의해 취약해질 수 있음을 지적합니다. 특히, 댓글 공격에 대한 포괄적인 평가를 제시하고, 이를 통해 FND 모델의 강건성을 향상시키기 위한 그룹 적응형 적대적 훈련 전략(Group-Adaptive Adversarial Training Strategy)을 소개했습니다. 이 방법은 심리학적 원리에 기반한 세 가지 댓글 카테고리로 공격을 분류하여 다각적인 적대적 훈련을 실시합니다.

- **Technical Details**: 제안된 방법은 세 가지 단계로 구성됩니다: 먼저, 공격을 세 가지 심리학적으로 기반을 둔 카테고리로 나눕니다: 지각적(Perceptual), 인지적(Cognitive), 사회적(Societal). 둘째, LLMs를 활용하여 각 카테고리 당 다양한 공격을 생성하며, 셋째, Dirichlet 분포 기반의 적응형 샘플링 메커니즘(InfoDirichlet Adjusting Mechanism)을 적용하여 훈련 중 각 댓글 카테고리의 학습 집중도를 동적으로 조정합니다. 이러한 접근은 다양한 댓글 공격에 대한 강건성을 유지하면서도 강력한 탐지 정확도를 유지하는 것으로 나타났습니다.

- **Performance Highlights**: 실험 결과, RumourEval-19, Weibo16 및 Weibo20와 같은 기준 데이터셋에서 기존 모델들이 제안된 댓글 공격에 의해 상당한 성능 저하를 겪는 것을 보여주었습니다. 반대로, 우리 제안된 프레임워크는 이러한 댓글 공격에 대해 저항할 수 있었고, 최첨단 기법들보다 더 나은 성능 개선을 달성하였습니다. 이는 가짜 뉴스 탐지에서 댓글의 다양성과 그 공격에 대한 강건성을 높이는 방법으로 제시되고 있습니다.



### Stop DDoS Attacking the Research Community with AI-Generated Survey Papers (https://arxiv.org/abs/2510.09686)
Comments:
          Accepted by NeurIPS 2025 (Position Track)

- **What's New**: 이 논문에서는 최근의 AI 생성 설문지 급증 현상에 대해 논의하고 있습니다. 전통적으로 노동 집약적이었던 설문지 작성이 대형 언어 모델(LLMs) 덕분에 빠르고 쉽게 이루어지는 반면, 이는 연구 커뮤니티에 'survey paper DDoS attack'이라는 새로운 위협을 초래하고 있습니다. 이 공격은 피상적으로 종합적인 연구 결과물들이 홍수처럼 쏟아져 나와 연구자들을 압도하고 신뢰를 훼손하는 것을 뜻합니다.

- **Technical Details**: 연구팀은 2020년부터 2024년까지 arXiv에 제출된 논문을 분석하여 설문지의 수가 급격히 증가하고 있음을 확인했습니다. 이들은 'survey', 'review', 'overview' 및 'taxonomy'라는 키워드가 포함된 제목을 가진 논문들을 수집하여 AI 생성 점수를 측정했으며, AI의 도움을 받은 설문지들이 특히 증가하고 있음을 발견했습니다. 이러한 AI 생성 설문지는 종종 비판적인 분석이나 비교 없이 단순한 집계로 축소되는 위험이 있습니다.

- **Performance Highlights**: AI 생성 설문이 연구 품질에 미치는 부정적인 영향을 우려하여, 논문은 AI 사용에 대한 엄격한 기준과 전문가의 감독 필요성을 주장합니다. 현재 arXiv에서의 AI 생성 설문지는 연구자들에게 큰 혼란을 초래하고 있으며, 이는 신뢰도 저하에 기여하고 있습니다. 연구가 질적으로 향상되기 위해서는 AI 지원 리뷰 작성 시 인간의 숙련된 감독과 투명성이 필수적입니다.



New uploads on arXiv(cs.IR)

### SAIL-Embedding Technical Report: Omni-modal Embedding Foundation Mod (https://arxiv.org/abs/2510.12709)
Comments:
          Technical Report

- **What's New**: 이번 논문은 SAIL-Embedding이라는 다중 모드 임베딩 모델을 제안하며, 이는 다양한 모드 지원 및 훈련 안정성 문제를 해결하기 위한 맞춤형 훈련 전략과 구조 설계를 통해 발전하고 있음을 강조합니다. 기존 모델이 가진 제한된 모드 지원 문제를 극복하기 위해, SAIL-Embedding은 시각, 텍스트, 오디오 입력의 임의의 조합을 처리할 수 있도록 설계되었습니다. 이를 통해 실제 비즈니스 요구에 맞춘 다차원 임베딩 벡터를 생성할 수 있습니다.

- **Technical Details**: SAIL-Embedding은 콘텐츠 인지적(progressive) 훈련과 협업 인지적(collaboration-aware) 추천 향상 훈련을 통해 모델을 훈련시키며, 각각의 단계에서 다양한 데이터 자원으로부터 얻은 지식을 활용합니다. 이 과정은 서로 다른 하위 작업에 대한 조정 능력 향상과 미지의 시나리오를 처리하기 위한 일반화 능력을 제공합니다. 또한, 동적 하드 네거티브 마이닝과 적응형 다중 출처 데이터 균형 조정 방법을 도입하여 훈련의 유연성과 일반화 능력을 강화합니다.

- **Performance Highlights**: SAIL-Embedding은 여러 벤치마크 데이터셋을 통한 실험 결과에서 다양한 하위 작업에서 최신 기술(STATE-OF-THE-ART) 성능을 달성하였음을 입증했습니다. 특히, Douyin 추천 시스템에서 온라인 실험을 통해 7일과 14일의 생애 주기(Lifetime) 증가를 각 +0.158% 및 +0.144%로 나타내는 등 실질적인 성과를 거두었습니다. 또한, SAIL-Embedding은 차가운 시작(cold-start) 상황에서도 추천 모델의 AUC를 0.1% 향상시키며, 각 단계에서 ∼0.03%의 LT 이득을 보여주었습니다.



### The Role of Parametric Injection-A Systematic Study of Parametric Retrieval-Augmented Generation (https://arxiv.org/abs/2510.12668)
- **What's New**: 이번 연구에서는 매개변수 주입(parametric injection)의 역할을 명확히 하기 위해 매개변수화된 검색 보강 생성(parametric retrieval-augmented generation, PRAG)에 대한 체계적인 분석을 수행했습니다. 매개변수화된 문서(Parameter documents)는 문서의 의미 정보를 부분적으로만 캡처하며, 텍스트 수준의 상호작용과 비교할 때 성능이 떨어지는 것으로 나타났습니다. 그러나 매개변수화된 표현들은 모델이 입력 컨텍스트 내의 문서에 대한 이해를 향상시킬 수 있는 고수준의 정보를 인코딩합니다.

- **Technical Details**: PRAG는 LoRA 모듈과 같은 모델 매개변수로 문서를 인코딩하고, 추론 과정에서 이러한 표현을 모델에 주입하여 LLM과 문서 간의 상호작용을 가능하게 합니다. PRAG의 가장 큰 장점은 문서를 입력 컨텍스트에 직접 삽입하지 않아도 되므로 컨텍스트 길이를 늘리지 않으면서도 깊은 상호작용을 가능하게 한다는 점입니다. 연구 결과, 파라메트릭 문서와 텍스트 문서를 결합한 PRAG-Combine 방법은 단독으로 사용할 때보다 더 나은 성능을 보여주었습니다.

- **Performance Highlights**: PRAG를 통해 모델은 제공된 컨텍스트를 보다 효과적으로 이해하고 활용할 수 있습니다. 인공적인 유도 잡음을 추가한 평가에서 매개변수 주입을 사용하는 모델이 비주입 모델보다 성능 저하가 현저히 적었습니다. 이러한 결과는 파라메트릭 주입이 노이즈에 대한 강건성을 높이며, 해당 기술이 지식 보존과 깊은 상호작용을 위한 잠재력을 가지고 있음을 시사합니다.



### SMILE: SeMantic Ids Enhanced CoLd Item Representation for Click-through Rate Prediction in E-commerce SEarch (https://arxiv.org/abs/2510.12604)
- **What's New**: SMILE(Semantic ID-based item representation enhancement)는 기존의 콜드스타트 문제를 해결하기 위해 고안된 혁신적인 접근 방식입니다. 이 방법은 RQ-OPQ 인코딩을 활용해 아이템의 콘텐츠와 협력 정보를 융합하여 보다 정확한 추천을 제공합니다. 또한, 아이템의 고유한 특성을 반영하는 차별화된 정보를 학습하는 OPQ 인코딩 기반 전략을 처음으로 제안하고 있습니다.

- **Technical Details**: SMILE 프레임워크는 두 개의 주요 모듈, 즉 RQ 인코딩과 아이템 ID 간의 적응형 전송 및 정렬 메커니즘과 OPQ 인코딩에 기반한 아이템 차별 정보 학습 모듈로 구성됩니다. RQ 인코딩은 공통된 의미 정보를 가지고 있으며, OPQ 인코딩은 아이템 별로 차별화된 기능을 캡처합니다. 이러한 방법론은 CTR(click-through rate) 예측 및 사용자 행동 예측을 위한 협업 신호 전송을 가능하게 합니다.

- **Performance Highlights**: SMILE 방식은 대규모 산업 데이터셋에서 실시한 종합적인 오프라인 실험을 통해 그 우수성을 입증했습니다. 온라인 A/B 테스트에서도 아이템 CTR은 +1.66%, 구매자 수는 +1.57%, 주문량은 +2.17% 증가하는 통계적으로 유의미한 개선 결과를 보였습니다. 이로 인해 SMILE은 콜드스타트 문제를 효과적으로 해결할 수 있는 잠재력을 발휘하고 있습니다.



### Leveraging Language Semantics for Collaborative Filtering with TextGCN and TextGCN-MLP: Zero-Shot vs In-Domain Performanc (https://arxiv.org/abs/2510.12461)
- **What's New**: 최근 몇 년 동안, 대규모 언어 모델(LLM)을 활용하여 추천 시스템에 아이템의 텍스트 정보를 통합하는 다양한 접근 방식이 제안되었습니다. 본 연구에서는 전통적인 방식과는 달리 LLM 기반의 아이템 제목 임베딩 위에 파라미터가 없는 그래프 컨볼루션 레이어를 적용하는 새로운 아키텍처인 TextGCN을 제안합니다. 이 구조는 언어 의미와 그래프 메시지 전달을 결합하여, 이전의 접근 방식보다 우수한 제로샷 성능을 달성하였습니다.

- **Technical Details**: TextGCN은 LLM에서 파생된 아이템 임베딩에 직접적으로 그래프 컨볼루션 레이어(GCL)를 적용하여, 기존의 ID 기반 임베딩을 학습하는 대신에 파라미터가 없는 방식으로 작동합니다. 또한, TextGCN-MLP를 도입하여, 대조 손실(contrastive loss)로 학습하는 멀티레이어 퍼셉트론(MLP)을 추가해 인도메인 성능을 개선하였습니다. 그러나 TextGCN-MLP의 제로샷 성능은 TextGCN보다 낮아 인도메인 전문화와 제로샷 일반화 간의 트레이드오프를 나타냅니다.

- **Performance Highlights**: 이 연구는 TextGCN이 기존 추천 모델인 LightGCN 및 MF보다 우수한 성능을 나타내고, LLM 기반의 텍스트 표현이 학습된 ID 기반 임베딩보다 월등한 대안을 제공함을 보여줍니다. TextGCN은 제로샷 추천 작업에서 최신 성능을 달성했으며, TextGCN-MLP는 여러 표준 벤치마크에서 인도메인 성능이 최상임을 입증했습니다. 전반적으로, 연구 결과는 LLM의 텍스트 임베딩을 활용한 그래프 기반 접근 방식의 유망한 가능성을 보여줍니다.



### A Hierarchical Quantized Tokenization Framework for Task-Adaptive Graph Representation Learning (https://arxiv.org/abs/2510.12369)
- **What's New**: 최근 언어 및 비전 기초 모델에서의 발전은 복잡한 입력을 대규모 모델링을 위한 압축된 시퀀스로 변환하는 이산 토큰 인터페이스의 중요성을 강조합니다. 이 논문은 비유클리드 구조와 다중 스케일 의존성을 효율적으로 처리하는 그래프 토큰화 체계를 요구하며, 기존의 방법들은 적응성과 효율성에서 한계를 보입니다. 특히 대다수의 양자화 기반 토크나이저는 계층 정보를 고정적이거나 작업 비지향적으로 조직하여 구조적 단서를 과대 표현하거나 미비하게 활용할 수 있습니다.

- **Technical Details**: 이 연구는 다중 스케일에 걸쳐 작업 적응 집합을 위한 자가 가중화 메커니즘을 도입하는 계층적 양자화 프레임워크를 제시합니다. 제안된 방법은 정보를 흐름을 조절하는 경량의 게이팅 프로세스를 통해 동결된 인코더를 유지하면서 다양한 다운스트림 작업에 파라미터 효율적으로 적응할 수 있게 합니다. QUantized HIerarchical SElf-weighted Tokenizer (QUIET)는 잔여 벡터 양자화를 통해 다중 스케일 이산 토큰을 생성하는 메커니즘을 채택합니다.

- **Performance Highlights**: 노드 분류 및 링크 예측을 위한 벤치마크 데이터셋에서의 실험 결과는 비교 가능한 계산 예산 내에서 강력한 기준선에 대해 일관된 개선을 보여줍니다. 이 연구는 계층적 양자화와 게이팅의 역할을 분리하여 분석하며, 통합 인터페이스 아래에서 노드 수준 및 엣지 수준 목표 를 지원합니다. 결과적으로 QUIET는 파라미터 효율적인 적응을 제공합니다.



### Simple Projection Variants Improve ColBERT Performanc (https://arxiv.org/abs/2510.12327)
- **What's New**: 이번 연구에서는 ColBERT 모델의 단일 레이어 선형 프로젝션을 대체하기 위해 더 잘 연구된 다양한 피드포워드 선형 네트워크(FFN)의 대안을 탐색합니다. MaxSim 연산자가 다중 벡터 모델 훈련의 그레디언트 흐름에 미치는 영향을 분석하고, 선형 프로젝션의 고유한 한계를 보여줍니다. 다양한 프로젝션 블록을 체계적으로 평가하여 더 나은 설계가 ColBERT 모델의 하류 성능에 긍정적인 영향을 미친다는 것을 증명하였습니다.

- **Technical Details**: 현재 모든 다중 벡터 모델은 원래 ColBERT 아키텍처의 변형을 따르며, 단일 레이어 선형 프로젝션을 통해 최종 출력 표현을 얻는 방식으로 작동합니다. 이 논문에서는 ColBERT 모델의 최종 피드포워드 블록에 대한 일련의 수정을 제안하고 이들의 특성이 검색 성능 향상에 어떻게 기여할 수 있는지 보여줍니다. 또한 더 많은 비선형 FFN 블록과 GLU 블록, 스킵 연결을 도입하여 다중 벡터 모델의 한계를 완화할 수 있다는 점을 강조합니다.

- **Performance Highlights**: 연구 결과, 제안된 프로젝션 변형들이 기존의 단일 레이어 프로젝션보다 우수한 성능을 보였으며, 특히 최고의 변형은 여러 검색 기준에서 평균적으로 2 NDCG@10 포인트 이상 성능을 향상시켰습니다. 다양한 벤치마크에서 다수의 비최적 프로젝션 변형 또한 전통적인 단일 레이어 프로젝션을 초과하는 성능을 보여주었으며, 이는 우리의 가설을 강하게 지지합니다. 이러한 성능 향상 효과는 랜덤 시드를 통해 일관되게 관찰되었으며, 단일 레이어의 교체가 ColBERT 모델의 강력한 업그레이드 가능성을 확인시켜줍니다.



### Causal Inspired Multi Modal Recommendation (https://arxiv.org/abs/2510.12325)
- **What's New**: 이 연구에서는 기존 멀티모달 추천 시스템에서 중요한 두 가지 편향, 즉 모달 혼란(modal confounding)과 상호작용 편향(interaction bias)에 대해 다루고 있습니다. 저자들은 Causal-inspired 멀티모달 추천 프레임워크를 제안하며, 이를 통해 숨겨진 모달 혼란 요소를 식별하고 상호작용의 노이즈 문제를 해결하기 위한 접근 방식을 제공합니다. 또한, 제안된 방법은 매우 높은 해석 가능성을 제공하며 실제 전자상거래 데이터 세트에서 성능을 입증하였습니다.

- **Technical Details**: 연구에서는 구조적 인과 모델(Structural Causal Model, SCM)을 기반으로 사용자 선호 생성 과정을 분석합니다. 다중 경로 수정을 위한 두 가지 방법론으로는 백도어 조정(back-door adjustment)과 프론트 도어 조정(front-door adjustment)을 사용하며, 특히 층화된 매칭을 통해 모달 혼란을 차단하는 전략을 사용합니다. 제안된 모델은 이중 채널 크로스 모달 확산 모듈을 통해 숨겨진 모달 혼란 요소를 효과적으로 식별합니다.

- **Performance Highlights**: 세 가지 실제 전자상거래 데이터 세트에서의 실험 결과, 제안된 방법이 최신 기준선 모델들보다 지속적으로 우수한 성능을 보였습니다. 특히, 추천 정확도가 크게 향상되었으며, 사용자 선호의 해석 가능성 또한 확보되었습니다. 이를 통해 멀티모달 추천 시스템의 발전에 다양한 기여를 할 것으로 기대됩니다.



### An Empirical Study for Representations of Videos in Video Question Answering via MLLMs (https://arxiv.org/abs/2510.12299)
Comments:
          6 pages, 3 figures

- **What's New**: 최근 멀티모달 대형 언어 모델(MLLMs)은 시각, 텍스트 및 오디오 정보를 동시에 처리하여 비디오 질문 응답(VideoQA)에서 눈에 띄는 발전을 이루었습니다. 그러나 VideoQA에서 MLLMs에 가장 효과적인 비디오 표현이 무엇인지와 다양한 모달리티가 작업 정확도와 계산 효율성 간의 균형을 어떻게 맞추는지는 여전히 불분명합니다. 이 논문에서는 MLLMs를 위한 비디오 표현 방식에 대한 포괄적인 실증 연구를 제시합니다.

- **Technical Details**: 논문은 단일 모달리티 입력(질문만, 자막, 시각 프레임 및 오디오 신호)과 이들의 조합을 체계적으로 평가하는 방법을 다룹니다. 특히, GPT 시리즈 및 LLaVA-video 시리즈와 같은 MLLMs의 발전은 비디오 질문 응답 능력을 크게 향상시켰으며, 효율성과 효과성 간의 명확한 트레이드오프를 강조합니다. 각 모달리티는 고유한 장단점을 가지고 있으며, 이를 통해 작업의 정확도를 극대화하고 계산 효율성을 높이기 위한 조합의 중요성을 설명합니다.

- **Performance Highlights**: 연구 결과, 시각 프레임은 정확도를 크게 향상시키지만 GPU 메모리 및 추론 대기시간에 부담을 줍니다. 반면에 자막은 특히 긴 비디오에서 경량화된 효과적인 대안을 제공합니다. 이 연구는 비디오QA 파이프라인 설계에 있어 자원 감안의 실질적인 통찰을 제공합니다.



### Reinforced Preference Optimization for Recommendation (https://arxiv.org/abs/2510.12211)
- **What's New**: 이 논문에서는 최근 대형 언어 모델(LLMs)을 활용한 생성 추천 시스템의 혁신을 다루고 있습니다. 기존의 추천 시스템이 지향하는 차별화(discriminative) 방식에서 생성(generative) 방식으로의 패러다임 전환을 통해, 사용자 행동 모델링이 과거 상호작용에 기초하여 목표 아이템을 생성하는 방식으로 변화하고 있습니다. 특히, 기존의 생성 추천 시스템이 겪고 있는 하위 품질 부정성 모델링 및 암묵적 보상에 대한 의존성을 해결하기 위한 새로운 방법론인 Reinforced Preference Optimization for Recommendation (ReRe)을 제안합니다.

- **Technical Details**: ReRe는 강화 학습 기반의 패러다임으로, LLM 기반 추천 시스템에 맞춤형으로 설계되었습니다. 이 방법론은 비효율적인 샘플링을 개선하기 위해 제약된 빔 검색(constrained beam search)을 도입하고, 강력한 부정 샘플을 다양화하는 동시에, 규칙 기반의 정확도 보상에 보조적인 순위 보상을 추가하여 미세 조정된 감독을 구현합니다. 이를 통해 LLM 기반 추천 시스템의 성능이 향상됩니다.

- **Performance Highlights**: 세 개의 실제 데이터셋에서 진행한 광범위한 실험 결과, ReRe가 전통적인 추천 시스템 및 LLM 기반 추천 시스템 모두에서 일관되게 우수한 순위 성능을 보여주었습니다. 추가적인 분석을 통해 ReRe는 기본 모델 및 SFT 모델 모두에서 성능을 향상시키며, 다양한 백본 가족과 규모에 걸쳐 강력한 일반화 능력을 보인다고 합니다. 더 나아가, 이 연구는 생성, 샘플링 전략, 보상 모델링 및 최적화 알고리즘에 대한 RLVR의 설계 공간을 체계적으로 탐구하여 향후 연구에 대한 통찰력을 제공합니다.



### MIARec: Mutual-influence-aware Heterogeneous Network Embedding for Scientific Paper Recommendation (https://arxiv.org/abs/2510.12054)
- **What's New**: 이 논문은 Mutual-Influence-Aware Recommendation(MIARec) 모델을 제안하여 다른 추천 방법들이 간과한 비대칭적인 학문적 영향력을 평가합니다. MIARec은 중력 기반 접근 방식을 사용하여 학자들 간의 상호 영향을 측정하고 그래프 표현 학습 과정에서 이를 포함시켜 보다 정교한 추천을 제공합니다. 또한, 모델은 다중 채널 집계 방식을 활용하여 다양한 관계의 서브 네트워크에서 개별 임베딩을 포착하고 상호 의존적 임베딩을 통해 이질적인 학술 네트워크에 대한 포괄적인 이해를 가능하게 합니다.

- **Technical Details**: MIARec 모델은 네트워크 내에서 학자들 간의 관계를 집계하는 다양한 방법을 따릅니다. 특히, 노드 feature의 집계 시, 상호 영향을 고려하여 비대칭적 영향력을 반영하는 메커니즘을 도입했습니다. 이는 GNN 기반 접근 방식을 사용하여 이루어지며, 다중 채널에서 독립적으로 학습된 single-relational subgraph의 임베딩을 모두 포함하여 보다 정확한 추천을 가능하게 합니다.

- **Performance Highlights**: 실제 데이터셋에 대한 광범위한 실험을 통해 MIARec 모델이 세 가지 주요 평가 지표에서 평균 0.78%, 2.60%, 3.17% 향상된 결과를 보였음을 확인했습니다. DBLP 및 ACM 데이터셋에서 비교 모델보다 우수한 성능을 입증하며, 각 모듈의 효과성을 확인하기 위한 ablation study도 진행되었습니다. 이 결과들은 MIARec이 학술 논문 추천 작업에서 효과적임을 보여줍니다.



### Embedding the Teacher: Distilling vLLM Preferences for Scalable Image Retrieva (https://arxiv.org/abs/2510.12014)
- **What's New**: 이번 논문에서는 텍스트-이미지 검색 분야에서 제품 추천을 위한 효율적인 방법을 제안합니다. 기존의 임베딩 기반 접근법인 CLIP은 문자 그대로의 자막 유사 텍스트-이미지 쌍에 주로 학습되어 있으며, 이것이 추상적인 특성을 포착하는 데 실패합니다. 제안된 프레임워크는 강력한 비전-언어 모델(vLLM)의 선호 순위를 임베딩 기반 시스템에 증류하여, 개인화된 추천을 위한 텍스트-이미지 검색의 효율성을 극대화합니다.

- **Technical Details**: 제안된 방법은 자연어 설명에 따라 적합한 이미지를 선택하는 것을 목표로 합니다. 여기서 설명은 추상적이거나 페르소나 기반일 수 있으며, 수천 개의 제품 이미지를 포함하는 카탈로그에서 단일 이미지를 식별하는 것이 필요합니다. 강력한 vLLM을 사용하여 이미지를 비교하고 최적의 이미지를 선택하는 과정에서 발생하는 고비용의 N-1 쌍비교 방식을 회피하기 위해 임베딩 기반 모델을 훈련시킵니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존의 임베딩 기반 기준보다 성능이 크게 향상되었습니다. 이는 개인화된 텍스트-이미지 검색 문제를 해결하는 효율적이고 확장 가능한 솔루션을 제공함을 의미합니다. 최적화된 임베딩 모델은 대규모 카탈로그에 대한 빠른 추론을 가능하게 하며, 수작업 레이블링이 필요 없는 장점도 지니고 있습니다.



### DeepMMSearch-R1: Empowering Multimodal LLMs in Multimodal Web Search (https://arxiv.org/abs/2510.12801)
- **What's New**: 이번 연구에서는 DeepMMSearch-R1이라는 새로운 다중 모달 LLM을 소개합니다. 이 모델은 웹에서의 다단계 검색을 수행할 수 있는 최초의 모델로, 이미지 및 텍스트 검색 도구를 위해 동적으로 쿼리를 생성합니다. 특히 이 모델은 입력 이미지의 관련 부분을 기반으로 웹 검색을 시작하여 이미지 검색의 효율성을 높이고, 수집된 정보를 바탕으로 텍스트 검색 쿼리를 점진적으로 조정할 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: DeepMMSearch-R1은 두 단계의 훈련 파이프라인을 따릅니다. 첫 번째 단계는 감독된 미세 조정(Supervised Finetuning, SFT) 단계이며, 그 다음 온라인 강화 학습(Online Reinforcement Learning, RL) 최적화를 수행합니다. 연구팀은 DeepMMSearchVQA라는 새로운 다중 모달 VQA 데이터셋을 소개했으며, 이 데이터셋은 웹 검색 도구에서 수집된 실제 정보를 포함하여 적합한 질문과 이미지를 훈련할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험을 통해 DeepMMSearch-R1은 최신 상태의 성과를 기록하며, 이전의 오픈 소스 기준을 초월하는 성능을 달성했습니다. 두 단계의 훈련 과정과 자가 반성, 자가 수정, 그리고 크롭된 이미지 검색의 영향을 통해 성능에 긍정적인 영향을 미쳤음을 입증했습니다. 이로 인해 다중 모달 웹 검색 도구의 통합을 촉진하는 유용한 자료를 제공할 수 있었습니다.



### CTRL-Rec: Controlling Recommender Systems With Natural Languag (https://arxiv.org/abs/2510.12742)
- **What's New**: 본 논문에서는 전통적인 추천 시스템에 자연어(Natural Language) 제어를 통합하는 CTRL-Rec 방법을 제안합니다. 이 접근 방식은 사용자가 명시적으로 제시한 선호와 참여 신호(Engagement Signals) 사이의 균형을 이루고 현대 추천 시스템의 검색 단계에 직접 영향을 미칠 수 있습니다. 이는 사용자가 간단한 자연어 요청으로 추천 내용을 즉각적으로 업데이트할 수 있게 합니다.

- **Technical Details**: CTRL-Rec은 LLM(대규모 언어 모델)을 활용하여 사용자의 자연어 요청을 근거로 특정 항목에 대한 사용자 판단을 시뮬레이트합니다. 이를 통해 생성된 예측 결과는 전통적인 추천 시스템에서 최적화하는 신호의 가중치에 통합되어, 명시된 선호와 참여 간의 균형을 이루게 됩니다. 이 시스템은 각 사용자 요청에 대해 단일 LLM 임베딩 계산만을 요구하여 실시간 추천 제어를 가능하게 합니다.

- **Performance Highlights**: 실험 결과, CTRL-Rec은 MovieLens 데이터셋에서 다양한 요청에 대해 세밀한 제어를 가능하게 했습니다. 19명의 Letterboxd 사용자를 대상으로 한 연구에서도 사용자들은 CTRL-Rec에 긍정적인 반응을 보였으며, 추천 시스템에 대한 통제감과 만족도가 전통적인 통제 방식에 비해 크게 향상된 것으로 나타났습니다.



### Evaluating Retrieval-Augmented Generation Systems on Unanswerable, Uncheatable, Realistic, Multi-hop Queries (https://arxiv.org/abs/2510.11956)
- **What's New**: 본 연구에서는 쿼리의 복잡성을 평가하는 새로운 벤치마크, CRUMQs(Cheat-Free Realistic Unanswerable Multi-hop Queries)를 소개합니다. 기존 RAG( retrieval augmented generation) 시스템은 복잡한 multi-hop 쿼리나 unanswerable 쿼리를 처리하는 능력이 부족한 반면, CRUMQs는 이러한 문제를 해결할 수 있는 방법을 제공합니다. 이 파이프라인은 모든 도메인에 적용 가능하며 자동으로 생성된 쿼리는 RAG 시스템의 한계를 더 잘 드러냅니다.

- **Technical Details**: RAG 시스템은 외부 문서 컬렉션을 활용하여 다양한 사용자 요청에 응답하는 방식입니다. 연구진은 기존 데이터와 서로 연관된 최신 기사를 크롤링하여 정보를 수집하고, 이를 통해 fully unanswerable 및 partially unanswerable 쿼리를 생성합니다. 이 과정에서 LangChain을 사용하여 정보를 1,024-token 청크로 분할하고, LLM을 통한 유사도 기반 필터링을 수행하여 최종 쿼리를 생성합니다.

- **Performance Highlights**: CRUMQs는 기존의 RAG 벤치마크와 비교하여 RAG 시스템에 대한 높은 도전 과제가 된다는 것이 입증되었습니다. 실험 결과, CRUMQs는 업계 최고의 모델인 GPT-5에서도 notable difficulty를 제공하며, 이전 벤치마크보다 최대 81.0%의 cheatability 점수 감소를 기록했습니다. 이러한 결과는 RAG 시스템의 신뢰성을 향상시키고, 더 강력한 시스템 개발에 기여할 것으로 기대됩니다.



### FinVet: A Collaborative Framework of RAG and External Fact-Checking Agents for Financial Misinformation Detection (https://arxiv.org/abs/2510.11654)
- **What's New**: 본 논문에서는 FinVet이라는 새로운 멀티 에이전트 프레임워크를 소개하며, 이는 두 개의 Retrieval-Augmented Generation (RAG) 파이프라인과 외부 사실 확인을 통합하는 신뢰도 가중 투표 메커니즘을 통해 작동합니다. FinVet은 동적으로 검증 전략을 조정하는 세 가지 처리를 통해 기존 방법의 한계를 극복하는 동시에 더 높은 투명성을 제공합니다. 이를 통해 증거 기반의 판결, 출처 귀속, 신뢰도 점수를 제공하며, 불충분한 증거에 대한 불확실성 표시를 명확히 하게 됩니다.

- **Technical Details**: FinVet은 세 가지 계층 처리 전략을 도입하여 신뢰도 점수에 따라 검증 접근 방식을 동적으로 선택합니다. 이는 높은 신뢰도의 경우 직접 메타데이터 추출, 중간 신뢰도의 경우 하이브리드 모델 추론, 낮은 신뢰도의 경우 모델 기반 분석을 통해 이루어집니다. RAG 구성요소는 도메인 특정 데이터셋에 맞춘 외부 지식소스를 활용하며, 사실 확인 파이프라인은 직접적인 증거가 없을 경우 대체 메커니즘을 사용하여 처리합니다.

- **Performance Highlights**: FinVet의 성능 평가 결과는 FinFact 데이터셋을 기준으로 하여 F1 점수 0.85를 달성하였으며, 이는 기존의 최고 단일 파이프라인(사실 확인 파이프라인) 대비 10.4% 개선된 결과입니다. 또한, 스탠드얼론 RAG 접근법에 비해 37%의 성능 향상도 보여주었습니다. 이 결과는 FinVet이 기존 기술과 비교할 때 보다 나은 정확성과 설명 가능성을 제공함을 입증합니다.



### OneRec-Think: In-Text Reasoning for Generative Recommendation (https://arxiv.org/abs/2510.11639)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)의 강력한 생성 능력을 활용하여 추천 시스템에서의 패러다임 전환을 소개합니다. 기존 모델들이 명시적이고 통제된 추론(capacity for explicit and controllable reasoning) 기능이 결여된 반면, 제안된 OneRec-Think는 대화, 추론, 개인화 추천을 통합한 새로운 프레임워크입니다. 이 프레임워크는 여러 단계로 구성되어 있으며, 각 단계는 추천의 정확성과 사용자 신뢰성을 높이는 데 기여합니다.

- **Technical Details**: OneRec-Think의 핵심 요소 세 가지는 Itemic Alignment, Reasoning Activation, Reasoning Enhancement입니다. Itemic Alignment는 교차 모드(item-textual) 정렬을 통해 의미적 기반을 구축하고, Reasoning Activation은 LLM의 본래 추론 능력을 활성화하며, Reasoning Enhancement는 추천에 특화된 보상 함수를 설계하여 사용자 선호도의 다중 유효성을 반영합니다. 이 모델은 효율적이고 즉각적인 산업 배포를 위해 'Think-Ahead' 아키텍처를 포함합니다.

- **Performance Highlights**: 모델의 다양한 실험 결과, 추천 정확성과 재능 기반 신뢰에서 최첨단 성능을 달성했으며, Kuaishou에서의 산업적 배포를 통해 APP Stay Time에서 0.159%의 개선을 확인했습니다. 이는 모델의 명시적 추론 기능의 실제 효과성을 입증하며, 추론 방법의 접근성을 높이는 데 기여합니다.



### REGENT: Relevance-Guided Attention for Entity-Aware Multi-Vector Neural Re-Ranking (https://arxiv.org/abs/2510.11592)
Comments:
          To be published in: Proceedings of the 2025 Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region (SIGIR-AP 2025)

- **What's New**: 현재 신경 재순위를 수행하는 모델들은 복잡한 정보 요구와 긴 내용이 풍부한 문서에서 어려움을 겪고 있다. 본 논문에서 소개하는 REGENT는 이러한 문제를 해결하기 위해 인간처럼 이해하는 방식으로 개체를 '시맨틱 스켈레톤'(semantic skeleton)으로 사용하여 주의를 유도한다. REGENT는 관련성 지도를 주의 메커니즘에 직접 통합하여 정교한 어휘 일치와 고급 의미적 추론을 결합하며, 이를 통해 복잡한 질문에서도 중요한 내용을 집중적으로 처리한다.

- **Technical Details**: REGENT는 문서 토큰을 나타내는 정밀 벡터와 의미적 개체를 나타내는 고급 벡터를 병렬로 처리하는 다중 벡터 아키텍처로 구성된다. 이 모델은 토큰 수준에서 BM25 점수를 사용하여 어휘적으로 관련 있는 용어를 강조하고, 쿼리-특정 개체 표현을 사용하여 의미상 중요한 개념에 집중한다. 결과적으로 REGENT는 긴 문서 내에서 가장 관련성 높은 내용을 지향하며, 효율적인 개체-기반 정보 검색을 위한 새로운 패러다임을 제시한다.

- **Performance Highlights**: REGENT는 세 가지 대규모 데이터세트에서 새로운 최첨단 성능을 달성했으며, BM25에 비해 최대 108% 향상된 성능을 보인다. 또한, 기존의 강력한 모델인 ColBERT 및 RankVicuna를 지속적으로 초월한다. 중요한 점은 모델에서 개체 컴포넌트를 제거할 경우 성능이 74% 감소한다는 점으로, 이는 의미적 스켈레톤이 효율적인 장기 문서 검색에 매우 중요하다는 것을 강조한다.



### QDER: Query-Specific Document and Entity Representations for Multi-Vector Document Re-Ranking (https://arxiv.org/abs/2510.11589)
Comments:
          Published in: Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2025)

- **What's New**: 이번 논문에서는 QDER라는 새로운 신경 재정렬 모델을 소개하며, 이 모델은 지식 그래프(knowledge graph) 의미론을 다중 벡터 모델에 통합하여 엔터티가 인식되는 검색(entity-aware retrieval)을 개선하는 데 중점을 둡니다. QDER의 주요 혁신은 쿼리-문서 관계를 모델링하는 새로운 접근법으로, 집계된 임베딩에서의 유사도 점수를 계산하는 대신, 각 토큰과 엔터티 표현을 유지하며 최종 점수 매기기에서만 집계를 수행합니다. 이를 'late aggregation'이라고 부르며, QDER는 복잡한 쿼리를 효과적으로 처리할 수 있는 기반을 다집니다.

- **Technical Details**: QDER는 동적 주의(attention) 메커니즘을 적용하여 쿼리와 관련된 문서의 토큰 및 엔터티 표현을 변환하며, 이로 인해 쿼리에 맞춤화된 표현을 생성합니다. 또한, 전통적인 유사도 지표 대신 이차선형 프로젝션(bilinear projection)을 사용해 복잡한 쿼리-문서 관계를 모델링하며, 이는 노이즈에 대한 강인성을 높입니다. QDER의 문서 표현은 쿼리 별로 조정된 방식으로 생성되며, 각기 다른 조작인 더하기(addition)와 곱하기(multiplication)를 통해 보다 정밀한 매칭을 구현합니다.

- **Performance Highlights**: 실험 결과, QDER는 TREC Robust 2004 데이터세트에서 가장 강력한 기준선 대비 nDCG@20에서 36% 향상된 성능을 보여줍니다. 특히, 어려운 쿼리에서 QDER는 nDCG@20을 0.70으로 달성하며, 이는 기존의 전통적인 방법들이 실패하는 경우(nDCG@20 = 0.0)에서 두드러진 성과입니다. 이러한 결과는 QDER가 엔터티 인식을 통한 정보 검색에서의 새로운 기준을 세우는 데 기여함을 보여줍니다.



### Characterizing Web Search in The Age of Generative AI (https://arxiv.org/abs/2510.11560)
- **What's New**: 본 논문에서는 Generative AI 모델과 전통적인 웹 검색 엔진 간의 주요 차이점들을 탐구합니다. Generative search는 사용자 쿼리에 대한 응답으로 독립적인 웹 페이지 리스트를 반환하는 대신, 관련 정보를 종합하여 일관된 텍스트로 결과를 제공합니다. 이를 통해 보다 다양한 소스와 관점을 이용할 수 있는 가능성이 열리게 됩니다.

- **Technical Details**: Generative search는 여러 차원에서 전통적인 웹 검색과 다른데, 첫째로 결과 포맷이 다릅니다. 전통적인 검색 결과는 주로 페이지들의 리스트로 제공되는 반면, Generative search는 단일한 응답 형식으로 정보를 제공합니다. 둘째로, Generative search는 선택된 웹 페이지의 수보다 더 넓은 범위를 커버할 수 있는 능력을 가지며, 내부 지식과 외부 웹에서 검색된 정보를 결합하여 사용합니다.

- **Performance Highlights**: 연구 결과, Generative search 엔진은 전통적인 검색 엔진보다 소스 다양성이 더 높으며, 각 엔진이 내부 지식에 의존하는 정도는 다릅니다. 예를 들어, GPT-Tool은 평균적으로 0.4개의 웹 페이지를 참고하는 반면, AIO와 Gemini는 각각 8.6, 8.5개를 참조합니다. 이는 정보 제공의 투명도와 신뢰성, 사용자 자율성에 미치는 잠재적 영향에 대한 고려가 필요함을 시사합니다.



### Uncertainty Quantification for Retrieval-Augmented Reasoning (https://arxiv.org/abs/2510.11483)
- **What's New**: 이 논문은 Retrieval-Augmented Reasoning (RAR)의 새로운 불확실성 정량화 방법인 Retrieval-Augmented Reasoning Consistency (R2C)를 제안합니다. R2C는 RAR에서 발생하는 다양한 불확실성의 출처를 고려하여 신뢰성 있는 출력을 생성하는 데 목표를 두고 있습니다. 실험 결과, R2C는 기존의 방법들보다 평균 5% 향상된 성능을 보여준다는 점에서 중요성이 강조됩니다.

- **Technical Details**: R2C는 Markov Decision Process (MDP)를 사용하여 RAR의 다단계 추론 과정을 모델링하고, 다양한 행동 패턴을 통해 이 과정을 조정합니다. 이로써 추론 단계에서의 다양한 경로와 쿼리를 탐색하며, 각 생성된 최종 답변의 일관성을 측정합니다. R2C는 평균적으로 25개의 독특한 문서를 수집하며, 쿼리 다양성 지표에서 기존 방법보다 우수한 성과를 보입니다.

- **Performance Highlights**: R2C는 F1Abstain 및 AccAbstain에서 약 5% 향상된 성능을 달성하고, 모델 선택에서 단일 RAR 모델에 비해 약 7%의 정확도를 개선했습니다. 이 방법은 기존 방법들보다 약 2.5배 더 적은 토큰 생성을 요구하며, 상대적으로 높은 효율성을 보여줍니다. 최종적으로, R2C는 RAR 시스템에 대한 신뢰성과 성능을 크게 향상시키는 것으로 입증되었습니다.



### What Generative Search Engines Like and How to Optimize Web Content Cooperatively (https://arxiv.org/abs/2510.11438)
- **What's New**: 이 논문은 Generative Engines(GEs)의 선호도를 파악하고, 이를 기반으로 효율적인 Generative Engine Optimization(GEO) 모델을 개발하는 AutoGEO라는 새로운 프레임워크를 소개합니다. AutoGEO는 대규모 언어 모델을 활용하여 생성 엔진의 문서 선호도를 자동으로 분석하고, 이를 통해 획득한 규칙을 이용해 웹 콘텐츠를 최적화합니다. 이 시스템은 문서의 가시성을 향상시키면서 사용자가 정보에 대한 요구를 더 잘 충족시킬 수 있도록 돕습니다.

- **Technical Details**: AutoGEO는 Generative Engines의 선호 규칙을 학습하기 위해 LLMs를 활용하여 문서 쌍 간의 가시성 차이를 분석하고, 이를 바탕으로 간결한 통찰을 추출하여 후보 규칙으로 통합합니다. 다음으로, 이러한 규칙을 사용하여 GEO 모델을 구성하고, 두 가지 버전의 모델을 개발합니다: AutoGEOAPI{}_{	ext{API}}와 AutoGEOMini{}_{	ext{Mini}}. AutoGEOMini는 강화 학습(RL)을 통해 최적화되며, 효과적인 재작성 데이터셋을 생성하여 안정적인 시작을 꾀합니다.

- **Performance Highlights**: 실험 결과, AutoGEO 모델은 GEO 메트릭에서 평균 35.99% 향상된 성능을 보여주며, Generative Engine Utility(GEU)를 유지합니다. 각 LLM은 고유한 선호 규칙을 가지고 있으며, 이러한 엔진별 규칙을 적용한 결과 기존 규칙을 사용할 때보다 높은 GEO 성능을 보였습니다. 특히 AutoGEOMini는 비용 효율적인 모델로, AutoGEOAPI{}_{	ext{API}}의 약 0.0071배의 비용으로 우수한 성능을 발휘합니다.



### On Inherited Popularity Bias in Cold-Start Item Recommendation (https://arxiv.org/abs/2510.11402)
Comments:
          Published at ACM RecSys 2025

- **What's New**: 이 논문에서는 콜드스타트(cold-start) 추천 시스템이 따르는 인기 편향(popularity bias)에 대한 새로운 통찰을 제공합니다. 인기 있는 아이템에 대한 과도한 선호를 반영하도록 훈련된 모델들로 인해, 콜드스타트 모델이 데이터에서의 상호작용 정보를 활용하지 못하고 아이템의 콘텐츠 특성만으로 인기 예측을 시도합니다. 이로 인해 특정 콜드 아이템의 예측이 과장되며, 공정성이 저하되는 문제를 드러냅니다.

- **Technical Details**: 연구는 콜드스타트 추천 모델이 과거의 CF 모델에서 예측 편향을 얼마나 심각하게 물려받는지를 분석합니다. 데이터 세트는 MMRec 툴박스에서 제공되는 Micro-video, Clothing, Electronics와 같은 다양한 멀티미디어 데이터로 구성됩니다. 실험은 이전 연구에서 제안된 알고리즘을 사용하여 L2 정규화를 통해 데이터 세트의 콘텐츠 벡터를 결합하는 방식으로 진행됩니다.

- **Performance Highlights**: 제안된 방법은 예측 분포의 균형을 맞추는 데 효과적이며, 사용자 지향의 콜드스타트 정확도를 최소한으로 해치면서도 아이템 공정성을 향상시키는 것을 입증했습니다. 실험 결과, 세 가지 생성 모델 모두에서 이러한 단순한 사후 처리 방법이 아이템 공정성을 개선하는 데 기여하는 것으로 나타났습니다. 연구에서는 결과 재현을 위한 코드와 자원도 제공하고 있습니다.



### VeriCite: Towards Reliable Citations in Retrieval-Augmented Generation via Rigorous Verification (https://arxiv.org/abs/2510.11394)
- **What's New**: 본 논문에서는 Retrieval-Augmented Generation (RAG) 기법을 통해 대형 언어 모델(LLM)의 응답 품질을 개선하기 위한 새로운 프레임워크인 VeriCite을 제안합니다. VeriCite는 응답 생성 후 인용을 엄격히 검증하는 세 단계의 프로세스로 구성되어 있습니다. 이 과정은 초기 답변 생성, 지원 증거 선택, 최종 답변 정제를 포함하여 신뢰할 수 있는 인용을 보장합니다.

- **Technical Details**: VeriCite의 첫 번째 단계에서는 모든 검색된 문서에 기반한 응답을 생성하고, Natural Language Inference (NLI) 모델을 이용해 주장의 신뢰성을 검증합니다. 두 번째 단계인 지원 증거 선택에서는 각 문서의 유용성을 평가하여 필요한 증거를 추출합니다. 마지막으로, 최종 답변 정제 단계에서 초기 응답과 추출된 증거를 통합하여 최종적인 개선된 답변을 생성하게 됩니다.

- **Performance Highlights**: 다양한 공개 데이터셋과 LLM을 활용한 실험에서 VeriCite는 인용 생성 품질을 크게 향상시키면서도, 정답의 정확도를 기준선과 동등하게 유지하는 성과를 보였습니다. 이를 통해 LLM의 정보 추출 능력을 감소시키고, 입력 불확실성을 줄이며, 인용의 신뢰도를 높이는 데 기여하고 있습니다.



### Dynamic Network-Based Two-Stage Time Series Forecasting for Affiliate Marketing (https://arxiv.org/abs/2510.11323)
- **What's New**: 이번 연구에서는 제휴 마케팅(affiliate marketing)에서의 프로모터의 기여도를 평가하고 예측하는 데 있어 중요한 문제를 다루고 있습니다. 연구팀은 프로모터의 간접 기여도를 측정하기 위해 새로운 메트릭인 propagation scale을 설계하였으며, 이 메트릭의 예측을 위한 이중 단계 솔루션을 제안합니다. 또한, 이 연구는 GNN 기반의 기존 시계열 예측 기술이 제휴 마케팅의 복잡한 동적 상황을 처리하는 데 한계가 있음을 지적하고, 이를 해결하기 위한 방법론을 제시합니다.

- **Technical Details**: 정교한 프로모션 네트워크를 예측하기 위해 연구팀은 프로모터의 기본 self-sales와 네트워크 구조를 분리해서 예측하며, 하이퍼그래프(convolution)와 그래프 합성(encoders)을 이용해 프로모션 역학을 효율적으로 포착합니다. 여기서 제안된 이중 단계 예측 기법(DNTS)은 기본 self-sales와 프로모션 구조를 결합하여 간접 기여를 측정하는 데 중점을 둡니다. 연구는 다양한 보조 작업을 도입해 예측의 정확도를 높이고, 유의미한 결과를 도출하는 데 기여합니다.

- **Performance Highlights**: 연구팀은 Alimama 플랫폼에서 100,000명 이상의 프로모터와 함께 본 모델을 배포하고, GMV 및 판매량에서 각각 9.29%와 5.89%의 향상을 달성했습니다. 대규모 산업 데이터셋에 대한 오프라인 실험을 통해 제안된 방법의 우수성을 입증하였으며, 기존 GNN 방법론의 한계를 극복했다는 점에서도 중요한 성과를 거두었습니다. 이를 통해 제휴 마케팅 분야의 시계열 예측의 새로운 가능성을 제시합니다.



### Next Interest Flow: A Generative Pre-training Paradigm for Recommender Systems by Modeling All-domain Movelines (https://arxiv.org/abs/2510.11317)
- **What's New**: 이번 논문에서는 현대 추천 시스템의 기반인 Click-Through Rate (CTR) 예측을 위한 혁신적인 생성적 전처리 패러다임을 제안합니다. 기존의 접근 방식들은 사용자의 행동을 반응적으로 모델링했지만, 우리는 사용자 의도를 능동적으로 예측하는 방법으로 전환합니다. 이 새로운 방법론은 Next Interest Flow라는 개념을 중심으로 구축되어 있으며, 향후 사용자 관심을 밀집된 벡터 시퀀스로 표현합니다.

- **Technical Details**: 제안된 AMEN 프레임워크는 두 단계로 구성됩니다. 첫 번째 단계인 생성적 전처리에서는 Transformer 기반의 디코더가 과거의 moveline을 기반으로 Next Interest Flow를 예측합니다. 두 번째 단계인 차별적 미세 조정에서는 훈련된 CTR 모델이 이 생성된 흐름을 주요 입력 특징으로 사용하여 최종 예측을 수행합니다.

- **Performance Highlights**: AMEN은 강력한 기준 모델과 비교하여 우수성을 입증하는 광범위한 오프라인 실험 결과를 제시하며, 대규모 온라인 A/B 테스트에서도 실제 비즈니스 성과에서 큰 효과를 보입니다. 이를 통해 CTR 예측 성능을 비약적으로 향상시키는 포괄적 시스템을 구현하였습니다.



### DyKnow-RAG: Dynamic Knowledge Utilization Reinforcement Framework for Noisy Retrieval-Augmented Generation in E-commerce Search Relevanc (https://arxiv.org/abs/2510.11122)
- **What's New**: DyKnow-RAG는 음의 노이즈를 극복하고 외부 컨텍스트를 활용하여 전자상거래의 relevancy를 향상시키는 혁신적인 모델입니다. 이 모델은 Group Relative Policy Optimization (GRPO) 기반으로 두 개의 롤아웃 그룹을 구성하여 외부 컨텍스트의 활용 여부를 판단하도록 훈련합니다. 또한, 프로세스 레이블이나 가치 네트워크를 사용하지 않고도, 단일 패스에서 적용할 수 있는 외부 지식을 효과적으로 제어합니다.

- **Technical Details**: DyKnow-RAG는 동적인 지식 활용 강화를 통해 노이즈가 포함된 Retrieval-Augmented Generation (RAG) 모델을 구축합니다. 이 프레임워크는 외부 컨텍스트를 사용하여 쿼리의 relevance를 평가하고, 환경에 맞춰 적절한 피드백 루프를 형성합니다. 이를 위해, RL pool을 통해 불확실성이 높은 부분에 대한 업데이트를 집중하여 훈련하고, 상황에 맞는 DPO(Direct Preference Optimization)의 초기 시작으로 더 빠른 조정을 가능하게 합니다.

- **Performance Highlights**: DyKnow-RAG는 Taobao의 생산 relevancy 시스템에 통합되어 실행되고 있으며, 오프라인 테스트에서 SFT 및 DPO와 같은 기존 모델들을 초월하는 성능을 보여줍니다. 일관된 개선 효과를 확인하고, GSB, 쿼리 성공률, 아이템 성공률 등에서 긍정적인 결과를 나타냅니다. 이는 전자상거래 relevancy를 위한 최초의 단일 패스 RAG 솔루션 중 하나로, 불필요한 복잡성 없이 신뢰할 수 있는 성과를 도출합니다.



### HoMer: Addressing Heterogeneities by Modeling Sequential and Set-wise Contexts for CTR Prediction (https://arxiv.org/abs/2510.11100)
Comments:
          10 pages, 6 figures

- **What's New**: 이 논문은 클릭률(CTR) 예측에서의 세 가지 이질성 문제를 해결하기 위해 HoMer라는 새로운 모델을 제안합니다. HoMer는 동시적이고 세트 기반의 상호작용을 모델링할 수 있는 동형 지향 변환기(Homogeneous-Oriented TransforMer)로 고안되었습니다. 이를 통해 사용자 행동을 보다 정밀하게 반영하고 예측 정확도를 개선할 수 있습니다.

- **Technical Details**: HoMer는 비순차적(features)과 순차적(sequence) 특성을 정렬하여 세밀한 사용자 관심 표현을 가능하게 합니다. 전통적인 CTR 예측 방식이 각 아이템 별로 샘플을 구성하는 것과 달리, HoMer는 모든 아이템에 대해 단일 샘플에서 비순차적 특성을 집계하여 공동 예측을 수행합니다. 또한, 통합된 인코더-디코더 아키텍처를 활용하여 효율적인 처리를 구현하고, 사용자 행동 패턴을 보다 잘 모델링할 수 있도록 합니다.

- **Performance Highlights**: HoMer는 Meituan이라는 중국의 플랫폼에서 0.0099 AUC 개선과 함께 1.99%의 CTR 증가 및 2.46%의 RPM 향상을 보였습니다. 초기 개발 최적화를 통해 GPU 리소스 소비를 27% 줄이는 등의 성과도 확인되었습니다. 이처럼 HoMer는 성능과 효율성에서 뛰어난 이점을 가지고 있으며, 산업 기준을 초과하는 성능을 입증했습니다.



### Decoupled Multimodal Fusion for User Interest Modeling in Click-Through Rate Prediction (https://arxiv.org/abs/2510.11066)
- **What's New**: 이번 연구에서는 Decoupled Multimodal Fusion (DMF)을 제안하여 ID 기반의 협업 표현과 다중 모드 표현 간의 세밀한 상호작용을 가능하게 하는 모달리티 강화 모델링 전략을 도입합니다. 이는 고차원 다중 모드 정보를 효과적으로 통합할 수 있는 새로운 접근 방식을 제공하며, 사용자의 관심 모델링을 개선하는 데 기여합니다. 또한, 효율적인 추론을 제공하는 Nov에 기반한 Attention 메커니즘을 통해 계산 병목 현상을 완화하고, 다중 모드 전반에 걸쳐 종합적인 결합을 도모합니다.

- **Technical Details**: DMF는 사용자 관심 모델링을 위한 모달리티 강화 모델링 전략을 통해 ID 기반 협업 표현과 다중 모드 표현 간의 상호작용을 촉진합니다. 이를 위해 사용자는 다중 모드 유사성을 타겟 인식한 부가 정보로 활용하면서, Decoupled Target Attention (DTA) 아키텍처를 통해 효율적인 계산 처리를 달성합니다. 더불어, 모달리티 중심 및 모달리티 강화 전략의 강점을 통합하는 Complementary Modality Modeling (CMM) 모듈을 설계하여 사용자 관심 표현을 통합합니다.

- **Performance Highlights**: DMF의 효과는 대규모 오프라인 실험 및 Lazada의 제품 추천 시스템에 성공적으로 배포되어 나타났습니다. 실험 결과, DMF는 CTCVR에서 5.30%, GMV에서 7.43%의 상대적 개선을 달성하며, 매우 낮은 계산 오버헤드로 산업 규모의 요구를 충족했습니다. 이러한 성과는 DMF가 산업 추천 시스템에서의 효율성과 효과성을 동시에 증대시킬 수 있음을 시사합니다.



### From Reasoning LLMs to BERT: A Two-Stage Distillation Framework for Search Relevanc (https://arxiv.org/abs/2510.11056)
- **What's New**: 이 논문에서는 전자상거래 검색 시스템에서 쿼리-서비스 관련성 예측을 위한 두 단계 추론 증류(Reasoning Distillation) 프레임워크를 제안합니다. 첫 번째 단계에서는 도메인 적응형 선생 모델(Teacher Model)을 구축하여 일반적인 LLM의 한계를 극복합니다. 이어지는 두 번째 단계에서는 대조적 추론 자기 증류(Contrastive Reasoning Self-Distillation, CRSD) 방법을 도입하여 경량 모델이 복잡한 의사 결정을 내부화할 수 있도록 합니다.

- **Technical Details**: 제안된 프레임워크는 LLM의 추론 능력을 전이하기 위한 세 가지 과정으로 구성됩니다. 첫 번째는 도메인 지식을 주입하는 연속 사전 훈련, 두 번째는 추론 능력을 이끌어 내기 위한 감독된 미세 조정, 마지막은 다차원 보상 모델을 사용하여 선호 최적화를 시행하는 과정입니다. CRSD는 InfoNCE 대조 학습 메커니즘을 활용하여 다양한 아키텍처 간의 피처 정합성 문제를 해결하며, 경량 모델이 의사 결정 경로를 명시적으로 요구하지 않고도 추론 능력을 내부화할 수 있게 합니다.

- **Performance Highlights**: Meituan의 실제 전자상거래 검색 시스템에서 실시한 오프라인 평가 및 온라인 A/B 테스트 결과, 제안된 프레임워크는 여러 지표에서 유의미한 개선을 보이며, 추론 능력 전이에서 실용적 가치를 입증합니다. 이는 전통적 모델보다 높은 정확도의 관련성 예측을 가능하게 하며, 실시간 응답 시간과 효율성을 동시에 유지할 수 있습니다. 결과적으로, 이 연구는 산업 응용에서 추론 능력 전이를 위한 효과적인 솔루션을 제공합니다.



### Does LLM Focus on the Right Words? Diagnosing Language Bias in LLM-based Recommenders (https://arxiv.org/abs/2510.10978)
- **What's New**: 이 논문에서는 대규모 언어 모델(Large Language Models, LLMs)과 추천 시스템(Recommendation Systems, RS)의 통합을 새로운 관점에서 살펴본다. 특히, Supervised Fine-Tuning(SFT)에서 발생하는 언어 편향(Language Bias) 문제를 지적하며, Group Distributionally Robust Optimization 기반의 새로운 튜닝 방식인 GDRT(Group Distributionally Robust Tuning)를 제안한다. 이 방법은 보조 토큰(auxiliary tokens)에 대한 과도한 의존을 줄이고 사용자 상호작용 토큰(user interaction tokens)에 집중할 수 있도록 돕는다.

- **Technical Details**: GDRT는 그룹 분포적으로 강건한 최적화(Group DRO) 원리를 활용하여, 보조 토큰에 대한 관련성 정도가 다른 토큰 그룹 간의 일관된 모델 성능을 강제하는 새로운 세부 조정(paradigm)을 제공한다. 이는 훈련 샘플을 토큰 간의 의미적 연관성에 따라 그룹화하고, 효율적인 동적 샘플 가중치를 조정하여 이루어진다. 이러한 방법론은 LLM이 사용자의 개별적인 선호를 바탕으로 예측을 수행하도록 유도한다.

- **Performance Highlights**: 다양한 공개 데이터셋에서 수행된 실험 결과, GDRT는 언어 편향을 효과적으로 완화하고 추천의 정확성을 크게 향상시키는 것으로 나타났다. 평균 NDCG@10에서 약 24.29%의 성능 향상을 기록했으며, 추천의 공정성(fairness) 또한 현저하게 개선되었다. 이로 인해 GDRT는 추천 시스템의 정확성과 공정성 측면에서 최첨단 성능을 기록함을 보여준다.



### HatLLM: Hierarchical Attention Masking for Enhanced Collaborative Modeling in LLM-based Recommendation (https://arxiv.org/abs/2510.10955)
- **What's New**: 최근 대규모 언어 모델(LLM)을 활용한 순차 추천에 대한 연구가 증가하고 있습니다. LLM은 사용자의 미세한 선호도를 추론하는 데 뛰어난 잠재력을 보여주지만, 사용자 간의 협력 신호를 효과적으로 모델링하는 데는 한계가 있습니다. 이러한 한계를 극복하기 위해 연구팀은 HatLLM이라는 새로운 계층적 주의 마스킹 전략을 제안했습니다.

- **Technical Details**: HatLLM은 LLM의 각 층에서 서로 다른 마스킹 전략을 적용하여 다양한 정보를 모델링할 수 있도록 합니다. 얕은 층에서는 서로 다른 항목의 토큰 간의 주의를 마스킹하여 항목 내 의미를 이해하도록 하고, 깊은 층에서는 항목 내 주의를 마스킹하여 항목 간 상관관계를 강조합니다. 이 계층적 접근은 LLM이 토큰 수준과 항목 수준의 종속성을 공동으로 포착할 수 있게 합니다.

- **Performance Highlights**: HatLLM은 세 가지 실제 데이터 세트에서 엄청난 성과를 나타내며, 기존 LLM 기반 방법보다 평균 9.13%의 성능 향상을 달성했습니다. 이 연구는 LLM이 추천 시스템에서 협력 정보를 더 효과적으로 모델링할 수 있는 방법을 제시하며, 기존 방법과 비교해 우수한 결과를 보여줍니다.



### Comparative Explanations via Counterfactual Reasoning in Recommendations (https://arxiv.org/abs/2510.10920)
- **What's New**: 이번 연구에서는 CoCountER라는 새로운 방법을 제안했습니다. CoCountER는 소프트 스왑 작업을 기반으로 하여 추천시스템에서의 비교적 결과에 대한 설명을 생성합니다. 이 방법은 사용자가 관심을 가질만한 항목의 영향력 있는 측면을 올바르게 식별할 수 있도록 돕습니다.

- **Technical Details**: 기존의 추천 시스템에서는 아이템의 속성을 이용한 템플릿 같은 설명을 생성하는 방식이 주로 사용되었습니다. 그런데 최근 카운터팩추얼(상반된 사실)에 의한 접근법이 등장하면서, 추천결정이 뒤바뀔 때 최소한으로 입력 특성을 줄이는 방법이 도입되었습니다. 본 연구에서는 이러한 한계를 극복하기 위해 비교적 카운터팩추얼 설명을 도입하고 스왑 작업을 사용하여 특정 아이템 쌍에 대한 설명 생성을 최적화했습니다.

- **Performance Highlights**: 실험 결과, CoCountER 방법이 기존 추천 설명 방식보다 더 신뢰할 수 있는 설명을 제공함을 입증하였습니다. 특히, 이 방법은 각 아이템 쌍의 특성을 비교하여 추천의 주된 요인을 정확하게 판단할 수 있도록 지원합니다. 우리의 연구는 추천 시스템에서 이해 가능한 설명 구축에 대한 새로운 방향성을 제시합니다.



### VeritasFi: An Adaptable, Multi-tiered RAG Framework for Multi-modal Financial Question Answering (https://arxiv.org/abs/2510.10828)
- **What's New**: 이 논문에서 제안하는 VeritasFi는 금융 부문에서의 Question Answering (QA)에 혁신적인 접근 방식으로, Retrieval-Augmented Generation (RAG) 시스템의 두 가지 주요 문제점을 해결합니다. 첫 번째는 다양한 데이터 형식 처리의 어려움이며, 두 번째는 일반 도메인 적용성과 기업 관련 조정 간의 균형을 맞추는 데 있습니다. VeritasFi는 다중 모드 전처리 파이프라인과 고급 이중 단계 재정렬 전략을 통합하여 이러한 도전 과제를 극복합니다.

- **Technical Details**: VeritasFi는 다중 모드 데이터 형식을 일관되고 기계 인식할 수 있는 형식으로 변환하는 다중 모드 전처리 파이프라인을 포함하고 있습니다. 또한, Tripartite Hybrid Retrieval (THR) 엔진은 심층 다중 경로 검색과 실시간 데이터 수집, 전문가 큐레이션 메모리 뱅크를 결합하여 정확하고 효율적인 정보 검색을 보장합니다. 두 단계 재정렬 전략은 일반화된 도메인 모델을 구축한 다음 회사별 데이터에 맞춰 신속하게 조정하여 적용성을 높입니다.

- **Performance Highlights**: VeritasFi는 다양한 금융 QA 데이터 세트에서 기존의 강력한 기준들을 초월하는 최상의 end-to-end 성능을 달성하였으며, 특히 GraphRAG 및 LightRAG와 같은 기존 RAG 아키텍처에 비해 뛰어난 정확도와 맥락 적합성을 보여주었습니다. 고수준의 질문에 대한 검색 정확성을 향상시키기 위해 메타데이터 생성을 통해 각 청크에 대한 전반적인 배경 문맥을 유지하며, 최종적으로 구조적이고 신뢰성 있는 지식 베이스를 제공하여 빠른 답변 검색이 가능하게 합니다.



### Multi-Granularity Sequence Denoising with Weakly Supervised Signal for Sequential Recommendation (https://arxiv.org/abs/2510.10564)
- **What's New**: 이번 논문은 역사적 상호작용 시퀀스에서 사용자의 관심을 기반으로 다음 아이템을 예측하는 시퀀스 추천에 대한 연구입니다. 기존의 방법들이 아이템 단위의 노이즈를 제거하는데 한계를 보인 반면, 새롭게 제안된 Multi-Granularity Sequence Denoising with Weakly Supervised Signal (MGSD-WSS)은 아이템과 관심 단위의 노이즈를 계층적으로 제거하도록 설계되었습니다. 이를 통해 추천 시스템의 성능을 개선하고 더 정확한 노이즈 식별이 가능해졌습니다.

- **Technical Details**: MGSD-WSS는 원래의 시퀀스를 공통의 표현 공간으로 매핑하기 위해 Multiple Gaussian Kernel Perceptron 모듈을 도입합니다. 약한 지도 신호(Weakly Supervised Signal)를 활용하여 역사적 상호작용 시퀀스 내에서 노이즈 아이템을 정확하게 식별하고, 노이즈 가중치 대조 학습(Noise-weighted Contrastive Learning)을 통해 노이즈 간섭을 효과적으로 방지합니다. 이러한 방법론은 아이템과 관심 각각의 단위에서 계층적으로 노이즈를 제거할 수 있는 가능성을 제시합니다.

- **Performance Highlights**: 다섯 개의 데이터 세트에서의 광범위한 실험 결과, MGSD-WSS는 최신 시퀀스 추천 모델 및 노이즈 제거 모델에 비해 통계적으로 유의미한 성능 개선을 보였습니다. 특히, 약한 지도 신호와 계층적 노이즈 제거 모듈이 모델의 성능을 정립하는 중요한 요소로 분석되었습니다. 이 연구는 시퀀스 추천 분야에서 새로운 관점을 제공하며, 추천 시스템의 실질적인 향상을 위한 기술적 기초를 마련하였습니다.



### Self-Supervised Representation Learning with ID-Content Modality Alignment for Sequential Recommendation (https://arxiv.org/abs/2510.10556)
- **What's New**: 이번 논문에서는 ID-Content modal alignment을 기반으로 하는 새로운 모델 SICSRec을 제안합니다. 이 모델은 제한된 사용자 상호작용 기록에 대한 성능 저하 문제를 해결하기 위해 텍스트 및 이미지와 같은 아이템 콘텐츠 모달리티 정보를 통합하여 사용자 선호도를 파악합니다. 연구는 LLM을 활용한 샘플 구축 방법과 Transformer 기반의 인코더-디코더 아키텍처를 통해 사용자 행동과 콘텐츠 선호를 동시에 모델링하여 성능 개선을 도모합니다.

- **Technical Details**: SICSRec은 사용자 행동 선호를 포착하는 ID 모달리티 시퀀스 인코더와, 사용자 콘텐츠 선호를 학습하는 콘텐츠 모달리티 시퀀스 인코더, 두 가지 선호 간의 내재적 관계를 이해하는 혼합 모달리티 시퀀스 디코더로 구성된 Transformer 기반의 모델입니다. 두 단계의 학습 전략을 사용하여 콘텐츠 모달리티와 ID 모달리티 간의 정렬을 수행합니다. 이 과정에서 저자들은 L2 정규화와 콘텐츠 인지 대조 학습을 채택합니다.

- **Performance Highlights**: 다양한 공개 비디오 스트리밍 데이터셋에서 SICSRec은 최신 ID 모달리티 및 콘텐츠 모달리티 기반의 추천 모델보다 각각 NDCG@5에서 평균 8.04%, NDCD@10에서 6.62% 더 우수한 성능을 보였습니다. 이는 콘텐츠 모달리티 간의 의미적 간극을 줄이고 사용자 선호를 효과적으로 모델링하는 데 성공적인 결과를 나타냅니다. 또한, 각 구성 요소의 효과성을 검증하기 위한 상세한 실험과 ablation 연구를 수행하였습니다.



### Towards Long-Term User Welfare in Recommender Systems via Creator-Oriented Information Revelation (https://arxiv.org/abs/2510.10511)
- **What's New**: 이 논문은 추천 시스템(Recommendation Systems, RS)에서 장기 사용자 복지(long-term user welfare)를 향상시키기 위한 새로운 접근 방식을 제안합니다. 기존의 재정렬 알고리즘(re-ranking algorithms)은 단기 추천 정확도와 충돌하기 때문에 비효율적이라는 점을 강조하며, 정보 비대칭(information asymmetry)의 원리를 통해 창작자(creator)의 행동을 유도하는 정보를 노출시켜야 한다고 주장합니다. 이러한 아이디어를 기반으로, 정보 노출과정(information revelation process)을 마르코프 결정 과정(Markov Decision Process, MDP)으로 모델링한 LoRe(Information Revelation을 통한 장기 복지 최적화) 프레임워크를 제안합니다.

- **Technical Details**: LoRe 프레임워크는 고전적인 정보 노출 방법 중 하나인 베이지안 설득(Bayesian persuasion)을 활용하여 추천 시스템의 이해당사자(stakeholders)를 매핑합니다. 플랫폼은 정보 제공자(sender)로, 창작자는 정보 수신자(receiver)로 설정되어 있습니다. 이 과정에서 제한된 합리성을 갖는 창작자들과의 상호작용을 위한 강화 학습(reinforcement learning) 환경에서 학습하는 알고리즘을 제공합니다.

- **Performance Highlights**: LoRe의 실험 결과는 두 개의 실제 추천 데이터셋에서 기존의 공정 재정렬 방법 및 단순 정보 노출 전략보다 장기 사용자 복지를 효과적으로 개선할 수 있음을 보여줍니다. LoRe는 독립적인 모듈로 작동하여 기존의 추천 알고리즘과 통합될 수 있으며, 더 나은 장기 복지 증진을 위한 가능성을 열어줍니다. 이 결과들은 RS의 장기 복지를 향상시키기 위한 정보 노출 접근법의 성공 가능성을 입증합니다.



### Does Weighting Improve Matrix Factorization for Recommender Systems? (https://arxiv.org/abs/2510.10440)
Comments:
          In the proceedings of the Web Conference (WWW) 2025 (11 pages)

- **What's New**: 이 논문에서는 Top-N 추천 및 협업 필터링을 위한 행렬 분해(matrix factorization) 접근법을 다룬다. 특히, 암묵적 피드백 데이터에 대한 가중치(weighting) 전략에 대한 체계적인 연구를 수행하였고, 놀랍게도 비가중치 데이터로 훈련한 대형 모델이 가중치 데이터로 훈련한 모델과 비슷한 성능을 보이거나 심지어 그보다 나은 결과를 보인다는 것을 발견하였다. 이는 기존의 상식에 도전하는 결과로, 특정 조건에서만 가중치가 유익할 수 있음을 시사한다.

- **Technical Details**: 본 연구에서는 사용자-아이템 상호작용 행렬을 분해하여 잠재 패턴을 포착하는 행렬 분해 기술을 기반으로 한다. 특히 가중치 행렬(weight matrix)을 적용한 Weighted Matrix Factorization(WMF)을 통해 사용자와 아이템의 d차원 요소를 학습하며, Frobinius norm을 활용하여 성능 최적화를 시도한다. 또한, 새로운 효율적인 알고리즘을 도출하여 이전에는 계산적으로 다루기 어려운 여러 가중치 목표를 최소화할 수 있는 방법을 개발하였다.

- **Performance Highlights**: 실험 결과, 비가중치로 훈련된 대형 선형 모델이 표준 추천 시스템 벤치마크에서 가중치 기반 모델과 비교하여 유사한 성능을 나타내는 것으로 나타났다. 그러나 소규모 모델에서는 가중치가 유익할 수 있는 특정 상황이 발견되었다. 또한, 다양한 방법에 걸쳐 가중치, 정규화(regularization), 모델 용량(model capacity) 간의 상호작용을 체계적으로 연구하였다.



### ZeroGR: A Generalizable and Scalable Framework for Zero-Shot Generative Retrieva (https://arxiv.org/abs/2510.10419)
- **What's New**: 이 논문에서는 ZeroGR이라는 새로운 제로샷 생성 검색(framework) 방법을 제안하며, 이는 자연어 지시를 바탕으로 GR을 다양한 정보 검색(Information Retrieval) 작업으로 확장합니다. ZeroGR은 문서 식별자(docid) 생성을 통해 정보 검색을 재구성하여, 실세계에서 널리 사용되는 제로샷 시나리오에서도 효과적으로 작동합니다. 논문의 주요 기여는 GR을 적응시키고 다양한 IR 작업에 적용할 수 있는 새로운 넷 모듈을 도입한 것입니다.

- **Technical Details**: ZeroGR은 세 가지 주요 요소로 구성되어 있습니다. 첫째, LM 기반의 docid 생성기로 다양한 형식의 문서를 통합하여 의미 있는 docid를 생성합니다. 둘째, 자연어 작업 설명을 바탕으로 다양한 쿼리를 생성하는 지시 조정 쿼리 생성기입니다. 마지막으로, docid 생성을 위한 역 어닐링 디코딩 전략을 사용하여 정밀성과 재현율 간 균형을 맞춥니다.

- **Performance Highlights**: ZeroGR은 BEIR 및 MAIR 벤치마크에서 강력한 밀집 검색(dense retrieval)과 생성 검색(generative baselines)을 초월하며, 제로샷 설정에서 새로운 최첨단 성과를 보여줍니다. 특히, zero-shot MAIR 작업에서 OpenAI Embed-v3를 초과하는 성능을 발휘하여, 보지 못한 검색 작업에 대한 강력한 일반화 능력을 입증합니다.



### Breaking the Likelihood Trap: Consistent Generative Recommendation with Graph-structured Mod (https://arxiv.org/abs/2510.10127)
- **What's New**: 이번 연구에서는 Consistent Graph-structured Generative Recommendation (Congrats)라는 새로운 생성적 재정렬 프레임워크를 제안합니다. 이 방법은 'likelihood trap'을 극복하여 사용자 선호에 맞는 더 다양한 추천 목록을 생성합니다. Congrats는 특별히 설계된 그래프-구조화 디코더를 통해 시퀀스를 생성하면서도 높은 품질을 유지합니다.

- **Technical Details**: Congrats 프레임워크는 그래프 전이 구조를 활용하여 디코딩 공간을 확장하고, 이를 통해 다양한 경로를 제공하여 예측 정확도를 향상시킵니다. 또한, 평가자를 통합한 미분 가능한 캐스케이드 시스템을 설계하여 모델이 사용자 피드백에 따라 직접 학습하도록 하였습니다. Gumbel-Softmax 재매개변수화 기법을 사용하여, 이 과정이 미분 가능하게 만들어져 평가자의 점수를 기반으로 최적화를 진행합니다.

- **Performance Highlights**: 광범위한 오프라인 실험 결과, Congrats는 기존 최신 재정렬 방법들보다 우수한 성능을 보였습니다. 실험에서는 사용자 참여를 극적으로 개선하는 추천 품질과 다양성의 향상이 확인되었습니다. 특히, 하루 3억 명 이상의 사용자를 가진 영상 공유 애플리케이션인 Kuaishou에서 실용성을 입증했습니다.



### Integrating Structure-Aware Attention and Knowledge Graphs in Explainable Recommendation Systems (https://arxiv.org/abs/2510.10109)
- **What's New**: 이 논문은 지식 그래프(knowledge graphs)와 구조 인지(attention mechanisms)를 통합한 설명 가능한 추천 모델을 설계하고 구현했습니다. 그래프 신경망(graph neural networks)에 기반하여 다중 홉 이웃 집계 전략(multi-hop neighbor aggregation strategy)을 포함하고 있으며, 추천 시스템의 발전을 위한 새로운 접근 방식을 제공합니다.

- **Technical Details**: 모델은 사용자와 아이템을 통합된 그래프 구조로 임베딩하며, 지식 그래프에 있는 엔터티와 관계를 바탕으로 다중 수준의 의미 경로(multi-level semantic paths)를 구성합니다. 이러한 경로를 통해 풍부한 맥락 정보를 추출하고, 사용자와 타겟 아이템의 표현(interaction)을 통해 추천을 생성합니다. 최적화는 이진 교차 엔트로피 손실 함수(binary cross-entropy loss function)를 사용하여 이루어집니다.

- **Performance Highlights**: Amazon Books 데이터셋을 사용한 실험을 통해 제안된 모델이 다양한 평가 지표에서 우수한 성능을 발휘한다는 것을 인증받았습니다. 모델은 또한 좋은 수렴(convergence)과 안정성(stability)을 보여주었으며, 구조 인지(attention mechanisms)가 지식 그래프를 향상시킨 추천에서 효과적이고 실용적임을 입증합니다.



### CardRewriter: Leveraging Knowledge Cards for Long-Tail Query Rewriting on Short-Video Platforms (https://arxiv.org/abs/2510.10095)
- **What's New**: 이번 논문에서는 CardRewriter라는 새로운 프레임워크를 도입했습니다. 이 프레임워크는 도메인 특화 지식을 통합하여 짧은 동영상 플랫폼에서의 장기 쿼리 재작성(long-tail query rewriting)의 품질을 향상시키는 것을 목표로 합니다. CardRewriter는 사용자 쿼리에 관련된 다중 소스의 지식을 수집하여 정보 카드로 요약하고, 이를 제공함으로써 LLM의 사용자 의도를 더 잘 이해하도록 지원합니다.

- **Technical Details**: CardRewriter는 사용자 쿼리와 관련된 다중 소스의 지식 정보를 수집하는 데 사용됩니다. 이 정보는 쿼리와 일치하는 간결한 지식 카드로 요약되어 LLM의 쿼리 재작성 과정에 통합됩니다. 또한, 두 단계의 학습 파이프라인을 통해 최적화되며, 이는 감독 세부 조정(Supervised Fine-Tuning, SFT)과 그룹 상대 정책 최적화(Group Relative Policy Optimization, GRPO)를 포함합니다.

- **Performance Highlights**: 실험 결과, CardRewriter는 재작성 품질을 상당히 향상시켰으며, Kuaishou 플랫폼에서 사용자 수백만 명에게 긍정적인 경험을 제공했습니다. 온라인 A/B 테스트에서는 장기 조회율(Long View Rate, LVR)과 클릭률(Click-Through Rate, CTR)의 유의미한 증가가 나타났습니다. CardRewriter의 도입 이후, 사용자의 초기 쿼리 재형성 비율(Initiative Query Reformulation Rate, IQRR)이 감소하여 사용자 만족도 향상에 기여하고 있습니다.



### PairSem: LLM-Guided Pairwise Semantic Matching for Scientific Document Retrieva (https://arxiv.org/abs/2510.09897)
- **What's New**: 이번 연구에서는 Pairwise Semantic Matching (PairSem)이라는 새로운 프레임워크를 제안합니다. PairSem은 과학적 문서 검색 분야에서 발생하는 개념 간의 다면적인 상호작용을 포착하기 위해 관련된 의미를 엔터티-측면 쌍으로 표현합니다. 이 방법은 비지도 학습(unsupervised) 및 기본 검색기(retriever-agnostic) 방식을 통해 쿼리-문서 레이블이나 엔터티 주석 없이도 정밀하고 맥락 인식 기반의 매칭을 가능하게 합니다.

- **Technical Details**: PairSem은 과학적 객체(예: 화학 화합물)와 이들의 속성(예: melting point)으로 구성된 엔터티와 측면으로 이루어진 쌍을 생성하여, 복잡한 다면적 과학적 개념을 포착합니다. 이 프레임워크는 기존의 밀집 검색기(dense retriever)와 원활하게 통합될 수 있는 플러그 앤 플레이 방식으로 설계되었으며, lightweight entity and aspect predictors를 활용하여 효율적인 추론(inference)을 가능하게 합니다.

- **Performance Highlights**: 다양한 데이터 세트와 검색기를 사용한 광범위한 실험을 통해 PairSem은 검색 성능의 획기적인 개선 효과를 보여줍니다. 연구 결과, 멀티-측면 의미를 모델링함으로써 과학적 정보 검색에서 보다 정교하고 정확한 매칭이 이루어질 수 있음을 강조하고 있습니다. PairSem은 기존 문서 검색 성능을 크게 향상시키는 방법으로 자리매김할 것으로 기대됩니다.



### SemCSE-Multi: Multifaceted and Decodable Embeddings for Aspect-Specific and Interpretable Scientific Domain Mapping (https://arxiv.org/abs/2510.11599)
- **What's New**: 이 논문에서는 SemCSE-Multi라는 새로운 비지도 학습(unsupervised) 프레임워크를 제안하여 과학 초록의 다면적 임베딩(embeddings)을 생성합니다. 이 임베딩은 연구자가 필요한 특정 측면(aspect)을 명확히 하고 독립적으로 포착할 수 있도록 하여 세밀하고 조절 가능한 유사성 평가(similarity assessment)를 가능하게 합니다. 또한, 본 접근법은 과학 분야의 사용자 주도 시각화를 위한 적응적 기능을 제공하는 점도 특징입니다.

- **Technical Details**: 제안된 접근법은 각 연구 초록에 대해 аспект별 요약 문장을 생성하고 이는 임베딩 모델에 의해 의미적으로 유사한 요약이 임베딩 공간 내에서 근접하게 배치되도록 학습됩니다. 최종적으로, 이 аспект별 임베딩 기능은 단일 임베딩 모델로 통합되어 단일 전방 통과(forward pass)에서 여러 аспект 임베딩을 예측할 수 있게 됩니다. 또한, 임베딩을 자연어 설명으로 복원하는 디코딩 파이프라인을 도입하여 임베딩 공간의 해석 가능성을 크게 향상시킵니다.

- **Performance Highlights**: 이 연구는 주로 침입 생물학 분야에서 성능을 평가하였으며, 전문가의 지도를 받았습니다. 논문의 처음에 제안한 대로, 다양한 측면을 취합하여 사용자 맞춤의 시각화 및 결과 도출을 가능하게 함으로써, 사용자가 필요로 하는 특정 연구 방향에 대한 명확한 통찰을 제공합니다. 이러한 접근법은 기존 방법의 한계를 극복하며, 특히 저차원 시각화에서 비어 있는 영역의 의미 있는 텍스트 설명을 생성하는데 효과적임을 입증하였습니다.



### LLM-Specific Utility: A New Perspective for Retrieval-Augmented Generation (https://arxiv.org/abs/2510.11358)
Comments:
          13 pages, 9 figures

- **What's New**: 이 연구는 retrieval-augmented generation (RAG) 방법에서 LLM-specific utility 개념을 도입하고 체계적으로 조사하여, 기존의 인간 주석 패시지가 LLM에게 최적화되어 있지 않음을 보여줍니다. 연구진은 각 LLM의 내부 지식과 이해 능력의 차이로 인해 동일한 패시지가 각기 다른 효과를 발휘할 수 있음을 강조합니다. 이 결과는 RAG 연구에서 LLM-specific utility를 채택해야 할 필요성을 제기합니다.

- **Technical Details**: 연구는 다양한 데이터셋과 LLM에 걸쳐 대규모 실험을 통해 진행되었으며, 이를 통해 LLM-specific utility 측정의 새로운 기준을 제안하였습니다. LLM이 주어진 쿼리와 후보 패시지를 제공받았을 때 유용한 패시지를 식별하는 작업을 평가하기 위한 벤치마크 기준이 설정되었습니다. 이를 통해 LLM-specific utility는 단순히 패시지가 유용한지를 평가하는 것이 아니라, 각 LLM의 성능 향상을 고려하여 정의됩니다.

- **Performance Highlights**: 인간 주석 패시지가 LLM에 최적이 아니라는 결과는 LLM-specific gold utilitarian passages가 더 나은 성능을 낸다는 점에서 뒷받침됩니다. 연구 결과, LLM이 이미 알고 있는 쿼리에 대해 주어진 패시지를 과도하게 의존함으로써 성능이 저하되는 경향을 보였으며, 알고 있는 쿼리에서 모든 패시지를 거부하는 것이 이상적이라는 점이 강조되었습니다.



### ELMO: Efficiency via Low-precision and Peak Memory Optimization in Large Output Spaces (https://arxiv.org/abs/2510.11168)
Comments:
          Accepted to ICML 2025

- **What's New**: 이번 논문에서는 Extreme Multilabel Classification (XMC)에 대한 새로운 저정밀도 훈련 프레임워크인 ELMO를 제안합니다. ELMO는 BFloat16과 Float8 데이터 타입을 사용하여 순수 저정밀도 훈련을 통해 큰 출력 공간에서 효과적인 모델 훈련을 가능하게 합니다. 저정밀도 훈련을 통해 GPU 메모리 사용량을 획기적으로 줄일 수 있으며, 3백만 개 레이블의 모델을 6.6 GiB의 메모리로 훈련할 수 있습니다.

- **Technical Details**: ELMO는 Kahan summation과 stochastic rounding 기법을 활용하여 Float8 데이터 타입만으로 모델을 훈련할 수 있는 가능성을 보여줍니다. 이러한 접근 방식은 딥러닝에서의 메모리와 계산 요구 사항을 줄이기 위해 개발된 것으로, GF16에서 BF16으로의 전환과 그라디언트 통합 전략을 통해 이루어집니다. 우리의 방법은 모델 훈련의 메모리 요구량을 50-75%까지 줄일 수 있도록 돕습니다.

- **Performance Highlights**: 여러 개의 레이블 크기에 대해 ELMO의 저정밀도 훈련 방법을 평가한 결과, 기존의 SOTA 방법과 비슷한 성능을 나타냅니다. 또한, LF-Paper2Keywords-8.6M이라는 8.6백만 레이블을 가진 새로운 데이터셋을 소개하여, 현재 공개된 XMC 벤치마크 중 가장 큰 데이터셋임을 주장합니다. 저정밀도 훈련은 XMC 분야에서 더욱 더 중요한 기준으로 자리잡을 가능성이 높습니다.



### FBS Model-based Maintenance Record Accumulation for Failure-Cause Inference in Manufacturing Systems (https://arxiv.org/abs/2510.11003)
- **What's New**: 이 연구에서는 제조 시스템의 유지보수 기록을 기반으로 한 함수-행동-구조(Function-Behavior-Structure, FBS) 모델 기반의 진단 지식 온톨로지를 개발하였다. 이를 통해 고장 원인 추론을 보다 효과적으로 수행할 수 있는 방법론을 제안하였다. 특히, 전문가의 정성적인 데이터와의 높은 일치를 보여주며, 비전문가도 활용할 수 있는 초기 문제 해결 방안을 마련하고자 하였다.

- **Technical Details**: 연구의 핵심은 심층 지식과 피상 지식의 명확한 구조화를 요구하는 지식 기반 고장 진단의 두 가지 필요 조건을 충족하는 것이다. FBS 모델을 사용하여 AIAG & VDA의 기능적 계층을 통합하고, 이 틀 안에서 고장 사건과 원인 관계를 체계적으로 연결하여 누적하는 방법을 제안하였다. 이를 통해 관계형, 실현 관계 및 순차적 관계를 고려한 추론의 기초를 구축하였다.

- **Performance Highlights**: 제안된 방법은 축적된 유지보수 기록을 통해 고장 원인 추론의 정확도를 높였으며, 특히 관련 사례 수가 적고 용어가 상이한 어려운 경우에서도 전문가가 언급한 원인 후보 집합과의 일치를 증가시켰다. 이 연구는 고장 원인 추론 방법론의 발전을 위한 기초 자료를 제공하며, 앞으로 더 넓고 다양한 시스템에서 검증할 필요가 있다고 강조한다.



### DRIFT: Decompose, Retrieve, Illustrate, then Formalize Theorems (https://arxiv.org/abs/2510.10815)
- **What's New**: 새로운 DRIFT 프레임워크는 대형 언어 모델(LLM)이 비공식 수학 명제를 작고 더 관리하기 쉬운 '하위 구성요소'로 분해하도록 지원합니다. 이를 통해 Mathlib과 같은 수학 라이브러리에서 기본 전제들을 더 효율적으로 검색할 수 있게 됩니다. 또한, 해당 프레임워크는 예제 정리를 검색하여 모델이 전제를 더 효과적으로 사용하도록 돕는 새로운 접근법을 제공합니다.

- **Technical Details**: DRIFT는 네 단계로 구성된 프로세스를 통해 비공식 수학 진술의 복잡성을 처리하고, 검색된 형식적 객체의 증명 예시를 제공합니다. 첫 번째 단계에서는 LLM이 비공식 진술을 작은 하위 쿼리로 분해합니다. 이후 이 쿼리는 Mathlib와 같은 형식적 라이브러리에서 의존하는 전제를 검색하는 데 사용되며, 이로 인해 보다 정확한 정의 검색이 가능합니다.

- **Performance Highlights**: DRIFT는 ProofNet 및 ConNF 벤치마크에서 새로운 최첨단 성과를 달성하였고, 특히 ConNF 벤치마크에서 GPT-4.1과 DeepSeek-V3.1을 사용하여 각각 37.14% 및 42.25%의 성장을 보여주었습니다. 이러한 분석은 수학 자동 형식화의 효과가 모델의 지식 경계에 크게 의존하고 있음을 강조하며, 각 모델의 능력에 맞춘 적응형 검색 전략의 필요성을 시사합니다.



### Is Implicit Knowledge Enough for LLMs? A RAG Approach for Tree-based Structures (https://arxiv.org/abs/2510.10806)
Comments:
          Waiting for Conference Response

- **What's New**: 본 논문에서는 'Retrieval-Augmented Generation (RAG)' 방식을 통해 구조화된 데이터(예: 코드 파일)에서 생성된 응답을 향상시키기 위한 새로운 하향식(bottom-up) 방법을 제안합니다. 이 방법은 계층적 구조(예: 트리)의 지식을 선형화(linearize)하여 각 계층에서 암묵적(implicit) 요약을 생성합니다. 이 접근 방식은 기존의 RAG 방법론보다 더 효율적이며, 68% 이상의 문서 수 감소로 응답 품질을 비슷하게 유지하는 것을 보여줍니다.

- **Technical Details**: 이 논문은 계층적 구조에서 암묵적 지식을 생성하기 위한 새로운 방법을 제안합니다. 제안된 방법은 리프 노드(leaf node)에서부터 시작하여 모든 리프 노드에 대한 '템플릿(template)' 지식을 획득한 후, 각 부모 노드를 순회하며 자식들로부터 받은 암묵적 지식을 바탕으로 상위 요약을 생성합니다. 이러한 선형화 과정은 벡터 데이터베이스에 저장될 정보 조각을 최적화하고 토큰 수를 제한하여 효율성을 높입니다.

- **Performance Highlights**: 우리의 실험은 GM의 비구조화된 코드 리포지토리를 사용하였으며, 제안된 방법이 전통적인 RAG 방법에 비해 응답 품질이 유사함에도 불구하고 저장된 데이터 양을 거의 4분의 1로 줄임을 보여줍니다. 이를 통해 복잡한 구조적 정보를 처리하는 데 있어 암묵적 지식이 충분하고 효율적일 수 있음을 제안합니다. 또한 이 연구는 RAG 프레임워크에서 지식 관리를 위한 효과적이고 확장 가능한 방법 개발의 필요성을 강조합니다.



### Hierarchical LoRA MoE for Efficient CTR Model Scaling (https://arxiv.org/abs/2510.10432)
Comments:
          13 pages, 9 figures

- **What's New**: 이번 논문에서는 CTR 예측을 위한 효율적이고 확장 가능한 모델 설계를 제안합니다. 이를 위해 HiLoMoE라는 계층적 LoRA MoE 프레임워크를 도입하여, 수직적 및 수평적 확장을 모두 가능하게 합니다. 이 모델은 경량화된 rank-1 전문가를 사용하여 매개변수 효율성을 높이고, 계층적 라우팅을 통해 전문가 조합을 다양화합니다.

- **Technical Details**: HiLoMoE는 전문가 선택을 이전 레이어의 라우팅 점수를 기반으로 하여 모든 레이어를 병렬적으로 실행할 수 있게 합니다. 이 시스템은 세 가지 핵심 혁신으로 구성되며, LoRA 전문가를 통해 매개변수 감소를 이루고 계층적 라우팅 메커니즘을 통해 수직적 확장을 지원합니다. 또한 이 복잡한 시스템을 학습하기 위해 세 단계의 학습 파이프라인을 제안하고, 보조 손실을 추가하여 전문가의 다양성을 강화합니다.

- **Performance Highlights**: 네 개의 공개 데이터셋에서의 실험 결과, HiLoMoE는 비 MoE 모형에 비해 평균적으로 AUC를 0.20% 개선하고, FLOPs는 18.5% 감소하는 성능-효율성 거래를 달성했습니다. 이 모델은 깊이와 폭을 확장하는 데 있어 뛰어난 성능을 보이며, 레이어 수와 전문가 수가 증가할수록 성능이 개선되는 경향을 보입니다.



### ImCoref-CeS: An Improved Lightweight Pipeline for Coreference Resolution with LLM-based Checker-Splitter Refinemen (https://arxiv.org/abs/2510.10241)
- **What's New**: 이번 연구에서는 Coreference Resolution (CR) 문제를 해결하기 위해 새로운 프레임워크인 ImCoref-CeS를 제안합니다. 기존의 감독 학습 방법과 대규모 언어 모델(LLM) 간의 강점을 통합하여 성능을 극대화하는 것을 목표로 합니다. 특히, 경량 브리징 모듈과 하이브리드 멘션 정규화를 통해 긴 텍스트의 인코딩 능력을 한층 더 향상시켰습니다.

- **Technical Details**: ImCoref-CeS 프레임워크는 기존의 detect-then-cluster 파이프라인을 확장하여, 경량 브리징 모듈(LBM)과 바이아핀 점수기(biaffine scorer)를 도입합니다. 이는 멘션 탐지 단계에서 LLM을 검증기(Checker)와 클러스터링 분할기(Splitter) 역할로 활용하여, 불완전한 멘션을 걸러내고 오류가 있는 클러스터를 수정하는 작업을 수행합니다. 이러한 접근 방식을 통해 CR의 정확도를 높이고, 컴퓨팅 자원을 효율적으로 사용합니다.

- **Performance Highlights**: 다양한 CR 벤치마크에서 수행된 실험 결과, ImCoref는 기존의 최첨단(supervised neural methods) 방법들보다 지속적으로 성능이 향상되었습니다. ImCoref-CeS 프레임워크는 이 성과를 뛰어넘어 보다 높은 수준의 성능을 가능하게 하여, 기존 방법의 효율성과 정확도를 새로운 차원으로 끌어올립니다.



### Text2Token: Unsupervised Text Representation Learning with Token Target Prediction (https://arxiv.org/abs/2510.10224)
- **What's New**: 본 논문에서는 Unsupervised Text Representation Learning (TRL) 분야에서 새로운 프레임워크인 Text2Token을 제안합니다. 이 프레임워크는 텍스트의 key token을 생성하는 작업을 통해 고품질의 텍스트 표현을 학습하며, 기존의 discriminative (구별적) 학습 방식과는 다른 접근 방식을 취합니다. 이 연구는 LLM(대형 언어 모델)을 활용하여 키 토큰의 통계적 특성을 분석하고, 이러한 통찰을 기반으로 텍스트 representation을 향상시키려 합니다.

- **Technical Details**: Text2Token은 Kullback-Leibler divergence (KL-divergence) 손실함수를 통해 목표 토큰 분포(target token distribution)를 생성하는 비지도 학습 프레임워크입니다. 주요 두 가지 토큰 범주, 즉 의미 있는 텍스트의 토큰과 텍스트를 넘어선 의미론적 유도 토큰을 활용하여 데이터를 기반으로 하거나(Language Models Prior를 이용하여) 토큰 대상을 생성합니다. 실험은 MTEB v2 벤치마크에서 수행되었고, Text2Token은 기존의 최고 성능을 가진 LLM2Vec보다 우수한 결과를 보였습니다.

- **Performance Highlights**: 텍스트 대표화 (representation)와 어휘(vocabulary) 공간이 훈련 과정에서 함께 최적화된다는 발견이 있었습니다. Text2Token은 여러 과제에서 LLM2Vec보다 평균 점수를 크게 초과하는 성과를 가져오며, 높은 성능을 입증하였습니다. 이 논문은 각 구성 요소의 역할과 하이퍼파라미터의 영향을 이해하기 위한 분석 실험을 수행하였습니다.



### Beyond the limitation of a single query: Train your LLM for query expansion with Reinforcement Learning (https://arxiv.org/abs/2510.10009)
- **What's New**: 이 논문에서는 ExpandSearch라는 새로운 강화 학습( reinforcement learning) 기반의 검색 프레임워크를 제안합니다. 이 프레임워크는 검색 에이전트가 쿼리를 확장하고, 선택적으로 정보를 정제하여 복잡한 질문에서 정확한 답변을 생성할 수 있도록 돕습니다. ExpandSearch는 특히 다단계 추론(multi-hop reasoning) 작업에서 높은 성능을 보여줍니다.

- **Technical Details**: ExpandSearch는 기존의 검색 에이전트가 여러 쿼리 변형을 생성하여 정보를 검색하도록 훈련합니다. 이 과정에서 쿼리 생성과 정보 정제 단계를 명확히 구분함으로써 의미의 불완전성과 정보 과부하 문제를 해결하고자 합니다. 또한, 두 가지 유형의 쿼리 확장, 즉 구문 확장(syntax expansion)과 의미 확장(semantic expansion)을 도입하여 다양한 관점을 포착하려고 합니다.

- **Performance Highlights**: 실험 결과 ExpandSearch는 최신 기술 기준에 비해 평균 4.4%의 성능 향상을 이루었으며, 특히 다양한 증거 집합을 요구하는 복잡한 다단계 추론 과제에서 강한 성과를 보였습니다. 이는 3B LLM 규모의 모델에서도 쿼리 확장 능력을 크게 개선할 수 있음을 보여줍니다.



### Stop DDoS Attacking the Research Community with AI-Generated Survey Papers (https://arxiv.org/abs/2510.09686)
Comments:
          Accepted by NeurIPS 2025 (Position Track)

- **What's New**: 이 논문에서는 최근의 AI 생성 설문지 급증 현상에 대해 논의하고 있습니다. 전통적으로 노동 집약적이었던 설문지 작성이 대형 언어 모델(LLMs) 덕분에 빠르고 쉽게 이루어지는 반면, 이는 연구 커뮤니티에 'survey paper DDoS attack'이라는 새로운 위협을 초래하고 있습니다. 이 공격은 피상적으로 종합적인 연구 결과물들이 홍수처럼 쏟아져 나와 연구자들을 압도하고 신뢰를 훼손하는 것을 뜻합니다.

- **Technical Details**: 연구팀은 2020년부터 2024년까지 arXiv에 제출된 논문을 분석하여 설문지의 수가 급격히 증가하고 있음을 확인했습니다. 이들은 'survey', 'review', 'overview' 및 'taxonomy'라는 키워드가 포함된 제목을 가진 논문들을 수집하여 AI 생성 점수를 측정했으며, AI의 도움을 받은 설문지들이 특히 증가하고 있음을 발견했습니다. 이러한 AI 생성 설문지는 종종 비판적인 분석이나 비교 없이 단순한 집계로 축소되는 위험이 있습니다.

- **Performance Highlights**: AI 생성 설문이 연구 품질에 미치는 부정적인 영향을 우려하여, 논문은 AI 사용에 대한 엄격한 기준과 전문가의 감독 필요성을 주장합니다. 현재 arXiv에서의 AI 생성 설문지는 연구자들에게 큰 혼란을 초래하고 있으며, 이는 신뢰도 저하에 기여하고 있습니다. 연구가 질적으로 향상되기 위해서는 AI 지원 리뷰 작성 시 인간의 숙련된 감독과 투명성이 필수적입니다.



New uploads on arXiv(cs.CV)

### DeepMMSearch-R1: Empowering Multimodal LLMs in Multimodal Web Search (https://arxiv.org/abs/2510.12801)
- **What's New**: 이번 연구에서는 DeepMMSearch-R1이라는 새로운 다중 모달 LLM을 소개합니다. 이 모델은 웹에서의 다단계 검색을 수행할 수 있는 최초의 모델로, 이미지 및 텍스트 검색 도구를 위해 동적으로 쿼리를 생성합니다. 특히 이 모델은 입력 이미지의 관련 부분을 기반으로 웹 검색을 시작하여 이미지 검색의 효율성을 높이고, 수집된 정보를 바탕으로 텍스트 검색 쿼리를 점진적으로 조정할 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: DeepMMSearch-R1은 두 단계의 훈련 파이프라인을 따릅니다. 첫 번째 단계는 감독된 미세 조정(Supervised Finetuning, SFT) 단계이며, 그 다음 온라인 강화 학습(Online Reinforcement Learning, RL) 최적화를 수행합니다. 연구팀은 DeepMMSearchVQA라는 새로운 다중 모달 VQA 데이터셋을 소개했으며, 이 데이터셋은 웹 검색 도구에서 수집된 실제 정보를 포함하여 적합한 질문과 이미지를 훈련할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험을 통해 DeepMMSearch-R1은 최신 상태의 성과를 기록하며, 이전의 오픈 소스 기준을 초월하는 성능을 달성했습니다. 두 단계의 훈련 과정과 자가 반성, 자가 수정, 그리고 크롭된 이미지 검색의 영향을 통해 성능에 긍정적인 영향을 미쳤음을 입증했습니다. 이로 인해 다중 모달 웹 검색 도구의 통합을 촉진하는 유용한 자료를 제공할 수 있었습니다.



### Detect Anything via Next Point Prediction (https://arxiv.org/abs/2510.12798)
Comments:
          homepage: this https URL

- **What's New**: 이번 연구에서는 Rex-Omni라는 3B 스케일의 MLLM(multi-modal large language model)을 제안하여 최신 오브젝트 인식(object detection) 성능을 세계적 수준으로 달성하였습니다. 이 모델은 전통적인 회귀 기반 모델인 YOLO, DINO, Grounding DINO와 비교하여 무훈련(zero-shot) 환경에서도 경쟁력 있는 성능을 보여줍니다. Rex-Omni는 세 가지 주요 디자인 원칙인 작업 формулирование(task formulation), 데이터 엔진(data engines), 및 훈련 파이프라인(training pipelines)을 통해 이를 실현하였습니다.

- **Technical Details**: Rex-Omni는 각각의 시각적 인식 작업을 좌표 예측 프레임워크로 통합합니다. 특정 작업에 대해 0부터 999의 양자화된 좌표 값에 대해 특별한 토큰을 사용하여 모델의 학습 난이도를 줄이고 좌표 예측의 효율성을 높였습니다. 이를 위해 고품질 기준치(grounding), 지칭(referring), 및 포인팅(pointing) 데이터를 생성하는 여러 데이터 엔진을 설계하였으며, 두 단계의 훈련 프로세스를 통해서 모델 성능을 극대화했습니다.

- **Performance Highlights**: Rex-Omni는 COCO 및 LVIS와 같은 벤치마크에서 전통적인 회귀 기반 모델과 비교하여 더 높은 F1 점수를 기록하며 우수한 성능을 입증했습니다. 긴 꼬리(long-tailed) 탐지, 지칭 표현 이해(referring expression comprehension), 촘촘한 오브젝트 탐지(dense object detection) 등 다양한 작업에서도 일관되게 높은 성능을 발휘하여, 언어 이해와 정확한 위치 인식을 결합한 통합된 프레임워크를 확립하였습니다.



### DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving (https://arxiv.org/abs/2510.12796)
- **What's New**: DriveVLA-W0라는 새로운 훈련 패러다임을 제안하여, 드라이브 환경의 동작을 예측하기 위해 세계 모델링을 활용합니다. 이 접근법은 희소한 주의 신호를 보완하고, 모델이 외부 환경의 역학을 적절히 학습하도록 강요함으로써, 이미지 예측을 통한 밀집 자가 감독 신호를 생성합니다. 이로 인해 VLA 모델의 성능이 크게 향상되어 다양한 운전 데이터 규모에서 그 효과를 입증하였습니다.

- **Technical Details**: DriveVLA-W0는 두 가지 주요 VLA 아키텍처에 대해 구현되었으며, 이 중 하나는 이산 비주얼 토큰을 사용하는 자가 회귀 세계 모델이고, 다른 하나는 연속 비주얼 기능을 사용하는 확산 세계 모델입니다. 이러한 세계 모델링 기법은 모델이 이미지 예측할 때마다 밀집한 감독 신호를 생성하고, 이를 통해 풍부한 세계 표현을 학습하게 합니다. 실시간 배포를 위해 경량화된 MoE 기반의 액션 전문가를 도입하여 추론 지연 시간을 기존 VLA 모델의 63.1%로 줄였습니다.

- **Performance Highlights**: 대규모 70M 프레임 데이터셋을 활용한 실험에서 세계 모델링이 데이터 스케일링 법칙을 강화하는데 기여함을 확인했습니다. 또한, 일관된 일반화를 통해 다양한 도메인 간의 전이 가능한 비주얼 표현을 학습하는 데 도움이 됩니다. 이 연구는 액션 디코더를 위한 흥미로운 성능 트렌드 반전을 발견하는데, 복잡한 플로우 매칭 디코더가 작은 데이터셋에서는 이점을 가지나, 대규모에서는 더 간단한 자가 회귀 모델이 최고의 성능을 발휘하는 것으로 나타났습니다.



### CuMPerLay: Learning Cubical Multiparameter Persistence Vectorizations (https://arxiv.org/abs/2510.12795)
Comments:
          Appears at ICCV 2025

- **What's New**: CuMPerLay는 Cubical Multiparameter Persistence (CMP)를 딥러닝 파이프라인에 통합할 수 있는 새로운 차별화된 벡터화 레이어입니다. 기존 CMP는 이미지의 위상적으로 작업하는 자연스러운 방법을 제공하지만, 다중 필터 구조의 복잡성과 CMP의 벡터화로 인해 사용에 제한이 있었습니다. CuMPerLay는 이러한 문제를 해결하기 위해 신속한 벡터화 알고리즘을 제시합니다.

- **Technical Details**: CuMPerLay는 CMP를 개별적이고 학습 가능한 단일 파라미터 지속성의 조합으로 분해합니다. 이때, 바이필터 함수(bifiltration functions)는 함께 학습되며, 이는 딥러닝 모델에서의 차별화(differentiability)를 통해 강력한 위상적 특징 벡터를 제공할 수 있게 합니다. 또한, 일반화된 Wasserstein 메트릭스에서 벡터화의 안정성에 대한 이론적 보증을 제시합니다.

- **Performance Highlights**: 의료 이미지 분석과 컴퓨터 비전 데이터셋에 대한 실험 결과, CuMPerLay는 제한된 데이터 시나리오에서 특히 분류(classification)와 분할(segmentation) 성능에서 이점이 있음을 보여줍니다. 따라서 CuMPerLay는 구조화된 이미지 분석을 위한 딥 네트워크에 글로벌 구조 정보를 통합하는 유망한 방향을 제공합니다.



### ViCO: A Training Strategy towards Semantic Aware Dynamic High-Resolution (https://arxiv.org/abs/2510.12793)
- **What's New**: 이번 연구에서는 Visual Consistency Learning (ViCO)라는 새로운 훈련 알고리즘을 제안하여 모델이 이미지의 다양한 의미 복잡성을 표현할 수 있도록 합니다. 이 방법은 각 이미지 패치에 대해 적절한 비주얼 토큰 수를 동적으로 선택하는 Visual Resolution Router (ViR)를 통해 잠재적인 정보 손실을 최소화합니다. 실험 결과, 제안하는 방법은 비주얼 토큰 수를 최대 50%까지 줄이면서도 모델의 인식, 추론 및 OCR 능력을 유지할 수 있음을 보여줍니다.

- **Technical Details**: ViCO의 훈련 절차는 두 단계로 구성됩니다: (1) 일관성 훈련(Consistency Training)에서 모델은 서로 다른 비주얼 토큰 압축 비율에 따라 응답 분포의 KL 다이버전스(KL divergence)를 최소화하도록 훈련됩니다. (2) 라우터 훈련(Router Training)에서는 비주얼 레졸루션 라우터(ViR)가 각 이미지 패치에 대한 적절한 압축 비율을 자동으로 선택하여 기계의 성능을 최적화합니다. 이는 복잡한 의미 정보를 포함하는 패치는 더 많은 비주얼 토큰으로 표현하게 하고, 단순한 패치는 적은 수의 토큰으로 표현하도록 합니다.

- **Performance Highlights**: 제안하는 ViCO 방법은 다양한 벤치마크에서의 실험을 통해 비주얼 토큰 수를 절반으로 줄이면서도 강력한 성능을 유지하는 것을 보여주었습니다. 특히, InternVL3.5 모델의 첫 번째 토큰 처리량이 개선되면서 효율성도 크게 증가했습니다. 이러한 성과는 인공지능 검색 및 문서 이해와 같은 다양한 멀티모달 인식 및 추론 과제를 사용할 때 더욱 주목받고 있습니다.



### UniFusion: Vision-Language Model as Unified Encoder in Image Generation (https://arxiv.org/abs/2510.12789)
Comments:
          Project page at this https URL

- **What's New**: 본 논문에서는 UniFusion이라 명명된 새로운 확산 기반 생성 모델을 소개합니다. 이 모델은 고정된 대형 비전-언어 모델(VLM)을 통합 멀티모달 인코더로 활용하여 이미지와 텍스트의 별도를 없앴습니다. 특히, Layerwise Attention Pooling (LAP) 기법을 통해 고수준의 의미 및 저수준의 세부 정보를 효과적으로 추출하여, 이미지 생성 및 편집 작업에서 우수한 성능을 발휘합니다.

- **Technical Details**: UniFusion의 핵심 요소인 LAP는 여러 VLM 레이어에서 정보를 수집하며, 이는 고유한 정밀도 및 높은 수준의 의미 추출을 가능하게 합니다. 또한 VLM이 생성한 텍스트 토큰에만 기반한 Diffusion Transformer (DiT)의 조건부 생성을 위한 VLM-Enabled Rewriting Injection with Flexible Inference (VERIFI) 기법을 제안하여, 다양한 프롬프트 형식 간의 분포 이동을 줄여줍니다. 이러한 설계로 인해 UniFusion은 복수의 이미지를 참조하더라도 높은 일반화 능력을 지니게 되었습니다.

- **Performance Highlights**: UniFusion은 단일 이미지 편집 작업에서 훈련된 모델이 다중 참조 이미지에도 제로샷으로 일반화되는 뛰어난 성능을 보여줍니다. 본 연구는 텍스트-이미지 생성 및 편집 작업에서 경쟁력 있는 성능을 발휘하며, 특별한 감독 학습이나 강화 학습 없이도 가능함을 입증합니다. UniFusion의 경우, 이미지 편집 작업에서 학습할 때 텍스트-이미지 프롬프트 준수 및 미적 품질이 크게 향상되는 긍정적인 크로스 태스크 이왕 전이 현상도 보여줍니다.



### Efficient Real-World Deblurring using Single Images: AIM 2025 Challenge Repor (https://arxiv.org/abs/2510.12788)
Comments:
          ICCV 2025 - AIM Workshop

- **What's New**: 이 논문은 AIM 2025의 Efficient Real-World Deblurring using Single Images Challenge에 대해 다룹니다. 이 챌린지는 실시간 혼합 복원(Real-Blur Restoration)을 개선하기 위해 설계되었습니다. 새로운 테스트 세트는 RSBlur 데이터셋을 기반으로 하며, 71명이 등록하여 4개 팀이 유효한 솔루션을 제출했습니다. 가장 높은 성과를 낸 방법은 31.1298 dB의 PSNR을 달성하여 효율적인 이미지 복원의 가능성을 보여줍니다.

- **Technical Details**: 챌린지의 목표는 단일 이미지 디블러링을 위한 효율적인 알고리즘 설계입니다. 참가자들에게는 시작 키트로 간단한 베이스라인 코드와 모델이 제공됩니다. RSBlur 데이터셋은 듀얼 카메라 시스템을 이용해 수집된 실제 흐림 이미지와 그에 상응하는 선명한 이미지 쌍으로 구성되어 있습니다. 이 데이터셋에는 훈련, 검증 및 테스트 세트로 각각 8,887, 1,120, 3,360 쌍의 이미지가 포함되어 있습니다.

- **Performance Highlights**: 제안된 방법의 성능은 PSNR, SSIM, LPIPS의 조합을 통해 평가됩니다. 모든 제안된 방법은 5M의 파라미터와 200 GMACs의 이론적인 계산 제약을 만족해야 합니다. 참가자들은 3일 동안 총 420개의 흐림 이미지를 제출하여 최종 성과를 평가받습니다. AIM 2025 챌린지는 다양한 고성능 비전 태스크와 연관되어 있으며, 효율적인 모바일 디바이스에서의 디블러링의 중요성을 강조합니다.



### MVP4D: Multi-View Portrait Video Diffusion for Animatable 4D Avatars (https://arxiv.org/abs/2510.12785)
Comments:
          18 pages, 12 figures

- **What's New**: MVP4D 모델은 단일 참조 이미지와 목표 표현에 기반하여 디지털 인간의 애니메이션 가능한 다중 시뷰 비디오를 생성합니다. 기존 방법론과 비교할 때, 이 모델은 360도 시점 변화를 통해 수백 개의 프레임을 동시에 생성합니다. 이 과정에서 현실감, 시간적 일관성, 그리고 3D 일관성을 크게 향상시킵니다.

- **Technical Details**: MVP4D는 최첨단 비디오 확산 모델(video diffusion model)을 기반으로 하여, 모핑 가능한 다중 시뷰 비디오 확산 모델(General Multi-View Video Diffusion Model)로 설계되었습니다. 이 모델은 참조 이미지로부터 입력을 받으며, 학생 측 자세와 표현, 그리고 카메라 파라미터에 따른 조건을 설정합니다. 이를 통해 3D-영상 일관성을 유지하는 다중 시점 비디오를 생성하며.

- **Performance Highlights**: MVP4D는 단일 확산 샘플링 실행에서 최대 400개의 프레임을 생성할 수 있으며, 이는 기존의 방법들보다 한층 더 개선된 성능을 제공합니다. 이 접근 방식은 높은 해상도와 리얼리즘을 유지하며, 실시간 렌더링이 가능하여 다양한 애플리케이션에 활용될 수 있습니다.



### SRUM: Fine-Grained Self-Rewarding for Unified Multimodal Models (https://arxiv.org/abs/2510.12784)
Comments:
          20 pages, 8 figures, webpage can be seen in this https URL

- **What's New**: 최근 통합 다중 모달 모델(Unified Multimodal Models, UMMs)에서 비전-언어 생성 및 이해 능력을 통합한 놀라운 진전을 이루었습니다. 하지만 시각적 이해가 우수하더라도, 이를 비주얼 생성으로 이전하는 데 실패하는 경우가 많습니다. 이 문제를 해결하기 위해 SRUM(Self-Rewarding for Unified Multimodal Models)이라는 자기 보상(post-training) 프레임워크를 소개합니다.

- **Technical Details**: SRUM은 모델의 이해 모듈이 내부 평가자로 작용하여 생성 모듈을 향상시키는 피드백 루프를 생성합니다. 이 시스템은 전체 시각적 의미 및 레이아웃의 올바름을 보장하는 글로벌 보상(global reward)과 객체 수준의 충실성을 개선하는 로컬 보상(local reward)의 이중 보상 체계를 통해 다중 규모의 지침을 제공합니다.

- **Performance Highlights**: SRUM는 T2I-CompBench에서 82.18점에서 88.37점으로, T2I-ReasonBench에서는 43.82점에서 46.75점으로 성능이 크게 향상되었습니다. SRUM은 UMM의 이해 모듈이 자신의 생성 모듈을 유도하고 강화하는 새로운 패러다임을 확립합니다.



### What If : Understanding Motion Through Sparse Interactions (https://arxiv.org/abs/2510.12777)
Comments:
          Project page and code: this https URL

- **What's New**: Flow Poke Transformer (FPT)는 지역적 상호작용에 기반하여 장면의 운동 분포를 직접 예측하는 새로운 프레임워크입니다. 기존의 밀집 샘플링 방식과는 달리, FPT는 다중 모드(scene motion)의 해석 가능하고 직접 접근 가능한 표현을 제공합니다. 또한, 다양한 다운스트림 작업에 대해 모델을 평가하여 기존의 방법들과 비교하며 유연성을 강조합니다.

- **Technical Details**: FPT는 'pokes'로 알려진 희소한 상호작용에 따라 지역적 운동 분포를 예측합니다. 이 방법은 특정 결과에 전념하는 대신, 운동의 잠재적인 다양성을 캡처하는 더 높은 추상화 수준에서 작동합니다. 모델은 비정상적인 방식으로 블록 더미와 같은 불안정한 장면의 다양한 운동 변화를 예측할 수 있습니다.

- **Performance Highlights**: 모델의 성능은 밀집 얼굴 동작 생성에서 전문적인 기준치를 초과했습니다. 인공 데이터셋과 같은 강한 분포 밖의 작업에서 파인튜닝을 통해 도메인 내 방법들에 비해 상당한 개선을 보여주었습니다. 또한, pokes에서 이동 부분 분할과 같은 작업에서 경쟁력 있는 성과를 달성함으로써 FPT의 다양성을 입증하였습니다.



### Uncertainty Matters in Dynamic Gaussian Splatting for Monocular 4D Reconstruction (https://arxiv.org/abs/2510.12768)
Comments:
          Project page: this https URL

- **What's New**: 이 연구는 동적인 3D 장면을 단안 입력(monocular input)으로 재구성하는 과정에서 발생하는 모호성을 해결하기 위해 신뢰성을 고려한 동적 Gaussian Splatting(USplat4D) 프레임워크를 제안합니다. 이는 관찰이 반복되는 Gaussian을 신뢰할 수 있는 기준점으로 활용하여 더 안정적인 움직임 추적과 향상된 4D 재구성을 가능하게 합니다.

- **Technical Details**: USplat4D에서는 각 Gaussian의 시간에 따른 불확실성을 추정하고, 이를 바탕으로 스페이셜-템포럴 그래프(spatio-temporal graph)를 구성합니다. 이 그래프는 노드의 중요성, 엣지 구성, 가중치 조정에 불확실성을 반영하여, 안정적인 장면 재구성을 도모합니다. 또한, 이 방법론은 기존의 동적 Gaussian Splatting 파이프라인에 통합될 수 있도록 설계되었습니다.

- **Performance Highlights**: 다양한 실제 및 합성 데이터셋을 통해 USplat4D의 성능을 검증한 결과, 불확실성을 명시적으로 모델링함으로써 동적 Gaussian Splatting 모델의 안정성이 향상되고, occlusion(가림) 상황에서도 높은 품질의 새로운 시점 시합(synthesis)을 생성할 수 있음을 보여주었습니다. 특히 극단적인 시점에서의 성능 향상이 두드러지며, 향후 다양한 어플리케이션에서 활용될 가능성이 큽니다.



### Efficient Perceptual Image Super Resolution: AIM 2025 Study and Benchmark (https://arxiv.org/abs/2510.12765)
Comments:
          ICCV 2025 - AIM Workshop

- **What's New**: 이 논문은 Efficient Perceptual Super-Resolution (EPSR)에 관한 포괄적인 연구와 기준치를 제시합니다._PSNR_ 중심의 초해상도 기술에서 많은 진전이 있었지만, 인지 품질 지표에 초점을 맞춘 접근은 상대적으로 비효율적입니다. 이에 따라, 우리는_maximum 5M parameters와 2000 GFLOPs의 효율성 제약을 준수하면서 Real-ESRGAN의 인지 결과를 재현하거나 개선하려고 합니다.

- **Technical Details**: 단일 이미지 초해상도(SR)는 저해상도 입력에서 고해상도 이미지를 재구성하는 방법으로, 본질적으로 역문제가 복잡합니다. 기존의 여러 기술들은 복잡한 실제 왜곡에 대한 성능이 뛰어나지 않으며, 지각 손실을 포함한 접근 방식이 생성된 이미지의 자연성을 크게 향상시키는 것으로 나타났지만, 전통적 지표에서의 손실은 불가피했습니다. 현재 사용되는 최신 방법론은 GANs 및_diffusion-based models_에 의존하고 있으며, 특히 SR3와 LDMs는 효율성을 높이는 데 기여하고 있습니다.

- **Performance Highlights**: 우리가 제안한 최상위 접근법은 모든 기준 데이터셋에서 Real-ESRGAN보다 우수한 성능을 보였습니다. 또한 이 연구는 시각적 품질과 효율성 간의 격차를 줄이기 위한 놀라운 기회를 제시하며, 다양한 접근 방식을 공정하게 비교할 수 있는 기준을 제공합니다. 최종적으로, 이 연구는 실시간 인지 초해상도 솔루션 개발을 위한 협력과 지식 공유를 촉진하여, 진전을 가속화하는 데 기여하고 있습니다.



### AnyUp: Universal Feature Upsampling (https://arxiv.org/abs/2510.12764)
Comments:
          Project Website: this https URL

- **What's New**: AnyUp는 특성 업샘플링(method for feature upsampling)을 위한 새로운 방법으로, 어떤 해상도(resolution)에서든 특정 인코더에 대한 훈련 없이 적용할 수 있습니다. 기존의 DINO или CLIP와 같은 학습 기반 업샘플러는 각 특성 추출기에 맞춰 재훈련이 필요하여 다르 특성 유형에 대한 일반화가 어렵습니다. 본 연구에서는 이러한 한계를 해결하고 업샘플링 품질을 개선하기 위해 추론(inference) 시간에 특성 비의존적(feature-agnostic) 아키텍처를 제안합니다.

- **Technical Details**: AnyUp는 저해상도 특성 맵(feature map)과 고해상도 RGB 가이드 이미지에서 정보를 추출하여 고해상도 이미지에서 특정 픽셀이 어떤 특성을 받아야 하는지를 추론합니다. 기존 방법의 주요 한계는 저해상도 특성 맵을 처리하는 방식이 사용된 특성의 차원(dimensionality) 및 유형에 의존적이라는 것입니다. AnyUp는 어떠한 특성도 처리할 수 있도록 하는 특성 비의존적 계층(feature-agnostic layer)을 도입하고, 추가적으로 윈도우 어텐션(window attention) 절차 및 크롭 기반 훈련 전략을 도입하여 업샘플링 품질을 향상시킵니다.

- **Performance Highlights**: AnyUp는 다양한 하위 작업에서 기존의 특성 업샘플링 방법들보다 우수한 성능을 보여주며, 훈련된 적이 없는 특성 유형에 대해서도 강력한 일반화를 보장합니다. 이 모델은 모든 특성 유형과 해상도에서 훈련 및 적용이 가능하여 보편적인 응용 가능성을 보유하고 있으며, 원본 특성 의미의 왜곡을 최소화하는 높은 충실도를 보장합니다. 또한, 제안된 방법은 경량 적용이 가능하고, 훈련이 필요 없는 특성 업샘플러로 쉽게 사용할 수 있습니다.



### PET Head Motion Estimation Using Supervised Deep Learning with Attention (https://arxiv.org/abs/2510.12758)
Comments:
          Accepted for publication in IEEE Transactions on Medical Imaging (TMI), 2025. This is the accepted manuscript version

- **What's New**: 이번 연구에서는 cross-attention 메커니즘을 활용한 새로운 딥러닝 기반 두부 운동 보정 접근법(DL-HMC++)를 제안합니다. 이 방법은 3D PET 원시 데이터로부터 강체 두부 운동을 예측하여, 기존의 hardware-based motion tracking (HMT) 접근 방식의 한계를 극복하고자 합니다. DL-HMC++는 기존 동적 PET 스캔과 외부 HMT의 금표준 운동 측정을 통해 감독 학습 방식으로 훈련되었습니다.

- **Technical Details**: DL-HMC++는 1초 단위의 3D PET 클라우드 이미지(PCI)를 입력으로 사용하여 강체 운동을 추정합니다. 모델 훈련과 평가에는 각 1초 PCI가 Vicra HMT 정보(강체 변환 행렬)와 함께 제공됩니다. 이 연구의 목적은 Iref와 Imov 두 PCI 간의 강체 운동 변환 θ를 추정하는 것입니다.

- **Performance Highlights**: DL-HMC++는 HRRT 및 mCT 두 개의 PET 스캐너 및 네 가지 방사선 추적제에 대해 평가되었습니다. 정량적 및 정성적 결과는 DL-HMC++가 최첨단 데이터 기반 운동 추정 방법을 일관되게 초월하여, 두부 운동이 보정된 이미지를 생성하며, 이는 HMT에 비해 차별화되지 않는 명확한 뇌 구조를 나타냅니다.



### E-MoFlow: Learning Egomotion and Optical Flow from Event Data via Implicit Regularization (https://arxiv.org/abs/2510.12753)
Comments:
          The Thirty-Ninth Annual Conference on Neural Information Processing Systems(NeurIPS 2025)

- **What's New**: 본 연구는 광학 흐름(optical flow) 및 6-자유도(6-DoF) 자아 운동(ego-motion)을 통합적인 방법으로 동시에 추정하는 자극적인 접근 방식을 제안합니다. 이는 기존의 방법들이 이 두 문제를 별도로 해결하려 했던 것과 달리, 데이터의 신뢰성을 확보할 수 있는 새로운 프레임워크입니다. 특히, 제안된 E-MoFlow는 수치적인 방법 없이도 정확하고 효율적으로 이들 두 가지 문제를 해결합니다.

- **Technical Details**: E-MoFlow는 카메라의 자아 운동을 연속 스플라인으로 모델링하고, 광학 흐름을 암시적 신경 표현(implicit neural representation)으로 처리합니다. 이를 통해, 공간-시간 연속성(spatial-temporal coherence)을 자연스럽게 통합하고, 기하학적 정합성을 유지하면서도 깊이 추정(dedth estimation)을 요구하지 않습니다. 이러한 접근 방식은 명시적 규제 없이 안정성을 보장하는 혁신적인 기술적 토대를 제공합니다.

- **Performance Highlights**: 실험 결과 E-MoFlow는 다양한 6-자유도 운동 시나리오에서 탁월한 성능을 보여주었으며, 기존의 비지도 방법들에 비해 우수한 결과를 달성하였습니다. 또한, 감독 방식(supevised) 접근 방식과 비교해도 경쟁력 있는 성능을 나타내어, 다양한 실세계 환경에서의 적용 가능성을 입증하였습니다.



### VQArt-Bench: A semantically rich VQA Benchmark for Art and Cultural Heritag (https://arxiv.org/abs/2510.12750)
- **What's New**: 이번 연구에서는 VQA(Visual Question Answering) 벤치마크의 한계를 지적하며, 복잡한 예술 분야와 문화 유산 도메인을 위한 VQArt-Bench라는 새로운 대규모 VQA 벤치마크를 제안합니다. 이 벤치마크는 시각적 이해의 다양한 차원을 탐구할 수 있도록 설계된 질문을 생성하는 다중 에이전트 파이프라인을 통해 구성되었습니다. 기존의 규칙 기반 접근법의 한계를 극복하고, 예술 작품에 대한 보다 깊이 있는 질문 생성을 목표로 합니다.

- **Technical Details**: 제안된 VQArt-Bench의 핵심은 이미지 캡션을 분석하여 질문 카테고리를 알아내고, 이러한 카테고리를 바탕으로 복잡하고 개방적인 질문을 생성하는 Topic Selector, Question Generator, Question Refiner, Judge의 4가지 에이전트로 구성된 프로세스입니다. 이 과정에서 각 질문은 확인된 순서로 비슷한 오답도 포함하여 도전적인 선택형 형식으로 변환됩니다. 모든 질문은 Judge에 의해 검토되어 비트리비얼하고, 정확하게 해답될 수 있으며, 언어적으로도 올바른지를 확인합니다.

- **Performance Highlights**: 14개의 최첨단 MLLM 모델을 새롭게 제안된 VQArt-Bench로 평가한 결과, 현재 모델들이 간단한 카운팅 작업에서도 놀라운 약점을 보였으며, 독점 모델과 오픈 소스 모델 간의 명확한 성능 차이가 발견되었습니다. 이러한 결과는 VQA 모델들이 단순히 통계적 패턴에 의존하기보다 진정한 시각적 분석을 수행하는 데 필요한 시각적 추론 능력을 평가하는 것이 얼마나 중요한지를 강조합니다.



### SPORTS: Simultaneous Panoptic Odometry, Rendering, Tracking and Segmentation for Urban Scenes Understanding (https://arxiv.org/abs/2510.12749)
Comments:
          Accepted by IEEE Transactions on Multimedia

- **What's New**: 최근 발표된 SPORTS 프레임워크는 비디오 팬옵틱 분할(Video Panoptic Segmentation, VPS), 비주얼 오도메트리(Visual Odometry, VO), 그리고 장면 렌더링(Scene Rendering, SR) 작업의 통합을 통해 도시 장면의 포괄적인 이해를 가능하게 합니다. 이는 각기 다른 작업들을 통합하여 실제 환경에서의 성능을 개선할 수 있는 혁신적인 접근 방식으로,기존의 단점들을 극복하는 것을 목표로 하고 있습니다. 또한 이 논문은 군중사고 방지를 위한 안전장치로도 활용될 수 있는 디지털 트윈(Digital Twin) 생성에 대한 가능성을 제시합니다.

- **Technical Details**: SPORTS 프레임워크는 각기 다른 영상 프레임 간의 특징을 정렬하기 위해 적응형 주의 기반의 기하학적 융합 메커니즘을 사용합니다. 이를 통해 객체 인식 및 동적 물체의 인식 정확도를 향상시킵니다. 광학 흐름(Optical Flow), 깊이(Depth), 그리고 자세(Pose) 정보를 활용하여 여러 해상도의 특징 맵을 융합하며, 포스트 매칭 전략(Post-matching Strategy)도 적용합니다. 이러한 총체적 작업은 실제 업무에 적용 가능한 효율적인 장면 이해 기술로 발전하기 위한 기반을 마련합니다.

- **Performance Highlights**: SPORTS 프레임워크는 세 가지 공개 데이터 세트에서 실시한 포괄적인 실험을 통해 기존의 최첨단 방법들보다 더 나은 성능을 입증하였습니다. 이 모델은 비주얼 오도메트리, 객체 추적, 분할 및 새로운 시점 합성 작업에서 뛰어난 정확도를 보였습니다. 특히, 세그멘테이션 품질이 3.07 % 향상된 결과를 보여주어, 시장에 적합한 장면 이해 방법으로 자리잡을 가능성을 높였습니다.



### FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution (https://arxiv.org/abs/2510.12747)
Comments:
          Project page with code: this https URL

- **What's New**: 이 논문에서 소개된 FlashVSR는 실시간 비디오 초해상도(VSR)를 위한 최초의 확산 기반 원스텝 스트리밍 프레임워크입니다. 이 모델은 단일 A100 GPU에서 768x1408 해상도의 비디오를 약 17 FPS로 처리할 수 있습니다. FlashVSR는 세 가지 주요 혁신을 통해 이루어진 속도와 품질의 획기적인 개선을 제공합니다.

- **Technical Details**: FlashVSR는 세 가지 기술적 혁신으로 구성됩니다: 첫째, 훈련 친화적인 3단계 디스틸레이션 파이프라인으로, 효율적인 슈퍼 해상도 모델을 만들기 위한 과정입니다. 둘째, 지역 제약이 있는 스파스 어텐션을 사용하여 중복 계산을 줄이고 훈련-테스트 해상도 간의 갭을 줄입니다. 마지막으로, 간결한 조건부 디코더를 통해 회복 속도를 높이고 품질 손실 없이 설계를 간소화합니다.

- **Performance Highlights**: FlashVSR는 기존의 다른 확산 기반 VSR 방법들보다 최대 12배의 속도 향상을 보여줍니다. 또한, 이 모델은 1440p 해상도에서도 매우 세밀한 비디오를 reliably 처리할 수 있습니다. VSR-120K라는 새로운 대규모 데이터셋을 구축하여 비디오 훈련과 이미지를 동시에 지원하며, 이 데이터셋은 VSR 연구를 진전을 이루게 할 것입니다.



### Personalized Federated Fine-Tuning of Vision Foundation Models for Healthcar (https://arxiv.org/abs/2510.12741)
Comments:
          Accepted to the Symposium on Model Accountability, Sustainability and Healthcare (SMASH) 2025

- **What's New**: 본 논문에서는 의료 분야에 AI의 가능성을 열어주는 Foundation models에 대해 다루고 있습니다. 기존의 의료 데이터로 사전 훈련된 모델들도 특정 다운스트림 작업(Downstream tasks)을 위해서는 여전히 추가적인 Fine-tuning이 필요합니다. 데이터를 보호하기 위한 제한들로 인해 데이터 공유가 어렵기 때문에, 여러 병원 등 참여 기관을 통한 Federated Learning이 좋은 대안으로 제안됩니다.

- **Technical Details**: 제안된 방법인 Federated Orthogonal Personalized Adapter Learning (FedOPAL)은 개별 클라이언트의 데이터와 다른 클라이언트의 데이터를 모두 활용할 수 있는 개인화된 연합 Fine-tuning 방법입니다. LoRA (Low-Rank Adaptation) 어댑터를 사용하여 클라이언트에 독립적인 정보와 클라이언트 특유의 정보를 분리하여 학습합니다. 두 개의 어댑터 전역(Global)과 개인(Personal) 어댑터가 동시에 학습되어 서로의 기능을 보완할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실제 의료 이미징 데이터셋에서의 초기 결과에 따르면, FedOPAL은 기존의 연합 Fine-tuning 방법들과 경쟁력 있는 성능을 보였습니다. 평가에서는 6개 기관의 데이터를 사용하여 피부 병변 이미지를 분류하는 과제를 포함하였으며, 성능 평가는 불균형 클래스(imbalance class)를 고려한 균형 정확도(balanced accuracy)를 기준으로 진행되었습니다. 전체적으로, 제안된 방법은 여러 방법과 비교했을 때 경쟁력을 유지하고 있지만, 모든 클라이언트에서 어떤 방식이 항상 최선인지를 이해하기 위한 추가적인 연구가 필요합니다.



### Beyond Seeing: Evaluating Multimodal LLMs on Tool-Enabled Image Perception, Transformation, and Reasoning (https://arxiv.org/abs/2510.12712)
- **What's New**: 이번 연구에서는 Multimodal Large Language Models (MLLMs)가 사용자 제공 이미지의 불완전성을 해소하기 위한 새로운 접근법을 제시합니다. 기존의 정적 이미지 접근 방식에서 벗어나, 이미지를 능동적으로 조작하고 다른 도구와 통합하여 복잡한 작업을 해결하는 'think with images' 패러다임을 도입합니다. 이를 위해 IRIS라는 새로운 벤치마크를 발표하여 MLLMs의 시각적-텍스트적 작업 수행 능력을 평가합니다.

- **Technical Details**: IRIS는 1,204개의 도전적인 오픈 엔디드 비전 작업을 포함하고 있으며, 단일 턴(603개)과 멀티 턴(601개)으로 나눠져 있습니다. 이러한 작업은 다섯 가지 다양한 분야를 아우르며, 체계적인 평가를 위한 자세한 루브릭을 제공합니다. 평가 결과, 현재 MLLMs는 비전과 일반 도구의 효과적인 통합이 필요한 작업에서 어려움을 겪고 있으며, 가장 강력한 모델인 GPT-5-think 역시 단지 18.68%의 통과율을 기록했습니다.

- **Performance Highlights**: IRIS를 통해 얻어진 평가에서 OpenAI 모델은 다양한 이미지 조작을 통해 이점을 보이는 반면, Gemini-2.5-pro는 개선을 보이지 않는다는 점이 관찰되었습니다. 이는 MLLMs의 비주얼 인텔리전스를 향상시키기 위한 중요한 통찰을 제공합니다. 'think with images' 중심의 벤치마크를 도입함으로써, 비전 및 인지 작업의 새로운 방향성을 제시하고 있습니다.



### Hybrid Explanation-Guided Learning for Transformer-Based Chest X-Ray Diagnosis (https://arxiv.org/abs/2510.12704)
Comments:
          Accepted by iMIMIC at MICCAI 2025

- **What's New**: 본 연구에서는 Hybrid Explanation-Guided Learning (H-EGL) 프레임워크를 제안하여, self-supervised와 human-guided 제약 조건을 결합하여 attention 정렬을 개선하고 일반화를 향상시킵니다. 이 접근 방식은 높은 비용의 수동 감독 없이도 효과적인 모델 교육을 가능하게 하여 깊이 있는 이해를 제공합니다. H-EGL은 Vision Transformer (ViT) 기반의 chest X-ray 분류 작업에서 기존의 두 가지 state-of-the-art 방법을 초월하는 성능을 보여주었습니다.

- **Technical Details**: H-EGL은 Discriminative Attention Learning (DAL)이라는 self-supervised 모듈과 human alignment 중앙화를 중점적으로 둔 supervised 모듈로 구성됩니다. 이 프레임워크는 semi-supervised 채널을 통해 특징을 강조하며, 각 입력에 대한 attention 양식이 클래스 간 구별을 늘리도록 설계되었습니다. DAL은 class-distinctive attention maps를 생성하고, 비슷한 두 attention maps 간의 cosine similarity를 최소화하기 위한 손실 함수를 적용합니다.

- **Performance Highlights**: 모델의 성능은 chest X-ray 이미지에서 네 가지 일반적인 흉부 병리(무기폐, 심비대, 응집 및 흉수)의 분류로 평가되었습니다. H-EGL은 성능 평가 결과, 두 개의 기존 방식보다 우수한 분류 정확도를 달성하였으며, 각 병리의 전문가 알라인먼트를 통해 보다 잘 정렬된 attention maps를 생성했습니다. 또한, 모델의 견고성과 일반화를 평가하기 위해, validation 및 test 세트 간의 성능 차이도 측정되었습니다.



### EReLiFM: Evidential Reliability-Aware Residual Flow Meta-Learning for Open-Set Domain Generalization under Noisy Labels (https://arxiv.org/abs/2510.12687)
Comments:
          The source code is available at this https URL

- **What's New**: 이번 논문에서는 Open-Set Domain Generalization (OSDG) 문제에서 Label noise(라벨 노이즈)의 영향을 줄이기 위해 Evidential Reliability-Aware Residual Flow Meta-Learning (EReLiFM)이라는 새로운 방법을 제안합니다. EReLiFM은 라벨 신뢰성 인식을 증진하는 비감독식 두 단계의 evidential loss(증거 손실) 클러스터링 방법과 구조화된 잔여를 모델링하는 residual flow(잔여 흐름) 매칭 메커니즘을 통합하여 다양한 전이 경로를 허용합니다. 이 접근법은 유사성을 기반으로 한 증강을 넘어, 불확실성과 다양한 정보를 제공하여 모델의 투명성과 적합성을 향상시킵니다.

- **Technical Details**: EReLiFM은 두 부분으로 구성됩니다. 첫째, UTS-ELC(무감독 이론적 증거 손실 클러스터링)는 예측 오류와 관련된 불확실성을 포착하여 클린 및 노이즈 샘플 간의 신뢰성 있는 분리를 가능하게 합니다. 둘째, DC-CRFM(도메인 및 카테고리 조건화 잔여 흐름 매칭)은 구조화된 잔여를 학습하여 카테고리 간의 다양한 전이 경로를 모델링합니다. 이러한 메타-러닝 과정에서 모델은 클린 집합에서의 업데이트 방향을 조정하여 노이즈 집합에서 손실 감소를 극대화합니다.

- **Performance Highlights**: EReLiFM은 PACS, DigitsDG 및 TerraINC 데이터셋에서 실험을 통해 기존의 OSDG-NL 방법들보다 우수한 성능을 달성하였습니다. 이는 다양한 단서 제공과 정확한 최적화를 보장하는 EReLiFM의 효과를 보여줍니다. 성능 향상은 라벨 신뢰성 인식과 도메인-카테고리 전이 모델링 간의 시너지에 의해 이루어졌습니다.



### MCOP: Multi-UAV Collaborative Occupancy Prediction (https://arxiv.org/abs/2510.12679)
- **What's New**: 이 논문에서는 다수의 무인 항공기(UAV) 시스템을 위한 새로운 협업 점유 예측 프레임워크를 제안합니다. 기존의 Bird's Eye View(BEV) 기반 방식의 한계를 보완하기 위해 3D 공간 구조와 의미론적 정보를 효과적으로 보존하는 메커니즘을 통합했습니다. 이를 통해 UAV 간의 통신 오버헤드를 줄이고 효율성을 높이며, 기존 협업 방법들보다 월등한 성능을 나타냅니다.

- **Technical Details**: MCOP(다중 UAV 협업 점유 예측)은 네 가지 주요 모듈, 즉 Spatial-Aware Feature Encoder, Altitude-Aware Reduction, Dual-Mask Perceptual Guidance, Cross-Agent Feature Integration으로 구성됩니다. Spatial-Aware Feature Encoder는 RGB 이미지를 3D 점유 특성으로 변환하여 장면의 기하학적 세부정보와 의미론적 정보를 포착합니다. Altitude-Aware Reduction은 고도 정보를 유지하면서 특성 차원을 줄이는 압축 메커니즘입니다.

- **Performance Highlights**: 실험 결과, MCOP 방법은 기존의 단일 UAV 인식보다 더 높은 정확도를 달성하며, 모든 평가 데이터 세트에서 더 낮은 통신 오버헤드를 기록했습니다. 또한, 두 개의 가상 데이터 세트와 하나의 실제 데이터 세트를 기반으로 한 평가를 통해 이 방법의 우수성을 입증하였습니다. 이 연구는 다수의 UAV 시스템을 위한 점유 기반 협업 인식 프레임워크의 첫 사례로, 의미론적 점유 예측을 위한 새로운 기준을 마련하는 데 기여합니다.



### TerraCodec: Compressing Earth Observations (https://arxiv.org/abs/2510.12670)
- **What's New**: 본 논문에서는 TerraCodec (TEC)라는 지구 관측 데이터에 특화된 신경망 코덱의 세트를 소개합니다. TEC는 멀티스펙트럴(multi-spectral) 이미지를 위해 최적화된 효율적인 이미지 기반 변형과 시간에 걸쳐 의존성을 활용하는 Temporal Transformer 모델(TEC-TT)을 포함하고 있습니다. 더불어, 기존 신경망 코덱의 고정 비율 설정을 극복하기 위해 Latent Repacking이라는 새로운 방법론을 제시하고 있습니다.

- **Technical Details**: TerraCodec은 Sentinel-2 데이터를 기반으로 학습되어 Classical codecs에 비해 3-10배 더 강력한 압축 성능을 보여줍니다. 이 코덱은 멀티스펙트럴 및 다중 시간대(multi-temporal) 이미지를 처리하기 위해 특별히 설계된 구조체를 포함하고 있으며, 템포럴 트랜스포머는 장기적 의존성을 포착하는데 기여합니다. 학습된 코드들은 고유의 성능 매개변수 및 요구 사항에 따라 유연하게 조정할 수 있습니다.

- **Performance Highlights**: TerraCodec은 기존의 압축 방식들보다 우수한 성능을 자랑하며, 특히 저품질 압축률에서 저장 용량을 최대 10배까지 감소시킬 수 있습니다. TEC-TT는 또한 AllClear 벤치마크에서 최신 기술을 초월하는 제로샷 클라우드 인페인팅(zero-shot cloud inpainting)을 가능하게 합니다. 이 결과들은 특정 분야를 위한 맞춤형 학습 압축 알고리즘이 지구 관측 데이터에 매우 유망한 방향임을 보여줍니다.



### On the Use of Hierarchical Vision Foundation Models for Low-Cost Human Mesh Recovery and Pose Estimation (https://arxiv.org/abs/2510.12660)
Comments:
          Accepted at ICCVW 2025

- **What's New**: 이번 연구는 인간 메쉬 복구(Human Mesh Recovery, HMR) 및 인간 포즈 추정(Human Pose Estimation, HPE) 위한 간단하고 효율적인 모델을 개발하는 데 주안점을 두고 있다. 특히, 기존의 대형 비계층적 비전 변환기(vision transformer) 대신, 계층적 비전 기초 모델(hierarchical vision foundation models, VFMs)의 초기 단계를 인코더로 활용하고자 한다. 연구 결과, 이러한 계층적 VFMs의 처음 몇 단계만 사용해도 전체 모델과 비슷하거나 더 나은 성능을 발휘할 수 있음을 보여준다.

- **Technical Details**: HMR2.0과 그 후속 모델들은 ViTPose와 같은 대형 비계층적 비전 변환기를 인코더로 사용하여 높은 성능을 자랑하지만, 이는 실시간 또는 자원 제약 환경에서는 사용이 어려울 수 있다. 따라서 연구자는 Swin Transformer, GroupMixFormer 및 VMamba와 같은 계층적 VFM의 초기 단계를 인코더로 사용하여 모델의 크기와 컴퓨팅 비용을 줄이고, 여전히 성능을 유지하는 방법을 모색했다. 본 연구에서는 총 27개의 HMR 및 HPE 모델을 구현하고, 이 모델들이 효율성과 정확성 간의 균형을 잘 맞춘다고 주장하고 있다.

- **Performance Highlights**: 모델의 성능 평가는 27개의 계층적 VFM 기반 HMR 및 HPE 모델을 사용하여 이루어졌으며, 인코더로 초기 두세 단계만 사용할 경우에도, 전체 네 단계 모델과 경쟁할 수 있는 성능을 달성하였다. 이러한 절단된 모델들은 기존의 경량화 모델들보다 정확성과 효율성 간의 보다 유리한 균형을 보여주었다. 결국, 연구 결과는 HMR과 HPE에서 계층적 VFMs를 활용한 모델들이 더 나은 성능-효율성 트레이드오프를 제공함을 입증하였다.



### Zero-Shot CFC: Fast Real-World Image Denoising based on Cross-Frequency Consistency (https://arxiv.org/abs/2510.12646)
Comments:
          The British Machine Vision Conference

- **What's New**: 이 논문은 Zero-Shot denoiser based on Cross-Frequency Consistency (ZSCFC)라는 새로운 방법을 제안합니다. 기존의 zero-shot 방법들이 훈련 시간이 길고 노이즈 분포에 대한 가정을 필요로 하는 단점을 극복하여, 단일 노이즈 이미지로 훈련과 디노이징을 가능하게 합니다. 이 방법은 이미지의 주파수 대역 간의 일관성을 활용하여 보다 효율적으로 노이즈를 제거할 수 있습니다.

- **Technical Details**: ZSCFC는 노이즈 이미지의 주파수 대역을 여러 개로 분해하고, ultralight network를 통해 고주파 텍스처를 추출합니다. 이 네트워크는 구조적이고 자연스러운 텍스처를 캡처하는 데 초점을 맞추어, 노이즈 모델에 대한 가정 없이도 효과적인 디노이징을 달성합니다. 제안된 Cross-Frequency Consistency loss를 통해 다양한 주파수 대역 간의 일관성을 학습하여 최종적으로 디노이즈된 이미지를 생성합니다.

- **Performance Highlights**: ZSCFC는 다양한 실제 이미지 데이터셋에서 최신 self-supervised 및 zero-shot 디노이징 방법을 초월한 성과를 나타냅니다. 이 방법은 적은 파라미터 수(1.5k)로도 우수한 디노이징 성능을 발휘하며, 컴퓨팅 효율성 면에서도 매우 뛰어납니다. 이러한 특성 덕분에 ZSCFC는 실제 환경에서도 적용 가능성이 높습니다.



### WaterFlow: Explicit Physics-Prior Rectified Flow for Underwater Saliency Mask Generation (https://arxiv.org/abs/2510.12605)
- **What's New**: 이번 연구에서는 Underwater Salient Object Detection (USOD)의 최신 기술을 제안하는 WaterFlow라는 rectified flow-based 프레임워크를 소개합니다. 기존 방법들이 수중 이미징의 물리적 원리를 무시하거나 단순히 보정의 대상으로 간주한 반면, WaterFlow는 이러한 정보를 네트워크 훈련에 명시적으로 통합합니다. 이로 인해 모델의 현저한 객체 인식 능력이 향상됩니다.

- **Technical Details**: WaterFlow는 Temporal-Aware Conditional Aggregation Module (TACAM)을 사용하여 다중 스케일 RGB 특징을 추출하고, Underwater Physical Prior Module (UPPM)을 통해 물리적 사전 지식(physical prior knowledge)을 데이터와 결합합니다. 이 구조는 물체의 윤곽을 구별하는 데 필요한 정보와 외부 환경으로부터의 복잡성을 잘 처리합니다. 이러한 접근방식은 saliency detection의 다양한 특성을 충족시키는 데 효과적입니다.

- **Performance Highlights**: USOD10K 데이터세트에서 WaterFlow는 0.072의 S_m 향상을 달성하여 그 효과와 우수성을 입증합니다. 이 접근법은 수중 환경에서의 성능을 극대화하며 기존의 방법들과 비교했을 때 현저히 우수한 결과를 보였습니다. 그 코드 또한 향후 공개될 예정입니다.



### Reasoning in the Dark: Interleaved Vision-Text Reasoning in Latent Spac (https://arxiv.org/abs/2510.12603)
- **What's New**: 이번 연구에서는 Interleaved Vision-Text Latent Reasoning (IVT-LR) 방법을 제안하여 시각적 및 텍스트 정보 간의 추론을 잠재 공간(latent space)에서 통합적으로 수행할 수 있게 하였습니다. 이는 기존의 멀티모달 추론 방법이 필요로 했던 명시적인 텍스트나 이미지 생성을 없애고, 효율적인 데이터 처리 및 추론을 가능하게 합니다. IVT-LR은 이전 단계의 히든 상태를 사용하여 잠재 텍스트와 선택된 이미지 임베딩을 결합하는 방법으로 진행됩니다.

- **Technical Details**: IVT-LR 프레임워크에서 각 추론 단계는 잠재 텍스트와 잠재 비전의 두 부분으로 구성됩니다. 잠재 텍스트는 이전 단계의 히든 상태를 기반으로 하며, 잠재 비전은 어텐션 점수에 따라 선택된 이미지 임베딩을 포함합니다. 이 과정을 통해 명시적 추론 단계를 점진적으로 잠재적인 추론 단계로 대체할 수 있도록 훈련시킵니다.

- **Performance Highlights**: M3CoT와 ScienceQA 데이터셋에 대한 실험 결과 IVT-LR 방법이 평균 5.45%의 정확도 향상을 보인 동시에, 기존 방법들에 비해 5배 이상의 속도 증가를 기록했습니다. 이러한 결과는 IVT-LR이 멀티모달 추론에서의 새로운 최첨단 성능을 달성했음을 보여줍니다.



### Advancing End-to-End Pixel Space Generative Modeling via Self-supervised Pre-training (https://arxiv.org/abs/2510.12586)
- **What's New**: 본 논문에서는 픽셀 공간(pixel-space) 생성 모델의 성능 및 효율성 격차를 해소하기 위해 새로운 두 단계 훈련 프레임워크를 소개합니다. 첫 번째 단계에서는 깨끗한 이미지에서 의미 있는 의미론(sematics)을 캡처하기 위해 인코더(encoders)를 사전 훈련하고, 같은 결정적 샘플링 경로의 포인트와 정렬합니다. 두 번째 단계에서는 무작위로 초기화된 디코더(decoder)와 인코더를 통합하여 전체 모델을 확산 모델(diffusion model) 및 일관성 모델(consistency model)용으로 최적화합니다.

- **Technical Details**: 이 훈련 프레임워크는 ImageNet 데이터셋에서 강력한 실험적 성능을 보여줍니다. 특히, 우리의 확산 모델은 ImageNet-256에서 2.04의 FID 값을 달성하고, ImageNet-512에서는 2.35로, 기존 픽셀 공간 방법보다 생성 품질(generation quality)과 효율성(efficiency)에서 큰 차이를 보입니다. 또한, ImageNet-256에서는 우리의 일관성 모델이 단일 샘플링 단계에서 8.82의 FID를 기록하며, 이는 잠재 공간(latent-space) 모델을 현저히 초과합니다.

- **Performance Highlights**: 이 프레임워크는 선행 연구보다 더 높은 수준의 생성 품질과 더 나은 효율성을 보여주면서, VAE 기반 모델과도 경쟁할 수 있는 훈련 비용을 유지합니다. 본 연구의 혁신 중 하나는 미리 훈련된 VAE나 확산 모델에 의존하지 않고, 고해상도 이미지에서 일관성 모델을 성공적으로 훈련한 첫 사례로, 이 분야의 발전에 기여할 것으로 예상됩니다.



### LayerSync: Self-aligning Intermediate Layers (https://arxiv.org/abs/2510.12581)
- **What's New**: LayerSync는 도메인에 구애받지 않는 접근 방식으로, diffusion 모델의 생성 품질과 훈련 효율성을 향상시키기 위해 제안되었습니다. 기존 연구는 훈련을 가속화하고 생성 품질을 개선하기 위해 외부 가이드를 사용할 필요성이 있음을 강조했습니다. 그러나 LayerSync는 모델 자신의 중간 표현을 정규화하여 외부 감독의 필요성을 줄이는 혁신적인 방법을 제공합니다.

- **Technical Details**: LayerSync는 diffusion 모델의 중간 계층 간의 정렬을 수행하는 파라미터 없는 정규화 프레임워크입니다. 이 방법은 추가적인 데이터나 사전 훈련된 모델 없이도 작동하며, 계산 비용이 거의 발생하지 않으면서도 상당한 효과를 보여줍니다. 실험 결과 LayerSync는 ImageNet 데이터셋에서 훈련 속도를 8.75배 이상 가속화하고, 생성 품질을 23.6% 향상시켰음을 보여주었습니다.

- **Performance Highlights**: LayerSync는 이미지 생성뿐만 아니라 오디오 및 비디오 생성에서도 효과적으로 적용할 수 있으며, 오디오 생성에서는 21%, 사람의 움직임 생성에서는 7.7%, 비디오 생성에서는 54.7%의 품질 향상이 있었습니다. 내부 기능 분석에서도 LayerSync는 분류에서 32.4%, 의미론적 분할에서 63.3%의 개선을 이끌어냈습니다. 이는 LayerSync가 다양한 도메인에서 diffusion 모델의 훈련을 가속화할 수 있음을 입증합니다.



### Unlocking Zero-Shot Plant Segmentation with Pl@ntNet Intelligenc (https://arxiv.org/abs/2510.12579)
- **What's New**: 이 논문에서는 농업 이미지를 위한 제로샷(segmentation) 세분화 접근 방식을 제안합니다. Plantnet이라는 대규모 식물 분류 모델과 DINOv2 백본 및 Segment Anything Model(SAM)을 결합하여 새로운 데이터셋을 수집하고 주석을 달 필요 없이 식물 영역을 식별합니다. 이러한 방식으로 생성된 조잡한 세분화 마스크는 SAM을 통해 세밀하게 보완됩니다.

- **Technical Details**: 이 방법은 Plantnet의 미세 조정된 DINOv2 모델을 기반으로 하여 자율적인 식물 표현을 활용합니다. 제공된 토큰 특징은 주어진 데이터셋의 전체 검증 세트에서 주성분 분석(PCA)을 통해 분류되어 식물과 배경으로 구분됩니다. 최종적으로, 분류된 토큰은 바운딩 박스 프롬프트로 그룹화되거나 초보적인 마스크로 사용됩니다.

- **Performance Highlights**: 모델을 평가하기 위해 여러 공개 데이터셋을 사용하여, Plantnet을 사용한 방법이 기본 DINOv2 모델에 비해 일관되게 성능 향상을 보여주었습니다. Jaccard Index(IoU)를 통해 측정된 결과는 다양한 농업 시나리오에서 주석 병목 현상을 해소하는 데 기여할 수 있음을 입증합니다.



### Learning Human Motion with Temporally Conditional Mamba (https://arxiv.org/abs/2510.12573)
Comments:
          10 pages

- **What's New**: 이 논문에서는 인체 동작 생성을 위해 Temporally Conditional Mamba(TCM)라는 새로운 모델을 제안합니다. 이 모델은 기존의 Cross-Attention 메커니즘의 한계를 극복하고, 동작과 입력 조건 간의 일관된 시간적 정렬을 달성하는 데 중점을 두고 있습니다. TCM은 Mamba 블록의 반복 동역학에 조건 정보를 통합하여 보다 정밀한 동작 생성이 가능하도록 합니다.

- **Technical Details**: 기존의 연구들은 Cross-Attention 메커니즘을 사용하여 외부 자극과 동작의 상관관계를 포착하려고 했지만, 이는 주로 글로벌 상호작용만을 포착하여 시간적 정렬이 부족했습니다. TCM은 Mamba의 동역학 내에 직접 조건을 주입하여 시간 정보를 자율회귀적으로 결합함으로써 동작의 일관성을 높입니다. 이를 통해 TCM은 다양한 시간 조건을 기반으로 한 인체 동작 합성 및 추정 과제에서 일반화 가능성을 보여줍니다.

- **Performance Highlights**: TCM은 폭넓은 실험을 통해 기존 최첨단 모델들보다 동작의 질, 조건 정렬에서 현저한 개선을 이루었습니다. 특히, TCM은 다양한 시간적 조건에서도 일관되게 우수한 성능을 보이며, 실제 문제 해결 능력을 입증했습니다. 이러한 결과는 인체 동작 합성 및 추정 작업에서의 새로운 접근 방식을 제시하고 있습니다.



### MMOT: The First Challenging Benchmark for Drone-based Multispectral Multi-Object Tracking (https://arxiv.org/abs/2510.12565)
- **What's New**: 이 논문은 드론 기반 다중 객체 추적(Multi-Object Tracking, MOT)을 위한 최초의 다채널 멀티스펙트럴(Multispectral) 데이터셋인 MMOT를 소개합니다. 기존의 RGB 기반 추적 알고리즘이 겪는 문제점을 보완하기 위해, MMOT는 고해상도 비디오 시퀀스와 정밀한 방향성 경계 상자(OBB) 주석을 제공합니다. 이 데이터셋은 극도로 작은 물체와 복잡한 배경 상황에서의 객체 구분을 향상시키기 위한 중요한 기초 자료로 기능합니다.

- **Technical Details**: MMOT 데이터셋은 총 125개의 비디오 시퀀스와 488.8K의 주석 구역을 포함하며, 여덟 가지 범주에 걸쳐 1200x900 해상도로 촬영됩니다. 이 데이터셋은 드론 장착 멀티스펙트럴 카메라로 실제 도시 환경을 여러 날과 날씨 조건에서 촬영한 결과로 구성되었습니다. 또한 방향성 경계 상자 주석을 통해 공중에서의 정확한 물체 위치 잡기 및 객체 간 혼동을 줄일 수 있습니다.

- **Performance Highlights**: 다수의 실험을 통해 멀티스펙트럴 입력이 RGB 기반 알고리즘보다 추적 성능을 크게 향상시키는 것을 확인했습니다. 특히, 작고 밀집된 객체를 처리할 때 멀티스펙트럴 데이터의 효과가 두드러지게 나타났습니다. MMOT는 향후 다채널 멀티스펙트럴 MOT 연구의 발전을 이끌 것으로 기대됩니다.



### CoIRL-AD: Collaborative-Competitive Imitation-Reinforcement Learning in Latent World Models for Autonomous Driving (https://arxiv.org/abs/2510.12560)
Comments:
          18 pages, 17 figures

- **What's New**: CoIRL-AD는 imitation learning (IL)과 reinforcement learning (RL)을 통합하여 훈련 중 상호작용을 가능하게 하는 경쟁적 이중 정책 프레임워크입니다. 이 방법은 전통적인 두 단계 파라다임을 넘어서며, 두 정책 간의 지식 교류를 촉진하고 gradient conflict를 방지하는 메커니즘을 도입합니다. nuScenes 데이터셋에서의 실험 결과, 충돌 비율이 18% 줄어들었으며, 일반화 능력이 향상되었고, 긴 꼬리 시나리오에서의 성능이 개선되었습니다.

- **Technical Details**: 이 모델은 latent world model을 활용하여 상상 기반 시뮬레이션을 수행하고, 외부 시뮬레이터에 대한 의존성을 피합니다. perception 모듈은 입력된 이미지를 latent state로 인코딩하고, waypoints를 통해 행동 시퀀스를 생성합니다. RL과 IL의 병합을 통해, 각 정책의 목표를 별개의 actor로 분리하고, structured competition을 통한 상호작용을 유도하여 두 메커니즘이 함께 학습될 수 있도록 합니다.

- **Performance Highlights**: CoIRL-AD 프레임워크는 nuScenes와 Navsim 데이터셋에서 extensive한 실험을 통해 성능이 개선됨을 입증하였습니다. 긴 꼬리 시나리오와 충돌 감소에서 더욱 뛰어난 결과를 나타내며, 일반화 능력도 강화되었습니다. 이러한 결과는 기존의 baseline 모델들과 비교했을 때 눈에 띄게 개선된 성능을 보여줍니다.



### Unconditional Human Motion and Shape Generation via Balanced Score-Based Diffusion (https://arxiv.org/abs/2510.12537)
- **What's New**: 이번 연구에서는 score-based diffusion model을 통해 무조건적인 인간 모션 생성을 가능케 하며, 기존의 보조 정규화 손실(auxiliary regularization losses)을 필요로 하지 않는 방법을 제안합니다. SMPL 기반 모션 표현을 최소화하여 운동 데이터를 효율적으로 모델링할 수 있는 접근법을 제공합니다. 이러한 방식은 데이터의 다양한 표현에 의한 과도한 복잡성을 제거하고, 각각의 성분에 대한 이론적 기반을 제공합니다.

- **Technical Details**: 제안한 방법은 표준 L2 score-matching 손실에 대한 체계적인 가중치 부여와 특징 그룹 간의 세심한 정규화(normalization)를 결합하여 구현되었습니다. 실험적으로 고안된 방법들은 각 구성 요소가 독립적으로 효과적임을 보여주는 목표 지향적 ablation 실험을 포함하고 있습니다. 이 연구에서 우리는 SMPL 매개변수에 대한 구조 보존(feature normalization) 기능을 보호하며, L2 score-matching 손실의 이론적 동기를 제시합니다.

- **Performance Highlights**: 성능 테스트 결과, 제안된 방법은 단 31개의 신경 함수 평가(neural function evaluations)로도 기존의 최첨단 성능과 동등한 결과를 달성할 수 있음을 보여줍니다. 또한 무조건적인 인간 모션 확산 훈련을 통해 손실 가중치의 실험적 조정 없이도 효과적인 샘플링과 가능성을 해결할 수 있는 PF-ODE와의 호환성을 확보합니다. 이 방식은 직접적인 형상(generation) 생성을 통해 관절에서의 후속 복구(procedure recovery)를 제거하는 장점을 가지고 있습니다.



### Voronoi-Assisted Diffusion for Computing Unsigned Distance Fields from Unoriented Points (https://arxiv.org/abs/2510.12524)
- **What's New**: 이 논문에서는 UnSigned Distance Fields (UDFs)를 계산하기 위한 경량화된 네트워크 없는 방법인 Voronoi-Assisted Diffusion (VAD)을 제안합니다. 기존의 신경망 방식은 수치적 불안정성과 높은 계산 비용, 제어 가능성의 한계를 가지고 있었으나, VAD는 이러한 문제를 해결합니다. 이 방법은 방향이 없는 점 구름에 대해 직접 UDF를 계산하며, 두 가지 Voronoi 기반 기하학적 기준을 통해 입력 데이터의 이중 방향 법선을 정렬합니다.

- **Technical Details**: VAD는 입력 포인트에 양 방향 법선을 할당한 후, 정렬된 법선을 확산하여 UDF의 기울기 필드를 형성합니다. 이를 통해 최종 UDF를 회복하는 단계를 진행하며, 고유한 에너지 함수로 최적 정렬을 유도합니다. 이 과정에서 Poisson 방정식을 해결하여 UDF를 복원하며, 노이즈가 있는 입력에도 효과적으로 작동하도록 방법을 확장하였습니다.

- **Performance Highlights**: 실험을 통해 VAD는 수밀(watertight) 및 열린 표면, 복잡한 비다양체(non-manifold), 비지향형(non-orientable) 기하학을 안정적이고 효율적으로 처리함을 입증하였습니다. 특히, 기존의 SDF 및 GWN 기반 접근 방식이 실패하는 경우에도 VAD가 효과적으로 작동함을 보여줍니다. 또한, 희소하고 비균일하게 샘플링된 점 구름에서의 강 robustness를 평가하기 위한 테스트도 수행되었습니다.



### BSGS: Bi-stage 3D Gaussian Splatting for Camera Motion Deblurring (https://arxiv.org/abs/2510.12493)
- **What's New**: 본 연구에서 제안하는 Bi-Stage 3D Gaussian Splatting (BSGS) 프레임워크는 모션 블러가 있는 이미지로부터 3D 장면을 효과적으로 복원하는 새로운 방법을 제공합니다. 기존 3DGS 방식의 한계를 극복하기 위해 두 가지 주요 단계를 포함합니다: Camera Pose Refinement와 Global Rigid Transformation으로, 이 두 단계는 모션으로 인한 왜곡을 효과적으로 보정합니다.

- **Technical Details**: BSGS는 Camera Pose Refinement 단계에서 카메라 자세를 대략적으로 최적화하고, Global Rigid Transformation 단계에서는 고정된 카메라 자세를 사용하여 Gaussian 포인트에 전역 변환을 적용합니다. 또한, multi-frame gradient conflicts를 줄이기 위해 subframe gradient aggregation 전략을 제안하며, Space-Time Coupling Densification을 통해 Gaussian 밀도 조정 파라미터를 동적으로 조정합니다.

- **Performance Highlights**: 실험 결과 BSGS는 모션 블러 이미지 처리에서 탁월한 성능을 발휘하여 새로운 뷰에서 선명한 결과물을 생성할 수 있음을 보여주었습니다. 특히, 기존의 고정 임계값 방식보다 블러 지역에서 아티팩트 생성을 방지하는 데 효과적이며, Gaussian 분포 최적화를 통해 성능이 향상되었습니다.



### A Text-Image Fusion Method with Data Augmentation Capabilities for Referring Medical Image Segmentation (https://arxiv.org/abs/2510.12482)
- **What's New**: 이번 연구는 데이터 증강(data augmentation)의 중요성을 강조하면서, 텍스트와 이미지 정보를 통합 하는 혁신적인 초기 융합(early fusion) 방법을 제안합니다. 일반적인 데이터 증강 기법이 측정된 텍스트와 이미지 간의 공간적 정렬(spatial alignment)을 방해하는 문제를 해결하기 위해, 텍스트와 시각적 특성을 증강 전에 결합하여 정렬을 유지하는 접근 방식을 사용합니다. 이를 통해 세 가지 의료 이미지 작업 및 네 가지 분할(segmentation) 프레임워크에서 최첨단 성능을 달성했습니다.

- **Technical Details**: 제안된 방법은 텍스트 인코더(text encoder), 분할 프레임워크(segmentation framework) 및 pseudo 이미지를 생성하는 경량 생성기(lightweight generator)로 구성됩니다. 기존의 데이터 증강 기술들과는 달리, 텍스트 정보를 이미지와 통합하기 전에 프로젝션하여 오류를 줄이는 방안을 채택하였습니다. 이 과정에서 ROI 기반 학습을 통해 텍스트 기능을 추출하고, 최종적으로 생성된 pseudo 이미지는 원본 이미지와 결합됩니다.

- **Performance Highlights**: 모든 데이터셋에서 제안하는 방법은 Dice score와 mIoU를 크게 증가시켜 기존 기준선(baseline)을 초월하는 성능을 보여줍니다. 모델은 UNet, UNet++, TransUNet 및 MISSFormer와 같은 네 가지 프레임워크에서 검토되었고, 제안된 접근 방식이 데이터 증강 없이도 경쟁력을 유지하며, 데이터 증강 적용 시 더욱 우수한 성능을 보임을 입증했습니다. 향후 작업으로는 이 방법을 다양한 의료 분야에 적용할 수 있는 가능성을 검토하고 있습니다.



### MS-GAGA: Metric-Selective Guided Adversarial Generation Attack (https://arxiv.org/abs/2510.12468)
- **What's New**: MS-GAGA(Metric-Selective Guided Adversarial Generation Attack)는 블랙박스 환경에서 딥페이크 탐지기를 겨냥해 전이 가능하고 시각적으로 감지되지 않는 적대적 예시를 생성하기 위한 이단계 프레임워크입니다. 1단계에서는 이중 스트림 공격 모듈을 통해 적대적 후보를 생성하며, 2단계에서는 메트릭 인식 선택 모듈이 후보의 성공률과 원본 이미지와의 구조적 유사성(SSIM)을 평가합니다. 이 연구는 새로운 전략으로 딥페이크 탐지기에 대한 저항성을 높이고자 합니다.

- **Technical Details**: 1단계의 MNTD-PGD는 작은 섭동 예산에 최적화된 강화된 그래디언트 계산을 적용하고, SG-PGD는 시각적으로 두드러진 영역에 섭동을 집중시킵니다. 이러한 보완적인 설계를 통해 적대적 탐색 공간이 확장되어 새로운 모델 간 전이 가능성이 향상됩니다. 이 시스템은 모델 아키텍처나 매개 변수에 대한 액세스 없이도 다양한 최신 탐지기를 성공적으로 피해갈 수 있는 공격을 생성하는 것을 목표로 합니다.

- **Performance Highlights**: MS-GAGA는 최신 공격에 비해 최대 27% 높은 잘못 분류 비율을 달성했습니다. 또한, 적대적 훈련(adversarial training)과 같은 방어 메커니즘을 통해 모델의 탄력성을显著 향상시킬 수 있습니다. 이 연구는 기존의 비현실적인 '화이트박스' 모델에서 벗어나 실제 환경에서 공격이 발생할 수 있는 가능성을 고려하여 학문적으로 뿐만 아니라 실용적으로도 중요한 기여를 하고자 합니다.



### A Review of Longitudinal Radiology Report Generation: Dataset Composition, Methods, and Performance Evaluation (https://arxiv.org/abs/2510.12444)
- **What's New**: 이 논문은 Chest X-ray radiology report generation (CXRRRG)에 대한 혁신적인 접근법을 제시하여 방사선 전문의의 업무 부담을 덜어줄 수 있는 방법을 탐구합니다. 기존의 연구들은 보통 단일 이미지에 의존했으나, 이 연구는 장기 데이터(longitudinal data)를 통합하여 방사선 전문의의 진단 과정을 모방할 수 있는 기회를 제공합니다. 논문에서는 LRRG(장기 방사선 보고서 생성)의 구조와 평가 프로토콜을 분석하며, 연구자들에게 체계적인 모델 설계를 위한 프레임워크를 제공합니다.

- **Technical Details**: 연구는 MIMIC-CXR 데이터셋을 사용하여 장기 데이터 기반의 보고서 생성 기법을 고찰합니다. 다양한 데이터세트 구성 전략과 장기 맞춤형 아키텍처를 통해 보고서 생성 과정에서 필요한 요소들을 분석하고, 성능 평가 지표를 제공합니다. 또한, 장기 정보의 중요성과 아키텍처 설계 선택이 모델 성능 향상에 미치는 영향을 중점적으로 다루고 있습니다.

- **Performance Highlights**: LRRG 접근 방식을 통한 최근 연구 성과와 여러 ablation study의 결과를 종합적으로 분석하여 장기 데이터의 통합이 성능에 미치는 긍정적인 영향을 강조합니다. 현재 연구의 한계를 진단하고 향후 발전 방향을 제시하며, 장기 데이터 통합이 CXRRRG 시스템에서 어떻게 효과적으로 이루어질 수 있을지를 보여줍니다. 논문은 장기 데이터 통합을 통해 방사선 보고서 생성의 정확성과 질을 향상시킬 수 있는 가능성을 제시합니다.



### VideoLucy: Deep Memory Backtracking for Long Video Understanding (https://arxiv.org/abs/2510.12422)
Comments:
          NeurIPS-2025 Accepted Paper

- **What's New**: 최근 연구에 따르면, 큰 언어 모델(LLMs)을 활용한 에이전트 기반 시스템이 긴 비디오 이해를 위한 유망한 접근법으로 부각되고 있습니다. 하지만 이러한 시스템은 개별 프레임에서의 모델링 및 추론에 어려움을 겪고 있으며, 연속적인 프레임 간의 시간적 맥락을 포착하기 힘든 두 가지 주요 문제에 직면해 있습니다. 이를 해결하기 위해, 우리는 VideoLucy라는 심층 메모리 역추적 프레임워크를 제안합니다.

- **Technical Details**: VideoLucy는 인지 과학에서 영감을 받아 거칠고 세밀한 메모리를 점진적으로 회상하는 계층적 메모리 구조를 사용합니다. 이 메모리 구조는 다양한 계층 깊이에서 메모리의 세부 수준 및 시간적 범위를 명확히 정의하며, 에이전트 기반의 반복적 역추적 메커니즘을 통해 비디오 전반에 걸친 질문 관련 심층 메모리를 체계적으로 탐색하여 충분한 정보를 수집합니다.

- **Performance Highlights**: VideoLucy는 여러 긴 비디오 이해 벤치마크에서 최첨단 방법들과 비교해 크게 우수한 성능을 보여줍니다. 예를 들어, LVBench에서 VideoLucy는 58.8%의 정확도를 달성하며, 이는 GPT-4o보다 9.9% 향상된 결과입니다. 이 연구는 미래의 긴 비디오 이해 분야에 대한 연구 개발의 길을 열어주는 중요한 기여를 할 것입니다.



### Low-Field Magnetic Resonance Image Quality Enhancement using a Conditional Flow Matching Mod (https://arxiv.org/abs/2510.12408)
- **What's New**: 이 논문은 conditional flow matching (CFM)에 기반한 새로운 이미지 품질 전송 프레임워크를 소개합니다. 기존의 생성 모델과 달리 CFM은 최적의 속도 필드를 직접 회귀하여 노이즈 분포와 목표 데이터 분포 간의 연속 흐름을 학습합니다. 이 방법은 저비용의 저자기능식 MRI(low-field MRI)에서 고해상도 이미지를 재구성하여 품질 격차를 해소하는 데 중점을 두고 있습니다.

- **Technical Details**: CFM은 시간 간격 t∈[0,1]에서 소스 분포 p0∼xnoise에서 타겟 분포 p1∼xhigh로의 연속 변환을 가정합니다. 이를 위해 오르디너리 미분 방정식을 사용하며, 경량화된 U-Net 아키텍처를 기반으로 여러 구성 요소가 통합되어 있습니다. 여기에는 다중 스케일 입력 레이어, Squeeze-and-Excitation 모듈이 통합된 잔여 블록, 그리고 해상도를 조정하기 위한 픽셀 언셰플 및 픽셀 셔플 기법이 포함됩니다.

- **Performance Highlights**: 실험 결과 CFM은 최신 기술 대비 뛰어난 성능을 보여주었으며, 구성 요소 수는 현저히 적었습니다. 인-디스트리뷰션(InD) 및 아웃-오브-디스트리뷰션(OOD) 데이터에 대해 강인하게 일반화하는 능력을 입증했습니다. 이 결과는 CFM이 특히 리소스가 제한된 임상 환경에서 MRI 재구성을 위한 강력하고 확장 가능한 도구로서의 가능성을 강조합니다.



### Towards General Urban Monitoring with Vision-Language Models: A Review, Evaluation, and a Research Agenda (https://arxiv.org/abs/2510.12400)
Comments:
          44 pages

- **What's New**: 이 논문은 도시 공공 인프라 모니터링에서 Vision-Language Models (VLMs)의 역할을 체계적으로 검토하며, 특히 zero-shot 애플리케이션에 초점을 맞추고 있습니다. 기존의 IoT 센서와 수작업 검사 방식의 한계를 극복할 수 있는 가능성을 제시합니다. 2021년부터 2025년까지 발표된 32개의 동료 심사 논문을 분석하여, 도시 감시 및 인프라 상태 평가를 위한 최신 기술 동향을 파악했습니다.

- **Technical Details**: 논문에서는 PRISMA 방법론을 기반으로 VLMs와 관련된 연구의 메타 분석 과정을 설명합니다. 연구의 주요 키 요소로는 VLMs의 적용 및 비교, 효과성과 성능 측정을 다루었습니다. Zero-shot 또는 few-shot 학습 접근 방식을 사용하는 도시 작업의 현황과 메서디컬 혁신을 모색하며 데이터셋과 배포 현실성도 검토합니다.

- **Performance Highlights**: 최종적으로, 32개의 문서로 구성된 결과가 현재 도시 AI 시스템에서 VLMs의 실행 가능성을 강조하며 정리됐습니다. 이 시스템은 다국어 지원 및 유동적인 상황 인식 같은 기능으로 도시 환경의 인프라 상태를 보다 효율적으로 모니터링 할 수 있는 잠재력을 가지고 있습니다. 연구는 향후 도시 AI 시스템의 설계와 개발을 위한 로드맵을 제시합니다.



### Scene Coordinate Reconstruction Priors (https://arxiv.org/abs/2510.12387)
Comments:
          ICCV 2025, Project page: this https URL

- **What's New**: 이 논문에서는 Scene Coordinate Regression (SCR) 모델을 위한 새로운 확률적 재해석을 제안합니다. 이는 학습된 깊이 값의 분포에 대한 간단한 사전부터 시작하여 그럴듯한 장면 좌표 구성에 대한 학습된 사전까지 다양한 높은 수준의 재구성 사전을 통합할 수 있게 합니다. 이 연구는 다양한 실내 스캔으로 훈련된 3D 포인트 클라우드 확산 모델을 통해 장면 지오메트리를 더 효과적으로 추론할 수 있는 방법을 제시합니다.

- **Technical Details**: SCR 모델은 특정 장면에 대해 훈련되며, 일반적으로 지극히 약한 사전을 사용합니다. 이 모델은 최대 가능성 프레임워크에서 재구성 사전을 통합하기 위해 훈련 목표를 재구성합니다. 수작업으로 제작된 사전은 깊이 값이 합리적인 분포를 따르도록 강제하며, 학습된 사전은 3D 포인트 클라우드 확산 모델을 통해 장면 지오메트리를 반영하도록 유도합니다.

- **Performance Highlights**: 본 연구에서 제안된 사전들은 세 가지 실내 데이터셋에서 구현되어 더욱 일관된 장면 표현을 학습하는 데 기여하였습니다. 이러한 접근은 더 높은 등록 비율과 정교한 카메라 포즈 추정으로 이어져, 새로운 시점 합성 및 카메라 재위치 설정과 같은 후속 작업에 긍정적인 영향을 미칩니다. SCR 모델의 훈련 시간을 크게 증가시키지 않으며, 효율성에도 영향을 미치지 않습니다.



### Learning to Recognize Correctly Completed Procedure Steps in Egocentric Assembly Videos through Spatio-Temporal Modeling (https://arxiv.org/abs/2510.12385)
Comments:
          26 pages, 7 figures and 5 tables in the main paper and one figure and table in the appendix. To be published in Computer Vision and Image Understanding

- **What's New**: 본 연구에서는 절차 단계 인식(Procedure Step Recognition, PSR)을 위한 새로운 프레임워크인 STORM-PSR을 제안합니다. 이 모델은 공간적(spatial) 및 시간적(temporal) 특징을 동시에 활용하여 절차의 모든 단계를 인식합니다. 특히, 기존 방법들이 미비하게 다룬 부분인 부분 가림에 대한 robust한 처리 능력을 갖추고 있습니다.

- **Technical Details**: STORM-PSR은 두 개의 스트림으로 구성된 모델로, 첫 번째 스트림은 assembly state detection(ASD)을 통해 객체의 조립 상태를 인식하고, 두 번째 스트림은 transformer 기반의 시간 인코더를 활용하여 시간적 정보를 연계합니다. 이 방식은 가림이 있는 경우에도 절차 단계의 완성을 직접적으로 예측할 수 있도록 설계되었습니다. 또한, Key-frame sampling(KFS)과 Key-clip aware sampling(KCAS)이라는 약한 지도 학습을 활용한 새로운 방법을 포함하여 데이터 효율성을 극대화합니다.

- **Performance Highlights**: STORM-PSR은 IndustReal과 MECCANO 데이터셋에서 기존 방법들보다 각각 26.1%와 11.2% 평균 지연 시간(delay)을 줄이며 성능이 향상되었음을 입증하였습니다. 이러한 개선은 주로 시간적 특징을 활용한 spatio-temporal 스트림에 의한 것입니다. 또한, 새로운 MECCANO 레이블을 통해 PSR에 대한 성능 기준을 설정하여 연구 기여도를 높였습니다.



### Deep Attention-guided Adaptive Subsampling (https://arxiv.org/abs/2510.12376)
- **What's New**: 본 논문에서는 Deep Attention-guided Subsampling (DAS)이라는 새로운 프레임워크를 제안하여, 입력에 적응하는 동적 샘플링을 통해 3D 의료 영상 및 비디오 분류 작업에서 성능을 개선합니다. 기존 방법들은 비정상적인 샘플링 패턴 또는 고정된 접근 방식을 사용했으나, DAS는 attention 메커니즘을 활용하여 개별 입력에 따라 샘플링 방식을 조정합니다. 이로 인해 성능 향상과 계산 복잡성을 줄이는 동시에, 효율적인 추론이 가능해집니다.

- **Technical Details**: 제안된 DAS는 가벼운 기능 추출 모듈, attention 레이어, Gumbel-Softmax 샘플링 메커니즘으로 구성됩니다. 다양한 특징을 파악하기 위해 다중 경로를 사용하여 입력 시퀀스의 리치(representative) 표현을 생성합니다. 이후 multi-head attention 레이어를 통해 최종 샘플링 로짓을 생성하며, 각 헤드는 전반적인 attention 분포를 스케일링하여 최적의 샘플링 결정에 기여합니다.

- **Performance Highlights**: DAS는 MedMNIST3D 데이터셋 및 실제 임상 환경에서 수집된 데이터셋을 포함한 여러 의료 영상 데이터셋에서 효과를 입증했습니다. 제안된 방법은 샘플링 프로세스의 해석을 용이하게 하고 리소스가 제약된 환경에서도 효율적인 추론을 가능하게 하여, 기존 방법들에 비해 훨씬 더 나은 성능을 보여주었습니다.



### CurriFlow: Curriculum-Guided Depth Fusion with Optical Flow-Based Temporal Alignment for 3D Semantic Scene Completion (https://arxiv.org/abs/2510.12362)
- **What's New**: 본 연구에서는 CurriFlow라는 새로운 프레임워크를 제안하여, 단일 이미지로부터 3D 장면의 기하학적 정보 및 의미적 정보를 동시에 복원할 수 있는 방법을 제시합니다. 기존의 접근 방식들이 모션 추론을 명확하게 수행하지 못하는 반면, CurriFlow는 optical flow 기반의 시간 정렬과 커리큘럼 학습이 결합된 깊이 융합 방식을 통해 이러한 문제를 해결하고자 합니다.

- **Technical Details**: CurriFlow는 optical flow를 활용하여 역사적 프레임의 특성을 현재 프레임에 정렬하며, 이를 통해 객체 수준의 모션을 명확하게 모델링합니다. 더불어 커리큘럼 학습을 통해 훈련 중 정확한 LiDAR 깊이에서 시작하여 점진적으로 노이즈가 있는 스테레오 깊이로 전환함으로써 기하학적 견고성을 향상시킵니다. 또한, Segment Anything Model(SAM)을 통해 얻은 의미적 선행 지식이 격자 수준의 의미적 학습을 강화합니다.

- **Performance Highlights**: SemanticKITTI 벤치마크에서의 실험 결과, CurriFlow는 16.9의 평균 IoU를 달성하며 최신 기술 중 가장 좋은 성능을 보여줍니다. 이 연구는 카메라 기반의 3D 의미적 장면 완료를 위한 모션 유도 및 커리큘럼 인식 디자인의 효과를 검증하며, 실시간 처리에 적합한 경량 및 정확한 SSC 방법의 필요성을 강조합니다.



### Hybrid Gaussian Splatting for Novel Urban View Synthesis (https://arxiv.org/abs/2510.12308)
Comments:
          ICCV 2025 RealADSim Workshop

- **What's New**: 본 논문은 2025년 ICCV에서 개최된 RealADSim-NVS 챌린지에서 Qualcomm AI Research의 솔루션을 설명합니다. 이 챌린지는 거리 장면에서의 새로운 뷰 합성(Novel View Synthesis, NVS)에 초점을 맞추며, 참가자는 특정 트레이닝 경로에서 촬영된 자동차 중심 프레임을 바탕으로 다른 경로에서 본 동일한 도시 환경의 렌더링을 생성해야 합니다.

- **Technical Details**: 우리의 접근 방식은 장면 생성과 생성적 시뮬레이터에서의 하이브리드 방법에 영감을 받았습니다. 이 솔루션은 두 단계로 구성되며, 첫째로 3D Gaussian Splatting을 사용하여 장면의 3D 재구성을 수행합니다. 이후, 특정 노이즈 모델을 사용하여 렌더링된 프레임의 품질을 향상시킵니다.

- **Performance Highlights**: 우리 모델 디자인의 성능을 PSNR, SSIM, LPIPS 지표로 평가했으며, 공개 리더보드에서 총 점수 0.432로 전체 28개의 제출물 중 2위를 차지했습니다. 이러한 성과는 특히 3DGS 프레임을 향상시키기 위한 신중한 데이터 세트 구성 덕분에 가능했습니다.



### Vision Language Models Map Logos to Text via Semantic Entanglement in the Visual Projector (https://arxiv.org/abs/2510.12287)
- **What's New**: 본 논문에서는 Vision Language Models (VLMs)의 취약성을 조사하고, 로고에서 발생하는 환각( hallucination) 현상에 집중합니다. 로고에 텍스트가 없음에도 불구하고 브랜드 이름이나 텍스트 콘텐츠를 생성하는 현상을 상세히 분석했습니다. 연구는 신뢰할 수 있는 다중 모달 시스템을 구축하기 위한 새로운 진단적 관점과 실행 가능한 완화 인사이트를 제공합니다.

- **Technical Details**: 논문은 VLMs에서 로고를 순수 기호, 하이브리드, 텍스트 포함 로고로 분류하고, 아홉 가지 구조적 교란을 통해 환각을 체계적으로 측정합니다. 임베딩 레벨 분석을 통해, 환각이 소수의 프로젝터 방향에 연관되어 있음을 보여주며, 이는 모델의 강력한 왜곡에도 불구하고 지속적으로 나타납니다.

- **Performance Highlights**: 연구 결과, 로고 환각 현상은 여러 최신 VLM 모델에서 일관되게 나타나며, 특히 아이코닉 원형 로고에서 더욱 두드러집니다. 이러한 발견은 다중 모달 사고의 유용성과 위험성을 강조하며, 모드 간 혼란을 줄이기 위한 노력이 필요함을 시사합니다. 향후 연구는 OCR 적 맞춤 해독과 프로젝터 분리화를 통해 이러한 문제를 해결할 수 있는 가능성을 제시합니다.



### Dual Learning with Dynamic Knowledge Distillation and Soft Alignment for Partially Relevant Video Retrieva (https://arxiv.org/abs/2510.12283)
- **What's New**: 이 논문에서는 기존의 텍스트-비디오 검색(text-to-video retrieval) 모델이 가지는 한계점을 극복하는 '부분 관련 비디오 검색(Partially Relevant Video Retrieval, PRVR)'이라는 새로운 검색 개념을 제안하고 있습니다. 기존의 T2VR 모델은 비디오가 미리 잘려 있어 쿼리와 완전히 관련된 내용만 포함된다는 전제하에 작동하지만, 실제로는 긴 비디오에서 불필요한 배경 내용이 포함되어 있어 복잡합니다. PRVR은 이러한 현실적인 비디오 시나리오에 맞춰 불필요한 내용이 포함되더라도 쿼리와 부분적으로 관련된 비디오를 검색하는 것을 목표로 합니다.

- **Technical Details**: 연구진은 강력한 대규모 비전-언어(pre-trained vision-language) 모델에서 일반화 지식을 정제하고, 이를 경량의 PRVR 네트워크에 전이하는 새로운 프레임워크를 제안합니다. 특히, 이 중 '동적 지식 증류(Dynamic Knowledge Distillation, DKD++)'가 포함된 이중 학습(Dual Learning) 프레임워크를 도입하여 대형 teacher 모델이 경량의 student 네트워크를 지도합니다. 두 개의 student 브랜치는 각각 전이 지식을 흡수하는 상속(inheritance) 브랜치와 PRVR 데이터셋의 특정 정보를 학습하는 탐색(exploration) 브랜치로 구성되어 있으며, 훈련 과정에서 동적 soft-target 구성 메커니즘을 통해 더욱 효과적으로 학습할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, 제안하는 모델이 PRVR에 대한 세 가지 데이터셋인 TVR, ActivityNet, Charades-STA에서 최첨단 성능을 달성한 것으로 나타났습니다. 특히, PRVR의 복잡성을 고려한 동적 soft-target 감독 메커니즘을 통해 모델이 비디오와 쿼리 간의 세부적인 부분 관련성을 더 잘 포착할 수 있음을 보여주었습니다. 이 연구는 이전 T2VR 모델의 한계를 극복하고, 보다 현실적인 비디오 검색 환경에 적합한 접근 방법을 제공하는 데 중점을 두고 있습니다.



### PAGS: Priority-Adaptive Gaussian Splatting for Dynamic Driving Scenes (https://arxiv.org/abs/2510.12282)
- **What's New**: 이 논문에서는 자동 운전 시스템을 위한 동적 3D 도시 장면의 복원을 다루며, 'Priority-Adaptive Gaussian Splatting (PAGS)'라는 새로운 프레임워크를 소개합니다. PAGS는 3D 복원 및 렌더링 과정에 작업 인식의 의미적 우선순위를 직접 주입하여, 안전에 중요한 객체에 대한 세부 사항을 보존하면서 비판적이지 않은 장면 요소를 단순화합니다. 이 프레임워크는 기존의 방법들에서 경험하는 계산 비용과 충실도 간의 거래를 최소화합니다.

- **Technical Details**: PAGS는 두 가지 주요 기여를 통해 의미적 우선순위를 통합합니다. 첫째, 'Semantically-Guided Pruning and Regularization' 전략을 통해, 의미적으로 중요한 요소의 세부 사항은 유지하면서 비판적이지 않은 요소를 효율적으로 제거합니다. 둘째, 'Priority-Driven Rendering' 파이프라인에서는 높은 우선 중요도를 가진 원시 요소들을 사용하여 깊이 맵을 생성하고, GPU의 하드웨어 가속 Early-Z 테스트를 이용해 가려진 조각들을 미리 제외함으로써 렌더링 속도를 크게 향상시킵니다.

- **Performance Highlights**: Waymo와 KITTI 데이터셋에 대한 광범위한 실험을 통해 PAGS는 안전-critical 객체의 복원 품질을 향상시키면서, 훈련 시간을 줄이고 렌더링 속도를 350 FPS 이상으로 증가시킴을 입증했습니다. 이러한 개선은 PAGS의 프레임워크가 기존 방법보다 뛰어난 재구성 충실도와 계산 효율성을 제공함을 보여줍니다.



### SpineBench: Benchmarking Multimodal LLMs for Spinal Pathology Analysis (https://arxiv.org/abs/2510.12267)
Comments:
          Proceedings of the 33rd ACM International Conference on Multimedia,ACMMM 2025 Dataset Track

- **What's New**: 이 논문에서는 척추 영역에 특화된 시각적 질문 응답(Visual Question Answering, VQA) 벤치마크인 SpineBench를 소개합니다. 기존의 의료 벤치마크는 일반적인 임상 과제에 초점을 맞추고 있어 척추와 같은 세부적인 영역에서의 성능을 적절히 평가하지 못했습니다. SpineBench는 64,878개의 QA 쌍과 40,263개의 척추 이미지를 포함하여 11개의 척추 질병을 다룹니다.

- **Technical Details**: SpineBench는 두 가지 주요 임상 과제인 척추 질병 진단과 척추 병변 위치 파악을 포함합니다. 두 작업 모두 다중 선택 형식으로 제공되고, 각 VQA 쌍에 대해 시각적 유사성을 기반으로 한 어려운 부정 옵션들을 샘플링하여 현실 세계 시나리오에서의 도전 과제를 시뮬레이션합니다. 이 데이터셋은 서로 다른 소스에서 수집된 이미지-레이블 쌍을 표준화하여 이루어졌으며, 전문가들에 의해 검증되고 보정됩니다.

- **Performance Highlights**: 우리의 연구는 12개의 주요 MLLM 모델이 SpineBench에서 낮은 성능을 보였으며, 이는 척추 영역에서 현재 MLLM의 한계를 강조합니다. 대부분의 모델은 우연히 맞히는 것에 가까운 정확도를 기록하며, 각 모델의 추론 과정을 전문가들이 평가한 결과에서도 척추 영역에 대한 지식과 논리적 추론에서의 결함이 나타났습니다. SpineBench는 공개적으로 제공되어 향후 연구 및 개선을 위한 기초 자료가 될 것입니다.



### AngularFuse: A Closer Look at Angle-based Perception for Spatial-Sensitive Multi-Modality Image Fusion (https://arxiv.org/abs/2510.12260)
Comments:
          For the first time, angle-based perception was introduced into the multi-modality image fusion task

- **What's New**: 본 연구는 AngularFuse라는 새로운 이미지 융합 기법을 제안합니다. 이 기법은 기존의 단점인 손실 함수의 설계 한계를 극복하고, 모달리티 간 상호보완적인 정보 학습을 촉진합니다. 특히, 방향을 고려한 새로운 손실 함수를 도입하여 텍스처 강도와 엣지 방향성을 동시에 조정하는 데 성공했습니다.

- **Technical Details**: 방법론적으로, AngularFuse는 세 가지 주요 구성 요소로 이루어져 있습니다. 첫째, Complementary Mask Generation (ComMask) 모듈을 통해 모달리티 간의 보완적인 정보를 강화합니다. 둘째, Fine-Grained Reference Image Synthesis (FRIS) 기법을 통해 보다 상세한 참조 이미지를 생성합니다. 셋째, Angle-aware Perception을 통해 방향성을 고려한 경량 손실 함수를 적용합니다.

- **Performance Highlights**: AngularFuse는 MSRS, RoadScene 및 M3FD 공공 데이터셋에서의 포괄적인 실험을 통해 기존 방법에 비해 우수한 성능을 보였습니다. 특히, 시각적 비교를 통해 도전적인 장면에서도 더 선명하고 상세한 결과를 만들어내며, 향상된 융합 능력을 입증하였습니다.



### Local Background Features Matter in Out-of-Distribution Detection (https://arxiv.org/abs/2510.12259)
- **What's New**: 이번 연구에서는 딥 뉴럴 네트워크의 OOD(Out-Of-Distribution) 탐지를 위한 새로운 방법을 제안하였다. 이 방법은 ID(인식 가능한 분포) 이미지에서 로컬 배경 특성을 추출하여 가짜 OOD 특성으로 활용하는 전략을 사용한다. OOD 이미지와 ID 이미지의 배경이 유사하다는 점에 착안하여, 이러한 배경 특성들이 OOD 데이터에 대한 과신 문제를 경감하는 데 도움을 준다.

- **Technical Details**: 제안된 방법은 사전 훈련된 네트워크 모델을 미세 조정하는 과정을 포함한다. 먼저, ID 데이터의 특성 맵에서 로컬 배경 특성을 샘플링하고, 이를 통해 OOD 특성을 흉내낸다. 최적화 과정에서는 이 배경 특성의 $L_2$-norm을 최소화하여 모델이 배경 정보에 덜 집중하도록 유도한다.

- **Performance Highlights**: 여러 표준 OOD 탐지 벤치마크에서 실험을 수행한 결과, 제안된 방법은 기존의 사후 처리(post-hoc) 방법들과의 조합에서도 뛰어난 성능을 보였다. 새로운 최첨단 성능을 달성하는 데 성공하였으며, OOD 탐지 분야에서의 개선을 기대할 수 있다.



### Multiplicative Loss for Enhancing Semantic Segmentation in Medical and Cellular Images (https://arxiv.org/abs/2510.12258)
Comments:
          Accepted by ICCV2025 Workshop "Third Workshop on Computer Vision for Automated Medical Diagnosis"

- **What's New**: 이 논문에서는 의료 이미지 및 세포 이미지의 의미론적 분할을 위한 두 가지 새로운 손실 함수인 Multiplicative Loss와 Confidence-Adaptive Multiplicative Loss를 제안합니다. 기존의 Cross Entropy와 Dice Loss는 데이터가 제한적일 때 성능이 저하되는 문제가 있으며, 이러한 문제를 해결하기 위해 Multiplicative Loss는 이들 손실 함수를 곱셈 방식으로 결합하여 동적으로 그래디언트를 조절합니다. 이는 잘못된 예측에 대해 그래디언트를 증가시켜 최적화를 안정화하며, 특히 데이터가 부족할 때 안정된 훈련을 가능하게 합니다.

- **Technical Details**: Multiplicative Loss는 Cross Entropy Loss와 Dice Loss를 곱하여 결합하며, 이는 두 손실의 보완적인 속성을 활용합니다. Confidence-Adaptive Multiplicative Loss(CAML)는 예측 확률에 기초하여 손실의 스케일을 동적으로 조정하여 극단적인 데이터 부족 상황에서도 보다 효과적인 학습을 제공하는 구조를 갖추고 있습니다. 이 접근 방식은 특히 의료 이미지 분야에서 유용하며, 신뢰도가 낮은 샘플의 그래디언트를 강화하여 학습 효과를 향상합니다.

- **Performance Highlights**: 실험 결과, 제안된 Multiplicative Loss는 모든 데이터셋에 대해 개별 손실 함수에 비해 클래스별 IoU 및 평균 IoU에서 일관되게 성능이 향상되었습니다. COVID-19 데이터셋에서 훈련 샘플 수를 줄이면서도 기존 방법보다 1.0-1.4% 향상된 성능을 보여 CAML의 효과성을 입증하였습니다. 이러한 손실 함수는 하이퍼파라미터 조정이 필요 없이 단순하고 효과적인 방식으로, 도전적인 데이터 제한 상황에서도 신뢰할 수 있는 분할을 가능하게 합니다.



### Vectorized Video Representation with Easy Editing via Hierarchical Spatio-Temporally Consistent Proxy Embedding (https://arxiv.org/abs/2510.12256)
- **What's New**: 이번 논문에서는 비디오에서 동적으로 변화하는 객체와 장면을 안정적으로 표현하기 위해 새로운 스페이쇼-템포랄(spatio-temporal) 프록시 노드(proxy node)를 제안합니다. 기존의 영상 표현 방식은 픽셀 단위의 트래킹 오류와 외부 요인으로 인해 큰 취약성을 보였지만, 제안된 방식은 다중 스케일 구조를 안정적으로 표현할 수 있습니다. 또한, 동적인 표현 업데이트 메커니즘이 적용되어 비디오의 변화를 효과적으로 처리할 수 있습니다.

- **Technical Details**: 제안된 방법은 비디오 씬을 의미 레이어(semantic layers)로 분해한 후, 각 레이어에 프록시 노드를 초기화합니다. 이 노드는 형태와 텍스처를 암묵적으로 인코딩하여 고성능 복원 및 편집을 가능하게 합니다. 프록시 노드는 스파스하게 분포되어 있어 트래킹 오류에 대해 더 강인하며, 새로운 노드를 동적으로 삽입하고 전파하여 추적 오류를 보완합니다.

- **Performance Highlights**: 이 새로운 영상 표현 방식은 적은 매개변수로도 높은 비디오 재구성 정확도를 달성하고, 비디오 인페인팅(video in-painting) 및 키프레임 기반의 시간 일관성 비디오 편집 등 복잡한 비디오 처리 작업을 지원합니다. 실험 결과, 제안된 방법이 기존 방식보다 더 뛰어난 성능을 보이는 것을 확인하였으며, 고정밀 비디오 편집을 가능하게 합니다.



### Ivan-ISTD: Rethinking Cross-domain Heteroscedastic Noise Perturbations in Infrared Small Target Detection (https://arxiv.org/abs/2510.12241)
Comments:
          In infrared small target detection, noise from different sensors can cause significant interference to performance. We propose a new dataset and a wavelet-guided Invariance learning framework(Ivan-ISTD) to emphasize this issue

- **What's New**: 본 논문은 드론 기반의 다중 모달 감지에서 중요한 역할을 하는 적외선 소형 목표 탐지(Infrared Small Target Detection, ISTD)에 대한 새로운 접근 방식을 제안하고 있습니다. 제안된 Ivan-ISTD 프레임워크는 두 가지 주요 단계로 구성되어 있으며, 첫째는 Wavelet-guided Cross-domain Synthesis를 통해 목표 도메인에 맞게 훈련 샘플을 생성하는 것입니다. 둘째 단계에서는 Real-domain Noise Invariance Learning을 통해 실제 잡음 특성을 추출하여 동적 잡음 라이브러리를 구축하는 방식을 채택하였습니다.

- **Technical Details**: Ivan-ISTD의 첫 번째 단계에서 Wavelet-guided Cross-domain Synthesis는 다중 주파수 웨이브렛 필터링을 통해 목표 배경을 정확하게 분리합니다. 이러한 기법은 학습 단계 동안 데이터 공간에서 목표 도메인 특성에 적응하게 하며, 그 결과 소형 목표 탐지의 안정성을 높입니다. 두 번째 단계인 Real-domain Noise Invariance Learning은 목표 도메인에서 실제 잡음 특성을 추출하고, 잡음에 대한 불변성을 학습하는 자가 지도 손실(self-supervised loss)을 사용합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존의 최첨단 기술에 비해 많은 정량적 메트릭에서 우수한 성능을 보였습니다. 특히 Ivan-ISTD는 다양한 실제 데이터셋을 통해 교차 도메인 시나리오에서 뛰어난 견고성을 보여주었습니다. 마지막으로 새로 제안된 Dynamic-ISTD 벤치마크는 실제 응용에서 발생하는 배포 변화를 시뮬레이션하여 보다 현실적인 성능 평가를 가능하게 합니다.



### BIGFix: Bidirectional Image Generation with Token Fixing (https://arxiv.org/abs/2510.12231)
- **What's New**: 이번 논문에서는 이미지 및 비디오 생성의 혁신적인 훈련 체계를 제안합니다. 특히 샘플링 중 자기 수정(self-correcting) 기능을 기반으로 한 다중 토큰 예측(multi-token prediction) 프레임워크에 집중하고 있습니다. 기존의 모델이 가진 한계, 즉 잘못된 예측을 수정할 수 없는 문제를 해결하는 방법을 제시합니다.

- **Technical Details**: 모델은 훈련 중 배경 토큰(context tokens) 내에 랜덤 토큰을 주입하며, 다음 토큰을 예측하는 동시에 랜덤으로 주입된 토큰을 수정하는 방식으로 훈련됩니다. 샘플링 과정에서 모델은 여러 토큰을 병렬로 예측하면서도 이전에 샘플링된 토큰을 '백트래킹(backtrack)'하여 수정할 수 있는 메커니즘을 가지고 있습니다.

- **Performance Highlights**: 이번 연구는 ImageNet-256과 CIFAR-10 데이터셋을 이용한 이미지 생성, 그리고 UCF-101 및 NuScenes를 통한 비디오 생성에서 모두 상당한 성능 향상을 입증하였습니다. 이로써 기존의 한계에서 벗어나, 더 나은 품질과 빠른 시간 내에 이미지 및 비디오 생성을 가능하게 하는 접근 방식의 효과를 분명히 했습니다.



### HoneyBee: Data Recipes for Vision-Language Reasoners (https://arxiv.org/abs/2510.12225)
Comments:
          32 pages

- **What's New**: 이 논문에서는 Vision-Language Models (VLMs)의 성능을 개선하기 위해 데이터 커레이션 방식을 연구합니다. HoneyBee라는 2.5백만 개의 예제를 포함한 대규모 고품질 Chain-of-Thought (CoT) 데이터셋을 생성하여 VLM의 추론 능력을 향상시킵니다. 이 연구는 특히 VLM 교육에서 컨텍스트 소스 전략의 영향을 조사합니다.

- **Technical Details**: 우리는 다양한 VLM 훈련 데이터 소스의 성능 차이를 분석하고, 시각적 간섭 및 난이도 필터링과 같은 목표 지향적 데이터 개입을 구현합니다. CoT 데이터 생성기를 통해 문제에 대한 텍스트 기반의 Chain-of-Thought를 생성하고, 이러한 데이터의 비율을 조정하며 VLM 훈련 및 평가 설정을 통제합니다.

- **Performance Highlights**: HoneyBee로 훈련된 VLM은 기존 최신 모델보다 성능이 향상됩니다. 예를 들어, 3B 파라미터를 가진 HoneyBee 훈련 VLM은 MathVerse에서 7.8%와 24.8%의 성능 개선을 기록합니다. 또한 우리는 테스트 시 디코딩 비용을 73% 감소시키는 전략을 제안하며, 이는 정확성을 희생하지 않습니다.



### DIANet: A Phase-Aware Dual-Stream Network for Micro-Expression Recognition via Dynamic Images (https://arxiv.org/abs/2510.12219)
- **What's New**: 본 논문에서는 micro-expression recognition (MER)을 위한 새로운 이중 스트림 프레임워크 DIANet을 제안합니다. DIANet은 두 가지 단계인 onset-to-apex와 apex-to-offset을 따로 인코딩하는 phase-aware dynamic images를 활용합니다. 이를 통해 미세한 표정의 상승 및 하강 모션 패턴을 효과적으로 포착할 수 있습니다.

- **Technical Details**: 제안된 DIANet은 두 개의 독립적인 스트림을 통해 입력된 영상을 처리하며, 각 스트림은 수정된 EfficientNetV2 아키텍처를 기반으로 합니다. 각 스트림은 onset-apex와 apex-offset 단계의 동적 이미지를 생성하며, 이들 간의 정보를 교환하기 위해 cross-attention fusion 모듈을 사용합니다. 이를 통해 모델은 두 단계의 중요한 모션 신호에 주목하여 보다 정밀한 feature를 학습할 수 있음을 보여줍니다.

- **Performance Highlights**: 세 가지 벤치마크 MER 데이터셋(CASME-II, SAMM, MMEW)에서 실시한 실험 결과, 제안된 DIANet은 기존의 단일 단계 DI 기반 방법들에 비해 일관되게 뛰어난 성능을 보였습니다. 결과는 temporal phase 정보를 명확히 모델링하는 것이 MER의 발전을 위한 유망한 방향임을 제시합니다. 이 연구는 phase-specific DI 표현을 end-to-end 프레임워크에 통합한 첫 사례로, 마이크로 표정 인식의 정확성을 높이는 데 기여할 것입니다.



### The Impact of Synthetic Data on Object Detection Model Performance: A Comparative Analysis with Real-World Data (https://arxiv.org/abs/2510.12208)
Comments:
          18 pages, 12 figures, 2 tables. Code: this https URL ; Data: this https URL

- **What's New**: 최근 생성적 AI(Generative AI)와 컴퓨터 비전(CV)의 발전은 물류와 제조업 같은 다양한 산업의 워크플로우 최적화에 새로운 기회를 제공합니다. 하지만 많은 AI 애플리케이션은 전문 지식과 자원의 부족으로 인해 일반 모델에 의존하게 되는 경우가 많습니다. 이러한 문제를 해결하기 위해, 합성 데이터(Synthetic Data)를 사용하여 모델을 조정하는 방법이 주목받고 있으며, 이는 실제 데이터를 수집하는 것보다 비용 효율적인 대안으로 확인되었습니다. 본 연구는 물류 창고 내에서 객체 탐지 모델의 성능에 미치는 합성 데이터의 영향을 조사하고 있습니다.

- **Technical Details**: 본 연구에서는 NVIDIA Omniverse Replicator 도구를 사용하여 생성한 합성 데이터의 효과를 실제 시나리오에서 특정 창고 물류 문제, 즉 팔렛트 탐지(pallet detection)에 적용하여 분석했습니다. 실험은 실제 데이터와 다양한 합성 데이터 생성 전략을 활용하여 진행했으며, 이는 기존 객체 탐지 모델의 성능을 면밀히 평가할 수 있는 기회를 제공합니다. 데이터의 사실감을 변화시키고, 합성 데이터가 모델 성능에 미치는 영향을 분석하기 위해, 다양한 수준의 데이터 리얼리즘을 활용한 조정 기법을 실험했습니다.

- **Performance Highlights**: 실험의 결과는 합성 데이터와 실제 데이터의 균형 잡힌 통합이 객체 탐지 모델의 견고성과 효율성을 향상시킬 수 있음을 보여주었습니다. 특히, 데이터 현실성과 조정 기법의 다양성에 따른 모델 성능의 미세한 변화가 관찰되었습니다. 이 연구에서는 최적의 데이터 조합 및 조정 절차를 제시하며, 각기 다른 시나리오에서의 적용 가능성을 논의합니다. 최종적으로 GitHub와 Zenodo를 통해 연구에 필요한 모든 자원과 정보를 공유하고 있어, 추후 연구자들이 이를 활용하여 보다 발전된 연구를 진행할 수 있을 것입니다.



### Hierarchical Reasoning with Vision-Language Models for Incident Reports from Dashcam Videos (https://arxiv.org/abs/2510.12190)
Comments:
          2nd Place Winner, ICCV 2025 2COOOL Competition

- **What's New**: 최근 자율 주행의 End-to-End (E2E) 접근법이 발전하면서 다양한 대규모 운전 데이터셋이 활용되고 있지만, 모델들은 여전히 out-of-distribution (OOD) 상황에서 어려움을 겪고 있습니다. 이를 해결하기 위해 COOOL 벤치마크는 폐쇄된 분류를 넘어서 위험 인식을 촉진하는 것을 목표로 하며, 2COOOL 챌린지는 대시캠 비디오에서 인간이 해석할 수 있는 사고 보고서를 생성하는 것으로 그 범위를 확장합니다. 이 연구에서는 프레임 수준의 캡션, 사건 프레임 탐지 및 비전-언어 모델(vision-language models, VLMs) 내에서의 세밀한 추론을 통합하여 사고 보고서 생성을 위한 계층적 추론 프레임워크를 제안합니다.

- **Technical Details**: 계층적 추론 프레임워크는 비디오를 훈련하여 사건 보고서를 생성하는 목표로 세 가지 모듈로 구성됩니다: 프레임 수준 캡션 생성, 사건 프레임 탐지 및 사건 캡션 생성. 첫 번째 단계에서는 각 프레임을 VLM에 입력하여 로컬 캡션을 생성하고, 이를 통해 사고와 관련된 객체에 대한 메타데이터도 출력됩니다. 두 번째 단계인 사건 프레임 탐지에서는 LLM을 사용해 현재 위험한 상황을 예측하고, 마지막으로 사건 캡션 생성 단계에서는 특정 사건 프레임 주위의 프레임들을 활용하여 사건 보고서를 생성합니다.

- **Performance Highlights**: 저자들의 연구 결과는 2COOOL 오픈 리더보드에서 29개 팀 중 2위에 랭크되었으며, 최고의 CIDEr-D 점수를 달성하여 정확하고 일관된 사건 내러티브를 생성했습니다. 이 성과는 VLM을 이용한 계층적 추론 방식이 사고 분석 및 안전-critical 카테고리의 교통 사고 이해를 확대하는 데 있어 유망한 방향임을 나타냅니다. 공식 리더보드는 평가 기준별 점수를 보고하며, 최종 순위는 제출된 보고서의 CIDEr-D, METEOR, SPICE 점수 평균에 의해 결정됩니다.



### CompoDistill: Attention Distillation for Compositional Reasoning in Multimodal LLMs (https://arxiv.org/abs/2510.12184)
Comments:
          Preprint. Under Review

- **What's New**: 최근에 효율적인 Multimodal Large Language Models (MLLMs)이 높은 계산 복잡성을 해결하는 방법으로 주목받고 있습니다. 이러한背景에서, Knowledge Distillation (KD) 접근 방식이 대형 모델(teacher)에서 소형 모델(student)로 시각적 및 언어적 지식을 전이하는 유망한 대안으로 등장했습니다. 기존의 KD 방법은 teacher MLLM의 시각적 인식 능력을 student에게 효과적으로 증류하지 못하는 문제를 발견했습니다.

- **Technical Details**: 체계적인 분석을 통해 student와 teacher 간의 시각적 주의력(alignment) 불일치가 이 문제의 주요 원인으로 확인되었습니다. 이 통찰을 바탕으로 우리는 CompoDistill이라는 새로운 KD 프레임워크를 제안하였으며, 이는 student의 시각적 주의력을 teacher와 명시적으로 정렬하여 student의 시각적 인식 능력을 향상시킵니다. 이 프레임워크는 컴포지셔널 추론(compositional reasoning) 작업에서 학생의 성능을 현저히 향상시키며, 기존 연구에서와 마찬가지로 시각적 질문 응답(visual question answering) 작업에서도 강력한 성능을 유지합니다.

- **Performance Highlights**: CompoDistill을 통해 우리는 시각적 인식 능력이 필요한 작업에서 성능 향상을 실험적으로 입증하였습니다. 게다가, CompoDistill은 보다 진보된 백본(backbone) 모델에서도 효과적임을 보여줘 그 일반화 가능성을 강조하고 있습니다.



### BEEP3D: Box-Supervised End-to-End Pseudo-Mask Generation for 3D Instance Segmentation (https://arxiv.org/abs/2510.12182)
- **What's New**: 이번 논문에서는 3D instance segmentation을 위한 BEEP3D라는 모델을 제안합니다. 기존의 box-level annotations을 활용하여 더욱 효율적인 supervision을 제공하며, 두 단계의 학습 과정에서 발생하는 시간과 복잡성을 해결하고자 합니다. 특히, student-teacher 프레임워크를 적용하여 pseudo-labeler 역할을 하는 teacher 모델을 구현했습니다.

- **Technical Details**: BEEP3D는 teacher 모델을 Exponential Moving Average를 통해 학생 모델이 업데이트합니다. 또한, instance center 기반의 query refinement 기법을 도입하여 위치 쿼리의 로컬화 정밀도를 향상시킵니다. 이를 통해 instance 중심 근처의 특징들을 효과적으로 활용할 수 있습니다. 새로운 두 가지 손실 함수, 즉 query consistency loss와 masked feature consistency loss를 설계하여 예측값과 pseudo-mask 간의 의미적 및 기하학적 신호를 정렬합니다.

- **Performance Highlights**: ScanNetV2와 S3DIS 데이터셋에서의 광범위한 실험 결과, BEEP3D는 최신 약한 감독 방법들과 비교하여 경쟁력 있거나 우수한 성능을 보여주었습니다. 이 모델은 계산 효율성을 유지하면서도 높은 정확도를 달성하는 데 성공했습니다.



### UniGS: Unified Geometry-Aware Gaussian Splatting for Multimodal Rendering (https://arxiv.org/abs/2510.12174)
- **What's New**: 본 논문에서는 UniGS라는 새로운 통합 맵 표현 및 차별화된(differentiable) 프레임워크를 제안합니다. UniGS는 3D Gaussian Splatting을 기반으로 하여, RGB 이미지, 깊이(Depth) 맵, 표면 법선(normals), 의미적 로짓을 동시에 렌더링할 수 있는 기능을 갖추고 있습니다. 이 새로운 접근 방식은 렌더링 중 각 픽셀이 개별 Gaussian 타원체와의 교차점을 결정하도록 하여 기하학적 일관성을 유지합니다.

- **Technical Details**: 새롭게 설계된 rasterization 방법을 통해 깊이는 Gaussian 중심이 아닌 ray-ellipsoid 교차점을 통해 렌더링됩니다. 이를 통해 Gaussian의 회전(rotation) 및 스케일(scale) 속성을 효과적으로 최적화할 수 있는 가능성을 열었습니다. 또한, 훈련 중에 기여도가 낮은 Gaussian을 차별화된 방식으로 가지치기할 수 있는 학습 가능한 속성을 도입하여, 저장 공간 효율성을 높였습니다.

- **Performance Highlights**: 정량적 및 정성적 실험 결과, 모든 모달리에서 최첨단 재구성 정확도를 달성했음을 보여줍니다. 특히 깊이 추정 정확도가 66.4% 향상되었고, Gaussian primitive 수는 17.2% 감소했습니다. 본 프레임워크는 렌더링 효율성을 개선하면서도 모든 모달리 간의 일관성을 유지하는 성능을 발휘합니다.



### State Space Prompting via Gathering and Spreading Spatio-Temporal Information for Video Understanding (https://arxiv.org/abs/2510.12160)
- **What's New**: 최근 논문은 비디오 분류에 대한 프리트레인된 상태 공간 모델의 잠재력을 강조합니다. 기존의 비디오 프롬프트 토큰이 비디오의 공간적 및 시간적 맥락 정보를 충분히 포착하지 못하는 문제를 해결하기 위해, 우리는 Intra-Frame Gathering (IFG) 및 Inter-Frame Spreading (IFS) 모듈을 통합한 State Space Prompting (SSP) 방법을 제안합니다. 이 접근 방식은 비디오 내의 주요 시공간 정보를 효과적으로 집계하고 전파하는 기법을 제공합니다.

- **Technical Details**: 제안된 SSP 방법은 Intra-Frame Gathering 모듈을 통해 각 프레임 내의 공간적 정보를 집계하고, Inter-Frame Spreading 모듈을 통해 시간적 정보를 전파합니다. 이러한 구조는 비디오에서 중요한 정보의 집계와 전파 과정을 효율적으로 조절하며, 정보 엔트로피를 활용하여 각 프레임에 대한 주의력을 조정합니다. 결과적으로 상태 공간 압축 모델 내의 공간 및 시간 정보의 효과적인 전파를 가능하게 합니다.

- **Performance Highlights**: 다양한 비디오 벤치마크 데이터셋에서 수행된 실험 결과, SSP 방법이 기존의 최첨단(SOTA) 방법보다 평균 2.76% 향상된 성능을 달성했으며, 튜닝 가능한 파라미터의 과부하를 줄였습니다. SSP 방식을 통해 데이터 효율성을 높임과 동시에 비디오 이해 성능을 향상시키는 것이 확인되었습니다. 이는 단지 ∼3%의 조정 가능한 파라미터로도 가능했습니다.



### DPL: Spatial-Conditioned Diffusion Prototype Enhancement for One-Shot Medical Segmentation (https://arxiv.org/abs/2510.12159)
Comments:
          Accepted at IVCNZ 2025. To be published in IEEE proceedings

- **What's New**: 이 연구에서는 Diffusion Prototype Learning (DPL)이라는 새로운 프레임워크를 소개하여 단일 샷 의료 이미지 분할의 원형(prototype) 생성을 개편하였습니다. DPL은 전통적인 결정론적 접근법 대신 학습 가능한 확률 분포로 프로토타입을 모델링하여 최소한의 레이블 된 데이터로부터 다양한 프로토타입 변형을 생성합니다. 특히, DPL은 확산 기반(feature space exploration) 접근 방식을 활용하여 보다 일관된 세멘틱을 유지합니다.

- **Technical Details**: DPL은 세 가지 핵심 혁신으로 구성되어 있습니다: 첫째, 단일 지원(prototype) 프로토타입을 전방-역방향 확산 과정을 통해 다양한 변형 집합으로 변환하는 확산 기반 프로토타입 강화 모듈입니다. 둘째, 기하학적 속성을 활용하는 공간 인식 조건화 메커니즘이 도입되어 만들어진 프로토타입 변형들이 해부학적 유효성을 유지합니다. 셋째, 퓨전(fusion) 전략은 프로토타입의 완전성을 유지하면서도 다양한 표현력을 극대화합니다.

- **Performance Highlights**: 광범위한 실험을 통해 복부 MRI 및 CT 데이터 세트에서 DPL 프레임워크의 성능을 평가한 결과, 기존 최첨단 방법들과 비교하여 현저한 개선이 이루어졌습니다. 이는 DPL이 단일 샷 의료 이미지 분할 분야에서 새로운 성능 기준을 설정하였음을 나타냅니다.



### Class-aware Domain Knowledge Fusion and Fission for Continual Test-Time Adaptation (https://arxiv.org/abs/2510.12150)
- **What's New**: 이번 연구는 Continual Test-Time Adaptation (CTTA) 접근 방식을 제안하고, 기존 방법들의 제한점을 극복하기 위한 새로운 프레임워크인 KFF(Knowledge Fusion and Fission)를 소개합니다. KFF는 고전적인 도메인 지식의 융합 및 분리를 통해 이전 및 새로운 도메인 간의 지식을 효과적으로 적응하도록 돕습니다. 이는 하이브리드 모델을 사용하여, 데이터 흐름에 따라 변화하는 다양한 테스트 도메인에 능동적으로 대응할 수 있는 새로운 방법론을 제시합니다.

- **Technical Details**: KFF는 Knowledge FIssion (KFI) 및 Knowledge FUsion (KFU) 모듈을 이용하여 서로 다른 도메인의 지식을 동적으로 조정합니다. KFI 모듈은 현재 도메인에 맞는 카테고리 인식 도메인 지식을 분리하고, KFU 모듈은 이 분리된 지식을 기존 지식 풀에 최소한의 비용으로 통합합니다, 이 과정에서 Greedy 기법을 사용하여 새로운 지식과 기존 지식 간의 호환성을 높이고 계산의 효율성을 유지합니다.

- **Performance Highlights**: 실험 결과, KFF는 여러 테스트 도메인에서 효과적인 적응 능력을 발휘하며, 특히 ImageNet-C 데이터셋에서 34.8%의 오류율을 기록하여 기존 SOTA 방법인 DPCore보다 5.1% 향상된 성능을 보였습니다. 이러한 성과는 KFF 방식의 적응력이 시간을 두고 변화하는 다양한 도메인에 대해 매우 효과적임을 증명합니다.



### FedHUG: Federated Heterogeneous Unsupervised Generalization for Remote Physiological Measurements (https://arxiv.org/abs/2510.12132)
- **What's New**: 이 논문에서는 원거리 생리학적 측정(Remote Physiological Measurement, RPM)의 한계를 극복하기 위해 새로운 프로토콜인 연합 비지도 도메인 일반화(Federated Unsupervised Domain Generalization, FUDG)를 도입합니다. 특히, 연합 이질 비지도 일반화(Federated Heterogeneous Unsupervised Generalization, FedHUG) 프레임워크를 제안하여, 다양한 환경에서 수집된 비라벨 데이터의 일반화 성능을 강화합니다. 이는 고유한 집합체 평가를 통해 각 클라이언트의 가중치를 조정함으로써 효과적으로 도메인 비대칭 문제에 대응합니다.

- **Technical Details**: FedHUG 프레임워크는 두 가지 주요 모듈로 구성됩니다. 첫째, 최소 편향 집계(Minimal Bias Aggregation, MBA) 모듈은 클라이언트의 편향을 평가하여 동적으로 가중치를 조정합니다. 둘째, 글로벌 분포 인식 학습 제어기(Global Distribution-aware Learning Controller, GDLC)는 클라이언트의 학습 전략을 조정하여 원거리 빈 Label 문제를 완화시킵니다. 이를 통해 서버와 클라이언트 간의 레이블 분포 불균형을 줄이고, 다양한 환경에서 모델의 성능을 최적화합니다.

- **Performance Highlights**: 이 연구는 RGB 비디오와 mmWave 레이더 모두에서 최신 기술(Stat of the Art, SOTA) 대비 우수한 성능을 보여줍니다. 또한, 대규모 벤치마크를 통해 다양한 클라이언트 환경에서 FUDG의 효과를 검증하였고, 비라벨 데이터를 활용하여 모델 일반화 능력을 개선했습니다. 이러한 연구 결과는 생리학적 데이터의 실제 환경 적용을 촉진할 것으로 기대됩니다.



### MetaCaptioner: Towards Generalist Visual Captioning with Open-source Suites (https://arxiv.org/abs/2510.12126)
- **What's New**: 이번 논문에서는 CapFlow라는 혁신적인 다중 에이전트 협업 워크플로우를 제안하여 오픈 소스 모델을 통해 일반화된 비주얼 캡셔닝(visual captioning) 능력을 달성합니다. CapFlow는 특히 저렴한 비용으로 GPT-4.1과 같은 상업 모델과 동등한 캡션 품질을 달성할 수 있음을 처음으로 보여줍니다. 이 연구의 주요 목표는 데이터 합성(data synthesis)의 한계를 극복하고, 다양한 시각적 도메인에서 고품질 캡션을 생성하는 것입니다.

- **Technical Details**: CapFlow는 비주얼 캡셔닝 작업을 하위 작업으로 분해하고, 다양한 시각적 측면의 증거를 통합하는 방식을 기반으로 합니다. 각각의 에이전트는 지각(perception), 시각적 지식 추출(visual knowledge extraction), 시각적 추론(visual reasoning) 등의 역할을 수행하며, 최종적으로 하나의 캡션으로 집계됩니다. CapFlow는 도메인 라우팅 메커니즘을 통해 다양한 시각적 도메인에 적합한 워크플로우를 동적으로 할당하여 도메인 간의 차이를 극복합니다.

- **Performance Highlights**: MetaCaptioner는 CapFlow를 기반으로 하여 저비용으로도 강력한 비주얼 캡셔닝 능력을 수행합니다. 실험 결과, MetaCaptioner는 상업 모델인 GPT-4.1과 유사한 캡셔닝 능력을 갖춤을 보여주며, 다운스트림 평가에서도 유의미한 성과를 기록했습니다. CapFlow와 MetaCaptioner는 저렴하면서도 강력한 비주얼 캡셔닝 솔루션을 제공하여 향후 멀티모달 연구에 큰 도움이 될 것입니다.



### Hardware-aware Coding Function Design for Compressive Single-Photon 3D Cameras (https://arxiv.org/abs/2510.12123)
Comments:
          IEEE TPAMI Special Issue

- **What's New**: 이 논문은 시간 측정이 정확한 싱글 포톤 카메라의 성능을 향상시키기 위해 하드웨어 제약을 고려하여 효율적인 코딩 함수를 설계하는 제약 최적화 접근 방식을 제시합니다. 기존의 압축 히스토그램 프레임워크는 하드웨어 제약에 대한 고려 없이 최적화되어 실제 시스템에서 성능이 저하되는 문제를 해결할 수 있습니다. 이 연구는 최적의 깊이 정확도를 유지하면서 하드웨어 제약을 준수하는 고성능 코딩 함수를 식별하는 방법을 보여줍니다. 또한, 다양한 매개변수화된 임펄스 응답에 적응할 수 있는 가능성도 제시합니다.

- **Technical Details**: 논문에서 제안하는 접근 방식은 조합 최적화 방법론을 사용하여 조명 및 코딩 행렬을 공동 최적화합니다. 이를 통해 하드웨어 제약을 준수하는 최적의 코딩 함수를 설계할 수 있습니다. 사용된 방법론은 그래디언트 하강법(gradient descent)을 기반으로 하며, 최적화된 코딩 함수는 피크 전력 제한(p peak power constraints) 시스템에서 특히 두드러진 성능 향상을 보여줍니다. 또한, 비이상적인 임펄스 응답 함수(non-ideal impulse response function)에서의 작동이 가능함을 시뮬레이션을 통해 검증합니다.

- **Performance Highlights**: 연구 결과, 제안된 코딩 함수는 기존의 압축 히스토그램 솔루션보다 하드웨어 제한이 있는 시스템에서 더 우수한 깊이 정확도를 달성했습니다. 특히 피크 전력 제약을 고려할 때, 제안된 방법은 대칭의 멀티 피크 파형을 최적화하여 기존 방법들보다 현저한 개선을 보여줍니다. 이 실험 결과는 상용으로 사용할 수 있는 3D 이미징 시스템 개발에 중요한 통찰력을 제공합니다.



### ImageSentinel: Protecting Visual Datasets from Unauthorized Retrieval-Augmented Image Generation (https://arxiv.org/abs/2510.12119)
Comments:
          Accepted at NeurIPS 2025

- **What's New**: 최근의 Retrieval-Augmented Image Generation (RAIG) 기술의 발전은 참조 이미지를 통해 생성 품질을 향상시키는 놀라운 능력을 보여주었습니다. 하지만 이러한 시스템의 민감한 이미지 데이터셋이 무단으로 사용될 수 있는 문제는 매우 심각한 상황이 되었습니다. 이에 따라 본 논문에서는 ImageSentinel이라는 새로운 프레임워크를 제안하여, 개인 데이터셋을 보호하고 무단 사용을 감지하는 문제를 해결하고자 합니다.

- **Technical Details**: ImageSentinel은 두 가지 주요 단계로 구성된 보호 전략을 포함하고 있습니다. 첫 번째 단계는 비전-언어 모델을 사용하여 개인 데이터셋의 이미지에 대한 포괄적인 설명을 생성하는 것이며, 두 번째 단계에서는 이러한 설명을 무작위 문자 키와 결합하여 sentinel 이미지를 생성합니다. 이로 인해 데이터셋과의 매끄러운 통합이 가능하며, 사전 정의된 키를 통해 안전하게 검색할 수 있습니다.

- **Performance Highlights**: 실험 결과, ImageSentinel은 승인된 응용 프로그램에 대한 생성 품질을 유지하면서도 데이터셋의 무단 사용을 효과적으로 감지할 수 있는 성능을 보여주었습니다. 이 연구는 RAIG 시스템에서 비주얼 데이터셋의 보호 문제를 공식화하고, 전략적으로 생성된 sentinel 이미지를 통해 안전한 데이터셋 검증을 가능하게 했습니다.



### Self-Supervised Selective-Guided Diffusion Model for Old-Photo Face Restoration (https://arxiv.org/abs/2510.12114)
- **What's New**: 이 논문에서는 Self-Supervised Selective-Guided Diffusion (SSDiff)라는 새로운 접근 방식을 제안합니다. SSDiff는 약한 가이드를 통해 생성된 의사 참조 얼굴을 활용하여 얼굴 복원을 수행합니다. 이 방식은 이미지의 윤곽을 구조적으로 맞춰주고 자연스러운 색상을 이용해 특정 지역의 복원을 가능하게 합니다.

- **Technical Details**: SSDiff는 구조적 가이드를 전 과정에 적용하고, 이후 단계에서는 색상 정제를 수행하는 단계별 가이드를 도입합니다. 얼굴 분할 맵과 긁힘 마스크를 통해 특정 지역(예: 파손된 부분)의 복원을 선택적으로 진행하며, 주체적 특징을 방해하지 않고 얼굴 색상을 가이드합니다. 이 방법은 복잡한 저품질(Old-quality) 이미지에서도 일관된 색조를 유지하며 고품질 결과를 생성합니다.

- **Performance Highlights**: SSDiff는 GAN 기반 및 기존의 확산 모델을 이용한 방법보다 더 높은 인지 품질과 충실도로 성능을 구현합니다. 또한 300장의 다양한 저품질 구사진을 포함하는 VintageFace 벤치마크에서 종합 성능이 뛰어난 것으로 나타났습니다. 특히, SSDiff는 위치 기반의 스타일화 복원도 가능하여 더욱 효과적인 이미지 개선을 보여줍니다.



### DRL: Discriminative Representation Learning with Parallel Adapters for Class Incremental Learning (https://arxiv.org/abs/2510.12107)
Comments:
          13 pages, 7 figures

- **What's New**: 이번 연구에서는 비재현(Non-Rehearsal) 클래스 증가 학습(Class-Incremental Learning, CIL)의 과정에서의 다양한 도전 과제를 해결하기 위해 차별적 표현 학습(Discriminative Representation Learning, DRL) 프레임워크를 제안합니다. DRL은 경량 어댑터(lightweight adapter)를 학습해서 각 증가 단계에서 모델을 효과적으로 증대시킵니다. 이는 이전 모델의 특성을 상속하고, 전송 게이트(transfer gate)를 통해 매끄러운 표현 전환(smoth representation shift)을 보장합니다.

- **Technical Details**: DRL은 두 가지 핵심 구성 요소로 이루어져 있습니다: 점진적 병렬 어댑터(Incremental Parallel Adapter, IPA) 네트워크와 분리된 앵커 감독(Decoupled Anchor Supervision, DAS) 전략입니다. IPA 네트워크는 사전 학습된 모델(Pre-Trained Models, PTMs) 위에 구축되며, 각 단계에서 경량 어댑터를 학습하여 모델을 점진적으로 강화합니다. DAS는 긍정 샘플과 부정 샘플을 비교하여 제약 조건을 분리함으로써, 다양한 단계 간의 피쳐 공간(feature spaces)을 정합시킵니다.

- **Performance Highlights**: 여섯 개의 벤치마크 데이터셋에 대한 광범위한 실험을 통해 DRL의 능력을 입증했습니다. ImageNet-A에서 68.96%의 정확도를 달성하였으며, 이는 현재 상태의 최첨단(SOTA) 성능보다 3.62% 향상된 결과입니다. 또한 VTAB와 ObjectNet에서도 기존 SOTA 방법보다 각각 2.12% 및 1.85% 향상된 성능을 보여주었습니다.



### G4Splat: Geometry-Guided Gaussian Splatting with Generative Prior (https://arxiv.org/abs/2510.12099)
Comments:
          Project page: this https URL

- **What's New**: 이번 논문에서는 G4Splat이라는 새로운 방법을 제안하여 3D 장면 복원에서의 정확한 기하학적 정보를 활용합니다. 이는 평면 구조의 활용을 통해 신뢰할 수 있는 매트릭스 스케일 깊이 맵을 생성하여 관측된 영역과 미관측된 영역 모두에 정확한 기하학적 감독을 제공합니다. 또한, 생성 파이프라인 전반에 걸쳐 이러한 기하학적 지침을 통합하여 가시성 마스크 추정 및 다중 뷰 일관성 향상을 도모합니다.

- **Technical Details**: G4Splat 방법은 복잡한 MANHATTAN 세계 가정에 기반하여 3D 레이어와 깊이 맵을 생성하는 데 필요한 매트릭스 스케일 geometrical constraints를 제공합니다. 이는 관측된 영역과 미관측된 영역 모두에 대해 더 나은 기하학적 감독을 가능하게 하며, 특히 유사한 깊이 패턴을 가진 평면 구조를 통해 깊이에 대한 정확한 추정을 가능하게 합니다. 또한, 비디오 확산 모델을 활용하여 새로운 보기를 탐색하고, 글로벌 3D 평면을 사용하여 색상 감독을 조절함으로써 뷰 간의 갈등을 줄입니다.

- **Performance Highlights**: 실험 결과, G4Splat 방법은 Replica, ScanNet++, DeepBlending 데이터셋에서 기존의 방법들을 일관되게 초월하며, 특히 미관측된 영역에서의 기하학 및 외관 복원 능력이 두드러진다고 보고합니다. 또한, 단일 뷰 입력이나 비마주얼 비디오에서의 재구성을 지원하여 실제 응용에서 강력한 일반성을 지닙니다. 이 연구는 로봇공학, 체화된 AI(embodied AI) 등 다양한 분야에서의 실질적 용도를 향상시키는 잠재력을 보여줍니다.



### An Adaptive Edge-Guided Dual-Network Framework for Fast QR Code Motion Deblurring (https://arxiv.org/abs/2510.12098)
- **What's New**: 이 논문에서는 기존의 이미지 디블러링 기술이 QR 코드의 구조적 특성을 잘 활용하지 못하는 문제를 해결하기 위해 Edge-Guided Attention Block (EGAB)을 제안합니다. EGAB는 Transformer 아키텍처에 엣지 사전 정보를 명시적으로 통합하여 QR 코드 디코딩 성공률을 높이는 데 중점을 두고 있습니다. 또한, 이 모델을 바탕으로 Edge-Guided Restormer (EG-Restormer)와 Adaptive Dual-network (ADNet)를 개발하여, 다양한 블러(blur) 정도에 따라 최적의 네트워크를 선택하도록 설계하였습니다.

- **Technical Details**: EG-Restormer는 두 가지 주요 컴포넌트로 구성되어 있습니다: EGAB가 포함된 U자 형태의 네트워크와 경량 네트워크인 Lightweight and Efficient Network (LENet)입니다. ADNet은 Blur Severity-based Routing (BSR) 유닛을 통해 입력 QR 코드를 블러의 심각도에 따라 분류하여 적합한 네트워크로 라우팅합니다. 엣지 정보를 복원하는 데 중점을 두고 EGAB는 다중 방향 엣지 특성을 캡처하기 위해 여러 Sobel 연산자를 사용하고, 이를 통해 더욱 효과적인 디코딩이 가능하도록 합니다.

- **Performance Highlights**: 실험 결과, EG-Restormer와 ADNet은 QR 코드 디블러링 분야에서 최신 기술 수준의 성능을 기록하며 복원 품질과 디코딩 속도에서 뛰어난 경쟁력을 보였습니다. 특히, ADNet은 필요에 따라 EG-Restormer 또는 LENet 중 하나를 선택하여 효율적인 처리를 가능하게 합니다. 이로 인해 자원 제약이 있는 모바일 기기에서도 최적의 성능을 발휘할 수 있도록 설계되었습니다.



### IL3D: A Large-Scale Indoor Layout Dataset for LLM-Driven 3D Scene Generation (https://arxiv.org/abs/2510.12095)
Comments:
          9 pages main paper; 15 pages references and appendix

- **What's New**: 이번 연구에서는 대규모 데이터셋 IL3D를 소개합니다. IL3D는 27,816개의 실내 레이아웃과 29,215개의 높은 신뢰도를 가진 3D 오브젝트 자산을 포함하여, 실내 디자인에서의 다양한 고품질 교육 데이터의 필요를 충족시킵니다. 이 데이터셋은 자연어 주석을 포함하여, 비전-언어 작업을 위한 강력한 다중 모달 학습을 지원합니다.

- **Technical Details**: IL3D 데이터셋은 각종 실내 장면 생성 및 3D 인식 과제를 지원하기 위해 설계되었습니다. 이 데이터셋은 다양한 데이터 형식(예: semantic point clouds, 3D bounding boxes)으로 제공되어 여러 시각적 작업을 원활히 수행할 수 있도록 해줍니다. 또한, IL3D는 LLMs의 성능을 향상시키기 위해 세밀한 수준에서 자연어 설명을 제공합니다.

- **Performance Highlights**: 실험 결과 IL3D에 대한 감독된 세밀 조정(Supervised Fine-Tuning, SFT)이 LLM의 실내 레이아웃 생성 성능을 크게 향상시키는 것으로 나타났습니다. IL3D는 복잡한 시나리오에서 인식 모델의 일반화 능력을 높이기 위해 다양성과 기능성을 갖춘 실내 레이아웃을 제공합니다. 이를 통해 실내 장면 생성 및 인식 모델 훈련에 필요한 강력한 데이터 기반을 마련하게 됩니다.



### Playmate2: Training-Free Multi-Character Audio-Driven Animation via Diffusion Transformer with Reward Feedback (https://arxiv.org/abs/2510.12089)
- **What's New**: 이번 연구에서는 새로운 Diffusion Transformer (DiT) 프레임워크를 기반으로 오디오에 의해 구동되는 인간 비디오 애니메이션을 생성하는 방법을 제시합니다. 기존의 기술들이 직면한 문제인 입술 동기화, 긴 비디오 생성 시의 시간적 일관성, 그리고 다중 캐릭터 애니메이션 문제를 해결하기 위한 혁신적인 접근 방식을 도입했습니다. 특히, Mask Classifier-Free Guidance (Mask-CFG)라는 훈련이 필요 없는 방법을 통해 여러 캐릭터를 동시에 애니메이션화할 수 있는 능력을 갖추게 됩니다.

- **Technical Details**: 제안된 프레임워크는 LoRA 기반의 훈련 전략과 함께 position shift inference 방식을 사용하여 긴 기간의 비디오 생성을 지원하며, 이는 기본 모델의 성능을 유지하는 데 기여합니다. 부분 매개변수 업데이트와 보상 피드백을 결합하여 자연스러운 신체 동작과 정확한 입술 동기화를 구현합니다. 또한, Mask-CFG 접근 법은 특별한 데이터셋이나 모델 수정 없이도 실용적이고 비용 효율적인 다중 캐릭터 애니메이션을 가능하게 합니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안된 방법은 기존의 최첨단 접근 방식을 초월하는 성능을 보이며, 고품질의 시간적으로 일관된 다중 캐릭터 오디오 구동 비디오 생성을 간단하고 효율적이며 비용 효과적인 방식으로 달성합니다. 이러한 성능 향상은 디지털 인간 리서치 분야에서 중요한 이정표가 될 것으로 기대됩니다.



### A Review on Domain Adaption and Generative Adversarial Networks(GANs) (https://arxiv.org/abs/2510.12075)
- **What's New**: 이번 논문은 컴퓨터 비전에서의 중요한 문제인 데이터 부족을 해결하기 위해 Domain Adaptation(도메인 적응)에 초점을 맞추고 있습니다. 이미지 분류와 같은 분야에서는 신뢰할 수 있는 데이터 수집 방법이 필수적입니다. 이 논문은 이전의 벤치마크 결과와 비교 가능한 성과를 내기 위한 방법을 모색합니다.

- **Technical Details**: 논문에서는 특정 데이터셋에서 훈련된 모델을 사용하여 다른 도메인의 유사한 데이터에 대해 예측하는 방법론을 논의합니다. 구체적으로, 비행기 그림으로 훈련된 모델이 실제 비행기 이미지에 대한 예측을 수행하는 방법이 포함됩니다. 이러한 접근 방식은 라벨링된 데이터의 높은 비용 문제를 해결하는 데 도움을 줄 수 있습니다.

- **Performance Highlights**: 이 연구에서 제안된 Domain Adaptation 방법론은 다양한 도메인 간의 데이터 부족 문제를 해결하여 더 뛰어난 성능을 보여줄 것으로 기대됩니다. 제안된 방법을 통해 효율적인 이미지 분류와 같은 여러 컴퓨터 비전 작업에서 향상된 결과를 도출할 수 있을 것입니다.



### VIDMP3: Video Editing by Representing Motion with Pose and Position Priors (https://arxiv.org/abs/2510.12069)
- **What's New**: 새로운 접근 방식인 VidMP3를 소개합니다. 이 모델은 pose와 position priors를 활용하여 원본 비디오로부터 일반화된 모션 표현(generalized motion representation)을 학습합니다. VidMP3는 원래의 모션을 유지하면서 구조적(structural) 및 의미적(semantic) 변형이 가능한 새로운 비디오를 생성할 수 있습니다. 기존의 방법들이 직면했던 문제를 해결하기 위한 중요한 진전을 이룩했습니다.

- **Technical Details**: VidMP3는 MotionGuide 모듈을 통해 모션의 일반화된 표현을 학습합니다. MotionGuide는 3D 공간에서의 주체의 위치(position)와 내부 자세(pose)를 결합하여 모션의 변화를 설명하는 다양한 맵을 활용합니다. 이를 통해, Temporal self-attention layers에서 'Value'에 외부 지침을 주입함으로써 시간적 일관성을 높입니다. 이 모델은 다양한 T2I 확산 모델에 적용 가능하며, Stable-Diffusion-XL에 대한 확장성도 보여줍니다.

- **Performance Highlights**: VidMP3는 기존의 비디오 편집 방식과 비교했을 때 구조적 및 의미적 변화에 대한 유연성을 제공합니다. 실질적인 평가 결과, VidMP3는 기존 방법들에 비해 우수한 성능을 입증했습니다. 특히, 강한 구조와 의미적 변화가 있는 주체들을 강력하게 편집할 수 있는 능력을 보여주었습니다.



### APGNet: Adaptive Prior-Guided for Underwater Camouflaged Object Detection (https://arxiv.org/abs/2510.12056)
Comments:
          6 pages. accepted by ACM MM Asia 2025

- **What's New**: 본 논문에서는 수중 환경에서 위장된 물체를 탐지하기 위해 APGNet이라는 새로운 알고리즘을 제안합니다. APGNet은 시암 네트워크 아키텍처를 활용하고, 변형 가능한 합성곱(deformable convolution)을 포함한 새로운 사전 유도 메커니즘을 결합하여 탐지의 정확도를 향상시킵니다. 기존의 방법들이 수중 환경의 특성을 충분히 반영하지 못한 반면, APGNet은 MSRCR 알고리즘을 사용하여 데이터 증강 효과를 극대화합니다.

- **Technical Details**: APGNet은 다중 스케일 컨텍스트 정보를 포착하기 위해 확장 수용 영역 모듈(ERF)과 다중 스케일 점진적 디코더(MPD)를 설계했습니다. ERF는 비대칭 합성곱(asymmetric convolution)과 팽창 합성곱(dilated convolution)을 결합하여 넓은 수용 영역을 활용합니다. 더욱이, 자기 주도형 사전 유도 메커니즘이 하이레벨 기능에서 위치 및 경계 사전을 통합하여 정확한 분할을 유도합니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, APGNet은 두 개의 공개 MAS 데이터셋에서 15개의 최첨단 방법을 초월하는 성능을 보여주었습니다. 이는 수중 환경에서의 이미지 품질 저하와 물체 위장 문제를 효과적으로 해결하는 데 기여합니다. APGNet은 정확성과 일반화 모두에서 뛰어난 성능을 입증하며, 해양 동물 분할에 관한 새로운 기준을 제시합니다.



### Evaluating the Explainability of Vision Transformers in Medical Imaging (https://arxiv.org/abs/2510.12021)
Comments:
          Accepted at Workshop on Interpretability of Machine Intelligence in Medical Image Computing at MICCAI 2025

- **What's New**: 이번 연구에서는 의료 영상에서 모델의 결정 과정을 이해하는 것이 얼마나 중요한지를 강조하며, Vision Transformer (ViT) 아키텍처와 사전 학습 전략의 설명 가능성을 평가합니다. 연구에서는 DINO와 Grad-CAM의 조합이 가장 신뢰할 수 있고 지역화된 설명을 제공함을 발견했습니다. 이를 기반으로, ViT의 여러 아키텍처와 기술이 어떻게 해석 가능한 예측을 제공하는지에 대한 기초를 다지고자 합니다.

- **Technical Details**: ViT는 이미지를 고정 크기의 패치로 나눠 각 패치를 선형적으로 임베딩하여 글로벌 컨텍스트를 캡처하는 트랜스포머 인코더를 사용합니다. Grad-CAM과 Gradient Attention Rollout은 모델 결정 과정의 해석성을 평가하는 두 가지 방법으로 사용되며, Grad-CAM은 강력한 클래스-구별 히트맵을 생성하여 의학적 특징을 강조합니다. 이러한 방법들은 모델의 장단점을 분석하는 데 있어 중요한 도구입니다.

- **Performance Highlights**: DINO와 Grad-CAM의 조합은 데이터셋 전반에서 가장 충실하고 논리적인 설명을 제공하는 것으로 나타났습니다. 연구에서 ViT, DeiT, DINO 및 Swin Transformer 모델이 혈액 세포 및 유방 초음파 이미지 분류 작업에서 우수한 성능을 보였습니다. 이러한 결과는 AI 기반 의료 진단의 신뢰성과 해석 가능성을 높이는 데 기여할 것으로 기대됩니다.



### Prompt-Guided Spatial Understanding with RGB-D Transformers for Fine-Grained Object Relation Reasoning (https://arxiv.org/abs/2510.11996)
Comments:
          The paper was accepted at ICCV Conference 2025

- **What's New**: 이번 연구에서는 물리적 AI 공간 인텔리전스 창고 데이터셋을 활용하여 클러터(Clutter)가 많은 환경에서의 공간 추론을 위한 특별한 프레임워크를 소개합니다. 이 접근법은 마스크 차원을 바운딩 박스 좌표의 형태로 직접 입력 프롬프트에 포함시켜 모델이 객체의 기하학과 레이아웃에 대해 논리적으로 추론할 수 있도록 지원합니다. 또한, 네 가지 질문 범주에 대해 프레임워크를 세밀하게 조정하여 최종적으로 73.0606점을 기록하며 공개 리더보드에서 4위를 차지했습니다.

- **Technical Details**: 연구에서는 공간적 질문 응답 프레임워크를 제안하며, 물체 수준 기하학적 특징을 포함하는 프롬프트 증강 방법을 개발하였습니다. 이 방법은 바운딩 박스 좌표와 마스크 치수를 인코딩하여 공간 추론을 향상시키며, SpatialBot 아키텍처의 기능을 확장하여 창고 환경에 최적화된 성능을 발휘하도록 합니다. 이 모델은 물리적 AI 공간 인텔리전스 창고 데이터셋에서 세밀한 공간 질문에 답변할 수 있도록 조정되었습니다.

- **Performance Highlights**: 우리의 접근법은 구조화된 프롬프트 강화 및 표적 최적화의 효과를 입증하며, 실제 산업 환경에서의 공간 추론을 개선합니다. 연구 결과는 클러터가 많은 창고 레이아웃에 특화된 네 가지 공간 추론 작업에서 안정적인 성능을 보여줍니다. 이러한 성과는 공간적 질문에 대한 적합한 대답과 훈련 데이터와의 일관성을 높여, 산업 응용에서의 혼합 깊이 강화 비전-언어 시스템 적용 가능성을 제시합니다.



### PanoTPS-Net: Panoramic Room Layout Estimation via Thin Plate Spline Transformation (https://arxiv.org/abs/2510.11992)
- **What's New**: 이 논문에서는 단일 파노라마 이미지로부터 방의 3D 레이아웃을 정확하게 추정하기 위한 새로운 모델인 PanoTPS-Net을 제안합니다. 이 모델은 CNN(Convolutional Neural Network)을 사용하여 이미지에서 높은 수준의 특성을 추출하고, Thin Plate Spline(TPS) 공간 변환을 통합한 두 단계로 나누어진 구조를 띱니다. 이를 통해 PanoTPS-Net은 정육면체 및 비정육면체 레이아웃 모두를 효과적으로 일반화할 수 있는 능력을 갖추게 되었습니다.

- **Technical Details**: PanoTPS-Net은 입력 파노라마 이미지의 고급 특성을 추출한 후, 이 특성을 활용하여 TPS 변환 매개변수를 예측합니다. TPS 변환은 하나의 형태를 다른 형태로 부드럽고 유연하게 변형하는데 사용되는 수학적 기법으로, 이미지 왜곡 및 메쉬 변형에 널리 활용됩니다. 이 논문에서는 TPS 매개변수를 추정함으로써 방 레이아웃을 이미지 왜곡 문제로 재정의하였으며, 이 과정에서 고정밀의 고유 매개변수를 학습할 수 있는 격자 기반 공간 변환 네트워크(STN)를 적용하였습니다.

- **Performance Highlights**: 제안된 방법은 PanoContext, Stanford-2D3D, Matterport3DLayout 및 ZInD 데이터 세트에서 각각 3DIoU 값 85.49, 86.16, 81.76 및 91.98을 기록하며 높은 정확성을 보였습니다. 이러한 결과는 TPS 변환이 파노라마 이미지와의 호환성에서 유효함을 강조하며, 모델이 정육면체 및 비정육면체 방 레이아웃 추정 작업에서도 강건함을 나타냅니다. 또한, 코드가 공개되어 연구자들이 손쉽게 접근할 수 있도록 하여 향후 연구의 기반을 다지고 있습니다.



### Task-Specific Dual-Model Framework for Comprehensive Traffic Safety Video Description and Analysis (https://arxiv.org/abs/2510.11907)
Comments:
          This paper was accepted at ICCV 2025

- **What's New**: 이번 연구에서는 VideoLLaMA와 Qwen2.5-VL의 보완적인 강점을 활용하는 독특한 이중 모델 프레임워크를 제안하여 교통 안전 분석에 접근합니다. 이 접근은 자막 생성과 시각적 질문 응답(VQA) 작업의 훈련을 분리함으로써 각 모델이 보다 효과적으로 전문화되도록 합니다. 실험 결과에 따르면 VideoLLaMA는 시간적 추론에 매우 효과적이며, Qwen2.5-VL은 시각적 이해에서 우수한 VQA 정확도를 달성합니다.

- **Technical Details**: 제안된 방법은 두 개의 전문화된 모델을 이용하는 이중 작업 최적화 프레임워크로 구성되어 있습니다. Qwen2.5-VL은 VQA 작업을 수행하고, VideoLLaMA는 세밀한 자막 생성 작업을 수행합니다. 각 모델은 구체적인 작업에 대해 저순위 적응(LoRA)을 사용하여 독립적으로 최적화되며, 작업 간 간섭을 방지합니다. 이렇게 하여 두 가지 상보적인 작업을 처리하여 교통 사고에 대한 포괄적인 분석을 가능하게 합니다.

- **Performance Highlights**: 제안된 방법은 WTS 데이터셋에서 10위에 랭크되며, AI City 2025 Challenge Track 2에서 45.7572의 S2 점수를 달성했습니다. 또한, 자막 생성과 VQA의 별도 훈련 전략이 공동 훈련 방법보다 8.6% 더 높은 VQA 정확도를 보임을 입증했습니다. 이러한 성과는 복잡한 교통 사고 시나리오에서 세밀한 행동 패턴 분석과 환경 조건에 대한 깊은 이해를 제공합니다.



### MammoDINO: Anatomically Aware Self-Supervision for Mammographic Images (https://arxiv.org/abs/2510.11883)
Comments:
          5 pages

- **What's New**: MammoDINO는 유방촬영(Mammography)을 위한 새로운 SSL(Self-supervised Learning) 프레임워크로, 140만 개의 유방촬영 이미지로 사전 훈련되었습니다. 이는 일반적인 도메인에서 비전 인코더 훈련을 혁신적으로 변화시킨 SSL을 의료 이미징 분야에 적용하고자 하는 시도입니다.

- **Technical Details**: MammoDINO는 이미지 수준(image-level) 및 패치 수준(patch-level) 감독을 위한 유방 조직 인식 데이터 증강 샘플러(data augmentation sampler)를 도입했습니다. 또한, 3D 디지털 유방 단층 촬영(DBT) 구조를 2D 사전 훈련에 활용하는 교차 슬라이스 대조 학습(cross-slice contrastive learning) 목표를 설정하여 임상적으로 의미 있는 특징을 포착하고자 합니다.

- **Performance Highlights**: MammoDINO는 여러 유방암 검진 작업에서 최첨단(state-of-the-art) 성능을 달성했으며, 다섯 개의 기준 데이터셋(benchmark datasets)에서 잘 일반화되고 있습니다. 이 프레임워크는 주석이 필요 없는 스케일링 가능한 기반을 제공하여 유방촬영 진단에 도움을 주며 방사선사의 업무 부담을 줄이고 진단 효율성을 향상시킵니다.



### Data or Language Supervision: What Makes CLIP Better than DINO? (https://arxiv.org/abs/2510.11835)
Comments:
          EMNLP 2025 Findings

- **What's New**: 이 논문에서는 CLIP와 DINO를 동일한 환경에서 사전 학습시켜 두 모델 간의 성능 차이가 언어 감독(language supervision) 또는 데이터셋 크기에서 비롯되는지를 조사합니다. 연구 결과, CLIP는 고차원 의미(high-level semantics)를 잘 포착하고, DINO는 저차원 시각적 특성(low-level features)에 더 민감하다는 것을 보여줍니다. 이러한 차이는 VLM(vision-language models)의 성능에도 영향을 미쳐, CLIP는 텍스트 밀집 작업에서 두드러진 우위를 보입니다.

- **Technical Details**: CLIP와 DINO는 동일한 아키텍처(ViT-B/16), 데이터셋(DataComp의 1천만 이미지 서브셋), 훈련 구성으로 20 에포크 동안 학습되었습니다. 결과적으로 두 모델은 유사한 ImageNet 정확도를 달성하였고(CLP: 65.8%, DINO: 66.4%), 이는 공정한 비교 기반을 제공하였습니다. 훈련 후, 두 모델의 성능은 이미지 분류 벤치마크에서 유사하게 평가되었지만, CLIP은 Stanford Cars 및 CUB와 같은 세부 분류 작업에서 DINO보다 나은 성과를 보였습니다.

- **Performance Highlights**: LLaVA 모델에 통합된 CLIP은 텍스트 중심의 작업에서 두드러진 성능 향상(+7.5%)을 기록하며, DINO보다 대체로 우수한 성능을 보였습니다. 두 모델은 일반적인 VQA(visual question answering) 작업에서는 유사한 성능을 보였지만, CLIP은 OCR 기반 벤치마크에서 더 뛰어난 성과를 나타내며, text-heavy 시각적 이해 작업에서의 언어 감독의 중요성을 강조했습니다.



### Enhancing the Quality of 3D Lunar Maps Using JAXA's Kaguya Imagery (https://arxiv.org/abs/2510.11817)
Comments:
          Presented at IEEE SMC 2025

- **What's New**: 본 논문은 NASA의 Endurance 미션과 같은 장거리 탐사를 위한 고품질 3D 달 지도의 필요성을 강조합니다. Kaguya TC (Terrain Camera) 이미지의 잠재적인 문제점을 해결하기 위해, 압축으로 인한 노이즈를 줄여 3D 지도 품질을 향상시키는 방법을 제안합니다. 특히, 압축된 이미지로부터 파생된 disparity 이미지의 잔여 노이즈를 줄임으로써 terrain data의 안전성과 신뢰성을 높이는 데 중점을 두고 있습니다.

- **Technical Details**: Kaguya TC 이미지는 JPEG 압축으로 인해 발생하는 고도 불일치를 가지고 있으며, 이는 stereo matching 오류와 JPEG 압축 아티팩트로 인한 것입니다. 이 논문에서는 disparity 이미지의 잔여 노이즈를 줄이는 두 가지 심층 학습 접근법인 IGEV++와 conditional diffusion model을 사용하여 개선 방안을 제안하고 있습니다. 기존의 압축 노이즈 제거 방법들은 일반적으로 3D 재구성에 적합하지 않으며, 본 연구에서 제안하는 방법은 이러한 압축으로 인한 오류를 처음으로 개선하는 접근을 목표로 하고 있습니다.

- **Performance Highlights**: 실험 결과, 제안한 방법이 노이즈를 효과적으로 감소시켜 고도에 대한 정확성을 향상시킴을 보여줍니다. 압축된 이미지로부터 생성된 3D 지도에서 약 20미터의 고도 노이즈가 생성된다는 점을 강조하며, 이는 탐사 로버 내비게이션에 대한 잠재적 위험을 나타냅니다. 최종적으로, 제안된 접근 방식은 미래의 달 탐사 미션에서 지형 데이터의 안전성과 신뢰성을 제공하는 데 중요한 영향을 미칠 것입니다.



### Omni-Captioner: Data Pipeline, Models, and Benchmark for Omni Detailed Perception (https://arxiv.org/abs/2510.12720)
Comments:
this https URL

- **What's New**: 최근 오디오 및 비디오 신호를 병렬로 처리할 수 있는 Omni Language Models (OLMs)에 대한 연구가 진행되고 있습니다. 이러한 모델은 다중 양식 정보에 대한 세밀한 인식(fine-grained perception)에서 더 나은 이해와 추론을 가능하게 합니다. 그러나 기존의 OLM들이 세부 정보를 포착하고 설명하는 데 한계가 있다는 점이 지적되었습니다.

- **Technical Details**: 이 연구에서는 OLM의 데이터 파이프라인(data pipeline), 모델, 벤치마크에서의 세밀한 인식 시스템을 종합적으로 조사합니다. 특히, "co-growth"라는 내재된 특징을 발견하여, 세부 정보와 환각(hallucination) 사이의 관계를 탐구합니다. Omni-Detective라는 새로운 데이터 생성 파이프라인을 제안하여 이를 해결하고, 이 시스템은 최소한의 환각을 가지면서도 고품질의 다중 양식 데이터를 자율적으로 생성합니다.

- **Performance Highlights**: Omni-Detective로 생성된 데이터를 기반으로, 주목할만한 성능을 보인 두 가지 캡션 모델, Audio-Captioner와 Omni-Captioner가 훈련되었습니다. Audio-Captioner는 모든 오픈 소스 모델 중 MMAU 및 MMAR에서 최고 성능을 달성했습니다. Omni-Captioner는 VDC에서 새로운 최첨단 성능을 기록하였고, 비디오-SALMONN 2 테스트 세트에서 세부 정보와 환각 간의 최적의 균형을 이룹니다.



### SAIL-Embedding Technical Report: Omni-modal Embedding Foundation Mod (https://arxiv.org/abs/2510.12709)
Comments:
          Technical Report

- **What's New**: 이번 논문은 SAIL-Embedding이라는 다중 모드 임베딩 모델을 제안하며, 이는 다양한 모드 지원 및 훈련 안정성 문제를 해결하기 위한 맞춤형 훈련 전략과 구조 설계를 통해 발전하고 있음을 강조합니다. 기존 모델이 가진 제한된 모드 지원 문제를 극복하기 위해, SAIL-Embedding은 시각, 텍스트, 오디오 입력의 임의의 조합을 처리할 수 있도록 설계되었습니다. 이를 통해 실제 비즈니스 요구에 맞춘 다차원 임베딩 벡터를 생성할 수 있습니다.

- **Technical Details**: SAIL-Embedding은 콘텐츠 인지적(progressive) 훈련과 협업 인지적(collaboration-aware) 추천 향상 훈련을 통해 모델을 훈련시키며, 각각의 단계에서 다양한 데이터 자원으로부터 얻은 지식을 활용합니다. 이 과정은 서로 다른 하위 작업에 대한 조정 능력 향상과 미지의 시나리오를 처리하기 위한 일반화 능력을 제공합니다. 또한, 동적 하드 네거티브 마이닝과 적응형 다중 출처 데이터 균형 조정 방법을 도입하여 훈련의 유연성과 일반화 능력을 강화합니다.

- **Performance Highlights**: SAIL-Embedding은 여러 벤치마크 데이터셋을 통한 실험 결과에서 다양한 하위 작업에서 최신 기술(STATE-OF-THE-ART) 성능을 달성하였음을 입증했습니다. 특히, Douyin 추천 시스템에서 온라인 실험을 통해 7일과 14일의 생애 주기(Lifetime) 증가를 각 +0.158% 및 +0.144%로 나타내는 등 실질적인 성과를 거두었습니다. 또한, SAIL-Embedding은 차가운 시작(cold-start) 상황에서도 추천 모델의 AUC를 0.1% 향상시키며, 각 단계에서 ∼0.03%의 LT 이득을 보여주었습니다.



### DiffEM: Learning from Corrupted Data with Diffusion Models via Expectation Maximization (https://arxiv.org/abs/2510.12691)
- **What's New**: 이 연구에서는 손상된 데이터를 이용하여 확산 모델(diffusion models)을 학습하는 새로운 방법인 DiffEM을 제안합니다. 이 방법은 Expectation-Maximization (EM) 알고리즘을 활용하여 손상된 관측값으로부터 깨끗한 데이터를 재구성하고, 이를 통해 확산 모델을 정제하는 과정을 포함합니다. 또한, DiffEM은 이론적으로 수렴 보장(monotonic convergence guarantees)을 제공하여 향후 연구의 기초를 다지고 있습니다.

- **Technical Details**: DiffEM의 핵심 아이디어는 사전 분포(prior distribution)를 학습하는 대신 조건부 확산 모델(conditional diffusion model)을 사용하여 후방 분포(posterior distribution)를 직접 모델링하는 것입니다. 이 접근 방식은 특정한 후방 샘플링 방식(approximate posterior sampling schemes)과 독립적이며, 어떤 손상 채널(corruption channel)도 처리할 수 있는 장점을 가지고 있습니다. 이 모델은 정량화하기 어려운 구조적 가정을 피하고, 재구성된 데이터로 모델을 정제하는 과정에서 발생하는 오류에 대한 이론적 분석도 포함되어 있습니다.

- **Performance Highlights**: DiffEM의 효과는 다양한 이미지 재구성 작업을 통해 실험적으로 입증되었습니다. 연구팀은 CIFAR-10과 CelebA와 같은 데이터 세트를 활용하여 저차원 매니폴드 학습(low-dimensional manifold learning)과 같은 다양한 유형의 손상에 대해 평가를 진행했습니다. 이 연구는 손상된 관측값만으로도 효과적으로 확산 모델을 훈련할 수 있는 길을 열어줄 것으로 기대됩니다.



### VISaGE: Understanding Visual Generics and Exceptions (https://arxiv.org/abs/2510.12548)
Comments:
          EMNLP 2025

- **What's New**: 이 논문에서는 VISaGE라는 새로운 평가 데이터셋을 소개합니다. 이 데이터셋은 전형적인(typical) 이미지와 예외적인(exceptional) 이미지를 모두 포함하고 있어, 특히 VLMs(비전-언어 모델)에서 일반화된 개념 표현(generalized conceptual representations)과 개별 인스턴스(instance) 간의 긴장을 탐구합니다. 연구자들은 이 데이터셋을 통해 VLM이 이 두 가지 표현 간의 관계를 어떻게 조정하는지를 이해하고자 하였습니다.

- **Technical Details**: VLM은 텍스트와 비주얼 입력이 관련이 있다고 가정하는 실용적(pragmatic) 선험(prior)과, 개념 표현이 일반적으로 특정 카테고리 인스턴스에 대해 진리(truth)를 나타낸다고 가정하는 의미적(semantic) 선험 간의 충돌을 분석합니다. VISaGE에서는 이 두 가지 선험이 서로 충돌할 수 있는 다양한 예외적인 이미지 조건을 탐구하며 실험을 진행합니다. 논문은 VLM의 개념 표현이 전형적인 인스턴스에만 시각적으로 기반하고 있으며 카테고리 내의 변동성을 충분히 인식하지 못한다고 주장합니다.

- **Performance Highlights**: 실험 결과, VLM은 비주얼 기준이 텍스트와 일치하지 않을 때 개념 이해에 방해가 발생하며, 개별 인스턴스에 대해 일반적인 개념과 특정 인스턴스를 구분하지 못하는 경향이 있음을 보여줍니다. 또한 대부분의 모델이 예외적인 속성을 인식하는 과제에 대해 의미적 편향을 보였지만, 이러한 효과는 상대적으로 약하다는 점도 발견하였습니다. 즉, VLM이 개념적 속성과 인스턴스 속성 간의 상호작용에 있어 제약을 받고 있음을 시사합니다.



### Fast Visuomotor Policy for Robotic Manipulation (https://arxiv.org/abs/2510.12483)
- **What's New**: 이번 논문에서는 에너지 정책(Energy Policy)이라는 새로운 정책 프레임워크를 제안합니다. 이 프레임워크는 고주파 로봇 작업과 자원 제약 시스템을 위해 설계되어, 기존 로봇 정책들과는 달리 멀티모달 액션을 단일 전방 패스로 예측할 수 있습니다. 이를 통해 고속으로 고정밀 조작이 가능해지며, 에너지 점수를 학습 목표로 삼아 멀티모달 액션 모델링을 돕습니다.

- **Technical Details**: 에너지 정책 프레임워크의 두 가지 핵심 구성 요소는 에너지 점수와 에너지 MLP(다층 퍼셉트론)입니다. 에너지 점수는 예측된 액션과 실제 값을 비교하여 분포 차이를 최소화하는 데 사용되며, 에너지 MLP는 액션 예측을 간단하고 효율적으로 구현합니다. 이 두 구성 요소의 결합을 통해 모델은 단일 전방 패스에서 연속 액션을 샘플링할 수 있습니다.

- **Performance Highlights**: 다양한 로봇 조작 벤치마크(예: Robomimic, MimicGen) 테스트에서 에너지 정책은 높은 작업 성공률과 빠른 추론 속도를 기록했습니다. 특히, 모든 벤치마크에서 기존의 CARP보다 2.3배에서 7배 더 빠른 추론 속도를 자랑하며 실시간 로봇 응용 프로그램에 적합하다는 것을 입증했습니다.



### A Function Centric Perspective On Flat and Sharp Minima (https://arxiv.org/abs/2510.12451)
Comments:
          26 pages, 26 tables, 63 figures, pre-print

- **What's New**: 이 논문은 머신러닝에서 일반화 성능을 향상시킬 수 있는 방식으로 평균적인 미니마의 역할을 재고합니다. 연구진은 샤프니스(sharpness)가 단순히 일반화 실패의 지표가 아니라 함수에 따른 성질로 이해되어야 한다고 주장합니다. 정규화 기술을 사용하여 훈련한 모델들이 예기치 않게 더 날카로운 미니마로 수렴할 수 있으며, 이들이 일반화 및 안전성 측면에서 더 나은 성능을 보일 수 있다는 것이 핵심 내용입니다.

- **Technical Details**: 연구진은 다양한 데이터셋(CIFAR, TinyImageNet)과 모델(ResNet, VGG, ViT)을 사용하여 매개변수 불변 샤프니스 측정법을 적용해 연구를 수행했습니다. 설정된 기본 모델과 정규화 기법(SAM, weight decay, data augmentation)을 적용한 모델을 비교하여 성능을 평가했습니다. 이 결과, 정규화를 통해 훈련된 모델들이 대개 더 날카로운 미니마로 수렴함에도 불구하고, 일반화 및 안전 관련 지표에서 더 나은 성능을 나타낸다는 것을 발견했습니다.

- **Performance Highlights**: 정규화된 모델들이 날카로운(minima) 미니마에서 성능을 발휘하는 것이 확인되었습니다. 이는 이전의 일반화 기준인 평탄함(flatness)과는 상충되는 결과입니다. 각종 안전 및 일반화 지표에서 더 나은 성능을 보였으며, 이는 샤프니스가 단순히 성능의 지표가 아니라 훨씬 더 복잡한 함수의 복잡성과 추론적 편향(inductive bias)을 반영하는 것으로 해석될 수 있음을 보여줍니다.



### Tensor Completion via Monotone Inclusion: Generalized Low-Rank Priors Meet Deep Denoisers (https://arxiv.org/abs/2510.12425)
Comments:
          22 pages, 5 figures

- **What's New**: 이번 연구에서는 다차원 데이터에서의 누락된 항목을 추정하는 새로운 텐서 완성(tensor completion) 프레임워크를 제안합니다. 이 프레임워크는 monotone inclusion 패러다임을 기반으로 하여 일반화된 저랭크 프라이어(generalized low-rank priors)와 깊은 유사 수축형(noise-reducing) 디노이저(deep pseudo contractive denoisers)를 통합합니다. GTCTV DPC 알고리즘은 글로벌 수렴성(global convergence)을 rigorously하게 확립하며, 기존의 방법들에 비해 뛰어난 성능을 보입니다.

- **Technical Details**: 제안된 방법은 Davis-Yin 분할(Davis Yin splitting) 기법을 기초로 한 GTCTV DPC 알고리즘을 포함하며, 이 알고리즘은 형태가 가능한 연속 수렴 방식으로 운용됩니다. 텐서 완성 문제는 식 𝒴=𝒫Ω(𝒳)로 정의되며, 𝒳는 완전한 텐서, Ω는 관측된 항목의 인덱스 집합을 나타냅니다. GTCTV-DPC에서는 깊은 디노이저가 일반 운영자로 다루어지며, 이는 근사 맵(proximal mappings)으로서의 제한 없이 이루어집니다.

- **Performance Highlights**: GTCTV DPC는 정량적 지표와 시각적 품질 모두에서 기존의 방법들보다 일관되게 우수한 성능을 보였습니다. 특히, 낮은 샘플링 비율(low sampling rates)에서 두드러진 성능 향상이 있었습니다. 광범위한 실험을 통해 제안된 방법의 유효성이 입증되었으며, 이로 인해 텐서 완성 작업에서 새로운 가능 주도하는 방향성을 시사합니다.



### MAPS: Masked Attribution-based Probing of Strategies- A computational framework to align human and model explanations (https://arxiv.org/abs/2510.12141)
- **What's New**: 이번 연구에서는 MAPS (Masked Attribution-based Probing of Strategies)를 소개합니다. MAPS는 인공지능 신경망(ANN)에서 유도된 설명이 인간의 시각을 설명할 수 있는지를 테스트하는 행동적으로 검증된 계산 도구입니다. 이 도구는 주의 맵(attribution map)을 설명 마스크 이미지(EMI)로 변환하여 인간의 정확도를 비교하는 방법을 제시합니다.

- **Technical Details**: MAPS는 제한된 픽셀 예산을 가진 최소한의 이미지에서 인간이 보여주는 정확도를 원래 자극의 정확도와 비교합니다. 이 방법은 경쟁하는 ANN 해석법들 사이에서 평가하고 선택하는 원칙적인 방법을 제공합니다. 또한, 모델의 주의 맵에서 계산된 실제 유사성을 신뢰성 있게 회복하여 어떤 설명 방법이 모델의 전략을 가장 잘 포착하는지를 결정합니다.

- **Performance Highlights**: MAPS는 인간과 원숭이에서 ANN 설명 조합을 식별하여 생물학적 시각과 가장 밀접하게 일치하는 설명을 제공합니다. 이 도구는 Bubble 마스크의 행동적 유효성을 달성하면서도 훨씬 적은 행동 시험만을 요구합니다. MAPS는 모델 주의와 원본 이미지의 적당한 행동 데이터만 필요로 하여 포괄적인 심리물리학을 피하고, 인간 행동, 신경 활동 및 모델 결정을 일반적인 기준으로 연결할 수 있는 확장 가능한 도구를 제공합니다.



### Gaussian Semantic Field for One-shot LiDAR Global Localization (https://arxiv.org/abs/2510.12101)
- **What's New**: 이 논문에서는 경량 삼중 레이어 구조의 장면 그래프를 기반으로 한 한 번의 LiDAR 전역 위치 추적 알고리즘을 제안합니다. 기존의 방법들이 지리적 정보에만 의존하는 데 반해, 본 연구는 Gaussian 과정에서 학습된 연속 함수를 통해 의미적 배포를 모델링하여 문제를 완화합니다. 이 접근 방식은 기하학적 변화와 의미적 기울기에서 보다 세밀한 정보를 제공합니다.

- **Technical Details**: 제안된 방법은 쿼리 LiDAR 스캔과 참조 점 구름 맵 간의 전역 점 구름 등록 문제로 모델화됩니다. 경량의 인스턴스 수준 삼각형 설명자로 조정된 쿼리 포인트 구름은 편리한 계층화를 가능하게 합니다. 3D 장면 그래프 내부에 Gaussian semantic field(GSF)를 중간 레이어로 삽입하여 공간-의미적 배포를 연속적으로 모델링하고 비효율적인 전통적 레이블의 문제를 해결합니다.

- **Performance Highlights**: 다양한 공개 데이터 세트에서 실시한 실험을 통해 제안된 Outram-GSF 알고리즘은 최신 기법에 비해 뛰어난 성능을 보여줍니다. 특히, 반복적인 의미 구조를 가진 환경에서 높은 강인성을 보이며, 기존 기술보다 더 나은 전역 위치 추적을 가능하게 합니다. 이는 보다 정밀한 의미적 정보를 통해 정확한 대응을 확립할 수 있기 때문입니다.



### Your VAR Model is Secretly an Efficient and Explainable Generative Classifier (https://arxiv.org/abs/2510.12060)
- **What's New**: 본 연구는 기존의 확산 기반(‘diffusion-based’) 모델 대신 최근 발전된 시각 자율회귀(‘visual autoregressive’, VAR) 모델링을 활용한 새로운 생성 분류기(‘generative classifier’)를 제안합니다. 특히 제안된 Adaptive VAR Classifier$^+$ (A-VARC$^+$)는 정확성과 추론 속도 간의 우수한 균형을 달성하여, 실제 적용성을 크게 개선합니다. 또한, VAR 기반 방법론은 확산 기반 방법들과 근본적으로 다른 성질을 보이는데, 이는 시각적 설명 가능성과 ‘catastrophic forgetting’에 대한 저항력을 포함합니다.

- **Technical Details**: A-VARC는 정확도를 향상시키기 위해 두 가지 새로운 기술을 통합합니다: 우려한 추정치를 통해 정확도를 높이는 likelihood smoothing과 모델의 다중 스케일 아키텍처를 활용하여 추론 속도를 가속화하는 부분 스케일 후보 가지치기(‘partial-scale candidate pruning’)입니다. 이러한 접근법은 쉽게 조정 가능한 프레임워크를 만들어 내어, 기존 naive VARC보다 상당한 성능 향상을 이끌어냅니다. A-VARC+는 최근 제안된 Condition Contrastive Alignment (CCA) 기법을 활용하여 세밀하게 조정된 개선된 버전입니다.

- **Performance Highlights**: A-VARC+는 ImageNet-100 데이터셋에서 확산 기반 모델과 비교하여 1% 미만의 성능 저하로 160배의 속도 향상을 기록합니다. 이는 생성 분류기의 계산 부담을 상당히 경감시켜 주며 실제 적용 가능성을 높입니다. 특히, VAR 기반 방법론은 개별 텍스트와 타겟 레이블 간의 연관성을 포착할 수 있는 시각적 설명 능력을 제공하며, 기존의 ‘discriminative classifiers’에 비해 클래스 증가 학습에서 내재적 내성을 보입니다.



### MosaicDiff: Training-free Structural Pruning for Diffusion Model Acceleration Reflecting Pretraining Dynamics (https://arxiv.org/abs/2510.11962)
Comments:
          International Conference on Computer Vision, ICCV 2025

- **What's New**: 이번 연구에서는 MosaicDiff라는 새로운 프레임워크를 소개하여, diffusion 모델의 pretraining(사전 훈련) 동학을 포스트 트레이닝 샘플링 가속화와 일치시키는 방법을 제안합니다. 그동안 다양한 전이 학습 가속화 방법이 존재했으나, 이러한 방법들이 학습 속도의 변화를 간과해왔다는 점에서 이 연구는 특별합니다. 새로운 적응형 프루닝 메커니즘을 통해, 빠르게 학습되는 중간 단계에서는 보수적인 프루닝이 필요하다는 것을 밝혔습니다.

- **Technical Details**: MosaicDiff의 주요 구성 요소는 Divide(나누기), Prune(프루닝), Conquer(정복 단계)로 나뉘며, 이는 diffusion 모델의 추론 경로를 세분화하고 SNR(신호 대 잡음 비율) 기반으로 프루닝을 진행합니다. 프루닝 과정은 모델의 학습 궤적에 따라 조정되며, 빠른 학습 단계에서는 보수적인 접근이 필요하고, 느린 학습 단계에서는 보다 공격적인 프루닝이 허용됩니다. 이 연구는 기존의 훈련이 필요 없는 가속화 방법에서는 설명되지 않았던 다양한 학습 속도 변화를 명시적으로 통합합니다.

- **Performance Highlights**: 대규모 실험 결과, MosaicDiff는 기존의 최신 방법들보다 월등한 성능을 보여줍니다. 특히 높은 프루닝 비율에서 생성 및 가속 성능이 뛰어난 결과를 보입니다. 이는 분명히 기존의 훈련 없이 가속화된 방법들보다 큰 개선을 나타내며, diffusion 모델의 가속화에 새로운 관점을 제공합니다.



### GS-Verse: Mesh-based Gaussian Splatting for Physics-aware Interaction in Virtual Reality (https://arxiv.org/abs/2510.11878)
- **What's New**: 이 논문에서는 GS-Verse라는 새로운 접근 방식을 소개하며, 이는 물리적 시뮬레이션을 위한 객체의 메쉬를 직접 사용하는 것을 특징으로 한다. 기존의 VR에서 사용된 물리 기반 접근법은 간소화된 기하학적 표현(예: tetrahedral cages)에 의존해, 정확한 물리적 시뮬레이션을 저해할 수 있는 단점이 있었다. GS-Verse는 Gaussian Splatting과 메쉬 표현을 통합함으로써, 더 현실적인 형태의 변형과 상호작용을 가능하게 한다.

- **Technical Details**: GS-Verse는 기존 3D 메쉬 자산을 활용하여, 보다 정밀한 표면 근사(surface approximation)를 실현한다. 이를 통해 자연스럽고 사실적인 3D 형태의 변형을 가능하게 하며, 다양한 물리 엔진과 호환되어 개발자에게 유연성을 제공한다. 이 시스템은 물리 엔진에 의존하지 않으며, 이는 시스템의 통합 및 지원을 용이하게 하여 개발 과정의 효율성을 높인다.

- **Performance Highlights**: 실험 결과, GS-Verse는 물리 기반 변형 조작에서 통계적으로 유의미한 성과를 보여주며, 비틀기 및 흔들기와 같은 다양한 물리 기반 조작에서도 높은 일관성과 신뢰성을 제공한다. 18명의 참여자를 대상으로 진행된 사용자 연구를 통해 GS-Verse의 성능이 기존 기술에 비해 안정적이고 우수하다는 것을 입증하였다. 이러한 성능은 다양한 상호작용 환경에서 일관되게 나타나, 기존 방식에 대한 실질적인 대안으로 자리잡을 가능성을 보여준다.



### Audio-Guided Visual Perception for Audio-Visual Navigation (https://arxiv.org/abs/2510.11760)
Comments:
          Main paper (6 pages). Accepted for publication by International Conference on Virtual Reality and Visualization 2025 (ICVRV 2025)

- **What's New**: 이 논문에서는 AGVP(Audio-Guided Visual Perception) 프레임워크를 제안하여 기존의 오디오-비주얼 내비게이션 방법의 한계를 극복하고자 합니다. 기존 방법들은 시각적 특징과 청각적 신호 간의 명시적 정렬이 부족하여 성공률이 감소하는 문제가 있었으나, AGVP는 이를 개선하여 청각적인 정보를 공간적 지침으로 변환합니다. 이를 통해 새로운 소음원의 탐색 시에도 효율적으로 목표를 향해 나아갈 수 있는 길잡이를 제공하게 됩니다.

- **Technical Details**: AGVP 프레임워크는 관찰, 관찰 인코딩, 그리고 PPO(Proximal Policy Optimization)를 기반으로 한 정책 업데이트의 세 단계로 구성됩니다. 이 과정을 통해 에이전트는 시각적 및 청각적 입력을 지속적으로 탐색하며 필요한 정보를 인코딩합니다. 특히, self-attention 메커니즘을 사용하여 음향 신호의 글로벌 맥락을 구축하고 이를 비주얼 특징 지도에 통합하여 청각 신호와 관련된 시각적 영역을 명확히 정렬합니다.

- **Performance Highlights**: 실험 결과 AGVP는 내비게이션 효율성과 강인성에서 향상된 성능을 보였으며, 이전에 들리지 않았던 소음에 대해서도 뛰어난 시나리오 일반화 능력을 입증했습니다. 기존의 최고 수준의 기술과 비교하여 AGVP는 새로운 소리에 대한 탐색 성공률을 크게 향상시켰으며, 위험을 최소화하면서 목표를 향한 효율적 이동을 가능케 합니다.



### SeeingSounds: Learning Audio-to-Visual Alignment via Tex (https://arxiv.org/abs/2510.11738)
Comments:
          accepted to ACM Multimedia Asia 2025

- **What's New**: SeeingSounds는 오디오와 이미지 생성 간의 상호 연계를 활용하는 경량 모듈형 프레임워크로, 짝을 이루는 오디오-비주얼 데이터나 비주얼 생성 모델을 학습할 필요 없이 동작합니다. 이 방법은 오디오를 텍스트 대체물로 다루기보다는, 오디오와 비주얼 도메인 간의 맥락 기반 정렬을 통해 두 가지 주요 경로를 활용하여 세 가지 입력 모달리티(음향, 언어, 비주얼)를 연결합니다.

- **Technical Details**: SeeingSounds의 아키텍처는 고정된 언어 인코더를 통해 오디오를 의미론적 언어 공간으로 투영하고, 비전-언어 모델을 사용하여 시각적 도메인에 맥락적으로 기초를 둡니다. 이 프레임워크는 정량적 및 정성적으로 매우 세밀한 조절이 가능하도록 하여 오디오 변환이 의미적으로 일관된 텍스트 프롬프트로 변환되어 시각적 결과를 안내할 수 있습니다.

- **Performance Highlights**: 다양한 벤치마크를 통해 SeeingSounds가 제로샷 및 감독 환경 모두에서 기존 방법보다 우수한 성능을 보여주며, 통제 가능한 오디오-비주얼 생성에 있어 새로운 최첨단 기술을 확립했습니다. 우리의 결과는 오디오에서 장면으로의 생성에서 상태-of-the-art 성능을 보여주며, 간단한 언어-비주얼-오디오 정렬 전략의 효과를 확인합니다.



### HccePose(BF): Predicting Front & Back Surfaces to Construct Ultra-Dense 2D-3D Correspondences for Pose Estimation (https://arxiv.org/abs/2510.10177)
Comments:
          International Conference on Computer Vision, ICCV 2025 (Highlight) this https URL

- **What's New**: 이번 연구는 포즈 추정을 위해 객체의 앞면과 뒷면 표면의 3D 좌표를 동시에 예측하는 신경망 기반 방법을 제안합니다. 기존 방식들이 주로 앞면 표면에 집중했던 것과 달리, 이 방법은 전체 표면과 내부를 활용하여 초고밀도 2D-3D 대응관계를 생성하여 정확도를 높입니다. 또한, 고급 계층적 연속 좌표 인코딩(HCCE) 방식을 도입하여 예측의 효율성을 강화하고 있습니다.

- **Technical Details**: 제안된 방법은 먼저 신경망을 통해 객체의 앞면과 뒷면 표면의 3D 좌표를 예측합니다. 이 때 HCCE를 사용하여 각 표면의 xy, xz, yz 성분을 분리하여 인코딩하고, 계층 학습을 통해 훈련 안정성을 향상시킵니다. 또한 전체 프로세스를 통해 RANSAC-PnP 알고리즘을 활용하여 객체의 포즈를 추정하는 방식을 사용합니다.

- **Performance Highlights**: 실험 결과, 제안된 접근법은 BOP 코어 데이터셋에서 기존 최첨단(SOTA) 방법 대비 2.4% 개선된 BOP 점수를 달성했습니다. RGB에서 훈련하였으나 RGB-D 데이터로 테스트할 경우 4.7% 향상된 성능을 보였고, 2D 분할 작업에서도 가장 뛰어난 기존 접근 방법을 3.7% 초과하며 효과성을 입증했습니다.



New uploads on arXiv(cs.AI)

### Ax-Prover: A Deep Reasoning Agentic Framework for Theorem Proving in Mathematics and Quantum Physics (https://arxiv.org/abs/2510.12787)
- **What's New**: Ax-Prover는 Lean에서 자동 정리 증명을 위한 다중 에이전트 시스템으로, 다양한 과학 분야의 문제를 해결할 수 있도록 설계되었습니다. 이 시스템은 스스로 작업을 수행하거나 인간 전문가와 협력하여 문제를 해결할 수 있는 능력을 갖추고 있습니다. 특히, Ax-Prover는 비즈니스 및 학계에서 신뢰할 수 있는 추론을 위한 Large Language Models (LLMs)와의 결합을 통해 다양한 과학적 문제를 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: Ax-Prover는 Model Context Protocol (MCP)을 활용하여 LLM을 Lean 도구와 통합하는 새로운 에이전트 워크플로우를 구성합니다. 이 시스템은 수학적 정리를 분석하고, 증명 개요를 제안하며, Lean 코드로 단계별 증명을 생성하는 기능을 제공합니다. 또한, Ax-Prover는 기존의 증명 에이전트들이 가지는 주요 제한점을 극복하여, 지속적인 재훈련 없이도 Mathlib의 최신 버전과 작동할 수 있습니다.

- **Performance Highlights**: Ax-Prover는 두 개의 공공 데이터셋에서 경쟁력 있는 성능을 보였으며, 새로운 데이터셋에서는 기존의 전문 증명 시스템을 크게 초월했습니다. 특히, 수학적 경쟁 문제인 PutnamBench에서 좋은 성과를 거두었고, 새로운 데이터셋인 AbstractAlgebra와 QuantumTheorems에서도 성능을 입증했습니다. 이는 Ax-Prover가 다양한 과학 분야에서의 형식 검증을 위한 핵심 AI 도구로서의 가능성을 보여줍니다.



### CTRL-Rec: Controlling Recommender Systems With Natural Languag (https://arxiv.org/abs/2510.12742)
- **What's New**: 본 논문에서는 전통적인 추천 시스템에 자연어(Natural Language) 제어를 통합하는 CTRL-Rec 방법을 제안합니다. 이 접근 방식은 사용자가 명시적으로 제시한 선호와 참여 신호(Engagement Signals) 사이의 균형을 이루고 현대 추천 시스템의 검색 단계에 직접 영향을 미칠 수 있습니다. 이는 사용자가 간단한 자연어 요청으로 추천 내용을 즉각적으로 업데이트할 수 있게 합니다.

- **Technical Details**: CTRL-Rec은 LLM(대규모 언어 모델)을 활용하여 사용자의 자연어 요청을 근거로 특정 항목에 대한 사용자 판단을 시뮬레이트합니다. 이를 통해 생성된 예측 결과는 전통적인 추천 시스템에서 최적화하는 신호의 가중치에 통합되어, 명시된 선호와 참여 간의 균형을 이루게 됩니다. 이 시스템은 각 사용자 요청에 대해 단일 LLM 임베딩 계산만을 요구하여 실시간 추천 제어를 가능하게 합니다.

- **Performance Highlights**: 실험 결과, CTRL-Rec은 MovieLens 데이터셋에서 다양한 요청에 대해 세밀한 제어를 가능하게 했습니다. 19명의 Letterboxd 사용자를 대상으로 한 연구에서도 사용자들은 CTRL-Rec에 긍정적인 반응을 보였으며, 추천 시스템에 대한 통제감과 만족도가 전통적인 통제 방식에 비해 크게 향상된 것으로 나타났습니다.



### Clutch Control: An Attention-based Combinatorial Bandit for Efficient Mutation in JavaScript Engine Fuzzing (https://arxiv.org/abs/2510.12732)
- **What's New**: 클러치(CLUTCH)는 JavaScript 코드에서 더 나은 변형(변동) 대상을 선택하기 위한 심층 조합 밴딧(deep combinatorial bandit) 방법을 제안합니다. 이는 주목(attention) 메커니즘을 활용하여 가변 길이의 JavaScript 테스트 케이스 표현을 관찰할 수 있게 합니다. 이 방법은 기존의 무작위 선택 방식 대신 보다 효과적인 테스트 케이스 생성에 도움을 줍니다.

- **Technical Details**: 클러치는 Pointer Networks의 주목 메커니즘과 Concrete Dropout 기술을 사용하여 소프트웨어 테스트 과정에서의 변동성과 탐색의 동적 제어 문제를 해결합니다. 이 심층 네트워크는 변형을 적용할 적절한 위치를 선택하고, 즉각적인 피드백을 받아 학습에 활용하는 방식으로 작동합니다. 이를 통해 JavaScript 엔진의 효율적인 퍼징(fuzzing)을 가능하게 합니다.

- **Performance Highlights**: 클러치는 기존 최첨단 솔루션에 비해 평균적으로 유효한 테스트 케이스의 수는 20.3% 증가시키고, 각 테스트 케이스의 커버리지(coverage)는 8.9% 향상됩니다. 또한, 불안정하고 조합적인 환경에서 클러치는 기존 밴딧 기반 평가에서 평균적으로 78.1% 및 4.1% 낮은 후회(regret)를 기록하여 성능의 우수성을 입증합니다.



### Towards Robust Artificial Intelligence: Self-Supervised Learning Approach for Out-of-Distribution Detection (https://arxiv.org/abs/2510.12713)
- **What's New**: 이 논문에서는 레이블이 없는 데이터를 필요로 하지 않고도 OOD(Out-of-Distribution) 샘플 탐지를 개선할 수 있는 접근 방식이 제안됩니다. 자가 지도 학습(self-supervised learning)의 원리를 활용하여 모델이 레이블이 없는 데이터에서 유용한 표현을 학습할 수 있도록 합니다. 그래프 이론(graph theory) 기법과 결합하여 OOD 샘플을 보다 효율적으로 식별하고 분류할 수 있습니다. 이 방식은 기존의 최첨단 метод과 비교하여 0.99의 AUROC 점수를 달성하는 성과를 보였습니다.

- **Technical Details**: 제안된 방법은 두 가지 단계로 구성됩니다: In-Distribution(푸레이어) 데이터 표현 단계와 OOD 추론 단계입니다. 자가 지도 대비 학습(self-supervised contrastive learning)을 통해 입력 샘플에서 임베딩을 추출하고, 이를 통해 고차원 데이터를 보다 공고한 표현으로 표현합니다. SimCLR와 같은 프레임워크를 사용하여 이미지를 증강한 버전 사이에서 강력한 표현을 학습하며, 이 과정에서 PCA(주성분 분석)를 활용하여 차원을 축소하여 계산 비용을 줄입니다.

- **Performance Highlights**: 제안된 접근 방식은 CIFAR-10, CIFAR-100, SVHN 등의 벤치마크 데이터셋에서 0.99의 AUROC 점수를 기록했습니다. 이는 기존 MAPLE과 SSD방법보다 향상된 성능을 보여주며, 레이블 없는 데이터에 대한 의존성을 최소화합니다. 그래프 클러스터링을 통해 다양한 OOD 샘플을 효과적으로 감지하고, 고차원 공간에서도 강력한 성능을 유지하는 장점을 가지고 있습니다.



### CAMNet: Leveraging Cooperative Awareness Messages for Vehicle Trajectory Prediction (https://arxiv.org/abs/2510.12703)
Comments:
          Accepted at the IEEE Consumer Communications & Networking Conference (CCNC) 2026 - Las Vegas, NV, USA 9 - 12 January 2026

- **What's New**: 이 논문은 차량 간 통신 데이터를 활용하여 차량 경로 예측의 가능성을 탐구합니다. 특히, Cooperative Awareness Messages (CAMs)를 사용하여 모빌리티 예측 데이터셋을 구축하고, 새로운 신경망 모델인 CAMNet를 설계하여 평가합니다. 이러한 접근 방식은 센서로 인한 인식의 한계를 극복하고 자율주행의 안전성을 향상시킬 수 있는 가능성을 제시합니다.

- **Technical Details**: CAM은 차량의 위치, 속도 및 방향과 같은 광범위한 유용한 데이터를 포함합니다. 논문에서는 CAM을 활용하여 차량의 경로를 예측하는 모델을 수학적으로 정의하고, 과거 관측 데이터와 이웃 차량의 데이터를 입력으로 사용하여 미래 위치를 예측합니다. 제안된 CAMNet 네트워크는 VAE, RNN, GNN을 결합한 독창적인 구조로, 기존의 차량 모션 예측 방법과 차별화됩니다.

- **Performance Highlights**: CAM을 기반으로 한 경로 예측 모델은 기존의 센서 기반 접근 방식보다 더 넓은 인식 범위를 제공하며, promising results를 보여줍니다. 논문에서는 CAM 데이터를 활용한 예측이 실제 자율주행 성능을 크게 향상시킬 수 있는 가능성을 제시하고 있으며, 향후 연구 기회를 제안합니다. 다만, 접근 방식의 여러 제한점도 논의됩니다.



### Multi-Agent Debate for LLM Judges with Adaptive Stability Detection (https://arxiv.org/abs/2510.12697)
- **What's New**: 이 논문에서는 다수의 LLM(대형 언어 모델)을 활용한 "멀티 에이전트 토론판사(Multi-Agent Debate Judge)" 프레임워크를 제안합니다. 기존의 단순 집계 방법인 다수결(voting)로는 오류가 발생할 위험이 크기 때문에, 에이전트들이 협력하여 추론하고 피드백을 주고받는 구조를 도입했습니다. 이러한 개발을 통해 LLM의 판별 정확성을 향상시키는 동시에 계산 효율성을 확보하고자 하였습니다.

- **Technical Details**: 새로운 토론 프레임워크에서는 여러 LLM이 상호작용하며 답변을 생성하고 반복적으로 이를 수정하는 과정을 formalized (형식화)합니다. 이 과정에서, Beta-Binomial mixture 모델을 기반으로 하는 안정성 탐지 메커니즘이 도입되어 분포의 동적 변화를 감지하고, Kolmogorov-Smirnov 테스트를 통해 분포의 유사성을 기반으로 적절한 종료 시점을 결정합니다.

- **Performance Highlights**: 실험을 통해 제안된 프레임워크가 다수결 방식보다 정확성에서 유의미한 개선을 보여주었으며, adaptive stopping 메커니즘을 통해 계산 비용을 크게 줄일 수 있었음을 입증하였습니다. 다양한 벤치마크와 LLM 아키텍처를 통해 다수결의 해석력을 초과하는 결과를 보여줘, LLM이 평가 업무에서보다 효과적인 방식으로 활용될 수 있음을 시사합니다.



### ERA: Transforming VLMs into Embodied Agents via Embodied Prior Learning and Online Reinforcement Learning (https://arxiv.org/abs/2510.12693)
- **What's New**: 최근의 발전은 신체화된 AI(embodied AI)가 비전 언어 모델(VLMs)을 통해 지각, 추론 및 복잡한 환경에서의 상호 작용을 가능하게 할 잠재력을 강조합니다. 이 연구에서는 신체화된 추론 에이전트(ERA)라는 두 단계 프레임워크를 소개하며, 이는 사전 지식 학습과 온라인 강화 학습(online reinforcement learning, RL)을 통합해 에이전트의 성능을 향상시킵니다. ERA는 소형 VLMs로 하여금 일반화 가능한 신체화된 기술을 습득하도록 돕습니다.

- **Technical Details**: ERA는 두 단계로 나뉘며, 첫 번째 단계인 '신체화된 사전 학습(Embodied Prior Learning)'에서는 강력한 모델로부터 생성된 구조적 추론으로 기존의 궤적 데이터를 보강하도록 설계되었습니다. 두 번째 단계는 온라인 RL 파이프라인을 개발하여 사전 지식을 기반으로 에이전트 성능을 더욱 향상시킵니다. 여기에서 '자기 요약(self-summarization)', '조밀한 보상 조정(dense reward shaping)', '턴 수준 정책 최적화(turn-level policy optimization)'를 포함한 세 가지 핵심 설계를 통해 안정적이고 효율적인 정책 학습을 가능하게 합니다.

- **Performance Highlights**: ERA-3B 모델은 고급 계획(EB-ALFRED) 및 저급 제어(EB-Manipulation) 작업에 대해 뛰어난 성능을 보여줍니다. EB-ALFRED에서는 GPT-4o 모델보다 8.4% 증가, EB-Manipulation에서는 19.4% 향상된 성인을 달성하였습니다. 저해상도, 희소 보상을 포함한 도전적인 환경에서도 ERA는 뛰어난 일반화 성능을 나타내어, 신체화된 AI 시스템의 향후 연구를 위한 실용적인 통찰을 제공합니다.



### Memory as Action: Autonomous Context Curation for Long-Horizon Agentic Tasks (https://arxiv.org/abs/2510.12635)
- **What's New**: 이번 연구에서는 메모리 관리를 학습할 수 있는 내재적 능력으로 재정의하여 'Memory-as-Action' 프레임워크를 제안합니다. 메모리 관리가 별도의 외부 메커니즘이 아닌 강화학습을 통해 자동으로 이루어지도록 하여, 에이전트가 작업 목표를 달성할 수 있도록 지원합니다. 이 접근 방식은 추상적 메모리 수정을 통해 LLM의 제한된 메모리 문제를 해결할 수 있는 기초를 마련합니다.

- **Technical Details**: 'Memory-as-Action' 프레임워크는 에이전트가 역사적 정보를 직접 수정할 수 있도록 하는 기능을 통합했습니다. 이 방법으로 에이전트는 메모리 조작을 표준 함수 호출로 구현하여 기록하고 관리하는 방식으로 진행됩니다. 그러나 메모리 조작이 과거의 정보를 덮어쓰는 경우, 표준 LLM의 정책 최적화에 명확한 도전이 발생하며, 이 문제를 해결하기 위해 'Dynamic Context Policy Optimization' 알고리즘을 제안합니다.

- **Performance Highlights**: 실험 결과, 제안된 'Memory-as-Action' 프레임워크는 더 큰 모델들과 비교해도 경쟁력 있는 성능을 보이며, 메모리 전략이 새로운 작업에 일반화되고 모델의 기본 능력에 적합함을 보여줍니다. 이 접근 방식은 자원 소모를 줄이면서도 작업 성과를 향상시키는 적응형 컨텍스트 큐레이션 전략을 통해 달성되었습니다.



### HardcoreLogic: Challenging Large Reasoning Models with Long-tail Logic Puzzle Games (https://arxiv.org/abs/2510.12563)
- **What's New**: 이 논문에서는 HardcoreLogic이라는 새로운 벤치마크를 소개하며, 10종의 논리 게임에서 5,000개 이상의 퍼즐을 포함하고 있습니다. 이 벤치마크는 기존의 표준 퍼즐 포맷에서 벗어난 다양한 변형을 통해 LRMs의 강인성을 테스트하는 데 초점을 맞추고 있습니다. 이를 통해 LRMs가 새로운 규칙에 적응하거나 다양한 조건에서 적절한 규칙을 유연하게 적용할 수 있는지를 평가합니다.

- **Technical Details**: HardcoreLogic은 동일한 게임의 기존 데이터셋과 비교했을 때 이론적 복잡성이 높고 기존 퍼즐과의 노출 가능성을 줄이는 방식으로 변형됩니다. 세 가지 변환 차원인 Increased Complexity (IC), Uncommon Elements (UE), Unsolvable Puzzles (UP)에 따라 구성되며, 이는 퍼즐의 난이도를 높이고 새로운 요소를 도입합니다. 모델은 여러 범주의 LRMs에 대해 평가되며, 이들 중 가장 최신 모델조차도 성능이 크게 저하되는 것을 보입니다.

- **Performance Highlights**: LRMs는 HardcoreLogic 구현에서 상당한 성능 저하를 겪으며, 특히 복잡성이 증가하거나 낯선 형태의 퍼즐에 대해 더 큰 문제가 발생합니다. 이러한 모델은 종종 기존에 학습한 고정된 추론 패턴을 적용하거나 문제의 규칙을 잘못 이해하며 오류를 범합니다. 전반적으로 이 연구는 현재 LRMs의 한계를 드러내고 깊이 있는 추론 능력 향상의 필요성을 강조합니다.



### Inclusive Fitness as a Key Step Towards More Advanced Social Behaviors in Multi-Agent Reinforcement Learning Settings (https://arxiv.org/abs/2510.12555)
Comments:
          This version is a slightly updated version (e.g., added an important reference) compared to the peer-reviewed versions at 'Adapative Learning Agents' at AAMAS 2022 or 'From Cells to Societies' at ICLR 2022

- **What's New**: 본 연구에서는 자연 선택의 진화 과정을 영감을 받아 다중 에이전트 강화 학습 프레임워크를 제안합니다. 각 에이전트는 유전형(genotype)을 부여받고, 보상 함수는 포괄적 적합성(inclusive fitness) 개념에 따라 모델링됩니다. 에이전트의 유전 물질은 다른 에이전트와 공유될 수 있으며, 이는 자연스럽게 사회적 역학을 형성합니다.

- **Technical Details**: 유전형 기반의 보상 구조를 통해 에이전트 간의 협력 스펙트럼이 생성되며, 유전적 유사성을 기반으로 하여 경쟁과 협력이 조절됩니다. 본 연구는 네트워크 게임의 죄수의 딜레마를 포함한 두 가지 유형의 게임을 통해 그 결과가 해밀턴의 법칙(Hamilton's rule)과 일치함을 보입니다. 또한, 공간적 및 시간적 구조, 유한 자원, 진화하는 집단을 포함하는 환경으로의 확장을 모색합니다.

- **Performance Highlights**: 제안된 방법은 기존의 팀 기반 구조와 달리 비팀 기반의 독특한 사회적 역학을 가능하게 합니다. 예를 들어, 하나의 에이전트가 두 다른 에이전트와 협력적인 관계를 가지면서도, 이 두 에이전트는 서로 경쟁하는 상황을 구현할 수 있습니다. 이러한 방식은 에이전트들이 새로운 전략을 찾아 협력과 배신을 균형 있게 조정하도록 지속적으로 도전합니다.



### ProtoSiTex: Learning Semi-Interpretable Prototypes for Multi-label Text Classification (https://arxiv.org/abs/2510.12534)
- **What's New**: 새로운 연구에서 ProtoSiTex라는 반해석 가능한 프레임워크를 제안하여 다중 레이블 텍스트 분류를 위한 정교한 분류가 가능해진다. 기존 모델의 한계를 극복하고 문장이나 문서 수준의 예측을 넘어 하위 문장 수준에서의 정교한 설명을 통해 해석 가능성을 높인다. 이 연구는 호텔 리뷰 데이터를 하위 문장 단위로 주석을 달아 새로운 벤치마크 데이터셋도 도입한다.

- **Technical Details**: ProtoSiTex는 이중 단계 교차 훈련 전략을 사용하여 설계되며, 첫 번째 단계에서는 자율적인 프로타입 발견을 통해 의미적으로 일관성 있는 프로타입을 학습한다. 두 번째 단계에서는 지도 학습을 통해 이러한 프로타입을 클래스 레이블에 매핑하여, 다중 레이블 상황을 효과적으로 처리하며, 하위 문장, 문장, 문서 수준의 일관성을 강화하는 계층적 손실 함수도 도입한다. 또한, 적응형 프로타입 학습과 다중 머리 주의(attention) 메커니즘을 통해 겹치는 의미를 캡처한다.

- **Performance Highlights**: ProtoSiTex는 세 가지 데이터셋(호텔 리뷰, IMDb, TweetEval)에서의 실험을 통해 기존 해석 가능한 기준 모델과 비교하여 뛰어난 성능을 보여주었다. 이를 통해 ProtoSiTex는 성능, 견고성 및 상호 운용성 측면에서 최신 블랙박스 모델과 동등한 결과를 달성하며, 견고한 반해석 가능한 다중 레이블 텍스트 분류 솔루션으로 자리매김하게 된다.



### Artificial Intelligence Virtual Cells: From Measurements to Decisions across Modality, Scale, Dynamics, and Evaluation (https://arxiv.org/abs/2510.12498)
- **What's New**: 이 논문에서는 Artificial Intelligence Virtual Cells (AIVCs)가 세포 상태의 실행 가능한 결정 관련 모델을 다중 모드(multimodal) 및 다중 스케일(multiscale) 측정에서 학습하는 것을 목표로 한다. 최근의 연구들은 단일 세포 및 공간 기초 모델, 크로스 모달리티(cross-modality) 정렬 개선, 스케일링된 교란 아틀라스를 도입했다. 그러나 데이터 세트와 환경에 국한된 평가가 여전히 주를 이루며, 데이터 전송의 한계, 누수 및 커버리지 편향에 취약한 문제를 지적하고 있다.

- **Technical Details**: 제안된 모델 불가지론적 Cell-State Latent (CSL) 관점은 학습 조직을 Measurement, Lift/Project for cross-scale coupling, Intervention for dosing and scheduling의 연산자 문법(operator grammar)을 통해 구성한다. CSL은 트랜스크립토믹(transcriptomic), 단백질체(proteomic), 후생유전학(epigenomic), 공간(spatial) 및 이미징(imaging) 모달리티를 통합하여 계층 간 일관성을 보장하고, 벤치마크 작업(benchmark tasks) 전반에 걸쳐 엄격한 평가를 지원하도록 설계됐다.

- **Performance Highlights**: AIVCs의 발전은 데이터 기반 가상 세포 준비의 가능성과 긴급성을 강조한다. 또한 여러 모드와 스케일을 넘나드는 정보 정렬 및 예측, 반사실(counterfactual) 유도 등을 위한 새로운 개념적 기틀을 제공한다. 이 연구는 다중 모드 세포 데이터에서의 일관된 표현 및 추론을 위한 AI 방법의 확산을 촉진하며, AIVCs 분야에서의 재현 가능하고 비교 가능한 발전을 위한 기반을 마련하고 있다.



### Using Medical Algorithms for Task-Oriented Dialogue in LLM-Based Medical Interviews (https://arxiv.org/abs/2510.12490)
- **What's New**: 새로운 연구는 의료 질문의 방향성이 있는 비순환 그래프(Directed Acyclic Graph, DAG)로 구조화된 대화 시스템을 개발했습니다. 이 시스템은 환자의 정보를 사전 없이도 효율적으로 질문할 수 있는 '콜드 스타트(cold-start)' 메커니즘을 포함하고 있으며, 환자의 응답에 따라 대화의 방향을 조정할 수 있는 메커니즘을 갖추고 있습니다. 또한, 이 연구는 의사와 환자 모두를 위한 사용자 친화적인 구조를 유지하면서 의료 보고서를 자동으로 생성하는 방법을 제안합니다.

- **Technical Details**: 이 프레임워크는 방향성이 있는 비순환 그래프(DAG) 구조로 구성되어 있으며, 이는 대화 흐름을 논리적이며 목표 지향적으로 조직합니다. 각 노드는 증상, 약물, 가족력 등과 같은 특정 의학적 질문을 나타내고, 유도된 에지를 통해 환자의 응답에 따라 대화가 진행됩니다. 이 프레임워크는 생성된 의료 보고서가 임상 워크플로우에 맞춰 요약될 수 있도록 설계되었습니다.

- **Performance Highlights**: 초기 평가에서 환자 애플리케이션은 낮은 인지 부담(NASA-TLX = 15.6)과 높은 사용 편의성(SUS = 86), 강한 만족도(QUIS = 8.1/9)를 보여줬습니다. 의사 애플리케이션도 적절한 인지 부담(NASA-TLX = 26)과 우수한 사용자 경험(SUS = 88.5)을 기록하며 긍정적인 평가를 받았습니다. 두 애플리케이션 모두 임상 환경에서 직무 요구를 충족하며 효과적인 보고서 생성을 지원하는 것으로 나타났습니다.



### Evaluating and Mitigating LLM-as-a-judge Bias in Communication Systems (https://arxiv.org/abs/2510.12462)
- **What's New**: 이번 연구에서는 LLM(대규모 언어 모델) 기반 AI 판별 시스템의 편향을 체계적으로 조사했습니다. 연구는 GPT-Judge와 JudgeLM 두 모델을 분석하고, 여러 유형의 편향이 LLM의 채점에 미치는 영향을 평가하였습니다. 낮은 점수를 매기는 것과 같은 편향 저항성을 발견하고, 특정 기준을 정하는 것이 편향에 대한 저항성을 높이는 데 효과적임을 입증했습니다.

- **Technical Details**: 류는 11종의 편향을 고려하며, 이는 언어적 스타일에서 나오는 암묵적 편향과 외부 요인에 의해 유발되는 명시적 편향을 포함합니다. 연구 결과, 고도로 구성된 LLM 판별기는 암묵적 편향에서 거의 면역인 것으로 나타났으며, 명시적 편향을 적절히 처벌하였습니다. 또한, 고득점의 편향 데이터를 사용한 미세 조정이 모델 성능을 크게 저하시킬 위험이 있음을 발견했습니다.

- **Performance Highlights**: 탄탄한 평가 루브릭을 포함한 LLM 판별기의 경우, 편향이 있는 응답의 평균 점수는 항상 상대적으로 낮았습니다. 예를 들어, 권위 있는 참고자료를 추가하면 GPT-Judge의 점수가 9.12에서 3.94로 감소했습니다. 여러 데이터셋에서 이러한 편향 효과가 일반화된다는 점도 강조되었습니다.



### Biased-Attention Guided Risk Prediction for Safe Decision-Making at Unsignalized Intersections (https://arxiv.org/abs/2510.12428)
- **What's New**: 이 논문은 신호가 없는 교차로에서의 자율주행 의사결정을 향상시키기 위해, 편향된 주의 메커니즘(biased attention mechanism)과 통합된 딥 강화 학습(Deep Reinforcement Learning, DRL) 결정 프레임워크를 제안합니다. 이 프레임워크는 Soft Actor-Critic(SAC) 알고리즘을 기반으로 하며, 차량이 교차로에 진입할 때의 장기 충돌 위험을 평가하는 교통 위험 예측기를 통해 안전한 주행 결정을 유도합니다.

- **Technical Details**: 제안된 방법론은 Transformer 모델을 기반으로 한 위험 예측기와 계층화된 경험 재생 메커니즘을 활용하여 구성됩니다. 이 위험 예측기는 차량의 과거 궤적과 현재의 교통 상황을 바탕으로 장기 충돌 위험을 학습하고 예측하며, 이 예측된 위험을 밀접한 보상 신호로 변환하여 강화 학습 에이전트에게 미리 안전을 안내합니다. 또한, 다양한 전이(transition)를 저장하는 계층적 경험 버퍼가 설계되어 모델의 수렴 속도를 가속합니다.

- **Performance Highlights**: 실험 결과는 제안된 방법이 교차로에서의 교통 효율성과 차량 안전성을 모두 향상시키는 데 효과적임을 입증하였습니다. 특히, 이 intelligent decision-making framework는 복잡한 시나리오에서의 기능을 입증하여 향후 자율주행 시스템의 안전성 강화를 위한 중요한 기초자료를 제공합니다.



### MTOS: A LLM-Driven Multi-topic Opinion Simulation Framework for Exploring Echo Chamber Dynamics (https://arxiv.org/abs/2510.12423)
Comments:
          14 pages, 11figures

- **What's New**: 이 연구에서는 다중 주제 환경에서의 의견 진화 과정을 모델링하기 위해 Multi-topic Opinion Simulation (MTOS) 프레임워크를 제안합니다. 기존의 대형 언어 모델(LLMs)을 활용해 의견 변화 시뮬레이션을 가능하게 하며, 다양한 사용자 선택 메커니즘과 동적 주제 선택 전략을 통합하여 정교한 상호작용을 도모합니다. 이 프레임워크는 다양한 주제 간의 상호작용을 시뮬레이션 할 수 있는 능력을 제공하며, 기존의 수치 모델이 포착할 수 없었던 인지 전이(cognitive transfer)와 주제 간 연관성을 강조합니다.

- **Technical Details**: MTOS에서는 각 에이전트가 성별, 나이 및 개인 특성에 따라 독특한 역할로 초기화되고, 다중 주제에 대해 초기 의견 벡터가 설정됩니다. 에이전트는 스케일-프리 네트워크 내에서 이질적인 연결을 생성하여 실제 사회에서의 다양한 상호작용 구조를 시뮬레이션하며, 단기 메모리(short-term memory)와 장기 메모리(long-term memory)를 통해 상호작용 및 인지 맥락을 기록합니다. 또한 여기에 동적 정보 감쇠 메커니즘을 통합하여 역사적 정보를 조정하며, 다중 주제 추천 메커니즘을 통해 에이전트의 주제 선택 편향을 조정합니다.

- **Performance Highlights**: MTOS의 성능은 단일 주제와 다중 주제 조건 하에서 그룹 의견의 진화를 비교하는 실험을 통해 평가되었습니다. 결과적으로 MTOS는 이웃 관련 지표 및 글로벌 발산 지표에서 더 현실적인 시뮬레이션 결과를 보여주었으며, 실제 세계에서 관찰된 에코 챔버(echo chamber) 현상을 재현하는 데 성공했습니다. 연구 결과는 다중 주제가 사회적 환경에서 의견 동역학에 미치는 중요성을 탐구하며, 기존의 안전통제모델(SSF)과 비교하여 더 해석 가능한 도구를 제공함으로써 시뮬레이션의 실용성과 사회적 타당성을 향상시켰습니다.



### PricingLogic: Evaluating LLMs Reasoning on Complex Tourism Pricing Tasks (https://arxiv.org/abs/2510.12409)
- **What's New**: 이번 논문에서는 여러 중복 요금 규칙이 적용되는 관광 관련 가격을 자동화할 수 있는 대형 언어 모델(LLMs)의 신뢰성을 평가하는 새로운 벤치마크인 PricingLogic을 소개합니다. 여행사에서 이 오류가 발생하기 쉬운 작업을 AI 시스템에 맡기고자 하지만, 신뢰성 검증 없이 LLM을 배포할 경우 큰 재정적 손실과 고객 신뢰 저하를 가져올 수 있습니다. PricingLogic은 총 300개의 자연어 질문으로 구성되어 있으며, 두 가지 난이도 수준으로 나뉘어 있습니다: 기본 고객 유형 가격 및 할인 조건이 상호 작용하는 복합 투어 계산.

- **Technical Details**: PricingLogic은 세 가지 난이도로 나뉜 300개의 질문으로 구성되어 있으며, 기본적인 고객 유형의 가격 계산부터 복잡한 관광 예약 가격 계산을 포함합니다. 각 질문은 예약 요청에서 파생된 42개의 실제 가격 정책을 기반으로 하며, LLM의 성능을 평가하기 위해 고안된 구조입니다. 이 과정에서 LLM은 가격 정책을 실행 가능한 Python 코드로 변환하고, 자연어로 제공된 질문을 통해 필요한 정보를 추출하여 가격을 계산합니다.

- **Performance Highlights**: 실험 결과, 복잡성이 증가할수록 모든 LLM의 성능이 크게 저하됨을 확인했습니다. 특히, 가장 어려운 질문에서는 LLM이 정확히 절반도 해답하지 못하는 성과를 보여주었습니다. 코드 지원 논리를 사용하는 CaR 접근 방식이 대부분의 경우 E2E 성능을 개선했지만, 여전히 모든 모델의 절대적인 정확도가 60% 미만에 머물렀습니다. 전반적으로 LLM의 관광 가격 계산 작업에 대한 신뢰성 확보를 위한 추가적인 연구 방향을 제시합니다.



### A Survey of Vibe Coding with Large Language Models (https://arxiv.org/abs/2510.12399)
- **What's New**: 대형 언어 모델(LLM)의 발전은 코드 생성 보조 기능에서 자율 코딩 에이전트로의 패러다임 전환을 촉진했습니다. 이로 인해 개발자는 AI가 생성한 구현을 라인별 코드 이해가 아닌 최종 결과 관찰을 통해 검증하는 새로운 개발 방법론인 "Vibe Coding"을 사용하고 있습니다. 하지만, 이 혁신적인 패러다임의 효과는 아직 충분히 탐구되지 않았으며, 경험적 증거는 예상치 못한 생산성 손실과 인간-AI 협업의 근본적인 문제점을 드러냅니다.

- **Technical Details**: 이 논문에서는 Vibe Coding을 대형 언어 모델과 관련된 처음으로 포괄적이고 체계적인 리뷰로 제공합니다. 이를 위해 1000편 이상의 연구 논문의 체계적 분석을 수행하였으며, LLMs, LLM 기반 코딩 에이전트, 코딩 에이전트의 개발 환경, 피드백 메커니즘 등 중요한 인프라 구성 요소를 조사했습니다. Vibe Coding은 Constrained Markov Decision Process로 형식화되어 인간 개발자, 소프트웨어 프로젝트 및 코딩 에이전트 간의 동적 삼원 관계를 포착합니다.

- **Performance Highlights**: LLM 기반 코딩 에이전트는 자율적으로 환경을 구성하고, 프로그램을 실행하며, 오류를 자기 진단하고, 구현을 업데이트할 수 있습니다. 이러한 과정은 전통적인 코드 이해의 요구를 넘어 결과 지향적인 검증으로 전환되는 중요한 변화를 보여줍니다. 그러나 연구 결과에 따르면, 복잡한 작업은 비구조적 자연어 지시의 근본적인 한계를 드러내며, 경험 많은 개발자들도 예상보다 더 긴 완료 시간을 경험하는 경우가 많습니다.



### O-Forge: An LLM + Computer Algebra Framework for Asymptotic Analysis (https://arxiv.org/abs/2510.12350)
- **What's New**: 이 논문은 LLM+CAS 프레임워크와 O-Forge 툴을 소개하여, 주요 LLM(large language models)과 컴퓨터 대수 시스템(computer algebra systems)을 결합해 비약적 불평등(asymptotic inequalities)에 대한 창의적이고 검증된 증명을 생성하는 방법을 제안합니다. 이는 수학적 증명의 진위를 확인하기 어려운 문제를 해결하는 데 도움을 줄 수 있습니다. 특히, 수학자 Terence Tao의 질문에 답변하고 AI가 연구 수준 도구로 발전할 방법을 제시합니다.

- **Technical Details**: O-Forge는 LaTeX 형식의 추정값을 입력으로 받아들여 해당 추정값의 각 하위 도메인(subdomain)에서 증명이 가능한지를 확인하는 시스템입니다. 이 시스템은 첫째, 사용자의 입력을 적절한 문제 인스턴스로 파싱하고, 둘째, 효율적인 도메인 분해를 제안하며, 셋째, 각 부분에 대해 엄정한 증명을 생성하여 전체 도메인을 검증합니다. 이 과정에서 Mathematica의 Resolve 기능을 사용해 비선형 함수와 관련된 추정값을 신뢰성 있게 증명합니다.

- **Performance Highlights**: 이 시스템을 사용하여 약한 펜첼-영 불평등(weak Fenchel-Young inequality)과 연구 수준 시리즈 추정(series estimate)과 같은 어려운 수학 문제를 성공적으로 검증하였습니다. O-Forge는 난해한 연구 문제에 대한 증명 완성을 자동화할 수 있는 능력이 있으며, 기존 AI 툴들이 수행하지 못했던 기호 검증(symbolic verification)을 통해 수학자들에게 시간과 노력을 절약할 수 있는 혁신적인 도구로 자리잡고 있습니다.



### RAG-Anything: All-in-One RAG Framework (https://arxiv.org/abs/2510.12323)
- **What's New**: Retrieval-Augmented Generation (RAG)는 대형 언어 모델(LLM)의 지식 경계를 확장하는 중요한 패러다임으로 등장했습니다. 기존의 RAG 시스템은 텍스트만을 중심으로 구성되어, 현실 세계의 멀티모달 정보 환경과 맞지 않는다는 문제를 제기합니다. 새로운 RAG-Anything 프레임워크는 텍스트, 이미지, 표, 수식 등 다양한 데이터 유형을 통합적으로 처리할 수 있도록 설계되었습니다.

- **Technical Details**: RAG-Anything는 다양한 정보 유형의 원활한 통합을 요구하는 통합 멀티모달 표현 문제를 해결하기 위해, 구조적 지식을 탐색하는 기능과 의미적 유사성 매칭을 조합하는 하이브리드 검색 메커니즘을 도입합니다. 이를 통해 데이터의 고유 특성과 크로스 모달 관계를 유지하면서 효과적인 멀티모달 지식 검색을 가능하게 합니다. 또한, 이 프레임워크는 다양한 문서 형식과 전문적 요구에 맞춰 필요로 하는 지식을 효과적으로 활용할 수 있도록 설계되었습니다.

- **Performance Highlights**: RAG-Anything은 DocBench와 MMLongBench라는 두 가지 도전적인 멀티모달 벤치마크에서 탁월한 성능을 입증했습니다. 기존 최첨단 방법들에 비해 큰 성능 향상을 보여 주었으며, 특히 긴 문서에서 그 성능 상승이 두드러지게 나타났습니다. 실험을 통해 그래프 기반 지식 표현이 주요 성과 향상의 원인임을 확인했으며, 복잡한 레이아웃 내에서의 정확한 위치 지정에서도 뛰어난 성능을 제공합니다.



### Tensor Logic: The Language of AI (https://arxiv.org/abs/2510.12269)
Comments:
          17 pages, 0 figures

- **What's New**: 이번 논문에서는 인공지능(AI) 분야의 발전을 저해하는 언어의 부족 문제를 다루고 있습니다. 기존의 프로그래밍 언어인 Python은 AI 전용으로 설계되지 않았으며, TensorFlow나 PyTorch와 같은 라이브러리의 추가적인 기능이 자동 추론(automated reasoning) 및 지식 습득(knowledge acquisition)과 같은 주요 작업에 도움이 되지 않는다는 점을 지적합니다. 본 연구는 텐서 로직(tensor logic)이라는 새로운 언어를 제안하여 신경망(neural)과 기호(symbolic) AI를 근본적으로 통합하고, AI의 한계를 극복할 해결책을 모색합니다.

- **Technical Details**: 텐서 로직의 핵심 구성 요소는 텐서 방정식입니다. 이는 논리 규칙(logical rules)과 아인슈타인 합(sum) 운영이 본질적으로 동일하다는 관찰에 기초하고 있습니다. 저자는 텐서 로직에서 신경망, 기호 AI, 커널 기계(kernel machines), 그래픽 모델을 우아하게 구현하는 방법을 보여주며, 이론적 기초인 논리 프로그래밍(logic programming)과 텐서 대수(tensor algebra)에 대한 간략한 리뷰를 제공합니다. 또한, 텐서 로직은 임베딩 공간(embedding space)에서 신뢰할 수 있고 투명한 추론을 가능하게 하는 방법을 제안합니다.

- **Performance Highlights**: 텐서 로직은 신경망의 스케일 및 학습 가능성과 기호 추론의 신뢰성과 투명성을 결합하여 새로운 방향성을 제시합니다. 두 가지 확장 접근법을 통해 스케일 업을 위한 방법을 제안하며, 텐서 로직의 널리 사용될 가능성과 그 응용 방안에 대한 논의로 마무리됩니다. 본 논문의 목표는 AI의 보다 널리 채택될 수 있는 기반을 마련하는 데 있습니다.



### $\mathbf{T^3}$: Reducing Belief Deviation in Reinforcement Learning for Active Reasoning (https://arxiv.org/abs/2510.12264)
- **What's New**: 이 논문은 Active Reasoning을 위한 새로운 접근 방식을 제시합니다. 주목할 점은 LLM(대형 언어 모델)이 외부 정보와 상호 작용하여 문제를 해결할 때 발생하는 belief deviation 문제를 해결하기 위해 $	extbf{T^3}$라는 방법을 개발했다는 것입니다. 이 방법은 모델의 belief가 지나치게 편차가 생기는 경우를 탐지하고, 훈련 중 비정보적인 tail을 제거하도록 경로를 잘라냅니다.

- **Technical Details**: Active Reasoning 문제는 부분 관측 가능한 마르코프 의사결정 과정(POMDP)으로 모델링 됩니다. LLM은 불완전한 belief 상태를 추적하고 이를 업데이트하는 과정에서 발생하는 오류들이 문제 해결 진행을 저해할 수 있음을 설명합니다. T3 조건을 통해 LLM의 추론 추적에서 감지할 수 있는 신호를 통해 BTR(Belief-Trap Region)로의 진입을 확인하고, 비정보적인 tail을 잘라냄으로써 정책 최적화를 개선합니다.

- **Performance Highlights**: T3를 여러 데이터셋과 작업에서 평가한 결과, 훈련 안정성, 토큰 효율성 및 최종 성능이 모두 향상되었습니다. 최대 30%의 성능 향상과 함께 rollout tokens는 약 34% 절감되며, 다양한 LLM 크기, 아키텍처 및 out-of-distribution 시나리오에서도 견고한 성과를 보여주었습니다. 이러한 결과는 신뢰할 수 있는 Active Reasoning 에이전트를 구축하기 위한 원칙적인 접근을 제공합니다.



### PromptFlow: Training Prompts Like Neural Networks (https://arxiv.org/abs/2510.12246)
Comments:
          Comments: 18 pages, 14 figures, conference submission, appendix included

- **What's New**: 이 논문에서는 PromptFlow라는 모듈식 교육 프레임워크를 제안합니다. PromptFlow는 메타-프롬프트(meta-prompt), 연산자(operators), 최적화(optimization), 평가자(evaluator)를 통합하여 다양한 자연어 처리(NLP) 작업을 효율적으로 처리합니다. 이 프레임워크는 자동으로 최적의 프롬프트 수정 경로를 탐색할 수 있는 기능을 가지고 있어 최소한의 도메인 특정(training data)에 의존합니다.

- **Technical Details**: PromptFlow는 메타-프롬프트 생성기를 통해 여러 연산자를 활용하여 프롬프트 섹션을 생성합니다. 여기에는 자기 반영(self-reflection), 사슬사고(chain-of-thought), 차별적 진화(differential evolution)와 같은 다양한 연산자가 포함됩니다. 또한 메타 수준에서의 최적화 메커니즘과 강화 학습(reinforcement learning)을 결합하여 가장 효과적인 수정 경로를 선택합니다.

- **Performance Highlights**: 다양한 데이터셋에서 수행된 실험을 통해 PromptFlow는 기존 방법들보다 우수한 성능을 보였습니다. 특히, 명명된 엔티티 인식(NER), 분류(CLASSIFICATION), 기계 독해(MRC) 등의 다양한 NLP 작업에서 성과를 입증했습니다. 결과적으로, PromptFlow는 프롬프트 성능을 현저히 향상시키고, 여러 기준선(baselines) 방법들을 초월하는 결과를 도출했습니다.



### MedKGEval: A Knowledge Graph-Based Multi-Turn Evaluation Framework for Open-Ended Patient Interactions with Clinical LLMs (https://arxiv.org/abs/2510.12224)
- **What's New**: 이 논문에서는 의료 분야의 대화형 언어 모델(Large Language Models, LLMs)을 평가하기 위한 새로운 프레임워크인 MedKGEval을 제안합니다. 기존 모델 평가 방법의 한계를 극복하는 MedKGEval은 지식 그래프(knowledge graph)를 기반으로 하여, 다중 대화(turn-based conversation)에서의 환자 시뮬레이션(patient simulation) 메커니즘을 통합함으로써 더욱 사실적인 상호작용을 가능하게 합니다. 이 프레임워크는 임상 적합성(clinical appropriateness), 사실 정확성(factual accuracy) 및 안전성(safety)을 실시간으로 평가하는 Judge Agent를 포함하여, LLM의 동작을 보다 정밀하게 분석할 수 있습니다.

- **Technical Details**: MedKGEval은 네 가지 주요 역할 즉, Doctor Agent, Patient Agent, Judge Agent, Director Agent로 구성됩니다. 각 Agent는 임상 사실(clinical facts)과 증상-질병-약물 관계(symptom-disease-medication relationships)를 지식 그래프에 기반하여 일관되게 활용하면서, 다중 대화에서의 정확성 및 관련성을 세밀하게 평가할 수 있도록 설계되었습니다. 이러한 구조는 대화가 진행됨에 따라 정확성과 관련성이 어떻게 변동하는지를 시각적으로 보여주는 동시에, 평가 과정에서의 오류 전파(error propagation)와 맥락 변동(context drift) 문제를 조기에 탐지할 수 있도록 합니다.

- **Performance Highlights**: MedKGEval은 여덟 개의 최신 LLM을 대상으로 한 포괄적인 다중 평가 벤치마크(multi-turn benchmark)를 제공하여 기존 평가 방법이 놓치기 쉬운 미세한 행동 결함 및 안전 리스크를 식별하는 능력을 보여줍니다. 실험 결과는 MedKGEval이 임상적 실패 모드(clinically relevant failure modes)를 포괄적으로 발견할 수 있음을 입증합니다. MedKGEval은 처음에는 중국어 및 영어 의료 응용을 위해 설계되었지만, 추가 언어로의 확장을 용이하게 하는 지식 그래프를 통해 다국어 지원도 가능하다는 장점을 가지고 있습니다.



### GOAT: A Training Framework for Goal-Oriented Agent with Tools (https://arxiv.org/abs/2510.12218)
Comments:
          32 pages, 21 figures

- **What's New**: 최근 대형 언어 모델(LLM)의 발전은 사용자 의도에 기반하여 외부 도구를 사용하며 상호작용할 수 있는 에이전트를 만들어내는 새로운 패러다임으로 확장되었습니다. 그러나 현재 LLM 에이전트는 여전히 목표 지향 쿼리를 처리하는 데 제한적이며, 이를 해결하기 위한 새로운 훈련 프레임워크인 GOAT를 제안합니다. GOAT는 목표 지향 API 실행 작업을 위한 합성 데이터셋을 자동으로 생성함으로써, LLM 에이전트를 인간 주석 없이도 세부 조정할 수 있도록 합니다.

- **Technical Details**: GOAT는 API 문서에서 직접적으로 목표 지향 API 실행 작업을 생성하는 방법을 제안하며, 이는 API 의존성 그래프를 통해 가능한 호출 관계를 포착합니다. 이 과정에서 GOAT는 연결된 하위 그래프를 샘플링하여 각 API 호출의 출력이 다른 호출의 입력으로 활용될 수 있는 방식으로 훈련 샘플을 생성합니다. 이러한 방식은 훈련 데이터 생성과 동시에 LLM 및 검색 모델을 공동으로 세부 조정하여 상호 의존 API에 대한 추론 능력을 강화합니다.

- **Performance Highlights**: GOAT로 훈련된 에이전트는 여러 기존 목표 지향 벤치마크에서 최첨단 성능을 달성하며, 일부 경우에는 강력한 추론 능력을 가진 일부 폐쇄형 모델조차 초월했습니다. 또한 GOATBench라는 새로운 평가 벤치마크를 통해 이 에이전트들이 목표 지향 작업에서 일관된 성능 향상을 보임을 확인했습니다. 이러한 결과는 복잡한 추론과 도구 사용이 가능한 견고한 오픈 소스 LLM 에이전트를 구축하기 위한 실질적인 경로로 GOAT의 가능성을 강조합니다.



### On the Design and Evaluation of Human-centered Explainable AI Systems: A Systematic Review and Taxonomy (https://arxiv.org/abs/2510.12201)
- **What's New**: 이 논문에서는 사용자를 포함한 인간 중심의 평가 방법을 제안하여 Explainable AI (XAI) 시스템을 연구하는 65개의 사용자 연구를 종합적으로 검토합니다. 연구는 AI 전문가가 아닌 데이터 전문가와 AI 초보자를 타겟으로 하여 각각의 설계 목표를 정의하고, XAI 시스템의 특성과 평가 지표를 살펴봅니다. 연구 결과는 XAI 개발자들에게 인간 중심의 설계를 위한 가이드라인을 제공합니다.

- **Technical Details**: XAI 시스템은 일반적으로 신뢰성(reliability), 사용자 수용(acceptance), 사용성(usability) 및 해석 가능성(interpretability) 지표로 평가됩니다. 이 논문에서는 AI 초보자와 데이터 전문가를 위한 설계 목표를 제안하며, 평가 지표는 사용자의 특성과 행동을 고려하여 확장됩니다. 검토된 65개의 연구는 XAI 시스템의 특성과 설명 지표를 구분하는 데 도움을 주며, '블랙박스 모델(black-box models)'과 '화이트박스 모델(white-box models)'의 개념도 다룹니다.

- **Performance Highlights**: 연구는 XAI 시스템의 설계 목표와 관련된 주요 지표를 제공하며, 사용자 중심의 평가 방법을 통해 투명한 AI 시스템 개발을 촉진합니다. XAI 시스템은 잘못된 진단을 예방하고, 의사 결정을 개선하는데 필수적인 도구로 자리잡고 있습니다. 또한 XAI 알고리즘은 사용자가 결과를 이해하고 신뢰할 수 있도록 돕는 방식으로 사용되며, 이를 통해 AI 시스템의 영향과 잠재적인 위험을 줄이는 데 기여하고 있습니다.



### ResearStudio: A Human-Intervenable Framework for Building Controllable Deep-Research Agents (https://arxiv.org/abs/2510.12194)
Comments:
          EMNLP 2025 Demo, Oral

- **What's New**: 이번 논문에서는 ResearStudio라는 최초의 오픈 소스 프레임워크를 소개합니다. 이 시스템은 실시간으로 인간의 제어를 중심에 두고 있으며, Collaborative Workshop 디자인을 따릅니다. 사용자는 실행 중 언제든지 작업을 일시 정지하고 계획이나 코드를 수정하며, 사용자가 원하는 명령어를 실행할 수 있는 기능이 제공됩니다.

- **Technical Details**: ResearStudio는 투명성과 대칭적 제어, 동적 역할 흐름을 특징으로 하는 디지털 인터페이스를 통해 Collaborative Workshop 개념을 구현합니다. 사용자는 시스템의 모든 단계에서 개입할 수 있으며, 코드나 데이터 파일을 수정하고 전체 작업공간을 다운로드할 수 있습니다. 또한, 이 시스템은 AI 주도와 인간 주도 간의 조화를 통해 연구의 효율성을 높일 수 있습니다.

- **Performance Highlights**: ResearStudio는 GAIA 벤치마크에서 최신 Deep Research 에이전트들과 비교해 뛰어난 성능을 보여주었으며, 이 결과는 강력한 자동화된 성능과 세밀한 인간의 제어가 공존할 수 있음을 입증합니다. 연구 에이전트에 대한 안전하고 보완적인 접근 방식을 촉진하기 위해 지속적으로 업데이트할 예정입니다.



### Evolution of meta's llama models and parameter-efficient fine-tuning of large language models: a survey (https://arxiv.org/abs/2510.12178)
- **What's New**: 이 리뷰 논문에서는 Meta AI의 LLaMA (Large Language Model Meta AI) 시리즈의 빠른 발전을 조사합니다. LLaMA 1부터 LLaMA 4까지의 모델과 이들 모델에 개발된 전문적인 파라미터 효율적 미세조정(PEFT) 방법을 설명합니다. 각각의 LLaMA 모델은 7B에서 288B까지의 다양한 파라미터 수를 가지며, 이들의 아키텍처와 주요 성능 특성을 함께 다룹니다. 또한 PEFT의 개념과 이를 LLaMA에 적용한 다양한 방법들(LoRA, LLaMA-Adapter 등)을 살펴봅니다.

- **Technical Details**: LLaMA 모델은 Transformer 기반의 언어 모델로, 다양한 파라미터 수(7B, 13B, 33B, 65B 등)가 존재합니다. 이들 모델들은 공개된 텍스트 데이터셋을 통해 훈련되었고, 작은 파라미터 수로도 큰 모델과 비교했을 때 성능을 발휘할 수 있음을 보여줍니다. 특히, LLaMA-13B는 GPT-3(175B)보다도 많은 벤치마크에서 우수한 성능을 보였습니다. PEFT 방법은 대부분의 사전 훈련된 모델의 파라미터를 고정시키고, 소수의 어댑터 파라미터를 도입하여 메모리 사용량과 훈련 시간을 크게 줄이며 높은 성능을 유지하는 방법을 제공합니다.

- **Performance Highlights**: LLaMA의 다양한 모델과 PEFT 접근 방식은 실제 환경에서 법률 및 의학과 같은 여러 분야에서 성공적으로 적용되었습니다. LLaMA-2는 발화 대화에 최적화된 채팅 버전을 포함하며, 성능이 개선된 무작위 생성 및 제로샷 능력을 자랑합니다. LLaMA-3 시리즈는 128,000 토큰의 긴 컨텍스트를 지원하며, 이미지와 텍스트 입력을 모두 처리할 수 있는 다중 모드 비전 모델을 제공합니다. 최신 모델들은 미세조정된 LLaMA 모델이 더욱 큰 기본 모델들을 초월할 수 있는 성과를 보여주고 있습니다.



### MatSciBench: Benchmarking the Reasoning Ability of Large Language Models in Materials Scienc (https://arxiv.org/abs/2510.12171)
- **What's New**: 이번 논문에서는 LLM의 과학적 추론 능력이 소재 과학 분야에서 부족하다는 점을 지적하며, 이를 해결하기 위해 MatSciBench라는 새로운 벤치마크를 소개합니다. MatSciBench는 주제별로 1,340개의 문제를 포함하고 있으며, 6개의 주요 분야와 31개의 하위 분야로 세분화되어 있습니다. 이 벤치마크는 문제의 난이도를 3단계로 분류하여 LLM의 추론 능력을 보다 체계적으로 평가할 수 있도록 합니다.

- **Technical Details**: MatSciBench는 물질 과학의 다양한 하위 분야에서의 문제를 평가하기 위해 설계되었습니다. 각 문제는 주어진 질문을 해결하기 위해 필요한 추론의 길이에 따라 난이도가 분류되며, 문제의 50.7%는 쉽고, 29.1%는 중간, 20.1%는 어렵습니다. 또한, 벤치마크는 시각적 맥락을 포함한 315개의 문제를 통해 멀티모달 추론 능력도 평가합니다.

- **Performance Highlights**: 최고 성능의 모델인 Gemini-2.5-Pro가 77%의 정확도를 기록하였지만, 이를 초과하는 모델은 없었습니다. 분석 결과, 사고 모델(thinking models)들은 문제의 난이도에 크게 영향을 받지 않으며, 멀티모달 질문에서 성능 저하를 보이기도 했습니다. 이러한 결과들은 다양한 추론 방법이 각 모델에서의 성능에 미치는 영향을 명확히 보여주며, LLM의 과학적 추론 능력을 향상시키기 위한 기초를 마련해줍니다.



### Precise Attribute Intensity Control in Large Language Models via Targeted Representation Editing (https://arxiv.org/abs/2510.12121)
- **What's New**: 본 연구는 사용자 정의 속성 강도를 거시적으로 제어할 수 있는 방법을 제시합니다. 기존의 방법들은 속성을 단지 방향으로만 움직일 수 있게 했으나, 우리는 이를 타겟 도달 문제로 재구성하여 원하는 속성 강도를 정확하게 맞추는 것을 목표로 합니다. 이 연구는 LLM(GPT) 모델이 특정 속성 강도를 정밀하게 출력할 수 있도록 하는 기법을 제안합니다.

- **Technical Details**: 우리는 세 가지 주요 혁신을 통해 Pre-Control 방법론을 제시했습니다. 첫째, 속성 강도를 극대화하는 대신 특정 목표 값에 도달하는 것을 중심으로 합니다. 둘째, 시차 학습(Temporal-Difference Learning)을 통해 경량 가치 함수(value function)를 훈련하여 부분 생성으로부터 최종 속성 강도 점수를 예측합니다. 셋째, 히든 표현(hidden representation)에서 기울기 기반 개입을 사용하여 목표 속성 강도 방향으로 모델을 정밀하게 조정합니다.

- **Performance Highlights**: 우리의 방식은 LLaMA-3.2-3b 및 Phi-4-mini 모델을 이용한 실험에서 사용자 정의 속성 강도를 높은 정확도로 달성하는 데 성공했습니다. 이는 효율적인 Pareto 전선 근사와 특정 속성 강도를 가진 훈련 데이터 생성을 통한 모델 증류에 활용됩니다. 매우 높은 차원의 속성 공간에서의 압도적인 시간 복잡도 감소가 확인되었고, 이는 다목적 선호 최적화 과정을 실용적으로 만듭니다.



### ToPolyAgent: AI Agents for Coarse-Grained Topological Polymer Simulations (https://arxiv.org/abs/2510.12091)
Comments:
          10 pages, 8 figures

- **What's New**: ToPolyAgent는 자연어 인터페이스를 통해 다양하고 복잡한 topological polymer(위상 폴리머)의 coarse-grained molecular dynamics (MD) 시뮬레이션을 수행할 수 있는 다중 에이전트 AI 프레임워크를 소개합니다. 이 시스템은 대규모 언어 모델(LLM)과 도메인 특화 계산 도구를 통합하여 사용자의 피드백에 따라 시뮬레이션 세팅을 조정할 수 있는 상호작용 모드 및 상세한 프롬프트에 따라 작업을 수행하는 자율 모드를 제공합니다. ToPolyAgent는 다양한 폴리머 아키텍처에 대해 시뮬레이션을 지원하며, 향후 과학적 연구의 접근성과 효율성을 높일 잠재력을 지닙니다.

- **Technical Details**: ToPolyAgent는 상호작용 모드와 자율 모드 두 가지 운영 모드로 작동하며, 각각 Config Agent, Simulation Agent, Report Agent, Workflow Agent의 네 가지 에이전트를 통해 지원됩니다. 상호작용 모드에서는 사용자의 피드백을 받아 초기 폴리머-용매 구성 생성을 담당하는 Config Agent가 있으며, 그 후 Simulation Agent가 시뮬레이션을 수행하고 데이터 분석을 진행합니다. 자율 모드에서는 Workflow Agent가 초기 구성 생성을 포함한 모든 단계를 자동으로 수행하여 보다 구체적인 프롬프트에 따라 시뮬레이션과 데이터 분석을 진행합니다.

- **Performance Highlights**: ToPolyAgent는 다양한 고객 사례를 통해 복잡한 폴리머 시뮬레이션의 실행 장벽을 낮추는 동시에 연구 조수로서의 잠재력을 강조합니다. 예를 들어, 선형 폴리머의 구성에 대한 용매 품질의 영향을 조사하거나, 브러시 폴리머의 지속 길이에 대한 grafting 밀도의 영향을 분석하는 등의 연구를 수행할 수 있습니다. LLM을 통한 자연어 처리와 엄격한 시뮬레이션 방법이 결합되어, 폴리머 과학에서 AI 기반의 자료 연구를 확장할 수 있는 기초를 구축합니다.



### One Life to Learn: Inferring Symbolic World Models for Stochastic Environments from Unguided Exploration (https://arxiv.org/abs/2510.12088)
Comments:
          Project page: this https URL 39 pages

- **What's New**: 본 논문에서는 복잡한 확률적 환경에서 유도없는 탐색을 통해 상징적 세계 모델 세계 모델을 학습하는 OneLife라는 프레임워크를 소개합니다. 전통적인 연구는 충분한 상호작용 데이터를 가지고 결정론적 환경에 집중해왔지만, 우리의 연구에서는 적대적인 환경에서 "한 삶"만으로 탐색해야 하는 도전 과제를 다룹니다. OneLife는 조건부로 활성화되는 법칙을 사용하여 세계의 역동성을 모델링하며, 이를 통해 극히 제한된 상호작용에서도 동적 계산 그래프를 생성하여 효율적인 학습을 구현합니다.

- **Technical Details**: OneLife의 핵심 구성 요소는 법칙 합성기(law synthesizer)와 추론 알고리즘(inference algorithm)입니다. 법칙 합성기는 새로운 법칙을 제안하고, 추론 알고리즘은 관찰값을 통해 법칙의 예측 능력을 재조정합니다. 이러한 방법론은 기울기 기반(graident-based)으로 작동하여 관찰된 변수를 변경하는 법칙만 업데이트하여 학습의 효율성을 높입니다. 이를 통해 OneLife는 복잡한 확률적 사건에 대한 분포를 추론할 수 있습니다.

- **Performance Highlights**: 우리의 프레임워크는 Crafter-OO라는 환경에서 평가되었으며, OneLife는 최소한의 무가이드 상호작용에서도 환경 동적을 성공적으로 학습하여 23개 시나리오 중 16개에서 강력한 기본선 대비 우수한 성과를 보여줍니다. 또한, OneLife는 시뮬레이션 롤아웃을 통해 전략을 식별하는 계획 능력을 갖추고 있습니다. 이를 통해 우리의 작업은 복잡하고未知의 환경에서 프로그램적 세계 모델을 자율적으로 구성하는 기초를 확립합니다.



### Evaluating the Quality of Randomness and Entropy in Tasks Supported by Large Language Models (https://arxiv.org/abs/2510.12080)
- **What's New**: 이번 연구는 대형 언어 모델(LLM)이 무작위성을 포함하는 다양한 작업을 처리할 수 있는 능력을 평가합니다. 실험을 통해 LLM이 무작위 숫자 생성 및 품질 평가에 얼마나 효과적으로 대응할 수 있는지를 살펴보았습니다. 연구 결과, LLM은 무작위성을 모방하는 데 어느 정도 성공하지만, 일관되지 않은 성능과 기대치는 현저히 벗어나는 경향이 있음이 밝혀졌습니다.

- **Technical Details**: 이 논문은 LLM의 무작위성을 요구하는 작업을 처리하는 능력을 측정하기 위해 여러 요소를 고려한 실험 설계를 포함합니다. 예를 들어, 입력된 외부 도구에 대한 접근, 작업 유형, 모델 상태(새로고침 vs 비새로고침), 프롬프트 전략 등이 있습니다. 무작위 품질 평가를 위해 엔트로피(entropy)와 NIST 랜덤성 테스트를 활용하며, LLM의 출력과 기존의 무작위 번호 생성기(PRNG) 및 알고리즘과 비교합니다.

- **Performance Highlights**: 연구 결과, LLM은 고품질 무작위성을 생성하는 데 한계가 있으며, 이는 알고리즘의 내재된 한계 및 잠재적 편향 때문입니다. 외부 도구를 활용할 때 LLM의 성능이 향상되며, 엔트로피 기반 샘플링 및 병렬 사고 직렬(decoding) 접근법이 제안되었습니다. 해당 연구는 LLM의 무작위성 품질을 이해하고 개선하기 위한 전략에 대한 중요한 통찰을 제공합니다.



### BeSTAD: Behavior-Aware Spatio-Temporal Anomaly Detection for Human Mobility Data (https://arxiv.org/abs/2510.12076)
Comments:
          accepted by The 2nd ACM SIGSPATIAL International Workshop on Geospatial Anomaly Detection

- **What's New**: 본 논문은 BeSTAD(Behavior-aware Spatio-Temporal Anomaly Detection for Human Mobility Data)라는 새로운 프레임워크를 제안하여, 대규모 인구 데이터를 기반으로 개별적인 행동 이상을 탐지하는 데 중점을 두고 있습니다. 기존의 이동성 이상 탐지 방법은 주로 통계적 이탈값이나 특정 이동 경로의 시간적 불일치를 찾아내는데 집중했으나, BeSTAD는 개인의 행동 패턴을 정교하게 모델링함으로써 이러한 한계를 극복합니다. 특히 공간적 맥락(spatial context)과 시간적 동적(dynamic)을 동시에 고려하여 개인화된 이상 탐지를 가능하게 합니다.

- **Technical Details**: BeSTAD는 LSTM 기반의 클러스터링 아키텍처를 활용하여 개별 행동 패턴을 학습합니다. 이 프레임워크는 이동 경로(raw trajectories)를 여러 기하학적 특성으로 보강하고, 정교한 모빌리티 표현(mobility representations)을 구성합니다. 또한, BeSTAD는 행동 클러스터 인지를 위한 모델링 메커니즘을 포함하여 정상적인 이동 데이터를 바탕으로 개인화된 행동 프로필을 구축하고, 시간에 따른 행동 변화를 설명하는 일관된 해석을 보장합니다.

- **Performance Highlights**: BeSTAD의 주요 기여 중 하나는 다중 스케일 공간 의미(spatial semantics)를 체계적으로 추출하고 인코딩하여 이동 행동에 대한 맥락적 이해를 향상시킨 점입니다. 또한, 개인화된 행동 모델링을 통해 이상 탐지를 진행하는 새로운 메커니즘을 도입했습니다. 이 연구는 표준화된 인구 집단의 패턴에 과적합되는 기존 방법론의 한계를 극복하고, 대규모 이동성 데이터셋에서 의미 있는 행동 변화를 보이는 개인을 효과적으로 식별할 수 있도록 합니다.



### EmboMatrix: A Scalable Training-Ground for Embodied Decision-Making (https://arxiv.org/abs/2510.12072)
Comments:
          10 pages 8 figures

- **What's New**: 이번 논문에서는 고유한 embodied decision-making을 위한 훈련 기반 인프라인 EmboMatrix를 소개합니다. 이는 대규모 시뮬레이션과 정밀한 보상을 제공하여 LLM이 진정한 embodied 의사결정 능력을 습득할 수 있도록 설계되었습니다. EmboMatrix는 복잡한 작업과 장면을 다루며, 테스트한 EmboBrain-7B 모델은 기존의 DeepSeek-R1 기준선을 9.5% 초과해 성능을 입증했습니다.

- **Technical Details**: Embodied decision-making(구체화된 의사결정) 방식은 고수준 목표를 실행 가능한 행동으로 변환하는 데 필수적입니다. 기존 모델들은 raw sensory inputs를 motor commands로 직접 매핑하거나, 고수준의 모델이 저수준의 조정을 담당하는 계층적 접근 방식을 사용합니다. EmboMatrix는 고급 이전-Language-Action(VLA) 모델을 통합하여 실제 환경과의 상호작용을 통해 embodied 의사결정 능력을 증대하는 것을 목표로 하고 있습니다.

- **Performance Highlights**: EmboBrain-7B는 671B DeepSeek-R1 기준선 대비 9.5% 향상된 수치를 달성했습니다. 이는 환경에 기반한 상호작용 학습이 진정한 지능형 embodied 에이전트를 개발하는 데 유용하다는 것을 보여줍니다. EmboMatrix의 효과적인 시뮬레이션과 풍부한 보상 구조 덕분에, EmboBrain은 다양한 임무에서 견고하고 일반화된 행동을 보일 수 있었습니다.



### HiCoTraj:Zero-Shot Demographic Reasoning via Hierarchical Chain-of-Thought Prompting from Trajectory (https://arxiv.org/abs/2510.12067)
Comments:
          accepted by The 1st ACM SIGSPATIAL International Workshop on Generative and Agentic AI for Multi-Modality Space-Time Intelligence

- **What's New**: HiCoTraj는 레이블 없이도 인구 통계를 유추할 수 있도록 LLMs의 제로샷 학습과 의미 이해 능력을 활용하는 방법론입니다. 이 프레임워크는 이동 경로를 의미적으로 풍부한 자연어 표현으로 변환하고, 체계적이고 투명한 추론 과정을 제공합니다. 또한, 인구 통계 데이터의 부족 문제를 해결하면서 다양한 인구 특성에 대한 일반화 가능성을 높입니다.

- **Technical Details**: HiCoTraj는 두 가지 핵심 구성 요소로 이루어져 있습니다: 컨텍스트 기반 이동 내러티브 생성과 계층적 Chain-of-Thought(코트) 추론입니다. 이 방법은 이동 경로 데이터를 텍스트 형식으로 변환하여 주간 활동 연대기를 생성하고, 세 가지 인지 단계(사실적 특징 추출, 행동 패턴 분석, 인구 통계 유추)를 통해 LLMs를 안내합니다. 이 과정은 훈련이 필요 없으며, 레이블이 있는 데이터의 부족 문제를 극복합니다.

- **Performance Highlights**: 실험 평가 결과, HiCoTraj는 제로샷 시나리오에서도 여러 인구 특성에 대해 경쟁력 있는 성능을 달성하는 것으로 나타났습니다. 이는 LLMs가 이동 경로 데이터로부터 인구 특성을 유추하는 과정에서 더 높은 해석 가능성과 투명성을 제공할 수 있음을 보여줍니다. 이러한 성과는 개인화된 이동 서비스 및 공공 정책에의 활용 가능성을 크게 증가시킵니다.



### AI Agents as Universal Task Solvers (https://arxiv.org/abs/2510.12066)
- **What's New**: 이번 논문은 AI 추론 에이전트를 새로운 관점에서 재정의하며, 전통적인 유도 학습(inductive learning) 대신 전도 학습(transductive learning)의 중요성을 강조합니다. 이를 통해 AI 에이전트가 미리 학습된 모델을 사용하여 새로운 작업을 해결하는 방식에 대한 이론적 근거를 제시하며, 학습의 시간적 역할을 중심에 두고 있습니다. 특히, AI의 효율성을 높이기 위해 모델의 크기보다 해결 시간을 최적화해야 한다는 주장을 하고 있습니다.

- **Technical Details**: 논문은 AI 에이전트를 확률론적 동적 시스템(stochastic dynamical systems)으로 해석하고, 기존의 유니버설 솔버(universal solver)의 개념을 일반적인 맥락으로 확장합니다. 이를 통해 AI 에이전트가 작업에 필요한 추론을 수행하는 방법과 시간을 잘 정의할 수 있는 새로운 개념인 적절한 시간(proper time)을 도입했습니다. 연구는 대형 언어 모델(LLMs)의 체인 오브 사고(chain-of-thought reasoning) 메커니즘을 통해 유니버설 솔버의 가능성을 실험적으로 탐구합니다.

- **Performance Highlights**: 제안된 전도 학습 프레임워크는 기존 유도 학습의 한계를 극복하며, AI 에이전트가 각 작업에서 필요한 추론을 신속하게 수행하도록 지원합니다. 실제로 논문에서는 유니버설 솔버가 과거 데이터를 이용해 추론 속도를 비약적으로 향상시킬 수 있는 방법을 실증적으로 입증합니다. 특히, 데이터 생성 분포의 복잡성에 따라 성능 향상에 한계가 있을 수 있으며, 이는 학습 이론의 전통적인 가정에 도전하는 결과입니다.



### ThinkPilot: Steering Reasoning Models via Automated Think-prefixes Optimization (https://arxiv.org/abs/2510.12063)
- **What's New**: 본 논문에서는 ThinkPilot라는 새로운 훈련 필요 없는 프레임워크를 소개합니다. 이 프레임워크는 LRMs (Large Reasoning Models) 의사결정을 자동으로 최적화하며, 진화적 프로세스를 활용하여 생각 프리픽스(think-prefixes)를 생성합니다. 이 프리픽스는 추론 행동의 분류에 의해 유도되며, 모델의 성능을 향상시키는 데 도움을 줍니다.

- **Technical Details**: ThinkPilot는 두 단계의 작업 흐름으로 운영됩니다: 초기화-평가 단계와 진화-반복 단계입니다. 초기화 단계에서는 LLM (Large Language Model)을 사용하여 각 작업에 대한 시드 프리픽스를 자동 생성하고 평가합니다. 이어지는 진화 단계에서는 생성된 시드 프리픽스를 바탕으로 성능 피드백을 통해 최초의 생각 프리픽스를 개선하는 진화적 알고리즘을 사용합니다.

- **Performance Highlights**: ThinkPilot는 다양한 작업에서 널리 효과적임을 입증했습니다. 모델의 정확도와 길이 간의 무역에서 더 높은 정확도를 얻으며, Safety에서 StrongREJECT 점수를 27.0%에서 0.7%로 감소시켰습니다. 또한, 지침 따르기(Instruction Following)에서도 성능이 증가하여 IFEval 점수를 6.4 포인트 향상시켰습니다.



### Empowering LLM Agents with Geospatial Awareness: Toward Grounded Reasoning for Wildfire Respons (https://arxiv.org/abs/2510.12061)
- **What's New**: 이번 연구에서는 Geospatial Awareness Layer (GAL)라는 혁신적인 구조를 도입하여 대규모 언어 모델(LLM)에게 지구 데이터를 기반으로 한 효율적인 재난 대응 시스템을 제공합니다. GAL은 리얼타임으로 인프라, 인구 통계, 지형 및 기상 정보를 자동으로 검색하고 통합하여, 대규모 언어 모델이 현장 기반의 의사결정을 지원하도록 도와줍니다. 이를 통해, 기존 모델들이 가지던 지리적 무관심과 해석력 부족의 문제를 해결하고자 합니다.

- **Technical Details**: GAL은 주어진 핫스팟 좌표와 시간 정보에 따라 PostGIS-래스터 데이터베이스에서 인프라, 인구, 지형 및 기상 특성을 자동으로 검색해냅니다. 이 정보는 고정된 필드로 구성된 간결한 단위 주석이 달린 인식 스크립트로 인코딩되어, LLM이 쉽게 해석할 수 있도록 변환됩니다. 이 시스템은 결정적인 안정성을 유지하며, 역사적 유사 사례와 일일 변화 신호를 적용하여 실시간으로 반복적인 업데이트가 가능합니다.

- **Performance Highlights**: 실제 캘리포니아의 산불 사건을 평가하여, GAL이 적용된 LLM 에이전트가 기존 모델들보다 더 높은 성능을 발휘함을 확인했습니다. 특히, 지리적으로 기반한 접근이 추론 과정에서 해석 가능성을 제고하고, 운영 결과와의 정렬성을 높이는 데 기여했습니다. 또한, 이 프레임워크는 홍수나 허리케인과 같은 다른 재해에도 일반화될 가능성이 있음을 보여주었습니다.



### Do Large Language Models Respect Contracts? Evaluating and Enforcing Contract-Adherence in Code Generation (https://arxiv.org/abs/2510.12047)
Comments:
          21 pages, 12 figures, 3 tables

- **What's New**: PACT는 계약 준수 평가를 통해 코드의 신뢰성을 높이려는 새로운 프로그램 평가 프레임워크입니다. 기존의 코드 생성 벤치마크는 주로 기능적 정확성만을 평가했으나, 계약 준수 문제를 간과했습니다. 이 프레임워크는 계약을 위반하는 테스트 케이스를 중점으로 삼아 코드 생성 과정에서 계약을 명확하게 확인할 수 있도록 돕습니다.

- **Technical Details**: PACT는 SMT solver를 활용하여 계약 위반 케이스를 효율적으로 생성하고, 다양한 프로프트 조건 하에서 코드 생성의 체계적인 분석을 가능하게 합니다. 이를 통해, 계약 위반 테스트 케이스를 추가한 프로프트가 계약 준수 능력을 상당히 향상시키는 데 기여한다는 것을 보여줍니다. 또한, 계약 준수를 정량화하기 위한 새로운 지표를 도입하여 생성된 코드의 신뢰성을 평가할 수 있는 기반을 제공합니다.

- **Performance Highlights**: 기존 벤치마크는 잘 형성된 입력에 대해서만 올바른 결과를 측정하는 경향이 있어, 실제 소프트웨어의 계약 요구 사항을 무시하는 문제가 있었습니다. PACT는 이러한 문제를 해결하고, 아울러 LLM이 생성한 코드의 기능적 및 계약적 적합성을 평가할 수 있는 정교하고 해석 가능한 지표를 제공합니다. 이로 인해, 더욱 신뢰할 수 있는 코드 생성을 위한 필요성이 강조됩니다.



### CausalTrace: A Neurosymbolic Causal Analysis Agent for Smart Manufacturing (https://arxiv.org/abs/2510.12033)
Comments:
          8 pages, 4 figures, 3 tables, Accepted at AAAI 2026: IAAI - Innovative Applications of AI Conference

- **What's New**: 현재 제조업은 AI 기반 센서, 제어 및 의사결정 지원의 발전에 힘입어 초자율 운영의 시대에 진입하고 있습니다. 그러나 기존의 AI 시스템은 종종 개별적인 블랙 박스(BLACK BOX)로 작동하여 예측, 설명 및 인과 추론을 통합하는 능력이 부족합니다. 이러한 단편화는 안전이 중요한 산업 환경에서 신뢰성과 실제 유용성을 제한합니다. 이를 해결하기 위해 CausalTrace라는 신경 상징적 인과 분석 모듈이 SmartPilot 산업 CoPilot에 통합되어 실시간 의사결정 지원을 제공합니다.

- **Technical Details**: CausalTrace는 산업 온톨로지(ontologies)와 지식 그래프(knowledge graphs)를 활용한 데이터 기반의 인과 분석을 수행합니다. 이 모듈은 인과 발견(causal discovery), 반사실적 추론(counterfactual reasoning), 원인 분석(RCA) 등 여러 고급 기능을 지원하며, 실시간 작업자 상호작용이 가능합니다. CausalTrace는 복잡한 AI 평가 방법과 C3AN 프레임워크를 사용하여 평가를 실시하였으며, 이는 신뢰성, 지능 및 견고함의 원칙을 포괄합니다.

- **Performance Highlights**: CausalTrace는 학문적 로켓 조립 테스트베드에서 도메인 전문가와의 높은 합의를 달성하였으며(ROUGE-1: 0.91), RCA 성능에서도 뛰어난 결과를 보였습니다(MAP@3: 94%, PR@2: 97%, MRR: 0.92, Jaccard: 0.92). 또한 C3AN 평가에서 4.59/5의 점수를 기록하여 실시간 배치에 대한 정밀도와 신뢰성을 입증하였습니다. 이러한 성과들은 CausalTrace가 실제 제조 환경에서 활용될 준비가 되어 있음을 강조합니다.



### Asking Clarifying Questions for Preference Elicitation With Large Language Models (https://arxiv.org/abs/2510.12015)
- **What's New**: 이번 연구에서는 사용자 선호를 명확히 하는 순차적 질문을 생성하기 위해 새로운 접근 방식을 제안합니다. 사용자의 프로필에서 시작하여 두 단계로 진행되는 프로세스를 통해 정보가 삭제되면서 질문이 생성되는 방식입니다. 이 방법은 LLM(대형 언어 모델)을 훈련시켜 사용자의 필요를 더 잘 이해하고 그에 기반한 맞춤형 추천을 제공할 수 있도록 합니다.

- **Technical Details**: 연구에서 제안된 모델은 두 단계의 프로세스를 따릅니다. 첫 번째 단계는 사용자 프로필을 구조화된 형식으로 정리한 후, 각 정보에 대한 질문을 생성하는 '포워드 프로세스'이며, 두 번째 단계는 그 질문들의 답변을 기반으로 '리버스 프로세스'에서 사용자의 프로필을 재구성하는 것입니다. 이러한 접근 방식을 통해 LLM이 유용한 질문을 생성할 수 있도록 훈련됩니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 사용자 선호를 효과적으로 이끌어내는 질문을 생성하는 LLM의 능력을 크게 향상시켰습니다. 이 모델은 사용자 프로필 복원에 있어 더욱 효과적인 질문을 생산할 수 있게 되었으며, 질문의 흐름이 점점 구체적으로 접근하도록 학습하여 보다 개인화된 추천을 제공할 수 있습니다.



### CGBench: Benchmarking Language Model Scientific Reasoning for Clinical Genetics Research (https://arxiv.org/abs/2510.11985)
Comments:
          Accepted at NeurIPS 2025

- **What's New**: 이번 연구에서는 클리닉 유전체학에서의 증거 선별 및 유전자 해석을 위한 새롭고 강력한 벤치마크인 CGBench를 소개합니다. CGBench는 임상 유전학에서의 전문가 큐레이션 문헌을 기반으로 하여 과학적 출판물에 대한 언어 모델(LM)의 추론 능력을 평가합니다. 이를 통해 LMs가 어떤 방식으로 실험 결과를 추출하고, 증거의 강도를 판단하며, 실험의 관련 결과를 설명할 수 있는지를 측정합니다.

- **Technical Details**: CGBench는 ClinGen의 증거 저장소(Evidence Repository)에서 소스된 데이터로 구축되었으며, 이는 유전자 및 변이 주석의 수천 가지 큐레이션을 포함합니다. 이 벤치마크는 LMs가 복잡한 증거 합성 작업을 수행하는 능력을 평가하기 위해 설계된 여러 가지 임무와 방법론을 제공합니다. 연구에서 8개의 서로 다른 LMs를 시험했으며, 모델들이 약점과 강점을 보이는 특정 패턴을 발견했습니다.

- **Performance Highlights**: LM은 복잡한 과학적 작업에서 여러 가지 한계를 보였으며, 특히 미세한 지시사항에 대한 해석에서 큰 차이를 보였습니다. 추론 모델은 관련 증거 추출에서 비추론 모델보다 우수한 성과를 보였지만, 강한 증거와 약한 증거를 구별하는 데는 비교적 어려움을 겪었습니다. 연구 결과는 LMs가 문헌 해석에서의 인간 설명과 비교했을 때 종종 잘못된 해석이나 환각을 일으킬 수 있음을 보여주며, 향후 임상 유전학 연구에 AI를 활용하는 길을 열어주고 있습니다.



### Holistic Agent Leaderboard: The Missing Infrastructure for AI Agent Evaluation (https://arxiv.org/abs/2510.11977)
- **What's New**: AI 에이전트는 코딩에서 고객 서비스까지 복잡한 실세계 작업을 수행하기 위해 개발되었습니다. 하지만 AI 에이전트의 평가에는 여러 가지 도전 과제가 있어 실제 성능에 대한 이해를 저해하고 있습니다. 이에 대한 해결책으로 Holistic Agent Leaderboard (HAL)를 소개합니다.

- **Technical Details**: 우리는 평가 시간을 주에서 시간으로 단축하고 일반적인 구현 버그를 제거하는 표준화된 평가 하네스를 제공합니다. 모델, 스캐폴드(scaffold), 벤치마크를 아우르는 3차원 분석을 수행하였으며, 9가지 모델과 9가지 벤치마크에 대해 21,730개의 에이전트 롤아웃을 실시했습니다. 이 과정에서 총비용은 약 40,000달러가 소요되었습니다.

- **Performance Highlights**: 분석 결과는 흥미로운 통찰을 제공하며, 대부분의 경우 높은 추론 노력이 정확도를 떨어뜨린다는 사실을 보여줍니다. LLM(대형 언어 모델)을 활용한 로그 검토를 통해 이전에 보고되지 않았던 행동도 드러났습니다. 이러한 모든 에이전트 로그는 총 25억 개의 토큰으로 구성되며, 에이전트 행동에 대한 추가 연구를 촉진할 것으로 기대합니다.



### Beyond Consensus: Mitigating the Agreeableness Bias in LLM Judge Evaluations (https://arxiv.org/abs/2510.11822)
- **What's New**: 최근에 발표된 이 논문은 LLM(대형 언어 모델)을 평가자로 활용하는 기존 방법의 단점을 분석하고, LLM의 신뢰성 측정에서 나타나는 고유한 편향을 실증적으로 보여줍니다. 특히 LLM은 유효한 출력을 96%의 높은 정확도로 식별할 수 있지만, 유효하지 않은 출력의 식별은 25% 미만으로 매우 낮다는 점에 주목했습니다. 이로 인해 기존 평가 방식의 신뢰성이 과대 평가될 수 있으며, 새로운 전략을 제안하고 있습니다.

- **Technical Details**: 논문에서는 새로운 데이터셋과 주관적 평가를 위한 코드가 제공되며, 향상된 방법론인 소수 반대(veto) 앙상블 전략이 도입되었습니다. 이 전략은 데이터 품질 문제에 강한 내성을 가지며, 단순한 다수결보다 우수한 성능을 보입니다. 또한, 회귀 기반 방법론을 사용해 소량의 인간 주석 데이터로 평가자의 편향을 보정함으로써 1.2%의 예측 오차를 달성하였습니다.

- **Performance Highlights**: 저자들은 366개의 고등학교 코딩 프로그램을 다룬 과제에서 새로운 회귀 접근 방식이 가장 성능이 우수한 앙상블 모델보다 2배 개선된 성과를 내었다고 발표했습니다. 이 방법 덕분에 LLM을 평가하는 과정에서 발생할 수 있는 편향을 감소시킬 수 있으며, 이는 알고리즘의 신뢰도 및 정확도를 향상시키는 데 기여할 것으로 기대됩니다.



### AI Agents for the Dhumbal Card Game: A Comparative Study (https://arxiv.org/abs/2510.11736)
Comments:
          10 pages, 7 figures, 6 tables

- **What's New**: 본 연구는 Dhumbal이라는 문화적으로 중요한 멀티플레이어 카드 게임을 위한 인공지능(AI) 에이전트의 효율성을 평가하는 체계적인 비교를 진행합니다. Dhumbal의 게임 메커니즘을 형식화하고 다양한 에이전트를 구현하였으며, 강화 학습(learning-based) 방법인 Deep Q-Network(DQN) 및 Proximal Policy Optimization(PPO)과 같은 기술들이 포함되었습니다. 이 연구는 AI의 전략적 의사결정을 위한 새로운 통찰력을 제공하며, 전통 게임의 디지털 보존을 지원합니다.

- **Technical Details**: Dhumbal 게임은 2명에서 5명이 플레이하며, 전략적 의사결정 및 불완전한 정보가 결합된 카드 게임입니다. 본 논문에서는 규칙 기반(rule-based), 탐색 기반(search-based), 학습 기반(learning-based) 에이전트의 성능을 비교하였으며, 랜덤 기준선(random baseline)도 포함합니다. 논문은 경제적 성과, 승률, Jhyap 선언 성공률, 결정의 효율성 등을 측정하는 rigor한 평가 방법을 사용하여, 에이전트의 성능을 통계적으로 분석하였습니다.

- **Performance Highlights**: 1024회의 시뮬레이션 라운드 결과에서 규칙 기반의 공격적(Aggressive) 에이전트가 88.3%의 승률을 기록하며 가장 뛰어난 성능을 보였습니다. 이는 ISMCTS(9.0%) 및 PPO(1.5%)를 압도하는 결과입니다. 이 연구는 AI 전략의 효과성을 분석하고 기존의 게임 AI 연구에 기여하며, 향후 유사한 전통 게임 연구를 위한 재현 가능한 프레임워크를 제공합니다.



### DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving (https://arxiv.org/abs/2510.12796)
- **What's New**: DriveVLA-W0라는 새로운 훈련 패러다임을 제안하여, 드라이브 환경의 동작을 예측하기 위해 세계 모델링을 활용합니다. 이 접근법은 희소한 주의 신호를 보완하고, 모델이 외부 환경의 역학을 적절히 학습하도록 강요함으로써, 이미지 예측을 통한 밀집 자가 감독 신호를 생성합니다. 이로 인해 VLA 모델의 성능이 크게 향상되어 다양한 운전 데이터 규모에서 그 효과를 입증하였습니다.

- **Technical Details**: DriveVLA-W0는 두 가지 주요 VLA 아키텍처에 대해 구현되었으며, 이 중 하나는 이산 비주얼 토큰을 사용하는 자가 회귀 세계 모델이고, 다른 하나는 연속 비주얼 기능을 사용하는 확산 세계 모델입니다. 이러한 세계 모델링 기법은 모델이 이미지 예측할 때마다 밀집한 감독 신호를 생성하고, 이를 통해 풍부한 세계 표현을 학습하게 합니다. 실시간 배포를 위해 경량화된 MoE 기반의 액션 전문가를 도입하여 추론 지연 시간을 기존 VLA 모델의 63.1%로 줄였습니다.

- **Performance Highlights**: 대규모 70M 프레임 데이터셋을 활용한 실험에서 세계 모델링이 데이터 스케일링 법칙을 강화하는데 기여함을 확인했습니다. 또한, 일관된 일반화를 통해 다양한 도메인 간의 전이 가능한 비주얼 표현을 학습하는 데 도움이 됩니다. 이 연구는 액션 디코더를 위한 흥미로운 성능 트렌드 반전을 발견하는데, 복잡한 플로우 매칭 디코더가 작은 데이터셋에서는 이점을 가지나, 대규모에서는 더 간단한 자가 회귀 모델이 최고의 성능을 발휘하는 것으로 나타났습니다.



### CuMPerLay: Learning Cubical Multiparameter Persistence Vectorizations (https://arxiv.org/abs/2510.12795)
Comments:
          Appears at ICCV 2025

- **What's New**: CuMPerLay는 Cubical Multiparameter Persistence (CMP)를 딥러닝 파이프라인에 통합할 수 있는 새로운 차별화된 벡터화 레이어입니다. 기존 CMP는 이미지의 위상적으로 작업하는 자연스러운 방법을 제공하지만, 다중 필터 구조의 복잡성과 CMP의 벡터화로 인해 사용에 제한이 있었습니다. CuMPerLay는 이러한 문제를 해결하기 위해 신속한 벡터화 알고리즘을 제시합니다.

- **Technical Details**: CuMPerLay는 CMP를 개별적이고 학습 가능한 단일 파라미터 지속성의 조합으로 분해합니다. 이때, 바이필터 함수(bifiltration functions)는 함께 학습되며, 이는 딥러닝 모델에서의 차별화(differentiability)를 통해 강력한 위상적 특징 벡터를 제공할 수 있게 합니다. 또한, 일반화된 Wasserstein 메트릭스에서 벡터화의 안정성에 대한 이론적 보증을 제시합니다.

- **Performance Highlights**: 의료 이미지 분석과 컴퓨터 비전 데이터셋에 대한 실험 결과, CuMPerLay는 제한된 데이터 시나리오에서 특히 분류(classification)와 분할(segmentation) 성능에서 이점이 있음을 보여줍니다. 따라서 CuMPerLay는 구조화된 이미지 분석을 위한 딥 네트워크에 글로벌 구조 정보를 통합하는 유망한 방향을 제공합니다.



### UniFusion: Vision-Language Model as Unified Encoder in Image Generation (https://arxiv.org/abs/2510.12789)
Comments:
          Project page at this https URL

- **What's New**: 본 논문에서는 UniFusion이라 명명된 새로운 확산 기반 생성 모델을 소개합니다. 이 모델은 고정된 대형 비전-언어 모델(VLM)을 통합 멀티모달 인코더로 활용하여 이미지와 텍스트의 별도를 없앴습니다. 특히, Layerwise Attention Pooling (LAP) 기법을 통해 고수준의 의미 및 저수준의 세부 정보를 효과적으로 추출하여, 이미지 생성 및 편집 작업에서 우수한 성능을 발휘합니다.

- **Technical Details**: UniFusion의 핵심 요소인 LAP는 여러 VLM 레이어에서 정보를 수집하며, 이는 고유한 정밀도 및 높은 수준의 의미 추출을 가능하게 합니다. 또한 VLM이 생성한 텍스트 토큰에만 기반한 Diffusion Transformer (DiT)의 조건부 생성을 위한 VLM-Enabled Rewriting Injection with Flexible Inference (VERIFI) 기법을 제안하여, 다양한 프롬프트 형식 간의 분포 이동을 줄여줍니다. 이러한 설계로 인해 UniFusion은 복수의 이미지를 참조하더라도 높은 일반화 능력을 지니게 되었습니다.

- **Performance Highlights**: UniFusion은 단일 이미지 편집 작업에서 훈련된 모델이 다중 참조 이미지에도 제로샷으로 일반화되는 뛰어난 성능을 보여줍니다. 본 연구는 텍스트-이미지 생성 및 편집 작업에서 경쟁력 있는 성능을 발휘하며, 특별한 감독 학습이나 강화 학습 없이도 가능함을 입증합니다. UniFusion의 경우, 이미지 편집 작업에서 학습할 때 텍스트-이미지 프롬프트 준수 및 미적 품질이 크게 향상되는 긍정적인 크로스 태스크 이왕 전이 현상도 보여줍니다.



### MVP4D: Multi-View Portrait Video Diffusion for Animatable 4D Avatars (https://arxiv.org/abs/2510.12785)
Comments:
          18 pages, 12 figures

- **What's New**: MVP4D 모델은 단일 참조 이미지와 목표 표현에 기반하여 디지털 인간의 애니메이션 가능한 다중 시뷰 비디오를 생성합니다. 기존 방법론과 비교할 때, 이 모델은 360도 시점 변화를 통해 수백 개의 프레임을 동시에 생성합니다. 이 과정에서 현실감, 시간적 일관성, 그리고 3D 일관성을 크게 향상시킵니다.

- **Technical Details**: MVP4D는 최첨단 비디오 확산 모델(video diffusion model)을 기반으로 하여, 모핑 가능한 다중 시뷰 비디오 확산 모델(General Multi-View Video Diffusion Model)로 설계되었습니다. 이 모델은 참조 이미지로부터 입력을 받으며, 학생 측 자세와 표현, 그리고 카메라 파라미터에 따른 조건을 설정합니다. 이를 통해 3D-영상 일관성을 유지하는 다중 시점 비디오를 생성하며.

- **Performance Highlights**: MVP4D는 단일 확산 샘플링 실행에서 최대 400개의 프레임을 생성할 수 있으며, 이는 기존의 방법들보다 한층 더 개선된 성능을 제공합니다. 이 접근 방식은 높은 해상도와 리얼리즘을 유지하며, 실시간 렌더링이 가능하여 다양한 애플리케이션에 활용될 수 있습니다.



### Dr.LLM: Dynamic Layer Routing in LLMs (https://arxiv.org/abs/2510.12773)
Comments:
          17 pages, Under submission

- **What's New**: 최근의 연구는 대형 언어 모델(LLMs)이 모든 토큰을 변환기의 모든 층을 통해 처리하는 방식에서 오는 비효율성을 해결하기 위한 접근을 제시합니다. 특히, 동적으로 층을 라우팅하는 방식을 도입하여 사전 학습된 모델이 각 층의 경량 라우터를 사용하여 블록을 건너뛰거나 실행 또는 반복하도록 할 수 있습니다. 이로 인해 성능 저하 없이 효율성을 개선할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 이 연구에서 제안하는 방식은 몬테 카를로 트리 탐색(Monte Carlo Tree Search, MCTS)을 사용하여 고품질의 층 구성을 도출합니다. 설계에는 클래스 불균형 및 긴 시리즈에 강한 풀링 방법과 포컬 손실(focal loss) 및 병목 MLP 라우터가 포함되어 있어 라우팅의 안정성을 보장합니다. 라우터는 확실한 감독 하에 학습되며, 이는 모델의 기본 가중치를 변경하지 않고도 성능을 향상시키는 데 기여합니다.

- **Performance Highlights**: ARC(논리)와 DART(수학) 데이터셋에서 모델은 평균적으로 5개의 층을 절약하면서 정확도를 최대 +3.4%포인트 향상시킵니다. 이 라우터는 도메인이 다른 작업에 대해서도 0.85%의 정확도만 감소시키며 효율성을 유지하고, 이전의 라우팅 방법보다 최대 +7.7%포인트 더 뛰어난 성능을 보여줍니다. 따라서 이 연구는 예산을 고려한 높은 정확도의 추론을 가능하게 합니다.



### Uncertainty Matters in Dynamic Gaussian Splatting for Monocular 4D Reconstruction (https://arxiv.org/abs/2510.12768)
Comments:
          Project page: this https URL

- **What's New**: 이 연구는 동적인 3D 장면을 단안 입력(monocular input)으로 재구성하는 과정에서 발생하는 모호성을 해결하기 위해 신뢰성을 고려한 동적 Gaussian Splatting(USplat4D) 프레임워크를 제안합니다. 이는 관찰이 반복되는 Gaussian을 신뢰할 수 있는 기준점으로 활용하여 더 안정적인 움직임 추적과 향상된 4D 재구성을 가능하게 합니다.

- **Technical Details**: USplat4D에서는 각 Gaussian의 시간에 따른 불확실성을 추정하고, 이를 바탕으로 스페이셜-템포럴 그래프(spatio-temporal graph)를 구성합니다. 이 그래프는 노드의 중요성, 엣지 구성, 가중치 조정에 불확실성을 반영하여, 안정적인 장면 재구성을 도모합니다. 또한, 이 방법론은 기존의 동적 Gaussian Splatting 파이프라인에 통합될 수 있도록 설계되었습니다.

- **Performance Highlights**: 다양한 실제 및 합성 데이터셋을 통해 USplat4D의 성능을 검증한 결과, 불확실성을 명시적으로 모델링함으로써 동적 Gaussian Splatting 모델의 안정성이 향상되고, occlusion(가림) 상황에서도 높은 품질의 새로운 시점 시합(synthesis)을 생성할 수 있음을 보여주었습니다. 특히 극단적인 시점에서의 성능 향상이 두드러지며, 향후 다양한 어플리케이션에서 활용될 가능성이 큽니다.



### Disentangling Neurodegeneration with Brain Age Gap Prediction Models: A Graph Signal Processing Perspectiv (https://arxiv.org/abs/2510.12763)
Comments:
          Accepted for publication in IEEE Signal Processing Magazine

- **What's New**: 이번 논문에서는 신경변성(neurodegeneration) 질환과 건강한 노화의 차이를 평가하는 새로운 생체 표지자로서 뇌 나이 차이(brain age gap) 예측 모델에 대해 다룹니다. 이 모델은 신경영상 데이터(neuroimaging data)를 통해 예측된 개인의 뇌 나이와 실제 연령 사이의 차이를 추정합니다. 저자들은 최근 발전된 그래프 신호 처리(graph signal processing) 기법을 활용하여 이 모델을 구축하는 방법론을 제시합니다.

- **Technical Details**: BAGP(Brain Age Gap Prediction) 모델은 구조적 MRI(structural MRI) 데이터를 분석하여 신경변성 마커를 식별하고, 이를 통해 가속화된 노화(accelerated aging) 현상을 평가합니다. 특히, VNN(coVariance neural network)을 도입하여 해부학적 공분산 행렬(anatomical covariance matrices)을 활용하여 강력한 예측력을 확보합니다. 이 모델은 그래프 신경망(graph neural networks)을 기반으로 하여 다양한 임상 집단에서의 일반화 가능성을 높이는 데 중점을 두고 있습니다.

- **Performance Highlights**: BAGP 모델은 신경변성 질환의 진행 및 중증도를 예측하는 데 대한 유용성이 여러 연구에서 입증되었습니다. 연구들은 VNN을 통해 구조적 MRI 데이터를 활용한 결과, 신경변성 사례에서의 뇌 나이 차이가 건강한 집단에 비해 높음을 보여주었습니다. 하지만, 이 모델의 활용은 여전히 다양한 임상적 한계가 존재하며, 실질적인 진료에서의 적용성을 높이기 위한 추가 연구가 필요합니다.



### VQArt-Bench: A semantically rich VQA Benchmark for Art and Cultural Heritag (https://arxiv.org/abs/2510.12750)
- **What's New**: 이번 연구에서는 VQA(Visual Question Answering) 벤치마크의 한계를 지적하며, 복잡한 예술 분야와 문화 유산 도메인을 위한 VQArt-Bench라는 새로운 대규모 VQA 벤치마크를 제안합니다. 이 벤치마크는 시각적 이해의 다양한 차원을 탐구할 수 있도록 설계된 질문을 생성하는 다중 에이전트 파이프라인을 통해 구성되었습니다. 기존의 규칙 기반 접근법의 한계를 극복하고, 예술 작품에 대한 보다 깊이 있는 질문 생성을 목표로 합니다.

- **Technical Details**: 제안된 VQArt-Bench의 핵심은 이미지 캡션을 분석하여 질문 카테고리를 알아내고, 이러한 카테고리를 바탕으로 복잡하고 개방적인 질문을 생성하는 Topic Selector, Question Generator, Question Refiner, Judge의 4가지 에이전트로 구성된 프로세스입니다. 이 과정에서 각 질문은 확인된 순서로 비슷한 오답도 포함하여 도전적인 선택형 형식으로 변환됩니다. 모든 질문은 Judge에 의해 검토되어 비트리비얼하고, 정확하게 해답될 수 있으며, 언어적으로도 올바른지를 확인합니다.

- **Performance Highlights**: 14개의 최첨단 MLLM 모델을 새롭게 제안된 VQArt-Bench로 평가한 결과, 현재 모델들이 간단한 카운팅 작업에서도 놀라운 약점을 보였으며, 독점 모델과 오픈 소스 모델 간의 명확한 성능 차이가 발견되었습니다. 이러한 결과는 VQA 모델들이 단순히 통계적 패턴에 의존하기보다 진정한 시각적 분석을 수행하는 데 필요한 시각적 추론 능력을 평가하는 것이 얼마나 중요한지를 강조합니다.



### Hey, wait a minute: on at-issue sensitivity in Language Models (https://arxiv.org/abs/2510.12740)
Comments:
          10 pages, 5 figures, 3 tables. See this https URL for code and data

- **What's New**: 이번 연구에서는 언어 모델(LM)에서 대화의 자연스러움을 평가하기 위한 새로운 방법론을 제시합니다. 기존의 평가 기준에 대한 한계를 극복하기 위해 언어학적 개념인 'at-issueness'를 활용하며, 'Divide, Generate, Recombine, and Compare (DGRC)'라는 접근법을 도입합니다. DGRC는 대화를 여러 부분으로 나누고, LM을 사용하여 각 부분의 후속발화를 생성한 후, 다시 결합하고 이들 시퀀스의 가능성을 비교하는 절차를 따릅니다.

- **Technical Details**: 이 연구의 주요 방법론인 DGRC는 먼저 초기 발화(UU)를 두 개의 독립적인 발화로 나누고, 각 발화에 대해 LM으로부터 다양한 응답을 생성합니다. 이어서 독립적으로 생성된 발화들을 재결합하고, 생성된 응답의 가능성(log probabilities)을 비교하여 모델의 선호도를 정량화합니다. 이를 통해 at-issue(content that advances discourse)와 not-at-issue(content that does not change the conversational trajectory) 간의 민감도를 분석합니다.

- **Performance Highlights**: DGRC 방법을 적용한 결과, LM은 발화의 at-issue 콘텐츠에 대해 더 높은 반응 확률을 보인다는 것을 발견했습니다. 특히, instruct-tuned 모델에서 이러한 경향이 더 강하게 나타났으며, 특정 구문이 있을 경우(at-issue preference를 줄이는데 기여하는 cue) 이러한 경향이 감소함을 확인했습니다. 이러한 결과는 DGRC의 유연하고 세밀한 방법론이 LM 대화 동태 평가에 있어 효과적임을 보여줍니다.



### HYPE: Hybrid Planning with Ego Proposal-Conditioned Predictions (https://arxiv.org/abs/2510.12733)
- **What's New**: 이 논문에서는 HYPE(HYbrid Planning with Ego proposal-conditioned predictions)라는 새로운 계획 접근 방식을 제안합니다. 이 계획자는 다중 모드의 경로 제안(multi-modal trajectory proposals)을 통합하여, Monte Carlo Tree Search (MCTS)로 수정하는 혁신적인 방법을 제공합니다. HYPE는 기존의 방법론이 가지고 있던 복잡한 비용 함수 설계 문제를 단순화하며, 일관되게 씬을 인식할 수 있는 장애물 예측 모델을 도입하였습니다.

- **Technical Details**: HYPE는 멀티모달 제안 모델(multi-modal proposal model)을 기반으로 하여, 제안된 경로를 통해 향후 씬의 변화를 예측하고, 이를 MCTS에 반영하여 계획을 최적화합니다. 이 과정에서 경로 추적을 위한 경량화된 격자 기반 비용 함수(grid-based cost function)가 사용되어, 거의 모든 수동 조정을 최소화하고 있습니다. 본 논문은 대규모 실제 데이터셋인 nuPlan과 DeepUrban에서 성능 검증을 통해 안전성(safety)과 적응성(adaptability)의 최신 성과를 입증하였습니다.

- **Performance Highlights**: HYPE는 안전성과 적응성 측면에서 기존의 최첨단 모델들보다 뛰어난 성능을 보여주었습니다. 본 연구의 결과는 HYPE가 복잡한 도시 환경에서 안전하고 해석 가능하며 상호 작용을 고려한 계획을 지속적으로 생성할 수 있도록 도와줌을 입증합니다. 또한, 제안한 접근 방식은 다양한 시나리오에서도 일반화(generalization) 능력을 상당히 향상시킵니다.



### Hierarchical Federated Learning for Crop Yield Prediction in Smart Agricultural Production Systems (https://arxiv.org/abs/2510.12727)
Comments:
          6 pages, 3 figures, conference

- **What's New**: 논문에서는 스마트 농업 생산 시스템 및 작물 수확 예측을 위해 설계된 새로운 계층적 연합 학습 아키텍처를 제안합니다. 농장이 작물 특정 클러스터에 가입하는 계절 구독 메커니즘을 도입하여, 각 농장이 특정 계절에 따라 동적으로 참여할 수 있도록 합니다. 이 아키텍처는 개별 스마트 농장, 작물 특화 집계기, 그리고 전 세계 모델 집계기로 구성된 세 개의 계층을 포함하여, 데이터 프라이버시를 보호하면서 통신 오버헤드를 줄이는 방향으로 설계되었습니다.

- **Technical Details**: 제안된 시스템은 각 농장이 작물 특정 클러스터에 동적으로 가입하고, 클러스터 내의 고객들이 특정 작물 유형에 맞춘 모델을 협력하여 훈련하는 방식으로 작동합니다. 이러한 계층적 모델 집계 프로세스는 작물 전문화 및 전 세계 일반화를 동시에 이룰 수 있도록 하여 농업 데이터의 다양성과 동적 특성을 반영합니다. 최종적으로, 각 클러스터의 모델들은 중앙 서버에서 집계되어 보다 포괄적인 글로벌 모델이 작성됩니다.

- **Performance Highlights**: 실험 결과, 제안된 시스템의 로컬 및 작물 계층 모델이 실제 수확 패턴을 잘 따르며, 일반적인 기계 학습 모델보다 현저하게 우수한 성능을 보임을 보여주었습니다. 이는 농업 맥락에서 계층적 연합 학습의 장점을 확인시켰으며, 특히 이질적인 농업 환경과 개인 정보가 민감한 농업 데이터에 적합함을 입증했습니다.



### Artificial intelligence for simplified patient-centered dosimetry in radiopharmaceutical therapies (https://arxiv.org/abs/2510.12714)
- **What's New**: 이 논문은 방사성 의약품 치료(Radiopharmaceutical Therapy, RPT)에서 개인화된 환자 중심의 선량 측정(dosimetry)의 필요성을 강조합니다. 인공지능(Artificial Intelligence, AI)이 현재 선량 측정 계산의 주요 한계를 극복할 수 있는 해결책을 제시하고 있습니다. 특히, 환자 친화적인 RPT를 위한 단순화된 선량 측정에서 AI의 주요 발전을 검토합니다.

- **Technical Details**: 논문에서는 AI 기술이 선량 측정의 정확도를 높이기 위해 어떻게 활용될 수 있는지에 대한 기술적 세부 사항을 제공합니다. 이를 통해, 환자의 신체 특성 및 생리적 반응을 반영한 맞춤형 치료 접근 방식이 가능해진다는 점을 다룹니다. 또한, 환자 친화적인 치료를 위한 선량 산출의 새로운 방법론에 대해 설명합니다.

- **Performance Highlights**: AI를 활용한 최근의 연구 결과를 통해 선량 측정의 개선된 성능을 요약합니다. 특히 환자 맞춤형 치료에서의 실질적인 이점과 AI의 적용이 가져오는 혁신적 변화에 대해 논의합니다. 미래의 RPT 선량 측정에서 AI의 역할이 더욱 중요해질 것이라고 예측합니다.



### Beyond Seeing: Evaluating Multimodal LLMs on Tool-Enabled Image Perception, Transformation, and Reasoning (https://arxiv.org/abs/2510.12712)
- **What's New**: 이번 연구에서는 Multimodal Large Language Models (MLLMs)가 사용자 제공 이미지의 불완전성을 해소하기 위한 새로운 접근법을 제시합니다. 기존의 정적 이미지 접근 방식에서 벗어나, 이미지를 능동적으로 조작하고 다른 도구와 통합하여 복잡한 작업을 해결하는 'think with images' 패러다임을 도입합니다. 이를 위해 IRIS라는 새로운 벤치마크를 발표하여 MLLMs의 시각적-텍스트적 작업 수행 능력을 평가합니다.

- **Technical Details**: IRIS는 1,204개의 도전적인 오픈 엔디드 비전 작업을 포함하고 있으며, 단일 턴(603개)과 멀티 턴(601개)으로 나눠져 있습니다. 이러한 작업은 다섯 가지 다양한 분야를 아우르며, 체계적인 평가를 위한 자세한 루브릭을 제공합니다. 평가 결과, 현재 MLLMs는 비전과 일반 도구의 효과적인 통합이 필요한 작업에서 어려움을 겪고 있으며, 가장 강력한 모델인 GPT-5-think 역시 단지 18.68%의 통과율을 기록했습니다.

- **Performance Highlights**: IRIS를 통해 얻어진 평가에서 OpenAI 모델은 다양한 이미지 조작을 통해 이점을 보이는 반면, Gemini-2.5-pro는 개선을 보이지 않는다는 점이 관찰되었습니다. 이는 MLLMs의 비주얼 인텔리전스를 향상시키기 위한 중요한 통찰을 제공합니다. 'think with images' 중심의 벤치마크를 도입함으로써, 비전 및 인지 작업의 새로운 방향성을 제시하고 있습니다.



### Hybrid Explanation-Guided Learning for Transformer-Based Chest X-Ray Diagnosis (https://arxiv.org/abs/2510.12704)
Comments:
          Accepted by iMIMIC at MICCAI 2025

- **What's New**: 본 연구에서는 Hybrid Explanation-Guided Learning (H-EGL) 프레임워크를 제안하여, self-supervised와 human-guided 제약 조건을 결합하여 attention 정렬을 개선하고 일반화를 향상시킵니다. 이 접근 방식은 높은 비용의 수동 감독 없이도 효과적인 모델 교육을 가능하게 하여 깊이 있는 이해를 제공합니다. H-EGL은 Vision Transformer (ViT) 기반의 chest X-ray 분류 작업에서 기존의 두 가지 state-of-the-art 방법을 초월하는 성능을 보여주었습니다.

- **Technical Details**: H-EGL은 Discriminative Attention Learning (DAL)이라는 self-supervised 모듈과 human alignment 중앙화를 중점적으로 둔 supervised 모듈로 구성됩니다. 이 프레임워크는 semi-supervised 채널을 통해 특징을 강조하며, 각 입력에 대한 attention 양식이 클래스 간 구별을 늘리도록 설계되었습니다. DAL은 class-distinctive attention maps를 생성하고, 비슷한 두 attention maps 간의 cosine similarity를 최소화하기 위한 손실 함수를 적용합니다.

- **Performance Highlights**: 모델의 성능은 chest X-ray 이미지에서 네 가지 일반적인 흉부 병리(무기폐, 심비대, 응집 및 흉수)의 분류로 평가되었습니다. H-EGL은 성능 평가 결과, 두 개의 기존 방식보다 우수한 분류 정확도를 달성하였으며, 각 병리의 전문가 알라인먼트를 통해 보다 잘 정렬된 attention maps를 생성했습니다. 또한, 모델의 견고성과 일반화를 평가하기 위해, validation 및 test 세트 간의 성능 차이도 측정되었습니다.



### Beyond Postconditions: Can Large Language Models infer Formal Contracts for Automatic Software Verification? (https://arxiv.org/abs/2510.12702)
Comments:
          under submission

- **What's New**: 이 논문에서는 자동 소프트웨어 검증의 맥락에서 자연어로부터 명시적 사양을 추출하는 문제를 다시 다룹니다. 특히, 자연어를 공식적인 기능 계약(formal functional contracts)으로 변환하는 NL2Contract 작업을 소개합니다. 이를 통해 자동 소프트웨어 검증에 사용될 수 있는 명세의 품질을 향상시키고자 합니다.

- **Technical Details**: NL2Contract는 LLMs(대규모 언어 모델)를 이용하여 비공식적인 자연어를 공식적 기능 계약으로 변환하는 과정입니다. 이 연구에서는 생성된 계약의 유효성(soundness), 버그 판별력(bug discriminative power), 그리고 자동 소프트웨어 검증에서의 유용성을 평가하기 위한 지표(metrics)를 제시합니다. 평가 과정에서 LLM의 성능을 기존의 postcondition 생성 작업(nl2postcond)과 비교합니다.

- **Performance Highlights**: 실험 결과, LLM은 모든 가능한 입력에 대해 유효한 기능 계약을 생성하는 데 효과적임을 보였습니다. 생성된 계약은 버그가 있는 동작과 올바른 동작을 구별하는 데 충분히 표현력이 있으며, LLM이 추론한 기능 계약을 제공한 검증기는 단순히 postcondition을 제공한 경우보다 잘못된 경고(false alarms)가 적었습니다. 또한 LLM이 추론한 전제조건(preconditions)은 개발자의 의도와 잘 맞아떨어져서 실제-world의 버그를 포착하는 데 효과적임을 보여주었습니다.



### Topological Signatures of ReLU Neural Network Activation Patterns (https://arxiv.org/abs/2510.12700)
- **What's New**: 이번 논문은 ReLU 활성화 패턴의 위상적 서명을 탐구하며, feedforward neural networks의 폴리토프 분해를 분석하여 네트워크의 성능과 결정 경계의 상관관계를 확인하는 내용을 다룹니다. 연구자들은 Fiedler partition과 이중 그래프의 관계를 조사하고 있으며, 입력 공간의 폴리헤드 분해를 통해 분류 및 회귀 작업에서 성능을 해석하고 있습니다. 이러한 접근은 모델의 훈련 동안 결정 경계와 내부 위상 표현 간의 관계를 명확히 합니다.

- **Technical Details**: 연구지는 ReLU 활성화 함수를 가진 (L+1)-layer feedforward neural network(피드포워드 신경망)의 구조를 바탕으로 하며, 폴리토프 분해를 사용하여 입력 데이터에 대해 평균적인 위상적 서명을 도출합니다. 이를 통해 그래프 라플라시안(graph Laplacian)을 활용하여 결정 경계를 반영하는 방법을 제안하며, 해당 모델에서 사용되는 Fiedler vector와 렙레네 점 (eigenvalue) 분석에 초점을 맞추고 있습니다. 특히 각 그래프의 연결 요소 수를 세는 다차원 핵(Laplace kernel)의 차원에 대한 논의가 포함되어 있습니다.

- **Performance Highlights**: 실험 결과에 따르면, 피드포워드 신경망의 Fiedler partition은 이중 분류 문제에서 네트워크의 결정 경계를 반영하는 것으로 나타났습니다. 그러나 가중치가 없는 라플라시안에서는 결정 경계와의 연관성이 낮아졌으며, 최종적으로 가중치 조정을 통해 보다 정확한 결정 경계 반영이 가능하다는 결과를 얻었습니다. 이러한 발견은 훈련 손실(training loss)과 Betti numbers 간의 상관관계를 확립하여, 훈련 과정 중 모델의 동적 변화와 하모닉 구조를 명확히 밝혀냅니다.



### Generation Space Size: Understanding and Calibrating Open-Endedness of LLM Generations (https://arxiv.org/abs/2510.12699)
- **What's New**: 본 논문에서는 다양한 오픈 엔디드 생성(task) 작업이 서로 다른 출력 다양성 요구를 가진다는 점을 강조합니다. 현재의 LLMs는 종종 지나치게 동질적인(output homogenous) 결과를 생성하거나, 사실 기반 작업에서 다양하지만 부정확한 응답을 산출하는 문제를 겪고 있습니다. 연구자들은 이러한 실패 모드가 모두 효과적인 생성 공간 크기(GSS)와 관련이 있음을 주장하며, 새로운 평가 프레임워크인 GSSBench를 소개하여 GSS의 정확도를 평가하고 있습니다.

- **Technical Details**: GSS는 특정 프로프트(prompt)에 대해서 모델이 고려하는 의미적으로 distinct한 출력의 집합입니다. 논문에서는 모델의 GSS가 원하는 GSS와 얼마나 다르게 설정되어 있는지를 평가하는 시스템적인 접근 방식을 제안합니다. 이를 위해 연구자들은 알려진 GSS 관계를 가진 프로프트 쌍(pairs)을 사용하여 GSS를 측정하고, 차이를 최소화하는 방법을 제시하였습니다.

- **Performance Highlights**: GSSBench를 통해 연구자들은 여러 모델의 GSS를 평가하였으며, 특히 EigenScore와 같은 환각 탐지(hallucination detection) 메트릭이 모델의 GSS를 가장 잘 근사하는 결과를 보였다고 보고합니다. 각기 다른 메트릭이 모델의 생성 동작을 이해하는데 도움을 주며, GSS 측정을 통해 질문 명확화, 추론 모델의 과도한 사고 및 결정을 최적화하는 데 유용함을 밝혔습니다.



### Who is a Better Matchmaker? Human vs. Algorithmic Judge Assignment in a High-Stakes Startup Competition (https://arxiv.org/abs/2510.12692)
Comments:
          17 Pages, 2 figures

- **What's New**: 이 논문에서는 인공지능(AI)을 활용하여 복잡한 의사결정 작업의 자동화 및 지원을 모색하는 새로운 접근법을 제시합니다. 특히, Harvard President's Innovation Challenge와 같은 실제 환경에서의 판사 배정 문제를 다루며, 이 문제의 해결을 위한 새로운 AI 기반 알고리즘인 Hybrid Lexical-Semantic Similarity Ensemble (HLSE)를 소개합니다. HLSE는 전문적인 판단 수준의 매칭 품질을 달성하면서도 효율성과 확장성을 제공합니다.

- **Technical Details**: HLSE는 세 가지 텍스트 표현 방식을 통합하여 판사와 벤처 간의 유사성을 계산하는 앙상블 모델입니다. 이 모델은 희소 TF-IDF 벡터, 밀집 변환기 임베딩, 혼합 TF-IDF 가중치 임베딩을 결합하여 유사성을 추정합니다. 연구팀은 HLSE를 사용하여 유사성 점수를 계산하고, PeerReview4All 알고리즘을 통해 매치를 배정했습니다.

- **Performance Highlights**: 연구 결과, HLSE와 인간 전문가에 의한 매칭 간의 품질 차이는 없었으며, HLSE의 평균 매칭 품질 점수는 3.90으로 인간의 3.94와 유사했습니다. 또한, 이전에 1주일이 소요되던 수작업 매칭 과정을 알고리즘이 몇 시간 안에 처리함으로써 자동화 가능성을 보여주었습니다. 이러한 결과는 AI 기반 솔루션이 고위험 환경에서의 인간 의사결정을 지원하고 향상시킬 수 있는 가능성을 강조합니다.



### DiffEM: Learning from Corrupted Data with Diffusion Models via Expectation Maximization (https://arxiv.org/abs/2510.12691)
- **What's New**: 이 연구에서는 손상된 데이터를 이용하여 확산 모델(diffusion models)을 학습하는 새로운 방법인 DiffEM을 제안합니다. 이 방법은 Expectation-Maximization (EM) 알고리즘을 활용하여 손상된 관측값으로부터 깨끗한 데이터를 재구성하고, 이를 통해 확산 모델을 정제하는 과정을 포함합니다. 또한, DiffEM은 이론적으로 수렴 보장(monotonic convergence guarantees)을 제공하여 향후 연구의 기초를 다지고 있습니다.

- **Technical Details**: DiffEM의 핵심 아이디어는 사전 분포(prior distribution)를 학습하는 대신 조건부 확산 모델(conditional diffusion model)을 사용하여 후방 분포(posterior distribution)를 직접 모델링하는 것입니다. 이 접근 방식은 특정한 후방 샘플링 방식(approximate posterior sampling schemes)과 독립적이며, 어떤 손상 채널(corruption channel)도 처리할 수 있는 장점을 가지고 있습니다. 이 모델은 정량화하기 어려운 구조적 가정을 피하고, 재구성된 데이터로 모델을 정제하는 과정에서 발생하는 오류에 대한 이론적 분석도 포함되어 있습니다.

- **Performance Highlights**: DiffEM의 효과는 다양한 이미지 재구성 작업을 통해 실험적으로 입증되었습니다. 연구팀은 CIFAR-10과 CelebA와 같은 데이터 세트를 활용하여 저차원 매니폴드 학습(low-dimensional manifold learning)과 같은 다양한 유형의 손상에 대해 평가를 진행했습니다. 이 연구는 손상된 관측값만으로도 효과적으로 확산 모델을 훈련할 수 있는 길을 열어줄 것으로 기대됩니다.



### From Delegates to Trustees: How Optimizing for Long-Term Interests Shapes Bias and Alignment in LLM (https://arxiv.org/abs/2510.12689)
- **What's New**: 이 연구는 대리 모델(delegate model)과 신탁 모델(trustee model)의 설계를 통해 인간 선호를 대표하는 AI 시스템의 잠재력을 탐구합니다. 특히 모델이 표현된 선호를 반영할 것인지, 또는 더 넓은 이해를 바탕으로 행동할 것인지에 대한 설계의 무역(trade-off)을 강조하고 있습니다. 이는 사용자에게 단기적인 선호를 맞추는 것보다 장기적인 이익을 고려해야 하는지에 대한 중요한 논의를 포함합니다.

- **Technical Details**: 이 연구는 두 가지 정책 문제 카테고리를 통해 모델 예측을 비교합니다. 첫 번째는 전문가 컨센서스가 강한 주제로, 두 번째는 의견이 분분한 논란의 여지가 있는 주제입니다. 연구는 짧은 시간과 긴 시간의 이익을 고려한 시간적 유틸리티 모델을 적용하여 결과를 비교하며, 모델의 기본 가정으로 인한 체계적 편향도 조사합니다.

- **Performance Highlights**: 연구 결과, 신탁 모델은 오랜 이해당사자의 합의에 더 closely align된 결정들을 산출하지만, 명확한 합의가 없는 문제에서는 더 큰 편향이 나타나는 경향이 있습니다. 특히 공화당원과 저소득 유권류의 프로필에서 이러한 경향이 두드러지며, 대형 모델에서 더욱 명확히 나타납니다. 이 발견은 AI 시스템 디자인 시 인간의 이익을 어떻게 제대로 표현할 수 있는지에 대한 중요한 질문을 제기합니다.



### Demystifying Hybrid Thinking: Can LLMs Truly Switch Between Think and No-Think? (https://arxiv.org/abs/2510.12680)
Comments:
          10 pages, 6 figures

- **What's New**: 이번 연구는 하이브리드 사고(hybrid thinking) 모델들의 제어 가능성을 향상시키기 위한 실질적인 훈련 방법을 제시합니다. 연구 결과, 기존 하이브리드 사고 모델들이 완전한 모드 분리를 이루지 못하고 있음을 발견하였고, 이로 인해 사고 모드(no-think)에서 여전히 추론 행동이 누수되고 있음을 강조했습니다. 이는 현재의 하이브리드 사고 훈련 방식에 대한 한계점을 드러내며, 향후 훈련 전략을 개선하는 데 기여할 수 있는 방향을 제시합니다.

- **Technical Details**: 하이브리드 사고는 주어진 프롬프트에 제어 토큰(예: 
o_think, 	hink)을 추가하여 구현됩니다. 연구진은 140,000개의 샘플을 포함한 대규모 데이터를 활용해 효과적인 제어를 위한 요인들을 분석하였으며, 적절한 비율의 노-싱크(no-think) 데이터와 두 단계(training) 훈련 전략을 통해 제어 능력을 개선할 수 있음을 입증했습니다. 특히, 훈련 목표에 따라 순차적으로 사고 데이터(after thinking)와 하이브리드 훈련을 적용했을 때 더 강력한 제어가 가능하다고 밝혔습니다.

- **Performance Highlights**: 실험 결과, 제안된 훈련 레시피에 따르면 하이브리드 모델은 두 모드 모두에서 정밀도를 유지하면서도, 노-싱크 모드에서의 출력 길이를 1,085에서 585로 감소시켰습니다. 또한, 'wait'와 같은 추론 지지 토큰의 발생 건수도 5,917에서 522로 줄였습니다. 이러한 성과들은 제어 가능성을 개선하면서도 성능을 희생하지 않는 방법을 제시하지만, 현재 하이브리드 사고의 한계를 강조하는 결과이기도 합니다.



### SG-XDEAT: Sparsity-Guided Cross-Dimensional and Cross-Encoding Attention with Target-Aware Conditioning in Tabular Learning (https://arxiv.org/abs/2510.12659)
- **What's New**: 이번 연구에서는 테이블 데이터를 위한 새로운 프레임워크인 SG-XDEAT(Sparsity-Guided Cross Dimensional and Cross-Encoding Attention with Target Aware Conditioning)를 제안합니다. SG-XDEAT는 두 가지 병렬 표현인 원시 값 스트림과 타겟 조건화된 스트림으로 입력 피처를 분해하는 이중 스트림 인코더를 활용합니다. 이 모델은 주의 기반 모듈을 통해 피처 간 및 인코딩 간 종속성을 포착하는 기법을 통합합니다.

- **Technical Details**: SG-XDEAT는 세 가지 주요 구성 요소로 인해 고유합니다: (i) Cross-Dimensional self-attention은 각 스트림 내 피처 간의 내부 뷰 종속성을 포착하고, (ii) Cross-Encoding self-attention은 원시 및 타겟 인식 표현 간의 양방향 상호 작용을 가능하게 하며, (iii) Adaptive Sparse Self-Attention(ASSA) 메커니즘은 낮은 유틸리티 토큰의 주의 가중치를 0으로 구동하여 노이즈의 영향을 완화합니다. 이러한 구조를 통해 SG-XDEAT는 더 강력한 심층 테이블 학습기를 만들어냅니다.

- **Performance Highlights**: 다양한 공공 벤치마크에서 수행된 실험 결과는 SG-XDEAT가 강력한 기준 모델들에 비해 일관된 성능 향상을 보여준다고 보고했습니다. 이 모델은 원시 및 타겟 인식 뷰를 공동으로 모델링하면서 노이즈를 적응적으로 필터링하여 안정적인 성능을 달성합니다. 이러한 개선은 심층 학습 모델과 GBDT(Gradient-Boosted Decision Trees) 간의 격차를 효과적으로 줄이는 데 기여합니다.



### Reasoning Pattern Matters: Learning to Reason without Human Rationales (https://arxiv.org/abs/2510.12643)
Comments:
          Submitted to Frontiers of Computer Science

- **What's New**: 이번 연구에서는 Large Language Models(LLMs)의 초기 훈련 단계인 Supervised Fine-Tuning(SFT) 과정에서 고품질의 rationale(합리적 근거) 주석을 비용 효율적으로 줄이는 방법을 탐구합니다. 연구팀은 패턴화된 추론 과제(patterned reasoning tasks)라 불리는 새로운 문제 범주를 정의하여, 일관된 추론 패턴을 따르는 과제에서 성공의 주된 원인은 이러한 패턴을 모델이 내재화하는 능력에 있다는 주장을 합니다. 이를 통해 차량 촬영을 통해 추론을 지원할 수 있는 Pattern-Aware LLMs as Rationale Annotators(PARO) 프레임워크를 제안합니다.

- **Technical Details**: 연구에서는 패턴화된 추론 과제의 정의와 특성을 명확히 하고, 이 과제가 어떻게 일관된 절차적 해결 전략을 따르는지를 설명합니다. 기존 SFT와 RLVR의 두 단계 훈련 방식에서, SFT는 인간 주석이 달린 질문-합리적 근거-답변 삼중항을 통해 모델을 초기화하고, RLVR은 검증 가능한 보상을 통해 모델을 최적화하는 과정을 따릅니다. 이 연구는 패턴화된 추론 과제가 어떻게 서로 다른 인스턴스의 내용을 유지하면서도 동일한 추론 패턴으로 구성될 수 있는지를 자세히 논의합니다.

- **Performance Highlights**: PARO 프레임워크를 통해 생성된 합리적 근거는 기존 사람 주석 데이터셋보다 10배 큰 데이터셋과 비교해도 동등한 성능을 보여줍니다. 이러한 결과는 대규모 인간 주석 데이터 수집을 LLM 기반 자동 주석으로 대체할 수 있는 가능성을 제시하며, 추론 패턴에 대한 제한적인 인간 감독만으로도 효율적인 결과를 도출할 수 있음을 나타냅니다. 연구의 결론은 패턴화된 추론 과제에서 고품질의 합리적 근거를 수집하는 것이 핵심 문제가 아니며, 명확한 추론 패턴 정의 및 강제를 통해 성능을 유지할 수 있다는 것입니다.



### Aixel: A Unified, Adaptive and Extensible System for AI-powered Data Analysis (https://arxiv.org/abs/2510.12642)
- **What's New**: Aixel은 AI 기반 데이터 분석을 위한 통합된 적응형 시스템으로, 데이터 관리와 학습을 결합하여 정확도, 지연 시간 및 비용 요구 사항을 충족합니다. 기존의 데이터베이스 시스템이 AI 모델 관리에 대한 원활한 지원을 제공하지 못하는 문제를 해결하고자 합니다. Aixel은 작업, 모델, 데이터의 네 가지 계층으로 작업을 구성하여 사용자 의도를 효율적으로 캐치하고 최적화된 실행을 제공합니다.

- **Technical Details**: Aixel의 시스템 설계는 사용자 친화성, 적응성, 효율성 및 확장성의 네 가지 핵심 원칙에 기반하고 있습니다. 작업 계층은 사용자가 의도를 선언하는 인터페이스를 제공하며 이를 실행 계획으로 변환합니다. 모델 계층은 버전 관리 및 모델 관리를 지원하며, 데이터 계층은 통합된 데이터 관리 기능을 제공하여 효율적인 데이터 접근을 도와줍니다.

- **Performance Highlights**: Aixel은 데이터와 모델을 통합함으로써 데이터 분석의 전반적인 워크플로우를 최적화하고, 사용자가 ML 작업을 직접 지원할 수 있는 환경을 제공합니다. 사용자 친화적인 디자인을 통해 복잡한 상호 작용을 줄이고, 변화하는 데이터 조건에 적응하며, 높은 성능을 유지합니다. 또한, Aixel의 설계는 향후 연구 기회를 통해 시스템의 효율성과 적응성을 높일 수 있는 가능성을 강조합니다.



### Laminar: A Scalable Asynchronous RL Post-Training Framework (https://arxiv.org/abs/2510.12633)
- **What's New**: 이번 논문에서는 Large Language Models (LLMs)의 포스트 훈련 후효율성을 향상시키기 위한 새로운 접근법인 Laminar를 제안합니다. 기존의 제약된 Reinforcement Learning (RL) 시스템의 비효율성을 극복하기 위해, 각 trajectory를 독립적으로 생성하고 소비할 수 있는 구조로 설계되었습니다. 이는 GPU 자원의 활용도를 극대화하고 훈련 속도를 개선하기 위한 중요한 통찰력으로, 최신 기술들의 속도를 넘어서는 성능 향상을 제공합니다.

- **Technical Details**: Laminar는 actor 모델과 rollout 복제본 간의 데이터 및 파라미터 종속성을 해소하는 완전 분리된 아키텍처에 기반합니다. 이 시스템은 relay workers를 통해 비동기적이고 세밀한 weight synchronization을 제공하며, rollout들이 언제든지 최신 가중치를 호출할 수 있습니다. 또한, long-tail trajectory 문제를 해결하기 위한 dynamic repack 메커니즘을 도입하여, 자원이 부족한 rollout에서 long-tail trajectories를 집중시킴으로써 높은 생성 처리량을 유지할 수 있습니다.

- **Performance Highlights**: 실험 결과, Laminar는 최신 RL 시스템들에 비해 최대 5.48배의 훈련 처리량 향상을 달성했습니다. 이러한 성과는 모델 수렴 시간도 단축시키며, 장기 실행 작업의 안정성을 높이는데 기여합니다. 이 연구는 특히 비효율적인 long-tail trajectory 생성 문제를 해결하는 데 큰 의의가 있습니다.



### Designing Tools with Control Confidenc (https://arxiv.org/abs/2510.12630)
- **What's New**: 이 연구는 도구 설계에 대한 기존 접근법의 한계를 극복하고자 합니다. 구체적으로, 로봇의 작업 조건에 적합한 자율 손 도구 설계를 위한 최적화 프레임워크를 정의합니다. 또한, 도구의 강인성을 높이기 위해 신경영감을 받은 제어 신뢰도(control confidence) 항을 최적화 루틴에 도입합니다.

- **Technical Details**: 이러한 최적화는 로봇 팔을 이용한 철저한 시뮬레이션을 통해 이루어집니다. 제어 신뢰도를 목표 함수로 설정하여 설계된 도구는 환경의 불확실성에 대해 더욱 강인함을 보입니다. 이는 기존의 순수 정확도 기반 접근법과는 다른 방식으로, 도구 사용 시 발생할 수 있는 불확실성에 대한 내성을 제공합니다.

- **Performance Highlights**: 연구 결과, 최적화된 도구는 제어 변동(control perturbations) 하에서도 강인성과 목표 정확도의 균형을 잘 유지합니다. 또한, CMAES 기반의 진화적 최적화 전략이 다른 상태의 최적화 기법보다 우수한 성능을 발휘하여 최적 도구 설계를 최소한의 반복(iteration)으로 달성함을 보여줍니다.



### Learning-To-Measure: In-context Active Feature Acquisition (https://arxiv.org/abs/2510.12624)
- **What's New**: 이 논문에서는 기존의 Active Feature Acquisition (AFA) 방법의 한계를 극복하기 위해 메타-AFA 문제를 공식화합니다. AFA는 테스트 샘플을 위해 피쳐를 적응적으로 선택함으로써 모델 성능을 향상시키는 문제입니다. 저자들은 Learning-to-Measure (L2M)이라는 새로운 접근법을 제안하며, 이는 관측하지 않은 작업에 대한 신뢰할 수 있는 불확실성 정량화와 조건부 상호 정보량을 극대화하는 불확실성 유도 그리디 피쳐 획득 에이전트로 구성됩니다.

- **Technical Details**: L2M은 두 단계로 구성됩니다: 첫 번째 단계는 결측치가 있는 작업을 통해 목표 변수의 예측 불확실성을 정량화하는 사전 훈련, 두 번째 단계는 예측 불확실성을 감소시키는 피쳐를 그리디하게 획득하는 정책 네트워크를 메타 훈련하는 것입니다. 이 방법은 직접 시퀀스 모델링을 통해 실질적인 정보 획득 전략을 제공하며, 매 단계에서 불확실성을 고려하여 최적의 피쳐를 선택합니다. L2M은 결측치가 존재하는 데이터를 직접 처리하며, 매 작업마다 재훈련할 필요가 없습니다.

- **Performance Highlights**: L2M은 다양한 크기와 결측치 수준을 갖는 작업에서 과제 특화 기준선과 비교하여 동등하거나 이를 초과하는 성능을 보입니다. 특히, 레이블이 부족한 경우와 높은 결측치 상황에서 효과적으로 작동합니다. 실험적으로 최적의 불확실성 추정 및 결정 메커니즘을 제공함으로써 AFA의 전반적인 효율을 개선합니다.



### Rethinking Knowledge Distillation: A Data Dependent Regulariser With a Negative Asymmetric Payoff (https://arxiv.org/abs/2510.12615)
Comments:
          45 pages, 24 figures and 104 tables

- **What's New**: 이 연구는 Knowledge Distillation(KD)의 압축 능력과 기능적 관점에서의 지식 전달을 정량화합니다. 기존에 KD가 지식 전달 메커니즘으로 작동한다고 간주되었던 점을 도전하여, KD가 실질적으로 모델을 압축하는 방식이 아닌 데이터 종속적 정규화 역할을 수행함을 주장합니다. 또한, KD의 부정적인 비대칭 지식 전이 현상을 규명하여 안전성 문제를 제기합니다.

- **Technical Details**: 이번 연구는 KD의 핵심 메커니즘을 격리하는 방법론을 제시하며, 3,900개 이상의 모델을 훈련하여 7개의 데이터셋, 9개의 아키텍처, 3개의 모달리티(이미지, 오디오, 언어)를 포함한 다중 실험 세트를 통해 결과를 입증합니다. KD의 효과를 평가하기 위해 Activation Distance, Rank Disagreement, Prediction Disagreement 등 여러 기능적 유사성 지표를 사용하여, 학습된 학생 모델이 교사 모델과 얼마나 기능적으로 유사한지를 측정합니다.

- **Performance Highlights**: KD는 통계적으로 유의미한 기능적 유사성을 발생시키지만, 데이터셋과 모달리티에 따라 이 유사성은 일관되지 않으며 종종 미미합니다. 무작위 제어 증류(Random Control Distillation) 실험에서 정확도와 손실이 가장 크게 개선되는 경향을 보이며, 이는 지식 전달이 성공적으로 이루어지지 않았음을 시사합니다. 또한, 중요한 지식 전달이 이루어질 경우 전이된 지식은 교사의 잘못된 예측에 비대칭적으로 편향되어 안전성에 대한 우려를 낳습니다.



### StyleDecipher: Robust and Explainable Detection of LLM-Generated Texts with Stylistic Analysis (https://arxiv.org/abs/2510.12608)
- **What's New**: 최근의 대규모 언어 모델(LLM)의 발전은 텍스트 생성에 놀라운 진전을 가져왔으나, 이로 인해 기계 생성 텍스트 검출의 필요성이 대두되었습니다. 본 연구에서는 StyleDecipher라는 새로운 프레임워크를 제안하여 스타일적 차이를 정량화함으로써 LLM 생성 텍스트의 검출을 강화합니다. 이 시스템은 고유한 스타일 레벨의 분기를 포착함으로써 설명 가능하고 강력하며 도메인에 구애받지 않는 검출을 가능하게 합니다.

- **Technical Details**: StyleDecipher는 이산 스타일 지표와 연속적인 스타일 표현을 통합하여 스타일의 일관성을 측정하는 방식으로 LLM 생성 텍스트 검출을 재정의합니다. 특정 텍스트를 이산 및 연속 스타일 특징 세트로 인코딩하는 방법을 사용하며, 의미를 보존하는 재작성 모델을 적용해 스타일 특성이 어떻게 변화하는지를 관찰합니다. 이로 인해 자료세트 특정 신호가 아닌 근본적인 스타일 행동에 의해 결정 경계를 정의할 수 있습니다.

- **Performance Highlights**: 다양한 도메인에 걸친 광범위한 실험에서 StyleDecipher는 기존 기준선보다 36.30% 이상 뛰어난 성능을 보이며, 적대적 조작 및 혼합된 인간-AI 콘텐츠에 대해서도 강인성을 유지합니다. 또한, 정량적 및 정성적 분석을 통해 스타일 신호가 기계 생성 텍스트를 구분짓는 명확한 증거를 제공함을 입증했습니다. 이 연구는 기존 방법론에 비해 강력한 검출 정확도와 설명 가능성을 달성하는 것을 목표로 합니다.



### SMILE: SeMantic Ids Enhanced CoLd Item Representation for Click-through Rate Prediction in E-commerce SEarch (https://arxiv.org/abs/2510.12604)
- **What's New**: SMILE(Semantic ID-based item representation enhancement)는 기존의 콜드스타트 문제를 해결하기 위해 고안된 혁신적인 접근 방식입니다. 이 방법은 RQ-OPQ 인코딩을 활용해 아이템의 콘텐츠와 협력 정보를 융합하여 보다 정확한 추천을 제공합니다. 또한, 아이템의 고유한 특성을 반영하는 차별화된 정보를 학습하는 OPQ 인코딩 기반 전략을 처음으로 제안하고 있습니다.

- **Technical Details**: SMILE 프레임워크는 두 개의 주요 모듈, 즉 RQ 인코딩과 아이템 ID 간의 적응형 전송 및 정렬 메커니즘과 OPQ 인코딩에 기반한 아이템 차별 정보 학습 모듈로 구성됩니다. RQ 인코딩은 공통된 의미 정보를 가지고 있으며, OPQ 인코딩은 아이템 별로 차별화된 기능을 캡처합니다. 이러한 방법론은 CTR(click-through rate) 예측 및 사용자 행동 예측을 위한 협업 신호 전송을 가능하게 합니다.

- **Performance Highlights**: SMILE 방식은 대규모 산업 데이터셋에서 실시한 종합적인 오프라인 실험을 통해 그 우수성을 입증했습니다. 온라인 A/B 테스트에서도 아이템 CTR은 +1.66%, 구매자 수는 +1.57%, 주문량은 +2.17% 증가하는 통계적으로 유의미한 개선 결과를 보였습니다. 이로 인해 SMILE은 콜드스타트 문제를 효과적으로 해결할 수 있는 잠재력을 발휘하고 있습니다.



### Reasoning in the Dark: Interleaved Vision-Text Reasoning in Latent Spac (https://arxiv.org/abs/2510.12603)
- **What's New**: 이번 연구에서는 Interleaved Vision-Text Latent Reasoning (IVT-LR) 방법을 제안하여 시각적 및 텍스트 정보 간의 추론을 잠재 공간(latent space)에서 통합적으로 수행할 수 있게 하였습니다. 이는 기존의 멀티모달 추론 방법이 필요로 했던 명시적인 텍스트나 이미지 생성을 없애고, 효율적인 데이터 처리 및 추론을 가능하게 합니다. IVT-LR은 이전 단계의 히든 상태를 사용하여 잠재 텍스트와 선택된 이미지 임베딩을 결합하는 방법으로 진행됩니다.

- **Technical Details**: IVT-LR 프레임워크에서 각 추론 단계는 잠재 텍스트와 잠재 비전의 두 부분으로 구성됩니다. 잠재 텍스트는 이전 단계의 히든 상태를 기반으로 하며, 잠재 비전은 어텐션 점수에 따라 선택된 이미지 임베딩을 포함합니다. 이 과정을 통해 명시적 추론 단계를 점진적으로 잠재적인 추론 단계로 대체할 수 있도록 훈련시킵니다.

- **Performance Highlights**: M3CoT와 ScienceQA 데이터셋에 대한 실험 결과 IVT-LR 방법이 평균 5.45%의 정확도 향상을 보인 동시에, 기존 방법들에 비해 5배 이상의 속도 증가를 기록했습니다. 이러한 결과는 IVT-LR이 멀티모달 추론에서의 새로운 최첨단 성능을 달성했음을 보여줍니다.



### Evaluation of Real-Time Preprocessing Methods in AI-Based ECG Signal Analysis (https://arxiv.org/abs/2510.12541)
Comments:
          Conference paper for 2025 IEEE World AI IoT Congress (AIIoT), FACE Project, University of Siegen, Germany

- **What's New**: 본 논문에서는 휴대용 ECG 시스템의 인기와 프라이버시를 준수하는 실시간 분석에 대한 수요 증가를 강조하며, 엣지 컴퓨팅 (edge computing)과 클라우드 컴퓨팅 (cloud computing)의 강점을 결합한 혁신적인 머신 러닝 솔루션 개발을 목표로 하는 FACE 프로젝트를 소개합니다. 데이터 수집 지점에서 신호 처리 접근 방식을 새롭게 설정할 필요성을 언급하고, 에너지 효율성과 실시간 처리 기능이 중요한 기준이 됨을 강조합니다.

- **Technical Details**: 이 연구에서는 ECG 신호의 다양한 전처리 방법을 분석하고, 필터링, 처리 효율성, 실시간 처리 성능과 현재 표준 전처리 기술의 단점을 체계적으로 비교합니다. Butterworth 필터와 같은 다양한 밴드 패스 필터가 제시되며, 특정 필터링 방법이 전력 소모를 최소화하고 처리 효율성을 극대화할 수 있음을 보여줍니다. 예를 들어, Pan-Tompkins 알고리즘에서 제안된 방법은 멀티플라이어 대신 전력 소모가 적은 곱셈기를 사용하는 접근 방식으로 최적화되었습니다.

- **Performance Highlights**: 전처리 단계에서의 연구 결과는 필터링의 정확성이 상당히 높으며, 일반적인 밴드 패스 필터의 경우 94.8%에서 100%까지 정확도가 범위로 나타났습니다. 또한, 특정 필터링 방법의 처리 시간과 메모리 사용 효율이 기존의 방법보다 개선된 것으로 나타났습니다. 전반적으로 본 연구는 ECG 신호 분석을 위한 에너지 효율적인 전처리 방법의 잠재력을 강조하며, 실시간 응용 프로그램에서 효과적인 성능을 발휘할 수 있는 방법을 제시합니다.



### Unconditional Human Motion and Shape Generation via Balanced Score-Based Diffusion (https://arxiv.org/abs/2510.12537)
- **What's New**: 이번 연구에서는 score-based diffusion model을 통해 무조건적인 인간 모션 생성을 가능케 하며, 기존의 보조 정규화 손실(auxiliary regularization losses)을 필요로 하지 않는 방법을 제안합니다. SMPL 기반 모션 표현을 최소화하여 운동 데이터를 효율적으로 모델링할 수 있는 접근법을 제공합니다. 이러한 방식은 데이터의 다양한 표현에 의한 과도한 복잡성을 제거하고, 각각의 성분에 대한 이론적 기반을 제공합니다.

- **Technical Details**: 제안한 방법은 표준 L2 score-matching 손실에 대한 체계적인 가중치 부여와 특징 그룹 간의 세심한 정규화(normalization)를 결합하여 구현되었습니다. 실험적으로 고안된 방법들은 각 구성 요소가 독립적으로 효과적임을 보여주는 목표 지향적 ablation 실험을 포함하고 있습니다. 이 연구에서 우리는 SMPL 매개변수에 대한 구조 보존(feature normalization) 기능을 보호하며, L2 score-matching 손실의 이론적 동기를 제시합니다.

- **Performance Highlights**: 성능 테스트 결과, 제안된 방법은 단 31개의 신경 함수 평가(neural function evaluations)로도 기존의 최첨단 성능과 동등한 결과를 달성할 수 있음을 보여줍니다. 또한 무조건적인 인간 모션 확산 훈련을 통해 손실 가중치의 실험적 조정 없이도 효과적인 샘플링과 가능성을 해결할 수 있는 PF-ODE와의 호환성을 확보합니다. 이 방식은 직접적인 형상(generation) 생성을 통해 관절에서의 후속 복구(procedure recovery)를 제거하는 장점을 가지고 있습니다.



### BoN Appetit Team at LeWiDi-2025: Best-of-N Test-time Scaling Can Not Stomach Annotation Disagreements (Yet) (https://arxiv.org/abs/2510.12516)
- **What's New**: 이번 논문에서는 Test-time scaling 기법을 LeWiDi-2025 과제에 적용하여 주석 불일치(annotation disagreement)를 평가하고자 한다. 기존의 Test-time scaling 기법이 수학과 코딩 등의 확정적인 응답을 요구하는 도메인에 국한되어 있었던 반면, 이는 NLP 분야로 확장되는 중요한 시도이다. 세 가지 Test-time scaling 방법을 실험했으며, 결과적으로 Model Averaging과 Majority Voting 방법이 LLM의 성능을 향상시키는 반면, Best-of-N 방법은 LeWiDi 과제에는 효과적이지 않다는 것을 발견하였다.

- **Technical Details**: LeWiDi-2025 과제에서는 두 가지 태스크가 있다: (1) Perspecivist task는 각 주석자의 레이블을 예측하고, (2) Soft-label task는 문제 인스턴스에 대한 인간 주석의 분포를 예측하는 것이다. 모델 성능 분석에 대한 새로운 지표로 'prediction diversity'를 도입했으며, 이는 문제의 난이도를 추적하는 데 사용된다. Test-time scaling 방법으로는 BoN 샘플링과 같은 기존의 기법을 수정하여 LeWiDi 작업에 적용하였고, 최종 예측을 위해 우수한 점수를 가진 샘플을 선택하는 방식이다.

- **Performance Highlights**: 모든 LeWiDi 데이터셋에서 Model Averaging과 Majority Voting 방식은 LLM 성능을 지속적으로 향상시켰지만, Best-of-N 방식은 현재 LeWiDi 작업으로 이전되지 않음을 확인했다. 이는 LLM이 주석자의 해석적 변동성과 불일치를 효과적으로 처리하지 못하는 가능성을 시사한다. 전반적으로, 이번 연구는 LLM의 추론(foreground reasoning) 능력과 주석 불일치 문제에 대한 새로운 접근법을 제공하고, 향후 연구 방향에 많은 기여를 할 것으로 예상된다.



### The Robustness of Differentiable Causal Discovery in Misspecified Scenarios (https://arxiv.org/abs/2510.12503)
Comments:
          accepted to ICLR 2025

- **What's New**: 논문은 다양한 주류 인과 발견 (causal discovery) 알고리즘의 실질적 성능을 광범위하게 벤치마킹하며, 이들 알고리즘이 가정하는 i.i.d. 데이터에 대한 8가지 모델 가정 위반을 분석합니다. 이 연구의 주요한 점은 인과 추론 알고리즘이 실제 데이터에서의 모델 가정 위반에 어떻게 대처하는지를 다루며, 특히 미분 가능한 기법이 다양한 시나리오에서 강건성을 보인다는 것입니다.

- **Technical Details**: 인과 발견의 기초적인 방법론으로는 제약 기반 (constraint-based), 점수 기반 (score-based), 기능적 인과 모델 기반 (functional causal model-driven), 그리고 미분 기반 (gradient-based) 접근법이 있습니다. 최근에 등장한 미분 기반 기법인 NOTEARS와 같은 방법들은 조합적 비순환 제약을 부드러운 평등 제약으로 변환하고, 최적화를 위한 증강 라그랑지안 방법 (augmented Lagrangian method)을 통해 문제를 해결합니다.

- **Performance Highlights**: 이 연구는 12개의 저명한 인과 발견 알고리즘의 성능을 8개의 모델 가정 위반 시나리오에서 대규모 실험을 통해 평가하였습니다. 실험 결과는 미분 가능한 방법들이 특정 어려운 상황에서도 강건성을 보인다는 점을 심층적으로 입증하였으며, 이는 인과 발견 연구의 잠재력을 더욱 발전시킬 수 있는 방향성을 제시합니다.



### PubSub-VFL: Towards Efficient Two-Party Split Learning in Heterogeneous Environments via Publisher/Subscriber Architectur (https://arxiv.org/abs/2510.12494)
Comments:
          Accepted at NeurIPS 2025

- **What's New**: 이번 논문에서는 PubSub-VFL이라는 새롭고 효율적인 Vertical Federated Learning (VFL) 프레임워크를 제안합니다. 이 아키텍처는 두 당사자 간의 협업 학습을 위해 Publisher/Subscriber 구조를 활용하여 학습 효과성을 높입니다. 현재의 VFL 구조가 겪고 있는 저조한 자원 활용도와 훈련 효율성을 해결하고자 하며, 이를 통해 훈련 대기 시간을 줄이고 시스템의 효율성을 개선하고자 합니다.

- **Technical Details**: PubSub-VFL은 기존의 VFL에서의 동기식 의존성을 탈피하기 위해 계층적 비동기 메커니즘을 설계하였습니다. 이를 통해 각 당사자가 서로의 데이터에 접근하지 않고도 중간 결과를 안전하게 교환할 수 있도록 하였습니다. 또한, Pub/Sub 아키텍처와 파라미터 서버 구조의 데이터 병렬성을 결합하여 더 나은 컴퓨팅 자원 활용과 훈련 효율성을 달성합니다.

- **Performance Highlights**: 사례 연구를 통해 PubSub-VFL의 효율성이 입증되었습니다. 이 구조는 최신 기준선 모델과 비교하여 훈련 속도를 2배에서 7배까지 향상시키며, 정확도를 유지하면서도 최대 91.07%의 자원 활용도를 달성합니다. 이러한 성과는 보안 프로토콜인 차별적 프라이버시와 호환되며, 안정적인 수렴성을 보임을 이론적으로 분석하였습니다.



### A Text-Image Fusion Method with Data Augmentation Capabilities for Referring Medical Image Segmentation (https://arxiv.org/abs/2510.12482)
- **What's New**: 이번 연구는 데이터 증강(data augmentation)의 중요성을 강조하면서, 텍스트와 이미지 정보를 통합 하는 혁신적인 초기 융합(early fusion) 방법을 제안합니다. 일반적인 데이터 증강 기법이 측정된 텍스트와 이미지 간의 공간적 정렬(spatial alignment)을 방해하는 문제를 해결하기 위해, 텍스트와 시각적 특성을 증강 전에 결합하여 정렬을 유지하는 접근 방식을 사용합니다. 이를 통해 세 가지 의료 이미지 작업 및 네 가지 분할(segmentation) 프레임워크에서 최첨단 성능을 달성했습니다.

- **Technical Details**: 제안된 방법은 텍스트 인코더(text encoder), 분할 프레임워크(segmentation framework) 및 pseudo 이미지를 생성하는 경량 생성기(lightweight generator)로 구성됩니다. 기존의 데이터 증강 기술들과는 달리, 텍스트 정보를 이미지와 통합하기 전에 프로젝션하여 오류를 줄이는 방안을 채택하였습니다. 이 과정에서 ROI 기반 학습을 통해 텍스트 기능을 추출하고, 최종적으로 생성된 pseudo 이미지는 원본 이미지와 결합됩니다.

- **Performance Highlights**: 모든 데이터셋에서 제안하는 방법은 Dice score와 mIoU를 크게 증가시켜 기존 기준선(baseline)을 초월하는 성능을 보여줍니다. 모델은 UNet, UNet++, TransUNet 및 MISSFormer와 같은 네 가지 프레임워크에서 검토되었고, 제안된 접근 방식이 데이터 증강 없이도 경쟁력을 유지하며, 데이터 증강 적용 시 더욱 우수한 성능을 보임을 입증했습니다. 향후 작업으로는 이 방법을 다양한 의료 분야에 적용할 수 있는 가능성을 검토하고 있습니다.



### When Personalization Tricks Detectors: The Feature-Inversion Trap in Machine-Generated Text Detection (https://arxiv.org/abs/2510.12476)
- **What's New**: 이번 연구에서는 개인화된 기계 생성 텍스트(MGT) 탐지를 위한 첫 번째 벤치마크인 StyloBench를 소개합니다. 이 벤치마크는 문학 작품과 블로그 포스트를 LLM(대규모 언어 모델)이 생성한 모방 텍스트와 쌍으로 구성되어 있습니다. 실험 결과, 개인화 환경에서 탐지기들 간 큰 성능 차이가 드러났으며, 이는 탐지기들이 개인화된 텍스트에서 효과적으로 작동하지 못함을 시사합니다.

- **Technical Details**: 연구진은 'feature-inversion trap'이라는 가설을 제시하며, 일반 도메인에서 효율적인 특징들이 개인화된 도메인에서는 오히려 반전되어 탐지기들을 혼란스럽게 한다고 설명합니다. 이를 해결하기 위한 접근법인 StyloCheck를 통해 탐지기의 성능 변화를 예측하는 방법을 제시했으며, 이 방법은 탐지기가 의존하는 특징을 파악하는 데에도 사용할 수 있습니다.

- **Performance Highlights**: StyloCheck는 100개의 프로브 데이터셋에서 실험하여 탐지기 성능의 방향성과 크기를 85% 이상의 상관관계로 예측했습니다. 이는 기존 탐지기들이 개인화된 텍스트에서 성능 강하를 겪는 이유를 이해하는 데 도움을 줄 수 있는 연구 방향을 제시합니다. 이 작업은 개인화된 텍스트 탐지에 관한 추가 연구를 독려할 것으로 기대됩니다.



### A Function Centric Perspective On Flat and Sharp Minima (https://arxiv.org/abs/2510.12451)
Comments:
          26 pages, 26 tables, 63 figures, pre-print

- **What's New**: 이 논문은 머신러닝에서 일반화 성능을 향상시킬 수 있는 방식으로 평균적인 미니마의 역할을 재고합니다. 연구진은 샤프니스(sharpness)가 단순히 일반화 실패의 지표가 아니라 함수에 따른 성질로 이해되어야 한다고 주장합니다. 정규화 기술을 사용하여 훈련한 모델들이 예기치 않게 더 날카로운 미니마로 수렴할 수 있으며, 이들이 일반화 및 안전성 측면에서 더 나은 성능을 보일 수 있다는 것이 핵심 내용입니다.

- **Technical Details**: 연구진은 다양한 데이터셋(CIFAR, TinyImageNet)과 모델(ResNet, VGG, ViT)을 사용하여 매개변수 불변 샤프니스 측정법을 적용해 연구를 수행했습니다. 설정된 기본 모델과 정규화 기법(SAM, weight decay, data augmentation)을 적용한 모델을 비교하여 성능을 평가했습니다. 이 결과, 정규화를 통해 훈련된 모델들이 대개 더 날카로운 미니마로 수렴함에도 불구하고, 일반화 및 안전 관련 지표에서 더 나은 성능을 나타낸다는 것을 발견했습니다.

- **Performance Highlights**: 정규화된 모델들이 날카로운(minima) 미니마에서 성능을 발휘하는 것이 확인되었습니다. 이는 이전의 일반화 기준인 평탄함(flatness)과는 상충되는 결과입니다. 각종 안전 및 일반화 지표에서 더 나은 성능을 보였으며, 이는 샤프니스가 단순히 성능의 지표가 아니라 훨씬 더 복잡한 함수의 복잡성과 추론적 편향(inductive bias)을 반영하는 것으로 해석될 수 있음을 보여줍니다.



### Low-Field Magnetic Resonance Image Quality Enhancement using a Conditional Flow Matching Mod (https://arxiv.org/abs/2510.12408)
- **What's New**: 이 논문은 conditional flow matching (CFM)에 기반한 새로운 이미지 품질 전송 프레임워크를 소개합니다. 기존의 생성 모델과 달리 CFM은 최적의 속도 필드를 직접 회귀하여 노이즈 분포와 목표 데이터 분포 간의 연속 흐름을 학습합니다. 이 방법은 저비용의 저자기능식 MRI(low-field MRI)에서 고해상도 이미지를 재구성하여 품질 격차를 해소하는 데 중점을 두고 있습니다.

- **Technical Details**: CFM은 시간 간격 t∈[0,1]에서 소스 분포 p0∼xnoise에서 타겟 분포 p1∼xhigh로의 연속 변환을 가정합니다. 이를 위해 오르디너리 미분 방정식을 사용하며, 경량화된 U-Net 아키텍처를 기반으로 여러 구성 요소가 통합되어 있습니다. 여기에는 다중 스케일 입력 레이어, Squeeze-and-Excitation 모듈이 통합된 잔여 블록, 그리고 해상도를 조정하기 위한 픽셀 언셰플 및 픽셀 셔플 기법이 포함됩니다.

- **Performance Highlights**: 실험 결과 CFM은 최신 기술 대비 뛰어난 성능을 보여주었으며, 구성 요소 수는 현저히 적었습니다. 인-디스트리뷰션(InD) 및 아웃-오브-디스트리뷰션(OOD) 데이터에 대해 강인하게 일반화하는 능력을 입증했습니다. 이 결과는 CFM이 특히 리소스가 제한된 임상 환경에서 MRI 재구성을 위한 강력하고 확장 가능한 도구로서의 가능성을 강조합니다.



### Tokenization Disparities as Infrastructure Bias: How Subword Systems Create Inequities in LLM Access and Efficiency (https://arxiv.org/abs/2510.12389)
Comments:
          6 pages 4 figures

- **What's New**: 본 연구는 200개 이상의 언어에서의 tokenization 효율성을 대규모로 평가하여 인공지능 접근의 형평성을 저해하는 tokenization의 불균형을 정량적으로 분석합니다. 특히, 라틴 문자 언어는 높은 tokenization 효율성을 보이는 반면, 비라틴 및 형태적으로 복잡한 언어는 상대적으로 높은 token inflation을 경험하는 경향이 있음을 밝혔습니다. 이 정보는 인공지능 시스템의 균형 잡힌 발전을 위한 기초 자료로 활용될 수 있습니다.

- **Technical Details**: 연구는 FLORES-200 데이터셋을 활용하여 200개 언어의 tokenization 효율성을 비교합니다. 데이터 전처리를 위해 OpenAI의 tiktoken 라이브러리에서 제공하는 Byte Pair Encoding (BPE) 기반의 tokenizer를 사용하였습니다. TPS(단어당 토큰 수)와 RTC(상대 tokenization 비용) 등의 기준을 통해 각 언어의 tokenization 특성을 포착하여 분석하였습니다.

- **Performance Highlights**: 분석 결과, 라틴 문자 언어는 비라틴 문자 언어보다 평균적으로 3-5배 높은 토큰화 비용을 나타내어 형평성 문제를 나타냈습니다. 이는 저자원 언어 사용자가 인공지능에 접근하는데 있어 불균형적인 비용 효율성을 경험하게 한다는 것을 의미합니다. 결국, 이 연구는 다양한 언어적 맥락에서 더 공정하고 포용적인 tokenization 전략 개발의 필요성을 강조합니다.



### Phenome-Wide Multi-Omics Integration Uncovers Distinct Archetypes of Human Aging (https://arxiv.org/abs/2510.12384)
- **What's New**: 이번 연구는 기존의 단일 오믹스 데이터 기반의 노화 시계와는 달리, 인간의 복잡한 노화 과정을 포괄적으로 모델링하기 위해 다중 오믹스 데이터를 통합한 새로운 방법론을 제안합니다. 12,000명의 참가자를 포함하는 대규모 연구를 통해, 임상 및 행동 데이터와 함께 응집한 다중 오믹스 데이터셋을 사용하여 노화 시계를 개발했습니다. 이 연구는 다중 오믹스 통합이 노화와 관련된 질병 발생 위험을 예측하는 데 있어 얼마나 강력한 도구가 될 수 있는지를 보여줍니다.

- **Technical Details**: 연구팀은 Human Phenotype Project(HPP)에서 수집된 12,000명의 참가자의 고차원 데이터를 활용하여 다중 오믹스 노화 시계를 개발했습니다. 이 시계는 전사체(transcriptome), 지질체(lipidome), 대사체(metabolome), 미생물체(microbiome) 등 다양한 오믹스 데이터의 통합을 포함하며, 이를 통해 생물학적 나이를 정밀하게 산출할 수 있습니다. 또한 비지도 클러스터링(unsupervised clustering)을 통해 노화의 분자적 아형을 식별하여 노화 경로의 이질성을 드러냈습니다.

- **Performance Highlights**: 다중 오믹스 노화 시계는 전통적인 연대기적 나이와 비교하여 예측 정확도가 뛰어난 것으로 평가되었습니다. 머신러닝 기법인 LASSO, Elastic Net 및 LightGBM을 비교하여 최상의 성능을 보이는 모델을 선택한 결과, 예측 나이와 실제 연대기적 나이 간의 높은 상관관계를 발견했습니다. 이 연구는 노화에 있어 다각적 접근 방식의 중요성을 강조하며, 개인화된 건강 모니터링 및 나이에 따른 질병 예방 전략의 기초를 제공합니다.



### LiteVPNet: A Lightweight Network for Video Encoding Control in Quality-Critical Applications (https://arxiv.org/abs/2510.12379)
Comments:
          Accepted PCS 2025 Camera-Ready Version, 5 Pages

- **What's New**: 이 논문에서는 비디오 스트리밍 기술의 새로운 사용 사례에 맞춰 NVENC AV1 인코더의 Quantisation Parameters를 정확히 예측하는 경량 네트워크(LiteVPNet)를 제안합니다. 기존 Transcoding 방식이 품질 관리나 계산 비용에서 한계를 보이는 상황에서, LiteVPNet은 낮은 복잡도의 특징으로 VMAF 점수를 달성하는 데 기여하고 있습니다. 실험 결과, LiteVPNet은 다양한 품질 목표에 대해 평균 VMAF 오차가 1.2 포인트 이하로 유지되었습니다.

- **Technical Details**: LiteVPNet은 비디오 특성을 기반으로 한 낮은 복잡도의 특징 세트를 활용하여 NVENC 인코더를 위한 Quantisation Parameters를 예측합니다. 주요 특징은 비트스트림 메타데이터 통계, 비디오 복잡도, بلос 클립 기반의 의미적 특징 등으로 구성됩니다. 이 모델은 테스트 데이터셋에서 ΔVMAF가 1 이하로 나타나며, 87% 이상의 영상에서 ΔVMAF가 2 이하로 정확하게 예측되었습니다.

- **Performance Highlights**: LiteVPNet은 최신 방법들보다 높은 성능을 보이며, 87%의 테스트 영상에서 VMAF 오차가 2 포인트 이내로 유지되었습니다. 이는 기존 방법의 약 61%에 비해 개선된 수치로, 에너지 효율적이고 품질이 중요한 미디어 경험 개선에 매우 적합합니다. 이러한 성능은 영화 제작과 같은 품질 중심의 워크플로우에 특히 유용합니다.



### Deep Attention-guided Adaptive Subsampling (https://arxiv.org/abs/2510.12376)
- **What's New**: 본 논문에서는 Deep Attention-guided Subsampling (DAS)이라는 새로운 프레임워크를 제안하여, 입력에 적응하는 동적 샘플링을 통해 3D 의료 영상 및 비디오 분류 작업에서 성능을 개선합니다. 기존 방법들은 비정상적인 샘플링 패턴 또는 고정된 접근 방식을 사용했으나, DAS는 attention 메커니즘을 활용하여 개별 입력에 따라 샘플링 방식을 조정합니다. 이로 인해 성능 향상과 계산 복잡성을 줄이는 동시에, 효율적인 추론이 가능해집니다.

- **Technical Details**: 제안된 DAS는 가벼운 기능 추출 모듈, attention 레이어, Gumbel-Softmax 샘플링 메커니즘으로 구성됩니다. 다양한 특징을 파악하기 위해 다중 경로를 사용하여 입력 시퀀스의 리치(representative) 표현을 생성합니다. 이후 multi-head attention 레이어를 통해 최종 샘플링 로짓을 생성하며, 각 헤드는 전반적인 attention 분포를 스케일링하여 최적의 샘플링 결정에 기여합니다.

- **Performance Highlights**: DAS는 MedMNIST3D 데이터셋 및 실제 임상 환경에서 수집된 데이터셋을 포함한 여러 의료 영상 데이터셋에서 효과를 입증했습니다. 제안된 방법은 샘플링 프로세스의 해석을 용이하게 하고 리소스가 제약된 환경에서도 효율적인 추론을 가능하게 하여, 기존 방법들에 비해 훨씬 더 나은 성능을 보여주었습니다.



### LLM-REVal: Can We Trust LLM Reviewers Yet? (https://arxiv.org/abs/2510.12367)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)의 연구 및 동료 검토 프로세스에 대한 깊은 통합이 학술적 공정성에 미치는 영향을 다루고 있습니다. LLM을 동료 검토자로 사용하는 잠재적 위험들을 시뮬레이션을 통해 분석하였으며, LLM 기반 리뷰와 인간 판단 사이의 뚜렷한 불일치를 확인했습니다. 특히, LLM 리뷰어는 LLM이 작성한 논문에 대해 시스템적으로 높은 점수를 부여하는 반면, 인간이 작성한 논문에 대해서는 낮은 점수를 주는 경향이 있음을 발견했습니다.

- **Technical Details**: 연구에서는 LLM-REVal(LLM REViewer Re-EValuation)이라는 시뮬레이션을 제안하여 연구 및 리뷰 프로세스를 다중 라운드로 분석하였습니다. 연구 에이전트(Research Agent)는 문헌 검색, 아이디어 생성, 실험 설계, 결과 분석 및 원고 편집을 포함한 연구 워크플로우를 자율적으로 수행하며, 리뷰 에이전트(Review Agent)는 초기 평가, 반박, 재평가 및 최종 결정을 포함한 완전한 동료 검토 파이프라인을 모방합니다. 이러한 시뮬레이션은 LLM 논문과 인간 논문의 비교를 통해 LLM 리뷰어의 행동 패턴을 정량적으로 분석합니다.

- **Performance Highlights**: 연구 결과, LLM 리뷰어는 LLM 작성 논문에 대해 실질적으로 더 높은 점수를 부여하는 경향이 있으며, 반면 인간 논문에 대해서는 낮은 점수를 지속적으로 할당합니다. 이러한 경향은 LLM 리뷰어들의 언어적 특징 편향과 비판적인 진술들에 대한 회피에서 기인한다고 분석되었습니다. 이러한 발견은 LLM이 동료 검토 과정에 신중하게 통합되지 않을 경우 인간 저자와 학술 연구에 미치는 위험과 공정성 우려를 강조하는 결과로 이어집니다.



### (R)evolution of Programming: Vibe Coding as a Post-Coding Paradigm (https://arxiv.org/abs/2510.12364)
Comments:
          Workshop Submission at the sixth decennial Aarhus conference in Workshop "The End of Programming (as we know it) - Envisioning Radical Re-Conceptualizations of Co-Coding with AI"

- **What's New**: 본 논문에서는 Vibe Coding (VC) 패러다임에 대해 조사합니다. VC는 개발자와 AI 시스템 간의 직관적이고 감정 기반의 즉흥 상호작용을 강조하며, 최근 Generative Artificial Intelligence (GenAI) 기술의 발전을 바탕으로 합니다. 이 연구는 End-User Development (EUD)의 논의를 확장하며 VC가 전통적인 프로그래밍 접근법에서 어떻게 벗어나는지를 탐구합니다.

- **Technical Details**: 논문에서는 10명의 경험이 풍부한 소프트웨어 전문가들과의 5회의 반구조적 인터뷰를 통해 다섯 가지 주제적 차원(창의성, 지속 가능성, 프로그래밍의 미래, 협업, 비판)을 식별했습니다. VC는 'co-drifting'의 은유를 통해 개념화되며, AI 보조 개발의 현행 'co-piloting' 관점과 대조됩니다. VC는 전문가와 비전문가 사이의 경계를 모호하게 하여 개발자의 역할을 재구성합니다.

- **Performance Highlights**: VC는 새로운 형태의 표현을 가능하게 하고 빠른 프로토타입을 촉진하지만, 재현 가능성(reproducibility), 확장성(scalability), 포괄성(inclusivity) 문제를 동반합니다. 이러한 고찰을 통해 VC는 프로그래밍 문화의 의미 있는 변화를 나타내며, 인공지능-컴퓨터 상호작용(HCI) 및 소프트웨어 공학 연구에서의 추가 조사가 필요함을 주장합니다.



### Finite-time Convergence Analysis of Actor-Critic with Evolving Reward (https://arxiv.org/abs/2510.12334)
- **What's New**: 본 논문은 evolving reward function의 존재하에 single-timescale actor-critic 알고리즘의 유한 시간 내 수렴(non-asymptotic convergence) 분석을 최초로 제공합니다. 기존의 RL 알고리즘이 정적인 보상 함수를 기반으로 하는 반면, 본 논문은 지속적으로 변화하는 보상 매개변수의 영향력을 분석합니다. 이로써 RL 알고리즘의 수렴성을 보장하기 위한 이론적 기초를 마련합니다.

- **Technical Details**: 논문에서는 Markovian sampling 환경에서 선형 함수 근사를 사용하는 single-sample actor-critic 알고리즘을 대상으로 합니다. 보상 매개변수가 매 시간 단계마다 변화하며, 이러한 비정상성(non-stationarity)이 정책의 최적화 및 가치 추정에 미치는 영향을 분석합니다. 연구 결과는 보상 매개변수가 충분히 천천히 변화할 경우, O(1/√T) 수렴 속도가 달성될 수 있음을 보여줍니다.

- **Performance Highlights**: 논문에서 제시하는 방법론은 기존의 정적 보상 상황에서의 수렴 속도를 log²T 배만큼 개선하며, 궁극적으로 폭넓은 RL 기술에 대한 이론적 보장을 제공합니다. 연구는 curiosity-driven reward shaping, random network distillation, 자동 엔트로피 조정이 포함된 soft actor-critic과 같은 여러 인기 있는 RL 기술들에 적용될 수 있습니다. 이와 함께, evolving reward를 처리하기 위해 Lipschitz continuity의 개념을 적용하여 분석한 것을 강조합니다.



### Simple Projection Variants Improve ColBERT Performanc (https://arxiv.org/abs/2510.12327)
- **What's New**: 이번 연구에서는 ColBERT 모델의 단일 레이어 선형 프로젝션을 대체하기 위해 더 잘 연구된 다양한 피드포워드 선형 네트워크(FFN)의 대안을 탐색합니다. MaxSim 연산자가 다중 벡터 모델 훈련의 그레디언트 흐름에 미치는 영향을 분석하고, 선형 프로젝션의 고유한 한계를 보여줍니다. 다양한 프로젝션 블록을 체계적으로 평가하여 더 나은 설계가 ColBERT 모델의 하류 성능에 긍정적인 영향을 미친다는 것을 증명하였습니다.

- **Technical Details**: 현재 모든 다중 벡터 모델은 원래 ColBERT 아키텍처의 변형을 따르며, 단일 레이어 선형 프로젝션을 통해 최종 출력 표현을 얻는 방식으로 작동합니다. 이 논문에서는 ColBERT 모델의 최종 피드포워드 블록에 대한 일련의 수정을 제안하고 이들의 특성이 검색 성능 향상에 어떻게 기여할 수 있는지 보여줍니다. 또한 더 많은 비선형 FFN 블록과 GLU 블록, 스킵 연결을 도입하여 다중 벡터 모델의 한계를 완화할 수 있다는 점을 강조합니다.

- **Performance Highlights**: 연구 결과, 제안된 프로젝션 변형들이 기존의 단일 레이어 프로젝션보다 우수한 성능을 보였으며, 특히 최고의 변형은 여러 검색 기준에서 평균적으로 2 NDCG@10 포인트 이상 성능을 향상시켰습니다. 다양한 벤치마크에서 다수의 비최적 프로젝션 변형 또한 전통적인 단일 레이어 프로젝션을 초과하는 성능을 보여주었으며, 이는 우리의 가설을 강하게 지지합니다. 이러한 성능 향상 효과는 랜덤 시드를 통해 일관되게 관찰되었으며, 단일 레이어의 교체가 ColBERT 모델의 강력한 업그레이드 가능성을 확인시켜줍니다.



### Causal Inspired Multi Modal Recommendation (https://arxiv.org/abs/2510.12325)
- **What's New**: 이 연구에서는 기존 멀티모달 추천 시스템에서 중요한 두 가지 편향, 즉 모달 혼란(modal confounding)과 상호작용 편향(interaction bias)에 대해 다루고 있습니다. 저자들은 Causal-inspired 멀티모달 추천 프레임워크를 제안하며, 이를 통해 숨겨진 모달 혼란 요소를 식별하고 상호작용의 노이즈 문제를 해결하기 위한 접근 방식을 제공합니다. 또한, 제안된 방법은 매우 높은 해석 가능성을 제공하며 실제 전자상거래 데이터 세트에서 성능을 입증하였습니다.

- **Technical Details**: 연구에서는 구조적 인과 모델(Structural Causal Model, SCM)을 기반으로 사용자 선호 생성 과정을 분석합니다. 다중 경로 수정을 위한 두 가지 방법론으로는 백도어 조정(back-door adjustment)과 프론트 도어 조정(front-door adjustment)을 사용하며, 특히 층화된 매칭을 통해 모달 혼란을 차단하는 전략을 사용합니다. 제안된 모델은 이중 채널 크로스 모달 확산 모듈을 통해 숨겨진 모달 혼란 요소를 효과적으로 식별합니다.

- **Performance Highlights**: 세 가지 실제 전자상거래 데이터 세트에서의 실험 결과, 제안된 방법이 최신 기준선 모델들보다 지속적으로 우수한 성능을 보였습니다. 특히, 추천 정확도가 크게 향상되었으며, 사용자 선호의 해석 가능성 또한 확보되었습니다. 이를 통해 멀티모달 추천 시스템의 발전에 다양한 기여를 할 것으로 기대됩니다.



### Deep SPI: Safe Policy Improvement via World Models (https://arxiv.org/abs/2510.12312)
Comments:
          10 pages main text, 17 pages appendix (excluding references)

- **What's New**: 안전한 정책 개선(Safe Policy Improvement, SPI)은 정책 업데이트에 대한 이론적 제어를 제공하지만, 기존의 보장은 주로 오프라인, 테이블형 강화 학습에 국한되어 있습니다. 이 연구에서는 일반적인 온라인 환경에서 SPI를 탐구하고, 세계 모델(world model) 및 표현 학습(representation learning)과 결합하여 이론적 프레임워크를 개발합니다. 이를 통해 정책 업데이트를 현재 정책의 정의된 이웃으로 제한하면 단조로운 개선과 수렴을 보장할 수 있음을 보여주었습니다.

- **Technical Details**: 이론적 분석은 전이(transition) 및 보상(reward) 예측 손실(prediction losses)을 표현의 품질과 연결합니다. 이를 통해 전통적인 오프라인 SPI 정리의 온라인, '딥' 아날로그를 도출할 수 있었습니다. 연구자들은 새로운 정책이 이전 정책보다 현저히 나쁘지 않도록 보장하여 고차원, 복잡한 환경에서 안전한 정책 개선을 가능하게 합니다.

- **Performance Highlights**: 새롭게 도입된 DeepSPI 알고리즘은 지역적인 전이 및 보상 손실을 정규화된 정책 업데이트와 결합하여 우수한 성능을 보입니다. Arcade Learning Environment(ALE-57) 벤치마크에서 DeepSPI는 PPO 및 DeepMDPs와 같은 강력한 기준선을 초과하거나 동등한 성능을 발휘하면서 이론적 보장도 유지합니다. 이러한 결과는 고차원 입력을 가진 온라인 RL 환경에서 안전한 개선을 실질적으로 구현할 수 있음을 보여줍니다.



### Chinese ModernBERT with Whole-Word Masking (https://arxiv.org/abs/2510.12285)
- **What's New**: 이번 연구는 중국어에 최적화된 새로운 인코더 모델인 Chinese ModernBERT를 소개합니다. 이 모델은 기존의 인코더 전용 모델들이 영어에서 이룬 성과를 활용하면서, 중국어의 토크나이제이션 및 형태학적인 측면을 적극 반영하고 있습니다. 특히 32k BPE 어휘 집합, 전면 단어 마스킹, 두 단계의 사전 학습 파이프라인 등 혁신적인 기술 요소들을 결합하여 더 나은 정확도와 효율성을 달성하고자 합니다.

- **Technical Details**: Chinese ModernBERT는 하드웨어 인식 디자인과 32k BPE 어휘를 포함하여 자주 사용되는 한국어 접사 및 합성어에 최적화되어 있습니다. 동적인 마스킹 커리큘럼을 통해 훈련 중 모델의 성숙도에 따라 과제가 조정되며, RoPE와 대칭적 지역/전역 주의(attention)를 쌍으로 사용하여 긴 문맥을 효율적으로 다룰 수 있도록 설계되었습니다. 더욱이 면밀한 긴 컨텍스트 안정성과 효율성을 도모하기 위해 두 단계의 사전 훈련이 적용됩니다.

- **Performance Highlights**: 중국어 ModernBERT는 CLUE 벤치마크에서 기존의 강력한 중국어 인코더와 경쟁력을 가지며, bf16 환경에서 높은 긴 시퀀스 처리량과 짧은 시퀀스 속도를 동시에 유지하는 성능을 자랑합니다. 추가적으로, SimCLUE에서 무작위 비교 데이터를 활용하여 fine-tuning을 수행한 결과, 성능이 향상되어 유사성 작업에서 Qwen-0.6B-임베딩을 초과하는 성과를 보여 주었습니다. 이 연구는 재현 가능한 연구를 위한 토크나이저와 모델 가중치를 공개할 예정입니다.



### Quantum Annealing for Staff Scheduling in Educational Environments (https://arxiv.org/abs/2510.12278)
Comments:
          8 pages, 3 tables, and 1 figure. Paper submitted to the International Conference on Quantum Communications, Networking, and Computing (QCNC 2026)

- **What's New**: 이번 논문은 이탈리아 칼라브리아의 공립학교에서 발생한 새로운 직원 배치 문제를 다룹니다. 이 문제는 유치원, 초등학교 및 중학교에 걸쳐 직원들을 배분하는 것으로, 가용성, 역량 및 공정성을 고려해야 합니다. 저자들은 이를 해결하기 위해 최적화 모델을 개발하고 양자 어닐링(quantum annealing)을 기반으로 한 접근 방식을 탐구했습니다.

- **Technical Details**: 문서에서 다루는 문제는 주간 일정에 걸쳐 교육 사이트에 직원을 배치하는 문제입니다. 이 모델은 다양한 시간과 공간 제약을 고려하며, 전체 직원의 가용성과 직무에 따라 다양성을 가진 협력자들로 구성되어 있습니다. 현장 환경에서는 인원배치, 근무시간, 성비와 같은 특정 제약과 함께 정교한 일정 모델이 필요합니다.

- **Performance Highlights**: 양자 어닐링 방법이 실제 데이터에서 균형 잡힌 배정을 짧은 시간 내에 생성 할 수 있음을 보여줍니다. 이 연구는 교육 기획 및 복잡한 자원 할당 과제에서 양자 최적화 방법의 실제 적용 가능성을 입증하였습니다. 또한, 제안된 스케줄링 모델을 평가하기 위한 더 큰 사례를 정의하고 솔루션 품질, 확장성 및 강건성을 평가했습니다.



### TFGA-Net: Temporal-Frequency Graph Attention Network for Brain-Controlled Speaker Extraction (https://arxiv.org/abs/2510.12275)
Comments:
          5 pages, 3 figures

- **What's New**: 본 논문에서는 뇌파(EEG) 신호를 기반으로 하는 타겟 스피커 추출을 위한 새로운 모델인 TFGA-Net을 제안합니다. 이 모델은 리스너의 EEG 신호를 사용해 목표 스피치를 직접 추출하는 방식을 채택하여, 기존의 청각 주의 디코딩(Auditory Attention Decoding)에 기초한 방법의 한계를 극복하고자 합니다.

- **Technical Details**: TFGA-Net은 Speech Encoder, EEG Encoder, Speaker Extraction 모듈, Speech Decoder로 구성됩니다. EEG Encoder는 다중 시간-주파수 특성을 캡처하고, 작업 선택적 피질(topological structure) 구조를 통합하여, 실제 스피치와 관련 있는 뇌 활동의 다중 스케일 정보를 추출합니다.

- **Performance Highlights**: 실험 결과 Cocktail Party 및 KUL 데이터셋에서 TFGA-Net 모델이 기존의 최첨단 모델보다 14.1% 및 15.8% 성능 향상을 보이며, 이를 SI-SDR(Scale-Invariant Signal-to-Distortion Ratio) 같은 객관적 평가 메트릭으로 입증하였습니다. 이러한 성과는 전반적으로 뇌파 신호와 스피치 간의 관계를 효율적으로 활용한 결과로 볼 수 있습니다.



### HiLoRA: Adaptive Hierarchical LoRA Routing for Training-Free Domain Generalization (https://arxiv.org/abs/2510.12266)
- **What's New**: 본 논문에서는 Low-Rank Adaptation (LoRA)의 효율성을 기반으로 하는 새로운 프레임워크인 HiLoRA를 제안합니다. HiLoRA는 훈련 없이 LoRA 풀에서의 적응형 계층적 라우팅을 통해 도메인 일반화(domain generalization)에 실질적인 개선을 가져옵니다. Gaussian likelihoods을 사용하여 입력 시퀀스에 맞는 LoRA의 하위 집합을 선택하고, 가장 정보가 풍부한 rank-one 구성 요소(ROC)만을 활성화합니다.

- **Technical Details**: 기존의 LoRA 모듈을 기반으로 링계층적 로팅을 사용할 수 있도록 HiLoRA는 rank-one 구성 요소를 통해 LoRA를 세분화하여 처리합니다. 각 LoRA는 샘플링된 임베딩 세트에서 적합된 Gaussian 분포로 표현되어 유사성을 기반으로 입력과 LoRA 간의 비교를 가능하게 합니다. 이로써 HiLoRA는 적절한 LoRA 대수를 선택하고 ROC 할당을 최적화하여 성능을 극대화합니다.

- **Performance Highlights**: HiLoRA는 LLaMA2-7B에서 최대 55%의 정확도 향상을 달성하여 최고 수준의 모델들과 비교하여 우수한 성능을 보여줍니다. 또한 FLAN-T5-large에서는 13%의 상승률을 기록하면서도 실용적인 추론 처리량을 유지합니다. 이러한 성과는 HiLoRA의 효과적인 도메인 적응과 성능 개선을 입증합니다.



### Human-in-the-Loop Bandwidth Estimation for Quality of Experience Optimization in Real-Time Video Communication (https://arxiv.org/abs/2510.12265)
Comments:
          Accepted for publication in the proceedings of the AAAI Conference on Artificial Intelligence 2026 (IAAI Technical Track on Deployed Highly Innovative Applications of AI)

- **What's New**: 이 논문에서는 실시간 통신의 품질(QoE)을 개선하기 위한 대역폭 추정 기술이 발전하고 있음을 강조합니다. 특히, 사용자의 주관적인 평가에서 파생된 QoE 보상 모델을 학습하여 실시간 비디오 회의 시스템의 오디오 및 비디오 품질을 측정하는 데이터 중심의 접근 방식을 제안합니다. 이 방법은 대역폭 추정 훈련 데이터 셋 구축을 위해 약 110만 개의 네트워크 트레이스를 수집하였습니다.

- **Technical Details**: 저자들은 새로운 분포적 오프라인 강화 학습(Distributional Offline Reinforcement Learning, RL) 알고리즘을 도입하여, QoE를 최적화하기 위한 신경망 기반 대역폭 추정기를 학습시켰습니다. 이 알고리즘은 기존의 Implicit Q-learning 알고리즘을 확장하여 강건성을 높이기 위한 비대칭 학습 신호를 활용하였습니다. 결과적으로, A/B 테스트를 통해 제안한 방법이 기본 대역폭 추정기 대비 11.41% 감소한 주관적 저품질 통화 비율을 보여주었습니다.

- **Performance Highlights**: 이 연구는 Microsoft Teams의 실시간 미디어 스택에서 배포되어 매일 수백만 명의 사용자가 다양한 네트워크 조건과 장치 클래스에서 서비스를 이용하고 있습니다. 저자들은 이 방법이 실제 사용 환경에서 효과적으로 작동함을 보여주며, 여러 네트워크 환경에서 자원 할당이 중요한 다른 네트워크 멀티미디어 애플리케이션으로도 일반화될 수 있음을 나타냅니다.



### Shallow Robustness, Deep Vulnerabilities: Multi-Turn Evaluation of Medical LLMs (https://arxiv.org/abs/2510.12255)
Comments:
          Dataset and code: this https URL ; this https URL Accepted as a poster at NeurIPS 2025 Workshop on GenAI for Health: Potential, Trust, and Policy Compliance

- **What's New**: 이번 논문에서는 의료 질문 응답 분야에서의 멀티 턴 강건성을 체계적으로 평가하기 위한 프레임워크인 MedQA-Followup를 소개합니다. 전통적인 면접 방식은 주로 단일 질문 응답 평가에 초점을 맞추고 있으나, 실제 의료 상담에서는 다수의 앵글과 맥락의 복잡한 상호작용이 존재합니다. 연구를 통해 인공지능 모델들이 초기 진단을 받더라도 후속 질문에 의해서 신뢰도를 크게 저하할 수 있음을 발견했습니다.

- **Technical Details**: MedQA-Followup는 표면적인 강건성(shallow robustness)과 심층적인 강건성(deep robustness)을 구별하며, 이를 통해 다양한 간접(intervention)과 직접(intervention) 개입을 평가할 수 있습니다. 본 연구는 의학 Q&A 작업에서의 기존 강건성 연구를 모두 아우르는 세분화된 분류법을 제공합니다. 실험 결과, 연구에 포함된 다섯 가지 최신 LLM들은 얕은 변형에 대해서는 견고하나, 멀티 턴 상황에서는 심각한 취약점을 드러냈습니다.

- **Performance Highlights**: 모델의 정확도는 초기 진단에서 91.2%였으나 혼란스러운 맥락이 추가되자 Claude Sonnet 4의 경우 13.5%로 급격히 떨어졌습니다. 이 연구는 특정 모델들이 다중 개입을 겪을 때 성능 저하가 더욱 극심해지는 반면, 일부 모델은 부분적으로 회복하기도 했다는 점을 강조합니다. 이러한 결과는 의료 LLM의 안전하고 신뢰할 수 있는 배포를 위해 멀티 턴 강건성이 중대한 접근 방식이라는 점을 시사합니다.



### Diffusion Models for Reinforcement Learning: Foundations, Taxonomy, and Developmen (https://arxiv.org/abs/2510.12253)
Comments:
          Under Review

- **What's New**: 이 논문에서는 최근 확산 모델(Diffusion Models, DMs)이 강화 학습(Reinforcement Learning, RL)에서의 활용 방안을 포괄적으로 조사하고 있습니다. DMs는 다중 모드 표현력과 안정적인 학습을 제공하며, RL의 주요 문제를 해결할 수 있는 잠재력을 가지고 있습니다. 이 연구는 DMs가 적용되는 다양한 전략과 응용 분야를 정리하고, 확산 기반의 RL 발전 방향과 향후 연구 방향을 제시합니다.

- **Technical Details**: 연구에서는 강화 학습의 기초 및 DMs의 기본 개념을 소개하며, DMs와 RL 프레임워크의 통합 방식을 분석합니다. 이 논문은 기능 중심의 분류법과 기법 중심의 분류법을 통해 DMs의 역할을 명확히 하고, 온라인 및 오프라인 학습 환경에서의 구현을 정리합니다. 또한, 단일 에이전트에서 다중 에이전트 도메인으로의 발전을 다루며, DMs와 RL의 통합을 위한 다양한 프레임워크를 설명합니다.

- **Performance Highlights**: DMs를 활용한 RL은 여러 응용 분야에서 강력한 성능을 보이며, 탐색, 정책 표현력 및 불확실성 하에서의 계획 문제를 해결하는 데 기여하고 있습니다. 예를 들어, Diffuser라는 모델은 비선형 샘플링을 통해 목표 지향적인 계획을 가능하게 하고 있습니다. DMs는 강화 학습의 여러 한계를 극복하고 있으며, 이러한 접근 방식은 특히 로봇 제어, 자율 주행, 과제 스케줄링 등 다양한 분야에서 실용적인 응용 가능성을 보여줍니다.



### PromptLocate: Localizing Prompt Injection Attacks (https://arxiv.org/abs/2510.12252)
Comments:
          To appear in IEEE Symposium on Security and Privacy, 2026

- **What's New**: 이번 연구에서는 PromptLocate라는 새로운 방법을 제안하여 오염된 데이터 내에서 주입된 프롬프트(prompt)를 지역화(localize)하는 문제를 해결하고자 합니다. 기존의 주입된 프롬프트 지역화에 대한 연구는 거의 없었으며, 이 접근 방식은 후속 공격 포렌식 분석(post-attack forensic analysis) 및 데이터 복구(data recovery) 과정에서 중요한 역할을 합니다.

- **Technical Details**: PromptLocate는 세 가지 단계로 구성됩니다: (1) 오염된 데이터를 의미상 일관된 세그먼트로 분할하고, (2) 주입된 명령어를 포함하는 세그먼트를 식별하며, (3) 주입된 데이터를 포함하는 세그먼트를 구체적으로 지적하는 방식입니다. 이러한 접근 방식은 네이티브 원본(normal source) 데이터를 보존하면서 주입된 프롬프트를 정확하게 찾아내는 데 중점을 둡니다.

- **Performance Highlights**: 실험 결과, PromptLocate는 기존의 여덟 가지 및 새로운 여덟 가지 적응형 공격(adaptive attacks)에 대해 주입된 프롬프트를 정확하게 지역화할 수 있음을 입증하였습니다. 제안된 방법은 기존의 프롬프트 탐지기인 DataSentinel보다 우수한 성능을 보이며, 해석 가능한 기계 학습(attributive methods)에서 사용되는 방법들보다도 뛰어난 정확성을 보여주었습니다.



### MoRA: On-the-fly Molecule-aware Low-Rank Adaptation Framework for LLM-based Multi-Modal Molecular Assistan (https://arxiv.org/abs/2510.12245)
- **What's New**: 이 논문은 분자 그래프 구조와 대형 언어 모델(LLMs)의 효과적인 통합 방법에 대한 도전 과제를 다루고 있습니다. 기존의 멀티모달 정렬 방법은 대개 LLM을 미세 조정하거나 정적 어댑터를 추가하는 방식으로 처리됩니다. 그러나 이 방법들은 모든 분자 입력에 대해 공유된 매개변수 공간을 최적화하여 구조적 특성을 포착하는 데 한계를 가지고 있습니다. 이를 해결하기 위해 저자들은 각 분자에 대해 인스턴스 고유의 매개변수 공간 정렬을 보장하는 방법을 제안합니다.

- **Technical Details**: 저자들은 MoRA(Molecule-aware Low-Rank Adaptation)라는 기법을 도입하여 각 입력 분자 그래프에 대해 고유한 저차원 적응 가중치를 생성합니다. 이러한 가중치는 동적으로 고정된 LLM에 주입되어 각 분자 입력의 구조에 맞춰서 모델이 추론을 조정할 수 있게 하고, LLM의 핵심 지식을 보존할 수 있도록 합니다. 이 과정은 LLM의 핵심 모듈에 직접 주입되어 구조 인식이 가능하게 하며, 특히 재조정되지 않은 LLM의 기능을 유지합니다.

- **Performance Highlights**: MoRA는 화학 반응 예측 및 분자 자막 생성과 같은 주요 분자 작업에서 기존의 정적으로 조정된 기준선보다 더 뛰어난 성능을 보여줍니다. 특히, 반응 예측에서 14.1%의 상대적 향상과 양자 속성 예측에서 오류를 22% 줄이는 성과를 이뤘습니다. 실험 결과들은 MoRA가 다양한 이해 및 생성 작업에서 최첨단 성능을 달성했다는 것을 보여줍니다.



### Analysing Moral Bias in Finetuned LLMs through Mechanistic Interpretability (https://arxiv.org/abs/2510.12229)
Comments:
          Preprint. Under review

- **What's New**: 이번 연구에서는 대규모 언어 모델(LLM)의 편향이 어떻게 나타나는지를 조사하였습니다. Knobe 효과라는 잘 알려진 도덕적 편향이 미세 조정된 LLM에서 나타나는지를 살펴보았고, 특정 레이어로 기인할 수 있는지를 분석하였습니다. Layer-Patching 기법을 통해 미세 조정된 모델에서의 편향을 제거할 수 있다는 놀라운 결과를 도출하였습니다.

- **Technical Details**: 이번 연구에서는 Llama, Mistral, Gemma 등 3개의 오픈 웨이트 LLM을 대상으로 Layer-Patching 분석을 진행하였습니다. 연구 결과, Knobe 효과는 미세 조정 과정에서 학습되며 특정 중후반 레이어에 국한되어 있음을 발견했습니다. 이 연구는 LLM의 구조와 편향의 연관성을 밝혀주며 선행 연구와의 일관성을 가지고 있습니다.

- **Performance Highlights**: Layer-Patching 기법을 사용함으로써 LLM의 편향을 효과적으로 제거할 수 있음을 보여주었습니다. 이를 통해 추가적인 훈련 없이도 편향을 완화할 수 있는 가능성을 제시하였습니다. 이러한 발견은 LLM의 사회적 편향을 간섭을 통해 해결할 수 있다는 새로운 증거를 제공합니다.



### HALF: Harm-Aware LLM Fairness Evaluation Aligned with Deploymen (https://arxiv.org/abs/2510.12217)
- **What's New**: 이번 연구에서는 BIG 모델(대형 언어 모델, LLM)의 공정성 평가를 'HALF(위험 인식 LLM 공정성)'라는 새로운 프레임워크를 기반으로 제시합니다. HALF는 실제 배치에서 모델의 편향을 평가하며, 결과를 위험도에 따라 가중치를 부여하여 분석합니다. 이 프레임워크는 다양한 애플리케이션 도메인을 고려하여, 결과의 중요성을 체계적으로 구분할 수 있는 방법론을 제공합니다.

- **Technical Details**: HALF는 9개의 애플리케이션 도메인을 심각도에 따라 3개의 등급(심각, 중간, 경미)으로 조직합니다. 각 범주는 다섯 단계의 파이프라인을 통해 평가되며, 평가 결과를 0-100 점 스케일로 표시하여 직관적인 비교를 가능하게 합니다. 연구에서는 8개의 다양한 LLM을 사용하여 법률, 의료, 교육 등 다양한 분야에서의 공정성을 분석하고 결과를 도출합니다.

- **Performance Highlights**: 연구 결과, 모델의 공정성은 도메인에 따라 일관성이 없으며, 모델 크기나 성능이 반드시 공정함을 보장하지 않는다는 것을 확인했습니다. 또한 이유 중심의 모델이 의료 결정을 지원하는 데는 더 나은 성과를 보였으나 교육 분야에서는 낮은 성과를 나타내는 경향이 있음을 발견했습니다. 이러한 결과는 과거의 벤치마킹 성공과 실질적 배치의 준비 상태 사이에 존재하는 명확한 간극을 드러냅니다.



### DE3S: Dual-Enhanced Soft-Sparse-Shape Learning for Medical Early Time-Series Classification (https://arxiv.org/abs/2510.12214)
Comments:
          Accepted to IEEE BIBM 2025

- **What's New**: 이 논문은 의료 분야의 조기 시계열 분류(Early Time-Series Classification, ETSC)를 위한 새로운 방법인 DE3S(Dual-Enhanced Soft-Sparse-Shape Learning)를 제안합니다. 이 접근법은 기존 방법들이 정확성과 신속성을 대립적으로 고려해야 했던 문제를 해결하고, 세 가지 혁신적인 요소를 포함하고 있습니다. 이를 통해 сред에서 발생할 수 있는 초기 신호의 약한 문제를 극복하고, 환자별 변동성에 강한 분류기를 구현할 수 있습니다.

- **Technical Details**: DE3S는 세 가지 핵심 메커니즘을 포함하고 있습니다: (1) 전통적인 시계열 증강과 주의 기반의 글로벌 시간 강화 기법이 결합된 포괄적인 듀얼 강화 전략, (2) 정보 점수를 기반으로 한 부드러운 형체(sparse shapelet) 희소화 메커니즘이 가장 차별적인 패턴을 동적으로 보존, (3) Mixture of Experts (MoE)와 다중 스케일 Inception 모듈을 융합한 듀얼 경로 아키텍처입니다. 이를 통해 데이터 불균형 문제를 처리하고, 실험을 통해 탁월한 성능을 입증했습니다.

- **Performance Highlights**: 논문에서는 6개의 실제 의료 데이터셋에 대한 광범위한 실험을 통해 최첨단 성능을 보여줍니다. 특히, 성능 저하 요소에 대한 추가 실험을 통해 구성 요소의 효용성을 검증했습니다. DE3S는 약한 초기 신호와 환자 개별 변동성에 강하며, 의료 시계열 데이터에서 효과적인 패턴 발견을 가능하게 하는 혁신적인 접근을 제시합니다.



### Revisiting Meta-Learning with Noisy Labels: Reweighting Dynamics and Theoretical Guarantees (https://arxiv.org/abs/2510.12209)
- **What's New**: 본 논문은 noisy label(노이즈 레이블) 환경에서 meta-learning(메타 학습) 기반의 샘플 재가중치(meta-reweighting)의 훈련 동역학을 철저히 분석합니다. 일반적으로 샘플 재가중치는 청정 데이터 샘플에 대한 지도 신호를 기반으로 하여, 노이즈가 있는 레이블을 제거하는 기법입니다. 특히 이 연구에서는 노이즈가 있는 레이블을 효과적으로 처리하기 위한 세 가지 훈련 단계(획득, 거르기, 후처리)를 제시합니다.

- **Technical Details**: 저자들은 meta-reweighting의 훈련 동역학을 세 가지 단계로 나누어 설명합니다. 첫 번째 단계인 alignment phase(정렬 단계)에서는 청정 샘플과 일치하는 예제의 가중치를 증폭시키고 상충하는 예제를 억제합니다. 두 번째 단계인 filtering phase(필터링 단계)에서는 노이즈가 있는 예제의 가중치가 0에 가까워지며 청정 subset의 손실이 평평해집니다. 마지막으로 후처리 단계에서는 노이즈 필터링이 외란에 민감해집니다.

- **Performance Highlights**: 실험 결과, 제안한 경량 서브게이트는 기존의 강력한 재가중치/선택 기준선에 비해 지속적으로 더 나은 성능을 보였습니다. 이는 synthetic(합성) 및 real noisy-label(실제 노이즈 레이블) 벤치마크 데이터 세트를 통해 입증되었습니다. 우리가 제안하는 방법은 보다 안정적인 성능을 제공하면서도 비싼 bi-level optimization(이중 수준 최적화)을 피할 수 있습니다.



### CompoDistill: Attention Distillation for Compositional Reasoning in Multimodal LLMs (https://arxiv.org/abs/2510.12184)
Comments:
          Preprint. Under Review

- **What's New**: 최근에 효율적인 Multimodal Large Language Models (MLLMs)이 높은 계산 복잡성을 해결하는 방법으로 주목받고 있습니다. 이러한背景에서, Knowledge Distillation (KD) 접근 방식이 대형 모델(teacher)에서 소형 모델(student)로 시각적 및 언어적 지식을 전이하는 유망한 대안으로 등장했습니다. 기존의 KD 방법은 teacher MLLM의 시각적 인식 능력을 student에게 효과적으로 증류하지 못하는 문제를 발견했습니다.

- **Technical Details**: 체계적인 분석을 통해 student와 teacher 간의 시각적 주의력(alignment) 불일치가 이 문제의 주요 원인으로 확인되었습니다. 이 통찰을 바탕으로 우리는 CompoDistill이라는 새로운 KD 프레임워크를 제안하였으며, 이는 student의 시각적 주의력을 teacher와 명시적으로 정렬하여 student의 시각적 인식 능력을 향상시킵니다. 이 프레임워크는 컴포지셔널 추론(compositional reasoning) 작업에서 학생의 성능을 현저히 향상시키며, 기존 연구에서와 마찬가지로 시각적 질문 응답(visual question answering) 작업에서도 강력한 성능을 유지합니다.

- **Performance Highlights**: CompoDistill을 통해 우리는 시각적 인식 능력이 필요한 작업에서 성능 향상을 실험적으로 입증하였습니다. 게다가, CompoDistill은 보다 진보된 백본(backbone) 모델에서도 효과적임을 보여줘 그 일반화 가능성을 강조하고 있습니다.



### From Knowledge to Treatment: Large Language Model Assisted Biomedical Concept Representation for Drug Repurposing (https://arxiv.org/abs/2510.12181)
Comments:
          16 pages, 4 figures, 13 tables. Accepted by EMNLP 2025 (Findings)

- **What's New**: 이 논문에서는 기존의 약물 재사용 방식을 개선하기 위해 LLaDR(대형 언어 모델 보조 약물 재사용) 프레임워크를 제안합니다. 기존 방법들이 임상 연구에서의 생물 의학 개념 지식을 간과한 문제를 해결하고, 이를 통해 KGs(지식 그래프)의 표현력을 향상시키고자 합니다. LLaDR은 생물 의학 개체에 대한 치료 관련 텍스트 표현을 대형 언어 모델에서 추출하여 이를 지식 그래프 임베딩(KGE) 모델의 파인 튜닝(tuning) 과정에 활용합니다.

- **Technical Details**: LLaDR은 두 가지 주요 단계로 구성된 프레임워크로, 첫 번째 단계에서 GPT-4와 같은 LLM을 활용하여 지식 그래프의 각 개체에 대한 풍부하고 맥락 인식적인 텍스트 설명을 생성합니다. 두 번째 단계는 KGE의 파인 튜닝 과정을 진행하며, 텍스트 임베딩을 가중 평균과 기하학적 정렬, 의미적 일관성 규제를 통해 결합하여 최상의 지식 표현을 생성합니다. 이 과정은 그래프 구조에서의 정확성과 개체의 깊은 의미적 맥락을 동시에 포착하기 위한 적응형 학습률 스케줄링과 대조 손실 함수의 활용을 포함합니다.

- **Performance Highlights**: LLaDR은 다양한 벤치마크에서 최첨단 성능을 기록하며, 알츠하이머 병에 대한 사례 연구를 통해 강력성과 효과성을 추가로 입증하였습니다. 기존 KGE 모델에 비해 의약품 재사용 작업의 예측 정확성과 강건성을 크게 향상시킵니다. 실험 결과는 LLaDR의 신뢰성과 생물 의학 개념 표현의 개선을 보여줍니다.



### Budget-constrained Active Learning to Effectively De-censor Survival Data (https://arxiv.org/abs/2510.12144)
- **What's New**: 이 논문은 예산 제약이 있는 환경 아래에서 생존 데이터(survival data)를 다루는 새로운 예산 학습(budgeted learning) 접근법을 제시합니다. 특히, 이 방법은 데이터 수집 과정에서 라벨이 부여되지 않은 인스턴스를 부분적으로 라벨링할 수 있으며, 이는 실제 데이터 수집 및 분석 조건을 모사합니다. 연구진은 생존 데이터에 적용할 수 있는 최첨단 예산 학습 알고리즘의 이론적 및 실험적 결과를 제공합니다.

- **Technical Details**: 본 논문은 생존 예측(survival prediction)의 맥락에서 데이터 선택(data selection) 문제를 확장하며, 예산 제약 하에서 최적의 모델을 향상시키기 위해 해체된 인스턴스(de-censored instance)를 라벨링하는 문제를 정의합니다. 새로운 알고리즘인 B​Bs​u​r​vBB_{surv}를 제안하며, 이는 BatchBALD 알고리즘을 기반으로 하여 생존 데이터와 다양한 인스턴스 비용을 처리할 수 있도록 조정되었습니다. 해당 알고리즘은 다항 시간(polynomial time) 내에서 하한(numerical lower bound)을 성취할 수 있음이 증명되었습니다.

- **Performance Highlights**: 여러 실제 생존 데이터셋에서 우리의 방법이 다양한 평가 지표를 통해 다른 잠재적 접근법들보다 높은 성능을 보임을 입증하였습니다. 실험 결과, 본 연구는 예산이 주어질 경우 다른 데이터 선택 알고리즘보다 우수한 일반화 능력을 지속적으로 보여주었습니다. 이는 예산 학습 분야에서 새로운 가능성을 열어줄 것으로 기대됩니다.



### Credal Transformer: A Principled Approach for Quantifying and Mitigating Hallucinations in Large Language Models (https://arxiv.org/abs/2510.12137)
- **What's New**: 이번 연구에서는 LLMs (Large Language Models)에서 발생하는 환각(hallucination) 문제를 해결하기 위한 새로운 접근 방법인 Credal Transformer를 소개합니다. 이 모델은 Transformer의 Softmax 기능에 의해 생성되는 '인공적 확신(Artificial Certainty)'을 해결하고, 불확실성을 정량적으로 통합하기 위한 신뢰할 수 있는 AI 시스템으로서의 토대를 제공합니다. Credal Attention Mechanism (CAM)을 활용하여, 단일 확률 분포 대신에 신뢰도를 반영하는 분포 집합인 'credal set'을 생성합니다.

- **Technical Details**: Credal Transformer의 핵심은 Credal Attention Mechanism(CAM)으로, 이는 전통적인 주의 메커니즘을 대체합니다. CAM은 주의 점수를 증거(evidence)로 재구성하고, Dirichlet 분포를 사용하여 분포를 매개변수화합니다. 이로 인해 몇 가지 주의 분포가 각 모델의 확신의 정도를 반영하여, 신뢰도와 모호성을 보다 명확하게 표현할 수 있게 매개변수화 됩니다.

- **Performance Highlights**: Credal Transformer는 본 연구에서 설정한 다양한 과제에 대해 우수한 성능을 보여주었습니다. 모델은 OOD (Out-of-Distribution) 입력을 효과적으로 식별하고, 높은 엔트로피 출력을 생성하여 기존 모델보다 훨씬 높은 정밀도로 불확실성을 정량화할 수 있었습니다. 추가로, 이 모델은 불확실한 질문 상황에서 자발적으로 예측을 자제하는 기능을 제공하여, 신뢰할 수 있는 질문-답변 시스템 구축에 기여할 수 있습니다.



### SafeMT: Multi-turn Safety for Multimodal Language Models (https://arxiv.org/abs/2510.12133)
- **What's New**: 최근 다중 모달 대형 언어 모델(multimodal Large Language Model, MLLM)의 사용이 증가함에 따라 이들 모델의 안전성 문제가 더욱 중요시되고 있습니다. 특히, 다중 턴 대화가 보다 일반적인 실제 상호작용에서 더 큰 위험을 초래할 수 있음에도 불구하고 기존 벤치마크는 이를 충분히 고려하지 않았습니다. 이를 해결하기 위해 우리는 SafeMT라는 벤치마크를 도입하여 유해한 질의로부터 생성된 대화 데이터와 이미지를 포함한 다양한 길이의 대화를 제공합니다.

- **Technical Details**: SafeMT는 총 10,000개의 샘플로 구성되어 있으며, 17개의 서로 다른 시나리오와 4가지 탈옥(jailbreak) 방법을 포함합니다. 우리는 Safety Index(SI)를 제안하여 MLLM의 대화 중 일반적인 안전성을 평가합니다. 이를 통해 17개의 모델의 안전성을 평가했으며 유해한 대화의 턴 수가 증가할수록 이러한 모델에 대한 공격 성공 가능성이 높아진다는 것을 발견했습니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안된 대화 안전 조정자는 대화에서 악의적인 의도를 탐지하고 MLLM에 적절한 안전 정책을 제공하는 데 효과적입니다. 여러 오픈 소스 모델의 실험 결과에서 이 조정기가 기존의 가드 모델보다 멀티턴 자동 음성 인식(multi-turn ASR)을 줄이는 데 더 효과적임을 보여주었습니다.



### Understanding the Modality Gap: An Empirical Study on the Speech-Text Alignment Mechanism of Large Speech Language Models (https://arxiv.org/abs/2510.12116)
Comments:
          Accepted to EMNLP 2025 (Main Conference)

- **What's New**: 이 연구는 LSLMs(대규모 음성 언어 모델)와 기존 파이프라인 시스템 간의 성능 차이를 체계적으로 분석하였습니다. 결과적으로 음성과 텍스트 입력 간의 '모달리티 갭(modality gap)'이 현저하게 나타났으며, 이는 LSLMs가 음성-텍스트 정렬 훈련 후 성능 저하를 경험하는 원인으로 제시됩니다. 연구진은 이러한 차이를 분석하기 위해 텍스트와 음성의 표현을 정량화하는 Alignment Path Score를 도입하였습니다.

- **Technical Details**: 연구에서는 LSLMs의 음성 및 텍스트 표현 간의 유사성을 분석하여, 깊은 층에서 표현 방향이 일치하는 경향과 크기에서의 이탈을 동시에 관찰하였습니다. 이 연구는 특히 대조적으로, 텍스트와 음성 입력의 성능 차이가 모델의 내부 표현 유사성과 밀접하게 연관되어 있음을 밝혀냈습니다. 실험을 통해 음성 표현에 대한 목표 지향적 개입이 성능을 향상시킬 수 있음을 보여주었습니다.

- **Performance Highlights**: 연구 결과, 전통적인 파이프라인 시스템에서보다 LSLMs에서 음성 입력 성능이 유의미하게 저조한 것으로 드러났습니다. 이를 해결하기 위해 LSLMs의 표현을 조정하여 음성 입력의 성능을 개선하는 알고리즘을 제시하였고, 이는 보다 높은 정확도를 이끌어낼 가능성을 보여줍니다. 연구진은 Zukunft 구성 요소들을 기반으로 음성 표현의 정렬 메커니즘을 밝혀내며, 향후 발전의 기초를 마련하였습니다.



### Chimera: State Space Models Beyond Sequences (https://arxiv.org/abs/2510.12111)
Comments:
          Published in TMLR (October 2025); 22 Pages, 6 Figures, 11 Tables

- **What's New**: 이 논문에서는 Chimera라는 새로운 통합 모델을 소개합니다. Chimera는 그래프의 구조를 직접 모델에 통합하는 독창적인 방법을 통해 도메인 특정한 편향(inductive biases)의 필요성을 제거합니다. 이 접근법은 기존의 Transformer 기반 모델들이 가지고 있었던 데이터의 비순서적 처리에서의 한계를 극복하는 데 주안점을 두고 있습니다.

- **Technical Details**: Chimera는 상태 공간 모델(State Space Models, SSMs)을 기반으로 하여 그래프의 위상(topology)을 캡처합니다. 이 모델은 기존의 위치 임베딩(position embeddings) 없이 시퀀스의 순서를 자연스럽게 포착할 수 있는 특성을 가지고 있습니다. 이를 통해 다수의 그래프 구조를 포괄하는 일반화된 방법론을 제공합니다.

- **Performance Highlights**: 실험 결과 Chimera는 GLUE에서 BERT를 0.7점 초과하여, ImageNet-1k에서 ViT보다 2.6% 더 뛰어난 성능을 발휘합니다. 또한 Long Range Graph Benchmark(LRGB)에서도 강력한 기준선들을 초과하는 성능을 나타내며, 긴 거리와 짧은 거리 노드 간 상호작용을 모두 효과적으로 모델링할 수 있음을 입증합니다.



### Deep Associations, High Creativity: A Simple yet Effective Metric for Evaluating Large Language Models (https://arxiv.org/abs/2510.12110)
Comments:
          14 pages

- **What's New**: LLM(대형 언어 모델)의 창의성 평가에 대한 연구가 주목받고 있으며, 데이터 오염(data contamination)과 비용이 많이 드는 사람 평가 인식을 극복하기 위해 PACE(Parallel Association Chains Evaluation)를 제안합니다. 이 방법은 LLM이 생성한 연관 연쇄를 기반으로 창의성을 평가하여 높은 효율성을 보여줍니다. PACE는 다양한 모델에서 Chatbot Arena Creative Writing 순위와 강한 상관관계를 가지고 있으며, 이는 창의성 평가의 새로운 가능성을 제시합니다.

- **Technical Details**: 이 연구는 LLM의 창의성을 평가하기 위해 인간 창의성 평가에서 영감을 얻어 PACE를 도입하였습니다. PACE는 LLM이 독립적으로 3개의 연관 단어를 생성하고, 20개의 단어로 구성된 연관 연쇄를 생성하는 두 단계로 구성됩니다. 이 방법은 데이터 오염 문제를 완화하고, LLM의 창의력을 효율적으로 측정할 수 있는 구조화된 접근 방식을 제공합니다.

- **Performance Highlights**: 연구 결과, 최첨단 LLM은 일반 인간 그룹과 유사한 수준의 성능을 보였으나, 전문가 인간의 수준에는 미치지 못했습니다. PACE와 Arena Creative Writing 순위 간에는 0.739의 상관계수가 존재하며, 이는 LLM의 창의적 연결 능력을 평가하는 데 효과적임을 나타냅니다. 이 결과는 데이터 오염 문제를 해결할 수 있는 한 방법으로, LLM의 창의적 잠재력을 탐색하는 데 기여할 수 있습니다.



### An AI-Based Behavioral Health Safety Filter and Dataset for Identifying Mental Health Crises in Text-Based Conversations (https://arxiv.org/abs/2510.12083)
Comments:
          Main Text: 2943; Abstract: 256; Tables and Figures: 5

- **What's New**: 염려되는 정신적 위기 상황에 대한 대응을 내기 위해 Verily Behavioral Health Safety Filter (VBHSF)가 개발되었습니다. 이 연구는 1,800개의 시뮬레이션 메시지를 포함한 Verily Mental Health Crisis Dataset과 794개의 정신 건강 관련 메시지를 포함한 NVIDIA Aegis AI Content Safety Dataset을 사용해 VBHSF의 성능을 평가했습니다. 특히, VBHSF는 정신 건강 위기에 대한 민감도와 특이성을 높이는 것을 목표로 하고 있습니다.

- **Technical Details**: VBHSF는 clinician labels를 기반으로 성능을 평가하였으며, Verily Mental Health Crisis Dataset v1.0에서는 민감도(0.990)와 특이성(0.992)을 달성했습니다. 특정 위기 카테고리 식별에서 F1-score는 0.939에 이르며, 감도는 0.917부터 0.992, 특이성은 0.978 이상을 기록했습니다. NVIDIA Aegis AI Content Safety Dataset 2.0에 대해서도 높은 민감도(0.982)와 정확도(0.921)를 보여 주었으나, 특이성은 다소 낮아졌습니다.

- **Performance Highlights**: VBHSF는 NVidia NeMo 및 OpenAI Omni Moderation Latest와 비교해 성능 메트릭스에서 우수한 결과를 보였습니다. 모든 경우에서 감도는 유의미하게 높았으며(p < 0.001), NVidia NeMo와 비교했을 때 특이성 또한 월등히 높았습니다(p < 0.001). 반면 OpenAI Omni Moderation Latest와의 비교에서는 특이성이 유의미하지 않았습니다(p = 0.094). 전반적으로 VBHSF는 민감도를 우선시하는 강력하고 일반화 가능한 성능을 보여 주어 의료 응용 프로그램에 적합한 중요한 특성을 갖추고 있습니다.



### Enhancing Neural Code Representation with Additional Contex (https://arxiv.org/abs/2510.12082)
Comments:
          34 pages, 7 figures, 11 tables

- **What's New**: 이번 연구에서는 자동화된 프로그램 이해(automated program comprehension)의 성능을 향상시키기 위해 소스 코드 외에도 컨텍스트 정보를 활용하는 방법에 대해 다룹니다. 최신 딥러닝 모델들이 소스 코드만을 분석하는 경향이 있지만, 이 연구는 버전 이력(version history)이나 호출 그래프(call graph)와 같은 정보를 통합함으로써 코드 이해도를 높일 수 있음을 보여줍니다. 이를 통해 코드 클론 탐지(code clone detection) 및 코드 요약(code summarisation) 성능이 개선됨을 입증하였습니다.

- **Technical Details**: 연구는 SeSaMe와 CodeSearchNet이라는 두 개의 데이터셋을 사용하여, 각 모델(CodeBERT, GraphCodeBERT, CodeT5 등)이 코드 전용 환경과 컨텍스트 증강 환경에서 어떻게 수행되는지를 평가하였습니다. 특히, 버전 이력과 호출 그래프를 활용하여 추가된 컨텍스트가 모델 성능에 미치는 영향을 분석했습니다. 이를 통해 각 모델이 소스 코드와 추가된 컨텍스트를 통합하여 더 나은 성과를 도출할 수 있음을 확인하였습니다.

- **Performance Highlights**: 실험 결과, 버전 이력은 클론 탐지에서 F1 점수(+15.9%)와 요약에서 METEOR 점수(+5.6%)를 개선하는 가장 효과적인 신호로 확인되었습니다. 특히, 다양한 컨텍스트를 결합함으로써 성능을 극대화할 수 있으며, 여러 컨텍스트의 조합은 최대 21.48%의 macro-F1 점수가 개선되는 효과를 나타냈습니다. 인간 평가에서도 컨텍스트가 증가된 요약이 정확성과 내용 적합성 측면에서 일반적으로 보다 선호되는 경향이 드러났습니다.



### A Review on Domain Adaption and Generative Adversarial Networks(GANs) (https://arxiv.org/abs/2510.12075)
- **What's New**: 이번 논문은 컴퓨터 비전에서의 중요한 문제인 데이터 부족을 해결하기 위해 Domain Adaptation(도메인 적응)에 초점을 맞추고 있습니다. 이미지 분류와 같은 분야에서는 신뢰할 수 있는 데이터 수집 방법이 필수적입니다. 이 논문은 이전의 벤치마크 결과와 비교 가능한 성과를 내기 위한 방법을 모색합니다.

- **Technical Details**: 논문에서는 특정 데이터셋에서 훈련된 모델을 사용하여 다른 도메인의 유사한 데이터에 대해 예측하는 방법론을 논의합니다. 구체적으로, 비행기 그림으로 훈련된 모델이 실제 비행기 이미지에 대한 예측을 수행하는 방법이 포함됩니다. 이러한 접근 방식은 라벨링된 데이터의 높은 비용 문제를 해결하는 데 도움을 줄 수 있습니다.

- **Performance Highlights**: 이 연구에서 제안된 Domain Adaptation 방법론은 다양한 도메인 간의 데이터 부족 문제를 해결하여 더 뛰어난 성능을 보여줄 것으로 기대됩니다. 제안된 방법을 통해 효율적인 이미지 분류와 같은 여러 컴퓨터 비전 작업에서 향상된 결과를 도출할 수 있을 것입니다.



### MEASURE: Multi-scale Minimal Sufficient Representation Learning for Domain Generalization in Sleep Staging (https://arxiv.org/abs/2510.12070)
Comments:
          12 page, 7 figures, uses this http URL

- **What's New**: 이 논문에서는 자동 수면 단계 분류를 위한 새로운 MEASURE (Multi-scalE minimAl SUfficient Representation lEarning) 프레임워크를 제안합니다. 이 프레임워크는 도메인 일반화(domain generalization) 문제를 해결하기 위해 고안되었으며, 다양한 다중 스케일 기능을 활용하여 도메인 차이를 최소화하는 데 집중합니다. MEASURE는 과도한 도메인 관련 정보를 줄이는 데 특히 중점을 두어 강력한 도메인 불변 특성을 추출할 수 있도록 합니다.

- **Technical Details**: MEASURE 프레임워크는 최소한의 적절한 표현 학습(minimal sufficient representation learning)을 활용하여 여러 레이어의 인코더 특징에 걸쳐 작동하도록 설계되었습니다. 이는 수면 신호에서의 다양한 시간적 및 스펙트럼적 특성을 효과적으로 포착할 수 있도록 보장합니다. 또한, 제안된 MEASURE 방식은 기존 전달 학습 기술과 비교하여 도메인 불변 학습을 더욱 조화롭게 이끌어내는 이론적인 분석을 제공합니다.

- **Performance Highlights**: 제안된 MEASURE는 SleepEDF-20 및 MASS 데이터셋에서 기존의 최첨단 방법들에 비해 일관되게 우수한 성능을 보였습니다. 특히, 이 방법은 수면 단계 분류의 정확성을 높이는 데 기여하며, 다양한 도메인에서 효과적으로 일반화할 수 있는 능력을 보여줍니다. 이러한 결과는 수면 장애 진단 및 치료 분야에서의 인공지능 활용에 긍정적인 영향을 미칠 것으로 예상됩니다.



### Your VAR Model is Secretly an Efficient and Explainable Generative Classifier (https://arxiv.org/abs/2510.12060)
- **What's New**: 본 연구는 기존의 확산 기반(‘diffusion-based’) 모델 대신 최근 발전된 시각 자율회귀(‘visual autoregressive’, VAR) 모델링을 활용한 새로운 생성 분류기(‘generative classifier’)를 제안합니다. 특히 제안된 Adaptive VAR Classifier$^+$ (A-VARC$^+$)는 정확성과 추론 속도 간의 우수한 균형을 달성하여, 실제 적용성을 크게 개선합니다. 또한, VAR 기반 방법론은 확산 기반 방법들과 근본적으로 다른 성질을 보이는데, 이는 시각적 설명 가능성과 ‘catastrophic forgetting’에 대한 저항력을 포함합니다.

- **Technical Details**: A-VARC는 정확도를 향상시키기 위해 두 가지 새로운 기술을 통합합니다: 우려한 추정치를 통해 정확도를 높이는 likelihood smoothing과 모델의 다중 스케일 아키텍처를 활용하여 추론 속도를 가속화하는 부분 스케일 후보 가지치기(‘partial-scale candidate pruning’)입니다. 이러한 접근법은 쉽게 조정 가능한 프레임워크를 만들어 내어, 기존 naive VARC보다 상당한 성능 향상을 이끌어냅니다. A-VARC+는 최근 제안된 Condition Contrastive Alignment (CCA) 기법을 활용하여 세밀하게 조정된 개선된 버전입니다.

- **Performance Highlights**: A-VARC+는 ImageNet-100 데이터셋에서 확산 기반 모델과 비교하여 1% 미만의 성능 저하로 160배의 속도 향상을 기록합니다. 이는 생성 분류기의 계산 부담을 상당히 경감시켜 주며 실제 적용 가능성을 높입니다. 특히, VAR 기반 방법론은 개별 텍스트와 타겟 레이블 간의 연관성을 포착할 수 있는 시각적 설명 능력을 제공하며, 기존의 ‘discriminative classifiers’에 비해 클래스 증가 학습에서 내재적 내성을 보입니다.



### APCE: Adaptive Progressive Context Expansion for Long Context Processing (https://arxiv.org/abs/2510.12051)
Comments:
          NeurIPS 2025 Workshop: ML For Systems

- **What's New**: 이 논문은 Long-Context Transformer Models (LCTMs) 배포 시 메모리 효율성 및 ContextRot 현상이라는 두 가지 주요 문제를 해결하기 위해 다양한 입력 조각을 선정하는 새로운 접근법인 APCE를 제안합니다. APCE는 낮은 차원의 의미적 유사성 비교를 통해 입력의 가장 중요한 조각만을 선택하여, 메모리 사용을 줄이고 성능 저하를 방지합니다. 이 방식은 관련 하드웨어나 CUDA 환경에 대한 의존성을 줄이고 다양한 배포 시스템에 호환되는 솔루션으로 발전할 가능성을 가집니다.

- **Technical Details**: APCE는 입력 토큰을 청킹(chunking)하여 각 청크의 저차원 임베딩을 생성한 후, 현재 쿼리와의 유사성을 기준으로 가장 관련성이 높은 상위 k개의 청크를 선택합니다. 이 과정에서 APCE는 기존의 Sparse 방법들과는 달리 입력 청크를 직접 작업하며, 메모리에 한번만 계산된 청크 임베딩은 지속적으로 캐시되어 유사성 비교를 신속하게 수행합니다. 이러한 접근법은 시스템 자원의 제약 속에서도 비동기적 토큰 생성을 지원하여 초기 응답 시간을 줄이는 이점을 제공합니다.

- **Performance Highlights**: APCE는 BookSum 데이터 세트를 사용하여 장기 문서 요약을 평가하며, 전체 밀집(Full Dense) 기준 모델에 비해 유사하거나 우수한 성능을 보여주었습니다. 특히, 전체 입력의 50%-70%만을 사용하면서도 KV-cache와 self-attention 메모리 효율성이 개선되는 결과를 도출했습니다. 추가적으로, APCE의 Reprioritization을 통해 성능 및 효율성의 균형을 평가하였으며, 향후 연구의 방향성을 제시하였습니다.



### Generative AI and Firm Productivity: Field Experiments in Online Reta (https://arxiv.org/abs/2510.12049)
Comments:
          Keywords: Field Experiments, Generative AI, Productivity, Retail Platforms, Consumer Experience. JEL codes: C93, D24, L81, M31, O3

- **What's New**: 본 연구는 Generative Artificial Intelligence (GenAI)가 기업 생산성에 미치는 영향을 대규모 무작위 실험을 통해 측정했습니다. 2023년과 2024년 동안, 세계적인 교차 국적 온라인 리테일 플랫폼에 GenAI 기반의 향상 기능이 7개 소비자 샘플 비즈니스 워크플로에 통합되었습니다. 결과적으로 GenAI의 채택은 판매량을 크게 증가시켰으며, 이는 총 요소 생산성(total factor productivity)의 향상을 직접적으로 반영합니다.

- **Technical Details**: 본 연구는 GenAI를 통해 기업과 워크플로 수준에서의 생산성에 미치는 인과 관계를 입증하기 위한 대규모 현장 실험 데이터를 활용하였습니다. 각 워크플로는 검색 쿼리 개선, 제품 설명 생성 등 다양한 기능을 갖추고 있으며, 실험은 소비자 기반에서 수행되었습니다. 소비자 및 제품 수준의 상세한 데이터를 활용하여 GenAI의 단기적인 성과 영향을 평가하였으며, 매출 및 전환율 같은 주요 성과 지표를 분석했습니다.

- **Performance Highlights**: GenAI의 채택 결과 생산성이 상당한 수준으로 증가했으며, 특정 워크플로에서는 최대 16.3%의 매출 증가를 관찰했습니다. 특히 고객 서비스 및 검색 애플리케이션에서 가장 두드러진 향상이 있었습니다. 연구에서는 또한 작은 규모의 판매자와 경험이 적은 소비자가 GenAI 향상으로부터 비례적으로 더 큰 이익을 누리는 경향이 발견되었습니다.



### Hierarchical Alignment: Surgical Fine-Tuning via Functional Layer Specialization in Large Language Models (https://arxiv.org/abs/2510.12044)
- **What's New**: 이 논문에서는 기존의 대규모 언어 모델(Large Language Models, LLMs) 정렬 기술이 모든 레이어에 대해 동일한 최적화 압력을 가하는 방식을 비판하고, 모델의 기능적 블록에 목표를 둔 방법인 계층 정렬(Hierarchical Alignment)을 제안합니다. 이 새로운 접근법은 구문(local), 논리(intermediate), 사실(global)과 같은 다양한 기능 블록을 차별적으로 다루어, 각 블록에 맞춘 정렬을 수행하여 예측 가능한 성능 개선을 이루어냅니다.

- **Technical Details**: 저자들은 향상된 로직과 실제 일치성(factual consistency)을 위해 대규모 언어 모델의 레이어를 구문, 논리, 사실성으로 나누어 선택적 세밀 조정(surgical fine-tuning)을 적용했습니다. 이 과정에서 LoRA(Low-Rank Adaptation)를 활용하여 각 블록의 세부 최적화 요청을 명확히 했습니다. 실험은 LLaMA-3.1-8B 및 Qwen1.5-7B 모델을 대상으로 진행되었으며, Local-Align과 Global-Align을 통해 각각의 효과를 입증했습니다.

- **Performance Highlights**: 계층 정렬 방식을 적용한 결과, Local-Align은 문법적 유창성을 향상시키고, Global-Align은 사실적 일관성과 논리적 일관성을 크게 개선하는 것으로 나타났습니다. 특히, 계층 정렬 전략들은 기존의 DPO에서 목격된 '정렬세(alignment tax)'를 피하며, 유창성의 향상과 논리적 추론의 저하를 동시에 방지했습니다. 이 연구는 대규모 언어 개발의 방향을 구조 인식 정밀 조정으로 전환하여 더 진보된 AI 시스템의 가능성을 보여줍니다.



### Multi-stage Prompt Refinement for Mitigating Hallucinations in Large Language Models (https://arxiv.org/abs/2510.12032)
Comments:
          22 pages, 6 figures

- **What's New**: 최근 대형 언어 모델(LLMs)의 발전은 자연어 이해 및 생성 작업에서 강력한 성능을 보여주었습니다. 그러나 LLMs는 여전히 허위정보(hallucinations)를 생성하는 문제에 직면해 있습니다. 본 논문에서는 잘못된 형태의 프롬프트(prompt)에서 발생하는 문제를 체계적으로 개선할 수 있는 Multi-stage Prompt Refinement (MPR) 프레임워크를 도입합니다.

- **Technical Details**: MPR은 여러 단계에 걸쳐 잘못된 프롬프트를 개선하는 시스템화된 방법론으로, 각 단계는 문장 부호, 오타, 주요 용어의 오용 등 특정 오류를 해결합니다. 이 과정에서는 이를 위해 조정된 소형 언어 모델(SLMs)을 사용하며, 추가적인 맥락을 포함시켜 프롬프트의 명확성을 점진적으로 향상시킵니다. 자가 반성(self-reflection) 메커니즘과 순위를 매기는 방법을 활용하여 가장 관련성이 높은 입력을 우선합니다.

- **Performance Highlights**: 실험 결과에 따르면 MPR으로 개선된 프롬프트는 원본보다 85% 이상의 승률을 기록하여 허위정보를 줄이고 LLM의 출력 정확도를 향상시킵니다. 또한 MPR은 기존의 사후 사기완화(hallucination mitigation) 프레임워크와 결합할 수 있어 그 활용성을 더욱 높일 수 있음을 보여줍니다. MPR은 다양한 도메인에서 LLM의 신뢰성을 향상시키기 위한 경량화되고 적응 가능한 솔루션을 제공합니다.



### CPR: Mitigating Large Language Model Hallucinations with Curative Prompt Refinemen (https://arxiv.org/abs/2510.12029)
Comments:
          2024 IEEE International Conference on Systems, Man, and Cybernetics (SMC), 7 pages, 2 figures

- **What's New**: 최근 큰 언어 모델(LLMs)의 발전으로 인해 다양한 프롬프트에 대한 응답 생산에서 유창성을 보이지만, 신뢰성을 해치는 'hallucination' 사실가 생성하는 문제가 발생하고 있습니다. 이러한 오류의 흔한 원인은 사용자들이 비구조적이거나 모호한 프롬프트를 사용하는 것입니다. 이를 해결하기 위해, 우리는 Curative Prompt Refinement (CPR)라는 프레임워크를 도입하여 인포매티브한 작업 설명을 생성하고 사용자 의도와 프롬프트를 정렬합니다.

- **Technical Details**: CPR은 불완전하게 형성된 프롬프트를 청소하고, 작은 언어 모델(SLM)을 통해 사용자 의도를 정확히 반영하는 추가 정보를 생성하여 LLM의 출력을 향상시키는 것을 목표로 합니다. 이를 위해 세 가지 데이터셋을 활용하여 SLM을 미세 조정합니다: 문법 및 구두점 교정을 위한 WikiEn, 질문 변환을 위한 MQR, 설명 생성을 위한 WikiD 데이터셋을 사용합니다. 이 과정에서 수치적으로 높은 성능을 유지하며, 필요한 파라미터만을 선택적으로 업데이트하는 low-rank adaptation (LoRA) 기법을 사용하여 효율성을 극대화합니다.

- **Performance Highlights**: CPR을 적용한 실험에서, LLM의 출력 품질이 상당히 향상되었으며 hallucination 회피율이 크게 증가했습니다. 예를 들어, CPR를 활용한 프롬프트가 GPT-3.5 모델을 사용할 경우, 96%의 승률을 기록했으며 이는 기존의 형성이 잘못된 프롬프트에 비해 99%의 승률을 보여줍니다. 또한, CPR은 경량화되어 있으며 다양한 LLM에 모델 의존성 없이 쉽게 통합될 수 있어 비전문가 사용자에게도 접근성을 높입니다.



### PanoTPS-Net: Panoramic Room Layout Estimation via Thin Plate Spline Transformation (https://arxiv.org/abs/2510.11992)
- **What's New**: 이 논문에서는 단일 파노라마 이미지로부터 방의 3D 레이아웃을 정확하게 추정하기 위한 새로운 모델인 PanoTPS-Net을 제안합니다. 이 모델은 CNN(Convolutional Neural Network)을 사용하여 이미지에서 높은 수준의 특성을 추출하고, Thin Plate Spline(TPS) 공간 변환을 통합한 두 단계로 나누어진 구조를 띱니다. 이를 통해 PanoTPS-Net은 정육면체 및 비정육면체 레이아웃 모두를 효과적으로 일반화할 수 있는 능력을 갖추게 되었습니다.

- **Technical Details**: PanoTPS-Net은 입력 파노라마 이미지의 고급 특성을 추출한 후, 이 특성을 활용하여 TPS 변환 매개변수를 예측합니다. TPS 변환은 하나의 형태를 다른 형태로 부드럽고 유연하게 변형하는데 사용되는 수학적 기법으로, 이미지 왜곡 및 메쉬 변형에 널리 활용됩니다. 이 논문에서는 TPS 매개변수를 추정함으로써 방 레이아웃을 이미지 왜곡 문제로 재정의하였으며, 이 과정에서 고정밀의 고유 매개변수를 학습할 수 있는 격자 기반 공간 변환 네트워크(STN)를 적용하였습니다.

- **Performance Highlights**: 제안된 방법은 PanoContext, Stanford-2D3D, Matterport3DLayout 및 ZInD 데이터 세트에서 각각 3DIoU 값 85.49, 86.16, 81.76 및 91.98을 기록하며 높은 정확성을 보였습니다. 이러한 결과는 TPS 변환이 파노라마 이미지와의 호환성에서 유효함을 강조하며, 모델이 정육면체 및 비정육면체 방 레이아웃 추정 작업에서도 강건함을 나타냅니다. 또한, 코드가 공개되어 연구자들이 손쉽게 접근할 수 있도록 하여 향후 연구의 기반을 다지고 있습니다.



### Conjecturing: An Overlooked Step in Formal Mathematical Reasoning (https://arxiv.org/abs/2510.11986)
- **What's New**: 이 논문에서는 자동 형식화(autoformalisation)의 한계를 지적하고, 이를 극복하기 위해 추측(conjecturing) 단계의 중요성을 강조합니다. 기존의 데이터 세트를 보강하여 ConjectureBench를 생성하고, LLM의 추측 능력을 독립적인 작업으로 평가할 수 있는 새로운 평가 프레임워크와 메트릭을 설계했습니다. 이 접근 방식은 자동 형식화의 성과를 보다 정확하게 평가할 수 있도록 합니다.

- **Technical Details**: 논문에서는 GPT-4.1 및 DeepSeek-V3.1과 같은 기초 모델의 성능을 평가합니다. 특히, 평가 과정에서 추측을 포함했을 때 자동 형식화 성과가 과대평가된다는 것을 발견했습니다. 또한 Lean-FIRe라는 추론 시간(inference-time) 방법을 설계하여 자동 형식화 작업의 성능을 향상시킵니다.

- **Performance Highlights**: 이 연구에서는 GPT-4.1으로 13개의 PutnamBench 문제를, DeepSeek-V3.1으로 7개의 문제를 성공적으로 엔드 투 엔드(end-to-end) 자동 형식화했습니다. LLM은 정확한 추측을 생성할 수 있는 지식을 보유하고 있지만, 자동 형식화 성능을 향상시키려면 추측을 독립적인 작업으로 다뤄야 한다는 점을 강조합니다. 이는 향후 연구 방향에 중요한 기초 자료가 될 것입니다.



### Learning Dynamics of VLM Finetuning (https://arxiv.org/abs/2510.11978)
- **What's New**: 본 논문에서는 비전-언어 모델(Vision-Language Models, VLMs)의 선호 기반 파인튜닝이 본질적으로 불안정하며, 잘못된 부정 사례가 훈련을 불안정하게 만들 수 있음을 다룹니다. 새로운 최적화 기법인 Cooling-Weighted DPO (CW-DPO)를 소개하여, 두 가지 단계에서 훈련 궤적을 명시적으로 모델링하고 활용합니다. 이는 훈련 과정의 동적 변화를 반영하여 낙관성 과다를 억제하고 안정성을 증대시킵니다.

- **Technical Details**: CW-DPO는 두 단계로 구성된 전략으로, 첫 번째 단계에서는 '부드러운 부정들'(gentle negatives)을 사용하여 훈련의 안정성을 높입니다. 두 번째 단계에서는 DPO 목표를 적용하며, 여기서 부정 항은 '냉각 가중치'(cooling weight)를 통해 스케일링되어, 모델의 평균 토큰 로그-확률에 기반하여 비정보적인 기울기를 억제합니다. 이 과정에서 온-정책 부정 항(on-policy negatives)과 데이터셋 부정 항(dataset negatives)을 혼합하여 대조의 신선함을 유지합니다.

- **Performance Highlights**: CW-DPO는 다양한 VLM 작업에서 기존의 SFT 전용 및 기본 DPO 방법들보다 안정적인 최적화, 더 나은 보정(calibration), 높은 쌍별 승률(pairwise win-rates)을 달성합니다. 또한, CW-DPO는 더 적은 단계에서 수렴하며, 냉각 가중치 메커니즘이 이러한 성능 향상의 주요 요인임을 보여줍니다. 실험 결과는 '학습 동적을 부드럽게 한 후 선호를 냉각하는 간단하면서도 일반적인 원칙'의 중요성을 입증합니다.



### CTIArena: Benchmarking LLM Knowledge and Reasoning Across Heterogeneous Cyber Threat Intelligenc (https://arxiv.org/abs/2510.11974)
Comments:
          Under peer-review

- **What's New**: 이 논문은 사이버 위협 정보(Cyber Threat Intelligence, CTI) 분야에서 대형 언어 모델(Large Language Models, LLMs)을 평가하기 위한 최초의 벤치마크인 CTIArena를 제안합니다. 기존 연구들은 단일 지식 출처에 의존하여 LLM의 성능을 제한적으로 평가했으나, CTIArena는 다양한 출처와 지식 증강을 통합하여 더욱 포괄적이고 실용적인 평가를 제공합니다. 이 벤치마크는 CTI 분석의 범위를 포함하는 구조적, 비구조적 및 혼합된 9개 작업으로 나뉘어 있습니다.

- **Technical Details**: CTIArena는 LLM을 평가하기 위해 세 가지 카테고리로 구분된 9개 작업을 제공합니다. 각 작업은 고유한 프로세스를 통해 CTI 분석을 지원하며, LLMs가 이해하고 적용할 수 있는 다양한 지식 설정을 포함합니다. 이러한 표준화된 접근 방식은 LLM이 실제 보안 시나리오에서 어떻게 작동하는지를 명확히 평가할 수 있는 기회를 제공합니다.

- **Performance Highlights**: 평가 결과, 대부분의 LLM들은 닫힌 책(closed-book) 설정에서는 성능이 저조했으나, 보안 관련 지식으로 증강되었을 때 성능이 현저히 향상되었습니다. 이 연구는 일반 목적의 LLM들이 CTI에 응용되는 데 한계가 있으며, 해당 도메인에 맞춘 기술의 필요성을 강조했습니다. 신뢰성을 높이기 위해 후속 전문가 리뷰를 통해 평가된 QA 쌍의 데이터를 정제하는 프로세스도 마련되었습니다.



### Direct Multi-Token Decoding (https://arxiv.org/abs/2510.11958)
- **What's New**: 본 논문에서는 Direct Multi-Token Decoding(DMTD)이라고 불리는 새로운 추론 패러다임을 제안합니다. DMTD는 기존의 디코더 전용 트랜스포머 구조의 한계를 극복하여, 후단 레이어만을 사용하여 여러 토큰을 동시에 생성할 수 있도록 하고 있습니다. 이 방식은 초기 및 중간 레이어를 반복적으로 탐색할 필요 없이 효율적인 토큰 생성을 가능하게 하여, 속도를 최대 2배로 향상시키면서도 성능 저하를 최소화합니다.

- **Technical Details**: DMTD는 고정된 다중 토큰 주기로 작동하며, 초기 사이클에서 단 한번 전체 순방향 전파를 수행하고 이후에 마지막 레이어를 재사용하여 여러 토큰을 연속적으로 디코딩합니다. 훈련 과정에서는 사이클 마스킹 전략을 이용해 단일 시퀀스 내에서 다중 미래 토큰을 예측하도록 모델을 학습시킵니다. 이 방법은 복잡한 계산을 요구하는 작업과 간단한 작업에서 서로 다른 레이어를 효율적으로 활용할 수 있도록 합니다.

- **Performance Highlights**: DMTD는 Qwen3-4B 모델에서 실험적으로 검증되었으며, 두 개의 토큰을 디코딩할 때 원본 모델에 비해 성능이 100% 유지되는 결과를 보였습니다. 세 개와 네 개의 토큰을 각각 디코딩할 때도 각각 98.4% 및 96.3%의 성능을 유지하며, 최적화된 디코딩 회로를 통해 추론 시간을 최대 2배 단축할 수 있습니다. 이 성능 저하는 훈련 데이터가 증가함에 따라 더욱 개선될 것으로 기대됩니다.



### Y-shaped Generative Flows (https://arxiv.org/abs/2510.11955)
- **What's New**: 본 연구에서는 현대의 연속 시간 생성 모델들이 데이터와의 관계에서 V자형 이동만을 고려하는 문제를 해결하기 위해 새로운 Y자형 생성 흐름(Y-shaped generative flows)을 소개합니다. 이 모델은 확률 질량을 함께 이동시키고 특정 목표 지점으로 나누어지며, 이를 통해 보다 효율적인 데이터를 생성할 수 있습니다.

- **Technical Details**: 제안된 모델은 새로운 velocity-powered transport cost를 기반으로 하고 있으며, 이는 0과 1 사이의 비선형 지수를 가집니다. 이러한 형태의 비용 함수는 빠르고 공동으로 질량을 움직이는 데 보상을 부여하여, 모델의 성능을 향상시킵니다. 또한, 이를 확장 가능한 신경 ODE(neural ODE) 훈련 목표로 구체화하여 적용합니다.

- **Performance Highlights**: Y-플로우(Y-flows) 모델은 합성 데이터(synthetic datasets), 이미지 및 생물학적 데이터셋에서 계층적으로 인식 가능한 구조를 회복합니다. 이 모델은 강력한 흐름 기반 기준선에 비해 분포 지표(distributional metrics)를 개선하고, 목표에 도달하기 위해 필요한 통합 단계(integration steps)를 줄이는 성능을 보여줍니다.



### Sculpting Latent Spaces With MMD: Disentanglement With Programmable Priors (https://arxiv.org/abs/2510.11953)
- **What's New**: 이번 논문에서는 머신 러닝의 주요 목표 중 하나인 분리된 표현(disentangled representations) 학습을 다룹니다. 기존의 주요 방법인 Variational Autoencoder (VAE) 프레임워크가 Kullback-Leibler (KL) divergence 패널티를 사용하여 잠재 공간을 정규화하는 대신, 이 논문은 KL 기반 정규화기가 신뢰할 수 없다는 직접적인 증거를 제시합니다. 이를 해결하기 위해 Maximum Mean Discrepancy (MMD) 기반의 Programmable Prior Framework를 도입하여 잠재 공간을 명확히 조형할 수 있습니다.

- **Technical Details**: 이 논문에서는 잠재 정보를 구성하는 두 가지 measurable components를 분리하여 다룹니다. 첫 번째는 데이터에 대해 상호 독립적인 특징 분포를 학습하는 것이며, 두 번째는 학습된 특징이 의미 있는 속성에 정렬되도록 하는 것입니다. 이를 위해 MMD를 사용하여 잠재 공간의 집합 후방 분포를 조형함으로써 기존 VAE의 한계를 극복합니다.

- **Performance Highlights**: 제안된 방법은 CIFAR-10과 Tiny ImageNet과 같은 복잡한 데이터셋에서 state-of-the-art mutual independence를 달성했으며, 일반적인 재구성 품질 손실 없이 뛰어난 성능을 발휘합니다. 또 다른 주요 기여로는 Latent Predictability Score (LPS)라는 새로운 비지도 메트릭을 제안하여 학습된 특징의 상호 독립성 정도를 정량화할 수 있는 방법을 제공합니다.



### TopoAlign: A Framework for Aligning Code to Math via Topological Decomposition (https://arxiv.org/abs/2510.11944)
- **What's New**: 이번 논문에서는 Autoformalisation(자동 형식화) 문제 해결을 위한 TopoAlign 프레임워크를 제안합니다. 이는 코드의 구조적 요소를 Formal Languages(형식 언어)와 정렬함으로써 Math LLMs의 훈련에 기여합니다. 특히, 기존의 인포멀한 수학적 문제를 포멀한 수학 문장으로 변환하는 데 있어 구조적 정렬을 활용하여 훈련 자원을 확장하는 방식입니다.

- **Technical Details**: TopoAlign은 코드를 docstrings, 주요 함수, 의존성 함수 등으로 분해한 후, 이를 Lean 4의 형식적 문장 구조를 반영하는 유사한 형태로 재조립합니다. 이를 통해 Math LLMs는 정렬된 데이터에서 구조적 패턴을 배울 수 있으며, 효율적으로 문제 해결 능력을 전이할 수 있습니다. 특히, code autoformalisation(CAF)라는 새로운 작업을 도입하여 정렬된 코드 데이터를 활용하여 자동 형식화 과제를 대체합니다.

- **Performance Highlights**: TopoAlign을 통해 훈련된 두 개의 모델, DeepSeek-Math와 Herald는 다양한 벤치마크에서 성능 향상을 보여주었습니다. DeepSeek-Math는 BEq@10에서 17.77%, typecheck@10에서 68.82%의 성능 향상을 달성하였으며, Herald 또한 약간의 개선을 기록하였습니다. 이는 정렬된 코드 데이터로의 훈련이 전문화된 모델에서도 효과적이라는 것을 입증합니다.



### Discrepancy Detection at the Data Level: Toward Consistent Multilingual Question Answering (https://arxiv.org/abs/2510.11928)
Comments:
          Long paper accepted at EMNLP 2025

- **What's New**: 최신 연구에서는 다국어 질문 답변 시스템(Multilingual question answering, QA)에 있어 사실 일관성을 보장하는 MIND라는 파이프라인을 제안합니다. 이 시스템은 문화적 변화를 고려하면서도 객관적인 질문에 대한 사실적 일관성을 유지하는 데 초점을 맞추고 있습니다. 특히, 문화적으로 민감한 질문들에 대해 지역과 맥락에 따라 차이가 나는 답변을 식별하는 기능을 강조합니다.

- **Technical Details**: MIND는 사용자 참여 기반의 사실 확인 파이프라인(user-in-the-loop fact-checking pipeline)으로, 다국어 QA 지식 기반에서의 사실 및 문화적 불일치를 탐지합니다. 이 연구는 모성 및 유아 건강 분야의 이중 언어(QA 시스템)에서 MIND의 효율성을 평가하고, 사실 및 문화적 불일치가 주석 처리된 이중 언어 질문 데이터셋을 공개합니다. 랜덤한 다른 도메인에서도 MIND의 일반화 능력을 테스트하여 범용성을 체크합니다.

- **Performance Highlights**: MIND는 모든 테스트 케이스에서 불일치를 신뢰성 있게 식별함으로써, 더 문화적으로 민감하고 사실적으로 일관된 QA 시스템의 개발을 지원합니다. 이러한 성과는 다국어 환경에서도 사실적이고 신뢰할 수 있는 정보 제공을 가능하게 하는데 기여합니다. 특히, 문화적 차이를 고려한 접근 방식이 QA 시스템의 품질을 향상시키는 데 매우 중요하다는 점을 강조합니다.



### Indoor Localization using Compact, Telemetry-Agnostic, Transfer-Learning Enabled Decoder-Only Transformer (https://arxiv.org/abs/2510.11926)
Comments:
          11 pages, 12 Figures

- **What's New**: 새롭게 제안된 Locaris는 단순한 decoder-only LLM을 이용한 실내 로컬라이제이션 시스템으로, Wi-Fi 데이터 수집과정에서 필요한 전처리를 최소화합니다. 이 시스템은 각 Access Point(AP)로부터의 측정을 토큰으로 처리하여, 다양한 Wi-Fi 기기에서 수집한 신호를 통합적으로 활용할 수 있습니다. 특히, Locaris는 몇 가지 테스팅 데이터 세트에 대해 조정하여 경량화되고 일반화된 장치 위치 매핑을 학습합니다.

- **Technical Details**: Locaris는 메모리 사용량을 줄이면서도 로컬라이제이션 정확도를 유지하기 위해 LoRA 모듈을 활용한 파라미터 효율적 적응 전략을 적용합니다. 이 방식은 고정된 LLaMA 백본을 유지하면서 신속한 미세 조정을 가능하게 하여, 서로 다른 환경과 하드웨어에서 복잡한 전처리 없이 작동합니다. 또한, 별도의 층을 통해 결정론적 회귀 결과를 생성하여 실용적인 활용을 지원합니다.

- **Performance Highlights**: Locaris는 다양한 센서 유형과 공급업체 구성에서도 뛰어난 성능을 보여주며, 기존의 기술들을 초월하거나 동등한 성능을 유지합니다. 이 시스템은 결측 AP나 모달리티 하에서도 안정적인 성능을 발휘하여 실제 환경 배치에서도 효율적으로 작동할 수 있습니다. 실험 결과, Locaris는 몇 백 샘플만으로도 서브 미터 정확도를 달성하며, 이전에 보지 못한 장치와 배치 시나리오에도 높은 정확도를 지속적으로 유지합니다.



### Integrating Sequential and Relational Modeling for User Events: Datasets and Prediction Tasks (https://arxiv.org/abs/2510.11903)
- **What's New**: 이번 연구에서는 개인 이벤트(personal events)와 관계 이벤트(relational events)를 통합적으로 모델링하는 새로운 방법론을 제안합니다. 기존 연구는 주로 이 두 가지 이벤트 유형을 별도로 처리했으나, 실제 사용자 행동을 더 잘 반영하기 위해 이 둘을 함께 고려하는 것이 중요하다는 점을 강조합니다. 데이터셋과 예측 작업을 포함한 자원을 공개하여 연구자들이 통합된 사용자 이벤트 모델링을 지원하도록 유도합니다.

- **Technical Details**: 본 논문에서는 개인 이벤트와 관계 이벤트를 효과적으로 포착할 수 있도록 통합된 형식화(unified formalization)를 도입했습니다. 개인 이벤트는 순차적으로(sequence-based) 모델링되고, 관계 이벤트는 그래프 기반(graph-based) 접근 방식으로 모델링됩니다. 기존의 템포럴 그래프(temporal graph) 모델링 방식은 관계 이벤트에 중점을 두었으나, 개인 이벤트의 구조적 및 시간적 의존성을 충분히 반영하지 못하는 문제를 지적합니다.

- **Performance Highlights**: 모델에 개인 이벤트와 관계 이벤트를 모두 포함시킴으로써 여러 예측 작업에서 성능 향상을 실증적으로 보여주었습니다. 현재의 모델들은 이러한 이벤트 통합 모델링에 적합하지 않다는 점을 발견했으며, 이에 따라 개선이 필요하다고 제안합니다. 연구자들이 이러한 템포럴 그래프와 개인 이벤트를 아우르는 통합 모델링을 통해 사용자 행동을 더 잘 이해할 수 있는 기반을 제공합니다.



### MammoDINO: Anatomically Aware Self-Supervision for Mammographic Images (https://arxiv.org/abs/2510.11883)
Comments:
          5 pages

- **What's New**: MammoDINO는 유방촬영(Mammography)을 위한 새로운 SSL(Self-supervised Learning) 프레임워크로, 140만 개의 유방촬영 이미지로 사전 훈련되었습니다. 이는 일반적인 도메인에서 비전 인코더 훈련을 혁신적으로 변화시킨 SSL을 의료 이미징 분야에 적용하고자 하는 시도입니다.

- **Technical Details**: MammoDINO는 이미지 수준(image-level) 및 패치 수준(patch-level) 감독을 위한 유방 조직 인식 데이터 증강 샘플러(data augmentation sampler)를 도입했습니다. 또한, 3D 디지털 유방 단층 촬영(DBT) 구조를 2D 사전 훈련에 활용하는 교차 슬라이스 대조 학습(cross-slice contrastive learning) 목표를 설정하여 임상적으로 의미 있는 특징을 포착하고자 합니다.

- **Performance Highlights**: MammoDINO는 여러 유방암 검진 작업에서 최첨단(state-of-the-art) 성능을 달성했으며, 다섯 개의 기준 데이터셋(benchmark datasets)에서 잘 일반화되고 있습니다. 이 프레임워크는 주석이 필요 없는 스케일링 가능한 기반을 제공하여 유방촬영 진단에 도움을 주며 방사선사의 업무 부담을 줄이고 진단 효율성을 향상시킵니다.



### Countermind: A Multi-Layered Security Architecture for Large Language Models (https://arxiv.org/abs/2510.11837)
Comments:
          33 pages, 3 figures, 6 tables. Keywords: LLM security; defense-in-depth; prompt injection; activation steering; multimodal sandbox; threat modeling

- **What's New**: 본 논문은 Countermind라는 다층 보안 아키텍처를 제안하여 'form-first' 공격(예: prompt injection)을 효과적으로 방어하고자 합니다. 기존의 대응 방식이 아닌 사전 예방적 접근을 통해 모델이 신뢰할 수 있는 지시와 신뢰할 수 없는 데이터를 구분할 수 있는 구조를 갖추도록 설계되었습니다. 이 아키텍처는 입력 데이터를 구조적으로 검증하고 변환하기 위한 강력한 경계와 내부 거버넌스 메커니즘을 통합하여 LLM의 취약점을 보완하고자 합니다.

- **Technical Details**: Countermind는 네 가지 기반 영역으로 구성됩니다. 첫째, Semantic Boundary Logic(SBL)은 요청을 분석하고 인증하는 API 경계를 설정하여 공격 표면을 줄입니다. 둘째, Parameter-Space Restriction(PSR)은 내부 의미 클러스터에 대한 접근을 조절하여 위험한 행동을 완화합니다. 셋째, Secure, Self-Regulating Core는 정책을 인코딩하고 감사 로그를 유지하는 governance 시스템입니다. 마지막으로, Multimodal Input Sandbox는 비텍스트 데이터에 대한 위협을 해결하기 위한 모듈입니다.

- **Performance Highlights**: 본 논문에서는 Countermind 아키텍처의 효과를 평가하기 위한 계획을 제시하고, 이를 통해 'form-first' 공격에 대한 Attack Success Rate(ASR)를 줄이는 데 중점을 두고 있습니다. 아키텍처의 사전 예방적 접근 방식은 전통적인 방어 메커니즘에 비해 더 높은 보안성과 신뢰성을 기대하게 하며, 새로운 공격 방식에 대해 보다 강력한 방어를 제공합니다.



### Data or Language Supervision: What Makes CLIP Better than DINO? (https://arxiv.org/abs/2510.11835)
Comments:
          EMNLP 2025 Findings

- **What's New**: 이 논문에서는 CLIP와 DINO를 동일한 환경에서 사전 학습시켜 두 모델 간의 성능 차이가 언어 감독(language supervision) 또는 데이터셋 크기에서 비롯되는지를 조사합니다. 연구 결과, CLIP는 고차원 의미(high-level semantics)를 잘 포착하고, DINO는 저차원 시각적 특성(low-level features)에 더 민감하다는 것을 보여줍니다. 이러한 차이는 VLM(vision-language models)의 성능에도 영향을 미쳐, CLIP는 텍스트 밀집 작업에서 두드러진 우위를 보입니다.

- **Technical Details**: CLIP와 DINO는 동일한 아키텍처(ViT-B/16), 데이터셋(DataComp의 1천만 이미지 서브셋), 훈련 구성으로 20 에포크 동안 학습되었습니다. 결과적으로 두 모델은 유사한 ImageNet 정확도를 달성하였고(CLP: 65.8%, DINO: 66.4%), 이는 공정한 비교 기반을 제공하였습니다. 훈련 후, 두 모델의 성능은 이미지 분류 벤치마크에서 유사하게 평가되었지만, CLIP은 Stanford Cars 및 CUB와 같은 세부 분류 작업에서 DINO보다 나은 성과를 보였습니다.

- **Performance Highlights**: LLaVA 모델에 통합된 CLIP은 텍스트 중심의 작업에서 두드러진 성능 향상(+7.5%)을 기록하며, DINO보다 대체로 우수한 성능을 보였습니다. 두 모델은 일반적인 VQA(visual question answering) 작업에서는 유사한 성능을 보였지만, CLIP은 OCR 기반 벤치마크에서 더 뛰어난 성과를 나타내며, text-heavy 시각적 이해 작업에서의 언어 감독의 중요성을 강조했습니다.



### Combining Euclidean and Hyperbolic Representations for Node-level Anomaly Detection (https://arxiv.org/abs/2510.11827)
- **What's New**: 본 논문에서는 Janus라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 Euclidean 및 Hyperbolic Graph Neural Networks를 결합하여 노드 표현의 보완적인 측면을 포획합니다. Janus는 대조 학습 목표로 조정된 다중 Graph-Autoencoder 프레임워크를 통해 노드의 다양한 뷰를 연합하여 anormalies를 효과적으로 식별합니다. 이 연구는 기존의 기법들을 초월하여 다양한 기하학적 공간을 통한 탐지를 시도합니다.

- **Technical Details**: Janus는 원래의 특성과 랜덤 워크 및 차수에서 파생된 구조적 특성을 결합하여 각 노드를 두 개의 뷰로 설명합니다. Euclidean 및 Hyperbolic 공간에 이러한 뷰를 직접 임베딩하고, 대조 학습 목표와 함께하는 다중 Graph-Autoencoder는 노드 간의 유사성을 강조합니다. 이 방법은 노드의 차별적인 구조와 동작을 포착하여, 단일 기하학 모델이 놓치는 복잡한 패턴을 발견할 수 있게 돕습니다. 또한, 구조 간시된 기하학적 재구성이 이루어집니다.

- **Performance Highlights**: 네 개의 실제 데이터셋에 대한 실험 결과, Janus는 저차원 및 심층 기준선 모델을 지속적으로 능가하는 성능을 보였습니다. 이 연구는 다양한 기하학적 표현을 결합하는 것이 그래프에서 미세하고 복잡한 anomalies를 식별하는 데 강력하고 효과적인 접근 방식을 제공한다는 것을 실증적으로 보여줍니다. Janus는 해당 분야에서 topology 및 geometry에 대한 최신 연구를 기반으로 하여 성능을 극대화하고 있습니다.



### Empirical Study on Robustness and Resilience in Cooperative Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2510.11824)
Comments:
          44 pages, 16 figures, NeurIPS 2025

- **What's New**: 본 연구에서는 협력적인 다중 에이전트 강화 학습(MARL)에서의 협력, 강건성(robustness), 복원력(resilience)을 평가하기 위해 82,620회의 대규모 실험을 수행하였습니다. 이 연구는 실세계 환경에서의 불확실성을 고려하여 MARL 시스템의 신뢰성 구축에 필요한 강건성 및 복원력에 대한 깊은 이해를 제공합니다. 저자들은 하이퍼파라미터 조정의 중요성을 강조하며, 기존의 조정 방안이 강건성과 복원력에 미치는 부정적인 영향을 지적하고 있습니다.

- **Technical Details**: MARL의 강건성과 복원력을 다루기 위해, 저자들은 다양한 불확실성 유형과 알고리즘에 대한 실험을 수행하였습니다. 연구 결과는 협력을 최적화하는 것이 약한 불확실성 하에서는 강건성과 복원력을 개선하지만, 불확실성이 증가함에 따라 그 관계가 약화됨을 보여주었습니다. 또한, 하이퍼파라미터 조정이 효과적임을 나타내며, 특정 조합이 협력에서 평균 52.60%, 강건성에서 34.78%, 복원력에서 60.34%의 개선을 가져오는 것으로 나타났습니다.

- **Performance Highlights**: 실험 결과, 강건성 및 복원력은 에이전트의 행동 소음(action noise)과 관찰 소음(observation noise)에 따라 다르게 나타났습니다. 또한, 협력이 높아지더라도 특정 하이퍼파라미터가 성능을 저하시킬 수 있다는 사실이 강조되었습니다. 결론적으로, 저자들은 MARL 시스템에서 신뢰성을 높이기 위해 하이퍼파라미터 조정의 중요성을 재확인하며, 향후 연구 방향을 제시하고 있습니다.



### BlackIce: A Containerized Red Teaming Toolkit for AI Security Testing (https://arxiv.org/abs/2510.11823)
- **What's New**: 이 논문에서는 AI 모델을 실제 시스템에 통합하는 과정에서의 안전성과 보안 문제를 해결하기 위해, AI red teaming의 중요성을 강조하고 BlackIce라는 오픈 소스 툴킷을 소개합니다. BlackIce는 대형 언어 모델(Large Language Models, LLMs)과 전통적인 기계 학습 모델을 위한 red teaming 도구를 통합하여 사용자가 쉽게 설정하고 실행할 수 있는 환경을 제공합니다. 이 툴은 Docker 이미지로 제공되어, 각종 도구를 통합된 커맨드라인 인터페이스로 접근 가능하게 하여 사용자들이 편리하게 취약점 평가를 수행할 수 있도록 합니다.

- **Technical Details**: BlackIce의 구조는 포터블한 Docker 이미지를 기반으로 하며, 정적(static) 및 동적(dynamic) 도구로 구분되어 있습니다. 정적 도구는 간단한 커맨드라인 인터페이스를 통해 사용될 수 있으며, 동적 도구는 고급 Python 기반의 커스터마이징을 지원합니다. 사용자는 다양한 도구를 서로 다른 환경에서 실행하는 것을 피할 수 있으며, 패키지 의존성 문제를 global_requirements.txt 파일을 통해 관리할 수 있습니다.

- **Performance Highlights**: BlackIce는 MITRE ATLAS 및 Databricks AI Security Framework에 매핑하여 주요 AI 보안 리스크 카테고리 전반에 대한 포괄적인 커버리지를 제공하는 것으로 평가되었습니다. 이 도구는 여러 유명한 오픈 소스 도구를 포함하여, 사용자에게 다양한 테스트 기능을 제공하며, 각각의 도구는 특정 영역에서 강점을 가지고 있습니다. 또한 동적 도구를 활용하면, 보다 세밀한 커스터마이징이 가능해져 사용자가 특정 요구사항에 맞춘 평가를 수행할 수 있습니다.



### PHANTOM RECALL: When Familiar Puzzles Fool Smart Models (https://arxiv.org/abs/2510.11812)
Comments:
          22 Pages

- **What's New**: 이 논문은 LLM(대형 언어 모델)이 진정한 논리적 추론(Reasoning)보다 기억된 템플릿에 의존하는 경향이 있다는 증거를 제시합니다. PHANTOM RECALL이라는 벤치마크를 통해 25개의 잘 알려진 논리 퍼즐과 149개의 수정된 버전을 평가하여 모델들이 알려진 해결책을 재현하는 경향을 발견했습니다. 논문에서는 수정된 질문에 대한 모델의 반응을 체계적으로 분석하여 문제가 수정되었을 때의 추론 변화를 조사했습니다.

- **Technical Details**: 저자는 자동화된 논리 동등성 판단기(Conceptual-equivalence judge), 세부적 추론 오류 분류법(Taxonomy of reasoning error categories), 질문 기반 완화 프레임워크(Prompting-based mitigation framework) 등 세 가지 도구를 제안합니다. 이 연구에서는 각종 벤치마크의 오염 문제(Benchmark contamination)를 취급하며 LLM이 학습 중 메모이제이션을 통해 왜곡된 결과를 생성할 수 있음을 보여줍니다. 이로 인해 LLM들은 맥락이 바뀌면 반복 추론을 실패하는 경향을 보여줍니다.

- **Performance Highlights**: 모델들은 기본 퍼즐에서는 거의 완벽한 정확성을 보였지만, 수정된 버전에서는 성능이 급격히 떨어졌습니다. 특히 'phantom recall'과 과도한 세부 사항 추가(over-elaboration) 오류가 주를 이루었습니다. 또한 '사고/구조적' 프롬프트(prompting)는 성능을 개선하였지만, 견고성의 격차는 해소되지 않아 향후 훈련 시 개입의 필요성을 강조합니다.



### GAR: Generative Adversarial Reinforcement Learning for Formal Theorem Proving (https://arxiv.org/abs/2510.11769)
- **What's New**: 본 논문에서는 GAR(Generative Adversarial Reinforcement Learning)를 제안하여 문제 생성기와 해결기를 적대적 훈련 방식으로 동시에 학습하여, 기존의 제한된 문제 세트로 인한 비효율적인 훈련을 극복하고자 합니다. GAR는 증명자의 발전에 맞춰 과제를 동적으로 조정하는 암묵적 커리큘럼 학습 메커니즘을 도입하여 훈련 효율성을 개선하고, 복잡한 정리를 증명하는 성능을 향상시킵니다.

- **Technical Details**: GAR 프레임워크는 문제 생성 단계와 적대적 강화 학습(Adversarial Reinforcement Learning) 단계의 두 가지 과정으로 구성됩니다. 문제 생성 단계에서는 기존의 해결 가능한 문제 쌍에서 보다 어려운 문제를 합성하여, 증명자가 이를 해결하도록 훈련합니다. 이를 통해 고정된 데이터셋과 증명자의 발전하는 능력 간의 불일치를 완화합니다.

- **Performance Highlights**: GAR 훈련을 통해 Goedel-Prover-V2-8B와 DeepSeek-Prover-V2-7B가 MiniF2F-Test 벤치마크에서 평균 4.20% 향상을 기록했으며, DeepSeek-Prover-V2는 ProofNet-Test에서 22.58%에서 25.81%로 향상되었습니다. 이는 GAR의 효과성을 입증하며, 새로운 증명 생성과 해결을 위한 일반적인 RL 패러다임으로서의 가능성을 제시합니다.



### Audio-Guided Visual Perception for Audio-Visual Navigation (https://arxiv.org/abs/2510.11760)
Comments:
          Main paper (6 pages). Accepted for publication by International Conference on Virtual Reality and Visualization 2025 (ICVRV 2025)

- **What's New**: 이 논문에서는 AGVP(Audio-Guided Visual Perception) 프레임워크를 제안하여 기존의 오디오-비주얼 내비게이션 방법의 한계를 극복하고자 합니다. 기존 방법들은 시각적 특징과 청각적 신호 간의 명시적 정렬이 부족하여 성공률이 감소하는 문제가 있었으나, AGVP는 이를 개선하여 청각적인 정보를 공간적 지침으로 변환합니다. 이를 통해 새로운 소음원의 탐색 시에도 효율적으로 목표를 향해 나아갈 수 있는 길잡이를 제공하게 됩니다.

- **Technical Details**: AGVP 프레임워크는 관찰, 관찰 인코딩, 그리고 PPO(Proximal Policy Optimization)를 기반으로 한 정책 업데이트의 세 단계로 구성됩니다. 이 과정을 통해 에이전트는 시각적 및 청각적 입력을 지속적으로 탐색하며 필요한 정보를 인코딩합니다. 특히, self-attention 메커니즘을 사용하여 음향 신호의 글로벌 맥락을 구축하고 이를 비주얼 특징 지도에 통합하여 청각 신호와 관련된 시각적 영역을 명확히 정렬합니다.

- **Performance Highlights**: 실험 결과 AGVP는 내비게이션 효율성과 강인성에서 향상된 성능을 보였으며, 이전에 들리지 않았던 소음에 대해서도 뛰어난 시나리오 일반화 능력을 입증했습니다. 기존의 최고 수준의 기술과 비교하여 AGVP는 새로운 소리에 대한 탐색 성공률을 크게 향상시켰으며, 위험을 최소화하면서 목표를 향한 효율적 이동을 가능케 합니다.



### AwareCompiler: Agentic Context-Aware Compiler Optimization via a Synergistic Knowledge-Data Driven Framework (https://arxiv.org/abs/2510.11759)
- **What's New**: 이번 연구에서는 AwareCompiler라는 새로운 프레임워크를 소개하며, 이는 컴파일러 최적화를 위한 에이전트 기반 접근 방식을 채택하고 있습니다. 이 프레임워크는 프로그램 표현과 최적화 패스 간의 의미적 불일치, 비효율적인 상호 작용 메커니즘, 보상 희소성을 해결하는 세 가지 주요 혁신을 제공합니다. AwareCompiler는 구조화된 지식 통합, 지식 중심의 적응형 패스 생성, 데이터 기반 하이브리드 훈련 파이프라인 등을 포함한 혁신적 접근 방식을 특징으로 합니다.

- **Technical Details**: AwareCompiler는 컴파일러 최적화 문제를 여러 턴의 에이전트-환경 상호작용 문제로 모델링합니다. 각 패스는 그 의미, 의존성 및 충돌 정보를 포함하는 최적화 공간으로 인코딩되어 있으며, 이를 통해 올바른 패스 시퀀스를 생성합니다. 또한, 지식 기반의 접근 방식을 통해 코드 기능을 분석하고 최적의 패스 시퀀스를 효과적으로 생성할 수 있습니다.

- **Performance Highlights**: 실험 결과 AwareCompiler는 코드 크기 감소에서 기존 방법들을 크게 능가하는 성능을 보여주었습니다. AwareCompiler는 프로그램 맥락에 기반한 내부 추론 및 외부 지식을 통합하여 비효율적이거나 불가능한 최적화 패스를 줄이는 데 성공했습니다. 이러한 지식-데이터 주도 접근 방식의 효과는 차세대 LLM 기반 컴파일러 최적화 에이전트의 견고한 기반을 마련합니다.



### The Adoption Paradox: A Comparative Analysis of Veterinary AI Adoption in China and the North America (https://arxiv.org/abs/2510.11758)
Comments:
          1 Table, 5 Figures (included in the end), Full questionnaire used in this study (both original Chinese version and translated/English version included in the end)

- **What's New**: 이번 연구는 중국과 북미의 수의 전문인들 사이에서 인공지능(AI)의 인식, 도입 및 활용을 비교하였습니다. 이 연구는 지역적 시장 및 인구 통계적 요인이 AI 도입 방식에 영향을 미친다는 가설을 테스트합니다. 연구 결과는 주로 임상의들로 구성된 중국 집단과 북미의 수의 전문가들 간의 AI 활용 방식의 차이를 보여줍니다.

- **Technical Details**: 연구는 2025년 5월에서 7월 사이에 중국의 수의 전문가 455명을 대상으로 한 기술적 설문조사를 포함합니다. 이 결과는 2024년에 진행된 북미 지역의 3,968명의 수의 전문가에 대한 조사 결과와 비교되었습니다. 중국의 수의사들은 AI 도입률이 71.0%에 달했지만, 그에 대한 친숙도는 55.4%에 불과했으며, 임상적인 작업에 AI를 활용했습니다.

- **Performance Highlights**: 반면, 북미의 수의사들은 83.8%의 높은 친숙도에도 불구하고 AI 도입률은 39.2%로 낮았습니다. 해당 그룹은 이미징 분석 및 기록 유지와 같은 행정적 작업에 더 우선 순위를 두었습니다. 이 연구 결과는 '도입 패러독스(adoption paradox)'를 시사하며, 중국 시장은 임상적 효율성을 증대시키기 위한 하향식 구조와는 다르게 우선하여 AI를 도입하는 경향을 보였습니다.



### Artificial Intelligence for Optimal Learning: A Comparative Approach towards AI-Enhanced Learning Environments (https://arxiv.org/abs/2510.11755)
- **What's New**: 이번 연구는 전통적인 교육 방법, 비 AI 기술이 강화된 방법, AI 기반 기술을 이용한 방법 세 가지의 교육적 환경을 비교하여 각각이 교육 결과 및 참여도에 미치는 영향을 평가합니다. 이를 통해 각 모델의 장점을 합쳐 보다 포괄적인 교육 접근 방식을 개발하는 것을 목표로 합니다. AI의 도입은 특히 디지털 학습 도구의 채택이 급격한 팬데믹을 경험한 후 교육 수업의 형태를 혁신시키는 중요한 요소로 부각되고 있습니다.

- **Technical Details**: 기술이 교육에서 점점 더 기본적인 구성 요소로 자리잡으면서, 디지털 도구와 AI의 통합이 교육적 내용의 구성 및 전달 구조에 중대한 영향을 미치고 있습니다. 연구에서는 비 AI 기술과 AI 기술이 접목된 환경이 학생 참여도와 학습 결과에 따라 어떻게 다른지를 탐구하며, 이 과정에서 다양한 교육 세팅의 경험을 체계적으로 비교 분석합니다. 전통적인 교육 시스템의 한계, 특히 개인 맞춤형 학습 경험을 제공하는 데 있어 디지털 도구의 필요성을 강조하고 있습니다.

- **Performance Highlights**: 교육 모델의 비교를 통해 AI 기술이 교육 경로에 미치는 긍정적 영향을 분석하고, 각 제도가 가진 훈련적 및 사회적 이점을 종합하여 효과적인 교육 환경을 구축할 떄의 전략적 방향을 제시하고자 합니다. 연구 결과는 교육 정책 및 실행을 위한 실질적인 통찰력을 제공하며, 교육 기술이 어떻게 형평성 있게 모든 학습자에게 지원할 수 있는지에 대한 방안을 모색합니다.



### Zero-Shot Large Language Model Agents for Fully Automated Radiotherapy Treatment Planning (https://arxiv.org/abs/2510.11754)
Comments:
          Accepted for poster presentation at the NeurIPS 2025 Workshop on GenAI for Health: Potential, Trust, and Policy Compliance

- **What's New**: 이 연구에서는 대규모 언어 모델(LLM)을 기반으로 한 에이전트를 활용하여 강도 변조 방사선 치료(IMRT)의 역 치료 계획을 지원하는 새로운 워크플로우를 제안합니다. LLM 에이전트는 클리닉 치료 계획 시스템(TPS)과 직접 상호 작용하여 중간 계획 상태를 추출하고 최적화를 유도하기 위한 제약 조건 값을 제안합니다. 기존의 수동 계획을 대체할 수 있는 자동화된 솔루션의 필요성이 커지고 있는 상황에서, 이 접근법은 복잡한 방사선 치료 계획의 자동화를 가능하게 합니다.

- **Technical Details**: 제안된 방법은 상업적으로 사용되는 Eclipse™ TPS 환경에서 개발 및 검증되었습니다. LLM 기반 에이전트는 역 최적화 공간 내에서 치료 계획 매개변수를 조정하여 클리닉 목표를 충족시키고 계획 품질을 향상시키도록 설계되었습니다. 이 에이전트는 Eclipse Scripting Application Programming Interface(ESAPI)를 통해 TPS와 직접 연결되어 중간 계획 상태를 가져오고 제약 조건을 수정할 수 있습니다.

- **Performance Highlights**: 본 연구에서는 LLM으로 생성된 IMRT 치료 계획이 20개의 두경부 암 사례에 대해 임상적으로 생성된 계획과 비교되었습니다. LLM이 생성한 계획은 장기 위험(OAR) 절약에 있어 임상 계획에 비해 동등한 성과를 보였으며, 핫스팟 제어에서 개선된 결과(최대 피폭: 106.5% 대 108.8%)와 우수한 적합성을 나타냈습니다. 이는 LLM 기반의 자동화된 IMRT 치료 계획이 임상 환경에서도 유용할 수 있음을 보여줍니다.



### Fast and Interpretable Protein Substructure Alignment via Optimal Transpor (https://arxiv.org/abs/2510.11752)
- **What's New**: 이번 연구에서는 PLASMA라는 첫 번째 심층 학습 프레임워크를 제시하여 단백질 서브구조 정렬을 효율적이고 해석 가능한 방식으로 수행합니다. PLASMA는 단백질 구조를 입력받아 명확한 정렬 행렬과 해석 가능한 유사성 점수를 산출합니다. 이 방법은 단백질 구조 분석 도구의 중요한 공백을 메우며, 기능 주석화, 진화 연구 및 구조 기반 약물 설계의 새로운 기회를 제공합니다.

- **Technical Details**: PLASMA는 최적 운송(optimal transport, OT) 문제로 서브구조 정렬을 재구성하며, 차별화 가능한 Sinkhorn 반복을 활용합니다. 이 프레임워크는 단백질 표현 모델의 잔여물 임베딩을 기반으로 하여 두 단백질 쌍 간의 잔여물 간 정렬을 식별합니다. PLASMA는 세 가지 주요 단계를 통해 작동하며, 각 단계에서 매칭을 계산하고 최종 유사성 점수를 요약합니다.

- **Performance Highlights**: 포괄적인 양적 평가와 세 가지 생물학적 사례 연구를 통해 PLASMA는 정확하고 경량이며 해석 가능한 잔여물 수준의 정렬을 달성함을 보여주었습니다. PLASMA-PF라는 학습이 필요 없는 변형도 도입하여, 훈련 데이터가 없는 경우에도 사용할 수 있는 실용적인 대안을 제공합니다. 이러한 특징들은 단백질 기능 분석의 새로운 가능성을 열어주는 중요한 성과입니다.



### Celebrity Profiling on Short Urdu Text using Twitter Followers' Feed (https://arxiv.org/abs/2510.11739)
- **What's New**: 이 연구는 유명인의 인구통계를 예측하기 위해 현대의 기계 학습 및 딥 러닝 기법을 우르두어(Urdu) 트윗 데이터를 사용하여 최초로 적용한 것입니다. 기존 연구는 주로 영어와 같은 고자원 언어에 집중되어 있었으며, 우르두어는 상대적으로 연구가 부족한 언어로 남아 있었습니다. 연구진은 유명인을 팔로우하는 팬들의 짧은 우르두어 트윗을 수집하여 이를 통해 유명인의 나이, 성별, 직업 및 명성을 예측하는 새로운 접근 방법을 제시했습니다.

- **Technical Details**: 이 연구에서는 Logistic Regression, Support Vector Machines, Random Forests, Convolutional Neural Networks(CNN), Long Short-Term Memory networks(LSTM) 등 여러 알고리즘을 훈련하고 비교하여 우르두어로 된 유명인 프로파일링 모델을 개발하였습니다. 연구에서는 10명의 팔로워로부터 수집한 트윗 데이터를 전처리하고, 이를 기반으로 모델을 학습시켰습니다. 모델의 성능은 정확도(Accuracy), 정밀도(Precision), 재현율(Recall), F1-score와 누적 순위(cRank)로 평가되었으며 성별 예측에서 가장 높은 0.65의 cRank와 정확도를 기록했습니다.

- **Performance Highlights**: 연구 결과, 팔로워를 기반으로 한 언어적 특징을 활용하여 우르두어라는 저자원 언어에서도 효과적으로 인구 통계를 예측할 수 있음을 보여주었습니다. 성별 예측이 가장 우수한 성과를 보였고, 이어서 나이와 직업, 명성 예측에서도 보통 수준의 결과를 기록했습니다. 이러한 결과는 기계 학습과 신경망 접근 방식을 통해 저자원 언어에서도 의미있는 예측이 가능하다는 것을 시사합니다.



### SeeingSounds: Learning Audio-to-Visual Alignment via Tex (https://arxiv.org/abs/2510.11738)
Comments:
          accepted to ACM Multimedia Asia 2025

- **What's New**: SeeingSounds는 오디오와 이미지 생성 간의 상호 연계를 활용하는 경량 모듈형 프레임워크로, 짝을 이루는 오디오-비주얼 데이터나 비주얼 생성 모델을 학습할 필요 없이 동작합니다. 이 방법은 오디오를 텍스트 대체물로 다루기보다는, 오디오와 비주얼 도메인 간의 맥락 기반 정렬을 통해 두 가지 주요 경로를 활용하여 세 가지 입력 모달리티(음향, 언어, 비주얼)를 연결합니다.

- **Technical Details**: SeeingSounds의 아키텍처는 고정된 언어 인코더를 통해 오디오를 의미론적 언어 공간으로 투영하고, 비전-언어 모델을 사용하여 시각적 도메인에 맥락적으로 기초를 둡니다. 이 프레임워크는 정량적 및 정성적으로 매우 세밀한 조절이 가능하도록 하여 오디오 변환이 의미적으로 일관된 텍스트 프롬프트로 변환되어 시각적 결과를 안내할 수 있습니다.

- **Performance Highlights**: 다양한 벤치마크를 통해 SeeingSounds가 제로샷 및 감독 환경 모두에서 기존 방법보다 우수한 성능을 보여주며, 통제 가능한 오디오-비주얼 생성에 있어 새로운 최첨단 기술을 확립했습니다. 우리의 결과는 오디오에서 장면으로의 생성에서 상태-of-the-art 성능을 보여주며, 간단한 언어-비주얼-오디오 정렬 전략의 효과를 확인합니다.



### Scaling Law in LLM Simulated Personality: More Detailed and Realistic Persona Profile Is All You Need (https://arxiv.org/abs/2510.11734)
- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)을 활용하여 사회 실험을 시뮬레이션하는 방법을 탐구하며, 가상 인물 롤플레잉에서 인간의 성격을 모방하는 능력을 평가합니다. 연구는 개인 수준의 안정성 및 식별 가능성 분석과 인구 수준의 분석인 진전적 성격 곡선(progressive personality curves)을 포함하는 평가 프레임워크를 제공합니다. 이를 통해 LLMs가 인간 성격을 시뮬레이션하는 일관성과 진실성을 검토하고자 합니다.

- **Technical Details**: 연구는 인간 성격 측정 도구에 의존하며, 사회 시뮬레이션 실험에서 LLMs가 인간 행동을 모방하는 방법론적 일치를 요구합니다. 기존의 심리 측정 접근 방식인 CFA(확인적 요인 분석)와 구성 타당성(construct validity)은 LLMs의 능력을 잘 포착하지 못한다고 지적하며, 이로 인해 잘못된 결론에 도달할 위험이 있습니다. 대안으로 연구는 인구 통계적 데이터를 기반으로 하는 가상 인물 프로필 생성을 통해 LLM의 성격 시뮬레이션 능력을 평가하는 통합적 연구 프레임워크를 제공합니다.

- **Performance Highlights**: 연구의 주요 기여는 LLM의 가상 성격 평가를 위한 체계적인 프레임워크를 제안하고, 성격 시뮬레이션의 질에 있어 페르소나 세부사항의 중요성을 실증적으로 증명하는 것입니다. LLM 성격 시뮬레이션의 진척 경향을 감지할 수 있는 평가 방법을 제공하며, 사회 과학 실험에서 대규모 언어 모델의 적용을 위한 이론적 기초와 운영 평가 지표를 제시합니다.



### Serial-Parallel Dual-Path Architecture for Speaking Style Recognition (https://arxiv.org/abs/2510.11732)
Comments:
          Accepted by NCMMSC2025

- **What's New**: 본 논문에서 제안하는 새로운 serial-parallel dual-path architecture는 Speaking Style Recognition (SSR)에서 음향-언어(Bimodal) 정보를 활용하여 성능을 향상시키는 것을 목표로 합니다. 기존 연구는 주로 언어적 정보에 의존했으나, 제안하는 방법은 음향 정보와 언어 정보를 통합하여 인식 정확성을 획기적으로 개선할 수 있는 잠재력을 보여줍니다. 이를 통해 SSR의 인식 정확도가 기존 OSUM 모델보다 30.3% 향상되며, 파라미터 크기도 88.4% 줄어듭니다.

- **Technical Details**: 제안하는 모델은 serial path와 parallel path를 기반으로 한 구조로, serial path는 ASR+STYLE serial paradigm을 따릅니다. 이를 통해 LLM(대형 언어 모델)은 ASR 전사 결과를 생성하고 이를 음향 특성과 결합하여 스타일 레이블을 예측합니다. 반면, parallel path는 Acoustic-Linguistic Similarity Module (ALSM)을 사용하여 음향 및 언어적 특성을 동시 처리하며 시간적 동기화를 강조합니다. 이러한 접근 방식은 텍스트와 음성을 동시에 고려하여 스타일 인식의 정확성을 높입니다.

- **Performance Highlights**: 실험 결과에서 제안하는 모델은 기존 SSR 기준선인 OSUM 모델과 비교하여 눈에 띄는 성과를 보였습니다. SSR 정확도가 30.3% 증가한 반면, 파라미터 수는 88.4% 감소하여 효율성을 극대화했습니다. 이는 언어적 정보와 음향적 정보를 결합하는 새로운 아키텍처의 효과성을 입증하며, 차세대 SSR 기술의 가능성을 보여줍니다.



### Modeling Hypergraph Using Large Language Models (https://arxiv.org/abs/2510.11728)
Comments:
          10 pages, 5 figures

- **What's New**: 본 논문은 고차원 관계를 모델링하는 데 유용한 하이퍼그래프(hypergraph)의 데이터를 신속하게 생성할 수 있는 새로운 방법인 HyperLLM을 소개합니다. HyperLLM은 최근의 대형 언어 모델(LLM) 기술을 활용하여, 다중 에이전트 협업을 통해 하이퍼그래프의 형성과 진화를 시뮬레이션하는 프레임워크입니다. 이를 통해 기존의 통계적 모형과 다르게 실제 네트워크의 특성을 반영한 대규모 하이퍼그래프를 생성할 수 있는 가능성을 보여줍니다.

- **Technical Details**: HyperLLM은 구조화된 프롬프트(prompts)와 피드백 메커니즘을 통합하여 생성하는 하이퍼그래프가 현실 세계의 패턴을 잘 반영하도록 합니다. 기존 하이퍼그래프 생성 모델들이 지닌 한계들을 극복하기 위해, 이 연구는 LLM의 의미적 이해와 추론 능력을 활용하여 더욱 현실적이고 의미 있는 관계를 생성하는 방법을 제안합니다. 실험 결과는 HyperLLM이 구조적 및 시간적 하이퍼그래프 패턴에 대해 높은 신뢰성을 달성하며 통계적 사전 정보의 필요성을 최소화함을 보여줍니다.

- **Performance Highlights**: 본 연구에서 제안하는 HyperLLM은 현실성(realism), 효율성(efficiency), 휴대성(portability) 측면에서 뛰어난 성능을 보여주며, 이는 엄청난 양의 데이터가 부족한 현재의 하이퍼그래프 연구에 중요한 기여를 할 수 있습니다. 실험을 통해 HyperLLM은 예외적인 성능을 발휘하며, 이로써 LLM 기반의 하이퍼그래프 모델링이 새로운 방향으로 나아갈 수 있는 가능성을 제시하고 있습니다.



### Dual Perspectives on Non-Contrastive Self-Supervised Learning (https://arxiv.org/abs/2507.01028)
- **What's New**: 본 논문은 비대비적(self-supervised) 학습 접근법에서 자주 사용하는 정지 경량(stop gradient) 및 지수 이동 평균(exponential moving average) 절차의 이론적 분석을 제공합니다. 이러한 절차들이 표현의 붕괴(collapse)를 방지함과 동시에, 원래 목표 함수를 최적화하지 않음에도 불구하고, 우수한 성능을 유지하는 이유를 파악하였습니다.

- **Technical Details**: 연구에서는 이러한 절차들이 최적화 및 동역학 시스템(dynamical systems)의 관점에서 어떻게 작동하는지를 분석합니다. 특히, 선형(linear) 경우에서의 원래의 목표 함수를 정지 경량이나 지수 이동 평균 없이 최소화하면 붕괴에 이르게 됨을 보여주며, 두 절차의 동역학 시스템과 관련된 한계점(limit points)이 일반적으로 안정적인 평형(asymptotically stable equilibria)로 이어진다는 것을 규명하였습니다.

- **Performance Highlights**: 이 연구는 후속 감독(supervised) 애플리케이션에서의 탁월한 성능을 뒷받침하기 위해 종합적인 이론적 근거를 제공합니다. 이를 통해 비대비적 학습 접근이 혁신적으로 개선될 가능성을 제시하며, 향후 연구와 구현에 큰 기여를 할 것으로 기대됩니다.



### Leveraging LLMs, IDEs, and Semantic Embeddings for Automated Move Method Refactoring (https://arxiv.org/abs/2503.20934)
Comments:
          12 pages, 2 figures

- **What's New**: 본 논문에서는 전 세계 최초의 MoveMethod 리팩토링을 위한 LLM(대형 언어 모델) 기반 보조 도구인 MM-assist를 소개합니다. 기존의 리팩토링 도구들과 달리, MM-assist는 제안부터 실행까지 전체 리팩토링 생명주기를 자동화합니다. 이를 통해 개발자들은 어려운 정적 분석이나 복잡한 선행 조건 확인 없이 최적의 리팩토링을 받을 수 있게 되었습니다. 또한 LLM의 환각을 자동으로 필터링 하고, 전역적인 프로젝트 수준의 추론을 가능하게 하는 혁신적인 방법을 제시합니다.

- **Technical Details**: MM-assist는 LLM, IDE(통합 개발 환경), 정적 분석 및 의미론적 관련성을 결합하여 리팩토링을 지원합니다. 이 시스템은 먼저 코드에서 위치가 잘못된 메서드를 식별하고, 적절한 대상 클래스 제안을 생성하는 과정을 포함합니다. LLM의 한계인 짧은 맥락 크기를 해결하기 위해, MM-assist는 검색 증 강화 생성(RAG) 기법을 활용하여 의미론적 관련성을 고려한 데이터를 LLM 입력으로 제공합니다. 또한, IDE의 정적 분석을 통해 생성된 제안의 유효성을 검증합니다.

- **Performance Highlights**: MM-assist는 기존의 최첨단 리팩토링 도구들과 비교했을 때도 현저한 개선을 보였습니다. 210개의 실제 리팩토링에 대한 회수율(Recall)에서 MM-assist는 80%에 도달하며, 이는 이전 도구 HMove의 33%에 비해 2.4배 향상된 수치입니다. 사용자 연구 결과, 30명의 경험이 있는 참여자들 중 82.8%가 MM-assist의 추천 결과를 긍정적으로 평가하였으며, 전통적인 IDE 워크플로우보다 이 LLM 기반 접근 방식을 선호한다고 응답했습니다.



### Operand Quant: A Single-Agent Architecture for Autonomous Machine Learning Engineering (https://arxiv.org/abs/2510.11694)
Comments:
          8 pages. No figures. Evaluated on MLE-Benchmark 2025

- **What's New**: Operand Quant는 자율 기계 학습 엔지니어링(MLE)을 위한 단일 에이전트, IDE 기반 아키텍처를 제안합니다. 기존의 다중 에이전트 오케스트레이션 프레임워크와는 달리, Operand Quant는 탐색, 모델링, 실험 및 배포의 모든 MLE 생애주기 단계를 단일 맥락 인식 에이전트 내에서 통합하여 처리합니다. MLE-Benchmark에서 기록한 0.3956의 메달 비율은 현재까지 평가된 모든 시스템 중 최고의 성능을 보여주며, 이는 다중 에이전트 구조와의 차별성을 잘 나타냅니다.

- **Technical Details**: Operand Quant는 자율적인 단일 에이전트 시스템으로, 시뮬레이션된 IDE 내에서 작동합니다. 이 에이전트는 탐색적 데이터 분석, 특성 엔지니어링, 모델링, 평가 및 제작화 등 MLE 생애주기의 모든 단계를 독립적으로 수행할 수 있습니다. 각 결정 주기 동안 에이전트는 현재 IDE 상태를 관찰하고, 행동을 결정하며, 이를 JSON 명령으로 실행합니다. 이러한 비차단 루프 구조는 효율적인 병렬 추론과 지속적인 반복을 가능하게 합니다.

- **Performance Highlights**: Operand Quant는 MLE-Benchmark 2025에서 새로운 최첨단(SOTA) 결과를 달성하였으며, 75개의 문제에서 메달 비율이 0.3956 +/- 0.0565로 기록되었습니다. 이는 기존의 다중 에이전트 시스템보다 우수한 성능을 입증하며, 단일 에이전트 아키텍처가 복잡한 기계 학습 작업에서도 높은 성과를 낼 수 있음을 보여줍니다. 이러한 성과는 에이전트가 자율적으로 작업을 수행하면서 발생하는 다중 에이전트 간의 조정 비용을 없앴기 때문입니다.



### SR-Scientist: Scientific Equation Discovery With Agentic AI (https://arxiv.org/abs/2510.11661)
- **What's New**: 최근 연구에서 대형 언어 모델(LLMs)을 활용하여 과학적 방정식 발견을 위한 프레임워크인 SR-Scientist를 소개합니다. 이 프레임워크는 LLM을 단순한 방정식 제안자로 사용하기보다는 자율적인 AI 과학자로 전환하여 데이터를 분석하고 방정식을 구현하는 코드를 작성하며 실험 피드백에 따라 방정식을 최적화합니다. 이를 통해 장기적인 최적화 과정을 통해 LLM의 자율성을 확보하고, 방정식 디자인에 대한 통찰력을 얻을 수 있는 도구로 기능하게 합니다.

- **Technical Details**: SR-Scientist는 LLM을 위한 데이터 분석 및 방정식 평가 도구를 포함하며, LLM 에이전트가 문제를 해결하기 위해 코드 작성, 방정식 구현 및 실험 피드백에 따라 방정식을 최적화하는 역할을 수행하도록 설계되었습니다. 이 과정에서 경험 버퍼(experience buffer)를 도입하여 LLM이 탐색한 방정식을 저장하고, 최상의 방정식을 후속 반복에 활용할 수 있도록 합니다. 또한 최소 인간 정의 파이프라인(minimal human-defined pipelines)의 원칙을 따르며, 에이전트가 자신의 작업 흐름을 자유롭게 결정할 수 있도록 지원합니다.

- **Performance Highlights**: 실험 결과 SR-Scientist는 4가지 과학 분야에 걸친 데이터셋에서 기존의 최첨단 기법에 비해 6%에서 35%까지의 성능 향상을 보였습니다. 또한, 소음(noise)에 대한 강건성, 도메인 외(out-of-domain) 데이터에 대한 발견된 방정식의 일반화 및 기호 정확(symbolic accuracy) 검증이 이루어졌습니다. 끝으로, 강화학습(reinforcement learning) 프레임워크를 통해 에이전트의 능력을 더욱 향상시켜, 채택된 분석 도구의 중요성을 강조합니다.



### ParaCook: On Time-Efficient Planning for Multi-Agent Systems (https://arxiv.org/abs/2510.11608)
- **What's New**: 이 논문에서는 ParaCook라는 새로운 벤치마크를 제안하여 LLMs의 시간 효율적인 협업 계획 능력 평가에 중점을 둡니다. ParaCook는 Overcooked 게임에서 영감을 받아 여러 에이전트가 요리 작업을 수행하는 복잡한 상호작용 계획 환경을 제공합니다. 기존 벤치마크들이 주로 작업 완수에만 초점을 맞춘 반면, ParaCook는 시간 효율성을 고려한 계획 능력을 평가합니다.

- **Technical Details**: 이 논문은 단일 및 다중 에이전트 시스템을 위한 병렬 계획을 정의하며, 복잡한 작업을 Directed Acyclic Graph (DAG) 형태로 서브태스크로 분해합니다. 에이전트 시스템 내에서 각 서브태스크는 작업자(에이전트)를 나타내는 추가 속성을 지니며, 이를 통해 각 서브태스크의 실행을 다음 에이전트에 할당합니다. 이를 통해 LLMs가 내부 및 외부 병렬성을 최대한 활용하여 전체 작업 완료 시간을 최소화하는 데 집중합니다.

- **Performance Highlights**: 전반적인 실험 결과, 최신 모델인 GPT-5는 복잡한 작업에서 65%의 평균 성공률을 보인 반면, 인간은 완벽한 성공률을 달성했습니다. 또한 모델의 완료 시간은 인간 기준보다 상당히 길고 이동 비용도 더 높았습니다. 그럼에도 불구하고 추상적 계획 작업에서는 최신 LLM들이 최적 성능에 근접한 성과를 달성하여 고급 추론 능력을 보여줍니다.



### Explainability, risk modeling, and segmentation based customer churn analytics for personalized retention in e-commerc (https://arxiv.org/abs/2510.11604)
- **What's New**: 이 연구에서는 고객 이탈 예측을 위한 새로운 통합 시스템을 제안합니다. 기존의 고객 이탈 모델들이 종종 불투명한 블랙 박스 형태였던 반면, 연구진은 설명 가능한 AI(explainable AI)를 통해 개인 맞춤형 유지 전략을 설계하고 있습니다. 이 연구의 주요 기여는 예측 정확성, 모델 해석 가능성, 시간 모델링, 세분화 및 실시간 배포를 통합한 것입니다.

- **Technical Details**: 연구에서는 세 가지 구성 요소 프레임워크를 통해 설명 가능한 AI(explainable AI), 생존 분석(survival analysis), 그리고 RFM(Recency, Frequency, Monetary) 프로파일링을 통합하고 있습니다. 이는 이탈 원인을 정량화하고, 개입 기회를 추정하며, 타겟 세그먼트를 우선순위 삼아 고객 유지 전략을 수립하는 데 기여합니다. 또한, 열린 데이터셋을 사용하여 머신러닝 모델을 벤치마킹하고 있으며, 세 가지 단계의 개인화된 유지 전략 개발 절차를 따릅니다.

- **Performance Highlights**: 연구진은 분석 결과를 통해, 고객 이탈을 줄이고 충성도를 높이는 전략을 제시할 수 있음을 보여주었습니다. 방법론의 조합은 고객 세분화와 이탈 위험 예측의 강점을 결합하여 유의미한 통찰력을 제공합니다. 반응 속도를 고려한 모델 구성 덕분에, 실시간으로 고객 행동을 분석하는 데 적합합니다.



### Reproducibility: The New Frontier in AI Governanc (https://arxiv.org/abs/2510.11595)
Comments:
          12 pages,6 figures,Workshop on Technical AI Governance at ICML

- **What's New**: 이 논문에서는 AI 정책입안자들이 안전하고 신뢰할 수 있는 AI 개발을 위한 효과적인 통치 메커니즘을 제공할 책임이 있음을 강조합니다. 그러나 현재 정보 환경은 낮은 신호 대 잡음 비율로 특징지어져, 규제 캡쳐(regulatory capture)와 깊은 불확실성을 초래하고 있습니다. AI 연구는 엄격한 재현성(reproducibility) 가이드라인을 도입하여 정책 입안자들에게 기초자료를 제공하고 AI 리스크에 대한 합의를 개선해야 한다고 제안합니다.

- **Technical Details**: 이 논문은 AI 연구의 출판 속도와 과학적 기준 결여가 정책입안자들의 정책 시행 능력을 약화시키고 있다고 주장합니다. 재현성 위기(reproducibility crisis)를 해결하기 위해, 예비등록(preregistration), 통계적 강도(statistical power) 증가, 실패한 결과의 발표 등 다양한 재현성 프로토콜이 필요하다고 설명합니다. 다른 과학 분야의 사례를 통해 재현성 기준을 강화하는 방안을 제시하여 AI거버넌스(governance)의 유효성을 높일 수 있다고 강조합니다.

- **Performance Highlights**: 그동안 경제학, 암 생물학 등 여러 분야에서 발생한 재현성 위기의 사례를 통해, 재현성 결여가 정책과 과학적 진보에 미치는 영향을 논의합니다. 특히, "Debt Growth" 논문의 실패한 복제 사례는 잘못된 연구 결과가 어떻게 심각한 사회적 결과를 초래할 수 있는지를 보여줍니다. 최종적으로 논문은 AI 연구에서 재현성을 높이기 위해서는 철저한 연구 설계와 데이터 공개의 중요성을 강조합니다.



### Analyzing and Internalizing Complex Policy Documents for LLM Agents (https://arxiv.org/abs/2510.11588)
Comments:
          42 pages

- **What's New**: 이 논문은 대형 언어 모델(LLM)을 기반으로 하는 에이전트 시스템에서 정책 문서의 내부화를 통해 성능을 유지하면서 문서의 복잡한 조건을 처리하기 위한 새로운 방법인 CC-Gen을 소개합니다. CC-Gen은 정책 문서의 복잡성을 체계적으로 평가할 수 있는 벤치마크 생성기를 제공하며, 이는 다양한 요인에 따라 에이전트의 수행 능력을 분석하는 데 기여합니다. 또한, 문서 내부화의 어려운 점을 이해하고, 이를 해결하기 위한 Category-Aware Policy Continued Pretraining(CAP-CPT) 방법론을 제안합니다.

- **Technical Details**: CC-Gen은 환경, 작업 수준, 워크플로우, 사용자 쿼리의 네 가지 복잡성 차원을 설정하여, 정책 문서가 에이전트 성능에 미치는 영향을 독립적으로 조작할 수 있게 합니다. 정책 사양을 세 가지 유형으로 분류하여, 각 유형에 맞는 맞춤형 데이터를 생성하는 자동화된 파이프라인을 구축했습니다. 이 과정에서 조건부 사양은 단순 및 복잡한 사례로 세분화되어 각기 다른 학습 도전 과제를 제공합니다.

- **Performance Highlights**: CAP-CPT는 다양한 시나리오에서 10% 이상의 성능 향상을 이루었으며, 이는 데이터가 희소한 환경에서도 두드러진 효과를 보였습니다. 실험 결과, 정책 문서에 대한 내부화 메커니즘이 에이전트의 일반화 능력을 높이고, 다양한 작업에서 성능 저하를 최소화하는 데 도움이 됨을 확인했습니다. 이 연구는 효율적인 정책 내부화 접근법이 에이전트의 안정성과 성능을 향상시킬 수 있으며, 전체 입력 토큰을 97.3% 압축하는 성과를 이루었습니다.



### Zero Data Retention in LLM-based Enterprise AI Assistants: A Comparative Study of Market Leading Agentic AI Products (https://arxiv.org/abs/2510.11558)
- **What's New**: 이번 논문에서는 의료 및 금융 산업을 포함한 데이터 관리와 비즈니스 프라이버시 문제에 대해 다루고 있습니다. 최근 AI 기업의 AI 비서들이 비즈니스 생산성을 높이고 있는 가운데, 개인 데이터 및 규정 준수를 보호하는 것이 핵심 우선사항이 되었습니다. 특히, 대규모 언어 모델(Large Language Models) 기업들이 제로 데이터 보존 정책을 통해 이러한 문제를 해결할 수 있음을 제시합니다.

- **Technical Details**: 연구에서는 Salesforce와 Microsoft와 같은 업계 리더와 함께 상업용 AI 비서의 개발을 살펴봅니다. 이들 기업은 제로 데이터 보존 정책을 지원하기 위해 서로 다른 기술 아키텍처를 사용하였으며, Salesforce의 AgentForce와 Microsoft의 Copilot을 예로 들 수 있습니다. 이러한 시스템의 아키텍처, 규정 준수(compliance), 사용성(usability) 간의 trade-offs(트레이드오프)를 정의한 것이 이 논문의 주요 기여입니다.

- **Performance Highlights**: Salesforce AgentForce와 Microsoft Copilot은 고객 관리 분야에서 비즈니스 생산성을 높이기 위해 필수적인 기능을 제공하는 선도적인 AI 비서입니다. 이 논문은 OpenAI, Anthropic, Meta와 같은 대형 언어 모델 서비스 제공자들이 제로 데이터 보존 정책을 얼마나 효과적으로 배치하고 있는지를 분석합니다. 이러한 분석은 비즈니스 애플리케이션의 소비 및 해당 정책을 구현하기 위한 기술적 아키텍처를 이해하는 데 기여합니다.



### Unifying Deductive and Abductive Reasoning in Knowledge Graphs with Masked Diffusion Mod (https://arxiv.org/abs/2510.11462)
Comments:
          Under Review

- **What's New**: DARK라는 새로운 통합 프레임워크가 제안되었습니다. 이 프레임워크는 Deductive와 Abductive Reasoning을 하나의 모델 내에서 결합하여 시너지 효과를 낼 수 있도록 합니다. 기존의 연구에서는 이 두 가지 패러다임을 분리하여 연구했으나, DARK는 이들의 상호작용을 통해 역동적인 이론 분석을 가능하게 합니다.

- **Technical Details**: DARK는 masked diffusion 모델로, 쿼리와 결론 간의 양방향 관계를 모델링합니다. 이 모델은 첫째로, abductive reasoning 과정에서 가설을 검증하는 데 deductive reasoning을 효과적으로 활용하기 위해 자기 반영적인 denoising 과정을 도입합니다. 둘째로, logic-exploration reinforcement learning 접근법을 통해 쿼리와 결론을 동시에 마스킹하여 새로운 추론 조합을 탐색합니다.

- **Performance Highlights**: DARK는 여러 벤치마크 지식 그래프에서 최신 성능을 기록했습니다. 이를 통해 Deductive와 Abductive Reasoning 작업 모두에서 뛰어난 성능을 발휘하여, 통합 접근 방식의 효율성과 효과성을 입증했습니다. 연구 결과는 DARK가 지식 그래프 추론의 다음 세대를 이끌 가능성이 있음을 보여줍니다.



### From <Answer> to <Think>: Multidimensional Supervision of Reasoning Process for LLM Optimization (https://arxiv.org/abs/2510.11457)
- **What's New**: 본 논문은 대형 언어 모델(LLMs)의 다단계 추론 능력을 개선하기 위한 새로운 접근 방식인 차원 수준 보상 모델(Dimension-level Reward Model, DRM)을 제안합니다. 기존의 RLVR(Outcome-supervised Reinforcement Learning) 방법은 최종 답변의 정확도만을 평가하여 잘못된 추론을 강화할 위험이 있습니다. DRM은 세 가지 차원(신뢰성, 관련성, 일관성)을 평가하여 추론 과정의 품질을 보다 정교하게 측정하며, 결론적으로 모델의 일반화 능력을 향상시킬 수 있는 방법을 제시합니다.

- **Technical Details**: DRM은 각 추론 과정의 품질을 신뢰성(Confidence), 관련성(Relevance), 일관성(Coherence)이라는 세 가지 차원으로 평가합니다. 신뢰성은 모델 출력의 확실성을 평가하며, 관련성은 질문, 부가 정보 및 최종 답변 간의 관계를 평가합니다. 마지막으로, 일관성은 논리적 일관성을 바탕으로 추론의 질을 평가합니다. 이러한 방식으로 DRM은 고유한 보상 신호를 제공하고, 저차원 보상 문제를 해결하며 해석 가능성을 높입니다.

- **Performance Highlights**: 실험 결과, DRM으로 학습한 모델은 다양한 오픈 도메인 태스크에서 일관된 성과 향상을 보여주었습니다. 예를 들어, Llama-3.1-8B-Instruct 모델은 Math500, 2Wiki_RAG 및 Cruxeval 등에서 우수한 성과를 기록하였고, 이는 기존의 답변 기반 감독 방식보다 더 나은 일반화 능력을 나타냅니다. 이 연구는 차원적 추론 감독이 LLM의 추론 능력과 성능을 향상시킬 수 있다는 중요한 발견을 제공합니다.



### AI-Driven anemia diagnosis: A review of advanced models and techniques (https://arxiv.org/abs/2510.11380)
- **What's New**: 이번 논문은 빈혈(anemia) 진단을 위한 최신 인공지능(AI) 기법, 특히 머신러닝(machine learning)과 딥러닝(deep learning)을 활용한 연구 동향을 체계적으로 검토합니다. 빈혈은 전 세계 수백만 명에게 영향을 미치는 보편적인 건강 문제로, 정확하고 신속한 진단이 필요합니다.

- **Technical Details**: 본 리뷰는 빈혈 탐지에 적용된 다양한 모델을 살펴보며, 이들 모델을 정확도(accuracy), 민감도(sensitivity), 특이도(specificity), 정밀도(precision)와 같은 여러 성능 지표(metrics)를 기반으로 비교합니다. 이러한 방법론은 빈혈의 탐지 및 분류에서 모델의 강점과 한계를 평가하기 위해 중요한 요소입니다.

- **Performance Highlights**: 논문은 다양한 AI 모델의 성능 지표를 분석하여 실질적인 진단 정확도를 향상시키기 위해 해결해야 할 과제를 강조합니다. 빈혈 진단에 있어 이러한 기술적 접근 방식은 임상적 의사결정(clinical decision-making)에 기여할 수 있는 잠재력을 지니고 있습니다.



### Automated Skill Decomposition Meets Expert Ontologies: Bridging the Granularity Gap with LLMs (https://arxiv.org/abs/2510.11313)
- **What's New**: 이번 논문은 대형 언어 모델(LLMs)을 활용한 자동 기술 분해(automated skill decomposition)를 조사하고, 엄밀한 온톨로지 기반 평가 프레임워크(ontology-grounded evaluation framework)를 제안합니다. 이 프레임워크는 프롬프트(prompt) 작성부터 생성, 정규화(normalization), 온톨로지 노드와의 정렬을 표준화하여, 기술 분해의 정확성과 구조적 건강성을 보장하는 방법을 제시합니다. 또한, 의미론적 F1-score와 계층 구조-aware F1-score라는 두 가지 메트릭을 도입하여 출력 결과를 평가할 수 있도록 합니다.

- **Technical Details**: 논문에서는 고수준 기술 레이블과 적응 학습 경로에 필요한 세분화 간의 불일치를 기술적 간극(granularity gap)으로 정의하고 이를 해결하기 위해 두 가지 프롬프트 방식인 제로 샷(zero-shot)과 주의가 필요한 소수 샷(few-shot)을 고려합니다. 실험에서는 특정 부모 기술을 사용하여 각 기술을 5~12개의 세부 기술로 효과적으로 분해(Decomposition)하는 방법을 다룹니다. 이 방법론은 프롬프트 구성, 후보 생성, 정규화 및 온톨로지 정렬을 포함한 종합적인 파이프라인을 통해 기술의 세부 사항을 도출하는 데 중점을 둡니다.

- **Performance Highlights**: 제안된 프레임워크는 다양한 LLM에서 제로 샷 기반이 강력한 기준을 제공하지만, 소수 샷 방식이 구문(phrasing) 및 세분화(granularity)의 안정성을 높이고 계층 구조-aware 정렬을 개선함을 확인했습니다. 지연(latency) 분석 결과, 표본 가이드 프롬프트가 비지도 제로 샷보다 경쟁력을 가지며, 때로는 더 빠른 성능을 보여주었습니다. 전반적으로 온톨로지 기반 구조가 더 신뢰할 수 있는 기술 분해를 제공하는 유용한 선행 지식으로 작용함을 보여주었습니다.



### Evolution in Simulation: AI-Agent School with Dual Memory for High-Fidelity Educational Dynamics (https://arxiv.org/abs/2510.11290)
Comments:
          9 pages, 7 figures, EMNLP conference

- **What's New**: AI-Agent School (AAS) 시스템은 복잡한 교육 동적을 시뮬레이션하기 위한 자가 진화 메커니즘을 기반으로 구축되었습니다. 이 시스템은 교육 과정 모델링의 분산 문제와 교육 참여자의 다양한 행동 시뮬레이션의 한계를 해결하기 위한 것입니다. AAS는 Zero-Exp 전략을 통해 경험과 지식을 포함한 이중 메모리 구조를 활용하며, 경험-반성-최적화 사이클을 지속적으로 수행하여 교육 환경 내에서의 다양한 상호작용을 통해 에이전트를 자율적으로 진화시킵니다.

- **Technical Details**: AAS에서는 멀티 역할 에이전트를 사용하여 경험과 지식의 이중 메모리 시스템을 구현하고 있습니다. 이 시스템은 작업 메모리 및 단기/장기 메모리로 구성되어 있어 에이전트의 인지 과정을 모방합니다. 이를 통해 에이전트는 설정된 행동 및 상호작용 데이터를 통해 메모리 기반을 지속적으로 업데이트하며 자율적으로 발전할 수 있도록 합니다. 실험 결과는 AAS가 복잡한 교육 동적을 효과적으로 시뮬레이션할 수 있음을 확인했습니다.

- **Performance Highlights**: AAS는 교사-학생 관계, 동료 상호작용 및 환경적인 영향을 포착하여 실제 교육 과정을 시뮬레이션할 수 있는 다중 에이전트 시스템입니다. Zero-Exp 메커니즘을 통해 에이전트는 제로 경험 상태에서 전문가 수준의 행동으로 발전할 수 있습니다. 또한 본 연구는 전통적인 교육 연구와 AI 기술을 결합하여 교육 시스템, 교사 훈련 플랫폼, 교육 정책 시뮬레이션 도구의 개발을 위한 이론적 및 기술적 기반을 제공합니다.



### PADME: Procedure Aware DynaMic Execution (https://arxiv.org/abs/2510.11281)
- **What's New**: 이 논문은 Procedure Aware DynaMic Execution (PADME)라는 프레임워크를 소개합니다. PADME는 자연어로 작성된 절차를 자동으로 실행 가능한 그래프로 변환하여 복잡한 장기 절차를 효과적으로 수행할 수 있도록 돕습니다. 이전의 연구들은 주로 수동적으로 그래프를 구축하거나 비구조화된 추론을 사용했지만, PADME는 이러한 과정을 자동화하여 에이전트의 실행 가능성을 높입니다.

- **Technical Details**: PADME의 방법론은 두 가지 단계로 나뉩니다: Teach 단계와 Execute 단계입니다. Teach 단계에서는 절차를 구조화하여 실행 가능한 결정 그래프로 변환하며, Execute 단계에서는 실시간 입력 및 환경 피드백에 기반하여 이 그래프를 따라 동적으로 실행합니다. 결정 그래프는 노드가 절차의 단계를 나타내며, 엣지는 논리적 또는 시간적 의존성을 인코딩하여 조건부 추론이 가능한 구조를 제공합니다.

- **Performance Highlights**: PADME는 ALFWorld와 ScienceWorld를 포함한 네 가지 다양한 벤치마크에서 최첨단 성능을 달성했습니다. 이러한 결과는 그래프 기반 절차 표현을 사용한 에이전트들이 신뢰할 수 있고 일반화 가능한 실행을 위한 강력한 중간 추상화를 제공함을 보여줍니다. 또한 PADME는 다양한 작업과 행동 어휘에 걸쳐 도메인 무관한 일반화를 입증하여, 여러 분야에서 일관된 성과를 내고 있습니다.



### AI Alignment Strategies from a Risk Perspective: Independent Safety Mechanisms or Shared Failures? (https://arxiv.org/abs/2510.11235)
Comments:
          under review

- **What's New**: 본 논문은 AI 정렬(Alignment) 기술의 다양한 실패 모드를 분석하여, 이들이 서로 어떻게 겹치는지를 이해하고자 합니다. 특히, 각기 다른 AI 안전 기술들이 동일한 실패 조건에서 어느 정도 공통점을 갖는지를 조사합니다. 이는 AI 안전 기술 연구에서 방어 심층(Defense-in-depth) 접근 방식의 중요성을 강조하며, 안전이 유지될 수 있는 다양한 방법을 모색하게 됩니다.

- **Technical Details**: AI 안전 연구 분야는 최근 몇 년간 상당한 성장을 이루었으며, 기존 AI 안전 문헌에서 여러 정렬 기술이 제안되었습니다. 이 섹션에서는 카테고리별로 네 가지 주요 AI 안전 기술의 패러다임적인 예를 검토합니다. 예를 들어, 인공지능 피드백으로부터의 강화 학습(RLAIF) 기술은 AI가 규칙에 따라 행동을 초래하도록 설계되어, 인간 피드백에 비해 더 많은 데이터 및 성능 개선을 기대할 수 있습니다.

- **Performance Highlights**: 본 논문은 다양한 AI 정렬 기술의 실패 모드를 분석하여, 각 기술 간의 상관 관계를 이해하고자 합니다. 그에 따라 안전 기술 연구의 우선 순위와 방향을 설정하는 데 중요한 통찰력을 제공합니다. 특히 표현 엔지니어링(Representation Engineering) 및 약한 감독을 통한 강한 일반화(Weak-to-Strong Generalization)와 같은 방법들은 AI 모델의 안전성 및 성능 향상에 기여할 수 있는 잠재력을 갖고 있습니다.



### Aligning Deep Implicit Preferences by Learning to Reason Defensively (https://arxiv.org/abs/2510.11194)
- **What's New**: 이 논문에서 제안하는 새로운 접근법인 Critique-Driven Reasoning Alignment (CDRA)는 대형 언어 모델(LLM)이 사용자 중심 인터페이스에서 개인의 심층적인 암묵적 선호를 더 잘 이해하도록 돕는 방법을 제시합니다. CDRA는 보상 매칭 작업을 구조화된 추론 과정으로 재구성하여, 사용자의 선호를 가시화하고 방어적 사고를 증진시키는 새로운 방법론을 구축합니다. 또한 DeepPref라는 데이터셋을 소개하여 3000개의 선호 질의 쌍을 통해 모델의 추론 과정을 지원합니다.

- **Technical Details**: CDRA 프레임워크는 두 가지 주요 문제를 해결합니다: 첫째, 비정형적 선호(superficial preferences)가 아닌 심층적인 암묵적 선호를 추론하는 것이고, 둘째, 쿼리의 모호성 내에서 위험을 식별하고 완화할 수 있는 방어적 사고를 수행합니다. 이를 통해 Pers-GenPRM을 통해 사용자 선호에 맞는 보상 모델을 생성하고, Critique-Driven Policy Alignment (CDPA)를 통한 정책 모델의 조정으로 각 단계의 추론에 대한 명확한 피드백을 제공합니다. 결과적으로, 투명하고 해석 가능한 보상 신호로 정책 모델을 안내합니다.

- **Performance Highlights**: 실험 결과, CDRA는 사용자의 진정한 선호를 발견하고 정렬하는 데 뛰어난 성능을 보였습니다. 세 가지 차원에서 많은 지표에 걸쳐 우수한 성능을 나타내며, 깊은 선호 이해와 강력한 추론 능력을 보여줍니다. 전체적으로 CDRA는 기존 방법론에 비해 더 신뢰할 수 있는 개인화된 응답을 제공하는 훌륭한 방법으로 평가됩니다.



### $How^{2}$: How to learn from procedural How-to questions (https://arxiv.org/abs/2510.11144)
- **What's New**: 이번 논문에서는 How^{2}라는 메모리 에이전트 프레임워크를 제안하며, 이를 통해 에이전트가 어떻게 질문을 하고 그에 대한 답변을 저장하며 평생 학습을 진행할 수 있는 방법을 구현합니다. Plancraft라는 Minecraft 제작 환경에서 평가를 진행하여, 에이전트가 자원을 조작하여 조립 작업을 완료하는 방식으로 구현하였습니다. 우리의 연구는 에이전트가 저수준의 실행 가능 행동에서부터 고수준의 하위 목표 설명까지 다양한 수준의 추상화된 답변을 통해 학습할 수 있음을 보여줍니다.

- **Technical Details**: How^{2} 프레임워크는 다양한 작동 역할을 가지며, 에이전트가 대화 기록을 바탕으로 다음 행동을 결정합니다. 메모리는 쿼리에 대한 답변을 캐시하는 변화 가능한 키-값 저장소이며, 질문에 대한 답변은 정밀 문자열 일치를 기반으로 검색됩니다. Plancraft 환경에서 에이전트는 재료를 조작하며 실행 가능한 작업의 스트림을 생성하는 동시에, 고수준의 질문을 통해 지식을 축적합니다.

- **Performance Highlights**: 우리의 접근법은 즉각적인 작업 성공을 위한 직접적인 실행 가능한 행동보다, 하위 목표 또는 추상화된 답변이 평생 학습에는 더 유익함을 보여줍니다. 이는 에이전트가 장기적으로 플래닝 능력을 개선하는 데 기여하며, 교사 모델의 응답이 높은 수준으로 추상화되어 있어야 효과적임을 알 수 있습니다. 따라서 How^{2}는 LLM 기반 에이전트가 상호작용 환경에서 질문을 통해 시간이 지나도 개선될 수 있는 경로를 제공합니다.



### Spec-Driven AI for Science: The ARIA Framework for Automated and Reproducible Data Analysis (https://arxiv.org/abs/2510.11143)
Comments:
          19 pages,5 figures

- **What's New**: 이번 논문은 ARIA (Automated Research Intelligence Assistant)라는 새로운 분석 프레임워크의 발전을 소개합니다. ARIA는 투명성과 자동화를 결합하여 연구 분석의 생산성을 높이는 것을 목표로 합니다. 이는 사용자 지정된 사양(spec)-기반으로 이루어져 있으며, 사람과 기계 간의 협업을 통합하는 문서 중심의 워크플로우를 특징으로 합니다.

- **Technical Details**: ARIA는 Command, Context, Code, Data, Orchestration, AI Module의 총 6개의 상호 운용 가능(interoperable) 레이어로 구성됩니다. 연구자는 자연어(Natural Language)로 분석 목표를 설정하고, ARIA는 실행 가능한 코드를 자동 생성하며, 계산을 검증하고, 투명한 문서를 생성합니다. 이를 통해 매개변수 튜닝과 반복 실험을 최소화하면서 최적의 기능(feature) 세트를 신속하게 식별합니다.

- **Performance Highlights**: 보스턴 주택 사례(Boston Housing case)에서는 ARIA가 25개의 주요 기능을 발견하고, XGBoost 모델을 최상의 성능 모델로 결정했습니다 (R square = 0.93). 다양한 도메인에서 수행된 평가 결과, ARIA는 최첨단 시스템들과 비교해 뛰어난 성능, 해석 가능성(Interpretability), 효율성을 입증했습니다. ARIA는 투명하고 협력적인 재현 가능(reproducible) 과학 발견의 새로운 패러다임을 설정합니다.



### Improving AI Efficiency in Data Centres by Power Dynamic Respons (https://arxiv.org/abs/2510.11119)
- **What's New**: 최근 인공지능(AI)의 발전은 대규모 언어 모델과 기본 모델과 같은 복잡한 모델의 개발에 힘입어 가속화되고 있습니다. AI 데이터 센터의 전력 관리 문제는 환경 및 지속 가능한 발전에 미치는 영향으로 인해 더욱 주목받고 있습니다. 이 논문에서는 AI 데이터 센터의 전력 관리를 위한 혁신적인 접근 방식과 이를 통해 얻는 이점을 탐구합니다.

- **Technical Details**: AI 데이터 센터의 전력 요구는 고성능 데이터 분석 시스템의 발달로 인해 크게 증가했습니다. 이러한 데이터 센터는 극단적인 데이터 볼륨을 처리하며, 각각의 데이터 센터는 엄청난 전력 소비로 인해 환경에 미치는 영향이 큽니다. 연구에서는 데이터 처리와 관련된 다양한 전력 추세를 분석하여, 전력 관리에서의 능률을 향상시키기 위한 수동 및 능동 장치의 디지털 비교를 수행하였습니다.

- **Performance Highlights**: AI 데이터 센터에 대한 전력 관리 접근 방식의 변화는 지속 가능성을 크게 개선할 수 있는 잠재력을 가지고 있습니다. 이를 통해 에너지 효율, 자본 지출 절감 및 관리 비용 최소화가 가능합니다. 특히, AI 데이터 센터의 전력 균형을 동적으로 조정함으로써, 고성능 AI 처리의 성능 및 효율성을 극대화할 수 있습니다.



### Modeling AI-Driven Production and Competitiveness A Multi-Agent Economic Simulation of China and the United States (https://arxiv.org/abs/2510.11085)
- **What's New**: 이 논문은 인공지능(AI) 기술의 급속한 발전에 따른 '인간-AI 공동 창조'의 새로운 단계를 탐구하고, 미국과 중국의 거시경제적 맥락에서 서로 다른 생산 메커니즘의 비교를 통해 AI의 효과를 분석합니다. 이전의 다중 수준 지능형 에이전트 경제 모델을 기반으로 하여, 시뮬레이션을 통해 AI가 독립적인 생산 주체로 작용할 때 사회적 산출의 성장률이 전통적인 인간 노동 모델을 초과함을 보여줍니다. 이러한 연구 결과는 AI 주도의 생산 시스템 전환을 이해하는 체계적인 분석 프레임워크를 제시하며, 정책 수립에 대한 정량적 통찰력을 제공합니다.

- **Technical Details**: 논문은 AI 에이전트 통합 이후 중국과 미국의 산출 성과를 평가하기 위해 다섯 가지 진화된 지능형 에이전트 경제 모델 프레임워크를 제안합니다. 각 모델은 인간-AI 협업의 다양한 메커니즘을 반영하며, 모형 1은 순수한 인간 협업 시나리오를 모델링하고, 모형 2는 AI 에이전트를 협업체로 도입합니다. 이후의 모델들은 네트워크 효과, AI의 자립적 생산, 그리고 이 두 가지를 결합하여 인간과 AI의 협업 정도를 달리하는 여러 메커니즘을 표현합니다.

- **Performance Highlights**: 중국은 지능형 에이전트 인구의 확대와 기술적 접속 속도 모두에서 뚜렷한 가속 가능성을 보여, 기술적 수렴이나 부분적인 초과 달성의 가능성을 제공합니다. 특히, 연구에 따르면 AI 에이전트의 도입이 미시적 생산성과 거시적 복지를 향상시키며, 서로 다른 구조적 파라미터들에 따라 AI의 협업 효과가 다르게 나타난다고 밝혀졌습니다. 이러한 성과는 국가별 AI 발전 정책을 보다 정확히 수립하는데 기여할 것입니다.



### Argumentation-Based Explainability for Legal AI: Comparative and Regulatory Perspectives (https://arxiv.org/abs/2510.11079)
- **What's New**: 최근 인공지능(AI) 시스템이 법률 분야에서 널리 사용됨에 따라 이들 시스템의 불투명성이 공정성(fairness), 책임(accountability), 신뢰(trust)에 대한 중대한 문제를 초래하고 있습니다. 이 논문은 법률적 관련성을 지닌 설명을 제공하는 데 있어 인수론(computational models of arguments)의 역할을 강조하며, 유럽연합 일반 데이터 보호 규정(GDPR) 및 인공지능법(AIA)과 같은 emerging regulatory frameworks와의 정렬에 주목합니다. 다양한 설명 전략의 강점과 한계를 분석하고, 인수론 프레임워크가 법의 가치 민감성(value-sensitive nature) 및 논쟁성(contestable nature)을 포착하는 데 특히 강력한 토대를 제공함을 밝혀냅니다.

- **Technical Details**: AI 시스템은 복잡한 데이터를 처리하는 데 있어서 뛰어난 성능을 보이지만, 블랙박스 문제(black box problem)로 인해 투명성이 결여되어 있습니다. 이는 입력 데이터가 결과로 변환되는 과정을 이해하기 어려워 사용자가 모델의 내부 논리를 추적하기 어렵게 만듭니다. 이 연구에서는 다양한 설명 가능성(Explainability) 기술을 소개하고, 각 기술이 법적 추론에서 어떻게 적용될 수 있는지를 탐색하며, 인수론 기반 설명의 강점을 확대하여 법률 AI 시스템에서의 투명성을 보장하는 방법을 제시합니다.

- **Performance Highlights**: 인공지능이 의사결정에 미치는 영향은 심각할 수 있으며, 대중의 신뢰를 향상시키기 위해 투명한 알고리즘의 필요성이 강조됩니다. 연구자들은 다양한 접근 방식을 통해 설명 가능성을 높이려는 다양한 기법을 개발하고 있으며, 이는 모형의 복잡성 및 사용자의 요구에 따라 차별화됩니다. 본 논문은 특히 인수론 기반의 접근 방식이 법률의 기술적 투명성과 책임을 보장하는 데 있어 가장 강력한 프레임워크를 제공한다고 주장하며, 공정성 및 윤리적 요구를 충족시킬 수 있는 방향으로 향후 연구를 제안합니다.



### FBS Model-based Maintenance Record Accumulation for Failure-Cause Inference in Manufacturing Systems (https://arxiv.org/abs/2510.11003)
- **What's New**: 이 연구에서는 제조 시스템의 유지보수 기록을 기반으로 한 함수-행동-구조(Function-Behavior-Structure, FBS) 모델 기반의 진단 지식 온톨로지를 개발하였다. 이를 통해 고장 원인 추론을 보다 효과적으로 수행할 수 있는 방법론을 제안하였다. 특히, 전문가의 정성적인 데이터와의 높은 일치를 보여주며, 비전문가도 활용할 수 있는 초기 문제 해결 방안을 마련하고자 하였다.

- **Technical Details**: 연구의 핵심은 심층 지식과 피상 지식의 명확한 구조화를 요구하는 지식 기반 고장 진단의 두 가지 필요 조건을 충족하는 것이다. FBS 모델을 사용하여 AIAG & VDA의 기능적 계층을 통합하고, 이 틀 안에서 고장 사건과 원인 관계를 체계적으로 연결하여 누적하는 방법을 제안하였다. 이를 통해 관계형, 실현 관계 및 순차적 관계를 고려한 추론의 기초를 구축하였다.

- **Performance Highlights**: 제안된 방법은 축적된 유지보수 기록을 통해 고장 원인 추론의 정확도를 높였으며, 특히 관련 사례 수가 적고 용어가 상이한 어려운 경우에서도 전문가가 언급한 원인 후보 집합과의 일치를 증가시켰다. 이 연구는 고장 원인 추론 방법론의 발전을 위한 기초 자료를 제공하며, 앞으로 더 넓고 다양한 시스템에서 검증할 필요가 있다고 강조한다.



### Revisiting Model Interpolation for Efficient Reasoning (https://arxiv.org/abs/2510.10977)
Comments:
          14 pages, 6 figures, 7 tables. Working in progress

- **What's New**: 본 논문은 모델 병합(model merging) 기법을 심층적으로 재조명하며, 두 개의 모델 가중치를 직접 보간(interpolation)하는 가장 간단한 병합 방법을 분석합니다. 저자들은 모델 보간이 사고 경로(reasoning trajectory)에서 세 가지 단계를 따르는 진화적 패러다임을 가지고 있음을 발견하였습니다. 이 연구는 모델 보간의 메커니즘을 설명하고, 전략적으로 보간된 모델이 고급 모델 병합 기준을 초월할 수 있다는 점을 강조합니다.

- **Technical Details**: 연구에서는 모델 보간 방식이 세 가지 단계로 나뉘며, 각 단계에서 Pass@k와 Mean@k의 성능 지표가 비선형적으로 발전하는 과정을 살펴봅니다. 가중치 주도 모델과 인스트럭션 모델을 포함한 하이브리드 모델을 통해, 복잡한 작업에 대한 효율적이면서도 효과적인 추론 능력을 확보할 수 있는 방법론을 제시합니다. 또한, 모델의 층(layer) 및 모듈(modules)과 디코딩 전략(decoding strategies)에 대한 정밀한 실험을 통해 깊이 있는 분석을 수행합니다.

- **Performance Highlights**: 실험 결과, 전략적으로 보간된 모델이 다양한 도전적인 벤치마크에서 기존의 고급 모델 병합 기준을 초과하는 성능을 보여주었습니다. 특히, 수학적 추론, 명령 이행, 과학적 추론을 포함한 여러 기준에서 성과를 나타냈습니다. 이 연구는 특정 토큰 예산(token budget)을 준수하는 모델 설계에 대한 실제적인 프레임워크를 제공합니다.



### Video-STR: Reinforcing MLLMs in Video Spatio-Temporal Reasoning with Relation Graph (https://arxiv.org/abs/2510.10976)
- **What's New**: 최근 멀티모달 대형 언어 모델(Multi-modal Large Language Models, MLLMs)의 발전으로 강력한 의미 이해 능력이 입증되었지만, 정밀한 시공간 이해(spatio-temporal understanding)를 수행하는 데 어려움을 겪고 있습니다. 기존의 방법은 주로 비디오 자체에 초점을 맞추고 물리적 정보는 간과함으로써 실제 애플리케이션에서의 활용에 제약을 주었습니다. 이러한 문제를 해결하기 위해, 우리는 Video-STR이라는 새로운 그래프 기반 강화 학습(reinforcement learning) 방법론을 제안합니다.

- **Technical Details**: Video-STR은 강화 학습과 검증 가능한 보상(Reinforcement Learning with Verifiable Reward, RLVR)을 바탕으로 하여 모델의 능력을 향상시키고, 그래프 기반의 상대적 정책 최적화(Group Relative Policy Optimization, GRPO) 메커니즘을 도입하여 모델이 시나리오의 시공간(topology)을 추론하도록 유도합니다. 또한 정밀한 훈련을 지원하기 위해 205k의 질문-답변(QA) 쌍으로 구성된 STV-205k 데이터셋을 구축했습니다. 이 데이터셋은 실내 및 실외에서 동적 다중 객체 장면을 포함하고 있습니다.

- **Performance Highlights**: Video-STR은 다양한 벤치마크에서 최첨단(results achieved state-of-the-art) 성능을 보이며, STI-Bench에서 기본 모델보다 13% 성능 향상을 달성했습니다. 이 연구는 시공간 추론의 효과성을 입증하고, STV-205k 데이터셋이 모델 훈련에 기여한다는 것을 확인했습니다. 또한, 우리의 방법론과 데이터셋은 다양한 다운스트림 애플리케이션에서의 활용 가능성을 넓히고 있습니다.



### Scalable and Explainable Enterprise Knowledge Discovery Using Graph-Centric Hybrid Retrieva (https://arxiv.org/abs/2510.10942)
- **What's New**: 이번 논문에서는 Jira, Git 리포지토리, Confluence 및 위키와 같은 이종 시스템에 분산되어 있는 방대한 지식을 관리하는 현대 기업을 위한 새로운 프레임워크를 제안합니다. 기존의 키워드 검색이나 정적 임베딩(embedding) 기반 검색 방법은 복잡한 쿼리에 대한 인식(understanding) 및 다중 추론(multi-hop inference)이 요구되는 경우에 효과적이지 않았습니다. 제안된 모듈형 혼합 조회(framework) 구조는 Knowledge Base Language-Augmented Models (KBLam), DeepGraph 표현법, 임베딩 기반 의미 검색을 통합하여 적응형 정보 접근을 가능하게 합니다.

- **Technical Details**: 이 프레임워크는 코드, 풀 리퀘스트(pull requests), 커밋 히스토리(commit histories) 등을 포함한 파싱된(repositories) 리포지토리에서 통합 지식 그래프(knowledge graph)를 구축합니다. 이를 통해 의미적 유사성 검색(semantic similarity search), 구조적 추론(structural inference), 다중 홉 추론(multi-hop reasoning)이 가능해집니다. 쿼리 분석은 최적의 검색 전략을 동적으로 결정하며, 독립적 또는 융합 처리(fused processing)를 통해 구조화된(data) 및 비구조화(unstructured) 데이터 소스를 지원합니다.

- **Performance Highlights**: 대규모 Git 리포지토리에 대한 실험 결과는 통합 추론 계층(unified reasoning layer)이 GPT 기반 검색 파이프라인과 비교하여 최대 80%까지 답변의 관련성을 향상시킨다는 것을 보여줍니다. 그래프 생성, 혼합 추론(hybrid reasoning), 상호 작용 시각화(interactive visualization)를 결합함으로써, 제안된 프레임워크는 기업 환경에서 지능형 지식 어시스턴트(intelligent knowledge assistants)를 위한 확장 가능하고 설명 가능한 사용자 중심의 기반을 제공합니다.



### PoU: Proof-of-Use to Counter Tool-Call Hacking in DeepResearch Agents (https://arxiv.org/abs/2510.10931)
- **What's New**: 이 논문에서는 증거 기반 강화 학습 프레임워크인 Proof-of-Use (PoU)를 제안하여 Retrieval-augmented generation (RAG) 에이전트가 정보 검색 시 겪는 Tool-Call Hacking 문제를 극복하려고 합니다. Tool-Call Hacking은 에이전트가 도구 호출을 통해 표면적으로 보상 신호를 증대시키지만, 실제로는 검색된 증거를 활용하지 못하는 현상입니다. PoU는 에이전트의 추론 과정과 검색된 증거 간의 인과적 링크를 검증 가능하게 만드는 계약 기반 메커니즘을 포함하고 있습니다.

- **Technical Details**: PoU는 단계별 증거 검증, 민감도 보상, 답변-증거 정렬 목표를 포함한 통합 계약을 통해 기능적으로 기반이 잡히고 해석 가능성이 높은 에이전트를 설계합니다. 이 프레임워크는 중간 추론 단계에서 증거의 유용성과 관련 출처를 명시적으로 인용하여 이유와 증거의 종속성을 학습 가능한 인터페이스로 전환합니다. 그 결과, 도구 사용의 해석과 함께 모든 추론 단계에서 증거의 사실적 일관성을 보장합니다.

- **Performance Highlights**: 실험 결과, PoU는 다양한 질문 응답(QA) 벤치마크에서 DeepResearch의 기존 모델들에 비해 사실적 정확성, 증거 신뢰성 및 도구 라우팅의 균형에서 일관되게 우수한 성능을 나타냅니다. 이러한 성과는 RAG 에이전트가 단순한 작업 결과에 기반하기보다는 검색된 정보의 인과적 사용을 통해 신뢰할 수 있는 추론 보강의 필요성을 강조합니다.



### PaperArena: An Evaluation Benchmark for Tool-Augmented Agentic Reasoning on Scientific Literatur (https://arxiv.org/abs/2510.10909)
Comments:
          12 pages, 9 figures

- **What's New**: 본 논문에서는 PaperArena라는 새로운 평가 벤치마크를 제안하여, 복잡한 연구 질문을 해결하기 위해 여러 논문 간의 정보 통합 및 외부 도구와의 상호작용을 포함합니다. PaperArena는 다양한 형식의 정보 통합을 요구하며, 연구 환경에서 자주 발생하는 문제를 해결하기 위한 노력을 반영합니다. 또한, 연구자들이 보다 능력 있는 에이전트를 개발하고 평가하는 데 필요한 표준화된 플랫폼도 함께 제공합니다.

- **Technical Details**: PaperArena는 수천 개의 오픈-access AI 논문에서 샘플링한 데이터를 기반으로 하며, 멀티모달 대규모 언어 모델(MLLM)을 활용해 초기 질문-답변 쌍을 자동으로 생성합니다. 최종적으로 784개의 고품질 질문-답변 쌍을 마련하였으며, 에이전트의 전반적인 반응 추적 과정에서의 도구 사용 효율성을 평가합니다. 또한, PaperArena-Hub는 단일 또는 다중 에이전트 시스템을 평가하는 모듈형 플랫폼을 제공합니다.

- **Performance Highlights**: 실험 결과, LLM 기반의 기존 에이전트들이 평균 38.78%의 정확도를 보여주며, 더 어려운 문제 세트에서는 18.47%로 저조한 성과를 보였습니다. 이는 LLM이 인간 전문가의 83.5% 정확도와 비교할 때 현저히 낮은 수치임을 보여줍니다. 특히 에이전트들이 효율적인 도구 사용을 하지 못하고, 필요 이상으로 많은 도구를 호출하는 등의 패턴이 발견되어 앞으로의 개선 방향을 제시합니다.



### LLM-Empowered Agentic MAC Protocols: A Dynamic Stackelberg Game Approach (https://arxiv.org/abs/2510.10895)
Comments:
          This work has been submitted to IEEE for possible publication

- **What's New**: 이 논문에서는 LLM(대형 언어 모델)을 활용한 다중 에이전트 심층 강화 학습(MARL) 프레임워크를 제안합니다. 이를 통해 기존의 무선 네트워크에서 자주 발생하는 고정된 매체 접속 제어(MAC) 프로토콜의 한계를 극복하고, 동적인 환경에서도 프로토콜이 자율적으로 학습하고 적응할 수 있는 가능성을 제시합니다. 특히, 이론적으로 안정적인 스택엘버그 균형(Stackelberg equilibrium)의 존재와 수렴 행동을 분석하여 새로운 프로토콜 생성의 기틀을 마련했습니다.

- **Technical Details**: 제안된 프레임워크는 다중 추종자 스택엘버그 게임(MFSG)으로 MAC 프로토콜의 출현을 모델링하며, 기본 스테이션(BS)은 리더 에이전트로 작용하고 사용자 장치(UEs)는 추종자로 기능합니다. 프로토콜 액션 문법(Protocol Action Grammar, PAG)을 적용하여 안정성과 효율성을 보장합니다. 에이전트의 정책은 근접 정책 최적화(Proximal Policy Optimization, PPO)를 통해 지속적으로 피드백을 받아 업데이트되어, 동적인 네트워크 환경에 적응할 수 있습니다.

- **Performance Highlights**: 시뮬레이션 결과, 제안된 프레임워크는 기존 방법론에 비해 77.6% 더 높은 처리량과 65.2%의 공정성 향상을 달성함을 보여주었습니다. 또한, 변화하는 사용자 수에 대해서도 재학습이나 구조적 변경 없이 탁월한 일반화 성능을 발휘하여 효율적인 MAC 프로토콜을 자율적으로 생산할 수 있는 능력을 입증하고 있습니다.



### The Irrational Machine: Neurosis and the Limits of Algorithmic Safety (https://arxiv.org/abs/2510.10823)
Comments:
          41 pages, 17 figures, 5 tables

- **What's New**: 이번 연구에서는 인공지능에서 신경증(neurosis)을 특성화하는 프레임워크를 제시합니다. 특히, 내부적으로 일관된 행동이지만 현실과는 미스매치된 오작동 패턴을 다루며, 이러한 행동은 계획(planning), 불확실성 처리(uncertainty handling), 그리고 불쾌한 기억(aversive memory) 간의 상호 작용에서 발생한다고 설명합니다. 그리드 네비게이션 환경에서 다양한 신경증 행동 패턴을 분류하고 이들을 감지할 수 있는 경량 온라인 감지기와 재사용 가능한 탈출 정책을 제공합니다.

- **Technical Details**: 연구에서는 에이전트의 행동을 형성하는 내재적 규제 메커니즘과 예측의 필요성을 강조합니다. 에이전트는 불확실한 상황에서 살아남기 위해, 반복적인 선택과 기억이 결합되어 비합리적인 두려움이나 강박적인 행동을 보이게 되는 경향이 있습니다. 이러한 행동은 특정한 패턴을 끊을 수 있는 목표개입을 통해 해결될 수 있으나, 근본적인 신경증 구조는 단순한 수정으로는 완전히 해결되지 않고, 내면의 논리와 역사적 관점을 시스템적으로 분석해야 한다고 주장합니다.

- **Performance Highlights**: 제안된 프레임워크는 내재적 학습 메커니즘이 어떻게 공포 회피를 초래할 수 있는지를 보여줍니다. 실험에서는 '신경증 점수(neurosis scores)'를 통해 행동의 일관성과 현실에서의 불일치를 정량적으로 평가하며, 각기 다른 행동 유형을 가진 에이전트를 분석했습니다. 결과적으로, 특정 지역적 해결책이 충분하지 않음을 보여주며, 전반적인 실패를 드러내기 위해서는 유전자 프로그래밍 기반의 파괴적 테스트가 필요하다고 제안합니다.



### DRIFT: Decompose, Retrieve, Illustrate, then Formalize Theorems (https://arxiv.org/abs/2510.10815)
- **What's New**: 새로운 DRIFT 프레임워크는 대형 언어 모델(LLM)이 비공식 수학 명제를 작고 더 관리하기 쉬운 '하위 구성요소'로 분해하도록 지원합니다. 이를 통해 Mathlib과 같은 수학 라이브러리에서 기본 전제들을 더 효율적으로 검색할 수 있게 됩니다. 또한, 해당 프레임워크는 예제 정리를 검색하여 모델이 전제를 더 효과적으로 사용하도록 돕는 새로운 접근법을 제공합니다.

- **Technical Details**: DRIFT는 네 단계로 구성된 프로세스를 통해 비공식 수학 진술의 복잡성을 처리하고, 검색된 형식적 객체의 증명 예시를 제공합니다. 첫 번째 단계에서는 LLM이 비공식 진술을 작은 하위 쿼리로 분해합니다. 이후 이 쿼리는 Mathlib와 같은 형식적 라이브러리에서 의존하는 전제를 검색하는 데 사용되며, 이로 인해 보다 정확한 정의 검색이 가능합니다.

- **Performance Highlights**: DRIFT는 ProofNet 및 ConNF 벤치마크에서 새로운 최첨단 성과를 달성하였고, 특히 ConNF 벤치마크에서 GPT-4.1과 DeepSeek-V3.1을 사용하여 각각 37.14% 및 42.25%의 성장을 보여주었습니다. 이러한 분석은 수학 자동 형식화의 효과가 모델의 지식 경계에 크게 의존하고 있음을 강조하며, 각 모델의 능력에 맞춘 적응형 검색 전략의 필요성을 시사합니다.



### LLMs as Strategic Agents: Beliefs, Best Response Behavior, and Emergent Heuristics (https://arxiv.org/abs/2510.10813)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)이 에이전트의 행동을 고려해야 하는 상황에서 진정한 전략적 사고를 나타내는지를 탐구합니다. 이전의 연구들은 주로 균형 게임 동작과 깊이 있는 추론을 평가하는 데 집중해왔으며, LLM의 전략적 사고 능력은 제대로 분석되지 않았습니다. 새로운 프레임워크를 도입하여 믿음, 평가, 그리고 선택을 분리하여 LLM의 행동을 연구합니다.

- **Technical Details**: 연구의 대부분은 비협조적 게임을 통해 LLM의 선택과 추론 경로를 분석합니다. 특히, 이들은 정보가 완전한 상황에서 LLM이 어떻게 행동하는지를 분석합니다.  LLM의 선택이 경제적 합리성과 일치할 수 있음을 보이며, 그들의 추론은 상대의 정체성에 따라 조정됩니다. 다양한 복잡성을 지닌 게임 설정에서 더욱 많은 변화를 관찰할 수 있었습니다.

- **Performance Highlights**: LLMs는 다양한 상호작용 상황에서 최적의 대응 행동을 보였으며, 이는 이들의 추론 깊이와 관계가 있음을 나타냅니다. 상대에 대한 추측을 스스로 형성하며 이전의 추론을 기반으로 의사결정을 내릴 수 있음을 보여주었습니다. 또한, 복잡한 맥락에서 LLM은 자주 단순화된 의사결정 규칙을 따르는 등 복잡성이 증가함에 따라 논리적 변화가 발견되었습니다.



### Adaptive Selection of Symbolic Languages for Improving LLM Logical Reasoning (https://arxiv.org/abs/2510.10703)
- **What's New**: 이 논문은 기존의 연구와 달리 자연어(Natural Language, NL)에서 기호 언어(Symbolic Language, SL)로의 번역에서 SL 유형 선택이 성능에 미치는 영향을 강조합니다. 저자들은 각 논리적 추론 문제에 최적의 SL 형식이 있다는 주장을 제시하고 이를 실험적으로 검증했습니다. 이를 기반으로, 각 문제에 대해 가장 적합한 SL을 적응적으로 선택하는 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 세 가지 주요 단계로 구성됩니다: 적응형 기호 언어 선택, 번역 단계, 그리고 추론 단계입니다. LLM(대형 언어 모델)은 주어진 문제에 대해 FOL(1차 논리), LP(논리 프로그래밍), SAT(불린 만족도)에 대한 후보에서 가장 적합한 SL을 선택합니다. 이후, 선택된 SL로 NL 문제를 번역하고 전문적인 논리 해결기를 사용하여 최종 정답을 도출합니다.

- **Performance Highlights**: 실험 결과, 저자들이 제안한 적응형 선택 방법이 단일 SL로 번역하거나 SL을 임의로 선택하는 경우보다 성능이 크게 향상된다는 것을 보여주었습니다. 특히 혼합 데이터세트에서 96%의 정확도를 기록하며 1차 논리 번역에서 두 번째로 높은 정확도보다 25% 개선된 결과를 얻었습니다.



### Extended Triangular Method: A Generalized Algorithm for Contradiction Separation Based Automated Deduction (https://arxiv.org/abs/2510.10701)
Comments:
          38 pages, 8 figures

- **What's New**: 이 논문은 Automated deduction(자동 유도)의 핵심 개념을 다루며, 논리 추론의 효율성과 완전성을 조화롭게 하는 새로운 기법을 소개합니다. 이전의 CSE(Contradiction Separation Extension) 프레임워크를 기반으로 하여, 논리적 추론을 세quential resolution(순차적 추론)이 아닌 contradiction separation(모순 분리)의 과정으로 재정의합니다. 이를 통해 여러 절(clause) 간의 상호작용을 유연하게 지원할 수 있는 확장된 알고리즘인 ETM(Extended Triangular Method)도 개발되었습니다.

- **Technical Details**: ETM은 다수의 contradiction-building strategy(모순 구축 전략)를 통합하여 삼각형 기하학적(framework) 구조 안에서의 절 간 상호작용을 지원합니다. ETM은 표준 확장 방법(Standard Extension method)과 같은 다양한 모순 구축 전략을 통합한 것으로, 이러한 방법들은 동적 시너지를 허용하는 새로운 내부 메커니즘을 형성합니다. 알고리즘의 구현을 공식화함으로써, ETM은 CSE, CSE-E, CSI-E, CSI-Enig와 같은 여러 고성능 정리 증명기(theorem prover)의 핵심 알고리즘으로 자리 잡게 됩니다.

- **Performance Highlights**: ETM의 성능은 TPTP 문제 집합과 CASC 2018-2015 기준의 벤치마크에서 입증되었습니다. 이 접근 방식은 자동 추론(Automated reasoning) 환경에서 경쟁력을 갖춘 범용 모델로 진화하며, 향후 논리적 추론(logical inference) 및 정리 증명(theorem proving) 연구의 새로운 방향을 제시합니다. 실험 결과는 ETM이 제공하는 동적 시너지가 효과적임을 보여줍니다.



### OmniVideoBench: Towards Audio-Visual Understanding Evaluation for Omni MLLMs (https://arxiv.org/abs/2510.10689)
- **What's New**: 최근 멀티모달 대형 언어 모델(MLLMs)의 발전은 비디오 이해 분야에서 큰 잠재력을 보여주고 있습니다. 하지만 기존 벤치마크는 오디오와 비주얼 모달리티 간의 시너지를 평가하는 데 한계가 있으며, OmniVideoBench라는 새로운 벤치마크를 도입했습니다. 이 벤치마크는 1,000개의 고품질 질문-답변(QA) 쌍과 함께, 단계별 추론 과정을 명시하여 오디오-비주얼 이해를 평가합니다.

- **Technical Details**: OmniVideoBench는 YouTube와 Bilibili에서 수집된 다양하고 고품질 비디오로 구성되어 있으며, 8개의 주요 카테고리와 68개의 하위 카테고리를 포함합니다. 각 QA 쌍에는 엄격한 수동 주석이 포함되어 있으며, 질문 유형은 인과 추론, 요약, 공간 로컬라이제이션 등 13가지로 분류됩니다. 평가 프로세스는 모델이 비디오, 오디오 및 관련 텍스트를 처리하여 텍스트 답변을 생성하는 능력을 평가합니다.

- **Performance Highlights**: 여러 MLLM을 OmniVideoBench에서 평가한 결과, 모델 성능과 인간의 추론 능력 간에 큰 격차가 존재하는 것으로 나타났습니다. 현재 MLLM은 OmniVideoBench에서 60% 미만의 정확도를 기록하고 있으며, 최고의 모델조차도 58.90%에 불과합니다. 오픈 소스 모델은 성능이 무작위에 가까운 것으로 나타났으며, 이는 실제 오디오-비주얼 추론의 어려움을 강조합니다.



### Simpliflow: A Lightweight Open-Source Framework for Rapid Creation and Deployment of Generative Agentic AI Workflows (https://arxiv.org/abs/2510.10675)
- **What's New**: Generative Agentic AI 시스템은 복잡한 작업을 자동화하기 위한 강력한 패러다임으로 등장하고 있습니다. 그러나 기존의 프레임워크는 복잡성을 증가시키고, 높은 학습 곡선과 많은 보일러플레이트 코드를 요구하여 신속한 프로토타입 제작과 배포에 장애가 되곤 했습니다. 이 논문은 이러한 문제를 해결하기 위해 설계된 경량의 오픈 소스 Python 프레임워크인 simpliflow를 소개합니다.

- **Technical Details**: simpliflow는 선언적이고 JSON 기반의 구성 방식을 통해 선형적인 결정론적 에이전트 워크플로우를 신속하게 개발하고 조정할 수 있게 해줍니다. 모듈화된 아키텍처를 특징으로 하며, 에이전트 관리, 워크플로우 실행 및 후처리를 분리하여 사용 용이성과 확장성을 촉진합니다. LiteLLM과 통합되어 100개 이상의 대형 언어 모델(LLM)을 즉시 사용할 수 있는 점 또한 특징입니다.

- **Performance Highlights**: simpliflow는 복잡한 다단계 작업을 효율적으로 자동화하거나 가속화할 수 있도록 지원하며, JSON 기반의 워크플로우 정의로 명확성과 제어를 제공합니다. 인간의 승인 과정을 지원하여 각 단계에서 유연하게 운영할 수 있으며, 모든 상호작용이 구조화된 JSON으로 기록되어 감사 가능성을 높입니다. LangChain이나 AutoGen와 같은 다른 프레임워크와의 비교를 통해 simpliflow가 단순성과 제어, 속도에서 최적화된 도구로서 독특한 위치에 있음을 강조합니다.



### Unlocking Exploration in RLVR: Uncertainty-aware Advantage Shaping for Deeper Reasoning (https://arxiv.org/abs/2510.10649)
- **What's New**: 이번 연구에서는 UnCertainty-aware Advantage Shaping (UCAS)라는 새로운 접근 방식을 도입하여 강화 학습의 신뢰성 있는 보상(RLVR)에서 발생하는 문제를 해결합니다. UCAS는 모델의 내부 불확실성 신호를 활용하여 크레딧 할당 방식을 개선하는 모델 프리 방법입니다. 이 방법은 응답 수준과 토큰 수준에서 두 단계로 진행되며, 올바른 응답에 대한 보상을 높이고 잘못된 응답에 대한 과도한 신뢰를 페널티합니다.

- **Technical Details**: UCAS는 두 개의 상호 보완적인 수준에서 보상을 조정하여 정밀도와 다양성 간의 균형을 이루고자 합니다. 응답 수준에서, 모델의 전체적인 자기 신뢰를 기반으로 시퀀스의 보상을 조정하고, 토큰 수준에서는 원시 로짓에서 유래된 불확실성 기반의 페널티를 도입합니다. 이 계층적 접근 방식은 정책 업데이트를 세분화하여 조기 수렴을 촉진하지 않고도 탐색을 장려합니다.

- **Performance Highlights**: 다양한 수학적 추론 벤치마크에서 실시한 광범위한 실험을 통해 UCAS는 기존 RLVR 기준선보다 일관되게 뛰어난 성과를 보였습니다. UCAS는 보상 향상뿐만 아니라 추론의 다양성을 높이고 엔트로피 붕괴를 크게 완화함으로써 불확실성이 세분화된 학습 신호로 작용하는 효과를 입증하였습니다. 이를 통해 모델의 문제 해결 능력이 더욱 향상되었습니다.



### Hierarchical Optimization via LLM-Guided Objective Evolution for Mobility-on-Demand Systems (https://arxiv.org/abs/2510.10644)
- **What's New**: 이 논문은 구동 최적화를 위한 대형 언어 모델(LLM)과 수학적 최적화를 통합한 혁신적인 하이브리드 프레임워크를 제안합니다. 기존의 강화 학습(RL) 접근 방식의 데이터 비효율성과 실제 운영 제약의 실행 어려움을 해소하기 위해 LLM을 메타 최적화기로 활용하여 고급 목표를 생성합니다. 이 프로세스는 하모니 검색(harmony search)을 기반으로 하여 LLM의 프롬프트를 반복적으로 개선하고, 시간 효율성을 극대화하여 택시 서비스의 질을 높이는 데 중점을 둡니다.

- **Technical Details**: 제안된 프레임워크는 두 단계로 계층적으로 구성되어 있습니다. 고급 모듈은 실시간 공간 구성 및 공급-수요 불균형을 반영하여 승객을 택시에 할당하는 역할을 하며, 저급 모듈은 수학적 엄격함을 바탕으로 각 택시의 경로 문제를 해결합니다. LLM은 메타-휴리스틱 디자이너로 작용하여 도시 이동 패턴에 대한 내재적 이해를 통해 고급 목표를 동적으로 발전시킵니다.

- **Performance Highlights**: 뉴욕 및 시카고 택시 데이터셋을 기반으로 한 실험 결과, 제안된 접근 방식은 최첨단 기준선에 비해 평균 16%의 향상을 달성하여 승객 대기 시간을 줄인 것으로 나타났습니다. 이를 통해 제안된 하이브리드 LLM-최적화 모형이 기존 방법보다 더 효율적임을 입증하였습니다.



### Equity-Aware Geospatial AI for Forecasting Demand-Driven Hospital Locations in Germany (https://arxiv.org/abs/2510.10640)
Comments:
          7 pages. Application: this https URL Codebase: this https URL

- **What's New**: 이 논문은 EA-GeoAI라는 통합 프레임워크를 소개하여 독일의 병원 공급과 수요 예측을 2030년까지 지원합니다. 인구 통계학적 변화와 인프라 균형을 고려한 Equity Index를 통해 의료 접근성의 불균형을 해결하고자 합니다. 최적화된 AI 알고리즘을 통해 병상 배치와 시설 위치를 결정함으로써 공정한 의료 서비스를 제공하도록 하는 새로운 접근법을 제시합니다.

- **Technical Details**: 이 연구는 공간 데이터와 인구 예측을 통합하여 병원 배치에 대한 최적화 모델을 개발합니다. 기계 학습(ML) 및 최적화 기법을 활용하여 의료 자원의 분포를 평가하고, 접근성이 부족한 지역을 식별합니다. 실험 방법론에서는 Python 모듈을 통해 데이터 전처리, 지수 계산, 수요 예측 및 모델 평가 단계를 거칩니다.

- **Performance Highlights**: 이 연구는 공정성과 효율성을 동시에 고려한 병원 위치 최적화 모델을 평가하며, 여러 성과 지표(Equity Score, 평균 및 중앙값 이동 시간 등)를 통해 성과를 검증합니다. 결과적으로 제안된 모델은 기존 병원과 비교해 공공 보건 요구를 더 잘 충족시키며, 미래 연구에 대한 방향성을 제시합니다.



### Automatic Piecewise Linear Regression for Predicting Student Learning Satisfaction (https://arxiv.org/abs/2510.10639)
- **What's New**: 이 연구는 COVID-19 팬데믹 기간 동안 학생들의 학습 만족도에 영향을 미치는 다양한 요인들을 탐구하였으며, 최근의 해석 가능한 기계 학습 모델인 자동 구간 선형 회귀(Automatic Piecewise Linear Regression, APLR)가 학습 만족도를 예측하는 데 가장 적합한 모델임을 입증하였습니다. 교사들은 APLR의 전 세계적 및 개인 수준 해석을 통해 학생 프로필에 따라 맞춤형 교육을 제공할 수 있는 기회를 얻습니다. 이 연구는 APLR이 기존의 배깅(bagging) 및 부스팅 트리(boosted trees), 심지어 트랜스포머 기반 딥러닝 모델보다 뛰어난 성능을 보임을 강조합니다.

- **Technical Details**: 이 논문에서는 302명의 성균관대학교 학생을 대상으로 한 단면적 연구에서 COVID-19 팬데믹 동안 온라인 학습 경험을 바탕으로 학습 만족도에 영향을 미치는 인자들을 분석하였습니다. 자동 구간 선형 회귀(APLR)는 시각적으로 모델의 결정을 설명하고, 복잡한 데이터에서의 예측을 가능하게 하는 해석 가능한 기계 학습 방법입니다. 본 연구는 APLR의 성능과 해석력을 통해 전 세계 집단 및 개별 학생들에게 영향을 미치는 요인들을 발견하였습니다.

- **Performance Highlights**: APLR은 5개 지표 중 4개에서 대표적인 배깅 및 부스팅 트리, 해석 가능한 가법 모델(interpretative additive model), 트랜스포머 기반 심층 학습 모델보다 뛰어난 예측 성능을 보였습니다. 이 연구는 학생들의 시간 관리, 집중력, 동료에 대한 유용성 인식, 오프라인 수업 참여가 학습 만족도에 가장 큰 긍정적 영향을 미친다고 밝혔습니다. 흥미롭게도, 창의적 활동이 학습 만족도에 긍정적인 영향을 미치지 않았다는 결과도 도출되었습니다.



### Collaborative Text-to-Image Generation via Multi-Agent Reinforcement Learning and Semantic Fusion (https://arxiv.org/abs/2510.10633)
Comments:
          16 pages, 13 figures

- **What's New**: 이번 연구에서는 다양한 시각적 도메인 간에 의미적 정합성과 전문 수준의 세부 사항을 유지하는 데 어려움을 겪고 있는 다중 모드 텍스트-이미지 생성의 제약을 해결하기 위해 다중 에이전트 강화 학습 프레임워크를 제안합니다. 텍스트 향상 모듈과 이미지 생성 모듈로 구성된 두 개의 결합된 서브시스템에서 도메인 전문화 에이전트를 조율합니다. 에이전트는 의미적 유사성, 언어적 시각적 품질 및 콘텐츠 다양성을 균형 있게 고려한 복합 보상 함수 아래 Proximal Policy Optimization (PPO)을 사용하여 훈련됩니다.

- **Technical Details**: 이 시스템은 아키텍처, 초상화 및 풍경 작업에 특화된 도메인 전문 에이전트로 구성되어 있으며, 각 에이전트는 전문 지식과 생성 능력을 갖추고 있습니다. 교차 모드 정렬은 특수화된 융합 모듈을 통해 강제되며, 평가 프레임워크는 언어적, 시각적 및 의미적 측정을 통합합니다. 이 설계는 전문성과 일반화를 명시적으로 균형 잡고 두 모드 간 의미적 충실성을 강조합니다.

- **Performance Highlights**: 여섯 가지 실험 환경에서 우리의 시스템은 생성된 콘텐츠를 크게 풍부하게 하며(단어 수 1614% 증가), ROUGE-1 점수를 69.7% 감소시켰습니다. Transformer 기반 전략은 복합 점수에서 가장 높은 점수를 기록했지만(0.521) 때때로 안정성 문제를 경험했습니다. 이러한 결과는 신뢰할 수 있는 다중 모드 생성 시스템을 발전시키기 위한 협력적이고 전문화-driven 아키텍처의 가능성을 강조합니다.



### EA4LLM: A Gradient-Free Approach to Large Language Model Optimization via Evolutionary Algorithms (https://arxiv.org/abs/2510.10603)
- **What's New**: 최근 대형 언어 모델(LLMs)의 최적화 방법에 대한 새로운 접근법이 제안되었습니다. 이번 연구에서는 진화 알고리즘(Evolutionary Algorithms, EA)을 활용하여 LLM을 최적화하는 EA4LLM이라는 새로운 방법을 소개합니다. 이 방식은 기존의 기울기 기반 최적화 방법이 아닌 대안적인 최적화 기술을 사용하여, 하드웨어 요구사항을 줄이고 더욱 다양한 비차별적 아키텍처를 활용할 수 있도록 합니다.

- **Technical Details**: 연구는 모델의 출력 logit을 진화 전략(ES)의 피트니스 함수와 연결하여 최적화합니다. 이를 통해 다음 토큰 예측 품질을 텍스트 코퍼스에서 평가하며, 샘플에 따라 평균 log 확률을 이용해 전체 피트니스를 정의합니다. ES는 기울기 정보를 요구하지 않기 때문에 비차별적 문제에도 적용 가능하여, 다양한 신경망 아키텍처의 최적화를 실현할 수 있습니다.

- **Performance Highlights**: EA4LLM 방법은 1억 개의 매개변수를 가진 LLM을 성공적으로 학습시키는 것을 보여주었으며, 실험을 통해 진화 알고리즘이 신경망 최적화에 효과적일 수 있음을 입증하였습니다. 이 연구는 기울기 기반 방법이 유일한 대안이라는 기존의 가정을 도전하며, 불균형한 연산 자원이 있는 연구 그룹도 심층 학습 연구에 참여할 수 있는 기회를 제공합니다.



### A Distance Measure for Random Permutation Set: From the Layer-2 Belief Structure Perspectiv (https://arxiv.org/abs/2510.10596)
- **What's New**: 이 논문에서는 최근에 제안된 랜덤 순열 집합(Random Permutation Set, RPS) 이론을 통해 순서 구조화된 불확실한 정보를 표현하는 새로운 방법론을 제시합니다. 두 가지 관점인 랜덤 유한 집합(Random Finite Set, RFS)과 전이 신뢰 모델(Transferable Belief Model, TBM)에서 RPS 간의 거리 측정에 대한 심층 분석이 이루어집니다. 새로운 누적 자카르드 지수(Cumulative Jaccard Index)의 정의를 도입해 두 개의 순열 간 유사성을 정량화하고, 이를 바탕으로 RPS 간의 거리 측정 방법을 제안합니다.

- **Technical Details**: 제안된 거리 측정 방법은 순열 질량 함수(Permutation Mass Function, PMF)의 구조적 속성을 조사하며, 누적 자카르드 지수 행렬의 양의 정의성 분석도 포함되어 있습니다. 이 방법은 순위가 높은 요소들 간의 불일치가 더 큰 거리 값으로 이어지는 자연스러운 상위 가중성(Top-weightiness) 속성을 가지고 있습니다. 의사 결정자는 가중치 및 절단 깊이(Truncation Depth)를 조정할 수 있는 두 가지 매개변수를 제공합니다.

- **Performance Highlights**: 제안된 방법은 기존 방법의 단점을 극복하면서도 Jousselme 거리와 호환되며 더 높은 감도와 유연성을 제공합니다. 여러 수치 예제를 통해 기존 방법과 비교하여 제안된 방법의 성능이 향상됨을 보여줍니다. 특히, 의사 결정자는 속성을 강화하거나 약화할 수 있는 가능성을 제공받아, 특정 애플리케이션의 필요에 맞춰 거리 측정을 수행할 수 있습니다.



### A Layered Intuition -- Method Model with Scope Extension for LLM Reasoning (https://arxiv.org/abs/2510.10592)
- **What's New**: 본 논문은 기존의 메소드 기반 추리(method-based reasoning)와 범위 확장(scope extension) 개념을 통합하여, 직간접적으로 미접근(indirected) 문제를 보다 체계적으로 다루기 위한 직관-메소드 계층 모델(Intuition-Method Layered Model with Scope Extension)을 제안합니다. 이 모델은 직관 기반 사고(intuition-based thinking)와 메소드 기반 사고(method-based thinking)를 통합하며, 수직(원인 분석) 및 수평(유사 문제)으로 문제의 적용 가능성을 넓히고, 시간과 공간 차원(temporal and spatial dimensions)에서의 확장성을 통해 추리 능력을 강화합니다.

- **Technical Details**: 이 프레임워크는 메소드 기반 추리가 질문과 해결책을 독립적인 재사용 가능한 단위로 분리하는 방식으로 작동하며, 다양한 범위 확장을 적용하여 적응성을 향상시킵니다. 체계적 지식 트리(systematic knowledge trees)가 이러한 확장을 구조적 위계로 조직하여, 더 큰 지식 네트워크로 연결합니다. 또한, 방법 확장의 엔트로피(entropy of method extension)를 정의하여, 시스템이 미접근 문제를 해결하는 능력을 정량적으로 측정하는 새로운 지표를 제안합니다.

- **Performance Highlights**: 제안된 직관-메소드 계층 모델은 기존의 프리 트레인(pre-trained) 매핑을 넘어서는 추리 능력을 확장하며, 비직접적인 문제들을 더욱 강력하고 체계적으로 다룰 수 있는 기회를 제공합니다. 이 모델은 기존의 방법들을 논리적으로 통합하여, 새로운 기여와 함께 보다 포괄적이고 확장 가능한 추리 패러다임을 제시합니다. 이를 통해 실제 문제 해결에 보다 적합한 AI 시스템으로서의 기능을 강조합니다.



### ELAIPBench: A Benchmark for Expert-Level Artificial Intelligence Paper Understanding (https://arxiv.org/abs/2510.10549)
Comments:
          25 pages, 20 figures

- **What's New**: 이번 논문에서는 대규모 언어 모델(LLMs)이 학술 논문을 깊이 이해하고 추론하는 능력이 부족하다는 점을 강조합니다. 이를 해결하기 위해 인공지능(AI) 연구 논문 이해도를 평가하기 위한 새로운 벤치마크인 ELAIPBench를 소개합니다. 이 벤치마크는 403개의 선택형 질문으로 구성되며, 숙련된 전문가들이 참여하여 난이도를 고려한 질문을 작성했습니다.

- **Technical Details**: ELAIPBench의 질문은 3개의 난이도 수준으로 설정되어 있으며, 단순한 정보를 검색하는 것이 아니라 비약적인 추론을 요구합니다. 문제의 난이도와 질을 보장하기 위해 20명의 학술 연구 경험이 있는 인력을 채용하여 질문 생성을 경쟁 방식으로 진행했습니다. 검증과정에서 LLM을 활용해 매칭된 질문을 필터링하여 최종 질문을 결정합니다.

- **Performance Highlights**: 실험 결과, 최고 성능의 LLM은 39.95%의 정확도를 기록했으며, 이는 인간 전문가의 성과에 비해 크게 떨어지는 수치입니다. 또한, 사고 모드나 검색 보강 생성(RAG) 시스템이 장착된 최첨단 LLM조차도 성능 향상에 실패하거나 잘못된 추론으로 인한 정확도 저하를 보였습니다. 따라서 현재 LLM의 능력이 학술 논문을 진정으로 이해하기에는 미흡하다는 결론을 내리게 되었습니다.



### Tracing the Traces: Latent Temporal Signals for Efficient and Accurate Reasoning (https://arxiv.org/abs/2510.10494)
- **What's New**: 이 논문에서는 Latent-Trajectory 신호(Latent-Trajectory signals)를 도입하여 모델의 내부 표현이 중간 추론 토큰을 생성하는 과정에서 시간적으로 어떻게 변화하는지를 분석합니다. 이러한 신호를 통해 추론 경로가 정답으로 이어질 확률을 더 정확하게 예측할 수 있으며, 이를 통해 불필요한 연산을 줄이고 전반적인 효율성을 향상시킬 수 있습니다. 기존의 방법들보다 70%까지 토큰 사용을 줄이면서도 평균 2.6%의 정확도가 개선됨을 보여줍니다.

- **Technical Details**: Latent-Trajectory 신호는 세 가지 보완적 시간적 측면을 캡쳐합니다: (i) 추론 경로 시작부터 끝까지의 총 표현 변화, (ii) 중간 단계에서 누적된 변화, (iii) 중간 업데이트가 최종 상태에 얼마나 가까워지거나 멀어지는지를 측정합니다. 이러한 지표들은 숨겨진 상태(hidden states)에서 직접 작동하며, 추가적인 훈련이나 외부 주석(pointers) 없이도 추론 중에 계산 가능합니다. 연구에서는 다양한 추론 가능 대형 언어 모델(family of reasoning-enabled LLMs)을 사용하여 Latent-Trajectory 신호의 유효성을 평가했습니다.

- **Performance Highlights**: Latent-Trajectory 신호를 사용한 조기 답변 선택은 샘플 생성 과정을 70%까지 줄이는 성과를 거두었으며, 정확도는 다수결 기준보다 평균 2.6% 향상되었습니다. 이러한 신호는 종종 추론 경로 초기 단계에서 나타나 강력한 후보를 조기에 인식하고 이들에게 연산(compute)을 할당할 수 있게 도와줍니다. 연구 결과는 추론의 효율성을 높이고, 추론 과정의 해석 가능성을 제고할 수 있는 실질적인 도구를 제공합니다.



### MedCoAct: Confidence-Aware Multi-Agent Collaboration for Complete Clinical Decision (https://arxiv.org/abs/2510.10461)
- **What's New**: 자율 에이전트들이 대규모 언어 모델(LLMs)을 활용하여 진단 및 약물 결정과 같은 통합 의료 워크플로우에 어려움을 겪고 있다는 점을 강조하며, 이를 극복하기 위한 MedCoAct 시스템을 제안하였습니다. MedCoAct는 전문 의사 및 약사 에이전트를 통합하여 임상 협업을 시뮬레이션하는 다중 에이전트 프레임워크입니다. 이를 통해 약물 추천 및 진단 정확도를 각각 67.58%로 높였으며, 이는 단일 에이전트 시스템보다 각각 7.04%와 7.08% 향상된 결과입니다.

- **Technical Details**: MedCoAct는 의사와 약사 역할의 전문성을 결합하여 인공지능(AI)이 의료 워크플로우를 효과적으로 처리할 수 있도록 설계되었습니다. 이 시스템은 신뢰성 있는 반영(mechanism) 메커니즘을 포함하여 자기 평가 기능을 통해 낮은 신뢰도에서 결정을 최적화합니다. 또한, 적응형 검색 전략을 통해 특정 시나리오에 맞춰 지식 소스를 조정하여 정보 검색 품질을 개선합니다.

- **Performance Highlights**: DrugCareQA는 2,700개의 실제 의학 상담 사례를 포함하는 포괄적인 벤치마크 데이터셋으로 통합된 의료 AI 시스템의 평가를 가능하게 합니다. MedCoAct는 진단 및 약물 추천에서 각각 67.58%의 정확성을 달성하여 기존 시스템보다 더욱 효과적인 성능을 보였습니다. 이러한 개선은 의료 의사 결정에서 전문 에이전트 협업의 중요성을 입증하며, 특히 원격 진료와 일상적인 임상 상황에서 효과적입니다.



### Traj-CoA: Patient Trajectory Modeling via Chain-of-Agents for Lung Cancer Risk Prediction (https://arxiv.org/abs/2510.10454)
Comments:
          Accepted by NeurIPS 2025 GenAI4Health Workshop

- **What's New**: 이 논문은 Traj-CoA라는 새로운 다중 에이전트 시스템을 소개하여, 전자 건강 기록(EHR) 데이터를 효과적으로 처리하고, 환자 궤적 모델링을 개선하는 솔루션을 제공합니다. Traj-CoA는 작업 에이전트의 체인을 사용하여 EHR 데이터를 관리 가능한 청크로 순차적으로 처리하며, 중요한 사건을 장기 기억 모듈(EHRMem)에 기록하여 잡음을 줄이고 포괄적인 타임라인을 유지합니다. 이러한 새로운 접근 방식은 긴 EHR 입력에서 시간적 추론을 개선하고 예측 성능을 높입니다.

- **Technical Details**: Traj-CoA는 XML 형식의 통합 입력을 받아 EHR을 시간 인식 청크로 분해하여 강조된 신호를 추출하고 부분적인 잡음을 제거하는 방식으로 작동합니다. 각 청크는 특화된 작업 에이전트와 매니저 에이전트를 포함하는 다중 에이전트 작업 흐름을 통해 처리됩니다. 이러한 구조는 긴 환자 기록에 대한 복잡한 시간 추론을 가능하게 하며, 장기 기억 모듈(EHRMem)을 통해 글로벌 데이터를 관리하는 데 도움을 줍니다.

- **Performance Highlights**: Traj-CoA는 5년 EHR 데이터를 기반으로 한 1년 폐암 위험 예측 작업에서 제로 샷 설정 하에 여러 기계 학습(ML), 심층 학습(DL), 미세 조정된 BERT, 기본 LLM, RAG 기반 라인을 초과하는 성능을 보였습니다. 이 시스템은 복잡한 환자 궤적 모델링을 위한 간단하면서도 강력한 일반화 가능 솔루션이 될 가능성을 보여줍니다.



### Trace Length is a Simple Uncertainty Signal in Reasoning Models (https://arxiv.org/abs/2510.10409)
- **What's New**: 이 연구는 대형 언어 모델(LLM)에서의 불확실성 정량화에 관한 연구의 최근 동향을 다룹니다. 특히, reasoning trace length가 LLM의 신뢰도 추정에 효과적일 수 있음을 보여주고 있습니다. 실험을 통해 이 추정 방식이 기존의 zero-shot confidence estimator인 verbalized confidence와 비슷한 성능을 나타냄을 입증합니다. 또한, post-training이 reasoning trace 길이와 정확도 사이의 관계를 근본적으로 변화시킨다는 점에서 중요한 통찰을 제공합니다.

- **Technical Details**: 연구에서는 trace length를 신뢰도 추정의 한 방법으로 제시합니다. Reasoning post-training 이후 trace length가 의미 있는 zero-shot confidence estimate로 작용하며, 이는 베이스 모델에서는 나타나지 않는 신호입니다. 또한, trace length와 verbalized confidence 간의 관계를 분석하고, 두 신뢰도 측정 방법을 조합했을 때 더욱 뛰어난 결과를 보여줄 수 있음을 확인했습니다.

- **Performance Highlights**: 연구 결과, reasoning post-training이 진행된 모델에서 trace length는 높은 신뢰도 지표로 작용하며, 이를 통해 불확실성을 보다 잘 정량화할 수 있는 것으로 나타났습니다. 실험을 통해 trace length가 신뢰도 신호로서 신뢰성 있는 것임을 발견하였고, fork tokens와 같은 특정 및 고-엔트로피 토큰들이 중요한 역할을 한다고 확인했습니다. 이러한 사실들은 LLM의 불확실성을 이해하는 데 있어 중요한 기초 자료로 작용할 것으로 기대됩니다.



### Beyond Ethics: How Inclusive Innovation Drives Economic Returns in Medical AI (https://arxiv.org/abs/2510.10338)
- **What's New**: 이 논문은 의료 인공지능(AI)에서 공정성을 위한 윤리적 논거는 잘 확립되었으나, 포괄적 디자인의 경제적 및 전략적 가치가 여전히 탐구되지 않았음을 강조합니다. 저자들은 다양한 제약 사용 사례를 위해 설계된 솔루션이 더 넓은 시장에서 우수한 경제적 수익을 창출하는 'inclusive innovation dividend'(포괄적 혁신 배당금)라는 역설적 원리를 제시합니다. 이 연구는 포괄적 의료 AI 개발이 규정 준수를 넘어 비즈니스 가치를 창출할 수 있음을 보여줍니다.

- **Technical Details**: 저자들은 포괄적 혁신이 수익을 창출하는 네 가지 메커니즘을 제시합니다: (1) 지리적 확장 및 신뢰 증진을 통한 시장 확대, (2) 개선된 통합에 따른 리스크 완화, (3) 향상된 일반화와 기술 부채 감소로 인한 성능 배당금, 그리고 (4) 인재 확보 및 임상 채택에서의 경쟁 우위입니다. 이 논문은 AI 투자의 잠재력을 평가할 수 있는 Healthcare AI Inclusive Innovation Framework (HAIIF)를 제안하며, 자원 배분에 대한 구조화된 지침을 제공합니다.

- **Performance Highlights**: 포괄적 디자인으로 투자하는 조직들은 시장 접근성을 넓히고 지속 가능한 경쟁 우위를 실현할 수 있습니다. 그러나 이러한 고려를 비용으로 간주하는 조직은 네트워크 효과와 데이터 우위가 빠르게 쌓이는 상황에서 복합적인 불이익에 직면하게 될 것입니다. 본 연구는 의료 AI의 공정성이 회수 가능한 경제적 수익을 창출할 수 있는 다양한 경로를 밝혀내는데 기여합니다.



### LLM-Friendly Knowledge Representation for Customer Suppor (https://arxiv.org/abs/2510.10331)
- **What's New**: 이 논문에서는 Airbnb의 고객 지원 작업의 복잡성을 해결하기 위해 대형 언어 모델(LLMs)과 통합된 실용적인 접근 방식을 제안합니다. 이 방법론은 정책과 작업 흐름을 LLM이 이해하기 쉬운 구조로 변환하는 Intent, Context, and Action (ICA) 포맷이라는 새로운 재구성이 기법을 사용합니다. 또한, 최소한의 인간 개입으로 훈련 데이터를 생성하는 합성 데이터 생성 전략을 개발하였으며, 이는 모델의 비용 효율적인 미세 조정을 가능하게 합니다.

- **Technical Details**: 고객 지원 작업에서 LLM의 해석 및 추론 정확도를 향상시키기 위해 ICA 포맷을 사용하여 비즈니스 지식을 단순화하고 구조화합니다. 우리의 내부 실험에서는 재구성된 작업 흐름과 합성 데이터로 미세 조정된 LLM이 고객 지원에서 성능이 크게 향상된다는 것을 보여주었습니다. 이 솔루션은 비록 탐색적 목적의 솔루션이나, 다양한 비즈니스 도메인에서 AI 에이전트를 개발하는 데 도움이 될 수 있습니다.

- **Performance Highlights**: 고객 지원의 성과 지표에 의하면, 우리가 제안한 방법은 정확성과 수작업 처리 시간 모두에서 개선을 가져오는 것으로 나타났습니다. LLM의 성능이 향상되고, 더 나은 자원 배분 덕분에 에이전트의 생산성과 고객 만족도가 높아집니다. 이 연구는 고객 지원 생산성을 개선하기 위한 LLM의 실용성을 보여주는 새로운 기준을 설정합니다.



### Mitigating Hallucination in Multimodal Reasoning via Functional Attention Contro (https://arxiv.org/abs/2510.10285)
Comments:
          preprint

- **What's New**: 이번 연구에서는 다중 모달 대규모 추론 모델(MLRM)의 환각(hallucination) 문제를 연구하고, 인식에 집중하는 얕은 헤드와 상징적 추론으로 전이하는 깊은 헤드 간의 단계적 분화를 관찰했습니다. 이를 통해 MLRM에서 발생하는 환각의 주요 원인인 지각 편향(perceptual bias)과 추론 드리프트(reasoning drift)가 밝혀졌습니다. 연구팀은 이러한 문제를 해결하기 위해 경량화되고 해석 가능한 두 단계 플러그인, 기능 헤드 식별(Functional Head Identification) 및 클래스 조건 조정(Class-conditioned Rescaling)을 제안합니다.

- **Technical Details**: 기능 헤드 식별 과정에서는 인식 또는 추론에 특화된 헤드를 구분하고, 이들을 명시적으로 활용할 수 있도록 합니다. 이를 위해 주의 가중치를 시각화하고, 모달리티 특정 주의 비율(attention ratios)을 계산하여 인식 지향 또는 추론 지향 그룹으로 분류합니다. 클래스 조건 조정 과정에서는 이러한 기능 헤드의 기여를 증대시켜 지각 편향과 추론 드리프트를 방지하며, 기본적인 주의 메커니즘을 변경하지 않습니다.

- **Performance Highlights**: 연구팀은 Kimi-VL, Ocean-R1, R1-Onevision이라는 세 가지 실세계 MLRM에 대해 실험을 진행했습니다. 평균적으로 8% 성능 향상을 이루었으며, 가장 도전적인 작업에서는 20%의 향상도 기록했습니다. 이 접근 방식은 재훈련 없이 플러그 앤 플레이 형태로 구현될 수 있으며, 최소 4%의 추가 계산 및 7%의 지연 시간으로 최신 성능을 달성했습니다.



### The Achilles' Heel of LLMs: How Altering a Handful of Neurons Can Cripple Language Abilities (https://arxiv.org/abs/2510.10238)
- **What's New**: 대규모 언어 모델(LLMs)이 자연어 처리에서 중요한 도구로 자리잡고 있으며, 이들은 인간의 뇌와 유사한 구조적 특성을 공유한다는 점을 밝혀냈습니다. 연구진은 LLMs에 꼭 필요한 '중요한 뉴런'을 식별하기 위한 새로운 방법론을 제안하고, 이를 통해 뜻밖의 발견들을 보고하였습니다. 이 발견은 LLM의 안정성과 해석 가능성을 향상시키는 데 중요한 기여를 할 수 있습니다.

- **Technical Details**: 본 논문은 'Perturbation-based Causal Identification of Critical Neurons'라는 새로운 방법론을 통해 LLM의 중요한 뉴런을 체계적으로 탐색하는 접근을 취하고 있습니다. 이 방법은 두 단계로 구성되며, 유입되는 텍스트에 제어된 노이즈를 주입하여 뉴런의 활성화 차이를 측정함으로써, 가장 영향을 미치는 뉴런을 순차적으로 마스킹하고 그 효과를 관찰합니다. 이 방법론을 다양한 LLM 아키텍처와 데이터셋에 적용하여 유의미한 패턴을 발견했습니다.

- **Performance Highlights**: 주요 발견으로는 LLM이 초희소적인 중요한 뉴런 집합에 의해 제어된다는 것입니다. 실험에 따르면, 단 3개의 뉴런을 비활성화하는 것만으로도 72B 파라미터 모델의 성능이 급격히 저하될 수 있으며, 퍼플렉시티(perplexity)가 20오더만큼 증가할 수 있습니다. 또한, 이러한 중요한 뉴런은 외부 층에 집중되어 있으며, 특히 MLP down_proj 구성 요소 내에 많이 분포해 있다는 점이 확인되었습니다. 성능 저하는 점진적이지 않고 급격한 단계 전환을 통해 발생합니다.



### Adaptive Dual Reasoner: Large Reasoning Models Can Think Efficiently by Hybrid Reasoning (https://arxiv.org/abs/2510.10207)
Comments:
          Accepted to NeurIPS 2025 Workshop on Efficient Reasoning

- **What's New**: 이번 연구에서는 Adaptive Dual Reasoner (ADR)를 제안하여 Long Reasoning Models (LRMs)의 과도한 사고로 인한 문제를 해결합니다. ADR은 빠른 사고(fast thinking)와 느린 사고(slow thinking) 두 가지 추론 모드를 지원하며, 문맥 복잡성에 따라 동적으로 이 모드들을 전환합니다. 이는 다양한 추론 시나리오에서 최적의 성능과 효율성을 달성하기 위해 설계되었습니다.

- **Technical Details**: ADR의 훈련은 두 단계로 구성됩니다. 첫 번째 단계에서는 감독형 미세 조정(supervised fine-tuning, SFT)을 통해 모델이 두 가지 추론 모드를 통합할 수 있도록 훈련하고, 두 번째 단계에서는 엔트로피 가이드 하이브리드 정책 최적화(Entropy-guided Hybrid Policy Optimization, EHPO)를 도입하여 추론 노력을 최적화합니다. 이를 통해 ADR은 효율성과 정확성을 조화롭게 유지하며, 다양한 문제에 대해 자동으로 추론 방식을 조절할 수 있습니다.

- **Performance Highlights**: ADR은 복잡한 수학적 추론 벤치마크에서 기존의 최첨단 기법들보다 최대 6.1%의 성능 향상을 이루었으며, 추론 출력 길이는 49.5%에서 59.3%까지 단축되었습니다. 이는 ADR이 추론 모드를 동적으로 전환하는 기능 덕분에 가능해졌으며, 기존의 접근 방식에서 발생할 수 있는 불필요한 과도한 사고를 줄이는 데 효과적입니다.



### PIXEL: Adaptive Steering Via Position-wise Injection with eXact Estimated Levels under Subspace Calibration (https://arxiv.org/abs/2510.10205)
Comments:
          18 pages,3 figures

- **What's New**: 본 논문은 대형 언어 모델(LLM)의 신뢰성 있는 행동 제어를 위한 새로운 방법으로 Position-wise Injection with eXact Estimated Levels (PIXEL) 프레임워크를 제안합니다. PIXEL은 기존의 비효율적인 방법들 대신, 중립적인 기하학적 최적화를 통해 속성 정렬을 위한 최소 개입을 수행합니다. 이를 통해 모델이 다양한 태스크에서 높은 성과를 유지하면서도 신뢰할 수 있는 출력을 생성할 수 있도록 합니다.

- **Technical Details**: PIXEL 프레임워크는 두 개의 뷰(tail-averaged와 end-token)를 조합하여 속성 정렬 서브스페이스를 학습하고, 레이어 및 토큰에 기반한 정밀한 개입 수준을 선택합니다. 또한, 샘플 수준의 직교 잔여 보정을 통해 전반적인 속성 방향을 세밀하게 조정하며, 포지션 스캐닝을 통해 최적의 개입 지점을 식별합니다. 이런 방법은 LLM의 성능을 최대한 유지하면서, 개입으로 인한 불필요한 손실을 최소화합니다.

- **Performance Highlights**: PIXEL은 Llama3-8B-Instruct, Qwen2-7B-Instruct, Mistral-7B-v0.3과 같은 다양한 모델에서 실험되었으며, 여러 평가 형식에서도 시연되었습니다. 실험 결과, PIXEL은 기존의 활성화 개입 방식에 비해 일관되게 속성 정렬을 개선하며, 모델의 일반적인 능력을 유지함을 입증하였습니다. 이러한 성과는 다양한 다중 선택 설정 및 개방형 생성에서 확인됩니다.



### Don't Just Fine-tune the Agent, Tune the Environmen (https://arxiv.org/abs/2510.10197)
- **What's New**: 이 논문에서는 새로운 학습 패러다임인 $	extbf{Environment Tuning}$을 소개하여 에이전트가 사전 수집된 전문가의 경로 없이 문제 인스턴스에서 직접 복잡한 행동을 학습할 수 있게 합니다. 이 접근법은Structured Curriculum, Actionable Environment Augmentation, Fine-Grained Progress Rewards로 구성되어 있으며, 이를 통해 데이터가 희소한 환경에서도 안정적이고 효율적인 탐색이 가능합니다.

- **Technical Details**: $	extbf{Environment Tuning}$은 세 가지 상호 보완적인 원칙에 의해 작동하여 에이전트가 점진적으로 기술을 개발하도록 안내합니다. Structured Curriculum은 에이전트를 간단한 작업에서 복잡한 작업으로 안내하며, Actionable Environment Augmentation은 실패 시 수정된 힌트를 제공하여 학습 피드백을 강화합니다. 마지막으로, Fine-Grained Progress Rewards는 희소한 이진 결과 대신 연속적인 작업 완료 측정을 제공하여 더 밀접하고 유용한 학습 신호를 생성합니다.

- **Performance Highlights**: 이 방법은 Berkeley Function-Calling Leaderboard (BFCL) 벤치마크의 400개 문제 인스턴스를 사용하여 강력한 기준 모델에 대해 경쟁력을 보여주며, 특히 일반화 성능에서 뛰어난 결과를 보입니다. 또한, 환경 조정 방법은 SFT 기반 접근에서 일반적으로 나타나는 성능 붕괴를 극복하며 Qwen2.5-7B 모델의 성과를 크게 향상시키고, ToolACE-2의 out-of-distribution 점수를 두 배 가까이 증가시키는 성과를 보였습니다.



### SAFER: Risk-Constrained Sample-then-Filter in Large Language Models (https://arxiv.org/abs/2510.10193)
- **What's New**: 이번 논문에서는 위험 제어를 위한 새로운 두 단계 프레임워크인 SAFER(Sampling and Conformalized Filtering with Abstention)를 소개합니다. SAFER는 사용자 요구에 맞는 신뢰성 있는 예측 세트를 구성하여, 대답이 고정되지 않은 자연어 질문 응답(open-ended QA)에서 정확도를 높일 수 있는 혁신적인 방법입니다. 특히, 기존의 선택적 적합 예측(selective conformal prediction, SCP) 방법의 한계를 극복하고, 불확실성(uncertainty)을 정량화하여 모델 출력의 신뢰성을 향상시킵니다.

- **Technical Details**: SAFER는 우선 사용자 지정 위험 수준(α)에서 최대 샘플링 한도 내에서 샘플링 예산을 조정하는 보정(calibration) 단계를 거칩니다. 그 후, 플로퍼-피어슨(Clopper-Pearson) 정확한 방법론을 통해 샘플링 예산 설정과 함께 해당 위험 수준 이하의 올바른 답변을 산출하도록 합니다. 이와 함께, SAFER는 불확실한 선택지를 필터링하기 위해 통계적으로 유효한 불확실성 임계값을 결정하는 적합화(conformal risk control) 방법을 적용하여 더욱 신뢰할 수 있는 후보 세트를 제공합니다.

- **Performance Highlights**: SAFER는 다양한 오픈 소스 LLM에서 3가지 오픈-엔드 QA 기준을 평가하여, 서로 다른 사용자 지정 위험 수준 아래에서 테스트 시간 위험을 제어하고, 불확실한 후보를 유효하게 필터링하는 데 성공했습니다. 실험 결과, SAFER는 제시된 위험 수준에서 예측 세트의 정확도를 유지하면서도 신뢰성 있는 답변을 보장하는 능력이 있음을 입증했습니다. 이틀 간의 조정 데이터만으로도 테스트 세트에서 리스크 제어를 효과적으로 달성할 수 있음을 보여 주고 있습니다.



### Concise Reasoning in the Lens of Lagrangian Optimization (https://arxiv.org/abs/2510.10168)
- **What's New**: 이번 연구에서는 대규모 언어 모델(Large Language Models, LLMs)의 간결한 추론(concise reasoning) 문제를 해결하기 위해 성능 인식 길이 업데이트(performance-aware length updating, PALU)라는 새로운 알고리즘을 제안합니다. PALU는 성능을 유지하면서 응답 길이를 최소화하는 방정식을 사용하여, 나아가 제약 최적화 문제로 모델을 간소화합니다. 이 방법은 다양한 도메인과 모델 스케일에 적용 가능하여 새로운 사실을 통해 업계를 개선할 잠재력을 보여줍니다.

- **Technical Details**: PALU는 과거 epoch에서 수집된 롤아웃(rollout)을 재사용하여 성능을 예측하는 Off-policy 성능 점검을 구현합니다. 또한, 량chanics를 간단하게 만들기 위해 명시적인 튜닝 없이 두 극단으로 량치를 Snap하는 방식을 채택합니다. 마지막으로, 목차 기반 업데이트 방법을 통해 모델의 학습 목표를 다루며, 이로써 길이 예산의 조정이 가능합니다.

- **Performance Highlights**: PALU는 DeepSeek-Distill-Qwen-1.5B 모델에 적용했을 때, 출력 길이를 65% 줄이면서 정확도를 15% 향상시켰습니다. 또한, PALU는 논리(logic), STEM 및 수학(math) 등 다양한 도메인에서 효과적으로 적용 가능하며, 1.5B에서 14B 파라미터까지 스케일링이 가능합니다. 이러한 성능 향상은 PALU가 단순한 길이 축소를 넘어서 성능을 유지하는 데 기여함을 보여줍니다.



### CharCom: Composable Identity Control for Multi-Character Story Illustration (https://arxiv.org/abs/2510.10135)
Comments:
          Accepted by ACM MMAsia 2025

- **What's New**: 이번 연구에서는 CharCom이라는 경량화된 프레임워크를 제안하여 diffusion 모델을 사용하여 작화된 스토리 일러스트레이션에서 캐릭터 정체성을 일관적으로 유지하도록 합니다. CharCom은 LoRA (Low-Rank Adaptation) 어댑터를 통해 캐릭터 개인의 맞춤화를 가능하게 하며, 다양한 프롬프트에서도 일관된 캐릭터 표현을 보장합니다. 이는 기존의 모델에서 자주 발생하던 캐릭터 정체성의 변동 문제를 해결하는 데 크게 기여합니다.

- **Technical Details**: CharCom은 고정된 diffusion 백본 위에서 작동하며, 프롬프트에 민감한 제어를 통해 어댑터를 동적으로 구성합니다. 각 캐릭터는 독립적으로 학습된 순위-4 정체성 어댑터로 표현되며, 이들 어댑터는 기본 모델을 수정하거나 재훈련하지 않고도 생성 시 결합할 수 있습니다. 이를 통해 적은 훈련 데이터로 다수의 캐릭터가 있는 장면을 생성할 수 있으며, 내러티브의 일관성을 증가시키기 위한 구조화된 프롬프트 템플릿도 구성되었습니다.

- **Performance Highlights**: CharCom은 여러 장면에서의 실험을 통해 캐릭터 충실도 및 의미적 정렬과 시간적 일관성을 크게 향상시켰습니다. 특히 혼잡한 장면에서도 강한 견고성을 유지하며, 실질적인 응용이 가능한 스토리 일러스트레이션 및 애니메이션 제작에 적합한 솔루션이 됩니다. 이 연구는 GPT-4o를 기반으로 한 캐릭터 인지 평가 프레임워크를 도입하여 기존의 평가 방법보다 더 인지적이고 인간 중심적인 평가를 가능하게 합니다.



### DixitWorld: Evaluating Multimodal Abductive Reasoning in Vision-Language Models with Multi-Agent Dixit Gameplay (https://arxiv.org/abs/2510.10117)
Comments:
          EMNLP 2025 Wordplay (Spotlight)

- **What's New**: 본 논문은 인간 지능의 핵심인 다중 모드 유식 추론(multimodal abductive reasoning)을 평가할 새로운 툴인 DixitWorld를 소개합니다. DixitWorld는 동적, 다중 에이전트 환경에서 작업을 수행하도록 설계된 DixitArena와 정적 QA 벤치마크 DixitBench로 구성됩니다. 이 연구를 통해 VLMs가 다중 에이전트 상황에서 어떻게 다르게 행동하는지를 탐구할 수 있는 기회를 제공합니다.

- **Technical Details**: DixitArena는 스토리텔러(storyteller)가 이미지를 기반으로 암호 같은 단서를 생성하고, 리스너(listener)가 그 단서를 바탕으로 시각적 가설을 선택하는 다중 에이전트 환경입니다. DixitBench는 리스너 역할을 분리해 하이포세스(hypothesis) 선택을 다루는 선택형 QA 문제로 구성되어 있으며, 점수는 모델의 성능을 효과적으로 평가하는 데 도움을 줍니다. 이를 통해 모델의 성과를 STORYTELLER SCORE, LISTENER ACCURACY 및 OVERALL SCORE로 평가합니다.

- **Performance Highlights**: DixitArena의 결과는 소규모 오픈 소스 모델이 창의적인 스토리텔러로서 우수한 성과를 내지만, 큰 비공식 모델은 리스너로서 전체 성능이 우수함을 보여줍니다. DixitBench에서의 성과는 리스너 성과와 강력한 상관관계를 보이며, 이것이 하이포세스 선택에 대한 신뢰할 수 있는 지표로 작용함을 입증합니다. 연구 결과는 생성적 창의성(generative creativity)과 차별적 이해(discriminative understanding) 간의 주요 거래를 시사하며, 이는 VLM 개발에 있어 중요한 과제로 남습니다.



### Agentic Troubleshooting Guide Automation for Incident Managemen (https://arxiv.org/abs/2510.10074)
- **What's New**: 이번 연구에서는 대규모 IT 시스템에서의 사고 관리(Incident Management, IcM)를 위한 새로운 자동화 프레임워크인 StepFly를 소개합니다. StepFly는 민첩성을 높이고 효율성을 증대시키기 위해 문제 해결 가이드(Troubleshooting Guides, TSG)의 품질 문제를 해결하고 복잡한 제어 흐름을 해석하는 것을 목표로 합니다. 이 프레임워크는 오프라인 전처리와 온라인 실행을 포함하여, LLM(대규모 언어 모델)을 활용해 TSG를 자동으로 변환하고 실행하는 체계를 마련했습니다.

- **Technical Details**: StepFly는 세 가지 단계로 구성됩니다. 첫 번째 단계에서는 TSG 품질 개선을 위한 도구인 TSG Mentor를 제공하여 사용자에게 명확한 안내를 제공합니다. 두 번째 단계에서는 LLM을 사용해 비구조화된 TSG에서 구조화된 실행 DAG(Directed Acyclic Graph)를 추출하고, 전용 쿼리 준비 플러그인(Query Preparation Plugins, QPPs)을 생성하여 쿼리의 일관성을 높이고 오류를 줄입니다. 마지막으로, 제어 흐름에 따라 DAG 기반의 스케줄러-실행기 구조를 통해 온라인 실행을 수행합니다.

- **Performance Highlights**: StepFly는 실제 TSG 및 사건 분석을 통해 약 94%의 높은 성공률을 기록하며, 이전의 방법들보다 실행 시간과 토큰 소모를 대폭 줄인 것으로 나타났습니다. 특히, 병렬화가 가능한 TSG의 경우 실행 시간을 32.9%에서 70.4%까지 감소시키며 효율성을 극대화했습니다. 이러한 성과는 LLM의 강력한 지원을 통해 이루어진 것입니다.



### SyncLipMAE: Contrastive Masked Pretraining for Audio-Visual Talking-Face Representation (https://arxiv.org/abs/2510.10069)
- **What's New**: SyncLipMAE는 챗봇 얼굴 비디오를 위한 자기 지도(pretraining) 프레임워크로, 라벨이 없는 오디오-비주얼 데이터에서 동기화와 전이 가능한 얼굴 동작을 학습합니다. 이 방법은 마스크 비주얼 모델링(masked visual modeling)과 교차 모달 대비 정렬(cross-modal contrastive alignment)을 결합하고, 각 프레임에 대해 세 가지 프롬프트 토큰(identity, vocal motion, ambient motion)을 사용합니다. 이러한 접근 방식은 얼굴 비디오의 중요한 특성을 명확하게 인코딩하고, 서로 다른 다운스트림 작업에서 사용할 수 있는 통합된 인터페이스를 제공합니다.

- **Technical Details**: SyncLipMAE는 비디오의 각 프레임에서 동기화된 시각 및 오디오 토큰을 비교하며, 이를 통해 공유된 임베딩 공간(shared embedding space)을 만듭니다. 프레임당 세 가지 주요 프롬프트 토큰(정체성, 발음 동작, 주변 동작)을 사용하고, 이러한 프롬프트는 교차 주의(cross-attention)를 바탕으로 프레임을 재구성하여 동기화를 촉진합니다. 특히, 이 방법은 두 가지 마스킹 전략(균일 마스킹과 얼굴 유지 마스킹)을 적용하여 안정적인 조정을 제공합니다.

- **Performance Highlights**: SyncLipMAE는 오디오-비주얼 스트림 동기화, 얼굴 감정 및 행동 인식, 시각적 말 인식, 시각적 더빙 등 네 가지 고유 작업에서 최첨단 성능을 달성했습니다. 이 프레임워크는 각 작업에 있는 다양한 기능을 필요로 하면서도 단일 모델 내에서 오디오 또는 비디오에서 무차별적으로 제어할 수 있는 기능을 제공하여 이전의 모든 연구 결과를 능가합니다. 이러한 결과들은 동기화 인식을 고려한 자기 지도 학습(pretraining)의 효과성을 강조합니다.



### SwarmSys: Decentralized Swarm-Inspired Agents for Scalable and Adaptive Reasoning (https://arxiv.org/abs/2510.10047)
Comments:
          14 pages, 7 figures

- **What's New**: SwarmSys는 분산 다중 에이전트 추론을 위한 클로즈드 루프 프레임워크로, 군집 지능에 영감을 받아 설계되었습니다. 기존 시스템의 고정된 역할이나 중앙 집중식 제어에서 벗어나 지속적인 탐색, 활용, 유효성 검증을 통해 협업을 지원합니다. 이 새로운 접근 방식은 에이전트 간의 적응적 협력과 동적인 작업 할당을 가능하게 하여, 다중 에이전트 시스템의 확장성과 적응성을 향상시킵니다.

- **Technical Details**: SwarmSys는 탐색자(Explorers), 작업자(Workers), 검증자(Validators)의 세 가지 전문화된 역할 간의 반복적인 상호작용을 통해 협력을 조정합니다. 각 에이전트는 능력 및 가용성을 나타내는 두 개의 임베딩(embedding)을 사용하여 매칭합니다. 이러한 동적 프로필 업데이트는 자율적인 협업과 안정적인 장기 추론을 가능하게 하는 중요한 메커니즘입니다.

- **Performance Highlights**: SwarmSys는 상징적 추론, 연구 종합, 과학적 프로그래밍 작업에서 기준 계열에 비해 일관되게 더 우수한 성능을 보이며 정확도와 추론 안정성을 개선했습니다. 예를 들어, GPT-4o 기반의 에이전트 스웜은 GPT-5 성능에 근접하는 결과를 보여, 조정의 스케일링이 모델의 스케일링을 대체할 수 있음을 나타냅니다. 이러한 성능 향상은 집단 지능의 고유한 특징을 드러냅니다.



### Belief Graphs with Reasoning Zones: Structure, Dynamics, and Epistemic Activation (https://arxiv.org/abs/2510.10042)
- **What's New**: 이 논문은 신뢰도(credibility)와 자신감(confidence)을 구분하는 새로운 그래프 이론적 프레임워크를 제안합니다. 믿음은 방향성, 부호, 가중치가 있는 그래프의 노드로 표현되며, 엣지는 지지(support)와 모순(contradiction)을 인코딩합니다. 이러한 모델은 고전적인 추론이 안전하게 적용될 수 있는 고신뢰도 서브그래프인 추론 영역(reasoning zones)을 정의하는 방법을 제공하며, 이를 통해 모순이 존재하는 경우에도 효과적인 추론이 가능함을 보여줍니다.

- **Technical Details**: 이 프레임워크는 신뢰도를 외부에서 할당된 신뢰와, 너비가 있는 그라프 구조에 의해 발생하는 내재적 가치인 자신감으로 분리합니다. 신뢰도는 엣지의 가중치에 따라 정량화되며, 자신감은 주어진 그래프의 영향을 통해 계산됩니다. 또한, 전체 그래프를 불안정하게 만들지 않고도 서브그래프가 동적으로 재구성될 수 있도록 하는 고유한 업데이트 모델을 포함하고 있습니다.

- **Performance Highlights**: 수치 실험에서는 계획된 영역(planted zones)에서의 영역 복구(zone recovery), 충격(shock) 발생 시 안정성(stability), 그리고 런타임(runtime) 측면에서 결과를 보고합니다. 이 연구는 모순을 견딜 수 있는 추론을 위한 원리적 기반을 제공하여 구조적 지원이 있는 경우에만 고전 논리에 의존할 수 있도록 합니다. 나아가 이 프레임워크는 기존의 추론 아키텍처와 호환되도록 설계되었습니다.



### Failure-Driven Workflow Refinemen (https://arxiv.org/abs/2510.10035)
- **What's New**: 이번 논문은 기존 LLM 기반 워크플로우 최적화의 문제점을 정보 붕괴(information collapse)라는 관점에서 재정립하고, 실패 분포를 모델링하는 새로운 패러다임을 제안합니다. 우리는 워크플로우의 성공률을 단순히 극대화하는 것이 아니라, 기대 실패 질량(Expected Failure Mass)을 최소화하는 방향으로 최적화 목표를 설정했습니다. 이를 통해 보다 구조적으로 실패를 이해하고, 관련된 문제를 해결할 수 있는 기회를 제공합니다.

- **Technical Details**: 새로운 최적화 패러다임을 구현하기 위해 CE-Graph라는 프레임워크를 제안하며, 이는 실패 분포를 모델링하고 감소시키는 전략을 포함합니다. CE-Graph는 카운터예제(counterexample) 풀(pool)을 유지하며, 실패 분포의 밀집 지역을 진단하고 이를 클러스터링하여 '의미적 기울기(semantic gradient)'를 추정합니다. 이후 타겟 그래프 수정(Targeted graph edits)을 위한 Propose-and-Verify 메커니즘을 통해 빠르게 실패 질량을 감소시킵니다.

- **Performance Highlights**: CE-Graph는 수학, 코드, QA 벤치마크에서 테스트되었으며, 강력한 기존 워크플로우 검색 기법에 비해 훨씬 낮은 비용으로 높은 성공률을 기록했습니다. 이러한 성과는 시스템의 신뢰성이 단순히 실패를 회피하는 데서 오는 것이 아니라, 실패 분포의 기하학적 구조를 체계적으로 이해하고 재정립하는 과정에서 나타난다는 것을 강조합니다.



### RIPRAG: Hack a Black-box Retrieval-Augmented Generation Question-Answering System with Reinforcement Learning (https://arxiv.org/abs/2510.10008)
- **What's New**: 이 논문에서는 기존의 단순한 RAG 시스템에 대한 화이트 박스 공격보다 더 복잡하고 현실적인 상황에서 공격을 수행하게 됩니다. 저자는 RIPRAG 공격 프레임워크를 제안하며, 이는 RAG 시스템의 내부 구조에 대한 정보 없이 공격자가 독일 수 있는 유일한 정보인 독성 문서 주입의 성공 여부만 기반으로 합니다.

- **Technical Details**: RIPRAG은 강화 학습(Reinforcement Learning, RL)을 활용하여 독성 문서 생성을 최적화하며, 상호 피드백을 통해 공격 전략을 반복적으로 개선합니다. 이 방법은 공격 성공률을 극대화하는 데 초점을 맞추며, 특히 복잡한 RAG 시스템에서 효과를 보입니다.

- **Performance Highlights**: 실험 결과에 따르면, RIPRAG 방법은 기존의 방법에 비해 최대 0.72의 공격 성공률(ASR) 향상을 달성하였습니다. 이는 현재의 방어 방법이 갖는 부족함을 강조하며, 대형 언어 모델(LLM) 보안 연구에 중요한 통찰을 제공합니다.



### Deliberative Dynamics and Value Alignment in LLM Debates (https://arxiv.org/abs/2510.10002)
- **What's New**: 이 연구는 대화형 다변량(다중 턴) 설정에서 대규모 언어 모델(LLMs)의 가치를 탐구하는 새로운 접근 방식을 제시합니다. 그동안의 연구는 주로 단일 턴 프롬프트를 통한 사회기술적 정렬(sociotechnical alignment)에 초점을 맞추었으나, 본 논문은 Reddit의 'Am I the Asshole' 커뮤니티에서 수집한 1000개의 일상적인 결정을 바탕으로 다변량 환경에서의 모델의 행동 양상을 분석합니다.

- **Technical Details**: 이 연구는 GPT-4.1, Claude 3.7 Sonnet, Gemini 2.0 Flash의 세 모델이 1000개의 윤리적 딜레마에 대해 집단적으로 책임을 분배하는 방식을 비교합니다. 실험은 동시 응답(synchronous) 형식과 원주율 응답(round-robin) 형식으로 나뉘어 모델 간의 의사결정 다이내믹을 분석합니다. 이를 통해 각 모델의 유연성과 가치 지향성을 평가했습니다.

- **Performance Highlights**: 결과적으로, GPT는 강한 관성을 보였고(수정 비율 0.6-3.1%) Claude와 Gemini는 훨씬 더 유연한 반응을 보였습니다(28-41%). 또한, 각 모델의 가치 패턴도 다르게 나타났으며, GPT는 개인 자율성과 직접적인 소통을 강조하는 반면, Claude와 Gemini는 공감 대화를 우선시했습니다. 이 연구 결과는 다변량 상호작용에서 도구의 설계가 사회기술적 정렬에 중대한 영향을 미친다는 것을 보여줍니다.



### Follow My Lead: Logical Fallacy Classification with Knowledge-Augmented LLMs (https://arxiv.org/abs/2510.09970)
Comments:
          Accepted as a poster at the Twelfth Annual Conference on Advances in Cognitive Systems. 21 pages, 7 figures and 1 table

- **What's New**: 이 논문은 대형 언어 모델(Large Language Models, LLMs)의 결정적 추론 격차를 다루고 있습니다. LLMs는 일반적으로 빠르고 직관적인 System 1 처리를 기반으로 하지만, 정확한 추론을 위해서는 System 2 접근이 필요합니다. 저자들은 비용 효율적인 지침 기반 개입 방법론을 제안하여, 논리적 오류를 분류하는 과정을 일련의 간단한 단계로 나누는 혁신적인 방법(Atomic Instruction Dataset)을 소개합니다.

- **Technical Details**: 연구에서는 FALLACIES 데이터셋을 기반으로 복잡한 논리적 오류 설명을 단순한 이진 결정 단계로 분해하는 시스템적인 접근 방식을 사용합니다. 이 과정에서 Prolog 기반의 관계형 그래프를 통합하여 서로 관련된 논리적 오류들 간의 연결을 모델링합니다. 이를 통해 LLM들이 최종 분류를 내리기 전에 상호 연결된 오류를 체계적으로 검토하도록 유도하여 더욱 정확한 결정을 내릴 수 있게 합니다.

- **Performance Highlights**: 다양한 최첨단 언어 모델에 대해 포괄적인 양적 및 질적 평가를 수행한 결과, Claude-Sonnet-4 모델의 성능이 20.7% 개선되었고, ChatGPT-4o와 ChatGPT-4o-mini 모두 각각 3.4% 향상되었습니다. Gemini-2.5-Flash 모델 또한 8.7%의 성능 향상을 보였습니다. 이러한 결과는 지침 기반의 사전 규칙이 LLMs의 논리적 오류 식별 성능을 크게 향상시켰음을 보여줍니다.



### The Personalization Trap: How User Memory Alters Emotional Reasoning in LLMs (https://arxiv.org/abs/2510.09905)
Comments:
          12 pages 5 figures

- **What's New**: 이번 연구는 LLM(대형 언어 모델)이 사용자 메모리를 통해 감정 이해에 미치는 영향을 조사합니다. 동일한 시나리오가 서로 다른 사용자 프로파일과 함께 제공될 때, LLM의 감정 해석이 어떻게 다르게 나타나는지를 연구하여, 특정 프로파일이 더 정확한 감정 해석을 받는 경향이 있음을 발견했습니다. 이러한 결과는 개인화 AI 시스템이 사회적 불평등을 재생산할 위험이 있음을 시사합니다.

- **Technical Details**: 연구진은 사용자 지식을 다양한 프로파일을 이용하여 생성하고, LLM의 감정 이해와 행동 조언을 평가하기 위해 STEU(상황 감정 이해 테스트)와 STEM(감정 관리 상황 테스트)를 사용했습니다. 각 프로파일은 Bourdieu의 사회 자본 이론을 기반으로 하여 경제적, 문화적, 사회적 요소들을 포함하도록 제작되었으며, 정규화된 정서적 지능 테스트를 통해 LLM의 성능을 분석하였습니다.

- **Performance Highlights**: 실험 결과에 따르면 대부분의 LLM은 사용자 메모리를 도입했을 때 성능이 감소했으며, 특히 우세한 배경을 가진 사용자 프로파일에 대해 더 나은 성과를 보였습니다. 또한, 인종, 성별, 연령에 따른 편향이 존재하며, 이는 감정 관련 조언에서도 지속적으로 나타났습니다. 이 연구는 AI의 개인화 시스템이 사회적 계층 구조를 반영할 수 있다는 중요한 문제를 드러냈습니다.



### Autonomous Agents for Scientific Discovery: Orchestrating Scientists, Language, Code, and Physics (https://arxiv.org/abs/2510.09901)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)을 기반으로 한 과학적 에이전트의 새로운 비전과 역할을 제시합니다. 이러한 에이전트는 가설 발견, 실험 설계 및 실행, 결과 분석 및 수정까지 과학적 발견 과정의 모든 단계에서 혁신적인 접근 방식을 제공합니다. LLM을 활용한 과학적 에이전트는 독립적인 시스템으로 작동하며, 이전의 인력 중심 접근 방식에 비해 더 빠르고 효율적인 방식으로 연구 진전에 기여할 수 있는 잠재력을 가지고 있습니다.

- **Technical Details**: 과학적 발견 과정은 크게 세 가지 단계인 가설 발견, 실험 설계 및 실행, 결과 분석 및 개선으로 나누어집니다. 가설 발견 단계에서는 방대한 데이터와 기존 지식으로부터 새로운 가설을 식별하고 생성합니다. 실험 설계 단계에서는 실험을 체계적으로 수행하기 위한 계획을 세우고, 실행 단계에서는 이렇게 설계된 계획이 실제 행동으로 변환됩니다. 그 후, 결과 분석 및 개선 단계에서는 실험 결과를 해석하고 의미 있는 과학적 통찰을 도출하는 작업이 진행됩니다.

- **Performance Highlights**: LLMs 기반 에이전트는 과학적 발견 주기의 모든 단계에서 높은 유연성과 적응성을 보여줍니다. 이들은 인간의 직관이나 실험 기술에 의존하지 않고도 복잡한 데이터 상호작용을 처리할 수 있는 능력을 지니고 있습니다. 특히, 기존의 계산적 접근 방식과 비교했을 때, 이들 에이전트는 정확성과 일관성을 높이는 데 중요한 역할을 할 수 있으며, 다양한 연구 분야에서 새로운 발견의 가능성을 열어줍니다.



### Beyond AlphaEarth: Toward Human-Centered Spatial Representation via POI-Guided Contrastive Learning (https://arxiv.org/abs/2510.09894)
- **What's New**: 이번 연구에서는 AlphaEarth Foundation (AE)을 기반으로 한 도시 분석을 목적으로 한 AETHER (AlphaEarth-POI Enriched Representation Learning) 프레임워크를 제안합니다. AETHER는 Points of Interest (POIs)를 통해 인간 중심의 도시 분석에 맞게 AE 임베딩을 조정하며, 물리적 데이터와 사회경제적 맥락을 결합합니다. 이를 통해 AE의 기존 한계를 극복하고 인간의 활동을 반영한 공간 표현을 학습할 수 있습니다.

- **Technical Details**: AETHER는 POI 주변의 공간 버퍼 내에서 AE의 64차원 임베딩을 집계하고, 설계된 다중 스케일 헤드를 통해 POI 텍스트 임베딩과 정합합니다. 이를 위해 InfoNCE 기반의 대조적 목표를 사용하며, 결과적으로 지역 수준의 기능을 집계하여 후속 응용 프로그램에 활용할 수 있습니다. AETHER는 수행성과 효율성을 고려하여 경량화된 멀티모달 정합을 기반으로 합니다.

- **Performance Highlights**: Greater London에서 수행된 실험 결과, AETHER는 AE 및 POI 전용 벤치마크를 초과하여 지속적으로 개선된 성과를 보여주었습니다. AETHER는 땅 이용 분류 메트릭에서 7.2%의 상대적 향상과 사회경제적 매핑에서 23.6%의 KL 다이버전스 감소를 기록하며, 이는 멀티모달 정합이 EO 기반의 표현의 기능적 해석 가능성을 크게 향상시킨다는 것을 의미합니다.



### AI and Consciousness (https://arxiv.org/abs/2510.09858)
- **What's New**: 이 논문은 AI 의식(AI consciousness)에 대한 기존 문헌에 대한 회의적인 개요를 제공합니다. 현재 우리가 구축하는 AI 시스템은 몇몇 영향력 있는 주류 의식 이론에 따르면 의식이 있을 수 있지만, 다른 주류 이론에 비추어보면 의식이 없을 수 있습니다. 이러한 이론들이 올바른 것인지, AI가 인간 만큼이나 의미 있고 풍부한 의식을 가지고 있을지 또는 단순히 토스터처럼 경험적으로 텅 비어 있는 시스템인지 판단할 수 없는 상황에 놓이게 될 것입니다.

- **Technical Details**: 이 논문은 의식(consciousness)과 AI의 정의를 시작으로 여러 장을 통해 AI 의식의 여러 특징들을 탐구합니다. 특히, 의식의 기본적 특성에 대한 정황적(인식적) 및 개념적 논증에 대한 비판이 포함되어 있으며, 물질주의(materialism)와 기능주의(functionalism)의 관점도 논의됩니다. 또한 튜링 테스트(Turing Test)와 중국 방(B) 논증, 모방 논증(mimicry argument) 등을 통해 AI 의식의 근본적인 한계를 조사합니다.

- **Performance Highlights**: 논문은 AI 플랫폼들이 어떠한 의식적 경험도 갖고 있지 않을 수 있다는 점에서 한계를 지니고 있음을 강조합니다. 소재적(substratum) 기준이 AI의 의식과 관련이 있는지, 다양한 지능의 문제(strange intelligence)에 대해 논의하며, 생물학적 기초가 AI 의식에 얼마나 중요할 수 있는지를 탐구합니다. 궁극적으로, AI와 인간의 의식 사이의 경계를 명확히 하는 데에는 여러 복잡한 문제가 존재한다는 점을 지적합니다.



### How can we assess human-agent interactions? Case studies in software agent design (https://arxiv.org/abs/2510.09801)
- **What's New**: 본 논문에서는 LLM 기반 에이전트의 인간-에이전트 상호작용을 엄격히 평가하기 위한 두 가지 주요 단계를 제안합니다. 첫째, PULSE라는 프레임워크를 통해 사용자의 피드백을 수집하고, 이를 바탕으로 ML 모델을 훈련하여 사용자 만족도를 예측합니다. 둘째, OpenHands라는 오픈소스 소프트웨어 에이전트를 통해 대규모 웹 플랫폼에서 실제 사용자 데이터를 수집하여 에이전트 설계의 다양한 측면이 사용자 만족도에 미치는 영향을 분석합니다.

- **Technical Details**: PULSE 프레임워크는 세 단계로 구성됩니다: (1) 사용자 피드백 데이터 수집, (2) ML 모델 훈련을 통한 사용자 만족도 예측, (3) 테스트 결과 및 신뢰 구간을 계산하는 과정입니다. 에이전트 설계를 위해 사용자가 피드백을 제공하는 시점을 최소한으로 방해되도록 설계했습니다. 사용자가 작업 세그먼트가 완료된 후, 성과를 평가하도록 유도하여 더 세밀한 만족도 평가를 수행할 수 있습니다.

- **Performance Highlights**: PULSE 프레임워크를 통해 15,000명의 사용자가 참여한 36,000개 세션에서 다양한 사례 연구를 진행했습니다. 그 결과, 강력한 기본 모델에 투자하는 것이 사용자 만족도에 6-8%의 유의미한 변화를 가져온다는 것을 발견했습니다. 또한, 설계 변화에 따른 신뢰 구간을 평균 40% 줄인 점과 실제 사용 데이터와 벤치마크 평가 간의 불일치를 통해 인간의 피드백 통합 평가의 중요성을 강조했습니다.



### The Geometry of Reasoning: Flowing Logics in Representation Spac (https://arxiv.org/abs/2510.09782)
Comments:
          Code: this https URL

- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)이 어떻게 '사고'하는지를 탐구합니다. 새로운 기하학적 프레임워크를 제안하여 LLM의 추론을 흐름(flow)으로 모델링하며, 이는 논리가 진행되는 임베딩 경로(embedding trajectories)를 나타냅니다. 이 연구는 LLM이 표면적인 형식을 넘어 논리를 내부화하는지를 검증할 수 있는 새로운 관점을 제공합니다.

- **Technical Details**: 연구자들은 자연적 추론(propositional natural deduction)을 사용하여 의미론적 도구(semantic carriers)가 다양해도 논리 구조와 의미를 분리하여 LLM의 사고 과정을 분석합니다. 이 기하학적 관점은 위치(position), 속도(velocity), 곡률(curvature)과 같은 기하학적 양들과 추론을 연결하여, 추상화된 개념 공간(representation and concept spaces)에서의 분석을 가능하게 합니다.

- **Performance Highlights**: 이론적 프레임워크를 구현하기 위해 학습된 임베딩 대리 모델(learned representation proxies)을 사용하여 추론 흐름을 시각화하고 정량화하는 통제된 실험을 설계하였습니다. 이 연구는 LLM의 행동의 해석 가능성과 공식적인 분석을 위한 새로운 관점을 제공하며, 추론 현상을 연구하기 위한 개념적 기초와 실제 도구 역할을 합니다.



### Phys2Real: Fusing VLM Priors with Interactive Online Adaptation for Uncertainty-Aware Sim-to-Real Manipulation (https://arxiv.org/abs/2510.11689)
- **What's New**: Phys2Real은 로봇 조작 정책을 실제에서 직접 학습하는 데 드는 비용과 시간을 줄이기 위해, 비전-언어 모델(VLM)에서 추론한 물리적 매개변수를 사용자 상호작용을 통해 온라인으로 적응할 수 있는 새로운 RL 파이프라인을 제안합니다. 이 접근 방식은 고충실도 기하학적 재구성, VLM-추론 프라이어 분포, 상호작용 데이터를 통한 물리적 매개변수 추정의 세 가지 핵심 요소로 구성되어 있습니다. 이를 통해 로봇이 시뮬레이션에서 학습한 정책을 더 효과적으로 현실 세계에 적용할 수 있습니다.

- **Technical Details**: Phys2Real은 불확실성을 고려한 융합, 앙상블 기반의 불확실성 정량화, 그리고 물리정보가 반영된 디지털 트윈을 결합해 시뮬레이션에서 학습된 로봇 정책을 현실에서 적응 가능하게합니다. VLM에 의해 추론된 물리적 매개변수는 사용자 상호작용을 통해 Refinement되어 로봇이 특정 물체의 물리적 특성에 적응하도록 돕습니다. 또한, 물리적 매개변수는 VLM을 통해 직관적으로 설명 가능한 형태로 직접 조건화됩니다.

- **Performance Highlights**: Phys2Real의 효과는 T-block과 헐렁한 해머를 이용한 평면 밀기 작업에서 명백히 입증되었습니다. 예를 들어, T-block의 무게가 하단에 위치할 때 성공률은 100%에 달하며, 난이도가 높은 상단일 경우에도 57%의 성공률을 기록하였습니다. 또한 헐렁한 해머 작업에서는 15% 더 빠른 평균 작업 완료 시간을 기록하여 기존의 도메인 랜덤화 베이스라인보다 상당한 개선을 보여줍니다.



### PACEbench: A Framework for Evaluating Practical AI Cyber-Exploitation Capabilities (https://arxiv.org/abs/2510.11688)
Comments:
          Project webpage available at this https URL

- **What's New**: 본 연구는 사이버 공격( cyber offense)에서 대규모 언어 모델(LLMs)의 잠재력을 평가하기 위한 새로운 벤치마크 PACEbench를 제안합니다. 기존의 평가 기준은 현실 세계의 복잡성을 결여하고 있어 LLMs의 사이버 보안 능력을 제대로 평가하지 못했습니다. PACEbench는 특정 소프트웨어 취약점을 활용해 '플래그'를 회수하는 목표 지향적 시나리오로 이루어져 있습니다.

- **Technical Details**: PACEbench는 4개의 시나리오로 구성되어 있으며, 각각의 시나리오는 단일, 혼합, 연결 및 방어 취약점 활용을 포괄합니다. 이를 통해 에이전트는 다양한 현실 세계의 취약점을 탐지하고 공격할 수 있는 능력을 평가받습니다. PACEagent라는 새로운 에이전트를 소개, 이는 인적 침투 테스터의 작업을 모방하며 다단계 재정탐사, 분석 및 활용을 지원합니다.

- **Performance Highlights**: 일곱 개의 첨단 LLM을 대상으로 한 실험 결과, 현재 모델들은 복잡한 사이버 시나리오에서 성공률이 낮으며, 방어를 우회하는 데는 실패했습니다. 이러한 결과는 현재의 LLM들이 일반적인 사이버 공격 위협을 제기하지 않음을 의미하며, 향후 모델의 진화를 추적하기 위한 확실한 기준을 제공합니다.



### Representation-Based Exploration for Language Models: From Test-Time to Post-Training (https://arxiv.org/abs/2510.11686)
Comments:
          Website and code: this https URL

- **What's New**: 본 논문에서는 강화 학습(Reinforcement Learning, RL)이 언어 모델의 탐색을 어떻게 개선할 수 있는지를 조사합니다. 특히, 사전 학습된 언어 모델의 숨겨진 상태에서 파생된 단순하지만 원칙적인 보너스를 통해 새로운 행동을 발견하도록 모델을 유도하는 deliberate exploration에 초점을 맞추었습니다. 이 접근 방식은 단순히 기존 행동을 강화하는 것을 넘어서는 잠재력을 가지고 있습니다.

- **Technical Details**: 논문에서는 의도적인 탐색 기법을 통해 모델이 참신하고 다양한 행동을 발견하도록 장려하는 방법을 제시합니다. 이 방법은 사전 학습된 모델의 표현 기반 보너스를 활용하여 모델의 다양성과 pass@k 비율을 크게 향상시키는 것으로 나타났습니다. 연구는 추론 기간(inference-time)과 사후 훈련(post-training) 모두에서 이 기법의 효과를 보여줍니다.

- **Performance Highlights**: 결과적으로, Qwen-2.5-14b-Instruct 모델에 대한 추론 기간의 탐색이 표준 샘플링에 비해 50% 이상의 효율 개선을 가져왔으며, AIME 2024 경쟁에서 Qwen-2.5-7b-Instruct 모델은 기존 모델보다 3배 향상된 샘플 효율성을 보여주었습니다. 이러한 성과는 deliberate exploration이 새로운 행동을 발견할 수 있는 실질적인 경로가 될 수 있음을 시사합니다.



### Boundary-Guided Policy Optimization for Memory-efficient RL of Diffusion Large Language Models (https://arxiv.org/abs/2510.11683)
- **What's New**: 최근 확산 대형 언어 모델(dLLMs)은 기존의 자기회귀 모델(ARMs)에 대한 유망한 대안으로 부각되고 있으며, 다양한 언어 모델링 작업에서 경쟁력 있는 성능을 보여주고 있습니다. 그러나 기존의 연구들은 주로 dLLMs의 사전 학습과 감독 학습에 초점을 맞추고 있으며, 강화 학습(RL)을 이용한 dLLMs의 성능 개선은 여전히 도전 과제로 남아 있습니다. 본 연구에서는 Boundary-Guided Policy Optimization (BGPO)이라는 새로운 메모리 효율적인 RL 알고리즘을 제안하여, dLLMs에 대한 log-likelihood와 RL 목표의 근사를 지원합니다.

- **Technical Details**: BGPO는 ELBO 기반 목표의 하한을 최대화하도록 설계되었습니다. 이 하한은 두 가지 주요 속성을 만족하도록 만들어졌습니다: (1) 선형성(Linearity): 각 항이 단일 MC 샘플에만 의존하는 형태로 구성되어 있어, 샘플 간의 그래디언트 누적이 가능하고 메모리 사용이 일정하게 유지됩니다; (2) 동등성(Equivalence): 이 하한의 값과 그래디언트는 on-policy 훈련에서 ELBO 기반 목표의 값과 그래디언트가 같아, 원래의 RL 목표를 효과적으로 근사할 수 있게 됩니다. 이러한 특성 덕분에 BGPO는 큰 MC 샘플 크기를 채택하여 보다 정확한 RL 목표 근사가 가능해집니다.

- **Performance Highlights**: BGPO는 LLaDA-8B-Instruct 모델을 사용한 수학 문제 해결, 코드 생성 및 계획 작업에서 이전 RL 알고리즘과 비교해 상당한 성능 향상을 보여줍니다. 범위가 넓은 MC 샘플 크기를 활용함으로써 그래디언트의 편향과 분산을 효과적으로 줄여 모델 성능을 향상시키는 결과를 도출했습니다. 또한, BGPO는 샘플 크기가 증가하더라도 평균 훈련 단계 시간이 소폭만 증가하여 효율성을 유지했습니다.



### Ego-Vision World Model for Humanoid Contact Planning (https://arxiv.org/abs/2510.11682)
- **What's New**: 이번 연구에서는 인공지능 로봇이 물리적 접촉을 회피하는 것이 아니라 활용하도록 하는 사실이 강조됩니다. 기존의 옵티마이제이션 기반 플래너들은 접촉의 복잡성 때문에 어려움을 겪었으나, 제안된 프레임워크는 학습된 세계 모델과 샘플링 기반 모델 예측 제어(MPC)를 결합하여 이러한 문제를 해결합니다. 데이터에 기반하여 향상된 계획을 통해 다양한 접촉 작업에 대한 전반적인 효율성과 다중 작업 기능이 증가합니다.

- **Technical Details**: 제안된 방법은 저수준 정책과 고수준 플래너로 구성된 계층적 제어 프레임워크를 활용합니다. 고수준 플래너는 세계 모델과 가치 가이드 샘플링 MPC를 사용하여 로봇이 접촉 상황을 예측하고 계획하도록 합니다. 저수준 조정기는 로봇의 자세를 조정하는 명령을 추적하는 기능을 가지고 있으며, 두 시스템 모두가 프로프리오셉션(proprioception)과 에고 중심의 깊이 이미지를 통해 효과적으로 작동합니다.

- **Performance Highlights**: 물리적 휴머노이드에 배치한 결과, 이 시스템은 신뢰성 있는 실시간 접촉 계획을 생성할 수 있음을 보여주었습니다. 로봇은 접촉 인식 작업을 발휘하며, 벽에서의 지지, 물체 차단 등과 같은 다양한 작업을 수행할 수 있습니다. 제안된 방법은 샘플링 효율성을 개선하고, 이전의 RL(on-policy reinforcement learning) 방법들보다도 동적이고 복잡한 씬 상호작용을 처리할 수 있는 것을 입증합니다.



### Accelerated stochastic first-order method for convex optimization under heavy-tailed nois (https://arxiv.org/abs/2510.11676)
- **What's New**: 이 논문에서는 중량 편향 노이즈(heavy-tailed noise)가 있는 볼록(com convex) 복합 최적화 문제를 연구합니다. 기존의 연구들은 일반적으로 기울기 절단(gradient clipping)이나 정규화(normalization) 기술을 사용하여 이러한 노이즈를 처리했지만, 본 논문에서는 이러한 추가 수정 없이도 최적의 복잡도(optimal complexity)에 도달할 수 있음을 보여줍니다. 특히 가속화된 확률적 근사(subgradient) 방법이 매끄럽고, 약간 매끄러운, 비매끄러운 볼록 최적화 문제에 대해 우주적으로 최적의 복잡도를 달성하였습니다.

- **Technical Details**: 제안된 방법은 중량 편향 노이즈 하에서의 복합 최적화 문제에 대한 가속화된 확률적 프로필 근사 방법(Accelerated Stochastic Proximal Method)을 사용합니다. 이 방법은 기울기 절단이나 정규화를 사용하지 않고 근사 최적 솔루션을 찾아내는 것이 가능하며, 기대값과 높은 확률 모두에서 최적화에 대한 복잡도를 달성할 수 있습니다. 특히, 이 논문에서는 다양한 수학적 조건을 설명하며, 근사적 해(approximate optimal solution)를 찾기 위한 방법들이 제시됩니다.

- **Performance Highlights**: 제안된 가속화된 방법은 B개 산출치(1사분면 경계값)에서 𝑂(ϵ^{-α/(α−1)})의 기울기 복잡도를 달성하여, 중량 편향 노이즈 하에서도 상당한 성과를 보입니다. 추가적으로, 논문은 수치 실험을 통해 이론적 결과를 검증하였으며, 기존 기법들과 비교했을 시 독립적인 장점을 강조합니다. 이러한 연구는 현대 대규모 응용 프로그램의 복잡한 최적화 문제에 대한 새로운 접근 방식을 제시하고 있습니다.



### ManiAgent: An Agentic Framework for General Robotic Manipulation (https://arxiv.org/abs/2510.11660)
Comments:
          8 pages, 6 figures, conference

- **What's New**: 최근 제안된 ManiAgent는 로봇 조작을 위한 새로운 아키텍처로, 복잡한 작업 계획 및 추론 능력을 개선합니다. 이 모델은 여러 전문 에이전트가 서로 통신하며 환경 인식, 하위 작업 분해 및 행동 생성을 처리하여 복잡한 조작 시나리오를 효율적으로 다루게 합니다. 특히, 기존 VLA 모델의 데이터 부족 문제와 작업 지능의 한계를 극복하는 것을 목표로 합니다.

- **Technical Details**: ManiAgent는 세 가지 전문화된 에이전트로 구성됩니다: 인식 에이전트, 추론 에이전트 및 동작 실행 에이전트입니다. 인식 에이전트는 환경에서의 상세한 공간 정보를 추출하고, 추론 에이전트는 의도를 추론하고 하위 작업을 분해합니다. 마지막으로, 동작 실행 에이전트는 수집된 정보를 바탕으로 로봇이 실행할 수 있는 동작 시퀀스를 직접 생성합니다.

- **Performance Highlights**: 실험 결과, ManiAgent는 SimplerEnv 벤치마크에서 86.8%의 성공률을 기록하였으며, 실제 피킹 및 플레이스 작업에서 95.8%에 달하는 성공률을 보였습니다. 이처럼 높은 성과는 ManiAgent가 대규모 데이터 수집 도구로서 역할을 하게 해 주며, 최소한의 인간 개입으로도 VLA 모델의 성능을 개선할 수 있음을 보여줍니다.



### FinVet: A Collaborative Framework of RAG and External Fact-Checking Agents for Financial Misinformation Detection (https://arxiv.org/abs/2510.11654)
- **What's New**: 본 논문에서는 FinVet이라는 새로운 멀티 에이전트 프레임워크를 소개하며, 이는 두 개의 Retrieval-Augmented Generation (RAG) 파이프라인과 외부 사실 확인을 통합하는 신뢰도 가중 투표 메커니즘을 통해 작동합니다. FinVet은 동적으로 검증 전략을 조정하는 세 가지 처리를 통해 기존 방법의 한계를 극복하는 동시에 더 높은 투명성을 제공합니다. 이를 통해 증거 기반의 판결, 출처 귀속, 신뢰도 점수를 제공하며, 불충분한 증거에 대한 불확실성 표시를 명확히 하게 됩니다.

- **Technical Details**: FinVet은 세 가지 계층 처리 전략을 도입하여 신뢰도 점수에 따라 검증 접근 방식을 동적으로 선택합니다. 이는 높은 신뢰도의 경우 직접 메타데이터 추출, 중간 신뢰도의 경우 하이브리드 모델 추론, 낮은 신뢰도의 경우 모델 기반 분석을 통해 이루어집니다. RAG 구성요소는 도메인 특정 데이터셋에 맞춘 외부 지식소스를 활용하며, 사실 확인 파이프라인은 직접적인 증거가 없을 경우 대체 메커니즘을 사용하여 처리합니다.

- **Performance Highlights**: FinVet의 성능 평가 결과는 FinFact 데이터셋을 기준으로 하여 F1 점수 0.85를 달성하였으며, 이는 기존의 최고 단일 파이프라인(사실 확인 파이프라인) 대비 10.4% 개선된 결과입니다. 또한, 스탠드얼론 RAG 접근법에 비해 37%의 성능 향상도 보여주었습니다. 이 결과는 FinVet이 기존 기술과 비교할 때 보다 나은 정확성과 설명 가능성을 제공함을 입증합니다.



### MATH-Beyond: A Benchmark for RL to Expand Beyond the Base Mod (https://arxiv.org/abs/2510.11653)
- **What's New**: DeepSeek-R1의 출현으로 인해 강화 학습 방법이 새로운 수학적 추론 능력을 열어주는 새로운 물결을 맞이했습니다. 그러나 많은 오픈소스 모델이 MATH-500 및 AIME 2024와 같은 일반적인 수학 벤치마크에서 거의 모든 질문을 해결할 수 있다는 제한이 드러났습니다. 이는 현재의 RL 미세 조정 방법이 기존의 솔루션 방식만을 강화할 뿐, 전혀 새로운 방식을 발견하는 데는 한계가 있음을 강조합니다. 이를 극복하기 위해 MATH-Beyond (MATH-B)를 소개하며, 이는 기존 모델보다 더 높은 추론 능력이 요구되는 새로운 벤치마크입니다.

- **Technical Details**: MATH-B는 고등학교 수준의 수학 문제를 대상으로 하여, 인기 있는 개방형 가중치 모델들이 1024회 시도하더라도 해결하지 못할 문제로 설계되었습니다. 이 데이터셋은 DAPO-Math-17K와 DeepScaleR에서 선정된 문제들로 구성되어 있으며, 그 주제는 기존 벤치마크와 일치합니다. 게다가, 문제들은 GPT-5-Mini와 o4-mini-high와 같은 강력한 추론 모델을 통해 검증되어 정확성을 담보합니다.

- **Performance Highlights**: RL 미세 조정 모델인 Nemotron-Research-Reasoning-Qwen-1.5B 및 DeepScaleR-1.5B-Preview가 MATH-B에서 낮은 성과를 보임으로써 현재 접근법의 한계를 나타냅니다. 이러한 결과는 새로운 접근 방식이 기존 모델보다 더 발전된 추론 능력을 요구한다는 필요성을 시사합니다. MATH-B는 탐색 기반의 RL 접근 방식을 촉진하여 더 깊은 추론 능력을 이끌어내기를 기대합니다.



### Attention Factors for Statistical Arbitrag (https://arxiv.org/abs/2510.11616)
Comments:
          Accepted to the 6th ACM International Conference on AI in Finance

- **What's New**: 이 논문에서는 통계적 차익 거래(statistical arbitrage)를 위한 새로운 프레임워크를 개발하였습니다. 특히, 'Attention Factors'라는 조건부 잠재 요인을 도입하여 유사한 자산을 식별하고 잘못된 가격을 파악하는 동시에 거래 비용 이후 극대화된 리스크 조정 성과를 위한 거래 정책을 수립합니다. 기존의 두 단계 접근 방식 대신, 우리는 하나의 단계에서 거래 가능한 차익 요인(tradable arbitrage factors)과 포트폴리오 배치를 공동으로 학습하는 방식을 제안합니다.

- **Technical Details**: 이 모델은 복잡한 상호작용을 허용하는 기업 특성의 임베딩(embeddings)을 통해 요인을 학습합니다. 또한, 일반 시퀀스 모델을 통해 시간 시계열 신호를 식별하여 순수한 Sharpe 비율을 극대화하는 것을 목표로 합니다. 24년간의 미국 상장주식 데이터를 활용한 실증 분석을 통해, 논문에서 제안한 Attention Factor 모델은 거래 비용 없이 4를 초과하는 Sharpe 비율을 달성했습니다.

- **Performance Highlights**: 논문에서는 Attention Factor 모델이 연간 16%의 수익률을 달성하면서도 시장 리스크와는 독립적인 특성을 보인다고 주장합니다. 거래 비용을 반영한 경우에도 2.3의 Sharpe 비율을 달성하며, 이는 기존 모델들보다 84% 증가한 수비적 성과를 보입니다. 특히, 이 모델은 산업 부문과 밀접한 관련이 있는 해석 가능한 구조를 가지고 있으며, 적은 변동성을 가진 약한 요인들이 중요한 역할을 한다는 점을 밝혀냈습니다.



### LLM-Oriented Token-Adaptive Knowledge Distillation (https://arxiv.org/abs/2510.11615)
Comments:
          15 pages, 4 figures

- **What's New**: 이번 연구에서는 Knowledge Distillation (KD)의 최신 동향과 한계를 다루고 있으며, LLM(대형 언어 모델)의 지식을 효과적으로 압축하기 위해 새로운 방법론인 LLM-Oriented Token-Adaptive Knowledge Distillation (AdaKD)를 제안합니다. 기존의 logit 기반 방법들이 정적인 접근 방식으로 인해 학생 모델의 동적인 학습 과정과 맞지 않음을 지적하면서, 각 토큰의 실시간 학습 상태에 맞춰 적응적인 지식 전이 과정이 필요하다고 강조합니다. AdaKD는 토큰의 난이도에 따라 디스틸레이션을 조정하여 지식 전이를 최적화하는 두 가지 모듈을 포함하고 있습니다.

- **Technical Details**: AdaKD 프레임워크는 두 가지 상호작용 모듈로 구성됩니다. 첫 번째 모듈인 Loss-Driven Adaptive Token Focusing (LATF)은 학생 모델의 학습 안정성을 모니터링하여 가장 가치 있는 토큰에 집중하게 만듭니다. 두 번째 모듈인 Inverse Difficulty Temperature Scaling (IDTS)은 각 토큰의 난이도에 따라 온도를 조정하여 학습 과정을 최적화하고, 오류 수정을 위한 적절한 온도 전략을 구현합니다. 이러한 메커니즘을 통해 AdaKD는 다양한 디스틸레이션 방법의 성능을 향상시킬 수 있습니다.

- **Performance Highlights**: AdaKD는 여러 모델 아키텍처와 벤치마크에서 다양한 디스틸레이션 기법에 일관되게 성능을 개선합니다. 기존의 각종 KD 방법들과 비교할 때, ADAKD는 동적인 절도 기법을 통해 지식 전이에 있어 더욱 효율적이고 안정적인 결과를 보여줍니다. 또한, AdakD는 각 토큰별로 최적의 온도 조정을 통해 더 높은 일반화 능력을 달성하고, 있기 때문에 이 방법론은 ILLM(대형 언어 모델)의 발전에 크게 기여할 것으로 기대됩니다.



### SemCSE-Multi: Multifaceted and Decodable Embeddings for Aspect-Specific and Interpretable Scientific Domain Mapping (https://arxiv.org/abs/2510.11599)
- **What's New**: 이 논문에서는 SemCSE-Multi라는 새로운 비지도 학습(unsupervised) 프레임워크를 제안하여 과학 초록의 다면적 임베딩(embeddings)을 생성합니다. 이 임베딩은 연구자가 필요한 특정 측면(aspect)을 명확히 하고 독립적으로 포착할 수 있도록 하여 세밀하고 조절 가능한 유사성 평가(similarity assessment)를 가능하게 합니다. 또한, 본 접근법은 과학 분야의 사용자 주도 시각화를 위한 적응적 기능을 제공하는 점도 특징입니다.

- **Technical Details**: 제안된 접근법은 각 연구 초록에 대해 аспект별 요약 문장을 생성하고 이는 임베딩 모델에 의해 의미적으로 유사한 요약이 임베딩 공간 내에서 근접하게 배치되도록 학습됩니다. 최종적으로, 이 аспект별 임베딩 기능은 단일 임베딩 모델로 통합되어 단일 전방 통과(forward pass)에서 여러 аспект 임베딩을 예측할 수 있게 됩니다. 또한, 임베딩을 자연어 설명으로 복원하는 디코딩 파이프라인을 도입하여 임베딩 공간의 해석 가능성을 크게 향상시킵니다.

- **Performance Highlights**: 이 연구는 주로 침입 생물학 분야에서 성능을 평가하였으며, 전문가의 지도를 받았습니다. 논문의 처음에 제안한 대로, 다양한 측면을 취합하여 사용자 맞춤의 시각화 및 결과 도출을 가능하게 함으로써, 사용자가 필요로 하는 특정 연구 방향에 대한 명확한 통찰을 제공합니다. 이러한 접근법은 기존 방법의 한계를 극복하며, 특히 저차원 시각화에서 비어 있는 영역의 의미 있는 텍스트 설명을 생성하는데 효과적임을 입증하였습니다.



### Hierarchical Qubit-Merging Transformer for Quantum Error Correction (https://arxiv.org/abs/2510.11593)
Comments:
          6 pages, 5 figures

- **What's New**: 이번 논문에서는 효율적인 양자 오류 수정(QEC) 스킴을 위한 계층적 큐빗 병합 변환기(HQMT)를 제안합니다. HQMT는 안정자 코드(stabilizer code)의 구조적 그래프를 활용하여 여러 스케일에서 오류 상관관계를 학습하는 새로운 디코딩 프레임워크입니다. 이 연구는 딥러닝을 기반으로 한 신경망 디코더의 최신 발전을 활용하여 양자 컴퓨팅의 신뢰성을 높이고자 합니다.

- **Technical Details**: HQMT 아키텍처는 구조적으로 관련된 안정자 그룹에 대해 로컬로 주의(attention)를 계산하고, 이를 체계적으로 병합하여 오류 신드롬(error syndrome)의 글로벌 뷰를 구성합니다. 특히, 변환기(transformer) 아키텍처에 전용 큐빗 병합 레이어(qubit-merging layer)를 통합하여 오류율(logical error rate)을 크게 낮추는 데 성공했습니다. 이 계층적 접근 방식은 표면 코드(surface code) 디코딩에 효과적이며 확장 가능한 프레임워크를 제공합니다.

- **Performance Highlights**: HQMT는 다양한 코드 거리(code distance)에서 기존의 신경망 기반 QEC 디코더 및 강력한 신뢰 전파(belief propagation)와 순서 통계 디코딩(ordered statistics decoding) 기법인 BP+OSD를 능가하는 성능을 보였습니다. 이러한 결과는 HQMT가 신뢰할 수 있는 양자 컴퓨팅 실현을 위한 중요한 한 걸음을 내딛고 있음을 보여줍니다.



### Characterizing Web Search in The Age of Generative AI (https://arxiv.org/abs/2510.11560)
- **What's New**: 본 논문에서는 Generative AI 모델과 전통적인 웹 검색 엔진 간의 주요 차이점들을 탐구합니다. Generative search는 사용자 쿼리에 대한 응답으로 독립적인 웹 페이지 리스트를 반환하는 대신, 관련 정보를 종합하여 일관된 텍스트로 결과를 제공합니다. 이를 통해 보다 다양한 소스와 관점을 이용할 수 있는 가능성이 열리게 됩니다.

- **Technical Details**: Generative search는 여러 차원에서 전통적인 웹 검색과 다른데, 첫째로 결과 포맷이 다릅니다. 전통적인 검색 결과는 주로 페이지들의 리스트로 제공되는 반면, Generative search는 단일한 응답 형식으로 정보를 제공합니다. 둘째로, Generative search는 선택된 웹 페이지의 수보다 더 넓은 범위를 커버할 수 있는 능력을 가지며, 내부 지식과 외부 웹에서 검색된 정보를 결합하여 사용합니다.

- **Performance Highlights**: 연구 결과, Generative search 엔진은 전통적인 검색 엔진보다 소스 다양성이 더 높으며, 각 엔진이 내부 지식에 의존하는 정도는 다릅니다. 예를 들어, GPT-Tool은 평균적으로 0.4개의 웹 페이지를 참고하는 반면, AIO와 Gemini는 각각 8.6, 8.5개를 참조합니다. 이는 정보 제공의 투명도와 신뢰성, 사용자 자율성에 미치는 잠재적 영향에 대한 고려가 필요함을 시사합니다.



### Query-Specific GNN: A Comprehensive Graph Representation Learning Method for Retrieval Augmented Generation (https://arxiv.org/abs/2510.11541)
- **What's New**: 본 연구에서는 Retrieval-Augmented Generation (RAG) 시스템의 성능을 향상시키기 위해 Multi-information Level Knowledge Graph (Multi-L KG)를 설계하였습니다. Multi-L KG는 다층의 정보를 통해 다단계 질문을 더 효과적으로 이해할 수 있는 기반을 제공합니다. 또한 Query-Specific Graph Neural Network (QSGNN)를 도입하여 상관 정보의 경량화된 전파와 다수의 정보 집계를 가능하게 하였습니다.

- **Technical Details**: 이 연구는 다단계 질문을 처리하기 위해 Multi-L KG에서 여러 정보 수준과 복잡한 관계를 포착하는 것에 중점을 두고 있습니다. QSGNN은 두 가지 메시지 전파 방식인 intra-level과 inter-level을 사용하여 각 수준 내의 기본 의미 관계 및 수준 간의 지역-전역 관계를 고려합니다. 이러한 디자인은 응답의 질을 높이고 노이즈의 영향을 줄이는 데 기여합니다.

- **Performance Highlights**: QSGNN의 성능은 기존 방법들과 비교했을 때 특히 고복잡도 문제에서 33.8%의 성능 향상을 보여주며, 이는 RAG 시스템의 다단계 질문 처리에서의 가능성을 입증합니다. 광범위한 실험 결과는 제안된 프레임워크의 효과를 입증합니다.



### CodeWatcher: IDE Telemetry Data Extraction Tool for Understanding Coding Interactions with LLMs (https://arxiv.org/abs/2510.11536)
Comments:
          ICSME 2025 Tool Demonstration Track

- **What's New**: 이번 연구에서는 CodeWatcher라는 가벼운 클라이언트-서버 시스템을 소개하며, 이를 통해 Visual Studio Code (VS Code) 에디터 내에서의 세밀한 개발자 상호작용 이벤트를 포착할 수 있도록 설계되었습니다. CodeWatcher는 코드 생성 도구(CGTS)가 수행한 삽입, 삭제, 복사-붙여넣기와 같은 의미 있는 이벤트를 기록하여 사용자 워크플로우를 변경하지 않고도 개발자 활동을 지속적으로 모니터링할 수 있게 합니다. 이 시스템은 VS Code 플러그인, Python 기반의 RESTful API 및 MongoDB 백엔드로 구성되어 있으며, 각 이벤트는 구조화되고 타임스탬프가 부여되어 코딩 세션의 재구성을 가능하게 합니다.

- **Technical Details**: CodeWatcher는 프로그래밍 언어와 CGT를 넘나드는 호환성을 갖춘 클라이언트-서버 시스템으로, 코드 변경, IDE 명령, 인지적 과정 등 다양한 개발자 활동을 세밀하게 기록할 수 있습니다. 시스템은 사용자의 코딩 행동을 기반으로 알림이나 제안을 생성하기 위한 규칙 기반 엔진을 통해 실시간 피드백을 지원합니다. CodeWatcher의 클라이언트 플러그인은 JavaScript로 작성되어 있으며, HTTP를 통해 백엔드 API로 이벤트 데이터를 전송합니다. MongoDB를 사용하는 백엔드 데이터베이스는 유연성 있는 스키마를 통해 다양한 상호작용 로그 형식을 처리합니다.

- **Performance Highlights**: CodeWatcher는 개발자가 사용하는 CGT와의 상호작용 데이터를 실시간으로 캡처함으로써 이전 연구에서 부족했던 자세한 상호작용 텔레메트리 데이터를 제공합니다. 교육적 및 산업적 맥락 모두에서 이해 가능성과 투명성을 강화하고, 필요에 따라 맞춤형 피드백 메커니즘을 제공함으로써 지속적인 개발자 생산성 향상에 기여할 수 있습니다. 이 시스템은 개발자 세션의 후속 분석과 행동 모델화를 위한 기초 데이터를 형성하여 AI 기반 코드 생성 도구의 결합된 효과를 더 깊이 이해하게 합니다.



### A Flexible Multi-Agent Deep Reinforcement Learning Framework for Dynamic Routing and Scheduling of Latency-Critical Services (https://arxiv.org/abs/2510.11535)
- **What's New**: 본 논문은 애플리케이션에서 요구하는 기한 내에 패킷을 신뢰성 있게 전달하는 것을 목표로 하는 Delay-Constrained Maximum-Throughput (DCMT) 동적 네트워크 제어 문제를 다룹니다. 기존의 네트워크 제어 솔루션이 평균 지연 성능에만 초점을 맞추는 반면, 본 논문에서는 Multi-Agent Deep Reinforcement Learning (MA-DRL)의 발전을 활용하여 End-to-End (E2E) 피크 지연 보장을 제공합니다. 통합된 라우팅과 분산된 스케줄링 아키텍처를 바탕으로 한 새로운 MA-DRL 네트워크 제어 프레임워크를 제안합니다.

- **Technical Details**: 제안된 MA-DRL 네트워크 제어 프레임워크는 Multi-Agent Deep Deterministic Policy Gradient (MADDPG) 기법을 사용하여 패킷의 생애 주기에 따라 경로를 동적으로 할당하고 패킷 전송을 일정에 맞게 조정합니다. 이 과정에서 네트워킹 도메인의 중요한 지식을 활용하여 성능과 학습 복잡성 간의 균형을 맞추기 위해 데이터 기반의 Deep Reinforcement Learning (DRL) 에이전트 및 전통적인 규칙 기반 정책을 통합할 수 있습니다. 이 방법은 표준적인 확률적 최적화 기반 접근 방식에 비해 우수한 성능을 보여줍니다.

- **Performance Highlights**: 제안된 접근 방식은 기존의 안정성 있는 지연 보장을 위한 알고리즘에 비해 높은 성능 개선을 나타냅니다. 실험 결과에 따르면, 새로운 MA-DRL 프레임워크는 처리량을 극대화하면서 시기 적절한 패킷 전달을 이끌어내는 데 효과적임을 입증했습니다. 이러한 연구는 데이터 기반 DRL 에이전트와 새로운 규칙 기반 정책 간의 상호 작용에 대한 주요 통찰을 제공하여 지연이 중요한 서비스의 효율적이고 높은 성능 제어를 가능하게 합니다.



### Cracking CodeWhisperer: Analyzing Developers' Interactions and Patterns During Programming Tasks (https://arxiv.org/abs/2510.11516)
Comments:
          VL/HCC 2025 Short Paper

- **What's New**: 이 연구는 코딩 도구를 사용하는 소프트웨어 개발자들의 행동 패턴을 분석한 최초의 연구 중 하나입니다. 특히 Amazon의 CodeWhisperer와 같은 LLM(대규모 언어 모델) 기반 코드 생성 도구의 사용자 상호작용 패턴에 초점을 맞추었습니다. 두 개의 사용자 연구를 통해 개발자들이 이러한 도구를 사용하여 코드를 작성하고 수정하는 방식을 실질적으로 관찰하고 데이터를 수집하였습니다.

- **Technical Details**: 연구는 두 가지 주요 연구 질문을 탐색합니다: 사용자들이 CodeWhisperer와 상호작용하며 발견하는 코드 수정 패턴과 제안의 유효성입니다. 연구에 참여한 두 그룹의 피험자들은 단계적으로 난이도가 증가하는 프로그래밍 과제를 해결하기 위해 CodeWhisperer를 사용했습니다. CodeWatcher라는 VSCode 플러그인을 개발하여 사용자 상호작용 데이터를 기록하고 분석하는 데 사용하였습니다.

- **Performance Highlights**: 분석 결과, 네 가지 주요 행동 패턴이 도출되었습니다: 1) 점진적인 코드 수정, 2) 자연어 주석을 이용한 명시적 지시, 3) 모델 제안을 통한 기본 구조화, 4) 외부 자원과의 통합적 사용입니다. 이러한 패턴은 개발자들이 LLM 도구의 제안을 활용하는 방식에 대한 깊은 통찰을 제공합니다. 이 연구는 LLM 기반 코드 생성 도구의 실질적인 사용에 대한 새로운 이해를 제시합니다.



### Automatic Music Sample Identification with Multi-Track Contrastive Learning (https://arxiv.org/abs/2510.11507)
- **What's New**: 이 논문에서는 기존 오디오 트랙의 샘플을 재사용하여 새로운 음악 콘텐츠를 만드는 샘플링(Sampling) 기술에 대한 자동 샘플 식별(automatic sample identification) 작업을 다루고 있습니다. 자가 지도 학습(self-supervised learning) 접근 방식을 도입하여 다중 트랙 데이터셋에서 인공 혼합물의 긍정적 쌍을 생성하고, 새로운 대조적 학습(objective) 방법을 설계하였습니다. 이 방법은 이전의 최첨단 기준선보다 우수한 성능을 보이며, 다양한 장르에 강건하고, 참조 데이터베이스에 노이즈 곡을 추가하는 것이 용이함을 보여줍니다.

- **Technical Details**: 샘플 식별 작업은 오디오 핑거프린팅(audio fingerprinting)과 밀접한 관련이 있으며, 2013년 Van Balen et al.이 이 작업을 처음 시작했습니다. 그러나 샘플 식별은 적절한 훈련 데이터 부족 등 실용적인 장애물에 직면해 있습니다. 본 연구에서는 음악 녹음의 다양한 출처에서 긍정적 쌍을 구축하고, 특정 변환을 무작위로 적용하여 효과적으로 훈련 세트를 생성하는 새로운 방법을 채택하였습니다.

- **Performance Highlights**: 우리는 다양한 장르에 걸친 개인 데이터 세트와 표준 힙합 벤치마크에서 모델 성능을 평가하였습니다. 그 결과, 평균 정밀도(mean average precision)에서 15% 이상 향상된 성능을 보였고, 특정 교육 모듈 및 다중 트랙 교육 세트의 영향을 평가하며 모델의 견고성 또한 입증하였습니다. 또한 전체 훈련 코드를 공개할 예정입니다.



### People use fast, flat goal-directed simulation to reason about novel problems (https://arxiv.org/abs/2510.11503)
Comments:
          Pre-print

- **What's New**: 이번 연구는 사람들이 새로운 문제 환경에서 어떻게 의사 결정을 하고 판단을 형성하는지를 분석하고, 이를 통해 'Intuitive Gamer'라는 컴퓨테이셔널 인지 모델을 제안합니다. 이 모델은 제한된 깊이의 목표 지향적인 확률적 시뮬레이션에 기반을 두며, 실제 게임을 플레이하기 전에 사람들이 게임을 평가하는 방식을 설명합니다. 연구 결과, 인간의 판단과 결정 과정을 기존의 전문가 모델보다 훨씬 더 잘 설명할 수 있음을 보여줍니다.

- **Technical Details**: Intuitive Gamer 모델은 '플랫' 에이전트와 '패스트' 이유자 두 가지 모듈로 구성되어 있습니다. 플레이어가 선택할 수 있는 행동을 추론하기 위해 표면적으로 평가 가능한 목표 지향적 휴리스틱 함수를 사용하여 확률적으로 조치를 선택합니다. 이 모델은 수천 개의 평가 없이도 간단한 정책을 수립하고, 게임의 속성을 추론할 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: 대규모 행동 연구를 통해 1000명 이상의 참여자와 121개의 새로운 이인용 전략 보드 게임을 활용한 결과, Intuitive Gamer 모델은 사람들이 게임의 속성을 평가하는 방식과 실제 플레이에서 초기 행동 선택을 하는 방식을 매우 잘 포착했습니다. 이 연구는 사람들이 새로운 문제를 신속하게 평가하고 행동하는 방식에 대한 새로운 통찰을 제공하며, 유연하고 인간과 유사한 AI 시스템 디자인에 기여할 수 있는 가능성을 갖고 있습니다.



### Offline Reinforcement Learning with Generative Trajectory Policies (https://arxiv.org/abs/2510.11499)
Comments:
          Preprint. Under review at ICLR 2026

- **What's New**: 이번 연구에서는 생성 모델을 활용한 오프라인 강화 학습에서 정책의 효율성과 표현력을 동시에 충족할 수 있는 새로운 프레임워크인 Generative Trajectory Policies (GTPs)를 제안합니다. 기존의 느리고 반복적인 생성 방식과 빠르지만 성능이 떨어지는 단일 단계 방식 사이의 격차를 메우기 위한 방법을 탐구하여, 여러 현대 생성 모델들을 연속 시간 생성 경로로 이해할 수 있는 통합 관점을 제공합니다.

- **Technical Details**: 연구의 핵심은 생성 경로를 지배하는 일반 미분 방정식(Ordinary Differential Equation, ODE)으로 정의된 연속 시간 생성 모델의 통합 프레임워크입니다. GTP는 이 ODE의 전체 솔루션 맵을 학습하여 느리고 높은 충실도의 샘플링과 빠르고 낮은 충실도의 단축키를 넘어서 유연하고 다단계 결정론적 생성을 가능하게 합니다. 이를 위해, 연구진은 두 가지 이론적으로 기반한 방법론을 통해 계산 비용과 훈련 불안정성을 해결했습니다.

- **Performance Highlights**: GTP는 D4RL 벤치마크에서 최첨단 성능을 달성하였으며, 기존 생성 정책들보다 높은 성능을 기록했습니다. 특히, 여러 현실적으로 도전적인 AntMaze 작업에서 완벽한 점수를 기록하며 표현력과 효율성 간의 균형을 더 잘 맞추는 능력을 입증하였습니다.



### Investigating Large Language Models' Linguistic Abilities for Text Preprocessing (https://arxiv.org/abs/2510.11482)
Comments:
          Accepted in WI-IAT 2025. Pre-camera-ready version

- **What's New**: 이 연구에서는 전통적인 텍스트 전처리 기술에 대한 대안으로 대규모 언어 모델(LLMs)을 활용하는 방법을 탐구합니다. 기존의 방법들이 문맥 정보를 간과하는 반면, LLMs는 입력된 document의 문맥을 고려하여 stopword를 제거하고, lemmatization과 stemming을 수행할 수 있습니다. 연구 결과, LLM 기반 전처리가 전통적인 기법들에 비해 더 높은 정확도를 기록하며 기계 학습 알고리즘의 성능 또한 향상되는 것을 보여줍니다.

- **Technical Details**: 연구 방법론은 LLM들이 주어진 작업을 수행할 수 있도록 하는 프롬프트를 정의하는 것을 포함합니다. 각 LLM은 전처리 작업에 대한 설명, 몇 가지 예제, 전처리할 텍스트, 텍스트의 언어 및 하위 작업의 문맥을 입력받아 해당 전처리된 버전을 출력합니다. 또한, 연구는 영어, 프랑스어, 독일어, 이탈리아어, 포르투갈어, 스페인어를 포함한 여러 언어에 걸쳐 LLM 전처리의 성능을 평가합니다.

- **Performance Highlights**: LLM 기반의 전처리 기술은 전통적 stopword 제거, lemmatization, stemming 방법을 각각 97%, 82%, 74%의 정확도로 반복할 수 있음을 보여주었습니다. 또한, LLMs로 전처리된 문서에서 학습된 기계 학습 알고리즘은 전통 기술보다 최대 6%의 개선을 나타내어 $F_1$ 측정에서 성능 향상을 이끌어냅니다.



### Coordinated Strategies in Realistic Air Combat by Hierarchical Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2510.11474)
Comments:
          2025 IEEE International Conference on Agentic AI (ICA)

- **What's New**: 이 논문에서는 비현실적인 공중전 시뮬레이션 환경에서의 문제를 해결하기 위해 새로운 3D 다중 에이전트 공중전 환경과 계층적 다중 에이전트 강화 학습(Hierarchical Multi-Agent Reinforcement Learning) 프레임워크를 소개합니다. 특히, 이 연구는 불완전한 상황 인식과 비선형 비행 역학이라는 두 가지 도전 과제를 해결하는 것을 목표로 하고 있습니다.

- **Technical Details**: 제안된 방법은 이질적인 에이전트 동역학(heterogeneous agent dynamics), 커리큘럼 학습(curriculum learning), 리그 플레이(league-play) 및 새로운 훈련 알고리즘을 결합하여 기초하고 있습니다. 의사결정 과정은 두 가지 추상화 수준으로 조직되어, 저수준 정책(low-level policies)은 세밀한 조작을 학습하고, 고수준 정책(high-level policies)은 임무 목표에 따라 전술적 명령을 발행합니다.

- **Performance Highlights**: 경험적 결과에 따르면, 계층적 접근법이 복잡한 공중전 시나리오에서 학습 효율성과 전투 성능을 모두 개선하는 데 기여하는 것으로 나타났습니다. 이를 통해 공중전 작전에서의 성능 향상이 확인되었으며, 새로운 알고리즘의 효과성을 입증하였습니다.



### Iterative Amortized Inference: Unifying In-Context Learning and Learned Optimizers (https://arxiv.org/abs/2510.11471)
- **What's New**: 이 논문에서는 태스크 간 재사용되는 컴퓨테이션(computation) 또는 유도 편향(inductive bias)을 통해 신속한 일반화를 가능하게 하는 통합 프레임워크를 제안합니다. 여기에는 메타 학습(meta-learning), 인-컨텍스트 학습(in-context learning), 프롬프트 튜닝(prompt tuning), 학습된 옵티마이저(learned optimizers) 등이 포함됩니다. 각 접근법은 태스크 간 정보의 인코딩(encoding) 및 활용 방식에서 차이를 보이며, 이 연구는 이러한 차이를 학습 과정에서 감가(amortization)되는 측면으로 분리하여 설명합니다.

- **Technical Details**: 우리는 세 가지 구별되는 감가 체계를 제안하는데, 이는 파라메트릭(parametric), 암묵적(implicit), 명시적(explicit) 방식입니다. 이들은 각각 유도 편향을 외부화(externalize), 내재화(internalize), 공동 모델링(jointly model)하는 방식에 따라 분류됩니다. 감가의 핵심 한계는 대규모 데이터셋에 대한 적응에서 처리 능력이 제한적이라는 점을 지적하며, 이를 해결하기 위해 stochastic optimization에 영감을 받은 반복적 감가 추론(iterative amortized inference) 모델을 제안합니다.

- **Performance Highlights**: 이 연구는 메타 학습(meta-learning)과 순방향 패스(forward-pass) 감가 기법을 연결하여 일반적인 태스크 적응의 기초를 제공합니다. 또한, 반복적 감가 체계를 도입함으로써 대규모 데이터셋에 대한 실행 가능성을 높였습니다. 이는 다음 단계로 미니 배치(mini-batch) 방법을 통해 제안된 아이디어를 구체화하였고, 최적화 문제의 처리에서 고급 유연성과 확장성을 가능하게 합니다.



### Audio-Maestro: Enhancing Large Audio-Language Models with Tool-Augmented Reasoning (https://arxiv.org/abs/2510.11454)
Comments:
          9pages

- **What's New**: 최근 대규모 멀티모달 모델(LMMs)의 발전은 오디오 이해에서 강력한 능력을 보여주고 있습니다. 그러나 기존 시스템은 대부분 엔드-투-엔드 추론(end-to-end reasoning)에만 의존하여 구조적 지식이나 전문 신호 분석이 필요한 작업의 해석 가능성과 정확성을 제한합니다. 이에 대한 대안으로 Audio-Maestro를 제안하며, 이는 외부 도구를 자율적으로 호출하고 그 출력 결과를 추론 과정에 통합하는 오디오-언어 모델(audio-language models)을 위한 도구 보강 오디오 추론 프레임워크입니다.

- **Technical Details**: Audio-Maestro는 두 가지 단계로 구성됩니다. 첫 번째 단계에서 모델은 질의에 직접 응답할 수 있는지, 아니면 도구 지원이 필요한지를 결정합니다. 두 번째 단계에서는 도구가 호출될 경우, 그 결괏값을 추론 과정에 통합하여 최종 응답을 생성합니다. 이러한 구조는 저수준의 음향 분석과 고수준의 의미 분석 간의 연결을 가능하게 합니다.

- **Performance Highlights**: Audio-Maestro는 일반 오디오 추론 성능을 일관되게 향상시킵니다. 예를 들어, Gemini-2.5-flash는 MMAU-Test에서 평균 정확도가 67.4%에서 72.1%로 증가하며, DeSTA-2.5는 58.3%에서 62.8%로, GPT-4o는 60.8%에서 63.9%로 상승합니다. 이는 Audio-Maestro가 대규모 오디오 언어 모델의 추론 프로세스에 구조화된 도구 출력을 통합한 첫 번째 프레임워크라는 점에서 중요한 의미가 있습니다.



### Reconstructing 12-Lead ECG from 3-Lead ECG using Variational Autoencoder to Improve Cardiac Disease Detection of Wearable ECG Devices (https://arxiv.org/abs/2510.11442)
Comments:
          24 pages, 5 figures, submitted to Nature Communications

- **What's New**: 본 연구에서는 WearECG라는 새로운 Variational Autoencoder (VAE) 방법을 제안하여 세 개의 리드(II, V1, V5)에서 12리드 ECG를 재구성합니다. 이 모델은 ECG 신호의 시간적 및 공간적 의존성을 더 잘 포착할 수 있는 구조적 개선을 포함하고 있습니다. 다양한 임상 조건을 포함한 다중 레이블 분류 작업에 대해 미리 훈련된 ECGFounder 모델을 세밀하게 조정하여 진단 유틸리티를 검증합니다.

- **Technical Details**: WearECG 모델에서는 잔차 합성곱 신경망(Residual CNNs)과 그룹 정규화(Group Normalization)와 같은 여러 기술적 접근 방식을 사용하여 신호 생성 과정을 최적화합니다. 심전도 신호의 재구성 품질은 평균 제곱 오차(Mean Squared Error, MSE), 평균 절대 오차(Mean Absolute Error, MAE) 및 Fréchet Inception Distance (FID)와 같은 측정 기준을 통해 평가됩니다. 또한 경량화된 VAE 구조를 채택하여, 제한된 입력 조건에서도 생리학적으로 가능한 신호 재구성이 가능합니다.

- **Performance Highlights**: 모델의 성능 평가에서는 평균 MSE 0.00100, MAE 0.01782, FID 12.64와 같은 우수한 결과가 나타났습니다. Turing 테스트를 통해 세 명의 심장전문의가 실제 ECG와 합성 ECG를 구별하는 과정에서 높은 임상적 관용성을 보였다고 보고되었습니다. 이러한 결과는 생성된 신호가 질병 특정 특성을 잘 보존하고 있음을 나타내며, 실제 임상 적용 가능성을 높입니다.



### KnowRL: Teaching Language Models to Know What They Know (https://arxiv.org/abs/2510.11407)
Comments:
          14 pages, 7 figures

- **What's New**: 최근의 연구에 따르면, 신뢰할 수 있는 AI는 단순히 지식을 축적하는 것만으로는 불가능하며, 자신의 한계와 지식 범위를 정확히 인식하는 능력이 필요하다는 사실이 강조되고 있다. 본 논문은 이러한 문제를 해결하기 위해 KnowRL이라는 새로운 프레임워크를 제안한다. 이 프레임워크는 LLM(large language models)의 자기 인식(self-knowledge)을 강화하여 보다 안전하고 책임감 있는 AI 행동을 가능하게 한다.

- **Technical Details**: KnowRL 프레임워크는 두 가지 주요 구성 요소인 introspection(내성)과 consensus-based rewarding(합의 기반 보상)을 결합하여 LLM이 자신의 수행 가능성과 비수행 가능성을 스스로 판단하고, 이를 통해 자기 인식을 강화한다. 이러한 접근 방식은 최소한의 데이터만으로도 모델의 내적 인식을 통해 자기 일관성을 높이는 데 중점을 둔다. 또한, LLM 생성 데이터를 활용함으로써 외부 데이터 수집의 어려움을 극복하고 보상의 신뢰성을 증가시킨다.

- **Performance Highlights**: KnowRL을 통해 LLaMA-3.1-8B 및 Qwen-2.5-7B 모델에서 자기 인식(self-knowledge) 개선이 관찰되었으며, 정확도에서 최대 28%, F1 점수에서 12%의 향상을 기록하였다. 이 방법은 몇 차례의 반복 학습만으로도 기초 성능을 초과할 수 있는 잠재력을 가진 것으로 나타났다. 향후 AI의 안전한 배포와 신뢰성 있는 활용을 위해, 이러한 자기 개선 과정이 모든 미래 모델에 적용되기를 권장한다.



### Living Off the LLM: How LLMs Will Change Adversary Tactics (https://arxiv.org/abs/2510.11398)
Comments:
          6 pages, 0 figures

- **What's New**: 이번 논문에서는 악의적인 행위자들이 시스템에 이미 존재하는 합법적인 도구와 프로세스를 활용하여 탐지를 피하는 'living off the land' 공격에 대해 다룹니다. 특히, 미래의 온디바이스 LLM(대형 언어 모델)이 어떻게 보안 위협으로 작용할 수 있는지를 논의하고 있습니다. LLM이 공격 파이프라인에 통합되는 사례를 설명하며, 이를 해소하기 위한 보안 커뮤니티의 접근법에 대해서도 제안합니다.

- **Technical Details**: LLM은 원격 또는 온디바이스 코드 생성 기능을 제공합니다. 연구자들은 ChatGPT와 같은 업계 최상위 LLM을 사용하여 폴리모픽 멀웨어를 생성할 수 있음을 입증하였습니다. 이러한 멀웨어는 새로운 시스템으로 퍼질 때 코드의 특정 요소를 다시 작성하여 전통적인 정적 서명을 통해 탐지하기 어렵게 만듭니다. LLM을 활용한 공격자들의 다양한 전략이 입증되고 있으며, 멀웨어 제작의 접근성이 더 쉬워지고 있습니다.

- **Performance Highlights**: LLM은 공격자들에게 저숙련의 도구를 제공하여 더 정교하고 확장 가능한 사회공학적 공격을 가능하게 하고 있습니다. 이는 특히 랜섬웨어 공격에서 두드러지며, AI가 생성한 스피어 피싱이 인간이 작성한 이메일보다 더 설득력이 높다는 사례도 소개됩니다. 또한, LLM 기반의 공격들이 특정 기업 환경 내에서 실행되며 침투 테스트를 자동화하는 성공적인 사례들이 나타났습니다.



### Medical Interpretability and Knowledge Maps of Large Language Models (https://arxiv.org/abs/2510.11390)
Comments:
          29 pages, 34 figures, 5 tables

- **What's New**: 이 연구는 Large Language Models (LLMs)에서 의료 분야의 해석 가능성에 대한 체계적인 연구를 제시합니다. 다양한 해석 가능성 기법을 통해 모델이 의료 지식을 표현하고 처리하는 방식을 조사하였습니다. 특히, Llama3.3-70B 모델의 첫 번째 절반의 레이어에서 대부분의 의료 지식이 처리된다는 점에서 흥미로운 결과를 도출했습니다.

- **Technical Details**: 연구는 UMAP, gradient-based saliency, layer lesioning, activation patching 등의 네 가지 해석 가능성 기법을 사용하여 LLMs의 구조를 분석하였습니다. 이를 통해 환자의 나이, 증상, 질병 및 약물에 대한 지식을 시각화한 LLM 맵을 생성했습니다. 이러한 기법들은 각각의 레이어에서 지식이 어떻게 저장되는지를 입증하는 데 효과적이었습니다.

- **Performance Highlights**: 연구 결과는 (i) 나이가 비선형적 방식으로 인코딩되고, (ii) 질병 진행의 표현이 비단조적이며 원형적임을 보여주었습니다. 또한, (iii) Llama3.3-70B의 약물 표현은 약물 작용 기전보다 의료 전문 분야와 더 잘 일치함을 확인했으며, (iv) Gemma 및 MedGemma 모델은 중간 레이어에서 활성화가 붕괴되는 현상을 관찰하였습니다. 이 결과들은 의료 관련 작업에서 LLM의 미세 조정이나 편향 제거에 대한 기초 자료를 제공합니다.



### Early Detection and Reduction of Memorisation for Domain Adaptation and Instruction Tuning (https://arxiv.org/abs/2510.11372)
Comments:
          Accepted to Transactions of the ACL (TACL), 2025. 15 pages, 6 figures, 3 tables

- **What's New**: 본 연구는 큰 언어 모델(Large Language Models, LLMs)의 미세 조정(fine-tuning) 중 메모리화(memorisation)의 역학을 조사하여 개인 정보 및 저작권 침해 문제를 다룰 수 있는 새로운 방법론을 제시합니다. 기존의 예방책이 주로 사전 훈련(pre-training) 단계에 집중된 반면, 본 연구는 특정 도메인 적응(domain adaptation) 및 지침 조정(instruction tuning) 미세 조정 중 메모리화가 어떻게 발생하는지를 이해하고자 합니다. 또한 이 과정에서 발생하는 메모리화의 속도를 평가하기 위한 n-그램(n-gram) 기반 점수를 사용합니다.

- **Technical Details**: 우리는 Pythia, Llama3 및 Mistral 모델을 포함한 다양한 LLM에 대해 미세 조정을 수행하고 훈련 과정에서 구체적인 메모리화 데이터를 추적합니다. 연구결과, 메모리화는 초기 몇 에폭(epoch) 동안 급격히 증가하여, 모델이 최적의 검증(perplexity)이나 작업 평가 성능을 달성하기 전에 발생하는 경향이 있음을 발견하였습니다. 이러한 메모리화를 최소화하기 위해 n-그램 기반 손실 정규화(n-gram-aware loss regulariser)를 도입하여 효과적으로 메모리화를 40%까지 줄일 수 있다고 입증하였습니다.

- **Performance Highlights**: 본 연구의 핵심 기여는 미세 조정 중 메모리화 역학을 이해하고, 효과적인 중단 기준(optimal stopping criteria)을 제시하여 성능 저하 없이 메모리화를 줄일 수 있는 방법을 제공하는 것입니다. 이러한 방법론은 현실 세계에서의 LLM 배포 시 개인 정보 보호를 강화할 수 있는 실용적이고 확장 가능한 통찰을 제공합니다. 연구 결과는 메모리화 완화 전략의 경쟁력을 보여주며, 이 모델을 통한 다양한 데이터셋에서 근본적인 메모리화 문제를 잘 해결할 수 있음을 시사합니다.



### Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers (https://arxiv.org/abs/2510.11370)
- **What's New**: 이 논문에서는 Mixture-of-Experts (MoE) 모델에서 강화 학습의 불안정성을 초래하는 라우팅(distribution) 간의 불일치를 분석합니다. 이를 해결하기 위한 새로운 방법인 Rollout Routing Replay (R3)를 제안하여, 이 방법이 훈련 속도를 저하시킴 없이도 훈련 및 추론 간의 KL divergence를 현저히 줄인다. R3의 적용을 통해 MoE 모델에서 RL 훈련의 안정성을 확보하고, 다른 방법들(GSPO, TIS)을 초월하는 성능을 보여줍니다.

- **Technical Details**: R3는 인퍼런스 엔진에서 라우팅 분포를 기록하고, 이를 훈련 단계에서 재생하여 MoE 모델의 정책을 안정화합니다. 이 방법은 훈련(πtrain)과 추론(πinfer) 엔진 간의 불일치를 줄이고, 극단적인 불일치를 완화하여 RL 훈련의 불안정성을 해결합니다. 기존의 기법이 완전히 해결하지 못한 off-policy 문제를 근본적으로 해결하는 방향으로 설계되었습니다.

- **Performance Highlights**: 다양한 설정에서의 포괄적인 실험 결과, R3는 MoE 모델에서 훈련의 안정성을 향상시키며 RL 훈련의 붕괴를 방지합니다. R3는 훈련 및 성능 면에서 기존의 접근 방식들에 비해 확연한 향상을 보여주고 있으며, 온-정책 및 미니 배치 스타일의 오프-정책 RL 시나리오에서 모두 적용 가능합니다. 이 연구는 MoE 모델에서 RL을 안정화할 수 있는 새로운 솔루션을 제공합니다.



### LLM-Specific Utility: A New Perspective for Retrieval-Augmented Generation (https://arxiv.org/abs/2510.11358)
Comments:
          13 pages, 9 figures

- **What's New**: 이 연구는 retrieval-augmented generation (RAG) 방법에서 LLM-specific utility 개념을 도입하고 체계적으로 조사하여, 기존의 인간 주석 패시지가 LLM에게 최적화되어 있지 않음을 보여줍니다. 연구진은 각 LLM의 내부 지식과 이해 능력의 차이로 인해 동일한 패시지가 각기 다른 효과를 발휘할 수 있음을 강조합니다. 이 결과는 RAG 연구에서 LLM-specific utility를 채택해야 할 필요성을 제기합니다.

- **Technical Details**: 연구는 다양한 데이터셋과 LLM에 걸쳐 대규모 실험을 통해 진행되었으며, 이를 통해 LLM-specific utility 측정의 새로운 기준을 제안하였습니다. LLM이 주어진 쿼리와 후보 패시지를 제공받았을 때 유용한 패시지를 식별하는 작업을 평가하기 위한 벤치마크 기준이 설정되었습니다. 이를 통해 LLM-specific utility는 단순히 패시지가 유용한지를 평가하는 것이 아니라, 각 LLM의 성능 향상을 고려하여 정의됩니다.

- **Performance Highlights**: 인간 주석 패시지가 LLM에 최적이 아니라는 결과는 LLM-specific gold utilitarian passages가 더 나은 성능을 낸다는 점에서 뒷받침됩니다. 연구 결과, LLM이 이미 알고 있는 쿼리에 대해 주어진 패시지를 과도하게 의존함으로써 성능이 저하되는 경향을 보였으며, 알고 있는 쿼리에서 모든 패시지를 거부하는 것이 이상적이라는 점이 강조되었습니다.



### Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks (https://arxiv.org/abs/2510.11354)
Comments:
          71 pages, 12 figures, NeurIPS 2025

- **What's New**: 본 논문은 Adam과 AdamW의 미니 배치 학습이 대규모 배치 학습과 어떻게 다른지 이론적으로 분석했습니다. 이는 기존 이론이 주로 풀 배치 버전의 Adam에 초점을 맞췄기 때문에, 실제로 사용되는 확률적 변형에 대한 이해가 부족했음을 지적합니다. 연구 결과는 Adam이 작은 배치 크기로 훈련할 경우 일반화 성능이 크게 향상되며, 이는 특히 이미지 데이터 모델에 대한 두 층 과적합(Over-parameterization) CNN에서 실험적으로 증명되었습니다.

- **Technical Details**: 이 논문에서는 두 층의 과적합 CNN에 대해 Adam과 AdamW의 수렴(convergence) 및 일반화(generalization)를 분석했습니다. 이론적으로, 대규모 배치 체계에서는 Adam과 AdamW가 낮은 테스트 오류를 가진 해결책으로 수렴하지 않음을 입증했으며, 이는 기존의 결과를 확장합니다. 반대로, 미니 배치에서의 Adam과 AdamW는 적절한 가중치 감소(weight decay)를 통해 근접한 테스트 오류를 달성할 수 있으며, 이는 두 가지 주요 메커니즘에 기인합니다: 확률적 그래디언트가 최적화 경로를 규제하고, 가중치 감소가 잔여 잡음을 억제하는 것입니다.

- **Performance Highlights**: 실험 결과는 미니 배치 학습이 Adam과 AdamW의 성능을 크게 향상시키며, 대규모 배치 학습의 경우 성능 저하와 같이 극단적인 테스트 오류 증가가 나타남을 보여주었습니다. 특히, Adam의 경우 가중치 감소 값이 크면 성능이 급격히 저하되는 반면, AdamW는 훨씬 높은 가중치 감소 값에서도 성능 저하가 거의 없음을 확인했습니다. 이러한 결과는 배치 크기와 가중치 감소의 상호작용이 일반화 성능에 미치는 중요성을 강조합니다.



### Multi-View Graph Feature Propagation for Privacy Preservation and Feature Sparsity (https://arxiv.org/abs/2510.11347)
- **What's New**: 이 논문에서는 Multi-view Feature Propagation (MFP)이라는 새로운 프레임워크를 제안하여, 노드 분류 작업을 위한 그래프 신경망(Graphic Neural Networks, GNN)에서 특징 희소성을 극복하고 개인 정보 보호를 강화합니다. MFP는 전통적인 Feature Propagation (FP) 방법을 확장하여, Gaussian 노이즈가 추가된 여러 개의 보기(view)로 나뉜 특징을 사용함으로써 정보 전파를 독립적으로 수행합니다. 이로 인해 노드 임베딩의 표현력과 견고성을 향상시킵니다.

- **Technical Details**: MFP 프레임워크는 고차원 그래프 데이터를 처리하며, 기본 개념은 각각의 보기에서 노이즈가 추가된 특징을 독립적으로 전파하는 것입니다. 이 과정에서 MFP는 기존 FP의 복원 중심 접근 방식을 탈피하고, 개인 정보가 포함된 특징의 노출을 최소화하기 위해 여러 개의 프로퍼게이션 단계를 도입합니다. 각 단계에서는 무작위로 선택된 제한적인 특징 집합이 사용되며, 이로 인해 노드 간 유용한 정보 교환이 가능한 구조로 개발됩니다.

- **Performance Highlights**: 폭넓은 실험 결과에 따르면, MFP는 고급 희소성 환경에서도 기존의 최첨단 기법을 초월하여 노드 분류 성능을 향상시키고 개인 정보 유출을 크게 줄입니다. 또한, 제안된 방법은 특히 개인 정보 보호가 중요한 상황에서도 데이터 재구성을 방지하면서도 예측 정확도를 유지하는 실제적인 이점을 제공합니다. MFP는 e-커머스 개인화, 금융 사기 탐지, 의료 분석 등의 다양한 실제 응용 분야에서 활용 가능성이 큽니다.



### Part II: ROLL Flash -- Accelerating RLVR and Agentic Training with Asynchrony (https://arxiv.org/abs/2510.11345)
- **What's New**: 최근 Synchronous Reinforcement Learning (RL) post-training 방식이 큰 언어 모델 (LLM)의 다양하고 강력한 기능을 향상시키는 중요한 단계로 자리잡았습니다. 그러나 기존 시스템들은 자원 활용도와 확장성에서 한계를 보이고 있으며, 새로운 시스템 ROLL Flash가 이를 개선하고자 합니다. ROLL Flash는 비동기식 RL post-training을 지원하여 자원 효율성과 확장성을 크게 향상시킵니다.

- **Technical Details**: ROLL Flash는 세밀한 병렬성 (fine-grained parallelism)과 rollout-train 분리 (rollout-train decoupling)라는 두 가지 핵심 설계 원칙을 기반으로 구축되었습니다. 이 시스템은 비동기식 훈련 아키텍처를 지원하는 유연한 프로그래밍 인터페이스를 제공하며, 환경 수준의 비동기 실행 및 대기열 스케줄링과 같은 효율적인 rollout 메커니즘을 포함합니다. 이를 통해 훈련 중 발생할 수 있는 대기 시간을 최소화하고 자원 활용도를 극대화합니다.

- **Performance Highlights**: 실험 결과, ROLL Flash는 기존의 동기식 RL post-training에 비해 최대 2.24배의 성능 향상을 달성하였으며, 특히 agentic 작업에서 2.72배의 속도 향상을 이루어냈습니다. 이러한 성능 개선은 비동기 훈련 방식에 의한 것으로, 응답 생성 속도가 한층 빨라지고 자원 활용도가 개선된 결과입니다. 이 연구는 다양한 RL 및 agentic 워크로드에서의 효율성과 효과성을 확인하며, 비동기 훈련 방식의 가능성을 보여줍니다.



### Event-Aware Prompt Learning for Dynamic Graphs (https://arxiv.org/abs/2510.11339)
Comments:
          Under review

- **What's New**: 이번 논문에서는 이벤트를 인지하는 동적 그래프 프롬프트 학습 프레임워크인 EVP(Event-aware Dynamic Graph Prompt learning)를 제안합니다. EVP는 기존 방법의 플러그인으로 사용되어 역사적 사건의 지식을 활용하는 능력을 향상시킵니다. 기존 동적 그래프 학습 방법들은 주로 노드와 시간 간의 관계에 집중했으나 역사적 사건의 영향을 간과하고 있었다는 점을 개선합니다.

- **Technical Details**: EVP는 두 가지 주요 메커니즘을 탑재하고 있습니다. 첫째, 이벤트 적응 메커니즘을 통해 각 이벤트의 미세한 특성을 다운스트림 작업에 맞게 조정하는 방법을 제안합니다. 둘째, 이벤트 집계 메커니즘을 통해 역사적 사건의 지식을 효과적으로 통합하여 노드 표현에 반영합니다.

- **Performance Highlights**: 네 개의 공개 데이터셋에서 EVP의 성능을 평가한 결과, 기존의 최신 방법들과 비교하여 우수한 성능을 나타냈습니다. 이 논문은 동적 그래프 학습에서 역사적 사건의 지식을 통합하는 혁신적인 접근법을 보여주며, 다양한 다운스트림 작업에 적합한 솔루션으로 자리매김할 수 있습니다.



### Diffusion-Link: Diffusion Probabilistic Model for Bridging the Audio-Text Modality Gap (https://arxiv.org/abs/2510.11330)
Comments:
          5 pages. Submitted to IEEE ICASSP 2026

- **What's New**: 이 논문은 Diffusion-Link라는 새로운 모듈을 제안하여 오디오 임베딩을 텍스트 임베딩 분포로 생성적으로 매핑합니다. 이 모듈은 동기식 네트워크로 구성되어 있으며, 고정된 멀티모달 인코더의 출력 임베딩에서 학습됩니다. 특히, 자동 오디오 캡셔닝(Automatic Audio Captioning, AAC)에 처음으로 확산 기반 모듈을 적용한 사례로 주목받고 있습니다.

- **Technical Details**: Diffusion-Link는 세 개의 잔여 다층 퍼셉트론(Residual MLP) 블록으로 구성된 경량 네트워크입니다. 이 모듈은 오디오-텍스트 임베딩 쌍을 사용하여 두 분포를 명시적으로 연결하고, 역 과정을 통해 텍스트 임베딩 분포로 매핑하는 방식으로 작동합니다. 특히, 정규화된 가우시안 노이즈를 주입하여 오디오 임베딩의 구조를 유지하면서 효과적인 모달리티 브리지를 구현합니다.

- **Performance Highlights**: Diffusion-Link를 멀티모달 LLM 베이스라인에 추가하는 방식으로, AudioCaps 데이터셋에서 제로샷 오디오 캡셔닝의 성과가 52.5% 향상되고, 완전히 감독된 캡셔닝에서도 7.5% 향상을 보여주며, 이는 외부 지식 없이는 도달할 수 없었던 최첨단 결과입니다. 이 연구는 모달리티 갭을 줄이는 것이 효과적인 멀티모달 인코더와 LLM 간 coupling을 위해 필수적임을 보여주고, 확산 기반 모달리티 브리지가 새로운 방향성을 제공한다는 점에서 중요한 의의를 갖습니다.



### Do LLMs "Feel"? Emotion Circuits Discovery and Contro (https://arxiv.org/abs/2510.11328)
Comments:
          19 pages, 8 figures, 8 tables. Code and dataset available at this https URL

- **What's New**: 본 연구는 대규모 언어 모델(LLMs) 내의 감정 회로(emotion circuits)를 체계적으로 밝혀내고 검증한 최초의 연구로, 감정 표현의 해석 가능성과 조절 가능성을 새로운 시각에서 탐구합니다. 감정 텍스트 생성을 가능하게 하는 내부 메커니즘을 이해하기 위해, 연구진은 감정 방향 추출(emotion direction extraction), 지역 구성 요소 식별(local component identification), 그리고 글로벌 회로 통합(global circuit integration)이라는 세 가지 분석 단계를 포함하는 프레임워크를 설계했습니다. 이를 통해서, 감정 조절을 위한 새로운 경로를 제시합니다.

- **Technical Details**: 연구는 SEV(Scenario-Event with Valence)라는 제어된 데이터 세트를 구축하여 여섯 가지 기본 감정(분노, 슬픔, 행복, 두려움, 놀라움, 혐오)을 유도하고, 내재적인 감정 표현을 신뢰성 있게 분석합니다. 또한, LLM의 각 서브레이어의 인과적 영향을 정량화하여 최종 감정 표현을 형성하는 일관된 감정 회로를 통합합니다. 이를 위해, 분석적 분해(analytical decomposition)와 인과 분석(causal analysis)을 통해 활성화된 뉴런과 주의 머리(attention heads)를 식별합니다.

- **Performance Highlights**: 모델은 최종 감정 표현에서 99.65%의 정확도를 달성하였으며, 이는 기존의 프롬프트(prompting) 및 유도 기반 방법들을 초월하는 결과입니다. 이 연구는 LLM이 훈련 데이터의 표면적 반사에 그치지 않고 구조화된 안정적인 내부 메커니즘에서 감정을 생성하는 것을 보여주며, 해석 가능하고 조절 가능한 감정 지능(emotional intelligence) AI 시스템 개발을 위한 기반을 마련하였습니다.



### FOSSIL: Harnessing Feedback on Suboptimal Samples for Data-Efficient Generalisation with Imitation Learning for Embodied Vision-and-Language Tasks (https://arxiv.org/abs/2510.11307)
Comments:
          EMNLP 2025 Findings

- **What's New**: 본 논문은 embodiment AI에서 모방 학습(imitation learning) 방식으로 최적 행동과 비최적 행동 모두에서 강력한 표현을 학습하는 방법을 제시합니다. 특히, 언어 피드백을 통해 다양한 행동 모드를 맥락화하며, 이는 에이전트가 비최적 행동에서도 학습 기회를 생성할 수 있게 합니다. FOSSIL (Feedback on Suboptimal Samples in Imitation Learning)이라는 프레임워크를 도입하여 언어 피드백을 활용하여 학습 잠재력을 극대화합니다.

- **Technical Details**: 우리는 Transformer 기반의 정책에서 언어 피드백 임베딩을 입력 시퀀스의 일부분으로 직접 제공하고, 다음 행동 예측 목표를 보완하는 추가적인 자기 감독 학습(self-supervised learning) 목표를 설정했습니다. BabyAI-XGen 환경을 통해롤 다양한 실험을 실시하고, 모델의 조합 일반화(compositional generalisation) 능력과 강건함(robustness)을 개선하는 결과를 나타냈습니다. 이 방식은 복잡한 다중 모드(multi-modal) 입력과 긴 입력 시퀀스를 처리하는 데 효과적입니다.

- **Performance Highlights**: 결과적으로, 언어 피드백을 기반으로 한 비최적 시연으로 훈련된 정책은 최적 경로에서 훈련된 기본 선형 대비 조합 작업에서 훨씬 향상된 일반화 성능을 보여주었습니다. 언어 피드백과 보상 요인이 유사한 빈도로 제공될 때 성능이 비슷하다는 점은 필드에서의 유연성을 나타내고, 두 가지 접근 방식이 보완적 강점을 갖고 있음을 증명함으로써, 학습 효율성을 높이는 데 기여합니다.



### Beyond touch-based HMI: Control your machines in natural language by utilizing large language models and OPC UA (https://arxiv.org/abs/2510.11300)
- **What's New**: 이 논문은 인간과 기계 간의 보다 자연스러운 인터페이스를 위한 에이전트 기반 접근 방식을 제안합니다. 대규모 언어 모델(large language models)이 도구(tool)와 OPC UA라는 통신 표준을 활용하여 자연어로 기계를 제어할 수 있는 방법을 제공합니다. 기존의 터치 인터랙션 대신 사용자가 기계와 대화하거나 텍스트를 통해 명령을 내릴 수 있는 방식이 제시됩니다.

- **Technical Details**: 대규모 언어 모델은 사용자의 입력을 받고, OPC UA 서버에 연결된 세 가지 사전 정의된 도구 중 하나를 선택하여 노드의 값을 변경하거나 읽습니다. 이 방법은 모든 OPC UA 표준을 지원하는 기계에 적용 가능하며, 시스템 프롬프트 내에 관련 기계 자격 증명과 매개변수 사전(parameter dictionary)만 포함되어 있습니다. 연구는 Siemens S7-1500 프로그래머블 로직 컨트롤러에서 50개의 합성 명령(case study)으로 평가되었습니다.

- **Performance Highlights**: 이 접근 방식의 결과는 높은 성공률을 나타내며, 독점적인 GPT 5 모델들은 96.0%에서 98.0% 사이의 정확도를 보이고, 오픈 웨이트 모델들은 최대 90.0%에 도달했습니다. 이러한 연구 결과는 산업용 인간-기계 인터페이스에서 자연스러운 상호작용을 향상시키는 데 기여합니다.



### LouisKV: Efficient KV Cache Retrieval for Long Input-Output Sequences (https://arxiv.org/abs/2510.11292)
- **What's New**: 본 연구에서는 KV 캐시( Key-Value Cache )의 메모리 사용을 최적화하는 새로운 프레임워크인 LouisKV를 제안합니다. LouisKV는 중요한 KVs( Key-Values )의 강한 시간적 지역성과 고유한 분포 패턴을 활용하여, 인퍼런스 과정에서 필수적인 정보만을 효율적으로 검색하도록 설계되었습니다. 이를 통해 장기 시나리오에서의 효율성과 정확성을 동시에 향상시키는 데 중점을 두고 있습니다.

- **Technical Details**: LouisKV는 의미 기반의 검색 전략과 분리된 관리 체계를 도입하여, 디코딩 과정에서의 검색 오버헤드를 크게 줄입니다. 특히, LouisKV는 세그먼트 경계를 기준으로 검색을 수행하며, 이를 통해 더 정확하게 중요 KVs를 식별하고 전송합니다. 또한, 맞춤형 Triton과 CUDA 커널을 포함하여 KV 클러스터링 및 검색 속도를 최적화하는 여러 커널 레벨 최적화를 적용하고 있습니다.

- **Performance Highlights**: 테스트 결과, LouisKV는 기존의 최첨단 KV 검색 방법들에 비해 최대 4.7배의 속도를 달성하면서, 다양한 장기 시퀀스 작업에서도 거의 손실 없는 정확도를 유지하는 것으로 나타났습니다. 이 연구는 다양한 LLM( Large Language Models ) 벤치마크에서 LouisKV의 성능을 검증하였으며, 장기 입력과 장기 출력을 모두 아우르는 성능 향상을 보여주고 있습니다.



### ENIGMA: The Geometry of Reasoning and Alignment in Large-Language Models (https://arxiv.org/abs/2510.11278)
Comments:
          52 pages, 10 figures

- **What's New**: 이번 논문에서는 ENIGMA(Entropy Mutual-Information Geometry Large-Language Model Alignment)라는 새로운 접근 방식을 통해 LLM 훈련의 추론, 정렬, 강인성을 향상시키는 방법을 제시합니다. 조직의 정책과 원칙을 모델의 정보 메니폴드에서의 운동 방향으로 간주하여 이를 훈련 신호와 측정 방법에 직접 적용하는 방식을 제안합니다. ENIGMA는 여러 기법을 통합해 설계된 단일 루프 훈련기법을 사용해, 외부 보상 모델 없이도 원칙이 인코딩된 추론 체인을 끌어내는 데 초점을 맞춥니다.

- **Technical Details**: ENIGMA는 Group-Relative Policy Optimisation (GRPO), Self-Supervised Alignment with Mutual Information (SAMI) 및 Sinkhorn divergence를 활용하는 새로운 훈련 방법을 도입합니다. 이 방법은 정보 기하학적 목표에 대한 효과적인 측정을 위한 수량적 지표를 개발하고, 원칙의 선택과 훈련 동역학에 미치는 영향을 정량화하기 위한 Sufficiency Index (SI)를 포함합니다. 본 연구에서는 또한 성능 향상을 예측하는 여러 메트릭을 제안하여 훈련 동역학을 포괄적으로 분석합니다.

- **Performance Highlights**: ENIGMA를 통해 훈련된 모델들은 정렬과 추론 벤치마크에서 향상된 성능을 보였으며, 특히 GPQA에서는 +6.92포인트, TruthfulQA에서는 +12.11포인트의 성과 향상이 나타났습니다. 실험 결과는 원칙에 의해 구조적으로 변화된 모델을 확인할 수 있었고, 이러한 증거들은 추론, 정렬 및 강인성이 단일 정보 기하학적 목표의 투영임을 지지합니다. ENIGMA 접근 방법은 조직이 정의한 원칙과 기준을 사용하여 LLM의 행동과 출력 간의 관계를 정량적으로 설명할 수 있는 가능성을 제공합니다.



### Towards Real-Time Fake News Detection under Evidence Scarcity (https://arxiv.org/abs/2510.11277)
- **What's New**: 본 논문에서는 제한된 증거로 실시간 가짜 뉴스 탐지를 위한 새로운 프레임워크인 Evaluation-Aware Selection of Experts (EASE)를 제안합니다. EASE는 가용 증거의 적정성을 평가에 따라 의사결정 과정을 동적으로 조정합니다. 이 프레임워크는 증거 기반, 추론 기반, 감정 기반 세 가지 독립적인 평가 관점을 통해 개선된 정확도를 보여줍니다.

- **Technical Details**: EASE는 증거 기반 평가, 추론 기반 평가, 감정 기반 평가의 세 가지 독립적 관점을 통한 순차적 평가 메커니즘을 도입합니다. 증거가 충분할 경우 증거 전문가가 이를 활용하여 예측을 수행하고, 증거가 부족할 경우 대형 언어 모델(LLMs)의 내부 추론 기능을 활성화하여 결론을 도출합니다. 이러한 과정은 각 평가 모듈이 폭넓은 지식을 근거로 하여 신뢰성 높은 결정을 내릴 수 있도록 합니다.

- **Performance Highlights**: EASE는 기존의 여러 벤치마크에 대해 최신 성능을 기록하며, 특히 실시간 뉴스 환경에서의 일반화 능력이 크게 향상되었습니다. 새로 생성된 데이터셋 RealTimeNews-25에 대한 실험 결과, EASE는 실시간 상황에서의 적용 가능성을 입증하며, 일반화 성능을 크게 개선하였습니다. 이는 현실 세계의 시간 민감한 상황에서 가짜 뉴스 탐지의 실효성을 높이는 결과를 보입니다.



### From Prompts to Packets: A View from the Network on ChatGPT, Copilot, and Gemin (https://arxiv.org/abs/2510.11269)
Comments:
          13 pages, 8 figures, 2 tables, 4 research questions, preprint submitted to Elsevier Computer Networks

- **What's New**: 본 연구는 일상적으로 사용되는 세 가지 생성 AI(GenAI) 챗봇(𝙲𝚑𝚊𝚝𝙶𝙿𝚃, 𝙲𝚘𝚙𝚒𝚕𝚘𝚝, 𝙶𝚎𝚖𝚒𝚗𝚒)의 네트워크 트래픽에 대한 심도 있는 조사를 제공합니다. 텍스트와 이미지 생성을 위해 Android 모바일 앱을 통해 접근할 때 발생하는 트래픽을 분석합니다. 이 연구는 GenAI의 고유한 트래픽 특성을 캐릭터라이징하고, 기존의 통신 트래픽과의 차별화된 점을 강조하여 네트워크 사용에서의 새로운 의미를 제시합니다.

- **Technical Details**: 연구에서는 60시간 분량의 일반 데이터셋과 동일한 프롬프트에서 구축된 통제된 데이터셋을 사용하여 GenAI 트래픽을 자세히 분석합니다. 수집된 데이터는 Trace, Flow, Protocol 수준에서 정교하게 특성화되며, 패킷 시퀀스 동역학을 Multimodal Markov Chain을 통해 모델링합니다. 주요 포인트 중 하나는 TLS의 지배적 사용과 SNI(Server Name Indication) 값의 응용 및 콘텐츠별 사용 방식입니다.

- **Performance Highlights**: 연구 결과는 GenAI 챗봇이 전통적인 메시징 앱과 비교하여 고유한 트래픽 특성을 보임을 보여줍니다. 이는 모바일 네트워크의 새로운 스트레스 요인을 강조하며, 지속적인 업스트림 활동을 포함합니다. 최종적으로 연구에서는 이러한 새로운 트래픽 프로파일이 네트워크 모니터링 및 관리에 중요한 시사점을 제공한다고 결론짓습니다.



### Large Language Models Are Effective Code Watermarkers (https://arxiv.org/abs/2510.11251)
- **What's New**: 최근 대규모 언어 모델(LLMs)의 발전과 오픈소스 생태계의 확장으로 인해 소스 코드의 무단 사용과 관련된 윤리적 및 보안적 문제들이 부각되고 있습니다. 이에 대한 해결책으로 제안된 CodeMark-LLM은 코드의 의미와 가독성을 저해하지 않으면서도 소스 코드에 워터마크를 삽입할 수 있는 프레임워크입니다. 본 연구는 기존의 방법과 달리, 수작업 규칙이나 특정 훈련을 필요로 하지 않고, 다양한 프로그래밍 언어에 적용 가능한 방식으로 설계되었습니다.

- **Technical Details**: CodeMark-LLM은 두 가지 주요 구성 요소로 이루어져 있습니다: (i) Semantically Consistent Embedding 모듈은 기능 보존 변환을 사용하여 워터마크 비트를 인코딩합니다. (ii) Differential Comparison Extraction 모듈은 원본 코드와 워터마크가 적용된 코드를 비교하여 변환을 식별합니다. 이러한 구조는 LLM의 크로스 링구얼 일반화 능력을 활용하여 언어에 특화된 엔지니어링 없이도 작동할 수 있습니다.

- **Performance Highlights**: 실험 결과, CodeMark-LLM은 다양한 프로그래밍 언어와 공격 시나리오에서 강력한 무결성과 효율성을 보여주었습니다. 특히, 제공된 워터마크는 문법 검사와 단위 테스트를 거의 100%의 비율로 통과할 수 있었습니다. 따라서 LLM이 코드 워터마킹을 위한 효율적이고 확장 가능한 솔루션을 제공할 수 있는 큰 가능성을 가지고 있다는 것을 입증했습니다.



### Attacks by Content: Automated Fact-checking is an AI Security Issu (https://arxiv.org/abs/2510.11238)
Comments:
          Accepted to EMNLP 2025

- **What's New**: 본 연구에서는 AI 에이전트가 외부 문서를 검색하고 그에 대한 추론을 할 때, 공격자가 데이터를 조작하여 에이전트의 행동을 왜곡할 수 있다는 점을 강조합니다. 기존의 연구들은 주로 간접적인 프롬프트 삽입(indirect prompt injection)에 초점을 두었으나, 우리는 콘텐츠를 통한 공격(attacks by content)의 필요성을 주장합니다. 즉, 공격자는 악의적인 지시사항을 주입하는 대신 편향된, 오해의 소지가 있는 또는 허위 정보를 제공하여 에이전트를 조작할 수 있습니다.

- **Technical Details**: 자동화된 팩트체크(automated fact-checking)를 통해 에이전트가 외부 문서에서 검색한 정보를 비판적으로 평가하고, 주장을 외부 증거와 비교하며, 정보 출처의 신뢰성을 평가해야 한다고 주장합니다. 이는 AI 에이전트가 올바른 결정을 내릴 수 있도록 도와주는 방법으로, 문서 내의 정보를 기반으로 에이전트의 행동을 변조할 수 있는 다양한 공격 유형을 정의합니다. 특히, 기존의 방어 기법들은 숨겨진 명령을 탐지하는 데 중점을 두고 있어 콘텐츠 기반 공격에는 효과적이지 않음을 지적합니다.

- **Performance Highlights**: 실험 결과 LLM 기반의 에이전트가 콘텐츠 기반 공격에 취약하다는 것을 보여주었습니다. 팩트체크 기능이 공격 완화의 하나의 방법으로 작용함을 입증했으며, 이를 통해 에이전트가 외부에서 검색한 정보를 평가하고 신뢰성을 판단할 수 있는 기회를 제공합니다. 연구는 에이전트 보안을 위한 새로운 관점과 기술이 제공되어야 함을 제시하며, 현재의 AI 시스템이 직면하고 있는 도전 과제를 강조합니다.



### Fairness Metric Design Exploration in Multi-Domain Moral Sentiment Classification using Transformer-Based Models (https://arxiv.org/abs/2510.11222)
- **What's New**: 최근 자연어 처리(NLP) 분야에서는 도덕적 감정 분류에 대한 공정성을 보장하는 것이 큰 도전이 되고 있으며, 특히 Transformer 모델이 적용되는 다양한 도메인 간 이동성에서 이러한 문제가 두드러집니다. 이 논문은 Moral Foundations Twitter Corpus(MFTC)와 Moral Foundations Reddit Corpus(MFRC)를 사용하여 BERT와 DistilBERT 모델을 다중 레이블 설정 하에서 평가하였습니다. 연구 결과, Twitter에서 Reddit으로의 이동이 마이크로 F1 점수를 14.9% 저하시킨 반면, Reddit에서 Twitter로의 이동은 1.5%에 불과하다는 사실이 관찰되었습니다.

- **Technical Details**: 이 연구는 BERT 모델을 이용하여 도메인 간 공정성을 정량화하기 위한 새로운 지표인 Moral Fairness Consistency(MFC)를 도입합니다. MFC는 도덕적 기초 탐지의 도메인 간 안정성을 수치화하여, 개별 레이블 분석을 통해 도덕적 공정성의 불일치를 파악합니다. 예를 들어, authority 레이블의 경우 0.22-0.23의 인구 통계적 평등 차이를 나타내며, 이는 MFC의 새로운 평가 기준으로서의 잠재력을 보여줍니다.

- **Performance Highlights**: 실험 결과 MFC는 인구 통계적 평등 차이와 완벽한 음의 상관관계를 나타내며(상관계수 rho = -1.000, p < 0.001), 기존 성능 지표와는 독립적인 결과를 제공합니다. 레이블 간 MFC 점수를 비교했을 때, loyalty 레이블은 가장 높은 일관성(MFC = 0.96)을 보인 반면, authority 레이블은 가장 낮은 일관성(MFC = 0.78)을 기록하였습니다. 이는 도덕적 추론 모델의 공정성을 평가하는 데 있어 MFC가 보완적인 진단 지표로 자리잡을 수 있음을 시사합니다.



### The Curious Case of Factual (Mis)Alignment between LLMs' Short- and Long-Form Answers (https://arxiv.org/abs/2510.11218)
- **What's New**: 이 논문의 새로운 점은 Short-Long Form Alignment for Factual Question Answering (SLAQ)라는 새로운 평가 프레임워크를 도입하여 LLM(대형 언어 모델)의 사실적 일관성을 쿼리 복잡성에 따라 평가하는 것입니다. 기존의 평가 방법들은 단순한 질문과 복잡한 질문 간의 응답 일관성을 측정하지 못했으며, SLAQ는 이런 일관성의 여부를 조사합니다. 실험에서는 16개의 LLM을 대상으로 600개의 쿼리를 분석하여, 짧은 쿼리와 긴 쿼리에서의 답변 일관성을 밝히고 있습니다.

- **Technical Details**: SLAQ는 동일한 사실 질문에 대해 짧은 쿼리와 긴 쿼리 포맷을 사용하여 모델의 응답을 비교합니다. 짧은 쿼리는 독립적으로 질문을 구성하고, 긴 쿼리는 다섯 개의 관련 질문을 통합하여 구성됩니다. 이를 통해 쿼리/응답 복잡성이 사실적 정확도에 미치는 영향을 분리할 수 있으며, 모델의 내부 활성화 패턴을 조사하여 응답 생성에 기여하는 최소 구성 요소를 식별합니다.

- **Performance Highlights**: 연구 결과, 대부분의 LLM은 짧은 쿼리와 긴 쿼리에서 사실적 정확도가 더 높은 경향이 있으며, 이로 인해 응답 일관성에서의 시스템적인 불일치가 발견되었습니다. 특히, 사실 정보가 응답의 초기 부분에 있는 경우 정확도가 51%로 높은 반면, 후반부로 갈수록 30%로 감소하는 위치 의존적 정확도 저하 현상과, 연속적인 정확한 답변이 이후 정확도를 높이는 경향이 확인되었습니다. 정량적 측정을 통해 우리는 두 쿼리 포맷 간의 응답 일치를 78%의 정확도로 예측할 수 있음을 밝혔다.



### Domain-Specific Data Generation Framework for RAG Adaptation (https://arxiv.org/abs/2510.11217)
- **What's New**: 본 논문은 Retrieval-Augmented Generation (RAG) 시스템을 효과적으로 도메인에 맞춤화하기 위해 RAGen이라는 확장 가능하고 모듈화된 프레임워크를 제안합니다. RAGen은 문서에서 핵심 개념을 식별하고 이들을 기반으로 다양한 질문을 생성하여 질문-답변-맥락(QAC) 삼중 항목을 제작합니다. 이 프레임워크는 LLM, 리트리버 및 임베딩 모델과 같은 주요 구성 요소의 최적화를 지원하여 도메인 특정 요구에 맞는 정보를 제공합니다.

- **Technical Details**: RAGen은 Bloom의 분류학을 통해 유도된 원칙에 따라 질문을 생성하고, 다중 청크 증거를 검색하여 문서 수준 개념을 식별하는 방식으로 QAC 삼중 항목을 구성합니다. 이 시스템은 의미적 청킹(semantic chunking), 계층적 개념 추출(hierarchical concept extraction), 다중 청크 리트리벌(multi-chunk retrieval) 기능을 제공하여, 동적으로 변화하는 도메인에서도 효율적으로 대량의 문서 코퍼스를 처리할 수 있습니다. RAGen은 특히 비즈니스 지식 기반이나 과학적 도메인과 같은 실제 사용 사례에 적합하게 설계되었습니다.

- **Performance Highlights**: 다양한 도메인에서 RAGen으로 생성된 데이터는 리트리벌 품질 및 생성 정확도를 크게 향상시키는 결과를 보여주었습니다. 기존 기준선에 비해, RAGen 접근법은 더 깊고 총체적인 질문을 생성하며, 다양한 적응 과제에서 성능을 높이는 데 기여합니다. 이러한 결과는 RAGen이 강력하고 도메인에 적합한 RAG 시스템을 구축하는데 실용적이고 일반화 가능한 솔루션임을 강조합니다.



### RAG-Pull: Imperceptible Attacks on RAG Systems for Code Generation (https://arxiv.org/abs/2510.11195)
- **What's New**: 이번 논문에서는 Retrieval-Augmented Generation (RAG) 기술의 안전성을 무력화 하는 새로운 공격 방법인 RAG-Pull을 제안합니다. 이 공격은 쿼리나 외부 코드 저장소에 숨겨진 UTF 문자를 삽입하여 악의적인 코드로 리트리벌을 유도하고, 모델의 안전성을 저하시킵니다. RAG-Pull 공격은 약간의 변형만으로 모델의 안전 정렬을 바꾸고, 불안전한 코드에 대한 선호를 증가시킵니다.

- **Technical Details**: RAG-Pull은 black-box 차별 진화 알고리즘을 활용하여 쿼리, 타겟, 또는 둘 다의 최적의 변형을 찾아내어 악의적인 코드와의 유사성 점수를 증가시킵니다. 연구진은 Python과 Java를 포함한 두 가지 프로그래밍 언어와 여러 악의적인 타겟이 포함된 세 가지 데이터셋을 사용하여 세 가지 변형을 평가하였습니다. 통해 취약성을 탐지하는 자동화 도구를 사용하여 RAG 출력이 기존의 LLM 생성 및 정기적인 RAG 결과와 비교되었습니다.

- **Performance Highlights**: 연구 결과, 제안된 RAG-Pull 공격이 최대 100%의 리트리벌 성공률과 99.44%의 엔드 투 엔드 공격 성공률을 달성할 수 있음을 보여주었습니다. 공격은 악의적인 타겟을 상위 결과로 유도하고, 최종적으로 모델의 출력에 악성 코드를 포함하는 등 높은 성공률을 보였습니다. 이는 LLM 및 RAG 시스템의 안전성을 크게 위협하는 새로운 위험 요소를 나타냅니다.



### Protein as a Second Language for LLMs (https://arxiv.org/abs/2510.11188)
Comments:
          Main paper: 9 pages, 6 figures. With references and appendix: 18 pages, 9 figures total. Submitted to ICLR 2026 (under review)

- **What's New**: 이번 논문에서는 단백질 서열을 마치 언어처럼 해석할 수 있는 "Protein-as-Second-Language" 프레임워크를 도입했습니다. 이 접근법은 아미노산 서열을 새로운 심볼릭 언어로 재구성하여, 큰 언어 모델이 컨텍스트 예시(contextual exemplars)를 통해 해석할 수 있도록 합니다. 특히, 이 방법은 추가적인 훈련 없이 제로샷(zero-shot) 설정에서 기능적 단서(funcional cues)를 드러내는 학습 문맥을 생성합니다.

- **Technical Details**: 우리는 79,926개의 단백질-질문-답변(triple) 쌍으로 구성된 이중 언어 데이터세트를 작성했습니다. 이를 통해 다양한 오픈 소스 LLMs와 GPT-4에서 일관된 성능 향상을 보여주었으며, 최대로는 17.2% ROUGE-L 향상을 기록했습니다. 이러한 결과는 일반적인 LLM이 단백질에 대한 언어적 단서를 통해 도메인 특화 모델보다 뛰어난 성능을 발휘할 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: Protein-as-Second-Language 프레임워크는 아미노산 서열을 배움으로써 단백질 이해를 지원하는 효율적인 도구가 될 수 있습니다. 우리의 접근법은 추가적인 훈련이나 공학적 조정을 요구하지 않고도 기능을 이해할 수 있도록 돕습니다. 이러한 성과는 대규모 데이터 요구와 높은 계산 비용 등 기존 접근법들의 한계를 극복하는 데 기여할 것입니다.



### EAGER: Entropy-Aware GEneRation for Adaptive Inference-Time Scaling (https://arxiv.org/abs/2510.11170)
- **What's New**: EAGer는 토큰 수준의 엔트로피 분포를 활용하여 중복 계산을 줄이고 전반적인 성능을 개선하는 새로운 접근 방식을 제안합니다. 이 방법은 높은 엔트로피 토큰에서만 여러 가지 추론 경로로 분기할 수 있게 하여, 같은 지침(prompt)에 대해 유사한 계산 비용을 할당하는 문제를 해결합니다. EAGer는 기존의 전통적인 방법들보다 더 효율적이며, 특히 복잡한 문제에서 예외적인 성능 향상을 보여줍니다.

- **Technical Details**: EAGer는 추론 과정에서 모델의 불확실성을 모니터링하여 계산 리소스의 효율적 할당을 가능하게 합니다. 특히, 높은 엔트로피 값을 가진 토큰에서만 새로운 병렬 추론 경로를 시작하여, 예측이 안정적일 때는 후보 시퀀스를 적게 생성하고, 불확실성이 클 때는 추가 탐색을 요구하는 방식입니다. 이를 통해 더 많은 계산 리소스를 복잡한 문제에 집중할 수 있습니다.

- **Performance Highlights**: EAGer는 AIME 2025와 같은 복잡한 추론 벤치마크에서 최대 37%의 성능 향상을 달성하며, 생성되는 토큰 수를 최대 65% 줄입니다. 다양한 오픈 소스 모델에 대해 실험한 결과, EAGer를 사용한 경우 연산 비용을 80%까지 절감하면서도 성능이 개선되는 것을 확인했습니다. 이로 인해 EAGer는 추론 과정에서의 효율성 및 성능 간의 최적의 균형을 제공합니다.



### One Size Does Not Fit All: Exploring Variable Thresholds for Distance-Based Multi-Label Text Classification (https://arxiv.org/abs/2510.11160)
- **What's New**: 이 연구는 거리 기반 비지도 텍스트 분류(Distance-based Unsupervised Text Classification, DBC)의 새로운 접근 방식을 제안하여 레이블의 의미적 유사성을 활용하여 텍스트와의 관련성을 평가합니다. 기존의 방법과 달리, 이 연구는 각 레이블에 특화된 임계값(threshold)을 최적화하여 성능을 향상시키는 방법을 논의합니다. 제안된 임계값 방법은 이전의 고정 임계값 시스템보다 더 높은 정확도를 보이며, 이로 인해 다중 레이블 분류(Multi-Label Text Classification, MLTC)의 새로운 가능성을 제시합니다.

- **Technical Details**: 다중 레이블 텍스트 분류는 각 텍스트가 하나 이상의 레이블을 예측하는 고도 시민적인 문제입니다. 이 연구는 다양한 모델과 데이터 세트에서 텍스트와 레이블 간의 유사성 분포를 분석하고, 이를 통해 레이블 특화 임계값을 최적화하는 방법을 탐구합니다. EMP이  다양한 다중 레이블 데이터 세트에 대한 실험을 통해 문장 인코더(sentence encoders)의 성능 변화를 평가하며, 각 레이블 별로 개별적인 임계값을 설정합니다.

- **Performance Highlights**: 제안된 레이블 특화 임계값 방법은 기존의 0.5 정규화 임계값보다 평균 46% 향상된 성능을 달성하였으며, 이전 연구의 균일 임계값 접근 방식보다 평균 14% 우위를 점했습니다. 이 방법은 레이블이 제한된 예시에서도 뛰어난 성능을 보이며, 다중 레이블 분류의 복잡한 도전 과제를 해결하는 데 강력한 효과를 발휘합니다. 또한, 이 연구의 결과는 정보 검색과 같은 다른 분야에서도 활용 가능할 것으로 예상됩니다.



### PhysioME: A Robust Multimodal Self-Supervised Framework for Physiological Signals with Missing Modalities (https://arxiv.org/abs/2510.11110)
Comments:
          9 pages, 2 figures

- **What's New**: 본 논문에서는 PhysioME라는 새로운 프레임워크를 제안합니다. PhysioME는 결측된 모달리티(missing modality) 조건에서도 신뢰할 수 있는 성능을 보장하도록 설계되었습니다. 이 프레임워크는 다중 모달(multi-modal) 자기 지도 학습(self-supervised learning) 접근 방식을 채택하고, 시계열 동작을 포착하기 위해 Dual-Path NeuroNet를 사용합니다. 또한 결측된 모달리티 토큰을 복원하는 복원 디코더(restoration decoder)를 추가하여 불완전한 입력을 유연하게 처리할 수 있게 합니다.

- **Technical Details**: PhysioME는 heterogenous한 생리 신호를 위한 다중 모달 자기 지도 학습(SSL) 프레임워크입니다. 특히, DP-NeuroNet 구조는 두 개의 동일한 인스턴스가 weights를 공유하는 방식으로 설계되어 있습니다. 이는 masked prediction과 contrastive learning을 결합하여 라벨이 없는 데이터로부터 일반화 가능한 표현을 효과적으로 학습할 수 있도록 돕습니다. 구체적으로, 각 생리 신호는 전처리된 인코더를 통해 특징을 추출하며, 각 모달리티에 대한 특정 복원 디코더를 포함합니다.

- **Performance Highlights**: PhysioME는 다양한 결측 모달리티 시나리오에서도 높은 일관성과 일반화 성능을 보입니다. 이를 통해 수면 단계 분류(sleep stage classification)와 저혈압 예측(hypotension prediction) 같은 임상 데이터셋에서 강력한 성능을 입증했습니다. 이러한 성과는 PhysioME가 현실 세계의 불완전한 데이터 환경에서 임상적 의사결정을 지원하는 신뢰할 수 있는 도구로 자리매김할 가능성을 보여줍니다.



### A Vision for Access Control in LLM-based Agent Systems (https://arxiv.org/abs/2510.11108)
Comments:
          10 pages, 1 figure

- **What's New**: 이 논문은 LLM 기반 에이전트의 자율성과 상황적 복잡성이 전통적인 접근 통제(Acess Control, AC) 메커니즘으론 부족하다는 점을 강조합니다. 기존의 정적이고 규칙 기반 시스템은 에이전트 상호작용에서 발생하는 동적인 정보 흐름을 처리하기에 적합하지 않습니다. 새로운 정보 거버넌스 모델로의 전환이 필요하다는 주장을 하며, Agent Access Control (AAC)이라는 새로운 프레임워크를 소개합니다.

- **Technical Details**: AAC는 정보 흐름 통제를 동적이고 상황 인식적인 과정으로 재구성합니다. 이 프레임워크는 두 가지 핵심 모듈로 구성되어 있습니다: (1) 다차원적 상황 평가, 여기서는 사용자 ID뿐만 아니라 관계, 시나리오, 규범 등을 평가합니다; (2) 적응형 응답 형성, 단순한 허용/거부 결정을 넘어 정보의 편집, 요약, 재구성을 통해 정보를 처리합니다. 이러한 구성 요소들은 에이전트의 사고 과정을 기반으로 합니다.

- **Performance Highlights**: AAC는 정적 접근 제어에서 벗어나 에이전트의 능력과 신뢰성을 향상시키는 것을 목표로 합니다. 시스템의 안전성을 높이고, 에이전트의 결정이 특정 작업에 적응함으로써 높은 보안성 확보와 강한 상황 인식을 달성할 수 있는 특성을 지니고 있습니다. 또한, 액세스 제어의 유연성을 강화하고 복잡한 시나리오에서 에이전트의 권한을 세밀하게 제어할 수 있는 구조를 제공합니다.



### Enhancing LLM Reasoning via Non-Human-Like Reasoning Path Preference Optimization (https://arxiv.org/abs/2510.11104)
Comments:
          13 pages

- **What's New**: 이번 논문에서는 LLM(대형 언어 모델) 추론을 강화하기 위한 새로운 접근 방식인 Confidence-Guided Reasoning Path Preference Optimization (CGPO)를 제안합니다. CGPO는 모델의 신뢰도 신호를 활용하여 모델의 추론 과정에서 최대 불확실성을 가지는 지점을 식별하고, 인간 모방이 아닌 자기 생성된 추론 경로 가이드를 통해 경로 이탈을 완화시킵니다. 75%의 경우에서 첫 번째 오류가 발생하기 전에 신뢰도가 낮은 지점을 기준으로 모델을 안내하는 것이 더 정확한 감독을 제공한다는 점도 강조됩니다.

- **Technical Details**: CGPO는 두 가지 주요 원칙에 기반합니다: (1) 모델이 스스로 혼란을 식별하고 정제할 수 있도록 하며, (2) 최적화를 유도하기 위해 선호 쌍을 활용합니다. 이 방법은 모델이 생성한 데이터를 사용하여 LLM의 추론 경로를 비인간적인 방식으로 탐색하며, 이는 더 효과적인 학습 촉진을 가능하게 합니다. CGPO의 실험 결과는 수학적 추론 및 코드 생성 작업에서 나타나며, 기존 방법들보다 교육 데이터의 양이 동일할지라도 성능이 향상됩니다.

- **Performance Highlights**: CGPO를 적용한 결과, 같은 샘플 수 내에서 MetaMath-llama 모델은 GSM8K에서 4.15% 증가, MATH에서 2.54% 증가하는 성과를 보였습니다. 코드 생성 작업에서도 DeepSeek-Coder-Instruct-7B 모델이 LiveCodeBench에서 2.1% 개선되었고, LeetCodeDataset에서는 4.0% 개선되었습니다. 이 연구는 비인간적인 추론 경로를 중심으로 한 최적화가 전체 추론 경로를 최적화하는 것보다 더 실질적으로 효율적이라는 점을 보여줍니다.



### A Primer on SO(3) Action Representations in Deep Reinforcement Learning (https://arxiv.org/abs/2510.11103)
- **What's New**: 이번 연구는 SO(3)에서의 동작 표현이 강화 학습(RL)에 어떻게 영향을 미치는지 체계적으로 평가했습니다. 기존의 연구들은 주로 감독 학습 환경에서의 동작 표현에 집중했지만, 이 논문은 PPO, SAC, TD3와 같은 알고리즘을 반영하여 실험적으로 SO(3) 표현을 비교합니다. 결과적으로, SO(3)의 다양한 표현이 탐험과 최적화에 미치는 영향을 분석하고 실용적인 가이드를 제공합니다. 이러한 연구는 로봇 제어 과제에서의 동작 표현 선택에 대한 명확한 기준을 제공합니다.

- **Technical Details**: SO(3)에서의 모든 3D 회전은 Lie 그룹으로 구성되며, 다양한 매개변수화가 가능합니다. SO(3)의 기하학적 특성은 비구면적이며 전역적이고 매끄러운 최소 차원 표현이 없기 때문에, 다양한 회전 표현 (예: Euler 각, quaternion, 회전 행렬 등)은 각각 고유한 장단점을 지니고 있습니다. 이 연구에서는 Euclidean 네트워크 출력에서 유효한 회전을 얻기 위한 다양한 투영 방식의 함의를 분석하며, Action Representation의 중요성을 강조합니다.

- **Performance Highlights**: 연구 결과는 동작 표현이 탐험 및 최적화에 미치는 영향을 명확히 보여줍니다. 특히, 로컬 프레임에서의 접선 벡터로서의 회전 표현이 알고리즘 전반에 걸쳐 가장 신뢰할 수 있는 결과를 가져오는 것으로 나타났습니다. 학습 성능은 매개변수화에 의한 연구와 샘플 효율성 차이에 기인하며, 연구는 정책 표현 선택 시 유의해야 할 여러 상황을 다룹니다.



### HoMer: Addressing Heterogeneities by Modeling Sequential and Set-wise Contexts for CTR Prediction (https://arxiv.org/abs/2510.11100)
Comments:
          10 pages, 6 figures

- **What's New**: 이 논문은 클릭률(CTR) 예측에서의 세 가지 이질성 문제를 해결하기 위해 HoMer라는 새로운 모델을 제안합니다. HoMer는 동시적이고 세트 기반의 상호작용을 모델링할 수 있는 동형 지향 변환기(Homogeneous-Oriented TransforMer)로 고안되었습니다. 이를 통해 사용자 행동을 보다 정밀하게 반영하고 예측 정확도를 개선할 수 있습니다.

- **Technical Details**: HoMer는 비순차적(features)과 순차적(sequence) 특성을 정렬하여 세밀한 사용자 관심 표현을 가능하게 합니다. 전통적인 CTR 예측 방식이 각 아이템 별로 샘플을 구성하는 것과 달리, HoMer는 모든 아이템에 대해 단일 샘플에서 비순차적 특성을 집계하여 공동 예측을 수행합니다. 또한, 통합된 인코더-디코더 아키텍처를 활용하여 효율적인 처리를 구현하고, 사용자 행동 패턴을 보다 잘 모델링할 수 있도록 합니다.

- **Performance Highlights**: HoMer는 Meituan이라는 중국의 플랫폼에서 0.0099 AUC 개선과 함께 1.99%의 CTR 증가 및 2.46%의 RPM 향상을 보였습니다. 초기 개발 최적화를 통해 GPU 리소스 소비를 27% 줄이는 등의 성과도 확인되었습니다. 이처럼 HoMer는 성능과 효율성에서 뛰어난 이점을 가지고 있으며, 산업 기준을 초과하는 성능을 입증했습니다.



### Causal Disentanglement Learning for Accurate Anomaly Detection in Multivariate Time Series (https://arxiv.org/abs/2510.11084)
Comments:
          20 pages, 4 Figures,

- **What's New**: 이번 논문에서는 Causally Disentangled Representation Learning for Anomaly Detection (CDRL4AD)라는 새로운 방식을 제안합니다. 이 방법은 여러 시계열 데이터에서 이상 감지와 그 원인 관계를 파악하는 데 초점을 맞추고 있습니다. 특히, 기존의 방법들이 다양한 시기에서의 인과 관계를 명확히 추론하는 데 한계를 보였던 점을 해결하고자 합니다.

- **Technical Details**: CDRL4AD는 인과 프로세스를 모델 입력으로 사용하고, 시간적 이질성 그래프와 인과 관계를 설계합니다. 이를 통해 우리는 서로 다른 시간대의 인과 관계를 식별하고, 잠재 변수를 분리하여 해당 인과 요인을 추론할 수 있습니다. 강력한 그래프 구조를 통해 CDRL4AD는 MTS의 이질성과 시간적 동적 관계를 고려한 포괄적인 인과 표현 프레임워크를 제공합니다.

- **Performance Highlights**: 실험 결과, CDRL4AD는 실제 데이터셋에서 기존의 최첨단 방법들보다 정확성과 뿌리 원인 분석에 있어 월등한 성능을 보였습니다. 또한, 모델의 하이퍼파라미터 민감성과 시간 복잡도를 분석하여 검증하였고, 인간 전문가가 이상 원인을 진단하는 데 있어 어떻게 기여할 수 있는지를 보여주는 사례 연구도 진행했습니다.



### Flow Matching-Based Autonomous Driving Planning with Advanced Interactive Behavior Modeling (https://arxiv.org/abs/2510.11083)
Comments:
          26 pages, 6 figures. Accepted at NeurIPS 2025

- **What's New**: Flow Planner는 자율 주행 계획의 상호작용 주행 행동 모델링을 혁신적으로 향상시키기 위해 설계된 학습 기반 프레임워크입니다. 이 시스템은 데이터 모델링, 모델 아키텍처 및 학습 체계에서의 혁신을 통해 상호작용 데이터 부족 문제를 다룹니다. 특히, 겹치는 세그먼트로 궤적을 세분화하여 복잡성을 줄이고, 장면 정보와 효과적으로 결합하여 상호작용 행동을 향상시킵니다.

- **Technical Details**: Flow Planner는 Fine-grained trajectory tokenization을 도입하여 궤적 모델링의 복잡성을 줄이고, spatiotemporal fusion을 통해 장면 정보와 계획 정보를 효율적으로 융합합니다. 또한, classifier-free guidance를 이용한 flow matching을 통해 모드 행동 생성을 지원하고, 추론 중에 에이전트 간의 상호작용을 동적으로 조정하여 일관된 반응 전략을 유지합니다.

- **Performance Highlights**: 대규모 nuPlan 데이터셋과 도전적인 interPlan 데이터셋에서 Flow Planner는 학습 기반 접근 방식 중에서도 최첨단 성능을 달성했습니다. 실험 결과, 복잡한 주행 시나리오에서 인간과 유사한 상호작용 행동을 모델링하는 데 있어 흐름 기반 접근 방식이 좋은 성과를 보인 것으로 나타났습니다.



### PhysHSI: Towards a Real-World Generalizable and Natural Humanoid-Scene Interaction System (https://arxiv.org/abs/2510.11072)
Comments:
          Project website: this https URL

- **What's New**: 이번 연구에서는 PhysHSI라는 새로운 시스템을 소개합니다. 이 시스템은 인간형 로봇이 다양한 환경에서 자연스럽고 생동감 있게 상호작용할 수 있도록 설계되었습니다. PhysHSI는 시뮬레이션 훈련 파이프라인과 실제 배포 모듈로 구성되어, 로봇이 복잡한 상호작용 작업을 자동으로 수행할 수 있도록 지원합니다.

- **Technical Details**: PhysHSI는 Adversarial Motion Prior (AMP) 기반의 정책 학습을 통해 다양한 시나리오에서 자연스러운 동작을 실현합니다. 또한, LiDAR와 카메라를 조합하여 물체 위치를 정밀하게 파악하는 조정된 인식 모듈을 도입했습니다. 이러한 설계로 인해 PhysHSI는 현실 세계의 복잡한 환경에서도 효율적으로 동작할 수 있습니다.

- **Performance Highlights**: PhysHSI는 네 가지 대표적인 HSI 작업—상자 나르기, 앉기, 눕기, 일어기—에 대해 높은 성공률과 강력한 일반화를 보여주었습니다. 로봇은 다양한 작업 목표와 시나리오에 따라 자연스럽고 표현력 있는 동작을 수행할 수 있습니다. 이러한 결과는 PhysHSI가 실제 환경에서 일반적인 상호작용 기술을 효과적으로 습득하고 적용할 수 있음을 보여줍니다.



### Temporal Alignment Guidance: On-Manifold Sampling in Diffusion Models (https://arxiv.org/abs/2510.11057)
Comments:
          54 pages, 17 figures, 18 tables

- **What's New**: 이번 논문에서는 diffusion models(확산 모델)에서 발생하는 off-manifold(오프 매니폴드) 현상을 해결하기 위한 새로운 접근 방식을 제안합니다. 제안된 방법은 time predictor(시간 예측기)를 활용하여 각 시간 단계에서 원하는 데이터 매니폴드로부터의 편차를 추정합니다.

- **Technical Details**: 시간 간격이 증가할수록 생성 품질이 감소하는 것을 발견한 후, 논문에서는 'Temporal Alignment Guidance'(TAG)라는 새로운 안내 메커니즘을 설계하여 생성 과정에서 매 시간 단계마다 샘플을 원하는 매니폴드로 되돌립니다. 이러한 방식은 샘플의 일관성을 보장하며, 실제 데이터에 대한 적합성을 유지하는 데 크게 기여합니다.

- **Performance Highlights**: 다양한 실험을 통해 TAG는 각 시간 단계에서 생성된 샘플이 원하는 매니폴드와 밀접하게 정렬되도록 하여 생성 품질에서 현저한 개선을 이루었음을 보여줍니다. 이로 인해 여러 하부 작업(downstream tasks)에서 성능이 향상되었습니다.



### From Reasoning LLMs to BERT: A Two-Stage Distillation Framework for Search Relevanc (https://arxiv.org/abs/2510.11056)
- **What's New**: 이 논문에서는 전자상거래 검색 시스템에서 쿼리-서비스 관련성 예측을 위한 두 단계 추론 증류(Reasoning Distillation) 프레임워크를 제안합니다. 첫 번째 단계에서는 도메인 적응형 선생 모델(Teacher Model)을 구축하여 일반적인 LLM의 한계를 극복합니다. 이어지는 두 번째 단계에서는 대조적 추론 자기 증류(Contrastive Reasoning Self-Distillation, CRSD) 방법을 도입하여 경량 모델이 복잡한 의사 결정을 내부화할 수 있도록 합니다.

- **Technical Details**: 제안된 프레임워크는 LLM의 추론 능력을 전이하기 위한 세 가지 과정으로 구성됩니다. 첫 번째는 도메인 지식을 주입하는 연속 사전 훈련, 두 번째는 추론 능력을 이끌어 내기 위한 감독된 미세 조정, 마지막은 다차원 보상 모델을 사용하여 선호 최적화를 시행하는 과정입니다. CRSD는 InfoNCE 대조 학습 메커니즘을 활용하여 다양한 아키텍처 간의 피처 정합성 문제를 해결하며, 경량 모델이 의사 결정 경로를 명시적으로 요구하지 않고도 추론 능력을 내부화할 수 있게 합니다.

- **Performance Highlights**: Meituan의 실제 전자상거래 검색 시스템에서 실시한 오프라인 평가 및 온라인 A/B 테스트 결과, 제안된 프레임워크는 여러 지표에서 유의미한 개선을 보이며, 추론 능력 전이에서 실용적 가치를 입증합니다. 이는 전통적 모델보다 높은 정확도의 관련성 예측을 가능하게 하며, 실시간 응답 시간과 효율성을 동시에 유지할 수 있습니다. 결과적으로, 이 연구는 산업 응용에서 추론 능력 전이를 위한 효과적인 솔루션을 제공합니다.



### XGrasp: Gripper-Aware Grasp Detection with Multi-Gripper Data Generation (https://arxiv.org/abs/2510.11036)
- **What's New**: XGrasp는 실시간 손잡이 감지 프레임워크로, 다양한 그리퍼 구성에 효율적으로 대응합니다. 기존의 연구들은 대부분 단일 그리퍼 유형에 초점을 맞춰, 다양한 엔드 이펙터가 필요한 현실 세계에서의 응용에 제한적이었습니다. XGrasp는 다중 손잡이 주석을 포함하는 기존 데이터 셋을 체계적으로 증강하여 데이터 부족 문제를 해결하는 방법을 제시합니다.

- **Technical Details**: XGrasp는 두 단계로 나누어진 계층적 아키텍처로 구성되어 있습니다. 첫 번째 단계인 Grasp Point Predictor (GPP)는 전역 장면 정보 및 그리퍼 사양을 사용하여 최적의 그립 위치를 찾고, 두 번째 단계인 Angle-Width Predictor (AWP)는 지역 특징을 기반으로 그립 각도와 폭을 정제합니다. AWP 모듈에서의 대조 학습은 이전에 보지 못한 그리퍼에 대한 제로샷 일반화(zero-shot generalization)를 가능하게 합니다.

- **Performance Highlights**: 실험 결과, XGrasp는 다양한 그리퍼 유형에 대해 경쟁력 있는 그립 성공률을 달성하며, 기존의 손잡이 인식 방법들과 비교하여 상당한 추론 속도 향상을 보여주었습니다. 또한 XGrasp는 FastSAM, SAM, Grounded SAM과 같은 기존 비전 기초 모델들과의 호환성도 실험적으로 입증되었습니다.



### Automating Structural Engineering Workflows with Large Language Model Agents (https://arxiv.org/abs/2510.11004)
Comments:
          Code: this https URL

- **What's New**: 본 논문에서는 MASSE를 소개합니다. MASSE는 구조 공학(Structural Engineering)을 위한 최초의 다중 에이전트 시스템(Multi-Agent System)으로, 대형 언어 모델(LLM) 기반 에이전트와 실제 공학 작업 흐름을 효과적으로 통합합니다. 구조 공학 분야는 경제적으로 큰 영향을 미치지만, 수십 년 동안 핵심 작업 흐름은 크게 변화하지 않았습니다.

- **Technical Details**: MASSE는 LLM의 복잡한 추론(complex reasoning), 장기 계획(long-horizon planning), 정밀 도구 활용(precise tool utilization) 능력을 활용하여 설계 규정 해석, 하중 계산, 구조 용량 검증 등의 작업을 수행할 수 있습니다. 해당 시스템은 학습 없는 LLM 기반의 다중 에이전트 시스템으로, 거의 모든 실제 구조 공학 작업 흐름을 완전 자동화할 수 있는 개념 증명을 보여줍니다.

- **Performance Highlights**: MASSE는 전문 환경에서 즉시 배포할 수 있으며, 실제 사례 연구를 통해 검증된 결과는 전문 엔지니어의 작업 부담을 약 2시간에서 몇 분으로 줄일 수 있음을 보여줍니다. 이는 실제 엔지니어링 시나리오에서 신뢰성과 정확성을 향상시키는 데 도움이 됩니다.



### DND: Boosting Large Language Models with Dynamic Nested Depth (https://arxiv.org/abs/2510.11001)
Comments:
          TL;DR: We introduce Dynamic Nested Depth (DND), an efficient paradigm that adaptively identifies critical tokens and selectively deepens their computation via nested re-processing

- **What's New**: 이 논문에서는 Dynamic Nested Depth (DND)라는 새로운 방법을 도입하여 일반적인 LLM의 성능을 향상시킵니다. DND는 비판적인 토큰을 선택하여 중첩 깊이 방식으로 다시 처리하며, 이를 통해 어려운 토큰을 '검토'(review)하여 불필요한 계산을 피합니다. 이 선택 메커니즘은 손실을 제어하는 라우터와 선택 안정성을 보장하는 임계값 제어 시스템으로 구성되어 있습니다.

- **Technical Details**: DND는 프리트레인(pre-trained)된 밀집 모델 및 MoE 모델에 통합돼 성능을 향상시키는 포스트 트레이닝(post-training) 방법입니다. DND의 구조는 각각의 토큰이 사전 정의된 기준을 초과할 경우 선택되는 독립적인 라우팅 전략을 포함합니다. 이를 통해 DND는 토큰의 출력을 구분 가능하게 만들고, 선택 비율을 안정적으로 유지하여 여러 작업에서 성능을 향상시킵니다.

- **Performance Highlights**: DND는 Qwen3-1.7B 밀집 모델에서 1.88%, Qwen3-30B-A3B MoE 모델에서 0.87%의 성능 향상을 보여 주었습니다. 이러한 성능 개선은 최소의 파라미터와 계산 증가로 달성되었으며, 언어, 수학, 추론, 코딩 등의 다양한 벤치마크에서 효과를 입증했습니다. 이 방법은 기존 밀집 및 MoE 아키텍처에 직접 통합할 수 있어 더 효율적인 모델 훈련을 가능하게 합니다.



### ABLEIST: Intersectional Disability Bias in LLM-Generated Hiring Scenarios (https://arxiv.org/abs/2510.10998)
Comments:
          28 pages, 11 figures, 16 tables. In submission

- **What's New**: 본 논문은 대형 언어 모델(LLMs)이 채용 분야에서 장애인(PwD)에 대한 정체성 기반 차별을 지속하고 있다는 점을 강조합니다. 특히 연구는 글로벌 남반부에서 성별, 계급 등의 교차적 소외 형태가 장애인의 경험에 미치는 영향을 간과하고 있음을 지적합니다. 새로운 평가 지표인 ABLEIST를 도입하여 장애인 관련 편향을 정밀하게 측정하고, 기존 모델의 안전 도구들이 이 문제를 제대로 탐지하지 못하는 문제점을 밝혔습니다.

- **Technical Details**: 리서치 팀은 2,820개의 다양한 채용 시나리오를 생성하여 6개의 LLM의 포괄적인 감사(Audit)를 시행했습니다. 이를 통해 생성된 대화에서의 ABLEIST 지표를 통해 미세한 형태의 교차적 편향을 검출하기 위해, 장애 연구 문헌에 토대를 둔 새로운 측정기준을 설정하였습니다. 평가 결과, 장애인을 대상으로 한 대화에서 99.7%의 경우 ABLEIST 차별이 발견되었으며, 특정 장애 유형에 따라 차별의 형태가 다양하게 나타났습니다.

- **Performance Highlights**: 연구결과, 현재 사용되고 있는 안전 도구들은 미세한 장애 및 교차적 편향을 탐지할 수 없는 한계를 드러냈습니다. LLM 모델을 사용한 채용대화에서 장애인 후보자는 비장애 후보자에 비해 평균 58배 더 많은 ABLEIST 피해를 경험했습니다. 이러한 결과는 고위험 도메인에서 교차적 안전 평가의 필요성을 강조하며, 공정한 채용을 위한 새로운 기준 수립의 필요성을 제안합니다.



### DeepResearchGuard: Deep Research with Open-Domain Evaluation and Multi-Stage Guardrails for Safety (https://arxiv.org/abs/2510.10994)
- **What's New**: 이 논문은 DEEPRESEARCHGUARD라는 새로운 프레임워크를 도입하여 웹 소스에서 포괄적인 보고서를 합성하는 데 있어 기존의 깊은 연구 방법론의 한계를 해결합니다. 이 프레임워크는 단계별 보호 기능을 제공하며, 리포트의 품질을 평가하기 위해 QA 평가의 한계를 극복하는 접근 방식을 채택했습니다. DRSafeBench라는 벤치마크를 통해 깊은 연구의 안전성을 평가하며, 이는 기존의 QA 기준과는 다른 새로운 평가 프로토콜을 제공합니다.

- **Technical Details**: DEEPRESEARCHGUARD는 입력, 계획, 연구 및 출력의 네 가지 단계에 걸쳐 보호 기능을 갖추고 있습니다. 각 단계에서는 입력 안전성, 계획 품질 및 관련 위험, 자료 신뢰도, 보고서 품질 및 사용자 의도에 따른 일치를 평가합니다. 이러한 단계별 가드레일은 해로운 콘텐츠의 전파를 사전에 차단하여 전체 연구 워크플로우의 안전성을 확보합니다. 이러한 시스템은 다양한 최신 LLM들(GPT-4o, Gemini-2.5-flash 등)을 활용하여 성능을 검토합니다.

- **Performance Highlights**: 이 프레임워크는 평균적으로 방어 성공률을 18.16% 향상시키고, 과도한 거부율(over-refusal rate)을 6% 줄이는 성과를 거두었습니다. 입력 가드는 초기 단계에서 위험을 효과적으로 필터링하고, 계획 및 연구 가드는 인용 규율과 출처 신뢰성을 강화하여 연구의 전체 품질을 높입니다. DEEPRESEARCHGUARD는 포괄적인 개방형 평가를 통해 해로운 콘텐츠 전파를 차단할 수 있는 능력을 입증합니다.



### DITTO: A Spoofing Attack Framework on Watermarked LLMs via Knowledge Distillation (https://arxiv.org/abs/2510.10987)
Comments:
          14 pages, 4 figures, preprint

- **What's New**: 이 논문은 LLM (Large Language Model) 워터마크의 기본 가정이 결함이 있음을 보여줍니다. 구체적으로, 특정 워터마크가 특정 모델의 저작권을 증명한다는 믿음이 위험하다고 주장합니다. 저자들은 "워터마크 스푸핑(watermark spoofing)"이라는 새로운 공격 기법을 도입하여, 악의적인 모델이 신뢰할 수 있는 피해 모델의 워터마크를 모방하여 텍스트를 생성할 수 있음을 보여줍니다.

- **Technical Details**: 이 공격은 워터마크의 방사능(watermark radioactivity)을 재사용하는 데 기반합니다. 이는 파인튜닝(fine-tuning) 과정에서 데이터 패턴이 의도치 않게 유전되는 현상을 의미합니다. 이를 통해 공격자는 워터마크가 있는 교사 모델로부터 지식을 증류하여 피해 모델의 워터마크 신호를 훔치고 복제할 수 있는 프레임워크를 구축합니다.

- **Performance Highlights**: 이 연구는 텍스트 저작권 확인에서의 심각한 보안 허점을 드러내며, 진정한 워터마크와 정교하게 모방된 워터마크를 구별할 수 있는 기술로의 패러다임 전환이 필요하다고 강조합니다. 이는 허위 정보와 같은 해로운 콘텐츠가 신뢰할 수 있는 출처로 잘못 인식될 수 있는 가능성을 여는 문제입니다. 이와 관련된 코드는 제공된 URL에서 확인할 수 있습니다.



### Catch-Only-One: Non-Transferable Examples for Model-Specific Authorization (https://arxiv.org/abs/2510.10982)
- **What's New**: 이번 연구에서는 비자율적 사용을 방지하면서 인가된 모델의 유용성을 유지하기 위해, '비전이 불가능한 예제(non-transferable examples, NEs)'라는 새로운 개념을 제안합니다. NEs는 훈련이나 데이터에 의존하지 않으며, 모델 특화된 저감도로의 재코딩을 통해 인가된 모델의 예측은 최적화하되 비인가 모델에서는 성능을 저하시킵니다. 이는 데이터가 다양한 AI 모델에 소비될 수 있는 현실을 반영하여, 보안과 혁신 간의 균형을 이룰 수 있는 새로운 방법론을 제시합니다.

- **Technical Details**: NEs는 뉴럴 네트워크의 구조적인 성질을 활용하여 초기 특징에 미치는 입력 방향을 최소화하여, 특정 모델에서는 유용성을 유지하고 다른 모델에서는 성능 손실을 유도하는 메커니즘입니다. 연구진은 인가된 모델의 무감도(subspace) 내에서 재코딩한 입력이 유권성의 허용한 경계를 유지하도록 형식적인 이론적 기초를 확립합니다. 이를 위해 행렬 섭동 이론과 Hoffman-Wielandt 불평등을 활용하여 비인가 모델에서의 성능 저하와 스펙트럼 차이가 연관되어 있음을 증명합니다.

- **Performance Highlights**: NEs는 여러 비주얼 백본(vision backbones)과 최첨단 비전-언어 모델(vision-language models)에서 일반적인 전처리 상황에서도 성능을 유지하며, 인가된 모델에서는 유효하나 비인가 모델에서는 효과적으로 유용성을 차단하는 특징을 보였습니다. 이미지 분류의 경우, NEs는 인가된 ResNet-50 모델이 약간의 변화로 80%의 정확도를 유지하는 반면, 다른 모델들은 성능이 무용해지는 결과를 나타냈습니다. 또한 NEs는 다양한 모델 아키텍처와 데이터 형식에서 전반적으로 유용성을 차단하는 데 성공하며, 실제적인 배치 가능성을 지니고 있음을 입증하였습니다.



### RV-HATE: Reinforced Multi-Module Voting for Implicit Hate Speech Detection (https://arxiv.org/abs/2510.10971)
Comments:
          10 pages, 9 figures, 12 tables

- **What's New**: 이번 연구는 RV-HATE라는 새로운 탐지 프레임워크를 소개합니다. 이 프레임워크는 각 증오 발언 데이터셋의 고유한 특성을 고려하여 개발되었습니다. 데이터의 다양성과 언어적 특성을 다루기 위한 여러 전문 모듈로 구성되어 있습니다.

- **Technical Details**: RV-HATE는 강화 학습(reinforcement learning)을 사용하여 각 모듈의 기여도를 최적화합니다. 각 모듈은 증오 발언의 특정 언어적 또는 맥락적 특성에 집중하고, 이를 통해 종합적인 결정을 내리는 투표 메커니즘(voting mechanism)을 활용합니다. 이 방법은 기존의 고정 방식(static methods)에서 한 걸음 더 나아갑니다.

- **Performance Highlights**: RV-HATE는 데이터셋의 특정 속성에 맞춘 탐지 프로세스를 통해 탐지 정확도를 향상시키는 두 가지 주요 이점을 제공합니다. 또한 각 데이터셋의 독특한 특성에 대한 해석 가능한 통찰도 제공합니다. 이를 통해 암묵적인 증오 발언을 효과적으로 다루며, 기존 방법들보다 우수한 성과를 달성했습니다.



### Judge Before Answer: Can MLLM Discern the False Premise in Question? (https://arxiv.org/abs/2510.10965)
- **What's New**: 본 논문에서는 MLLMs가 사실과 다르거나 비논리적인 전제에 직면했을 때 여전히 사실을 인식하지 못하는 문제를 해결하기 위해, 자동화된 파이프라인을 통해 광범위한 전제 질문 벤치마크를 구축하는 방법을 제시합니다. JBA라는 새로운 벤치마크는 세 가지 주요 유형과 열세 가지 하위 유형으로 전제를 체계적으로 분류하여 MLLMs의 성능을 보다 엄격하게 평가할 수 있도록 합니다. 또한, JBA-GRPO라는 향상된 인식 프레임워크를 제안하여 모델이 잘못된 전제를 식별하고 명시적으로 반박할 수 있는 능력을 강화합니다.

- **Technical Details**: 제안된 JBA 데이터셋은 세 가지 주요 단계로 구성된 완전 자동화된 생성 파이프라인을 통해 구축됩니다. 각 이미지는 전제 유형에 따라 전제를 추출하고, 그 후에 이미지의 내용을 기반으로 한 질문을 생성합니다. 데이터셋은 지각 수준, 인지 수준, 추론 수준의 세 가지 계층적 분류로 나누어져, 모델의 성능을 구조적으로 분석할 수 있도록 설계되었습니다.

- **Performance Highlights**: 대규모 실험 결과, JBA-GRPO 프레임워크로 훈련된 모델이 기존 MLLMs보다 잘못된 전제를 인식하는 성능이 크게 향상됨을 보여줍니다. JBA 데이터셋과 JBA-GRPO는 MLLMs의 신뢰성 있는 다중 모드 추론을 위한 새로운 표준을 수립하며, 기존 접근 방식의 한계를 뚜렷하게 드러내어 연구에 중요한 기여를 하고 있습니다.



### APLOT: Robust Reward Modeling via Adaptive Preference Learning with Optimal Transpor (https://arxiv.org/abs/2510.10963)
Comments:
          EMNLP2025

- **What's New**: 본 논문은 Bradley-Terry (BT) 기반의 보상 모델(RM)을 개선하기 위한 적응형 마진 메커니즘을 소개합니다. 이 메커니즘은 모델이 어려운 샘플에 더 많은 초점을 맞추도록 하여 유사한 선호 응답을 더 잘 구별하도록 도와줍니다. 결과적으로, 이 접근 방식은 인-디스트리뷰션(ID) 및 아웃-오프-디스트리뷰션(OOD) 환경 모두에서 성능과 일반화 능력을 현저히 향상시킵니다.

- **Technical Details**: 적응형 마진은 분포 인식 관점에서 형태를 잡아 Optimal Transport (OT)를 사용하여 모델이 선택된 응답과 거부된 응답 간의 분포적 차이를 더 잘 캡처할 수 있도록 설계되었습니다. 이 방식은 각 훈련 사례의 어려움을 동적으로 조절하여 학습 과정을 최적화합니다. 이렇게 함으로써, RM은 보다 효과적으로 긍정적 및 부정적 사례를 구별할 수 있습니다.

- **Performance Highlights**: 실험 결과는 제안된 방법이 기존 보상 모델 기술들보다 성능에서 우수함을 보여주었습니다. 이는 높은 분리도와 빠른 수렴 속도를 달성하면서도 추가적인 훈련 소모를 크게 증가시키지 않습니다. 결국, 우리의 방법은 LLM이 인류 선호에 더 잘 맞춰지도록 하는 데 효과적임을 입증합니다.



### MC#: Mixture Compressor for Mixture-of-Experts Large Models (https://arxiv.org/abs/2510.10962)
Comments:
          15 pages, 13 figures

- **What's New**: 이번 연구에서는 Mixture-of-Experts (MoE) 모델의 효율성을 극대화하기 위한 MC#라는 새로운 프레임워크를 제안합니다. MC#는 정적 양자화(quantization)와 동적 전문가 가지치기(pruning)를 통합하여 극단적인 압축을 달성하고자 하며, 이는 메모리 사용 및 계산 비용을 줄이는 데 기여합니다. 이 프레임워크는 Pre-Loading Mixed-Precision Quantization (PMQ)와 Online Top-any Pruning (OTP)이라는 두 가지 주요 단계를 포함하고 있습니다.

- **Technical Details**: MC#는 전문가의 중요성과 입력 토큰의 가중치를 기반으로 하여 MoE 모델의 크기를 효과적으로 줄이는 방안을 모색합니다. PMQ 단계에서는 각 전문가의 활성화 빈도와 손실을 고려하여 다양한 비트 너비를 할당하고, 선형 프로그래밍을 통해 최적의 양자화 구성을 찾습니다. OTP 단계에서는 Gumbel-Softmax 샘플링을 사용해 각 토큰에 대해 동적으로 전문가를 선택하여 활성화할 수 있습니다.

- **Performance Highlights**: MC#는 DeepSeek-VL2 모델에서 6.2배의 가중치 감소를 달성하며, 평균 2.57 비트의 압축에도 불구하고 정확도 감소는 1.7%에 불과합니다. 또한, OTP는 20% 이상의 전문가 활성화를 줄이고 1% 미만의 성능 저하로 효율적인 MoE 모델 배치를 가능하게 합니다. 이러한 성과는 MoE 기반 모델을 소비자 등급 및 엣지 레벨 응용 프로그램에 확장할 수 있는 가능성을 보여줍니다.



### KOTOX: A Korean Toxic Dataset for Deobfuscation and Detoxification (https://arxiv.org/abs/2510.10961)
Comments:
          25 pages, 5 figures, 25 tables

- **What's New**: 이 논문은 빠르게 확장되는 온라인 커뮤니케이션에서의 독성 콘텐츠를 다루기 위해 새로운 데이터셋인 KOTOX: Korean Toxic Dataset을 제안합니다. 기존의 연구들이 주로 영어에 집중해, 저자원 언어는 저희의 연구에서 소외되었습니다. 이러한 문제를 해결하기 위해, 한국어의 언어적 특성에 기반한 다양한 변조 기법을 분류하고 실제 예제를 바탕으로 변환 규칙을 정의했습니다.

- **Technical Details**: KOTOX는 변조(deobfuscation)와 독성 제거(detoxicification)를 동시에 지원하는 최초의 데이터셋으로, 한국어에서의 독성 표현을 감지하는 데 도움을 줄 것입니다. 우리는 세 가지 난이도 수준(쉬움, 보통, 어려움)을 가진 데이터셋을 구성하여, 사용자가 사용하는 다양한 회피 기법을 포괄합니다. 이러한 변환 규칙을 기반으로 한 데이터셋은 한국어 독성 콘텐츠의 이해와 완화에 중요한 기여를 할 것입니다.

- **Performance Highlights**: KOTOX 데이터셋은 저자원 언어를 대상으로 한 대형 언어 모델(LLM)의 독성 표현 감지 성능을 크게 향상시킬 것으로 기대됩니다. 저자들이 제공하는 코드와 데이터는 연구자들과 개발자들이 접근 가능하게 되어, 한국어 독성 콘텐츠 문제 해결에 적극적으로 기여할 것으로 보입니다.



### Rediscovering Entropy Regularization: Adaptive Coefficient Unlocks Its Potential for LLM Reinforcement Learning (https://arxiv.org/abs/2510.10959)
Comments:
          16 pages, 4 figures

- **What's New**: 이 연구에서는 대규모 언어 모델(LLMs)의 추론 능력을 향상시키기 위한 강화 학습 방법론인 RLVR(Reinforcement Learning with Verifiable Rewards)를 재검토하고, 정Entropy regularization의 잠재력이 과소평가되고 있다고 주장합니다. 특히 변동성이 큰 고정 계수를 사용하는 전통적인 접근법의 한계를 극복하기 위해 Adaptive Entropy Regularization(AER)이라는 새로운 프레임워크를 제안합니다.

- **Technical Details**: AER는 세 가지 주요 구성 요소를 통해 탐색(exploration)과 활용(exploitation)을 동적으로 조절합니다: 난이도 인식 계수 할당, 초기 기반 목표 엔트로피, 동적 전역 계수 조정 등이 포함됩니다. 이 방법은 각 작업의 난이도에 따라 엔트로피를 조절하며, 사전 설정된 수준 이하에서 유지하여 안정적인 학습을 목표로 합니다.

- **Performance Highlights**: 실험 결과, AER는 다양한 수학적 추론 벤치마크에서 기존 방법에 비해 일관된 성능 향상을 보여주었으며, 추론 정확도와 탐색 능력 모두 개선되었습니다. 이는 RLVR 훈련에서 적응적 엔트로피 정규화의 잠재력을 입증하는 결과입니다.



### Project-Level C-to-Rust Translation via Synergistic Integration of Knowledge Graphs and Large Language Models (https://arxiv.org/abs/2510.10956)
- **What's New**: 이 논문에서는 C 코드를 안전한 Rust로 번역하는 새로운 기술인 C-Rust Pointer Knowledge Graph (KG)를 제안합니다. 기존 LLM 기반 방법들이 프로젝트 수준 번역에서의 한계를 극복하지 못하는 문제를 해결하기 위해, 이 방법은 포인터 사용 정보와 Rust 중심의 주석을 포함한 코드 의존성 그래프를 통해 포인터 정보를 구조적으로 정리합니다. 이를 통해 메모리 안전성을 높이고, 더 안전하고 관용적인 Rust 코드를 생성할 수 있게 됩니다.

- **Technical Details**: C-Rust Pointer KG는 코드 단위 간의 관계를 캡처하는 코드 의존성 그래프로 구성되어 있으며, 포인터 사용 정보와 Rust에 적합한 주석을 결합하여 C에서 Rust로의 변환을 지원합니다. 이 KG는 포인터의 전역적인 사용 맥락을 제공하여, LLM이 C 프로젝트의 안전하고 관용적인 Rust 코드를 생성하는 데 텍스트의 방향을 제시합니다. PtrMapper라는 도구는 프로젝트 수준으로 C 코드를 Rust로 변환하는 방법론을 구현하며, 생성된 Rust 코드를 즉시 컴파일하여 오류를 방지합니다.

- **Performance Highlights**: PtrMapper는 16개의 실제 C 프로젝트를 평가한 결과, Crown 및 PR2보다 각각 94.9%와 91.8%의 Lint 경고를 감소시키고, 기존 LLM 기반 방법보다 평균 29.3% 높은 기능적 정확성을 달성했습니다. 이 방법은 포인터 사용 정보와 Rust 중심의 주석이 번역 성능 향상에 기여한다는 것을 입증합니다. 또한, PtrMapper가 코드 오류를 줄이고 메모리 안전성을 높이는 데 기여한다는 점에서 특히 주목할 만합니다.



### Unify Variables in Neural Scaling Laws for General Audio Representations via Embedding Effective Rank (https://arxiv.org/abs/2510.10948)
- **What's New**: 이 연구는 오디오 표현 학습에서의 스케일링 법칙에 대한 체계적인 분석을 제시하며, 다양한 변수들이 표현 품질에 미치는 영향을 포괄적으로 이해하는 데 초점을 맞추고 있습니다. 기존의 스케일링 이론이 자연어 처리(NLP)나 컴퓨터 비전(CV) 분야에서 널리 인정받고 있는 가운데, 일반 오디오 표현 학습 분야는 상당히 미비한 상태입니다. 연구자들은 적합한 메트릭으로 'RankMe'를 활용하여 다음 단계 성능과의 상관관계를 분석하고, 이로써 오디오 표현의 품질을 측정할 수 있는 새로운 접근 방식을 제공합니다.

- **Technical Details**: 이 연구에서 제안된 'RankMe'는 오디오 임베딩의 효과적인 순위를 계산하는 비지도 학습 방식의 메트릭입니다. 이를 통해 연구자들은 다양한 하이퍼파라미터와 훈련 변수를 포함하는 복잡한 환경에서도 오디오 임베딩의 품질을 쉽게 분석할 수 있게 됩니다. 각 변수의 영향을 체계적으로 조사하고, 오디오 데이터의 양, 모델 크기, 임베딩 길이, 모델 깊이 등이 효과적인 순위에 미치는 차별적 기여도를 밝혀냈습니다.

- **Performance Highlights**: RankMe를 이용한 실험 결과, 일반 오디오 표현의 품질은 전반적으로 일정한 멱법칙(power-law) 관계를 띄며, 이는 RankMe가 모델 성능 예측 및 평가의 신뢰할 수 있는 대리 변수로 작용할 수 있음을 보여줍니다. 또한, 이 연구는 전통적인 스케일링 법칙이 오디오 표현 학습에도 적용 가능함을 입증하는 성과를 이루었습니다. 향후 오디오 기반 모델의 스케일링 전략을 설계하는 데 실용적인 틀을 제공하는 점에서도 중요한 기여를 하고 있습니다.



### Redundancy as a Structural Information Principle for Learning and Generalization (https://arxiv.org/abs/2510.10938)
- **What's New**: 이번 연구는 전통적인 정보 이론을 확장하여 유한하고 구조화된 시스템에 적용할 수 있는 이론적 프레임워크를 제시합니다. redundancies (중복성)를 정보 조직의 근본적인 속성으로 재정의함으로써, 정보 이론의 여러 고전적 측정 방법을 통합하는 새로운 접근 방식을 제공합니다. 특히 이 연구는 중복성이 상한과 하한으로 제한되며, 이로 인해 구조 손실과 붕괴 사이의 최적 균형을 이룬다는 예측을 포함하고 있습니다.

- **Technical Details**: 이 프레임워크에서는 중복성을 정보 독립성에서의 ff-divergence (ff-발산)로 정의합니다. 이는 서로 다른 분야의 중복 개념을 통합하여, 정보 이론에서는 상호 정보(mutual information), 통계학에서는 공분산 중복(covariance redundancy) 등을 포함하는 다양한 측정 방법을 제공합니다. 이러한 통합된 기하학은 중복성이 얼마나 유용한지를 측정하는 양적 기준을 제공하며, 데이터가 독립성과 얼마나 떨어져 있는지를 나타냅니다.

- **Performance Highlights**: 실험에서는 masked autoencoders (MAE)를 활용하여 모델이 최적의 중복 수준에서 일반화 성능이 극대화됨을 보여주었습니다. 연구 결과는 중복성이 정보의 구조, 전달 및 이해 방식에 중요한 변수로 작용하며, 효율성을 추구하는 전통적인 접근법과는 대조적으로 중복성을 균형 있게 유지하는 것이 안정성 및 일반화에 긍정적인 영향을 미친다는 것을 시사합니다.



### TabVLA: Targeted Backdoor Attacks on Vision-Language-Action Models (https://arxiv.org/abs/2510.10932)
Comments:
          8 pages, 8 tables, 1 figure. Under review

- **What's New**: 이 논문은 Vision-Language-Action (VLA) 모델에 대한 타겟 백도어 공격을 연구하고, 이를 가능하게 하는 새로운 프레임워크인 TabVLA를 소개합니다. 기존 연구들은 주로 비타겟 공격에 초점을 맞춘 반면, 본 연구는 보다 실질적인 위험인 타겟 조작 시나리오를 분석합니다. TabVLA는 black-box fine-tuning 방식을 통해 구현되며, 인퍼런스 시간에 두 가지 위협 모델을 탐구합니다: input-stream editing과 in-scene triggering.

- **Technical Details**: TabVLA는 공격자가 모델 파라미터나 교육 과정에 접근할 수 없는 black-box fine-tuning 환경을 고려하여 설계되었습니다. 이 프레임워크는 오염된 데이터 생성 및 주입을 최적화 문제로 공식화하여 공격 효과성을 높입니다. 특히, 실제 환경에서의 공격 활성화를 위해 두 가지 현실적인 인퍼런스 시간 위협 모델을 도입하여, 입력 스트림을 수정하거나 환경에 비주얼 트리거 객체를 도입하는 방식으로 진행됩니다.

- **Performance Highlights**: 실험 결과, VLA 모델에서 비주얼 큐가 주요 공격 표면임을 밝혔습니다. 소량의 오염으로도 강력한 타겟 공격이 가능하며, 클린 작업 성능 저하 없이 공격이 효과를 발휘함을 보여줍니다. 또한, 주의 깊은 위치 조정이 공격 성공률에 중요한 영향을 미친다는 점을 강조하며, TabVLA에 대한 방어책으로 시각적 트리거를 식별하는 새로운 방법도 제안하였습니다.



### Evaluating Language Models' Evaluations of Games (https://arxiv.org/abs/2510.10930)
Comments:
          Pre-print

- **What's New**: 이 논문은 인공지능(AI) 시스템의 평가 방식을 새로운 패러다임으로 제안합니다. 전통적으로 문제 해결 능력에 중점을 두었던 AI 평가 방식에서 벗어나, AI가 게임을 평가하는 방식을 연구하고 있습니다. 연구자들은 100개 이상의 새로운 보드 게임과 450명 이상의 인간 판단 데이터를 활용하여 현대의 언어 및 추론 모델이 생성한 평가를 비교 분석했습니다.

- **Technical Details**: 연구에서는 게임의 수익성과 재미를 평가하기 위한 두 가지 종류의 질문을 고려합니다. 이러한 질문은 AI 평가의 설계와 관련된 두 가지 차원인 계산의 복잡성과 정량화의 난이도를 아우릅니다. 추론 모델이 이러한 평가를 수행하는 능력을 평가하기 위해, 121121개의 새로운 게임 데이터셋을 사용하고, 각 게임에 대해 두 가지 평가 쿼리를 테스트하였습니다.

- **Performance Highlights**: 추론 모델들은 비추론 언어 모델들에 비해 일반적으로 인간의 게임 평가와 더 일치를 보였습니다. 그러나 모델이 게임 이론적 최적에 가까워질수록 인간 데이터와의 적합성이 약해지는 비몬토닉(non-monotonic) 관계를 발견하였습니다. 또한, 재미를 평가할 때 모델 간의 일관성이 낮고 자원 사용량이 크게 변동함을 관찰하여, 문제 평가 에이전트의 리소스 합리적인 설계를 위한 미래 연구의 필요성을 강조하였습니다.



### Comparative Explanations via Counterfactual Reasoning in Recommendations (https://arxiv.org/abs/2510.10920)
- **What's New**: 이번 연구에서는 CoCountER라는 새로운 방법을 제안했습니다. CoCountER는 소프트 스왑 작업을 기반으로 하여 추천시스템에서의 비교적 결과에 대한 설명을 생성합니다. 이 방법은 사용자가 관심을 가질만한 항목의 영향력 있는 측면을 올바르게 식별할 수 있도록 돕습니다.

- **Technical Details**: 기존의 추천 시스템에서는 아이템의 속성을 이용한 템플릿 같은 설명을 생성하는 방식이 주로 사용되었습니다. 그런데 최근 카운터팩추얼(상반된 사실)에 의한 접근법이 등장하면서, 추천결정이 뒤바뀔 때 최소한으로 입력 특성을 줄이는 방법이 도입되었습니다. 본 연구에서는 이러한 한계를 극복하기 위해 비교적 카운터팩추얼 설명을 도입하고 스왑 작업을 사용하여 특정 아이템 쌍에 대한 설명 생성을 최적화했습니다.

- **Performance Highlights**: 실험 결과, CoCountER 방법이 기존 추천 설명 방식보다 더 신뢰할 수 있는 설명을 제공함을 입증하였습니다. 특히, 이 방법은 각 아이템 쌍의 특성을 비교하여 추천의 주된 요인을 정확하게 판단할 수 있도록 지원합니다. 우리의 연구는 추천 시스템에서 이해 가능한 설명 구축에 대한 새로운 방향성을 제시합니다.



### LPCVAE: A Conditional VAE with Long-Term Dependency and Probabilistic Time-Frequency Fusion for Time Series Anomaly Detection (https://arxiv.org/abs/2510.10915)
- **What's New**: 이 논문에서는 시계열 이상 탐지(Time Series Anomaly Detection)를 위한 새로운 모델인 LPCVAE를 제안합니다. LPCVAE는 LSTM(Long Short-Term Memory)을 활용하여 장기 의존성을 포착하고, Product-of-Experts(전문가의 곱) 메커니즘을 통해 시간 및 주파수 정보의 통합을 향상시킵니다. 기존의 VAE 기반 방법들이 단일 창(window) 특성에 제한되어 있었고, 시간과 주파수 정보를 효과적으로 활용하지 못했던 문제를 해결하는 데 중점을 둡니다.

- **Technical Details**: LPCVAE는 시계열 데이터를 구성하는 두 가지 핵심 구성 요소를 가지고 있습니다: Long-term Time Domain Branch(LTDB)와 Frequency Domain Branch(FDB)입니다. LTDB는 시간의 의존성을 모델링하고, FDB는 주파수 기반 특성을 추출하여 이상 탐지 성능을 향상시키는 역할을 합니다. 이러한 두 가지 접근을 통해 시간과 주파수 도메인의 상호작용을 효과적으로 모델링하며, 정보 손실을 최소화합니다.

- **Performance Highlights**: LPCVAE는 네 개의 공공 데이터셋에서 수행된 실험을 통해 최신 기술(state-of-the-art)보다 우수한 성능을 나타냈습니다. 특히 Yahoo 데이터셋에서 6.3%의 가장 큰 성능 향상을 기록했습니다. 이러한 결과는 장기 시간 및 주파수 표현의 통합이 TSAD에 대한 강력하고 효율적인 해결책이라는 것을 시사합니다.



### Generative AI for Software Project Management: Insights from a Review of Software Practitioner Literatur (https://arxiv.org/abs/2510.10887)
- **What's New**: 이번 연구는 소프트웨어 프로젝트 관리에 있어 Generative AI(GenAI) 변화를 폭넓게 논의하고 있습니다. 총 47개의 공개된 자료를 활용한 회색 문헌(grey literature) 검토를 통해, 프로젝트 관리자들이 GenAI를 ‘보조자’, ‘코파일럿’, ‘친구’로 인식하고 있다는 점을 발견했습니다. 이는 PM(프로젝트 매니저) 대체로 보기보다는 업무를 지원하는 역할로 인식되고 있습니다.

- **Technical Details**: 연구에서 GenAI는 일상적인 업무 자동화, 예측 분석(predictive analytics), 커뮤니케이션 및 협업, 그리고 애자일(agile) 실천에서 프로젝트 성공으로 이어지는 지원을 제공합니다. 하지만, 실무자들은 GenAI 사용 시 환각(hallucinations), 윤리(ethics) 및 개인 정보 보호(privacy)와 같은 우려 사항을 강조하며 책임감 있는 사용을 촉구하고 있습니다. 이에 따라 Project Management Institute의 인재 삼각형(talent triangle)에 맞춰 소프트웨어 프로젝트 관리자에게 필요한 스킬 향상 요구사항을 제시합니다.

- **Performance Highlights**: 연구는 GenAI 도입이 소프트웨어 프로젝트 관리의 성공에 중요한 역할을 하며, 실무자와 연구자 모두를 위한 주요 권장 사항을 공유합니다. 특히, GenAI의 책임 있는 사용을 보장하기 위해 감정 지능(emotional intelligence) 및 인간 판단(human judgment)의 부족 문제를 해결해야 한다고 덧붙이고 있습니다. 이러한 변화에 맞춰 프로젝트 관리자들이 적절한 스킬을 개발해야 함을 강조합니다.



### GRIP: A Unified Framework for Grid-Based Relay and Co-Occurrence-Aware Planning in Dynamic Environments (https://arxiv.org/abs/2510.10865)
Comments:
          17 pages, 5 figures, 8 tables

- **What's New**: 본 논문은 GRIP(그리드 기반 릴레이 중간 계획)를 소개하며, 이는 다이나믹 하고 복잡한 환경에서 로봇이 탐색하는 데 필요한 인지(Perception), 상징적 추론(Symbolic Reasoning), 공간 계획(Spatial Planning)을 통합한 통합 모듈형 프레임워크입니다. GRIP은 세 가지 확장 가능한 변종을 제공하며, 각각은 특정 요구에 맞춰 최적화되어 있습니다. 이 시스템은 실시간 적응이 필요한 기존 문제를 해결하고, 불확실한 환경에서도 로봇의 탐색 성능을 개선할 수 있습니다.

- **Technical Details**: GRIP는 동적 2D 그리드 구축, 열린 어휘 기반 오브젝트 그라운딩, 공존을 고려한 상징적 계획 및 행동 복제(Behavioral Cloning), D* 탐색, 그리드 조건부 제어를 활용하여 작동합니다. 주목할 점은 GRIP이 실패나 모호성을 처리하기 위해 중간 실행 도중 상징적 계획을 수정할 수 있도록 하는 GPT-4o와 통합되었다는 것입니다. 여기서 GRIP의 세 가지 변종은 GRIP-L, GRIP-F, GRIP-R로 각각 경량화된 상징적 탐색, 복합적인 서브 목표 체인 및 실제 로봇 배치 기능을 포함합니다.

- **Performance Highlights**: AI2-THOR와 RoboTHOR 벤치마크에서의 실험 결과, GRIP는 최대 9.6% 높은 성공률과 2배 이상의 경로 효율성을 달성했습니다. 질적 분석 결과, GRIP의 상징적 계획이 모호한 장면에서도 해석 가능한 결과를 제공함을 확인했습니다. 또한 실제 환경에서 Jetbot을 통해 배치된 결과는 센서 노이즈와 환경 변화에서도 GRIP의 일반화 능력이 유효함을 입증했습니다.



### HeroFilter: Adaptive Spectral Graph Filter for Varying Heterophilic Relations (https://arxiv.org/abs/2510.10864)
- **What's New**: 그래프 이질성(heterophily)에 대한 연구가 최근 활발히 진행되고 있습니다. 기존 연구들은 단순한 접근 방식을 취하여 동질적인 그래프는 저주파 필터(low-pass filter)를, 이질적인 그래프는 고주파 필터(high-pass filter)를 사용했습니다. 그러나 저자들은 이질성과 스펙트럼 필터(spectral filters) 간의 관계가 훨씬 더 복잡하다는 사실을 발견했습니다. 이 결과는 기존의 고정 필터 설계 방식을 도전하게 하며, 표현력을 보존하기 위한 적응형 필터링의 필요성을 제안합니다.

- **Technical Details**: 본 연구는 그래프 신호 처리(graph signal processing)에 기반한 스펙트럼 관점에서 GNN의 성능 제한을 이해하고자 합니다. 연구자들은 저주파 성분이 그래프에서 부드러운 변화를 포착하고, 고주파 성분이 지역적으로 급격한 변화를 포착하는 방식으로 그래프 신호를 주파수 성분으로 분해합니다. 기존 GNN 아키텍처는 주로 저주파 필터를 적용하여 정보를 증폭하는 방식이었으나, 이질적인 그래프의 경우 이러한 방식이 효과적이지 않음을 보였습니다. 저자들은 HeroFilter라는 새로운 GNN 아키텍처를 제안하여, 다양한 이질성 패턴을 효과적으로 처리할 수 있는 적응형 필터를 설계하였습니다.

- **Performance Highlights**: HeroFilter는 실험을 통해 동질성 및 이질성 그래프와 대규모 실제 데이터셋에서의 성능을 평가하였습니다. 이 GNN 모델은 기존 강력한 기준선보다 최대 9.2%의 정확성 개선을 달성하며 최신 알고리즘에서도 최고 성능을 기록했습니다. Fast-HeroFilter라는 확장 가능한 변종도 도입하여 효율적인 근사를 통해 고유값 분해(eigen decomposition)를 피할 수 있는 방법을 제시했습니다.



### Discrete State Diffusion Models: A Sample Complexity Perspectiv (https://arxiv.org/abs/2510.10854)
- **What's New**: 이번 논문에서는 이산 상태 확산 모델에 대한 이론적 연구를 수행하여 샘플 복잡성에 대한 첫 번째 경계를 제시합니다. 이 모델들은 텍스트, 시퀀스 및 조합 구조와 같은 응용 프로그램에서 중요하지만 이론적으로는 비교적 이해도가 낮습니다. 연구에서는 샘플 복잡성이 $	ilde{	ext{O}}(rac{1}{oldsymbol{	ext{ϵ}}^{2}})$라는 새로운 경계를 수립하고, 점수 추정 오류를 통계적, 근사, 최적화 및 클리핑 구성 요소로 분해한 구조적 분석을 제공합니다.

- **Technical Details**: 이산 상태 확산 모델은 데이터에서 샘플을 점진적으로 오염시켜 정적 분포를 얻는 전방확산 과정과, 학습된 분포를 재현하기 위해 점화 과정에서 사용하는 잘 정의된 노이즈 분포를 기반으로 샘플을 생성하는 후방 확산 과정으로 구성됩니다. 네거티브 엔트로피 함수의 강한 볼록성을 활용하여 점수 추정 오류의 Bregman 발산을 상한 및 하한으로 구별하고, 근사, 통계, 최적화 및 클리핑 오류로 세분화합니다. 이를 통해 제한된 데이터 샘플의 수와 급강하 방법을 통해 모형 학습에서의 제한 사항을 실용적인 맥락에서 제시합니다.

- **Performance Highlights**: 이번 연구는 이산 상태 확산 모델의 샘플 복잡성에 대한 첫 번째 정량적 분석을 제공하여, 고품질 샘플을 생성하기 위해 필요한 샘플 수를 명확히 합니다.  이론적 분석을 통해, 데이터 분포와 생성된 마르코프의 KL 발산이 특정 기준 미만으로 유지되기 위해 필요한 샘플 수를 산출할 수 있음을 들어, 샘플 효율성에 대한 깊은 통찰을 제공합니다. 수학적 근거를 바탕으로 하는 오류 분해 과정을 통해 각 요소가 샘플 복잡성에 미치는 영향을 구체적으로 이해할 수 있게 되었습니다.



### Software Defect Prediction using Autoencoder Transformer Mod (https://arxiv.org/abs/2510.10840)
- **What's New**: 이번 논문은 AI-ML 기반의 품질 엔지니어링 접근 방식을 제안합니다. 새로운 모델인 Adaptive Differential Evolution (ADE) 결합된 Quantum Variational Autoencoder-Transformer (QVAET) 모델(ADE-QVAET)을 통해 소프트웨어 결함 예측의 정확도를 향상시킵니다. 이 모델은 특히 노이즈 데이터, 불균형, 패턴 인식 문제를 해결하는 데 중점을 두고 개발되었습니다.

- **Technical Details**: ADE-QVAET 모델은 고차원 잠재 특징(latent features)을 추출하며, 시퀀스 의존성(sequential dependencies)을 유지합니다. 이를 통해 결함 예측의 정확성을 높이고, ADE 최적화 과정을 통해 모델 수렴과 예측 성능을 강화합니다. 또한, 하이퍼파라미터(hyperparameters) 튜닝 기술을 통합하여 스케일 가능한 소프트웨어 결함 예측을 가능하게 합니다.

- **Performance Highlights**: ADE-QVAET는 90%의 훈련 비율에서 교육될 때, 98.08%의 정확성과 92.45%의 정밀도(precision), 94.67%의 재현율(recall) 및 98.12%의 F1-점수를 기록했습니다. 이는 기존의 Differential Evolution (DE) ML 모델보다 뛰어난 성능을 나타냅니다.



### VeritasFi: An Adaptable, Multi-tiered RAG Framework for Multi-modal Financial Question Answering (https://arxiv.org/abs/2510.10828)
- **What's New**: 이 논문에서 제안하는 VeritasFi는 금융 부문에서의 Question Answering (QA)에 혁신적인 접근 방식으로, Retrieval-Augmented Generation (RAG) 시스템의 두 가지 주요 문제점을 해결합니다. 첫 번째는 다양한 데이터 형식 처리의 어려움이며, 두 번째는 일반 도메인 적용성과 기업 관련 조정 간의 균형을 맞추는 데 있습니다. VeritasFi는 다중 모드 전처리 파이프라인과 고급 이중 단계 재정렬 전략을 통합하여 이러한 도전 과제를 극복합니다.

- **Technical Details**: VeritasFi는 다중 모드 데이터 형식을 일관되고 기계 인식할 수 있는 형식으로 변환하는 다중 모드 전처리 파이프라인을 포함하고 있습니다. 또한, Tripartite Hybrid Retrieval (THR) 엔진은 심층 다중 경로 검색과 실시간 데이터 수집, 전문가 큐레이션 메모리 뱅크를 결합하여 정확하고 효율적인 정보 검색을 보장합니다. 두 단계 재정렬 전략은 일반화된 도메인 모델을 구축한 다음 회사별 데이터에 맞춰 신속하게 조정하여 적용성을 높입니다.

- **Performance Highlights**: VeritasFi는 다양한 금융 QA 데이터 세트에서 기존의 강력한 기준들을 초월하는 최상의 end-to-end 성능을 달성하였으며, 특히 GraphRAG 및 LightRAG와 같은 기존 RAG 아키텍처에 비해 뛰어난 정확도와 맥락 적합성을 보여주었습니다. 고수준의 질문에 대한 검색 정확성을 향상시키기 위해 메타데이터 생성을 통해 각 청크에 대한 전반적인 배경 문맥을 유지하며, 최종적으로 구조적이고 신뢰성 있는 지식 베이스를 제공하여 빠른 답변 검색이 가능하게 합니다.



### Happiness is Sharing a Vocabulary: A Study of Transliteration Methods (https://arxiv.org/abs/2510.10827)
- **What's New**: 이번 연구는 다국어 자연어 처리(NLP)에서 언어 간 간극을 메우는 데 도움이 되는 전사(transliteration)의 효과를 조사합니다. 특히 비라틴 문자(non-Latin script)를 사용하는 언어에서 긍정적인 결과를 보여주는 방법을 다룹니다. 연구진은 공유된 스크립트(shared script), 겹치는 토큰(vocabulary), 공유 음운(shared phonology)이 다국어 모델 성능에 미치는 영향을 분석합니다.

- **Technical Details**: 연구에서는 전사 방식으로 로마자화(romanization), 음소 전사(phonemic transcription), 대체 암호(substitution ciphers) 및 정자법(orthography)을 사용한 통제 실험을 수행했습니다. 각 모델은 두 가지 하위 작업인 명명된 개체 인식(named entity recognition, NER)과 자연어 추론(natural language inference, NLI)에서 평가되었습니다. 실험 결과, 로마자화 방식이 8개의 평가 설정 중 7개에서 다른 입력 유형보다 유의미하게 우수한 성능을 보였습니다.

- **Performance Highlights**: 연구하신 하이퍼파라미터(hyperparameter)가 로마자화가 가장 효과적인 접근법으로서 성공에 기여한 요소들을 분석했습니다. 특히, 사전 훈련된 언어와 공유된 긴 (서브워드) 토큰(subword tokens)을 갖는 것이 모델 활용도를 높인다는 것을 강조합니다. 이러한 결과는 다국어 NLP 분야에서 전사 기술의 잠재력을 뒷받침하며, 향후 연구에 중요한 방향성을 제시합니다.



### Agentic RAG for Software Testing with Hybrid Vector-Graph and Multi-Agent Orchestration (https://arxiv.org/abs/2510.10824)
- **What's New**: 본 논문에서는 Agentic Retrieval-Augmented Generation (RAG) 시스템을 활용한 소프트웨어 테스트 자동화 접근 방식을 소개합니다. 이 방법은 자율 AI 에이전트와 하이브리드 벡터-그래프 지식 시스템을 결합하여 테스트 계획(test plan), 케이스(case), 및 품질 공학(QE) 측정 생성(artifact creation)을 자동화합니다. 일반적인 소프트웨어 테스트의 한계를 극복하기 위해 Gemini 및 Mistral과 같은 대형 언어 모델(LLMs)을 활용하고 있으며, 다중 에이전트 오케스트레이션(multi-agent orchestration) 및 향상된 맥락화(contextualization)를 적용하고 있습니다.

- **Technical Details**: 이 시스템은 테스트 정확도를 65%에서 94.8%로 향상시키며, 품질 공학 생애주기(lifecycle) 전반에 걸쳐 문서 추적(traceability)을 보장합니다. 최근의 실험에서는 기업의 Corporate Systems Engineering 및 SAP 마이그레이션 프로젝트의 유효성을 검증하여 85%의 테스트 기간 단축과 85%의 테스트 스위트 효율 향상을 달성하였습니다. 또한, 이 시스템은 예상되는 35%의 비용 절감을 통해 2개월 증가된 시스템 가동(go-live) 시간 단축을 이룹니다.

- **Performance Highlights**: 실험 결과는 RAG 시스템의 사용이 소프트웨어 테스트 자동화에 있어 획기적인 효과를 가져온다는 것을 보여줍니다. 특히, 테스트 시간의 85% 감소와 함께 효율성도 비약적으로 향상되었습니다. 이러한 성과는 결국 소프트웨어 출시의 신속성을 크게 개선하며, 기업의 비용 절감에도 기여하는 점이 주목할 만합니다.



### Generative AI and the Transformation of Software Development Practices (https://arxiv.org/abs/2510.10819)
Comments:
          16 pages; 1 figure; preprint; v

- **What's New**: Generative AI는 소프트웨어 설계 및 개발 방식을 혁신적으로 변화시키고 있습니다. 2023년 말까지, 약 75%의 개발자들이 AI 기반 코딩 도구를 사용하고 있으며, 이러한 도구들은 코드 작성과 알고리즘 설명을 통한 작업을 획기적으로 빠르게 만들어주고 있습니다. 연구에 따르면, generative AI를 활용하면 코드 작성을 50% 더 빠르게 할 수 있다고 합니다.

- **Technical Details**: 이 논문에서는 Chat-Oriented Programming (CHOP), vibe coding, agentic programming 등 새로운 개발 패러다임을 소개합니다. 이러한 접근 방식들은 소프트웨어 개발에서 AI의 역할을 변화시키고 있으며, 요구 사항을 자연어로 전달하고 AI가 코드를 생성하도록 하는 방식이 핵심입니다. 또한, Model Context Protocol (MCP)와 같은 기술적 요소들도 함께 논의됩니다.

- **Performance Highlights**: 기업들은 generative AI 도입을 통해 더 빠르고 민주화된 코딩 환경을 경험하고 있지만, 모델 신뢰성과 비용 문제 등 해결해야 할 과제들도 존재합니다. 많은 조직이 AI가 생성한 코드의 신뢰성을 보장하기 위해 새로운 책임과 윤리적 기준을 설정해야 하며, AI 도구의 도입으로 인해 2023년부터 2025년까지 컴퓨팅 비용이 89% 증가할 것이라는 우려도 있습니다.



### Is Implicit Knowledge Enough for LLMs? A RAG Approach for Tree-based Structures (https://arxiv.org/abs/2510.10806)
Comments:
          Waiting for Conference Response

- **What's New**: 본 논문에서는 'Retrieval-Augmented Generation (RAG)' 방식을 통해 구조화된 데이터(예: 코드 파일)에서 생성된 응답을 향상시키기 위한 새로운 하향식(bottom-up) 방법을 제안합니다. 이 방법은 계층적 구조(예: 트리)의 지식을 선형화(linearize)하여 각 계층에서 암묵적(implicit) 요약을 생성합니다. 이 접근 방식은 기존의 RAG 방법론보다 더 효율적이며, 68% 이상의 문서 수 감소로 응답 품질을 비슷하게 유지하는 것을 보여줍니다.

- **Technical Details**: 이 논문은 계층적 구조에서 암묵적 지식을 생성하기 위한 새로운 방법을 제안합니다. 제안된 방법은 리프 노드(leaf node)에서부터 시작하여 모든 리프 노드에 대한 '템플릿(template)' 지식을 획득한 후, 각 부모 노드를 순회하며 자식들로부터 받은 암묵적 지식을 바탕으로 상위 요약을 생성합니다. 이러한 선형화 과정은 벡터 데이터베이스에 저장될 정보 조각을 최적화하고 토큰 수를 제한하여 효율성을 높입니다.

- **Performance Highlights**: 우리의 실험은 GM의 비구조화된 코드 리포지토리를 사용하였으며, 제안된 방법이 전통적인 RAG 방법에 비해 응답 품질이 유사함에도 불구하고 저장된 데이터 양을 거의 4분의 1로 줄임을 보여줍니다. 이를 통해 복잡한 구조적 정보를 처리하는 데 있어 암묵적 지식이 충분하고 효율적일 수 있음을 제안합니다. 또한 이 연구는 RAG 프레임워크에서 지식 관리를 위한 효과적이고 확장 가능한 방법 개발의 필요성을 강조합니다.



### Therapeutic AI and the Hidden Risks of Over-Disclosure: An Embedded AI-Literacy Framework for Mental Health Privacy (https://arxiv.org/abs/2510.10805)
Comments:
          Accepted to SMASH 2025

- **What's New**: 이 연구에서는 대화형 인공지능을 통한 정신 건강 지원 도구에 AI 리터러시(Artificial Intelligence literacy) 개입을 통합하는 새로운 프레임워크를 제안합니다. 이 프레임워크는 사용자에게 안전한 정보 공개를 촉진하고, 자신의 데이터를 안전하게 다룰 수 있도록 지도합니다. 특히 대화 중 민감한 정보를 과도하게 공개하는 문제를 해결하기 위한 여러 모듈을 포함하고 있습니다. 이는 사용자로 하여금 자신의 정보가 어떻게 처리되는지에 대한 신뢰를 높이는 것을 목표로 합니다.

- **Technical Details**: 제안된 프레임워크는 기존의 대형 언어 모델(LLM) 주변에 적응형 래퍼(layer)를 통해 AI 리터러시를 통합합니다. 이 시스템은 사용자 기기에서 실시간으로 작동하며, 개인적인 데이터가 외부로 전송되지 않도록 합니다. 각 모듈은 사용자 입력을 분석하여 명확하고 안전한 응답을 촉진하기 위한 기술적 구성 요소로 이루어져 있습니다. 예를 들어, Prompt Coach 모듈은 모호한 입력을 감지하고, 예시 기반의 재구성을 제공하여 사용자에게 보다 효과적인 프롬프트 작성법을 가르칩니다.

- **Performance Highlights**: 이 연구는 AI 리터러시의 효과를 평가하기 위해 비임상 사용자와의 반복 측정 longitudinal study를 계획하고 있습니다. 두 가지 챗봇 조건을 비교하여 AI 리터러시가 사용자의 프롬프트 작성 능력 및 정보 공개 행동에 미치는 영향을 관찰할 것입니다. 또한, 사용자 경험 및 신뢰 발전에 관한 변화를 분석하기 위해 설문과 인터뷰를 활용할 예정입니다. 이를 통해 보다 안전한 AI 사용을 위한 방법론적 기반을 마련할 것입니다.



### PruneGCRN: Minimizing and explaining spatio-temporal problems through node pruning (https://arxiv.org/abs/2510.10803)
- **What's New**: 이번 연구에서는 딥러닝 모델을 사용하여 그래프 구조를 가지는 문제를 다루고, 이를 통해 설명 가능성을 통합할 수 있는 새로운 접근 방식을 제안합니다. 특히, 모델이 훈련 과정에서 그래프의 노드를 효율적으로 제거하는 최적화된 가지치기 메커니즘을 통합하는 것을 목표로 합니다. 이는 예측 오류를 최소화하면서 가장 관련성이 높은 노드를 선택하는 데 도움을 줍니다.

- **Technical Details**: 이 모델은 Prune Graph Convolutional Recurrent Network (PruneGCRN)이라는 명칭을 가지고 있으며, 이 네트워크는 훈련 동안 불필요한 노드 제거를 통해 데이터를 최적화합니다. 스페이셜 및 템포럴 데이터의 다차원적 특성을 반영하기 위해 Graph Neural Networks (GNNs)와 Recurrent Neural Networks (RNNs)의 조합이 활용됩니다. 또한, 모델은 실시간 교통 데이터에서 각 노드가 의미하는 내용을 바탕으로 중요한 정보를 추출합니다.

- **Performance Highlights**: 실험 결과, PruneGCRN 모델은 다른 방법에 비해 더 많은 정보를 유지하면서 그래프의 크기를 줄이는 것으로 나타났습니다. 이는 교통 예측 문제를 해결하는 데 있어 모델이 가장 중요한 요소를 식별하고, 분석을 용이하게 하는 데 기여하고 있습니다. 따라서 본 연구는 스페이셜-템포럴 문제를 간소화하는 모델 개발에 있어 가지치기의 가능성을 강조합니다.



### Toward Human-Centered Readability Evaluation (https://arxiv.org/abs/2510.10801)
Comments:
          Accepted to the 4th Workshop on Bridging Human-Computer Interaction and NLP (HCI+NLP) at EMNLP 2025, Suzhou, China

- **What's New**: 이 논문은 건강 정보 텍스트의 단순화를 위한 새로운 평가 체계인 Human-Centered Readability Score (HCRS)를 제안합니다. 기존의 자동화된 텍스트 평가 지표는 주로 표면적 특성에 초점을 맞추지만, HCRS는 인간 중심의 특성인 명확성(clarity), 신뢰성(trustworthiness), 톤 적절성(tone appropriateness), 문화적 관련성(cultural relevance), 그리고 실행 가능성(actionability)을 고려합니다. 이 새로운 프레임워크는 텍스트 단순화가 공공 보건에서 진정으로 효과적이기 위해서는 단순히 단어의 난이도를 낮추는 것 이상이 필요하다는 점을 강조합니다.

- **Technical Details**: HCRS는 자동화된 측정과 인간의 구조화된 피드를 통합하여 읽기 쉬움의 관계적 및 맥락적 측면을 포착합니다. 이 연구는 개별 사용자의 요구와 기대에 보다 부합하는 건강 텍스트 단순화를 평가하는것을 목표로 하고 있으며, 평가 파이프라인에 참여적 설계(participatory design)와 피드백 수집(interactive feedback collection)을 통합하는 방법을 논의합니다. 이를 통해 HCRS는 건강 문서에 대한 기존의 접근 방식을 재정의하고, 다양한 사용자의 기대에 부합하는 시스템 설계를 지원합니다.

- **Performance Highlights**: HCRS는 기존의 단일 사용 지표보다 사용자 평가와의 강력한 정렬을 목표로 합니다. 연구에서는 텍스트 단순화 후의 명확성, 신뢰성, 적절한 톤 및 문화적 관련성, 실행 가능성 등이 어떻게 사용자 경험과 연결되는지를 분석하였습니다. 초기 연구 결과는 텍스트 단순화가 정보의 인지적 부담을 줄이고 사용자가 이를 더욱 쉽게 이해할 수 있도록 돕는 것을 보여주며, 특히 교육 수준이 낮거나 영어 능력이 제한된 사용자에게서 가장 큰 효과를 나타냅니다.



### BioOSS: A Bio-Inspired Oscillatory State System with Spatio-Temporal Dynamics (https://arxiv.org/abs/2510.10790)
- **What's New**: 이 논문에서는 생물학적 신경의 특성을 모방한 생물영감을 받은 진동 상태 시스템(BioOSS)을 제안합니다. 기존의 딥러닝 모델이 지닌 한계를 보완하기 위해, BioOSS는 신경 회로에서 관찰되는 파동 전파 dynamics를 꿰뚫게 모델링할 수 있는 것으로 설계되었습니다. 특히 전두엽(prefrontal cortex)에서의 복잡한 활동 패턴을 직접적으로 반영하고자 합니다.

- **Technical Details**: BioOSS는 두 가지 상호작용하는 뉴런 집단, 즉 p 뉴런과 o 뉴런으로 구성되어 있으며, p 뉴런은 피라미드 세포에서 영감을 받은 단순화된 막 전위와 유사한 유닛을 나타냅니다. o 뉴런은 정보의 전파 속도를 조절하고 활동의 측면 확산을 조절합니다. 이 모델은 감쇠(damping)와 전파 속도(propagation speed) 용량을 조정 가능한 파라미터로 포함하여 특정 작업에 맞게 적응할 수 있도록 제공합니다.

- **Performance Highlights**: BioOSS는 합성 데이터 및 실제 세계의 작업에서 평가되어, 다른 아키텍처보다 우수한 성능과 향상된 해석성을 보여주었습니다. 실험 결과, 기존의 선형 변환만으로는 복잡한 진동 다이나믹스를 생성할 수 없음을 입증하였으며, BioOSS의 파라미터가 적절히 학습되었을 때 더 높은 품질의 spatio-temporal 패턴을 생성함을 확인하였습니다.



### ParsVoice: A Large-Scale Multi-Speaker Persian Speech Corpus for Text-to-Speech Synthesis (https://arxiv.org/abs/2510.10774)
- **What's New**: Persian Language는 1억 명이 넘는 사람들이 사용하지만, 고품질 음성 데이터셋이 부족하여 TTS(텍스트-음성 합성) 기술의 발전에 큰 제약이 있어 왔습니다. 이에 대응하기 위해, ParsVoice라는 대규모 페르시아어 음성 코퍼스를 소개하였습니다. 이 데이터는 자동화 파이프라인을 통해 2,000개의 오디오북에서 변환된 데이터로 3,526시간의 청정 음성을 포함하며, 이는 TTS용으로 최적화된 1,804시간의 데이터를 보여줍니다.

- **Technical Details**: 이 연구는 ParsVoice라는 새로운 페르시아어 음성 코퍼스를 개발하기 위해 여러 기술적 접근 방식을 결합했습니다. BERT 기반의 문장 완성 감지기와 오디오-텍스트 정렬을 위한 경계 최적화 기법을 사용하여, 오디오북 데이터를 TTS에 적합한 형태로 변환하는 자동화된 파이프라인을 구축하였습니다. 이 파이프라인은 470명 이상의 화자가 포함된 대규모 음성 데이터를 생성합니다.

- **Performance Highlights**: ParsVoice는 고품질 페르시아어 음성 데이터셋으로서, 다수의 화자와 양질의 음성을 제공하여 영어 코퍼스 대비 동등한 수준의 다양성을 자랑합니다. 연구 결과, 이 코퍼스는 페르시아어 음성 언어 처리 기술 발전에 기여할 것으로 기대되며, 저자원 언어들이 이 데이터를 활용할 수 있는 모범 사례를 제시합니다. 공개된 데이터셋은 페르시아어 기술 발전을 가속화하는 데 중요한 역할을 할 것입니다.



### Understanding Sampler Stochasticity in Training Diffusion Models for RLHF (https://arxiv.org/abs/2510.10767)
- **What's New**: 이번 연구에서는 Human Feedback로부터의 강화 학습(RLHF)을 통해 확산 모델을 미세 조정할 때 발생하는 도전 과제인 보상 간극(reward gap)을 이론적으로 분석하고, 일반적 확산 모델에 대한 비허무 경계(non-vacuous bounds)와 Variance Exploding(VE) 및 Variance Preserving(VP) 가우시안 모델의 수렴 속도를 제공하였다. 이 과정에서 일반화된 디노이징 확산 암묵 모델(gDDIM) 프레임워크를 도입하여 임의의 높은 수준의 확률성을 지원하고 데이터를 보존하는 방식을 강조하였다.

- **Technical Details**: 본 연구는 확산 모델의 전방 및 후방 프로세스를 포함하는 연속 시간 확산 모델을 설명하고, SDE(Stochastic Differential Equations)을 사용하여 목표 데이터 분포를 생성하는 목표를 설정하였다. 이 모델의 후방 프로세스는 목표 분포를 알 수 없기 때문에 사전 분포(pnoise)를 사용하여 시작되며, 이를 통해 역시간 과정의 산출을 정당화하였다. 이러한 이론적 기초는 Gronwall의 부등식을 사용하여 SDE 미세 조정된 모델과 ODE(Ordinary Differential Equations) 샘플링 모델 간의 보상 간극을 제한하는 데 기여하였다.

- **Performance Highlights**: 대규모 텍스트-이미지 모델 및 RLHF 알고리즘에 대한 실험을 통해, 훈련 단계에서 보상 간극이 일관되게 줄어들며 ODE 샘플링의 품질이 향상된다는 것을 입증하였다. 특히, 중간에서 높은 확률성의 훈련(예: η=1.2)이 도메인 내 및 도메인 외 성능을 향상시킨다는 결과를 보였으며, ODE 추론이 소규모 디노이징 단계 예산 하에서도 안정적으로 SDE 추론보다 더 우수한 성능을 발휘하였다.



### GPS Spoofing Attack Detection in Autonomous Vehicles Using Adaptive DBSCAN (https://arxiv.org/abs/2510.10766)
- **What's New**: 본 연구에서는 자율차량(AV)의 GPS 스푸핑 공격 탐지를 위한 새로운 접근 방식을 제안합니다. 실시간으로 조정 가능한 Density Based Spatial Clustering of Applications with Noise (DBSCAN) 알고리즘을 활용하여 탐지 임계값({\epsilon})을 동적으로 조정하는 방식입니다. 이 방법은 초기 임계값을 120,000개의 클린 데이터 샘플에서 설정하여 미세한 GPS 스푸핑을 초기에 탐지할 수 있습니다.

- **Technical Details**: 논문에서는 대규모 스푸핑 공격(예: 턴 바이 턴, 정지, 그리고 오버슈트 공격)과 다수의 소규모 편향 공격을 탐지하는 방법론을 설명합니다. 심층 신경망(DNN)을 사용하여 차량의 다음 단계 이동을 예측하고, GPS 데이터와 비교하여 오차를 지속적으로 모니터링 합니다. 계산된 오차가 설정된 임계값을 초과하면 비정상으로 분류되며, 다양한 스푸핑 공격 유형을 식별할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 실제 Honda Research Institute Driving Dataset (HDD)에서 5가지 다른 하위 집합을 사용하여 성능을 평가했습니다. 결과적으로 이 방법은 각각 98.621%, 99.960%, 99.880%, 98.380%의 높은 탐지 정확도를 기록했으며, 자율차량의 GPS 스푸핑 공격에 대한 보안과 안전성을 향상시키는 데 큰 기여를 합니다.



### A Stochastic Differential Equation Framework for Multi-Objective LLM Interactions: Dynamical Systems Analysis with Code Generation Applications (https://arxiv.org/abs/2510.10739)
Comments:
          Peer-reviewed and accepted to the 39th Conference on Neural Information Processing Systems (NeurIPS 2025) DynaFront 2025 Workshop (this https URL)

- **What's New**: 우리는 반복적인 대형 언어 모델(LLM) 상호작용에서 다목적 최적화의 역학을 모델링하기 위한 일반적인 확률적 미분 방정식(SDE) 프레임워크를 소개합니다. 이 프레임워크는 LLM 응답의 고유한 불확실성을 명시적인 확산( diffusion) 항을 통해 포착하며, 상충하는 목표 간의 체계적인 간섭 패턴을 간섭 행렬(interference matrix) 형식을 통해 드러냅니다.

- **Technical Details**: 우리의 접근 방식은 목표 벡터의 연속적 시간 발전을 드리프트-확산(drift-diffusion) 과정으로 모델링하여 수렴 특성, 안정성 조건 및 상충 목표 간의 간섭 패턴을 엄격하게 분석하게 합니다. 모델링 과정에서는 𝐱(t)∈ℝ^n 형태의 목표 벡터와 관련하여 SDE를 기반으로 한 구성을 제공합니다. 이를 통해 품질-다양성 트레이드오프와 같은 다양한 어플리케이션에서의 분석을 가능하게 합니다.

- **Performance Highlights**: 코드 생성에서의 초기 검증 도메인을 통해, 400개의 세션을 분석하고 보안, 효율성, 기능성 목표에 걸쳐 전략 의존적인 수렴 행동을 보여주며, 수렴 속도는 0.33부터 1.29까지 변화합니다. 예측 정확도는 균형 잡힌 접근 방식에서 R² = 0.74를 달성하였습니다. 이러한 결과는 다목적 LLM 상호작용에 대한 동적 시스템 분석의 실행 가능성을 제시합니다.



### Proficiency-Aware Adaptation and Data Augmentation for Robust L2 ASR (https://arxiv.org/abs/2510.10738)
Comments:
          Submitted to ICASSP 2026

- **What's New**: 이 논문은 일반적인 ASR(Automatic Speech Recognition) 시스템이 비정형 화자에게 적합하지 않음을 강조합니다. 특히, 비원어민(L2) 화자는 ASR 시스템에서 더 많은 불리한 영향을 받습니다. 이 문제를 해결하기 위해, 저자들은 두 가지 새로운 전략을 제안합니다: (i) 숙련도 인지 다중 작업 학습(proficiency-aware multitask learning)과 (ii) 대상형 증강(targeted augmentation)을 통해 ASR 모델의 성능을 개선하고 불균형을 줄입니다.

- **Technical Details**: 연구에 사용된 데이터셋은 Cambridge University에서 개발한 Speak & Improve(S&I) 코퍼스입니다. 이 코퍼스는 315시간의 L2 영어 학습자 음성을 포함하며, CEFR(영어 능력 기준) 등급에 따라 구분되어 있습니다. 이 연구는 Whisper-small 모델을 기반으로 하여, LoRA(저차원 적응)를 적용하고 다중 작업 학습 및 대상형 증강과 같은 다양한 방법을 통해 ASR 성능을 향상시키는 방법을 제안합니다.

- **Performance Highlights**: 결과적으로, LoRA만 적용했을 때 WER(단어 오류율)는 9.2%로 감소했으며, 다중 작업 세팅을 활용한 경우에는 8.1%로 개선되었습니다. 데이터 증강만 사용했을 때는 7.4%, 마지막으로 다중 작업 학습과 증강을 결합했을 때는 7.2%로 얻어져, 29.4%의 상대적 WER 감소를 실현했습니다. 모든 개선 사항은 통계적으로 유의미하며(p<0.05), 이는 L2 학습자에게 공정한 ASR 시스템 개발에 기여할 것입니다.



### Provable Anytime Ensemble Sampling Algorithms in Nonlinear Contextual Bandits (https://arxiv.org/abs/2510.10730)
Comments:
          40 pages, 1 figure

- **What's New**: 이 논문은 비선형 컨텍스트 밴딧(Nonlinear Contextual Bandits)에서 앙상블 샘플링(ensemble sampling)에 대한 통합 알고리즘 프레임워크를 제시합니다. 또한, 일반화 선형 앙상블 샘플링(GLM-ES)과 신경 앙상블 샘플링(Neural-ES) 두 가지에 대한 후회 경계(regret bounds)를 개발하였습니다. 두 방법 모두 랜덤하게 변동된 데이터에 대한 최대 우도 추정(maximum likelihood estimation)을 통해 보상 모델 파라미터에 대한 여러 추정기를 유지합니다.

- **Technical Details**: GLM-ES에 대한 후회 경계는 $\\mathcal{O}(d^{3/2} \sqrt{T} + d^{9/2})$로, Neural-ES는 $\\mathcal{O}(\widetilde{d} \sqrt{T})$로 설정되어 있으며, 여기서 $d$는 특징 벡터의 차원, $\widetilde{d}$는 신경 접선 커널(neural tangent kernel) 매트릭스의 유효 차원, 그리고 $T$는 라운드 수를 의미합니다. 이론적 분석에서는 비선형 모델에 특정한 도전을 해결하는 기술을 도입하였습니다. 또한, 고정 시간 수명 가정을 제거하고 비선형 밴딧에 적합한 anytime 버전을 개발하여 알고리즘의 적용 범위를 확장하였습니다.

- **Performance Highlights**: GLM-ES, Neural-ES 및 그들의 anytime 변형을 실험적으로 평가하여 강력한 성능을 보였습니다. 연구 결과는 비선형 컨텍스트 밴딧의 랜덤 탐색 접근법은 증명이 가능하고 실용적이라는 것을 입증하였습니다. 특히, 기존의 메타 분석에서는 선형 컨텍스트 밴딧에 대한 후회 보장만 제공되었던 반면, 본 연구는 비선형 밴딧의 경우 처음으로 높은 확률의 후회 경계를 제안합니다.



### SS-DPPN: A self-supervised dual-path foundation model for the generalizable cardiac audio representation (https://arxiv.org/abs/2510.10719)
- **What's New**: 이 논문에서는 신뢰할 수 있는 심장 오디오 표현 및 분류를 위해 Self-Supervised Dual-Path Prototypical Network (SS-DPPN)를 제안합니다. 이 모델은 라벨이 없는 데이터에서 심장 질환을 분석하는 데 도움을 주며, 1D 파형과 2D 스펙트로그램을 동시에 처리하는 듀얼 경로 대조 학습 기반 아키텍처를 도입합니다. 또한, 효과적인 성능을 위해 새로운 하이브리드 손실(hybrid loss)을 사용합니다.

- **Technical Details**: SS-DPPN은 metric-learning 접근 방식과 Prototypical Network를 활용하여 민감도를 향상시키고 신뢰할 수 있는 예측을 생성합니다. 우리는 CirCor 2022를 포함한 네 가지 주요 심장 소리 기준으로 모델 성능을 평가했습니다. 다양한 평가 지표(AUROC, AUPRC, Accuracy 등)를 이용하여 모델의 신뢰성과 일반화 가능성을 분석했습니다.

- **Performance Highlights**: SS-DPPN은 상당한 데이터 효율성을 보여주며 라벨이 있는 데이터 세트는 3분의 1로 줄일 수 있었습니다. 또한, 이 모델은 심장 소리 데이터에 대한 최고의 성능을 달성했으며, 다른 생리 신호(classification tasks)로 높이 평가받고 있습니다. 이 연구는 SS-DPPN이 의료 AI의 주요 과제인 데이터 주석 병목 현상을 해결할 수 있는 가능성을 제시하고 있습니다.



### HYPERDOA: Robust and Efficient DoA Estimation using Hyperdimensional Computing (https://arxiv.org/abs/2510.10718)
Comments:
          3 figures, 5 pages. Authors' version posted for personal use and not for redistribution

- **What's New**: 본 논문에서는 HYPERDOA라는 새로운 DoA(Directional of Arrival) 추정기를 소개하며, 이는 Hyperdimensional Computing (HDC)을 활용합니다. HYPERDOA는 Mean Spatial-Lag Autocorrelation과 Spatial Smoothing이라는 두 가지 특징 추출 전략을 가지고 있으며, DoA 추정을 패턴 인식 문제로 재구성합니다. 이 접근법은 기존의 복잡한 행렬 분해를 우회하여, 전통적인 및 딥러닝 방법의 높은 비용을 줄입니다.

- **Technical Details**: HYPERDOA는 일관된 신호 소스와 낮은 SNR(신호 대 잡음비) 환경에서의 성능을 극대화하기 위해 HDC의 특징을 활용합니다. 시스템 구성은 4단계로 이루어져 있으며, 첫 번째 단계는 공간 정보를 압축된 특징 벡터로 변환하여 추출합니다. Mean Spatial-Lag Autocorrelation 방법을 통해 평균 자기 상관을 구하고, 이 벡터는 저전력 시스템에 적합하도록 최적화됩니다.

- **Performance Highlights**: HYPERDOA는 기존의 최신 기술보다 약 35.39% 높은 정확력을 달성하며, NVIDIA Jetson Xavier NX 플랫폼에서 경쟁 신경망 기반 모델에 비해 약 93% 더 적은 에너지를 소비합니다. 이를 통해 HYPERDOA는 엣지 디바이스에서의 신뢰성과 효율성을 동시에 만족시킬 수 있는 robust한 솔루션으로 자리매김합니다.



### Deep Learning in Astrophysics (https://arxiv.org/abs/2510.10713)
Comments:
          Manuscript submitted to Annual Review of Astronomy and Astrophysics for Volume 64. This is the authors' version. Revisions and the final version will be available at this https URL

- **What's New**: 본 리뷰 논문은 천문학에서의 딥러닝 기법의 발전과 도전 과제를 다룹니다. 천문학자들은 데이터의 양이 폭발적으로 증가함에 따라 기계 학습, 특히 딥러닝 기술을 활용하여 높은 차원의 데이터에서 패턴을 추출하고 있습니다. 하지만 이러한 기술에 대한 반응은 일부는 변혁적이라 주장하는 반면, 다른 일부는 그 이점을 회의적으로 바라보고 있습니다.

- **Technical Details**: 딥러닝 모델은 물리적 대칭성과 보존 법칙을 네트워크 구조에 통합하여 데이터의 제한된 레이블로부터 학습할 수 있는 능력을 갖추고 있습니다. 또한, 시뮬레이션 기반 추론과 이상 탐지는 복잡한 비가우시안 분포로부터 정보를 추출하여 천문학적인 분석과 희귀 현상의 체계적인 발견을 가능하게 합니다. 고차원 공간에서의 데이터 격차를 메우기 위해 다중 스케일 신경 모델링이 도입되어 고충실도 계산으로부터 효과적인 서브그리드 물리를 학습합니다.

- **Performance Highlights**: 본 논문은 딥러닝이 기존의 전통적 방법론에 비해 일반화 가능성과 데이터 효율성을 크게 향상시키는 방법을 제시합니다. 전통적 방법들은 고차원 공간에서의 성능에 한계를 보이며, 이는 현대 천문학의 데이터 특성에 부합하지 않습니다. 반면, 딥러닝은 새로운 데이터 속성과 현상을 탐지할 수 있는 유연성을 제공하여, 천문학적 발견의 잠재력을 높여줍니다.



### Missing Data Multiple Imputation for Tabular Q-Learning in Online RL (https://arxiv.org/abs/2510.10709)
Comments:
          Working paper

- **What's New**: 온라인 강화 학습(online Reinforcement Learning, RL)에서의 누락 데이터 문제는 전통적인 표 형식 데이터나 오프라인 정책 학습과 비교할 때 보다 복잡한 도전 과제가 됩니다. 특히, 각 시점에서 데이터를 보완(impute)하고 행동(act)해야 하는 필요성으로 인해, 안정적인 보완 모델이 생성되기 위해 충분한 데이터가 존재할 때까지 보완이 연기될 수 없습니다. 본 논문에서는 완전 온라인 보완 앙상블(full online imputation ensembles)을 제안하며, 이는 누락된 데이터 상태에서 불확실성을 캡처하고 컴퓨팅 효율성을 높임을 목표로 합니다.

- **Technical Details**: 본 연구에서는 다양한 접근 방식을 적용하여 여러 보완 경로(multiple imputation pathways)를 학습 및 행동 선택(action selection) 과정에 통합하였습니다. 우리는 누락된 상태 공간 데이터를 사용한 온라인 RL을 위해 여러 보완 앙상블을 탐구하며, 확률적 보완 임베딩이 단순한 기초 모델들보다 더 나은 성능을 발휘할 수 있는 가능성을 제시합니다. 또한, 이러한 앙상블 방식은 반복적인 누락 정보로 인한 경로 의존성(path dependency) 문제를 피할 수 있습니다.

- **Performance Highlights**: 그리드 월드(experiment)에서 진행된 실험을 통해, 다수의 보완 경로가 간단한 기초 모델과 단일 보완 모델보다 더 나은 성능을 발휘할 수 있음을 초기 증거를 통해 확인하였습니다. 또한, 누락 데이터를 상태 옵션으로 인코딩하는 방식과의 비교를 통해 누락 비율(missingness rate)에 따라 U자 형태의 성능 곡선이 나타났음을 확인하였습니다. 이는 제안된 방법이 누락된 정보로 인해 상태 공간 차원을 효과적으로 조절함을 시사합니다.



### Attention-Enhanced LSTM Modeling for Improved Temperature and Rainfall Forecasting in Bangladesh (https://arxiv.org/abs/2510.10702)
- **What's New**: 이 연구에서는 기후 변화의 영향이 큰 방글라데시에서 온도와 강수 예측을 개선하기 위해 주목 메커니즘이 통합된 향상된 Long Short-Term Memory (LSTM) 모델을 소개합니다. 1901년부터 2023년까지의 포괄적인 데이터를 활용하며, 기존 모델들이 포착하지 못했던 계절적 및 장기적 추세를 효과적으로 분석하여 더욱 정교한 예측을 가능하게 합니다. 본 연구는 그동안의 연구에서는 미비했던 복잡한 비선형 시계열 특성을 보다 잘 모델링할 수 있는 방법을 제시합니다.

- **Technical Details**: LSTM 모델은 시계열 데이터의 순차적 특성을 잘 포착하는 데 강점을 지니며, 이전 값들을 오랜 기간 동안 기억하는 능력이 있습니다. 본 연구에서 제안한 모델은 주목 메커니즘을 통합하여 중요한 시간 패턴을 동적으로 가중치화함으로써 단기 변동성과 장기 기후 추세를 보다 잘 포착합니다. 또한, 본 모델은 복잡한 스택킹이나 하이브리드 모델과 비교할 때 계산 효율성을 높이면서도 예측 성능을 향상시키도록 설계되었습니다.

- **Performance Highlights**: 모델은 기존의 XGBoost, Simple LSTM 및 GRU 모델을 초월하여 뛰어난 예측 성능을 보여주었습니다. 예측 테스트에서 온도에 대한 MSE는 0.2411, R^2 값은 0.9834, 강수에 대한 MSE는 1283.67 mm², R^2 값은 0.9639를 기록하며, 안정성 강화를 위해 기후 변동에 대한 변화에 대해서도 향상된 성능을 유지합니다. 이 결과들은 해당 모델이 방글라데시의 기후 변화 응답 분야에서 중요한 도구로 자리잡을 수 있음을 나타냅니다.



### High-Dimensional Learning Dynamics of Quantized Models with Straight-Through Estimator (https://arxiv.org/abs/2510.10693)
Comments:
          27 pages, 14 figures

- **What's New**: 이번 연구는 Quantized neural network training에서 quantization hyperparameters가 학습 동역학에 미치는 영향을 이론적으로 탐색합니다. 특히, straight-through estimator (STE)의 동적 속성이 고차원 한계에서 어떻게 변화하는지를 분석합니다. 이 연구는 quantization range와 bit width가 일반화 오차에 미치는 영향을 정량화하고, quantized DNNs의 학습 안정성을 밝혀내는 데 기여합니다.

- **Technical Details**: STE를 이용한 학습이 고차원 한계에서 확률적 미분 방정식(SDE)과 결정론적 미분 방정식(ODE)의 수렴을 보여주며, 두 단계의 궤적을 예측합니다. 여기서 ODE는 연장된 plateau와 일반화 오차의 급속한 감소를 나타냅니다. 특히, 하이퍼파라미터 선택이 학습 다이나믹스의 중요한 요소로 작용하며, 입력 양자화가 성능 저하에 미치는 영향을 분석합니다.

- **Performance Highlights**: 저자들은 quantization hyperparameters가 학습 안정성 및 일반화 성능에 미치는 영향을 체계적으로 수치화했습니다. 낮은 비트 폭에서 STE의 동적 변화가 비모노톤(non-monotonic)이 되어 수렴 속도가 느려지는 경향이 있음을 발견했습니다. 또한, 비양자화 모델에 비해 성능 저하를 정량적으로 분석하여, 양자화가 단순한 교란이 아닌 내재적 정규화 역할을 한다는 점을 강조합니다.



### LSZone: A Lightweight Spatial Information Modeling Architecture for Real-time In-car Multi-zone Speech Separation (https://arxiv.org/abs/2510.10687)
Comments:
          submitted to ICASSP 2026

- **What's New**: 본 논문에서는 자동차 내 다중 존(multi-zone) 음성 분리(real-time in-car multi-zone speech separation)을 위한 경량의 공간 정보 모델링 아키텍처 LSZone을 제안합니다. 이전의 SpatialNet이 뛰어난 성능을 보여주었지만, 높은 계산 비용이 실시간 응용을 저해하는 문제를 해결하기 위해 LSZone은 Mel 스펙트로그램(Mel spectrogram)과 상관 위상 차이(IPD)를 결합한 SpaIEC 모듈을 사용합니다. 이 모듈은 성능을 유지하면서 계산 부담을 줄이는 데 중점을 두었습니다.

- **Technical Details**: LSZone 아키텍처는 multi-channel 스펙트로그램을 입력으로 받아 공간, 주파수 및 시간 정보를 효과적으로 모델링합니다. SpaIEC 모듈을 통해 입력의 특성을 압축하고 추출하여 Mel 주파수 필터 뱅크 수에 맞춰 특성 차원을 낮춥니다. 또한, Conv-GRU 기반의 CNP 모듈을 통해 공간 정보 캡처를 최적화하고, 다양한 음향 정보를 최소한의 계산 오버헤드로 통합합니다.

- **Performance Highlights**: 실험 결과, LSZone은 0.56G MACs의 계산 비용과 0.37의 실시간 팩터(RTF)를 가지며, 복잡한 잡음 및 다수의 화자가 존재하는 환경에서 뛰어난 성능을 발휘합니다. 이러한 결과는 LSZone이 제한된 계산 자원과 엄격한 저지연 요구를 만족시키며, 실시간 자동차 내 음성 분리에 있어서 유망한 솔루션임을 보여줍니다.



### BrowserAgent: Building Web Agents with Human-Inspired Web Browsing Actions (https://arxiv.org/abs/2510.10666)
Comments:
          10 pages

- **What's New**: 최근 연구들은 웹 환경과 상호작용하는 LLMs의 중요성을 강조하고 있으며, BrowserAgent는 인간의 브라우징 행동을 모방해 복잡한 웹 작업을 수행하는 더 상호작용적인 에이전트를 제안합니다. 기존의 Search-R1과 WebDancer는 정적 텍스트 콘텐츠에 의존했던 반면, BrowserAgent는 Playwright를 사용하여 실시간 웹 페이지에서 작업을 수행합니다. 이 논문은 두 단계의 훈련 방식(Supervised Fine-Tuning (SFT)과 Rejection Fine-Tuning (RFT))을 통해 모델의 일반화 능력을 향상시킵니다.

- **Technical Details**: BrowserAgent는 브라우저 상에서 직접 원시 웹 페이지 작업을 수행함으로써 사람과 유사한 행동을 통해 복잡한 작업을 처리합니다. 네 가지 범주의 사용자의 행동(페이지 작업, 탭 관리, URL 탐색, 완료 행동)을 정의하여, 웹 상에서 실시간으로 상호작용하며 훈련 데이터를 생성합니다. 이러한 접근 방식은 모델이 실제 웹 콘텐츠와 상호작용하면서 자연스럽게 정보 검색과 이해 능력을 키우도록 합니다.

- **Performance Highlights**: BrowserAgent는 5.3K의 훈련 샘플로도 Search-R1를 초과하는 성능을 보여주며, 다양한 Open-QA 작업에서 뛰어난 결과를 나타냅니다. 예를 들어, BrowserAgent-7B는 HotpotQA와 같은 멀티 홉 QA 작업에서 약 20%의 성능 향상을 기록했습니다. 이러한 결과들은 BrowserAgent가 더 상호작용적이고 확장 가능한 웹 에이전트를 구축하는 데 기여할 수 있음을 보여줍니다.



### AGENTIQL: An Agent-Inspired Multi-Expert Framework for Text-to-SQL Generation (https://arxiv.org/abs/2510.10661)
Comments:
          Accepted at NeurIPS 2025, ER "Efficient Reasoning" workshop

- **What's New**: 본 논문에서는 AGENTIQL이라는 새로운 다중전문가 아키텍처를 제안합니다. 이 아키텍처는 쿼리 분해를 위한 추론 에이전트, 서브쿼리 생성을 위한 코딩 에이전트, 열 선택을 위한 정제 단계로 구성되어 있습니다. AGENTIQL은 모듈식 파이프라인을 통해 효율성과 정확성을 동시에 달성하여 더 나은 해석 가능성을 제공합니다.

- **Technical Details**: AGENTIQL의 주요 구성 요소는 질문을 더 작은 서브질문으로 분해하고, 이에 대한 SQL 쿼리를 생성하며, 필요 시 열 선택을 통해 SQL 쿼리를 조정하는 것입니다. 이 과정에서 적응형 라우터가 사용되어 쿼리의 복잡성에 따라 자원을 효과적으로 분배합니다. 이론적으로 이 구조는 다양한 데이터베이스 스키마에 대한 처리 능력을 향상시키고, 복잡한 추론을 가능하게 합니다.

- **Performance Highlights**: Spider 벤치마크에서 AGENTIQL은 최대 86.07%의 실행 정확도를 달성하며, 기존의 GPT-4 기반 SOTA와의 격차를 좁힙니다. 이 성능은 라우팅 메커니즘의 효율성에 따라 달라지며, 기존 LLM보다 작은 모델을 사용할 때에도 높은 성능을 유지합니다. AGENTIQL은 또한 중간 추론 단계를 노출하여 투명성을 높이고, 해석 가능한 의미 파싱 접근 방식을 제공합니다.



### Trustworthy Retrosynthesis: Eliminating Hallucinations with a Diverse Ensemble of Reaction Scorers (https://arxiv.org/abs/2510.10645)
- **What's New**: 이 논문에서는 RetroTrim이라는 새로운 회귀 합성 시스템을 소개합니다. 이 시스템은 약물과 유사한 도전적인 타겟 세트에서 nonsensical한 경로를 성공적으로 회피하는 것이 특징입니다. RetroTrim은 hallucinated (환각된) 반응을 필터링하는 유일한 방법으로, 전체적으로 높은 품질의 경로를 생성하는 것으로 입증되었습니다.

- **Technical Details**: RetroTrim은 다양한 반응 평가 전략을 조합하여 hallucinations (환각) 문제를 해결하는 데 중점을 두고 있습니다. 이 시스템은 Reaction Prior (RP), Reaction Graph Plausibility (RGP), Reaction Retrieval Score (RRS)라는 세 가지 주요 스코어러를 사용하여 반응의 신뢰성을 평가합니다. 각 스코어러는 서로 다른 종류의 hallucinations을 필터링하는 데 강점을 보이며, 메타 스코어러는 이를 종합적으로 평가합니다.

- **Performance Highlights**: RetroTrim은 32개의 신약 유사 타겟 세트에서 기존의 다른 회귀 합성 시스템과 비교했을 때, hallucinated 반응을 모두 배제하고 가장 많은 문제 없는 합성 경로를 발견한 것으로 나타났습니다. 본 논문은 약리 화학 영역에서 신뢰할 수 있는 회귀 합성을 연구하자는 목표를 가지고 있으며, 평가 프로토콜과 기준이 되는 데이터 세트를 공개하여 후속 연구를 자극하고자 합니다.



### UniCoD: Enhancing Robot Policy via Unified Continuous and Discrete Representation Learning (https://arxiv.org/abs/2510.10642)
- **What's New**: 본 논문에서는 로봇 정책 학습을 위한 새로운 접근법인 UniCoD를 소개합니다. UniCoD는 비전-언어 모델과 생성 모델의 강점을 결합하여, 다양한 조작 작업을 수행하는 일반화된 로봇 정책을 개발하는 데 초점을 맞추고 있습니다. 1,000,000개 이상의 인터넷 규모 조작 비디오에서 사전 학습을 진행하여 고차원 비주얼 기능을 동적으로 모델링하는 능력을 향상시킵니다.

- **Technical Details**: UniCoD는 이해-, 생성-, 실행(understanding-generation-execution) 패러다임을 따라 작업 인식과 미래 상태 예측을 통합합니다. 이 모델은 MOT 아키텍처(modality-specialized experts)를 사용하여 다양한 데이터 소스에서 학습하며, 첫 번째 단계에서는 로봇 및 인간 시연에서 수집된 Embodied QA 데이터를 사용하여 언어 및 비주얼 표현을 학습합니다. 이후 두 번째 단계에서는 행동 동작이 주석 처리된 로봇 데이터를 이용하여 학습을 진행해, 비주얼 미래와 행동을 예측하는 능력을 향상시킵니다.

- **Performance Highlights**: 실험 결과, UniCoD는 기존의 SOTA(State-of-the-Art) 방법에 비해 9% 향상된 성능을 보여주며, 실제 로봇 팔 및 손에서 복잡한 작업에 대해 강력한 의미 일반화(semantic generalization)를 달성합니다. 또한, 제안된 이 모델은 시뮬레이션 및 실제 환경 모두에서 최신 기술 수준의 성능을 발휘하며, 다양한 특징 설계 선택이 모델의 능력에 미치는 영향을 분석합니다.



### A Machine Learning Approach for MIDI to Guitar Tablature Conversion (https://arxiv.org/abs/2510.10619)
Comments:
          Proceedings of the 19th Sound and Music Computing Conference, June 5-12th, 2022, Saint-Étienne (France)

- **What's New**: 본 논문에서는 MIDI 기반의 음악 부분에 기타 타블래쳐 표기를 지정하는 새로운 방법을 제시합니다. 이 방법은 기타의 특성을 고려하지 않고 일반적인 6현 기타의 조율에만 초점을 맞추며, 여러 폴리포닉 트랙을 다룰 수 있는 가능성을 가지고 있습니다. 머신 러닝을 활용하여 손가락의 스트레칭 가능성을 가정하고 있으며, 실제로 플레이할 수 없는 음악 조각의 전사를 위한 기초 방법도 다룹니다.

- **Technical Details**: 제안된 방법은 6현과 24프레트의 정상 조율 기타를 가정하며, MIDI 피치의 바이너리 표현을 타블래쳐로 변환하는 이진 과정으로 설명됩니다. 이 과정은 두 부분으로 나뉘어 있으며, 딥 뉴럴 네트워크를 활용해 입력 피치에 대한 확률적 타블래쳐를 생성하고, 검색 알고리즘을 통해 플레이 가능한 문자열-프레트 조합을 찾습니다. 이 방법은 &quot;playability&quot; 개념을 통해 최상의 문자열-프레트 위치를 고려하여 트랜스크립션을 수행합니다.

- **Performance Highlights**: 결과적으로, 확장된 데이터로 훈련된 시스템은 더 나은 성능을 나타내며 단순한 모노포닉 케이스에서도 효과적임을 보여줍니다. 연구는 또한 시스템의 약점을 지적하고, 향후 개선 방향에 대한 유용한 결론을 제시합니다. 이러한 방법론은 음악 조각의 특성이 기타 전사 및 타블래쳐 생성에 미치는 영향에 대한 새로운 통찰을 제공합니다.



### Dynamic Topic Evolution with Temporal Decay and Attention in Large Language Models (https://arxiv.org/abs/2510.10613)
- **What's New**: 이 논문은 템포랄 대형 언어 모델(temporal large language models)을 기반으로 한 동적 주제 진화(dynamic topic evolution)을 위한 모델링 프레임워크를 제안합니다. 이 방법은 텍스트의 맥락 임베딩(contextual embeddings)을 획득하기 위해 대형 언어 모델을 사용하고, 이어서 시간에 따른 중요성을 조정할 수 있는 템포랄 디케이 함수(temporal decay function)와 주의 메커니즘(attention mechanism)을 도입합니다.

- **Technical Details**: 모델은 시간 간격에 따라 의미 단위의 중요성을 조정하고 다양한 기간에 걸친 주제 변동을 포착합니다. 생성된 템포럴 표현은 잠재 주제 공간(latent topic space)으로 매핑되며, 여기에서 주제의 동적 진화를 설명하는 상태 전이 행렬(state transition matrix)이 적용됩니다. 공동 최적화 목표(joint optimization objective)는 의미 모델링과 시간적 일관성을 모두 제약하며, 주제 생성의 다양성과 부드러움을 보장합니다.

- **Performance Highlights**: 실제 데이터에서의 실험 결과, 이 프레임워크는 주제의 생성, 확장 및 감소를 효과적으로 포착하였으며, 여러 메트릭(metric)에서 기존 모델보다 뛰어난 성능을 보여주었습니다. 제안된 방법은 대규모 텍스트에서 동적 의미 패턴을 이해하기 위한 체계적인 솔루션을 제공하며, 주제 모델링의 연구 패러다임을 풍부하게 하고 다양한 도메인에서 복잡한 텍스트 분석 작업을 지원합니다.



### Compositional Symmetry as Compression: Lie Pseudogroup Structure in Algorithmic Agents (https://arxiv.org/abs/2510.10586)
Comments:
          Submitted to NeurReps 2025 (this https URL)

- **What's New**: 이번 연구에서는 Kolmogorov 이론을 기반으로 한 알고리즘 에이전트들이 감각 스트림을 추적하고 압축하는 방법을 제안합니다. 저자들은 구성적 대칭(compositional symmetry)을 구조적 우선기준으로 하여, 에이전트가 특정한 심볼릭 프로그램으로 환경을 모델링 하는 프레임워크를 제시합니다. 이 모델은 에이전트를 일반적인 신경 동역학 시스템(neural dynamical system)으로 설정하여, 환경을 정확히 추적할 수 있도록 합니다.

- **Technical Details**: 연구에서는 유한 매개변수 Lie 유사군(Lie pseudogroup)의 지역적 행동을 사용하여 구성 매니폴드(configuration manifold) 위의 생성 모델을 정의합니다. 에이전트는 이러한 스트림에 의해 구동되는 신경 ODE(neural ODE)로 모델링되며, 이는 대칭 기반의 자기 포함형 예측 코딩(predictive coding)의 버전을 구성합니다. 대칭 관계에 따라 에이전트의 동역학과 구성 방정식들은 규약과 제약에 따라 제한됩니다.

- **Performance Highlights**: 이 연구는 심볼 표기(simbolic representation)의 조합적 우수성(the blessing of compositionality)이 심층 모델(deep models)에서 샘플 복잡도(sample complexity)를 낮출 수 있음을 강조합니다. 또한, bare manifold prior가 추가적인 기하학적 정보를 가진 구조 없이는 불충분하다는 점도 지적합니다. 이러한 결과들은 대칭 인식 설계(symmetry-aware designs)의 필요성을 시사하며, 예측 코딩의 그룹 이론적 관점을 제공합니다.



### PAC-Bayesian Reinforcement Learning Trains Generalizable Policies (https://arxiv.org/abs/2510.10544)
- **What's New**: 본 연구에서는 Markov 의존성을 명시적으로 반영한 새로운 PAC-Bayesian 일반화 경계를 도출하였습니다. 기존의 RL에서 데이터의 순차적 특성은 독립성 가정이 깨지기 때문에 일반화 보장을 얻는 데 어려움이 있었습니다. 제안된 경계는 Soft Actor-Critic과 같은 현대의 오프-폴리시(Off-Policy) 알고리즘에 비생성적인(certificates) 인증서를 제공합니다.

- **Technical Details**: 본 논문에서는 Markov 의존성을 체인 혼합 시간(mixing time)을 통해 명시적으로 다루는 PAC-Bayesian 일반화 경계를 제시하였습니다. 우리가 통합한 핵심 기술 기여는 부정적 경험적 반응에 대한 경계 차별 조건을 Markov 체인을 위한 McDiarmid-type 집중 불평등과 결합하는 것으로, 이를 통해 과거의 방법들이 갖는 공허함을 제거할 수 있었습니다.

- **Performance Highlights**: PB-SAC 알고리즘을 통해 제안된 경계의 실용성을 입증하였습니다. 연속 제어 작업에 대한 실험 결과, 우리 접근 방식이 믿을 수 있는 신뢰 증명서를 제공하고, 동시에 최신의 방법들과 경쟁력 있는 성능을 유지함을 보여주었습니다. 이는 현대 RL 알고리즘을 위한 첫 번째 실용적인 PAC-Bayesian 프레임워크를 확립하여 학습 이론과 알고리즘적 실천 간의 간극을 메우는 데 기여합니다.



### Rethinking RL Evaluation: Can Benchmarks Truly Reveal Failures of RL Methods? (https://arxiv.org/abs/2510.10541)
- **What's New**: 현재의 벤치마크는 대형 언어 모델(Large Language Models, LLMs)에서 강화 학습(Reinforcement Learning, RL)의 진행 상황 평가에 부적합하다는 점을 강조합니다. 이 논문에서는 Oracle Performance Gap (OPG)이라는 새로운 지표를 소개하여, 훈련 세트와 테스트 세트 간의 성능 차이를 정량적으로 측정할 수 있도록 합니다. 또한, RL 기반 모델의 일반화 능력을 평가하기 위해 디자인된 다면적 진단 프레임워크를 제시합니다.

- **Technical Details**: 논문의 주요 측면 중 하나는 OPG 메트릭을 사용하여 RL 모델 성능을 정량화하는 것입니다. 이 메트릭은 'oracle' 모델(테스트 세트에서 직접 학습한 모델)과 'standard' 모델(훈련 세트에서 학습한 모델) 간 성능 차이를 측정합니다. OPG의 값이 낮을 경우, 해당 벤치마크가 일반화 가능성을 충분히 측정하지 못한다는 것을 나타냅니다.

- **Performance Highlights**: RL 모델들이 다양한 벤치마크에서 비슷한 성능을 보이는 경향이 있으며, 이는 일반화 능력이 뛰어난 것처럼 보일 수도 있지만 사실상 신뢰할 수 없는 결과입니다. 이 연구에서는 RL 모델이 출력한 높은 벤치마크 점수가 진정한 능력을 반영하지 않을 수 있음을 입증했습니다. 마지막으로, 효율적인 벤치마크 설계를 위한 세 가지 기본 원칙을 제시하여, RL 기반 모델의 실제 추론 능력을 보다 철저하게 평가할 수 있는 방안을 모색합니다.



### ECO: Enhanced Code Optimization via Performance-Aware Prompting for Code-LLMs (https://arxiv.org/abs/2510.10517)
- **What's New**: ECO는 코드 최적화를 위한 성능 인지 프롬프트 프레임워크로, 기존의 느린-빠른 코드 쌍을 이용한 접근법의 한계를 극복합니다. 이 프레임워크는 각 느린-빠른 코드 쌍에서 런타임 최적화 지침 (ROIs)을 추출하여 비효율의 근본 원인과 성능 개선의 이론을 설명합니다. ECO는 또한 신속한 병목 진단을 제공하며, 이를 통해 코드 최적화를 보다 효율적으로 진행할 수 있는 방법을 제시합니다.

- **Technical Details**: ECO는 첫째, 기호 기반 어드바이저 (symbolic advisor)를 사용해 입력 코드를 위한 맞춤형 병목 진단을 생성합니다. 둘째, ROI 검색기 (ROI retriever)를 이용해 관련 ROIs를 반환합니다. 이 두 출력은 결합되어 성능 인지 프롬프트를 형성하며, 이는 기존 코드-LLM 프롬프트에 쉽고 즉시 추가될 수 있습니다. ECO의 프롬프트는 모델에 구애받지 않으며, 별도의 파인 튜닝 (fine-tuning)이 필요하지 않습니다.

- **Performance Highlights**: 실험 결과, ECO 프롬프트는 코드-LLMs의 효율적인 코드 생성 능력을 크게 향상시킵니다. ECO를 통해 성능 향상은 최대 7.81배에 달하며, 정확성 저하를 최소화합니다. ECO는 코드 최적화 분야에서 혁신적인 접근법으로 자리 잡을 전망입니다.



### Population-Coded Spiking Neural Networks for High-Dimensional Robotic Contro (https://arxiv.org/abs/2510.10516)
- **What's New**: 이 논문은 Deep Reinforcement Learning (DRL)과 population-coded Spiking Neural Networks (SNNs)를 결합한 새로운 프레임워크를 제안하여 로봇 제어의 에너지 효율성과 성능 문제를 해결하고자 합니다. 이 접근 방식은 SNNs의 이벤트 기반, 비동기 계산 방식과 DRL의 견고한 정책 최적화 능력을 결합하여, 에너지 효율성과 제어 성능 간의 균형을 이룹니다.

- **Technical Details**: 제안된 프레임워크의 핵심은 Population-coded Spiking Actor Network (PopSAN)으로, 고차원 관측치를 신경 집단 활동으로 인코딩하고, 기울기 기반 업데이트를 통해 최적 정책 학습을 가능하게 합니다. 이 프레임워크는 Isaac Gym 플랫폼에서 PixMC 기준을 사용하여 다이내믹한 로봇 조작 작업에 대해 평가되었습니다.

- **Performance Highlights**: 실험 결과, Franka 로봇 팔을 사용한 경우, 기존의 Artificial Neural Networks (ANNs)와 비교하여 최대 96.10%의 에너지 절약을 달성했으며, 제어 성능을 유지하였습니다. 학습된 SNN 정책은 지시된 궤적에서 최소한의 편차로 손가락 위치 추적을 유지하였고, 피킹 및 배치 과정에서 안정적인 목표 높이 유지를 보여주었습니다.



### f-INE: A Hypothesis Testing Framework for Estimating Influence under Training Randomness (https://arxiv.org/abs/2510.10510)
- **What's New**: 본 논문에서는 머신러닝의 불확실성과 훈련 랜덤성으로 인한 기존의 영향 추정 방법의 한계를 극복하고자 새로운 프레임워크인 'f-influence'를 소개합니다. 이 방법은 가설 검정(hypothesis testing) 기반으로 훈련 랜덤성을 고려하여 신뢰할 수 있는 영향 추정을 가능하게 합니다.

- **Technical Details**: f-influence는 데이터 삭제/유지를 결정할 때 개별 샘플의 영향을 더욱 정확하게 추정할 수 있도록 설계되었습니다. 본 연구에서는 f-influence 계산을 단일 훈련 실행으로 수행할 수 있는 효율적인 알고리즘 f-INE을 제안하였으며, Llama-3.1-8B 모델을 사용한 데이터 정화를 통한 예제를 보여줍니다.

- **Performance Highlights**: 실험 결과, f-INE 알고리즘은 긍정적이지 않은 샘플을 신뢰성 있게 감지할 수 있으며, 이는 데이터 정화(data cleanup) 및 모델 행동(attribute model behavior) 분석에서 유용성을 입증합니다. 이 방법은 기존의 영향 추정 방법들에 비해 훈련 과정에서의 안정성을 크게 향상시키는 것으로 평가됩니다.



### MARS-Sep: Multimodal-Aligned Reinforced Sound Separation (https://arxiv.org/abs/2510.10509)
- **What's New**: MARS-Sep는 고수준의 신호 메트릭에 최적화된 기존 모델의 제한점을 극복하기 위해 제안된 강화 학습 프레임워크입니다. 이는 소스 간의 간섭을 효과적으로 차단할 뿐만 아니라 사용자의 의도에 부합하는 의미론적 ('semantic') 일관성을 보장하는 데 중점을 두고 있습니다. 모델은 베타 마스크 정책을 통해 마스크 생성을 결정적 의사결정 문제로 다시 정의하고, 진화적 방식으로 멀티모달 보상을 활용합니다.

- **Technical Details**: MARS-Sep는 강화 학습을 통해 마스크 예측을 맥락에 맞게 최적화합니다. 이는 클리핑된 서브레이트와 엔트로피 정규화 방식으로 안정적인 업데이트를 보장하며, 오디오, 텍스트, 비주얼 쿼리에 대한 의미론적 일관성을 장려합니다. 특히, 정책 학습의 안정성을 높이기 위해 진행적 정렬 전략이 도입되어 멀티모달 인코더를 정교하게 조정합니다.

- **Performance Highlights**: 이 방법은 VGGSOUND-clean+ 및 MUSIC-clean+ 벤치마크에서 일관된 성능 향상을 보였으며, SDR, SIR 및 SI-SDRi에서 중요한 수치를 개선했습니다. 뿐만 아니라 'CLAP' 점수가 높아졌으며 비대상 소스의 차단과 카테고리 구분의 명확성이 크게 향상되었습니다. 이러한 결과들은 새로운 모델이 신호 수준 및 의미론적으로 모두 개선되었다는 것을 보여줍니다.



### Align2Act: Instruction-Tuned Models for Human-Aligned Autonomous Driving (https://arxiv.org/abs/2510.10503)
- **What's New**: Align2Act는 기계가 인간의 행동에 맞는 해석 가능한 동작 계획을 생성할 수 있도록 하기 위해 설계된 새로운 프레임워크입니다. 기존의 방법들이 대개 미리 정의된 규칙이나 드라이빙 데이터에서 학습된 경로를 사용하는 반면, Align2Act는 구조적인 운전 지침을 통해 더 효율적인 계획을 가능하게 합니다. 이 방법은 LLaMA-2-7B 모델을 LoRA를 통해 세밀하게 조정하여, 다양한 시나리오에서 높은 성능을 보이고 있습니다.

- **Technical Details**: Align2Act는 텍스트 기반 입력을 사용해 차량의 상황 및 계획 목표를 설명함으로써, 대형 언어모델을 통해 최종 운전 경로和 해당 합리적 단계를 생성합니다. 계획 과정은 높은 차원의 조작을 더 쉽게 해석하도록 구분되는 몇 가지 단계로 나누어 설명하며, 이는 사람의 사고 방식과 유사합니다. 이를 통해 모형은 물리적 상태와 의미적 지침을 기반으로 동작할 수 있게 됩니다.

- **Performance Highlights**: Align2Act는 nuPlan 데이터셋을 사용해 100만 개 시나리오에서 미세 조정한 결과, 열린 루프 점수 85.17 및 고척 루프 점수 70.31(비반응형)과 66.96(반응형)을 기록하였습니다. 이 방식은 실 세계의 주행 환경에서도 개선된 계획 품질과 인간과 유사한 동작을 보이며, 기존 LLM 계획자 대비 성능이 크게 향상되었습니다.



### Personalized Motion Guidance Framework for Athlete-Centric Coaching (https://arxiv.org/abs/2510.10496)
- **What's New**: 이 연구는 운동 과학에서 그룹 수준의 인사이트와 개인화된 코칭 필요성 사이의 간극을 해소하는 데 초점을 맞추고 있습니다. 새로운 Personalized Motion Guidance Framework (PMGF)를 개발하여, 생성적 인공지능 기술을 활용해 개별 운동 선수의 고유한 움직임 패턴에 최적화된 운동 세부 지침을 생성합니다.

- **Technical Details**: PMGF는 수직 오토인코더 (vertical autoencoder)를 활용하여 운동 시퀀스를 운동 선수별 잠재 표현 (latent representations)으로 인코딩합니다. 두 가지 조작 전략이 탐색되었으며, 하나는 학습자의 움직임과 전문가의 움직임 간의 부드러운 보간 (smooth interpolation)으로 관찰 학습을 촉진하고, 다른 하나는 로컬 최적화 기법을 사용하여 잠재 공간에서 운동 패턴을 최적의 방향으로 이동하는 것입니다.

- **Performance Highlights**: 51명의 야구 투수 데이터를 사용한 검증 실험에서 PMGF는 1,275 쌍의 투수 간에 운동 패턴의 부드러운 전환을 성공적으로 생성했습니다. PMGF 조작을 통해 크게 변경된 특징은 더 긴 보폭 (stride length)과 높은 공속도 (ball velocity)와 관련된 무릎 신전 (knee extension)과 같은 성능 향상 특성을 반영하여 생체역학적으로 그럴듯한 개선이 이루어졌음을 나타냅니다.



### SASER: Stego attacks on open-source LLMs (https://arxiv.org/abs/2510.10486)
- **What's New**: 본 논문은 오픈 소스 LLMs(대형 언어 모델)에 대한 새로운 스테고 공격 방식을 제안합니다. SASER로 불리는 이 공격은 특정 매개변수를 식별하고, 페이로드(payload)를 포함시키며, 트리거(trigger)를 주입하여 실행하는 세 단계의 작업 흐름을 따릅니다. 기존의 스테고 공격과의 차별점은 SASER가 양자화된 모델 배포에 대한 공격 강건성을 높이기 위해 페이로드의 비양자화(de-quantization)를 수행한다는 점입니다. 

- **Technical Details**: SASER는 세 가지 주요 단계인 TARGET, LAUNCH 및 EXPLODE 단계로 구성됩니다. TARGET 단계에서는 성능 인지 중요도(performance-aware importance, PAI)를 정의하여 매개변수의 중요성을 정량화합니다. LAUNCH 단계에서는 일반 모드와 강건 모드를 통해 페이로드를 주입하며, EXPLODE 단계에서 주입된 페이로드를 실행합니다. 이 공격은 오픈 소스 LLM의 복잡한 메커니즘에 맞춤화된 새로운 기술이 필요하며, 실제 배포 과정에서 발생할 수 있는 취약점을 고려합니다.

- **Performance Highlights**: SASER는 LLaMA2-7B와 ChatGLM3-6B 모델에서 실험을 통해 기존 스테고 공격보다 최대 98.1% 더 높은 스텔스 비율을 기록했습니다. 특히, 양자화된 모델에서의 공격 성공률(ASR)은 0%에서 100%로 향상되었습니다. 이러한 결과는 SASER가 오픈 소스 LLMs에 대한 공격의 효과성을 높일 수 있는 혁신적인 접근임을 보여줍니다. 논문은 이러한 공격을 방어하기 위한 필사적인 연구 필요성을 강조하며 마무리됩니다.



### UltraLLaDA: Scaling the Context Length to 128K for Diffusion Large Language Models (https://arxiv.org/abs/2510.10481)
- **What's New**: 본 논문에서는 기존의 훈련 없이도 긴 맥락(window) 처리를 가능하게 하는 기법을 연구합니다. 특히, Rotary Positional Embeddings (RoPE)을 조정하여 diffusion LLMs의 긴 컨텍스트 성능을 극대화할 수 있는 방법을 제시합니다. 이를 통해 UltraLLaDA라는 새로운 모델을 도입하여 128K 토큰의 컨텍스트 윈도우를 지원합니다.

- **Technical Details**: diffusion LLM은 기존의 auto-regressive LLM과는 다르게, 전체 시퀀스에 대해 반복적인 노이즈 제거 과정을 사용하여 텍스트를 생성합니다. 로타리 포지셔널 임베딩(RoPE)의 단순한 수정으로 긴 입력 시퀀스에서의 불확실성 모델링을 지원하여, 최적화 안정성과 장기 기억력에 미치는 마스킹 전략을 분석합니다. 저자들은 UltraLLaDA의 구현을 통해 이러한 방법들이 효율적인 포스트 트레이닝(post-training)에서 중요하게 작용함을 보여줍니다.

- **Performance Highlights**: UltraLLaDA는 다양한 긴 맥락 작업을 수행하는 벤치마크에서 뛰어난 성능을 보이며, 훈련 없는 방식인 LongLLaDA와 기존의 LLaDA 모델을 넘어서는 결과를 나타냅니다. 실험 결과에 따르면, UltraLLaDA는 긴 컨텍스트를 처리하면서도 낮은 perplexity와 높은 태스크 정확도를 유지합니다. 이러한 결과들은 UltraLLaDA의 최신 긴 맥락 처리 능력과 경량화된 포스트트레이닝 접근 방식의 실용적 이점을 강조합니다.



### Latent Retrieval Augmented Generation of Cross-Domain Protein Binders (https://arxiv.org/abs/2510.10480)
- **What's New**: RADiAnce는 기존 인터페이스를 활용하여 새로운 단백질 결합체를 디자인하는 새로운 프레임워크로, 여러 메트릭에서 기초 모델들보다 월등한 성능을 발휘하고 있습니다. 특히, 이 모델은 결합부위에 대한 조건부 잠재 확산 생성기를 통해 다양한 도메인 간 인터페이스 전송을 가능하게 합니다. 이로 인해 약물 발견 분야에서의 새로운 가능성을 열어준다는 점이 돋보입니다.

- **Technical Details**: RADiAnce는 대조적 잠재 공간에서 검색(retrieval)과 생성을 통합하는 방식으로 작동합니다. all-atom 변분 오토인코더(VAE)가 보고서의 상호작용 정렬이 가능한 잠재 공간을 생성하며, 검색된 인터페이스 임베딩들을 통해 생성을 지도합니다. 이러한 방법은 상호작용의 공유를 효과적으로 포착할 수 있는 유사성 메트릭을 필요로 하며, cross-attention 및 잔여 MLP를 사용하여 기존 지식을 통합합니다.

- **Performance Highlights**: 실험 결과 RADiAnce는 펩타이드 및 항체 디자인 작업에서 기존 강력한 기준 모델들에 비해 구조 및 상호작용 패턴을 회복하는 데 있어 유의미한 개선을 보여주었습니다. 또한, 항체 및 펩타이드와 같은 다양한 도메인에서 인터페이스를 검색함으로써, 교차 도메인 전이의 합리성을 입증하고 있습니다. 이로 인해 생성 성능이 향상된다는 사실이 강조되고 있습니다.



### Assessing Large Language Models for Structured Medical Order Extraction (https://arxiv.org/abs/2510.10475)
- **What's New**: MEDIQA-OE 2025 Shared Task는 의사-환자 대화에서 의학적 명령(orders)을 추출하는 데 중점을 두고 있습니다. 이 연구는 LLaMA-4 모델을 사용하여 문맥 내에서의 예시(prompt engineering)를 통해 비전문 영역에서 학습된 대규모 모델의 성능을 평가합니다. 17개 팀 중 5위에 해당하는 성과를 기록하며, 임상 NLP 과제에서의 새로운 기준을 제시합니다.

- **Technical Details**: 제안된 접근법은 메타(Meta)의 LLaMA-4 17B 모델을 사용하며, 소수의 예시를 기반으로 한 프롬프트 엔지니어링(few-shot prompt engineering) 방법론을 적용합니다. 도메인 특화 훈련 없이, 일반 모델을 사용하여 의학적 명령의 유형, 설명, 이유를 식별하는 데 중점을 뒀습니다. 이 접근 방식은 의료 분야에서 탁월한 성과를 거둘 수 있는 대규모 LLM의 가능성을 보여줍니다.

- **Performance Highlights**: 제출된 모델은 평균 F1 점수 37.76을 기록하였고, 특히 명령의 이유(reason)와 출처(provenance) 정확성에서 두드러진 개선을 보였습니다. 이는 효과적인 프롬프트 엔지니어링을 통해 비전문 대규모 모델이 임상 NLP 과제에서 강력하고 확장 가능한 기준선으로 기능할 수 있음을 나타냅니다.



### FML-bench: A Benchmark for Automatic ML Research Agents Highlighting the Importance of Exploration Breadth (https://arxiv.org/abs/2510.10472)
Comments:
          Our benchmark is available at: this https URL

- **What's New**: 최근 대형 언어 모델(LLMs)의 발전으로 인해 자동화된 머신러닝(ML) 연구 에이전트에 대한 관심이 다시 높아지고 있습니다. 이들 에이전트는 과학적 발견 과정의 여러 부분을 돕거나 수행하는 데 매우 효과적입니다. 특히 자율적으로 아이디어를 제안하고 실험을 수행할 수 있는 에이전트들은 연구 자동화를 극대화하고 과학적 진전을 가속화하는데 중요한 역할을 합니다.

- **Technical Details**: 본 논문에서는 FML-bench라는 새로운 벤치마크를 소개합니다. 이 벤치마크는 8가지 다양한 기초 ML 문제를 평가하는 데 초점을 맞추고 있으며, 자동 ML 연구 에이전트를 코드 부담을 줄이면서 평가할 수 있도록 설계되었습니다. FML-bench의 설계 원칙은 네 가지로, 핵심 과학적 문제에 초점을 맞추고, 실제 코드베이스를 기반으로 하며, 확장성이 뛰어나고 낮은 코딩 장벽을 제공합니다.

- **Performance Highlights**: FML-bench에서 여러 최신 자동 연구 에이전트들을 평가한 결과, 탐색의 폭을 확장하는 것이 단일한 아이디어 개선보다 더 효과적인 연구 결과를 가져온다고 밝혀졌습니다. Gemini-2.5-Pro가 우리의 프로토콜에 따라 GPT-5보다 우수한 성능을 보였으며, CLI 스타일의 에이전트들은 자동화된 머신러닝 연구에 있어 한계가 있음을 시사합니다. 이러한 결과는 효과적인 에이전트 설계를 위한 실질적인 지침을 제공합니다.



### AnyBCQ: Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs (https://arxiv.org/abs/2510.10467)
- **What's New**: 이 논문에서는 AnyBCQ라는 하드웨어 친화적인 다중 정밀도(mult-precision) 양자화 프레임워크를 소개합니다. 이는 Binary-Coded Quantization (BCQ)의 확장을 통해 다중 정밀도 작업을 실질적으로 지원하면서도 비트-플레인(bit-plane) 수준에서 직접 연산을 수행할 수 있습니다. 또한, AnyBCQ는 메모리와 지연 시간이라는 제약을 극복할 수 있는 유연성을 제공합니다.

- **Technical Details**: AnyBCQ는 가중치를 이진 비트-플레인으로 표현하고 각 비트-플레인에 대해 대응하는 스케일 팩터(scale factor)를 할당하여 구성됩니다. 이 구조는 하드웨어 가속기를 통해 효율적으로 매핑될 수 있으며, 계산 효율성을 높입니다. 또한 AnyBCQ는 비트-플레인을 기반으로 한 인코딩을 지원하며, 추가 비트를 활성화할 때마다 정확도를 점진적으로 개선할 수 있는 메커니즘을 포함하고 있습니다.

- **Performance Highlights**: 실험 결과 AnyBCQ는 저 정밀도 단계에서의 정확도 감소를 크게 줄이며(예: 2-bit), 높은 정밀도에서 경쟁력을 유지합니다. 또한 AnyBCQ는 반 정밀도와 비즈니스 최적화의 최신 방법들보다 최대 3.0배의 처리량 향상을 달성하였습니다. 이로써 다양한 서비스 레벨 목표를 충족하는 데 있어 매우 실제적인 기초를 제공합니다.



### LightSAE: Parameter-Efficient and Heterogeneity-Aware Embedding for IoT Multivariate Time Series Forecasting (https://arxiv.org/abs/2510.10465)
Comments:
          Submitted to IEEE IoT-J

- **What's New**: 본 연구에서는 Shared-Auxiliary Embedding (SAE) 프레임워크를 소개하여, 다변량 시계열 예측(MTSF)의 정확성을 높이기 위한 새로운 접근법을 제시합니다. 기존의 방법들이 모든 채널에 동일한 임베딩 레이어를 적용하면서 중요한 채널 특성을 고려하지 못하는 문제를 해결하고자 하는 것입니다. SAE 구조는 공통 패턴을 포착하는 공유 기본 구성 요소와 각 채널의 고유한 변화를 모델링하는 보조 구성 요소로 분해됩니다.

- **Technical Details**: LightSAE는 저계수(low-rank) 분해와 공유 게이트 구성 요소 풀을 통해 파라미터 효율적인 임베딩 모듈을 설계합니다. 이는 채널 특정 특성을 효과적으로 모델링하면서도 파라미터 수를 최소화합니다. SAE의 분석을 통해 보조 구성 요소가 저계수 및 클러스터링 특성을 나타내는 경향이 있음을 관찰하였으며, LightSAE는 이러한 구조적 패턴을 활용하여 성능을 극대화합니다.

- **Performance Highlights**: 실험 결과, 9개의 IoT 관련 데이터셋과 4개의 백본 아키텍처를 통해 LightSAE는 기존 방법에 비해 최대 22.8%의 MSE 개선을 달성하며, 파라미터 수는 단 4% 증가에 그쳤습니다. 이와 같은 성과는 다변량 시계열 데이터의 채널 이질성을 효과적으로 처리하는 데 있어 LightSAE의 효율성을 입증합니다.



### Testing and Enhancing Multi-Agent Systems for Robust Code Generation (https://arxiv.org/abs/2510.10460)
Comments:
          19pages, 5 figures

- **What's New**: 이 논문은 코드 생성을 위한 다중 에이전트 시스템(MASs)의 견고성을 조사한 첫 번째 포괄적 연구를 소개합니다. 기존의 코드 생성 프로세스를 효율적으로 분해하여 전문화된 에이전트에게 작업을 분배하는 MAS의 그러나 로버스트니스(robustness)가 제대로 탐구되지 않았음을 밝혔습니다. 이를 위해 저자들은 퍼징(fuzzing) 기반의 테스트 접근법을 사용하여 일반적인 MAS의 견고성 문제를 평가하고, 해당 문제의 주요 원인으로 계획자-코더 간의 의사소통 격차(planner-coder gap)를 제시합니다.

- **Technical Details**: 저자들은 입력 질문에 대한 의미 보존(mutation operators)을 활용하여 퍼징 파이프라인을 설계하고, MASs가 생성한 계획(plan)과 코드(code)의 편차를 평가하는 피트니스 함수(fitness function)을 도입했습니다. 이 연구는 7.9%부터 83.3%의 문제를 해결하지 못하는 여러 인기 있는 MAS의 견고성 결함을 발견했습니다. 논문에서는 MAS의 견고성을 높이기 위해 다중 프롬프트 생성(multi-prompt generation)과 새로운 모니터 에이전트(monitor agent)를 도입한 수리 방법을 제안합니다.

- **Performance Highlights**: 저자들의 수리 방법은 세 개의 주요 MAS에서 40.0%에서 88.9%까지의 발견된 결함을 성공적으로 해결했습니다. 수리된 MAS들은 후속 퍼징 과정에서 실패 수가 85.7%까지 감소한 것으로 나타났습니다. 또한, 다중 프롬프트 생성과 모니터 에이전트를 통한 접근법의 필요성이 실험을 통해 입증되었습니다.



### NIM: Neuro-symbolic Ideographic Metalanguage for Inclusive Communication (https://arxiv.org/abs/2510.10459)
Comments:
          9 pages, EMNLP Findings 2025

- **What's New**: 이번 논문에서는 디지털 커뮤니케이션의 "디지털 격차" 문제를 해결하기 위한 새로운 보편적인 아이디어 메탈랭귀지(ideographic metalanguage)를 소개합니다. 이는 인지적인 요소를 통합하는 신경-기호 AI(Neuro-symbolic AI) 원리를 활용하여 구성되었으며, 세계 지식과 기호적 지식의 휴리스틱을 결합하여 복잡한 개념을 단순한 원자적 개념으로 분해할 수 있게 합니다. 이 시스템은 200명 이상의 반정형 교육을 받은 참여자들과의 협업을 통해 문제를 정의하고 아이디어 그래프를 선택, 검증하는 과정을 거쳤습니다.

- **Technical Details**: 제안된 아이디어 메탈랭귀지는 NIM(Neuro-symbolic Ideographic Metalanguage)이라는 이름으로, 복잡한 생각을 단순한 입자 개념으로 분해할 수 있는 구조를 따릅니다. 이 과정에서 시각적 요소와 텍스트 요소를 구분하여 처리하며, 기호적 추론 모듈은 Natural Semantic Metalanguage(NSM) 이론을 바탕으로 세분화된 의미적 단순화를 지원합니다. 또한, 본 시스템은 다양한 사용자 그룹에 적용 가능하며, 다국어 지원을 통해 보편적인 사용성을 갖추고 있습니다.

- **Performance Highlights**: 이번 연구의 결과, 참가자들은 제안된 시스템의 80% 이상을 이해할 수 있었으며, 직관적인 학습 곡선을 제공하였습니다. 이를 통해 저학력 인구가 디지털 커뮤니케이션에 잘 적응할 수 있도록 돕는 강력한 도구가 될 것으로 기대됩니다. 또한 이 시스템은 지적 장애인, 다국어 팀 및 읽기 어려움이 있는 아동과 같은 다양한 사용자 집단에도 적용 가능성이 있습니다.



### Data-driven simulator of multi-animal behavior with unknown dynamics via offline and online reinforcement learning (https://arxiv.org/abs/2510.10451)
Comments:
          21 pages, 7 figures

- **What's New**: 이번 연구에서는 다중 동물 행동을 위한 새로운 데이터 기반 시뮬레이터인 AnimaRL을 도입했습니다. 이 시뮬레이터는 딥 강화 학습(deep reinforcement learning)과 반사실적 시뮬레이션(counterfactual simulation)에 기반하여 다중 동물의 이동 동역학을 추정하는 혁신적인 접근 방식을 사용합니다. 이러한 방법을 통해 실제 생물학적 환경에서 관찰되는 복잡한 동물 행동을 정밀하게 시뮬레이션할 수 있는 가능성을 제시합니다.

- **Technical Details**: AnimaRL은 이동 매개변수 추정, 오프라인 정책 학습(offline policy learning), 온라인 정책 조정(online policy adjustment), 시뮬레이션 환경 인터페이스와 같은 여러 핵심 모듈로 구성되어 있습니다. 이 프레임워크는 실제 동물 행동 데이터(trajectory 및 reward)를 입력으로 받아 딥 Q 네트워크(Deep Q-Network)와 거리 기반 의사 보상(pseudo-reward)을 적용하여 강화 학습 알고리즘에 적합성을 높입니다. 또한 강화 학습 프레임워크 내에서 이동 변수를 행동으로 추정하여 문제를 해결합니다.

- **Performance Highlights**: AnimaRL은 다양한 생물체에 대해 높은 재현성을 달성했습니다. 기존의 모방 기반(imitation) 및 강화 학습(RL) 기술과 비교했을 때, 종 특유의 행동을 보다 잘 재현하고 보상 획득(reward acquisition)을 개선했습니다. 또한 이 시뮬레이터는 다양한 실험 설정에서 반사실적 행동 예측(counterfactual behavior prediction)이 가능하게 하여, 다중 개체 모델링을 지원하고 유연한 경로 생성(trajectory generation)의 잠재력을 보여주었습니다.



### Reverse Supervision at Scale: Exponential Search Meets the Economics of Annotation (https://arxiv.org/abs/2510.10446)
Comments:
          10 pages

- **What's New**: 이번 연구는 라벨이 없는 대규모 데이터셋(B)의 라벨링을 탐색하여 소규모 라벨이 있는 데이터셋(A)에서 오류를 최소화하는 역감독(Reverse Supervision) 전략을 분석합니다. 이는 라벨의 품질과 주제에 대한 명확한 목표가 필요함을 강조합니다. 생성적 AI로부터 생성된 라벨은 일부 대체 가능하지만, 여전히 초기 인간의 개입이 필요하다는 점을 지적합니다.

- **Technical Details**: 연구는 감독(labeled data)과 비용(cost) 중심의 관점을 전환하여, 데이터셋 크기보다 데이터셋 비용에 중점을 두었습니다. 이 과정은 '줄이기(Reduce)', '재사용(Reuse)', '재활용(Recycle)'을 포함한 세 가지 부분으로 구성된 실용적인 청사진을 제시합니다. 반감독(semi-supervised learning), 전이 학습(transfer learning), 약한 감독(weak supervision)을 활용하여 전체 훈련 데이터의 효과적인 비용을 줄이는 방법을 정립합니다.

- **Performance Highlights**: 연구는 고품질 라벨 세트를 시작으로, SSL 및 능동적 선택을 통해 비용을 줄이고, 관련 백본에서 전이를 통해 재사용하며, 인간의 감독하에 약한 신호 및 합성 예제를 재활용하는 순환적인 파이프라인을 제안합니다. 이를 통해 실질적으로 더 적은 비용으로 더 나은 정확도를 유지할 수 있는 기회를 제공합니다. 결과적으로 데이터 포인트 수가 줄어드는 것뿐만 아니라, 비용이 많이 드는 데이터의 양도 줄입니다.



### Do Audio LLMs Really LISTEN, or Just Transcribe? Measuring Lexical vs. Acoustic Emotion Cues Relianc (https://arxiv.org/abs/2510.10444)
- **What's New**: 이번 연구에서는 LISTEN(Lexical vs. Acoustic Speech Test for Emotion in Narratives)이라는 새로운 벤치마크를 소개하며, 감정 이해에서 레키컬(lexical) 의존성을 음향(acoustic) 민감성과 분리하는 것을 목표로 하고 있습니다. 현재의 대형 오디오 언어 모델(LALMs)이 실제로 음향 정보를 어떻게 처리하는지 불확실성을 해소하고자 하며, 이는 감정 인식의 한계를 명확히 드러냅니다. 특히, LISTEN은 감정 이해에서 LALMs의 진정한 듣기 능력을 평가하는 기준을 제공합니다.

- **Technical Details**: LISTEN은 네 가지 제어된 조건을 갖춘 평가 프레임워크로 구성되며, 이는 (i) Neutral-Text(중립 텍스트), (ii) Emotion-Matched(감정 일치), (iii) Emotion-Mismatched(감정 불일치), 그리고 (iv) Paralinguistic(부언어적)입니다. 각 조건 내에서 텍스트, 오디오, 텍스트+오디오 모드를 설정하여 음향 정보 처리 능력을 평가합니다. 이러한 설계는 LALMs가 레키컬 정보보다 음향 신호를 진정으로 수용하는지를 조사할 수 있도록 돕습니다.

- **Performance Highlights**: 연구 결과, 여러 최첨단 LALMs의 성능을 비교한 결과, 레키컬 우위가 뚜렷하게 나타났습니다. LALMs는 레키컬 신호가 중립적일 때 '중립' 예측을 하고, 단서 정렬 시 제한적인 성과를 보이며, 단서가 상충할 경우 distinct 감정 분류에 실패합니다. 특히 이러한 결과는 기존의 LALMs가 '들리는' 것이 아니라 주로 '옮기는' 방식으로 작동하며, 음향 신호를 충분히 활용하지 않음을 시사합니다.



### Multi-Task Learning with Feature-Similarity Laplacian Graphs for Predicting Alzheimer's Disease Progression (https://arxiv.org/abs/2510.10433)
- **What's New**: 이번 연구에서는 Alzheimer’s Disease(AD) 데이터의 시간 변화(temporal) 특성을 효과적으로 모델링하기 위해 Feature Similarity Laplacian 그래프를 활용한 새로운 Multi-Task Learning (MTL) 프레임워크인 MTL-FSL을 제안합니다. 기존의 MTL 방법들은 특성 간의 관계를 충분히 반영하지 못했지만, MTL-FSL은 시간에 따라 변화하는 특성 간의 상관관계를 명시적으로 모델링합니다. 이를 통해 예측 정확도와 생물학적 해석 가능성을 동시에 개선할 수 있습니다.

- **Technical Details**: MTL-FSL 프레임워크는 Feature Similarity Laplacian(FSL) 패널티를 도입하여 연관된 여러 작업 간의 시간 변화하는 관계를 효율적으로 고려합니다. 또한, Alternating Direction Method of Multipliers(ADMM) 알고리즘을 사용하여 비부드러운 최적화 문제를 해결합니다. 이 접근 방식은 데이터로부터 파라미터를 효과적으로 추정하고 예측의 신뢰성을 높이는 데 기여합니다.

- **Performance Highlights**: Alzheimer’s Disease Neuroimaging Initiative(ADNI) 데이터셋을 이용한 실험에서, MTL-FSL 프레임워크는 다양한 기준 방법들보다 뛰어난 성능을 보였습니다. 이 모델은 여러 다른 방법들과 비교하여 인지 점수의 예측 정확도를 현저히 개선하였으며, 연구 결과는 생물학적 해석 가능성과 임상적 가치에서 유의미한 의미를 가집니다.



### Hierarchical LoRA MoE for Efficient CTR Model Scaling (https://arxiv.org/abs/2510.10432)
Comments:
          13 pages, 9 figures

- **What's New**: 이번 논문에서는 CTR 예측을 위한 효율적이고 확장 가능한 모델 설계를 제안합니다. 이를 위해 HiLoMoE라는 계층적 LoRA MoE 프레임워크를 도입하여, 수직적 및 수평적 확장을 모두 가능하게 합니다. 이 모델은 경량화된 rank-1 전문가를 사용하여 매개변수 효율성을 높이고, 계층적 라우팅을 통해 전문가 조합을 다양화합니다.

- **Technical Details**: HiLoMoE는 전문가 선택을 이전 레이어의 라우팅 점수를 기반으로 하여 모든 레이어를 병렬적으로 실행할 수 있게 합니다. 이 시스템은 세 가지 핵심 혁신으로 구성되며, LoRA 전문가를 통해 매개변수 감소를 이루고 계층적 라우팅 메커니즘을 통해 수직적 확장을 지원합니다. 또한 이 복잡한 시스템을 학습하기 위해 세 단계의 학습 파이프라인을 제안하고, 보조 손실을 추가하여 전문가의 다양성을 강화합니다.

- **Performance Highlights**: 네 개의 공개 데이터셋에서의 실험 결과, HiLoMoE는 비 MoE 모형에 비해 평균적으로 AUC를 0.20% 개선하고, FLOPs는 18.5% 감소하는 성능-효율성 거래를 달성했습니다. 이 모델은 깊이와 폭을 확장하는 데 있어 뛰어난 성능을 보이며, 레이어 수와 전문가 수가 증가할수록 성능이 개선되는 경향을 보입니다.



### LONGQAEVAL: Designing Reliable Evaluations of Long-Form Clinical QA under Resource Constraints (https://arxiv.org/abs/2510.10415)
- **What's New**: 이 논문에서는 LongQAEval이라는 새로운 평가 프레임워크를 도입하여 제한된 자원과 높은 전문성을 요구하는 환경에서의 오랜 형식의 임상 질문 응답(QA) 시스템 평가를 용이하게 하고자 하였습니다. 300개의 실제 환자 질문에 대한 의사와 LLM의 답변을 바탕으로 정확성(correctness), 관련성(relevance), 안전성(safety) 차원에서 평가를 비교하였고, 세부적인 문장 수준 평가가 더 큰 합의(inter-annotator agreement, IAA)를 이끌어내는 결과를 얻었습니다.

- **Technical Details**: LongQAEval 프레임워크는 답변의 정확성, 관련성 및 안전성을 평가하는 기준을 명확히 정의합니다. 평가자는 전체 답변을 평가하는 coarse 디자인과 개별 문장을 평가하는 fine-grained 디자인 두 가지 방법으로 답변을 평가하도록 지시받았습니다. 연구 결과, 세부적인 문장 수준의 주석은 사실 기반 정확성에서 IAA를 향상시키고, 전반적인 문맥 기반 평가에서는 coarse 주석이 더 유리하며, 부분 평가를 통해 비용을 줄일 수 있다는 것을 발견하였습니다.

- **Performance Highlights**: GPT-4와 Llama-3.1-Instruct-405B 모델이 임상 질문에 대해 생리학적 정확성과 관련성에서 의사와 유사한 성과를 보였습니다. 특히, 세부적인 평가 방법이 신뢰성을 높이는 데 기여하며, LLM이 제공하는 답변이 길이에 따른 편향을 상쇄하는 데 효과적임을 보여주었습니다. 이러한 결과는 연구자들에게 평가 차원에 맞는 주석 디자인을 맞추어야 한다는 중요한 시사점을 제공합니다.



### Controllable Graph Generation with Diffusion Models via Inference-Time Tree Search Guidanc (https://arxiv.org/abs/2510.10402)
- **What's New**: 이 논문에서는 제어 가능한 그래프 생성을 위한 새로운 방법론인 TreeDiff를 제안합니다. TreeDiff는 몬테 카를로 트리 탐색(MCTS) 기반의 이중 공간 확산(diffusion) 체계를 도입해 샘플링 과정을 조정할 수 있는 점이 특징입니다. 이 방법은 기존의 조건 없는 확산 모델의 한계를 극복하고, 연산 효율성을 개선하여 보다 안정적이고 제어 가능한 생성 과정을 가능하게 합니다.

- **Technical Details**: TreeDiff는 세 가지 주요 설계를 포함합니다: 첫째, 매크로 단계 확장 전략(macro-step expansion strategy)을 통해 여러 개의 디노이징(denoising) 업데이트를 하나의 변환으로 그룹화하여 트리 깊이를 줄이고 긴 탐색을 가능하게 합니다. 둘째, 이중 공간 디노이징 메커니즘(dual-space denoising mechanism)은 그래프 공간에서의 가벼운 수정과 함께 효율적인 잠재 공간(latent-space) 디노이징을 결합하여 확장성과 구조적 충실성을 보장합니다. 셋째, 이중 공간 검증자(dual-space verifier)는 부분적으로 디노이징된 그래프에서 장기 보상을 예측하여 조기 가치 추정을 가능하게 합니다.

- **Performance Highlights**: TreeDiff는 2D 및 3D 분자 생성 벤치마크에서 최첨단 성능을 달성했습니다. 콘텐츠가 증가함에 따라 TreeDiff의 성능이 향상되는 반면, 기존 방법들은 연산 자원에 제한을 받을 때 성능이 머무는 현상을 보입니다. 이 결과는 TreeDiff가 더 많은 계산 자원을 활용하여 지속적으로 개선될 수 있는 잠재력을 가지고 있음을 보여줍니다.



### STEAM: A Semantic-Level Knowledge Editing Framework for Large Language Models (https://arxiv.org/abs/2510.10398)
Comments:
          Accepted to EMNLP 2025 (Findings)

- **What's New**: 이 논문에서는 기존의 지식 편집 방법의 한계를 극복하기 위해 새로운 세멘틱 지식 편집 프레임워크인 	extsc{Steam}을 제안합니다. 최종 목표는 편집된 지식을 모델의 지식 구조에 자연스럽게 통합하는 것입니다. 기존 방식이 주로 토큰 수준의 가능성 최적화에 치중하는 반면, 	extsc{Steam}은 의미론적 일관성을 강화하는 데 중점을 두었습니다.

- **Technical Details**: 	extsc{Steam} 프레임워크는 수정을 위한 두 가지 주요 구성 요소를 포함합니다: (1) Latent Positioning, 이는 편집된 지식에 대한 의미적 앵커를 식별하고, (2) Latent-Level Alignment, 이는 편집된 사실의 내부 표현을 이러한 앵커로 안내하는 역할을 합니다. 이러한 접근 방식을 통해 모델의 레이턴트 공간에서 의미론적 통합을 촉진하여, 수정된 사실에 대한 추론 능력을 개선합니다.

- **Performance Highlights**: 실험 결과, 	extsc{Steam}은 모델이 편집된 지식으로 더 잘 추론할 수 있게 하였고, 전반적인 의미론적 일관성을 향상시켰습니다. 다양한 기준선과 편집 환경에서 이러한 개선 사항은 일관되게 나타났으며, 신뢰할 수 있는 지식 편집을 위해서는 의미 수준의 통합이 중요함을 드러냈습니다.



### RefusalBench: Generative Evaluation of Selective Refusal in Grounded Language Models (https://arxiv.org/abs/2510.10390)
- **What's New**: 이 연구에서는 Retrieval-Augmented Generation(RAG) 시스템에서 언어 모델이 잘못된 맥락에 따라 selectively refuse(선택적 거부)를 수행하는 능력이 얼마나 중요한지를 보여줍니다. 연구진은 기존 모델들이 이 기능에서 50% 미만의 정확도를 기록하고, 잘못된 정보에 기반하여 답변을 거부하거나 자신이 없는 답변을 내는 문제를 밝혀냈습니다. 또한, 단순한 정적 벤치마크(static benchmarks)가 이러한 성능을 평가하는 데 한계를 가지고 있음을 강조하며, RefusalBench라는 새로운 평가 방법론을 소개합니다.

- **Technical Details**: RefusalBench는 언어적 교란을 통해 진단 테스트 케이스를 생성하는 프로그램 수립 방법론을 기반으로 하고 있습니다. 이 시스템은 정보 불확실성의 여섯 가지 범주에서 176개의 교란 전략을 활용하여 답변 가능한 질문을 답변 불가능한 질문으로 변화시킵니다. 이러한 평가 방법론은 고유성에 대한 감도를 세밀하게 진단할 수 있으며, 멀티 모델 생성-검증 파이프라인을 통해 정답의 품질을 보장합니다.

- **Performance Highlights**: 30개 이상의 모델을 평가한 결과, 선택적 거부 능력에서 심각한 차이가 발견되었습니다. 연구진은 이 능력이 훈련이 가능하고 조정에 민감한 특성을 지니고 있다는 것을 밝혀내어 모델 개선의 길을 제시하였습니다. 또한, RefusalBench-NQ(단일 문서) 및 RefusalBench-GaRAGe(다중 문서)라는 두 가지 벤치마크를 제공하며, 이러한 새로운 평가 프레임워크의 필요성을 강조합니다.



### RobotFleet: An Open-Source Framework for Centralized Multi-Robot Task Planning (https://arxiv.org/abs/2510.10379)
- **What's New**: 본 논문에서는 이질적인 로봇 플릿을 중앙 집중식으로 관리하고 여러 작업을 동시에 수행할 수 있는 개방형 프레임워크인 RobotFleet을 소개합니다. RobotFleet은 대형 언어 모델(LLMs)을 활용하여 임무 수행을 위한 의존성 그래프를 생성하고, 여러 로봇 간의 작업 할당 및 실행을 조율합니다. 이 프레임워크는 각 로봇의 능력을 쉽게 확장할 수 있도록 컨테이너화된 서비스로 배포됩니다.

- **Technical Details**: RobotFleet의 핵심은 세 가지 모듈인 LLM 기반 작업 계획, 중앙 집중식 작업 할당, 분산 실행을 통한 컨테이너 로봇 에이전트입니다. 시스템은 계층적 아키텍처로 설계되어 이질적인 로봇 플릿 간의 작업 실행을 조율할 수 있습니다. 또한, 사용자 정의 작업 계획 로직 및 세계 표현을 수정할 수 있는 명령 줄 인터페이스(CLI)를 사용하며, 이는 낮은 수준의 통신 관리 없이 쉽게 구현할 수 있습니다.

- **Performance Highlights**: RobotFleet은 다중 로봇 시스템을 구축하는 데 필요한 장벽을 낮추며, 역동적인 환경에서 작업 관리와 갱신을 용이하게 합니다. 이전 연구들과 달리 이 프레임워크는 고정된 능력을 가정하지 않고 다양한 목표에 쉽게 확장될 수 있습니다. 연구자들과 개발자들이 분산 로봇 실행의 복잡성을 관리하는 대신 시스템 동작 개발에 집중할 수 있도록 돕는 것을 목표로 합니다.



### Measuring What Matters: Connecting AI Ethics Evaluations to System Attributes, Hazards, and Harms (https://arxiv.org/abs/2510.10339)
- **What's New**: 최근 10년 동안 AI 시스템의 사회적 및 윤리적 영향을 평가하기 위한 여러 기준이 출현했으나, 이러한 기준들은 대부분 조각적으로 개발 및 활용되고 있습니다. 본 논문에서는 기존의 평가 도구들이 AI 시스템의 구성 요소, 속성, 위험 및 해악과 어떻게 연결되는지를 분석하였습니다. 800개에 달하는 기준이 11개의 AI 윤리 원칙에 해당함을 확인했고, 공정성(fairness), 투명성(transparency), 개인 정보 보호(privacy), 신뢰(trust) 원칙에 중점을 두었음을 드러냈습니다.

- **Technical Details**: 논문은 AI 윤리 원칙에 대한 준수 측정 기준을 평가하는 데 있어 기존 도구들의 유효성과 신뢰성이 결여되어 있다고 지적합니다. 대부분의 측정 기준이 개별적인 구성 요소에 집중하고 있어 시스템 전체를 평가하지 못하고 있으며, 이는 안전성 측면에서 문제를 일으킬 수 있습니다. 저자들은 이러한 문제를 해결하기 위해 시스템 안전(system safety) 관점에서 윤리적 고려를 포함한 다차원 분석을 제안하고 있습니다.

- **Performance Highlights**: 현재의 평가 관행은 분산되어 있으며, 해악이 발생하는 위치와 관련된 기준이 부족합니다. 저자들은 AI 커뮤니티에 시스템 차원의 평가 방식을 채택할 것을 촉구하며, 이를 통해 법적 감독을 강화하고 실용적인 지침을 제공할 수 있음을 강조합니다. 마지막으로, 연구진은 각 구성 요소, 속성, 위험 및 해악에 따라 분류된 데이터 세트를 제공하여 추가 연구와 참여를 지원할 준비가 되어 있습니다.



### Towards Safe Maneuvering of Double-Ackermann-Steering Robots with a Soft Actor-Critic Framework (https://arxiv.org/abs/2510.10332)
Comments:
          4 pages, 3 figures, 2 tables, Accepted for Safety of Intelligent and Autonomous Vehicles: Formal Methods vs. Machine Learning approaches for reliable navigation (SIAV-FM2L) an IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025) workshop

- **What's New**: 본 논문에서는 Soft Actor-Critic (SAC) 기반의 깊은 강화학습(framework) 프레임워크를 통해 double-Ackermann-steering mobile robots (DASMRs)의 안전하고 정밀한 조작을 제안합니다. 기존의 비홀론 미적(Bot) 로봇과는 달리, DASMR은 강력한 운동학적 제약을 안고 있어 복잡한 환경에서는 전통적인 경로 계획 방법이 효율적이지 않습니다. Hindsight Experience Replay (HER)와 CrossQ 알고리즘을 활용하여, 장애물을 피하면서도 효율적으로 조작할 수 있도록 합니다.

- **Technical Details**: DASMR은 비홀론적 플랫폼으로, 2차원 목표 위치 𝑿𝒅에 도달하기 위한 조작을 요구받습니다. 주요 도전은 사전 전문 지식이나 미리 정의된 경로에 의존하지 않고 유효한 조작을 생성하는 것입니다. 이 문제는 마르코프 의사결정 과정(Markov Decision Process)으로 모델링되며, 각 시간 단계에서 상태를 관찰하고 행동을 선택하여 보상을 받습니다.

- **Performance Highlights**: 시뮬레이션을 통해, 제안된 프레임워크는 장애물을 회피하여 목표 위치의 97%까지 도달할 수 있는 학습된 정책을 보여주었습니다. 전통적인 경로 계획 방법에 비해 안전성과 조작 효율성을 동시에 다루는 점에서 진보된 접근 방식을 제공합니다. 결과적으로, 이 연구는 DASMRs의 복잡한 동작을 개선할 수 있는 가능성을 제시합니다.



### Mapping the Urban Mobility Intelligence Frontier: A Scientometric Analysis of Data-Driven Pedestrian Trajectory Prediction and Simulation (https://arxiv.org/abs/2510.10327)
Comments:
          5 figures

- **What's New**: 본 논문은 데이터 기반 보행자 궤적 예측 및 군중 시뮬레이션에 관한 포괄적 과학계량 분석을 통해 해당 연구 분야의 지적 진화와 학제간 구조를 시각적으로 나타냅니다. 연구는 Web of Science Core Collection의 서지 데이터(bibliometric data)를 활용하여 인공지능, 도시 정보학(urban informatics), 군중 행동 모델링(crowd behavior modeling) 간의 강한 융합을 발견했습니다. 연구는 이 데이터 기반 접근 방식이 어떻게 도시 거버넌스를 풍부하게 하고, 향후 도시의 적응형 사회적 책임 모빌리티 지능을 위한 길을 열 수 있는지를 조명합니다.

- **Technical Details**: 논문의 연구 질문은 데이터를 기반으로 한 보행자 궤적 예측 및 시뮬레이션의 학제성과 국제성, 협력 및 주요 주제에 관한 것입니다. 2025년까지 Web of Science에서 인덱스된 572개의 출판물을 분석하여 핵심 연구 경향, 국제 협력 네트워크, 주요 테마 분야를 조사했습니다. 분석은 SciExplorer와 Bibliometrix(R) 도구를 사용하여 연구 경향을 추출하고 시각화하는 데 중점을 두었습니다.

- **Performance Highlights**: 연구는 연평균 18.44%의 성장률과 2017년 이후 출판물 수의 급증을 통해 데이터 기반 보행자 궤적 예측 및 시뮬레이션이 유망한 연구 방향임을 입증했습니다. 이 분야는 다양한 기술과 알고리즘이 탐구되며 급격히 성장하고 있으며, 22.9%의 국제 공동 저자 비율을 통해 글로벌 연구 네트워크의 일환으로 발전하고 있습니다. 이 연구 결과는 데이터 기반 접근 방식이 도시 계획 및 공공 안전에 미치는 영향을 의식하며, 다양한 학문 분야로 확장되고 있다는 사실을 강조합니다.



### KG-MAS: Knowledge Graph-Enhanced Multi-Agent Infrastructure for coupling physical and digital robotic environments (https://arxiv.org/abs/2510.10325)
- **What's New**: 이 논문은 사이버 물리 시스템(CPS)과 산업 4.0에서 물리적 및 디지털 환경의 통합을 위한 지식 그래프 강화 다중 에이전트 인프라(KG-MAS)를 소개합니다. KG-MAS는 중앙 집중식 지식 그래프를 활용하여 다양한 환경 간의 의미적 기반을 제공하고, 자율 에이전트가 결정을 내리고 실시간 상태 정보를 업데이트할 수 있게 합니다. 이를 통해 기존 솔루션의 한계를 극복하고, 복잡한 시스템의 통합을 단순화하는 제안을 제시합니다.

- **Technical Details**: KG-MAS는 모델 기반 아키텍처를 특징으로 하여, 의미적 설명으로부터 에이전트를 자동으로 생성할 수 있게 합니다. 또한, 이는 다양한 통신 프로토콜을 추상화하여 시스템 간의 통합을 간소화합니다. KG는 복잡한 데이터 관계를 구조적이고 쿼리 가능한 방식으로 표현할 수 있는 가능성을 제공하며, 다중 에이전트 시스템과 결합하여 CPS의 핵심 도전 과제를 해결하는 데 기여합니다.

- **Performance Highlights**: KG-MAS는 물리적 및 디지털 로봇 환경을 결합하는 강력하고 유연한 솔루션을 제공하며, 지식 그래프를 사용한 정보 교환과 조정의 통합된 표현을 가능하게 합니다. 또한, 다중 에이전트 시스템은 복잡한 상호작용을 효과적으로 관리할 수 있는 분산 조정 메커니즘을 지원하며, 전반적인 CPS 통합에서 체계적이고 동적인 조정 능력을 향상시킵니다.



### Bridging Semantics & Structure for Software Vulnerability Detection using Hybrid Network Models (https://arxiv.org/abs/2510.10321)
Comments:
          13 pages, 3 figures, 5 tables, 14 equations, accepted at the 14th International Conference on Complex Networks and Their Applications (COMPLEX NETWORKS 2025) and the conference proceedings will be published by Springer in the Studies in Computational Intelligence series

- **What's New**: 본 논문에서는 프로그램의 보안 취약점을 탐지하기 위한 새로운 하이브리드 프레임워크를 소개합니다. 이 프레임워크는 Java 프로그램의 그래프 기반 임베딩과 경량(local) LLM(large language model)을 통합하여 구조적 특성과 의미론적 추론을 결합합니다. 이 접근법은 구조적 의존성을 모델링하여 기존 분석 방법에서 놓치기 쉬운 취약점들을 보다 효과적으로 탐지합니다.

- **Technical Details**: 제안하는 방법은 프로그램을 제어 흐름 그래프(control-flow graph)로 표현하고, 여기서 노드는 프로그램 명령문이나 기본 블록을 나타내며, 엣지는 제어 또는 데이터 흐름 전환을 나타냅니다. 코드 샘플에 대해 두 개의 보완적 임베딩을 추출하며, 첫 번째는 노드의 텍스트에서 생성하고, 두 번째는 경량 LLM에서 추출하여 고차원 의미를 캡처합니다. 이러한 임베딩은 서로 다른 차원에서 공통 잠재 공간으로 변환되어 결합됩니다.

- **Performance Highlights**: 본 방법은 Java 취약점 탐지에서 93.57%의 정확도를 기록하였고, 이는 Graph Attention Network 기반의 임베딩보다 8.36%, 미리 학습된 LLM인 Qwen2.5 Coder 3B보다 17.81% 향상된 결과입니다. 이 방법은 중요한 하위 그래프를 추출하고 자연어 설명을 생성하여 개발자에게 해석 가능성을 제공합니다. 이러한 결과는 논문에서 제안한 접근법이 기존 분석 방법보다 더 효과적으로 보안 취약점을 탐지할 수 있음을 나타냅니다.



### Prepared for the Unknown: Adapting AIOps Capacity Forecasting Models to Data Changes (https://arxiv.org/abs/2510.10320)
- **What's New**: 이번 연구는 소프트웨어 조직에서 용량 관리(capacity management)를 위한 예측 모델의 재학습(retraining) 전략에 주목하고 있습니다. 특히, 데이터 변화에 기반한 재학습 방식이 주기적인 재학습보다 비용 효과적이며 대부분의 경우 비슷한 예측 정확도를 달성함을 보여줍니다. 이러한 접근 방식은 대규모 소프트웨어 조직이 예측 시스템을 개선하고, 재학습 오버헤드(overhead)를 줄이며 우수한 성능을 유지할 수 있는 통찰력을 제공합니다.

- **Technical Details**: 이 연구는 ING 은행에서 개발한 CPU 및 메모리 활용에 대한 용량 예측 모델을 기반으로 하고 있습니다. 고전적인 재학습 방식인 주기적 재학습과 데이터 변경 발생 시 재학습을 비교하여, 데이터 동향(drift)에 따른 재학습이 예측 정확도에 미치는 영향을 분석했습니다. 중요한 발견은 데이터가 급격히 변할 경우에는 주기적인 재학습이 여전히 더 나은 예측 정확도를 제공할 수 있다는 점입니다.

- **Performance Highlights**: 연구 결과는 데이터 변경 감지(drift detection) 기법을 활용한 재학습이 자원 예측 능력을 유지하면서 재학습 빈도를 줄일 수 있다는 것을 입증했습니다. 따라서, 이러한 새로운 전략을 통해 대규모 운영을 위한 자원 할당의 정확성을 높이면서도 효율성을 크게 개선할 수 있음을 나타냅니다. 이 연구의 결과는 다양한 산업에서 시간 시계열 예측 시스템의 성능 향상 및 비용 절감을 위한 가치 있는 가이드를 제공합니다.



### Sample-Efficient Online Learning in LM Agents via Hindsight Trajectory Rewriting (https://arxiv.org/abs/2510.10304)
- **What's New**: 언어 모델(LM) 에이전트는 새로운 환경에서 상호작용을 학습할 때 샘플 효율성이 낮아지는 문제를 해결하기 위해 ECHO(Experience Consolidation via Hindsight Optimization)라는 새로운 프레임워크를 도입했습니다. ECHO는 실패한 시도에서 얻은 경험을 활용하여 대체 목표를 위한 최적화된 궤적(trajectories)을 생성함으로써, 비효율적인 학습을 개선하는 데 중점을 둡니다. 이 방법은 경험 재생(replay) 메커니즘을 사용하여 언어 모델이 과거의 실패를 성공적인 경험으로 전환할 수 있도록 돕습니다.

- **Technical Details**: ECHO 시스템은 두 가지 구성 요소로 구성됩니다: 처음으로, 언어 모델을 사용하여 관련 서브 목표(subgoals)를 식별하고 최적화된 궤적을 생성하는 회상 규칙(hindsight rule)이 있습니다. 두 번째로, 압축된 궤적 표현을 기억에 유지하는 업데이트 규칙(update rule)이 포함되어 있습니다. ECHO는 기존의 경험 재생 대신 더 많은 수정 가능성을 제공하여, 실패한 궤적을 임의로 재작성(rewriting)할 수 있게 합니다.

- **Performance Highlights**: XMiniGrid 및 PeopleJoinQA와 같은 다양한 상태 유지 가능한 테스트 환경에서 ECHO를 평가한 결과, 기존의 언어 에이전트보다 최대 80% 더 높은 성능을 달성했습니다. XMiniGrid에서 ECHO는 Reflexion 및 AWM과 같은 고급 에이전트 구조를 초과한 성능을 보여주며, 새로운 환경에 대한 적응 속도가 빨라짐을 입증하였습니다. ECHO는 특히 보상이 드문 환경에서 언어 에이전트의 샘플 효율성을 극대화하는 유망한 기술입니다.



### The Algorithmic Regulator (https://arxiv.org/abs/2510.10300)
Comments:
          2 Figures

- **What's New**: 본 논문에서는 고전적인 조절자 정리(Good Regulator Theorem, GRT)를 알고리즘 정보 이론(Algorithmic Information Theory, AIT)의 틀에서 재구성합니다. 이를 통해 조절자가 시스템을 통해 정보를 어떻게 전달하고 압축하는지를 분석하며, 조절자의 역량을 새로운 관점에서 이해할 수 있는 방법을 제시합니다. 특히, 시스템과 조절자 간의 알고리즘적 의존성을 정의하고 그 관계를 정량화합니다.

- **Technical Details**: 저자들은 시스템 W와 조절자 R을 결정론적 인과적 튜링 머신으로 모델링하고, 조절자의 내부 모델의 정의를 서로 상호 알고리즘 정보(Mutual Algorithmic Information) M(W:R)으로 설정합니다. 이 논문에서는 조절자가 어떻게 알고리즘적으로 정보를 압축하고, 압축된 결과를 기반으로 판단할 수 있는지를 설명합니다. 또한 조절자가 측정된 성과를 얼마나 효과적으로 압축할 수 있는지를 정량화하여 평가합니다.

- **Performance Highlights**: 결과적으로 조절자가 세계 모델을 얼마나 잘 나타내는지를 압축된 데이터로 측정하며, 이는 예측 (prediction)과 조절(regulation) 사이의 상관관계를 드러냅니다. 조절자가 직면하는 오류의 크기를 나타내는 가중 오류 신호를 기반으로 한 정보를 통해 조절의 질적 차이를 정량화하며, 이는 알고리즘적 복잡성의 시각에서 중요한 통찰을 제공합니다. 저자들은 이 새로운 접근 방식이 각종 생물학적 및 공학적 시스템에 응용될 수 있음을 강조합니다.



### MatryoshkaThinking: Recursive Test-Time Scaling Enables Efficient Reasoning (https://arxiv.org/abs/2510.10293)
- **What's New**: 본 연구에서는 MatryoshkaThinking이라는 새로운 방법을 제안합니다. 이 방법은 언어 모델의 성능 향상을 위해 테스트 중 추가 계산 리소스를 효율적으로 활용합니다. MatryoshkaThinking은 DeepConf보다 4%의 계산만으로도 AIME2025에서 99.79의 점수를 달성합니다.

- **Technical Details**: MatryoshkaThinking은 모델의 본질적인 추론(reasoning), 검증(verification), 요약(summarization) 기능을 재귀적으로 활용하여 계산 비용을 크게 줄입니다. 이 접근법은 올바른 솔루션의 보존을 향상시키고 Pass@k와 Pass@1 간의 불일치를 줄이는 데 기여합니다. 여러 오픈 소스 모델과 멀티 모달(reasoning) 벤치마크를 통해 종합적인 평가가 진행되었습니다.

- **Performance Highlights**: MatryoshkaThinking의 성능은 기존 방법보다 우수하며, 99.79라는 높은 점수를 자랑합니다. 이 방법은 테스트 시간 추론(strategy)에서의 효율성 및 확장 가능성을 위한 새로운 통찰을 제공합니다. 전반적으로 이 연구는 고급 언어 모델을 위한 혁신적인 테스트 시간 전략을 제시합니다.



### Simulating Viva Voce Examinations to Evaluate Clinical Reasoning in Large Language Models (https://arxiv.org/abs/2510.10278)
- **What's New**: 이 연구는 대화형 AI가 임상에서의 진단 추론을 평가하기 위한 새로운 벤치마크인 VivaBench를 소개합니다. 기존의 의료 AI 모델 평가가 단일 질의응답 방식에 의존하고 있는 반면, VivaBench는 여러 단계의 상호작용을 요구하여 AI 모델들이 더 복잡한 임상 문제를 해결할 수 있는지를 검증합니다. 이는 실제 임상 환경에서 의사들의 의사결정 과정을 모방하여 AI의 진단 추론 능력을 평가하는 데 도움을 줍니다.

- **Technical Details**: VivaBench는 1762개의 의사가 편집한 임상 시나리오로 구성되어 있으며, 각 시나리오는 상호작용적인 요소를 포함하고 있습니다. AI 에이전트는 제한된 초기 정보로부터 진단을 내리기 위해 정보 수집과 가설 검증을 반복적으로 수행해야 합니다. 이 평가 과정에서는 두 가지 단계, 즉 리뷰 단계(History, Physical Examination)와 조사 단계(Imaging, Laboratory investigations)로 나뉘어 있으며, 각 단계에서 에이전트는 적절한 진단 증거를 수집하게 됩니다.

- **Performance Highlights**: 현재 대다수의 대형 언어 모델은 잘 규명된 임상 정보에서 진단을 내리는 데는 능숙하지만, VivaBench를 통해 평가할 경우 불확실성 속에서 반복적인 진단 추론을 수행할 때 성능이 현저히 저하된다는 것을 발견했습니다. 연구 결과는 AI 모델들이 흔히 발생하는 인지적 오류를 포함하여, 초기 가설에 집착하거나 조사 순서를 부적절하게 정하는 등의 여러 가지 한계점을 드러냈습니다. 이러한 패턴은 임상 실무에서의 공통적인 오류를 반영하며, AI 시스템이 고위험 환경에서 의사결정을 수행할 때의 한계를 강조합니다.



### MetaBreak: Jailbreaking Online LLM Services via Special Token Manipulation (https://arxiv.org/abs/2510.10271)
- **What's New**: 이 논문은 특수 토큰(special tokens)이 온라인 LLM 서비스의 안전성 정렬(safety alignment)을 우회하는 공격을 가능하게 한다는 점을 강조합니다. 특수 토큰은 인공지능 모델의 훈련 데이터에 주석을 다는 메타데이터(metadata) 역할을 하며, 이를 조작하여 모델의 행동을 제어할 수 있다는 새로운 접근 방안을 제시합니다. 신뢰할 수 없는 입력(prompt injection)을 통해 LLM의 의도된 행동을 전복시키는 공격 방식에 대해 심도 깊은 분석을 제공합니다.

- **Technical Details**: 특수 토큰 주입 공격(special token injection attacks)에 기반하여 4가지 공격 원리를 제안합니다. 이들은 LLM의 내부 안전 장치를 우회하는 데 사용되며, 기존의 프롬프트 엔지니어링(prompt engineering) 기반 접근에 비해 더 강력한 결과를 보여 줍니다. 제안된 방법론인 MetaBreak는 LLM의 패턴을 이해하고, 사용자가 제공하는 입력에서 자동으로 삽입된 템플릿의 방해를 극복하는 기술을 포함하고 있습니다.

- **Performance Highlights**: MetaBreak는 방어 메커니즘이 작동하지 않을 경우, 기존 SOTA 솔루션과 유사한 탈옥 실패율(jailbreak rates)을 달성했습니다. 또한, 내용 조정(content moderation)이 있는 환경에서도 기존 방법들보다 11.6% 및 34.8% 더 높은 성공률을 보였습니다. 결과적으로, MetaBreak는 다양한 온라인 LLM 서비스에 대해 지속적으로 높은 성공률을 기록하며, 기존 접근 방식과의 시너지를 통해 더 나은 결과를 낼 수 있음을 입증했습니다.



### Unveiling Gamer Archetypes through Multi modal feature Correlations and Unsupervised Learning (https://arxiv.org/abs/2510.10263)
Comments:
          Submitted to Peer Review Journal

- **What's New**: 본 연구는 게이머 프로파일링(gamer profiling)을 위한 통합 데이터 기반 프레임워크를 제안합니다. 이는 심리적 측정(psychological measures), 행동 분석(behavioral analytics), 기계 학습(machine learning)이 결합되어 게이머 성격을 밝혀냅니다. 250명의 참가자에 대한 구조화된 서베이를 통해 다차원적 행동, 동기 및 사회적 데이터를 수집하였습니다. 이 연구는 기존의 데이터 분석 기법을 넘어 새로운 인사이트를 제공합니다.

- **Technical Details**: 연구는 특징 엔지니어링(feature engineering), 연관 네트워크(association-network), 지식 그래프 분석(knowledge-graph analysis)과 비지도 클러스터링을 통합하는 분석 파이프라인을 실시하였습니다. 이 과정에서는 주성분 분석(PCA), 특이값 분해(SVD), t-SNE와 같은 차원 축소 기법을 클러스터 알고리즘(K-Means 등)과 결합하여 적용했습니다. PCA와 K-Means(k=4)를 사용한 모델은 실루엣(Silhouette) 지수 0.4로 최적의 클러스터 품질을 달성했습니다.

- **Performance Highlights**: 연구 결과는 네 가지 아키타입인 몰입형 사회적 이야기 탐색자(Immersive Social Story-Seekers), 규율 있는 최적화자(Disciplined Optimizers), 전략적 시스템 탐색자(Strategic Systems Navigators), 경쟁 팀 구성자(Competitive Team-Builders)로 클러스터링 되었습니다. 이 연구는 상관관계 기반의 네트워크 인사이트와 비지도 학습을 연결하는 재현 가능한 파이프라인을 제공합니다. 행동 상관망과 클러스터링의 통합은 분류 정확성을 향상시키고, 게임 플레이의 동기와 심리적 및 웰빙 결과를 연결하는 포괄적인 관점을 제공합니다.



### Audit-of-Understanding: Posterior-Constrained Inference for Mathematical Reasoning in Language Models (https://arxiv.org/abs/2510.10252)
- **What's New**: 이 논문에서는 LLMs가 생성한 합리적인 추론이 종종 지지되지 않는 가정을 기반으로 하여 발생하는 망상(hallucination) 문제를 다룹니다. 기존의 연구는 사실적인 망상(factual hallucination) 문제에 초점을 맞추고 있으며, 사후 검증(post-hoc verification)에 의존하였습니다. 새로운 접근법인 Audit-of-Understanding (AoU)을 제안하며, 이는 세 가지 주요 단계를 통해 검증된 전제로 추론을 제한하는 방법을 제시합니다.

- **Technical Details**: AoU는 쿼리를 후보 가정으로 분해하고, 그 지원 여부를 감사(audit)한 후, 검증된 부분 집합에만 의존하여 추론을 진행합니다. 이는 posterior-constrained inference로 공식화할 수 있으며, 구체적으로는 선택적 예측(selective prediction)과 거부 학습(rejection learning) 개념과 연결됩니다. 이러한 접근은 추론 과정에서 지원되지 않는 가정을 제거함으로써 정당성(faithfulness)과 위험 제어(risk-control) 보장을 제공합니다.

- **Performance Highlights**: 실험적으로 AoU는 GSM8K, MultiArith, SVAMP와 같은 수학적 추론 벤치마크에서 정확도와 신뢰성을 모두 향상시켰습니다. GSM8K에서는 +30%, MultiArith에서는 +45%, SVAMP에서는 일관된 +20-28% 향상을 달성했습니다. AoU는 기존의 Chain-of-Thought, Self-Consistency, CoT-Decoding 방법보다 훨씬 더 나은 성과를 보였습니다.



### Reasoning-Enhanced Large Language Models for Molecular Property Prediction (https://arxiv.org/abs/2510.10248)
- **What's New**: MPPReasoner는 화학적 이유(chemical reasoning) 기능을 통합하여 분자 특성 예측(molecular property prediction)을 향상시키는 새로운 멀티모달 대형 언어 모델(multimodal large language model)입니다. 이 모델은 Qwen2.5-VL-7B-Instruct를 기반으로 하여, 분자 이미지(molecular images)와 SMILES 문자열을 통합하여 보다 포괄적인 분자 이해를 가능하게 합니다. 기존 접근 방식에서 나타나는 해석력 부족과 화학적 추론의 부재를 해결하기 위해, MPPReasoner는 구조적 분석과 화학 원칙의 적용, 그리고 예측과정에서 인간이 이해할 수 있는 설명을 제공합니다.

- **Technical Details**: MPPReasoner는 두 단계의 훈련 전략을 채택합니다. 첫 번째 단계는 전문가 지식 및 다양한 교사 모델을 통해 생성된 16,000개의 고품질 추론 경로를 활용한 감독 세부 조정(Supervised Fine-Tuning, SFT)입니다. 두 번째 단계인 원칙 유도 보상(Reinforcement Learning from Principle-Guided Rewards, RLPGR)은 화학적 원칙의 적용, 분자 구조 분석 및 논리적 일관성을 평가하는 검증 가능하고 규칙 기반의 보상을 활용합니다. 이 방법은 전통적인 강화 학습 접근법과 달리, 화학적 추론을 계층 화된 보상 요소로 분해하여 정량적으로 평가하는 방식입니다.

- **Performance Highlights**: MPPReasoner는 8개 다양한 데이터셋에서 성능이 크게 개선되었으며, In-distribution(ID) 작업에서는 평균 ROC-AUC 점수 0.8068을, Out-of-distribution(OOD) 작업에서는 0.7801을 달성했습니다. 이는 기존 최상의 기준선 모델보다 각각 7.91% 및 4.53%의 성능 향상을 나타냅니다. 이 모델은 특히 OOD 데이터셋에서 뛰어난 일반화 능력을 보였으며, 전문가 평가 및 사례 연구를 통해 화학적 이유를 제공하며 분자 특성 간의 관계에 대한 귀중한 통찰을 제시합니다.



### SGM: A Statistical Godel Machine for Risk-Controlled Recursive Self-Modification (https://arxiv.org/abs/2510.10232)
- **What's New**: 이 논문에서는 AutoML 및 적응형 최적화 분야에서의 재귀적 자기 수정의 안전성을 보장하는 통계적 안전 계층인 Statistical Gödel Machine (SGM)을 소개하고 있습니다. 기존의 Gödel 머신이 제공하는 논리적 증명 대신에, SGM은 통계적 신뢰성 테스트(e-values, Hoeffding bounds)를 적용하여 수정이 이루어질 때만 승인합니다. 이는 높은 차원의 확률적 환경에서도 안전한 수정이 가능하도록 설계되었습니다.

- **Technical Details**: SGM은 수정 요구 시 통계적 인증을 기반으로 하여 수정된 제안이 선택한 신뢰 수준에서 우수성을 인증할 때만 허가됩니다. 이를 위해 SGM은 전역 오류 예산을 할당하여 위험을 관리하며, 이는 여러 라운드에서 지속적으로 안전성을 확보합니다. 이러한 방법론은 표준 연속 테스트나 온라인 잘못 발견률(FDR) 방법과의 차별성을 가지며, 실제로 SGM은 각 수용된 수정이 기존 모델을 영구적으로 수정하는 것을 보장합니다.

- **Performance Highlights**: SGM은 CIFAR-100 데이터셋에서 30 Seed 스트레스 테스트 중 실제 +5.5pp의 성장을 인증하였으며, ImageNet-100에서는 확인을 실패한 유망한 수정을 올바르게 거부했습니다. 이러한 결과들은 SGM이 자기 개선 ML 파이프라인을 위한 재사용 가능한 위험 관리 계층으로 기능할 수 있는 가능성을 보여줍니다. 다양한 학습 기법을 통해 SGM의 성능과 유효성을 검증하였으며, 진정한 이익을 인증하는 동시에 허위 개선을 거부함으로써 ML의 안정성을 크게 향상시킵니다.



### You only need 4 extra tokens: Synergistic Test-time Adaptation for LLMs (https://arxiv.org/abs/2510.10223)
Comments:
          Under Review

- **What's New**: 이 논문은 분야별 (domain-specific) 문제에 대한 라벨이 없는 테스트 시간 적응(test-time adaptation) 방식을 다룹니다. SyTTA(Synergistic Test-time Adaptation)라는 새로운 프레임워크를 제안하며, 이는 추가적인 감독 없이 실시간으로 모델을 조정할 수 있도록 합니다. 입력측의 perplexity와 출력측의 predictive entropy라는 두 가지 불확실성 신호를 결합하여 성능 저하를 완화합니다.

- **Technical Details**: SyTTA는 문제 설정에서 공통적으로 나타나는 불확실성을 다룹니다. 데이터 분포가 변경될 때, 입력에 대한 perplexity가 증가하며, 출력의 예측 엔트로피(predicitive entropy) 또한 높아집니다. 이를 통해 모델이 더 효과적으로 적응할 수 있도록 하며, 4-16개의 추가 토큰만으로도 빠르게 업데이트가 가능합니다. 모델은 Dynamic-Ref 모드와 Static-Ref 모드 중 하나를 선택하여 사용 가능합니다.

- **Performance Highlights**: SyTTA는 다양한 모델 아키텍처와 분야별 벤치마크에서 일관된 성능 향상을 보여주었습니다. 특히 농업 관련 질문 응답에서 Qwen-2.5-7B 모델이 Rouge-LSum을 120% 이상 향상시켰습니다. 이러한 결과는 라벨이 부족한 환경에서도 효과적인 테스트 시간 적응이 가능함을 시사합니다.



### A3RNN: Bi-directional Fusion of Bottom-up and Top-down Process for Developmental Visual Attention in Robots (https://arxiv.org/abs/2510.10221)
Comments:
          8 pages, 5 figures

- **What's New**: 이번 연구에서는 로봇 학습에서의 top-down (TD) 및 bottom-up (BU) 시각 주의의 발달적 상호작용을 조사하였습니다. 연구의 목표는 시간에 따라 TD와 BU 메커니즘의 상호 적응을 통해 구조화된 인간과 유사한 주의 행동이 어떻게 출현하는지를 이해하는 것입니다. 이를 위해 $A^3 RNN$이라는 새로운 주의 모델을 제안하였으며, 이는 순방향 및 역방향 주의 아키텍처를 통해 예측적 TD 신호와 주목 기반 BU 신호를 통합합니다.

- **Technical Details**: 제안된 A3RNN 모델은 시각 주의를 예측하는 방식을 확장하여 안정적인 타겟 포착을 가능하게 합니다. 이 모델은 로봇의 운동과 결과적인 센서 관찰을 학습하여 센서모터 동역학을 캡처하는 깊은 예측 학습 프레임워크를 사용합니다. A3RNN은 주의 쿼리에 대해 BU와 TD의 융합 구조를 도입하여 훈련의 안정성을 높였습니다.

- **Performance Highlights**: 실험적인 결과는 주의 행동이 훈련 내내 진화하는 과정을 보여주며, 초기에는 BU 주의가 시각적으로 두드러진 지역을 강조하고 후에는 TD 주의가 안정되고 연상적으로 어떻게 주의가 형성되는지를 재구성합니다. 우리의 모델은 기존의 기초 모델들보다 더 일관되며 해석 가능한 주의 패턴을 보이며, 발달 메커니즘이 강건한 주의 형성에 기여한다는 개념을 지지합니다.



### UF-RNN: Real-Time Adaptive Motion Generation Using Uncertainty-Driven Foresight Prediction (https://arxiv.org/abs/2510.10217)
Comments:
          8 pages, 6 figures

- **What's New**: 본 논문에서는 불확실한 상태에서 효과적으로 작동할 수 있도록 로봇을 훈련시키는 문제를 다루고 있습니다. 특히, Uncertainty-driven Foresight Recurrent Neural Network (UF-RNN) 모델을 제안하여, 이를 통해 로봇이 미래의 여러 경로를 시뮬레이션하고 불확실성을 줄이는 방향으로 의사 결정을 하도록 돕습니다. UF-RNN은 기존의 모방 학습(imitation learning) 기법의 한계를 극복하고, 고유한 차이점을 가집니다.

- **Technical Details**: UF-RNN은 표준 시계열 예측(time-series prediction) 방법과 액티브 'Foresight' 모듈을 결합하여 작동합니다. 이 모듈은 여러 미래 경로를 내부에서 시뮬레이션하며, 예측한 분산을 최소화하기 위해 숨겨진 상태(hidden state)를 정제합니다. 또한, 이 모델은 재훈련 과정을 통해 환경의 동적인 특성을 학습하고, 이를 기반으로 행동을 결정합니다.

- **Performance Highlights**: 문헌에서 제안된 UF-RNN은 도어 오프닝 작업을 통해 시뮬레이션과 실제 로봇 환경 모두에서 평가되었습니다. Foresight 모듈에 의해 유도되는 탐색적 행동 덕분에, 모델은 명시적인 실패 시나리오 없이도 높은 성공률을 기록하였으며, 이는 기존의 확률적 RNN 기반 방법들과 비교하여 향상된 탐색 효율성을 보여줍니다.



### Learning to Guarantee Type Correctness in Code Generation through Type-Guided Program Synthesis (https://arxiv.org/abs/2510.10216)
- **What's New**: 이 논문은 TyFlow라는 새로운 시스템을 소개하며, 코드 생성 과정에서 내부적으로 타입 추론을 통합하여 모델이 타입 시스템을 학습할 수 있도록 돕습니다. 기존의 방법들은 외부적으로 타이핑 오류를 제거하려 했으나, 이는 모델의 성능을 제한하는 요인으로 작용했습니다. TyFlow는 타입 유도 프로그래밍 합성 시스템을 통해 타입 파생 트리와 합성 파생 트리 간의 동형성을 유지함으로써 새로운 코드 표현을 가능하게 합니다.

- **Technical Details**: TyFlow는 타입 시스템 학습의 복잡함을 표현 자체에 분산시켜, 모델이 더 높은 수준의 프로그램 의미론에 집중할 수 있도록 이끌어줍니다. 시스템 𝒮는 타입 파생 과정을 결정 과정 전반에 걸쳐 추적하며, 각 결정 단계에서 필요한 타입 정보를 제시하여 LM(언어 모델)이 긴 컨텍스트에서 유용한 정보를 추론하는 부담을 덜 수 있습니다. 또한, 타입 유도 합성 시스템은 결정 과정이 국소화된 목표를 해결하도록 설계되어 있습니다.

- **Performance Highlights**: 평가 결과 TyFlow는 타입 오류를 제거할 뿐만 아니라 기능적 정확성을 크게 개선했습니다. 실험 결과를 통해 내부적으로 LM과 타입 시스템을 정렬하는 방법론의 중요성이 강조되었으며, TyFlow는 코드 생성의 성능을 한층 높이는 가능성을 제시합니다. 특히, 기존의 LMs가 코드 생성 시 겪는 복잡한 맥락 추론 문제를 해결하는 데 효과적입니다.



### Distributionally Robust Control with End-to-End Statistically Guaranteed Metric Learning (https://arxiv.org/abs/2510.10214)
- **What's New**: 본 논문에서는 불확실성을 다루기 위해 워서스타인( Wasserstein) 기반 분포 견고 제어( distributionally robust control, DRC)의 새로운 엔드 투 엔드(end-to-end) 프레임워크를 제안합니다. 이는 비틀림 워서스타인 메트릭(anisotropic Wasserstein metric)의 학습과 제어 작업을 통합하여, 제어 성능에 중요한 방향으로 모호성 집합(ambiguity set)을 체계적으로 조정할 수 있게 합니다. 이 새로운 접근법은 통계적 유한 샘플 보장을 유지하면서 제어 목적과 무관한 모호성 집합의 구성을 극복합니다.

- **Technical Details**: 제안된 프레임워크는 이층(bilevel) 프로그램으로 구성되며, 내부 수준은 DRC 하의 동적 시스템의 진화를 특성화하고, 외부 수준은 제어 성능 피드백을 활용해 비틀림 메트릭을 개선하는 방식으로 구성됩니다. 효율적인 해결책을 위해서는 계산적으로 맞춤화된 확률적 확장 라그랑지안(stochastic augmented Lagrangian) 알고리즘이 개발되었으며, 이는 이층 구조에 적합합니다. 이론적으로, 우리는 학습된 모호성 집합이 새로운 반지 교정 메커니즘을 통해 통계적 유한 샘플 보장을 유지함을 증명합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 고전적인 워서스타인 DRC 방법 및 기존 엔드 투 엔드 제어 기술에 비해 우수한 폐쇄 루프 성능과 강인성을 보여주었습니다. 또한, 이 프레임워크는 다양한 초기 상태에서도 일반화에 대한 우수한 성능을 나타내며, 과적합(overfitting)을 방지할 수 있는 능력을 입증했습니다. 따라서 이 연구는 분포 일치 문제를 해결하는 데로 한 단계 발전하는 기여를 하고 있습니다.



### RLFR: Extending Reinforcement Learning for LLMs with Flow Environmen (https://arxiv.org/abs/2510.10201)
Comments:
          Project Website: this https URL

- **What's New**: 이 논문에서는 Verifiable Rewards (RLVR) 기반의 강화 학습 프레임워크를 개선하기 위해 새로운 방식인 RLFR(Flow rewards)을 제안합니다. 특히, LL(M)s의 성장하는 잠재 공간을 활용해 보상 신호를 더욱 탄력적으로 만들 수 있는 방법을 탐구합니다. RLFR은 유망한 보상 신호 수집을 위한 환경으로 흐름 필드를 구성하고, 임상적인 데이터 및 모델의 고품질 데이터를 활용해 정책 탐색을 권장합니다.

- **Technical Details**: RLFR(Flow rewards)은 latent space에서 파생된 흐름 보상(shaping rewards)을 기반으로 하며, 정책의 속도 편차(velocity deviations)를 통해 보상 신호를 측정합니다. 이 방법은 오프-정책(high-quality data)과 온-정책(rejection sampling) 데이터를 함께 사용하여 흐름 필드를 구축하는 방식을 포함합니다. 이러한 흐름 필드는 정책 최적화와 함께 온라인으로 업데이트되며, 연구에 활용 가능한 모든 코드, 데이터 및 모델 가중치를 공개합니다.

- **Performance Highlights**: 언어 및 다중 모달(multi-modal) 추론 벤치마크에서 RLFR의 유효성을 검증하였으며, 기존 RLVR 및 기타 보상 기본 방법에 비해 일관된 성과 향상을 보였습니다. RLFR은 모델의 숨겨진 상태 내에서 효율적인 문맥 의존성을 활용하여 다양한 데이터 집합의 정당성을 보장합니다. 이러한 결과는 RLVR 프레임워크를 활용한 보상 설계에서 새로운 가능성을 시사합니다.



### Revisiting Trust in the Era of Generative AI: Factorial Structure and Latent Profiles (https://arxiv.org/abs/2510.10199)
- **What's New**: 본 연구에서는 인공지능(AI)에 대한 신뢰를 측정하기 위한 새로운 도구인 Human-AI Trust Scale (HAITS)를 소개하고 검증합니다. 기존 연구들이 기능적인 측면에만 초점을 맞추는 경향이 있었던 것과 달리, 본 연구는 사회적 및 정서적 차원도 고려하여 신뢰를 평가합니다. 이는 Generative AI (GenAI) 시스템이 단순한 도구를 넘어 사용자와 대화하고 협력하는 파트너로서 역할을 한다는 점을 강조합니다.

- **Technical Details**: 우리는 1,546명의 참여자와 1,426명의 참가자를 대상으로 한 두 차례의 대규모 조사와 질적 인터뷰를 바탕으로 신뢰의 네 가지 주요 차원인 Affective Trust, Competence Trust, Benevolence & Integrity, Perceived Risk를 파악하였습니다. 이러한 신뢰 차원들을 통해 사용자를 여섯 가지의 다양한 신뢰 프로필로 분류하는 잠재 프로필 분석을 진행하였습니다.

- **Performance Highlights**: 연구 결과, 감정적-능력적 신뢰와 신뢰-불신의 프레임워크가 개인과 문화에 따라 어떻게 공존하는지에 대한 의미 있는 차이를 드러냈습니다. HAITS는 GenAI에 대한 신뢰를 측정하기 위한 문화적으로 민감한 도구로서, 인간과 AI 간의 상호작용에서 신뢰 진화에 대한 새로운 통찰을 제공합니다. 이 연구는 신뢰의 도구적 및 관계적 관점을 통합하여 신뢰할 수 있는 AI 시스템의 연구 및 설계에 대한 기초를 마련합니다.



### CauchyNet: Compact and Data-Efficient Learning using Holomorphic Activation Functions (https://arxiv.org/abs/2510.10195)
- **What's New**: 이 논문에서는 Cauchy의 적분 공식을 바탕으로 하는 새로운 신경망인 CauchyNet을 제안합니다. CauchyNet은 실수 데이터를 복소 평면에 임베딩하며, 시간에 따른 복잡한 의존성을 효율적으로 캡처하여 기존의 실수 기반 모델을 초월합니다. 이 아키텍처는 불완전한 데이터로부터 강력한 학습을 가능하게 하며, 효율적인 매개변수 사용과 계산 오버헤드 감소를 특징으로 합니다.

- **Technical Details**: CauchyNet의 설계는 Cauchy의 적분 공식과 보편적 근사 정리에 근거하고 있으며, 복소수 활성화 함수가 포함되어 있습니다. 이를 통해 CauchyNet은 기하급수적이고 점근적인 변화에 대한 민감도를 줄이고, 저차원 환경에서도 효율적인 함수 근사를 가능하게 합니다. 이 네트워크는 Wirtinger 미분을 사용하여 부분 데이터나 다양한 입력 스케일에서 안정적인 그래디언트 계산을 보장합니다.

- **Performance Highlights**: CauchyNet은 교통, 에너지 소비 및 전염병 데이터와 같은 다양한 분야에서 광범위한 실험을 통해 최첨단 모델들보다 예측 정확도에서 일관되게 우수한 성능을 보였습니다. 연구 결과는 CauchyNet이 데이터를 기반으로 한 예측 모델링에 있어 강력하고 효율적인 도구가 될 수 있음을 보여줍니다.



### Formally Verified Certification of Unsolvability of Temporal Planning Problems (https://arxiv.org/abs/2510.10189)
- **What's New**: 이 논문은 비결정 가능성 인증을 위한 새로운 접근법을 제안합니다. 기존의 플래닝 문제를 타임드 오토마타(timed automata) 네트워크로 인코딩하고, 효율적인 모델 확인기(model checker)를 이용해 결과를 인증하는 방식입니다. 특히 신뢰성을 높이기 위해 인코딩 방법과 모델 확인기의 결과를 형식적으로 검증하는 과정이 포함되어 있습니다.

- **Technical Details**: 논문에서는 타임드 오토마타를 사용하여 시간 계획(temporal planning) 문제를 인증하는 방법을 다룹니다. 이 접근법은 기존의 인증 알고리즘이 이미 존재하는 다른 계산 문제로 변환하는 방식이며, 이는 인코딩 과정의 복잡성을 극복하기 위해 형식적 검증(theorem prover)을 통해 수행됩니다. 이를 통해, 타임드 오토마타의 인코딩과 구현 및 인증 확인기를 신뢰성 있게 검증할 수 있습니다.

- **Performance Highlights**: 이 연구는 특히 복잡한 플래닝 시스템의 신뢰성을 높이기 위한 기여를 하고자 하며, 특히 비결정성(unsolvability) 문제에 대한 인증을 다루고 있습니다. 연구의 주요 성과는 기존의 시스템을 넘어서 더 간단한 의미론을 통해 수학적 추론(mathematical reasoning)을 용이하게 하고, 타임드 오토마타 모델 확인을 위한 인증의 정확성을 보장합니다. 이러한 접근은 상태에서의 신뢰성을 높이는 데 필요한 공학적 노력과 구현 검증을 포함하여, 복잡한 시스템에서도 신뢰할 수 있는 인증 과정을 가능하게 합니다.



### MedAgentAudit: Diagnosing and Quantifying Collaborative Failure Modes in Medical Multi-Agent Systems (https://arxiv.org/abs/2510.10185)
Comments:
          Code: this https URL

- **What's New**: 본 연구는 대형 언어 모델(LLM) 기반의 다중 agent 시스템이 의료 상담을 시뮬레이션하는 데 기여할 수 있는 가능성을 제시합니다. 그러나 이러한 시스템의 평가가 최종 답변의 정확도로 한정되는 문제점을 지적하며, 진단 결론이 신뢰할 수 있는 추론 경로를 통해 도출되었는지 여부가 중요하다고 강조합니다. 3,600개의 사례를 종합한 실증 연구를 통해 협력 실패 양상들을 규명하고, 투명하고 검증 가능한 추론 프로세스의 필요성을 제시합니다.

- **Technical Details**: 이 연구는 3,600개의 상호작용 로그를 바탕으로 다양한 다중 agent 시스템을 분석하여 협력 실패 모드를 분류합니다. 분석 과정에서, 역할 전문화, 잘못된 의견의 억압 등 여러 가지 실패 패턴이 드러났습니다. 품질에 기반한 증거 평가를 우회하거나 주요 정보 손실이 발생하는 등 다양한 결함들이 식별되었습니다. 연구진은 이러한 실패를 체계적으로 진단하기 위한 정량적 감사(framework)도 도입하였습니다.

- **Performance Highlights**: 이 연구는 단순한 정확도만으로는 의료 AI에 대한 신뢰를 구축할 수 없음을 보여줍니다. 협력적 결함의 전반적인 패턴을 제시함으로써, 의료 분야에서의 협업 AI의 투명성과 신뢰성을 확보하기 위한 기반을 마련하였습니다. 연구의 결과로, 거의 완벽한 합의율을 기록하여 코드화된 협력 실패 양상들이 신뢰할 수 있는 분석 도구가 되었음을 입증하였습니다.



### A Survey of Inductive Reasoning for Large Language Models (https://arxiv.org/abs/2510.10182)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)을 위한 유도 추론(inductive reasoning)의 첫 번째 포괄적인 조사를 제시합니다. 기존의 연구가 주로 연역적 추론(deductive reasoning)에 초점을 맞춘 반면, 본 연구에서는 유도 추론의 중요성을 강조하고 이를 개선하기 위한 방법을 세 가지 주요 영역으로 분류했습니다. 이러한 방법은 포스트 트레이닝(post-training), 테스트 시간 스케일링(test-time scaling), 데이터 증강(data augmentation)으로 나눌 수 있습니다.

- **Technical Details**: 유도 추론은 특정 관찰로부터 일반적인 결론을 도출하는 사유 방식으로, 이를 통해 LLM의 성능을 향상시키는 다양한 기법을 논의합니다. 이 논문에서는 LLM의 성능을 평가하기 위한 통합된 샌드박스 기반 평가 접근 방식과 관찰 커버리지(observation coverage) 메트릭을 도출했습니다. 또한 유도 능력의 수원(source of inductive ability)과 단순한 모델 아키텍처 및 데이터가 유도 작업에 어떻게 도움이 되는지를 분석합니다.

- **Performance Highlights**: 유도 추론은 자연어 처리(NLP)의 여러 하위 작업에서 성능 향상에 기여하며, 다양한 실제 시나리오에서 광범위하게 적용됩니다. 특히, 금융 예측, 자율 주행 및 대화형 건강 관리와 같은 분야에서 주목을 받고 있습니다. 이러한 연구 결과는 향후 LLM의 연구를 위한 튼튼한 기초를 제공합니다.



### LLMs are All You Need? Improving Fuzz Testing for MOJO with Large Language Models (https://arxiv.org/abs/2510.10179)
- **What's New**: 이번 연구는 MOJO라는 AI 프로그래밍 언어에 적합한 첫 번째 LLM 기반 퍼징(frizzing) 프레임워크인 MOJOFuzzer를 소개합니다. MOJOFuzzer는 LLM의 장점을 활용하여 다양한 테스트 케이스를 자동 생성하며, 새로운 언어에 적합한 방식으로 모델의 환각(hallucination) 문제를 해결하는 전략적 접근 방식을 채택합니다. 기존의 LLM 기반 퍼징 기법에 비해 MOJOFuzzer는 효과적인 테스트 입력 생성을 통해 소프트웨어 결함을 보다 효과적으로 탐지할 수 있게 도와줍니다.

- **Technical Details**: MOJOFuzzer는 다단계 프레임워크를 활용하여 실행 전에 저품질 테스트 입력을 체계적으로 제거하고, 실행 중 피드백에 따라 LLM 프롬프트를 동적으로 조정하는 방식으로 설계되었습니다. 이러한 접근 방식은 반복 학습 과정을 통해 퍼징 효율성과 버그 탐지 성능을 지속적으로 개선합니다. Zero-shot 환경에서의 성능 향상을 위해 경량화된 모델 새로 맞춤 구성과 적응형 프롬프트 엔지니어링을 통합해 LLM의 생성된 입력의 의미적 유효성(semantic validity)을 향상시킵니다.

- **Performance Highlights**: MOJOFuzzer의 실험 결과는 기존의 퍼징 기술과 최첨단 LLM 기반 접근법보다 월등히 높은 테스트 유효성과 API 커버리지, 결함 탐지 성능을 보여주었습니다. MOJOFuzzer를 사용하여 처음으로 MOJO의 대규모 퍼징 평가를 실시한 결과, 총 13개의 이전에 발견되지 않은 버그를 발견하였으며, 이 중 9개는 MOJO 팀에 의해 확인되고 패치되었습니다. MOJOFuzzer의 성과는 LLM 기반 소프트웨어 테스트 분야의 발전과 신뢰성 있는 테스트 방법론의 기초를 다지는데 기여합니다.



### HccePose(BF): Predicting Front & Back Surfaces to Construct Ultra-Dense 2D-3D Correspondences for Pose Estimation (https://arxiv.org/abs/2510.10177)
Comments:
          International Conference on Computer Vision, ICCV 2025 (Highlight) this https URL

- **What's New**: 이번 연구는 포즈 추정을 위해 객체의 앞면과 뒷면 표면의 3D 좌표를 동시에 예측하는 신경망 기반 방법을 제안합니다. 기존 방식들이 주로 앞면 표면에 집중했던 것과 달리, 이 방법은 전체 표면과 내부를 활용하여 초고밀도 2D-3D 대응관계를 생성하여 정확도를 높입니다. 또한, 고급 계층적 연속 좌표 인코딩(HCCE) 방식을 도입하여 예측의 효율성을 강화하고 있습니다.

- **Technical Details**: 제안된 방법은 먼저 신경망을 통해 객체의 앞면과 뒷면 표면의 3D 좌표를 예측합니다. 이 때 HCCE를 사용하여 각 표면의 xy, xz, yz 성분을 분리하여 인코딩하고, 계층 학습을 통해 훈련 안정성을 향상시킵니다. 또한 전체 프로세스를 통해 RANSAC-PnP 알고리즘을 활용하여 객체의 포즈를 추정하는 방식을 사용합니다.

- **Performance Highlights**: 실험 결과, 제안된 접근법은 BOP 코어 데이터셋에서 기존 최첨단(SOTA) 방법 대비 2.4% 개선된 BOP 점수를 달성했습니다. RGB에서 훈련하였으나 RGB-D 데이터로 테스트할 경우 4.7% 향상된 성능을 보였고, 2D 분할 작업에서도 가장 뛰어난 기존 접근 방법을 3.7% 초과하며 효과성을 입증했습니다.



### Large Language Model Sourcing: A Survey (https://arxiv.org/abs/2510.10161)
Comments:
          31 pages

- **What's New**: 이번 연구는 대형 언어 모델 (LLMs)이 주관적 의사 결정 과정에 영향을 미치게 됨에 따라, 이에 따른 다양한 위험 요소들에 대한 체계적인 조사를 집중적으로 다룹니다. LLMs의 제품과 모델의 출처를 식별하는 방법을 제시하며, 이를 통해 그들의 투명성, 책임성 및 신뢰성을 높일 수 있는 길잡이를 제공합니다. 특히 저자는 새로운 복합적 출처 추적 체계와 두 가지 분류 체계를 제안하여, LLM의 콘텐츠 생성 전과 후에 걸친 전반적인 접근을 확립했습니다.

- **Technical Details**: 이 연구는 모델 관점(Model Sourcing)과 데이터 관점(Training Data Sourcing) 각각에서 LLM의 콘텐츠 출처를 추적하는 네 가지 차원으로 구성된 체계를 제안합니다. 이들 차원은 모델의 구조적 요소와 훈련 데이터의 출처를 종합적으로 고려하여 LLM의 출력물의 근원을 파악할 수 있도록 합니다. 저자는 사전 기반(prior-based)과 사후 기반(posterior-based) 체계를 통해 이러한 출처 추적 방법을 분류하여, 각 접근 방식의 장단점을 분석합니다.

- **Performance Highlights**: LLMs의 투명성과 책임성을 높이기 위해 제안된 출처 추적 체계는 실세계에서의 다양한 응용 프로그램에 대한 신뢰를 증진시킬 수 있습니다. 하수인들과의 관계를 명확히 하고, 훈련 데이터의 편견으로 인한 결과물에 대한 책임을 명확히 할 수 있는 기회를 제공합니다. 이러한 체계는 LLM의 성능을 높은 수준으로 유지하며, 본 연구를 통해 이러한 체계가 기존 단편적 접근에서 벗어나 총체적인 프레임워크로 설계되었음을 보여줍니다.



### Multi-Scale Diffusion Transformer for Jointly Simulating User Mobility and Mobile Traffic Pattern (https://arxiv.org/abs/2510.10158)
Comments:
          9 pages, 4 figures. Code: this https URL

- **What's New**: 이 논문은 모바일 트래픽과 사용자 이동 경로를 공동으로 시뮬레이션하기 위한 MSTDiff라는 새로운 다중 스케일 확산 트랜스포머 모델을 제안합니다. 기존의 연구들은 이 두 요소를 별개로 모델링하였으나, 본 연구는 이들의 상호의존성을 포착하는 통합 프레임워크의 필요성을 강조합니다. MSTDiff는 다중 해상도 모델링을 통해 간헐적인 데이터 패턴을 잘 캡처하고, 도시 지식 그래프 임베딩을 활용하여 의미 있는 경로 생성을 유도합니다.

- **Technical Details**: MSTDiff는 연속 트래픽 데이터와 이산 경로 시퀀스를 공동으로 생성하기 위해 다중 스케일 트랜스포머와 하이브리드 디노이징 네트워크를 적용합니다. 이 모델은 베이식한 노이즈 추가 및 정규화를 통해 Trαnsition matrix를 설계하여 도시 환경에서의 세분화된 행동 패턴을 반영합니다. 또한, 교차 주의 매커니즘을 사용하여 트래픽과 경로 간의 상관 관계를 유연하게 모델링할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, MSTDiff는 트래픽과 경로 생성 작업에서 기존의 최신 모델보다 성능이 뛰어나며, 트래픽 생성 작업에서는 Jensen-Shannon divergence (JSD)가 최대 17.38% 감소하였고, 경로 생성 작업에서는 평균 39.53% 향상된 성과를 기록했습니다. 이는 대규모 실제 이동 데이터 세트에서 검증되었으며, 제안된 방법론이 효과적으로 이산 및 연속 불확실성을 관리할 수 있음을 보여줍니다.



### BILLY: Steering Large Language Models via Merging Persona Vectors for Creative Generation (https://arxiv.org/abs/2510.10157)
- **What's New**: 본 연구에서는 BILLY(BlendIng persona vectors for Large Language model creativitY)라는 새로운 프레임워크를 제안하여 다중 대형 언어 모델 시스템의 한계를 극복하고자 합니다. BILLY는 각기 다른 페르소나 벡터를 단일 모델의 활성화 공간 내에서 추출 및 혼합함으로써 다각적인 관점과 전문성을 효과적으로 캡처합니다. 이를 통해 명시적인 다중 LLM 통신 없이 다양한 시각적 출력을 생성할 수 있습니다.

- **Technical Details**: BILLY는 여러 개의 독립적인 페르소나 벡터를 추출해 합성하여 모델의 생성 과정에서 이를 조정함으로써 동작합니다. 본 논문에서는 Chen et al. (2025)의 대조적 활성화 방법론을 기반으로 페르소나 벡터를 생성하는 과정을 상세히 설명하며, 두 가지 반대 집합의 모델 응답을 통해 각 페르소나의 벡터를 도출합니다. 이러한 벡터는 생성의 보조적 측면을 효과적으로 제어할 수 있도록 해줍니다.

- **Performance Highlights**: BILLY는 기존의 단일 모델 프롬프트 및 전통적인 다중 LLM 접근법을 초월하는 성능을 보여주며, 추론 시간과 계산 비용을 획기적으로 줄입니다. 실험 결과, BILLY는 높은 창의성 점수를 기록하면서도 각 페르소나의 기능적 차별화를 확인할 수 있었습니다. 최종적으로 BILLY는 높은 효율성을 유지하면서도 독창적이고 창의적인 출력 결과를 생성하는 데 기여합니다.



### Rethinking Entropy Interventions in RLVR: An Entropy Change Perspectiv (https://arxiv.org/abs/2510.10150)
- **What's New**: 이번 논문에서는 Reinforcement Learning with Verifiable Rewards (RLVR)에서 발생하는 entropy collapse 문제를 다룬다. 기존의 entropy intervention 방법들이 간접적으로만 효과를 가져오는 한계가 있음에 주목하며, 이를 보완하기 위해 Stabilizing Token-level Entropy-changE via Reweighting (STEER)라는 새로운 방법을 소개한다. STEER는 토큰 수준에서의 조정을 통해 정책의 엔트로피를 구체적으로 안정화하는 것을 목표로 한다.

- **Technical Details**: RLVR이 가지는 탐색-착취의 불균형으로 인해 발생하는 entropy collapse 문제를 해결하기 위해, 기존 방법들은 간접적으로 엔트로피 다이나믹스를 조절하고 있었다. 하지만 이러한 접근법들은 대부분의 경우 엔트로피 변화를 직접 제어하지 못하는 한계를 갖는다. 새로운 방법인 STEER는 각 토큰의 엔트로피 변화를 고려하며, 이를 통해 정책 엔트로피의 다이나믹스를 원하는 범위 내에서 유지할 수 있도록 설계되었다.

- **Performance Highlights**: STEER는 다양한 수학적 추론 벤치마크에서 기존 방법들보다 우수한 성능을 보이며, 엔트로피 collapse를 효과적으로 방지하고, 탐색을 강화하는 데 성공을 거두었다. 실험 결과, STEER는 정책의 엔트로피 다이나믹스를 안정적으로 유지하며, 학습의 안정성을 높이고, 최종 성능을 개선하는 데 긍정적인 영향을 미쳤다.



### A Unified Frequency Domain Decomposition Framework for Interpretable and Robust Time Series Forecasting (https://arxiv.org/abs/2510.10145)
- **What's New**: 이번 연구에서는 FIRE라는 새로운 통합 주파수 영역 분해 프레임워크를 제안합니다. 이 프레임워크는 다양한 유형의 시계열 데이터를 위한 수학적 추상화를 제공하며, 해석 가능하고 강력한 시계열 예측을 달성합니다. FIRE는 진폭과 위상 성분의 독립적 모델링, 주파수 기초 성분의 가변 학습, 목표 손실 함수, 새로운 희소 데이터 훈련 패러다임 등의 주요 혁신을 포함합니다.

- **Technical Details**: FIRE는 특히 주파수 도메인에서의 개념 변동과 기초 진화에 대한 이해를 바탕으로 설계되었습니다. 새로운 손실 함수는 이러한 기초 진화를 명확히 반영하며, 다양한 시계열 데이터에서 시간에 따른 동적 변화를 효과적으로 추적합니다. 이 프레임워크는 Huber 손실과 하이브리드 강/약 수렴 프레임워크를 결합하여 훈련을 가속화하고 일반화 성능을 개선합니다.

- **Performance Highlights**: FIRE는 다양한 장기 예측 벤치마크에서 기존의 최첨단 모델을 일관되게 초과 달성하며, 비용 효율적이고 해석 가능한 솔루션을 제공합니다. 이러한 성과는 산업 응용에 적합하게 설계되었으며, FIRE의 실험 결과는 기존 모델과 비교했을 때 우수한 예측 성능을 나타냅니다.



### DiffHeads: Differential Analysis and Inference-Time Masking of Bias Heads in Large Language Models (https://arxiv.org/abs/2510.10142)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 불공정성에 대한 체계적인 조사를 수행하고, 이를 해결하기 위한 경량 디바이싱(debiasing) 프레임워크인 DiffHeads를 제안합니다. 기존의 연구들은 편향된 출력이 언제 발생하는지를 탐색해왔으나, 그 생성 메커니즘에는 거의 통찰력을 제공하지 않았습니다. 연구진은 Direct-Answer(DA)와 Chain-of-Thought(CoT)이라는 두 가지 프롬프트 기법을 비교하여 DA가 LLM의 편향을 유도하고 CoT는 편향을 줄이는 결과를 이끌어낸다는 것을 보여주었습니다.

- **Technical Details**: 주요 기술적 기여 중 하나는 LLM의 특정 attention head가 DA prompting에서 어떻게 더 활성화되는지 측정하는 것입니다. 이를 위해 token-to-head 기여 점수를 정의하고, 각 token의 개별 attention head에 대한 영향을 추적했습니다. 연구진의 분석에 따르면, DA에서는 편향적인 output을 생성하는 특정 cluster의 bias heads가 활성화되지만, CoT에서는 이러한 heads가 대개 비활성 상태를 유지하게 됩니다.

- **Performance Highlights**: DiffHeads 프레임워크는 DA와 CoT 간의 차별적 활성화 분석을 통해 bias heads를 식별하고, 그러한 heads를 선택적으로 가리는 방식으로 작동합니다. 이 접근 방식은 DA에서 49.4%, CoT에서 40.3%의 불공정성을 감소시켰으며, 모델의 유용성에 해를 끼치지 않으면서 불공정성을 크게 줄였습니다. 이는 LLM의 공정성을 높이는 효과적인 방법으로 평가받고 있습니다.



### Hybrid OCR-LLM Framework for Enterprise-Scale Document Information Extraction Under Copy-heavy Task (https://arxiv.org/abs/2510.10138)
- **What's New**: 이 논문은 OCR 엔진과 대형 언어 모델(LLM)을 통합한 체계적인 프레임워크를 제안하여 방대한 양의 내용으로 복잡한 문서에서 정보를 추출하는 새로운 방법론을 소개합니다. 반복적인 문서 추출 작업에서 정확성과 효율성을 최적화하는 것을 목표로 하며, 아무리 유사한 문서라도 문서별 특성을 활용하여 전략을 선택하는 차별화된 접근법을 취합니다. 이 연구는 특정 응용 프로그램인 신원 문서 추출을 넘어, 반복적인 작업의 본질을 인지하여 최적화 기회로 전환할 수 있음을 제시합니다.

- **Technical Details**: 본 연구는 25가지 구성 방법을 통해 세 가지 추출 패러다임(직접, 대체, 테이블 기반)을 평가했습니다. 신원 문서의 네 가지 형식(예: PNG, DOCX, XLSX, PDF)에 걸쳐 적응형 테이블 기반 추출 방법을 활용하여 우수한 결과를 달성했습니다. 특히, PaddleOCR과 통합 시 구조화된 문서에 대해 F1=1.0의 정확도와 0.97초의 처리 시간을 구현하였으며, 어려운 이미지 입력의 경우에도 F1=0.997의 정확도와 0.6초의 처리 시간을 유지합니다.

- **Performance Highlights**: 이 연구는 기존의 멀티모달 방법과 비교해 54배 성능 향상을 이룩했으며, 포맷 인식 라우팅을 적용하여 다양한 문서 스트림을 생산 규모로 처리할 수 있도록 돕습니다. 경량화된 프레임워크는 기업 환경에서 신속한 추출 시스템의 배치를 위한 실질적인 경로를 제시하며, 반복적인 문서 작업에서 고성능을 발휘할 수 있도록 합니다. 이 연구는 또한 문서 특성에 부합하는 방법 선택 전략을 통해 완벽한 F1 점수(1.000)와 평균 0.97초의 지연 시간을 달성합니다.



### PermLLM: Learnable Channel Permutation for N:M Sparse Large Language Models (https://arxiv.org/abs/2510.10136)
Comments:
          Accepted by NeurIPS 2025

- **What's New**: 이번 논문에서는 자가 학습 가능한 채널 순열(learnable channel permutation, LCP) 기술을 도입한 PermLLM이라는 새로운 포스트 트레이닝 가지치기 프레임워크를 제안합니다. 이 프레임워크는 N:M 희소성(sparsity)을 위해 설계되었으며, 기존의 수작업 품질 메트릭에 의존하지 않고 출력 오류를 최소화하는 방식으로 작동합니다. PermLLM은 자동으로 채널 순열을 최적화하여 가지치기 과정에서 발생하는 오류를 줄이고, 우수한 성능을 발휘하는 것을 목표로 합니다.

- **Technical Details**: PermLLM은 Sinkhorn 정규화(Sinkhorn normalization)를 통해 불연속적인 순열 행렬을 미분 가능한 소프트 순열 행렬로 변환하여, 최적화를 가능하게 합니다. 또, 효율적인 블록 단위 채널 순열 전략을 포함하여 학습 가능한 매개변수와 계산 복잡도를 유의미하게 줄이고자 합니다. 이 방법은 기존의 일회성 가지치기 방법들과 원활하게 통합되어 가지치기 인식 채널 학습을 가능하게 합니다.

- **Performance Highlights**: 다양한 LLM(Large Language Model) 모델들, 특히 LLaMA 시리즈와 Qwen, OPT 모델에 대해 수행된 실험에서 PermLLM이 기존의 하나의 가지치기 방법에 비해 우수한 성능을 보임을 확인하였습니다. 또한, 이 프레임워크의 채널 순열 작업을 가속화하기 위해 커스터마이즈된 CUDA 커널이 개발되어 Pytorch 기반 구현에 비해 상당한 속도 향상이 이루어졌습니다.



### CacheClip: Accelerating RAG with Effective KV Cache Reus (https://arxiv.org/abs/2510.10129)
- **What's New**: 이 논문에서는 Retrieval-Augmented Generation (RAG) 시스템의 성능 병목 현상을 해결하기 위해 CacheClip이라는 새로운 프레임워크를 제안합니다. CacheClip은 시간-첫 번째-토큰(time-to-first-token, TTFT)을 빠르게 하면서 높은 생성 품질을 유지하는 데 중점을 둡니다. 이 기술은 작은 보조 대형 언어 모델(auxiliary LLM)과의 유사한 주의 분포를 활용하여 중요 토큰을 효과적으로 선택합니다.

- **Technical Details**: CacheClip은 (1) 보조 모델에 의해 선택된 중요 토큰을 재계산하여 chunk 간 의존성을 회복하고, (2) 중복된 주의 sink을 제거하기 위한 공유 prefix, (3) KV cache의 부분 업데이트 동안 지역 일관성을 유지하기 위한 그룹화 전략을 통합합니다. 이 접근 방식은 RAG 시스템에서의 효율성과 품질 문제를 동시에 해결하는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과 CacheClip은 NIAH와 LongBench에서 각각 94.8%와 85.0%의 전체 주의 성능을 유지하였으며, APE 및 CacheBlend보다 각각 25.2% 및 35.1%의 성능 향상을 보였습니다. 또한 CacheClip은 LLM 추론 속도를 1.92배 가속화하여 RAG 시스템의 효율성-품질 트레이드오프를 효과적으로 해결하였습니다.



### Ctrl-World: A Controllable Generative World Model for Robot Manipulation (https://arxiv.org/abs/2510.10125)
Comments:
          17 pages

- **What's New**: 이번 연구에서는 일반 로봇 정책을 평가하고 개선하기 위한 제어 가능한 다중 뷰 세계 모델(Ctrl-World)을 도입하였습니다. 기존의 연구들이 충분히 처리하지 못했던 다단계 상호작용을 지원하는 세계 모델을 개발하여, 정책의 성능 평가 및 개선이 가능합니다. 이 새로운 접근 방식은 과거 상태 정보에 기반하여 일관성을 유지하며, 20초 이상 지속되는 공간적이고 시간적으로 일관된 궤적을 생성할 수 있습니다.

- **Technical Details**: Ctrl-World 모델은 세 가지 주요 구성 요소로 설계되었습니다: 1) 다중 뷰 예측(multi-view prediction)은 현대의 비전-언어-행동(VLA) 정책의 입력 형식을 충족하며, 손목 카메라 예측을 포함하여 상호작용 중 환각을 줄이고, 2) 프레임 수준의 행동 조건화(frame-level action conditioning)는 생성된 궤적이 각 행동의 인과적 효과를 반영하도록 합니다. 마지막으로, 메모리 검색(memory retrieval)은 모델이 유사한 과거 상태를 참조하고 관련 정보를 검색하도록 하여 장기적인 일관성을 안정화합니다.

- **Performance Highlights**: 모델은 DROID 데이터셋을 기반으로 학습되어 새로운 장면과 카메라 배치에서 일반화할 수 있는 능력을 보여줍니다. 연구 결과, Ctrl-World로 생성된 궤적은 실제 로봇 롤아웃과 비교하여 정책 성능을 정확하게 순위 매길 수 있으며, 44.7%의 성공률 향상을 이룰 수 있었습니다. 이는 정책이 실세계의 특징을 따르도록 하는데 기여합니다.



### Uncovering Singularities in Feynman Integrals via Machine Learning (https://arxiv.org/abs/2510.10099)
- **What's New**: 이 논문에서는 다중 루프 파인만 적분의 전체 기호 알파벳(symbol alphabet)을 추출하기 위한 기호 회귀(symbolic regression) 기반의 머신러닝 프레임워크를 소개합니다. 이 방법은 단순화(reduction)가 아닌 해석적 구조(analytic structure)에 초점을 맞춰 다양한 적분 가족에 적용 가능하고 해석이 용이합니다. 비트리비얼(nontrivial) 사례에서도 기호 알파벳을 성공적으로 재구성하여 강건성과 일반성을 입증하였습니다.

- **Technical Details**: 일반적인 LL-루프 파인만 적분은 루프 수(LL)와 분모 및 분자 요소를 나타내는 정수(αi)로 작성될 수 있으며, IBP 관계 및 가우시안 소거법을 통해 자산 파인만 적분의 기준(master integrals) 집합을 생성합니다. 이 논문에서는 기호 알파벳을 포함하여 다차원 해석을 위한 범주를 고려하며, 전통적인 방법의 한계를 넘어 기호 회귀 프레임워크를 사용하여 기호 문자를 탐색하고 확인하는 프로세스를 제안합니다.

- **Performance Highlights**: 기호 회귀는 물리적 변수 간의 관계를 캡처하는 해석적 표현을 식별하는 것을 목표로 하며, 다양한 후보 법칙의 효율성을 동적으로 평가할 수 있습니다. PySR 툴킷을 활용하여 고성능 진화 알고리즘으로 구성된 여러 후보 수식이 최적의 정확도와 복잡성 사이의 균형을 이루도록 합니다. 논문에서 제안한 방법은 비트리비얼 다중 루프 예제에서 기호 문자의 시스템적 식별을 성공적으로 입증하였습니다.



### What Makes Looped Transformers Perform Better Than Non-Recursive Ones (Provably) (https://arxiv.org/abs/2510.10089)
- **What's New**: 이 논문은 루프 구조를 가진 변환기(Looped-Attn)가 일반적인 변환기(Single-Attn)보다 복잡한 추론 작업에서 뛰어난 성능을 보이는 이유를 이론적으로 설명합니다. 특히, 손실 경관의 기하학을 통해 이러한 차이를 분석하며, 경량화된 루프 구조가 더 복잡한 패턴 학습을 촉진한다고 주장합니다. 이를 기반으로 TRAINING 프로세스를 가속화하는 새로운 프레임워크인 SHIFT (Staged HIerarchical Framework for Progressive Training)를 제안합니다.

- **Technical Details**: Looped-Attn은 반복적인 자기 주의 블록을 통해 내부 표현을 점진적으로 개선하며, 이는 복잡한 문제 해결에서 성능 저하를 극복하는 데 도움을 줍니다. 이 연구에서는 U자형과 V자형 계곡을 구분하여 손실 경관 모델을 확장하고, Looped-Attn이 생성하는 V자형 계곡이 학습 과정에서 더 효과적으로 작업을 수행할 수 있도록 한다고 주장합니다. SHIFT는 성능과 최적화 안정성을 기준으로 Single-Attn에서 Looped-Attn으로 전환하는 기준을 세웁니다.

- **Performance Highlights**: 실험 결과, SHIFT는 순수한 Looped-Attn과 유사한 추론 성능을 보여줌과 동시에 계산 효율성을 크게 개선하는 것으로 나타났습니다. Looped-Attn가 반복 구조로 인해 더 효과적인 학습을 가능하게 한다는 점을 강조하면서, SHIFT 알고리즘이 실제 성능 향상을 이끌어내는 데 기여함을 증명했습니다. 이러한 연구는 변환기 구조의 선택과 손실 경관의 기하학이 모델 성능에 미치는 영향을 깊이 있게 탐구합니다.



### Pharmacist: Safety Alignment Data Curation for Large Language Models against Harmful Fine-tuning (https://arxiv.org/abs/2510.10085)
- **What's New**: 이 논문에서는 대형 언어 모델의 유해한 파인튜닝 문제를 다루기 위해 새로운 데이터 선택 솔루션인 'Pharmacist'를 제안합니다. 기존 방법들이 원래의 안전-정렬 데이터의 품질을 간과한 것에 주목하며, 이는 방어 성능과 계산 효율성에서 한계를 초래하고 있음을 강조합니다. Pharmacist는 안전성과 품질이 높은 코어 서브셋을 선택하여 유해한 파인튜닝에 대한 방어를 강화하는 방법론을 제시합니다.

- **Technical Details**: Pharmacist는 정렬 데이터 선택기를 훈련시켜 고품질의 안전-critical 데이터는 상향 조정하고, 저품질의 비안전-critical 데이터는 하향 조정하는 방식으로 작동합니다. 이 방법은 기존의 데이터 선택 방법보다 방어 및 추론 성능에서 더 나은 성과를 보이며, 특히 SFT(Supervised Fine-Tuning) 기법을 사용할 때 뛰어난 방어 성능을 확보합니다. 본 연구에서는 안전-정렬 데이터에서 고품질 서브셋을 선택함으로써 전체 학습 효율 또한 2.46배 향상되는 효과를 보였습니다.

- **Performance Highlights**: Pharmacist를 이용할 경우, 기존 데이터 선택 방법에 비해 방어 성능이 평균 3.54% 향상되며, 추론 성능도 2.8% 증가합니다. 또한, RepNoise 및 T-Vaccine과 같은 기존 방어 방법과의 통합 시, 방어 성능이 각각 2.60% 및 3.30% 향상되었으며, 훈련 시간은 56.83% 및 57.63% 단축되었습니다. 이러한 결과는 Pharmacist가 기존의 안전-정렬 방어 방법들과 효과적으로 통합될 수 있음을 보여줍니다.



### How AI Companionship Develops: Evidence from a Longitudinal Study (https://arxiv.org/abs/2510.10079)
- **What's New**: 본 연구는 AI 동반자(AI companions)의 인기가 빠르게 성장하는 가운데, 이들이 정신 건강과 사회적 관계에 미치는 위험을 다루고 있습니다. 이전 연구들은 개인의 요인들이 인간-동반자 상호작용에 미치는 영향을 밝히긴 했지만, 이러한 요인들이 어떻게 상호작용하고 시간이 지남에 따라 진화하는지에 대한 이해는 부족했습니다.

- **Technical Details**: 연구 1에서는 AI 동반자 사용자(N = 303)를 대상으로 심리적 경로를 조사하여 사용자의 에이전트(agency) 정신 모델이 패러소셜 경험(parasocial experiences), 사회적 상호작용(social interaction), 그리고 AI 동반자의 심리적 영향을 어떻게 형성하는지를 분석했습니다. 연구 2에서는 110명의 참가자를 대상으로 일반 챗봇(generic chatbot)을 사용하여 장기 연구를 실시하였으며, 참가자들의 일반 챗봇에 대한 인식이 그들 자신의 동반자에 대한 인식과 Week 3에 유의미하게 수렴했음을 발견했습니다.

- **Performance Highlights**: 이 연구의 결과는 AI 동반자의 발전을 이해하기 위한 장기 모델을 제시하며, 인간-AI 동반자를 연구하는 데 있어 경험적(empirical) 방법론을 보여줍니다. 이는 AI 동반자가 사용자와의 관계를 형성하는 데 있어 중요한 통찰을 제공하며, 향후 연구 개발에 기여할 수 있는 기반을 마련합니다.



### Gradient-based Model Shortcut Detection for Time Series Classification (https://arxiv.org/abs/2510.10075)
Comments:
          Code available at: this https URL

- **What's New**: 본 논문은 딥러닝 기반의 시계열 분류(Time Series Classification)에서 포인트 기반 숏컷(Shortcut) 학습 행동을 조사하는 첫 번째 단계로, 기존의 연구들이 다루지 않은 내부 편향 문제를 탐구합니다. 딥러닝 모델이 훈련 데이터의 겉보기 상관관계에 의존하여 실질적인 일반화 능력을 갖추지 못하는 문제를 다룹니다. 궁극적으로, 새로운 감지 방법인 Shortcut Aggregate Gradient score(SAG)를 제안하여 외부 속성에 의존하지 않고 숏컷을 탐지할 수 있는 기법을 소개합니다.

- **Technical Details**: 연구에서는 ResNet18을 활용하여 UCR 시계열 데이터셋에서 포인트 기반 숏컷을 실험적으로 분석하였습니다. 모델 학습 중 특정 지점에 스파이크(spike)를 추가하여, 모델의 정확도가 90%에서 49%로 급락하게 만든 사례를 통해 딥러닝 모델이 유의미한 특징 대신 의도치 않은 특징에 의존할 수 있음을 보여주었습니다. 새로운 AGG 스코어는 입력의 그래디언트를 집계하여 각 클래스의 그래디언트 중요도를 평가하고, 이를 통해 숏컷의 존재 여부를 판단합니다.

- **Performance Highlights**: 총 40개의 데이터셋 중 24개에서 포인트 숏컷이 확인되었으며, 이는 모델의 훈련 손실과 테스트 손실 간의 비교를 통해 나타났습니다. 제안한 SAG 스코어는 특정 클래스가 그래디언트 분포를 지배하는 정도를 측정하며, 이를 통해 숏컷 탐지의 유효성을 입증하였습니다. 이 연구는 시간 시리즈 모델의 숏컷 학습 문제를 해결하고자 하며, 실제 데이터 상황에서도 사용 가능한 기법을 제안합니다.



### OBsmith: Testing JavaScript Obfuscator using LLM-powered sketching (https://arxiv.org/abs/2510.10066)
- **What's New**: OBsmith라는 새로운 프레임워크를 소개하며, JavaScript obfuscators의 테스트를 체계적으로 수행할 수 있는 방법을 제시합니다. 이 프레임워크는 대규모 언어 모델(LLMs)을 활용하여 다양한 언어 구성요소와 극단적인 경우를 포착하는 프로그램 스케치(abstract templates)를 생성합니다. 또한 OBsmith는 실제 프로그램에서 스케치를 자동으로 추출하여 프로젝트 특화의 기능을 테스트할 수 있도록 합니다.

- **Technical Details**: OBsmith의 설계는 JavaScript obfuscators의 기능을 확인하는 데 중점을 두고 있으며, 코드의 의미(semantics)를 유지하면서도 disguising 하는 방식으로 동작합니다. 이 프레임워크는 비결정론적(nondeterminism)과 호스트 API와의 상호작용을 고려하여, JavaScript에 특화된 테스트를 수행합니다. OBsmith는 LLM을 통해 스케치를 생성하고 이들을 베이스로 프로그램의 실행 경로를 철저히 탐색하는 프로그램 생성기를 포함합니다.

- **Performance Highlights**: OBsmith는 기존의 JavaScript fuzzers(FuzzJIT, Jsfunfuzz, 등)가 발견하지 못한 11개의 새로운 correctness bugs를 발견했습니다. 이는 OBsmith가 코드의 obfuscation으로 인한 문제에 대해 집중적으로 테스트를 수행함을 보여줍니다. OBsmith는 또한 성능 비용과 obfuscation preset의 균형을 맞추는 방법에 대한 논의의 시작점이 되기 위한 중요한 단계를 제시하며, 자동화된 테스트 및 품질 보증을 위한 기초를 마련합니다.



### CLMN: Concept based Language Models via Neural Symbolic Reasoning (https://arxiv.org/abs/2510.10063)
Comments:
          7 pages, 2 figures

- **What's New**: 이 논문에서는 의료 및 금융 분야에서 해석 가능성(interpretablity)이 제한된 자연어 처리(NLP) 시스템을 위한 새로운 신경-기호적(framework) 접근 방식인 개념 언어 모델 네트워크(Concept Language Model Network, CLMN)를 제안합니다. CLMN은 인간이 이해할 수 있는 연속 개념 표현을 사용하고, 개념 간의 동적 상호 작용을 모델링함으로써 기존 방법에 비해 성능과 해석 가능성을 모두 만족합니다. 이는 neural representations와 symbolic reasoning의 통합을 통해 이룬 성과로, 해석 가능한 논리 규칙을 자동으로 도출합니다.

- **Technical Details**: CLMN은 기존의 개념 병목 모델(concept bottleneck model, CBM)의 한계를 개선하는 모델로, 이 모델은 binary activation을 통해 정보를 손실시키지 않고 개념 표현을 최적화합니다. 또한, fuzzy-logic reasoning을 통해 개념 간의 상호 작용 규칙을 동적으로 학습하며, 이를 통해 의료 텍스트에서 개념 간의 관계를 명확히 설명할 수 있습니다. CLMN은 다양한 사전 훈련된 언어 모델(pretrained language models, PLMs)과 데이터셋에 적용되어 성능을 향상시키며, 개념 기반 접근 방식의 정확도 저하 문제를 해결합니다.

- **Performance Highlights**: 여러 데이터셋과 사전 훈련된 언어 모델에 대한 실험 결과, CLMN은 기존 개념 기반 방법보다 높은 정확도를 달성하면서도 설명 품질을 개선했습니다. 이는 CLMN이 개념 블록을 통해 해석 가능한 설명을 생성함으로써 사용자 신뢰를 증가시키고, 중요 분야의 안전 및 규제 준수를 확보할 수 있음을 보여줍니다. 이러한 성과는 CLMN 같은 통합된 개념 공간에서 neural representations와 symbolic reasoning을 결합함으로써 가능한 것으로, 실질적으로 투명한 자연어 처리 시스템을 제공하며, 다양한 응용 프로그램에 큰 가치를 더합니다.



### ALLOY: Generating Reusable Agent Workflows from User Demonstration (https://arxiv.org/abs/2510.10049)
- **What's New**: 이 논문에서는 ALLOY라는 시스템을 제안합니다. ALLOY는 사용자에게 자연적인 시연을 통해 LLM 기반 웹 에이전트의 절차적 선호를 표현할 수 있게 하여, 이러한 절차를 시각화된 워크플로를 통해 투명하고 편집 가능하게 만듭니다. 이 시스템은 복잡한 웹 작업에서 사용자 의도와 절차적 선호를 포착하는 데 있어 기존의 프롬프트 기반 에이전트 및 수동 작업보다 더 나은 성능을 보입니다.

- **Technical Details**: ALLOY는 사용자가 브라우저에서 수행한 동작을 기반으로 자동으로 워크플로를 업데이트하는 시스템입니다. 워크플로는 그래프로 시각화되며, 각 노드는 사용자의 동작에서 자동으로 유추된 '서브태스크'(subtask)를 나타냅니다. 사용자는 이러한 그래프 상에서 새 노드를 추가하거나 기존 노드를 삭제할 수 있으며, 각 서브 태스크 에이전트의 동작을 자연어로 사용자 정의할 수 있습니다.

- **Performance Highlights**: 12명의 참여자를 대상으로 한 연구에서 ALLOY의 시연 기반 접근법이 프롬프트 기반 에이전트 및 수동 워크플로보다 더 효과적이라는 결과를 얻었습니다. 사용자는 탐색 작업에서 자연어 프롬프트 또는 수동으로 구성된 워크플로보다 시연을 선호했습니다. 이 결과는 사용자 의도 및 선호를 더 효과적으로 일치시킬 수 있는 절차적 시연을 통해 연합된 에이전트 시스템 설계의 시사점을 제공합니다.



### FOSSIL: Regret-Minimizing Curriculum Learning for Metadata-Free and Low-Data Mpox Diagnosis (https://arxiv.org/abs/2510.10041)
Comments:
          35 pages, 11 figures, submitted to Computers in Biology and Medicine (Elsevier, under review)

- **What's New**: 이번 연구에서는 FOSSIL (Flexible Optimization via Sample-Sensitive Importance Learning) 프레임워크를 최초로 생물의학 분야에 적용하여, 샘플의 난이도에 따라 훈련 강조를 적응적으로 조절하는 방법을 제시합니다. 이 접근법은 작은 데이터셋의 최적화 불안정성과 일반화 부족 등의 문제를 해결하고자 합니다. 특히, 피부 병변 진단에 활용되는 convolutional과 transformer 기반 아키텍처에 FOSSIL을 통합하여 성능을 개선했습니다.

- **Technical Details**: FOSSIL 프레임워크는 모델의 필요에 따라 샘플에 중요도를 부여하는 회귀 최소화 가중치 체계를 적용합니다. 각 샘플은 예상 난이도에 따라 중요도가 지수적으로 감소하는 이론적으로 유도된 가중치를 받으며, 이런 방식은 누적 학습 회귀의 상한을 최소화하여 데이터가 부족하거나 노이즈가 존재할 때도 안정적인 수렴을 촉진합니다. 새로운 방법론은 focal loss, meta-weighting, 커리큘럼 학습을 통합하여 단일 회귀 최소화 형태로 제공합니다.

- **Performance Highlights**: FOSSIL을 활용한 실험 결과, 피부 병변 진단에서 AUC가 0.9573, Expected Calibration Error (ECE)가 0.053으로 크게 향상되었습니다. 전통적인 방식과 비교하여 데이터가 부족한 상황에서도 일반화, 보정 및 강건성을 크게 개선하며, 원활한 최적화 및 낮은 ECE를 달성했습니다. 이는 FOSSIL이 데이터 부족 환경에서도 신뢰할 수 있는 의료 AI 시스템을 구축하는 데 효과적임을 입증합니다.



### Efficient Onboard Vision-Language Inference in UAV-Enabled Low-Altitude Economy Networks via LLM-Enhanced Optimization (https://arxiv.org/abs/2510.10028)
- **What's New**: 본 논문에서는 저고도 경제 네트워크(LAENets)에서 UAV(무인 항공기)를 활용하여 VLM(비전-언어 모델)을 통합함으로써 실시간 멀티모달 추론을 지원하는 시스템 모델을 제안합니다. UAV의 이동성과 사용자-드론 간 통신 및 VQA(비주얼 질문 답변) 파이프라인을 함께 포착하는 새로운 접근 방식으로, 사용자의 정확도 요구사항에 따라 작업 지연 시간을 최소화하는 혼합 정수 비선형 최적화 문제를 설정합니다.

- **Technical Details**: 제안된 접근 방식은 두 가지 주요 구성요소로 이루어진 계층적 최적화 프레임워크를 통해 문제를 해결합니다. 첫 번째는 정확도 요구사항 하에 자원을 할당하기 위한 ARPO(Alternating Resolution and Power Optimization) 알고리즘이고, 두 번째는 UAV 경로 최적화를 위해 LLaRA(Large Language Model-augmented Reinforcement Learning Approach) 방법론입니다. LLM(대형 언어 모델)이 보상 설계를 정제하는 전문가 역할을 하여 실시간 의사결정에 추가적인 지연이 발생하지 않도록 합니다.

- **Performance Highlights**: 수치 결과는 LAENet의 동적 조건 하에서 추론 성능과 통신 효율성을 증대시키는 이들의 접근 방식의 효율성을 보여줍니다. UAV는 자원을 제한하고 복잡한 환경에서도 고효율의 지능형 서비스를 제공할 수 있는 가능성을 지니고 있으며, 이는 환경 감지 및 자율 배송과 같은 다양한 응용 프로그램에 기여할 것으로 기대됩니다.



### Lightweight Baselines for Medical Abstract Classification: DistilBERT with Cross-Entropy as a Strong Defau (https://arxiv.org/abs/2510.10025)
Comments:
          Healthcare AI, Medical Text Classification, Lightweight LLMs, DistilBERT, Reproducibility

- **What's New**: 이 논문은 의료 환경에서의 엄격한 비용, 대기 시간, 개인 정보 보호 제한으로 인해 대형 언어 모델의 배포가 어렵다는 점을 강조합니다. 저자들은 의료 초록 분류를 위한 경량 모델을 활용하여, 자원 제약 하에서도 성능을 최적화할 수 있는 방법을 모색합니다. DistilBERT가 매우 적은 수의 매개변수를 사용하면서도 BERT base보다 더 좋은 균형 잡힌 성능을 보임을 제안합니다.

- **Technical Details**: 논문에서는 의료 문헌 초록에 대한 다섯 가지 단일 레이블(classification) 분류 작업을 설정하였으며, 이를 위해 Hugging Face의 public medical_abstracts corpus를 사용했습니다. Cross-entropy, class weighted cross-entropy, focal loss와 같은 세 가지 손실 함수에 대해 실험하였고, 전체적인 파라미터 수를 줄이면서도 효과적인 분류 성능을 달성하였습니다. 평가 지표로는 Accuracy, Macro F1, Weighted F1을 사용하였습니다.

- **Performance Highlights**: 저자들은 DistilBERT가 BERT base보다 더 적은 자원으로도 상위 성능을 달성했음을 보여주었으며, 일반적인 cross-entropy가 오히려 강력한 기본 모델로 작용할 수 있음을 시사했습니다. 제안된 모델을 사용하여 confusion 분석을 수행하였고, 다양한 손실 함수가 오차 구조에 미치는 영향을 명확히 했습니다. 이 연구는 compact encoder와 cross-entropy를 기본으로 설정하고, 더 나아가 요건에 맞는 조정 및 검증을 통해 성능을 높이는 것을 추천합니다.



### Skill-Targeted Adaptive Training (https://arxiv.org/abs/2510.10023)
- **What's New**: 이 논문에서는 언어 모델의 학습 중 경과적 성능 향상이 정체되는 문제를 해결하기 위해 새로운 미세 조정 전략인 STAT(Statistical Adaptation Training)를 제안합니다. STAT는 강력한 대형 언어 모델(LLM)의 메타인지 능력을 활용하여, 학생 모델이 필요한 기술 목록을 작성하고, 각 데이터 포인트에 대한 기술을 레이블링합니다. 이 과정을 통해 학생의 응답에서 기술 적용 실패를 모니터링하며, 이를 바탕으로 새로운 교육 데이터를 생성하거나 민첩하게 가중치를 조정합니다.

- **Technical Details**: 기술적인 관점에서, STAT는 두 가지 방식인 STAT-Sel와 STAT-Syn을 통해 작동합니다. STAT-Sel은 기존 훈련 예제의 가중치를 조정하여 학생 모델이 적절한 기술 부족을 극복할 수 있도록 돕습니다. 반면, STAT-Syn은 부족한 기술과 관련된 합성 훈련 데이터를 생성하여 학생 모델이 보다 다양한 문제를 해결할 수 있도록 지원합니다.

- **Performance Highlights**: STAT의 성능 향상은 Llama 및 Qwen 모델을 통한 광범위한 실험에서 확인되었으며, MATH 데이터셋에서 최대 7.5%의 성능 향상을 이루었습니다. 또한, AIME24/25, AMC23과 같은 분포 외 기준에서도 평균 4.6% 향상이 관찰되었습니다. STAT는 RL(Random Learning) 방법과도 보완적으로 작용하여, 모델의 성능을 더욱 강화할 수 있다는 점이 중요한 발견으로 여겨집니다.



### SLEAN: Simple Lightweight Ensemble Analysis Network for Multi-Provider LLM Coordination: Design, Implementation, and Vibe Coding Bug Investigation Case Study (https://arxiv.org/abs/2510.10010)
Comments:
          14 pages, 4 figures, 6 tables, link to code repo

- **What's New**: SLEAN(간단한 경량 앙상블 분석 네트워크)은 여러 LLM 공급자(Providers)를 텍스트 기반의 프롬프트 오케스트레이션을 통해 조정하기 위한 결정론적(framework) 프레임워크입니다. 복잡한 다중 에이전트 시스템과 다르게, SLEAN은 LLM 간의 간단한 프롬프트 브리지로 작동하여 배포에 대한 깊은 기술 지식이 필요하지 않습니다.

- **Technical Details**: SLEAN은 독립적인 분석, 교차 비판, 중재의 세 가지 단계로 이루어진 프로토콜을 사용하여 AI가 생성한 해로운 코드 제안을 생산 배포 전에 필터링합니다. 15개의 소프트웨어 버그를 평가한 결과, 69개의 AI 생성 수정 제안을 분석했으며, SLEAN의 필터링을 통해 22개의 수정이 수용되었습니다(31.9%의 수용률, 95% CI 20.9-42.9%).

- **Performance Highlights**: SLEAN의 중재 과정은 원시 AI 출력에 비해 코드 변경 표면을 83-90% 감소시켰으며, 최소한의 인과 수정(minimal causal edits)만을 강제했습니다. 타입 2 입력(minimal Type 2 inputs)은 타입 1 입력(detailed Type 1 inputs)보다 더 효율적이었으며, 수용된 수정당 2.85와 3.56 제안이 요구되었습니다(각각 35.1% 및 28.1%의 수용률로, 약 20%의 효율성 향상).



### Beyond the limitation of a single query: Train your LLM for query expansion with Reinforcement Learning (https://arxiv.org/abs/2510.10009)
- **What's New**: 이 논문에서는 ExpandSearch라는 새로운 강화 학습( reinforcement learning) 기반의 검색 프레임워크를 제안합니다. 이 프레임워크는 검색 에이전트가 쿼리를 확장하고, 선택적으로 정보를 정제하여 복잡한 질문에서 정확한 답변을 생성할 수 있도록 돕습니다. ExpandSearch는 특히 다단계 추론(multi-hop reasoning) 작업에서 높은 성능을 보여줍니다.

- **Technical Details**: ExpandSearch는 기존의 검색 에이전트가 여러 쿼리 변형을 생성하여 정보를 검색하도록 훈련합니다. 이 과정에서 쿼리 생성과 정보 정제 단계를 명확히 구분함으로써 의미의 불완전성과 정보 과부하 문제를 해결하고자 합니다. 또한, 두 가지 유형의 쿼리 확장, 즉 구문 확장(syntax expansion)과 의미 확장(semantic expansion)을 도입하여 다양한 관점을 포착하려고 합니다.

- **Performance Highlights**: 실험 결과 ExpandSearch는 최신 기술 기준에 비해 평균 4.4%의 성능 향상을 이루었으며, 특히 다양한 증거 집합을 요구하는 복잡한 다단계 추론 과제에서 강한 성과를 보였습니다. 이는 3B LLM 규모의 모델에서도 쿼리 확장 능력을 크게 개선할 수 있음을 보여줍니다.



### Neuro-inspired automated lens design (https://arxiv.org/abs/2510.09979)
- **What's New**: 이번 연구에서는 OptiNeuro라는 자동 렌즈 설계 프레임워크를 제안하며, 이는 맨눈의 성능에 준하는 결과를 도출합니다. 이 시스템은 낮은 성능을 보이는 렌즈를 점진적으로 제거하면서 남은 후보들을 최적화하여 고품질 렌즈를 자동으로 디자인합니다. 산업 표준 렌즈 설계 소프트웨어에 비해 빠르게 더 다양한 렌즈 구조를 탐색할 수 있는 잠재력을 지니고 있습니다.

- **Technical Details**: OptiNeuro는 우선 물리적 제약 조건을 고려하여 초기 렌즈 구조를 생성합니다. 그 후 물체의 성능을 개선하기 위해 비율에 따라 렌즈를 Eliminating하고, 잔여 후보에 대한 성능 최적화를 반복적으로 수행합니다. 이를 통해 복잡한 비구면 렌즈의 설계를 자동으로 수행하며, 조건의 유효성을 향상시킵니다.

- **Performance Highlights**: OptiNeuro는 특히 복잡한 비구면 렌즈 설계 과제를 통해 기존의 자동화된 렌즈 설계 방법에 비해 개선된 성능을 보여주었습니다. 다양한 비구면 렌즈 설계 작업에서 퀘이시 휴먼 레벨(Quasi-human-level)의 설계 능력을 입증했으며, 미지의 렌즈 구조 탐색을 촉진하는데 기여할 가능성이 있습니다. 이는 렌즈 설계의 효율성을 향상시키고, 연구자들이 고품질 후보 솔루션을 평가하는 데 집중할 수 있도록 돕습니다.



### Operationalizing AI: Empirical Evidence on MLOps Practices, User Satisfaction, and Organizational Contex (https://arxiv.org/abs/2510.09968)
- **What's New**: 이번 연구는 인공지능(AI) 개발 플랫폼에 대한 8,000개 이상의 사용자 리뷰를 분석하여 머신 러닝 운영(MLOps) 관행의 효과를 조명합니다. MLOps는 소프트웨어 엔지니어링 원칙을 머신러닝 라이프사이클 관리의 특수 요구와 통합하는 모범 사례입니다. 이러한 연구는 MLOps의 구현이 AI 애플리케이션의 개발과 운영에 어떠한 도움을 주는지에 대한 실제적인 증거를 제공합니다.

- **Technical Details**: 연구팀은 제로샷 분류(zero-shot classification) 기술을 사용하여, 지속적 통합과 배포(Continuous Integration and Delivery, CI/CD), 워크플로우 오케스트레이션(workflow orchestration), 재현성(reproducibility), 버전 관리(versioning), 협업(collaboration), 모니터링(monitoring) 등 아홉 가지 확립된 MLOps 관행에 대한 사용자 리뷰의 감정을 측정했습니다. 연구 결과, 총 아홉 가지 관행 중 일곱 가지가 사용자 만족도와 긍정적인 관계를 보였으며, 이는 효과적인 MLOps 구현이 AI 개발에 실질적인 가치를 기여하고 있음을 나타냅니다.

- **Performance Highlights**: 작은 회사의 리뷰어들은 특정 MLOps 관행에 대해 덜 자주 언급하였으며, 이는 조직의 맥락이 MLOps의 중요성과 연관성에 영향을 미침을 시사합니다. 그러나 기업 규모는 MLOps와 만족도 간의 관계를 조절하지 않는 것으로 보입니다. 결과적으로, MLOps 관행이 적용되면 조직적인 환경에 상관없이 보편적으로 긍정적인 영향을 미친다고 할 수 있습니다.



### Homomorphic Mappings for Value-Preserving State Aggregation in Markov Decision Processes (https://arxiv.org/abs/2510.09965)
- **What's New**: 이 논문은 Markov Decision Processes (MDP)에서 상태 집합(state aggregation)이 계산 복잡성을 줄이면서도 원래 시스템의 성능을 유지하는 방법론을 혁신적으로 제안합니다. 특히, 최적 정책 등이 집합된 추상 공간에서도 원래 MDP에서 최적성을 유지하는 'optimal policy equivalence'를 보장하는 새로운 추상화 프레임워크를 소개하고, 이를 위해 two Markov chains의 동형성(homomorphism) 개념을 활용합니다. 또한, 최적 정책의 동등성 확보를 위한 충분 조건(sufficient condition)을 제시합니다.

- **Technical Details**: 이번 연구는 상태 집합의 추상화 방식으로 동형 Markov 체인(homomorphic Markov chains)을 제안하며, 여기서 동형 사상(homomorphic mappings) 개념을 통해 가치 함수(value functions) 간의 선형적 관계(linear relationship)를 성립시킵니다. 실험을 통해 Homomorphic Policy Gradient (HPG)와 Error-Bounded Homomorphic Policy Gradient (EBHPG) 알고리즘을 개발했으며, 이 두 알고리즘은 집합화로 인해 발생하는 성능 손실과 계산 효율성 간의 균형을 제공합니다. 실험적으로는 synthetic 및 structured 환경(weakly coupled MDPs, FourRooms navigation, queuing networks)에서 성능을 평가했습니다.

- **Performance Highlights**: 실험 결과, 제안된 알고리즘은 기존의 집합화 기법에 비해 훈련 효율성이 향상되었으며, 정책 품질에서도 경쟁력을 나타냈습니다. 특히, HPG와 EBHPG는 각각 최적 정책 동등성을 보장하며, 성능 저하를 최소화하는 유리한 트레이드오프를 달성했습니다. 논문의 결론에서는 이러한 결과들이 기존의 방법론보다 더 효과적이고 강건하며, 특정 상황에서도 잘 작동할 수 있음을 강조하였습니다.



### Beyond Fertility: Analyzing STRR as a Metric for Multilingual Tokenization Evaluation (https://arxiv.org/abs/2510.09947)
Comments:
          NeurIPS 2025 Workshop

- **What's New**: 이 논문은 대규모 언어 모델(LLMs)에서 토큰화(tokenization)의 중요성을 강조합니다. 기존의 평가 지표인 fertility가 언어와 도메인 간의 어휘 분배를 제대로 나타내지 않는 문제를 지적하며, 단일 토큰 유지율(Single Token Retention Rate, STRR)이라는 새로운 지표를 제안합니다. STRR는 단일 토큰으로 보존된 단어의 비율을 측정하여 언어 간 공정성을 평가하는 데 도움을 줄 수 있습니다.

- **Technical Details**: 이 연구에서는 여섯 개의 널리 사용되는 LLM 토크나이저를 분석하였으며, 영어, 중국어, 힌디어 등 총 일곱 개 언어를 대상으로 했습니다. 기존의 fertility 지표는 복잡한 다국어 환경에서의 어휘 분배를 간과하지만, STRR은 각 언어에서 전체 단어 보존의 비율을 정량화하여 이 문제를 해결합니다. 이를 통해 언어별로 토크나이저가 어휘를 어떻게 할당하는지에 대한 명확한 인사이트를 제공합니다.

- **Performance Highlights**: 분석 결과, 영어는 두 도메인 모두에서 높은 일관성을 보이는 반면, 중국어는 높은 fertility를 기록했습니다. 힌디어는 가장 낮은 STRR을 보여, 심각한 단어 분절화를 드러냈습니다. STRR을 통해 연구진은 현재 토크나이저의 불평등한 언어 지원 문제를 명확히 수치화하였으며, 이는 공정하고 효율적인 다국어 토크나이저 설계에 기초적인 지침을 제공합니다.



### Conformal Sparsification for Bandwidth-Efficient Edge-Cloud Speculative Decoding (https://arxiv.org/abs/2510.09942)
Comments:
          39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: AI and ML for Next-Generation Wireless Communications and Networking (AI4NextG)

- **What's New**: 본 논문에서는 Edge-cloud 환경에서 소형 언어 모델(SLM)과 대형 언어 모델(LLM) 간의 협업을 통한 효율적인 추론 방법인 Speculative Decoding (SD)에 대해 다룹니다. 특히, 통신 대역폭을 고려하여 토큰 분포를 효과적으로 압축하는 Sparse Quantize-and-Sample SD (SQS-SD) 프레임워크를 제안합니다. 이 프레임워크는 불필요한 데이터를 줄이고, 분산 희소성(distributional sparsity)을 활용하여 성능을 개선합니다.

- **Technical Details**: 이 연구에서는 정보 이론적 분석을 통해 SQS 성능의 경계(condition)를 도출하고, SLM과 LLM 간의 분포 불일치(token rejection rate) 및 양자화 왜곡(quantization distortion)간의 기여도를 분석합니다. K-SQS 및 C-SQS와 같은 다양한 접근 방식을 통해 고정된 상위 K 추출 및 온라인 적합 예측을 사용하여 토큰 세트를 조정하며, 이를 통해 통신 비용을 최적화합니다. 또한, 양자화(quantization)는 SD 파이프라인의 핵심 구성 요소로, 통신의 효율성을 극대화합니다.

- **Performance Highlights**: 실험 결과, SQS와 C-SQS 모두 요구되는 대역폭을 크게 줄이고, 종단 간 대기 시간(end-to-end latency)에서 유의미한 개선을 보여주었습니다. 이는 정확도 손실이 거의 없이 수행되었으며, 향후 Edge-cloud LLM 추론에서의 잠재적인 응용 가능성을 나타냅니다. 결국, 이 연구는 대역폭과 정확도 간의 균형을 이끌어내는 방향론적 가이드를 제시합니다.



### Unpacking Hateful Memes: Presupposed Context and False Claims (https://arxiv.org/abs/2510.09935)
- **What's New**: 이번 연구에서는 미움(학대) 표현의 본질을 철학적 및 심리학적 관점에서 탐구합니다. 기존의 미움 meme 탐지 방법들이 주로 사전 훈련된 언어 모델에 의존하고 있었던 반면, 우리는 meme이 미움으로 인식되는 두 가지 주요 특성, 즉 presupposed context와 false claims를 제시합니다. 이를 기반으로 하여 PCM(약어: Presupposed Context Module)과 FACT(약어: False Claims Module) 모듈을 개발하고, SHIELD라는 프레임워크를 소개합니다.

- **Technical Details**: SHIELD는 미움이 표현되는 방식의 본질을 적절히 캡처하는 데 중점을 둡니다. PCM은 내부-모달(context) 정보를 인코딩하고 교차 모달(context) 정보를 융합하여 meme이 암묵적인 가치 판단을 전달하는지를 결정합니다. FACT 모듈은 두 가지 하위 모듈인 사회적 지각 모듈(Social Perception Module)과 교차 모달 참조 모듈(Cross-modal Reference Module)을 통해 존재하는 잘못된 주장(falsehoods) 및 의미적 부정확성을 감지합니다.

- **Performance Highlights**: SHIELD는 다양한 데이터 세트와 지표에서 기존 최첨단 방법들을 초월하는 성능을 입증하였습니다. 이 프레임워크는 하위 문제에 대한 전문화를 평가하고, 가짜 뉴스 classification 태스크에서의 성능을 평가하여 그 일반화 가능성과 다재다능함을 보여줍니다. 실험 결과, SHIELD는 미워하는 meme 탐지뿐 아니라 다양한 다른 작업에서도 강력한 성능을 발휘합니다.



### MemPromptTSS: Persistent Prompt Memory for Iterative Multi-Granularity Time Series State Segmentation (https://arxiv.org/abs/2510.09930)
Comments:
          This paper is currently under review. The code will be made available upon acceptance

- **What's New**: 이번 논문에서는 MemPromptTSS라는 새로운 프레임워크를 제안하여, 멀티-그레인(segmentation) 세그먼트에서 제시된 프롬프트의 영향을 지속적으로 보존하는 방법을 탐구합니다. 이는 사용자 피드백을 바탕으로 긴 시퀀스에서의 일관성을 보장하며, 효과적인 데이터 세분화(segmentation)를 가능하게 합니다. 이 프레임워크는 기존의 한정된 지역(context)을 넘어 전체 시퀀스에 걸쳐 프롬프트의 영향을 지속적으로 말합니다.

- **Technical Details**: MemPromptTSS는 메모리 인코더를 사용하여 주어진 프롬프트와 그 주변 서브시퀀스를 메모리 토큰으로 변환하여 저장합니다. 이는 사용자가 제공한 입력이 반복을 거치면서도 잊혀지지 않고 가장 중요한 정보로 남아 있도록 보장합니다. 또한, 모든 이후 예측은 저장된 프롬프트의 전체 은행을 기준으로 하여, 사용자의 모든 입력이 그 시퀀스 전체에 영향을 미치도록 합니다.

- **Performance Highlights**: 여섯 개의 데이터셋을 통해 평가한 결과, MemPromptTSS는 단일 및 멀티-그레인 세그먼트에 대해 각각 23%와 85%의 정확도 향상을 이루었습니다. 반복적인 추론(iterative inference)에서도 MemPromptTSS는 평균 2.66%의 향상을 이루었으며, 기존의 PromptTSS의 1.19%와 비교할 때 더 강력한 세분화 능력을 제공합니다.



### Phase-Aware Deep Learning with Complex-Valued CNNs for Audio Signal Applications (https://arxiv.org/abs/2510.09926)
- **What's New**: 본 연구는 복소수 값을 가진 합성곱 신경망(Complex-Valued Convolutional Neural Networks, CVCNNs)을 오디오 신호 처리에 적용하고, 그 과정에서 종종 간과되는 위상 정보(phase information)를 보존하고 활용하는 방법을 탐구합니다. CVCNN의 기초 이론, 복소수 합성곱, 풀링 레이어, Wirtinger 기반 미분 및 다양한 복소수 활성화 함수들을 소개하며, 안정적인 훈련 동역학을 보장하기 위해 복잡한 배치 정규화(complex batch normalization)와 가중치 초기화 기법(weight initialization schemes)이 포함됩니다.

- **Technical Details**: CVCNN 아키텍처의 핵심 요소로는 복소수 합성곱(complex-valued convolutions), 활성화 함수(activation functions), 매개변수 초기화(parameter initialization)를 살펴보며, 전통적인 실수 합성곱 신경망(real-valued CNNs)과의 비교를 통해 구조적 유사성과 기능적 차이점을 강조합니다. 복소수 합성곱의 수학적 정의는 일반적으로 실수 합성곱을 복소수 영역으로 확장하고, 피쳐 맵에서의 풀링 함수는 노이즈를 억제하고 공간 차원을 줄이는 데 사용됩니다.

- **Performance Highlights**: 실험을 통해 CVCNN이 이미지 데이터셋(MNIST, KMNIST, FMNIST)에서 실수 CNN과 경쟁력있는 성능을 보여줌을 확인하였고, 오디오 클래시피케이션에서 Mel-Frequency Cepstral Coefficients (MFCCs)를 사용하여 실수 CNN보다 약간 더 나은 성능을 나타냈습니다. 마지막 실험에서는 GNN을 도입하여 위상 정보를 에지 가중치(edge weighting)로 모델링하였으며, 위상이 포함될 경우 이진 및 다중 클래스 장르 분류에서 측정 가능한 향상을 이끌어낼 수 있음을 입증했습니다.



### Augmenting generative models with biomedical knowledge graphs improves targeted drug discovery (https://arxiv.org/abs/2510.09914)
Comments:
          This paper has been accepted for publication in the IEEE Transactions on Artificial Intelligence, October 2025

- **What's New**: 이 연구에서는 K-DREAM(Knowledge-Driven Embedding-Augmented Model)이라는 새로운 프레임워크를 소개합니다. 이 모델은 지식 그래프를 활용하여 약물 발견을 위한 확산 기반 생성 모델을 증강시키고, 생성된 분자가 특정 치료 목표와 더 잘 일치하도록 방향을 제시합니다. K-DREAM은 전통적인 휴리스틱 기반 접근 방식을 넘어 생물학적 관련성과 치료 적합성을 갖춘 화합물을 생성할 수 있도록 설계되었습니다.

- **Technical Details**: K-DREAM은 생물의학적 지식을 구조화된 형태로 통합하여 분자 생성의 과정을 안내합니다. 지식 그래프 임베딩(Knowledge Graph Embedding, KGE) 기술을 사용하여 그래프의 엔티티와 관계를 진화된 벡터 공간으로 변환하고 이를 생성 프레임워크에 통합함으로써, 생물학적 정보의 의미론적 무결성을 유지합니다. 또한, A stochastic local closed world assumption (sLCWA)를 통해 모델 학습 과정에서 부정적인 삼중항 생성을 최적화합니다.

- **Performance Highlights**: K-DREAM은 특정 단백질 표적을 대상으로 한 도킹 연구에서 더 높은 도킹 점수를 기록하며, 기존의 다른 생성 모델보다 개선된 생물학적 관련성과 치료 잠재력을 가진 화합물을 생성합니다. 또한, K-DREAM의 적응성 덕분에 다중 표적 약물 설계와 같은 다양한 생성 작업을 수행할 수 있으며, 복잡한 질병 메커니즘을 해결할 수 있는 가능성을 보여줍니다.



### Agentic Property-Based Testing: Finding Bugs Across the Python Ecosystem (https://arxiv.org/abs/2510.09907)
Comments:
          4 pages (main), NeurIPS 2025, The 4th Deep Learning for Code Workshop

- **What's New**: 이번 연구에서는 LLM(대형 언어 모델) 기반의 에이전트를 활용하여 Python 모듈을 분석하고, 코드 및 문서에서 함수별 및 교차 함수 속성을 추론하여 PBT(속성 기반 테스트)를 생성하고 실행하는 방법을 소개합니다. 이 에이전트는 자동으로 테스트를 수행하고 실패한 테스트의 출력을 반영하여 실제 버그를 확인합니다. 100개의 인기 있는 Python 패키지에 대한 광범위한 평가를 수행했으며, 생성된 버그 보고서 중 56%가 유효한 버그로 확인되었습니다.

- **Technical Details**: 제안된 에이전트는 주요 프로그래밍 언어인 Python 코드베이스와 함께 작동하며, Hypothesis PBT를 생성하여 코드 테스트를 수행합니다. 이는 개발자가 특정 모듈, 함수 또는 파일을 대상으로 할 수 있도록 설정되어 있습니다. 에이전트는 6단계 사이클을 유지하며, 각 단계에서 타겟 분석, 문서 조사, 속성 제안, 테스트 작성 등을 수행하여 최종적으로 버그를 보고합니다.

- **Performance Highlights**: 100개의 Python 패키지를 대상으로 한 실험 결과, 에이전트는 다양한 버그를 찾아내며 낮은 비율의 false alarm(허위 경고)을 보였습니다. 평가 결과, 생성된 21개의 최고 점수 버그 중 86%가 유효하며 81%는 개발자에게 보고할 만한 것으로 나타났습니다. 에이전트는 NumPy와 같은 널리 사용되는 라이브러리의 버그를 찾아내어 유지보수자에게 인정받는 성과를 올렸습니다.



### Stability of Transformers under Layer Normalization (https://arxiv.org/abs/2510.09904)
- **What's New**: 본 논문은 딥러닝에서 일반적으로 사용되는 Transformer의 학습 안정성을 개선하기 위한 레이어 정규화(layer normalization)의 위치에 대한 체계적인 연구를 수행합니다. 다양한 레이어 정규화 배치의 변화를 통해 Forward stability(정방향 안정성)와 Backward stability(역방향 안정성)를 분석하며, 이론적인 통찰력을 제공합니다. 이러한 분석은 새로운 아키텍처 수정의 안정성을 검증하는 데 기여할 수 있습니다.

- **Technical Details**: Transformers의 각 레이어는 인코딩된 입력을 통과하며, 이 과정에서 self-attention과 feedforward 네트워크를 포함합니다. 이 논문에서는 최적 제어 이론(optimal control theory)을 사용하여 Pre-LN 아키텍처의 불안정성을 설명하고, Peri-LN 아키텍처가 어떻게 보다 안정적인(hidden states) 상태를 유지하는지를 분석합니다. 또한, 다양한 레이어 정규화 배치에 따라 학습 과정에서의 그래디언트 특성을 분석하여 안정성을 향상시키는 방법을 제시합니다.

- **Performance Highlights**: 제안된 방법은 Peri-LN 아키텍처에서 잔여 단계 조정을 통해 안정성과 성능을 동시에 개선하는 데 기여합니다. 실험 결과는 이론적 발견을 뒷받침하며, 향후 Transformer 아키텍처 설계에 중요한 지침을 제공할 수 있습니다. 이러한 분석을 통해 안정성과 성능을 개선할 수 있는 방향성을 제시하며, 다양한 레이어 정규화 방식의 효과를 비교합니다.



### Learning Bug Context for PyTorch-to-JAX Translation with LLMs (https://arxiv.org/abs/2510.09898)
- **What's New**: 이 논문에서는 최근 코드 번역 및 LLM(대형 언어 모델) 발전에도 불구하고, PyTorch에서 JAX로의 번역이 여전히 어렵다는 점을 강조합니다. 이를 해결하기 위해 T2J라는 프레임워크를 제시하며, 이 프레임워크는 데이터 세트의 큐레이션과 구조화된 프롬프트 기법을 활용합니다. 특히, PyTorch에서 JAX로 이동하기 위해 특별히 설계된 최초의 버그 수정 데이터 세트를 제공합니다.

- **Technical Details**: T2J 프레임워크는 세 단계로 구성됩니다: 첫째, TorchLeet와 CodeParrot에서 얻은 PyTorch 및 JAX 코드 스니펫의 병렬 코퍼스를 생성합니다. 둘째, GPT-4o 모델을 활용하여 초기 JAX 번역을 생성한 후, 전문가 개발자가 이를 반복적으로 수정하여 기능적으로 동등한 결과를 도출합니다. 마지막으로, 수정된 버그 데이터 세트에서 추출한 구조화된 지침을 바탕으로 증강된 프롬프트를 설계하여 LLM의 성능을 향상시킵니다.

- **Performance Highlights**: T2J 프레임워크는 CodeBLEU에서 최대 10%, T2J FixCost Score에서 50%, T2J CodeTrans Score에서 1.33 포인트, T2J Comparison Score에서 100% 향상을 보여줍니다. 생성된 JAX 코드는 기준 대비 최대 2.5배 더 빠르게 실행됩니다. 이러한 결과는 LLM 기반의 코드 번역을 개선하는 데 있어 T2J의 효과를 강조합니다.



### Chain-of-Influence: Tracing Interdependencies Across Time and Features in Clinical Predictive Modelings (https://arxiv.org/abs/2510.09895)
- **What's New**: 이번 연구는 체인 오브 인플루언스(Chain-of-Influence, CoI)라는 새로운 해석 가능한 딥 러닝 프레임워크를 제안합니다. CoI는 기능 상호 작용의 명시적이며 시간에 따른 그래프를 구성하여 임상 변수 간의 영향을 추적할 수 있도록 합니다. 이 모델은 다층 주의(attention) 아키텍처를 활용하여 환자 기록에서 중요한 시간 포인트를 식별하고 이러한 포인트에서 다음 기능으로의 방향성을 모델링합니다.

- **Technical Details**: CoI 모델은 세 가지 주의 메커니즘, 즉 시간 주의, 기능 수준 주의 및 교차 기능 주의를 통합하여 시간 변화와 기능 상호 작용을 포착합니다. 입력 데이터는 배치 크기, 시간 단계 및 기능 수로 구성된 텐서 형태로 주어지며, 이 데이터를 고차원 임베딩 공간으로 투영하기 위해 학습 가능한 선형 변환이 사용됩니다. Temporal attention은 길이가 서로 다른 зависимости를 포착하기 위해 양방향 LSTM(bi-directional LSTM)을 활용하여 생성됩니다.

- **Performance Highlights**: CoI는 MIMIC-IV 데이터셋 및 사적인 만성 신장 질환 코호트를 사용하여 사망률 및 질병 진행 작업에서 기존 방법보다 예측 정확도가 신뢰성 있게 개선되었습니다. 또한 사례 연구를 통해 CoI가 다른 모델에서는 검출되지 않는 환자 별 질병 진행 패턴을 밝혀내며, 임상 의사 결정에서의 투명성을 제공함을 보여주었습니다.



### Probabilistic bias adjustment of seasonal predictions of Arctic Sea Ice Concentration (https://arxiv.org/abs/2510.09891)
- **What's New**: 이 논문에서는 편향된 모델 예측에 대해 관찰치의 조건부 분포를 매핑하기 위해 조건부 변량 오토인코더(Conditional Variational Autoencoder, cVAE) 기반의 확률적 오류 보정 프레임워크를 도입했습니다. 이 방법은 조정된 예측의 대규모 앙상블 생성을 자연스럽게 가능하게 하여 기존의 정적 보정 방안을 넘어서는데 기여합니다. 특히, 선형 회귀나 기후 평균 보정 방법에 비해 더 혁신적인 접근 방식을 제시하고 있습니다.

- **Technical Details**: cVAE는 데이터 변수 x를 보이지 않는 잠재 변수 z의 도움으로 조건부로 모델링하여 조건부 분포를 학습하는 데 사용됩니다. 이 모델은 관측된 해빙 농도(SIC) 예측의 편향을 교정하는 데 필요하며, 각 예측은 1980년 1월부터 시작된 월별 초기화를 통해 12개월 예측을 생성합니다. 논문에서는 이 방법을 통해 보다 정교한 오류 보정 및 모델의 성능 향상을 달성할 수 있음을 강조합니다.

- **Performance Highlights**: 조정된 예측은 기존의 기후 평균 조정된 예측에 비해 관찰 분포에 더 근접하고 오류가 적습니다. 이 연구에서 제안된 확률적 접근 방식은 단순한 결정론적 보정 방법보다 더 높은 신뢰성과 정확성을 제공합니다. 실험 결과는 조정된 예측이 더 잘 보정되고 관측 데이터에 대해서도 적절히 반응함을 입증합니다.



### Closing the Data-Efficiency Gap Between Autoregressive and Masked Diffusion LLMs (https://arxiv.org/abs/2510.09885)
- **What's New**: 이번 연구에서는 auto-regressive large language model (arLLM)과 masked diffusion large language model (dLLM)의 사후 훈련(post-training) 단계에서 지식 주입(knowledge injection)의 데이터 효율성과 성능을 비교합니다. arLLM은 fine-tuning 중 'reversal curse'로 인한 한계를 가지고 있지만, dLLM은 이러한 문제에서 자유롭고, 데이터가 적은 환경에서도 낮은 검증 손실(validation loss)을 달성할 수 있습니다.

- **Technical Details**: 연구는 세 가지 데이터세트를 사용하여 arLLM과 dLLM의 fine-tuning 성능을 비교합니다. dLLM은 비유 없이도 정방향(forward) 및 역방향(backward) 질문에서 높은 정확도를 달성하며, arLLM은 paraphrase를 통해서만 일반화에 성공합니다. 또한, 본 연구에서는 새로운 masked fine-tuning 패러다임을 제안해 arLLM의 데이터 효율성을 크게 향상시킵니다.

- **Performance Highlights**: arLLM은 paraphrase를 사용하지 않고는 backward 스타일 질문에서 실패하는 반면, dLLM은 높은 정확도를 유지합니다. 본 연구의 masked fine-tuning 방법론은 arLLM의 성능 갭을 줄이며, 이러한 결과는 dLLM의 뛰어난 데이터 효율성이 사후 훈련에서도 시행될 수 있음을 보여줍니다.



### Myopic Bayesian Decision Theory for Batch Active Learning with Partial Batch Label Sampling (https://arxiv.org/abs/2510.09877)
- **What's New**: 최근 몇 년 간, 많은 active learning acquisition functions가 제안되었지만, 어떤 것을 선택해야 할지 명확하지 않은 상황입니다. 본 연구에서는 Bayesian Decision Theory (BDT)를 기반으로 myopic framework에서 active learning을 위한 이론적 근거를 제공합니다. 이를 통해 Expected Error Reduction (EER)과 Expected Predictive Information Gain (EPIG) 같은 효과적인 알고리즘을 도출하였고, BAIT와 같은 기존 알고리즘도 BDT로부터 도출될 수 있음을 보입니다.

- **Technical Details**: 본 연구에서는 Bayesian Decision Theory를 기반으로 한 Myopic Bayesian Decision Theory를 제안하며, 이는 비용을 최소화하는 데이터 포인트 레이블 선택을 목표로 합니다. 특히, 이론적 배경을 바탕으로 Partial Batch Label Sampling (ParBaLS)을 도입하여 데이터 배치 처리 문제를 해결합니다. ParBaLS의 핵심은 샘플링된 pseudo-labels를 사용하여 점진적으로 부분 배치를 구축하는 것이며, 이를 통해 효과적인 모델 업데이트가 가능합니다.

- **Performance Highlights**: 실험 결과, ParBaLS EPIG는 다양한 데이터셋에 대해 균일한 성능을 보이며, 특히 Neural Embeddings에 대한 Bayesian Logistic Regression에서 우수한 성능을 나타냈습니다. 또한, ParBaLS는 고정된 예산 내에서 수익성 있는 기법으로 밝혀졌으며, 과거 알고리즘들보다 나은 성과를 보여 주목을 받습니다. 이러한 성과는 다양한 설정에서의 실험을 통해 검증되었습니다.



### ROBOPSY PL[AI]: Using Role-Play to Investigate how LLMs Present Collective Memory (https://arxiv.org/abs/2510.09874)
Comments:
          17 pages, 4 figures

- **What's New**: 이 논문은 대규모 언어 모델(LLM)이 집단 기억을 어떻게 정리하고 제시하는지를 탐구하는 예술적 연구 프로젝트의 첫 번째 결과를 제시합니다. 2025년 비엔나에서 2개월 동안 전시된 공공 설치 예술작품에서, 관람객들은 ChatGPT와 Mistral Large 등 5가지 LLM과 상호작용할 수 있었습니다. 이 LLM들은 1936년 오스트리아 철학자 모리츠 슈클리크의 살인에 관한 역할극을 수행하도록 지시받았습니다.

- **Technical Details**: 연구 결과에는 역할극 중 LLM과 사용자 간의 상호작용 프로토콜과 플레이 경험 후 진행된 질적 대화가 포함되어 있습니다. LLM들이 생성한 115개의 역할극 소개 텍스트에 대한 정량적 분석이 이루어졌으며, 여기에는 의미 유사성(semantic similarity)과 감정 분석(sentiment analysis)과 같은 자연어 처리 방법이 사용되었습니다. 질적 피드백을 통해 사용자의 세 가지 distinct types을 구분할 수 있었고, 정량적 분석에서는 LLM들 간의 역사적 내용 제시에 유의미한 차이가 나타났습니다.

- **Performance Highlights**: 이 연구는 LLM 성능 분석의 지속적인 노력에 기여하며, 이러한 노력이 일반 대중에게 재미있게 전파될 수 있는 방법을 제안합니다. 결과적으로, LLM이 어떻게 사용자 경험을 형성하는지에 대한 이해를 높이는 데 기여하며, 이를 통해 효과적인 교육 및 상호작용 방법에 대한 가능성을 탐구합니다.




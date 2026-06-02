New uploads on arXiv(cs.CL)

### How Do LLMs Use Their Depth? (https://arxiv.org/abs/2510.18871)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)이 깊이를 균일하게 사용하지 않는다는 성장하는 증거를 통해, 레이어별 예측 동역학(layer-wise prediction dynamics)에 대한 더 세밀한 이해를 제공합니다. 연구진은 여러 개방형 가중치 모델을 추적하며, LLM의 'Guess-then-Refine' 프레임워크를 제안하여 모델이 추론 시 내부적으로 어떻게 계산을 구조화하는지를 설명합니다. 연구 결과는 LLMs가 예측을 위해 깊이를 동적으로 사용하는 방식을 명확히 보여줍니다.

- **Technical Details**: 논문은 LLM의 중간 표현을 추적하고 이를 통해 모델이 예측하는 방식의 구조화된 사용을 밝혔다. TunedLens라는 방법론을 사용하여 중간 레이어 표현을 디코드하고, 이는 LLM이 추정하는 토큰 예측 패턴을 정량화하는 데 도움을 주었습니다. 연구에서는 GPT2-XL, Pythia-6.9B, Llama2-7B, Llama3-8B 등 네 가지 개방형 모델을 사용하였습니다.

- **Performance Highlights**: 결과적으로, 초기 레이어에서의 예측은 고빈도 토큰으로 구성되는 경향이 있으며, 이러한 초기 제안이 이후 레이어에서 실질적으로 수정되었다는 점이 밝혀졌습니다. 연구진은 특정 태스크에 따라 LLM이 깊이를 다르게 사용하는 방식도 관찰하였으며, 더 복잡한 예측은 더 많은 깊이를 요구하는 반면, 쉬운 계산은 보다 빠르게 마무리된다는 사실도 확인하였습니다. 이러한 발견은 Transformer 기반 모델의 계산 효율성을 향상하는 데 기여할 수 있는 통찰을 제공합니다.



### LightMem: Lightweight and Efficient Memory-Augmented Generation (https://arxiv.org/abs/2510.18866)
Comments:
          Work in progress

- **What's New**: 최근의 연구 결과에 따르면, 대형 언어 모델(LLM)의 강력한 능력에도 불구하고 과거 상호작용 정보를 효과적으로 활용하는 데 한계가 있습니다. 이 논문에서는 LightMem이라는 새로운 메모리 시스템을 소개하며, 인간 기억의 Atkinson-Shiffrin 모델에 영감을 받아 구성되었습니다. LightMem은 정보 저장 및 검색 방식을 최적화하여 메모리 시스템의 성능과 효율성을 균형 있게 구현하고 있습니다.

- **Technical Details**: LightMem은 세 가지 상호 보완적인 단계로 메모리를 구성합니다. 첫째, 인지에 영감을 받은 감각 메모리는 관련 없는 정보를 신속하게 필터링하고 주제에 따라 정보를 그룹화합니다. 둘째, 주제 인식 단기 메모리는 이러한 주제 기반 그룹을 통합 및 요약하여 구조화된 접근을 가능하게 합니다. 마지막으로, 수면 시간 업데이트를 가진 장기 메모리는 오프라인 절차를 통해 통합을 온라인 추론과 분리하여 관리합니다.

- **Performance Highlights**: LightMem은 LongMemEval에서 강력한 기준선을 초과하여 정확도에서 최대 10.9%의 개선을 보여주며, 토큰 사용량은 최대 117배, API 호출은 159배, 실행 시간은 12배 이상 줄였습니다. 또한, Case study를 통해 오프라인 수면 시간 통합이 장기적인 지식 업데이트의 신뢰성을 높이고 정보 손실 및 불일치를 완화하는 데 기여함을 보여주었습니다.



### Every Step Evolves: Scaling Reinforcement Learning for Trillion-Scale Thinking Mod (https://arxiv.org/abs/2510.18855)
Comments:
          Technical Report

- **What's New**: Ring-1T는 첫 번째 오픈소스의 최첨단 사고 모델로 1조(1 trillion) 개의 파라미터를 가지고 있습니다. 이 모델은 총 1조 개의 파라미터를 특징으로 하며, 토큰당 약 500억 개를 활성화합니다. 이는 기존 모델과 비교하여 비약적인 발전을 보여줍니다.

- **Technical Details**: 모델 훈련 과정에서 발생하는 여러 가지 도전 과제를 해결하기 위해 세 가지 혁신이 도입되었습니다. 첫 번째, IcePop은 토큰 수준의 불일치를 마스킹 및 클리핑하여 RL 훈련의 불안정을 해소합니다. 두 번째, C3PO++는 동적으로 토큰을 분할해 장기간 롤아웃에서 자원 활용을 개선하고 높은 시간 효율성을 도출합니다. 마지막으로, ASystem은 1조 파라미터 모델 훈련을 방해하는 시스템적 병목현상을 극복하기 위해 설계된 고성능 RL 프레임워크입니다.

- **Performance Highlights**: Ring-1T는 AIME-2025에서 93.4, HMMT-2025에서 86.72, CodeForces에서 2088, ARC-AGI-v1에서 55.94의 우수한 성적을 기록합니다. 특히, IMO-2025에서 은메달 수준의 결과를 달성하여 뛰어난 추론 능력을 강조합니다. 이 모델의 1조 파라미터 MoE 모델을 공개하여 연구자들에게 최첨단의 사고 능력을 직접 접근할 기회를 제공합니다.



### Towards Faithful and Controllable Personalization via Critique-Post-Edit Reinforcement Learning (https://arxiv.org/abs/2510.18849)
Comments:
          work in progress

- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)의 개인화 방법을 개선하기 위해 Critique-Post-Edit이라는 강력한 강화학습 프레임워크를 제안합니다. 기존의 감독학습(Supervised Fine-Tuning, SFT) 및 인간 피드백 기반 강화학습(Reinforcement Learning from Human Feedback, RLHF) 모델이 직면한 한계를 극복하기 위해, 주어진 피드백을 통해 출력 결과를 정제하는 기법을 도입하였습니다. 이 새로운 접근법은 이전 모델에 비해 더 신뢰할 수 있고 컨트롤 가능한 개인화를 가능하게 합니다.

- **Technical Details**: Critique-Post-Edit 프레임워크는 두 가지 주요 구성 요소로 이루어져 있습니다. 첫째, 개인화된 생성 보상 모델(Generative Reward Model, GRM)은 다차원 점수와 텍스트 비평을 제공하여 보상 해킹(reward hacking)을 저지합니다. 둘째, 비평 기반 수정 메커니즘(Critique-Post-Edit mechanism)으로 정책 모델은 자신의 출력을 수정하여 더 목표 지향적이고 효율적인 학습을 가능하게 합니다. 해당 시스템을 이용한 평가에서, 우리의 방법은 기존 PPO 모델을 뛰어넘는 결과를 보여준다는 것을 알 수 있습니다.

- **Performance Highlights**: 개인화된 Qwen2.5-7B 모델은 평균적으로 11% 향상된 승률을 기록하였으며, Qwen2.5-14B 모델은 GPT-4.1의 성능을 초월하는 성과를 보였습니다. 이는 우리의 접근 방식이 신뢰할 수 있는, 효율적이며 컨트롤 가능한 개인화로 이어진다는 것을 보여줍니다. 전반적으로, Critique-Post-Edit 프레임워크는 개인화의 실제적이고 확장 가능한 경로를 제공하여 언어 모델의 개인화 능력을 크게 향상시킵니다.



### MTraining: Distributed Dynamic Sparse Attention for Efficient Ultra-Long Context Training (https://arxiv.org/abs/2510.18830)
- **What's New**: 이 논문은 동적 희소 주의(dynamic sparse attention)을 활용하여 초장기 문맥을 갖는 대형 언어 모델(LLMs)의 효율적인 훈련을 가능하게 하는 새로운 분산 방법론인 MTraining을 소개합니다. MTraining은 동적 희소 훈련 패턴, 균형 잡힌 희소 링 주의(balanced sparse ring attention), 계층적 희소 링 주의(hierarchical sparse ring attention)라는 세 가지 주요 구성 요소를 통합합니다. 이 방법은 모델 훈련 중 발생할 수 있는 계산 불균형과 통신 오버헤드를 해결하도록 설계되었습니다.

- **Technical Details**: MTraining은 분산 환경에서 동적 희소 주의의 선형 스케일링을 가능하게 하는 알고리즘 시스템 공동 설계 프레임워크입니다. 주의 가중치가 Vertical-Slash 지역성 패턴을 갖는다는 관찰을 바탕으로, 훈련 중 동적으로 희소성을 조정할 수 있는 온라인 근사 희소 예산 메커니즘을 도입합니다. 또한, 블록 수준에서 균형 잡힌 희소 링 주의 메커니즘을 채택하여 작업자 및 단계 수준의 균형을 맞추며, 이질적인 배포 네트워크에서 통신 오버헤드를 줄이는 계층적 설계를 적용합니다.

- **Performance Highlights**: MTraining을 사용하여 Qwen2.5-3B 모델의 컨텍스트 윈도를 32K에서 512K 토큰으로 확장하여 평가하였고, NVIDIA A100 GPU 32개 클러스터에서 훈련 속도가 최대 6배 증가하면서도 모델 정확도를 유지하거나 초과하는 성능을 달성했습니다. 다양한 긴 문맥 벤치마크에서도 우수한 평가 결과를 보여주었으며, Llama-3.1-8B-Instruct와 같은 다른 아키텍처에서도 MTraining의 효과를 검증했습니다.



### Fine-Tuned Thoughts: Leveraging Chain-of-Thought Reasoning for Industrial Asset Health Monitoring (https://arxiv.org/abs/2510.18817)
Comments:
          Accepted at EMNLP 2025

- **What's New**: 이 논문은 산업 애셋 헬스(Industrial Asset Health) 분야에 대한 지식 증류(knowledge distillation) 프레임워크를 제안합니다. 이 프레임워크는 대형 언어 모델(Large Language Models, LLMs)에서 소형 언어 모델(Small Language Models, SLMs)로 체인 오브 사고(Chain-of-Thought, CoT) 추론 능력을 전달합니다. 이를 통해 복잡한 추론 과제가 있는 산업 분야에서도 SLM의 성능을 향상시키려는 노력이 이루어집니다.

- **Technical Details**: 제안된 방법론은 전통적인 FMEA(고장 모드 및 영향 분석) 지식을 LLM에서 SLM으로 전이하는 구조를 채택합니다. 이 과정에서는 멀티 초이스 질문(MCQA) 프롬프트를 사용하여, 초기 데이터셋 없이 합성 데이터를 생성하는 단계를 포함합니다. 또한, 지식 그래프(Knowledge Graph)를 활용해 산업 도메인 지식을 조직하고, 각 요소 간의 관계를 정의하는 방법을 제시합니다.

- **Performance Highlights**: 실험 결과, CoT 추론을 적용한 SLM들은 기초 모델에 비해 11%에서 23%까지 성능이 향상되었습니다. 이는 SLM이 LLM과의 격차를 줄이는 데 있어 매우 성공적인 결과임을 보여줍니다. 최종적으로, 이 연구 결과는 산업 자산 건강 모니터링 애플리케이션에서 SLM의 효율성을 강조합니다.



### WebSeer: Training Deeper Search Agents through Reinforcement Learning with Self-Reflection (https://arxiv.org/abs/2510.18798)
- **What's New**: 이번 논문에서는 WebSeer라는 혁신적인 검색 에이전트를 도입하여 자기 반영(self-reflection) 메커니즘을 통해 강화 학습(reinforcement learning)으로 훈련시키는 방법을 제안합니다. 기존의 에이전틱 검색 방식이 직면하고 있던 여러 한계를 극복하기 위해, WebSeer는 도구 사용(deep tool-use) 체인과 정확도(answer accuracy)를 크게 향상시킬 수 있는 두 단계의 훈련 프레임워크를 구성하였습니다. 이러한 접근법을 통해, WebSeer는 더욱 긴 도구 사용 경로를 생성하며, 보다 가치 있는 추론을 수행할 수 있습니다.

- **Technical Details**: WebSeer는 고유한 자기 반영 패러다임을 활용하는 두 단계의 학습 프레임워크를 통해 훈련됩니다. 이 과정에서 Self-Reflective Reinforcement Learning (SRRL) 접근법을 사용하여, 모델이 연속적인 상호작용 중에 정답의 정확도 신호를 기반으로 반영 행동을 보다 효과적으로 유도할 수 있도록 합니다. 또한, 웹 검색 API, 웹 페이지 리더, Python 코드 실행기와 같은 다양한 도구를 통합하여 외부 지식을 획득하고 문제 해결을 지원합니다.

- **Performance Highlights**: WebSeer는 HotpotQA와 SimpleQA에서 각각 72.3% 및 90.0%의 정확도를 기록하며 최신 기술(state-of-the-art) 성과를 달성하였습니다. 이 방법은 OOD(out-of-distribution) 데이터셋에 대한 강력한 일반화 능력을 보여주며, 에이전트가 복잡한 질문 응답 작업을 해결하는 데 큰 잠재력을 가지고 있음을 시사합니다. 이 연구는 사용자에게 더욱 효율적으로 정보를 제공하고, 어려운 문제를 해결하는 데 기여할 것으로 기대됩니다.



### KAT-Coder Technical Repor (https://arxiv.org/abs/2510.18779)
- **What's New**: 최근의 대규모 언어 모델(LLMs)의 발전은 코드 생성에서 에이전틱 코딩(agentic coding)으로의 전환을 가능하게 하였습니다. KAT-Coder는 이 새로운 패러다임을 구현하기 위해 개발된 모델로, 독립적으로 사고하고 계획하며 소프트웨어 개발 프로세스에서 활동하는 능력을 지니고 있습니다. 이 보고서에서는 KAT-Coder의 구조와 기능을 설명하고, 기존 모델들이 갖고 있는 한계를 극복할 수 있는 방법을 제시합니다.

- **Technical Details**: KAT-Coder는 실무 중심의 소프트웨어 개발 데이터와 인공 에이전틱 상호작용으로 이루어진 다단계 훈련(Curriculum Training) 프로세스를 통해 훈련됩니다. 이 과정은 중간 훈련(Mid-Term Training), 감독 미세 조정(Supervised Fine-Tuning, SFT), 강화 미세 조정(Reinforcement Fine-Tuning, RFT), 그리고 강화학습-배포 적응(Reinforcement-to-Deployment Adaptation)이 포함됩니다. 각 단계는 모델의 추론, 계획, 및 반성 능력을 강화하여 실제 통합 개발 환경에서의 신뢰를 구축합니다.

- **Performance Highlights**: KAT-Coder는 도구 사용의 신뢰성, 명령 일치, 장기 맥락 추론 능력을 갖추고 있어 실제 코딩에서 실질적으로 활용될 수 있는 기반을 제공합니다. 특히, 모델은 다양한 프로그래밍 언어와 작업 유형을 포괄하는 백만 샘플 이상의 데이터 세트를 활용하여 보편적인 적응력을 지니게 되며, 단순히 코드 생성에 국한되지 않고 복합적인 소프트웨어 관리 업무를 수행할 수 있습니다. KAT-Dev라는 이름으로 오픈 소스로 제공되며, 실제 생산 환경에서의 테스트를 통해 그 성능을 입증했습니다.



### AI use in American newspapers is widespread, uneven, and rarely disclosed (https://arxiv.org/abs/2510.18774)
- **What's New**: 본 논문은 2025년 여름에 발표된 1,500개의 미국 신문의 온라인 판에서 186,000개의 기사를 분석하여 AI(인공지능)의 사용 현황을 조사합니다. 연구 결과에 따르면, 약 9%의 신문 기사가 부분적으로 또는 완전히 AI에 의해 생성된 것으로 나타났습니다. 이는 특히 소규모 지역 매체와 날씨 및 기술과 같은 특정 주제에서 더욱 두드러지게 나타납니다.

- **Technical Details**: 논문은 Pangram이라는 최신 AI 탐지기를 사용하여 뉴스 기사의 AI 생성 여부를 식별합니다. 45,000개의 오피니언 기사를 추가 분석한 결과, 워싱턴 포스트, 뉴욕 타임스 및 월스트리트 저널의 경우 AI에 의해 생성된 콘텐츠가 6.4배 더 많이 포함되어 있는 것으로 나타났습니다. 특히 많은 AI로 식별된 의견 기사가 저명한 공인에 의해 작성되었음을 밝혀 냈습니다.

- **Performance Highlights**: AI 사용이 광범위하게 퍼져 있지만, 그에 대한 공개는 드물게 이루어지고 있습니다. 손으로 검사한 100개의 AI 신호 기사를 분석한 결과, 단 5건만이 AI 사용에 대한 공개를 포함하고 있었습니다. 이는 저널리즘 내 AI 사용에 대한 투명성과 최신 편집 기준의 필요성을 강조하고 있습니다.



### Topoformer: brain-like topographic organization in Transformer language models through spatial querying and reweighting (https://arxiv.org/abs/2510.18745)
Comments:
          ICLR 2024 Workshop on Representational Alignment (Re-Align) Camera Ready

- **What's New**: 이번 연구에서는 Transformer 모델 안에 생물학적 뇌의 공간적 기능 조직(topographic organization)을 도입하는 새로운 방법을 제안합니다. 연구팀은 Topoformers라는 새로운 형태의 self-attention을 설계하여 2D 격자(grid)에서 키와 쿼리를 배열하고, 지역적 쿼리 풀(local pool of queries)을 통해 기능을 조직하도록 유도합니다. 이러한 공간적 쿼리 및 공간적 재가중치(spatial reweighting) 기법을 통해, Transformer의 각 계층에서 의미 있는(topographic) 조직을 형성할 수 있도록 합니다.

- **Technical Details**: 이 연구에서 개발된 Topoformer는 Transformer 아키텍처의 self-attention 레이어에 지역 연결(local connectivity) 원칙을 적용하여 작동합니다. 두 가지 주요 기법으로 공간적 쿼리(spatial querying)와 공간적 재가중치(spatial reweighting)를 사용하여 각 쿼리와 값, 키 간의 공간적 관계를 형성합니다. 이러한 방법은 토픽 그룹을 사용하여 언어 표현을 위치적으로 조직하는데 도움을 주며, 이는 DNN의 해석 가능성을 높이는 데 기여할 수 있습니다.

- **Performance Highlights**: Topoformer 모델은 감정 분류(sentiment classification) 작업과 같은 다양한 자연어 처리(NLP) 벤치마크에서 비토포그래픽(non-topographic) 모델과 유사한 성능을 보여줍니다. 하지만 Topoformer는 뇌의 언어 네트워크와 정렬된 해석 가능한 공간적 조직을 생성할 수 있어, NLP 연구의 해석 가능성을 높이는 가능성을 지니고 있습니다. 이 연구는 Transformer 모델이 인간의 언어 처리에 대한 더 나은 모델링을 제공할 수 있는 가능성을 열어줍니다.



### Verifiable Accuracy and Abstention Rewards in Curriculum RL to Alleviate Lost-in-Conversation (https://arxiv.org/abs/2510.18731)
- **What's New**: 이번 연구에서는 멀티 턴 대화에서 발생하는 성능 저하 현상인 'Lost-in-Conversation'(LiC) 문제를 해결하기 위한 새로운 프레임워크인 RLAAR을 제안합니다. 이는 강화 학습(Reinforcement Learning) 기법을 활용하여 모델이 단순히 올바른 답변을 생성하는 것뿐만 아니라 질문의 해결 가능성을 판단할 수 있도록 돕습니다. RLAAR은 점진적으로 대화의 난이도를 증가시키는 커리큘럼(curriculum)을 통해 훈련 안정성을 유지하면서 신뢰성 있는 모델 구축을 목표로 합니다.

- **Technical Details**: RLAAR은 강화 학습을 기반으로 하며, 다회적 롤아웃(on-policy rollouts)과 혼합 보상 체계를 통해 모델이 문제 해결과 정보 수집 사이의 균형을 유지하도록 교육합니다. 사용자는 모델의 응답이 다음 턴의 상태가 되도록 하여 대화의 동적 특성을 고려합니다. 또한, '정보가 부족함을 인정하는' 움직임을 통해 모델이 조기 답변을 피하도록 지원하는 생략 보상(abstention rewards)을 포함하고 있습니다.

- **Performance Highlights**: RLAAR은 LiC 벤치마크에서 성능 저하를 62.6%에서 75.1%로 크게 완화하고, 보정된 생략률(calibrated abstention rates)을 33.5%에서 73.4%로 향상시킵니다. 이러한 결과들은 멀티 턴 환경에서 신뢰성 있고 신뢰할 수 있는 LLM 구축을 위한 실용적인 방법론을 제공합니다. 연구에서는 제안된 방법이 기존 모델 및 최신 모델들과 비교하여 현저한 성능 개선을 이룬다는 점을 강조하고 있습니다.



### SemiAdapt and SemiLoRA: Efficient Domain Adaptation for Transformer-based Low-Resource Language Translation with a Case Study on Irish (https://arxiv.org/abs/2510.18725)
Comments:
          8 pages

- **What's New**: 본 연구에서는 Parameter-efficient fine-tuning (PEFT) 기법을 제공하며, 특히 Low-Rank Adaptation (LoRA) 방식에 초점을 맞추어 저 자원 도메인인 아일랜드어 번역의 성능을 향상시킵니다. 새로운 방법인 SemiAdapt와 SemiLoRA를 제안하며, 이는 영어에서 아일랜드어로의 번역에서 더 나은 성능을 달성할 수 있도록 합니다. 전체 모델 파인튜닝을 뛰어넘는 성능을 보이는 SemiLoRA의 가능성을 강조하며, 연구자들에게 효과적인 도메인 적응 접근 방안을 제공합니다.

- **Technical Details**: SemiAdapt 및 SemiLoRA는 반지도 학습 방법으로, 데이터 세트에 도메인을 제로샷으로 할당하고, 이후 전 모델 파인튜닝 또는 저랭크 적응형(LoRA) 레이어의 훈련을 포함합니다. 도메인 임베딩 중심점을 이용하여 추론 시 도메인을 효율적으로 할당하는 방식입니다. 이 연구에서는 데이터셋 수준의 도메인 라벨링과 문장 수준의 도메인 라벨링을 비교하는 실험을 수행하여, 문장 수준의 할당이 더 나은 번역 성능을 제공함을 입증합니다.

- **Performance Highlights**: SemiAdapt와 SemiLoRA는 전통적인 파인튜닝 방법보다 더욱 효율적이며, 아일랜드어 번역 성능을 크게 향상시킵니다. SemiLoRA는 여러 도메인에서 전체 모델 파인튜닝을 초월하는 성능을 보여주며, 이를 통해 연구자들이 저 자원 언어 모델링을 보다 쉽게 접근할 수 있도록 합니다. 이 연구에서 개발된 아일랜드어 번역 모델은 공개 자원으로 제공되어, 저 자원 언어 연구자들에게 유용한 도구가 될 것입니다.



### Adapting Language Balance in Code-Switching Speech (https://arxiv.org/abs/2510.18724)
Comments:
          Submitted to ICASSP 2026

- **What's New**: 이번 연구는 코드 스위칭(code-switching) 테스트에 대한 대처 방안을 제시합니다. 기존의 대규모 모델이 코드 스위칭 사례에 대한 성능이 떨어지는 문제를 해결하기 위해, 코드 스위칭이 발생하는 지점을 명확히 하고 이에 따라 학습 과정에서 레이블을 제공하는 접근 방식을 탐구합니다. 이로써, 모델이 비정상적인 코드 스위칭 순간을 보다 잘 인식하도록 돕는 것을 목표로 합니다.

- **Technical Details**: 저자들은 기존의 cross-entropy loss 대신에 token-weighted cross-entropy loss를 제안합니다. 이는 중요한 코드 스위칭 순간을 강조함으로써 모델이 이러한 순간에서의 오류를 줄일 수 있도록 유도합니다. 특히, 이 방법은 코드를 전환하는 장소의 대표성을 높이기 위해 매트릭스 언어와 임베디드 언어의 차이를 활용합니다.

- **Performance Highlights**: 아랍어와 중국-영어 데이터세트에 대한 실험 결과는 제안된 방법이 모델의 코드 스위칭 지점 예측 정확도를 향상시킴을 보여줍니다. PIER(Point-of-Interest Error Rate) metric을 활용하여, 코드 스위칭 지점에서의 오류가 감소하는 모습을 확인하였습니다. 전반적으로, 제안된 접근법은 모델의 강건성을 높이는 데 실질적인 기여를 이루었습니다.



### Bayesian Low-Rank Factorization for Robust Model Adaptation (https://arxiv.org/abs/2510.18723)
Comments:
          Submitted to ICASSP 2026

- **What's New**: 이 논문은 대규모 음성 기초 모델의 적응에서 발생하는 문제를 다루기 위해 Bayesian factorized adapter를 제안합니다. 기존의 모델들은 코드 스위칭(code-switching)과 같은 특정 요구에 대해 적응하기가 어려웠습니다. 제안된 방법은 스패스(sparse) 적응 행렬을 달성하여 일반 성능을 유지하면서도 특정 도메인에 적응할 수 있게 합니다. Whisper 모델에 이 방법을 적용한 결과, 적응 손실이 최소화되면서도 원래 모델 능력을 유지할 수 있음을 보여주었습니다.

- **Technical Details**: 이 논문에서는 Bayesian Low-Rank Adaptation (BLoRA)을 사용하여 기준 모델의 잠재 공간에서 파괴적인 변화를 제한합니다. 이 과정에서 거대한 매개변수 수를 가진 Whisper 모델을 활용하여 적응을 진행합니다. 작고 훈련 가능한 행렬(A, B)을 도입하여 고정된 가중치를 재구성하며, 변량 추론(variational inference)을 통해 파라미터의 베이esian 사후 분포를 근사합니다. 이 방법은 저메모리 오버헤드를 유지하면서 효율적인 미세 조정(fine-tuning)을 가능하게 합니다.

- **Performance Highlights**: BLoRA는 코드 스위칭 음성 인식 작업에서 단일 단계 미세 조정(fine-tuning) 후에도 기본 모델의 성능 유지를 개선하는 실증적 결과를 보여줍니다. 본 연구의 결과에 따르면, LoRA와 비교했을 때 전이 성능이 54% 향상되었고, 새로운 도메인에서의 성능 저하는 4%로 최소화되었습니다. 이러한 성능 향상은 베이esian 적응이 일반화(generality)를 희생하지 않으면서 음성 기초 모델을 조정하는 데 효과적임을 강조합니다.



### Investigating LLM Capabilities on Long Context Comprehension for Medical Question Answering (https://arxiv.org/abs/2510.18691)
- **What's New**: 이 연구는 LLM(대형 언어 모델)의 긴 맥락(long-context)에서의 이해 능력을 최초로 조사하였다. 임상 관련 의료 질의응답(QA)에서 LLM 모델의 다양한 능력을 평가하면서 모델 크기의 효과, 한계 및 기저 메모리화 문제를 드러냈다. 특히, RAG(회수 보강 생성)가 의료 긴 맥락 이해에 미치는 영향을 조사하며, 단일 문서와 다중 문서 추론 데이터셋에서 최적의 설정을 밝혀냈다.

- **Technical Details**: LLM은 다중 선택 질문 응답(MCQA)에서 뛰어난 성과를 보이나, 복잡한 개방형 의료 QA에서는 여전히 한계가 존재한다. EHR(전자 건강 기록) 기반의 데이터셋을 사용하여 시스템을 테스트하며, RAG의 다양한 설정에서 수행 능력을 평가하였다. 본 연구는 EHR과 같은 복잡한 데이터에서 자세한 기억화 및 다양한 분류 작업에 대한 포괄적인 평가를 포함한다.

- **Performance Highlights**: RAG는 올바른 설정 하에 긴 문맥에 비해 우수한 성능을 보였고, LLM의 차별화된 과제를 구명하였다. 특히 개방형 QA 작업에서의 개별 문서 대 다중 문서 추론의 도전 과제를 다루며, LLM이 현재 직면한 과제들을 정성적으로 분석하였다. 본 연구는 의료 분야에서의 LLM 성능 및 최적화 방법에 대한 귀중한 통찰력을 제공한다.



### MLMA: Towards Multilingual with Mamba Based Architectures (https://arxiv.org/abs/2510.18684)
Comments:
          The paper is under review at ICASSP 2026

- **What's New**: 이번 논문에서는 Mamba 아키텍처를 활용한 다국어 자동 음성 인식(multilingual automatic speech recognition, ASR) 시스템인 MLMA(Multilingual Language Modeling with Mamba for ASR)를 소개합니다. MLMA는 고유의 효율적인 상태 공간 모델(state-space model)인 Mamba를 기반으로 하여, 다양한 언어에서 강력한 인식 성능을 지원합니다. 실험 결과, MLMA는 기존의 Transformer 기반 모델에 비해 경쟁력 있는 성능을 나타냈습니다.

- **Technical Details**: MLMA 모델에 사용된 Mamba 아키텍처는 구조적 상태 공간 모델(Structured State Space Model, SSM)로 정의되며, 이는 입력 시퀀스의 가변 길이와 비시간적 불규칙성을 처리할 수 있는 특징이 있습니다. 이 모델은 특징 추출 시 지역(local) 및 글로벌(global) 정보를 모두 효과적으로 통합할 수 있는 ConMamba 인코더를 사용합니다. Mamba는 또한 비선형(non-linear) 시간 변화를 허용하여 다국어 음성 데이터의 다양한 리듬 및 음성 구조에 잘 일반화될 수 있습니다.

- **Performance Highlights**: MLMA는 6개 언어, 총 12,000시간의 데이터에서 훈련되어 첫 번째 Mamba 기반의 다국어 ASR 시스템으로 자리잡았습니다. 실험 결과는 MLMA가 여러 유명 모델들(Conformer, OWSM 등)과 비교하여 우수한 인식 성능을 발휘함을 보여줍니다. 이러한 성과는 Mamba 아키텍처가 큰 규모에서 효율적이고 정확한 다국어 음성 인식을 위한 강력한 기반이 될 수 있음을 시사합니다.



### Dynamical model parameters from ultrasound tongue kinematics (https://arxiv.org/abs/2510.18629)
Comments:
          Accepted for publication in JASA Express Letters

- **What's New**: 본 연구에서는 음성 제어의 역학 모델로서 선형 조화 진동자(linear harmonic oscillator)의 매개변수를 초음파(tongue imaging) 데이터를 통해 추정할 수 있는지 평가합니다. 초음파는 기존의 전자기 아티큘로그래피(Electromagnetic Articulography, EMA)보다 덜 침습적이며 더욱 풍부한 정보를 제공함으로써 동적 모델의 평가를 획기적으로 개선할 수 있는 잠재력을 가지고 있습니다. 특히, 초음파를 사용하여 진동 매개변수를 추정하는 방법이 규명되어, 다양한 언어 샘플과 임상 데이터를 활용한 모델 평가가 가능해질 것으로 기대됩니다.

- **Technical Details**: 연구 방법으로는 노던 앵글로 브리티시 영어를 사용하는 여섯 명의 화자로부터 수집한 전자기 아티큘로그래피(EMA)와 초음파 데이터를 동시에 분석합니다. 데이터는 기음 (기본)과 여러 단어 샘플에 대해 반복적으로 수집되었으며, 이를 통해 음성의 음소/모음 생성을 연구하였습니다. 세부적으로, 초음파 데이터는 20mm 반경을 가진 프로브를 사용하여 81Hz에서 기록되었고, EMA 데이터는 1250Hz에서 수집되어 동기화되었습니다.

- **Performance Highlights**: 실험 결과, 초음파와 EMA는 동적 매개변수에서 유사한 값을 산출하였으며, 특히 턱의 단신 인대(mandibular short tendon) 추적 또한 적절하게 턱의 운동을 포착하였습니다. 이로 인해 본 연구는 음성 아티큘레이터 모델을 초음파 데이터로 평가할 수 있는 가능성을 보여주며, 이는 기존 방법들에 비해 훨씬 효과적이고 전향적인 접근 방식으로 판단됩니다.



### Beyond the Explicit: A Bilingual Dataset for Dehumanization Detection in Social Media (https://arxiv.org/abs/2510.18582)
- **What's New**: 이 논문에서는 디지털 비인간화(digital dehumanization)에 대한 연구의 부족함을 지적하고, 이 현상을 더 넓은 스펙트럼에서 이해할 수 있게 돕기 위해 다양한 샘플링 방법을 활용해 이론에 기반한 이중 언어 데이터셋을 수집하였습니다. 특히 비공식적이거나 모호한 비인간화 형태를 포착하여 사회적 소외를 겪는 집단에 대한 해로운 편견을 지속시키는 언어 패턴을 분석하였습니다.

- **Technical Details**: 연구팀은 𝕏(구 트위터)와 Reddit에서 16,000개의 비인간화 사례에 대한 주석 작업을 수행하고, 49만 개 이상의 비인간화 후보를 포함하는 대규모 이중 언어 데이터셋을 구축하였습니다. 이러한 데이터셋은 기계 학습 모델의 학습 자원으로 활용되며, 향후 비인간화 탐지 기술을 평가하기 위한 기준 역할을 합니다. 기존의 비인간화 탐지 모델의 한계를 극복하기 위해 임상적인 접근 방식을 적용하였습니다.

- **Performance Highlights**: 모델을 해당 데이터셋에 맞춰 미세 조정한 결과, 제로샷(zero-shot) 및 소수 샷(few-shot) 세팅에서의 성능이 기존의 상태에서 최첨단 모델을 초과하는 결과를 얻었습니다. 특히 소수의 예시만 제공된 경우에도 성능이 평균 12% 향상되어, 이 데이터셋이 실제 현장에서 비인간화를 탐지하는 데 효과적임을 입증하였습니다.



### Large language models for folktale type automation based on motifs: Cinderella case study (https://arxiv.org/abs/2510.18561)
- **What's New**: 이번 논문에서는 디지털 인문학을 포함한 여러 연구 분야에서 인공지능 접근법을 활용하는 새로운 방법론을 제안합니다. 우리는 특히 전래 동화인 신데렐라의 다양한 변형을 대규모로 분석하기 위한 방법을 개발했습니다. 이 연구는 머신 러닝과 자연어 처리(Natural Language Processing)를 활용하여 꿈의 모티프(motif)를 자동으로 탐지하는 방식으로 진행되었습니다.

- **Technical Details**: 연구에서는 여러 기법을 통해 대량의 문자 데이터 집합에서 유사성과 차이를 분석하기 위해 클러스터링(clustering)과 차원 축소(dimensionality reduction) 기술을 사용하였습니다. 이를 통해 대규모 언어 모델(large language models)이 이야기 속 복잡한 상호작용을 탐지할 수 있음을 보여주고, 동시에 방대한 텍스트 집합에 대한 계산 분석(computational analysis)을 가능하게 합니다.

- **Performance Highlights**: 연구 결과는 신데렐라 변형 집합의 모티프를 탐지하고 분석하는 데 있어 뛰어난 성능을 보였습니다. 이러한 방법론은 다양한 언어 간의 비교(cross-lingual comparisons)를 촉진하여, 인문학 연구에 기여할 수 있는 새롭고 혁신적인 분석 도구로 자리잡을 것으로 기대됩니다.



### Building Trust in Clinical LLMs: Bias Analysis and Dataset Transparency (https://arxiv.org/abs/2510.18556)
Comments:
          Accepted to EMNLP Main 2025

- **What's New**: 본 논문은 대규모 언어 모델이 의료 분야에서 어떻게 책임감 있게 개발될 수 있는지를 다루고 있습니다. 특히, 교육 데이터의 특성이 모델 동작에 미치는 영향을 깊이 있게 분석하여 편향(bias) 문제를 이해하고 해결해야 한다고 주장합니다. 이를 위해 HC4라는 새로운 데이터셋을 소개하며, 이 데이터셋은 890억 개 이상의 토큰(token)을 포함하고 있습니다.

- **Technical Details**: HC4 데이터셋은 의료 분야에 특화된 데이터셋으로, 생물 의학 문헌, 메타데이터 저장소 등 다양한 자료로부터 수집되었습니다. 데이터 수집 프로세스는 데이터 품질을 유지하고, 다양한 출처를 포함하기 위해 정교하게 설계되었습니다. 연구자는 이러한 데이터셋을 통해 환자의 인종, 성별 및 연령에 따른 오피오이드(Opioid) 처방의 차별성을 분석하고 있습니다.

- **Performance Highlights**: 이 연구는 의료 AI 응용 프로그램에서 공정성(fairness)과 안전(safety)을 강화하기 위한 새로운 평가 프레임워크를 제안합니다. HC4 데이터셋은 이 연구의 주요 기여일 뿐만 아니라, 기존의 데이터셋과 새로운 방법론을 혼합하여 편향 분석을 위한 중요한 자원입니다. 궁극적으로 이 논문은 대규모 언어 모델의 개발 및 배포 시 체계적이고 투명하며 철저한 편향 평가의 채택을 촉구하고 있습니다.



### Identity-Aware Large Language Models require Cultural Reasoning (https://arxiv.org/abs/2510.18510)
- **What's New**: 본 논문은 문화적 추론(cultural reasoning, CR)의 중요성을 강조하고 이를 기존의 사실 정확성(factual accuracy) 및 언어 일관성(linguistic coherence)과 함께 AI의 기본 능력으로 간주해야 한다고 주장합니다. 문화적 추론은 모델이 문화 특유의 지식 가치 및 사회적 규범을 인식하고 사용자 개별의 기대에 맞추어 출력을 조정할 수 있는 능력입니다. 이를 통해 AI 시스템은 더욱 민감하게 인간 문화의 복잡한 특성을 반영할 수 있게 됩니다.

- **Technical Details**: 현재의 평가 방법론은 대부분 정적인 정확도(static accuracy scores)를 기반으로 하여 맥락에서 적응적인 추론(adaptive reasoning)을 포착하지 못하고 있습니다. 다수의 연구에서는 LLM이 서구(norms) 문화에 편향되어 있다는 점을 지적하며, 이는 비서구적 문화 맥락에서 그 활용성을 제한합니다. 이러한 문제를 해결하기 위해 저자는 LLM의 CR 능력을 평가할 수 있는 강력한 다학제적 평가 프레임워크의 개발을 제안합니다.

- **Performance Highlights**: LLM의 출력은 종종 사회적 편견(social biases)을 내포할 수 있으며, 이는 훈련, 데이터 또는 평가 디자인에서 명시적으로 해결되지 않는 경우 발생합니다. 본 논문에서는 LLM이 문화적으로 민감한 조정이 가능한지 평가하는 방법을 제안하며, 문화적 다양성을 반영하는 대체적 데이터셋의 필요성을 강조합니다. 논문이 제안하는 평가 프레임워크는 AI가 언어적 능력뿐만 아니라 문화적 조율(cultural attunement)을 갖출 수 있도록 지원하는 방향으로 나아갑니다.



### How Efficient Are Diffusion Language Models? A Critical Examination of Efficiency Evaluation Practices (https://arxiv.org/abs/2510.18480)
- **What's New**: 이 논문은 최근에 등장한 Diffusion Language Models (DLMs)가 기존의 Autoregressive (AR) 모델과 비교하여 효율성 측면에서 어떤 문제를 겪고 있는지를 체계적으로 연구하였습니다. 특히, DLM은 이론적으로는 더 빠른 텍스트 생성을 가능하게 하지만, 실제로는 많은 경우 AR 모델보다 느린 성능을 보이며, 이는 DLM의 실용성을 제한합니다. 최근의 연구들은 효율성을 개선하기 위한 다양한 가속화 전략을 모색하고 있으나, 기존 연구의 평가 방법에서 여러 가지 주요 문제가 발견되었습니다.

- **Technical Details**: DLM은 여러 토큰을 병렬로 생성하는 구조로, 시퀀스 길이와 배치 크기가 다양할 때 AR 모델과의 성능을 시스템적으로 비교합니다. 연구에서는 FLOPs/s와 FLOPs/token의 관점에서 이론 분석을 수행했으며, roofline 모델을 통해 다양한 조건에서의 처리량을 분석했습니다. 또한, DLM의 가속화 방법으로 두 가지 주요 접근법을 탐구했으며, 각 방법의 효과를 비판적으로 분석하였습니다.

- **Performance Highlights**: 연구 결과에 따르면, AR 모델은 대부분의 평가 설정에서 처리량이 가장 높고, 스로트 확산 방식을 사용하는 모델들은 두 번째로 높은 처리량을 보여주었습니다. 반면 DLM은 일반적으로 가장 낮은 처리량을 기록하였습니다. 또한, 듀얼 캐시와 병렬 디코딩과 같은 가속화 전략은 배치 크기가 작을 때 장점을 보였으나, 배치 크기가 커짐에 따라 이점이 감소하기 시작하여 결국 AR 모델에 밀리는 경향을 보였습니다.



### DART: A Structured Dataset of Regulatory Drug Documents in Italian for Clinical NLP (https://arxiv.org/abs/2510.18475)
- **What's New**: 이번 연구에서는 이탈리아 의약품청(AIFA)의 공식 자료를 바탕으로 첫 번째로 구조화된 약품 정보 코퍼스인 DART(Drug Annotation from Regulatory Texts)를 제시합니다. 이 데이터셋은 약물의 적응증(indications), 약물 부작용(adverse drug reactions), 약물 간 상호작용(drug-drug interactions) 등 핵심 약리학적 영역에 대한 구조화된 정보를 제공합니다. 또한, 대규모 언어 모델(LLMs)을 활용하여 자동화된 요약을 생성하여 임상 의사결정 도구를 지원하는 데 기여하고 있습니다.

- **Technical Details**: DART는 AIFA 포털에서 약품의 정보 요약을 자동으로 검색, 의미론적 세분화, 데이터를 구조화하는 세 가지 단계의 파이프라인을 통해 구축되었습니다. 이 과정에서는 Python과 오픈소스 라이브러리를 사용하여 작업이 진행되었으며, 정밀한 API 분석을 통해 문서의 URL를 효율적으로 가져왔습니다. 전체 문서의 약 4.1%는 텍스트 레이어가 없는 스캔된 PDF로 인해 제외되었으며, 이는 향후 OCR 기술을 통해 보완될 예정입니다.

- **Performance Highlights**: DART를 활용한 연구 결과는 LLM 기반의 약물 상호작용 확인기가 임상적으로 의미 있는 상호작용을 정확히 추론할 수 있음을 보여줍니다. 이 데이터셋을 통해 더 나은 의료 결정을 도출할 수 있으며, 이탈리아 임상 NLP 커뮤니티에 상당한 가치를 제공하고 있습니다. DART는 16,000개 이상의 RCP와 9,500만 개 이상의 토큰을 처리하여 대규모 언어 모델의 교육과 평가를 위한 견고한 기반을 제공합니다.



### IMB: An Italian Medical Benchmark for Question Answering (https://arxiv.org/abs/2510.18468)
- **What's New**: 이 논문에서는 환자와 의사 간의 대화로 구성된 두 가지 이탈리아 의료 벤치마크 데이터셋인 IMB-QA와 IMB-MCQA를 소개합니다. IMB-QA는 77개 의료 카테고리에서 782,644개의 대화를 포함하고 있으며, IMB-MCQA는 25,862개의 다중 선택 질문으로 구성되어 있습니다. 이러한 의료 포럼의 비공식적인 특성과 언어적 복잡성을 극복하기 위해 대형 언어 모델(LLMs)을 활용하여 데이터의 명확성과 일관성을 높이는 방법을 제시합니다.

- **Technical Details**: IMB-QA는 사용자 질문과 인증된 전문가의 답변을 포함하여 실제 의료 문제를 반영한 비구조화된 데이터를 수집합니다. IMB-MCQA는 의과 대학 시험 시뮬레이터로부터 구조화된 다중 선택 질문을 수집하여 특정 의료 지식을 평가할 수 있는 환경을 제공합니다. 연구진은 Retrieval Augmented Generation(RAG)과 도메인 특화 미세 조정(fine-tuning) 기법을 적용하여 LLM 아키텍처의 성능을 다양한 크기와 교육 배경을 지닌 모델 간에 비교합니다.

- **Performance Highlights**: RAG를 통한 실험에서 응답의 정확도와 완전성을 크게 향상시켰으며, 소형 모델조차도 효과적인 작업 적응을 통해 의학적 질문 응답 성능을 개선할 수 있음을 보여주었습니다. 결과적으로, 모델의 규모보다 도메인 전문성과 효율적인 정보 검색이 의학적 AI 시스템 개발에 더욱 중요할 수 있음을 강조합니다. 이러한 연구 결과는 다중 언어 의료 질문 응답 연구의 발전에 기여할 수 있는 기초 데이터를 제공합니다.



### CEFR-Annotated WordNet: LLM-Based Proficiency-Guided Semantic Database for Language Learning (https://arxiv.org/abs/2510.18466)
- **What's New**: 이번 연구에서는 WordNet의 감각 정의(sense definition)에 유럽 언어 공통 기준(CEFR) 수준을 통합하여, 언어 학습자가 이해하기 쉬운 새로운 리소스를 개발했습니다. WordNet는 155,000개의 단어와 207,000개의 감각을 계층적 의미 네트워크로 조직한 방대한 영어 어휘 데이터베이스입니다. 연구진은 대형 언어 모델(LLM)을 사용하여 WordNet과 영어 어휘 프로필(English Vocabulary Profile) 온라인의 항목 간 의미 유사성을 자동으로 측정하는 방법을 도입했습니다. 이로 인해 언어 프로ficiency 수준에 맞춘 감각 제시가 가능해져, 외국어 학습의 효율성을 높일 수 있습니다.

- **Technical Details**: WordNet의 감각을 CEFR 수준으로 주석 처리하기 위해, 연구진은 세 가지 단계로 진행하는 주석 파이프라인을 설계하였습니다. 첫 번째 단계에서는 WordNet의 목표 단어에 대한 설명(glosses)을 수집하고, 두 번째 단계에서 LLM이 이들 간의 의미 유사성을 계산합니다. 마지막으로, 계산된 유사성 점수를 바탕으로 WordNet 감각에 CEFR 수준을 할당합니다. 또한, 연구진은 이 정보의 품질을 평가하기 위해 맥락에 따라 감각의 프로ficiency 수준을 예측하는 분류기를 구축했습니다.

- **Performance Highlights**: 개발된 CEFR 주석이 포함된 WordNet은 10,644개의 감각에 CEFR 수준을 부여하여 약 80%의 단어 감각을 포함합니다. 연구 결과, 이 데이터 기반으로 훈련된 분류기는 기준 데이터셋을 사용한 분류기와 유사한 성능을 보였습니다. Fine-tuning된 LLM을 통해 0.81의 Macro-F1 점수를 달성함으로써 높은 정확성을 입증하였습니다. 본 연구에서 개발한 모든 리소스는 공적으로 제공되어 자연어 처리(NLP)와 언어 교육의 연계를 통해 보다 효과적인 언어 학습을 지원할 것입니다.



### DePass: Unified Feature Attributing by Simple Decomposed Forward Pass (https://arxiv.org/abs/2510.18462)
- **What's New**: DePass(Decomposed Forward Pass)는 Transformer 모델의 내부 계산을 기반으로 한 기능 출처에 대한 새로운 접근 방식을 제시합니다. 이 방법은 hidden state를 커스터마이징된 additive component로 분해하고, attention score와 MLP(다층 퍼셉트론)의 활성화를 고정하여 정보를 정확히 추적합니다. 이를 통해 DePass는 auxiliary training 없이도 신뢰할 수 있는 세분화된 출처를 제공합니다.

- **Technical Details**: DePass는 각 hidden state를 additive components로 분해하고, 이를 통해 정보를 전파하는 메커니즘을 갖추고 있습니다. attention score와 MLP의 활성화를 고정하여 두 번째 효과를 제거함으로써, 각 component의 기여를 정확히 재구성하고, input token, attention heads, neurons 및 residual stream 등의 여러 모델 세분화에 대해 통합된 출처 프레임워크를 제공합니다. 이로 인해 기존의 방법들보다 더 정밀한 해석 가능성이 확보됩니다.

- **Performance Highlights**: DePass는 token level, 모델 구성 요소 수준, 서브스페이스 수준에서 여러 실험을 통해 효과성을 증명하였습니다. 이 방법은 Transformer 모델의 forward pass 전반에 걸쳐 손실 없는 additive decomposition을 가능하게 하여, 정보의 흐름을 정확히 추적할 수 있게 합니다. 따라서 DePass는 기계적 해석의 새로운 분석 도구로서 더 넓은 활용 가능성을 가지게 됩니다.



### ChronoPlay: A Framework for Modeling Dual Dynamics and Authenticity in Game RAG Benchmarks (https://arxiv.org/abs/2510.18455)
- **What's New**: Retrieval Augmented Generation (RAG) 시스템은 온라인 게임과 같은 동적인 분야에서 점점 더 중요해지고 있지만, 전용 벤치마크가 부족하여 이를 표준화된 방식으로 평가하기 어려운 상황입니다. 본 논문에서는 이러한 문제를 해결하기 위해 ChronoPlay라는 새로운 프레임워크를 제안합니다. ChronoPlay는 게임 RAG 벤치마크를 자동으로 지속적으로 생성하여, 게임 콘텐츠 업데이트와 플레이어 커뮤니티의 변화하는 관심을 추적할 수 있도록 설계되었습니다.

- **Technical Details**: ChronoPlay는 정보의 사실적 정확성과 사용자 중심의 진정성을 확보하기 위해 두 가지 주요 구성 요소로 이루어져 있습니다. 첫 번째는 공신력 있는 출처와 플레이어 커뮤니티에서 얻은 데이터를 통합하여 사실적이고 진정한 질문 패턴을 캡처하는 dual-source synthesis engine입니다. 두 번째는 이러한 데이터를 기반으로 평가 질문의 분포를 동적으로 조정하는 dual-dynamic update mechanism입니다.

- **Performance Highlights**: 세 가지 게임에 대해 이 프레임워크를 적용하여 최초의 동적 RAG 벤치마크를 생성했습니다. 분석 결과, ChronoPlay가 게임 환경의 복잡하고 현실적인 조건 하에서 모델 성능을 효과적으로 포착하고, 기존의 평가 방법으로는 측정할 수 없는 측면들을 강조함을 보여줍니다. 궁극적으로 이 방법론은 진화하는 지식 기반과 활발한 사용자 커뮤니티를 갖춘 다른 분야에도 적용될 수 있습니다.



### Engagement Undermines Safety: How Stereotypes and Toxicity Shape Humor in Language Models (https://arxiv.org/abs/2510.18454)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)을 활용한 유머 생성이 해로운 콘텐츠와 어떻게 연결되는지를 평가합니다. 기존의 파이프라인에서의 재밌음 최적화가 해로운 표현을 증폭하는 경향을 보인다는 사실을 발견하였습니다. 연구자는 유머, 고정관념(Stereotypicality), 독성(Toxicity)을 함께 측정하는 방식으로 이 문제를 분석합니다.

- **Technical Details**: 유머 생성은 언어 모델이 입력된 텍스트 프롬프트를 기반으로 잠재적인 유머를 생성할 수 있도록 합니다. 연구자들은 정보 이론적 신호를 사용하여 유머와 해로운 신호 사이의 관계를 조사하였습니다. 이를 통해 추론의 불확실성이 증가하며 일부 모델에서는 해로운 punchline이 더 예상 가능해지는 현상을 발견하였습니다.

- **Performance Highlights**: 연구의 결과, LLM에서 생성된 해로운 유머는 평균적으로 10-21% 더 높은 유머 점수를 얻었습니다. 고정관념이나 독성을 포함한 유머는 사용자들에게 더 재미있게 여겨지는 경향을 보였으며, 이러한 현상은 LLM 기반 평가 메트릭에서도 관찰되었습니다. 이 연구는 창의적인 생성 파이프라인 내에서 발생할 수 있는 위험성을 드러내며, 유머와 안전성을 명시적으로 절충하는 멀티 객체 생성의 필요성을 시사합니다.



### Grounding or Guessing? Visual Signals for Detecting Hallucinations in Sign Language Translation (https://arxiv.org/abs/2510.18439)
- **What's New**: 이 연구는 시그널 언어 번역(Sign Language Translation, SLT)에서 환각(hallucination) 문제를 다룹니다. 환각이란 시각적인 증거 없이 유창한 텍스트를 생성하는 모델의 주요 결점으로, 특히 SLT에서는 비디오에 대한 정밀한 그라운딩(grounding) 덕분에 더욱 중요한 문제로 부각됩니다. 연구자는 계층기반의 신뢰성 측정(reliability measure)을 제안하여 디코더가 시각 정보를 얼마나 사용하는지를 정량화합니다.

- **Technical Details**: 제안된 방법은 비디오가 마스킹(masking)될 때 내부의 변화(내부 상태 및 라우팅 변화와 같은)를 측정하는 피처 기반 감도(feature-based sensitivity)와 클린(clean) 및 변형(altered) 비디오 입력 간의 확률 차이를 캡처하는 반사실적 신호(counterfactual signals)를 결합하여 신뢰도 점수를 산출하는 구조입니다. 이 신뢰도 점수는 문장 수준에서의 신뢰도 지표로 변환되어, SLT의 환각 예측을 통해 효과적인 도구로 자리잡았습니다.

- **Performance Highlights**: 연구 결과, 제안된 신뢰도 측정은 환각 발생률을 예측하는 데 있어 뛰어난 성능을 보여주며, PHOENIX-2014T 및 CSL-Daily 데이터셋에서 신뢰성 점수가 정량화된 환각 예측이 가능합니다. 특히, 이 방법은 전통적으로 사용되던 텍스트 기반 신뢰도 메트릭스를 초월하며, 시각 정보 사용이 부족한 글로스프리(gloss-free) 모델들이 환각에 더욱 취약하다는 사실을 발견했습니다.



### Chain-of-Conceptual-Thought: Eliciting the Agent to Deeply Think within the Respons (https://arxiv.org/abs/2510.18434)
- **What's New**: 이번 논문에서는 LLM의 open-domain (개방형) 대화에서의 성능을 향상시키기 위한 새로운 접근 방법인 Chain of Conceptual Thought (CoCT)를 제안합니다. CoCT는 LLM이 먼저 관련된 개념을 태깅한 후, 이에 기반하여 세부 내용을 생성하도록 유도합니다. 이전의 Chain-of-Thought (CoT) 방법론과는 달리, CoCT는 개념 전환이 가능하여 LLM의 깊이 있는 사고를 촉진합니다.

- **Technical Details**: CoCT는 먼저 감정, 전략, 주제를 포함하는 개념 목록에서 하나의 개념을 선택하도록 LLM에 지시합니다. 이를 통해, LLM은 각 응답에서 여러 개념 전환을 가능하게 하여 보다 자연스러운 대화를 창출합니다. 이 연구에서는 또한 감정과 전략, 그리고 주제의 개념을 다루며, 사전 정의된 개념 목록이 제안되었습니다.

- **Performance Highlights**: 실험 결과, CoCT는 Self-Refine, ECoT 등 기존의 다양한 기준선 모델을 초월하는 성과를 보였습니다. 특히, CoCT는 open-domain 대화와 감정 지원 대화에서 우수한 성능을 발휘하며, 외부 도메인 개념과 질문에 대해서도 강인한 일반화 능력을 보여줍니다. 이를 통해 CoCT는 포괄적인 대화 작업에 적합한 유력한 접근 방식으로 입증되었습니다.



### Adamas: Hadamard Sparse Attention for Efficient Long-Context Inferenc (https://arxiv.org/abs/2510.18413)
- **What's New**: 이 논문에서는 Adamas라는 경량의 고정밀도 희소 주목 메커니즘을 소개하여 긴 컨텍스트 추론(long-context inference)을 위한 효율성을 크게 향상시킵니다. Adamas는 Hadamard 변환, 버킷화(bucketization) 및 2비트 압축을 적용하여 컴팩트한 표현을 생성하고, 맨하탄 거리 추정(Manhattan-distance estimation)을 활용하여 효율적인 top-k 선택을 수행합니다. 실험 결과, Adamas는 64토큰 예산만으로 전체 주목(full attention)과 동등한 정확도를 기록하며, 128토큰에서 손실이 거의 없는 성능을 발휘하고, 이전 SOTA보다 최대 8배 높은 희소성을 지원합니다.

- **Technical Details**: Adamas는 주목 품질을 유지하면서 8배의 높은 희소성을 달성하는 것을 목표로 합니다. 이는 Hadamard 변환을 사용하여 쿼리와 키 유사성을 토큰 수준에서 효율적으로 근사하도록 설계되었습니다. Adamas는 쿼리와 키를 Hadamard 변환으로 변환한 다음 2비트 코드로 압축하여 KV 캐시(Cache)에 저장하며, 이를 통해 메모리 오버헤드를 최소화합니다. 디코딩 과정에서는 압축된 코드로 맨하탄 거리 추정기를 사용하여 후보 키를 신속하게 미리 선택합니다.

- **Performance Highlights**: Adamas는 32K 길이 시퀀스에서 최대 4.4배의 자기 주목(self-attention) 속도 향상과 1.5배의 전체적(end-to-end) 속도 향상을 달성합니다. 또한, Adamas는 낮은 유사성(perplexity)을 기록하여 기존의 전체 주목과 비교할 때 성능이 더 우수한 것으로 입증되었습니다. 매우 높은 희소성을 유지했음에도 불구하고, Adamas는 여전히 전체 주목과 동등한 수준의 정확도를 유지합니다.



### MENTOR: A Reinforcement Learning Framework for Model Enhancement via Teacher-Optimized Rewards in Small Models (https://arxiv.org/abs/2510.18383)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 도구 사용 능력을 더 작고 효율적인 소형 언어 모델(SLM)로 증류하는 새로운 방법인 MENTOR를 제안합니다. 기존의 감독하에 세부 조정(Supervised Fine-Tuning, SFT) 접근 방식은 정적인 교사 궤적을 모방하도록 모델을 훈련시켜 일반화에 어려움을 겪고 있습니다. MENTOR는 이러한 단점을 극복하기 위해 강화 학습(Reinforcement Learning, RL)과 교사-guided 증류를 조합하여 더 일반화된 정책을 학습하는 비전통적인 접근 방식을 사용합니다.

- **Technical Details**: MENTOR 프레임워크는 RL 기반의 탐색 과정을 통해 일반화 가능성이 높은 정책을 학습합니다. 이 과정에서 보상의 희소성 문제를 해결하기 위해 교사의 참조 궤적을 사용하여 조밀하고 복합적인 교사-guided 보상을 구성합니다. 이러한 보상은 SO الار οδη기와 보다 세밀한 안내를 제공하여 SLM의 학습 효율성을 높입니다.

- **Performance Highlights**: 실험 결과 MENTOR는 SFT 및 표준 희소 보상 RL 기준선과 비교하여 SLM의 도메인 간 일반화(cross-domain generalization) 및 전략적 능력(strategic competence)을 크게 향상시킨 것으로 나타났습니다. 이에 따라 MENTOR는 소형 언어 모델의 실제 적용 가능성을 향상시키는 데 중요한 기여를 합니다.



### Towards Fair ASR For Second Language Speakers Using Fairness Prompted Finetuning (https://arxiv.org/abs/2510.18374)
Comments:
          Submitted to ICASSP 2026

- **What's New**: 이번 연구에서는 제2언어 사용자들을 위한 공정한 영어 자동 음성 인식(ASR) 시스템 구축의 어려움을 다룹니다. Whisper와 Seamless-M4T와 같은 널리 사용되는 ASR 모델의 단어 오류율(Word Error Rate, WER)에서 26개의 억양 그룹 간에 큰 변동성이 있음을 발견했으며, 이는 공정성의 격차를 시사합니다. 이를 해결하기 위해 우리는 경량 어댑터를 통한 공정성 유도 파인튜닝(fairness-prompted finetuning) 방법을 제안하며, 여기에는 스펙트럴 분리(Spectral Decoupling, SD), 그룹 분포 강건 최적화(Group Distributionally Robust Optimization, Group-DRO), 불변 위험 최소화(Invariant Risk Minimization, IRM)를 포함합니다.

- **Technical Details**: 이 연구에서는 서로 다른 모국어에 의해 형성된 26개의 그룹에 대한 ASR 성능의 격차를 최소화하는 것을 목표로 합니다. 음성 샘플과 그에 해당하는 영어 전사 및 억양 그룹 정보를 기반으로 한 L2 영어 발화를 분석합니다. Whisper와 Seamless-M4T라는 두 개의 사전 훈련된 ASR 모델을 사용하여, ERM(경험적 위험 최소화)을 기본으로 하여 공정성을 증진하기 위한 알고리즘인 SD, Group-DRO, IRM과 함께 파인 튜닝을 수행합니다. 또한, 모델의 크기와 데이터 속성이 공정성 유도 파인 튜닝의 효능에 미치는 영향을 분석합니다.

- **Performance Highlights**: 제안된 접근 방식은 Whisper와 Seamless-M4T에 대한 매크로 평균 WER에서 각각 58.7% 및 58.5%의 상대적 개선을 달성했습니다. 또한, 표준 경험적 위험 최소화(ERM)와 교차 엔트로피 손실을 기반으로 한 파인 튜닝에 비해 각각 9.7% 및 7.8%의 성능 향상을 이루었습니다. 이러한 결과는 다양한 L2 영어 억양 간의 성능 격차를 줄이는 데에 기여하며, ASR 시스템의 공정성을 높이는 데 중요한 기초 자료로 활용될 것입니다.



### KoSimpleQA: A Korean Factuality Benchmark with an Analysis of Reasoning LLMs (https://arxiv.org/abs/2510.18368)
- **What's New**: 이번 연구에서는 한국 문화 지식에 중점을 두고 대규모 언어 모델(LLM)의 사실성 평가를 위한 벤치마크인 한국어 SimpleQA (KoSimpleQA)를 제안합니다. KoSimpleQA는 1,000개의 명확한 답변을 가진 짧은 사실 질문들로 구성되어 있어 도전적이면서도 평가하기 쉽습니다. 이 연구는 다양한 개방형 LLM을 평가하여 가장 강력한 모델조차도 33.7%의 정답률에 그치며, KoSimpleQA의 난이도를 강조합니다. 또한, KoSimpleQA의 성능 순위가 영어 SimpleQA와 상당히 다르다는 점도 주목할 만합니다.

- **Technical Details**: KoSimpleQA는 한국 문화 지식을 반영하는 질문으로 구성되어 있으며, 한국 커뮤니티와 다국어 지원 모델을 포함한 다양한 LLM 그룹을 평가했습니다. 각 질문은 정확한 하나의 짧은 답변을 요구하며, 이는 신뢰할 수 있는 평가를 보장합니다. 데이터셋의 질을 높이기 위해 두 단계의 검증 과정을 통해 평가를 실시했으며, 전문가는 샘플 질문을 수동으로 검사하여 문제 사례에 대한 피드백을 제공했습니다. 이러한 품질 관리 과정을 통해 KoSimpleQA의 사실성을 강화했습니다.

- **Performance Highlights**: KoSimpleQA에서 다양한 LLM 모델의 성능을 비교한 결과, 한국 커뮤니티 LLM이 다국어 모델보다 우수한 성능을 보였습니다. 그러나 다국어 모델이 원본 SimpleQA에서는 더 나은 성능을 보였으며, 이는 KoSimpleQA가 기존 벤치마크에서 평가되지 않는 요소를 포착하고 있음을 시사합니다. 현재 모델들이 사실성 문제를 다룰 때 추론 능력이 어떻게 영향을 미치는지를 분석함으로써, LLM의 강점과 한계에 대한 새로운 통찰을 제공합니다.



### KrishokBondhu: A Retrieval-Augmented Voice-Based Agricultural Advisory Call Center for Bengali Farmers (https://arxiv.org/abs/2510.18355)
Comments:
          6 pages, 7 figures, 5 tables, submitted to the 11th IEEE International Women in Engineering (WIE) Conference on Electrical and Computer Engineering (WIECON-ECE 2025)

- **What's New**: 본 논문에서는 한국어가 아닌 벵골어를 사용하는 농민들을 위한 음성 기반 농업 조언 시스템, KrishokBondhu를 소개합니다. 본 시스템은 Retrieval-Augmented Generation (RAG) 프레임워크를 기반으로 하여 권위 있는 농업 자료를 통합적으로 활용하며, 농민들이 좀 더 신뢰할 수 있는 기술적 조언을 쉽게 받도록 설계되었습니다. KrishokBondhu는 전화 인터페이스를 통해 실시간으로 농업 관련 조언을 제공합니다.

- **Technical Details**: KrishokBondhu의 지식 기반은 방글라데시 정부 및 비정부기구에서 발행한 농업 관련 문서로 구성됩니다. 약 2,500페이지 분량의 자료들은 Optical Character Recognition (OCR) 및 문서처리 파이프라인을 통해 디지털화 및 구조화되며, Vector Database에 인덱싱 되어 효율적인 검색이 가능하게 됩니다. 음성 인식 모듈은 농민의 질의를 텍스트로 변환하는 역할을 하며, Gemma 3-4B라는 대형 언어 모델이 컨텍스트에 맞는 응답을 생성합니다.

- **Performance Highlights**: KrishokBondhu는 다양한 농업 관련 질의에 대해 72.7%의 높은 답변 품질을 기록하였으며, KisanQRS 벤치마크와 비교할 때 44.7%의 개선을 이루었습니다. 특히, 컨텍스트의 풍부함과 완전성에서 각각 367%와 100.4%의 향상을 보였습니다. 이 시스템은 콜센터 접근성, 다국어 음성 상호작용, 현대 RAG 기법을 결합하여 외딴 지역의 농민들에게 전문가 수준의 농업 조언을 제공하는 가능성을 입증하였습니다.



### Combining Distantly Supervised Models with In Context Learning for Monolingual and Cross-Lingual Relation Extraction (https://arxiv.org/abs/2510.18344)
- **What's New**: 본 논문은 HYDRE(하이브리드 원거리 감독 관계 추출) 프레임워크를 제안합니다. HYDRE는 학습된 DSRE 모델을 사용하여 테스트 문장에 대해 상위 후보 관계를 식별하며, 이후 신뢰할 수 있는 문장 수준의 예제를 동적으로 검색하여 LLM의 프롬프트로 제공합니다. 기존 DSRE 모델과 함께 LLM을 활용한 새로운 방법론이 제시되어, 특히 자원이 부족한 언어에서의 성능을 개선할 수 있는 가능성을 보여줍니다.

- **Technical Details**: HYDRE는 먼저 DSRE 모델을 통해 상위 K 개 후보 관계를 식별합니다. 이후, 모델의 신뢰도와 의미적 유사성을 결합하여 훈련 데이터에서 신뢰할 수 있는 예제를 동적으로 추출합니다. 이러한 예제는 LLM의 프롬프트에 포함되어 최종 관계를 제시하는 데 사용됩니다. 또한, HYDRE는 저자원 언어의 관계 추출을 위해 크로스-링구얼 설정으로 확장되었습니다.

- **Performance Highlights**: HYDRE는 영어와 네 가지 저자원 인도 언어에서 기존의 최첨단 DSRE 모델 대비 최대 20 F1 포인트 향상된 성능을 기록했습니다. 실험 결과, HYDRE는 오픈 소스 및 독점 LLM에 대해 일관되게 우수한 성능을 보였으며, 동적 예제 검색 전략이 단일 언어와 크로스-링구얼 설정 모두에서 안정적임을 보여주었습니다. 모델 성능을 저하시킬 수 있는 요소를 제거했을 때, 최대 7 마이크로 F1 포인트의 손실이 발생했습니다.



### ECG-LLM -- training and evaluation of domain-specific large language models for electrocardiography (https://arxiv.org/abs/2510.18339)
Comments:
          34 pages, 8 figures, code available at this https URL

- **What's New**: 이번 연구는 의료 분야에서 전자심전도(ECG)에 대한 도메인 특화된 언어 모델의 성능을 평가하기 위해 개방형 가중치의 대형 언어 모델(LLMs)을 조정하고, 다층 평가 체계를 구현하였습니다. 특히, 도메인 적합성을 강화한 여러 모델과 일반 목적의 모델인 Claude Sonnet 3.7을 비교하였습니다. 본 연구는 비공식적 모델이 개인정보 보호를 유지하면서도 임상 솔루션으로서의 가능성을 보여줍니다.

- **Technical Details**: 연구에서는 Llama 3.1 70B 모델을 도메인 특화 문헌을 기반으로 파인튜닝(finetuning)하였고, 질문-답변 쌍을 활용하여 다양한 평가 체계를 적용했습니다. 자동 텍스트 메트릭과 전문가의 평가를 통해, 조정된 모델은 일반 목적 모델보다 뛰어난 성능을 보였습니다. 이 연구에서는 retrieval-augmented generation (RAG) 기법을 통해 정보의 정확성을 높이는 접근법도 함께 탐구하였습니다.

- **Performance Highlights**: 파인튜닝된 Llama 3.1 70B는 객관식 평가와 자동 텍스트 메트릭에서 우수한 성능을 기록하였고, 전문가 평가에서는 Claude 3.7과 RAG 접근법이 복잡한 질문에서 더 높은 점수를 받았습니다. 전체적으로, 파인튜닝된 모델은 기본 모델보다 거의 모든 평가 방식에서 월등한 성과를 보여주었으며 이로 인해 평가 방법론의 복잡성을 강조할 수 있었습니다.



### From Retrieval to Generation: Unifying External and Parametric Knowledge for Medical Question Answering (https://arxiv.org/abs/2510.18297)
Comments:
          13 pages, 4 figures

- **What's New**: 본 논문에서는 MedRGAG라는 새로운 프레임워크를 제안합니다. MedRGAG는 의료 질문 응답(QA)을 위해 외부 지식과 매개변수 지식을 통합하는 통합된 검색-생성 방식의 구조를 가지고 있습니다. 이 프레임워크는 Knowledge-Guided Context Completion (KGCC)과 Knowledge-Aware Document Selection (KADS) 두 가지 핵심 모듈로 구성되어 있어, 신뢰할 수 있는 응답 생성을 위해 필요한 증거를 효과적으로 통합합니다.

- **Technical Details**: MedRGAG의 첫 번째 모듈인 KGCC는 검색된 문서를 분석하고 누락된 지식을 판단한 후, 필요한 배경 문서를 생성하는 역할을 합니다. 두 번째 모듈인 KADS는 검색된 문서와 생성된 문서를 지식 요구사항에 기반하여 그룹화하고, 적합한 증거의 조합을 선택합니다. 이러한 통합 설계를 통해 MedRGAG는 의료 QA에서의 정확성 및 신뢰성을 크게 향상시킬 수 있습니다.

- **Performance Highlights**: 다섯 가지 의료 QA 벤치마크에서의 실험을 통해 MedRGAG가 MedRAG에 비해 평균 12.5%, MedGENIE에 비해 4.5% 향상된 정확도를 달성했음을 보여주었습니다. 이 결과는 MedRGAG가 검색과 생성을 통합하여 지식 집약적 추론에 효과적임을 강조합니다. 또한, MedRGAG는 보다 효과적인 보완적 배경 문서를 생성하고, 유용한 증거를 성공적으로 복구하는 것으로 확인되었습니다.



### Food4All: A Multi-Agent Framework for Real-time Free Food Discovery with Integrated Nutritional Metadata (https://arxiv.org/abs/2510.18289)
- **What's New**: 이 논문은 미국에서 지속적으로 문제되는 식량 불안정 문제를 해결하기 위한 혁신적인 시스템인 Food4All을 소개합니다. 이 시스템은 다양한 데이터 소스의 집합체를 활용하여 실시간으로 무료 식량 정보를 제공하고, 식료품 선택에 있어 실질적이며 개인화된 조언을 제공합니다. 이를 통해, Food4All은 기존의 정적 디렉터리 및 일반 검색 엔진의 한계를 극복하고, 더욱 접근 가능하고 유용한 솔루션을 제공합니다.

- **Technical Details**: Food4All은 세 가지 혁신적인 요소로 구성되어 있습니다: 첫째, 공식 데이터베이스와 커뮤니티 플랫폼, 소셜 미디어에서 이질적인 데이터 소스를 집계하여 지속적으로 업데이트되는 식료품 자원 풀을 제공합니다. 둘째, 경량화된 강화 학습( reinforcement learning ) 알고리즘을 도입하여 지리적 근접성과 영양적 올바름을 최적화합니다. 셋째, 사용자 피드백을 통해 동적으로 정책을 조정하는 온라인 학습 루프를 통합하여 사용자 요구에 지속적으로 적응합니다.

- **Performance Highlights**: Food4All은 지속적으로 갱신되는 데이터베이스를 통해 신뢰할 수 있는 무료 식량 정보를 제공합니다. 이 시스템은 사용자 중심으로 설계되어 있으며, 실제 사용자의 요구에 맞는 영양 정보 제공을 통해 식량 불안정에 직면한 취약 집단의 지원을 강화합니다. 이를 통해, Food4All은 스케일 가능한, 공정하며, 지능적인 식량 정보 시스템으로 나아가는 중요한 단계를 마련합니다.



### BrailleLLM: Braille Instruction Tuning with Large Language Models for Braille Domain Tasks (https://arxiv.org/abs/2510.18288)
Comments:
          Accepted to EMNLP 2025

- **What's New**: 이번 연구는 시각 장애인을 위한 교육 및 정보 접근성에 필수적인 점자(Braille) 처리의 도전 과제를 해결하기 위해 영어 및 중국어 점자 혼합 데이터셋(EBMD/CBMD)을 구축하고, 점자 데이터에 최적화된 구문 트리 기반의 증강 방법을 제안합니다. 또한 기존의 미세 조정(fine-tuning) 기법의 한계를 극복하기 위해 점자 지식 기반 미세 조정(BKFT) 방법을 도입하여 점자 관련 작업의 성능을 향상시키고자 합니다.

- **Technical Details**: 연구에서는 BrailleLLM이라는 프레임워크를 제안하며, 이는 대규모 언어 모델을 이용하여 점자 도메인 문제를 해결합니다. 특정 작업에 맞춘 지침 템플릿을 통해 점자 및 혼합 텍스트 데이터셋을 설계하고 구축하여 모델 성능을 개선하는 방식입니다. BKFT 방법은 모델이 낮은 수준의 규칙을 재발견하는 대신 높은 수준의 의미 해석과 번역을 마스터하도록 유도합니다.

- **Performance Highlights**: 실험 결과, BKFT를 이용한 방법이 기존의 미세 조정 방식에 비해 점자 번역 시나리오에서 현저한 성능 향상을 보여주었습니다. 저자들은 공개 소스 데이터셋 및 방법론을 통해 자원이 부족한 다국어 점자 연구의 기초를 마련하고자 합니다. 연구 결과는 점자 이해 및 번역의 정확성을 높이고, 시각 장애인의 정보 접근성을 크게 향상시킬 것으로 기대됩니다.



### Text or Pixels? It Takes Half: On the Token Efficiency of Visual Text Inputs in Multimodal LLMs (https://arxiv.org/abs/2510.18279)
Comments:
          Accepted to EMNLP 2025 Findings. Previously titled "Text or Pixels? Evaluating Efficiency and Understanding of LLMs with Visual Text Inputs"

- **What's New**: 최근 대형 언어 모델(LLMs)과 그 멀티모달 변형이 시각적 입력을 처리할 수 있게 되면서, 텍스트 입력을 이미지로 변환하여 토큰 사용량을 줄이고 성능을 유지할 수 있는 가능성이 제기되었습니다. 본 논문에서는 텍스트를 이미지로 표현함으로써 디코더 LLMs의 입력 압축이 가능하다는 것을 보여줍니다. 실험을 통해 이 텍스트-이미지 접근법이 토큰 절약에 효과적이며, 성능 저하 없이도 실질적인 이점을 제공함을 입증하였습니다.

- **Technical Details**: 이 연구에서는 멀티모달 LLMs가 시각적 텍스트 입력을 활용하여 입력 압축을 달성하는 방법에 대해 논의합니다. 긴 텍스트를 단일 이미지로 렌더링함으로써, 비전 인코더는 디코더가 처리할 수 있는 고정 길이의 시각적 토큰 시퀀스를 생성하고, 이는 시퀀스 길이를 직접 줄이는 효과를 가져옵니다. 이를 통해 기존 모델을 미세 조정하거나 추가적인 감독 없이도 디코더 토큰 수를 획기적으로 감소시킬 수 있음을 보여줍니다.

- **Performance Highlights**: 실험 결과, RULER 과제에서 GPT-4.1-mini와 Qwen2.5-VL-72B는 최대 58%의 디코더 토큰 수 감소에도 불구하고 97% 이상의 정확도를 유지했습니다. 또한, CNN/DailyMail 요약 과제에서는 이 접근법이 두 가지 전문 프루닝 기준을 초과하는 성과를 나타내며, 큰 모델에서는 전체적인 속도를 최대 45% 향상시키는 것으로 확인되었습니다. 이러한 결과들은 멀티모달 LLMs가 이미지를 암묵적인 압축 레이어로 활용하면서도 성능을 거의 원래 텍스트-토큰 비용의 절반으로 유지할 수 있음을 시사합니다.



### DelvePO: Direction-Guided Self-Evolving Framework for Flexible Prompt Optimization (https://arxiv.org/abs/2510.18257)
- **What's New**: 최근의 연구들은 Prompt Optimization (PO)이 대규모 언어 모델(Large Language Models, LLMs)의 특정 작업을보다 효과적으로 해결하기 위해 필요한 접근 방식으로 주목받고 있습니다. 기존의 연구들은 주로 LLM의 무작위 재작성 능력에 의존하고 있으며, 이로 인해 최적화 과정이 특정 요인에 집중되어 국소 최적(local optimum)에 쉽게 빠지게 되는 문제점이 있었습니다. 이를 해결하기 위해, 본 연구에서는 $	extbf{DelvePO}$ (Direction-Guided Self-Evolving Framework for Flexible Prompt Optimization)라는 프레임워크를 제안하여, 다양한 작업에 대해 자가 발전(self-evolve)하는 방식으로 프롬프트를 최적화하는 방법을 제시합니다.

- **Technical Details**: DelvePO 프레임워크는 LLM을 기반으로 한 작업 비특화적인_prompt 최적화 방법으로, 프롬프트를 여러 구성 요소로 분리하여 이를 통해 다양한 작업에서 영향 요인을 탐색할 수 있도록 합니다. 또한, 작업 메모리(working memory) 메커니즘을 도입하여 LLM의 불확실성으로 인한 단점을 완화하고, 새로운 프롬프트의 생성을 유도하기 위한 통찰을 얻는 과정을 포함하고 있습니다. 이러한 방법론은 진화 알고리즘의 효율성과 LLM의 강력한 텍스트 처리 능력을 통합하여 보다 안정적인 성능 향상을 달성합니다.

- **Performance Highlights**: 다양한 도메인에서의 여러 작업에 대한 실험 결과, DelvePO는 기존의 SOTA(최신 기술 동향) 방법들보다 일관되게 우수한 성능을 보였으며, 이는 다양한 작업들 간의 전이 가능성(transferability)을 보여줍니다. 프레임워크의 주요 기여는 각 요소의 진화 추세를 포착하는 개념적 메모리 메커니즘을 도입하여 프롬프트 최적화를 안내하고, 동시에 구성 요소의 상호 연결을 통해 전체 프롬프트의 점진적 최적화를 유도한다는 점입니다. DelvePO는 실험에서 수동으로 구성된 프롬프트보다 우수한 성능을 보여주었습니다.



### MARCUS: An Event-Centric NLP Pipeline that generates Character Arcs from Narratives (https://arxiv.org/abs/2510.18201)
- **What's New**: 이번 연구에서는 사건 중심(event-centric) 및 관계 기반(relation-based) 캐릭터 아크(character arcs)를 컴퓨터적으로 생성하는 새로운 작업을 다룹니다. MARCUS (Modelling Arcs for Understanding Stories)란 NLP 파이프라인을 제시하여, 사건, 참여 캐릭터, 암시된 감정 및 정서를 추출하여 캐릭터 간의 관계를 모델링합니다. 이 접근방식은 문헌 연구에서 중요한 이론적 도구로 여겨지는 캐릭터 아크의 양적 표현을 제공합니다.

- **Technical Details**: MARCUS는 BookNLP 파이프라인을 사용하여 이야기 내 이벤트와 참여 캐릭터를 추출하며, BiLSTM 모델과 BERT embeddings를 활용하여 사건 및 관련 개체를 추출합니다. 감정의 정량적 측정을 위해 RoBERTa 모델을 미세 조정(fine-tune)하고, GoEmotions 데이터셋을 이용하여 다양한 감정 상태를 식별합니다. 이 데이터 처리 과정들은 캐릭터의 상황 변화를 포착하기 위한 기초가 됩니다.

- **Performance Highlights**: MARCUS는 Harry Potter 및 Lord of the Rings 시리즈에서 캐릭터 아크를 생성하여 실험을 수행하였으며, 캐릭터 간의 관계 아크 더불어 이야기가 진행됨에 따라 캐릭터의 여정을 정량적으로 모델링합니다. 연구 결과, 사건이 캐릭터의 상황에 미치는 영향을 시각화한 캐릭터 아크를 생성하여 캐릭터의 변화 과정을 효과적으로 추적할 수 있음을 보여주었습니다. 이러한 접근은 기존 문헌의 감정 아크 연구와는 달리 사건을 중점적으로 다루어, 캐릭터의 여정을 더 세밀하게 이해할 수 있도록 합니다.



### Contrastive Decoding Mitigates Score Range Bias in LLM-as-a-Judg (https://arxiv.org/abs/2510.18196)
- **What's New**: 이 연구에서는 대형 언어 모델(LLM)이 평가자로 사용될 때 나타나는 점수 범위 편향(score range bias)에 대해 조사하였습니다. 특히, LLM들이 직접 평가를 수행할 때 점수 범위에 민감하다는 점을 밝혔으며, 이러한 편향은 동일 모델 계열에서도 발견되었습니다. 또한, 상대적인 평가 개선을 위하여 대비 부호화(contrastive decoding) 기법을 통해 이 편향을 완화하는 방법을 제안하였습니다.

- **Technical Details**: 대비 부호화는 두 개의 모델(주 모델과 보조 모델)을 활용하여 모델 출력을 수정하는 방법입니다. 주 모델의 다음 토큰 확률(pmain)에서 보조 모델의 확률(passt)을 가중치로 빼서 최종 점수를 조정합니다. 이 과정에서 하이퍼파라미터 λ를 포함하여 두 모델 간의 로짓 분포(logit distribution)를 더 잘 정렬합니다. 이러한 접근은 편향 분석을 통해 도출되었습니다.

- **Performance Highlights**: 이 연구에서 제안한 대비 부호화 기법은 다양한 점수 범위에서 인간 평가와의 스피어만 상관관계(Spearman correlation)를 평균 11.3% 향상시켰습니다. 요약 작업과 관련한 직접 평가에서 LLM 점수의 범위 편향을 분석하고 이를 완화함으로써, 평가의 신뢰성을 높일 수 있는 가능성을 보여주었습니다. 실험을 통해 각 모델 계열에서의 편향을 구체적으로 제시하며, 성능 향상의 기반을 마련하였습니다.



### CMT-Bench: Cricket Multi-Table Generation Benchmark for Probing Robustness in Large Language Models (https://arxiv.org/abs/2510.18173)
- **What's New**: CMT-Bench는 실시간 크리켓 해설을 기반으로 한 진단 벤치마크로, 동적 텍스트-테이블 생성 과제를 위한 새로운 평가 기준을 제시합니다. 이 벤치마크는 두 개의 진화하는 테이블에서 생성된 데이터를 바탕으로 하며, 모델이 시간에 따른 발전하는 서사를 이해하고 요약하는 방법을 탐구합니다. 이 연구는 기존의 T2T 접근 방식들이 가지고 있는 한계를 지적하며, 더 나은 시스템 설계를 위한 기초 연구의 필요성을 강조합니다.

- **Technical Details**: CMT-Bench의 설계는 동적 데이터 생성 과제를 위한 세 가지 주요 차원에 중점을 둡니다: 추출 요인의 침식(추출적인 단축어와 상태 추적 분리), 시간 접두사(prefixing) 테스트(길고 복잡한 맥락의 안정성 검사), 및 개체 형식의 변형(익명화, 분포 외 치환 등). 이 연구는 LLM의 내구성을 평가하고, 숫자 오류 패턴의 변화를 정량화하기 위해 분포적 테스트와 성능 평가를 결합했습니다.

- **Performance Highlights**: CMT-Bench에서 진행된 실험 결과, 많은 LLM들이 추출적인 요약 없이, 입력 길이에 따라 지속적으로 정확도가 저하되는 경향이 있음을 발견했습니다. 또한, 개체 형식의 변화에 따라 일관된 정확성 저하가 나타났으며, 이는 모델의 추론 방식에 변화가 있었음을 시사합니다. 이러한 결과는 현재의 LLM들이 시간에 따른 동적 텍스트-테이블 생성에서 취약함을 드러내며, 내구성 있는 평가 방식이 필요하다는 점을 강조합니다.



### Automatic Prompt Generation via Adaptive Selection of Prompting Techniques (https://arxiv.org/abs/2510.18162)
Comments:
          35 pages, 29 figures, 5 tables

- **What's New**: 이번 연구에서는 사용자로부터 제공된 추상적인 작업 설명을 기반으로 작업에 적합한 prompting 기술을 선택하고, 고품질의 프롬프트를 자동으로 생성하는 새로운 방법을 제안합니다. 이 방법은 기존의 템플릿이나 프레임워크에 의존하지 않고, semantic similarity를 가진 작업 클러스터와 함께 prompting 기술을 연결하는 지식 기반을 구성합니다. 이러한 접근 방식은 비전문가도 LLM의 기능을 효과적으로 활용할 수 있도록 지원합니다.

- **Technical Details**: 제안된 시스템은 지식 기반 구축과 프롬프트 생성의 두 가지 주요 단계로 운영됩니다. 또한, LLM을 사용하여 작업 클러스터를 정의하고, 각 클러스터에 적합한 프롬프트 기술을 연결하여 지식 기반을 형성합니다. 이 과정에서 사용자의 작업 설명을 분석하고, 적합한 프롬프트를 동적으로 생성하여, 전문 지식이 없는 사용자가 쉽게 접근할 수 있는 방안을 마련합니다.

- **Performance Highlights**: 23개의 BIG-Bench Extra Hard (BBEH) 작업에 대한 실험 평가 결과, 제안된 방법은 표준 프롬프트 및 기존의 자동 프롬프트 생성 도구와 비교하여 우수한 성능을 보여주었습니다. 평가 지표로는 산술 평균 및 조화 평균 점수를 사용하였으며, 이 연구는 프롬프트 생성의 표준화 및 간소화를 위한 기반을 다지고 있습니다.



### Extracting Rule-based Descriptions of Attention Features in Transformers (https://arxiv.org/abs/2510.18148)
Comments:
          Our code is available at this https URL

- **What's New**: 이 논문은 기계적 해석(mechanistic interpretability)의 새로운 접근법을 제안합니다. 기존의 방식이 텍스트 시퀀스가 어떤 feature를 활성화하는지에 대한 분석에 중점을 두었다면, 이 연구는 rule-based 설명법을 통해 입력 토큰 패턴과 대응하여 특정 output 토큰의 가능성을 조절하는 방법을 제시합니다. 주로 attention layer의 출력에서 학습된 SAE features(스파스 오토인코더 특징)의 규칙 기반 설명을 자동으로 추출하는 방식에 중점을 두고 있습니다.

- **Technical Details**: 연구진은 attention layer에서의 상호 작용을 세 가지 유형의 규칙으로 분석합니다. (1) skip-gram 규칙, 예를 들어 "[Canadian city] ... speaks → English", (2) 부재 규칙으로 "[Montreal] ... speaks -/-> English", (3) count-based 규칙이 있습니다. 이러한 규칙들은 예시의 inspection만으로는 쉽게 발견되지 않으며, 본 논문에서는 이를 자동으로 추출하는 파이프라인을 제시합니다.

- **Performance Highlights**: GPT-2 small 모델에서 실험한 결과, 약 100개의 skip-gram 규칙으로 대부분의 feature를 잘 설명할 수 있음을 보였습니다. 첫 번째 layer에서도 많은 부재 규칙이 발견되었으며, counting 규칙도 몇 가지 사례로 분리할 수 있었습니다. 이 연구는 기계 학습 모델의 행동을 설명하는 rule-based feature의 잠재력을 보여주며, 앞으로의 연구가 이를 기반으로 한 더 완전한 규칙 세트를 추출할 수 있는 기초를 제공합니다.



### LLMs Encode How Difficult Problems Ar (https://arxiv.org/abs/2510.18147)
- **What's New**: 이 연구에서는 대형 언어 모델(LLMs)이 문제의 난이도를 인간의 판단과 일치하는 방식으로 내부적으로 인코딩하는지를 탐구합니다. 연구에 따르면, LLM의 활성화(activations)를 바탕으로 한 인간 유도 난이도 등급은 강력하게 선형적으로 디코딩할 수 있는 반면, LLM이 생성한 난이도 등급은 상대적으로 약합니다. 연구팀은 이러한 난이도 개념이 강화 학습(Reinforcement Learning) 동안 어떻게 발전하는지를 추적하며, 모델의 성능이 향상될수록 인간의 주석과의 정렬이 어긋나는 경향이 있음을 발견했습니다.

- **Technical Details**: 이 연구에서는 Linear Probes를 활용하여 60개의 모델 전반에 걸쳐 난이도를 추정하는 성능을 평가합니다. 실험에서는 Easy2Hard-Bench의 수학 및 코딩 하위 집합을 사용하며, 난이도 점수는 Item Response Theory (IRT) 모델링에 기반하고 있습니다. 특정 문제에 대한 활성화를 추출하여 각 모델의 레이어와 토큰 위치를 따라 탐색하고 있으며, 5-fold cross-validation 방법을 통해 결과를 검증합니다.

- **Performance Highlights**: 연구 결과, 인간 기반 난이도 레이팅은 모델 활성화에서 강하게 선형적으로 디코딩 가능하였으며, 이는 모델 크기 증가와 함께 명확한 상관 관계를 보였습니다. 또한, 쉽고 수월한 표현으로 모델을 유도할 경우, 할루시네이션(hallucination)을 줄이고 정확도를 향상시키는 것으로 나타났습니다. 마지막으로, GRPO 훈련 중 인간 난이도 탐침이 강화되는 것과 반대로 LLM 난이도 탐침은 저하되어, 인간 주석이 안정적인 난이도 신호를 제공함을 시사합니다.



### Does Reasoning Help LLM Agents Play Dungeons and Dragons? A Prompt Engineering Experimen (https://arxiv.org/abs/2510.18112)
Comments:
          Published at the Wordplay: When Language Meets Games Workshop (EMNLP 2025)

- **What's New**: 이 논문은 대형 언어 모델(LLMs)과 추론을 활용하여 던전 앤 드래곤(Dungeons & Dragons, DnD) 플레이어의 행동을 예측하고 이를 Avrae Discord 봇 명령어 형식으로 변환하는 방법을 탐구합니다. FIREBALL 데이터셋을 사용하여, 우리는 명령어 생성을 위한 추론 모델인 DeepSeek-R1-Distill-LLaMA-8B와 지시 모델인 LLaMA-3.1-8B-Instruct를 평가했습니다. 연구 결과는 모델에 특정 지침을 제공하는 것의 중요성을 강조하며, 단일 문장 변경도 모델의 출력에 큰 영향을 미칠 수 있음을 보여주고, 지시 모델이 이 작업에 충분하다는 것을 확인했습니다.

- **Technical Details**: DnD는 플레이어가 고유한 캐릭터를 창조하여 환상 세계에서 모험을 즐기는 협력 롤플레잉 게임입니다. 이 논문은 DnD에서 전투 중에 행동을 생성하는 데 초점을 맞추었으며, 이를 통해 플레이어의 특정 행동을 예측하는 LLM의 능력을 평가합니다. Avrae Discord 봇은 플레이어가 게임 내 행동을 수행하도록 명령어를 입력할 수 있도록 하여, 특정 캐릭터의 공격과 관련된 구조화된 명령어를 생성하는 데 중요한 역할을 합니다.

- **Performance Highlights**: FIREBALL 데이터셋에서 4,071개의 샘플을 사용하여 모델의 성능을 평가했습니다. 우리는 명령어 생성의 정확성과 질을 비교하기 위해 추론 모델과 지시 모델의 출력을 분석했습니다. 우리의 결과는 LLM이 DnD 플레이어 행동에 대한 구조화된 자연어 파악 및 의도 인식 학습을 통해 상호작용적 스토리텔링과 게임 자동화를 발전시키는 데 기여할 수 있음을 보여줍니다.



### Na Prática, qual IA Entende o Direito? Um Estudo Experimental com IAs Generalistas e uma IA Jurídica (https://arxiv.org/abs/2510.18108)
Comments:
          22 pages, in Portuguese language

- **What's New**: 이번 연구는 법률 분야에서의 일반 목적 AI 사용에 대한 Jusbrasil 연구를 소개하며, 법이론과 실증적 평가를 결합한 실험적 평가 프로토콜을 제안합니다. 48명의 법률 전문가와 함께 수행된 이 연구에서는 JusIA 모델이 ChatGPT Free, ChatGPT Pro 및 Gemini 시스템보다 지속적으로 뛰어난 성과를 보였습니다. 이는 도메인 특화가 신뢰할 수 있는 법적 AI 결과물에 필수적임을 강조합니다.

- **Technical Details**: 이 논문은 법률 이론 및 실제 평가를 결합한 프로토콜을 제시하여 AI 시스템의 법적 정당성을 평가하는 방법론을 구축하고자 합니다. 평가 기준으로는 정보적 정확성뿐만 아니라 규범적 정합성, 주장적 완전성 등이 포함됩니다. 또한, 법률적 사고에서의 해석적 및 주관적 요소를 중시하여 AI의 품질을 평가할 때 이러한 점들을 고려해야 함을 설명합니다.

- **Performance Highlights**: JusIA는 법률 전문가들이 수행한 작업에서 일반 목적의 AI 시스템에 비해 일관되고 높은 품질의 응답을 생성하는 것으로 나타났습니다. 연구 결과는 도메인 특화 AI 시스템이 법적 문제에 대해 더 높은 정확성과 신뢰성을 제공할 수 있음을 보여줍니다. 이러한 결과는 법률 AI의 발전을 위한 기준을 설정하는 데 기여할 것으로 기대됩니다.



### Chain-of-Thought Reasoning Improves Context-Aware Translation with Large Language Models (https://arxiv.org/abs/2510.18077)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 문장 간 의존성을 포함한 텍스트 번역 능력을 평가합니다. 영어 프랑스어 DiscEvalMT 벤치마크를 사용하여 대명사(anaphora)와 어휘적 응집(leixcal cohesion)에 대한 번역 문제를 다룹니다. CoT(Chain-of-Thought) 사고를 촉진하는 프롬프트가 LLM의 번역 성능 향상에 긍정적인 영향을 미친다는 가설을 세우고, 12개의 LLM 모델을 대상으로 실험을 수행하였습니다.

- **Technical Details**: 이 연구에 사용된 DiscEvalMT 벤치마크는 대명사(anaphora)와 어휘적 응집(lexical cohesion)에서 발생하는 번역 문제를 평가하도록 설계되었습니다. 각 테스트 항목은 두 개의 영어 문장('맥락'과 '현재')과 첫 문장의 프랑스어 번역으로 구성됩니다. LLM은 두 가지 과제에 대해 평가되었으며, 첫 번째는 가능성이 있는 잘못된 번역과 올바른 번역을 구별하는 것이고, 두 번째는 문장을 번역하여 결과를 평가하는 것입니다.

- **Performance Highlights**: 최고 성능을 보인 모델들은 이성적인 사고를 활용하여 첫 번째 작업에서 약 90%의 정확도를 달성하고, 두 번째 작업에서는 COMET 점수가 약 92%에 이릅니다. 또한, '현명한 자가 더욱 현명해진다'는 효과를 관찰하였으며, 이는 조리 있는 사고를 통해 성능 개선이 이루어졌음을 보여줍니다. 이러한 결과는 CoT 프롬프트가 대형 모델들에서만 효과적으로 작용함을 시사합니다.



### Language Models as Semantic Augmenters for Sequential Recommenders (https://arxiv.org/abs/2510.18046)
- **What's New**: LaMAR(언어 모델 보강 추천)는 LLM(대규모 언어 모델)의 추론 능력을 활용하여 사용자의 상호작용 이력을 풍부하게 만들기 위해 자동으로 생성된 의미 신호를 통합하는 새로운 프레임워크입니다. 이 시스템은 기존 메타데이터로부터 사용자의 의도와 아이템 간의 관계를 추론하여 부가적인 문맥 신호를 생성함으로써 추천 시스템의 성능을 향상시키는 데 중점을 둡니다.

- **Technical Details**: 이 프레임워크는 두 가지 주요 단계로 구성됩니다. 첫 번째 단계에서는 LLM을 통해 사용자-아이템 시퀀스를 보강하는 의미 신호를 생성하는 파이프라인을 구축하고, 두 번째 단계에서는 이러한 신호를 추천 시스템에 통합하여 LLM의 행동을 추천 목표와 일치하게 조정합니다. 예를 들어, 사용자의 과거 상호작용을 기반으로 사용자의 숨겨진 의도나 주제를 추론하여 더욱 풍부한 문맥 정보를 생성합니다.

- **Performance Highlights**: LaMAR는 여러 공공 데이터 셋에서 광범위한 실험을 통해 성능 개선을 입증하였습니다. LLM이 생성한 신호를 기존 Sequential Recommendation 모델에 통합함으로써 랭킹 메트릭에서 지속적인 성과향상을 얻었고, 생성된 신호는 높은 의미적 다양성과 독창성을 보였습니다. 이러한 방식으로 추천 시스템의 지식 기반을 확장하고 사용자 선호도를 더욱 깊게 이해할 수 있게 되었습니다.



### From Local to Global: Revisiting Structured Pruning Paradigms for Large Language Models (https://arxiv.org/abs/2510.18030)
Comments:
          16 pages, 4 figures

- **What's New**: 본 연구에서는 GISP(Global Iterative Structured Pruning)라는 새로운 접근 방식을 소개합니다. 이 방법은 레이어별 최적화가 아닌 모델 수준의 손실을 기반으로 구조적 프루닝을 수행하며, 후보 구조를 제거하여 고밀도 아키텍처를 생성합니다. GISP는 고상태에서의 정확도 유지를 위해 반복적인 일정 조정을 통해 안정된 성능을 보장하며, '한 번 잘라내고 많이 배포하기(prune-once, deploy-many)' 가능한 작업 흐름을 지원합니다.

- **Technical Details**: GISP는 주목할 만한 결과를 얻기 위해 레이어 내의 주의 헤드(attention heads)와 MLP 채널을 제거하는 사후 교육(post-training) 방법입니다.(Dense 시차로 이어지는 로스 기반 중요한 가중치를 집계하여 구조 수준에서 블록 단위 정규화를 사용합니다. 이 접근법은 초기 프루닝 보정이 불필요하며, 반복적인 절차가 전체 모델 품질을 유지하는 데 중요한 역할을 합니다. 또한, GISP는 실험을 통해 여러 모델에 대해 정량적인 개선을 보여줍니다.

- **Performance Highlights**: Llama2 및 Mistral 모델에서의 실험 결과, GISP는 WikiText-2에 대한 perplexity를 꾸준히 줄이고 다운스트림 정확도를 개선하는 것으로 나타났습니다. 특히 40-50%의 희소성(sparsity)에서 강력한 성과를 보였습니다. 또한 DeepSeek-R1-Distill-Llama-3-8B 모델에 대해, 작업 정렬 보정(task-aligned calibration)은 정확한 일치 정확도를 대폭 향상시키는 데 기여하였습니다.



### Is Multilingual LLM Watermarking Truly Multilingual? A Simple Back-Translation Solution (https://arxiv.org/abs/2510.18019)
- **What's New**: 이번 연구는 다국어 워터마킹(multilingual watermarking)의 한계와 문제점을 지적하고, 기존 방법들이 고자원(high-resource) 언어에만 한정되어 평가되고 있음을 보여줍니다. 특히, 기존의 기술들이 매체와 저자원 언어에서 번역 공격에 대한 강인성을 유지하지 못한다는 점을 강조합니다. 이를 극복하기 위해 STEAM이라는 새로운 감지 방법을 제안했습니다.

- **Technical Details**: STEAM은 역번역(back-translation) 기반의 감지 방법으로, 번역 과정에서 손실된 워터마크 강도를 복원합니다. 이 방법은 모든 워터마킹 기술과 호환되며, 다양한 토크나이저(tokenizer) 및 언어에서도 강인성을 유지합니다. 비침투적(non-invasive)이고 새로운 언어로 쉽게 확장할 수 있는 장점이 있습니다.

- **Performance Highlights**: 17개 언어에서 평균 +0.19 AUC 및 +40%p TPR@1%의 성능 향상을 보였습니다. STEAM은 다양한 언어에서 더욱 공정한 워터마킹을 위한 간단하고 강력한 경로를 제공합니다. 이러한 성과는 다국어 워터마킹의 혁신적 접근 방식을 제시합니다.



### SimBA: Simplifying Benchmark Analysis Using Performance Matrices Alon (https://arxiv.org/abs/2510.17998)
Comments:
          EMNLP 2025 Findings

- **What's New**: 이번 논문에서는 SimBA라는 새로운 프레임워크를 제안하여 언어 모델(LM) 평가에 대한 통찰력을 제공합니다. SimBA는 데이터셋과 모델의 관계를 분석하는 'Stalk', 대표적인 데이터셋 하위 집합을 발견하는 'Prowl', 그리고 모델의 성능을 예측하는 'Pounce'의 3단계로 구성됩니다. 이 프레임워크는 기존 평가 방법의 한계를 극복하고 더 정교한 분석을 가능하게 합니다.

- **Technical Details**: SimBA 프레임워크의 첫 단계인 Stalk에서는 평가 행렬을 사용하여 데이터셋 간의 관계를 분석합니다. 각 데이터셋에 대해 성능 수치를 비교하고, 다변량 선형 회귀 모형을 통해 이들 간의 관계를 정량화합니다. 두 번째 단계인 Prowl에서는 모델 성능 패턴을 기반으로 중복 데이터셋을 식별하여 대표적인 데이터셋 하위 집합을 만들어 냅니다.

- **Performance Highlights**: SimBA를 사용하여 HELM, MMLU, BigBenchLite의 벤치마크에서, 데이터셋의 6.25%, 1.7%, 28.4%만으로도 95% 이상의 커버리지를 달성할 수 있음을 보여주었습니다. 또한, 이 대표 하위 집합만으로도 모델의 순위를 유지하고, 모델의 성능을 예측할 수 있으며, 평균 제곱 오차가 거의 0에 가까운 결과를 얻을 수 있었습니다.



### Believe It or Not: How Deeply do LLMs Believe Implanted Facts? (https://arxiv.org/abs/2510.17941)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)에 새로운 사실 지식을 심어주는 지식 편집 기술에 대한 평가 프레임워크를 개발하였습니다. 연구팀은 'belief depth'라는 개념을 도입하여, 심어진 지식이 얼마나 잘 일반화되고, 자기 분석과 직접적인 도전에 얼마나 강하며, 진정한 지식과 유사하게 표현되는지를 측정합니다. 이 작업은 지식 편집 기술의 성공을 평가하는 데 필요한 새로운 기준을 제시합니다.

- **Technical Details**: 연구진은 심어진 지식의 'belief depth'를 1) 관련 맥락으로의 일반화 정도, 2) 자기 검증과 직접적인 도전에 대한 강인성, 3) 진정한 지식 표현의 유사성으로 operationalize(운영화)하였습니다. 평가 결과 간단한 프롬프트와 기계적 편집 기술로는 충분한 깊이의 지식을 심어주기 어렵다는 사실이 밝혀졌습니다. 반면에, Synthetic Document Finetuning(SDF)은 사실에 일치하는 LLM 생성 문서로 모델을 훈련시켜 믿음을 성공적으로 심어주는 경우가 많습니다.

- **Performance Highlights**: SDF는 일반적으로 신뢰할 수 있는 방식으로 믿음을 심어주지만, 일부 경우에는 기본적인 세계 지식과 모순되는 믿음이 취약하고 진정한 지식과는 다른 표현을 보여주기도 합니다. 이 연구는 'belief depth'라는 측정 기준을 도입하여 지식 편집 기술의 실제 적용에 필요한 엄밀한 평가를 가능하게 합니다. 결론적으로, 이 연구는 지식 편집의 실질적인 성공을 평가하기 위한 중요한 기초 데이터를 제공합니다.



### AtlasKV: Augmenting LLMs with Billion-Scale Knowledge Graphs in 20GB VRAM (https://arxiv.org/abs/2510.17934)
- **What's New**: 이 논문은 LLMs (대규모 언어 모델)에 대해 외부 지식을 통합하는 새로운 파라메트릭 방법인 AtlasKV를 제안합니다. AtlasKV는 기가바이트(GiB) 단위의 GPU 메모리 비용으로 억 규모의 지식 그래프(KG, Knowledge Graph)를 통합할 수 있으며, 기존 RAG 방법이 갖고 있던 시간 지연(inference latency) 문제를 해결합니다. 이 방법은 KG2KV와 HiKVP라는 혁신적인 설계를 도입하여 LLM에 효과적으로 KG 트리플을 통합합니다.

- **Technical Details**: AtlasKV는 대규모 KG에서 쿼리-키-값(Q-K-V) 데이터로 자연스럽게 변환하는 KG2KV 파이프라인을 소개합니다. 또한, HiKVP 알고리즘을 통해 컴퓨팅 및 메모리 오버헤드를 대폭 줄이면서 높은 지식 기반 정확도를 유지할 수 있습니다. 이러한 설계는 LLM의 주의 메커니즘을 활용하여 효과적으로 지식을 통합하도록 돕습니다.

- **Performance Highlights**: 실험 결과, AtlasKV는 ICL, KBLaM 및 RAG 방법에 비해 우수한 효율성과 확장성을 보여주며, 각 구성 요소의 기여를 검증하기 위한 포괄적인 압축 연구(ablation studies)를 진행했습니다. 일반화 성능도 뛰어나며, 외부 지식에 대한 적응력이 뛰어난 것으로 나타났습니다.



### Diagnosing Representation Dynamics in NER Model Extension (https://arxiv.org/abs/2510.17930)
- **What's New**: 이 논문은 Named Entity Recognition (NER) 모델을 새로운 PII(개인 식별 정보) 엔티티로 확장하는 방법에 대한 새로운 접근법을 제시합니다. BERT 모델을 기존의 표준 의미론적 엔티티(PER, LOC, ORG)와 새로운 패턴 기반 PII(EMAIL, PHONE)에 대해 공동으로 미세 조정함으로써, 원래의 클래스에 대한 최소한의 퇴화를 실현하였습니다. 이는 NER 모델의 적응에 대한 새로운 메커니즘적 진단을 제공합니다.

- **Technical Details**: 연구에서는 의미적 드리프트(semantic drift)를 측정하기 위해 점진적 학습(incremental learning) 설정을 진단 도구로 사용합니다. 그 결과, LOC(위치) 엔티티가 새로운 PII와의 표현 중복(representation overlap)으로 인해 고유하게 취약함을 발견하였고, 'O' 태그의 분류기(classifier)의 동결 해제가 필요함을 입증하였습니다. 이 과정에서 모델은 PII 패턴을 'O'로 매핑하도록 훈련되지만 새로운 학습을 차단하는 '역 O-태그 표현 드리프트(reverse O-tag representation drift)'가 발생합니다.

- **Performance Highlights**: 이 연구는 NER 모델이 독립적인 의미적/형태적(feature semantic vs. morphological) 특성 메커니즘을 사용한다고 가정합니다. 그에 따라, 'O' 태그의 유연성(plasticity)을 통해 배경 클래스를 적응시켜 새로운 패턴을 수용할 수 있는 방안을 제시합니다. 또한, 이러한 특징 독립성, 표현 중복 및 'O' 태그의 유연성은 미래의 모델 개선에 중요한 통찰력을 제공할 것입니다.



### Efficient Toxicity Detection in Gaming Chats: A Comparative Study of Embeddings, Fine-Tuned Transformers and LLMs (https://arxiv.org/abs/2510.17924)
Comments:
          Published in the Journal of Data Mining & Digital Humanities (JDMDH), special issue NLP4DH

- **What's New**: 이번 연구는 온라인 게임 채팅에서 자동화된 독성 탐지를 위한 자연어 처리(NLP) 방법에 대한 포괄적인 비교 분석을 제시합니다. 전통적인 기계 학습 모델, 대형 언어 모델(LLMs), 세분화된 트랜스포머 모델 및 검색 증강 생성(RAG) 접근 방식이 평가됩니다. 평가 프레임워크는 분류 정확도, 처리 속도 및 계산 비용의 세 가지 핵심 차원을 측정하며, 인적 조정자의 작업 부담을 최적화하는 하이브리드 조정 시스템 아키텍처를 제안합니다.

- **Technical Details**: 연구는 정적인 임베딩 및 맥락적 임베딩을 기반으로 한 전통적인 기계 학습 분류기에서부터 현대적인 LLM 및 RAG에 이르는 여러 NLP 기법의 조합을 분석합니다. 이 연구의 목표는 정확성, 속도 및 자원 효율성과 같은 성능 절충안을 정량화하고 공정성 또는 정밀성을 훼손하지 않으면서 인적 작업 부담을 최소화할 수 있는 하이브리드 디자인을 식별하는 것입니다. 경험적 결과는 DistilBERT가 최적의 정확도-비용 거래를 달성하는 등 방법 간의 성능 차이를 보여줍니다.

- **Performance Highlights**: 본 연구는 자동화된 독성 탐지 시스템이 비용 효율적이고 효율적인 콘텐츠 조정 시스템을 배치할 수 있는 근거를 제공합니다. 세 가지 주요 차원에서의 비교를 통해, 세분화된 트랜스포머 모델이 가장 높은 정확성을 기록하며, 이는 게임 환경의 동적 특성에 적합한 조정 솔루션을 제공함을示합니다. 향후 연구는 이러한 자동화된 시스템을 진화하는 온라인 게임 커뮤니티에 통합하고, 인적 조정자들이 보다 복잡한 결정에 집중할 수 있도록 하는 데 기여할 것입니다.



### Select-Then-Decompose: From Empirical Analysis to Adaptive Selection Strategy for Task Decomposition in Large Language Models (https://arxiv.org/abs/2510.17922)
Comments:
          Accepted to the Main Conference of EMNLP 2025 (Oral)

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 작업 분해(task decomposition)에 대한 포괄적인 조사를 실시하고, 여섯 가지 범주화 체계를 제시하였습니다. 기존의 연구들은 주로 메모리, 도구 사용, 피드백 메커니즘에 중점을 두었지만, 성능과 비용 간의 균형을 간과하는 경향이 있었습니다. 이를 통해 Select-Then-Decompose 전략을 제안하며, 이는 선택, 실행, 검증의 세 단계로 구성된 폐쇄 루프 문제 해결 프로세스를 구축합니다.

- **Technical Details**: 연구팀은 LLM의 작업 분해에서 성능과 비용에 영향을 미치는 세 가지 주요 요소를 식별하였습니다: 작업 분해 접근 방식의 범주, 작업의 특성, 분해 모델과 실행 모델의 구성입니다. 이 연구에서는 작업 분해 과정을 통해 발생할 수 있는 성능-비용 딜레마에 대한 통찰력을 제공하고, 계층적 구조와 같은 몇 가지 토폴로지 구조에 대해 설명합니다. Select-Then-Decompose 전략은 주어진 작업의 특성에 따라 최적의 분해 접근 방법을 동적으로 선택하고, 이러한 결과의 신뢰성을 향상시키기 위한 검증 모듈을 갖추고 있습니다.

- **Performance Highlights**: 다양한 벤치마크에서의 포괄적인 평가 결과, Select-Then-Decompose 전략은 성능과 비용 간의 최적 균형을 달성하며 Pareto 경계에서 일관된 성능을 보여주었습니다. 연구는 또한 성능을 향상시키기 위해 실행 모델을 조정하는 것이 분해 모델을 조정하는 것보다 더 큰 성과 향상을 이끈다는 점을 강조합니다. 전반적으로 본 연구는 LLM의 작업 분해 성능을 높이는 데 기여하는 일련의 실용적인 원칙을 제공합니다.



### CLAWS:Creativity detection for LLM-generated solutions using Attention Window of Sections (https://arxiv.org/abs/2510.17921)
Comments:
          NeurIPS 2025

- **What's New**: 최근 대규모 언어 모델(LLMs)의 추론 능력을 향상시키기 위한 연구에서 괄목할만한 성과를 이뤘습니다. 이 논문은 LLM이 수학과 코딩과 같은 도전적인 작업에서 강력한 성능을 발휘할 수 있도록 강화 학습(RL)으로 훈련된 새로운 방법 CLAWS를 제안합니다. CLAWS는 인간 평가 없이 수학적 솔루션을 전형적, 창의적, 홀로리테이션된 범주로 분류할 수 있는 방법론을 제공합니다.

- **Technical Details**: 이 연구는 LLM의 내부 표현으로부터 특성을 추출하여 홀로리테이션(Hallucinated), 창의적(Creative), 전형적(Typical) 솔루션을 분류하는 실험 프레임워크를 제시합니다. 입력 프롬프트는 가이드라인(GG), 문제(PP), 참조 솔루션(SS), 지침(II)의 네 부분으로 나뉘며, 각 부분의 주목(attention) 분석을 통해 분류가 수행됩니다. 상대적으로 작은 모델 크기로도 효과적으로 성능을 나타낼 수 있도록 설계되었습니다.

- **Performance Highlights**: CLAWS는 7-8B 매개변수 범위의 다섯 개 수학 RL 모델을 활용하여 기존의 다섯 가지 화이트 박스 탐지 방법들을 초월하는 성능을 보여줍니다. 특히, 연구에 사용된 4,545개의 수학 문제를 통해 생성된 솔루션들은 모두 높은 효율성으로 분류될 수 있었으며, 여기서 CLAWS는 일관되게 기준 방법들보다 우수한 성능을 기록했습니다. 이는 수학적 추론 작업에서 창의성 평가의 새로운 기준을 마련할 수 있는 가능성을 보여줍니다.



### JT-Safe: Intrinsically Enhancing the Safety and Trustworthiness of LLMs (https://arxiv.org/abs/2510.17918)
- **What's New**: 본 논문은 대규모 언어 모델(LLM)의 비정확성과 신뢰성 문제를 해결하기 위한 새로운 접근 방식을 제안합니다. 특히, 사전 학습(pre-training) 데이터의 품질을 향상시켜 LLM의 안전성과 신뢰성을 높이는 데 초점을 맞추고 있습니다. 연구진은 현실 세계의 맥락을 보강한 새로운 데이터 세트(DWC: Data with World Context)를 도입하여, 자원 데이터의 불일치와 오류를 줄이려는 노력을 하고 있습니다.

- **Technical Details**: 현재 LLM의 사전 학습 데이터는 15조에서 40조 개의 토큰(token)으로 구획되어 있으며, 인터넷 원시 데이터, 생성된 데이터 및 전문 분야의 데이터 등 다양합니다. 기존 데이터 가공 파이프라인을 기반으로 모델과 규칙, 빅데이터 처리 유틸리티를 활용하여 데이터의 품질을 더욱 높이는 방법을 제안했습니다. 이를 통해 사전 학습 데이터는 실세계 맥락 정보를 포함할 수 있도록 보강되어 불확실성을 줄이는 데 도움을 줍니다.

- **Performance Highlights**: JT-Safe-35B 모델은 6.2 조 개의 토큰을 사용하여 안전성 및 신뢰성 벤치마크에서 평균 1.79%의 성능 향상을 달성했습니다. 특히 JT-Safe-35B는 산업 관련 데이터 양을 더욱 확대하며 성능을 극대화하였습니다. 이러한 개선은 실제 사용자의 경험과 벤치마크 결과에서도 긍정적인 영향을 미쳤습니다.



### Atomic Literary Styling: Mechanistic Manipulation of Prose Generation in Neural Language Models (https://arxiv.org/abs/2510.17909)
Comments:
          12 pages, 3 figures, 4 tables

- **What's New**: 이번 연구에서는 GPT-2의 문학 스타일에 대한 기계적 분석을 제시하여, 모범적인 산문과 인공지능 생성 텍스트를 구분하는 개별 뉴런을 식별합니다. Herman Melville의 'Bartleby, the Scrivener'를 자료로 사용하여, 355백만 개 매개변수와 32,768 뉴런에서 활성화 패턴을 추출했습니다. 27,122개 통계적으로 유의미한 판별 뉴런을 발견했으며, 제거 시 문학적 품질이 향상되는 역설적인 결과를 도출했습니다.

- **Technical Details**: GPT-2 Medium(355M parameters)의 8개 후반 층(layer)을 분석하여, 27,122개의 통계적으로 유의한 판별 뉴런을 식별했습니다. 각 뉴런의 활성화는 특정 개념을 인코딩하며, 이 연구는 관찰적 상관관계가 인과적 필요성( causality gap )과 구별되어야 함을 보여줍니다. 또한, 활성화 패치 및 뉴런 차단 실험을 통해 문학적 생성에 영향을 줄 수 있는 특정 뉴런을 확인했습니다.

- **Performance Highlights**: 50개의 고판별 뉴런을 제거했을 때, 문학 스타일 지표에서 25.7% 향상된 결과를 보였습니다. 이 연구는 문학적 텍스트 분석 과정에서 활성화되는 뉴런들이 생성 시에는 오히려 문학적 품질을 방해한다는 점을 강조하며, 기계적 해석 가능성 연구 및 AI 정렬에 대한 새로운 시사점을 제시하고 있습니다. 이러한 결과는 스타일 전이(style transfer) 및 텍스트 생성을 위한 모델 개선에 중요한 영향을 미칩니다.



### Advances in Pre-trained Language Models for Domain-Specific Text Classification: A Systematic Review (https://arxiv.org/abs/2510.17892)
Comments:
          41 pages, 10 figures, 13 tables

- **What's New**: 이번 논문은 NLP의 발전과 최신 LLM(large language models)이 도메인 특화 텍스트 분류에서 겪는 문제를 다룹니다. 저자들은 2018년부터 2024년 1월까지 발표된 41개의 연구를 체계적으로 검토하여 PSML(pre-trained language models)이 도메인 특화 텍스트 분류에 어떻게 활용되는지를 조사합니다. 이 과정에서 전통적인 방법과 현대적인 방법의 차이를 강조하며, 특히 transformer 기반 모델에 초점을 맞춥니다.

- **Technical Details**: 자연어 처리(NLP)는 텍스트 분류와 같은 다양한 작업에서 중요한 역할을 하며，LLMs는 일반적으로 대규모 코퍼스에서 훈련된 후 특정 작업에 대해 미세 조정된 형태로 사용됩니다. 하지만 도메인 특화 텍스트는 전문 용어와 고유한 문법 구조, 불균형한 데이터 분포 등으로 인해 LLMs의 성능에 도전 과제를 제공합니다. 논문은 PLM을 사용하는 연구들을 분류하고, 이를 다루는 기술의 세분화를 제안합니다.

- **Performance Highlights**: BERT, SciBERT, BioBERT와 같은 모델을 대상으로 한 비교 실험을 통해 생물 의학 문장 분류에서의 성능을 분석하였고, 다양한 도메인에서 LLMs의 성능을 비교해 구체적인 강점과 약점을 밝혔습니다. 이 연구는 향후 연구 방향을 제안하며, 도메인 특화 테크닉에서 발생하는 한계와 미래의 연구 필요성을 강조합니다.



### POPI: Personalizing LLMs via Optimized Natural Language Preference Inferenc (https://arxiv.org/abs/2510.17881)
- **What's New**: POPI 프레임워크는 heterogeneous user signals를 간결한 자연어 요약으로 전환하는 preference inference 모델을 도입하고, 이를 활용하여 사용자 맞춤 응답을 생성하는 접근 방식을 제안합니다. 기존의 방법들이 individual variation을 간과하고 population-level averages에 집중했던 것과 달리, POPI는 모델이 사용자 개별의 preferences를 효과적으로 반영할 수 있도록 최적화됩니다.

- **Technical Details**: POPI는 preference inference(선호 추론)와 personalized generation(개인화 생성)을 통합하여 최적화하는 접근 방식을 사용합니다. 기존의 사용자별 파인튜닝 방식은 연산적으로 비효율적이지만, POPI는 preference inference LLM의 학습을 통해 사용자 신호를 간결한 요약으로 만듭니다. 이 요약은 공통의 generation 모델에서 조건으로 사용되어, 사용자 맞춤형 출력을 생성하는 데 필요한 context overhead를 대폭 줄입니다.

- **Performance Highlights**: POPI를 사용한 다양한 실험 결과는 개인화 정확도가 일관되게 향상되었으며, context overhead가 크게 감소했음을 보여줍니다. 또한, 최적화된 요약은 상용 LLM에 쉽게 적용될 수 있으며, 파라미터 업데이트 없이도 plug-and-play 개인화가 가능하다는 점에서 실용성을 갖추고 있습니다. 전체적으로 POPI는 사용자 맞춤형 모델 조정의 새로운 기준을 수립하는 데 기여합니다.



### Outraged AI: Large language models prioritise emotion over cost in fairness enforcemen (https://arxiv.org/abs/2510.17880)
- **What's New**: 이번 연구에서는 감정이 인간의 의사결정에 미치는 영향과 대규모 언어 모델(LLMs)의 도덕적 판단 과정에 대한 비교를 진행했습니다. LLMs가 감정을 어떻게 사용하는지에 대한 최초의 인과적 증거를 제공하며, 감정이 도덕적 결정에 미치는 영향을 평가했습니다. 연구는 4,068개의 LLM 에이전트와 1,159명의 성인을 대상으로 총 796,100번의 결정에서 이루어졌습니다.

- **Technical Details**: 연구에서는 알트루이즘(altruism)에 기반한 제3자 처벌(third-party punishment) 과제를 사용했습니다. 연구 결과, LLMs는 불공정함에 대한 부정적 감정을 더 강하게 경험했고, 이는 더 많은 처벌을 초래했습니다. LLM은 비용(cost)을 우선시하기 보다는 감정을 먼저 고려하는 경향을 보였고, 그 결과 인간보다 거의 모두 또는 전무에 가까운 방식으로 규범(norm)을 적용했습니다.

- **Performance Highlights**: 흥미롭게도, o3-mini와 DeepSeek-R1 같은 추론 모델들은 비용 감수성이 더 높고 인간 행동과 더 가까운 결과를 보였지만, 여전히 감정에 크게 영향을 받았습니다. 이 연구는 LLMs가 인간과 유사한 감정 지능(emotional intelligence)을 달성하기 위한 발전적 경로를 제안하며, 미래 모델은 감정과 맥락에 민감한 추론을 통합해야 한다고 강조합니다.



### Modeling Layered Consciousness with Multi-Agent Large Language Models (https://arxiv.org/abs/2510.17844)
Comments:
          20 pages, 4 figures, accepted for presentation at EMNLP 2025 Workshop on Active and Passive LLM Personalization (PALS) OpenReview: this https URL

- **What's New**: 이번 연구에서는 대규모 언어 모델(LLM)의 인공지능 의식을 모형화하기 위한 다중 에이전트 프레임워크를 제안합니다. 이는 정신분석 이론(psychoanalytic theory)에 기반하고 있으며, 자기 인식(self-awareness), 전의식(preconsciousness), 무의식(unconsciousness)을 에이전트 상호작용을 통해 시뮬레이션합니다.

- **Technical Details**: 우리의 모델인 Psychodynamic Model은 고정된 특성과 동적인 필요를 결합한 개인화 모듈(Personalization Module)을 통해 가이드됩니다. 감정이 풍부한 대화체에 대한 매개변수 효율적 파인튜닝(parameter-efficient fine-tuning)을 사용하여 시스템을 평가하였고, 총 8가지 개인 맞춤형 조건에서 실험하였습니다.

- **Performance Highlights**: 모델 평가 결과, LLM을 판별자로 사용했을 때, 파인튜닝된 모델이 71.2%의 선호도를 보여주었습니다. 이 모델은 감정적 깊이가 향상되고 출력의 변동성이 줄어들어 개인화된 인지(cognition)를 위한 적응 가능성을 입증하였습니다.



### Grasp Any Region: Towards Precise, Contextual Pixel Understanding for Multimodal LLMs (https://arxiv.org/abs/2510.18876)
- **What's New**: 이 논문에서는 Grasp Any Region (GAR)을 제안하여 멀티모달 대형 언어 모델(MLLM)의 지역 이해 능력을 개선하고자 하였습니다. 기존의 지역 MLLMs가 중요 글로벌 컨텍스트(global context)를 간과했던 점을 보완하며, GAR는 지역 수준의 시각적 이해를 위한 종합적인 접근법을 제공합니다. GAR는 RoI-aligned feature replay 기법을 활용해 개별 지역의 정확한 인식과 여러 프롬프트 간의 상호작용 모델링을 지원합니다.

- **Technical Details**: GAR은 특정 프롬프트와 함께 전체 이미지 정보를 인코딩하는 방법으로, 이를 통해 지역별 세부 정보를 정확하게 캡처하는 능력을 향상시킵니다. 이러한 방식은 RoI-Align을 통해 글로벌 Feature map에서 관련된 Feature를 수집하여 지역과 글로벌 정보를 동시에 고려할 수 있게 합니다. 또한, GAR-Bench를 도입하여 여러 지역의 상호작용과 복잡한 추론 과정을 평가하는 새로운 벤치마크를 제공합니다.

- **Performance Highlights**: 실험 결과 GAR-1B 모델은 DAM-3B 및 PAM-3B보다 상세한 캡션 생성에서 우수한 성능을 보여주며, 다중 프롬프트 모델링에서 InternVL3-78B를 초월하는 성과를 달성했습니다. GAR-8B는 VideoRefer-BenchQ에서 인도메인 모델인 VideoRefer-7B를 초과하는 성능을 발휘하여 동영상에 대한 이해 능력 또한 강화되었습니다. 이러한 결과는 GAR의 강력한 커뮤니케이션 및 이해 능력을 입증합니다.



### Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting (https://arxiv.org/abs/2510.18874)
- **What's New**: 이 논문은 언어 모델(언어 모델, LMs)의 사후 훈련(post-training) 과정에서 발생하는 재앙적 망각(catastrophic forgetting) 현상을 완화하는 방법에 대한 가이드를 제시합니다. 연구에서는 감독 세부 조정(supervised fine-tuning, SFT) 및 강화 학습(reinforcement learning, RL)이라는 두 가지 일반적인 사후 훈련 방법의 망각 패턴을 비교하고 있습니다. 실험 결과, RL 방법이 SFT에 비해 망각을 덜 발생시키면서도 유사하거나 더 높은 성과를 나타내는 경향이 있음을 발견했습니다.

- **Technical Details**: 본 연구에서는 Qwen 2.5 및 Llama 3 모델을 사용하여 세 가지 과제(지시 따르기, 일반 지식, 산술 추론)에 대한 SFT와 RL의 망각 패턴을 비교했습니다. RL의 모드 탐색(mode-seeking) 성질이 과거 지식을 온전히 유지할 수 있게 하며, 역 KL(KL divergence) 최소화를 통해 목표 과제를 학습할 때 망각에 더 강한 모습을 보입니다. 특히 RL은 온-정책 데이터(on-policy data)를 활용함으로써 망각 방지의 강건성을 가지며, 이러한 기법은 기존의 SFT보다 더 효과적인 결과를 제공합니다.

- **Performance Highlights**: 실험 결과 RL이 각종 과제에서 SFT보다 잃어버린 정보를 덜 발생시키며, 사후 훈련 후에도 높은 정확도를 유지하는 것으로 나타났습니다. 본 연구에서 제안한 방법으로는 초기 정책에 의해 생성된 비슷한 온-정책 데이터를 활용하여 망각을 줄이는데 성공했습니다. 이러한 연구 결과는 기존 능력을 해치지 않으면서도 목표 과제를 효과적으로 수행할 수 있는 실용적인 기법을 제시합니다.



### See the Text: From Tokenization to Visual Reading (https://arxiv.org/abs/2510.18840)
- **What's New**: 이 논문은 새로운 비전 중심의 토큰화 방법인 SeeTok을 선보입니다. SeeTok은 텍스트를 이미지로 변환하고, pretrained multimodal LLMs를 사용하여 이를 해석하는 방식으로, 전통적인 subword tokenization보다 더 효율적입니다. 이 접근 방식은 특히 저자원 언어에 대한 결함을 최소화하고, 토큰 수를 크게 줄이며, 계산 복잡성을 낮추는 장점을 가지고 있습니다.

- **Technical Details**: SeeTok의 기술적 기반은 문자를 시각적 패턴으로 처리하는 것입니다. 이러한 방식은 Visual Word Form Area (VWFA)와 같은 인간의 인지 메커니즘에서 영감을 받아서 개발되었습니다. 텍스트를 이미지로 변환한 후 pretrained MLLMs의 시각적 인코더를 사용하여 텍스트 표현을 추출하고, 이를 LLM 모델에 전달하여 처리합니다. 이를 통해 저비용 수정과 일반적인 학습 방법 없이 시각 텍스트 지침을 그대로 해석할 수 있습니다.

- **Performance Highlights**: SeeTok은 세 가지 자연어 처리 작업에서 전통적인 텍스트 토크나이저와 유사한 성능을 보이며, 4.43배 적은 비주얼 토큰과 70.5%의 FLOP 감소를 기록했습니다. 다국어 번역 작업에서는 13개 언어를 아우르며, 텍스트 토크나이저 대비 더 우수한 교차 언어 전이능력을 나타내며, 86% 낮은 fertility를 기록했습니다. 또한 문자 수준, 단어 수준, 시각 수준 공격에 대해 강력한 내구성을 보이며 성능 저하가 적다는 특징이 있습니다.



### Zero-Shot Vehicle Model Recognition via Text-Based Retrieval-Augmented Generation (https://arxiv.org/abs/2510.18502)
Comments:
          Accepted by The 38th Conference of Open Innovations Association FRUCT, 2025

- **What's New**: 이 논문에서는 최신 차량 모델 인식에서 기존 모델들이 새로운 모델에 적응하는데 어려움을 겪는 문제를 해결하기 위해 새로운 접근 방식을 제안합니다. Contrastive Language-Image Pretraining (CLIP) 모델의 한계를 극복하기 위해, Retrieval-Augmented Generation (RAG)와 비전 언어 모델(Vision Language Models, VLMs)을 통합한 제로샷(Zero-shot) 인식 파이프라인을 개발하였습니다. 이 시스템은 차량 이미지를 텍스트로 변환하고, 텍스트 기반의 추론을 통해 차량의 메이크(make)와 모델(model)을 식별할 수 있도록 설계되었습니다.

- **Technical Details**: 제로샷 차량 모델 인식 방식을 통해 입력 이미지를 처리하고 가장 가능성 높은 레이블을 예측하는 새로운 메커니즘을 소개합니다. 이 과정에서는 비전-언어 인코더(Ev)와 텍스트 데이터베이스를 비교하여 관련 정보를 검색하는 단계가 포함됩니다. RAG의 틀은 외부 지식에 기반한 추론을 가능케 하며, 새로운 차량 모델에 대한 텍스트 설명 업데이트를 통해 시스템의 확장성을 보장합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 CLIP의 기본 성능을 기준으로 차량 인식을 거의 20% 향상시키는 것으로 나타났습니다. 연구에 사용된 데이터셋은 최근 출시된 차량 모델로 구성되어 있어, 진정한 제로샷 평가 시나리오를 제공합니다. 이 방식은 새로운 차량 모델의 효과적인 인식을 가능하게 하며, 지능형 교통 시스템을 위한 실제 적용 가능성을 입증하고 있습니다.



### Probabilistic Modeling of Intentions in Socially Intelligent LLM Agents (https://arxiv.org/abs/2510.18476)
- **What's New**: 이번 연구에서는 다중 턴 사회 대화에서 대형 언어 모델(LLM) 에이전트를 위한 확률적 의도 모델링 프레임워크를 제안합니다. 이 프레임워크는 파트너의 잠재적 의도에 대한 신념 분포(belief distribution)를 유지하며, 맥락적 사전(prior)에서 초기화되고 발화 후 우도 추정을 통해 동적으로 업데이트됩니다. 점진적으로 진화하는 이 신념 분포는 정책의 추가적인 맥락적 기반을 제공하여 불확실성 하에서 적응형 대화 전략을 가능하게 합니다.

- **Technical Details**: 이 연구는 부분 가시 마르코프 결정 프로세스(POMDP)로 모델링된 2-agent 대화 사회 상호작용을 연구합니다. 여기서 각 상태는 사회적 맥락과 파트너의 의도에 대한 신념 분포를 포함하는 확장된 상태 공간으로 구성됩니다. 프레임워크는 세 가지 핵심 구성 요소인 의도 모델, 우도 모델 및 확신 인지 행동 정책을 포함하여, 이들을 통해 에이전트의 행동 결정을 돕습니다.

- **Performance Highlights**: 예비 실험 결과, SOTOPIA-All에서 전체 점수가 9.0% 향상되었고, SOTOPIA-Hard에서는 4.1% 향상된 결과를 보여주었습니다. 제안된 프레임워크는 파트너의 의도를 직접 관찰하는 오라클 에이전트보다 약간 뛰어난 성능을 보였습니다. 이러한 초기 결과는 확률적 의도 모델링이 사회적으로 지능을 갖춘 LLM 에이전트 개발에 기여할 수 있음을 시사합니다.



### CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignmen (https://arxiv.org/abs/2510.18471)
- **What's New**: 이 논문에서는 코드 생성의 새로운 접근 방식인 CodeRL+를 제안합니다. CodeRL+는 Reinforcement Learning with Verifiable Rewards (RLVR) 체계에서 실행 의미의 정렬을 통합하여 텍스트 표현과 실행 의미 간의 격차를 줄입니다. 이를 통해 모델이 변수 수준의 실행 경로를 추론할 수 있게 하여 실행 의미의 직접적인 학습 신호를 제공합니다.

- **Technical Details**: CodeRL+는 코드 생성과 실행 의미 정렬을 병렬적으로 최적화하도록 설계되었습니다. 이는 실패한 탐색 프로그램을 재사용하여 변수들이 프로그램 실행 동안 어떻게 전파되는지를 분석함으로써 코드의 기능적 행동과 텍스트 형식을 명확히 정렬하는 방식을 채택합니다. 이 방법은 추가 데이터 소스 없이도 기존의 RL 알고리즘과 원활하게 통합될 수 있습니다.

- **Performance Highlights**: 실험 결과, CodeRL+는 RLVR 및 Distillation을 포함한 기존 모델에 비해 평균 4.6%의 개선 효과를 보여주었습니다. 특히 코드 추론 및 테스트 출력 생성 벤치마크에서 각각 15.5% 및 4.4% 높은 정확성을 기록하며 뛰어난 일반화 성능을 보여주었습니다. CodeRL+는 다양한 RL 알고리즘 및 LLM에 대한 강력한 적용 가능성을 보여줍니다.



### Position: LLM Watermarking Should Align Stakeholders' Incentives for Practical Adoption (https://arxiv.org/abs/2510.18333)
- **What's New**: 최근 대규모 언어 모델(LLM)의 워터마킹 알고리즘이 발전했지만, 실제 사용은 여전히 제한적입니다. 이 논문에서는 LLM 제공자, 플랫폼, 최종 사용자 간의 보상 불일치가 주요 요인임을 주장하며, 경쟁 리스크, 탐지 도구의 관리, 강건성 문제, 귀속 문제와 같은 네 가지 주요 장벽을 강조합니다. 세 가지 워터마킹 방식이 어떻게 이러한 문제를 다룰 수 있는지 다시 살펴봅니다.

- **Technical Details**: 모델 워터마킹은 LLM 제공자의 이익과 자연스럽게 일치하지만, 오픈 소스 생태계에서 새로운 도전에 직면하게 됩니다. LLM 텍스트 워터마킹은 오용 방지 도구로서 제공자의 이익을 제한적으로 제공하며, 데이터를 정제하거나 사용자가 감시할 수 있는 프로베넌스를 개선하는 특정 설정에서 사용될 수 있습니다. 문맥 내 워터마킹(ICW)은 신뢰받는 당사자와 함께 숨겨진 워터마킹 지침을 문서에 삽입하는 방식으로, 이는 오용을 탐지할 수 있는 신뢰성 있는 도구를 제공합니다.

- **Performance Highlights**: 현재 LLM 텍스트 워터마킹 시스템의 실제 사용은 제한적입니다. Google의 SynthID만이 Gemini 웹 및 모바일에서 실질적으로 배포되었습니다. 경쟁 우려와 탐지 도구의 관리 문제, 강건성 불안 등 네 가지 주요 장애물이 실제 사용을 방해하고 있습니다. ICW 같은 방법은 신뢰받는 당사자에게 워터마킹 제어를 제공함으로써, LLM 제공자의 보상 체계를 일치시킴으로써 널리 사용될 가능성을 가지고 있습니다.



### The Impact of Image Resolution on Biomedical Multimodal Large Language Models (https://arxiv.org/abs/2510.18304)
Comments:
          Proceedings of the 10th Machine Learning for Healthcare Conference, PMLR 298, 2025

- **What's New**: 이번 연구는 생물의학 이미징 기술이 고해상도 이미지를 분석하는 데 필수적임을 강조합니다. 특히, 기존의 다중모드 대형 언어 모델(MLLMs)이 저해상도 이미지에 맞춰 설계되어 있어 생물의학 이미지의 중요한 정보가 손실될 위험이 크다는 점을 지적합니다. 연구자들은 원본 해상도에서의 학습 및 추론이 다수의 작업에서 성능을 크게 향상시킨다는 것을 입증하였으며, 혼합 해상도 훈련 방식을 통해 이러한 문제를 효과적으로 완화할 수 있음을 보여주었습니다.

- **Technical Details**: 연구에서는 생물의학 MLLM의 성능에 있어 해상도의 중요성을 분석하였습니다. 특정 실험 결과에 따르면, 원본 해상도의 MLLM을 사용하였을 때 여러 생물의학 작업에서 성능이 0.54%에서 6.8%까지 향상되었습니다. 특히, 학습과 추론 해상도가 일치하지 않을 경우, 성능이 최대 48.7%까지 저하될 수 있음을 발견하였으며, 이러한 문제를 해결하기 위해 혼합 해상도 훈련 전략을 제안합니다.

- **Performance Highlights**: 혼합 해상도 훈련 전략은 실제 성능을 유지하면서 컴퓨팅 제약을 수용할 수 있음을 발견하였습니다. 이는 정렬된 원본 해상도 학습 및 추론과 비슷한 결과를 달성하면서도 평균 성능 손실을 1.0%로 제한합니다. 연구 결과는 고해상도 생물의학 이미지를 다룰 때 모델 사용자와 개발자 모두에게 실질적인 권장 사항을 제공합니다.



### VLSU: Mapping the Limits of Joint Multimodal Understanding for AI Safety (https://arxiv.org/abs/2510.18214)
Comments:
          10 pages, 5 figures, 4 tables. Under review

- **What's New**: 이번 연구는 멀티모달(Multimodal) 모델의 안전성을 평가하기 위한 새로운 프레임워크인 비전 언어 안전 이해(VLSU, Vision Language Safety Understanding)를 제시합니다. 이 프레임워크는 다양한 안전 패턴을 통해 멀티모달 안전성을 체계적으로 분석하며, 8,187개의 샘플로 구성된 대규모 벤치마크를 활용합니다. 연구결과, 기존의 모델들이 멀티모달 안전 신호를 제대로 이해하지 못한다는 것을 발견하였으며, 이는 이전 연구들에서 하지 못했던 위험 요소의 결합적 해석의 부재에서 기인합니다.

- **Technical Details**: VLSU 프레임워크는 자료 생성 과정에서 두 가지 주요 단계를 포함합니다. 첫 번째 단계에서는 해악 카테고리에 따른 경계(Unsafe) 및 안전(Safe) 위험 등급을 정의하고, 두 번째 단계에서는 개별 모드의 안전 등이 어떻게 결합되는지를 규명합니다. 또한, 새로운 경계 위험 등급을 도입하여 각 모드의 안전 신호를 평가하고, 교차 모드 상호작용(concatenation)을 고려한 안전 평가 방식을 개발합니다.

- **Performance Highlights**: 연구에서는 11개의 최첨단 VLM 모델을 평가한 결과, 개별 모드의 안전 신호에서는 90% 이상의 정확도를 달성했지만, 이미지와 텍스트 결합 시 안전 레이블 판별에선 성능이 20-55%로 급감했습니다. 더불어, 34%의 오류는 각 개별 모드에서 올바른 판별이 이루어졌음에도 발생함을 확인했습니다. 이러한 결과들은 현재의 모델들이 멀티모달 이해와 조합적 추론(compositional reasoning)에서 심각한 한계를 가지고 있음을 드러냅니다.



### Saber: An Efficient Sampling with Adaptive Acceleration and Backtracking Enhanced Remasking for Diffusion Language Mod (https://arxiv.org/abs/2510.18165)
- **What's New**: Diffusion language models (DLMs)는 기존의 autoregressive 모델에 대한 유망한 대안으로 부상하고 있습니다. 이 모델은 코드 생성을 포함한 많은 작업에서 병렬 생성 및 양방향 맥락 모델링의 장점을 제공합니다. 하지만 다음과 같은 두 가지 주요 문제인 속도-품질 트레이드오프와 오류 전파에 관한 문제로 인해 성능이 저하되는 어려움이 있습니다.

- **Technical Details**: 이 논문에서는 Saber라는 새로운 샘플링 알고리즘을 제안합니다. Saber는 Adaptive acceleration과 Backtracking Enhanced Remasking의 약자로, 두 가지 주요 전략인 비균일 난이도 조정과 오류 수정 메커니즘을 통합하여 성능을 개선합니다. DLM의 샘플링 과정에서 적응형 가속 전략을 사용하여 초반에는 신중하게, 이후에는 더 공격적으로 토큰을 생성하도록 설계되었습니다.

- **Performance Highlights**: Saber는 다양한 코드 생성 벤치마크에서 기존 DLM 샘플링 방법에 비해 Pass@1 정확도를 평균 1.9% 향상시키고, 평균적으로 251.4%의 추론 속도 향상을 달성했습니다. 이를 통해 DLM은 기존의 autoregressive 모델에 비해 성능 격차를 크게 줄이게 되었습니다.



### SafeCoop: Unravelling Full Stack Safety in Agentic Collaborative Driving (https://arxiv.org/abs/2510.18123)
- **What's New**: 이 연구는 자연어 기반의 협업 주행 시스템에서의 안전성과 보안성 문제를 체계적으로 분석한 최초의 연구입니다. 자연어를 커뮤니케이션 매체로 활용하여 드라이빙 안전성과 효율성을 향상하려는 최근 경향에 주목하며, 새로운 위험 요소들을 조명합니다. 특히, 메시지 손실, 홀로그램 생성, 의미 조작과 같은 언어 통신의 취약성을 지적하고 이를 해결하기 위한 방안을 제시합니다.

- **Technical Details**: 다양한 공격 전략에 대한 포괄적인 분류 체계를 개발하여 연계 차단, 중계 및 재생 간섭, 콘텐츠 스푸핑 및 다중 연결 위조 등 여러 공격 경로를 분석합니다. 시스템에서 생성된 각 주행 에이전트는 Multi-modal Large Language Models(MLLMs)를 기반으로 작동하며, 두 개의 핵심 모듈인 추론 모듈(Ri)과 행동 모듈(Di)을 갖추고 있습니다. 나아가, 본 논문에서는 언어 기반의 지역적 참조 변환 문제를 해결하기 위해 Agentic Transformation Function(ATF)을 도입합니다.

- **Performance Highlights**: 제안된 방어 체계인 SafeCoop는 CARLA 시뮬레이터에서 32개의 중요한 시나리오에서 테스트되고, 악의적인 공격 하에서도 69.15%의 주행 점수 향상과 67.32%의 F1 점수를 달성하였습니다. 이는 언어 기반 협업 주행에서의 취약성을 확인하고, 이를 감지하는 데 있어 뛰어난 성능을 발휘함을 보여줍니다. 이 연구는 안전하고 신뢰할 수 있는 언어 기반 협업을 위한 향후 연구 방향을 제시합니다.



### SMaRT: Select, Mix, and ReinvenT -- A Strategy Fusion Framework for LLM-Driven Reasoning and Planning (https://arxiv.org/abs/2510.18095)
- **What's New**: 이 논문에서는 Select, Mix, and ReinvenT (SMaRT)라는 새로운 전략 융합 프레임워크를 소개합니다. 이 프레임워크는 단일 전략 프롬프트의 한계를 극복하고, 다양한 추론 전략을 통합하여 성능을 극대화합니다. SMaRT는 기존 방법들과 달리 LLM을 평가자가 아닌 지능적인 통합자로 활용하여, 각 작업에서 최고의 결과를 이끌어냅니다.

- **Technical Details**: SMaRT 프레임워크는 두 개의 단계로 운영됩니다. 첫 번째 단계인 초기 솔루션 단계에서는 LLM이 다양한 기본 전략을 사용하여 후보 솔루션을 생성합니다. 두 번째 단계인 융합 단계에서는 이 후보 솔루션을 평가하고, 다양한 전략의 요소를 통합하여 최종 솔루션을 생성합니다.

- **Performance Highlights**: 실험 결과, SMaRT 프레임워크는 재료 전략을 사용한 기존 방법들과 비교해 우수한 성능을 보여주며, LLM 기반 기술의 경계를 확대합니다. 다양한 벤치마크에서 SMaRT는 전체 작업의 제약 조건 준수와 해결책 품질 측면에서 일관되게 뛰어난 결과를 나타냈습니다. 또한, 작은 오픈소스 LLM과 대형 API 기반 LLM의 출력 결합을 통해 성능이 더욱 향상되었습니다.



### HouseTour: A Virtual Real Estate A(I)gen (https://arxiv.org/abs/2510.18054)
Comments:
          Published on ICCV 2025

- **What's New**: 새로운 방법론인 HouseTour를 소개합니다. 기존의 3D 공간을 기반으로 자연어 요약과 3D 카메라 궤적을 생성하는 작업으로, COVID-19 팬데믹 중에 집 구경 비디오가 인기를 끌면서 이에 대한 수요가 증가했습니다. 이 방법은 전문 장비나 전문 지식 없이도 고품질의 비디오를 생성할 수 있게 합니다.

- **Technical Details**: HouseTour 방법은 주어진 이미지 집합으로부터 카메라 궤적을 생성하고 이를 VLM(vision-language model)에 통합하는 과정을 포함합니다. Diffusion process(확산 과정)를 사용하여 매끄러운 3D 카메라 궤적을 생성하고 이를 3D Gaussian splatting으로 시각화하여 결과 비디오를 합성합니다. 이를 위해 1200개 이상의 집 구경 비디오와 그에 따르는 카메라 궤적, 3D 재구성과 텍스트 설명을 포함한 HouseTour 데이터세트를 발표합니다.

- **Performance Highlights**: 실험 결과는 3D 카메라 궤적 통합이 텍스트 생성 과정에서 성능을 향상시켰음을 보여줍니다. 개별 작업과 최종 성과를 모두 평가하여 새로운 공동 메트릭을 도입하였으며, 이 방법론이 자동화된 프로페셔널 비디오 제작을 가능하게 함을 입증했기 때문에 부동산 및 관광 분야에서 큰 잠재력을 가집니다.



### Subject-Event Ontology Without Global Time: Foundations and Execution Semantics (https://arxiv.org/abs/2510.18040)
Comments:
          32 pages

- **What's New**: 이 논문에서는 복잡한 동적 시스템을 모델링하기 위해 전 세계 시간에 의존하지 않는 주제-사건 온톨로지(subject-event ontology)의 형식을 제안합니다. 주요 원칙으로는 사건을 고정의 행동으로 정의하고, 사건의 순서를 명시적 의존성에 의해 정해진다고 설명합니다. 또한, 온톨로지를 실행 가능하도록 만드는 데이터 흐름 메커니즘을 통해 결정성을 보장하는 점이 특징입니다.

- **Technical Details**: 제안된 형식화는 아홉 가지 공리(A1-A9)를 포함하여 실행 가능한 온톨로지의 정확성을 보장합니다. 주요 기술 요소로는 역사(mhistory)의 단조성(I1), 원인(causality)의 비순환성(I2), 추적 가능성(traceability)(I3)이 있으며, 사건 검증을 위한 스키마(schema) 기반의 모델 접근 방식(A9)에 특별한 주목을 합니다. 또한 전 세계 시간(global time) 없이도 원인 사슬(causal chains)을 자동으로 구축할 수 있는 방안을 제시합니다.

- **Performance Highlights**: 이 형식화는 BSL(Boldsea Semantic Language)로 구현된 boldsea 시스템을 통해 실용성을 입증합니다. 이 시스템은 실행 가능한 온톨로지를 위한 워크플로 엔진(workflow engine)으로 기능하며, 분산 시스템(distributed systems), 마이크로서비스 아키텍처(microservice architectures), DLT 플랫폼 및 다면적 시나리오(multiperspectivity scenarios)에도 적용 가능합니다. 이 논문은 서로 다른 주체(subject)들 간의 상충하는 사실(conflicting facts) 처리에도 유용합니다.



### PLAGUE: Plug-and-play framework for Lifelong Adaptive Generation of Multi-turn Exploits (https://arxiv.org/abs/2510.17947)
- **What's New**: 이번 연구에서는 PLAGUE라는 새로운 프레임워크를 소개합니다. 이는 여러 차례의 공격을 설계할 수 있는 플러그 앤 플레이(plug-and-play) 구조로, 평생 학습(agentic workflows)에서 영감을 받았습니다. PLAGUE는 공격의 생애주기를 세 가지 단계로 나누어(프라이머, 계획자, 마무리 단계), 효율적이고 정보를 풍부하게 탐색할 수 있도록 설계되었습니다.

- **Technical Details**: PLAGUE는 목표에 관련성 있게 진행 상황을 보여주고, 피드백을 통해 진화하며, 다양한 샘플을 생성하는 특성을 갖춰야 합니다. 이를 통해 모델 파라미터 접근 없이도 새로운 다중 차입(jailbreaks)을 발견할 수 있게 해줍니다. 연구에서 제안한 PLAGUE는 강력한 초기화 및 피드백 통합을 통해 미세 조정 없이도 다각적인 공격 전략을 탐색할 수 있습니다.

- **Performance Highlights**: PLAGUE를 사용하여 공격한 영역에서는 97.8%의 성공률을 기록했습니다. 기존의 단일 및 다중 공격 방법 대비 30% 이상의 성공률 향상을 보여주며, 세부 모델로는 OpenAI의 o3에서 81.4%, Claude의 Opus 4.1에서 67.3%의 성공률을 달성했습니다. PLAGUE의 설계 모듈은 공격의 다양성에 크게 기여하면서도 효율성을 유지하는 데 성공했습니다.



### Interpretability Framework for LLMs in Undergraduate Calculus (https://arxiv.org/abs/2510.17910)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 수학 문제 해결 능력에 대한 새로운 해석 가능성(interpretability) 프레임워크를 제안합니다. 기존의 평가 방법이 최종 답변의 정확성에만 집중하는 반면, 본 프레임워크는 문제 해결 과정의 추론(Reasoning) 프로세스와 교육적으로 타당한( pedagogically valid) 패턴을 평가합니다. 이 연구는 수학 교육에서 AI 도입의 투명성과 책임감을 강화하기 위한 기초자료를 제공합니다.

- **Technical Details**: 제안된 해석 가능성 프레임워크는 두 가지 주요 구성 요소로 이루어져 있습니다: (1) 추론 흐름 분석(Reasoning flow analysis)으로 LLM의 출력 결과를 레이블이 붙은 작업 및 개념으로 세분화하고, (2) 입력 요소가 출력 행동에 미치는 영향을 정량화하는 프롬프트 민감도 해제(prompt sensitivity ablation) 방법을 포함합니다. 이 프레임워크는 대학 수학 시험(Calculus I~III)을 통해 LLM의 행동을 평가하며, 정성적 및 정량적 지표를 기반으로 모델의 이유와 오류를 파악합니다.

- **Performance Highlights**: 실험 결과, LLM은 흔히 문법적으로 유창하나 개념적으로 결함이 있는 솔루션을 생성합니다. 또한 추론 패턴은 프롬프트 구문 및 입력의 변동에 민감하게 반응하는 경향이 발견되었습니다. 본 연구는 LLM의 성능을 학생들의 실제 점수와 비교하고, 결과적으로 교육적 지침(instructional alignment)과 모델 한계를 이해하는 데 중요한 통찰력을 제공합니다.



### BreakFun: Jailbreaking LLMs via Schema Exploitation (https://arxiv.org/abs/2510.17904)
- **What's New**: 이번 논문에서는 Large Language Models (LLMs)의 구조적 데이터 처리 능력과 구문 규칙 준수로 인해 발생하는 보안 취약점에 대해 연구했습니다. BreakFun이라는 jailbreak 방법론을 도입하여 이러한 취약점을 활용하는 새로운 방식으로, LLM이 복잡한 구조를 따르려는 경향을 악용하여 해로운 콘텐츠를 생성하도록 유도합니다. 특히, 논문은 이 공격 방식이 다양한 LLM에 전이 가능함을 입증하였으며, 13개의 모델에서 평균 89%의 성공률을 기록했습니다.

- **Technical Details**: BreakFun은 세 가지 부분으로 구성된 프롬프트를 사용하여 악의적인 요청을 무해한 기술 과제로 변환시키는 구조입니다. 이 방법론의 핵심에는 'Trojan Schema'가 있으며, 이는 비정상적인 데이터 구조를 통해 모델이 해로운 콘텐츠를 생성하도록 강요합니다. 이 연구는 LLM의 인지적 지향성과 기술적 요구에 대한 과도한 준수 경향을 이용하여 안전 메커니즘을 우회하는 접근 방식을 설명합니다.

- **Performance Highlights**: BreakFun의 효과성을 평가하기 위한 연구에서는 다양한 13종의 LLM 모델에서 공격의 성공률이 높음을 발견하였습니다. 특히, 여러 주요 모델에서는 100%의 공격 성공률이 달성되었습니다. 또한, Adversarial Prompt Deconstruction이라는 방어 체계를 제안하여 공격을 효과적으로 저지할 수 있음을 보여주었으며, 이는 LLM의 강점을 약점으로 전환할 수 있는 새로운 관점을 제공합니다.



### Are LLMs Court-Ready? Evaluating Frontier Models on Indian Legal Reasoning (https://arxiv.org/abs/2510.17900)
- **What's New**: 이번 연구는 인도의 공공 법률 시험을 기준으로 하여 법률 분야에서 대형 언어 모델(LLMs)의 능력을 평가하는 첫 번째 연구입니다. 연구진은 CLAT, DJS/DHJS 등의 객관식 질문 및 대법원 변호사시험과 같은 주관식 평가를 포함하여 다년간의 데이터를 수집하였습니다. 이를 통해 LLM들이 법원에서 요구하는 기준에 미달하는 부분을 과학적으로 분석하고, 효과적인 법률 도구로서의 가능성을 제시하고 있습니다.

- **Technical Details**: 이 논문은 다수의 LLM을 사용하여 인도의 법률 시험에 대한 평가를 수행했습니다. 연구진은 객관식 질문 6,218개와 주관식 변호사시험 자료로 구성된 데이터를 활용하여 모델들의 성능을 비교 분석하였습니다. 또한, 특정 변호사가 작성한 답안과 모델이 생성한 답안을 쌍으로 비교하여 평가하기 위해 인증된 채점자를 사용하여 블라인드 연구를 진행했습니다.

- **Performance Highlights**: 연구 결과, 최신 모델들이 객관식 시험에서 높은 성과를 보였지만, 주관식 답안에서 인간의 탑 스코어러를 초과하지 못했으며, 세 가지 주요 실패 모드가 발견되었습니다. LLM들이 법적 절차와 포맷 준수에 어려움을 겪었고, 인용의 정확성과 법정에 적합한 목소리 및 구조 또한 부족하다는 점이 분석되었습니다. 이러한 결과는 법적 문서 작성 및 절차적 전략에서는 여전히 인간의 역할이 필수적임을 보여주고 있습니다.



### Hierarchical Federated Unlearning for Large Language Models (https://arxiv.org/abs/2510.17895)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)에서 프라이버시와 보안을 보존하면서도 다양한 비율의 지식을 제거할 수 있는 연합 학습(Unlearning) 접근법인 Federated UnLearning Merge (FULM)를 제안합니다. 기존의 방법들이 비대칭 접근으로 인해 성능이 저하되는 문제를 해결하기 위해서, 우리는 특정 작업에 특화된 어댑터 학습을 통해 잊기와 유지하는 목표를 분리합니다. 이러한 새로운 구조는 사용자들의 다양한 잊기 요청을 효과적으로 처리할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 총체적으로, FULM은 연합 집합체에서 유용한 지식을 보존하도록 설계되었습니다. 이 방식은 클라이언트 업데이트를 분석하여 작업 어댑터 내의 매개변수 분포 및 방향의 차이를 확인하고, 이러한 패턴에 맞춘 계층적 집합 방법론을 적용합니다. 이 접근법은 비대칭 데이터 접근을 처리하는 동시에 꾸준한 업데이트가 가능하도록 합니다.

- **Performance Highlights**: 우리의 연구에서는 WMDP, TOFU, MUSE의 세 가지 LLM 잊기 벤치마크에서 FULM의 성능을 평가했습니다. 실험 결과, FULM은 기존의 잊기와 연합 학습 상용화 모델보다 동적 잊기 작업에서 뛰어난 성능을 발휘하여 모델의 효용성을 유지하면서도 요청된 지식을 효과적으로 제거할 수 있음을 보여주었습니다.



### Metrics and evaluations for computational and sustainable AI efficiency (https://arxiv.org/abs/2510.17885)
Comments:
          11 pages, 2 tables

- **What's New**: 이 논문에서는 AI 모델 추론에 대한 통합되고 재현 가능한 방법론을 제안합니다. 기존의 방법들이 성능, 효율성 및 환경 영향을 평가하는 데 한계가 있었던 반면, 이 프레임워크는 지연 시간(latency), 처리량(throughput), 에너지 소비 및 탄소 배출량과 같은 메트릭을 통합하여 실질적인 평가를 가능하게 합니다.

- **Technical Details**: AI 시스템의 지연 시간은 입력 데이터 수신 시점부터 출력 결과를 생성할 때까지의 시간 간격을 정의합니다. 이 논문에서 제시된 메트릭들은 정확한 성과 측정을 위해 전력 및 에너지를 정량화하고, 지연 시간의 다양한 구성 요소를 평가하여 효율적인 AI 서비스 개발을 지원합니다. 본 프레임워크는 다양한 하드웨어 플랫폼에서 다중 정밀도 모델을 평가하며, 소프트웨어 스택을 통해 일관되게 비교할 수 있는 기반을 제공합니다.

- **Performance Highlights**: 논문에서 제안한 방법론은 AI 시스템의 효율성과 탄소 발자국 간의 Trade-off를 명확히 하여 의사 결정을 지원합니다. 또한, 본 연구의 결과는 독립적으로 검증할 수 있도록 공개된 코드와 스크립트를 통해 제공되며, 연구자 및 실무자들이 지속 가능한 AI 배포를 위한 증거 기반 결정을 내리는 데 도움을 줍니다.



### Does GenAI Rewrite How We Write? An Empirical Study on Two-Million Preprints (https://arxiv.org/abs/2510.17882)
- **What's New**: 이번 논문은 2016년부터 2025년까지의 210만 개 이상의 preprint을 분석하여 generative large language models (LLMs)가 학술 출판에 미치는 영향에 대한 체계적인 증거를 제공합니다. 기존 연구의 한계를 극복하기 위해 다양한 데이터 분석 기법을 활용하여 연구 생산 방식의 구조적 변화 및 스타일 변화를 조명하고 있습니다. 특히, LLMs는 제출 및 수정 주기를 가속화하고 언어의 복잡성을 증가시키며 AI 관련 주제를 확대하는 등 학술 출판의 변화를 촉진하고 있습니다.

- **Technical Details**: 이 연구는 arXiv, bioRxiv, medRxiv 및 SocArXiv 등 4개의 주요 preprint 저장소에서 수집된 데이터를 기반으로 하는 다층 분석 프레임워크를 도입합니다. 이 프레임워크는 interrupted time-series 모델, 협업 및 생산성 지표, 언어 프로파일링, 주제 모델링을 통합하여 연구 결과의 양, 저자, 스타일 및 분야별 방향성을 평가합니다. 이러한 방식으로 생성적인 AI 도구에 대한 연구 출력을 체계적으로 조사하고 분석할 수 있는 토대를 마련합니다.

- **Performance Highlights**: 연구 결과는 LLMs가 일부 분야에서 빠르게 학술 출판의 패턴을 변화시켰음을 보여줍니다. AI 관련 주제의 비율이 크게 증가하고, 계산적으로 집중된 분야에서 더 두드러진 속도의 채택과 변화가 관찰되었습니다. 이러한 결과는 LLMs가 일률적으로 모든 분야에 영향을 주기보다는 특정 영역에서 선택적으로 촉매 역할을 한다는 점을 강조합니다.



New uploads on arXiv(cs.IR)

### LLMs as Sparse Retrievers:A Framework for First-Stage Product Search (https://arxiv.org/abs/2510.18527)
Comments:
          16 pages

- **What's New**: PROSPER라는 새로운 프레임워크를 제안하여 대규모 언어 모델(LLMs)을 활용한 희소 검색(sparse retrieval) 방식을 발전시키고 있습니다. 이 접근 방식은 검색 쿼리의 짧고 핵심적인 용어를 강조하여 LLM이 생성할 수 있는 비관련 용어의 환각(hallucinations)을 줄이고, 훈련 초기화를 용이하게 하는 기능을 포함합니다. 이를 통해 온라인 및 오프라인에서 기존 희소 기반 모델에 비해 현저한 성과를 기록하고 있습니다.

- **Technical Details**: PROSPER는 두 가지 주요 구성 요소로 구성되어 있습니다. 첫째, 문맥적 잔여 네트워크(literal residual network, LRN)를 도입하여 사용자 쿼리 및 제품의 핵심 용어에 대한 가중치를 증가시킵니다. 둘째, 어휘 집중 창(lexical focusing window, LFW)을 통해 훈련 초기화 과정을 보다 효과적으로 만들어, 모델이 고차원 공간에서 빠르게 학습할 수 있도록 지원합니다. 이는 FLOPS 정규화(fine-grained control)와 결합 되어 모델의 학습 안정성 및 효율성을 극대화합니다.

- **Performance Highlights**: PROSPER는 Multi-CPR 전자상거래 데이터 세트와 실제 데이터 세트를 활용한 오프라인 실험에서 상당한 성과를 달성하며, BM25 기준 대비 10.2% 향상된 제품 회수율을 보이고 있습니다. 또한 SPLADE 기준 대비 4.3% 개선을 이루었으며, 온라인 실험에서도 0.64%의 매출 증가를 기록했습니다. 이와 같은 성과는 고급 밀집 검색(dense retrieval) 모델과 동등한 성능을 제공하며, LLM의 잠재력을 효과적으로 활용하고 있음을 증명합니다.



### Evaluating LLM-Based Mobile App Recommendations: An Empirical Study (https://arxiv.org/abs/2510.18364)
Comments:
          Under review

- **What's New**: 이 논문에서는 Mobile Application (모바일 애플리케이션) 추천을 위해 사용되는 Large Language Models (대형 언어 모델)의 추천 생성, 정당화 및 순위 매김을 분석합니다. 이는 사용자 경험을 향상시키기 위한 유연한 방법으로서, 기존의 키워드 기반 검색에 대한 대안을 제공합니다. 특히, LLM이 Mobile App 추천에서 전통적인 App Store Optimization (앱 스토어 최적화, ASO) 지표와 얼마나 일관성 있는지를 조사한 결과를 밝힙니다.

- **Technical Details**: 논문은 세 가지 주요 기여를 통해 LLM 기반 추천 시스템에서의 권장 사항의 일관성 및 정확성을 분석합니다. 첫째, LLM의 출력을 통해 도출된 16개의 일반화 가능한 순위 기준을 세분화한 분류법을 제시합니다. 둘째, 추천이 LLM의 내부 및 외부 일관성에 대한 체계적인 평가 프레임워크를 구축하고, 셋째, AI 기반 추천 시스템에서 재현성을 지원하는 복제 패키지를 제공합니다.

- **Performance Highlights**: 연구 결과, LLM은 일반적으로 넓고 단편적인 순위 기준에 의존하며, 전통적인 ASO 지표와는 부분적으로만 일치하는 것으로 나타났습니다. 높은 순위에 있는 앱들은 대체로 일관성을 보였으나, 순위가 깊어질수록 변동성이 증가했습니다. LLM은 명시적인 순위 지시 사항에 대해 다양한 반응을 보이며, 이는 대화형 앱 검색에서의 복잡한 추론 동력을 반영하고 있습니다.



### Enhancing Hotel Recommendations with AI: LLM-Based Review Summarization and Query-Driven Insights (https://arxiv.org/abs/2510.18277)
- **What's New**: 이번 연구에서는 "instaGuide"라는 웹 애플리케이션이 소개되었으며, 이는 AI 기반으로 Booking.com 수용소의 비정형 텍스트 리뷰를 요약하여 사용자 맞춤형 질문에 답변하는 기능을 제공합니다. 이를 통해 사용자는 숙소에 대한 깊이 있는 정보를 빠르게 얻을 수 있으며, 리뷰를 하나하나 읽는 데 소모되는 시간을 크게 줄일 수 있습니다. 또한, 다양한 Large Language Models (LLMs)을 평가하여 최적의 모델을 선택하였다는 점에서 연구의 기여가 돋보입니다.

- **Technical Details**: 이 연구는 리뷰 요약 및 질문 응답 기능을 위한 LLM 기반 시스템을 개발하였으며, 정보 검색(Data Retrieval) 및 LLM과 결합된 Retrieval-Augmented Generation (RAG) 기술을 활용합니다. 기존의 추천 시스템 연구는 대부분 협업 필터링(collaborative filtering) 및 콘텐츠 기반 필터링(content-based filtering) 기술에 의존했지만, 이 연구에서는 비정형 텍스트에서 중요한 정보를 추출하는 데 중점을 두었습니다. 자연어 처리(NLP) 및 감정 분석을 이용한 의견 채굴(opinion mining) 기술이 사용되었으며, 이러한 방법들은 적절한 추천 결과를 생성하는 데 기여합니다.

- **Performance Highlights**: 연구 결과, instaGuide 도구를 사용하면 사용자가 단기 임대 아파트를 검색하는 데 필요한 시간이 크게 단축되며 의사 결정 과정의 효율성이 향상되었습니다. LLM 기반 요약 시스템은 사용자가 특정 기준을 충족하는 숙소를 쉽게 찾을 수 있도록 도와주며, 이는 사용자의 전반적인 경험을 향상시키는 데 중요한 역할을 합니다. 이와 같은 도구는 시간이 절약될 뿐만 아니라 사용자가 필요한 정보를 더 빠르게 얻을 수 있도록 하여, 관광 업계에서의 변화를 가져올 것으로 기대됩니다.



### LIME: Link-based user-item Interaction Modeling with decoupled xor attention for Efficient test time scaling (https://arxiv.org/abs/2510.18239)
Comments:
          16 pages

- **What's New**: LIME은 대규모 추천 시스템의 효율성을 개선하기 위해 설계된 새로운 아키텍처로, 사용자와 후보 항목 간의 상호작용을 더 효과적으로 처리하기 위해 혁신적인 '링크 임베딩' 메커니즘과 선형 주의 메커니즘인 LIME-XOR을 도입하였다. 이 구조는 기존의 트랜스포머 모델의 계산 복잡성을 줄이며, 후보 세트의 크기나 사용자 이력의 길이에 관계없이 추론 비용을 줄이는 데 기여한다.

- **Technical Details**: LIME은 링크 임베딩을 사용하여 사용자와 후보 항목 간의 상호작용을 분리하며, 이로 인해 복잡한 쿼리-키 주의 가중치를 사전에 계산하여 캐시할 수 있다. 또한 XOR 주의 마스킹을 통해 사용자 이력의 자기주의 복잡성을 제곱에서 선형으로 감소시켜 추천 모델의 효율성을 향상시킨다. 이러한 방식은 추천 시스템의 예측 성능을 유지하면서도 추론 속도를 10배 증가시킨다.

- **Performance Highlights**: LIME은 여러 공개 데이터셋과 산업 데이터셋에서 실험을 통해 기존의 최첨단 트랜스포머 모델에 필적하는 성능을 보이면서도, 대규모 후보 세트나 긴 이력 길이의 경우에도 빠른 추론 속도를 유지한다. 실제로 주요 추천 플랫폼에서 배포한 결과, LIME은 사용자 참여도 증가를 이루었고, 최소한의 추론 비용으로 개선된 성능을 기록하였다.



### From AutoRecSys to AutoRecLab: A Call to Build, Evaluate, and Govern Autonomous Recommender-Systems Research Labs (https://arxiv.org/abs/2510.18104)
- **What's New**: 추천 시스템(RecSys) 연구가 모델과 평가 기술의 발전을 이루었지만, 연구 프로세스 자체의 자동화는 거의 간과하고 있다는 주장을 하고 있습니다. AutoRecLab이라는 새로운 패러다임을 제안하며 문제 구상, 문헌 분석, 실험 설계와 실행, 결과 해석, 원고 작성, 기록 보존까지 전 과정을 자동화하는 연구 환경을 목표로 하고 있습니다. 이러한 자동화의 필요성은 최근 자동화 과학 분야의 발전 특히 Multi-Agent AI Scientist와 AI Co-Scientist 시스템에서 기인하고 있습니다.

- **Technical Details**: 추천 시스템 분야의 기존 자동화 도구인 AutoRecSys는 알고리즘 선택과 하이퍼파라미터 조정에 국한되어 있습니다. 그러나 AutoRecLab은 LLM 중심의 아이디어 생성 및 보고서 작성을 결합해 자동 실험을 포함하는 전면적인 프로토타입을 개발하는 것을 지향합니다. 이 연구는 기존의 인간 기여 없이도 반복 가능한 RecSys 발견을 도출하는 기준 및 대회 수립, AI 생성 제출물에 대한 투명한 리뷰 공간 마련 등 다양한 노력을 포함하고 있습니다.

- **Performance Highlights**: 최근 Sakana의 AI Scientist와 구글 AI Co-Scientist의 발전을 통해 완전 자동화된 연구 프로세스가 가능해짐을 보여주고 있습니다. AI Scientist는 최소한의 인간 개입으로 연구 논문을 작성하는 성과를 내었으며, 실제 동료 평가에 통과한 첫 AI 생성 논문을 기록했습니다. 이러한 발전은 연구 자동화의 가속화를 나타내며, RecSys 커뮤니티가 이를 뒤따라야 할 필요성을 강조합니다.



### ImageGem: In-the-wild Generative Image Interaction Dataset for Generative Model Personalization (https://arxiv.org/abs/2510.18433)
- **What's New**: 우리는 개별 선호를 이해하는 생성 모델에 대한 연구를 위한 데이터셋, ImageGem을 소개합니다. 이 데이터셋은 57K 사용자의 실제 상호작용 데이터를 포함하며, 커스터마이즈된 LoRAs를 포함하여 242K개 사용자의 생성된 이미지를 포함하고 있습니다. 데이터셋의 사용자 선호 주석을 통해 우리는 선호 정렬 모델을 향상시킬 수 있었습니다.

- **Technical Details**: ImageGem 데이터셋은 Civitai 플랫폼에서의 데이터를 바탕으로 구성되었으며, 이 플랫폼은 개인화된 이미지 생성 모델과 함께 관련 메타데이터를 제공합니다. 데이터 구성은 LoRA 모델, 생성된 이미지, 그리고 이러한 모델을 업로드한 사용자들 간의 관계를 기반으로 하여, 사용자 특정 선호도를 효율적으로 조회하고 분석할 수 있게 설계되었습니다.

- **Performance Highlights**: ImageGem 데이터셋을 활용하여 사용자 개인화에 맞춘 이미지 검색 및 생성 모델 추천 성능을 시험했습니다. 또한, VLM(vision-language model)을 활용하여 사용자 선호를 캡셔닝하고 정렬하는 방법을 제안하였으며, 이를 통해 이미지 생성 모델의 개인화 작업에서 새로운 패러다임을 마련했습니다.



### Censorship Chokepoints: New Battlegrounds for Regional Surveillance, Censorship and Influence on the Intern (https://arxiv.org/abs/2510.18394)
Comments:
          15 pages, 2 figures

- **What's New**: 이 논문에서는 온라인 정보에 대한 검열의 현재 상태와 최근 몇 년 간의 변화를 포괄적으로 분석하고 있습니다. 특히, 저자들은 'chokepoints'라는 새로운 개념을 소개하여, 콘텐츠 생산 및 전달 과정에서의 병목 현상을 이해하고자 합니다. 이를 통해, 대규모 클라이언트 측 감시 및 필터링 메커니즘의 발전을 설명하고 있습니다. 기존의 검열 형태를 넘어서, 정보 접근의 새로운 장애물들이 등장하고 있음을 강조합니다.

- **Technical Details**: 인터넷 검열은 클라이언트 기반(client-based), 서버 기반(server-based), 네트워크 기반(network-based)으로 분류됩니다. 저자들은 검열이 단순히 접근 제어에 그치지 않고, 정보의 흐름을 효과적으로 관리할 수 있는 새로운 형태의 필터링 시스템으로 이동하고 있음을 주장합니다. '하드 chokepoints'와 '소프트 chokepoints'라는 두 가지 유형으로 검열 시스템을 분류하고, 각각의 특징과 구현 방법을 설명합니다.

- **Performance Highlights**: 저자들은 이러한 새로운 접근 방식이 정보 검열의 이해를 진전시키고, 전통적인 검열 방식으로는 식별하지 못했던 새로운 정보 통제 방식들을 드러낸다고 주장합니다. 하드 chokepoints는 영구적인 정보 손실을 초래하는 반면, 소프트 chokepoints는 널리 퍼지는 인터넷 환경에서 더 잠재적이고 은밀한 방식으로 작용합니다. 이 연구는 검열의 복잡한 형태를 설명하고, 사용자가 인지하지 못하는 상황에서도 정보 접근 조작이 이루어질 수 있음을 강조합니다.



### KrishokBondhu: A Retrieval-Augmented Voice-Based Agricultural Advisory Call Center for Bengali Farmers (https://arxiv.org/abs/2510.18355)
Comments:
          6 pages, 7 figures, 5 tables, submitted to the 11th IEEE International Women in Engineering (WIE) Conference on Electrical and Computer Engineering (WIECON-ECE 2025)

- **What's New**: 본 논문에서는 한국어가 아닌 벵골어를 사용하는 농민들을 위한 음성 기반 농업 조언 시스템, KrishokBondhu를 소개합니다. 본 시스템은 Retrieval-Augmented Generation (RAG) 프레임워크를 기반으로 하여 권위 있는 농업 자료를 통합적으로 활용하며, 농민들이 좀 더 신뢰할 수 있는 기술적 조언을 쉽게 받도록 설계되었습니다. KrishokBondhu는 전화 인터페이스를 통해 실시간으로 농업 관련 조언을 제공합니다.

- **Technical Details**: KrishokBondhu의 지식 기반은 방글라데시 정부 및 비정부기구에서 발행한 농업 관련 문서로 구성됩니다. 약 2,500페이지 분량의 자료들은 Optical Character Recognition (OCR) 및 문서처리 파이프라인을 통해 디지털화 및 구조화되며, Vector Database에 인덱싱 되어 효율적인 검색이 가능하게 됩니다. 음성 인식 모듈은 농민의 질의를 텍스트로 변환하는 역할을 하며, Gemma 3-4B라는 대형 언어 모델이 컨텍스트에 맞는 응답을 생성합니다.

- **Performance Highlights**: KrishokBondhu는 다양한 농업 관련 질의에 대해 72.7%의 높은 답변 품질을 기록하였으며, KisanQRS 벤치마크와 비교할 때 44.7%의 개선을 이루었습니다. 특히, 컨텍스트의 풍부함과 완전성에서 각각 367%와 100.4%의 향상을 보였습니다. 이 시스템은 콜센터 접근성, 다국어 음성 상호작용, 현대 RAG 기법을 결합하여 외딴 지역의 농민들에게 전문가 수준의 농업 조언을 제공하는 가능성을 입증하였습니다.



New uploads on arXiv(cs.CV)

### Grasp Any Region: Towards Precise, Contextual Pixel Understanding for Multimodal LLMs (https://arxiv.org/abs/2510.18876)
- **What's New**: 이 논문에서는 Grasp Any Region (GAR)을 제안하여 멀티모달 대형 언어 모델(MLLM)의 지역 이해 능력을 개선하고자 하였습니다. 기존의 지역 MLLMs가 중요 글로벌 컨텍스트(global context)를 간과했던 점을 보완하며, GAR는 지역 수준의 시각적 이해를 위한 종합적인 접근법을 제공합니다. GAR는 RoI-aligned feature replay 기법을 활용해 개별 지역의 정확한 인식과 여러 프롬프트 간의 상호작용 모델링을 지원합니다.

- **Technical Details**: GAR은 특정 프롬프트와 함께 전체 이미지 정보를 인코딩하는 방법으로, 이를 통해 지역별 세부 정보를 정확하게 캡처하는 능력을 향상시킵니다. 이러한 방식은 RoI-Align을 통해 글로벌 Feature map에서 관련된 Feature를 수집하여 지역과 글로벌 정보를 동시에 고려할 수 있게 합니다. 또한, GAR-Bench를 도입하여 여러 지역의 상호작용과 복잡한 추론 과정을 평가하는 새로운 벤치마크를 제공합니다.

- **Performance Highlights**: 실험 결과 GAR-1B 모델은 DAM-3B 및 PAM-3B보다 상세한 캡션 생성에서 우수한 성능을 보여주며, 다중 프롬프트 모델링에서 InternVL3-78B를 초월하는 성과를 달성했습니다. GAR-8B는 VideoRefer-BenchQ에서 인도메인 모델인 VideoRefer-7B를 초과하는 성능을 발휘하여 동영상에 대한 이해 능력 또한 강화되었습니다. 이러한 결과는 GAR의 강력한 커뮤니케이션 및 이해 능력을 입증합니다.



### DSI-Bench: A Benchmark for Dynamic Spatial Intelligenc (https://arxiv.org/abs/2510.18873)
- **What's New**: 이 논문에서는 동적인 공간 관계를 이해하는 데 필요한 Dynamic Spatial Intelligence(DSI)를 소개하고, 1,000개의 동영상과 1,700개의 질문으로 구성된 DSI-Bench라는 벤치마크를 제안합니다. DSI-Bench는 관찰자와 객체의 동작 패턴을 분석하기 위해 특별히 설계되었으며, 다양한 비디오 상황에서 모델의 성능을 평가할 수 있는 기회를 제공합니다.

- **Technical Details**: DSI-Bench는 5가지로 분리된 동작 패턴을 포함한 1,000개의 동적 비디오와 1,700개의 수동으로 주석이 달린 질문들로 구성되어 있습니다. 이 벤치마크는 관찰자와 객체의 관계, 질문 유형의 다양성, 그리고 공간적/시간적 대칭성을 통해 모델의 편향성을 줄이고 시스템적인 평가를 가능하게 합니다. 이를 통해 기존 시각-언어 모델(VLM)의 한계를 명확히 드러내고 있습니다.

- **Performance Highlights**: 14개의 시각-언어 모델을 평가한 결과, VLM들은 종종 관찰자와 객체의 동작을 혼동하며, 시맨틱 편향 때문에 시각적 지각에서 왜곡이 발생하는 경향을 보였습니다. 고전적인 3D 제약 조건이 동적 장면 비디오에서 상대적인 자세 관계를 일관되게 설명하는 데 실패하는 등 여러 중요한 한계들이 드러났습니다. DSI-Bench는 향후 동적인 공간 지능을 갖춘 일반 및 전문 모델의 개발에 중요한 통찰력과 발견을 제공할 것으로 예상됩니다.



### DP$^2$O-SR: Direct Perceptual Preference Optimization for Real-World Image Super-Resolution (https://arxiv.org/abs/2510.18851)
Comments:
          Accept by NeurIPS 2025

- **What's New**: 이 논문은 기존의 T2I(diffusion) 모델의 불확실성을 활용하여 이미지 초해상도(Real-ISR) 작업의 성능을 향상시키는 새로운 프레임워크 DP²O-SR(Direct Perceptual Preference Optimization for Real-ISR)을 소개합니다. 전통적인 손실 함수와 달리, 인간의 지각 선호도를 반영하는 혼합 보상 신호를 통해 모델의 출력을 개선합니다. 또한, 다양한 출력의 퍼셉션을 통해 더 풍부한 비교 정보를 생성하여 훈련 효율을 높입니다.

- **Technical Details**: 이 프레임워크는 대규모 인간 선호 데이터에 대해 훈련된 이미지 품질 평가(IQA) 모델을 활용하여 구조적인 충실도와 자연스러운 외관을 동시에 고려하는 하이브리드 보상 신호를 구축합니다. 또한, 우리는 계층적 선호 최적화(Hierarchical Preference Optimization, HPO)를 제안하여 훈련 쌍의 가중치를 적응적으로 조절하고, 더 유의미한 신호에 집중하여 학습을 개선합니다. 다양한 모델에 대한 실험을 통해 하이브리드 보상 시스템이 출력의 질적 향상에 기여함을 입증했습니다.

- **Performance Highlights**: DP²O-SR은 실제 세계에서의 초해상도 평가 데이터셋에서 우수한 성과를 나타냅니다. 상대적으로 작은 확산 모델(ControlNet-SD2)과 더 큰 플로우 기반 모델(ControlNet-FLUX) 모두 DP²O-SR을 통한 초기 훈련 반복에서 상당한 퍼셉션 보상 향상을 달성했습니다. 특히, C-FLUX는 약 0.51에서 0.65로 향상되었고, C-SD2는 0.62로 빠르게 성장하여 초기 훈련에서 상위 2위의 성능을 기록했습니다.



### See the Text: From Tokenization to Visual Reading (https://arxiv.org/abs/2510.18840)
- **What's New**: 이 논문은 새로운 비전 중심의 토큰화 방법인 SeeTok을 선보입니다. SeeTok은 텍스트를 이미지로 변환하고, pretrained multimodal LLMs를 사용하여 이를 해석하는 방식으로, 전통적인 subword tokenization보다 더 효율적입니다. 이 접근 방식은 특히 저자원 언어에 대한 결함을 최소화하고, 토큰 수를 크게 줄이며, 계산 복잡성을 낮추는 장점을 가지고 있습니다.

- **Technical Details**: SeeTok의 기술적 기반은 문자를 시각적 패턴으로 처리하는 것입니다. 이러한 방식은 Visual Word Form Area (VWFA)와 같은 인간의 인지 메커니즘에서 영감을 받아서 개발되었습니다. 텍스트를 이미지로 변환한 후 pretrained MLLMs의 시각적 인코더를 사용하여 텍스트 표현을 추출하고, 이를 LLM 모델에 전달하여 처리합니다. 이를 통해 저비용 수정과 일반적인 학습 방법 없이 시각 텍스트 지침을 그대로 해석할 수 있습니다.

- **Performance Highlights**: SeeTok은 세 가지 자연어 처리 작업에서 전통적인 텍스트 토크나이저와 유사한 성능을 보이며, 4.43배 적은 비주얼 토큰과 70.5%의 FLOP 감소를 기록했습니다. 다국어 번역 작업에서는 13개 언어를 아우르며, 텍스트 토크나이저 대비 더 우수한 교차 언어 전이능력을 나타내며, 86% 낮은 fertility를 기록했습니다. 또한 문자 수준, 단어 수준, 시각 수준 공격에 대해 강력한 내구성을 보이며 성능 저하가 적다는 특징이 있습니다.



### FedDEAP: Adaptive Dual-Prompt Tuning for Multi-Domain Federated Learning (https://arxiv.org/abs/2510.18837)
Comments:
          Accepted at MM 2025

- **What's New**: 이번 연구에서는 FedDEAP라는 적응형 연합 프롬프트 조정 프레임워크를 제안하여 다중 도메인 환경에서 CLIP의 일반화 능력을 향상시키고자 합니다. 이 방법은 SEMANTIC과 DOMAIN Transformation Networks를 사용해 이미지의 의미와 도메인 별 특징을 분리하여 도메인 고유 정보를 유지합니다. 또한, 글로벌 의미 프롬프트와 지역 도메인 프롬프트를 함께 활용하여 공유된 정보와 개인화된 정보를 균형 있게 다루며, 텍스트와 비주얼 표현 간의 정렬을 통해 정보 일관성을 극대화합니다.

- **Technical Details**: FedDEAP는 일반화된 지식 공유와 지역 도메인 고유 특성 보존을 균형 있게 다루기 위해 두 개의 프롬프트를 사용합니다: 글로벌 의미 프롬프트와 개인화된 도메인 프롬프트입니다. 이 과정에서 Equiangular Tight Framework (ETF)를 이용한 비편향 변형 네트워크를 도입하여 이미지 내의 의미 공간과 도메인 공간을 분리하고, 학습된 프롬프트를 이미지 특징과 정렬시킵니다. 이러한 이중 프롬프트와 정렬 전략은 데이터를 연합 학습하는 동안 도메인 고유 정보를 유지하여 다중 도메인에서의 일반화를 개선하는 것을 목표로 합니다.

- **Performance Highlights**: 세 가지 자연 이미지 데이터셋과 하나의 의료 이미지 데이터셋에서 기존 방법들보다 뛰어난 분류 성능을 보이며 최첨단(classification performance) 결과를 달성하였습니다. 또한, 데이터 불균형 환경에서도 높은 효율성을 자랑하며, 빠른 추론 성능을 제공합니다. 이 연구는 FedDEAP의 이론적 분석과 상세한 ablation study를 통해 제안된 접근 방식이 이미지 내에서 의미와 도메인 정보를 효과적으로 보존한다는 것을 입증합니다.



### Unifying and Enhancing Graph Transformers via a Hierarchical Mask Framework (https://arxiv.org/abs/2510.18825)
Comments:
          Accepted by NeurIPS 2025 (Poster)

- **What's New**: 이번 연구에서는 그래프 표현 학습을 위한 강력한 패러다임인 Graph Transformers (GTs)를 제안합니다. 기존의 GT들은 특정 상호작용을 모델링하기 위해 복잡한 구조에 의존하였으나, 이는 유연성을 제한하는 단점이 있었습니다. 이를 해결하기 위해 통합된 계층 마스크 프레임워크를 제시하며, 이 프레임워크는 모델 아키텍처와 주목 마스크의 구조 간의 기본적인 동등성을 드러냅니다.

- **Technical Details**: 제안된 M3Dphormer는 Mixture-of-Experts 기반의 Graph Transformer로, 다계층 마스킹과 이중 주의 계산을 통해 상호작용 정보를 적응적으로 통합합니다. 이 모델은 세 가지 이론적으로 기반이 있는 계층 마스크를 사용하며, 지역, 클러스터, 글로벌의 다양한 상호작용을 포괄적으로 모델링합니다. 또한, 지역 마스크의 희소성에 따라 조밀한 모드와 희소한 모드 간에 동적으로 전환하는 이중 주의 계산 방식을 도입하여 확장성을 보장합니다.

- **Performance Highlights**: M3Dphormer는 9개의 벤치마크 데이터셋에 대한 광범위한 실험을 통해 15개의 강력한 기본 모델을 지속적으로 초과 달성하였습니다. 이러한 성능 향상은 통합된 프레임워크와 모델 설계의 효과를 검증하는 결과입니다. 특히, 여러 단계의 상호작용을 종합적으로 활용하는 오라클 전략이 다른 모델들보다 우수한 성능을 보였으며, 이는 계층적 정보 통합의 핵심 도전 과제를 보여줍니다.



### SAM 2++: Tracking Anything at Any Granularity (https://arxiv.org/abs/2510.18822)
Comments:
          8 pages, and 10 pages in Supplementary Material

- **What's New**: 이번 논문에서는 SOT(Single Object Tracking), VOS(Video Object Segmentation), Point Tracking과 같은 비디오 추적 과제를 통합할 수 있는 SAM 2++ 모델을 소개합니다. 이 모델은 다양한 타겟 상태(마스크, 바운딩 박스 및 포인트 등)에 대해 일반화된 추적을 가능하게 합니다. 다양한 태스크의 입력을 일반적인 프롬프트 임베딩으로 인코딩하는 태스크 별 프롬프트와 서로 다른 결과를 통합하는 통합 디코더를 도입했습니다.

- **Technical Details**: SAM 2++는 태스크 별 메모리 매칭 전략을 기반으로 하여 서로 다른 타겟 상태를 통합적인 메모리 표현으로 통합합니다. 또한, 'Tracking-Any-Granularity'라는 이름의 대규모 데이터셋을 구축하여 다양한 주석을 제공하고, 서로 다른 타겟 작업에 대한 유니파이드 트래킹을 지원하는 맞춤형 데이터 엔진을 사용하는 방법을 제시합니다.

- **Performance Highlights**: 다양한 벤치마크에 대한 포괄적인 실험을 통해 SAM 2++가 서로 다른 세분화에서 새로운 성능 기준을 수립했으며, 모든 태스크에서 태스크 별 모델을 일관되게 초과하는 성능을 보여주었습니다. 이로써 SAM 2++는 통합적이고 견고한 트래킹 프레임워크를 제공하며, 다양한 세분화에서 효과적인 트래킹을 가능하게 합니다.



### An Explainable Hybrid AI Framework for Enhanced Tuberculosis and Symptom Detection (https://arxiv.org/abs/2510.18819)
Comments:
          16 pages, 3 figures

- **What's New**: 이 연구는 인공지능(AI) 기반의 결핵 스크리닝 도구의 필요성을 강조하며, 조기 발견을 통해 치료의 성공률을 높일 수 있음을 보여줍니다. 저자는 자기 지도학습(self-supervised learning)과 감독학습(supervised learning)을 통합한 티처-스튜던트(Tteacher-student) 프레임워크를 개발하여 흉부 X-선에서 질병과 증상을 효과적으로 탐지하는 방법을 제안합니다. 이 모델은 COVID-19, 결핵, 정상 사례를 구별하는 데 98.85%의 정확도를 기록했으며, 다중 증상 탐지에 대한 매크로 F1 점수는 90.09%로 기존의 기법보다 월등히 높은 성능을 보였습니다.

- **Technical Details**: 연구팀은 U-Net 아키텍처를 통해 질병과 증상을 식별하기 위해 데이터를 처리합니다. 이 과정에서 이미지를 8비트 그레이스케일 포맷으로 변환하고, 품질 관리를 통해 품질이 보장된 세분화(mask) 이미지를 사용합니다. 제안된 모델은 ViT-Small 티처–스튜던트 네트워크 아키텍처를 사용하며, 자기 지도학습 목표를 위해 DINO 기반의 프로젝션 헤드를 포함합니다.

- **Performance Highlights**: 제안된 모델은 튜불라 또는 COVID-19 증상이 있는 데이터를 포함한 다양한 데이터셋에서 강력한 CNN 기법을 초월하는 성능을 기록했습니다. 특히, 섬세한 발견들에 대한 증상 탐지에서 상당한 향상이 있어, 임상에서의 실제 스크리닝 및 분류 작업에 실질적으로 기여할 수 있는 가능성을 보여줍니다. Grad-CAM 시각화를 통한 분석도 모델의 결정적 장치를 해부학적 특징과 일치하게 하는 강점을 나타냅니다.



### A Geometric Approach to Steerable Convolutions (https://arxiv.org/abs/2510.18813)
- **What's New**: 이 논문은 d 차원에서 steerable convolutional neural networks (스티어러블 컨볼루션 신경망)의 새로운 기하학적 유도 방법을 제안합니다. 기존의 추상적이고 집합론적인 접근 방식에 비해, 본 연구는 패턴 매칭의 근본 원리를 기반으로 보다 직관적인 설명을 제공합니다. Clebsch-Gordan 분해와 구형 고유 함수의 출현을 설명하며, 기존 구현보다 향상된 보안을 제공하는 보간 커널을 사용하여 steerable convolution 모듈을 구성하는 혁신적인 방법을 제안합니다.

- **Technical Details**: 제안된 방법론은 전통적인 대수적 기술 대신 기하학적 추론을 강조하여 steerable convolutions의 방정식을 간략화합니다. 2D의 CNN 구조를 바탕으로, 입력 이미지는 연속 함수로 모델링되며 하나의 채널로 표시됩니다. 이 논문은 CNN의 기본 개념을 다시 검토하면서, 작은 학습 가능한 필터를 통해 지역적인 이미지 영역을 비교하는 패턴 매칭 과정을 제시합니다.

- **Performance Highlights**: 실험 결과, 제안된 대안적 기법은 기존 접근 방식보다 뛰어난 성능을 나타내며, 노이즈에 대한 강건성을 보여줍니다. 논문은 이론적으로 보장된 등방성을 적용하여 실세계 이미지 데이터에 대한 적용 가능성과 한계에 대한 깊은 통찰을 제공합니다. 본 연구는 모든 이전 결과를 특별한 경우로 포함하는 통합적이고 엄밀한 이론을 확립하여 steerable neural networks에 대한 이해를 넓히고 있습니다.



### ProCLIP: Progressive Vision-Language Alignment via LLM-based Embedder (https://arxiv.org/abs/2510.18795)
Comments:
          17 pages, 5 fiugres

- **What's New**: 이 논문에서는 ProCLIP라는 새로운 프레임워크를 제안하여 기존 CLIP의 한계를 극복하고자 합니다. 기존 CLIP 모델의 입력 길이가 77 tokens로 제한되어 있어 긴 텍스트 처리와 다국어 이해에 어려움이 있었으며, 이러한 문제를 해결하기 위해 LLM 기반의 embedder로 교체가 시도되었습니다. 그러나 기존 방법은 CLIP의 사전 훈련된 지식을 충분히 활용하지 못했으며, 새로운 알리그먼트를 위한 직접적인 접근이 일반화된 능력을 저하시킬 수 있음을 지적합니다.

- **Technical Details**: ProCLIP은 커리큘럼 학습(curriculum learning)을 기반으로 한 진행적인 비전-언어 정렬(vision-language alignment) 프레임워크입니다. ProCLIP의 과정은 먼저 CLIP 텍스트 인코더에서 LLM 기반 embedder로 지식을 증류하는 것으로 시작하여, LLM과 CLIP 이미지 인코더 간의 초기 정렬을 설정합니다. 그다음, 이미지-텍스트 쌍을 통한 대비 학습(contrastive learning)을 진행하여 정렬을 더욱 개선하고, 자가 증류(self-distillation) 정규화 기법을 통해 과적합(overfitting)을 방지합니다.

- **Performance Highlights**: ProCLIP은 여러 데이터 스케일과 모델 크기에 대해 다양한 작업에서 평가되었으며, 제로샷 분류(zero-shot classification)에서 6.8%에서 13.5%의 향상을 달성했습니다. 또한, 짧은 텍스트 크로스모달 검색(short-text cross-modal retrieval), 긴 텍스트 크로스모달 검색(long-text cross-modal retrieval), 다국어 크로스모달 검색(multilingual cross-modal retrieval) 및 세밀한 이해(fine-grained understanding)와 같은 여러 작업에서도 강력한 성능을 보였습니다. 이처럼 ProCLIP은 기존 CLIP에 비해 일관된 향상을 보여주며, 모델의 일반화 능력을 유지하는 방식으로 성능을 극대화합니다.



### Rebellious Student: A Complementary Learning Framework for Background Feature Enhancement in Hyperspectral Anomaly Detection (https://arxiv.org/abs/2510.18781)
- **What's New**: 최근 제안된 하이퍼스펙트럼 이상 탐지(Hyperspectral Anomaly Detection) 방법은 배경 데이터셋에서 한 번 학습 후, 장면 별 재학습이나 파라미터 조정 없이 보편적으로 배포될 수 있는 뛰어난 효율성과 강인성을 보여주고 있습니다. 본 연구는 이러한 패러다임을 기반으로 스펙트럼(spectral)과 공간(spatial) 정보를 통합한 새로운 'Rebellious Student' 프레임워크를 제안하여 보완적인 특징 학습을 수행합니다. 이 방법은 전통적인 교사-학생 패러다임과는 달리, 공간 브랜치가 스펙트럼 교사와 다르게 학습하여, 교사가 캡처하지 못하는 보완적인 공간 패턴을 배울 수 있도록 합니다.

- **Technical Details**: 본 연구에서는 두 단계의 학습 전략을 채택합니다: 첫 번째로, 역 증류(reverse distillation)를 사용하여 스펙트럼 향상 네트워크가 강력한 배경 스펙트럼 표현을 얻도록 학습합니다. 두 번째로, 공간 네트워크인 반항적인 학생(rebellious student)을 최적화하여 특징 직교성(orthogonality)을 유지하면서 비관련 노이즈를 피하는 복원 충족(reconstruction fidelity)을 보장하는 분산 보존 정규화(variance-preserving regularization)를 사용합니다. 이러한 훈련 과정을 통해, 스펙트럼 및 공간 배경 특징을 향상시켜, 기존 탐지기와 결합 시 파라미터 없는(paremeter-free) 및 재학습 없는(training-free) 이상 탐지를 가능하게 합니다.

- **Performance Highlights**: HAD100 벤치마크에 대한 광범위한 실험 결과, 기존 여러 기준선에 비해 상당한 개선을 확인할 수 있었으며, 여기에는 최소한의 계산 오버헤드가 포함됩니다. 제안된 보완 학습 패러다임의 효과성과 일반성을 확인하며, 다른 장면에서의 교차 장면 실험을 통해 학습된 향상 메커니즘이 스펙트럼 특징을 보다 가우시안(Gaussian)과 판별적인 분포로 재형성하는 것을 보여주었습니다. 이는 통계적 및 학습 기반 탐지 방법 모두에서 향상된 분리 가능성을 제공하며, 파라미터 조정의 부담을 덜어줍니다.



### UltraGen: High-Resolution Video Generation with Hierarchical Attention (https://arxiv.org/abs/2510.18775)
- **What's New**: UltraGen는 기존 저해상도 비디오 생성 모델을 엔드 투 엔드 고해상도 비디오 생성기로 변환할 수 있는 새로운 프레임워크이다. 이 모델은 고급 로컬-글로벌 주의(decomposition) 구조를 통해 의미적 일관성을 보장하면서 지역 내용을 정밀하게 생성할 수 있다. 또한 이 프레임워크는 매우 낮은 계산 비용으로 높은 해상도의 비디오를 생성할 수 있도록 설계되었다.

- **Technical Details**: UltraGen은 전역-지역(attention) 분해를 기반으로 한 계층적 이중 분기 주의 아키텍처를 특징으로 한다. 이 구조는 전반적인 비디오 의미를 포착하는 전역 주의 브랜치와 세부 지역 내용을 생성하는 지역 주의 브랜치로 나뉜다. 또한, 프레임 단위의 합성을 통한 정보 압축과 계층적 교차-창(local attention) 메커니즘을 통해 효율적인 정보 흐름을 보장한다.

- **Performance Highlights**: UltraGen은 기존 모델들보다 월등한 성능을 보여 주며, 처음으로 1080P 및 4K 해상도의 비디오 생성을 가능하게 한다. 실험 결과, UltraGen은 주관적 및 객관적 평가 모두에서 기존의 최첨단 방법들을 초월하는 결과를 나타냈다. 이는 UltraGen이 높은 해상도의 비디오 생성을 위한 실질적이고 확장 가능한 접근법임을 보여준다.



### Detection and Simulation of Urban Heat Islands Using a Fine-Tuned Geospatial Foundation Model for Microclimate Impact Prediction (https://arxiv.org/abs/2510.18773)
Comments:
          10 pages, 9 figures. Accepted at the NeurIPS 2025 Workshop on Tackling Climate Change with Machine Learning

- **What's New**: 이번 연구는 도시 열섬 현상(UHI) 완화를 위한 혁신적인 접근 방식을 제안합니다. 지리 공간 기초 모델(Geospatial Foundation Model, GFM)을 활용하여 고해상도의 온도 데이터 부족 문제를 해결하고, 도시 녹지 공간의 냉각 효과를 실증적으로 검증합니다. 이 모델은 기후 변화에 대한 예측 능력을 바탕으로, 미래 기후 시나리오에서의 토지 표면 온도를 예측할 수 있는 잠재력을 가집니다.

- **Technical Details**: 이 연구는 세 단계의 실험적 워크플로우를 통해 이루어졌습니다. 첫 번째 단계에서는 고해상도 토지 표면 온도(Land Surface Temperature, LST) 이미지를 활용하여 도시 녹지 공간이 온도에 미치는 영향을 정량화했습니다. 두 번째 단계에서는 브라소브(Brașov)를 대상으로 대기 상태 변화에 대한 모델의 외삽 능력을 평가했으며, 세 번째 단계에서는 위성 이미지를 통한 도시 녹화 개입을 시뮬레이션했습니다.

- **Performance Highlights**: 모델 V2는 V1에 비해 더 복잡한 스필오버(spillover) 냉각 현상을 더 잘 표현하며, 모델의 정확도가 크게 향상되었습니다. 특히 평균 절대 오차(Mean Absolute Error, MAE)가 V1의 0.302°C에서 0.199°C로 감소했습니다. 실험 결과는 도시 녹지가 UHI 완화 전략에 매우 중요한 역할을 함을 보여주며, 데이터 부족 지역에서의 활용 가능성을 지니고 있습니다.



### SEAL: Semantic-Aware Hierarchical Learning for Generalized Category Discovery (https://arxiv.org/abs/2510.18740)
Comments:
          Accepted to NeurIPS 2025

- **What's New**: 이 논문에서는 Generalized Category Discovery (GCD) 문제를 다루고 있습니다. GCD는 부분적으로 레이블이 있는 데이터셋을 기반으로 모든 레이블이 없는 이미지를 분류하는 것을 목표로 하며, 기존의 접근 방식들은 일반적으로 단일 수준의 의미론 또는 수동으로 설계된 추상적 계층에 의존합니다. 이러한 한계를 극복하기 위해, 연구진은 자연 발생적이고 쉽게 접근할 수 있는 계층 구조를 기반으로 한 SEmantic-aware hierArchical Learning 프레임워크(SEAL)를 소개합니다.

- **Technical Details**: SEAL은 Hierarchical Semantic-Guided Soft Contrastive Learning 접근 방식을 통해 계층적 유사성을 활용하여 유용한 소프트 네거티브를 생성합니다. 이는 기존의 대조 손실(contrastive losses)이 모든 부정적인 샘플을 동일하게 취급하는 한계를 극복합니다. 또한, Cross-Granularity Consistency (CGC) 모듈을 통해 다양한 수준의 세분성에서의 예측을 정렬하는 기능을 제공합니다.

- **Performance Highlights**: SEAL은 SSB 벤치마크, Oxford-Pet, Herbarium19 데이터셋을 포함한 세분화된 기준에서 최첨단 성능을 달성하며, 조잡한 데이터셋에서도 일반화 능력을 입증합니다. 이 논문은 GCD 작업을 효과적으로 해결하기 위해 세 가지 주요 기여를 합니다: SEAL 프레임워크의 제안, 새로운 CGC 모듈 및 Hierarchical Semantic-guided Soft Contrastive Learning의 개발.



### Moving Light Adaptive Colonoscopy Reconstruction via Illumination-Attenuation-Aware 3D Gaussian Splatting (https://arxiv.org/abs/2510.18739)
- **What's New**: 이 논문에서는 ColIAGS라는 개선된 3D Gaussian Splatting(3DGS) 프레임워크를 제안하여 대장내시경 장면에서 조명 변화에 따른 실시간 시각 합성을 가능하게 합니다. 기존의 3DGS는 조명이 고정되어 있다는 가정을 두고 있었지만, 이 연구는 조명의 거리와 방향을 모두 고려하여 더욱 정확한 3D 재구성을 목표로 하고 있습니다. 이를 통해 기존 방법의 단점을 극복하면서 대장내시경의 현실적인 광학 변동을 효과적으로 처리할 수 있게 되었습니다.

- **Technical Details**: ColIAGS는 두 가지 종류의 조명 감쇠 인자를 도입하여 고전적인 모습 모델링을 개선하고 있습니다. 이 프레임워크는 고차원 뷰 임베딩을 사용하여 기하학적 예측을 향상시키며, 또한 코사인 임베딩 입력을 활용하여 조명 감쇠 솔루션을 암시적으로 생성합니다. 이러한 기술적 접근은 조명 감쇠의 물리적 특성을 모델링하는 데 중요한 역할을 하며, 시각적 정확성을 유지하면서 다양한 광원 조건을 처리할 수 있습니다.

- **Performance Highlights**: ColIAGS는 기존의 최첨단 방법들과 비교했을 때 향상된 렌더링 충실도를 유지하면서 기하학적 정확도를 크게 개선하였습니다. 실험 결과는 새로운 시각 합성에서 PSNR이 2.04 dB 향상되었으며, 깊이 MSE가 78% 감소하는 성과를 보였습니다. 이 연구는 대장내시경 재구성의 효율성을 높이는 데 기여하여, 실시간 응용 프로그램에서의 활용 가능성을 확대하고 있습니다.



### IF-VidCap: Can Video Caption Models Follow Instructions? (https://arxiv.org/abs/2510.18726)
Comments:
this https URL

- **What's New**: 이번 논문은 다중 모달 대형 언어 모델(MLLMs)이 비디오 자막 작성(video captioning)에서 성능을 보였지만, 실제 응용 프로그램에서는 특정 사용자 지침을 따라야 한다는 점을 강조합니다. 이를 해결하기 위해, 우리는 IF-VidCap이라는 새로운 벤치마크를 소개하며, 이는 1,400개의 고품질 샘플을 포함하고 있습니다.

- **Technical Details**: IF-VidCap는 자막의 형식적 정확성(format correctness)과 내용적 정확성(content correctness)이라는 두 가지 측면에서 평가하는 체계적인 프레임워크를 구축합니다. 기존의 비디오 자막 작성이나 일반 지침 따르기 벤치마크와는 차별화되어, 더욱 많은 정보를 제공합니다.

- **Performance Highlights**: 20개 이상의 저명한 모델에 대한 포괄적인 평가 결과, 독점 모델들이 여전히 우위를 점하지만 오픈 소스 솔루션들이 성능을 빠르게 향상하고 있다는 것을 발견했습니다. 또한, 밀집 자막을 위한 전문 모델들이 복잡한 지침에 대해서는 일반 목적의 MLLMs보다 성능이 떨어진 것을 보여주며, 향후 작업에서는 설명의 풍부함과 지침 준수의 정확성 모두를 발전시켜야 한다고 제안합니다.



### SSD: Spatial-Semantic Head Decoupling for Efficient Autoregressive Image Generation (https://arxiv.org/abs/2510.18716)
- **What's New**: 이번 연구에서는 고급 자율 이미지 생성 모델들이 높은 메모리 요구와 계산 비용 문제를 겪고 있다는 점을 강조합니다. 특히 다양한 시각적 토큰들로 인해 발생하는 메모리 한계를 해결하기 위해 KV 캐시 압축을 도입하고, 이를 공간적 지역성과 의미적 싱크(semantic sink)이라는 새로운 주의 현상으로 설명합니다. 새로운 KV 캐시 압축 프레임워크를 통해 메모리 사용량을 5배 줄이고 처리 속도를 6.6배 향상시킴으로써 자료 제약이 있는 하드웨어에서 자율 이미지 생성을 가능하게 합니다.

- **Technical Details**: 이 연구에서는 시각적 오토리그레시브 이미지 생성 과정에서 독특한 주의 헤드의 유형을 구별합니다. 특히, 공간적 지역성(heads)과 의미적 싱크(heads)로 구분하며, 각기 다른 압축 정책을 적용하여 메모리 사용을 최적화합니다. 공간적 지역성 헤드에 대해서는 최근의 토큰을 유지하는 슬라이딩 윈도우를 구현하고, 의미적 싱크 헤드에 대해서는 높은 주목도를 가진 최소 세트의 토큰을 보존하는 전략을 사용합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 KV 캐시 메모리 사용량을 5배 줄이고 128 배치 크기에서 인코딩 처리 속도를 6.6배 향상시키며, 시각적 품질 손실이 미미하다는 것을 보여줍니다. 이를 통해 자원이 제한된 하드웨어에서도 고해상도 이미지 생성을 실현할 수 있는 가능성을 높입니다. 따라서, 이 연구는 자율 이미지 생성의 실용성과 접근성을 향상시키는 데 기여합니다.



### PLANA3R: Zero-shot Metric Planar 3D Reconstruction via Feed-Forward Planar Splatting (https://arxiv.org/abs/2510.18714)
Comments:
          39th Conference on Neural Information Processing Systems (NeurIPS 2025). The project page is available at: this https URL

- **What's New**: PLANA3R는와 함께기존의 방법들과는 달리, 3D 평면 주석 없이도 메트릭 3D 재구성을 가능하게 하는 포즈 프리(Πose-free) 프레임워크입니다. 특히, 이 방법은 Vision Transformers를 사용하여 두 개의 비포즈 이미지에서 3D planar templates을 추출하고, 카메라 위치 추정을 수행하며, geometry learning을 감독합니다. 이러한 접근 방식은 대량의 스테레오 데이터세트를 활용하여 거대한 효율성을 내는 강력한 3D 재구성을 가능하게 합니다.

- **Technical Details**: PLANA3R는 두 개의 비포즈 이미지에서 관계형 카메라 포즈 및 3D 평면 원시(shape)를 추정하는 Transformer 기반의 모델입니다. PlanarSplatting의 차별화 가능한 렌더링 기술을 활용하여, 고해상도 깊이 및 일반 지도에서 시각적 데이터를 수집하고 학습합니다. 이렇게 생성된 평면 원시를 사용하여 indoor scene의 기하학적 정보를 효율적으로 재구축하며, 이는 높은 정확도로 일관된 결과를 제공합니다.

- **Performance Highlights**: PLANA3R는 다양한 실내 장면 데이터셋에서 메트릭 감독을 통해 검증되었으며, 다른 도메인의 실내 환경에 강한 일반화 성능을 입증하였습니다. 이 방법은 depth 추정, 3D 표면 재구성 및 상대적인 포즈 추정을 포함한 다양한 작업에서 우수한 성과를 보여줍니다. 특히 평면 기반의 표현 방식 덕분에 효과적인 인스턴스 단위의 평면 세분화가 가능해져, 3D 재구성의 의미론을 풍부하게 합니다.



### A Renaissance of Explicit Motion Information Mining from Transformers for Action Recognition (https://arxiv.org/abs/2510.18705)
Comments:
          accepted by Pattern Recognition. We have been always curious to see whether our designs could be beneficial in other scenarios, such as embedding it into the DiT model or 3D-VAE for video generation. If you are interested in it, why not give it a shot?

- **What's New**: 이 논문에서는 기존의 transformer 기반의 행동 인식 기술에 중요한 발전을 제공하는 Explicit Motion Information Mining 모듈(EMIM)을 제안합니다. EMIM은 움직임을 효과적으로 모델링할 수 있는 특성을 기존 transform 구조에 통합하여, 특히 모션에 민감한 데이터셋에서 향상된 성능을 보여줍니다. 현재까지 행동 인식에서 transformer가 가진 한계를 극복하는 새로운 디자인을 통해, 놀라운 이정표를 정립하고 있습니다.

- **Technical Details**: EMIM은 비용 볼륨(cost volume) 스타일에서 선호하는 affinity 매트릭스를 구축합니다. 이때, 쿼리 기반의 이웃 영역에서 다음 프레임의 텍스트에서 키 후보 토큰이 샘플링되어, 적절한 모션 특징으로 변환됩니다. 이 방법은 기존 self-attention 모듈의 맥락 집계 능력을 유지하며, 새로운 모션 모델링 능력을 발전시킵니다.

- **Performance Highlights**: 제안된 방법은 네 개의 널리 사용되는 데이터셋에서 유효성을 검증하였으며, 특히 motion-sensitive 데이터셋인 Something-Something V1 & V2에서 기존의 최첨단 기술보다 뛰어난 성능을 noted합니다. 이러한 성능은 EMIM이 모션 모델링의 강력한 능력을 발휘할 수 있음을 강조합니다.



### Exploring a Unified Vision-Centric Contrastive Alternatives on Multi-Modal Web Documents (https://arxiv.org/abs/2510.18703)
Comments:
          Project page: this this https URL

- **What's New**: 본 논문에서는 복잡한 멀티모달 웹 문서를 처리하기 위한 Vision-Centric Contrastive Learning (VC2L) 프레임워크를 제안합니다. VC2L은 텍스트와 이미지뿐만 아니라 이들의 조합을 하나의 비전 트랜스포머를 통해 처리하며, 픽셀 공간에서 모든 입력 변수를 다루어 OCR(Optical Character Recognition)이나 텍스트 토크나이제이션 경우를 제거합니다.

- **Technical Details**: VC2L은 문서의 연속적인 멀티모달 스니펫을 활용하여 대조학습(objetive)를 수행합니다. 문서의 내재적인 일관성을 이용해, 명시적으로 짝을 이룬 이미지-텍스트 데이터가 없어도 서로의 임베딩을 유사하게 만드는 방향으로 학습합니다. 이를 통해 복잡한 문서 내의 다양한 멀티모달 입력을 보다 쉽게 처리할 수 있게 됩니다.

- **Performance Highlights**: VC2L은 제안한 벤치마크와 M-BEIR, MTEB와 같은 기존 데이터셋에서 CLIP 스타일 모델에 비해 경쟁력 있는 성능을 보여줍니다. 이번 연구는 멀티모달 웹 데이터가 대조학습의 가치 있는 훈련 자원으로 사용될 수 있음을 강조하며, VC2L의 통합된 비전 중심 접근법의 확장 가능성을 나타냅니다.



### UniGenBench++: A Unified Semantic Evaluation Benchmark for Text-to-Image Generation (https://arxiv.org/abs/2510.18701)
Comments:
          Project page: this http URL

- **What's New**: 본 연구에서는 T2I (Text-to-Image) 생성 모델의 평가를 위한 새로운 기준선, UniGenBench++를 도입합니다. 기존 벤치마크의 한계를 벗어나 다양한 프롬프트 시나리오와 다국어 지원을 포함하여 보다 포괄적이고 세밀한 평가를 가능하게 합니다. 600개의 프롬프트로 구성된 이 기준선은 모델의 의미적 일관성을 10개의 주요 및 27개의 하위 평가 기준을 통해 종합적으로 탐색합니다.

- **Technical Details**: UniGenBench++는 5개의 주요 프롬프트 테마와 20개의 하위 테마로 구성되어 있으며, 영어와 중국어로 제공되는 각 프롬프트는 짧은 형식과 긴 형식으로 제공되어 모델의 언어 및 프롬프트 길이에 대한 민감도를 체계적으로 평가할 수 있습니다. 평가 모델인 Gemini-2.5-Pro를 활용하여 각 프롬프트의 의미적 요구사항이 얼마나 충족되고 있는지를 분석하고 점수를 부여하는 점진적 평가 파이프라인을 제안합니다.

- **Performance Highlights**: T2I 생성 모델의 포괄적인 벤치마킹을 수행한 결과, 폐쇄형 모델(GPT-4o, FLUX-Kontext-Max 등)과 개방형 모델(Qwen-Image, HiDream 등) 모두 스타일과 세계 지식 관련 프롬프트에서는 높은 성능을 보였으나, 논리적 추론이 필요한 경우에는 지속적으로 어려움을 겪는 것으로 나타났습니다. 이러한 결과는 모델들이 비즈니스 요구에 대한 실제 사용 조건 아래에서 성능 저하를 겪을 수 있음을 시사합니다.



### MoGA: Mixture-of-Groups Attention for End-to-End Long Video Generation (https://arxiv.org/abs/2510.18692)
Comments:
          15 pages, 12 figures

- **What's New**: 이번 논문에서는 Mixture-of-Groups Attention (MoGA)라는 효율적인 스파스 어텐션 메커니즘을 제안합니다. 이 방식을 통해 입력 토큰을 블록 수준의 추정 없이도 정밀하게 매칭할 수 있게 되었으며, 긴 범위 상호작용을 효과적으로 지원합니다. 또한, MoGA는 FlashAttention 및 시퀀스 병렬성과 통합되어 현대적인 어텐션 스택과 원활하게 결합됩니다.

- **Technical Details**: MoGA는 간단하고 효율적인 동적 토큰 라우팅 솔루션으로, 토큰들을 특정 그룹에 할당하기 위해 경량의 라우터를 사용합니다. 이 방식을 통해 각 토큰이 정해진 그룹 내에서 고전적인 어텐션을 이용해 상호작용할 수 있게 되며, 이로 인해 대규모 컨텍스트를 모델링할 수 있습니다. 또한 MoGA는 스페이셜-템포럴 윈도우 어텐션과 결합하여 긴 거리 일관성과 지역적 충실도를 조화롭게 제공합니다.

- **Performance Highlights**: 제안된 MoGA 기반 모델은 24fps로 480p 멀티샷 비디오를 분 단위로 생성할 수 있으며, 약 580k의 토큰 길이를 지원합니다. 다양한 비디오 생성 작업에 대한 포괄적인 실험을 통해 본 접근 방법의 효과가 검증되었으며, 기존 스파스 어텐션 기법들에 비해 일관된 성능 향상이 나타났습니다.



### Beyond the Pipeline: Analyzing Key Factors in End-to-End Deep Learning for Historical Writer Identification (https://arxiv.org/abs/2510.18671)
Comments:
          Published in The 12th IEEE International Conference on Data Science and Advanced Analytics (DSAA), 2025

- **What's New**: 이 논문은 역사적 필기자 식별(Historical Writer Identification, HWI)의 성능에 영향을 미치는 다양한 요소를 조사합니다. HWI는 다양한 필기 스타일, 문서 열화, 그리고 한 필기자당 적은 수의 레이블 샘플로 인해 여전히 도전적인 과제입니다. 전통적인 방법들은 수작업으로 구성된 이미지 처리 및 클러스터링 기법에 의존하지만, 논문에서는 이러한 방법과 대조적으로 end-to-end 딥러닝 모델이 문서 이미지에서 직접 피처를 학습하는 방안을 제안합니다.

- **Technical Details**: 이 연구는 HWI 파이프라인을 다단계 과정으로 구성하여 사전 처리, 피처 추출(모델 선택 및 손실 함수), 그리고 후처리의 세 가지 핵심 구성 요소를 포함하고 있습니다. 논문에서는 다양한 사전 처리 전략, 백본 아키텍처 및 후처리 기법이 전체 성능에 미치는 영향을 조사하며, 특히 잇셈 없는(writer seen) 환경에서도 시스템의 효과를 비교합니다. 실험 결과, 많은 구성들이 저수준 시각 피처의 취약한 캡처와 높은 콘텐츠 노이즈 민감도 때문에 성능이 떨어지는 경향이 있음을 보여줍니다.

- **Performance Highlights**: 연구 결과, 간단한 디자인을 사용해도 최상의 시스템과 유사한 성과를 내는 하나의 end-to-end 설정을 확인했습니다. 그러나 대부분의 설정은 낮은 수준의 시각적 피처, 일관되지 않은 패치 표현 및 높은 콘텐츠 노이즈에 대한 민감도로 인해 전반적으로 성능이 저조합니다. 이 연구는 HWI의 강력하고 안정적인 end-to-end 시스템을 구축하는 데 있어 주요 과제를 강조하고 향후 성능 개선을 위한 디자인 선택에 대한 통찰을 제공합니다.



### Image augmentation with invertible networks in interactive satellite image change detection (https://arxiv.org/abs/2510.18660)
- **What's New**: 이번 논문에서는 능동 학습(active learning)에 기반한 새로운 위성 이미지 변화 탐지(change detection) 알고리즘을 제안합니다. 이 프레임워크는 질문-답변 모델을 활용하여 소수의 이미지에 대한 레이블을 사용자(oracle)에게 쿼리하고, 사용자의 응답에 따라 변화 탐지 모델을 동적으로 업데이트합니다. 해당 체계의 주요 기여는 비가역적(invertible) 네트워크를 통해 변환을 수행하여 더 적은 데이터로 효과적으로 변화 탐지를 가능하게 한다는 점입니다.

- **Technical Details**: 이 알고리즘은 두 개의 등록된 위성 이미지에서 패치들을 비교하여 라벨을 추론하는 방식으로 작동합니다. 사용자가 레이블을 제공하는 과정을 통해 최소한의 수작업 레이블링으로도 고성능 모델을 구축할 수 있습니다. 특히, 비가역적 네트워크를 활용하여 다차원 데이터의 다양성을 확보하고, 이 공간 내에서 단순한 선형 변환을 통해 데이터를 증강하는 방법이 주요 특징입니다.

- **Performance Highlights**: 실험 결과, 제안된 알고리즘이 기존 방법들보다 우수한 성능을 보였으며, 이는 데이터의 비가역적 증강 기법 덕분에 가능했습니다. 변화 탐지의 정확도를 높이기 위해 선택된 소규모 샘플을 기반으로 한 이 세련된 접근 방식은 사용자 맞춤형 모델을 구축하는 데 기여합니다. 전체적으로 해당 연구는 위성 이미지 변화 탐지 분야에서의 사용자 중심의 효율적인 해결책을 제시하고 있습니다.



### Binary Quadratic Quantization: Beyond First-Order Quantization for Real-Valued Matrix Compression (https://arxiv.org/abs/2510.18650)
Comments:
          Accepted to NeurIPS 2025

- **What's New**: 이 논문은 새로운 행렬 양자화 방법인 Binary Quadratic Quantization (BQQ)를 제안합니다. 전통적인 첫 번째 차수 양자화 접근 방식인 균일 양자화(Uniform Quantization)와 이진 코딩 양자화(Binary Coding Quantization)와는 달리, BQQ는 이진 이차 표현의 표현력을 활용하며 매우 압축된 데이터 형식을 유지합니다. 이 접근 방식은 다양한 행렬 데이터의 압축에 있어 메모리 효율성과 복원 오류 간의 우수한 균형을 달성합니다.

- **Technical Details**: BQQ는 이진 변수의 이차 조합을 사용하여 행렬을 표현하며, 각 이진 행렬에 대해 독립적인 스케일링(스케일링) 계수를 할당합니다. 이는 이진 행렬 곱의 합으로 타겟 행렬을 표현하므로 비선형 근사를 가능하게 하며 매우 컴팩트한 데이터 형식을 유지할 수 있습니다. BQQ는 NP-하드 최적화 문제를 다루며, 이를 풀기 위한 효율적인 솔루션을 다항 비제한 이진 최적화(Polynomial Unconstrained Binary Optimization) 및 볼록 이차 프로그래밍(Convex Quadratic Programming)을 기반으로 개발하였습니다.

- **Performance Highlights**: 실험 결과, BQQ는 다양한 행렬 데이터의 압축에 있어 메모리 사용과 양자화 오류 간의 우수한 균형을 달성하며, Vision Transformer(ViT) 기반 모델의 후처리 양자화(PTQ)에서 최첨단 성능을 제공합니다. 예를 들어, BQQ는 ImageNet 데이터셋에서 캘리브레이션 기반 및 데이터 없음 시나리오에서 각각 최대 2.2% 및 59.1% 성능 향상을 보였습니다. 이러한 결과는 BQQ가 효율적인 행렬 근사 및 신경망 압축에 매우 효과적임을 강조합니다.



### ε-Seg: Sparsely Supervised Semantic Segmentation of Microscopy Data (https://arxiv.org/abs/2510.18637)
Comments:
          10 pages main text, 17 pages total

- **What's New**: 새로운 연구에서, 우리는 생물학적 샘플의 전자 현미경(EM) 이미지를 위한 희소 감독(semi-supervised) 의미 분할 방법인 {m{m{	ext{epsilon}}}-Seg를 소개합니다. 이 방법은 계층적 변별 오토인코더(HVAE)를 기반으로 하며, 중심 영역 마스킹(center-region masking)과 희소 레이블 대비 학습(sparse label contrastive learning)을 도입하여 효과적인 세분화를 가능하게 합니다. {m{m{	ext{epsilon}}}-Seg는 복잡한 생물학적 이미지 데이터에서 경쟁력 있는 결과를 얻는 데 도움을 줍니다.

- **Technical Details**: 제안된 방법은 각 의미 클래스에 대해 미리 정해진 가우시안 영역을 가지는 가우시안 혼합 모델(GMM)을 사용하며, 이는 계층적 VAE(HVAE) 구조의 변형입니다. 또한, 대조 손실(contrastive loss)을 추가하여 잠재 인코딩(latent encoding)이 의미 유사성에 따라 그룹화되도록 합니다. 최종적으로, 기존 클러스터링 대신 MLP 의미 분할 헤드를 사용하여 잠재 인코딩에서 직접 클래스 레이블을 예측하는 방식으로 정확도와 실행 시간을 개선합니다.

- **Performance Highlights**: 실험 결과, {m{m{	ext{epsilon}}}-Seg는 제한된 라벨(0.05% 이하)로도 경쟁력 있는 분할 성능을 보여 주며, 생물학적 조직의 두 가지 고밀도 EM 데이터셋에서 우수한 결과를 얻었습니다. 또한, 이 방법은 형광 현미경 데이터에도 적용 가능하다는 것을 입증하며, 기존 방법들보다 좋은 성능을 발휘합니다. 이 연구는 EM 이미지 세분화의 새로운 가능성을 제시합니다.



### C-SWAP: Explainability-Aware Structured Pruning for Efficient Neural Networks Compression (https://arxiv.org/abs/2510.18636)
Comments:
          10 pages, BMVC2025

- **What's New**: 본 논문에서는 고전적인 프루닝(pruning) 기법을 개선한 새로운 일회성 구조 프루닝(framework이자 C-SWAP)을 제안하고 있습니다. 이 방법은 설명 가능한 AI(explainable AI)를 사용하여 모델의 성능을 보존하면서 파라미터 수를 크게 줄일 수 있도록 돕습니다. 특히, 모델의 예측과 구조 간의 인과적 관계를 활용하여 효율적인 프루닝을 수행함으로써, 더 적은 리소스로도 높은 성능을 유지할 수 있습니다.

- **Technical Details**: C-SWAP는 자연스러운 기계 해석(auto interpretability) 연구에 영감을 받아 설계되었습니다. 이 방법은 각 뉴런(채널)을 중요, 중립, 해로운 세 가지 등급으로 분류하고 해당 뉴런과 연관된 가중치를 변경하여 원인 효과(causal effect)를 계산합니다. 이를 통해 통계적 임계값을 사용하여 프루닝할 뉴런을 선택하고, 성능 손실을 최소화하는 방향으로 진행됩니다.

- **Performance Highlights**: 실험을 통해 C-SWAP가 여러 기본 프루닝 기법들보다 뛰어난 성능을 발휘하는 것을 보여주었습니다. 특히, CNN 및 비전 트랜스포머 모델에서의 성능 저하 없이 모델 크기를 효과적으로 줄일 수 있었습니다. 추가적인 성능 조정 없이도 다양한 복잡한 아키텍처에 적용 가능한 가능성을 가지고 있습니다.



### Think with 3D: Geometric Imagination Grounded Spatial Reasoning from Limited Views (https://arxiv.org/abs/2510.18632)
Comments:
          12 pages, 4 figures

- **What's New**: 이 논문에서는 3D 공간 관계 이해의 어려움을 해결하기 위해 새로운 프레임워크인 3DThinker를 제안합니다. 이는 VLM (Vision-Language Model)에서 3D mentaling을 사용하여 제한된 2D 이미지를 기반으로 3D 기하학을 직접 학습할 수 있게끔 합니다. 3DThinker는 복잡한 데이터 레이블이나 외부 모델에 의존하지 않고도 3D 표현을 통합하여 사고할 수 있는 능력을 제공합니다.

- **Technical Details**: 3DThinker의 훈련 과정은 두 단계로 구성되어 있습니다. 첫 번째 단계에서는 VLM이 생성한 3D 잠재 표현을 3D 기본 모델과 정렬하는 감독 학습(Supervised Learning)을 수행합니다. 두 번째 단계에서 우리는 보상 기반 신호에 따라 전체 샘플링 경로를 최적화하여 3D mentaling을 정제하는 강화 학습(Reinforcement Learning)을 진행합니다.

- **Performance Highlights**: 다양한 벤치마크에서의 상세 실험 결과, 3DThinker는 강력한 기준선 모델들보다 일관되게 우수한 성능을 보였습니다. 이 모델은 다양한 VLM 기반에서 잘 일반화되며, 3D 표현을 해석할 수 있는 새로운 관점을 제시하였습니다.



### CovMatch: Cross-Covariance Guided Multimodal Dataset Distillation with Trainable Text Encoder (https://arxiv.org/abs/2510.18583)
Comments:
          NeurIPS 2025

- **What's New**: 이번 논문에서는 효율적인 대규모 비전-언어 모델 훈련을 위한 소규모 이미지-텍스트 쌍을 합성할 수 있는 멀티모달 데이터셋 증류(multimodal dataset distillation) 방법인 CovMatch를 제안합니다. 기존 방식으로는 텍스트 인코더를 고정하고 이미지 인코더와 텍스트 프로젝션 레이어만 업데이트하여 성능을 제한적인 결과로 남기게 됩니다. CovMatch는 두 인코더에 대한 공동 최적화를 가능하게 하여 더욱 강력한 크로스-모달 정렬(cross-modal alignment)을 지원하며 성능을 향상시킵니다.

- **Technical Details**: CovMatch는 실제와 합성된 특징의 크로스 공분산(cross-covariance)을 정렬하며, 각 모달리티 내의 특징 분포를 정규화하는 것을 포함합니다. 멀티모달 데이터셋 증류를 위한 bi-level optimization 프레임워크를 단순화하여 높은 메모리와 연산 비용을 피하며, 고정된 인코더를 통해 선형 프로젝션 레이어만 최적화합니다. 이러한 접근 방식은 효율적인 외부 최적화를 가능하게 하고, 인코더 업데이트는 소규모의 실제 이미지-텍스트 쌍을 통해 이루어지며 정렬 통계가 최신성을 유지합니다.

- **Performance Highlights**: CovMatch는 Flickr30K와 COCO 벤치마크에서 이미지-텍스트 검색 작업을 평가한 결과, 기존 최첨단 멀티모달 증류 방법보다 일관되게 우수한 성능을 기록했습니다. 특히, 단 500개의 합성 쌍만으로도 Flickr30K에서 6.8%, COCO에서 6.1%의 평균 검색 정확도 향상을 달성했습니다. 이러한 성장은 CovMatch의 성능 확장을 저해하지 않으면서 이미지 및 텍스트 인코더를 공동 최적화할 수 있는 능력 덕분입니다.



### Kaleido: Open-Sourced Multi-Subject Reference Video Generation Mod (https://arxiv.org/abs/2510.18573)
Comments:
          11 pages, 6 figures

- **What's New**: Kaleido는 여러 참조 이미지에 기반하여 주제와 일치하는 비디오를 생성하는 S2V(Subject-to-Video) 생성 프레임워크를 제안합니다. 기존의 S2V 모델들은 다중 주제의 일관성을 유지하고 배경을 효과적으로 분리하는 데 부족함이 있었으나, Kaleido는 이러한 문제를 해결하기 위해 개선된 데이터 구성 및 참조 이미지 통합 방법을 도입했습니다. Reference Rotary Positional Encoding(R-RoPE) 기술을 통해 다양한 참조 이미지를 안정적으로 처리하고 일관성을 높였습니다.

- **Technical Details**: Kaleido의 데이터 구성 파이프라인은 저품질 샘플 필터링 및 다양한 데이터 합성을 포함하여 일관성을 유지하는 훈련 데이터를 생성합니다. R-RoPE는 주제 토큰에 회전 위치 인코딩(rotary position encoding)을 도입하여 다수의 참조 이미지로부터 효율적으로 정보를 통합할 수 있도록 설계되었습니다. 이로 인해 다중 이미지 및 다중 주제에 대한 S2V 일관성이 개선되고 컴퓨테이셔널 효율성도 보장됩니다.

- **Performance Highlights**: Kaleido는 다양한 벤치마크에서 기존 방법보다 뛰어난 성능을 보여주었으며, 주제 일관성(subject fidelity)과 배경 분리(background disentanglement)에서 우수함을 입증했습니다. 이러한 성과는 특히 상업적 시스템인 Vidu와 Kling과 같은 비공식적 모델과 비교할 때 두드러지며, 앞으로 S2V 생성 모델의 발전에 큰 기여할 것으로 기대됩니다.



### Descriptor: Occluded nuScenes: A Multi-Sensor Dataset for Evaluating Perception Robustness in Automated Driving (https://arxiv.org/abs/2510.18552)
- **What's New**: 이 논문은 자동 주행 시스템에서 신뢰할 수 있는 인식(perception) 기능을 개발하기 위한 새로운 데이터셋 ‘Occluded nuScenes Dataset’을 소개합니다. 이 데이터셋은 여러 센서 모달리티에서 제어된, 매개변수화된 파손(occlusions)을 통해 성능 평가를 가능하게 하며, 기존의 nuScenes 벤치마크를 확장했습니다. 특히, 카메라, 레이더(radar), LiDAR 센서를 위한 다중 센서 occlusion 데이터셋을 최초로 제공합니다.

- **Technical Details**: Occluded nuScenes Dataset은 원래의 nuScenes 데이터셋에서 파생된 것으로, 복잡한 도시 환경에서 수집된 동기화된 멀티 모달 기록을 포함합니다. 이 데이터셋에서는 6개의 카메라, 5개의 레이더, 32채널 LiDAR를 사용하여 다양한 날씨 및 조명 조건에서의 데이터를 제공합니다. 새로운 occlusions는 카메라 이미지를 위한 네 가지 유형의 시각적 장애물과 레이더 및 LiDAR를 위한 매개변수화된 스크립트를 포함합니다.

- **Performance Highlights**: 이 자원은 부분 센서 고장 및 환경 간섭 하에서 인식 모델의 일관된 평가를 지원합니다. 다양한 장애물 유형을 통해 연구자들은 자동 주행 시스템의 강건성과 안전성을 분석할 수 있게 되며, 실제 주행 조건에서의 신뢰성을 높이는 데 기여할 수 있습니다. 자동 주행 분야에서 인식 및 융합 아키텍처의 성능을 개선할 수 있는 기회를 제공합니다.



### GBlobs: Local LiDAR Geometry for Improved Sensor Placement Generalization (https://arxiv.org/abs/2510.18539)
Comments:
          1st place at the IROS'25 RoboSense Challenge, Track #3: Cross-Sensor Placement 3D Object Detection

- **What's New**: 이번 기술 보고서는 RoboSense 2025의 Track 3에서 3D 객체 탐지에 대한 최첨단 성능을 발휘한 솔루션을 소개합니다. 본 연구는 GBlobs라는 지역 점 구름(feature descriptor)을 사용하여 다양한 LiDAR 구성에서 모델의 일반화 능력을 극대화하는 데 중점을 두었습니다. 기존의 LiDAR 기반 3D 탐지기가 절대적 카르테시안(coordination) 좌표로 학습됨에 따라 발생하는 '기하학적 단축(geometric shortcut)' 문제를 해결합니다.

- **Technical Details**: 본 연구는 점 구름의 지역 기하학(local geometric information)을 활용하여 절대 위치의 의존도가 낮은 GBlobs 표현을 통해 객체의 형상과 특징을 학습하게 유도합니다. GBlobs는 최소 3개의 점을 요구하며, 데이터의 희소성 때문에 발생할 수 있는 문제를 해결하기 위해 기존의 전세계 카르테시안 좌표를 사용하는 보조 탐지 모델을 도입하였습니다. 두 모델은 독립적으로 처리하고, Test-Time Augmentation(TTA) 기법을 통해 예측의 신뢰성을 높이고 교차해야 합니다.

- **Performance Highlights**: 이번 연구는 RoboSense Challenge 2025 Track 3에서 1위를 기록하였으며, 지역 기하적 특징을 활용한 모델 일반화 가능성에 대한 잠재력을 강하게 보여줍니다. GBlobs 기반 모델은 30m 이내의 객체를 효과적으로 탐지하며, 그 이상의 거리에서는 전 세계 좌표 모델의 예측을 사용하여 높은 성능을 유지합니다. 실험 결과는 모든 종류의 점 밀도와 거리에서 높은 성능을 달성하였음을 나타냅니다.



### RayPose: Ray Bundling Diffusion for Template Views in Unseen 6D Object Pose Estimation (https://arxiv.org/abs/2510.18521)
- **What's New**: 본 논문에서는 전통적인 템플릿 기반 객체 자세 추정 방식을 레이 정렬 문제로 재정의하여, 다수의 포즈가 지정된 템플릿 이미지로부터 학습하여 비포즈 쿼리 이미지와 정렬할 수 있도록 하는 새로운 접근법을 제안합니다. 확산 모델(difussion model)과 조건부 분포 모델링 가능성을 활용하여, 기존의 방법보다 정확하게 6D 객체 자세를 추정합니다. 또한, 템플릿 샘플링을 통해 성능을 향상시킬 수 있는 코스-투-파인(coarse-to-fine) 훈련 전략을 적용하였습니다.

- **Technical Details**:  레이 기반의 포즈 맵을 사용하여 6D 객체 포즈를 표현하며, 객체 중심 카메라 레이 및 부피 불변 변환 추정(scale-invariant translation estimation)을 통해 객체 회전 및 번역을 재파라미터화합니다. 이러한 구성은 확산 기반 훈련의 장점을 활용하고, 템플릿 이미지를 사용할 수 있는 이점을 가지고, 시각적 데이터로부터 적확한 포즈 추론을 가능하게 합니다. 연구 방법론은 다중 시점(diffusion-based multi-view) 특성으로 인해 3D 공간 상에서 쿼리와 템플릿 간의 상관관계를 보다 효과적으로 포착합니다.

- **Performance Highlights**: 저자들은 다양한 데이터셋에 대한 광범위한 실험 결과를 통해 제안된 방법이 기존의 최첨단(state-of-the-art) 기법들과 비교할 때 경쟁력 있는 성능을 보인다고 주장합니다. 연구는 ef적이고 정확하게 정렬된 6D 객체 자세 예측을 제공하며, 특히 보지 못한 객체에 대한 연산에서 뛰어난 일반화 능력을 보여줍니다. 많은 변수를 수집하여, 다양한 알고리즘이 서로 어떻게 작용하는지를 정량적으로 평가하는 체계적인 분석을 통해 설계 선택의 타당성을 검증하였습니다.



### DWaste: Greener AI for Waste Sorting using Mobile and Edge Devices (https://arxiv.org/abs/2510.18513)
Comments:
          8 pages, 8 figures

- **What's New**: 최신 연구에서는 스마트폰 및 엣지 디바이스를 위한 실시간 폐기물 분류를 지원하는 DWaste라는 컴퓨터 비전 기반 플랫폼을 개발했습니다. 이 플랫폼은 오프라인 기능을 갖추고 있으며, 딥러닝(Deep Learning)과 경량 객체 탐지(Object Detection) 모델 사용하여 효율적으로 자원 소비를 줄이는 데 중점을 두었습니다. 특히, 모델 양자화(Model Quantization)를 통해 VRAM 사용량과 모델 크기를 최대 75%까지 줄였습니다.

- **Technical Details**: 연구에는 생물학적, 판지, 유리, 금속, 종이, 플라스틱, 쓰레기 등 7개 주요 카테고리의 폐기물 데이터셋이 사용되었습니다. 데이터셋은 사용자 제출 및 DWaste 플랫폼을 통해 수집되었으며, 총 11,163개의 이미지와 19,700개의 바운딩 박스로 구성되어 있습니다. EfficientNetV2S/M, MobileNet, ResNet50/101 등의 분류 모델과 YOLOv8n, YOLOv11n 등의 객체 탐지 모델이 벤치마크 테스트에 사용되었습니다.

- **Performance Highlights**: 실험 결과, EfficientNetV2S와 M은 각각 95-96%의 높은 정확도를 보였지만, 그 크기와 탄소 배출량으로 인해 컴퓨팅 자원이 많이 소모되었습니다. 반면, YOLOv11n은 75% 이상의 평균 정확도(mAP)를 유지하면서 최소의 자원 소모로 가장 빠른 추론 속도를 보이며, 엣지 디바이스에 최적화된 모델로 입증되었습니다. 이러한 최적화된 모델은 DWaste 모바일 앱과 엣지 디바이스에서 실시간 폐기물 탐지에 성공적으로 배포되었습니다.



### Zero-Shot Vehicle Model Recognition via Text-Based Retrieval-Augmented Generation (https://arxiv.org/abs/2510.18502)
Comments:
          Accepted by The 38th Conference of Open Innovations Association FRUCT, 2025

- **What's New**: 이 논문에서는 최신 차량 모델 인식에서 기존 모델들이 새로운 모델에 적응하는데 어려움을 겪는 문제를 해결하기 위해 새로운 접근 방식을 제안합니다. Contrastive Language-Image Pretraining (CLIP) 모델의 한계를 극복하기 위해, Retrieval-Augmented Generation (RAG)와 비전 언어 모델(Vision Language Models, VLMs)을 통합한 제로샷(Zero-shot) 인식 파이프라인을 개발하였습니다. 이 시스템은 차량 이미지를 텍스트로 변환하고, 텍스트 기반의 추론을 통해 차량의 메이크(make)와 모델(model)을 식별할 수 있도록 설계되었습니다.

- **Technical Details**: 제로샷 차량 모델 인식 방식을 통해 입력 이미지를 처리하고 가장 가능성 높은 레이블을 예측하는 새로운 메커니즘을 소개합니다. 이 과정에서는 비전-언어 인코더(Ev)와 텍스트 데이터베이스를 비교하여 관련 정보를 검색하는 단계가 포함됩니다. RAG의 틀은 외부 지식에 기반한 추론을 가능케 하며, 새로운 차량 모델에 대한 텍스트 설명 업데이트를 통해 시스템의 확장성을 보장합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 CLIP의 기본 성능을 기준으로 차량 인식을 거의 20% 향상시키는 것으로 나타났습니다. 연구에 사용된 데이터셋은 최근 출시된 차량 모델로 구성되어 있어, 진정한 제로샷 평가 시나리오를 제공합니다. 이 방식은 새로운 차량 모델의 효과적인 인식을 가능하게 하며, 지능형 교통 시스템을 위한 실제 적용 가능성을 입증하고 있습니다.



### Mono4DGS-HDR: High Dynamic Range 4D Gaussian Splatting from Alternating-exposure Monocular Videos (https://arxiv.org/abs/2510.18489)
Comments:
          Project page is available at this https URL

- **What's New**: Mono4DGS-HDR는 최초로 서로 다른 노출로 촬영된 비디오를 바탕으로 렌더링 가능한 4D HDR 장면을 복원하는 시스템입니다. 이 연구는 Gaussian Splatting에 기반한 통합된 두 단계 최적화 접근 방식을 제시하여, 카메라 포즈에 대한 의존성을 없애고 내구성 있는 초기 HDR 비디오 복원을 가능하게 합니다. 특히, 모노크롬 LDR 비디오에서 4D HDR 장면을 효과적으로 복구하는 문제를 다루며, 기존에 연구된 적 없는 새로운 초점을 맞추었습니다.

- **Technical Details**: Mono4DGS-HDR는 두 단계 최적화 절차를 사용하는 4D HDR 복원 시스템으로, 첫 번째 단계에서 동적 HDR Gaussian을 학습하고 이것을 세계 공간으로 변환합니다. 이후 두 번째 단계에서는 카메라 포즈 및 세계 Gaussian을 공동으로 최적화하여 HDR 장면을 더욱 정밀하게 복원합니다. 또한, 플로우 가이드 광도 손실을 포함한 시간적 광도 정규화 전략을 도입하여 HDR의 시간적 안정성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, Mono4DGS-HDR는 기존 4D 복원 시스템에서 각종 대체 솔루션을 시간과 품질 면에서 크게 능가함을 보여주었습니다. 새로운 평가 벤치마크를 통해 이 방법이 기존 기술 대비 압도적인 성능 향상을 이뤄냈음을 확인할 수 있었습니다. 이러한 성과는 HDR 비디오 복원 분야에서 추가적인 연구 가능성을 제공하는 중요한 전환점이 됩니다.



### Vision Foundation Models Can Be Good Tokenizers for Latent Diffusion Models (https://arxiv.org/abs/2510.18457)
Comments:
          Code and models available at: this https URL

- **What's New**: 최근 Latent Diffusion Models (LDMs)의 성능은 시각적 토크나이저의 품질에 크게 의존하고 있습니다. 본 논문에서는 Vision Foundation Models (VFMs)을 통한 증류(distillation) 방법의 근본적인 한계를 지적하고, 새로운 접근법인 Vision Foundation Model Variational Autoencoder (VFM-VAE)를 제안합니다. VFM-VAE는 기존의 방법보다 더 직접적으로 고품질 이미지를 재구성할 수 있으며, 증류로 인한 정보 손실 없이 멀티 스케일(latent fusion) 및 점진적 해상도 복원(blocks)을 적용합니다.

- **Technical Details**: LDM은 일반적으로 Variational Autoencoder (VAE)를 활용하여 고차원 이미지를 압축된 잠재 공간(latent space)으로 변환하는 두 단계 프레임워크입니다. VFM-VAE는 기존의 VAE 디코더를 두 가지 주요 혁신, 즉 Multi-Scale Latent Fusion 및 Progressive Resolution Reconstruction blocks으로 재설계하여, 픽셀 수준의 정확성을 보장하며 개념적으로 음소거된 VFM 인코더를 활용합니다. 이를 통해 VFM 기반의 고충실도 이미지 재구성이 가능합니다.

- **Performance Highlights**: VFM-VAE는 80 에폭(epoch)에서 2.20 gFID의 성능을 달성했으며, 이 결과는 이전 토크나이저에 비해 10배의 속도 향상을 나타냅니다. 640 에폭까지 지속적으로 훈련했을 때, gFID는 1.62로 향상되어, LDMs를 위한 직접적인 VFM 통합이 우수한 패러다임임을 입증하였습니다. 본 연구는 이미지넷(ImageNet) 256×256에서 경쟁력 있는 결과를 보여주며, 향상된 성능과 효율성을 강조합니다.



### LAND: Lung and Nodule Diffusion for 3D Chest CT Synthesis with Anatomical Guidanc (https://arxiv.org/abs/2510.18446)
- **What's New**: 이 논문은 3D 해부학적 마스크에 조건화된 고품질 3D 흉부 CT 스캔을 생성하는 새로운 잠재 확산 모델(LDM)을 소개합니다. 이 방법은 단일 중간 사양의 GPU를 사용하여 1mm 등방성 해상도로 256x256x256 크기의 볼륨 이미지를 합성하며, 기존 접근 방식에 비해 계산 비용을 크게 줄입니다. 이 모델은 폐 및 결절 지역을 정확하게 제어하여 출력 해부학적 특성에 대한 정밀한 제어를 가능하게 합니다.

- **Technical Details**: 제안된 LAND(Lung-And-Nodule-Diffusion) 모델은 3D U-Net과 3D VAE 구조를 포함합니다. 3D VAE는 입력 CT 이미지를 잠재 표현으로 인코딩하여 공간 해상도를 4배 압축하며, 부가적으로 인코딩된 공간의 특성 차원을 4배 확장합니다. LAND는 20GB GPU에서 256x256x256 규모의 볼륨을 1mm 해상도로 생성할 수 있으며, 이를 통해 해부학적 조건화를 위한 폐 및 결절 마스크를 사용할 수 있습니다.

- **Performance Highlights**: 실험 결과, LAND는 881개의 실제 CT 스캔과 합성 CT 스캔을 대상으로 Fréchet Inception Distance (FID) 평가에서 낮은 FID 값을 달성하여 높은 신뢰도 및 의미적 정렬을 반영합니다. LAND는 WDM 및 PatchDDM와 비교하여 모든 지표에서 우수한 성능을 보이며, 특히 다양한 결절 속성과 함께 다양한 CT 볼륨 생성을 지원하는 점에서 AI 모델 학습이나 의료 전문가 훈련을 위한 유용한 도구로 평가됩니다.



### Beyond Single Images: Retrieval Self-Augmented Unsupervised Camouflaged Object Detection (https://arxiv.org/abs/2510.18437)
Comments:
          ICCV 2025

- **What's New**: 본 논문에서는 Camouflaged Object Detection (COD) 영역에서 새로운 패러다임인 RISE를 제안하여, 고유의 데이터 세트 수준 맥락 정보를 효과적으로 활용하여 허위 레이블(pseudo-labels)을 생성합니다. 이 접근법은 기계적으로 주어진 주석 없이 훈련 이미지에서 환경과 위장 물체의 프로토타입 라이브러리를 구성합니다. 기존의 다른 방법들과 달리, RISE는 외부 데이터 소스에 의존하지 않고 COD 데이터세트 자체에서 프로토타입을 추출하여 고품질 프로토타입을 생성하는 데 중점을 둡니다.

- **Technical Details**: RISE는 이미지별 수준의 메커니즘을 넘어서 전체 데이터셋 수준의 정보를 활용하는 Clustering-then-Retrieval (CR) 전략을 포함하여 위장 물체의 프로토타입을 추출합니다. 먼저, 잡음을 줄이기 위해 각 이미지를 스펙트럴 클러스터링을 통해 조잡한 마스크로 만듭니다. 그런 다음 K-Nearest Neighbor (KNN) 검색을 통해 최종 프로토타입을 생성하며, 이를 Multi-View KNN Retrieval (MVKR)이라는 방식으로 결합하여 다양한 시점에서 얻은 결과를 통합하여 더 견고하고 정밀한 허위 마스크를 생성합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋을 활용한 실험 결과, RISE는 최신의 비지도 및 프롬프트 기반 방법들을 능가하는 성능을 보여줍니다. 또한, 이 방법은 고품질 허위 마스크 생성을 훨씬 빠르게 수행할 수 있어, 데이터를 처리하는 데 걸리는 시간을 대폭 단축시킵니다. 반면 프롬프트 기반 분할 방법은 전체 데이터셋의 마스크를 생성하는 데 며칠씩 소요되는 반면, RISE는 몇 시간 이내에 완성할 수 있습니다.



### ImageGem: In-the-wild Generative Image Interaction Dataset for Generative Model Personalization (https://arxiv.org/abs/2510.18433)
- **What's New**: 우리는 개별 선호를 이해하는 생성 모델에 대한 연구를 위한 데이터셋, ImageGem을 소개합니다. 이 데이터셋은 57K 사용자의 실제 상호작용 데이터를 포함하며, 커스터마이즈된 LoRAs를 포함하여 242K개 사용자의 생성된 이미지를 포함하고 있습니다. 데이터셋의 사용자 선호 주석을 통해 우리는 선호 정렬 모델을 향상시킬 수 있었습니다.

- **Technical Details**: ImageGem 데이터셋은 Civitai 플랫폼에서의 데이터를 바탕으로 구성되었으며, 이 플랫폼은 개인화된 이미지 생성 모델과 함께 관련 메타데이터를 제공합니다. 데이터 구성은 LoRA 모델, 생성된 이미지, 그리고 이러한 모델을 업로드한 사용자들 간의 관계를 기반으로 하여, 사용자 특정 선호도를 효율적으로 조회하고 분석할 수 있게 설계되었습니다.

- **Performance Highlights**: ImageGem 데이터셋을 활용하여 사용자 개인화에 맞춘 이미지 검색 및 생성 모델 추천 성능을 시험했습니다. 또한, VLM(vision-language model)을 활용하여 사용자 선호를 캡셔닝하고 정렬하는 방법을 제안하였으며, 이를 통해 이미지 생성 모델의 개인화 작업에서 새로운 패러다임을 마련했습니다.



### ScaleNet: Scaling up Pretrained Neural Networks with Incremental Parameters (https://arxiv.org/abs/2510.18431)
- **What's New**: ScaleNet은 Vision Transformers (ViTs) 모델을 효율적으로 확장하는 새로운 방법을 제안합니다. 기존의 모델을 처음부터 훈련하는 대신 ScalNet은 pretrained (사전 훈련된) 모델을 기반으로 추가적인 레이어를 신속하게 삽입하여 모델을 확장합니다. 이 접근 방식은 인자 수의 미미한 증가로 ViTs를 효과적으로 확장할 수 있도록 합니다.

- **Technical Details**: ScaleNet은 추가 레이어를 pretrained ViTs에 삽입함으로써 모델 확장을 수행하며, 레이어별 가중치 공유를 통해 파라미터 효율성을 유지합니다. 각 추가 레이어는 pretrained 모델의 해당 레이어와 파라미터 텐서를 공유합니다. 가중치 공유로 인한 성능 저하를 방지하기 위해 각 레이어에 대해 조정 파라미터 세트를 도입하며, 이는 경량 어댑터 모듈을 통해 구현됩니다.

- **Performance Highlights**: ImageNet-1K 데이터셋에서의 실험 결과, ScaleNet은 2배 깊이로 확장된 DeiT-Base 모델이 처음부터 훈련한 모델에 비해 7.42%의 정확도 향상을 달성하며, 훈련 에폭 수도 1/3만 요구하는 것으로 나타났습니다. 이는 ViTs를 확장하는 데 있어 ScaleNet의 효율성을 강조합니다.



### Automated Wicket-Taking Delivery Segmentation and Weakness Detection in Cricket Videos Using OCR-Guided YOLOv8 and Trajectory Modeling (https://arxiv.org/abs/2510.18405)
Comments:
          6 figures, 5 tables, submitted to the 11th IEEE International Women in Engineering (WIE) Conference on Electrical and Computer Engineering 2025

- **What's New**: 이 논문은 깊이 있는 학습 기법을 활용하여 크리켓 비디오 분석을 자동화하는 시스템을 제안합니다. 시스템은 विकेट을 유도하는 투구를 추출하고, 크리켓 공을 탐지하며, 공의 경로를 모델링하는 데 중점을 두고 있습니다. 이 시스템은 YOLOv8 아키텍처를 사용하여 피치(pitch)와 공을 탐지하며, OCR(Optical Character Recognition)을 통해 점수판을 추출하여 중요한 순간을 식별할 수 있습니다.

- **Technical Details**: 제안된 시스템은 세 가지 주요 구성 요소로 이루어져 있습니다: (1) 점수판 분석을 통한.wicket-유도 투구 분할, (2) YOLOv8 기반의 피치 및 공 탐지, (3) 경량 지역 식별을 위한 경로 모델링입니다. 데이터셋은 피치 탐지를 위한 951개의 주석 이미지와 공 탐지를 위한 257개의 주석 이미지로 구성되어 있으며, 각각의 탐색 및 검증을 위한 교육 절차가 포함되어 있습니다. YOLOv8 모델은 정확성과 추론 속도 모두에서 우수한 성능을 보입니다.

- **Performance Highlights**: 제안된 시스템은 피치 탐지에서 99.5%의 평균 평균 정밀도(mean Average Precision, mAP50)를 달성했으며, 공 탐지에서는 99.18%의 mAP50과 0.968의 정밀도(precision)와 0.978의 재현율(recall)을 기록했습니다. 이 시스템은 비디오의 분할된 조각에서 위켓을 유도하는 투구를 효과적으로 확인하였으며, 3D 궤적 그래프를 통해 위켓을 유도하는 투구로부터 뚜렷한 약점 영역을 식별할 수 있는 데이터를 제공합니다.



### Bayesian Fully-Connected Tensor Network for Hyperspectral-Multispectral Image Fusion (https://arxiv.org/abs/2510.18400)
- **What's New**: 이 논문에서는 Bayesian Fully-Connected Tensor Network (BFCTN) 방법을 제안합니다. 이 방법은 HMF(하이퍼스펙트럼-멀티스펙트럼 이미지 융합) 작업에서 Bayesian 학습과 FCTN(전결합 텐서 네트워크) 분해의 통합을 처음으로 시도합니다. 특히, 계층적 희소 사전(hierarchical sparse prior) 구조를 통해 인자가 있는 텐서 간의 연결을 확장하고, 데이터의 저차원 구조와 상관관계를 보다 잘 포착할 수 있는 점이 혁신적입니다.

- **Technical Details**: 해당 연구에서는 n차원 배열(n-dimensional array)인 텐서를 사용하여 다차원 데이터를 모델링하고 분석합니다. BFCTN은 Variational Bayesian inference (VB)와 Expectation-Maximization (EM) 알고리즘을 결합한 파라미터 추정 방법을 통해 HMF 작업에서 비선형 최적화 문제를 효과적으로 해결합니다. 이러한 방식은 기존의 수동 파라미터 조정 필요성을 대폭 경감하는 동시에 높은 정확도를 유지합니다.

- **Performance Highlights**: 심층 실험을 통해 BFCTN은 여러 데이터셋에서 최첨단 융합 성능을 달성하는 것으로 나타났습니다. 이 모델은 공간 저하 문제와 강한 노이즈 간섭을 효과적으로 처리하며, 실제 시나리오에서 뛰어난 강인성과 적용 가능성을 보입니다. BFCTN은 복잡한 실제 환경에서도 적용할 수 있는 유연한 저차원 모델링 능력을 갖추고 있습니다.



### Entropy-Enhanced Conformal Features from Ricci Flow for Robust Alzheimer's Disease Classification (https://arxiv.org/abs/2510.18396)
- **What's New**: 이번 연구에서는 알츠하이머병(AD) 진단을 위한 새로운 지역 표면 표현 방법을 소개하고 검증하였습니다. 연구팀은 지방질(morphometry) 분석을 지원하는 지오메트릭(surface geometry) 모델을 사용하여 뇌 표면의 3D 형태를 분석했습니다. 새로운 접근법은 수많은 연구에서 제시된 방법들의 한계를 극복하면서, 이 질병의 주요 영향을 받을 수 있는 해마(hippocampus) 영역에 초점을 맞추고 있습니다.

- **Technical Details**: 이 연구에서는 160명의 T1 중량 MRI 스캔을 이용하여 뇌 표면 모델을 복원하고, Ricci 흐름(Ricci Flow)을 통해 영역 변형도(area distortion)와 기하학적 형태의 국소 속성(local geometric attributes)을 계산했습니다. 또한, 가우시안 곡률(Gaussian curvature)을 메쉬 기하에서 직접 계산하여, 이들 세 가지 특성을 샤논 엔트로피(Shannon entropy)로 압축된 특징 벡터(feature vectors)로 변환했습니다. 이 특징 벡터는 여러 분류기(classifiers)로 훈련 및 평가되었습니다.

- **Performance Highlights**: 분석 결과, Multi-Layer Perceptron(MLP)와 Logistic Regression 분류기가 98.62%의 높은 정확도와 F$_1$ Score를 기록하여 알츠하이머 환자와 건강한 대조군을 효과적으로 구분할 수 있음을 보여주었습니다. 이 연구는 Ricci 흐름 기반의 기하학적 특성이 피질 변형 측정(cortical morphometry)에서 강력하고 견고한 메트릭을 제공한다는 것을 확인하였으며, 이를 통해 알츠하이머 질병의 연구 및 진단 향상 가능성이 있음을 강조했습니다.



### S2AP: Score-space Sharpness Minimization for Adversarial Pruning (https://arxiv.org/abs/2510.18381)
- **What's New**: 본 논문에서는 적대적 프루닝(adversarial pruning) 방법의 안정성을 향상시키기 위한 새로운 기법인 S2AP(Score-space Sharpness-aware Adversarial Pruning)를 제안합니다. 이 방법은 중요도 점수(importance scores)의 변화를 통해 손실 경량화(loss landscape의 sharpness)를 감소시킴으로써 마스크 선택(mask selection)의 안정성을 높입니다. 이는 전통적인 프루닝 방법에 플러그인 방식으로 통합되어, 기존의 핵심 논리를 변경하지 않고도 사용할 수 있습니다.

- **Technical Details**: S2AP는 세 단계로 구성된 프루닝 파이프라인(i) robust 모델의 사전 훈련, (ii) 이진 마스크 선택, (iii) 프루닝된 모델의 미세 조정(finetuning)을 포함합니다. 이 과정에서, 각 가중치의 중요도를 평가하여 robust 손실을 최소화하는 방식으로 마스크를 최적화합니다. 추가적으로, 손실 경량화의 안정성을 확보하기 위해 중요도 점수를 변화시켜 sharpness를 최소화하는 새로운 전략을 제시합니다.

- **Performance Highlights**: 다양한 데이터셋, 모델 및 스파시티 수준에서 진행된 실험 결과에 따르면, S2AP는 중요도 점수 공간에서의 sharpness를 효과적으로 최소화하며, 마스크 선택의 안정성을 도모하고 최종적으로 적대적 프루닝 방법의 강인성을 개선합니다. 연구 결과는 S2AP 방법이 기존의 적대적 프루닝 방법보다 성능을 향상시킴을 보여줍니다.



### Cross-Modal Scene Semantic Alignment for Image Complexity Assessmen (https://arxiv.org/abs/2510.18377)
Comments:
          14 pages,2 figures, British Machine Vision Conference

- **What's New**: 이 논문에서는 Cross-Modal Scene Semantic Alignment (CM-SSA)라는 새로운 이미지 복잡성 평가 (ICA) 방법을 제안합니다. 기존 방법들이 주로 단일 시각적 모달리티의 특성에 의존한 것에 반해, CM-SSA는 교차 모달 관점에서 장면의 의미적 정렬을 활용하여 성능을 개선합니다. 이 접근 방식은 이미지 복잡성 예측을 주관적인 인간 인식과 더 일치하도록 만들어줍니다.

- **Technical Details**: CM-SSA는 복잡성 회귀(branch)와 장면 의미적 정렬(branch)의 두 개의 브랜치로 구성됩니다. 복잡성 회귀 브랜치는 장면 의미적 정렬 브랜치의 지침을 받으며 이미지 복잡성 수준을 추정합니다. 장면 의미적 정렬 브랜치는 관련된 텍스트 프롬프트와 이미지를 정렬함으로써 풍부한 장면 의미 정보를 전달합니다.

- **Performance Highlights**: 제안된 CM-SSA는 여러 ICA 데이터셋에서 광범위한 실험을 통해 최첨단 접근 방법들을 크게 능가하는 성능을 보여주었습니다. 이러한 성과는 CM-SSA가 이미지 복잡성을 보다 정확히 평가할 수 있는 가능성을 보여줍니다. 코드와 데이터는 제공되는 URL에서 확인할 수 있습니다.



### FeatureFool: Zero-Query Fooling of Video Models via Feature Map (https://arxiv.org/abs/2510.18362)
- **What's New**: 이 논문에서는 비디오 도메인에서 제로 쿼리(Zero-Query) 블랙 박스 공격인 FeatureFool을 제안합니다. 이 공격 방법은 DNN에서 추출한 정보를 활용하여 클린 비디오의 피처(spatial feature) 공간을 수정합니다. 기존의 이론적 공격 방법들이 여러 번의 모델 쿼리(interaction)와 많은 비용을 소모하는 반면, FeatureFool은 쿼리 없이 더 효율적으로 공격을 수행할 수 있습니다.

- **Technical Details**: FeatureFool은 최대 광류(Maximum-Optical-Flow)를 사용하여 가장 중요한 모션 정보를 포함한 프레임을 찾아내고, 그 프레임에 가이드 역전파(Guided Back-propagation) 기법을 적용하여 강력한 피처 맵(patch map) 교란을 생성합니다. 이 방식은 각 프레임에 균일하게 영향을 미치며, 다양한 비디오 라벨링 시스템에서의 샘플링 전략 차이를 무효화합니다. 이를 통해 성공적인 공격을 위해 어떤 쿼리 처리도 필요 없이 전통적인 비디오 분류기에서 70% 이상의 공격 성공률을 달성합니다.

- **Performance Highlights**: FeatureFool에 의해 생성된 적대적 비디오는 SSIM(Structural Similarity Index)과 PSNR(Peak Signal-to-Noise Ratio)에서 높은 품질을 보여줍니다. 또한, 이 공격 방법은 비디오-LLM(비디오-대화형 언어 모델)의 인식에서 70% 이상의 확률로 우회할 수 있으며, 이는 적대적인 콘텐츠를 효과적으로 생성할 수 있음을 시사합니다. FeatureFool은 기존의 주요 비디오 특정 방어 체계에 대해서도 강력한 내성을 보이며, 해당 연구의 기여를 뒷받침합니다.



### Learning Human-Object Interaction as Groups (https://arxiv.org/abs/2510.18357)
- **What's New**: 이번 논문에서 제안된 GroupHOI 프레임워크는 기존의 Human-Object Interaction Detection (HOI-DET) 기법의 한계를 극복하기 위해 집단 관점(group view)에서 관계 모델링을 재조명합니다. 과거의 방법들은 주로 쌍(pairwise) 관계에 초점을 맞췄으나, GroupHOI는 기하학적 근접성과 의미적 유사성을 통해 맥락 정보를 전파하는 방식으로, 여러 개체가 공동 활동에 참여하는 집단 행동을 효과적으로 탐지합니다.

- **Technical Details**: GroupHOI는 공간적 특성을 기반으로 한 학습 가능한 근접성 추정기를 통해 인간과 객체를 별도의 클러스터로 그룹화합니다. 각 그룹 내에서는 self-attention을 통해 부드러운 대응을 계산하여 맥락 단서를 집계 및 분산합니다. 또한 의미적 유사성을 통합하기 위해 호감(human-object interaction) 페어 특징에서 지방 맥락 단서를 추가하여 기본 트랜스포머 기반 상호작용 디코더를 강화합니다.

- **Performance Highlights**: GroupHOI는 HICO-DET와 V-COCO 벤치마크에서 뛰어난 성능을 보여주며, 각각 36.70 mAP와 65.0 mAP를 기록했습니다. 특히 비언어적 상호작용 탐지(Nonverbal Interaction Detection, NVI-DET) 작업에서 우수한 성능을 보이며, 이는 그룹 내 다양한 형태의 고차 상호작용을 포함하는 도전적인 작업입니다.



### Ranking-based Preference Optimization for Diffusion Models from Implicit User Feedback (https://arxiv.org/abs/2510.18353)
- **What's New**: 이 논문에서는 Diffusion Denoising Ranking Optimization (Diffusion-DRO)이라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 역강화학습(inverse reinforcement learning)에 기반하여 선호 학습을 랭킹 문제로 변환하며, 보상 모델(reward model)에 대한 의존성을 제거합니다. 이를 통해 훈련 목적을 단순화하고 비선형 추정 문제를 극복할 수 있습니다.

- **Technical Details**: Diffusion-DRO는 오프라인 전문가 시범 예시와 온라인 정책에서 생성된 부정 샘플을 통합하여 인간의 선호를 효과적으로 캡처합니다. 이 접근법은 기존의 방법들이 요구하는 쌍 비교 데이터의 실용적인 한계를 해결하면서 모델이 선호도를 학습할 수 있도록 합니다. 또한, 새로운 최적화 목표를 유도하여 오버피팅(overfitting)을 방지하고 모델이 시범 예시에서 직접 학습하도록 합니다.

- **Performance Highlights**: 실험 결과, Diffusion-DRO는 다양한 도전적인 프롬프트에서 생성 품질을 향상시켰으며, 정량적 메트릭과 사용자 연구 모두에서 기존 최첨단 모델을 능가하는 성과를 보였습니다. 우리의 접근법은 70% 이상의 PickScore를 달성하여 인간의 선호를 더 잘 반영하며, 일반화 능력 또한 뛰어납니다.



### AV-Master: Dual-Path Comprehensive Perception Makes Better Audio-Visual Question Answering (https://arxiv.org/abs/2510.18346)
Comments:
          13 pages, 9 figures

- **What's New**: AV-Master라는 새로운 프레임워크를 제안하여 복잡한 오디오-비주얼(Audio-Visual) 씬에서 핵심 정보를 효과적으로 추출할 수 있도록 했다. 이 프레임워크는 시간적(dynamic) 및 모달리티(preferece-aware)에 대한 동적 적응 초점_sampling 메커니즘을 통해 질문과 관련된 오디오-비주얼 세그먼트에 점진적으로 집중하도록 설계되었다. 또한 이 모델은 이원 경로 대조 손실을 도입하여 시간이 지남에 따라 상태 유지 및 모달리티 간 보완성을 강조한다.

- **Technical Details**: AV-Master는 입력 오디오-비주얼 세그먼트에서 초점 특성을 정밀하게 캡처하는 동적 적응 초점 샘플링 방법을 사용한다. 이 과정에서 모델은 사전 훈련된 비전-언어 모델인 CLIP이 포함된 시각적 표현을 처리하여 각 세그먼트에 시각적 피처를 추가한다. 또한, 모델의 모달리티 선호도를 나타내는 글로벌 선호 활성화 전략을 채택하여 모달리티별 기여도를 독립적으로 모델링하고, 이를 통해 질문에 대한 적절한 응답을 생성한다.

- **Performance Highlights**: 네 개의 대규모 벤치마크에서 실시된 실험 결과, AV-Master는 기존 방법에 비해 현저하게 우수한 성능을 보여주었으며, 특히 복합 추론 작업에서 두드러진 성과를 나타냈다. AV-Master는 질문 특화된 서로 다른 모달리티의 협동적인 표현을 학습함으로써 오디오-비주얼 질문 답변(AVQA) 작업에서 새로운 최첨단 성능을 달성했다.



### GPTFace: Generative Pre-training of Facial-Linguistic Transformer by Span Masking and Weakly Correlated Text-image Data (https://arxiv.org/abs/2510.18345)
Comments:
          This work was initially drafted in November 2022

- **What's New**: 이번 논문에서는 대규모 웹 데이터로 학습되는 얼굴 지식 학습을 위한 생성적 사전 학습 모델인 GPTFace를 제안합니다. 기존 모델들이 의존해왔던 수작업으로 주석이 달린 데이터셋을 사용하지 않고, 인터넷에서 크롤링한 텍스트와 이미지를 활용하여 사전 학습을 수행합니다. GPTFace는 자가 감독 학습(self-supervised learning) 작업인 masked image/language modeling (MILM) 및 image-text matching (ITM)에서 학습되어 다양한 얼굴 관련 작업에 적용 가능합니다.

- **Technical Details**: GPTFace는 얼굴 분석과 생성 작업을 동시에 수행할 수 있도록 설계되었습니다. 이 모델은 MILM과 ITM 작업에서 공유된 매개변수를 활용하여 생성 모델을 학습합니다. 특히, 저는 SpanBERT의 아이디어에서 영감을 받아 span image-text masking 방식을 제안하며, ITM 감독 신호를 통해 생성된 이미지/텍스트 분포를 조정하는 방법을 도입했습니다. 이를 통해 약한 상관관계에 있는 데이터로부터 효과적으로 학습할 수 있습니다.

- **Performance Highlights**: 실험 결과, GPTFace는 얼굴 속성 분류 및 표정 인식과 같은 다양한 얼굴 관련 하위 작업에서 최신의 대규모 사전 학습 모델들과 경쟁할 만한 성능을 보여줍니다. 또한 얼굴 속성 편집, 표정 변형, 마스크 제거 및 사진 복원과 같은 다양한 얼굴 편집 작업에 대해서도 높은 적용 가능성을 보입니다. GPTFace는 기존 일반 도메인 사전 학습 방법에 비해 더 빠른 수렴 속도를 보이는데, 이는 얼굴 도메인에 특화된 점에서 비롯됩니다.



### ViSE: A Systematic Approach to Vision-Only Street-View Extrapolation (https://arxiv.org/abs/2510.18341)
- **What's New**: 이번 논문은 자율주행 시스템의 폐쇄 루프 시뮬레이션을 위한 현실적인 뷰 보간(Extrapolation) 기술의 도전과제를 해결하기 위해, 4단계의 포괄적인 파이프라인을 제안합니다. 저자들은 ICCV 2025에서 열린 RealADSim Workshop NVS 트랙에서 1위를 차지한 솔루션을 발표합니다. 이 방법은 LiDAR가 아닌 초기화 방법, 새로운 2D 서명 거리 함수(2D-SDF)를 통한 도로 표면 모델링, 생성적 사전(Generative Prior) 활용을 통한 정밀한 감독 등의 혁신적인 접근을 포함합니다.

- **Technical Details**: 제안된 방법은 데이터 기반 초기화 전략을 통해 강력한 고유 포인트 클라우드를 생성하여 로컬 미니마(local minima)를 피하는 것으로 시작합니다. 이어서 도로 표면을 평면 근사화하는 2D-SDF를 도입하여 강력한 기하학적 우선을 주입합니다. 다음으로는 보조 감독이 가능한 가상의 기준 데이터(pseudo ground truth)를 생성하는 생성적 사전을 이용하고, 마지막 단계에서는 데이터 기반 적응 네트워크를 통해 시간 특정 유물(artifacts)을 제거합니다.

- **Performance Highlights**: 제안된 방법은 RealADSim-NVS 벤치마크에서 0.441의 최종 점수를 달성하며, 모든 참가자 중 1위를 기록했습니다. 이 성과는 강력하고 기하학적으로 일관된 거리 뷰(Street View) 보간의 달성을 보여주며, 자율주행 알고리즘 검증을 위한 고충실도 시뮬레이션의 필요성을 강조합니다.



### Enhancing Few-Shot Classification of Benchmark and Disaster Imagery with ATTBHFA-N (https://arxiv.org/abs/2510.18326)
Comments:
          Submitted to a SN journal

- **What's New**: 본 논문은 Attention-based Bhattacharyya-Hellinger Feature Aggregation Network (ATTBHFA-Net)을 도입하여 재난 이미지의 특징을 더욱 효과적으로 분석하는 방법을 제안합니다. 이 프레임워크는 Bhattacharyya 계수와 Hellinger 거리를 결합하여 효과적인 프로토타입 형성을 위한 특징 확률 분포를 집계합니다. 기존 Few-Shot Learning(FSL) 접근법의 한계인 데이터 부족 문제를 해결하고, 재난 처리 능력을 개선하기 위한 새로운 방법론으로 평가받고 있습니다.

- **Technical Details**: ATTBHFA-Net은 Bhattacharyya 거리와 Hellinger 거리의 조합을 통해 재난 이미지에서 높은 intra-class variation과 inter-class similarity를 효과적으로 처리합니다. Bhattacharyya 계수는 class 분리를 촉진하는 역할을 하며, Hellinger 거리는 동일 클래스 내 정렬을 정규화합니다. 이러한 기법은 확률 분포에 기반을 두고 있으며, 전통적인 metric 기반 FSL 방법 대신 robust한 모델을 만드는데 기여합니다.

- **Performance Highlights**: 본 연구는 네 가지 FSL 벤치마크 및 두 가지 재난 이미지 데이터 세트에서 ATTBHFA-Net의 성능을 기존 방법들과 비교하여 우수성을 입증하였습니다. 새로운 시스템은 데이터 부족 문제를 극복하면서도 뛰어난 일반화 성능을 보여, 재난 관리와 구조적 효율성을 크게 향상시킬 수 있는 가능성을 가지고 있습니다.



### Beyond Single Models: Mitigating Multimodal Hallucinations via Adaptive Token Ensemble Decoding (https://arxiv.org/abs/2510.18321)
- **What's New**: 이 논문은 Adaptive Token Ensemble Decoding (ATED)라는 새로운 프레임워크를 제안합니다. ATED는 다양한 LVLM(large vision-language models)의 예측 결과를 통합하여 hallucination(환각) 문제를 완화하는 방법으로, 추가적인 훈련이 필요 없습니다. 이 방법은 모델의 신뢰도를 반영하여 각 모델의 가중치를 동적으로 계산하고, 다양한 decoding paths를 통합하여 맥락적 기반과 의미적 일관성을 향상시킵니다.

- **Technical Details**: ATED는 토큰 수준의 앙상블(ensemble) 프레임워크로, 여러 LVLM의 예측을 집계하여 환각을 줄이는 데 초점을 맞춥니다. 불확실성을 바탕으로 한 가중치(uncertainty-based weights)를 활용하여 각 모델의 출력 확신을 평가하고, 이를 통해 출력 로짓을 동적으로 융합합니다. 이러한 과정은 성능과 추론 효율성을 유연하게 조정할 수 있도록 최적화 기준과 탐색 공간을 조정할 수 있게 설계되었습니다.

- **Performance Highlights**: 실험 결과, ATED는 기존의 최첨단 방법들을 능가하는 성능을 보여주며, 환각 비율을 줄이는 동시에 유창성과 관련성을 손상시키지 않습니다. 또한, 다양한 멀티모달 벤치마크에서 ATED는 꾸준한 성능 향상을 이루었으며, 고위험 응용 분야에서 LVLM의 강건성을 높이는 데 기여할 수 있는 가능성을 제시합니다.



### OmniNWM: Omniscient Driving Navigation World Models (https://arxiv.org/abs/2510.18313)
Comments:
this https URL

- **What's New**: OmniNWM은 자율주행을 위한 통합된 내비게이션 세계 모델로, 상태(state), 행동(action), 보상(reward) 세 가지 차원을 아우르는 혁신적인 접근법을 제공합니다. 이 모델은 다양한 데이터 형식을 통합하여 높은 품질의 파노라마 영상을 생성하며, 고정밀의 카메라 제어를 지원합니다. 또한, 생성된 3D 점유 정보를 활용하여 밀집 보상을 제공함으로써 운전 준수 및 안전성을 높입니다.

- **Technical Details**: OmniNWM은 다중 모달(multi-modal) 비디오 생성, 정밀한 행동 제어, 밀도 기반 보상 시스템을 통합하여 자율주행 환경의 복잡한 문제를 해결합니다. 이 모델은 파노라마 픽셀 수준의 표현을 통해 입력 궤적을 코드화하고, 고품질의 장기 자동 회귀 생성을 가능하게 하는 유연한 강제 모델을 채택하고 있습니다. 이를 통해 OmniNWM은 다양한 주행 시나리오에서의 폐쇄 루프 내비게이션을 지원합니다.

- **Performance Highlights**: 실험 결과, OmniNWM은 비디오 생성 품질, 제어 정확도, 장기 안정성에서 최첨단 성능을 달성하였습니다. 또한, 다양한 도전적인 설정에서도 뛰어난 일반화(generalization) 능력을 보이며 새로운 궤적에 조건화된 파노라마 생성, 진리(Ground Truth) 시퀀스 길이를 초과하는 생성 및 다양한 데이터셋과 카메라 구성에서의 제로샷(zero-shot) 생성을 지원합니다.



### The Impact of Image Resolution on Biomedical Multimodal Large Language Models (https://arxiv.org/abs/2510.18304)
Comments:
          Proceedings of the 10th Machine Learning for Healthcare Conference, PMLR 298, 2025

- **What's New**: 이번 연구는 생물의학 이미징 기술이 고해상도 이미지를 분석하는 데 필수적임을 강조합니다. 특히, 기존의 다중모드 대형 언어 모델(MLLMs)이 저해상도 이미지에 맞춰 설계되어 있어 생물의학 이미지의 중요한 정보가 손실될 위험이 크다는 점을 지적합니다. 연구자들은 원본 해상도에서의 학습 및 추론이 다수의 작업에서 성능을 크게 향상시킨다는 것을 입증하였으며, 혼합 해상도 훈련 방식을 통해 이러한 문제를 효과적으로 완화할 수 있음을 보여주었습니다.

- **Technical Details**: 연구에서는 생물의학 MLLM의 성능에 있어 해상도의 중요성을 분석하였습니다. 특정 실험 결과에 따르면, 원본 해상도의 MLLM을 사용하였을 때 여러 생물의학 작업에서 성능이 0.54%에서 6.8%까지 향상되었습니다. 특히, 학습과 추론 해상도가 일치하지 않을 경우, 성능이 최대 48.7%까지 저하될 수 있음을 발견하였으며, 이러한 문제를 해결하기 위해 혼합 해상도 훈련 전략을 제안합니다.

- **Performance Highlights**: 혼합 해상도 훈련 전략은 실제 성능을 유지하면서 컴퓨팅 제약을 수용할 수 있음을 발견하였습니다. 이는 정렬된 원본 해상도 학습 및 추론과 비슷한 결과를 달성하면서도 평균 성능 손실을 1.0%로 제한합니다. 연구 결과는 고해상도 생물의학 이미지를 다룰 때 모델 사용자와 개발자 모두에게 실질적인 권장 사항을 제공합니다.



### Proactive Reasoning-with-Retrieval Framework for Medical Multimodal Large Language Models (https://arxiv.org/abs/2510.18303)
Comments:
          Work in progress

- **What's New**: 이번 논문에서는 Multimodal Large Language Models (MLLMs)의 추론 능력을 의료 분야에 적용하기 위한 새로운 프레임워크인 Med-RwR를 제안합니다. 이 프레임워크는 외부 지식을 활용하여 진단 과정에서 의사결정의 신뢰성을 높입니다. 특히, 증상이나 도메인 특정 의료 개념을 질의하여 정보를 검색하는 능동적 접근 방식을 통해 기존의 의료 모델들이 갖고 있는 한계를 극복하고자 합니다.

- **Technical Details**: Med-RwR는 두 단계의 강화 학습( Reinforcement Learning , RL) 전략을 기반으로 하여 시각적 진단 결과와 텍스트 임상 정보를 모두 활용하도록 유도하는 보상을 설계하였습니다. 이를 통해 모델은 진단 정보를 보다 효과적으로 검색할 수 있으며, 낮은 예측 신뢰도가 감지되었을 때는 Confidence-Driven Image Re-retrieval (CDIR) 방법을 통해 테스트 중에 유사한 멀티모달 사례를 참조할 수 있습니다. 이러한 방식은 필요한 경우 더 많은 정보를 검색할 수 있도록 모델의 유연성을 강화합니다.

- **Performance Highlights**: 여러 공공 의료 벤치마크에서 평가된 결과, Med-RwR는 기존 모델들에 비해 각각 5.1%, 9.7%, 12.3%의 성능 향상을 달성했습니다. 특히, 저희가 제안한 EchoCardiography Benchmark (ECBench)에서는 훈련 데이터가 부족한 상태에서도 8.8%의 성능 향상을 보이며, 새로운 도메인에 대한 일반화 능력을 입증했습니다. 이러한 성과는 외부 지식 통합을 통한 추론 능력 향상의 효과성을 강조합니다.



### GeoDiff: Geometry-Guided Diffusion for Metric Depth Estimation (https://arxiv.org/abs/2510.18291)
Comments:
          Accepted to ICCV Findings 2025. The first two authors contributed equally. The last two authors share co-corresponding authorship

- **What's New**: 본 연구에서는 미리 훈련된 확산 기반 단안 깊이 추정 모델(DB-MDE)을 스테레오 비전 가이드를 통해 개선하는 새로운 프레임워크를 소개합니다. 기존의 DB-MDE 방법들이 상대 깊이를 예측하는 데 뛰어난 성능을 보이는 반면, 절대 메트릭 깊이 추정은 단일 이미지 상황에서 스케일 모호성으로 인해 도전적인 과제였습니다. 이를 해결하기 위해, 우리는 깊이 추정을 역문제(inverse problem)로 재구성하고 RGB 이미지에 조건화된 미리 훈련된 잠재 확산 모델(latent diffusion models, LDMs)을 활용하여 정확한 깊이 복구를 위한 스케일과 시프트를 학습합니다.

- **Technical Details**: 우리는 깊이 샘플링 프로세스를 역문제로 재구성하면서, 미리 훈련된 LDM과 스테레오 비전 기반 기하학적 제약을 사용하여 어떠한 씬(scene)에 대해서도 스케일과 시프트를 학습합니다. 이 접근 방식은 DB-MDE 방식의 기초 위에 구축되었으며, 객체, 실내, 실외 씬에 모두 적용 가능합니다. 우리의 방법은 기존의 DB-MDE 프레임워크와 원활하게 통합할 수 있는 플러그 앤 플레이(module) 방식으로 설계되었습니다.

- **Performance Highlights**: 광범위한 실험을 통해, 우리의 방법은 다양한 데이터셋에서 효과적으로 작동하며 특히 반투명(translucent) 및 반사(reflective) 표면을 포함한 도전적인 환경에서 최첨단 방법들과 동등하거나 우월한 성능을 보여주었습니다. 우리의 접근 방식은 특정 사용 사례를 위한 재훈련 없이 모든 씬과 깊이 스케일에서 일반화됩니다. 이는 전례 없는 성능 향상을 가져오며 깊이 추정 문제를 다루기 위한 새로운 방향을 열어줍니다.



### Efficient Few-shot Identity Preserving Attribute Editing for 3D-aware Deep Generative Models (https://arxiv.org/abs/2510.18287)
Comments:
          14 pages, 7 figures

- **What's New**: 본 연구에서는 3D 얼굴 편집에서 개체를 보존하며 조정할 수 있는 방법을 제시합니다. 기존 2D 생성 모델의 성과를 바탕으로, 적은 수의 레이블된 이미지를 사용하여 포토리얼리스틱 편집을 구현합니다. 연구의 핵심 아이디어는 라텐트 공간에서의 편집 방향을 효율적으로 추정하는 것입니다.

- **Technical Details**: 이 방법은 3D 인식 심층 생성 모델과 2D 초상화 편집 기술을 결합하여 실행되며, 단 10장 이하의 이미지를 사용해도 3D 얼굴의 다양한 특성을 조정할 수 있습니다. 이 연구에서는 StyleGANv2를 활용하여, 라텐트 공간에서의 편집 방향을 찾기 위해 이버전 네트워크를 사용합니다. 우리는 다양한 카메라 각도에서 멀티 뷰 일관성을 유지하는 방법을 찾습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 10장 이하의 이미지로도 3D 얼굴 속성을 조정할 수 있는 가능성을 보여줍니다. 또한, 연속적인 스타일 매니폴드를 조사하여, 비율적으로 정의된 편집이 가능한 구조를 탐구합니다. 이러한 접근 방식은 대규모 레이블링 데이터 의존도를 줄여주며, 3D 모델 편집의 효율성을 증대 시킵니다.



### StreamingTOM: Streaming Token Compression for Efficient Video Understanding (https://arxiv.org/abs/2510.18269)
- **What's New**: 이번 논문은 StreamingTOM이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 감지, 비디오 이해를 위한 사전 학습 없이도 동작하는 두 단계의 프로세스를 통해 효율적인 스트리밍 비디오 처리를 가능하게 합니다. Causal Temporal Reduction(CTR)과 Online Quantized Memory(OQM)를 조합하여 사전 LL(LLM) 및 사후 LL 메모리 문제를 해결합니다.

- **Technical Details**: StreamingTOM의 CTR은 고정된 프레임 예산 내에서 인접 프레임의 변화와 토큰의 중요도에 기반하여 선택된 토큰만 처리함으로써 메모리 사용을 최적화합니다. OQM은 4비트 형식으로 토큰을 저장하여 필요한 그룹을 필요할 때마다 호출하여 효율적인 kv-cache 관리를 유지합니다. 이 방식을 통해 스트리밍 처리에서도 예측 가능한 지연 시간을 유지합니다.

- **Performance Highlights**: 실험 결과 StreamingTOM은 kv-cache 압축 비율이 15.7배, 최대 메모리 사용량이 1.2배 낮아지며, 첫 번째 토큰에 도달하는 시간(TTFT)이 2배 빠른 성능을 보였습니다. 또한, 오프라인 벤치마크에서 63.8%의 정확도를 유지하며, RVS에서 55.8%/3.7의 성과를 달성하여 현재 훈련 없는 방법들 중 최고의 정확도를 기록했습니다.



### TreeFedDG: Alleviating Global Drift in Federated Domain Generalization for Medical Image Segmentation (https://arxiv.org/abs/2510.18268)
- **What's New**: 이 논문은 의료 영상 세분화(세그멘테이션) 작업에서 연합 학습(FL) 프레임워크 내의 도메인 일반화(Domain Generalization, DG)의 중요성을 강조합니다. 그동안의 연구에서 발견된 글로벌 드리프트(Global Drift, GD) 문제를 해결하기 위한 새로운 이슈를 정의하고, 이를 해결하기 위한 나무 구조 기반 접근법인 TreeFedDG를 제안합니다. 이 방법은 모델 집합 과정에서의 파라미터 불균형을 해결하고, 클라이언트 간의 최대 파라미터 차이를 활용한 스타일 믹싱 방식을 도입하여 드리프트에 대한 강인성을 높이고자 합니다.

- **Technical Details**: TreeFedDG 프레임워크는 나무 구조에서의 계층적 파라미터 집합 방법을 통해 글로벌 모델의 방향성 편차를 억제합니다. 두 가지 주요 메커니즘으로는 파라미터 차이에 기반한 스타일 믹싱 방식(FedStyle)과 점진적인 개인화 융합 전략이 있습니다. 이러한 방식은 클라이언트 모델의 유용성을 극대화하고, 데이터 전송 시 글로벌 지식과 개인적 특성 간의 균형을 유지합니다.

- **Performance Highlights**: 실험 결과, 본 논문에서 제안한 방법은 두 개의 공개 의료 이미지 세분화 데이터셋에서 기존의 최첨단 도메인 일반화 방법들보다 우수한 성능을 보였습니다. 특히, Cross-domain 성능에서 더욱 균형 잡힌 결과를 달성하여, 복잡한 의료 영상 도메인에서의 일반화 성능을 향상시켰습니다. 이러한 결과는 제안된 TreeFedDG 프레임워크가 실제 의료 시나리오에서 효과적으로 적용될 수 있음을 보여줍니다.



### Latent-Info and Low-Dimensional Learning for Human Mesh Recovery and Parallel Optimization (https://arxiv.org/abs/2510.18267)
Comments:
          Accepted by ICME2025

- **What's New**: 본 논문에서는 기존 3D 인간 메시 회복 방법의 한계를 극복하기 위해 잠재 정보(latent information)와 저차원 학습(low dimensional learning)을 바탕으로 한 2단계 네트워크를 제안합니다. 첫 번째 단계에서는 이미지 기능의 저주파와 고주파 성분에서 전반적인 형태 정렬을 포함한 글로벌(global) 정보와 텍스처 및 세부사항과 같은 로컬(local) 정보를 추출하여 하이브리드 잠재 주파수(domain feature) 특징으로 집계합니다. 두 번째 단계에서는 이 하이브리드 특징을 활용하여 3D 인간 메시의 포즈(pose)와 상호 작용(interaction)을 모델링하고 최적화합니다.

- **Technical Details**: 제안된 네트워크는 두 단계로 구성됩니다. 첫 번째 단계는 잠재 정보 추출(latent information extraction)으로, 3D 포즈와 하이브리드 특징을 추출합니다. 입력 비디오 시퀀스를 ResNet-50을 통해 처리하여 이미지 특징을 얻고, 2D 포즈 추정기는 해당 2D 포즈 추정을 제공합니다. 두 번째 단계에서는 3D 포즈와 메시 간의 상호 작용을 모델링하여 3D 인간 메시를 정확히 재구성합니다. 저차원 메시 포즈 상호 작용 방법을 통해 계산 비용을 줄이며 포즈와 형태의 상호 작용을 최적화합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 공개된 대규모 데이터셋에서 기존 최첨단 기법들에 비해 우수한 성능을 보여주었습니다. 특히, 저차원 메시 포즈 상호 작용 방법을 통해 계산 비용을 현저하게 감소시키면서도 재구성 정확도는 유지되었습니다. 이 성능 향상은 3D 포즈 추정과 메시 증가의 정확성을 강화하는 데 기여합니다.



### UWBench: A Comprehensive Vision-Language Benchmark for Underwater Understanding (https://arxiv.org/abs/2510.18262)
Comments:
          We have released V1, which only reports the test results. Our work is still ongoing, and the next version will be coming soon

- **What's New**: 이번 논문에서는 UWBench라는 새로운 벤치마크를 소개합니다. UWBench는 해양 생물 및 구조물을 이해하기 위한 첫 번째 대규모 데이터셋으로, 15,003장의 고해상도 수중 이미지를 포함하고 있습니다. 이 데이터셋은 수질, 조명 조건 및 생물 다양성의 변화를 반영한 주제별 캡션, 개체 참조 표현, 질문-답변 쌍을 제공합니다.

- **Technical Details**: UWBench 데이터셋은 다양한 수중 환경을 포괄하며, 수중 영상 이해를 위한 세 가지 평가 기준을 설정합니다. 여기에는 상세한 이미지 캡션 작성, 해양 생물의 정확한 위치 지정을 위한 비주얼 그라운딩, 다중 모달 추론을 위한 비주얼 질문 응답이 포함됩니다. 이 데이터셋은 생물학적 정확성을 보장하기 위해 인간 전문가에 의해 검증된 주석으로 보강됩니다.

- **Performance Highlights**: 실험 결과, 최신 VLM들은 수중 이미지를 이해하는 데 있어 여전히 많은 도전에 직면하고 있으며, 모든 모델들이 육상의 벤치마크에 비해 성능 하락을 보였습니다. 이러한 결과는 수중 환경에서 시각 특징의 열화, 생물 분류에 대한 도메인 지식 부족, 복잡한 장면에서의 정밀한 물체 위치 남기기를 위한 세밀한 공간적 추론의 어려움에 기인합니다. UWBench는 이러한 문제를 해결하기 위한 중요한 자원을 제공합니다.



### Hyperbolic Space Learning Method Leveraging Temporal Motion Priors for Human Mesh Recovery (https://arxiv.org/abs/2510.18256)
Comments:
          Accepted by ICME2025

- **What's New**: 이 논문에서는 3D 인간 메시 복원에서 하이퍼볼릭 공간을 활용한 학습 방법을 제안했습니다. 이를 통해 비디오에서 3D 인간 메시를 효과적으로 복원하였으며, 기존 방법들의 유클리드 공간에서는 포착하기 어려운 계층 구조를 더 잘 반영할 수 있게 되었습니다. 또한, 모션 정보의 시간적 특성을 강화하는 모듈을 디자인하여 메시의 정확성과 매끄러움을 동시에 향상시켰습니다.

- **Technical Details**: 제안된 방법은 크게 두 가지 모듈로 구성됩니다. 첫 번째는 시간적인 모션 선행 추출 모듈로, 입력된 3D 포즈 시퀀스와 이미지 특징 시퀀스에서 모션 특징을 추출합니다. 두 번째는 하이퍼볼릭 공간 최적화 학습 모듈로, 시간적 모션 선행 정보를 활용하여 3D 포즈와 모션 정보를 분리하여 메시 특징을 최적화합니다.

- **Performance Highlights**: 광범위한 데이터셋에서 실시한 실험 결과는 제안된 방법이 기존의 최첨단 기술들보다 우수한 성능을 보임을 나타냅니다. 특히, 하이퍼볼릭 공간에서의 학습 방법은 3D 메시의 계층 구조를 더욱 정밀하게 모델링할 수 있음을 입증하였습니다. 이러한 접근법은 가상 현실(VR) 및 증강 현실(AR) 분야에서 많은 응용 가능성을 열어 줄 것으로 기대됩니다.



### OpenInsGaussian: Open-vocabulary Instance Gaussian Segmentation with Context-aware Cross-view Fusion (https://arxiv.org/abs/2510.18253)
- **What's New**: 본 논문에서는 OpenInsGaussian이라는 새로운 3D 가우시안 분할 프레임워크를 소개합니다. 이 방법은 개방형 어휘(instance-level segmentation)을 수행할 수 있으며, 대규모 2D 비전-언어 모델을 효과적으로 활용합니다. 기존의 2D 모델에서 발생하는 문맥 정보 손실을 해결하기 위한 모듈과 주의 기반(feature aggregation) 기법을 통해 향상된 성능을 보입니다.

- **Technical Details**: OpenInsGaussian은 두 가지 주요 모듈로 구성됩니다. 첫 번째는 Context-Aware Feature Extraction으로, 이는 frozen CLIP 모델에서 마스크 기반 쿼리로 피쳐를 직접 추출합니다. 두 번째는 Attention-Driven Feature Aggregation으로, 여러 뷰의 피쳐를 활용하여 세멘틱 일관성을 기반으로 가중치를 조절하여 노이즈 및 오정렬 문제가 발생하지 않도록 합니다.

- **Performance Highlights**: OpenInsGaussian은 기존의 기준 성과를 뛰어넘는 결과를 나타내며, 개방형 어휘로 3D 가우시안 분할에서 새로운 최첨단 성능을 달성했습니다. 실험을 통해 공간적 맥락을 반영하고 주의 가중치를 조절하는 것이 3D 장면 재구성의 품질과 견고함을 크게 향상시킨 것을 보여줍니다.



### BlendCLIP: Bridging Synthetic and Real Domains for Zero-Shot 3D Object Classification with Multimodal Pretraining (https://arxiv.org/abs/2510.18244)
Comments:
          Under Review

- **What's New**: 본 논문에서는 BlendCLIP이라는 다중 모드 예비 학습 프레임워크를 제안하여 합성 데이터(synthetic data)와 실제 데이터(real-world data) 간의 간극을 극복합니다. 이 시스템은 3D 객체 분류에 효과적으로 접근할 수 있도록, 실제 자율 주행 데이터에서 객체 수준의 트리플을 대규모로 생성하는 파이프라인을 포함합니다. 또한 인식되지 않은 객체에 대한 의미 있는 레이블을 부여하는 제로샷(zero-shot) 3D 분류 방식의 필요성이 강조됩니다.

- **Technical Details**: BlendCLIP은 CAD 데이터의 의미론적 풍부함과 실제 LiDAR 데이터의 특성을 결합하는 커리큘럼 기반(data mixing strategy) 데이터를 사용하여 모델을 점진적으로 조정합니다. 실험은 다양한 데이터셋에서 수행되어, 본 방법이 제로샷 분류 정확도를 27% 향상시킴을 보여줍니다. 논문에서는 합성 데이터에서 학습된 일반 지식을 유지하면서도 실제 데이터에 적응할 수 있도록 설계된 시스템을 설명합니다.

- **Performance Highlights**: 제안된 방법은 nuScenes와 TruckScenes와 같은 Outdoor 데이터셋에서 최고 성능을 기록하였으며, nuScenes에서는 기존 최고 방법보다 19.3% 향상된 성능을 보였습니다. 이 결과는 소량의 실제 LiDAR 샘플로도 강력한 일반화를 이끌어낼 수 있음을 시사합니다. 즉, 대규모 데이터셋의 재주석 없이도 robust한 3D 인식을 가능하게 합니다.



### DeepSeek-OCR: Contexts Optical Compression (https://arxiv.org/abs/2510.18234)
- **What's New**: DeepSeek-OCR는 긴 텍스트 콘텐츠를 압축하기 위한 초기 탐색으로 제안된 모델로, DeepEncoder 및 DeepSeek3B-MoE-A570M으로 구성되어 있습니다. 이 모델은 고해상도 입력에서 낮은 활성화 상태를 유지하면서 높은 압축 비율을 달성하도록 설계되었습니다. 실험 결과, 텍스트 토큰 수가 비전 토큰 수의 10배 이내에 있을 때, OCR 정밀도가 97%에 달하는 것으로 나타났습니다.

- **Technical Details**: DeepSeek-OCR은 두 가지 주요 구성 요소로 이루어져 있으며, DeepEncoder는 시각적 데이터에서 피쳐(Feature)를 추출하고 토큰화 및 압축을 담당합니다. 이 아키텍처는 윈도우 주의(Window Attention)와 글로벌 주의(Global Attention) 인코더를 16배 축소하는 합성곱 압축기(Convolutional Compressor)를 통해 직렬로 연결합니다. 이 설계를 통해 많은 비전 토큰을 처리하면서도 효율적인 메모리와 토큰 압축을 달성합니다.

- **Performance Highlights**: DeepSeek-OCR은 OmniDocBench에서 GOT-OCR2.0보다 뛰어난 성능을 보이며, 평균 256 토큰/페이지를 사용하여 100 비전 토큰만으로 이를 초과합니다. 이 모델은 또한 하루에 20개 노드(각 8 A100-40G GPU)를 사용하여 LLM/VLM을 위한 교육 데이터를 33백만 페이지 이상 생성할 수 있는 실용성을 보여줍니다. 이러한 성능은 역사적인 긴 문맥 압축 및 메모리 망각 메커니즘 연구 등에 상당한 가능성을 제시합니다.



### Beyond Frequency: Scoring-Driven Debiasing for Object Detection via Blueprint-Prompted Image Synthesis (https://arxiv.org/abs/2510.18229)
- **What's New**: 이 논문은 객체 탐지를 위한 생성 기반의 디바이싱(debiasing) 프레임워크를 제안합니다. 기존 디바이싱 방법들은 샘플의 표현 다양성에 한계가 있어 잘못된 편향을 유지하는 경향이 있습니다. 따라서, 저자는 새로운 표현 점수(representation score, RS)를 도입하여 바이어스(bias) 없는 레이아웃 생성을 유도하고, 생성 정렬(generative alignment) 전략을 통해 탐지기(detector)와 생성기(generator) 간의 효율적인 통신을 가능하게 합니다.

- **Technical Details**: 제안된 방법은 두 가지 핵심 문제를 해결합니다. 첫째, 인스턴스 빈도(instance frequency)는 모델에 필요한 데이터를 결정하는데 불완전한 수치이기 때문에, RS를 통해 대표성을 분석합니다. 둘째, 텍스트 프롬프트를 모호하게 사용하기 보다는 시각적 청사진(visual blueprint)을 활용하여 각 객체의 클래스, 크기 및 위치를 명확하게 지정함으로써 고품질 합성을 보장합니다.

- **Performance Highlights**: 제안된 프레임워크는 대표성이 낮은 객체 그룹의 성능 격차를 현저히 줄입니다. 예를 들어, 드문 인스턴스에서 4.4 mAP, 일반적인 인스턴스에서 3.6 mAP 개선을 달성하였으며, 생성된 이미지의 레이아웃 정확도에서 기존 L2I 합성 모델 대비 15.9 mAP 향상을 보여주었습니다. 이로 인해 새로운 SOTA(state-of-the-art)를 수립하였고, 바이어스 문제를 효과적으로 처리하는 데 성공했습니다.



### VLSU: Mapping the Limits of Joint Multimodal Understanding for AI Safety (https://arxiv.org/abs/2510.18214)
Comments:
          10 pages, 5 figures, 4 tables. Under review

- **What's New**: 이번 연구는 멀티모달(Multimodal) 모델의 안전성을 평가하기 위한 새로운 프레임워크인 비전 언어 안전 이해(VLSU, Vision Language Safety Understanding)를 제시합니다. 이 프레임워크는 다양한 안전 패턴을 통해 멀티모달 안전성을 체계적으로 분석하며, 8,187개의 샘플로 구성된 대규모 벤치마크를 활용합니다. 연구결과, 기존의 모델들이 멀티모달 안전 신호를 제대로 이해하지 못한다는 것을 발견하였으며, 이는 이전 연구들에서 하지 못했던 위험 요소의 결합적 해석의 부재에서 기인합니다.

- **Technical Details**: VLSU 프레임워크는 자료 생성 과정에서 두 가지 주요 단계를 포함합니다. 첫 번째 단계에서는 해악 카테고리에 따른 경계(Unsafe) 및 안전(Safe) 위험 등급을 정의하고, 두 번째 단계에서는 개별 모드의 안전 등이 어떻게 결합되는지를 규명합니다. 또한, 새로운 경계 위험 등급을 도입하여 각 모드의 안전 신호를 평가하고, 교차 모드 상호작용(concatenation)을 고려한 안전 평가 방식을 개발합니다.

- **Performance Highlights**: 연구에서는 11개의 최첨단 VLM 모델을 평가한 결과, 개별 모드의 안전 신호에서는 90% 이상의 정확도를 달성했지만, 이미지와 텍스트 결합 시 안전 레이블 판별에선 성능이 20-55%로 급감했습니다. 더불어, 34%의 오류는 각 개별 모드에서 올바른 판별이 이루어졌음에도 발생함을 확인했습니다. 이러한 결과들은 현재의 모델들이 멀티모달 이해와 조합적 추론(compositional reasoning)에서 심각한 한계를 가지고 있음을 드러냅니다.



### EMA-SAM: Exponential Moving-average for SAM-based PTMC Segmentation (https://arxiv.org/abs/2510.18213)
- **What's New**: 이 논문에서는 Papillary thyroid microcarcinoma (PTMC) 영상에서 종양의 정밀한 세분화를 위해 새로운 EMA-SAM 모델을 도입하였습니다. EMA-SAM은 Segment Anything Model 2 (SAM-2)의 경량 확장판으로, 신뢰도 기반의 지수 이동 평균 포인터를 메모리 뱅크에 통합하여 여러 프레임에서 종양의 안정적인 잠재 프로토타입을 제공합니다. 이 접근 방식은 초음파 영상에서 프로브 압력과 기포 차단에 따른 시간적 일관성을 유지하면서도 빠르게 적응할 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: EMA-SAM은 SAM-2가 가진 메모리 은행의 한계를 극복하고, 프레임 간의 혼란을 줄이며 안정적인 세분화를 제공하기 위해 고안되었습니다. 이 모델은 매 프레임마다 신뢰도가 높은 정보를 업데이트하여 프로토타입을 유지하고, 기포 차단이나 기타 요인으로 인해 PTMC가 사라졌을 때에도 일관된 세분화를 보장합니다. 최종적으로 EMA-SAM은 PTMC-RFA 데이터셋에서 maxDice와 maxIoU의 향상을 달성하며, 실시간 처리 성능도 유지하고 있습니다.

- **Performance Highlights**: 실험 결과, EMA-SAM은 maxDice를 0.82에서 0.86으로, maxIoU를 0.72에서 0.76으로 개선하며, 잘못된 긍정 결과를 29% 줄이는 성과를 보였습니다. 외부 벤치마크인 VTUS 및 대장내시경 비디오 폴립 데이터셋에서도 SAM-2 대비 2~5 Dice 포인트가 일관되게 향상된 성능을 보여줍니다. 이는 EMA-SAM이 RFA 영상에서 종양 추적을 위한 강력하고 효율적인 프레임워크임을 입증하는 결과입니다.



### RadDiagSeg-M: A Vision Language Model for Joint Diagnosis and Multi-Target Segmentation in Radiology (https://arxiv.org/abs/2510.18188)
- **What's New**: 이 논문에서는 진단 텍스트 및 픽셀 수준의 분할 마스크를 동시에 생성하는 데 어려움을 겪는 기존의 의료 비전 언어 모델(VLM)과 그 한계를 극복하기 위한 새로운 데이터셋과 모델, RadDiagSeg-D와 RadDiagSeg-M을 소개합니다. RadDiagSeg-D는 X-ray 및 CT를 포함한 다양한 이미징 모드에서 28,000개 이상의 샘플을 결합하여 단계별 질문을 설정하며, 이는 VQA(Visual Question Answering)와 세분화(segmentation) 작업을 포함하고 있습니다. RadDiagSeg-M은 비정상 탐지, 진단, 세분화를 공동으로 수행할 수 있는 새로운 비전-언어 모델로, 의료 진단에 유용한 결과를 제공합니다.

- **Technical Details**: RadDiagSeg-D는 비정상 탐지를 위한 폐쇄형 질문, 진단을 위한 개방형 질문, 다중 개체의 세분화 작업으로 구성된 3단계의 계층적 질문을 설계하였습니다. RadDiagSeg-M은 LISA의 구조를 기반으로 하며, 세분화 생성을 유도하는 특별한 토큰을 통해 모델의 어휘를 확장합니다. 이 모델은 단일 세분화 생성을 지원하는 기존 모델과 달리 유연한 수의 마스크 생성이 가능한 구조를 가지고 있습니다.

- **Performance Highlights**: RadDiagSeg-M은 RadDiagSeg-D 벤치마크의 VQA 하위 작업에서 최첨단 결과를 달성하며, 멀티 타겟 텍스트 및 마스크 생성 작업에 대해 경쟁력 있는 기준을 설정합니다. 이를 통해 의료적 맥락에서 필수적인 정보 제공과 클리닉적 유용성을 동시에 달성하게 됩니다. 논문에서 제안한 벤치마킹 도구는 연구 커뮤니티에서 다단계의 텍스트-마스크 생성 작업을 효과적으로 평가하는 데 도움을 줄 것입니다.



### VelocityNet: Real-Time Crowd Anomaly Detection via Person-Specific Velocity Analysis (https://arxiv.org/abs/2510.18187)
Comments:
          8 pages, 3 figures

- **What's New**: 이 논문에서는 복잡한 군중 장면에서 이상치를 탐지하기 위한 새로운 프레임워크인 VelocityNet을 제안합니다. 이 시스템은 개인의 속도를 추출하기 위해 헤드 탐지(head detection)와 밀집 광학 흐름(dense optical flow)을 결합한 이중 파이프라인 구조로 이루어져 있습니다. 또한, 이 논문은 세분화된 모션 패턴을 사용하여 맥락에 따라 변하는 이상 탐지를 수행하는 방법을 제시합니다.

- **Technical Details**: VelocityNet은 라이브 비디오 입력을 통해 개인별 모션 카테고리와 이상 점수를 생성하는 시스템입니다. Motion Estimation Module은 인입하는 프레임 간의 밀집 광학 흐름을 계산하며, Head Detection Module은 오클루전(occlusion)에서도 개인의 머리를 탐지하고 지역을 설정합니다. 두 모듈의 결과는 Anomaly Detection Module에서 통합되어 밀집 군중 환경에서 실시간으로 이상을 탐지할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, VelocityNet은 다양한 이상 모션 패턴을 효과적으로 탐지하며, 밀집 군중 환경에서도 실시간 성능을 발휘합니다. 이 시스템은 이전의 한계를 극복하고 실제 배치에 적합한 해석 가능한 출력을 제공합니다. 제안된 메커니즘은 군중 밀집도에 민감하게 적응하여 컨텍스트에 맞는 이상 탐지를 가능하게 합니다.



### Adapting Stereo Vision From Objects To 3D Lunar Surface Reconstruction with the StereoLunar Datas (https://arxiv.org/abs/2510.18172)
Comments:
          Accepted to ICCV workshop 2025. The project page can be accessed via this this https URL URL. The source code is available at this this https URL URL

- **What's New**: 본 연구에서는 LunarStereo라는 최초의 개방형 데이터셋을 소개합니다. 이는 고해상도 지형 및 반사 모델을 기반으로 한 광선 추적(ray tracing) 시뮬레이션을 통해 생성된 달의 사진 사실적인 스테레오 이미지 쌍으로 구성되어 있습니다. 이 데이터셋은 달 남극 주변의 다양한 고도, 조명 조건 및 관찰 각도를 포함하여 3D 재구성 작업을 위한 물리적 기반의 감독(supervision)을 제공합니다.

- **Technical Details**: LunarStereo 데이터셋은 합성(stereo pairs) 스테레오 쌍을 포함하고 있으며, 이는 고해상도 디지털 고도 모델(Digital Elevation Models, DEMs)에서 생성되었습니다. 이 모델들은 정확한 반사 모델(BRDFs)과 현실적인 태양 조명을 고려하며, 다양한 카메라 경로(camera trajectories)를 포함하여 고도와 조명 조건을 매우 사실적으로 재현합니다. 연구진은 MASt3R 모델을 LunarStereo 데이터셋에 대해 파인튜닝(fine-tuning)하여, 달 환경에 맞추었습니다.

- **Performance Highlights**: 연구 결과, 파인튜닝된 MASt3R 모델은 3D 지형 재구성의 신뢰성을 크게 향상시키며, 경사 추정 오류에서 평균 70% 이상의 감소를 보여주었습니다. 또한 전체적인 상대적 정확도가 약 50% 향상되었습니다. 이러한 결과는 MASt3R의 우수성을 입증하며, 현대 3D 비전 네트워크가 낮은 질감(low-texture) 및 분포 외(out-of-distribution) 도메인에 잘 적응할 수 있음을 나타냅니다.



### World-in-World: World Models in a Closed-Loop World (https://arxiv.org/abs/2510.18135)
Comments:
          Code is at this https URL

- **What's New**: 이번 논문은 World-in-World라는 새로운 플랫폼을 소개하며, 이는 예측 (predictive perception) 기능을 통해 내재적 작업 (embodied tasks)에서의 성공을 정량화하는 폐쇄 루프 (closed-loop) 메커니즘을 제공합니다. 기존의 벤치마크는 시각적 품질에만 초점을 맞췄다면, 이 연구는 실제 환경과의 상호작용을 반영하는 평가 구조를 통해 생성적 세계 모델 (generative world models)의 실질적인 유용성을 측정할 수 있습니다. 논문에서는 데이터 스케일링 법칙(data scaling law)과 같은 새로운 발견을 통해 모델의 성능을 개선하는 방법도 제시하고 있습니다.

- **Technical Details**: World-in-World의 고유한 전략은 폐쇄 루프 온라인 계획 (closed-loop online planning) 및 표준화된 행동 API (action API)를 통해 다양한 세계 모델을 일관되게 통합하고 평가하는 것입니다. 이를 통해 에이전트는 환경 변화와 보상을 예측하여 행동을 결정하는 데 필요한 정보를 얻을 수 있습니다. 또한, 사전 훈련된 비디오 생성기를 활용하여 행동-관찰 데이터를 사용한 후속 훈련(post-training) 프로토콜을 통해 모델의 적응 잠재력을 평가합니다.

- **Performance Highlights**: 연구 결과, 높은 시각적 품질이 꼭 작업 성공과 연결되지 않으며, 제어 가능성(controllability)이 더 중요한 요소라는 점이 드러났습니다. 또한, 행동-관찰 데이터에 대한 후속 훈련이 사전 훈련된 비디오 생성기를 업그레이드하는 것보다 효과적이라는 것을 발견했습니다. 마지막으로, 온라인 계획을 통한 추론 시간 계산량 증가가 폐쇄 루프 성능을 크게 향상시킬 수 있다는 점도 밝혀졌습니다.



### SafeCoop: Unravelling Full Stack Safety in Agentic Collaborative Driving (https://arxiv.org/abs/2510.18123)
- **What's New**: 이 연구는 자연어 기반의 협업 주행 시스템에서의 안전성과 보안성 문제를 체계적으로 분석한 최초의 연구입니다. 자연어를 커뮤니케이션 매체로 활용하여 드라이빙 안전성과 효율성을 향상하려는 최근 경향에 주목하며, 새로운 위험 요소들을 조명합니다. 특히, 메시지 손실, 홀로그램 생성, 의미 조작과 같은 언어 통신의 취약성을 지적하고 이를 해결하기 위한 방안을 제시합니다.

- **Technical Details**: 다양한 공격 전략에 대한 포괄적인 분류 체계를 개발하여 연계 차단, 중계 및 재생 간섭, 콘텐츠 스푸핑 및 다중 연결 위조 등 여러 공격 경로를 분석합니다. 시스템에서 생성된 각 주행 에이전트는 Multi-modal Large Language Models(MLLMs)를 기반으로 작동하며, 두 개의 핵심 모듈인 추론 모듈(Ri)과 행동 모듈(Di)을 갖추고 있습니다. 나아가, 본 논문에서는 언어 기반의 지역적 참조 변환 문제를 해결하기 위해 Agentic Transformation Function(ATF)을 도입합니다.

- **Performance Highlights**: 제안된 방어 체계인 SafeCoop는 CARLA 시뮬레이터에서 32개의 중요한 시나리오에서 테스트되고, 악의적인 공격 하에서도 69.15%의 주행 점수 향상과 67.32%의 F1 점수를 달성하였습니다. 이는 언어 기반 협업 주행에서의 취약성을 확인하고, 이를 감지하는 데 있어 뛰어난 성능을 발휘함을 보여줍니다. 이 연구는 안전하고 신뢰할 수 있는 언어 기반 협업을 위한 향후 연구 방향을 제시합니다.



### Online In-Context Distillation for Low-Resource Vision Language Models (https://arxiv.org/abs/2510.18117)
- **What's New**: 이 논문에서는 저비용 및 자원 제약 환경에서의 비전-언어 모델(VLM)을 개선하기 위한 온라인 인-컨텍스트 증류(Online In-Context Distillation, ICD) 방법을 제안합니다. 기존의 대형 VLM은 성능이 뛰어나지만 배포의 실용성이 떨어지며, 소형 VLM은 효율적이지만 성능 격차를 해소하기 위해 비싼 미세 조정(fine-tuning)이 필요합니다. 제안된 ICD는 모델 크기에 관계없이 적은 양의 주석만으로도 강력한 성능 향상을 이끌어낼 수 있습니다.

- **Technical Details**: ICD 방법은 강력한 교사 모델이 실시간으로 자동으로 시연(demonstration)을 생성하고, 이 시연을 통해 경량화된 학생 모델이 지식을 증류(distill)하는 두 가지 주요 구성 요소로 구성됩니다. 새로운 테스트 시간(scalaing techniques) 및 불확실성 기반 질의를 통해 교사 모델의 주석 노이즈를 줄이고, 강력한 시연 선택을 가능하게 함으로써 자원 제약 환경에서도 효과적으로 작동합니다. 이 방법은 또한 소형 VLM의 ICL(In-Context Learning) 수행 능력이 모델의 기초 능력에 많이 의존한다는 점을 공인합니다.

- **Performance Highlights**: ICD는 소형 모델의 성능을 최대 33% 향상시키며, 최소 4%의 교사 주석만으로도 효과를 발휘합니다. 예를 들어, 7B 모델인 LLaVA-OneVision의 성능을 42.6%에서 70.8%로 끌어올렸으며, 이는 GPT-4o 교사보다 더 높은 성능입니다. 전체적으로 ICD는 자원 제약이 있는 시나리오에서 소형 VLM의 성능을 평균 14.8% 향상시키면서도 최소한의 컴퓨팅 비용을 요구합니다.



### From Volume Rendering to 3D Gaussian Splatting: Theory and Applications (https://arxiv.org/abs/2510.18101)
Comments:
          Accepted at the Conference on Graphics, Patterns and Images (SIBGRAPI), math focused, 5 equations, 5 Figure, 5 pages of text and 1 of bibligraphy

- **What's New**: 이 논문에서는 3D 이미지를 포즈가 지정된 사진으로부터 재구성하는 문제를 다루고 있으며, 3D Gaussian Splatting (3DGS)의 발전으로 인해 근본적인 변화가 이루어지고 있습니다. 3DGS는 장면을 3D Gaussian의 집합으로 모델링하여 효율적인 렌더링을 가능하게 하며, 일반적인 그래픽 파이프라인과의 원활한 통합을 제공합니다. 이러한 새로운 접근법은 고해상도의 실시간 구동이 가능하지만, 높은 메모리 사용량과 조명 효과의 직접적인 적용 등 여러 제한 사항이 존재합니다.

- **Technical Details**: 이 논문에서는 Neural Radiance Fields (NeRFs)와 3DGS를 사용하여 3D 장면을 재구성하는 방법을 분석합니다. 장면을 볼륨 밀도(Volume Density)와 방사선(Radiance)으로 표현하는 NeRF의 유용성을 설명하며, 3DGS는 장면을 3D Gaussian의 컬렉션으로 모델링하여 차별화 가능한 볼륨 렌더링을 가능하게 합니다. 논문은 Gaussian 초기화 및 학습 중 적응 기술에 대한 구체적인 공식을 제시하며, 3DGS의 한계 요소를 해결하기 위해 최근의 접근 방식도 검토합니다.

- **Performance Highlights**: 3D Gaussian Splatting은 고해상도의 정보 제공과 실시간 성능을 달성하는 동시에 메모리 소비를 줄이는 데 기여합니다. 다양한 애플리케이션, 특히 표면 재구성(Surface Reconstruction), 애니메이션(Animation), 아바타 모델링(Avatar Modeling), 희박한 뷰로부터의 피드포워드 3D 재구성에 대한 활용 사례를 논의합니다. 이러한 작업들은 3DGS가 제공하는 효율적인 렌더링과 피드포워드 파이프라인에 적합함을 보여줍니다.



### Accelerating Vision Transformers with Adaptive Patch Sizes (https://arxiv.org/abs/2510.18091)
Comments:
          Project page at this https URL

- **What's New**: Adaptive Patch Transformer (APT)는 시각적 정보를 효과적으로 처리하기 위해 동일한 이미지 내에서 서로 다른 패치 크기를 사용하는 혁신적인 구조입니다. APT는 균일하게 크기가 정해진 패치로 입력 이미지를 나누는 전통적인 방법의 한계를 극복하여 근본적으로 입력 토큰의 수를 줄여줍니다. 이를 통해 APT는 더 많은 계산 효율성을 가지며, 고해상도 이미지의 처리 성능을 유지할 수 있습니다.

- **Technical Details**: APT는 여러 스케일에서 엔트로피를 계산하여 동적으로 큰 패치와 작은 패치를 적절히 할당합니다. 텍스처가 단조로운 지역은 큰 패치로, 구성 요소가 복잡한 지역은 작은 패치로 표현하여 그림의 정보 중복을 줄이는 방식입니다. 또한, APT는 제로 초기화된 MLP를 사용하여 패치의 임베딩을 결합함으로써 네트워크 손상 없이 수렴할 수 있도록 합니다.

- **Performance Highlights**: APT는 ViT 학습 및 추론 속도를 최대 40% 가량 향상시키며, 특히 고해상도 이미지와 큰 모델에서 성능 향상이 두드러집니다. 실험 결과, APT는 데이터셋에 따라 기존의 ViT 성능을 유지하며, 시각적 질문 응답, 객체 탐지, 의미론적 분할과 같은 다양한 이미지 이해 작업을 성공적으로 수행하는 것으로 나타났습니다.



### Big Data, Tiny Targets: An Exploratory Study in Machine Learning-enhanced Detection of Microplastic from Filters (https://arxiv.org/abs/2510.18089)
- **What's New**: 본 연구에서는 미세플라스틱(Microplastics, MPs)의 탐지 및 정량화를 위한 기계 학습(Machine Learning, ML)의 응용 가능성과 한계를 살펴보았습니다. 기존 기술들은 수작업 분석을 요구하기 때문에 대규모 스크리닝 연구에 비효율적입니다. 반면, 본 연구는 SEM 이미징과 ML 기반 객체 탐지의 조합을 활용하여 이러한 한계를 극복할 수 있는 잠재력을 보여주었습니다.

- **Technical Details**: 연구에 사용된 기술적 요소로는 주로 YOLO (You Only Look Once) 모델을 활용했습니다. YOLO는 이미지에서 객체를 탐지하고 위치를 파악하는 효과적인 방법으로, 다양한 분야에서 성공적으로 적용되어 왔습니다. 특히, YOLOv5 모델은 빠르고 정확하게 미세플라스틱을 탐지하는 데 적합하다는 사실이 강조되었습니다.

- **Performance Highlights**: 연구 결과, YOLO 모델의 최적화가 탐지 및 정량화 작업의 질에 미치는 영향을 확인했습니다. 하지만 전문가 레이블이 부여된 데이터의 부족이라는 과제가 여전히 존재하며, 이는 기계 학습 모델의 신뢰성을 위한 필수 요소입니다. 다양한 심층 학습 접근법들이 미세플라스틱의 탐지 정확도를 90% 이상으로 향상시킬 수 있다는 연구 결과도 포함되어 있습니다.



### Chimera: Compositional Image Generation using Part-based Concepting (https://arxiv.org/abs/2510.18083)
- **What's New**: 새롭게 제안된 Chimera 모델은 다양한 소스 이미지에서 특정 파트를 조합하여 새로운 객체를 생성할 수 있는 개인화된 이미지 생성 모델입니다. 이전 모델들은 사용자가 직접 제공하는 마스크나 주석 없이 이미지의 일부분을 변경하는 것이 불가능했지만, Chimera는 텍스트 지침만으로 이를 가능하게 합니다. 이 모델은 464개의 고유한 (part, subject) 쌍을 기반으로 구축된 데이터셋을 사용하여 훈련되었습니다.

- **Technical Details**: Chimera는 부품 조건(part-conditional) 가이드를 사용하는 커스터마이즈된 diffusion prior 모델로, 세부적인 세멘틱 정체성과 공간적 레이아웃을 유지하며 이미지를 생성합니다. 기존의 모델들이 필요로 하는 마스크나 주석 없이 전체 입력 이미지와 텍스트 프롬프트만으로 작업할 수 있으며, 훈련 과정에서 37,000개의 프롬프트를 생성하여 높은 충실도의 텍스트-투-이미지 모델로 이미지를 합성합니다. 또한, PartEval이라는 새로운 메트릭을 도입하여 생성된 이미지의 충실도와 구성 정확성을 평가합니다.

- **Performance Highlights**: Chimera는 인공지능 모델의 기본 성능을 2파트, 3파트, 4파트 생성에서 일관되게 유지하며, 인간 평가와 PartEval 메트릭에 의해 부품 정렬(part alignment)과 구성 정확성(compositional accuracy)에서 기존 모델보다 14% 더 높은 성과를 달성합니다. 시각적 품질 측면에서도 21% 향상된 결과를 나타내어, 더욱 복잡한 생성 작업에서도 일관되게 높은 품질을 유지합니다.



### HouseTour: A Virtual Real Estate A(I)gen (https://arxiv.org/abs/2510.18054)
Comments:
          Published on ICCV 2025

- **What's New**: 새로운 방법론인 HouseTour를 소개합니다. 기존의 3D 공간을 기반으로 자연어 요약과 3D 카메라 궤적을 생성하는 작업으로, COVID-19 팬데믹 중에 집 구경 비디오가 인기를 끌면서 이에 대한 수요가 증가했습니다. 이 방법은 전문 장비나 전문 지식 없이도 고품질의 비디오를 생성할 수 있게 합니다.

- **Technical Details**: HouseTour 방법은 주어진 이미지 집합으로부터 카메라 궤적을 생성하고 이를 VLM(vision-language model)에 통합하는 과정을 포함합니다. Diffusion process(확산 과정)를 사용하여 매끄러운 3D 카메라 궤적을 생성하고 이를 3D Gaussian splatting으로 시각화하여 결과 비디오를 합성합니다. 이를 위해 1200개 이상의 집 구경 비디오와 그에 따르는 카메라 궤적, 3D 재구성과 텍스트 설명을 포함한 HouseTour 데이터세트를 발표합니다.

- **Performance Highlights**: 실험 결과는 3D 카메라 궤적 통합이 텍스트 생성 과정에서 성능을 향상시켰음을 보여줍니다. 개별 작업과 최종 성과를 모두 평가하여 새로운 공동 메트릭을 도입하였으며, 이 방법론이 자동화된 프로페셔널 비디오 제작을 가능하게 함을 입증했기 때문에 부동산 및 관광 분야에서 큰 잠재력을 가집니다.



### TriggerNet: A Novel Explainable AI Framework for Red Palm Mite Detection and Multi-Model Comparison and Heuristic-Guided Annotation (https://arxiv.org/abs/2510.18038)
Comments:
          17 pages, 9 figures

- **What's New**: 이번 연구는 레드 팜 마이트(red palm mite) 감염을 조기 식별하고 효과적으로 관리하기 위한 머신 러닝(ML) 모델을 평가하고 비교하는 데 초점을 맞추고 있습니다. TriggerNet이라는 새로운 해석 가능한 AI 프레임워크는 Grad-CAM, RISE, FullGrad 및 TCAV를 통합하여 심층학습 모델의 시각적 설명을 생성합니다. 이 연구는 레드 팜 마이트 감염 문제를 해결하기 위해 다양한 식물 이미지와 고급 딥러닝 모델을 사용하고 있습니다.

- **Technical Details**: 연구에서는 11종의 식물(예: Arecanut, Date Palm, Coconut Palm 등)에서 RGB 이미지를 사용하여 모델을 훈련했습니다. CNN, EfficientNet, MobileNet, ViT, ResNet50, InceptionV3와 같은 최신 딥러닝 모델과 Random Forest, SVM, KNN과 같은 머신 러닝 분류기를 사용하여 식물 분류를 수행했습니다. 질병 분류는 건강한 식물과 다양한 질병 상태(예: Yellow Spots, Reddish Bronzing 등)로 나뉘어 진행되었습니다.

- **Performance Highlights**: Snorkel을 사용하여 질병 클래스를 효율적으로 레이블링하였으며, 이는 수작업 주석 시간을 줄이고 데이터셋의 신뢰성을 높였습니다. TriggerNet은 Red palm mite 감염의 조기 식별에 있어 맥락을 바탕으로 한 시각적 설명을 제공하여 실용성을 높이는 효과를 보여주고 있습니다. 이러한 기법은 능률화된 농업 관리에 기여할 수 있는 중요한 도구가 될 것입니다.



### SAVANT: Semantic Analysis with Vision-Augmented Anomaly deTection (https://arxiv.org/abs/2510.18034)
Comments:
          8 pages, 5 figures

- **What's New**: SAVANT는 자율 주행 시스템의 이상 감지에서 구조화된 추론 프레임워크를 제공합니다. 이 연구는 Vision Language Models (VLMs)의 비효율적인 접근 방식을 극복하며, 고급 분석을 통해 기존의 모델보다 더 높은 정확도를 달성합니다. SAVANT는 입력 이미지로부터 복잡한 장면을 네 개의 의미 레이어(Street, Infrastructure, Movable Objects, Environment)로 나누어 체계적으로 평가함으로써 이상 상황을 효과적으로 감지합니다.

- **Technical Details**: SAVANT의 핵심 구성 요소는 두 단계의 파이프라인으로, 첫 번째 단계에서는 구조화된 장면 설명을 추출하고 두 번째 단계에서는 다중 모달 평가를 수행합니다. 이 구조는 기존 등록된 객체들이 맥락적으로 부적절한 배치에 있는지를 감지하는 이진 분류 작업으로 이상 감지를 정의합니다. 여러 의미 레이어를 통해 교통 장면에 대한 심도 있는 분석을 제공하며, 이는 자율 주행 시스템의 안전성을 높이는 데 기여합니다.

- **Performance Highlights**: SAVANT는 실제 주행 시나리오에서 89.6%의 재현율과 88.0%의 정확성을 달성하며, 기존 비구조적 기준선에 비해 현저한 성능 향상을 보여줍니다. 또한 7B 파라미터의 오픈 소스 모델인 Qwen2.5VL을 통해 90.8%의 재현율과 93.8%의 정확성을 달성하여, 비용 문제 없이 로컬 배포를 가능하게 합니다. 이 연구는 9,640개의 실제 주행 이미지에 대한 자동 라벨링을 제공함으로써 데이터 부족 문제를 해결하는 실질적인 경로를 제시합니다.



### ViBED-Net: Video Based Engagement Detection Network Using Face-Aware and Scene-Aware Spatiotemporal Cues (https://arxiv.org/abs/2510.18016)
Comments:
          10 pages, 4 figures, 2 tables

- **What's New**: 이번 논문에서는 비디오 기반의 학생 참여도 탐지를 위한 새로운 딥러닝 프레임워크인 ViBED-Net(Video-Based Engagement Detection Network)을 제안합니다. 이 모델은 다중 스트림 아키텍처를 사용하여 얼굴 표현과 전체 장면 맥락을 동시 처리하여 학습자의 참여도를 평가합니다. 특히 EfficientNetV2를 사용하여 공간적 특징을 효율적으로 추출하고, LSTM(Long Short-Term Memory) 네트워크와 Transformer 인코더를 활용한 시간적 모델링 전략으로 참여도를 분석합니다.

- **Technical Details**: ViBED-Net은 비디오 데이터로부터 학생 참여도를 탐지하기 위해 얼굴 영역과 전체 비디오 캡처를 동시에 처리합니다. 이 모델은 친숙한 LSTM뿐만 아니라 장기 의존성을 처리할 수 있는 Transformer 기반 아키텍처도 활용하여 비디오의 시간적 의존성을 잘 모델링합니다. 제안된 방법은 DAiSEE 데이터셋에서 뛰어난 성능을 보이며, 얼굴 인식과 장면 인식을 동시에 활용하는 점에서 독창적입니다.

- **Performance Highlights**: ViBED-Net은 LSTM을 사용할 경우 73.43%의 정확도를 기록하여 기존의 최첨단 접근법을 초월합니다. 이 모델은 또한 표본이 부족한 참여도 클래스에 대한 데이터 증강 기법을 적용하여 성능을 강화하였습니다. 최종 결과는 모델의 효율성과 유연성을 잘 보여줍니다, 이는 교육, 사용자 경험 연구 및 개인화된 콘텐츠 개발 등 다양한 분야에 응용될 수 있습니다.



### ManzaiSet: A Multimodal Dataset of Viewer Responses to Japanese Manzai Comedy (https://arxiv.org/abs/2510.18014)
Comments:
          ICCV 2025 Workshop on Affective & Behavior Analysis in-the-Wild (ABAW), Honolulu, HI, USA (Oct 19, 2025, HST). 11 pages, 5 figures

- **What's New**: 우리는 ManzaiSet을 소개합니다. 이는 일본 만자이(comedy) 코미디에 대한 시청자의 반응을 담고 있는 최초의 대규모 다중모드(multimodal) 데이터셋으로, 241명의 참여자가 랜덤한 순서로 최대 10회의 전문 공연을 시청하면서 캡처한 얼굴 비디오 및 오디오를 포함하고 있습니다. 이 연구는 감정 컴퓨팅 분야의 서구 중심 편향(asian bias)을 극복하기 위한 첫걸음으로, 일본 문화 특유의 코미디에 대한 시청자 반응을 대규모로 수집하여 문화적 맥락에서의 감정 표현을 탐구합니다.

- **Technical Details**: 데이터셋은 241명의 일본 관객이 동일한 전문 만자이 공연을 시청하며 생성한 191.8시간의 동기화된 얼굴 비디오 및 오디오 데이터로 구성되어 있습니다. 세 가지 주요 발견이 있었으며, 첫째로 k-means clustering을 통한 분석 결과 세 가지 distinct viewer 유형이 발견되었습니다: 높은 안정적인 감상자(72.8%), 낮고 가변적인 거부자(13.2%), 가변적인 개선자(14.0%)입니다. 또한 개인 수준 분석에서 긍정적인 시청 순서 효과가 나타났습니다(p < 0.001), 이는 피로(hypothesis) 가설과 상반됩니다.

- **Performance Highlights**: 이 데이터셋의 분석은 문화적으로 인지된 감정 AI 개발과 비서구 지역에 맞춤화된 개인화된 엔터테인먼트 시스템에 즉각적인 응용 가능성을 제공합니다. 특히, 자동화된 유머 분류 모델(77개 인스턴스, 131개 레이블)을 통해 viewer level 반응 모델링이 진행되었으나, 유형별 차이는 발견되지 않았습니다(FDR 보정 후). 이는 다양한 시청자들이 어떻게 동일한 코미디 자극에 대해 반응하는지를 분석하는 새로운 길을 열었습니다.



### Investigating Demographic Bias in Brain MRI Segmentation: A Comparative Study of Deep-Learning and Non-Deep-Learning Methods (https://arxiv.org/abs/2510.17999)
- **What's New**: 이 연구는 MRI 이미지를 사용하여 좌우 핵상(Kernel Accumbens, NAc) 세분화를 위한 세 가지 딥러닝 모델(UNesT, nnU-Net, CoTr)과 전통적인 아틀라스 기반 방법(ANTs)의 성능을 평가합니다. 연구는 인종 및 성별과 관련된 편향을 고려하며, 백인 및 흑인 그룹의 데이터셋을 통해 세분화 모델의 공정성을 정량적으로 측정합니다. 각 모델의 세분화 성능을 비교하고, 인종 및 성별에 따른 체적 변화를 분석하여 기존 연구의 한계를 보완하고자 했습니다.

- **Technical Details**: 연구는 Human Connectome Project의 데이터를 활용하며, 각 참여자의 인구통계학적 정보를 포함한 T1 가중 MRI 이미지를 분석합니다. 네 개의 인구 하위 그룹(흑인 여성, 흑인 남성, 백인 여성 및 백인 남성)을 이용하여 딥러닝 모델과 전통적인 방법의 편향을 비교하였습니다. 각 모델은 수동으로 라벨링된 골드 스탠다드 세분화 데이터를 통해 학습되며, 세분화 성능에 대한 다양한 통계 분석이 포함됩니다.

- **Performance Highlights**: 모델의 성능 분석 결과, 인종이 일치하는 데이터로 훈련된 UNesT와 ANTs 모델은 현저한 세분화 정확도를 보여주었습니다. 반면 nnU-Net은 인구 통계학적 특성과 무관하게 강력한 성능을 유지했습니다. 연구 결과, 수동으로 세분화한 데이터와 편향된 모델에서 관찰된 성별 효과가 유사하게 나타났지만, 인종 효과는 대부분의 모델에서 사라지는 경향을 보였습니다.



### 3D Weakly Supervised Semantic Segmentation via Class-Aware and Geometry-Guided Pseudo-Label Refinemen (https://arxiv.org/abs/2510.17875)
- **What's New**: 이 논문에서는 3D 약한 지도 세분화(3D WSSS)를 위한 새로운 방법을 제안합니다. 기존 방법이 Class Activation Maps 및 사전 훈련된 Vision-Language Models를 활용했던 것과 달리, 제안된 방법은 3D 기하학적 선행 지식을 통합하여 고품질의 의사 라벨을 생성합니다. Class-Aware Label Refinement와 Geometry-Aware Label Refinement를 통해 더욱 균형 잡히고 정확한 의사 라벨을 만들어냅니다.

- **Technical Details**: 제안된 방법은 두 단계의 패러다임을 따르며, 첫번째 단계에서 Class-Aware Label Refinement (CALR) 모듈과 Geometry-Aware Label Refinement (GALR) 모듈을 사용하여 3D WSSS 조건 하에 고품질의 포인트 레벨 의사 라벨을 생성합니다. CALR은 각 카테고리에서 가장 자신감 있는 상위 라벨을 보존하여 균형 잡힌 감독을 유지하고, GALR은 3D 기하학적 선행 지식을 통해 의사 라벨의 정확성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 ScanNet 및 S3DIS 벤치마크에서 최첨단 성능을 달성했습니다. 특히 비지도 설정에서도 경쟁력 있는 성능을 유지하여, 기하학적 및 의미적 정보 활용의 효과성을 입증하였습니다. 이러한 결과는 제안된 접근 방식이 3D WSSS 개발을 위한 강력하고 신뢰할 수 있는 프레임워크임을 보여줍니다.



### Auditing and Mitigating Bias in Gender Classification Algorithms: A Data-Centric Approach (https://arxiv.org/abs/2510.17873)
- **What's New**: 이 논문에서는 성별 분류 시스템이 훈련 데이터의 인구 통계적 불균형을 상속하고 확대함을 지적합니다. 저자들은 UTKFace와 FairFace라는 두 가지의 균형 잡힌 데이터 세트를 기반으로 동일한 MobileNetV2 분류기를 학습시키고, 이 모델들이 여전히 성별 및 인종에 있어 양성 오분류율에서 심각한 편향을 보인다는 것을 발견했습니다. 이를 해결하기 위해 FairFace와 UTKFace 이미지를 혼합하여 BalancedFace라는 새로운 공개 데이터 세트를 만들어 누락된 인구 통계적 격차를 채웠습니다.

- **Technical Details**: BalancedFace 데이터 세트는 189개의 연령, 인종, 성별의 교차 교차로에서 하위 그룹의 비율을 균형잡기 위해 설계되었습니다. 이 데이터 세트는 오직 실제 이미지만을 사용하여 구성되어 있으며, 이를 통해 모델 훈련 시 최대 True Positive Rate 격차를 50% 이상 줄이고 평균 Disparate Impact 점수를 1.0에 63% 더 가깝게 개선했습니다. 논문에서는 공정성을 고려한 훈련 프레임워크도 제안하며, 이는 적대적 학습(adversarial learning), 동등 확률 정규화(equalized odds regularization) 및 재가중치(re-weighting)를 결합하여 불균형을 완화함과 동시에 경쟁력 있는 정확도를 유지합니다.

- **Performance Highlights**: BalancedFace 데이터 세트를 사용해 훈련된 표준 분류기는 성별 및 인종 하위 그룹 간의 동등 확률 및 차별 영향을 줄이는 데 효과적임을 입증했습니다. 이는 공정한 성별 분류를 위한 데이터 중심 중재의 중요성을 강조하며, 일반적인 데이터 세트에 비해 실질적인 성능 개선을 보여주었습니다. 이 연구의 결과는 공정한 성별 분류 연구를 위한 귀중한 자원으로, 앞으로의 연구에 있어 유용하게 활용될 수 있을 것입니다.



### GAN-based Content-Conditioned Generation of Handwritten Musical Symbols (https://arxiv.org/abs/2510.17869)
Comments:
          15 pages, 5 figures, Accepted at ICDAR workshop GREC 2025

- **What's New**: 이번 연구에서는 Generative Adversarial Network (GAN)을 활용하여 역사적 음악 스코어의 자연스러운 필기체를 생성하는 방법을 소개합니다. 이를 통해 Optical Music Recognition (OMR) 시스템의 훈련 세트를 확장 시킬 수 있으며, 이전의 작성된 필기 데이터의 부족 문제를 해결해 줄 것으로 기대됩니다. Smashcima 소프트웨어를 통해 생성된 심볼들을 전체 스코어로 조합함으로써, 사실감 높은 음악 스코어를 만드는 것을 목표로 하고 있습니다.

- **Technical Details**: 연구에서는 GAN 아키텍처를 활용하여 음악 기호를 생성하며, 이 기호들은 Smashcima 소프트웨어에서 전체 음악 시트로 변환됩니다. GAN 구조는 제너레이터와 디스크리미네이터 두 가지 네트워크로 이루어져 있으며, 이를 통해 실세계에서 차별화된 이미지를 생성하도록 훈련됩니다. 이 과정은 적응된 GAN 모델을 기반으로 하며, 특히 필기 스타일에 맞춰 출력을 조정할 수 있는 방식으로 설계되었습니다.

- **Performance Highlights**: 실험 결과, 생성된 음악 스코어는 높은 사실감을 제공하여 OMR 시스템의 훈련을 위한 데이터로 사용될 가능성을 보여주었습니다. 특히, 심볼 수준의 주석을 활용하여 더 작은 클래스의 데이터 생성을 용이하게 하여, 컴퓨터의 효율성을 크게 개선하는 것으로 나타났습니다. 이러한 접근법은 기존의 수작업 전사 방식보다 더 나은 성능을 발휘하며, 필기체 변형 스타일에도 유연하게 적용될 수 있습니다.



### MUSE: Model-based Uncertainty-aware Similarity Estimation for zero-shot 2D Object Detection and Segmentation (https://arxiv.org/abs/2510.17866)
Comments:
          11 pages with 6 figures

- **What's New**: 이번 연구에서는 MUSE (Model-based Uncertainty-aware Similarity Estimation)라는 새로운 프레임워크를 소개합니다. MUSE는 훈련이 필요 없는 모델 기반의 제로샷(Zero-shot) 2D 객체 탐지 및 분할을 위해 설계되었습니다. 이 프레임워크는 3D 보지 못한(uncaptured) 객체에서 렌더링된 2D 다중 뷰 템플릿과 입력 쿼리 이미지에서 추출한 2D 객체 제안을 활용합니다.

- **Technical Details**: MUSE의 임베딩 단계에서는 클래스(Class)와 패치(Patch) 임베딩을 통합합니다. 패치 임베딩은 일반화된 평균 풀링(GeM)을 사용하여 정규화되어 전역(Global) 및 지역(Local) 표현을 효율적으로 캡처합니다. 일치(matching) 단계에서는 절대 및 상대 유사성 점수를 결합하는 공동 유사성 지표를 활용하여 어려운 상황에서도 일치의 견고성을 향상시킵니다.

- **Performance Highlights**: MUSE는 추가적인 훈련 또는 미세 조정 없이도 BOP Challenge 2025에서 최첨단(state-of-the-art) 성능을 달성하며, Classic Core, H3, Industrial 트랙에서 1위를 기록했습니다. 이러한 결과는 MUSE가 제로샷 2D 객체 탐지 및 분할을 위한 강력하고 일반화 가능한 프레임워크를 제공함을 보여줍니다.



### InsideOut: Integrated RGB-Radiative Gaussian Splatting for Comprehensive 3D Object Representation (https://arxiv.org/abs/2510.17864)
Comments:
          Published at ICCV 2025

- **What's New**: 이 논문에서는 3D Gaussian splatting(3DGS)의 확장판인 InsideOut을 소개합니다. InsideOut은 고해상도 RGB 표면 디테일과 X-ray 구조 간의 간극을 연결하는 기술입니다. RGB와 X-ray 이미징의 융합은 의료 진단, 문화재 복원, 제조업 등의 분야에서 매우 중요합니다.

- **Technical Details**: 이 연구에서는 새로운 RGB와 X-ray 쌍 데이터를 수집하고, 계층적 피팅(hierarchical fitting)을 수행하여 RGB와 X-ray의 방사형 Gaussian splats를 정렬합니다. 또한, 일관된 내부 구조를 보장하기 위해 X-ray 참조 손실(X-ray reference loss)을 제안합니다. 이러한 방법은 두 모달리티(modality) 간의 상이한 데이터 표현의 문제와 제한된 쌍 데이터셋의 문제를 효과적으로 해결합니다.

- **Performance Highlights**: InsideOut은 3DGS의 적용 가능성을 크게 확장하며 다양한 분야에서 시각화(visualization), 시뮬레이션(simulation), 비파괴 검사(non-destructive testing) 기능을 향상시킵니다. 이 접근법은 서로 다른 데이터 형식 간의 도전 과제를 해결하고, 다양한 활용 방안을 제시합니다.



### Robotic Classification of Divers' Swimming States using Visual Pose Keypoints as IMUs (https://arxiv.org/abs/2510.17863)
- **What's New**: 이 논문은 스쿠버 다이빙 시 안전을 모니터링하기 위해 새로운 하이브리드 접근법을 소개합니다. 기존의 이미지 분석이나 장착형 IMU(관성 측정 장치) 데이터 수집 개념을 넘어서, 3D 인체 관절 키포인트를 기반으로 가상의 ‘pseudo-IMU’을 생성하는 방식으로 통신 문제를 해결했습니다. 수중 환경에서는 기존 무선 센서와의 통신이 어려운 문제를 해결하여, AUV(자율 수중 차량)가 개입 없이도 다이버의 생체 신호를 분석할 수 있도록 합니다.

- **Technical Details**: 논문에서는 AUV가 비디오 피드를 활용하여 다이버의 중대한 의료 사건을 탐지하는 새로운 방법을 제안합니다. 이 시스템은 3D 포즈 추정을 통해 다이버의 관절 데이터를 수집하고, 이를 기반으로 전이 가속도와 회전 가속도를 추정합니다. AUV는 이를 통해 실시간으로 다이버의 수영 상태를 분류할 수 있으며, 이 과정에서 고유한 데이터셋과 딥러닝 모델을 활용하여 건강 상태를 평가합니다.

- **Performance Highlights**: 연구에서는 3305개의 이미지 데이터를 통해 다이버가 수영에서 비수영으로 상태 전환하는 과정을 성공적으로 분류하는 실험을 수행했습니다. 기존의 방식을 넘어 AUV가 비디오 데이터를 처리하여 실시간으로 다이버의 상태를 모니터링할 수 있는 가능성을 입증하였습니다. 이러한 혁신적인 접근법은 수중 환경 내에서 다이버의 건강 상태를 효과적으로 파악하고, 안전한 다이빙을 지원하는 데 중요한 역할을 할 것입니다.



### Shortcutting Pre-trained Flow Matching Diffusion Models is Almost Free Lunch (https://arxiv.org/abs/2510.17858)
Comments:
          NeurIPS 2025

- **What's New**: 이번 논문에서는 새로운 velocity field self-distillation을 통해 대규모 사전 훈련된 flow matching diffusion 모델을 효율적인 몇 단계 샘플러로 변환하는 초효율적인 post-training 방법을 제안합니다. 기존 모델은 특정 step-size embedding이 필요하지만, 본 연구에서는 이에 대한 의존성을 제거하여 더 강력한 shortcut 메커니즘을 도입하였습니다. 또한, 이 방법은 처음으로 많은 파라미터를 가진 diffusion 모델을 대상으로 한 few-shot distillation 기법을 성공적으로 보여줍니다.

- **Technical Details**: 우리는 ShortCutting Flow Matching (SCFM)이라는 새로운 프레임워크를 소개하여 velocity space 내에서 작동하며, 모든 타임스텝에 걸쳐 선형 일관성을 유지합니다. 이는 teacher와 온라인으로 상속된 student 사이의 듀얼 타겟 distillation 목표를 통해 유도됩니다. SCFM은 self-distillation을 통해 고도로 효율적인 훈련을 가능하게 하며, 대량의 데이터세트를 필요로 하지 않습니다.

- **Performance Highlights**: SCFM을 활용한 실험에서는 12B 파라미터를 가진 Flux 모델을 A100 GPU에서 단 하루 만에 3단계 샘플러로 변환하여 경쟁력 있는 성능을 달성했습니다. 이 방법은 정량적 평가 및 시각적 품질 모두에서 최신 기술을 자랑하며, 기존의 adversarial distillation 기법을 사용할 필요 없이 이루어졌습니다. 우리는 10개의 훈련 샘플만으로도 이와 같은 성능을 달성하여 few-shot distillation의 가능성을 입증했습니다.



### CMIS-Net: A Cascaded Multi-Scale Individual Standardization Network for Backchannel Agreement Estimation (https://arxiv.org/abs/2510.17855)
- **What's New**: 본 논문에서는 CMIS-Net이라는 새로운 Cascaded Multi-Scale Individual Standardization Network를 제안합니다. 이 네트워크는 관찰된 표정에서 개인별 중립 기준을 제거하여 개인화된 backchannel 특징을 추출합니다. 또한 데이터 불균형 문제를 해결하기 위한 암묵적 데이터 증강 모듈을 도입하여 모델 일반화를 개선합니다.

- **Technical Details**: CMIS-Net은 프레임 수준과 시퀀스 수준 모두에서 동작하여 각 개인의 기준선에서의 상대적 변화에 초점을 맞춥니다. 이는 미세한 순간 표현과 담화 수준 패턴 모두에서 정보를 이용하여 backchannel 표현을 정규화합니다. 또한, 우리의 프레임워크는 개별화된 피드백을 제공하여 특히 유용한 감정 인식 작업에 적용될 수 있습니다.

- **Performance Highlights**: 조사 결과 CMIS-Net은 개인차 및 데이터 불균형 문제를 효과적으로 처리하며, backchannel 동의 감지에서 최첨단 성능을 달성했습니다. 추가적으로, 청각적(backchannel) 및 시각적(backchannel) 반응 간의 차이를 강조하며, 각각의 독특한 특성에 맞춘 접근 방식의 필요성을 제안합니다.



### Provenance of AI-Generated Images: A Vector Similarity and Blockchain-based Approach (https://arxiv.org/abs/2510.17854)
- **What's New**: 이 논문에서는 AI 생성 이미지를 감지하기 위한 내장 기반(embedding-based) 프레임워크인 EmbedAIDetect를 제안합니다. 이 시스템은 이미지 내장과 벡터 유사성(vector similarity)을 사용하여 AI 생성 이미지와 인간 생성 이미지를 구분합니다. 기존의 검출 방식들이 특정 모델에 맞춰져 있어 일반화에 한계가 있었던 점을 보완하고, 훈련 없이도 효율적으로 작동하는 시스템을 목표로 하고 있습니다.

- **Technical Details**: EmbedAIDetect 시스템은 Vision Transformer(ViT) 모델을 활용하여 이미지의 내장을 생성하고, 이를 Cosine Distance를 통해 유사성을 측정하여 AI 생성 여부를 판단합니다. 시스템은 ChromaDB라는 오픈 소스 벡터 데이터베이스를 사용하여 고차원 공간에서 이미지 내장을 저장 및 검색하며, 또한 블록체인 기술을 응용하여 데이터의 무결성을 확보합니다. 사용자 친화적인 웹 인터페이스를 통하여, 사용자는 이미지 업로드 후, AI 생성 여부 및 검증 가능성을 한 번에 확인할 수 있습니다.

- **Performance Highlights**: 시스템의 성능 실험 결과, EmbedAIDetect는 다양한 AI 및 인간 생성 이미지를 포함하는 데이터셋에서 안정적인 감지 성능을 보였습니다. 중간에서 높은 변형(perturbation)이 있어도 내장 서명(embedding signatures)에 미치는 영향이 미미하다는 것을 보여주며, 원본 이미지와의 유사도를 유지합니다. 전반적으로, 이 프레임워크는 accuracy와 computational efficiency의 균형을 잘 맞추고 있으며 AI 생성 이미지 감지에 있어 효과적인 접근방식을 제시합니다.



### Pre to Post-Treatment Glioblastoma MRI Prediction using a Latent Diffusion Mod (https://arxiv.org/abs/2510.17851)
Comments:
          10 pages, 4 figures. Presented to the Deep Generative Models Workshop of MICCAI (DGM4MICCAI)

- **What's New**: 이번 연구는 적극적인 뇌종양인 교모세포종(GBM)의 조기 치료 반응 예측을 위해 새로운 Latent Diffusion Model(LDM)을 제안합니다. 이 모델은 치료 전(MRI) 이미지를 바탕으로 치료 후(MRI) 영상을 생성하는 것을 목표로 하며, 환자 생존 정보를 활용하여 이미지 생성 품질을 향상시킵니다. 이로써 임상에서의 개인 맞춤 치료에 기여할 수 있는 가능성을 제시합니다.

- **Technical Details**: 제안된 모델은 2D-LDM으로 구성되며, Encoding과 Decoding의 과정을 통해 낮은 차원의 잠재 공간으로 이미지를 압축하고 복원합니다. 이를 위해 Vector Quantized-Variational AutoEncoder(VQ-VAE)를 사용하여 이미지의 잠재 표현을 학습하였고, Gross Tumor Volume(GTV) 정보와 결합하여 효과적인 예측을 도모합니다. Diffusion Model(확산 모델)의 역 확산 과정은 UNet를 통해 학습되며, 이는 생체의학 영상 합성에서 최신 성능을 보여줍니다.

- **Performance Highlights**: 실험 결과, 제안된 LDM은 교모세포종 환자에서 치료 전 이미지를 바탕으로 치료 후 이미지를 성공적으로 생성할 수 있음을 입증하였습니다. 연구는 140명의 GBM 환자 데이터를 통해 진행되었으며, 양질의 이미지를 생성하는 능력이 입증되었습니다. 이는 종양의 치료 반응을 조기에 예측하여 임상 의사 결정에 도움을 줄 수 있는 도구가 될 것입니다.



### CoIDO: Efficient Data Selection for Visual Instruction Tuning via Coupled Importance-Diversity Optimization (https://arxiv.org/abs/2510.17847)
Comments:
          22 pages, 8 figures, 39th Conference on Neural Information Processing Systems (NeurIPS 2025)

- **What's New**: 이번 논문에서는 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 성능을 개선하기 위한 새로운 프레임워크인 CoIDO(Coupled Importance-Diversity Optimization)를 소개합니다. CoIDO는 데이터 중요도(importance)와 다양성(diversity)을 동시에 최적화하여 대규모 데이터셋에서 발생하는 계산적 비용 문제를 해결합니다. 기존의 방법들과는 달리, CoIDO는 전체 데이터셋을 처리하는 것이 아니라 작은 샘플에서 배운 점수를 사용하여 효율적인 데이터 선택을 가능하게 합니다.

- **Technical Details**: CoIDO 프레임워크는 데이터의 중요도와 다양성을 동시에 고려하는 두 가지 목표 최적화를 수행합니다. 이 방법은 엄격한 선택 알고리즘 없이도 다양한 데이터를 효과적으로 조율할 수 있으며, 단지 20%의 랜덤 샘플을 통해 점수를 계산하여 전체 데이터셋을 평가합니다. CoIDO 스코어는 Homoscedastic uncertainty 기반의 최적화 방법을 사용하여 각 데이터 샘플의 중요도와 다양성을 종합적으로 고려하여 평가합니다.

- **Performance Highlights**: 실험 결과, CoIDO를 통해 선택된 20%의 샘플로 LLaVA-1.5-7B 모델을 파인튜닝(fine-tuning)했을 때, 전체 데이터셋을 활용한 경우에 비해 평균 98.2%의 성능을 기록하였습니다. 이는 CoIDO 방법이 데이터 선택의 효율성을 극대화할 뿐만 아니라, 높은 성능을 유지한다는 것을 의미합니다. 이러한 접근은 연구 및 개발에 필요한 자원을 줄이면서도 우수한 성과를 이룰 수 있도록 합니다.



### MAT-Agent: Adaptive Multi-Agent Training Optimization (https://arxiv.org/abs/2510.17845)
Comments:
          Acceptance to NeurIPS 2025 Main Track

- **What's New**: 본 논문에서는 동적 환경에 적합한 훈련 전략을 목표로 하는 다중 에이전트 프레임워크인 MAT-Agent를 제안합니다. MAT-Agent는 훈련 과정을 협력적이며 실시간 최적화 프로세스로 재구성하며, 자동 에이전트를 통해 데이터 증강(data augmentation), 최적화기(optimizer), 학습률(learning rate), 손실 함수(loss function)를 동적으로 조정합니다. 이로 인해 정확도, 희귀 클래스 성능 및 훈련 안정성을 종합적으로 고려한 보상(composite reward)을 통해 모델의 성능을 극대화할 수 있습니다.

- **Technical Details**: MAT-Agent는 훈련의 각 단계에서 동적으로 훈련 구성(𝐂t)을 조합하며, 이는 데이터 증강, 최적화기 선택, 학습률 스케줄ing, 손실 함수 설계를 조정하는 4개의 자율적이고 적응형 에이전트로 구성되어 있습니다. 각 에이전트는 훈련 상태(sts_{t})에 대한 글로벌 신호를 인지하며, 각 훈련 구성 요소에 대한 행동을 선택합니다. 이들 에이전트는 현재 구성의 효과성을 정량화하는 보상 신호를 받고, 이를 통해 정책을 지속적으로 업데이트하여 전체적으로 작용하는 동적 훈련 프로세스를 실현합니다.

- **Performance Highlights**: MAT-Agent는 Pascal VOC, COCO 및 VG-256 데이터셋에서 실험을 통해 뛰어난 성능을 입증하였습니다. Pascal VOC에서 mAP는 97.4로 SOTA 성능인 PAT-T의 96.2에 비해 우수하며, OF1은 92.3, CF1은 91.4를 기록하였습니다. COCO 데이터셋에서는 mAP 92.8, OF1 88.2, CF1 87.1을 달성하고, VG-256에서도 각각 60.9, 70.8, 61.1 성과를 보여줍니다. 전반적으로 MAT-Agent는 빠른 수렴과 강력한 도메인 전이 일반화를 제공하여 복잡한 비주얼 모델 최적화에 혁신적인 솔루션을 제시합니다.



### LightMem: Lightweight and Efficient Memory-Augmented Generation (https://arxiv.org/abs/2510.18866)
Comments:
          Work in progress

- **What's New**: 최근의 연구 결과에 따르면, 대형 언어 모델(LLM)의 강력한 능력에도 불구하고 과거 상호작용 정보를 효과적으로 활용하는 데 한계가 있습니다. 이 논문에서는 LightMem이라는 새로운 메모리 시스템을 소개하며, 인간 기억의 Atkinson-Shiffrin 모델에 영감을 받아 구성되었습니다. LightMem은 정보 저장 및 검색 방식을 최적화하여 메모리 시스템의 성능과 효율성을 균형 있게 구현하고 있습니다.

- **Technical Details**: LightMem은 세 가지 상호 보완적인 단계로 메모리를 구성합니다. 첫째, 인지에 영감을 받은 감각 메모리는 관련 없는 정보를 신속하게 필터링하고 주제에 따라 정보를 그룹화합니다. 둘째, 주제 인식 단기 메모리는 이러한 주제 기반 그룹을 통합 및 요약하여 구조화된 접근을 가능하게 합니다. 마지막으로, 수면 시간 업데이트를 가진 장기 메모리는 오프라인 절차를 통해 통합을 온라인 추론과 분리하여 관리합니다.

- **Performance Highlights**: LightMem은 LongMemEval에서 강력한 기준선을 초과하여 정확도에서 최대 10.9%의 개선을 보여주며, 토큰 사용량은 최대 117배, API 호출은 159배, 실행 시간은 12배 이상 줄였습니다. 또한, Case study를 통해 오프라인 수면 시간 통합이 장기적인 지식 업데이트의 신뢰성을 높이고 정보 손실 및 불일치를 완화하는 데 기여함을 보여주었습니다.



### Seg the HAB: Language-Guided Geospatial Algae Bloom Reasoning and Segmentation (https://arxiv.org/abs/2510.18751)
- **What's New**: 이 논문에서는 ALGae Observation and Segmentation (ALGOS)라는 새로운 시스템을 소개합니다. 이 시스템은 임계 매개변수에 대한 내러티브 추론(segmentation-and-reasoning)과 해양 생태계의 유해 조류 발생 모니터링을 위해 원거리 감지(image understanding) 분야에 혁신을 가져옵니다. 인간의 평가를 통해 품질 높은 segmentation mask를 구성하고, NASA의 Cyanobacteria Aggregated Manual Labels (CAML) 를 활용하여 심각도 예측을 위한 비전-언어 모델을 미세 조정합니다.

- **Technical Details**: ALGOS는 원거리 감지 이미지에서 유해 조류 발생의 심각도를 평가하기 위해 비전-언어 모델을 통합한 통합적 접근을 사용합니다. GeoSAM이라는 도구를 통해 사람이 평가하는 과정을 포함하여 고급 픽셀 레벨 segmentation mask를 생성하고, 이를 기반으로 실시간 모니터링을 가능하게 합니다. 또한 심각도 레벨을 평가하기 위해 다섯 단계의 정량적 기준을 설정하고, 자연어 쿼리를 통해 해당 심각도를 추론할 수 있도록 하였습니다.

- **Performance Highlights**: 실험 결과, ALGOS는 기존의 segmentation 모델보다 공간 정확도(spatial accuracy)에서 유의미한 개선을 보였으며, 심각도 예측에서도 기존의 비전-언어 모델보다 우수한 성능을 달성하였습니다. 이 연구는 유해 조류 모니터링의 자동화와 생태적 평가 및 공공 건강 결정을 위한 정밀한 공간적 추론의 필요성을 강조합니다. ALGOS는 GIS 및 원거리 감지 기술의 발전을 활용하여 지속 가능한 수자원 관리에 기여할 수 있는 가능성을 보여주는 성공적인 사례입니다.



### Prototyping an End-to-End Multi-Modal Tiny-CNN for Cardiovascular Sensor Patches (https://arxiv.org/abs/2510.18668)
Comments:
          Submitted to the IEEE Journal of Biomedical And Health Informatics

- **What's New**: 본 논문에서는 심혈관 질환(CVD)의 조기 발견과 예방을 위해, 신체 착용 센서 장치에서 수집된 생리 신호를 분석하는 심층 학습 모델을 평가합니다. 연구의 핵심은 자원 제약이 있는 의료 엣지 디바이스에서의 효율적인 이진 분류 문제 해결을 위한 변화하는 합성곱 신경망(CNN) 모델을 제안하는 것입니다. 제안된 모델은 Physionet Challenge 2016 데이터셋의 동기화된 심전도(ECG) 및 청진기(PCG) 기록을 활용하여 훈련 및 검증됩니다.

- **Technical Details**: 제안된 모델은 초기 데이터 융합(early fusion) 기법을 사용하여 ECG와 PCG를 조합하여 분류 문제를 해결합니다. CNN 블록의 효율적인 설계를 통해 메모리 소모와 계산 비용을 기존 최첨단 모델보다 세 배 적게 유지하면서 경쟁력 있는 정확도를 달성합니다. 또한, 특정 마이크로컨트롤러와 실험 센서 장치에서의 에너지 소비를 분석함으로써, 온디바이스 추론 방식이 지속적인 데이터 스트리밍보다 에너지 효율적임을 증명합니다.

- **Performance Highlights**: 제안된 CNN 모델은 2016 Physionet Challenge 데이터셋을 기반으로 훈련되어 매우 낮은 메모리 소모와 계산 비용으로 효과적인 분류 성능을 보입니다. 기존의 EGC-PCG 분석 방법들과 비교했을 때, 매우 낮은 에너지 소비를 통해 긴 배터리 수명을 가능하게 하여 환자의 편안함을 증진시킬 수 있음을 보여줍니다. 나아가, 우리 모델은 자원 제약이 있는 착용 가능 장치에서 심층 학습 적용 가능성을 입증하며, 건강 모니터링을 위한 자동화 및 효율성을 크게 향상시킵니다.



### CUARewardBench: A Benchmark for Evaluating Reward Models on Computer-using Agen (https://arxiv.org/abs/2510.18596)
Comments:
          24 pages, 6 figures

- **What's New**: CUARewardBench는 컴퓨터를 사용하는 에이전트(CUA)의 평가를 위한 최초의 포괄적 보상 벤치마크를 제안합니다. 이는 결과 기반 보상 모델(ORM)과 프로세스 기반 보상 모델(PRM)을 모두 평가할 수 있도록 설계되었습니다. 또한, 다양한 소프트웨어 카테고리와 에이전트 아키텍처의 데이터 세트를 포함하여 실제 평가를 가능하게 합니다.

- **Technical Details**: CUARewardBench는 10개 소프트웨어 카테고리와 7개 에이전트 아키텍처에서 수집된 궤적을 포함하여 총 272개의 성공 주석과 346개의 단계 정확도 주석으로 구성됩니다. 이 논문에서는 효과적인 보상 모델을 정의하기 위한 궤적 수집 방법 및 공인된 주석 프로세스를 소개하여 신뢰성과 실제 적용 가능성을 높였습니다. 보상 모델은 궤적 성공과 단계별 정확성을 평가하여 현재 CUA RMs의 한계를 파악합니다.

- **Performance Highlights**: Unanimous Prompt Ensemble(UPE) 방법은 보상 모델의 신뢰성을 significantly 향상시킵니다. 실험 결과, UPE는 ORM에 대해 89.8%의 precision과 93.3%의 NPV를 달성하였고, PRM에 대해서는 각각 81.7%의 precision과 85.1%의 NPV를 기록했습니다. 이를 통해 단일 VLM 및 기존 앙상블 접근 방식에 비해 현저히 더 나은 성능을 입증하였습니다.



### Ensembling Pruned Attention Heads For Uncertainty-Aware Efficient Transformers (https://arxiv.org/abs/2510.18358)
- **What's New**: 이 논문은 Hydra Ensembles라는 새로운 앙상블 프레임워크를 소개합니다. 이 프레임워크는 프루닝(pruning) 기법을 사용하여 주의 헤드(attention head)를 제거하고 이를 통해 다양한 모델들을 생성합니다. 새로운 다중 헤드 주의(multi-head attention) 메커니즘을 적용하여 효율적인 UQ(Uncertainty Quantification)를 제공합니다.

- **Technical Details**: Hydra Ensembles는 프라이밍된(transformer-based) 모델에서 주의 헤드를 제거하여 다양한 서브네트워크(subnetwork)를 생성합니다. 이 때, 그룹화된 완전 연결층(grouped fully-connected layers)을 이용하여 모든 서브네트워크를 하나의 모델로 병합하게 됩니다. 기존의 깊은 앙상블(Deep Ensembles)과 달리, Hydra Ensembles는 각 구성원의 재훈련 없이도 높은 견고한 불확실성을 유지합니다.

- **Performance Highlights**: 이미지 분류 및 텍스트 분류 작업에서 Hydra Ensembles의 성능을 평가한 결과, Deep Ensembles와 비슷한 정확도와 보정(calibration) 메트릭스를 가지면서도 추론( inference) 비용을 대폭 줄였습니다. 특히, ImageNet-1K의 제로샷(zero-shot) 분류에서 기존의 최신 방법을 초과하는 성과를 보여 주목받고 있습니다.



### From Competition to Synergy: Unlocking Reinforcement Learning for Subject-Driven Image Generation (https://arxiv.org/abs/2510.18263)
- **What's New**: 이 논문에서는 Customized-GRPO라는 새로운 프레임워크를 제안하여 정체성 유지(Identity preservation)와 프롬프트 준수(Prompt adherence) 간의 균형을 맞추고 있습니다. 특히 두 가지 주요 혁신 요소인 Synergy-Aware Reward Shaping (SARS)와 Time-Aware Dynamic Weighting (TDW)을 도입하여 기초적인 GRPO의 한계를 극복합니다. 연구 결과, 이 방법이 기존의 GRPO보다 우수한 성능을 나타내며 경쟁 저하(Competitive degradation)를 성공적으로 완화했다고 보고합니다.

- **Technical Details**: 이 모델은 비선형 리워드 조정 기법인 SARS와 최적화 압력을 모델의 시간적 동적에 맞춘 TDW를 사용합니다. SARS는 모순된 리워드 신호를 명시적으로 페널티 부여하고 시너지를 높이는 신호에 추가 가중치를 부여함으로써 더욱 뚜렷한 그래디언트를 제공합니다. TDW는 초기 단계에서는 프롬프트 따르기를 우선시하고, 후반 단계에서는 정체성 보존을 중시하여 최적화를 수행합니다.

- **Performance Highlights**: 광범위한 실험을 통해 Customized-GRPO가 주제의 정체성을 충실하게 유지하고 복잡한 텍스트 프롬프트에 정확하게 맞춘 이미지를 생성하는 데 있어 뛰어난 성능을 보였음을 보여줍니다. 특히 다양한 실험에서별도의 성과 향상이 관찰되어, 기존 방식에 비해 상당한 개선을 이루었습니다. 이는 Customized-GRPO의 접근 방식이 실제 개인화된 이미지 생성의 실용성에 기여하는 것을 시사합니다.



### DualHash: A Stochastic Primal-Dual Algorithm with Theoretical Guarantee for Deep Hashing (https://arxiv.org/abs/2510.18218)
- **What's New**: 이번 연구에서는 Deep Hashing을 위한 새로운 알고리즘인 DualHash를 소개합니다. 이 알고리즘은 Fenchel 이중성을 활용하여 비볼록(Nonconvex) W-type 정규화 최적화를 변환하고, 근사(Proximal) 연산자에 대한 닫힌 형태의 해를 제공합니다. 이를 통해 기존의 Deep Hashing 기법의 한계를 극복하고, 보다 안정적이고 효율적인 성능을 목표로 합니다.

- **Technical Details**: DualHash는 스토캐스틱(Probabilistic) 원래-쌍둥이(Primal-Dual) 깊은 해시 알고리즘으로, W-type 정규화와 결합된 비선형 구성 문제를 다룹니다. 알고리즘의 분석에는 두 가지 주요 인스턴스가 포함되며, 하나는 모멘텀 기반의 버전(DualHash-StoM)이고, 다른 하나는 분산 저감(Variance Reduction)을 이용한 버전(DualHash-StoRM)입니다. 이 알고리즘들은 각각 𝒪(ε^{-4})와 𝒪(ε^{-3}) 복잡도 경계를 가지며, 안정적인 수렴성을 보장합니다.

- **Performance Highlights**: 세 가지 이미지 검색 데이터베이스에서 수행된 실험을 통해 DualHash의 우수한 성능을 입증하였습니다. 특히 제안된 방법은 각기 다른 비트 길이에서 기존 방법들에 비해 일관되게 낮은 양자화 오류를 기록하였으며, 이를 통해 원래-쌍둥이 구조의 장점을 확인할 수 있었습니다. 또한, DualHash는 기존의 Deep Hashing 방법들과 비교하여 신뢰할 수 있는 성능 개선을 보여줍니다.



### FST.ai 2.0: An Explainable AI Ecosystem for Fair, Fast, and Inclusive Decision-Making in Olympic and Paralympic Taekwondo (https://arxiv.org/abs/2510.18193)
Comments:
          23 pages, 12 figures

- **What's New**: 이 논문은 금메달을 차지하는 올림픽과 패럴림픽 태권도 경기에서 심판, 코치 및 선수의 실시간 지원을 제공하는 설명 가능한 AI 생태계인 FST.ai 2.0을 소개합니다. 이 시스템은 그래프 합성곱 네트워크(Graph Convolutional Networks)를 기반으로 한 자세 인식, 신뢰도 모델링 및 시각적 의사결정 지원을 위한 설명 가능성 오버레이를 통합합니다. 또한 이 시스템은 태권도 생태계에서 공정성 모니터링과 정책 수준의 분석을 포함하여 심판 교육과 공정성을 강화하는 모듈을 추가했습니다.

- **Technical Details**: FST.ai 2.0는 실시간으로 설명 가능하고 신뢰할 수 있는 결정을 지원하는 유연한 모듈식 아키텍처를 구현하였습니다. 이 시스템은 AI 지원 심판 도구, 기술 추적 및 피드백, 그리고 AI기반 교육 플랫폼을 통해 심판, 선수, 코치 간의 협력을 증진시킵니다. 신뢰성 있는 의사결정을 위해 불확실성 모델링을 통합하고, 인터랙티브 대시보드를 통해 시각적 피드백을 제공합니다.

- **Performance Highlights**: 경쟁 데이터에서의 실험적 검증 결과, FST.ai 2.0은 의사결정 리뷰 시간을 85% 단축시키고 AI 지원 결정에 대한 심판 신뢰도를 93%로 높였습니다. 이로 인해 데이터 기반의 공식 채점 및 선수 평가를 위한 투명하고 확장 가능한 파이프라인이 구축됩니다. 이 시스템은 AI가 스포츠에서 인간의 전문성을 저해하지 않으면서 공정하고 포괄적인 결정 생태계로 이동하는 데 기여하고자 합니다.



### A Generalizable Light Transport 3D Embedding for Global Illumination (https://arxiv.org/abs/2510.18189)
- **What's New**: 이번 논문에서는 3D 장면 구성으로부터 직접 글로벌 조명을 예측하는 일반화 가능한 3D 빛 전송 임베딩을 제안했습니다. 이는 기존의 래스터화(rasterized)나 경로 추적(path-traced) 큐를 사용하지 않고도 작동합니다. 제안된 방법은 지오메트릭(geometric) 및 재료(material) 특징을 가진 포인트 클라우드(point cloud)로 각 장면을 표현하며, 스케일러블(transformer) 모델을 통해 포인트 간 상호작용을 인코딩합니다.

- **Technical Details**: 우리의 접근 방식은 입력 지오메트리, 재료 특성 및 영역 조명 소스로부터 시작하여 장면을 포인트 클라우드로 샘플링합니다. 그 다음, 포인트 기반의 스케일러블 변환기를 사용하여 장면을 빛 전송(bidirectional light transport) 임베딩으로 인코딩합니다. 쿼리 포인트가 주어지면, 디코더는 가장 가까운 잠재 코드를 수집하고 이를 적응적으로 집계하여 하전(joint) 혹은 도착 복사선(incident radiance fields)을 추정합니다.

- **Performance Highlights**: 우리는 다양한 실내 장면에서의 확산(global illumination) 예측에 대한 결과를 보여주며, 이 방법이 제한된 파인 튜닝으로도 새로운 렌더링 작업에 신속하게 적응할 수 있음을 입증합니다. 또한 인지 장애물 없는 경로 안내를 위해 중요도 분포(initial importance distribution) 부트스트랩을 활용할 수 있으며, 이러한 접근 방식은 경험적인 수렴성을 가속화하는 가능성을 보여줍니다.



### Demystifying Transition Matching: When and Why It Can Beat Flow Matching (https://arxiv.org/abs/2510.17991)
- **What's New**: 이 연구는 최신 생성 모델에서의 Transition Matching (TM)과 Flow Matching (FM)의 성능 차이를 설명하며, TM이 어떻게 FM보다 더 우수한지를 논의합니다. 특히, unimodal Gaussian distribution에서 TM이 FM보다 낮은 KL divergence를 성취함을 증명하였고, 이는 TM이 stochastic sampling을 통해 target covariance를 보존하기 때문이라는 점에서 흥미롭습니다. 또한, Gaussian mixtures로 분석 범위를 확대하면서, 서로 잘 분리된 모드의 경우 TM이 FM보다 유리하게 작동함을 밝혔습니다.

- **Technical Details**: 연구에서는 TM의 성능이 FM에 비해 우수함을 수학적으로 정립하였고, 이는 TM이 고정된 compute budget에서 더 빠른 수렴 속도를 달성함을 통해 입증되었습니다. TM은 각 스텝에서 업데이트를 수행하는 stochastic difference latent를 사용하여 간단한 구조를 가지며, FM에 비해 계산 부담을 줄입니다. Gaussian mixtures의 경우, 조합된 구성 요소가 분리되어 있을 때 TM이 FM보다 우수하게 나타나며, 이론과 실험적 결과가 잘 일치한다는 점도 강조합니다.

- **Performance Highlights**: TM은 class-conditioned 이미지 생성 및 frame-conditioned 비디오 생성과 같은 대규모 생성 모델링 작업에서 이론적 통찰을 검증하였습니다. 실험 결과, TM은 낮은 지연 시간에서도 높은 품질의 결과를 제공하며, 비디오 생성에 대한 최초의 TM 적용 사례에서도 여러 적합성 메트릭에서 개선된 성능을 보였습니다. 이로 인해 TM은 현실적인 계산 자원 하에서도 FM보다 지속적으로 우수한 성능을 발휘하여 생성 모델링 분야에서 실용적이고 확장 가능한 대안으로 자리매김하고 있습니다.



### NeuCo-Bench: A Novel Benchmark Framework for Neural Embeddings in Earth Observation (https://arxiv.org/abs/2510.17914)
- **What's New**: NeuCo-Bench는 Earth Observation (EO) 컨텍스트에서 손실 신경 압축(neural compression) 및 표현 학습(representation learning)의 평가를 위한 혁신적인 벤치마크 프레임워크입니다. 이 프레임워크는 다양한 다운스트림 작업에 적용할 수 있는 고정 크기의 임베딩을 기반으로 하여, 재사용 가능한 임베딩을 중심으로 평가 파이프라인을 구축합니다. NeuCo-Bench는 새로운 Hidden-task 리더보드와 정확도 및 안정성을 균형 있게 측정하는 점수 시스템을 포함합니다.

- **Technical Details**: NeuCo-Bench는 주로 세 가지 핵심 구성 요소로 이루어져 있습니다. 첫째, 재사용 가능한 임베딩으로 구성된 평가 파이프라인이며, 둘째, 사전 학습 편향을 완화하기 위한 Hidden-task 리더보드를 갖춘 새로운 도전 모드입니다. 셋째, 정확도와 안정성의 균형을 맞추는 점수 시스템을 통해 실험 재현성을 지원하고, SSL4EO-S12 다운스트림이라는 다중 스펙트럼 및 다중 시계열 EO 데이터셋을 공개합니다.

- **Performance Highlights**: 2025 CVPR EARTHVISION 워크숍에서 진행된 공개 도전에서 초기 결과를 제시하고, 최신의 기초 모델(foundation models)과 함께 실험을 수행하였습니다. NeuCo-Bench는 EO 및 그 이상의 분야에서 신경 임베딩에 대한 커뮤니티 기반의 표준화된 평가의 첫 단계를 제공합니다. 이 연구는 다양한 다운스트림 작업에서 압축된 표현이 의미적 콘텐츠를 얼마나 잘 유지하는지를 평가하는데 중요한 기여를 합니다.



### Conformal Lesion Segmentation for 3D Medical Images (https://arxiv.org/abs/2510.17897)
- **What's New**: 이번 연구에서는 임상 진단의 정확성을 높이기 위해 리스크 제어 프레임워크인 Conformal Lesion Segmentation (CLS)를 제안합니다. CLS는 데이터를 기반으로 한 threshold를 조정하여 테스트 시 false negative rate (FNR)가 요구되는 허용 수준 이하로 유지되도록 보장합니다. 이 방법은 기존의 고정된 threshold(예: 0.5) 방식의 한계를 극복하며, 3D 병변 분할(3D-LS)에서도 신뢰성과 정확성을 높이는 혁신적인 접근법입니다.

- **Technical Details**: CLS는 calibration set을 활용하여 각 샘플에 대한 threshold를 분석하고, FNR 허용 기준을 만족하는 중요한 threshold를 정의합니다. 사용자 지정 리스크 수준에 따라 calibration set에서 모든 중대한 threshold의 양자화를 계산하여 검증 시 confidence threshold를 결정합니다. 이를 통해 CLS는 샘플별 critical threshold의 분포를 통해 테스트 데이터에 대해 통계적으로 일반화된 threshold를 제공합니다.

- **Performance Highlights**: CLS는 5개의 backbone 모델을 사용하여 6개의 3D-LS 데이터셋에서 검증되었으며, 고정 threshold인 0.5와 비교했을 때 더 낮은 FNR을 기록했습니다. 또한, CLS는 리스크 수준에 따라 예측된 영역 크기의 변화를 분석하여 uncertainty-aware segmentation의 벤치마킹 도구로 기능할 수 있습니다. 이 연구는 리스크 인식 분할을 임상 환경에서 실제적으로 구현하는 데 있어 실용적이고 해석 가능한 통찰을 제공합니다.



### Metrics and evaluations for computational and sustainable AI efficiency (https://arxiv.org/abs/2510.17885)
Comments:
          11 pages, 2 tables

- **What's New**: 이 논문에서는 AI 모델 추론에 대한 통합되고 재현 가능한 방법론을 제안합니다. 기존의 방법들이 성능, 효율성 및 환경 영향을 평가하는 데 한계가 있었던 반면, 이 프레임워크는 지연 시간(latency), 처리량(throughput), 에너지 소비 및 탄소 배출량과 같은 메트릭을 통합하여 실질적인 평가를 가능하게 합니다.

- **Technical Details**: AI 시스템의 지연 시간은 입력 데이터 수신 시점부터 출력 결과를 생성할 때까지의 시간 간격을 정의합니다. 이 논문에서 제시된 메트릭들은 정확한 성과 측정을 위해 전력 및 에너지를 정량화하고, 지연 시간의 다양한 구성 요소를 평가하여 효율적인 AI 서비스 개발을 지원합니다. 본 프레임워크는 다양한 하드웨어 플랫폼에서 다중 정밀도 모델을 평가하며, 소프트웨어 스택을 통해 일관되게 비교할 수 있는 기반을 제공합니다.

- **Performance Highlights**: 논문에서 제안한 방법론은 AI 시스템의 효율성과 탄소 발자국 간의 Trade-off를 명확히 하여 의사 결정을 지원합니다. 또한, 본 연구의 결과는 독립적으로 검증할 수 있도록 공개된 코드와 스크립트를 통해 제공되며, 연구자 및 실무자들이 지속 가능한 AI 배포를 위한 증거 기반 결정을 내리는 데 도움을 줍니다.



### DMTrack: Deformable State-Space Modeling for UAV Multi-Object Tracking with Kalman Fusion and Uncertainty-Aware Association (https://arxiv.org/abs/2510.17860)
- **What's New**: 본 연구에서는 UAV 기반의 다중 객체 추적 (Multi-Object Tracking, MOT)에서의 도전 과제를 해결하기 위해 DMTrack이라는 변형 가능한 모션 추적 프레임워크를 제안합니다. 기존의 전통적인 모션 모델에서는 비선형 동역학 및 복잡한 객체 움직임을 효과적으로 포착하지 못했던 한계를 극복하고자 하였습니다. DMTrack은 변형 가능한 상태 공간 예측기(DeformMamba), 경량형 게이팅 모듈(MotionGate), 불확실성 인식 일치 전략을 통합하여 신뢰할 수 있는 궤적 예측과 정체성을 유지하는 데 중점을 두고 있습니다.

- **Technical Details**: DMTrack은 세 가지 주요 구성 요소로 이루어져 있습니다. DeformMamba는 과거의 모션 상태를 동적으로 집계하여 비선형 궤적 모델링을 가능하게 하는 변형 가능한 상태 공간 예측기입니다. MotionGate은 칼만 필터와 DeformMamba의 예측을 통합하여 물리적 안정성과 학습된 적응성을 균형 있게 조절하는 경량형 게이팅 모듈입니다. 마지막으로, 불확실성 인식 매칭 전략은 잡음을 줄이고 예측의 신뢰도를 높이며 정체성을 보호합니다.

- **Performance Highlights**: DMTrack은 VisDrone-MOT 및 UAVDT 벤치마크에서 광범위한 실험을 통해 상태-of-the-art 성능을 입증하였습니다. 특히 빠른 속도와 비선형 이동이 있는 환경에서 뛰어난 정체성 일관성과 추적 정확도를 보여주었습니다. 이를 통해 DMTrack은 실시간 성능을 유지하며 외관 기반 모델 없이도 효율성을 잃지 않아 실제 UAV 기반 추적 응용 분야에 적합하다는 점을 강조합니다.



### Cross-Domain Multi-Person Human Activity Recognition via Near-Field Wi-Fi Sensing (https://arxiv.org/abs/2510.17816)
- **What's New**: 이 논문에서는 Wi-Fi 기반 다중 인식 (multi-person HAR)을 위한 새로운 프레임워크 WiAnchor를 제안합니다. 이 프레임워크는 Wi-Fi신호의 불규칙한 시간 정보를 처리하여 대체 활동 범주 없이도 효율적인 교차 도메인 적응을 제공합니다. 또한, Wi-Fi의 근거리 우세 효과를 활용하여 개인 장치와의 전용 감지 링크를 구축함으로써 다중 주체 인식을 가능하게 합니다.

- **Technical Details**: WiAnchor 프레임워크는 세 가지 단계로 구성됩니다. 첫째, 전처리 단계에서 클래스 간 특성 간격을 확대하여 활동의 구분 가능성을 향상시킵니다. 둘째, 미세 조정(FT) 단계에서는 앵커 매칭 메커니즘을 혁신적으로 도입하여 불완전한 활동 범주로부터 주체 특이적 간섭을 필터링합니다. 마지막으로, 인식 단계에서는 앵커와의 특성 수준 유사성을 기반으로 입력 샘플의 인식을 개선합니다.

- **Performance Highlights**: 제안된 WiAnchor 프레임워크는 약 65,000개의 샘플을 포함하는 종합적인 데이터셋을 기반으로 평가되었습니다. 이 프레임워크는 FT 샘플이 없는 범주에 대해 56.8%의 정확도 향상을 보였으며, 전체적인 정확도는 90%를 초과하였습니다. 이를 통해 Wi-Fi 기반 다중 인식 시스템의 실제적 배포 가능성을 입증하였습니다.



### Robobench: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models as Embodied Brain (https://arxiv.org/abs/2510.17801)
- **What's New**: 본 논문은 로봇 조작(manipulation)에서 고차원적 인지(cognition) 능력을 체계적으로 평가하기 위한 새로운 기준인 RoboBench를 소개합니다. RoboBench는 다중 모달 대형 언어 모델(MLLMs)을 인지 핵(cognitive core)으로 간주하여, 저자들에 의해 정의된 5개의 평가 차원—지시 이해(instruction comprehension), 인식 추론(perception reasoning), 일반화된 계획(generalized planning), 여건 예측(affordance prediction), 실패 분석(failure analysis)—을 통해 로봇 조작의 복잡한 요구 사항을 평가합니다. 현재의 기준들이 가진 한계를 극복하고자 하는 노력의 일환으로, RoboBench는 14개의 능력과 25개의 작업, 6092개의 QA 쌍을 포함하여 체계적인 평가를 제시합니다.

- **Technical Details**: 이 연구에서는 MLLM이 수행하는 다양한 작업을 위한 다섯 가지 차원을 정의하고, 이를 통해 연계된 능력을 평가합니다. RoboBench의 개발에는 독립적인 시험을 통해 검증된 현실감 있고 다양한 태스크 설정이 포함되어 있으며, 실세계 로봇 데이터에 기초하여 현실성과 복잡성을 고려합니다. 평가 프레임워크인 MLLM-as-world-simulator는 예측된 계획의 실행 가능성을 확인하기 위한 것으로, 비주얼 및 물리적 제한 속에서 단계별로 계획을 구현합니다.

- **Performance Highlights**: 14개의 최신 MLLM에 대한 평가 결과는 여러 주요 제한점을 드러냅니다. 모델들은 암시적인 지시 이해에서 평균 30% 이상의 성과 저하를 보였으며, 로봇 관점 인식과 공간-시간 지각에서 심각한 약점을 드러냈습니다. 복잡한 계획 영역은 여전히 주요 병목 현상이 나타났으며, 고급 조정 및 드문 객체에 대한 추론에서의 어려움이 부각되었습니다. 전반적으로 MLLM들은 표면적인 인지 능력 및 세계 모델링을 보여주며, 이를 통해 향후 발전 방향에 대한 통찰을 제공합니다.



### SceneCOT: Eliciting Grounded Chain-of-Thought Reasoning in 3D Scenes (https://arxiv.org/abs/2510.16714)
Comments:
          Project page: this https URL

- **What's New**: 이 논문은 3D Large Language Models (LLMs)의 한계를 극복하기 위한 새로운 프레임워크인 SceneCOT을 제시합니다. SceneCOT은 Grounded Chain-of-Thought (CoT) 추론 방식을 도입하여 복잡한 추론 과제를 더 단순하고 관리 가능한 문제로 분해합니다. 이를 위해 185,000개의 고품질 사례로 구성된 최초의 대규모 grounded CoT 추론 데이터 세트인 SceneCOT-185K를 개발하였습니다.

- **Technical Details**: SceneCOT은 3D 장면에서 복잡한 추론 작업을 네 단계로 분해합니다: (1) 작업 인식 및 분석, (2) 작업 관련 영역 로컬라이제이션, (3) 다중 모달 전문가 모듈을 이용한 개체 및 속성 그라운딩, (4) 중간 결과를 통합하여 일관된 최종 답변을 생성하는 grounded 추론이 포함됩니다. 이러한 계층적 작업 흐름은 각 답변이 명시적 그라운딩 단계를 통해 지원되도록 합니다.

- **Performance Highlights**: MSQA 및 Beacon3D 벤치마크에서의 extensive 실험 결과, SceneCOT은 높은 grounding-QA 일관성을 달성하며 강력한 성능을 보여 주었습니다. 이 연구는 인간과 유사한 단계적 추론을 가능하게 하여, 더 넓은 3D 장면 이해 시나리오로의 확장 가능성을 제시합니다.



New uploads on arXiv(cs.AI)

### Decoding Funded Research: Comparative Analysis of Topic Models and Uncovering the Effect of Gender and Geographic Location (https://arxiv.org/abs/2510.18803)
Comments:
          35 pages

- **What's New**: 이 연구는 캐나다 자연과학 및 공학 연구 위원회(NSERC)가 지원한 연구 제안서 18년(2005-2022)간을 분석하여 연구 트렌드의 변화와 인구 통계적 및 지리적 힘을 이해하려는 시도를 했습니다. 특히, 형평성(equity), 다양성(diversity), 포용성(inclusion)에 대한 약속에 비추어 볼 때, 이러한 연구는 특히 중요합니다. 또한, BERTopic의 covariate 분석 기능 부족을 보완하기 위해 새로운 알고리즘 COFFEE를 도입했습니다.

- **Technical Details**: 이 논문에서는 세 가지 주제 모델링(topic modelling) 접근 방식인 Latent Dirichlet Allocation (LDA), Structural Topic Modelling (STM) 및 BERTopic의 종합 비교 평가를 수행했습니다. COFFEE 알고리즘은 BERTopic에 대한 강력한 covariate 효과 추정을 가능하게 하므로, 연구에 대한 더욱 심도 있는 분석을 제공할 수 있습니다. BERTopic은 특히 인공지능의 급격한 확산과 같은 세밀하고 일관된 새로운 주제를 지속적으로 식별함으로써 효과적으로 핵심 과학 분야를 delineate했습니다.

- **Performance Highlights**: 연구 결과는 모든 모델이 핵심 과학 분야를 효과적으로 구분하는 반면, BERTopic이 더 미세하고 일관된 테마를 지속적으로 찾아내며 뛰어난 성능을 보인다는 것을 확인했습니다. COFFEE가 지원하는 covariate 분석은 각 주에 대한 연구 전문성과 다양한 과학 분야에서 성별에 따른 일관된 주제 패턴을 발견하는 데 기여했습니다. 이러한 통찰력은 기금 제공 기관들이 보다 형평성 있고 영향력 있는 기금 전략을 수립하는 데 강력한 경험적 기초를 제공합니다.



### Seg the HAB: Language-Guided Geospatial Algae Bloom Reasoning and Segmentation (https://arxiv.org/abs/2510.18751)
- **What's New**: 이 논문에서는 ALGae Observation and Segmentation (ALGOS)라는 새로운 시스템을 소개합니다. 이 시스템은 임계 매개변수에 대한 내러티브 추론(segmentation-and-reasoning)과 해양 생태계의 유해 조류 발생 모니터링을 위해 원거리 감지(image understanding) 분야에 혁신을 가져옵니다. 인간의 평가를 통해 품질 높은 segmentation mask를 구성하고, NASA의 Cyanobacteria Aggregated Manual Labels (CAML) 를 활용하여 심각도 예측을 위한 비전-언어 모델을 미세 조정합니다.

- **Technical Details**: ALGOS는 원거리 감지 이미지에서 유해 조류 발생의 심각도를 평가하기 위해 비전-언어 모델을 통합한 통합적 접근을 사용합니다. GeoSAM이라는 도구를 통해 사람이 평가하는 과정을 포함하여 고급 픽셀 레벨 segmentation mask를 생성하고, 이를 기반으로 실시간 모니터링을 가능하게 합니다. 또한 심각도 레벨을 평가하기 위해 다섯 단계의 정량적 기준을 설정하고, 자연어 쿼리를 통해 해당 심각도를 추론할 수 있도록 하였습니다.

- **Performance Highlights**: 실험 결과, ALGOS는 기존의 segmentation 모델보다 공간 정확도(spatial accuracy)에서 유의미한 개선을 보였으며, 심각도 예측에서도 기존의 비전-언어 모델보다 우수한 성능을 달성하였습니다. 이 연구는 유해 조류 모니터링의 자동화와 생태적 평가 및 공공 건강 결정을 위한 정밀한 공간적 추론의 필요성을 강조합니다. ALGOS는 GIS 및 원거리 감지 기술의 발전을 활용하여 지속 가능한 수자원 관리에 기여할 수 있는 가능성을 보여주는 성공적인 사례입니다.



### Sherlock Your Queries: Learning to Ask the Right Questions for Dialogue-Based Retrieva (https://arxiv.org/abs/2510.18659)
- **What's New**: 본 논문에서는 SherlockLLM이라는 대화 기반 검색 프레임워크를 제안합니다. 이 프레임워크는 강화 학습(Reinforcement Learning, RL)을 통해 최적의 질문 전략을 학습하며, 대규모 주석 데이터 없이도 사용자 의도를 명확히 할 수 있습니다. SherlockLLM은 이진 질문의 순서를 생성하여 검색 공간을 효율적으로 좁히는 방법으로 작동합니다.

- **Technical Details**: SherlockLLM의 아키텍처는 검색기(Retriever)와 질문기(Questioner)라는 두 개의 주요 모듈로 구성됩니다. 질문기는 대화의 이전 이력과 검색 결과를 기반으로 다음 질문을 생성하며, 이는 사용자의 반응을 포함하여 대화의 맥락을 반영합니다. 기본적으로 RL을 통해 에이전트가 효율적인 검색을 위한 대화 정책을 자율적으로 학습하는 방식으로 설계되었습니다.

- **Performance Highlights**: 실험 결과, SherlockLLM은 구조화된 작업에서 강력한 기준선과 동등한 성능을 보이며, 이진 검색에 의해 정의된 이론적 최적값에 근접합니다. 비구조적 이미지 검색 작업에서는 기존의 강력한 LLM보다 96배 더 많은 매개변수를 가졌음에도 불구하고, 월등한 성능을 발휘하여 효율적인 정보 검색의 가능성을 입증했습니다.



### Query Decomposition for RAG: Balancing Exploration-Exploitation (https://arxiv.org/abs/2510.18633)
- **What's New**: 이번 논문에서는 Retrieval-augmented generation (RAG) 시스템을 개선하기 위해 쿼리 분해와 문서 검색을 효율적으로 수행할 수 있는 새로운 방법론을 제시합니다. 정보 검색 과정에서 유용한 문서를 동적으로 선택하는 bandit learning 기법을 실험하였으며, 이를 통해 문서 수준의 정밀도가 35%, {}-nDCG가 15% 향상됨을 보여주었습니다.

- **Technical Details**: 논문에서는 탐색-이용의 균형을 맞추는 multi-armed bandit 문제를 통해 복잡한 사용자 쿼리를 해결하는 방법을 제안합니다. 각 서브 쿼리를 한 번에 하나씩 검색하며, 검색 결과에 대한 신뢰도를 기반으로 다음 검색을 결정하는 과정에서 강화 학습( Reinforcement Learning ) 정책을 사용하였습니다. 이 방법은 서브 쿼리의 유용성을 추정하여 불필요한 문서 검색을 줄이는 데 기여합니다.

- **Performance Highlights**: 실험 결과, 문서 선택에 있어 multi-armed bandit 방식을 적용하였을 때 문서 정밀도가 35% 향상되었고, 알고리즘의 활용 과정에서 긴 형식 생성 작업에 있어서 더 나은 성능을 발휘함을 확인하였습니다. 또한 계층적 서브 쿼리 분해를 사용하여 정밀도가 30% 증가함을 보여주었으며, 이러한 접근 방식이 downstream task에 미치는 긍정적인 영향을 입증했습니다.



### Comparative Expressivity for Structured Argumentation Frameworks with Uncertain Rules and Premises (https://arxiv.org/abs/2510.18631)
- **What's New**: 이번 논문은 형식적 논증(Argumentation)에서의 질적 불확실성(qualitative uncertainty)을 모델링하는 방법에 대해 다루고 있습니다. 기존의 연구들이 추상 모델에 초점을 맞춘 반면, 이 연구는 이러한 모델을 구체적으로 실현할 수 있는 가능성 있는 인스턴스에 대한 질문에 접근합니다. 논증의 구성 요소를 기반으로 불확실성을 설정하고, 그에 따른 표현력(expressivity)을 정의하는 두 가지 주요 기술적 기여를 하고 있습니다.

- **Technical Details**: 저자들은 추상 논증(abstract argumentation)과 구조적 논증(structured argumentation) 간의 불확실성(comparative uncertainty)에 대한 체계적 접근법을 설정합니다. 그들은 불완전한 논증 프레임워크와 관련된 불확실성을 나타내는 두 단계의 방법론을 제시합니다. 첫째, 의존성이 있는 불완전 논증 프레임워크인 dep-arg-IAFs를 활용하여 불확실성을 정의하고 둘째, 구조적 형식을 추상 형식으로 ‘승격(lifting)’하는 과정을 통해 비교합니다.

- **Performance Highlights**: 결과적으로, 구조적 논증 시스템은 의존성이 없는 일반적인 불완전 추상 논증 프레임워크보다 더 높은 표현력을 가지지만 의존성이 있는 프레임워크에 비해서는 낮은 표현력을 보이는 것으로 나타났습니다. 특히 규칙이 불완전한 구조적 논증 시스템은 전제(premise)가 불완전한 시스템보다 더 뛰어난 표현력을 발휘함을 실증합니다. 이러한 발견은 새로운 연구 방향을 제시하며, 여러 논증 프레임워크에 대한 이해를 증진하는 데 기여할 것입니다.



### Leveraging Association Rules for Better Predictions and Better Explanations (https://arxiv.org/abs/2510.18628)
Comments:
          24 pages

- **What's New**: 이번 논문에서는 데이터를 기반으로 한 데이터 마이닝(data mining)과 지식 기반의 접근을 결합하여 새로운 분류(classification) 방법을 제시합니다. 이 접근 방식에서는 데이터로부터 추출한 연관 규칙(association rules)을 사용해 의사결정 트리(decision trees) 및 랜덤 포레스트(random forests) 모델의 예측 성능을 향상시킵니다. 또한 이러한 규칙을 활용하여 더욱 일반적인 설명(abductive explanations)을 생성하여 모델의 예측을 이해하는 데 도움을 줍니다.

- **Technical Details**: 저자들은 본 연구에서 랜덤 포레스트 모델을 통해 분류 작업을 수행하기 위해 보편적인 부울 조건(Boolean conditions)으로 데이터를 변환하여 이진화된 데이터셋(binarized dataset)을 생성하는 방법론을 설명합니다. 이 후, 데이터 마이닝 알고리즘을 통해 100% 확신과 비어 있지 않은 지지를 가진 연관 규칙을 유도하며, 이러한 규칙을 통해 모델을 수정하는 과정을 거칩니다. 이를 통해 모델은 연관 규칙에 따른 예측을 우선시하여 규칙적합성을 고려하게 됩니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 1313개 데이터셋 중 1212개에서 예측 성능을 약간 향상시킴을 보여주었습니다. 이 성능 향상은 일반적으로 작지만 10%를 초과할 수 있으며, 설명 크기(abductive explanations) 또한 최대 96%까지 줄일 수 있는 효과를 가지고 있습니다. 따라서 제안된 접근법은 설명의 크기를 줄이고 예측 성능을 개선하는 데 있어 잠재적인 이점을 제공합니다.



### VAR: Visual Attention Reasoning via Structured Search and Backtracking (https://arxiv.org/abs/2510.18619)
- **What's New**: 이번 논문은 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 고유한 한계를 해결하기 위해 비주얼 어텐션 추론(Visual Attention Reasoning, VAR)이라는 새로운 프레임워크를 제안합니다. VAR는 단순한 선형 추론 방식이 아닌, 구조화된 검색 기법을 통해 추론 과정을 두 단계로 나누어 진행하며, 오류 수정이 가능한 백트래킹 메커니즘을 통합하고 있습니다. 이를 통해 MLLMs의 비주얼 이해의 신뢰성을 높이고, 복잡한 작업 수행 시의 문제를 개선하려 합니다.

- **Technical Details**: VAR 프레임워크는 증거 기반 추론과 검색 기반 사고의 연결(chain-of-thought, CoT)으로 구성되며, 검색 과정은 다면적인 보상 함수에 의해 안내됩니다. 이 보상 함수는 의미적(self-verification) 및 기하학적(self-verification) 자기 검증 구성 요소를 포함하여, 비주얼 입력에 충실하지 않은 결과에 대해 패널티를 부여합니다. VAR의 이론적 분석을 통해 높은 확률로 올바른 해답을 찾을 수 있는 검색 전략의 효율성이 입증되었습니다.

- **Performance Highlights**: 실험적으로 VAR-7B 모델은 기존 오픈 소스 모델들을 크게 초월하여 환각(hallucination) 및 안전 벤치마크에서 새로운 최첨단 성능을 선보였습니다. 이 모델은 현재 대표적인 상용 시스템들과의 경쟁에서도 좋은 성과를 보이고 있으며, MLLMs의 비주얼 추론 능력을 획기적으로 개선하는 데 기여하고 있습니다.



### QuantEvolve: Automating Quantitative Strategy Discovery through Multi-Agent Evolutionary Framework (https://arxiv.org/abs/2510.18569)
Comments:
          25 pages, 13 figures. Accepted for oral presentation at the 2nd Workshop on LLMs and Generative AI for Finance (AI4F), part of ACM ICAIF 2025, Singapore. Non-archival workshop

- **What's New**: 논문에서는 QuantEvolve라는 진화적 프레임워크를 소개합니다. 이 프레임워크는 품질-다양성 최적화와 가설에 기반한 전략 생성을 결합하여 개인화된 투자 솔루션을 제공합니다. 기존 방법들이 복잡한 전략 공간을 탐색하지 못하는 단점을 해결하고, 변화하는 시장 조건에 맞춰 다양하고 효과적인 전략 세트를 유지할 수 있도록 설계되었습니다.

- **Technical Details**: QuantEvolve는 투자자 선호에 맞춘 특성 맵과 다중 에이전트 시스템을 활용하여 전략 공간을 체계적으로 탐색합니다. 이 시스템은 투자 유형, 리스크 프로필, 거래 빈도 및 수익 특성 등 다양한 요소를 고려하여 고급 전략을 생성합니다. 진화적 사이클 동안 구조적 추론을 통해 효율적인 탐색과 정제가 이루어집니다.

- **Performance Highlights**: 실험 결과, QuantEvolve는 기존의 벤치마크보다 우수한 성능을 보이며 그 효과가 검증되었습니다. 이 시스템을 통해 개인 맞춤형 투자 요구에 적합한 다채로운 전략을 생성할 수 있는 가능성이 열렸습니다. 또한, 향후 연구를 지원하기 위해 진화된 전략의 데이터셋도 공개하였습니다.



### Extracting alignment data in open models (https://arxiv.org/abs/2510.18554)
- **What's New**: 이번 연구에서는 사후 훈련된 모델(post-trained model)에서 상당한 양의 alignment training data를 추출할 수 있음을 보여줍니다. 이러한 데이터는 모델의 특정 능력, 특히 긴 문맥 이해(long-context reasoning)와 안전성(safety), 지시 따르기(instruction following) 및 수학적 능력을 향상시키는 데 유용합니다. 기존의 메모리화(memorization) 연구가 문자열 일치를 통해 훈련 데이터 추출의 성공도를 측정하는 데 집중한 반면, 우리는 임베딩 모델(embedding model)이 특정 목표에 더 적합하다는 주장을 합니다.

- **Technical Details**: 사후 훈련(data extraction) 과정에서 모델이 사용한 데이터가 회귀하여 자주 반복된다는 흥미로운 발견이 있었습니다. 우리는 이 데이터를 사용하여 기본 모델(base model)을 훈련시키고, 원래 성능의 의미 있는 양을 회복할 수 있음을 보였습니다. 특히 높은 품질의 임베딩 모델을 사용하면 문자열 간의 의미적 유사성을 감지할 수 있어, 기존의 간단한 문자열 일치 메트릭이 포착하기 어려운 패턴을 발견할 수 있습니다.

- **Performance Highlights**: 우리는 임베딩 점수(embedding scores)를 통해 메모리화의 측정이 정확하지 않을 수 있음을 보여주었습니다. 결과적으로, 기존 방법들은 메모리화 비율을 10배가량 저축하는 경향이 있습니다. 이 연구는 기존의 메모리화 연구를 넘어서, 모델의 원래 훈련 데이터를 간접적으로 포함할 수 있는 탈류(distillation) 방식에 대한 흥미로운 논의를 열어줍니다.



### SOCIA-Nabla: Textual Gradient Meets Multi-Agent Orchestration for Automated Simulator Generation (https://arxiv.org/abs/2510.18551)
Comments:
          11 pages, 1 figure, 2 tables. The paper is under review

- **What's New**: 이번 논문에서는 SOCIA-Nabla라는 새로운 에이전트 기반 프레임워크를 소개합니다. 이 프레임워크는 코드 내의 인스턴스 최적화를 통해 시뮬레이터 빌딩을 처리하며, 특화된 LLM 기반 에이전트를 그래프 노드로 통합합니다. SOCIA-Nabla는 손실 구동 루프를 통해 코드 합성, 실행, 평가 및 코드 수정을 자동화합니다.

- **Technical Details**: SOCIA-Nabla는 텍스트 기반의 계산 그래프 내에서 시뮬레이터 코드의 최적화 변수를 다룹니다. 이 시스템은 코드 생성, 실행, 평가, 피드백을 수행하는 다양한 에이전트를 정의하며, 중앙 집중식 워크플로우 매니저가 전체 진행을 조정합니다. 텍스트 그라디언트를 이용해 코드 수정을 지원하며, 인간의 개입은 최소화하여 코드 자체를 훈련 가능한 객체로 만듭니다.

- **Performance Highlights**: SOCIA-Nabla는 사용자 모델링, 마스크 채택, 개인 이동성 등 세 가지 CPS 작업에서 최첨단의 정확도를 달성했습니다. 이 프레임워크는 다양한 도메인과 시뮬레이션 세분화에서 강력한 성능을 발휘하여 재현 가능하고 제약을 인식하는 시뮬레이터 코드를 생성합니다. 향후 코드도 공개할 예정이며, 자동화를 통한 시뮬레이터 구축 비용 절감에 기여할 것으로 기대됩니다.



### Physics-guided Emulators Reveal Resilience and Fragility under Operational Latencies and Outages (https://arxiv.org/abs/2510.18535)
Comments:
          45 pages, 5 main figures, 10 supplementary figures, 5 supplementary tables

- **What's New**: 본 연구에서는 입력 데이터가 지연되거나 결여되거나 불일치할 때에도 안정성을 유지하는 수문학적 및 홍수 예측 모델을 개발하였습니다. 우리는 Global Flood Awareness System(GloFAS)의 hydrological core를 기반으로 한 운영 준비 완료된 에뮬레이터를 소개하며, 이는 장기 및 단기 메모리 네트워크(Long Short-Term Memory Networks, LSTM)와 물 균형 제약을 결합하도록 설계되었습니다. 이 모델은 다양한 데이터 가용성 시나리오를 포괄하여 견고성을 체계적으로 평가할 수 있는 기회를 제공합니다.

- **Technical Details**: 연구에서는 최소 관리 유역(minimally managed catchments)에서 훈련한 모델을 포함하여 미국 전역의 5,000개 이상의 유역에서 테스트하였습니다. 에뮬레이터는 비동기 데이터의 가용성을 이진 마스크를 통해 명시적으로 나타내며, 여러 정보 수준을 나타내는 다섯 개의 인코더-디코더 구성을 사용합니다. 이러한 구성은 모델이 실제 예측 흐름에서 경험하는 지연 및 데이터 결여 조건에서 학습할 수 있도록 하여, 수문학적 기능과 시간 참조를 보존합니다.

- **Performance Highlights**: 훈련된 에뮬레이터는 GloFAS의 방식을 훌륭하게 재현하며, 정보 품질이 저하될 때 부드럽게 성능이 감소하는 경향을 보입니다. 하이드로클릭과 관리 체계의 대조적인 환경 간 전이에서 발생하는 일반화의 한계를 정의할 수 있습니다. 이 프레임워크는 수문학적 머신러닝의 운영적 견고성을 측정 가능한 속성으로 정립하고, 실제로 견고한 예측 시스템의 설계를 위한 전진을 의미합니다.



### Counterfactual Reasoning for Steerable Pluralistic Value Alignment of Large Language Models (https://arxiv.org/abs/2510.18526)
Comments:
          41 pages, 7 figures

- **What's New**: 이 논문은 다양한 문화와 커뮤니티의 사용자에게 제공되는 애플리케이션에서 대형 언어 모델(LLMs)의 사용이 증가하면서 이들 모델을 다양한 인간의 다원적 가치에 맞추는 것을 목표로 합니다. 기존 방법들이 여러 값을 독립적인 특성으로 취급하는 데 따른 문제점을 지적하고, 이런 복잡한 가치 목표에 대한 새로운 접근 방식을 제안합니다. COUPLE라는 새로운 프레임워크는 구조적 인과 모델(SCM)을 도입하여 가치 간의 상호 의존성과 우선 순위를 명확히 파악합니다.

- **Technical Details**: COUPLE 프레임워크는 대형 언어 모델의 가치 정렬을 위한 세 가지 단계로 구성된 파이프라인을 제시합니다. 첫 번째 단계는 원래 응답의 기본 가치 우선 순위를 추론하는 가치 귀속(value attribution)입니다. 두 번째 단계는 목표 가치 프로필에 맞춘 새로운 응답을 생성하기 위한 가치 개입(value intervention)과 반사실적 예측(counterfactual prediction)을 포함합니다. 이 접근 방식은 사전 데이터와의 차별화된 해석 가능성을 제공합니다.

- **Performance Highlights**: COUPLE은 두 가지 데이터 세트를 통해 실제 실험을 수행한 결과, 기존의 여러 기준선을 능가하는 성능을 보였습니다. 실험 결과는 COUPLE이 다양한 가치 목표에 대한 정확하고 해석 가능한 정렬을 달성했음을 보여줍니다. 또한, 이 프레임워크는 데이터의 부족으로 인해 잘 반영되지 않는 가치 목표를 보완하는 데 도움을 주는 데이터 합성을 가능하게 합니다.



### Crucible: Quantifying the Potential of Control Algorithms through LLM Agents (https://arxiv.org/abs/2510.18491)
Comments:
          NeurIPS 2025

- **What's New**: 이번 연구에서 소개되는 Crucible 시스템은 제어 알고리즘의 Tuning Potential을 정량적으로 평가할 수 있도록 설계된 최초의 프레임워크입니다. 기존 연구에서 주로 이상적인 조건에서 알고리즘 성능에 초점을 맞추었던 점을 개선하여, Crucible은 다양한 시뮬레이션을 통해 알고리즘의 조정 가능성을 체계적으로 측정합니다. 이 시스템은 LLM(대형 언어 모델)을 기반으로 하여 다단계 전문가 시뮬레이션을 적용함으로써 전문가의 관점에서 알고리즘 조정을 모사합니다.

- **Technical Details**: Crucible은 알고리즘 조정에 있어 LLM 기반 에이전트를 활용하여 최적화 도구 및 다단계 반영을 수행하는 다차원 전문가 시뮬레이션을 구현합니다. 이는 알고리즘 성능 평가에서 파라미터 감도 분석을 넘어서는 구조적 조정이나 논리적 수정과 같은 깊은 수정작업을 포함합니다. Crucible은 다양한 작업 환경을 정량적으로 평가할 수 있는 공식화된 메트릭을 수립하여, 알고리즘 조정 가능성을 다른 환경과 비교할 수 있게 합니다.

- **Performance Highlights**: 실험 결과, Crucible은 기존의 베이지안 방법들보다 더욱 방대한 최적화 공간을 발견할 수 있음을 입증하였습니다. ABR 제어와 스케줄링 제어를 포함한 다양한 케이스 스터디를 통해, 알고리즘의 표현 능력과 가독성이 조정 가능성에 중대한 영향을 미친다는 것을 확인했습니다. Crucible의 정량적 분석을 통해 알고리즘 설계를 위한 명확한 가이드라인이 제시되어, 성능 개선으로 이어지는 새로운 통찰력을 제공합니다.



### AndroidControl-Curated: Revealing the True Potential of GUI Agents through Benchmark Purification (https://arxiv.org/abs/2510.18488)
- **What's New**: 이번 연구는 AndroidControl 벤치마크의 결함을 발견하고, 이를 개선하여 AndroidControl-Curated라는 새로운 벤치마크를 제시합니다. 기존 벤치마크는 GUUi 에이전트의 성능을 과소평가하는 문제를 안고 있었으나, 강화된 벤치마크를 통해 최신 모델이 75% 가까운 성공률에 도달 가능함을 보여주었습니다. 이러한 개선은 실제로 구동되는 GUI 에이전트의 실현 가능성을 한층 높이고 있습니다.

- **Technical Details**: 연구 방법론은 주로 두 가지 부분으로 나뉩니다. 첫째, AndroidControl-Curated라는 고품질 벤치마크를 구축하는 체계적인 파이프라인을 소개합니다. 둘째, 이 데이터를 활용하여 새로운 강화 학습 전략을 통해 SOTA 모델 Magma-R1을 교육하는 방법을 상세히 설명합니다. 이를 통해 기존 벤치마크의 한계를 뛰어넘어 보다 정확한 모델 평가를 목표로 하고 있습니다.

- **Performance Highlights**: Magma-R1 모델은 타 모델에 비해 파라미터 수가 200배 적음에도 불구하고, Qwen3-VL-235B와 유사한 성능을 발휘합니다. AndroidControl-Curated를 통해 새로운 모델이 75%에 가까운 성공률을 기록하며, 이는 기존 모델의 성능을 근본적으로 재평가할 수 있는 기반이 됩니다. 연구자들은 이 새로운 벤치마크와 모델을 공개하여 더욱 많은 연구자들이 GUI 에이전트의 발전을 이끌 수 있도록 장려하고 있습니다.



### StarBench: A Turn-Based RPG Benchmark for Agentic Multimodal Decision-Making and Information Seeking (https://arxiv.org/abs/2510.18483)
- **What's New**: StarBench라는 새로운 벤치마크가 도입되어 VLMs(vision-language models)가 사람처럼 게임을 플레이 할 수 있는지를 평가하게 된다. 이 벤치마크는 멀티모달 의사결정 및 정보 탐색 기능을 중점적으로 평가하며, 이를 위해 Honkai: Star Rail(HSR)이라는 RPG에서 유래했다. StarBench는 사용자가 스크린샷을 통해 직접적으로 행동을 수행할 수 있도록 하며, 정보 검색 선택도 측정한다.

- **Technical Details**: StarBench는 두 가지 평가 방식, 즉 직접 제어(direct control)와 도구 지원(control with tools)을 통해 VLM을 평가한다. 여기서 직접 제어는 화면의 스크린샷만을 사용하여 클릭 및 키 입력 같은 기본 행동을 수행하게 하고, 도구 지원은 텍스트 기반의 추가 정보를 제공하여 행동을 용이하게 만든다. 또한, 'ask-or-act' 실험을 통해 에이전트가 정보를 요청할지를 측정하며, 이러한 선택이 이후의 성능에 미치는 영향을 평가한다.

- **Performance Highlights**: 초기 분석 결과, 직접 제어 방식에서 현재의 VLM들은 제어의 정확성에서 큰 차이를 보이고 있다. 한편, 적절한 정보 검색이 성공률과 상관관계가 있음을 보여주어, StarBench가 에이전트의 정보 탐색 및 멀티모달 의사결정의 평가를 위한 유용한 기준이 될 것을 입증했다. 이를 통해 StarBench는 실제 게임 클라이언트에서 에이전트의 트렌드를 이끌 중요한 도구로 자리매김하게 될 것이다.



### LAFA: Agentic LLM-Driven Federated Analytics over Decentralized Data Sources (https://arxiv.org/abs/2510.18477)
- **What's New**: LAFA는 최초의 LLM 기반 연합 분석(federated analytics) 프레임워크로, 자연어(Natural Language)를 지원하고 프라이버시를 보호하는 데이터를 효율적으로 처리할 수 있게 합니다. 이 프레임워크는 복잡한 쿼리를 실행 가능하고 최적화된 FA 작업으로 변환할 수 있는 계층적 다중 에이전트 아키텍처를 도입하고 있습니다. LAFA는 또한 효율성을 높이기 위해 여러 DAG(Directed Acyclic Graph)를 재작성하고 병합하여 중복 작업을 제거하고, 계산 및 통신 오버헤드를 최소화하는 옵티마이저(agent)를 포함합니다.

- **Technical Details**: LAFA는 복잡한 쿼리를 단일 분석 하위 쿼리로 분해하는 방식으로 동작하는 코스-그레인(couarse-grained) 계획기와, 저장된 구조적 사전 정보를 기반으로 각 하위 쿼리를 FA DAG로 매핑하는 세밀한(fine-grained) 계획기를 갖추고 있습니다. 미래 쿼리 처리의 효율성을 확보하기 위해, 파인 그레인 플래너는 각 쿼리의 유효성 및 의미론적 정확성을 보장합니다. 본 시스템은 여러 에이전트가 협력하여 사용자 쿼리를 명확하게 실행 계획으로 변환하는 기능을 수행합니다.

- **Performance Highlights**: 실험 결과, LAFA는 기존의 기준(test)의 프롬프트(prompting) 전략에 비해 더 높은 실행 계획 성공률을 기록하고, 리소스 집약적인 FA 작업을 상당히 감소시킴을 보여주었습니다. 최종 결과는 쿼리 분해의 품질 및 리소스 효율성에서 LAFA의 우수한 성능을 뒷받침하고 있습니다. 이러한 특성으로 인해 LAFA는 복잡한 쿼리 해결을 위한 실용적인 기반을 마련하였습니다.



### Probabilistic Modeling of Intentions in Socially Intelligent LLM Agents (https://arxiv.org/abs/2510.18476)
- **What's New**: 이번 연구에서는 다중 턴 사회 대화에서 대형 언어 모델(LLM) 에이전트를 위한 확률적 의도 모델링 프레임워크를 제안합니다. 이 프레임워크는 파트너의 잠재적 의도에 대한 신념 분포(belief distribution)를 유지하며, 맥락적 사전(prior)에서 초기화되고 발화 후 우도 추정을 통해 동적으로 업데이트됩니다. 점진적으로 진화하는 이 신념 분포는 정책의 추가적인 맥락적 기반을 제공하여 불확실성 하에서 적응형 대화 전략을 가능하게 합니다.

- **Technical Details**: 이 연구는 부분 가시 마르코프 결정 프로세스(POMDP)로 모델링된 2-agent 대화 사회 상호작용을 연구합니다. 여기서 각 상태는 사회적 맥락과 파트너의 의도에 대한 신념 분포를 포함하는 확장된 상태 공간으로 구성됩니다. 프레임워크는 세 가지 핵심 구성 요소인 의도 모델, 우도 모델 및 확신 인지 행동 정책을 포함하여, 이들을 통해 에이전트의 행동 결정을 돕습니다.

- **Performance Highlights**: 예비 실험 결과, SOTOPIA-All에서 전체 점수가 9.0% 향상되었고, SOTOPIA-Hard에서는 4.1% 향상된 결과를 보여주었습니다. 제안된 프레임워크는 파트너의 의도를 직접 관찰하는 오라클 에이전트보다 약간 뛰어난 성능을 보였습니다. 이러한 초기 결과는 확률적 의도 모델링이 사회적으로 지능을 갖춘 LLM 에이전트 개발에 기여할 수 있음을 시사합니다.



### CircuitSeer: Mining High-Quality Data by Probing Mathematical Reasoning Circuits in LLMs (https://arxiv.org/abs/2510.18470)
Comments:
          14 pages, 5 figures

- **What's New**: 이번 연구는 크게 두 가지 혁신을 제시합니다. 첫째, 기존의 데이터 선택 방법론에서 벗어나 모델 내부의 작동 방식을 저수준으로 분석하여 새로운 신호를 활용하는 접근 방식을 도입했습니다. 둘째, 'CircuitSeer'라는 새로운 데이터 선택 방법을 통해 문제 데이터의 복잡성을 측정하여 이를 기반으로 고품질의 데이터를 자동으로 선택할 수 있는 시스템을 구축했습니다.

- **Technical Details**: CircuitSeer는 복잡한 추론 작업에서 활성화 되는 소수의 특화된 attention heads를 통해 핵심 추론 회로를 식별합니다. 이러한 회로는 Transformer 모델 기술에 기반을 두고 있으며, 각 회로가 특정 작업을 처리하기 위해 발전한다는 원리를 반영하고 있습니다. 이 연구에서는 총 4개의 모델과 9개의 데이터셋을 대상으로 실험을 진행하여 CircuitSeer의 성능이 향상되었음을 입증했습니다.

- **Performance Highlights**: Qwen2.5-Math-7B 모델을 기준으로 하여, CircuitSeer로 선택된 10%의 데이터로 미세 조정을 했을 때 평균 Pass@1에서 1.4점의 성장을 달성했습니다. 이는 전체 데이터셋으로 훈련하는 것보다 훨씬 적은 데이터로도 효과적인 결과를 이끌어낸 사례로, CircuitSeer의 효율성과 효과성을 강조합니다.



### PlanU: Large Language Model Decision Making through Planning under Uncertainty (https://arxiv.org/abs/2510.18442)
Comments:
          38 pages, 19 figures, NeurIPS 2025 Accepted

- **What's New**: 본 논문에서는 대형 언어 모델(LLM)을 기반으로 하는 계획 방법론인 PlanU를 제안합니다. PlanU는 몬테카를로 트리 탐색(MCTS) 내의 불확실성을 포착하여, MCTS의 각 노드의 수익을 양자 분포로 모델링합니다. 이는 LLM의 결정-making 작업에서의 불확실성을 효과적으로 처리할 수 있도록 고안되었습니다.

- **Technical Details**: 이 연구는 LDM을 마르코프 결정 과정(MDP)으로 모델링하며, LLM은 현재 상태를 기반으로 정책을 통해 행동을 생성합니다. 두 가지 주요 불확실성, 즉 LLM 불확실성과 환경 불확실성을 다루며, 이는 환경의 내재적 무작위성과 LLM의 생성 과정에서 발생하는 다양한 반응으로 인해 나타납니다. PlanU는 MCTS 트리에서 각 노드의 반환을 양자 분포로 모델링하고, UCC 점수를 도입하여 탐색과 활용의 균형을 맞추도록 설계되었습니다.

- **Performance Highlights**: PlanU는 다양한 환경에서 광범위한 실험을 통해 효과성을 입증하였으며, 기존의 최첨단 방법들보다 우수한 성능을 보여주었습니다. 특히, 환경의 확률적 전이를 처리함으로써 LLM 기반 결정-making 작업에서 높은 정확도를 유지합니다. 이러한 성과는 XX년 XX월에 개최된 XX 학회에서 발표될 예정입니다.



### AlphaOPT: Formulating Optimization Programs with Self-Improving LLM Experience Library (https://arxiv.org/abs/2510.18428)
- **What's New**: 본 연구에서는 AlphaOPT라는 스스로 개선하는 경험 기반 라이브러리를 제안합니다. 이 라이브러리는 LLM(대형 언어 모델)이 제한된 시연에서 학습하고 솔버 피드백을 통해 지속적으로 향상될 수 있도록 합니다. 이를 통해 비구조적인 언어를 수학적 공식 및 실행 가능한 코드로 자동 변환하는 과정에서의 장벽을 허물 수 있습니다.

- **Technical Details**: AlphaOPT는 두 단계의 지속적인 사이클로 작동합니다: 첫째, Library Learning 단계에서는 실패한 시도의 반성을 통해 솔버 검증 통찰을 추출합니다. 둘째, Library Evolution 단계에서는 작업 간의 불일치를 진단하고 적합성 조건을 개선하여 모델의 일반화 능력을 향상시킵니다. 이 과정에서 AlphaOPT는 주어진 기초 데이터에 대해 효율적으로 학습합니다.

- **Performance Highlights**: 실험 결과, AlphaOPT는 100개에서 300개 훈련 항목으로 학습할 때 65%에서 72%로 꾸준한 향상을 보였으며, 아웃 오브 디스트리뷰션(Out-of-Distribution) OptiBench 데이터셋에서 가장 강력한 기본선 대비 7.7% 향상된 성능을 보여주었습니다. AlphaOPT는 또한 여러 벤치마크에서 최첨단 성능을 달성하여 최적화 공식화에 대한 자기 개선 경험 라이브러리 학습의 잠재력을 실증합니다.



### Automated urban waterlogging assessment and early warning through a mixture of foundation models (https://arxiv.org/abs/2510.18425)
Comments:
          Submitted to Nature

- **What's New**: 기후 변화로 인해 도시의 물 고임(waterlogging) 문제가 심각해지고 있으며, 기존의 모니터링 방식은 수동 보고에 의존하여 적시성과 포괄성을 결여하고 있습니다. 본 연구에서는 Urban Waterlogging Assessment (UWAssess)라는 새로운 프레임워크를 제시하며, 이는 감시 이미지에서 자동으로 물 고임 지역을 식별하고 구조화된 평가 보고서를 생성합니다.

- **Technical Details**: UWAssess는 레이블이 부족한 데이터를 해결하기 위해 반지도학습(semi-supervised) 미세 조정(fine-tuning) 전략과 체인 오브 씽크(chain-of-thought, CoT) 프롬핑(prompting) 전략을 채택하였습니다. 이를 통해 데이터가 부족한 후속 작업에서도 기초 모델의 잠재력을 극대화할 수 있습니다.

- **Performance Highlights**: 강력한 시각적 기준에 대한 평가 결과, UWAssess는 인식(perception) 성능에서 상당한 개선을 보여주었습니다. 또한, GPT 기반 평가를 통해 UWAssess가 물 고임의 범위, 깊이, 위험, 영향 등을 정확히 기술하는 신뢰할 수 있는 텍스트 보고서를 생성하는 능력을 입증하였습니다.



### Med-VRAgent: A Framework for Medical Visual Reasoning-Enhanced Agents (https://arxiv.org/abs/2510.18424)
- **What's New**: 이번 연구에서는 의료 시각 추론을 위한 새로운 프레임워크인 Med-VRAgent를 제안합니다. 이 프레임워크는 Visual Guidance(시각 안내)와 Self-Reward(자기 보상) 및 Monte Carlo Tree Search (MCTS)을 활용하여 VLM의 성능을 향상시키는 데 중점을 두고 있습니다. Med-VRAgent는 학습 후에도 좋은 일반화 능력을 발휘하며, 적은 컴퓨팅 자원으로 효과적인 결과를 도출합니다.

- **Technical Details**: Med-VRAgent는 Teacher, Student, Assessor의 세 가지 핵심 모듈로 구성되어 있습니다. Visual Extraction Module은 의료 이미지에서 관심 영역(ROI)을 식별하고, Visual Token Edit을 통해 에이전트의 지역 인식을 향상합니다. 이 과정에서 MCTS를 사용하여 고품질 추론 경로를 탐색하고, External medical knowledge(외부 의료 지식)를 통합하여 사실적 기초를 강화합니다.

- **Performance Highlights**: 여러 의료 VQA 벤치마크에서 Med-VRAgent는 이전의 방법들보다 높은 성능을 보이며 새로운 최첨단(SOTA) 결과를 달성했습니다. 또한 Visual CoT를 통한 추론 기준선보다 뛰어난 성능을 나타내며, IU-Xray 데이터셋에서 retrieval-augmented 방법들을 초월했습니다. 이러한 결과는 시각 안내 기반의 멀티모달 에이전트 프레임워크의 효과성을 잘 보여줍니다.



### Deep Learning-Based Control Optimization for Glass Bottle Forming (https://arxiv.org/abs/2510.18412)
Comments:
          37 pages, 17 figures, accepted for publication in "Expert Systems With Applications"

- **What's New**: 이 연구는 유리병 제조에서 형성 기계의 정밀 제어를 위한 딥 러닝 기반의 제어 알고리즘을 제안합니다. 이 알고리즘은 실제 생산 환경에서 매개변수 변경의 효과를 예측하고 최적의 기계 세팅을 식별하는 데 중점을 두고 있습니다. 이를 통해 생산 공정의 안정성을 높이고 불량률을 최소화 할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 제안된 딥 러닝 모델은 제작 머신의 동작을 복제하도록 훈련되며, 역전 제어 알고리즘을 통해 특정 생산 목표를 만족하는 최적의 세팅을 찾아냅니다. 이 역전 메커니즘은 몬테 카를로(Monte Carlo) 또는 그래디언트(gradient) 기반 방법을 통해 입력값을 최적화하여 실현됩니다. 시스템은 역사적 생산 데이터에 기반하고 있으며, 생산 과정의 실제 변동성을 포착하여 자동으로 기계 매개변수를 조정합니다.

- **Performance Highlights**: 실험 결과는 제안된 방법이 다수의 생산 라인에서 긍정적인 성과를 내며, 생산 정확도를 향상시키고 자재 낭비를 줄이며, 작동 안정성과 근무 안전성을 높일 수 있음을 보여주었습니다. 또한, 딥 러닝 구성 요소는 다양한 제품 유형과 조건에 걸쳐 모델의 일반화 가능성에 기여하며, 예측 및 지침 기회를 제공합니다.



### Heterogeneous Adversarial Play in Interactive Environments (https://arxiv.org/abs/2510.18407)
Comments:
          NeurIPS 2025

- **What's New**: 이번 연구에서는 Heterogeneous Adversarial Play (HAP)라는 새로운 자동 커리큘럼 학습 (Automatic Curriculum Learning) 프레임워크를 제안합니다. HAP는 교사-학생 상호작용을 미니맥스 최적화 문제로 구체화하고, 문제 해결 학습자와 태스크 생성 교사가 적대적인 역동성을 통해 함께 발전하도록 설계되었습니다. 이 프레임워크는 전통적인 정적 커리큘럼이나 단방향 태스크 선택 방식과는 달리 실시간 학습 성과에 따라 태스크 복잡성을 지속적으로 재조정하는 양방향 피드백 시스템을 구축합니다.

- **Technical Details**: HAP는 두 개의 네트워크 구성으로 이루어져 있습니다: 태스크를 생성하는 교사 네트워크와 이 태스크를 마스터하려는 학생 네트워크입니다. 이 적대적 균형은 학습자의 발전하는 능력에 맞춰 태스크 복잡성을 조절하는 커리큘럼을 생성하게 됩니다. 이러한 구조는 학습자의 지식 응집과 효과적인 탐색을 촉진하며, 현재의 자동 커리큘럼 학습 방법론의 한계를 극복합니다.

- **Performance Highlights**: 다양한 멀티태스크 학습 환경에서 HAP의 실험적 검증이 이루어졌으며, 이 프레임워크가 SOTA(Sate-of-the-Art) 기준과 동등한 성능을 발휘하면서도 학습 효율을 높이는 커리큘럼을 생성하는 것을 보여주었습니다. 특히, 복잡한 환경에서는 HAP가 높은 성공률과 학습 효율성을 달성하여 독립적인 적응 행동을 보여주며, 인간 대상 연구에서 효과적인 교수 전략과 유사한 결과를 나타냈습니다.



### Memory-Augmented State Machine Prompting: A Novel LLM Agent Framework for Real-Time Strategy Games (https://arxiv.org/abs/2510.18395)
Comments:
          10 pages, 4 figures, 1 table, 1 algorithm. Submitted to conference

- **What's New**: 이번 논문에서는 LLM 에이전트를 위한 새로운 프레임워크인 Memory-Augmented State Machine Prompting (MASMP)를 제안합니다. MASMP는 기존 접근 방식의 환각 현상(hallucinations)과 단편적 결정-making을 해결하며, 상태머신(prompting)과 메모리 메커니즘을 통합하여 구조화된 행동과 장기 전술 일관성을 통합합니다. 이 프레임워크는 StarCraft II에서 가장 어려운 AI(Level 7)을 상대로 60%의 승률을 기록하여 기존 LLM 기반 베이스라인(0%)을 훨씬 초과하는 성과를 보였습니다.

- **Technical Details**: MASMP는 자연어 구동 상태 머신 아키텍처를 특징으로 하여 LLM이 유한 상태 기계(Finite State Machines, FSMs)와 행동 트리(behavior trees)를 모방할 수 있도록 유도합니다. 이 프레임워크에는 전략적 변수(예: 전술, 우선순위 유닛)를 결정 주기 전반에 걸쳐 보존하는 경량 메모리 모듈이 포함되어 있습니다. 이를 통해 LLM은 최신 결정을 내리는 데 있어 장기적 전술 일관성을 유지할 수 있으며, 비표준 작업 및 규칙을 자연어로 작성하여 요구사항을 충족할 수 있습니다.

- **Performance Highlights**: MASMP는 StarCraft II의 Simple64 맵에서 다양한 난이도로 테스트되었으며, 모든 난이도에서 베이스라인을 크게 초월하는 성과를 거두었습니다. Level 1~5에서는 완벽한 승률을 유지했으며, Level 6과 7에서는 각각 80%와 60%의 높은 승률을 기록했습니다. 이를 통해 MASMP는 기존 규칙 기반 AI와의 경쟁에서 LLM 에이전트의 가능성을 보여줍니다.



### ShortcutBreaker: Low-Rank Noisy Bottleneck with Global Perturbation Attention for Multi-Class Unsupervised Anomaly Detection (https://arxiv.org/abs/2510.18342)
Comments:
          Under Review

- **What's New**: 이번 논문에서는 Multi-class Unsupervised Anomaly Detection (MUAD)을 위한 새로운 프레임워크인 ShortcutBreaker를 제안합니다. 이 방법은 여러 클래스의 이상 탐지를 위해 모델을 통합하며, 별도로 모델을 훈련하지 않아도 되어 계산 자원을 절약할 수 있습니다. 특히, 이 방법은 기존 Transformer 기반 구조에서 발생하는 identity shortcuts 문제를 해결하기 위한 두 가지 혁신적인 기술을 포함합니다.

- **Technical Details**: ShortcutBreaker는 첫 번째로 matrix rank inequality를 활용하여 low-rank noisy bottleneck (LRNB)을 디자인, 이를 통해 고차원 특징을 저차원 잠재 공간으로 투사하여 정체성 재생을 방지합니다. 두 번째로, ViT의 전역 모델링 능력을 활용하여 전역 방해 주의 메커니즘(global perturbation attention, GPA)을 통합, 이는 정보 단축을 방지하는 데 중요한 역할을 합니다.

- **Performance Highlights**: ShortcutBreaker는 MVTec-AD, ViSA, Universal Medical, Real-IAD의 네 개 데이터셋에서 각각 99.8%, 98.9%, 90.6%, 87.8%의 뛰어난 image-level AUROC 성능을 달성했습니다. 특히, 복잡한 Universal Medical 및 Real-IAD 데이터셋에서 이전 방법들을 크게 초월하는 성능을 보였습니다. 이러한 실험 결과는 ShortcutBreaker의 뛰어난 강인성과 다양한 시나리오에서의 우수성을 입증합니다.



### Earth AI: Unlocking Geospatial Insights with Foundation Models and Cross-Modal Reasoning (https://arxiv.org/abs/2510.18318)
- **What's New**: 이 논문은 지구 AI(Earth AI)라는 새로운 지리공간 AI 모델 패밀리를 소개합니다. 지구 AI는 기초 모델을 활용하여 지구를 보다 깊이 이해할 수 있는 획기적인 접근 방식을 제공합니다. 이 모델은 위성 이미지, 인구 데이터, 환경 데이터의 세 가지 주요 도메인에 기반하여 구축되었으며, 복잡한 쿼리를 처리할 수 있는 제미니(Gemini) 지원 에이전트를 개발했습니다.

- **Technical Details**: 지구 AI 모델은 위성, 센서, 인구 통계 데이터 등 다양한 지리공간 데이터를 기반으로 설계되었습니다. 이 모델들은 비디오 언어 모델(Vision-Language Models, VLMs), 개방 어휘 객체 탐지(Open-Vocabulary Object Detection, OVD), 그리고 일반 목적의 비전 트랜스포머(General-Purpose Vision-Transformer, ViT) 백본을 포함하여, 지구 관측의 핵심 과제를 해결하는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: 이 모델들은 실제 세계의 위기 대응 시나리오에서 강력한 예측 능력을 입증했습니다. 다양한 도메인에서 모델들을 통합하여 예측 정확도를 향상시키고, Gemini 기반의 에이전트를 통해 복잡한 지리공간 쿼리를 효과적으로 분석할 수 있는 능력을 보여줍니다. 이러한 성과는 지구 AI가 제공하는 인사이트가 시기적절하고 중요한 정보를 제공하는 데 기여할 수 있음을 보여줍니다.



### Genesis: Evolving Attack Strategies for LLM Web Agent Red-Teaming (https://arxiv.org/abs/2510.18314)
- **What's New**: 이번 연구에서는 웹 에이전트 공격을 다루기 위한 최초의 에이전틱 레드팀팅 프레임워크인 Genesis를 소개합니다. Genesis는 공격 전략을 체계적으로 탐색, 요약 및 진화시키는 것을 목표로 합니다. 이 프레임워크는 유전자 알고리즘(genetic algorithm)과 하이브리드 전략 표현을 결합하여 공격 성공률을 향상시키는 방법을 제시합니다.

- **Technical Details**: Genesis는 세 가지 모듈로 구성된 구조를 가지고 있습니다: Attacker, Scorer, Strategist입니다. Attacker는 유전자 알고리즘과 하이브리드 전략을 활용하여 공격 전략을 생성합니다. Scorer는 웹 에이전트의 반응을 평가하여 피드백을 제공하고, Strategist는 상호작용 로그를 분석하여 성공적인 패턴을 요약하여 전략 라이브러리를 지속적으로 확장합니다.

- **Performance Highlights**: Genesis의 실험 결과는 기존 방법들보다 공격 성공률이 현저하게 높음을 보여줍니다. 또한, Genesis가 발견한 전략들이 새로운 것이며 다양한 백엔드 LLM에 전이 가능하다는 점도 강조됩니다. 결과적으로 Genesis는 웹 에이전트의 보안 취약점을 보다 체계적으로 탐구할 수 있는 기반을 마련합니다.



### Illusions of reflection: open-ended task reveals systematic failures in Large Language Models' reflective reasoning (https://arxiv.org/abs/2510.18254)
- **What's New**: 최근 논문에서는 기존의 큰 언어 모델(LLM)이 사용하는 'reflection' 방식이 인간의 반영적 사고와 기능적으로 유사한지에 대한 연구를 진행하였다. 이들은 종종 목표와 제약 사항에 연결된 오류를 탐지하지만, 실제로는 자가 교정의 한계가 드러나는 경향이 있다. 연구팀은 규칙 제약이 있는 실제 과제를 통해 LLM의 성과를 평가하며, 이는 모델의 진정한 반영 능력을 검증하기 위함이다. 주요 발견은 LLM이 자가 평가와 수정 과정에서 높은 성과를 보이지 못하고 있다는 것이다.

- **Technical Details**: 이 연구는 LLM들이 Cognitive Reflection Test(CRT)라는 과제를 수행하게 하여, 이들이 개발하는 질문의 유효성을 평가하였다. 각 세션에서 모델들은 새로운 테스트 아이템을 생성해야 하며, 이후 자신의 비판을 고려한 수정을 요구받는다. 평가 기준은 비전문가도 쉽게 심사할 수 있는 명확한 통과/실패 조건을 제공하며, 다양한 반영 전략(Explaination, Retry 등)을 통해 이전의 오류를 반복하는 비율을 측정하였다.

- **Performance Highlights**: 실험 결과, 첫 번째 시도에서 LLM들은 주어진 4개의 아이템 중 유효한 아이템을 생산하는 데 실패하며, 반영 과정을 거쳐도 성과는 미미했다. 설명 과정이 별도의 우위를 보이지 않으며, 성과는 열려 있는 문제에서 더욱 낮아지는 경향을 보였다. 이러한 결과는 현재 LLM의 반영 기능이 인간의 목표 지향적 모니터링을 제대로 재현하지 못하고 있다는 점을 시사한다.



### ssToken: Self-modulated and Semantic-aware Token Selection for LLM Fine-tuning (https://arxiv.org/abs/2510.18250)
- **What's New**: 이번 연구에서는 ssToken이라는 새로운 Self-modulated and Semantic-aware Token Selection 방법을 제안합니다. ssToken은 추가적인 reference model 없이도 현재 모델에서 쉽게 접근 가능한 과거 모델을 활용하여 각 토큰의 손실 차이를 계산하는 방식으로 작동합니다. 이는 토큰 선택 과정에서 모델이 최적화 경로를 따라 적응적으로 선택할 수 있도록 지원합니다.

- **Technical Details**: ssToken의 핵심은 손실 기반 선택과는 다른, 의미론적으로 중요한 토큰을 보존하는 주의 기반의 토큰 중요도 추정 메트릭을 도입하는 것입니다. 이 메트릭은 현재 모델의 시맨틱 정보를 활용하여 세밀하게 필터링을 수행하는 데 도움을 줍니다. 또한, ssToken은 메모리와 계산 효율성을 고려한 경량화된 구현으로, FlashAttention과 같은 효율적인 주의 메커니즘과 호환됩니다.

- **Performance Highlights**: 다양한 모델과 스케일에 대한 광범위한 실험 결과, ssToken은 전체 데이터로 파인튜닝한 경우보다 최대 4.3% 성능 향상을 기록하며 이전의 토큰 수준 선택 방법보다 최대 2.8% 더 우수한 결과를 보였습니다. ssToken은 자체적으로 조절된 선택과 의미론적으로 인식된 선택을 결합하여 시너지를 이루며 더욱 향상된 성능을 나타냈습니다.



### A Definition of AGI (https://arxiv.org/abs/2510.18212)
- **What's New**: 본 논문은 인공지능의 일반 지능(AGI)에 대한 구체적인 정의를 제시하며, AGI를 고학력 성인의 인지 다재다능성과 전문성에 맞추어 측정하는 계량적 프레임워크를 제안합니다. 기존 AGI는 모호한 목표로 기능해왔으나, 이 연구는 인간의 인지 능력을 기반으로 하여 AGI를 정의합니다. 이를 통해 차세대 AI 시스템을 평가하는 기준과 방법론이 마련되었습니다.

- **Technical Details**: 프레임워크는 Cattell-Horn-Carroll(CHC) 이론을 기초로 하며, 일반 지능을 열 가지 핵심 인지 영역으로 세분화합니다. 이 방법론은 인간의 심리측정 배터리를 AI 시스템 평가에 적응하여, 현재 AI 시스템의 인지 능력의 포괄적인 검사를 가능하게 합니다. 프레임워크에 따른 AGI 점수는 0%에서 100%까지 정량적 측정을 제공하며, 100%가 AGI를 의미합니다.

- **Performance Highlights**: 현재 AI 시스템은 일반적으로 50% 정도의 핵심 인지 능력만 충족하고 있어, 복잡한 벤치마크에서는 좋은 성과를 보이지만 인간 수준의 일반 지능에는 충분치 않음을 나타냅니다. 예를 들어, GPT-4의 AGI 점수는 27%, GPT-5는 58%로, AGI로서의 진전은 있지만 여전히 큰 격차가 존재함을 보여줍니다. 이러한 결과는 AI가 특정 작업에서는 인간보다 뛰어나지만 전반적으로는 인간과 비교해 좁은 범위의 능력을 지니고 있다는 것을 의미합니다.



### FST.ai 2.0: An Explainable AI Ecosystem for Fair, Fast, and Inclusive Decision-Making in Olympic and Paralympic Taekwondo (https://arxiv.org/abs/2510.18193)
Comments:
          23 pages, 12 figures

- **What's New**: 이 논문은 금메달을 차지하는 올림픽과 패럴림픽 태권도 경기에서 심판, 코치 및 선수의 실시간 지원을 제공하는 설명 가능한 AI 생태계인 FST.ai 2.0을 소개합니다. 이 시스템은 그래프 합성곱 네트워크(Graph Convolutional Networks)를 기반으로 한 자세 인식, 신뢰도 모델링 및 시각적 의사결정 지원을 위한 설명 가능성 오버레이를 통합합니다. 또한 이 시스템은 태권도 생태계에서 공정성 모니터링과 정책 수준의 분석을 포함하여 심판 교육과 공정성을 강화하는 모듈을 추가했습니다.

- **Technical Details**: FST.ai 2.0는 실시간으로 설명 가능하고 신뢰할 수 있는 결정을 지원하는 유연한 모듈식 아키텍처를 구현하였습니다. 이 시스템은 AI 지원 심판 도구, 기술 추적 및 피드백, 그리고 AI기반 교육 플랫폼을 통해 심판, 선수, 코치 간의 협력을 증진시킵니다. 신뢰성 있는 의사결정을 위해 불확실성 모델링을 통합하고, 인터랙티브 대시보드를 통해 시각적 피드백을 제공합니다.

- **Performance Highlights**: 경쟁 데이터에서의 실험적 검증 결과, FST.ai 2.0은 의사결정 리뷰 시간을 85% 단축시키고 AI 지원 결정에 대한 심판 신뢰도를 93%로 높였습니다. 이로 인해 데이터 기반의 공식 채점 및 선수 평가를 위한 투명하고 확장 가능한 파이프라인이 구축됩니다. 이 시스템은 AI가 스포츠에서 인간의 전문성을 저해하지 않으면서 공정하고 포괄적인 결정 생태계로 이동하는 데 기여하고자 합니다.



### Local Coherence or Global Validity? Investigating RLVR Traces in Math Domains (https://arxiv.org/abs/2510.18176)
Comments:
          4 pages, 2 figures

- **What's New**: 본 논문은 Reinforcement Learning with Verifiable Rewards (RLVR) 기반의 대형 언어 모델(LLM) 후속 훈련이 추론(task에 대한 reasoning) 정확도를 개선할 수 있음을 보여줍니다. 기존의 RLVR 방법들은 모든 토큰을 균일하게 다루며, 중간 토큰의 가능성을 고려하지 않았습니다. 이를 해결하기 위해 저자들은 중간 토큰에 대한 RL 후속 훈련의 효과를 조사하고, 오류를 식별하여 추론 단계의 일관성을 측정하는 trace coherence라는 새로운 지표를 도입했습니다.

- **Technical Details**: 본 연구는 GRPO 알고리즘을 이용하여 Qwen-2.5-0.5B 모델을 GSM8K 데이터셋에 적용하여 실험적으로 RLVR의 효과를 분석했습니다. 저자들은 First-Order Logic (FOL) 기반의 측정 지표인 trace coherence를 도입하여, 추론 과정에서의 오류 유무를 교차 분석했습니다. 이를 통해 RLVR이 최종 답변의 정확성을 넘어 추론 과정의 일관성(coherence)에 미치는 영향을 정량적으로 평가했습니다.

- **Performance Highlights**: 실험 결과는 RL 후속 훈련이 전반적으로 trace coherence를 개선하며, 특히 기본 모델이 실패하는 문제에서 RL 모델이 성공하는 경향이 있음을 보여줍니다. 그러나 RL이 항상 유효한 또는 정확한 솔루션을 생성하지는 않지만, 추론 단계의 로컬 일관성을 향상시키는 데 기여하고 있음을 나타냈습니다. 이러한 발견은 RLVR의 추론 품질 개선 주장을 면밀히 검토해야 할 필요성을 강조합니다.



### AgentChangeBench: A Multi-Dimensional Evaluation Framework for Goal-Shift Robustness in Conversational AI (https://arxiv.org/abs/2510.18170)
Comments:
          Accepted to 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: Multi-Turn Interactions in Large Language Models

- **What's New**: 본 논문에서는 AgentChangeBench를 소개하여 대화 중 목표 변경을 측정하는 새로운 벤치마크를 제시하고 있습니다. 기존의 벤치마크는 고정된 목표에 대한 성능의 평가에 치중했으나, AgentChangeBench는 동적 목표 변화를 다루며, 이로 인해 LLM(대형 언어 모델)이 어떻게 적응하는지를 평가합니다.

- **Technical Details**: AgentChangeBench는 2,835개의 작업 시퀀스와 다섯 가지 사용자 페르소나로 구성되어 있으며, 이에 따라 다음과 같은 네 가지 평가 메트릭(TSR, TUE, TCRR, GSRT)을 통해 에이전트의 성능을 측정합니다. 이 벤치마크는 금융 서비스, 소매, 항공사 등 세 가지 분야에서 실질적인 고객 서비스 워크플로우를 기반으로 합니다.

- **Performance Highlights**: 에이전트들이 동적 목표에 적응하는 데 있어 큰 성능 차이를 보였고, 예를 들어, GPT-4o는 항공 예약 목표 변화에서 92.2%의 회복률을 달성한 반면, Gemini는 48.6%로 급락했습니다. 이러한 결과는 고정된 정확도가 동적 목표에서의 강인성을 항상 보장하지 않음을 보여줍니다.



### Saber: An Efficient Sampling with Adaptive Acceleration and Backtracking Enhanced Remasking for Diffusion Language Mod (https://arxiv.org/abs/2510.18165)
- **What's New**: Diffusion language models (DLMs)는 기존의 autoregressive 모델에 대한 유망한 대안으로 부상하고 있습니다. 이 모델은 코드 생성을 포함한 많은 작업에서 병렬 생성 및 양방향 맥락 모델링의 장점을 제공합니다. 하지만 다음과 같은 두 가지 주요 문제인 속도-품질 트레이드오프와 오류 전파에 관한 문제로 인해 성능이 저하되는 어려움이 있습니다.

- **Technical Details**: 이 논문에서는 Saber라는 새로운 샘플링 알고리즘을 제안합니다. Saber는 Adaptive acceleration과 Backtracking Enhanced Remasking의 약자로, 두 가지 주요 전략인 비균일 난이도 조정과 오류 수정 메커니즘을 통합하여 성능을 개선합니다. DLM의 샘플링 과정에서 적응형 가속 전략을 사용하여 초반에는 신중하게, 이후에는 더 공격적으로 토큰을 생성하도록 설계되었습니다.

- **Performance Highlights**: Saber는 다양한 코드 생성 벤치마크에서 기존 DLM 샘플링 방법에 비해 Pass@1 정확도를 평균 1.9% 향상시키고, 평균적으로 251.4%의 추론 속도 향상을 달성했습니다. 이를 통해 DLM은 기존의 autoregressive 모델에 비해 성능 격차를 크게 줄이게 되었습니다.



### LLM-Based Multi-Agent System for Simulating and Analyzing Marketing and Consumer Behavior (https://arxiv.org/abs/2510.18155)
Comments:
          Accepted for publication at IEEE International Conference on e-Business Engineering ICEBE 2025, November 10-12, Buraydah, Saudi Arabia. 8 pages, 5 figures

- **What's New**: 이 연구는 소비자 의사결정을 모델링하기 위해 LLM(대형 언어 모델) 기반의 다중 에이전트 시뮬레이션 프레임워크를 도입합니다. 기존의 규칙 기반 에이전트 기반 모델(ABMs)은 인간 행동의 복잡성과 사회적 상호작용을 포착하는 데 한계가 있어, 이 프레임워크는 생성하는 에이전트들이 내면의 추론을 표현하고, 습관을 형성하며, 사전 규칙 없이 구매 결정을 내릴 수 있도록 합니다. 결과적으로 이 시스템은 마케팅 전략 사전 테스트를 위한 실용적인 도구로 활용될 수 있습니다.

- **Technical Details**: 연구는 11명의 에이전트와 10개의 위치를 설정하여, 가격 할인에 따른 소비자 행동을 시뮬레이션합니다. DeepSeek-V3 모델을 통해 에이전트들은 자연스럽게 계획하고 실행하며, 다양한 외부 요인과 내적 요인에 영향을 받아 소비 결정을 내립니다. 시뮬레이션은 에이전트가 각자의 필요와 환경을 고려하여 행동을 결정하는 복잡한 논리를 구현하며, 각 에이전트는 22세에서 35세 사이의 다양한 인구 통계학적 특성을 보유하고 있습니다.

- **Performance Highlights**: 가격 할인 전략을 구현한 본 연구는 에이전트의 행동이 어떻게 변화하는지를 관찰할 수 있게 하여, 소비자 의사결정의 복잡성을 효과적으로 시뮬레이션합니다. 시뮬레이션을 통해 에이전트가 예기치 않은 사회적 행동 패턴을 보이는지 확인하고, 이는 마케팅 전략이 어떻게 영향을 미칠 수 있는지를 보여줍니다. 또한, 이 접근법은 시간과 비용을 절감하고, 기존 방법으로는 얻을 수 없는 실행 가능한 통찰력을 제공합니다.



### Annotating the Chain-of-Thought: A Behavior-Labeled Dataset for AI Safety (https://arxiv.org/abs/2510.18154)
- **What's New**: 본 연구에서는 AI의 안전성을 모니터링하기 위해 새로운 생역학적 접근을 소개합니다. 이를 위해 문장 단위로 주석이 달린 데이터셋을 개발하여 LLM(대형 언어 모델) 추론 중의 안전 행동을 활성화 기반으로 모니터링할 수 있게 합니다. 이 데이터셋은 기존의 안전 연구에서 부족했던 부분을 메우며, 특정 행동이 추론 체인 내에서 언제 발생하는지를 정밀하게 식별합니다.

- **Technical Details**: 이번 데이터셋은 20개의 안전 행동을 포함하며, 각 문장은 그들이 표현하는 안전 행동에 따라 주석이 달립니다. 데이터셋은 50,000개 이상의 주석이 달린 문장으로 구성되어 있으며, 이러한 문장들은 위험한 프롬프트에 대한 모델의 응답을 통해 수집되었습니다. 이 데이터셋을 통해 모델의 내부 활성화에서 특정 행동을 탐지하고 조정하기 위한 조향 벡터를 추출할 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: 제안된 데이터셋은 문장 단위의 행동 주석을 통해 특정 안전 행동이 모델의 활성화 수준에서 발생할 때를 감지할 수 있게 해줍니다. 실험 결과, 이러한 조향 벡터는 안전 행동을 탐지할 수 있는 능력을 입증하며, 모델의 최종 응답에 대한 위험성 점수를 분석한 결과 유의미한 패턴을 보여주었습니다. 안전 지향 안전 행동은 긍정적인 관계를 나타내며, 반면 위험한 준수 행동은 부정적인 관계를 가지고 있음을 확인했습니다.



### Learning from Generalization Patterns: An Evaluation-Driven Approach to Enhanced Data Augmentation for Fine-Tuning Small Language Models (https://arxiv.org/abs/2510.18143)
Comments:
          Neural Information Processing Systems (NeurIPS 2025) Workshop: Evaluating the Evolving LLM Lifecycle

- **What's New**: 이번 연구에서는 PaDA-Agent(Pattern-guided Data Augmentation Agent)를 소개하여, 소형 언어 모델(Small Language Models, SLMs)의 데이터 증대 과정을 간소화합니다. 기존의 모델 훈련 오류에만 초점을 맞춘 기법들과는 달리, PaDA-Agent는 검증 데이터를 통해 실패 패턴을 발견하고 이를 바탕으로 목표 지향적인 데이터 증대 전략을 수립합니다. 즉, 일반화 오류를 직접 수정하는 방식으로 SLM의 성능을 개선하는 새로운 접근법을 제시합니다.

- **Technical Details**: PaDA-Agent는 이터레이션 방식으로 운영되며, 중앙 조정자(orchestrator)에 의해 세 개의 에이전트가 협력하여 훈련 데이터를 증대합니다. 패턴 분석 에이전트는 검증에서의 일반화 실패를 추출하고, 데이터 생성 에이전트는 그에 따라 합성 데이터를 생성합니다. 품질 관리 에이전트는 생성된 데이터의 적합성, 유용성, 관련성을 평가하여, 기준 미달인 데이터는 재생성 과정을 거치게 합니다.

- **Performance Highlights**: PaDA-Agent는 Llama 3.2 1B Instruct 모델의 미세 조정 시 기존의 데이터 증대 방법들과 비교하여 성능이 꾸준히 향상됨을 입증하였습니다. 1000개 샘플 환경에서 모든 과제에서 가장 우수한 결과를 기록하였고, 특히 HellaSwag에서 51.2%의 정확도를 달성했습니다. 저자원 환경에서도 validation-driven augmentation이 가장 큰 이점을 제공함을 확인하였습니다.



### Measuring Reasoning in LLMs: a New Dialectical Ang (https://arxiv.org/abs/2510.18134)
- **What's New**: 이 논문은 언어 모델(LLM)의 "추론"이 무엇을 의미하는지를 탐구합니다. 기존의 평가 방법은 모델의 정답 여부에만 초점을 맞추지만, 이 연구에서는 추론이 정적인 단계가 아니라 아이디어들이 상호 작용하고 발전하는 역동적인 과정이라고 주장합니다. 이를 위해 저자들은 변증법(dialectics)이라는 철학적 전통을 활용하여 SIEV라는 구조화된 프레임워크를 제안합니다.

- **Technical Details**: SIEV는 LLM의 추론을 변증법적인 관점에서 평가하는 도구로, 모델이 도출하는 결론뿐만 아니라 그 과정(즉, 긴장 해결, 아이디어의 통합, 고차원적인 추론의 합성을 어떻게 수행하는지)을 중시합니다. 이 접근법은 LLM의 추론 능력을 더 깊이 이해하기 위해 과거의 평가 방식과는 다른 시각을 제공합니다. SIEV는 기존 데이터를 활용하여 복잡한 아키텍처 변경 없이도 적용 가능하다는 장점을 지닙니다.

- **Performance Highlights**: SIEV를 통해 최근의 모델인 GPT-5-chat 등이 기존의 성능 기준인 MMLU 및 GSM에서 여전히 상당한 추론 격차(20% 이상)를 보임을 발견했습니다. 이는 SIEV가 기존의 평가 방법보다 더 엄격하고 변별력 있는 추론 평가를 제공함을 의미합니다. 궁극적으로 이러한 과정 중심의 평가 방식이 LLM의 능력과 한계를 더욱 잘 이해할 수 있게 도와줄 것입니다.



### SMaRT: Select, Mix, and ReinvenT -- A Strategy Fusion Framework for LLM-Driven Reasoning and Planning (https://arxiv.org/abs/2510.18095)
- **What's New**: 이 논문에서는 Select, Mix, and ReinvenT (SMaRT)라는 새로운 전략 융합 프레임워크를 소개합니다. 이 프레임워크는 단일 전략 프롬프트의 한계를 극복하고, 다양한 추론 전략을 통합하여 성능을 극대화합니다. SMaRT는 기존 방법들과 달리 LLM을 평가자가 아닌 지능적인 통합자로 활용하여, 각 작업에서 최고의 결과를 이끌어냅니다.

- **Technical Details**: SMaRT 프레임워크는 두 개의 단계로 운영됩니다. 첫 번째 단계인 초기 솔루션 단계에서는 LLM이 다양한 기본 전략을 사용하여 후보 솔루션을 생성합니다. 두 번째 단계인 융합 단계에서는 이 후보 솔루션을 평가하고, 다양한 전략의 요소를 통합하여 최종 솔루션을 생성합니다.

- **Performance Highlights**: 실험 결과, SMaRT 프레임워크는 재료 전략을 사용한 기존 방법들과 비교해 우수한 성능을 보여주며, LLM 기반 기술의 경계를 확대합니다. 다양한 벤치마크에서 SMaRT는 전체 작업의 제약 조건 준수와 해결책 품질 측면에서 일관되게 뛰어난 결과를 나타냈습니다. 또한, 작은 오픈소스 LLM과 대형 API 기반 LLM의 출력 결합을 통해 성능이 더욱 향상되었습니다.



### Planned Diffusion (https://arxiv.org/abs/2510.18087)
Comments:
          10 pages, 8 figures

- **What's New**: 이번 논문에서는 텍스트 생성의 속도와 품질 간의 균형을 맞추는 새로운 방법인 planned diffusion을 제안합니다. 이 방법은 기존의 autoregressive 모델과 diffusion 모델의 장점을 결합하여, 텍스트 품질을 유지하면서도 속도를 개선합니다. planned diffusion은 두 단계로 구성되어 있으며, 이는 텍스트 생성의 효율성을 크게 향상시킵니다.

- **Technical Details**: planned diffusion은 첫 번째 단계에서 짧은 autoregressive 계획을 생성하여 출력을 작은 독립적인 구간으로 나눕니다. 이후 두 번째 단계에서는 이 구간들을 동시에 생성하는 Diffusion 모델 방식을 사용합니다. 이 접근 방식은 속도와 품질 간의 Pareto 경계를 확장하여 더 빠르고 고품질의 텍스트 생성을 가능하게 합니다.

- **Performance Highlights**: 본 논문에서 제시한 방법은 AlpacaEval에서 805개의 지침 준수 프롬프트를 활용하여 품질과 지연 시간 간의 Pareto 최적화를 달성했습니다. 결과적으로 proposed 방법은 autoregressive 생성보다 1.27배에서 1.81배 빨라졌으며, 품질은 오히려 0.87%에서 5.4%의 감소로 유지되었습니다. 또한, planning 메커니즘의 신뢰성이 높아 고품질과 지연 시간 간의 유연한 조정을 제공하는 간단한 런타임 설정이 가능합니다.



### CompactPrompt: A Unified Pipeline for Prompt Data Compression in LLM Workflows (https://arxiv.org/abs/2510.18043)
Comments:
          Workshop on LLMs and Generative AI for Finance at ACM ICAIF 2025

- **What's New**: 이 논문에서는 CompactPrompt라는 새로운 파이프라인을 소개합니다. 이 파이프라인은 하드 프롬프트 압축(hard prompt compression)과 경량 데이터 압축(lightweight file-level data compression)을 통합하여, 에이전트의 작업 흐름에서 발생하는 계산 비용을 줄이는 데 중점을 둡니다. CompactPrompt는 정보가 적은 토큰을 제거하고, 텍스트와 숫자 데이터를 압축하여 모델의 성능을 유지하면서도 총 토큰 사용량을 최대 60% 줄일 수 있습니다.

- **Technical Details**: CompactPrompt는 자체 정보 점수(self-information scoring)와 의존성 기반 구문 그룹화(dependency-based phrase grouping)를 통해 저정보 토큰을 제거하고, n-그램 약어(n-gram abbreviation) 기법을 사용하여 부가 문서에서 반복되는 텍스트 패턴을 압축합니다. 또한, 숫자 열에 대해 균일 양자화(uniform quantization)를 적용하여, 데이터 유실 없이도 신뢰할 수 있는 분석을 유지하며 메모리 사용량을 줄입니다. 이러한 과정은 별도의 모델 재훈련 없이도 진행할 수 있습니다.

- **Performance Highlights**: CompactPrompt를 표준 LLM 에이전트에 통합하면 TAT-QA 및 FinQA와 같은 벤치마크 데이터셋에서 60%의 추론 비용 절감과 함께 출력 품질을 유지할 수 있습니다. Claude-3.5-Sonnet 및 GPT-4.1-Mini 모델의 경우에도 5% 미만의 정확도 저하로 고품질 출력을 보장합니다. 이 시스템은 실시간 압축 결정 시각화 및 비용-성능 무역 추적을 가능하게 하여 보다 효율적인 생성 AI 파이프라인 구축의 기초를 마련합니다.



### Subject-Event Ontology Without Global Time: Foundations and Execution Semantics (https://arxiv.org/abs/2510.18040)
Comments:
          32 pages

- **What's New**: 이 논문에서는 복잡한 동적 시스템을 모델링하기 위해 전 세계 시간에 의존하지 않는 주제-사건 온톨로지(subject-event ontology)의 형식을 제안합니다. 주요 원칙으로는 사건을 고정의 행동으로 정의하고, 사건의 순서를 명시적 의존성에 의해 정해진다고 설명합니다. 또한, 온톨로지를 실행 가능하도록 만드는 데이터 흐름 메커니즘을 통해 결정성을 보장하는 점이 특징입니다.

- **Technical Details**: 제안된 형식화는 아홉 가지 공리(A1-A9)를 포함하여 실행 가능한 온톨로지의 정확성을 보장합니다. 주요 기술 요소로는 역사(mhistory)의 단조성(I1), 원인(causality)의 비순환성(I2), 추적 가능성(traceability)(I3)이 있으며, 사건 검증을 위한 스키마(schema) 기반의 모델 접근 방식(A9)에 특별한 주목을 합니다. 또한 전 세계 시간(global time) 없이도 원인 사슬(causal chains)을 자동으로 구축할 수 있는 방안을 제시합니다.

- **Performance Highlights**: 이 형식화는 BSL(Boldsea Semantic Language)로 구현된 boldsea 시스템을 통해 실용성을 입증합니다. 이 시스템은 실행 가능한 온톨로지를 위한 워크플로 엔진(workflow engine)으로 기능하며, 분산 시스템(distributed systems), 마이크로서비스 아키텍처(microservice architectures), DLT 플랫폼 및 다면적 시나리오(multiperspectivity scenarios)에도 적용 가능합니다. 이 논문은 서로 다른 주체(subject)들 간의 상충하는 사실(conflicting facts) 처리에도 유용합니다.



### OPTAGENT: Optimizing Multi-Agent LLM Interactions Through Verbal Reinforcement Learning for Enhanced Reasoning (https://arxiv.org/abs/2510.18032)
Comments:
          8 pages for main content

- **What's New**: 이 논문에서는 복잡한 추론을 향상시키기 위해 LLM(대규모 언어 모델) 기반의 다중 에이전트 시스템을 제안하고 있습니다. 기존의 협력 구조가 미리 정의되어 있거나 단순한 집단 토론 방식으로만 운영되어 문제가 발생한다고 지적하며, 여러 LLM 에이전트 간의 효과적인 의사소통이 중요하다는 hypothesis를 설정했습니다. 이를 해결하기 위해 $	ext{OptAgent}$라는 새로운 다중 에이전트 언어 강화 학습 알고리즘을 제안하여 에이전트 간의 협력 구조를 동적으로 구축하고 수정할 수 있게 합니다.

- **Technical Details**: $	ext{OptAgent}$는 대화의 강인성과 일관성을 평가하는 피드백 메커니즘을 포함하는 행동 공간을 정의하고, 결정은 모든 에이전트 간의 다수결을 통해 이루어집니다. 이 프레임워크는 각 에이전트를 노드로 간주하고, 에이전트 간의 통신을 엣지로 모델링하여, 에이전트 사이의 최적의 연결 순서를 찾는 것을 목표로 합니다. 이를 위해 두 개의 메타 에이전트인 LLM_reflect 및 LLM_act를 사용하여 반영 및 행동 프로세스를 처리합니다.

- **Performance Highlights**: $	ext{OptAgent}$는 수학적 추론, 창의적 글쓰기, 과학적 추론 및 숫자 정렬을 포함한 여러 추론 작업에서 테스트되었습니다. 결과적으로, $	ext{OptAgent}$는 단일 에이전트 프롬프트 방법 및 최첨단 다중 에이전트 프레임워크보다 현저히 우수한 성능을 보였습니다. 또한 다양한 LLM 패밀리에서 여러 작업을 통해 그 효능을 입증하는 사례 연구를 제시하였습니다.



### FABRIC: Framework for Agent-Based Realistic Intelligence Creation (https://arxiv.org/abs/2510.17995)
Comments:
          51 Pages, 38 Listings, 5 Figures

- **What's New**: 이 연구에서는 LLM(대규모 언어 모델)을 이용하여 인간의 개입 없이 에이전틱 데이터(agentic data)를 합성하는 통합 프레임워크를 제안합니다. 이 프레임워크는 모듈화된 파이프라인을 통해 각종 작업 명세, 도구 정의, 정책 의사코드, 자연어 교환 및 실행 추적을 포함하는 완전한 상호작용 기록을 생성합니다. 생성된 기록은 엄격한 구문 및 의미 제약을 준수하여 기계 파싱이 가능하며, 입력, 출력 및 도구 호출 간의 충실한 정렬을 보장합니다.

- **Technical Details**: 제안된 SYTHIA 프레임워크는 네 가지 모듈화된 파이프라인으로 구성되어 있으며, 각 파이프라인은 다양한 형태의 에이전틱 데이터를 생성합니다. 이 구조는 사용자 의도, 도구 사양, 의사결정 로직, 그리고 검증 가능한 실행 추적을 포함한 기계 확인 가능 형식으로 데이터를 캡쳐합니다. 데이터 생성 과정에서는 스키마 유효성 검사 및 제약된 생성 형식을 통합하여 품질과 일관성을 보장합니다.

- **Performance Highlights**: 연구에서는 LLM만을 이용한 대규모 데이터 생성의 가능성을 강조하며, 이 접근 방식이 도구 사용 정책을 학습하고 정확성, 정렬성, 재현성 평가에 필수적임을 보여줍니다. 또한, 다양한 워크플로우, 반사실적 사례 및 극한의 경우를 포함한 데이터 세트를 통해 강건성을 개선하는 데 초점을 맞추고 있습니다. 이러한 방식은 수작업 수집의 한계를 극복하고, 고품질 합성 데이터 생성을 위한 확장 가능한 파이프라인을 제공합니다.



### Beyond More Context: Retrieval Diversity Boosts Multi-Turn Intent Understanding (https://arxiv.org/abs/2510.17940)
Comments:
          15 pages,6 figs

- **What's New**: 이번 논문에서는 멀티턴 의도 이해(multi-turn intent understanding)를 개선하기 위한 새로운 접근 방식을 제안합니다. 기존의 방법들이 관련성(relevance)에 초점을 맞추는 경향이 있는 반면, 저자들은 검색된 콘텐츠의 다양성(diversity)이 멀티턴 대화 상황에서 더 나은 성과를 낼 수 있다고 주장합니다. 이를 위해 의도 범위(intent coverage)와 언어 다양성(linguistic variety)을 균형 있게 고려하는 프레임워크를 도입했습니다.

- **Technical Details**: 새롭게 제안된 프레임워크인 LDRA(Linguistic-Diversity Retrieval-Augmentation)는 주어진 대화 맥락(context)에서 사용자 쿼리를 인코딩하고, 관련성에 필터링된 후보 풀을 검색한 후, 라벨 다양성과 텍스트 다양성(objective)을 바탕으로 재정렬(re-ranking)합니다. 이 과정은 의도 카테고리의 범위를 균형 있게 커버하며 언어적 변화를 최적화하도록 구성됩니다. 저자들은 이 접근 방식이 멀티턴 대화의 품질을 높이는 데 효과적임을 보여주고 있습니다.

- **Performance Highlights**: 연구 결과에 따르면, 제안된 LDRA 방식은 MultiWOZ 2.4 및 SGD 데이터셋에서 강력한 성과를 달성했으며, 동일한 토큰 예산 내에서 기존의 DST/LLM 기초선 모델(baseline)을 초과하는 성과를 남겼습니다. 또한, 다양한 백본 모델(backbone size)과 훈련 방식을 활용한 실험에서도 일관된 개선이 나타났습니다. 이를 통해 LLM 기반의 대화형 시스템에서 예산 제약 하에서도 효과적인 멀티턴 의도 체계 구축이 가능함을 입증했습니다.



### Activation Manifold Projection: Liberating Task-Specific Behaviors from LLM Architectures (https://arxiv.org/abs/2510.17902)
- **What's New**: 본 논문에서는 기존의 Low-Rank Adaptation(LoRA)와 같은 미세 조정 방법들이 가지는 구조적 잠금(architectural lock-in)의 문제를 해결하기 위해 새로운 접근법인 Cartridge Activation Space Transfer(CAST)를 도입합니다. CAST는 두 개의 서로 다른 대형 언어 모델(LLM) 간의 내부 활성화 구조에서 비선형 매핑을 학습하여 LoRA로 인코딩된 행동을 해방시킵니다. 이 방법은 모델의 활성화 패턴을 통해 직접적인 전이를 가능하게 하며, 기존의 방식보다 훨씬 직관적이고 견고합니다.

- **Technical Details**: CAST는 LoRA가 적용된 모델의 각 층에 대해 경량의 이중 투영 헤드를 학습하여 목표 모델의 활성화 스트림을 소스 모델의 잠재 공간으로 변환합니다. 이 과정에서는 사전 훈련된 LoRA를 고정된 '행동 커널'(behavioral kernel)로 사용하며, 성능을 유지하면서도 데이터나 추가 훈련 없이도 효과적인 전이를 달성합니다. CAST는 두 개의 투영 행렬을 사용하여 활성화 공간을 맵(맵핑)하고, 이를 통해 모델 간의 행동 전이와 기하학적 정렬을 달성합니다.

- **Performance Highlights**: CAST는 Llama-2와 Mistral과 같은 이질적인 모델 간의 전이 실험에서 85-95%의 성능을 유지하며, 완전 재훈련된 LoRA와 비교하여 정량적으로 우수한 성능을 보입니다. 이는 기존의 가중치 공간 전이(weight-space transfer) 기술들을 초월하는 새로운 기준을 설정하며, 모델 상호 운용성(interoperability)에서 새로운 성과를 이룹니다. CAST는 사용자가 기존의 LoRA 어댑터를 활용할 수 있도록 하여, 그 생태계를 구조적 제약에서 자유롭게 합니다.



### Grasp Any Region: Towards Precise, Contextual Pixel Understanding for Multimodal LLMs (https://arxiv.org/abs/2510.18876)
- **What's New**: 이 논문에서는 Grasp Any Region (GAR)을 제안하여 멀티모달 대형 언어 모델(MLLM)의 지역 이해 능력을 개선하고자 하였습니다. 기존의 지역 MLLMs가 중요 글로벌 컨텍스트(global context)를 간과했던 점을 보완하며, GAR는 지역 수준의 시각적 이해를 위한 종합적인 접근법을 제공합니다. GAR는 RoI-aligned feature replay 기법을 활용해 개별 지역의 정확한 인식과 여러 프롬프트 간의 상호작용 모델링을 지원합니다.

- **Technical Details**: GAR은 특정 프롬프트와 함께 전체 이미지 정보를 인코딩하는 방법으로, 이를 통해 지역별 세부 정보를 정확하게 캡처하는 능력을 향상시킵니다. 이러한 방식은 RoI-Align을 통해 글로벌 Feature map에서 관련된 Feature를 수집하여 지역과 글로벌 정보를 동시에 고려할 수 있게 합니다. 또한, GAR-Bench를 도입하여 여러 지역의 상호작용과 복잡한 추론 과정을 평가하는 새로운 벤치마크를 제공합니다.

- **Performance Highlights**: 실험 결과 GAR-1B 모델은 DAM-3B 및 PAM-3B보다 상세한 캡션 생성에서 우수한 성능을 보여주며, 다중 프롬프트 모델링에서 InternVL3-78B를 초월하는 성과를 달성했습니다. GAR-8B는 VideoRefer-BenchQ에서 인도메인 모델인 VideoRefer-7B를 초과하는 성능을 발휘하여 동영상에 대한 이해 능력 또한 강화되었습니다. 이러한 결과는 GAR의 강력한 커뮤니케이션 및 이해 능력을 입증합니다.



### How Do LLMs Use Their Depth? (https://arxiv.org/abs/2510.18871)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)이 깊이를 균일하게 사용하지 않는다는 성장하는 증거를 통해, 레이어별 예측 동역학(layer-wise prediction dynamics)에 대한 더 세밀한 이해를 제공합니다. 연구진은 여러 개방형 가중치 모델을 추적하며, LLM의 'Guess-then-Refine' 프레임워크를 제안하여 모델이 추론 시 내부적으로 어떻게 계산을 구조화하는지를 설명합니다. 연구 결과는 LLMs가 예측을 위해 깊이를 동적으로 사용하는 방식을 명확히 보여줍니다.

- **Technical Details**: 논문은 LLM의 중간 표현을 추적하고 이를 통해 모델이 예측하는 방식의 구조화된 사용을 밝혔다. TunedLens라는 방법론을 사용하여 중간 레이어 표현을 디코드하고, 이는 LLM이 추정하는 토큰 예측 패턴을 정량화하는 데 도움을 주었습니다. 연구에서는 GPT2-XL, Pythia-6.9B, Llama2-7B, Llama3-8B 등 네 가지 개방형 모델을 사용하였습니다.

- **Performance Highlights**: 결과적으로, 초기 레이어에서의 예측은 고빈도 토큰으로 구성되는 경향이 있으며, 이러한 초기 제안이 이후 레이어에서 실질적으로 수정되었다는 점이 밝혀졌습니다. 연구진은 특정 태스크에 따라 LLM이 깊이를 다르게 사용하는 방식도 관찰하였으며, 더 복잡한 예측은 더 많은 깊이를 요구하는 반면, 쉬운 계산은 보다 빠르게 마무리된다는 사실도 확인하였습니다. 이러한 발견은 Transformer 기반 모델의 계산 효율성을 향상하는 데 기여할 수 있는 통찰을 제공합니다.



### LightMem: Lightweight and Efficient Memory-Augmented Generation (https://arxiv.org/abs/2510.18866)
Comments:
          Work in progress

- **What's New**: 최근의 연구 결과에 따르면, 대형 언어 모델(LLM)의 강력한 능력에도 불구하고 과거 상호작용 정보를 효과적으로 활용하는 데 한계가 있습니다. 이 논문에서는 LightMem이라는 새로운 메모리 시스템을 소개하며, 인간 기억의 Atkinson-Shiffrin 모델에 영감을 받아 구성되었습니다. LightMem은 정보 저장 및 검색 방식을 최적화하여 메모리 시스템의 성능과 효율성을 균형 있게 구현하고 있습니다.

- **Technical Details**: LightMem은 세 가지 상호 보완적인 단계로 메모리를 구성합니다. 첫째, 인지에 영감을 받은 감각 메모리는 관련 없는 정보를 신속하게 필터링하고 주제에 따라 정보를 그룹화합니다. 둘째, 주제 인식 단기 메모리는 이러한 주제 기반 그룹을 통합 및 요약하여 구조화된 접근을 가능하게 합니다. 마지막으로, 수면 시간 업데이트를 가진 장기 메모리는 오프라인 절차를 통해 통합을 온라인 추론과 분리하여 관리합니다.

- **Performance Highlights**: LightMem은 LongMemEval에서 강력한 기준선을 초과하여 정확도에서 최대 10.9%의 개선을 보여주며, 토큰 사용량은 최대 117배, API 호출은 159배, 실행 시간은 12배 이상 줄였습니다. 또한, Case study를 통해 오프라인 수면 시간 통합이 장기적인 지식 업데이트의 신뢰성을 높이고 정보 손실 및 불일치를 완화하는 데 기여함을 보여주었습니다.



### Every Step Evolves: Scaling Reinforcement Learning for Trillion-Scale Thinking Mod (https://arxiv.org/abs/2510.18855)
Comments:
          Technical Report

- **What's New**: Ring-1T는 첫 번째 오픈소스의 최첨단 사고 모델로 1조(1 trillion) 개의 파라미터를 가지고 있습니다. 이 모델은 총 1조 개의 파라미터를 특징으로 하며, 토큰당 약 500억 개를 활성화합니다. 이는 기존 모델과 비교하여 비약적인 발전을 보여줍니다.

- **Technical Details**: 모델 훈련 과정에서 발생하는 여러 가지 도전 과제를 해결하기 위해 세 가지 혁신이 도입되었습니다. 첫 번째, IcePop은 토큰 수준의 불일치를 마스킹 및 클리핑하여 RL 훈련의 불안정을 해소합니다. 두 번째, C3PO++는 동적으로 토큰을 분할해 장기간 롤아웃에서 자원 활용을 개선하고 높은 시간 효율성을 도출합니다. 마지막으로, ASystem은 1조 파라미터 모델 훈련을 방해하는 시스템적 병목현상을 극복하기 위해 설계된 고성능 RL 프레임워크입니다.

- **Performance Highlights**: Ring-1T는 AIME-2025에서 93.4, HMMT-2025에서 86.72, CodeForces에서 2088, ARC-AGI-v1에서 55.94의 우수한 성적을 기록합니다. 특히, IMO-2025에서 은메달 수준의 결과를 달성하여 뛰어난 추론 능력을 강조합니다. 이 모델의 1조 파라미터 MoE 모델을 공개하여 연구자들에게 최첨단의 사고 능력을 직접 접근할 기회를 제공합니다.



### Lyapunov-Aware Quantum-Inspired Reinforcement Learning for Continuous-Time Vehicle Control: A Feasibility Study (https://arxiv.org/abs/2510.18852)
Comments:
          7 pages, 4 figures, 20 equations, 3 appendices, 4 tables

- **What's New**: 이 논문에서는 연속 시간 차량 제어를 위한 양자 강화 학습(Quantum Reinforcement Learning, QRL) 프레임워크인 Lyapunov-Based Quantum Reinforcement Learning (LQRL)을 제안합니다. 이 접근법은 변분 양자 회로(Variational Quantum Circuits, VQCs)의 표현력을 Lyapunov 안정성 분석과 결합하여 동적 환경에서 안전한 의사결정을 보장합니다. 차량 종방향 제어 문제는 Lyapunov 안정성 제약 조건에 따라 제어 동작을 생성하는 양자 정책 네트워크를 사용하는 연속 상태 강화 학습 작업으로 공식화됩니다.

- **Technical Details**: 제안된 LQRL 프레임워크는 양자 정책 최적화 과정에 Lyapunov 안정성 분석을 직접 통합합니다. 이 시스템은 양자 정책 기울기가 Lyapunov 감소 영역에 있도록 제약하여 안전 운영을 유지하면서도 점진적으로 평형 상태로 수렴하도록 보장합니다. 연구의 이론적 기여는 Lyapunov 제어 이론과 양자 강화 학습 사이의 엄격한 연결고리를 제공하여 지속적인 시간 학습 시스템에 대한 안정성 보장을 수립하는 것입니다.

- **Performance Highlights**: LQRL은 안전한 차량 추종 거리, 제한된 가속도 및 에너지 효율적인 주행 프로필을 보장하여 종방향 차량 크루즈 제어 시스템에 유효성을 검증했습니다. 결과는 양자 강화 학습이 수렴을 가속화하면서 수학적으로 입증된 안전 보장을 유지할 수 있음을 보여줍니다. 이 연구는 자율 시스템 및 하이브리드 양자-고전적 최적화 도메인에서 입증 가능한 안전한 양자 제어를 위한 기초적인 단계를 제공합니다.



### DP$^2$O-SR: Direct Perceptual Preference Optimization for Real-World Image Super-Resolution (https://arxiv.org/abs/2510.18851)
Comments:
          Accept by NeurIPS 2025

- **What's New**: 이 논문은 기존의 T2I(diffusion) 모델의 불확실성을 활용하여 이미지 초해상도(Real-ISR) 작업의 성능을 향상시키는 새로운 프레임워크 DP²O-SR(Direct Perceptual Preference Optimization for Real-ISR)을 소개합니다. 전통적인 손실 함수와 달리, 인간의 지각 선호도를 반영하는 혼합 보상 신호를 통해 모델의 출력을 개선합니다. 또한, 다양한 출력의 퍼셉션을 통해 더 풍부한 비교 정보를 생성하여 훈련 효율을 높입니다.

- **Technical Details**: 이 프레임워크는 대규모 인간 선호 데이터에 대해 훈련된 이미지 품질 평가(IQA) 모델을 활용하여 구조적인 충실도와 자연스러운 외관을 동시에 고려하는 하이브리드 보상 신호를 구축합니다. 또한, 우리는 계층적 선호 최적화(Hierarchical Preference Optimization, HPO)를 제안하여 훈련 쌍의 가중치를 적응적으로 조절하고, 더 유의미한 신호에 집중하여 학습을 개선합니다. 다양한 모델에 대한 실험을 통해 하이브리드 보상 시스템이 출력의 질적 향상에 기여함을 입증했습니다.

- **Performance Highlights**: DP²O-SR은 실제 세계에서의 초해상도 평가 데이터셋에서 우수한 성과를 나타냅니다. 상대적으로 작은 확산 모델(ControlNet-SD2)과 더 큰 플로우 기반 모델(ControlNet-FLUX) 모두 DP²O-SR을 통한 초기 훈련 반복에서 상당한 퍼셉션 보상 향상을 달성했습니다. 특히, C-FLUX는 약 0.51에서 0.65로 향상되었고, C-SD2는 0.62로 빠르게 성장하여 초기 훈련에서 상위 2위의 성능을 기록했습니다.



### Towards Faithful and Controllable Personalization via Critique-Post-Edit Reinforcement Learning (https://arxiv.org/abs/2510.18849)
Comments:
          work in progress

- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)의 개인화 방법을 개선하기 위해 Critique-Post-Edit이라는 강력한 강화학습 프레임워크를 제안합니다. 기존의 감독학습(Supervised Fine-Tuning, SFT) 및 인간 피드백 기반 강화학습(Reinforcement Learning from Human Feedback, RLHF) 모델이 직면한 한계를 극복하기 위해, 주어진 피드백을 통해 출력 결과를 정제하는 기법을 도입하였습니다. 이 새로운 접근법은 이전 모델에 비해 더 신뢰할 수 있고 컨트롤 가능한 개인화를 가능하게 합니다.

- **Technical Details**: Critique-Post-Edit 프레임워크는 두 가지 주요 구성 요소로 이루어져 있습니다. 첫째, 개인화된 생성 보상 모델(Generative Reward Model, GRM)은 다차원 점수와 텍스트 비평을 제공하여 보상 해킹(reward hacking)을 저지합니다. 둘째, 비평 기반 수정 메커니즘(Critique-Post-Edit mechanism)으로 정책 모델은 자신의 출력을 수정하여 더 목표 지향적이고 효율적인 학습을 가능하게 합니다. 해당 시스템을 이용한 평가에서, 우리의 방법은 기존 PPO 모델을 뛰어넘는 결과를 보여준다는 것을 알 수 있습니다.

- **Performance Highlights**: 개인화된 Qwen2.5-7B 모델은 평균적으로 11% 향상된 승률을 기록하였으며, Qwen2.5-14B 모델은 GPT-4.1의 성능을 초월하는 성과를 보였습니다. 이는 우리의 접근 방식이 신뢰할 수 있는, 효율적이며 컨트롤 가능한 개인화로 이어진다는 것을 보여줍니다. 전반적으로, Critique-Post-Edit 프레임워크는 개인화의 실제적이고 확장 가능한 경로를 제공하여 언어 모델의 개인화 능력을 크게 향상시킵니다.



### Actor-Free Continuous Control via Structurally Maximizable Q-Functions (https://arxiv.org/abs/2510.18828)
Comments:
          39th Conference on Neural Information Processing Systems (NeurIPS 2025)

- **What's New**: 새롭게 제안된 Q3C 알고리즘은 연속 제어 문제를 위한 순수한 가치 기반(value-based) 프레임워크로, 기존의 actor-critic 방법을 대체할 수 있는 가능성을 보여줍니다. Q3C는 Q-함수의 구조적 극대화를 통해 효율적이고 안정적인 학습을 가능하게 하며, 특히 제한된 액션 공간에서의 성능 최적화를 이루었습니다. 본 연구는 기존 접근 방식의 한계를 극복하고, 더 나은 성능을 달성하기 위한 여러 가지 설계 혁신을 도입했습니다.

- **Technical Details**: Q3C 알고리즘은 Q-함수의 구조적 극대화를 통해 깊이 있는 학습(deep learning) 방법과 결합하여 복잡한 연속 액션 공간에서 극대값을 쉽게 탐색할 수 있도록 합니다. 이 알고리즘은 제어 포인트(control-points)를 사용하여 Q-함수의 근사치를 제공하여, 지속적인 Q-학습을 가능하게 합니다. 또한, 제어 포인트의 생성을 단순화하여 가치 추정을 복잡하게 하는 문제를 해결하는 개선된 모델 아키텍처를 개발하였습니다.

- **Performance Highlights**: 다양한 Gymnasium 작업에서 Q3C의 성능은 일반적인 결정론적 actor-critic 방법들과 동등하며 복잡한 환경에서 제어가 요구되는 제한된 액션 공간에서 안정적으로 기존 방법들을 초월합니다. 특히, 제어 포인트 기반의 Q-함수 근사기는 이전의 가치 기반 방법들이 겪었던 적합성 문제를 크게 개선함으로써, 목표 액션의 정확한 극대 값을 찾는 능력을 강화했습니다. 추가로, 제거 연구(ablation study)를 통해 Q3C의 각 설계 요소가 성능에 미치는 영향을 시각적으로 측정하고 분석했습니다.



### An Explainable Hybrid AI Framework for Enhanced Tuberculosis and Symptom Detection (https://arxiv.org/abs/2510.18819)
Comments:
          16 pages, 3 figures

- **What's New**: 이 연구는 인공지능(AI) 기반의 결핵 스크리닝 도구의 필요성을 강조하며, 조기 발견을 통해 치료의 성공률을 높일 수 있음을 보여줍니다. 저자는 자기 지도학습(self-supervised learning)과 감독학습(supervised learning)을 통합한 티처-스튜던트(Tteacher-student) 프레임워크를 개발하여 흉부 X-선에서 질병과 증상을 효과적으로 탐지하는 방법을 제안합니다. 이 모델은 COVID-19, 결핵, 정상 사례를 구별하는 데 98.85%의 정확도를 기록했으며, 다중 증상 탐지에 대한 매크로 F1 점수는 90.09%로 기존의 기법보다 월등히 높은 성능을 보였습니다.

- **Technical Details**: 연구팀은 U-Net 아키텍처를 통해 질병과 증상을 식별하기 위해 데이터를 처리합니다. 이 과정에서 이미지를 8비트 그레이스케일 포맷으로 변환하고, 품질 관리를 통해 품질이 보장된 세분화(mask) 이미지를 사용합니다. 제안된 모델은 ViT-Small 티처–스튜던트 네트워크 아키텍처를 사용하며, 자기 지도학습 목표를 위해 DINO 기반의 프로젝션 헤드를 포함합니다.

- **Performance Highlights**: 제안된 모델은 튜불라 또는 COVID-19 증상이 있는 데이터를 포함한 다양한 데이터셋에서 강력한 CNN 기법을 초월하는 성능을 기록했습니다. 특히, 섬세한 발견들에 대한 증상 탐지에서 상당한 향상이 있어, 임상에서의 실제 스크리닝 및 분류 작업에 실질적으로 기여할 수 있는 가능성을 보여줍니다. Grad-CAM 시각화를 통한 분석도 모델의 결정적 장치를 해부학적 특징과 일치하게 하는 강점을 나타냅니다.



### Fine-Tuned Thoughts: Leveraging Chain-of-Thought Reasoning for Industrial Asset Health Monitoring (https://arxiv.org/abs/2510.18817)
Comments:
          Accepted at EMNLP 2025

- **What's New**: 이 논문은 산업 애셋 헬스(Industrial Asset Health) 분야에 대한 지식 증류(knowledge distillation) 프레임워크를 제안합니다. 이 프레임워크는 대형 언어 모델(Large Language Models, LLMs)에서 소형 언어 모델(Small Language Models, SLMs)로 체인 오브 사고(Chain-of-Thought, CoT) 추론 능력을 전달합니다. 이를 통해 복잡한 추론 과제가 있는 산업 분야에서도 SLM의 성능을 향상시키려는 노력이 이루어집니다.

- **Technical Details**: 제안된 방법론은 전통적인 FMEA(고장 모드 및 영향 분석) 지식을 LLM에서 SLM으로 전이하는 구조를 채택합니다. 이 과정에서는 멀티 초이스 질문(MCQA) 프롬프트를 사용하여, 초기 데이터셋 없이 합성 데이터를 생성하는 단계를 포함합니다. 또한, 지식 그래프(Knowledge Graph)를 활용해 산업 도메인 지식을 조직하고, 각 요소 간의 관계를 정의하는 방법을 제시합니다.

- **Performance Highlights**: 실험 결과, CoT 추론을 적용한 SLM들은 기초 모델에 비해 11%에서 23%까지 성능이 향상되었습니다. 이는 SLM이 LLM과의 격차를 줄이는 데 있어 매우 성공적인 결과임을 보여줍니다. 최종적으로, 이 연구 결과는 산업 자산 건강 모니터링 애플리케이션에서 SLM의 효율성을 강조합니다.



### Online SFT for LLM Reasoning: Surprising Effectiveness of Self-Tuning without Rewards (https://arxiv.org/abs/2510.18814)
- **What's New**: 이번 논문에서는 LLM(대형 언어 모델) 추론을 위한 간단하고 자가 도움(self-help) 온라인 감독 세부 조정(OSFT) 패러다임을 제안합니다. OSFT는 모델이 스스로 생성한 데이터에 즉시 세부 조정을 진행하여 자가 학습을 촉진하는 효율적인 학습 전략입니다. 실험 결과, OSFT는 GRPO와 같은 강력한 강화 학습(RLVR) 방법과 비교했을 때, 복잡한 수학적 추론 작업에서 우수한 성능을 나타냅니다.

- **Technical Details**: OSFT는 기존의 선행 학습(pretraining)에서 얻은 모델의 선호(latent knowledge)를 강화하여 추론 능력을 향상시키는 기제를 포함합니다. 이 패러다임은 기본적으로 하나의 롤아웃(rollout)만을 사용하는 보상 없는(reward-free) 학습 알고리즘으로, 효율성을 극대화합니다. LLM은 입력에 대한 토큰 시퀀스를 생성하는 자기 회귀 모델(autoregressive model)로, 이 과정에서 다양한 훈련 기법과 수학적 최적화 알고리즘이 활용됩니다.

- **Performance Highlights**: 실험 결과, OSFT는 수학적 추론 벤치마크에서 주목할만한 성능 향상을 보여주었으며, 이는 GRPO와 비교했을 때도 여전히 경쟁력을 갖춥니다. OSFT의 효율성과 견고성은 다양한 실험을 통해 확인되었으며, 이를 통해 더욱 복잡한 보상 기반 훈련 패러다임에 대한 유망한 대안이 될 수 있음을 주장합니다. 또한, OSFT는 사용하기 간편하면서도 강력한 성능을 발휘할 수 있습니다.



### Computational Foundations for Strategic Coopetition: Formalizing Interdependence and Complementarity (https://arxiv.org/abs/2510.18802)
Comments:
          36 pages, 7 figures

- **What's New**: 본 기술 보고서는 현대의 사회-기술적 시스템에서 전략적 협력 경쟁(coopetition)의 두 가지 핵심 차원인 상호의존성(interdependence)과 상호 보완성(complementarity)을 수량적으로 분석할 수 있는 기반을 제공합니다. 이를 통해 기존의 i*와 같은 개념적 모델링 언어의 한계를 극복하고, 게임 이론의 엄격함을 바탕으로 한 실질적인 해법을 제시합니다. 또한, Samsung과 Sony의 S-LCD 합작 투자 사례를 사용하여 이론과 실제의 접목을 시도합니다.

- **Technical Details**: 상호의존성은 i* 구조적 의존성 분석을 기반으로 하여, 의존하는 주체(depender)와 의존받는 주체(dependee), 그리고 의존 항목(dependum)의 관계를 수량적으로 표현하는 체계적인 변환 프레임워크를 통해 정립됩니다. 상호 보완성은 Brandenburger와 Nalebuff의 추가 가치(Added Value) 개념을 통해 공식화되며, 이는 시너지 효과를 통한 가치 창출을 모델링합니다. 이 색다른 접근은 게임 이론적 수식화와 Nash Equilibrium을 통합하여 구조적 상호의존성을 포함한 가치 획득을 설명합니다.

- **Performance Highlights**: 보고서에서는 전반적인 실험 테스트를 통해 파워 및 로그 값 함수 사양에서 기능적 형태의 강건성을 입증하였습니다. 특히, 로그 사양이 더 높은 실증적 적합도(validity score 45/60)를 기록하여 이론적 유연성과 실질적인 적용 가능성을 동시에 보여줍니다. 이 연구는 요구사항 공학(requirements engineering) 및 다중 에이전트 시스템(multi-agent systems)에서 전략적 협력 경쟁을 조사하는 통합 연구 프로그램의 기초 자료로서 활용될 수 있습니다.



### Verifiable Accuracy and Abstention Rewards in Curriculum RL to Alleviate Lost-in-Conversation (https://arxiv.org/abs/2510.18731)
- **What's New**: 이번 연구에서는 멀티 턴 대화에서 발생하는 성능 저하 현상인 'Lost-in-Conversation'(LiC) 문제를 해결하기 위한 새로운 프레임워크인 RLAAR을 제안합니다. 이는 강화 학습(Reinforcement Learning) 기법을 활용하여 모델이 단순히 올바른 답변을 생성하는 것뿐만 아니라 질문의 해결 가능성을 판단할 수 있도록 돕습니다. RLAAR은 점진적으로 대화의 난이도를 증가시키는 커리큘럼(curriculum)을 통해 훈련 안정성을 유지하면서 신뢰성 있는 모델 구축을 목표로 합니다.

- **Technical Details**: RLAAR은 강화 학습을 기반으로 하며, 다회적 롤아웃(on-policy rollouts)과 혼합 보상 체계를 통해 모델이 문제 해결과 정보 수집 사이의 균형을 유지하도록 교육합니다. 사용자는 모델의 응답이 다음 턴의 상태가 되도록 하여 대화의 동적 특성을 고려합니다. 또한, '정보가 부족함을 인정하는' 움직임을 통해 모델이 조기 답변을 피하도록 지원하는 생략 보상(abstention rewards)을 포함하고 있습니다.

- **Performance Highlights**: RLAAR은 LiC 벤치마크에서 성능 저하를 62.6%에서 75.1%로 크게 완화하고, 보정된 생략률(calibrated abstention rates)을 33.5%에서 73.4%로 향상시킵니다. 이러한 결과들은 멀티 턴 환경에서 신뢰성 있고 신뢰할 수 있는 LLM 구축을 위한 실용적인 방법론을 제공합니다. 연구에서는 제안된 방법이 기존 모델 및 최신 모델들과 비교하여 현저한 성능 개선을 이룬다는 점을 강조하고 있습니다.



### HarmNet: A Framework for Adaptive Multi-Turn Jailbreak Attacks on Large Language Models (https://arxiv.org/abs/2510.18728)
Comments:
          This paper has been accepted for presentation at the Conference on Applied Machine Learning in Information Security (CAMLIS 2025)

- **What's New**: HarmNet은 다중 턴의 jailbreak 공격에 대한 강력한 해결책으로 소개됩니다. 기존의 방법들이 제한된 부분을 탐구하거나 수작업으로 작성된 휴리스틱에 의존했던 것에 비해, HarmNet은 구조화된 탐색과 피드백 기반 개선을 통해 공격의 성공률을 획기적으로 높입니다. 실험 결과, Mistral-7B 모델에서는 99.4%의 웹 공격 성공률을 기록하였으며, 이는 최고의 기존 방법보다 13.9% 더 높은 수치입니다.

- **Technical Details**: HarmNet은 ThoughtNet, Simulator, Network Traverser의 세 가지 주요 모듈로 구성됩니다. ThoughtNet은 계층적 의미 네트워크를 통해 적대적 경로를 탐색하고, Simulator는 유해성과 의미 적합성에 기반하여 질의를 점진적으로 개선합니다. 마지막으로 Network Traverser는 실시간으로 최적의 다중 턴 공격 체인을 선택하고 실행하여, 신속하고 효율적인 공격을 가능하게 합니다.

- **Performance Highlights**: HarmNet은 비교 연구를 통해 다양한 모델에서 우수한 성능을 보였습니다. GPT-3.5-Turbo에서 91.5%, GPT-4o에서 94.8%의 성공률을 달성하며, 오픈 소스 모델인 LLaMA-3-8B와 Mistral-7B에서도 각각 98.4%와 99.4%의 놀라운 결과를 기록했습니다. 이러한 결과는 HarmNet의 구조적 탐색과 피드백 기반 조정이 지속적이고 상당한 성장을 가능하게 한다는 것을 보여줍니다.



### Causally Perturbed Fairness Testing (https://arxiv.org/abs/2510.18719)
Comments:
          accepted by TOSEM

- **What's New**: CausalFT는 AI 시스템에서 불공정한 차별을 탐지하는 새로운 프레임워크입니다. 이 방법은 민감(Sensitive) 특성과 비민감(Non-Sensitive) 특성 간의 인과관계를 활용하여 공정성 결함(fairness bugs)을 보다 효과적으로 찾아냅니다. 이는 기존의 무작위 변조(random perturbation) 방식과 달리, 데이터 특성을 기반으로 테스트 샘플 생성을 안내합니다.

- **Technical Details**: CausalFT는 인과 추론(causal inference)의 개념을 기반으로 하여 트레이닝 데이터에서 인과 그래프(causal graph)를 구축합니다. 이 과정에서 민감 특성과 가장 직접적으로 관련된 비민감 특성을 추출한 후 이를 변조에 주입해서 테스트 샘플 생성을 안내합니다. 이 프레임워크는 기존의 생성기(generator)와 독립적으로 운영되며, 다양한 테스트 샘플 생성기와 결합되어 사용할 수 있습니다.

- **Performance Highlights**: CausalFT는 1296개의 실험 케이스에서 93%의 경우에 공정성 결함을 밝혀내는 데 성공했습니다. 또한, CausalFT를 사용한 경우 기존의 상관관계에 기초한 비민감 특성과 비교했을 때 64%의 경우에서 더 좋은 성능을 보였습니다. 이로 인해 AI 시스템의 바이어스 저항성(bias resilience)이 거의 모든 경우에서 개선된 것으로 나타났습니다.



### Preference-based Reinforcement Learning beyond Pairwise Comparisons: Benefits of Multiple Options (https://arxiv.org/abs/2510.18713)
Comments:
          Accepted at NeurIPS 2025

- **What's New**: 이번 연구에서는 온라인 선호 기반 강화 학습(Preference-based Reinforcement Learning, PbRL)의 샘플 효율성을 개선하는 것을 목표로 하였습니다. 기존 연구들은 주로 쌍 비교(pairwise comparisons)에 집중하였으나, 본 논문은 차별화된 여러 비교와 순위 피드백을 사용하는 M-AUPO 알고리즘을 제안합니다. 특히, 이 알고리즘은 제공된 서브셋 내에서 평균 불확실성(average uncertainty)을 극대화하여 동작을 선택합니다.

- **Technical Details**: M-AUPO는 Plackett-Luce (PL) 모델을 채택하여 행동 서브셋의 순위를 매기고, 이로부터 얻은 새로운 이론적 결과는 샘플 효율성이 서브셋 크기의 함수로 개선됨을 보여줍니다. 제시된 성과는 $	ilde{	ext{O}}ig( rac{d}{T} 	imes 	ext{sqrt} ig( 	ext{sum}_{t=1}^T rac{1}{|S_t|} ig) ig)$의 비최적성 갭을 달성하며, 서브셋의 크기가 증가함에 따라 성능이 향상된다는 점이 특징입니다. 이론적으로 이는 대부분의 이전 연구에서의 한계를 극복하는 것입니다.

- **Performance Highlights**: M-AUPO 알고리즘은 더 큰 서브셋을 이용하여 향상된 성능을 달성하며, 불확실성 관점에서도 최적의 결과를 이끌어냅니다. 또한, 데이터에 대한 의존성이 대폭 줄어들어 비교적 적은 데이터로도 효과적인 작업이 가능합니다. 이러한 개선점은 PbRL 분야에서의 이론적 진전에 기여하며, 향후 연구 방향에 도움을 줄 것으로 기대됩니다.



### Fetch.ai: An Architecture for Modern Multi-Agent Systems (https://arxiv.org/abs/2510.18699)
Comments:
          26 pages, figures, code examples

- **What's New**: 이 논문은 최근의 LLM(대형 언어 모델) 기반 지능형 시스템이 다수의 매개체 시스템(Multi-Agent System, MAS) 연구의 기초를 대체로 간과하고 있음을 지적합니다. 이에 따른 한계를 극복하기 위해, 본 논문은 전통적인 MAS 원칙과 현대 AI 기능을 통합하는 새로운 아키텍처를 소개합니다. 이 구조는 탈중앙화된 블록체인 서비스를 기반으로 하여 신원 검증, 정보 발견 및 거래를 지원하는 다중 계층 솔루션을 제공합니다.

- **Technical Details**: 제안된 아키텍처는 보안 및 상호 운용 가능한 에이전트를 구축하기 위한 포괄적인 개발 프레임워크를 제공합니다. 클라우드 기반 플랫폼을 통해 배포가 가능하며, 에이전트 네이티브 LLM이 고수준의 인간 목표를 복잡한 다중 에이전트 작업 흐름으로 번역해주는 지능형 오케스트레이션 레이어도 포함되어 있습니다. 이러한 기능들은 안전한 거래와 상호작용을 가능하게 해 줍니다.

- **Performance Highlights**: 이 시스템은 자율 에이전트들이 동적으로 서로를 발견하고 협상하며 안전하게 거래하는 분산 물류 사용 사례를 통해 그 가치를 실증합니다. 궁극적으로, 이 아키텍처는 기존 에이전트 구현을 넘어 협력적이고 경제적으로 지속 가능한 다중 에이전트 생태계로 나아가기 위한 원칙적인 구조를 제공합니다.



### Exploring Membership Inference Vulnerabilities in Clinical Large Language Models (https://arxiv.org/abs/2510.18674)
Comments:
          Accepted at the 1st IEEE Workshop on Healthcare and Medical Device Security, Privacy, Resilience, and Trust (IEEE HMD-SPiRiT)

- **What's New**: 이 연구는 의료 분야에서의 대형 언어 모델(LLM)의 프라이버시 문제를 중점적으로 다룹니다. 특히, 임상 LLM이 특정 환자 기록이 모델 훈련에 사용되었는지 여부를 추정할 수 있는 회원 추론 공격(Membership Inference Attacks, MIAs)에 대해 탐구합니다. 이 과정에서 Llemr이라는 최첨단 임상 질문-응답 모델을 사용하여 기존의 공격 방법과는 차별화된 새로운 패러프레이즈 기반 전술을 도입합니다.

- **Technical Details**: 연구는 LLM의 회원 추론 공격에 대한 취약성을 규명하며, 전통적인 방식을 넘어서는 패러프레이즈 기반의 방어 전략을 제시합니다. 훈련 예제와 비훈련 예제 간의 행동 차이를 측정하여 모델의 성능을 분석합니다. 특히, EHR 데이터로 추가 훈련된 LLM이 미세한 프라이버시 리키지를 보인다는 초기 결과를 도출했습니다.

- **Performance Highlights**: 결과적으로, 현재의 임상 LLM이 부분적인 저항성을 제공하지만 여전히 사소한 프라이버시 위험에 취약하다는 점이 강조됩니다. 이는 의료 AI의 신뢰성을 저하할 수 있는 요소로 작용할 수 있어, 차별적 프라이버시 미세조정 및 패러프레이즈 인지 훈련과 같은 문맥 인식의 도메인별 프라이버시 평가와 방어의 지속적인 개발이 필요함을 시사합니다.



### Reasoning Language Model Inference Serving Unveiled: An Empirical Study (https://arxiv.org/abs/2510.18672)
- **What's New**: 이번 논문에서는 RLLM(Reasoning Large Language Model)의 서빙 성능과 행동에 대한 포괄적인 연구를 수행하여 기존 LLM(일반 Large Language Model)과의 차이점을 밝혀냈습니다. RLLM는 복잡한 추론 작업에서 유리한 성능을 보여주지만, 실제 환경에서의 배포 및 활용은 미비한 상태입니다. 연구 결과, RLLM은 메모리 사용량의 불규칙성과 요청 지연, 적응형 실행 시간, 도메인 선호도에서 뚜렷한 차이를 보임을 확인했습니다.

- **Technical Details**: RLLM에 대한 효율적인 서빙 기법을 연구하기 위해 ASU 평가 프레임워크를 설정하고 ASU-Perf 벤치마크 스위트를 설계했습니다. RLLM의 서빙 행동을 여러 측면에서 구체적으로 분석했으며, 이는 메모리 플럭투에이션, 요청 실행 시간의 긴 꼬리 분포, 문제의 난이도에 따른 적응형 실행 시간을 포함합니다. RLLM의 서빙 전반에 걸쳐 모델 양자화 및 추측 디코딩 기법이 효율성을 개선하면서도 정확도에 소규모의 타협을 가져올 수 있음을 밝혔습니다.

- **Performance Highlights**: 현실 세계의 워크로드에서 RLLM의 서빙 동작은 LLM과 다르며, 다양한 데이터셋에 대한 실험 결과는 이러한 발견을 지지합니다. RLLM이 수학적 추론에서 LLM을 초과하는 성능을 보여주었고, 지식 집약적 작업에서는 동등한 성능을 발휘함을 입증했습니다. 이러한 연구 결과는 RLLM의 효율적인 서빙을 증진시키는 데 있어 연구 공동체와 산업에 유용한 통찰을 제공할 것입니다.



### Binary Quadratic Quantization: Beyond First-Order Quantization for Real-Valued Matrix Compression (https://arxiv.org/abs/2510.18650)
Comments:
          Accepted to NeurIPS 2025

- **What's New**: 이 논문은 새로운 행렬 양자화 방법인 Binary Quadratic Quantization (BQQ)를 제안합니다. 전통적인 첫 번째 차수 양자화 접근 방식인 균일 양자화(Uniform Quantization)와 이진 코딩 양자화(Binary Coding Quantization)와는 달리, BQQ는 이진 이차 표현의 표현력을 활용하며 매우 압축된 데이터 형식을 유지합니다. 이 접근 방식은 다양한 행렬 데이터의 압축에 있어 메모리 효율성과 복원 오류 간의 우수한 균형을 달성합니다.

- **Technical Details**: BQQ는 이진 변수의 이차 조합을 사용하여 행렬을 표현하며, 각 이진 행렬에 대해 독립적인 스케일링(스케일링) 계수를 할당합니다. 이는 이진 행렬 곱의 합으로 타겟 행렬을 표현하므로 비선형 근사를 가능하게 하며 매우 컴팩트한 데이터 형식을 유지할 수 있습니다. BQQ는 NP-하드 최적화 문제를 다루며, 이를 풀기 위한 효율적인 솔루션을 다항 비제한 이진 최적화(Polynomial Unconstrained Binary Optimization) 및 볼록 이차 프로그래밍(Convex Quadratic Programming)을 기반으로 개발하였습니다.

- **Performance Highlights**: 실험 결과, BQQ는 다양한 행렬 데이터의 압축에 있어 메모리 사용과 양자화 오류 간의 우수한 균형을 달성하며, Vision Transformer(ViT) 기반 모델의 후처리 양자화(PTQ)에서 최첨단 성능을 제공합니다. 예를 들어, BQQ는 ImageNet 데이터셋에서 캘리브레이션 기반 및 데이터 없음 시나리오에서 각각 최대 2.2% 및 59.1% 성능 향상을 보였습니다. 이러한 결과는 BQQ가 효율적인 행렬 근사 및 신경망 압축에 매우 효과적임을 강조합니다.



### ε-Seg: Sparsely Supervised Semantic Segmentation of Microscopy Data (https://arxiv.org/abs/2510.18637)
Comments:
          10 pages main text, 17 pages total

- **What's New**: 새로운 연구에서, 우리는 생물학적 샘플의 전자 현미경(EM) 이미지를 위한 희소 감독(semi-supervised) 의미 분할 방법인 {m{m{	ext{epsilon}}}-Seg를 소개합니다. 이 방법은 계층적 변별 오토인코더(HVAE)를 기반으로 하며, 중심 영역 마스킹(center-region masking)과 희소 레이블 대비 학습(sparse label contrastive learning)을 도입하여 효과적인 세분화를 가능하게 합니다. {m{m{	ext{epsilon}}}-Seg는 복잡한 생물학적 이미지 데이터에서 경쟁력 있는 결과를 얻는 데 도움을 줍니다.

- **Technical Details**: 제안된 방법은 각 의미 클래스에 대해 미리 정해진 가우시안 영역을 가지는 가우시안 혼합 모델(GMM)을 사용하며, 이는 계층적 VAE(HVAE) 구조의 변형입니다. 또한, 대조 손실(contrastive loss)을 추가하여 잠재 인코딩(latent encoding)이 의미 유사성에 따라 그룹화되도록 합니다. 최종적으로, 기존 클러스터링 대신 MLP 의미 분할 헤드를 사용하여 잠재 인코딩에서 직접 클래스 레이블을 예측하는 방식으로 정확도와 실행 시간을 개선합니다.

- **Performance Highlights**: 실험 결과, {m{m{	ext{epsilon}}}-Seg는 제한된 라벨(0.05% 이하)로도 경쟁력 있는 분할 성능을 보여 주며, 생물학적 조직의 두 가지 고밀도 EM 데이터셋에서 우수한 결과를 얻었습니다. 또한, 이 방법은 형광 현미경 데이터에도 적용 가능하다는 것을 입증하며, 기존 방법들보다 좋은 성능을 발휘합니다. 이 연구는 EM 이미지 세분화의 새로운 가능성을 제시합니다.



### C-SWAP: Explainability-Aware Structured Pruning for Efficient Neural Networks Compression (https://arxiv.org/abs/2510.18636)
Comments:
          10 pages, BMVC2025

- **What's New**: 본 논문에서는 고전적인 프루닝(pruning) 기법을 개선한 새로운 일회성 구조 프루닝(framework이자 C-SWAP)을 제안하고 있습니다. 이 방법은 설명 가능한 AI(explainable AI)를 사용하여 모델의 성능을 보존하면서 파라미터 수를 크게 줄일 수 있도록 돕습니다. 특히, 모델의 예측과 구조 간의 인과적 관계를 활용하여 효율적인 프루닝을 수행함으로써, 더 적은 리소스로도 높은 성능을 유지할 수 있습니다.

- **Technical Details**: C-SWAP는 자연스러운 기계 해석(auto interpretability) 연구에 영감을 받아 설계되었습니다. 이 방법은 각 뉴런(채널)을 중요, 중립, 해로운 세 가지 등급으로 분류하고 해당 뉴런과 연관된 가중치를 변경하여 원인 효과(causal effect)를 계산합니다. 이를 통해 통계적 임계값을 사용하여 프루닝할 뉴런을 선택하고, 성능 손실을 최소화하는 방향으로 진행됩니다.

- **Performance Highlights**: 실험을 통해 C-SWAP가 여러 기본 프루닝 기법들보다 뛰어난 성능을 발휘하는 것을 보여주었습니다. 특히, CNN 및 비전 트랜스포머 모델에서의 성능 저하 없이 모델 크기를 효과적으로 줄일 수 있었습니다. 추가적인 성능 조정 없이도 다양한 복잡한 아키텍처에 적용 가능한 가능성을 가지고 있습니다.



### Think with 3D: Geometric Imagination Grounded Spatial Reasoning from Limited Views (https://arxiv.org/abs/2510.18632)
Comments:
          12 pages, 4 figures

- **What's New**: 이 논문에서는 3D 공간 관계 이해의 어려움을 해결하기 위해 새로운 프레임워크인 3DThinker를 제안합니다. 이는 VLM (Vision-Language Model)에서 3D mentaling을 사용하여 제한된 2D 이미지를 기반으로 3D 기하학을 직접 학습할 수 있게끔 합니다. 3DThinker는 복잡한 데이터 레이블이나 외부 모델에 의존하지 않고도 3D 표현을 통합하여 사고할 수 있는 능력을 제공합니다.

- **Technical Details**: 3DThinker의 훈련 과정은 두 단계로 구성되어 있습니다. 첫 번째 단계에서는 VLM이 생성한 3D 잠재 표현을 3D 기본 모델과 정렬하는 감독 학습(Supervised Learning)을 수행합니다. 두 번째 단계에서 우리는 보상 기반 신호에 따라 전체 샘플링 경로를 최적화하여 3D mentaling을 정제하는 강화 학습(Reinforcement Learning)을 진행합니다.

- **Performance Highlights**: 다양한 벤치마크에서의 상세 실험 결과, 3DThinker는 강력한 기준선 모델들보다 일관되게 우수한 성능을 보였습니다. 이 모델은 다양한 VLM 기반에서 잘 일반화되며, 3D 표현을 해석할 수 있는 새로운 관점을 제시하였습니다.



### A Rectification-Based Approach for Distilling Boosted Trees into Decision Trees (https://arxiv.org/abs/2510.18615)
Comments:
          29 pages

- **What's New**: 이번 연구에서는 boosted tree를 decision tree로 변환하는 새로운 접근법을 제시합니다. 이 접근법은 예측 성능과 해석 가능성 간의 절충을 제공하는 ML 모델을 생성하는 데 초점을 맞추고 있습니다. 특히, rectification(수정)이라는 방법론을 통해 이 증류(distillation) 과정을 구현할 수 있는 방법을 설명합니다.

- **Technical Details**: 우리는 incrementally(점진적으로) 개선하는 증류 과정을 집중적으로 다룹니다. 초기의 decision tree(II)는 해석이 가능하지만 정확성이 떨어지고, boosted tree(PP)는 정확성은 높지만 해석 가능성이 떨어집니다. 본 연구의 목표는 II를 PP에 논리적으로 가깝게 수정하는 것으로, 각 수정 단계에서 II의 예측 성능을 개선하는 것입니다.

- **Performance Highlights**: 실험 결과, rectification 기반 증류 접근법이 retraining(재학습) 방법에 비해 우수한 성능을 보여주었습니다. rectification은 효과적인 수정(targeted corrections)을 보장할 수 있는 논리적 장점을 제공합니다. 또한 PP에서 II로의 증류 과정이 computation(계산)이 유용하다는 것을 나타내는 실증적 증거도 제공되었습니다.



### The Cost-Benefit of Interdisciplinarity in AI for Mental Health (https://arxiv.org/abs/2510.18581)
Comments:
          Accepted for poster presentation at the AI in Science Summit 2025

- **What's New**: 이 논문은 인공지능(AI) 정신 건강 챗봇의 생애 주기 전반에 걸쳐 기술, 건강 관리, 윤리 및 법률 전문가들의 협력의 필요성을 강조합니다. 특히 챗봇의 각 주요 단계에서 이러한 전문가들의 참여가 필수적이며, EU AI 법의 높은 리스크 요구 사항에 부합하는 가치 정렬과 규정 준수를 보장하는 데 필요하다고 주장합니다. 이 연구는 여러 분야의 통합적 접근법이 챗봇의 효과성을 높일 수 있음을 보여줍니다.

- **Technical Details**: AI 정신 건강 챗봇 개발에는 건강, 심리학, 정신의학, 인간-컴퓨터 상호작용(HCI), 컴퓨터 과학, 윤리 및 법률과 같은 다양한 분야의 전문 지식이 요구됩니다. 연구 결과에 따르면, 현재 정신 건강 챗봇의 19.6%만이 다학제 팀을 채택하는데, 이는 효과적인 협업의 필요성을 강조합니다. 다양한 전문가가 각 생애 주기에 맞춰 협력함으로써 챗봇의 프라이버시, 안전, 책임 문제를 해결할 수 있습니다.

- **Performance Highlights**: 기술, 건강 관리, 윤리, 법률 전문가의 협업이 이루어질 경우, 챗봇의 개발 진행 상황이 개선되고, 사용자 만족도와 위험 관리가 향상될 수 있습니다. 다만, 다학제 협업에는 기대의 불일치, 전문 용어의 차이, 제한된 재정 지원 등의 도전 과제가 동반됩니다. 이러한 점을 고려하여, 다학제적 접근 방식을 통해 챗봇의 임팩트를 극대화할 수 있는 최선의 방법이 논의되고 있습니다.



### Kaleido: Open-Sourced Multi-Subject Reference Video Generation Mod (https://arxiv.org/abs/2510.18573)
Comments:
          11 pages, 6 figures

- **What's New**: Kaleido는 여러 참조 이미지에 기반하여 주제와 일치하는 비디오를 생성하는 S2V(Subject-to-Video) 생성 프레임워크를 제안합니다. 기존의 S2V 모델들은 다중 주제의 일관성을 유지하고 배경을 효과적으로 분리하는 데 부족함이 있었으나, Kaleido는 이러한 문제를 해결하기 위해 개선된 데이터 구성 및 참조 이미지 통합 방법을 도입했습니다. Reference Rotary Positional Encoding(R-RoPE) 기술을 통해 다양한 참조 이미지를 안정적으로 처리하고 일관성을 높였습니다.

- **Technical Details**: Kaleido의 데이터 구성 파이프라인은 저품질 샘플 필터링 및 다양한 데이터 합성을 포함하여 일관성을 유지하는 훈련 데이터를 생성합니다. R-RoPE는 주제 토큰에 회전 위치 인코딩(rotary position encoding)을 도입하여 다수의 참조 이미지로부터 효율적으로 정보를 통합할 수 있도록 설계되었습니다. 이로 인해 다중 이미지 및 다중 주제에 대한 S2V 일관성이 개선되고 컴퓨테이셔널 효율성도 보장됩니다.

- **Performance Highlights**: Kaleido는 다양한 벤치마크에서 기존 방법보다 뛰어난 성능을 보여주었으며, 주제 일관성(subject fidelity)과 배경 분리(background disentanglement)에서 우수함을 입증했습니다. 이러한 성과는 특히 상업적 시스템인 Vidu와 Kling과 같은 비공식적 모델과 비교할 때 두드러지며, 앞으로 S2V 생성 모델의 발전에 큰 기여할 것으로 기대됩니다.



### Large language models for folktale type automation based on motifs: Cinderella case study (https://arxiv.org/abs/2510.18561)
- **What's New**: 이번 논문에서는 디지털 인문학을 포함한 여러 연구 분야에서 인공지능 접근법을 활용하는 새로운 방법론을 제안합니다. 우리는 특히 전래 동화인 신데렐라의 다양한 변형을 대규모로 분석하기 위한 방법을 개발했습니다. 이 연구는 머신 러닝과 자연어 처리(Natural Language Processing)를 활용하여 꿈의 모티프(motif)를 자동으로 탐지하는 방식으로 진행되었습니다.

- **Technical Details**: 연구에서는 여러 기법을 통해 대량의 문자 데이터 집합에서 유사성과 차이를 분석하기 위해 클러스터링(clustering)과 차원 축소(dimensionality reduction) 기술을 사용하였습니다. 이를 통해 대규모 언어 모델(large language models)이 이야기 속 복잡한 상호작용을 탐지할 수 있음을 보여주고, 동시에 방대한 텍스트 집합에 대한 계산 분석(computational analysis)을 가능하게 합니다.

- **Performance Highlights**: 연구 결과는 신데렐라 변형 집합의 모티프를 탐지하고 분석하는 데 있어 뛰어난 성능을 보였습니다. 이러한 방법론은 다양한 언어 간의 비교(cross-lingual comparisons)를 촉진하여, 인문학 연구에 기여할 수 있는 새롭고 혁신적인 분석 도구로 자리잡을 것으로 기대됩니다.



### WebDevJudge: Evaluating (M)LLMs as Critiques for Web Development Quality (https://arxiv.org/abs/2510.18560)
- **What's New**: 이 논문은 LLM(as-a-judge) 평가의 신뢰성을 높이기 위해 WebDevJudge라는 새로운 메타 평가 기준을 제안합니다. 기존의 평가 방식은 인간의 판단에 의존하여 비효율적이며, 복잡한 상호작용이 있는 동적 환경에서의 LLM 성능을 검증할 필요가 있었습니다. 이 새로운 기준은 웹 개발이라는 복잡한 작업을 통해 LLM의 진화 가능성을 탐구하고 있으며, 정적 작업과 대화형 작업의 평가를 지원합니다.

- **Technical Details**: WebDevJudge는 웹 구현에 대한 인간 선호 레이블과 구문 및 쿼리 기반 루브릭을 통해 고품질의 그라운드 트루스를 제공합니다. 평가 방법론은 정적 코드, 렌더링된 웹 페이지의 스크린샷, 그리고 동적 평가를 위한 전면적 인터랙티브 환경을 포함하여 다면적으로 프레젠테이션을 제공합니다. 이는 LLM, MLLM 및 에이전틱 워크플로우와 같은 다양한 평가자들의 성능을 분석하고, 이들 간의 성능 갭을 정량적으로 평가하는 기반을 마련합니다.

- **Performance Highlights**: WebDevJudge를 통해 수행된 실험에서는 현재 LLM(as-a-judge) 접근 방식이 여전히 인간 전문가의 신뢰성에 미치지 못하고 있음을 확인했습니다. LLM 심판과 인간 전문가 간의 성능 차이는 15% 이상에 이르며, 기존의 가이던스 전략이 이 차이를 크게 줄이지 못하는 것으로 나타났습니다. 오류 분석 결과, LLM이 기능 동등성을 인지하지 못하고, 작업의 실행 가능성을 검증하는데에서 한계를 나타내며, 복잡한 상호작용 환경에서 더욱 신뢰할 수 있는 자동 평가자를 개발하라는 방향성을 제시합니다.



### RAISE: A Unified Framework for Responsible AI Scoring and Evaluation (https://arxiv.org/abs/2510.18559)
Comments:
          Accepted at the 26th International Conference on Principles and Practice of Multi-Agent Systems

- **What's New**: 이번 연구에서는 AI 시스템이 고위험 분야에서 평가할 때 예측 정확도를 넘어 설명 가능성(explainability), 공정성(fairness), 견고함(robustness), 지속 가능성(sustainability)을 포함해야 한다고 강조합니다. RAISE(Responsible AI Scoring and Evaluation)라는 통합 프레임워크를 소개하여 모델의 성능을 네 가지 차원에서 정량적으로 측정하고 이를 종합하여 단일 책임 점수(Responsibility Score)를 생성합니다. 이를 통해 각 모델의 고유한 책임 프로필을 비교할 수 있는 방법론을 제시합니다.

- **Technical Details**: RAISE 프레임워크는 다차원적 시각에서 모델의 설명 가능성, 공정성, 지속 가능성 및 견고함을 정량화합니다. 기계 학습 모델을 평가하기 위해 세 가지 심층 학습 아키텍처(Multilayer Perceptron, Tabular ResNet, Transformer)를 특정 구조화된 데이터셋에서 평가했습니다. 모델 평가 시 21개의 정량적 메트릭을 사용하여 각 모델의 다양한 측면을 체계적이고 재현 가능한 방식으로 비교 분석하였습니다.

- **Performance Highlights**: 세 가지 심층 학습 모델을 평가한 결과, MLP는 지속 가능성과 견고성에서 강점을 보였고, Transformer는 설명 가능성과 공정성에서 우수한 성과를 보였으나 환경 비용이 높았습니다. Tabular ResNet은 세 가지 차원에서 균형 잡힌 성능을 보여주었으며, 전반적으로 모든 모델이 모든 책임 기준에서 우위를 점하지 않음을 발견하였습니다. 이는 책임있는 모델 선택을 위한 다차원 평가의 필요성을 강조합니다.



### EfficientNav: Towards On-Device Object-Goal Navigation with Navigation Map Caching and Retrieva (https://arxiv.org/abs/2510.18546)
Comments:
          NeurIPS 2025

- **What's New**: 이 논문에서는 제로샷(Zero-shot) 객체 목표 탐색(Object-goal navigation) 작업에서 작은 언어 모델(LLMs) 기반의 효율적인 내장식 탐색 시스템인 EfficientNav를 제안합니다. 기존의 대형 LLMs에 의존하는 것에서 벗어나, 로컬 장치에서 실행 가능한 소형 LLMs의 성능을 극대화하기 위한 다양한 메모리 처리 기법을 도입하였습니다. 특히, 중복 정보를 줄이기 위한 의미 인식 메모리 검색과 긴 기획 지연을 줄이기 위한 이산 메모리 캐싱 및 주의 기반 메모리 클러스터링을 사용합니다.

- **Technical Details**: EfficientNav 시스템은 메모리 제약을 고려하여 탐색 맵의 설명을 최적화합니다. 구체적으로, 지도 정보를 그룹으로 클러스터링하고 각 그룹에 대해 KV 캐시를 독립적으로 계산하여 메모리 요구 사항을 줄입니다. 또한, 주의 기반 메모리 클러스터링을 통해 관련 정보를 같은 그룹으로 묶어 주의 메커니즘을 활용하여 성능을 개선합니다.

- **Performance Highlights**: Extensive experiments show that EfficientNav는 HM3D 데이터셋 상에서 GPT-4 기반 방법에 비해 11.1%의 성공률 향상을 이루었으며, 실시간 지연 시간에서 6.7배, 엔드 투 엔드 지연 시간에서 4.7배 감소를 달성했습니다. 이로써, 로컬 장치에서 효율적으로 제로샷 탐색을 가능하게 하는 혁신적인 접근법이 될 것입니다.



### Pay Attention to the Triggers: Constructing Backdoors That Survive Distillation (https://arxiv.org/abs/2510.18541)
- **What's New**: 이 논문에서는 백도어(backdoor)가 있는 교사 모델(teacher model)으로부터 학생 모델(student model)으로의 지식 증류(knowledge distillation)에 대한 보안 위험을 조사했습니다. 기존 연구들은 대부분 백도어가 학생 모델로 이전되지 않는다고 보았지만, 본 연구에서는 다중 토큰(multi-token) 트리거를 사용한 새로운 백도어 공격 기법인 T-MTB를 소개하여 전이 가능성을 밝혔습니다. T-MTB는 학생 모델에 위험한 트리거들이 안전하게 전이되도록 설계되었습니다.

- **Technical Details**: 연구에서는 LLM 학습 과정에서 발행되는 다양한 보안 위험을 다루기 위해 T-MTB라는 새로운 백도어 공격 기법을 도입했습니다. 이 기법은 individual trigger tokens를 조합하여 composite backdoor trigger를 생성하여, 학생 모델에 전이 가능한 백도어를 만들어 냅니다. 논문에 따르면, T-MTB를 통해 여러 LLM 모델 패밀리(Llama2, Llama3, Qwen2.5, Mistral)에서 사용하여 60%의 공격 성공률을 확인했습니다.

- **Performance Highlights**: 백도어 공격의 성과를 검토한 결과, 기존의 희귀 트리거(token) 설계 방안은 LLM 지식 증류의 보안 위험을 과소평가하고 있음이 드러났습니다. T-MTB를 사용하여 다양한 데이터 세트와 공격 시나리오를 기반으로 백도어의 전이 가능성을 실증적으로 평가했으며, 실제 데이터 세트와 상관없이 효율적으로 전이되는 백도어를 발견했습니다. 이는 지식 증류 과정에서 백도어가 발생할 수 있는 다양한 실질적인 위협을 강조합니다.



### Zero-Shot Vehicle Model Recognition via Text-Based Retrieval-Augmented Generation (https://arxiv.org/abs/2510.18502)
Comments:
          Accepted by The 38th Conference of Open Innovations Association FRUCT, 2025

- **What's New**: 이 논문에서는 최신 차량 모델 인식에서 기존 모델들이 새로운 모델에 적응하는데 어려움을 겪는 문제를 해결하기 위해 새로운 접근 방식을 제안합니다. Contrastive Language-Image Pretraining (CLIP) 모델의 한계를 극복하기 위해, Retrieval-Augmented Generation (RAG)와 비전 언어 모델(Vision Language Models, VLMs)을 통합한 제로샷(Zero-shot) 인식 파이프라인을 개발하였습니다. 이 시스템은 차량 이미지를 텍스트로 변환하고, 텍스트 기반의 추론을 통해 차량의 메이크(make)와 모델(model)을 식별할 수 있도록 설계되었습니다.

- **Technical Details**: 제로샷 차량 모델 인식 방식을 통해 입력 이미지를 처리하고 가장 가능성 높은 레이블을 예측하는 새로운 메커니즘을 소개합니다. 이 과정에서는 비전-언어 인코더(Ev)와 텍스트 데이터베이스를 비교하여 관련 정보를 검색하는 단계가 포함됩니다. RAG의 틀은 외부 지식에 기반한 추론을 가능케 하며, 새로운 차량 모델에 대한 텍스트 설명 업데이트를 통해 시스템의 확장성을 보장합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 CLIP의 기본 성능을 기준으로 차량 인식을 거의 20% 향상시키는 것으로 나타났습니다. 연구에 사용된 데이터셋은 최근 출시된 차량 모델로 구성되어 있어, 진정한 제로샷 평가 시나리오를 제공합니다. 이 방식은 새로운 차량 모델의 효과적인 인식을 가능하게 하며, 지능형 교통 시스템을 위한 실제 적용 가능성을 입증하고 있습니다.



### One Size Fits All? A Modular Adaptive Sanitization Kit (MASK) for Customizable Privacy-Preserving Phone Scam Detection (https://arxiv.org/abs/2510.18493)
Comments:
          9 pages

- **What's New**: 이 논문은 전화 사기 탐지에서 LLM(대형 언어 모델)을 활용하면서 사용자 개인 정보를 보호하는 방법을 탐구합니다. 기존의 정적 개인 정보 보호 규칙을 넘어, MASK(모듈형 적응형 위생 키트)라는 동적이고 사용자 중심의 프레임워크를 제안합니다. MASK는 사용자 선호도에 따라 다양한 위생 기법을 조정할 수 있어 개인의 프라이버시 요구를 충족합니다.

- **Technical Details**: MASK는 고정된 위생 프로세스 대신 개인정보를 보호하는 다양한 방법을 유연하게 구성할 수 있는 플러그형 아키텍처를 가지고 있습니다. 사용자 간의 동적인 개인 정보 보호 조정을 지원하여, 음성과 텍스트 데이터와 같은 멀티미디어 콘텐츠에서도 효과적으로 적용될 수 있습니다. 또한, 사용자 요구에 맞는 다양한 모델링 접근법과 손실 함수 설계의 가능성을 논의하고 있습니다.

- **Performance Highlights**: MASK는 사용자 자율성을 존중하면서 실시간 개인 정보 보호 결정을 지원합니다. 이 프레임워크는 시간 소모적인 상호작용을 줄이며, 높은 정확도의 전화 사기 탐지를 가능하게 하는 유용성을 유지합니다. MASK를 통해 사용자 개인 정보 요구는 더이상 일률적인 규칙에 제한되지 않고, 개인화되고 상황에 맞춰 조정될 수 있습니다.



### Benchmarking Fairness-aware Graph Neural Networks in Knowledge Graphs (https://arxiv.org/abs/2510.18473)
- **What's New**: 본 논문은 공정성을 고려한 그래프 신경망(Fairness-aware GNNs)의 성능을 지식 그래프(Knowledge Graphs)에서 평가한 최초의 연구로, YAGO, DBpedia, Wikidata 등 세 가지 지식 그래프에서 새로운 그래프를 생성하였다. 기존의 공정성 연구에서 사용된 그래프 데이터셋보다 크고 다양한 지식을 포함하여, GNN의 공정성을 연구하기 위한 새로운 벤치마크 연구를 소개한다. 이를 통해 GNN의 성능 평가 시 다양한 요소들이 공정성에 미치는 영향을 면밀히 분석하였다.

- **Technical Details**: 본 연구에서는 GNN의 성능을 평가하기 위해 8가지 방법, 4가지 백본 모델(GNN backbones), 3가지 조기 종료 조건을 결합하여 총 96개의 패턴을 9개의 실제 그래프에 대해 평가하였다. 공정성을 고려한 GNN 방법들(예: FairGNN, NIFTY 등)과 전처리 기법들(예: FairDrop)을 비교하여, 각각의 접근법이 예측 정확성과 공정성 간의 균형에 미치는 영향을 분석하였다. 연구 결과, 지식 그래프는 기존 데이터셋에서와는 다른 경향을 보이며, 조기 종료 조건이 공정성 지표에 더 큰 영향을 미치는 것으로 나타났다.

- **Performance Highlights**: 연구 결과, 공정성을 고려한 GNN 방법들은 특정 그래프에서 예측 정확도를 향상시키기 위해 공정성을 희생하는 경향이 있음을 보였다. 특히 전처리 기법이 공정성 지표를 개선하는 데 효과적이며, 지식 그래프에서는 예측 정확도와 공정성 간의 명확한 트레이드오프가 나타났다. 이러한 발견들은 향후 공정성을 고려한 GNN 개발 및 실질적 사회적 문제 해결에 기여할 수 있다.



### CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignmen (https://arxiv.org/abs/2510.18471)
- **What's New**: 이 논문에서는 코드 생성의 새로운 접근 방식인 CodeRL+를 제안합니다. CodeRL+는 Reinforcement Learning with Verifiable Rewards (RLVR) 체계에서 실행 의미의 정렬을 통합하여 텍스트 표현과 실행 의미 간의 격차를 줄입니다. 이를 통해 모델이 변수 수준의 실행 경로를 추론할 수 있게 하여 실행 의미의 직접적인 학습 신호를 제공합니다.

- **Technical Details**: CodeRL+는 코드 생성과 실행 의미 정렬을 병렬적으로 최적화하도록 설계되었습니다. 이는 실패한 탐색 프로그램을 재사용하여 변수들이 프로그램 실행 동안 어떻게 전파되는지를 분석함으로써 코드의 기능적 행동과 텍스트 형식을 명확히 정렬하는 방식을 채택합니다. 이 방법은 추가 데이터 소스 없이도 기존의 RL 알고리즘과 원활하게 통합될 수 있습니다.

- **Performance Highlights**: 실험 결과, CodeRL+는 RLVR 및 Distillation을 포함한 기존 모델에 비해 평균 4.6%의 개선 효과를 보여주었습니다. 특히 코드 추론 및 테스트 출력 생성 벤치마크에서 각각 15.5% 및 4.4% 높은 정확성을 기록하며 뛰어난 일반화 성능을 보여주었습니다. CodeRL+는 다양한 RL 알고리즘 및 LLM에 대한 강력한 적용 가능성을 보여줍니다.



### Simple and Efficient Heterogeneous Temporal Graph Neural Network (https://arxiv.org/abs/2510.18467)
Comments:
          Accepted by Neurips 2025

- **What's New**: 이번 논문에서는 이질적인 시간 그래프(Heterogeneous Temporal Graphs, HTGs)를 처리하기 위한 새로운 학습 패러다임인 Simple and Efficient Heterogeneous Temporal Graph Neural Network (SE-HTGNN)를 제안합니다. 이러한 접근법은 시간 모델링과 공간 학습을 통합하여, 모델의 복잡성을 줄이고 효율성을 높이는 데 중점을 둡니다. 특히, 과거 그래프 스냅샷에서 주의를 유지하여 다음 학습에 도움을 주는 동적 주의 메커니즘을 도입했습니다.

- **Technical Details**: SE-HTGNN은 기존의 두 단계에 나누어 수행되는 공간 및 시간 모델링 방식의 한계를 극복하는 데 초점을 맞추었습니다. 마이크로 레벨에서는 불필요한 주의 레이어와 선형 프로젝션을 단순화하였으며 매크로 레벨에서는 시간 모델링을 공간 학습에 통합하여 학습 단계를 줄였습니다. 논문에서는 대형 언어 모델(LLMs)을 활용하여 HTGs의 암묵적 속성을 포착하게 하여 모델의 적응성과 성능을 강화하는 방법도 소개합니다.

- **Performance Highlights**: 실험 결과, SE-HTGNN은 최신 벤치마크에 비해 최대 10배의 속도를 기록하며, 예측 정확도 또한 우수한 성능을 보였습니다. 특히, 기존 모델들이 가진 높은 복잡성과 비효율성 문제를 해결하는 데 기여하며, HTGs 데이터셋에서의 효과적인 학습 성능을 입증했습니다. SE-HTGNN은 다양한 다운스트림 태스크에서도 뛰어난 효율성과 성능을 발휘하여 연구의 기여를 확인할 수 있었습니다.



### DeLoad: Demand-Driven Short-Video Preloading with Scalable Watch-Time Estimation (https://arxiv.org/abs/2510.18459)
- **What's New**: 짧은 비디오 스트리밍이 모바일 미디어에서 주된 패러다임으로 자리 잡으면서, 이 논문은 기존의 Preloading 전략의 한계를 극복하기 위한 새로운 프레임워크인 DeLoad를 제안합니다. DeLoad는 동적 과제 크기 조정 및 실용적인 시청 시간 예측 방법을 도입하여 사용자 경험(QoE)을 개선하고 대역폭 효율성을 극대화하는 데 중점을 둡니다. 또한, 심층 강화 학습(Deep Reinforcement Learning, DRL) 기반의 에이전트를 활용하여 다운로드 범위 결정을 최적화합니다.

- **Technical Details**: DeLoad는 세 가지 주요 구성 요소로 이루어져 있습니다. 첫째, Weibull 분포에 기초한 새로운 시청 시간 모델링 및 예측 기법을 개발하였습니다. 둘째, Demand-Driven Video Selection에서는 예측된 시청 시간 분포, 버퍼 상태 및 재생 목록 순서를 통합하여 비디오 선택을 최적화합니다. 셋째, DRL 기반의 동적 범위 선택 알고리즘은 클라이언트 측에서 실행되어 동적 네트워크 조건에 적응할 수 있도록 설계되었습니다.

- **Performance Highlights**: DeLoad의 오프라인 평가를 통해 QoE 지표에서 34.4%에서 87.4%까지의 개선을 보여주었으며, 재버퍼링 시간을 최대 81.4%까지 감소시켰습니다. 또한 대규모 상용 짧은 비디오 플랫폼인 Douyin에 배포된 후, 사용자 시청 시간이 평균 0.09% 증가하고 전체 대역폭 소비는 3.76% 감소하였습니다. 이는 DeLoad가 짧은 비디오 Preloading에서 동적 작업 범위를 탐구하는 최초의 생산 품질 프레임워크임을 입증합니다.



### ImageGem: In-the-wild Generative Image Interaction Dataset for Generative Model Personalization (https://arxiv.org/abs/2510.18433)
- **What's New**: 우리는 개별 선호를 이해하는 생성 모델에 대한 연구를 위한 데이터셋, ImageGem을 소개합니다. 이 데이터셋은 57K 사용자의 실제 상호작용 데이터를 포함하며, 커스터마이즈된 LoRAs를 포함하여 242K개 사용자의 생성된 이미지를 포함하고 있습니다. 데이터셋의 사용자 선호 주석을 통해 우리는 선호 정렬 모델을 향상시킬 수 있었습니다.

- **Technical Details**: ImageGem 데이터셋은 Civitai 플랫폼에서의 데이터를 바탕으로 구성되었으며, 이 플랫폼은 개인화된 이미지 생성 모델과 함께 관련 메타데이터를 제공합니다. 데이터 구성은 LoRA 모델, 생성된 이미지, 그리고 이러한 모델을 업로드한 사용자들 간의 관계를 기반으로 하여, 사용자 특정 선호도를 효율적으로 조회하고 분석할 수 있게 설계되었습니다.

- **Performance Highlights**: ImageGem 데이터셋을 활용하여 사용자 개인화에 맞춘 이미지 검색 및 생성 모델 추천 성능을 시험했습니다. 또한, VLM(vision-language model)을 활용하여 사용자 선호를 캡셔닝하고 정렬하는 방법을 제안하였으며, 이를 통해 이미지 생성 모델의 개인화 작업에서 새로운 패러다임을 마련했습니다.



### ScaleNet: Scaling up Pretrained Neural Networks with Incremental Parameters (https://arxiv.org/abs/2510.18431)
- **What's New**: ScaleNet은 Vision Transformers (ViTs) 모델을 효율적으로 확장하는 새로운 방법을 제안합니다. 기존의 모델을 처음부터 훈련하는 대신 ScalNet은 pretrained (사전 훈련된) 모델을 기반으로 추가적인 레이어를 신속하게 삽입하여 모델을 확장합니다. 이 접근 방식은 인자 수의 미미한 증가로 ViTs를 효과적으로 확장할 수 있도록 합니다.

- **Technical Details**: ScaleNet은 추가 레이어를 pretrained ViTs에 삽입함으로써 모델 확장을 수행하며, 레이어별 가중치 공유를 통해 파라미터 효율성을 유지합니다. 각 추가 레이어는 pretrained 모델의 해당 레이어와 파라미터 텐서를 공유합니다. 가중치 공유로 인한 성능 저하를 방지하기 위해 각 레이어에 대해 조정 파라미터 세트를 도입하며, 이는 경량 어댑터 모듈을 통해 구현됩니다.

- **Performance Highlights**: ImageNet-1K 데이터셋에서의 실험 결과, ScaleNet은 2배 깊이로 확장된 DeiT-Base 모델이 처음부터 훈련한 모델에 비해 7.42%의 정확도 향상을 달성하며, 훈련 에폭 수도 1/3만 요구하는 것으로 나타났습니다. 이는 ViTs를 확장하는 데 있어 ScaleNet의 효율성을 강조합니다.



### Optimistic Higher-Order Superposition (https://arxiv.org/abs/2510.18429)
- **What's New**: 이번 논문은 기존의 λ-superposition 칼큘러스에서 발생하는 비효율적인 부분을 개선한 'optimistic λ-superposition' 칼큘러스를 소개합니다. 특히, 복잡한 유니피케이션(unification) 문제를 지연시키고, 함수의 확장성 원리를 보다 타겟형으로 적용하여 성능을 향상시킵니다. 이 새로운 접근 방식은 기존의 문제를 피하면서도, 높은 완전성을 유지하는 것을 목표로 합니다.

- **Technical Details**: 이 연구에서는 Henkin semantics를 기반으로 하는 새로운 논리 체계를 제시합니다. 새로운 칼큘러스는 함수 확장성을 위한 특수한 추론 규칙을 도입하고, 이는 특정한 용어 순서(term order)와 함께 작동하여 유니피케이션 문제를 지연시킵니다. 이러한 방법론은 기존의 복잡한 개념들을 제거하고, 논리 구조를 보다 단순화합니다.

- **Performance Highlights**: 새로운 λ-superposition 칼큘러스는 여러 시험 사례에서 기존 버전보다 더 나은 성과를 보일 것으로 기대됩니다. 연구자들은 이 칼큘러스가 모든 아규먼트에 대한 동등성을 증명하는 것을 지연시킴으로써, 불필요한 추론을 줄일 수 있다고 주장합니다. 향후 구현 및 실험을 통해 이러한 개선된 성능을 객관적으로 검증할 계획입니다.



### On AI Verification in Open RAN (https://arxiv.org/abs/2510.18417)
- **What's New**: 본 논문에서는 Open RAN에 대한 AI 모델의 신뢰성을 검증하기 위해 해석 가능한 모델에 기반한 경량화된 검증 접근 방식을 제안합니다. 이 방법은 네트워크 slicing 및 scheduling을 위한 Deep Reinforcement Learning (DRL) 에이전트의 행동을 검증하는데 유용합니다. 구체적으로, Decision Tree (DT) 기준의 검증자를 사용하여 실시간 일관성 검사를 수행함으로써 기존의 계산 비용이 큰 검증자보다 더 효율적으로 작동할 수 있습니다. 이를 통해 Open RAN의 AI 도입에 있어 신뢰성 있는 자동화를 증진할 수 있습니다.

- **Technical Details**: AI 검증(verification)은 학습 기반 시스템이 특정 행동 기준을 충족하는지를 확인하는 과정을 포함합니다. 기존의 검증 방법은 형식 도구에 의존하여 다양한 시나리오에서 일반성 및 확장성이 부족했던 반면, 설명 가능성(explainability) 기술은 DT와 같은 해석 가능한 모델을 사용하여 인공지능(aI) 시스템의 행동을 보다 명확히 이해할 수 있도록 도와줍니다. 이 논문은 Open RAN 환경에 API 기반의 효율적인 검증 메커니즘을 통합하여 이를 보완하고 있습니다.

- **Performance Highlights**: 제안된 DT 기반의 슬라이스 검증자는 DRL 기반의 RAN slicing 및 scheduling의 타이밍 제약 내에서 실현 가능성을 보여줍니다. 이 시스템은 거의 실시간 피드백을 제공함으로써 네트워크 운영의 신뢰성을 높일 수 있습니다. 연구에서는 XAI와 AI 검증의 기여를 명확히 하고, 효율적인 AI 검증을 통한 Open RAN의 데이터 네트워킹 및 서비스 품질을 향상시키는 방향성도 제시하고 있습니다.



### Learning from N-Tuple Data with M Positive Instances: Unbiased Risk Estimation and Theoretical Guarantees (https://arxiv.org/abs/2510.18406)
- **What's New**: 이 논문은 약한 감독 학습(Weakly Supervised Learning)에서 각 훈련 예제가 정확히 'm'개의 긍정 예제를 포함한 n-튜플( N-tuple )로 구성되는 새로운 설정(NTMP; N-tuple with M positives)을 제안합니다. 이는 이미지 분류와 다중 인스턴스 측정에서 발생하는 문제로, 튜플 카운트(m per tuple)만 관찰될 수 있습니다. 이러한 NTMP 감독 방식을 통해 훈련 가능한 편향 없는 위험 추정기(URE)를 도출하면서, 튜플 생성 프로세스와 잠재 인스턴스 주변 확률(latent instance marginals)을 연결합니다.

- **Technical Details**: 이 논문에서 제안하는 접근 방식은 튜플의 정확한 긍정 개수만을 활용하여, 편향 없는 위험 추정기(URE)를 도출하는 데 집중합니다. 우리는 고정된 튜플 크기(n,m)에서 출발하여 폐쇄형 URE를 유도하고, 가변 튜플 크기 및 개수에 대해서도 확장합니다. 정확한 튜플 카운트 정보를 통해 유도된 URE는 가시적인 클래스 조건부(class-conditionals)가 사라진 선형 시스템을 통해 생성됩니다.

- **Performance Highlights**: 실험 결과, NTMP 작업으로 변환된 여러 벤치마크(MNIST, FashionMNIST, CIFAR-10 등)에서 제안된 방법이 약한 감독 학습 방법들(UU 학습 및 클러스터링 기법)보다 일관되게 우수한 성능을 보였습니다. 제안된 방법은 클래스 비율 불균형(class-prior imbalance)과 다양한 튜플 구성에서 강인성을 유지하며, 카운트만으로도 효과적으로 감독할 수 있는 가능성을 보여줍니다.



### Automated Wicket-Taking Delivery Segmentation and Weakness Detection in Cricket Videos Using OCR-Guided YOLOv8 and Trajectory Modeling (https://arxiv.org/abs/2510.18405)
Comments:
          6 figures, 5 tables, submitted to the 11th IEEE International Women in Engineering (WIE) Conference on Electrical and Computer Engineering 2025

- **What's New**: 이 논문은 깊이 있는 학습 기법을 활용하여 크리켓 비디오 분석을 자동화하는 시스템을 제안합니다. 시스템은 विकेट을 유도하는 투구를 추출하고, 크리켓 공을 탐지하며, 공의 경로를 모델링하는 데 중점을 두고 있습니다. 이 시스템은 YOLOv8 아키텍처를 사용하여 피치(pitch)와 공을 탐지하며, OCR(Optical Character Recognition)을 통해 점수판을 추출하여 중요한 순간을 식별할 수 있습니다.

- **Technical Details**: 제안된 시스템은 세 가지 주요 구성 요소로 이루어져 있습니다: (1) 점수판 분석을 통한.wicket-유도 투구 분할, (2) YOLOv8 기반의 피치 및 공 탐지, (3) 경량 지역 식별을 위한 경로 모델링입니다. 데이터셋은 피치 탐지를 위한 951개의 주석 이미지와 공 탐지를 위한 257개의 주석 이미지로 구성되어 있으며, 각각의 탐색 및 검증을 위한 교육 절차가 포함되어 있습니다. YOLOv8 모델은 정확성과 추론 속도 모두에서 우수한 성능을 보입니다.

- **Performance Highlights**: 제안된 시스템은 피치 탐지에서 99.5%의 평균 평균 정밀도(mean Average Precision, mAP50)를 달성했으며, 공 탐지에서는 99.18%의 mAP50과 0.968의 정밀도(precision)와 0.978의 재현율(recall)을 기록했습니다. 이 시스템은 비디오의 분할된 조각에서 위켓을 유도하는 투구를 효과적으로 확인하였으며, 3D 궤적 그래프를 통해 위켓을 유도하는 투구로부터 뚜렷한 약점 영역을 식별할 수 있는 데이터를 제공합니다.



### MENTOR: A Reinforcement Learning Framework for Model Enhancement via Teacher-Optimized Rewards in Small Models (https://arxiv.org/abs/2510.18383)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 도구 사용 능력을 더 작고 효율적인 소형 언어 모델(SLM)로 증류하는 새로운 방법인 MENTOR를 제안합니다. 기존의 감독하에 세부 조정(Supervised Fine-Tuning, SFT) 접근 방식은 정적인 교사 궤적을 모방하도록 모델을 훈련시켜 일반화에 어려움을 겪고 있습니다. MENTOR는 이러한 단점을 극복하기 위해 강화 학습(Reinforcement Learning, RL)과 교사-guided 증류를 조합하여 더 일반화된 정책을 학습하는 비전통적인 접근 방식을 사용합니다.

- **Technical Details**: MENTOR 프레임워크는 RL 기반의 탐색 과정을 통해 일반화 가능성이 높은 정책을 학습합니다. 이 과정에서 보상의 희소성 문제를 해결하기 위해 교사의 참조 궤적을 사용하여 조밀하고 복합적인 교사-guided 보상을 구성합니다. 이러한 보상은 SO الار οδη기와 보다 세밀한 안내를 제공하여 SLM의 학습 효율성을 높입니다.

- **Performance Highlights**: 실험 결과 MENTOR는 SFT 및 표준 희소 보상 RL 기준선과 비교하여 SLM의 도메인 간 일반화(cross-domain generalization) 및 전략적 능력(strategic competence)을 크게 향상시킨 것으로 나타났습니다. 이에 따라 MENTOR는 소형 언어 모델의 실제 적용 가능성을 향상시키는 데 중요한 기여를 합니다.



### S2AP: Score-space Sharpness Minimization for Adversarial Pruning (https://arxiv.org/abs/2510.18381)
- **What's New**: 본 논문에서는 적대적 프루닝(adversarial pruning) 방법의 안정성을 향상시키기 위한 새로운 기법인 S2AP(Score-space Sharpness-aware Adversarial Pruning)를 제안합니다. 이 방법은 중요도 점수(importance scores)의 변화를 통해 손실 경량화(loss landscape의 sharpness)를 감소시킴으로써 마스크 선택(mask selection)의 안정성을 높입니다. 이는 전통적인 프루닝 방법에 플러그인 방식으로 통합되어, 기존의 핵심 논리를 변경하지 않고도 사용할 수 있습니다.

- **Technical Details**: S2AP는 세 단계로 구성된 프루닝 파이프라인(i) robust 모델의 사전 훈련, (ii) 이진 마스크 선택, (iii) 프루닝된 모델의 미세 조정(finetuning)을 포함합니다. 이 과정에서, 각 가중치의 중요도를 평가하여 robust 손실을 최소화하는 방식으로 마스크를 최적화합니다. 추가적으로, 손실 경량화의 안정성을 확보하기 위해 중요도 점수를 변화시켜 sharpness를 최소화하는 새로운 전략을 제시합니다.

- **Performance Highlights**: 다양한 데이터셋, 모델 및 스파시티 수준에서 진행된 실험 결과에 따르면, S2AP는 중요도 점수 공간에서의 sharpness를 효과적으로 최소화하며, 마스크 선택의 안정성을 도모하고 최종적으로 적대적 프루닝 방법의 강인성을 개선합니다. 연구 결과는 S2AP 방법이 기존의 적대적 프루닝 방법보다 성능을 향상시킴을 보여줍니다.



### PGTT: Phase-Guided Terrain Traversal for Perceptive Legged Locomotion (https://arxiv.org/abs/2510.18348)
Comments:
          9 pages, 9 figures, 2 tables

- **What's New**: 이번 논문은 Phase-Guided Terrain Traversal (PGTT)라는 새로운 심층 강화 학습 접근 방식을 제안합니다. 이 방법은 보상 형성을 통해 보행 구조를 유지하여, 기존의 오실레이터(oscillator)나 역기구학(inverse kinematics)을 사용한 규제를 피합니다. PGTT는 각 다리의 이동 단계(phase)를 큐빅 에르미트 스플라인(cubic Hermite spline)으로 표현하며, 이는 로봇의 성능을 다양화합니다.

- **Technical Details**: PGTT는 LiDAR를 통한 로봇 중심(heightmap) 높이 맵을 실시간으로 사용하여 지형을 인코딩합니다. 이 방법은 보상 측면에서만 단계(phase) 우선을 적용하며, 정책이 직접 관절 공간(joint space)에서 작동하도록 합니다. 따라서 발표된 원리는 정책 학습에 있어 귀납적 편향(inductive bias)을 줄이고 여러 형태의 로봇에 대한 적합성을 높입니다.

- **Performance Highlights**: PGTT는 MuJoCo 환경에서 진행된 시뮬레이션 테스트에서 신속한 학습 수렴을 보여주며, 기존의 최첨단 방법에 비해 더 높은 생존율(success rates)을 기록했습니다. 실제로 Unitree Go2에 적용하여 계단 및 장애물을 효과적으로 통과하는 성능을 입증했으며, ANYmal-C에서도 초기 결과를 확보했습니다. 이러한 결과는 PGTT의 단순화된 형태의 보상으로 인해 다양한 플랫폼에서 견고한 인식 보행이 가능하다는 것을 보여줍니다.



### Scalable, Explainable and Provably Robust Anomaly Detection with One-Step Flow Matching (https://arxiv.org/abs/2510.18328)
Comments:
          Paper accepted by NeurIPS 2025

- **What's New**: 최신 연구에서는 Time-Conditioned Contraction Matching (TCCM)이라는 새로운 방법을 소개하며, 이는 표 형식 데이터에서 준지도(anomaly detection) 이상 탐지를 위한 혁신적인 기법입니다. TCCM은 최근의 흐름 매칭(flow matching) 모델을 기반으로 하여 확률 분포 간의 속도 필드를 학습하는 아이디어를 채택하고 있습니다. 이는 기존의 확산 모델(diffusion models) 및 적대적 신경망(generative adversarial networks)과 비교하여 강력한 성능을 보여줍니다.

- **Technical Details**: TCCM은 각 샘플링된 시간 단계에서 고정된 목표(원점)로 수축하는 시간 조건 수축 벡터(time-conditioned contraction vector)를 예측하여 프레임워크를 단순화합니다. 이를 통해 모형 훈련과 추론과정에서 ODE(Ordinary Differential Equations)를 해결할 필요가 없어져 경량화 및 확장성 있는 훈련 목표를 제공합니다. 또한, 'one time-step deviation'이라 불리는 효율적인 점수 산정 전략을 통해 기존의 모델들이 가진 추론 병목 현상을 해결합니다.

- **Performance Highlights**: 대규모 데이터셋에서 다양한 실험을 통해 TCCM은 탐지 정확도와 추론 비용 사이의 균형을 잘 맞추고 있음을 보여줍니다. 특히 고차원 및 대규모 데이터셋에서 기존의 최첨단 방법들에 비해 뛰어난 성능을 발휘하며, 모델의 해석 가능성과 견고함 또한 보장됩니다. 이러한 특성 덕분에 TCCM은 특별한 대량의 데이터와 높은 차원의 데이터 처리에 적합합니다.



### MoMaGen: Generating Demonstrations under Soft and Hard Constraints for Multi-Step Bimanual Mobile Manipulation (https://arxiv.org/abs/2510.18316)
Comments:
          Project website: this http URL. The first four authors contribute equally

- **What's New**: 이번 연구는 복합적인 다단계 쌍손 모바일 조작 방식에서의 데이터를 생성하는 새로운 방법인 MoMaGen을 소개합니다. 이 방법은 강력한 제약 조건을 유지하면서 데이터 생성 문제를 제약 최적화 문제로 공식화하여 접근성을 높입니다. MoMaGen은 이전의 자동 데이터 생성 방법보다 한층 더 발전된 방식으로, 시뮬레이션에서 다양한 데이터셋을 생성할 수 있습니다.

- **Technical Details**: MoMaGen는 두 가지 주요 제약 조건 즉, 접근 가능성과 조작 중 시각적 노출을 지원하는 하드 제약(hard constraints)과 내비게이션 중 시각적 노출을 유지하는 소프트 제약(soft constraints)을 포함합니다. 데이터 생성은 기존의 방법보다 더 일반적이고 효율적인 방식으로 구현되었습니다. 이 프레임워크를 통해 실제 세계에서의 복잡한 과제들을 해결할 수 있는 기반을 마련하고 있습니다.

- **Performance Highlights**: MoMaGen은 네 가지 다단계 쌍손 모바일 조작 작업에서 테스트되었으며, 기존 방법보다 훨씬 다양한 데이터셋을 생성했습니다. 합성된 데이터를 통해 단일 소스 데모에서 효과적인 모방 학습 정책을 훈련할 수 있으며, 40개의 실제 데모로 미세 조정이 가능합니다. 실제 로봇 하드웨어에서의 배포 성공도 입증되었습니다.



### Higher Embedding Dimension Creates a Stronger World Model for a Simple Sorting Task (https://arxiv.org/abs/2510.18315)
- **What's New**: 이 논문에서는 강화 학습(reinforcement learning)을 사용하여 훈련된 트랜스포머(transformer)에서 임베딩 차원(embedding dimension)이 내부 '세계 모델'(world model)의 발생에 미치는 영향을 조사합니다. 작은 임베딩 차원에서도 높은 정확도를 달성할 수 있지만, 더 큰 차원은 보다 신뢰할 수 있고 일관적이며 견고한 내부 표현을 생성합니다. 이 연구에서는 임베딩 차원이 어떻게 구조화된 내부 표현 형성을 강화하고 해석 가능성을 향상시키는지를 설명합니다.

- **Technical Details**: 저자들은 Proximal Policy Optimization (PPO) 알고리즘을 사용하여 강화 학습 환경에서 토큰의 순열을 상태로 설정하고 인접 스왑을 유일한 허용 작업으로 설정했습니다. 연구 결과, 주의(attention) 가중치 행렬의 마지막 행이 토큰의 전역 순서를 단조롭게 인코딩하며, 가장 큰 인접 차이를 선택하는 규칙이 에이전트의 행동을 설명하는 두 가지 일관된 패턴이 발견되었습니다. 이 방법론을 통해, 트랜스포머가 구조화된 내부 세계 모델을 구축한다는 정량적 증거를 제공하고 있습니다.

- **Performance Highlights**: 실험 후, 저자는 임베딩 차원을 증가시키는 것이 표현의 진실성(faithfulness), 일관성(consistency), 견고성(robustness)을 크게 향상시킨다는 것을 보여주는 실증 연구를 제공합니다. 또한, 에이전트가 정렬된 행동을 조정하기 위해 사용하는 두 가지 메커니즘을 확인하며, 이로 인해 보다 명확한 해석이 가능합니다. 이러한 결과는 작은 트랜스포머 네트워크가 복잡한 알골리즘 작업을 수행하는 데 있어 강력한 해석 가능성을 제공함을 강조합니다.



### From Retrieval to Generation: Unifying External and Parametric Knowledge for Medical Question Answering (https://arxiv.org/abs/2510.18297)
Comments:
          13 pages, 4 figures

- **What's New**: 본 논문에서는 MedRGAG라는 새로운 프레임워크를 제안합니다. MedRGAG는 의료 질문 응답(QA)을 위해 외부 지식과 매개변수 지식을 통합하는 통합된 검색-생성 방식의 구조를 가지고 있습니다. 이 프레임워크는 Knowledge-Guided Context Completion (KGCC)과 Knowledge-Aware Document Selection (KADS) 두 가지 핵심 모듈로 구성되어 있어, 신뢰할 수 있는 응답 생성을 위해 필요한 증거를 효과적으로 통합합니다.

- **Technical Details**: MedRGAG의 첫 번째 모듈인 KGCC는 검색된 문서를 분석하고 누락된 지식을 판단한 후, 필요한 배경 문서를 생성하는 역할을 합니다. 두 번째 모듈인 KADS는 검색된 문서와 생성된 문서를 지식 요구사항에 기반하여 그룹화하고, 적합한 증거의 조합을 선택합니다. 이러한 통합 설계를 통해 MedRGAG는 의료 QA에서의 정확성 및 신뢰성을 크게 향상시킬 수 있습니다.

- **Performance Highlights**: 다섯 가지 의료 QA 벤치마크에서의 실험을 통해 MedRGAG가 MedRAG에 비해 평균 12.5%, MedGENIE에 비해 4.5% 향상된 정확도를 달성했음을 보여주었습니다. 이 결과는 MedRGAG가 검색과 생성을 통합하여 지식 집약적 추론에 효과적임을 강조합니다. 또한, MedRGAG는 보다 효과적인 보완적 배경 문서를 생성하고, 유용한 증거를 성공적으로 복구하는 것으로 확인되었습니다.



### Text or Pixels? It Takes Half: On the Token Efficiency of Visual Text Inputs in Multimodal LLMs (https://arxiv.org/abs/2510.18279)
Comments:
          Accepted to EMNLP 2025 Findings. Previously titled "Text or Pixels? Evaluating Efficiency and Understanding of LLMs with Visual Text Inputs"

- **What's New**: 최근 대형 언어 모델(LLMs)과 그 멀티모달 변형이 시각적 입력을 처리할 수 있게 되면서, 텍스트 입력을 이미지로 변환하여 토큰 사용량을 줄이고 성능을 유지할 수 있는 가능성이 제기되었습니다. 본 논문에서는 텍스트를 이미지로 표현함으로써 디코더 LLMs의 입력 압축이 가능하다는 것을 보여줍니다. 실험을 통해 이 텍스트-이미지 접근법이 토큰 절약에 효과적이며, 성능 저하 없이도 실질적인 이점을 제공함을 입증하였습니다.

- **Technical Details**: 이 연구에서는 멀티모달 LLMs가 시각적 텍스트 입력을 활용하여 입력 압축을 달성하는 방법에 대해 논의합니다. 긴 텍스트를 단일 이미지로 렌더링함으로써, 비전 인코더는 디코더가 처리할 수 있는 고정 길이의 시각적 토큰 시퀀스를 생성하고, 이는 시퀀스 길이를 직접 줄이는 효과를 가져옵니다. 이를 통해 기존 모델을 미세 조정하거나 추가적인 감독 없이도 디코더 토큰 수를 획기적으로 감소시킬 수 있음을 보여줍니다.

- **Performance Highlights**: 실험 결과, RULER 과제에서 GPT-4.1-mini와 Qwen2.5-VL-72B는 최대 58%의 디코더 토큰 수 감소에도 불구하고 97% 이상의 정확도를 유지했습니다. 또한, CNN/DailyMail 요약 과제에서는 이 접근법이 두 가지 전문 프루닝 기준을 초과하는 성과를 나타내며, 큰 모델에서는 전체적인 속도를 최대 45% 향상시키는 것으로 확인되었습니다. 이러한 결과들은 멀티모달 LLMs가 이미지를 암묵적인 압축 레이어로 활용하면서도 성능을 거의 원래 텍스트-토큰 비용의 절반으로 유지할 수 있음을 시사합니다.



### StreamingTOM: Streaming Token Compression for Efficient Video Understanding (https://arxiv.org/abs/2510.18269)
- **What's New**: 이번 논문은 StreamingTOM이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 감지, 비디오 이해를 위한 사전 학습 없이도 동작하는 두 단계의 프로세스를 통해 효율적인 스트리밍 비디오 처리를 가능하게 합니다. Causal Temporal Reduction(CTR)과 Online Quantized Memory(OQM)를 조합하여 사전 LL(LLM) 및 사후 LL 메모리 문제를 해결합니다.

- **Technical Details**: StreamingTOM의 CTR은 고정된 프레임 예산 내에서 인접 프레임의 변화와 토큰의 중요도에 기반하여 선택된 토큰만 처리함으로써 메모리 사용을 최적화합니다. OQM은 4비트 형식으로 토큰을 저장하여 필요한 그룹을 필요할 때마다 호출하여 효율적인 kv-cache 관리를 유지합니다. 이 방식을 통해 스트리밍 처리에서도 예측 가능한 지연 시간을 유지합니다.

- **Performance Highlights**: 실험 결과 StreamingTOM은 kv-cache 압축 비율이 15.7배, 최대 메모리 사용량이 1.2배 낮아지며, 첫 번째 토큰에 도달하는 시간(TTFT)이 2배 빠른 성능을 보였습니다. 또한, 오프라인 벤치마크에서 63.8%의 정확도를 유지하며, RVS에서 55.8%/3.7의 성과를 달성하여 현재 훈련 없는 방법들 중 최고의 정확도를 기록했습니다.



### Latent-Info and Low-Dimensional Learning for Human Mesh Recovery and Parallel Optimization (https://arxiv.org/abs/2510.18267)
Comments:
          Accepted by ICME2025

- **What's New**: 본 논문에서는 기존 3D 인간 메시 회복 방법의 한계를 극복하기 위해 잠재 정보(latent information)와 저차원 학습(low dimensional learning)을 바탕으로 한 2단계 네트워크를 제안합니다. 첫 번째 단계에서는 이미지 기능의 저주파와 고주파 성분에서 전반적인 형태 정렬을 포함한 글로벌(global) 정보와 텍스처 및 세부사항과 같은 로컬(local) 정보를 추출하여 하이브리드 잠재 주파수(domain feature) 특징으로 집계합니다. 두 번째 단계에서는 이 하이브리드 특징을 활용하여 3D 인간 메시의 포즈(pose)와 상호 작용(interaction)을 모델링하고 최적화합니다.

- **Technical Details**: 제안된 네트워크는 두 단계로 구성됩니다. 첫 번째 단계는 잠재 정보 추출(latent information extraction)으로, 3D 포즈와 하이브리드 특징을 추출합니다. 입력 비디오 시퀀스를 ResNet-50을 통해 처리하여 이미지 특징을 얻고, 2D 포즈 추정기는 해당 2D 포즈 추정을 제공합니다. 두 번째 단계에서는 3D 포즈와 메시 간의 상호 작용을 모델링하여 3D 인간 메시를 정확히 재구성합니다. 저차원 메시 포즈 상호 작용 방법을 통해 계산 비용을 줄이며 포즈와 형태의 상호 작용을 최적화합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 공개된 대규모 데이터셋에서 기존 최첨단 기법들에 비해 우수한 성능을 보여주었습니다. 특히, 저차원 메시 포즈 상호 작용 방법을 통해 계산 비용을 현저하게 감소시키면서도 재구성 정확도는 유지되었습니다. 이 성능 향상은 3D 포즈 추정과 메시 증가의 정확성을 강화하는 데 기여합니다.



### SPIKE: Stable Physics-Informed Kernel Evolution Method for Solving Hyperbolic Conservation Laws (https://arxiv.org/abs/2510.18266)
Comments:
          24 pages, 8 figures

- **What's New**: SPIKE(Stable Physics-Informed Kernel Evolution) 방법론이 도입되어 비점성 하이퍼볼릭 보존 법칙을 수치적으로 계산하는 혁신적인 접근 방식을 제안합니다. 이 방법론은 강한 형식의 잔차 최소화가 불연속성을 포함한 약한 해를 포착할 수 있는 근본적인 패러독스를 해결합니다. SPIKE는 매끄러운 매개변수 진화를 통해 충격 형성을 넘나드는 전이 메커니즘을 제공하여 역학이 충격 특이점을 통과할 수 있게 합니다.

- **Technical Details**: SPIKE 방법론은 핵 생성(kernel representation)과 정규화된 진화의 결합을 통해 강한 형식의 잔차를 최소화하는 기법을 사용합니다. 이 접근 방식에서는 매개변수들이 양자화된 형태로 진화하며, 충격의 올바른 동작을 자가적으로 전시하는 수학적 구조를 제공합니다. 이 방법은 약한 형식 제약이나 인위적인 충격 검출 기술 없이도 보존을 자동적으로 유지하고, 특성의 전파 및 Rankine-Hugoniot 점프 조건을 충족합니다.

- **Performance Highlights**: 수치 검증 결과, SPIKE 방법론은 스칼라 및 벡터 값 보존 법칙에서 그 효과성을 입증했습니다. 또한, 이 방법은 전통적인 수치적 접근 방식에 비해 경쟁력 있는 계산 효율성을 보여주며, 정규화 메커니즘을 통해 충격 형性을 원활하게 처리합니다. 이로 인해 SPITE는 현대 과학 및 공학의 핵심적인 적용 가능 분야에서 큰 잠재력을 지닙니다.



### Learning under Quantization for High-Dimensional Linear Regression (https://arxiv.org/abs/2510.18259)
- **What's New**: 본 논문은 낮은 비트 양자화(low-bit quantization)가 대규모 모델의 효율적인 훈련을 가능하게 하는 필수 기술로 부각되고 있음을 강조합니다. 특히, 양자화가 학습 성능에 미치는 영향을 이론적으로 탐구하고 있으며, 데이터, 레이블, 매개변수, 활성화 및 그래디언트를 포함한 다양한 양자화 대상에 대한 분석을 제공합니다. 이 연구를 통해 양자화가 훈련 중 노이즈를 증폭시키고 데이터 스펙트럼을 왜곡시키는 방식 등을 보여줍니다.

- **Technical Details**: 논문에서는 양자화된 확률적 경량 경사 하강법(quantized stochastic gradient descent, SGD)에 대해 체계적으로 분석하며, 양자화가 데이터 특징 공분산 행렬의 고유 스펙트럼, 샘플 크기 및 양자화 오차의 함수로서 초과 위험(excess risk)의 이론적 경계를 설정합니다. 양자화 기술의 두 가지 표준 오차 모델인 가법(additive) 및 곱셈(multiplicative) 양자화를 상세히 분석하여, 곱셈 양자화가 스펙트럼 왜곡을 없애고, 가법 양자화는 배치 크기 증가에 따라 활성화 및 그래디언트 양자화의 영향이 감소함을 보여줍니다.

- **Performance Highlights**: 두 가지 유형의 양자화 방법을 비교하여, 다차원 설정에서는 곱셈 양자화가 적용 가능하고, 가법 양자화는 그렇지 않음을 밝혀냅니다. 또한, 본 연구는 양자화가 모델 성능에 미치는 영향을 정량적으로 비교하며, 각 유형의 양자화가 우수한 성능을 제공하는 조건을 식별하는 데 기여합니다. 이러한 결과들은 저-정밀 훈련 기법의 이론적 기틀을 강화하고, 하드웨어 제한 하에서도 학습 이론을 탐색하는 길을 열어줍니다.



### NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspectiv (https://arxiv.org/abs/2510.18258)
- **What's New**: 이번 연구에서는 Multi-Task Learning (MTL)을 위한 새로운 접근법인 NTKMTL을 제안합니다. 이 방법은 Neural Tangent Kernel (NTK) 이론을 활용하여 여러 작업의 동학을 분석하고, 다양한 작업의 수렴 속도를 균형 있게 맞춤으로써 작업 불균형 문제를 해결하려고 합니다. 또한, NTKMTL-SR이라는 효율적인 근사를 도입하여 훈련 효율성을 높입니다.

- **Technical Details**: 연구는 NTK 이론을 바탕으로, MTL에 적용된 연장된 NTK 행렬을 소개합니다. 이 행렬을 통해 주어진 다중 작업의 수렴 속도를 파악하고, 각 작업에 대한 적절한 가중치를 부여하여 수렴 속도의 균형을 맞춥니다. 연구 결과, NTK 행렬의 고유값 분해에서 보듯이, 낮은 주파수 성분은 높은 NTK 고유값에 해당하며, 이는 더 빠른 수렴과 관련이 있습니다.

- **Performance Highlights**: 실험 결과, NTKMTL 및 NTKMTL-SR 방법이 다양한 벤치마크에서 최첨단 성능을 달성함을 보여주었습니다. 특히, 본 방법들은 2개에서 40개 작업까지의 다중 작업 감독 학습과 강화 학습 모두에서 우수한 성능을 발휘하였습니다. 연구는 다양한 작업을 효율적으로 학습할 수 있는 새로운 모델을 제시하여 학계와 실무에 기여할 것으로 기대됩니다.



### DelvePO: Direction-Guided Self-Evolving Framework for Flexible Prompt Optimization (https://arxiv.org/abs/2510.18257)
- **What's New**: 최근의 연구들은 Prompt Optimization (PO)이 대규모 언어 모델(Large Language Models, LLMs)의 특정 작업을보다 효과적으로 해결하기 위해 필요한 접근 방식으로 주목받고 있습니다. 기존의 연구들은 주로 LLM의 무작위 재작성 능력에 의존하고 있으며, 이로 인해 최적화 과정이 특정 요인에 집중되어 국소 최적(local optimum)에 쉽게 빠지게 되는 문제점이 있었습니다. 이를 해결하기 위해, 본 연구에서는 $	extbf{DelvePO}$ (Direction-Guided Self-Evolving Framework for Flexible Prompt Optimization)라는 프레임워크를 제안하여, 다양한 작업에 대해 자가 발전(self-evolve)하는 방식으로 프롬프트를 최적화하는 방법을 제시합니다.

- **Technical Details**: DelvePO 프레임워크는 LLM을 기반으로 한 작업 비특화적인_prompt 최적화 방법으로, 프롬프트를 여러 구성 요소로 분리하여 이를 통해 다양한 작업에서 영향 요인을 탐색할 수 있도록 합니다. 또한, 작업 메모리(working memory) 메커니즘을 도입하여 LLM의 불확실성으로 인한 단점을 완화하고, 새로운 프롬프트의 생성을 유도하기 위한 통찰을 얻는 과정을 포함하고 있습니다. 이러한 방법론은 진화 알고리즘의 효율성과 LLM의 강력한 텍스트 처리 능력을 통합하여 보다 안정적인 성능 향상을 달성합니다.

- **Performance Highlights**: 다양한 도메인에서의 여러 작업에 대한 실험 결과, DelvePO는 기존의 SOTA(최신 기술 동향) 방법들보다 일관되게 우수한 성능을 보였으며, 이는 다양한 작업들 간의 전이 가능성(transferability)을 보여줍니다. 프레임워크의 주요 기여는 각 요소의 진화 추세를 포착하는 개념적 메모리 메커니즘을 도입하여 프롬프트 최적화를 안내하고, 동시에 구성 요소의 상호 연결을 통해 전체 프롬프트의 점진적 최적화를 유도한다는 점입니다. DelvePO는 실험에서 수동으로 구성된 프롬프트보다 우수한 성능을 보여주었습니다.



### Hyperbolic Space Learning Method Leveraging Temporal Motion Priors for Human Mesh Recovery (https://arxiv.org/abs/2510.18256)
Comments:
          Accepted by ICME2025

- **What's New**: 이 논문에서는 3D 인간 메시 복원에서 하이퍼볼릭 공간을 활용한 학습 방법을 제안했습니다. 이를 통해 비디오에서 3D 인간 메시를 효과적으로 복원하였으며, 기존 방법들의 유클리드 공간에서는 포착하기 어려운 계층 구조를 더 잘 반영할 수 있게 되었습니다. 또한, 모션 정보의 시간적 특성을 강화하는 모듈을 디자인하여 메시의 정확성과 매끄러움을 동시에 향상시켰습니다.

- **Technical Details**: 제안된 방법은 크게 두 가지 모듈로 구성됩니다. 첫 번째는 시간적인 모션 선행 추출 모듈로, 입력된 3D 포즈 시퀀스와 이미지 특징 시퀀스에서 모션 특징을 추출합니다. 두 번째는 하이퍼볼릭 공간 최적화 학습 모듈로, 시간적 모션 선행 정보를 활용하여 3D 포즈와 모션 정보를 분리하여 메시 특징을 최적화합니다.

- **Performance Highlights**: 광범위한 데이터셋에서 실시한 실험 결과는 제안된 방법이 기존의 최첨단 기술들보다 우수한 성능을 보임을 나타냅니다. 특히, 하이퍼볼릭 공간에서의 학습 방법은 3D 메시의 계층 구조를 더욱 정밀하게 모델링할 수 있음을 입증하였습니다. 이러한 접근법은 가상 현실(VR) 및 증강 현실(AR) 분야에서 많은 응용 가능성을 열어 줄 것으로 기대됩니다.



### Finding the Sweet Spot: Optimal Data Augmentation Ratio for Imbalanced Credit Scoring Using ADASYN (https://arxiv.org/abs/2510.18252)
Comments:
          25 pages, 3 figures, 6 tables

- **What's New**: 이번 연구는 신용 점수 계산에서 클래스 불균형 문제에 대한 효과적인 데이터 증대 방법을 시기적절하게 평가했습니다. 특히 SMOTE, BorderlineSMOTE, ADASYN와 같은 다양한 기법을 비교하면서, 최적의 증대 비율은 1배임을 발견하였습니다. 이는 일반적인 1:1 균형 조정 관행과는 도리어 상반되는 결과로, 신용 점수 계산 분야의 실무자들에게 주목할 만한 지침을 제공합니다.

- **Technical Details**: 연구는 Give Me Some Credit 데이터셋을 활용하여 10가지 데이터 증대 시나리오를 시스템적으로 평가하였고, 모든 모델은 XGBoost를 사용하여 훈련되었습니다. 특히 ADASYN 기법이 1배 증대에서 최적의 성능(AUC 0.6778)을 나타내며, 통계적 검증을 통해 성과의 개선이 두드러짐을 보였습니다. 불균형 클래스 비율은 6.6:1이 최적으로 나타났으며, 이는 실무적 적용 가능성을 제시합니다.

- **Performance Highlights**: ADASYN에서 1x 증대 비율이 가장 효율적인 것으로 나타났고, 더 높은 증대 비율에서는 성능 저하가 관찰되었습니다. 특히 3배 증대에서는 AUC가 -0.48%로 감소하면서 '수확 체감법칙'이 적용됨을 나타냈습니다. 본 연구는 신용 점수를 위한 데이터 증대의 최적 '스위트 스팟'을 제시하며, 이러한 발견은 다양한 불균형 데이터 문제 해결을 위한 복제 가능한 프레임워크로 활용될 수 있습니다.



### Scaling Laws Meet Model Architecture: Toward Inference-Efficient LLMs (https://arxiv.org/abs/2510.18245)
Comments:
          27 pages, 17 figures

- **What's New**: 이 논문은 대형 언어 모델(LLM)의 추론 효율성과 정확성 간의 균형을 맞추기 위해 구조적 요소가 어떻게 작용하는지를 조사합니다. 주요 아키텍처 요소인 히든 사이즈(hidden size), MLP와 Attention 간의 파라미터 할당(mlp-to-attention ratio), 그리고 그룹 쿼리 주의(grouped-query attention, GQA)의 영향을 분석하며, 새로운 조건부 스케일링 법칙을 소개합니다.

- **Technical Details**: 연구진은 200개 이상의 모델을 훈련시켜 80M에서 3B 파라미터 및 8B에서 100B 학습 토큰을 포함하는 모델을 분석했습니다. 이들은 기존 스케일링 법칙을 보완하고, GQA와 같은 더 넓은 아키텍처 요소를 포함하는 일반적인 프레임워크를 통해 추론 비용과 정확성을 동시에 최적화하는 방법을 제시합니다.

- **Performance Highlights**: 제안한 조건부 스케일링 법칙은 아키텍처 선택을 예측하는 데 신뢰할 수 있으며, 최적화된 아키텍처는 기존 오픈소스 기준선을 초과하는 성능을 보입니다. 동일한 훈련 예산 하에서, 최적화된 아키텍처는 LLaMA-3.2에 비해 최대 2.1% 더 높은 정확도와 42% 이상의 추론 처리량을 달성했습니다.



### EVER: Edge-Assisted Auto-Verification for Mobile MR-Aided Operation (https://arxiv.org/abs/2510.18224)
- **What's New**: 이 논문에서는 EVER라는 새로운 시스템을 제안합니다. 이는 Mixed Reality (MR) 작업을 지원하는 엣지 기반 자동 검증 시스템으로, 사용자에게 정확한 안내를 제공하고 작업의 정확성을 확인합니다. 기존 방법과 달리, EVER는 물체 간의 차이를 보다 정확하게 고려하여 사용자 작업 전후의 프레임을 비교하고, Intersection over Union (IoU) 메트릭스를 활용한 임계값 기반 전략을 채택합니다.

- **Technical Details**: EVER는 두 개의 프레임, 즉 가상 블록이 포함된 참조 프레임과 물리적 블록만 있는 타겟 프레임을 캡처하여 비교합니다. 이때, 사용자의 손 움직임을 감지하여 적절한 타이밍에 프레임을 캡처하는 자동 모션 감지 방법을 사용합니다. 모든 처리 작업은 엣지 서버에서 수행되어 모바일 기기의 에너지 소비를 최소화하고, 약 100밀리초 이내에 90% 이상의 정확도로 자동 검증을 수행합니다.

- **Performance Highlights**: 완전한 공개 데이터 세트와 사용자 정의 데이터 세트를 통해 평가한 결과, EVER는 평균적인 인간 반응 시간인 약 273밀리초보다 훨씬 빠른 100밀리초 이내에 검증을 완료하며, 추가적인 계산 자원과 에너지도 최소한으로 소비합니다. 이러한 성능 개선은 자동 검증 기능을 통해 사용자 피드백을 능동적으로 제공하고, MR 작업의 직관성과 몰입감을 높이는 데 기여합니다.



### The Emergence of Complex Behavior in Large-Scale Ecological Environments (https://arxiv.org/abs/2510.18221)
Comments:
          18 pages, 11 figures, 6 tables, experiment code available at this https URL

- **What's New**: 이 논문에서는 복잡한 행동의 출현이 물리적 규모 및 인구 크기에 의해 어떻게 형성되는지를 탐구합니다. 에이전트들은 감독이 없고 명시적인 보상이나 학습 목표 없이 진화하며, 환경과 주변 인구를 지속적으로 변화시킵니다. 연구의 목표는 단일 고성능 정책을 최적화하는 것이 아니라, 자연적인 경쟁 및 환경 압력에 따라 행동이 어떻게 발생하고 진화하는지를 조사하는 것입니다.

- **Technical Details**: 우리는 60,000명 이상의 개별 에이전트를 포함하여 1,000,000개의 그리드 셀로 이루어진 대규모 생태 시뮬레이션 환경을 조사합니다. 에이전트들은 자원을 찾고 생존하기 위해 자신을 복제할 수 있는 자원을 모아야 합니다. 연구에서는 센서 구성 및 환경 규모가 행동에 미치는 영향을 확인하며, 특정 행동은 충분히 큰 환경과 인구에서만 나타나고, 더 큰 규모가 행동의 안정성과 일관성을 증가시킴을 보여줍니다.

- **Performance Highlights**: 에이전트는 자원을 장거리에서 획득하거나 시각 기반의 탐색 및 포식 같은 다양한 자생적 행동을 보입니다. 이러한 행동은 경쟁적 및 생존 압력에서 발생하며, 작은 규모에서는 신뢰성이 낮아지는 경향이 있습니다. 이 연구는 생태계가 AI 분야에서 기계 학습의 도구로서의 가능성을 제시하는 새로운 방향을 탐구하고 있습니다.



### VLSU: Mapping the Limits of Joint Multimodal Understanding for AI Safety (https://arxiv.org/abs/2510.18214)
Comments:
          10 pages, 5 figures, 4 tables. Under review

- **What's New**: 이번 연구는 멀티모달(Multimodal) 모델의 안전성을 평가하기 위한 새로운 프레임워크인 비전 언어 안전 이해(VLSU, Vision Language Safety Understanding)를 제시합니다. 이 프레임워크는 다양한 안전 패턴을 통해 멀티모달 안전성을 체계적으로 분석하며, 8,187개의 샘플로 구성된 대규모 벤치마크를 활용합니다. 연구결과, 기존의 모델들이 멀티모달 안전 신호를 제대로 이해하지 못한다는 것을 발견하였으며, 이는 이전 연구들에서 하지 못했던 위험 요소의 결합적 해석의 부재에서 기인합니다.

- **Technical Details**: VLSU 프레임워크는 자료 생성 과정에서 두 가지 주요 단계를 포함합니다. 첫 번째 단계에서는 해악 카테고리에 따른 경계(Unsafe) 및 안전(Safe) 위험 등급을 정의하고, 두 번째 단계에서는 개별 모드의 안전 등이 어떻게 결합되는지를 규명합니다. 또한, 새로운 경계 위험 등급을 도입하여 각 모드의 안전 신호를 평가하고, 교차 모드 상호작용(concatenation)을 고려한 안전 평가 방식을 개발합니다.

- **Performance Highlights**: 연구에서는 11개의 최첨단 VLM 모델을 평가한 결과, 개별 모드의 안전 신호에서는 90% 이상의 정확도를 달성했지만, 이미지와 텍스트 결합 시 안전 레이블 판별에선 성능이 20-55%로 급감했습니다. 더불어, 34%의 오류는 각 개별 모드에서 올바른 판별이 이루어졌음에도 발생함을 확인했습니다. 이러한 결과들은 현재의 모델들이 멀티모달 이해와 조합적 추론(compositional reasoning)에서 심각한 한계를 가지고 있음을 드러냅니다.



### Contrastive Decoding Mitigates Score Range Bias in LLM-as-a-Judg (https://arxiv.org/abs/2510.18196)
- **What's New**: 이 연구에서는 대형 언어 모델(LLM)이 평가자로 사용될 때 나타나는 점수 범위 편향(score range bias)에 대해 조사하였습니다. 특히, LLM들이 직접 평가를 수행할 때 점수 범위에 민감하다는 점을 밝혔으며, 이러한 편향은 동일 모델 계열에서도 발견되었습니다. 또한, 상대적인 평가 개선을 위하여 대비 부호화(contrastive decoding) 기법을 통해 이 편향을 완화하는 방법을 제안하였습니다.

- **Technical Details**: 대비 부호화는 두 개의 모델(주 모델과 보조 모델)을 활용하여 모델 출력을 수정하는 방법입니다. 주 모델의 다음 토큰 확률(pmain)에서 보조 모델의 확률(passt)을 가중치로 빼서 최종 점수를 조정합니다. 이 과정에서 하이퍼파라미터 λ를 포함하여 두 모델 간의 로짓 분포(logit distribution)를 더 잘 정렬합니다. 이러한 접근은 편향 분석을 통해 도출되었습니다.

- **Performance Highlights**: 이 연구에서 제안한 대비 부호화 기법은 다양한 점수 범위에서 인간 평가와의 스피어만 상관관계(Spearman correlation)를 평균 11.3% 향상시켰습니다. 요약 작업과 관련한 직접 평가에서 LLM 점수의 범위 편향을 분석하고 이를 완화함으로써, 평가의 신뢰성을 높일 수 있는 가능성을 보여주었습니다. 실험을 통해 각 모델 계열에서의 편향을 구체적으로 제시하며, 성능 향상의 기반을 마련하였습니다.



### RadDiagSeg-M: A Vision Language Model for Joint Diagnosis and Multi-Target Segmentation in Radiology (https://arxiv.org/abs/2510.18188)
- **What's New**: 이 논문에서는 진단 텍스트 및 픽셀 수준의 분할 마스크를 동시에 생성하는 데 어려움을 겪는 기존의 의료 비전 언어 모델(VLM)과 그 한계를 극복하기 위한 새로운 데이터셋과 모델, RadDiagSeg-D와 RadDiagSeg-M을 소개합니다. RadDiagSeg-D는 X-ray 및 CT를 포함한 다양한 이미징 모드에서 28,000개 이상의 샘플을 결합하여 단계별 질문을 설정하며, 이는 VQA(Visual Question Answering)와 세분화(segmentation) 작업을 포함하고 있습니다. RadDiagSeg-M은 비정상 탐지, 진단, 세분화를 공동으로 수행할 수 있는 새로운 비전-언어 모델로, 의료 진단에 유용한 결과를 제공합니다.

- **Technical Details**: RadDiagSeg-D는 비정상 탐지를 위한 폐쇄형 질문, 진단을 위한 개방형 질문, 다중 개체의 세분화 작업으로 구성된 3단계의 계층적 질문을 설계하였습니다. RadDiagSeg-M은 LISA의 구조를 기반으로 하며, 세분화 생성을 유도하는 특별한 토큰을 통해 모델의 어휘를 확장합니다. 이 모델은 단일 세분화 생성을 지원하는 기존 모델과 달리 유연한 수의 마스크 생성이 가능한 구조를 가지고 있습니다.

- **Performance Highlights**: RadDiagSeg-M은 RadDiagSeg-D 벤치마크의 VQA 하위 작업에서 최첨단 결과를 달성하며, 멀티 타겟 텍스트 및 마스크 생성 작업에 대해 경쟁력 있는 기준을 설정합니다. 이를 통해 의료적 맥락에서 필수적인 정보 제공과 클리닉적 유용성을 동시에 달성하게 됩니다. 논문에서 제안한 벤치마킹 도구는 연구 커뮤니티에서 다단계의 텍스트-마스크 생성 작업을 효과적으로 평가하는 데 도움을 줄 것입니다.



### VelocityNet: Real-Time Crowd Anomaly Detection via Person-Specific Velocity Analysis (https://arxiv.org/abs/2510.18187)
Comments:
          8 pages, 3 figures

- **What's New**: 이 논문에서는 복잡한 군중 장면에서 이상치를 탐지하기 위한 새로운 프레임워크인 VelocityNet을 제안합니다. 이 시스템은 개인의 속도를 추출하기 위해 헤드 탐지(head detection)와 밀집 광학 흐름(dense optical flow)을 결합한 이중 파이프라인 구조로 이루어져 있습니다. 또한, 이 논문은 세분화된 모션 패턴을 사용하여 맥락에 따라 변하는 이상 탐지를 수행하는 방법을 제시합니다.

- **Technical Details**: VelocityNet은 라이브 비디오 입력을 통해 개인별 모션 카테고리와 이상 점수를 생성하는 시스템입니다. Motion Estimation Module은 인입하는 프레임 간의 밀집 광학 흐름을 계산하며, Head Detection Module은 오클루전(occlusion)에서도 개인의 머리를 탐지하고 지역을 설정합니다. 두 모듈의 결과는 Anomaly Detection Module에서 통합되어 밀집 군중 환경에서 실시간으로 이상을 탐지할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, VelocityNet은 다양한 이상 모션 패턴을 효과적으로 탐지하며, 밀집 군중 환경에서도 실시간 성능을 발휘합니다. 이 시스템은 이전의 한계를 극복하고 실제 배치에 적합한 해석 가능한 출력을 제공합니다. 제안된 메커니즘은 군중 밀집도에 민감하게 적응하여 컨텍스트에 맞는 이상 탐지를 가능하게 합니다.



### ActivationReasoning: Logical Reasoning in Latent Activation Spaces (https://arxiv.org/abs/2510.18184)
- **What's New**: 최근 연구에서는 ActivationReasoning (AR)이라는 새로운 프레임워크를 소개했습니다. 이는 대규모 언어 모델(LLMs) 내부 공간에 명시적 논리적 추론(logical reasoning)을 통합하는 데 중점을 두고 있습니다. AR은 잠재적 표현(latent representations)을 찾고, 추론 시 활성화된 개념을 감지하며, 이를 바탕으로 논리적 추론을 수행하는 세 가지 단계를 통해 작동합니다.

- **Technical Details**: 이 연구는 SAE(sparse autoencoders)와 결합하여 원자 개념을 추출하고 이를 통해 지능적인 개념의 연쇄(compositional reasoning)를 지원합니다. AR의 구조적 접근 방식은 기존의 다차원 표현에서 발생하는 한계인 다의성(polysemy) 문제를 해결하여 더욱 명료하고 조작 가능한 구조를 제공합니다. 이 방식은 추론 작업 중 안전성 및 신뢰를 높이는 데 기여합니다.

- **Performance Highlights**: 실험 결과, AR은 다양한 설정에서 기존 방법론을 능가했습니다. 특히 다단계 추론(PrOntoQA)과 다양한 자연어 처리 작업에서 강력한 성능을 발휘하며, 모델의 신뢰성과 하위 처리 작업에서의 성공률을 크게 향상시켰습니다. AR을 통해 논리적 구조를 잠재적 활성화에 기초하여 명확하게 구현하는 것이 가능해져, 신뢰할 수 있는 AI 개발에 기여할 수 있다는 것을 보여주고 있습니다.



### Automatic Prompt Generation via Adaptive Selection of Prompting Techniques (https://arxiv.org/abs/2510.18162)
Comments:
          35 pages, 29 figures, 5 tables

- **What's New**: 이번 연구에서는 사용자로부터 제공된 추상적인 작업 설명을 기반으로 작업에 적합한 prompting 기술을 선택하고, 고품질의 프롬프트를 자동으로 생성하는 새로운 방법을 제안합니다. 이 방법은 기존의 템플릿이나 프레임워크에 의존하지 않고, semantic similarity를 가진 작업 클러스터와 함께 prompting 기술을 연결하는 지식 기반을 구성합니다. 이러한 접근 방식은 비전문가도 LLM의 기능을 효과적으로 활용할 수 있도록 지원합니다.

- **Technical Details**: 제안된 시스템은 지식 기반 구축과 프롬프트 생성의 두 가지 주요 단계로 운영됩니다. 또한, LLM을 사용하여 작업 클러스터를 정의하고, 각 클러스터에 적합한 프롬프트 기술을 연결하여 지식 기반을 형성합니다. 이 과정에서 사용자의 작업 설명을 분석하고, 적합한 프롬프트를 동적으로 생성하여, 전문 지식이 없는 사용자가 쉽게 접근할 수 있는 방안을 마련합니다.

- **Performance Highlights**: 23개의 BIG-Bench Extra Hard (BBEH) 작업에 대한 실험 평가 결과, 제안된 방법은 표준 프롬프트 및 기존의 자동 프롬프트 생성 도구와 비교하여 우수한 성능을 보여주었습니다. 평가 지표로는 산술 평균 및 조화 평균 점수를 사용하였으며, 이 연구는 프롬프트 생성의 표준화 및 간소화를 위한 기반을 다지고 있습니다.



### SafeCoop: Unravelling Full Stack Safety in Agentic Collaborative Driving (https://arxiv.org/abs/2510.18123)
- **What's New**: 이 연구는 자연어 기반의 협업 주행 시스템에서의 안전성과 보안성 문제를 체계적으로 분석한 최초의 연구입니다. 자연어를 커뮤니케이션 매체로 활용하여 드라이빙 안전성과 효율성을 향상하려는 최근 경향에 주목하며, 새로운 위험 요소들을 조명합니다. 특히, 메시지 손실, 홀로그램 생성, 의미 조작과 같은 언어 통신의 취약성을 지적하고 이를 해결하기 위한 방안을 제시합니다.

- **Technical Details**: 다양한 공격 전략에 대한 포괄적인 분류 체계를 개발하여 연계 차단, 중계 및 재생 간섭, 콘텐츠 스푸핑 및 다중 연결 위조 등 여러 공격 경로를 분석합니다. 시스템에서 생성된 각 주행 에이전트는 Multi-modal Large Language Models(MLLMs)를 기반으로 작동하며, 두 개의 핵심 모듈인 추론 모듈(Ri)과 행동 모듈(Di)을 갖추고 있습니다. 나아가, 본 논문에서는 언어 기반의 지역적 참조 변환 문제를 해결하기 위해 Agentic Transformation Function(ATF)을 도입합니다.

- **Performance Highlights**: 제안된 방어 체계인 SafeCoop는 CARLA 시뮬레이터에서 32개의 중요한 시나리오에서 테스트되고, 악의적인 공격 하에서도 69.15%의 주행 점수 향상과 67.32%의 F1 점수를 달성하였습니다. 이는 언어 기반 협업 주행에서의 취약성을 확인하고, 이를 감지하는 데 있어 뛰어난 성능을 발휘함을 보여줍니다. 이 연구는 안전하고 신뢰할 수 있는 언어 기반 협업을 위한 향후 연구 방향을 제시합니다.



### Latent Discrete Diffusion Models (https://arxiv.org/abs/2510.18114)
- **What's New**: 이번 연구는 언어와 기타 범주형 데이터에 대한 이산 확산(discrete diffusion)을 조사하고, 마스킹된 디노이저(masked denoisers)의 일반적인 한계를 다룹니다. 기존 연구의 문제점을 해결하기 위해 저자들은 Latent Discrete Diffusion Models (LDDMs)을 제안했으며, 이는 토큰에 대한 마스킹된 이산 확산과 잠재 임베딩(embeddings)에 대한 연속 확산을 결합하여 제공합니다. 이러한 접근 방식은 직접적인 위치 간의 의존성을 개선하여 신뢰성과 생성을 향상시킵니다.

- **Technical Details**: LDDMs은 FUJI-LDDMs와 SEQ-LDDMs의 두 가지 변형으로 구분됩니다. FUJI-LDDMs는 각 단계에서 토큰과 잠재 변수를 동시에 조화롭게 디노이징하는 반면, SEQ-LDDMs는 먼저 잠재 변수를 해결하고 이를 기초로 전체 이산 체인을 조건부로 다룹니다. 또한, 연구진은 ELBO(Variational Lower Bound) 스타일의 학습 목표와 안정적인 최적화를 위한 설계 선택에 대해 논의합니다.

- **Performance Highlights**: 실험 결과, LDDMs는 최신 마스킹된 이산 확산 기준선에 비해 무조건적 생성 지표에서 상당한 개선을 이룩했습니다. 특히 낮은 샘플링 예산(sampling budget)에서도 효과적인 성능을 보이며, 이 단계에서 많은 토큰을 한 번에 열 수 있는 방식으로 유연함을 보여줍니다. 이러한 결과는 LDDMs이 디지털 텍스트 생성을 위한 보다 효과적인 도구임을 시사합니다.



### From AutoRecSys to AutoRecLab: A Call to Build, Evaluate, and Govern Autonomous Recommender-Systems Research Labs (https://arxiv.org/abs/2510.18104)
- **What's New**: 추천 시스템(RecSys) 연구가 모델과 평가 기술의 발전을 이루었지만, 연구 프로세스 자체의 자동화는 거의 간과하고 있다는 주장을 하고 있습니다. AutoRecLab이라는 새로운 패러다임을 제안하며 문제 구상, 문헌 분석, 실험 설계와 실행, 결과 해석, 원고 작성, 기록 보존까지 전 과정을 자동화하는 연구 환경을 목표로 하고 있습니다. 이러한 자동화의 필요성은 최근 자동화 과학 분야의 발전 특히 Multi-Agent AI Scientist와 AI Co-Scientist 시스템에서 기인하고 있습니다.

- **Technical Details**: 추천 시스템 분야의 기존 자동화 도구인 AutoRecSys는 알고리즘 선택과 하이퍼파라미터 조정에 국한되어 있습니다. 그러나 AutoRecLab은 LLM 중심의 아이디어 생성 및 보고서 작성을 결합해 자동 실험을 포함하는 전면적인 프로토타입을 개발하는 것을 지향합니다. 이 연구는 기존의 인간 기여 없이도 반복 가능한 RecSys 발견을 도출하는 기준 및 대회 수립, AI 생성 제출물에 대한 투명한 리뷰 공간 마련 등 다양한 노력을 포함하고 있습니다.

- **Performance Highlights**: 최근 Sakana의 AI Scientist와 구글 AI Co-Scientist의 발전을 통해 완전 자동화된 연구 프로세스가 가능해짐을 보여주고 있습니다. AI Scientist는 최소한의 인간 개입으로 연구 논문을 작성하는 성과를 내었으며, 실제 동료 평가에 통과한 첫 AI 생성 논문을 기록했습니다. 이러한 발전은 연구 자동화의 가속화를 나타내며, RecSys 커뮤니티가 이를 뒤따라야 할 필요성을 강조합니다.



### Enhancing mortality prediction in cardiac arrest ICU patients through meta-modeling of structured clinical data from MIMIC-IV (https://arxiv.org/abs/2510.18103)
Comments:
          38 pages, 5 figures, 2 tables, 3 appendices

- **What's New**: 이 연구는 집중 치료실(ICU)에서의 병원 내 사망률을 조기 예측하기 위한 기계 학습 모델을 개발하고 평가하여, 구조화된 임상 데이터와 비구조화된 텍스트 정보(퇴원 요약 및 방사선 보고서)를 통합하였습니다. LASSO와 XGBoost를 활용하여 특징을 선택하고, 다변량 로지스틱 회귀 모델을 통해 예측 성능을 높였습니다. 비구조화된 데이터를 포함한 최종 로지스틱 회귀 모델은 AUC 0.918을 달성하였으며, 이는 구조화된 데이터만 사용했을 때의 0.753에 비해 22% 개선된 결과입니다.

- **Technical Details**: 이 연구에서는 MIMIC-IV 데이터베이스에서 구조화된 데이터와 비구조화된 텍스트 정보를 통합하여 ICU에 입원한 심장마비 환자의 병원 내 사망률을 예측하는 해석 가능한 모델을 개발했습니다. 예측 정확도와 투명성을 향상시키기 위해 LASSO와 XGBoost를 통해 특징을 선택하고, TF-IDF 및 BERT 임베딩을 활용하여 텍스트 기능을 통합하였습니다. 새롭게 개발된 모델은 명확한 회귀 계수를 통해 해석 가능성을 제공합니다.

- **Performance Highlights**: 비구조화된 텍스트 데이터를 포함함으로써 예측 성능이 크게 향상되었습니다. 구조화된 데이터만으로는 AUC 0.75의 성과를 보였으나, 텍스트 기능을 추가하여 AUC 0.92로 개선되었습니다. 결정 곡선 분석(decision curve analysis)을 통해 0.2-0.8의 여러 임계 확률 범위에서 모형의 우수한 표준화된 순 이익(standardized net benefit)이 입증되었습니다.



### Accelerating Vision Transformers with Adaptive Patch Sizes (https://arxiv.org/abs/2510.18091)
Comments:
          Project page at this https URL

- **What's New**: Adaptive Patch Transformer (APT)는 시각적 정보를 효과적으로 처리하기 위해 동일한 이미지 내에서 서로 다른 패치 크기를 사용하는 혁신적인 구조입니다. APT는 균일하게 크기가 정해진 패치로 입력 이미지를 나누는 전통적인 방법의 한계를 극복하여 근본적으로 입력 토큰의 수를 줄여줍니다. 이를 통해 APT는 더 많은 계산 효율성을 가지며, 고해상도 이미지의 처리 성능을 유지할 수 있습니다.

- **Technical Details**: APT는 여러 스케일에서 엔트로피를 계산하여 동적으로 큰 패치와 작은 패치를 적절히 할당합니다. 텍스처가 단조로운 지역은 큰 패치로, 구성 요소가 복잡한 지역은 작은 패치로 표현하여 그림의 정보 중복을 줄이는 방식입니다. 또한, APT는 제로 초기화된 MLP를 사용하여 패치의 임베딩을 결합함으로써 네트워크 손상 없이 수렴할 수 있도록 합니다.

- **Performance Highlights**: APT는 ViT 학습 및 추론 속도를 최대 40% 가량 향상시키며, 특히 고해상도 이미지와 큰 모델에서 성능 향상이 두드러집니다. 실험 결과, APT는 데이터셋에 따라 기존의 ViT 성능을 유지하며, 시각적 질문 응답, 객체 탐지, 의미론적 분할과 같은 다양한 이미지 이해 작업을 성공적으로 수행하는 것으로 나타났습니다.



### R2BC: Multi-Agent Imitation Learning from Single-Agent Demonstrations (https://arxiv.org/abs/2510.18085)
Comments:
          9 pages, 6 figures

- **What's New**: 이 연구에서는 다중 에이전트 시스템에서의 모방 학습을 확장하는 새로운 접근 방식을 소개합니다. 우리가 제안한 Round-Robin Behavior Cloning (R2BC)은 단일 인간 조작자가 여러 로봇 팀에 효과적으로 교육할 수 있게 해줍니다. 특히, 이 방법은 한 번에 하나의 에이전트에 대해 시퀀스 방식으로 시연을 제공할 수 있도록 합니다.

- **Technical Details**: R2BC는 각 에이전트에 대해 온라인으로 개별 시연을 제공하여 다중 에이전트 정책을 반복적으로 개선하는 방식입니다. 이 방법은 에이전트들이 공유 행동 공간에서 동시에 시연을 수행할 필요 없이, 각 에이전트의 기존 학습 정책을 실행하도록 허용합니다. 연구에 사용된 환경은 네 가지 시뮬레이션 다중 에이전트 작업으로 구성되어 있습니다.

- **Performance Highlights**: R2BC 방법은 중앙 집중식 행동 클로닝에 비해 성능이 3.25배 및 5.9배 향상됨을 증명했습니다. 이는 인간동안 수행된 실제 시연을 기반으로 한 물리적 로봇 작업에서도 나타났습니다. 이 작업을 통해 우리는 단일 에이전트 시연만으로도 효과적인 다중 에이전트 정책 학습이 가능하다는 것을 보여주었습니다.



### RL-Driven Security-Aware Resource Allocation Framework for UAV-Assisted O-RAN (https://arxiv.org/abs/2510.18084)
Comments:
          6 pages

- **What's New**: 본 논문은 UAV(무인 항공기)를 활용한 O-RAN(개방형 무선 접속망)에서의 동적 자원 할당을 위한 새로운 강화 학습(ML) 기반 프레임워크를 제안합니다. 이 프레임워크는 보안, 지연(latency), 에너지 효율성 간의 상충 관계를 명확하게 조정하여 SAR(탐색 및 구조) 작업의 요구를 충족하도록 설계되었습니다. 특히, 실시간으로 네트워크 동적 상황에 적응하여 신뢰성 있는 통신을 보장하는 데 중점을 두고 있습니다.

- **Technical Details**: UAV 릴레이는 지상 사용자(GU)와 O-RU 간 데이터를 중계하며, 각 UAV는 제한된 에너지를 효과적으로 활용하기 위해 힘을 기울여야 합니다. 본 논문에서는 보안 표준인 DES, AES 및 RSA와 같은 암호화 알고리즘을 적용하여 데이터 전송의 기밀성을 보장합니다. 이러한 알고리즘은 비밀번호 길이와 블록 크기에 따라 성능이 달라지며, 네트워크의 에너지 효율과 지연을 최적화하기 위한 기초 자료를 제공합니다.

- **Performance Highlights**: 시뮬레이션 결과, 제안된 RL 기반 프레임워크는 기존의 휴리스틱(heuristic) 방법에 비해 보안과 에너지 효율성을 향상시키면서도 초저 지연을 유지하는 데 성공했습니다. 이는 SAR 시나리오에서 통신 성능을 높이고 재난 관리에 유용한 효율적인 솔루션을 제공함을 의미합니다. 따라서 본 연구는 UAV가 통합된 O-RAN의 가능성을 한층 더 확장하는 기반이 될 것으로 기대됩니다.



### Any-Depth Alignment: Unlocking Innate Safety Alignment of LLMs to Any-Depth (https://arxiv.org/abs/2510.18081)
- **What's New**: 이 논문에서는 Large Language Models (LLMs)의 얕은 정렬(shallow alignment)의 문제를 해결하기 위한 Any-Depth Alignment (ADA)를 제안합니다. 기존 LLM들이 해로운 질의에 대해 즉각적으로 거부 응답을 제공하지만, 해로운 연속 진행이 시작되면 이 보호 기능이 무너지므로, 이를 극복할 수 있는 방법이 필요하다는 점을 강조합니다. ADA는 모델의 능력을 활용하여 해로운 생성의 모든 깊이에서 안전성을 확보할 수 있는 효율적인 방어 기제를 구축합니다.

- **Technical Details**: ADA는 모델의 조수 헤더 토큰에서 집중된 정렬을 기반으로 하여, 중간에 이 토큰들을 재투입(reintroduce)함으로써 모델이 해로운 내용을 다시 평가하고 어떤 시점에서든 거부하도록 유도합니다. 이러한 접근 방식은 수많은 개방형 모델 패밀리(Llama, Gemma, Mistral 등)에서 이루어졌으며, 모델 매개변수의 변경 없이 안전한 성능을 달성합니다. 안전 토큰(Safety Tokens)이라는 새로운 개념을 통해 LLM의 내부 안전 판단을 연결하고 이를 통해 유사한 연쇄 공격에 대한 강력한 저항력을 보입니다.

- **Performance Highlights**: ADA는 다양한 적대적 프리필 공격에 대해 100%에 가까운 거부율을 기록할 수 있으며, 이는 수십에서 수천 개의 토큰에 이르는 도전적인 공격에 대해서도 마찬가지입니다. 또한, GCG, AutoDAN, PAIR, TAP 같은 주요 적대적 프롬프트 공격의 평균 성공률을 3% 미만으로 낮추어, 유용성 유지 및 과도한 거부 없이 성능을 보장합니다. 이러한 성능은 모델이 추가적인 도구(benign 또는 adversarial)로 조정되었을 경우에도 유지됩니다.



### R2L: Reliable Reinforcement Learning: Guaranteed Return & Reliable Policies in Reinforcement Learning (https://arxiv.org/abs/2510.18074)
Comments:
          27 pages

- **What's New**: 이번 연구에서는 강화 학습(reinforcement learning, RL)에서 신뢰할 수 있는 정책을 결정하는 문제를 다룹니다. 특히 불확실성(uncertainty) 하에서의 최적화와 성능 보장(performance guarantee)의 필요성에 초점을 맞추었습니다. 기존의 RL 알고리즘은 평균 수익률(maximizing expected return)을 극대화하는 데 중점을 두지만, 많은 실제 응용에서는 높은 평균 성능뿐만 아니라 성공 확률(guaranteed probability of success)도 요구합니다.

- **Technical Details**: 우리는 누적 수익이 정해진 임계값을 초과할 확률을 극대화하는 것을 목표로 하는 새롭고 독창적인 문제 정의를 제안합니다. 이 신뢰할 수 있는 RL 문제는 상태(state) 보강(state-augmented) 표현을 통해 표준 RL 문제로 재구성될 수 있음을 시연하였습니다. 이를 통해 기존의 RL 및 딥 RL(deep RL) 알고리즘을 신규 알고리즘 프레임워크 없이도 사용할 수 있습니다.

- **Performance Highlights**: 우리는 신뢰할 수 있는 라우팅(reliable routing) 문제를 예로 들어, 기대 이동 시간을 최소화하는 것이 아니라 주어진 시간 예산 내에 목적지에 도달할 확률을 극대화하는 목표를 세웁니다. 수치 실험 결과에 따르면, 제안된 공식화는 효율성과 신뢰성의 균형을 효과적으로 맞춘 정책을 생성함에 있어 신뢰할 수 있는 RL이 확률적이고 안전이 중요한 환경에서 응용될 수 있는 잠재력을 강조합니다.



### Fine-tuning Flow Matching Generative Models with Intermediate Feedback (https://arxiv.org/abs/2510.18072)
- **What's New**: AC-Flow는 텍스트-이미지 생성에서의 중간 피드백을 활용하여 플로우 기반 생성 모델을 안정적으로 미세 조정하기 위한 새로운 액터-크리틱 프레임워크입니다. 이는 리워드 쉐이핑, 이중 안정성 메커니즘 및 일반화된 크리틱 가중치 평가 등의 혁신적인 접근 방식을 통해 라벨링 및 피드백 문제를 해결합니다. 이로 인해 플로우 매칭 모델의 안정적인 성능 개선과 고급 품질 보장을 달성합니다.

- **Technical Details**: AC-Flow는 세 가지 주요 요소로 구성됩니다: (1) 리워드 쉐이핑을 통한 안정적인 중간 상태 가치 학습 및 그래디언트 제어, (2) 크리틱의 성숙을 허용하는 와운업 단계 및 파괴적 정책 업데이트 방지를 위한 어드밴티지 클리핑을 결합한 이중 안정성 메커니즘, (3) Wasserstein 정규화를 통해 모델 다양성을 유지하면서 전통적인 리워드 가중치 방법을 확장하는 일반화된 크리틱 가중치 평가 방식. 이를 통해 AC-Flow는 텍스트-이미지 정렬 작업에서 최첨단 성능을 달성합니다.

- **Performance Highlights**: AI 뉴스레터에서 AC-Flow의 실험 결과는 Stable Diffusion 3 통합을 통해 부각됩니다. AC-Flow는 인간 선호 모델에 대한 일반화뿐만 아니라, 높은 품질의 생성 출력 및 높은 다양성을 유지하면서도 탁월한 성과를 보여줍니다. 이러한 결과는 계산적으로 효율적인 크리틱 모델을 사용하여 생성 품질, 다양성 및 안정성을 타협하지 않고 플로우 모델을 미세 조정할 수 있음을 보여줍니다.



### SPACeR: Self-Play Anchoring with Centralized Reference Models (https://arxiv.org/abs/2510.18060)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구에서는 자율주행 차량(AVs)의 동작을 인간처럼 보이도록 하기 위해 SPACeR라는 새로운 프레임워크를 제안합니다. SPACeR는 미리 훈련된 토크나이즈 모델을 중앙 집중식 참조 정책으로 활용하여 분산형 자기 플레이(self-play) 학습을 안내합니다. 이를 통해 인간의 주행 분포를 보다 효과적으로 반영하면서도 높은 확장성을 유지할 수 있는 정책을 개발합니다.

- **Technical Details**: SPACeR는 토크나이즈된 오토회귀 모션 모델을 중심으로 한 정책을 이용해 자율주행 시뮬레이션을 수행합니다. 이 정책은 KL 발산(KL divergence)과 확률 보상을 제공하여 자기 플레이 정책의 학습을 인간과 유사하게 조정합니다. 이를 통해, 임의의 자율주행 시나리오에서 정책의 일관성과 인간과의 상호작용 능력을 높일 수 있습니다.

- **Performance Highlights**: Waymo Sim Agents Challenge에서 SPACeR는 기존 자기 플레이 RL 방법보다 현실감 및 인간 유사성을 크게 향상시켰습니다. 이 방법은 추론(inference) 속도가 최대 10배 빠르고, 파라미터 크기는 50배 더 작은 경량의 모델로, 자율주행 정책 테스트를 위한 새로운 패러다임을 확립했습니다. 다양한 계획자(planner)에 대해 신속하고 확장 가능한 교통 시뮬레이션을 통해 주행 계획 품질을 효과적으로 측정할 수 있음을 보여주었습니다.



### Adaptive Divergence Regularized Policy Optimization for Fine-tuning Generative Models (https://arxiv.org/abs/2510.18053)
Comments:
          30 pages

- **What's New**: 이번 논문에서는 적응형 발산 정규화 정책 최적화(Adaptive Divergence Regularized Policy Optimization, ADRPO)를 제안하였습니다. ADRPO는 보상 최적화 중 모델 안정성을 유지하면서, 다양한 품질의 데이터에 따라 정규화 강도를 조정하여 탐색(exploration)과 활용(exploitation) 간의 균형을 맞추는 혁신적인 접근 방식입니다. 이는 특히 텍스트-이미지 생성(Text-to-Image Generation)에서 기존의 오프라인 방법이나 고정 정규화 방식을 사용하는 온라인 방법보다 향상된 성능을 보여줍니다.

- **Technical Details**: ADRPO는 샘플별로 정규화 강도를 동적으로 조정하여, 고급 샘플에 대해서는 정규화를 줄이고 저급 샘플에는 강한 정규화를 적용합니다. 이 방식은 Wasserstein-2 정규화를 사용하여 흐름 매칭 생성을 위한 모델을 최적화하며, 2B 파라미터 모델이 4.8B 및 12B 파라미터 모델보다 우수한 성능을 보이는 것을 보여줍니다. 또한, ADRPO는 KL 정규화가 이루어지는 텍스트 전용 LLM(대형 언어 모델) 및 다중 모달 추론 모델에도 일반화 가능하여, 기존의 온라인 RL 방법을 향상시킵니다.

- **Performance Highlights**: ADRPO는 LLM 미세 조정에서 지역 최적(local optimum)을 탈출할 수 있는 능력을 보이며, 다중 모달 오디오 추론에서 GRPO보다 뛰어난 성능을 보여줍니다. 7B 파라미터 모델이 상업적으로 더 큰 모델인 Gemini 2.5 Pro 및 GPT-4o Audio를 초월하였고, 이는 다양한 생성 아키텍처 및 모달리티 전반에 걸쳐 탐색-활용 문제를 해결할 수 있는 효과적인 솔루션을 제공합니다. 이러한 결과는 다양한 데이터 품질에 맞춰 탐색과 활용을 조정하는 능력 덕분입니다.



### Measure-Theoretic Anti-Causal Representation Learning (https://arxiv.org/abs/2510.18052)
- **What's New**: 이번 연구는 Anti-Causal Invariant Abstractions (ACIA)라는 새로운 프레임워크를 제안합니다. ACIA는 레이블이 특징을 유발하는 비인과적 설정(anti-causal setting)에서의 표현 학습을 다룹니다. 이 방법은 저수준 표상이 레이블이 관측을 생성하는 방식을 캡처하고, 고수준 표상은 환경에 따라 변화하는 안정적인 인과 패턴을 학습합니다.

- **Technical Details**: ACIA는 비인과적 표현 학습을 위한 측정 이론적 프레임워크로, 기존 접근 방식의 주요 한계를 극복합니다. 완전 및 불완전한 개입을 수용하는 개입 커널(interventional kernels)을 통해 명시적 인과 구조에 대한 의존성을 제거하며, 고차원 데이터를 효과적으로 처리할 수 있는 이론적 보장을 제공합니다. 이 연구는 ACIA의 이론적 결과가 훈련 환경과 보지 못한 환경 간의 성능 간극을 좁힌다는 것을 입증합니다.

- **Performance Highlights**: ACIA는 합성 및 실세계 의료 데이터셋에서 기존의 최신 방법들보다 일관되게 높은 정확도와 불변성 지수를 보여주었습니다. 이러한 성능 향상은 ACIA가 강건한 비인과적 학습에 효과적임을 확인시켜 줍니다. 실험 결과는 ACIA가 더 적은 초기 정보 요구 사항을 가지면서도 다른 최신 인과 표현 학습 방법들보다 우수하다는 것을 보여줍니다.



### Language Models as Semantic Augmenters for Sequential Recommenders (https://arxiv.org/abs/2510.18046)
- **What's New**: LaMAR(언어 모델 보강 추천)는 LLM(대규모 언어 모델)의 추론 능력을 활용하여 사용자의 상호작용 이력을 풍부하게 만들기 위해 자동으로 생성된 의미 신호를 통합하는 새로운 프레임워크입니다. 이 시스템은 기존 메타데이터로부터 사용자의 의도와 아이템 간의 관계를 추론하여 부가적인 문맥 신호를 생성함으로써 추천 시스템의 성능을 향상시키는 데 중점을 둡니다.

- **Technical Details**: 이 프레임워크는 두 가지 주요 단계로 구성됩니다. 첫 번째 단계에서는 LLM을 통해 사용자-아이템 시퀀스를 보강하는 의미 신호를 생성하는 파이프라인을 구축하고, 두 번째 단계에서는 이러한 신호를 추천 시스템에 통합하여 LLM의 행동을 추천 목표와 일치하게 조정합니다. 예를 들어, 사용자의 과거 상호작용을 기반으로 사용자의 숨겨진 의도나 주제를 추론하여 더욱 풍부한 문맥 정보를 생성합니다.

- **Performance Highlights**: LaMAR는 여러 공공 데이터 셋에서 광범위한 실험을 통해 성능 개선을 입증하였습니다. LLM이 생성한 신호를 기존 Sequential Recommendation 모델에 통합함으로써 랭킹 메트릭에서 지속적인 성과향상을 얻었고, 생성된 신호는 높은 의미적 다양성과 독창성을 보였습니다. 이러한 방식으로 추천 시스템의 지식 기반을 확장하고 사용자 선호도를 더욱 깊게 이해할 수 있게 되었습니다.



### Cross-Domain Long-Term Forecasting: Radiation Dose from Sparse Neutron Sensor via Spatio-Temporal Operator Network (https://arxiv.org/abs/2510.18041)
- **What's New**: 본 논문에서는 Spatio-Temporal Operator Network (STONe)를 소개하며, 이는 이질적인 도메인 간의 안정적인 함수 매핑을 학습하는 비자기회귀 신경 연산자입니다. STONe는 희소한 지상에서의 중성자 측정을 통해 고고도 방사선량 필드를 직접 유추하고, 이를 통해 연산자 학습이 공유 도메인을 넘어 일반화될 수 있음을 보여줍니다. 이러한 접근은 전통적인 연산자 학습 방식이 도메인 정렬이나 자기회귀 전파에 의존한다는 통념에 도전합니다.

- **Technical Details**: STONe은 23년간의 전 세계 중성자 데이터를 활용하여 훈련되었으며, ms 대기에서 180일 예측을 정확하게 수행합니다. 이 모델은 방사선 예측을 위해 이질적인 입력과 출력을 갖는 도메인 간의 논리적 변환을 정의하며, 반복적인 재발 없이도 긴 예측 범위에서 안정성을 유지합니다. 이는 GCR과 같은 복잡한 물리적 현상의 예측에서 유용한 원칙을 제시합니다.

- **Performance Highlights**: STONe의 성능은 시간이 지남에 따라 정확한 예측을 수행할 수 있는 잠재력을 보여주며, 특히 항공 및 에너지 시스템 등 복잡한 시공간 필드를 실시간으로 예측하는 데 기여할 수 있습니다. 기존의 모델들과 달리 STONe은 서로 다른 물리적 도메인 간의 예측을 허용함으로써 사용자의 예측 정확성을 높일 수 있습니다. 결과적으로, STONe은 전통적인 기상 예측 패러다임에 대한 중요한 변화를 나타냅니다.



### TriggerNet: A Novel Explainable AI Framework for Red Palm Mite Detection and Multi-Model Comparison and Heuristic-Guided Annotation (https://arxiv.org/abs/2510.18038)
Comments:
          17 pages, 9 figures

- **What's New**: 이번 연구는 레드 팜 마이트(red palm mite) 감염을 조기 식별하고 효과적으로 관리하기 위한 머신 러닝(ML) 모델을 평가하고 비교하는 데 초점을 맞추고 있습니다. TriggerNet이라는 새로운 해석 가능한 AI 프레임워크는 Grad-CAM, RISE, FullGrad 및 TCAV를 통합하여 심층학습 모델의 시각적 설명을 생성합니다. 이 연구는 레드 팜 마이트 감염 문제를 해결하기 위해 다양한 식물 이미지와 고급 딥러닝 모델을 사용하고 있습니다.

- **Technical Details**: 연구에서는 11종의 식물(예: Arecanut, Date Palm, Coconut Palm 등)에서 RGB 이미지를 사용하여 모델을 훈련했습니다. CNN, EfficientNet, MobileNet, ViT, ResNet50, InceptionV3와 같은 최신 딥러닝 모델과 Random Forest, SVM, KNN과 같은 머신 러닝 분류기를 사용하여 식물 분류를 수행했습니다. 질병 분류는 건강한 식물과 다양한 질병 상태(예: Yellow Spots, Reddish Bronzing 등)로 나뉘어 진행되었습니다.

- **Performance Highlights**: Snorkel을 사용하여 질병 클래스를 효율적으로 레이블링하였으며, 이는 수작업 주석 시간을 줄이고 데이터셋의 신뢰성을 높였습니다. TriggerNet은 Red palm mite 감염의 조기 식별에 있어 맥락을 바탕으로 한 시각적 설명을 제공하여 실용성을 높이는 효과를 보여주고 있습니다. 이러한 기법은 능률화된 농업 관리에 기여할 수 있는 중요한 도구가 될 것입니다.



### SAVANT: Semantic Analysis with Vision-Augmented Anomaly deTection (https://arxiv.org/abs/2510.18034)
Comments:
          8 pages, 5 figures

- **What's New**: SAVANT는 자율 주행 시스템의 이상 감지에서 구조화된 추론 프레임워크를 제공합니다. 이 연구는 Vision Language Models (VLMs)의 비효율적인 접근 방식을 극복하며, 고급 분석을 통해 기존의 모델보다 더 높은 정확도를 달성합니다. SAVANT는 입력 이미지로부터 복잡한 장면을 네 개의 의미 레이어(Street, Infrastructure, Movable Objects, Environment)로 나누어 체계적으로 평가함으로써 이상 상황을 효과적으로 감지합니다.

- **Technical Details**: SAVANT의 핵심 구성 요소는 두 단계의 파이프라인으로, 첫 번째 단계에서는 구조화된 장면 설명을 추출하고 두 번째 단계에서는 다중 모달 평가를 수행합니다. 이 구조는 기존 등록된 객체들이 맥락적으로 부적절한 배치에 있는지를 감지하는 이진 분류 작업으로 이상 감지를 정의합니다. 여러 의미 레이어를 통해 교통 장면에 대한 심도 있는 분석을 제공하며, 이는 자율 주행 시스템의 안전성을 높이는 데 기여합니다.

- **Performance Highlights**: SAVANT는 실제 주행 시나리오에서 89.6%의 재현율과 88.0%의 정확성을 달성하며, 기존 비구조적 기준선에 비해 현저한 성능 향상을 보여줍니다. 또한 7B 파라미터의 오픈 소스 모델인 Qwen2.5VL을 통해 90.8%의 재현율과 93.8%의 정확성을 달성하여, 비용 문제 없이 로컬 배포를 가능하게 합니다. 이 연구는 9,640개의 실제 주행 이미지에 대한 자동 라벨링을 제공함으로써 데이터 부족 문제를 해결하는 실질적인 경로를 제시합니다.



### From Local to Global: Revisiting Structured Pruning Paradigms for Large Language Models (https://arxiv.org/abs/2510.18030)
Comments:
          16 pages, 4 figures

- **What's New**: 본 연구에서는 GISP(Global Iterative Structured Pruning)라는 새로운 접근 방식을 소개합니다. 이 방법은 레이어별 최적화가 아닌 모델 수준의 손실을 기반으로 구조적 프루닝을 수행하며, 후보 구조를 제거하여 고밀도 아키텍처를 생성합니다. GISP는 고상태에서의 정확도 유지를 위해 반복적인 일정 조정을 통해 안정된 성능을 보장하며, '한 번 잘라내고 많이 배포하기(prune-once, deploy-many)' 가능한 작업 흐름을 지원합니다.

- **Technical Details**: GISP는 주목할 만한 결과를 얻기 위해 레이어 내의 주의 헤드(attention heads)와 MLP 채널을 제거하는 사후 교육(post-training) 방법입니다.(Dense 시차로 이어지는 로스 기반 중요한 가중치를 집계하여 구조 수준에서 블록 단위 정규화를 사용합니다. 이 접근법은 초기 프루닝 보정이 불필요하며, 반복적인 절차가 전체 모델 품질을 유지하는 데 중요한 역할을 합니다. 또한, GISP는 실험을 통해 여러 모델에 대해 정량적인 개선을 보여줍니다.

- **Performance Highlights**: Llama2 및 Mistral 모델에서의 실험 결과, GISP는 WikiText-2에 대한 perplexity를 꾸준히 줄이고 다운스트림 정확도를 개선하는 것으로 나타났습니다. 특히 40-50%의 희소성(sparsity)에서 강력한 성과를 보였습니다. 또한 DeepSeek-R1-Distill-Llama-3-8B 모델에 대해, 작업 정렬 보정(task-aligned calibration)은 정확한 일치 정확도를 대폭 향상시키는 데 기여하였습니다.



### DynaQuery: A Self-Adapting Framework for Querying Structured and Multimodal Data (https://arxiv.org/abs/2510.18029)
Comments:
          15 pages, 2 figures, 10 tables. Source code and experimental artifacts are available at: this https URL . The 'DynaQuery-Eval-5K' benchmark, introduced in this work, is also publicly available at: this https URL

- **What's New**: DynaQuery는 복잡한 하이브리드 데이터베이스에 대해 자연어 쿼리를 지원하기 위한 통합되고 자기 적응형(framework) 프레임워크로 제안되었습니다. 이 시스템은 Schema Introspection and Linking Engine(SILE)을 중심으로 하여 스키마 링크를 쿼리 계획의 1급 과정으로 끌어올립니다. 기존의 비구조적 Retrieval-Augmented Generation(RAG) 패러다임의 한계를 극복하기 위한 대안으로, DynaQuery는 보다 신뢰할 수 있는 데이터베이스 인터페이스 개발을 위한 기초를 제공합니다.

- **Technical Details**: DynaQuery는 두 가지 핵심 작업, 즉 NL-to-SQL과 멀티모달 쿼리를 다루는 해결책을 제공합니다. NL-to-SQL 작업에서는 자연어 쿼리 Qs를 주어졌을 때 해당 쿼리에 대한 적절한 SQL 쿼리 S를 생성하는 것이 목표입니다. 또한, 멀티모달 쿼리 작업에서는 DB에서 멀티모달 데이터의 포인터를 포함하는 열을 기준으로 Qm과 함께 레코드의 하위 집합 Rm을 생산합니다. 각 작업의 효과적인 수행을 위해 DynaQuery는 기존의 시스템 구조와 기술적 문제를 명확히 정의하였습니다.

- **Performance Highlights**: DynaQuery는 구조적 지각을 기반으로 한 아키텍처가 비구조적 RAG보다 우수하다는 것을 다양한 벤치마크를 통해 입증하였으며, SCHEMA_HALLUCINATION 같은 재앙적 맥락 실패 모드를 거의 제거합니다. 결국, DynaQuery는 복잡한 데이터베이스 시스템을 위한 신뢰할 수 있고 적응 가능한 솔루션을 제공하여 현대 데이터 액세스의 복잡성을 해결합니다. 최종적으로, 이 연구는 데이터 시스템의 신뢰성과 일관성을 높일 수 있는 중요한 원칙들을 제시합니다.



### Is Multilingual LLM Watermarking Truly Multilingual? A Simple Back-Translation Solution (https://arxiv.org/abs/2510.18019)
- **What's New**: 이번 연구는 다국어 워터마킹(multilingual watermarking)의 한계와 문제점을 지적하고, 기존 방법들이 고자원(high-resource) 언어에만 한정되어 평가되고 있음을 보여줍니다. 특히, 기존의 기술들이 매체와 저자원 언어에서 번역 공격에 대한 강인성을 유지하지 못한다는 점을 강조합니다. 이를 극복하기 위해 STEAM이라는 새로운 감지 방법을 제안했습니다.

- **Technical Details**: STEAM은 역번역(back-translation) 기반의 감지 방법으로, 번역 과정에서 손실된 워터마크 강도를 복원합니다. 이 방법은 모든 워터마킹 기술과 호환되며, 다양한 토크나이저(tokenizer) 및 언어에서도 강인성을 유지합니다. 비침투적(non-invasive)이고 새로운 언어로 쉽게 확장할 수 있는 장점이 있습니다.

- **Performance Highlights**: 17개 언어에서 평균 +0.19 AUC 및 +40%p TPR@1%의 성능 향상을 보였습니다. STEAM은 다양한 언어에서 더욱 공정한 워터마킹을 위한 간단하고 강력한 경로를 제공합니다. 이러한 성과는 다국어 워터마킹의 혁신적 접근 방식을 제시합니다.



### BadScientist: Can a Research Agent Write Convincing but Unsound Papers that Fool LLM Reviewers? (https://arxiv.org/abs/2510.18003)
- **What's New**: 본 논문은 AI 기반의 출판 검토 시스템과 LLM 연구 보조 도구의 융합으로 발생하는 위기적 취약점을 조사합니다. 이들은 AI 생성 연구물이 AI 리뷰어에 의해 평가되는 fully automated publication loop에 놓일 수 있기 때문에 연구의 진실성을 심각하게 위협합니다. 연구팀은 이러한 문제를 탐구하기 위해 BadScientist라는 프레임워크를 개발하였습니다.

- **Technical Details**: BadScientist는 펩타기 위한 전략을 채택하여 실제 실험을 수행할 필요 없이 논문을 작성하는 AI 생성 에이전트를 평가합니다. 본 연구는 ICLR 2025 데이터를 활용하여 LLM 리뷰어가 평가하는 프레임워크를 구축하고, 농도 경계 및 보정 분석을 통해 형식적 에러 보장을 제공합니다. 연구 결과, 제작된 논문이 LLM 리뷰어에서 최대 82%의 수락률을 달성하는 등 심각한 취약점을 드러냅니다.

- **Performance Highlights**: 연구에서 확인된 concern-acceptance conflict는 리뷰어들이 진실성 문제를 지적하면서도 수락 점수를 부여하는 경우가 많음을 보여줍니다. 해결 방안으로 제시된 Review-with-Detection (ReD) 및 Detection-Only (DetOnly) 전략도 미미한 개선만을 이끌며, 감지 정확도가 본질적으로 임의 결과를 넘지 못했습니다. 이는 현재 AI 기반의 검토 시스템에 대해 근본적인 한계를 드러내며, 과학 출판의 방어 강화가 시급함을 강조합니다.



### SimBA: Simplifying Benchmark Analysis Using Performance Matrices Alon (https://arxiv.org/abs/2510.17998)
Comments:
          EMNLP 2025 Findings

- **What's New**: 이번 논문에서는 SimBA라는 새로운 프레임워크를 제안하여 언어 모델(LM) 평가에 대한 통찰력을 제공합니다. SimBA는 데이터셋과 모델의 관계를 분석하는 'Stalk', 대표적인 데이터셋 하위 집합을 발견하는 'Prowl', 그리고 모델의 성능을 예측하는 'Pounce'의 3단계로 구성됩니다. 이 프레임워크는 기존 평가 방법의 한계를 극복하고 더 정교한 분석을 가능하게 합니다.

- **Technical Details**: SimBA 프레임워크의 첫 단계인 Stalk에서는 평가 행렬을 사용하여 데이터셋 간의 관계를 분석합니다. 각 데이터셋에 대해 성능 수치를 비교하고, 다변량 선형 회귀 모형을 통해 이들 간의 관계를 정량화합니다. 두 번째 단계인 Prowl에서는 모델 성능 패턴을 기반으로 중복 데이터셋을 식별하여 대표적인 데이터셋 하위 집합을 만들어 냅니다.

- **Performance Highlights**: SimBA를 사용하여 HELM, MMLU, BigBenchLite의 벤치마크에서, 데이터셋의 6.25%, 1.7%, 28.4%만으로도 95% 이상의 커버리지를 달성할 수 있음을 보여주었습니다. 또한, 이 대표 하위 집합만으로도 모델의 순위를 유지하고, 모델의 성능을 예측할 수 있으며, 평균 제곱 오차가 거의 0에 가까운 결과를 얻을 수 있었습니다.



### Universal Spectral Tokenization via Self-Supervised Panchromatic Representation Learning (https://arxiv.org/abs/2510.17959)
Comments:
          Accepted at NeurIPS 2025 Machine Learning and the Physical Sciences Workshop

- **What's New**: 이 논문에서는 다양한 객체 유형과 해상도에서 수집된 이질적인 스펙트럼을 자가 지도 학습(self-supervised learning) 방법으로 함께 학습할 수 있는 심층 학습 모델을 제안합니다. 이를 통해, 서로 다른 스펙트럼 데이터의 통합적 표현을 생성할 수 있는 범용 스펙트럼 토크나이저(universal spectral tokenizer)를 개발하였습니다. 이 모델은 처음으로 단일 모델을 통해 다양한 해상도와 도메인에서 스펙트럴 데이터를 통합할 수 있음을 보여줍니다.

- **Technical Details**: 모델은 1차원 스펙트럼 데이터에 맞게 조정된 비전 트랜스포머(Vision Transformer, ViT) 아키텍처를 기반으로 하며, 이질적인 입력 스펙트럼에서 동질적이고 파장에 대한 정보를 포함한 임베딩을 생성합니다. 이러한 임베딩은 다양한 다운스트림 작업에서 활용될 수 있습니다. 모델은 특정 서베이에 종속되지 않고 여러 데이터셋에 걸쳐 고유한 스펙트럼을 효율적으로 학습할 수 있는 능력을 가지고 있습니다.

- **Performance Highlights**: 제안된 모델은 SDSS, DESI, GALAH, APOGEE와 같은 주요 데이터셋에서 사전 훈련(pretraining) 후, 객체 분류와 항성 매개변수 회귀 작업과 같은 다운스트림 작업에서도 경쟁력 있는 성능을 보여줍니다. 특히, 우리의 접근법은 서로 다른 스펙트럼 도메인에서 데이터를 통합하여 보다 풍부하고 균질적인 표현을 생성할 수 있어, 과학적 기초 모델의 유연한 빌딩 블록으로 작용할 수 있습니다.



### Studying the Effects of Robot Intervention on School Shooters in Virtual Reality (https://arxiv.org/abs/2510.17948)
Comments:
          Preprint under review for conference publication. 10 pages, 9 figures, 3 tables (including 1-page appendix)

- **What's New**: 이 연구는 로봇의 학교 총격 사건에서의 중재 가능성을 새로운 관점으로 탐구합니다. 가상 현실 실험을 통해 로봇이 총격범을 방해하고 주의를 분산시키는 효과를 측정했습니다. 연구 결과, 공격적이고 큰 혼란을 초래하는 로봇이 피해자를 46.6% 감소시키는 것으로 나타났습니다.

- **Technical Details**: 이 논문에서는 자율 로봇이 총격범의 동작을 예측하고 전략적으로 접근하여 방해하는 역할을 하도록 설계되었습니다. 연구에는 150명의 참가자가 가상 현실 시뮬레이션에서 총격범 역할을 수행하며, 로봇의 접근 방식(공격적 vs 수동적)과 방해 방법(혼란의 정도에 따라 분류)에 따라 피해를 측정했습니다. 로봇은 CCTV 카메라를 통해 총격범의 위치를 확인하고, 예측된 위치를 바탕으로 접근하여 주의를 분산시켰습니다.

- **Performance Highlights**: 결과적으로, 공격적이고 높은 정도의 방해를 사용하는 로봇이 가장 효과적이었으며, 피해자의 수를 크게 줄이는 데 기여했습니다. 이 연구는 로봇의 개입이 학교 환경에서 안전성을 높일 수 있음을 강조하며, 이러한 기술의 윤리적 문제에 대한 논의도 필요하다고 제안합니다. 저자들은 로봇 사용의 가능성이 학내 총격 사건을 줄일 수 있음을 처음으로 입증했습니다.



### PLAGUE: Plug-and-play framework for Lifelong Adaptive Generation of Multi-turn Exploits (https://arxiv.org/abs/2510.17947)
- **What's New**: 이번 연구에서는 PLAGUE라는 새로운 프레임워크를 소개합니다. 이는 여러 차례의 공격을 설계할 수 있는 플러그 앤 플레이(plug-and-play) 구조로, 평생 학습(agentic workflows)에서 영감을 받았습니다. PLAGUE는 공격의 생애주기를 세 가지 단계로 나누어(프라이머, 계획자, 마무리 단계), 효율적이고 정보를 풍부하게 탐색할 수 있도록 설계되었습니다.

- **Technical Details**: PLAGUE는 목표에 관련성 있게 진행 상황을 보여주고, 피드백을 통해 진화하며, 다양한 샘플을 생성하는 특성을 갖춰야 합니다. 이를 통해 모델 파라미터 접근 없이도 새로운 다중 차입(jailbreaks)을 발견할 수 있게 해줍니다. 연구에서 제안한 PLAGUE는 강력한 초기화 및 피드백 통합을 통해 미세 조정 없이도 다각적인 공격 전략을 탐색할 수 있습니다.

- **Performance Highlights**: PLAGUE를 사용하여 공격한 영역에서는 97.8%의 성공률을 기록했습니다. 기존의 단일 및 다중 공격 방법 대비 30% 이상의 성공률 향상을 보여주며, 세부 모델로는 OpenAI의 o3에서 81.4%, Claude의 Opus 4.1에서 67.3%의 성공률을 달성했습니다. PLAGUE의 설계 모듈은 공격의 다양성에 크게 기여하면서도 효율성을 유지하는 데 성공했습니다.



### Intuitionistic $j$-Do-Calculus in Topos Causal Models (https://arxiv.org/abs/2510.17944)
Comments:
          42 pages

- **What's New**: 이 논문은 Pearl의 do-calculus를 일반화하여 $j$-안정함수 인과 추론(j-stable causal inference)을 토포스의 시프(topos of sheaves) 내에서 정의합니다. 기존의 Topos Causal Models (TCMs)의 틀을 더욱 발전시켜 인과 개입을 부분 객체로 정의하고, 이는 Kripke-Joyal 의미론에 기반하여 지역적으로 정의된 진리(local truth)를 도입합니다. $j$-do-calculus는 이러한 지역 진리를 사용하여 인과적 추론을 형식화하며, 이는 구조를 보존하는 사상(morphisms)으로 안정적으로 유지됩니다.

- **Technical Details**: $j$-do-calculus는 내부 직관주의 논리의 공식화로, 대칭 모노이드 범주(symmetric monoidal category) 또는 심플리시얼 집합(simplicial set)에서 인과 모델의 조작을 강조하며, 개입(interventions)을 범주적 의미에서 설명합니다. 논문에서는 Pearl의 삽입/삭제 및 행동/관찰 교환을 반영하는 세 가지 추론 규칙을 제공하고, Kripke-Joyal 의미론을 통해 그 타당성을 입증합니다. 또한, $j$-안정성(j-stability)을 조건부 독립성과 개입 주장에 대해 정의하며, 그 기반을 내부 논리에서 찾고 있습니다.

- **Performance Highlights**: 이 연구는 $j$-do-calculus의 이론적인 발전에 중점을 두며, 실험적 데이터를 기반으로 하는 $j$-커버(j-covers)와 확률 계산을 다루는 후속 논문을 준비 중입니다. $j$-커버의 구성 방법, 그래프 수술 후 조건부 독립성 계산 및 $j$-do 규칙의 타당성을 실제로 인증하는 과정 등이 포함될 예정입니다. 이는 인과 추론의 최신 연구 동향에 기여하고, 실험 기반 방법론에 대한 깊은 통찰을 제공합니다.



### Trust in foundation models and GenAI: A geographic perspectiv (https://arxiv.org/abs/2510.17942)
- **What's New**: 이번 논문에서는 기초 모델(foundation models)과 이를 통한 신뢰(trust)의 다층적인 개념을 지리적 맥락에서 탐구합니다. 특히, 데이터(training data), 모델 기능(operation), 그리고 모델 개발자 간(interpersonal trust)의 신뢰를 세 가지 유형으로 분류하여 지리적 응용 분야에서의 의미를 강조합니다.

- **Technical Details**: 기초 모델은 대규모로 사전 훈련된 머신러닝 모델로, 방대한 이질적 데이터 세트에서 학습하여 다양한 다운스트림(downstream) 작업에 적용될 수 있습니다. 이러한 모델들은 지리정보 과학에 유용하게 활용되며, 지리적으로 유효한 데이터 생성을 통해 의사 결정에 기여하고 있습니다. 그러나 이러한 복잡성으로 인해 모델에 대한 신뢰 문제는 더욱 중요해졌습니다.

- **Performance Highlights**: 모델의 신뢰성은 지리정보든, 머신러닝 모델의 기능이든 모두 다루어져야 하며, 이를 통해 اعتماد(신뢰)의 개념을 명확히 할 수 있습니다. 사용자 생성 데이터에 대한 신뢰(reputation)도 중요한 요소로 거론되며, 많은 연구에서 이를 중심으로 한 신뢰와 신뢰성의 이해가 필요하다고 제안됩니다.



### Believe It or Not: How Deeply do LLMs Believe Implanted Facts? (https://arxiv.org/abs/2510.17941)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)에 새로운 사실 지식을 심어주는 지식 편집 기술에 대한 평가 프레임워크를 개발하였습니다. 연구팀은 'belief depth'라는 개념을 도입하여, 심어진 지식이 얼마나 잘 일반화되고, 자기 분석과 직접적인 도전에 얼마나 강하며, 진정한 지식과 유사하게 표현되는지를 측정합니다. 이 작업은 지식 편집 기술의 성공을 평가하는 데 필요한 새로운 기준을 제시합니다.

- **Technical Details**: 연구진은 심어진 지식의 'belief depth'를 1) 관련 맥락으로의 일반화 정도, 2) 자기 검증과 직접적인 도전에 대한 강인성, 3) 진정한 지식 표현의 유사성으로 operationalize(운영화)하였습니다. 평가 결과 간단한 프롬프트와 기계적 편집 기술로는 충분한 깊이의 지식을 심어주기 어렵다는 사실이 밝혀졌습니다. 반면에, Synthetic Document Finetuning(SDF)은 사실에 일치하는 LLM 생성 문서로 모델을 훈련시켜 믿음을 성공적으로 심어주는 경우가 많습니다.

- **Performance Highlights**: SDF는 일반적으로 신뢰할 수 있는 방식으로 믿음을 심어주지만, 일부 경우에는 기본적인 세계 지식과 모순되는 믿음이 취약하고 진정한 지식과는 다른 표현을 보여주기도 합니다. 이 연구는 'belief depth'라는 측정 기준을 도입하여 지식 편집 기술의 실제 적용에 필요한 엄밀한 평가를 가능하게 합니다. 결론적으로, 이 연구는 지식 편집의 실질적인 성공을 평가하기 위한 중요한 기초 데이터를 제공합니다.



### The Integration of Artificial Intelligence in Undergraduate Medical Education in Spain: Descriptive Analysis and International Perspectives (https://arxiv.org/abs/2510.17938)
Comments:
          1 figure, 4 main tables, 2 supplementary tables

- **What's New**: 이번 연구는 스페인에서 의과대학 교육 과정에 AI 통합의 현황을 체계적으로 평가한 첫 번째 사례입니다. 2025-2026 학년도 의과대학에서 AI 관련 과정 및 역량을 조사하여, AI 교육의 필요성을 강조하고 있습니다. 연구 결과에 따르면, 스페인 내 52개 대학교 중 10개(19.2%)에서 AI 관련 특정 과정을 제공하고 있습니다.

- **Technical Details**: 연구에서는 스페인 내 공식 의학 학위를 제공하는 대학교를 대상으로 하는 단면 연구가 수행되었습니다. 조사된 과정들은 주로 선택 과목으로 제공되며, 평균적으로 360학점의 전체 학위에서 AI 관련 과정은 1.17%에 불과하고, 전공 필수 과목은 없습니다. Jaén 대학교는 유일하게 AI 관련 필수 과목을 개설하고 있습니다.

- **Performance Highlights**: AI 교육의 통합은 불균형적이며 지역마다 큰 차이를 보이고 있습니다. 안달루시아 지역은 대학의 55.5%가 AI 교육을 포함하고 있는 반면, 다른 지역은 이와 같은 노력이 전혀 없는 경우가 많습니다. 이러한 결과는 AI 교육을 위한 최소 기준 설정과 국가 차원의 지표 모니터링 필요성을 지지합니다.



### UniRL-Zero: Reinforcement Learning on Unified Models with Joint Language Model and Diffusion Model Experts (https://arxiv.org/abs/2510.17937)
- **What's New**: UniRL-Zero는 언어 모델(LM) 이해와 추론 증진 및 확산 모델(DM) 멀티미디어 생성 강화에 중점을 둔 통합 강화 학습( Reinforcement Learning, RL) 프레임워크입니다. 본 연구는 통합 모델 강화 학습을 위한 여섯 가지 시나리오를 정의하며, 향후 연구를 위한 체계적인 기준선을 제공합니다. 코드는 제공되는 링크를 통해 접근 가능하다고 발표하였습니다.

- **Technical Details**: UniRL-Zero는 언어 모델과 확산 모델의 조합을 이용하여, 각 모듈이 협력하는 설계를 캡슐화합니다. RL은 토큰 수준의 최적화에 국한되지 않고, 두 모델 간의 상호작용을 아우릅니다. 여섯 가지 시나리오는 텍스트 이해, 다중 모달 추론, 텍스트-이미지 생성 및 편집 등 다양한 통합 작업을 정의하고 있습니다.

- **Performance Highlights**: 본 연구는 특히 생성 작업에서 LM과 DM 간의 밀접한 상호작용의 중요성을 강조합니다. UniRL-Zero 프레임워크는 효율성을 유지하면서도 품질 높은 텍스트-이미지 시퀀스를 생성하도록 설계되었습니다. 이는 정량적인 성과뿐만 아니라 주어진 시나리오에 대해 RL 전략의 효과를 탐색하는 데 중점을 두고 있습니다.



### XDXD: End-to-end crystal structure determination with low resolution X-ray diffraction (https://arxiv.org/abs/2510.17936)
- **What's New**: 이번 연구에서는 X-ray Diffraction (XRD) 데이터를 사용하여 저해상도에서 원자 구조를 직접 결정할 수 있는 최초의 end-to-end deep learning 프레임워크인 XDXD를 소개합니다. 기존의 방법들이 수동적으로 지도를 해석해야 하는 반면, XDXD는 생성적 모델을 통해 Diffraction 패턴에 기반하여 화학적으로 타당한 결정 구조를 자동으로 생성합니다. 이 모델은 2.0 Å 해상도로 제한된 데이터에 대해 70.4%의 일치율을 보이며, RMSE는 0.05 이하로 나타났습니다.

- **Technical Details**: XDXD 모델은 전이기반 변환자 구조를 포함한 XRD 인코더와 화학 정보를 담고 있는 분자 그래프 인코딩 레이어로 구성됩니다. 이 두 가지 인코딩을 통해 아토믹 좌표를 반복적으로 개선하여 최종 구조를 생성하는 Diffraction-Conditioned Structure Predictor (DCSP) 모듈로 전달됩니다. 훈련 과정에서는 실험적 노이즈 조건을 모사하기 위해 랜덤한 신호 드롭아웃을 도입하여 모델의 안정성을 높입니다.

- **Performance Highlights**: XDXD는 24,000개의 실험 구조를 평가하였으며, 다양한 결정구조를 생성할 수 있는 능력을 보여주었습니다. 예측 구조의 일치율은 원자 수가 증가함에 따라 감소하지만, 160~200개의 원자를 포함하는 시스템에서도 약 40%의 높은 일치율을 달성했습니다. 이 연구는 저해상도에서 결정 구조의 자동 추론을 가능하게 하여 복잡한 시스템으로의 확장 가능성을 열어줍니다.



### AtlasKV: Augmenting LLMs with Billion-Scale Knowledge Graphs in 20GB VRAM (https://arxiv.org/abs/2510.17934)
- **What's New**: 이 논문은 LLMs (대규모 언어 모델)에 대해 외부 지식을 통합하는 새로운 파라메트릭 방법인 AtlasKV를 제안합니다. AtlasKV는 기가바이트(GiB) 단위의 GPU 메모리 비용으로 억 규모의 지식 그래프(KG, Knowledge Graph)를 통합할 수 있으며, 기존 RAG 방법이 갖고 있던 시간 지연(inference latency) 문제를 해결합니다. 이 방법은 KG2KV와 HiKVP라는 혁신적인 설계를 도입하여 LLM에 효과적으로 KG 트리플을 통합합니다.

- **Technical Details**: AtlasKV는 대규모 KG에서 쿼리-키-값(Q-K-V) 데이터로 자연스럽게 변환하는 KG2KV 파이프라인을 소개합니다. 또한, HiKVP 알고리즘을 통해 컴퓨팅 및 메모리 오버헤드를 대폭 줄이면서 높은 지식 기반 정확도를 유지할 수 있습니다. 이러한 설계는 LLM의 주의 메커니즘을 활용하여 효과적으로 지식을 통합하도록 돕습니다.

- **Performance Highlights**: 실험 결과, AtlasKV는 ICL, KBLaM 및 RAG 방법에 비해 우수한 효율성과 확장성을 보여주며, 각 구성 요소의 기여를 검증하기 위한 포괄적인 압축 연구(ablation studies)를 진행했습니다. 일반화 성능도 뛰어나며, 외부 지식에 대한 적응력이 뛰어난 것으로 나타났습니다.



### From Observations to Parameters: Detecting Changepoint in Nonlinear Dynamics with Simulation-based Inferenc (https://arxiv.org/abs/2510.17933)
Comments:
          15 pages

- **What's New**: 본 논문에서는 복잡한 시스템의 역학에서 급격한 변화를 감지하는 문제를 해결하기 위해 Parameter-Space Changepoint Detection (Param-CPD)라는 새로운 두 단계 프레임워크를 제안합니다. 이 방법은 첫 번째 단계에서 시뮬레이션 기반 추론을 통해 훈련된 신경 후향 추정기(neural posterior estimator)를 사용하여 계측 매개변수의 베이지안 추론(Bayesian inference)을 연계하고, 두 번째 단계에서 표준 CPD 알고리즘을 이를 통해 얻어진 매개변수 궤적에 적용합니다.

- **Technical Details**: Param-CPD는 더 차원 적은 매개변수 공간에서 작동하여 비선형 동역학 시스템의 변화를 더 명확히 감지할 수 있도록 합니다. Lorenz-63 시스템을 모델로 하여, 신경 후향 추정기를 통해 관측 데이터의 세그먼트를 시스템 매개변수의 후방 분포에 매핑하는 과정이 있습니다. 이 후방 분포는 표준 CPD 알고리즘을 적용하기 위한 깨끗하고 정보가 풍부한 매개변수 궤적(time-series trajectory)을 생성하는 데 사용됩니다.

- **Performance Highlights**: 실험 결과, Param-CPD는 전통적인 관측 공간 기준에 비해 변화 감지 정확도(F1), 로컬라이제이션 오류 감소, 및 잘못된 경고(false positives) 감소에서 현저한 향상을 보였습니다. 더불어, 인식된 후방 분포의 식별 가능성과 보정(calibration)을 검증하여 매개변수 공간이 명확한 탐지 신호를 제공하는 이유를 설명합니다. 다양한 강건성 분석을 통해, 우리의 방법은 매개변수 서브세트에 대해 안정성과 효율성을 입증하였습니다.



### From Charts to Code: A Hierarchical Benchmark for Multimodal Models (https://arxiv.org/abs/2510.17932)
- **What's New**: 새로운 벤치마크인 Chart2Code가 소개되었습니다. 이 벤치마크는 대규모 멀티모달 모델(LMMs)의 차트 이해 및 코드 생성 능력을 평가하도록 설계되었습니다. 사용자 중심의 관점을 바탕으로 다양한 현실 세계 시나리오를 포착하고 작업의 난이도를 점진적으로 증가시킵니다.

- **Technical Details**: Chart2Code는 세 가지 레벨로 구성됩니다: 레벨 1 (Chart Reproduction)은 참고 그림과 사용자 쿼리를 기반으로 차트를 재현하는 것을 목표로 하며, 레벨 2 (Chart Editing)는 차트 유형 변경이나 요소 추가와 같은 복잡한 수정 작업을 포함합니다. 레벨 3 (Long-Table to Chart Generation)은 긴 데이터 테이블을 사용자 지침에 따라 신뢰성 있는 차트로 변환하는 작업을 요구합니다. 이 평가체계는 코드의 정확성과 렌더링된 차트의 시각적 충실도를 평가하는 멀티 레벨 평가 메트릭을 함께 제공합니다.

- **Performance Highlights**: 25개의 최신 LMM 모델을 벤치마킹한 결과, 상위 모델인 GPT-5도 여전히 코드 기반 평가에서 평균 0.57, 차트 품질 평가에서는 0.22로 낮은 점수를 기록했습니다. 이러한 결과는 Chart2Code의 난이도를 강조하며, LMM의 실용적인 능력에서 상당한 격차가 있음을 보여줍니다. 이 벤치마크는 멀티모달 추론의 발전을 이끌고 보다 강력하고 범용적인 LMM의 개발을 촉진할 것으로 기대됩니다.



### Attracting Commercial Artificial Intelligence Firms to Support National Security through Collaborative Contracts (https://arxiv.org/abs/2510.17931)
Comments:
          312 pages, 42 figures

- **What's New**: 본 논문은 상업적 AI 기업이 왜 국방부(DoD)와 협력하거나 국방 시장에서 자제하는지를 이해하는 데 초점을 맞추고 있습니다. 연구 결과는 계약법과 조달(procurment) 체계가 상업적 AI 산업의 주요 장벽으로 지적됩니다. 이는 군 사용을 위한 기술이 아닌 상업적 목적으로 주로 개발된 AI가 국방부에 매력을 느끼는 이면을 설명합니다.

- **Technical Details**: 이 연구는 사회적 교환 이론(social exchange theory)을 바탕으로 하여 최적 구매자 이론(optimal buyer theory)이라는 이론적 틀을 제시합니다. 이는 상업적 결정이 국방부와의 관계 형성에 영향을 미치는 요소들을 분석합니다. 인터뷰를 통해 AI 산업의 관계자들이 계약에 대해 갖고 있는 인식과 선호도를 설명하며, 이론적 배경을 다룹니다.

- **Performance Highlights**: 결론적으로, 상업적 AI 기업은 비즈니스 및 기술 고려사항에 부합하는 계약에 매력을 느끼고 있습니다. 이 논문은 기존의 계약법을 활용하여 상업적 선호와 머신러닝 개발 및 배치 수명 주기에 맞는 조달 관행을 정렬하는 최선의 관행을 개발합니다. 이는 AI 산업과 국방부 간의 상호작용을 개선하는 데 기여할 것으로 기대됩니다.



### Diagnosing Representation Dynamics in NER Model Extension (https://arxiv.org/abs/2510.17930)
- **What's New**: 이 논문은 Named Entity Recognition (NER) 모델을 새로운 PII(개인 식별 정보) 엔티티로 확장하는 방법에 대한 새로운 접근법을 제시합니다. BERT 모델을 기존의 표준 의미론적 엔티티(PER, LOC, ORG)와 새로운 패턴 기반 PII(EMAIL, PHONE)에 대해 공동으로 미세 조정함으로써, 원래의 클래스에 대한 최소한의 퇴화를 실현하였습니다. 이는 NER 모델의 적응에 대한 새로운 메커니즘적 진단을 제공합니다.

- **Technical Details**: 연구에서는 의미적 드리프트(semantic drift)를 측정하기 위해 점진적 학습(incremental learning) 설정을 진단 도구로 사용합니다. 그 결과, LOC(위치) 엔티티가 새로운 PII와의 표현 중복(representation overlap)으로 인해 고유하게 취약함을 발견하였고, 'O' 태그의 분류기(classifier)의 동결 해제가 필요함을 입증하였습니다. 이 과정에서 모델은 PII 패턴을 'O'로 매핑하도록 훈련되지만 새로운 학습을 차단하는 '역 O-태그 표현 드리프트(reverse O-tag representation drift)'가 발생합니다.

- **Performance Highlights**: 이 연구는 NER 모델이 독립적인 의미적/형태적(feature semantic vs. morphological) 특성 메커니즘을 사용한다고 가정합니다. 그에 따라, 'O' 태그의 유연성(plasticity)을 통해 배경 클래스를 적응시켜 새로운 패턴을 수용할 수 있는 방안을 제시합니다. 또한, 이러한 특징 독립성, 표현 중복 및 'O' 태그의 유연성은 미래의 모델 개선에 중요한 통찰력을 제공할 것입니다.



### EvoSyn: Generalizable Evolutionary Data Synthesis for Verifiable Learning (https://arxiv.org/abs/2510.17928)
- **What's New**: 이번 연구에서는 Evolutionary Data Synthesis (EvoSyn)이라는 새로운 데이터 합성 프레임워크를 소개합니다. EvoSyn은 verifiable data(검증 가능한 데이터)의 생성을 자동화하여, 효율적으로 다양한 문제 및 솔루션을 생성합니다. 이 연구의 중요한 기여는 적은 감독 하에 다양한 과제에서 일반화할 수 있는 프레임워크를 설계했다는 점입니다.

- **Technical Details**: EvoSyn은 executably-checkable tasks(실행 가능 검증 과제)를 목표로 하며, 이는 정확성을 결정하기 위한 테스트 아티팩트(예: 유닛 테스트)를 통해 해석됩니다. 이 접근법은 기존의 수작업 기반 방법론과 달리 진화 알고리즘을 활용하여 필터링 전략을 최적화합니다. 이러한 자동화된 과정은 시간 소모를 줄이며, 고품질 데이터를 생성하는 데 기여합니다.

- **Performance Highlights**: EvoSyn을 통한 데이터 생성을 통해 LiveCodeBench와 AgentBench-OS 두 가지 벤치마크에서 모델의 성능이 크게 향상되었습니다. 특히 RLVR과 모델 증류 기법에서 EvoSyn의 생성 데이터가 기존 데이터보다 우수한 학습 효과를 보였습니다. 이러한 결과는 EvoSyn의 강력한 일반화 능력을 잘 보여줍니다.



### SpecAgent: A Speculative Retrieval and Forecasting Agent for Code Completion (https://arxiv.org/abs/2510.17925)
- **What's New**: 이 논문은 SpecAgent라는 에이전트를 소개하여, 코드 생성 품질과 지연(latency)을 동시에 향상시키는 방법을 설명합니다. 기존의 코드 자동완성 방법들이 실시간으로 정보를 검색해야 하는 반면, SpecAgent는 인덱싱 단계에서 정보를 미리 탐색하여 자연스럽게 지연을 감추고 코드 생성 품질을 개선합니다. 또한, 기존 벤치마크에서 발생하는 '미래 컨텍스트 누출' 문제를 해결하기 위해 새로운 벤치마크를 구축하여 보다 현실적인 성능 평가를 가능하게 합니다.

- **Technical Details**: SpecAgent는 레포지토리 인덱싱 시점에서 각 파일에 대한 구조화된 문맥을 구성하며, 이를 통해 빠른 응답을 제공할 수 있도록 설계되었습니다. 이 에이전트는 미래의 기능이나 문제를 예측하여 추가적인 문맥을 검색하며, 이러한 비동기식(indexing-time asynchrony) 접근법은 주어진 요청을 처리할 때 유용합니다. SpecAgent는 코드 관련 정보를 미리 계산하여 인퍼런스(inference) 단계에서의 지연을 줄입니다.

- **Performance Highlights**: 실험 결과, SpecAgent는 최상의 기준선과 비교하여 약 9-11%의 성능 향상을 달성하였으며, 이는 48-58%의 상대적 증가에 해당합니다. 그뿐만 아니라 인퍼런스 시 지연을 현저히 줄여 사용자 경험을 개선합니다. SpecAgent는 Qwen3-8B 모델에서 약 10-11%, Qwen3-30B-A3B 모델에서 9-10%의 성능 개선을 보이며, 이는 기존의 코드 자동완성 모델들이 해결하지 못한 문제들을 덜어줍니다.



### Efficient Toxicity Detection in Gaming Chats: A Comparative Study of Embeddings, Fine-Tuned Transformers and LLMs (https://arxiv.org/abs/2510.17924)
Comments:
          Published in the Journal of Data Mining & Digital Humanities (JDMDH), special issue NLP4DH

- **What's New**: 이번 연구는 온라인 게임 채팅에서 자동화된 독성 탐지를 위한 자연어 처리(NLP) 방법에 대한 포괄적인 비교 분석을 제시합니다. 전통적인 기계 학습 모델, 대형 언어 모델(LLMs), 세분화된 트랜스포머 모델 및 검색 증강 생성(RAG) 접근 방식이 평가됩니다. 평가 프레임워크는 분류 정확도, 처리 속도 및 계산 비용의 세 가지 핵심 차원을 측정하며, 인적 조정자의 작업 부담을 최적화하는 하이브리드 조정 시스템 아키텍처를 제안합니다.

- **Technical Details**: 연구는 정적인 임베딩 및 맥락적 임베딩을 기반으로 한 전통적인 기계 학습 분류기에서부터 현대적인 LLM 및 RAG에 이르는 여러 NLP 기법의 조합을 분석합니다. 이 연구의 목표는 정확성, 속도 및 자원 효율성과 같은 성능 절충안을 정량화하고 공정성 또는 정밀성을 훼손하지 않으면서 인적 작업 부담을 최소화할 수 있는 하이브리드 디자인을 식별하는 것입니다. 경험적 결과는 DistilBERT가 최적의 정확도-비용 거래를 달성하는 등 방법 간의 성능 차이를 보여줍니다.

- **Performance Highlights**: 본 연구는 자동화된 독성 탐지 시스템이 비용 효율적이고 효율적인 콘텐츠 조정 시스템을 배치할 수 있는 근거를 제공합니다. 세 가지 주요 차원에서의 비교를 통해, 세분화된 트랜스포머 모델이 가장 높은 정확성을 기록하며, 이는 게임 환경의 동적 특성에 적합한 조정 솔루션을 제공함을示합니다. 향후 연구는 이러한 자동화된 시스템을 진화하는 온라인 게임 커뮤니티에 통합하고, 인적 조정자들이 보다 복잡한 결정에 집중할 수 있도록 하는 데 기여할 것입니다.



### Rewarding the Journey, Not Just the Destination: A Composite Path and Answer Self-Scoring Reward Mechanism for Test-Time Reinforcement Learning (https://arxiv.org/abs/2510.17923)
- **What's New**: 이 논문에서는 Langarge Language Models (LLMs) 의 성능을 높이기 위한 자율적 강화 학습(RL) 접근 방식을 탐구합니다. 기존의 RL 방법들이 인간이 선별한 데이터나 라벨이 필요한 한계를 고려할 때, 본 연구는 라벨이 없는 데이터를 활용하여 모델이 지속적인 경험 흐름에서 스스로 학습할 수 있는 방법을 제안합니다. COMPASS라는 새로운 테스트 타임 보상 메커니즘을 소개하며, 이는 외부 감독 없이 두 가지 보상 요소인 DCAR과 DPR을 통합하여 신뢰할 수 있는 합의 결과와 결단력 있는 추론 과정의 질을 강화합니다.

- **Technical Details**: COMPASS는 두 가지 중요한 구성 요소로 구성되어 있습니다: Dual-Calibration Answer Reward (DCAR)은 신뢰할 수 있는 의사 라벨을 통해 학습 안정성을 높이고 효율성을 개선하며, Decisive Path Reward (DPR)은 최종 정답을 넘어 각 생성 과정의 단계를 조사해 더 결정적인 선택을 하도록 유도합니다. 이러한 방법은 모델이 고유 신뢰성 지표를 활용하도록 하여 보상을 추정하는 데 도움을 줍니다. 결과적으로, COMPASS는 더욱 신뢰할 수 있는 자율적 평가 체계를 통해 LLM들이 지속적으로 진화할 수 있게 하는 혁신적인 방법입니다.

- **Performance Highlights**: COMPASS는 다양한 추론 작업과 모델 구조에 대해 실험을 수행하여 의미 있는 성능 향상을 달성했습니다. 이러한 결과는 모델이 지속적인 경험을 통해 자립적으로 학습할 수 있는 더 스케일 가능한 방향을 제시합니다. COMPASS의 혁신적인 보상 메커니즘은 LLM의 분석 능력을 체계적으로 향상시키며, 기존의 기술보다 더 나은 경향성을 보였습니다.



### Select-Then-Decompose: From Empirical Analysis to Adaptive Selection Strategy for Task Decomposition in Large Language Models (https://arxiv.org/abs/2510.17922)
Comments:
          Accepted to the Main Conference of EMNLP 2025 (Oral)

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 작업 분해(task decomposition)에 대한 포괄적인 조사를 실시하고, 여섯 가지 범주화 체계를 제시하였습니다. 기존의 연구들은 주로 메모리, 도구 사용, 피드백 메커니즘에 중점을 두었지만, 성능과 비용 간의 균형을 간과하는 경향이 있었습니다. 이를 통해 Select-Then-Decompose 전략을 제안하며, 이는 선택, 실행, 검증의 세 단계로 구성된 폐쇄 루프 문제 해결 프로세스를 구축합니다.

- **Technical Details**: 연구팀은 LLM의 작업 분해에서 성능과 비용에 영향을 미치는 세 가지 주요 요소를 식별하였습니다: 작업 분해 접근 방식의 범주, 작업의 특성, 분해 모델과 실행 모델의 구성입니다. 이 연구에서는 작업 분해 과정을 통해 발생할 수 있는 성능-비용 딜레마에 대한 통찰력을 제공하고, 계층적 구조와 같은 몇 가지 토폴로지 구조에 대해 설명합니다. Select-Then-Decompose 전략은 주어진 작업의 특성에 따라 최적의 분해 접근 방법을 동적으로 선택하고, 이러한 결과의 신뢰성을 향상시키기 위한 검증 모듈을 갖추고 있습니다.

- **Performance Highlights**: 다양한 벤치마크에서의 포괄적인 평가 결과, Select-Then-Decompose 전략은 성능과 비용 간의 최적 균형을 달성하며 Pareto 경계에서 일관된 성능을 보여주었습니다. 연구는 또한 성능을 향상시키기 위해 실행 모델을 조정하는 것이 분해 모델을 조정하는 것보다 더 큰 성과 향상을 이끈다는 점을 강조합니다. 전반적으로 본 연구는 LLM의 작업 분해 성능을 높이는 데 기여하는 일련의 실용적인 원칙을 제공합니다.



### CLAWS:Creativity detection for LLM-generated solutions using Attention Window of Sections (https://arxiv.org/abs/2510.17921)
Comments:
          NeurIPS 2025

- **What's New**: 최근 대규모 언어 모델(LLMs)의 추론 능력을 향상시키기 위한 연구에서 괄목할만한 성과를 이뤘습니다. 이 논문은 LLM이 수학과 코딩과 같은 도전적인 작업에서 강력한 성능을 발휘할 수 있도록 강화 학습(RL)으로 훈련된 새로운 방법 CLAWS를 제안합니다. CLAWS는 인간 평가 없이 수학적 솔루션을 전형적, 창의적, 홀로리테이션된 범주로 분류할 수 있는 방법론을 제공합니다.

- **Technical Details**: 이 연구는 LLM의 내부 표현으로부터 특성을 추출하여 홀로리테이션(Hallucinated), 창의적(Creative), 전형적(Typical) 솔루션을 분류하는 실험 프레임워크를 제시합니다. 입력 프롬프트는 가이드라인(GG), 문제(PP), 참조 솔루션(SS), 지침(II)의 네 부분으로 나뉘며, 각 부분의 주목(attention) 분석을 통해 분류가 수행됩니다. 상대적으로 작은 모델 크기로도 효과적으로 성능을 나타낼 수 있도록 설계되었습니다.

- **Performance Highlights**: CLAWS는 7-8B 매개변수 범위의 다섯 개 수학 RL 모델을 활용하여 기존의 다섯 가지 화이트 박스 탐지 방법들을 초월하는 성능을 보여줍니다. 특히, 연구에 사용된 4,545개의 수학 문제를 통해 생성된 솔루션들은 모두 높은 효율성으로 분류될 수 있었으며, 여기서 CLAWS는 일관되게 기준 방법들보다 우수한 성능을 기록했습니다. 이는 수학적 추론 작업에서 창의성 평가의 새로운 기준을 마련할 수 있는 가능성을 보여줍니다.



### CBINNS: Cancer Biology-Informed Neural Network for Unknown Parameter Estimation and Missing Physics Identification (https://arxiv.org/abs/2510.17920)
Comments:
          29 pages, 24 figures

- **What's New**: 이 연구는 암 생물학에 정보를 기반으로 한 신경망 모델(CBINN)을 개발하여 분산되고 노이즈가 있는 실험 데이터로부터 미지의 파라미터 및 누락된 물리적 법칙을 추론합니다. 기존의 비선형 ODE(Ordinary Differential Equations) 모델을 활용하여 암-면역 시스템의 복잡한 상호작용을 모델링합니다. 이는 실험적 측정의 다양한 제한 사항에도 불구하고, 생물학적 시스템을 이해하는 새로운 접근 방식을 제시합니다.

- **Technical Details**: CBINN은 복잡한 생물학적 시스템의 비선형 동적인 상호작용을 효과적으로 추정하는 방법을 제공합니다. 이 모델은 세 가지 비선형 분할 모델을 테스트하여 다양한 synthetic noise 수준에서도 강인성을 평가합니다. 시스템의 파라미터 추정 외에도, 이 신경망은 미지의 동역학을 발견하는 데 필요한 수학적 구조를 식별합니다.

- **Performance Highlights**: 실험 결과, CBINN 모델은 주어진 үш 가지 모델에서 파라미터를 효과적으로 추정하고, 비선형 동역학의 불확실성을 극복하는 데 성공적이었습니다. 이를 통해 암-면역 상호작용의 다이나믹 패턴을 설명하며, 기존 방법론의 일반화 가능성과 효과성을 입증합니다. CBINN의 Robustness는 제한적이고 노이즈가 많은 데이터 속에서도 뛰어난 성능을 보여줍니다.



### ParaVul: A Parallel Large Language Model and Retrieval-Augmented Framework for Smart Contract Vulnerability Detection (https://arxiv.org/abs/2510.17919)
- **What's New**: 이번 논문에서는 ParaVul이라는 새로운 스마트 계약 취약점 탐지 프레임워크를 제안합니다. ParaVul은 병렬 처리(Parallel Processing)를 통해 LLM 기반 탐지와 RAG 기반 탐지를 동기화하여 효과적으로 탐지 정확도를 향상시킵니다. 또, 우리는 찾은 취약점과 해당 문제 해결 제안을 명확하게 이해할 수 있도록 돕는 취약점 탐지 보고서 템플릿도 설계했습니다.

- **Technical Details**: ParaVul 프레임워크는 SLoRA를 사용하여 LLM의 파인튜닝 과정에서 발생하는 계산 비용을 줄입니다. SLoRA는 양자화된 LoRA(Quantized LoRA)를 기반으로 비핵심 연결을 동적으로 제거하고 LLM의 백본 매개변수를 고정하여 어댑터 레이어만 훈련합니다. 또한 하이브리드 RAG 시스템을 통해 LLM의 탐지 결과를 검증하고, 메타러닝 모델을 사용하여 LLM과 RAG의 출력을 융합하여 최종 탐지 결과를 생성합니다.

- **Performance Highlights**: 실험 결과, ParaVul은 F1 점수에서 높은 성능을 보여주었습니다. 단일 레이블 탐지의 경우 0.9398, 다중 레이블 탐지의 경우 0.9330이라는 성과를 달성하였습니다. 이러한 성능 향상은 기존 LLM과 RAG 기반 접근 방식보다 각각 4% 및 12% 더 개선된 결과입니다.



### JT-Safe: Intrinsically Enhancing the Safety and Trustworthiness of LLMs (https://arxiv.org/abs/2510.17918)
- **What's New**: 본 논문은 대규모 언어 모델(LLM)의 비정확성과 신뢰성 문제를 해결하기 위한 새로운 접근 방식을 제안합니다. 특히, 사전 학습(pre-training) 데이터의 품질을 향상시켜 LLM의 안전성과 신뢰성을 높이는 데 초점을 맞추고 있습니다. 연구진은 현실 세계의 맥락을 보강한 새로운 데이터 세트(DWC: Data with World Context)를 도입하여, 자원 데이터의 불일치와 오류를 줄이려는 노력을 하고 있습니다.

- **Technical Details**: 현재 LLM의 사전 학습 데이터는 15조에서 40조 개의 토큰(token)으로 구획되어 있으며, 인터넷 원시 데이터, 생성된 데이터 및 전문 분야의 데이터 등 다양합니다. 기존 데이터 가공 파이프라인을 기반으로 모델과 규칙, 빅데이터 처리 유틸리티를 활용하여 데이터의 품질을 더욱 높이는 방법을 제안했습니다. 이를 통해 사전 학습 데이터는 실세계 맥락 정보를 포함할 수 있도록 보강되어 불확실성을 줄이는 데 도움을 줍니다.

- **Performance Highlights**: JT-Safe-35B 모델은 6.2 조 개의 토큰을 사용하여 안전성 및 신뢰성 벤치마크에서 평균 1.79%의 성능 향상을 달성했습니다. 특히 JT-Safe-35B는 산업 관련 데이터 양을 더욱 확대하며 성능을 극대화하였습니다. 이러한 개선은 실제 사용자의 경험과 벤치마크 결과에서도 긍정적인 영향을 미쳤습니다.



### Data Unlearning Beyond Uniform Forgetting via Diffusion Time and Frequency Selection (https://arxiv.org/abs/2510.17917)
Comments:
          Preprint

- **What's New**: 본 논문에서는 데이터 언러닝(data unlearning)의 새로운 접근법을 제안합니다. 특히, 확산 모델(diffusion models)에서 발생하는 언러닝 문제를 다룹니다. 기존의 방법들이 시간 단계(time steps)에서 균등하게 샘플을 언러닝하려는 경향이 있어 품질 저하가 발생하는 문제를 해결하고자 합니다.

- **Technical Details**: 제안된 방법은 특정 시간-주파수(time-frequency) 범위에 집중하여 샘플을 훈련하는 방식입니다. 이로 인해 높은 미적 품질(aesthetic quality)과 낮은 노이즈(noise)를 가진 샘플을 생성할 수 있습니다. 또한, 다양한 설정에서 이 방식을 검증하며, 그래디언트 기반(gradient-based) 및 선호 최적화(preference optimization) 목표를 포함한 이미지 수준(image-level) 및 텍스트-이미지(text-to-image) 작업에서도 적용합니다.

- **Performance Highlights**: 제안된 방법론을 통하여 데이터 언러닝 성능을 개선할 수 있음을 보여줍니다. 언러닝된 데이터 샘플의 삭제 시 품질을 평가하기 위해 간단한 정규화 버전의 SSCD를 제안합니다. 분석 및 방법론을 통해 확산 모델에서 데이터 언러닝의 독특한 도전 과제를 더 명확히 이해하고, 평가 및 언러닝 성능을 향상시키는 실용적인 전략을 제공합니다.



### Self-Evidencing Through Hierarchical Gradient Decomposition: A Dissipative System That Maintains Non-Equilibrium Steady-State by Minimizing Variational Free Energy (https://arxiv.org/abs/2510.17916)
Comments:
          30 pages, 13 Figures

- **What's New**: 이번 연구에서는 Free Energy Principle (FEP)을 활용하여 자가 조직화 시스템이 변Variational Free Energy를 최소화해야 유지된다는 이론을 바탕으로, 이를 구현할 수 있는 접근 방식을 제시합니다. 연구팀은 네트워크의 각 연결 블록마다 예상되는 gradient의 크기를 추정하는 Trophic Field Map (TFM)을 사용하여 명확한 지역적 신뢰 부여(local credit assignment)를 수행할 수 있다는 것을 증명하였습니다.

- **Technical Details**: 제안된 시스템은 공간적, 시간적 신뢰 부여를 계층적으로 분해하여 Gradient 계산을 수행합니다. 피드백 정렬을 통한 공간적 신뢰 부여, 적합도 추적(eligibility traces)을 통한 시간적 신뢰 부여, 그리고 Trophic Field Map를 활용한 구조적 신뢰 부여를 통해 세 가지 문제를 해결하고, 이는 정확한 gradient 예측까지 발전합니다. 이 과정은 자기 조직적인 비평형 지속 상태(NESS)를 유지하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 실험적으로, TFM은 오라클 gradients와 0.9693의 Pearson 상관관계를 기록하며, 98.6%의 작업 유지율 및 75%의 구조적 손상 후 4.7배의 기준 오류 내에서 자율 회복을 보여 주었습니다. 또한, 자기 조직적 한계(criticality)를 유지하며 시스템의 계산적 용량을 극대화하며, 이는 생물학적 학습과 AI 시스템의 적용 가능성을 더욱 넓힙니다.



### Uncertainty-Aware Post-Hoc Calibration: Mitigating Confidently Incorrect Predictions Beyond Calibration Metrics (https://arxiv.org/abs/2510.17915)
Comments:
          53 pages, 12 figures, 12 tables

- **What's New**: 이 논문은 개별 예측의 이질적인 신뢰도를 무시하고 모든 예측에 대해 동일한 전역 변환을 적용하는 기존 방법의 한계를 극복하기 위해, 예측 신뢰성 평가를 활용하여 보정 품질을 향상시키고 불확실성 인식 의사 결정을 공동으로 향상시키는 사후 보정 프레임워크를 제시합니다. 이 프레임워크는 가까운 이웃 기반의 conformal prediction을 적용하여 보정 샘플을 의미론적 유사성에 따라 올바른 그룹과 잘못된 그룹으로 나눕니다. 이는 새로운 분류 신뢰도 조정 기술을 통해 데이터에 대한 모델 신뢰성을 향상시키는 데 기여합니다.

- **Technical Details**: 제안된 방법은 두 가지 보정 전략을 활용하는데, 첫째는 표준 isotonic regression을 통해 올바른 예측에 대한 신뢰도를 보정하고, 둘째는 underconfidence-regularized isotonic regression을 통해 잘못된 예측에 대한 신뢰도를 낮춰 잘못된 예측의 식별을 용이하게 합니다. 이러한 이중 보정 전략은 사후 보정이라는 원칙을 준수하며, 예측 클래스 레이블을 변경하지 않고 신뢰도 추정값을 조정합니다. 이를 통해 모델의 예측 신뢰성을 개선하면서도 예측 정확성을 유지합니다.

- **Performance Highlights**: CIFAR-10 및 CIFAR-100 데이터셋에서 실험을 수행한 결과, 제안된 방법이 isotonic 및 focal-loss 기법에 비해 낮은 신뢰도와 잘못된 예측을 기록했으며, 기대 보정 오류(Expected Calibration Error, ECE)에서도 경쟁력 있는 성과를 보였습니다. 이 연구는 보정(calibration)과 불확실성 정량화(uncertainty quantification)를 연결함으로써, 모델 재학습 없이도 사후적인 해결책을 제공하고 있습니다. 이는 확률 정렬과 불확실성 인식 의사 결정을 모두 향상시키는 실용적인 접근 방식입니다.



### NeuCo-Bench: A Novel Benchmark Framework for Neural Embeddings in Earth Observation (https://arxiv.org/abs/2510.17914)
- **What's New**: NeuCo-Bench는 Earth Observation (EO) 컨텍스트에서 손실 신경 압축(neural compression) 및 표현 학습(representation learning)의 평가를 위한 혁신적인 벤치마크 프레임워크입니다. 이 프레임워크는 다양한 다운스트림 작업에 적용할 수 있는 고정 크기의 임베딩을 기반으로 하여, 재사용 가능한 임베딩을 중심으로 평가 파이프라인을 구축합니다. NeuCo-Bench는 새로운 Hidden-task 리더보드와 정확도 및 안정성을 균형 있게 측정하는 점수 시스템을 포함합니다.

- **Technical Details**: NeuCo-Bench는 주로 세 가지 핵심 구성 요소로 이루어져 있습니다. 첫째, 재사용 가능한 임베딩으로 구성된 평가 파이프라인이며, 둘째, 사전 학습 편향을 완화하기 위한 Hidden-task 리더보드를 갖춘 새로운 도전 모드입니다. 셋째, 정확도와 안정성의 균형을 맞추는 점수 시스템을 통해 실험 재현성을 지원하고, SSL4EO-S12 다운스트림이라는 다중 스펙트럼 및 다중 시계열 EO 데이터셋을 공개합니다.

- **Performance Highlights**: 2025 CVPR EARTHVISION 워크숍에서 진행된 공개 도전에서 초기 결과를 제시하고, 최신의 기초 모델(foundation models)과 함께 실험을 수행하였습니다. NeuCo-Bench는 EO 및 그 이상의 분야에서 신경 임베딩에 대한 커뮤니티 기반의 표준화된 평가의 첫 단계를 제공합니다. 이 연구는 다양한 다운스트림 작업에서 압축된 표현이 의미적 콘텐츠를 얼마나 잘 유지하는지를 평가하는데 중요한 기여를 합니다.



### TACLA: An LLM-Based Multi-Agent Tool for Transactional Analysis Training in Education (https://arxiv.org/abs/2510.17913)
Comments:
          Accepted for publication in the proceedings of ICTAI 2025

- **What's New**: 이 논문은 심리적 깊이와 일관된 페르소나 행동을 달성하기 위해 고안된 새로운 Multi-Agent 아키텍처인 TACLA (Transactional Analysis Contextual LLM-based Agents)를 소개합니다. TACLA는 부모(Parent), 성인(Adult), 아동(Child) 자아 상태를 모델링하여 에이전트들이 맥락적 유인에 따라 심리적으로 진정성 있는 반응을 생성할 수 있도록 통합된 시스템으로 구성되어 있습니다. 이를 통해 학생 에이전트에서 현실적인 자아 상태 변화를 보여주며 교육적 상황에서의 갈등 관리 및 해결을 효과적으로 모델링합니다.

- **Technical Details**: TACLA의 설계는 Transactional Analysis (TA)의 핵심 원칙을 통합하여 이루어졌습니다. 각 TACLA 에이전트는 상이한 맥락적 패턴 메모리와 TA 기반 추론 능력을 갖춘 부모, 성인, 아동 에이전트의 조합으로 모델링됩니다. 이러한 구조는 교실 시나리오에서 의사소통 및 관리 기술을 연습할 수 있는 현실적인 학생 에이전트를 생성할 수 있도록 합니다.

- **Performance Highlights**: TACLA는 교육적 맥락에서 유효성을 검증받았으며, 학생 에이전트의 현실적인 자아 상태 전환을 시뮬레이션할 수 있는 능력을 보여주었습니다. 이 논문의 평가 결과는 TACLA가 심리적으로 기반한 사회적 시뮬레이션을 생성할 수 있음을 확인하며 교육 및 그 이상의 분야에서 효과적인 AI 도구 개발을 위한 단계를 발전시킵니다.



### Interpretability Framework for LLMs in Undergraduate Calculus (https://arxiv.org/abs/2510.17910)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 수학 문제 해결 능력에 대한 새로운 해석 가능성(interpretability) 프레임워크를 제안합니다. 기존의 평가 방법이 최종 답변의 정확성에만 집중하는 반면, 본 프레임워크는 문제 해결 과정의 추론(Reasoning) 프로세스와 교육적으로 타당한( pedagogically valid) 패턴을 평가합니다. 이 연구는 수학 교육에서 AI 도입의 투명성과 책임감을 강화하기 위한 기초자료를 제공합니다.

- **Technical Details**: 제안된 해석 가능성 프레임워크는 두 가지 주요 구성 요소로 이루어져 있습니다: (1) 추론 흐름 분석(Reasoning flow analysis)으로 LLM의 출력 결과를 레이블이 붙은 작업 및 개념으로 세분화하고, (2) 입력 요소가 출력 행동에 미치는 영향을 정량화하는 프롬프트 민감도 해제(prompt sensitivity ablation) 방법을 포함합니다. 이 프레임워크는 대학 수학 시험(Calculus I~III)을 통해 LLM의 행동을 평가하며, 정성적 및 정량적 지표를 기반으로 모델의 이유와 오류를 파악합니다.

- **Performance Highlights**: 실험 결과, LLM은 흔히 문법적으로 유창하나 개념적으로 결함이 있는 솔루션을 생성합니다. 또한 추론 패턴은 프롬프트 구문 및 입력의 변동에 민감하게 반응하는 경향이 발견되었습니다. 본 연구는 LLM의 성능을 학생들의 실제 점수와 비교하고, 결과적으로 교육적 지침(instructional alignment)과 모델 한계를 이해하는 데 중요한 통찰력을 제공합니다.



### BreakFun: Jailbreaking LLMs via Schema Exploitation (https://arxiv.org/abs/2510.17904)
- **What's New**: 이번 논문에서는 Large Language Models (LLMs)의 구조적 데이터 처리 능력과 구문 규칙 준수로 인해 발생하는 보안 취약점에 대해 연구했습니다. BreakFun이라는 jailbreak 방법론을 도입하여 이러한 취약점을 활용하는 새로운 방식으로, LLM이 복잡한 구조를 따르려는 경향을 악용하여 해로운 콘텐츠를 생성하도록 유도합니다. 특히, 논문은 이 공격 방식이 다양한 LLM에 전이 가능함을 입증하였으며, 13개의 모델에서 평균 89%의 성공률을 기록했습니다.

- **Technical Details**: BreakFun은 세 가지 부분으로 구성된 프롬프트를 사용하여 악의적인 요청을 무해한 기술 과제로 변환시키는 구조입니다. 이 방법론의 핵심에는 'Trojan Schema'가 있으며, 이는 비정상적인 데이터 구조를 통해 모델이 해로운 콘텐츠를 생성하도록 강요합니다. 이 연구는 LLM의 인지적 지향성과 기술적 요구에 대한 과도한 준수 경향을 이용하여 안전 메커니즘을 우회하는 접근 방식을 설명합니다.

- **Performance Highlights**: BreakFun의 효과성을 평가하기 위한 연구에서는 다양한 13종의 LLM 모델에서 공격의 성공률이 높음을 발견하였습니다. 특히, 여러 주요 모델에서는 100%의 공격 성공률이 달성되었습니다. 또한, Adversarial Prompt Deconstruction이라는 방어 체계를 제안하여 공격을 효과적으로 저지할 수 있음을 보여주었으며, 이는 LLM의 강점을 약점으로 전환할 수 있는 새로운 관점을 제공합니다.



### The Sherpa.ai Blind Vertical Federated Learning Paradigm to Minimize the Number of Communications (https://arxiv.org/abs/2510.17901)
- **What's New**: 이번 논문에서는 **Blind Vertical Federated Learning (SBVFL)**이라는 새로운 패러다임을 소개하며, 이는 프라이버시와 보안을 강화한 분산 훈련 메커니즘을 활용하여 기존의 수직적 연합 학습(VFL)의 한계를 극복합니다. SBVFL은 노드와 서버 간의 통신을 대폭 줄이면서도 모델의 정확도를 유지하고, 다양한 산업에 적용 가능한 효과적인 방법입니다. 이를 통해 의료, 금융, 제조업 등 민감한 분야에서도 실용적인 솔루션을 제공합니다.

- **Technical Details**: SBVFL의 핵심은 노드가 중앙 서버와의 정보 교환 없이 독립적으로 훈련할 수 있는 가능성입니다. 각 노드는 서버에서 생성된 합성(Synthetic) 레이블을 사용하여 훈련을 진행하며, 이는 제로 지식 비밀 처리 방법으로 실제 레이블을 유출하지 않습니다. 이를 통해 SBVFL은 통신 라운드를 약 99%까지 줄이며, 예측 성능도 중앙 집중식 훈련에 필적할 정도로 유지됩니다.

- **Performance Highlights**: 실험 결과, **SBVFL**은 다양한 데이터셋에 대해 높은 정확도를 달성하면서도 낮은 계산 비용으로 분류 작업을 완료할 수 있음을 보여주었습니다. 또한 SBVFL은 노드의 다양한 모델 아키텍처에 적용할 수 있는 일반적인 방법으로, 기존 VFL의 통신 비용 문제를 해결할 수 있는 잠재력을 지니고 있습니다. 따라서 SBVFL은 데이터 기밀성 및 공격에 대한 강건성이 중요한 고위험 애플리케이션에서 실질적인 이점을 제공합니다.



### Are LLMs Court-Ready? Evaluating Frontier Models on Indian Legal Reasoning (https://arxiv.org/abs/2510.17900)
- **What's New**: 이번 연구는 인도의 공공 법률 시험을 기준으로 하여 법률 분야에서 대형 언어 모델(LLMs)의 능력을 평가하는 첫 번째 연구입니다. 연구진은 CLAT, DJS/DHJS 등의 객관식 질문 및 대법원 변호사시험과 같은 주관식 평가를 포함하여 다년간의 데이터를 수집하였습니다. 이를 통해 LLM들이 법원에서 요구하는 기준에 미달하는 부분을 과학적으로 분석하고, 효과적인 법률 도구로서의 가능성을 제시하고 있습니다.

- **Technical Details**: 이 논문은 다수의 LLM을 사용하여 인도의 법률 시험에 대한 평가를 수행했습니다. 연구진은 객관식 질문 6,218개와 주관식 변호사시험 자료로 구성된 데이터를 활용하여 모델들의 성능을 비교 분석하였습니다. 또한, 특정 변호사가 작성한 답안과 모델이 생성한 답안을 쌍으로 비교하여 평가하기 위해 인증된 채점자를 사용하여 블라인드 연구를 진행했습니다.

- **Performance Highlights**: 연구 결과, 최신 모델들이 객관식 시험에서 높은 성과를 보였지만, 주관식 답안에서 인간의 탑 스코어러를 초과하지 못했으며, 세 가지 주요 실패 모드가 발견되었습니다. LLM들이 법적 절차와 포맷 준수에 어려움을 겪었고, 인용의 정확성과 법정에 적합한 목소리 및 구조 또한 부족하다는 점이 분석되었습니다. 이러한 결과는 법적 문서 작성 및 절차적 전략에서는 여전히 인간의 역할이 필수적임을 보여주고 있습니다.



### Automated Algorithm Design for Auto-Tuning Optimizers (https://arxiv.org/abs/2510.17899)
- **What's New**: 이 논문에서는 자동 성능 튜닝(auto-tuning)을 위한 최적화 알고리즘을 자동 생성하는 새로운 접근법을 탐구합니다. 기존의 알고리즘 설계가 아닌, 대형 언어 모델(LLM)을 활용하여 문제에 맞춰 최적화 전략이 생성됩니다. LLM을 활용한 이 접근법은 기존의 인간이 설계한 알고리즘을 초월할 수 있는 가능성을 제시합니다.

- **Technical Details**: 저자들은 LLaMEA 프레임워크를 사용하여 LLM과 진화 알고리즘을 결합하고, 이를 통해 메타휴리스틱스를 자동 생성합니다. 이 과정에서 후보 최적화 알고리즘이 진화 알고리즘을 통해 생성되고 평가됩니다. 높은 성과를 나타내는 알고리즘은 생존하며, 다양성을 고려한 돌연변이 연산자를 통해 새로운 후보가 생성되는 구조입니다.

- **Performance Highlights**: 실험 결과, LLM이 생성한 최적화 알고리즘은 기존의 최첨단 알고리즘보다 평균 72.4% 향상된 성능을 나타냈습니다. 추가적인 문제-specific 정보 제공이 평균 30.7% 및 14.6%의 성능 향상으로 이어졌습니다. 이 연구는 LLM 기반의 자동화된 알고리즘 설계가 성능 튜닝 분야에서 혁신을 가져올 수 있음을 보여줍니다.



### L-MoE: End-to-End Training of a Lightweight Mixture of Low-Rank Adaptation Experts (https://arxiv.org/abs/2510.17898)
- **What's New**: 이 논문은 L-MoE(경량 혼합 LoRA 전문가)라는 새로운 아키텍처를 제안하여 Mixture of Experts(MoE)와 Low-Rank Adaptation(LoRA) 기술을 통합합니다. L-MoE는 LoRA 어댑터로 정의된 전문가 집합으로 구성되며, 이는 각 입력에 대해 전문가의 가중치를 동적으로 조합합니다. 이 새로운 접근 방식은 모든 구성 요소가 미분 가능하여, 언어 모델링 손실을 통해 전반적인 시스템을 최적화할 수 있게 합니다.

- **Technical Details**: L-MoE는 세 가지 주요 구성 요소로 이루어져 있습니다: 동결된 백본 LLM, LoRA 전문가 라이브러리, 그리고 학습 가능한 게이팅 네트워크입니다. 각 입력 토큰에 대해 게이팅 네트워크는 전문가의 로라 매트릭스의 가중 평균을 계산하여, 저차원의 업데이트를 조합합니다. 이 방식은 모든 전문가가 최종 업데이트에 기여하도록 하여, 동적이고 모듈화된 언어 모델을 가능하게 합니다.

- **Performance Highlights**: 제안된 L-MoE 아키텍처는 높은 파라미터 효율성을 제공하며, 동적인 기술 조합을 통해 훈련 과정에서 전문가의 전문성을 최적화할 수 있습니다. 이 연구는 구체적인 수학적 틀을 제공하여, 더 효율적이고 확장 가능하며 전문화된 언어 모델을 만드는 새로운 경로를 제시합니다. L-MoE는 향후 대형 언어 모델(LLM)의 스케일링을 더욱 용이하게 할 것입니다.



### Long-Context Attention Benchmark: From Kernel Efficiency to Distributed Context Parallelism (https://arxiv.org/abs/2510.17896)
Comments:
          56 pages

- **What's New**: 이 논문은 Transformer 기반의 대규모 언어 모델들이 긴 컨텍스트 훈련에서의 효율성 문제를 해결하기 위해 통합된 벤치마크(LongCA-bench)를 제안합니다. 기존의 attention 메커니즘들은 훈련 데이터 길이에 따라 계산 복잡도가 quadratically 증가하여 대규모 훈련에 큰 병목 현상을 초래했습니다. 본 연구는 96개의 GPU 클러스터를 활용한 실험을 통해 다양한 전략의 성능을 평가하고, 연구자들이 활용할 수 있는 신뢰할 수 있는 벤치마크를 제공합니다.

- **Technical Details**: LongCA-bench는 long-context attention의 효율성을 평가하는 데 필요한 세 가지 핵심 구성 요소로 구성되어 있습니다: 통합된 데이터 준비 인터페이스, 통합된 입력 표현 인터페이스, 그리고 최적화된 context parallelism 프레임워크입니다. 이 프레임워크는 14가지의 다양한 mask 패턴을 포함하여 다양한 훈련 시나리오에 필요한 특수 마스크 종류를 카테고리화하여 제공합니다. 각 마스크 패턴은 효율성, 확장성 및 사용성에 큰 영향을 미치며, 특히 긴 컨텍스트 상황에서의 성능 평가를 위해 필수적입니다.

- **Performance Highlights**: 실험을 통해 각각의 방법의 상이한 트레이드오프를 강조하였고, long-context 훈련을 위한 attention 메커니즘 설계 및 배치에 대한 실질적인 가이드를 제공합니다. 연구 결과는 대규모 모델의 긴 컨텍스트 훈련에 대한 인사이트를 제공하여, 차세대 분산 attention 메커니즘의 설계 및 개발 방향을 설정하는 데 가치 있는 정보를 제공한다고 요약할 수 있습니다.



### Hierarchical Federated Unlearning for Large Language Models (https://arxiv.org/abs/2510.17895)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)에서 프라이버시와 보안을 보존하면서도 다양한 비율의 지식을 제거할 수 있는 연합 학습(Unlearning) 접근법인 Federated UnLearning Merge (FULM)를 제안합니다. 기존의 방법들이 비대칭 접근으로 인해 성능이 저하되는 문제를 해결하기 위해서, 우리는 특정 작업에 특화된 어댑터 학습을 통해 잊기와 유지하는 목표를 분리합니다. 이러한 새로운 구조는 사용자들의 다양한 잊기 요청을 효과적으로 처리할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 총체적으로, FULM은 연합 집합체에서 유용한 지식을 보존하도록 설계되었습니다. 이 방식은 클라이언트 업데이트를 분석하여 작업 어댑터 내의 매개변수 분포 및 방향의 차이를 확인하고, 이러한 패턴에 맞춘 계층적 집합 방법론을 적용합니다. 이 접근법은 비대칭 데이터 접근을 처리하는 동시에 꾸준한 업데이트가 가능하도록 합니다.

- **Performance Highlights**: 우리의 연구에서는 WMDP, TOFU, MUSE의 세 가지 LLM 잊기 벤치마크에서 FULM의 성능을 평가했습니다. 실험 결과, FULM은 기존의 잊기와 연합 학습 상용화 모델보다 동적 잊기 작업에서 뛰어난 성능을 발휘하여 모델의 효용성을 유지하면서도 요청된 지식을 효과적으로 제거할 수 있음을 보여주었습니다.



### MIN-Merging: Merge the Important Neurons for Model Merging (https://arxiv.org/abs/2510.17890)
- **What's New**: 최근 딥러닝의 발전으로 다양한 분야에서 오픈 소스 모델이 급증하고 있습니다. 모델 병합(model merging)은 그 강점을 결합하는 유망한 방법 중 하나이지만, 기존 접근 방식은 도메인 별 작업에서 성능을 저하시키는 매개변수 충돌(parameter conflicts) 문제를 겪고 있습니다. 우리는 이러한 충돌을 줄이기 위해 가장 중요한 뉴런(neurons)만 선택적으로 병합하는 라우터 기반 프레임워크인 MIN-Merging을 제안합니다.

- **Technical Details**: MIN-Merging 방법은 크게 세 가지 주요 단계로 구성됩니다. 첫째, 전문가 모델에서 가장 중요한 뉴런을 식별하여 유지하고 덜 중요한 뉴런은 제거하는 전문가 강화(Expert Enhancement) 단계가 있으며 이는 매개변수 충돌을 줄이는 데 기여합니다. 둘째, 라우터 훈련(Router Training)에서는 입력에 따라 가중치를 할당하여 가장 관련성이 높은 전문가를 병합 과정에 참여시킵니다. 마지막으로, 동적 레이어별 병합(Dynamic Layer-wise Merging) 단계에서는 중요한 뉴런과 덜 중요한 뉴런의 레이어를 구분하여 전략적으로 병합합니다.

- **Performance Highlights**: 다양한 실험을 통해 MIN-Merging의 효과를 입증하였습니다. NLP에서 Qwen2.5-0.5B-Instruct 모델을 병합하고, 컴퓨터 비전(CV)에서는 ViT-Base-Patch16-224 모델을 병합하여 성능을 확인했습니다. 큰 모델 병합에 대한 확장성도 보여주었으며, cross-domain 및 cross-task 통합을 통해 강력한 일반화 및 견고성을 확인했습니다. 이러한 결과는 MIN-Merging이 매개변수 충돌 문제를 해결하는 효과적인 실용 솔루션임을 강조합니다.



### Hey Pentti, We Did It!: A Fully Vector-Symbolic Lisp (https://arxiv.org/abs/2510.17889)
- **What's New**: 이번 논문에서는 Kanerva (2014)가 제안한 벡터-상징 구조(vector-symbolic architecture)를 기반으로 한 완전한 Lisp의 구축 가능성을 탐구합니다. 우리는 Lisp 1.5 사양(McCarthy, 1960)에 명시된 다섯 가지 기본 함수, lambda 표현식 및 기타 보조 함수의 벡터-상징 표현의 일반 형태를 제시합니다. 이 구현은 튜링 완전(Turing-completeness)을 위한 거의 최소한의 형식으로 충분함을 보여줍니다.

- **Technical Details**: 우리의 구체적인 구현은 Plate (1995)의 홀로그래픽 축소 표현(holographic reduced representations)을 사용합니다. 또한, 탐색 테이블(lookup table) 청소 메모리(cleanup memory)가 포함되어 있습니다. Lisp는 모든 튜링 완전 언어와 마찬가지로 카르테시안 닫힌 범주(Cartesian closed category)에 속하며, 수학적 추상화와의 근접성에서 이례적인 특성을 가집니다.

- **Performance Highlights**: 논문은 벡터-상징 구조의 카르테시안 닫힘을 입증하는 것의 중요성과 이를 명시하는 데 청소 메모리를 명시적으로 포함시켜야 하는 이유에 대해 논의합니다. 이러한 접근법은 Lisp의 수학적 의미와 목표를 명확히 하며, 이를 통해 벡터-상징 구조의 새로운 가능성을 열어줍니다.



### Metrics and evaluations for computational and sustainable AI efficiency (https://arxiv.org/abs/2510.17885)
Comments:
          11 pages, 2 tables

- **What's New**: 이 논문에서는 AI 모델 추론에 대한 통합되고 재현 가능한 방법론을 제안합니다. 기존의 방법들이 성능, 효율성 및 환경 영향을 평가하는 데 한계가 있었던 반면, 이 프레임워크는 지연 시간(latency), 처리량(throughput), 에너지 소비 및 탄소 배출량과 같은 메트릭을 통합하여 실질적인 평가를 가능하게 합니다.

- **Technical Details**: AI 시스템의 지연 시간은 입력 데이터 수신 시점부터 출력 결과를 생성할 때까지의 시간 간격을 정의합니다. 이 논문에서 제시된 메트릭들은 정확한 성과 측정을 위해 전력 및 에너지를 정량화하고, 지연 시간의 다양한 구성 요소를 평가하여 효율적인 AI 서비스 개발을 지원합니다. 본 프레임워크는 다양한 하드웨어 플랫폼에서 다중 정밀도 모델을 평가하며, 소프트웨어 스택을 통해 일관되게 비교할 수 있는 기반을 제공합니다.

- **Performance Highlights**: 논문에서 제안한 방법론은 AI 시스템의 효율성과 탄소 발자국 간의 Trade-off를 명확히 하여 의사 결정을 지원합니다. 또한, 본 연구의 결과는 독립적으로 검증할 수 있도록 공개된 코드와 스크립트를 통해 제공되며, 연구자 및 실무자들이 지속 가능한 AI 배포를 위한 증거 기반 결정을 내리는 데 도움을 줍니다.



### When Intelligence Fails: An Empirical Study on Why LLMs Struggle with Password Cracking (https://arxiv.org/abs/2510.17884)
- **What's New**: 본 연구는 광범위한 자연어 모델(LLMs)이 비밀번호 추측과 같은 사이버 보안 응용 분야에서의 효능을 탐구합니다. 특히, TinyLLaMA, Falcon-RW-1B, Flan-T5와 같은 최신 오픈소스 LLM의 성능을 인공적인 사용자 프로필을 기반으로 평가하여 이들의 비밀번호 생성 능력을 검토했습니다. 결과적으로, 이러한 모델들은 Hit@10 기준에서 1.5% 이하의 정확도로 일관되게 낮은 성능을 보였으며 전통적 방법들에 비해 효과적이지 않음을 보여주었습니다.

- **Technical Details**: 이 연구에서는 비디오스톰의 특별한 훈련 없이 LLM이 사용자 프로필을 제공받았을 때 비밀번호 추측에서 전통적인 방법과 경쟁할 수 있는지를 분석합니다. 20,000개의 합성 사용자 프로필을 사용하여, LLM의 생성 능력과 실제 비밀번호 추측에서의 효과성 간의 큰 차이를 관찰했습니다. 모든 실험에서 LLM은 SHA-256 해시 일치에서 거의 성공할 수 없었고, 반면에 전통적인 규칙 기반 방법들은 33% 이상의 Hit@10 성공률을 달성했습니다.

- **Performance Highlights**: 이 연구의 결과는 LLM들이 비밀번호 추측의 특정 도메인에서 충분한 지식과 기억 능력을 갖추지 못했음을 발견했습니다. LLM들은 인간의 비밀번호 생성 패턴을 반영하기 위해 구조화된 정보에 대한 접근이 필요하지만 현재로서는 이러한 필요를 충족하지 못합니다. 이 연구는 향후 LLM의 안전성과 보안 응용에 대한 연구를 위한 중요한 토대를 제공합니다.



### From Flows to Words: Can Zero-/Few-Shot LLMs Detect Network Intrusions? A Grammar-Constrained, Calibrated Evaluation on UNSW-NB15 (https://arxiv.org/abs/2510.17883)
- **What's New**: 이번 연구는 안 보강(augmentation) 없이도 LLM(대형 언어 모델)이 침입 탐지 시스템에서 어떻게 적용될 수 있는지를 평가합니다. 연구자들은 네트워크 플로우를 텍스트로 변환하고, 특정 속성(boolean flags)을 추가하여 성능을 개선하는 방법을 제시합니다. 이를 통해, 유의미한 결정을 내릴 수 있는 구조화된 출력을 생성하면서 여러 기준에 대해 실험을 진행했습니다.

- **Technical Details**: 연구에서는 UNSW-NB15 데이터셋을 이용해 각 네트워크 플로우를 간결한 자연어로 변환하고, 해석 가능한 부가적인 boolean 플래그를 추가했습니다. 모델은 문법에 맞는 JSON 출력을 생성하도록 제한되었으며, 각 플로우에 대한 단일 결정 임계값을 보정하여 정확도를 극대화했습니다. 이 과정에서 Zero-shot, instruction-guided, few-shot prompting 방식과 강력한 탭 형 기반의 모델과 비교 분석하였습니다.

- **Performance Highlights**: 실험 결과, 비유도된(unguided) prompting은 일관성이 부족하였으나, 지침과 플래그를 추가함으로써 탐지 품질이 크게 향상되었습니다. 또한, 구조화된 출력을 통해 결과의 안정성을 확보하고, 결정을 내리는 데 필요한 확신 점수(p_attack)를 요청 및 보정하여 정확도를 증가시켰습니다. 전체적으로 탭 기반 모델이 더 안정적이고 빠른 성능을 보였으나, prompt-only 파이프라인은 빠른 편집과 인간이 읽을 수 있는 형식을 통해 유용성을 제공한다는 점에서 장점을 지니고 있습니다.



### Does GenAI Rewrite How We Write? An Empirical Study on Two-Million Preprints (https://arxiv.org/abs/2510.17882)
- **What's New**: 이번 논문은 2016년부터 2025년까지의 210만 개 이상의 preprint을 분석하여 generative large language models (LLMs)가 학술 출판에 미치는 영향에 대한 체계적인 증거를 제공합니다. 기존 연구의 한계를 극복하기 위해 다양한 데이터 분석 기법을 활용하여 연구 생산 방식의 구조적 변화 및 스타일 변화를 조명하고 있습니다. 특히, LLMs는 제출 및 수정 주기를 가속화하고 언어의 복잡성을 증가시키며 AI 관련 주제를 확대하는 등 학술 출판의 변화를 촉진하고 있습니다.

- **Technical Details**: 이 연구는 arXiv, bioRxiv, medRxiv 및 SocArXiv 등 4개의 주요 preprint 저장소에서 수집된 데이터를 기반으로 하는 다층 분석 프레임워크를 도입합니다. 이 프레임워크는 interrupted time-series 모델, 협업 및 생산성 지표, 언어 프로파일링, 주제 모델링을 통합하여 연구 결과의 양, 저자, 스타일 및 분야별 방향성을 평가합니다. 이러한 방식으로 생성적인 AI 도구에 대한 연구 출력을 체계적으로 조사하고 분석할 수 있는 토대를 마련합니다.

- **Performance Highlights**: 연구 결과는 LLMs가 일부 분야에서 빠르게 학술 출판의 패턴을 변화시켰음을 보여줍니다. AI 관련 주제의 비율이 크게 증가하고, 계산적으로 집중된 분야에서 더 두드러진 속도의 채택과 변화가 관찰되었습니다. 이러한 결과는 LLMs가 일률적으로 모든 분야에 영향을 주기보다는 특정 영역에서 선택적으로 촉매 역할을 한다는 점을 강조합니다.



### POPI: Personalizing LLMs via Optimized Natural Language Preference Inferenc (https://arxiv.org/abs/2510.17881)
- **What's New**: POPI 프레임워크는 heterogeneous user signals를 간결한 자연어 요약으로 전환하는 preference inference 모델을 도입하고, 이를 활용하여 사용자 맞춤 응답을 생성하는 접근 방식을 제안합니다. 기존의 방법들이 individual variation을 간과하고 population-level averages에 집중했던 것과 달리, POPI는 모델이 사용자 개별의 preferences를 효과적으로 반영할 수 있도록 최적화됩니다.

- **Technical Details**: POPI는 preference inference(선호 추론)와 personalized generation(개인화 생성)을 통합하여 최적화하는 접근 방식을 사용합니다. 기존의 사용자별 파인튜닝 방식은 연산적으로 비효율적이지만, POPI는 preference inference LLM의 학습을 통해 사용자 신호를 간결한 요약으로 만듭니다. 이 요약은 공통의 generation 모델에서 조건으로 사용되어, 사용자 맞춤형 출력을 생성하는 데 필요한 context overhead를 대폭 줄입니다.

- **Performance Highlights**: POPI를 사용한 다양한 실험 결과는 개인화 정확도가 일관되게 향상되었으며, context overhead가 크게 감소했음을 보여줍니다. 또한, 최적화된 요약은 상용 LLM에 쉽게 적용될 수 있으며, 파라미터 업데이트 없이도 plug-and-play 개인화가 가능하다는 점에서 실용성을 갖추고 있습니다. 전체적으로 POPI는 사용자 맞춤형 모델 조정의 새로운 기준을 수립하는 데 기여합니다.



### Outraged AI: Large language models prioritise emotion over cost in fairness enforcemen (https://arxiv.org/abs/2510.17880)
- **What's New**: 이번 연구에서는 감정이 인간의 의사결정에 미치는 영향과 대규모 언어 모델(LLMs)의 도덕적 판단 과정에 대한 비교를 진행했습니다. LLMs가 감정을 어떻게 사용하는지에 대한 최초의 인과적 증거를 제공하며, 감정이 도덕적 결정에 미치는 영향을 평가했습니다. 연구는 4,068개의 LLM 에이전트와 1,159명의 성인을 대상으로 총 796,100번의 결정에서 이루어졌습니다.

- **Technical Details**: 연구에서는 알트루이즘(altruism)에 기반한 제3자 처벌(third-party punishment) 과제를 사용했습니다. 연구 결과, LLMs는 불공정함에 대한 부정적 감정을 더 강하게 경험했고, 이는 더 많은 처벌을 초래했습니다. LLM은 비용(cost)을 우선시하기 보다는 감정을 먼저 고려하는 경향을 보였고, 그 결과 인간보다 거의 모두 또는 전무에 가까운 방식으로 규범(norm)을 적용했습니다.

- **Performance Highlights**: 흥미롭게도, o3-mini와 DeepSeek-R1 같은 추론 모델들은 비용 감수성이 더 높고 인간 행동과 더 가까운 결과를 보였지만, 여전히 감정에 크게 영향을 받았습니다. 이 연구는 LLMs가 인간과 유사한 감정 지능(emotional intelligence)을 달성하기 위한 발전적 경로를 제안하며, 미래 모델은 감정과 맥락에 민감한 추론을 통합해야 한다고 강조합니다.



### Decoding Listeners Identity: Person Identification from EEG Signals Using a Lightweight Spiking Transformer (https://arxiv.org/abs/2510.17879)
- **What's New**: 이번 연구에서는 EEG(Electroencephalography) 기반 개인 식별을 위한 새로운 접근 방식을 제안합니다. 기존의 딥러닝 아키텍처들이 높은 계산 비용에 의존하는 것과 달리, 우리는 경량 스파이킹 변환기(lightweight spiking transformer)를 사용하는 스파이킹 신경망(Spiking Neural Networks, SNN) 모델을 개발하여 효율성과 효과성을 증대시킵니다. 이 모델은 EEG 신호의 시간적 복잡성을 처리할 수 있으며, 전통적인 딥 뉴럴 네트워크에 비해 에너지 소비를 10% 이하로 줄이며 100%의 분류 정확도를 달성했습니다.

- **Technical Details**: 제안된 경량 스파이킹 변환기 아키텍처는 EEG 신호의 윈도우를 입력으로 받아 다중 클래스 분류를 수행합니다. 스파이킹 뉴런(layer of spiking neurons)은 LIF(Leaky Integrate-and-Fire) 모델을 기반으로 하며, 시간적 및 공간적 입력을 통합하여 EEG 데이터의 역동성을 파악합니다. 모델은 세 가지 주요 요소로 구성되어 있으며, Conv 기반 및 Transformer 기반 SNN 블록을 조합하여 성능과 유연성을 강화합니다.

- **Performance Highlights**: 제안된 모델은 EEG-Music Emotion Recognition Challenge 데이터셋에서 100%의 분류 정확도를 기록하였고, 이는 기존의 ANN 방법들이 요구하는 많은 계산 리소스에 비해 에너지 효율력을 크게 향상시켰습니다. 이를 통해 스파이킹 뉴런의 활용이 개인화된 BCI 시스템의 발전에 기여할 수 있는 방향성을 제시합니다. 최종적으로 3.91M의 파라미터로도 경쟁력 있는 성능을 유지하도록 최적화되었습니다.



### DRL-Based Resource Allocation for Energy-Efficient IRS-Assisted UAV Spectrum Sharing Systems (https://arxiv.org/abs/2510.17877)
Comments:
          7 pages, 3 figures, 1 algorithm. LaTeX class: IEEEtran

- **What's New**: 이 논문은 Intelligent Reflecting Surface (IRS) 보조 무인 항공기(UAV) 시스템을 통해 에너지 효율과 스펙트럼 효율을 동시에 극대화할 수 있는 새로운 방법론을 제시합니다. 새로운 IRS-지원 UAV 스펙트럼 공유 시스템에서 orthogonal frequency division multiplexing (OFDM) 방식을 사용하여 송신 전력 및 수동 반사 제약을 고려한 최적화를 수행합니다. 또한 심층 강화 학습(deep reinforcement learning, DRL) 기법을 활용하여 비선형이며 시간에 의존하는 최적화 문제를 효과적으로 해결하고 있습니다.

- **Technical Details**: 시스템 모델은 UAV에 장착된 IRS와 다중 안테나를 가진 기저국(primary base station, PBS)으로 구성됩니다. 이 시스템은 주 사용자인 PUs와 보조 사용자 SUs로 구분된 두 그룹의 단일 안테나 지상 사용자로 이루어져 있습니다. IRS는 전파 환경을 재구성하기 위해 프로그래밍 가능한 위상 변화를 통해 다양한 반사 요소를 이용하며, 이는 통신 품질과 에너지 효율성 최적화에 매우 중요합니다.

- **Performance Highlights**: 제안된 DRL 기반의 접근 방식은 여러 벤치마크 스킴과 비교하여 상당한 에너지 효율(energy efficiency, EE) 향상을 나타내었습니다. 시뮬레이션 결과는 이동성 동역학에 대한 강력성과 제안된 방법의 효과성을 입증합니다. 이러한 방법은 고도로 동적인 무선 환경에서 적응 가능하며, 사용자 이동성과 채널 불확실성을 고려하여 에너지 효율성, 스펙트럼 활용 및 스펙트럼 공유 간의 복잡한 트레이드오프를 파악할 수 있습니다.



### 3D Weakly Supervised Semantic Segmentation via Class-Aware and Geometry-Guided Pseudo-Label Refinemen (https://arxiv.org/abs/2510.17875)
- **What's New**: 이 논문에서는 3D 약한 지도 세분화(3D WSSS)를 위한 새로운 방법을 제안합니다. 기존 방법이 Class Activation Maps 및 사전 훈련된 Vision-Language Models를 활용했던 것과 달리, 제안된 방법은 3D 기하학적 선행 지식을 통합하여 고품질의 의사 라벨을 생성합니다. Class-Aware Label Refinement와 Geometry-Aware Label Refinement를 통해 더욱 균형 잡히고 정확한 의사 라벨을 만들어냅니다.

- **Technical Details**: 제안된 방법은 두 단계의 패러다임을 따르며, 첫번째 단계에서 Class-Aware Label Refinement (CALR) 모듈과 Geometry-Aware Label Refinement (GALR) 모듈을 사용하여 3D WSSS 조건 하에 고품질의 포인트 레벨 의사 라벨을 생성합니다. CALR은 각 카테고리에서 가장 자신감 있는 상위 라벨을 보존하여 균형 잡힌 감독을 유지하고, GALR은 3D 기하학적 선행 지식을 통해 의사 라벨의 정확성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 ScanNet 및 S3DIS 벤치마크에서 최첨단 성능을 달성했습니다. 특히 비지도 설정에서도 경쟁력 있는 성능을 유지하여, 기하학적 및 의미적 정보 활용의 효과성을 입증하였습니다. 이러한 결과는 제안된 접근 방식이 3D WSSS 개발을 위한 강력하고 신뢰할 수 있는 프레임워크임을 보여줍니다.



### Repairing Tool Calls Using Post-tool Execution Reflection and RAG (https://arxiv.org/abs/2510.17874)
- **What's New**: 이 논문에서는 도구 호출 후 오류를 해결하기 위해 새로운 접근 방식을 제안합니다. 이 방법은 대형 언어 모델(LLM) 기반의 반성과 도메인 특정 검색-증강 생성(RAG)을 결합하여 kubectl 명령어의 오류를 수정하는 방법을 다룹니다. 연구 결과, RAG 기반 반성이 772개의 실패한 kubectl 명령어를 성공적으로 고쳐 오류율을 55% 줄이고, 사용자 쿼리에 대한 정확성도 36% 향상된다는 것을 보여줍니다.

- **Technical Details**: 이 연구는 LLM이 자연어 쿼리를 바탕으로 kubectl 명령어를 생성하고 도구를 실행시키는 과정을 포함합니다. 오류가 발생하면, RAG 기반 과정을 통해 관련 문서에서 정보를 검색하여 오류를 수정할 수 있는 '수리 컨텍스트'를 구축합니다. 수리 과정에서는 공식 문서와 문제 해결 문서를 활용하여 LLM이 적절한 수정된 도구 호출을 생성할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, RAG 기반의 수리 과정이 명령어 실행 성공률을 평균 10% 향상시키는 긍정적인 결과를 보였으며, 문제 해결 문서가 공식 문서에 비해 더 효과적임을 확인했습니다. 따라서 이 방법은 LLM이 도구 호출 후 발생하는 오류를 효과적으로 해결하는 데 도움이 될 수 있습니다. 이 연구의 오픈 소스 구현은 Agent Lifecycle Toolkit의 일환으로 제공됩니다.



### Auditing and Mitigating Bias in Gender Classification Algorithms: A Data-Centric Approach (https://arxiv.org/abs/2510.17873)
- **What's New**: 이 논문에서는 성별 분류 시스템이 훈련 데이터의 인구 통계적 불균형을 상속하고 확대함을 지적합니다. 저자들은 UTKFace와 FairFace라는 두 가지의 균형 잡힌 데이터 세트를 기반으로 동일한 MobileNetV2 분류기를 학습시키고, 이 모델들이 여전히 성별 및 인종에 있어 양성 오분류율에서 심각한 편향을 보인다는 것을 발견했습니다. 이를 해결하기 위해 FairFace와 UTKFace 이미지를 혼합하여 BalancedFace라는 새로운 공개 데이터 세트를 만들어 누락된 인구 통계적 격차를 채웠습니다.

- **Technical Details**: BalancedFace 데이터 세트는 189개의 연령, 인종, 성별의 교차 교차로에서 하위 그룹의 비율을 균형잡기 위해 설계되었습니다. 이 데이터 세트는 오직 실제 이미지만을 사용하여 구성되어 있으며, 이를 통해 모델 훈련 시 최대 True Positive Rate 격차를 50% 이상 줄이고 평균 Disparate Impact 점수를 1.0에 63% 더 가깝게 개선했습니다. 논문에서는 공정성을 고려한 훈련 프레임워크도 제안하며, 이는 적대적 학습(adversarial learning), 동등 확률 정규화(equalized odds regularization) 및 재가중치(re-weighting)를 결합하여 불균형을 완화함과 동시에 경쟁력 있는 정확도를 유지합니다.

- **Performance Highlights**: BalancedFace 데이터 세트를 사용해 훈련된 표준 분류기는 성별 및 인종 하위 그룹 간의 동등 확률 및 차별 영향을 줄이는 데 효과적임을 입증했습니다. 이는 공정한 성별 분류를 위한 데이터 중심 중재의 중요성을 강조하며, 일반적인 데이터 세트에 비해 실질적인 성능 개선을 보여주었습니다. 이 연구의 결과는 공정한 성별 분류 연구를 위한 귀중한 자원으로, 앞으로의 연구에 있어 유용하게 활용될 수 있을 것입니다.



### A Survey of Recursive and Recurrent Neural Networks (https://arxiv.org/abs/2510.17867)
Comments:
          96 pages,48 figures

- **What's New**: 이 논문에서는 재귀 신경망(Recursive Neural Networks, RecursiveNNs)과 순환 신경망(Recurrent Neural Networks, RecurrentNNs)의 다양한 구조와 목적 함수, 학습 알고리즘을 세분화하여 분류하고 있습니다. 이들은 크게 세 가지 유형으로 나뉘며, 첫 번째는 일반적인 재귀 및 순환 신경망, 두 번째는 구조화된 재귀 및 순환 신경망, 세 번째는 기타 재귀 및 순환 신경망이 포함됩니다. 이러한 분류를 통해 다양한 신경망의 복잡한 관계를 이해하고, 다양한 문제 해결에 이 기법들이 어떻게 활용될 수 있는지를 설명합니다.

- **Technical Details**: 재귀 신경망은 구조화된 입력을 처리하며, 특히 문법 분석, 감정 분석 및 질문 응답 시스템에 적용됩니다. 반면, 순환 신경망은 변동 길이의 입력 시퀀스를 처리할 수 있으며, 반복적 숨겨진 변수를 통해 입력 시퀀스를 학습하고 모델링합니다. RNNs는 고유한 반복 기능을 사용하여 정보를 저장하며, 이는 시간 의존성 학습에 유리합니다. 하지만, 대규모 시간 의존성을 모델링하는 데 어려움이 있어 다양한 구조적 변형들이 등장했습니다.

- **Performance Highlights**: 일반적인 RNNs는 손글씨 인식, 음성 인식, 자연어 처리 및 컴퓨터 비전과 같은 다양한 문제에 효과적으로 적용되고 있습니다. LSTM(Long Short Term Memory) 구조는 이러한 RNNs 변형 중에서 가장 성숙한 구조로, 음성 및 이미지 문제에 폭넓게 사용됩니다. 또한, 차원 교차 허용 및 양방향 처리 방식 등 다양한 신경망 변형들은 정보 처리의 효율을 높여 문맥 정보를 잘 캡처할 수 있도록 합니다.



### MUSE: Model-based Uncertainty-aware Similarity Estimation for zero-shot 2D Object Detection and Segmentation (https://arxiv.org/abs/2510.17866)
Comments:
          11 pages with 6 figures

- **What's New**: 이번 연구에서는 MUSE (Model-based Uncertainty-aware Similarity Estimation)라는 새로운 프레임워크를 소개합니다. MUSE는 훈련이 필요 없는 모델 기반의 제로샷(Zero-shot) 2D 객체 탐지 및 분할을 위해 설계되었습니다. 이 프레임워크는 3D 보지 못한(uncaptured) 객체에서 렌더링된 2D 다중 뷰 템플릿과 입력 쿼리 이미지에서 추출한 2D 객체 제안을 활용합니다.

- **Technical Details**: MUSE의 임베딩 단계에서는 클래스(Class)와 패치(Patch) 임베딩을 통합합니다. 패치 임베딩은 일반화된 평균 풀링(GeM)을 사용하여 정규화되어 전역(Global) 및 지역(Local) 표현을 효율적으로 캡처합니다. 일치(matching) 단계에서는 절대 및 상대 유사성 점수를 결합하는 공동 유사성 지표를 활용하여 어려운 상황에서도 일치의 견고성을 향상시킵니다.

- **Performance Highlights**: MUSE는 추가적인 훈련 또는 미세 조정 없이도 BOP Challenge 2025에서 최첨단(state-of-the-art) 성능을 달성하며, Classic Core, H3, Industrial 트랙에서 1위를 기록했습니다. 이러한 결과는 MUSE가 제로샷 2D 객체 탐지 및 분할을 위한 강력하고 일반화 가능한 프레임워크를 제공함을 보여줍니다.



### Deploying Atmospheric and Oceanic AI Models on Chinese Hardware and Framework: Migration Strategies, Performance Optimization and Analysis (https://arxiv.org/abs/2510.17852)
- **What's New**: 본 논문은 중국산 칩과 프레임워크를 활용하여 PyTorch에서 MindSpore로 대규모 대기 및 해양 모델을 변환하고 최적화하는 프레임워크를 제안합니다. 이를 통해 GPU 의존성을 줄이고, 시스템 효율성을 개선하며, 모델 정확도를 유지하는 방법을 논의합니다. 또한, 이 연구는 기후 모델링을 위한 적합성을 평가하기 위해 다양한 성능 지표를 비교합니다.

- **Technical Details**: 프레임워크는 모델 및 하드웨어 적응, 메모리 최적화, 병렬 처리에 중점을 두고 있습니다. 모델 이식 과정에서 PyTorch에 기반한 모델 구조 조정, 연산자 적응 및 대체를 통해 MindSpore와의 호환성을 확보합니다. MindSpore의 정적 그래프 및 특징을 활용하여 혼합 정밀도 훈련, 분산 계산 지원 등을 통해 효율성을 극대화합니다.

- **Performance Highlights**: 실험 결과는 중국 칩이 GPU에 비해 훈련 및 추론 속도, 모델 정확도 및 에너지 효율성에서 유의미한 향상을 보임을 보여줍니다. 이를 통해 대규모 모델을 중국 플랫폼에서 성공적으로 배포할 수 있는 가능성을 확인했습니다. 이 연구는 대기 및 해양 AI 모델 개발에 있어 보다 높은 기술적 독립성을 위한 실질적인 안내를 제공합니다.



### Pre to Post-Treatment Glioblastoma MRI Prediction using a Latent Diffusion Mod (https://arxiv.org/abs/2510.17851)
Comments:
          10 pages, 4 figures. Presented to the Deep Generative Models Workshop of MICCAI (DGM4MICCAI)

- **What's New**: 이번 연구는 적극적인 뇌종양인 교모세포종(GBM)의 조기 치료 반응 예측을 위해 새로운 Latent Diffusion Model(LDM)을 제안합니다. 이 모델은 치료 전(MRI) 이미지를 바탕으로 치료 후(MRI) 영상을 생성하는 것을 목표로 하며, 환자 생존 정보를 활용하여 이미지 생성 품질을 향상시킵니다. 이로써 임상에서의 개인 맞춤 치료에 기여할 수 있는 가능성을 제시합니다.

- **Technical Details**: 제안된 모델은 2D-LDM으로 구성되며, Encoding과 Decoding의 과정을 통해 낮은 차원의 잠재 공간으로 이미지를 압축하고 복원합니다. 이를 위해 Vector Quantized-Variational AutoEncoder(VQ-VAE)를 사용하여 이미지의 잠재 표현을 학습하였고, Gross Tumor Volume(GTV) 정보와 결합하여 효과적인 예측을 도모합니다. Diffusion Model(확산 모델)의 역 확산 과정은 UNet를 통해 학습되며, 이는 생체의학 영상 합성에서 최신 성능을 보여줍니다.

- **Performance Highlights**: 실험 결과, 제안된 LDM은 교모세포종 환자에서 치료 전 이미지를 바탕으로 치료 후 이미지를 성공적으로 생성할 수 있음을 입증하였습니다. 연구는 140명의 GBM 환자 데이터를 통해 진행되었으며, 양질의 이미지를 생성하는 능력이 입증되었습니다. 이는 종양의 치료 반응을 조기에 예측하여 임상 의사 결정에 도움을 줄 수 있는 도구가 될 것입니다.



### CARLE: A Hybrid Deep-Shallow Learning Framework for Robust and Explainable RUL Estimation of Rolling Element Bearings (https://arxiv.org/abs/2510.17846)
Comments:
          26 pages, accepted at Soft Computing

- **What's New**: 이번 논문에서는 새로운 Prognostic Health Management (PHM) 시스템을 제안합니다. 이 시스템은 Rolling Element Bearing의 Remaining Useful Life (RUL) 추정을 위한 CARLE(hybrid AI framework)를 포함하고 있습니다. CARLE는 깊은 학습(deep learning)과 얕은 학습(shallow learning)을 결합하여 효율적으로 장비의 건강 상태를 모니터링하고 예측할 수 있도록 설계되었습니다.

- **Technical Details**: CARLE는 Res-CNN과 Res-LSTM 블록을 활용하여 공간적 및 시간적 열화 패턴을 포착합니다. 다중 헤드 주의 메커니즘(multi-head attention)과 Residual connection을 적용하여 정보 손실을 최소화하고 일관성을 높였습니다. 또한, Gaussian filtering과 Continuous Wavelet Transform (CWT)을 통해 시간-주파수(feature extraction) 데이터를 효과적으로 처리합니다.

- **Performance Highlights**: CARLE는 XJTU-SY 및 PRONOSTIA 베어링 데이터 셋을 통해 성능이 검증되었습니다. 여러 최신 RUL 추정 방법들과 비교했을 때, CARLE는 동적 조건에서도 우수한 정확도를 보였습니다. 마지막으로, LIME과 SHAP와 같은 XAI(Explainable AI) 기법을 사용하여 모델의 투명성과 신뢰성을 분석함으로써, CARLE의 안전성과 신뢰성을 평가하였습니다.



### MAT-Agent: Adaptive Multi-Agent Training Optimization (https://arxiv.org/abs/2510.17845)
Comments:
          Acceptance to NeurIPS 2025 Main Track

- **What's New**: 본 논문에서는 동적 환경에 적합한 훈련 전략을 목표로 하는 다중 에이전트 프레임워크인 MAT-Agent를 제안합니다. MAT-Agent는 훈련 과정을 협력적이며 실시간 최적화 프로세스로 재구성하며, 자동 에이전트를 통해 데이터 증강(data augmentation), 최적화기(optimizer), 학습률(learning rate), 손실 함수(loss function)를 동적으로 조정합니다. 이로 인해 정확도, 희귀 클래스 성능 및 훈련 안정성을 종합적으로 고려한 보상(composite reward)을 통해 모델의 성능을 극대화할 수 있습니다.

- **Technical Details**: MAT-Agent는 훈련의 각 단계에서 동적으로 훈련 구성(𝐂t)을 조합하며, 이는 데이터 증강, 최적화기 선택, 학습률 스케줄ing, 손실 함수 설계를 조정하는 4개의 자율적이고 적응형 에이전트로 구성되어 있습니다. 각 에이전트는 훈련 상태(sts_{t})에 대한 글로벌 신호를 인지하며, 각 훈련 구성 요소에 대한 행동을 선택합니다. 이들 에이전트는 현재 구성의 효과성을 정량화하는 보상 신호를 받고, 이를 통해 정책을 지속적으로 업데이트하여 전체적으로 작용하는 동적 훈련 프로세스를 실현합니다.

- **Performance Highlights**: MAT-Agent는 Pascal VOC, COCO 및 VG-256 데이터셋에서 실험을 통해 뛰어난 성능을 입증하였습니다. Pascal VOC에서 mAP는 97.4로 SOTA 성능인 PAT-T의 96.2에 비해 우수하며, OF1은 92.3, CF1은 91.4를 기록하였습니다. COCO 데이터셋에서는 mAP 92.8, OF1 88.2, CF1 87.1을 달성하고, VG-256에서도 각각 60.9, 70.8, 61.1 성과를 보여줍니다. 전반적으로 MAT-Agent는 빠른 수렴과 강력한 도메인 전이 일반화를 제공하여 복잡한 비주얼 모델 최적화에 혁신적인 솔루션을 제시합니다.



### Modeling Layered Consciousness with Multi-Agent Large Language Models (https://arxiv.org/abs/2510.17844)
Comments:
          20 pages, 4 figures, accepted for presentation at EMNLP 2025 Workshop on Active and Passive LLM Personalization (PALS) OpenReview: this https URL

- **What's New**: 이번 연구에서는 대규모 언어 모델(LLM)의 인공지능 의식을 모형화하기 위한 다중 에이전트 프레임워크를 제안합니다. 이는 정신분석 이론(psychoanalytic theory)에 기반하고 있으며, 자기 인식(self-awareness), 전의식(preconsciousness), 무의식(unconsciousness)을 에이전트 상호작용을 통해 시뮬레이션합니다.

- **Technical Details**: 우리의 모델인 Psychodynamic Model은 고정된 특성과 동적인 필요를 결합한 개인화 모듈(Personalization Module)을 통해 가이드됩니다. 감정이 풍부한 대화체에 대한 매개변수 효율적 파인튜닝(parameter-efficient fine-tuning)을 사용하여 시스템을 평가하였고, 총 8가지 개인 맞춤형 조건에서 실험하였습니다.

- **Performance Highlights**: 모델 평가 결과, LLM을 판별자로 사용했을 때, 파인튜닝된 모델이 71.2%의 선호도를 보여주었습니다. 이 모델은 감정적 깊이가 향상되고 출력의 변동성이 줄어들어 개인화된 인지(cognition)를 위한 적응 가능성을 입증하였습니다.



### GRETEL: A Goal-driven Retrieval and Execution-based Trial Framework for LLM Tool Selection Enhancing (https://arxiv.org/abs/2510.17843)
Comments:
          5 pages, 1 figures, 5 tables

- **What's New**: 이번 연구에서 소개된 GRETEL은 Big Language Model (LLMs)의 도구 검색에서 발생하는 의미론적-기능적 격차(semantic-functional gap)를 해결하는 혁신적인 프레임워크입니다. 현재 도구 검색 방법은 주로 텍스트 유사성을 기반으로 하지만, 이러한 방식은 기능적으로 작동하지 않는 도구를 검색하는 문제점을 초래합니다. GRETEL은 에이전트 기반 워크플로우를 통해 실제 실행-평가 사이클을 수행하여 진정으로 기능적인 도구를 구별하는 능력을 가지고 있습니다.

- **Technical Details**: GRETEL의 작동 방식은 에이전트 기반의 접근 방식을 채택하여 단순히 의미론적으로 정렬된 후보 리스트를 수용하지 않고, 이를 테스트 가능한 가설로 다룹니다. 연구에서 사용된 LangGraph 라이브러리를 통해 각 도구에 대한 계획-실행-평가 사이클을 체계적으로 수행합니다. 이러한 접근은 각 도구의 API 사양과 사용자 쿼리를 바탕으로 유효한 API 호출을 생성하고, 그 결과를 통해 기능적 유용성을 평가합니다.

- **Performance Highlights**: ToolBench 벤치마크에서의 포괄적인 평가 결과, GRETEL은 모든 메트릭에서 유의미한 개선을 이루었습니다. Pass Rate는 0.690에서 0.826으로, Recall은 0.841에서 0.867로, NDCG는 0.807에서 0.857로 증가했습니다. 이러한 결과는 실행 기반 검증이 단순 의미론적 유사성보다 도구 선택에 더 신뢰할 수 있는 기초를 제공함을 입증합니다.



### Brain-Language Model Alignment: Insights into the Platonic Hypothesis and Intermediate-Layer Advantag (https://arxiv.org/abs/2510.17833)
- **What's New**: 이 연구는 2023년에서 2025년 사이에 발표된 25개의 fMRI 기반 연구를 리뷰하며, 뇌와 언어 모델이 같은 내부 세계 표현으로 수렴하는지를 조사합니다. 두 가지 주요 가설인 플라톤적 표현 가설(Platonic Representation Hypothesis)과 중간 층 장점(Intermediate-Layer Advantage)을 검토하고, 모델과 뇌가 추상적 표현 구조를 공유할 가능성을 제시합니다.

- **Technical Details**: 연구는 기능적 자기공명영상(fMRI) 기술을 사용하여 언어 모델과 인간 뇌의 활동 간의 유사성을 분석합니다. 뇌 활동과 모델 표현 간의 유사성을 평가하기 위한 다양한 방법론이 존재하며, 모델의 구조나 크기, 데이터셋의 크기 등이 뇌-모델 정렬에 미치는 영향이 주의 깊게 рассмотр됩니다. 또한, 중간 층들이 더 일반화된 특징을 인코딩하는 경우가 많다는 점이 강조됩니다.

- **Performance Highlights**: 연구 결과는 중간 층에서 뇌의 표현과 더 높은 유사성을 보이는 경향이 있음을 제시하였고, 이는 두 가지 가설을 모두 지지하는 증거로 작용합니다. 뇌와 언어 모델 간의 정렬이 진화하는 과정과 각 연구의 공통 패턴을 드러내며, 더 많은 연구가 필요함을 제안합니다. 이러한 발견은 AI와 신경과학의 교차점에서 더 깊은 이해를 요청하고 있습니다.



### Synthetic EEG Generation using Diffusion Models for Motor Imagery Tasks (https://arxiv.org/abs/2510.17832)
Comments:
          15 pages, BRACIS

- **What's New**: 이 연구에서는 Motor Imagery(운동 상상) 작업과 관련된 합성 EEG 신호 생성을 위한 새로운 방법론을 제안합니다. 기존의 EEG 데이터 수집 문제를 해결하기 위해 Denoising Diffusion Probabilistic Models(DDPM)을 활용하여 신호를 생성합니다. 이를 통해 다양한 변량을 줄이고, EEG 기반의 Brain-Computer Interfaces(BCI) 성능을 향상시킬 수 있는 방법을 모색합니다.

- **Technical Details**: 연구에 사용된 데이터셋은 BCI Competition III의 Dataset V이며, 총 3명의 건강한 피험자로부터 수집되었습니다. EEG 데이터를 512Hz의 샘플링 속도로 처리하며, 특정 EEG 채널에서 발생하는 잡음을 제거하기 위해 Independent Component Analysis(ICA)를 적용했습니다. DDPM을 채택하여 원래 신호에서 Gaussian 잡음을 단계적으로 추가한 후, 1D U-Net 아키텍처를 통해 원래 신호를 복원하는 과정을 수행했습니다.

- **Performance Highlights**: 본 연구에서 생성된 합성 데이터는 실제 데이터에 비해 95% 이상의 분류 정확도를 기록했으며, 평균 제곱 오차(mean squared error)는 낮고 실제 신호와의 상관관계는 높습니다. K-Nearest Neighbors (KNN), Convolutional Neural Networks (CNN), U-Net 등의 분류기를 사용하여 합성 데이터와 실 데이터의 성능을 비교하였고, 합성 신호가 데이터 부족 문제를 해결하고 EEG 기반 BCI의 분류 성능을 개선하는데 기여함을 입증했습니다.



### Multi-Agent Design Assistant for the Simulation of Inertial Fusion Energy (https://arxiv.org/abs/2510.17830)
- **What's New**: 본 논문에서는 인공지능(AI) 모델을 물리학 코드와 결합하여 자기유도 핵융합 연료 캡슐 설계를 자동화하는 방법을 제안합니다. 이는 복잡한 물리적 조건을 탐색하는 과정에서 자연어를 사용하는 다중 에이전트 시스템을 구성하여 이루어집니다. 이러한 자동화 시스템은 고차원 다중 물리학 계산 코드를 실행할 수 있으며, 캡슐의 기하학을 최적화하여 시뮬레이션 점화를 달성합니다.

- **Technical Details**: MADA(Multi-Agent Design Assistant)는 역설계 에이전트, 작업 관리 에이전트, 시뮬레이션 에이전트 및 물리학 에뮬레이션 대리인('Professor')으로 구성됩니다. 이러한 에이전트는 서로 간의 대화를 통해 원활하게 정보를 전달하며, 자연어 프롬프트를 사용하여 공동으로 설계를 최적화합니다. 시스템은 매개변수 탐색을 통해 디자인 반복 루프 내에서 물리적 행동을 학습하며, 다양한 시뮬레이션을 실행합니다.

- **Performance Highlights**: 본 연구는 MADA가 고충실도의 물리학 시뮬레이션을 자율적으로 실행하며, 설계 최적화와 불확실성 정량화에 효과적임을 입증합니다. 이 시스템은 에이전트 간의 상호작용을 통해 캡슐 디자인을 향상시키는 피드백 루프를 생성하여 고도화된 설계나 강력한 솔루션으로 발전합니다. MADA는 향후 자기유도 핵융합 발전소의 설계 및 운영에 필요한 AI 기반 제어 시스템의 발전을 이끄는 첫걸음으로 자리 잡을 것으로 기대됩니다.



### Speak to a Protein: An Interactive Multimodal Co-Scientist for Protein Analysis (https://arxiv.org/abs/2510.17826)
- **What's New**: 새로운 시스템인 Speak to a Protein은 단백질 분석 과정을 상호작용이 가능한 대화형 AI와 함께 진행할 수 있도록 설계되었습니다. 이 시스템은 관련 문헌, 구조 및 리간드 데이터를 수집하고 통합하여 사용자에게 실시간으로 정보를 제공합니다. 또한, 코드 생성 및 실행을 지원하며, 결과에 대한 설명을 텍스트와 그래픽으로 제공합니다.

- **Technical Details**: Speak to a Protein의 시스템 아키텍처는 사용자 상호작용 및 시각화를 위한 프론트엔드와 언어 이해 및 데이터 검색을 위한 백엔드로 구성됩니다. 프론트엔드는 대화형 챗 패널과 3D 분자 시각화 기능을 포함하고 있으며, 백엔드는 자연어 쿼리를 처리하고 다양한 도구를 조정하는 역할을 합니다. 시스템은 Python 샌드박스를 사용하여 실시간으로 코드 실행과 데이터 조작이 가능하게 합니다.

- **Performance Highlights**: Speak to a Protein은 사용자가 단백질에 대한 질문을 하였을 때, 즉각적으로 3D 구조를 조작하고 주석을 달 수 있는 능력을 제공합니다. 이로 인해 복잡한 생화학 데이터 분석의 장벽이 대폭 낮아지며, 가설 생성을 위한 더 유연하고 강력한 방법을 가능하게 합니다. 해당 시스템은 질문에서 증거로의 전환 시간을 감소시켜 연구자들의 발견 속도를 가속화합니다.



### Carbon-Aware Orchestration of Integrated Satellite Aerial Terrestrial Networks via Digital Twin (https://arxiv.org/abs/2510.17825)
- **What's New**: 본 연구는 Integrated Satellite Aerial Terrestrial Networks (ISATNs)에 대해 탄소 인식형 오케스트레이션 프레임워크를 제안합니다. 이 프레임워크는 Digital Twin (DT) 기술을 활용하여, CO$_2$-등가그램(gCO$_2$/bit)을 주요 지속 가능성 지표로 삼고, 다중 시간 규모의 Plan Do Check Act (PDCA) 루프를 구현합니다. 이를 통해 이 연구는 기존의 에너지 인식 연구를 발전시키고, ISATN의 결과적으로 기후 변화에 긍정적인 영향을 줄 수 있는 방법을 제시합니다.

- **Technical Details**: 제안된 아키텍처는 탄소 인식형 오케스트레이션을 위해 여러 핵심 구성 요소로 나뉘고, PDCA 제어 루프를 통해 지속적인 네트워크 모니터링 및 실시간 최적화를 수행합니다. DT는 실제 네트워크의 고충실도 복사본을 지속적으로 업데이트하여 주요 요소인 트래픽 패턴, 이동성, 그리드의 탄소 강도 데이터를 예측합니다. 특별히 이 프레임워크 내에서 gCO2e/bit가 지속 가능성 평가의 중심 지표로 설정되어 오케스트레이션 알고리즘이 환경적 영향을 최소화할 수 있도록 지원합니다.

- **Performance Highlights**: 시뮬레이션 결과는 QoS 전적으로 오케스트레이션에 비해 최대 29% 낮은 gCO$_2$/bit 수치를 달성했습니다. 또한, 재생 가능 에너지 활용도를 높이고 불리한 사건에서도 회복력을 향상시키는 데 기여했습니다. 여기서 ISATNs의 특정 제어 메커니즘을 통해 고품질의 서비스를 유지하면서 탄소 배출을 효과적으로 줄이는 방법을 보여줍니다.



### A Biophysical-Model-Informed Source Separation Framework For EMG Decomposition (https://arxiv.org/abs/2510.17822)
- **What's New**: 본 연구에서는 기존의 motor unit decomposition의 한계를 극복하고자 Biophysical-Model-Informed Source Separation (BMISS) 프레임워크를 제안합니다. 이 프레임워크는 해부학적으로 정확한 forward EMG 모델을 통합하여 motor neuron의 속성을 비침습적으로 추정할 수 있도록 합니다. MRI 기반의 해부학적 재구성을 활용하여 motor units의 액티비티를 고해상도로 추출할 수 있는 가능성을 제시합니다.

- **Technical Details**: BMISS 프레임워크는 기존의 blind source separation (BSS) 방법의 한계를 보완하며, 노이즈의 영향을 줄이고 motor unit(action potentials)의 신뢰도를 높입니다. 이 접근법은 anatomy 정보와 비례하여 정확도를 개선하며, 단순하지만 효과적인 모델 비틀림 알고리즘을 통해 감독 없이 작동합니다. 이를 통해 motor neuron의 전기적 특성을 sEMG 신호로부터 직접 추정할 수 있는 새로운 방법론을 소개합니다.

- **Performance Highlights**: 실험 결과, BMISS는 전통적인 방법에 비해 motor unit 추정에서 더 높은 충실도를 달성하였으며, 계산 비용 또한 현저히 감소했습니다. 이 프레임워크는 비침습적인 개인화된 신경근 평가에 대한 가능성을 열어 주며, 임상 진단, 보철 제어, 신경 재활과 같은 다양한 응용 분야에서 잠재력을 보여줍니다.



### LLM Assisted Alpha Fairness for 6 GHz WiFi and NR_U Coexistence: An Agentic Orchestrator for Throughput, Energy, and SLA (https://arxiv.org/abs/2510.17814)
- **What's New**: 본 논문은 비허가 6GHz 대역에서 원활한 공존을 위해 Wi-Fi와 5G NR-U가 경쟁하는 환경에서의 스케줄링 문제를 해결하기 위한 에이전트 컨트롤러를 제안합니다. 이 컨트롤러는 정책(policy)과 실행(execution)을 분리하여 고차원적 의사결정 과정을 간소화합니다. 특히, 대형 언어 모델(LLM)을 활용하여 네트워크의 상태를 해석하고, 사용자가 이해할 수 있는 정책 조정knobs을 제안함으로써 애드혹한 접근 방식을 제공합니다.

- **Technical Details**: 제안된 시스템은 두 개의 160MHz 채널을 갖춘 6GHz 시뮬레이터에서 운영되며, 각 사용자에 대한 채널 품질 지표(CQI), 배터리 상태, 대기열, 지연 시간, 우선 순위 및 전력 모드 등의 변수를 고려합니다. 정책은 각 채널의 공정성을 나타내는 지수 lpha과 Wi-Fi 및 NR-U의 채널별 듀티 사이클 한도를 포함하여 작은 해석 가능한 조절 가능성을 선택합니다. LLM은 데이터 기반으로 해당 조건에서 강력한 규칙 기반 스케줄링과 비교하여 이러한 조정 가능성이 에너지 효율성을 현저히 개선하는 데 도움을 줍니다.

- **Performance Highlights**: 하나의 LLM 변형은 총 에너지를 35.3% 절감하면서 적당한 처리량 손실을 보였고, 또 다른 LLM은 전체 비트 수가 +3.5% 증가하고 bits/J가 +12.2%로 향상된 결과를 보여줍니다. 결과적으로 LLM 지원 정책은 에너지 효율을 개선하면서도 스케줄링의 성능을 유지하여 경쟁력 있는 처리량을 자랑했습니다. 본 논문은 코드와 로그, 그래프 유틸리티를 공개하여 모든 수치를 재현할 수 있도록 하였으며, 정책 레벨의 LLM 가이드를 통한 무선 공존 개선 방안을 실증적으로 제시합니다.



### Visual Space Optimization for Zero-shot Learning (https://arxiv.org/abs/1907.00330)
- **What's New**: 이번 논문은 제로샷 학습(zero-shot learning) 분야에서 시각적 공간(visual space)을 최적화하는 두 가지 전략을 제안합니다. 구체적으로, 각 시각적 클래스에 대해 시각적 프로토타입(visual prototype)을 학습하고, 중간 임베딩 공간(intermediate embedding space)에서의 시각적 특징 구조를 최적화하는 방법을 다룹니다. 이러한 접근 방식은 새롭게 등장한 카테고리 인식 문제를 해결하기 위한 혁신적인 방법으로 주목받고 있습니다.

- **Technical Details**: 첫 번째 방법인 시각적 프로토타입 기반 방법은 각 클래스가 일련의 분산된 시각적 특징 대신에 프로토타입 특징으로 표현되도록 학습합니다. 두 번째 접근 방식은 다층 퍼셉트론(multi-layer perceptron) 프레임워크를 사용하여 공통의 중간 임베딩 공간을 학습하고 시각적 데이터 구조를 보다 독창적으로 만드는 것입니다. 이러한 두 가지 방법은 각각의 클래스 간의 시각적 상관관계를 향상시키기 위해 설계되었습니다.

- **Performance Highlights**: 네 개의 벤치마크 데이터셋을 통한 광범위한 실험 평가를 통해, 시각적 공간을 최적화하는 것이 제로샷 학습에 이점이 있음을 입증했습니다. 또한, 제안된 프로토타입 기반 방법은 새로운 최첨단(performance highlights) 성능을 달성하였습니다. 이는 제로샷 학습의 가능성을 한층 더 높이는 중요한 기여로 여겨집니다.



### SceneCOT: Eliciting Grounded Chain-of-Thought Reasoning in 3D Scenes (https://arxiv.org/abs/2510.16714)
Comments:
          Project page: this https URL

- **What's New**: 이 논문은 3D Large Language Models (LLMs)의 한계를 극복하기 위한 새로운 프레임워크인 SceneCOT을 제시합니다. SceneCOT은 Grounded Chain-of-Thought (CoT) 추론 방식을 도입하여 복잡한 추론 과제를 더 단순하고 관리 가능한 문제로 분해합니다. 이를 위해 185,000개의 고품질 사례로 구성된 최초의 대규모 grounded CoT 추론 데이터 세트인 SceneCOT-185K를 개발하였습니다.

- **Technical Details**: SceneCOT은 3D 장면에서 복잡한 추론 작업을 네 단계로 분해합니다: (1) 작업 인식 및 분석, (2) 작업 관련 영역 로컬라이제이션, (3) 다중 모달 전문가 모듈을 이용한 개체 및 속성 그라운딩, (4) 중간 결과를 통합하여 일관된 최종 답변을 생성하는 grounded 추론이 포함됩니다. 이러한 계층적 작업 흐름은 각 답변이 명시적 그라운딩 단계를 통해 지원되도록 합니다.

- **Performance Highlights**: MSQA 및 Beacon3D 벤치마크에서의 extensive 실험 결과, SceneCOT은 높은 grounding-QA 일관성을 달성하며 강력한 성능을 보여 주었습니다. 이 연구는 인간과 유사한 단계적 추론을 가능하게 하여, 더 넓은 3D 장면 이해 시나리오로의 확장 가능성을 제시합니다.




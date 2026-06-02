New uploads on arXiv(cs.CL)

### SEAP: Training-free Sparse Expert Activation Pruning Unlock the Brainpower of Large Language Models (https://arxiv.org/abs/2503.07605)
Comments:
          15 pages, 7 figures, 8 tables

- **What's New**: 이 논문에서는 Sparse Expert Activation Pruning (SEAP)이라는 새로운 pruning 방법을 제안합니다. SEAP는 특정 작업에 관련된 파라미터를 선택적으로 유지하여 inference 오버헤드를 줄이는 훈련이 필요 없는 메소드를 제공합니다. 이는 인공지능 모델의 성능을 유지하면서도 계산 효율성을 높일 수 있는 가능성을 제시합니다.

- **Technical Details**: SEAP는 작업 특정 활성화 패턴을 기반으로 동적으로 pruning하여 계산 오버헤드를 줄입니다. 이를 위해 다양한 작업에서 수집된 데이터셋을 사용하여 hidden state를 추출하고, 각 뉴런의 중요성을 평가하여 pruning 결정을 내립니다. 이 방법은 task-adaptive pruning 전략을 채택하여 특정 작업의 복잡도에 따라 pruning 비율을 조절합니다.

- **Performance Highlights**: 실험 결과 SEAP는 기존 방법들과 비교하여 뛰어난 성능을 보였습니다. 50% pruning 후에도 WandA와 FLAP보다 20% 이상 높은 성능을 기록하며, 20% pruning에서는 밀집 모델에 비해 단 2.2%의 성능 저하에 그쳤습니다. 이러한 결과는 SEAP가 대규모 LLM 최적화에 있어 매우 유망한 접근 방식임을 증명합니다.



### Implicit Reasoning in Transformers is Reasoning through Shortcuts (https://arxiv.org/abs/2503.07604)
- **What's New**: 이번 연구에서는 GPT-2 모델을 처음부터 훈련시키고 multi-step (다단계) 수학적 추론 데이터셋을 사용하여 언어 모델이 어떻게 implicit reasoning (암묵적 추론)을 수행하는지를 조사했습니다. 특히, implicit reasoning이 어떻게 언어 모델의 높은 정확도를 가져오는지, 그리고 이는 고정 패턴 데이터에 대한 훈련에서만 나타난다는 점에 주목했습니다.

- **Technical Details**: 연구 결과에 따르면, 언어 모델은 단계별 추론을 수행할 수 있으며, 이는 고정 패턴 데이터에 기반하여 in-domain (도메인 내) 및 out-of-domain (도메인 외) 테스트 모두에서 높은 정확도를 보였습니다. 반면, 고정되지 않은 패턴 데이터로부터 얻어진 implicit reasoning 능력은 특정 패턴에 과적합(overfitting)되며 더 이상 일반화되지 못한다는 한계를 가지고 있습니다.

- **Performance Highlights**: 이 연구는 고급 추론 능력이 implicit reasoning 스타일에서 왜 나타나지 않는지를 설명합니다. 핵심 발견 중 하나는, 언어 모델이 유사한 패턴의 작업에서 강력한 성능을 발휘하지만, 이러한 능력이 shortcut learning (우회 학습)을 통해 획득되었다는 점입니다. 따라서, 이러한 모델들은 특정 패턴에 대해 잘 작동하지만, 일반화 능력이 결여되어 있습니다.



### Detection Avoidance Techniques for Large Language Models (https://arxiv.org/abs/2503.07595)
- **What's New**: 본 논문에서는 대형 언어 모델의 인기가 높아짐에 따라 가짜 뉴스의 체계적 전파와 같은 다양한 리스크가 발생하고 있음을 강조하고 있습니다. DetectGPT와 같은 분류 시스템의 개발이 필수적이며, 이 시스템들이 회피 기술에 취약하다는 점도 실험을 통해 확인되었습니다. 이러한 결과는 대형 언어 모델에 대한 신뢰성을 더욱 요구하게 만듭니다.

- **Technical Details**: 연구에서는 생성 모델의 온도(temperature) 조정을 통해 얕은 학습(detectors)이 가장 신뢰할 수 없음을 증명하였습니다. 또한, 강화 학습(reinforcement learning)을 통한 생성 모델의 미세 조정(fine-tuning)이 BERT 기반(detectors) 회피를 가능하게 했습니다. 최종적으로, 문장을 재구성하여 DetectGPT와 같은 제로샷(detectors) 회피율이 90%를 초과하는 성과를 보였습니다.

- **Performance Highlights**: 제시된 방법은 기존 연구와의 비교를 통해 더 나은 성능을 발휘함을 강조합니다. 가짜 뉴스를 식별하는 데 있어 더욱 강력한 의사 결정을 가능하게 하며, 사회적 함의와 향후 연구 방향에 대한 논의도 포함됩니다. 이 연구는 언어 모델의 신뢰성 문제 해결을 위한 중요한 기초 자료를 제공합니다.



### KSOD: Knowledge Supplement for LLMs On Demand (https://arxiv.org/abs/2503.07550)
- **What's New**: 이 논문에서는 KSOD(Knowledge Supplement for LLMs On Demand)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 LLMs의 지식 기반으로 슈퍼바이즈드 파인 튜닝(Supervised Fine-Tuning, SFT)을 통해 LLM의 성능을 개선합니다. KSOD는 LLM의 오류 원인을 지식 부족 관점에서 분석하고, 부족한 지식을 바탕으로 LLM을 보완하여 성능을 향상시키는 데 중점을 둡니다.

- **Technical Details**: KSOD는 세 가지 주요 단계를 포함합니다: Knowledge Identification(지식 식별), Knowledge Verification(지식 검증), Knowledge Supplement(지식 보충). 이 과정에서 KSOD는 기존 데이터셋에서 지식을 추출하고, 이를 통해 LLM의 오류를 유발할 수 있는 지식을 확인한 후, 해당 지식을 LLM에 보충하여 성능을 개선합니다. 알고리즘을 통해 LLM이 특정 지식을 결여하고 있는지를 검증하는 방법론을 제시하고 있습니다.

- **Performance Highlights**: KSOD를 통한 실험 결과는 LLM의 오류를 억제하고 필요한 지식을 보충함으로써 4개의 일반 작업에서 성능 변화를 최소화하면서도 개선을 보여주었습니다. 특히, 특정 지식을 보완했을 때 LLM의 다른 작업에 대한 성능 저하가 거의 없거나 소폭 변화가 있었으며, 일부 경우에는 성능이 개선되기도 했습니다. 이러한 연구 결과는 지식을 보충하여 LLM의 성능을 향상시킬 수 있는 잠재력을 보여줍니다.



### XIFBench: Evaluating Large Language Models on Multilingual Instruction Following (https://arxiv.org/abs/2503.07539)
- **What's New**: 본 논문에서는 XIFBench라는 포괄적인 벤치마크를 소개하고, 이를 통해 다중 언어 설정에서 대형 언어 모델(LLMs)의 지시 수행 능력을 평가합니다. XIFBench는 다섯 가지 제약 조건 카테고리를 기반으로 한 분류법과 465개의 병렬 지시문을 포함하고 있으며, 다양한 자원 수준의 여섯 개 언어에서 평가를 수행합니다. 이 연구는 LLM의 다언어 지시 수행 능력에 대한 새로운 통찰을 제공합니다, 특히 중간 및 낮은 자원 언어에서의 성능 변화를 분석합니다.

- **Technical Details**: XIFBench는 Content, Style, Situation, Format, Numerical의 다섯 가지 주요 카테고리로 구성된 제약 조건 분류 체계를 바탕으로 하여 대형 언어 모델의 지시 수행을 평가합니다. 각 카테고리는 구체적인 차원과 기준을 포함하여, LLM이 응답 내에서 어떻게 과학적 연구를 포함하도록 유도하는 등의 세부 지침을 제공합니다. 또한, 영어의 요구사항을 사용해 복잡한 지시를 평가 요구사항으로 분해하여, 다른 언어 간의 평가의 일관성을 보장하는 요구 기반 프로토콜을 개발하였습니다.

- **Performance Highlights**: 다양한 LLM을 통해 실시된 광범위한 실험은 자원 수준에 따라 지시 수행 성능에 상당한 변동이 있음을 보여주었습니다. 성능에 영향을 미치는 주요 요인으로는 제약 조건 카테고리, 지시 복잡성, 문화적 특수성이 포함됩니다. 이 연구는 LLM의 다국어 지시 수행 능력에 대한 보다 세밀한 분석과 평가 방법론의 필요성을 강조합니다.



### LMM-R1: Empowering 3B LMMs with Strong Reasoning Abilities Through Two-Stage Rule-Based RL (https://arxiv.org/abs/2503.07536)
- **What's New**: 이 논문에서는 대규모 다중 모달 모델(Large Multimodal Models, LMMs)에서 추론(Reasoning)을 향상시키기 위한 새로운 접근 방식인 	extbf{Foundational Reasoning Enhancement (FRE)}와 	extbf{Multimodal Generalization Training (MGT)}를 제안합니다. 기존의 텍스트 전용 분야에서는 강력한 성과를 보여주었지만, 다중 모달 환경에서는 한계가 있었습니다. 이 연구는 이러한 한계를 극복하기 위해 텍스트 데이터로부터 추론 능력을 강화한 후, 이를 다중 모달 환경에 일반화하는 이중 단계 프레임워크를 도입합니다.

- **Technical Details**: 제안된 	extbf{	extit{method}}는 먼저 FRE 단계에서 규칙 기반 강화 학습(Rule-based Reinforcement Learning, RL)을 통해 텍스트 전용 데이터의 추론 능력을 향상시킵니다. 그 다음, MGT 단계에서 이러한 향상된 능력을 다중 모달 도메인으로 일반화하여 적용합니다. 이 방식을 통해 데이터의 부족과 복잡한 추론 예시의 부족 문제를 해결하고, 다중 모달 예비 학습으로 인한 기초적 추론 능력의 저하를 완화합니다.

- **Performance Highlights**: 실험 결과 	extbf{	extit{method}}는 Qwen2.5-VL-Instruct-3B 모델에서 다중 모달 벤치마크에서 4.83% 및 텍스트 전용 벤치마크에서 4.5%의 평균 개선률을 보였으며, 복잡한 축구 게임(Football Game) 과제에서는 3.63%의 성과 향상을 기록했습니다. 이러한 결과는 텍스트 기반의 추론 강화가 효과적인 다중 모달 일반화를 가능하게 함을 입증하며, 고품질의 다중 모달 훈련 데이터에 대한 필요성을 줄이는 데이터 효율적인 패러다임을 제공합니다.



### TokenButler: Token Importance is Predictab (https://arxiv.org/abs/2503.07518)
- **What's New**: 이번 논문에서는 대형 언어 모델(Large Language Models, LLMs)의 Key-Value (KV) 캐시의 메모리 병목을 해결하기 위해 "TokenButler"라는 새로운 토큰 중요도 예측기를 도입하였습니다. 기존 모델들은 고정된 규칙에 따라 토큰을 배제하거나 전체 KV 캐시를 유지관리하였으나, TokenButler는 쿼리에 의존한 동적 예측을 통해 중요한 토큰을 선택합니다. 이로 인해 기존 방법보다 8% 이상 향상된 정확도를 보여줍니다.

- **Technical Details**: TokenButler는 경량 예측기로, 1.2% 이하의 파라미터 오버헤드를 가지며 중요한 토큰의 문맥적 중요도를 기반으로 우선순위를 매깁니다. 이는 정밀한 토큰 중요도 추정을 가능하게 하여, 모델의 perplexity와 하위 정확도를 개선합니다. 기존의 KV 캐시 문제를 해결하기 위해 작은 컨텍스트(<512 토큰)에서 수행된 실험을 통해, 자기참조와 관련된 정밀도가 높음을 입증합니다.

- **Performance Highlights**: TokenButler는 기존의 토큰 희소성 메트릭을 초과하여, 8% 이상의 향상을 가져오는 경량화된 예측기를 통해 중요한 토큰을 식별합니다. 이러한 성능은 TokenButler가 새로운 합성 작은 컨텍스트 자기참조 검색 작업에서 거의 오라클에 가까운 정확도를 달성했다는 점에서 더욱 두드러집니다. 코드 및 모델은 논문에서 제공되는 새 URL을 통해 확인할 수 있습니다.



### Language Models Fail to Introspect About Their Knowledge of Languag (https://arxiv.org/abs/2503.07513)
- **What's New**: 최근 대규모 언어 모델(LLMs)의 내부 상태를 반추할 수 있는 능력에 대한 관심이 증가하고 있습니다. 이러한 능력은 LLM의 해석 가능성을 높이고, 표준적인 반추적 방법을 통해 모델의 문법적 지식을 평가하는 데 유용할 수 있습니다. 이 연구는 21개의 오픈소스 LLM을 체계적으로 조사하며, 문법 지식과 단어 예측이라는 두 가지 분야에 초점을 맞춥니다. 특히, 모델의 내부 언어 지식은 문자열 확률의 직접 측정을 기반으로 이론적으로 구체화됩니다.

- **Technical Details**: 우리는 두 가지 방법을 사용하여 각 모델을 평가합니다. Direct 방법은 문자열에 할당된 로그 확률을 비교하여 모델의 '진정한' 지식 상태를 이론적으로 나타내고, Meta 방법은 반추적 프롬프트에 대한 반응에 할당된 로그 확률을 비교합니다. 이 두 가지 방법의 일치를 통해, 모델이 반추할 수 있다면 메탈링귀스틱 반응이 자신의 내부 확률에 보다 충실할 것으로 예상합니다. 연구 결과는 반추의 여부를 평가하는 데 있어 현재의 연구 방법론의 한계를 극복하려는 시도를 다룹니다.

- **Performance Highlights**: 실험 결과, 테스트된 LLM에서 반추에 대한 강력한 증거를 발견하지 않았습니다. 오히려 서로 유사한 모델은 메탈링귀스틱과 직접 측정된 행동 간의 강한 상관관계를 보였습니다. 즉, 모델의 메탈링귀스틱 반응이 문자열 확률에 대한 지식과 별개로 존재한다는 결론에 도달했습니다. 이러한 발견은 최근의 모델 반추에 대한 주장에 복잡성을 더하며, 모델의 언어 능력을 연구하는 데 중요한 시사점을 제공합니다.



### MedAgentsBench: Benchmarking Thinking Models and Agent Frameworks for Complex Medical Reasoning (https://arxiv.org/abs/2503.07459)
- **What's New**: MedAgentsBench는 기존 의학 질의응답 벤치마크에 비해 다단계 클리닉 추론, 진단 수립 및 치료 계획 수립을 필요로 하는 도전적인 의학 질문에 초점을 맞춘 새로운 벤치마크입니다. 이 벤치마크는 기존 평가의 세 가지 주요 한계를 해결하며, 다양한 기존 의학 데이터셋에서 수집된 데이터를 기반으로 합니다. 또한, 최신 모델 DeepSeek R1과 OpenAI o3가 복잡한 의학적 추론 작업에서 뛰어난 성능을 발휘하는 것이 입증되었습니다.

- **Technical Details**: MedAgentsBench는 8개의 잘-known 의료 데이터셋에서 수집된 데이터를 기반으로 설정된 기준으로, 복잡한 추론 시나리오에 초점을 맞춘 질문들을 포함합니다. 이 벤치마크는 질문에 대해 모델의 정답 비율을 분석하여, 50% 이하의 성공률을 보이는 질문을 '어려운 질문'으로 분류하여 현재 모델들이 어려움을 겪고 있는 질문을 선정합니다. 이를 통해 세밀한 성능 평가의 기준을 제공합니다.

- **Performance Highlights**: 실험 결과, DeepSeek R1 및 OpenAI o3와 같은 고급 모델들이 기존 방법론에 비해 15-25% 높은 정확도를 보였습니다. 또한 검색 기반 에이전트 방식은 전통적인 접근법보다 더 우수한 비용-성능 비율을 보이며, 오픈 소스 모델이 실용적인 운영 비용으로 경쟁력 있는 결과를 달성할 수 있는 가능성을 보여줍니다. 이는 복잡한 의학적 질문에 대한 모델 간의 성능 격차를 드러내고 최적 모델 선택을 위한 인사이트를 제공합니다.



### LLMs syntactically adapt their language use to their conversational partner (https://arxiv.org/abs/2503.07457)
Comments:
          4 pages, 1 table, 1 figure, submitted to ACL

- **What's New**: 본 논문에서는 인간이 대화 중에 언어 사용을 어떻게 조정하는지를 바탕으로, 대형 언어 모델(LLMs)도 유사한 행동을 보이는지를 실증적으로 연구합니다. LLM 간의 대화 corpus를 구축하고, 대화가 진행됨에 따라 이러한 모델들이 어떻게 문법적 선택을 더욱 유사하게 만드는지를 분석했습니다. 결과적으로, 현대 LLM이 대화 상대에 따라 언어 사용을 기본적으로 적응시키는 경향이 있음을 확인했습니다.

- **Technical Details**: LLM의 문법적 적응을 측정하기 위해, 인간 대화의 경우를 분석한 Reitter와 Moore(2014)의 방법론을 적용했습니다. 대화의 첫 49%를 Prime, 마지막 49%를 Target으로 나누어 문법 규칙의 사용 빈도를 분석하고, 두 부분 사이의 규칙 반복 가능성을 비교했습니다. 이러한 방법론을 통해, 대화 세트에서 문법적 규칙의 상호작용을 추적하고, LLM 모델이 사용자의 언어에 얼마나 효과적으로 적응하는지를 평가했습니다.

- **Performance Highlights**: 분석 결과, LLM 대화의 문법적 반복이 대화 내에서 주는 변화를 정량적으로 평가할 수 있었고, LLM 간의 적응 경향이 통계적으로 유의미하다는 것을 발견했습니다. 이 발견은 인간 대화에서 문법적 정렬 현상을 재현한 것으로, LLM도 대화 중 언어 사용을 조정하는 능력이 있음을 시사합니다. 또한, 대화 내내 이러한 적응이 지속적인 과정임을 보여주며, LLM에서 ‘인간과 유사한’ 정렬이 나타날 수 있음을 논의합니다.



### Revisiting Noise in Natural Language Processing for Computational Social Scienc (https://arxiv.org/abs/2503.07395)
Comments:
          PhD thesis. Under the supervision of Prof. Isabelle Augenstein

- **What's New**: 이 논문은 Computational Social Science (CSS) 분야에서의 잡음(noise)의 다양한 양상과 그로 인한 도전 과제를 탐구하고 있습니다. 특히, 과거 데이터 처리에서 발생하는 문자 수준의 오류, 시대에 뒤떨어진 언어 사용, 그리고 주관적이며 모호한 작업에서의 주석 불일치 등을 중심으로 다룹니다. 저자는 잡음이 연구에 있어 무익하거나 해로운 것이라는 기존의 관념에 도전하며, 오히려 일부 형태의 잡음이 CSS 연구에 중요한 정보를 제공할 수 있음을 주장합니다.

- **Technical Details**: CSS 연구에서의 잡음은 Optical Character Recognition (OCR) 과정 중의 문자 오류에서부터 애매모호한 언어 사용에 이르기까지 다양한 형태로 나타납니다. 종래의 관점은 잡음을 제거하거나 완화해야 할 불완전한 요소로 간주했지만, 이 논문은 잡음이 독특한 의사소통 양식이나 문화의 영향을 받은 데이터셋을 암호화하고 있다는 점에 주목합니다. 저자는 잡음의 처리 방법이 해당 유형과 맥락에 따라 달라져야 한다고 강조합니다.

- **Performance Highlights**: 저자의 연구는 CSS에서의 잡음을 다루는 체계적인 연구가 부족한 상황에서 매우 독창적인 접근법을 보여줍니다. 다양한 사례 연구를 통해 저자는 잡음의 유형별로 효과적인 처리 전략을 제시하고 있으며, 이는 연구자들에게 새로운 시각을 제공합니다. 이 논문은 CSS의 발전에 기여할 뿐만 아니라, 머신러닝 및 NLP(Natural Language Processing) 기술의 발전과도 밀접하게 연결되어 있습니다.



### Is My Text in Your AI Model? Gradient-based Membership Inference Test applied to LLMs (https://arxiv.org/abs/2503.07384)
- **What's New**: 이 연구는 LLMs에 대한 텍스트 분류 작업에서 gradient-based Membership Inference Test (gMINT)를 적용하고 연구합니다. MINT는 특정 데이터가 머신러닝 모델 훈련에 사용되었는지 확인하는 일반적인 접근 방식이며, 본 연구는 자연어 처리 분야에서의 활용에 초점을 맞추고 있습니다. 특히, 데이터 개인 정보 보호에 대한 우려를 해결하기 위해 gMINT가 데이터 샘플의 훈련 포함 여부를 확인하는 능력을 강조합니다.

- **Technical Details**: gMINT는 각 훈련 샘플이 모델의 훈련 및 최적화에 미치는 영향을 분석하기 위해 훈련 과정에서 생성된 gradient를 활용합니다. 연구에서는 250만 문장을 포함한 6개의 데이터셋과 7개의 Transformer 기반 모델을 사용하여 gMINT를 평가했습니다. 이 과정에서 gMINT의 강력함이 입증되어 데이터 크기와 모델 구조에 따라 85%에서 99% 사이의 AUC 점수를 기록했습니다.

- **Performance Highlights**: gMINT는 훈련 과정의 어떤 샘플이 데이터에 포함되었는지를 높은 정확도로 식별하는 효과적인 능력을 보여주었습니다. 실험 결과는 gMINT가 머신러닝 모델의 감사를 위한 확장 가능하고 신뢰할 수 있는 도구로서의 잠재력을 강조합니다. 이를 통해 AI/NLP 기술의 배포에서 투명성을 보장하고 민감한 데이터를 보호하며 윤리적 준수를 촉진할 수 있습니다.



### RepoST: Scalable Repository-Level Coding Environment Construction with Sandbox Testing (https://arxiv.org/abs/2503.07358)
- **What's New**: RepoST(Repository-Level Sandbox Testing)는 코드 생성을 위한 환경을 구축하여 피드백을 제공하는 확장 가능한 방법을 제시합니다. 기존의 전체 저장소를 구축하는 접근법과 달리, RepoST는 특정 함수와 그 종속성을 분리된 스크립트에 배치하여 샌드박스 테스트를 통해 피드백을 제공합니다. 이를 통해 외부 종속성의 복잡성을 줄이고 대규모로 환경을 구축할 수 있게 되었습니다.

- **Technical Details**: RepoST는 GitHub 저장소에서 함수를 추출하고, 이를 샌드박스하여 지역 종속성을 분리된 스크립트에서 실행합니다. 이 과정은 Rust-like 호출 그래프를 활용하여 수행되며, LLM(대규모 언어 모델)을 이용해 테스트 케이스를 생성하여 환경의 실행 가능성을 높입니다. 또한, 각 함수의 기능이 유지되는지를 검증하기 위해 여러 품질 통제 단계를 수행합니다.

- **Performance Highlights**: RepoST-Train은 현재 8억개 이상의 함수를 포함하는 가장 큰 저장소 기반 코드 생성 데이터셋으로, Qwen2.5-Coder는 HumanEval의 Pass@1에서 5.5% 향상된 성능을 기록했습니다. RepoST-Eval은 296개 함수 샘플을 포함하며 12개의 코드 생성 모델을 평가하여, 가장 우수한 모델도 개선 여지가 크다고 평가되었습니다.



### Assessing the Macro and Micro Effects of Random Seeds on Fine-Tuning Large Language Models (https://arxiv.org/abs/2503.07329)
Comments:
          7 pages, 5 tables, 3 figures

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)에서 랜덤 시드의 영향을 체계적으로 평가합니다. GLUE 및 SuperGLUE 벤치마크를 사용하여 정확도, F1 점수와 같은 전통적인 지표의 평균과 분산을 계산하여 성능 변동의 매크로 레벨 영향을 분석합니다. 또한, 예측의 일관성을 측정하는 새로운 지표를 도입하여 개별 예측의 안정성에 대한 마이크로 레벨 효과를 포착합니다.

- **Technical Details**: 랜덤 시드의 매크로 레벨 영향을 평가하기 위해 여러 시드에 대한 성능 지표의 분산을 계산합니다. 이러한 성능 지표에는 F1 점수 또는 회귀 작업을 위한 Pearson 상관 등이 포함될 수 있으며, VAR 값이 작을수록 매크로 수준의 성능 변화가 적다는 것을 의미합니다. 또한, '일관성(consistency)'을 정의하여 다양한 하이퍼파라미터 설정으로 미세 조정된 LLM 간의 예측 일관성을 측정합니다.

- **Performance Highlights**: 실험 결과, 표준 메트릭과 일관성 메트릭 모두에서 상당한 변동성이 있음을 보여주었습니다. 이는 LLM의 미세 조정 및 평가에서 랜덤 시드 기반 변동을 고려해야 할 필요성을 강조합니다. 연구의 주요 기여는 이러한 변동성을 제대로 이해하고, 보다 신뢰할 수 있으며 재현 가능한 결과를 위한 평가 기준을 설계하는 데 도움이 되는 것입니다.



### Benchmarking Chinese Medical LLMs: A Medbench-based Analysis of Performance Gaps and Hierarchical Optimization Strategies (https://arxiv.org/abs/2503.07306)
- **What's New**: 이번 연구는 의료 분야에서 대규모 언어 모델(LLMs)의 평가와 개선이 얼마나 중요한지를 다룹니다. 주목할 점은 기존의 프레임워크가 도메인 별 오류 패턴을 충분히 분석하지 못하고, 다중 모드(multi-modal) 과제를 해결하지 못한다는 것입니다. 이 연구는 10개의 모델을 분석하여 오류를 8가지 유형으로 구분하는 세밀한 오류 분류 체계를 소개합니다.

- **Technical Details**: 의료 모델의 평가에서 10개의 모델을 조사하였으며, 정확도(accuracy)가 0.86인 반면, 중요한 추론(task)에서는 96.3%의 누락이 발생했습니다. 또한, 안전 윤리 평가에서는 변형된 옵션 하에서 0.79의 강건성 점수를 보여 주목할 만한 불일치가 나타났습니다. 이러한 발견은 지식 경계 enforcement와 다단계(reasoning) 분석의 체계적인 약점을 드러냈습니다.

- **Performance Highlights**: 이 연구는 4단계로 나누어진 최적화 전략을 제안하며, 이는 프롬프트 엔지니어링(prompt engineering)과 지식 보강 검색(knowledge-augmented retrieval)부터 하이브리드 신경 기호 구조(hybrid neuro-symbolic architectures), 인과 추론(causal reasoning) 프레임워크에 이릅니다. 제안된 로드맵은 임상적으로 강건한 LLM을 개발하기 위한 실행 가능한 방향을 제시하며, 오류 기반 인사이트(error-driven insights)를 통해 평가 패러다임을 재정립하고 의료 환경에서 AI의 안전성과 신뢰성을 높이는데 기여합니다.



### An Information-Theoretic Approach to Identifying Formulaic Clusters in Textual Data (https://arxiv.org/abs/2503.07303)
- **What's New**: 이번 연구에서는 고차원 텍스트 공간에서 비지도 방식으로 공식적인(formulaic) 패턴을 식별하기 위한 정보 이론적 알고리즘을 개발했다. 이 알고리즘은 가중치가 부여된 자기 정보 분포(weighted self-information distributions)를 활용하여 텍스트 내의 구조적 패턴을 탐지한다. 이는 전통적인 공분산 기반 방법론이 소규모 샘플에서 불안정해지는 문제를 해결하는 데 기여한다.

- **Technical Details**: 연구에서 사용된 정보 이론 측정 도구 중 하나는 엔트로피(entropy)이다. 이는 데이터 세트의 예측 불가능성 또는 무작위성을 측정하며, 공식적인 클러스터를 비공식적인 것과 구별하는 강력한 도구이다. Shannon의 엔트로피 공식을 기반으로 하여, 이 연구는 고전적인 자기 정보 측정을 연속적으로 확장하였다.

- **Performance Highlights**: 히브리어 성경에서의 저자 구분에 이 방법을 적용한 결과, 성공적으로 스타일적 계층을 분리했으며, 이는 텍스트의 층화(compositional patterns 분석을 위한 정량적 프레임워크를 제공한다. 이러한 접근 방식은 복잡한 저작 및 편집 과정에 의해 형성된 텍스트의 문학적 및 문화적 진화를 보다 깊이 이해하는 데 도움을 준다.



### A Graph-based Verification Framework for Fact-Checking (https://arxiv.org/abs/2503.07282)
Comments:
          13pages, 4figures

- **What's New**: 이번 연구에서는 그래프 기반 사실 검증 프레임워크(GraphFC)를 제안하여 기존의 사실 검증 방식의 한계를 극복하고자 하였습니다. 기존의 언어 모델(LLM)을 활용한 주장 분해 방식은 불충분한 분해와 언급의 애매성을 주된 문제로 나타내었으며, 이러한 문제를 해결하기 위해 주장을 <주체, 관계, 객체>로 이루어진 그래프로 변환하였습니다. 이를 통해 경량화된 분해 및 언급의 명확화를 동시에 달성할 수 있음을 보여주었습니다.

- **Technical Details**: GraphFC 프레임워크는 세 가지 주요 구성 요소로 이루어져 있습니다: (1) 그래프 구축, 여기서는 주장 그래프 및 해당 주장에 대한 증거 그래프를 함께 구성합니다. (2) 그래프 기반 계획, 이 단계에서는 각 삼중항(triplet)의 검증 순서를 우선순위에 따라 결정합니다. (3) 그래프 기반 검증, 여기서는 주장을 바탕으로 한 그래프 일치와 미완의 삼중항을 위한 그래프 보완을 수행합니다. 이러한 구조를 통해 사실 검증의 맥락 정보를 유지하면서도 애매성을 줄일 수 있습니다.

- **Performance Highlights**: 다양한 사실 검증 데이터셋(HOVER, FEVEROUS, SciFact)에서의 실험을 통해 GraphFC의 효과성을 입증하였으며, 기존의 방법들과 비교하여 최첨단 성능을 달성했습니다. 이 프레임워크는 미세한 분해를 통해 문맥의 혼란을 해소하고, 정확한 분류를 가능하게 해 사실 검증의 신뢰성을 크게 향상시킵니다. 결과적으로 GraphFC는 미래의 사실 검증 연구에 중요한 기여를 하고 있습니다.



### SemEval-2025 Task 11: Bridging the Gap in Text-Based Emotion Detection (https://arxiv.org/abs/2503.07269)
Comments:
          SemEval2025 Task11 (Task Description Paper). arXiv admin note: text overlap with arXiv:2502.11926

- **What's New**: 이번 논문에서는 7개 언어 가족에서 더해 30개 이상의 언어에 대한 텍스트 기반 감정 탐지(shared task)를 소개하고 있습니다. 각 언어는 주로 저자원(low-resource) 언어로서 여러 대륙에서 사용됩니다. 참여자들은 단일 언어 설정에서 감정 레이블 예측, 감정 강도 점수 예측, 그리고 다국어 설정에서의 감정 레이블 예측 등 세 가지 경로로 작업을 수행했습니다.

- **Technical Details**: 참여자들에게 제공된 데이터셋은 다양한 출처에서 수집된 10만 개 이상의 다중 라벨 인스턴스로, 여섯 가지 감정 클래스(joy, sadness, anger, fear, surprise, disgust)로 주석 처리되었습니다. 각 팀은 한 가지, 두 가지 또는 세 가지 경로의 결과를 제출할 수 있었고, 공식 평가 메트릭으로는 Tracks A와 C에 대한 평균 F-score와 Track B에 있는 인간 판단과의 일치를 평가하는 Pearson 상관 계수를 사용했습니다.

- **Performance Highlights**: 총 700명 이상의 참가자가 있었고, 220개의 최종 제출과 93개 팀에서 시스템 설명 논문이 제출되었습니다. Track A(다중 라벨 감정 탐지)는 가장 많은 제출(114)을 기록했으며, 일반적으로 각 팀은 평균 11개 언어로 참여했습니다. 이 작업은 2024년 CodaBench에서 가장 인기 있는 대회로 꼽혔습니다.



### LLM-C3MOD: A Human-LLM Collaborative System for Cross-Cultural Hate Speech Moderation (https://arxiv.org/abs/2503.07237)
Comments:
          Accepted to NAACL 2025 Workshop - C3NLP (Workshop on Cross-Cultural Considerations in NLP)

- **What's New**: 이 논문은 콘텐츠 조정(content moderation)의 새로운 접근 방식을 제안합니다. 특히, 많은 기술 플랫폼이 자원이 풍부한 언어에 집중하는 반면, 자원이 부족한 언어의 모더레이터는 부족하다는 문제를 다룹니다. 효과적인 조정은 문화적 맥락을 이해하는 데 의존하기 때문에, 이 불균형은 비원어민 모더레이터의 문화 이해 부족으로 인해 부적절한 조정의 위험을 증가시킵니다.

- **Technical Details**: 연구팀은 비원어민 모더레이터가 문화특화 지식(culturally-specific knowledge), 감정(sentiment), 인터넷 문화(internet culture)를 해석하는 데 어려움을 겪는다는 것을 발견했습니다. 이를 해결하기 위해 LLM-C3MOD라는 인간-LLM 협력 파이프라인을 제시하며, 세 단계로 구성됩니다: (1) RAG-enhanced cultural context annotations; (2) 초기 LLM 기반 조정; (3) LLM 합의가 없는 경우의 인간 조정(targeted human moderation).

- **Performance Highlights**: 이 시스템은 한국의 혐오 발언(hate speech) 데이터셋을 이용하여 평가되었으며, 인도네시아와 독일 참가자들이 참여했습니다. 결과적으로 78%의 정확도(기준인 GPT-4o의 71%를 초과)를 달성했으며, 인간의 작업량은 83.6% 줄어들었습니다. 특히, 인간 모더레이터는 LLM이 힘들어하는 미세한 내용(nuanced contents)에서 뛰어난 성과를 보였습니다.



### Cross-Lingual IPA Contrastive Learning for Zero-Shot NER (https://arxiv.org/abs/2503.07214)
Comments:
          17 pages, 6 figures

- **What's New**: 이 논문에서는 저자들이 CONtrastive Learning with IPA (CONLIPA) 데이터세트를 제안하여 10개 고자원 언어와 영어의 IPA 쌍을 포함하고 있습니다. 이는 유사한 발음을 가진 언어간의 포네믹 표현 과정에서 갭을 줄이고, 고자원 언어로 훈련된 모델이 저자원 언어에서 효과적으로 작동할 수 있도록 지원하는 방법론을 다룹니다. 또한, 본 연구는 Zero-Shot NER을 위한 새로운 접근 방식으로, 대규모 언어 모델(LLMs)을 활용하여 동족어 쌍을 추출해 훈련합니다.

- **Technical Details**: CONLIPA 데이터세트는 10개의 주요 언어 가족에서 추출된 고자원 언어와 영어 간의 IPA 쌍으로 구성되어 있습니다. 이 데이터세트를 사용하여 Cross-lingual IPA Contrastive learning 방법(IPAC)을 제안하며, 이는 다양한 언어의 포네믹 표현 간의 유사성을 확보하는 데 중점을 둡니다. 논문에서는 자가 감독(self-supervised) 학습 방식을 적용하고, InfoNCE 손실을 통해 유사한 데이터 포인트를 찾아내는 방법을 사용합니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식은 베이스라인과 비교하여 유의미한 성능 향상을 보여주었고, 특히 저자원 언어에서 Zero-Shot NER 작업에 있어 주요 이점을 입증했습니다. 분석된 결과는 새로운 모델이 유사한 발음을 가진 단어들 간의 표현을 효과적으로 가까이 가져오는 능력을 갖추었음을 나타냅니다. 이렇게 한의 발음 기록 방식을 활용하여 향후 저자원 언어 연구에 기여할 수 있는 가능성이 높아 보입니다.



### Contextual Cues in Machine Translation: Investigating the Potential of Multi-Source Input Strategies in LLMs and NMT Systems (https://arxiv.org/abs/2503.07195)
Comments:
          11 pages

- **What's New**: 기존의 다원 번역 시스템과 LLM 모델(GPT-4o)을 비교하여 다소원 입력이 기계 번역(MT) 품질에 미치는 영향을 탐구합니다. 본 연구는 중간 언어 번역을 활용한 컨텍스트 정보를 통해 영어 및 중국어의 포르투갈어 번역 품질을 향상시키는 방법을 평가하고, 언어 쌍 간의 전략적 언어 선택의 중요성을 강조합니다. 특히 다원 입력이 도메인 특정 데이터셋에서 번역 품질에 긍정적인 영향을 미치는지에 대한 연구를 수행합니다.

- **Technical Details**: 이 연구는 3,000개의 테스트 문장을 포함하는 5개의 데이터셋을 사용하여, 다수의 컨텍스트 언어의 영향을 평가합니다. 연구에는 OpenAI의 GPT-4o 모델과 NVIDIA NeMo 툴킷을 이용한 전통적인 다언어 NMT 시스템이 포함되었습니다. 또한, 모델의 출력을 최적화하기 위해 shallow fusion 기법을 적용하여 여러 소스 언어의 로그 확률을 결합하는 방식으로 진행되었습니다.

- **Performance Highlights**: 실험 결과, 중간 언어를 사용할 때 번역 품질이 유의미하게 향상됨을 보여주었고, 특히 도메인 특정 데이터셋에서 보다 뚜렷한 개선이 관찰되었습니다. 그러나 높은 언어 변이가 있는 벤치마크에서는 수확이 감소하는 경향이 나타났습니다. GPT-4o와 전통적인 NMT 시스템의 비교 평가를 통해 각 모델의 강점과 한계를 분석하며, 컨텍스트 정보를 활용한 적응력에 대한 통찰을 제공하였습니다.



### Strategies for political-statement segmentation and labelling in unstructured tex (https://arxiv.org/abs/2503.07179)
Comments:
          Accepted to NLP4DH 2025 @ NAACL 2025

- **What's New**: 본 연구는 정치적 맥락에서의 발언 분할 및 분류를 위한 새로운 접근 방식을 제안하고 있습니다. 현재 주로 제공된 발언 경계에 의존하여 제한된 현장 적용 가능성이 있는 기존 방법을 개선하기 위해, 다양한 통합 분할 및 레이블링 프레임워크를 실험하였습니다. 제안된 방법은 raw 텍스트에서 발언을 동시에 세그멘테이션(Segmentation)하고 분류(Classification)하는 것을 목표로 하고 있습니다.

- **Technical Details**: 연구는 linear-chain CRFs, fine-tuned text-to-text models, 그리고 in-context learning과 constrained decoding의 조합을 사용하여 발언을 세그멘테이션하고 레이블을 부착하는 여러 모델을 시험하였습니다. 모델은 Transformer 아키텍처에 근거하여 구성되며, XLM-RoBERTa (XLM-R) 인코더를 이용한 token-wise emission scores로 CRF와 결합되어 있습니다. 입력 텍스트는 여러 개의 겹치는 윈도우로 나누어 프로세스 되어, 각 토큰의 표현이 적절한 문맥을 바탕으로 생성되도록 하였습니다.

- **Performance Highlights**: 제안된 방법은 비교적 높은 정확도를 나타내며, 원시 텍스트의 정치적 선언문에 적용했을 때 높은 성능을 보였습니다. 특히 fine-tuned T5 모델은 최고의 성능을 달성했지만, 높은 컴퓨팅 요구와 느린 추론 속도로 인해 대규모 실험에 실용성이 떨어지는 것으로 나타났습니다. 반면, CRF 기반 모델은 성능이 약간 저하되지만 빠른 추론 시간을 제공하여 대규모 적용에 더 적합합니다.



### DeFine: A Decomposed and Fine-Grained Annotated Dataset for Long-form Article Generation (https://arxiv.org/abs/2503.07170)
- **What's New**: 이번 연구에서는 DeFine이라는 새로운 데이터셋을 소개합니다. DeFine은 Long-Form Article Generation (LFAG)을 지원하기 위해 계층적으로 분해되고 세밀하게 주석이 달린 데이터셋입니다. 이 데이터셋은 각각의 생성 프로세스를 세 가지 단계를 구분하여 일정한 논리적 일관성과 내용 구성을 보장합니다. 본 논문은 DeFine이 가진 독특한 구조적 특성과 구체적인 주석을 통해 긴 형식의 기사 작성을 향상시킬 수 있음을 밝혔다.

- **Technical Details**: DeFine 데이터셋은 데이터 채굴(Data Miner), 인용 검색(Cite Retriever), 질문-답변 주석 생성(Q&A Annotator) 및 데이터 정리(Data Cleaner)로 구성된 다중 에이전트 협업 파이프라인을 이용해 구축되었습니다. 각 에이전트는 데이터셋 생성의 특정 측면에 전문화되어 있으며, 이는 논리적 일관성과 내용 조직을 보장하는데 기여합니다. 데이터 마이너는 Wikipedia에서 구조적 요소를 추출하고, 인용 검색기는 참조된 URL을 정제하며, Q&A 주석 생성기는 맥락 인식 프롬프트를 사용합니다. 마지막으로 데이터 클리너는 데이터의 무결성을 확보합니다.

- **Performance Highlights**: 실험 결과, DeFine로 학습된 Qwen2-7b-Instruct 모델은 기존의 LFAG 방법과 비교하여 논리적 일관성, 사실 정확성 및 인용 신뢰성에서 유의미한 개선을 보였습니다. 이를 통해 DeFine이 긴 형식의 기사 생성에서 질적 향상을 가능하게 하는 효과적인 도구임을 입증하였습니다. DeFine 데이터셋은 연구자들에게 공개되어 향후 연구를 촉진할 수 있도록 도와줍니다.



### MRCEval: A Comprehensive, Challenging and Accessible Machine Reading Comprehension Benchmark (https://arxiv.org/abs/2503.07144)
Comments:
          Under review

- **What's New**: 이 논문에서는 기계 독해 이해(MRC) 평가를 위한 포괄적이고 도전적인 벤치마크인 MRCEval을 소개합니다. 기존 MRC 데이터셋은 주로 특정 독해 이해(RC) 능력을 평가했지만, MRCEval은 LLMs(Large Language Models)의 13가지 독해 기술을 포괄하여 총 2,103개의 고품질 다중 선택 질문을 포함하고 있습니다. 이는 LLMs의 RC 능력을 효과적으로 검토하며, 그들이 직면하는 도전 과제를 강조합니다.

- **Technical Details**: MRCEval은 세 가지 주요 작업과 13개의 하위 작업으로 구성되어 있으며, 각 작업은 MRC의 세 가지 핵심 측면인 맥락 이해(Context Comprehension), 외부 지식 이해(External Knowledge Comprehension), 그리고 추론(Reasoning)에 초점을 맞추고 있습니다. 이 벤치마크는 GPT-4o를 활용하여 샘플을 생성하고, 세 가지 경량 모델이 샘플의 질을 판별하는 평가자로 사용됩니다.

- **Performance Highlights**: 28개의 유명한 오픈 소스 및 폐쇄형 모델을 대상으로 한 평가 결과, 가장 경쟁력 있는 모델인 o1-mini 및 Gemini-2.0-flash조차도 MRCEval에서 통계적으로 낮은 성과를 보였습니다. 이는 LLMs의 성능이 기존 기준에서는 높게 평가되지만, MRC 문제에서는 여전히 상당한 도전 과제가 존재함을 보여줍니다.



### A Systematic Comparison of Syntactic Representations of Dependency Parsing (https://arxiv.org/abs/2503.07142)
- **What's New**: 본 논문에서는 다양한 주석 방식(annotation schemes)에 대한 전이 기반 파서(transition-based parser)의 성능을 비교합니다. 또한, 보편적 의존 트리뱅크(universal dependency treebanks)에서 관찰된 몇몇 특정 구문 구성을 더 표준화된 표현으로 변환할 것을 제안합니다.

- **Technical Details**: 연구는 프로젝트의 모든 언어에 대한 파싱 성능을 평가합니다. 우리는 "표준" 구성이 항상 더 나은 파싱 성능으로 이어지지 않음을 보여주며, 점수가 언어에 따라 상당히 변동함을 강조합니다.

- **Performance Highlights**: 이 논문은 언어 간 파서 성능의 차이를 통찰하고, 표준화된 구성이 모든 언어에서의 성능 개선을 보장하지 않는다는 점을 강조하고 있습니다. 이를 통해 앞으로의 연구 방향에 대한 중요한 시사점을 제공합니다.



### Application of Multiple Chain-of-Thought in Contrastive Reasoning for Implicit Sentiment Analysis (https://arxiv.org/abs/2503.07140)
- **What's New**: 이번 연구에서는 암묵적 감정 분석(implicit sentiment analysis)의 성능을 향상시키기 위한 새로운 방법론인 Dual Reverse Chain Reasoning (DRCR) 프레임워크를 제안합니다. 이 프레임워크는 감정의 양극성을 가정하고, 이를 반증한 뒤, 두 가지 추론 경로를 대조하여 최종 감정 양극성을 도출하는 세 가지 주요 단계를 포함합니다. 또한, 무작위 가설의 한계를 해결하기 위해 Triple Reverse Chain Reasoning (TRCR) 프레임워크도 도입했습니다.

- **Technical Details**: DRCR 프레임워크는 연역적 추론(deductive reasoning)에 영감을 받아 구성되며, 감정 가정(hypothesis)과 그에 대한 추론 과정을 제시합니다. 두 가지 경로를 대조하는 과정에서, 대조 메커니즘(contrastive mechanisms)과 다단계 추론(multi-step reasoning)을 결합하여 감정 분류의 정확도를 높입니다. TRCR는 DRCR의 단점을 보완하고 더 정밀한 분석을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, 제안된 두 가지 방법은 다양한 모델 스케일(model scales)에서 기존 방법들보다 우수한 성능을 보이며 최첨단 수준의 정확도를 달성했습니다. 이러한 결과는 암묵적 감정 분석에 있어 대조적 추론과 다단계 추론을 결합하는 것이 효과적임을 입증합니다. 연구의 결과는 앞으로의 암묵적 감정 분석에 큰 기여를 할 것으로 기대됩니다.



### ASTRA: A Negotiation Agent with Adaptive and Strategic Reasoning through Action in Dynamic Offer Optimization (https://arxiv.org/abs/2503.07129)
- **What's New**: 이번 논문에서는 자율적인 협상 에이전트를 개발하기 위해 ASTRA라는 새로운 프레임워크를 도입했습니다. ASTRA는 상대 모델링(opponent modeling)과 Tit-for-Tat(TFT) 상호 작용과 같은 두 가지 핵심 원칙을 기반으로 한 턴별 제안 최적화(turn-level offer optimization)를 특징으로 합니다. 이 에이전트는 시뮬레이션과 인간 평가를 통해 상대방의 행동 변화에 효과적으로 적응하며, 협상 성과를 향상시킵니다.

- **Technical Details**: ASTRA 에이전트는 세 가지 단계로 운영됩니다: (1) 상대방의 행동 해석, (2) 선형 프로그래밍(Linear Programming, LP) 해결기를 통한 반대 제안 최적화, (3) 협상 전술 및 수용 가능성에 기반한 제안 선택입니다. 이 에이전트는 상대의 선호도를 추론하고 이를 기반으로 제안을 조정하는 핵심 요소들을 통합하여 복잡한 협상에서도 유용하게 작동할 수 있도록 설계되었습니다. 이러한 접근 방식은 협상의 복잡성을 이해하고 전략적으로 대응할 수 있는 능력을 높입니다.

- **Performance Highlights**: ASTRA는 다양한 에이전트 유형 및 인간 평가를 통해 상대방의 변하는 태도에 맞춰 성공적으로 협상 목표를 달성했습니다. 이 에이전트는 전략적 피드백을 제공하고 최적의 제안 추천을 통해 협상 코칭 도구로도 활용될 수 있습니다. 연구 결과는 ASTRA가 협상에서의 적응력 및 전략적 추론을 크게 개선했다는 것을 보여줍니다.



### A Novel Ophthalmic Benchmark for Evaluating Multimodal Large Language Models with Fundus Photographs and OCT Images (https://arxiv.org/abs/2503.07094)
- **What's New**: 최근 대규모 언어 모델(LLMs)은 다양한 의료 응용 분야에서 뛰어난 잠재력을 보여주었습니다. 이번 연구는 LLM을 비주얼 모델과 통합한 다중 모드 대규모 언어 모델(MLLMs)을 통해 임상 데이터와 의료 이미지를 처리하는 방법을 소개합니다. 특히 안과 분야에서는 MLLMs가 광학 단층 촬영 보고서를 분석하고 질병 분류 및 치료 결과 예측을 돕는 데 사용되었습니다.

- **Technical Details**: MLLM의 성능 평가를 위해, 저자들은 439장의 망막 사진과 75장의 OCT 이미지를 포함한 데이터셋을 작성했습니다. 이 데이터셋은 정밀한 품질 관리와 전문가 주석을 통해 구축되었으며, 7개의 주요 MLLM을 API 기반의 표준화된 프레임워크를 통해 평가했습니다. 연구는 협의된 진단 기준에 따라 이미지의 해석 능력을 분석하는 데 중점을 두었습니다.

- **Performance Highlights**: 연구에서는 당뇨망막병증과 노인성 황반변성 같은 질병에 대한 진단에서는 양호한 성과를 보였지만, 신생혈관형 황반병증과 근시 등의 경우에는 일관성 부족을 나타냈습니다. 이는 MLLMs의 성능 개선과 보다 임상에 적합한 벤치마크 개발의 필요성을 강조합니다. 최종적으로 MLLMs는 안과 진단과 치료의 변화를 가져올 수 있는 잠재력을 가지고 있습니다.



### Linguistic Knowledge Transfer Learning for Speech Enhancemen (https://arxiv.org/abs/2503.07078)
Comments:
          11 pages, 6 figures

- **What's New**: 본 연구에서는 Cross-Modality Knowledge Transfer (CMKT) 학습 프레임워크를 제안하여 사전 훈련된 대규모 언어 모델(LLM)을 활용하여 언어적 지식을 음성 향상(SE) 모델에 통합합니다. 이 접근 방식은 추론 중에 텍스트 입력이나 LLM이 필요 없기 때문에 실용성을 높입니다. 또한, 기존의 언어 기반 SE 방법들이 가지는 데이터 의존성을 극복할 수 있는 가능성을 제공합니다.

- **Technical Details**: CMKT 프레임워크는 텍스트 정보를 직접 사용하지 않고 LLM의 임베딩을 통해 언어적 지식을 전달합니다. 연구에서는 시간적 이동(misalignment strategy)을 도입하여 모델이 더욱 강력한 표현을 학습하도록 유도하고, 주의 가중치 맵(attention weighting maps)을 사용하여 예측 작업의 해석 가능성을 높입니다. 또한, 다양한 SE 아키텍처와 LLM 임베딩에서 CMKT의 성능을 평가하고, 그 효과성을 입증합니다.

- **Performance Highlights**: CMKT는 다양한 SE 아키텍처와 구성에서 기준 모델 대비 일관되게 더 나은 성능을 보여줍니다. 실험은 만다린과 영어 데이터셋을 통해 진행되어, 다양한 언어적 조건에서도 효과적임을 확인했습니다. 텍스트 데이터 없이도 적용 가능하다는 점에서 실제 SE 작업에 매우 유용한 솔루션이 됩니다.



### DistiLLM-2: A Contrastive Approach Boosts the Distillation of LLMs (https://arxiv.org/abs/2503.07067)
Comments:
          The code will be available soon at this https URL

- **What's New**: 이 논문에서는 DistiLLM-2라는 새로운 대조적 접근법을 제안하여, 교사 모델과 학생 모델 간의 응답 가능성을 조정함으로써 LLM (Large Language Model) 증류의 효율성을 극대화했습니다. DistiLLM-2는 다양한 데이터 유형과 손실 기능 간의 시너지를 활용하여 학생 모델의 성능 향상을 도모합니다. 이를 통해 다양한 작업에서 높은 성능을 보이는 학생 모델을 구축할 수 있습니다.

- **Technical Details**: DistiLLM-2는 대칭이 없는 손실 역학을 분석하여, 교사와 학생 모델의 응답에 대해 서로 다른 손실 함수를 적용하는 대조적 방법(CALD; Contrastive Learning for Distillation)을 개발합니다. 이를 통해 손실 기능과 데이터 관점 간의 시너지를 효과적으로 통합합니다. 더불어, 데이터셋 커리는 최적화되었으며, 커리큘럼 기반의 적응형 손실 메커니즘이 도입되어, DistiLLM-2는 실무자들을 위한 강력한 지침을 제공합니다.

- **Performance Highlights**: DistiLLM-2는 다양한 텍스트 생성 작업에서 최첨단 성능을 달성하며, 지침 수행, 수학적 추론, 코드 생성 등을 포함합니다. 또한, 이 접근법은 선호 정렬(preference alignment)과 비전-언어 모델 확장과 같은 다양한 애플리케이션을 지원하여, 넓은 범위의 활용 가능성을 보여줍니다.



### DatawiseAgent: A Notebook-Centric LLM Agent Framework for Automated Data Scienc (https://arxiv.org/abs/2503.07044)
- **What's New**: 이번 논문에서 제안하는 DatawiseAgent는 데이터 과학의 작업을 유연하고 적응적으로 자동화하기 위한 노트북 중심의 LLM(대규모 언어 모델) 에이전트 프레임워크입니다. 이 프레임워크는 사용자, 에이전트 및 계산 환경 간의 상호작용을 통합하여 데이터 과학의 복잡한 작업을 효과적으로 처리할 수 있도록 설계되었습니다. 기존의 LLM 기반 접근 방식들이 특정 단계에 집중하는 경향이 있지만, DatawiseAgent는 연속적인 데이터 과학 작업의 의존성을 잘 반영합니다.

- **Technical Details**: DatawiseAgent는 유한 상태 변환기(Finite State Transducer, FST)를 기반으로 하는 다단계 설계를 사용하여 네 가지 주요 단계인 DFS(Depth First Search) 계획, 점진적 실행, 자기 디버깅 및 후처리를 조율합니다. 특히 DFS-like 계획 단계는 솔루션 공간을 체계적으로 탐색하고, 점진적 실행 단계는 실시간 피드백을 활용하여 제한된 LLM의 능력을 극대화합니다. 자기 디버깅 및 후처리 모듈은 오류를 진단하고 수정하여 신뢰성을 향상시키는 역할을 합니다.

- **Performance Highlights**: 실험 결과, DatawiseAgent는 다양한 데이터 과학 작업, 특히 데이터 분석, 시각화 및 데이터 모델링에서 기존의 최첨단 방법들과 비교하여 우수하거나 동등한 성능을 보여주었습니다. 가장 도전적인 데이터 모델링 작업에서도 DatawiseAgent는 90% 이상의 작업 완료율과 함께 40 이상의 상대 성능 격차(Relative Performance Gap)를 기록하며 탁월한 결과를 달성했습니다. 이러한 결과는 DatawiseAgent가 데이터 과학 시나리오에서 일반화할 수 있는 잠재력을 강조하며, 더욱 효율적이고 완전 자동화된 워크플로우를 위한 기초를 마련합니다.



### TCM-3CEval: A Triaxial Benchmark for Assessing Responses from Large Language Models in Traditional Chinese Medicin (https://arxiv.org/abs/2503.07041)
- **What's New**: 이 논문에서는 기존의 대형 언어 모델(LLM)이 전통 중국 의학(TCM) 분야에서 어떻게 평가될 수 있는지를 다루기 위해 TCM3CEval이라는 벤치마크를 도입합니다. 이 벤치마크는 핵심 지식 숙달, 고전 텍스트 이해 및 임상 의사 결정의 세 가지 차원에서 LLM을 평가합니다. 다양한 모델이 평가되었고, 전반적으로 전문 분야에서의 한계가 드러났습니다.

- **Technical Details**: TCM3CEval는 TCM에 대한 지식과 이해를 높이기 위해 LLM의 성능을 다차원적으로 평가하기 위한 체계적인 프레임워크를 제공합니다. 평가 기준에는 TCM 이론의 기초 이해, 고전 텍스트에 대한 해석력, 임상 사례에 대한 응답 능력이 포함됩니다. 각 차원에서는 다수의 질문 세트를 통해 모델의 성과를 평가합니다.

- **Performance Highlights**: 연구 결과, 중국 언어적 및 문화적 배경을 가진 모델이 고전 텍스트 해석과 임상 추론에서 더 나은 성능을 보였습니다. 또한, 기존의 LLM들이 TCM의 복잡한 이론과 개별화된 치료 과정을 충분히 반영하지 못하고 있다는 점에서 발전이 필요함이 확인되었습니다. 결국, TCM3CEval는 LLM이 TCM에 적합하도록 최적화할 수 있는 통찰력을 제공합니다.



### Bot Wars Evolved: Orchestrating Competing LLMs in a Counterstrike Against Phone Scams (https://arxiv.org/abs/2503.07036)
- **What's New**: 이번 논문에서는 'Bot Wars'라는 프레임워크를 소개합니다. 이 프레임워크는 대화형 적대적 기법을 통해 전화 사기를 방어하기 위해 대형 언어 모델(LLMs)을 활용합니다. 핵심 기여는 명시적인 최적화 없이 사고의 연쇄를 통한 전략의 출현(Strategy Emergence)에 대한 정형화된 기반을 제시하는 것입니다.

- **Technical Details**: Bot Wars는 두 계층의 프롬프트 아키텍처를 통해 전략적 일관성을 유지하면서 인구통계적으로 진정한 피해자 페르소나를 생성할 수 있는 LLM의 능력을 극대화합니다. 시스템은 사기꾼 에이전트와 피해자 에이전트 각각의 상대적인 목표를 가진 두 개의 적대적 에이전트를 포함하여 작동합니다. 프레임워크는 3,200개의 사기 대화 데이터셋을 기반으로 하여 179시간의 인간 사기 방어 상호작용과 검증된 평가를 실시합니다.

- **Performance Highlights**: 실험 평가 결과, GPT-4가 대화의 자연스러움과 페르소나의 진정성에서 뛰어난 성능을 보이는 반면, Deepseek는 더 나은 참여 지속성을 보여주었습니다. 다양한 지표를 통해 대화의 효과성을 정량화하였으며, 이에 따라 Bot Wars 프레임워크의 효과성을 입증하였습니다.



### Multimodal Human-AI Synergy for Medical Imaging Quality Control: A Hybrid Intelligence Framework with Adaptive Dataset Curation and Closed-Loop Evaluation (https://arxiv.org/abs/2503.07032)
- **What's New**: 이번 연구에서는 의학 영상 품질 관리(QC)를 위한 표준화된 데이터셋과 평가 프레임워크를 구축하여 대형 언어 모델(LLM)들을 체계적으로 평가했습니다. 기존의 QC 방법들이 인력 소모가 크고 주관적이었던 반면, LLM들이 이러한 문제를 해결할 가능성을 보여주고 있습니다. 특히, 161개의 흉부 X-레이(Chest X-ray, CXR) 및 219개의 CT 보고서를 사용하여 기술적 오류 및 불일치를 탐지하는 데 초점을 맞추었습니다.

- **Technical Details**: 연구에서 사용된 데이터셋은 임상 경력이 15년 이상인 방사선 전문의 다섯 명의 감독 하에 수집된 161개의 흉부 X-레이 및 219개의 CT 보고서입니다. 이 데이터는 Python-GDCM 라이브러리를 사용하여 DICOM 헤더를 익명화하고, LLM 기반 자연어 처리(NLP) 기법으로 문서의 민감한 정보를 제거했습니다. 시각적 품질 확보를 위해 ACR 및 IEC 국제 가이드라인에 맞춘 11가지의 기술 기준을 적용하였고, CT 보고서 오류를 8개 주요 유형으로 체계화하였습니다.

- **Performance Highlights**: 실험 결과, Gemini 2.0-Flash는 CXR 과제에서 Macro F1 점수 90을 기록해 강력한 일반화 능력을 입증하였으나, 세부 성능은 제한적이었습니다. DeepSeek-R1은 CT 보고서 감사에서 62.23%의 재현율을 보여 다른 모델에 비해 우수한 성과를 기록했습니다. 반면, 몇몇 변형 모델의 성능은 떨어졌으며, InternLM2.5-7B-chat은 추가 발견률에서 가장 높은 성과를 보였으나 정확성은 떨어지는 경향을 보였습니다.



### Toward Multi-Session Personalized Conversation: A Large-Scale Dataset and Hierarchical Tree Framework for Implicit Reasoning (https://arxiv.org/abs/2503.07018)
Comments:
          Preprint

- **What's New**: 이 연구는 대화 에이전트의 복잡한 특정성을 더욱 증대시키기 위해 ImplexConv라는 새로운 대화 데이터셋을 소개합니다. 기존 데이터셋은 복잡하고 실제적인 개인화 요소를 제공하지 않으며, 암묵적 추론을 포착하지 못했습니다. 이 데이터셋은 2,500개의 예제를 포함하며, 각 예제는 약 100회의 대화를 포함하여 이상적인 개인화 대화를 연구하기 위해 설계되었습니다. 또한 TaciTree라는 새로운 계층적 프레임워크를 제안하여 대화 기록을 여러 요약 수준으로 구조화합니다.

- **Technical Details**: ImplexConv는 암묵적 추론을 지원하는 여러 전혀 다른 개인적 특성을 결합합니다. TaciTree는 대화 이력을 트리 구조로 조직하여, 모델이 필요할 때 적절한 정보를 선택해 추가적인 세부 사항으로 들어갈 수 있게 합니다. 이러한 계층적 접근 방식은 검색 공간을 크게 줄이면서도 높은 정확도를 유지하여, LLM들이 그들의 추론 능력을 활용할 수 있게 합니다. ImplexConv 데이터셋의 각 대화 세션은 약 100회의 대화로 이루어져 있으며, 이는 대화의 일관성을 유지합니다.

- **Performance Highlights**: 실험 결과에 따르면, TaciTree는 기존의 기억 기반 메모리 가져오기에서 30% 높은 정확도를 달성했습니다. ImplexConv는 기존 데이터셋에 비해 20% 낮은 의미 유사성을 나타내며, 이는 높은 암묵성의 도전 과제를 반영합니다. 특히, TaciTree는 40-60% 더 적은 토큰으로도 높은 정확도를 유지하여, 필요한 정보의 효율적인 추출을 가능하게 합니다. 데이터셋과 소스 코드는 공개되어 있어, 다른 연구자들이 활용할 수 있습니다.



### Large Language Models Often Say One Thing and Do Another (https://arxiv.org/abs/2503.07003)
Comments:
          Published on ICLR 2025

- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 신뢰성을 평가하는 데 있어 언급된 내용과 실제 행동 간의 일관성을 그 핵심 과제로 다룹니다. 이를 위해 새롭게 개발된 평가 기준인 Words and Deeds Consistency Test (WDCT)를 소개하며, 이 기준은 다양한 도메인에서의 언어 기반 질문과 행동 기반 질문 간의 엄격한 일치를 설정합니다. 실험 결과는 LLM 모델들이 단어와 행동 간에 광범위한 비일관성을 보이고 있다는 사실을 드러냅니다.

- **Technical Details**: WDCT는 단어 질문과 행동 질문의 쌍을 통해 LLM의 일관성을 정량적으로 측정할 수 있도록 설계되었습니다. 각 질문 쌍은 특정 상황과 행동에서의 모델의 의견, 가치 등을 직접적으로 파악할 수 있도록 구성됩니다. 평가 결과, LLM의 단어와 행동 간의 비일관성이 일반적이고 상당한 문제임을 밝히며, 개별 조정이 오히려 비일관성을 악화시킨다는 점이 강조됩니다.

- **Performance Highlights**: 논문에서는 언어와 행동의 조정이 LLM의 일관성에 미치는 영향을 실험으로 분석하였습니다. 실험 결과, 단어나 행동 중 하나에 대한 조정은 다른 측면에 악영향을 미치는 경향을 보였으며, 이는 LLM이 단어와 행동 선택에 있어 통합된 지식 공간을 갖지 않음을 시사합니다. 마지막으로, 데이터 증식이나 명확한 추론 방식 등이 일관성 향상에 기여하지 않는다는 사실을 발견하게 되었습니다.



### Social Bias Benchmark for Generation: A Comparison of Generation and QA-Based Evaluations (https://arxiv.org/abs/2503.06987)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)의 사회적 편향을 측정하기 위해 'Bias Benchmark for Generation (BBG)'라는 새로운 벤치마크를 제안합니다. 기존의 QA 기반 평가 방식은 긴 형식의 생성에서 발생하는 편향을 충분히 포착하지 못하므로, LLM이 이야기의 진행을 생성하는 과정을 통해 편향을 측정하는 방식을 채택했습니다. BBG는 영어와 한국어에서 구축되어, LLM의 생성 결과에서 중립적 및 편향된 생성의 확률을 측정합니다.

- **Technical Details**: BBG는 이야기 생성의 편향을 측정하기 위해 두 가지 지표, 중립성 점수(ntr_gen) 및 편향 점수(bias_gen)를 도입합니다. 중립성 점수는 모델이 특정 개인에 대한 연관을 하지 않거나, 두 개의 서로 다른 버전에서 동일한 순서대로 개인에 대해 일관되게 매핑할 때의 비율을 측정합니다. 편향 점수는 비편향적 생성과 편향적 생성의 비율 차이로 정의되며, 이를 통해 LLM의 사회적 편향을 정량화합니다.

- **Performance Highlights**: 실험 결과, 10개의 LLM 중 대부분이 생성하는 출력의 49%에서 69%가 중립적이며, 편향에 맞는 생성의 확률이 편향에 반하는 출력보다 10%에서 25% 더 높습니다. 기존의 BBQ 평가와 비교했을 때, 두 접근 방식의 결과는 일관성이 없음을 보여주며, 동일한 모델 내에서도 일반 성능이 높은 모델이 QA 작업에서는 낮은 편향 점수를, 생성 작업에서는 높은 편향 점수를 보이는 경향이 있음을 확인했습니다.



### Exploring Multimodal Perception in Large Language Models Through Perceptual Strength Ratings (https://arxiv.org/abs/2503.06980)
Comments:
          under review, 15 pages

- **What's New**: 이번 연구에서는 대형 언어 모델(Large Language Models, LLMs)의 다중 양식 인식(multimodal perception)을 조사하였으며, 인간과 유사한 감각 양식(sensory modalities)에서의 강도 평정(perceptual strength ratings)을 포착하는 능력에 초점을 맞추었습니다. GPT-3.5, GPT-4, GPT-4o, GPT-4o-mini 모델들이 비교되었으며, 다중 양식 입력이 언어적 추론(linguistic reasoning) 및 그라우딩(grounding)에 미치는 영향을 강조했습니다.

- **Technical Details**: 연구는 감각 강도 평정 수치를 기준으로 하여 모델들의 성능을 비교했습니다. GPT-4와 GPT-4o는 인간 평가와의 높은 일치를 보였지만, 상대적으로 작은 모델들보다 큰 진전을 이루었습니다. 하지만, 퀄리티 분석(qualitative analyses)은 다중 감각 과대 평가(multisensory overrating) 및 느슨한 의미 연관(semantic associations)에 대한 의존성을 드러내며, 처리 양식에서 뚜렷한 차이를 밝혔습니다.

- **Performance Highlights**: GPT-4o는 다중 양식 기능을 통합했음에도 불구하고, GPT-4에 비해 더 나은 그라우딩을 보이지 않았습니다. 이는 인간과 유사한 그라우딩을 향상시키는 역할에 대한 질문을 제기합니다. 이러한 결과는 LLM이 언어적 패턴에 의존하여 인간의 신체적 인지(embodied cognition)를 어느 정도 근사(approximate)하고, 때로는 차이가 있음을 보여주며, 감각 경험(replicating sensory experiences)을 복제하는 데 한계를 가지고 있음을 드러냅니다.



### CtrlRAG: Black-box Adversarial Attacks Based on Masked Language Models in Retrieval-Augmented Language Generation (https://arxiv.org/abs/2503.06950)
- **What's New**: 이 연구에서는 Retrieval-Augmented Generation (RAG) 시스템의 보안 취약점을 다루고 있습니다. 특히, 악의적인 사용자가 지식베이스에 악성 콘텐츠를 주입하여 모델의 출력을 조작할 수 있는 새로운 공격 방법인 CtrlRAG를 제안합니다. 이 연구는 기존의 화이트박스 공격 방식과는 달리 블랙박스 환경에서의 공격을 중점적으로 탐구하고 있어 실용적인 상황에 적합합니다.

- **Technical Details**: CtrlRAG 공격 방법은 Masked Language Model (MLM)을 활용하여 환경에 따라 동적으로 악의적인 콘텐츠를 최적화하는 기법을 포함합니다. 두 가지 유형의 악성 콘텐츠인 지침(instructions)과 지식(knowledge)을 분류하고, 이를 바탕으로 초기 악성 텍스트를 생성하여 검색 결과에 포함시키는 과정을 설명합니다. 또한 이 방법은 검색된 문맥의 변화에 따라 악성 텍스트를 조정하여 최적의 성능을 발휘하도록 구성되어 있습니다.

- **Performance Highlights**: 실험 결과, CtrlRAG는 감정 조작(emotional manipulation) 및 환각 강화(hallucination amplification) 목표에서 총 세 가지 기준선을 초과하는 성능을 보여주었습니다. 특히, CtrlRAG는 다양한 RAG 시스템에서 70%가 넘는 공격 성공률(ASR)을 달성하였고, 감정적 영향을 미치는 출력에서도 유의미한 결과를 나타냈습니다. 기존의 방어 메커니즘은 CtrlRAG에 대해 효과가 제한적임을 보여주어 더욱 강력한 방어책의 필요성을 강조합니다.



### Lshan-1.0 Technical Repor (https://arxiv.org/abs/2503.06949)
- **What's New**: 이 보고서에서는 첫 번째 세대 추론 모델인 Lshan-1.0을 소개합니다. 이 모델은 중국의 법률 분야에 특화된 대형 언어 모델로, 다양한 실제 요구를 충족할 수 있는 종합적인 기능을 제공합니다. 기존의 법률 LLM들은 컴퓨터 과학 관점에서 주로 설계되었기 때문에 법률 전문성과 논리적 사고가 부족하여 고정밀 법률 응용에 문제가 있었습니다.

- **Technical Details**: Lshan-1.0은 중국의 31개 성에서 20종 이상의 범죄를 포괄하는 수백만 개의 법률 문서를 수집하여 모델 훈련에 사용합니다. 데이터셋에서 고품질 데이터를 선택하여 감독하에 미세 조정(supervised fine-tuning)을 수행하며, 그 이후에 추가 감독 없이 대규모 강화학습(large-scale reinforcement learning)을 진행하여 추론 능력과 설명 가능성을 강화합니다.

- **Performance Highlights**: Lshan-1.0은 복잡한 법률 응용의 효과를 검증하기 위해 법률 전문가와의 인간 평가(human evaluations)를 통해 그 유효성을 평가합니다. 또한 이 모델은 DeepSeek-R1-Distilled 버전 기반으로 미세 조정된 모델이 개발되었으며, 14B, 32B, 70B의 세 가지 밀집 구성으로 제공됩니다.



### Effect of Selection Format on LLM Performanc (https://arxiv.org/abs/2503.06926)
- **What's New**: 이번 논문은 대규모 언어 모델(LLM)의 성능에 중요한 요소인 분류 작업 옵션의 최적 형식에 대해 조사합니다. 우리는 많은 실험을 통해 두 가지 선택 형식인 불릿 포인트(bullet points)와 일반 영어(plain English)를 비교하여 모델 성능에 미치는 영향을 검토했습니다. 연구 결과, 불릿 포인트 형식이 일반적으로 더 나은 결과를 내는 것으로 나타났습니다.

- **Technical Details**: 이 연구는 LLM의 성능을 평가하기 위해 10개의 서로 다른 도메인별 작업을 대상으로 실험을 수행했습니다. 각 실험에서는 불릿 포인트와 일반 설명 형식으로 제시된 프롬프트를 비교하여, 이를 통해 LLM의 반응을 측정했습니다. 평가 지표로는 가중 평균 정밀도(precision), 재현율(recall), 그리고 F1 점수를 사용했습니다.

- **Performance Highlights**: 실험 결과, 불릿 포인트 형식을 사용한 경우가 일반 설명 형식을 사용할 때보다 더 높은 성능을 보인 것으로 확인되었습니다. 이 연구는 LLM의 취지에 맞게 프롬프트 설계를 최적화하는 데 기여할 수 있는 중요한 통찰을 제공할 것으로 기대됩니다. 이후 연구는 프롬프트 형식을 추가로 탐구하여 모델 성능을 더욱 개선할 방향을 모색해야 합니다.



### Automatic Speech Recognition for Non-Native English: Accuracy and Disfluency Handling (https://arxiv.org/abs/2503.06924)
Comments:
          33 pages, 10 figures

- **What's New**: 이번 연구는 비원어민 액센트 영어 음성을 인식하는 최신 자동 음성 인식 시스템(ASR)의 정확도를 평가합니다. L2-ARCTIC 코퍼스의 음성 샘플을 사용하여, 아랍어, 중국어, 힌디어, 한국어, 스페인어, 베트남어를 사용하는 화자들이 포함되었습니다. 이는 읽기 음성과 자발적 음성 모두에 대해 평가되었으며, 현재의 ASR 기술이 언어 학습 애플리케이션에 얼마나 적합한지를 보여줍니다.

- **Technical Details**: 연구에서는 24명의 화자로부터 2,400개의 문장 단위로 읽은 음성과 22명의 화자로부터의 내러티브 음성을 포함하여 두 가지 유형의 음성을 분석했습니다. Whisper와 AssemblyAI는 각각 0.054와 0.056의 평균 일치 오류율(Match Error Rate, MER)로 읽기 음성에서 가장 높은 정확도를 달성했습니다. 자발적 음성에 대해서는 RevAI가 평균 MER 0.063으로 가장 높은 성능을 보였습니다.

- **Performance Highlights**: 연구는 각 ASR 시스템이 필러 단어, 반복 및 수정과 같은 비유창성(disfluency)을 처리하는 방식도 검토하며, 시스템 간 및 비유창성 유형에 따라 성능에서 상당한 차이를 발견했습니다. 처리 속도는 시스템 간에 차이가 컸으며, 처리 시간이 길어질수록 정확도가 반드시 향상되는 것은 아니었습니다. 이 연구는 비원어민 영어 음성에 대한 최신 ASR 시스템의 성능을 자세히 설명하여 언어 교수 및 연구자들이 각 시스템의 장단점을 이해하고 특정 용도에 적합한 시스템을 식별하는 데 도움을 줍니다.



### KwaiChat: A Large-Scale Video-Driven Multilingual Mixed-Type Dialogue Corpus (https://arxiv.org/abs/2503.06899)
- **What's New**: 이번 연구에서는 비디오 기반의 다중 참가자 대화 시스템의 한계를 지적하고, 이 시스템이 다양한 대화 유형을 지원하는 필요성을 언급합니다. 새로운 과제로 비디오 구동 다국어 혼합형 대화를 생성하는 방법을 제안하며, KwaiChat이라는 데이터셋을 생성했습니다. KwaiChat은 총 93,209개의 비디오와 246,080개의 대화로 구성되어 있으며, 4가지 대화 유형과 30개 도메인, 4개 언어를 포함합니다.

- **Technical Details**: KwaiChat 데이터셋은 비디오 공유 플랫폼인 Kwai에서 수집된 다양한 언어로 구성된 혼합형 다중 참가자 대화를 포함합니다. 이 데이터셋은 세 가지 데이터 필터링 전략 및 적응형 비디오 균형 방법을 사용하여 품질을 유지하고, 데이터의 장기적 분포 문제를 해결합니다. 실험은 7개의 대형 언어 모델(LLMs)을 대상으로 하여, 0-shot, in-context learning, fine-tuning 성능을 평가하였습니다.

- **Performance Highlights**: 실험 결과, 최신 LLM들도 KwaiChat에서 비디오 기반 혼합형 다중 참가자 대화를 생성하는 데 한계를 보였습니다. 특히, in-context learning과 fine-tuning을 적용했음에도 불구하고 낮은 성능을 보여 이 과제가 단순하지 않음을 증명했습니다. 이는 추가 연구의 필요성을 강하게 시사합니다.



### A LongFormer-Based Framework for Accurate and Efficient Medical Text Summarization (https://arxiv.org/abs/2503.06888)
Comments:
          Paper accepted by 2025 8th International Conference on Advanced Algorithms and Control Engineering (ICAACE 2025)

- **What's New**: 이 논문에서는 LongFormer를 기반으로 한 의학 텍스트 요약 방법을 제안합니다. 기존 모델들이 긴 의학 텍스트를 처리할 때 겪는 어려움을 해결하고자 하며, 전통적인 요약 방법의 단기 기억의 한계를 극복합니다. LongFormer는 장기적 의존성(long-range dependencies)을 효과적으로 포착하여 더 많은 주요 정보를 보존하고 요약의 정확성과 정보 보존을 향상시킵니다.

- **Technical Details**: LongFormer는 장기 자기 주의(long-range self-attention) 메커니즘을 도입하여 긴 텍스트에서 중요한 정보를 유지합니다. 실험 결과, LongFormer 기반 모델은 RNN, T5, BERT와 같은 전통적인 모델보다 자동 평가 메트릭인 ROUGE에서 높은 성과를 보였습니다. 전문가 평가에서도 높은 점수를 획득하였으며, 정보 보존(information retention) 및 문법적 정확성(grammatical accuracy) 측면에서 특히 뛰어난 성능을 보였습니다.

- **Performance Highlights**: 하지만 간결성(conciseness)과 가독성(readability) 측면에서는 더 개선할 여지가 있습니다. 일부 전문가들은 생성된 요약이 중복 정보를 포함하고 있어 간결성에 영향을 미친다고 언급했습니다. 앞으로의 연구는 모델 구조 최적화에 중점을 두어 더 효율적인 의학 텍스트 요약을 목표로 하고 있습니다. 의료 데이터가 계속 증가함에 따라 자동 요약 기술은 의학 연구, 임상 의사 결정 지원, 지식 관리 등 다양한 분야에서 점점 더 중요한 역할을 할 것입니다.



### Lost-in-the-Middle in Long-Text Generation: Synthetic Dataset, Evaluation Framework, and Mitigation (https://arxiv.org/abs/2503.06868)
- **What's New**: 이 논문에서는 기존의 긴 텍스트 생성 방식이 짧은 입력에서 긴 텍스트를 생성하는 것에 집중하고, 긴 입력과 긴 출력 작업을 소홀히 하고 있음을 지적합니다. 이를 해결하기 위해 Long Input and Output Benchmark (LongInOutBench)를 도입하고, 여기에는 합성 데이터셋과 포괄적인 평가 프레임워크가 포함되어 있습니다. 이러한 벤치마크는 기존의 문제를 해결하는 데 기여할 것입니다.

- **Technical Details**: 이 연구에서는 Retrieval-Augmented Long-Text Writer (RAL-Writer)를 개발하여, 중요한 내용들을 검색하고 재진술하는 방식으로 'lost-in-the-middle' 문제를 해결합니다. RAL-Writer는 명시적인 프롬프트를 구성하여 정보를 효과적으로 복원합니다. 연구팀은 LongInOutBench를 사용하여 RAL-Writer의 성능을 유사한 기준선과 비교했습니다.

- **Performance Highlights**: 실험 결과, RAL-Writer는 제안한 벤치마크에 대해 기존의 방법들보다 우수한 성능을 보였습니다. 이는 긴 입력과 긴 출력을 효과적으로 처리할 수 있는 잠재력을 보여줍니다. 최종적으로, 연구팀은 그들의 코드를 공개하여 연구자들이 이 작업을 재현하고 발전시킬 수 있도록 지원하고 있습니다.



### Enhanced Multi-Tuple Extraction for Alloys: Integrating Pointer Networks and Augmented Attention (https://arxiv.org/abs/2503.06861)
Comments:
          17 pages, 5 figures

- **What's New**: 본 연구는 다중 주성분 합금(multi-principal-element alloys)에서 기계적 특성을 효과적으로 추출하기 위한 새로운 방법을 제안합니다. 제안된 방법은 MatSciBERT 기반의 개체 추출 모델과 포인터 네트워크(pointer networks), 그리고 상호 및 내부 개체 주의(attention) 메커니즘을 활용하는 할당 모델을 통합하고 있습니다. 이를 통해, 다양하고 복잡한 형태의 튜플 정보를 정확하게 추출할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 기계적 특성을 포함하는 다중 튜플(multi-tuple)을 추출하기 위해, 본 연구에서는 255개의 문장으로 구성된 데이터셋을 사용하여 각 문장 내 튜플 수에 따라 데이터셋을 분리하였습니다. MatSciBERT 모델과 포인터 네트워크를 활용하여 개체를 효과적으로 식별하고 데이터를 구조화합니다. 이 모델은 다수의 튜플을 포함하는 문장을 처리할 수 있도록 설계되었습니다.

- **Performance Highlights**: 모델의 성능은 1, 2, 3, 4개의 튜플 데이터셋에 대해 각각 0.963, 0.947, 0.848, 0.753의 F1 점수를 기록하였으며, 무작위로 선택된 데이터셋에서는 0.854의 F1 점수를 달성했습니다. 이러한 성과는 제안된 접근 방식이 대규모 자연어 처리 모델과 비교할 때 보다 정교하고 효과적임을 입증합니다.



### On the Mutual Influence of Gender and Occupation in LLM Representations (https://arxiv.org/abs/2503.06792)
Comments:
          In submission

- **What's New**: 이 연구는 대규모 언어 모델(LLMs)이 직업 맥락에서 성별을 어떻게 인식하는지를 분석합니다. 특히, 이름이 주는 성별 편견이 실제 성별 통계와 상관관계가 있음을 보여주고, 이러한 성별 표현이 직업 예측 작업에 미치는 영향을 탐구합니다. 이 연구는 기존의 블랙 박스 접근 방식에서 벗어나 LLM의 내부 성별 표현 및 편향된 행동 간의 상관관계를 밝혀내는 데 중점을 두고 있습니다.

- **Technical Details**: 연구에서는 먼저 성별 방향을 근사하기 위해 기존의 성별 방향 알고리즘을 활용합니다. 네 가지 개방형 LLM(예: Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3 등)을 사용하여 각 모델의 내부 임베딩을 분석합니다. 이를 통해 모델들이 성별과 직업에 대한 편견을 어떻게 반영하고 있는지를 체계적으로 조사합니다.

- **Performance Highlights**: 이 연구의 결과는 LLM이 특정 이름과 관련된 성별이 직업 예측에 미치는 영향을 강조합니다. 예를 들어, 여성의 이름이 주어지면 여성 중심의 직업에 대해 더 높은 확률을 보이고, 반대로 남성의 이름은 남성 중심의 직업에 대한 예측을 높이는 경향을 보입니다. 그러나 이러한 성별 표현을 기반으로 편향을 탐지하는 것은 여전히 도전적인 과제로 남아 있습니다.



### Dr Genre: Reinforcement Learning from Decoupled LLM Feedback for Generic Text Rewriting (https://arxiv.org/abs/2503.06781)
Comments:
          29 pages, 4 figures, 25 tables

- **What's New**: 이번 연구에서는 다양한 재작성 목표를 처리할 수 있는 일반적인 모델을 도입합니다. 특히 사실 수정(factuality), 스타일 변환(style transfer), 대화형 재작성(conversational rewriting)에 능숙한 모델을 개발하였습니다. 이를 위해, 자연스러운 지침을 제공하는 대화형 데이터셋인 ChatRewrite를 구성하였습니다. 또한 Dr Genre라는 새로운 프레임워크를 통해 성능 개선을 위한 목표 지향 보상 모델을 제안합니다.

- **Technical Details**: 이 모델은 다양한 NLP 작업에서 요구되는 재작성 능력을 통합하기 위해 세 가지 주요 목표를 분리하여 설정합니다. 목표는 '합의'(agreement), '일관성'(coherence), 그리고 '간결성'(conciseness)으로 정의되며, 각 목표는 재작성의 품질 향상을 위해 밀접하게 연결되어 있습니다. Dr Genre는 이를 기반으로 한 세밀한 조정을 통해 작업 요구 사항에 따라 목표 지향 보상 모델의 가중치를 조정할 수 있는 능력을 제공합니다.

- **Performance Highlights**: 실험 결과, Dr Genre는 모든 목표 지정 작업에서 높은 품질의 재작성을 제공하며, 지침 준수, 내적 일관성 및 불필요한 수정 최소화와 같은 여러 목표 개선을 나타냅니다. 특히, 사용자 요구에 맞춘 대화형 재작성 작업에서 두드러진 성과를 보였으며, 이는 사용자 적용 가능성을 크게 향상시킵니다.



### Large Language Models Are Effective Human Annotation Assistants, But Not Good Independent Annotators (https://arxiv.org/abs/2503.06778)
Comments:
          9 pages, 4 figures

- **What's New**: 이번 연구는 이벤트 주석(annotation)이 시장 변화 및 사회적 트렌드를 이해하는 데 얼마나 중요한지를 강조합니다. 기존의 전문가 주석이 효율성이 떨어지는 점을 해결하기 위해 LLM(대형 언어 모델) 기반 자동 주석을 도입하여 변수 주석의 시간과 정신적 노력을 줄일 수 있는 가능성을 보여줍니다.

- **Technical Details**: 이 연구는 복합적인 워크플로를 평가하여 관련 없는 문서를 제거하고, 동일한 이벤트에 대한 문서를 병합하며, 이벤트를 주석 처리하는 방법을 제시합니다. 특히 LLM이 전문가 주석자에게 지원이 될 수 있는 방법을 모델링하여, 문서의 정보 추출 과정을 보다 효율적으로 만들 수 있습니다.

- **Performance Highlights**: 이벤트 주석 과정에서 LLM 기반 시스템이 기존의 TF-IDF 방법이나 이벤트 세트 큐레이션(Event Set Curation)보다 우수한 성능을 보임을 나타냅니다. 그러나 LLM이 완전히 자동화된 주석자보다는 인간 전문가와의 협업을 통해 더욱 향상되는 성과를 거두게 됩니다.



### Effectiveness of Zero-shot-CoT in Japanese Prompts (https://arxiv.org/abs/2503.06765)
Comments:
          NLP2025 Workshop on Japanese Language Resources (JLR2025)

- **What's New**: 이 연구는 일본어와 영어에서의 zero-shot Chain-of-Thought (CoT) 프로팅의 효과성을 비교합니다. 요약하자면, zero-shot CoT는 문제 해결 전에 "단계별로 생각해봅시다"라는 문구를 추가하여 사고를 촉진하는 기술입니다. 그 결과, 일본어에서의 성과 일부 카테고리에서는 개선이 나타났지만, 더 진보된 모델인 GPT-4o-mini에서는 오히려 상당한 성능 저하가 발생했습니다.

- **Technical Details**: 이 연구는 일본어와 영어에서의 zero-shot CoT 프로팅을 분석하기 위해 Multi-task Language Understanding Benchmark (MMLU)와 일본어 버전인 JMMLU를 사용했습니다. GPT-3.5에서는 일부 프롬프트 카테고리에서 성과 향상이 관찰되었으나, GPT-4o-mini에서는 전반적으로 부정적인 영향을 미쳤습니다. 각 모델의 성과 변화를 이해하기 위해 다양한 주제 영역에서 데이터 분석을 수행했습니다.

- **Performance Highlights**: 전반적인 성과 분석 결과, CoT 프롬프트가 적용된 경우 두 언어 모두에서 부정적인 영향을 미쳤습니다. 특히 일본어 프롬프트의 경우 초등 수학에서 가장 큰 개선이 있었으나, 국제법과 같은 특정 과목에서는 오히려 성능이 저하되었습니다. 통계 분석 결과, 일본어에서 CoT 노출이 유의미한 영향을 미친 것으로 나타난 반면, 영어에서는 그 효과가 더욱 변동성이 있었습니다.



### Gender Encoding Patterns in Pretrained Language Model Representations (https://arxiv.org/abs/2503.06734)
Comments:
          Proceedings of the 5th Workshop on Trustworthy Natural Language Processing (TrustNLP 2025)

- **What's New**: 이 연구는 미리 훈련된 언어 모델(PLMs)에서 성별 편향이 어떻게 형성되고 전파되는지를 정보 이론적 접근법을 통해 분석합니다. 특히, PLM의 내부 표현에서 성별 정보와 편향이 어떻게 인코딩되는지를 탐구하며, 기존의 디바이징 기법의 효과를 검토합니다. 그 결과, 다양한 모델에서 성별 인코딩의 일관된 패턴이 발견되었고, 일부 디바이징 기법은 오히려 내부 표현에서 편향을 증가시키는 경향을 보였습니다.

- **Technical Details**: 연구에서는 Minimum Description Length (MDL) 프로빙 기법을 사용하여 다양한 인코더 기반 아키텍처에서 성별 편향이 어떻게 인코딩되는지를 분석합니다. PLM의 여러 층을 검사함으로써 편향이 가장 두드러지게 나타나는 층을 규명하고, 사전 훈련된 디바이징 목표가 포스트 호크(post-hoc) 완화 접근법보다 인코딩된 편향을 줄이는 데 더 효과적임을 입증합니다. 이는 디바이징 기법이 모델의 편향된 대표성을 다루는 데 있어 중요한 통찰을 제공합니다.

- **Performance Highlights**: 이 연구는 PLM의 다양한 아키텍처 간에 성별 인코딩의 지속적인 패턴을 드러냅니다. 또한, 디바이징 기법의 효과성에 대한 통찰을 제공하며, 많은 경우 이러한 기법들이 오히려 내부 표현에서의 편향을 증가시키는 결과를 초래함을 지적합니다. 이를 통해 편향 완화 전략을 개발하는 데 있어 가치 있는 지침을 제공합니다.



### Topology of Syntax Networks across Languages (https://arxiv.org/abs/2503.06724)
Comments:
          Final Thesis for MSc in Computational and Applied Mathematics at UC3M

- **What's New**: 본 연구에서는 다양한 언어의 구문 네트워크를 분석하여 언어 간의 유사성과 차이점을 발견하려고 한다. 특히, 이전 연구보다 더 많은 언어와 특성을 고려하여 분석의 깊이를 더한다. 실험적 접근법을 통해 각 언어의 문법적 클래스(Part of Speech)와 유사한 기능을 가진 단어의 집합을 조사한다. 이를 통해 구문 네트워크 간의 보편적으로 유지되는 구조적 패턴을 발견하고자 한다.

- **Technical Details**: 이 논문은 구문 네트워크를 구성하기 위해 특정 언어의 문장에서 직접적으로 의존하는 단어들 간의 관계를 고려한다. 언어 네트워크는 고유 단어를 노드로 하고, 서로 구문적으로 의존하는 두 단어를 연결하여 복잡한 그래프를 생성한다. 네트워크의 분석은 단어들이 서로 비슷한 위상적 특성을 가진 커뮤니티로 클러스터링되는 방식을 포함하며, 이를 통해 언어 간의 구문적 차이를 비교하였다. 또한, 전체 구문 네트워크의 글로벌 위상적 특성 및 개별 네트워크의 단어 클러스터링을 분석한다.

- **Performance Highlights**: 연구 결과는 50개 언어의 구문 네트워크에서 나타나는 보편적인 구조를 발견하였다. 이는 스페인어를 토대로 구체적인 사례를 통해 설명되며, 네트워크의 구조적 특성을 드러낸다. 연구에 나타난 결과는 비교 언어학 및 신경언어학의 분야에서도 중요한 의미를 가질 것으로 예상된다. 기존 문헌에서 보이지 않았던 의문점들을 해결하고, 각 언어 간의 새로운 유사성을 도출해낼 수 있는 가능성을 보여준다.



### Delusions of Large Language Models (https://arxiv.org/abs/2503.06709)
- **What's New**: 이번 연구에서는 LLM(대형 언어 모델)에서 흔히 발생하는 사실과 상관없이 그럴듯한 출력을 생성하는 'hallucination'과는 달리, LLM delusion이라는 더 심각한 현상을 밝혀냈습니다. LLM delusion은 비정상적으로 높은 확신을 가진 잘못된 출력을 의미하며, 이러한 잘못된 내용은 낮은 불확실성으로 지속되어 모델의 신뢰성을 떨어뜨립니다. 이는 검출 및 완화가 특히 어려운 도전과제가 됩니다.

- **Technical Details**: 연구에서 사용된 방법론으로는 세 가지 불확실성 추정 기술이 있습니다: logit 기반 방법으로 토큰의 확률 분포를 평가하고, 언어화된 신뢰도에서는 모델이 자신의 신념을 명시적으로 표현하며, 일관성 기반 방법은 여러 샘플의 응답 안정성을 평가합니다. LLM delusion은 특정 신념 임계점을 초과하는 잘못된 응답으로 정의되며, 이는 전체 데이터셋에서 정답에 대한 평균 신뢰도로 empirically 결정됩니다.

- **Performance Highlights**: 연구 결과, LLM은 hallucination에 비해 delusion에서 사실 honesty가 낮고, fine-tuning이나 self-reflection을 통해 덜 수정됩니다. 특히, 잘못된 응답을 거부하도록 훈련해도 delusion은 여전히 높은 비율로 지속되고, 생성된 응답을 재고하도록 요청했을 때 오히려 delusional 출력을 계속 고수하는 경향이 있음을 확인했습니다. 외부 검증 방법인 retrieval-augmented generation 및 다중 에이전트 토론 시스템이 delusion을 줄이는 데 도움을 줄 수 있지만, 완전한 제거에는 여전히 많은 도전과제가 남아 있습니다.



### Alignment for Efficient Tool Calling of Large Language Models (https://arxiv.org/abs/2503.06708)
- **What's New**: 최근 도구 학습(tool learning)의 발전으로 인해 대형 언어 모델(LLM)은 외부 도구를 통합할 수 있게 되어, 작업 성능을 향상시키며 지식의 경계를 넓히고 있습니다. 그러나 도구 사용에 의존하면 성능, 속도 및 비용 사이의 트레이드오프가 발생하며, LLM이 도구 사용에 지나치게 의존하거나 자신감을 과신하는 경향이 있을 수 있습니다. 이 논문에서는 LLM이 그들의 지식 경계에 따라 더 지능적인 결정을 내릴 수 있도록 돕기 위한 다중 목표 정렬 프레임워크(multimodal alignment framework)를 제안합니다.

- **Technical Details**: 제안된 프레임워크는 확률적 지식 경계 추정(probabilistic knowledge boundary estimation) 및 동적 의사결정(dynamic decision making)을 결합하여, LLM이 자신감에 따라 도구를 사용할 때를 더 잘 평가할 수 있도록 합니다. 이 프레임워크는 두 가지 지식 경계 추정 방법(일관성 기반 추정과 절대 추정)을 포함하고, 이러한 추정을 모델의 의사결정 과정에 통합하기 위한 두 가지 훈련 전략을 제안합니다. 이를 통해 LLM은 불필요한 도구 사용을 줄이고 효율성을 높일 수 있습니다.

- **Performance Highlights**: 다양한 도구 호출 시나리오에서의 실험 결과, 제안된 프레임워크의 효과가 입증되어 불필요한 도구 호출을 크게 줄이고 도구 효율성(tool efficiency)을 개선했습니다. 또한, 다양한 도구 사용 상황에서의 평가를 통해, 제안된 알고리즘이 실제 상황에서 LLM의 도구 사용에 대한 의사결정을 어떻게 개선하는지를 보여주었습니다. 전반적으로, 이 연구는 LLM이 도구 사용을 더 효율적으로 관리하도록 도와줘서 작업 성과를 향상시키는 데 기여합니다.



### PFDial: A Structured Dialogue Instruction Fine-tuning Method Based on UML Flowcharts (https://arxiv.org/abs/2503.06706)
- **What's New**: 본 연구에서는 Process Flow Dialogue (PFDial) 데이터셋을 구축하여 고객 서비스 및 장비 유지보수에서 중요한 역할을 하는 프로세스 기반 대화 시스템의 성능을 향상시키는 것을 목표로 합니다. 이 데이터셋은 440개의 플로우차트에서 유래된 12,705개의 고품질 중국어 대화 명령어를 포함하고 있으며, 이는 5,055개의 프로세스 노드를 포괄합니다. 이 연구는 기존의 Large Language Models (LLMs)가 프로세스 제약을 유지하기에 어려움을 겪고 있음을 보여주고, 새로운 데이터셋을 통해 이를 해결하고자 합니다.

- **Technical Details**: PFDial 데이터셋의 구성은 PlantUML 명세를 기반으로 하여, 각 UML 플로우차트를 원자적 대화 단위로 전환하고 구조화된 5-튜플 (flowchart description, current state, user input, next state, robot output) 형식으로 제공합니다. 이를 통해 모델은 자연어 대화 능력을 유지하면서 정확한 상태 전이를 학습할 수 있습니다. 또한, PFDial-Hard 데이터셋을 추가하여 복잡한 역방향 전이를 처리하는 데 중점을 두고 분류된 90개의 특정 비즈니스 시나리오를 다루었습니다.

- **Performance Highlights**: 실험 결과, 0.5B 모델이 98.99%의 인-도메인 정확도와 92.79%의 아웃-오브-도메인 정확도를 달성하였으며, 8B 모델은 97.02%의 정확도로 GPT-4o를 11.00% 초과하며 결정 브랜치에서 43.88%의 향상을 보여주었습니다. 특히, 7B 모델은 단지 800개의 훈련 샘플로도 90% 이상의 정확도를 달성할 수 있음을 시사합니다. 이러한 성능은 PFDial 데이터셋의 우수성을 입증하며, 모형의 제어된 추론 능력을 효과적으로 향상시킵니다.



### InftyThink: Breaking the Length Limits of Long-Context Reasoning in Large Language Models (https://arxiv.org/abs/2503.06692)
- **What's New**: 본 논문에서는 InftyThink라는 새로운 패러다임을 도입하여 대규모 언어 모델의 장기 추론( long-context reasoning)에서 발생하는 컴퓨팅 비효율성을 극복합니다. 기존의 연속된 논리적 사고를 반복적인 과정으로 전환하여 중간 요약을 포함함으로써, 무한한 깊이의 추론을 가능하게 하고 단위 컴퓨팅 비용을 유지합니다. InftyThink는 전통적인 접근 방식에 비해 계산 복잡성을 크게 감소시키는 톱니형 메모리 패턴을 창출합니다.

- **Technical Details**: InftyThink는 복잡한 추론을 여러 개의 짧은 연관된 세그먼트로 나누어 컴퓨팅 효율성을 유지하면서 일관된 사고 흐름을 보장합니다. 각 세그먼트는 효율적인 길이를 유지하며, 이전 추론에서 요약된 내용을 기반으로 다음 세그먼트를 구축합니다. 이는 인간의 인지 과정에서 영감을 받아 설계되었으며, 복잡한 문제를 관리 가능한 부분으로 나누고 중간 진행 상황을 요약하는 방법과 유사합니다.

- **Performance Highlights**: 실험 결과, Qwen2.5-Math-7B 모델이 MATH500에서 3%, AIME24에서 8%, GPQA_diamond에서 6% 성능 향상을 보였습니다. InftyThink는 장기 추론 관련 데이터를 재구성하여 훈련 세트를 강화하고, 기존의 복잡한 문제 해결 능력을 보다 효율적으로 개선합니다. 이러한 결과는 깊이 있는 추론과 계산 효율성 간의 상상을 뛰어넘는 한계를 해결하였음을 보여줍니다.



### Enhancing NLP Robustness and Generalization through LLM-Generated Contrast Sets: A Scalable Framework for Systematic Evaluation and Adversarial Training (https://arxiv.org/abs/2503.06648)
- **What's New**: 이번 연구는 표준 NLP 벤치마크의 한계를 보완하기 위해, 데이터셋 아티팩트와 무의미한 상관관계로 인한 취약점을 포착하는 데 집중했습니다. 대조 세트(contrast sets)를 자동 생성하여 모델의 결정을 방어하는 경계 근처에서 도전하는 것입니다. 이 방법은 연구 분야에서의 접근성을 높이고, 다양한 대조 세트를 제공할 수 있게 합니다.

- **Technical Details**: 이 연구에서는 SNLI 데이터셋을 사용하여 3,000개의 예제로 구성된 대조 세트를 생성했습니다. 이 과정은 대형 언어 모델을 활용하여 자동화되었으며, 이는 인공지능 모델의 강인성을 평가하고 향상하는 데 중요한 역할을 합니다. 모델을 이러한 대조 세트로 미세 조정(fine-tuning)함으로써, 데이터에 대한 시스템적 변화를 처리하는 능력을 향상시켰습니다.

- **Performance Highlights**: 미세 조정을 통해, 모델은 시스템적으로 변화된 예제에 대한 성능을 향상시켰으며, 기존 테스트 정확도는 유지되었습니다. 또한, 새로운 변화에 대한 일반화(generalization) 능력도 소폭 향상되었습니다. 이 자동화된 접근 방식은 NLP 모델의 평가 및 개선을 위한 확장 가능한 솔루션(scalable solution)을 제공합니다.



### Beyond Decoder-only: Large Language Models Can be Good Encoders for Machine Translation (https://arxiv.org/abs/2503.06594)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)과 신경 기계 번역(NMT) 세계의 융합을 통해 보편적이고 효율적인 번역 모델을 탐색합니다. 전통적으로 사용되어온 인코더-디코더 아키텍처에서 변화하여, LLM을 NMT 인코더로 사용하는 새로운 접근법인 LaMaTE를 제안합니다. 이를 통해 훨씬 더 유연하고 최적화가 쉬운 번역 시스템을 구성할 수 있습니다.

- **Technical Details**: LaMaTE 모델은 복잡한 인코더와 경량 디코더를 결합하여 높은 품질의 번역을 낮은 디코딩 비용으로 생성할 수 있게 합니다. 이를 평가하기 위하여 새로운 벤치마크인 Comprehensive Machine Translation benchmark (ComMT)를 개발하여 다양한 번역 관련 작업을 측정하고자 합니다. 이 모델은 대규모 신경망을 사용하므로 연산 비용이 크지만, 이를 통해 얻는 유연성이 많은 이점을 제공합니다.

- **Performance Highlights**: 우리의 실험 결과, LaMaTE 모델은 다양한 작업에서 기존의 기준 시스템에 비해 동등하거나 더 나은 성과를 내며, 2.4부터 6.5배 빠른 추론 속도와 75%의 KV 캐시 메모리 차지 감소를 기록했습니다. 이러한 결과는 대형 언어 모델 시대에서도 NMT의 원리가 여전히 유효하다는 것을 시사합니다. LaMaTE는 여러 번역 작업에서 강력한 일반화 능력을 보여주어 차세대 번역 시스템 개발에 대한 흥미로운 방향성을 제시합니다.



### WildIFEval: Instruction Following in the Wild (https://arxiv.org/abs/2503.06573)
- **What's New**: 최근에 발전한 LLM(대규모 언어 모델)은 사용자 지침을 따르는 데 있어서 놀라운 성공을 거두고 있지만, 여러 가지 제약조건을 가진 지침을 처리하는 것은 여전히 큰 도전 과제입니다. 이에 따라 WildIFEval이라는 12K의 실제 사용자 지침 데이터셋을 소개하며, 이는 다양한 제약조건을 포함하고 있습니다. 이 데이터셋은 기존의 자료와 달리 자연스러운 사용자 프롬프트를 통해 폭넓은 렉시컬(lexical) 및 주제적(topic) 제약을 포괄하고 있습니다.

- **Technical Details**: WildIFEval은 LLM이 복잡한 다중 제약 지침을 따르는 능력을 평가하기 위해 설계된 대규모 벤치마크입니다. 이 데이터셋은 12K의 사용자 생성 지침을 포함하고 있으며, 각 지침은 요구되는 제약 조건의 집합으로 분해되어 있습니다. 이러한 분해 과정은 LLM이 제약을 충족하는 능력을 평가하는 데 필요한 세분화된 평가를 가능하게 합니다.

- **Performance Highlights**: 실험 결과, 평가된 모든 모델은 제약 조건의 수가 증가할수록 성능이 저하되는 현상을 보였습니다. 특히, 길이 관련 제약 조건을 포함한 작업은 모든 모델에 대해 더 어려운 것으로 나타났습니다. 주목할 점은 특정 제약 유형이 모델 성능에 미치는 중요한 역할을 한다는 것입니다.



### BingoGuard: LLM Content Moderation Tools with Risk Levels (https://arxiv.org/abs/2503.06550)
Comments:
          10 pages, 4 figures, 4 tables. ICLR 2025 poster

- **What's New**: 이번 논문에서는 대규모 언어 모델(LLMs)에 의해 생성된 악성 콘텐츠의 해로움 정도를 평가할 수 있는 새로운 방법론을 제안합니다. 기존의 LLM 기반 콘텐츠 조정 시스템이 위험 수준을 제대로 평가하지 못하고 있다는 점을 지적하며, 각 주제에 따른 심각도 기준(per-topic severity rubrics)를 도입합니다.

- **Technical Details**: 연구진은 11개 위험 주제별로 심각도를 평가하기 위해 BingoGuard라는 LLM 기반 조정 시스템을 구축하였습니다. 이 시스템은 안전성 라벨 및 심각도 수준을 이진적으로 예측할 수 있으며, 저품질 응답을 걸러내는 확장 가능한 generate-then-filter 프레임워크를 제안합니다.

- **Performance Highlights**: BingoGuardTrain 데이터셋을 통해 학습한 BingoGuard-8B는 WildGuardTest 및 HarmBench를 포함한 여러 조정 벤치마크에서 최첨단 성능을 기록했습니다. 이 시스템은 기존의 최고 성능 모델인 WildGuard를 4.3% 초과하는 성과를 낼 수 있음을 보여주었으며, 심각도 수준을 훈련에 포함하는 것이 탐지 성능을 크게 향상시킨다고 분석하였습니다.



### KréyoLID From Language Identification Towards Language Mining (https://arxiv.org/abs/2503.06547)
Comments:
          8 main pages

- **What's New**: 이번 연구에서는 자동 언어 식별(Automatic Language Identification, LID)을 멀티클래스 분류 문제로 바라보던 기존의 관점을 재정립하여 이를 데이터 마이닝(Data Mining) 문제로 접근하는 새로운 시각을 제안합니다. 특히, 저자들은 프랑스어 기반 크리올 언어(Frence-based Creoles)를 위한 새롭고 효율적인 코퍼스 생성 파이프라인을 소개하며, 이를 통해 기존 대비 보다 빠르고 효율적인 자원 분배가 가능하다고 주장합니다. 이는 특히 웹 문서의 대다수가 중요하지 않은 언어에 쓰인 경우에 특히 유용하다고 설명합니다.

- **Technical Details**: 본 연구는 ‘Language Mining’이라는 개념을 제안하며, 이를 통해 대규모 웹 크롤링에서 소규모 프랑스 크리올 언어 클러스터를 효율적으로 식별하는 방법을 제시합니다. 저자들은 문서 수준의 Bag-of-Types 전략을 사용하며, 문서 필터링 시스템을 통해노이즈(web data)의 영향을 최소화하면서 정확한 언어 식별 결과를 도출하는 방법을 설명합니다. 이 시스템은 2.6억 페이지의 Common Crawl 스냅샷에서 99% 이상의 방해 문서를 제거할 수 있으며, 경쟁력 있는 회수율(recall) 유지가 가능합니다.

- **Performance Highlights**: 연구 결과, 저자들은 제안된 문서 필터링 시스템이 빠른 속도와 높은 정확도를 바탕으로 언어 식별을 수행할 수 있음을 증명하였습니다. 또한, 이 시스템은 최근 시도된 다른 필터링 시스템들보다 성능이 뛰어난 것으로 나타났습니다. 향후 2024년 12월에 Common Crawl 스냅샷에서 필터링된 프랑스어 기반 크리올 언어에 대한 추가 데이터 세트도 공개할 예정입니다.



### SafeSpeech: A Comprehensive and Interactive Tool for Analysing Sexist and Abusive Language in Conversations (https://arxiv.org/abs/2503.06534)
Comments:
          NAACL 2025 system demonstration camera-ready

- **What's New**: 이번 연구에서는 SafeSpeech라는 플랫폼을 소개하여 독립적인 메시지 수준의 분류와 대화 수준의 분석을 통합하는 도구를 제공합니다. 이 플랫폼은 유해하고 맥락에 의존하는 언어를 감지하기 위한 고급 분류기와 대규모 언어 모델(LLMs)을 통합하여 다각적인 탐지가 가능합니다. SafeSpeech는 사용자에게 문자 및 대화 수준 모두에서 독성과 관련된 컨텐츠를 시스템적으로 평가하고 벤치마크할 수 있는 새로운 기회를 제공합니다. 또한, 이 플랫폼은 모델의 예측력을 탐구하는 설명 가능성 메커니즘을 포함하여 이해를 돕습니다.

- **Technical Details**: SafeSpeech 플랫폼은 메시지 수준 분류와 대화 수준 분석의 간극을 해소합니다. 이 시스템은 세분화된 단일 메시지 분류에서 다중 순환 대화의 맥락 이해까지 지원합니다. 사용자는 맞춤 프롬프트 템플릿을 적용하여 분류 작업을 수행하고, 언어적 요소들이 예측에 미치는 영향을 강조하는 소프트하다. 커스터마이징 가능한 워크플로우와 최신 LLM 기능을 통해 분석 결과를 개선할 수 있습니다.

- **Performance Highlights**: SafeSpeech는 다양한 벤치마크 데이터셋에서 평가되어, 성차별, 모욕적 언어, 증오발언 탐지에서 최첨단 성능을 입증했습니다. 이 플랫폼은 독성이 포함된 대화 요약 및 페르소나 분석을 위한 모듈도 포함하여, 다중 턴 대화의 발화자 행동과 역학을 심층적으로 조사할 수 있도록 합니다. 따라서 SafeSpeech는 유해한 행동을 정교하게 탐지하고 분석하는 데 있어 중요한 장점을 제공합니다.



### MetaXCR: Reinforcement-Based Meta-Transfer Learning for Cross-Lingual Commonsense Reasoning (https://arxiv.org/abs/2503.06531)
- **What's New**: 이번 논문은 Cross-lingual Low-Resource Commonsense Reasoning을 위한 Multi-source adapter인 MetaXCR을 제안합니다. 기존의 CR 데이터셋이 대부분 영어로 되어 있어, 새로운 저자원(target language) 작업에는 간단히 적용하기 어렵습니다. MetaXCR는 다양한 소스 데이터셋을 활용하여 모델이 제한된 라벨 데이터를 가진 새로운 다국어 데이터셋에 적응할 수 있도록 합니다.

- **Technical Details**: MetaXCR는 메타 학습(meta learning)과 강화 학습(reinforcement learning)을 접목시킨 프레임워크로, 여러 개의 훈련 데이터셋을 통합하여 다양한 작업(task) 간의 일반화(generalization)를 배웁니다. 이 과정에서, 모델은 소스 언어의 CR 작업을 임의로 샘플링하는 강화 기반 알고리즘을 사용하여 알려진 데이터셋에 기반하여 타겟 언어의 CR 작업을 수월하게 적응할 수 있도록 합니다. 또한, 메타 업데이트(meta-update)를 통해 어댑터(adapter) 매개변수를 개선합니다.

- **Performance Highlights**: MetaXCR는 XCOPA 데이터셋에서 최첨단 성능을 기록하며, 기존의 모델들보다 적은 수의 훈련 파라미터(1.6%)로도 높은 성능을 유지합니다. 이 결과는 다양한 저자원 언어 작업을 위한 기존 CR 메타 학습 연구의 발전 가능성을 보여줍니다. 실험 결과는 MetaXCR의 효율성 및 효과성을 입증합니다.



### GFlowVLM: Enhancing Multi-step Reasoning in Vision-Language Models with Generative Flow Networks (https://arxiv.org/abs/2503.06514)
- **What's New**: 이 논문에서는 VLM(비전-언어 모델)의 한계 극복을 위한 새로운 프레임워크인 GFlowVLM을 제안합니다. 기존의 Supervised Fine-Tuning(SFT) 및 Reinforcement Learning(RL) 방법들이 가지는 한계를 인식하고, Generative Flow Networks(GFlowNets)를 활용하여 다채로운 해결책을 생성하는 구조적 사고를 지원하는 방식으로 발전했습니다. GFlowVLM은 비선형 결정을 모델링하여 복잡한 추론 작업에서 장기적 의존성을 캡쳐할 수 있습니다.

- **Technical Details**: GFlowVLM은 GFlowNets를 사용하여 비선형 차원에서 VLM을 파인 튜닝하는 방식입니다. 이는 상태 간의 논리적 의존성을 고려하여 연속적인 상태에서 구조적 추론 프로세스를 강화할 수 있도록 도와줍니다. 저자들은 GFlowNets가 제시하는 보상 함수에 기반하여 다양한 샘플링 전략을 이용해 높은 보상을 받을 수 있는 궤적을 통해 방대한 문제 공간에서의 제한적인 솔루션을 극복하고자 했습니다.

- **Performance Highlights**: GFlowVLM의 성능은 카드 게임(NumberLine, BlackJack) 및 구현 계획(ALFWorld)과 같은 복잡한 작업에서 검증되었습니다. 이 프레임워크는 기존의 SFT 및 RL 기반 방법들과 비교하여 높은 교육 효율성과 다양한 솔루션 생성 능력을 보여주며, 특히 일반화 성능이 향상되었음을 입증합니다. 세부 실험 결과는 GFlowVLM이 성공률 및 문제 해결의 다양성을 높이는 데 기여함을 분명하게 나타냅니다.



### VisualSimpleQA: A Benchmark for Decoupled Evaluation of Large Vision-Language Models in Fact-Seeking Question Answering (https://arxiv.org/abs/2503.06492)
- **What's New**: 이번 논문에서 소개하는 VisualSimpleQA는 대형 비전-언어 모델(LVLM) 평가를 위한 새로운 다중 모달 사실 탐색 벤치마크입니다. VisualSimpleQA의 특징으로는 비주얼(visual)과 언어적(linguistic) 평가를 단순화하고, 인간 주석을 안내하는 명확한 난이도 기준을 포함합니다. 또한, 이 데이터셋은 VisualSimpleQA-hard라는 도전적인 서브셋을 포함하여, 최첨단 모델들의 성능 평가에 도움을 줄 수 있습니다.

- **Technical Details**: VisualSimpleQA는 다중 모달 질문과 정답, 그리고 답변의 근거(rationale)와 텍스트 전용 질문을 포함하여 평가를 수행합니다. 이 평가 방식은 언어 모듈과 시각 모듈의 성능을 개별적으로 분석할 수 있게 해 주며, 검증된 난이도 기준은 각 샘플의 도전 수준을 정량화합니다. 또한, 데이터셋 내의 모든 샘플은 최소 1년의 경험이 있는 인간 주석자들에 의해 제작됩니다.

- **Performance Highlights**: 15개의 LVLM을 대상으로 한 실험 결과, 최신 모델인 GPT-4o도 VisualSimpleQA에서 60% 이상의 정답률을 기록했으며, VisualSimpleQA-hard에서는 30% 이상의 정답률을 보였습니다. 이는 LVLM들이 복잡한 시각 인식 작업을 처리하는 데 한계가 있음을 나타내며, 향후 개선의 여지가 크다는 점을 강조합니다. 이 연구는 LVLM의 사실 탐색 QA 능력 개선을 위한 기초 자료로 활용될 수 있습니다.



### MoFE: Mixture of Frozen Experts Architectur (https://arxiv.org/abs/2503.06491)
Comments:
          NAACL 2025 Industry

- **What's New**: 논문에서는 Parameter-efficient Fine-tuning (PEFT)과 Mixture of Experts (MoE) 아키텍처를 결합한 Mixture of Frozen Experts (MoFE) 아키텍처를 제안하고 있습니다. MoFE는 MoE 프레임워크 내에서 Feed Forward Network (FFN) 레이어를 동결하여 학습 가능한 파라미터의 수를 크게 줄이고, 이는 훈련 효율성을 높입니다. 연구 결과에 따르면, MoFE는 다른 PEFT 방법들과 비교했을 때 유의미한 효율성 향상을 보여주며, 자원이 제한된 환경에서 실용적인 솔루션이 될 수 있습니다.

- **Technical Details**: MoFE 아키텍처는 세 가지 구성 요소로 이루어져 있습니다: 기본 모델, 전문가 모델, 그리고 라우터입니다. 기본 모델은 임베딩 및 셀프 어텐션 레이어를 제공하고, 전문가 모델은 FFN 레이어를 제공합니다. 실험에서는 TinyLlama라는 11억 파라미터를 가진 사전 훈련된 모델을 기반으로 하여, FFN 블록이 동결되고 라우터와 다른 요소만 업데이트되어 학습 가능한 파라미터의 크기가 고정됩니다.

- **Performance Highlights**: 실험 결과, MoFE는 다른 PEFT 방법들과 비교하여 MMLU 및 MedMCQA 데이터셋에서 가장 좋은 성능을 기록했습니다. 또한, MoFE는 훈련 시간에서 가장 적은 자원을 요구하며, 훈련 가능한 파라미터 수는 고정되어 즉각적인 효율성을 제공합니다. MoFE는 성능이 약간 낮을 수 있지만, 자원 제약이 있는 환경에서 경쟁력 있는 선택지를 제공하고 있습니다.



### Graph Retrieval-Augmented LLM for Conversational Recommendation Systems (https://arxiv.org/abs/2503.06430)
Comments:
          Accepted by PAKDD 2025

- **What's New**: 이 논문에서는 G-CRS(Graph Retrieval-Augmented Large Language Model for Conversational Recommender Systems)라는 새로운 툴을 소개합니다. G-CRS는 훈련이 필요 없는 프레임워크로, 그래프 기반 정보 검색과 ICL(In-Context Learning)을 결합하여 LLM(대형 언어 모델)의 추천 능력을 향상시킵니다. 기존의 방법들이 가지는 도메인 지식 부족 문제를 해결하고, 의미론적 관계와 사용자 상호작용을 더 효과적으로 캡처합니다.

- **Technical Details**: G-CRS는 두 단계의 검색-추천 아키텍처를 사용하여, 첫 단계에서 GNN(그래프 신경망) 기반 추론기가 후보 아이템을 식별하고, 이후 Personalized PageRank(PPR) 알고리즘을 통해 사용자 관심에 맞는 추가 아이템을 탐색합니다. 이 과정에서 기존 대화의 이력을 활용하여 LLM이 현재 대화에서의 사용자 선호를 더 잘 이해할 수 있도록 돕습니다. 이 방법은 기존의 RAG(검색을 통한 생성) 접근법을 향상시키며, 특정 작업 훈련 없이도 효과적인 추천이 가능합니다.

- **Performance Highlights**: G-CRS는 두 개의 공개 데이터셋에서 실험을 통해 기존 방법들보다 우수한 추천 성능을 보였습니다. 추가적인 모델 훈련 없이도 G-CRS의 프레임워크는 추천 정확도를 향상시키는 데 성공하였으며, 이는 대화 기반 추천 시스템의 실용성을 크게 높이는 결과로 이어집니다. 이 연구는 대화형 추천 시스템의 발전에 기여할 뿐 아니라, 도메인 특정 지식의 필요성을 줄이는 효과를 가지고 있습니다.



### Training LLM-based Tutors to Improve Student Learning Outcomes in Dialogues (https://arxiv.org/abs/2503.06424)
- **What's New**: 이번 연구에서는 Generative AI를 활용하여 학생의 학습 결과를 극대화하기 위한 새로운 접근 방식을 제안합니다. 기존의 AI 튜터는 pedagogy(교육학적 원칙)를 따르도록 훈련되었지만, 학생의 반응을 극대화하는 데에는 한계가 있었습니다. 본 연구에서는 LLM 기반 학생 모델과 GPT-4o를 활용하여 최적의 튜터 발언을 생성하고 평가하는 방법을 개발했습니다.

- **Technical Details**: 제안된 방법은 세 가지 주요 단계를 포함합니다. 첫 번째 단계에서는 다양한 출처에서 여러 후보 튜터 발언을 생성합니다. 두 번째 단계에서는 후보 발언이 학생이 올바른 응답을 유도하는지 예측하고, pedagogy 원칙에 부합하는지를 평가합니다. 마지막으로, 좋은 후보 발언과 나쁜 후보 발언을 대조하여 Llama 3.1 8B를 Direct Preference Optimization(DPO)를 통해 미세 조정합니다.

- **Performance Highlights**: 본 연구의 결과, 생성된 튜터 발언은 학생의 올바른 응답 가능성을 상당히 높이며, 기존의 대형 LLM과 유사한 교육학적 품질을 달성했습니다. 질적 분석과 인간 평가를 통해 제안된 모델이 높은 품질의 튜터 발언을 생성하는 것을 확인하였고, 이러한 훈련 방식에서 나타나는 새로운 튜터링 전략도 밝혀냈습니다.



### How LLMs Learn: Tracing Internal Representations with Sparse Autoencoders (https://arxiv.org/abs/2503.06394)
Comments:
          Our code, demo, SAE weights are available at: this https URL

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)의 내부 표현이 훈련 과정에서 어떻게 진화하는지를 분석하여 이 모델의 다국어 능력과 개념적 이해의 발전 과정을 탐구합니다. 연구팀은 복잡한 모델의 내부 메커니즘을 이해하기 위한 결과로, LLM들이 언어별 지식과 교차언어 개념을 어떻게 독립적으로 학습하는지를 제시하며, 이는 LLM의 일반화 능력의 기초가 됩니다.

- **Technical Details**: 이 연구에서는 희소 오토인코더(sparse autoencoder, SAE)의 변형인 TopK-SAE를 사용하여 LLM의 내부 표현을 분석합니다. TopK-SAE는 은닉층에서 희소성 제약을 강제하며, 상대적으로 훈련이 용이하면서도 높은 재구성 성능을 발휘합니다. 두 개의 하이퍼파라미터(n과 K)를 조정함으로써 최적의 성능과 해석 가능성을 조화시키고 있습니다.

- **Performance Highlights**: 초기 실험에서 연구팀은 훈련된 LLM의 12번째 레이어 출력을 사용하여 SAE 하이퍼파라미터를 조정하고 결과 피처의 해석 가능성을 검증했습니다. 이 과정에서 입력 데이터의 추출과 정규화 방법을 세심하게 설정하여, 일본어와 영어를 포함한 1.7조 개의 토큰을 다루었습니다. 결과적으로 LLM이 개별 언어 지식부터 시작하여 교차 언어 매핑과 추상 개념으로 진행하는 과정을 확인했습니다.



### States of LLM-generated Texts and Phase Transitions between them (https://arxiv.org/abs/2503.06330)
Comments:
          Published as a conference paper at MathAI 2025

- **What's New**: 본 연구는 인간이 작성한 텍스트와 LLM(대형 언어 모델)에서 생성된 텍스트의 자기상관 분포가 qualitatively 다른 점을 실증적으로 보여줍니다. 특히, 텍스트 생성 시 온도(temperature) 매개변수의 변화에 따라 텍스트를 solid, critical state, gas로 분류할 수 있다는 것을 입증합니다. 이는 확률론적 자기회귀 언어 모델의 통계적 속성이 잘 이해되지 않고 있다는 점을 보완하고자 하는 시도입니다.

- **Technical Details**: 자기상관 함수(autocorrelation function)와 같은 다양한 통계적 지표를 통해 LLM이 생성한 텍스트의 세 가지 국면 - 주기적(Periodic), 임계(Critical) 및 비정형(Amorphous) - 을 정의하였습니다. 연구팀은 LLM 생성 텍스트의 임계온도에서 질서 있는 상태에서 비정형 상태로의 상전이가 발생한다고 발견했습니다. 이 과정에서 온도가 0.7에서 1.0 사이일 때 길이가 최대 2000단어에 이르는 자기상관이 전력법(Power law)으로 감소하는 현상을 확인하였습니다.

- **Performance Highlights**: Lippid와 Takahashi의 선행 연구와 비교할 때, 본 연구에서는 두 가지 최신 LLM인 Qwen2.5와 1.5B를 사용하여 보다 정확한 상전이 온도와 자기상관의 거동을 제시하였습니다. 특히, 주기적, 비정형의 상태 확립에 대한 실험 결과를 통해 LLM의 텍스트 생성 능력을 더욱 강화하였음을 알 수 있습니다. 이 결과는 기계 생성 텍스트와 인간 작성 텍스트 간의 중요한 통계적 차이를 보여주며, 각 모델이 텍스트 생성을 할 때 적용되는 매개변수가 갖는 의미에 대한 새로운 통찰을 제공합니다.



### MoEMoE: Question Guided Dense and Scalable Sparse Mixture-of-Expert for Multi-source Multi-modal Answering (https://arxiv.org/abs/2503.06296)
Comments:
          To appear at NAACL Industry Track

- **What's New**: 이 논문에서는 질문-응답 생성(QAG) 작업을 위해 다중정보 원천 및 다중 모드 데이터를 처리하는 새로운 프레임워크를 제안합니다. 기존 모델은 일반적으로 단일 정보 원천에 의존하여 언어 또는 시각 신호에서만 응답을 생성해 왔지만, 본 연구는 이러한 제한을 극복하여 여러 원천에서 정보를 수집하고 이를 통합하는 방법을 모색합니다. 제안된 질문 유도 주의(attention) 메커니즘은 다양한 출처에서 정보를 학습하고 이를 기반으로 안정적이고 편향 없는 응답을 생성할 수 있습니다.

- **Technical Details**: 모델은 질문에 따라 정보 출처에서 주의(attention) 패턴을 자동으로 인식하고, 이를 통해 각 출처에서 질문에 적합한 정보의 주의를 분산합니다. 또한, 질문, 컨텍스트, 이미지를 별도로 임베딩하여 질문-이미지 및 질문-컨텍스트 쌍의 상관관계를 최대화함으로써 각 출처 내에서 적절한 주의 패턴을 학습합니다. 단일 모델이 다양한 질문 유형을 처리하는 것이 어렵기 때문에, 우리는 전문가 혼합(sparse MoE) 모델을 도입하여 전문가들이 특정 질문 유형에 대한 전문성을 가지도록 했습니다.

- **Performance Highlights**: 실험에서는 T5 및 Flan-T5 모델을 활용한 세 가지 데이터셋에서 우수한 성능을 보였습니다. 제안된 방법은 속성 기반 응답 생성을 위한 새로운 기준 성능을 확립했으며, 각 모델 구성 요소의 기여를 분석하기 위한 제거(ablation) 연구도 진행되었습니다. 이러한 연구 결과는 제출된 방식이 연관된 문제에 효과적으로 기여한다는 것을 입증합니다.



### IteRABRe: Iterative Recovery-Aided Block Reduction (https://arxiv.org/abs/2503.06291)
Comments:
          8 pages

- **What's New**: 본 논문에서는 IteRABRe라는 새로운 반복적 차단 가지치기(iterative pruning) 방법을 소개합니다. 이 방법은 모델 압축을 효과적으로 수행하면서도 최소한의 컴퓨팅 자원만 요구합니다. 특히 2.5M 토큰만으로 성능 복구를 달성하고 Llama3.1-8B 및 Qwen2.5-7B 모델에서 평균 3% 향상된 성능을 보여줍니다.

- **Technical Details**: IteRABRe는 대형 언어 모델의 모든 층을 반복적으로 가지치기 및 복구하는 과정을 통해 특정 목표 모델 크기에 도달하는 프레임워크입니다. 본 방법은 계층 단위의 가지치기를 직접적으로 수행하고, 과도한 데이터 요구 없이도 경쟁력 있는 성능을 달성하는 효율적인 복구 과정을 포함합니다. 손실 최소화를 위해 모델 품질에 대한 각 계층의 기여도를 평가합니다.

- **Performance Highlights**: IteRABRe는 언어 관련 작업에서 기존 기준보다 5% 더 나은 성능을 보여 언어적 능력 보존에 특히 강점을 나타냅니다. 이 방법은 또한 영어 데이터만을 사용하여 독일어와 같은 다국어 능력도 유지하는 제로샷(zero-shot) 크로스링구얼(cross-lingual) 기능을 보여줍니다. 전체적으로 IteRABRe는 다양한 LLM 모델에서 원하는 크기의 압축을 달성하면서도 성능 손실을 최소화하는 혁신적인 접근 방식입니다.



### Integrating Chain-of-Thought for Multimodal Alignment: A Study on 3D Vision-Language Learning (https://arxiv.org/abs/2503.06232)
- **What's New**: 본 연구에서는 Chain-of-Thought (CoT) 추론을 3D 비전-언어 학습에 통합하여 구조화된 추론을 조정 훈련 과정에 포함시키는 새로운 접근 방식을 제안합니다. 특히, 3D-CoT 벤치마크라는 데이터셋을 만들어 형상 인식, 기능 유추, 인과 추론에 대한 계층적 CoT 주석을 포함하였습니다. 이러한 연구는 CoT가 다중 모달 추론을 크게 향상시킬 수 있음을 시사합니다.

- **Technical Details**: 3D-CoT 벤치마크는 기존의 3D 데이터셋에 계층적 추론 주석을 추가하여 3D 비전-언어 정렬에 대한 새로운 기준을 제공합니다. 본 연구에서는 큰 추론 모델(LRMs)과 일반-purpose 언어 모델(LLMs) 간의 성능 차이를 분석하기 위해 표준 텍스트 주석과 CoT-구조화 주석을 비교하는 제어 실험을 수행했습니다. 이 평가 방법론은 intermediate reasoning 품질과 최종 추론의 정확성을 개별적으로 측정합니다.

- **Performance Highlights**: 실험 결과, CoT-구조화 주석이 3D 추론을 상당히 개선하는 것으로 나타났습니다. CoT를 통해 훈련된 모델은 텍스트 설명과 3D 구조 간의 정렬이 향상되었으며, 특히 객체의 affordance 인식과 상호작용 예측에서 두드러진 성과를 보였습니다. LRM은 CoT를 더 효과적으로 활용하여 구조화된 추론에서 보다 큰 이점을 얻는 것으로 나타났습니다.



### KnowLogic: A Benchmark for Commonsense Reasoning via Knowledge-Driven Data Synthesis (https://arxiv.org/abs/2503.06218)
- **What's New**: 이번 논문에서는 구조화된 주석이 부족했던 commonsense reasoning 평가의 한계를 극복하기 위해 KnowLogic이라는 벤치마크를 소개합니다. KnowLogic은 다양한 commonsense 지식과 합리적인 시나리오를 통합하여 현존하는 LLMs에 대한 주요 과제를 제공합니다. 이 벤치마크는 3,000개의 이중언어 질문(중국어 및 영어)으로 구성되어 있으며, 난이도 수준을 조정할 수 있는 장점을 가지고 있습니다.

- **Technical Details**: KnowLogic은 개체(entities), 명제(propositions), 시나리오(scenarios)의 세 가지 핵심 개념으로 구성된 지식 기반을 기반으로 합니다. 각 개체는 고유의 속성을 가지고 있으며, 명제는 개체의 속성과 상호 관계를 정의합니다. 시나리오는 commonsense reasoning에 필요한 맥락을 제공하여,상황 안에서 올바른 정보를 추론할 수 있도록 돕습니다.

- **Performance Highlights**: KnowLogic 데이터셋은 3000개의 질문으로 구성되어 있으며, 현재 최고 성능을 기록한 모델은 단지 69.57%의 정확도를 달성했습니다. 평가 결과, LLMs의 commonsense reasoning에서의 주요 문제점을 식별했으며, 이는 저빈도 commonsense에 대한 오해와 논리적 일관성의 부족 등입니다. 이 연구는 LLMs의 commonsense reasoning 능력을 평가하고 향상시키는 데 중요한 도구를 제공하고 있습니다.



### Text-Speech Language Models with Improved Cross-Modal Transfer by Aligning Abstraction Levels (https://arxiv.org/abs/2503.06211)
- **What's New**: 이 논문은 Text-Speech Language Models (TSLMs)의 훈련 방법에 변화를 제안합니다. 기존의 방법은 사전 훈련된 텍스트 LM의 어휘에 새로운 임베딩을 추가하는 방식이었지만, 이로 인해 크로스 모달 전이가 제한된다고 주장합니다. 새로운 기법인 	extsc{SmolTolk}를 통해, 모델의 다양한 레이어에서 추상화 수준을 보다 잘 정렬할 수 있도록 모듈을 추가하는 방식으로 개선했습니다.

- **Technical Details**: 제안된 	extsc{SmolTolk} 모델은 135백만에서 17억 개의 파라미터를 가진 SmolLM 모델 계열에 적용됩니다. 새로운 입력 및 출력에 대해 음성 특화 레이어를 추가하고, 학습 가능한 동적 레이어 풀링 메커니즘을 도입하여 텍스트 LM의 출력 표현이 저수준 및 다음 단어 예측 표현에 적절히 전환할 수 있도록 설계했습니다. 이러한 구조적 변화는 기존의 TSLM 방법론에 비해 크로스 모달 전이를 촉진합니다.

- **Performance Highlights**: 실험 결과, 	extsc{SmolTolk}는 일반적인 어휘 확장 방법으로 훈련된 기준 모델들을 지속적으로 초월하는 성능을 보였습니다. 가장 큰 모델인 SmolTolk-2B는 수십 배 더 큰 TSLM 모델과 경쟁하거나 그 이상으로 성능을 달성했습니다. 다양한 아블레이션 및 표현 분석을 통해 제안된 각 요소가 전반적인 성능 향상에 기여함을 확인했습니다.



### CUPCase: Clinically Uncommon Patient Cases and Diagnoses Datas (https://arxiv.org/abs/2503.06204)
Comments:
          Accepted to AAAI 2025

- **What's New**: 이 논문에서는 Clinically Uncommon Patient Cases and Diagnosis Dataset (CUPCase)를 구축하여, LLMs (Large Language Models)의 진단 능력을 평가했습니다. CUPCase는 3,562개의 실제 환자 사례를 기반으로 하여, 병의 진단 정보를 가진 개방형 텍스트와 선택형 질문을 포함합니다. 이 데이터셋은 기존의 의료 영역 기준을 보완하여, 보다 다양한 임상 사례를 제시합니다.

- **Technical Details**: CUPCase 데이터셋은 BMC에서 제공하는 의료 사례 보고서에서 추출한 데이터로 구성됩니다. 연구에서는 두 가지 과제를 통해 LLMs의 성능을 평가했습니다: 첫 번째는 선택형 질문(multiple-choice question) 평가, 두 번째는 개방형 질문(open-ended question) 작성입니다. GPT-4o는 두 평가 모두에서 뛰어난 성능을 보여주었으며, 특히 87.9%의 정확도를 달성했습니다.

- **Performance Highlights**: 연구 결과, GPT-4o는 선택형 질문에서 평균 87.9%의 정확도를, 개방형 질문에서는 BERTScore F1이 0.764으로 측정되었습니다. 또한, GPT-4o는 전체 임상 정보의 20%만으로도 87%와 88%의 성능을 유지할 수 있음을 나타내어, 실제 임상 사례에서 조기 진단을 지원할 수 있는 가능성을 강조했습니다. CUPCase는 LLMs의 임상 의사결정 지원 능력을 공개적이고 재현 가능한 방식으로 평가할 수 있도록 합니다.



### Sample-aware Adaptive Structured Pruning for Large Language Models (https://arxiv.org/abs/2503.06184)
- **What's New**: 이 논문에서는 LLMs(대형 언어 모델)의 구조적 가지치기(strcutured pruning) 기법에 대한 새로운 접근법인 AdaPruner를 소개합니다. AdaPruner는 기존의 무작위 선택 방식 대신 샘플 인식(sample-aware) 기반으로 최적의 보정 데이터(calibration data)와 중요도 추정 지표(importance estimation metrics)를 동시에 최적화하는 점이 핵심입니다. 이를 통하여 LLMs의 비효율적인 파라미터를 효과적으로 제거함으로써 성능을 극대화하는 방법을 제안합니다.

- **Technical Details**: AdaPruner는 베이지안 최적화(Bayesian optimization)를 활용하여, 보정 데이터와 중요도 추정 지표를 동시적으로 탐색하는 구조로 되어 있습니다. 첫째, AdaPruner는 구조적인 가지치기와 관련된 데이터와 메트릭의 하위 공간을 생성하여 상호 의존성을 고려합니다. 둘째, Taylor 확장(Taylor expansion)을 기반으로 파라미터 제거가 손실에 미치는 영향을 정량화하여, 효과적인 구조 제거를 가능하게 합니다.

- **Performance Highlights**: 실험 결과에 따르면, AdaPruner는 다양한 가지치기 비율에서 기존의 구조적 가지치기 방법들보다 우수한 성능을 보여주었습니다. 특히, 20% 가지치기에서 AdaPruner로 가지치기한 모델은 비가지치기 모델의 97%의 성능을 유지했습니다. 이는 LLaMA 시리즈 모델에서 기존의 LLM-Pruner보다 평균 1.37% 높은 성능을 달성함으로써 효과성과 강건성을 입증했습니다.



### GRP: Goal-Reversed Prompting for Zero-Shot Evaluation with LLMs (https://arxiv.org/abs/2503.06139)
Comments:
          Ongoing Work

- **What's New**: 이 논문에서는 Goal-Reversed Prompting (GRP)이라는 새로운 접근 방식을 제안합니다. 기존의 방법들이 더 나은 답변을 선택하도록 요구하는 것과 달리, GRP는 더 나쁜 답변을 선택하게끔 LLM(대형 언어 모델)에게 유도합니다. 이를 통해 LLM이 역으로 사고하도록 하여 평가 능력을 향상시키고자 합니다.

- **Technical Details**: 논문에서는 LLM 기반의 평가는 일반적으로 더 나은 답을 평가하는 것으로 이뤄지지만, GRP를 통해 이 과제를 더 나쁜 답을 선택하도록 변경합니다. Chain-of-Thought (CoT) 패러다임에 기반 경쟁력 있는 평가 방법인 SOP(Standard Operating Procedures)의 장점을 활용하여 더 나은 성과를 도출할 수 있습니다. 이는 약한 응답을 식별함으로써 전반적인 평가 메커니즘을 개선합니다.

- **Performance Highlights**: 실험 결과, GRP를 도입한 평가 방식이 기존의 목표를 가진 프롬프트에 비해 성능이 개선되었음을 확인했습니다. 특히, LLM의 역 목표 프롬프트를 사용함으로써 GPT-4o와 Claude-3.5-Sonnet 모두에서 평가 정확도가 유의미하게 향상되었습니다. 이러한 결과는 GRP가 LLM의 평가 능력을 대폭 개선하는 데 기여함을 시사합니다.



### Evaluating Discourse Cohesion in Pre-trained Language Models (https://arxiv.org/abs/2503.06137)
- **What's New**: 이 논문은 사전 훈련된 언어 모델의 응집력(cohesive ability)을 평가하기 위해 새로운 테스트 세트를 제안합니다. 다양한 문장 간의 응집 현상을 포함하여 모델의 성능을 비교하고 분석하고자 합니다. 또한, 담화 응집(discourse cohesion)에 대한 관심을 높이기 위해 이 연구가 시작되었습니다.

- **Technical Details**: 이 연구에서는 Halliday의 응집 개념을 채택하여 참조(reference), 대체(substitution), 생략(ellipsis), 접속(conjunction), 어휘 응집(lexical cohesion) 등을 포함한 다섯 가지 주요 응집 유형을 분석합니다. 테스트 세트는 총 1554개의 응집 예시를 포함하며, 인접한 문장과 비인접 문장 사이의 관계를 모두 고려합니다. 주어진 데이터는 'ROC stories corpus'에서 발췌되었으며, 일상적인 사건 간의 관계를 잘 포착한 고품질의 이야기입니다.

- **Performance Highlights**: 사전 훈련된 언어 모델 BERT, BART, RoBERTa를 이용한 실험에서 모델의 다수의 응집 현상에 대한 성능을 평가하며, 이 모델들이 담화의 복잡성을 얼마나 잘 이해하고 생성할 수 있는지를 분석합니다. 결과적으로, 기존의 연구와 비교해 보다 종합적인 담화 응집 능력을 가진 모델들이 개발될 수 있음을 제안하고 있습니다.



### Theta Theory: operads and coloring (https://arxiv.org/abs/2503.06091)
Comments:
          26 pages LaTeX

- **What's New**: 이 논문에서는 생성 언어학의 최소주의 모델에서 세타 이론(theta theory)을 구현하는 colored operad의 생성 집합을 명시적으로 구성하는 방법을 제시합니다. 이는 구문 객체(syntactic objects)에 대한 색상 알고리즘(coloring algorithm)의 형태로 나타납니다. 또한 작업 공간(workspaces)에서의 coproduct operation이 세타 기준(theta criterion)의 재귀적 구현을 가능하게 한다고 설명합니다.

- **Technical Details**: 구조가 Merge에 의해 자유롭게 형성될 때 색상 규칙(coloring rules)으로 필터링하는 방법이 하프한 Merge의 색상 버전(더블 Merge)으로 구조 형성 구조의 과정과 동등하다는 것을 보여줍니다. 이를 통해 colored operad의 생성자의 형태는 External Merge와 Internal Merge 간의 의미론적(semantics) 이분법(dichotomy)을 의미하게 됩니다. 특히 Internal Merge는 비 세타 위치(non-theta positions)로만 움직입니다.

- **Performance Highlights**: 이 연구 결과는 구문 결합(syntactic combination)의 색상화(coloring) 방법론을 통해 이론을 더욱 명확하게 할 수 있는 가능성을 보여줍니다. 색상화된 구조는 다양한 언어 현상(linguistic phenomena)을 설명하는 데 있어 중요한 도구가 될 수 있습니다. 이로 인해 최소주의 최소 구조 이론(minimalist theory of structures)에 대한 새로운 통찰을 제공합니다.



### Multi-Attribute Multi-Grained Adaptation of Pre-Trained Language Models for Text Understanding from Bayesian Perspectiv (https://arxiv.org/abs/2503.06085)
Comments:
          Extended version accepted by AAAI 2025

- **What's New**: 이번 연구는 비독립적이고 동일 분포가 아닌(non-IID) 정보를 어떻게 다루는지에 대한 새로운 통찰력을 제공합니다. 이를 통해, 사전 학습된 언어 모델(PLMs)의 성능 향상을 위한 베이지안 관점으로 접근하고, 다중 특성과 다중 세분화된(multi-grained) 프레임워크인 M2A를 제안합니다. M2A는 경량화된 방식으로 PLM의 불확실성을 완화하고, 다양한 데이터 소스로부터 수집된 복잡한 특성을 효과적으로 통합합니다.

- **Technical Details**: M2A 프레임워크는 다중 특성과 세분화된 뷰를 통합하여 효과적인 PLM 적응을 가능하게 합니다. 베이지안 추론을 통해 두 가지 특성, 즉 비독립적이고 동일 분포가 아닌(non-IID)과 IID 정보를 연결하고, 이들의 관계를 분석합니다. 이러한 방식은 PLM의 적응성을 증가시키고 데이터 이질성 문제를 해결하는 데 중점을 둡니다.

- **Performance Highlights**: 다양한 텍스트 이해 데이터셋을 활용하여 M2A의 성능을 평가한 결과, 특히 데이터가 암묵적으로 비독립적이고 동일 분포가 아닌(non-IID) 경우에 우수한 성능을 보였습니다. 실험을 통해 M2A는 다양한 PLM과 함께 활용될 때 뛰어난 적응 능력을 보여주며, 이는 이전의 방법들과 비교하여 큰 개선점으로 작용합니다.



### An Empirical Study of Causal Relation Extraction Transfer: Design and Data (https://arxiv.org/abs/2503.06076)
- **What's New**: 이 논문에서는 인과 관계 추출(causal relation extraction)을 위한 신경망 아키텍처와 데이터 전송 전략(data transfer strategies)의 실증 분석을 수행합니다. 실험을 통해 BioBERT-BiGRU 모델이 다양한 웹 기반 소스와 주석 전략에 대해 다른 아키텍처보다 더 나은 일반화를 보임을 입증하였습니다. 또한 명사구(localization) 강조된 새로운 성능 평가 지표인 $F1_{phrase}$를 소개하여, 데이터 전송 실험에서 도메인과 주석 스타일에 관계없이 성능 향상을 확인하였습니다.

- **Technical Details**: 기술적으로, 다양한 컨텍스트 임베딩 층(contextual embedding layers)과 아키텍처 요소를 실험하여, BioBERT와 RoBERTa와 같은 Transformer 기반의 변환기 컨텍스트 단어 임베딩이 가장 효과적임을 보여줍니다. 또한, BiLSTM과 CRF와 같은 일반적인 관계 추출 아키텍처가 열려 있는 도메인에서의 인과 관계 추출 모델에 필요하지 않다는 것을 발견하였습니다. 이 연구는 명시적 및 암시적인 근거의 관계를 추출하는 데 초점을 맞추며, 데이터 구조와 훈련 크기가 성능에 미치는 영향을 통해 인과 데이터에서 빠르고 효율적으로 작업할 수 있는 방법을 제시하고 있습니다.

- **Performance Highlights**: 성능 하이라이트로는 BioBERT-BiGRU 모델이 다양하고 노이즈가 많은 웹 소스를 통해 인과 관계를 추출하는 데 뛰어난 효율성을 발휘하는 점이 있습니다. F1_{phrase} 메트릭의 도입으로 인해 데이터 증강이 성능을 향상시키는 데 기여하며, 특히 암시적 인과 문장의 비율이 중요하다는 점이 강조됩니다. 이 연구는 궁극적으로 노이즈가 많은 원천으로부터 인과 지식을 더 풍부하게 추출하는 방안으로 작용할 수 있다는 것을 보여줍니다.



### Towards Conversational AI for Disease Managemen (https://arxiv.org/abs/2503.06074)
Comments:
          62 pages, 7 figures in main text, 36 figures in appendix

- **What's New**: 이번 연구에서는 Articulate Medical Intelligence Explorer (AMIE)를 활용한 새로운 LLM 기반 시스템을 소개하며, 이 시스템은 기존의 진단 능력을 넘어 질병 관리(Clinical Management)와 대화(Dialogue)를 최적화하는데 중점을 두고 있다는 점이 눈에 띕니다. AMIE는 여러 환자 방문 사례와 치료 반응에 대한 추론을 포함하며, 전문적인 약물 처방 능력을 강화합니다. 또한, AMIE는 Gemini의 긴 맥락(Long Context) 능력을 활용해 최신 임상 지침과 약물 포뮬러리에 기반한 합리적인 추론을 제공합니다.

- **Technical Details**: AMIE는 구조적 추론(Structured Reasoning)과 인-context 검색(In-Context Retrieval)을 결합하여 의료 지식의 권위에 기반한 해결책을 도출합니다. 연구에서는 21명의 일반 진료 의사(Primary Care Physicians, PCPs)와 비교하여 AMIE의 관리 추론 능력을 평가하였고, 이는 영국 NICE 지침과 BMJ Best Practice 가이드를 반영한 100개의 다중 방문 사례로 구성되었습니다. 이 과정에서 전문 의사들이 평가한 결과, AMIE는 PCP들에게 뒤지지 않는 성능을 기록하였습니다.

- **Performance Highlights**: AMIE는 진단의 정밀성과 조사에 있어 PCP들보다 더 높은 점수를 받았으며, 임상 지침에 따른 관리 계획의 적절성 면에서도 우수한 성과를 보였습니다. 추가로, 약물 추론을 벤치마킹하기 위해 제작된 RxQA를 통해 AMIE는 어려운 질문에 대한 정확도에서 PCP들을 초월하는 결과를 나타냈습니다. 비록 실세계 적용을 위한 추가 연구가 필요하지만, AMIE의 강력한 성능은 질병 관리에서 대화형 AI의 중요한 진전을 의미합니다.



### GEM: Empowering MLLM for Grounded ECG Understanding with Time Series and Images (https://arxiv.org/abs/2503.06073)
- **What's New**: 최근 다중 모달 대형 언어 모델(MLLMs)은 자동화된 ECG 해석에서 발전을 이루었지만, 여전히 두 가지 주요 한계에 직면해 있습니다. 첫째, 시계열 신호와 시각적 ECG 표현 간의 결합이 불충분하며, 둘째, 진단을 미세한 파형 증거에 연결하는 데 한계가 있습니다. 이를 해결하기 위해 GEM을 도입하여 시계열 데이터, 12리드 ECG 이미지 및 텍스트를 통합한 최초의 MLLM을 제안합니다.

- **Technical Details**: GEM은 이중 인코더 프레임워크를 사용하여 시계열과 이미지 특성을 보완적으로 추출하고, 크로스모달 정렬(Cross-modal Alignment)을 통해 효과적인 다중 모달 이해를 가능하게 합니다. 또한, 지식 기반의 지침 생성을 통해 고해상도의 그라운딩 데이터를 생성하여 진단을 측정 가능한 매개변수와 연계합니다. 이러한 구조는 GEM이 임상과 유사한 진단 과정을 시뮬레이션하도록 돕습니다.

- **Performance Highlights**: GEM은 기존 및 제안된 벤치마크에서 실험적으로 예측 성능을 7.4% 향상시키고, 설명 가능성은 22.7% 증가시키며, 그라운딩 성능을 24.8% 개선하였습니다. 이러한 성과는 GEM이 실제 임상 환경에서 더 적합하고 믿을 수 있는 진단 도구로 자리 잡을 수 있도록 합니다.



### A Survey on Post-training of Large Language Models (https://arxiv.org/abs/2503.06072)
Comments:
          87 pages, 21 figures, 9 tables

- **What's New**: 이 논문은 Post-training Language Models (PoLMs)에 대한 최초의 포괄적 설문조사를 제공하며, 모델의 발전과 문제점들을 다섯 가지 핵심 패러다임으로 체계적으로 분석합니다. 이러한 패러다임에는 파인튜닝(Fine-tuning), 정렬(Alignment), 추론(Reasoning), 효율성(Efficiency), 통합 및 적응(Integration and Adaptation)이 포함됩니다. PoLM의 진화를 통해 데이터셋 활용 방식 및 도메인 적응력 향상에 대한 기여를 설명합니다.

- **Technical Details**: 이 연구에서는 조건에 맞는 대화 시스템에서부터 과학적 탐색에 이르기까지 다양한 분야에 걸쳐 LLM의 발전을 다루고 있습니다. PoLM은 GPT-4와 DeepSeek-R1를 포함하여 고도화된 모델 성능을 획득하는 데 중점을 둡니다. 이 과정에서 모델의 특정 작업에 대한 조정 및 사용자 요구 사항에 대한 적절한 대응 방안을 설정합니다.

- **Performance Highlights**: PoLM의 발전은 특정 작업과 요건에 대한 적응 능력을 향상시키는데 크게 기여했습니다. 예를 들어, DeepSeek-R1은 추론 능력을 강화하고 사용자 선호에 맞춘 정렬 및 도메인 적응성을 개선했습니다. 앞으로의 연구 방향은 모델의 정밀도 및 신뢰성을 향상시키는 것으로, LRM의 성장과 함께 더욱 민감한 기술 개발이 이루어질 것입니다.



### Fine-Grained Bias Detection in LLM: Enhancing detection mechanisms for nuanced biases (https://arxiv.org/abs/2503.06054)
Comments:
          Bias detection, Large Language Models, nuanced biases, fine-grained mechanisms, model transparency, ethical AI

- **What's New**: 최근 인공지능의 발전, 특히 Large Language Models (LLMs)에서의 향상은 자연어 처리를 혁신적으로 변화시켰습니다. 그러나 이러한 모델에 내재된 편향을 감지하는 것은 여전히 도전 과제가 되고 있습니다. 본 연구는 LLM에서 섬세한 편향을 식별하기 위한 새로운 탐지 프레임워크를 제시하여 윤리적 우려를 해소하고자 합니다.

- **Technical Details**: 이 접근 방식은 문맥 분석(contextual analysis), 주의 메커니즘(attention mechanisms)을 통한 해석 가능성, 그리고 반사실 데이터 증강(counterfactual data augmentation)을 통합하여 언어적 맥락에 숨겨진 편향들을 캡쳐합니다. 비교적(proven) 프롬프트와 합성 데이터셋(synthetic datasets)을 사용해 모델의 행동을 문화적, 이념적, 인구통계적 시나리오에 따라 분석합니다. 이는 벤치마크 데이터셋을 통한 정량적 분석과 전문가 리뷰를 통한 정성적 평가로 효과성을 검증합니다.

- **Performance Highlights**: 연구 결과는 인종, 성별, 사회정치적 맥락에 대한 모델 응답의 차이를 강조하지 못하는 기존 방법에 비해 섬세한 편향을 탐지하는 데 있어 개선된 성능을 보여줍니다. 또한, 훈련 데이터와 모델 아키텍처의 불균형에서 발생하는 편향을 식별하며, 지속적인 사용자 피드백을 통해 적응성과 정교함을 보장합니다. 이는 교육, 법률 시스템 및 의료와 같은 민감한 응용 분야에서의 책임 있는 LLM 배포를 지원하고 향후 실시간 편향 모니터링과 교차 언어 일반화에 중점을 둡니다.



### Constructions are Revealed in Word Distributions (https://arxiv.org/abs/2503.06048)
- **What's New**: 이번 연구에서는 RoBERTa 언어 모델을 사용하여 통계적 친화력(statistical affinity) 패턴으로서의 구성(constructions) 인식을 탐구합니다. 기존 연구들과는 달리, PLM을 학습자의 역할에서 분포의 시뮬레이션으로 사용하여 구성 학습 가능성을 검토합니다. 기존의 접근 방식에서 통계적 친화력만으로는 모든 구성을 식별하는 데 한계가 있음을 보여줍니다.

- **Technical Details**: 이 연구는 RoBERTa 언어 모델을 활용하여 전역 친화력(global affinity)과 지역 친화력(local affinity)을 비교하는 두 가지 방법을 개발하였습니다. 이를 통해 구성 간의 통계적 관계를 분석하며, 텍스트의 맥락에 따라 출력 분포의 변화를 관찰합니다. 이 방법론은 구성이 나타내는 통계적 패턴의 본질을 해명하는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: 실험 결과, 다양한 종류의 구성을 robust하게 구분할 수 있음을 보여주었습니다. 특히, 외관상 유사하지만 의미적으로 다른 구성 사례를 처리할 수 있는 능력도 확인되었습니다. 이전 연구에서의 실패 사례들에서도 이러한 접근법을 사용하여 구성을 효과적으로 회복할 수 있음을 입증했습니다.



### Mitigating Memorization in LLMs using Activation Steering (https://arxiv.org/abs/2503.06040)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 메모리화(memorization) 문제를 완화하기 위해 새로운 방법인 Activation Steering을 사용했습니다. 이번 접근 방식은 모델의 활성화를 조작하여 메모리화된 내용을 억제하면서 일반화(generalization) 능력을 유지하는 것을 목표로 합니다. 실험 결과, 이 방법은 모델의 성능 저하를 최소화하면서 기억된 내용을 효과적으로 억제하는 데 성공했습니다.

- **Technical Details**: 연구에서는 Activation Steering 기법을 통해 모델의 활성화를 직접 조정함으로써 기억화된 내용을 효과적으로 제거할 수 있는 가능성을 탐구했습니다. 다양한 요인과 구성에서의 정량적 실험이 이루어졌고, 메모리화 저감, 언어 능력 및 일반 능력 간의 영향을 분석했습니다. 이를 통해 Activation Steering을 효과적으로 적용하기 위한 최적의 방법론을 도출하였습니다.

- **Performance Highlights**: 연구 결과, Activation Steering은 LLMs의 메모리화를 효과적으로 억제하며, 모델의 전체적인 성능을 거의 저하시키지 않는 것으로 나타났습니다. 특히, 프롬프트(promise)에 따라 다른 벤치마크에서 다양한 성능을 평가하여 일반화된 메모리화 완화 방법을 마련했습니다. 이 연구는 더 안전하고 프라이버시를 중시하는 LLM 개발에 기여할 수 있는 실용적이고 효율적인 메커니즘을 제시하고 있습니다.



### SmartBench: Is Your LLM Truly a Good Chinese Smartphone Assistant? (https://arxiv.org/abs/2503.06029)
Comments:
          23 pages

- **What's New**: 최근 large language models (LLMs)가 스마트폰의 지능형 비서로 자리잡으면서, on-device LLM의 역량을 평가하기 위한 새로운 표준이 필요해졌습니다. 기존의 벤치마크는 주로 수학이나 코딩처럼 객관적인 과제에 집중하고 있습니다. 그러나 이러한 평가가 실제 모바일 상황에서의 on-device LLM의 활용도를 반영하지 못하는 문제를 해결하기 위해 SmartBench를 제안합니다. SmartBench는 모바일 환경에서의 on-device LLM의 기능을 평가하기 위해 특별히 설계된 첫 번째 중국어 벤치마크입니다.

- **Technical Details**: SmartBench는 애플, 화웨이, 오포, 비보, 샤오미와 같은 대표적인 스마트폰 제조업체가 제공하는 on-device LLM 기능을 분석하여 다섯 가지 카테고리로 나눕니다: 텍스트 요약, 텍스트 Q&A, 정보 추출, 콘텐츠 생성 및 알림 관리입니다. 각 카테고리는 구체적인 20개 작업으로 세분화되어 있으며, 각 작업에 대해 50에서 200개의 질문-답변 쌍을 포함하는 고품질 데이터셋이 구성됩니다. SmartBench에서는 각 카테고리/task에 특화된 자동 평가 기준도 개발하였습니다.

- **Performance Highlights**: SmartBench를 사용하여 여러 on-device LLM과 MLLM의 포괄적인 평가를 수행했으며, 실제 스마트폰의 NPU에서 양자화된 배치를 통한 성능 평가도 실시하였습니다. 이 연구는 중국어를 사용하는 스마트폰에 최적화된 on-device LLM 평가를 위한 표준화된 프레임워크를 제공하며, 이 분야의 발전과 최적화를 촉진하는 기초 자료로 활용될 수 있습니다. 코드와 데이터는 해당 URL에서 확인할 수 있습니다.



### GenieBlue: Integrating both Linguistic and Multimodal Capabilities for Large Language Models on Mobile Devices (https://arxiv.org/abs/2503.06019)
Comments:
          14 pages

- **What's New**: 최근 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 발전은 모바일 장치에서의 배치를 가능하게 했습니다. 그러나 언어 능력 유지와 하드웨어 호환성이라는 문제는 여전히 남아 있습니다. 본 연구에서는 GenieBlue라는 구조적 설계를 제안하여, 언어 능력과 멀티모달 능력을 통합하여 모바일 디바이스에서 효율적으로 작동할 수 있도록 합니다.

- **Technical Details**: GenieBlue는 MLLM 훈련 중 원래 대형 언어 모델(LLM) 파라미터를 동결하여 언어 능력을 유지합니다. 특정 transformer 블록을 복제하고 경량 LoRA(Low-Rank Adaptation) 모듈을 추가하여, 멀티모달 능력을 확보합니다. 이러한 접근 방식은 효율적인 고유 언어 능력 보존을 통해, 높은 훈련량으로도 경쟁력 있는 멀티모달 성능을 제공합니다.

- **Performance Highlights**: GenieBlue는 실제 스마트폰의 NPU에서 배치되며, 모바일 디바이스에 적합한 효율성과 실용성을 입증합니다. 다양한 MLLM에서 나타나는 성능 저하 문제를 분석하고, GenieBlue에 의해 직접적인 언어 작업 성능 감소를 예방합니다. 이 연구는 모델 구조 설계와 훈련 데이터에 대한 분석을 통해, MLLMs의 순수 언어 성능 유지 방법을 구체적으로 설명합니다.



### Intent-Aware Self-Correction for Mitigating Social Biases in Large Language Models (https://arxiv.org/abs/2503.06011)
Comments:
          18 pages. Under review

- **What's New**: 이 연구에서는 Large Language Models (LLMs)의 출력 품질을 개선하기 위해 Feedback을 기반으로 하는 Self-Correction 기법을 제시합니다. Self-Correction은 인지 심리학의 System-2 사고와 유사하게 작용하여 사회적 편견을 줄일 가능성을 지니고 있습니다. 각각의 구성 요소인 instruction, response, feedback에서 의도를 분명히 이해하고 반영하는 것이 중요하다는 것을 보여줍니다.

- **Technical Details**: Self-Correction은 세 가지 주요 단계로 이루어져 있으며, 초기 응답 생성, 피드백 생성 및 정제 단계를 포함한 iterative한 과정입니다. 응답 생성 과정에서는 편향을 줄이기 위한 명시적인 프롬프트와 Chain-of-Thought (CoT) 접근을 통해 사고 과정을 명확히 합니다. 피드백 생성 과정에서는 적합한 평가 기준을 정의하고, 각 기준에 대해 점수를 부여하여 보다 명확한 피드백을 제공합니다.

- **Performance Highlights**: 실험을 통해 Self-Correction 프레임워크가 bias를 줄이는 데 있어 기존 방법보다 보다 일관되고 강력한 효과를 나타냄을 입증하였습니다. Cross-model correction을 통해 높은 편향 모델에 대한 편향 완화 능력이 향상되었으며, Feedback의 출처와 생성자가 피드백 품질에 미치는 영향을 분석하였습니다. 또한, Refinement 품질은 Feedback 생성자에 의해 크게 좌우되는 것으로 확인되었습니다.



### SINdex: Semantic INconsistency Index for Hallucination Detection in LLMs (https://arxiv.org/abs/2503.05980)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)이 생성한 사실적으로 부정확한 출력을 탐지하기 위한 새로운 자동화된 프레임워크를 소개합니다. 제안된 방법은 문장 임베딩과 계층적 클러스터링을 기반으로 하고, 새로운 불일치 측정인 SINdex를 활용하여 LLM의 환각 현상을 보다 정확하게 감지하도록 설계되었습니다. 미셀린세서리 기술 없이도 다양한 LLM에 적용할 수 있는 블랙박스 프레임워크로서, 기존 기술에 비해 최대 9.3%의 AUROC 개선을 보여주었습니다.

- **Technical Details**: 제안된 방법론은 두 가지 주요 구성 요소로 이루어져 있습니다. 첫째, 의미가 유사한 출력들을 클러스터링하는 의미 클러스터링이며, 둘째, 이러한 클러스터 내에서 의미적 불일치를 측정하는 SINdex입니다. 이 프레임워크는 외부 데이터나 수동 개입 없이도 응답을 집계하여 생성된 정보의 정확성을 평가합니다.

- **Performance Highlights**: 제안된 방법은 TriviaQA, NQ, SQuAD 및 BioASQ와 같은 여러 유명한 오픈북 및 클로즈북 질의응답 데이터셋에서 기존의 최첨단 기법들에 비해 상당한 성능 향상을 보였습니다. 또한 대규모 설정에서의 스케일러빌리티 실험을 통해, 기존 환각 탐지 기법에 비해 60배 빠른 처리 속도를 달성했습니다.



### SANDWiCH: Semantical Analysis of Neighbours for Disambiguating Words in Context ad Hoc (https://arxiv.org/abs/2503.05958)
Comments:
          15 pages, 2 figures, 7 tables, NAACL 2025

- **What's New**: 본 논문에서는 다국어 단어 의미 구별(Word Sense Disambiguation, WSD)을 위한 새로운 프레임워크인 SANDWiCH를 소개합니다. 이 프레임워크는 전통적인 개별 의미 구별에서 시맨틱 네트워크의 클러스터 구별로 문제를 재구성합니다. 우리의 접근 방식은 BabelNet를 사용해 정제된 시맨틱 네트워크를 활용하며, 파라미터 수를 72% 줄여 성능을 향상시킵니다.

- **Technical Details**: SANDWiCH는 두 수준의 프레임워크로 구성되어 있으며, 먼저 의미를 구분한 시맨틱 네트워크를 처리하고, 이웃 개념을 훈련 데이터의 일부로 포함하여 모델을 구분합니다. 주어진 문맥에서 후보 의미 간의 시맨틱 유사성을 평가하여 클러스터 기반의 접근 방식을 취합니다. 실험 결과, 모델은 영어 단어 의미 구별 작업에서 F1 점수를 8% 개선했으며, 다양한 작업에서 우수한 성과를 보였습니다.

- **Performance Highlights**: 영어 및 다국어 WSD 데이터셋에서 이전의 최첨단 결과를 상회하는 성능을 달성했습니다. 특히 저자원이 언어에서도 유의미한 개선을 보여주었으며, 기존 솔루션 대비 모든 데이터셋과 품사에서 뛰어난 성과를 달성했습니다. 이로 인해 SANDWiCH 프레임워크는 다양한 언어의 WSD 성능 격차를 줄이는 데 기여할 것으로 기대됩니다.



### DETQUS: Decomposition-Enhanced Transformers for QUery-focused Summarization (https://arxiv.org/abs/2503.05935)
Comments:
          12 pages, 2 figures, Accepted to NAACL 2025 main conference

- **What's New**: DETQUS(Decomposition-Enhanced Transformers for QUery-focused Summarization)는 사용자의 쿼리에 기반하여 테이블 데이터를 요약하는 새로운 시스템입니다. 기존의 트랜스포머 모델의 토큰 제한과 대형 테이블의 복잡한 추론 문제를 해결하기 위해 테이블 분해(tabular decomposition) 기법을 통해 성능을 향상시킵니다. 이 모델은 대형 언어 모델을 사용하여 쿼리에 관련된 열만 선택적으로 유지하며, 이를 통해 처리 효율성과 요약 품질을 동시에 향상시킵니다.

- **Technical Details**: DETQUS는 쿼리에 기반하여 테이블을 역동적으로 축소하고 관련 정보만 선택하여 요약을 생성합니다. 이 시스템은 입력 길이의 제약을 줄이고 중요한 콘텐츠를 보존함으로써 복잡한 쿼리와 대형 테이블을 효과적으로 처리할 수 있도록 설계되었습니다. 이전의 REFACTOR 모델을 초과하는 0.4437의 ROUGE-L 점수를 기록하며, DETQUS는 더 구조적이고 해석 가능한 접근 방식을 제공합니다.

- **Performance Highlights**: 연구 결과, DETQUS는 여러 트랜스포머 기반 모델 간의 성능 평가에서 개선된 요약 품질을 입증하였습니다. 특히, QTSUMM 데이터셋을 사용하여 테스트한 결과, 기존 모델들과 비교해 더욱 효율적인 쿼리 기반 요약을 생성하였습니다. 이는 대형 테이블을 처리하는 데 있어 DETQUS의 유용성을 보여주며, 사용자 요구에 맞춘 적시 정보를 제공하는 데 매우 효과적입니다.



### Training and Inference Efficiency of Encoder-Decoder Speech Models (https://arxiv.org/abs/2503.05931)
- **What's New**: 이번 연구에서는 최근의 최첨단 음성 모델인 Whisper, OWSM, Canary-1B의 주요 아키텍처인 Attention Encoder-Decoder 모델의 효율성을 분석합니다. 특히 이러한 음성 모델들이 훈련하는 데 드는 데이터와 계산 자원의 요구 사항을 효율적으로 극대화할 수 있는 방안을 제시합니다. 우리는 이러한 모델 훈련에서 가장 심각한 비효율적인 요소가 시퀀스 데이터를 샘플링하는 전략과 관련이 있다고 주장하며, 개선할 수 있는 방법을 알아봅니다.

- **Technical Details**: 모델 훈련의 효율성을 정의하는 두 가지 목표는 하드웨어 자원 활용의 극대화와 불필요한 계산의 최소화입니다. 이 연구에서는 2D bucketing 기법을 통해 길이가 비슷한 샘플로 미니 배치를 구성하여 패딩을 최소화하는 방법을 논의합니다. 이에 따라 Canary-1B의 훈련을 최적화하여 GPU 활용도를 대폭 향상시키고, 같은 처리 시간 내에 GPU 수를 4배 줄이는 성과를 이뤘습니다.

- **Performance Highlights**: 최종적으로, autoregressive decoder 단계에서의 병목 문제를 개선하기 위한 아키텍처 조정을 통해 추론 속도를 3배 향상했습니다. 최적화된 설정에 따르면 Canary-1B는 동일한 처리 시간 내에 4배 적은 GPU로 훈련이 가능하며, 고정 배치 크기 훈련과 비교할 때 2배 더 빠르게 수렴하였습니다. 훈련 코드 및 모델은 오픈 소스 소프트웨어로 제공될 예정입니다.



### IDEA Prune: An Integrated Enlarge-and-Prune Pipeline in Generative Language Model Pretraining (https://arxiv.org/abs/2503.05920)
- **What's New**: 최근 대형 언어 모델의 발전은 제한된 추론 예산 내에서 효율적이고 배포 가능한 모델의 필요성을 증가시켰습니다. 본 논문에서는 이전 연구에서 종종 간과된 확대 모델의 사전 훈련을 가지치기(Pruning) 파이프라인에 통합하는 것을 제안합니다. 우리는 'enlarge-and-prune' 파이프라인을 통해 모델의 성능을 저하시키지 않으면서도 개선할 수 있는 방법을 연구합니다.

- **Technical Details**: 우리는 통합된 'IDEA Prune' 파이프라인을 제안하여 확대 모델 훈련, 가지치기, 복구 단계를 하나의 cosine annealing learning rate 일정 아래 통합합니다. 이 접근법은 학습 속도의 상승에서 오는 지식 손실을 완화하고 생존하는 뉴런 사이에서 모델 용량을 효과적으로 재분배합니다. 또한, 점진적인 파라미터 제거를 위한 새로운 반복적 구조화 가지치기 방법을 적용하여 FB에 대한 압축을 진행합니다.

- **Performance Highlights**: 우리의 실험 결과에 따르면, 2.8B 모델을 1.3B로 압축하면서 최대 2T의 훈련 토큰을 사용했습니다. IDEA Prune은 기존의 방법들과 비교했을 때 일관된 성능 개선을 보여주며, 특히 MMLU 정확도가 46.4%에 달해 기본 방법의 31.4-33.4%보다 현저하게 향상되었습니다. 우리는 추가적인 연구를 통해 중간 체크포인트가 가지치기 시작점을 더욱 유리하게 제공할 수 있음을 확인했습니다.



### From Style to Facts: Mapping the Boundaries of Knowledge Injection with Finetuning (https://arxiv.org/abs/2503.05919)
- **What's New**: 이번 연구는 언어 모델의 맞춤화(finetuning)가 특정 작업(task)이나 응답 양식(response styles)에 대해 신뢰성을 높이고 효율적인 방법임을 제시합니다. 기존의 일반적인 관념과 달리, 지식을 주입(injecting knowledge)하는 과정이 불안정한 성능을 초래한다고 본다는 것에 반발합니다. 연구자들은 '작업 맞춤화(task customization)'와 '지식 주입(knowledge injection)'이라는 이분법이 실제로는 차이가 없다는 주장을 펼칩니다.

- **Technical Details**: 연구는 Gemini v1.5 모델 패밀리를 대상으로, 여러 유형의 데이터셋을 활용한 대규모 실험을 진행했습니다. 이 데이터셋들은 맞춤화의 강점과 실패 양상을 인터폴레이션(interpolate)하기 위해 인위적으로 설계되었습니다. 연구 결과에 따르면 질문-답변 형식의(training data formats) 데이터는 문서/기사 스타일의 데이터에 비해 훨씬 강력한 지식 일반화(knowledge generalization)를 제공함을 확인했습니다.

- **Performance Highlights**: 또한, 숫자 정보(numerical information)는 범주형 정보(categorical information)에 비해 유지하기 더 어려움이 있음을 보여주었습니다. 모델들은 유사한 예제로 훈련되었음에도 불구하고 다단계 추론(multi-step reasoning) 중에 맞춤화된 지식을 적용하는 데 어려움을 겪는 것으로 나타났습니다. 반면, 현실 세계 사건에 대한 정보와 모델의 글쓰기 스타일에 대한 정보를 맞춤화하는 것은 본질적으로 동일하게 어렵지 않다는 점도 확인했습니다.



### MastermindEval: A Simple But Scalable Reasoning Benchmark (https://arxiv.org/abs/2503.05891)
Comments:
          9 pages, 2 figures, 4 tables

- **What's New**: 최근의 대형 언어 모델(LLMs) 발전이 언어 이해 및 수학적 과제에서 뛰어난 성능을 거두며, 이러한 모델들의 실제 추론 능력을 평가하는 연구가 증가하고 있습니다. 본 논문에서는 Mastermind에서 영감을 받은 새로운 추론 벤치마크인 MastermindEval을 소개합니다. 이 벤치마크는 에이전트가 자율적으로 게임을 플레이하거나 사전에 플레이한 게임 상태를 기반으로 추론하는 두 가지 평가 방법론을 지원합니다.

- **Technical Details**: MastermindEval은 코드 길이(c), 기호 수(n), 허용된 추측 횟수(g)의 세 가지 주요 매개변수로 구성됩니다. 예를 들어, c=4, n=6인 경우, 1296개의 가능한 코드 조합이 존재합니다. 기존 모델들은 단순한 Mastermind 게임에서도 어려움을 겪고 있으며, 이는 모델이 정보를 결합하는 능력에 한계가 있음을 보여줍니다.

- **Performance Highlights**: 본 연구 결과에 따르면, 현재의 LLM들이 Mastermind와 같은 간단한 게임 과제를 해결하는 데 어려움을 겪고 있으며, 향후 더 발전된 모델에 대한 확장 가능성을 보여줍니다. 이는 모델의 추론 능력을 향상시키기 위한 새로운 데이터셋과 프레임워크의 필요성을 강조합니다.



### QG-SMS: Enhancing Test Item Analysis via Student Modeling and Simulation (https://arxiv.org/abs/2503.05888)
Comments:
          Under Review

- **What's New**: 이 연구에서는 교육 평가에서 질문 생성(Question Generation, QG) 작업의 평가 방식에 대한 새로운 접근법을 제시합니다. 기존의 QG 평가 방식은 교육적 가치와의 명확한 연계를 결여하고 있어, 테스트 항목 분석(test item analysis)을 도입하여 질문의 품질을 평가하는 방법을 강조하고 있습니다. 연구진은 품질이 다른 질문 쌍을 구성하여 기존의 QG 평가 방식이 이러한 차이를 효과적으로 구분할 수 있는지 살펴보았습니다.

- **Technical Details**: 연구에서는 질문 품질을 평가하기 위해 교육 분야에서 널리 사용되는 테스트 항목 분석(Method)은 주제 범위(topic coverage), 항목 난이도(item difficulty), 항목 변별력(item discrimination), 주의 산만 선택지의 효율성(distractor efficiency) 등 네 가지 차원에서 이루어집니다. 기존의 QG 평가 방식은 주제 범위에 대해서는 잘 작동하지만, 항목 난이도 및 변별력과 같은 후처리 분석의 차원을 정확하게 평가하는 데 큰 한계를 보였습니다. 이러한 문제를 해결하기 위해, 연구자들은 새로운 QG 평가 프레임워크인 QG-SMS를 제안하였습니다.

- **Performance Highlights**: QG-SMS는 다양한 학생 모델링 및 시뮬레이션을 활용하여 질문 품질 평가에서 성능을 높이며, 이는 기존의 LLM 기반 평가 방법의 한계를 극복합니다. 본 연구에서는 대규모 실험과 인간 평가를 통해 QG-SMS의 효과성과 강건성을 입증하였습니다. 연구 결과는 교육적 측면에서의 질문 품질 평가에 있어 significant improvements를 보여줍니다.



### This Is Your Doge, If It Please You: Exploring Deception and Robustness in Mixture of LLMs (https://arxiv.org/abs/2503.05856)
Comments:
          35 pages, 9 figures, 16 tables

- **What's New**: Mixture of large language model (LLMs) Agents (MoA) 아키텍처는 AlpacaEval 2.0과 같은 주요 벤치마크에서 뛰어난 성과를 달성했습니다. 하지만 이러한 MoA의 안전성과 신뢰성에 대한 평가가 부족했습니다. 이 연구에서는 고의적으로 오해를 일으키는 응답을 제공하는 기만적 LLM 에이전트에 대한 MoA의 강건성을 평가하는 최초의 포괄적 연구를 제시합니다.

- **Technical Details**: MoA는 여러 LLM의 전문성을 집약하여 성능을 극대화하는 multi-layer 구조를 가진 시스템입니다. 이 시스템에서는 제안자(proposers)와 집계자(aggregators) 간의 협력을 통해 다양한 관점을 수집하고 응답의 품질을 향상시킵니다. MoA의 특징 중 하나는 탈중앙화(decentralization) 배치가 가능하다는 점이며, 이는 잠재적인 취약점을 초래할 수 있습니다.

- **Performance Highlights**: 연구 결과, 단 하나의 잘 조작된 기만적 에이전트를 MoA에 도입할 경우 성능이 37.9%로 감소되어 MoA의 모든 이점을 무력화할 수 있음을 보여줍니다. 또한, QuALITY 다항선택 이해 과제에서 정확도가 48.5% 급락하는 심각한 영향을 받습니다. 본 연구는 MoA의 안전성을 확보하기 위한 여러 방어 메커니즘을 제안합니다.



### Extracting and Emulsifying Cultural Explanation to Improve Multilingual Capability of LLMs (https://arxiv.org/abs/2503.05846)
Comments:
          under review, 18pages

- **What's New**: 최근 연구에서는 다국어 능력을 개선하기 위한 새로운 접근법인 EMCEI(Extract and emulsify Cultural explanation)를 제안합니다. EMCEI는 문화적 맥락을 통합하여 LLMs의 응답을 더욱 정확하고 적절하게 만듭니다. 이를 통해 기존의 다국어 프롬프트 방식이 간과한 문화적 요소를 반영하여 다양한 언어로 질높은 응답을 제공합니다.

- **Technical Details**: EMCEI는 두 단계 프레임워크를 따릅니다. 첫 번째 단계에서는 LLM의 파라메트릭 지식을 활용하여 관련 문화적 정보를 추출합니다. 그 후, 두 번째 단계에서는 추출한 정보를 사용하여 다양한 유형의 질문에 맞춤형 응답을 생성하도록 LLM에 요청합니다.

- **Performance Highlights**: EMCEI의 효과는 여러 다국어 벤치마크에서 검증되었습니다. 예를 들어, EMCEI는 전통적인 방법에 비해 평균 16.4%의 향상을 보였고, 특히 자원이 적은 언어에서는 32.0%의 개선이 나타났습니다. 이러한 결과는 EMCEI가 문화적 지식을 효과적으로 활용하여 비영어 쿼리를 처리하는 데 큰 기여를 한다는 것을 보여줍니다.



### FedMentalCare: Towards Privacy-Preserving Fine-Tuned LLMs to Analyze Mental Health Status Using Federated Learning Framework (https://arxiv.org/abs/2503.05786)
Comments:
          9 pages, 3 figures, 2 tables and 2 algorithms

- **What's New**: 해당 연구에서는 FedMentalCare라는 프레임워크를 제안하며, 이는 Federated Learning(FL)과 Low-Rank Adaptation(LoRA)을 결합하여 대규모 언어 모델(LLM)을 정신 건강 분석에 맞게 조정하기 위해 데이터 프라이버시를 보장하는 것을 목표로 합니다. 또한, 클라이언트 데이터 양의 변화 및 모델 아키텍처가 FL 환경에서 성능에 미치는 영향을 조사합니다. 이 프레임워크는 데이터 보안과 계산 효율성을 해결하는 데 중점을 두어 실제 정신 건강 관리에 LLM을 배포할 수 있는 확장 가능하고 프라이버시 인식 접근 방식을 제공합니다.

- **Technical Details**: 대규모 언어 모델(LLMs)은 최근에 transformer 아키텍처를 통해 큰 발전을 이루었으며, 그 결과 Pre-trained Language Models(PLMs)와 같은 다양한 모델들이 등장했습니다. 특히, MobileBERT와 MiniLM과 같은 소형 언어 모델들을 사용하여 Federated Learning(FL) 환경에서 성능을 분석하는 방법을 모색합니다. FL은 사용자 데이터를 노출하지 않고 모델 훈련을 가능하게 하여 데이터 유출 위험을 감소시킵니다.

- **Performance Highlights**: 정신 건강 지원을 위한 LLMs의 통합은 방대한 관심을 받고 있으며, 텍스트 분석을 통한 우울증과 불안의 초기 징후를 탐지하는 데 기여하고 있습니다. 하지만, 현재 FL 환경에서의 최신 LLM들이 정신 건강 지원에 얼마나 효과적인지는 여전히 많은 의문을 남깁니다. 따라서, 이 연구는 FL을 활용하여 소셜 미디어 텍스트에서의 정신 건강 징후를 탐지할 수 있는 LLM의 활용 가능성을 탐구합니다.



### Medical Hallucinations in Foundation Models and Their Impact on Healthcar (https://arxiv.org/abs/2503.05777)
- **What's New**: 이번 논문은 의료 분야에서의 Foundation Models의 다중 모드(multi-modal) 데이터 처리 및 생성 능력에 대해 다루고 있으며, 특히 의료적 환각(medical hallucination)에 대한 문제점을 강조합니다. 이러한 환각은 잘못된 정보가 생성되어 임상 결정 및 환자 안전에 영향을 줄 수 있습니다. 연구진은 의료 환각의 정의, 특성 및 임상 시나리오에서의 실질적 영향을 탐구합니다.

- **Technical Details**: 논문에서는 의료 환각을 이해하고 해결하기 위한 분류법(taxonomy)을 제시하고, 의료 환각 데이터셋을 사용하여 모델을 벤치마킹(benchmarking)합니다. 또한, 임상 사례에 대한 의사 주석(physician-annotated) LLM 응답을 비교하여 환각이 임상에 미치는 직접적인 통찰을 제공합니다. Chain-of-Thought(CoT) 및 Search Augmented Generation과 같은 추론(inference) 기법이 환각 비율을 감소시키는 데 효과적임을 보여주지만, 여전히 비주얼적인 환각이 존재합니다.

- **Performance Highlights**: 연구 결과에 따르면 의료 환각의 비율을 낮추기 위한 여러 접근 방식이 제시되었음에도 불구하고 상당한 수준의 환각이 여전히 지속되고 있음을 알 수 있습니다. 이는 AI가 의료 분야에 통합됨에 따라 환자 안전과 임상 무결성을 유지하기 위한 규제 정책의 필요성을 강조합니다. 의료진의 피드백을 통해 기술적 발전뿐만 아니라 윤리적 및 규제적 지침이 필요하다는 점도 강조되었습니다.



### Graph Masked Language Models (https://arxiv.org/abs/2503.05763)
- **What's New**: 이 논문에서는 언어 모델(LMs)과 구조화된 지식 그래프(KGs) 간의 상호작용을 개선하기 위해 새로운 접근 방식을 제안합니다. 특히, 그래프 마스킹 언어 모델(GMLM)이 노드 분류 작업을 위해 도입되며, 이는 의미적 마스킹 전략과 소프트 마스킹 메커니즘을 통해 특징적인 그래프 정보를 효과적으로 활용할 수 있도록 설계되었습니다. 이러한 접근은 노드의 구조적 중요성을 고려하여 중요한 그래프 구성 요소가 학습에 실질적으로 기여할 수 있도록 합니다.

- **Technical Details**: 제안된 GMLM은 이중 가지 모델 아키텍처를 통해 구조 그래프 정보와 컨텍스트 임베딩을 결합합니다. 이 모델은 정량적인 정보 흐름과 훈련 과정 중 세부 정보 보존을 가능하게 하는 소프트 마스킹 메커니즘을 통해 노드 표현을 생성합니다. 연구는 또한 노드의 중요성을 반영한 의미적 마스킹 전략을 사용하여, 훈련 동안 신뢰성과 안정성을 강조합니다.

- **Performance Highlights**: 여섯 개의 노드 분류 베치마크에서 진행된 실험에 따르면, GMLM은 최첨단(SOTA) 성능을 달성하며 다양한 데이터셋에서 강력한 안정성과 견고함을 보여줍니다. 이러한 결과는 GMLM의 적용이 기존 GNN 아키텍처의 한계를 극복하고 노드 표현 학습에서의 효과성을 극대화할 수 있음을 시사합니다.



### CSTRL: Context-Driven Sequential Transfer Learning for Abstractive Radiology Report Summarization (https://arxiv.org/abs/2503.05750)
Comments:
          11-pages main paper with 2-pages appendices

- **What's New**: 이 논문에서는 방사선 보고서에서 Findings에서 Impression을 자동으로 생성하는 방법을 제시합니다. 기계 학습을 활용하여 임상 맥락을 유지하면서 핵심 정보를 추출하는 Sequential Transfer Learning 접근법을 도입했습니다. 이 방법은 Fisher matrix 정규화를 통해 지식 손실 문제를 해결하고, 방사선 전문 용어의 복잡성을 다룹니다.

- **Technical Details**: 논문에서 제안된 모델은 두 단계로 구성된 Sequential Transfer Learning 방식을 사용하여 T5 모델을 미세 조정합니다. 첫 단계에서는 GSG(Gap Sentence Generation) 작업을 통해 중요한 문장을 학습하고, 두 번째 단계에서는 임상 요약을 위해 GSG를 통한 학습된 가중치를 활용합니다. 이를 통해 지식 전이를 효과적으로 수행하고 차원 축소를 이룰 수 있습니다.

- **Performance Highlights**: CSTRL-Context-driven Sequential Transfer Learning 모델은 MIMIC-CXR 및 Open-I 데이터셋에서 기존 연구에 비해 BLEU 및 ROUGE 점수에서 현저한 향상을 보였습니다. BLEU-1에서 56.2%, BLEU-2에서 40.5%, ROUGE-1에서 28.9% 등의 성과를 기록하며, 일본어 임상 맥락을 유지하는 데 있어 사실적 일관성 점수 또한 분석되었습니다.



### What Are They Filtering Out? A Survey of Filtering Strategies for Harm Reduction in Pretraining Datasets (https://arxiv.org/abs/2503.05721)
- **What's New**: 이 논문은 데이터 필터링 전략이 취약 그룹에게 미치는 실제 영향에 대한 체계적인 분석을 제공하는 최초의 연구입니다. 연구에서는 총 55개의 기술 보고서를 조사하여 기존의 데이터 필터링 전략을 파악하고, 이를 활용하여 선별된 전략들이 취약 그룹의 언급 비율을 저하시킨다는 것을 실험적으로 입증하였습니다. 이 결과는 데이터 필터링이 긍정적인 효과를 가지더라도, 취약 그룹의 등장을 줄이는 부작용을 동반할 수 있음을 보여줍니다.

- **Technical Details**: 본 연구에서는 데이터 필터링 전략을 이해하기 위해 다루는 55개의 기술 문헌에서 정보를 수집하고, 이를 분석하기 위해 방법론을 확립하였습니다. 필터링 전략의 유형을 분류하기 위한 세분화된 세 가지 기준을 설정하였고, 소수자 내지 약자 그룹에 대한 언급을 조사하는 파이프라인을 설계하여 필터링 전략의 유형별 효과를 검토했습니다.

- **Performance Highlights**: 연구 결과는 여성 그룹이 필터링 전략의 영향을 가장 많이 받는 것으로 나타났으며, 각 전략의 필터링 결과는 상이합니다. 선택되는 필터링 전략에 따라 특정한 해의 원천에 집중하면서도 다른 원천의 해는 간과할 수 있는 점이 확인되었습니다. 이러한 결과들은 필터링 전략이 취약 그룹의 언급 비율을 저하시킬 수 있다는 점에서 더 큰 문제의 심각성을 시사합니다.



### VisBias: Measuring Explicit and Implicit Social Biases in Vision Language Models (https://arxiv.org/abs/2503.07575)
Comments:
          9 pages

- **What's New**: 이 연구는 Vision-Language Models (VLMs)에서 나타나는 명시적 및 암묵적 사회 편향을 조사합니다. 연구진은 성별과 인종 차이에 관한 여러 질문을 통해 명시적 편향을 분석하고, 사용자 요청에 도움을 주면서도 편향을 드러내는 작업을 통해 암묵적 편향을 이해하고자 합니다. 이에 따라, 사용자의 요청에 도움이 되는 질문 및 작업을 설계해 편향 행동을 자세히 분석하려고 합니다.

- **Technical Details**: 편향 분석을 위해 연구진은 네 가지 평가 시나리오를 설계했습니다: (I) 다중 선택 질문, (II) 예-아니오 질문, (III) 이미지 설명, (IV) 양식 작성입니다. 이들 질문은 주로 개인의 외모를 기반으로 모델이 결정을 내리도록 설계되었으며, 인종, 성별 및 직업의 만남을 고려하여 이미지를 수집합니다. 또한, 편향 노출을 방지하기 위한 방법으로 Caesar cipher를 활용한 '탈옥'(jailbreak) 기법이 적용됩니다.

- **Performance Highlights**: 연구진은 GPT-4(V), GPT-4o, Gemini-1.5-Pro, LLaMA-3.2 (Vision), LLaVA-v1.6와 같은 VLM들을 평가했습니다. 실험 결과, 이러한 모델들이 명시적 질문의 경우 잘 수행하였으나, 암묵적 작업에서는 성별, 인종, 종교 등의 편향이 여전히 문제로 나타났습니다. 이러한 발견은 고급 VLM에서 사회 편향을 완화하려는 지속적인 노력이 필요함을 강조합니다.



### Optimizing Test-Time Compute via Meta Reinforcement Fine-Tuning (https://arxiv.org/abs/2503.07572)
- **What's New**: 이 논문은 LLM의 reasoning (추론) 성능을 개선하기 위한 테스트 시간 컴퓨팅(test-time compute)의 최적화 문제를 메타 강화 학습(meta-reinforcement learning) 관점에서 다루고 있습니다. 기존 방법들이 주로 0/1 결과 보상을 사용하는 RL(reinforcement learning) 방식으로 이루어졌다면, 이 연구에서는 누적 후회(cumulative regret) 개념을 도입해 성과를 측정하고자 합니다. 이로 인해 LLM의 출력 스트림을 여러 에피소드로 나누어 분석하는 새로운 접근법이 가능하게 됩니다.

- **Technical Details**: 논문에서는 테스트 시간 컴퓨팅을 최적화하기 위해 메타 강화 학습을 수립하고, 이를 통해 LLM의 출력 토큰의 누적 후회를 최소화하는 방법을 제안합니다. 저자들은 에피소드의 맥락에서 오류 검출과 전략적 백트래킹을 구현할 수 있는 구조화된 접근법인 '백트래킹 검색(backtracking search)'을 탐구합니다. 이 설정에서는 해답을 생성하고, 이전 시도의 오류를 검출한 후 적절한 단계로 되돌아가는 과정을 통해 모델의 오류 감지 능력을 향상시킵니다.

- **Performance Highlights**: Meta Reinforcement Fine-Tuning (MRT) 기법을 도입함으로써, 실험 결과 LLM은 결과 보상을 사용하는 RL 기술 대비 2-3배의 성능 향상을 달성하게 됩니다. 또한, MRT는 수학적 추론에서 약 1.5배의 토큰 효율성을 개선한 것으로 나타났습니다. 전반적으로 MRT는 알고리즘의 진행(progress)을 증대시키는 효능이 있음을 보여주며, 에피소드 간의 추론과 오류 교정을 통해 모델의 학습 효율성을 높이는 데 기여하고 있습니다.



### Building English ASR model with regional language suppor (https://arxiv.org/abs/2503.07522)
Comments:
          5 pages, 3 figures

- **What's New**: 이 논문에서는 영어 자동 음성 인식 시스템에서 힌디 쿼리를 효과적으로 처리하기 위한 새로운 접근 방법을 제안합니다. 제안된 모델은 SplitHead with Attention(SHA)라는 명칭의 새로운 음향 모델(acoustic model)로, 공유된 숨겨진 계층과 언어별 프로젝션 계층을 사용하여 자기 주의 메커니즘(self-attention mechanism)으로 결합합니다. 이 시스템은 영어 쿼리에 대한 성능을 저하시키지 않고 힌디 쿼리의 성능을 향상시키는 것을 목표로 합니다.

- **Technical Details**: 우리는 언어별 프로젝션 계층을 통해 각 언어의 은닉 표현(hidden representations)을 생성하며, 주의 모델(attention model)을 이용해 각 언어의 가중치를 계산합니다. 특히, 기존의 공유 숨겨진 계층 모델링 접근 방식과는 달리, 모든 언어는 동일한 청소(chensones) 세트를 사용하도록 제한함으로써 언어 간 조합을 가능하게 합니다. 이 외에도, n-그램(language model) 모델을 기반으로 한 언어 모델은 힌디 텍스트의 저작물과 영어 n-그램 모델을 보간(interpolate)하는 방법을 포함합니다.

- **Performance Highlights**: 우리의 접근 방식은 힌디와 영어 테스트 세트에서 각각 69.3% 및 5.7%의 단어 오류율(word error rate) 감소를 보여주며 이는 기존의 단일 언어 모델에 비해 우수한 성능을 나타냅니다. 이러한 성과는 컴퓨터 자원을 증가시키지 않고도 힌디 쿼리에 대한 인식 정확도를 크게 향상시킨 것입니다. 즉, 이 향상된 en-IN ASR 모델은 기존의 단일 언어 모델을 대체할 수 있는 실질적이고 상업적으로 유용한 솔루션을 제공합니다.



### GRITHopper: Decomposition-Free Multi-Hop Dense Retrieva (https://arxiv.org/abs/2503.07519)
Comments:
          Under Review at ACL Rolling Review (ARR)

- **What's New**: GRITHopper-7B는 혼합적 언어 모델링과 밀집 검색 훈련을 통합한 새로운 다단계 밀집 검색 모델이다. 이는 분해 기반 접근방식의 한계를 극복하고, 분해 없는 방법이 긴 다단계 문제에서 어려움을 겪는 현상을 해결하고자 한다. 이 모델은 다양한 질문-답변 및 사실 확인 작업에 대한 다단계 데이터셋에서 훈련되었다.

- **Technical Details**: GRITHopper는 GRITLM을 기반으로 하여 인과(language modeling) 언어 모델링과 밀집 검색 훈련을 결합하였으며, 밀접하게 관련된 정보 습득을 위해 최종 답변 등의 요소를 훈련 과정에 포함시킨다. 이 접근법은 '포스트-검색 언어 모델링'(post-retrieval language modeling)으로 정의되며, 검색 체인 이후의 추가 정보를 통해 밀집 검색 성능을 향상시킨다. 이를 통해 참가자들에게 더 효과적으로 관련 정보를 검색하는 법을 학습하게 된다.

- **Performance Highlights**: GRITHopper-7B는 최신 멀티홉 밀집 검색에서 최상위 성능을 달성하며, 기존 모델들과 비교했을 때 다양한 일반화 성능을 보여준다. 이 모델은 특히 분해 없는 방식으로 다단계 문제를 잘 처리하며, 이전 접근 방식에서 발생한 계산 비용 문제를 해결하는 데 도움을 준다. 궁극적으로 GRITHopper-7B는 향후 다단계 추론 및 검색 능력이 필요한 연구와 응용에 적합한 견고한 솔루션을 제공한다.



### Sometimes the Model doth Preach: Quantifying Religious Bias in Open LLMs through Demographic Analysis in Asian Nations (https://arxiv.org/abs/2503.07510)
- **What's New**: 본 연구는 Large Language Models(LLMs)가 생성하는 의견이 주로 비대표적인 데이터 수집에서 유래된 편향을 어떻게 반영하는지를 정량적으로 분석하는 새로운 방법을 제안합니다. 기존 연구가 주로 서구 사회에 맞춰져 있었던 반면, 우리는 비서구 지역에서의 LLM의 문화적 민감성에 주목하였습니다. 인도 및 기타 아시아 국가들의 설문조사를 통해 현대의 오픈 LLM들이 어떻게 반응하는지를 평가하며, 종교적 관용과 정체성에 관한 주제를 분석합니다.

- **Technical Details**: 우리는 Hamming Distance를 사용하여 LLM의 응답과 설문 응답자의 거리 측정을 통해 모델에서 반영된 인구 통계적 특성을 유추합니다. 연구에서 다룬 오픈 LLM은 Llama와 Mistral로, 이들은 공용으로 접근 가능한 아키텍처와 교육 코드 등을 갖추고 있어 쉽게 수정 및 공유가 가능합니다. 우리의 연구는 모델의 편향을 이해하는 첫 단계로, 다양한 인구 통계 변수의 상호 연결성을 인정하는 독창적인 방법을 제안합니다.

- **Performance Highlights**: 분석 결과, 대부분의 오픈 LLM은 단일 동질적인 프로필을 가지며, 이는 각 국가 및 지역에 따라 다르게 나타났습니다. 이러한 사실은 LLM이 지배적인 세계관을 촉진함으로써 다양한 소수자의 관점을 undermining할 수 있는 위험을 제기합니다. 우리는 연구 결과와 실험 코드베이스를 GitHub에 공개하여 추가 연구와 개발에 기여할 수 있도록 하였습니다.



### Is a Good Foundation Necessary for Efficient Reinforcement Learning? The Computational Role of the Base Model in Exploration (https://arxiv.org/abs/2503.07453)
- **What's New**: 이 논문은 언어 모델의 효율적인 탐색을 위한 새로운 계산 프레임워크를 소개합니다. 기존의 알고리즘 설계 원리에 대한 이해가 부족한 가운데, 이 연구는 강력한 사전 훈련된 생성 모델을 활용하여 탐색의 효율성을 개선하는 방법을 제시합니다. 특히, SpannerSampling라는 새로운 알고리즘을 도입하여 데이터 효율성을 최적화하고 탐색의 효과적인 검색 공간을 줄입니다.

- **Technical Details**: 저자는 선형 소프트맥스 모델 파라미터화를 중심으로 효율적인 탐색의 계산-통계적 트레이드오프를 밝혀냅니다. 주요 발견은 커버리지의 필요성, 추론 시간 탐색, 훈련 시간 개입의 한계 및 다중 턴 탐색의 계산적 이점에 관한 것입니다. 커버리지는 모델이 최적 응답을 얼마나 잘 포함하고 있는지를 의미하며, 이는 알고리즘의 런타임을 하한으로 제한합니다.

- **Performance Highlights**: SpannerSampling 알고리즘은 사전 훈련된 모델이 충분한 커버리지를 가질 때 최적의 데이터 효율성과 계산적 효율성을 달성합니다. 이 연구는 효과적인 다중 턴 탐색을 통해 런타임 개선을 보여주며, 이는 시퀀스 수준의 커버리지를 토큰 수준으로 교체하여 가능합니다. 이러한 결과는 언어 모델의 능력을 최대화하는 데 기여할 것으로 보입니다.



### VizTrust: A Visual Analytics Tool for Capturing User Trust Dynamics in Human-AI Communication (https://arxiv.org/abs/2503.07279)
Comments:
          Accepted by ACM CHI conference 2025

- **What's New**: 이 논문은 사용자 신뢰(user trust)를 실시간으로 분석할 수 있는 VizTrust라는 시각 분석 도구를 소개합니다. 기존의 사용자 신뢰 측정 방식은 복잡한 상호작용 중에 신뢰의 변화를 포착하지 못했지만, VizTrust는 이를 해결하고자 합니다. 이 도구는 협업 시스템(multi-agent collaboration system)을 활용하여 인간-에이전트 간의 소통에서의 신뢰 동적 변화를 시각적으로 나타냅니다.

- **Technical Details**: VizTrust는 네 가지 중요한 신뢰 차원—유능성(competence), 무결성(integrity), 선의(benevolence), 예측가능성(predictability)—에 기반하여 사용자 신뢰를 평가합니다. 이 도구는 자연어 처리(NLP) 및 머신러닝 기술을 사용하여 대화 중 발생하는 세부적인 신뢰 형성 신호를 분석하고, 시계열(time series) 시각화를 통해 인간-에이전트 간의 상호 작용을 정교하게 평가합니다. 또한, VizTrust는 대화에서의 사회적 신호(social signals)와 언어적 전략을 실시간으로 제공하여 대화 설계에 유용한 통찰을 제공합니다.

- **Performance Highlights**: VizTrust의 대시보드는 사용자의 신뢰 동력학, 참여도, 정서적 톤(또는 감정적 반응), 그리고 예의 이론(theory of politeness) 등을 시각적으로 표현하여 설계 이해관계자가 신뢰의 변화 지점을 확인할 수 있도록 도와줍니다. 이 도구는 실시간으로 사용자와 대화하는 에이전트의 반응을 바탕으로 신뢰 변화를 이해하고, 사용자와의 장기적인 상호작용에서 신뢰 요소를 최적화하는 데 기여합니다. VizTrust는 대화의 각 회전(turn)에서 수집된 결과를 종합적으로 시각화하여 효율적인 에이전트 설계를 지원하는 데 중요한 역할을 합니다.



### WISE: A World Knowledge-Informed Semantic Evaluation for Text-to-Image Generation (https://arxiv.org/abs/2503.07265)
Comments:
          Code, data and leaderboard: this https URL

- **What's New**: 본 논문에서는 텍스트-이미지(T2I) 모델의 평가를 위한 새로운 벤치마크인 WISE(World Knowledge-Informed Semantic Evaluation)를 제안합니다. 기존의 평가 기준들이 이미지 현실성과 단순한 텍스트-이미지 정합성에 주로 집중하는 반면, WISE는 보다 복잡한 의미 이해와 세계 지식 통합을 평가합니다. 또한 WiScore라는 새로운 정량적 메트릭을 도입하여 지식-이미지 정합성을 보다 rigorously 평가할 수 있도록 합니다.

- **Technical Details**: WISE는 자연 과학, 시공간 추론, 문화적 상식과 같은 세 가지 주요 영역을 포함하며, 25개의 서브 도메인에 걸쳐 1000개의 평가 프롬프트를 제공합니다. WiScore는 생성된 이미지 내에서 객체와 개체의 정확한 묘사를 강조하는 새로운 복합 메트릭으로, 일관성(Consistency), 현실성(Realism), 미적 품질(Aesthetic Quality)의 세 가지 주요 요소의 가중 평균으로 계산됩니다. 이러한 체계적인 접근 방식을 통해, 기존 T2I 모델들이 세계 지식을 효과적으로 통합하고 적용하는 데 제한적이라는 것을 보여줍니다.

- **Performance Highlights**: 20개의 T2I 모델(10개의 전용 모델 및 10개의 통합 멀티모달 모델)을 평가한 결과, 기존 T2I 모델들은 복잡한 의미 이해 및 세계 지식 통합에 있어 상당한 한계를 드러냈습니다. 통합 멀티모달 모델조차도 전용 T2I 모델에 비해 이미지 생성에 있어 우위가 확인되지 않았으며, 이는 현재의 통합 접근 방법이 이미지 생성에서 세계 지식을 효과적으로 활용하는 데 한계를 나타냅니다. 이러한 결과는 T2I 모델의 발전을 위한 주요 경로를 제시합니다.



### Multi-Modal 3D Mesh Reconstruction from Images and Tex (https://arxiv.org/abs/2503.07190)
Comments:
          under review

- **What's New**: 이 논문에서는 보지 못한 객체의 6D 포즈 추정을 위한 언어 기반의 소수 샷(몇가지 샷) 3D 재구성 방법을 제안합니다. 기존의 방법들이 대규모 데이터셋과 고비용의 컴퓨팅 자원을 요구하는 반면, 우리는 몇개의 이미지와 언어 정보를 이용하여 3D 메쉬를 재구성하는 혁신적인 접근 방식을 제시합니다. 이를 통해 실시간 응용 프로그램에서의 3D 모델 생성을 보다 효율적이고 실용적으로 수행할 수 있습니다.

- **Technical Details**: 제안된 방법은 입력 이미지 집합과 언어 쿼리를 받아들입니다. GroundingDINO와 Segment Anything Model(SAM)의 조합을 통해 세그먼트 마스크를 생성하고, 이를 사용하여 VGGSfM을 이용해 드문포인트 클라우드를 재구성합니다. 그 후, Gaussian Splatting 기법인 SuGAR를 사용해 메쉬를 생성하며, 최종 단계에서는 여러 아티팩트를 제거하여 쿼리된 객체의 최종 3D 메쉬가 완성됩니다.

- **Performance Highlights**: 우리는 재구성의 정확성 및 기하학적, 텍스처 품질을 평가합니다. 실험은 AMD Ryzen 9 5950X CPU와 NVIDIA RTX 3090 GPU를 갖춘 시스템에서 수행되었으며, 재구성된 3D 기하 구조는 Chamfer Distance와 Intersection over Union을 사용해 평가됩니다. 낮은 Chamfer Distance 값은 정확한 기하학적 정렬을 나타내며, IoU는 재구성된 모델의 부피적 유사성을 측정하여 성능을 확인합니다.



### PoseLess: Depth-Free Vision-to-Joint Control via Direct Image Mapping with VLM (https://arxiv.org/abs/2503.07111)
- **What's New**: PoseLess는 로봇 손의 제어를 위한 새로운 프레임워크로, 명시적인 pose 추정이 필요 없이 2D 이미지를 직접적으로 관절 각도로 매핑하는 방법을 제안합니다. 합성 교육 데이터로 생성된 무작위 관절 구성( joint configurations)을 활용하여 실제 시나리오 및 다양한 손 형태 간의 전이( transfer)를 가능하게 하며, 이 과정에서 pose 추정 없이도 신뢰성 높은 제어를 달성합니다.

- **Technical Details**: PoseLess는 비전-언어 모델( vision-language model)을 사용하여 시각적 입력을 토큰화(tokenization)하고 이를 관절 각도로 디코딩하는 혁신적인 방법론을 제시합니다. 이는 복잡한 두 단계 파이프라인에서 발생할 수 있는 오류 전파를 줄이며, 다양한 손 형태에 대한 일반화 능력을 강화합니다. 또한, 합성 데이터 파이프라인을 통해 무한한 학습 예제를 생성하여 수작업으로 레이블을 붙인 데이터셋의 필요성을 제거합니다.

- **Performance Highlights**: 실험 결과, PoseLess는 인간 레이블 데이터셋에 의존하지 않으면서도 관절 각도 예측 정확도에서 경쟁력 있는 성능을 보여줍니다. 특히, 깊이 추정에 의존하지 않고도 제어가 가능하다는 증거를 제시하여, 로봇 공학 연구에서 자주 사용되는 카메라의 한계를 극복하는 가능성을 제시합니다. 다양한 손 형태 간의 전이를 성공적으로 이뤄내며, 이는 이전 연구에서 다루지 않았던 중요한 발전입니다.



### ProjectEval: A Benchmark for Programming Agents Automated Evaluation on Project-Level Code Generation (https://arxiv.org/abs/2503.07010)
Comments:
          17 pages (9 Appendix pages), 4 figures, 7 tables

- **What's New**: 최근 LLM(대형 언어 모델) 에이전트가 프로그래밍 능력 향상에 빠른 발전을 이뤘습니다. 그러나 현재의 벤치마크는 사용자 관점에서 자동으로 평가할 수 있는 기능이 부족합니다. 이를 해결하기 위해, ProjectEval이라는 새로운 벤치마크를 소개합니다. 이 벤치마크는 사용자 상호작용을 시뮬레이션하여 LLM 에이전트의 코드 생성 능력을 평가할 수 있도록 설계되었습니다.

- **Technical Details**: ProjectEval은 총 284개의 테스트 케이스로 구성된 20개의 실제 작업을 포함하고, 웹 기반 프로젝트와 배치/콘솔 기반 프로그램 두 가지 작업 유형을 지원합니다. 각 미션은 세 가지 입력 유형으로 구성되며, 이를 통해 LLM 에이전트가 복잡한 사용자 주도 작업을 정확하고 적응력 있게 수행할 수 있는지를 평가합니다. 이 시스템은 사용자 관점에서의 평가를 위해 세분화된 파라미터 분석과 자동화된 테스트 스위트를 통합한 개념을 특징으로 합니다.

- **Performance Highlights**: ProjectEval은 기존 벤치마크들과 비교했을 때 더 복잡한 테스트 케이스와 더 많은 테스트를 제공하여 LLM 에이전트의 프로그래밍 능력을 포괄적으로 평가할 수 있게 합니다. 이 과정에서 에이전트는 실사용자의 관점에서 프로젝트를 구성하고 실행하는 능력을 테스트하게 됩니다. 이를 통해 개발자들이 LLM 에이전트를 실제 프로덕션 환경에 효과적으로 배치할 수 있도록 돕는 통찰력을 제공합니다.



### Silent Hazards of Token Reduction in Vision-Language Models: The Hidden Impact on Consistency (https://arxiv.org/abs/2503.06794)
- **What's New**: 이 논문은 비주얼 랭귀지 모델(VLMs)의 토큰 감소가 출력 안정성에 미치는 영향을 분석합니다. 연구 결과, 기존의 성능 지표가 반영하지 못하는 모델 출력의 불일치를 발견했습니다. 이러한 불일치는 의료 시스템과 같이 신뢰성 높은 성능이 요구되는 실용적인 응용 분야에 심각한 문제를 초래할 수 있습니다. 저자들은 이러한 문제를 해결하기 위해 LoFi라는 새로운 훈련 없는 토큰 감소 방법을 제안합니다.

- **Technical Details**: 논문에서는 VLM의 내부 표현에서 에너지 분포가 토큰 감소에 의해 어떻게 변화하는지를 분석합니다. 세부적으로는 SVD(Singular Value Decomposition)를 사용하여 주성분 방향의 변화를 모니터링하며, IPR(Inverse Participation Ratio) 값을 통해 에너지 분포의 집중도를 평가합니다. 이러한 접근법은 각 레이어에서 토큰 감소가 내부 표현에 미치는 영향을 계량적으로 파악하는 데 도움을 줍니다.

- **Performance Highlights**: LoFi는 기존의 최첨단 방법들보다 높은 출력 일관성과 함께 계산 비용을 크게 줄였습니다. 실험 결과는 LoFi가 성능 저하를 최소화하면서도 원래의 모델과 상당히 일관된 출력을 제공함을 보여줍니다. 이 연구는 비주얼 랭귀지 모델이 높은 정확도를 유지하면서도 안정성을 높일 수 있는 새로운 방향을 제시하고 있습니다.



### Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models (https://arxiv.org/abs/2503.06749)
- **What's New**: DeepSeek-R1-Zero는 강화 학습(Reinforcement Learning, RL)만으로 LLM의 추리 능력을 성공적으로 증명했습니다. 이에 영감을 받아, 이번 연구에서는 MLLM의 추리 능력을 향상시키기 위해 RL을 어떻게 활용할 수 있을지 탐구합니다. 다만, MLLMs에서 직접적인 RL 학습은 고품질 다중 모달 추리 데이터의 부족으로 인해 복잡한 추리 능력을 이끌어내기에는 한계를 겪고 있습니다. 이를 극복하기 위해 Vision-R1이라는 추리 MLLM을 제안합니다.

- **Technical Details**: 이번 연구에서는 기존 MLLM과 DeepSeek-R1을 활용하여 사람의 주석 없이도 20만 개의 고품질 다중 모달 CoT(Coach-of-Thought) 데이터셋, Vision-R1-cold 데이터셋을 구축합니다. Vision-R1은 초기 데이터 셋을 통해 추리 능력을 높이는 한편, Progressive Thinking Suppression Training (PTST) 전략을 도입하여 모델의 올바른 추리 과정을 학습할 수 있도록 돕습니다. 특히, Group Relative Policy Optimization (GRPO) 기법과 하드 포맷 결과 보상 함수를 사용하여 10K 다중 모달 수학 데이터셋에서 모델의 성능을 점진적으로 개선합니다.

- **Performance Highlights**: 포괄적인 실험 결과, Vision-R1은 다양한 다중 모달 수학 추리 벤치마크에서 평균 6% 개선된 성능을 보였습니다. 특히, Vision-R1-7B는 MathVista 벤치마크에서 73.5% 정확도를 기록하며, 최고의 추리 모델인 OpenAI O1과 오직 0.4% 차이로 성능을 발휘했습니다. 이번 연구에서 제안한 데이터셋과 코드도 공개되어 연구자들 간의 상호 작용을 촉진할 것으로 기대됩니다.



### DependEval: Benchmarking LLMs for Repository Dependency Understanding (https://arxiv.org/abs/2503.06689)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)들이 코드 생성에서 유망한 성과를 보였지만, 실제 소프트웨어 개발에서는 advanced repository-level reasoning이 필요하다는 점을 강조합니다. 특히, 의존성(depencies) 이해와 프로젝트 구조를 파악하는 것이 중요하며, 이러한 복잡한 코드 리포지토리에 대한 LLM의 이해는 아직 충분히 탐구되지 않았습니다. 이를 해결하기 위해 DependEval이라는 새로운 계층적 벤치마크를 도입하였습니다.

- **Technical Details**: DependEval 벤치마크는 15,576개의 실제 웹사이트에서 수집된 리포지토리를 기반으로 하며, 의존성 인식(Dependency Recognition), 리포지토리 구성(Repository Construction), 멀티 파일 편집(Multi-file Editing)의 세 가지 핵심 작업을 평가합니다. 이 벤치마크는 실제 코드 리포지토리에서 8개 프로그래밍 언어를 아우르며, LLM들이 리포지토리 레벨의 코드 이해를 얼마나 잘 수행하는지를 측정하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 25개 이상의 LLM을 평가한 결과, 성능 차이가 상당하며, 이는 리포지토리 수준에서의 코드 이해에 대한 귀중한 통찰을 제공합니다. 특히, 리포지토리의 복잡한 구조와 파일 간의 관계를 효과적으로 이해하는 능력이 LLM의 성능을 좌우하는 것으로 보입니다. 이러한 연구 결과는 LLM의 코드 생성 능력을 한층 더 발전시키는 데 기여할 것입니다.



### Attention, Please! PixelSHAP Reveals What Vision-Language Models Actually Focus On (https://arxiv.org/abs/2503.06670)
- **What's New**: PixelSHAP는 Vision-Language Models (VLMs)에 대한 해석 가능성을 높이기 위해 Shapley 기반 분석을 구조적 시각 개체에 확장한 새로운 모델 무관 프레임워크입니다. 기존의 토큰 기반 접근 방식과는 달리, PixelSHAP는 이미지 개체를 체계적으로 변화시키고 이를 통해 모델의 반응에 미치는 영향을 정량화합니다. 이 프레임워크는 모델 내부에 대한 접근 없이 입력-출력 쌍만으로 작동하므로 오픈 소스 및 상용 모델 모두와 호환됩니다.

- **Technical Details**: PixelSHAP는 섹멘테이션(Segmentation) 모델을 활용하여 객체 마스크를 생성하고, 객체의 변화를 통해 모델의 출력을 평가합니다. 이러한 방식으로 픽셀 단위가 아닌 객체 단위의 perturbation을 수행하여 계산 효율성을 높입니다. 이 메서드는 객체의 중요성을 평가하기 위해 Shapley 기반의 중요도 추정을 적용하며, 복잡한 시각 장면의 해석 가능성을 크게 증가시킵니다.

- **Performance Highlights**: PixelSHAP는 자율 주행과 같은 높은 위험의 응용 분야에서 중요한 해석 가능성을 향상시키는 능력을 보여줍니다. 본 시스템의 개선된 신뢰성 및 투명성 덕분에 사용자는 모델의 결정에 어떤 객체가 영향을 미쳤는지 더 잘 이해할 수 있습니다. 이러한 기능들은 향후 설명 가능한 AI 연구의 진전을 위한 강력한 도구 역할을 할 것입니다.



### Evaluating and Aligning Human Economic Risk Preferences in LLMs (https://arxiv.org/abs/2503.06646)
- **What's New**: 이 연구에서는 대형 언어 모델(LLMs)의 리스크 선호도가 개인화된 페르소나에 따라 어떻게 달라지는지 탐구합니다. 특히, LLM의 응답이 개인의 리스크 회피 또는 리스크 추구 행동을 적절히 반영하는지를 평가합니다. 연구 결과, LLM은 단순한 리스크 상황에서는 합리적인 결정을 내리지만, 복잡한 경제 의사결정 작업에서는 성능이 저하되는 것으로 나타났습니다. 이를 해결하기 위해 페르소나에 맞춘 리스크 정렬 방법을 제안함으로써 AI의 의사결정 과정이 인간과 더 잘 일치할 수 있도록 합니다.

- **Technical Details**: 연구에서는 세 가지 실험을 통해 LLM의 리스크 선호도를 평가합니다. 첫 번째는 리스크 선호도를 직접 묻는 간단한 질문이며, 두 번째는 투자 시나리오 시뮬레이션입니다. 세 번째는 프로스펙트 이론을 활용하여 LLM의 경제적 합리성을 분석합니다. 연구는 400개의 다양한 페르소나 데이터를 활용하여 성별 및 연령에 따른 LLM의 리스크 클래스 분류 결과를 분석했습니다.

- **Performance Highlights**: 실험 결과, LLM은 성별에 따라 남성을 리스크 선호, 여성을 리스크 회피로 분류하는 경향이 있었습니다. 또한 연령이 증가할수록 리스크 선호도가 감소하는 것으로 나타났습니다. 복잡한 의사결정 작업에서 LLM의 리스크 정렬 방법이 유의미하게 향상되는 것을 보여주며, 특히 리스크와 관련된 의사결정에서 AI의 경제적 합리성을 개선하는 데 기여합니다.



### Is Your Benchmark (Still) Useful? Dynamic Benchmarking for Code Language Models (https://arxiv.org/abs/2503.06643)
Comments:
          14 pages, 7 figures

- **What's New**: 이 논문에서는 모델 평가 시 코드 벤치마크의 유용성을 유지하는 문제를 다룹니다. 이를 해결하기 위해 우리가 제안한 동적 벤치마킹 프레임워크는 코드 이해 또는 추론 벤치마크를 다이나믹하게 변형하여 구문적으로 새롭고 의미적으로 동일한 벤치마크를 생성합니다. 이 프레임워크는 여러 언어 모델을 평가했으며 결과적으로 모델들의 성능이 이전보다 크게 떨어짐을 보여줍니다.

- **Technical Details**: 동적 벤치마킹 프레임워크는 기존 코드 벤치마크의 각 테스트 사례를 의미를 보존하는 변형을 통해 변환합니다. 변형된 코드 조각은 원본 벤치마크와 구문적으로 다르지만 동일한 실행 동작을 가집니다. 이러한 변환에서는 변수 명명과 코드 구조 수준의 세 가지 핵심 구조(대입, 조건 분기, 반복)에 중점을 둡니다.

- **Performance Highlights**: 평가 결과, 동적 벤치마크에서 모든 모델의 성능이 원래 벤치마크보다 최대 40%까지 감소했습니다. 또한, 동적 벤치마크는 검증된 모델 평가에 여전히 유용함이 입증되었습니다. 이 연구는 데이터 오염 문제에 저항할 수 있는 평가 방법론을 제시하여 모델의 진정한 이해 및 추론 능력을 측정할 수 있도록 합니다.



### Revisiting Early Detection of Sexual Predators via Turn-level Optimization (https://arxiv.org/abs/2503.06627)
Comments:
          Accepted as a main conference paper at NAACL 2025

- **What's New**: 이 연구에서는 온라인 grooming(그루밍)을 탐지하기 위해 speed control reinforcement learning (SCoRL) 방법을 제안합니다. SCoRL는 Luring Communication Theory (LCT)를 기반으로 하여, 대화의 각 턴에서 위험도를 평가합니다. 이 방법은 보다 세밀한 턴 수준의 위험 레이블을 활용하여 임박한 위험을 미리 탐지하고 최적의 개입 시점을 파악하는 데 초점을 맞추고 있습니다.

- **Technical Details**: SCoRL에서는 턴 수준의 위험 레이블을 기반으로 하고, 속도 제어 보상 함수(speed control reward function)를 설계하여 속도와 정확성 사이의 균형을 맞추고 있습니다. 새로운 벤치마크인 Turn-Level eSPD를 도입하여, 기존의 채팅 수준 메트릭스의 한계를 극복하고 턴 수준의 위험 요소를 평가할 수 있도록 했습니다. 이를 통해 SCoRL은 더 효과적이고 실시간으로 탐지할 수 있는 성능을 보입니다.

- **Performance Highlights**: 실험 결과, SCoRL이 기존의 방법들보다 eSPD 성능에서 유의미한 향상을 보여주었습니다. 실제 온라인 grooming 대화에 대한 분석을 통해, SCoRL 모델이 정량적 성과뿐만 아니라 직관적으로 조기 개입의 optimal points를 식별하는 데에도 효과적임을 확인했습니다. 또한, 이전 메트릭의 허점에도 불구하고 보다 정확한 평가가 가능해졌습니다.



### Multimodal Programming in Computer Science with Interactive Assistance Powered by Large Language Mod (https://arxiv.org/abs/2503.06552)
Comments:
          Accepted in Proceedings of the 27th International Conference on. Human-Computer Interaction, 2025

- **What's New**: 이번 연구에서는 DeepSeek R1 기반의 대화형 숙제 도움 시스템을 개발하여 대규모 컴퓨터 과학 입문 프로그래밍 과정 수강생들에게 처음으로 구현하였습니다. 이 시스템은 잘 알려진 코드 편집기에서의 도움 버튼과 커맨드라인 자동 평가기 내 피드백 옵션을 통합하여 운영됩니다.

- **Technical Details**: 학생의 작업을 개인화된 프롬프트로 감싸 교육 목표를 지원하며 즉각적인 답변을 제공하지 않습니다. 본 시스템은 학생들이 겪는 개념적인 어려움을 이해하고 교육적으로 적절한 방식으로 아이디어, 계획 및 템플릿 코드를 제공하는 능력을 가지고 있습니다.

- **Performance Highlights**: 그러나 시스템은 때때로 올바른 학생 코드를 잘못 라벨링하거나 수업에 적합하지 않은 방법을 제안하는 등의 오류를 범할 수 있습니다. 이는 학생들에게 긴장되고 헷갈리는 경험을 초래할 수 있으며, 시스템의 발전 및 배포 관련 여러 문제에 대해 논의한 후 향후 행동 방안을 제시합니다.



### Less is More: Adaptive Program Repair with Bug Localization and Preference Learning (https://arxiv.org/abs/2503.06510)
Comments:
          accepted by AAAI2025 Oral

- **What's New**: 이 논문에서는 Adaptive Program Repair (AdaPR)이라는 새로운 작업을 제안한다. 기존의 Automated Program Repair (APR) 연구는 주로 올바른 패치를 생성하는 데 집중했지만, 수정된 코드와 원본의 일관성을 무시하는 경우가 많았다. AdaPR은 최소한의 수정으로 버그를 고치고, 원래 코드와의 일관성을 유지하는 것을 목표로 한다.

- **Technical Details**: 제안된 방법은 두 단계로 구성된 AdaPatcher를 사용하여, 버그 위치를 정확히 찾아내는 Bug Locator와 수정된 코드의 일관성을 유지하는 Program Modifier로 구분된다. Bug Locator는 LLM을 이용한 자기 디버깅 학습을 통해 버그의 위치를 식별하고, Program Modifier는 위치 인식 수리 학습, 하이브리드 훈련 전략, 적응 선호 학습을 활용하여 최소 수정으로 패치를 생성한다.

- **Performance Highlights**: 실험 결과, 제안된 AdaPatcher는 여러 기준 모델들에 비해 성능이 크게 향상되었음을 보였다. 이 두 단계 접근법은 AdaPR 작업의 효과성을 입증하며, 패치 생성 과정에서의 일관성과 정확성을 모두 고려할 수 있는 가능성을 보여준다. 따라서, 이 연구는 실제 소프트웨어 개발 분야에서 더 실용적이고 적용 가능한 패치 생성을 지원한다.



### SKG-LLM: Developing a Mathematical Model for Stroke Knowledge Graph Construction Using Large Language Models (https://arxiv.org/abs/2503.06475)
- **What's New**: 이번 연구에서는 SKG-LLM을 소개합니다. SKG-LLM은 뇌졸중 관련 논문에서 출발하여 지식 그래프(knowledge graph, KG)를 구축하는 데 수학적 모델과 대형 언어 모델(large language models, LLMs)을 활용합니다. 이 방법을 통해 생물 의학 문헌에서 복잡한 관계를 추출하고 정리하여 뇌졸중 연구에서 KG의 정확성과 깊이를 향상시킵니다.

- **Technical Details**: 제안된 방법에서는 데이터 전처리(data pre-processing)에 GPT-4를 사용하였으며, KG 구축 과정 전반에 걸쳐 임베딩(extraction of embeddings) 추출 또한 GPT-4를 통해 이루어졌습니다. 제안된 모델의 성능은 정밀도(Precision)와 재현율(Recall) 두 가지 평가 기준으로 테스트되었고, 추가 검증에서도 GPT-4가 활용되었습니다.

- **Performance Highlights**: Wikidata 및 WN18RR와 비교했을 때, SKG-LLM 접근 방식은 정밀도와 재현율 면에서 더욱 우수한 성능을 보였습니다. SKG-LLM 모델은 전처리 과정에서 GPT-4를 포함하여 0.906의 정밀도 점수와 0.923의 재현율 점수를 달성했습니다. 전문가 리뷰를 통해 결과를 추가 개선함으로써 정밀도는 0.923, 재현율은 0.918로 증가하였습니다. SKG-LLM에 의해 구축된 지식 그래프는 2692개의 노드와 5012개의 엣지를 포함하고 있으며, 13종의 고유한 노드와 24종의 엣자를 지니고 있습니다.



### HuixiangDou2: A Robustly Optimized GraphRAG Approach (https://arxiv.org/abs/2503.06474)
Comments:
          11 pages

- **What's New**: 이 논문에서는 HuixiangDou2라는 새로운 GraphRAG 프레임워크를 소개합니다. 이 프레임워크는 지식 집약적인 도메인에서 LLM(대규모 언어 모델)의 성능을 극대화하기 위해 설계되었습니다. 기존 GraphRAG 방법론을 통합하여 효율성과 정확성을 높이는 데 초점을 맞추고 있습니다. 또한, 새로운 다단계 검증 메커니즘을 도입하여 계산 비용을 증가시키지 않고도 검색의 강인성을 향상시킵니다.

- **Technical Details**: HuixiangDou2는 쿼리 분해 및 두 수준 검색과 같은 다양한 검색 메커니즘을 평가하고 최적화하는 데 중점을 둡니다. 이 프레임워크는 구조화된 지식 표현을 이용하여 동적 검색을 가능하게 하며, 알고리즘은 데이터 세트에서의 성능 향상을 입증하기 위한 체계적인 실험을 기반으로 합니다. 알고리즘 1에 설명된 인덱싱, 검색 및 생성 프로세스는 성능을 결정짓는 주요 요소입니다.

- **Performance Highlights**: Qwen2.5-7B-Instruct 모델의 경우 초기 성능이 60점에서 74.5점으로 개선되는 등 유의미한 성능 향상을 달성하였습니다. 이는 기존 LLM이 어려움을 겪던 복잡한 쿼리 구조에서도 정확한 검색을 가능하게 함을 보여줍니다. 후보 데이터셋에서 실험이 이루어졌으며, 두 수준 검색이 불확실성 일치(fuzzy matching)을 향상시키고 논리 기반 검색이 구조적 추론을 개선하는 데 기여했습니다.



### Think Twice, Click Once: Enhancing GUI Grounding via Fast and Slow Systems (https://arxiv.org/abs/2503.06470)
- **What's New**: 이번 논문에서는 Focus라는 새로운 GUI grounding 프레임워크를 소개합니다. 이 프레임워크는 속도와 분석을 결합하여 작업의 복잡성에 따라 빠른 예측과 체계적인 분석을 동적으로 전환합니다. 이러한 접근 방식은 기존의 빠른 예측 방법이 가지던 복잡한 인터페이스 이해의 한계를 극복하고자 하며, 이는 인간의 두 가지 사고 시스템에서 영감을 받았습니다.

- **Technical Details**: Focus 프레임워크는 GUI grounding을 세 가지 단계로 세분화하여 각 단계에서 인터페이스 요약, 집중 분석 및 정밀 좌표 예측을 수행합니다. 이 프로세스는 복잡한 인터페이스와 시각적 관계를 체계적으로 이해하는 데 도움을 줍니다. Focus는 300K의 훈련 데이터를 사용하여 2B 파라미터 모델을 통해 기초적인 성능을 지속적으로 개선해왔습니다.

- **Performance Highlights**: Focus는 특히 복잡한 GUI 시나리오에서 탁월한 성능을 보여줍니다. ScreenSpot 데이터세트에서 평균 77.4%의 정확도를 기록했으며, 더 어려운 ScreenSpot-Pro에서 13.3% 개선된 결과를 보였습니다. 이번 결과는 Focus의 이중 시스템 접근 방식이 복잡한 GUI 상호작용 시나리오를 개선하는 데 잠재력이 있음을 보여줍니다.



### TI-JEPA: An Innovative Energy-based Joint Embedding Strategy for Text-Image Multimodal Systems (https://arxiv.org/abs/2503.06380)
- **What's New**: 본 연구에서는 인공지능의 다중모달 정렬에 대한 새로운 접근법인 Text-Image Joint Embedding Predictive Architecture (TI-JEPA)를 소개합니다. TI-JEPA는 에너지 기반 모델 (Energy-based Model, EBM) 프레임워크를 활용하여 텍스트와 이미지 간의 복잡한 관계를 포착하고, 멀티모달 감정 분석(task)과 같은 다양한 멀티모달 기반 작업에서 기존의 사전 훈련 방법론보다 우수한 성능을 보여줍니다. 이러한 접근법은 다운스트림 애플리케이션에 상당한 개선을 제안합니다.

- **Technical Details**: TI-JEPA는 크로스 어텐션 메커니즘을 통합하여 텍스트와 시각 정보를 정렬하는 방식으로 작동합니다. 명시된 아키텍처에는 이미지 인코더와 텍스트 인코더가 포함되어 있으며, 이러한 인코더는 각각의 모달리티에서 임베딩 벡터를 생성하여 성과를 낼 수 있도록 지원합니다. 또한, TI-JEPA는 타겟 및 컨텍스트 임베딩을 생성하기 위한 두 단계의 접근법을 설계하여, 각각의 이미지를 패치(patch)로 분할해 처리합니다.

- **Performance Highlights**: TI-JEPA는 다양한 텍스트-이미지 정렬 벤치마크에서 최첨단 성능을 기록하였으며, 특히 감정 분석(task)에서 정확도와 F1 스코어에서 우수한 결과를 보여주었습니다. 이러한 성과는 TI-JEPA의 유연하고 동적인 멀티모달 학습 프레임워크 덕분에 가능했으며, 이는 다양한 다운스트림 작업에서도 효율적인 성능을 발휘하는 데 도움을 줍니다. 이 연구는 또한 에너지 기반 프레임워크를 활용해 멀티모달 융합의 가능성을 여는 데 기여합니다.



### General Scales Unlock AI Evaluation with Explanatory and Predictive Power (https://arxiv.org/abs/2503.06378)
- **What's New**: 이 논문에서는 AI 평가를 위한 일반적인 척도(general scales)를 소개하며, 기존 AI 벤치마크가 실제로 무엇을 측정하는지 설명하고, AI 시스템의 능력 프로파일(ability profiles)을 추출하며, 새로운 작업 인스턴스에 대한 성능을 예측하는 방법을 제안합니다. 18개 새로운 '요구 수준 주석' (demand-level-annotation) 기준을 통해 모든 테스트 인스턴스에 적용 가능하며, AI의 안전하고 효과적인 사용을 위한 안정적인 배포를 지원합니다. 또한, 새로운 방법론을 통해 AI 시스템의 다양성과 복잡성을 효과적으로 다룰 수 있는 가능성을 제시합니다.

- **Technical Details**: 이 논문에서 제안하는 방법론은 18개의 정성적인 기준을 활용하여 AI의 과제를 정의하고 평가 기준의 객관성을 높이며, 자동화된 방식으로 실험됩니다. 각 요구 수준은 0에서 무한대까지를 아우르며, AI 시스템의 능력을 측정할 수 있는 상위 개념으로 자리 잡습니다. 이를 통해 AI 시스템의 성능 예측이 가능해지며, 특히 새로운 작업이나 벤치마크에 대해서도 높은 예측력을 발휘합니다.

- **Performance Highlights**: 실험 결과, 다양한 대형 언어 모델에 대해 18개의 능력 추정치가 생성되며, 이는 AI 시스템의 강점과 약점을 명확하게 파악할 수 있는 이점을 제공합니다. 특히, 기존의 블랙박스 기반 예측 모델과 비교했을 때, 인스턴스 수준에서보다 우수한 성능을 발휘했습니다. 이러한 결과는 AI 모델의 선택 및 안전한 운영 영역을 정의하는 데에 유용한 가능성을 엽니다.



### Phraselette: A Poet's Procedural Pa (https://arxiv.org/abs/2503.06335)
- **What's New**: 이번 논문에서는 기존 자동 쓰기 도구들의 규범적 기초가 작가의 가치와 잘 맞지 않음을 지적하며, 텍스트 소재 발굴, 처리 및 변형을 유연하게 지원하는 실험적 시를 위한 작성 지원 도구 'Phraselette'를 소개합니다. Phraselette는 작가의 의도에 따라 텍스트 자료를 제공하여 작가의 창의적인 가능성을 확장하는 것을 목표로 하고 있습니다. 이러한 새 도구는 기존의 기존 LLM 기반 애플리케이션의 자동성을 반대하는 방식으로 설계되었습니다.

- **Technical Details**: Phraselette는 기계 학습 언어 모델의 제안 기능을 활용하지만, 사용자가 요청한 특정 지점에서만 텍스트 자료를 제공합니다. 이 도구는 기존의 쓰기 지원 도구와 달리 단어 검색뿐만 아니라 문구 검색도 지원하여 작가의 창의성을 증가시킵니다. 텍스트의 맥락에 따라 제공되는 검색 기능은 다양한 창작 통제 방안을 제공합니다.

- **Performance Highlights**: 10명의 발간된 시인에 대한 전문가 평가 결과, Phraselette는 실제로 유용하며 문서의 소유권을 부정적으로 영향을 미치지 않는 것으로 나타났습니다. 나아가, Phraselette의 설계 철학은 시와 예술 제작 이론과 잘 통합되어 있음을 보여주며, 작가의 창작 과정과의 정렬을 강조하는 데 기여하고 있습니다.



### Advancing Autonomous Vehicle Intelligence: Deep Learning and Multimodal LLM for Traffic Sign Recognition and Robust Lane Detection (https://arxiv.org/abs/2503.06313)
Comments:
          11 pages, 9 figures

- **What's New**: 이 논문에서는 자율주행차(AV)의 안전한 내비게이션을 위한 고급 심층 학습 기법과 다중모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)을 결합한 통합 접근 방식을 소개합니다. 이 접근 방식은 복잡하고 동적인 환경에서 도로 감지를 향상시키기 위한 것입니다. 특히, 다양한 기법을 평가하여 교통 표지판 인식에서 최고의 성능을 달성하고, 경량화된 MLLM 기반 프레임워크를 제안하여 소규모 데이터셋으로 직접적인 학습 조정을 가능하게 했습니다.

- **Technical Details**: 교통 표지판 인식에는 ResNet-50, YOLOv8, RT-DETR을 평가하여 각각 99.8%, 98.0%, 96.6%의 높은 정확도를 달성했습니다. 차선 감지에서는 CNN 기반의 세그멘테이션 방법에 다항 곡선 맞춤(polygonal curve fitting)을 추가하여 높은 정확도를 이끌어냈습니다. 제안된 MLLM 기반 프레임워크는 초기 사전 훈련 없이 다양한 차선 유형과 복잡한 교차로를 효과적으로 처리하여 차선 검출의 신뢰성을 높였습니다.

- **Performance Highlights**: 이 다중모달 접근법은 훈련 자원의 제약에도 불구하고, 다양한 조건에서도 뛰어난 추론 능력을 보여줍니다. 명확한 조건에서는 99.6%, 야간에는 93.0%의 차선 검출 정확도를 달성했고, 비 오는 날의 차선 불가시(88.4%)나 도로 노후화(95.6%)에 대한 추론에서도 탁월한 성능을 보였습니다. 이 포괄적인 프레임워크는 자율주행차의 인식 신뢰성을 크게 향상시켜, 다양한 도로 시나리오에서 안전한 자율주행에 기여합니다.



### Critical Foreign Policy Decisions (CFPD)-Benchmark: Measuring Diplomatic Preferences in Large Language Models (https://arxiv.org/abs/2503.06263)
- **What's New**: 이번 연구는 인공지능(AI)과 국가 안전 보장 간의 통합이 증가함에 따라 대형 언어 모델(LLM)의 편향을 평가하기 위한 독창적인 벤치마크를 제안합니다. 연구는 국제 관계(International Relations, IR)와 관련된 400개의 전문가 제작 시나리오를 이용해 여러 주요 모델의 편향과 선호도를 분석했습니다. 이를 통해 Qwen2 72B, Gemini 1.5 Pro-002, Llama 3.1 8B Instruct 모델이 다른 모델보다 군사적 에스컬레이션을 강화하는 추천을 더 많이 제공한다는 점을 발견했습니다.

- **Technical Details**: 이 연구에서는 군사 에스컬레이션, 군사 및 인도적 개입, 국제 시스템 내 협력적 행동, 동맹 역학과 같은 네 가지 주제를 중심으로 모델들의 추천을 분석했습니다. 각 모델은 특정 국가에 대한 편향을 보이며, 예를 들어 중국과 러시아에 대해서는 덜 에스컬레이션하고 개입적인 행동을 추천하는 경향이 있음을 관찰했습니다. 이 연구는 AI 모델의 응용이 부정적인 위험 없이 이루어지도록 하는 점검이 필수적임을 강조합니다.

- **Performance Highlights**: 모델 응답에서 중요한 차이가 나타났으며, 특히 에스컬레이션과 개입 분야에서 두드러진 변화를 보였습니다. 모델들은 편향된 행동을 보여주며, 특정 국가들에 대해 다르게 반응하는 것을 발견했습니다. 연구 결과는 LLM을 고위험 국가 안보 및 외교 정책 시나리오에 배치하는 것의 위험성을 강조하며, 향후 연구에서 도메인별 벤치마킹 및 평가가 중요하다고 강조합니다.



### A Noise-Robust Turn-Taking System for Real-World Dialogue Robots: A Field Experimen (https://arxiv.org/abs/2503.06241)
- **What's New**: 이 논문에서는 턴 테이킹(turn-taking)에 중점을 두어, 대화 로봇에서 실시간으로 대화를 활성화하기 위한 노이즈 로버스트(noise-robust) 음성 활동 프로젝션(Voice Activity Projection, VAP) 모델을 제안합니다. 이전 연구에서는 통제된 환경에서 턴 테이킹 모델을 탐구했으나, 실제 환경에서의 강건성은 충분히 조명되지 않았습니다. 우리는 일본의 한 쇼핑몰에서 제안된 시스템의 효과를 평가하기 위해 실험을 수행했습니다.

- **Technical Details**: VAP 모델은 Transformer 아키텍처에 기반하여 설계되었으며, 음성 입력을 통해 대화 참여자의 미래 음성 활동을 예측합니다. 모델은 고정된 오디오의 파형을 직접 예측하여 음성 인식 오류에 강한 내성을 가지며, 실시간 처리를 위한 경량 설계로 구성되어 있습니다. 이 연구에서는 다양한 노이즈 환경에서 훈련된 VAP 모델을 통해 품질 향상을 도모하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 VAP 시스템은 응답 지연을 유의미하게 줄여 더 자연스러운 대화를 만들어내며, 사용자가 더 빠르게 반응하는 것을 관찰할 수 있었습니다. 사용자 평가에서 더 빠른 응답이 상호작용 경험을 개선한다는 결과를 확인했습니다. 이러한 연구는 새로운 평가 방법을 통해 대화 로봇의 상호작용 품질 개선을 보여주는 혁신적인 접근 방식이었습니다.



### Explainable Synthetic Image Detection through Diffusion Timestep Ensembling (https://arxiv.org/abs/2503.06201)
Comments:
          13 pages, 5 figures

- **What's New**: 이 연구에서는 최근 발전한 diffusion 모델이 생성한 이미지의 식별을 위한 새로운 방법을 제안합니다. 기존의 감지 방법이 진품 사진과 합성 이미지를 구분하는 데 어려움을 겪고 있는 가운데, 본 논문은 이미지의 고주파 성분에서 발생하는 차이를 활용하여 합성 이미지를 효과적으로 탐지할 수 있는 가능성을 보여줍니다. 또한, 이는 추론 기능을 추가하여 AI 생성 이미지의 결함을 식별하는 데도 기여합니다.

- **Technical Details**: ESIDE(Explainable Synthetic Image Detection through Diffusion Timestep Ensembling)라는 프레임워크를 제안하며, 이는 여러 타임스탬프에서 노이즈가 추가된 원본 이미지를 통해 분류기를 훈련시키는 방식을 채택하고 있습니다. 해당 프레임워크는 DDIM 반전 과정을 통해 생성된 여러 노이즈 버전을 이용하여 이미지를 처리하며, 이들 노이즈 이미지는 CLIP 이미지 인코더를 통해 특징 표현을 추출한 후 AdaBoost 모델에 입력됩니다. 이를 통해 각 모델은 이미지가 합성인지의 여부를 평가하며, 최종 예측은 이러한 평가를 기반으로 한 가중 합산으로 도출됩니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 GenImage 기준으로 정규 샘플에서 98.91%, 더 어려운 샘플에서 95.89%의 높은 탐지 정확도를 기록하며, 기존 방법보다 각각 2.51%, 3.46% 향상된 성능을 보였습니다. 또한, 이 방법은 다른 diffusion 모델이 생성한 이미지에 대해서도 효과적으로 일반화되는 특성을 가지고 있습니다. 공급된 데이터셋인 GenHard와 GenExplain은 더욱 높은 난이도의 탐지 샘플과 AI 생성 이미지에 대한 고품질 설명을 제공합니다.



### AF-KAN: Activation Function-Based Kolmogorov-Arnold Networks for Efficient Representation Learning (https://arxiv.org/abs/2503.06112)
Comments:
          25 pages

- **What's New**: 이 논문에서는 ReLU-KAN을 기반으로 한 Activation Function-Based Kolmogorov-Arnold Networks (AF-KAN)를 제안합니다. AF-KAN은 다양한 활성화 함수 조합을 통해 데이터 표현 학습을 개선하고, 주의 메커니즘(attention mechanisms)과 데이터 정규화(data normalization) 기법을 통합하여 네트워크의 매개변수 수를 낮추는 것을 목표로 합니다. 이를 통해 AF-KAN은 이미지 분류 데이터셋에서 MLP 및 기존 KAN 모델에 비해 성능이 향상된 결과를 보여주었습니다.

- **Technical Details**: AF-KAN은 ReLU-KAN의 구조를 개선하여 ReLU 외에도 다양한 활성화 함수(function)와 그 조합을 사용합니다. 모델 훈련 전에 L2 norm과 min-max scaling을 결합한 정규화(normalization) 방식을 도입하여 데이터의 범위를 관리합니다. 또한, AF-KAN은 선형 변환 전에 사전 선형 정규화(pre-linear normalization)를 실행하여 모델 성능을 향상시킵니다.

- **Performance Highlights**: AF-KAN은 실험 결과 MLP, ReLU-KAN, 그리고 동일한 매개변수를 가진 다른 KAN보다 상당히 뛰어난 성능을 보였습니다. AF-KAN은 같은 네트워크 구조를 유지하면서도 매개변수를 줄이는데 성공하며, 학습 시간은 길어지지만, FLOPs의 수치가 증가하게 됩니다. 또한 AF-KAN은 6배에서 10배 이하의 매개변수로도 경쟁력을 유지하는 것으로 나타났습니다.



### A Novel Trustworthy Video Summarization Algorithm Through a Mixture of LoRA Experts (https://arxiv.org/abs/2503.06064)
- **What's New**: 이번 연구에서는 사용자 생성 콘텐츠가 급증하는 비디오 공유 플랫폼에서 비디오의 효율적인 검색과 탐색을 위한 새로운 접근 방식이 제안됩니다. MiLoRA-ViSum은 비디오 데이터의 복잡한 시간적 동역학과 공간적 관계를 효율적으로 포착하기 위해 설계된 새로운 모델입니다. 기존의 Video-llama에서 발생하는 자원 소모 문제를 해결하기 위해 저자들은 파라미터 수를 조절할 수 있는 방법을 모색했습니다.

- **Technical Details**: MiLoRA-ViSum은 전통적인 Low-Rank Adaptation (LoRA)을 혼합 전문가(mixture-of-experts) 패러다임으로 확장하여 시간적 및 공간적 적응 메커니즘을 통합합니다. 이 모델은 각각의 LoRA 전문가가 특정 시간적 또는 공간적 차원에 맞춰 조정되어 비디오 요약 작업을 수행합니다. 이 동적 통합 방식은 복잡한 비디오 데이터의 다양한 특성을 효과적으로 다루기 위해 특별히 설계되었습니다.

- **Performance Highlights**: 비교 평가에서 MiLoRA-ViSum은 VideoXum와 ActivityNet 데이터셋에서 최신 모델들과 비교해 최고의 요약 성능을 발휘했습니다. 또한, 다른 모델들에 비해 현저하게 낮은 계산 비용을 유지하면서도 효과성을 보장합니다. 혼합 전문가 전략과 이중 적응 메커니즘은 비디오 요약 기능을 향상시킬 수 있는 잠재력을 강조합니다.



### DSGBench: A Diverse Strategic Game Benchmark for Evaluating LLM-based Agents in Complex Decision-Making Environments (https://arxiv.org/abs/2503.06047)
Comments:
          43 pages, 5 figures, conference

- **What's New**: DSGBench는 전략적 의사결정에 대한 평가를 위한 새로운 플랫폼으로, 최신 LLM 기반 에이전트를 평가하기 위해 설계되었습니다. 이 플랫폼은 복잡한 전략 게임 6종을 포함하여, 장기적이며 다차원적인 의사결정 요구를 충족합니다. DSGBench는 다양한 난이도를 설정하고 다수의 목표를 지원하여 더 맞춤화된 평가를 가능하게 합니다.

- **Technical Details**: DSGBench는 다섯 가지 특정 차원에서의 성능을 분석하는 세분화된 평가 점수 체계를 사용하여 의사결정 능력을 평가합니다. 또한, 자동화된 의사결정 추적 메커니즘을 도입하여 에이전트의 행동 패턴과 전략의 변화를 심층적으로 분석할 수 있습니다. 이는 복잡한 결정작업에 대한 LLM 에이전트의 실제 능력을 정밀하게 검토하는 데 도움을 줍니다.

- **Performance Highlights**: DSGBench를 여러 인기 있는 LLM 기반 에이전트에 적용하여 이 플랫폼이 제공하는 귀중한 통찰을 입증했습니다. 이 연구 결과는 전략 결정 작업에서 LLM 기반 에이전트를 선택하는 데 유용하며, 향후 개발을 개선하는 데 기여할 수 있음을 보여줍니다. DSGBench는 [이 URL](https://)에서 이용할 수 있습니다.



### Rank-R1: Enhancing Reasoning in LLM-based Document Rerankers via Reinforcement Learning (https://arxiv.org/abs/2503.06034)
- **What's New**: 논문에서는 Rank-R1이라는 새로운 LLM(large language model) 기반의 재정렬기가 사용자 쿼리와 후보 문서에 대한 추론을 수행한 후 문서를 정렬하는 접근 방식을 소개합니다. 기존의 문서 재정렬 방법은 LLM을 프롬프트(prompt)하거나 미세 조정(fine-tuning)하여 관련성에 따라 후보 문서를 배치하는 데 의존합니다. 반면, Rank-R1은 소규모의 관련성 라벨을 활용하여 강화 학습(reinforcement learning) 알고리즘을 사용하여 LLM 재정렬기의 추론 능력을 향상시킵니다.

- **Technical Details**: Rank-R1은 DeepSeek의 강화 학습 프레임워크를 바탕으로 LLM 기반 문서 재정렬기의 추론 능력을 강화합니다. 이 방법론은 Setwise prompting 접근법을 채택하여 쿼리와 후보 문서 세트를 입력으로 받아 LLM이 가장 관련성이 높은 문서를 선택하도록 유도합니다. 원래 Setwise 접근법은 쿼리와 문서 간의 관련성에 대한 추론을 유도하지 않지만, Rank-R1은 추론 지침을 추가해 문서의 관련성을 이해하는 능력을 향상시킵니다.

- **Performance Highlights**: TREC DL 및 BRIGHT 데이터셋을 통해 Rank-R1의 실험 결과, 복잡한 쿼리에 대해 매우 효과적임을 확인했습니다. 특히 인도메인 데이터셋에서는 감독 미세 조정 방법과 동등한 성능을 보이며, 미세 조정 방법이 사용하는 데이터의 18%만으로도 충분한 성능을 발휘합니다. 또한, Rank-R1은 복잡한 쿼리를 포함하는 아웃오브도메인 데이터셋에서도 기존 방법보다 뛰어난 성능을 보였습니다.



### Psycholinguistic Analyses in Software Engineering Text: A Systematic Literature Review (https://arxiv.org/abs/2503.05992)
- **What's New**: 본 연구는 소프트웨어 공학(Software Engineering, SE)에서 심리언어학 도구의 중요성을 분석하고, Linguistic Inquiry and Word Count (LIWC) 사용 사례를 포괄적으로 정리한 것을 발표합니다. LIWC는 개발자 팀의 커뮤니케이션을 분석하고 감정을 감지하며, AI 생성 텍스트와 인간 작성 텍스트를 비교하는 등의 다양한 용도로 활용되고 있습니다. 또한, LIWC의 SE 특정 어휘 처리의 어려움과 같은 한계도 논의되고 있습니다.

- **Technical Details**: 소프트웨어 공학에서 LIWC는 심리학적 구성 개념을 분석하는 데 중요한 역할을 합니다. 연구자들은 LIWC를 통해 팀의 동학, 협업, 동기 부여 및 개발자 생산성과 같은 행동 소프트웨어 공학(Behavioral Software Engineering, BSE) 개념을 연구하였습니다. LIWC는 Q&A 게시물, IRC 채널의 개발자 커뮤니케이션, 이메일 아카이브 등 다양한 텍스트 데이터 유형을 분석하는 데 사용됩니다.

- **Performance Highlights**: LIWC는 43개의 관련 논문에서 사용되었으며, 17개의 연구는 LIWC의 효과성을 공식적으로 평가했습니다. 그러나 43개 논문 중 26개는 LIWC의 형식적인 평가를 수행하지 않았습니다. LIWC의 사용은 팀 관리, 감성 분석, 그리고 Stack Overflow와 같은 플랫폼의 게시물 품질 평가 등에서 광범위하게 활용되며, 도구의 한계를 극복해 나가는 방향으로의 발전 가능성도 시사하고 있습니다.



### Bimodal Connection Attention Fusion for Speech Emotion Recognition (https://arxiv.org/abs/2503.05858)
- **What's New**: 이번 연구에서는 Bimodal Connection Attention Fusion (BCAF)이라는 새로운 방법을 제안합니다. 이 방법은 오디오 및 텍스트 간의 상호작용을 모델링하기 위한 인터랙티브 연결 네트워크와 bimodal attention 네트워크, 상관관계 주의 네트워크로 구성되어 있습니다. 이 방법은 감정 인식 시스템의 성능을 향상시키기 위해 모달리티 간의 연결 및 상호작용을 효과적으로 포착하는 데 중점을 둡니다.

- **Technical Details**: BCAF 아키텍처는 유니 모달 표현 모듈, 연결 주의 융합 모듈, 분류 모듈로 구성되어 있습니다. 오디오 모달리티 인코더로는 wav2vec 모델이 사용되어 1024차원 오디오 표현을 추출합니다. 텍스트 모달리티에는 RoBERTa 모델을 사용하여 1024차원 텍스트 표현을 생성하며, 이러한 표현을 통해 서로 다른 모달리티 간의 세밀한 관계를 학습합니다.

- **Performance Highlights**: MELD 및 IEMOCAP 데이터셋에서의 실험 결과, 제안한 BCAF 방법이 기존의 최첨단 기준을 능가하는 성능을 보여주었습니다. BCAF는 독립적인 모달리티와 융합 모델을 동시에 적용하여 감정 인식의 정확도와 강건성을 높였습니다. 이러한 성능 향상은 감정 인식 시스템의 실제 애플리케이션에서의 활용 가능성을 높입니다.



### MedSimAI: Simulation and Formative Feedback Generation to Enhance Deliberate Practice in Medical Education (https://arxiv.org/abs/2503.05793)
- **What's New**: MedSimAI는 AI가 지원하는 시뮬레이션 플랫폼으로, 의사-환자 커뮤니케이션에 대한 임상 기술 훈련을 혁신적으로 변화시킵니다. 이 플랫폼은 대화형 환자 만남을 통해 고의적인 연습(deliberate practice)과 자기주도 학습(self-regulated learning, SRL)을 허용하며, 자동화된 평가를 제공합니다. MedSimAI는 대형 언어 모델(large language models, LLMs)을 활용하여 현실적인 임상 상호작용을 생성하고, 의료 평가 기준에 따라 구조화된 즉각적인 피드백을 제공합니다.

- **Technical Details**: MedSimAI는 전문가들 간의 협업을 통해 설계되었으며, 임상 기술의 기초 역량을 연습할 수 있는 AI 표준화 환자(AI standardized patients) 시뮬레이션 프레임워크를 포함합니다. 이 플랫폼은 즉각적인 피드백을 제공하기 위한 유연한 평가 프레임워크와 임상 체크리스트를 통해 교육적 요구를 충족할 수 있도록 설계되었습니다. 특히 SRL 원칙을 통합하여 학습자들이 설정한 목표에 따라 사용할 수 있는 다양한 임상 시나리오를 제공합니다.

- **Performance Highlights**: 파일럿 연구에서 104명의 1학년 의대생들이 MedSimAI를 통해 환자 이력 취득을 반복적으로 연습하는 데 도움을 받았습니다. 학생들은 이 플랫폼이 실질적이고 반복 가능한 훈련을 제공한다고 평가했으나, 일부 고차원 기술이 간과되는 경향을 보였습니다. 그러나 전반적으로 학생들은 체계적인 역사 취득과 공감적 경청에 높은 성과를 보여, MedSimAI가 전통적인 시뮬레이션 기반 교육의 주요 한계를 해결할 수 있는 가능성을 입증했습니다.



### Emergent Abilities in Large Language Models: A Survey (https://arxiv.org/abs/2503.05788)
- **What's New**: 이 논문은 인공지능 일반 지능(Artificial General Intelligence)을 향한 새로운 기술 혁명을 이끌고 있는 대형 언어 모델(Large Language Models, LLMs)의 급증에 대한 이해를 심화시키고자 한다. 최근 LLMs는 여러 가지 예상치 못한 'emergent abilities'를 나타내어 과학적 논쟁의 중심에 서게 되었다. 이러한 emergent abilities는 외부 요인, 훈련 동역학(training dynamics), 과제 유형 등에 따라 달라질 수 있을지에 대한 의문을 제기하며, 이 논문은 그 복잡성을 파악하기 위한 종합적인 검토를 제공한다.

- **Technical Details**: 논문은 emergent abilities라는 개념을 비판적으로 분석하고, 이를 바탕으로 LLMs 및 대형 추론 모델(Large Reasoning Models, LRMs)의 성능을 결정짓는 조건들을 조사한다. 이 연구에서는 스케일링 법칙(scaling laws), 과제 복잡성(task complexity), 사전 훈련 손실(pre-training loss) 등의 요소를 분석하여 emergent abilities가 나타나는 맥락을 규명한다. 또, 기존 LLMs의 한계를 넘어서기 위해 강화 학습(reinforcement learning)과 추론 시간 탐색(inference-time search)을 통합한 LRMs를 다룬다.

- **Performance Highlights**: LLMs의 성능은 훈련 데이터의 규모와 모델 파라미터 수에 따라 잘 정의된 관계가 있다. 그러나 특정 과제에서는 모델의 크기와 성능 사이에 비연속적인 관계가 나타나는 등, 예측 불가능한 성과도 존재한다. 이에 따라 emergent abilities는 시스템적 신뢰성과 안전성, 특히 유해한 행동의 예측을 보장하기 위한 핵심 요소로 부각되며, AI 시스템의 안전과 관리에 대한 우려가 커지고 있다.



### Where is my Glass Slipper? AI, Poetry and Ar (https://arxiv.org/abs/2503.05781)
Comments:
          36 pages, 0 figures, I have updated the submission to the correct submission standards apologies. The paper is a Literature Review so there are no formulas or results tables and images

- **What's New**: 이 문헌 리뷰는 인공지능(AI), 시(poetry), 예술(art)之间의 교차점을 탐구하며 디지털 창작 관행에서의 역사적 진화 및 현재의 논의를 포괄적으로 조사합니다. 초기 템플릿 기반 시스템에서 생성 모델(generative models)에 이르기까지 컴퓨터 생성 시의 발전을 추적하며, Turing Test, FACE 모델, ProFTAP 등과 같은 평가 프레임워크를 비판적으로 분석합니다. 이 연구는 AI 생성 텍스트에서 창의성, 의미적 일관성, 그리고 문화적 관련성을 측정하기 위한 노력과 인간의 시적 표현을 복제하는 데 있어 지속적인 도전 과제를 강조합니다.

- **Technical Details**: 리뷰는 AI 기업들이 기술을 인간화하기 위해 사용하는 세련된 언어와 의인화 은유를 분석하는 마케팅 이론(Marketing Theory) 논의를 포함하고 있습니다. 이러한 마케팅 내러티브(narratives)의 환원적 성격과 알고리즘적 정밀성과 인간 경험의 현실 간의 긴장을 드러냅니다. 또한, 이 리뷰는 자신의 구성 과정에 대한 자기 반성적(auto-ethnographic) 설명을 포함하여 저작권 및 객관성에 대한 기존의 개념을 불안정하게 만듭니다.

- **Performance Highlights**: 결국 이 리뷰는 기술 혁신과 인간 주관성(human subjectivity)의 상호 의존성을 인식하는 창작 과정의 재평가를 촉구합니다. 윤리적, 문화적, 철학적 문제를 다루면서 예술 생산의 경계를 재구상해야 한다고 주장합니다. 또한, 이 대화는 학문적 담론에서의 탈구성(deconstruction)과 로고 중심(logocentrism) 가정에 도전하는 내용을 담고 있습니다.



### DreamNet: A Multimodal Framework for Semantic and Emotional Analysis of Sleep Narratives (https://arxiv.org/abs/2503.05778)
Comments:
          10 pages, 5 figures, new research contribution

- **What's New**: 이 논문은 꿈 내러티브(dream narratives)의 체계적인 분석을 위해 인공 지능을 사용한 연구가 부족했던 점을 해결하기 위해 새로운 딥러닝 프레임워크인 DreamNet을 도입합니다. DreamNet은 텍스트 기반의 꿈 보고서에서 의미적 주제와 감정 상태를 추출하며, REM 단계의 EEG 데이터를 통해 능력을 보강할 수 있습니다. 특히, 이 모델은 1500개의 익명 꿈 내러티브로 구성된 데이터셋에서 텍스트 전용 모드에서 92.1%의 정확도와 88.4%의 F1-score를 달성하며, EEG 통합 시에는 99.0%의 정확도와 95.2%의 F1-score로 성능이 향상됩니다.

- **Technical Details**: DreamNet은 RoBERTa를 기반으로 한 transformer 모델로, 꿈 내러티브에서 비행, falling, 추적, 상실과 같은 의미적 주제를 추출합니다. 이 모델은 감정 상태를 나타내는 8개의 이진 벡터를 사용하여 다양한 감정(예: 두려움, 기쁨, 불안, 슬픔 등)을 분류합니다. 또한, DreamNet은 REM 단계의 EEG 데이터를 통합하여 생리적 데이터를 포함한 다중 모드 분석을 구현하며, 이는 텍스트 기반 데이터와 생리적 맥락을 모두 포착할 수 있게 합니다.

- **Performance Highlights**: DreamNet은 텍스트 전용 모드(DNet-T)에서 92.1%의 정확도와 88.4%의 F1-score를 기록했으며, REM 단계의 EEG 데이터 통합 후에는 99.0%의 정확도와 95.2%의 F1-score로 7%의 성능 향상을 보여줍니다. 연구 결과는 정신 건강 진단 및 개인 맞춤형 치료에 활용될 수 있는 잠재력을 가지고 있으며, 꿈과 감정 간의 강한 상관관계(예: falling-anxiety, r = 0.91, p < 0.01)를 보여줍니다. 이 작업은 AI와 심리 연구 간의 교량 역할을 하며, 스케일러블한 도구와 공개 데이터셋을 제공합니다.



### Effect of Gender Fair Job Description on Generative AI Images (https://arxiv.org/abs/2503.05769)
Comments:
          ISCA/ITG Workshop on Diversity in Large Speech and Language Models

- **What's New**: 이 연구는 OpenAI DALL-E 3와 Black Forest FLUX를 사용하여 STEM 직업 이미지에서 성별 표현을 분석하였습니다. 150개의 프롬프트를 사용하여 3개의 언어 형태(독일어 일반 남성형, 독일어 쌍형, 그리고 영어)로 생성된 이미지를 분석한 결과, 모든 형태에서 남성 편향이 두드러졌습니다. 연구 결과는 생성적 인공지능이 사회적 편견을 강화하는 데 기여하고 있다는 점을 강조하며, 다각적인 다양성 논의의 필요성을 시사합니다.

- **Technical Details**: 이 연구에서는 DALL-E와 Black Forest Flux의 두 가지 이미지 생성기를 사용하여 50개의 STEM 직업을 포함한 210개의 이미지를 생성하였습니다. 각 직업에 대해 150개의 STEM 프롬프트와 60개의 사회적 직업에 대한 프롬프트가 사용되었으며, 모든 직업은 세 가지 언어 범주에서 균등하게 표현되었습니다. 개별 이미지의 성별 식별은 세 명의 저자가 독립적으로 수행하였으며, DALL-E와 FLUX의 각 생성 결과를 비교하였습니다.

- **Performance Highlights**: 연구 결과, DALL-E와 FLUX가 생성한 이미지는 성별 및 인종적 다양성이 낮으며, 특히 STEM 분야에서 남성을 과대표현하고 있습니다. 또한, 생성된 이미지의 성별 구별 기준이 대단히 보수적이어서, 성별 확인이 용이하다는 점이 강조되었습니다. 이러한 결과는 생성적 인공지능의 편견을 해결하기 위한 포괄적인 접근이 필요함을 보여줍니다.



### Uncertainty-Aware Fusion: An Ensemble Framework for Mitigating Hallucinations in Large Language Models (https://arxiv.org/abs/2503.05757)
Comments:
          Proceedings of the ACM Web Conference 2025, WWW 25

- **What's New**: 이 연구에서는 LLM(대형 언어 모델)의 허위 출력(hallucination) 문제를 해결하기 위한 새로운 방법으로 Uncertainty-Aware Fusion (UAF)이라는 앙상블 프레임워크를 제안합니다. 최근 연구에 따르면, LLM은 허위 출력을 생성할 가능성을 자체적으로 평가할 수 있는 불확실성 추정(uncertainty estimation)이 가능합니다. UAF는 다양한 LLM의 정확도와 자기 평가 기능을 활용하여 효과적으로 허위 출력을 줄이는 것을 목표로 합니다.

- **Technical Details**: UAF는 두 가지 주요 모듈로 구성됩니다: SELECTOR와 FUSER입니다. SELECTOR는 주어진 성능 메트릭에 따라 N개의 LLM 풀에서 K개의 LLM을 선택하고, FUSER는 선택된 LLM의 출력을 결합하여 최종 결과를 생성합니다. 이러한 구조는 추가적인 학습이나 미세 조정 없이 여러 LLM의 보완적인 강점을 활용하여 총체적인 정확도를 향상시킵니다.

- **Performance Highlights**: UAF는 TruthfulQA, TriviaQA 및 FACTOR와 같은 여러 공개 벤치마크 데이터세트에서 성능이 입증되었습니다. UAF는 기존의 최첨단(상태의 최첨단) 허위 출력 완화 방법보다 8% 향상된 사실 정확도를 보여주었으며, GPT-4와의 성능 차이를 좁히거나 심지어 초과하는 결과를 나타내었습니다.



### ChatWise: AI-Powered Engaging Conversations for Enhancing Senior Cognitive Wellbeing (https://arxiv.org/abs/2503.05740)
- **What's New**: 이번 연구는 노인들을 위해 고안된 LLM 기반의 챗봇 ChatWise를 소개합니다. 이는 기존의 단순한 상호작용을 넘어 중복 대화 지원을 위한 전략적 대화 설계를 목표로 하고 있습니다. ChatWise는 정서적 상태와 대화 맥락을 고려하여 향상된 대화 참여를 제공합니다. 이러한 접근 방식은 노인의 인지 기능을 개선하고 사회적 고립감을 줄이기 위한 효과적인 대안으로 자리 잡고자 합니다.

- **Technical Details**: ChatWise는 이중 수준의 대화 생성 구조를 채택하여, 먼저 매크로 수준의 정보를 도출하여 대화 전략을 제안하고, 이후 사용자 참여를 최대화하기 위한 미세 수준의 발화를 생성합니다. 대화 전략 후보는 실제 원격 건강 관리 임상 시험에서 추출된 대화 행위(Dialogue Acts)와 정서적 지원 데이터 세트에서 가져온 전략을 통합하여 구성되었습니다. 이 접근 방식은 LLM의 고급 추론 능력을 활용하여 구성되었습니다.

- **Performance Highlights**: ChatWise는 다중 회전 대화에서 특히 우수한 성과를 보이며, 사용자의 인지 및 정서 상태를 크게 향상시키는 것으로 입증되었습니다. 디지털 트윈을 활용한 실험 결과, ChatWise는 경도 인지 장애가 있는 사용자에게서도 효과를 보여주었습니다. 또한,  대화 분석 결과, 장기적인 대화 지원이 노인의 정서적 웰빙을 개선하는 데 중요한 역할을 한다는 점이 강조되었습니다.



### That is Unacceptable: the Moral Foundations of Canceling (https://arxiv.org/abs/2503.05720)
- **What's New**: 이번 연구는 Canceling Attitudes Detection (CADE) 데이터셋을 소개하며, 소셜 미디어에서의 취소 태도에 대한 분석을 시도합니다. 이 데이터셋은 사람들의 도덕적 관점이 취소 태도 평가에 미치는 영향을 연구하기 위해 주석이 달린 취소 사례들의 집합으로 구성되어 있습니다. 연구자들은 주석자들이 평가하는 사건에 따라 그들의 도덕적 판단이 어떻게 다르게 나타나는지를 보여줍니다.

- **Technical Details**: CADE 데이터셋은 여섯 명의 유명인과 관련된 논란의 여지가 있는 사건들에 대한 주석이 담긴 YouTube 댓글들을 포함합니다. 주석 분석을 위해 취소 태도와 관련한 주석자들의 도덕적 프로필을 파악하는 데 Moral Foundations Theory (MFT)를 사용했습니다. 각 댓글에 대해 주석자들이 취소 태도를 어떻게 평가하는지를 분석하여, 그들의 도덕적 기준이 사건의 성격에 따라 다르게 나타나는 것을 발견했습니다.

- **Performance Highlights**: 연구 결과, 6666개의 LLMs(large language models)가 사람들의 도덕적 평가와 비교되었으며, 다양한 LLM들이 각각 다른 도덕적 관점을 보인다는 사실이 밝혀졌습니다. 이는 콘텐츠 모더레이션(content moderation)에서 LLM의 도덕적 기준이 얼마나 중요한지를 드러냅니다. 연구를 통해 소셜 미디어에서의 공적 망신(public shaming) 문제를 해결하기 위한 방향성을 제시하고 있습니다.



### Beyond English: Unveiling Multilingual Bias in LLM Copyright Complianc (https://arxiv.org/abs/2503.05713)
Comments:
          Work in progress

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)의 저작권 보호에서 다국적 편향을 분석한 최초의 연구입니다. 저작권이 보호된 콘텐츠를 다루는 데 있어 LLMs가 언어 방향에 따라 차별적인 태도를 보이는 것을 발견했습니다. 연구팀은 영어, 프랑스어, 중국어 및 한국어로 된 인기 노래 가사를 포함한 데이터셋을 구축하여 이 모델들을 평가하였습니다.

- **Technical Details**: 연구진은 다양한 모델에 대해 직접적인 프로빙 형식의 프롬프트를 사용하여 저작권 콘텐츠의 생성을 조사했습니다. 평가에서 '최장 공통 부분 문자열(Longest Common Substring, LCS)'과 'ROUGE-L' 점수를 사용하여 실질적인 저작권 복제량을 측정했습니다. 또한, 생성된 콘텐츠에서 허위 생성 비율(Hallucination Rate)도 평가하여 저작권 준수 여부를 보다 철저히 분석했습니다.

- **Performance Highlights**: 연구 결과, 여러 LLMs가 언어에 따라 저작권 보호를 불균형적으로 시행하고 있다는 사실이 드러났습니다. 예를 들어, OpenAI의 GPT-3.5-Turbo 모델은 영어 저작권 가사에 대한 거부율이 가장 높았지만, 한국어와 중국어 가사에 대해서는 상대적으로 낮은 보호를 보였습니다. 이 모델들은 다국적 저작권 보호 메커니즘에서의 한계를 나타내며, 저작권 정책 적용의 일관성을 보장하기 위한 추가적인 연구 개발이 필요하다는 점을 강조합니다.



### Russo-Ukrainian war disinformation detection in suspicious Telegram channels (https://arxiv.org/abs/2503.05707)
Comments:
          CEUR-WS, Vol-3777 ProfIT AI 2024 4th International Workshop of IT-professionals on Artificial Intelligence 2024

- **What's New**: 이 논문은 러시아-우크라이나 갈등과 관련된 Telegram 채널의 허위 정보(disinformation) 식별을 위한 고급 접근법을 제안합니다. 최신 심층 학습(deep learning) 기술 및 전이 학습(transfer learning)을 활용하여, 기존의 수작업 검증(manual verification) 및 규칙 기반 시스템(rule-based systems)의 한계를 극복하고자 합니다. 이를 통해 허위 정보 탐지 분야에서의 효율성을 높이고자 합니다.

- **Technical Details**: 제안된 시스템은 LLM 모델을 포함한 심층 학습 알고리즘을 사용하며, 검증된 허위 정보와 합법적(content)인 자료로 구성된 맞춤형 데이터셋에 대해 미세 조정(fine-tuning)됩니다. 기존의 전통적인 기계 학습(machine learning) 기법보다 더 나은 맥락 이해(contextual understanding) 및 신흥 허위 정보 전략(emerging disinformation strategies)에 대한 적응성을 제공합니다.

- **Performance Highlights**: 논문 결과에 따르면, 제안된 접근법은 전통적인 기계 학습 기법을 크게 능가하며, 빠르게 변화하는 여론 공작(tactics)에 맞서 효과적인 대응이 가능합니다. 이러한 성과는 허위 정보 식별 작업의 품질을 높여주며, 대량의 데이터가 생성되는 환경에서도 높은 정확도를 유지합니다.



### OPTIC: Optimizing Patient-Provider Triaging & Improving Communications in Clinical Operations using GPT-4 Data Labeling and Model Distillation (https://arxiv.org/abs/2503.05701)
Comments:
          15 pages, 8 figures. submitted to Journal of the American Medical Informatics Association

- **What's New**: COVID-19 팬데믹은 전자 의료 포털을 통한 원격 의료(telemedicine)와 환자 메시징의 채택을 가속화했습니다. 이러한 플랫폼은 환자의 의료 접근성을 향상시키지만, PMARs(Patient Medical Advice Requests)의 급증으로 의료 제공자에게 부담을 가중시켰습니다. 이에 대한 해결책으로 본 연구에서는 의사 업무 부담을 줄이고 환자와 제공자 간의 커뮤니케이션을 개선하기 위해 메시지 분류 도구인 OPTIC을 개발했습니다.

- **Technical Details**: OPTIC은 GPT-4를 활용한 데이터 라벨링과 BERT 모델의 증류(distillation)를 통해 메시지를 효과적으로 분류하는 강력한 도구입니다. 본 연구는 2020년 1월부터 6월까지 Johns Hopkins Medicine의 405,487건의 환자 메시징 데이터를 사용하여 진행되었습니다. GPT-4 기반 프롬프트 엔지니어링을 통해 생성된 고품질 라벨 데이터를 BERT 모델의 학습에 사용하였으며, 이를 통해 메시지를 "Admin"과 "Clinical"로 분류합니다.

- **Performance Highlights**: BERT 모델은 GPT-4 라벨링으로 검증된 테스트 세트에서 88.85% 정확도로 메시지를 분류하였으며, 민감도(sensitivity) 88.29%, 특이도(specificity) 89.38% 및 F1 점수(F1 score) 0.8842를 기록했습니다. 또한, BERTopic 분석을 통해 81개의 별개의 주제를 식별하였고, 58개 주제의 80% 이상을 정확하게 분류했습니다. 이 시스템은 Epic의 Nebula Cloud Platform을 통해 성공적으로 배포되어 의료 환경에서의 실질적인 효과를 입증했습니다.



New uploads on arXiv(cs.IR)

### Talking to GDELT Through Knowledge Graphs (https://arxiv.org/abs/2503.07584)
- **What's New**: 이 연구에서는 Retrieval Augmented Generation (RAG) 접근 방식을 통해 질문-응답 분석에서 각 접근 방식의 장단점을 파악합니다. 이를 위해 우리는 Global Database of Events, Language, and Tone (GDELT) 데이터셋의 사례 연구와 온라인 뉴스 기사에서 스크래핑한 원시 텍스트 코퍼스를 사용합니다. RAG 시스템에서 전통적인 벡터 저장소와 최신 대형 언어 모델(LLM) 기반 접근 방식을 구현하여 지식 그래프(KG)를 자동으로 구축하고 관련 서브 그래프를 검색합니다.

- **Technical Details**: 본 연구에서는 GDELT 데이터셋을 활용하여 KG 구축을 위한 새로운 온톨로지를 개발하고, LLM을 사용하여 이 KG와의 상호작용을 통해 유용성을 검토합니다. GDELT는 전 세계 사건에 대한 실시간 컴퓨터 기록을 제공하며, 다양한 뉴스 소스와 블로그, 소셜 미디어 플랫폼에서 정보를 수집합니다. RAG 접근 방식은 비구조적 텍스트 기사를 활용하는 전통적인 방법에서 넘어, 그래프 구조를 가진 데이터를 다루는 기술로 발전하고 있습니다.

- **Performance Highlights**: 연구 결과, 온톨로지 기반 KG는 질문-응답 작업에서 유용성이 있지만 자동 서브 그래프 추출에는 어려움이 있는 반면, LLM이 생성한 KG는 사건 요약을 포착하지만 일관성과 해석 가능성이 부족하다는 것을 발견했습니다. 이러한 발견은 온톨로지와 LLM 기반 KG 구성이 상호 보완적으로 접근할 때 이점이 있음을 시사하며, 향후 연구 방향에 대한 제안을 포함합니다.



### GRITHopper: Decomposition-Free Multi-Hop Dense Retrieva (https://arxiv.org/abs/2503.07519)
Comments:
          Under Review at ACL Rolling Review (ARR)

- **What's New**: GRITHopper-7B는 혼합적 언어 모델링과 밀집 검색 훈련을 통합한 새로운 다단계 밀집 검색 모델이다. 이는 분해 기반 접근방식의 한계를 극복하고, 분해 없는 방법이 긴 다단계 문제에서 어려움을 겪는 현상을 해결하고자 한다. 이 모델은 다양한 질문-답변 및 사실 확인 작업에 대한 다단계 데이터셋에서 훈련되었다.

- **Technical Details**: GRITHopper는 GRITLM을 기반으로 하여 인과(language modeling) 언어 모델링과 밀집 검색 훈련을 결합하였으며, 밀접하게 관련된 정보 습득을 위해 최종 답변 등의 요소를 훈련 과정에 포함시킨다. 이 접근법은 '포스트-검색 언어 모델링'(post-retrieval language modeling)으로 정의되며, 검색 체인 이후의 추가 정보를 통해 밀집 검색 성능을 향상시킨다. 이를 통해 참가자들에게 더 효과적으로 관련 정보를 검색하는 법을 학습하게 된다.

- **Performance Highlights**: GRITHopper-7B는 최신 멀티홉 밀집 검색에서 최상위 성능을 달성하며, 기존 모델들과 비교했을 때 다양한 일반화 성능을 보여준다. 이 모델은 특히 분해 없는 방식으로 다단계 문제를 잘 처리하며, 이전 접근 방식에서 발생한 계산 비용 문제를 해결하는 데 도움을 준다. 궁극적으로 GRITHopper-7B는 향후 다단계 추론 및 검색 능력이 필요한 연구와 응용에 적합한 견고한 솔루션을 제공한다.



### Advancing Vietnamese Information Retrieval with Learning Objective and Benchmark (https://arxiv.org/abs/2503.07470)
- **What's New**: 이 연구에서는 베트남어 정보 검색 (Information Retrieval, IR) 작업을 위한 새로운 벤치마크인 베트남어 컨텍스트 검색(VCS)을 도입하고 있습니다. 이 벤치마크는 기존 베트남어 데이터셋을 수정하여 구성되었으며, 검색(retrieval) 및 리랭킹(reranking) 작업에 대한 평가를 중점적으로 다룹니다. 또한, 기존의 InfoNCE 손실 함수에서 개선된 새로운 학습 목표 함수를 제시하여 베트남어 임베딩 모델의 성능을 향상시키고자 합니다.

- **Technical Details**: 본 연구는 새로운 벤치마크를 구축하기 위해 기존 베트남어 데이터셋을 활용하며, 이는 텍스트 임베딩 모델이 관련 문서를 정확히 검색할 수 있도록 돕습니다. 본 벤치마크는 검색 및 리랭킹 작업에서 베트남어 언어 모델의 성능을 종합적으로 평가하기 위해 고안되었습니다. 또한, 정보 검색 작업에서 성능을 향상시키기 위해 하이퍼파라미터인 온도(temperature)의 영향을 분석합니다.

- **Performance Highlights**: 베트남어 컨텍스트 검색(VCS) 벤치마크를 통해 여러 베트남어 임베딩 모델의 성능을 비교 평가하며, 각각의 모델이 실제 문서 검색에 얼마나 효과적인지를 점검합니다. 연구 결과, 새로운 학습 목표 함수가 기존 모델보다 뛰어난 성능을 보이며, 임베딩 모델 성능의 다양한 변화를 직접 확인할 수 있습니다. 이러한 결과는 베트남어 자연어 처리(NLP) 연구의 발전을 촉진할 것으로 기대됩니다.



### Process-Supervised LLM Recommenders via Flow-guided Tuning (https://arxiv.org/abs/2503.07377)
- **What's New**: 최근 대규모 언어 모델(LLM)을 활용한 추천 시스템이 급속히 발전하고 있으며, 이는 추천 정확도를 향상시킬 수 있는 잠재력을 보여주고 있습니다. 그러나 기존의 지도 학습 미세 조정(Supervised Fine-Tuning, SFT) 방법은 추천의 다양성과 공정성을 저해하는 경향이 있습니다. 이러한 상황을 해결하기 위해, 논문은 GFlowNet 기반의 플로우 유도 미세 조정 추천 시스템인 Flower를 제안합니다.

- **Technical Details**: Flower의 주요 혁신은 항목의 보상을 개별 토큰 보상으로 분해하여, 각 토큰의 생성 확률과 그 보상 신호 간의 직접적인 정렬을 가능하게 합니다. GFlowNet은 보상 함수에 비례하여 항목을 샘플링하는 강화 학습 알고리즘으로, 주어진 보상을 극대화하는 것이 아닌 확률을 샘플링합니다. 이 프레임워크는 각 분기 지점에서 흐름 값을 계산하여 다음 토큰 예측 작업을 위한 토큰 수준 보상을 제공합니다.

- **Performance Highlights**: 실험 결과, Flower는 전통적인 SFT에 비해 공정성, 다양성 및 정확성을 향상시키는 데 뛰어난 성능을 보이는 것으로 나타났습니다. 또한 맞춤형 선호를 포함하는 유연한 통합이 가능하여, 추천의 개인화 및 정확성을 증대시키는 데 기여합니다. 이러한 장점들은 PPO 및 DPO와 같은 추가적인 정렬 방법이 적용되는 경우에도 유지됩니다.



### Weak Supervision for Improved Precision in Search Systems (https://arxiv.org/abs/2503.07025)
Comments:
          Accepted to the AAAI 2025 Workshop on Computational Jobs Marketplace

- **What's New**: 이 논문에서는 대규모 검색 시스템의 정밀도를 향상시키기 위해 Learning to Rank 프레임워크 내에서 쿼리-문서 쌍의 질을 추론하는 약한 지도 학습(weak supervision) 접근법을 제시합니다. 기존의 데이터 라벨링 방식이 시간과 비용이 많이 드는 반면, 사용자의 클릭 및 활동 로그를 대안으로 활용하여 효율적으로 학습 라벨을 생성할 수 있는 방법을 제안합니다. 이를 통해 '골드' 데이터셋 작성의 비용을 줄이고, 보다 신뢰성 있는 훈련 데이터를 생성합니다.

- **Technical Details**: 제안된 시스템 아키텍처는 Apache Spark를 통해 구현되며, 라벨링 함수(Label Functions, LFs)를 각 기록에 대해 실행합니다. 다양한 외부 데이터베이스, 모델 및 분류 체계를 활용하여 결정들을 내립니다. 제안된 방식은 기존의 Snorkel 프레임워크와는 달리 이진 라벨에 중점을 둡니다. LFs의 출력은 True(긍정), False(부정), 또는 null(기권)로 표현되며, 이 결과들을 결합하여 단일 확률적 라벨로 집계합니다.

- **Performance Highlights**: 훈련된 모델은 수억 개의 데이터 포인트에 대해 최적화된 DNN으로, TensorFlow와 Horovod를 사용하여 분산 환경에서 학습됩니다. 모델 성능은 NDCG@k를 통해 원래 레이블, 업데이트 된 레이블, 그리고 약한 라벨러의 예측 세트에 대해 평가됩니다. 실험 결과, 약한 지도 학습을 통해 정밀도 및 재현율이 개선되었음을 발견하였습니다.



### Multi-Behavior Recommender Systems: A Survey (https://arxiv.org/abs/2503.06963)
Comments:
          Accepted in the PAKDD 2025 Survey Track

- **What's New**: 최근 다중 행동 추천 시스템(multi-behavior recommender systems)에 대한 연구가 급격히 증가하고 있으며, 이러한 시스템은 사용자 선호도를 예측하기 위해 다양한 사용자-아이템 상호작용을 활용합니다. 특히 사용자의 클릭, 장바구니에 담기, 위시리스트 저장과 같은 행동은 기존의 구매와 평점 한정 방식에 비해 더욱 풍부한 통찰을 제공합니다. 이 문서에서는 데이터 모델링, 인코딩, 훈련의 세 가지 주요 단계에 초점을 맞춰 다양한 다중 행동 추천 기술을 종합적으로 검토합니다.

- **Technical Details**: 다중 행동 추천 시스템은 데이터 모델링(data modeling), 인코딩(encoding), 훈련(training) 세 가지 주요 단계를 통해 구성됩니다. 데이터 모델링 단계에서는 다양한 사용자 행동을 그래프(graph) 또는 시퀀스(sequence)와 같은 특정 데이터 구조로 표현합니다. 인코딩 단계에서는 모델링된 데이터를 벡터 표현(vector representations)으로 변환하고, 훈련 단계에서는 이러한 다중 행동 정보를 효과적으로 학습하고 활용하기 위해 파라미터를 최적화합니다.

- **Performance Highlights**: 최신의 다중 행동 추천 시스템(MULE)은 목표 행동(target behavior)인 구매의 예측에서 기존 방법에 비해 최대 463%의 성능 향상을 보입니다. 이러한 성능 향상은 사용자의 다양한 행동 데이터를 효과적으로 통합함으로써 달성되었습니다. 다중 행동 추천 시스템의 성능 평가는 수많은 벤치마크 데이터셋을 통해 이루어지며, 인기 있는 데이터셋 중 일부는 Tmall, Taobao, Yelp 등 다양한 도메인에서 수집된 것입니다.



### AlignPxtr: Aligning Predicted Behavior Distributions for Bias-Free Video Recommendations (https://arxiv.org/abs/2503.06920)
Comments:
          video recommendation. 7 page, 1 figure

- **What's New**: 이번 논문에서는 비디오 추천 시스템에서 사용자 행동(예: 시청 시간, 좋아요, 팔로우)이 사용자 선호를 추론하는 데 흔히 사용되지만, 지속 시간 편향, 인구 통계학적 편향 등 여러 가지 편향의 영향을 받는다는 점을 강조합니다. 저자들은 사용자 관심과 편향이 독립적이라는 가설을 바탕으로 새로운 방법론을 제안하며, 이는 예측된 행동 분포를 정량적 매핑(quantile mapping)을 통해 교정하여 편향 변수와 진정한 사용자 관심 간의 상관관계를 제거합니다. 이러한 접근법은 여러 편향 차원을 동시에 처리하면서도 실시간 대규모 시스템에서 효과적으로 동작할 수 있는 계산 효율성도 확보합니다.

- **Technical Details**: 논문에서는 사용자 행동을 관찰했을 때(예: 시청 시간, 좋아요) 이 행동이 사용자 관심(Z)과 여러 가지 편향(Y) 모두의 영향을 받는 구조를 제시합니다. 이 관계는 인과 구조(causal structure)로 설명되며, Z와 Y는 각각 독립적으로 존재하고, 이 둘은 잠재 행동 분포(X)에 영향을 미칩니다. 저자들은 편향 요인과 사용자 관심 간의 상호 정보를 최소화하기 위해 Z의 조건부 분포가 Y의 모든 값에 대해 동일해야 한다고 주장하여, 이를 통해 편향이 없는 사용자 관심을 추출할 수 있음을 이론적으로 입증합니다.

- **Performance Highlights**: 제안된 방법은 Kuaishou Lite 및 Kuaishou의 온라인 A/B 테스트를 통해 검증되었습니다. 결과적으로 사용자 활동 일수에서 각각 0.267% 및 0.115%의 누적 증가, 평균 앱 사용 시간에서 각각 1.102% 및 0.131%의 개선을 보였습니다. 이러한 결과는 제안한 방법이 장기적으로 사용자 참여 및 유지율을 향상시키는 데 효과적임을 나타내며, 다양한 플랫폼에서 긍정적인 결과를 확인할 수 있었습니다.



### Improving Access to Trade and Investment Information in Thailand through Intelligent Document Retrieva (https://arxiv.org/abs/2503.06489)
- **What's New**: 이 논문은 초보자들이 해외 투자와 무역을 보다 쉽게 이해할 수 있도록 돕는 챗봇 시스템을 소개합니다. 이 시스템은 자연어 처리(Natural Language Processing)와 정보 검색(Information Retrieval) 기술을 통합하여 복잡한 문서 검색 과정을 간소화합니다. 사용자는 제안된 시스템을 통해 해외 무역 및 투자 환경을 더 효율적으로 탐색할 수 있습니다.

- **Technical Details**: 제안된 시스템은 BM25 모델과 딥 러닝(Deep Learning) 모델을 결합하여 문서를 순위화하고 검색합니다. 이 방법론은 문서 내용에서 노이즈(noise)를 줄이고 결과의 정확성을 향상시키는 것을 목표로 합니다. 태국어 자연어 쿼리를 사용한 실험은 시스템의 문서 검색 능력을 입증하였습니다.

- **Performance Highlights**: 사용자 만족도 조사에서는 응답자 대부분이 시스템이 유용하다고 느꼈으며, 제안된 문서에 대해 동의하였습니다. 이러한 결과는 시스템이 태국 기업가들이 해외 무역과 투자에서 길잡이 역할을 할 수 있는 강력한 도구가 될 수 있음을 나타냅니다.



### HuixiangDou2: A Robustly Optimized GraphRAG Approach (https://arxiv.org/abs/2503.06474)
Comments:
          11 pages

- **What's New**: 이 논문에서는 HuixiangDou2라는 새로운 GraphRAG 프레임워크를 소개합니다. 이 프레임워크는 지식 집약적인 도메인에서 LLM(대규모 언어 모델)의 성능을 극대화하기 위해 설계되었습니다. 기존 GraphRAG 방법론을 통합하여 효율성과 정확성을 높이는 데 초점을 맞추고 있습니다. 또한, 새로운 다단계 검증 메커니즘을 도입하여 계산 비용을 증가시키지 않고도 검색의 강인성을 향상시킵니다.

- **Technical Details**: HuixiangDou2는 쿼리 분해 및 두 수준 검색과 같은 다양한 검색 메커니즘을 평가하고 최적화하는 데 중점을 둡니다. 이 프레임워크는 구조화된 지식 표현을 이용하여 동적 검색을 가능하게 하며, 알고리즘은 데이터 세트에서의 성능 향상을 입증하기 위한 체계적인 실험을 기반으로 합니다. 알고리즘 1에 설명된 인덱싱, 검색 및 생성 프로세스는 성능을 결정짓는 주요 요소입니다.

- **Performance Highlights**: Qwen2.5-7B-Instruct 모델의 경우 초기 성능이 60점에서 74.5점으로 개선되는 등 유의미한 성능 향상을 달성하였습니다. 이는 기존 LLM이 어려움을 겪던 복잡한 쿼리 구조에서도 정확한 검색을 가능하게 함을 보여줍니다. 후보 데이터셋에서 실험이 이루어졌으며, 두 수준 검색이 불확실성 일치(fuzzy matching)을 향상시키고 논리 기반 검색이 구조적 추론을 개선하는 데 기여했습니다.



### Image is All You Need: Towards Efficient and Effective Large Language Model-Based Recommender Systems (https://arxiv.org/abs/2503.06238)
- **What's New**: 이 논문에서는 LLM 기반 추천 시스템에서 항목 표현방식의 비효율성과 효율성 간의 균형을 다루기 위해 이미지 기반 접근 방식인 I-LLMRec을 제안합니다. 이 방법은 긴 텍스트 설명 대신 이미지를 활용하여 아이템을 표현함으로써 토큰 사용량을 줄이고 아이템 설명의 풍부한 의미 정보를 보존하는 것을 목표로 합니다. 여러 실제 데이터셋을 통해 I-LLMRec이 기존 방식에 비해 효율성과 효과성 모두에서 향상된 성과를 보임을 입증합니다.

- **Technical Details**: I-LLMRec은 이미지와 텍스트 간의 정보 겹침을 활용하여, 아이템 표현 시 이미지와 LLM의 언어 공간 간의 불일치를 극복하기 위해 학습 가능한 어댑터(Adaptor)와 이미지를 LLM에 맞추기 위한 기법을 도입합니다. 이를 통해 LLM이 적은 토큰 수로도 이미지에 대한 풍부한 의미 정보를 포착할 수 있도록 지원합니다. 주요 기술 도전과제는 아이템 이미지 공간과 언어 공간 사이의 불일치입니다.

- **Performance Highlights**: I-LLMRec은 설명 기반 표현 방식을 2.93배 향상시키고, 속도 및 효율성에 있어 Attribute 기반 표현 방식보다 22%의 성능 향상을 달성했습니다. 이 방법은 또한 아이템 설명의 노이즈에 대한 민감도를 줄여 보다 강력한 추천을 제공합니다. 실험 결과는 I-LLMRec이 다양한 자연어 기반 표현 방식보다 효과성과 효율성 모두에서 우수함을 입증합니다.



### Rank-R1: Enhancing Reasoning in LLM-based Document Rerankers via Reinforcement Learning (https://arxiv.org/abs/2503.06034)
- **What's New**: 논문에서는 Rank-R1이라는 새로운 LLM(large language model) 기반의 재정렬기가 사용자 쿼리와 후보 문서에 대한 추론을 수행한 후 문서를 정렬하는 접근 방식을 소개합니다. 기존의 문서 재정렬 방법은 LLM을 프롬프트(prompt)하거나 미세 조정(fine-tuning)하여 관련성에 따라 후보 문서를 배치하는 데 의존합니다. 반면, Rank-R1은 소규모의 관련성 라벨을 활용하여 강화 학습(reinforcement learning) 알고리즘을 사용하여 LLM 재정렬기의 추론 능력을 향상시킵니다.

- **Technical Details**: Rank-R1은 DeepSeek의 강화 학습 프레임워크를 바탕으로 LLM 기반 문서 재정렬기의 추론 능력을 강화합니다. 이 방법론은 Setwise prompting 접근법을 채택하여 쿼리와 후보 문서 세트를 입력으로 받아 LLM이 가장 관련성이 높은 문서를 선택하도록 유도합니다. 원래 Setwise 접근법은 쿼리와 문서 간의 관련성에 대한 추론을 유도하지 않지만, Rank-R1은 추론 지침을 추가해 문서의 관련성을 이해하는 능력을 향상시킵니다.

- **Performance Highlights**: TREC DL 및 BRIGHT 데이터셋을 통해 Rank-R1의 실험 결과, 복잡한 쿼리에 대해 매우 효과적임을 확인했습니다. 특히 인도메인 데이터셋에서는 감독 미세 조정 방법과 동등한 성능을 보이며, 미세 조정 방법이 사용하는 데이터의 18%만으로도 충분한 성능을 발휘합니다. 또한, Rank-R1은 복잡한 쿼리를 포함하는 아웃오브도메인 데이터셋에서도 기존 방법보다 뛰어난 성능을 보였습니다.



### From Limited Labels to Open Domains: An Efficient Learning Paradigm for UAV-view Geo-Localization (https://arxiv.org/abs/2503.07520)
- **What's New**: 본 논문에서는 기존의 다양한 UAV-view Geo-Localization (UVGL) 방법의 한계를 극복하기 위해 새로운 방식인 cross-domain invariance knowledge transfer network (CDIKTNet)를 제안합니다. CDIKTNet는 cross-domain invariance sub-network (CDIS)와 cross-domain transfer sub-network (CDTS)로 구성되어, 특정 도메인에서의 새로운 매칭 관계를 필요로 하지 않고도 강력한 성능을 발휘합니다. 이 방법은 paired data에 크게 의존하지 않으면서도 뛰어난 feature representation 및 지식 전이를 가능하게 합니다.

- **Technical Details**: CDIKTNet의 CDIS는 다양한 관점에서의 구조적 및 공간적 불변성을 학습하여 공유된 피처 공간을 구축합니다. CDTS는 이러한 불변 피처를 앵커로 사용하고, dual-path contrastive memory learning mechanism을 통해 unpaired data에서 latent cross-domain correlation patterns를 탐색합니다. 논문은 두 가지 주요 부품의 상호 작용을 통해 불변성의 특징을 극대화하고, 학습에 필요한 paired data의 양을 최소화하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과에 따르면 CDIKTNet는 전통적인 방법들에 비해 두 가지 주요 벤치마크에서 뛰어난 성능을 보여주었습니다. 특히, 단 2%의 paired data로도 기존의 지도 학습 방법과 유사한 성능을 낼 수 있었고, unpaired data에서의 효과적인 feature learning을 통해 다양한 시나리오에 쉽게 전이할 수 있는 능력을 보였습니다. 따라서 CDIKTNet는 다양한 응용 프로그램에서 개방적인 환경을 지원할 수 있는 가능성을 지니고 있습니다.



### Zero-Shot Hashing Based on Reconstruction With Part Alignmen (https://arxiv.org/abs/2503.07037)
- **What's New**: 본 논문에서는 이미지 부분 정렬(Part Alignment)을 기반으로 한 새로운 제로샷 해싱(zero-shot hashing) 알고리즘인 RAZH를 제안합니다. 기존의 제로샷 해싱 알고리즘은 전반적인 이미지에서 추출된 특징을 활용하지만, RAZH는 이미지의 개별 부분에 대한 특성을 매칭하는 데 중점을 둡니다. 이를 통해 특성과 이미지 부분 간의 정밀한 정렬을 달성하여 정확도를 높이고, 특정 특성과 해당 이미지 부위 간의 관계를 포착하는 방법을 제시합니다.

- **Technical Details**: RAZH는 먼저 군집화(Clustering) 알고리즘을 사용하여 유사한 패치를 그룹화하여 각 부분에 맞는 특성을 정렬합니다. 그 후, 이미지 부분을 해당 특성 벡터로 대체하고, 각 부분과 가장 가까운 특성과 점진적으로 정렬하는 재구성 전략(Reconstruction Strategy)을 사용합니다. 이러한 접근 방식은 다양한 크기의 이미지 부분과 특성 간의 정렬 문제를 해결하기 위한 새로운 틀을 제공합니다.

- **Performance Highlights**: 광범위한 실험 결과에서 RAZH 방법은 여러 최신 기술(State-of-the-art) 방법들보다 우수한 성능을 보였습니다. 이 연구는 제로샷 해싱 문제를 해결하는 데 있어 이미지 부분과 속성 간의 정밀한 정렬이 핵심적이라는 점을 강조하며, 이를 통해 새로운 클래스에 대한 해시 코드를 효과적으로 생성할 수 있음을 보여줍니다.



### Graph Retrieval-Augmented LLM for Conversational Recommendation Systems (https://arxiv.org/abs/2503.06430)
Comments:
          Accepted by PAKDD 2025

- **What's New**: 이 논문에서는 G-CRS(Graph Retrieval-Augmented Large Language Model for Conversational Recommender Systems)라는 새로운 툴을 소개합니다. G-CRS는 훈련이 필요 없는 프레임워크로, 그래프 기반 정보 검색과 ICL(In-Context Learning)을 결합하여 LLM(대형 언어 모델)의 추천 능력을 향상시킵니다. 기존의 방법들이 가지는 도메인 지식 부족 문제를 해결하고, 의미론적 관계와 사용자 상호작용을 더 효과적으로 캡처합니다.

- **Technical Details**: G-CRS는 두 단계의 검색-추천 아키텍처를 사용하여, 첫 단계에서 GNN(그래프 신경망) 기반 추론기가 후보 아이템을 식별하고, 이후 Personalized PageRank(PPR) 알고리즘을 통해 사용자 관심에 맞는 추가 아이템을 탐색합니다. 이 과정에서 기존 대화의 이력을 활용하여 LLM이 현재 대화에서의 사용자 선호를 더 잘 이해할 수 있도록 돕습니다. 이 방법은 기존의 RAG(검색을 통한 생성) 접근법을 향상시키며, 특정 작업 훈련 없이도 효과적인 추천이 가능합니다.

- **Performance Highlights**: G-CRS는 두 개의 공개 데이터셋에서 실험을 통해 기존 방법들보다 우수한 추천 성능을 보였습니다. 추가적인 모델 훈련 없이도 G-CRS의 프레임워크는 추천 정확도를 향상시키는 데 성공하였으며, 이는 대화 기반 추천 시스템의 실용성을 크게 높이는 결과로 이어집니다. 이 연구는 대화형 추천 시스템의 발전에 기여할 뿐 아니라, 도메인 특정 지식의 필요성을 줄이는 효과를 가지고 있습니다.



New uploads on arXiv(cs.CV)

### AlphaDrive: Unleashing the Power of VLMs in Autonomous Driving via Reinforcement Learning and Reasoning (https://arxiv.org/abs/2503.07608)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문에서는 AlphaDrive라는 새로운 프레임워크를 제안하며, 이는 자율 주행에서 Vision-Language Models (VLMs)의 계획(Planning) 성능을 향상시키기 위한 Reinforcement Learning (RL)과 추론(Reasoning) 전략을 통합합니다. AlphaDrive는 GRPO 기반의 RL 보상을 도입하고, SFT와 RL을 결합한 두 단계의 훈련 전략을 채택하여 훈련 효율성과 성능을 크게 개선합니다. 또한, AlphaDrive는 RL 훈련 후에 나타나는 다중 모드 계획 능력을 통해 자율 주행의 안전성과 효율성을 향상시키는 잠재성을 보여줍니다.

- **Technical Details**: AlphaDrive는 네 가지 GRPO 기반의 RL 보상을 도입하여 계획 성능을 최적화합니다. 계획 정확도 보상, 행동 가중치 보상, 계획 다양성 보상, 계획 형식 보상으로 구성된 이 보상들은 자율 주행 모델의 적합성을 높입니다. 나아가, 지식 증류(Knowledge Distillation)를 기반으로 한 두 단계의 훈련 전략을 통해 초기 단계의 불안정성과 착시 현상을 완화하고, 계획 성능을 개선하는 데 기여합니다.

- **Performance Highlights**: 대규모 자율 주행 데이터셋에서 실험한 결과, AlphaDrive는 SFT 모델에 비해 계획 정확도를 25.52% 대폭 향상시켰으며, 훈련 데이터의 20%로도 SFT 모델보다 35.31% 더 뛰어난 성능을 보여주었습니다. 이 연구는 AlphaDrive가 자율 주행 분야에서 GRPO 기반의 RL과 계획 추론을 통합한 최초의 프레임워크임을 강조하고, 코드가 공개될 예정이므로 향후 연구에 기여할 수 있는 가능성을 탐색합니다.



### VoD: Learning Volume of Differences for Video-Based Deepfake Detection (https://arxiv.org/abs/2503.07607)
- **What's New**: 이번 논문은 Deepfake 감지를 위한 새로운 프레임워크인 Volume of Differences (VoD)를 제안합니다. VoD는 연속된 비디오 프레임 간의 시공간적 불일치를 활용하여 감지 정확성을 향상시킵니다. 특히, 이는 Consecutive Frame Differences (CFD)를 통해 시간 정보를 추출하고, 이를 여러 축(axes)에서 활용하여 미세한 차이를 학습하는 방식으로 작동합니다.

- **Technical Details**: VoD 프레임워크는 데이터의 특정 구성을 최적화하기 위해 여러 파라미터를 조정할 수 있는 유연성을 제공합니다. 이를 통해 Deepfake 탐지의 성능을 크게 향상시킬 수 있는 기반을 마련합니다. 이 연구는 FaceForensics++ (FF++) 데이터셋을 통해 검증되었으며, 성능이 기존의 최첨단 방법들(SOTA)보다 우수함을 보여줍니다.

- **Performance Highlights**: 연구 결과, VoD 방법은 훈련된 데이터에서 뛰어난 성능을 보였으며, 새로운 데이터에 대한 적응력도 강한 것으로 나타났습니다. 특히, 다양한 구성을 통한 ablation studies는 각 구성 요소가 Deepfake 감지 성능에 미치는 영향을 상세히 분석하였습니다. 이러한 통찰력은 더욱 정확하고 효율적인 Deepfake 감지 시스템 개발에 기여할 것으로 기대됩니다.



### Should VLMs be Pre-trained with Image Data? (https://arxiv.org/abs/2503.07603)
Comments:
          ICLR 2025

- **What's New**: 이 논문은 전처리된 LLM(대규모 언어 모델)이 이미지 데이터로 추가 훈련될 때의 비전-언어(Vision-Language) 작업 성능 개선에 대해 연구합니다. 특히, 비전 토큰을 모델에 도입하는 최적의 시점과 방법이 모델 성능에 어떤 영향을 미치는지 분석합니다. 300개의 다양한 모델을 훈련시켜 이미지 데이터의 중요성을 정량적으로 평가했습니다.

- **Technical Details**: 제안된 실험은 3단계 과정으로 구성되어 있습니다: 1) 부분 텍스트 전처리, 2) 이미지-텍스트 쌍을 혼합한 텍스트 연속 전처리, 3) 다중 작업 파인튜닝입니다. 이 연구에서는 기존의 VLM(비전-언어 모델)과 달리 모델이 완전히 전처리되기 전에 이미지 데이터를 도입하여 훈련하는 방식을 채택했습니다. 다양한 비율의 이미지와 텍스트 데이터 셋을 조합하여 실험을 진행하였습니다.

- **Performance Highlights**: 실험 결과, 비전-언어 작업에서의 평균 성능은 이미지 데이터를 전처리 단계에서 적절히 도입할 때 향상되었으며, 일반적으로 1B 파라미터 모델에서 이미지 데이터가 80%로 도입될 때 평균적으로 2% 향상된 성능을 나타냈습니다. 각 도메인에서 강력한 성능을 발휘하기 위해 적절한 비율의 시각적 데이터(10%~20%)와 파인튜닝 시점을 조정하는 것이 매우 중요합니다.



### DreamRelation: Relation-Centric Video Customization (https://arxiv.org/abs/2503.07602)
Comments:
          Project Page: this https URL

- **What's New**: 이번 논문에서는 DreamRelation이라는 혁신적인 방식이 소개되었다. 이는 사용자가 지정한 관계를 개인화한 비디오를 생성할 수 있도록 설계된 것으로, 기존의 방법들이 놓쳤던 복잡한 관계 모델링 문제를 해결하려고 한다. DreamRelation은 Relational Decoupling Learning과 Relational Dynamics Enhancement 두 가지 주요 요소를 활용하여 다양한 주제 간 관계를 강화하는 데 중점을 두고 있다.

- **Technical Details**: DreamRelation의 핵심 기술인 Relational Decoupling Learning에서는 관계와 주제의 외형을 분리하는 관계 LoRA triplet과 하이브리드 마스크 훈련 전략을 사용한다. 이를 통해 다양한 관계에 대한 일반성을 높이고, MM-DiT의 주의 메커니즘 내에서 쿼리, 키, 값 특징을 분석하여 최적의 관계 LoRA triplet 구조를 개발한다. 또한, Relational Dynamics Enhancement에서는 시공간 관계적 대조 손실(space-time relational contrastive loss)을 도입하여 관계 동적성을 강조한다.

- **Performance Highlights**: 광범위한 실험 결과, DreamRelation은 기존의 최신 기술들과 비교하여 관계 비디오 개인화 작업에서 우수한 성능을 보였다. 연구팀은 26개의 인체 상호작용을 기반으로 데이터셋을 구성하고, 다양한 텍스트 프롬프트를 사용하여 평가는 진행되었다. 연구 결과는 DreamRelation이 관계 비디오 개인화의 가장 강력한 방법임을 입증하고 있으며, 코드와 모델이 공개될 예정이다.



### Balanced Image Stylization with Style Matching Scor (https://arxiv.org/abs/2503.07601)
Comments:
          Project page: this https URL

- **What's New**: 이번 논문에서는 이미지 스타일화(Stylization)를 위한 새로운 최적화 방법, Style Matching Score (SMS)를 제안합니다. 기존 방법들과의 차별점은 이미지 스타일화를 스타일 분포 매칭 문제로 재구성했다는 점이며, 이를 통해 효과적인 스타일 전이와 콘텐츠 보존 간의 균형을 맞추고자 합니다. 이 방법은 주로 오프더셸프 스타일 종속 LoRA를 사용하여 스타일 분포를 추정하고, 점진적으로 주파수 도메인에서 콘텐츠 보존을 위해 작동하는 Progressive Spectrum Regularization 기법을 도입합니다.

- **Technical Details**: SMS에서 스타일 전이는 스타일 매칭 목표(Style Matching Objective)를 통해 이루어지며, 콘텐츠 보존은 Progressive Spectrum Regularization을 통해 달성됩니다. 이들은 Kullback–Leibler (KL) 발산을 최소화하는 과정에서 스타일 분포와 생성된 이미지의 분포를 일치시키는 데 주력합니다. 또한, Semantic-Aware Gradient Refinement 기술을 통해 의미론적으로 중요한 영역의 스타일링을 선택적으로 수행하여 스타일과 콘텐츠를 효과적으로 균형 있게 조화시킵니다.

- **Performance Highlights**: SMS는 스타트 벤치마크 방법들을 초월하는 성능을 보이며, 여러 실험을 통해 이러한 점이 검증되었습니다. 제안된 방법은 픽셀 공간에서 파라미터 공간으로 스타일링을 확장하여 경량 피드포워드 생성기에도 손쉽게 적용할 수 있어, 효율적인 스타일화 솔루션을 제공합니다. 이로 인해 SMS는 실시간 스타일 전이 및 보다 넓은 다양한 응용에서의 활용 가능성을 제공합니다.



### VACE: All-in-One Video Creation and Editing (https://arxiv.org/abs/2503.07598)
- **What's New**: VACE는 비디오 생성 및 편집을 위한 올인원 모델로, 참조-비디오 생성, 비디오-비디오 편집, 마스킹된 비디오-비디오 편집을 포함한 다양한 과제를 지원합니다. 이 모델은 다양한 비디오 작업 입력을 체계적으로 정리하여 Video Condition Unit (VCU)라는 통합 인터페이스를 제공합니다. VACE는 기존의 텍스트-비디오 생성 모델을 기반으로 하여 긴 비디오 시퀀스를 처리하기 위한 뛰어난 기본 기능과 확장성을 제공합니다.

- **Technical Details**: 비디오 작업에 대한 필요를 고려하여 VACE는 멀티모달 입력을 통합한 구조로 설계되었습니다. 영상과 비디오 편집, 참조, 마스킹 작업 등을 통합하여 VCU 하의 새로운 멀티모달 입력 형식을 설계하였습니다. 또한, Context Adapter 구조를 활용하여 임의 비디오 합성 작업을 유연하게 처리할 수 있도록 다양한 작업 개념을 모델에 주입합니다.

- **Performance Highlights**: VACE 모델은 기존의 특화된 모델과 비교하여 다양한 하위 작업에서 동등한 성능을 보입니다. 실험을 통해 정량적 및 정성적 분석 모두에서 경쟁력을 입증하였습니다. 이 모델은 사용자에게 프리미엄 비디오 콘텐츠 생성과 편집의 새로운 가능성을 열어줍니다.



### HumanMM: Global Human Motion Recovery from Multi-shot Videos (https://arxiv.org/abs/2503.07597)
Comments:
          CVPR 2025; Project page: this https URL

- **What's New**: 이번 논문은 샷 간 전환이 있는 야외 비디오로부터 3D 인간 모션을 월드 좌표에서 재구성하기 위한 새로운 프레임워크를 제안합니다. 기존 방법들은 일반적으로 단일 샷 비디오에 초점이 맞춰져 있으며, 여러 샷 간의 연속성을 유지하기 어려운 문제를 가지고 있습니다. 우리는 샷 전환 감지기와 강력한 정렬 모듈을 포함한 카메라 포즈 추정 방법을 통합하여 정확한 포즈 및 방향 연속성을 보장합니다.

- **Technical Details**: 이 프레임워크는 Human Motion Recovery (HMR) 기술을 사용하여 카메라 간의 포즈를 정대하고, 각 샷이 별도로 포즈를 계산하는 과정에서의 불일치를 해소합니다. 우리는 샷 전환 감지기를 개발하고, 장기 특징 점 추적을 포함한 SLM 방법을 통해 카메라 포즈를 보다 견고하게 추정합니다. 또한, 인접 샷 간에도 인간의 방향을 align하도록 다중 샷 HMR 인코더를 사용하는 정렬 모듈을 구현합니다.

- **Performance Highlights**: 우리는 대규모 멀티 샷 데이터 세트를 구축하였으며, 이는 공개 3D 인간 데이터 세트를 기반으로 합니다. Extensive 평가를 통해 우리의 방법이 월드 좌표에서 현실적인 인간 모션을 효과적으로 재구성하는 데 강력함을 증명하였습니다. 특히, foot sliding 문제를 완화하고 시간적 일관성을 보장하여 재구성된 3D 인간 모션의 전체 모션 일관성을 향상시켰습니다.



### Hierarchical Cross-Modal Alignment for Open-Vocabulary 3D Object Detection (https://arxiv.org/abs/2503.07593)
Comments:
          AAAI 2025 (Extented Version). Project Page: this https URL

- **What's New**: 이 논문에서는 Open-vocabulary 3D object detection (OV-3DOD)을 위한 새로운 계층적 프레임워크인 HCMA를 제안합니다. 기존의 연구들이 놓치고 있는 3D 인식에 필요한 풍부한 장면 맥락을 통합할 수 있도록 설계되었습니다. HCMA는 로컬 객체와 글로벌 장면 정보를 동시에 학습하여 이전의 솔루션들보다 더 효과적인 결과를 보여줍니다.

- **Technical Details**: HCMA는 Hierarchical Data Integration (HDI) 방법론을 도입하여 3D-이미지-텍스트 데이터를 조합하여 객체 중심의 지식을 추출합니다. Interactive Cross-Modal Alignment (ICMA) 전략을 통해 다중 계층 기능 연결을 강화하고, Object-Focusing Context Adjustment (OFCA) 모듈을 통해 다양한 수준에서 객체 관련 기능을 정교하게 조정합니다. 이러한 기술적 기여는 3D 장면에서의 객체 탐지를 보다 정교하게 만듭니다.

- **Performance Highlights**: 방대한 실험 결과, 제안된 HCMA 방법은 기존의 OV-3DOD 벤치마크에서 최신 기술(SOTA)보다 뛰어난 성능을 보였습니다. 특히, 3D 주석이 없는 상태에서도 우수한 OV-3DOD 결과를 달성하여, 이 프레임워크의 적용 가능성과 효과성을 잘 나타냈습니다. 이로 인해 HCMA는 3D 객체 탐지 연구 분야에서 중요한 이정표가 될 것으로 기대됩니다.



### Filter Images First, Generate Instructions Later: Pre-Instruction Data Selection for Visual Instruction Tuning (https://arxiv.org/abs/2503.07591)
Comments:
          Accepted at Computer Vision and Pattern Recognition Conference (CVPR) 2025

- **What's New**: 이번 논문에서는 Visual Instruction Tuning (VIT) 데이터 선택 방식인 PreSel을 소개합니다. PreSel은 유효한 레이블이 없는 이미지에서 가장 유용한 샘플을 직접 선택하고, 선택된 이미지에 대해서만 명령어를 생성함으로써 비용을 절감하는 접근 방식을 제안합니다. 또한, 기존의 VIT 방법과 달리, 고비용의 명령어 생성 단계 이전에 이미지 선택을 수행하여 효율성을 크게 증가시킵니다.

- **Technical Details**: PreSel 접근 방식은 두 단계로 구성됩니다. 첫 번째는 각 비전 작업의 중요성을 자동으로 추정하여 샘플링 예산을 결정하는 것입니다. 두 번째 단계에서는 경량 비전 인코더를 사용하여 unlabeled 이미지의 특징을 추출하고, 클러스터링 기법을 통해 각 클러스터 내의 대표 이미지를 선택합니다. 마지막으로, 선택된 이미지에 대해 명령어를 생성하여 LVLM (Large Vision-Language Models) 훈련을 준비합니다.

- **Performance Highlights**: 실험 결과, LLaVA-1.5와 Vision-Flan 데이터셋을 활용하여, PreSel은 전체 데이터의 15%만 이용하였음에도 불구하고 VIT에서 발생하는 비용을 15%로 줄이는 동시에 성능면에서도 유사한 결과를 달성했습니다. PreSel은 다양한 아키텍처와 크기의 LVLM에서 사용할 수 있는 전이 가능성을 입증하며, 효율적인 모델 개발을 위해 선택된 서브셋을 공개합니다.



### When Large Vision-Language Model Meets Large Remote Sensing Imagery: Coarse-to-Fine Text-Guided Token Pruning (https://arxiv.org/abs/2503.07588)
Comments:
          12 pages, 6 figures, 7 tables

- **What's New**: 이 논문에서는 대규모 원격 센싱 이미지(RSI)의 효율적인 비전-언어 이해를 위한 새로운 접근법을 제안합니다. 제안된 방법은 Text-Guided Token Pruning(텍스트 기반 토큰 프루닝)과 Dynamic Image Pyramid(DIP) 통합을 통해 이미지를 처리하며, 이를 통해 정보 손실을 최소화하면서 계산 복잡성을 줄이는 것을 목표로 합니다. 또한, 새로운 벤치마크인 LRS-VQA를 통해 7,333개의 질문-답변(QA) 쌍을 수집하여 LVLM의 평가 기준을 개선하였습니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 구성요소, 즉 Region Focus Module(RFM)과 Dynamic Image Pyramid(DIP)을 포함합니다. RFM은 텍스트 관련 비전 토큰을 식별하기 위해 텍스트 인식 지역 로컬라이징 능력을 이용합니다. 이어서 DIP는 중요 이미지 타일을 선택하고 비전 토큰을 프루닝하는 코스-투-파인 전략을 적용하여 효율적으로 전체 대규모 이미지를 처리합니다.

- **Performance Highlights**: 제안된 방법은 동일한 데이터를 사용하는 네 가지 데이터 세트에서 기존의 고해상도 전략을 초과하는 성능을 보였습니다. 또한 기존의 토큰 축소 방법들에 비해 고해상도 설정에서 더 높은 효율성을 나타내었습니다. 이 연구는 아키텍처에 독립적이며, 성능 개선과 효율성 상승을 모두 달성합니다.



### Robusto-1 Dataset: Comparing Humans and VLMs on real out-of-distribution Autonomous Driving VQA from Peru (https://arxiv.org/abs/2503.07587)
Comments:
          A pre-print. 26 pages. Link to Code + Data: this https URL

- **What's New**: 이 논문에서는 로부스토 데이터셋(Robusto-1 dataset)을 소개합니다. 이 데이터셋은 페루의 대시캠 비디오 데이터로 구성되어 있으며, 페루는 세계에서 운전이 가장 난폭한 나라 중 하나로 알려져 있습니다. 본 연구는 자율주행차가 어떻게 비표준 상황에서 사람과 유사하게 반응하는지를 연구하고자 합니다.

- **Technical Details**: 본 연구는 비주얼 질문 응답(Visual Question Answering, VQA) 방법을 사용하여 비즈니스의 인지적 수준에서 기초 비주얼 언어 모델(Foundational Visual Language Models, VLMs)과 인간을 비교합니다. 이에 대한 분석은 대표 유사도 분석(Representational Similarity Analysis, RSA)이라는 시스템 신경과학에서 유래한 방법을 통해 수행됩니다.

- **Performance Highlights**: 초기 분석 결과에 따르면, VLMs와 인간의 인지적 정렬은 질문의 종류에 따라 크게 달라집니다. 이는 자율주행차가 다양한 비표준 상황에서 얼마나 잘 일반화할 수 있는지를 평가하기 위한 중요한 지표로 작용할 수 있습니다. 이 데이터셋과 평가 프레임워크는 앞으로 자율주행차의 성능 평가에 유용할 것으로 기대됩니다.



### VisBias: Measuring Explicit and Implicit Social Biases in Vision Language Models (https://arxiv.org/abs/2503.07575)
Comments:
          9 pages

- **What's New**: 이 연구는 Vision-Language Models (VLMs)에서 나타나는 명시적 및 암묵적 사회 편향을 조사합니다. 연구진은 성별과 인종 차이에 관한 여러 질문을 통해 명시적 편향을 분석하고, 사용자 요청에 도움을 주면서도 편향을 드러내는 작업을 통해 암묵적 편향을 이해하고자 합니다. 이에 따라, 사용자의 요청에 도움이 되는 질문 및 작업을 설계해 편향 행동을 자세히 분석하려고 합니다.

- **Technical Details**: 편향 분석을 위해 연구진은 네 가지 평가 시나리오를 설계했습니다: (I) 다중 선택 질문, (II) 예-아니오 질문, (III) 이미지 설명, (IV) 양식 작성입니다. 이들 질문은 주로 개인의 외모를 기반으로 모델이 결정을 내리도록 설계되었으며, 인종, 성별 및 직업의 만남을 고려하여 이미지를 수집합니다. 또한, 편향 노출을 방지하기 위한 방법으로 Caesar cipher를 활용한 '탈옥'(jailbreak) 기법이 적용됩니다.

- **Performance Highlights**: 연구진은 GPT-4(V), GPT-4o, Gemini-1.5-Pro, LLaMA-3.2 (Vision), LLaVA-v1.6와 같은 VLM들을 평가했습니다. 실험 결과, 이러한 모델들이 명시적 질문의 경우 잘 수행하였으나, 암묵적 작업에서는 성별, 인종, 종교 등의 편향이 여전히 문제로 나타났습니다. 이러한 발견은 고급 VLM에서 사회 편향을 완화하려는 지속적인 노력이 필요함을 강조합니다.



### Alligat0R: Pre-Training Through Co-Visibility Segmentation for Relative Camera Pose Regression (https://arxiv.org/abs/2503.07561)
- **What's New**: 이번 논문에서는 Alligat0R라는 새로운 프리트레이닝(pre-training) 접근 방식을 소개합니다. 이 방법은 크로스 뷰(completion) 학습을 공동 가시성(segmentation) 작업으로 재구성하고, 이미지 쌍 간의 겹침(overlap)과 상관없이 픽셀의 위치를 예측합니다. 이는 이전 방식의 한계를 극복하고, 해석 가능한 예측을 제공합니다.

- **Technical Details**: Alligat0R는 델타-뷰(deep-view) 이미지 쌍 처리에 있어 균형 잡힌 아키텍처를 따릅니다. 이 접근법은 두 이미지 I1과 I2를 비겹치는 패치로 나누고, 각각의 패치를 독립적으로 처리하여 공동 가시성을 예측하는 구조로 되어 있습니다. 우리의 모델은 상대적인 자세 회귀(relative pose regression) 작업에 맞춰서 추가 조정될 수 있습니다.

- **Performance Highlights**: 실험 결과, Alligat0R는 크로코(CroCo) 대비 상대 자세 회귀에서 유의미한 성능 향상을 보였습니다. 특히, 겹침이 제한된 장면에서도 뛰어난 성능을 발휘했습니다. 또한, Alligat0R와 Cub3 데이터셋은 공개될 예정이며, 이는 비주얼 representations의 이해에 기여할 것입니다.



### LBM: Latent Bridge Matching for Fast Image-to-Image Translation (https://arxiv.org/abs/2503.07535)
- **What's New**: 이 논문에서는 Latent Bridge Matching (LBM)이라는 새로운 방법을 소개합니다. LBM은 잠재 공간(latent space)에서 Bridge Matching을 활용하여 빠른 이미지 간 변환을 구현하는 스케일러블(scalable) 방법입니다. 단일 추론 단계(inference step)만으로 다양한 이미지 변환 작업에서 최신 성능을 달성할 수 있음을 보여줍니다. 또한, 객체 제거, 깊이 추정(depth estimation), 객체 조명 변경(object relighting) 등의 다양한 이미지 변환 작업에 대한 효율성과 다재다능성을 입증합니다.

- **Technical Details**: LBM은 소스 도메인(source domain)에서 타겟 도메인(target domain)으로 이미지를 변환하는 작업을 위한 새로운 방법입니다. 이 방법은 두 분포 간의 변환 맵을 찾는 것입니다. LBM은 조건부 프레임워크를 도출하고 이를 통해 제어 가능한 이미지 조명 변경 및 그림자 생성 작업을 수행했습니다. 논문은 LBM의 다양한 구성 요소가 미치는 영향을 조사하기 위해 포괄적인 ablation study를 수행했습니다.

- **Performance Highlights**: LBM은 고해상도 이미지를 포함한 다양한 이미지 간 작업에서 매우 효과적입니다. 또한, LBM은 여러 샘플링 단계가 필요한 기존의 diffusion 기반 방법이나 flow matching 모델과 비교하여 우수한 성능을 자랑합니다. 이 방식은 단일 단계에서의 생성(single-step generation)을 가능하게 하여 실시간(real-time) 응용 프로그램에서의 활용성을 높입니다.



### VisRL: Intention-Driven Visual Perception via Reinforced Reasoning (https://arxiv.org/abs/2503.07523)
Comments:
          18pages,11 figures

- **What's New**: 최근 연구에서는 대규모 멀티모달 모델(large multimodal models, LMMs)을 통해 인간의 의도-driven 시각 이해를 최적화할 수 있는 새로운 접근법인 VisRL을 제안했습니다. 기존 방식들이 정밀한 주석 요구로 인해 확장성에 한계를 두고 있는 반면, VisRL은 보상 신호를 통한 Reinforcement Learning(강화 학습) 방식을 통해 이 문제를 극복합니다. 이 방법은 작업에 대한 성공이나 실패를 보상 신호로 사용하여 모델이 스스로 집중할 영역을 선택하도록 학습합니다.

- **Technical Details**: VisRL은 시각적 추론제를 완전한 보상 신호를 기반으로 최적화하는 데 필요한 새로운 학습 프레임워크를 제공합니다. 이 프레임워크는 데이터 생성 단계와 모델 최적화 단계를 거치며, 특정 질문에 대한 적절한 난이도를 고려하는 필터링 메커니즘을 도입합니다. DPO(Direct Preference Optimization) 알고리즘을 적용하여 모델이 시각적 추론 과정의 모든 단계를 최적화할 수 있도록 합니다.

- **Performance Highlights**: 다양한 벤치마크 실험 결과 VisRL은 기존의 강력한 기준(baselines)보다 일관되게 우수한 성능을 보여주었습니다. 또한, VisRL의 효과성이 다양한 멀티모달 모델에서 잘 일반화되는 것을 확인할 수 있었으며, 이는 이 접근법의 폭넓은 적용 가능성을 강조합니다. 이러한 결과는 사용자 의도에 기반한 시각적 인식의 혁신적 발전을 예고합니다.



### From Limited Labels to Open Domains: An Efficient Learning Paradigm for UAV-view Geo-Localization (https://arxiv.org/abs/2503.07520)
- **What's New**: 본 논문에서는 기존의 다양한 UAV-view Geo-Localization (UVGL) 방법의 한계를 극복하기 위해 새로운 방식인 cross-domain invariance knowledge transfer network (CDIKTNet)를 제안합니다. CDIKTNet는 cross-domain invariance sub-network (CDIS)와 cross-domain transfer sub-network (CDTS)로 구성되어, 특정 도메인에서의 새로운 매칭 관계를 필요로 하지 않고도 강력한 성능을 발휘합니다. 이 방법은 paired data에 크게 의존하지 않으면서도 뛰어난 feature representation 및 지식 전이를 가능하게 합니다.

- **Technical Details**: CDIKTNet의 CDIS는 다양한 관점에서의 구조적 및 공간적 불변성을 학습하여 공유된 피처 공간을 구축합니다. CDTS는 이러한 불변 피처를 앵커로 사용하고, dual-path contrastive memory learning mechanism을 통해 unpaired data에서 latent cross-domain correlation patterns를 탐색합니다. 논문은 두 가지 주요 부품의 상호 작용을 통해 불변성의 특징을 극대화하고, 학습에 필요한 paired data의 양을 최소화하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과에 따르면 CDIKTNet는 전통적인 방법들에 비해 두 가지 주요 벤치마크에서 뛰어난 성능을 보여주었습니다. 특히, 단 2%의 paired data로도 기존의 지도 학습 방법과 유사한 성능을 낼 수 있었고, unpaired data에서의 효과적인 feature learning을 통해 다양한 시나리오에 쉽게 전이할 수 있는 능력을 보였습니다. 따라서 CDIKTNet는 다양한 응용 프로그램에서 개방적인 환경을 지원할 수 있는 가능성을 지니고 있습니다.



### FastInstShadow: A Simple Query-Based Model for Instance Shadow Detection (https://arxiv.org/abs/2503.07517)
- **What's New**: 이 논문은 Instance shadow detection(인스턴스 섀도우 탐지) 방법 중 FastInstShadow라는 새로운 접근 방식을 제안합니다. 기존 방법들이 그림자와 객체를 독립적으로 감지한 후 연결하는 구조에서 벗어나, 새로운 쿼리 기반 아키텍처를 통해 더 높은 정확도를 달성합니다. FastInstShadow는 그림자와 객체 간의 관계를 파악하는 두 개의 이중 경로 transformer decoder로 구성되며, 검증된 SOBA dataset을 사용한 실험에서도 우수한 성능을 보였습니다.

- **Technical Details**: FastInstShadow(FIS)는 FastInst라는 쿼리 기반 인스턴스 세분화 방법을 기반으로 하며, 실시간 성능과 높은 정확도를 구현합니다. 이 방법은 그림자와 객체의 특성을 직접 집계할 수 있는 쿼리를 설계하여 두 가지 훈련 전략인 shadow direction learning과 box-aware mask loss를 포함합니다. 이러한 혁신은 기존의 쌍짓기 프로세스를 제거하고 서로의 관계를 학습하도록 모델을 돕습니다.

- **Performance Highlights**: FIS는 간단한 아키텍처에도 불구하고 SOTA(State-Of-The-Art) 방법보다 훨씬 더 높은 정확도를 자랑하며, 중간 해상도의 이미지에서 초당 30프레임 이상의 처리 속도를 달성합니다. 실험 결과, 기존 방법들보다 모든 기준에 대해 우수한 성능을 보여주는 것을 확인했습니다. 이러한 성과는 다양한 비전 작업과 이미지 생성 분야에서의 활용 가능성을 높이며, 현재 인스턴스 섀도우 탐지의 새로운 기준이 될 것으로 기대됩니다.



### CPAny: Couple With Any Encoder to Refer Multi-Object Tracking (https://arxiv.org/abs/2503.07516)
- **What's New**: 이번 연구에서는 Referring Multi-Object Tracking (RMOT) 분야에서의 한계를 극복하기 위해 CPAny라는 새로운 인코더-디코더 프레임워크를 제안합니다. CPAny는 Contextual Visual Semantic Abstractor (CVSA) 및 Parallel Semantic Summarizer (PSS)라는 두 가지 핵심 구성 요소를 도입하여 기존 모델보다 더 효율적이고 유연한 멀티 오브젝트 추적을 가능하게 합니다. 이 모델은 다양한 비주얼/텍스트 인코더와의 호환성을 보장하면서도 상대적으로 낮은 계산 비용으로 여러 표현을 동시에 처리할 수 있습니다.

- **Technical Details**: CPAny는 두 단계의 RMOT 프레임워크로, 인코더가 아니라 자체적으로 구축된 통합 의미 공간을 사용하는 점이 특징입니다. CVSA는 전역 시각적 표현에서 경로 인식 기능을 추출하여 통합된 의미 공간으로 투사합니다. PSS는 언어적 특징과 시각적 의미를 융합하여 참조 점수를 생성하는 멀티모달 디코더로, 두 단계 RMOT의 추론 속도를 크게 향상시킵니다. 이러한 방식으로 CPAny는 높은 계산 효율성과 확장성을 갖추게 됩니다.

- **Performance Highlights**: CPAny는 Refer-KITTI 및 Refer-KITTI-V2 데이터 세트에서 현재의 SOTA 방법들과 비교하여 성능과 효율성 모두에서 우수한 결과를 보여주었습니다. 특히, Refer-KITTI-V2에서 7.77%의 HOTA 개선을 이루어냈습니다. 다양한 비주얼/텍스트 인코더 조합에서도 뛰어난 성능을 나타내어 향후 연구에 대한 가능성을 열어두고 있습니다.



### PE3R: Perception-Efficient 3D Reconstruction (https://arxiv.org/abs/2503.07507)
- **What's New**: 최근 2D에서 3D로의 인식(2D-to-3D perception) 발전에 따라 2D 이미지에서 3D 장면을 이해하는 데 큰 개선이 있었습니다. 그러나 기존 방법들은 장면 전반에 걸친 일반화의 한계, 인식 정확도의 저하, 느린 재구성 속도 등의 문제에 직면해 있습니다. 이를 해결하기 위해 우리는 PE3R(Perception-Efficient 3D Reconstruction)이라는 새로운 프레임워크를 제안하여 정확성과 효율성을 향상시킵니다. PE3R는 빠른 3D 의미 필드 재구성을 가능하게 하는 피드포워드 아키텍처를 채택하여 다양한 장면과 객체에서 강력한 제로샷 일반화를 보여줍니다.

- **Technical Details**: PE3R는 픽셀 임베딩 난명 해소(pixel embedding disambiguation), 의미 필드 재구성(semantic field reconstruction), 글로벌 뷰 인식(global view perception) 등 3 가지 핵심 모듈을 통해 인식 및 재구성 기능을 강화합니다. 픽셀 임베딩 난명 해소는 크로스 뷰의 다중 레벨 의미 정보를 통합하여 계층 객체 간의 모호성을 해소하고 시점 일관성을 보장합니다. 의미 필드 재구성은 의미 정보를 재구성 과정에 직접 통합하여 정확도를 개선하며, 글로벌 뷰 인식은 단일 시점이 도입하는 노이즈를 줄여줍니다.

- **Performance Highlights**: PE3R는 2D에서 3D로의 개방적 어휘 세분화(2D-to-3D open-vocabulary segmentation) 및 3D 재구성과 같은 작업에 대해 평가되었으며, Mipnerf360, Replica, KITTI 등의 다양한 데이터셋을 사용하여 성능을 검증했습니다. 재구성 속도에서 최소 9배의 개선을 이루었으며, 세분화 정확도와 재구성 정확도에서도 상당한 증진을 확인했습니다. 이러한 결과는 PE3R이 업계에서 새로운 성능 기준을 설정했다고 자랑스럽게 말할 수 있게 합니다.



### Think Before You Segment: High-Quality Reasoning Segmentation with GPT Chain of Thoughts (https://arxiv.org/abs/2503.07503)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 복잡한 모든 형태의 질의에 대한 세분화(mask)를 위한 새로운 접근법인 ThinkFirst를 소개합니다. ThinkFirst는 훈련 없이 GPT의 사고 체계(chain-of-thought)를 활용하여 세분화 품질을 향상시키고, 특히 복잡한 구조의 물체에 대한 문제를 해결합니다. 이 프레임워크는 다양한 사용자 입력(텍스트 및 이미지 스크레치 등)을 통해 세분화 에이전트와의 상호작용을 쉽게 할 수 있게 만들어줍니다.

- **Technical Details**: Reasoning segmentation은 이미지에서 특정 객체를 세분화하고 시각적 데이터를 처리하기 위한 고급 인지 능력이 요구되는 작업입니다. ThinkFirst는 사용자 질의 및 이미지를 직접 모델에 전달하는 대신, 마치 사고 체계의 흐름을 따라가듯이 이미지 분석을 수행하고 그 결과를 바탕으로 세분화 모델을 유도합니다. 이 접근법은 기존의 reasoning segmentation 모델과 호환 가능하여, 효과적인 'zero-shot-CoT' 전략으로 작동합니다.

- **Performance Highlights**: ThinkFirst의 성능은 COD-10K와 Marine Video Kit와 같은 도전적인 데이터셋에서 정량적 및 정성적으로 평가되었습니다. 실험 결과, ThinkFirst는 세분화 품질을 현저히 향상시킴과 동시에 사용자가 제공한 복잡한 질의에 대해 덜 민감하게 반응합니다. 이러한 결과는 다양한 도메인 및 복잡한 이미지에서도 ThinkFirst의 강력한 일반화 능력을 보여줍니다.



### AthletePose3D: A Benchmark Dataset for 3D Human Pose Estimation and Kinematic Validation in Athletic Movements (https://arxiv.org/abs/2503.07499)
- **What's New**: 이 논문은 기존에 부족한 스포츠에 특화된 3D 포즈 추정 데이터셋을 해결하기 위해 AthletePose3D를 소개합니다. 이 데이터셋은 다양한 스포츠에서 이루어진 12가지 유형의 운동을 포함하며, 130만 개의 프레임과 16만 개의 개별 자세를 수집하여 고속 및 고가속의 운동을 안정적으로 캡처합니다. 또한, 기존의 데이터셋으로는 성능이 떨어지는 최첨단 단안 2D 및 3D 포즈 추정 모델의 성능을 AthletePose3D로 파인튜닝함으로써 오류를 69% 이상 감소시킬 수 있음을 보여줍니다.

- **Technical Details**: AthletePose3D는 달리기, 육상 및 피겨 스케이팅과 같이 다양한 종목에서 수행되는 인간의 3D 포즈를 포괄적으로 캡처합니다. 기존 데이터셋들, 특히 Human3.6M 및 MPI-INF-3DHP가 통제된 환경 내에서의 운동에 초점을 맞춘 반면, AthletePose3D는 고속의 복잡한 운동을 중시하여 이를 분석할 수 있는 독특한 특징을 제공합니다. 이 논문의 방법론 섹션에서는 이러한 고속 운동의 포즈를 정확하게 측정하기 위한 접근 방식을 상세히 설명합니다.

- **Performance Highlights**: 기존 데이터셋에 기반하여 학습된 모델은 AthletePose3D에 포함된 운동에 대해 상당히 낮은 성능을 보였습니다. 하지만 AthletePose3D에서 모델을 파인튜닝한 결과, 214mm에서 65mm로 평균 관절 위치 오류(MPJPE)를 감소시켜 69% 이상 개선된 성과를 기록하였습니다. 본 논문은 제안된 데이터셋을 통해 모니터링된 관절 각도 추정은 강한 상관관계를 보이나 속도 추정에는 제한이 있음을 강조하면서, 스포츠 환경 내에서의 단안 포즈 추정의 필요성과 가능성을 제시합니다.



### V2Flow: Unifying Visual Tokenization and Large Language Model Vocabularies for Autoregressive Image Generation (https://arxiv.org/abs/2503.07493)
Comments:
          11 pages, 6 figures

- **What's New**: V2Flow는 높은 충실도로 시각적 토큰을 생성하며, 대형 언어 모델(LLMs)의 어휘 공간과의 구조적 및 잠재적 분포 정렬을 보장하는 새로운 토크나이저입니다. 이 방법은 자연 언어 처리에서의 새로운 패러다임을 활용하여 비주얼 생성에 대한 자율 회귀 모델링을 가능하게 만들었습니다. V2Flow는 시각적 토큰화 과정을 흐름 일치 문제로 공식화하여, LLM 어휘 공간에 내장된 토큰 시퀀스에 따라 계속되는 이미지 분포를 학습합니다.

- **Technical Details**: V2Flow의 주요 설계는 두 가지로 나뉩니다: 첫째로, 비주얼 어휘 재샘플러를 제안합니다. 이는 시각 데이터를 압축하여 LLM의 어휘에 대한 부드러운 범주형 분포로 표현되는 간결한 토큰 시퀀스로 변환합니다. 둘째로, 마스크된 자율 회귀 정류 흐름(RECTIFIED FLOW) 디코더를 제시하며, 맥락적으로 풍부한 임베딩을 생성하기 위해 마스크된 변환기 인코더-디코더 구조를 채택합니다.

- **Performance Highlights**: 상 extensive 실험을 통해 V2Flow는 주류 VQ 기반 토크나이저보다 뛰어난 성능을 보여주며, 기존 LLM 위에서 자율 회귀 비주얼 생성을 용이하게 합니다. 전반적으로, V2Flow는 LLMs의 어휘에 원활하게 통합되며 고성능 이미지 재구성을 달성합니다. 이 토크나이저는 자율적인 시각 생성의 효율성과 효과성을 크게 향상시키는 것을 목표로 하고 있습니다.



### LLaVA-RadZ: Can Multimodal Large Language Models Effectively Tackle Zero-shot Radiology Recognition? (https://arxiv.org/abs/2503.07487)
- **What's New**: 최근 다중 모달 대형 모델 (MLLMs)은 시각적 이해 및 추론에서 뛰어난 성능을 보여주고 있습니다. 하지만 의료 질병 인식에 있어서는 제로샷(zero-shot) 방식으로 성능이 저조한데, 이는 포착된 특징과 의료 지식을 완전히 활용하지 못하기 때문입니다. 이를 해결하기 위해, 우리는 LLaVA-RadZ라는 간단하면서도 효과적인 프레임워크를 제안하며, 이는 제로샷 의료 질병 인식을 위한 것입니다.

- **Technical Details**: LLaVA-RadZ는 Decoding-Side Feature Alignment Training (DFAT)이라는 새로운 훈련 전략을 설계하였습니다. 이 전략은 MLLM 디코더 아키텍처의 특징을 활용하고, 다양한 모달리티에 맞춘 특수 토큰을 통합하여 이미지와 텍스트 표현을 더욱 효과적으로 이용할 수 있도록 합니다. 또한, Domain Knowledge Anchoring Module (DKAM)을 도입하여 이미지-텍스트 정렬 과정에서 발생하는 의미적 카테고리 간의 간극을 줄이며, 질병 인식의 정확도를 높입니다.

- **Performance Highlights**: LLaVA-RadZ는 여러 벤치마크에서 기존의 전통적인 MLLMs보다 제로샷 질병 인식에서 월등한 성능을 보이며, CLIP 기반 접근 방식과 비교했을 때 최신의 성능을 자랑합니다. 또한, 과학적 실험을 통해 이러한 성능 개선이 기존의 모델들과 비교했을 때 어떻게 증가하였는지를 명확하게 보여줍니다.



### Chameleon: Fast-slow Neuro-symbolic Lane Topology Extraction (https://arxiv.org/abs/2503.07485)
Comments:
          ICRA 2025, Project Page: this https URL

- **What's New**: 본 논문에서는 지도 없는 자율 주행을 위한 핵심 작업인 차선 구조 추출을 다룬다. 차선 및 교통 요소를 감지하고 그 관계를 파악하는 작업은 복잡한 추론을 필요로 한다. 이를 해결하기 위해, 우리는 비전-언어 기초 모델(VLMs)을 활용한 신경-기호 방식(neuro-symbolic methods)을 도입한다. 새롭게 제안된 Chameleon 알고리즘은 빠른 시스템과 느린 시스템을 번갈아 가며 사용하여 효율과 성능을 균형 있게 유지한다.

- **Technical Details**: Chameleon 알고리즘은 감지된 인스턴스에 대해 생성된 프로그램을 사용하여 직접적으로 추론하는 빠른 시스템과, 복잡한 상황을 처리하는 느린 시스템으로 구성된다. 이 알고리즘은 dense visual prompting과 neuro-symbolic reasoning의 장점을 결합하여, 각각의 시각 입력에 맞춰 프로그램을 최적화한다. 제안된 방법은 OpenLane-V2 데이터셋에서 평가되어 성능 향상을 보여주며, 다양한 기준 감지기(baseline detectors)와 일관된 개선을 기록하였다.

- **Performance Highlights**: Chameleon은 기존 신경-기호 방식의 한계를 뛰어넘어, 시각 정보(VLMs)를 통합하여 차선 구조 추출을 가능하게 한다. 이 시스템은 다양한 코너 케이스를 정확히 식별하고 처리할 수 있으며, 실시간 로봇 응용 분야에서도 활용 가능한 효율적인 솔루션을 제공한다. 또한, 논문에서는 성능 평가를 위한 포괄적인 데이터셋과 벤치마크를 제시하여, 향후 연구자들이 이 기술을 활용할 수 있는 기반을 마련하고 있다.



### VLRMBench: A Comprehensive and Challenging Benchmark for Vision-Language Reward Models (https://arxiv.org/abs/2503.07478)
Comments:
          12 pages, 4 figures. This work is in progress

- **What's New**: 이 논문에서는 VLRMBench라는 종합적인 비주얼-언어 모델(VL) 벤치마크를 소개합니다. 이는 12,634개의 질문으로 구성되어 있으며, 수학적 추론, 환각 이해, 멀티-이미지 이해에 대한 세 가지 구별된 데이터 범주를 포함합니다. 기존의 벤치마크들은 VL 모델의 특정 역량만 평가했으나, VLRMBench는 이러한 한계를 극복하고 여러 측면에서 성능을 평가할 수 있습니다.

- **Technical Details**: VLRMBench는 세 가지 주요 테마인 단계 기반, 결과 기반, 비판 기반의 12개 작업으로 구성되어 있으며, 각각의 작업은 VLRM의 프로세스 이해, 결과 판단, 비판 생성을 평가합니다. 또한, 연구진은 21개의 오픈소스 모델과 5개의 고급 클로즈드 소스 모델에서 광범위한 실험을 수행하였습니다. 이 과정에서 다단계 필터링과 자동 데이터 생성 파이프라인을 통해 고품질 샘플을 확보했습니다.

- **Performance Highlights**: VLRMBench에서의 실험 결과, 예를 들어 `Forecasting Future`라는 이진 분류 작업에서는 GPT-4o가 겨우 76.0%의 정확도만을 달성하는 등 여전히 큰 도전 과제가 있음을 강조합니다. 이 논문은 VLRMBench의 다양한 작업에서 오픈소스 모델과 클로즈드 소스 모델의 성능 격차를 좁히는 데 도움이 될 것으로 기대됩니다.



### SOGS: Second-Order Anchor for Advanced 3D Gaussian Splatting (https://arxiv.org/abs/2503.07476)
Comments:
          Accepted by CVPR 2025

- **What's New**: 이번 논문에서는 anchor-based 3D Gaussian splatting (3D-GS)의 새로운 기술인 SOGS를 제안합니다. SOGS는 second-order anchors를 도입하여 anchor feature의 차원을 줄이면서도 우수한 렌더링 품질을 달성하도록 설계되었습니다. 이는 Gaussian의 예측 속성을 향상시키며, 렌더링 품질과 모델 크기 간의 균형을 맞추는 데 기여합니다. 또한 선택적 그래디언트 손실을 통해 장면 텍스처와 기하학의 최적화를 향상시킵니다.

- **Technical Details**: SOGS는 covariance 기반의 second-order 통계와 anchor feature 차원 간의 상관관계를 통합하여 anchor 내의 feature를 보강합니다. 기술적으로, 각 Gaussian은 위치(μ), 투명도(α), 색상(coefficients)으로 정의되며, 그라디언트 맵을 쉽게 계산합니다. 이 맵은 렌더링된 이미지와 실제 이미지 간의 차이를 강조하여 어려운 텍스처와 구조가 렌더링되는 데 도움을 줍니다. 요약하자면, SOGS는 compact anchor size에서도 높을 품질의 이미지를 렌더링할 수 있습니다.

- **Performance Highlights**: 다양한 벤치마크에서의 실험 결과는 SOGS가 novel view synthesis에서 우수한 렌더링 품질을 달성함을 보여줍니다. 특히, anchor size와 모델 크기를 효과적으로 줄이는 동시에 뛰어난 성능을 보였습니다. SOGS는 기존의 최첨단 기술들을 능가하며, 향상된 렌더링 품질을 제공하는 것으로 확인되었습니다. 이를 통해 3D 컴퓨터 비전 분야에서의 적용 가능성이 한층 더 확장될 것으로 기대됩니다.



### A Review on Geometry and Surface Inspection in 3D Concrete Printing (https://arxiv.org/abs/2503.07472)
- **What's New**: 이 논문은 건설 분야에서의 적층 제조(Additive Manufacturing in Construction, AMC) 기술이 발전함에 따라 더욱 복잡한 Printed specimen의 품질 보증이 필요함을 강조하고 있습니다. 3D Concrete Printing (3DCP)을 위한 기하학 및 표면 품질 관리의 다양한 측면을 탐구하며, 주입 및 샷크리트 방법에 중점을 두고 있습니다. 기존의 품질 관리(QC) 방법에 대한 포괄적인 개요를 제공하고, 데이터 캡처 기술의 네 가지 범주와 그들 각각의 장단점을 상세히 논의합니다.

- **Technical Details**: 적층 제조 기술은 많은 제조 공정을 포함하고 있으며, 이 기술의 발전에 따라 스케일과 로봇 기술별로 다양한 프린팅 전략이 필수적입니다. 특정 프린팅 기술인 Extrusion과 Shotcrete 3D Printing(SC3DP)에 중점을 두며, SC3DP는 더 높은 기하학적 자유도와 인접한 필라멘트 간의 결합을 개선하여 콜드 조인트의 위험을 극복할 수 있는 잠재력을 가지고 있습니다. 이 논문은 또한 정확한 데이터 캡처를 위한 다양한 센서의 선택과 이들의 사용에 대한 방법론을 제시합니다.

- **Performance Highlights**: 이 연구는 3DCP의 QC 절차를 단계별로 분류하여 각 공정에서 품질 관리의 필요성을 강조하고 있습니다. 전통적인 QC 방법은 인쇄 마지막 단계에서만 이루어지므로 수정할 기회를 제한하지만, 자동화된 단계별 QC를 통해 지속적인 검사와 데이터 흐름의 쌍방향 통신이 가능합니다. 각 단계에서의 QC 기법과 센서 사용을 통해 인쇄 과정 전반에서의 일관성과 정밀도를 유지할 수 있는 방법을 제시하고 있습니다.



### YOLOE: Real-Time Seeing Anything (https://arxiv.org/abs/2503.07465)
Comments:
          15 pages, 9 figures;

- **What's New**: 이번 논문에서는 YOLOE라는 새로운 모델을 소개합니다. YOLOE는 다양한 객체 탐지 및 분할을 위한 오픈 프롬프트 메커니즘을 통합한 고효율 모델로, 실시간으로 모든 것을 볼 수 있는 능력을 가지고 있습니다. 텍스트 프롬프트에 대해 Re-parameterizable Region-Text Alignment(RepRTA) 전략을 적용하여, 사전 훈련된 텍스트 임베딩을 효율적으로 개선하고 시각적-텍스트 정렬을 향상시킵니다.

- **Technical Details**: YOLOE는 컴퓨터 비전의 핵심 과제인 객체 탐지 및 분할을 위해 설계된 통합 모델입니다. 이 모델은 텍스트 프롬프트, 비주얼 프롬프트 및 프롬프트가 없는 상황에 대해 각각 RepRTA, Semantic-Activated Visual Prompt Encoder(SAVPE), Lazy Region-Prompt Contrast(LRPC) 전략을 적용하여 최적의 성능을 기록합니다. 각 전략은 객체 탐지 및 분할의 효율성을 높이는데 기여하며, 전체 아키텍처는 YOLO 모델과 동일한 구조로 추가 오버헤드 없이 운영됩니다.

- **Performance Highlights**: YOLOE는 다양한 프롬프트 메커니즘에서 뛰어난 탐지 및 분할 성능을 보여줍니다. LVIS 데이터셋에서 YOLOE-v8-S는 YOLO-Worldv2-S를 3.5 AP 상승시키며, 3배 적은 훈련 비용과 1.4배 빠른 추론 속도를 달성했습니다. COCO 데이터셋에서도 YOLOE-v8-L은 YOLOv8-L에 비해 0.6 AP 향상을 도모하며, 약 4배 적은 훈련 시간을 기록하여 효율성을 강화하였습니다.



### Anatomy-Aware Conditional Image-Text Retrieva (https://arxiv.org/abs/2503.07456)
Comments:
          16 pages, 10 figures

- **What's New**: 이 연구에서는 해부학적 위치에 기반한 이미지-텍스트 검색(Anatomical Location-Conditioned Image-Text Retrieval, ALC-ITR) 프레임워크를 제안하며, 이는 의사나 방사선사에게 더욱 효과적인 진단 지원을 제공합니다. 이는 질병과 증상이 동일한 해부학적 위치에서 나타나는 유사 환자 사례를 검색하는 데 초점을 맞춥니다. 제안한 시스템은 세분화된 환자 사례 이해를 돕고, 관련성을 존중하는 지역 수준의 설명을 제공합니다.

- **Technical Details**: 제안된 시스템은 약하게 감독되는 지역-상관 의료 비전 언어 모델(Region-Relevance-Aligned Vision Language, RRA-VL)을 사용하여 해부학적 위치 기반 멀티모달 검색(located-conditioned multimodal retrieval, LC-MMR)을 수행합니다. 이 모델은 일반화 가능한 멀티모달 표현을 생성하며, 의료 이미지와 보고서 간의 지역 및 단어 수준 정렬을 통해 지역적 정렬을 수행합니다. 또한, 위치-조건 대조 학습(location-conditioned contrastive learning)을 통해 향상된 검색 성능을 달성합니다.

- **Performance Highlights**: 제안한 RRA-VL 모델은 상태-최상의(localization) 성능을 실현하고, 여러 하위 작업에서 경쟁력을 보여줍니다. ALC-ITR 시스템은 해부학적 지역 조건에 기반하여 설명 가능성을 높이고 초기 진단 보고서를 제공합니다. 또한 기존 솔루션과 비교할 때 우수한 멀티모달 검색 성능을 보여주며, 환자 사례 간 설명력을 강화해 기여합니다.



### EigenGS Representation: From Eigenspace to Gaussian Image Spac (https://arxiv.org/abs/2503.07446)
- **What's New**: EigenGS는 전통적인 PCA 방식과 3D Gaussian Splatting을 효과적으로 결합하여 이미지 표현 과정에서의 혁신적인 변환 파이프라인을 제공합니다. 이 방법은 새로운 이미지에 대한 Gaussian 매개변수를 즉시 초기화할 수 있어, 이전 이미지별 최적화 과정을 생략하고 수렴 속도를 대폭 향상시킵니다. 또한, 주파수 인식을 반영한 학습 메커니즘을 도입하여 여러 스케일에 적응하는 Gaussian을 모델링해 고해상도 복원 시 아티팩트를 방지합니다.

- **Technical Details**: EigenGS는 eigenspace와 이미지 공간의 Gaussian 표현을 연결하는 과정을 통해 새로운 이미지의 Gaussian 표현을 빠르게 추정할 수 있도록 설계되었습니다. Gaussian 모델의 미리 훈련된 설정을 활용함으로써, 새로운 입력 이미지에 대한 Gaussian 매개변수를 자동으로 결정할 수 있습니다. 이 방법은 Gaussian의 스케일과 이미지의 공간 주파수 간의 관계도 학습하여, 다양한 주파수를 효과적으로 모델링합니다.

- **Performance Highlights**: 광범위한 실험 결과는 EigenGS가 직관적인 2D Gaussian 피팅에 비해 우수한 복원 품질을 달성하며, 필요한 매개변수 수 및 훈련 시간을 줄일 수 있음을 보여줍니다. 특히, EigenGS는 다양한 해상도와 다채로운 카테고리의 이미지를 다룰 수 있는 능력으로 인해 실시간 애플리케이션에서도 높은 품질을 유지합니다. 이 연구는 Gaussian 기반 이미지 표현의 가능성을 제시하며, 고품질 출력과 효율적인 학습 속도를 동시에 실현합니다.



### Divide and Conquer Self-Supervised Learning for High-Content Imaging (https://arxiv.org/abs/2503.07444)
- **What's New**: 이 논문에서는 기존의 self-supervised representation learning (SSL) 방법에서 발생하는 한계를 극복하기 위해 Split Component Embedding Registration (SpliCER)이라는 새로운 구조를 소개하고 있습니다. SpliCER는 이미지를 여러 섹션으로 나누어 각 섹션에서 정보를 증류하여 모델이 더욱 섬세하고 복잡한 특징을 학습하도록 돕습니다. 이는 의료 및 지리공간 이미징과 같은 최신 기술 환경에서 효과적으로 그 성능을 발휘합니다.

- **Technical Details**: SpliCER는 이미지의 구성 요소를 분해하여 각 부분에서 특징을 학습하도록 유도하는 구조로, 현재의 self-supervised loss function과 호환되며 기존 작업에 쉽게 통합될 수 있습니다. SpliCER를 사용하면 모델이 훈련 중 각 부분에서 복잡한 특징을 잊지 않고 학습 할 수 있게 됩니다. 이를 통해 고차원 이미지의 섬세한 특징을 놓치지 않고 촘촘한 정보 맵을 생성합니다.

- **Performance Highlights**: SpliCER는 실제 의료 및 지리공간 이미징 분야에서 우수한 성능을 입증했습니다. 복잡한 특징을 학습할 수 있는 능력을 통해 모델의 다운스트림 성능 향상에 기여하며, 기존의 self-supervised 방법을 통해 발생하는 간략화된 솔루션(predictive shortcuts)을 극복할 수 있습니다. 이로 인해 과학적 발견과 분석에 필요한 중요한 정보들을 학습하는 데 효과적인 도구가 되고 있습니다.



### Open-Set Gait Recognition from Sparse mmWave Radar Point Clouds (https://arxiv.org/abs/2503.07435)
- **What's New**: 최근 mmWave 레이더 장치의 인체 감지, 특히 보행 인식에 대한 관심이 급증하고 있습니다. 이 연구는 기존의 닫힌 집합(closed-set) 시나리오와는 달리, 알려지지 않은 주체가 존재할 수 있는 열린 집합(open-set) 보행 인식 문제를 다룹니다. 이를 위해 저자는 희소한 포인트 클라우드(point clouds) 데이터를 사용하여 새로운 신경망 아키텍처를 제안합니다.

- **Technical Details**: 제안된 새로운 신경망 구조는 지도 학습(supervised classification)과 비지도 학습(unsupervised reconstruction)을 결합해 포인트 클라우드의 특징을 추출합니다. 이 아키텍처는 강력하고 정규화된(latent space) 공간을 생성하여 주행 특성을 분석합니다. 또한, 제안된 확률적 신뢰성 탐지 알고리즘은 구조화된 잠재 공간을 활용하여 미지의 주체를 탐지하고 예측 정확도와 추론 속도의 조율 가능성을 제공합니다.

- **Performance Highlights**: 저자들은 제안된 방법이 최첨단 기법보다 평균 24%의 F1 점수 향상을 보였음을 보여주었습니다. 또한, 새로운 인간 보행 데이터세트인 mmGait10을 출시하여 10명의 주체로부터 5시간 분량의 다양한 보행 모달리티를 수집했습니다. 이 연구는 여전히 제한적인 수의 대상만 있는 상황에서 효과적으로 작동하는 OSGR 시스템의 중요성을 강조합니다.



### Analysis of 3D Urticaceae Pollen Classification Using Deep Learning Models (https://arxiv.org/abs/2503.07419)
- **What's New**: 이 논문은 기후 변화로 인한 알레르기 환자의 증가와 함께, 정확한 꽃가루(classification of pollen) 분류가 필요한 상황이라는 점을 강조합니다. 기존의 2D 현미경 이미지를 사용하는 방법 대신, 3D 이미지를 전체 스택으로 활용하여 꽃가루를 분류하는 새로운 접근법을 제안합니다. 이 방법은 각 꽃가루의 알레르기 잠재력에 따라 선택된 두 개의 식물 속(속(Genus))인 Urtica와 Parietaria의 꽃가루를 대상으로 하고 있습니다.

- **Technical Details**: 논문에서는 수집된 3D 이미지 데이터셋을 사용하여, 다양한 딥러닝 모델(Deep Learning Models)의 성능을 평가합니다. ResNet3D 모델을 이용하여 최적의 레이어 선택과 확장된 학습 epochs로 학습하였고, 이를 통해 98.3%의 F1-score를 달성했습니다. 이 접근법은 데이터의 3D 정보를 최대한 보존하여, 꽃가루 분류의 정확성을 높이는 가능성을 탐구합니다.

- **Performance Highlights**: 예비 실험을 통해, 전통적인 2D 분류 방법보다 훨씬 뛰어난 성능을 보임을 확인했습니다. 모델 성능 비교 시, ResNet3D 모델이 가장 높은 정확도로 꽃가루를 분류하였으며, 이는 향후 알레르기 예방 전략 개발에 기여할 수 있음을 시사합니다. 이 연구는 꽃가루 모니터링을 위한 자동화된 분류 시스템 개발에 중요한 기초 자료를 제공합니다.



### AR-Diffusion: Asynchronous Video Generation with Auto-Regressive Diffusion (https://arxiv.org/abs/2503.07418)
Comments:
          Accepted by CVPR 2025

- **What's New**: 이번 연구에서는 Auto-Regressive Diffusion (AR-Diffusion)라는 새로운 모델을 제안합니다. 이 모델은 auto-regressive와 diffusion 모델의 장점을 결합하여 유연한 비동기 영상 생성을 실현합니다. 기존 방법들이 가진 훈련과 추론 간의 불일치 문제를 해결하며, 다양한 길이의 영상 생성에 적합한 특징을 가지고 있습니다.

- **Technical Details**: AR-Diffusion 모델은 비디오 프레임을 점진적으로 오염시키는 diffusive 방식을 통해 훈련과 추론에서의 불균형을 줄입니다. 이를 위해 non-decreasing constraint를 도입하여 이전 프레임이 후속 프레임보다 더 명확하도록 하는 제한을 두어, 자연스럽고 일관된 콘텐츠 진행을 유지합니다. 또한, FoPP 및 AD timestep scheduler를 통해 훈련 중 샘플링 균형을 맞추고 추론 시 유연한 timestep 차이를 지원합니다.

- **Performance Highlights**: 제안된 AR-Diffusion 모델은 다양한 벤치마크에서 경쟁력 있는 성능을 기록하였습니다. 특히, UCF-101 데이터셋에서는 이전의 동기화 비디오 확산 모델보다 60.1% 향상된 FVD 점수를 기록했습니다. 실험을 통해 적절한 timestep 차이가 비디오 생성 성능에 미치는 긍정적인 영향을 증명하였으며, Taichi-HD 데이터셋에서 최적의 timestep 차이는 각각 14.6 및 5.4의 FVD 점수 개선을 가져왔습니다.



### GM-MoE: Low-Light Enhancement with Gated-Mechanism Mixture-of-Experts (https://arxiv.org/abs/2503.07417)
- **What's New**: 이번 연구에서는 저조도 이미지 향상을 위한 Gated-Mechanism Mixture-of-Experts (GM-MoE)라는 첫 번째 프레임워크를 제안합니다. GM-MoE는 동적으로 조정되는 가중치 조건 네트워크와 세 개의 전문가 네트워크로 구성되어 있어 각각의 향상 작업에 특화되어 있습니다. 이 모델은 다양한 조명 장면에 적응할 수 있는 동적 가중치 조정 메커니즘을 통합하여 정보 활용도를 크게 향상시킵니다.

- **Technical Details**: GM-MoE는 개선된 U-Net 아키텍처를 바탕으로 한 시스템으로, 서로 다른 이미지 향상 작업인 색상 보정, 세부 사항 회복과 같은 문제를 해결하기 위한 세 개의 서브 전문가 네트워크를 포함합니다. 각 서브 전문가 네트워크는 다양한 조명 조건에 따라 적절한 가중치를 할당받아 균형 있는 이미지 향상을 달성합니다. 이 과정에서 지역 및 전역 특징 융합이 통합되어 다중 스케일 특징을 캡처하여 이미지 품질을 향상시킵니다.

- **Performance Highlights**: 실험 결과 GM-MoE는 25개 비교 방법에 비해 뛰어난 일반화 성능을 달성하고, 5개의 벤치마크에서 PSNR에서 최첨단 성능을, 4개의 벤치마크에서 SSIM에서 우수한 성능을 보였습니다. GM-MoE는 여러 데이터 세트 및 하위 작업에서도 뛰어난 성능을 입증하여 저조도 이미지 향상 분야에서의 적용 가능성을 높였습니다.



### TimeStep Master: Asymmetrical Mixture of Timestep LoRA Experts for Versatile and Efficient Diffusion Models in Vision (https://arxiv.org/abs/2503.07416)
Comments:
          17 pages, 5 figures, 13 tables

- **What's New**: 이번 논문에서는 Low-Rank Adaptation (LoRA) 기법을 활용하여 Diffusion 모델의 효율적인 미세 조정을 제안합니다. 새로운 TimeStep Master (TSM) 패러다임을 도입하여, 서로 다른 타임스텝에서 다양한 LoRA 모듈을 적용함으로써 모델의 성능을 향상시키는 방법을 설명합니다. 이는 각 타임스텀에 따라 전문가를 구성하여 발생하는 노이즈 레벨을 효과적으로 캡처할 수 있도록 합니다.

- **Technical Details**: 기존의 LoRA는 모든 타임스텝에 대해 동일한 설정을 사용하여 미세 조정을 진행하는 반면, TSM은 두 단계로 구성된 효율적인 접근 방식을 활용합니다. 첫 번째 단계인 "fostering"에서는 서로 다른 LoRA를 사용하여 각 타임스텝을 조정하고, 두 번째 단계인 "assembling"에서는 여러 스케일의 전문가들이 협력하여 모델의 성능을 강화합니다. 이를 통해 각 타임스텝에서 노이즈 레벨을 보다 정확하게 예측하고, 다양한 맥락을 통합할 수 있습니다.

- **Performance Highlights**: TSM은 도메인 적응, 사전 훈련 후 조정, 모델 증류와 같은 여러 LoRA 관련 작업에서 최상의 성능을 나타냅니다. UNet, DiT, MM-DiT 구조에서 모두 최고의 결과를 기록했으며, 특히 T2I-CompBench에서 두드러진 성과를 보였습니다. 자원 소모가 적으면서도 COCO2014에서 9.90의 FID를 달성하여 모델의 일반화 능력을 입증하였습니다.



### REF-VLM: Triplet-Based Referring Paradigm for Unified Visual Decoding (https://arxiv.org/abs/2503.07413)
- **What's New**: 이번 연구에서 우리는 REF-VLM을 제안합니다. 이는 다양한 비쥬얼 디코딩 작업을 통합하여 훈련할 수 있는 엔드 투 엔드 프레임워크입니다. REF-VLM은 Mask-Guided Aggregation, Latent Embeddings Router, Parallel Group Hungarian Matching과 같은 새로운 구성 요소를 통합하여 다중 작업 성능과 적응성을 향상시킵니다.

- **Technical Details**: REF-VLM의 핵심은 Triplet-Based Referring Paradigm (TRP)으로 기존의 디코딩 방식에서 발생할 수 있는 의미적 모호성을 해결합니다. TRP는 시각적 개념, 디코딩 유형, 참조 토큰으로 구성된 삼중 구조를 통해, 다중 세부 수준의 참조를 지원하며, 이는 MLLM의 성능과 정확성을 크게 향상시킵니다. 또한 이를 위해 Visual Decoding Chain-of-Thought (VD-CoT)를 도입하여, 모델이 과제를 수행하기 전에 이미지를 개관하고 관련 정보를 요약하도록 합니다.

- **Performance Highlights**: REF-VLM은 Visual Understanding, Referring Expression, Open-Vocabulary Identification과 같은 다양한 표준 벤치마크에서 기존 MLLMs보다 우수한 성능을 보입니다. 연구 결과 REF-VLM은 25가지 작업 유형이 포함된 1억 개 이상의 고품질 다중 모달 대화 샘플로 구성된 VT-Instruct 데이터셋을 통해 다양한 비쥬얼 유닛의 이해와 디코딩을 위해 robust한 성능을 보여줍니다.



### Keeping Representation Similarity in Finetuning for Medical Image Analysis (https://arxiv.org/abs/2503.07399)
Comments:
          12 pages, 6 figures

- **What's New**: 이번 연구에서는 자연 이미지 데이터로 사전훈련된 Foundation 모델을 의료 이미지 분석에 효과적으로 적응시킬 수 있는 새로운 세밀조정(finetuning) 방법인 RepSim을 제안합니다. 이 방법은 세밀조정 과정에서 원래의 사전훈련된 표현(representation) 기능인 일반화 가능성을 잘 보존할 수 있음을 입증하고, 기존 방법에 비해 표현 간 유사성을 30% 이상 개선합니다.

- **Technical Details**: RepSim은 사전훈련된 표현과 세밀조정된 표현 간의 거리를 최소화하는 데 초점을 맞추며, 유사성 불변성에 기초한 학습 가능한 직교 다체(orthogonal manifold)를 제약하여 수행됩니다. 이를 위해 통계적 정보인 공분산(covariance)을 활용하여 여러 표현을 동시에 처리하는 방식으로, 효율적인 소프트 정규화(soft regularization)를 통해 근본적으로 개선된 세밀조정 방법론을 제시합니다.

- **Performance Highlights**: RepSim을 통해 다섯 개의 의료 이미지 분류 데이터셋에서 기존의 세밀조정 방법들과 비교하여 42%의 샤프니스(sharpness) 감소와 함께 경쟁력 있는 정확도를 유지하며, 여러 작업에서의 성능을 향상시킬 수 있는 잠재력을 확인하였습니다. 이러한 접근법은 다중작업 학습(multi-task learning)과 같은 다양한 상황에서도 유용할 것으로 예상됩니다.



### Brain Inspired Adaptive Memory Dual-Net for Few-Shot Image Classification (https://arxiv.org/abs/2503.07396)
- **What's New**: 이번 연구에서 제안된 SCAM-Net은 인간의 보완적 학습 시스템에서 얻은 영감을 바탕으로 하여, 일반화 최적화된 시스템 통합(Generalization-optimized Systems Consolidation) 메커니즘을 채택한 이중 네트워크 모델입니다. 이는 극소 샷(Few-shot) 학습 시 의의 있는 특징을 식별하는 어려움을 해결하여, 의미적(feature) 대표성을 높이고 적응형 메모리 조정 메커니즘을 도입하여 신뢰성을 더욱 개선합니다.

- **Technical Details**: SCAM-Net은 해마(Hippocampus)와 신피질(Neocortex) 이중 네트워크로 설계되어 있으며, 구조화된 정보를 통합하고 장기 기억을 통해 적응적으로 조절하는 특징이 있습니다. Neocortex 모델은 공간적(spatial) 및 의미적(semantic) 특징을 통합하여 해마 모델의 표현력을 향상시키고, 해마 모델은 네트워크 가중치의 지수 이동 평균(Exponential Moving Average, EMA)의 느린 업데이트를 통해 Neocortex 모델의 학습을 제어합니다.

- **Performance Highlights**: 광범위한 실험 결과, 제안된 SCAM-Net 모델은 4개의 벤치마크 데이터셋에서 최첨단 성능(State-of-the-art performance)을 달성하였습니다. 미니이미지넷(miniImagenet) 데이터셋에서의 적응형 메모리 조정(Ablation study) 및 시각화 결과는 모델의 우수성을 입증하고 있습니다.



### SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion Models (https://arxiv.org/abs/2503.07392)
- **What's New**: 본 논문에서는 T2I (text-to-image) 확산 모델에서 개념 삭제(concept erasure)의 필요성을 강조하며, 새로운 접근 방식인 SPEED를 제안합니다. SPEED는 비대상 개념(non-target concepts)에서 영향력 있는 요소를 유지하고, 효율적으로 특정 개념을 삭제할 수 있는 기술입니다. 이 방법은 전통적인 방식과 달리 이미지 품질을 유지하면서도 빠르고 정확한 개념 삭제를 가능하게 합니다.

- **Technical Details**: SPEED는 무-null 공간(null-space) 제약 조건을 활용하여 개념 삭제를 수행하는 편집 기반(editing-based) 방법입니다. 특히, Influence-based Prior Filtering (IPF), Directed Prior Augmentation (DPA), Invariant Equality Constraints (IEC) 같은 기술을 도입하여 편집 과정에서 원래 의도를 유지하면서도 효과적인 개념 삭제를 보장합니다. 이를 통해 여러 개념을 동시에 효율적으로 삭제할 수 있는 가능성을 높였습니다.

- **Performance Highlights**: SPEED는 5초 만에 100개의 개념을 삭제할 수 있으며, 기존 방법들과 비교하여 비대상 개념의 보존을 지속적으로 뛰어난 성능을 보입니다. 다양한 삭제 작업에서 SPEED는 고품질의 결과를 도출하며, 컴퓨팅 비용을 최소화하면서도 개념 삭제의 효율성을 높였습니다.



### PersonaBooth: Personalized Text-to-Motion Generation (https://arxiv.org/abs/2503.07390)
- **What's New**: 이 논문은 Motion Personalization이라는 새로운 작업을 소개하며, 이는 Persona가 포함된 기본 모션을 사용하여 텍스트 설명에 맞춰 개인화된 모션을 생성하는 것입니다. 이를 위해 여러 배우의 독특한 페르소나를 포착한 새로운 대규모 모션 데이터셋인 PerMo(또는 PersonaMotion)를 제안합니다. 또한, 사전 훈련된 모션 확산 모델인 PersonaBooth의 다중 모달 파인튜닝 방법도 제안하여, 개인화된 모션을 현실적으로 구현합니다.

- **Technical Details**: 모델 PersonaBooth는 특정 페르소나 피처를 캡처하기 위해 학습 가능한 페르소나 토큰을 도입하고, 텍스트와 비주얼 양쪽에 대한 적응을 위한 변환 스킴을 제안합니다. 이 과정에서, 서로 다른 입력 모션 간의 일관된 페르소나 추출을 위해 контрастив learning 기법을 활용하여 같은 페르소나를 가진 샘플 간의 응집력을 강화합니다. 또한, 다양한 입력 모션의 페르소나 신호를 최대화하기 위해 context-aware fusion 메커니즘을 도입하여 페르소나 큐를 통합합니다.

- **Performance Highlights**: PersonaBooth는 현재 최신 모션 스타일 전송 방법들을 초월하는 성능을 발휘하며, 모션 개인화라는 새로운 벤치마크를 수립하였습니다. 실험 결과, Motion Personalization 및 모션 스타일 전송(MST) 작업 모두에서 뛰어난 성능을 보여주며, 여러 모션 입력의 조합을 통해 최적의 결과를 생성합니다. 이로써, 가상 환경 내 개인화된 인터랙션을 가능하게 하는 중요한 발전을 이루었습니다.



### TRCE: Towards Reliable Malicious Concept Erasure in Text-to-Image Diffusion Models (https://arxiv.org/abs/2503.07389)
- **What's New**: 최근의 텍스트-이미지 생성 모델이 고해상도 이미지를 만들 수 있는 가능성을 보여주고 있지만, 이로 인해 NSFW(Not Safe For Work) 이미지와 같은 위험한 콘텐츠 생성의 우려가 커지고 있습니다. 이러한 위험을 줄이기 위한 방법으로 개념 소거(concept erasure) 기법이 연구되고 있으나, 기존 연구는 은유적 표현 또는 적대적 프롬프트와 같이 프롬프트에 암묵적으로 삽입된 공격적인 개념을 완전히 소거하는 데 어려움을 겪고 있습니다. 본 연구에서는 TRCE라는 새로운 방법을 제안하여, 이러한 개념을 효과적으로 소거하면서도 모델의 정상적인 이미지 생성 능력을 유지할 수 있도록 하였습니다.

- **Technical Details**: TRCE는 두 단계 개념 소거 전략을 사용하여 악성 개념을 신뢰성 있게 소거하고 지식 보존을 동시에 달성합니다. 첫 번째 단계에서는 프롬프트에 암묵적으로 포함된 악성 의미를 소거하는 'Textual Semantic Erasure'를 수행하며, [EoT] 임베딩을 활용하여 악성 프롬프트를 안전한 개념을 포함한 맥락적으로 유사한 프롬프트로 매핑합니다. 두 번째 단계인 'Denoising Trajectory Steering'에서는 초기의 노이즈 제거 과정에서 안전 방향으로 예측을 조정하여 악성 콘텐츠 생성을 방지합니다.

- **Performance Highlights**: TRCE는 다양한 악성 개념 소거 벤치마크에 대해 포괄적인 평가를 수행하였으며, 그 결과 악성 개념을 효과적으로 소거하면서도 모델의 원래 생성 능력을 보다 잘 보존하는 성능을 보여주었습니다. 이 방식은 다양한 프롬프트 및 다중 개념 소거 시나리오에서 검증되었으며, 실험 결과는 TRCE가 보편적으로 잘 작동하는 것을 입증하였습니다.



### Probabilistic Segmentation for Robust Field of View Estimation (https://arxiv.org/abs/2503.07375)
- **What's New**: 본 논문은 자율주행 차량(AV)의 안전한 배치를 위협하는 감지 및 인식 공격에 대한 해결책을 제시합니다. 특히, 이 연구는 자율적인 FOV(시야) 추정을 위한 최초의 알고리즘을 개발하고, 이를 위한 진실 데이터 기반 FOV 라벨을 포함한 최초의 데이터셋을 생성했습니다. 학교에서 수집한 데이터는 자율주행 차량의 실제 적용에 필요한 FOV 추정의 강인성을 높이기 위해 보강되었습니다.

- **Technical Details**: 연구에서 제안하는 새로운 FOV 추정기는 학습 기반 세분화 모델을 채택하며, Monte Carlo dropout(MCD)를 통해 불확실성을 정량화합니다. 이 모델은 FOV의 특성을 포착하고, 신뢰도 지도를 기반으로 이상 탐지를 수행합니다. 또한, 심층 신경망(DNN)을 사용하여 FOV 예측의 확률 분포를 형성하고, 적대적 환경에서도 높은 강인성을 발휘합니다.

- **Performance Highlights**: 논문은 다양한 환경에서 FOV 추정기의 강인성과 일반화 능력을 평가하며, MCD를 통한 불확실성 인지를 개선하여 보안 감지 솔루션을 강화를 추진합니다. DNN 기반 FOV 추정은 높은 속도와 정확성을 목표로 하여 자율 시스템에 효과적으로 배포될 수 있는 가능성을 보여줍니다. 이 연구는 자율주행 환경에서의 보안과 안정성을 강화하는 데 기여할 것으로 기대됩니다.



### HGO-YOLO: Advancing Anomaly Behavior Detection with Hierarchical Features and Lightweight Optimized Detection (https://arxiv.org/abs/2503.07371)
Comments:
          10 pages

- **What's New**: 이번 연구에서는 HGO-YOLO라는 모델을 제안하고 있습니다. 이 모델은 YOLOv8에 HGNetv2 아키텍처를 통합하여, 하드웨어 제약이 있는 환경에서 정확성과 속도를 균형 있게 유지할 수 있도록 설계되었습니다. 간소화된 모델 복잡성을 위해 GhostConv를 활용한 점이 특징입니다.

- **Technical Details**: HGO-YOLO는 확장된 receptive field를 통해 더 넓은 범위의 특징을 포착하며, OptiConvDetect라는 경량 탐지 헤드를 도입하여 효과적인 탐지 헤드 구축을 위해 파라미터 공유를 활용합니다. 이 모델은 단지 4.6 MB의 크기로, CPU에서 56 FPS 프레임 속도를 기록하고 있습니다. 성능 평가 결과, 제안된 알고리즘은 mAP@0.5가 87.4%, recall rate는 81.1%를 달성했습니다.

- **Performance Highlights**: HGO-YOLO는 정확도를 3.0% 향상시키는 동시에 계산 부하를 51.69% 줄였습니다 (8.9 GFLOPs에서 4.3 GFLOPs로 감소). 추가로, 실시간 테스트는 Raspberry Pi4와 NVIDIA 플랫폼에서 수행되어 HGO-YOLO 모델이 이상 행동 탐지에서 우수한 성능을 발휘함을 나타냈습니다.



### LEGO-Motion: Learning-Enhanced Grids with Occupancy Instance Modeling for Class-Agnostic Motion Prediction (https://arxiv.org/abs/2503.07367)
Comments:
          8 pages, 4 figures

- **What's New**: 이번 연구에서는 LEGO-Motion이라는 새로운 현상 기반(class-agnostic) 모션 예측 프레임워크를 제안합니다. 이 프레임워크는 인스턴스(instance) 기능을 Bird's Eye View (BEV) 공간에 통합하여 정확하고 신뢰할 수 있는 모션 예측을 가능하게 합니다. 기존의 객체 중심(object-centric) 방법의 한계를 극복하고, 물리적 일관성을 보장함으로써 교통 참여자 간의 상호작용을 반영합니다. 또한, nuScenes 데이터 세트에서 최첨단 성능을 기록하며, 실제 활용 가능성을 보여줍니다.

- **Technical Details**: LEGO-Motion은 다음 세 가지 주요 요소로 구성됩니다: (1) BEV 인코더, (2) 상호작용 증강 인스턴스 인코더(Interaction-Augmented Instance Encoder), (3) 인스턴스 강화 BEV 인코더(Instance-Enhanced BEV Encoder). 이 구조는 상호작용 관계와 물리적 일관성을 개선하여 환경을 보다 정확하고 견고하게 이해할 수 있도록 돕습니다. 특히, 인스턴스 인코딩 과정에서 소셜 상호작용 관계를 추출할 수 있는 주의 메커니즘을 포함하고 있습니다.

- **Performance Highlights**: LEGO-Motion은 nuScenes 데이터 세트의 다양한 도전적인 시나리오에서 모션 예측 정확도와 견고성을 크게 향상시켜 기존의 최첨단 방법을 초월하는 성능을 보였습니다. 또한, 고급 FMCW LiDAR 벤치마크에서도 효과성을 검증하여 실제 응용 가능성을 입증합니다. 코드가 공개될 예정이며, 이는 향후 연구를 촉진하는 데 기여할 것입니다.



### MM-Eureka: Exploring Visual Aha Moment with Rule-based Large-scale Reinforcement Learning (https://arxiv.org/abs/2503.07365)
- **What's New**: 이번 논문에서는 멀티모달(Multimodal) 추론 모델인 MM-Eureka를 소개하며, 대규모 규칙 기반 강화 학습(RL)을 성공적으로 멀티모달 추론에 적용하였음을 보여줍니다. 기존의 텍스트 기반 RL 시스템에서 나타나는 주요 특성(예: 정확성 보상 및 응답 길이의 지속적 증가)을 멀티모달 환경에서도 재현하여, 주목할만한 데이터 효율성을 발휘한다고 밝혔습니다. 이 연구는 감독된 미세 조정 없이도 강력한 멀티모달 추론 능력을 개발할 수 있음을 보여줍니다.

- **Technical Details**: MM-Eureka 모델은 InternVL2.5를 기반으로 하여, 다양한 크기의 모델에서 대규모 RL을 적용하였습니다. 이 연구에서는 DeepSeek-R1과 유사한 RL 알고리즘을 사용하여, 훈련에 필요한 규칙 기반 보상 체계를 마련하였고, 다양한 시나리오에서 실험을 통해 그 성능을 분석하였습니다. 훈련 과정에서 시각적 'aha moments'와 같은 반성 및 되짚기 과정을 관찰하며, 규칙 기반 RL의 효율성을 입증하였습니다.

- **Performance Highlights**: MM-Eureka는 54K개의 이미지-텍스트 데이터로 훈련되어 여러 벤치마크에서 평균 성능이 우수하며, MPO로 훈련된 1M 데이터 모델보다 뛰어난 성능을 보여줍니다. 특히 MM-Eureka-Zero는 8K 이미지-텍스트 데이터만으로도 특정 벤치마크에서 기존 모델을 능가하는 성과를 기록하며, K12 벤치마크에서 8.2%의 정확도 향상을 달성했습니다. 이 논문은 커뮤니티와의 협업을 위해 코드, 모델 및 데이터를 완전하게 오픈소스로 공개하며, 멀티모달 추론 연구를 촉진할 수 있는 포괄적인 프레임워크를 제안하고 있습니다.



### Inversion-Free Video Style Transfer with Trajectory Reset Attention Control and Content-Style Bridging (https://arxiv.org/abs/2503.07363)
- **What's New**: 이 논문에서는 Trajectory Reset Attention Control (TRAC)라는 새로운 비가역 방식의 스타일 전송 방법을 소개합니다. 이 방법은 데이터를 손실 없이 스타일을 전송할 수 있도록 해 주며, 기존의 스타일과 내용이 분리되는 방법의 문제를 해결합니다. TRAC는 중간 잠재(latent)를 매 타임스텝에서 초기화하여 스타일과 내용을 함께 통합하는 방식으로 작동합니다.

- **Technical Details**: TRAC는 스타일 전송 과정에서 원본 내용 구조를 유지하면서 직접 스타일을 주입할 수 있게 설계되었습니다. 이 방법은 스타일을 추출하기 위해 Multimodal Large Language Model (MLLM)을 사용하고, Style Medium이라는 개념을 도입하여 콘텐츠와 스타일을 연결하는 매개체로 작용합니다. 이는 GPU 사용을 최소화함으로써 계산 비용을 줄이는데 큰 도움이 됩니다.

- **Performance Highlights**: 실험 결과, 제안된 TRAC 프레임워크는 콘텐츠 무결성을 유지하면서도 다양한 스타일화된 출력물을 생성할 수 있음을 보여줍니다. 이 프레임워크는 높은 품질의 스타일이 일관된 비디오를 제작하며, 스타일 정보와 내용 정보의 통합이 효과적으로 이루어졌습니다. 최종적으로, 이 방법은 다양한 상황에서 유연하고 강력한 솔루션을 제공함을 증명했습니다.



### Certifiably Optimal Anisotropic Rotation Averaging (https://arxiv.org/abs/2503.07353)
- **What's New**: 이 논문은 회전 평균화(rotation averaging) 문제를 해결하는 데 있어, 비등방성(anisotropic) 비용을 포함하는 방식을 제안합니다. 기존의 많은 방법들이 등방성(isotropic) 설정에 초점을 맞추고 있는 반면, 비등방성 설정에서는 정밀한 불확실성(uncertainty) 정보를 최적화 작업에 포함시킴으로써 해결 품질이 개선될 수 있음을 입증합니다. 이 연구는 이러한 비등방성 비용을 적절히 통합할 수 있는 공인된 최적의 방법론을 제시합니다.

- **Technical Details**: 회전 평균화 문제를 효과적으로 해결하기 위해, 이 논문은 비등방성 코드 거리(anisotropic chordal distance)를 최적화하는 SDP(semidefinite programming) 공식을 개발합니다. 비등방성 비용을 최적화하면 솔루션이 비결정적(detached) 매트릭스로 나오는 문제가 발생할 수 있으며, 이는 비등방성 비용에 대한 직접적인 수정을 통해 해결하기 어렵습니다. 이에 대해, 저자들은 회전 매트릭스의 볼록 껍질(convex hull)을 제약하는 새로운 완화(relaxation)을 제시하여, 글로벌 최적값(global optimum)을 복원할 수 있음을 입증합니다.

- **Performance Highlights**: 실험 결과, 제안한 비등방성 모델은 표준 등방성 코드 벌점보다 일반적으로 더 높은 정확도를 달성하며, 다양한 데이터셋에서 글로벌 최적 솔루션을 복원하는 데 성공했습니다. 이 논문에서 소개된 방정식은 오랜 시간 동안 돌아온 회전 평균화 문제 해결에 실질적인 진전을 가져오며, 앞으로 이 분야의 연구에 기여할 것으로 예상됩니다.



### Fully Unsupervised Annotation of C. Elegans (https://arxiv.org/abs/2503.07348)
- **What's New**: 이번 연구에서는 가우시안 분포를 가정하는 새로운 비지도 다중 그래프 매칭 접근 방식을 제시합니다. 자가지도 학습(self-supervised learning)을 위해 사이클 일관성(cycle consistency)을 손실(loss)로 활용하고, 베이지안 최적화(Bayesian Optimization)를 통해 가우시안 파라미터를 결정합니다. 이 방식은 대규모 데이터셋에서 효과적으로 확장할 수 있으며, 세포 핵(annotation of cell nuclei)을 분류하는 정확도가 최신 감독 방법과 유사한 수준인 96.1%에 도달하였습니다.

- **Technical Details**: 제안된 방법은 C. elegans의 세포 핵에 대한 통계적 아틀라스(statistical atlas)를 구축하는 방식으로 구성됩니다. 이 아틀라스는 주어진 학습 세트에서 세포 인스턴스 분할(cell instance segmentation)만을 사용하여 구축되며, 수동 주석(manual labeling)이 필요하지 않습니다. 또한, 이 연구는 가우시안 매칭 비용의 파라미터를 직접 최적화하는 베이지안 최적화를 활용하여, 고급 생물 의학 이미지에 적합한 일반화를 제공합니다.

- **Performance Highlights**: 최신 감독 방법의 정확도에 필적하는 성능을 달성하며, 3D 현미경 이미지에서 C. elegans 세포 핵을 주석 처리하는 과제를 해결했습니다. 제안된 접근 방식은 C. elegans의 무감독 아틀라스를 최초로 생성하며, 추가적인 생물 모델에 대해서도 적용 가능성을 보여줍니다. 이를 통해 다양한 생물학적 연구에 있어서 비교 개발 연구(comparative developmental studies)를 촉진할 잠재력을 지니고 있습니다.



### DaD: Distilled Reinforcement Learning for Diverse Keypoint Detection (https://arxiv.org/abs/2503.07347)
- **What's New**: 이번 논문에서는 Structure-from-Motion (SfM) 시스템을 위한 완전 자기 감독(self-supervised) 및 설명자(descriptor) 없는 키포인트(keypoint) 탐지 목표를 제안합니다. 기존의 방법들은 설명자에 의존하고 있어 바람직하지 않으며, 이를 해결하기 위해 강화 학습(reinforcement learning)을 활용합니다. 또한, 우리는 균형 잡힌 top-K 샘플링 전략을 통해 이를 훈련 동안 저하되지 않도록 보장하고, 두 가지 질적으로 다른 종류의 탐지기가 출현하는 문제를 해결하기 위해 새로운 탐지기 DaD를 훈련합니다.

- **Technical Details**: 논문에서는 키포인트 탐지를 위해 강화 학습을 사용하는 접근 방식을 설명합니다. 이 과정에서, 균형 잡힌 top-K 샘플링 전략을 통해 탐지기의 성능을 개선하고, 경량(light) 및 암흑(dark) 키포인트를 모두 탐지할 수 있는 다양한 탐지기로 처리하기 위한 포인트-와이즈 최대 증류(point-wise maximum distillation) 목표를 제안합니다. 이러한 방법론은 기존의 설명자를 통해 유도되었던 의존성을 제거하고, 균형 잡힌 탐지기 훈련을 가능하게 합니다.

- **Performance Highlights**: 제안된 DaD 탐지기는 다양한 벤치마크에서 새로운 최첨단(state-of-the-art) 성능을 기록하며, 512개에서 8192개까지의 키포인트 예산에 대해 뛰어난 결과를 보여줍니다. 그러므로 이 방식이 기존의 모든 지표를 초월하는 결과를 도출했음을 나타냅니다. 이러한 성과는 정량적 실험을 통해 확인되었으며, 연구 결과는 공개될 예정입니다.



### Now you see me! A framework for obtaining class-relevant saliency maps (https://arxiv.org/abs/2503.07346)
- **What's New**: 이 논문에서는 신경망(Neural Networks)의 입력 특성이 특정 예측을 위해 어떻게 작용하는지를 이해하기 위한 새로운 프레임워크를 제안합니다. 기존의 Saliency Maps는 일반적으로 너무 포괄적이며, 특정 분류에 기여하는 정보를 정확하게 포착하지 못하는 경향이 있었습니다. 이 연구는 클래스 전반의 기여도를 통합하여 진정으로 클래스와 관련된 정보를 반영하는 Saliency Maps를 생성하는 방법을 제시합니다.

- **Technical Details**: 제안된 프레임워크는 여러 모델 아키텍처와 기여도 기법에 독립적이어서 다양한 상황에서 유연하게 적용될 수 있습니다. 연구에서는 grid-pointing 게임과 랜덤화 기반의 산성 검사를 포함한 여러 검증 된 벤치마크에서 성능을 평가하였습니다. 이 프레임워크는 어떤 모델 예측에서 구별되는 특성과 공유되는 특성을 식별하는 데 도움이 됩니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 표준 Saliency Map 접근 방식의 성능을 크게 향상시켰습니다. 새로운 프레임워크는 신경망의 예측을 해석하는 데 있어 이해 가능성과 투명성을 높이는 데 기여합니다. 이러한 결과는 신뢰성이 중요한 고위험 설정에서도 더욱 유용할 것으로 기대됩니다.



### Unleashing the Potential of Large Language Models for Text-to-Image Generation through Autoregressive Representation Alignmen (https://arxiv.org/abs/2503.07334)
- **What's New**: Autoregressive Representation Alignment (ARRA)은 LLMs의 이미지 생성 능력을 극대화하기 위해 글로벌 제약 조건을 주입한 새로운 훈련 프레임워크입니다. 기존의 복잡한 아키텍처 변경 없이 LLM의 은닉 상태를 외부의 비주얼 기초 모델과 정렬하여 텍스트-이미지 생성에서의 일관성을 강화합니다. 이 프레임워크는 <HYBNEXT>라는 하이브리드 토큰을 도입하여 로컬 최적화와 글로벌 의미 증류를 동시에 수행할 수 있도록 합니다.

- **Technical Details**: ARRA는 글로벌 비주얼 정렬 손실(global visual alignment loss)을 통해 LLM의 잠재 표현을 외부 모델(예: BioMedCLIP)과 정렬시킴으로써 훈련 목표를 재정의합니다. 이 과정에서 <HYBNEXT>라는 새로운 토큰을 도입하여 로컬 차원과 글로벌 차원 모두에서 작업할 수 있게 합니다. 이 방법은 기존 아키텍처를 변경하지 않고도 LLM의 이미지 생성능력을 증진시키는데 기여합니다.

- **Performance Highlights**: ARRA는 Chameleon과 LlamaGen과 같은 고급 LLM에서 MIMIC-CXR, DeepEyeNet, ImageNet 데이터셋에 대해 FID를 각각 25.5%, 8.8%, 7.5% 감소시키며, 아키텍처의 변경 없이 뛰어난 성능을 보여줍니다. 또한 의료 이미징과 같은 특수 도메인으로의 적응이 가능하며, 일반적인 LLM을 특수 모델과 결합하여 FID를 18.6% 줄이는 성과를 달성하였습니다.



### Mitigating Hallucinations in YOLO-based Object Detection Models: A Revisit to Out-of-Distribution Detection (https://arxiv.org/abs/2503.07330)
- **What's New**: 이 논문은 객체 탐지 시스템에서 동적 환경에서의 안전한 의사결정을 보장하기 위한 연구이다. 특히, 과신(overconfidence)으로 인한 허위 탐지(hallucination)를 줄이기 위한 새로운 접근 방식을 제안한다. 기존의 Out-of-distribution (OoD) 탐지 방법의 한계를 분석하고, YOLO 모델의 성능 개선을 위한 방법론을 제시하였다.

- **Technical Details**: 논문에서는 기존 OoD 검증 데이터셋의 품질 문제를 분석하고, 이를 통해 높은 허위 양성률(false positive rates)의 원인을 규명하였다. 제안된 방법론은 객체 탐지기의 결정 경계를 조정하는 방식으로, 직접적으로 객체 존재를 인식하는 경우에만 OoD 필터링이 작동하도록 한다. 또한, 새롭게 정의한 '주변 OoD(proximal OoD)' 샘플을 사용하여 자율주행 기준 BDD-100K에서 전체 허위 탐지 오류를 88% 감소시키는 결과를 얻었다.

- **Performance Highlights**: 제안된 방식은 기존의 YOLO 기반 탐지기에서 허위 탐지를 효과적으로 줄이는 것으로 입증되었으며, 실험 결과는 OoD 샘플에 대한 미세 조정(fine-tuning) 전략이 효과적이라는 것을 보여준다. 논문에서 제시한 방법은 ID 데이터와 유사한 특성을 가진 OoD 샘플을 활용하여 결정 경계를 형성하여, 보다 나은 성능을 발휘하도록 했다. 이러한 접근법은 새로운 객체나 이상 탐지 시 허위 탐지를 줄이는 데 기여할 수 있다.



### Automated Movie Generation via Multi-Agent CoT Planning (https://arxiv.org/abs/2503.07314)
Comments:
          The code and project website are available at: this https URL and this https URL

- **What's New**: 이 논문에서는 MovieAgent를 소개하며, 자동화된 영화 생성의 패러다임을 탐구하고 정의합니다. MovieAgent는 스크립트와 캐릭터 뱅크를 기반으로 연결된 내러티브를 가진 다중 장면 및 다중 샷의 긴 형식 비디오를 생성할 수 있습니다. 이 프레임워크는 캐릭터 일관성, 자막 동기화 및 음향 안정성을 보장하며, 이는 기존의 수동 촬영 방식의 단점을 극복합니다.

- **Technical Details**: MovieAgent는 계층적 CoT(Chain of Thought) 기반의 추론 프로세스를 적용하여 자동적으로 장면을 구성하고 촬영 설정을 최적화합니다. 이 시스템은 감독, 각본가, 스토리보드 아티스트 및 위치 관리자의 역할을 모의하는 여러 LLM(대형 언어 모델) 에이전트를 사용하여 제작 파이프라인을 간소화합니다. 이 계층적 프레임워크는 결정 과정을 단계별로 해석 가능하게 하여 미세한 조정을 가능하게 합니다.

- **Performance Highlights**: 실험 결과에 따르면, MovieAgent는 스크립트 충실성, 캐릭터 일관성 및 내러티브 응집력에서 새로운 최첨단 성능을 달성했습니다. 이 시스템은 완전 자동화된 영화 생성 분야에서는 뛰어난 성능을 자랑하며, 고도의 계획을 필요로 하지 않고도 매끄러운 내러티브 구조를 만듭니다. 이로 인해 공동 작업을 통해 영화 제작의 중요한 요소를 쉽고 일관되게 구성할 수 있습니다.



### AttenST: A Training-Free Attention-Driven Style Transfer Framework with Pre-Trained Diffusion Models (https://arxiv.org/abs/2503.07307)
- **What's New**: Diffusion 모델은 스타일 전송 작업에서 놀라운 발전을 이루었지만, 기존의 방법들은 보통 미세 조정(fine-tuning) 또는 사전 훈련된 모델 최적화에 의존하여 높은 계산 비용과 콘텐츠 보존(content preservation)과 스타일 통합(style integration) 간 균형을 맞추는 데 어려움을 겪습니다. 이를 해결하기 위해 제안된 AttenST는 훈련 없이 스타일 전송을 수행할 수 있는 주의(attention) 기반 프레임워크입니다. 스타일 가이드 자가 주의 메커니즘(style-guided self-attention)은 콘텐츠 이미지의 쿼리(query)을 유지하면서 스타일 이미지의 키(key) 및 값(value)로 교체하여 효과적인 스타일 피처 통합을 가능하게 합니다.

- **Technical Details**: AttenST는 스타일 전송 품질을 최적화하기 위해 5th 및 6th transformer 블록에서 스타일 가이드 자가 주의 메커니즘을 구현하였습니다. 또한, 스타일 정보를 손실을 줄이기 위한 스타일 보존 역전(style-preserving inversion) 전략을 도입하며, 여러 재샘플링 단계를 거쳐 역전 정확도를 개선합니다. 콘텐츠 통계(content statistics)를 통합하는 Content-Aware Adaptive Instance Normalization(CA-AdaIN)을 통해 스타일 융합(style fusion)을 최적화하고 콘텐츠 저하(content degradation)를 완화합니다.

- **Performance Highlights**: AttenST는 기존의 방법들보다 뛰어난 성능을 보여주며, 스타일 전송 데이터셋에서 최첨단(state-of-the-art) 성과를 달성했습니다. 이 연구에서는 duual-feature cross-attention 메커니즘을 통해 콘텐츠와 스타일 피처(feature)를 융합하여 구조적 충실도(structural fidelity) 및 스타일 표현(stylistic expression)의 조화를 이끌어 냈습니다. 실험 결과 AttenST는 스타일 주입(style injection)과 콘텐츠 보존을 효과적으로 균형 잡아 더욱 매력적인 이미지를 생성하는 데 성공했습니다.



### ALLVB: All-in-One Long Video Understanding Benchmark (https://arxiv.org/abs/2503.07298)
Comments:
          AAAI 2025

- **What's New**: 최근 발표된 ALLVB (ALL-in-One Long Video Understanding Benchmark)는 멀티모달 대형 언어 모델(Multi-modal LLMs, MLLMs)의 긴 비디오 이해 능력을 평가하기 위한 포괄적인 벤치마크입니다. 이 벤치마크는 9개의 주요 비디오 이해 작업을 통합하여 단일 평가 기준으로 9가지 다양한 비디오 이해 능력을 평가하는 데 중점을 두고 있습니다. 또한, 완전 자동화된 주석 처리 파이프라인을 통해 수천 개의 Q&A(질문과 답변)를 생성하여 기존의 긴 비디오 벤치마크 소스에서의 확장성과 유지 관리를 용이하게 합니다.

- **Technical Details**: ALLVB는 1,376개의 긴 비디오와 총 252,000개의 Q&A를 포함하고 있으며, 평균 길이는 거의 2시간에 달합니다. 이 벤치마크는 91개의 서브 태스크를 포함해 비디오 이해에 필요한 다양한 질문 템플릿을 설계하여 MLLMs의 능력을 종합적으로 평가합니다. 텍스트 기반의 비디오 콘텐츠 설명과 GPT-4o를 사용한 두 단계의 세분화 방법을 통해 Q&A 생성을 자동화하고, 품질 관리를 위해 여러 단계의 수동 검토를 수행합니다.

- **Performance Highlights**: ALLVB를 통해 다양한 대표적인 MLLMs를 테스트한 결과, 가장 발전된 상업 모델조차도 개선의 여지가 상당함을 보여주었습니다. 이 결과는 벤치마크의 도전적인 성격을 반영하며 긴 비디오 이해 분야에서의 발전 가능성을 강조합니다. 현재 ALLVB는 비디오 수, 평균 길이 및 Q&A 수에서 가장 큰 긴 비디오 이해 벤치마크로 자리 잡고 있습니다.



### Distilling Knowledge into Quantum Vision Transformers for Biomedical Image Classification (https://arxiv.org/abs/2503.07294)
Comments:
          Submitted for MICCAI 2025

- **What's New**: 이번 연구에서는 양자 비전 변환기(QViTs)가 비전 변환기(ViTs)와 양자 신경망(QNNs)의 장점을 결합한 새로운 모델로 제안됩니다. QViTs는 기존의 선형 레이어를 QNN으로 대체하여 자기 주목 메커니즘의 기능을 향상시킵니다. 이 하이브리드 접근법은 낮은 파라미터 수로도 뛰어난 성능을 달성할 수 있는 가능성을 보여줍니다.

- **Technical Details**: QViTs는 이미지 데이터를 패치 시퀀스로 처리하고, 양자역학의 특성을 활용하여 Hilbert 공간에서 정보를 표현합니다. 이를 통해 QNN은 복잡한 패턴을 더욱 효율적으로 학습할 수 있습니다. 또한, 지식 증류(Knowledge Distillation, KD)를 통해 고품질의 고전 모델로부터 얻은 지식을 바탕으로 QViTs의 성능을 향상시키는 방법도 탐구합니다.

- **Performance Highlights**: 연구 결과, QViTs는 비교 가능한 ViTs와 대조할 때 평균 ROC AUC와 정확도에서 각각 0.017 및 0.023의 향상을 이루었습니다. 또한, QViTs는 상대적으로 적은 파라미터 수에도 불구하고, 여러 분류 작업에서 전통적인 SOTA 모델과도 경쟁력을 갖추며, GFLOPs와 파라미터 수를 각각 89% 및 99.99% 감소시키는 효율성을 보여줍니다.



### A Systematic Review of ECG Arrhythmia Classification: Adherence to Standards, Fair Evaluation, and Embedded Feasibility (https://arxiv.org/abs/2503.07276)
- **What's New**: 이번 논문은 2017년부터 2024년까지 발표된 ECG(심전도) 신호 분류 연구를 체계적으로 분석한 리뷰 논문입니다. 이 연구는 AAMI(Association for the Advancement of Medical Instrumentation) 기준을 따르며, 환자 간 패러다임을 구현하고, 심전도 분류 모델의 실제 배포 가능성을 평가하는 데 중점을 두고 있습니다. 연구에 따르면, 많은 기존 연구들이 환자 독립적인 분할 및 하드웨어 제한을 적절히 고려하지 않고 있어, 실제 적용 가능성에 대한 문제가 있음이 지적되었습니다.

- **Technical Details**: 연구에서는 E3C(Embedded, Clinical, and Comparative Criteria) 기준을 준수하는 최첨단 방법들을 식별하고, 정확성(accuracy), 추론 시간(inference time), 에너지 소비(energy consumption), 메모리 사용(memory usage) 등의 요소를 비교 분석했습니다. 또한, 저전력 시스템에 배포 가능한 ECG 분류 모델을 위한 최적화 기법으로 양자화(quantization), 가지치기(pruning), 지식 증류(Knowledge distillation) 등의 방법이 중요하다고 강조하였습니다. 이와 함께 FPGA(Field-Programmable Gate Arrays)를 활용한 다양한 최적화 방안도 논의되었습니다.

- **Performance Highlights**: 전반적으로, 많은 연구가 낮은 전력 소비와 효율적인 추론 성능을 목표로 하고 있지만, 대다수의 모델이 고정된 매개변수로 외부 훈련된 모델을 기반으로 하여 제한된 적응력을 보이고 있습니다. 본 리뷰는 AAMI 권고 기준을 준수하고, 환자 간 패러다임을 효과적으로 구현한 연구들이 더 현실적인 성능 평가를 제공할 것임을 시사합니다. 이를 통해 작성된 가이드라인은 향후 심전도 분류 시스템의 실질적인 진전을 이끌어내는 데 기여할 것입니다.



### Customized SAM 2 for Referring Remote Sensing Image Segmentation (https://arxiv.org/abs/2503.07266)
- **What's New**: 이 논문에서는 텍스트 설명을 기반으로 원거리 감지 (Remote Sensing) 이미지를 세분화하는 Referring Remote Sensing Image Segmentation (RRSIS)에서 SAM 2의 적용을 위한 도전 과제를 해결하기 위해 RS2-SAM 2라는 새로운 프레임워크를 제안합니다. 이 구조는 SAM 2의 원거리 감지 특징과 텍스트 특징을 정렬하고, 의사 마스크(pseudo-mask)에 기반한 치밀한 프롬프트를 제공하며, 경계 제약(boundary constraints)을 강화하는 것을 목표로 합니다.

- **Technical Details**: RS2-SAM 2는 비주얼과 텍스트 입력을 공동으로 인코딩하는 유니온 인코더를 사용하여 시맨틱적으로 정렬된 비주얼 및 텍스트 임베딩을 생성합니다. 그 후, 이 모델은 비주얼 증강 텍스트 임베딩과 조정된 원거리 감지 비주얼 특징을 계층적으로 정렬하는 양방향 계층 합성 모듈을 설계합니다. 또한, 비주얼 임베딩과 클래스 토큰을 입력으로 받아서 SAM 2의 치밀한 프롬프트로 사용할 의사 마스크를 생성하는 마스크 프롬프트 생성기도 도입하였습니다.

- **Performance Highlights**: 본 논문의 실험 결과는 여러 RRSIS 벤치마크에서 RS2-SAM 2가 최첨단 성능을 달성했음을 보여줍니다. 특히 제안된 프레임워크는 안정적인 경계 제약을 강화하여 세분화 정확도를 높이는 데 효과적이며, 기존의 RRSIS 방법들과 비교했을 때 현저한 개선을 나타냅니다. 이러한 성과는 RRSIS 분야에서의 진전을 위한 중요한 기여로 평가됩니다.



### WISE: A World Knowledge-Informed Semantic Evaluation for Text-to-Image Generation (https://arxiv.org/abs/2503.07265)
Comments:
          Code, data and leaderboard: this https URL

- **What's New**: 본 논문에서는 텍스트-이미지(T2I) 모델의 평가를 위한 새로운 벤치마크인 WISE(World Knowledge-Informed Semantic Evaluation)를 제안합니다. 기존의 평가 기준들이 이미지 현실성과 단순한 텍스트-이미지 정합성에 주로 집중하는 반면, WISE는 보다 복잡한 의미 이해와 세계 지식 통합을 평가합니다. 또한 WiScore라는 새로운 정량적 메트릭을 도입하여 지식-이미지 정합성을 보다 rigorously 평가할 수 있도록 합니다.

- **Technical Details**: WISE는 자연 과학, 시공간 추론, 문화적 상식과 같은 세 가지 주요 영역을 포함하며, 25개의 서브 도메인에 걸쳐 1000개의 평가 프롬프트를 제공합니다. WiScore는 생성된 이미지 내에서 객체와 개체의 정확한 묘사를 강조하는 새로운 복합 메트릭으로, 일관성(Consistency), 현실성(Realism), 미적 품질(Aesthetic Quality)의 세 가지 주요 요소의 가중 평균으로 계산됩니다. 이러한 체계적인 접근 방식을 통해, 기존 T2I 모델들이 세계 지식을 효과적으로 통합하고 적용하는 데 제한적이라는 것을 보여줍니다.

- **Performance Highlights**: 20개의 T2I 모델(10개의 전용 모델 및 10개의 통합 멀티모달 모델)을 평가한 결과, 기존 T2I 모델들은 복잡한 의미 이해 및 세계 지식 통합에 있어 상당한 한계를 드러냈습니다. 통합 멀티모달 모델조차도 전용 T2I 모델에 비해 이미지 생성에 있어 우위가 확인되지 않았으며, 이는 현재의 통합 접근 방법이 이미지 생성에서 세계 지식을 효과적으로 활용하는 데 한계를 나타냅니다. 이러한 결과는 T2I 모델의 발전을 위한 주요 경로를 제시합니다.



### COMODO: Cross-Modal Video-to-IMU Distillation for Efficient Egocentric Human Activity Recognition (https://arxiv.org/abs/2503.07259)
- **What's New**: 본 논문에서는 COMODO라는 새로운 크로스 모달(self-supervised) 자기 지도 증류(framework)를 제안합니다. 이는 비디오 모달리티(video modality)의 풍부한 의미 지식을 IMU 모달리티로 전이하여 라벨이 없는 상태에서도 활용할 수 있는 방법입니다. COMODO는 비디오 인코더를 고정하여 동적 인스턴스 큐를 구축하고, 비디오와 IMU 임베딩 간의 피처 분포를 정렬하는 방식을 도입합니다.

- **Technical Details**: COMODO는 두 가지 모달리티의 시간 해상도와 피처 공간이 다름에도 불구하고, 효과적으로 비디오 도메인의 우수한 지식을 IMU 도메인으로 증류할 수 있는 기술을 필요로 합니다. 또한, 다양한 비디오 및 IMU 인코더 쌍에 대해 적용할 수 있도록 구조적 접근 방식을 갖추고 있음을 강조합니다. 이 방법은 자가 지도(self-supervised) 학습의 이점을 살려 데이터 효율성을 높이며, IMU 센서의 인지 성능을 향상시킵니다.

- **Performance Highlights**: COMODO는 세 가지 벤치마크 데이터세트에서 실험을 실시하여, 완전 감독된 모델보다 높은 성능을 보이는 결과를 보여줍니다. 또한, COMODO는 데이터의 부족함에도 불구하고 강한 크로스 데이터셋 일반화 성능을 유지합니다. 이러한 결과들은 고급 리소스 센서와 저급 리소스 센서 간의 간극을 해소할 수 있는 가능성을 제시합니다.



### AnomalyPainter: Vision-Language-Diffusion Synergy for Zero-Shot Realistic and Diverse Industrial Anomaly Synthesis (https://arxiv.org/abs/2503.07253)
Comments:
          anomaly synthesis,anomaly detection

- **What's New**: AnomalyPainter라는 새로운 zero-shot 프레임워크가 발표되었습니다. 이 프레임워크는 Vision Language Large Model (VLLM), Latent Diffusion Model (LDM), 그리고 Tex-9K라는 새로운 텍스처 라이브러리를 결합하여 다양성과 현실성을 모두 갖춘 이상치 샘플을 합성하는 데 중점을 두고 있습니다. Tex-9K는 75개 카테고리와 8,792개의 텍스처 자산으로 구성된 전문 텍스처 라이브러리로, 다양한 이상치 합성을 지원합니다. 기존 방법에서 발견된 다양성과 현실성 간의 균형 문제를 해결하며, 효과적인 실험 결과를 통해 AnomalyPainter가 기존 방법들보다 우수한 성능을 보임을 입증했습니다.

- **Technical Details**: AnomalyPainter는 세 가지 주요 단계인 전문 텍스처 라이브러리 구축, 이상치 설명 생성 및 매칭, 그리고 적응형 텍스처 이상치 생성을 구현합니다. Latent Diffusion Models는 원본 데이터 공간에서 저해상도 잠재 공간으로의 양방향 매핑을 설정하는 오토인코더와 주어진 특정 시간 단계 t와 텍스트 프롬프트 임베딩 p를 기반으로 시끄러운 잠재값을 제거하는 잠재 노이즈 제거 네트워크를 포함합니다. 이 과정은 노이즈 추가 및 노이즈 제거 프로세스를 통해 진행되며, ControlNet을 통해 정상 이미지에 텍스처를 적용하게 됩니다.

- **Performance Highlights**: AnomalyPainter는 기존의 최첨단 기술들보다 현실감과 다양성, 일반화 성능에서 뛰어난 결과를 보였습니다. 실험 결과, AnomalyPainter가 생성한 이상치 샘플은 더 자연스러운 전환을 가지고 있으며, 다양한 이상치 유형을 잘 반영합니다. 이는 궁극적으로 다운스트림 이상치 탐지 작업의 성능 향상에 기여하게 됩니다.



### Semantic Communications with Computer Vision Sensing for Edge Video Transmission (https://arxiv.org/abs/2503.07252)
- **What's New**: 이번 논문에서는 엣지 비디오 전송을 위한 SC(computer Vision Sensing) 프레임워크(SCCVS)를 제안합니다. SCCVS는 static 또는 dynamic 프레임에 따라 압축 비율(CR)을 조절해 스펙트럼 자원을 효과적으로 절약하는 CRSC 모델을 포함합니다. 또한, 실제 변화를 감지하고 각 프레임의 중요성을 평가하는 OSMS 스킴을 구현하였습니다.

- **Technical Details**: SCCVS 프레임워크는 두 개의 경량 AI 모델로 구성되어 있습니다. CRSC 모델은 Bilateral Transformer(BiFormer)와 Kolmogorov-Arnold Networks(KAN)를 사용해 의미 추출과 인코딩을 수행하며, OSMS는 Yolov10과 FastSAM을 활용하여 실시간 객체 탐지 및 의미 분할을 수행합니다. 이러한 구성은 자원 제약이 있는 엣지 환경에서도 효과적으로 작동할 수 있게 설계되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 SCCVS 프레임워크는 VIRAT Video Dataset에서 데이터 전송량을 기본 방법에 비해 약 90% 줄이는 성과를 보였습니다. 이는 스펙트럼 효율성을 개선하며 중요한 의미 정보를 손실하지 않고 전송 효율성을 향상시키는 데 기여합니다.



### Text-IRSTD: Leveraging Semantic Text to Promote Infrared Small Target Detection in Complex Scenes (https://arxiv.org/abs/2503.07249)
- **What's New**: 이 논문에서는 텍스트 기반의 안내를 활용하여 적외선 소형 표적 탐지(Infrared Small Target Detection, IRSTD) 문제를 해결하는 새로운 접근 방식을 제안합니다. 이 방법은 기존 IRSTD의 개념을 확장하여 텍스트 안내 IRSTD(Text-IRSTD)로 발전시키며, 복잡한 배경에서의 탐지 성능을 개선하는 데 기여합니다. 또한, 모호한 표적 카테고리를 수용하기 위해 새로운 퍼지 의미 텍스트 프롬프트(fuzzy semantic text prompt)를 고안하고, 텍스트와 이미지 간의 정보를 융합하기 위해 점진적 교차 모달 의미 상호 작용 디코더(Progressive Cross-Modal Semantic Interaction Decoder, PCSID)를 제안합니다.

- **Technical Details**: 제안된 Text-IRSTD는 두 가지 주요 구성 요소로 이루어져 있습니다. 첫 번째는 텍스트 안내 기능 집합(Text-Guided Feature Aggregation, TGFA) 블록으로, 이는 텍스트 정보를 이용하여 핵심 특징을 추출하는 데 도움을 줍니다. 두 번째는 텍스트 안내 의미 상호 작용(TGSI) 블록으로, 이는 두 가지 서로 다른 관점에서 특징 변조를 수행하여 텍스트와 이미지 간의 정보 상호 작용을 보다 정교하게 만들어냅니다. 또한, 다양한 시나리오에서 수집된 2,755개의 적외선 이미지로 구성된 새로운 벤치마크 데이터셋 FZDT도 구축하였습니다.

- **Performance Highlights**: 제안된 방법은 여러 공공 데이터셋에서 기존의 최첨단 알고리즘들보다 우수한 탐지 성능을 달성하여 표적 윤곽 복원에서도 뛰어난 결과를 보여주었습니다. Text-IRSTD는 보이지 않는 탐지 시나리오에서도 강력한 일반화 능력을 발휘하며, 다양한 응용 가능성을 보여줍니다. 이러한 결과는 텍스트 정보의 추가적 통합이 복잡한 환경에서의 탐지 성능을 크게 개선할 수 있음을 시사합니다.



### Retinex-MEF: Retinex-based Glare Effects Aware Unsupervised Multi-Exposure Image Fusion (https://arxiv.org/abs/2503.07235)
- **What's New**: 이번 논문은 다수의 저동적 범위 이미지를 하나의 고동적 범위 이미지로 통합하는 Multi-exposure image fusion(MEIF) 방법론을 소개합니다. Retinex 이론(Retinex theory)을 기반으로 하여 과다 노출로 인한 glare 효과를 모델링하고 이를 개선하기 위해 비지도식으로 조절 가능한 접근 방식을 제안합니다. 본 연구는 조명 구성 요소와 공유 반사 구성 요소를 분리하여 glare 효과를 효과적으로 완화할 수 있는 방법을 개발했습니다.

- **Technical Details**: 제안된 방법은 다수의 노출 이미지를 조명 구성 요소와 공유 반사 구성 요소로 분해하여 처리합니다. Bidirectional loss constraint를 활용하여 공통 반사 구성 성분을 학습하며, 조명 퓨전 기준을 설정하여 밝기 변화를 조절할 수 있도록 설계되었습니다. Retinex 이론을 사용하여 다양한 노출 수준에서 물체의 실제 특성을 유지하며, 각 이미지의 노출 조건을 독립적으로 처리함으로써 더욱 유연한 퓨전이 가능합니다.

- **Performance Highlights**: 여러 데이터셋에서 수행된 실험 결과, 본 방법은 저 노출-과다 노출 퓨전, 노출 제어 퓨전 및 동질적 극단 노출 퓨전에서 효과적인 분해 및 유연한 융합 성능을 입증했습니다. 공공 데이터셋에서 우수한 융합 성능을 나타내었으며, 노출 제어 실험을 통해 사용자의 조정 가능성을 드러냈습니다. 이러한 성과는 고해상도 이미지 생성에 중요한 기여를 할 것으로 기대됩니다.



### CoT-Drive: Efficient Motion Forecasting for Autonomous Driving with LLMs and Chain-of-Thought Prompting (https://arxiv.org/abs/2503.07234)
- **What's New**: 본 연구는 CoT-Drive라는 새로운 접근 방식을 제안하며, 이는 대규모 언어 모델(LLMs)과 연쇄적 사고(CoT) 프롬프트 기법을 활용하여 움직임 예측을 강화합니다. CoT-Drive는 경량 언어 모델(LMs)에 LLM의 고급 장면 이해 능력을 효과적으로 전달하기 위한 교사-학생 지식 증류 전략을 도입하여, 경량 모델이 실시간으로 작동할 수 있도록 합니다. 또한 Highway-Text 및 Urban-Text라는 두 가지 새로운 장면 설명 데이터셋을 제공하여 문맥에 맞는 의미론적 주석 생성을 위한 경량 LMs의 미세 조정을 지원합니다.

- **Technical Details**: CoT-Drive 프레임워크는 LLM의 고급 장면 이해 기능을 경량, 엣지 배포 모델에 통합하는 것을 목표로 합니다. 이 연구에서는 LLM GPT-4 Turbo가 "교사" 역할을 하여 경량 "학생" 모델로 지식을 전달하는 교사-학생 지식 증류 방법을 도입하였습니다. CoT 프롬프트 기법은 LLM의 통찰력을 인간 같은 인지 프로세스에 맞게 조정하여, AV의 정확한 예측을 가능하게 합니다.

- **Performance Highlights**: 다양한 실제 데이터셋을 기반으로 한 포괄적인 평가 결과, CoT-Drive는 기존 모델들을 능가하였으며 복잡한 교통 시나리오를 처리하는 데 있어 효과성과 효율성을 증명했습니다. 이 연구는 LLMs의 실제 적용 가능성을 고려한 첫 번째 사례로, 경량 LLM 대리 모델의 훈련 및 사용을 선도하여 새로운 기준을 설정하고 AD 시스템에 LLM을 통합할 잠재력을 보여줍니다.



### Boosting Diffusion-Based Text Image Super-Resolution Model Towards Generalized Real-World Scenarios (https://arxiv.org/abs/2503.07232)
- **What's New**: 본 논문에서는 저해상도 텍스트 이미지 복원을 위한 새로운 프레임워크를 제안합니다. 이는 특히 텍스트의 사실성과 스타일을 보존하는 것에 중점을 두고 있습니다. 제안된 방법은 텍스트 이미지의 복원에서 확산 모델의 일반화 능력을 향상시키기 위해 다양한 이미지 타입을 포함하는 점진적 데이터 샘플링 전략을 채택합니다.

- **Technical Details**: 모델 아키텍처는 사전 학습된 SR을 활용하여 견고한 공간적 추론 능력을 제공하며, 이는 텍스트 정보의 보존 능력을 향상시킵니다. 교차 주의 메커니즘(cross-attention mechanism)을 사용하여 텍스트 선행 정보와의 통합을 개선하고, 교수 점수(confidence scores)를 활용하여 훈련 중 텍스트 특성의 중요성을 동적으로 조정합니다.

- **Performance Highlights**: 실제 환경 데이터셋에서 수행된 광범위한 실험을 통해, 제안된 접근법이 텍스트 이미지의 시각적 리얼리즘을 향상시킬 뿐만 아니라, 텍스트 구조의 정확성도 개선됨을 보여주었습니다. 또한, 새롭게 도입된 텍스트 이미지 SR 방식이 기존 방법들에 비해 더 나은 성능과 품질을 달성했음을 확인했습니다.



### A Deep Learning Architecture for Land Cover Mapping Using Spatio-Temporal Sentinel-1 Features (https://arxiv.org/abs/2503.07230)
Comments:
          Submitted to IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing

- **What's New**: 본 연구에서는 스테 레 마이크로파 레이더(SAR) 데이터를 기반으로 계절적으로 합성된 시공간 이미지와 변환기 기반의 Swin-Unet 아키텍처를 결합하여 지표 유형을 분류하는 새롭고 혁신적인 접근 방식을 제안합니다. 힘을 실어주는 이러한 방법론은 특히 시베리아와 같은 데이터의 시간이 불규칙한 지역에서 성능 향상에 기여하고 있습니다. 전통적인 밀집 시퀀스 대신 계절적 특성 시퀀스를 활용함으로써 제안된 방법론은 다양한 생태 지역에서 높은 정확도를 달성하도록 돕고 있습니다.

- **Technical Details**: 이 연구는 Sentinel-1(S1) SAR 데이터를 활용하여 시공간적 특성을 추출하고 이를 계절 클러스터로 조직하여 지표 유형을 분류합니다. 주요 기술적 장점으로는 전통적인 머신 러닝(Machine Learning) 기법과 달리, Convolutional Neural Networks(CNN)와 Vision Transformers(ViT)의 조합을 통해 데이터를 처리할 수 있다는 점입니다. 이를 통해 시각적 패턴을 효과적으로 학습하고, 시계열 데이터의 텍스처 및 공간적 특징을 증대시키는 능력이 강화되었습니다.

- **Performance Highlights**: 본 연구의 결과는 특히 훈련 데이터가 제한된 지역에서도 높이 전반적인 정확도(Overall Accuracy, O.A.)를 달성하는 데 성공하였습니다. 이 방법론은 아마존, 아프리카, 시베리아 등 다양한 생태 지역에서의 모델 성능 평가를 통해 그 유용성을 입증하였습니다. 전반적으로 제안된 방법론은 SAR 데이터의 고유한 요구 사항을 고려하여 LC 매핑의 정확도를 크게 향상시킬 수 있다는 점에서 의미가 있습니다.



### Synthetic Lung X-ray Generation through Cross-Attention and Affinity Transformation (https://arxiv.org/abs/2503.07209)
- **What's New**: 이 논문에서는 비용 효율적인 대안을 제공하는 합성 데이터 생성을 위한 새로운 방법을 소개합니다. 이는 텍스트-이미지 쌍으로 훈련된 안정적 확산 모델을 기반으로 하는 합성 폐 X-ray 이미지에서 정확한 의미 마스크를 자동으로 생성하는 기술로, 데이터 수집 과정을 혁신적으로 줄입니다. 특히, 텍스트 기반의 교차 주의(cross-attention) 정보를 활용하여 의미 마스크 생성을 수행하는 점이 주목할 만합니다.

- **Technical Details**: 이 방법은 텍스트와 이미지 간의 교차 주의 매핑을 사용하여 텍스트에 의해 유도된 이미지 합성을 의미 마스크 생성을 통해 확장합니다. 주의(attention) 맵을 정밀한 이진 마스크로 변환하는 적응형 임계값(thresholding) 기법을 도입하고, 데이터 다양성을 높이기 위한 검색 기반 프롬프트를 통해 화소 레벨의 데이터 라벨링을 생략할 수 있도록 합니다. 이로 인해 사용자 수작업 없이 무한한 주석 처리된 이미지를 생성할 수 있습니다.

- **Performance Highlights**: 실험 결과, 이 방법을 사용하여 합성된 데이터로 훈련된 분할(segmentation) 모델들이 실제 데이터셋으로 훈련된 모델들과 비교했을 때 동등하거나 때때로 더 우수한 성능을 보였습니다. 이는 이 방법의 효율성과 의료 이미지 분석에 혁신적인 잠재력을 보여줍니다. DiffMask라는 접근법은 의료 이미지 분할의 새로운 패러다임을 제시하여 데이터 수집과 주석 과정에서의 많은 비용을 절감할 수 있는 가능성을 엿봅니다.



### Endo-FASt3r: Endoscopic Foundation model Adaptation for Structure from motion (https://arxiv.org/abs/2503.07204)
- **What's New**: Endo-FASt3r는 로봇 보조 수술(RAS) 장면에서 깊이 및 카메라 자세 추정을 위해 처음으로 고안된 자가 지도 학습(SSL) 기반의 단안 모델이다. 이 프레임워크는 Foundation 모델을 기반으로 하여, 자세 추정(Pose Estimation)과 깊이 추정을 동시에 처리할 수 있도록 설계되었다. 또한, Reloc3r의 변형인 Reloc3rX를 도입하여 SSL의 수렴 문제를 해결하고, DoMoRA라는 새로운 적응 기법을 통해 높은 차수 업데이트를 가능하게 하였다.

- **Technical Details**: Endo-FASt3r는 기존의 Low-Rank Adaptation(LoRA) 기법에 의존하지 않고, 높은 차수의 업데이트를 통해 최적의 성능을 달성하도록 설계되었다. Reloc3rX는 Robust한 수렴을 위해 변경된 Reloc3r 모델로, 앵글 거리 수치에 기반한 지도 학습 방식으로 훈련되었다. DoMoRA는 Low-Rank 및 Full-Rank 업데이트를 결합하여 빠른 수렴을 가능하게 하는 새로운 기술로, 비례-적응과정을 개선했다.

- **Performance Highlights**: SCARED, Hamlyn, StereoMIS 등 3개의 공개 데이터셋에서 Endo-FASt3r는 자세 추정에서 최대 10%, 깊이 추정에서 2%의 성능 향상을 보였다. 이렇게 얻은 성과는 최신 기술(SOTA)과 비교했을 때도 유의미한 개선으로 나타났으며, 다양한 데이터셋에 대한 일반화 가능성을 강조한다. 기존 방법들과 비교하여 Endo-FASt3r는 로봇 보조 수술의 깊이 및 자세 추정에 있어 차별화된 성능을 보여준다.



### Effective and Efficient Masked Image Generation Models (https://arxiv.org/abs/2503.07197)
- **What's New**: 본 연구에서는 서로 다른 목적을 가진 masked image generation 모델과 masked diffusion 모델이 단일 프레임워크 내에서 통합될 수 있음을 관찰하였다. 이를 기반으로 eMIGM이라는 모델을 개발하여, ImageNet 생성에서 뛰어난 성능을 입증하였다. 특히, eMIGM은 유사한 모델 파라미터와 함수 평가수(NFEs)로 기존의 VAR 모델을 능가하였다.

- **Technical Details**: eMIGM 모델의 훈련 및 샘플링 전략을 최적화하기 위해 masked image modeling과 masked diffusion 모델의 장점을 결합하였다. 훈련 과정에서 이미지의 높은 중복성으로 인해 높은 masking 비율의 이점을 발견하였고, 새로운 CFG with Mask 기법을 통해 성능을 더욱 향상시켰다. 또한, 클래스 토큰 대신 마스크 토큰을 사용하여 무조건적인 생성을 강화하였다.

- **Performance Highlights**: eMIGM은 ImageNet의 256x256 해상도에서 VAR과 비교하여 지속적으로 뛰어난 성능을 보였으며, 모델 파라미터와 NFE가 증가함에 따라 성능이 비례적으로 향상되었다. 특히, eMIGM의 대형 모델인 eMIGM-H는 최신 diffusion 모델들과 비슷한 성능을 보여주었고, ImageNet 512x512 해상도에서도 60%의 NFE만으로도 뛰어난 결과를 냈다.



### Multi-Modal 3D Mesh Reconstruction from Images and Tex (https://arxiv.org/abs/2503.07190)
Comments:
          under review

- **What's New**: 이 논문에서는 보지 못한 객체의 6D 포즈 추정을 위한 언어 기반의 소수 샷(몇가지 샷) 3D 재구성 방법을 제안합니다. 기존의 방법들이 대규모 데이터셋과 고비용의 컴퓨팅 자원을 요구하는 반면, 우리는 몇개의 이미지와 언어 정보를 이용하여 3D 메쉬를 재구성하는 혁신적인 접근 방식을 제시합니다. 이를 통해 실시간 응용 프로그램에서의 3D 모델 생성을 보다 효율적이고 실용적으로 수행할 수 있습니다.

- **Technical Details**: 제안된 방법은 입력 이미지 집합과 언어 쿼리를 받아들입니다. GroundingDINO와 Segment Anything Model(SAM)의 조합을 통해 세그먼트 마스크를 생성하고, 이를 사용하여 VGGSfM을 이용해 드문포인트 클라우드를 재구성합니다. 그 후, Gaussian Splatting 기법인 SuGAR를 사용해 메쉬를 생성하며, 최종 단계에서는 여러 아티팩트를 제거하여 쿼리된 객체의 최종 3D 메쉬가 완성됩니다.

- **Performance Highlights**: 우리는 재구성의 정확성 및 기하학적, 텍스처 품질을 평가합니다. 실험은 AMD Ryzen 9 5950X CPU와 NVIDIA RTX 3090 GPU를 갖춘 시스템에서 수행되었으며, 재구성된 3D 기하 구조는 Chamfer Distance와 Intersection over Union을 사용해 평가됩니다. 낮은 Chamfer Distance 값은 정확한 기하학적 정렬을 나타내며, IoU는 재구성된 모델의 부피적 유사성을 측정하여 성능을 확인합니다.



### Evaluation of Alignment-Regularity Characteristics in Deformable Image Registration (https://arxiv.org/abs/2503.07185)
- **What's New**: 이 연구에서는 고해상도 이미지 정합(deformable image registration, DIR)의 평가를 위한 새로운 평가 방식을 제안합니다. 제안된 방식은 alignment-regularity characteristic (ARC) 곡선을 기반으로 하여 정합 정확성과 변형 일관성 간의 트레이드오프를 체계적으로 분석합니다. 또한 HyperNetwork 기반의 접근 방식을 도입하여 ARC 곡선 구조화 과정을 가속화하고 샘플 밀도를 개선합니다.

- **Technical Details**: ARC 곡선은 주어진 정합 알고리즘의 성능을 정합 및 정규성 메트릭으로 측정하여 스펙트럼으로 설명합니다. 본 논문에서는 다양한 정규화 수준에서 정합 결과를 비교하기 위해 여러 DIR 알고리즘을 실험하였습니다. 이 과정에서 다양한 데이터 세트를 활용하여 ARC 평가 방식을 검증하였으며, 여러 네트워크 아키텍처를 평가했습니다.

- **Performance Highlights**: 실험 결과, 기존의 평가 방식에서는 드러나지 않는 여러 가지 통찰을 제공하며, 모델 선택에 도움이 되는 일반적인 권장 사항을 제시합니다. 평가의 정확성과 정규성 간의 균형을 포괄적으로 분석하는 ARC 곡선의 유용성을 확인하였습니다. 본 연구의 모든 코드는 공개되어 있어 연구자들이 활용할 수 있습니다.



### Towards Spatial Transcriptomics-guided Pathological Image Recognition with Batch-Agnostic Encoder (https://arxiv.org/abs/2503.07173)
Comments:
          Accepted to ISBI 2025

- **What's New**: 이 논문은 공간 전사체학(Spatial Transcriptomics, ST)을 이미지 인식에 활용하는 새로운 접근 방식을 제안합니다. 기반 기술은 병리 이미지를 포함한 유전자 발현 데이터를 공간적으로 분석하여, 질병의 하위 유형(classification of subtypes) 식별에 도움을 줄 수 있습니다. 기존에 통상적으로 사용된 데이터의 배치 효과(batch effects)를 해결하면서, 여러 환자로부터 일관된 신호를 추출할 수 있도록 설계된 새로운 대조 학습 프레임워크(contrastive learning framework)는 주목할 만한 발전입니다.

- **Technical Details**: 제안된 프레임워크는 두 가지 단계를 포함합니다. 첫 번째 단계에서는 배치 효과를 제거하기 위해 변이 추정(variational inference) 기법을 도입하여 견고한 유전자 발현 인코더를 얻습니다. 두 번째로, 이 인코더를 사용하여 이미지와 유전자 발현이 쌍으로 구성된 데이터로 대조 학습을 수행합니다. 이를 통해 생물학적인 배경과 무관한 일관된 신호를 확보하여 이미지 인식을 개선하는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과는 제안된 프레임워크가 이전 대조 학습 기반 방법들보다 유의미하게 우수한 성능을 발휘함을 보여주었습니다. 특히, 배치 교정(batch correction)의 중요성과 유전자 발현 데이터를 이미지 인식에 적용하는 데 있어 해결해야 할 주요 도전 과제가 강조되었습니다. 공용 데이터셋을 활용한 평가 결과는 제안된 방법의 유효성을 입증하며, 향후 의료 이미지 분석 분야에서의 활용 가능성을 제시합니다.



### HisTrackMap: Global Vectorized High-Definition Map Construction via History Map Tracking (https://arxiv.org/abs/2503.07168)
- **What's New**: 본 논문에서는 고해상도(HD) 맵 구축을 위한 새로운 전반 추적 프레임워크인 HisTrackMap을 제안합니다. 기존의 방법들은 시간에 따른 일관된 인식 결과를 유지하는 데 어려움을 겪었으나, HisTrackMap은 맵 요소의 역사적 경로를 시간적으로 추적하여 이러한 문제를 해결합니다. 이 접근법은 인스턴스 수준의 역사 맵을 명시적으로 유지하여 향후 내비게이션 인식에 필요한 정보를 제공합니다.

- **Technical Details**: 첫째, 인스턴스 수준의 역사적 레스터화 맵 표현을 설계하여 과거 인식 결과를 명확히 저장합니다. 둘째, 맵-경로 사전 융합(Map-Trajectory Prior Fusion) 모듈을 도입하여 추적된 인스턴스에 대한 역사적 정보를 활용, 시간적 매끄러움과 연속성을 향상시킵니다. 셋째, HD 맵 내의 시간적 기하 구조 품질을 평가하기 위한 글로벌 관점 기준(Global Perspective Metric)을 제안합니다.

- **Performance Highlights**: 제안된 HisTrackMap은 nuScenes 및 Argoverse2 데이터셋에서 기존 최첨단(SOTA) 방법들과 비교하여 뛰어난 성능을 보였습니다. 이는 단일 프레임 및 시간적 메트릭 모두에서 확인되었으며, 역사 맵을 사용한 접근법이 그래도 내재적 변환보다 효과적임을 입증했습니다. 새로운 G-mAP 기반의 평가 메트릭을 통해 맵 구축 결과의 전반적인 일관성을 인정받았습니다.



### Temporal Overlapping Prediction: A Self-supervised Pre-training Method for LiDAR Moving Object Segmentation (https://arxiv.org/abs/2503.07167)
- **What's New**: 본 논문은 자율주행 차량과 같은 시스템을 위한 LiDAR 포인트 클라우드의 Moving Object Segmentation (MOS)에 대한 새로운 접근 방식을 제안합니다. 우리는 기존의 수동 라벨링의 부담을 덜어주기 위해 Temporal Overlapping Prediction (TOP)이라는 자가 지도(pre-training) 방법을 소개합니다. 이 방법은 시간적 중첩된 포인트를 탐구하여 공간-시간적 표현을 학습하고 MOS의 성능을 향상시킵니다.

- **Technical Details**: TOP은 현재 스캔과 인접한 이전 또는 미래 스캔에서 관찰되는 시간적 중첩 포인트의 점유 상태를 예측함으로써 학습을 수행합니다. 비유적으로, 이 접근법은 LiDAR의 물리적 특성을 활용하여 스캔 간 오류를 줄이고, 더 효과적으로 점유 상태를 재구성합니다. 또한, 현 상태 이해를 위해 재구성 용어를 추가하여 모델의 구조적 인식을 강화합니다.

- **Performance Highlights**: 우리는 nuScenes 및 SemanticKITTI 데이터셋에서 실험을 수행하였으며, TOP이 기존의 감독 학습 방법이나 다른 자가 지도(pre-training) 방법들보다 최대 28.77%의 상대적 개선을 보여줍니다. TOP의 전이 가능성과 일반화 능력 또한 여러 다른 작업에 대한 뛰어난 성능을 입증하며, 공개될 코드와 사전 학습된 모델은 연구 커뮤니티에 기여할 것입니다.



### MIRAM: Masked Image Reconstruction Across Multiple Scales for Breast Lesion Risk Prediction (https://arxiv.org/abs/2503.07157)
- **What's New**: 이번 연구에서는 Self-supervised learning (SSL) 접근법을 통해 의료 이미지에서 강력한 특징을 획득하는 새로운 방법을 제안하고 있습니다. 특히, 다중 스케일 이미지 복원(multi-scale image reconstruction)을 기반으로 한 과제가 모델이 더 세부적인 공간적 정보를 포착하도록 돕습니다. 이 새로운 접근법은 기존의 Masked Image Modeling (MIM)보다 더 복잡한 과제를 통해 모델의 성능을 향상시키는 것을 목표로 합니다.

- **Technical Details**: 제안된 방법은 Masked Autoencoder (MAE) 아키텍처를 기반으로 하며, 이미지 복원 및 다중 스케일을 처리하기 위해 여러 개의 디코더를 활용합니다. 각 디코더는 특정 스케일에 대한 이미지를 복원하고, 각각의 복원 손실(reconstruction loss)을 계산하여 전체 손실을 평균하여 최적화합니다. 이는 고해상도 이미지 재구성을 통해 모델이 세부 사항을 학습하도록 안내하여 성능 향상을 도모합니다.

- **Performance Highlights**: 실험 결과, 제안된 MIRAM 접근법은 병리 분류에서 평균 정확도(Average Precision, AP)가 3% 향상되었으며, 수신기 작동 특성 곡선(Area Under the Curve, AUC)에서도 1% 증가했습니다. 또한, 종양 경계 다중 레이블 분류에서는 AP가 4% 증가하고 AUC는 2% 향상되었습니다. 이러한 결과는 SSL 기반의 특징 학습이 의료 이미지 분류 작업에서 강력한 효과를 발휘할 수 있음을 보여줍니다.



### Controllable 3D Outdoor Scene Generation via Scene Graphs (https://arxiv.org/abs/2503.07152)
Comments:
          Project Page: this https URL

- **What's New**: 이번 논문은 사용자 친화적인 제어 방식을 통해 야외 3D 장면 생성을 개선하는 방법을 제안합니다. 씬 그래프(scene graph)를 기반으로 한 새로운 프레임워크를 통해 스파스(sparse)한 씬 그래프를 밀집된 Bird’s Eye View 임베딩 맵(BEV embedding map)으로 변환하여 조건부 확산 모델(conditional diffusion model)을 사용해 3D 장면을 생성합니다. 이러한 접근법은 대규모 야외 장면 생성을 가능하게 하며, 사용자들이 씬 그래프를 쉽게 수정하거나 만들 수 있도록 합니다.

- **Technical Details**: 씬 그래프는 객체, 속성 및 이들 간의 관계를 구조적으로 표현하는 방법입니다. 이 논문에서는 Graph Neural Network(GNN)를 활용해 씬 그래프의 정보를 집계하고, Allocation 모듈을 도입해 공간 위치를 배정하여 Bird’s Eye View 임베딩 맵(BEM)을 생성합니다. 이 BEM은 3D 피라미드 이산 확산 모델(3D Pyramid Discrete Diffusion Model)로 조건화되어 최종 3D 장면을 생성하게 됩니다. 또한, GNN과 확산 모델은 통합을 위해 함께 훈련됩니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안한 방법은 입력된 씬 그래프 정의에 밀접하게 일치하는 고품질 3D 도시 장면을 지속적으로 생성하는 것으로 나타났습니다. 또한, 논문에서 제작한 대규모 데이터셋은 씬 그래프와 3D 장면의 쌍으로 구성되어 있으며, 이는 모델 훈련에 중요한 역할을 합니다. 최종적으로, 사용자 친화적인 시스템을 통해 사용자들이 쉽게 자신만의 씬 그래프를 생성하고 수정할 수 있어 3D 장면 생성의 유연성이 개선됩니다.



### A Light Perspective for 3D Object Detection (https://arxiv.org/abs/2503.07133)
- **What's New**: 본 논문은 카메라와 LIDAR 데이터를 결합하여 3D 객체 탐지의 정확성을 높이는 새로운 접근 방식을 제시합니다. 특히, 기존의 무거운 백본 대신 최신 Deep Learning 기술을 활용하는 경량 모델 NextBEV를 소개하여, 이전 모델에 비해 성능을 유지하면서도 효율성을 강화했습니다. KITTI 3D Monocular 벤치마크에서 NextBEV는 약 2.39%의 정확성 향상을 이뤄냈습니다.

- **Technical Details**: NextBEV는 카메라 이미지에서 직접 Bird-Eye-View(BEV) 피쳐 텐서로 변환하는 종단 간(end-to-end) 접근 방식을 제공합니다. 또한, LIDAR 백본을 재분석하여 아키텍쳐 크기와 계산 요구량, 추론 시간을 줄이면서도 정확성을 유지하도록 개량했습니다. 전체적으로 이러한 수정된 경량 경로를 통합하여, 실시간 예측의 요구를 충족할 수 있는 성능을 갖춘 센서 융합 모델을 완성했습니다.

- **Performance Highlights**: NextBEV는 MobileNetV2의 절반도 안 되는 파라미터로 기존 모델들을 초월하는 높은 정확도를 달성했습니다. 또한, LIDAR 경로의 개선 덕분에 추론 시간을 10 ms까지 감소시켜 경량화된 접근 방식을 효과적으로 구현할 수 있었습니다. 통합된 센서 융합 모델은 단일 센서 기반 접근 방식에 비해 20% 향상된 F1-score를 기록하며, 잘못된 긍정(False Positives)과 잘못된 부정(False Negatives)의 발생도 최소화했습니다.



### Learning A Zero-shot Occupancy Network from Vision Foundation Models via Self-supervised Adaptation (https://arxiv.org/abs/2503.07125)
Comments:
          preprint

- **What's New**: 이 연구는 2D monocular 이미지에서 3D 세계를 추정하기 위한 새로운 접근법을 제시합니다. 특히, 3D 감독(supervision)을 이미지 레벨의 원시 정보(semantics, geometry)로 분리하여 2D Vision Foundation Models (VFMs)와 연결하는 방법을 강조합니다. 이러한 방식은 3D 주석(annotation)을 쉽게 획득할 수 있도록 하여, 필요한 경우 비지도 방식으로도 3D 정보를 활용할 수 있게 합니다.

- **Technical Details**: 연구에서는 2D VFMs로부터 얻은 상대 깊이(relative depth)를 최적화하여 미터 규모의 깊이(metric depth)로 변환하는 두 단계의 최적화 전략을 설명합니다. 초기 단계에서는 전체 장면에 적용되는 후보 규모(candidates of scales)를 선택하고, 이후 화소(pixel) 단위로 규모와 오프셋을 조정합니다. 이를 통해 3D 점유 예측(occupancy prediction)을 할 때, 기존 방법들에 비해 향상된 성능을 달성합니다.

- **Performance Highlights**: 제안된 프레임워크는 nuScenes 및 SemanticKITTI 데이터셋에서 3D 점유 예측에서 탁월한 성능을 보여주었습니다. 예를 들어, 이 연구의 방법은 nuScenes에서 voxel occupancy 예측에 대해 현재의 최첨단 방법들을 3.34% mIoU로 초월했습니다. 이러한 성과는 2D VFMs를 효과적으로 활용하여 3D 작업에 대한 새로운 가능성을 제시합니다.



### Exposure Bias Reduction for Enhancing Diffusion Transformer Feature Caching (https://arxiv.org/abs/2503.07120)
- **What's New**: Diffusion Transformer (DiT)는 뛰어난 생성 능력을 갖고 있지만, 높은 계산 복잡도로 인해 많은 도전 과제를 안고 있습니다. 본 논문에서는 캐싱이 중간 과정 생성에 미치는 영향을 분석하고, Exposure Bias 문제를 해결하기 위한 새로운 캐시 전략인 EB-Cache를 제안합니다. EB-Cache는 비노출 편향(diffusion) 프로세스를 정렬하여 성능을 최적화하고 동시에 가속화하는 혁신적인 접근 방식을 제공합니다.

- **Technical Details**: 이 연구는 캐싱이 노이즈 제거 절차를 강화하여 노드에서 발생하는 왜곡된 편향 문제를 심화시킨다는 점을 강조합니다. 새롭게 제안된 EB-Cache 전략은 단계별 및 구조별 설계를 통해 캐싱 전략을 일관성 있게 조정하여 노출 편향을 조절하고 있습니다. STEP-WISE 캐시 테이블을 활용하고, 자기 주의(self-attention) 및 피드포워드(feedforward) 네트워크에 대해 독립적으로 캐싱 전략을 채택하는 방식으로 구성됩니다.

- **Performance Highlights**: EB-Cache는 50-step 생성 과정에서 1.49배의 가속화와 3.69에서 0.63 FID 낮춤을 기록하며 기존 가속화 방법들을 초월한 성과를 보였습니다. 이를 통해 다수의 시나리오에서 실험 결과의 효과성과 신뢰성을 입증하였고, 기존 캐싱 방법에 플러그인하여 성능을 더욱 향상시킬 수 있음을 강하게 시사합니다.



### YOLOMG: Vision-based Drone-to-Drone Detection with Appearance and Pixel-Level Motion Fusion (https://arxiv.org/abs/2503.07115)
Comments:
          9 pages, 8 figures

- **What's New**: 최근 드론 간 탐지에서 비전 기반 접근법의 중요성이 부각되면서, 본 연구에서는 복잡한 환경에서 소형 드론을 정확하게 식별할 수 있는 새로운 end-to-end 프레임워크를 제안합니다. 제안된 방법은 움직임 차이 맵(motion difference map)을 생성하여 소형 드론의 움직임 특성을 포착하고, 이를 RGB 이미지와 결합하여 적응적인 특징 학습이 가능하도록 합니다. 특히, ARD100이라는 새로운 데이터셋을 공개하여 다양한 도전적인 조건에서의 소형 드론 탐지를 위한 연구를 강화합니다.

- **Technical Details**: 본 논문에서는 소형 드론 탐지에서 효과적인 알고리즘 개발을 위한 YOLOMG라는 새로운 모션 기반 객체 탐지 기법을 제안합니다. 이를 위해 픽셀 수준의 모션 특징을 추출하는 모션 특징 향상 모듈(motion feature enhancement module)을 도입하고, 움직임 차이 맵을 RGB 이미지와 결합하여 드론의 특징을 적응적으로 학습합니다. 마지막으로, YOLOv5 기반의 향상된 경량 백본과 탐지 네트워크를 통해 탐지 결과를 생성하여 성능을 극대화합니다.

- **Performance Highlights**: ARD100 데이터셋을 통해 검증된 결과, 제안된 방법은 도전적인 조건에서 뛰어난 성능을 보여주며 평균 정밀도에서 최첨단 알고리즘보다 22% 향상된 성능을 기록했습니다. 또한, NPS-Drones 데이터셋에서도 최신 기술을 능가하는 성능을 달성하였으며, 일반 객체 탐지기와 대비하여 높은 효율성과 강력한 일반화 성능을 유지합니다.



### SimROD: A Simple Baseline for Raw Object Detection with Global and Local Enhancements (https://arxiv.org/abs/2503.07101)
- **What's New**: 이번 논문에서는 SimROD라는 경량화된 RAW 객체 탐지 접근법을 제안합니다. RAW 데이터는 ISP 처리를 거치기 전에 더 풍부한 센서 정보를 보존하여 객체 탐지 정확도를 향상시킬 수 있는 장점이 있습니다. 기존의 복잡한 방법들과 달리 SimROD는 Global Gamma Enhancement (GGE) 모듈을 통해 효율성을 유지하며 성능을 개선하는 것을 목표로 합니다.

- **Technical Details**: SimROD는 두 가지 핵심 인사이트에 기반하여 설계되었습니다. 첫째, 조정된 글로벌 변환이 미세한 작업에 중요하다는 점과, 둘째, RGGB Bayer 패턴에서 녹색 채널의 우수한 정보성을 활용하여 모듈을 구성합니다. GGE 모듈은 네 개의 학습 가능한 매개변수를 사용하여 효과적인 글로벌 감마 변환을 적용하고, Green-Guided Local Enhancement (GGLE) 모듈은 녹색 채널을 이용해 지역 세부 사항을 보강합니다.

- **Performance Highlights**: SimROD는 ROD, LOD, Pascal-Raw와 같은 여러 RAW 객체 탐지 벤치마크에서 기존의 방법들보다 우수한 성능을 보였습니다. 예를 들어, Pascal-Raw 벤치마크에서는 RAW-Adapter와 같은 방법보다 일관된 성능 향상을 달성했습니다. 또한, DIAP에 대한 강력한 기준선도 설정되었으며, 이는 ROD 데이터셋에서 mAP 성능을 24.0%에서 30.7%로 향상시켰습니다.



### OmniSAM: Omnidirectional Segment Anything Model for UDA in Panoramic Semantic Segmentation (https://arxiv.org/abs/2503.07098)
- **What's New**: OmniSAM 프레임워크는 SAM2를 사용하여 파노라마 세그멘테이션 작업에 적용하는 최초의 시도입니다. 이 프레임워크는 파노라마 이미지를 패치로 분할하고, 각 패치를 비디오 세그멘테이션 작업과 유사하게 처리합니다. SAM2의 메모리 메커니즘을 활용해 패치 간 상관관계를 추출함으로써, 넓은 시야각(FoV)의 변화를 효과적으로 극복합니다.

- **Technical Details**: OmniSAM은 Low-Rank Adaptation (LoRA) 방식으로 미세 조정된 SAM2 이미지 인코더를 기반으로 하며, 맞춤형 세그멘테이션 디코더를 포함합니다. FoV 기반 프로토타입 적응(FPA) 모듈과 동적 가짜 레이블 업데이트 전략을 통해 서로 다른 도메인 간 특성 정합을 강화합니다. 이 형태로 처리된 이미지를 소스 모델 학습과 타겟 도메인 적응에 사용하여, 패치의 프로토타입을 추출하고 이를 기반으로 정합성을 부여합니다.

- **Performance Highlights**: OmniSAM은 최신 기술(SOTA) 방법들에 비해 79.06%에서 10.22% 향상된 성능을 기록하였고, CS13-to-DP13와 같은 시나리오에서 62.46%에 6.58% 향상된 결과를 보였습니다. 실험 결과는 적은 학습 파라미터 수(26MB 미만)에서 높은 성과를 달성함을 보여줍니다. 이러한 결과는 파노라마 세멘틱 분석 작업에서 OmniSAM의 잠재력을 잘 나타냅니다.



### FaceID-6M: A Large-Scale, Open-Source FaceID Customization Datas (https://arxiv.org/abs/2503.07091)
- **What's New**: 이번 논문에서는 최초의 대규모 오픈 소스 FaceID 데이터셋인 FaceID-6M을 수집하고 공개하였습니다. 이 데이터셋은 6백만 개의 고품질 텍스트-이미지 쌍으로 구성되어 있으며, LAION-5B에서 필터링 과정을 거쳐 제작되었습니다. 논문은 이 데이터셋이 FaceID 맞춤화 모델의 훈련에 적합하도록 세심하게 품질 관리 과정을 통해 구축되었다고 강조합니다.

- **Technical Details**: FaceID-6M 데이터셋은 이미지 필터링 및 텍스트 필터링 과정을 포함하여 구성됩니다. 이미지 필터링에서는 인간의 얼굴이 없는 이미지나 해상도가 낮은 이미지를 제거하며, 텍스트 필터링에서는 인물, 국적 및 직업과 관련된 용어가 포함된 설명을 남깁니다. 이러한 과정을 통해 FaceID-6M은 강력한 FaceID 맞춤화 모델 훈련을 최적화된 데이터셋을 제공합니다.

- **Performance Highlights**: FaceID-6M 데이터셋으로 훈련된 모델은 기존 산업 모델에 비해 경쟁력 있는 성능을 보여줍니다. 실험 결과, FaceID-6M으로 훈련된 InstantID 모델이 COCO-2017 테스트 세트에서 0.63의 FaceID 충실도 점수를 달성하여 이전 모델인 InstantID의 0.59 점수를 초과하였습니다. 이러한 결과는 FaceID-6M이 FaceID 맞춤화 커뮤니티에 중요한 기여를 할 수 있음을 시사합니다.



### On the Generalization of Representation Uncertainty in Earth Observation (https://arxiv.org/abs/2503.07082)
Comments:
          18 pages

- **What's New**: 최근 컴퓨터 비전 분야에서 사전 훈련된 표현 불확실성(pretrained representation uncertainty) 개념이 발전하여 제로샷 불확실성 추정(zero-shot uncertainty estimation)이 가능해졌다. 이는 신뢰성이 중요한 지구 관측(Earth Observation, EO) 분야에 크게 기여할 수 있으나, EO 데이터의 복잡성은 불확실성 인식 방법의 도전에 직면하게 한다. 이 연구는 EO의 고유한 의미적 특성을 고려하여 표현 불확실성의 일반화를 조사하였다.

- **Technical Details**: 연구자들은 큰 EO 데이터셋을 활용하여 불확실성을 사전 훈련(pretrain)하고, 다중 레이블 분류(multi-label classification) 및 세분화(segmentation) EO 작업에서 제로샷 성능을 평가하는 프레임워크를 제안하였다. 이 방법은 대규모 사전 훈련된 모델의 표현 공간에서 직접 불확실성을 추출하여, 큰 확장성을 가진 제로샷 불확실성 추정을 가능하게 한다. 또한, EO의 의미적 요인(Semantic Factors, SFs)을 정의하고 이들이 표현 불확실성의 일반화에 미치는 영향을 평가하였다.

- **Performance Highlights**: 연구 결과, 자연 이미지에서 사전 훈련된 불확실성과 달리 EO 사전 훈련은 보지 못한 EO 도메인, 지리적 위치 및 목표 세분성에 대해 강력한 일반화를 나타냈다. GSD(ground sampling distance)에 대한 민감성을 유지하면서도, EO 사전 훈련 불확실성이 하류 작업(downstream tasks)에서의 작업 특화 불확실성과 잘 맞아 떨어짐을 보여주었다. 이 연구는 EO 분야에서의 표현 불확실성의 강점과 제한을 논의하며, 향후 연구의 길을 제시한다.



### NFIG: Autoregressive Image Generation with Next-Frequency Prediction (https://arxiv.org/abs/2503.07076)
Comments:
          10 pages, 7 figures, 2 tables

- **What's New**: 이번 논문에서는 이미지 생성을 위한 새로운 프레임워크인 Next-Frequency Image Generation(NFIG)을 제안합니다. NFIG는 이미지 생성 과정을 여러 주파수 기반 단계로 분해하여 처리하며, 이를 통해 전반적인 구조를 설정하는 저주파 구성 요소를 생성한 후 고주파 세부 사항을 점진적으로 추가합니다. 이 접근 방식은 이미지 구성 요소 간의 인과 관계를 더 잘 포착하여 이미지 품질을 향상시키고, 계산 비용을 크게 줄입니다.

- **Technical Details**: NFIG의 핵심은 주파수 분석을 바탕으로 한 Frequency-guided Residual-quantized VAE(FR-VAE)입니다. 이는 이미지에서의 저주파 및 고주파 요소를 분리하여 저주파 구성 요소로 전반적인 구조를 수용하고 고주파 구성 요소로 세부 사항을 유지합니다. FR-VAE는 자원 효율성을 극대화하며, 전체 주파수 스펙트럼에서 정보를 효율적으로 표현하는 방법론으로 설계되었습니다.

- **Performance Highlights**: NFIG는 뛰어난 이미지 생성 품질을 보여주며, ImageNet-256 벤치마크에서 FID 2.81을 기록하여 최신 성능을 달성했습니다. 또한, 1.25배의 속도 향상을 보여주는 등 개선된 효율성을 입증했습니다. 이 논문은 이미지 생성 모델에 대한 새로운 통찰력을 제공하며, 향후 연구의 방향을 제시하고자 합니다.



### XR-VLM: Cross-Relationship Modeling with Multi-part Prompts and Visual Features for Fine-Grained Recognition (https://arxiv.org/abs/2503.07075)
- **What's New**: 본 연구에서는 XR-VLM이라는 새로운 메커니즘을 제안하여 유사한 하위 카테고리를 구별할 수 있는 미세한 차이를 찾아내는데 중점을 두고 있습니다. 현재의 기존 방법들이 시각적 특성과 클래스 프롬프트 간의 상호작용을 충분히 반영하지 못하고 있음을 지적하며, 이를 극복하기 위한 접근법을 소개합니다. 우리의 접근법은 각 클래스의 모든 정보와의 관계를 통합하여 예측을 수행하는 모델을 새롭게 구축하였습니다.

- **Technical Details**: XR-VLM은 멀티파트 비주얼 특징 추출 모듈과 멀티파트 프롬프트 학습 모듈을 도입하여 다중 특성 시나리오에서 우수한 성능을 발휘하도록 설계되었습니다. 이 방법은 시각적 특징과 모든 클래스 프롬프트 특성을 결합하여 클래스 간의 상호작용을 추적하고 정보가 풍부한 예측을 가능하게 합니다. 또한, 비주얼과 언어 정보를 동시에 처리하여 미세한 차이를 정교하게 캡처할 수 있도록 돕습니다.

- **Performance Highlights**: 우리의 방법은 다양한 미세한 데이터 세트에 대한 광범위한 실험을 수행하였으며, 현재 고급(VLM) 적응 방법들에 비해 상당한 개선 효과를 보였습니다. 실험 결과, 여러 데이터 세트에서 최상의 성능을 달성하였고, 클래스 관계 모델링이 전통적인 정렬 기반 패턴에 비해 더 나은 성능을 이끌어냄을 입증하였습니다. 이는 최적의 성능을 위해 시각적 및 언어적 특징을 더욱 잘 활용할 수 있음을 보여줍니다.



### Boosting the Generalization and Reasoning of Vision Language Models with Curriculum Reinforcement Learning (https://arxiv.org/abs/2503.07065)
- **What's New**: 본 연구에서는 소규모 비전-언어 모델(VLM) 교육을 위한 새로운 포스트 트레이닝 패러다임인 Curriculum Reinforcement Finetuning (Curr-ReFT)을 제안합니다. 기존의 대규모 언어 모델에 비해 소규모 모델이 겪는 OOD(Out-Of-Domain) 일반화 및 추론 능력의 한계를 극복하기 위해 재설계된 Curr-ReFT는 단계적 접근 방식을 적용합니다.

- **Technical Details**: Curr-ReFT는 두 단계로 구성되며, 첫 번째는 난이도에 맞춘 보상을 통해 모델 능력을 점진적으로 발전시키는 Curriculum Reinforcement Learning입니다. 두 번째 단계인 Rejected Sampling 기반 Self-improvement는 고품질의 멀티모달 및 텍스트 예시로부터 선택적 학습을 통해 모델의 기초 능력을 유지합니다.

- **Performance Highlights**: 실험 결과, Curr-ReFT로 학습된 모델은 다양한 비전 과제에서 최신 성능을 달성하며, 심지어 3B 모델이 32B 모델의 성능을 초과하는 경우도 관찰되었습니다. 이러한 결과는 Curr-ReFT의 효율적인 훈련 패러다임이 소규모와 대규모 모델 간의 격차를 효과적으로 해소할 수 있음을 보여줍니다.



### Breaking the Limits of Quantization-Aware Defenses: QADT-R for Robustness Against Patch-Based Adversarial Attacks in QNNs (https://arxiv.org/abs/2503.07058)
- **What's New**: 본 연구는 Quantized Neural Networks (QNNs)이 저비용 고속 연산 환경에서 모델 크기와 계산 비용을 줄이는 데 효과적이라는 점을 강조합니다. 하지만 2비트와 같은 극단적인 비트 너비 하에서도 패치 기반 적대적 공격에 대해 높은 전이 가능성을 보여주었고, 이는 양자화가 적대적 위협을 완화한다는 일반적인 가정에 도전합니다. QADT-R이라는 새로운 방어 전략을 제안하여 패치 기반 공격에 대한 저항력을 높였습니다.

- **Technical Details**: QADT-R은 Adaptive Quantization-Aware Patch Generation (A-QAPA), Dynamic Bit-Width Training (DBWT), Gradient-Inconsistent Regularization (GIR)을 포함하고 있습니다. A-QAPA는 적대적 패치를 양자화된 모델 내에서 생성하고, DBWT는 특정 양자화 설정에 과적합하지 않도록 비트 너비를 주기적으로 변경합니다. 마지막으로 GIR은 적대적 최적화를 방해하기 위해 그래디언트의 무작위 방해를 주입합니다.

- **Performance Highlights**: 연구 결과, QADT-R은 CIFAR-10과 ImageNet에서 기존의 방어 방법들 (PBAT, DWQ)과 비교하여 최대 25%의 공격 성공률(ASR) 감소를 보여주며, 보지 못한 공격 구성에 대해서는 40% 이상 감소했습니다. 이 연구는 각각의 구성 요소가 가지는 중요성을 입증하기 위한 조건부 실험을 수행하였고, QADT-R이 QNNs의 강력한 방어 메커니즘임을 확립했습니다.



### TIDE : Temporal-Aware Sparse Autoencoders for Interpretable Diffusion Transformers in Image Generation (https://arxiv.org/abs/2503.07050)
- **What's New**: 이번 논문에서는 기존 U-Net 기반의 diffusion 모델과 비교하여 Diffusion Transformers (DiTs)의 새로운 해석 가능성 탐구를 위한 TIDE 프레임워크(Tide Framework)를 제안합니다. TIDE는 Sparse Autoencoders (SAEs)를 사용하여 활성화 레이어 내에서의 시간적 재구성을 개선하며, diffusion 모델이 생성적 사전 학습 과정에서 계층적 기능을 스스로 학습한다는 것을 밝혀냅니다. 이 연구는 DiT의 내부 작업을 이해하는 데 필수적인 기여를 하며, 다양한 응용 분야에서 향상된 제어 가능성을 제공합니다.

- **Technical Details**: TIDE는 시간적 변화를 포착하고 denoising 과정 전반에 걸쳐 희소하고 해석 가능한 기능을 추출하기 위해 설계된 프레임워크입니다. 이 연구에서는 프로그레시브 스파시티 스케줄링(progressive sparsity scheduling) 및 랜덤 샘플링 증강(random sampling augmentation)과 같은 특수 교육 전략을 도입하여, 높은 충실도를 유지하면서 학습 속성을 최적화합니다. TIDE는 또한 multi-level features를 학습하여 다양한 다운스트림 작업에 적합하게 설계되었습니다.

- **Performance Highlights**: TIDE는 평균 제곱 오차(MSE) 1e-3 및 코사인 유사도 0.97을 기록하며, state-of-the-art 재구성 성능을 자랑합니다. TIDE는 사용자 친화적이고 효율적인 성능을 제공하며, 다수의 실험을 통해 고해상도 이미지에서의 활성화 역학을 효과적으로 포착하는 것이 입증되었습니다. 이미지 편집, 스타일 전송 등 다양한 분야에서의 응용 가능성을 통해 생성 시스템의 신뢰성과 제어 가능성을 높이는 데 기여하고 있습니다.



### Recovering Partially Corrupted Major Objects through Tri-modality Based Image Completion (https://arxiv.org/abs/2503.07047)
Comments:
          17 pages, 6 page supplementary

- **What's New**: 이번 논문에서는 이미지 보완(Image completion) 작업에 있어 텍스트 프롬프트(Text prompts)와 함께 사용되는 새로운 시각적 도구로 캐주얼 스케치(Casual sketch)를 도입합니다. 이는 기존의 텍스트 프롬프트가 세부 구조 정보를 회복하는 데 한계가 있음을 인식하고, 시각적 단서를 통해 생성 모델이 배경과 자연스럽게 통합된 객체 구조를 생성하도록 돕습니다. 저자들은 Visual Sketch Self-Aware (VSSA) 모델을 제안하며, 이는 확산 과정의 각 반복 단계에 스케치를 통합하여 특정 효과를 제공합니다.

- **Technical Details**: VSSA 모듈은 캐주얼 스케치를 확산 과정의 각 단계에 통합하여, 손상된 이미지의 다중 규모 문맥과 스케치에서 유래된 특징을 혼합합니다. 이 방식은 텍스트 프롬프트 가이드를 활용하여 복원된 객체와 원래 지역 간에 의미와 구조적 일관성을 유지하는 데 도움을 줍니다. 또한, CUB-sketch 및 MSCOCO-sketch라는 두 개의 신규 데이터셋을 구축하여 이미지, 스케치, 텍스트의 삼중 모달 데이터로 구성된 이미지 복원의 연구에 기여하고 있습니다.

- **Performance Highlights**: 저자들은 제안한 방법이 여러 최신 기술과 비교하여 정성적 및 정량적 성과에서 우수함을 입증했다고 강조합니다. 실험 결과, 스케치 기반 가이드를 통한 부분적으로 손상된 객체의 복원에서 더욱 정밀하고 자연스러운 결과를 나타냈습니다. 이는 복원된 객체 부분이 주변의 손상되지 않은 이미지 영역과 일관되게 연결될 수 있도록 하기 위한 개선된 접근법을 보여줍니다.



### MambaFlow: A Mamba-Centric Architecture for End-to-End Optical Flow Estimation (https://arxiv.org/abs/2503.07046)
- **What's New**: 이번 연구에서 제안하는 MambaFlow 프레임워크는 Optical Flow(옵티컬 플로우) 추정의 새로운 접근 방식을 제공합니다. 특히, Mamba 아키텍처를 활용하여 지역적인 상관관계를 캡처하면서도 글로벌 정보를 유지하는 데 중점을 두고 있습니다. 이는 기존의 Transformer 기반 방법들이 가진 계산적 복잡성을 효과적으로 개선하여 시간을 절약할 수 있는 혁신적인 방법으로 주목받고 있습니다.

- **Technical Details**: MambaFlow는 두 가지 주요 구성 요소로 이루어져 있으며, 각각 Feature Enhancement Mamba (FEM) 모듈과 Flow Propagation Mamba (FPM) 모듈입니다. FEM 모듈은 Self-Mamba 및 Cross-Mamba 메커니즘을 사용하여 추출된 특징 간의 관계를 향상시킵니다. FPM 모듈은 특징 유사성을 활용하여 흐름 정보를 효과적으로 전파하며, 특히 가려진 영역에서의 문제를 해결하는 데 디자인되었습니다.

- **Performance Highlights**: MambaFlow는 Sintel 벤치마크에서 1.60의 EPE(End-Point Error)를 기록하며 GMFlow의 1.74를 초과하는 성과를 보였습니다. 또한, MambaFlow는 0.113초의 실행 시간으로 GMFlow보다 18% 빠르며, MS-RAFT+에 비해서는 10배 더 빠른 성능을 자랑합니다. 소스 코드는 논문이 수락될 경우 공개될 예정입니다.



### Find your Needle: Small Object Image Retrieval via Multi-Object Attention Optimization (https://arxiv.org/abs/2503.07038)
- **What's New**: 이 논문은 Small Object Image Retrieval (SoIR)이라는 도전 과제를 다루고 있습니다. SoIR은 복잡한 장면에서 특정 작은 객체가 포함된 이미지를 검색하는 것을 목표로 하며, 기존 방법의 한계를 분석하고 새로운 벤치마크를 도입하여 SoIR 평가를 지원합니다. 새롭게 제안된 Multi-object Attention Optimization (MaO) 프레임워크는 여러 객체를 하나의 이미지 디스크립터로 통합하는 혁신적인 방식을 제공합니다.

- **Technical Details**: MaO는 개별 객체를 인지하기 위해 open-vocabulary detector (OVD)를 사용하여 이미지를 분해하고, 그 후 각 객체의 특성을 주의 기반(feature-based)으로 추출하여 통합하는 과정을 포함합니다. 이 방법은 이미지의 배경 요소를 필터링하면서, 각 객체가 균등하게 표현될 수 있도록 설계되었습니다. MaO의 포스트 트레이닝 단계에서는 attention maps와의 정렬 최적화를 통해 이미지 디스크립터의 검색 정확도를 제고하는 과정이 포함됩니다.

- **Performance Highlights**: 우리의 방법은 기존의 SoIR 방법에 비해 대폭 향상된 성능을 보여주며, 특히 제로샷(zero-shot) 및 경량(multi-object fine-tuning) 환경에서의 두드러진 개선을 이끌어 냈습니다. 이러한 결과는 복잡한 장면에서 작은 객체를 포함한 이미지를 효과적으로 검색할 수 있는 능력을 입증합니다. 최종적으로, MaO는 SoIR의 향상을 목표로 한 미래 연구의 기초를 마련한다고 기대합니다.



### Zero-Shot Hashing Based on Reconstruction With Part Alignmen (https://arxiv.org/abs/2503.07037)
- **What's New**: 본 논문에서는 이미지 부분 정렬(Part Alignment)을 기반으로 한 새로운 제로샷 해싱(zero-shot hashing) 알고리즘인 RAZH를 제안합니다. 기존의 제로샷 해싱 알고리즘은 전반적인 이미지에서 추출된 특징을 활용하지만, RAZH는 이미지의 개별 부분에 대한 특성을 매칭하는 데 중점을 둡니다. 이를 통해 특성과 이미지 부분 간의 정밀한 정렬을 달성하여 정확도를 높이고, 특정 특성과 해당 이미지 부위 간의 관계를 포착하는 방법을 제시합니다.

- **Technical Details**: RAZH는 먼저 군집화(Clustering) 알고리즘을 사용하여 유사한 패치를 그룹화하여 각 부분에 맞는 특성을 정렬합니다. 그 후, 이미지 부분을 해당 특성 벡터로 대체하고, 각 부분과 가장 가까운 특성과 점진적으로 정렬하는 재구성 전략(Reconstruction Strategy)을 사용합니다. 이러한 접근 방식은 다양한 크기의 이미지 부분과 특성 간의 정렬 문제를 해결하기 위한 새로운 틀을 제공합니다.

- **Performance Highlights**: 광범위한 실험 결과에서 RAZH 방법은 여러 최신 기술(State-of-the-art) 방법들보다 우수한 성능을 보였습니다. 이 연구는 제로샷 해싱 문제를 해결하는 데 있어 이미지 부분과 속성 간의 정밀한 정렬이 핵심적이라는 점을 강조하며, 이를 통해 새로운 클래스에 대한 해시 코드를 효과적으로 생성할 수 있음을 보여줍니다.



### Universal Incremental Learning: Mitigating Confusion from Inter- and Intra-task Distribution Randomness (https://arxiv.org/abs/2503.07035)
Comments:
          10 pages, 4 figures, 4 tables

- **What's New**: 본 논문에서는 새롭게 제안하는 범용 증분 학습(Universal Incremental Learning, UIL) 시나리오를 탐구합니다. 기존의 증분 학습 방법들이 일정한 클래스나 도메인의 증가만을 가정하는 것에 비해, UIL은 예기치 않은 클래스와 도메인의 무작위적 증가를 처리합니다. 이러한 불확실성을 통해 모델이 모든 작업 분포에서 지식을 확실하게 학습하는 것을 방지하며 더 일반적이고 현실적인 학습 시나리오를 성립합니다.

- **Technical Details**: UIL에서는 각 작업이 무작위적인 증가 유형을 가지고 있으며, 이는 이전 도메인에 새로운 클래스가 포함되거나, 새로운 도메인에 이전 클래스가 포함되거나, 완전히 새로운 클래스와 도메인이 모두 포함될 수 있습니다. 이를 해결하기 위해 MiCo라는 프레임워크를 제안하고, 다중 목표 학습(multi-objective learning) 방식과 방향 및 크기 조정 모듈(decoupled recalibration modules)을 도입하여 작업 간 및 작업 내 배포의 혼란(confusion)을 완화합니다.

- **Performance Highlights**: 실험 결과, 제안한 MiCo 방법이 세 가지 기준에 대한 연구에서 기존의 최첨단 방법보다 우수한 성능을 보였음을 입증했습니다. 특히, UIL 시나리오와 기존의 다목적 증분 학습(Versatile Incremental Learning, VIL) 시나리오에서도 눈에 띄는 개선된 결과를 나타냈습니다. 이는 UIL이 실제 환경에서의 다양한 증분 유형을 잘 처리할 수 있는 가능성을 보여줍니다.



### Learning a Unified Degradation-aware Representation Model for Multi-modal Image Fusion (https://arxiv.org/abs/2503.07033)
- **What's New**: 이번 논문에서는 복합적인 장면을 다루기 위해 이미지의 퇴화를 인식하고 뛰어난 품질의 융합 이미지를 생성하는 새로운 모델인 LURE를 제안합니다. LURE는 다양한 모드에서 발생하는 퇴화를 연관시켜, 기존의 데이터 문제를 해결하고 강력한 복원 훈련을 가능하게 합니다. 이 접근 방식은 데이터 간의 관계를 극복하고, 고품질의 실제 복원 데이터셋을 활용 할 수 있다는 점에서 혁신적입니다.

- **Technical Details**: LURE 모델은 데이터 수준에서 다중 모드 및 다중 품질 데이터를 분리하고 통합된 잠재 피처 공간(ULFS)에서 재결합하여 퇴화를 인식하는 새로운 손실 함수를 제안합니다. 작고 공간적 편향을 줄이기 위해, Text-Guided Attention(TGA) 기능을 활용하여 텍스트-이미지 상호작용을 강화하고 상세한 시각적 데이터를 보존합니다. 이 과정에서 inner residual 구조를 도입하여 효율적인 잠재 표현 학습을 추가합니다.

- **Performance Highlights**: 실험 결과, LURE는 일반적인 융합, 퇴화 인식 융합, 그리고 다운스트림 작업에 대해 최신의 SOTA 방법들을 능가하는 성능을 보여주었습니다. 이 모델은 데이터를 효율적으로 사용하여 퇴화 인식 모형을 신속하게 학습시키고, 외부 모형들과의 유연한 비교를 가능하게 합니다. 또한 코드는 공개되어 연구자들이 쉽게 활용할 수 있게 됩니다.



### Availability-aware Sensor Fusion via Unified Canonical Space for 4D Radar, LiDAR, and Camera (https://arxiv.org/abs/2503.07029)
Comments:
          Arxiv preprint

- **What's New**: 이번 연구에서는 자율주행(AD) 시스템을 위한 새로운 방법인 availability-aware sensor fusion (ASF)을 제안합니다. ASF는 각 센서의 특징을 통합하여 일관성을 보장하는 unified canonical projection (UCP)을 사용하며, 센서 고장 및 열화에 저항할 수 있는 cross-attention across sensors along patches (CASAP)를 도입하였습니다. 이 방법은 다양한 기상 조건과 센서 열화 상황에서도 기존의 최첨단 융합 방법들보다 뛰어난 성능을 발휘합니다.

- **Technical Details**: ASF는 깊이 연결된 융합(DCF) 및 센서별 크로스 어텐션 융합(SCF)의 한계를 동시에 해결합니다. UCP는 모든 센서의 특징을 통일된 공간에 투영하여 일관성을 없애며, CASAP-PN은 센서의 가용성을 고려하여 크로스 어텐션을 수행합니다. 이를 통해 ASF는 복잡한 위치 임베딩을 제거하고 계산 효율성을 향상시킵니다.

- **Performance Highlights**: K-Radar 데이터셋에서 ASF는 기존 SOTA(상태-최고) 융합 방법보다 9.7% 향상된 AP BEV(87.2%)와 20.1% 향상된 AP 3D(73.6%)를 기록했습니다. 이러한 성능은 센서 고장이나 저하와 같은 극단적인 상황에서도 유지됩니다. ASF는 낮은 계산 비용으로 높은 성능을 달성하여 자율주행의 신뢰성과 강인성을 향상시킵니다.



### EasyControl: Adding Efficient and Flexible Control for Diffusion Transformer (https://arxiv.org/abs/2503.07027)
- **What's New**: 최근 Unet 기반 확산 모델의 발전은 ControlNet과 IP-Adapter와 같은 효과적인 공간 및 주제 제어 메커니즘을 소개했습니다. 하지만 DiT(Diffusion Transformer) 아키텍처는 효율적이고 유연한 제어에서 여전히 어려움을 겪고 있습니다. 이를 감안하여 우리는 효율성과 유연성을 통합한 새로운 프레임워크인 EasyControl을 제안합니다.

- **Technical Details**: EasyControl은 세 가지 주요 혁신을 바탕으로 구축됩니다. 첫째, 경량 Condition Injection LoRA 모듈을 도입하여 조건 신호를 독립적으로 처리하며, 이를 플러그 앤 플레이 솔루션으로 활용할 수 있도록 설계했습니다. 둘째, 고정 해상도로 입력 조건을 표준화하여 임의의 종횡비로 이미지를 생성할 수 있는 Position-Aware Training Paradigm을 제안합니다. 셋째, 조건 생성 작업에 적합하게 조정된 Causal Attention 메커니즘과 KV Cache 기법을 개발하여 이미지 합성의 지연 시간을 크게 줄입니다.

- **Performance Highlights**: EasyControl은 다양한 응용 시나리오에서 뛰어난 성능을 달성하며 효율성과 유연성을 동시에 확보합니다. 각 조건에 대해 독립적인 조건 브랜치를 만들어 커스터마이징된 모델과의 원활한 통합을 지원합니다. 프레임워크는 또한 다양한 해상도와 종횡비의 이미지 생성을 가능하게 하여 고품질 생성과 다양한 요구 사항에 대한 적응성을 균형 있게 유지합니다.



### Erase Diffusion: Empowering Object Removal Through Calibrating Diffusion Pathways (https://arxiv.org/abs/2503.07026)
Comments:
          accepted by CVPR 2025

- **What's New**: 이 논문에서는 객체 제거를 위한 새로운 방식인 EraDiff를 제안합니다. EraDiff는 기존의 표준 확산(diffusion) 패러다임의 최적화 방법을 개선하여, 객체 제거의 효과성을 극대화할 수 있는 경로를 설정합니다. 이 모델은 Chain-Rectifying Optimization (CRO)라는 새로운 최적화 패러다임을 도입하여 객체 제거를 위한 혁신적인 확산 전이 경로를 구축합니다.

- **Technical Details**: EraDiff의 구조는 객체를 효과적으로 제거하기 위해 Self-Rectifying Attention (SRA) 메커니즘을 사용합니다. SRA 메커니즘은 자가 주의(self-attention) 활성화를 조절하여 모델이 아티팩트를 회피하고 생성된 콘텐츠의 일관성을 향상시킵니다. 이 최적화 접근법은 기존의 확산 경로가 아닌, 객체에서 배경으로의 직접적인 확산 경로를 통해 객체 제거를 수행할 수 있도록 합니다.

- **Performance Highlights**: 제안된 EraDiff는 공개된 OpenImages V5 데이터셋에서 최첨단 성능을 달성했으며, 실제 환경에서도 우수한 결과를 보여줍니다. 이 접근법은 객체를 제거하는 과정에서 발생할 수 있는 예상치 못한 아티팩트를 효과적으로 회피하여, 보다 자연스러운 배경을 생성합니다.



### HybridReg: Robust 3D Point Cloud Registration with Hybrid Motions (https://arxiv.org/abs/2503.07019)
Comments:
          2025, Association for the Advancement of Artificial Intelligence

- **What's New**: 이 논문은 HybridReg라는 새로운 3D 포인트 클라우드 등록 접근법을 제시하며, 동적 전경에서 하이브리드 모션을 처리하기 위한 불확실성 마스크를 학습합니다. 기존의 실내 데이터셋은 주로 고정된 동작을 가정해 비고정 동작을 가진 장면들을 효과적으로 처리하지 못했습니다. 또, 비고정 데이터셋은 객체 수준에서 제한적으로 적용되어 복잡한 장면에 잘 일반화되지 못하는 문제를 해결하고자 합니다.

- **Technical Details**: HybridReg는 배경에 rigid(강체) 동작을, 전경에 non-rigid(비강체) 및 rigid 동작을 고려하여 하이브리드 모션을 수용할 수 있는 포괄적인 데이터셋 HybridMatch를 구축했습니다. 이 데이터셋은 다양한 변형 전경을 제어 가능한 방식으로 정렬하도록 설계되었습니다. 또한, 효율적인 Feature extraction과 correspondence matching을 위해 불확실성 마스크를 추론하는 확률적 모델을 도입하여, 불필요한 간섭을 줄이는 방법으로 설정했습니다.

- **Performance Highlights**: 광범위한 실내 및 실외 데이터셋에 대한 실험을 통해 HybridReg의 뛰어난 성능을 입증했습니다. 이 방법은 비고정 동작을 포함하는 복잡한 장면에서 state-of-the-art 성능을 달성하며, 특히 HybridMatch의 사용이 중요한 기여를 하고 있습니다. 이러한 성과는 하이브리드 모션을 처리하는 최초의 연구임을 강조하며, 기존의 접근 방식에 비해 성능 저하가 없음을 나타내고 있습니다.



### SDFA: Structure Aware Discriminative Feature Aggregation for Efficient Human Fall Detection in Video (https://arxiv.org/abs/2503.07008)
Comments:
          Published IEEE Transactions on Industrial Informatics

- **What's New**: 이 논문에서는 노인들이 흔히 경험하는 낙상을 효과적으로 탐지할 수 있는 새로운 모델, SDFA(Skeleton-based Fall Detection Algorithm)를 제안합니다. 기존의 하드웨어와 소프트웨어의 복잡성 또는 사생활 침해 문제를 넘어, 저해상도 비디오에서 추출한 인체 스켈레톤(skeleton) 데이터를 활용하여 보다 저렴하고 안전한 모니터링 방안을 모색하고 있습니다.

- **Technical Details**: SDFA 모델은 여러 차원으로 각 joint 위치에서의 운동 정보를 포착하여 이들을 고차원 공간에 투영합니다. 이를 통해 얻은 통합 joint 및 motion 특성을 통해 구조적 변화를 인식합니다. separable convolution과 Graph Convolutional Networks (GCN)를 결합하여 높은 성능과 낮은 계산 복잡도를 실현했습니다. 또한 이 시스템은 고해상도 카메라 없이도 효과적으로 기능할 수 있습니다.

- **Performance Highlights**: 연구 결과, SDFA 모델은 다섯 개의 대규모 데이터셋에서 실행되었으며, 기존 모델보다 빠르게 작동하고 뛰어난 성능을 나타냈습니다. 특히, 다양한 활동과 낙상을 정확하게 탐지하는 능력을 보여주며, 메모리 효율과 연산 성능이 기존의 최첨단 기법들과 비교했을 때 현저히 낮음을 입증했습니다.



### NukesFormers: Unpaired Hyperspectral Image Generation with Non-Uniform Domain Alignmen (https://arxiv.org/abs/2503.07004)
- **What's New**: 이 논문에서는 비례에 맞지 않는 히퍼스펙트럼 이미지 생성(UnHIG)의 과제를 해결하기 위해, Range-Null Space Decomposition (RND) 방법론을 활용하여 null space의 상호작용을 모델링하였습니다. 특히, 비례 데이터의 기하학적 및 스펙트럼 분포를 효과적으로 정렬하기 위해 대비 학습을 도입했습니다. 이를 통해, 기존의 unpaired HIG 방법이 직면한 문제들을 해결하고, 새로운 벤치마크를 설정하는 데 기여하고 있습니다.

- **Technical Details**: 제안된 방법론은 비례하지 않는 히퍼스펙트럼 데이터를 연속적인 성분과 감쇠된 성분으로 체계적으로 분해합니다. RND 방법론에 기반하여 range space와 null space로 나눈 후, dual-dimensional contrastive learning을 통해 range space 내에서의 연속 요소를 효과적으로 집계합니다. 또한, Kolmogorov-Arnold Networks (KANs)를 기반으로 한 Non-Uniform Matrix Object-Aware Mechanism을 도입하여 null space의 보상을 강화하고 고차원 맞춤화를 지원합니다.

- **Performance Highlights**: 실험을 통해 제안된 방법이 다양한 벤치마크에서 최신 성능을 달성했음을 입증하였습니다. 특히, 고주파 성분 보상 및 효과적인 cross-domain 상호작용이 강조되고 있으며, 이로 인해 UnHIG 분야에서의 연구에 새로운 방향성을 제시하고 있습니다. 연구 결과는 산업 적용 가능성을 높이는 데 중요한 역할을 할 것으로 기대됩니다.



### Taking Notes Brings Focus? Towards Multi-Turn Multimodal Dialogue Learning (https://arxiv.org/abs/2503.07002)
- **What's New**: 이번 논문에서는 실제 인간 대화를 보다 정확하게 반영하기 위해 다중 턴 멀티모달 대화 데이터셋인 MMDiag를 소개합니다. 이 데이터셋은 질문 간의 강한 상관성을 바탕으로 생성되어, 시각적 정보와 질문 상호작용 간의 복잡한 관계를 모델의 학습에 도움이 됩니다. MMDiag는 다중 턴 멀티모달 대화 학습을 위한 강력한 벤치마크 역할을 하며, 모델의 근거 및 추론 능력에 더 많은 도전 과제를 제시합니다.

- **Technical Details**: 다양한 멀티모달 태스크에서의 성능 향상을 위해 MMDiag는 시각적 세부정보를 포함한 복잡한 대화 예시들로 구성되어 있습니다. 이 논문에서 제안하는 DiagNote 모델은 Deliberate와 Gaze라는 두 가지 주요 모듈로 이루어져 있으며, 이 두 모듈은 다중 턴 대화 전반에 걸쳐 상호작용합니다. 이를 통해 모델은 시각적 정보와 언어 정보를 보다 효과적으로 처리하고 추론할 수 있는 능력을 갖추게 됩니다.

- **Performance Highlights**: DiagNote 모델은 MMDiag와 다른 벤치마크에서 추론 및 근거 능력에 대한 평가를 통해 기존의 MLLMs에 비해 성능이 현저히 향상된 것을 입증했습니다. 이 모델은 다중 턴 대화에서 정확하고 맥락에 적합한 응답을 생성하는 데 성공했으며, MMDiag는 멀티모달 대화의 더 많은 도전 과제를 제공하여 이 분야의 연구에 중요한 기여를 하고 있습니다.



### Frequency-Aware Density Control via Reparameterization for High-Quality Rendering of 3D Gaussian Splatting (https://arxiv.org/abs/2503.07000)
Comments:
          Accepted to AAAI2025

- **What's New**: 이번 연구에서는 3D Gaussian Splatting (3DGS)의 밀도와 스케일 간의 명확한 관계를 설정하는 새로운 접근 방식을 제안합니다. 밀도를 적응적으로 제어하고 고주파 정보가 많은 영역에서 더 많은 Gaussians를 생성함으로써 장면 세부 정보를 개선할 수 있다는 점이 주목할 만합니다. 특히 밀도와 스케일 간의 직접적인 관계를 통해 렌더링 품질을 향상시키는 방법을 모색합니다.

- **Technical Details**: 제안된 방법은 densification과 deletion 전략을 활용하여 밀도를 제어합니다. 밀도 정보를 각 위치에서 Gaussians의 수로 정량화하고, 이를 바탕으로 스케일을 재파라미터화하여 밀도와 스케일 간의 관계를 명시적으로 설정합니다. 또한, 고주파가 있는 영역에서의 밀도를 더욱 개선하기 위해 동적 임계값과 스케일 기반 필터를 적용합니다.

- **Performance Highlights**: 실험 결과, 제안된 FDS-GS 방법은 기존의 최첨단 기술들과 비교하여 정량적 및 정성적으로 우수한 성능을 보여주었습니다. 특히, 적은 수의 Gaussians로 더 높은 렌더링 품질을 달성하여, NVS(새로운 뷰 합성) 응용 분야에서의 가능성을 확장합니다. 이로 인해 VR 및 AR 산업에서의 활용이 더욱 기대됩니다.



### SOYO: A Tuning-Free Approach for Video Style Morphing via Style-Adaptive Interpolation in Diffusion Models (https://arxiv.org/abs/2503.06998)
- **What's New**: 이 논문에서는 새로운 확산 기반 프레임워크인 SOYO를 소개합니다. SOYO는 비디오 스타일 모핑(video style morphing)을 가능하게 하여 다양한 스타일 간의 매끄러운 전환을 지원합니다. 이 방법은 미세 조정 없이 사전 훈련된 텍스트-이미지 확산 모델을 활용하여 구조적 일관성을 유지하며 스타일 전환을 구현합니다. 또한, 기존의 선형 간격 보간이 부자연스러운 스타일 모핑을 유발할 수 있는 문제를 해결하기 위해, 두 스타일 이미지 간의 적응형 샘플링 스케줄러를 제안하여 영상의 시각적 효과를 극대화합니다.

- **Technical Details**: SOYO 프레임워크는 시퀀스 프레임 간의 구조적 일관성을 유지하는 동시에 텍스처 요소를 부드럽게 통합하는 방식으로 작동합니다. 키와 값 특징을 동적으로 보간해 텍스처 전환을 매끄럽게 하며, 원본 비디오에서 구조적 정보를 보존하는 쿼리 주입(query injection) 메커니즘을 활용합니다. AdaIN(a Statistical Parameter의 동적 적용)을 통해 스타일 특성을 점진적으로 변조하여 색상 분포와 텍스처 특성이 일관되도록 합니다. 이 논문은 이러한 구성 요소를 통해 비디오 스타일 모핑의 성능을 크게 향상시킵니다.

- **Performance Highlights**: 다양한 실험 결과, SOYO는 기존 방법들에 비해 비디오 스타일 모핑 성능에서 눈에 띄게 우수한 결과를 보였습니다. 특히, SOYO는 구조적 일관성을 유지하면서도 부드럽고 안정적인 스타일 전환을 가능하게 하여, 복합적 감정이나 주제의 발전을 전달하는 데 효과적입니다. 소규모 다수의 씬을 바탕으로 한 멀티-씬 벤치마크를 제공하여 포괄적인 평가가 가능하도록 하고, 이는 예술적 제작의 요구를 충족시키는 데 기여합니다.



### Public space security management using digital twin technologies (https://arxiv.org/abs/2503.06996)
- **What's New**: 최근 공공 공간의 보안을 향상시키기 위한 혁신적인 접근 방식으로 디지털 트윈(Digital Twin) 기술이 부각되고 있습니다. 이 연구에서는 그리스 아테네의 한 지하철역에 대한 디지털 트윈을 구축하여 공공 안전 관리에 대한 새로운 통찰을 제공합니다. 이 모델은 다양한 보안 위협 시나리오를 적용하고 최적의 감시 시스템 구성을 파악하는 데 중점을 두었습니다.

- **Technical Details**: 이 연구에서는 FlexSim 시뮬레이션 소프트웨어를 사용하여 디지털 트윈 모델을 생성하고, 주요 지점과 승객 흐름을 모델링했습니다. 카메라 배치 및 각도 최적화를 통해 보안 효과성을 평가하며, 모델은 의심스러운 행동을 탐지하고 잠재적 보안 위협을 예측하는 기능을 갖추었습니다. DTaaSS (Digital Twin-as-a-Security-Service) 프레임워크를 통해 디지털 트윈의 구조와 운용을 재현하고 있습니다.

- **Performance Highlights**: 연구 결과, 감시 카메라의 전략적 배치와 각도 조정이 의심스러운 행동 탐지에 큰 영향을 미쳤습니다. 디지털 트윈의 활용을 통해 다양한 시나리오를 평가하고 각 경우에 가장 적합한 카메라 설치를 도출할 수 있었습니다. 이 연구는 현실 시간 시뮬레이션 및 데이터 기반 보안 관리의 가치와 스마트 보안 솔루션 개발에 기여할 것으로 기대됩니다.



### Bridge Frame and Event: Common Spatiotemporal Fusion for High-Dynamic Scene Optical Flow (https://arxiv.org/abs/2503.06992)
- **What's New**: 이 논문은 고속 동적 장면에서의 광학 흐름(optical flow) 추정을 위한 새로운 방법을 제안합니다. 기존의 방법들은 프레임과 이벤트 모달리티 간의 직접적인 융합을 시도하지만, 이는 서로 다른 데이터 표현으로 인해 비효율적입니다. 본 연구에서는 공통 잠재 공간(common-latent space)을 통해 모달리티 간의 간극을 줄여주며, 특히 시각 경계(localization)와 움직임 상관관계(fusion)의 개념을 도입합니다.

- **Technical Details**: 제안된 방법은 공통 시공간 융합(common spatiotemporal fusion) 프레임워크인 ComST-Flow를 기준으로 세 가지 주요 단계를 포함합니다. 첫째, 이벤트 모달리티와 프레임 모달리티 간의 유사성 분포를 분석하여 참조 경계를 설정하고, 둘째, 시공간 그래디언트(spatiotemporal gradient)를 사용하여 서로 다른 모달리티 간의 복합적인 움직임 지식을 융합합니다. 마지막으로, 크로스 모달 변환기(cross-modal transformer) 아키텍처를 도입하여 시공간 연관(feature correlation)을 보다 명확하게 융합합니다.

- **Performance Highlights**: 제안된 방법은 고속 동적 장면에서의 밀집(dense)하고 연속적인(optical flow) 광학 흐름 추정에서 우수한 성능을 보입니다. 실험 결과, 기존 방법 대비 더 나은 스파시오-타임(feature representation)과 일관된 흐름 추정을 달성하는 데 성공하였습니다. 이로 인해, 본 연구는 고속 동작에서의 광학 흐름 추정 개선에 기여할 것으로 기대됩니다.



### ConcreTizer: Model Inversion Attack via Occupancy Classification and Dispersion Control for 3D Point Cloud Restoration (https://arxiv.org/abs/2503.06986)
- **What's New**: 이 논문은 자율주행차에서 3D 포인트 클라우드 데이터의 사용 증가로 인한 개인 정보 보호 문제를 다루고 있습니다. 특히, 3D 데이터에서 모델 역전 공격(model inversion attack)의 적용이 미흡한 상황을 해결하기 위해 ConcreTizer라는 새로운 공격 방식을 제안합니다. 분석을 통해 3D 포인트 클라우드의 고유한 도전 과제가 드러났으며, Voxel Occupancy Classification과 Dispersion-Controlled Supervision 기법이 포함되어 효율적으로 포인트 클라우드를 복원하는 방법을 제시합니다.

- **Technical Details**: ConcreTizer는 3D 포인트 클라우드를 복원하기 위한 모델 역전 공격으로, 주로 voxel 기반 데이터 처리에 맞춰 고안되었습니다. 이 모델은 빈 voxel과 비어 있지 않은 voxel을 구별하는 Voxel Occupancy Classification을 사용하고, 비어 있지 않은 voxel의 분산을 조절하는 Dispersion-Controlled Supervision을 채택하여 문제를 해결합니다. 실험에서는 KITTI와 Waymo와 같은 널리 사용되는 데이터셋을 활용해 ConcreTizer의 성능을 입증합니다.

- **Performance Highlights**: ConcreTizer는 3D 포인트 클라우드 장면을 성공적으로 복원하며, 정량적 및 정성적 평가를 통해 일반화된 성능을 보여줍니다. 또한, 3D 객체 감지와 같은 특정 임무에서도 우수한 폭격 성능을 제공합니다. 연구 결과는 3D 데이터의 취약성과 강력한 방어 전략의 필요성을 강조합니다.



### Griffin: Aerial-Ground Cooperative Detection and Tracking Dataset and Benchmark (https://arxiv.org/abs/2503.06983)
Comments:
          8 pages, 7 figures. This work has been submitted to IROS 2025 for possible publication

- **What's New**: 이번 연구는 자율주행 기술의 한계를 극복하기 위한 공중-지상 협력 3D 인식 시스템을 제안합니다. 새로운 멀티 모달 데이터셋 Griffin을 소개하고, 다양한 환경에서의 UAV의 동적 관찰 및 3D 주석을 통합하여 실용적인 인식 환경을 제공합니다. 또한, AGILE 프레임워크와 베차밍(Benchmarking) 프로토콜을 통해 통신 효율성과 반응성을 평가할 수 있는 체계를 갖추었습니다.

- **Technical Details**: Griffin 데이터셋은 200개 이상의 역동적인 장면과 30,000 프레임 이상의 이미지를 포함하며, 다양한 날씨 조건과 UAV의 비행 높이에 대한 3D 주석을 제공합니다. 이 데이터는 CARLA와 AirSim의 공동 시뮬레이션을 통해 생성되었습니다. 연구는 조기 융합(Early Fusion), 중간 융합(Intermediate Fusion), 후기 융합(Late Fusion) 전략을 설명하며, UAV와의 협력이 이를 어떻게 개선할 수 있는지를 분석했습니다.

- **Performance Highlights**: 이 연구에서 제안하는 AGILE 프레임워크는 크로스 뷰 기능을 동적으로 조정하여 통신 오버헤드와 인식 정확성 사이의 균형을 유지합니다. 실험 결과, 공중-지상 협력 인식 시스템의 효과성을 입증했으며, 비교 분석을 통해 개발할 추가 연구 방향성을 제시했습니다. 데이터셋과 코드는 제공된 링크를 통해 공개되어, 연구자들이 쉽게 접근할 수 있도록 하였습니다.



### Lightweight Multimodal Artificial Intelligence Framework for Maritime Multi-Scene Recognition (https://arxiv.org/abs/2503.06978)
Comments:
          19 pages, 4 figures, submitted to Engineering Applications of Artificial Intelligence

- **What's New**: 이번 연구는 해양 다중 장면 인식을 위한 혁신적인 멀티모달 인공지능 프레임워크를 제안합니다. 이 프레임워크는 이미지 데이터, 텍스트 설명, 그리고 다중모달 대규모 언어 모델(Multimodal Large Language Model, MLLM)에 의해 생성된 분류 벡터를 통합하여 인식 정확도를 높이고 풍부한 의미를 제공합니다. 실험 결과, 새로운 모델이 98%의 정확도로 이전의 최첨단 모델보다 3.5% 향상되었습니다.

- **Technical Details**: 프레임워크의 핵심은 이미지 특징 추출을 위한 첨단 기술인 Swin Transformer와 텍스트 데이터를 처리하는 BERT, 그리고 분류 벡터를 처리하는 다층 퍼셉트론(Multi-Layer Perceptron, MLP)으로 구성된 효율적인 기능 추출 과정입니다. 이 시스템은 각 모달리티의 가장 관련성 높은 정보를 강화하여 전체 모델의 견고성을 보장합니다. 또한, 활성화 인식 가중치 양자화(Activation-aware Weight Quantization, AWQ)를 통해 모델 크기를 68.75MB로 줄이면서도 실시간 해양 장면 인식을 위한 성능을 유지합니다.

- **Performance Highlights**: 제안된 프레임워크는 해양 환경에서의 실시간 배치 작업에 매우 적합하며, 자율 해양 차량(Autonomous Surface Vehicles, ASVs)의 환경 모니터링과 재난 대응에 이로운 결과를 제공합니다. 이 연구는 다양한 해양 환경에서 인식 정확도를 개선하기 위한 다중모달 융합 기술의 가능성을 보여주며, 자원 제한된 환경에서도 높은 성능을 발휘하는 솔루션을 제시합니다.



### Task-Specific Knowledge Distillation from the Vision Foundation Model for Enhanced Medical Image Segmentation (https://arxiv.org/abs/2503.06976)
Comments:
          29 pages, 10 figures, 16 tables

- **What's New**: 이번 연구에서는 제한된 데이터로 의료 이미지 세분화를 위해 대형 Vision Foundation Model (VFM)의 지식을 소형 작업-특화 모델에 효과적으로 활용할 수 있는 새로운 지식 증류 프레임워크를 제안합니다. 이 방법은 기존의 일반적인 지식 증류 방식에서 벗어나, 최적화된 세분화 작업에 맞춰 VFM을 미세 조정(fine-tuning)하여 작업-특화 피쳐를 캡처하는 데 중점을 둡니다.

- **Technical Details**: 제안된 방법에서는 Low-Rank Adaptation (LoRA)을 사용하여 VFM의 재조정을 효율적으로 수행함으로써 계산 비용을 줄이고, 확산 모델(difffusion models)을 활용해 생성된 합성 데이터를 통해 데이터 부족 시나리오에서 모델 성능을 향상시킵니다. 이 프레임워크는 중간 표현(intermediate representations)과 최종 세분화 출력(segmentation outputs)을 정렬하여 소형 모델이 높은 정확도의 세분화를 위해 필요한 중요 특성을 상속받도록 합니다.

- **Performance Highlights**: 다양한 데이터가 제한된 상황에서도 제안된 작업-특화 지식 증류 방법이 최신 기법들과 비교할 때 일관되게 우수한 성능을 보이는 것을 실험적으로 입증했습니다. KidneyUS 데이터 세트에서는 작업-특화 KD가 80개의 레이블 샘플을 사용하여 작업 비특화 KD 대비 28% 더 높은 Dice 점수를 달성하였고, CHAOS 데이터 세트에서는 100개의 레이블 샘플로 MAE 대비 11% 개선된 결과를 보였습니다.



### Asymmetric Visual Semantic Embedding Framework for Efficient Vision-Language Alignmen (https://arxiv.org/abs/2503.06974)
Comments:
          9 pages, 5 figures, The 39th Annual AAAI Conference on Artificial Intelligence

- **What's New**: 새로운 연구 프레임워크인 Asymmetric Visual Semantic Embedding (AVSE)을 제안하여 이미지와 텍스트 간의 시각적 의미 유사성을 학습합니다. 이 프레임워크는 서로 다른 텍스트 입력에 맞춤화된 이미지의 다양한 영역에서 특징을 동적으로 선택하여 유사성을 계산합니다. 특히, AVSE는 비대칭 이미지와 텍스트 임베딩 간의 시각적 의미 유사성을 효율적으로 계산하기 위한 새로운 모듈을 도입합니다.

- **Technical Details**: AVSE는 이미지를 여러 관점에서 샘플링하기 위해 방사형 편향 샘플링 모듈을 설계하여 이미지 패치를 샘플링합니다. 주요 구성 요소인 Asymmetric Embedding Optimal Matching (AEOM) 모듈은 메타-의미 임베딩을 통해 두 가지 모달리티의 임베딩을 분해하고, 최적의 일치를 찾으며 시각적 의미 유사성을 계산합니다. 이 과정에서 차원 정규화 손실 함수를 도입하여 다양한 뷰에서 추출된 특징의 의미적 풍부함을 강화합니다.

- **Performance Highlights**: AVSE 모델은 MS-COCO 및 Flickr30K 데이터세트에서 폭넓게 평가되어, 최근의 최첨단 방법들에 비해 뛰어난 성능을 보여줍니다. 특히, AVSE는 로컬 매칭 방법에 비해 상당히 빠른 속도를 자랑하며, 다중 뷰 이미지 특징을 획득할 수 있어 이미지-텍스트 검색에서의 효율성을 높입니다.



### A Multimodal Benchmark Dataset and Model for Crop Disease Diagnosis (https://arxiv.org/abs/2503.06973)
Comments:
          Accepted by ECCV 2024 (14 pages, 8 figures)

- **What's New**: 이번 연구에서는 농업 분야의 질병 진단을 위한 다중 모달 대화형 AI의 가능성을 탐구하며, CDDM(Crop Disease Domain Multimodal) 데이터셋을 소개합니다. 이 데이터셋은 137,000개의 농작물 질병 이미지와 100만 개의 질문-답변 쌍으로 구성되어 있으며, 농업 지식의 다양한 범위를 포괄합니다. 이를 통해 고급 질문-응답 시스템을 개발하고 농업 전문가에게 유용한 조언을 제공할 수 있는 토대를 마련하고자 합니다.

- **Technical Details**: CDDM 데이터셋은 다양한 농작물 질병과 관련된 질문-답변의 상호작용을 통해 다중 모달 학습 기법을 적용합니다. 특히, 시각 인코더, 어댑터 및 언어 모델을 동시에 미세 조정하는 새로운 LoRA(low-rank adaptation) 전략을 채택하여 농작물 질병 진단의 정확성을 개선했습니다. 이 연구의 방법론은 농업 기술 연구를 촉진하기 위한 벤치마크와 함께 제공됩니다.

- **Performance Highlights**: CDDM 데이터셋을 통해 최신 다중 모달 모델을 미세 조정한 결과, 농작물 질병 진단의 질적 향상이 나타났습니다. 특히, 일반적인 VQA 시스템에서 겪는 탄력성 문제를 해결함으로써 보다 신뢰할 수 있는 진단 결과를 제공할 수 있게 되었습니다. 이 데이터셋과 모델은 오픈소스로 제공되어 연구자들이 농업 분야에서 다중 모달 학습을 발전시키는 데 기여할 것입니다.



### MIGA: Mutual Information-Guided Attack on Denoising Models for Semantic Manipulation (https://arxiv.org/abs/2503.06966)
- **What's New**: 이 논문에서는 일반적으로 사용되는 이미지 디노이징 모델의 근본적인 취약점을 드러내는 Mutual Information-Guided Attack (MIGA)라는 새로운 적대적 공격 방법을 제안합니다. 기존의 적대적 공격은 주로 시각적 선명도를 저하시키는 데 초점을 맞추었지만, MIGA는 의미론적 정보의 왜곡에 중점을 두었습니다. 이로 인해 디노이징 모델은 시각적으로 깨끗한 이미지를 생성하면서도 의미론적 내용이 손상된 결과를 도출하도록 강요받습니다.

- **Technical Details**: MIGA는 원본 이미지와 디노이즈된 이미지 간의 상호 정보(mutual information)를 최소화함으로써 작동합니다. 이 방법은 두 가지 손실을 동시에 최적화하여 디노이징 과정 중 의미론적 특성을 조작하는 미세한 섭동을 도입합니다. MIGA는 알려진 다운스트림 작업과 미지의 작업에 따라 다르게 적응하며, 후자의 경우 참조 이미지를 사용하여 의미론적인 수정을 통해 조정합니다.

- **Performance Highlights**: 연구에서는 MIGA를 네 가지 디노이징 모델과 다섯 개 데이터셋에서 테스트하였으며, 새로운 평가 지표를 설계하여 의미론적 조작을 측정하였습니다. 결과는 MIGA가 시각적으로는 깨끗하지만 의미론적으로는 왜곡된 출력을 생성하여 현실 세계의 다운스트림 작업에서 오작동을 유도할 수 있음을 증명했습니다. 이 연구는 딥 러닝 기반의 디노이징 모델들이 항상 강력하지 않으며, 실제 응용 프로그램에서 보안 위험을 초래할 수 있다는 점을 강조합니다.



### SeCap: Self-Calibrating and Adaptive Prompts for Cross-view Person Re-Identification in Aerial-Ground Networks (https://arxiv.org/abs/2503.06965)
- **What's New**: AGPReID(공중-지상 인물 재식별) 작업에 대한 새로운 접근 방식인 SeCap(자기 보정 및 적응형 프롬프트) 메서드를 제안합니다. 이 방법은 입력에 따라 동적으로 프롬프트를 보정하여 다양한 시점에서도 일관성 있는 특성을 추출할 수 있도록 돕습니다. 또한, 새로운 대규모 데이터셋 LAGPeR와 G2APS-ReID를 발표하여 AGPReID 연구를 위한 귀중한 데이터 지원을 제공합니다.

- **Technical Details**: SeCap 구조는 Prompt Re-calibration Module (PRM)과 Local Feature Refinement Module (LFRM)으로 구성되어, 입력에 기반하여 프롬프트를 자동으로 재보정합니다. PRM은 현재 보는 각도에 적합한 프롬프트를 생성하여 다양한 시점에 적응할 수 있도록 합니다. LFRM은 로컬 기능을 보정하여 시점 간 불변 정보를 추출하는 데 중점을 두고 있으며, 두 방향의 주의(attention) 메커니즘을 사용하여 다양한 기능을 동기화합니다.

- **Performance Highlights**: 실험 결과, SeCap은 기존 AGPReID 기법들에 비해 현저한 성능 향상을 보여주며, 상태 최적(SOTA) 결과를 달성했습니다. AGPReID 데이터셋에 대한 철저한 평가를 진행하여, 모델의 효과성을 확인했습니다. 따라서 SeCap은 AGPReID 작업을 위한 실용적이고 효과적인 솔루션으로 자리매김 할 것으로 기대됩니다.



### A Data-Centric Revisit of Pre-Trained Vision Models for Robot Learning (https://arxiv.org/abs/2503.06960)
Comments:
          Accepted by CVPR 2025

- **What's New**: 이번 연구는 프리트레인된 비전 모델(PVMs)의 최적 구성을 탐구하며, DINO와 iBOT은 MAE보다 뛰어난 성능을 보이지만, 비객체 중심 데이터(NOC)에서 학습할 경우 성능이 떨어진다는 것을 발견했습니다. 즉, NOC 데이터를 사용하여 객체 중심 표현을 학습할 수 있는 능력이 PVM의 성공에 중요한 요소임을 강조합니다. 이를 바탕으로, 우리는 객체 중심 표현을 유도하는 SlotMIM 방법을 디자인하였습니다.

- **Technical Details**: SlotMIM는 마스크 이미지 모델링(MIM) 및 대조 학습(contrastive learning)을 통합하여 NOC 데이터로부터 효과적인 표현 학습을 가능하게 합니다. 방법의 핵심 아이디어는 패치 수준의 이미지 토큰을 객체 중심 피쳐 추상화인 '슬롯'으로 그룹화하는 것입니다. 이를 통해 객체 중심 기술을 효과적으로 적용할 수 있도록 만듭니다.

- **Performance Highlights**: 실험 결과, SlotMIM는 다양한 작업에서 기존 방법에 비해 성능이 크게 개선되었고, 241K 샘플로 훈련하였을 때에도 1M 이상의 샘플을 사용한 방법을 초능가했습니다. NOC 데이터는 적절히 활용되면 이전에 생각했던 것보다 더 확장 가능하고 효율적인 학습 자원으로 나타났습니다.



### LatexBlend: Scaling Multi-concept Customized Generation with Latent Textual Blending (https://arxiv.org/abs/2503.06956)
Comments:
          cvpr2025

- **What's New**: 본 논문에서는 LaTexBlend라는 새로운 프레임워크를 제안하여 다중 개념(customized concept) 텍스트-이미지 생성의 품질 및 계산 효율성을 효과적으로 향상시킵니다. LaTexBlend는 단일 개념을 표현하고, 여러 개념을 Latent Textual 공간에서 혼합하는 방식으로 다중 개념 생성의 확장성과 일관성을 동시에 확보합니다. 이 프레임워크는 개념 정보를 집약한 라텐트 텍스트 특징(latent textual features)을 활용하여, 높은 충실도를 유지하는 동시에 레이아웃의 일관성도 보장합니다.

- **Technical Details**: LaTexBlend는 각 개념을 개별적으로 커스터마이즈하고, 이를 compact한 라텐트 텍스트 특징으로 구성된 개념 은행(concept bank)에 저장합니다. 훈련 과정에서는 보조적인 텍스트 인코딩 플로우를 도입하여 그래디언트 전파를 교정하고, 다중 개념 생성 시 라텐트 텍스트 공간에서 자유롭게 조합할 수 있도록 합니다. 이를 통해, LaTexBlend는 다양한 사용자 요구를 충족하는 동시에 계산 효율성을 대폭 증가시킵니다.

- **Performance Highlights**: 실험 결과, LaTexBlend는 여러 개념을 수월하게 통합하여 높은 주제 충실도(subject fidelity)와 일관된 레이아웃을 유지하는 데 성공했습니다. 특히, LaTexBlend는 생성 품질과 계산 효율성을 동시에 개선하여 기존 방법들에 비해 현저한 성과를 보였습니다. 또한, 이 방법은 개념 수가 증가해도 복잡도가 선형으로 증가하며 추가적인 추론 비용을 발생시키지 않아 매우 효율적입니다.



### Motion Anything: Any to Motion Generation (https://arxiv.org/abs/2503.06955)
- **What's New**: 이 논문은 'Motion Anything'라는 새로운 다중 모드 모션 생성 프레임워크를 제안합니다. 기존의 마스크 자가회귀 방법론을 넘어, 조건에 따라 동적 프레임과 신체 부위를 우선시하는 메커니즘을 도입하여 더욱 정교한 제어가 가능해졌습니다. 또한 본 연구는 텍스트와 음악으로 구성된 2,153개의 새로운 모션 데이터셋인 Text-Motion-Dance (TMD)를 소개하여 연구 커뮤니티의 요구를 충족시킵니다.

- **Technical Details**: 제안된 접근 방식은 Attention-based Mask Modeling을 통해 시간적(temporal) 및 공간적(spatial) 차원에서 모션 시퀀스를 보다 잘 생성하기 위해 조건을 동적으로 인코딩합니다. 모델에서는 상황에 따라 키 프레임 및 액션을 마스킹하여 전체 모션 제어를 조정합니다. 특히, 타임라인에서는 서로 다른 입력 모달리티의 정렬을 처리하여 동기화된 모션 생성이 이루어집니다.

- **Performance Highlights**: Motion Anything은 여러 벤치마크에서 최첨단 기법보다 15% 향상된 FID 점수를 기록하였고, AIST++ 및 TMD 데이터셋에서도 일관된 성능 향상을 보여주었습니다. 이는 다중 모달 조건을 효과적으로 통합하여 제어력을 높이고, 생성된 모션의 일관성을 더욱 강화하는 데 중점을 두었기 때문입니다.



### Approximate Size Targets Are Sufficient for Accurate Semantic Segmentation (https://arxiv.org/abs/2503.06954)
- **What's New**: 이 논문은 이미지 수준의 타겟(target)으로 세그멘테이션(segmentation)을 개선하는 새로운 방법을 소개합니다. 이 방법은 바이너리 클래스 태그를 사용해 대략적인 객체 크기 분포를 근사하는 접근법으로, 기존의 네트워크 구조를 활용하여 세그멘테이션 문제를 해결할 수 있게 합니다. 저자들은 PASCAL VOC 데이터셋을 사용하여 새로운 인간 주석(annotations)을 통해 이 아이디어를 검증하고, COCO 및 의료 데이터에서도 결과를 보여줍니다.

- **Technical Details**: 저자들은 기존의 픽셀 정밀한 주석을 사용하지 않고, Kullback-Leibler divergence를 이용한 손실(loss) 함수를 통해 평균 예측을 모델링합니다. 세그멘테이션 네트워크는 이미지에서 각 픽셀마다 K개의 객체 클래스를 확률 분포로 예측하며, 이는 해당 이미지의 객체 크기에 대한 예측을 포함합니다. 저자들은 다양한 형태의 약한 지도(weak supervision) 학습 방법을 논의하고, 자기 지도(self-supervised) 학습 및 CRF 기반(pairwise) 손실 함수를 활용하여 세그멘테이션의 성능을 높이고자 합니다.

- **Performance Highlights**: 연구 결과, 제안된 방법은 이미지 수준의 주석만을 사용하였음에도 불구하고 픽셀 수준의 정밀한 감독에 대한 정확도와 유사한 성능을 보여주었습니다. 특히, 특정 클래스의 경우, 검증 정확도가 픽셀 수준 감독보다 높은 것으로 나타났습니다. 이러한 성과는 잘못된 태그(target)로 인한 오차에 강한 내성을 보이며, 앞으로 더 간단하고 일반적인 해결책을 제시할 수 있는 기반이 될 것입니다.



### Large Language Model Guided Progressive Feature Alignment for Multimodal UAV Object Detection (https://arxiv.org/abs/2503.06948)
- **What's New**: 이번 연구에서 제안하는 LPANet은 Multimodal UAV object detection의 성능 개선을 위해 Large Language Model (LLM)을 활용한 Progressive feature Alignment Network입니다. 기존의 방법들이 모달리티 간의 의미적 불일치를 간과한 것과는 달리, 본 연구는 ChatGPT를 통해 생성된 세부 객체 설명을 활용하여 모달리티 간의 의미적 및 공간적 정렬을 점진적으로 진행합니다. 이 접근법을 통해 두 개의 공공 데이터셋에서 최고 성능의 멀티모달 UAV 객체 탐지기를 초과하는 결과를 보였습니다.

- **Technical Details**: LPANet은 세 가지 주요 모듈로 구성되어 있습니다. 첫째, Semantic Alignment Module (SAM)은 다양한 모달에서 추출된 의미적 특징을 근접하게 정렬하여 의미 차이를 완화합니다. 둘째, Explicit Spatial Alignment Module (ESM)은 의미 관계를 이용해 특징 레벨의 오프셋을 추정하여 모달리티 간의 공간적 정렬을 가능케 합니다. 마지막으로, Implicit Spatial Alignment Module (ISM)은 인접 영역으로부터 중요한 특징을 집계하여 암묵적인 공간 정렬을 수행합니다.

- **Performance Highlights**: Two public datasets인 DroneVehicle 및 VEDAI에서 진행된 실험에서 LPANet은 기존의 최첨단 방법을 초과하는 성능을 보여주었습니다. 구체적으로, 제안된 Semantic Alignment Module과 Explicit 및 Implicit Spatial Alignment Module의 결합은 모달리티 간의 의미적 및 공간적 불일치를 효과적으로 줄여줍니다. 통합된 프레임워크는 다양한 환경과 조건에서도 높은 검출 성능을 유지하는 특징을 가지며, 상업적 응용 가능성도 높입니다.



### Aligning Instance-Semantic Sparse Representation towards Unsupervised Object Segmentation and Shape Abstraction with Repeatable Primitives (https://arxiv.org/abs/2503.06947)
Comments:
          15 pages, 15 figures, 8 tables

- **What's New**: 이번 연구에서 처음으로 제안된 것은 완전히 감시 없이 단일 단계로, 카테고리 없는 학습 기반의 3D 의미론적 표현입니다. 이는 인공 물체 포인트 클라우드를 위한 것으로, 인스턴스 형태 추상화, 인스턴스 분할, 의미론적 형태 추상화 등 다섯 가지 표현을 통합합니다. 기존의 방법은 다단계 훈련을 요구하는데 반해, 우리는 하나의 프레임워크 내에서 이를 모두 해결할 수 있는 방법을 제시합니다.

- **Technical Details**: 이 연구에서는 고차원의 특성들이 저차원 부분공간에 존재하는 경향을 이용하여 의미론적 형태 표현을 달성합니다. 우리는 희소 표현(sparse representation)과 특성 정렬(feature alignment)을 통해 인스턴스와 의미론적 파트 특성을 정렬하고, 이를 통해 형상을 재구성하여 보다 일반화된 결과를 도출합니다. 이를 위해 Sparsemax 함수를 포함한 희소 잠재 멤버십 추구 방법(Sparse Latent Membership Pursuit, SLMP)을 도입하여 형상 추상화를 & 인스턴스 분할 및 의미론적 일관성을 확보합니다.

- **Performance Highlights**: 우리의 접근 방식은 지도 없이 유기성 있는 3D 의미론적 표현을 달성하는 데 중요한 발전을 이루었습니다. 또한, 변형 가능한 슈퍼쿼드릭(Deformable Superquadrics) 매개 변수를 사용하여 더 강력한 의미 해석을 위한 이중 학습을 제공, 의미의 중의성을 줄이고 형상을 보다 정교하게 표현할 수 있음을 보여주었습니다. 실험 결과, 다양한 형상으로부터 의미론적 추론의 충족도를 높이며, 형태 추상화에서도 기존의 방법보다 향상된 성능을 보입니다.



### CineBrain: A Large-Scale Multi-Modal Brain Dataset During Naturalistic Audiovisual Narrative Processing (https://arxiv.org/abs/2503.06940)
Comments:
          14 pages, 13 figures

- **What's New**: 이번 논문에서는 EEG(전기뇌파도)와 fMRI(기능적 자기공명영상) 기록을 동시에 수집한 첫 번째 대규모 데이터셋인 CineBrain을 소개합니다. CineBrain은 총 6명의 참가자가 시청한 '빅뱅이론' 에피소드를 기반으로 한 약 6시간의 내러티브 콘텐츠를 제공합니다. 본 연구는 이러한 독특한 데이터셋을 기반으로 EEG와 fMRI 신호를 융합하여 복잡한 오디오비주얼 자극을 효과적으로 재구성하는 CineSync라는 혁신적인 다중 모달 디코딩 프레임워크를 제안합니다. 또한, Cine-Benchmark라는 평가 프로토콜을 도입하여 의미적 및 지각적 차원에서 재구성을 평가합니다.

- **Technical Details**: CineSync는 다중 모달 융합 인코더(Multi-Modal Fusion Encoder)와 확산 기반 신경 잠재 디코더(diffusion-based Neural Latent Decoder)로 구성됩니다. 이 프레임워크는 EEG와 fMRI 신호를 융합하여 신호 재구성 품질을 크게 향상시키며, 두 모달리티의 특성을 보완합니다. 인코더는 두 개의 Transformer 기반 아키텍처를 적용하여 각 모달리티에서 멀티프레임 신호를 별도로 인코딩하고, 결합 대비 손실(combined contrastive loss)을 사용하여 시각 및 텍스트 표현과 함께 멀티모달 특징을 정렬합니다. 이후 융합된 뇌 표현은 신경 잠재 디코더의 입력으로 사용됩니다.

- **Performance Highlights**: 실험 결과, CineSync는 비디오 재구성에서 최첨단 성능을 달성하며, EEG와 fMRI 모달리티를 효과적으로 통합하여 복잡한 오디오비주얼 자극의 재구성 품질을 개선하는 데 성공했습니다. Cine-Benchmark는 재구성 품질을 종합적으로 평가하기 위한 프로토콜로 설계되었으며, 비디오와 오디오의 재구성을 더욱 엄격하게 검토합니다. 본 연구는 fMRI와 EEG 데이터의 결합을 통한 재구성 품질 향상을 최초로 탐구하며, 다중 모달 뇌 디코딩 연구에 기여합니다.



### Modeling Human Skeleton Joint Dynamics for Fall Detection (https://arxiv.org/abs/2503.06938)
Comments:
          Published in 2021 Digital Image Computing: Techniques and Applications (DICTA)

- **What's New**: 인구 고령화의 가속화는 노인을 위한 더욱 나은 돌봄과 지원 시스템의 필요성을 강조하고 있습니다. 노인의 낙상 사고는 심각한 장기 건강 문제를 야기할 수 있으며, 기존의 낙상 탐지 방법들은 사생활 보호 문제로 인해 실생활 응용에 적합하지 않았습니다. 본 논문에서는 효과적인 그래프 컨볼루션 네트워크 모델을 제안하여 인체 관절의 시공간 종속성(spatio-temporal joint dependencies)을 활용하여 정확한 낙상 탐지를 가능하게 합니다.

- **Technical Details**: 제안된 모델은 인체 관절의 동적 표현을 활용하여 운동 역학과 후속 자세 변화(transition)를 통한 낙상 탐지를 효율적으로 수행합니다. 기존의 이미지 처리 기반 모델들이 주요 관절 간의 종속성을 간과한 것과 달리, 본 연구에서는 모든 뼈대 관절을 그래프로 다루어 시공간 데이터를 분석합니다. 세 개의 대규모 데이터셋(NTU 데이터셋 포함)을 통해 성능을 평가하였으며, 모델 크기가 다른 기존 모델들에 비해 상당히 작음에도 불구하고 최첨단 결과를 달성했습니다.

- **Performance Highlights**: 실험 결과, 제안한 모델은 노인 낙상 탐지에서 우수한 성능을 보였으며, 크게 개선된 정확도를 기록했습니다. 특히, 기존의 방법들이 소규모 데이터셋에서 평가되었다는 점을 고려했을 때, 본 연구는 일상 생활 활동 120종을 포함한 대규모 데이터셋에서 높은 정확도를 입증했습니다. 이러한 성능 향상은 노인 낙상 탐지 시스템의 실용성을 크게 증가시킬 것으로 기대됩니다.



### LLaFEA: Frame-Event Complementary Fusion for Fine-Grained Spatiotemporal Understanding in LMMs (https://arxiv.org/abs/2503.06934)
- **What's New**: 이번 연구에서는 LLaFEA(대형 언어 및 프레임-이벤트 어시스턴트)를 소개합니다. 이 모델은 이벤트 카메라를 활용하여 시간적으로 밀집된 감지 및 프레임-이벤트 융합을 수행하는 새로운 접근 방식을 제안합니다. 이를 통해 언어적 및 시각적 표현 간의 증가된 정밀도가 확보되어, 대형 멀티모달 모델(LMMs)에 시간과 위치에 따른_scene 해석 능력을 크게 향상시킵니다.

- **Technical Details**: LLaFEA는 크로스 어텐션 메커니즘을 사용하여 보완적인 공간적 및 시간적 특징을 통합하고, 자기 어텐션 매칭으로 전 세계 스페이-타임 연관성을 수립합니다. 이 과정은 공간적 밀도가 높은 프레임 기반 특성과 지속적으로 변화하는 이벤트 기반 특성을 통합하는 계층적 융합 프레임워크를 적용합니다. 텍스트의 위치 및 지속 기간 토큰은 융합된 시각적 공간에 삽입되어 세밀한 정렬을 높이는 역할을 합니다.

- **Performance Highlights**: 제안하는 방법은 높은 역동성과 저조도 환경에서도 효과적으로 작동하며, 광범위한 실험을 통해 이의 효율성을 검증하였습니다. 실제 세계의 프레임-이벤트 데이터 세트를 구축하였고, 이를 통해 세밀한 시공간 이해의 개선이 이루어졌음을 보여주었습니다. LLaFEA는 다양한 위치와 시간에서 장면 컨텐츠를 이해하고 추론할 수 있는 새로운 가능성을 제공합니다.



### Post-Training Quantization for Diffusion Transformer via Hierarchical Timestep Grouping (https://arxiv.org/abs/2503.06930)
- **What's New**: Diffusion Transformer (DiT)가 이미지 생성 모델을 구축하는 데 있어서 널리 선호되는 선택이 되었습니다. 이전의 convolution 기반 UNet 모델과는 달리, DiT는 transformer 블록으로만 구성되어 있어, 큰 데이터 세트와 모델 크기를 수용할 수 있는 뛰어난 확장성을 제공합니다. 하지만 증가하는 모델 크기와 다단계 샘플링 패러다임은 배포와 추론 과정에서 상당한 압박을 초래합니다.

- **Technical Details**: 이 논문에서는 Diffusion Transformer의 도전 과제를 해결하기 위한 후처리 양자화(post-training quantization, PTQ) 프레임워크를 제안합니다. 먼저, 시간 의존성이 있는 채널 특정 이상값(outliers)으로 인해 양자화가 어렵다는 점을 확인했습니다. 이 문제를 해결하기 위해, 우리는 시간 단계에 인식한 shift-and-scale 전략을 통해 활성화 분포를 부드럽게 하여 양자화 오류를 줄입니다.

- **Performance Highlights**: 종합적인 실험 결과, 제안된 PTQ 방법은 Diffusion Transformer 모델을 8비트 가중치 및 8비트 활성화(W8A8)로 성공적으로 양자화하며, 최첨단 FiD 점수를 기록했습니다. 또한, 우리의 방법은 생성 품질을 저하시키지 않고 DiT 모델을 4비트 가중치 및 8비트 활성화(W4A8)로 추가로 양자화할 수 있습니다.



### From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers (https://arxiv.org/abs/2503.06923)
Comments:
          13 pages, 14 figures

- **What's New**: 본 논문에서는 기존의 'cache-then-reuse' 방식을 대체하고 'cache-then-forecast'라는 새로운 패러다임을 제안합니다. 이를 통해 변화하는 시점에서의 특징을 예측할 수 있는 방법론을 개발하여, 지속적인 모델링을 통해 다가오는 시점에서의 diffusion 모델 특징을 예측할 수 있게 합니다. 이 접근법은 특징의 안정적인 궤적을 활용하여 높은 비율의 가속화를 가능하게 하며, 기존 캐싱 방식을 넘어서는 성능을 자랑합니다.

- **Technical Details**: TaylorSeer는 Taylor 급수를 이용하여 다양한 시점에서의 특징 궤적을 근사하고, 고차 도함수를 활용하여 차별화된 속도로 특징을 예측합니다. 이 방식은 캐싱된 특징을 단순히 재사용하는 것과는 달리, 특징의 변화 연속성을 활용하여 미래의 특징을 예측하는데 중점을 둡니다. 결과적으로, 추가 훈련 없이도 높은 가속비와 함께 효율적인 소스 재사용이 가능해짐을 보여줍니다.

- **Performance Highlights**: TaylorSeer는 DiT, FLUX, HunyuanVideo에서 각각 2.5배, 4.99배, 5.00배의 가속화를 달성하며, 높은 품질의 생성을 유지합니다. 성능 지표에서도 이전 상태에서 기술(SOTA)보다 36배의 품질 손실 감소를 보여주며, 6배 이상의 가속 환경에서도 효과적으로 작동합니다. 이로써 유의미한 성능 향상을 이루며, diffusion 모델의 새로운 가속 방법론으로 자리매김할 수 있습니다.



### When Lighting Deceives: Exposing Vision-Language Models' Illumination Vulnerability Through Illumination Transformation Attack (https://arxiv.org/abs/2503.06903)
- **What's New**: 이 논문에서는 비전-언어 모델(VLM)에 대한 조명 변화에 대한 강건성을 체계적으로 평가하는 최초의 프레임워크인 조명 변환 공격(Illumination Transformation Attack, ITA)을 제안합니다. 기존 연구는 조명의 영향을 탐구했지만, 주로 CNN 기반 모델에 한정되었고, VLM에 적합하지 않았습니다. ITA는 조명의 다변화를 실현하기 위해 조명의 렌더링 방정식을 기반으로 전역 조명을 여러 파라미터화된 점 광원으로 분리하는 혁신적인 접근 방식을 채택하였습니다.

- **Technical Details**: ITA는 두 가지 주요 구성 요소로 나뉘며, 첫 번째는 조명 모델링을 위한 정밀 조정입니다. 이를 통해 각 광원의 기여를 선형 합으로 표현하여 복잡한 조명 변화를 정확하게 모델링할 수 있습니다. 두 번째는 조명 재구성 모델의 잠재 공간을 통해 자연스러운 외관을 유지하며 생성되는 이미지를 제어합니다. 이를 통해 물리적 조명 우선 순위를 보존하고 원본 이미지와의 시각적 일관성을 유지하기 위한 추가적인 지각적 제약을 구현합니다.

- **Performance Highlights**: 광범위한 실험을 통해 ITA가 고급 VLM의 성능을 상당히 저하시키는 것을 입증하였습니다. LLaVA-1.6과 같은 모델에서 조명을 최적화하여 생성된 조명 인식 적대적 예제가 조명에 대한 민감성을 성공적으로 노출시킵니다. 이러한 연구 결과는 VLM의 조명 취약성을 드러내며, 더욱 현실적인 조명 조건에서도 비슷한 성능 저하를 보였습니다.



### Iterative Prompt Relocation for Distribution-Adaptive Visual Prompt Tuning (https://arxiv.org/abs/2503.06901)
- **What's New**: 이 논문에서는 시각적 프롬프트 튜닝(Visual Prompt Tuning, VPT) 기법의 성능을 향상시키기 위해 적응형 분포 최적화(adaptive distribution optimization, ADO)를 탐구합니다. 기존의 방법들이 고정된 프롬프트 분포를 사용한 것과는 달리, 저자들은 각 작업의 특성에 따라 분포를 동적으로 조절해야 함을 강조합니다. 새로운 프레임워크인 PRO-VPT(Iterative Prompt RelOcation-based VPT)를 제안하여, 프롬프트의 최적 분포를 반복적으로 학습하며 성능을 향상시키는 방법을 제시합니다.

- **Technical Details**: PRO-VPT는 두 가지 중요한 최적화 단계로 구성된 프롬프트 리로케이션(prompt relocation, PR) 전략을 개발하여, 비활성 프롬프트를 식별하고 제거한 다음 최적의 블록에 재배치합니다. 이 과정을 반복적으로 수행함으로써, 각 작업에 적합한 프롬프트 분포를 적응적으로 학습하여 VPT의 잠재력을 최대한 활용할 수 있습니다. 또한, ADO의 정의를 명확히 하고, 이 정의에 기초한 더 나은 프롬프트 분포 최적화를 위한 전략을 제안합니다.

- **Performance Highlights**: PRO-VPT는 VTAB-1k 벤치마크에서 VPT 기법 대비 평균 정확도가 1.6% 향상된 78.0%의 최첨단 성능을 달성했습니다. 이 연구는 프롬프트 기반 방법들이 보다 높은 성능을 달성할 수 있음을 보여주는 중요한 결과를 제공합니다. 실험을 통해 제안된 방법이 기존의 최신 VPT 방법들을 능가함을 입증하였습니다.



### DirectTriGS: Triplane-based Gaussian Splatting Field Representation for 3D Generation (https://arxiv.org/abs/2503.06900)
Comments:
          Accepted by CVPR 2025

- **What's New**: DirectTriGS 프레임워크는 Gaussian Splatting(GS)을 활용한 3D 객체 생성의 새로운 접근 방식을 제안합니다. 기존의 generative modeling 방법과 달리, GS를 직접 생성하는 데 대한 연구는 부족했습니다. 본 논문에서는 GS의 복잡한 데이터 구조를 극복하기 위해 triplane 표현을 사용하여 GS를 이미지와 유사한 연속 필드로 나타내는 방식을 소개합니다.

- **Technical Details**: TriRenderer를 통해 우리는 GS의 기하학적 정보와 텍스처 정보를 효과적으로 인코딩하고, 분리된 경로로 압축할 수 있습니다. 우리 연구에서는 Variational Autoencoder(VAE)를 사용해 triplane 표현을 잠재 공간(latent space)으로 변환하여, 3D 객체를 생성하기 위한 latent diffusion을 적용합니다. 또한, TriRenderer는 전적으로 미분 가능(differentiable)하여 2D 감독(supervision)만을 사용해도 3D GS의 텍스처와 기하학을 다룰 수 있습니다.

- **Performance Highlights**: 우리의 실험 결과, 제안된 DirectTriGS는 텍스트 기반의 3D 생성(task)에서 높은 품질의 3D 객체 기하학과 렌더링 결과를 보여주었습니다. 특히, 메모리 효율성과 다양한 GS 생성 가능성을 가지고 있으며, 기존의 sparse GS point cloud에 비해 convolution 기반 인코더와 더 호환됩니다. 결과적으로, DirectTriGS는 3D 생성 분야에서 경쟁력 있는 성능을 입증하였습니다.



### Illuminating Darkness: Enhancing Real-world Low-light Scenes with Smartphone Images (https://arxiv.org/abs/2503.06898)
- **What's New**: 본 연구에서는 저조도 이미지를 개선하기 위해 대규모 고해상도 저조도 개선(SLLIE) 데이터셋을 제안합니다. 이 데이터셋은 6,425개의 고유한 초점 정렬 이미지 쌍으로 구성되어 있으며, 0.1에서 200 루멘까지 다양한 조명 조건에서 스마트폰 센서로 촬영되었습니다. 또한, 본 연구에서는 실제 저조도 이미지를 개선하기 위해 독특한 구조를 가진 Transformer-CNN 모델인 TFFormer를 도입했습니다.

- **Technical Details**: TFFormer는 조도와 색상 정보를 별도로 학습하는 tuning fork 형태의 모델로, 기존 Retinex 기반 네트워크와 달리 단일 네트워크 내에서 복잡한 장면의 노이즈 감소 및 과도한 개선을 다룹니다. 또한 Luminance-Chrominance Cross-Attention Block(LC CAB)과 Luminance-Chrominance Guided Refinement Block(LC GRB)을 포함하여 이미지 재구성을 더욱 향상하고 시각적으로 일관된 결과를 만들기 위한 Luminance-Chrominance Perceptual Loss(LC PL)을 도입했습니다.

- **Performance Highlights**: TFFormer는 다양한 하드웨어와 실제 상황에서 효과성을 입증했습니다. 또한, 우리 방법은 다수의 사용자 연구 및 정량적 비교를 통해 기존 저조도 이미지 개선 기술보다 월등한 성능을 보였습니다. 이러한 결과는 다양한 비전 작업(vision tasks)에서 효율성을 더욱 높일 수 있는 가능성을 제시합니다.



### HiSTF Mamba: Hierarchical Spatiotemporal Fusion with Multi-Granular Body-Spatial Modeling for High-Fidelity Text-to-Motion Generation (https://arxiv.org/abs/2503.06897)
Comments:
          11pages,3figures,

- **What's New**: HiSTF Mamba는 텍스트-모션 생성(model)에서의 혁신적인 접근법으로, 기능 중복을 줄이고 섬세한 조인트 수준의 세부사항을 고려하여 spatio-temporal 특성을 효과적으로 융합하는 것을 목표로 합니다. 이 프레임워크는 Dual-Spatial Mamba, Bi-Temporal Mamba, Dynamic Spatiotemporal Fusion Module (DSFM)와 같은 세 가지 주요 모듈로 구성되어 있습니다. HiSTF Mamba는 자연어 설명에 기반하여 인간의 움직임을 생성하는 데 필요한 유연성과 효율성을 제공합니다.

- **Technical Details**: Dual-Spatial Mamba는 'Part-based + Whole-based' 평행 모델링을 통해 전체 신체 협조와 세밀한 관절 역학을 동시에 표현합니다. Bi-Temporal Mamba는 양방향 스캔 전략을 채택하여 단기 움직임 디테일과 장기 의존성을 효과적으로 인코딩합니다. DSFM은 중복 제거 및 보완 정보 추출을 수행하여 공간 및 시간 정보를 통합함으로써 더욱 표현력 있는 spatio-temporal 표현을 생성합니다.

- **Performance Highlights**: HumanML3D 데이터셋에 대한 실험 결과, HiSTF Mamba는 여러 평가 지표에서 최첨단 성능을 달성하며 FID(Fréchet Inception Distance) 점수를 0.283에서 0.189로 약 30% 감소시키는 데 성공했습니다. 이러한 결과는 텍스트와의 높은 의미 일치와 사실감 있는 고해상도 모션 생성을 가능하게 하여 HiSTF Mamba의 효과성을 입증합니다.



### CATANet: Efficient Content-Aware Token Aggregation for Lightweight Image Super-Resolution (https://arxiv.org/abs/2503.06896)
Comments:
          Accepted by CVPR2025

- **What's New**: 이 논문에서는 기존의 클러스터링 기법의 한계를 극복하기 위해 경량화된 Content-Aware Token Aggregation Network(CATANet)를 제안합니다. CATANet은 효율적인 Content-Aware Token Aggregation 모듈을 통해 로컬 및 글로벌 정보 상호작용을 확대하고, 토큰 간의 긴 의존성을 효과적으로 캡처합니다. 이는 기존의 SR 방법들과 비교하여 최대 0.33dB의 PSNR 개선과 함께 추론 속도를 거의 두 배로 증가시키는 성과를 나타냅니다.

- **Technical Details**: CATANet의 핵심 구성 요소인 Token-Aggregation Block은 Content-Aware Token Aggregation 모듈과 Intra-Group Self-Attention, Inter-Group Cross-Attention으로 구성됩니다. CATA 모듈은 훈련 단계에서만 토큰 센터를 업데이트하여 모델의 추론 속도에 미치는 영향을 최소화합니다. 이를 통해 CATANet은 더 섬세한 길고 지역적 상호작용을 가능하게 하여 긴 의존성을 효과적으로 끌어낼 수 있습니다.

- **Performance Highlights**: CATANet은 기존의 클러스터링 기반 경량화 SR 방법인 SPIN과 비교했을 때, 최대 0.33dB의 PSNR 개선을 보였으며, 추론 속도도 거의 두 배 더 빠릅니다. 이러한 성능 향상은 CATANet의 효율적인 토큰 집합 및 세밀한 주의 메커니즘을 통해 달성되었습니다. 실험 결과는 CATANet이 경량 모델의 적용 가능성을 높이는 데 기여함을 보여줍니다.



### Improving cognitive diagnostics in pathology: a deep learning approach for augmenting perceptional understanding of histopathology images (https://arxiv.org/abs/2503.06894)
- **What's New**: 이번 논문은 Digital Technologies가 Computational-Pathology 분야에서 인간의 건강, 인지, 및 인식을 증진시킬 수 있는 새로운 접근방법을 제시합니다. Vision Transformers (Vit)와 GPT-2를 결합한 멀티모달 모델을 통해 Histopathology 이미지 분석을 향상시키는 것을 목표로 하고 있습니다. 이 모델은 의료 및 학술 자원에서 파생된 Dense 이미지 캡션으로 전문화된 Arch-Dataset을 기반으로 세밀하게 조정되었습니다.

- **Technical Details**: 모델은 조직 형태, 염색 변이, 병리적 상태와 같은 Histopathology 이미지의 복잡성을 포착하기 위해 설계되었습니다. 컨텍스트에 맞는 정확한 캡션을 생성함으로써, 의료 전문가의 인지 능력을 증대시켜 질병 분류(classification), 분할(segmentation), 및 분별(detection)을 더 효율적으로 수행할 수 있도록 합니다.

- **Performance Highlights**: 이 모델은 종종 간과되는 미세한 병리적 특징들을 인지하는 능력을 향상시켜 진단 정확성을 개선합니다. 본 연구는 디지털 기술이 의료 이미지 분석에서 인간의 인지 능력을 증진시킬 수 있는 가능성을 입증하고, 보다 개인화되고 정확한 의료 결과를 향한 발걸음을 보여줍니다.



### Accessing the Effect of Phyllotaxy and Planting Density on Light Use Efficiency in Field-Grown Maize using 3D Reconstructions (https://arxiv.org/abs/2503.06887)
Comments:
          17 pages, 8 figures

- **What's New**: 이 연구는 옥수수의 생장 밀도 증가로 인한 햇빛 차단 문제를 해결하기 위해, 3D 재구성과 photosynthetically active radiation (PAR) 모델링을 통합한 새로운 프레임워크를 제안합니다. 이 프레임워크는 다양한 옥수수 유전자형의 가상 필드를 구축하고 필드 측정값과 validation을 통해 그 효과를 측정합니다. 이를 통해 옥수수에서 캐노피 구조와 라이트 캡쳐의 관계를 이해하는 데 중요한 통찰을 제공합니다.

- **Technical Details**: 연구에서는 LiDAR 스캐너를 이용하여 옥수수 필드에서 포인트 클라우드를 캡쳐하고, 이를 통해 정확한 3D 모델을 복원합니다. 이어서 다양한 심기 밀도, 줄 방향 및 잎의 방위 각도에 대한 PAR 차단 분석을 수행합니다. LiDAR와 태양 복사 시뮬레이션을 결합한 이 프레임워크는 다양한 밀도와 심기 조건 하에서 PAR 차단 dynamics에 대한 깊은 분석을 제공합니다.

- **Performance Highlights**: 연구 결과, 심기 밀도와 캐노피 방향에 따라 빛 차단 효율에서 유의미한 변화를 보였습니다. 특히, 높은 밀도의 심기에서 주변 식물의 음영으로 인한 수확량 감소가 관찰되었습니다. 캐노피 건축학과 빛 포착의 관계를 이해함으로써, 이 연구는 다양하고 복잡한 농업 환경에서 옥수수 육종 및 재배 전략을 최적화하는 데 중요한 지침을 제공합니다.



### ProBench: Judging Multimodal Foundation Models on Open-ended Multi-domain Expert Tasks (https://arxiv.org/abs/2503.06885)
- **What's New**: 본 연구에서는 전문적인 전문성을 요구하는 오픈 엔디드 사용자 쿼리의 벤치마크인 ProBench를 도입합니다. ProBench는 사용자의 일상 생산성 요구에 따라 전문가들이 독립적으로 제출한 4,000개의 고품질 샘플로 구성되어 있습니다. 이 벤치마크는 과학, 예술, 인문학, 코딩, 수학, 창의적 글쓰기 등 10개 분야와 56개 하위 분야에 걸쳐 있습니다.

- **Technical Details**: ProBench는 MLLM(as-a-Judge) 시스템을 이용하여 24개의 최신 모델을 평가하고 비교합니다. 실험적으로, 이 모델들은 다양한 과제를 수행하는 데 있어 시각적 인식, 텍스트 이해, 도메인 지식 및 고급 추론 분야에서 상당한 도전 과제를 제시합니다. 이 연구는 다중 모달(AI) 모델의 평가를 위한 새로운 기준을 마련하고 있습니다.

- **Performance Highlights**: 결과적으로, 최고의 오픈 소스 모델들은 상용 모델들과 대등한 성능을 보이고 있지만, ProBench의 어려움은 여전히 시각적 인식과 텍스트 이해, 도메인 지식, 고급 추론에서 도전적인 과제를 남겨두고 있습니다. 이러한 발견은 앞으로의 다중 모달 AI 연구 노력에 귀중한 방향성을 제공하고 있습니다.



### Text-to-Image Diffusion Models Cannot Count, and Prompt Refinement Cannot Help (https://arxiv.org/abs/2503.06884)
- **What's New**: 이번 논문에서는 AI 커뮤니티에서 가장 중요한 문제 중 하나인 Generative modeling에 대해 다루고, text-to-image generation에서 diffusion models의 성공 사례를 소개합니다. 하지만 이 모델들이 사용자 지침의 숫자 제약을 준수하는 데 있어 근본적인 제한이 있음을 강조하고, 이에 대한 체계적인 평가가 부족하다고 지적합니다. 이러한 문제를 해결하기 위해 T2ICountBench라는 새로운 벤치마크를 제시하여, 최신 diffusion 모델의 카운팅 능력을 rigorously 평가하고자 합니다.

- **Technical Details**: T2ICountBench는 다양한 generative models를 포함하며, 이 모델들은 오픈 소스와 프라이빗 시스템을 포괄합니다. 이 벤치마크는 카운팅 성능을 다른 기능에서 분리하고, 다양한 난이도 수준을 제공하여 구조적으로 평가할 수 있게 설계되었습니다. 또한, 정확도를 높이기 위해 인적 평가(human evaluations)를 포함하여 높은 신뢰성을 보장합니다.

- **Performance Highlights**: T2ICountBench를 통한 평가 결과, 모든 최신 diffusion 모델들이 올바른 객체 수를 생성하는 데 실패하며, 객체의 수가 증가함에 따라 정확도가 현저히 감소하는 것으로 나타났습니다. 추가적으로, 프롬프트 개선(prompt refinement)에 대한 탐색적 연구 결과, 이러한 간단한 개입이 카운팅 정확도를 향상시키는 데 generally 효과적이지 않음을 보여주었습니다. 이러한 발견은 diffusion 모델 내에서 수치 이해에 대한 고유한 도전 과제를 강조하고, 향후 개선 방향에 대한 유망한 실마리를 제공합니다.



### Interactive Medical Image Analysis with Concept-based Similarity Reasoning (https://arxiv.org/abs/2503.06873)
Comments:
          Accepted CVPR2025

- **What's New**: 이번 논문에서는 Concept-based Similarity Reasoning 네트워크(CSR)를 소개합니다. 이 네트워크는 모델의 의사결정을 심층적으로 이해하고 중재할 수 있도록 설계되었습니다. CSR은 패치 수준의 프로토타입과 내재적인 개념 해석을 제공하며, 공간적인 상호작용을 통해 의사가 이미지의 특정 영역에 직접 개입할 수 있도록 합니다. 이를 통해 의료 이미지 분석에서 투명하고 직관적인 도구로 활용될 수 있습니다.

- **Technical Details**: CSR 모델은 (i) 설명 가능한 개념 특성을 추출하기 위한 Concept 모델, (ii) 개념 특성 공간을 강화하는 피쳐 프로젝터, (iii) 개념 유사성 점수를 기반으로 분류하는 과제 헤드로 구성됩니다. 입력 이미지를 받아들여 개념 점수를 예측하는 대신, CSR은 주어진 이미지와 각 개념의 프로토타입 간의 유사성 점수를 계산합니다. 이러한 유사성 점수를 활용하여 최종 분류를 지원하는 구조입니다.

- **Performance Highlights**: CSR은 세 가지 생물 의학 데이터셋에서 기존의 해석 가능한 방법들 대비 최대 4.5%의 성능 향상을 보여주었습니다. 특히, 의사들이 이미지 분석에 필요한 일관성 있는 피드백을 통해 모델의 신뢰성을 높이는 데 기여할 수 있는 기능을 강조합니다. 이러한 성능 향상은 의료 이미지 분석의 해석 가능성을 크게 개선하는 데 기여할 것으로 기대됩니다.



### Towards Generalization of Tactile Image Generation: Reference-Free Evaluation in a Leakage-Free Setting (https://arxiv.org/abs/2503.06860)
- **What's New**: 이번 연구에서는 신뢰할 수 있는 평가 프로토콜과 새로운 평가지표(TMMD, I-TMMD, CI-TMMD, D-TMMD)를 제안하여 촉각 이미지 생성에서의 데이터 누수를 해결하였습니다. 기존 데이터셋에서 훈련 및 테스트 샘플의 중복 문제로 인해 성능 지표가 부풀려지는 현상을 분석하고, 이를 해결하기 위한 효과적인 방법론을 개발했습니다. 또한, 시각 데이터를 촉각 이미지로 변환하기 위해 텍스트 설명을 중간 매개변수로 활용하는 방법을 제안하였습니다.

- **Technical Details**: 이 연구는 시각-촉각 생성(task of vision-to-touch generation)을 위한 프레임워크를 제안하며, 감지된 촉각 속성에 대한 간결한 텍스트 설명을 훈련 과정에 포함시킵니다. 본 접근법은 촉각 특징을 더 효과적으로 캡처하고, 생성된 촉각 이미지의 품질을 개선하는 데 기여합니다. 이 과정에서 우리는 데이터 누수를 견제하기 위해 흡수된 데이터셋을 분석하고, 새로운 참고 없이 평갈 방법을 소개하였습니다.

- **Performance Highlights**: 우리가 제안한 방법은 Touch and Go와 HCT라는 두 가지 인기 있는 데이터셋에서 진행된 실험에서 우수한 성능을 보이며, 보다 강건한 일반화를 보여주었습니다. 특히 데이터 누수가 없는 설정에서 평가하여 실질적인 성능을 반영하였고, 이는 실제 환경에서의 적용 가능성이 높음을 의미합니다. 이 결과들은 우리의 방법이 촉각 데이터 생성 및 평가의 새로운 기준을 제시하고 있음을 시사합니다.



### ActiveInitSplat: How Active Image Selection Helps Gaussian Splatting (https://arxiv.org/abs/2503.06859)
- **What's New**: 본 논문에서는 Gaussian splatting (GS)의 훈련 이미지 선택 문제를 해결하기 위한 새로운 프레임워크인 'ActiveInitSplat'를 제안합니다. 기존의 GS는 수동적으로 선택된 2D 이미지를 기반으로 했던 반면, ActiveInitSplat는 다양한 시점에서의 훈련 이미지 선택을 통해 더 나은 장면 커버리지를 확보하고 3D 구조와 잘 정렬된 초기화된 가우시안 함수를 생성하는 데 중점을 둡니다. 이를 통해 GS 렌더링 성능이 크게 향상됩니다.

- **Technical Details**: ActiveInitSplat는density(밀도) 및 occupancy(점유율) 기준을 활용하여 훈련 이미지의 적극적인 선택을 위한 혁신적인 최적화 공식을 제안합니다. 이러한 접근 방식은 GS의 초기화 단계에서 밀접하게 관련되며, 이후의 GS 단계와 독립적입니다. 이는 적은 수의 이미지로도 효과적인 GS 렌더링을 보장하며, GS 아키텍처의 확장된 변형에 쉽게 통합될 수 있습니다.

- **Performance Highlights**: Numerical tests를 통해 ActiveInitSplat가 기존 수동 선택 방식에 비해 현저한 렌더링 성능 향상을 가져온다는 점이 입증되었습니다. 독립된 3D 장면 테스트에서, ActiveInitSplat는 정보가 풍부한 훈련 이미지를 다양한 시점에서 능동적으로 포착함으로써 효과적인 성능을 나타냅니다. 실험은 저밀도 및 고밀도 뷰 모두에서 우수한 결과를 보였으며, 3D 공간 내 어떤 위치에서도 최적의 이미지를 선택할 수 있는 기능을 갖추었습니다.



### From Image- to Pixel-level: Label-efficient Hyperspectral Image Reconstruction (https://arxiv.org/abs/2503.06852)
- **What's New**: 현재 다루고 있는 논문에서는 Pixel-Level Spectral Super-Resolution (Pixel-SSR)라는 새로운 패러다임을 제안합니다. 이 접근 방식은 RGB 이미지와 포인트 스펙트럼을 기반으로 HSI를 재구성하여 정확성과 효율성을 동시에 고려합니다. 특히, 이 방법은 새로운 장면에서의 일반화와 효과적인 정보 추출을 통해 재구성 정확도를 높이는 두 가지 주요 도전에 대응하고자 합니다.

- **Technical Details**: Pixel-SSR는 포인트 스펙트럼의 내재적 특성을 고려하여 Gamma 분포를 모델링합니다. 이를 통해 새로운 장면에서도 HSI 재구성이 가능하게 하며, Dynamic Prompt Mamba (DyPro-Mamba)를 활용하여 RGB와 포인트 스펙트럼에서 상호보완적인 정보를 추출합니다. 각각의 프롬프트 구조는 전반적인 spatial distribution과 edge details, spectral dependencies를 캡처하는 데 기여합니다.

- **Performance Highlights**: 논문에서는 다양한 검증 지표를 활용하여 Pixel-SSR 방법이 기존의 최신 기술들과 비교해 경쟁력 있는 재구성 정확도를 보인다고 설명합니다. 특히, 이 방법은 레이블 소비를 효율적으로 유지하면서도 강력한 성능을 보여주어, 여러 벤치마크에서 실험을 통해 균형 잡힌 성능을 입증합니다.



### MADS: Multi-Attribute Document Supervision for Zero-Shot Image Classification (https://arxiv.org/abs/2503.06847)
- **What's New**: 이 논문에서는 다중 속성(document) 지도를 지원하는 새로운 프레임워크인 MADS를 제안합니다. 이 프레임워크는 문서 수집 및 모델 학습 단계에서 노이즈를 제거하여 더 깨끗한 보조 정보를 제공합니다. 특히, 대형 언어 모델을 활용하여 비시각적 설명을 자동으로 제거하고 다양한 속성 관점에서 문서를 풍부하게 만듭니다. 이로 인해 지식 이전 과정이 개선되어, 모델의 성능과 해석 가능성을 높입니다.

- **Technical Details**: MADS 프레임워크는 대형 언어 모델의 세계 지식을 활용하여 노이즈를 제거하는 혁신적인 프롬프트 알고리즘을 도입합니다. 이 알고리즘은 시각적 속성 관점을 기반으로 문서를 섹션으로 나누고, 비시각적 단어에 대한 주의를 줄이는 모델 불가지론적 초점 손실(focus loss)을 사용합니다. 또한, 다중 속성 문서에서 전이 가능한 지식을 추출하고, 이를 통해 이미지와 문서 간의 의미적 정렬을 달성하는 새로운 네트워크 구조를 제시합니다.

- **Performance Highlights**: MADS는 기존의 SOTA(State-Of-The-Art) 대비 평균적으로 7.2%에서 8.2% 개선된 성능을 보이며, 문서 기반 ZSL 설정 및 GZSL 설정에 대한 세 가지 벤치마크에서 검증되었습니다. 이 모델은 훈련 중 비시각적 정보를 명시적으로 감지하여, 모델의 주의를 향상시킴으로써, 이전 방법들보다 성능 향상을 이루었습니다. 또한, 프레임워크는 다양한 속성 관점에서 해석 가능한 예측 결과를 제공합니다.



### Improving Visual Place Recognition with Sequence-Matching Receptiveness Prediction (https://arxiv.org/abs/2503.06840)
Comments:
          8 pages, 5 figures, under review

- **What's New**: 본 논문에서는 비주얼 장소 인식(Visual Place Recognition, VPR) 기술을 위한 새로운 감독 학습 방법을 제안합니다. 이 방법은 프레임별 시퀀스 매칭 수용도(sequence matching receptiveness, SMR)를 예측하여 시스템이 언제 시퀀스 매칭의 출력을 신뢰할지를 선택적으로 결정하도록 합니다. 이 접근법은 여러 최신 VPR 기법에 대해 광범위한 성능 개선 효과를 나타내며, 안정적인 VPR 시스템 구축에 기여합니다.

- **Technical Details**: 새로운 SMR 예측기는 VPR 기술의 기본 원리에 영향을 받지 않으며, 다양한 조건에서 시퀀스 매칭의 성능 기여를 평가합니다. 이를 위해 거리 행렬(distance matrix)을 구성하고, 다층 퍼셉트론(Multi-layer Perceptron, MLP)을 이용해 SMR을 분류하는 방식으로 동작합니다. 이를 통해 예측된 결과는 잘못된 예측이 수정되거나, 반대로 정확한 예측이 부정확한 것으로 전환될 가능성을 포함합니다.

- **Performance Highlights**: 제안된 방식은 CosPlace, MixVPR, EigenPlaces, SALAD, AP-GeM, NetVLAD 및 SAD 등 다양한 최신 VPR 기법을 통해 실험적으로 검증되었습니다. 여러 벤치마크 데이터셋(Nordland, Oxford RobotCar, SFU-Mountain)에서 VPR 성능이 상당히 향상되었으며, SMR 예측기가 평균 성능을 크게 개선하는 결과를 보여주었습니다. 이러한 결과는 시퀀스 길이의 변화와 SMR 예측기의 상호작용에 대한 통찰력도 제공합니다.



### AttFC: Attention Fully-Connected Layer for Large-Scale Face Recognition with One GPU (https://arxiv.org/abs/2503.06839)
- **What's New**: 본 논문에서는 Attention Fully Connected (AttFC) 레이어를 제안하여 얼굴 인식(FL) 모델의 훈련에 필요한 계산 자원을 대폭 줄였습니다. AttFC는 생성적 클래스 센터(Generative Class Center, GCC)를 생성하기 위해 attention loader를 활용하고 역동적인 클래스 컨테이너(Dynamic Class Container, DCC)를 통해 클래스를 저장합니다. 여기서 DCC는 FC 레이어의 모든 클래스 센터 중 일부만 저장하기 때문에, 모델 파라미터 수가 대幅적으로 감소합니다.

- **Technical Details**: AttFC는 attention 메커니즘과 Momentum Contrast를 바탕으로 설계되었으며, 두 개의 인코더(Feature Encoder와 Class Encoder)를 사용하여 GCC 생성을 위한 다양한 이미지의 기여도를 계산합니다. 인식 품질에 따라 주의 가중치(attention weight)를 적용함으로써 저품질 이미지의 영향을 축소하고 고품질 이미지의 중요성을 높입니다. 결과적으로 업데이트된 GCC는 실제 클래스 센터(True Class Center, TCC)와 더욱 비슷해집니다.

- **Performance Highlights**: 실험 결과, AttFC를 활용했을 때 FD 모델은 기존의 최신 방법들과 유사한 성능을 보이면서도 훈련 자원을 크게 절감할 수 있음을 입증했습니다. 이는 대형 데이터셋을 사용할 때 계산 시간과 메모리 요구 사항을 줄여주어 하드웨어 자원의 부담을 경감시킵니다. 따라서 AttFC는 얼굴 인식에 있어 효율적인 모델 훈련을 가능하게 합니다.



### GUIDE-CoT: Goal-driven and User-Informed Dynamic Estimation for Pedestrian Trajectory using Chain-of-Though (https://arxiv.org/abs/2503.06832)
Comments:
          10 pages, 5 figures, will be published on The 24th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2025)

- **What's New**: 본 논문에서는 보행자 궤적 예측을 위한 새로운 방법, GUIDE-CoT(Goal-driven and User-Informed Dynamic Estimation using Chain-of-Thought)를 제안합니다. 이 접근법은 시각적 정보를 효과적으로 활용하는 목표 지향적 비주얼 프롬프트와 궤적 생성을 위한 Chain-of-Thought(CoT) LLM을 결합하여 실현됩니다. 이러한 혁신적인 모듈은 예측 정확성을 높이고 사용자가 궤적을 조정할 수 있는 조작 가능성을 제공합니다.

- **Technical Details**: 이 방법은 두 개의 주요 모듈로 구성됩니다: (1) 목표 지향적 비주얼 프롬프트는 사전 훈련된 시각 인코더와 결합하여 목표 예측 정확성을 향상시킵니다. (2) 체인 오브 생각(CoT) LLM은 예측된 목표를 바탕으로 보다 현실적인 궤적을 생성합니다. 이를 통해 시각적 및 공간적 정보를 모두 활용하면서 보다 정밀한 궤적 예측이 가능해집니다.

- **Performance Highlights**: ETH/UCY 벤치마크 데이터세트를 통한 실험 결과, 제안한 방법은 최신 기술 대비 뛰어난 성능을 보이며, 궤적 예측의 정확성과 적응성을 크게 개선합니다. 실험은 또한 사용자가 궤적을 제어할 수 있는 새로운 기능을 도입하여 동적인 궤적 조정을 가능하게 합니다.



### HierDAMap: Towards Universal Domain Adaptive BEV Mapping via Hierarchical Perspective Priors (https://arxiv.org/abs/2503.06821)
Comments:
          The source code will be made publicly available at this https URL

- **What's New**: 본 논문에서는 BEV (Bird's-Eye View) 지도 기법의 발전을 위해 HierDAMap이라는 계층적 관점 우선 지식 기반의 도메인 적응 프레임워크를 제안합니다. 기존 연구는 주로 이미지 수준의 학습에 집중하였지만, 본 연구는 글로벌, 스파스 그리고 인스턴스 수준에서 관점 우선 지식의 가이드를 탐색합니다. 이 프레임워크는 세 가지 주요 컴포넌트를 포함하며, 각 단계에서 효과적으로 BEV 도메인 적응을 달성하기 위한 접근을 제공합니다.

- **Technical Details**: HierDAMap은 Semantic-Guided Pseudo Supervision (SGPS), Dynamic-Aware Coherence Learning (DACL), Cross-Domain Frustum Mixing (CDFM) 등의 세 가지 모듈로 구성되어 있습니다. SGPS는 2D 공간에서 생성된 pseudo labels를 통해 교차 도메인 일관성을 확보하고, DACL은 불확실성을 고려하여 BEV 레이블을 동적으로 유도합니다. CDFM은 서로 다른 도메인에서 관점 마스크를 활용하여 다중 관점 이미지를 혼합, 학습하는 데 도움을 주며, 이를 통해 교차 도메인 뷰 변환 및 인코딩 학습을 안내합니다.

- **Performance Highlights**: 제안된 방법은 nuScenes 및 Argoverse 데이터 세트를 사용하여 여러 BEV 매핑 작업에서 검증되었으며, 다양한 실험 환경에서 이전 도메인 적응 모델들을 초월하는 성능을 보였습니다. 연구 결과는 고급 BEV 매핑 작업에서 상태-of-아트 성능을 달성함을 보여줍니다. 연구진은 HierDAMap을 통해 다양한 BEV 매핑 과제에 적응할 수 있는 보편적인 도메인 적응 프레임워크를 제시하며, 이는 실제 응용에서도 유망한 방법이 될 것입니다.



### Towards Fine-Grained Video Question Answering (https://arxiv.org/abs/2503.06820)
- **What's New**: MOMA-QA 데이터셋이 소개되었으며, 이는 Video Question Answering (VideoQA)의 현재 한계를 극복하기 위해 설계되었습니다. 이 데이터셋은 시간적 로컬라이제이션(temporal localization)과 공간적 관계(spatial relationship) 추론을 강조하여 보다 정교한 비디오 이해 모델 개발을 목표로 합니다. 또한, SGVLM이라는 새로운 비디오-언어 모델이 소개되어 더 나은 관계 이해 및 시간을 기반으로 한 로컬라이제이션 능력을 제공합니다.

- **Technical Details**: MOMA-QA 데이터셋은 시간 간격 주석과 진실 장면 그래프(ground truth scene graphs)를 포함하고 있어 비디오의 특정 순간과 객체 간의 관계를 이해하는 데 필요한 데이터의 깊이를 제공합니다. 데이터셋의 질문 중 71.6%는 공간적 관계 이해를 요구하며, 각각의 프레임에 대한 장면 그래프 주석이 포함되어 있습니다. SGVLM 모델은 모티프 기반(scene graph)의 장면 그래프 생성기, 효율적인 프레임 검색기, 사전 학습된 대규모 언어 모델을 통합하여 고급 추론 능력을 가지게 됩니다.

- **Performance Highlights**: MOMA-QA 및 기타 공개 데이터셋에 대한 평가 결과, SGVLM 모델이 비디오 질문 응답에서 우수한 성능을 보이며 새로운 기준을 설정했습니다. 특히, 이 모델은 비디오의 시간적 및 공간적 관계를 효과적으로 처리가능하며, 모델의 예측 결정 경로를 해석할 수 있는 데에도 기여합니다. 데이터셋의 새로운 주석 방식과 모델의 혁신이 결합되어 더 정교하고 신뢰할 수 있는 VideoQA 성능을 제공합니다.



### Sub-Image Recapture for Multi-View 3D Reconstruction (https://arxiv.org/abs/2503.06818)
Comments:
          5 pages, 4 figures

- **What's New**: 이 논문에서는 대형 이미지를 더 작은 하위 이미지로 나누어 개별적으로 처리하는 방식인 서브 이미지 재캡처(sub-image recapture, SIR) 접근법을 개발했습니다. 이 방식은 기존의 3D 재구성 알고리즘이 메모리 요구사항을 크게 줄이면서도 확장성을 개선할 수 있도록 합니다. 이로 인해 기존의 학습 기반 다중 뷰 스테레오 알고리즘(MVS)을 적용할 수 있는 가능성이 열립니다.

- **Technical Details**: SIR 접근법은 원본 이미지를 더 작은 하위 이미지로 나누어 각각을 독립적인 이미지로 취급합니다. 이 과정은 향후 깊이 정보를 계산할 때 더 집중적인 하위 이미지를 제공하여 공간적 해상도는 유지한 채 세부 손실 없이 처리됩니다. 각 하위 이미지는 고유한 카메라 매개변수와 연결되어 독립적인 재캡처 카메라로 취급됩니다.

- **Performance Highlights**: 이 접근법은 MVS 단계에서 메모리 사용을 크게 줄이면서도, 동일한 영역의 깊이 추정 정확도를 높일 수 있는 장점을 제공합니다. 또한, 인접 이미지 사이의 오버랩이 적은 경우 메모리 사용을 더욱 효과적으로 사용할 수 있으며, MVS 알고리즘의 병렬 구현이 개선됩니다. 이러한 점에서 SIR은 고해상도 3D 재구성 파이프라인을 효율적으로 지원하는 방식으로 나타납니다.



### Multimodal Emotion Recognition and Sentiment Analysis in Multi-Party Conversation Contexts (https://arxiv.org/abs/2503.06805)
Comments:
          5 pages

- **What's New**: 이번 연구에서는 멀티모달(multi-modal) 접근 방식을 통해 감정 인식(emotion recognition) 및 감성 분석(sentiment analysis) 문제를 다루는 시스템을 제안합니다. 이 시스템은 텍스트를 위한 RoBERTa, 음성을 위한 Wav2Vec2, 얼굴 표정을 위한 FacialNet, 비디오 분석을 위한 CNN+Transformer 아키텍처 등 네 가지 주요 모달리티(modality)를 통합합니다. 각 모달리티의 특징 벡터를 결합하여 멀티모달 벡터를 생성하고, 이를 사용해 감정 및 감성 레이블을 예측합니다. 실험 결과, 제안한 멀티모달 시스템은 단일 모달(unimodal) 접근 방식에 비해 우수한 성능을 보였습니다.

- **Technical Details**: 본 연구에서 사용된 MELD 데이터셋은 여러 모달리티를 포함하는 포괄적인 멀티모달 데이터셋으로, TV 시리즈 'Friends'의 대화에서 파생되었습니다. 이 데이터셋은 다양한 감정을 표현한 1,400개 이상의 대화 인스턴스와 13,000개 이상의 발화를 포함하며, 각 발화는 일곱 가지 감정 라벨과 세 가지 감성 라벨로 주석이 달려 있습니다. Wav2Vec2 및 RoBERTa와 같은 사전 학습(pre-trained) 모델을 활용하여 데이터의 부족 문제를 해결하고, 다양한 융합 기법(fusion techniques)을 통해 각 모달리티의 정보를 효과적으로 종합합니다.

- **Performance Highlights**: 제안된 멀티모달 시스템은 감정 인식에서 66.36%의 정확도를 달성했으며, 감성 분석에서는 72.15%의 정확도를 기록했습니다. 이러한 성과는 기존의 단일 모달 분석 방식에 비해 뚜렷하게 우수한 결과로, 실제 대화 상황에서 감정과 감성을 보다 잘 인식하고 분석할 수 있는 가능성을 보여줍니다. 실험 결과는 멀티모달 접근 방식의 유효성을 뒷받침하며, 더욱 복잡한 인간-기계 상호작용(Human-Machine Interaction)을 지원하는 데 기여할 것으로 기대됩니다.



### VideoPhy-2: A Challenging Action-Centric Physical Commonsense Evaluation in Video Generation (https://arxiv.org/abs/2503.06800)
Comments:
          41 pages, 33 Figures

- **What's New**: 이번 연구에서는 VideoPhy-2라는 데이터셋을 소개하여, 생성된 비디오가 물리적 상식(physical commonsense)을 얼마나 잘 따르는지를 평가하고 있습니다. 이 데이터셋은 200가지 다양한 액션을 포함하며, 현대 생성 모델의 비디오 합성을 위한 상세한 프롬프트(prompts)를 제공합니다. 연구진은 비디오에서의 의미 일치(semantic adherence), 물리적 상식, 그리고 물리적 규칙의 준수를 평가하기 위해 인간 평가를 수행했습니다.

- **Technical Details**: VideoPhy-2 데이터셋은 최대 600개의 액션 목록을 검토하여 197개의 액션을 선별하고, 각각의 액션에 대해 20개의 프롬프트를 생성하여 총 394,039개의 상세 프롬프트를 작성하였습니다. 이 과정에는 Gemini-2.0-Flash-Exp라는 대형 언어 모델(LLM)을 활용하였고, 물리 법칙(physical laws)을 평가하기 위한 기준도 포함되었습니다. 연구진은 비디오 모델이 이러한 물리 법칙을 얼마나 잘 준수하는지를 평가하기 위해 수동 및 자동 평가 모델인 VideoPhy-2-AutoEval을 개발했습니다.

- **Performance Highlights**: 연구 결과, 최고의 모델인 Wan2.1-14B가 VideoPhy-2의 어려운 하위 집합에서 22%의 조합 성과(joint performance)만을 달성한 것으로 밝혀졌습니다. 모델들이 물리적 상식, 특히 질량 및 운동량을 포함한 보존 법칙(conservation laws)을 잘 따르지 못하는 경향을 보였습니다. VideoPhy-2는 모던 비디오 생성 모델을 평가하는 데 매우 중요한 기준선이 되며, 향후 연구의 방향을 제시합니다.



### Silent Hazards of Token Reduction in Vision-Language Models: The Hidden Impact on Consistency (https://arxiv.org/abs/2503.06794)
- **What's New**: 이 논문은 비주얼 랭귀지 모델(VLMs)의 토큰 감소가 출력 안정성에 미치는 영향을 분석합니다. 연구 결과, 기존의 성능 지표가 반영하지 못하는 모델 출력의 불일치를 발견했습니다. 이러한 불일치는 의료 시스템과 같이 신뢰성 높은 성능이 요구되는 실용적인 응용 분야에 심각한 문제를 초래할 수 있습니다. 저자들은 이러한 문제를 해결하기 위해 LoFi라는 새로운 훈련 없는 토큰 감소 방법을 제안합니다.

- **Technical Details**: 논문에서는 VLM의 내부 표현에서 에너지 분포가 토큰 감소에 의해 어떻게 변화하는지를 분석합니다. 세부적으로는 SVD(Singular Value Decomposition)를 사용하여 주성분 방향의 변화를 모니터링하며, IPR(Inverse Participation Ratio) 값을 통해 에너지 분포의 집중도를 평가합니다. 이러한 접근법은 각 레이어에서 토큰 감소가 내부 표현에 미치는 영향을 계량적으로 파악하는 데 도움을 줍니다.

- **Performance Highlights**: LoFi는 기존의 최첨단 방법들보다 높은 출력 일관성과 함께 계산 비용을 크게 줄였습니다. 실험 결과는 LoFi가 성능 저하를 최소화하면서도 원래의 모델과 상당히 일관된 출력을 제공함을 보여줍니다. 이 연구는 비주얼 랭귀지 모델이 높은 정확도를 유지하면서도 안정성을 높일 수 있는 새로운 방향을 제시하고 있습니다.



### GenDR: Lightning Generative Detail Restorator (https://arxiv.org/abs/2503.06790)
- **What's New**: 최근 연구는 텍스트-이미지(T2I) 확산 모델을 실제 초해상도(SR)에 적용하여 놀라운 성과를 도출했습니다. 그러나 T2I와 SR 목표 간의 근본적인 불일치는 추론 속도와 세부 사항의 충실도 간의 딜레마를 야기합니다. 본 논문에서는 이러한 간극을 해소하기 위해 1단계 확산 모델인 GenDR을 제시하며, 이는 맞춤형 확산 모델에서 증류되어 큰 잠재 공간을 활용합니다.

- **Technical Details**: GenDR은 고도화된 잠재 공간을 확보하기 위해 16채널 기반 모델 SD2.1-VAE16을 사용합니다. 기존 방법보다 적은 추론 단계로 세부 사항 복원을 가능케 하기 위해 꾸준한 스코어 아이덴티티 증류(CiD) 기법을 적용했습니다. 게다가, CiD를 적대적 학습 및 표현 정렬(CiDA)로 확장하여 훈련을 가속화하고 지각 품질을 향상시킵니다.

- **Performance Highlights**: 실험 결과 GenDR은 정량적 메트릭과 시각적 충실도에서 최첨단 성능을 달성했습니다. 특히, 기존 모델들과 비교해 목표 품질 및 효율성 메트릭에서 큰 향상을 보이며, 주관적인 시각 비교 및 인간 평가에서도 긍정적인 결과를 얻었습니다. GenDR의 간소화된 파이프라인은 효율적인 배포를 위해 VAE와 UNet만으로 구성되어 있습니다.



### Investigating Image Manifolds of 3D Objects: Learning, Shape Analysis, and Comparisons (https://arxiv.org/abs/2503.06773)
- **What's New**: 이 논문은 3D 물체의 이미지 집합이 형성하는 저차원 다양체(low-dimensional manifold)의 본질을 탐구합니다. 포즈 이미지 다양체(pose image manifold)를 저차원 잠재 공간(latent space)으로 매핑하는 기하학 보존 변환을 이용하여, 다양한 물체의 포즈 다양체가 비선형이며 부드러운 특성을 가짐을 발견하였습니다. 이 연구는 기하학적 관점을 통해 매니폴드 학습(manifold learning)과 분석을 결합하여 기계 학습 알고리즘의 성공을 설명할 수 있는 통찰력을 제공합니다.

- **Technical Details**: 저자들은 3D 물체에 대한 이미지 다양체를 특정 조건에서의 회전, 조명 변화 등을 통해 정의하며, 이 다양체가 어떻게 구조화되어 있는지를 조사합니다. 연구는 포즈, 조명, 기타 다양한 조건 하에서의 이미지 처리에 따른 다양체의 기하학을 분석합니다. 이러한 과정에서 켄달(Kendall)의 형태 분석(shape analysis) 기법을 사용하여 모습의 다양체를 비교하고, 객체 간의 유사성을 근거로 클러스터링을 실시합니다.

- **Performance Highlights**: 연구 결과, 동일 객체 클래스에 속하는 물체들의 포즈 다양체가 자주 함께 군집화된다는 흥미로운 발견이 있었습니다. 이러한 기하학을 활용하면 고차원 이미지 데이터에서 물체 인식, 예측 성능 향상 및 학습 방법에 대한 통찰력을 얻는 등 시각 및 이미지 처리 작업을 간소화할 수 있습니다. 논문에서는 효율적인 비전 작업 설계를 위한 통계적 분석 및 생성 모델에 대한 탐구도 다루고 있습니다.



### SemHiTok: A Unified Image Tokenizer via Semantic-Guided Hierarchical Codebook for Multimodal Understanding and Generation (https://arxiv.org/abs/2503.06764)
Comments:
          Under Review

- **What's New**: 최근 SemHiTok의 개발에 따라 시각 정보의 이해 및 생성 작업을 위한 통합된 이미지 토크나이저가 제시되었습니다. 이 모델은 Semantic-Guided Hierarchical codebook을 통해 텍스처와 의미 특징을 동시에 잘 추출할 수 있도록 설계되었습니다. 기존의 접근 방식들이 직면했던 시맨틱(feature)과 텍스처(feature) 간의 균형 잡기 문제를 해결하기 위해, SemHiTok은 각각의 작업에 최적화된 단계별 학습을 가능하게 합니다.

- **Technical Details**: SemHiTok에서 사용되는 Semantic-Guided Hierarchical codebook은 두 가지 가지 기능을 갖추고 있습니다. 첫째로, 이는 고급 시맨틱 기능과 세부 텍스처 정보를 분리하여 학습하게 하여 가장 적합한 양의 정보만을 코드화할 수 있게 합니다. 둘째로, 이는 전이 학습 기법을 통해 더 나은 성능을 발휘하며, 기존 방안에 비해 학습의 복잡성을 줄이는데 도움을 줍니다.

- **Performance Highlights**: SemHiTok은 기존의 통합 토크나이저와 비교하여 256×256 해상도에서 state-of-the-art의 rFID 점수를 달성하였습니다. 또한, 다중 모달 이해 및 생성 작업에서 경쟁력 있는 성능을 보이며, 텍스트에서 이미지 생성 작업에서도 우수한 성과를 나타내는 등의 뛰어난 성능을 자랑합니다. 이러한 결과는 SemHiTok의 절충된 텍스처 및 의미 기능 추출 능력이 다중 모달 대형 모델에 적용될 가능성을 극대화함을 보여줍니다.



### Gaussian RBFNet: Gaussian Radial Basis Functions for Fast and Accurate Representation and Reconstruction of Neural Fields (https://arxiv.org/abs/2503.06762)
Comments:
          Our code is available at this https URL

- **What's New**: 최근 딥러닝 기반의 신경장(field)인 DeepSDF와 Neural Radiance Fields가 RGB 이미지 및 비디오에서 새로운 시각 합성과 3D 재구성을 혁신하고 있습니다. 하지만 고품질의 표현을 위해서는 훈련과 평가 속도가 느린 깊은 신경망이 필요합니다. 본 논문에서는 훈련 및 추론 속도가 빠르며 경량화된 새로운 신경 표현을 제안합니다. Radial Basis Function(RBF) 커널을 이용해 복잡한 비선형 함수를 단일 레이어로 매핑할 수 있음을 보여줍니다.

- **Technical Details**: 전통적인 MLP(다층 퍼셉트론) 기반의 신경장에서는 각 뉴런이 점곱 연산 후 ReLU 활성화를 수행하는데, 이로 인해 복잡한 함수를 매개화하기 위해 넓고 깊은 네트워크가 필요합니다. 본 논문에서는 RBF 커널을 사용함으로써 뉴런이 단순히 점곱 이상의 연산을 수행하도록 하여 네트워크의 깊이와 폭을 줄일 수 있음을 보입니다. 이러한 compact한 표현은 저해상도 특징 그리드를 통해 2D(RGB 이미지), 3D(형상), 5D(방사 필드) 신호를 정확히 표현할 수 있습니다.

- **Performance Highlights**: 제안된 모델은 NVIDIA GEFORCE RTX4090 GPU를 활용해 3D 형상 표현의 훈련을 15초 이내에 수행할 수 있으며, 새로운 시각 합성은 15분 이내에 완료됩니다. 런타임에서는 60fps 이상의 속도로 새로운 시각을 합성할 수 있으면서도 품질을 저하시킬 필요가 없습니다. 이러한 성능은 메모리 효율성을 높이며 훈련 속도를 단축시키는 데 큰 기여를 합니다.



### Revisiting Invariant Learning for Out-of-Domain Generalization on Multi-Site Mammogram Datasets (https://arxiv.org/abs/2503.06759)
- **What's New**: 이번 연구는 유방암 예측을 위한 전 장벽 유방촬영술(whole mammogram) 데이터에 대한 불변 학습(invariant learning) 방법의 적용을 재평가합니다. 기존 연구들은 특정 대륙의 데이터에서 훈련된 CNN 모델의 성능 저하를 경험했으나, 이 연구는 다수의 다중 사이트 데이터셋을 활용하여 이 문제를 해결하고자 합니다. 이 연구는 의료 영상에서의 변동성을 해결하기 위한 다양한 불변 학습 방법들을 비교 분석하며, 각 방법의 이점과 한계를 조명합니다.

- **Technical Details**: 연구에서는 잔여 네트워크(ResNet) 및 현대적인 합성곱 신경망(ConvNeXt)을 기반으로 한 모델 훈련 방법론을 제안합니다. 불변 위험 최소화(Invariant Risk Minimization, IRM)와 위험 외삽(Risk Extrapolation, REx) 기술을 통해 다양한 데이터 분포에서 일관된 모델을 구축하고, 의료 진단에의 효과적인 일반화를 도모합니다. 이 과정은 각 병원이나 기관에서 수집된 데이터셋을 서로 다른 환경으로 간주하여 수행됩니다.

- **Performance Highlights**: 실험은 CBIS-DDSM, EMBED, INbreast, BMCD 등 다양한 공개 데이터셋을 포함하며, 얻어진 결과는 out-of-domain 환경에서도 모델의 견고성을 입증합니다. 실험을 통해 IRM 및 REx 방법의 성능을 비교하고, 평균 정밀도(average precision) 및 곡선 아래 면적(area under the curve) 등의 평가 지표를 사용하여 그 효과성을 평가합니다. 또한, 클래스 활성화 맵(class activation maps)을 활용하여 모델의 해석 가능성도 분석하였습니다.



### Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models (https://arxiv.org/abs/2503.06749)
- **What's New**: DeepSeek-R1-Zero는 강화 학습(Reinforcement Learning, RL)만으로 LLM의 추리 능력을 성공적으로 증명했습니다. 이에 영감을 받아, 이번 연구에서는 MLLM의 추리 능력을 향상시키기 위해 RL을 어떻게 활용할 수 있을지 탐구합니다. 다만, MLLMs에서 직접적인 RL 학습은 고품질 다중 모달 추리 데이터의 부족으로 인해 복잡한 추리 능력을 이끌어내기에는 한계를 겪고 있습니다. 이를 극복하기 위해 Vision-R1이라는 추리 MLLM을 제안합니다.

- **Technical Details**: 이번 연구에서는 기존 MLLM과 DeepSeek-R1을 활용하여 사람의 주석 없이도 20만 개의 고품질 다중 모달 CoT(Coach-of-Thought) 데이터셋, Vision-R1-cold 데이터셋을 구축합니다. Vision-R1은 초기 데이터 셋을 통해 추리 능력을 높이는 한편, Progressive Thinking Suppression Training (PTST) 전략을 도입하여 모델의 올바른 추리 과정을 학습할 수 있도록 돕습니다. 특히, Group Relative Policy Optimization (GRPO) 기법과 하드 포맷 결과 보상 함수를 사용하여 10K 다중 모달 수학 데이터셋에서 모델의 성능을 점진적으로 개선합니다.

- **Performance Highlights**: 포괄적인 실험 결과, Vision-R1은 다양한 다중 모달 수학 추리 벤치마크에서 평균 6% 개선된 성능을 보였습니다. 특히, Vision-R1-7B는 MathVista 벤치마크에서 73.5% 정확도를 기록하며, 최고의 추리 모델인 OpenAI O1과 오직 0.4% 차이로 성능을 발휘했습니다. 이번 연구에서 제안한 데이터셋과 코드도 공개되어 연구자들 간의 상호 작용을 촉진할 것으로 기대됩니다.



### DiffAtlas: GenAI-fying Atlas Segmentation via Image-Mask Diffusion (https://arxiv.org/abs/2503.06748)
Comments:
          11 pages

- **What's New**: 본 논문에서는 DiffAtlas라는 새로운 생성 프레임워크를 제안합니다. 이는 딥러닝 기반의 의료 이미지 분할 방법의 한계를 극복하기 위해 개발되었습니다. 전통적인 이미지-마스크 매핑에 의존하지 않고, 확산 모델(diffusion model)을 통해 이미지와 마스크를 동시에 모델링함으로써 유연성과 강건성을 제공합니다.

- **Technical Details**: DiffAtlas는 아틀라스 기반의 세분화를 현대적인 생성 AI 기술로 개선한 방법입니다. 이는 훈련 중 이미지-마스크 쌍의 조인트 분포를 캡처하여 아틀라스의 강건성과 해부학적 일관성을 유지합니다. 더불어, 예측 과정에서 노이즈가 있는 이미지를 사용하여 최종 마스크를 생성하는 과정을 통해 안정성과 정확성을 향상시킵니다.

- **Performance Highlights**: DiffAtlas는 CT 및 MRI 데이터를 사용한 실험에서 제한된 데이터 및 제로-샷 모드 세분화 조건에서 기존 방법보다 우수한 성능을 보였습니다. MM-WHS 및 TotalSegmentator 데이터셋을 활용하여 다양한 세팅에서 최첨단 성과를 달성하였습니다. 이로 인해 DiffAtlas는 의료 이미지 분석의 새로운 가능성을 열어줍니다.



### Color Alignment in Diffusion (https://arxiv.org/abs/2503.06746)
Comments:
          CVPR 2025

- **What's New**: 이 논문에서는 색상 정렬(color alignment) 알고리즘을 소개하며, 확산 모델(diffusion models)에서 생성 과정을 주어진 색상 패턴에 맞추는 방법을 제안합니다. 기존 이미지 합성 방법이 원하는 픽셀 조건에 맞지 않는 경우가 많았으나, 새로운 접근법을 통해 색상 분포와 정렬된 결과물을 생성할 수 있습니다. 실험 결과는 높은 품질의 다양성을 유지하면서 색상 조정을 효과적으로 수행하는 수준 있는 성능을 보여줍니다.

- **Technical Details**: 제안된 색상 정렬 방법은 이미지를 입력 색상 공간으로 투사하여 확산 모델의 예측을 단순화합니다. 이 과정에서는 이미지 색상 조건을 흐리게(blurring) 하는 이점을 통해 지역 고주파 정보를 줄이고, latent representation에서 더 정확한 색상 인코딩을 허용합니다. 색상 정렬은 최종 결과물의 세부 사항을 정제하는 데 필요한 late time step에서 중단되어, 자연스러운 조명과 선명한 의미를 갖춘 색상을 생성할 수 있도록 합니다.

- **Performance Highlights**: 제안된 방법은 기존의 상업 제품과 비교했을 때 색상 조건 이미지 합성에서 유효성을 증명하였습니다. 특히, 색상 정렬이 없는 상태에서는 생성된 결과물이 점묘적(texter)이고 조각난 느낌을 주는 반면, 색상 정렬이 있는 경우에는 더 매끄러운 결과물이 도출되었습니다. 정량적 평가 결과는 제안된 방법의 유효성을 더욱 강화하며, 다양한 색상 조건에서 균형 잡힌 이미지 합성을 보여주었습니다.



### CoDa-4DGS: Dynamic Gaussian Splatting with Context and Deformation Awareness for Autonomous Driving (https://arxiv.org/abs/2503.06744)
- **What's New**: 이 논문에서는 자율 주행을 위한 동적인 장면 렌더링을 개선하기 위해 4D Gaussian Splatting (4DGS) 접근 방식을 소개합니다. 이 방법은 맥락(context)과 시간 변형(temporal deformation) 인지 능력을 통합하여 동적인 장면을 더 정확하게 표현할 수 있도록 합니다. 이를 통해 자율 주행 알고리즘의 검증에 필요한 포토리얼리스틱(photorealistic) 데이터 기반의 폐쇄 루프 시뮬레이션을 가능하게 합니다.

- **Technical Details**: 4DGS의 핵심은 2D 의미 분할(semantic segmentation) 기초 모델을 사용하여 Gaussian의 4D 의미적 특성을 자율적으로 학습한다는 점입니다. 이 방법은 각 Gaussian이 인접 프레임에서 시간 변형을 추적하여 3D 공간 내에서 변형 보상을 위한 단서를 제공합니다. 이러한 세부 기술들은 동적 장면을 더욱 정밀하게 표현하는 데 필수적입니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 동적 장면 렌더링 시 세부 사항을 더 잘 포착하는 4DGS의 능력을 향상시켰습니다. 또한, 기존의 자가 감독(self-supervised) 방법들과 비교했을 때 4D 재구성(4D reconstruction) 및 새로운 뷰 합성(novel view synthesis)에서 뛰어난 성능을 보였습니다. 추가적으로, CoDa-4DGS는 각 Gaussian에 대해 의미적 특성을 변형시킬 수 있어 더 넓은 응용 가능성을 제공합니다.



### D3DR: Lighting-Aware Object Insertion in Gaussian Splatting (https://arxiv.org/abs/2503.06740)
- **What's New**: 이번 연구에서는 D3DR이라는 방법을 제안하여 3D Gaussian Splatting(3DGS) 장면에 3DGS 매개변수화된 객체를 자연스럽게 삽입하는 문제를 해결한다. 이는 조명, 그림자 및 기타 시각적 아티팩트를 수정하여 장면과의 일관성을 보장하는 방식으로, 이전에 성공적으로 해결되지 않았던 문제이다. 본 연구는 확산 모델(diffusion models)의 발전을 활용해 실제 세계 데이터를 기반으로 훈련된 모델이 올바른 장면 조명을 이해하게끔 하고 있다.

- **Technical Details**: 3D객체 삽입 및 재조명(relighting) 방법이 확산 기반의 접근 방식을 통해 조명 일관성을 유지하도록 구현되었다. 객체를 새 장면으로 삽입 후, 우리는 Delta Denoising Score (DDS)에 기반한 최적화를 통해 3D Gaussian 매개변수를 조정한다. 이 방법은 객체의 구조적 세부 사항을 보존하면서 주변 환경과의 일관성을 유지하는 데 초점을 맞춘다.

- **Performance Highlights**: 기존 방법과 비교하여, D3DR의 효과를 입증하기 위해 0.5 PSNR 및 0.15 SSIM 증가라는 측정 결과를 달성하여 재조명 품질에서 향상됨을 보여준다. 이 연구는 3DGS 장면과 객체의 다양성을 지닌 데이터 세트를 수집하였으며, 광범위한 벤치마킹을 통해 기존 방법들 보다 현저히 우수한 성능을 입증하였다.



### Continuous Online Adaptation Driven by User Interaction for Medical Image Segmentation (https://arxiv.org/abs/2503.06717)
- **What's New**: 이번 연구에서는 사용자 상호작용(Interaction)을 통해 의료 영상 분할을 향상시키기 위한 온라인 적응 프레임워크(Online Adaptation Framework)를 도입했습니다. 이 프레임워크는 사용자의 클릭을 통해 모델을 지속적으로 학습하고, 새로운 데이터 분포에 대해 성능을 개선하는 기능을 제공합니다. 이를 통해 모델 배포 후에도 사용자 피드백을 활용해 영상 분할 성능을 자동으로 조정할 수 있게 되었습니다.

- **Technical Details**: 주요 기술적 기여는 Gaussian Point Loss (GPL) 함수의 도입과 이중 단계의 온라인 최적화 방법으로 나뉩니다. GPL은 사용자 클릭 주변의 일관되고 정확한 분할을 보장하도록 설계되었습니다. 모델은 첫 번째 단계에서 단일 클릭을 기반으로 최적화되고, 두 번째 단계에서는 추가적인 수정 클릭을 통해 세밀한 조정을 받습니다.

- **Performance Highlights**: 실험 결과, 5가지 안저 영상(db)과 4가지 뇌 MRI 데이터셋을 기반으로 한 실험에서 제안된 방법이 기존의 접근 방식보다 우수한 성능을 보였습니다. 특히 다양한 데이터 분포 이동(data distribution shifts) 환경에서 뛰어난 성능을 입증하였으며, 이는 임상 작업 흐름에서의 신뢰성을 더욱 높여줍니다.



### MemorySAM: Memorize Modalities and Semantics with Segment Anything Model 2 for Multi-modal Semantic Segmentation (https://arxiv.org/abs/2503.06700)
- **What's New**: MemorySAM은 다양한 센서에서 수집된 다중 시각 모달리티를 기반으로 다중 모달 의미 분할(MMSS) 문제를 해결하기 위한 최신 프레임워크입니다. 특히, MemorySAM은 세 가지 주요 요소인 메모리 메커니즘, 클래스 수준의 프로토타입 메모리 모듈(SPMM) 및 프로토타입 적응 손실을 결합하여 SAM2 모델의 성능을 향상시킵니다. 이 차별점은 다중 모달 데이터의 이해에 새로운 패러다임을 제시하고, 현재 문제를 해결하는 데 있어 높은 효율성을 제공합니다.

- **Technical Details**: MemorySAM 프레임워크는 다중 모달 데이터를 동일한 장면을 나타내는 일련의 프레임으로 간주하여 모달리티 관련 정보를 효과적으로 캐치합니다. 또한, SPMM을 사용하여 클래스 수준의 프로토타입을 학습하며, 두 가지 유형의 프로토타입(현재 프로토타입과 전역 프로토타입)을 통해 의미적 지식을 획득합니다. 프로토타입 적응 손실을 통해 두 프로토타입 간의 정렬을 반복적으로 수행하여 SAM2의 의미적 이해 능력을 더욱 향상시키고 있습니다.

- **Performance Highlights**: MemorySAM은 DELIVER와 MCubeS 데이터셋에서 각각 65.38%와 52.88%의 성능으로 기존의 선진적 방법들(State-of-the-Art)을 크게 초월하는 성과를 거두었습니다. 이는 MemorySAM이 다중 모달 의미 분할 분야에서 그 효용성과 성능을 널리 인증받았음을 나타냅니다. 향후 공개될 소스 코드를 통해 더 많은 연구자들이 MemorySAM의 기능을 활용할 수 있게 될 것입니다.



### Asymmetric Decision-Making in Online Knowledge Distillation:Unifying Consensus and Divergenc (https://arxiv.org/abs/2503.06685)
- **What's New**: 이번 논문에서는 Asymmetric Decision-Making (ADM)이라는 혁신적인 온라인 지식 증류 방법론을 제안합니다. OKD(Online Knowledge Distillation) 프레임워크를 활용하여, 교사 모델과 학생 모델 간의 특징 학습을 동시에 강화하는 접근법을 개발했습니다. 이 방법은 효과적으로 기존 기술과 차별화를 두며, 지식 전이의 여러 문제를 해결합니다.

- **Technical Details**: 이 논문은 두 가지 주요 인사이트를 바탕으로 합니다. 첫 번째로, 교사 모델과 학생 모델 간의 유사한 특징은 주로 전경 물체에 집중된다는 것이고, 두 번째로, 교사 모델이 학생 모델보다 전경 물체를 강조한다는 것입니다. ADM은 이러한 특성을 활용하여 학생 모델의 특징 학습을 가속화하고 교사 모델의 다양성을 유지합니다.

- **Performance Highlights**: 광범위한 실험 결과 ADM이 다양한 온라인 지식 증류 설정에서 기존 OKD 방법을 지속적으로 초월했음을 보여줍니다. 특히, 분산 증류(dispersion distillation), 의미 세그멘테이션(semantic segmentation) 및 오프라인 지식 증류 분야에서도 우수한 성능을 달성했습니다.



### PixelPonder: Dynamic Patch Adaptation for Enhanced Multi-Conditional Text-to-Image Generation (https://arxiv.org/abs/2503.06684)
- **What's New**: PixelPonder라는 새로운 통합 제어 프레임워크를 소개하는 이 논문은 여러 비주얼 컨디션을 단일 제어 구조 아래에서 효과적으로 관리할 수 있는 방법을 제시하고 있습니다. 전통적인 ControlNet과 같은 방법들이 다중 이질적인 제어 신호를 동시에 유지하면서 높은 비주얼 퀄리티를 제공하는 데 어려움을 겪던 반면, PixelPonder는 공간적으로 관련성이 높은 제어 신호를 동적으로 우선순위로 조정하여 정확한 로컬 가이드를 가능하게 합니다.

- **Technical Details**: PixelPonder는 패치 레벨의 적응형 조건 선택 메커니즘을 통해 시각적 컨디션을 효율적으로 결합하는 방법을 제공합니다. 이 메커니즘은 각 제어 조건을 시간 단계에 따라 조절하여 다양한 시각적 신호의 사용자 정의 조합을 통해 노이즈 제거 과정에서 다르게 작용하도록 합니다. 또한, 시간 인식 제어 주입 전략을 통해 구조 보존에서 텍스처 정제까지 점진적으로 전환하는 과정을 통해 서로 다른 카테고리에서 최대한의 제어 정보를 활용하게 됩니다.

- **Performance Highlights**: 논문은 PixelPonder가 다양한 벤치마크 데이터셋에서 이전 방법들을 능가하며, 공간 정렬 정확도를 높이는 동시에 높은 텍스처 의미 일치를 유지하는 데 성공했다고 강조합니다. extensive experiments를 통해 PixelPonder는 복잡한 비주얼 조건 하에서도 양호한 성능을 보여줍니다. 이러한 결과는 다중 비주얼 조건을 동시 제어하려는 최신 연구의 발전을 보여주는 중요한 성과로 해석될 수 있습니다.



### Dynamic Dictionary Learning for Remote Sensing Image Segmentation (https://arxiv.org/abs/2503.06683)
- **What's New**: 이 논문은 원거리 감지(remote sensing) 이미지 분할(segmentation)에서 발생하는 지속적인 문제를 해결하기 위한 새로운 접근법을 소개합니다. 기존의 방법들이 명시적이지 않은 표현 학습(implicit representation learning)에 의존하는 반면, 본 연구는 클래스 ID 임베딩(class ID embeddings)을 명시적으로 모델링하는 동적 사전 학습(dynamic dictionary learning) 프레임워크를 제안합니다. 이를 통해 다양한 장면 변화를 효과적으로 조정할 수 있는 메커니즘을 개발했습니다.

- **Technical Details**: 연구의 핵심 기여는 다단계 교차 주의(multi-stage alternating cross-attention) 쿼리를 통해 이미지 특징(image features)과 사전 임베딩(dictionary embeddings) 간의 상호작용을 통해 클래스에 민감한 의미 임베딩(class-aware semantic embeddings)을 점진적으로 업데이트하는 새로운 사전 구성 메커니즘입니다. 이러한 프로세스는 입력의 특성에 맞춘 적응형 표현 학습(adaptive representation learning)을 가능하게 하여, 클래스 내 이질성과 클래스 간 동질성의 모호함을 효과적으로 해결합니다.

- **Performance Highlights**: 이 논문에서는 차별성을 높이기 위해 사전 공간(dictionary space)에 대조적 제약(contrastive constraint)을 적용하여, 클래스 내 분포를 압축하고 클래스 간 분리를 극대화합니다. 다양한 데이터셋에 대한 광범위한 실험을 통해 기존 최첨단 방법들보다 일관된 성능 개선을 보여주었으며, 특히 두 개의 온라인 테스트 벤치마크인 LoveDA와 UAVid에서 우수한 결과를 기록했습니다.



### Gamma: Toward Generic Image Assessment with Mixture of Assessment Experts (https://arxiv.org/abs/2503.06678)
- **What's New**: 이번 논문에서는 다양한 이미지 평가 시나리오를 효과적으로 처리할 수 있는 Gamma라는 일반화된 이미지 평가 모델을 소개합니다. 이 모델은 혼합 데이터셋 훈련을 통해 이미지를 평가하며, 기존의 방법들이 개인 시나리오에 국한되었던 것과는 달리, 여러 평가 시나리오를 포괄하는 통합 접근 방식을 제공합니다. 특히, Mixture of Assessment Experts (MoAE) 모듈과 Scene-based Differential Prompt (SDP) 전략을 통해 각 데이터셋의 특성에 적응할 수 있도록 설계되었습니다.

- **Technical Details**: Gamma 모델은 다양한 데이터셋으로부터 공통 및 특정 지식을 동적으로 학습하도록 설계된 MoAE 모듈을 포함합니다. 이 모듈은 공유 전문가와 적응형 전문가로 구성되어 있으며, 각 데이터셋에 따라 다르게 활성화됩니다. 또한, 각 데이터셋의 장면에 따라 상이한 프롬프트를 사용하여 훈련 과정에서의 가이드를 제공하는 SDP 전략을 도입했습니다. 이를 통해 이미지 평가의 통합 훈련에서의 주된 도전 과제인 평균 의견 점수(MOS) 편향을 완화할 수 있습니다.

- **Performance Highlights**: Gamma 모델은 6개의 이미지 평가 시나리오를 포함하는 12개의 데이터셋에서 훈련 및 평가를 실시했으며, 기존의 혼합 훈련 방법들에 비해 우수한 성능을 보였습니다. 어떤 벤치마크에서는 Gamma가 특정 작업에 최적화된 SOTA 방법들을 초월하는 성과를 달성하였습니다. 또한 MoAE가 장착된 Gamma 모델을 특정 데이터셋에 세밀하게 조정함으로써 SOTA 성능을 달성할 수 있습니다.



### REArtGS: Reconstructing and Generating Articulated Objects via 3D Gaussian Splatting with Geometric and Motion Constraints (https://arxiv.org/abs/2503.06677)
Comments:
          11pages, 6 figures

- **What's New**: 이번 연구에서는 REArtGS라는 새로운 프레임워크를 제안하여 고품질의 텍스처가 풍부한 아티큘레이티드(objects)가 있는 객체의 표면 복원 및 생성을 수행합니다. 이 프레임워크는 다각도 RGB 이미지를 통해 3D Gaussian primitives를 사용하여 동적 형상을 생성할 수 있게 설계되었습니다. 또한, 각 아티큘레이티드 객체의 운동 구조를 활용하여 동적 표면 메쉬를 비지도 방식으로 생성할 수 있습니다.

- **Technical Details**: 우리는 먼저 비편향 Signed Distance Function(SDF) 지침을 도입하여 Gaussian opacity 필드의 기하학적 제약을 강화하여 복원 품질을 개선합니다. 이후, 3D Gaussian primitives의 최적화된 기하학적 초기화를 사용하여 동적 표면 생성을 위한 연속 변형 필드를 설정합니다. 이는 아티큘레이티드 객체의 운동 구조에 제약을 받으며, 이는 고품질의 표면 복원을 돕습니다.

- **Performance Highlights**: 제안된 REArtGS는 PartNet-Mobility와 AKB-48 데이터베이스에서 평가되었으며, 기존 방법들과 비교해 표면 복원 및 생성 작업에서 월등히 뛰어난 성능을 보였습니다. 실험 결과는 다양한 아티큘레이티드 객체 범주에서 REArtGS가 여러 조건에서 유용하다는 것을 보여줍니다. 논문에서 설명된 모든 코드와 데이터는 다음 4개월 이내에 공개될 예정입니다.



### Seeing Delta Parameters as JPEG Images: Data-Free Delta Compression with Discrete Cosine Transform (https://arxiv.org/abs/2503.06676)
Comments:
          15 pages, 7 figures

- **What's New**: 본 논문에서는 Delta-DCT라는 새로운 데이터 없는(delta-free) delta 압축 방법을 소개합니다. 이 방법은 전통적인 JPEG 이미지 압축에서 영감을 받아, Transformer 기반 모델의 개선된 저장 및 배포 효율성을 제공하여 다양한 작업에서의 모델 성능 저하를 줄입니다. Delta-DCT는 훈련이나 데이터 교정 없이도 성능을 유지하거나 기존의 미세 조정된 모델을 초과하는 결과를 보여줍니다.

- **Technical Details**: Delta-DCT는 레이어 내의 delta 파라미터를 패치(patch)로 그룹화하고, 각 패치의 중요성을 평가하여 서로 다른 양자화 비트(width)를 할당하는 방식으로 작동합니다. 패치가 DCT(Digital Cosine Transform) 도메인으로 변환 후 할당된 비트 폭에 기반하여 양자화를 수행하여 데이터 없는 delta 압축을 구현합니다. 이 과정에서 데이터나 추가 훈련 없이 delta 압축을 실현하게 됩니다.

- **Performance Highlights**: 제안된 Delta-DCT는 다양한 모델에서 우수한 성능을 보여줍니다. 예를 들어, 7B에서 13B 크기의 LLMs와 같은 최근에 발표된 모델들에서 delta 압축을 통해 성능 저하를 최소화하며, RoBERTa와 T5와 같은 상대적으로 작은 언어 모델에서도 기존 방법들을 초월합니다. 또한, 시각 변환기 모델과 다중 모드 BEiT-3 모델에 대해서도 탁월한 성능을 입증하여, 다양한 작업에 걸쳐 Delta-DCT의 높은 유용성을 강조합니다.



### Learning Few-Step Diffusion Models by Trajectory Distribution Matching (https://arxiv.org/abs/2503.06674)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 Trajectory Distribution Matching (TDM)이라고 불리는 새로운 방법을 제안합니다. TDM은 두 가지 방식인 distribution matching과 trajectory matching의 장점을 결합하여 몇 단계의 샘플링을 학습합니다. 또한, data-free score distillation objective를 도입하여, 학생 모델의 경로를 교사 모델과 분포 수준에서 일치시킵니다. 이 방법을 통해 TDM은 고유한 이미지 품질과 사용자 선호도를 증가시킵니다.

- **Technical Details**: Diffusion Model (DMs)은 데이터에 Gaussian 노이즈를 점진적으로 주입하는 과정으로 정의됩니다. TDM은 이전의 방법들보다 더 나은 효율성을 제공하며, 특히 동적 샘플링_STEP에 대해 분리된 학습 목표를 설정합니다. 이 접근 방식은 모델이 다양한 샘플링 단계를 통해 더 쉽게 조정할 수 있도록 하여, 더 나은 성능을 발휘합니다. TDM은 하이퍼 파라미터 설정이 변경될 때도 유연하게 대응할 수 있습니다.

- **Performance Highlights**: TDM은 SDXL, PixArt-$\alpha$ 등 다양한 기반 모델에서 기존 방법들보다 우수한 성능을 보여줍니다. 특히, 본 연구에서는 TDM을 활용하여 4-step 생성기를 훈련시킨 결과, 실제 사용자 선호도에서 교사 모델을 초과했습니다. TDM은 오직 500번의 반복학습과 2 A800 시간만으로 이루어지며, 이는 교사의 훈련 비용의 단 0.01%에 불과합니다.



### Emulating Self-attention with Convolution for Efficient Image Super-Resolution (https://arxiv.org/abs/2503.06671)
- **What's New**: 이번 연구는 lightweight image super-resolution (SR)을 위한 transformer의 높은 계산 부담에 도전하며, Convolutional Attention (ConvAttn) 모듈을 소개하여 self-attention의 장거리 모델링 능력과 인스턴스 의존 가중치를 단일 공유 대형 커널과 동적 커널로 구현합니다. ConvAttn 모듈을 활용하여 self-attention에 대한 의존도를 대폭 줄이고 메모리 제약 작업을 완화했습니다. 또한 flash attention을 통합하여 lightweight SR 영역에서의 메모리 병목 현상을 해결하여 성능을 더욱 개선했습니다.

- **Technical Details**: ConvAttn 모듈은 self-attention의 장거리 상호작용을 단순화하는 한편, 입력에 의존하는 가중치를 캡처하기 위한 동적 커널 생성을 통해 self-attention의 장점인 대표성 있는 모델링을 보존합니다. 랜덤하게 생성된 동적 커널은 각 입력에 따라 가중치를 조정하며, ConvAttn의 적용으로 self-attention의 대부분을 대체하면서 이미지 복원 과정에서 메모리 소비를 줄였습니다. 우리는 Window의 크기를 32x32로 확장하여 flash attention을 통해 성능을 개선하며 FLOP과 메모리 사용량을 각각 16배, 12.2배 줄였습니다.

- **Performance Highlights**: 새롭게 제안된 ESC (Emulating Self-attention with Convolution) 네트워크는 Urban100×4 벤치마크에서 HiT-SRF에 비해 PSNR이 0.27dB 향상되었으며, 지연 시간과 메모리 사용량이 각각 3.7배, 6.2배 감소했습니다. ESC는 ATD-light와 비교하여 Urban100×4에서 PSNR을 0.1dB 개선하고 속도는 8.9배 빨라졌습니다. 전체 실험 결과, ESC는 ConvAttn 모듈로 대체된 self-attention에도 불구하고 transformer의 이점인 장거리 모델링, 데이터 확장성, 대표성 있는 능력을 완전히 활용할 수 있음을 증명했습니다.



### Attention, Please! PixelSHAP Reveals What Vision-Language Models Actually Focus On (https://arxiv.org/abs/2503.06670)
- **What's New**: PixelSHAP는 Vision-Language Models (VLMs)에 대한 해석 가능성을 높이기 위해 Shapley 기반 분석을 구조적 시각 개체에 확장한 새로운 모델 무관 프레임워크입니다. 기존의 토큰 기반 접근 방식과는 달리, PixelSHAP는 이미지 개체를 체계적으로 변화시키고 이를 통해 모델의 반응에 미치는 영향을 정량화합니다. 이 프레임워크는 모델 내부에 대한 접근 없이 입력-출력 쌍만으로 작동하므로 오픈 소스 및 상용 모델 모두와 호환됩니다.

- **Technical Details**: PixelSHAP는 섹멘테이션(Segmentation) 모델을 활용하여 객체 마스크를 생성하고, 객체의 변화를 통해 모델의 출력을 평가합니다. 이러한 방식으로 픽셀 단위가 아닌 객체 단위의 perturbation을 수행하여 계산 효율성을 높입니다. 이 메서드는 객체의 중요성을 평가하기 위해 Shapley 기반의 중요도 추정을 적용하며, 복잡한 시각 장면의 해석 가능성을 크게 증가시킵니다.

- **Performance Highlights**: PixelSHAP는 자율 주행과 같은 높은 위험의 응용 분야에서 중요한 해석 가능성을 향상시키는 능력을 보여줍니다. 본 시스템의 개선된 신뢰성 및 투명성 덕분에 사용자는 모델의 결정에 어떤 객체가 영향을 미쳤는지 더 잘 이해할 수 있습니다. 이러한 기능들은 향후 설명 가능한 AI 연구의 진전을 위한 강력한 도구 역할을 할 것입니다.



### AA-CLIP: Enhancing Zero-shot Anomaly Detection via Anomaly-Aware CLIP (https://arxiv.org/abs/2503.06661)
Comments:
          8 pages, 7 figures

- **What's New**: 본 연구에서는 Anomaly-Aware CLIP (AA-CLIP)이라는 새로운 접근 방식을 제안합니다. AA-CLIP은 CLIP의 이상 탐지(discrimination) 능력을 개선하며, 텍스트와 시각적 공간 모두에서 정상 및 비정상 특징을 보다 명확하게 구분합니다. 이 두 단계를 통해 클립 모델을 능동적으로 변형하여 이상 탐지 성능을 저하시키지 않으면서도 향상된 효과를 이끌어냅니다.

- **Technical Details**: AA-CLIP는 단순하지만 효과적인 두 단계 접근 방식을 활용합니다. 첫 번째 단계에서는 'anchors'를 생성하여 각 클래스에 대해 비정상적인 의미를 텍스트 공간에서 명확히 구분하고, 두 번째 단계에서는 이러한 텍스트 앵커와 패치 수준의 시각적 특징을 정렬하여 이상을 정밀하게 탐지합니다. Residual Adapters 구조를 사용하여 클립의 미리 훈련된 지식을 보존합니다.

- **Performance Highlights**: AA-CLIP은 공업과 의료 분야에서의 실험을 통해 자원 효율적인 제로샷 이상 탐지 성능을 달성하였습니다. 제한된 데이터 환경, 즉, 각 클래스당 하나의 정상 샘플과 하나의 비정상 샘플(2-shot)로 훈련하여도 다른 기존 CLIP 기반 기법들과 유사한 성능을 나타냅니다. 또한, 64샷 훈련으로 다양한 데이터셋에서 SOTA 결과를 달성하였습니다.



### AxisPose: Model-Free Matching-Free Single-Shot 6D Object Pose Estimation via Axis Generation (https://arxiv.org/abs/2503.06660)
- **What's New**: AxisPose는 기존의 2D-3D 또는 2D-2D 포즈 추정 방식의 한계를 극복한 혁신적인 단일 샷 (single-shot) 솔루션입니다. 이 방법은 객체의 6D 포즈 추정을 위해 복잡한 매칭 과정 없이, 단일 뷰에서 직접 강력한 6D 포즈를 추론합니다. 이를 위해 AxisPose는 객체 축의 잠재적 기하학적 분포를 캡처하기 위해 확산 모델 (diffusion model)을 사용합니다.

- **Technical Details**: AxisPose는 축 생성 모듈 (AGM)을 통해 객체 축의 기하학적 분포를 배우고, 주어진 노이즈 추정에서 기하학적 일관성 손실의 그래디언트를 주입하여 노이즈를 조정합니다. 이 과정은 생성된 삼축 (tri-axis) 프로젝션을 통해 이루어지며, 이후 삼축 백 프로젝션 모듈 (TBM)을 통해 최종 6D 포즈를 회복합니다. 기존의 깊이 데이터나 CAD 모델을 사용하지 않고도, 단일 뷰만으로도 강력한 성능을 발휘할 수 있습니다.

- **Performance Highlights**: AxisPose는 참조 이미지 없이 단일 뷰로 입력을 받고, 다양한 인스턴스에 대해 하나의 모델로 작업할 수 있는 뛰어난 범용성을 입증합니다. 이를 통해 새로운 객체에 대한 포즈 추정 문제에 대한 가능성을 제시하며, 기존 방법들보다 향상된 일반화 능력을 보여주고 있습니다. 연구 결과, AxisPose는 실질적인 응용 가능성을 가지고 있으며, 기존 방법들이 가지는 입력 및 환경적 제약을 넘어 새로운 접근법을 제시합니다.



### Adding Additional Control to One-Step Diffusion with Joint Distribution Matching (https://arxiv.org/abs/2503.06652)
- **What's New**: 이 논문에서는 Joint Distribution Matching (JDM)이라는 새로운 접근법을 소개합니다. JDM은 이미지 조건 조합 분포 간의 역 Kullback-Leibler (KL) 발산을 최소화하며, 이는 모델이 알려지지 않은 새로운 제어를 처리할 수 있게 합니다. 이 방법은 기존의 생략된 학습 방식에 비해 더 효율적이고 유연한 방법을 제공합니다.

- **Technical Details**: JDM은 fidelity learning과 condition learning을 분리하여 처리합니다. 이 비대칭 증류 방식은 기존의 모델이 처리하지 못했던 제어 수단을 학생 모델이 사용할 수 있게 돕습니다. 이는 classifier-free guidance (CFG) 및 human feedback learning (HFL) 통합을 통해 제어성과 생성 이미지 품질을 모두 향상시킵니다.

- **Performance Highlights**: 실험 결과 JDM은 기존의 multi-step ControlNet 모델을 한 단계로 초월하며, 평균 FID 점수를 낮추고 (14.58 vs 15.21), 일관성 점수에서 24% 향상을 보였습니다. 특히 text-to-image 생성에서 JDM은 CFG나 HFL을 통합하여 새로운 최첨단 성능을 달성했습니다.



### Personalized Class Incremental Context-Aware Food Classification for Food Intake Monitoring Systems (https://arxiv.org/abs/2503.06647)
- **What's New**: 새로운 논문에서는 개인 맞춤형(class-incremental) 식품 분류 모델을 도입하여 건강한 식단 유지를 위한 정확한 식품 섭취 모니터링을 개선하고자 합니다. 기존의 고정 크기 데이터셋으로 인해 발생하는 한계를 극복하기 위해, 이 모델은 새로운 식품 클래스에 신속하게 적응할 수 있도록 설계되었습니다. 사용자 개인의 식사 빈도, 시간 및 장소와 같은 요소를 반영하여 분류 정확도를 높이며, 실시간 피드백 시스템을 포함하여 지속적으로 학습할 수 있는 구조를 갖추고 있습니다.

- **Technical Details**: 이 연구에서는 개인 맞춤형 동적 지원 네트워크(personalized dynamic support network, PDSN)를 중심으로 한 식품 섭취 모니터링 시스템을 제안합니다. 이 시스템은 스마트 스케일을 활용하여 음식의 중량을 추정하고, 영양소 데이터베이스를 활용해 매크로 영양소의 양을 계산합니다. 또한, FOOD101-Personal과 VFN-Personal와 같은 새로운 벤치마크 데이터셋을 평가하여 모델의 효과성을 입증하였습니다.

- **Performance Highlights**: 제안된 개인 맞춤형 클래스 증가 식품 분류 모델은 새로운 클래스와 기존 클래스 모두에서 높은 분류 정확도를 달성하며 시스템의 적용성을 향상시킵니다. 실험 결과, 이 모델은 기존의 고정식 데이터셋 기반 모델들의 한계를 극복하고 사용자의 식습관에 맞춰 진화하는 능력을 보여줍니다. 이는 건강한 식이요법 유지 및 영양 관련 질병 예방에 기여할 수 있을 것으로 예상됩니다.



### CLICv2: Image Complexity Representation via Content Invariance Contrastive Learning (https://arxiv.org/abs/2503.06641)
- **What's New**: 본 논문은 기존의 CLIC(Contrastive Learning for Image Complexity) 방법론을 재정의하여 CLICv2를 제안합니다. CLICv2는 이미지 품질을 평가하는데 있어 Content Invariance(내용 불변성)를 보장하며, 패치 기반 접근 방식을 통해 긍정 샘플 생성의 편향을 최소화합니다. 따라서 복잡한 이미지를 보다 효과적으로 인식하고 평가할 수 있도록 돕습니다.

- **Technical Details**: CLICv2는 'Shifted Patchify'라는 새로운 샘플 선택 방법론을 적용하여 이미지를 여러 패치로 나누고 무작위 방향으로 이동시키는 작업을 수행합니다. 이를 통해 동일 위치의 패치가 긍정적인 쌍을 형성하여 내용에 의존하지 않는 학습을 가능하게 합니다. 또한 패치 단위로 Contrastive Loss를 적용하여 지역적 복잡성 표현을 강화하고, Masked Image Modeling(MIM) 기법을 보조 작업으로 도입하여 복잡성 인식을 향상시킵니다.

- **Performance Highlights**: IC9600 데이터 세트에서 진행된 실험 결과, CLICv2는 기존 비지도 방법보다 PCC(피어슨 상관 계수)와 SRCC(스피어만 상관 계수) 지표에서 현저한 성과를 달성했습니다. CLICv2는 긍정 쌍의 편향을 줄이고, 내용 불변성 있는 복잡성 표현을 구현함으로써 기존 방법들을 초월하는 성능을 입증했습니다.



### CLAD: Constrained Latent Action Diffusion for Vision-Language Procedure Planning (https://arxiv.org/abs/2503.06637)
- **What's New**: CLAD(Constrained Latent Action Diffusion) 모델을 제안하여 시각-언어 절차 계획을 효과적으로 수행합니다. 이 모델은 주어진 시작 및 목표 상태의 시각적 관찰을 기반으로 중간 행동을 예측하는데 초점을 맞추고 있으며, 언어 설명과 결합하여 더욱 고도화된 계획 수립이 가능합니다. 기존의 시각적 데이터만을 사용하는 방법에서 벗어나, CLAD는 다중 모드 입력을 활용하여 절차 계획 문제를 해결합니다.

- **Technical Details**: CLAD는 변분 오토인코더(VAE)를 활용하여 행동과 관찰의 잠재적 표현을 학습하고, 이들을 확산 과정에 통합하여 중간 행동을 생성합니다. 또한, 잠재적 제약을 통해 생성된 행동 시퀀스의 정확성을 높이는 데 기여합니다. 이를 위해, 시작 및 목표 상태의 표현을 잠재 공간에 주입하여 신경망의 가장 깊은 층에서 모델의 생성 과정을 유도합니다.

- **Performance Highlights**: CLAD는 CrossTask, Coin, NIV 데이터셋에서 광범위한 실험을 통해 이전의 최첨단 방법들보다 월등한 성능을 기록했습니다. 실험 결과는 제안된 방법이 기존의 기본 모델들과 비교하여 큰 폭으로 성능 개선이 이루어졌음을 확인시켜 줍니다. 특히, VAE로 학습된 잠재적 제약이 확산 모델이 더 나은 행동 시퀀스를 생성할 수 있도록 돕는다는 점이 중요하게 평가되고 있습니다.



### Towards More Accurate Personalized Image Generation: Addressing Overfitting and Evaluation Bias (https://arxiv.org/abs/2503.06632)
Comments:
          18

- **What's New**: 이번 논문은 사용자 제공 주제를 바탕으로 개인화된 이미지를 생성하는 새로운 훈련 파이프라인을 제안합니다. 이 파이프라인은 훈련 이미지의 방해 요소를 걸러내는 attractor(어트랙터)를 통합하여, 모델이 기본 주제에 집중 행할 수 있도록 지원합니다. 그 결과, 고품질 이미지를 생성하고 사용자 프롬프트(prompt)와의 정합성을 개선합니다.

- **Technical Details**: 제안된 방법은 Latent Diffusion Model (LDM)을 사용하여 텍스트에 조건화된 이미지를 생성합니다. 기존 조정 기반(image personalization) 기술과 원활하게 통합되며, 데이터셋은 훈련 세트와 별도의 테스트 세트를 포함하여 평가의 일관성을 보장합니다. 따라서 기존 자동 평가 프레임워크의 한계를 극복할 수 있는 성능 측정 기준을 제시합니다.

- **Performance Highlights**: 이 연구는 새로운 베인치마크(developed benchmark) 데이터셋을 통해 모델 성능을 신뢰할 수 있게 평가할 수 있도록 합니다. 기존 방법에 비해 주제의 일관성을 개선하고 과적합(overfitting) 위험을 줄이며, 자동 평가 지표의 신뢰성을 향상시킵니다. 이를 통해 사용자 맞춤형 이미지 생성의 정확한 진행 상황을 추적할 수 있습니다.



### DiffCLIP: Differential Attention Meets CLIP (https://arxiv.org/abs/2503.06626)
Comments:
          Under review

- **What's New**: 본 연구에서는 DiffCLIP을 제안합니다. 이는 CLIP 아키텍처에 차별적(attention) 주의 메커니즘을 통합한 새로운 비전-언어 모델입니다. 차별적 주의 메커니즘은 관련된 상황을 강조하고 잡음(noise)을 제거하기 위해 개발되었으며, 본 논문에서는 이를 CLIP의 이중 인코더(dual encoder) 프레임워크에 통합했습니다. 결과적으로 DiffCLIP은 이미지-텍스트 이해 작업에서 탁월한 성능을 거두었음을 보여줍니다.

- **Technical Details**: DiffCLIP은 기본 CLIP 모델에 비해 불필요한 잡음을 제거하여 이미지와 텍스트의 정렬을 더욱 정교하게 만듭니다. 두 개의 주의 맵을 학습하고 하나를 다른 것에서 빼내는 방식으로 작동하며, 이는 모델 파라미터와 연산 비용에 미미한 부하만 추가합니다. 차별적 주의 메커니즘을 활용하여 여러 벤치마크에서 CLIP보다 일관된 성능 향상을 보여주고 있습니다. Experiments on various datasets을 통해 DiffCLIP의 효과를 입증하였습니다.

- **Performance Highlights**: DiffCLIP은 zero-shot 분류(zero-shot classification), 검색(retrieval), 강건성(robustness) 벤치마크에서 기본 CLIP 모델을 꾸준히 능가했습니다. 특히, 성능 향상의 경우 0.003%의 추가 파라미터로 가능하다는 점에서 효율성을 강조합니다. 이 연구는 차별적 주의가 다중 모드(multimodal) 표현을 크게 향상시킬 수 있다는 점을 입증하고 있으며, 이는 향후 연구에 중요한 기초를 제공합니다.



### Similarity-Guided Layer-Adaptive Vision Transformer for UAV Tracking (https://arxiv.org/abs/2503.06625)
- **What's New**: 이번 연구에서는 UAV(무인 항공기) 트래킹을 위해 경량 비전 트랜스포머(ViT) 기반 트래커의 레이어 중 복잡성을 분석하였습니다. 이를 통해 많은 레이어가 중복된 타겟 표현을 학습한다는 사실을 발견하고, 이를 바탕으로 동적인 레이어 비활성화 접근 방식을 제안하였습니다. 이 방법은 효율성을 높이면서도 정확성과 속도 간의 Trade-off를 최적화하는 데 중점을 둡니다. 또한, 이를 통해 새로운 효율적 트래커인 SGLATrack을 개발하였습니다.

- **Technical Details**: SGLATrack은 입력 데이터의 복잡성에 따라 ViT의 레이어를 동적으로 비활성화하는 방법을 사용하여, 성능 저하 없이도 특정 레이어에서 특징을 추출합니다. 이 접근 방식은 MLP(다층 퍼셉트론)를 사용하여 각 후속 레이어의 선택 확률을 출력합니다. 연구팀은 이러한 레이어 선택 모듈에 레이어 간 유사성 손실을 포함하여 모델이 타겟에 더욱 집중하도록 최적화하였습니다. 실험 결과, SGLATrack은 AUC(신뢰성 곡선 아래 면적) 66.9%를 달성하면서 225 FPS(초당 프레임 수)의 뛰어난 속도를 기록하였습니다.

- **Performance Highlights**: SGLATrack은 UAV 추적 벤치마크 6개에서의 실험을 통해 뛰어난 성능을 입증하였습니다. 특히 SGLATrack은 기존 ViT 아키텍처의 효율성을 극대화하여 최신 기술 수준의 실시간 트래킹 속도를 기록하였습니다. 연구진은 SGLATrack이 기존의 경량 ViT 기반 트래커보다 더욱 우수한 성능을 제공하며, 양질의 데이터와의 조합으로 효과적인 결과를 보여준다고 강조하였습니다.



### Chameleon: On the Scene Diversity and Domain Variety of AI-Generated Videos Detection (https://arxiv.org/abs/2503.06624)
Comments:
          17 pages

- **What's New**: 이 논문에서는 AI 생성 비디오 탐지(AI-generated videos detection)의 새로운 데이터셋인 Chameleon을 구축하였습니다. 기존 데이터셋들의 한계를 극복하기 위해 다양한 생성 도구와 실제 비디오 소스를 활용하여 비디오를 생성하고, 장면 전환 및 동적 시각 변화 등 현실 세계의 복잡성을 유지하였습니다. 이를 통해 인물의 표정과 행동 변화, 그리고 환경 생성까지 포함하는 폭넓은 비디오 카테고리를 다루고 있습니다.

- **Technical Details**: Chameleon 데이터셋은 뉴스 방송, 공개 연설 및 제품 추천 비디오 등 세 가지 주요 원본 비디오 유형에서 비디오를 선정하여 구성됩니다. 비디오 세그먼트는 FFmpeg를 사용하여 5초로 나눈 후, 장면 전환 및 도메인 다양성을 기준으로 선별합니다. 최종적으로 600개의 다양한 비디오가 만들어지며, 이는 탐지 알고리즘의 성능을 향상시키기 위한 다양한 벤치마크로 활용됩니다.

- **Performance Highlights**: Chameleon 데이터셋은 기존 AI 생성 비디오 탐지 작업에서 다루지 못했던 복잡한 장면 전환 및 다중 상호작용을 포함하여 훨씬 향상된 탐지 환경을 제공합니다. 이 연구는 AI 생성 데이터셋의 구축과 실세계 포렌식 수요 간의 격차를 해소하는 중요한 기초 자료를 제공하며, AI 생성 컨텐츠의 진화하는 위협에 효과적으로 대응할 수 있도록 돕습니다.



### Transforming Weather Data from Pixel to Latent Spac (https://arxiv.org/abs/2503.06623)
Comments:
          8 pages, 6 figures

- **What's New**: 이번 연구에서는 날씨 데이터를 픽셀 공간(pixel space)에서 잠재 공간(latent space)으로 변환하는 Weather Latent Autoencoder(WLA)를 제안합니다. 이 접근법은 날씨 모델링의 효율성을 향상시키고, 날씨 재구성과 하위 작업(downstream tasks)을 분리하여 결과의 정확성과 선명도를 높입니다. 또한, 압력 변수 통합 모듈(Pressure-Variable Unified Module, PVUM)을 통해 여러 압력 변수의 통합 표현을 제공함으로써 다양한 날씨 시나리오에 대한 모델의 적응성을 개선합니다.

- **Technical Details**: WLA는 다양한 날씨 데이터를 보다 낮은 저장공간 요구로 통합된 잠재 공간으로 인코딩하는 모델입니다. 이를 통해 날씨 모델은 직접 잠재 공간에서 작업을 수행할 수 있으며, 데이터 저장 및 계산 비용을 크게 줄일 수 있습니다. 특히 WLA는 소규모 날씨 구조의 재구성을 고품질로 유지할 수 있도록 사전 훈련된 모델을 활용하며, 유연한 압력 변수 표현을 통해 다양한 날씨 데이터 입력을 수용합니다.

- **Performance Highlights**: 실험 결과, WLA는 ERA5 데이터로부터 244.34 TB의 원본 데이터를 0.43 TB로 압축하는 매우 우수한 성능을 보였습니다. 또한, 낮은 데이터 비용으로 여러 압력 변수 집합에 적용할 수 있는 하위 작업 모델을 통해 픽셀 공간 모델 대비 뛰어난 성능을 달성했습니다. 이와 같은 결과는 날씨 연구에 대한 새로운 가능성을 제시하며, 잠재 공간에서의 작업에 있어 효율성을 크게 증가시킵니다.



### Dynamic Updates for Language Adaptation in Visual-Language Tracking (https://arxiv.org/abs/2503.06621)
- **What's New**: 이번 논문에서는 DUTrack이라는 새로운 시각-언어(VL) 추적 프레임워크를 제안합니다. DUTrack은 다중 모달 참조를 동적으로 업데이트하여 추적 객체의 최신 상태를 캡처하여 일관성을 유지합니다. 이를 통해 기존의 정적 다중 모달 참조에 의존하던 문제를 해결하며, 정확한 추적 결과를 제공합니다.

- **Technical Details**: DUTrack의 핵심은 동적 언어 업데이트 모듈(Dynamic Language Update Module, DLUM)과 동적 템플릿 캡처 모듈(Dynamic Template Capture Module, DTCM)입니다. DLUM은 대형 언어 모델을 이용하여 시각적 특징 및 객체 카테고리에 기반하여 동적 언어 설명을 생성하고, DTCM은 검색 이미지에서 동적 언어 설명과 가장 잘 일치하는 영역을 캡처합니다. 추가적으로 업데이트 전략을 통해 목표의 위치 변화 및 스케일 변화 등을 고려하여 업데이트 빈도를 조정합니다.

- **Performance Highlights**: DUTrack은 LaSOT, TNL2K, GOT-10K 등 네 가지 주요 VL 추적 벤치마크에서 새로운 최고 성능을 달성했습니다. 덕분에 DUTrack은 현재의 시각-언어(VL) 및 시각 전용 추적기와 비교할 때 악기적 경쟁력을 유지하고 있습니다. 실험 결과, 동적 비주얼 및 언어 정보를 결합하여 다중 모달 참조를 업데이트하는 방식이 VL 추적기의 성능을 크게 향상시키는 데 기여함을 보여주었습니다.



### Pixel to Gaussian: Ultra-Fast Continuous Super-Resolution with 2D Gaussian Modeling (https://arxiv.org/abs/2503.06617)
Comments:
          Tech Report

- **What's New**: 본 논문에서는 ContinuousSR 프레임워크를 제안하며, 이는 저해상도(LR) 이미지를 사용하여 2D 연속 고해상도(HR) 신호를 명시적으로 재구성합니다. 기존의 임픽트 신경 표현(Implicit Neural Representation) 기법의 한계를 극복하고, 시간 소모적인 upsampling과 decoding에 대한 의존성을 없애 빠른 응답성을 구현합니다. 이 구조는 Gaussian Splatting을 활용하여 극단적으로 빠른 초해상도 이미지를 생성할 수 있습니다.

- **Technical Details**: ContinuousSR은 Gaussian modeling을 활용해 2D 연속 HR 신호를 재구성하고, 고유의 Deep Gaussian Prior(DGP)를 적용하여 최적화합니다. DGP는 Gaussian 필드 매개변수의 분포를 파악하고, Adaptive Position Drifting 모듈을 통해 Gaussian 커널의 공간적 위치를 동적으로 조정하여 구조적 정확성을 향상시킵니다. 이 혁신적인 접근법은 기존 초해상도 기법에서의 성능을 초월합니다.

- **Performance Highlights**: 이 방법은 7개의 벤치마크에 대해 최첨단 성능을 달성하고, 전통적인 방법에 비해 최대 0.9dB의 재구성 성능 향상을 보였습니다. ContinuousSR은 40 스케일에서 연속적으로 upsampling을 수행하는 동안 19.5배의 속도 향상을 기록하였습니다. 이러한 성과는 실제 응용에 있어 상당한 진전을 나타냅니다.



### GroMo: Plant Growth Modeling with Multiview Images (https://arxiv.org/abs/2503.06608)
Comments:
          7 pages, 5 Figures, 3 Tables

- **What's New**: 이번 연구에서는 식물 성장 동태를 이해하기 위한 Grow Modelling (GroMo) 챌린지를 소개합니다. 주된 두 가지 과제는 식물 나이 예측과 잎 수 추정입니다. 새로운 데이터셋인 GroMo25는 네 가지 작물의 이미지로 구성되어 있으며, 다양한 각도에서 촬영된 다중 시점 이미지를 제공합니다.

- **Technical Details**: GroMo25 데이터셋은 농작물인 밀, 겨자, 무, 그리고 오크라의 이미지를 포함하며, 각 작물은 여러 식물 인스턴스를 가지고 있습니다. 데이터는 24개의 다양한 각도로 촬영된 이미지로 구성되어 있으며, 이를 통해 occlusion 문제를 해결하고 성장 추정의 정확성을 향상시키는 것을 목표로 합니다. MVVT(Multi-view Vision Transformer) 모델은 다중 시점 이미지를 처리하고 복잡한 관계를 캡처하는데 설계되었습니다.

- **Performance Highlights**: MVVT 모델은 식물 나이 예측에서 평균 MAE 7.74, 잎 수 추정에서 MAE 5.52라는 성능을 보여주었습니다. GroMo 챌린지는 식물 성장 추적 및 예측을 위한 혁신적인 솔루션을 장려하며, 공개된 GitHub 리포지토리에서 관련 자료를 확인할 수 있습니다.



### Steerable Pyramid Weighted Loss: Multi-Scale Adaptive Weighting for Semantic Segmentation (https://arxiv.org/abs/2503.06604)
Comments:
          9 pages, 4 figures

- **What's New**: 이 논문에서는 새로운 steerable pyramid 기반의 weighted (SPW) loss function을 제안하고 있습니다. 기존의 weight map 기반 손실 함수들은 사전 계산된 거리 변환(distance transforms)에 의존하였으나, SPW loss는 동적으로 적응 가능한 weight maps를 효율적으로 생성합니다. 이 방법은 멀티스케일 및 멀티 방향 이미지 분해 기술을 활용하여 모델 학습 중의 정교한 구조에 중점을 두고 세분화 정확도를 향상시킵니다.

- **Technical Details**: SPW loss는 ground truth와 네트워크 예측을 기반으로 적응적인 weight maps를 생성합니다. steerable pyramids는 주파수 영역에서 폴라-분리형 분해(polar-separable decomposition)를 수행하여 스케일과 방향을 독립적으로 표현할 수 있습니다. 이 방식을 활용하여 SPW loss는 다양한 주파수 대역에서 지역을 강조하여 세부 구조를 캡처하고 전체 컨텍스트를 유지합니다.

- **Performance Highlights**: 제안된 SPW loss는 SNEMI3D, GlaS, 그리고 DRIVE 데이터셋에서 평가되었으며, 11개의 최첨단 손실 함수와 비교하여 향상된 성능을 보였습니다. SPW loss는 픽셀 정밀도와 세분화 정확도가 뛰어나며, 계산 비용은 최소화된다고 확인되었습니다. 멀티스케일 특성 표현이 필수적인 응용 분야에서 SPW loss는 효과적이고 효율적인 솔루션을 제공합니다.



### StructVPR++: Distill Structural and Semantic Knowledge with Weighting Samples for Visual Place Recognition (https://arxiv.org/abs/2503.06601)
- **What's New**: 이번 논문에서는 구조적 및 의미론적 지식을 RGB 전역 표현에 통합하는 StructVPR++ 프레임워크를 제안합니다. 이 프레임워크는 세분화 이미지에서 파생된 지식을 활용하여 이미지 쌍 간 명확한 의미적 일치를 가능하게 합니다. 또한, 샘플 간의 신뢰성 우선개월 적용하는 가중 지식 증류 전략을 도입하여 훈련의 신뢰성을 높입니다. 이를 통해 기존의 방법들보다 5-23% 높은 성능을 보였습니다.

- **Technical Details**: StructVPR++는 세분화된 이미지에서 구조적 및 의미적 정보를 RGB 이미지로 이전하는 두 단계로 구성된 훈련 방식을 사용합니다. 첫 번째 단계에서는 세분화된 이미지와 RGB 특징 추출을 위한 분리된 브랜치를 훈련시키고, 두 번째 단계에서는 고정된 세그멘테이션 브랜치가 교사 네트워크 역할을 하여 RGB 모델에 지식을 전달합니다. 이 과정에서 표본의 중요도를 동적으로 평가하여 유용한 샘플을 우선시합니다.

- **Performance Highlights**: StructVPR++는 네 개의 데이터셋에서 실험을 진행하였으며, 각 데이터셋에서 기존의 최신 기법 대비 성능이 크게 향상되었습니다. Recall@1 성능에서 5-23%의 절대적인 향상이 있었으며, 실시간 효율성도 개선되었습니다. 이러한 결과는 환경 변화에 강한 특징 추출이 가능함을 시사합니다.



### MultiCo3D: Multi-Label Voxel Contrast for One-Shot Incremental Segmentation of 3D Neuroimages (https://arxiv.org/abs/2503.06598)
Comments:
          13 pages, 6 figures, 6 tables

- **What's New**: 이번 연구에서는 새로운 다중 레이블 voxel 대비 프레임워크인 MultiCo3D를 제안합니다. 이 프레임워크는 One-shot Class Incremental Segmentation (OCIS) 시나리오에서 브레인 트랙의 다중 레이블 분할을 위한 혁신적인 접근 방식을 제공합니다. 기존의 단일 레이블 접근 방식의 한계를 극복하고, 새로운 트랙의 학습 중 특성 겹침 문제를 완화하는 향상된 방법론을 제공합니다.

- **Technical Details**: MultiCo3D는 공유 인코더와 작업별 디코더로 구성된 클래식한 다중 작업 아키텍처를 채택하며, 학습된 기본 트랙 분할 지식을 보존하기 위해 불확실성 증류(uncertainty distillation) 기술을 사용합니다. 또한 다중 레이블 voxel 대비 모듈을 도입하여 특성 정렬을 개선하고 전체 손실을 조정하는 다중 손실 동적 가중치 모듈을 포함하고 있습니다. 이러한 구성 요소들은 다중 레이블 분할 작업의 복잡성을 해결하는 데 기여합니다.

- **Performance Highlights**: 다양한 실험 설정에서 MultiCo3D의 성능을 측정한 결과, HCP 및 Preto 데이터셋에서 One-shot 클래스 증가 트랙 분할 정확도를 유의미하게 향상시키는 결과를 보였습니다. 제안된 방법은 최신 기술(state-of-the-art) 접근법들과 비교하여 뛰어난 성능을 나타냈으며, 이는 특히 브레인 임상 연구 및 수술 개선에 기여할 수 있는 가능성을 제시합니다.



### Introducing Unbiased Depth into 2D Gaussian Splatting for High-accuracy Surface Reconstruction (https://arxiv.org/abs/2503.06587)
- **What's New**: 최근 2D Gaussian Splatting (2DGS) 기술이 3D Gaussian Splatting (3DGS)보다 우수한 기하학적 재구성 품질을 보여주었습니다. 2D surfels를 사용하여 얇은 표면을 근사화함으로써 성능을 향상시킨 것입니다. 특히, glossy 표면에서 발생하는 구멍 문제를 해결하기 위해 depth bias를 도입했습니다. 따라서, 깊이 연속성의 문제를 해결하는 새로운 depth convergence loss를 적용했습니다.

- **Technical Details**: 기술적으로, 우리는 2DGS의 깊이 왜곡 손실을 새로운 depth convergence loss로 대체하였습니다. 이 새로운 손실은 Gaussian 깊이를 연속적이고 매끄럽게 유지하도록 강한 제약을 부과합니다. 또한, 교차하는 Gaussian 수와 축적된 불투명도를 고려하여 표면을 올바르게 결정하는 새로운 기준인 깊이 수정(depth correction)을 도입하였습니다. 이를 통해 깊이의 편향 문제를 해결하고 재구성 품질을 극대화하였습니다.

- **Performance Highlights**: 우리의 방법은 다양한 데이터셋에서 질적 및 양적 평가를 통해 2DGS보다 기하학적 정확성을 획기적으로 향상시킨 것으로 나타났습니다. 특히 glossy 영역에서 더욱 완전하고 정확한 표면을 재구성할 수 있게 되었습니다. 실험 결과 뛰어난 재구성 품질을 보여 줬으며, 이는 새로운 구조화된 깊이 추정 방법을 통해 가능해졌습니다.



### Global-Aware Monocular Semantic Scene Completion with State Space Models (https://arxiv.org/abs/2503.06569)
- **What's New**: 이번 논문에서는 GA-MonoSSC라는 하이브리드 아키텍처를 소개하며, 이는 MonoSSC의 성능을 크게 향상시킵니다. 이 방법은 2D 이미지 도메인과 3D 공간 모두에서 전역적(global) 컨텍스트를 포착하도록 설계되었습니다. 새로운 Dual-Head Multi-Modality Encoder와 Frustum Mamba Decoder를 활용하여 깊이 정보의 손실을 보완하고 더욱 효과적인 3D 표현을 학습할 수 있도록 합니다.

- **Technical Details**: GA-MonoSSC는 Transformer 아키텍처를 이용한 Dual-Head Multi-Modality Encoder를 통하여 2D 이미지 내의 공간적 관계를 포착합니다. 또한, Frustum Mamba Decoder는 State Space Model(SSM)에 기반하여 3D 공간 내에서 장거리 의존성을 효율적으로 처리합니다. 이들 구성 요소는 서로 다른 의미적 및 기하학적 특성을 분리하여 학습함으로써 이전 방법들보다 더 세밀한 표현이 가능합니다.

- **Performance Highlights**: 제안된 GA-MonoSSC는 Occ-ScanNet과 NYUv2 데이터셋에서 최첨단 성능을 달성하였음을 입증하였습니다. 실험 결과는 이 방법이 복잡한 실내 환경에서의 MonoSSC 작업에 효과적이며, 3D에서의 정보 손실 회복을 극대화함을 보여줍니다. 이 연구의 코드는 논문 수락 시 공개될 예정입니다.



### Conceptrol: Concept Control of Zero-shot Personalized Image Generation (https://arxiv.org/abs/2503.06568)
- **What's New**: 이번 논문에서는 개인화된 이미지 생성을 위한 새로운 프레임워크 Conceptrol을 제안한다. 기존의 zero-shot adapter(어댑터)들이 텍스트 프롬프트와 개인화된 콘텐츠 간의 균형을 잘 잡지 못하는 문제를 식별하였다.

- **Technical Details**: Conceptrol은 텍스트 개념 마스크를 사용하여 시각적 명세의 주의(attention) 점수를 조정하며, 이를 통해 개인화된 이미지 생성의 정확성을 높인다. 이는 기존의 IP-Adapter와 OminiControl에 비해 최대 89%의 성능 향상을 보여준다.

- **Performance Highlights**: 종합적인 실험 결과, Conceptrol은 기존의 fine-tuning(미세 조정) 방법보다 뛰어난 성능을 발휘하였다. 특히, DreamBench++를 통한 평가에서 개인화된 이미지 생성을 위한 새로운 기준을 제시하였다.



### Future-Aware Interaction Network For Motion Forecasting (https://arxiv.org/abs/2503.06565)
- **What's New**: 이번 연구에서는 Future-Aware Interaction Network (FINet)을 제안하여 자율주행 시스템에서의 움직임 예측(Motion forecasting) 과제를 개선하고자 하였다. FINet은 미래의 가능한 궤적을 장면 인코딩(scene encoding)에 통합함으로써, 역사적인 궤적과 미래 상태의 동시 최적화를 가능하게 한다. 이로 인해 교통의 포괄적인 표현을 학습할 수 있다.

- **Technical Details**: FINet은 공간적(spatial) 및 시간적(temporal) 모델링을 위해 State Space Model(SSM)인 Mamba를 활용한다. Mamba는 비순차 데이터에 적응하도록 변환하는 Adaptive Reorder Strategy(ARS)를 사용하여, 궤적 생성을 위한 효율성을 높인다. FINet은 모든 교통 참여자가 상호 인식(mutual awareness)을 갖게 함으로써, 일관된 예측을 제공한다.

- **Performance Highlights**: FINet은 Argoverse 1과 Argoverse 2 데이터셋에서 이전 접근 방식에 비해 뛰어난 성능을 입증하며, 처리 효율성과 예측의 정확도를 향상시켰다. 낮은 지연 시간과 최소의 GPU 메모리 사용을 바탕으로 실제 응용에 적합하다. 이 방법은 더 다양한 예측을 가능하게 하고, 현실적인 시나리오의 더 넓은 범위를 아우르는 데 기여한다.



### TR-DQ: Time-Rotation Diffusion Quantization (https://arxiv.org/abs/2503.06564)
- **What's New**: 이 논문에서는 Time-Rotation Diffusion Quantization (TR-DQ)라는 새로운 양자화 방법을 제안합니다. TR-DQ는 샘플링 과정에서 타임스텝의 변동성을 고려하고, 회전 행렬(rotation matrix)을 통해 활성화와 가중치를 동적으로 부드럽게 합니다. 기존 방법들이 놓치고 있던 중요한 활성화의 영향을 반영하여 성능 저하를 최소화하는 데 중점을 두고 있습니다.

- **Technical Details**: TR-DQ는 타임스텝 기반 최적화와 회전 방식을 통합하여 양자화를 진행합니다. 이 과정에서는 각 타임스텝에 대해 전용 하이퍼파라미터를 사용하여 적응형 타이밍 모델링을 가능하게 합니다. 연구는 이미지 및 비디오 생성 작업에서 TR-DQ가 타임스텝을 감안한 양자화를 통해 더 나은 성능을 발휘하도록 도와줍니다.

- **Performance Highlights**: TR-DQ는 이미지 및 비디오 생성 작업에서 최첨단(SOTA) 성능을 달성하며, 기존 양자화 방법 대비 1.38-1.89배의 속도 향상과 1.97-2.58배의 메모리 절약을 이끌어냅니다. 이는 생성 능력을 유지하면서도 효율성을 크게 개선한 결과입니다.



### MMARD: Improving the Min-Max Optimization Process in Adversarial Robustness Distillation (https://arxiv.org/abs/2503.06559)
- **What's New**: 이 논문에서는 Adversarial Robustness Distillation (ARD) 기법의 한계를 극복하기 위한 새로운 방법론인 Min-Max optimization Adversarial Robustness Distillation (MMARD)를 제안하고 있습니다. MMARD는 내적 최적화 과정에서 교사의 강력한 예측을 활용하여 훈련 예제들을 교사의 결정 경계에 더 가깝게 위치하도록 조정합니다. 외적 최적화 과정에서는 자연 및 강력한 시나리오 간의 상관 관계를 통해 모델이 다양한 시나리오를 이해하도록 돕습니다.

- **Technical Details**: 이 연구는 최소-최대 최적화(min-max optimization) 과정에서 합성 대항 예제(synthetic adversarial examples)와 학생 모델의 훈련을 최적화하는 두 가지 과정을 다룹니다. 내적 과정에서는 교사의 예측을 활용하여 훈련 예제를 교사의 결정 경계에 더 가까워지게 하여 강력한 지식을 탐색합니다. 외적 과정에서는 진짜 레이블(true labels), 자연 시나리오의 예측(predictions in natural scenarios), 그리고 강력한 시나리오의 예측(predictions in robust scenarios) 간의 삼각 관계(triangular relationship)를 제안하여 모델의 적응력을 향상시킵니다.

- **Performance Highlights**: MMARD는 여러 벤치마크에서 최첨단 성과를 달성하며, 기존 ARD 방법의 강력한 포화 문제를 극복하는 데에 효과적임을 보여줍니다. 또한, MMARD는 다른 기존 방법들과 쉽게 결합할 수 있는 플러그 앤 플레이(plug-and-play) 특성을 가지고 있습니다. 실험 결과, MMARD는 모델의 강인성을 향상시키고, 각 시나리오에 대한 이해도를 높이는 데 기여하는 것으로 나타났습니다.



### QuantCache: Adaptive Importance-Guided Quantization with Hierarchical Latent and Layer Caching for Video Generation (https://arxiv.org/abs/2503.06545)
Comments:
          The code and models will be available at this https URL

- **What's New**: 최근 Diffusion Transformers (DiTs)는 비디오 생성 분야에서 U-Net 기반 모델들의 성능을 초월하는 주요 아키텍처로 떠올랐습니다. 하지만 이러한 DiTs의 향상된 기능은 계산 비용 및 메모리 요구 사항의 증가로 인해 자원 제한 장치에서의 배치를 제한하는 단점을 동반하고 있습니다. 이에 본 연구에서는 QuantCache라는 새로운 훈련 없이 추론 속도를 향상시키는 프레임워크를 제안합니다.

- **Technical Details**: QuantCache는 계층적 잠재 캐싱, 적응적 중요도 기반 양자화 및 구조적 중복 인식 가지치기를 동시에 최적화합니다. 이 방법은 중복 계산을 줄이면서도 생성 품질을 최소한으로 손실한 채 Open-Sora에서 6.72배의 속도 향상을 달성했습니다. QuantCache는 기존의 정적 휴리스틱에 의존하는 기존 기술의 한계를 극복하고, 동적인 분산 과정에 적응할 수 있는 방법론을 제공합니다.

- **Performance Highlights**: 본 연구의 방법론은 여러 비디오 생성 벤치마크에서 실험을 통해 입증되었으며, 효율적인 DiT 추론을 위한 새로운 표준을 설정했습니다. QuantCache는 특히 캐싱, 양자화 및 가지치기를 공동으로 최적화함으로써, Diffusion Transformers의 표현성을 보존하면서도 계산 비용을 최소화할 수 있는 혁신적인 접근을 제공합니다.



### ARMOR v0.1: Empowering Autoregressive Multimodal Understanding Model with Interleaved Multimodal Generation via Asymmetric Synergy (https://arxiv.org/abs/2503.06542)
- **What's New**: ARMOR은 문자열 및 이미지를 동시에 이해하고 생성할 수 있는 자원 효율적인 자가 회귀(framework) 프레임워크로, 기존의 다중 모달 대형 언어 모델(MLLMs)을 개선합니다. 기존 UniMs 모델은 상당한 컴퓨팅 자원을 요구하며, 복잡한 텍스트-이미지 생성에 어려움을 겪습니다. ARMOR는 비대칭 인코더-디코더 아키텍처를 도입하여 저렴한 비용으로 자연스러운 텍스트-이미지 혼합 생성을 가능하게 합니다.

- **Technical Details**: ARMOR는 모델 아키텍처, 훈련 데이터, 훈련 알고리즘의 세 가지 측면에서 기존 MLLMs를 확장합니다. 특히, 비대칭 이미지 디코더를 도입하여 모델의 이해 능력을 거의 손상시키지 않으면서도 텍스트-이미지 혼합 생성을 가능하게 하며, 데이터셋을 기초로 한 ‘생성 방법’ 알고리즘을 통하여 모델을 점진적으로 훈련합니다.

- **Performance Highlights**: 실험 결과는 ARMOR가 기존 MLLMs를 Superior UniMs로 업그레이드하며, 제한된 훈련 자원으로 높은 이미지 생성 능력을 입증합니다. ARMOR은 9개의 기준에서 기존 모델들보다 뛰어난 성능을 보여주며, 특히 multimodal understanding에서 큰 차이를 만들어냅니다. 최종적으로 ARMOR는 UNI 모델 구축을 위한 자가 회귀 아키텍처의 가능성을 확고히 합니다.



### One-Step Diffusion Model for Image Motion-Deblurring (https://arxiv.org/abs/2503.06537)
- **What's New**: 이번 연구에서는 단일 이미지 디블러링을 위한 새로운 프레임워크 One-Step Diffusion Model for Deblurring (OSDD)를 제안합니다. 기존의 다단계 디노이징 프로세스를 단일 단계로 줄이며, 추론 효율성을 높이면서도 높은 화질을 유지하는 것을 목표로 합니다. 또한, 향상된 변분 오토인코더(Enhanced Variational Autoencoder, eVAE)를 도입해 구조적 복원력도 향상시켰습니다.

- **Technical Details**: 연구에서는 고품질의 합성 디블러링 데이터셋을 구축하였으며, 이를 통해 기존 디퓨전 모델의 감각적 붕괴(perceptual collapse)를 완화하는 방법을 제시합니다. 동적 듀얼 어댑터(Dynamic Dual-Adapter, DDA) 메커니즘을 통해 사전 학습된 두 개의 모델을 동적으로 융합하여, 높은 고유성과 인식 품질을 동시에 개선할 수 있는 구조를 갖추고 있습니다.

- **Performance Highlights**: 많은 실험을 통해 제안된 방법론은 전체 및 비참조(Both full and no-reference) 메트릭에서 뛰어난 성능을 달성하였습니다. 특히, LPIPS와 같은 전체 참조 메트릭에서 우수한 성과를 보였으며, 이는 디퓨전 모델의 일반화 능력을 상당히 향상시킵니다. 그 결과, 높은 품질의 디블러링이 가능해져 실제 응용에서도 효과적으로 사용될 것으로 기대됩니다.



### TimeLoc: A Unified End-to-End Framework for Precise Timestamp Localization in Long Videos (https://arxiv.org/abs/2503.06526)
Comments:
          Code & models will be released at this https URL. The first 4 authors contributes equally

- **What's New**: 이 논문에서는 비디오에서 특정 타임스탬프를 식별하는 Temporal Localization의 중요성을 강조하며, 이를 위해 TimeLoc라는 통합 엔드 투 엔드 프레임워크를 제안합니다. 이 프레임워크는 텍스트 쿼리를 입력으로 받아 여러 작업을 처리할 수 있는 간단하면서도 효과적인 한 단계를 통한 로컬라이제이션 모델을 사용합니다. 템포럴 청킹(temporal chunking) 기술을 통해 30,000 프레임 이상의 긴 비디오를 효율적으로 처리할 수 있도록 하였습니다.

- **Technical Details**: TimeLoc은 비디오 인코더와 로컬라이제이션 모델을 엔드 투 엔드 방식으로 공동 훈련하여 비디오 인식의 정확성을 높입니다. 또한, 미세 조정된 텍스트 인코더를 다단계 훈련 전략을 통해 성능을 더욱 향상시키는 방법을 제시합니다. TimeLoc의 유니버설(Universal) 모델 아키텍처는 다양한 타임스탬프 로컬라이제이션 작업을 지원합니다.

- **Performance Highlights**: TimeLoc은 여러 개의 벤치마크에서 최첨단(SoTA) 성능을 달성하였으며, THUMOS14와 EPIC-Kitchens-100에서 각각 1.3%와 1.9% mAP의 향상을 보였습니다. Kinetics-GEBD에서는 1.1%의 개선을, QVHighlights에서는 2.94%의 mAP 향상을 이루었습니다. 특히, TACoS와 Charades-STA에서 강력한 성능 향상을 기록하며, 영상 그라운딩(grounding)에서 11.5% 및 6.7%의 성능 향상을 보였습니다.



### SGA-INTERACT: A 3D Skeleton-based Benchmark for Group Activity Understanding in Modern Basketball Tactic (https://arxiv.org/abs/2503.06522)
Comments:
          None

- **What's New**: 이 논문에서는 그룹 활동 인식을 위한 새로운 데이터셋 SGA-INTERACT를 소개합니다. 이 데이터셋은 3D 스켈레톤 기반의 벤치마크로, 농구 전술에서 영감을 받은 복잡한 활동을 특징으로 합니다. 또한, Temporal Group Activity Localization (TGAL)이라는 새로운 작업을 도입하여 비편집(uncut) 시퀀스에서 그룹 활동 이해를 확장합니다.

- **Technical Details**: SGA-INTERACT는 고품질의 다중 뷰 모션 캡처(MoCAP) 환경에서 데이터를 수집하여 3D 스켈레톤 기반의 그룹 활동 인식을 위한 첫 번째 벤치마크를 구축합니다. 연구에서는 One2Many이라는 새로운 프레임워크를 제안하여 사전 학습된 3D 스켈레톤 백본을 이용한 통합 개별 특성 추출을 가능하게 합니다. 이 프레임워크는 RGB 기반 방법과의 정합성도 둡니다.

- **Performance Highlights**: SGA-INTERACT에서 수행된 광범위한 평가 결과는 기존 방법들의 일반적인 낮은 성능을 강조합니다. 이는 그룹 활동 이해 작업에서 더 나은 모델링을 위한 도전 과제를 제시합니다. 실험을 통해 SGA-INTERACT이 데이터 수집의 편리함과 더불어 그룹 활동 이해를 위한 중대한 연구 방향을 제공함을 입증하였습니다.



### Seg-Zero: Reasoning-Chain Guided Segmentation via Cognitive Reinforcemen (https://arxiv.org/abs/2503.06520)
- **What's New**: Seg-Zero는 새로운 프레임워크로, 기존의 감독 학습 방식을 넘어 다중 도메인에서 뛰어난 일반화 능력을 보여줍니다. 이 모델은 Reasoning model과 Segmentation model을 분리된 아키텍처로 구성하여, 사용자 의도를 해석하고 명시적인 reasoning chain을 생성합니다. Seg-Zero는 순수한 강화 학습(reinforcement learning)을 통해 훈련되며, 다른 모델들과의 비교에서 인상적인 성능 향상을 보여줍니다.

- **Technical Details**: Seg-Zero는 MLLM(다중모달 대형 언어 모델)을 사용하여 reasoning과 segmentation을 결합한 모델입니다. 이는 사용자의 입력을 바탕으로 bounding box와 픽셀 수준의 포인트를 생성하며, 이를 이용해 정밀한 segmentation mask를 생성합니다. 복잡한 보상 메커니즘이 내장되어 있어, output을 규제하고 reasoning 능력을 향상시키는 효과를 제공합니다.

- **Performance Highlights**: 실험 결과, Seg-Zero-7B는 ReasonSeg 벤치마크에서 57.5의 제로샷(zero-shot) 성능을 기록하며, 이전에 최고 성능을 보였던 LISA-7B보다 18% 향상된 결과를 보여주었습니다. 본 모델은 in-domain 데이터는 물론 out-of-distribution(OOD) 데이터에서도 강력한 성능을 발휘하며, VQA 훈련 데이터 없이도 시각 QA(visual QA) 작업에서 뛰어난 능력을 유지합니다.



### Instance-wise Supervision-level Optimization in Active Learning (https://arxiv.org/abs/2503.06517)
Comments:
          Accepted at CVPR2025

- **What's New**: 이 논문에서는 Instance-wise Supervision-Level Optimization (ISO)라는 새로운 능동 학습 프레임워크를 소개합니다. 이 프레임워크는 단순히 주석을 달 인스턴스를 선택하는 데 그치지 않고, 고정된 주석 예산 내에서 각 인스턴스의 최적 주석 수준을 결정합니다. ISO는 각 인스턴스의 가치 대비 비용 비율(value-to-cost ratio, VCR)을 활용하면서 선택된 인스턴스 간의 다양성을 보장합니다.

- **Technical Details**: ISO는 불확실성 기반 샘플링(uncertainty-based sampling) 및 약한 감독(weak supervision)을 결합한 하이브리드 접근법입니다. 이 방법은 모든 레이블 없는 인스턴스를 세 가지 카테고리, 즉 완전 감독(fully supervised), 약한 감독(weakly supervised), 레이블 없는 상태로 자동 분류합니다. 주석 수준은 주어진 예산 내에서 동적으로 최적화되며, 각 인스턴스에 대해 적절한 주석이 부여됩니다.

- **Performance Highlights**: 실험 결과 ISO는 기존의 전통적인 능동 학습 방법을 지속적으로 능가했으며, 전체 비용이 낮으면서도 더 높은 정확도를 달성했습니다. ISO는 약한 레이블을 활용하여 적은 비용으로도 비슷한 정확도를 유지했으며, 여러 데이터셋에서 주석 예산을 보다 효율적으로 사용한 결과를 보여주었습니다. 최첨단 방법과 비교하여 ISO는 인스턴스별 정보를 활용하여 상당한 비용 효율성을 달성했습니다.



### SAQ-SAM: Semantically-Aligned Quantization for Segment Anything Mod (https://arxiv.org/abs/2503.06515)
- **What's New**: 이번 논문에서는 Segment Anything Model(SAM)의 PTQ(Post-Training Quantization)를 향상시키기 위해 SAQ-SAM을 제안합니다. 기존의 PTQ 방법들이 SAM의 특정 구조와 특징으로 인해 효과적이지 않았던 문제를 해결하고자 합니다. 특히, Perceptual-Consistency Clipping과 Prompt-Aware Reconstruction을 통해 성능 저하를 최소화하면서 정밀한 정량화를 구현합니다.

- **Technical Details**: SAQ-SAM은 SAM의 마스크 디코더에서 발견된 극단 값(outlier)을 조절하기 위해 공격적인 클리핑 방법을 사용합니다. 이를 통해 기존의 MSE와 같은 전통적인 지표로는 달성할 수 없는 효과적인 클리핑을 구현합니다. 또한, 시각적-프롬프트 상호작용을 통합한 Prompt-Aware Reconstruction을 통해 마스크 디코더 내부에서의 상호작용을 최적화하였습니다.

- **Performance Highlights**: SAQ-SAM은 다양한 세그멘테이션 과제를 수행하는 실험에서 기존 방법들보다 우수한 성능을 보였습니다. 예를 들어, 4비트로 양자화된 SAM-B의 경우, 인스턴스 세그멘테이션 작업에서 11.7%의 mAP 향상을 달성했습니다. 이러한 성과는 SAQ-SAM이 시맨틱 정렬을 기반으로 하여 높은 정밀도를 유지함을 보여줍니다.



### A Light and Tuning-free Method for Simulating Camera Motion in Video Generation (https://arxiv.org/abs/2503.06508)
Comments:
          18 pages in total

- **What's New**: LightMotion은 비디오 생성에서 카메라 움직임을 시뮬레이션하기 위한 경량, 조정-free 접근 방식을 제안합니다. 이 방법은 latent space에서 작동하며, 기존 모델들에서 요구되는 추가적인 fine-tuning, inpainting 및 depth estimation을 제거합니다. LighMotion의 주요 혁신점은 permutation operation을 통해 다양한 카메라 움직임을 효과적으로 시뮬레이션할 수 있다는 것입니다.

- **Technical Details**: 이 논문에서는 latent space permutation 작업과 resampling 전략을 통해 새로운 관점을 정확하게 채우면서도 프레임 간 일관성을 유지하는 방법을 설명합니다. 또한, noise를 재도입하여 SNR shift를 완화하고 비디오 생성 품질을 향상시키는 latent space correction 메커니즘을 제안합니다. 이러한 처리 과정은 기존의 복잡한 모델을 사용하지 않고도 효율적인 결과를 제공하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 실험 결과, LightMotion은 기존의 다양한 카메라 움직임을 조정하는 방법들과 비교했을 때 정량적, 정성적으로 더 나은 성능을 보입니다. 또한, 사용자가 정의한 다양한 파라미터 조합을 쉽게 지원하여 비디오 생성 과정의 유연성과 다재다능성을 극대화합니다. 이러한 특성 덕분에 LightMotion은 비디오 생성 분야에서의 널리 보급 가능성을 높이고 있습니다.



### Fine-Grained Alignment and Noise Refinement for Compositional Text-to-Image Generation (https://arxiv.org/abs/2503.06506)
- **What's New**: 본 논문에서는 텍스트 프롬프트에서 발생하는 상세 정보의 누락 및 관계 오류와 같은 복잡한 문제를 해결하기 위한 혁신적인 트레이닝 프리(training-free) 방법을 제안합니다. 이 방법은 텍스트 제약 사항을 적용하기 위해 맞춤형 목표를 통합하여 보다 유연한 장면 구성을 가능하게 합니다. 기존의 레이아웃 기반 방법과 달리, 제안된 접근법은 불필요한 구조적 제약 없이 텍스트에서 추출한 제약만을 반영합니다.

- **Technical Details**: 논문에서는 네 가지 주요 손실 함수를 정의하여 생성 과정을 개선했습니다: (1) object missing loss, (2) object mixing loss, (3) attribute-binding loss, (4) spatial relation loss. 이러한 손실 함수는 최초 생성 단계에서 통합 손실로 적용됩니다. 또한, 생성된 이미지를 평가하고 오류를 수정하는 피드백 기반 시스템이 도입되어 초기 노이즈를 정제하는 역할을 합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 인간 평가에서 24% 향상 및 공간 관계에서 25%의 이득을 보여주었습니다. 초기 노이즈 정제 기법은 효과적이며 성능을 최대 5% 향상시키는 데 기여했습니다. 전체적으로 이러한 개선은 기존 방법들과 비교하여 분명한 우위를 제공하고 있습니다.



### DynamicID: Zero-Shot Multi-ID Image Personalization with Flexible Facial Editability (https://arxiv.org/abs/2503.06505)
Comments:
          17 pages, 16 figures

- **What's New**: 이번 논문에서는 DynamicID라는 새로운 튜닝이 필요 없는 프레임워크를 제안합니다. 이 프레임워크는 단일 ID 및 다중 ID 개인화 생성 모두를 지원하며, 높은 정체성 충실도와 유연한 얼굴 편집성을 보장합니다. 주요 혁신 기술로는 Semantic-Activated Attention (SAA)와 Identity-Motion Reconfigurator (IMR)가 있습니다. SAA는 특정 이미지 Latent 쿼리에 기반하여 ID 특성을 효과적으로 주입하는 메커니즘을 제공하며, IMR은 얼굴 동작과 정체성 특징을 효과적으로 분리하고 재결합할 수 있도록 합니다.

- **Technical Details**: DynamicID는 두 가지 독창적인 디자인으로 구성되어 있으며, 첫 번째는 쿼리 레벨의 활성화 게이팅을 활용하여 ID 특성을 원활하게 주입할 수 있도록 하는 SAA입니다. 두 번째는 IMR로, 이 모델은 대조 학습을 이용하여 얼굴의 동작 및 정체성 특징을 분리한 후 다시 결합할 수 있는 기능을 가지고 있습니다. 이러한 구조는 원래 모델의 행동을 방해하지 않으면서 다중 ID 개인화 생성을 가능하게 합니다. 데이터 세트로는 VariFace-10k를 사용하여 10,000명의 고유한 개인을 대상으로 각각 35개의 개별 얼굴 이미지를 제공합니다.

- **Performance Highlights**: 실험 결과, DynamicID는 정체성과 얼굴 편집 가능성, 다중 ID 개인화 능력에서 기존의 최신 기법들을 초월하는 성능을 보였습니다. 특히, 다중 ID 생성이라는 어려움을 극복하며 높은 충실도를 유지하는 능력이 두드러집니다. 이러한 결과는 DynamicID가 단일 및 다중 ID 시나리오에서 모두 잘 작동한다는 것을 증명합니다.



### TextInPlace: Indoor Visual Place Recognition in Repetitive Structures with Scene Text Spotting and Verification (https://arxiv.org/abs/2503.06501)
Comments:
          8 pages,5 figures

- **What's New**: 텍스트 기반의 VPR(Visual Place Recognition) 기술인 TextInPlace를 제안합니다. 이 프레임워크는 인도어 환경에서의 반복 구조 문제를 해결하기 위해 Scene Text Spotting(STS)을 통합합니다. TextInPlace는 VPR과 STS 두 개의 분기 구조를 채택하여 고유한 글로벌 디스크립터를 생성하고 장면 텍스트를 탐지합니다.

- **Technical Details**: TextInPlace는 로컬 파라미터 공유 네트워크의 이중 분기 아키텍처로 설계되었습니다. VPR 분기는 주로 주의 기반 집합 방식으로 이미지의 글로벌 디스크립터를 추출하고, STS 분기는 장면 텍스트를 탐지하여 인식합니다. 이러한 구조는 연산 효율성을 극대화하기 위해 최적화되었습니다.

- **Performance Highlights**: Maze-with-Text라는 새로운 데이터셋을 구축하여 반복적인 인도어 환경에서의 VPR의 도전 과제를 강조합니다. 실험 결과, TextInPlace는 기존 VPR 방법보다 우수한 성능을 보이며, 특히 반복적인 내부 구조에서의 텍스트 기반 공간 검증의 강인성을 입증하였습니다.



### ExGes: Expressive Human Motion Retrieval and Modulation for Audio-Driven Gesture Synthesis (https://arxiv.org/abs/2503.06499)
- **What's New**: ExGes는 오디오 기반 인간 제스처 생성을 향상시키기 위해 새로운 검색 강화(diffusion) 프레임워크로, 동작 기반 구성(Motion Base Construction), 동작 검색(Motion Retrieval), 정밀 제어(Precision Control) 모듈로 구성됩니다. 본 연구의 핵심은 외부 보조 가이드를 통합하여 표현력이 풍부하고 의미적으로 정합된 제스처를 생성하는 것입니다. 이는 기존 방법의 한계를 극복하고, 다양한 감정 상태 및 개인화된 스타일을 반영할 수 있도록 합니다.

- **Technical Details**: ExGes는 세 가지 주요 모듈로 구성되어 있으며, 각각의 모듈은 특정 기능을 수행하여 전체 시스템의 성능을 향상시킵니다. 동작 기반 구성 모듈은 훈련 데이터셋을 사용하여 제스처 라이브러리를 구축하고, 동작 검색 모듈은 대조 학습(constrative learning)과 모멘텀 증류(momentum distillation)를 통해 세밀한 참고 포즈를 검색합니다. 마지막으로, 정밀 제어 모듈은 부분 마스킹(partial masking)과 확률적 마스킹(stochastic masking)을 통합하여 유연하고 세밀한 제어를 가능하게 합니다.

- **Performance Highlights**: 실험 결과, ExGes는 EMAGE보다 Fréchet Gesture Distance를 6.2% 낮추고, 모션 다양성을 5.3% 향상시키는 것으로 나타났습니다. 사용자 연구에 따르면, ExGes의 자연스러움과 의미적 관련성에 대해 71.3%의 선호도를 보였습니다. 이러한 결과는 ExGes가 기존 방법들보다 더 강력한 성능을 발휘함을 입증하고 있습니다.



### Evaluation of Safety Cognition Capability in Vision-Language Models for Autonomous Driving (https://arxiv.org/abs/2503.06497)
- **What's New**: 이 논문에서는 자율주행 시스템에서 비전-언어 모델(VLM)의 안전성을 평가하기 위해 안전 인지 드라이빙 벤치마크(SCD-Bench)라는 새로운 평가 방법을 제안합니다. 기존 연구가 주로 전통적인 벤치마크 평가에 중점을 둔 반면, SCD-Bench는 VLM이 인간과 상호작용할 때의 안전 인지 능력을 평가하는 데 초점을 맞추고 있습니다. 또한, 대규모 주석 문제를 해결하기 위해 자율주행 이미지-텍스트 주석 시스템(ADA)을 개발하였습니다.

- **Technical Details**: SCD-Bench는 VLM의 안전 인지 능력을 1) 명령 오해, 2) 악의적 결정, 3) 지각 유도, 4) 윤리적 딜레마의 네 가지 차원에서 평가합니다. 이 평가에서는 VLM이 모호한 명령을 처리하고, 악의적인 의도를 인식하며, 지각을 왜곡하는 정보를 피하는 능력을 중점적으로 살펴봅니다. 논문에서 제안하는 자동화된 평가 방법은 LLM을 기반으로 하여 구현되었으며, 본 연구의 자동 평가와 전문가 평가 간의 일치율이 99.74%에 달한다고 보고합니다.

- **Performance Highlights**: 초기 실험 결과에서 기존 오픈 소스 모델은 안전 인지 능력이 부족함을 보였으며, 특히 경량 모델(1B-4B)는 안전 인지에서 미흡한 성과를 보였습니다. 또한, SCD-Bench의 테스트 케이스는 5,043개로 구성되어 있으며, 안전 인지의 측면에서 VLM의 답변 능력을 정량적으로 평가합니다. VLM이 비행 안전을 충족하는 데 있어 도전 과제가 남아 있으며, 이는 경량 모델과 효율성을 유지하는 것의 중요성을 강조합니다.



### PerturboLLaVA: Reducing Multimodal Hallucinations with Perturbative Visual Training (https://arxiv.org/abs/2503.06486)
- **What's New**: 이번 논문은 밀접한 이미지 캡셔닝(dense image captioning) 작업에서 멀티모달 대형 언어 모델(MLLMs)의 환각(hallucination) 문제를 해결하는 것을 목표로 합니다. 기존의 캡션 품질을 측정하는 지표가 부족한 상황에서, 새로운 지표인 HalFscore를 제안하고 이를 통해 캡션의 정확성과 완전성을 정량적으로 평가합니다. 또한, 모델이 언어 선행 언어(prior)에 과도하게 의존한다는 문제를 발견하고, PerturboLLaVA라는 훈련 전략을 통해 이러한 의존성을 줄여 더 신뢰할 수 있는 결과를 만들어냅니다.

- **Technical Details**: HalFscore는 언어 그래프(language graph)를 기반으로 하여 개념 수준에서의 설명 품질을 측정합니다. 이 지표는 캡션의 정확성과 완전성을 평가하며, 잘못된 요소와 누락된 세부 사항을 식별하여 모델 성능을 정량적으로 분석할 수 있습니다. 논문에서는 PerturboLLaVA 방법을 통해 적대적으로 변형된 텍스트를 훈련에 포함시켜 모델이 시각적 입력에 더욱 집중하도록 유도하는 간단하면서 효과적인 방법을 제시합니다.

- **Performance Highlights**: PerturboLLaVA는 부가적인 계산 비용 없이 MLLMs의 환각 문제를 효과적으로 억제하며, 기존의 선진 방법론에 비해 더 뛰어난 성능을 보여줍니다. 이 방법은 일반적인 다중 모달 벤치마크에서 성능 향상 또한 가져오며, 특히 환각이 발생하는 상황에서도 보다 정확하고 신뢰할 수 있는 이미지 기반 설명을 생성하는 데 성공합니다. 결론적으로, HalFscore와 PerturboLLaVA는 MLLMs의 시각적 이해 능력을 높이고, 훨씬 더 효율적이고 확장 가능한 솔루션을 제공합니다.



### A Mesh Is Worth 512 Numbers: Spectral-domain Diffusion Modeling for High-dimension Shape Generation (https://arxiv.org/abs/2503.06485)
- **What's New**: 이 논문은 고차원 형상에서 유래한 잠재 코드를 학습하는 최신 발전을 바탕으로 향상된 3D 생성 모델링 결과를 제시합니다. 특히, 배우지 않는 방식인 스펙트럼 도메인 확산(Spectral-domain diffusion)을 활용하여 고품질의 형상 생성을 위한 SpoDify라는 새로운 프레임워크를 도입합니다. 이 접근법은 고유 벡터(eigenvector)를 저장하여 차후 디코딩에 사용하고, 생성 모델링은 이러한 고유 특성(eigenfeatures)에 대해 수행됩니다.

- **Technical Details**: SpoDify는 형상을 저차원 스펙트럼 특성으로 인코딩하는 학습 없는 파이프라인을 사용하며, 이는 이후 확산 기반 깊은 생성 모델(DGM)의 학습에 사용될 수 있는 잠재 변수로 기능합니다. 이 방법은 복잡한 메시를 연속적 암묵적 표현으로 효율적으로 인코딩하며, 예를 들어 15,000 정점(mesh vertex)을 512차원 잠재 코드로 인코딩할 수 있습니다. 이 방식은 특히 샘플이 제한적이거나 GPU 자원이 부족한 상황에서 상당한 이점을 제공합니다.

- **Performance Highlights**: SpoDify를 적용한 메시 생성 과제에서, 이 방법은 최신 기술들(state-of-the-art)과 비교해 비슷한 품질의 형상을 생성하였습니다. 특히, 512차원 스펙트럼 공간에서 생성 모델링을 수행함으로써 일부 경우에는 오히려 더 우수한 결과를 도출했습니다. 이는 학습 기반 인코더나 대규모 데이터 표현에 의존하지 않고도 가능하다는 점에서 큰 의미가 있습니다.



### Sign Language Translation using Frame and Event Stream: Benchmark Dataset and Algorithms (https://arxiv.org/abs/2503.06484)
Comments:
          In Peer Review

- **What's New**: 이 논문에서는 장애인을 위한 효과적인 의사소통 수단으로서 정확한 수화 이해의 중요성을 강조하고, 전통적인 RGB 카메라의 한계를 극복하기 위해 이벤트 카메라를 활용한 방식으로 수화 번역 알고리즘을 제안한다. 이 연구는 VECSL이라는 대규모 RGB-Event 수화 번역 데이터셋을 수집하여 개발하였으며, 이는 15,676개의 샘플과 2,568개의 중국 문자를 포함하고 있어 다양한 환경에서의 수화 번역의 도전 과제를 반영한다.

- **Technical Details**: VECSL 데이터셋은 DVS346 카메라를 사용하여 수집되며, 다양한 조명 조건과 카메라 이동을 포함하여 다채로운 실내외 환경에서의 샘플을 제공합니다. 저자들은 기존의 SLT 알고리즘을 재훈련하고 평가하여 새로운 벤치마크를 제공하며, RGB-이벤트 기반 수화 번역을 위한 M2-SLT 프레임워크를 제안한다. 이 프레임워크는 미세한 마이크로-사인 및 거친 매크로-사인 탐색 모듈을 포함하여, 텍스트 디코더 mBART를 활용하여 고정밀, 견고한 수화 번역 결과를 도출한다.

- **Performance Highlights**: VECSL 데이터셋을 기반으로 한 M2-SLT 프레임워크는 기존의 수화 번역 알고리즘에 비해 뛰어난 성능을 보여준다. 이 연구는 새로운 데이터셋의 구축과 함께 수화 번역 연구에서 향후 다양한 연구가 진행될 수 있도록 강력한 기반을 제공하는 것을 목표로 한다. 저자들은 이 연구 결과들이 이후 연구에 영감을 줄 수 있다고 믿으며, 데이터셋과 소스 코드를 공개한다고 언급하고 있다.



### PathVQ: Reforming Computational Pathology Foundation Model for Whole Slide Image Analysis via Vector Quantization (https://arxiv.org/abs/2503.06482)
- **What's New**: 본 논문에서는 암 진단을 위한 기계학습 기반 패스톨로지 기술을 개선하기 위한 새로운 방법론을 제안합니다. 특히, 전체 슬라이드 이미지(Whole-Slide Images, WSI) 분석에서 발생하는 초고해상도 문제를 해결하기 위해 패치 토큰을 효과적으로 압축하는 방식을 제시했습니다. 이를 통해 높은 저장공간 비용과 훈련 비용을 줄이면서도 우수한 성능을 유지할 수 있습니다.

- **Technical Details**: 제안된 벡터 양자화(distillation, VQ) 기법은 패치 피처의 차원을 1024에서 16으로 줄이며, 64배의 압축률을 달성하면서도 재구성 충실도를 보존합니다. 또한, 다중 스케일 VQ(multi-scale VQ, MSVQ) 전략을 통해 슬라이드 수준의 자기 지도 학습(Self-supervised Learning, SSL) 감독 타겟을 설정하여 효과적인 프리트레이닝을 가능하게 합니다. 이 과정에서 모든 공간적 패치 토큰을 활용하여 WSIs 분석에서 중요한 정보를 유지합니다.

- **Performance Highlights**: 여러 데이터세트를 대상으로 한 광범위한 평가를 통해 제안된 방법이 WSI 분석에서 최신 성능을 달성했다는 것을 입증했습니다. 암 진단 및 예후 예측에 대한 해석 가능성 및 정확성을 높이는 효과를 보였습니다. 또한, 본 연구에서 제시한 방법론은 향후 임상 응용에 실제적인 영향을 미칠 것으로 기대됩니다.



### PDB: Not All Drivers Are the Same -- A Personalized Dataset for Understanding Driving Behavior (https://arxiv.org/abs/2503.06477)
- **What's New**: 새로운 Personalized Driving Behavior (PDB) 데이터셋은 운전자의 개인적 행동을 포착하기 위해 설계된 다중 모달 데이터셋이다. 기존 데이터셋들이 모든 운전자를 동질적으로 취급하는 반면, PDB는 자연적인 운전 조건 하에서 운전자의 변화를 캡처한다. 이 데이터셋은 일정한 경로, 차량, 조명 조건을 유지하여 외부 영향을 최소화하고, 128라인 LiDAR, 전방 카메라 비디오, GNSS, 9축 IMU, CAN 버스 데이터와 같은 다양한 센서를 통해 수집된 데이터를 포함한다.

- **Technical Details**: PDB 데이터셋은 12명의 참가자로부터 약 270,000개의 LiDAR 프레임과 1.6백만 개의 이미지, 6.6 TB의 원시 센서 데이터를 포함하고 있다. 또한, 각 10초로 분리된 1,669 개의 진행 경로(segment)가 포함되어 있어, 운전자의 행동을 보다 명확하게 연구할 수 있다. 데이터 수집은 모든 세션에서 동일한 차량 모델을 사용함으로써 운전 스타일의 변화를 분석할 수 있게 한다.

- **Performance Highlights**: PDB는 운전자의 개별성과 행동 특성을 연구할 수 있는 중요한 자원으로, 적응형 운전자 보조 시스템 및 행동 인식 기반 경로 예측과 같은 다양한 응용 프로그램에 기여한다. 이 데이터셋은 기존 데이터셋에 비해 더 풍부한 운전자의 특정 데이터를 제공하며, 운전 스타일의 더 포괄적인 표현을 가능하게 한다. PDB의 출시는 인간 중심의 지능형 교통 시스템 발전에 기여할 것으로 기대된다.



### Enhancing Layer Attention Efficiency through Pruning Redundant Retrievals (https://arxiv.org/abs/2503.06473)
Comments:
          11 pages, 7 figures

- **What's New**: 본 논문에서는 깊은 신경망의 층 간 상호작용을 향상시키는 새로운 접근 방식을 소개합니다. 특히, Kullback-Leibler (KL) divergence를 활용하여 인접 층 간의 중복성을 정량화하는 방법을 제안합니다. 이를 통해 기존의 layer attention 메커니즘에서 발생하는 중복 문제를 해결하고, Enhanced Beta Quantile Mapping (EBQM) 알고리즘을 도입하여 중복 층을 효과적으로 건너뛰도록 합니다.

- **Technical Details**: 논문에서 제안하는 Efficient Layer Attention (ELA) 구조는 중복성을 줄여 30%의 훈련 시간 단축과 함께 성능 향상을 이룹니다. KL divergence를 사용하여 각 층의 attention weights의 유사성을 평가하고, 이를 통해 비슷한 성능을 가진 인접 층을 식별합니다. 또한, EBQM 알고리즘은 KL divergence의 분포를 적절하게 조정하여 어떤 층을 제거할지 안정적으로 결정할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, ELA 구조는 다양한 작업에서 최첨단 기법 대비 속도와 정확성을 모두 향상시켰습니다. 예를 들어, 이미지 분류 및 객체 탐지 작업에서 성능이 개선되었으며, 이는 제안된 접근 방식이 실질적인 결과를 제공함을 보여줍니다. 본 연구는 기존의 layer attention 메커니즘의 한계를 극복함으로써 모델의 성능과 훈련 효율성을 동시에 강화하는 데 기여합니다.



### CalliReader: Contextualizing Chinese Calligraphy via an Embedding-Aligned Vision-Language Mod (https://arxiv.org/abs/2503.06472)
Comments:
          11 pages

- **What's New**: 이번 논문에서는 중국 서예의 맥락화 문제인 Chinese Calligraphy Contextualization (CC²)을 해결하기 위한 새로운 비전-언어 모델인 CalliReader를 제안합니다. CalliReader는 세 가지 혁신적 기술을 활용하여 문자 인식의 정확도를 높이는 동시에 데이터 부족 문제를 해결합니다. 이를 통해 기존의 OCR 및 질문-응답(VQA) 시스템의 문제점을 보완하고, 더욱 깊이 있는 이해를 가능하게 합니다.

- **Technical Details**: CalliReader는 문자 단위로 이미지를 나누는 character-wise slicing, 비주얼-텍스트 토큰 압축 및 정렬을 위한 CalliAlign, 데이터 부족을 해소하기 위한 embedding instruction tuning (e-IT) 기술을 통합하고 있습니다. character-wise slicing은 페이지에서 개별 문자를 추출해 인식의 효율성을 높이고, CalliAlign은 시각적 토큰을 정렬하여 대규모 데이터셋에서의 성능을 향상시킵니다. 이러한 방식으로, 모델 훈련의 부담을 줄이면서도 비전-언어 모델의 능력을 극대화할 수 있습니다.

- **Performance Highlights**: CalliReader는 기존의 최고 수준의 모델들과 비교하여 페이지 단위의 서예 인식 및 해석에서 더 높은 정확도를 달성했습니다. 이 논문에서 수행된 다양한 실험 결과는 CalliReader가 원주율을 능가하며, 심지어 서예 전문가들의 성과를 초월하는 것으로 나타났습니다. 사용자 연구 결과는 CalliReader의 우수성을 보여주며, 문서 및 홀로그램 데이터에 대한 적응력도 뛰어난 것으로 평가받고 있습니다.



### Online Dense Point Tracking with Streaming Memory (https://arxiv.org/abs/2503.06471)
- **What's New**: 이 논문에서는 SPOT 프레임워크를 통해 밀집 포인트 트래킹을 위한 새로운 경량 모델을 제시합니다. SPOT는 비디오 처리에서 효율적인 정보 전달을 가능케 하며, 기존 알고리즘들이 가지는 외관의 이동 문제와 시간적 일관성을 고려할 수 있습니다. 기존의 슬라이딩 윈도우 접근 방식 대신에, 온라인으로 밀집 포인트 트래킹을 수행하는 데 필요한 안정성과 실시간 성능을 제공합니다.

- **Technical Details**: SPOT은 세 가지 핵심 구성 요소로 이루어져 있습니다: (1) 특성 향상을 위한 맞춤형 메모리 읽기 모듈, (2) 단기 운동 동역학 모델링을 위한 감각 메모리, (3) 정보 전파를 위한 가시성 기반 스플래팅 모듈입니다. 이 구조는 높은 시간 효율성과 정확성을 제공하며, 10배 더 적은 파라미터 수로 이전 모델보다 최소 2배 더 빠른 속도를 자랑합니다.

- **Performance Highlights**: CVO 벤치마크에서 SPOT는 최첨단 성능을 달성하였으며, TAP-Vid 및 RoboTAP과 같은 희박한 트래킹 벤치마크에서도 비교적 우수한 성능을 보입니다. 특히, SPOT은 12.4 FPS의 속도로 512x512 비디오를 밀집하게 트래킹하며, GPU 메모리를 4.15GB 소모합니다. 이러한 성능 개선은 분명한 설계 선택과 세밀한 구조 덕분입니다.



### Vector Quantized Feature Fields for Fast 3D Semantic Lifting (https://arxiv.org/abs/2503.06469)
- **What's New**: 이 논문에서는 per-view 마스크를 도입하여 lifting을 semantic lifting으로 일반화하는 방법을 제안합니다. 이 마스크는 멀티스케일 픽셀 정렬 피쳐 맵에서 쿼리하여 결정되며, 이는 2D 이미지를 3D로 변환하는 데 도움을 줍니다. 이를 통해 텍스트 기반의 지역 편집 및 임베디드 질문 응답의 효율성을 개선합니다.

- **Technical Details**: Vector-Quantized Feature Field(VQ-FF)가 도입되어 픽셀 정렬된 마스크를 경량으로 검색할 수 있는 방법을 제공합니다. 이 기법은 이미지의 각 부분에 relevance 마스크를 할당하여 semantic lifting을 실현할 수 있게 하고, 복잡한 장면에서도 효과를 보여줍니다. VQ-FF는 메모리 사용량을 극적으로 줄이면서도 필요한 의미적 충실도를 유지합니다.

- **Performance Highlights**: 이 시스템은 LERF 데이터셋의 다양한 장면을 통해 평가되었으며, compact representation이 쿼리의 충실도를 유지하거나 개선할 수 있음을 입증하였습니다. semantic lifting과 함께 사용되었을 때, VQ-FF는 즉각적인 객체 감지 및 효율적인 EQA를 가능하게 합니다. 이는 3D 장면의 이해와 편집을 혁신적으로 변화시킬 것으로 기대됩니다.



### SP3D: Boosting Sparsely-Supervised 3D Object Detection via Accurate Cross-Modal Semantic Prompts (https://arxiv.org/abs/2503.06467)
Comments:
          11 pages, 3 figures

- **What's New**: 이 논문에서는 SP3D라는 부스팅 전략을 제안하여, 밀도가 낮은 주석 환경에서도 3D 객체 탐지의 성능을 향상시킵니다. SP3D는 대규모 다중 모달 모델(LMMs)에서 생성된 크로스 모달 의미 프롬프트를 활용하여 3D 탐지기의 성능을 강화하는 데 중점을 둡니다. 이는 적은 수의 주석으로도 고차원의 특징 식별 능력을 자랑합니다.

- **Technical Details**: SP3D는 두 단계로 구성된 훈련 전략으로, 첫 번째 단계에서는 LMMs를 통해 2D 이미지로부터 의미를 추출하고 이를 3D 포인트 클라우드에 전이하여 가짜 레이블을 생성합니다. 두 번째 단계에서는 이 가짜 레이블을 기반으로 3D 탐지기를 미세 조정(fine-tuning)하며, Confident Points Semantic Transfer (CPST) 모듈과 Dynamic Cluster Pseudo-label Generation (DCPG) 모듈이 그 과정을 지원합니다.

- **Performance Highlights**: 실험 결과, SP3D는 KITTI와 Waymo Open Dataset(WOD)에서 sparsely-supervised 탐지기의 성능을 크게 향상시켰습니다. 또한 SP3D는 레이블이 없는 상태에서도 기존의 최신 방법들을 초월하는 성능을 보였습니다. 이러한 결과는 SP3D가 효율적이고 신뢰할 수 있는 탐지기 초기화를 제공함을 의미합니다.



### StructGS: Adaptive Spherical Harmonics and Rendering Enhancements for Superior 3D Gaussian Splatting (https://arxiv.org/abs/2503.06462)
- **What's New**: 최근 3D 복원 기술의 발전과 함께 신경 렌더링 기법이 사진 실사 같은 3D 장면 생성에 크게 기여하고 있으며, 이는 학술 및 산업 응용 분야에 많은 영향을 미치고 있습니다. 본 연구에서는 3D Gaussian Splatting(3DGS)을 향상시킨 StructGS라는 새로운 프레임워크를 소개하며, 이는 효과적인 새로운 시점 합성을 가능하게 합니다. StructGS는 패치 기반 SSIM 손실, 동적 구형 함수 초기화 및 다중 스케일 잔차 네트워크(MSRN)를 혁신적으로 통합하여 기존 모델들이 가지고 있는 한계점을 해결하고 있습니다.

- **Technical Details**: StructGS는 고해상도 이미지를 생성할 수 있도록 3DGS의 성능을 향상시키는 프레임워크입니다. 이 기술은 비선형 구조 유사성을 효과적으로 캡처하기 위해 패치 SSIM 손실을 사용하고, Gaussian 구의 투명도를 고려하여 구형 함수의 초기화 및 최적화를 위한 동적 조정 전략을 설계합니다. 또한, 미리 훈련된 MSRN을 통합하여 저해상도 입력 이미지로부터 고해상도의 고품질 이미지를 생성할 수 있습니다.

- **Performance Highlights**: StructGS는 최신 3DGS 기반 모델들보다 우수한 성능을 보이며, 훈련 반복 횟수를 줄이면서 더 나은 품질을 달성하고 있습니다. 또한, 설계된 손실 함수와 MSRN을 통해 고해상도 이미지의 렌더링 품질이 향상된다는 점이 실험적으로 입증되었습니다. 이를 통해 더 정교한 텍스처와 복잡한 기하학을 처리할 수 있는 능력을 갖추게 되었습니다.



### Long-tailed Adversarial Training with Self-Distillation (https://arxiv.org/abs/2503.06461)
Comments:
          ICLR 2025

- **What's New**: 이 연구에서는 불균형 데이터셋(imbalanced dataset)에서의 적대적 훈련(adversarial training) 성능 문제를 다루고 있습니다. 기존의 연구들은 주로 균형 잡힌 데이터셋에서 성능 검증을 하여 불균형 데이터에 대한 적대적 강건성(adversarial robustness)에 대한 공백을 드러냈습니다. 저자는 제안한 새로운 두 단계의 프레임워크를 통해 긴 꼬리 분포(long-tailed distribution)에서 비율이 낮은 클래스의 성능을 향상시키려는 노력을 하고 있습니다.

- **Technical Details**: 이 논문은 전통적인 긴 꼬리 분포의 분류 기술을 활용하여 새로운 자기 증류 기법(self-distillation technique)을 적용한 적대적 훈련 방법을 제안합니다. 먼저, 원본 데이터셋에서 균형 잡힌 하위 데이터셋을 구축하고, 이를 사용하여 자가 교사 모델(self-teacher model)을 훈련합니다. 그런 다음, 이 균형 잡힌 모델을 활용하여 자기 증류(self-distillation)를 적용함으로써 긴 꼬리 클래스의 성능을 크게 향상시키는 방식입니다.

- **Performance Highlights**: 저자들은 CIFAR-10, CIFAR-100, Tiny-ImageNet 데이터셋에서 긴 꼬리 클래스에 대한 정확도(tail class accuracy)가 각각 20.3, 7.1, 3.8% 향상되었다고 보고합니다. 이 방법은 AutoAttack에 대해 최고의 정확도를 달성하며 긴 꼬리 클래스의 성능 개선이 두드러집니다. 이를 통해 제안된 방식이 기존의 적대적 훈련 방법에 비해 더욱 효과적임을 입증하고 있습니다.



### Reconstructing Depth Images of Moving Objects from Wi-Fi CSI Data (https://arxiv.org/abs/2503.06458)
- **What's New**: 이 연구는 Wi-Fi 채널 상태 정보(CSI)를 사용하여 특정 영역 내에서 움직이는 물체의 깊이 이미지를 재구성하는 새로운 딥 러닝 방법인 Wi-Depth를 제안합니다. 본 연구에서는 보안 및 노인 돌봄과 같은 분야에서 Wi-Fi 기반 깊이 이미징 기법의 혁신적인 응용 가능성을 강조합니다. Wi-Depth는 깊이 이미지 재구성 작업에서 표면, 깊이, 위치의 세 가지 핵심 정보를 동시에 추정할 수 있도록 설계되었습니다.

- **Technical Details**: Wi-Depth는 변분 오토인코더(VAE) 기반의 교사-학생 구조를 이용하여 깊이 이미지를 세 가지 핵심 구성 요소로 분해하는 것을 기본 아이디어로 삼고 있습니다. 이를 통해 높은 차원의 CSI 데이터에서 깊이 이미지로의 매핑을 효율적으로 학습할 수 있습니다. 세부적으로, 깊이 이미지는 물체의 표면, 깊이 및 위치로 구성되며, 이 정보는 CSI로부터 유도된 기본 정보와 관련이 있습니다.

- **Performance Highlights**: 제안된 모델의 유효성이 네 가지 실제 환경에서 검증되었으며, 이는 Wi-Fi CSI로부터 움직이는 물체의 깊이 이미지를 추정하는 최초의 작업으로 볼 수 있습니다. 또한 VAE 기반의 교사-학생 네트워크를 통해 일관된 깊이 이미지를 추정하는 효율적인 아키텍처가 제안되었습니다. 이 방법은 깊이 이미지의 품질을 높이는 데에 기여하며, 기존 Wi-Fi 이미징 연구의 한계를 극복합니다.



### Geometric Knowledge-Guided Localized Global Distribution Alignment for Federated Learning (https://arxiv.org/abs/2503.06457)
Comments:
          Accepted by CVPR 2025

- **What's New**: 이번 연구에서는 Federated Learning (FL)에서의 데이터 이질성 문제를 해결하기 위해 새로운 데이터 생성 방법을 제안합니다. 이 방법은 로컬에서 글로벌 임베딩 분포를 시뮬레이션하여 데이터의 불균형 문제를 완화하고자 합니다. 특히, 레이블 스큐(label skew)와 도메인 스큐(domain skew)가 공존하는 상황에서도 모델의 성능을 향상시킬 수 있는 접근법을 제시합니다.

- **Technical Details**: 이 연구에서는 데이터 분포의 기하학적 형태를 개념화하고, 이를 통해 로컬과 글로벌 데이터 분포 간의 불일치를 해결하는 데 중점을 둡니다. GGEUR(Global Geometry-Guided Embedding Uncertainty Representation) 방법을 통해 로컬 클라이언트에서 신규 샘플을 생성하고, MLP를 통해 해당 데이터로 훈련할 수 있습니다. 또한, 다중 도메인 시나리오에서도 각 카테고리의 기하학적 분포가 유사하다는 점을 이용하여 글로벌 기하학적 형태를 근사화합니다.

- **Performance Highlights**: 실험 결과, 본 연구에서 제안한 방법이 레이블 스큐와 도메인 스큐 상황에서 기존 접근법의 성능을 크게 개선함을 보여줍니다. 이러한 개선은 Highly Heterogeneous Data 시나리오에서 최신 성과(state-of-the-art results)를 달성하는 데 기여합니다. 이 연구는 시각적 모델과 기하학적 지식을 결합하여 FL 분야에서의 시너지를 보여주는 전형적인 사례로 평가됩니다.



### DynCIM: Dynamic Curriculum for Imbalanced Multimodal Learning (https://arxiv.org/abs/2503.06456)
- **What's New**: 이번 논문에서는 DynCIM이라는 동적 커리큘럼 학습 프레임워크를 소개합니다. 이 프레임워크는 샘플 및 모달리티 관점에서 본질적인 불균형을 정량화하며, 학습 과정에서 동적으로 각 샘플의 난이도를 평가할 수 있도록 설계되었습니다. 또한, 모달리티 기여를 조정하는 게이팅 기반의 동적 융합 메커니즘을 도입하여 불필요한 중복을 줄이고 융합 효과성을 최적화합니다.

- **Technical Details**: DynCIM은 샘플 레벨과 모달리티 레벨의 커리큘럼을 활용하여, 각 샘플의 예측 편차, 일관성 및 안정성에 따라 난이도를 평가합니다. 모달리티 레벨의 커리큘럼은 글로벌 및 로컬 관점에서 모달리티의 기여를 측정합니다. 동적 weighting 전략을 통해 변동성이 큰 메트릭을 중시하여 컨디션에 따라 샘플 평가의 중요한 측면을 강조합니다.

- **Performance Highlights**: DynCIM은 빔모달 및 트림모달 시나리오를 포함하는 여섯 가지 멀티모달 벤치마크 데이터셋에서 최첨단 방법론을 지속적으로 초과하는 성능을 보여주었습니다. 이 연구는 멀티모달 학습 작업에서 모달리티와 샘플의 불균형을 완화하고, 적응성과 견고함을 향상시키는 데 효과적임을 입증했습니다.



### A Quantitative Evaluation of the Expressivity of BMI, Pose and Gender in Body Embeddings for Recognition and Identification (https://arxiv.org/abs/2503.06451)
- **What's New**: 이번 연구에서는 사람 재식별(Person Re-identification, ReID) 모델의 특성 표현(encoding) 및 표출(expressivity)을 분석하여 신체 관련 특성이 모델의 성능에 미치는 영향을 평가했습니다. 특히, BMI(Body Mass Index)와 같은 민감한 속성이 ReID 모델의 특성 발현에서 어떻게 중대한 역할을 하는지를 강조합니다. 이 연구는 ReID 시스템의 편향(bias)을 감소시키고 공정성(fairness)을 강화하기 위한 새로운 방법론을 제시합니다.

- **Technical Details**: 이 연구에서 표출은 특성 벡터(feature vector)와 특정 속성(attribute) 간의 상호정보량(mutual information)으로 정의되어, 2차 신경망을 이용하여 계산됩니다. SemReID라는 자가 감독(self-supervised) ReID 모델을 통해, 신체 속성의 표출 순서를 BMI > Pitch > Yaw > Gender로 도출하여, 신체 속성이 네트워크의 예측에 미치는 상대적 중요성을 나타냅니다. 이를 통해 신체 속성이 ReID 성능에 미치는 영향을 동적으로 분석했습니다.

- **Performance Highlights**: SemReID 네트워크의 마지막 주의 레이어에서 신체 속성의 표출 순위는 BMI, Pitch, Yaw 및 Gender입니다. 이는 각각의 속성이 예측에 미치는 영향의 차이를 나타내며, ReID 모델의 성능을 향상시키기 위한 효율적인 인사이트를 제공합니다. 또한, 각 네트워크 레이어와 훈련 과정에서의 특성-속성 상관관계의 변화를 분석함으로써, 모델의 해석 가능성을 높이고 알고리즘의 잠재적 편향을 파악하는 데 기여하고 있습니다.



### M$^3$amba: CLIP-driven Mamba Model for Multi-modal Remote Sensing Classification (https://arxiv.org/abs/2503.06446)
- **What's New**: M$^3$amba는 CLIP 모델을 기반으로 한 최신 다중 모달 융합(end-to-end CLIP-driven Mamba model) 모델로, 기존의 기법에서 발생하는 의미적 정보 부족과 낮은 계산 효율성 문제를 해결합니다. CLIP의 강력한 의미 정보 추출 능력을 활용하여, 모달별 적응기(modality-specific adapters)를 도입하여 각 도메인에 대한 이해 편향을 최소화합니다. 이는 다중 모달 정보 간의 완전한 융합을 위한 통합 프레임워크를 제공합니다.

- **Technical Details**: M$^3$amba는 CLIP 이미지 인코더와 모달 전용 적응기를 통해, 최소한의 훈련으로 다양한 모달의 종합적인 의미적 이해를 달성합니다. Cross-SS2D 모듈을 포함한 다중 모달 Mamba 융합 아키텍처는 선형 복잡성을 기반으로 하여 효율적인 정보 상호작용을 고려합니다. 이를 통해 각 모달의 의미 연결성을 강화하고, 훈련 시 효율성을 극대화했습니다.

- **Performance Highlights**: M$^3$amba는 원거리 센싱 분야의 다중 모달 하이퍼스펙트럼 이미지 분류 작업에서 최소 5.98%의 평균 성능 향상을 달성하며, 훈련 효율성에서도 뛰어난 결과를 보여줍니다. 이는 정확도와 효율성을 동시에 두 배 개선한 것을 나타내며, CLIP 및 Mamba 기반의 통합 프레임워크의 큰 잠재력을 나타냅니다.



### OT-DETECTOR: Delving into Optimal Transport for Zero-shot Out-of-Distribution Detection (https://arxiv.org/abs/2503.06442)
Comments:
          The first two authors contributed equally to this work

- **What's New**: 본 논문은 현실 세계의 머신 러닝 모델의 신뢰성과 안전성을 보장하기 위해 매우 중요한 Out-of-Distribution (OOD) 탐지 기술에 대해 다루고 있습니다. 특히, 제로샷(zero-shot) OOD 탐지가 클립(CLIP)과 같은 비전-언어 모델의 등장과 함께 새로운 가능성을 보이게 되었지만, 기존 방법들은 주로 의미적 매칭(sementic matching)에 집중하며 분포 차이(distributional discrepancies)를 완전히 포착하지 못하고 있습니다. 이러한 제한 사항을 해결하기 위해 저자들은 Optimal Transport (OT)를 활용한 새로운 프레임워크인 OT-DETECTOR를 제안합니다.

- **Technical Details**: OT-DETECTOR는 테스트 샘플과 ID 레이블 간의 의미적 차이 및 분포적 차이를 정량화하는 것을 목표로 합니다. 이를 위해 상호 모달 교통량(cross-modal transport mass)과 교통 비용(transport cost)을 각각 의미적 및 분포 점수로 도입하여 OOD 샘플의 탐지를 보다 강력하게 수행할 수 있습니다. 또한, 저자들은 ID 레이블의 의미적 단서를 활용하여 ID 샘플과 어려운 OOD 샘플 간의 분포 차이를 증폭하는 Semantic-aware Content Refinement (SaCR) 모듈을 제시합니다.

- **Performance Highlights**: 다양한 벤치마크에서의 광범위한 실험을 통해 OT-DETECTOR는 여러 OOD 탐지 작업에서 최첨단 성능을 달성했음을 보여줍니다. 특히, 어려운 hard-OOD 시나리오에서 두드러진 성과를 보였으며, 이는 이 프레임워크가 더욱 견고하게 OOD 샘플을 탐지할 수 있도록 설계되었음을 시사합니다.



### SEED: Towards More Accurate Semantic Evaluation for Visual Brain Decoding (https://arxiv.org/abs/2503.06437)
Comments:
          Under Review

- **What's New**: SEED(Se mantic E valuation for Visual Brain D ecoding)는 시각 뇌 디코딩 모델의 성능 평가를 위한 새로운 메트릭입니다. 세 가지 보완 메트릭을 통합하여 이미지 간의 의미적 유사성을 다양한 측면에서 포착합니다. SEED는 기존 메트릭보다 인간 평가와 더 높은 정합성을 보여주며, 중요한 정보가 종종 손실되는 현상을 밝히고 있습니다.

- **Technical Details**: SEED는 Object F1과 Cap-Sim이라는 두 가지 새로운 메트릭과 EffNet을 통합하여 GT(ground-truth)와 재구성 이미지 간의 의미적 유사성을 평가합니다. Object F1은 MM-Grounding-DINO를 사용하여 이미지의 주요 객체를 자동으로 식별하고 캡션 유사도는 GIT로 생성된 캡션을 활용하여 평가합니다. 이 메트릭들은 해석이 용이하고 독립적으로 이미지 품질을 평가할 수 있습니다.

- **Performance Highlights**: SEED는 기존 메트릭에 비해 인간 평가와의 정합성이 가장 높으며, 최신 디코딩 모델조차도 핵심 객체를 정확히 재현하지 못한다는 것을 밝혔습니다. 발표된 새로운 손실 함수는 CLIP 이미지 임베딩과 GT 임베딩 간의 정렬을 개선하여, 현재 메트릭과 SEED 모두에서 성능을 향상시킵니다. 이 연구는 훨씬 발전된 뇌 디코딩 모델의 평가 방법 개발을 장려하기 위해 인간 평가 데이터를 오픈소스합니다.



### OV-SCAN: Semantically Consistent Alignment for Novel Object Discovery in Open-Vocabulary 3D Object Detection (https://arxiv.org/abs/2503.06435)
- **What's New**: 이번 연구에서는 OV-SCAN이라는 새로운 오픈 어휘(Open-Vocabulary) 3D 오브젝트 탐지 프레임워크를 제안합니다. OV-SCAN은 새로운 객체 발견을 위한 의미적으로 일관된 정렬(Semantically Consistent Alignment)을 강화하는 방식으로 개발되었습니다. 이 시스템은 복잡한 3D 데이터와 텍스트 간의 변환 오차를 줄이기 위해 3D 주석의 정확성을 발견하고 불량한 정렬 쌍을 필터링합니다.

- **Technical Details**: OV-SCAN은 SC-NOD(Semantically-Consistent Novel-Object Discovery) 모듈을 통해 고유한 3D 박스 검색을 비선형 최적화 문제로 재구성하여 2D 제안을 기반으로 3D 경계 상자 매개변수를 효율적으로 최적화합니다. 또한, H2SA(Hierarchical Two-Stage Alignment) 헤드를 통해 객체들의 계층적 분류를 구현하여, 세부 클래스 간의 정밀한 정렬을 가능하게 합니다. 이러한 고급 기술들은 자율 주행 환경에서의 객체 인식을 향상시키는 데 기여합니다.

- **Performance Highlights**: nuScenes 데이터셋을 사용한 실험 결과, OV-SCAN은 기존의 오픈 어휘 3D 탐지 방법들보다 우수한 성능을 보여주었습니다. 이 연구는 특정 클래스에 대한 별도의 인간 주석 데이터 없이도 높은 탐지 성능을 달성함으로써, 자율 주행 시스템의 안전성과 효율성을 향상시킬 수 있는 가능성을 제시합니다.



### Consistent Image Layout Editing with Diffusion Models (https://arxiv.org/abs/2503.06419)
- **What's New**: 이번 논문에서는 기존의 이미지 레이아웃 편집 방법의 한계를 극복하기 위해 새로운 두 단계 이미지 레이아웃 편집 방법을 제안합니다. 이 방법은 실제 이미지를 특정 레이아웃으로 재배치할 수 있을 뿐만 아니라 편집 전과 객체의 시각적 외관을 일치시키는 것도 가능합니다. 또한, 초기 노이즈 설계를 통해 레이아웃 조정 과정이 용이해지는 혁신적인 방법을 포함하고 있습니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 구성 요소로 나뉩니다. 첫째, 다중 개념 학습(mult-concept learning) 체계를 활용해 단일 이미지에서 다양한 객체의 개념을 학습하여 레이아웃 편집의 시각적 일관성을 유지합니다. 둘째, 확산 모델의 중간 기능에서의 의미론적 일관성(semantic consistency)을 활용하여 객체의 외관 정보를 직접적으로 원하는 영역으로 투영하는 방식으로 작업합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 레이아웃 정렬(layout alignment) 및 시각적 일관성(visual consistency) 측면에서 기존 방법들을 능가함을 보여주었습니다. 더불어, 이 연구는 이미지 레이아웃 편집을 위한 첫 번째 공개 데이터셋(Layout-Bench)을 수집하였으며, 이를 통해 제안된 방법의 효과성을 평가합니다.



### Polygonal network disorder and the turning distanc (https://arxiv.org/abs/2503.06415)
- **What's New**: 이번 연구에서는 다각형 평면 네트워크의 'turning disorders' 개념을 도입하였으며, 이는 네트워크 면들 사이의 turning distance 평균을 기반으로 정의됩니다. 이를 통해 규칙적인 다각형이나 원과 같은 'ordered shapes'와 비교를 통해 네트워크의 무질서도를 측정합니다. 연구는 다양한 통계적 프로세스에서 이러한 무질서도를 탐구하며, 결과적으로 네트워크의 복잡성을 이해하는 데 기여하고 있습니다.

- **Technical Details**: 연구에 사용된 turning distance는 두 다각형간의 경계에 대한 접선 벡터의 각도를 추적하는 step functions를 활용하여 구성됩니다. 이때, L2 노름을 사용할 경우, 'turning distance'를 계산하는 데 소요되는 시간 복잡도는 O(mnlog(m+n))로 중복성을 줄였습니다. 연구에서는 같은 유형의 면을 가진 다각형에 대해 turning distance 계산의 연산량을 O((m+n)log(m+n))으로 축소할 수 있는 조밀한 공식을 도출했습니다.

- **Performance Highlights**: 실제 사례를 통해 Archimedean lattices와 같은 정규 타일링 구조에서 얻은 turning distance의 정확한 수식을 제공하였으며, 이는 나노 스케일 과학의 여러 문제와도 연결될 수 있음을 보여줍니다. 또한, 스프링 네트워크와 같은 확률적 프로세스에 'turning disorders'를 적용함으로써, 네트워크의 지오메트릭 특성을 심층적으로 분석할 수 있는 가능성을 열었습니다. 본 연구는 무질서도의 측정이 네트워크 구조 분석에 중요한 역할을 할 수 있음을 시사합니다.



### FEDS: Feature and Entropy-Based Distillation Strategy for Efficient Learned Image Compression (https://arxiv.org/abs/2503.06399)
Comments:
          16 pages

- **What's New**: 이번 논문에서는 Swin-Transformer V2 기반의 attention 모듈과 추가적인 residual block을 통합하여 높은 용량의 teacher 모델을 구축함으로써, 기존의 learned image compression (LIC) 방법들이 가진 단점을 극복하려고 합니다. 이를 통해 효율적인 압축 성능을 달성하고, lightweight student 모델로의 효과적인 knowledge transfer를 위한 FEDS (Feature and Entropy-based Distillation Strategy)를 제안합니다. 이 방법은 teacher 모델의 핵심 지식을 student 모델에 전이하는 동시에, 복잡성을 크게 줄입니다.

- **Technical Details**: FEDS의 핵심 요소는 intermediate feature representations의 정렬과 가장 유용한 latent channel을 강조하는 entropy 기반 손실(loss)을 통한 것입니다. 모델의 훈련 과정은 세 단계로 나뉘며, feature alignment, channel-level distillation, 및 최종 fine-tuning을 포함합니다. 이러한 접근 방식은 student 네트워크가 teacher 네트워크의 압축 성능을 거의 동일하게 유지하면서, 약 63%의 파라미터를 줄이고 인코딩/디코딩 속도를 약 73% 증가시킵니다.

- **Performance Highlights**: Kodak, Tecnick, CLIC 데이터셋에서 실험 결과, student 모델은 teacher 모델과 거의 동일한 성능을 유지하면서도 압축 효율성을 높이고, 속도를 개선하는 탁월한 성과를 보여줍니다. 특히 Kodak 데이터셋에서는 BD-Rate가 1.24% 증가했음에도 파라미터 수가 약 67% 줄어드는 효과를 낳았습니다. 이러한 연구 결과는 자원 한정 환경에서도 효과적으로 적용될 수 있는 다재다능한 압축 기술을 제안합니다.



### Removing Averaging: Personalized Lip-Sync Driven Characters Based on Identity Adapter (https://arxiv.org/abs/2503.06397)
- **What's New**: 최근 diffusion 기반의 lip-sync generative 모델의 발전은 시각적 더빙을 위한 고도로 동기화된 화자 비디오 생성 능력을 입증했습니다. 하지만 이러한 모델은 생성된 이미지에서 미세한 얼굴 세부사항을 유지하는 데 어려움을 겪습니다. 이 연구에서는 "lip averaging" 현상을 확인하고, UnAvgLip이라는 방법을 제안하여 lip 동기화를 정확하게 유지하면서도 얼굴의 독창적인 특성을 보존하는 방안을 제시합니다.

- **Technical Details**: UnAvgLip은 두 가지 주요 구성 요소로 구성됩니다: (1) Identity Perceiver 모듈은 얼굴 임베딩을 오디오 특징에 맞춰 정렬하며, (2) ID-CrossAttn 모듈은 생성 과정에 얼굴 임베딩을 주입하여 정체성 유지 능력을 향상시킵니다. 이 프레임워크는 강화된 정체성 보존 능력과 precise lip synchronization을 동시에 달성하도록 설계되었습니다.

- **Performance Highlights**: 다양한 실험을 통해 UnAvgLip은 lip inpainting의 ‘averaging’ 현상을 효과적으로 완화하고 독특한 얼굴 특성을 보존하며 정확한 lip 동기화를 유지하는 것으로 나타났습니다. 기존 방법과 비교해 정체성 일관성 지표에서 5% 개선, SSIM 지표에서 2% 개선을 달성하였습니다.



### TI-JEPA: An Innovative Energy-based Joint Embedding Strategy for Text-Image Multimodal Systems (https://arxiv.org/abs/2503.06380)
- **What's New**: 본 연구에서는 인공지능의 다중모달 정렬에 대한 새로운 접근법인 Text-Image Joint Embedding Predictive Architecture (TI-JEPA)를 소개합니다. TI-JEPA는 에너지 기반 모델 (Energy-based Model, EBM) 프레임워크를 활용하여 텍스트와 이미지 간의 복잡한 관계를 포착하고, 멀티모달 감정 분석(task)과 같은 다양한 멀티모달 기반 작업에서 기존의 사전 훈련 방법론보다 우수한 성능을 보여줍니다. 이러한 접근법은 다운스트림 애플리케이션에 상당한 개선을 제안합니다.

- **Technical Details**: TI-JEPA는 크로스 어텐션 메커니즘을 통합하여 텍스트와 시각 정보를 정렬하는 방식으로 작동합니다. 명시된 아키텍처에는 이미지 인코더와 텍스트 인코더가 포함되어 있으며, 이러한 인코더는 각각의 모달리티에서 임베딩 벡터를 생성하여 성과를 낼 수 있도록 지원합니다. 또한, TI-JEPA는 타겟 및 컨텍스트 임베딩을 생성하기 위한 두 단계의 접근법을 설계하여, 각각의 이미지를 패치(patch)로 분할해 처리합니다.

- **Performance Highlights**: TI-JEPA는 다양한 텍스트-이미지 정렬 벤치마크에서 최첨단 성능을 기록하였으며, 특히 감정 분석(task)에서 정확도와 F1 스코어에서 우수한 결과를 보여주었습니다. 이러한 성과는 TI-JEPA의 유연하고 동적인 멀티모달 학습 프레임워크 덕분에 가능했으며, 이는 다양한 다운스트림 작업에서도 효율적인 성능을 발휘하는 데 도움을 줍니다. 이 연구는 또한 에너지 기반 프레임워크를 활용해 멀티모달 융합의 가능성을 여는 데 기여합니다.



### Spectral State Space Model for Rotation-Invariant~Visual~Representation~Learning (https://arxiv.org/abs/2503.06369)
- **What's New**: 이번 연구에서는 기존의 Vision Transformers(ViTs)에서 나타나는 한계를 극복하기 위해 Spectral VMamba라는 새로운 접근 방식을 소개합니다. Spectral VMamba는 이미지 패치 간의 관계를 모델링하는 데 있어 그래프 라플라시안에서 유도된 스펙트럼 정보를 활용하여 글로벌 구조를 효과적으로 포착합니다. 이를 통해 모델은 이미지 회전과 같은 변환에 대한 불변성을 유지하면서도 패치 간의 관계를 독립적으로 인코딩할 수 있습니다.

- **Technical Details**: Spectral VMamba는 스펙트럼 그래프 분석에 기반하여 이미지 패치를 탐색합니다. 이를 통해 패치 간의 유사성을 클러스터링하고, 공간적 근접성에 상관없이 관계를 정의하여 특징 중요도를 평가하는 데 있어 더 나은 성능을 제공합니다. 특히, Rotational Feature Normalizer(RFN) 모듈을 도입하여 회전과 같은 등거리 변환에 대한 모델 강인성을 높입니다.

- **Performance Highlights**: 실험 결과, Spectral VMamba는 기존의 SSM 기반 모델을 능가하며, 이미지 분류 작업에서도 우수한 성능을 보였습니다. 특히, 모델이 회전 불변성을 유지하면서도 유사한 런타임 효율성을 제공합니다. 이러한 성과는 SSM이 복잡한 데이터 패턴을 캡처하고 해석하는 데 더욱 효과적일 수 있음을 보여줍니다.



### VORTEX: Challenging CNNs at Texture Recognition by using Vision Transformers with Orderless and Randomized Token Encodings (https://arxiv.org/abs/2503.06368)
- **What's New**: 본 논문에서는 패턴 인식을 위한 Vision Transformers (ViT) 기반의 새로운 텍스처 인식 방법인 VORTEX를 제안합니다. VORTEX는 다중 깊이 토큰 임베딩을 추출하고, 경량 모듈을 통해 계층적 특징을 집계하여 순서 없는 인코딩을 수행합니다. 이 접근 방식은 기존의 CNN 기술과 비교하여 텍스처 분석에서 우수한 성능을 보여줄 수 있는 가능성을 가지고 있습니다.

- **Technical Details**: VORTEX는 ViT 백본에서 추출한 텍스처 특징을 기반으로 하며, 기존 모델과의 통합이 용이하여 아키텍처의 수정이나 백본의 미세 조정 없이도 활용됩니다. 특징은 단순 선형 SVM에 공급되며, 실험은 아홉 개의 다양한 텍스처 데이터 세트에서 수행되어 결과를 검증하였습니다. VORTEX 구조의 전반적인 모습은 입력 텍스처 이미지와 ViT 백본을 사용하여 이미지 표현을 얻는 방식으로 이루어져 있습니다.

- **Performance Highlights**: 제안된 VORTEX는 다양한 텍스처 분석 시나리오에서 SOTA 성능을 달성하거나 초과할 수 있는 능력을 입증했습니다. VORTEX는 비슷한 비용을 가진 CNN과 비교했을 때 더욱 향상된 계산 효율성을 보여주며, 이는 텍스처 인식 분야에서 ViT 기반 모델의 도입을 촉진할 것으로 기대됩니다.



### Generative Video Bi-flow (https://arxiv.org/abs/2503.06364)
- **What's New**: 본 논문에서는 비디오 생성을 위한 새로운 생성 모델을 제안합니다. 이 모델은 과거 비디오 프레임에서 미래 프레임으로 직접 매핑하는 방법을 사용하여, 기존의 노이즈 기반 프레임 생성을 대체합니다. 특히, 훈련 과정에서 누적된 오류를 제거하는 방법도 함께 학습하여 안정적인 생성을 돕습니다.

- **Technical Details**: 저자들은 neural Ordinary Differential Equation (ODE) 흐름을 활용하여 비디오 프레임의 픽셀 변화를 모델링합니다. 이를 위하여, 기존의 flow matching 목표를 비디오 생성에 맞게 재구성하여, 시간에 따른 이미지 데이터의 변화를 효과적으로 학습합니다. 이 과정에서는 bi-linear interpolation을 통해 효율적인 비디오 흐름을 학습하며, 훈련시 적절한 수준의 노이즈를 주입하여 안정성을 강화합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 조건부 확산 모델과 비교하여 안정성과 효율성에서 우수한 성능을 보여줍니다. 비디오 생성을 실시간으로 수행할 수 있어, 게임 등 상호작용이 필요한 응용 분야에 적합합니다. 전체적으로, 이 연구는 초보자도 이해할 수 있는 수준에서 비디오 생성의 새로운 가능성을 제시합니다.



### Adaptive Audio-Visual Speech Recognition via Matryoshka-Based Multimodal LLMs (https://arxiv.org/abs/2503.06362)
- **What's New**: 이번 논문에서는 Audio-Visual Speech Recognition (AVSR)을 위한 첫 번째 Matryoshka 기반의 다중 모달 LLM인 Llama-MTSK을 제안합니다. Llama-MTSK는 다양한 계산적 제약에 따라 오디오-비주얼 토큰 할당을 유연하게 조정할 수 있도록 설계되었습니다. 이 모델은 Matryoshka Representation Learning (MRL)에서 영감을 받아 서로 다른 압축 레벨에 맞춰 여러 세밀한 표현을 하나의 모델에서 인코딩합니다.

- **Technical Details**: Llama-MTSK은 패러미터 효율적인 LoRA 기반의 미세 조정 전략을 통해 사전 훈련된 LLM을 조정합니다. 이는 전 세계적인 LoRA 모듈과 스케일-특화된 LoRA 모듈을 사용하여 다양한 오디오-비주얼 정보의 논리를 처리합니다. 이를 통해 해당 모델은 압축 비율에 따라 원하는 성능과 효율성 간의 균형을 설정할 수 있습니다.

- **Performance Highlights**: Llama-MTSK는 LRS2 및 LRS3와 같은 가장 큰 AVSR 데이터셋에서 최첨단 결과를 달성했습니다. 독립적으로 교육된 각 압축 레벨에서 훈련된 기존 모델들을 초과하는 성능을 보여주며, ASR, VSR 및 AVSR 작업 전반에서 일관되게 우수한 결과를 기록했습니다.



### Adversarial Robustness of Discriminative Self-Supervised Learning in Vision (https://arxiv.org/abs/2503.06361)
Comments:
          53 pages

- **What's New**: 이번 연구는 자가 지도 학습(self-supervised learning, SSL) 방식의 다양한 모델에 대한 적대적 강건성을 종합적으로 평가합니다. 총 7개의 식별적 자가 지도 모델과 1개의 감독 모델을 대상으로 이미지 분류, 전이 학습, 분할, 탐지 작업 등 다양한 과제를 수행한 결과를 제시합니다. 연구 결과는 SSL 모델이 감독 모델보다 적대적 공격에 대해 더 나은 강건성을 보이며, 이는 전이 학습에도 긍정적인 영향을 미친다는 점을 강조합니다.

- **Technical Details**: 이 연구에서는 Barlow Twins, BYOL, DINO, MoCoV3, SimCLR, SwAV, VICReg 등의 7가지 자가 지도 학습 모델과 감독 모델을 비교하여 다양한 적대적 공격에 대한 강건성을 분석합니다. 평가에서 사용된 주요 모델 아키텍처는 ResNet 기반의 전통적인 비전을 넘어, 최근의 Vision Transformer를 포함합니다. 훈련 기간, 데이터 증강 및 배치 크기와 같은 여러 요인도 적대적 강건성에 미치는 영향을 고려하고 있습니다.

- **Performance Highlights**: 이미지넷(ImageNet) 데이터셋에서 SSL 모델들이 감독 모델보다 일관되게 더 높은 적대적 강건성을 나타냈으며, 특히 MoCoV3 모델이 최상의 성능을 보였습니다. 전이 학습과 세분화 및 탐지 작업에서는 모든 모델이 유사한 취약성을 보였으나, 적대적 공격의 경우 SSL 모델들은 일반적으로 우수한 저항력을 보였습니다. 연구는 SSL 각각의 아키텍처가 적대적 강건성에 미치는 영향을 세밀히 분석하며, 연장된 훈련 기간이 모델의 성능 극대화에 기여한다고도 언급합니다.



### Accurate and Efficient Two-Stage Gun Detection in Video (https://arxiv.org/abs/2503.06317)
- **What's New**: 이번 논문은 비디오에서 총기를 탐지하는 새로운 접근 방식인 두 단계 총기 탐지 방법을 제안합니다. 첫 번째 단계에서는 영상 분류 모델을 통해 '총기' 동영상과 '비총기' 동영상을 분류합니다. 두 번째 단계에서는 총기로 분류된 동영상에서 물체 탐지 모델을 사용하여 총기의 정확한 위치를 찾아내는 방식입니다. 이러한 방법은 기존 모델들보다 효과성과 효율성을 향상시킵니다.

- **Technical Details**: 본 연구에서는 작은 객체(예: 총기)를 탐지하는 데 있어 모델의 성능에 영향을 미치는 여러 요인들, 예를 들어 총기의 크기와 형태의 변동, 비디오 품질 저하 및 유사 객체의 간섭 등을 논의합니다. 고전적인 영상 인식 및 물체 탐지 모델들은 주로 행동 인식을 위해 개발되었으나, 본 논문은 물체 탐지 모델과 영상 분류 모델 통합을 통해 성능을 향상시킵니다. 특히, 이미지 보강(image-augmented) 모델을 활용하여 영상 내 총기 특성을 효과적으로 추출합니다.

- **Performance Highlights**: 실험 결과, 제안된 도메인 특화 방법이 기존 기술들에 비해 성능이 현저히 향상되었음을 입증했습니다. 비디오 내 총기 탐지 정확도가 증가하고 비효율적인 처리 시간을 줄이는 데 기여했습니다. 향후 연구 방향에서는 총기 탐지에서의 도전 과제와 개선 방안을 제시합니다.



### End-to-End Action Segmentation Transformer (https://arxiv.org/abs/2503.06316)
- **What's New**: 이번 논문은 기존의 액션 세그멘테이션 방식을 개선하기 위해 End-to-End Action Segmentation Transformer (EAST)를 도입합니다. 이 모델은 기존 방식들이 가지고 있던 프레임 특징의 비효율성과 액션 세그먼트의 명확한 모델링 부재 문제를 해결하는 데 중점을 두고 설계되었습니다. 특별히 세 가지 주요 기여를 통해 성능을 향상시키고 있습니다: 경량화된 Contract-Expand Adapter 설계, 세그멘테이션-바이-디텍션 프레임워크, 그리고 제안 기반 데이터 증강 기법을 도입했습니다.

- **Technical Details**: EAST는 프레임 특징의 압축과 확장을 통해 복잡성을 줄이는 Contract-Expand Adapter (CEA)를 사용하여 효율적인 end-to-end 훈련을 가능하게 합니다. 또한, 비디오의 고르지 않은 샘플링을 통해 액션 프로포절을 감지하는 방식으로 액션 세그멘테이션을 수행합니다. 이 방법은 다운샘플링된 비디오에서 모든 프레임을 다루지 않고도 효율적이며, 감지된 액션 인스턴스를 명시적으로 추론하여 프레임별 분류를 개선합니다.

- **Performance Highlights**: EAST는 GTEA, 50Salads, Breakfast 및 Assembly-101과 같은 표준 벤치마크에서 최첨단의 성능을 발휘합니다. 기존 방법들과 비교했을 때, 모든 지표에서 높은 성능을 기록하였으며, 특히 데이터가 제한된 액션 세그멘테이션 작업에서 효과적으로 기능하고 있습니다. 모델과 관련 코드도 공개될 예정이어서 다른 연구자들도 이 기법을 활용할 수 있게 될 것입니다.



### Advancing Autonomous Vehicle Intelligence: Deep Learning and Multimodal LLM for Traffic Sign Recognition and Robust Lane Detection (https://arxiv.org/abs/2503.06313)
Comments:
          11 pages, 9 figures

- **What's New**: 이 논문에서는 자율주행차(AV)의 안전한 내비게이션을 위한 고급 심층 학습 기법과 다중모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)을 결합한 통합 접근 방식을 소개합니다. 이 접근 방식은 복잡하고 동적인 환경에서 도로 감지를 향상시키기 위한 것입니다. 특히, 다양한 기법을 평가하여 교통 표지판 인식에서 최고의 성능을 달성하고, 경량화된 MLLM 기반 프레임워크를 제안하여 소규모 데이터셋으로 직접적인 학습 조정을 가능하게 했습니다.

- **Technical Details**: 교통 표지판 인식에는 ResNet-50, YOLOv8, RT-DETR을 평가하여 각각 99.8%, 98.0%, 96.6%의 높은 정확도를 달성했습니다. 차선 감지에서는 CNN 기반의 세그멘테이션 방법에 다항 곡선 맞춤(polygonal curve fitting)을 추가하여 높은 정확도를 이끌어냈습니다. 제안된 MLLM 기반 프레임워크는 초기 사전 훈련 없이 다양한 차선 유형과 복잡한 교차로를 효과적으로 처리하여 차선 검출의 신뢰성을 높였습니다.

- **Performance Highlights**: 이 다중모달 접근법은 훈련 자원의 제약에도 불구하고, 다양한 조건에서도 뛰어난 추론 능력을 보여줍니다. 명확한 조건에서는 99.6%, 야간에는 93.0%의 차선 검출 정확도를 달성했고, 비 오는 날의 차선 불가시(88.4%)나 도로 노후화(95.6%)에 대한 추론에서도 탁월한 성능을 보였습니다. 이 포괄적인 프레임워크는 자율주행차의 인식 신뢰성을 크게 향상시켜, 다양한 도로 시나리오에서 안전한 자율주행에 기여합니다.



### GeoLangBind: Unifying Earth Observation with Agglomerative Vision-Language Foundation Models (https://arxiv.org/abs/2503.06312)
Comments:
          code & weights: this https URL

- **What's New**: GeoLangBind는 다양한 Земля 관측 (Earth Observation, EO) 데이터 모달리티를 통합할 수 있는 새로운 비전-언어 (vision-language) 기초 모델입니다. 이 모델은 동종 데이터 간의 간극을 메우고자 언어를 통합 매개체로 사용하여, 다양한 센서 데이터에서 보충적 특징 학습을 가능하게 합니다. 이를 위해 저자들은 GeoLangBind-2M이라는 대규모 멀티모달 이미지-텍스트 데이터셋을 구축하여, 무작위 EO 데이터 채널 처리가 가능한 제로샷 (zero-shot) 기초 모델을 개발했습니다.

- **Technical Details**: GeoLangBind는 여러 시나리오에서 다양한 EO 데이터 모달리티를 지원하는 데 필요한 세 가지 주요 설계를 포함합니다: 1) 여러 데이터 모달리티를 처리할 수 있는 파장 인식 동적 인코더, 2) 지식 집합화 모듈 (Knowledge Agglomeration)인 MaKA, 3) 여러 모달리티에 걸쳐 모델을 효율적으로 규모화하는 진전적 가중치 병합 전략. 이러한 접근 방식은 다양한 데이터 모달리티에 걸쳐 통일된 언어 특징 공간으로 조정하여, 더 매끄러운 분석이 가능하도록 합니다.

- **Performance Highlights**: GeoLangBind는 제로샷 분류, 의미론적 분할 및 교차 모드 이미지 검색(semantic segmentation and cross-modal image retrieval)과 같은 여러 작업에서 우수한 성능을 발휘합니다. 실험 결과, GeoLangBind 모델은 다양한 EO 애플리케이션에서 뛰어난 성능과 다재다능성을 보여주며, 환경 모니터링 및 분석 작업에 대한 강력한 프레임워크를 제공합니다. 추가로, GeoLangBind 및 사전 훈련된 모델은 공개적으로 이용 가능하게 될 계획입니다.



### Text2Story: Advancing Video Storytelling with Text Guidanc (https://arxiv.org/abs/2503.06310)
Comments:
          15 pages, 6 figures

- **What's New**: 본 논문에서는 텍스트 프롬프트만을 사용하여 일관된 장기 비디오 시퀀스를 생성하는 새로운 스토리텔링 접근 방식을 소개합니다. 기존의 확산 기반 모델들이 짧은 비디오 합성에 우수한 성능을 보이는 것과 달리, 텍스트로부터의 장기 스토리텔링은 시간적 일관성(temporal coherency) 및 의미 보존(semantic meaning) 등의 도전 과제가 있어 충분히 탐구되지 않았습니다. 우리 방법은 자연스러운 동작 전환과 구조화된 내러티브를 통해 비디오 생성을 매끄럽게 합니다.

- **Technical Details**: 우리는 장기 비디오 생성의 세그먼트 간 시간적 일관성을 보장하기 위해 양방향 시간 가중치(latent blending) 전략을 제안합니다. 또한, 이미지 생성에 대한 프롬프트 혼합(promt mixing)에서 Black-Scholes 알고리즘을 확장하여 비디오 생성에 적용하며, 구조화된 텍스트 조건을 통해 제어된 동작 진화를 가능하게 합니다. 우리의 접근 방식은 동작 유사성에 기반한 전환 조정을 통해 매끄러운 동작 변경을 보장하고, 장면 내 객체 간의 공간적 일관성을 유지합니다.

- **Performance Highlights**: 광범위한 실험을 통해, 우리는 기존 기법들에 비해 장기 비디오 내러티브에서 시간적 일관성과 시각적 매력을 크게 향상시켰음을 입증했습니다. 추가적인 훈련 없이도 비주얼적으로 매력적인 비디오 내러티브를 생성할 수 있었으며, 우리의 접근 방법은 짧은 클립과 장기 비디오 간의 격차를 해소하여 GenAI 기반 텍스트 비디오 합성의 새로운 패러다임을 확립합니다.



### ACAM-KD: Adaptive and Cooperative Attention Masking for Knowledge Distillation (https://arxiv.org/abs/2503.06307)
Comments:
          8 pages, 10 tables, 3 figures

- **What's New**: 이 논문에서는 지식 증류(knowledge distillation, KD)의 새로운 방법인 Adaptive student-teacher Cooperative Attention Masking (ACAM-KD)을 제안합니다. ACAM-KD는 기존 방법의 한계를 극복하기 위해 두 가지 주요 구성 요소인 Student-Teacher Cross-Attention Feature Fusion (STCA-FF)과 Adaptive Spatial-Channel Masking (ASCM)을 도입합니다. 이 방법은 학생 모델과 교사 모델 간의 상호작용을 통해 효과적으로 특성을 통합하고, 공간적 및 채널적 중요성을 동적으로 업데이트하여 학생의 학습 상태에 적응하도록 돕습니다.

- **Technical Details**: ACAM-KD의 STCA-FF 모듈은 학생 및 교사 모델의 특성을 통합하여 보다 상호작용적인 distillation 과정을 촉진합니다. ASCM 모듈은 공간적 및 채널 차원에서의 중요성을 고려한 마스킹을 동적으로 생성합니다. 이를 통해 기존의 고정된 또는 교사 주도의 마스크 선택 방식과는 달리, ACAM-KD는 학생의 발전에 맞춰 주의를 동적으로 조정할 수 있는 구조를 제공합니다.

- **Performance Highlights**: 실험 결과, ACAM-KD는 여러 기준에서 기존의 최첨단 KD 방법들보다 우수한 성능을 기록하였습니다. 예를 들어, COCO2017 기준에서 ResNet-50 학생이 ResNet-101 교사로부터 distillation할 때, 객체 감지 성능이 최대 1.4 mAP 증가하였으며, Cityscapes에서 DeepLabV3-MobileNetV2 학생 모델을 사용할 경우, mIoU가 3.09 향상되었습니다. 이러한 성과는 ACAM-KD가 다양한 dense prediction 작업에 있어 효과적임을 입증합니다.



### Your Large Vision-Language Model Only Needs A Few Attention Heads For Visual Grounding (https://arxiv.org/abs/2503.06287)
- **What's New**: 이번 연구에서는 대형 비전-언어 모델(LVLM)의 특정 주의 헤드가 시각적 기초 설계에 유용하다는 점을 발견했습니다. 특히, 저자들은 이러한 주의 헤드를 '로컬라이제이션 헤드(localization heads)'로 정의하고, 이를 활용하여 훈련이 필요 없는 간단하고 효과적인 시각적 기초 프레임워크를 제안합니다. 이 프레임워크는 텍스트-이미지 주의 맵을 이용해 대상 물체를 식별합니다.

- **Technical Details**: 로컬라이제이션 헤드는 개체에 대한 텍스트 의미와 관련된 지역을 지속적으로 캡처하는 몇 가지 주의 헤드를 나타냅니다. 연구에서는 이러한 헤드를 구체적으로 식별하는 두 가지 기준을 제시하며, (1) 각 주의 헤드가 이미지에 얼마나 집중하는지를 측정하고, (2) 특정 이미지 영역에 주의를 기울이는 헤드를 선택합니다. 이를 통해 최종적으로 세 개의 로컬라이제이션 헤드만으로도 적절한 경계 상자나 마스크를 예측할 수 있습니다.

- **Performance Highlights**: 혼합 성능 평가 결과, 제안된 방법은 기존의 훈련 기반 방법들보다 큰 폭으로 성능을 향상시켰습니다. 또한, 훈련이 필요 없는 방법 중에서도 우수한 성과를 거두며, 특정 훈련을 받은 LVLM과 비교해서도 유사한 성능을 보여줍니다. 이러한 결과들은 LVLM이 텍스트 표현과 일관된 영역을 본질적으로 식별할 수 있는 효과적인 도구임을 입증합니다.



### From Dataset to Real-world: General 3D Object Detection via Generalized Cross-domain Few-shot Learning (https://arxiv.org/abs/2503.06282)
- **What's New**: 이번 연구에서는 LiDAR 기반 3D 객체 검출에 대한 generalized cross-domain few-shot learning (GCFS) 작업을 처음으로 정의합니다. GCFS는 다양한 객체 범주에서의 모델의 성능을 개선하기 위해 사전 훈련된 모델을 적응시키는 데 중점을 둡니다. 본 연구는 데이터 부족과 도메인 적응 관련 문제를 통합적으로 해결하기 위해 멀티모달 융합과 대조 강화 프로토타입 학습을 포함한 솔루션을 제시합니다.

- **Technical Details**: GCFS 전략은 일반적인 객체 카테고리와 새로운 타겟 클래스의 존재를 가정합니다. 방법론은 vision-language models (VLMs)를 사용하여 포인트 클라우드와 정렬된 이미지에서 세분화된 시맨틱 지식을 추출합니다. 다양한 구조적 복잡성에 따른 포인트 분포의 편향을 해결하기 위해 물리적으로 인식 가능한 박스 검색 전략을 도입하여 3D 박스 제안을 개선합니다.

- **Performance Highlights**: 세 가지 GCFS 벤치마크 설정을 통해 제안된 솔루션의 효과가 입증되었습니다. 제안된 솔루션은 데이터 부족과 도메인 적응 문제를 종합적으로 해결하며, 한정된 타겟 데이터에 대해서도 모델의 적응력을 높입니다. 이러한 연구 결과와 기법들은 향후 연구에 대한 포괄적인 평가 프레임워크를 제공합니다.



### STiL: Semi-supervised Tabular-Image Learning for Comprehensive Task-Relevant Information Exploration in Multimodal Classification (https://arxiv.org/abs/2503.06277)
Comments:
          16 pages (including 5 pages of supplementary materials), accepted by CVPR 2025

- **What's New**: 이번 논문은 제한된 레이블 데이터 문제를 해결하기 위해 멀티모달 이미지-테이블 학습에서 세미-슈퍼바이즈드 러닝(SemiSL)을 탐구하는 최초의 연구입니다. 저자는 Modality Information Gap을 정의하고 이를 완화하기 위해 STiL이라는 새로운 프레임워크를 제안합니다. STiL은 레이블이 있는 데이터와 레이블이 없는 데이터를 동시에 활용하여 작업 관련 정보를 효과적으로 탐색합니다.

- **Technical Details**: STiL은 새로운 disentangled contrastive consistency 모듈을 특징으로 하며, 이는 교차 모달 불변 표현을 학습하면서 모달리티-특정 정보를 보존합니다. 또한, 지도 분류기들의 합의에 기반하여 신뢰성 있는 의사 레이블을 생성하는 consensus-guided pseudo-labeling 전략을 도입합니다. 저자는 또한 프로토타입 임베딩을 통해 의사 레이블의 품질을 개선하는 prototype-guided label smoothing 기법을 제안합니다.

- **Performance Highlights**: 자연 및 의료 이미지 데이터셋에서 수행된 실험 결과, STiL은 기존의 이미지 및 멀티모달 감독/자체 지도 학습(SL)/세미-슈퍼바이즈드 러닝(SemiSL) 방법들을 뛰어넘는 성과를 보여주었습니다. 특히 레이블이 부족한 상황에서도 우수한 성능을 보이며, 작업 관련 정보 학습을 극대화합니다.



### Exploring Adversarial Transferability between Kolmogorov-arnold Networks (https://arxiv.org/abs/2503.06276)
- **What's New**: 이 논문에서는 Kolmogorov-Arnold Networks (KANs)에 대한 적대적 공격의 취약성을 조사하고, AdvKAN이라는 첫 번째 전이 공격 방법을 제안합니다. KANs는 기존의 고정 선형 가중치를 학습 가능한 일변수 함수로 대체하여 네트워크의 유연성과 해석성을 증가시킵니다. 하지만 KAN들이 특정 기초 함수에 과적합(overfitting)되어 있어 적대적 전이성(adversarial transferability)이 떨어진다는 문제점이 있습니다.

- **Technical Details**: 이 연구에서는 AdvKAN의 두 가지 주요 요소를 소개합니다: 첫째, Breakthrough-Defense Surrogate Model (BDSM)이 포함되어 있으며, 이는 특정 KAN 구조에 대한 과적합을 완화하기 위해 훈련됩니다. 둘째, Global-Local Interaction (GLI) 기술이 적용되어, 적대적 기울기의 글로벌 및 로컬 상호작용을 촉진하여 손실 표면(loss surface)을 부드럽게 만듭니다. 이 두 요소가 결합되어 KAN 간의 적대적 전이성을 효과적으로 향상시킵니다.

- **Performance Highlights**: 다양한 KAN 아키텍처와 데이터셋에 대한 실험 결과, AdvKAN은 기존 최첨단 기술보다 월등히 우수한 공격 능력을 보여줍니다. 이 연구는 KAN의 고유한 취약성을 드러내면서도, 전이 공격이 KAN의 보안을 강화하는 데 기여할 수 있음을 보입니다. 논문은 최종 승인 시 코드도 공개할 계획입니다.



### Zero-AVSR: Zero-Shot Audio-Visual Speech Recognition with LLMs by Learning Language-Agnostic Speech Representations (https://arxiv.org/abs/2503.06273)
- **What's New**: 본 논문은 Zero-AVSR이라는 새로운 제로샷(Zero-shot) Audio-Visual Speech Recognition (AVSR) 프레임워크를 제안합니다. 이 프레임워크는 특정 언어의 오디오-비주얼 음성 데이터 없이도 해당 언어의 음성을 인식할 수 있게 해줍니다. 특히, 입력된 다언어 오디오-비주얼 음성 데이터를 기반으로 Roman 텍스트를 예측하는 Audio-Visual Speech Romanizer (AV-Romanizer)를 도입하여 언어 독립적인 음성 표현을 학습합니다.

- **Technical Details**: 제로샷 AVSR을 실현하기 위해 AV-Romanizer에서 예측한 Roman 텍스트를 미리 훈련된 대형 언어 모델(Large Language Model, LLM)을 사용하여 언어 특정한 그래프음(graphemes)으로 변환합니다. 우리는 AV-Romanizer와 LLM을 연결하는 두 가지 작업을 통해 성능을 향상시키는 Zero-AVSR 모델을 제안합니다. 첫 번째 작업은 학습된 텍스트 공간에 audio-visual speech representation을 통합할 수 있도록 LLM과 AV-Romanizer를 미세 조정하는 것입니다.

- **Performance Highlights**: 제안한 Zero-AVSR 프레임워크는 82개 언어의 2,916시간의 오디오-비주얼 데이터로 구성된 Multilingual Audio-Visual Romanized Corpus (MARC)를 통해 다양한 언어 지원을 확장하는 가능성을 보여줍니다. 또한, 이 모델은 Cascaded Zero-AVSR과 개선된 Zero-AVSR 두 가지 형태로 구현되어 이전의 모든 LLM과 함께 사용될 수 있습니다. 실험 결과는 제안한 프레임워크가 다양한 언어에 대한 지원을 넓힐 수 있는 잠재력을 갖추고 있음을 확인합니다.



### SplatTalk: 3D VQA with Gaussian Splatting (https://arxiv.org/abs/2503.06271)
- **What's New**: 이번 논문에서는 SplatTalk라는 새로운 방법론을 제안합니다. SplatTalk는 일반화 가능한 3D Gaussian Splatting (3DGS) 프레임워크를 활용하여 3D 토큰을 생성하며, 이를 통해 사전 훈련된 LLM에 직접 입력할 수 있도록 합니다. 이 접근방식은 단순히 촬영된 이미지만으로 3D 시각적 질의 응답(3D VQA)을 효과적으로 수행할 수 있습니다.

- **Technical Details**: SplatTalk는 주어진 RGB 이미지를 통해 언어적 특징을 추가하여 3D Gaussian Splatting 표현을 구축합니다. 여기서 생성된 3D 토큰은 사전 훈련된 LLM에게 직접 제공되어 복잡한 3D VQA 질문에 대한 답변을 가능하게 합니다. 이 연구는 3D 장면에 대한 이해를 촉진하기 위해 언어 특징 통합 및 LLM에 입력할 수 있는 토큰 추출 방법을 제안합니다.

- **Performance Highlights**: SplatTalk는 다양한 3D VQA 벤치마크에서 3D 모델보다 우수한 성능을 보였습니다. 또한, 이전의 이미지 기반 방법론을 능가하여 3D 장면 표현 구축의 중요성을 입증했습니다. 마지막으로, SplatTalk는 최신 3D LMM보다도 경쟁력 있는 성과를 보여줍니다.



### Get In Video: Add Anything You Want to the Video (https://arxiv.org/abs/2503.06268)
Comments:
          Project page:this https URL

- **What's New**: 이번 논문에서는 'Get-In-Video Editing'이라는 새로운 편집 패러다임을 제시하며, 사용자가 제공하는 참조 이미지를 통해 특정 시각적 요소를 비디오에 통합하는 방식을 설명합니다. 기존 방식의 한계를 보완하기 위해, 저자들은 GetIn-1M이라는 대규모 데이터셋을 구축하였으며, 이는 비디오 편집 쌍을 포함하고 있습니다. 또한, 제안된 GetInVideo는 3D 완전 주의(diffusion transformer architecture) 아키텍처를 활용하여 참조 이미지, 비디오 및 마스크를 동시에 처리할 수 있는 혁신적인 프레임워크입니다.

- **Technical Details**: GetIn-1M 데이터셋은 자동화된 Recognize-Track-Erase 파이프라인을 통해 생성되며, 비디오 캡셔닝, 주목할 만한 인스턴스 식별, 객체 감지, 시간적 추적 및 인스턴스 제거 기능을 포함합니다. GetInVideo 모델은 프롬프트, 참조 이미지, 마스크, 조건 비디오와 같은 네 가지 주요 입력을 동시에 처리하여 편집된 비디오를 출력합니다. 이 모델은 전에 훈련된 T2V 파라미터의 효과적인 상속을 가능하게 하는 레이턴트 공간 결합 접근 방식을 채택하였습니다.

- **Performance Highlights**: GetInBench는 참조 기반 비디오 객체 추가를 위한 최초의 포괄적 벤치마크로, 단일 물체 및 다중 물체 참조 시나리오에 대한 평가를 지원합니다. 실험 결과, GetInVideo는 기존 방법들보다 시각적 품질에서 우수성을 보였으며, 다수의 평가 지표에서 상당한 양적 우위를 달성했습니다. 이 연구는 고품질의 특정 현실 대상 통합을 가능하게 하여 개인화된 비디오 편집 능력을 크게 향상시키는 기여를 하고 있습니다.



### Segment Anything, Even Occluded (https://arxiv.org/abs/2503.06261)
- **What's New**: 새로운 논문에서는 Amodal instance segmentation의 개념을 도입하여 가려진 부분을 포함한 객체의 세분화를 목표로 합니다. 기존 방법들이 전방 탐지기와 마스크 디코더를 동시에 훈련해야 하는 불편함을 해결하기 위해 SAMEO라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 Segment Anything Model (SAM)을 기반으로 하여 다양한 전방 탐지기와 인터페이스가 가능하도록 혁신적인 마스크 디코더로 설계되었습니다.

- **Technical Details**: SAMEO는 EfficientSAM 구조를 활용하여 이미지 인코더, 프롬프트 인코더, 마스크 디코더를 포함합니다. 이를 통해 부분적으로 가려진 객체에 대한 마스크 예측이 가능해지며, 두 가지 유형의 입력(이미지와 경계 상자 프롬프트)을 사용하여 학습합니다. 학습 과정에서는 Dice loss, Focal loss 및 IoU 추정에 대한 L1 손실을 결합하여 모델의 성능을 극대화합니다.

- **Performance Highlights**: 제안된 방법은 Amodal-LVIS라는 새로운 대규모 데이터셋에서 훈련된 결과, COCOA-cls와 D2SA 벤치마크에서 뛰어난 제로 샷(zero-shot) 성능을 달성했습니다. 이러한 결과는 기존의 감독(supervised) Amodal 세분화 방법을 초월하며, 새로운 접근방식의 가능한 일반화(generalization) 능력을 강조합니다.



### From Captions to Rewards (CAREVL): Leveraging Large Language Model Experts for Enhanced Reward Modeling in Large Vision-Language Models (https://arxiv.org/abs/2503.06260)
- **What's New**: 대규모 비전-언어 모델(LVLM)을 인간의 선호도와 맞추는 것은 고급 다중 모달 선호 데이터 부족 때문에 어려운 도전입니다. 기존의 방법들은 저신뢰도 데이터로 인해 최적의 성능을 발휘하지 못하는 단점을 가지고 있습니다. 이에 대한 해결책으로 제안된 CAREVL은 신뢰성 있는 고신뢰도 및 저신뢰도 데이터를 활용한 선호 보상 모델링의 새로운 방법입니다.

- **Technical Details**: CAREVL은 첫째로, 보조 전문가 모델(auxiliary expert models)인 텍스트 보상 모델(textual reward models)을 활용하여 이미지 캡션을 약한 지도 신호(weak supervision signals)로 활용하여 고신뢰도 데이터를 필터링합니다. 그런 다음, 이 고신뢰도 데이터를 사용하여 LVLM을 미세 조정(fine-tuning)합니다. 둘째로, 저신뢰도 데이터를 활용하여 다양한 선호 샘플을 생성하며, 이 샘플들은 별도로 채점하여 신뢰 가능한 선택-거부 쌍을 구성하여 추가 훈련에 사용됩니다.

- **Performance Highlights**: CAREVL은 VL-RewardBench 및 MLLM-as-a-Judge 벤치마크에서 기존의 증류(distillation) 기반 방법들에 비해 성능 개선을 달성하였습니다. 이는 다양한 신뢰도 수준의 데이터를 효과적으로 활용함으로써 가능합니다. 앞으로 코드가 곧 공개될 예정입니다.



### Can Atomic Step Decomposition Enhance the Self-structured Reasoning of Multimodal Large Models? (https://arxiv.org/abs/2503.06252)
- **What's New**: 이번 논문에서는 '느린 사고' (slow thinking) 능력을 다중 모드 대형 언어 모델(MLLMs)에 통합하여 다중 모달 수학적 추론의 도전 과제를 다룹니다. 제안된 Self-structured Chain of Thought (SCoT) 패러다임은 복잡성에 따라 문제에 적합한 동적 추론 구조를 생성할 수 있도록 합니다. 기존 방법과 달리 구조화된 템플릿이나 자유형 패러다임에 의존하지 않고, 인지적 CoT 구조를 생성하여 과도한 사고를 줄일 수 있습니다.

- **Technical Details**: AtomThink라는 새로운 프레임워크는 데이터 엔진, 감시된 미세 조정 과정, 정책 기반의 다중 턴 추론 방법, 그리고 원자적 능력 메트릭을 포함한 네 개의 핵심 모듈로 구성됩니다. 이를 통해 20,000개의 고급 수학적 문제와 124,000개의 원자적 단계 주석이 포함된 AMATH 데이터셋을 생성했습니다. 이 모델은 원자 단계를 기반으로 하여 멀티 모달 작업에서 자가 구조화된 추론 능력을 활성화합니다.

- **Performance Highlights**: 우리는 MathVista, MathVerse 및 MathVision에서 10% 이상의 정확도 향상을 보여주었으며, LLaVA-CoT에 비해 500%의 데이터 활용성과 80% 이상의 추론 효율성을 개선했습니다. 이러한 성능 향상은 다중 모달 고도 추론 개선을 위한 중요한 기초를 제공합니다. 또한, 다양한 이해 능력의 분포와 MLLM의 성능을 분석하여 공유합니다.



### Rethinking Lanes and Points in Complex Scenarios for Monocular 3D Lane Detection (https://arxiv.org/abs/2503.06237)
Comments:
          CVPR2025

- **What's New**: 이번 논문은 자율주행에서 단안 3D 차선 검출의 중요성을 강조하며, 기존의 희소 점(sparse-point) 방법들이 차선 기하 구조를 충분히 활용하지 못하고 있음을 지적합니다. 이를 해결하기 위해, 새로운 패칭 전략(patching strategy)과 EndPoint head (EP-head) 모듈을 제안하여 더 높은 F1-score를 달성할 수 있도록 합니다.

- **Technical Details**: 제안된 EP-head는 차선의 끝점을 예측하는 거리를 추가하여, 모델이 기본적으로 더 많은 차선 표현을 예측할 수 있도록 하고, PointLane attention (PL-attention) 모듈은 차선 구조에 대한 이전 기하학적 지식을 주의(attention) 메커니즘에 통합합니다. 이는 각 차선 내의 점들 간의 관계를 학습할 수 있도록 도와줍니다.

- **Performance Highlights**: 실험을 통해, Persformer는 4.4점, Anchor3DLane은 3.2점, LATR은 2.8점의 F1-score 향상을 이뤘습니다. 이러한 성과는 각 모델이 주어진 한계 내에서 더 완전한 차선 표현을 할 수 있도록 하여 새로운 최첨단 성능을 달성하는 데 기여하고 있습니다.



### Dynamically evolving segment anything model with continuous learning for medical image segmentation (https://arxiv.org/abs/2503.06236)
- **What's New**: EvoSAM은 의료 이미지 세분화(Medical Image Segmentation) 분야에서 동적으로 발전하는 모델로, 다양한 시나리오와 작업에서 얻은 새로운 지식을 지속적으로 축적하여 세분화 능력을 향상시킵니다. 이전의 전통적인 접근법이 단일 학습(threaded learning)에 의존하는 것과 달리, EvoSAM은 다채로운 세분화 요구를 충족하기 위해 진화합니다.

- **Technical Details**: EvoSAM은 다양한 수술 이미지에서 혈관 세분화 및 다중 사이트 전립선 MRI 세분화(multi-site prostate MRI segmentation)에 대한 광범위한 평가를 통해 그 성능이 입증되었습니다. 이 모델은 세분화 정확성(segmentation accuracy)을 개선하고 재앙적 망각(catastrophic forgetting)을 완화하는 데 효과적입니다.

- **Performance Highlights**: 수술 Clinician들이 수행한 혈관 세분화 실험에서 EvoSAM은 사용자 프롬프트(prompt)에 따라 세분화 효율성을 높이는 것으로 나타났습니다. 이는 EvoSAM이 임상 응용 프로그램(clinical applications)에 매우 유망한 도구로 자리 잡을 수 있는 가능성을 강조합니다.



### StreamGS: Online Generalizable Gaussian Splatting Reconstruction for Unposed Image Streams (https://arxiv.org/abs/2503.06235)
Comments:
          8 pages

- **What's New**: 이 논문에서는 Camera Parameters가 없는 이미지 스트림으로부터 온라인에서 일반화된 3D Gaussian Splatting(3DGS) 재구성을 지원하는 StreamGS라는 새로운 방법을 제안합니다. StreamGS는 각 프레임에 대해 Gaussian을 예측하고 집계하여 이미지를 점진적으로 3D Gaussian 스트림으로 변환합니다. 또한, 기존의 초기 점 재구성 방법의 제한을 극복하고, 서로 인접한 프레임 간의 픽셀 대응 관계를 확립하여 일관성을 높입니다.

- **Technical Details**: StreamGS는 프레임 간의 신뢰할 수 있는 대응 관계를 구축하여 내용 적응형(descriptor) 정제를 통해 재구성을 수정합니다. 이 방법은 불필요한 중복 세트를 가지기 때문에 Gaussian 밀도를 감소시켜 온라인 재구성의 컴퓨팅 및 메모리 비용을 현저히 줄입니다. 우리는 DUSt3R와 같은 사전 훈련된 모델을 활용하여 이전 프레임을 참조로 삼아 3D 포인트를 예측하고 각 프레임에서 Gaussian을 통합합니다.

- **Performance Highlights**: 실험 결과, StreamGS는 최적화 기반 접근 방식에 비해 질적으로 동등한 성능을 보이며, 재구성 속도는 150배 더 빠릅니다. 또한, OOD(Out Of Domain) 장면을 다루는 데 있어 기존의 자세 의존 3DGS 방식보다 뛰어난 일반화 능력을 보여줍니다. 이러한 성과는 다양한 데이터 세트에서 검증되었습니다.



### Reinforced Diffuser for Red Teaming Large Vision-Language Models (https://arxiv.org/abs/2503.06223)
- **What's New**: 이번 연구에서는 Red Team Diffuser (RTD)라는 새로운 프레임워크를 도입하여 대규모 비전-언어 모델(VLM)의 보호 메커니즘을 우회하고, 유해한 텍스트 연속성을 유도하는 방법을 탐구합니다. 기존의 연구들이 해로운 지침에 대한 VLM의 취약점에 초점을 맞춘 반면, 이 논문은 독성이 있는 텍스트 연속성 작업에서 발생하는 중요한 취약성을 밝히고 있습니다. 실험을 통해 RTD가 VLM 출력의 독성 비율을 유의미하게 증가시키는 데 효과적임을 입증했습니다.

- **Technical Details**: RTD는 강화 학습(reinforcement learning)을 활용하여 높은 독성을 가진 이미지 프롬프트를 생성하는 두 단계의 프로세스를 통해 구조화되어 있습니다. 첫 번째 단계에서는 LLM이 VLM의 반응에서 독성을 극대화하는 이미지 프롬프트를 탐색하고, 두 번째 단계에서는 독성과 정렬 보상을 통해 생성된 출력의 유해성을 증대시키는 방식으로 확산 모델을 미세 조정합니다. 이 연구의 주요 기여는 VLMs의 안전 메커니즘 부족을 강조하고, 보다 강력하고 적응 가능한 정렬 메커니즘의 필요성을 강조하는 것입니다.

- **Performance Highlights**: 실험 결과, RTD는 원래 공격 세트에서 LLaVA 모델의 독성 비율을 10.69% 증가시켰으며, 보유 세트에서는 8.91%의 증가를 보였습니다. 또한, Gemini와 LLaMA 모델에 대해서도 각각 5.1%와 26.83%의 독성 비율 증가를 관찰하여 RTD의 강력한 모델 간 전이 가능성을 보여주었습니다. 이러한 성과들은 VLMs의 현재 정렬 전략의 중대한 결함을 드러내며, 유해한 연속성을 방지하기 위한 추가 연구의 필요성을 제기합니다.



### Vision-based 3D Semantic Scene Completion via Capture Dynamic Representations (https://arxiv.org/abs/2503.06222)
- **What's New**: CDScene은 비전 기반 강력한 의미 장면 완성을 위한 새로운 방법론을 제안합니다. 이 모델은 동적 표현을 포착하여 명시적 의미 모델링 및 장면 정보의 동적 및 정적 특징을 분리합니다. 이 접근 방식은 자율주행 시나리오에서 견고하고 정확한 장면 완성을 가능하게 합니다.

- **Technical Details**: CDScene은 다중 모달 대규모 모델(Large Multimodal Models)을 활용하여 2D 명시적 의미를 추출하고 이를 3D 공간에 정렬합니다. 또한 모노큘러 및 스테레오 깊이의 특징을 활용하여 장면 정보를 동적 및 정적 특징으로 분리합니다. 이 과정에서 동적 특징은 동적 객체 주변의 구조적 관계를 포함하고, 정적 특징은 밀집된 문맥 공간 정보를 포함합니다.

- **Performance Highlights**: CDScene의 성능은 SemanticKITTI, SSCBench-KITTI360, SemanticKITTI-C 데이터셋에 대한 광범위한 실험을 통해 검증되었으며, 기존 방법들을 크게 초월하는 결과를 보였습니다. 이 모델은 자율주행 시나리오에서 강력하고 정확한 의미 장면 완성을 달성하며, 최신 접근 방식을 넘어 우수성과 견고성을 입증하였습니다.



### StreamMind: Unlocking Full Frame Rate Streaming Video Dialogue through Event-Gated Cognition (https://arxiv.org/abs/2503.06220)
- **What's New**: 이번 논문에서는 Streaming Video Dialogue의 필요성을 강조하며, ultra-FPS (100 fps) 비디오 처리를 실행할 수 있는 새로운 LLM 프레임워크인 \\sys를 소개합니다. 이 프레임워크는 사용자 개입 없이도 실시간으로 반응할 수 있는 능력을 제공합니다. 논문은 기존 LLM의 한계를 극복하기 위해 'event-gated LLM invocation'이라는 새로운 패러다임을 제안합니다.

- **Technical Details**: 새로운 기술에서 비디오 인코더와 LLM 사이에 Cognition Gate 네트워크를 도입하여 관련 이벤트 발생 시에만 LLM이 호출되도록 합니다. Event-Preserving Feature Extractor (EPFE)를 통해 일정한 비용으로 이벤트 피처를 추출하며, 이로 인해 spatiotemporal feature를 생성합니다. 이러한 기법들은 비디오 LLM에 실시간 인지 반응을 가능하게 합니다.

- **Performance Highlights**: Ego4D 및 SoccerNet 스트리밍 작업에서의 실험 결과는 모델의 성능 및 실시간 효율성을 보여주며, ultra-high-FPS 응용 프로그램에 대한 가능성을 열어줍니다. 다양한 평가 지표에서 최고 성능을 달성했으며, COIN 및 Ego4D LTA 데이터셋에서도 우수한 결과를 기록했습니다. 이 프레임워크는 실제 인간-AI 상호작용 응용을 위한 튼튼한 기반을 마련합니다.



### VLScene: Vision-Language Guidance Distillation for Camera-Based 3D Semantic Scene Completion (https://arxiv.org/abs/2503.06219)
Comments:
          Accept by AAAI-2025(Oral)

- **What's New**: 이번 논문에서는 VLScene이라는 새로운 방법을 제안합니다. 이 방법은 비전-언어 모델을 활용하여 고차원 세미틱 프라이어(semantic priors)를 추출하고, 이를 통해 3D 장면 인식을 향상시킵니다. 특히, 이미지의 세부적인 물체 정보와 관련된 공간적 맥락을 강화하는 것이 핵심입니다.

- **Technical Details**: VLScene은 비전-언어 가이던스 증류(vision-language guidance distillation) 프로세스를 도입하여 이미지 특성을 개선합니다. 또한, 지역 기하학 전파(neighborhood geometry propagation)와 희소 세미틱 상호작용(sparse semantic interaction) 모듈을 통해 기하학적 구조를 효과적으로 전파하고 맥락적으로 희소한 세미틱 정보를 강화합니다.

- **Performance Highlights**: VLScene은 SemanticKITTI와 SSCBench-KITTI-360 벤치마크에서 17.52%와 19.10%의 mIoU를 기록하여 최첨단 성능을 달성했습니다. 이 모델은 47.4M의 파라미터만으로도 경쟁력 있는 결과를 자랑합니다.



### Explainable Synthetic Image Detection through Diffusion Timestep Ensembling (https://arxiv.org/abs/2503.06201)
Comments:
          13 pages, 5 figures

- **What's New**: 이 연구에서는 최근 발전한 diffusion 모델이 생성한 이미지의 식별을 위한 새로운 방법을 제안합니다. 기존의 감지 방법이 진품 사진과 합성 이미지를 구분하는 데 어려움을 겪고 있는 가운데, 본 논문은 이미지의 고주파 성분에서 발생하는 차이를 활용하여 합성 이미지를 효과적으로 탐지할 수 있는 가능성을 보여줍니다. 또한, 이는 추론 기능을 추가하여 AI 생성 이미지의 결함을 식별하는 데도 기여합니다.

- **Technical Details**: ESIDE(Explainable Synthetic Image Detection through Diffusion Timestep Ensembling)라는 프레임워크를 제안하며, 이는 여러 타임스탬프에서 노이즈가 추가된 원본 이미지를 통해 분류기를 훈련시키는 방식을 채택하고 있습니다. 해당 프레임워크는 DDIM 반전 과정을 통해 생성된 여러 노이즈 버전을 이용하여 이미지를 처리하며, 이들 노이즈 이미지는 CLIP 이미지 인코더를 통해 특징 표현을 추출한 후 AdaBoost 모델에 입력됩니다. 이를 통해 각 모델은 이미지가 합성인지의 여부를 평가하며, 최종 예측은 이러한 평가를 기반으로 한 가중 합산으로 도출됩니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 GenImage 기준으로 정규 샘플에서 98.91%, 더 어려운 샘플에서 95.89%의 높은 탐지 정확도를 기록하며, 기존 방법보다 각각 2.51%, 3.46% 향상된 성능을 보였습니다. 또한, 이 방법은 다른 diffusion 모델이 생성한 이미지에 대해서도 효과적으로 일반화되는 특성을 가지고 있습니다. 공급된 데이터셋인 GenHard와 GenExplain은 더욱 높은 난이도의 탐지 샘플과 AI 생성 이미지에 대한 고품질 설명을 제공합니다.



### Removing Multiple Hybrid Adverse Weather in Video via a Unified Mod (https://arxiv.org/abs/2503.06200)
- **What's New**: 이 논문에서는 다양한 악천후 조건에서 비디오를 처리하기 위한 새로운 통합 모델인 UniWRV를 제안합니다. 이는 여러 종류의 날씨로부터 비디오 데이터를 복원할 수 있는 강력한 능력을 가지고 있으며, 기존의 알고리즘들이 개별적으로 훈련된 모델에 의존하는 방식에서 벗어나 통합적인 접근 방식을 제공합니다. 또한, 이 모델은 대응하는 기후 조건에 대한 고유한 특징을 유추하기 위해 특정 날씨 프라이어(guidance)를 활용하여 더 효과적으로 기능을 수행합니다.

- **Technical Details**: UniWRV는 이질적인 공간 및 시간적 특징을 효과적으로 처리하기 위해 설계되었습니다. 공간적 특징 처리를 위한 날씨 프라이어 가이드 모듈(WPGM)과 시간적 특징 집합을 위한 동적 라우팅 집합 모듈(DRA)을 통해 서로 다른 영상 프레임의 최적 경로를 선택하여 특징을 통합합니다. 또한, 방대한 양의 합성 비디오 데이터셋인 HWVideo를 구축하여 여러 날씨 조건 하의 비디오 복원을 학습하고 벤치마킹하는 데 기여합니다.

- **Performance Highlights**: UniWRV는 악천후 제거 작업뿐만 아니라 일반적인 비디오 복원 작업에서도 최첨단 성능을 발휘합니다. 다양한 영상 환경에서도 일반화 가능성을 자랑하며, 실험 결과 기존 방법들을 크게 초월하는 성과를 보여줍니다. 이 모델은 특히 자율 주행 시스템 및 감시 시스템과 같은 실제 응용 프로그램에 매우 유용할 것입니다.



### NeuroADDA: Active Discriminative Domain Adaptation in Connectomic (https://arxiv.org/abs/2503.06196)
Comments:
          8 pages, 3 figures, 3 tables

- **What's New**: 이 논문의 가장 큰 특징은 전이 학습(transfer learning)과 도메인 적응(domain adaptation)을 통해 새로운 전자 현미경(connectomics) 데이터셋에서 세그멘테이션(segmentation) 성능을 향상시키기 위한 방법을 모색하는 것입니다. 기존 데이터셋에서 학습된 모델을 활용하여 주석(annotation) 비용을 줄이고 효율을 높일 수 있는 가능성을 제시하고 있습니다. NeuroADDA라는 새로운 메소드를 통해 최고의 도메인을 선택하고, 소스 데이터 없이도 효과적으로 학습할 수 있는 방안을 제안하였습니다.

- **Technical Details**: NeuroADDA는 최대 평균 불일치(Maximum Mean Discrepancy, MMD)를 사용하여 데이터가 최적의 소스 도메인을 식별하는 역할을 합니다. 또한, 몇 가지 주석 달린 샘플을 사용하여 새로운 데이터셋에 대해 지식을 이전하고 인스턴스 세그멘테이션 성능을 최적화하는 알고리즘을 설계했습니다. 이를 통해 다양한 데이터셋에서 세그멘테이션 성능이 유의미하게 향상된 것을 확인하였습니다.

- **Performance Highlights**: NeuroADDA는 여러 데이터셋 및 샘플 크기에서 무작정 학습하는 방식보다 일관되게 우수한 성능을 보였으며, $n=4$ 샘플 일 때 25-67%의 정보 변동성(Variation of Information, VI) 감소를 보였습니다. 또한 다양한 종간의 뉴런 이미지 간의 분포 차이를 분석하여 이러한 도메인 '거리'가 생물학적 계통 구분과 상관관계를 가진다는 것을 밝혔습니다.



### MSConv: Multiplicative and Subtractive Convolution for Face Recognition (https://arxiv.org/abs/2503.06187)
- **What's New**: 이번 논문에서는 얼굴 인식 작업에서 중요한 특징인 salient feature와 differential feature의 균형 잡힌 학습을 위한 새로운 합성곱 모듈 MSConv(Multiplicative and Subtractive Convolution)를 제안합니다. 최근의 주목 메커니즘 기반 접근법들이 주로 salient feature에 집중했던 반면, 저자들은 facial images의 복잡한 샘플 처리 시 필요한 시각적 세부정보를 간과할 수 있다는 점을 지적하며, differential feature의 중요성을 강조합니다. MSConv는 multi-scale mixed convolution을 통해 두 종류의 특징을 효과적으로 캡처하고, 모델의 전반적인 인식 성능을 향상시키는 데 기여합니다.

- **Technical Details**: MSConv 모듈은 특정 조건에서 학습의 효과를 극대화하기 위해 Multiplication Operation (MO)과 Subtraction Operation (SO)을 활용합니다. MO는 두 개의 서로 다른 feature를 다중 스케일 합성곱 방법으로 얻어내고 곱하여 수행되며, SO는 feature map 간의 미세한 차이를 포착하는 데 사용됩니다. 이러한 접근법을 통해 salient feature와 differential feature를 동시에 효과적으로 사용할 수 있도록 합니다. 저자들은 SKNet 모델에서의 softmax를 sigmoid로 변환하는 과정을 통해 MSConv의 구조적 변화를 이루어내었습니다.

- **Performance Highlights**: 실험 결과, MSConv는 오직 salient feature만을 중점적으로 다룬 모델들보다 뛰어난 성능을 보였습니다. 두 종류의 feature를 조합함으로써 얼굴 인식 작업에서의 인식 정확도를 높였고, noise나 occlusion에 대해 보다 효과적으로 대처할 수 있음을 보였습니다. 이로써 MSConv는 얼굴 인식 분야에서 더욱 신뢰할 수 있는 성능을 제공하며, 복잡한 시나리오에서도 일관된 결과를 나타냅니다.



### PTDiffusion: Free Lunch for Generating Optical Illusion Hidden Pictures with Phase-Transferred Diffusion Mod (https://arxiv.org/abs/2503.06186)
Comments:
          Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2025)

- **What's New**: 이 논문에서는 Optical illusion hidden picture(착시 숨은 그림)이라는 흥미로운 시각적 현상을 다룹니다. 이를 위해 기존의 text-to-image (T2I) diffusion 모델을 기반으로 하여, 훈련 없이 사용할 수 있는 새로운 텍스트 주도 이미지-투-이미지 (I2I) 변환 프레임워크인 Phase-Transferred Diffusion Model(PTDiffusion)을 제안합니다. 이 모델은 입력된 참조 이미지를 텍스트 프롬프트에 따라 다양한 장면에 통합할 수 있도록 설계되었습니다.

- **Technical Details**: PTDiffusion은 plug-and-play phase transfer 메커니즘을 중심으로 작동하며, 이로 인해 확산 처리 과정에서의 phase spectrum(상 단계 스펙트럼)을 동적으로 이식하여 참조 이미지를 샘플링된 환상 이미지로 재구성합니다. 또한, 비동기적 phase transfer(상 단계 전송)를 통해 숨겨진 콘텐츠의 인식 가능성에 대한 유연한 조절이 가능합니다. 이러한 방식으로 구조적 정보와 텍스트의 의미 정보의 조화로운 융합이 이루어집니다.

- **Performance Highlights**: 제안된 방법은 모델 훈련 및 세밀한 조정을 필요로 하지 않으면서도, 이미지 품질, 텍스트 충실도, 시각적 인식 가능성 및 문맥적 자연성에서 기존 방법들을 크게 초월하는 성능을 보여줍니다. 이들은 광범위한 정성적 및 정량적 실험을 통해 입증되었습니다. 따라서, 이 연구는 착시 그림 합성 분야에서 새로운 기준을 제시하고 있습니다.



### FORESCENE: FOREcasting human activity via latent SCENE graphs diffusion (https://arxiv.org/abs/2503.06182)
- **What's New**: 본 연구에서는 FORESCENE이라는 새로운 Scene Graph Anticipation (SGA) 프레임워크를 소개합니다. FORESCENE은 관찰된 비디오 세그먼트를 Graph Auto-Encoder를 통해 잠재 표현(latent representation)으로 인코딩하고, Latent Diffusion Model(LDM)을 사용하여 미래의 Scene Graph(SG)를 예측합니다. 이 접근법은 그래프의 내용이나 구조에 대한 가정 없이도 상호작용 역학을 지속적으로 예측할 수 있게 합니다.

- **Technical Details**: FORESCENE은 두 단계로 운영되며, 첫 번째 단계에서는 사용자 정의된 Graph Auto-Encoder(GAE)에 의해 관찰된 비디오를 잠재적 표현으로 변환합니다. 두 번째 단계에서는 Latent Diffusion Model이 이 표현을 근거로 SG의 시간적 진화를 예측합니다. 이 방법은 사용자가 개체와 그들의 쌍관계에 대해 사전의 제약 없이 예측할 수 있도록 합니다.

- **Performance Highlights**: Action Genome 데이터셋에서 FORESCENE을 평가한 결과, 기존 SGA 방법들을 능가하는 성능을 보였습니다. FORESCENE은 물체 발견(object discovery)과 관계 발견(relation discovery) 두 가지 주요 작업에서 뛰어난 성과를 기록하며, 복잡한 환경에서도 물체의 등장과 소실을 예측하는 데 성공하였습니다.



### ForestSplats: Deformable transient field for Gaussian Splatting in the Wild (https://arxiv.org/abs/2503.06179)
- **What's New**: 이 논문에서는 ForestSplats라는 새로운 접근 방식을 제안합니다. 이 방법은 변형 가능한 transient field와 superpixel-aware mask를 활용하여 정적 장면에서 transient 요소를 효과적으로 분리합니다. 기존 방법들의 단점을 보완하며, Vision Foundation Model(VFM) 없이도 높은 품질의 렌더링을 제공합니다.

- **Technical Details**: ForestSplats는 각 뷰에 대한 transient 요소를 포착하는 변형 가능한 transient field를 설계했습니다. 또한 photometric 오류와 superpixel을 고려하여 occluder의 경계를 명확히 정의하는 superpixel-aware mask를 도입했습니다. 이 방법을 통해 정적 필드의 경계 내에서 Gaussian을 생성하지 않도록 하는 불확실성 인식 densification가 포함되어 있습니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋에 대한 광범위한 실험을 통해 ForestSplats는 기존 방법들보다 뛰어난 성능을 보였으며, transient 요소의 메모리 효율성을 나타냈습니다. 이로 인해, 최신 기술들과 비교하여 동적 요소와 정적 장면을 성공적으로 분리하는 데 기여하였습니다.



### Treble Counterfactual VLMs: A Causal Approach to Hallucination (https://arxiv.org/abs/2503.06169)
- **What's New**: 이번 연구에서는 Vision-Language Models (VLMs)의 환각(hallucination) 문제를 해결하기 위해 인과론적(causal) 관점을 도입했습니다. 기존의 연구들은 통계적 편향(statistical biases)이나 언어적 선입견(language priors)과 같은 요인들을 중심으로 환각을 분석했지만, 본 연구에서는 그 원인을 인과적 그래프(causal graph)와 반사실적 분석(counterfactual analysis)을 통해 구조적으로 이해하려고 하였습니다. 이를 통해 VLM의 출력에서 비의도적인 직적 영향(unintended direct influence)을 체계적으로 제거할 수 있는 방법론을 제시합니다.

- **Technical Details**: 연구에 따르면, VLM에서 환각 현상은 비전 모달리티(vision modality)와 텍스트 모달리티(text modality) 간의 잘못된 통합으로 발생한다고 가정합니다. 이를 해결하기 위해 구조적 인과 그래프(structural causal graphs)를 설계하고, 각 모달리티의 자연 직적 효과(Natural Direct Effect, NDE)를 추정하여 모델의 각 모달리티 의존도를 조절하는 동적 개입 모듈(test-time intervention module)을 개발하였습니다. 이러한 접근법은 다양한 변형된 이미지와 언어 임베딩을 활용하여 각 모달리티의 특성을 보다 명확히 평가할 수 있게 합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법이 기존의 환각 완화 연산보다 우수한 성능을 보이며 VLM의 신뢰성을 크게 향상시켰습니다. 특히, 단순히 기존 방법을 초월하여 두 개의 다양한 벤치마크에서 두 개의 VLM에서 일관되게 뛰어난 성과를 나타냈습니다. 이 연구는 VLM의 신뢰성을 높이는 데 기여할 수 있는 강력하고 해석 가능한 프레임워크를 제공하며, 관련 코드 또한 공개되어 접근성과 재현성을 강조하고 있습니다.



### Feature-EndoGaussian: Feature Distilled Gaussian Splatting in Surgical Deformable Scene Reconstruction (https://arxiv.org/abs/2503.06161)
Comments:
          14 pages, 5 figures

- **What's New**: 본 논문은 3D Gaussian Splatting (3DGS)을 기반으로 한 Feature-EndoGaussian (FEG)이라는 새로운 접근법을 제안합니다. 기존의 Neural Radiance Fields (NeRFs)보다 데이터 요구량이 적고 렌더링 속도가 빠른 FEG는 실시간 의미적 및 장면 재구성을 가능하게 합니다. 이 기술은 2D 세그멘테이션 정보를 3D 렌더링에 통합하여, 수술 중 내비게이션의 정확성을 획기적으로 향상시킵니다.

- **Technical Details**: FEG는 Gaussian 변형 프레임워크 내에서 세그멘테이션 기능을 증류하여 재구성의 정확성과 세그멘테이션의 신뢰성을 모두 강화합니다. 제안된 레스터라이저(rasterizer)는 색상, 깊이 및 의미적 특징 맵을 동시에 렌더링할 수 있도록 여러 가지 개선사항을 포함하고 있습니다. 이는 Gaussian 원시 체적을 이미지 공간으로 렌더링하며, 각 원시의 매개변수에 대한 그래디언트 전달을 가능하게 합니다.

- **Performance Highlights**: EndoNeRF 데이터셋에서 FEG는 0.97의 SSIM, 39.08의 PSNR, 0.03의 LPIPS를 달성하여 기존의 주요 방법들보다 우수한 성능을 보였습니다. 또한 EndoVis18 데이터셋에서도 경쟁력 있는 클래스별 세그멘테이션 메트릭스를 보여주며, 모델 크기와 실시간 성능을 균형 있게 유지합니다.



### UrbanVideo-Bench: Benchmarking Vision-Language Models on Embodied Intelligence with Video Data in Urban Spaces (https://arxiv.org/abs/2503.06157)
Comments:
          22 pages

- **What's New**: 이 연구는 도시 내에서의 움직임 중 지각 능력에 대한 멀티모달 모델(Video-LLMs)의 능력을 평가하기 위한 새로운 벤치마크인 UrbanVideo-Bench를 소개합니다. 이 벤치마크는 1.5천 개의 동영상 클립과 5.2천 개의 다지선다형 질문을 포함하여, 도시에서의 체화된 인지 능력에 대한 통찰력을 제공합니다. 이를 통해 현재의 멀티모달 모델이 도시 환경에서 인지 적응력 부족한 부분을 수치적으로 분석하였습니다.

- **Technical Details**: 연구는 4가지 능력(Recall, Perception, Reasoning, Navigation)로 나뉜 16개의 특정 과제를 정리합니다. 드론을 이용해 실제 도시와 시뮬레이터에서 동영상 데이터를 수집하였으며, 이 과정에서 발생할 수 있는 신호 손실과 같은 여러 도전과제를 해결하였습니다. 채집된 데이터는 모델의 인지 능력을 평가하기 위한 다지선다형 질문 생성 파이프라인과 결합되었습니다.

- **Performance Highlights**: 17개의 일반적으로 사용되는 Video-LLMs를 평가한 결과, 이들은 도시 환경에서의 체화된 인지 능력에 현저한 한계를 보였습니다. 인과적 추론(causal reasoning)은 기억력, 지각력, 내비게이션과 강한 상관관계를 가지는 반면, 반사실적 및 연결적 추론(capabilities for counterfactual and associative reasoning)은 다른 작업과 낮은 상관관계를 보였습니다. 이는 도시 환경에서의 체화된 인지 학습을 위한 기준틀을 마련한 중요한 발견입니다.



### SRM-Hair: Single Image Head Mesh Reconstruction via 3D Morphable Hair (https://arxiv.org/abs/2503.06154)
Comments:
          Under review

- **What's New**: 이 논문에서는 3D Morphable Models (3DMMs)를 활용하여 독립적인 3D 모발 메쉬를 단일 이미지로부터 재구성하는 혁신적인 방법인 Semantic-consistent Ray Modeling of Hair (SRM-Hair)를 소개합니다. 이 방법은 고급 데이터 세트를 통해 3D 모발의 표현을 가능하게 하며, 모발의 additivity(덧셈성)와 adaptability(적응성)를 강조합니다. 이는 사용자가 스타일을 조합하고 새로운 아이덴티티로 전환할 수 있음을 의미합니다.

- **Technical Details**: SRM-Hair는 고해상도 3D 머리 스캔에서 세부적인 모발 메쉬를 추출하고, 고정된 multi-ray (다중 레이) 방향 템플릿을 사용하여 모발 표면의 3D 정점 시퀀스를 순서대로 결정을 합니다. 이 방식은 모발의 두께 수정, 플리핑 및 여러 스타일 패턴의 융합을 가능하게 하는 여러 뛰어난 특성을 제공합니다. 마지막적으로, SRM-Hair는 효율적인 모양 제어가 가능하며, 3D 머리 메쉬의 정확한 재구성을 목표로 합니다.

- **Performance Highlights**: 정량적 및 정성적 실험 결과, SRM-Hair는 이전의 방법들을 초월하여 3D 메쉬 재구성에서 최첨단 성능을 달성합니다. 이 연구는 250개 이상의 고충실도 실제 모발 스캔으로 구성된 데이터 세트를 기반으로 하며, 3D 얼굴 데이터와 함께 사용되어 향후 가상 아바타 생성 및 사실적인 애니메이션에 활용될 수 있습니다. SRM-Hair는 단일 이미지에서 모발 메쉬를 재구성하는 데 있어 기존의 방법들보다 뛰어난 효율성을 보여줍니다.



### BioMoDiffuse: Physics-Guided Biomechanical Diffusion for Controllable and Authentic Human Motion Synthesis (https://arxiv.org/abs/2503.06151)
- **What's New**: BioMoDiffuse는 생체 역학(Biomechanics) 인식을 통해 기존의 한계를 극복한 새로운 모션 생성 프레임워크입니다. 이 시스템은 인간 움직임에 대한 특정 요구 사항을 반영하여 근육 활성화 패턴 및 관절 조정을 고려합니다. BioMoDiffuse는 물리적으로 그럴싸한 움직임을 생성하기 위해 EMG 신호와 운동학적 특징을 통합한 가벼운 생동감 네트워크를 특징으로 합니다.

- **Technical Details**: 이 프레임워크의 핵심 혁신은 수정된 오일러-라그랑주 방정식(Euler-Lagrange equations)을 기반으로 물리적 가이드를 통합한 물리 지향 확산 프로세스(Physics-guided diffusion process)를 포함합니다. 독립적인 속도와 의미 컨트롤을 위한 분리된 제어 메커니즘(Decoupled control mechanism) 또한 제공하여 더욱 정교한 동작 조정이 가능합니다. 이를 통해 다양한 평가 프로토콜과 병행하여 생체 역학적 기준을 포함한 새로운 평가 방법을 구축했습니다.

- **Performance Highlights**: BioMoDiffuse는 HumanML3D 및 KIT-ML 데이터 세트를 통해 기존 방법에 비해 현저한 성능 향상을 보여주었습니다. 모션 생성에 있어 생리학적 현실성을 유지하며 물리적 법칙을 준수하는 새로운 기준을 수립하였습니다. 또한 기존의 모션 생성 지표(FID, R-precision 등)에 더하여 새로운 생체 역학적 기준을 도입함으로써 전반적인 평가 체계를 완성했습니다.



### OpenRSD: Towards Open-prompts for Object Detection in Remote Sensing Images (https://arxiv.org/abs/2503.06146)
Comments:
          11 pages, 4 figures

- **What's New**: 이 논문은 OpenRSD라는 새로운 오픈 프롬프트(remote sensing object detection) 원거리 감지 프레임워크를 제안합니다. 이 프레임워크는 다중 모달 프롬프트(multi-modal prompts)를 지원하며, 다양한 시나리오에서 객체 감지의 정확도와 실시간 성능을 동시에 충족할 수 있도록 설계되었습니다. 이를 통해 기존의.closed-set QR 코드 감지 방식의 한계를 극복하고, 다양한 객체의 탐지를 가능하게 합니다.

- **Technical Details**: OpenRSD는 이미지 및 텍스트 프롬프트를 이용한 객체 탐지를 지원하며, 다중 임무(multi-task) 감지 헤드(detecion heads)를 통합하여 높은 정확도와 신속한 성능을 보장합니다. 특히, 두 개의 감지 헤드를 통해 비슷한 대상을 인식하는 정렬 헤드(alignment head)와, 정보를 상호작용할 수 있는 융합 헤드(fusion head)로 구성되어 있습니다. 또한, 다단계 훈련 파이프라인(multi-stage training pipeline)은 데이터 세트를 효과적으로 활용하고 모델의 일반화 능력을 향상시키는데 기여합니다.

- **Performance Highlights**: OpenRSD는 공개 데이터 세트인 HRSC2016, SpaceNet 등에서 평가되었으며, 가로 및 세로 바운딩 박스 탐지에서 우수한 성능을 나타냅니다. 특히, YOLO-World와 비교해 8.7% 높은 평균 정확도를 기록했으며, 20.8 FPS의 빠른 추론 속도로 대규모 원거리 이미지 분석이 가능함을 증명했습니다. 이를 통해 OpenRSD는 실제 RS 객체 탐지 작업에 매우 적합한 모델로 자리잡을 것입니다.



### VLForgery Face Triad: Detection, Localization and Attribution via Multimodal Large Language Models (https://arxiv.org/abs/2503.06142)
- **What's New**: 이번 연구에서는 고급 Deepfake 탐지를 위한 MLLMs(멀티모달 대형 언어 모델)와 DMs(확산 모델) 기반의 얼굴 포렌식 기술을 결합하여 VLForgery라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 위조된 얼굴 이미지의 예측, 부분 합성에 대한 위치 지정, 그리고 특정 생성 방법에 대한 속성을 부여하는 기능을 포함합니다. 특히, VLF(비주얼 언어 포렌식)라는 새로운 데이터셋을 소개하여 MLLMs와 비주얼, 언어 모달리티 간의 상호작용을 풍부하게 하고 있습니다.

- **Technical Details**: VLForgery 프레임워크는 1) 탐지(detection), 2) 위치 지정(localization), 3) 속성 부여(attribution)의 세 가지 주요 작업을 수행하도록 설계되었습니다. 이를 위해, EkCot(외부 지식 기반의 설명 생성 방법)과 낮은 수준의 비전 비교 파이프라인을 사용할 수 있습니다. 이 기술들은 MLLMs가 이미지를 신속하게 이해하고, 실제 이미지와 위조 이미지를 평가하도록 합니다.

- **Performance Highlights**: 실험 결과, VLForgery는 기존의 포렌식 접근법보다 탐지 정확도에서 우수성을 보였습니다. 또한, 위조된 지역의 위치 지정 및 속성 분석에서도 잠재력을 가지고 있습니다. 특히, 기존의 연구들은 개별적인 탐지, 위치 지정 및 속성 부여에 초점을 맞추었다면, VLForgery는 이 세 가지를 통합적으로 다루어 새로운 가능성을 제시하고 있습니다.



### Next Token Is Enough: Realistic Image Quality and Aesthetic Scoring with Multimodal Large Language Mod (https://arxiv.org/abs/2503.06141)
- **What's New**: 이번 연구에서는 사용자 생성 콘텐츠 (UGC) 이미지를 평가하기 위한 새로운 데이터셋인 Realistic image Quality and Aesthetic (RealQA)를 제안합니다. 이 데이터셋에는 14,715개의 UGC 이미지가 포함되어 있으며, 각 이미지는 10개의 세부적인 속성으로 주석이 달려 있습니다. 본 연구는 이미지 품질 평가 (IQA)와 미적 평가 (IAA)에서 멀티모달 대형 언어 모델 (MLLMs)을 효과적으로 활용하는 방법에 대한 심층적인 조사를 수행합니다.

- **Technical Details**: RealQA 데이터셋은 세 가지 수준으로 인간의 인식을 세분화합니다: 낮은 수준(예: 이미지 선명도), 중간 수준(예: 주제의 완전성) 및 높은 수준(예: 구도). 연구에서는 MLLMs가 수치 점수를 직접 예측할 수 있는 가능성을 탐구하며, 주목할 만한 발견 중 하나는 다음 토큰 예측 패러다임을 통해 2자리 유효 숫자를 추가로 예측하면 훨씬 더 좋은 성능을 발휘한다는 점입니다. 이러한 접근 방식은 특히 차가운 시작 전략을 활용하여 공공 데이터셋의 속성을 자동으로 주석 처리하는 데 기여합니다.

- **Performance Highlights**: 제안된 방법은 5개의 공공 IQA 및 IAA 데이터셋에서 SOTA 성능을 초과하며, 비디오 품질 평가 (VQA) 작업에서도 강력한 제로샷 일반화를 보여줍니다. 예를 들어, Koniq-10k 데이터셋에서 Q-Align보다 +1.8% 향상된 PLCC를 달성하였으며, KoNViD 데이터셋에서는 이미지 데이터를 통해 훈련된 모델이 +36.4% SRCC 개선을 기록했습니다. 이러한 성과는 MLLMs가 인간의 지각을 더 잘 반영할 수 있는 가능성을 열어줍니다.



### Boosting the Local Invariance for Better Adversarial Transferability (https://arxiv.org/abs/2503.06140)
- **What's New**: 이 논문은 전이 기반 공격의 전반적인 문제를 다루며, 대체 모델(surrogate model)에서 생성된 적대적 예시(adversarial examples)가 새로운 피해 모델(victim model)을 속일 수 있는 가능성을 탐구합니다. 최근 연구에서 이들은 대부분 적대적 교란(adversarial perturbations)과 입력 이미지(input images) 간의 관계를 간과하고 있음을 지적합니다. 저자들은 적대적 교란이 특정 클린 이미지(clean image)와 모델에 대해 낮은 전이 불변성(transfer invariance)을 보인다는 사실을 발견하고, 이를 통해 적대적 전이 가능성을 높이는 방법인 LI-Boost를 제안합니다.

- **Technical Details**: LI-Boost 방식은 주어진 모델에서의 적대적 교란의 로컬 불변성(local invariance)을 최적화하여 공격의 전이 가능성을 향상하는 전략입니다. 이 접근법은 여러 번의 반복(iteration)을 통해 적대적 예시의 기울기를 사용하여 여러 번 변환된 교란을 적용함으로써 로컬 불변성을 강조합니다. 실험은 표준 이미지넷 데이터 세트(ImageNet dataset)를 기반으로 진행되었으며, 저자들은 이 방법이 CNNs(Convolutional Neural Networks)와 ViTs(Vision Transformers) 모두에서 효과적임을 입증했습니다.

- **Performance Highlights**: LI-Boost는 다양한 전이 기반 공격에 대해 상당한 성능 향상을 보여주었으며, 방어 기제(defense mechanisms)가 있든 없든 높은 일반성과 우수성을 입증합니다. 이 방법은 적대적 전이 가능성을 향상시키는 새로운 방향을 제시하며, 향후 연구에 중요한 기여를 할 것으로 기대됩니다. 요약하자면, LI-Boost는 물리적인 대상 탐지 및 보안 분야에서 심각한 위협이 될 수 있는 전이 기반 공격에 대한 대응책으로서 중요한 전환점을 제공합니다.



### GSV3D: Gaussian Splatting-based Geometric Distillation with Stable Video Diffusion for Single-Image 3D Object Generation (https://arxiv.org/abs/2503.06136)
- **What's New**: 본 논문에서는 이미지 기반 3D 생성의 최신 연구 결과를 소개하며, 특히 로봇 공학, 게임 및 가상 현실에서의 응용 가능성을 강조합니다. 기존의 3D 확산 모델(3D diffusion models)에서는 데이터셋 부족이나 강력한 사전 훈련 모델의 결여로 인한 한계가 있었으며, 2D 확산 기반 접근법은 기하학적 일관성을 유지하는 데 어려움을 겪었습니다. 제안하는 방법은 2D 확산 모델의 암묵적인 3D 추론 능력을 활용하고, Gaussian-splatting 기반의 기하학적 증류를 통해 3D 일관성을 보장합니다.

- **Technical Details**: 제안된 Gaussian Splatting Decoder는 SV3D의 잠재 출력(latent outputs)을 명시적인 3D 표현으로 변환함으로써 3D 일관성을 강제합니다. 이 과정에서 다중 관점(latents)을 구조적으로 정렬된 3D 표현으로 변환하며, 이러한 기하학적 제약은 뷰 간의 불일치를 수정하고 견고한 기하학적 일관성을 제공합니다. 이와 같은 방법은 3D 확산 모델에 비해 높은 충실도(fidelity)와 다양한 질감을 생성할 수 있는 강력한 솔루션으로 자리 잡고 있습니다.

- **Performance Highlights**: 실험 결과는 다중 관점 일관성(multi-view consistency) 및 다양한 데이터셋에서의 강력한 일반화 능력을 입증합니다. 본 연구는 2D 입력으로부터 일관된 3D 객체를 생성하는 데 있어 뛰어난 성능을 보이며, 코드도 수락 시 공개될 예정입니다. 더욱이, 제안하는 프레임워크는 단일 이미지 기반의 3D 생성을 위한 확장 가능한 솔루션을 제공하며, 2D 확산 모델의 다양성과 3D 구조의 일관성 간의 격차를 메우는 데 기여하고 있음을 보여줍니다.



### X2I: Seamless Integration of Multimodal Understanding into Diffusion Transformer via Attention Distillation (https://arxiv.org/abs/2503.06134)
Comments:
this https URL

- **What's New**: 본 논문에서는 다중 모드 입력을 이해할 수 있는 X2I 프레임워크를 제안합니다. 이 프레임워크는 Diffusion Transformer(DiT) 모델에 다양한 모드를 이해할 수 있는 능력을 부여하며, multilingual text, 이미지, 비디오 및 오디오를 포함합니다. X2I는 단 100K 영어 말뭉치와 160 GPU 시간을 활용하여 훈련됩니다.

- **Technical Details**: X2I는 기존의 DiT 모델을 기반으로 하여 새로운 디스틸레이션 방법을 도입하였습니다. 이 과정에서 AlignNet 구조를 설계하여 중간 브릿지 역할을 통해 T2I 모델의 성능 저하를 1% 미만으로 유지하면서 다중 모드 이해 능력을 획득할 수 있도록 합니다. LightControl을 통해도 이미지 편집의 충실도를 높이는 기능이 추가됩니다.

- **Performance Highlights**: X2I를 사용한 테스트 결과, T2I 모델의 성능 저하가 1% 미만으로 유지되면서 다양한 다중 모드 이해 능력을 입증했습니다. 더욱이, 전통적인 LoRA 훈련을 수행하여 IT2I 작업의 산업 요건을 충족함으로써, X2I는 기존 T2I 모델에 비해 개선된 적응성을 보여주었습니다. 전체 실험 결과, X2I는 효율성, 다기능성 및 이식성을 입증하였습니다.



### USP: Unified Self-Supervised Pretraining for Image Generation and Understanding (https://arxiv.org/abs/2503.06132)
- **What's New**: 본 논문은 통합적 자기 지도 사전 학습(USP, Unified Self-supervised Pretraining) 프레임워크를 제안하여, diffusion 모델을 변형된 VAE(latent space)에서 초기화함으로써 여러 다운스트림 비주얼 작업에 대한 성능을 향상시킵니다. 이 프레임워크는 representation learning과 diffusion 기반의 이미지 생성을 동시에 지원하여, 두 영역 간의 격차를 줄이는 데 중점을 두고 있습니다. 이를 통해 기존의 비효율적인 사전 학습 및 미세 조정 과정을 단순화할 수 있습니다.

- **Technical Details**: USP는 이미지 생성을 위한 확산 모델(difussion model)을 VAE(latent space)에서 마스킹된 latent 모델링(masked latent modeling)을 통해 초기화합니다. 이 과정에서 사전 훈련된 VAE를 활용하여 이미지의 latent feature를 획득하고, 이를 통해 다운스트림 작업에서 우수한 representation을 유지합니다. 이러한 방식은 라벨이 없는 데이터로도 학습이 가능하며, 작업별 손실(loss)의 차이를 해소합니다.

- **Performance Highlights**: 제안한 프레임워크는 두 가지 transformer 기반의 diffusion 모델, DiT와 SiT에서 뛰어난 성능을 나타납니다. USP는 각각 600K 및 150K 스텝에서 훈련하여 기존의 7M 스텝 모델보다 11.7배 및 46.6배 더 빠른 수렴 속도를 보이며, 이미지 인식에 대한 강력한 representation 능력을 유지합니다. 또한, 추가적인 훈련 비용이나 메모리 오버헤드를 발생시키지 않고 높은 효율성을 보장합니다.



### Viewport-Unaware Blind Omnidirectional Image Quality Assessment: A Flexible and Effective Paradigm (https://arxiv.org/abs/2503.06129)
- **What's New**: 본 논문에서는 기존의 Blind Omnidirectional Image Quality Assessment (BOIQA) 모델들이 viewport 생성에 의존하고 있다는 점을 지적하고, 이러한 문제점을 해결하기 위해 Viewport-Unaware BOIQA (VU-BOIQA)라는 새로운 패러다임을 제안합니다. 이 모델은 equirectangular projection (ERP) 이미지로부터 패치를 효율적으로 추출하는 adaptive prior-equator sampling (APS) 모듈과, 변형에 강한 progressive deformation-unaware feature fusion (PDFF) 모듈을 포함합니다. 이를 통해 사용자의 시각적 행동을 전제로 하지 않고도 이미지의 전반적인 품질을 평가할 수 있게 됩니다.

- **Technical Details**: VU-BOIQA 모델은 두 가지 주요 모듈로 구성됩니다. 첫째, APS 모듈은 패치의 크기와 샘플링 영역을 스스로 조정하여 의미 있는 정보를 추출합니다. 둘째, PDFF 모듈은 비정형적인 변형을 처리하는 데 도움이 되는 변형 가능한 합성곱을 사용하여 서로 다른 스케일과 레이어의 효과적인 특징 통합을 구현합니다. 마지막으로, local-to-global quality aggregation (LGQA) 모듈을 통해 지역 품질 정보를 종합하여 전반적인 품질을 도출합니다.

- **Performance Highlights**: 제안하는 VU-BOIQA 모델은 네 개의 OIQA 데이터베이스에서 포괄적인 실험을 진행한 결과, 최신 모델들과 비교하여 뛰어난 성능을 보였으며, 낮은 복잡도로 경쟁력 있는 결과를 나타냈습니다. 특히, 2D 이미지 품질 평가 (2D-IQA)에 대한 적용 가능성도 검증되었습니다. 이를 통해, 기존의 BOIQA 모델에 비해 보다 효율적인 이미지 품질 평가 방식으로 자리잡을 수 있을 것으로 기대됩니다.



### SecureGS: Boosting the Security and Fidelity of 3D Gaussian Splatting Steganography (https://arxiv.org/abs/2503.06118)
Comments:
          Accepted by ICLR 2025

- **What's New**: SecureGS는 3D Gaussian Splatting(3DGS) 스테가노그래피의 새로운 프레임워크로, 실시간 렌더링과 높은 품질의 출력을 제공하여 3D 자산의 개인 정보 보호 필요성을 강조합니다. 기존 GS 스테가노그래피 방법들은 렌더링 정확도 감소, 증가된 계산 요구 및 보안 문제로 고통받고 있으며, SecureGS는 이러한 문제를 해결하기 위한 안전하고 효율적인 솔루션을 제안합니다. 이 프레임워크는 네트워크를 통해 인가된 사용자만이 접근할 수 있도록 비밀 정보를 암호화하여 저장하는 하이브리드 방식의 알고리즘을 사용하고 있습니다.

- **Technical Details**: SecureGS는 고유한 하이브리드 분리형 가우시안 암호화 메커니즘을 사용하여 원래 3D 장면과 숨겨진 물체의 렌더링을 위한 두 세트의 밀집 가우시안 포인트를 동적으로 예측합니다. 이 과정에서 숨겨진 3D 물체의 위치를 숨기기 위한 오프셋 예측기를 도입하고, 숨겨진 물체의 기하학적 구조가 노출되지 않도록 밀도 영역 인식 앵커 성장 전략을 혁신적으로 제안합니다. 이를 통해, 기존 GS 스테가노그래피 방법보다 뛰어난 안정성과 효율성을 유지합니다.

- **Performance Highlights**: 다양한 실험 결과에 따르면 SecureGS는 기존 3D 스테가노그래피 방법에 비해 렌더링 정확도, 속도, 보안성이 모두 크게 향상되었습니다. 특히, 이 프레임워크는 3D 물체, 이미지 및 비트를 원본 3D 장면 내에서 효과적으로 숨기고 정확하게 추출할 수 있는 능력을 보여줍니다. SecureGS의 도입으로 3DGS 스테가노그래피의 다양한 응용 가능성이 더욱 넓어질 것으로 기대됩니다.



### NeuraLoc: Visual Localization in Neural Implicit Map with Dual Complementary Features (https://arxiv.org/abs/2503.06117)
Comments:
          ICRA 2025

- **What's New**: 최근 신경 방사장 모델(NeRF)이 시각적 로컬라이제이션 분야에서 큰 주목을 받고 있지만, 기존 NeRF 기반 접근법은 기하학적 제약이 부족하거나 기능 매칭을 위해 방대한 저장 용량이 필요하여 실용성에 한계가 있다. 이를 해결하기 위해, 우리는 보완적 특징(complementary features)을 기반으로 한 효율적인 신경 암묵적 지도(neural implicit map)에 기초한 새로운 시각적 로컬라이제이션 접근법을 제안한다. 이를 통해 저장 용량을 줄이고 기하학적 제약을 강화하여 2D-3D 대응을 보다 정확하게 생성할 수 있다.

- **Technical Details**: 제안된 방법은 3D 키포인트 설명자 필드(descriptor field)를 암묵적으로 학습하여 포인트 클라우드 정보를 명시적으로 저장하지 않는다. 또한, 추가적인 의미적 맥락 특징(semantic contextual features) 필드를 도입하여 설명자의 의미적 애매성을 줄이고 정확한 매칭 그래프를 구축한다. 마지막으로, descriptor 유사성 분포 정렬을 통해 2D와 3D 특징 공간 간의 도메인 갭(domain gap)을 최소화함으로써 더욱 향상된 성능을 이끈다.

- **Performance Highlights**: 우리의 방법은 최근 NeRF 기반 접근법과 비교하여 3배 빠른 훈련 속도와 45배의 모델 저장 용량 감소를 달성하였다. 두 개의 널리 사용되는 데이터세트에서의 광범위한 실험 결과, 우리는 다른 최첨단 NeRF 기반 시각적 로컬라이제이션 방법들보다 우수한 성능을 보여주었다. 특히, 제안된 방법은 정확한 6-자유도(6-DoF) 포즈 추정에서의 뛰어난 성능을 보여줍니다.



### Feature Fusion Attention Network with CycleGAN for Image Dehazing, De-Snowing and De-Raining (https://arxiv.org/abs/2503.06107)
- **What's New**: 본 논문은 Feature Fusion Attention (FFA) 네트워크와 CycleGAN 아키텍처를 결합하여 이미지의 안개 제거를 위한 새로운 접근 방식을 제시합니다. 이 방법은 감독 학습(supervised learning)과 비감독 학습(unsupervised learning) 기술을 활용하여 안개를 효과적으로 제거하면서도 이미지의 핵심 세부사항을 보존할 수 있습니다. 제안된 하이브리드 아키텍처는 기존의 방법에 비해 PSNR 및 SSIM 점수에서 뛰어난 성능 향상을 보여줍니다.

- **Technical Details**: 논문에서는 CVPR 2019 데이터셋인 RESIDE와 DenseHaze를 사용하여 다양한 가상 및 실제 상황에서도 안개가 낀 이미지를 효과적으로 처리할 수 있음을 실험적으로 보여주었습니다. CycleGAN은 쌍을 이루지 않은 하늘 이미지와 깨끗한 이미지를 효과적으로 처리하며, 이 아키텍처 조합은 강력한 데이터 적응능력을 제공합니다. 모델은 FFA 네트워크와 CycleGAN의 통합을 통해 저해상도 상태에서의 이미지를 복원하는 데 필요한 다양한 기능을 학습합니다.

- **Performance Highlights**: 제안된 방법은 25, 20, 10, 5, 0개의 깨끗한 샘플에 대한 여러 모델 변형을 구현하고 평가하여 성능을 측정하였습니다. SSIM과 PSNR과 같은 지표를 통해 실시간으로 평가할 수 있는 웹 인터페이스를 개발했습니다. 또한, 기존 방법과 비교하여 통계적으로 유의미한 성능 향상을 확인하였습니다.



### Vision-aware Multimodal Prompt Tuning for Uploadable Multi-source Few-shot Domain Adaptation (https://arxiv.org/abs/2503.06106)
Comments:
          Accepted by AAAI 2025

- **What's New**: 본 논문은 기존의 다중 소스 도메인 몇 샷 적응(MFDA) 문맥에서 클립(CLIP)의 언어 기반 감독의 이점과 효율적인 프롬프트 전이가 주목을 받으면서 업로드 가능한 다중 소스 몇 샷 도메인 적응(UMFDA) 체계를 제안합니다. 이 새로운 체계는 에지 장치의 계산 부하를 최소화하면서도 제한된 주석 데이터로 효과적으로 엣지 협업 학습을 가능케 합니다. 또한, 비전 인식 멀티모달 프롬프트 조정 프레임워크(VAMP)가 도입되어 특정 도메인에 대한 텍스트 프롬프트를 안내하며, 의식적이면서도 효율적으로 각 엣지 모델을 최적화합니다.

- **Technical Details**: UMFDA는 에지 학습을 위해 분산된 방식으로 작동하는 독창적인 프레임워크로, ELIP의 고정된 구조에서 도메인 특정 이미지 추출기 및 텍스트 분류기를 구축하는 데 사용됩니다. 주요 기술적 발전은 교차 모달 의미 정렬(CSA)와 도메인 분포 정렬(DDA)을 통해 각 에지 모델 간의 협업 학습을 극대화하는 것입니다. VAMP 프레임워크는 서로 다른 손실함수를 통해 원활한 협업 학습을 지원하며, 그로 인해 도메인 정보의 인식 및 유지 또한 가능하게 됩니다.

- **Performance Highlights**: OfficeHome과 DomainNet 데이터셋에 대한 광범위한 실험을 통해 제안된 VAMP의 효과성을 입증했으며, 이전의 프롬프트 조정 방법들보다 더 나은 성능을 보였습니다. VAMP는 다중 소스 도메인에 걸쳐 중요한 도메인 정보를 효과적으로 통합하고 모델의 학습 능력을 향상시키는 데 기여합니다. 이 연구는 적은 리소스 환경에서도 강력한 성능을 발휘할 수 있도록 지원하여 실제 적용 가능성을 높이고 있습니다.



### Handwritten Digit Recognition: An Ensemble-Based Approach for Superior Performanc (https://arxiv.org/abs/2503.06104)
Comments:
          11 pages,6 figures

- **What's New**: 이번 논문에서는 손글씨 숫자 인식 분야에서 새로운 앙상블(ensemble) 기반 접근법을 제시합니다. 이 방법은 Convolutional Neural Networks (CNNs)와 전통적인 머신러닝 기법을 결합하여 인식 정확도와 강건성을 향상시키고자 합니다. 본 연구는 70,000개의 손글씨 숫자 이미지로 구성된 MNIST 데이터셋을 활용하여 평가됩니다.

- **Technical Details**: 하이브리드 모델은 CNNs를 특징 추출(feature extraction)에 사용하고, Support Vector Machines (SVMs)를 분류(classification)에 이용합니다. 이 조합을 통해 99.30%의 정확도를 달성하였으며, 데이터 증강(data augmentation) 및 다양한 앙상블 기법을 활용한 효과성도 탐구합니다.

- **Performance Highlights**: 본 연구 결과, 제안된 접근법은 높은 정확도를 달성할 뿐만 아니라 다양한 손글씨 스타일에서의 일반화(generalization) 능력도 개선됨을 보여줍니다. 이러한 발견은 더 신뢰할 수 있는 손글씨 숫자 인식 시스템 개발에 기여하며, 딥러닝과 전통적인 머신러닝 기법의 결합 가능성을 강조합니다.



### Patch-Depth Fusion: Dichotomous Image Segmentation via Fine-Grained Patch Strategy and Depth Integrity-Prior (https://arxiv.org/abs/2503.06100)
- **What's New**: 이번 논문에서는 고해상도 자연 이미지에서의 고정밀 이분 이미지 분할(Dichotomous Image Segmentation, DIS) 과제를 다루고 있습니다. 기존 방법들이 국소적 세부 사항에 집중한 반면, 객체의 완전성을 모델링하는 데 어려움이 있었던 점을 지적하며, 이는 Depth Anything Model v2가 생성한 의사 깊이 맵과 이미지 패치의 세부 정보들을 조화를 이루게 함으로써 해결할 수 있다고 주장합니다. 저자들은 이러한 통찰을 바탕으로 새로운 패치-깊이 융합 네트워크(Patch-Depth Fusion Network, PDFNet)를 설계하였습니다.

- **Technical Details**: PDFNet는 3가지 핵심 요소로 구성되며, 첫째로 멀티 모달 입력 융합을 통해 객체 인식을 향상시킵니다. 특히 패치 정교화 전략을 활용해 감도를 개선하며, 둘째로 의사 깊이 맵에 분포된 깊이 완전성 우선순위를 활용하여 균일성을 높이고자 합니다. 마지막으로 공유 인코더의 특성을 이용하고 간단한 깊이 정제 디코더를 통해 깊이 관련 정보를 더욱 정교하게 캡처할 수 있도록 합니다.

- **Performance Highlights**: DIS-5K 데이터 세트에서의 실험 결과, PDFNet은 최신 비확산(non-diffusion) 방법들보다 월등한 성능을 보여줍니다. 깊이 완전성 우선순리를 도입함으로써 PDFNet은 최신 확산 기반(diffusion-based) 방법들과 유사하거나 그 이상의 성능을 보이면서도 파라미터 수는 11% 이하로 유지할 수 있음을 입증했습니다.



### PointDiffuse: A Dual-Conditional Diffusion Model for Enhanced Point Cloud Semantic Segmentation (https://arxiv.org/abs/2503.06094)
Comments:
          8 pages, 3 figures, 7 tables

- **What's New**: 본 연구에서는 점 구름(3D Point Cloud)에서 의미적(segmentation) 분할을 위한 새로운 접근 방식을 제안합니다. 기존의 확산(또는 Diffusion) 모델을 고정된 점 위치에서 색상 대신 점 라벨을 생성하는 데 활용합니다. 이를 통해 점 구름에서의 소음 제거를 효율적으로 향상시키는 새로운 레이블 임베딩 메커니즘을 도입하였습니다.

- **Technical Details**: 제안된 모델(PointDiffuse)은 듀얼 조건을 활용하여 나쁜 소음으로부터 점 라벨을 생성하는 과정에서 초기 의미 정보를 제공하는 작업을 수행합니다. 이 모델은 포지션 조건을 통해 로컬 피쳐의 효율적인 조정과 구성을 가능하게 하며, 포인트 주파수 변환기(Point Frequency Transformer)를 사용해 고수준 컨텍스트 처리 성능을 강화합니다. 또한, Denoising PointNet을 통해 고해상도 점 구름의 처리 과정에서 계산 비용을 절감할 수 있습니다.

- **Performance Highlights**: 점Diffuse는 S3DIS, SWAN, ShapeNet 등 다섯 개의 벤치마크 데이터셋에서 상당한 성과를 보였습니다. 이 모델은 S3DIS의 Area 5에서 74.2%, 6-fold에서 81.2%의 현재 최고 mIoU를 달성하며, SWAN에서는 64.8%를 기록했습니다. 이러한 결과는 PointDiffuse 모델의 효과성과 점 구름에 대한 높은 적합성을 입증합니다.



### ZO-DARTS++: An Efficient and Size-Variable Zeroth-Order Neural Architecture Search Algorithm (https://arxiv.org/abs/2503.06092)
Comments:
          14 pages, 8 figures

- **What's New**: ZO-DARTS++는 Differentiable Neural Architecture Search (NAS) 방법의 새롭고 효율적인 접근을 제공합니다. 이 방법은 자원 제한 아래에서 성능과 자원 제약을 균형 있게 조정하기 위한 제로 차수 근사(zeroth-order approximation)를 통합하여 효과적인 기울기 처리를 가능하게 합니다. 또한, 해석 가능한 아키텍처 분포를 위해 온도 변화를 적용한 sparsemax 함수를 사용하였으며, 크기 가변 탐색 방식(size-variable search scheme)을 도입하여 컴팩트하면서도 정확한 아키텍처를 생성합니다.

- **Technical Details**: ZO-DARTS++는 기존의 DARTS 및 다른 차별화된 NAS 방법의 저효율성과 구조 선택 과정의 개선, 자원 소비 문제를 해결하기 위해 고안되었습니다. 제로 차수 근사 기법을 도입하여 고립적인 변수의 기울기를 쉽게 대체하여 탐색 과정의 정확성과 효율성을 높입니다. 또한, sparsemax 함수를 사용해 이해 가능성을 증가시키고, 자원 소비를 효과적으로 줄일 수 있는 크기 가변 탐색 체계를 제안했습니다.

- **Performance Highlights**: ZO-DARTS++는 의료 이미징 데이터셋을 기반으로 한 광범위한 테스트에서 평균 정확성을 표준 DARTS 기반 방법보다 최대 1.8% 향상시키고, 탐색 시간을 약 38.6% 단축시켰습니다. 자원 제한 변형을 통해 35% 이상의 파라미터 수를 줄이면서 경쟁력 있는 정확도 수준을 유지할 수 있음을 보여주었습니다. 이러한 점에서 ZO-DARTS++는 실제 의료 애플리케이션에 적합한 고품질의 자원 인식 DL 모델을 생성하는 데 있어 다재다능하고 효율적인 프레임워크를 제공합니다.



### Fish2Mesh Transformer: 3D Human Mesh Recovery from Egocentric Vision (https://arxiv.org/abs/2503.06089)
- **What's New**: 본 논문에서는 착용카메라의 1인칭 시점에서 사용자 신체의 자세와 형태를 추정하는 에고센트릭(egocentric) 인간 신체 추정 기술을 다룹니다. 특히, Fish2Mesh라는 새로운 모델을 소개하여 3D 에고센트릭 인간 메시 복구(3D human mesh recovery)에서의 한계를 극복하고자 합니다. 이 모델은 fisheye 이미지의 왜곡을 줄이기 위해 Swin Transformer와의 연계를 통해 새로운 에고센트릭 위치 임베딩 블록을 제안합니다.

- **Technical Details**: Fish2Mesh는 다중 작업 헤드를 활용하여 SMPL 파라미터 회귀와 카메라 변환을 동시에 추정하는 구조로, 3D 및 2D 관절을 보조 손실로 활용하여 모델 학습을 지원합니다. 또한, efish2Mesh는 기존 데이터셋의 부족 문제를 해결하기 위해 4D-Human 모델과 제3자 카메라를 사용한 훈련 데이터셋을 생성을 제안합니다. 이러한 기법들은 위치 임베딩의 비선형 왜곡을 설명할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과를 통해 Fish2Mesh는 기존의 최첨단 3D HMR 모델들을 능가하는 성능을 보여줍니다. 특히, 이 모델은 에고센트릭 시점에서의 신체 메시 복구 품질을 높이기 위해 데이터 수집과 모델링 기술을 개선하였으며, 실제 환경에서의 다양한 움직임을 성공적으로 포착하는데 기여합니다. 이는 다양한 일상 활동에서 신체 구조를 정밀하게 추정할 수 있는 가능성을 열어줍니다.



### Exploring Interpretability for Visual Prompt Tuning with Hierarchical Concepts (https://arxiv.org/abs/2503.06084)
Comments:
          10 pages, 9 figures

- **What's New**: 본 논문에서는 해석 가능성을 탐구하기 위해 계층적 개념 프로토타입을 도입하여 시각적 프롬프트 튜닝을 위한 새로운 프레임워크인 Interpretable Visual Prompt Tuning (IVPT)을 제안합니다. 이를 통해 시각적 프롬프트가 인간이 이해할 수 있는 의미적 개념과 연결되며, 각 개념은 이미지의 특정 영역에 해당합니다. 기존의 추상적인 프로프트 벡터 학습 접근법과는 다르게, IVPT는 이미지의 다양한 세부 정보를 계층적으로 설명하는 해석 가능한 프롬프트를 생성합니다.

- **Technical Details**: 이 연구에서는 미리 훈련된 Transformer 모델을 기반으로 시각적 프롬프트 튜닝을 수행하며, 이 과정에서 프롬프트 임베딩만을 업데이트하고 Transformer의 백본 구조는 고정해 두고 사용합니다. IVPT는 서로 다른 의미 수준을 가진 프롬프트들의 관계를 정립하기 위해 세밀한 세부사항을 캡처하는 정교한 프롬프트 토큰과 보다 광범위한 개념을 나타내는 대략적인 토큰 간의 정렬을 사용합니다. 이를 통해 카테고리에 구애받지 않는 개념 프로토타입을 학습하게 되어, 다양한 카테고리 간의 공통 개념을 포착할 수 있게 됩니다.

- **Performance Highlights**: IVPT는 정교한 분류 기준과 병리 이미지에 대한 포괄적인 정성 및 정량 평가를 통해 기존의 시각적 프롬프트 튜닝 방법과 기존의 해석 가능한 방법들에 비해 뛰어난 해석 가능성과 성능을 보여줍니다. 이 연구의 접근 방식은 모델의 예측 설명 시 더욱 일관된 결과를 제공하는데 기여하며, 안전이 중요한 분야인 의료 및 자율 주행 영역에서 AI 시스템의 신뢰성을 높이는 데 도움을 줄 것으로 기대됩니다.



### TransParking: A Dual-Decoder Transformer Framework with Soft Localization for End-to-End Automatic Parking (https://arxiv.org/abs/2503.06071)
- **What's New**: 최근 자율 주행 시스템에서는 완전 미분 가능한 end-to-end 시스템이 주목받고 있습니다. 본 논문에서는 전문가의 궤적을 기반으로 한 순수 비전 기반 변환기(Transformer) 모델을 소개하며, 이는 복잡한 주차 환경에서 정확한 주차를 가능하게 합니다. 제안된 모델은 카메라로 촬영된 데이터를 입력으로 받아 미래 궤적 좌표를 직접 출력합니다.

- **Technical Details**: 전통적인 주차 시스템은 일반적으로 인식(perception), 계획(planning), 제어(control) 세 가지 모듈로 구성됩니다. 본 논문은 비전 정보만을 기반으로 하는 end-to-end 자율 주행 접근 방식을 제안하고, 전문가의 궤적에서 샘플링한 궤적 포인트를 토큰으로 활용하여 encoder 및 Dual-decoder 구조로 궤적 예측을 수행합니다. 이 과정에서 x와 y 좌표를 분리하여 입력하여 더 정확한 궤적 예측을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, 본 모델의 다양한 오류가 기존 최첨단(end-to-end) 궤적 예측 알고리즘에 비해 약 50% 감소한 것으로 나타났습니다. 제안된 방법은 주차 조작을 위한 효과적인 솔루션을 제공하며, 기존의 연구들보다 정확하고 강력한 미래 궤적 예측을 성취하였습니다.



### A Novel Trustworthy Video Summarization Algorithm Through a Mixture of LoRA Experts (https://arxiv.org/abs/2503.06064)
- **What's New**: 이번 연구에서는 사용자 생성 콘텐츠가 급증하는 비디오 공유 플랫폼에서 비디오의 효율적인 검색과 탐색을 위한 새로운 접근 방식이 제안됩니다. MiLoRA-ViSum은 비디오 데이터의 복잡한 시간적 동역학과 공간적 관계를 효율적으로 포착하기 위해 설계된 새로운 모델입니다. 기존의 Video-llama에서 발생하는 자원 소모 문제를 해결하기 위해 저자들은 파라미터 수를 조절할 수 있는 방법을 모색했습니다.

- **Technical Details**: MiLoRA-ViSum은 전통적인 Low-Rank Adaptation (LoRA)을 혼합 전문가(mixture-of-experts) 패러다임으로 확장하여 시간적 및 공간적 적응 메커니즘을 통합합니다. 이 모델은 각각의 LoRA 전문가가 특정 시간적 또는 공간적 차원에 맞춰 조정되어 비디오 요약 작업을 수행합니다. 이 동적 통합 방식은 복잡한 비디오 데이터의 다양한 특성을 효과적으로 다루기 위해 특별히 설계되었습니다.

- **Performance Highlights**: 비교 평가에서 MiLoRA-ViSum은 VideoXum와 ActivityNet 데이터셋에서 최신 모델들과 비교해 최고의 요약 성능을 발휘했습니다. 또한, 다른 모델들에 비해 현저하게 낮은 계산 비용을 유지하면서도 효과성을 보장합니다. 혼합 전문가 전략과 이중 적응 메커니즘은 비디오 요약 기능을 향상시킬 수 있는 잠재력을 강조합니다.



### Multi-Layer Visual Feature Fusion in Multimodal LLMs: Methods, Analysis, and Best Practices (https://arxiv.org/abs/2503.06063)
Comments:
          Accepted by CVPR2025

- **What's New**: 이 논문은 다중 모드 큰 언어 모델(Multimodal Large Language Models, MLLMs)에서 시각적 특징의 층 선택 및 융합 전략에 대한 새로운 접근 방식을 제시합니다. 이전 방법들이 임의의 설계 선택에 의존하는 경향이 있었으나, 이 연구에서는 효과적인 시각층 선택과 최적 융합 방식에 대한 체계적인 조사를 실시했습니다. 실험 결과, 여러 단계에서의 시각적 특징 결합이 일반화 성능을 개선하지만 동일 단계의 추가 특징을 포함하면 성능이 저하됨을 발견했습니다.

- **Technical Details**: 이 논문에서는 세 가지 고유 아키텍처(pre-cross, post-cross, parallel)를 통해 ‘Gated Xatten’ 모듈을 사용하는 다양한 Internal Modular Fusion을 채택하였습니다. 또한, 사전 훈련을 위한 558K 개의 LAION-CC-SBU 이미지-텍스트 쌍이 포함된 데이터셋과 665K 개의 지시를 따르는 데이터셋을 주요 훈련 데이터로 사용했습니다. 제각기 다른 환경에서 MLLM을 평가하기 위해, 일반, OCR, CV-중심, 환각(Hallucination) 등 네 가지 유형으로 분류된 일곱 가지 기준을 준비했습니다.

- **Performance Highlights**: 다양한 융합 전략의 성능 차이를 선보였으며, 처음부터 여러 층을 융합하는 것이 효과적임을 입증했습니다. 외부 직접 융합이 내부 직접 융합보다 평균적으로 더 나은 성능을 보였습니다. 또한, 매개변수 또는 데이터셋 크기에서 모델 스케일링을 연구했으며, 최대 7B 파라미터로 제한된 실험을 통해 실용적인 맥락에서 의미 있는 결과를 제공하고자 하였습니다.



### Pathological Prior-Guided Multiple Instance Learning For Mitigating Catastrophic Forgetting in Breast Cancer Whole Slide Image Classification (https://arxiv.org/abs/2503.06056)
Comments:
          ICASSP2025(Oral)

- **What's New**: 이번 연구에서는 유방암 Whole Slide Images (WSI) 분류에서 catastrophic forgetting을 완화하기 위한 새로운 프레임워크인 PaGMIL을 제안합니다. PaGMIL은 MIL(multiple instance learning) 모델 아키텍처에 두 개의 핵심 구성 요소를 통합하며, 미세 및 거시적 병리학적 선행지식을 활용하여 보다 정확하고 다양한 패치를 선택할 수 있도록 설계되었습니다.

- **Technical Details**: PaGMIL의 주요 모듈은 Patch Selector (PS)와 Prompt Guide (PG)이며, PS 모듈은 병리학적 선행정보를 통해 패치를 선별하고, PG 모듈은 WSI를 썸네일로 변환해 프롬프트를 생성합니다. 이러한 패치 생성을 통해 미세한 병리 정보와 함께 각 작업에 적합한 분류 머리를 선택하는 구조를 갖추고 있어, 각 데이터셋에 대한 적응성을 높이고 있습니다.

- **Performance Highlights**: 연구에서는 PaGMIL이 여러 공공 유방암 데이터셋에서 지속적인 학습 성능을 평가한 결과, 기존의 CLAM 모델보다 현재 작업의 성능과 이전 작업의 지식 유지를 더 잘 균형 잡는 것으로 나타났습니다. PaGMIL은 이러한 균형을 통해 기존의 지속적인 학습 방법보다 더 효과적으로 성능을 발휘했습니다.



### DropletVideo: A Dataset and Approach to Explore Integral Spatio-Temporal Consistent Video Generation (https://arxiv.org/abs/2503.06053)
- **What's New**: 이 논문에서는 비디오 생성에서 공간 시간 일관성(spatio-temporal consistency)의 개념을 새롭게 정의하고, 플롯 진행(plot progression)과 카메라 기법(camera techniques) 간의 상호 작용을 강조합니다. 특히, 카메라의 움직임이 비디오 내 내러티브에 미치는 장기적인 영향을 고려하여, 이전의 콘텐츠가 이후 생성에 미치는 영향을 연구하였습니다. 이를 위해 DropletVideo-10M이라는 대규모 오픈 소스 데이터셋을 구축하고, 이 데이터셋을 기반으로 DropletVideo라는 비디오 생성 모델을 개발했습니다.

- **Technical Details**: DropletVideo-10M 데이터셋은 1천만 개의 비디오로 구성되어 있으며, 각 비디오에는 물체의 동작과 카메라의 움직임에 관한 세부 캡션이 추가되어 있습니다. 이 데이터셋은 비디오 생성에서 공간 시간 일관성을 보존하는 데 최적화되어 있어, 물체의 시각적 특성과 장면의 일관성을 유지할 수 있도록 설계되었습니다. DropletVideo 모델은 다변량 샘플링(variable frame rate sampling) 전략을 통해 비디오 생성 속도와 시각적 전환의 템포를 정밀하게 조절할 수 있습니다.

- **Performance Highlights**: DropletVideo 모델은 광범위한 실험을 통해 시간적 및 공간적 차원에서 콘텐츠 일관성을 효과적으로 유지하는 것으로 확인되었습니다. 연구에 의해, DropletVideo는 기존 모델에 비해 더 복잡한 다중 플롯 내러티브를 생성할 수 있는 가능성을 보여주었으며, 이는 카메라 움직임과 부드러운 장면 전환을 통해 이루어집니다. 이로써 오픈 소스 데이터셋과 모델이 일반 대중에게 공개되어 알고리즘 혁신을 촉진하고, 폐쇄형 모델에 대한 대안을 제시할 수 있기를 기대합니다.



### Improving SAM for Camouflaged Object Detection via Dual Stream Adapters (https://arxiv.org/abs/2503.06042)
- **What's New**: 이번 논문에서는 기존의 Segment Anything Model (SAM)의 단점을 보완하기 위해 CAMOUFLAGED Object Detection (COD) 작업으로 발전된 SAM-COD 모델을 제안합니다. SAM-COD는 RGB-D 입력을 사용하여 camouflaged 객체를 탐지하며, 새로운 dual stream adapters를 통해 RGB 이미지와 depth 이미지 간의 상호 보완적인 정보를 학습하도록 설계되었습니다. 이 접근 방식은 두 가지 유형의 이미지 임베딩을 정제하고 일관성을 유지하면서 예측 성능을 향상시키는 것을 목표로 합니다.

- **Technical Details**: 논문에서 제안된 SAM-COD는 기존 SAM 아키텍처를 유지하면서 dual stream adapters를 병렬로 적용하여 이미지 인코더의 attention block에 통합합니다. 이를 통해 RGB 이미지와 depth 이미지에서 얻는 상이한 정보를 활용하여 서로 다른 피쳐를 보완할 수 있습니다. 또한, bidirectional knowledge distillation을 활용하여 dual stream embeddings 간의 상관 관계를 극대화하며, 이를 통해 RGB 이미지에서의 dense semantic feature를 깊이 이미지의 structural feature와 결합하여 더 정교한 mask prediction을 수행합니다.

- **Performance Highlights**: 제안된 SAM-COD는 4개의 COD 벤치마크에서 실험을 진행한 결과, 기존의 SAM 모델 대비 뛰어난 탐지 성능 향상을 보여주었습니다. 또한 최신 상태의 결과(State-of-the-art)를 기록하며 특정 파인튜닝 패러다임을 갖춘 다양한 모델들과 비교에서도 우수한 결과를 나타냈습니다. 이러한 성과는 camouflaged imaging에 대한 효과적인 모델 배포에 기여할 것으로 기대됩니다.



### A Label-Free High-Precision Residual Moveout Picking Method for Travel Time Tomography based on Deep Learning (https://arxiv.org/abs/2503.06038)
- **What's New**: 본 연구에서는 Residual Moveout (RMO) 선별을 위한 새로운 딥러닝 기반의 캐스케이드 피킹 방법을 제안합니다. 해당 방법은 세그멘테이션 네트워크와 트렌드 회귀 기반의 포스트 프로세싱 기술을 통해 정확하고 견고한 RMO를 구별합니다. 또한, 합성 데이터셋을 활용한 데이터 합성 방식을 도입하여 실제 필드 데이터에서의 피킹을 효과적으로 수행할 수 있도록 합니다. 이 연구는 RMO 선별의 품질을 정량화하기 위한 새로운 지표를 제안하는 데도 중점을 두고 있습니다.

- **Technical Details**: 제안된 캐스케이드 RMO 피킹 프레임워크는 세그멘테이션 네트워크와 경사장 제약 포스트 프로세싱 기술을 결합합니다. 해당 방법은 총 4단계로 구성되어 있으며, 첫 번째 단계에서 CIG에서 혼합된 특징을 추출하고, 주의 세그멘테이션 네트워크에 입력하여 곡률 세그멘테이션 맵을 추론합니다. 이후 후속 단계에서는 경사장 제약을 중심으로 각 곡률을 추출하는 포스트 프로세싱을 수행하여, 곡률의 윤곽을 찾아내고 유사 곡률들을 클러스터링 하며, 베이지안 추론을 통해 견고한 RMO 피킹을 제공합니다.

- **Performance Highlights**: 실험 결과는 제안된 방법이 기존의 유사도 기반 방법에 비해 더 높은 피킹 밀도와 정확성을 달성함을 보여줍니다. 모델 데이터 및 실제 데이터를 기반으로 한 평가를 통해, 본 연구의 방법이 기존 방법들보다 더 효과적임을 입증했습니다. 또한, 제안된 새로운 지표를 통해 자동으로 피킹된 RMO의 품질을 정량화할 수 있는 기초를 마련했습니다.



### Towards Universal Text-driven CT Image Segmentation (https://arxiv.org/abs/2503.06030)
- **What's New**: OpenVocabCT는 3차원 CT 이미지에서 텍스트 기반 분할을 위해 사전 훈련된 비전-언어 모델로, 기존의 분할 모델들에 비해 더 넓은 적용 범위를 제공합니다. 특히, 대규모 CT 데이터셋인 CT-RATE를 활용하여 진단 보고서를 세부 장기 설명으로 분해하여 멀티-그레인 대조 학습을 수행합니다. 이를 통해, 병원 환경에서 자주 발생하는 다양한 텍스트 프롬프트에 효과적으로 대응할 수 있는 능력을 확보하였습니다.

- **Technical Details**: OpenVocabCT는 대규모 3D CT 이미지에 대해 비전-언어 모델을 사전 훈련하는 접근 방식을 사용하여, 장기 및 종양 분할을 위한 포괄적인 텍스트 기반 모델을 구축합니다. 이 모델은 CT-RATE 데이터셋에 대한 이미지-텍스트 쌍을 생성하고, 멀티-라벨 대조 손실을 활용하여 비전-언어 모델을 정렬시킵니다. 이를 통해, 모델은 다양한 프롬프트에 대해 유연하게 일반화할 수 있습니다.

- **Performance Highlights**: OpenVocabCT는 아홉 개의 공개 데이터셋에서 장기 및 종양 분할 작업을 평가하여 기존 방법들과 비교했을 때 우수한 성능을 보였습니다. 특히, 단일 목표 분할 작업에서 텍스트 기반 모델보다 더 나은 성능을 발휘하며, 자연어를 통한 모델과의 상호작용을 통해 임상의들의 사용성을 크게 향상시킵니다.



### Towards Ambiguity-Free Spatial Foundation Model: Rethinking and Decoupling Depth Ambiguity (https://arxiv.org/abs/2503.06014)
Comments:
          32 pages, 31 figures, github repo: this https URL

- **What's New**: 이 논문은 깊이 추정(depth estimation)의 복잡성을 해결하기 위한 새로운 접근 방식을 소개합니다. 특히, 투명한 장면에서는 단일 깊이 추정이 3D 구조를 온전히 표현하지 못하는 한계가 있습니다. 기존의 모델들은 결정론적(predeterministic) 예측에 국한되어 실제 다층 깊이를 간과하는 경향이 있습니다.

- **Technical Details**: 우리는 MD-3k라는 벤치마크를 제시하여 전문가 모델과 기초 모델의 깊이 편향을 드러내는 다층 공간 관계(label) 및 새로운 메트릭스를 도입했습니다. 또한 Laplacian Visual Prompting (LVP)라는 교육이 필요 없는 스펙트럼 프롬프팅 기법을 통해 RGB 입력을 라플라시안 변환하여 숨겨진 깊이를 추출합니다. 이는 모델 재교육 없이도 LVP로 추정된 깊이를 기존 RGB 기반 추정치와 결합하여 다층 깊이를 유도합니다.

- **Performance Highlights**: 결과적으로 LVP는 제로샷(zero-shot) 다층 깊이 추정에서의 유효성을 검증했으며, 이는 보다 강력하고 포괄적인 형상 조건 시각 생성(geometry-conditioned visual generation) 및 3D 기반 공간 추론(3D-grounded spatial reasoning)이 가능하게 합니다. 또한, 시간에 일관된 동영상 수준의 깊이 추론(temporal depth inference)도 지원하여 다양한 응용 분야에 기여할 전망입니다.



### End-to-End HOI Reconstruction Transformer with Graph-based Encoding (https://arxiv.org/abs/2503.06012)
- **What's New**: 이 논문에서는 End-to-End HOI Reconstruction Transformer with Graph-based Encoding(HOI-TG)라는 혁신적인 프레임워크를 제안하여 인간과 물체 간의 상호작용을 효과적으로 재구성하는 방법을 다룹니다. 이 방법은 자기 주의 메커니즘(self-attention mechanism)을 활용하여 인간과 물체 간의 상호작용을 암묵적으로 학습하며, 그래프 잔여 블록(graph residual blocks)을 통해 각기 다른 공간 구조 간의 토폴로지를 통합합니다. 이를 통해 전 세계 및 지역 표현 간의 균형을 잘 맞춥니다.

- **Technical Details**: HOI-TG는 초기 3D 코디네이트와 그리드 샘플링(grid sampling)된 특성을 결합하여 입력으로 사용하며, 자신의 변환기(transformer) 아키텍처 내의 Graph Convolutional 구조를 통해 인간과 물체의 로컬 정보를 통합합니다. 이 아키텍처는 감지된 상호작용을 모델링하면서, 복잡한 세부 사항에서의 구별 능력을 높입니다. 결과적으로, 이 방식은 BEHAVE 및 InterCap 데이터셋에서 최첨단 성능을 달성하였습니다.

- **Performance Highlights**: HOI-TG는 BEHAVE 데이터셋에서 3D 재구성 정확도에서 8.0% (인간) 및 5.0% (물체) 향상을 보였으며, InterCap 데이터셋에서는 각각 8.9% (인간) 및 8.6% (물체) 향상되었습니다. 또한, 사람과 물체의 접촉 정확도가 각각 3.4% 및 5.8% 증가했습니다. 실험 결과, HOI-TG 프레임워크는 상호작용과 복잡한 구조를 정확하게 모델링하는 데 있어 뛰어난 효과를 보임을 강조합니다.



### Integrating Frequency-Domain Representations with Low-Rank Adaptation in Vision-Language Models (https://arxiv.org/abs/2503.06003)
Comments:
          8 pages, 4 figures

- **What's New**: 이 연구는 비주얼 랭귀지 모델(Visual Language Model, VLM)의 새로운 프레임워크를 제안합니다. 이 모델은 주파수 도메인 변환(frequency domain transformations)과 저순위 적응(Low-Rank Adaptation, LoRA)을 활용하여 피처 추출을 향상시키고 효율성을 높입니다. 기존의 VLM들이 공간 도메인(spatial-domain) 표현에만 의존했던 반면, 본 연구의 접근법은 잡음이 있거나 시야가 낮은 상황에서도 강력한 성능을 발휘합니다.

- **Technical Details**: 저순위 근사(Low-Rank Approximation)의 중심 개념인 특잇값 분해(Singular Value Decomposition, SVD)를 통해 매트릭스를 세 개의 구성 요소로 분해합니다. 연구자는 주파수 도메인 변환과 LoRA를 통합하여 각 인풋과 출력 간의 상관관계를 최적화합니다. 이를 통해 모델은 고차원 데이터에서도 더욱 효과적으로 학습하고 적응할 수 있도록 설계되었습니다.

- **Performance Highlights**: 다양한 수준의 가우시안 노이즈가 포함된 벤치마크 데이터셋에서 제안된 모델을 평가한 결과, 평가 지표에서 CLIP ViT-L/14와 SigLIP 등 최신 VLM과 타협 없는 성능을 보여주었습니다. 정성적 분석 또한 모델이 복잡한 실제 이미지를 처리할 때 더 자세하고 맥락에 적합한 응답을 생성한다는 것을 밝혔습니다.



### MagicInfinite: Generating Infinite Talking Videos with Your Words and Voic (https://arxiv.org/abs/2503.05978)
Comments:
          MagicInfinite is publicly accessible at this https URL. More examples are at this https URL

- **What's New**: MagicInfinite는 기존의 초상 애니메이션 한계를 극복한 새로운 diffusion Transformer (DiT) 프레임워크입니다. 이 프레임워크는 사실적인 인간, 전신 형체, 그리고 스타일화된 애니메이션 캐릭터 전반에 걸쳐 높은 충실도를 제공합니다. 특히, 다양한 얼굴 포즈를 지원하고, 멀티 캐릭터 씬에서 정확한 화자 지정을 위해 입력 마스크를 활용하여 단일 또는 여러 캐릭터를 애니메이션화할 수 있습니다.

- **Technical Details**: MagicInfinite는 3D full-attention 메커니즘과 슬라이딩 윈도우 잡음 제거 전략을 통해 무한한 비디오 생성을 가능하게 하며, 시간적 일관성과 시각적 품질을 보장합니다. 또한, 두 단계의 커리큘럼 학습 체계를 통해 음성과 텍스트 통합을 활용하여 긴 시퀀스에 대한 유연한 다중 모드 제어를 가능하게 합니다. 지역별 마스크와 적응형 손실 함수를 사용하여 전역 텍스트 제어와 지역 오디오 안내의 균형을 맞추고 있습니다.

- **Performance Highlights**: MagicInfinite는 새로운 벤치마크에서 오디오-입술 동기화, 정체성 보존, 그리고 다양한 시나리오에서의 움직임 자연성 면에서 우수성을 입증했습니다. 특히, 복잡한 모션과 액션을 포함한 여러 상황에서도 뛰어난 시각 품질과 시간적 일관성을 유지합니다. 혁신적인 단계 및 분류기 없는 지도(CFG) 증류 기술을 통해 20배의 추론 속도 향상을 달성하여, GPU 자원 소모를 최소화하면서도 품질 손실 없이 낮은 시간 내에 비디오를 생성합니다.



### Is Your Video Language Model a Reliable Judge? (https://arxiv.org/abs/2503.05977)
- **What's New**: 비디오 언어 모델(Video Language Models, VLMs)의 사용이 증가함에 따라, 이들의 성능을 평가하는 방법과 관련된 연구가 진행되고 있습니다. 기존의 전문가 기반 평가 방식은 일관성과 확장성에서 한계가 있으며, 이러한 문제를 해결하기 위해 VLM을 사용하여 VLM을 평가하는 자동화 방법에 대한 관심이 높아지고 있습니다. 그러나 VLM이 평가지로서의 신뢰성에 대한 연구는 미비한 실정입니다.

- **Technical Details**: 이 논문에서는 VLM이 다른 VLM의 성능을 평가할 때의 한계를 조사하며, 특히 신뢰할 수 있는 모델과 그렇지 않은 모델을 혼합하여 사용했을 때 집단적 사고(Collective Thought) 접근이 평가의 신뢰성을 얼마나 향상시키는지를 분석합니다. 최근의 연구들은 일반적으로 하나의 VLM에게만 의존하지만, 이는 편향된 결과를 초래하거나 신뢰성을 저하시킬 수 있음을 강조합니다. 이를 위해 Video-LLaVA라는 낮은 성능 모델을 미세 조정(Fine-tuning)하고, 신뢰성에 영향을 미치는 요소를 탐구합니다.

- **Performance Highlights**: 연구 결과에 따르면, 신뢰성이 낮은 평가자가 포함된 집단적 평가가 반드시 최종 평가의 정확성을 개선하지 않는다는 사실이 밝혀졌습니다. 신뢰성이 낮은 모델은 결과에 노이즈를 추가하여 집단의 이점을 상쇄할 수 있다는 점이 발견되었습니다. 이러한 결과는 VLM의 평가 신뢰성을 높이기 위한 보다 정교한 방법의 필요성을 강조하며, 평가 프레임워크의 설계에 대한 통찰력을 제공합니다.



### Bayesian Fields: Task-driven Open-Set Semantic Gaussian Splatting (https://arxiv.org/abs/2503.05949)
- **What's New**: 이번 논문에서는 오픈셋(open-set) 의미 맵핑을 위한 과제를 해결하기 위해 새로운 방식인 Bayesian Fields를 제시합니다. 이 방법은 작업에 기반한 태스크 드리븐(task-driven) 접근 방식을 통해 객체의 세분화 질문을 다루고, 각기 다른 시점을 통한 관찰을 확률적 방식으로 융합하는 새로운 방법론을 제시합니다. Bayesian Fields는 높은 정밀도와 낮은 메모리 사용량으로 3D 장면을 재구성할 수 있게 해줍니다.

- **Technical Details**: Bayesian Fields는 3D Gaussian을 사용하여 장면을 세분화하고, 각 객체의 태스크 적합성을 확률적으로 평가합니다. 이 모델은 여러 2D 관찰을 통해 얻은 시맨틱(semiantics) 지식을 Bayesian 업데이트 방법론을 통해 통합합니다. 또한 객체와 태스크 사이의 관계를 더 정교하게 파악하기 위한 비율적인 해석을 제공합니다.

- **Performance Highlights**: Bayesian Fields는 기존 기법보다 빠르게 작업을 수행하며(수 분 내에 완료), 추가적인 학습이 필요하지 않습니다. 이 방법은 3D 객체 추출을 가능하게 하며, 이전의 방법들이 가졌던 높은 메모리 요구사항 문제를 해결하고 있습니다. 이 논문은 Bayesian Fields의 오픈 소스를 제공하여 연구자들이 쉽게 접근할 수 있도록 하고 있습니다.



### CASP: Compression of Large Multimodal Models Based on Attention Sparsity (https://arxiv.org/abs/2503.05936)
- **What's New**: 이번 연구에서는 대규모 다중 모달 모델(large multimodal models, LMMs)의 극단적인 압축 기법을 제안합니다. 기존 연구는 대규모 언어 모델(large language models, LLMs)의 압축에 주로 집중했으나, LMMs의 저비트 압축에 대한 연구는 상대적으로 부족했습니다. 우리는 주의 메트릭스의 희소성이 Query와 Key 가중치 행렬의 압축 오류를 제한한다는 것을 이론적으로 및 실험적으로 증명하였고, 이를 기반으로 CASP라는 모델 압축 기법을 소개합니다.

- **Technical Details**: CASP는 데이터 인식 저계수 분해를 사용하여 Query 및 Key 가중치 행렬을 압축합니다. 이후 최적 비트 할당 프로세스를 기반으로 모든 레이어에서 양자화를 수행합니다. 이 방법은 어떤 양자화 기법과도 호환되며, 최신 2비트 양자화 방법인 AQLM과 QuIP#를 평균 21% 향상시키는 성과를 보였습니다. CASP의 이론적 근거로는 주의 가중치의 압축 오류가 주의 맵의 희소성에 의해 제한된다는 점을 들 수 있습니다.

- **Performance Highlights**: CASP는 다양한 이미지 및 비디오 언어 기준을 기준으로 2비트 양자화 방법의 성능을 평균 35% 및 7% 향상시키는 결과를 불러왔습니다. 또한, CASP는 LLMs에도 적용될 수 있으며, 언어 전용 기준에서 AQLM과 QuIP#의 성능을 각각 평균 11% 및 2.7% 개선합니다. 본 연구는 LMM 압축을 위한 2비트 기법을 탐구한 최초의 연구로, LMM과 LLM 모두에 유효한 새로운 접근법을 제시합니다.



### Denoising Score Distillation: From Noisy Diffusion Pretraining to One-Step High-Quality Generation (https://arxiv.org/abs/2503.07578)
Comments:
          First Author and Second Author contributed equally to this work. The last two authors equally advised this work

- **What's New**: 이번 논문에서는 변수로 오염된 데이터에서 고품질 생성 모델을 훈련하는 새로운 기법인 Denoising Score Distillation(DSD)을 도입했습니다. DSD는 오로지 노이즈가 섞인 샘플로 확산 모델을 사전 훈련한 후, 이를 단일 단계 생성기로 정제된 출력을 생성할 수 있도록 변형합니다. 이 접근법은 고품질 훈련 데이터의 부족 문제를 해결하고 과학적 분야의 여러 응용에서 큰 잠재력을 보여줍니다.

- **Technical Details**: DSD는 확산 모델(training diffusion model)의 사전 훈련을 통해 소음 있는 데이터에서 생성 모델을 더욱 효과적으로 학습할 수 있도록 합니다. 이 과정에서 점진적인 노이즈 주입(forward process)과 노이즈 제거(reverse process) 기술을 활용하여 원본 데이터의 분포를 회수합니다. 특별히, DSD는 선형 모델 설정에서 노이즈 분포 점수에 대해 정렬함으로써 깨끗한 데이터 분포의 고유값 공간을 식별하는 메커니즘을 통해 생성 모델 개선을 달성합니다.

- **Performance Highlights**: DSD는 다양한 노이즈 수준 및 데이터셋에서 생성 성능을 일관되게 향상시키는 결과를 나타냈습니다. 특히 기존의 기대와는 다르게, DSD는 저품질 교사 모델을 통해 노이즈가 섞인 데이터에서도 샘플 품질을 크게 개선할 수 있다는 것을 보여줍니다. 이러한 성과는 실질적인 경험적 증거로 뒷받침되며, 이로 인해 노이즈 데이터의 가치에 대한 새로운 인식을 제공합니다.



### PointVLA: Injecting the 3D World into Vision-Language-Action Models (https://arxiv.org/abs/2503.07511)
- **What's New**: PointVLA는 미리 훈련된 비전-언어-행동 모델에 포인트 클라우드 입력을 통합하는 새로운 프레임워크를 제안합니다. 이 방법은 기존의 2D 데이터 세트를 활용하면서 3D 기능을 주입하여 모델의 공간적 이해를 향상시키고자 합니다. 실험을 통해 이 방법이 현재 가장 앞선 2D 모방 학습 기법들을 초월하는 성과를 거두었다는 것을 입증하였습니다.

- **Technical Details**: PointVLA는 3D 포인트 클라우드 입력을 주입하기 위해 경량 모듈 블록을 사용하는 방법론을 채택합니다. 특정 블록에서만 3D 특징을 주입하여 미리 훈련된 VLA의 특징 표현이 유지되도록 하며, 이 과정에서 간섭을 최소화합니다. 실험 결과, PointVLA는 RoboTwin 시뮬레이션 플랫폼 및 실제 로봇에서 성과를 발휘하였습니다.

- **Performance Highlights**: PointVLA는 일부 우수한 특성을 보여 줍니다: (1) 20개의 시연으로 네 가지 작업을 수행하는 Few-shot multi-tasking; (2) 실물과 사진을 구별할 수 있어 보다 안전하고 신뢰할 수 있는 작업 수행; (3) 테이블 높이에 적응할 수 있어 다양한 환경에서의 변화에 잘 대응합니다. 또한, 빠르게 변화하는 컨베이어 벨트에서 작업하는 것과 같이 장기 과제에서도 뛰어난 성과를 보였습니다.



### ADROIT: A Self-Supervised Framework for Learning Robust Representations for Active Learning (https://arxiv.org/abs/2503.07506)
- **What's New**: 이 논문은 작업 인식(Task Awareness)에 최적화된 샘플 선택을 통해 주석 비용을 줄이는 액티브 러닝(Active Learning) 프레임워크를 소개합니다. 새롭게 제안된 ADROIT 접근법은 VAE(Variational Autoencoder)를 기반으로 하며, 다양한 소스(재구성, 적대적, 자기 지도(Self-Supervised), 지식 증류(Knowledge Distillation), 분류 손실)를 통합하여 통합된 표현 학습을 수행합니다. ADROIT은 레이블이 있는 데이터와 없는 데이터를 모두 활용하여 잠재 코드(Latent Code)를 학습하며, 레이블이 있는 데이터와 함께 Proxy Classifier를 활용하여 작업 인식을 강화합니다.

- **Technical Details**: ADROIT 프레임워크는 통합 표현 생성기(VAE), 상태 판별기(State Discriminator), Proxy Task-Learner 및 분류기로 구성되어 있습니다. 상태 판별기는 레이블이 있는 데이터와 없는 데이터를 구분하여 정보가 풍부한 샘플을 선택하는 역할을 합니다. VAE와 상태 판별기 간의 동적 상호작용은 경쟁적인 환경을 형성하며, VAE는 판별기를 속이려 하고 상태 판별기는 레이블이 있는 입력과 없는 입력을 구분하는 법을 배웁니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋에 대한 실험 결과는 ADROIT이 기존의 여러 액티브 러닝 기본 모델에 비해 우수한 성능을 보인다는 것을 입증합니다. 이 논문에서는 ADROIT의 주요 구성 요소의 중요성을 확인하는 심층적인 분석(Ablation Study) 또한 수행하였습니다. ADROIT은 레이블이 주어진 데이터와 비주어진 데이터 모두를 사용하여 학습을 수행함으로써, 현재 방법들보다 효과적으로 모델의 성능을 향상시키는 것으로 나타났습니다.



### NeAS: 3D Reconstruction from X-ray Images using Neural Attenuation Surfac (https://arxiv.org/abs/2503.07491)
- **What's New**: 이번 연구에서 제안된 새로운 접근법은 Neural Attenuation Surface (NeAS)를 통해 2D X-ray 이미지에서 3D 구조를 더욱 정확하게 재구성할 수 있다는 점입니다. NeAS는 표면 기하학과 감쇠 계수 필드를 동시에 캡처하며, 부호화된 거리 함수(Signed Distance Function, SDF)를 활용하여 3D 표면을 효과적으로 추출합니다. 기존 기술의 한계를 극복하기 위해 복수의 물질을 포함하는 경우를 위한 아키텍처도 제안했습니다.

- **Technical Details**: NeAS는 X-ray 이미지에서 체적 재구성을 위한 새로운 프레임워크로, 기하학적 표면을 정확하게 재구성하기 위한 방법론을 제시합니다. 본 시스템은 여러 개의 감쇠 필드를 동시에 추정하여 뼈와 피부와 같은 여러 물질 표면을 정밀하게 처리할 수 있도록 설계되었습니다. 또한, 외적 매개변수에서 발생하는 오류를 보정하기 위한 포즈 정제 방법이 사용되었습니다.

- **Performance Highlights**: 실험 결과, NeAS는 2D X-ray 이미지만으로도 3D 표면을 정확하게 추출할 수 있는 능력을 보여주었습니다. 이 접근법은 새로운 시점 합성(novel view synthesis) 작업에서도 향상된 성능을 발휘하며, 서로 다른 물질이 포함된 장면에서도 신뢰할 수 있는 표면 추출이 가능함을 입증했습니다. 따라서 CT 재구성의 효율성을 높이면서도 방사선 노출을 줄일 수 있는 기술적 기초를 마련했습니다.



### CATPlan: Loss-based Collision Prediction in End-to-End Autonomous Driving (https://arxiv.org/abs/2503.07425)
- **What's New**: 최근 자율주행(AD) 시스템의 설계, 훈련 및 평가에 대한 관심이 높아지고 있다. 특히, 이러한 시스템이 예측하는 계획된 경로의 불확실성을 추정하는 것이 중요한 문제로 대두되고 있으며, 이를 통해 안전성과 강건성을 확보할 수 있다. 본 논문에서는 불확실성 정량화 문헌에서 제안된 손실 예측 방식을 응용하여 Collision Alert Transformer for Planning (CATPlan)이라고 불리는 경량 모듈을 도입한다.

- **Technical Details**: CATPlan은 주어진 AD 모델의 모션 및 계획 쿼리를 디코드하여 충돌 손실 값을 예측하는 데 훈련된다. 이 충돌 손실 값은 추론 시 이진 충돌 예측에 대한 신뢰도로 해석될 수 있다. NeuroNCAP 벤치마크를 통해 CATPlan을 안전-critical 시나리오에서 평가하며, 이는 현실감 있는 니어렌더링을 기반으로 한 폐쇄 루프 시뮬레이터이다.

- **Performance Highlights**: CATPlan은 GMM 기반 기준선에 비해 평균 정밀도에서 54.8%의 상대적인 개선을 달성하며, 안전한 자율주행 시스템으로 발전하는 데 기여할 수 있음을 보여준다. 또한, CATPlan은 실제 데이터에서 50%의 충돌을 포착하면서 45.6%의 정밀도를 기록하여 각종 평가에서 뛰어난 성능을 나타낸다.



### Skelite: Compact Neural Networks for Efficient Iterative Skeletonization (https://arxiv.org/abs/2503.07369)
- **What's New**: 본 연구에서는 iterative skeletonization 알고리즘을 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 synthetic data(합성 데이터), task-specific augmentation(작업별 증대), 그리고 모델 증류 전략을 활용하여 학습 가능한 신경망을 통해 얇고 연결된 skeleton을 생성합니다. 이전의 skeletonization 알고리즘보다 100배 빠른 속도를 자랑하며, 높은 정확도를 유지하면서 새로운 도메인으로의 일반화 능력을 보여줍니다.

- **Technical Details**: Skeletonization의 기본 원리는 이미지를 매개축(medial axis)으로 근사하는 얇은 표현으로 축소하는 것입니다. 제안된 알고리즘은 convolutional neural network(CNN)으로 매개화된 함수 f_θ를 사용하여 N단계에서 skeleton을 근사합니다. 각 반복 단계는 삭제 제안, 제안 평가, skeleton 및 이미지 업데이트의 세 가지 단계로 구성되며, max-pooling을 사용하여 연산의 미분 가능성을 유지합니다.

- **Performance Highlights**: 제안된 알고리즘은 2D 및 3D 데이터셋에서 실험을 통해 계산 효율성과 새로운 curvilinear 데이터셋에 대한 일반화 능력을 입증합니다. 본 방법은 segmentation 파이프라인에 통합했을 때 연속성을 보존하는 데 있어 향상된 성능을 보여 주며, 전통적인 topology-constrained 알고리즘에 비해 더 낮은 계산 비용으로 접근이 용이합니다.



### Dynamic Path Navigation for Motion Agents with LLM Reasoning (https://arxiv.org/abs/2503.07323)
- **What's New**: 이 논문은 대형 언어 모델(LLM)의 공간 경로 계획 및 장애물 없는 궤적 생성 능력을 탐구합니다. LLM은 사용자-에이전트 상호작용을 지원하고 복잡한 시스템에서 전역 제어를 제공하는 능력이 있어 탐색에 큰 잠재력이 있습니다. 저자들은 제로샷(Zero-shot) 탐색과 경로 생성을 연구하기 위해 새로운 데이터셋을 구축하고 평가 프로토콜을 제안하였습니다.

- **Technical Details**: 경로는 직선으로 연결된 앵커 포인트를 사용하여 표현되며, 다양한 방향으로의 이동을 가능하게 합니다. 이 접근법은 기존 방법에 비해 더 큰 유연성과 실용성을 제공하면서도 LLM에 대해 간단하고 직관적입니다. 연구 결과에 따르면, 작업이 잘 구조화되면 LLM이 장애물을 피하는 데 있어 상당한 계획 능력을 보이며, 자율적으로 생성된 동작으로 목표에 도달할 수 있습니다.

- **Performance Highlights**: 단일 LLM 모션 에이전트가 정적 환경에서 공간 추론 능력을 갖추고 있는 것 외에도, 이 능력은 동적 환경에서 다중 모션 에이전트의 조정으로 원활하게 일반화될 수 있습니다. 전통적인 접근 방식과는 달리, 우리는 훈련이 필요 없는 LLM 기반 방법을 통해 글로벌 동적 폐쇄 루프(planning) 계획 수립과 자율적인 충돌 문제 해결을 가능하게 합니다.



### Group-robust Sample Reweighting for Subpopulation Shifts via Influence Functions (https://arxiv.org/abs/2503.07315)
Comments:
          Accepted to the 13th International Conference on Learning Representations (ICLR 2025). Code is available at this https URL

- **What's New**: 최근 기계 학습 모델은 데이터 배포 중 서브인구 집단(shifts) 간의 성능이 불균형한 문제를 안고 있다. 이러한 서브인구 집단의 비율 변화로 인해 모델의 일반화 능력이 저하되는 것에 대한 대안을 제시하고 있다. 본 논문에서는 Group-robust Sample Reweighting (GSR)라는 새로운 접근 방식을 제안하여 고품질의 그룹 레이블을 재사용하여 그룹 레이블의 효율성을 높이고 있다.

- **Technical Details**: GSR은 두 단계로 이루어진 방법론으로, 첫 번째 단계에서 그룹 레이블이 없는 데이터를 통해 표현을 학습하고, 두 번째 단계에서는 역량 기반(deep learning) 기술인 influence functions를 사용하여 가중치 조정을 수행한다. Pseudo-hessian를 통해 고차 미분 계수를 활용함으로써 훈련 과정을 통틀어 계산할 필요 없이 샘플 가중치 업데이트를 효율적으로 추정할 수 있다. 이 방법은 last-layer retraining (LLR)을 통해 최적화 문제가 단순히 변경될 수 있도록 지원하여 복잡성을 줄인다.

- **Performance Highlights**: GSR은 지난해의 최첨단 방법에 비해 평균적으로 1.0%의 개선된 절대 worst-group accuracy를 달성했다. 특히, 같은 양의 그룹 레이블을 사용할 때 GSR이 다른 방법들보다 더 나은 성능을 보여주며, 학습 속도가 더 빠르고 경량화된 방식으로 실질적인 개선을 제공한다. 이러한 결과는 비전 및 자연어 처리(NLP) 작업에서의 그룹 강건성(group robustness)의 향상으로 입증되었다.



### Goal Conditioned Reinforcement Learning for Photo Finishing Tuning (https://arxiv.org/abs/2503.07300)
Comments:
          38th Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 이 연구에서는 자동 사진 마무리 조정을 위한 새로운 접근법으로 목표 조건 강화 학습(Goal-conditioned Reinforcement Learning) 프레임워크를 제안합니다. 기존 방법들은 비효율적이거나 매개변수 수의 증가 시 속도가 저하되는 한계가 있었으며, 이번 연구는 블랙 박스(black box) 이미지 처리 파이프라인을 효과적으로 조정할 수 있는 방법을 제공합니다. 이를 통해 사용자는 목표 이미지를 기반으로 보다 정교하게 매개변수를 튜닝할 수 있습니다.

- **Technical Details**: 제안된 방식은 목표 이미지와 현재 조정된 이미지를 입력으로 활용하여 보다 나은 매개변수 세트를 탐색합니다. 여기서 사용하는 상태 표현은 CNN 기반의 특징 표현, 사진 통계 representación, 그리고 과거 행동의 임베딩을 포함하여, 강화 학습 정책이 효과적으로 다음 매개변수를 생성할 수 있도록 돕습니다. 또한, 보상 함수가 사진 마무리 조정과 스타일화 조정을 위한 두 가지로 설계되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 강화 학습 기반 프레임워크가 효율성과 이미지 품질 측면에서 기존 방법들보다 상당히 뛰어난 성능을 보임을 확인했습니다. 목표 조건에 기반한 정책이 다양한 목표에 맞춰 이미지 처리 파이프라인 매개변수의 미세 조절을 수행할 수 있는 능력이 입증되었습니다. 예를 들어, RL 기반 조정은 38.92dB까지 도달했으며, 이는 기존 방법이 도달한 18.69dB보다 훨씬 높은 결과입니다.



### Efficient Distillation of Classifier-Free Guidance using Adapters (https://arxiv.org/abs/2503.07274)
- **What's New**: 이번 논문에서는 classifier-free guidance (CFG)의 비효율성을 줄이기 위해 adapter guidance distillation (AGD)이라는 새로운 접근 방식을 제안합니다. AGD는 하나의 전방향 패스 (forward pass)에서 CFG를 시뮬레이션하여 샘플링 속도를 두 배로 증가시키는 한편, 샘플의 품질을 유지하거나 개선하는 데 초점을 맞추고 있습니다. 기존의 디스틸레이션 방법이 전체 모델을 조정하는 데 비해, AGD는 기본 모델을 고정하고 최소한의 추가 파라미터만 훈련시킵니다.

- **Technical Details**: AGD는 lightweight adapters를 활용하여 CFG를 근사합니다. 이 방법은 약 2%의 파라미터만을 훈련시키기 때문에 디스틸레이션 단계에서의 자원 요구를 크게 줄일 수 있습니다. 이외에도, AGD는 기존의 가이드라인 디스틸레이션 방법의 훈련과 추론 간의 주요 불일치를 해결하며, 표준 확산 경로(diffusion trajectories) 대신 CFG 가이드 경로(CFG-guided trajectories)에서 훈련합니다.

- **Performance Highlights**: 광범위한 실험을 통해 AGD는 여러 아키텍처에서 CFG에 비해 FID(Frechet Inception Distance) 값이 유사하거나 우수한 성능을 달성하며, NFEs(Neural Function Evaluations)를 절반만 사용합니다. 특히, 제안된 방법은 약 26억 개의 파라미터를 가진 대형 모델을 단일 소비자 GPU(24GB VRAM)에서 디스틸할 수 있게 하여, 앞서의 접근 방식보다 더 많은 접근성을 제공합니다. 이 논문에서는 우리의 방법의 구현을 공개할 계획입니다.



### AI-Driven Automated Tool for Abdominal CT Body Composition Analysis in Gastrointestinal Cancer Managemen (https://arxiv.org/abs/2503.07248)
- **What's New**: 이번 연구에서는 중국에서 특히 높은 발생률을 보이는 위장관 암에 대한 신속하고 효율적인 처리를 위해 복부 CT 스캔 분석을 자동으로 수행하는 AI 기반 도구를 개발하였습니다. 이 도구는 복부의 근육, 피하 지방 및 내장 지방을 식별하고 세분화할 수 있도록 설계되었습니다. 또한, 사용자가 세분화 결과를 정제할 수 있는 인터랙티브한 인터페이스를 제공합니다.

- **Technical Details**: 이 도구는 다중 관점 로컬라이제이션 모델(multi-view localization model)과 고정밀 2D nnUNet 기반 세분화 모델을 통합하여 사용합니다. 복부 영역을 정확하게 감지하기 위해 다중 관점 융합 모듈을 적용하며, 최종 세분화 결과의 Dice Score Coefficient는 0.967로 보고되었습니다. 전반적으로 이 시스템은 복부 CT 이미지의 전처리 및 특징 추출을 통해 세부적인 분석을 가능하게 합니다.

- **Performance Highlights**: AI 도구의 성능은 90%의 로컬라이제이션 정확도와 함께 높은 세분화 품질을 입증하며, 복부 조직의 정량적 매개변수를 자동으로 계산할 수 있는 기능을 제공합니다. 환자 평가 및 치료 계획 수립에 있어 임상 의사에게 유용한 메트릭스를 제공함으로써 위장관 암 관리에서의 효율성을 크게 향상시킬 것으로 기대됩니다. 이 도구는 복부 조직 분석의 표준화된 방법을 제공하여 보다 효과적인 환자 관리에 기여할 것으로 보입니다.



### ReelWave: A Multi-Agent Framework Toward Professional Movie Sound Generation (https://arxiv.org/abs/2503.07217)
- **What's New**: ReelWave는 영화 제작 과정을 모방한 멀티 에이전트 프레임워크로, 비디오에 기반한 오디오 생성의 새로운 접근 방식을 제안합니다. 이 프레임워크는 청각 효과, 음악, 대화 등 다양한 요소를 포함한 보다 몰입감 있는 비주얼 스토리텔링을 목표로 합니다. 생성되는 오디오의 시간적으로 변화하는 요소인 loudness, pitch, timbre를 고려하여, 각기 다른 역할을 맡은 에이전트를 통해 감독의 지휘 하에 상호작용적으로 오디오를 생성할 수 있습니다.

- **Technical Details**: ReelWave는 전체 음향 디자인 과정을 감독하는 Sound Director 에이전트를 포함하여, 각기 다른 역할을 수행하는 여러 에이전트로 구성됩니다. 이 프레임워크는 비디오 상황에 대한 음향 예측 모듈을 통해 '온스크린' 음향을 생성하고, '오프스크린' 음향은 다양한 에이전트의 협력적 상호작용을 통해 보완됩니다. 이러한 다단계 에이전트 설정은 공동 작업의 품질을 향상시키고 생성된 오디오의 정밀도를 증가시킵니다.

- **Performance Highlights**: ReelWave는 다양한 비디오 장면에서 동기화된 오디오 생성을 가능하게 하며, 실제 영화 제작에 필요한 복잡한 오디오 작업을 처리하는 데 뛰어난 성능을 보입니다. 특히, 본 프레임워크는 자동 피드백 메커니즘을 통해 생성 과정에서의 품질을 지속적으로 개선할 수 있는 장점을 지닙니다. 마지막으로, ReelWave는 다양한 비주얼 및 시간적 조건을 통합하여 보다 정교하고 상황에 적합한 오디오 생성을 실현하고 있어, 전문적인 영화 음향 디자인에 기여할 수 있을 것으로 기대합니다.



### All That Glitters Is Not Gold: Key-Secured 3D Secrets within 3D Gaussian Splatting (https://arxiv.org/abs/2503.07191)
- **What's New**: 최신 3D 가우시안 스플래팅(3DGS)의 발전은 장면 재구성을 혁신적으로 변화시켰으며, 새로운 3D 스테가노그래피의 가능성을 열었습니다. 본 논문에서는 기존의 3DGS 기능을 최적화하고, 고충실도의 재구성을 유지하면서도 스테가노그래피의 탐지 위험을 줄이기 위한 Key-Secured 3D Steganography (KeySS) 프레임워크를 제안합니다. 이 접근법은 비밀 메시지의 발견을 방지하며, 여러 비밀 정보를 안전하게 숨길 수 있는 키 제어 메커니즘을 포함합니다.

- **Technical Details**: KeySS는 커버 3D 가우시안을 비밀 3D 가우시안으로 직접 변환하는 엔드-투-엔드 학습 프레임워크를 제공합니다. 이 과정에서 표준 3DGS 형식과 렌더링 파이프라인의 호환성을 유지하며, 광범위한 분석을 통해 가우시안 특성이 스테가노그래피의 효과에 불균형적으로 기여한다는 사실을 밝혔습니다. 또한, 3D-Sinkhorn 거리 분석이라는 새로운 보안 평가 메트릭스를 도입하여 보안과 충실도를 균형 있게 유지하는 방법을 제시합니다.

- **Performance Highlights**: 실험적으로, KeySS는 커버와 비밀 재구성 모두에서 최첨단 성능을 달성하며 높은 보안 수준을 유지합니다. 비밀 정보의 시각적 품질과 재구성 충실도가 뛰어나며, 부정한 추출 시도에 대한 강력한 저항력을 입증하였습니다. 이 연구는 3D 스테가노그래피 분야에서의 발전을 나타냅니다.



### The 4D Human Embryonic Brain Atlas: spatiotemporal atlas generation for rapid anatomical changes using first-trimester ultrasound from the Rotterdam Periconceptional Cohor (https://arxiv.org/abs/2503.07177)
- **What's New**: 이 연구에서는 태아의 뇌 발달에 대한 자세한 통찰력을 제공하기 위해 심층 학습 기반의 접근 방식을 활용하여 4D Human Embryonic Brain Atlas를 생성하였습니다. 이 아틀라스는 임신 8주에서 12주 사이의 3D 초음파 이미지를 기반으로 하여 생성되었습니다. 연구에서는 시간이 지남에 따라 초기 아틀라스를 설정하고 그에 대한 편차를 처벌함으로써 신생아 설계를 통해 빠른 발달을 유지하는 방법을 도입하였습니다.

- **Technical Details**: 제안된 방법은 3D 이미지를 통해 생성된 시간 및 형태적 아틀라스를 사용하는 것이며, 두 개의 신경망으로 구성되어 있습니다. 첫 번째 네트워크는 초기 아틀라스를 입력으로 받아 시간에 따라 아틀라스를 생성하며, 두 번째 네트워크는 비강체 등록(nonrigid registration) 네트워크로 아틀라스와 해당 이미지 간의 변형 필드를 출력합니다. 이 접근법은 빠른 해부학적 변화에 대해 아틀라스를 설정하기 위한 것입니다.

- **Performance Highlights**: 연구 결과, 시간 의존적 초기 아틀라스와 그에 대한 처벌을 포함한 접근 방식이 해부학적으로 정확한 결과를 만들어냈습니다. 아블레이션 연구(ablation study)를 통해 시간 의존 초기 아틀라스를 사용하는 것이 중요하다는 것을 입증하였고, 기존의 ex-vivo 아틀라스와의 비교를 통해 아틀라스의 해부학적 정확성이 확인되었습니다. 제안된 아틀라스는 조기 생애 기간의 뇌 발달을 이해하고, 태아 신경 발달 장애의 조기 진단과 예방, 치료 향상에 기여할 Potential을 가지고 있습니다.



### VidBot: Learning Generalizable 3D Actions from In-the-Wild 2D Human Videos for Zero-Shot Robotic Manipulation (https://arxiv.org/abs/2503.07135)
Comments:
          Accepted to CVPR 2025

- **What's New**: 본 연구에서는 일상적인 인간 비디오를 활용하여 로봇의 조작 기술을 제로샷(zero-shot)으로 학습할 수 있는 VidBot 프레임워크를 제안합니다. 기존에는 인체를 모사한 비디오와 같은 고비용의 군중 기반 데이터 수집 방법에 의존했으나, VidBot은 RGB 비디오에서 3D affordance를 추출하여 다양한 환경과 로봇 유형에 적용 가능하도록 합니다. 이를 통해 로봇 학습의 확장성을 획기적으로 개선할 수 있는 가능성을 제시합니다.

- **Technical Details**: VidBot은 구조에서 동작을 추출하는 Structure-from-Motion (SfM) 기법과 깊이 기반 모델을 결합하여 3D 손 궤적을 복원합니다. 이 시스템은 로봇이 다양한 환경에서 사람의 조작을 관찰하여 학습할 수 있는 3D affordance 표현을 생성하며, 초급 직관 행동을 추출한 후, 이를 바탕으로 정밀한 상호작용 궤적을 생성하기 위해 디퓨전 모델을 사용합니다. 이러한 방식은 로봇이 새로운 장면이나 형태에서도 신뢰성 있게 행동을 수행할 수 있도록 돕습니다.

- **Performance Highlights**: VidBot은 13개 조작 작업에서 제로샷 적용 설정을 통해 기존의 방법들보다 성능이 현저히 뛰어난 결과를 보였습니다. 이 모델은 실시간 시뮬레이션 및 실제 환경 모두에서 우수한 성능을 보이며, 로봇 시스템에 원활하게 배포될 수 있습니다. 이러한 연구는 일상적인 인간 비디오를 활용하여 로봇 학습을 더욱 효율적이고 확장가능하게 만드는 길을 열어줍니다.



### Towards Experience Replay for Class-Incremental Learning in Fully-Binary Networks (https://arxiv.org/abs/2503.07107)
- **What's New**: 이 논문에서는 Fully-Binarized Neural Networks (FBNNs)에서 Class Incremental Learning (CIL)을 가능하게 하는 방법을 제시합니다. 특히, 기존 Binary Neural Networks (BNNs)와는 달리, FBNNs의 설계 및 훈련 절차를 revisits하면서 새로운 메트릭과 함께 비교 분석을 수행합니다. 또한, loss balancing 기법을 통해 과거 클래스와 현재 클래스의 성능 간의 균형을 맞추는 방법을 탐구합니다. 마지막으로, Latent replay와 Native replay라는 두 가지 전통적 CIL 방법을 심층 비교합니다.

- **Technical Details**: FBNN은 효율적인 하드웨어 매핑을 제공하며, 모든 계산을 이진 입력으로 된 스칼라 곱으로 수행합니다. 이 네트워크는 단일 단계에서 훈련되고, 채널별 Batch Normalization을 피하기 위해 층 단위의 스케일링 팩터를 사용합니다. 또한, 입력 이미지 크기를 축소하기 위해 전역 평균 풀링을 채택하며, 입력 이미지의 픽셀을 휘도/크로미넌스 영역으로 변환하여 thermometer encoding 형식으로 전환합니다.

- **Performance Highlights**: CORE50 데이터셋을 통해 검증된 결과에서, 제안된 3Mb-FBNN 모델은 기존의 실수 기반 대형 NN 모델과 비교하여 동등하거나 더 나은 성능을 보였습니다. 특히 Latent replay는 Native replay보다 더 높은 정확도를 유지했으며, Native replay는 뛰어난 적응도를 보여주었습니다. 마지막으로, 제안된 기법은 메모리 풋프린트가 352배 작은 3Mb 모델에서도 향상된 성능을 발휘합니다.



### Global Context Is All You Need for Parallel Efficient Tractography Parcellation (https://arxiv.org/abs/2503.07104)
Comments:
          8 pages, 2 pages references, 3 figures, 2 tables

- **What's New**: 이번 연구에서는 PETParc라는 새로운 접근법을 제안하여, 병렬 효율적 트랙토그래피 파셀레이션(parcellation)을 수행하는 방법을 제시합니다. 이 방법은 트랜스포머(transformer) 아키텍처를 활용하고, 모든 스트림라인을 랜덤하게 서브 트랙토그램(sub-tractograms)으로 나누어 병렬로 분류합니다. PETParc는 기존의 TractCloud에 비해 처리 속도가 최대 두 배로 향상되며, GPU 없이도 임상 환경에서 사용할 수 있습니다.

- **Technical Details**: PETParc는 각 스트림라인을 단일 토큰으로 간주하며, 셀프 어텐션(self-attention) 메커니즘을 통해 분류에 관련된 다른 스트림라인들을 결정합니다. 이 모델은 고유한 flip-invariant embedding을 사용하여 스트림라인의 방향성 부족을 보완하며, 데이터 증강(data augmentation)의 일환으로 랜덤 플립(random flips)을 사용합니다. 이는 모델이 다양한 연령대 및 건강 상태에서도 일반화될 수 있도록 돕습니다.

- **Performance Highlights**: 이 새로운 접근법은 TractCloud보다 공격적인 상황에서도 더 나은 성능을 보이며, 특히 hemispherotomy를 받은 환자들에 대해 우수한 결과를 나타냅니다. 전반적으로 PETParc는 이전 방법들보다 개선된 정확도를 보이며, 두 가지 방법 모두에서 처리 속도가 크게 향상된 것으로 나타났습니다. 연구진은 결과를 바탕으로 모델의 코드와 사전 훈련된 모델을 공개할 예정입니다.



### A Comprehensive Survey on Magnetic Resonance Image Reconstruction (https://arxiv.org/abs/2503.07097)
- **What's New**: 이 논문은 기존의 MRI 제어 방법을 체계적으로 검토하고 새로운 딥러닝 기반 접근법을 포함한 다양한 MRI 재구성 방법을 제공합니다. 데이터 수집 및 전처리, 공개 데이터셋, 단일 및 다중 모달 재구성 모델, 훈련 전략 및 이미지 재구성 평가 지표를 다룹니다. MRI 재구성이 여전히 해결되지 않은 과제라는 점도 강조하고 있으며, 향후 개발 방향에 대한 통찰을 제공합니다.

- **Technical Details**: MRI는 비침습적 방법으로, 외부 자기장과 고주파(RF) 펄스를 이용하여 수소 원자의 스핀 행동을 촬영합니다. k-space에는 주파수 영역에서 신호가 저장되며, 이 데이터를 Inverse Fourier Transform을 통해 이미지로 재구성합니다. 연구자들은 특정 언더 샘플링 마스크를 적용하여 훈련 및 평가에 필요한 입력-레이블 쌍을 생성하고, 구조적 샘플링과 무작위 샘플링을 통해 다양한 재구성 방법을 개발하고 있습니다.

- **Performance Highlights**: 딥러닝 기반 재구성 방법은 언더 샘플링된 데이터를 효과적으로 학습하여 높은 품질의 이미지를 복원하는 데 도움이 되고 있습니다. 전반적으로 MRI의 재구성 품질이 향상됨으로써, 임상 진단의 정확성을 높이고 조기 질병 발견 및 치료 계획에 기여하고 있습니다. 이러한 발전은 환자의 치료와 결과를 개선하는 데 중요한 역할을 하며, MRI 재구성 연구의 향후 방향 제시에도 기여하고 있습니다.



### RS2V-L: Vehicle-Mounted LiDAR Data Generation from Roadside Sensor Observations (https://arxiv.org/abs/2503.07085)
Comments:
          7 pages, 4 figures

- **What's New**: 본 논문에서는 차량 장착 LiDAR 데이터를 도로 측면 센서 관측값에서 재구성하고 합성하는 새로운 프레임워크인 RS2V-L을 소개합니다. 이 방법은 도로 측면 LiDAR 포인트 클라우드를 차량 장착 LiDAR 좌표계로 변환하여 고충실도 데이터 생성을 가능하게 합니다. 이 연구는 도로 측면 센서 입력에서 차량 장착 LiDAR 데이터를 재구성하는 첫 번째 접근방식으로, 데이터 수집 비용을 줄이고 자율주행 모델의 강건성을 향상시킬 잠재력을 강조합니다.

- **Technical Details**: RS2V-L 시스템의 기술적 구현에 대한 상세 설명이 포함됩니다. 특히, 도로 측면 LiDAR 데이터를 차량 장착 좌표계로 매핑하는 좌표계 변환 및 데이터 정렬 프로세스가 소개됩니다. 이 방법은 객체 및 지면 포인트 클라우드에 대한 의미론적 분할을 위해 Patchwork++를 사용하고, 비지면 및 지면 포인트 클라우드를 별도로 모델링하여 차량 장착 LiDAR 포인트 클라우드를 합성합니다.

- **Performance Highlights**: 대규모 실험 평가를 통해 생성된 데이터를 모델 훈련에 통합하면, KITTI 데이터세트를 보완하여 3D 객체 감지 정확도가 30% 이상 향상됨을 보였습니다. 또한, 엔드 투 엔드 자율주행 데이터 생성의 효율성이 10배 이상 개선되는 것으로 나타났습니다. 이러한 결과는 제안된 방법의 효과성을 강력하게 입증하며, 저비용의 데이터 생성이 가능함을 보여줍니다.



### Multimodal Human-AI Synergy for Medical Imaging Quality Control: A Hybrid Intelligence Framework with Adaptive Dataset Curation and Closed-Loop Evaluation (https://arxiv.org/abs/2503.07032)
- **What's New**: 이번 연구에서는 의학 영상 품질 관리(QC)를 위한 표준화된 데이터셋과 평가 프레임워크를 구축하여 대형 언어 모델(LLM)들을 체계적으로 평가했습니다. 기존의 QC 방법들이 인력 소모가 크고 주관적이었던 반면, LLM들이 이러한 문제를 해결할 가능성을 보여주고 있습니다. 특히, 161개의 흉부 X-레이(Chest X-ray, CXR) 및 219개의 CT 보고서를 사용하여 기술적 오류 및 불일치를 탐지하는 데 초점을 맞추었습니다.

- **Technical Details**: 연구에서 사용된 데이터셋은 임상 경력이 15년 이상인 방사선 전문의 다섯 명의 감독 하에 수집된 161개의 흉부 X-레이 및 219개의 CT 보고서입니다. 이 데이터는 Python-GDCM 라이브러리를 사용하여 DICOM 헤더를 익명화하고, LLM 기반 자연어 처리(NLP) 기법으로 문서의 민감한 정보를 제거했습니다. 시각적 품질 확보를 위해 ACR 및 IEC 국제 가이드라인에 맞춘 11가지의 기술 기준을 적용하였고, CT 보고서 오류를 8개 주요 유형으로 체계화하였습니다.

- **Performance Highlights**: 실험 결과, Gemini 2.0-Flash는 CXR 과제에서 Macro F1 점수 90을 기록해 강력한 일반화 능력을 입증하였으나, 세부 성능은 제한적이었습니다. DeepSeek-R1은 CT 보고서 감사에서 62.23%의 재현율을 보여 다른 모델에 비해 우수한 성과를 기록했습니다. 반면, 몇몇 변형 모델의 성능은 떨어졌으며, InternLM2.5-7B-chat은 추가 발견률에서 가장 높은 성과를 보였으나 정확성은 떨어지는 경향을 보였습니다.



### CAPT: Class-Aware Prompt Tuning for Federated Long-Tailed Learning with Vision-Language Mod (https://arxiv.org/abs/2503.06993)
- **What's New**: 이 논문에서는 비독립적 및 비동일 분포(non-IID) 데이터와 긴 꼬리(long-tailed) 분포의 공동 문제를 효과적으로 처리하는 새로운 접근 방식인 Class-Aware Prompt Learning for Federated Long-tailed Learning (CAPT)를 제안합니다. CAPT는 사전 훈련된 비전-언어 모델(VLM)을 활용하여 데이터의 이질성과 긴 꼬리 분포를 동시에 처리할 수 있도록 설계되었습니다. 이 시스템은 일반 프롬프트와 클래스 인식 프롬프트를 결합하여 지식 공유와 협력을 효과적으로 이루어질 수 있도록 합니다.

- **Technical Details**: CAPT 메커니즘은 이중 프롬프트 구조를 도입하여, 도메인 불변 기능을 학습할 수 있는 일반 프롬프트와 각 클래스의 세부 정보를 캡처하는 클래스 인식 프롬프트를 제공합니다. 또한, 데이터 분포에 따라 클라이언트를 그룹화하는 이질성 인식 클러스터링 전략을 머리와 꼬리 클래스를 통합적으로 학습하는 데 활용합니다. 이를 통해 각 클라이언트가 유사한 긴 꼬리 특성을 가진 클라이언트와 효율적으로 협력할 수 있도록 지원합니다.

- **Performance Highlights**: CAPT는 다양한 긴 꼬리 데이터셋에 대한 광범위한 실험을 통해 머리 클래스와 꼬리 클래스 간의 성능 격차를 효과적으로 줄이고, 꼬리 클래스의 성능을 획기적으로 개선합니다. CAPT는 연합 학습(FL) 시나리오에서도 경쟁력 있는 전체 정확도를 유지하면서 꼬리 클래스 성능을 크게 향상시킵니다. 기존 최첨단 연합 긴 꼬리 학습 방법보다 더 우수한 성능을 보임으로써, 기존의 프롬프트 조정 방법의 한계를 극복하는 새로운 가능성을 열었습니다.



### Are We Truly Forgetting? A Critical Re-examination of Machine Unlearning Evaluation Protocols (https://arxiv.org/abs/2503.06991)
- **What's New**: 이 논문은 머신 언러닝(machine unlearning)의 새로운 평가 프레임워크를 제안합니다. 기존의 연구는 주로 정확도(logit-based metrics)에 중점을 두고 소규모 데이터셋에서의 언러닝을 평가했던 반면, 이 연구는 대규모 데이터셋에서의 표현 기반 평가를 통해 언러닝이 실제로 평균적인 표현을 제거하는지를 점검합니다. 이로써 머신 언러닝의 이론과 실제 간의 격차를 해소하는 데 기여하고자 하며, 향후 연구의 방향을 제시할 기반을 마련합니다.

- **Technical Details**: 기존의 머신 언러닝 알고리즘은 주로 로그 기반 평가(logit-based evaluation)에서 발생한 제한된 점에 대해 명확한 대안을 제시합니다. 새로운 평가 방법에서는 Centered Kernel Alignment (CKA)와 k-Nearest Neighbors (k-NN)와 같은 기술을 사용하여 인간 통계의 표현을 분석하고, 다양한 다운스트림 데이터셋에서 모델의 특징 표현 품질을 평가합니다. 이 연구는 모델의 특징적인 표현의 유사성을 고려하여 언러닝 방법들이 목표로 한 데이터를 진정으로 잊게 할 수 있는지를 평가합니다.

- **Performance Highlights**: 연구 결과, 최신 언러닝 알고리즘들이 대규모 언러닝 시나리오에서 실제로 효과적으로 작동하지 않는 경향이 있음을 발견하였습니다. 대규모 모델에서 언러닝 후의 표현이 원본 모델에 더 유사한 것으로 나타났으며, 이는 기존 언러닝 방법들이 정보 제거에 있어 한계가 있다는 것을 강조합니다.  새로운 평가 세트인 Top Class-wise Forgetting은 언러닝이 자주 평가되는 방식의 취약함을 극복하고, 보다 포괄적인 평가를 가능하게 합니다.



### Utilizing Jailbreak Probability to Attack and Safeguard Multimodal LLMs (https://arxiv.org/abs/2503.06989)
- **What's New**: 최근 다중 모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)은 다중 모달 콘텐츠를 이해하는 뛰어난 능력을 보여줍니다. 그러나 이들은 악의적인 반응을 생성할 수 있는 관여 공격(jailbreak attacks)에 취약합니다. 본 연구에서는 MLLM의 jailbreak 가능성을 수치화하기 위해 'jailbreak probability'를 도입하고 이를 예측할 수 있는 네트워크인 Jailbreak Probability Prediction Network (JPPN)를 제안합니다.

- **Technical Details**: 제안된 방법은 입력의 숨겨진 상태(hidden states)와 jailbreak 가능성 간의 관계를 모델링하고, 이를 통해 입력을 최적화합니다. 특히, Jailbreak-Probability-based Attack (JPA)을 통해 입력에 대한 공격을 최적화하고, Jailbreak-Probability-based Defense methods인 JPF 및 JPDN을 통해 MLLM의 파라미터와 입력 공간에서 jailbreak 가능성을 최소화합니다.

- **Performance Highlights**: 실험 결과, JPA는 기존 방법보다 공격 성공률에서 최대 28.38% 향상을 보였으며, JPF와 JPDN은 각각 60% 이상 jailbreaking를 감소시키는데 성공했습니다. 이 결과들은 입력의 jailbreak 가능성을 활용한 공격 및 방어 방법의 효과를 뒷받침합니다.



### Synchronized Video-to-Audio Generation via Mel Quantization-Continuum Decomposition (https://arxiv.org/abs/2503.06984)
Comments:
          Accepted to CVPR-25

- **What's New**: 이번 논문에서는 Mel Quantization-Continuum Decomposition (Mel-QCD)이라는 새로운 접근 방식을 통해 비디오에 효과적으로 동기화된 오디오 트랙을 생성하는 방법을 제안합니다. 비디오에서 필요한 신호를 추출하여 성숙한 텍스트-오디오 생성 확산 모델을 제어하는 방법을 중점적으로 다루며, mel-spectrogram의 표현을 완전성과 복잡성 측면에서 균형 있게 조정합니다. 이 연구는 오디오 생성을 위한 ControlNet과 텍스트 반전 디자인을 활용하여, 신호 예측과 멜의 재조합을 통한 오디오 생성 과정의 손질을 보여줍니다.

- **Technical Details**: Mel-QCD는 mel-spectrogram을 의미적, 에너지, 표준 편차 벡터의 세 가지 구성요소로 분해하여 각각을 양자화하거나 연속적으로 처리하는 방법을 사용합니다. 이 과정에서 세멘틱 벡터를 이산 코드로 양자화 하고, 의미 구성 요소를 나타내는 코드북을 작성하여 복잡성을 줄이는 동시에 정보의 손실을 최소화합니다. 또한, 낮은 차원 데이터인 에너지 및 표준 편차 벡터는 연속 분포 유지에 강한 의존성을 분석하여 원래 표현을 유지하는 것을 지지합니다.

- **Performance Highlights**: 제안된 Mel-QCD 방법은 VGGSound 벤치마크에서 기존의 주요 파이프라인과 비교하여 생성 품질, 동기화 및 의미 일관성의 세 가지 측면으로 평가하며, 여덟 가지의 지표에서 최첨단 성능을 입증하였습니다. 이 연구는 비디오에 의해 조건 지어진 고품질 오디오 생성을 위한 보다 효율적인 프레임워크를 탐색하는 것을 목표로 하며, 폭넓은 분석 실험을 통해 제안된 통찰력을 검증합니다.



### Dynamic Cross-Modal Feature Interaction Network for Hyperspectral and LiDAR Data Classification (https://arxiv.org/abs/2503.06945)
Comments:
          Accepted by IEEE TGRS 2025

- **What's New**: 이 논문에서는 하이퍼스펙트럴 이미지(HSI)와 LiDAR 데이터의 결합 분류를 위한 혁신적인 Dynamic Cross-Modal Feature Interaction Network (DCMNet)라는 새로운 프레임워크를 제안합니다. DCMNet은 HSI와 LiDAR의 분류를 위해 동적 라우팅 메커니즘을 활용하며, 세 가지 특징 상호작용 블록인 Bilinear Spatial Attention Block (BSAB), Bilinear Channel Attention Block (BCAB), 그리고 Integration Convolutional Block (ICB)를 도입합니다. 이러한 블록들은 공간적, 스펙트럼적, 그리고 변별적 특징 상호작용을 효과적으로 증진시키도록 설계되었습니다.

- **Technical Details**: DCMNet은 하이퍼스펙트럴 이미지와 LiDAR 데이터의 다중 소스 통합 분류를 위한 첫 번째 다층 동적 블록 구성의 프레임워크입니다. 각 블록은 HSI와 LiDAR 데이터 간의 다양한 특징 상호작용을 포착하기 위해 설계되었으며, 동적 라우팅 메커니즘을 통해 입력 데이터의 복잡성에 따라 최적의 계산 경로를 결정합니다. 이러한 접근 방식은 여러 소스의 보완적 특징을 활용하여 더욱 효과적인 결과를 도출합니다.

- **Performance Highlights**: 세 개의 공개 HSI 및 LiDAR 데이터셋에 대한 광범위한 실험 결과, DCMNet은 최신 기술들에 비해 뛰어난 성능을 보여주었습니다. 특히, 성능 비교 및 시각적 분석에서도 DCMNet의 우수성이 입증되었습니다. 이를 통해 DCMNet은 다중 소스 원격 감지 데이터의 효과적인 분류를 위한 새로운 기준을 제시합니다.



### CAFusion: Controllable Anatomical Synthesis of Perirectal Lymph Nodes via SDF-guided Diffusion (https://arxiv.org/abs/2503.06919)
- **What's New**: CAFusion이라는 새로운 접근법을 소개하며, 이는 의료 이미징에서의 합성 병변 생성을 혁신적으로 이끌어갈 것으로 기대됩니다. 기존 방법들은 주로 텍스처 합성에 초점을 맞추고 있었지만, CAFusion은 해부학적으로 복잡한 구조를 정밀하게 모델링하는 데 중점을 두었습니다. 특히, 이 방법은 signed distance functions (SDF)를 활용하여 매우 사실적인 3D 해부학적 구조를 생성하고, 해부학적 및 텍스처 특성에 대해 유연한 제어를 제공합니다.

- **Technical Details**: CAFusion은 제어 가능한 3D 합성을 위한 구조로, 특히 주위의 여러 변화에 적응할 수 있도록 고안되었습니다. 이 접근 방식은 anatomical guidance (AG)와 textural guidance (TG)를 통합하여 림프절의 형상과 신호 강도를 정밀하게 조정하며, 이는 형태학적 속성을 분리하여 생성하는 데 도움을 줍니다. 또한, Signed Distance Functions (SDF)를 통해 고유의 연속성을 활용하여 복잡한 형태를 보다 잘 모델링합니다.

- **Performance Highlights**: 실험 결과, 생성된 합성 데이터는 segmentation 성능을 크게 향상시켜, Dice coefficient가 6.45% 향상되었습니다. 특히, 경험이 풍부한 방사선의사들은 합성 병변과 실제 병변을 구별하는 데 어려움을 겪었으며, 이는 CAFusion의 모델이 수행하는 사실성과 해부학적 정확성이 매우 높음을 나타냅니다. 이로 인해 CAFusion이 의료 이미지 처리 응용 분야에서 높은 품질의 합성 병변을 생성하는 데 효과적임을 입증하였습니다.



### HIF: Height Interval Filtering for Efficient Dynamic Points Remova (https://arxiv.org/abs/2503.06863)
- **What's New**: 본 논문에서는 동적인 객체를 제거하기 위한 새로운 방식인 Height Interval Filtering (HIF) 방법을 제안합니다. 이 방법은 복잡한 환경에서의 지도 구축 시 성능 저하 문제를 해결하는 데 초점을 맞추고 있으며, 실시간 처리 요구사항을 충족할 수 있도록 설계되었습니다. 또한, 알려지지 않은 공간을 탐지할 수 있는 저높이 보존 전략을 새롭게 도입하여 차단된 영역에서의 오분류를 줄이도록 하였습니다. 실험 결과, HIF는 기존 SOTA 방법과 비교하여 7.7배의 시간 효율성을 보여주었습니다.

- **Technical Details**: HIF는 공간을 전역적으로 일관된 기둥(pillar)으로 분할하고, 해시 혼합 인덱싱(hash-mixed indexing) 작업을 통해 검색 차원을 줄여 계산 효율성을 높입니다. 이 과정은 베이지안 필터링(Bayesian filtering)을 통해 높이 간격을 업데이트하며, 동적 객체 제거를 위한 모든 단계에서 레이 추적(ray tracing) 및 지면 점(segmenting ground points) 세분화 모듈에 대한 의존도를 없앴습니다. 이러한 방식은 컴퓨터 자원 소모를 줄이고, 실시간 처리를 가능하게 합니다.

- **Performance Highlights**: KITTI 데이터셋을 기반으로 실험을 진행한 결과, HIF 방법은 동적 객체 제거 작업에서 FPS(fram per second) 성능이 기존 방법보다 6-7배 향상되었습니다. 그럼에도 불구하고 높은 정확도를 유지하고 있어 실시간 환경에서 안정적인 동작이 가능합니다. 이는 HIF가 동적 객체 제거 분야에서 기존의 많은 방법들보다 더 뛰어난 성능을 보여주는 것을 의미합니다.



### One-Shot Dual-Arm Imitation Learning (https://arxiv.org/abs/2503.06831)
Comments:
          Accepted at ICRA 2025. Project Webpage: this https URL

- **What's New**: 이번 논문에서는 One-Shot Dual-Arm Imitation Learning (ODIL) 방식을 소개합니다. ODIL은 이중 팔 로봇이 단 한 번의 시연만으로도 정밀하고 조정된 일상 작업을 배우도록 해줍니다. 이 방법은 새로운 3단계 시각 서보 제어(3-VS)를 사용하여 엔드 이펙터와 목표 물체 간의 정확한 정렬을 수행하며, 단일 시연 후에는 재생(replay)만으로 작업을 수행할 수 있도록 합니다. 추가 데이터 수집이나 교육 없이 이뤄지는 ODIL의 rob성은 다양한 상황에서도 안정성을 보여줍니다.

- **Technical Details**: ODIL은 기본 위치 제어기를 갖춘 이중 팔 로봇을 기반으로 하며, 하나의 눈-손(global camera) 카메라와 하나의 손-눈(wrist camera) 카메라를 사용하여 정밀하고 강력한 시각 정렬을 구현합니다. 이 시스템의 주요 과제는 단일 시연에서 조정된 경로(coordinated trajectory)를 추출하여 실제 환경에서의 데이터 수집 없이 정확한 정렬을 수행하는 것입니다. ODIL의 중심에는 Dual-Arm Coordination Paradigm과 3-VS 제어기가 결합되어 있으며, 이는 시연에서의 물체 상호작용 경로를 파라미터화합니다.

- **Performance Highlights**: ODIL은 4-DoF 및 6-DoF 설정에서 다양한 시나리오에 대해 실험을 수행함으로써 기존의 최신 기술들보다 월등한 성능을 발휘했습니다. 특히, 다른 최첨단 방법에 비해 정밀 및 조정된 작업에서 현저한 성과를 보였으며, 유사한 연구 결과들과의 비교를 통해 그 우수함을 입증했습니다. 이 연구는 기존의 이중 팔 조작 기술의 한계를 극복할 수 있는 길을 제시하며, 향후 로봇 기술의 발전에 기여할 것으로 기대됩니다.



### Towards a Multimodal MRI-Based Foundation Model for Multi-Level Feature Exploration in Segmentation, Molecular Subtyping, and Grading of Glioma (https://arxiv.org/abs/2503.06828)
- **What's New**: 본 논문은 비침습적인 (noninvasive) 방법으로 뇌종양 (glioma)의 정확한 특성을 분석하는 Multi-Task SWIN-UNETR (MTS-UNET) 모델을 제안합니다. 기존의 방법들이 조직 샘플링에 의존하여 종양의 공간 이질성을 포착하지 못했던 한계를 극복하고, 다중 과제를 동시에 수행하는 신기술을 도입했습니다. MTS-UNET는 대규모 뇌 영상 데이터로 사전 학습된 BrainSegFounder 모델을 기반으로 하여, 새로운 통합 모델의 가능성을 제시합니다.

- **Technical Details**: MTS-UNET는 종양의 세분화(glioma segmentation), 조직 등급(histological grading), 및 분자 세분화(molecular subtyping) 작업을 동시에 수행합니다. 이 모델은 주목할 만한 두 가지 모듈, 즉 다중 스케일 특징 추출(Tumor-Aware Feature Encoding, TAFE)와 IDH 변이와 관련된 섬세한 신호를 강조하는 교차 모달 차별 (Cross-Modality Differential, CMD)을 포함합니다. 2,249명의 다양한 뇌종양 환자 데이터로 교육 및 검증을 진행하였으며, 이를 통해 모델의 효용을 입증하였습니다.

- **Performance Highlights**: MTS-UNET는 세분화에서 평균 Dice 점수 84%를 달성했으며, IDH 변이에 대한 AUC는 90.58%, 1p/19q 동시 결실 예측에 대한 AUC는 69.22%, 등급 예측에 대해 87.54%를 기록했습니다. 이러한 결과는 기존의 기준 모델들보다 유의미하게 뛰어난 성과로, p<=0.05의 차이를 보였습니다. 구조적 기여를 통해 TAFE와 CMD 모듈의 필수성을 검증한 이번 연구는 다양한 MRI 데이터셋에서의 일반화 가능성을 강하게 보여주고, 비침습적이고 개인 맞춤형 뇌종양 관리의 발전 가능성을 제시합니다.



### Two-stage Deep Denoising with Self-guided Noise Attention for Multimodal Medical Images (https://arxiv.org/abs/2503.06827)
Comments:
          IEEE Transactions on Radiation and Plasma Medical Sciences (2024)

- **What's New**: 이 연구는 의료 이미지의 잡음을 제거하는 데 있어 인공지능(AI) 기반의 두 단계 학습 전략을 제안하며, 이를 통해 기존의 방법들이 가지는 시각적 아티팩트 문제를 해결하고자 합니다. 특히 연구에서는 잔여 잡음을 추정하고 이를 노이즈 주의 메커니즘(noise attention mechanism)과 결합하여 효과적으로 잡음을 제거하는 방식으로 이루어집니다. 또한, 다양한 의료 이미지 형태와 잡음 패턴을 일반화하기 위해 다중 모달 학습 전략(multimodal learning strategy)을 활용합니다.

- **Technical Details**: 제안하는 방법은 두 단계로 구성되어 있습니다. 첫 번째 단계에서는 주어진 노이즈 이미지로부터 잔여 잡음 패턴을 학습하며, 이후 두 번째 단계에서는 이 추정된 잡음을 이용해 결과물을 정제(refine)합니다. 특히, 픽셀 수준에서 노이즈와 입력 이미지와의 상관관계를 인지하는 새로운 자기 유도 노이즈 주의 메커니즘(self-guided noise attention)이 도입되어 있습니다. 이 연구는 약 30,000개의 의료 이미지를 수집하여 이를 기반으로 잡음 패턴을 합성(synthesize)하는 알고리즘을 개발했습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존의 의료 이미지 잡음 제거 방법들에 비해 정량적 및 정성적 비교 모두에서 월등한 성능을 보였습니다. 예를 들어, PSNR(피크 신호 대 잡음비)에서 7.64, SSIM(구조적 유사도 지수)에서 0.1021, DeltaE(ΔE)에서 0.80, VIFP(시각 정보 충실도)에서 0.1855, MSE(평균 제곱 오차)에서 18.54의 성능 향상을 기록하며 최신 기술 수준(state-of-the-art)을 달성했습니다. 이러한 성과들은 다양한 의료 이미지 모달리티와 잡음 패턴에 대한 일반화 가능성을 입증합니다.



### Semi-Supervised Medical Image Segmentation via Knowledge Mining from Large Models (https://arxiv.org/abs/2503.06816)
Comments:
          18 pages, 2 figures

- **What's New**: 본 연구는 대규모 비전 모델인 SAM(Segment Anything Model)의 폭넓은 시각적 지식을 활용하여 제한된 레이블 데이터에서 U-Net++ 모델의 성능을 향상시키는 전략적 지식 마이닝 방법을 제안합니다. 이를 통해 SAM의 출력을 활용하여 '유사 레이블(pseudo labels)'을 생성하고, 이를 통해 훈련 데이터셋을 풍부하게 하여 작은 딥러닝 모델의 성능을 극대화합니다. 연구 결과는 우리가 제안한 방법이 기존의 U-Net++ 모델을 초월하는 성과를 내었음을 보여줍니다.

- **Technical Details**:  연구에서는 Kvasir SEG와 COVID-QU-Ex 데이터셋을 활용하여 U-Net++ 모델을 제한된 레이블 데이터로 훈련한 후, SAM이 생성한 출력으로부터 유사 레이블을 추출하는 방법을 사용했습니다. 이 과정은 레이블이 없는 데이터셋에 대한 SAM의 일반화된 시각적 지식을 활용하며, U-Net++ 모델의 학습이 이를 통해 향상됩니다. SAM의 예측 결과를 개선하기 위해 iterative한 과정이 적용되며, 결과적으로 경량의 U-Net++가 높은 인퍼런스 속도를 유지하면서도 성능을 극대화합니다.

- **Performance Highlights**: 우리가 제안한 방법은 Kvasir SEG와 COVID-QU-Ex 데이터셋에서 각각 3%와 1% 성능 향상을 이루어냈습니다. 본 연구는 100% 레이블 데이터로만 훈련된 기존 U-Net++ 모델과 비교했을 때도 제안한 방법이 우수한 성과를 보였다고 강조합니다. 이러한 결과는 대규모 모델인 SAM의 폭넓은 지식을 활용한 지식 마이닝이 데이터 제한 사항을 극복할 수 있음을 보여줍니다.



### Unlocking Generalization for Robotics via Modularity and Sca (https://arxiv.org/abs/2503.06814)
Comments:
          CMU Robotics PhD Thesis, 185 pages

- **What's New**: 본 논문에서는 일반ist 로봇 시스템을 구축하는 방법을 제시합니다. 모듈성과 대규모 학습을 통합하여 여러 로봇 제어 작업을 수행할 수 있는 에이전트를 개발하는 것을 목표로 합니다. 특히, 각 모듈의 독립적인 일반화 능력을 활용해 효율적인 로봇 학습자를 만드는 방법에 대해 논의합니다.

- **Technical Details**: 논문에서는 로봇 학습 시스템에 모듈성과 계층 구조를 구축하는 데 초점을 맞춥니다. 모듈화를 계획을 통해 시행함으로써 에이전트가 더욱 효과적이고 능력 있는 로봇 학습자가 될 수 있도록 합니다. 대규모 데이터와 다양한 아키텍처, 감독 소스를 확보하기 위해 클래식 계획(classical planning)을 활용하여 시뮬레이션 내에서 정책 학습을 감독합니다.

- **Performance Highlights**: 결국, 모듈성과 대규모 정책 학습을 통합하여 제로샷 조작(zero-shot manipulation)을 성공적으로 수행할 수 있는 실제 로봇 시스템을 구축하는 방법을 제시합니다. 이 접근법을 통해 하나의 일반ist 에이전트가 실제 환경에서 복잡한 장기 조작 작업을 해결할 수 있음을 보여줍니다.



### Interactive Tumor Progression Modeling via Sketch-Based Image Editing (https://arxiv.org/abs/2503.06809)
Comments:
          9 pages, 4 figures

- **What's New**: 본 논문에서는 의료 이미징에서 종양 진행 상황을 정확하게 시각화하고 편집하는 방법으로 SkEditTumor라는 스케치 기반의 확산 모델을 제안합니다. 이 접근 방식은 스케치를 구조적 선행지식으로 활용하여 종양 지역의 정밀한 수정을 가능하게 하며, 구조적 무결성과 시각적 현실성을 유지합니다. SkEditTumor는 BraTS, LiTS, KiTS 및 MSD-Pancreas 등 4개의 공개 데이터셋에서 평가되었으며, 다룰 수 있는 장기와 이미징 모달리티가 다양합니다.

- **Technical Details**: 이 방법은 섹션 마스크를 생성하고 가장자리를 감지하여 선형 구조를 추출하는 Swin UNETR와 nnU-NET 기술을 통합합니다. 기존 스케치와 실제 심각함 간의 간극을 좁히기 위해, 랜덤 배단점과 왜곡 필드를 도입하여 스케치를 변형시키는 전략을 사용합니다. 이렇게 생성된 스케치는 U-Net 네트워크를 통해 정제되며, 실제 사용자가 제공한 스케치에 대한 모델의 적응성을 높입니다.

- **Performance Highlights**: 실험 결과는 SkEditTumor가 최첨단 기법을 초월하여, 이미지 충실도와 분할 정확도에서 우수한 성능을 보여주었다고 보고합니다. 이 연구는 의료 이미지 편집을 위한 스케치 기반 프레임워크를 도입하고, 종양 움직임에 대한 세밀한 제어를 가능하게 합니다. 다양한 데이터셋과 이미징 모달리티에서의 광범위한 검증을 통해 이 방법이 새로운 기준을 설정했음을 입증하였습니다.



### Robotic Ultrasound-Guided Femoral Artery Reconstruction of Anatomically-Representative Phantoms (https://arxiv.org/abs/2503.06795)
- **What's New**: 이 연구는 자율 로봇 시스템을 통해 이분화된 대퇴동맥의 초음파 스캐닝을 수행하고, 실제 환자의 CT 데이터를 기반으로 만든 다섯 개의 혈관 팬텀에서 이를 검증한 첫 번째 사례입니다. 연구팀은 혈관 이미징을 위한 비디오 기반의 딥러닝 초음파 세그멘테이션 네트워크를 제안하여 3D 동맥 재구성을 향상시킵니다. 제안된 네트워크는 새로운 혈관 데이터셋에서 89.21%의 Dice 점수와 80.54%의 Intersection over Union을 달성했습니다.

- **Technical Details**: 이 논문에서는 자율 로봇 시스템이 초음파(US) 이미지를 수집하고, 이를 바탕으로 동맥 세그멘테이션 및 3D 중심선을 재구성하는 방법을 소개합니다. 연구는 다섯 개의 환자 특화된 팬텀을 이용하여 자율 로봇 스캐닝 및 3D 재구성을 시연하였으며, 실제 CT 데이터에 대한 재구성 정확성을 종합적으로 평가하였습니다. 이 시스템은 기존의 단순화된 팬텀 모델로는 다루기 힘든 인체 해부학적 복잡성을 반영할 수 있도록 개발되었습니다.

- **Performance Highlights**: 재구성된 동맥 중앙선의 정확성은 원래의 CT 데이터와 비교하여 평균 L2 편차가 0.91±0.70mm, 평균 하우스도르프 거리(Hausdorff distance)가 4.36±1.11mm로 나타났습니다. 이는 제안된 로봇 시스템의 성능이 실제 환자 해부학에 대한 신뢰도 높은 평가를 제공함을 의미합니다. 또한, 연구는 자율 로봇 시스템의 혈관 이미징 및 개입에 대한 평가를 위한 더 철저한 프레임워크를 제시합니다.



### Infinite Leagues Under the Sea: Photorealistic 3D Underwater Terrain Generation by Latent Fractal Diffusion Models (https://arxiv.org/abs/2503.06784)
Comments:
          10 pages

- **What's New**: 이 논문은 해양 3D 지형의 표현 생성을 다루며, DreamSea라는 새로운 생성 모델을 소개합니다. DreamSea는 해양 로봇 조사에서 수집된 실제 이미지 데이터베이스를 기반으로 훈련되어, 매우 사실적인 해양 장면을 생성할 수 있도록 설계되었습니다. 이 모델은 노이즈와 아티팩트를 포함하는 실제 해저 이미지를 활용해, 고품질의 RGBD 기반 이미지를 생성합니다.

- **Technical Details**: DreamSea는 시각적 기초 모델을 사용하여 데이터에서 3D 기하학 및 의미 정보를 추출하고, 새로운 분산(latent embedding) 기반의 방식으로 생성된 이미지를 3D 맵으로 융합합니다. 훈련에는 깊이 센서 및 LiDAR와 같은 3D 스캔 정보가 없이 RGB 이미지만 사용되며, 이는 생성된 장면들이 공간적으로 일관된 기하 구조를 가지도록 합니다. 이러한 혁신적인 접근 방식은 데이터가 주석이 없는 경우에도 효과적으로 작동합니다.

- **Performance Highlights**: DreamSea는 대규모 해양 장면을 강력하게 생성할 수 있는 능력을 입증하였으며, 제작된 장면은 일관성과 다양성, 고품질 화풍이 특징입니다. 이 연구는 영화 제작, 게임 및 로봇 시뮬레이션과 같은 여러 분야에 걸쳐 영향을 미치며, 수중 환경의 3D 시뮬레이션 가능성을 크게 확장합니다.



### X-GAN: A Generative AI-Powered Unsupervised Model for High-Precision Segmentation of Retinal Main Vessels toward Early Detection of Glaucoma (https://arxiv.org/abs/2503.06743)
Comments:
          11 pages, 8 figures

- **What's New**: 본 논문에서는 X-GAN이라는 새로운 생성적 AI을 기반으로 한 비지도 학습(segmentation) 모델을 제안합니다. 이 모델은 Optical Coherence Tomography Angiography(OCTA) 이미지를 통해 주요 혈관을 추출하는 데에 최적화되어 있습니다. 특히 X-GAN은 레이블이 있는 데이터 없이도 거의 100%의 분할 정확도를 달성하며, 자원의 제약 없이 빠른 재구성을 가능하게 합니다. 또한 GSS-RetVein이라는 최신 데이터셋을 생성하여 데이터 부족 문제를 해결합니다.

- **Technical Details**: X-GAN의 경우 Space Colonization Algorithm(SCA)을 사용하여 혈관의 스켈레톤을 빠르게 생성하고, 이를 통해 2차원 및 3차원의 혈관 재구성을 실현합니다. 논문에서는 또한 GAN(generative adversarial networks)과 생물통계(biostatistics) 모델링을 통합하여 혈관 반경을 주요 요소로 활용하여 보다 정확한 혈관 세그멘테이션을 수행합니다. GSS-RetVein 데이터셋은 혼합 2D 및 3D 이미지를 제공하며, 세밀한 혈관 구조를 구성하여 모델의 강건성을 테스트하는 데 기여합니다.

- **Performance Highlights**: X-GAN은 GSS-RetVein, OCTA-500 및 ROSE 데이터셋에서 실시한 실험을 통해 최신 모델(SOTA) 보다 매우 뛰어난 성능을 입증하였습니다. 특히, X-GAN의 분할 정확도는 거의 100%에 달하며, GSS-RetVein 데이터셋의 우수성도 확인되었습니다. 이는 이 데이터셋이 주요 혈관 세그멘테이션 평가에 있어 신뢰할 수 있는 기준을 제공한다는 것을 의미합니다.



### Unsupervised Multi-Clustering and Decision-Making Strategies for 4D-STEM Orientation Mapping (https://arxiv.org/abs/2503.06699)
Comments:
          32 pages, 5 figures, 5 figures in SI

- **What's New**: 본 연구에서는 비지도 학습(un supervised learning)과 의사결정 전략(decision-making strategies)을 통합하여 4D-STEM 데이터셋의 고급 분석을 위한 새로운 방법을 제시합니다. 주요 클러스터링 방법으로 비음수 행렬 분해(non-negative matrix factorization, NMF)를 중점적으로 활용하며, 최적의 구성 요소(components) 수(k)를 결정하기 위한 체계적인 프레임워크를 도입하였습니다.

- **Technical Details**: K-Component Loss 방법과 이미지 품질 평가(Image Quality Assessment, IQA) 지표를 활용하여 재구성 충실도(reconstruction fidelity)와 모델 복잡성(model complexity)을 효과적으로 균형 잡았습니다. 데이터셋 전처리(preprocessing)가 클러스터링의 안정성과 정확도를 향상시키는 데 중요한 역할을 한다는 점도 강조하였습니다. 또한, 공간 가중치 행렬(spatial weight matrix) 분석을 통해 데이터셋 내 겹치는 영역에 대한 통찰을 제공합니다.

- **Performance Highlights**: NMF와 고급 IQA 지표 및 전처리 기술을 결합하여 신뢰할 수 있는 방향 매핑(orientation mapping) 및 구조 분석(structural analysis)을 가능하게 함을 보여주었습니다. 이러한 접근법은 다차원 물질 특성화(multi-dimensional material characterization)에서의 응용 가능성을 열어줍니다.



### What's in a Latent? Leveraging Diffusion Latent Space for Domain Generalization (https://arxiv.org/abs/2503.06698)
- **What's New**: 이번 논문은 Domain Generalization(도메인 일반화)을 위한 새로운 접근 방식을 제공합니다. 사전 훈련된 feature space(특징 공간)에서 pseudo-domain(유사 도메인) 구조를 발견하고 이를 통해 다채로운 unseen test domains(보지 못한 테스트 도메인)에 보다 잘 일반화될 수 있도록 합니다. 특히, diffusion models(확산 모델)에서 뛰어난 feature(특징) 분리 능력을 활용하여, 도메인 레이블 없이도 고유한 도메인 정보를 캡쳐하는 방법론을 제시합니다.

- **Technical Details**: 연구에서는 사전 훈련된 모델의 feature landscape(특징 풍경)를 이해하기 위해 다양한 pre-training objectives(사전 훈련 목표)와 architecture(구조)의 영향을 분석합니다. diffusion models의 특정 내부 상태가 photo styles(사진 스타일), camera angles(카메라 각도) 등의 추상적인 정보를 효과적으로 캡쳐할 수 있음을 발견했습니다. 주요 방법론으로, pseudo-domain 구조를 비지도 학습 방식으로 발견하고, 이 정보를 기존의 classifier(분류기)에 통합하여 성능을 개선하는 접근 방식을 도입합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 5개의 데이터셋에서 최대 4%의 테스트 정확도 개선을 보였으며, 특히 TerraIncognita 데이터셋에서는 기존의 Empirical Risk Minimization(ERM) 대비 4.3% 향상된 결과를 기록하였습니다. 이 연구는 최신 고급 모델들이 unseen domains에 대한 일반화 성능을 높이기 위한 새로운 방법을 제시하고 있으며, prototype network의 필요 없이 사전 훈련된 모델을 효과 활용하는 접근법의 중요성을 강조합니다.



### ImplicitCell: Resolution Cell Modeling of Joint Implicit Volume Reconstruction and Pose Refinement in Freehand 3D Ultrasound (https://arxiv.org/abs/2503.06686)
- **What's New**: 이번 연구에서는 ImplicitCell이라는 새로운 프레임워크를 제안하여 Implicit Neural Representation (INR)과 초음파 해상도 셀 모델을 결합하여 3D 초음파 재구성을 최적화합니다. 이 프레임워크는 노이즈가 있는 EM (electromagnetic) 추적 데이터를 사용하는 임상 환경에서도 강력한 성능을 보여줍니다. ImplicitCell은 초음파 이미지를 기반으로 하여, 픽셀 간의 공간적 연관성을 효과적으로 활용하여 더 높은 품질의 3D 재구성을 실현합니다.

- **Technical Details**: ImplicitCell 프레임워크는 초음파 해상도 셀 모델을 통합하여, 2D 초음파 이미지와 그에 대한 자세 신호를 통해 3D 볼륨 이미지를 생성합니다. 이 과정에서, 재구성 품질을 높이고 자세 정밀도를 개선하기 위해 물리 기반 모델링을 활용합니다. 연구에서는 다양한 정규화 기법을 사용하여 손 떨림과 노이즈와 같은 문제를 해결하고, 자유로운 손 3D 초음파 데이터의 특수성을 반영한 최적화 과정을 제시합니다.

- **Performance Highlights**: 실험 결과는 ImplicitCell이 기존 방법들에 비해 재구성 아티팩트를 현저히 줄이고, 특히 노이즈가 많은 데이터에서도 높은 품질의 볼륨 재구성을 가능하다는 것을 보여줍니다. Phantom, 자원 봉사자 및 임상 데이터셋을 포함한 다양한 테스트를 통해 프레임워크의 효용성을 입증하였습니다. 최종 결과는 의료 분야에서의 3D 초음파의 진단 정보 제공에 기여할 것으로 기대됩니다.



### AgiBot World Colosseo: A Large-scale Manipulation Platform for Scalable and Intelligent Embodied Systems (https://arxiv.org/abs/2503.06669)
Comments:
          Project website: this https URL, Code: this https URL

- **What's New**: 본 논문에서는 AgiBot World라는 대규모 로봇 데이터 플랫폼을 소개하며, 217개의 작업에서 100만 개 이상의 궤적을 포함하고 있어 기존 데이터셋에 비해 데이터 규모가 수십 배 확대되었습니다. AgiBot World는 고품질 및 다양한 데이터 분포를 보장하고 있으며, 손가락과 같은 정밀한 기술 습득을 지원합니다. 또한 Genie Operator-1 (GO-1)이라는 새로운 일반 정책을 도입하여 데이터 활용을 극대화하고, 증가된 데이터 양에 따라 성능이 예측 가능하게 향상됨을 입증합니다.

- **Technical Details**: AgiBot World는 다섯 가지 배치 시나리오를 포함한 4000제곱미터의 시설로 구성되어 있으며, 다양한 일상적인 시나리오에서 고충실도의 데이터 수집을 위해 설계되었습니다. 데이터 수집은 표준화된 파이프라인을 통해 수행되어 높은 품질과 확장성을 확보하고 있으며, 모든 실험은 여러 카메라 뷰와 시각-촉각 센서를 통해 멀티모달 데이터를 수집합니다. GO-1 정책은 비슷한 구조를 가진 데이터셋과 로봇의 이질적 데이터를 결합하여 훈련되며, 뛰어난 일반화 및 손재주를 보여줍니다.

- **Performance Highlights**: AgiBot World 데이터셋에서 사전 훈련된 정책은 Open X-Embodiment로 훈련된 정책보다 평균 30% 향상된 성과를 기록하였으며, 데이터를 적게 사용한 경우에도 18%의 일반화 개선을 보였습니다. GO-1은 복잡한 작업에서 60% 이상의 성공률을 달성하고, 이전 RDT 방법보다 32% 성능이 향상되었습니다. 이러한 성과는 AgiBot World의 데이터셋이 실제 로봇 응용에서 실질적인 변화를 이끌 수 있음을 보여줍니다.



### Speech Audio Generation from dynamic MRI via a Knowledge Enhanced Conditional Variational Autoencoder (https://arxiv.org/abs/2503.06588)
- **What's New**: 본 논문은 동적 MRI(자기공명영상)로부터 음성 오디오 신호를 생성하기 위한 새로운 접근법인 Knowledge Enhanced Conditional Variational Autoencoder (KE-CVAE)를 제안합니다. 최신 기술을 통해 무연산 MRI 데이터 통합 및 변분추론 아키텍처를 포함해 음성 생성 모델링 능력을 향상시킵니다. KE-CVAE는 이미지를 직접 음성으로 변환하는 혁신적인 두 단계 프레임워크를 기반으로 하며, 이는 현재까지의 연구 중에서 중요한 첫 번째 시도 중 하나로 평가됩니다.

- **Technical Details**: 제안된 KE-CVAE 모델은 교사 네트워크와 학생 네트워크로 구성되어 있으며, 방문 변환기(vision transformer) 기반의 구조를 사용합니다. 모델은 대규모의 목소리 기구 MRI 데이터를 활용하여 학습하며, 세 가지 보완적인 손실 함수를 통해 효과적으로 잠재 변수를 학습합니다. 각 손실 함수는 일관성 손실, 재구성 손실, KoLeo 정규화로 구성되어 있으며, 이를 통해 다양한 음성 형상 구조를 세밀히 추출합니다.

- **Performance Highlights**: 실험 결과, KE-CVAE는 기존의 딥 러닝 기반 합성 방법을 초월하여 자연스럽고 정확한 음성 파형을 생성하는 데 효과적임을 입증하였습니다. 본 연구는 정량적 메트릭(상관관계, PESQ 등)과 주관적 MOS 점수에서 모두 우수한 성과를 보였으며, 이는 동적 MRI의 독특한 음향 문제를 해결하는 데 기여합니다.



### LSA: Latent Style Augmentation Towards Stain-Agnostic Cervical Cancer Screening (https://arxiv.org/abs/2503.06563)
- **What's New**: 이번 연구에서는 Latent Style Augmentation (LSA)이라는 새로운 프레임워크를 제안하여 자궁경부암 진단을 위한 Whole Slide Images (WSIs)에서 staining 변이를 효과적으로 보완하는 방법을 소개합니다. 기존의 patch-level stain augmentation 방법들은 gigapixel 해상도의 WSIs에 확장할 때 두 가지 주요 제한점을 보였습니다. LSA는 WSI 수준에서 발생하는 latent features에 직접적으로 온라인으로 stain augmentation을 수행함으로써 이 문제를 해결합니다.

- **Technical Details**: LSA는 WSI의 latent feature에서 stain augmentation을 수행하는 새로운 접근법을 제공합니다. WSAug라는 방법을 통해 전체 WSI의 패치들 간에 stain을 일관되게 유지하면서 데이터 효율성을 개선합니다. Stain Transformer를 설계하여 latent space 내에서 특정 스타일을 시뮬레이션하며, 이를 통해 WSI 수준의 분류기의 강건함을 높입니다.

- **Performance Highlights**: multi-scanner WSI 데이터셋을 통해 검증한 결과, 단일 스캐너 데이터로 훈련된 LSAs이 다른 스캐너의 데이터에서 우수한 성능 향상을 보였습니다. 이러한 결과는 LSA의 접근 방식이 기존의 도메인 변화 문제를 해결하며, 다양한 스캐너에서 안정적인 진단 성능을 유지할 수 있음을 시사합니다.



### ProJudge: A Multi-Modal Multi-Discipline Benchmark and Instruction-Tuning Dataset for MLLM-based Process Judges (https://arxiv.org/abs/2503.06553)
- **What's New**: 최근 다중 모달 대형 언어 모델(MLLMs)은 과학 문제를 해결하는 데 인상적인 능력을 보여주고 있습니다. 하지만 이 모델들은 종종 비효율적인 추론을 통해 올바른 답변을 생성하기 때문에 중간 과정에 대한 평가의 중요성이 강조됩니다. ProJudgeBench라는 새로운 벤치마크가 제안되어 MLLM 기반 프로세스 판단자의 평가 능력을 포함한 포괄적이고 체계적인 시험을 제공합니다.

- **Technical Details**: ProJudgeBench는 2,400개의 테스트 사례와 50,118개의 단계 수준의 레이블을 포함하고 있으며, 수학, 물리학, 화학, 생물학과 같은 다양한 과학 분야를 아우릅니다. 각 단계는 전문인력에 의해 정확성, 오류 유형 및 설명으로 세심하게 주석이 달려 있어, 모델의 오류 탐지 및 진단 능력을 체계적으로 평가할 수 있습니다. 이 연구에서는 또한 ProJudge-173k라는 대규모 지침 조정 데이터세트를 제안하고, 모델이 문제 해결 과정에서 명시적으로 추론하도록 요구하는 동적 이중 단계 조정 전략을 통해 평가 능력을 높이고자 합니다.

- **Performance Highlights**: ProJudgeBench를 통해 개방형 소스 모델과 상용 모델 간의 성능 차이가 상당히 크게 나타났습니다. ProJudge-173k와 DDP 전략을 통해 개방형 소스 모델의 프로세스 평가 능력을 크게 향상시킬 수 있었으며, 이들 개선 사항은 프로세스 평가의 신뢰성을 높이는 데 기여합니다. 모든 자원은 향후 신뢰할 수 있는 다중 모달 프로세스 평가 연구를 촉진하기 위해 공개될 예정입니다.



### AnywhereDoor: Multi-Target Backdoor Attacks on Object Detection (https://arxiv.org/abs/2503.06529)
- **What's New**: 이 논문은 객체 탐지에 대한 다중 대상 백도어 공격 방식인 AnywhereDoor를 제안합니다. 기존 연구들은 단일 목표 설정에 기반했으나, AnywhereDoor는 공격자가 추론 시 컨텍스트에 따라 다양한 목표를 선택할 수 있도록 합니다. 이 방법은 물체가 사라지거나, 새로운 물체를 생성하거나, 잘못된 레이블을 부여하는 등의 유연성을 제공합니다.

- **Technical Details**: AnywhereDoor는 세 가지 주요 혁신을 통해 구현됩니다. 첫째, 목표를 분리하여 지원하는 목표 수를 확장하는 objective disentanglement; 둘째, region 기반 탐지기를 상대로 강력한 영향을 보장하는 trigger mosaicking; 셋째, 객체 수준 데이터 불균형 문제를 해결하기 위한 전략적 배치 전략이 포함되어 있습니다. 이러한 접근은 공격 성공률을 기존 방법에 비해 26% 향상시킵니다.

- **Performance Highlights**: 다양한 객체 탐지 알고리즘과 데이터셋에 대한 광범위한 실험 결과, AnywhereDoor는 기존 방법보다 공격 성공률이 26% 향상됨을 입증했습니다. 또한, 공격이 이뤄질 때 깨끗한 샘플에 대한 성능이 유지되는 것도 확인되었습니다. 이 연구는 객체 탐지 보안 연구의 새로운 방향을 제시합니다.



### GFlowVLM: Enhancing Multi-step Reasoning in Vision-Language Models with Generative Flow Networks (https://arxiv.org/abs/2503.06514)
- **What's New**: 이 논문에서는 VLM(비전-언어 모델)의 한계 극복을 위한 새로운 프레임워크인 GFlowVLM을 제안합니다. 기존의 Supervised Fine-Tuning(SFT) 및 Reinforcement Learning(RL) 방법들이 가지는 한계를 인식하고, Generative Flow Networks(GFlowNets)를 활용하여 다채로운 해결책을 생성하는 구조적 사고를 지원하는 방식으로 발전했습니다. GFlowVLM은 비선형 결정을 모델링하여 복잡한 추론 작업에서 장기적 의존성을 캡쳐할 수 있습니다.

- **Technical Details**: GFlowVLM은 GFlowNets를 사용하여 비선형 차원에서 VLM을 파인 튜닝하는 방식입니다. 이는 상태 간의 논리적 의존성을 고려하여 연속적인 상태에서 구조적 추론 프로세스를 강화할 수 있도록 도와줍니다. 저자들은 GFlowNets가 제시하는 보상 함수에 기반하여 다양한 샘플링 전략을 이용해 높은 보상을 받을 수 있는 궤적을 통해 방대한 문제 공간에서의 제한적인 솔루션을 극복하고자 했습니다.

- **Performance Highlights**: GFlowVLM의 성능은 카드 게임(NumberLine, BlackJack) 및 구현 계획(ALFWorld)과 같은 복잡한 작업에서 검증되었습니다. 이 프레임워크는 기존의 SFT 및 RL 기반 방법들과 비교하여 높은 교육 효율성과 다양한 솔루션 생성 능력을 보여주며, 특히 일반화 성능이 향상되었음을 입증합니다. 세부 실험 결과는 GFlowVLM이 성공률 및 문제 해결의 다양성을 높이는 데 기여함을 분명하게 나타냅니다.



### VisualSimpleQA: A Benchmark for Decoupled Evaluation of Large Vision-Language Models in Fact-Seeking Question Answering (https://arxiv.org/abs/2503.06492)
- **What's New**: 이번 논문에서 소개하는 VisualSimpleQA는 대형 비전-언어 모델(LVLM) 평가를 위한 새로운 다중 모달 사실 탐색 벤치마크입니다. VisualSimpleQA의 특징으로는 비주얼(visual)과 언어적(linguistic) 평가를 단순화하고, 인간 주석을 안내하는 명확한 난이도 기준을 포함합니다. 또한, 이 데이터셋은 VisualSimpleQA-hard라는 도전적인 서브셋을 포함하여, 최첨단 모델들의 성능 평가에 도움을 줄 수 있습니다.

- **Technical Details**: VisualSimpleQA는 다중 모달 질문과 정답, 그리고 답변의 근거(rationale)와 텍스트 전용 질문을 포함하여 평가를 수행합니다. 이 평가 방식은 언어 모듈과 시각 모듈의 성능을 개별적으로 분석할 수 있게 해 주며, 검증된 난이도 기준은 각 샘플의 도전 수준을 정량화합니다. 또한, 데이터셋 내의 모든 샘플은 최소 1년의 경험이 있는 인간 주석자들에 의해 제작됩니다.

- **Performance Highlights**: 15개의 LVLM을 대상으로 한 실험 결과, 최신 모델인 GPT-4o도 VisualSimpleQA에서 60% 이상의 정답률을 기록했으며, VisualSimpleQA-hard에서는 30% 이상의 정답률을 보였습니다. 이는 LVLM들이 복잡한 시각 인식 작업을 처리하는 데 한계가 있음을 나타내며, 향후 개선의 여지가 크다는 점을 강조합니다. 이 연구는 LVLM의 사실 탐색 QA 능력 개선을 위한 기초 자료로 활용될 수 있습니다.



### Think Twice, Click Once: Enhancing GUI Grounding via Fast and Slow Systems (https://arxiv.org/abs/2503.06470)
- **What's New**: 이번 논문에서는 Focus라는 새로운 GUI grounding 프레임워크를 소개합니다. 이 프레임워크는 속도와 분석을 결합하여 작업의 복잡성에 따라 빠른 예측과 체계적인 분석을 동적으로 전환합니다. 이러한 접근 방식은 기존의 빠른 예측 방법이 가지던 복잡한 인터페이스 이해의 한계를 극복하고자 하며, 이는 인간의 두 가지 사고 시스템에서 영감을 받았습니다.

- **Technical Details**: Focus 프레임워크는 GUI grounding을 세 가지 단계로 세분화하여 각 단계에서 인터페이스 요약, 집중 분석 및 정밀 좌표 예측을 수행합니다. 이 프로세스는 복잡한 인터페이스와 시각적 관계를 체계적으로 이해하는 데 도움을 줍니다. Focus는 300K의 훈련 데이터를 사용하여 2B 파라미터 모델을 통해 기초적인 성능을 지속적으로 개선해왔습니다.

- **Performance Highlights**: Focus는 특히 복잡한 GUI 시나리오에서 탁월한 성능을 보여줍니다. ScreenSpot 데이터세트에서 평균 77.4%의 정확도를 기록했으며, 더 어려운 ScreenSpot-Pro에서 13.3% 개선된 결과를 보였습니다. 이번 결과는 Focus의 이중 시스템 접근 방식이 복잡한 GUI 상호작용 시나리오를 개선하는 데 잠재력이 있음을 보여줍니다.



### Pre-Training Meta-Rule Selection Policy for Visual Generative Abductive Learning (https://arxiv.org/abs/2503.06427)
Comments:
          Published as a conference paper at IJCLR'24

- **What's New**: 이 논문은 시각 생성 유도 학습 (visual generative abductive learning) 분야의 최신 연구를 다루고 있습니다. 메타-규칙 선택 정책 (meta-rule selection policy)을 학습하기 위한 사전 훈련 전략을 제안하여, 시각 생성 과정이 유도된 논리 규칙에 의해 안내될 수 있도록 합니다. 이 과정은 시간 비용을 줄이는 데 기여하며, 이는 특히 대규모의 논리 기호 집합과 복잡한 논리 규칙이 있을 때 중요합니다.

- **Technical Details**: 제안된 사전 훈련 방법은 기호 기초 (symbol grounding) 학습 없이 순수 기호 데이터에서 수행됩니다. 이는 전체 학습 과정이 낮은 비용으로 진행된다는 장점을 제공합니다. 또한 선택 모델은 사례의 기호 기초와 메타-규칙의 임베딩 표현을 기반으로 구축되어, 신경 모델과 논리 추론 시스템에 효과적으로 통합될 수 있습니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안된 방법은 시각 생성 유도 학습에서 메타-규칙 선택 문제를 효과적으로 해결하여, 유도 과정의 효율성을 크게 향상시킬 수 있음을 보여줍니다. 주목할 점은 사전 훈련 중에 보지 못한 기호 기초 오류를 수정할 수 있는 선택 정책의 강력한 관용성을 관찰할 수 있었다는 것입니다. 이는 주의 메커니즘의 기억 능력과 기호 패턴의 상대적 안정성을 증명하는 결과입니다.



### Federated Learning for Diffusion Models (https://arxiv.org/abs/2503.06426)
- **What's New**: 본 논문에서는 비독립적이고 비동질적(non-IID) 데이터를 다룰 때의 어려움을 해결하기 위해 FedDDPM- 이상적인 Denoising Diffusion Probabilistic Model을 사용하는 연합(FL) 학습을 제안합니다. 이 방법은 각 클라이언트가 자신의 데이터를 사용해 훈련한 로컬 모델을 서버에게 업로드한 뒤, 이를 통해 글로벌 데이터 분포를 근사적으로 나타내는 보조 데이터를 생성합니다. FedDDPM은 이러한 기법을 통해 학습의 품질을 향상시키고, 비효율적인 학습 과정을 해결할 새로운 알고리즘인 FedDDPM+도 소개합니다.

- **Technical Details**: FedDDPM은 로컬 클라이언트에서 훈련한 디퓨전 모델을 활용하여 서버가 집계된 모델을 개선할 수 있도록 하는 알고리즘입니다. 또한, 이 모델의 장점을 최대한 활용하여 비독립적이고 비동질적인(non-IID) 데이터에서 발생할 수 있는 편향을 교정하기 위해 보조 데이터셋을 사용하여 서버에서 추가적인 최적화를 수행합니다. FedDDPM+는 상대적으로 느린 학습을 감지하면 보조 데이터셋을 활용하여 원샷(one-shot) 수정 방법을 적용하여 학습 효율성을 높입니다.

- **Performance Highlights**: 실험 결과, FedDDPM과 FedDDPM+는 MNIST, CIFAR10 및 CIFAR100 데이터셋에서 기존의 최첨단 FL 알고리즘보다 우수한 성능을 보였습니다. 특히, FedDDPM+는 FL 효율성을 저하시키지 않으면서도 성능을 크게 향상시켰습니다. 이러한 결과는 비독립적이고 비동질적인 데이터에서 디퓨전 모델의 성능 향상을 위한 많은 가능성을 제시합니다.



### A Good Start Matters: Enhancing Continual Learning with Data-Driven Weight Initialization (https://arxiv.org/abs/2503.06385)
Comments:
          Preprint

- **What's New**: 최근 현실 데이터 스트림에 적응하기 위해 지속적인 학습(Continual Learning, CL) 시스템에서는 새로운 개념을 신속하게 배우는 동시에 이전 지식을 보존해야 합니다. 본 논문에서는 Neural Collapse(NC)에서 영감을 얻어 CL에서 학습 효율성을 개선하기 위한 가중치 초기화 전략을 제안합니다. 새로운 범주에 대한 분류기 가중치를 데이터 기반(data-driven) 방식으로 초기화함으로써 초기 손실 스파이크를 완화하고 새로운 작업에 대한 적응 속도를 높입니다.

- **Technical Details**: 본 연구는 새로운 클래스의 가중치를 무작위로 초기화하는 기존의 표준 관행이 초기 훈련 손실을 높이고 성능 저하를 초래한다는 점을 강조합니다. 우리는 Feature 통계에 기반한 데이터 구동 가중치 초기화 전략을 도입하여 이러한 문제를 해결하고, 최소 제곱(Least-Squares, LS) 기반 접근 방식을 통해 분류기 가중치를 최적으로 설정하는 방법을 모색합니다. 이러한 LS 기반 초기화는 추가 훈련이나 하이퍼파라미터 없이 마지막 층의 특성에서 단독으로 계산될 수 있습니다.

- **Performance Highlights**: 제안된 기법은 대규모 CL 환경에서 평가되었으며, 이를 통해 훈련 손실 스파이크를 줄이고 효율적 적응을 증진할 수 있음을 보였습니다. 특히 다양한 CL 조건에서도 데이터 기반 초기화가 새로운 개념 학습에 있어 성능을 크게 향상시킬 수 있다는 것을 보여주었습니다. 연구 결과를 바탕으로 CL 성능을 최적화하기 위한 새로운 연구 방향성을 제시합니다.



### X-LRM: X-ray Large Reconstruction Model for Extremely Sparse-View Computed Tomography Recovery in One Second (https://arxiv.org/abs/2503.06382)
Comments:
          A large reconstruction model and the largest dataset (16K samples) for sparse-view CT recovery

- **What's New**: 이번 연구에서는 극도로 희소한 3D CT 재구성을 위한 X-ray Large Reconstruction Model (X-LRM)을 제안합니다. 특히, X-LRM은 10개 미만의 X-ray 투영을 입력으로 사용하여 높은 품질의 재구성을 가능하게 합니다. 기존 CNN 기반 아키텍처의 한계를 극복하기 위해 MLP 기반의 이미지 토크나이저와 Transformer 기반의 인코더를 도입하였습니다. 이를 통해 다양한 입력 수를 유연하게 처리할 수 있습니다.

- **Technical Details**: X-LRM은 X-former와 X-triplane의 두 가지 주요 구성 요소로 구성됩니다. X-former는 MLP 기반 image tokenizer를 사용해 입력 이미지에서 패치 토큰으로 분할하며, 이후 Transformer 인코더를 통해 self-attention을 계산합니다. X-triplane은 3D 방사 밀도를 묘사하는 새로운 3D 표현을 제공하며, 대규모 학습을 지원하기 위해 Torso-16K라는 데이터셋을 구축했습니다. 이 데이터셋은 16K 이상의 볼륨-투영 쌍을 포함하고 있어 X-LRM의 훈련 성능을 극대화할 수 있습니다.

- **Performance Highlights**: X-LRM은 PSNR 지표에서 기존 최첨단 방법보다 1.5 dB 향상된 성능을 보이며, 27배 더 빠른 추론 속도를 달성하였습니다. 또한, 폐 분할(task) 평가에서 제안하는 방법의 실제 가치도 입증되었습니다. X-LRM의 코드, 미리 훈련된 모델 및 데이터셋은 추후 공개될 예정입니다.



### Learning to Unlearn while Retaining: Combating Gradient Conflicts in Machine Unlearning (https://arxiv.org/abs/2503.06339)
- **What's New**: 본 논문에서는 Machine Unlearning(MU) 문제를 다루며, 특정 데이터에 대한 지식을 효과적으로 제거하는 동시에 모델의 성능을 유지하는 새로운 접근 방식인 Learning to Unlearn while Retaining(LUR)을 제안합니다. LUR는 unlearning과 retention 목표 간의 경량화된 충돌을 피하기 위한 기법으로, 모델의 성능 향상과 unlearning의 효과를 동시에 추구합니다. 이 접근법은 gradient의 충돌을 최소화하여, 더욱 효과적인 unlearning을 가능하게 합니다.

- **Technical Details**: LUR는 forget loss를 기반으로 파라미터를 업데이트하면서 retain set의 성능을 고려합니다. 이를 통해, retain loss가 forget loss에 대한 파라미터 업데이트에 어떻게 반응하는지를 분석하여 충돌이 최소화되는 방향으로 파라미터를 조정합니다. 우리의 분석은 LUR가 retain과 forget loss의 내적을 극대화하는 기제가 있음을 밝혀내, 최적의 조건에서 gradient 방향을 조정하도록 합니다.

- **Performance Highlights**: LUR는 다양한 작업에서 검증되었으며, 분류 및 생성 모델 과제 모두에서 우수한 unlearning 효과와 모델 성능 유지를 보여줍니다. 기존의 방법에 비해 성능 격차가 적고, unlearning 효율성이 향상됨을 증명하며, 이러한 성과는 LUR의 gradient 충돌 최소화 원리에 기인합니다. 우리는 이러한 개선된 성능이 실제 응용에서 중요한 의미를 가진다고 강조합니다.



### Enhanced Pediatric Dental Segmentation Using a Custom SegUNet with VGG19 Backbone on Panoramic Radiographs (https://arxiv.org/abs/2503.06321)
- **What's New**: 이번 연구에서는 소아 치과 세분화를 위한 맞춤형 SegUNet 모델이 소개되었으며, VGG19 백본을 활용하여 개발되었습니다. 어린이의 치과 파노라마 방사선 사진 데이터셋에 처음으로 이 모델이 적용되어 최첨단 성능을 기록했습니다. SegUNet 아키텍처는 세밀한 세분화를 향상시키는 데 효과적인 성능을 보였습니다.

- **Technical Details**: 이 연구는 13층 인코더-디코더 모델로 구성된 SegUNet 아키텍처를 기반으로 합니다. VGG19 백본은 특징 추출 성능을 크게 향상시켜, 소아 치과 데이터의 복잡한 구조를 다루는 데 용이성을 제공합니다. 데이터 전처리 과정에서는 이미지를 256x256 픽셀로 조정하고 Numpy 배열로 변환하여 정규화를 수행했습니다.

- **Performance Highlights**: 모델은 97.53%의 정확도와 92.49%의 다이스 계수, 91.46%의 IOU를 달성하였으며, 이는 소아 치과 세분화에 대한 새로운 기준을 설정했습니다. 정밀도, 재현율 및 특이성 등 다양한 메트릭을 통해 이 방법의 견고성을 나타내며, 다양한 치과 구조에 대한 일반화 능력도 입증되어 클리닉에서의 검진 도구로서의 유용성이 강조되었습니다.



### Integrating Chain-of-Thought for Multimodal Alignment: A Study on 3D Vision-Language Learning (https://arxiv.org/abs/2503.06232)
- **What's New**: 본 연구에서는 Chain-of-Thought (CoT) 추론을 3D 비전-언어 학습에 통합하여 구조화된 추론을 조정 훈련 과정에 포함시키는 새로운 접근 방식을 제안합니다. 특히, 3D-CoT 벤치마크라는 데이터셋을 만들어 형상 인식, 기능 유추, 인과 추론에 대한 계층적 CoT 주석을 포함하였습니다. 이러한 연구는 CoT가 다중 모달 추론을 크게 향상시킬 수 있음을 시사합니다.

- **Technical Details**: 3D-CoT 벤치마크는 기존의 3D 데이터셋에 계층적 추론 주석을 추가하여 3D 비전-언어 정렬에 대한 새로운 기준을 제공합니다. 본 연구에서는 큰 추론 모델(LRMs)과 일반-purpose 언어 모델(LLMs) 간의 성능 차이를 분석하기 위해 표준 텍스트 주석과 CoT-구조화 주석을 비교하는 제어 실험을 수행했습니다. 이 평가 방법론은 intermediate reasoning 품질과 최종 추론의 정확성을 개별적으로 측정합니다.

- **Performance Highlights**: 실험 결과, CoT-구조화 주석이 3D 추론을 상당히 개선하는 것으로 나타났습니다. CoT를 통해 훈련된 모델은 텍스트 설명과 3D 구조 간의 정렬이 향상되었으며, 특히 객체의 affordance 인식과 상호작용 예측에서 두드러진 성과를 보였습니다. LRM은 CoT를 더 효과적으로 활용하여 구조화된 추론에서 보다 큰 이점을 얻는 것으로 나타났습니다.



### Attention on the Wires (AttWire): A Foundation Model for Detecting Devices and Catheters in X-ray Fluoroscopic Images (https://arxiv.org/abs/2503.06190)
- **What's New**: 이 논문은 최소 침습 심혈관 시술에서 사용되는 중재 장치 및 도관의 위치를 X선 형광 영상에서 식별하기 위한 새로운 주의 메커니즘(attention mechanism)을 제안합니다. 이 주의 메커니즘은 다중 스케일 가우시안 도함수 필터와 점곱 기반(attention layer) 주의 레이어를 포함하여, 심혈관 시술에 사용되는 장치에서 전선(wire)을 효과적으로 인식할 수 있도록 합니다.

- **Technical Details**: 제안된 모델은 컨볼루션 신경망(convolutional neural network, CNN)을 기반으로 하며, 12,438개의 X선 이미지를 이용하여 훈련되고 검증되었습니다. 모델은 객체를 동시에 탐지(detect)하고 높은 정밀도를 유지하여 실시간 속도(real-time speed)로 작동할 수 있도록 설계되었습니다. 성능은 Intersection-over-Union (IoU) 지표를 통해 평가되었습니다.

- **Performance Highlights**: 모델은 에코 프로브(echo probe) 탐지에서 0.88의 정확도를, 인공 판막(artificial valve) 탐지에서는 0.87의 정확도를 기록했습니다. 10개의 전극 도관(detecting a 10-electrode catheter)의 탐지 성공률은 99.8%, 절제 도관(ablation catheter)의 탐지 성공률은 97.8%에 달했습니다. 이러한 높은 성능 덕분에 의료 응용 및 로봇 보조 수술에 적합한 다양한 임상 환경에 활용될 수 있습니다.



### Object-Centric World Model for Language-Guided Manipulation (https://arxiv.org/abs/2503.06170)
- **What's New**: 이 연구에서는 언어 지침에 의해 안내되는 개체 중심의 월드 모델을 처음으로 제안합니다. 제안된 모델은 현재 상태를 개체 중심의 표현으로 인식하고, 자연어 지침을 조건으로 미래 상태를 예측합니다. 기존의 확산 기반 생성 모델보다 더 효율적이며 샘플 및 계산 효율성에서 우수한 성능을 보여줍니다.

- **Technical Details**: 제안된 모델은 슬롯 어텐션(Slot Attention) 기반 인코더를 사용하여 개체 중심의 표현을 얻고, 그 표현을 바탕으로 즉각적인 행동 예측을 수행합니다. 이 방법은 자동 인코딩 구조를 활용하여 무감독(un-supervised) 방식을 통해 프레임을 재구성하고, 이를 통해 제어 작업에서 효율성과 성능 향상을 도모합니다. 또한, 모델은 언어 지침에 의해 유연한 예측이 가능하여 객체 인식이 중요한 조작 작업에서 유리합니다.

- **Performance Highlights**: 모델은 비주얼-언어-모터 제어 과제에서 기존의 최첨단 모델보다 우수한 샘플 및 계산 효율성으로 성과를 달성했습니다. 제안된 방법은 보지 못한 과제 설정에 대한 일반화 성능도 탐구하고 있으며, 객체 중심 표현을 사용해 행동 예측을 위한 다양한 방법을 연구했습니다. 최종적으로, 제안된 모델은 언어 기반 지침을 통해 향상된 조작 작업의 수행 능력을 보여줍니다.



### VACT: A Video Automatic Causal Testing System and a Benchmark (https://arxiv.org/abs/2503.06163)
- **What's New**: 이번 논문에서는 텍스트에 기반한 Video Generation Models (VGMs)의 발전이 실제 세계 수준의 비디오 생성의 접근성과 비용 효율성을 높이는 데 기여하고 있음을 강조합니다. 그러나 생성된 비디오의 정확성이 떨어지고 기본 물리 법칙에 대한 이해가 부족하다는 문제를 지적합니다. 이를 해결하기 위해 VACT라는 새로운 자동화된 프레임워크를 제안하여 VGMs의 인과적 이해(causal understanding)를 평가하는 시스템을 개발했습니다.

- **Technical Details**: VACT는 인과 분석(causal analysis) 기법과 대형 언어 모델(large language model) 보조 도구를 결합하여 다양한 시나리오에서 VGMs의 인과적 행동을 인간 주석 없이 평가할 수 있도록 설계되었습니다. 이 프레임워크는 다양한 맥락에서 모델의 인과적인 측면을 측정하고, 다층 인과 평가 메트릭(multi-level causal evaluation metrics)을 통해 VGMs의 인과적 성능(performance)에 대한 세부 분석을 제공합니다.

- **Performance Highlights**: 제안된 프레임워크를 통해 여러 기존 VGMs을 벤치마킹하여 그들의 인과적 추론(capabilities of causal reasoning) 능력에 대한 통찰을 제공합니다. 이러한 연구는 VGMs의 신뢰성을 높이고, 실제 응용 가능성을 개선하는 데 기여할 기반을 마련합니다.



### RGB-Phase Speckle: Cross-Scene Stereo 3D Reconstruction via Wrapped Pre-Normalization (https://arxiv.org/abs/2503.06125)
Comments:
          Submitted to ICCV 2025

- **What's New**: 이 연구에서는 Active Stereo 카메라 시스템을 기반으로 한 RGB-Speckle이라는 새로운 3D reconstruction 프레임워크를 소개합니다. 이는 phase pre-normalization encoding-decoding 기법을 도입하여 환경 간섭을 완화함으로써, cross-domain 3D reconstruction의 안정성을 크게 향상시킵니다. 이 방법은 phase shift 맵을 랜덤으로 변형하고 RGB 채널에 통합하여 색깔 스펙클 패턴을 생성하는 독창적인 접근 방식을 포함합니다.

- **Technical Details**: 제안된 RGB-Speckle 접근 방식은 복잡한 시나리오에서의 3D reconstruction을 위해 새로운 phase pre-normalization 방법론을 적용합니다. 이 기법은 입력 이미지 쌍의 분야 간 일관성을 보장하면서, 다양한 환경 조명과 텍스처 간섭의 영향을 줄입니다. 또한, sub-pixel 정밀도의 대규모 RGB 스펙클 데이터 셋을 구축하여 실제적이고 도전적인 장면을 촬영하여 성능을 평가합니다.

- **Performance Highlights**: 실험 결과는 RGB-Speckle 모델이 cross-domain 및 cross-scene 3D reconstruction 작업에서 상당한 이점을 제공함을 보여줍니다. 제안된 방법은 복잡한 환경에서도 모델의 일반화 능력을 강화하여 더 나은 재구성을 가능하게 합니다. 공적인 데이터 셋과 스펙클 데이터 셋에서 수행된 정량적 및 정성적 실험은 이 접근법의 견고성과 일반성을 입증합니다.



### Pathology-Guided AI System for Accurate Segmentation and Diagnosis of Cervical Spondylosis (https://arxiv.org/abs/2503.06114)
- **What's New**: 이번 연구에서는 AI-assisted Expert-based Diagnosis System을 제안하여 경추( cervical ) 추간판 탈출증과 관련한 진단의 정밀도를 향상시키고자 합니다. 이 시스템은 MRI 이미지를 기반으로 자동으로 병변을 분할하고 진단을 수행하도록 설계되었습니다. 특히, 960개의 경추 MRI 이미지 데이터셋을 활용하여 병리학적 프레임워크를 통합한 자동화 진단 접근법을 개발했습니다.

- **Technical Details**: 연구팀은 nnUNet 기반의 병리 유도 분할 모델을 사용하여 경추 해부학 구조를 정확하게 분할할 수 있는 프레임워크를 설계했습니다. 이 시스템은 병리학적 히트맵을 생성하여 분할의 정확도를 높이며, 진단 프로세스에서 전문가 수준의 정확성을 목표로 합니다. 시스템은 5개의 주요 임상 지표를 기반으로 포괄적인 진단 프레임워크를 구축하며, T2 하이퍼인텐시티 검출을 위한 새로운 매개변수를 추가했습니다.

- **Performance Highlights**: 시스템의 분할 모델은 경추 해부학의 평균 다이스 계수가 0.90을 초과하는 인상적인 성과를 보였으며, 진단 평가에서는 C2-C7 Cobb 각도의 평균 절대 오차가 2.44도, 최대 척수 압박(coefficient)에서는 3.60%로 나타났습니다. 또한, 경추의 탈출 부위나 K선 상태 평가, T2 하이퍼인텐시티 검출 등에서 높은 정확도와 재현율, F1 점수를 기록하며 기존 방법들보다 뛰어난 성능을 입증했습니다.



### GEM: Empowering MLLM for Grounded ECG Understanding with Time Series and Images (https://arxiv.org/abs/2503.06073)
- **What's New**: 최근 다중 모달 대형 언어 모델(MLLMs)은 자동화된 ECG 해석에서 발전을 이루었지만, 여전히 두 가지 주요 한계에 직면해 있습니다. 첫째, 시계열 신호와 시각적 ECG 표현 간의 결합이 불충분하며, 둘째, 진단을 미세한 파형 증거에 연결하는 데 한계가 있습니다. 이를 해결하기 위해 GEM을 도입하여 시계열 데이터, 12리드 ECG 이미지 및 텍스트를 통합한 최초의 MLLM을 제안합니다.

- **Technical Details**: GEM은 이중 인코더 프레임워크를 사용하여 시계열과 이미지 특성을 보완적으로 추출하고, 크로스모달 정렬(Cross-modal Alignment)을 통해 효과적인 다중 모달 이해를 가능하게 합니다. 또한, 지식 기반의 지침 생성을 통해 고해상도의 그라운딩 데이터를 생성하여 진단을 측정 가능한 매개변수와 연계합니다. 이러한 구조는 GEM이 임상과 유사한 진단 과정을 시뮬레이션하도록 돕습니다.

- **Performance Highlights**: GEM은 기존 및 제안된 벤치마크에서 실험적으로 예측 성능을 7.4% 향상시키고, 설명 가능성은 22.7% 증가시키며, 그라운딩 성능을 24.8% 개선하였습니다. 이러한 성과는 GEM이 실제 임상 환경에서 더 적합하고 믿을 수 있는 진단 도구로 자리 잡을 수 있도록 합니다.



### STAR: A Foundation Model-driven Framework for Robust Task Planning and Failure Recovery in Robotic Systems (https://arxiv.org/abs/2503.06060)
- **What's New**: 최근 로봇 시스템은 산업 자동화에서 가정용 보조, 우주 탐사와 같은 다양한 동적 환경에서 자율적으로 작동하도록 요구받고 있습니다. 그러나 기존의 로봇 설계는 경직된 규칙 기반 프로그래밍에 의존해 이러한 예측 불가능한 작업을 처리하는 데 한계를 보이고 있습니다. 이를 개선하기 위해, 우리는 SMART (Smart Task Adaptation and Recovery)라는 새로운 프레임워크를 제안하며, 이는 Foundation Models (FMs)과 동적으로 확장되는 Knowledge Graphs (KGs)를 통합하여 효율적인 작업 계획과 자율적인 실패 회복을 가능하게 합니다.

- **Technical Details**: STAR는 자연어 명령어를 실행 가능한 계획으로 변환하고, 실행 중에 실시간 센서 데이터를 기반으로 장애의 원인을 진단합니다. 이 과정에서, KG는 역사적인 실패 패턴과 환경 제약을 포함한 지식을 저장하며, 이를 통해 상황에 최적화된 회복 전략을 생성합니다. STAR는 기존의 고정된 복구 프로토콜에 의존하지 않고, 새로운 실패 상황에 동적으로 적응하여 작업의 효율성과 신뢰성을 개선합니다.

- **Performance Highlights**: STAR는 포괄적인 데이터 세트를 활용한 실험을 통해 86%의 작업 계획 정확도와 78%의 회복 성공률을 달성했습니다. 이는 기존 방법들에 비해 상당한 개선을 보여주며, STAR는 구조화된 지식 표현을 유지하면서 경험으로부터 지속적으로 학습할 수 있는 능력이 특히 중요한 장기 배치에 적합하다는 것을 강조합니다.



### Zero-Shot Peg Insertion: Identifying Mating Holes and Estimating SE(2) Poses with Vision-Language Models (https://arxiv.org/abs/2503.06026)
Comments:
          Under submission

- **What's New**: 이 논문에서는 Vision-Language Models (VLMs)를 활용하여 제로샷(Zero-shot) 페그 삽입을 위한 새로운 프레임워크를 제안합니다. 이 방법은 사전 지식 없이 호환 가능한 홀을 식별하고 자세를 추정할 수 있는 인식 시스템을 구축합니다. 실험 결과, 이 시스템은 90.2%의 정확도를 달성하며, 다양한 이전에 보지 못한 페그-홀 쌍에서도 높은 성능을 보였습니다.

- **Technical Details**: 제로샷 페그 삽입을 위한 이 프레임워크는 비전-언어 모델을 활용하여 페그와 홀의 이미지를 입력받고, 이들의 호환성을 판단합니다. 그런 다음, 모델은 여러 후보 홀 중 최적의 결합 홀을 선택하고, 삽입 각도를 바탕으로 상대 자세를 추정합니다. 이를 통해, 고전적인 방법들에 비해 작업 전용 훈련 없이 다양한 산업용 연결 장치와 복잡한 형상의 물체에 잘 적용됩니다.

- **Performance Highlights**: 본 연구에서는 공백(panel) 상에 있는 산업용 커넥터를 실제로 삽입하는 과정을 평가하였고, 88.3%의 성공률을 기록했습니다. 또, 점검 연구를 통해 입력 및 출력의 변형을 체계적으로 개선하여 이 방법의 유효성을 입증하였습니다. 이러한 결과는 VLM 기반 제로샷 추론이 로봇 조립에서 강인하고 일반화 가능한 성능을 보여준다는 것을 강조합니다.



### GenieBlue: Integrating both Linguistic and Multimodal Capabilities for Large Language Models on Mobile Devices (https://arxiv.org/abs/2503.06019)
Comments:
          14 pages

- **What's New**: 최근 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 발전은 모바일 장치에서의 배치를 가능하게 했습니다. 그러나 언어 능력 유지와 하드웨어 호환성이라는 문제는 여전히 남아 있습니다. 본 연구에서는 GenieBlue라는 구조적 설계를 제안하여, 언어 능력과 멀티모달 능력을 통합하여 모바일 디바이스에서 효율적으로 작동할 수 있도록 합니다.

- **Technical Details**: GenieBlue는 MLLM 훈련 중 원래 대형 언어 모델(LLM) 파라미터를 동결하여 언어 능력을 유지합니다. 특정 transformer 블록을 복제하고 경량 LoRA(Low-Rank Adaptation) 모듈을 추가하여, 멀티모달 능력을 확보합니다. 이러한 접근 방식은 효율적인 고유 언어 능력 보존을 통해, 높은 훈련량으로도 경쟁력 있는 멀티모달 성능을 제공합니다.

- **Performance Highlights**: GenieBlue는 실제 스마트폰의 NPU에서 배치되며, 모바일 디바이스에 적합한 효율성과 실용성을 입증합니다. 다양한 MLLM에서 나타나는 성능 저하 문제를 분석하고, GenieBlue에 의해 직접적인 언어 작업 성능 감소를 예방합니다. 이 연구는 모델 구조 설계와 훈련 데이터에 대한 분석을 통해, MLLMs의 순수 언어 성능 유지 방법을 구체적으로 설명합니다.



### GrInAdapt: Scaling Retinal Vessel Structural Map Segmentation Through Grounding, Integrating and Adapting Multi-device, Multi-site, and Multi-modal Fundus Domains (https://arxiv.org/abs/2503.05991)
- **What's New**: 이번 연구에서는 GrInAdapt라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 레이블의 세분화와 모델의 일반화를 향상시키기 위해 다중 시점 영상을 활용하여, 광학 코히어런스 단층촬영 혈관조영법(OCTA)에서의 망막 혈관 분할을 지원합니다. GrInAdapt는 등록을 통한 공통 앵커 공간 정의, 다중 시점의 예측 통합을 통한 레이블 합의 도출, 다양한 목표 영역으로의 모델 적응이라는 직관적인 세 단계 접근 방식을 따릅니다.

- **Technical Details**: GrInAdapt는 기계 학습 기반의 다중 목표 도메인 적응(Multi-Target Domain Adaptation) 기술을 활용합니다. 이 기술은 단일 출처 도메인에 기반한 모델이 여러 라벨이 있는 목표 도메인에서 정확하게 작동하도록 돕습니다. 연구에서는 OCTA 및 색상 망막 사진(CFP)과 같은 다양한 시각적 정보에서 집합적인 추론을 수행하여 강력한 레이블 세분화의 정확성을 달성했습니다.

- **Performance Highlights**: 전문적인 실험 결과, GrInAdapt는 기존 도메인 적응 방법들에 비해 평균 4%의 Dice 점수 향상과 0.42 ASSD 감소를 보여주었습니다. 이 연구는 다수의 장치와 다양한 사이트에서 발생하는 데이터 분포 변화에도 불구하고, 강력한 성능을 유지함을 입증했습니다. GrInAdapt의 결과는 자동화된 망막 혈관 분석 발전에 기여할 가능성을 강조합니다.



### HealthiVert-GAN: A Novel Framework of Pseudo-Healthy Vertebral Image Synthesis for Interpretable Compression Fracture Grading (https://arxiv.org/abs/2503.05990)
- **What's New**: 이 연구에서는 새로운 척추 골절(VCFs) 평가 방법을 제시하며, HealthiVert-GAN이라는 이름의 새로운 모델을 통해 골절이 발생하기 전의 상태를 시뮬레이션하여 피사체를 복원하는 기술을 향상시켰습니다. 이 모델은 주어진 CT 이미지를 기반으로 가짜 건강한 척추 이미지를 생성하는 데 강력한 GAN 기법을 사용하여 골절의 심각도를 측정합니다. 또한, 본 논문은 상대적 높이 손실 지표(RHLV) 및 지지 벡터 머신(SVM)을 통한 분석으로 진단의 정확성을 높이고, 새로운 3D 생성 프레임워크를 제안합니다.

- **Technical Details**: HealthiVert-GAN은 coarse-to-fine 방식의 합성 네트워크를 활용하여 척추의 앞쪽, 중간, 뒷부분을 각각 측정하고, RHLV로 골절의 크기와 특성을 정량화합니다. 이 모델은 인접 건강한 척추의 형태학적 정보를 활용하여 분석의 일관성을 높이고, Self-adaptive Height Restoration Module(SHRM)을 사용하여 생성된 척추의 높이를 동적으로 조정합니다. 이 방법론은 기존의 2차원 X-Ray 또는 단일 CT 슬라이스에 의존하지 않으며, 3D 분석을 통해 골절 형태를 보다 정교하게 평가합니다.

- **Performance Highlights**: 제안된 방법은 Verse2019 데이터셋과 자체 데이터셋에서 최첨단 성능을 달성하며, 척추 높이 손실의 단면 배포 맵도 제공합니다. 이 연구의 접근 방식은 임상 환경에서 진단 민감성을 높이고 수술 결정에 도움이 되는 실용적인 도구로 자리 잡을 수 있습니다. 또한, 기존 방법에 비해 높이 손실의 정량화 및 골절 등급 지정의 정확도를 향상시켜 임상 의사결정을 지원하는 데 기여합니다.



### LapLoss: Laplacian Pyramid-based Multiscale loss for Image Translation (https://arxiv.org/abs/2503.05974)
Comments:
          Accepted at the DeLTa Workshop, ICLR 2025

- **What's New**: 이번 연구에서는 이미지 대 이미지 변환(I2IT)에 대한 새로운 접근 방식인 LapLoss를 제안합니다. LapLoss는 라플라시안 피라미드를 중심으로 한 네트워크를 기반으로 하며, 저조도 및 과도조명 조건에서도 고수준 특성을 캡처할 수 있도록 다중 분별자 아키텍처를 활용합니다. 이 방법은 다양한 스케일에서 손실을 계산하여 이미지를 생성할 때의 재구성 정확성과 지각 품질을 균형 있게 향상시킵니다.

- **Technical Details**: 제안된 방법론은 라플라시안 피라미드를 활용하여 다양한 해상도에서 높은 수준의 특징을 포착하고, 동시에 낮은 수준의 세부 사항과 텍스처를 유지합니다. 다중 스케일에서의 손실 계산은 고급 이미지 생성의 품질을 확보하며, 경량 아키텍처 덕분에 빠른 추론 시간을 자랑합니다. GANs(Generative Adversarial Networks)를 활용하여 생성 네트워크의 출력을 개선하였으며, 경쟁 과정에서 생성기가 더 미세한 변화를 모사하도록 발전합니다.

- **Performance Highlights**: 제안한 LapLoss는 SICE 데이터셋에서 다양한 조명 조건에서 우수한 성능을 발휘하며, 기존의 최첨단 대비 향상 기술을 초월하는 것을 목표로 합니다. 아홉 가지 다른 대비 수준을 대상으로 한 실험에서도 적응성과 일반화 가능성을 보여주었고, 교차 검증을 통해 제안된 손실 함수의 강건성을 입증하였습니다. 이는 저조도 및 과부하 조건 모두에서 경쟁력 있는 성능을 달성하는 데 기여합니다.



### OSCAR: Object Status and Contextual Awareness for Recipes to Support Non-Visual Cooking (https://arxiv.org/abs/2503.05962)
Comments:
          CHI 2025 Late Breaking Work

- **What's New**: OSCAR(레시피를 위한 객체 상태 맥락 인식 시스템)는 시각 장애인이 요리를 보다 효과적으로 할 수 있도록 도와주는 혁신적인 접근 방식을 제안합니다. 이 시스템은 요리 과정에서 객체의 상태를 추적하여 레시피 진행 상황을 기록하고, 상황 인식 기반 피드백을 제공합니다. 또한, OSCAR는 대형 언어 모델(LLM)과 비전-언어 모델(VLM)의 조합을 활용하여 레시피 단계를 조작하고, 요리 진행 로그를 생성합니다.

- **Technical Details**: OSCAR는 레시피 단계를 표준화하고 객체 상태 정보를 추출하여 시각 데이터와 정렬함으로써 요리 과정에서 동적인 변화를 인식합니다. 이 시스템은 OpenCV를 사용하여 비디오에서 유의미한 프레임을 추출하고, VLM을 통해 레시피 단계와 결합합니다. 또한, 타임 인과 모델을 통해 단계별 예측의 일관성을 유지하며, 사용자가 요리 과정을 쉽게 추적할 수 있도록 돕습니다.

- **Performance Highlights**: 본 연구에서는 173개의 유튜브 요리 비디오와 12개의 비시각적 요리 비디오를 활용하여 OSCAR의 기능을 평가했습니다. 결과적으로 객체 상태를 활용하여 다른 VLM에 비해 20% 이상의 성능 향상을 달성했음을 보여줍니다. 또한, 우리는 레시피 단계를 추적하는 데 영향을 미치는 요인들을 식별하고, 평가 기준으로 활용 가능한 비시각적 요리 비디오 데이터셋을 기여하였습니다.



### Beyond H&E: Unlocking Pathological Insights with Polarization via Self-supervised Learning (https://arxiv.org/abs/2503.05933)
- **What's New**: 이번 연구에서는 PolarHE라는 새로운 이중 모드 융합 프레임워크를 제안합니다. 이 프레임워크는 H&E(hematoxylin and eosin) 염색 이미지와 편광 이미지(polarization imaging)를 통합하여 조기 질병 탐지와 조직 특성 평가를 개선합니다. 편광 이미징은 조직의 미세 구조 변화를 포착하는 데 유용하며, 기존 H&E 이미지로는 확보할 수 없는 병리학적 특징을 강조합니다.

- **Technical Details**: PolarHE는 공통 및 모드 특이(feature-specific) 특징을 분리하는 기능 분해(feature decomposition) 전략을 활용합니다. 이는 H&E 및 편광 이미징으로부터의 정보를 보존하여 보다 견고하고 일반화된 표현을 가능하게 합니다. 각 모드에 대해 두 개의 증강된 뷰를 처리하고, 배치 정규화(batch normalization)를 적용하여 데이터셋 전반에 걸쳐 일관성을 유지합니다. 교차 상관관계 행렬을 계산하여 모드 간의 특성을 적절히 분리합니다.

- **Performance Highlights**: PolarHE는 패치 수준의 분류 작업에서 이전 방법들보다 유의미한 성능 향상을 보였습니다. Chaoyang 데이터셋에서 86.70%, MHIST 데이터셋에서 89.06%의 정확도를 달성했습니다. 또한, t-SNE 시각화를 통해 모델이 공유된 특징과 고유 모드 특징을 효과적으로 캡처하고 있다는 것이 확인되었습니다. 이 연구는 편광 이미징의 잠재력을 강조하며 병리학 모델의 해석 가능성과 일반화 가능성을 향상시키는 방향을 제시합니다.



### SAS: Segment Anything Small for Ultrasound -- A Non-Generative Data Augmentation Technique for Robust Deep Learning in Ultrasound Imaging (https://arxiv.org/abs/2503.05916)
Comments:
          25 pages, 8 figures

- **What's New**: 이번 논문에서는 초음파(ultrasound) 이미지에서 작은 해부학적 구조물의 정확한 분할을 위한 효과적인 데이터 보강 기법인 Segment Anything Small(SAS)를 소개합니다. SAS는 조직의 크기와 질감을 동시에 고려하여, 작은 구조물의 분할 성능을 향상시키기 위해 두 가지 변환 전략을 활용합니다. 마지막으로, SAS는 기존 생성 모델과 달리 비생성 방식으로 훈련 데이터를 다양화하여 노이즈를 줄이고, 오차를 최소화합니다.

- **Technical Details**: SAS 방법론은 두 가지 단계로 구성됩니다. 첫 번째 단계에서는 초음파 윈도우 내에서 관심 영역(ROI)을 추출하고, 이를 재조정하여 썸네일을 생성한 뒤 검은 배경에 배치합니다. 두 번째 단계에서는 세그멘테이션 마스크에서 정의된 기관 영역에 노이즈를 주입하여 다양한 질감을 모사합니다. 이를 통해 다양한 크기와 형태의 해부학적 구조에 대한 모델의 적응성과 일반성을 향상시킵니다.

- **Performance Highlights**: SAS에서 미세 조정된 모델은 내부 및 외부 데이터셋에서 실험을 진행한 결과, 최대 0.35의 Dice 점수 향상과 평균적으로는 0.16까지 성능이 개선되었습니다. 또한, SAS는 두 개의 클릭 프롬프트로 바운딩 박스 프롬프트와 유사한 성능을 달성하며, 희소한 데이터와 풍부한 데이터 환경 모두에서 소형 구조물의 분할 성능을 크게 향상시킵니다.



### Generalizable Image Repair for Robust Visual Autonomous Racing (https://arxiv.org/abs/2503.05911)
Comments:
          8 pages, 4 figures, Submitted to 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025)

- **What's New**: 이번 논문에서는 비전 기반 자율 주행의 성능을 향상시키기 위해 실시간 이미지 복구 모듈을 제안합니다. 이 모듈은 센서 노이즈, 악천후 및 조명 변화와 같은 환경 요인으로 인해 발생하는 이미지 손상을 회복하여 제어 성능을 유지하는데 중점을 둡니다. 제안된 방법은 CycleGAN과 pix2pix 같은 생성적 적대 신경망을 활용하여 이미지를 실시간으로 복원하며, 기존의 방법들과는 다른 획기적인 접근 방식을 제공합니다.

- **Technical Details**: 논문에서는 비전 기반 시스템에서 이미지 관찰 공간과 제어 행동 공간을 명시하며, 각각 카메라 이미지와 제어 명령을 포함합니다. 제안된 이미지 복구 모듈은 실시간으로 이미지 손상을 복원하고, 이를 통해 훈련 데이터와 실제 테스트 데이터 간의 분포 차이를 최소화합니다. 이를 위해 제어 중심의 손실 함수(control-focused loss function)를 도입하여, 복원된 이미지가 정확한 제어 출력을 생성할 수 있도록 보장합니다.

- **Performance Highlights**: 제안한 방법은 다양한 시각적 손상이 존재하는 시뮬레이션된 자율 레이싱 환경에서 평가되었습니다. 그 결과, 기존 방법들과 비교해 성능을 크게 향상시켰으며, 이는 배경 변화에 대한 저항성을 높이고 제어기의 신뢰성을 증대시킨 것으로 나타났습니다. 실험 결과는 자율 주행 성능 개선에 있어 새로운 가능성을 제시합니다.



### Encrypted Vector Similarity Computations Using Partially Homomorphic Encryption: Applications and Performance Analysis (https://arxiv.org/abs/2503.05850)
- **What's New**: 본 논문은 부분 동형 암호화(Partially Homomorphic Encryption, PHE)를 활용하여 암호화된 벡터 유사도 검색을 연구하였습니다. 특히 얼굴 인식과 이미지 검색, 추천 시스템, 대형 언어 모델(large language models, LLM) 등 다양한 응용 분야에 적용 가능한 가능성을 보여줍니다. 논문에서 제안하는 방법은 벡터를 사전에 정규화하여 PHE를 사용해 코사인 유사도를 계산할 수 있는 방법을 제시하여, FHE에 비해 실용적인 대안을 제공하고 있습니다.

- **Technical Details**: 부분 동형 암호화(PHE)는 제한적 연산(덧셈 및 스칼라 곱셈)만을 지원하여 계산 비용 및 저장 요구 사항을 크게 줄입니다. 본 연구에서는 LFW(Labeled Faces in the Wild) 데이터셋을 사용하여 DeepFace의 여러 모델에서 얼굴 이미지로부터 벡터 임베딩을 추출하고, 이 임베딩을 LightPHE를 통해 사전 암호화합니다. 이러한 방식은 이미지 캡처와 암호화된 판독계산을 가능하게 하여, 다양한 암호화 알고리즘을 비교하고 평가합니다.

- **Performance Highlights**: 실험 결과는 PHE가 FHE에 비해 훨씬 더 효율적이며, 메모리 제약이 있는 환경과 같은 실제 사용 사례에서 개인정보 보호를 위한 유사도 검색을 가능하다는 것을 시사합니다. 테스트는 80비트 및 112비트 보안 수준에서 수행되었으며, 암호화 및 복호화 시간, 작동 시간, 코사인 유사도 손실 등의 주요 요소가 비교되었습니다. PHE는 FHE보다 계산 집약도가 낮고, 속도가 빠르며 작은 암호문/키를 생산하여 실제 사용에 적합한 기술로 평가받고 있습니다.



### Decadal analysis of sea surface temperature patterns, climatology, and anomalies in temperate coastal waters with Landsat-8 TIRS observations (https://arxiv.org/abs/2503.05843)
Comments:
          Submitted to GIScience & Remote Sensing

- **What's New**: 이 연구는 Landsat-8의 Thermal Infrared Sensor(TIRS)를 활용하여 해수면 온도(SST)를 정밀하게 추출하는 새로운 방법론을 제안하고, 2014년부터 2023년까지의 SST 클리마톨로지(daily SST climatology)를 설정하여 비정상적인 SST 이벤트를 탐지하는 데 기여합니다. 또한, 연구지역에 대한 SST 이미지와 확률 맵을 통해 SST 이상 현상을 분석하였습니다. 이 연구의 결과는 고해상도 SST 데이터를 생성하여 남호주의 해양 생태계에 중요한 영향을 미치는 이상 현상에 대한 이해를 높이는 데 도움을 줄 것입니다.

- **Technical Details**: 연구는 Landsat-8 TIRS 센서를 사용하여 SST를 추출하는 데 있어 높은 품질의 데이터 획득을 목표로 합니다. 이를 위해 고급 방사전달 모델링을 적용하고, 각 위성 이미지 획득 시점의 특정 대기 조건을 활용하여 대기 영향을 정정합니다. 연구는 100m 해상도의 SST 클리마톨로지를 구축하고 이를 통해 SST 이상 현상의 발생 확률을 도출하였습니다.

- **Performance Highlights**: 연구 결과는 위성에서 유도된 SST 데이터가 현장 측정 SST 값과 잘 일치함을 보여주며, 연구 지역 내에서 SST의 계절적 변동성을 나타내는 패턴을 확인하였습니다. 특히, 스펜서 만(Spencer Gulf)과 세인트 빈센트 만(St Vincent Gulf) 근처의 얕은 지역에서 높은 SST 이상확률을 기록했으며, 이상 현상이 따뜻한 달에 더 빈번하게 발생함을 밝혔습니다.



### Enhancing AUTOSAR-Based Firmware Over-the-Air Updates in the Automotive Industry with a Practical Implementation on a Steering System (https://arxiv.org/abs/2503.05839)
Comments:
          Bachelor's thesis

- **What's New**: 자동차 산업은 차량 기능을 관리하기 위해 소프트웨어에 점점 더 의존하게 되며, 효율적이고 안전한 펌웨어 업데이트가 필수적입니다. 전통적인 펌웨어 업데이트 방식은 OBD 포트를 통한 물리적 연결을 요구하여 불편하고 비용이 많이 들며 시간도 많이 소요됩니다. 이 연구에서는 최신 차량에 맞춘 고급 FOTA(Firmware Over-the-Air) 시스템을 설계하고 구현하여, AUTOSAR 아키텍처를 통합하고 델타 업데이트를 활용하여 펌웨어 업데이트 크기를 최소화하는 방안을 제시합니다.

- **Technical Details**: 이 시스템은 UDS 0x27 프로토콜을 통합하여 업데이트 과정에서 인증 및 데이터 무결성을 보장합니다. 전자 제어 유닛(ECU) 간의 통신은 CAN 프로토콜을 사용하여 이루어지며, ESP8266 모듈과 마스터 ECU는 SPI를 통해 데이터를 전송합니다. 시스템의 아키텍처에는 원활한 펌웨어 업데이트를 위한 부트로더, 부트 매니저 및 부트로더 업데이트 구성 요소가 포함되어 있습니다.

- **Performance Highlights**: 시스템의 기능은 점등 LED와 차선 유지 보조 시스템(LKA)을 통해 시연되었으며, 이는 자동차의 중요한 기능을 처리하는 데 있어 시스템의 다양한 활용 가능성을 보여줍니다. 이 프로젝트는 차량 기술의 중요한 발전을 나타내며, 사용자 중심, 효율적이며 안전한 자동차 펌웨어 관리 솔루션을 제공합니다.



### Randomized based restricted kernel machine for hyperspectral image classification (https://arxiv.org/abs/2503.05837)
- **What's New**: 본 논문에서는 랜덤 벡터 기능 링크(RVFL) 네트워크와 제한된 커널 머신(RKM)의 강점을 결합한 새로운 랜덤 기반 제한 커널 머신($R^2KM$) 모델을 제안합니다. $R^2KM$은 가시 변수와 숨겨진 변수를 모두 활용하는 레이어 구조를 도입해 복잡한 데이터 상호작용과 비선형 관계를 효과적으로 캡처합니다. 이를 통해 기존 모델의 한계를 극복하고, 다양한 복잡한 데이터 상황에서 모델 성능을 향상시키는 포괄적인 솔루션을 제공합니다.

- **Technical Details**: $R^2KM$ 모델은 커널 방법을 통해 비선형 관계를 효과적으로 처리할 수 있는 능력을 갖추고 있습니다. 이 모델은 제한된 볼츠만 머신(RBM)과 유사한 기능을 하는 에너지 함수의 개념을 적용하여, 커널 기반 모델의 해석 능력을 향상시킵니다. 또한, Fenchel-Young 불평등에 기반한 공약수 특성 이중성을 도입하여 모델의 유연성과 확장성을 높이고, 복잡한 데이터 분석 작업에 효율적인 솔루션을 제공합니다.

- **Performance Highlights**: 광범위한 실험 결과는 $R^2KM$이 고광학 이미지 데이터셋과 UCI 및 KEEL 리포지토리의 실제 데이터에서 기준 모델들을 초월하는 성능을 보임을 보여줍니다. 이 모델은 분류(classification) 및 회귀(regression) 작업에서 특히 효과적임을 입증하며, 기존 모델보다 강력한 일반화 성능을 보여줍니다. 이 결과는 복잡한 데이터 구조 처리에 있어 $R^2KM$의 우수성을 강조합니다.



### Illuminant and light direction estimation using Wasserstein distance method (https://arxiv.org/abs/2503.05802)
- **What's New**: 이번 연구에서는 이미지 처리에서 조명 추정(illumination estimation) 문제를 해결하기 위한 새로운 방법을 제시합니다. 기존의 RGB histograms와 GIST descriptors와 같은 전통적인 접근법은 복잡한 조명 환경에서 효과적이지 못했습니다. 제안하는 방법은 최적 운송 이론(optimal transport theory)에 기반한 Wasserstein 거리(Wasserstein distance)를 활용하여 이미지를 통해 조명의 방향과 조명원을 추정합니다.

- **Technical Details**: 이 연구는 다양한 실내 장면, 흑백 사진, 야경 이미지를 대상으로 한 실험을 통해 제안된 방법의 효과성을 입증합니다. 특히, 복잡한 조명 환경에서 우세한 조명원을 감지하고 그 방향을 정확히 추정하는 데 성공했다고 보고합니다. 기존의 통계적 방법(statistical methods)보다 더 뛰어난 성능을 보일 뿐만 아니라, 조명 소스의 로컬라이제이션(light source localization), 이미지 품질 평가(image quality assessment), 객체 탐지(object detection) 강화 등 다양한 응용 가능성을 제시합니다.

- **Performance Highlights**: 이 방법은 복잡한 조명 조건에서도 안정적인 결과를 낼 수 있어, 로봇 공학 분야에서 실세계 조명 문제를 해결하는 데 큰 도움이 될 것으로 기대됩니다. 향후 연구에서는 적응형 임계값(adaptive thresholding) 및 그래디언트 분석(gradient analysis)을 통합하여 정확도를 더욱 향상시킬 예정입니다. 이러한 요소들은 실제 환경에서의 조명 문제를 해결하기 위한 확장 가능한 솔루션을 제공할 것으로 보입니다.



### Self is the Best Learner: CT-free Ultra-Low-Dose PET Organ Segmentation via Collaborating Denoising and Segmentation Learning (https://arxiv.org/abs/2503.03786)
Comments:
          8 pages, 5 figures

- **What's New**: 본 연구에서는 저선량 양전자 방출 단층 촬영(PET)에서의 장기 분할을 위한 새로운 파이프라인인 LDOS를 제안합니다. 이 방법은 CT(Computed Tomography) 의존성을 없애고, Masked Autoencoders(MAE)의 아이디어를 활용하여 저선량 PET을 자연적으로 마스킹된 풀 도즈 PET으로 재해석합니다. LDOS는 동시에 분리하여 밀도가 높은 이미지를 조정하는 것을 통해 장기 경계 인식을 향상시킵니다.

- **Technical Details**: LDOS는 간단하면서도 효과적인 구조를 갖추고 있는데, 공유 인코더가 일반화된 특성을 추출하고, 작업 전용 디코더가 독립적으로 출력을 정제합니다. 방법론적으로, LDOS는 픽셀 수준의 마스킹 접근 방식을 이용하여 의료 영상 내에서 중요한 세밀한 정보를 보존합니다. 이를 통해 CT에서 유래된 장기 주석을 덴오이징 프로세스에 통합하여 PET/CT 정렬 문제를 완화합니다.

- **Performance Highlights**: LDOS는 5% 저선량 PET에서 18개의 장기에 대해 평균 Dice 점수 73.11%(18F-FDG) 및 73.97%(68Ga-FAPI)를 기록하며, 기존 최신 기술 대비 우수한 성능을 입증합니다. 이는 간혹 고도화된 기존 방법에 비해 거리 경계를 보다 정교하게 인지할 수 있도록 합니다. 현재 LDOS에 대한 코드는 공개되어 있어, 향후 연구에 활용될 수 있습니다.



### R1-Omni: Explainable Omni-Multimodal Emotion Recognition with Reinforcement Learning (https://arxiv.org/abs/2503.05379)
- **What's New**: 이번 연구에서는 감정 인식(emotion recognition) 분야에서 Omni-multimodal 대형 언어 모델(large language model)에 Reinforcement Learning with Verifiable Reward (RLVR)의 첫 번째 적용을 제시합니다. RLVR을 통해 Omni 모델의 성능을 최적화하여 추론 능력(reasoning capability)과 감정 인식 정확도(emotion recognition accuracy), 일반화 능력(generalization ability)을 크게 향상시켰습니다.

- **Technical Details**: RLVR의 도입은 모델의 전반적인 성능을 개선할 뿐만 아니라, 데이터 분포가 다른(out-of-distribution) 데이터셋에 대한 평가에서도 뛰어난 강건성(robustness)을 보입니다. 이 과정에서 시각(visual) 및 음성(audio) 정보의 기여를 명확하게 분석할 수 있는 향상된 추론 능력이 확보되었습니다.

- **Performance Highlights**: 최적화된 Omni 모델은 감정 인식 프로세스에서 다양한 양식(modalities)의 기여를 분석하는 데 가치 있는 통찰(insights)을 제공합니다. 이 연구는 멀티모달 대형 언어 모델의 최적화에 대한 새로운 방향성을 제시하며, 감정 인식에서의 성능 향상에 기여하고 있습니다.



New uploads on arXiv(cs.AI)

### A Representationalist, Functionalist and Naturalistic Conception of Intelligence as a Foundation for AGI (https://arxiv.org/abs/2503.07600)
- **What's New**: 이번 논문은 인공지능(AGI) 생성을 위한 기초 원리를 분석합니다. 특히 AGI가 어떻게 목표를 달성할 수 있는지를 이해하는 데 중점을 두고 있으며, 인간의 인지적 한계를 넘어서는 접근 방법을 제안합니다. 다양한 추론 방법을 바탕으로 한 세계 모델 개발에 초점을 맞추고, 기존 AI 접근법의 한계를 지적합니다.

- **Technical Details**: 논문에서는 인지적 능력과 그 변별력을 이해하기 위한 두 가지 정의인 과정 지향적(process-oriented) 정의와 결과 지향적(result-oriented) 정의를 제시합니다. Cattell-Horn-Carroll 이론을 통해 인간 지능의 상호작용을 설명하며, AGI를 위한 지능은 새로운 기술을 생성하는 능력으로 정의됩니다. 이 정의에 따르면 AGI는 변동하는 조건에서 목표를 달성할 수 있는 새로운 기술을 창출할 수 있어야 합니다.

- **Performance Highlights**: AGI 개발의 현재 상태는 특정 분야의 제어된 환경에서는 신뢰성을 가지지만, 더 복잡하고 현실적인 작업에서는 실패율이 높습니다. 이로 인해 AGI가 모든 일상적인 작업을 인간처럼 신뢰성 있게 수행할 수 있는 능력이 부족함을 보여줍니다. 따라서 새로운 기술 생성 능력에 초점을 맞춘 인공지능의 정의가 AGI 개발에는 필수적이라는 주장을 합니다.



### Queueing, Predictions, and LLMs: Challenges and Open Problems (https://arxiv.org/abs/2503.07545)
- **What's New**: 본 논문에서는 대기(queueing) 시스템에 머신러닝 예측(prediction)을 적용하여 시스템 성능을 향상시키는 가능성을 다룹니다. 특히, 서비스 시간(service time)을 예측하는 기술이 스케줄링(scheduling) 결정에 어떻게 효과적으로 활용될 수 있는지를 탐구합니다. 최근의 연구들을 검토하며, 예측의 효용성과 대기 성능에 대한 열린 질문(open questions)을 제시하고 있습니다.

- **Technical Details**: 논문은 대기 시스템에서 예측된 서비스 시간을 고려한 스케줄링 문제를 다루고 있으며, 특히 대형 언어 모델(LLM) 시스템에서의 예측 활용에 중점을 둡니다. LLM 시스템에서의 추론(inference) 요청은 복잡한 특성을 가지며, 변수 추론 시간(variable inference times)과 키-값 저장소(key-value store) 메모리 제한으로 인해 동적 메모리 사용(dynamics memory footprints)이 발생합니다. 이로 인해 성능에 다양한 영향을 미치는 재개(preemption) 접근 방식이 있습니다.

- **Performance Highlights**: LLM 시스템에서 예측을 통해 성능을 개선할 수 있는 새로운 스케줄링 도전 과제를 강조합니다. 대기 이론(queueing theory)의 통찰(insights)을 LLM 시스템의 스케줄링에 적용할 수 있는 중요한 기회가 존재함을 논의하며, 새로운 모델과 문제 설정을 제안합니다. 이러한 접근법은 대기 성능을 향상시키는 데 기여할 가능성이 큽니다.



### AI-Enabled Knowledge Sharing for Enhanced Collaboration and Decision-Making in Non-Profit Healthcare Organizations: A Scoping Review Protoco (https://arxiv.org/abs/2503.07540)
Comments:
          14 pages

- **What's New**: 이번 프로토콜은 자원이 제한된 비영리 의료 조직에서 AI 기반 지식 공유에 관한 기존 증거를 체계적으로 정리하는 스코핑 리뷰를 outlines합니다. 특히, USAID의 운영 중단 이후 외부 지원이 감소하는 상황에서 이러한 기술들이 협력 및 의사결정을 어떻게 향상시키는지를 조사하는 것을 목표로 합니다.

- **Technical Details**: 이 연구는 Resource-Based View, Dynamic Capabilities Theory, Absorptive Capacity Theory의 세 가지 이론적 프레임워크를 기반으로 진행됩니다. 또한, PRISMA-ScR 지침에 따라 체계적인 검색 전략, 포함 및 제외 기준, 구조화된 데이터 추출 프로세스를 포함하는 엄격한 방법론적 접근을 detail합니다.

- **Performance Highlights**: 이 스코핑 리뷰는 이론적 통찰과 실증적 증거를 통합하여 문헌에서의 주요 공백을 확인하고 비영리 의료 환경에서 효과적인 AI 솔루션 설계를 위한 정보를 제공합니다. AI가 전략적 자원 및 조직 학습과 민첩성(agility)을 촉진하는 역할을 탐구합니다.



### From Idea to Implementation: Evaluating the Influence of Large Language Models in Software Development -- An Opinion Paper (https://arxiv.org/abs/2503.07450)
Comments:
          The project is partially supported by the DkIT Postgraduate Scholarship, Research Ireland under Grant number 13/RC/2094_2, and Grant number 21/FFP-A/925

- **What's New**: 이 논문은 대규모 언어 모델(LLMs)의 소프트웨어 개발에서의 활용과 혁신적인 잠재력에 대한 전문가의 의견을 분석합니다. 전문가들의 조사 결과, LLMs는 생산성을 향상시키고 코딩 시간을 단축시키는 데 긍정적인 역할을 하고 있지만, 과다 의존의 위험과 윤리적인 고려사항도 지적되었습니다. LLMs의 도입은 특히 코드 생성, 디버깅 및 문서 작성과 같은 업무에서 효율성을 높이는 데 기여할 수 있습니다.

- **Technical Details**: 연구 방법론은 11명의 소프트웨어 개발 전문가들을 대상으로 하는 비구조화된 인터뷰를 통해 진행되었습니다. 전문가들은 코드 생성 및 검토, 자연어 이해(NLU), 품질 보증, 예측 분석 및 개발자 협업과 같은 다양한 측면에서 LLMs의 역할에 대한 견해를 제시했습니다. 이 논문은 또한 LLMs가 소프트웨어 개발 과정에서 어떻게 활용될 수 있는지를 깊이 있게 다루고 있습니다.

- **Performance Highlights**: 전문가들은 LLMs의 사용이 소프트웨어 개발 업무를 크게 경감시킬 수 있으며, 이를 통해 개발자들이 더 창의적인 작업에 집중할 수 있도록 지원한다고 평가했습니다. 그러나, LLMs가 맥락을 완전히 이해하지 못할 수 있는 한계와 그로 인해 발생할 수 있는 오류, 그리고 데이터 기반의 편향을 우려하는 목소리도 있었습니다. 추가로, LLMs의 직업 안정성에 대한 우려와 윤리적 고려사항도 함께 강조되었습니다.



### From Text to Visuals: Using LLMs to Generate Math Diagrams with Vector Graphics (https://arxiv.org/abs/2503.07429)
- **What's New**: 본 논문은 LLM(대규모 언어 모델)을 활용하여 수학 교육에서 시각화의 중요성을 탐구합니다. 특히, 학생들이 문제를 이해하고 해결하는 데 필요한 도형을 자동으로 생성하는 방법에 대해 논의합니다. 기존의 문제 생성 외에도, LLM을 통해 SVG(Scalable Vector Graphics) 형식으로 수학 관련 도형을 제작하여 교육적 지원을 향상하는 새로운 접근 방식을 제시합니다.

- **Technical Details**: 이 연구는 수학 문제에 대한 텍스트 힌트를 바탕으로 SVG 기반 도형을 생성하는 과제를 정의합니다. 각 수학 문제는 명제 및 단계별 힌트로 구성되며, 각 단계는 텍스트 힌트와 관련된 SVG 도형으로 나뉘어집니다. 또한, LLM이 어떻게 SVG 코드 생성을 통해 시각적 정보를 지원할 수 있는지에 대해 구조적인 접근 방식을 사용하고 있습니다.

- **Performance Highlights**: 자동으로 생성된 도형의 품질을 평가하기 위해 VQA(Visual Question Answering) 기반의 평가 설정을 도입하였습니다. 연구를 통해 LLM이 도형을 생성하는 효과를 개선하기 위한 주요 전략을 식별하고, 여러 실험을 통해 파이프라인의 변화가 도형 품질과 정확성에 미치는 영향을 평가하고 있습니다. 이는 문제 해결 능력을 향상시킬 수 있는 중요하고 실용적인 방법으로 교육 현장에서 활용될 수 있습니다.



### Encoding Argumentation Frameworks to Propositional Logic Systems (https://arxiv.org/abs/2503.07351)
Comments:
          31 pages

- **What's New**: 본 연구는 인공지능에서 유용한 툴인 논증 프레임워크(Argumentation Framework, AF)의 인코딩 방법을 일반화하여 다양한 프로포지션 논리 시스템에 AF를 논리 공식으로 인코딩하는 방식을 제안합니다. 특히, Dung의 고전적 의미론과 Gabbay의 방정식 의미론을 포함한 논증 의미론에 따른 AF의 모델 간 관계를 연구합니다. 추가적으로, 이 연구는 명확한 인코딩 함수의 증명 및 3값 프로포지션 논리 시스템과 퍼지 프로포지션 논리 시스템으로의 AF 인코딩을 포함합니다.

- **Technical Details**: AF는 노드로 논증을 나타내고, 방향이 있는 엣지로 공격을 표시하는 주 Directed Graph입니다. 본 논문은 AF와 프로포지션 논리 시스템 간의 관계를 탐구하며, 2값 논리 시스템에 대한 정규 인코딩 함수의 증명을 보완합니다. 또한, Kleene의 3값 프로포지션 논리 시스템과 Łukasiewicz의 시스템으로 AF를 인코딩하여 모델 관계를 설명하는 새로운 방법론을 제공합니다.

- **Performance Highlights**: 연구 성과로, AF를 3값 및 퍼지 프로포지션 논리 시스템으로 인코딩함으로써 새로운 방정식 의미론을 구성할 수 있는 길을 열었습니다. Gabbay의 방정식 접근법과 인코딩 방법의 관계를 탐구하며, 이로 인해 AF와 논리 시스템 간의 연관성이 높아졌습니다. 이러한 새로운 방법들은 논증 이론의 발전에 기여할 것으로 기대되며, 향후 연구에도 적용 가능성이 큽니다.



### Human Machine Co-Adaptation Model and Its Convergence Analysis (https://arxiv.org/abs/2503.07319)
- **What's New**: 본 연구에서는 협력적 적응 마르코프 의사결정 프로세스(Cooperative Adaptive Markov Decision Process, CAMDP) 모델을 바탕으로 로봇 보조 재활에서 인간-기계 인터페이스의 혁신적인 접근법을 제시합니다. 기존의 인터페이스 설계는 기계제어 알고리즘에 주로 초점을 맞추었지만, 본 논문은 환자와 기계의 필요를 동시에 충족시키는 방법을 제안합니다. 이러한 새로운 접근법은 상호작용 학습 과정의 근본적 측면을 다루고 있으며, 이론적 통찰과 실제 지침을 제공합니다.

- **Technical Details**: CAMDP의 수렴을 보장하기 위한 충분 조건을 설정하여, 시스템이 고유한 내시 균형(Nash equilibrium) 지점으로 수렴함을 보장합니다. 본 연구에서는 공동 적응(co-adaptation) 과정에서의 수렴 특성을 분석하고 가치 평가(Value Evaluation) 및 정책 개선(Policy Improvement) 알고리즘을 조정하는 전략을 개발합니다. 이러한 알고리즘은 로봇 보조 재활 설정에서 더욱 효과적인 적응 시스템을 위한 기초를 제공합니다.

- **Performance Highlights**: 제안된 조건과 알고리즘의 효과성을 수치적 실험을 통해 입증하고, 실제 적용 가능성과 견고성을 보여줍니다. 본 연구의 결과는 로봇 보조 재활에서 두 대행자(에이전트), 즉 환자와 로봇 간 최적 정책의 공동 적응을 향상시키는데 기여하며, 환자 결과 개선을 위한 이론적 기반을 강화합니다. 이 연구를 통해 제안된 다수의 내시 균형 지점과 정책 개선 접근법은 실제 환경에서의 효과적인 응용 가능성을 가집니다.



### Automatic Curriculum Design for Zero-Shot Human-AI Coordination (https://arxiv.org/abs/2503.07275)
- **What's New**: 이번 연구에서 우리는 제로 샷 (zero-shot) 인간-AI 조정의 새로운 방법론을 제시합니다. 기존의 멀티 에이전트 환경 설계 기법(UED)을 확장하여 인간-에이전트 간의 조정을 더욱 효과적으로 지원하는 유틸리티 함수 및 동료 플레이어 샘플링을 도입합니다. 이 방법은 복잡한 환경에서도 높은 조정 성능을 달성하여 실제 인간과의 협력에서 우수한 성능을 발휘합니다.

- **Technical Details**: 제로 샷 인간-AI 조정 setting에서는 반응 기반 유틸리티 함수를 사용하여 에고 에이전트와 동료의 협동 능력을 평가합니다. 그 결과, 다양한 시나리오와 동료 플레이어에 대한 조정 전략을 개선할 수 있습니다. 기존의 방법들이 초점 맞추고 있던 최적의 플레이어 환경과의 상호 작용이 아닌 실제 인간과의 협력에 중점을 두고 있습니다.

- **Performance Highlights**: 우리의 접근 방식은 Overcooked-AI 환경에서 평가되었으며, 실제 인간 및 인간 프록시 에이전트를 대상으로 테스트했습니다. 실험 결과, 우리는 기존의 기준 모델들보다 높은 조정 성능을 달성했습니다. 이러한 성과는 제로 샷 설정에서 에고 에이전트가 다양한 환경에 잘 적응할 수 있게 해줍니다.



### A Zero-shot Learning Method Based on Large Language Models for Multi-modal Knowledge Graph Embedding (https://arxiv.org/abs/2503.07202)
- **What's New**: 본 논문에서는 제로샷 학습(zero-shot learning, ZL)을 위한 새로운 프레임워크인 ZSLLM을 제안합니다. ZSLLM은 대형 언어 모델(large language models, LLMs)을 사용하여 다중 모달 지식 그래프(multi-modal knowledge graph, MMKG)의 임베딩 학습을 수행합니다. 이를 통해 보지 않은 카테고리의 시멘틱(sementic) 정보 전송을 보다 효과적으로 처리할 수 있습니다.

- **Technical Details**: ZSLLM은 보지 않은 카테고리에 대한 텍스트 모달리티 정보를 프롬프트(prompt)로 활용하여 LLM의 추론 능력을 극대화합니다. 논문에서는 다중 모달 지식 그래프의 임베딩 표현 학습 과정과 이를 통한 시멘틱 정보 전송 과정을 자세히 설명합니다. 또한, 모델 기반 학습을 통해 MMKG 내에서 보지 않은 카테고리의 임베딩 표현을 강화하는 방법을 설명합니다.

- **Performance Highlights**: 다양한 실제 데이터셋에 대한 광범위한 실험을 통해, 제안한 ZSLLM의 성능이 최신의 방법들보다 우수함을 입증하였습니다. 이 연구는 제로샷 학습이 자연어 처리, 이미지 분류 및 크로스 링구얼 전이 등 다양한 과제에서 중요한 역할을 할 수 있음을 강조합니다. 따라서 본 프레임워크는 개방형 도메인에서의 확장성과 실용성을 크게 향상시킬 것으로 기대됩니다.



### Lawful and Accountable Personal Data Processing with GDPR-based Access and Usage Control in Distributed Systems (https://arxiv.org/abs/2503.07172)
Comments:
          Submitted for review to the Journal of AI and Law, 49 pages (including)

- **What's New**: 이 논문은 GDPR(General Data Protection Regulation) 준수에 따른 개인 데이터 처리의 법적 정당성을 자동으로 추론하는 방법론을 제안합니다. 여러 조직 간 데이터 공유가 증가하는 상황에서, 법적 논거를 수립하기 위해 프라이버시 전문가의 참여를 통한 사례 일반화 방법을 도입합니다. 이를 통해 투명성과 책임성을 높이고 데이터 처리 시스템에 효과적으로 통합될 수 있도록 합니다.

- **Technical Details**: 논문에서는 eFLINT라는 도메인 특정 언어를 통해 법적 규칙을 정의하고, XACML 아키텍처를 활용하여 GDPR 기반의 규범적 추론을 데이터 처리 시스템에 통합하는 방법을 설명합니다. 법적 요구 사항은 사례별로 상이하고, 이는 정책 입안자와 프라이버시 전문가의 지속적인 주의가 필요한 부분입니다. 구조적 복잡성과 GDPR의 다수의 용어가 통합된 이 시스템은 유연성과 투명성을 보장합니다.

- **Performance Highlights**: 제안된 시스템은 인간과 자동화된 시스템 간의 책임 분담을 통해 법적 요구 사항의 준수를 도와줍니다. 정책이 변경될 경우 소프트웨어에 적절하게 반영이 가능하여 비효율성 및 비준수 위험을 단축하는 한편, 의사결정의 책임을 프로세스에 기록하여 보고 체계를 수립합니다. 이는 특히 조직 간 데이터 공유의 맥락에서 협업을 개선하는 데 기여할 것입니다.



### Generative AI in Transportation Planning: A Survey (https://arxiv.org/abs/2503.07158)
Comments:
          56 pages

- **What's New**: 이번 논문은 Generative Artificial Intelligence (GenAI)를 교통 계획에 통합하는 시스템적 프레임워크의 필요성을 강조합니다. 연구팀은 GenAI의 적용을 위한 첫 번째 종합적인 프레임워크를 소개하며, 이를 통해 교통 계획 작업과 컴퓨팅 기법을 두 가지 관점에서 분류하는 새로운 분류 체계를 제시합니다. GenAI는 데이터 준비, 도메인 특화 조정, 사례 생성 등의 측면에서 교통 시스템의 효율성을 높이는 데 기여할 수 있습니다.

- **Technical Details**: 논문에서는 교통 계획의 다양한 작업을 자동화하기 위한 GenAI의 역할을 살펴봅니다. GenAI는 descriptive, predictive, generative, simulation, explainable 작업을 자동화하여 교통 시스템을 개선할 수 있도록 합니다. 또한, retrieval-augmented generation 및 zero-shot learning과 같은 최신 인사이트를 통해 데이터 집합을 준비하고, 교통 데이터에 맞춘 모델을 Fine-tuning하여 실험적 접근 방식을 제시합니다.

- **Performance Highlights**: 논문에 따르면, GenAI는 교통 수요 예측, 트래픽 시뮬레이션 및 정책 모델링 등에서 속도, 정확성 및 범위를 증가시킵니다. 예를 들어, GenAI는 지리적 패턴, 트래픽 카운트 및 환경 지표를 분석하여 미래 인프라 수요를 예측합니다. 또한, GenAI의 도입으로 인해 실시간 트래픽 관리 및 멀티모달 여행 최적화를 통해 교통 계획의 효율성과 포용성을 개선할 수 있습니다.



### Hierarchical Neuro-Symbolic Decision Transformer (https://arxiv.org/abs/2503.07148)
- **What's New**: 이번 연구에서 제안하는 계층적 신경-기호적 제어 프레임워크는 전통적인 기호 계획(symbolic planning)과 트랜스포머 기반의 정책(policy)을 결합하여 복잡하고 장기적인 의사 결정(task) 문제를 해결하는 데 중점을 두고 있습니다. 이 모델은 고수준의 기호 계획자가 논리적 제안을 바탕으로 해석 가능한 연산자 시퀀스를 구성하고, 저수준에서는 각 기호 연산자를 서브 목표(token)로 변환하여 트랜스포머가 불확실하고 고차원적인 환경에서 세밀한 행동 시퀀스를 생성하도록 합니다.

- **Technical Details**: 제안된 계층적 신경-기호적 결정 트랜스포머(decision transformer)는 이산 기호 계획자(symbolic planner)와 결정 트랜스포머(decision transformer) 간의 양방향 인터페이스를 통해 제어 구조를 형성합니다. 이러한 구조는 기호 연산자를 논리적으로 만들고, 이를 바탕으로 신경 정책이 반응형의 세밀한 행동으로 다듬을 수 있도록 합니다. 이 연구는 기호 계획과 신경 실행 계층에서의 근사 오류(approximation errors)가 누적되는 과정을 이론적으로 분석하여 설명합니다.

- **Performance Highlights**: 실험 결과, 제안한 계층적 접근 방식이 순수한 엔드-투-엔드 신경 정책보다 성공률, 샘플 효율성 및 궤적 길이 측면에서 상당히 우수함을 보여주었습니다. 특히 여러 단계와 복잡한 상태 전이 및 논리적 제약을 요구하는 작업에서 두드러진 성과를 나타냈습니다. 이 연구는 다양한 그리드 기반 환경에서의 실험을 통해 제안된 방법의 효과성을 검증하였습니다.



### Correctness Learning: Deductive Verification Guided Learning for Human-AI Collaboration (https://arxiv.org/abs/2503.07096)
- **What's New**: 이 논문에서는 안전-critical 분야에서 AI와 의사결정 기술의 발전에도 불구하고, 의사결정 출력 스킴의 정확성을 검증하고 설계하는 데 여전히 도전 과제가 남아있음을 강조합니다. 새로운 개념인 correctness learning (CL)을 제안하여, 과거의 고품질 스킴과의 협업을 통해 인공지능-인간 협업을 향상시킵니다. 논문은 특히, 'correctness pattern'을 활용하여 의사결정 모델을 개선하고 자원 최적화를 이루는 방법에 대해 설명합니다.

- **Technical Details**: 이 연구는 데이터 분석에 있어 deductive verification 방식을 사용합니다. 저자들은 historical high-quality schemes에서 추출한 행동 패턴을 통해 시스템 에이전트의 적응 행동을 모델링하고 추론하는 pattern-driven correctness learning (PDCL) 방안을 소개합니다. PDCL은 시스템 에이전트의 행동을 형식적으로 규명하여 의사결정 출력 스킴의 정확성을 보장하는 메커니즘을 제공하고, 자원 관리 및 분배 문제를 해결하기 위한 수학적 추론을 활용합니다.

- **Performance Highlights**: 광범위한 실험을 통해 PDCL의 효과가 입증되었습니다. 네 가지 벤치마크 알고리즘의 성능이 평균 8.4%, 3.9%, 1.6%, 및 5.7% 향상되었으며, 각 알고리즘의 효과를 평가하는 데 있어 주요 파라미터의 영향을 분석했습니다. 이는 PDCL이 기존의 스케줄링 방법을 초월하여 더 효율적인 자원 할당을 달성했음을 나타냅니다.



### Rule-Based Conflict-Free Decision Framework in Swarm Confrontation (https://arxiv.org/abs/2503.07077)
- **What's New**: 이 논문에서는 전통적인 rule-based 의사결정 방식의 한계를 극복하기 위해 새로운 의사결정 프레임워크를 제안합니다. 이 프레임워크는 probabilistic finite state machine, deep convolutional networks, 및 reinforcement learning를 통합하여 swarm confrontation에서의 의사결정 효율성을 높입니다. 이것은 특히 jitter (떨림)와 deadlock (교착 상태) 문제를 해결하여 에이전트의 안정적이고 적응력 있는 결정을 가능하게 합니다.

- **Technical Details**: 제안된 프레임워크는 에이전트가 겪는 동적이고 적대적인 환경에서 실시간 결정을 내리고, 이 과정에서 발생하는 상태 간 충돌을 예방합니다. 각 상태에 대한 전이 확률 행렬을 구축하는 대신, 딥러닝 방법을 사용하여 이 행렬을 접근하고, 강화 학습 방법을 통해 에이전트의 결정 정확성을 향상시켜 내부 충돌을 해결합니다. 이를 통해 전통적인 rule-based 방법이 가진 비해 더 나은 설명 가능성과 신뢰성을 제공합니다.

- **Performance Highlights**: 이 연구는 실제 실험에서의 평가를 통해 인간 유사 협동 및 경쟁 전략의 향상된 수행 능력을 나타냅니다. 제안된 접근법은 기존 방법들에 비해 더 효율적으로 swarm confrontation에서의 의사결정을 수행하며, 전반적인 성능 향상을 보여줍니다. 결과적으로, 이 프레임워크는 실시간 의사결정이 필요한 미래의 무인 시스템에 긍정적인 영향을 미칠 것으로 기대됩니다.



### ReAgent: Reversible Multi-Agent Reasoning for Knowledge-Enhanced Multi-Hop QA (https://arxiv.org/abs/2503.06951)
Comments:
          25pages, 3 figures

- **What's New**: ReAgent는 복구 가능한 다중 에이전트 협업 추론 프레임워크로, 명시적인 백트래킹 메커니즘을 통해 다중 통로 질문 응답(multi-hop QA)에 대한 새로운 접근을 제시합니다. 기존의 Chain-of-Thought(CoT) 방법이 오히려 오류를 누적시킬 수 있는 반면, ReAgent는 중간 추론 중에 오류를 감지하고 수정할 수 있는 기능을 제공합니다. 이는 방송(Networking)된 메모리 체계를 활용하여 불일치 상황에서 실시간으로 정보를 조정할 수 있도록 합니다.

- **Technical Details**: 이 프레임워크는 다중 에이전트 시스템을 활용하여 정보 검색, 검증 및 모순 해결과 같은 하위 작업을 전문화된 에이전트가 수행하도록 하고, 이들의 결과를 동기화하여 보다 효과적으로 작업을 수행합니다. 각 에이전트는 자체적으로 내부 모순을 감지하고 자신의 추론을 철회할 수 있는 로컬 수정(local correction) 기능과, 여러 에이전트 간의 불일치를 조정하여 동기화된 롤백(global rollback)을 수행할 수 있는 기능을 갖추고 있습니다. 이러한 비가역적(non-monotonic) 접근 방식은 에이전트가 이전의 유효한 상태로 되돌아가 다양한 대안을 재탐색하고 수정 사항을 앞으로 전파할 수 있게 합니다.

- **Performance Highlights**: 실험 결과, ReAgent는 평균적으로 근본 모델에 비해 약 6%의 성능 향상을 기록하였습니다. 이 시스템은 명확한 백트래킹 흔적을 통해 오류를 고립하고 수정하는 과정을 밝혀내며, 결과적으로 최종 답변의 정확성과 해석 가능성을 향상시킵니다. ReAgent는 비선형적인(non-linear) 추론을 데이터 기반 질문 응답에 연결하여 보다 투명하고 오류 내성이 강한 다중 통로 추론 파이프라인을 제공합니다.



### Enhancing Time Series Forecasting via Logic-Inspired Regularization (https://arxiv.org/abs/2503.06867)
- **What's New**: 최근의 연구들은 Transformer 기반의 시계열 예측(Time Series Forecasting, TSF) 방법들이 모든 토큰 간의 의존성을 동등하게 다루고 있다는 점을 지적합니다. 본 논문에서는 이러한 접근 방식의 한계를 극복하고자 하여, 효과적인 토큰 의존성이 다양한 예측 시나리오에서 어떻게 달라지는지를 탐구합니다. 우리는 이는 예측 성능에 영향을 미치는 중요한 요소라고 밝혔습니다.

- **Technical Details**: 이에 따라, 논문은 논리적 프레임워크를 정립하고, 이와의 정렬을 통해 효과적인 토큰 의존성을 구체적으로 정의합니다. 구체적으로, Atomic Formula와 같은 개념을 활용하여, 토큰 표현이 원자 공식으로 형성될 때의 이점을 강조합니다. 이러한 이해를 토대로, Attention Logic Regularization (Attn-L-Reg) 방법을 제안하여 모델이 더 적지만 효과적인 의존성을 사용하도록 유도합니다.

- **Performance Highlights**: 제안된 Attn-L-Reg는 기존의 Transformer 기반 TSF 방법에 쉽게 적용할 수 있으며, 성능 향상을 가져오는 것으로 입증되었습니다. 실험과 이론적 분석을 통해 이 방법이 모델의 일반화 경계의 타이트함을 개선할 수 있음을 증명하였고, TSF에서 예측 성능을 향상시키는 데 기여할 수 있음을 밝혔습니다.



### Dubito Ergo Sum: Exploring AI Ethics (https://arxiv.org/abs/2503.06788)
Comments:
          10 pages, 1 figure, HICSS 57: Hawaii International Conference on System Sciences, Honolulu, HI, published January 2024

- **What's New**: 본 연구는 AI 윤리(ethics) 분야에서 데카르트의 유명한 격언 "나는 의심한다, 그렇기 때문에 나는 존재한다(I doubt, therefore I am)"를 패러프레이즈(paraphrase)합니다. AI는 스스로 의심할 수 없기 때문에 도덕적 대리(moral agency)를 가질 수 없다는 점을 제안합니다. 이는 인간의 마음과 AI 간의 근본적인 차이를 탐구하는 출발점이 됩니다.

- **Technical Details**: 연구의 주요 초점은 인간의 감각적 기반(sensory grounding)과 이해의 행위(act of understanding) 그리고 스스로 의심할 수 있는 것의 중요성입니다. 윤리의 기초는 인류 역사에서 가장 오래되고 방대한 지식 프로젝트로, 우리(인간)는 아직 이를 완전히 이해하지 못했습니다. 도덕 심리학(moral psychology)의 접근 방식은 도덕적 결정(moral decisions)이 직관적이며, 윤리 모델은 오로지 우리가 스스로를 설명할 때만 관련이 있음을 강조합니다.

- **Performance Highlights**: 본 연구는 AI 윤리에 대한 해결책을 제시하지 않고, 여러 아이디어를 탐구하여 논제의 문제를 열어둡니다. 이는 우리가 AI 윤리를 논의할 때 어떤 관점을 취해야 하는지에 대해 더욱 나은 이해를 제공하는 데 기여할 것으로 기대됩니다. 연구의 과정을 통해 인간의 도덕적 복잡성을 이해하는 데 기여하는 방향으로 나아가고자 합니다.



### Beyond Black-Box Benchmarking: Observability, Analytics, and Optimization of Agentic Systems (https://arxiv.org/abs/2503.06745)
Comments:
          14 pages, 19 figures

- **What's New**: 이 논문은 다양한 작업을 수행하기 위해 협력하는 에이전트 시스템(agentic AI systems)의 분석과 최적화에서의 새로운 도전 과제를 탐구합니다. 전통적인 벤치마킹 방법은 이러한 시스템의 비결정적이고 동적인 특성에 효과적으로 대처하지 못하고 있습니다. 저자들은 에이전트 시스템의 행동을 벤치마킹하는 새로운 방법론을 제안하여 기존의 평가 방법의 한계를 극복하고자 합니다.

- **Technical Details**: 에이전트 시스템은 LLM(대형 언어 모델)을 기반으로 하여 동적인 환경에서 작동하며, 그 결정 과정과 결과는 입력의 미세한 차이에 따라 달라질 수 있습니다. 논문에서는 에이전트 시스템의 행동을 분석하기 위해 여러 가지 세 가지 주요 요소를 정의하고, 표준 관측 가능성 프레임워크를 확장하여 수집 방법을 제안합니다. 또한, 기존 방법론의 주요 제한 사항을 해결하기 위해 새로운 벤치마킹 프레임워크를 도입합니다.

- **Performance Highlights**: 사용자 연구를 통해 비결정적 흐름(non-deterministic flow)이 에이전트 시스템의 주요 도전 과제로 작용한다는 79%의 일치율을 보여주었습니다. 이 논문의 기여는 에이전트 시스템의 행동 벤치마킹의 필요성을 강조하며, 행동 분석 및 성능 분석을 위한 새롭고 체계적인 세분화 체계를 도입합니다. 마지막으로, ABBench(에이전트 분석 행동 벤치마크) 데이터셋을 소개하며 이 데이터셋이 기존 평가 방식을 넘어서는 평가를 가능하게 함을 보여줍니다.



### Agent models: Internalizing Chain-of-Action Generation into Reasoning models (https://arxiv.org/abs/2503.06580)
- **What's New**: 이번 논문은 전통적인 에이전트 워크플로우가 외부 프롬프트에 의존하여 도구 및 환경과의 상호작용을 관리하는 반면, Large Agent Models (LAMs)는 Chain-of-Action (CoA) 생성을 내재화하여 모델이 외부 도구를 언제, 어떻게 사용할지 자율적으로 결정할 수 있게 된다고 설명합니다. 제안된 AutoCoA 프레임워크는 감독형 미세 조정(Supervised Fine-Tuning, SFT)과 강화 학습(Reinforcement Learning, RL)을 결합하여 모델이 추론과 행동 간 전환을 원활하게 할 수 있도록 합니다. 이로 인해 에이전트 모델의 작업 완료율이 현저히 개선됩니다.

- **Technical Details**: AutoCoA의 주요 구성 요소에는 단계별 행동 트리거, 궤적 수준의 CoA 최적화, 실제 환경 상호작용 비용을 줄이기 위한 내부 세계 모델이 포함됩니다. 이 모델은 사용자가 요청한 작업을 기반으로 CoT와 CoA의 생성을 동시에 관리하여, 더 복잡한 지식 검색 작업을 수행할 수 있도록 합니다. 에이전트 모델은 생성된 각 행동이 도구를 호출하며, 이는 POMDP(부분 관찰 마르코프 결정 프로세스)로 형식화됩니다.

- **Performance Highlights**: 열린 도메인 질의 응답(QA) 작업에서의 평가 결과, AutoCoA로 훈련된 에이전트 모델은 ReAct 기반의 워크플로우에 비해 작업 완료에서 현저한 성능 향상을 보여줍니다. 특히 장기적 사고와 다단계 행동이 요구되는 작업에서도 더욱 두드러진 성과를 나타냈습니다. 향후 연구를 통해 대형 모델과 더 다양한 도구 유형을 포함한 실험 검증을 진행할 계획입니다.



### ProJudge: A Multi-Modal Multi-Discipline Benchmark and Instruction-Tuning Dataset for MLLM-based Process Judges (https://arxiv.org/abs/2503.06553)
- **What's New**: 최근 다중 모달 대형 언어 모델(MLLMs)은 과학 문제를 해결하는 데 인상적인 능력을 보여주고 있습니다. 하지만 이 모델들은 종종 비효율적인 추론을 통해 올바른 답변을 생성하기 때문에 중간 과정에 대한 평가의 중요성이 강조됩니다. ProJudgeBench라는 새로운 벤치마크가 제안되어 MLLM 기반 프로세스 판단자의 평가 능력을 포함한 포괄적이고 체계적인 시험을 제공합니다.

- **Technical Details**: ProJudgeBench는 2,400개의 테스트 사례와 50,118개의 단계 수준의 레이블을 포함하고 있으며, 수학, 물리학, 화학, 생물학과 같은 다양한 과학 분야를 아우릅니다. 각 단계는 전문인력에 의해 정확성, 오류 유형 및 설명으로 세심하게 주석이 달려 있어, 모델의 오류 탐지 및 진단 능력을 체계적으로 평가할 수 있습니다. 이 연구에서는 또한 ProJudge-173k라는 대규모 지침 조정 데이터세트를 제안하고, 모델이 문제 해결 과정에서 명시적으로 추론하도록 요구하는 동적 이중 단계 조정 전략을 통해 평가 능력을 높이고자 합니다.

- **Performance Highlights**: ProJudgeBench를 통해 개방형 소스 모델과 상용 모델 간의 성능 차이가 상당히 크게 나타났습니다. ProJudge-173k와 DDP 전략을 통해 개방형 소스 모델의 프로세스 평가 능력을 크게 향상시킬 수 있었으며, 이들 개선 사항은 프로세스 평가의 신뢰성을 높이는 데 기여합니다. 모든 자원은 향후 신뢰할 수 있는 다중 모달 프로세스 평가 연구를 촉진하기 위해 공개될 예정입니다.



### ChatGPT-4 in the Turing Test: A Critical Analysis (https://arxiv.org/abs/2503.06551)
Comments:
          14 pages, 1 Appendix

- **What's New**: 본 논문은 'Turing Test에서 ChatGPT-4'이라는 Restrepo Echavarría의 최근 출판물을 비판적으로 분석합니다. 이 논문은 ChatGPT-4가 Turing Test를 통과하지 못했다는 주장을 재검토하고, 마땅히 고려되어야 할 최소한의 심각한 테스트 구현의 부재를 challenge 합니다. 기존 비판이 엄격한 기준과 제한된 실험 데이터에 기반한다는 점을 지적하며, 이는 충분히 정당화되지 않는다고 주장합니다.

- **Technical Details**: 이 논문은 세 가지와 두 가지 플레이어 테스트라는 두 가지 상이한 포맷이 유효하다는 것을 입증하며, 각 테스트는 고유한 방법론적 함의를 지닙니다. 저자는 테스트 통과에 대한 절대적 기준(세 플레이어 포맷에서 50% 식별율을 반영)과 상대적 기준(기계 성능이 인간에 얼마나 가까운지 측정하는)을 구별하여, 더 정교한 평가 프레임워크를 제공합니다. 또한 두 테스트 유형의 확률적 기초를 Bernoulli 실험으로 모델링하여 명확히 하며, 세 플레이어 버전에서 상호 연관되고, 두 플레이어 버전에서는 독립적인 성격을 갖습니다.

- **Performance Highlights**: 이 연구는 테스트를 통과하는 이론적 기준과 실험 데이터 간의 엄격한 분리를 가능하게 합니다. 논문은 비판받고 있는 연구의 주요 측면들을 반박함과 동시에, AI의 행동이 인간의 행동과 얼마나 밀접하게 일치 또는 벗어나는지를 측정하는 객관적 기준을 위한 기초를 마련합니다. 이러한 기여들은 Turing Test 구현에 대한 이해를 풍부하게 하는 데 중요한 역할을 합니다.



### ExKG-LLM: Leveraging Large Language Models for Automated Expansion of Cognitive Neuroscience Knowledge Graphs (https://arxiv.org/abs/2503.06479)
- **What's New**: 이 논문은 대 규모 언어 모델(large language models, LLM)을 사용하여 인지 신경 과학 지식 그래프(cognitive neuroscience knowledge graphs, CNKG)의 자동 확장을 위한 ExKG-LLM 프레임워크를 소개합니다. 기존 도구의 한계를 극복하고 정확성(accuracy), 완전성(completeness), 유용성(usefulness)을 향상시키는 데 주안점을 두고 있습니다.

- **Technical Details**: 이 프레임워크는 방대한 과학 논문과 임상 보고서 데이터셋을 활용하여 최신 LLM을 적용해 새로은 엔터티(entity)와 관계(relationship)를 추출하고 최적화(optimize)하며 통합(integrate)합니다. 평가 지표에는 정밀도(precision), 재현율(recall), 그래프 밀도(graph density)가 포함됩니다.

- **Performance Highlights**: 결과적으로 정밀도는 0.80(+6.67%), 재현율은 0.81(+15.71%), F1 점수는 0.805(+11.81%)로 크게 향상되었습니다. 엣지 노드는 각각 21.13% 및 31.92% 증가하였으며, CNKG의 지름(diameter)은 15로 증가하여 더 분산된 구조를 나타냅니다. 전반적으로 시간 복잡도는 O(n log n)으로 개선되었으나, 공간 복잡도는 O(n2)로 증가하여 메모리 사용량이 많아짐을 나타냅니다.



### SKG-LLM: Developing a Mathematical Model for Stroke Knowledge Graph Construction Using Large Language Models (https://arxiv.org/abs/2503.06475)
- **What's New**: 이번 연구에서는 SKG-LLM을 소개합니다. SKG-LLM은 뇌졸중 관련 논문에서 출발하여 지식 그래프(knowledge graph, KG)를 구축하는 데 수학적 모델과 대형 언어 모델(large language models, LLMs)을 활용합니다. 이 방법을 통해 생물 의학 문헌에서 복잡한 관계를 추출하고 정리하여 뇌졸중 연구에서 KG의 정확성과 깊이를 향상시킵니다.

- **Technical Details**: 제안된 방법에서는 데이터 전처리(data pre-processing)에 GPT-4를 사용하였으며, KG 구축 과정 전반에 걸쳐 임베딩(extraction of embeddings) 추출 또한 GPT-4를 통해 이루어졌습니다. 제안된 모델의 성능은 정밀도(Precision)와 재현율(Recall) 두 가지 평가 기준으로 테스트되었고, 추가 검증에서도 GPT-4가 활용되었습니다.

- **Performance Highlights**: Wikidata 및 WN18RR와 비교했을 때, SKG-LLM 접근 방식은 정밀도와 재현율 면에서 더욱 우수한 성능을 보였습니다. SKG-LLM 모델은 전처리 과정에서 GPT-4를 포함하여 0.906의 정밀도 점수와 0.923의 재현율 점수를 달성했습니다. 전문가 리뷰를 통해 결과를 추가 개선함으로써 정밀도는 0.923, 재현율은 0.918로 증가하였습니다. SKG-LLM에 의해 구축된 지식 그래프는 2692개의 노드와 5012개의 엣지를 포함하고 있으며, 13종의 고유한 노드와 24종의 엣자를 지니고 있습니다.



### Think Twice, Click Once: Enhancing GUI Grounding via Fast and Slow Systems (https://arxiv.org/abs/2503.06470)
- **What's New**: 이번 논문에서는 Focus라는 새로운 GUI grounding 프레임워크를 소개합니다. 이 프레임워크는 속도와 분석을 결합하여 작업의 복잡성에 따라 빠른 예측과 체계적인 분석을 동적으로 전환합니다. 이러한 접근 방식은 기존의 빠른 예측 방법이 가지던 복잡한 인터페이스 이해의 한계를 극복하고자 하며, 이는 인간의 두 가지 사고 시스템에서 영감을 받았습니다.

- **Technical Details**: Focus 프레임워크는 GUI grounding을 세 가지 단계로 세분화하여 각 단계에서 인터페이스 요약, 집중 분석 및 정밀 좌표 예측을 수행합니다. 이 프로세스는 복잡한 인터페이스와 시각적 관계를 체계적으로 이해하는 데 도움을 줍니다. Focus는 300K의 훈련 데이터를 사용하여 2B 파라미터 모델을 통해 기초적인 성능을 지속적으로 개선해왔습니다.

- **Performance Highlights**: Focus는 특히 복잡한 GUI 시나리오에서 탁월한 성능을 보여줍니다. ScreenSpot 데이터세트에서 평균 77.4%의 정확도를 기록했으며, 더 어려운 ScreenSpot-Pro에서 13.3% 개선된 결과를 보였습니다. 이번 결과는 Focus의 이중 시스템 접근 방식이 복잡한 GUI 상호작용 시나리오를 개선하는 데 잠재력이 있음을 보여줍니다.



### Explaining Control Policies through Predicate Decision Diagrams (https://arxiv.org/abs/2503.06420)
- **What's New**: 이 연구에서는 Decision Trees (DTs)와 Binary Decision Diagrams (BDDs)의 장점을 통합한 Predicate Decision Diagrams (PDDs)를 소개합니다. PDDs는 DTs의 해석 가능성을 유지하면서도 BDDs의 축소 기법을 활용할 수 있도록 설계되었습니다. 이것은 안전 중요 시스템의 제어기를 더 효율적으로 표현할 수 있는 새로운 방법을 제시합니다.

- **Technical Details**: PDDs는 DTs의 표현성과 BDDs의 축소 기법을 결합한 구조로, bit representation을 사용하여 더 작은 크기의 표현을 가능하게 합니다. 이 구조에서 PDD는 DT에서 추출한 predicates를 변수를 활용하여 BDD에서 일반적인 축소 기법을 적용합니다. 이를 통해 연산상의 긴밀성과 일관성을 지키며 설명이 용이한 제어기를 생성할 수 있습니다.

- **Performance Highlights**: 실험 결과, PDD는 기존의 DT 및 bit-blasted BDD에 비해 크기 축소 효과를 보였고, 표준 벤치마크에 대한 평균적으로 88%의 크기 차이를 줄였습니다. 또한, PDD는 DT와 동일한 수준의 해석 가능성을 유지하면서도 결합된 축소 기법을 사용하여 크기를 줄이는 데 성공했습니다. 이로 인해 PDD는 DT에 대한 유망한 대안으로 자리 잡을 수 있습니다.



### Advancing AI Negotiations: New Theory and Evidence from a Large-Scale Autonomous Negotiations Competition (https://arxiv.org/abs/2503.06416)
- **What's New**: 본 논문에서는 인공지능(AI) 협상 에이전트의 발전에도 불구하고, 컴퓨터 과학 연구와 기존 협상 이론의 통합이 제한적임을 지적합니다. 이를 해결하기 위해 국제 AI 협상 대회를 개최하여 참가자들이 대형 언어 모델(LLM) 협상 에이전트를 위해 프롬프트를 설계하고 개선하는 과정을 반복적으로 진행했습니다. 이 경쟁을 통해 120,000건 이상의 협상이 이루어졌으며, AI-AI 협상에서 기존의 인간 간 협상 이론의 원리가 여전히 중요하다는 사실이 밝혀졌습니다.

- **Technical Details**: 연구의 주요 발견 중 하나는, 높은 친밀감을 가진 에이전트가 상대방의 주관적 가치를 증가시키고 거래 성사율을 높인다는 것입니다. 그러나 거래가 성사되었을 경우, 친밀한 에이전트는 적은 가치를 주장하는 반면, 지배적인 에이전트는 더 많은 가치를 주장했습니다. 또한, AI-AI 협상에서 나타나는 고유한 역학관계는 기존 협상 이론만으로는 완전히 설명되지 않는 경향이 있으며, 특히 체인 오브 사고(chain-of-thought reasoning) 및 프롬프트 주입(prompt injection)과 같은 AI 특화 전략의 효과에 대한 부분이 그러합니다.

- **Performance Highlights**: 이번 연구에서 경쟁에서 우승한 에이전트는 전통적인 협상 준비 프레임워크와 AI 특화 방법론을 혼합한 접근 방식을 사용했습니다. 이러한 결과들은 기존 협상 이론과 AI 특화 전략을 통합하여 에이전트 성능을 최적화하기 위한 새로운 AI 협상 이론 수립의 중요성을 시사합니다. 이 새로운 이론은 자율 에이전트의 독특한 특성을 고려해야 하며, 자동화된 환경에서 전통적인 협상 이론의 적용 조건을 명확히 설정해야 함을 강조합니다.



### Performant LLM Agentic Framework for Conversational AI (https://arxiv.org/abs/2503.06410)
Comments:
          6 pages, 3 figures

- **What's New**: 최근 Voice AI 산업에서 에이전틱 응용 프로그램 및 자동화의 증가로 인해, 복잡한 그래프 기반 로직 워크플로를 탐색하기 위해 대형 언어 모델(LLM)에 대한 의존도가 증가하고 있습니다. 이에 따른 문제를 해결하기 위해, 연구진은 Performant Agentic Framework (PAF)라는 새로운 시스템을 도입하여 LLM이 적절한 노드를 선택하고 액션을 차례로 수행하도록 지원합니다. PAF는 LLM 기반 이성과 수학적 백터 스코어링 메커니즘을 결합해 정확도를 높이면서 지연시간을 줄이는 데 성공했습니다.

- **Technical Details**: PAF는 기본 PAF와 최적화 PAF의 두 가지 구성 요소로 이루어져 있습니다. 이 프레임워크는 노드 및 엣지로 구성된 그래프를 탐색해 주어진 워크플로를 실행할 수 있도록 설계되었습니다. LLM을 이용하여 에이전트의 위치를 동적으로 식별하고, 논리적 조건에 따라 노드를 따라 작업을 수행하게 되며, 이는 생성 작업과 다운스트림 모듈 간의 분리를 통해 최적의 지연시간을 제공합니다.

- **Performance Highlights**: 실험 결과, PAF는 기존의 기준 방법보다 정확도와 지연시간 면에서 유의미한 성과를 보였습니다. 특히, 벡터 기반 점수 매김을 통한 최적화된 방법으로 워크플로의 복잡성을 줄이고 성능을 크게 향상시켰습니다. 이러한 결과는 PAF가 복잡한 비즈니스 환경에서의 실시간 대화형 AI 시스템에 적합한 확장 가능한 접근 방식을 제공함을 시사합니다.



### Optimizing Minimum Vertex Cover Solving via a GCN-assisted Heuristic Algorithm (https://arxiv.org/abs/2503.06396)
- **What's New**: 이 논문에서는 대규모 그래프의 Minimum Vertex Cover (MVC) 문제를 해결하기 위해 GCNIVC라는 새로운 휴리스틱 검색 알고리즘을 제안합니다. GCNIVC는 Graph Convolutional Network (GCN)를 활용하여 그래프의 전반적인 구조를 포착하고, 고품질의 초기 솔루션을 생성합니다. 또한, 새로운 휴리스틱을 도입하여 이중으로 덮인 엣지(dc-edges) 개념과 세 가지 컨테이너를 활용함으로써 검색의 효율성을 향상시킵니다.

- **Technical Details**: GCNIVC 알고리즘은 이론적 기초로 GCN 프레임워크를 기반으로 하여, 신경망이 생성한 정점의 확률을 이용해 초기 솔루션을 신속하게 생산합니다. 초기화를 통해 모델은 정점 이웃 간의 관계와 그래프의 전체 구조를 이해하여 초기 솔루션의 품질을 향상시킵니다. 이 알고리즘은 제안된 이중 덮인 엣지(dc-edges) 개념을 통해 구조적 특성을 체계적으로 활용하게 됩니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋에 대한 실험 결과, GCNIVC는 정확성과 효율성 측면에서 최신 MVC 알고리즘들보다 우수한 성능을 보였습니다. 이러한 결과는 GCNIVC의 GCN 기반 초기화 방법과 엣지 정보 활용 검색 전략의 효과성을 입증합니다. GCNIVC는 대규모 그래프 최적화 문제를 해결하기 위한 새로운 도구가 될 것으로 기대됩니다.



### Causal Discovery and Inference towards Urban Elements and Associated Factors (https://arxiv.org/abs/2503.06395)
- **What's New**: 이번 연구에서는 시민, 위치 및 이동성 간의 인과관계를 탐구하는 새로운 도시 인과 컴퓨팅 프레임워크를 제안합니다. 기존의 상관 분석은 인과관계를 정확히 반영하지 못할 수 있기 때문에, 연구자들은 인과 그래프를 발견하여 도시 요소 간의 복잡한 관계를 특성화하고자 노력하였습니다.

- **Technical Details**: 본 연구의 핵심은 강화 학습 알고리즘을 설계하여 도시 요소 간의 잠재적 인과 그래프를 발견하는 것입니다. 이 그래프는 쌍별 도시 요소 간의 인과 효과를 추정하기 위한 가이드를 제공하며, Propensity Score Matching 방법을 활용하여 상관 분석에서 발생할 수 있는 혼란 요인을 제거합니다.

- **Performance Highlights**: 실험 결과, 발견된 인과 그래프는 계층적 구조를 보여주며, 시민이 위치에 영향을 미치고 두 요소가 함께 이동성 행동을 변화시킨다는 것을 밝혔습니다. 이 연구는 또한 인과 분석이 도시 컴퓨팅 과제의 성능을 향상시킬 수 있음을 증명하였고, 인과 관계의 중요성을 발휘하여 도시 데이터를 보다 효과적으로 활용할 수 있음을 보여줍니다.



### General Scales Unlock AI Evaluation with Explanatory and Predictive Power (https://arxiv.org/abs/2503.06378)
- **What's New**: 이 논문에서는 AI 평가를 위한 일반적인 척도(general scales)를 소개하며, 기존 AI 벤치마크가 실제로 무엇을 측정하는지 설명하고, AI 시스템의 능력 프로파일(ability profiles)을 추출하며, 새로운 작업 인스턴스에 대한 성능을 예측하는 방법을 제안합니다. 18개 새로운 '요구 수준 주석' (demand-level-annotation) 기준을 통해 모든 테스트 인스턴스에 적용 가능하며, AI의 안전하고 효과적인 사용을 위한 안정적인 배포를 지원합니다. 또한, 새로운 방법론을 통해 AI 시스템의 다양성과 복잡성을 효과적으로 다룰 수 있는 가능성을 제시합니다.

- **Technical Details**: 이 논문에서 제안하는 방법론은 18개의 정성적인 기준을 활용하여 AI의 과제를 정의하고 평가 기준의 객관성을 높이며, 자동화된 방식으로 실험됩니다. 각 요구 수준은 0에서 무한대까지를 아우르며, AI 시스템의 능력을 측정할 수 있는 상위 개념으로 자리 잡습니다. 이를 통해 AI 시스템의 성능 예측이 가능해지며, 특히 새로운 작업이나 벤치마크에 대해서도 높은 예측력을 발휘합니다.

- **Performance Highlights**: 실험 결과, 다양한 대형 언어 모델에 대해 18개의 능력 추정치가 생성되며, 이는 AI 시스템의 강점과 약점을 명확하게 파악할 수 있는 이점을 제공합니다. 특히, 기존의 블랙박스 기반 예측 모델과 비교했을 때, 인스턴스 수준에서보다 우수한 성능을 발휘했습니다. 이러한 결과는 AI 모델의 선택 및 안전한 운영 영역을 정의하는 데에 유용한 가능성을 엽니다.



### Higher-Order Belief in Incomplete Information MAIDs (https://arxiv.org/abs/2503.06323)
- **What's New**: 본 논문에서는 불완전 정보 멀티 에이전트 영향도 다이어그램(incomplete information multi-agent influence diagrams, II-MAIDs)을 도입합니다. 기존의 멀티 에이전트 영향도 다이어그램(MAIDs)에서는 에이전트들이 세계에 대한 공통된 믿음을 공유한다고 가정하지만, II-MAIDs는 에이전트들이 서로 다른 1차 및 2차 신념을 가지고 있는 상황을 모델링합니다.

- **Technical Details**: II-MAIDs는 무한 및 유한 깊이 구조를 갖는 두 가지 형태로 정의되며, 불완전 정보가 있는 확대형 게임(extensive form game, EFG)과의 등가 관계가 증명됩니다. 이 연구에서는 일반적으로 존재하는 균형 개념들이 에이전트들의 비일관적인 신념 하에서는 직관적이지 않다는 점을 강조하며, 더 나아가 재귀적 최선 응답(recursive best-response) 기반의 새로운 해결 개념을 제시합니다.

- **Performance Highlights**: II-MAIDs는 기존의 해결 개념을 상속받아 내시 균형(Nash equilibrium)을 구현할 수 있지만, 기존의 개념이 현실적이지 않은 경우가 많다는 점을 지적합니다. 일반적인 신념 계층을 사용한 II-MAIDs의 유한 깊이 변형을 통해, 정책들이 반복적으로 최선 응답으로 할당됨으로써 보다 자연스럽고 직관적인 해결 개념이 정의됩니다.



### LapSum -- One Method to Differentiate Them All: Ranking, Sorting and Top-k Selection (https://arxiv.org/abs/2503.06242)
- **What's New**: 본 논문에서는 soft ranking, soft top-k 선택 및 soft permutations를 포함한 미분 가능한 순서형 작업을 구성하는 새로운 기술을 제시합니다. 특히, Laplace 분포의 합으로 정의되는 함수인 LapSum의 역함수에 대해 효율적인 닫힌 형식을 활용하여, 높은 활성도를 선택하는 과정에서 낮은 계산 및 메모리 복잡도를 보장합니다. 이 접근 방식은 $O(n\log n)$의 시간복잡도로 손실 및 기울기 계산을 가능하게 합니다. 이는 고차원 벡터와 큰 k 값에 대해 기존의 최첨단 기술보다 우수한 성능을 입증합니다.

- **Technical Details**: 이 연구의 주요 목표는 임의 밀도를 기반으로 하는 일반 이론을 구축하는 것으로, 이를 통해 Laplace 분포를 포함하면 모든 이전 문제에 대해 닫힌 형식의 안정적이고 효율적인 솔루션을 제공합니다. 제안한 LapSum 접근 방식은 O(n\log n)의 낮은 시간 복잡도와 O(n)의 메모리 복잡도를 가지며 이론적으로도 검증된 바 있습니다. 또한, CPU와 CUDA 모두에서 사용하기 쉬운 코드를 제공하여 대규모 최적화 문제에 대한 실용성을 높였습니다.

- **Performance Highlights**: 저자에 따르면, LapSum 방법은 기존의 다른 모든 방법들과 비교하여 시간 및 메모리 효율성이 뛰어나며, 이론적으로도 안정적인 성과를 나타냅니다. 실험을 통해 고차원 벡터에 적용했을 때 우수한 성능을 보여 주었고, 기존 방법들에 비해 손실과 기울기 계산의 효율성을 크게 향상시켰습니다. 또한, 대규모 순위 및 미분 가능한 정렬 문제에서의 활용 가능성을 강조합니다.



### Breaking Free from MMI: A New Frontier in Rationalization by Probing Input Utilization (https://arxiv.org/abs/2503.06202)
- **What's New**: 이번 논문에서는 최대 상호 정보량(MMI) 기준이 감소하는 한계 효용(diminishing marginal utility) 문제를 겪고 있다는 점을 강조합니다. 기존 방법론과는 달리, 본 연구는 신경망이 실제로 활용할 수 있는 입력 부분을 찾아내기 위해 가중치 행렬의 능력 공간에 따라 다른 후보 방안을 비교하는 새로운 목표를 제안합니다. 제안된 방법은 해석 가능성의 새로운 길을 열어줄 것으로 기대됩니다.

- **Technical Details**: RNP(이성적 신경 예측) 프레임워크는 정보가 가장 중요한 입력 부분인 이성(rationale)을 식별하고 이를 예측자에게 전송하여 예측을 수행하는 협력적 게임을 기반으로 합니다. 본 연구는 낮은 차수(low-rank) 가중치 행렬의 특징을 활용하여 해당 입력이 신경망에 의해 잘 학습되는지, 즉 비제로 순위(subspace) 방향과 일치하는지를 기준으로 rationales를 찾고자 합니다. 가중치 행렬을 통한 특징의 방향성을 고려하여, 이 연구는 기존 MMI 기준의 한계를 극복하고자 합니다.

- **Performance Highlights**: 우리의 방법은 세 가지 네트워크 아키텍처(GRUs, BERT, GCN)와 함께 4개의 텍스트 분류 데이터셋 및 1개의 그래프 분류 데이터셋으로 실험하여 기존 MMI 및 그 개선 변형들보다 우수한 성능을 보였습니다. 또한, 우리 방법은 LLM(LLM, llama-3.1-8b-instruct)과 비교했을 때 유사한 결과를 보였고, 경우에 따라서는 더 뛰어난 성능을 나타내기도 했습니다.



### Object-Centric World Model for Language-Guided Manipulation (https://arxiv.org/abs/2503.06170)
- **What's New**: 이 연구에서는 언어 지침에 의해 안내되는 개체 중심의 월드 모델을 처음으로 제안합니다. 제안된 모델은 현재 상태를 개체 중심의 표현으로 인식하고, 자연어 지침을 조건으로 미래 상태를 예측합니다. 기존의 확산 기반 생성 모델보다 더 효율적이며 샘플 및 계산 효율성에서 우수한 성능을 보여줍니다.

- **Technical Details**: 제안된 모델은 슬롯 어텐션(Slot Attention) 기반 인코더를 사용하여 개체 중심의 표현을 얻고, 그 표현을 바탕으로 즉각적인 행동 예측을 수행합니다. 이 방법은 자동 인코딩 구조를 활용하여 무감독(un-supervised) 방식을 통해 프레임을 재구성하고, 이를 통해 제어 작업에서 효율성과 성능 향상을 도모합니다. 또한, 모델은 언어 지침에 의해 유연한 예측이 가능하여 객체 인식이 중요한 조작 작업에서 유리합니다.

- **Performance Highlights**: 모델은 비주얼-언어-모터 제어 과제에서 기존의 최첨단 모델보다 우수한 샘플 및 계산 효율성으로 성과를 달성했습니다. 제안된 방법은 보지 못한 과제 설정에 대한 일반화 성능도 탐구하고 있으며, 객체 중심 표현을 사용해 행동 예측을 위한 다양한 방법을 연구했습니다. 최종적으로, 제안된 모델은 언어 기반 지침을 통해 향상된 조작 작업의 수행 능력을 보여줍니다.



### VACT: A Video Automatic Causal Testing System and a Benchmark (https://arxiv.org/abs/2503.06163)
- **What's New**: 이번 논문에서는 텍스트에 기반한 Video Generation Models (VGMs)의 발전이 실제 세계 수준의 비디오 생성의 접근성과 비용 효율성을 높이는 데 기여하고 있음을 강조합니다. 그러나 생성된 비디오의 정확성이 떨어지고 기본 물리 법칙에 대한 이해가 부족하다는 문제를 지적합니다. 이를 해결하기 위해 VACT라는 새로운 자동화된 프레임워크를 제안하여 VGMs의 인과적 이해(causal understanding)를 평가하는 시스템을 개발했습니다.

- **Technical Details**: VACT는 인과 분석(causal analysis) 기법과 대형 언어 모델(large language model) 보조 도구를 결합하여 다양한 시나리오에서 VGMs의 인과적 행동을 인간 주석 없이 평가할 수 있도록 설계되었습니다. 이 프레임워크는 다양한 맥락에서 모델의 인과적인 측면을 측정하고, 다층 인과 평가 메트릭(multi-level causal evaluation metrics)을 통해 VGMs의 인과적 성능(performance)에 대한 세부 분석을 제공합니다.

- **Performance Highlights**: 제안된 프레임워크를 통해 여러 기존 VGMs을 벤치마킹하여 그들의 인과적 추론(capabilities of causal reasoning) 능력에 대한 통찰을 제공합니다. 이러한 연구는 VGMs의 신뢰성을 높이고, 실제 응용 가능성을 개선하는 데 기여할 기반을 마련합니다.



### System 0/1/2/3: Quad-process theory for multi-timescale embodied collective cognitive systems (https://arxiv.org/abs/2503.06138)
Comments:
          Under review

- **What's New**: 이 논문은 이중 프로세스 이론을 확장하여 System 0/1/2/3 프레임워크를 소개합니다. 이 프레임워크는 인지의 쿼드 프로세스 모델을 사용하며, 물리적 배경과 집단 지성을 통합하여 다양한 시간적 차원을 아우릅니다. 특히 System 0은 사전 인지적인 과정, System 3은 기호의 출현을 대상으로 하여 효율적인 적응 능력을 설명하죠.

- **Technical Details**: System 0/1/2/3 모델은 빠른 직관적 사고(System 1)와 느린 심의적 사고(System 2)에 추가로 사전 인지적 프로세스(System 0)와 집단적 지능(System 3)을 통합합니다. 이러한 모델은 여러 시간적 스케일에서 인지 과정의 다양성을 설명하고, 행동의 적응을 위한 통합된 이론적 토대를 제공합니다. 또한, 뇌의 내부에서 발생하는 자유 에너지 원칙(free-energy principle)이 생물학적 적응을 이해하는 데 중요한 역할을 합니다.

- **Performance Highlights**: System 0/1/2/3 프레임워크는 전통적인 이중 프로세스 이론을 확장하여 인지의 복잡한 동적 구조를 설명합니다. 이를 통해 인공지능, 로봇공학, 그리고 집단 지성에서의 도전을 다루고, 인간 사회에서 기호 체계의 관점에서 혁신적인 통찰을 제공합니다. 또한, 다양한 시간 스케일을 아우르는 인지의 상호작용을 통해 협력적이고 적응적인 행동의 발전 가능성을 보여줍니다.



### MANDARIN: Mixture-of-Experts Framework for Dynamic Delirium and Coma Prediction in ICU Patients: Development and Validation of an Acute Brain Dysfunction Prediction Mod (https://arxiv.org/abs/2503.06059)
- **What's New**: 이 연구에서는 MANDARIN (Mixture-of-Experts Framework for Dynamic Delirium and Coma Prediction in ICU Patients)이라는 새로운 신경망 모델을 제안합니다. MANDARIN은 실시간으로 급성 뇌 장애(ABD)를 예측할 수 있도록 설계되었으며, 약 150만 개의 매개변수를 가진 혼합 전문가 네트워크입니다. 이 모델은 ICU 환자의 뇌 상태를 예측하기 위해 시간적 및 정적 데이터를 통합하여 사용합니다.

- **Technical Details**: MANDARIN 모델은 2008년부터 2019년까지 두 병원에서 수집된 92,734명의 환자 데이터로 훈련되었습니다. 모델은 12시간에서 72시간 후의 뇌 상태를 예측하며, 현재 상태를 고려하기 위해 다중 분기(multi-branch) 접근법을 사용합니다. 또한 외부 검증은 15개 병원에서의 11,719명의 환자 데이터로 수행되었습니다.

- **Performance Highlights**: MANDARIN 모델은 외부 및 예측 코호트 모두에서 기존의 신경학적 평가 도구(GCS, CAM, RASS)보다 유의미하게 높은 성능을 보였습니다. 특히, 외부 데이터에서의 섬망 예측(AUROC 75.5%)과 혼수 예측(AUROC 87.3%)에서 개선된 결과를 나타냈습니다. 이 도구는 ICU 환자의 지속적인 뇌 상태 모니터링을 통해 임상의의 의사 결정 지원에 기여할 가능성을 갖고 있습니다.



### DSGBench: A Diverse Strategic Game Benchmark for Evaluating LLM-based Agents in Complex Decision-Making Environments (https://arxiv.org/abs/2503.06047)
Comments:
          43 pages, 5 figures, conference

- **What's New**: DSGBench는 전략적 의사결정에 대한 평가를 위한 새로운 플랫폼으로, 최신 LLM 기반 에이전트를 평가하기 위해 설계되었습니다. 이 플랫폼은 복잡한 전략 게임 6종을 포함하여, 장기적이며 다차원적인 의사결정 요구를 충족합니다. DSGBench는 다양한 난이도를 설정하고 다수의 목표를 지원하여 더 맞춤화된 평가를 가능하게 합니다.

- **Technical Details**: DSGBench는 다섯 가지 특정 차원에서의 성능을 분석하는 세분화된 평가 점수 체계를 사용하여 의사결정 능력을 평가합니다. 또한, 자동화된 의사결정 추적 메커니즘을 도입하여 에이전트의 행동 패턴과 전략의 변화를 심층적으로 분석할 수 있습니다. 이는 복잡한 결정작업에 대한 LLM 에이전트의 실제 능력을 정밀하게 검토하는 데 도움을 줍니다.

- **Performance Highlights**: DSGBench를 여러 인기 있는 LLM 기반 에이전트에 적용하여 이 플랫폼이 제공하는 귀중한 통찰을 입증했습니다. 이 연구 결과는 전략 결정 작업에서 LLM 기반 에이전트를 선택하는 데 유용하며, 향후 개발을 개선하는 데 기여할 수 있음을 보여줍니다. DSGBench는 [이 URL](https://)에서 이용할 수 있습니다.



### Empowering Edge Intelligence: A Comprehensive Survey on On-Device AI Models (https://arxiv.org/abs/2503.06027)
- **What's New**: 이번 논문은 AI 기술의 발전에 따라 엣지 디바이스에서의 AI 모델 배포에 대한 현황 및 기술적 도전 과제를 포괄적으로 살펴보고 있다. 특히, 엣지 컴퓨팅과 IoT의 확산 속에 로컬 데이터 처리를 위한 AI 모델의 중요성이 강조되고 있으며, 데이터 프라이버시와 실시간 성능이 중요한 특성으로 언급된다. 또한, 데이터 전처리, 모델 압축, 하드웨어 가속화와 같은 최적화 및 구현 전략에 대한 논의도 포함되어 있다.

- **Technical Details**: 이 연구는 엣지 환경에서의 AI 모델 구현 시 겪는 기술적 문제를 분석한다. 일반적으로, 엣지 디바이스는 제한된 자원속에서 빠른 데이터 처리를 위해 AI 모델의 효율성을 높이기 위한 최적화 전략이 필요하다. 또한, 데이터 프라이버시 보호를 위해 데이터를 로컬에서 처리하도록 설계된 AI 모델의 다양한 사례에 대해서도 다룬다.

- **Performance Highlights**: 연구는 실시간, 자원 제약, 데이터 프라이버시라는 세 가지 주요 특성을 가진 엣지 디바이스에서의 AI 모델들이 어떻게 발전하고 있는지를 설명한다. 이러한 모델들이 스마트폰, 의료 장비, 자율주행차 등 다양한 분야에서 어떻게 적용되는지에 대한 통찰을 제공하며, 특히 Industry 4.0에서의 활용 가능성에 대해 강조하고 있다. 따라서 전반적으로, 해당 연구는 엣지 디바이스에서의 AI 모델의 미래 방향성과 발전을 위한 중요한 기초 자료로서의 역할을 한다.



### Bayesian Graph Traversa (https://arxiv.org/abs/2503.05963)
Comments:
          26 pages, 7 tables, 2 figures

- **What's New**: 이번 연구는 불확실한 그래프를 탐색하는데 있어 Bayesian 결정 분석 접근법을 고려하고 있습니다. 여행자는 그래프의 인접 행렬을 알고 있으며 시작 위치는 알고 있지만, 보상과 비용은 알지 못합니다. Bayesian적 관점으로, 여행자는 자신의 신념을 가우시안 프로세스 사전으로 인코딩하고 기대 효용을 극대화하려고 합니다. 이는 정보 수집과 네트워크 경로 탐색 문제를 결합한 의사 결정 분석 관점에서 해결됩니다.

- **Technical Details**: 연구에서는 결정 분석 원칙을 채택하여 여행자가 기대 효용이 부정적이거나 정해진 수의 의사 결정을 내릴 때까지 탐색을 중단하도록 설계되었습니다. 문제는 NP-Hard로 밝혀지며, 최적 경로의 성질을 도출하여 탐색과 활용의 균형을 맞추는 휴리스틱을 제공합니다. 이 연구는 각 노드의 첫 방문 시 보상을 받고, 모든 에지 통과 시 비용을 부과받는 문제 설정을 가지고 있습니다. 또한, 가우시안 프로세스 사전을 사용하여 보상과 비용에 대한 신뢰 구간을 정의하고 있습니다.

- **Performance Highlights**: 실제 사례 연구를 통해 무인 항공 시스템을 이용한 공공 안전을 위한 경로 탐색을 분석했습니다. 다양한 Erdos-Renyi 환경에서 정책 성능을 경험적으로 연구하여 여행자가 최적의 결정을 내리는 데 있어 본 연구가 제공하는 모델과 정책의 유용성을 강조합니다. 또한, 경쟁 연구와 비교했을 때, 실시간 학습을 통해 여행자가 그래프와 상호작용하며 정보를 수집하는 과정의 중요성을 부각하고 있습니다.



### Enhancing Reasoning with Collaboration and Memory (https://arxiv.org/abs/2503.05944)
Comments:
          17 pages, 6 figures

- **What's New**: 이 논문은 연속적인 협업 학습 시스템을 제안하며, 다양한 LLM (Large Language Model) 에이전트들이 메모리를 공동으로 구축하여 문제를 해결하는 방식을 탐구합니다. 독창적인 점은 동일한 에이전트를 사용한 자가 일관성(self-consistency) 방법을 넘어서, 다양한 맥락을 가진 에이전트와 요약자 에이전트를 도입한 것입니다. 이를 통해 LLM의 추론 성능 향상에 기여할 가능성을 제시하고 있습니다.

- **Technical Details**: 이 연구에서는 체인 오브 사고 (chain-of-thought), 다중 에이전트 협업(multi-agent collaboration), 그리고 메모리 은행(memory bank)을 결합하여 LLM의 추론 과정을 향상시키려고 합니다. 또한, 다양한 메모리 검색 방식 및 예시를 기반으로 하여, 고정된, 무작위 선택, 유사성 기반 검색 등을 비교 평가합니다. 이는 LLM 모델이 다양한 관점에서 문제를 접근함으로써 더 나은 성능을 발휘할 수 있도록 돕습니다.

- **Performance Highlights**: 실험을 통해 다양한 맥락을 가진 에이전트가 자가 일관성 이상의 성능을 발휘하고, 무작위 검색 방식이 유사성 기반 검색보다 더 높은 정확도를 가진다는 것을 발견했습니다. 구동 중 학습된 메모리는 비교적 효율적으로, 고정된 메모리와 유사한 성능을 나타내며, 아날로지적 프롬프트(analogical prompting)는 메모리 변화에 대해 더 견고한 반응을 보였습니다. 마지막으로, 요약자 에이전트는 에이전트 성능이 낮을 때 더 큰 도움이 되는 것으로 나타났습니다.



### Quantum-like cognition and decision making in the light of quantum measurement theory (https://arxiv.org/abs/2503.05859)
- **What's New**: 이번 논문에서는 квантовая (quantum) 측정의 다양한 방법이 인지 (cognition) 및 의사결정 (decision making) 모형에 어떻게 적용될 수 있는지를 다룹니다. 저자들은 기존의 프로젝티브 측정 (projective measurements)만으로는 모든 인지 효과를 설명할 수 없음을 강조하며, 새로운 클래스인 {	extit{sharp repeatable non-projective measurements}} - ${	extcal{SRar{P}}}$의 필요성을 제시하고 있습니다. 이 측정법은 양자 물리학에서 거의 사용되지 않는 방식으로, 기존 연구와 다른 방향성을 제안합니다.

- **Technical Details**: 양자 측정 이론은 일반적으로 프로젝티브 측정에 의해 표현됩니다. 하지만 논문에서는 이러한 방식이 인지에서 발생하는 모든 효과를 설명하지 못한다고 주장합니다. 저자들은 인지의 경우에 사용될 수 있는 비프로젝티브 측정의 중요성을 설명하며, 양자 측정 이론의 복잡성을 강조합니다. 이 과정에서 두 종류의 비가환성 (noncommutativity) 즉, 관측 가능 항목의 비가환성과 상태 업데이트 맵의 비가환성이 나옵니다.

- **Performance Highlights**: 또한 저자들은 양자역학에서 비가환성이 인지 모델링에 미치는 영향을 탐구합니다. 이 연구는 관측치 (observable)와 도구 (instrument)가 각각 어떤 비가환성을 가지고 있는지를 분석하여 두 가지 다른 형태의 비가환성이 양자 물리학 및 인지 모델의 '비고전성 (non-classicality)'에 중대한 영향을 미친다고 제안합니다. 따라서 인지에서의 양자적 특성을 명확히 구별하는 것의 중요성을 강조하며 앞으로의 연구 방향성을 제공합니다.



### Market-based Architectures in RL and Beyond (https://arxiv.org/abs/2503.05828)
Comments:
          Accepted at AAMAS 2025

- **What's New**: 이 논문에서는 시장 기반 에이전트(market-based agents)의 새로운 알고리즘을 제안합니다. 기존의 RL(강화학습) 알고리즘과 달리, 상태는 '상품(goods)'이라 불리는 몇 개의 축으로 분해되어 더 큰 전문화(specialization)와 병렬성(parallelism)을 가능하게 합니다. 추가적으로, 시장 기반 알고리즘이 AI의 여러 도전 과제를 해결할 수 있는 가능성을 제시하고 있습니다.

- **Technical Details**: 시장 기반 알고리즘은 두 가지 일반적인 프레임워크를 제시합니다: (1) '딥 마켓(deep market)'과 (2) '와이드 마켓(wide market)'. 기존 알고리즘과 큰 차이는 없는 딥 마켓은 모든 POMDP(부분 관찰 마코프 결정 프로세스)에 적용 가능하며, 와이드 마켓은 상태 공간을 상품으로 분할하여 실제 세계 시장의 성공 사례를 더 잘 반영합니다. 이는 더 큰 전문화와 병렬 처리를 가능하게 하며, 시장 기반 알고리즘을 생성하기 위한 파이썬 라이브러리가 출판과 함께 제공될 예정입니다.

- **Performance Highlights**: 이 연구는 시장 기반 에이전트들이 기본적인 감독학습(supervised learning) 작업에도 적용될 수 있으며, 신경망(neural networks)이 이들 에이전트의 특수한 경우로 나타날 수 있다는 점을 보여줍니다. 특히 시장은 검색(search), 완전한 피드백(complete feedback), 동적 확장(dynamic scaling) 등 여러 현재 AI 문제를 자연스럽게 해결하는 프레임워크로 작용할 수 있습니다. 또한, LLMs(대형 언어 모델)와 결합하여 인간 피드백 메커니즘을 개선하는 데에도 사용될 수 있는 새로운 방법들이 제시되었습니다.



### DriveGen: Towards Infinite Diverse Traffic Scenarios with Large Models (https://arxiv.org/abs/2503.05808)
Comments:
          8 pages, 3 figures

- **What's New**: 이 논문에서는 DriveGen이라는 새로운 교통 시뮬레이션 프레임워크를 제안합니다. 이 프레임워크는 보다 다양한 교통 흐름 생성을 목표로 하며, 사용자 맞춤형 디자인을 지원합니다. 기존의 데이터 기반 접근 방식을 넘어 여러 모델과 기술을 활용하여 더욱 풍부한 시뮬레이션 환경을 제공합니다.

- **Technical Details**: DriveGen은 두 가지 내부 단계를 포함합니다. 초기화 단계에서는 대형 언어 모델(large language model)과 검색 기술(retrieval technique)을 사용하여 지도(map)와 차량 자산(vehicle assets)을 생성합니다. 롤아웃 단계에서는 시각 언어 모델(visual language model)과 특별히 설계된 디퓨전 플래너(diffusion planner)를 통해 선택된 웨이포인트 목표(waypoint goals)에 따라 경로(trajectories)를 출력합니다.

- **Performance Highlights**: 실험 결과, DriveGen에서 생성된 시나리오와 코너 케이스(corner cases)는 기존의 최첨단 기준(state-of-the-art baselines)에 비해 우수한 성능을 보였습니다. 또한, DriveGen의 합성된 교통 시뮬레이션은 일반적인 주행 알고리즘의 성능 최적화를 지원하여, 전체 프레임워크의 효율성을 입증하였습니다.



### A Comprehensive Survey of Fuzzy Implication Functions (https://arxiv.org/abs/2503.05702)
- **What's New**: 이번 논문에서는 퍼지 논리(Fuzzy Logic)에서 퍼지 함의 함수(Fuzzy Implication Functions)의 다양한 가족에 대한 체계적이고 포괄적인 개요를 제공합니다. 최근 10년 동안 다양한 퍼지 함의 함수가 소개되었으나, 기존 문헌은 제한된 수의 가족에 초점을 맞추는 경향이 있었습니다. 새로운 가족들은 각기 다른 구조적 방법과 고유의 주요 속성을 가지고 있습니다.

- **Technical Details**: 이 문서에서는 퍼지 함의 함수의 동기, 속성 및 잠재적 응용 측면을 강조합니다. 퍼지 함의 함수는 고전 논리의 조건부를 [0,1] 구간에서 진리 정도를 다룰 수 있도록 확장합니다. 특히, 특정 구조적 방법에 의해 정의된 다양한 가족이 논의됩니다.

- **Performance Highlights**: 이 연구는 이론적 연구자들이 중복을 피할 수 있도록 자료를 조직화하고, 실무자들이 특정 응용에 적합한 연산자를 선택하는 데 도움이 되는 유용한 자원으로 기능합니다. 따라서 퍼지 논리 분야에서 실용성을 높일 수 있는 가능성을 제시합니다.



### NeuroChat: A Neuroadaptive AI Chatbot for Customizing Learning Experiences (https://arxiv.org/abs/2503.07599)
Comments:
          16 pages, 6 figures, 1 table

- **What's New**: 이 논문은 Generative AI가 개인화된 학습 경험을 제공하는 방법에 대해 다루고 있습니다. 특히, NeuroChat이라는 새로운 neuroadaptive AI 튜터를 제시하는데, 이는 실시간 EEG(전기 생리학적 뇌파) 기반의 참여도 추적을 통해 학습자에게 최적화된 피드백을 제공합니다. 이를 통해 AI 튜터는 학습자의 인지 상태를 실시간으로 파악하고 학습 내용을 동적으로 조정할 수 있습니다.

- **Technical Details**: NeuroChat은 닫힌 루프 시스템(closed-loop system)을 활용하여 학습자의 인지 참여(cognitive engagement)를 지속적으로 모니터링합니다. 이 시스템은 학습 내용의 복잡성(content complexity), 응답 스타일(response style), 학습 속도(pacing)를 실시간으로 조절합니다. 이러한 기능은 EEG를 통해 학습자의 신경 상태를 분석하여 달성됩니다.

- **Performance Highlights**: 파일럿 연구(참여자 24명) 결과, NeuroChat은 표준 LLM 기반 챗봇과 비교하여 인지적 및 주관적 참여를 향상시키는 것으로 나타났습니다. 그러나 학습 결과에 즉각적인 영향을 미치지는 않는 것으로 보입니다. 이러한 발견은 LLM에서의 실시간 인지 피드백(real-time cognitive feedback)의 가능성을 보여주며, 적응형 학습(adaptive learning) 및 AI 튜터링의 새로운 방향성을 나타냅니다.



### Denoising Hamiltonian Network for Physical Reasoning (https://arxiv.org/abs/2503.07596)
- **What's New**: 본 논문에서는 Denoising Hamiltonian Network (DHN)를 제안하며, 이는 해밀토니안 역학의 연산자를 더 유연한 신경망 연산자로 일반화합니다. DHN은 비국소적인 시간 관계를 포착하고, 노이즈 제거 메커니즘을 통해 수치적 적분 오류를 완화합니다. 또한, DHN은 글로벌 컨디셔닝을 지원하여 여러 시스템 모델링을 가능하게 합니다.

- **Technical Details**: DHN은 시스템 상태의 그룹을 토큰으로 취급하여 시스템 역학에 대한 총체적인 추론을 가능하게 합니다. 노이즈 제거 목표를 통합하여 장기 예측의 안정성을 향상시키고, 다양한 작업 문맥에서 유연한 훈련과 추론을 지원합니다. 글로벌 조건 부여를 통해 여러 물리적 시스템을 모델링할 수 있으며, 시스템 특성(예: 질량, 진자 길이)을 인코딩하는 공유 글로벌 잠재 코드를 사용합니다.

- **Performance Highlights**: DHN의 효과성과 유연성을 평가하기 위해 세 가지 물리적 추론 작업을 수행합니다: (i) 궤적 예측 및 완성, (ii) 부분 관측으로부터의 물리적 매개변수 추론, (iii) 점진적 초해상도로 희소 궤적 보간. 이러한 작업은 DHN이 전통적인 전방 시뮬레이션 및 다음 상태 예측을 넘어서는 물리적 추론을 위한 더 넓은 응용 가능성을 열어준다는 점을 입증합니다.



### Filter Images First, Generate Instructions Later: Pre-Instruction Data Selection for Visual Instruction Tuning (https://arxiv.org/abs/2503.07591)
Comments:
          Accepted at Computer Vision and Pattern Recognition Conference (CVPR) 2025

- **What's New**: 이번 논문에서는 Visual Instruction Tuning (VIT) 데이터 선택 방식인 PreSel을 소개합니다. PreSel은 유효한 레이블이 없는 이미지에서 가장 유용한 샘플을 직접 선택하고, 선택된 이미지에 대해서만 명령어를 생성함으로써 비용을 절감하는 접근 방식을 제안합니다. 또한, 기존의 VIT 방법과 달리, 고비용의 명령어 생성 단계 이전에 이미지 선택을 수행하여 효율성을 크게 증가시킵니다.

- **Technical Details**: PreSel 접근 방식은 두 단계로 구성됩니다. 첫 번째는 각 비전 작업의 중요성을 자동으로 추정하여 샘플링 예산을 결정하는 것입니다. 두 번째 단계에서는 경량 비전 인코더를 사용하여 unlabeled 이미지의 특징을 추출하고, 클러스터링 기법을 통해 각 클러스터 내의 대표 이미지를 선택합니다. 마지막으로, 선택된 이미지에 대해 명령어를 생성하여 LVLM (Large Vision-Language Models) 훈련을 준비합니다.

- **Performance Highlights**: 실험 결과, LLaVA-1.5와 Vision-Flan 데이터셋을 활용하여, PreSel은 전체 데이터의 15%만 이용하였음에도 불구하고 VIT에서 발생하는 비용을 15%로 줄이는 동시에 성능면에서도 유사한 결과를 달성했습니다. PreSel은 다양한 아키텍처와 크기의 LVLM에서 사용할 수 있는 전이 가능성을 입증하며, 효율적인 모델 개발을 위해 선택된 서브셋을 공개합니다.



### When Large Vision-Language Model Meets Large Remote Sensing Imagery: Coarse-to-Fine Text-Guided Token Pruning (https://arxiv.org/abs/2503.07588)
Comments:
          12 pages, 6 figures, 7 tables

- **What's New**: 이 논문에서는 대규모 원격 센싱 이미지(RSI)의 효율적인 비전-언어 이해를 위한 새로운 접근법을 제안합니다. 제안된 방법은 Text-Guided Token Pruning(텍스트 기반 토큰 프루닝)과 Dynamic Image Pyramid(DIP) 통합을 통해 이미지를 처리하며, 이를 통해 정보 손실을 최소화하면서 계산 복잡성을 줄이는 것을 목표로 합니다. 또한, 새로운 벤치마크인 LRS-VQA를 통해 7,333개의 질문-답변(QA) 쌍을 수집하여 LVLM의 평가 기준을 개선하였습니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 구성요소, 즉 Region Focus Module(RFM)과 Dynamic Image Pyramid(DIP)을 포함합니다. RFM은 텍스트 관련 비전 토큰을 식별하기 위해 텍스트 인식 지역 로컬라이징 능력을 이용합니다. 이어서 DIP는 중요 이미지 타일을 선택하고 비전 토큰을 프루닝하는 코스-투-파인 전략을 적용하여 효율적으로 전체 대규모 이미지를 처리합니다.

- **Performance Highlights**: 제안된 방법은 동일한 데이터를 사용하는 네 가지 데이터 세트에서 기존의 고해상도 전략을 초과하는 성능을 보였습니다. 또한 기존의 토큰 축소 방법들에 비해 고해상도 설정에서 더 높은 효율성을 나타내었습니다. 이 연구는 아키텍처에 독립적이며, 성능 개선과 효율성 상승을 모두 달성합니다.



### Robusto-1 Dataset: Comparing Humans and VLMs on real out-of-distribution Autonomous Driving VQA from Peru (https://arxiv.org/abs/2503.07587)
Comments:
          A pre-print. 26 pages. Link to Code + Data: this https URL

- **What's New**: 이 논문에서는 로부스토 데이터셋(Robusto-1 dataset)을 소개합니다. 이 데이터셋은 페루의 대시캠 비디오 데이터로 구성되어 있으며, 페루는 세계에서 운전이 가장 난폭한 나라 중 하나로 알려져 있습니다. 본 연구는 자율주행차가 어떻게 비표준 상황에서 사람과 유사하게 반응하는지를 연구하고자 합니다.

- **Technical Details**: 본 연구는 비주얼 질문 응답(Visual Question Answering, VQA) 방법을 사용하여 비즈니스의 인지적 수준에서 기초 비주얼 언어 모델(Foundational Visual Language Models, VLMs)과 인간을 비교합니다. 이에 대한 분석은 대표 유사도 분석(Representational Similarity Analysis, RSA)이라는 시스템 신경과학에서 유래한 방법을 통해 수행됩니다.

- **Performance Highlights**: 초기 분석 결과에 따르면, VLMs와 인간의 인지적 정렬은 질문의 종류에 따라 크게 달라집니다. 이는 자율주행차가 다양한 비표준 상황에서 얼마나 잘 일반화할 수 있는지를 평가하기 위한 중요한 지표로 작용할 수 있습니다. 이 데이터셋과 평가 프레임워크는 앞으로 자율주행차의 성능 평가에 유용할 것으로 기대됩니다.



### Denoising Score Distillation: From Noisy Diffusion Pretraining to One-Step High-Quality Generation (https://arxiv.org/abs/2503.07578)
Comments:
          First Author and Second Author contributed equally to this work. The last two authors equally advised this work

- **What's New**: 이번 논문에서는 변수로 오염된 데이터에서 고품질 생성 모델을 훈련하는 새로운 기법인 Denoising Score Distillation(DSD)을 도입했습니다. DSD는 오로지 노이즈가 섞인 샘플로 확산 모델을 사전 훈련한 후, 이를 단일 단계 생성기로 정제된 출력을 생성할 수 있도록 변형합니다. 이 접근법은 고품질 훈련 데이터의 부족 문제를 해결하고 과학적 분야의 여러 응용에서 큰 잠재력을 보여줍니다.

- **Technical Details**: DSD는 확산 모델(training diffusion model)의 사전 훈련을 통해 소음 있는 데이터에서 생성 모델을 더욱 효과적으로 학습할 수 있도록 합니다. 이 과정에서 점진적인 노이즈 주입(forward process)과 노이즈 제거(reverse process) 기술을 활용하여 원본 데이터의 분포를 회수합니다. 특별히, DSD는 선형 모델 설정에서 노이즈 분포 점수에 대해 정렬함으로써 깨끗한 데이터 분포의 고유값 공간을 식별하는 메커니즘을 통해 생성 모델 개선을 달성합니다.

- **Performance Highlights**: DSD는 다양한 노이즈 수준 및 데이터셋에서 생성 성능을 일관되게 향상시키는 결과를 나타냈습니다. 특히 기존의 기대와는 다르게, DSD는 저품질 교사 모델을 통해 노이즈가 섞인 데이터에서도 샘플 품질을 크게 개선할 수 있다는 것을 보여줍니다. 이러한 성과는 실질적인 경험적 증거로 뒷받침되며, 이로 인해 노이즈 데이터의 가치에 대한 새로운 인식을 제공합니다.



### Optimizing Test-Time Compute via Meta Reinforcement Fine-Tuning (https://arxiv.org/abs/2503.07572)
- **What's New**: 이 논문은 LLM의 reasoning (추론) 성능을 개선하기 위한 테스트 시간 컴퓨팅(test-time compute)의 최적화 문제를 메타 강화 학습(meta-reinforcement learning) 관점에서 다루고 있습니다. 기존 방법들이 주로 0/1 결과 보상을 사용하는 RL(reinforcement learning) 방식으로 이루어졌다면, 이 연구에서는 누적 후회(cumulative regret) 개념을 도입해 성과를 측정하고자 합니다. 이로 인해 LLM의 출력 스트림을 여러 에피소드로 나누어 분석하는 새로운 접근법이 가능하게 됩니다.

- **Technical Details**: 논문에서는 테스트 시간 컴퓨팅을 최적화하기 위해 메타 강화 학습을 수립하고, 이를 통해 LLM의 출력 토큰의 누적 후회를 최소화하는 방법을 제안합니다. 저자들은 에피소드의 맥락에서 오류 검출과 전략적 백트래킹을 구현할 수 있는 구조화된 접근법인 '백트래킹 검색(backtracking search)'을 탐구합니다. 이 설정에서는 해답을 생성하고, 이전 시도의 오류를 검출한 후 적절한 단계로 되돌아가는 과정을 통해 모델의 오류 감지 능력을 향상시킵니다.

- **Performance Highlights**: Meta Reinforcement Fine-Tuning (MRT) 기법을 도입함으로써, 실험 결과 LLM은 결과 보상을 사용하는 RL 기술 대비 2-3배의 성능 향상을 달성하게 됩니다. 또한, MRT는 수학적 추론에서 약 1.5배의 토큰 효율성을 개선한 것으로 나타났습니다. 전반적으로 MRT는 알고리즘의 진행(progress)을 증대시키는 효능이 있음을 보여주며, 에피소드 간의 추론과 오류 교정을 통해 모델의 학습 효율성을 높이는 데 기여하고 있습니다.



### Runtime Detection of Adversarial Attacks in AI Accelerators Using Performance Counters (https://arxiv.org/abs/2503.07568)
Comments:
          7 pages, 8 figures

- **What's New**: AI 기술의 빠른 채택에 따라 AI 응용 프로그램의 기밀성과 무결성을 위협하는 적대적 변조(adversarial perturbation)와 같은 보안 문제가 대두되고 있습니다. 이를 해결하기 위해 제안된 SAMURAI 프레임워크는 AI 하드웨어의 악용 방지 및 공격에 대한 복원력을 강화하는 혁신적인 접근 방식을 제시합니다. 이 프레임워크는 AI Performance Counter(APC)와 머신러닝 분석 엔진인 TANTO를 통합하여 AI 모델의 동적 동작을 기록하고 보안을 강화합니다.

- **Technical Details**: SAMURAI는 여러 보안 위협으로부터 AI 하드웨어를 보호하기 위해 멀티 레이어 저비용 보안 접근 방식을 적용합니다. APC는 AI 작업의 하드웨어 이벤트를 기록하고, TANTO는 이 데이터를 실시간으로 분석하여 악용 가능성을 탐지합니다. 이 프레임워크는 온라인 데이터 전송 없이 온디바이스(on-device)에서 학습과 추론을 수행하여 데이터의 무결성과 보안을 유지합니다.

- **Performance Highlights**: SAMURAI는 다양한 AI 모델에서 최대 97%의 정확도로 적대적 공격을 탐지하며, 전통적인 소프트웨어 기반 방법론을 크게 초월하는 성능을 보여줍니다. 이를 통해 AI 가속기의 보안 및 규제 준수를 향상시키고, 범용 AI 애플리케이션에서의 적절한 사용을 보장합니다. 실험 결과는 이 프레임워크가 상당한 성능 저하 없이 지속적이고 적응적인 모니터링을 지원함을 입증합니다.



### Inductive Moment Matching (https://arxiv.org/abs/2503.07565)
- **What's New**: 이 논문에서는 Inductive Moment Matching (IMM)이라는 새로운 생성 모델 클래스를 제안하여, 빠른 샘플링을 가능하게 하면서도 안정성을 유지합니다. 기존의 확산 모델(diffusion models)이나 Flow Matching과 비교했을 때, IMM은 단일 단계 학습 절차를 통해 더 짧은 단계로 샘플을 생성할 수 있습니다. 이 모델은 사전 훈련(pre-training)이나 두 개의 네트워크 최적화를 요구하지 않는 점에서 이전 방법들과 차별화됩니다.

- **Technical Details**: IMM 방식은 Consistency Models와 달리 분포 수준의 수렴(distribution-level convergence)을 보장합니다. 이 방법은 다양한 하이퍼파라미터(hyperparameters)와 표준 모델 아키텍처(architecture)에서도 안정적으로 작동합니다. 결과적으로 IMM은 기존의 모델들보다 더 빠른 샘플링을 지원하면서도 높은 품질의 이미지를 생성하는 것을 목표로 합니다.

- **Performance Highlights**: 원 논문에서 제시한 결과에 따르면 IMM은 ImageNet-256x256에서 1.99 FID를 달성하며, 단 8번의 추론(inference) 단계로 고품질 이미지를 생성할 수 있습니다. 또한 CIFAR-10 데이터셋에서는 2단계 FID 1.98을 기록하여, 처음부터 훈련된 모델이지만 최신 기술(state-of-the-art)로 평가받고 있습니다.



### KSOD: Knowledge Supplement for LLMs On Demand (https://arxiv.org/abs/2503.07550)
- **What's New**: 이 논문에서는 KSOD(Knowledge Supplement for LLMs On Demand)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 LLMs의 지식 기반으로 슈퍼바이즈드 파인 튜닝(Supervised Fine-Tuning, SFT)을 통해 LLM의 성능을 개선합니다. KSOD는 LLM의 오류 원인을 지식 부족 관점에서 분석하고, 부족한 지식을 바탕으로 LLM을 보완하여 성능을 향상시키는 데 중점을 둡니다.

- **Technical Details**: KSOD는 세 가지 주요 단계를 포함합니다: Knowledge Identification(지식 식별), Knowledge Verification(지식 검증), Knowledge Supplement(지식 보충). 이 과정에서 KSOD는 기존 데이터셋에서 지식을 추출하고, 이를 통해 LLM의 오류를 유발할 수 있는 지식을 확인한 후, 해당 지식을 LLM에 보충하여 성능을 개선합니다. 알고리즘을 통해 LLM이 특정 지식을 결여하고 있는지를 검증하는 방법론을 제시하고 있습니다.

- **Performance Highlights**: KSOD를 통한 실험 결과는 LLM의 오류를 억제하고 필요한 지식을 보충함으로써 4개의 일반 작업에서 성능 변화를 최소화하면서도 개선을 보여주었습니다. 특히, 특정 지식을 보완했을 때 LLM의 다른 작업에 대한 성능 저하가 거의 없거나 소폭 변화가 있었으며, 일부 경우에는 성능이 개선되기도 했습니다. 이러한 연구 결과는 지식을 보충하여 LLM의 성능을 향상시킬 수 있는 잠재력을 보여줍니다.



### Geometric Retargeting: A Principled, Ultrafast Neural Hand Retargeting Algorithm (https://arxiv.org/abs/2503.07541)
Comments:
          Project Website: this https URL

- **What's New**: 본 논문에서는 Teleoperation을 위해 개발된 Geometric Retargeting (GeoRT)이라는 새로운 신경망 손 리타게팅 알고리즘을 소개합니다. GeoRT는 인간의 손가락 키포인트를 로봇 손 키포인트로 변환하며, 1KHz의 속도와 뛰어난 정확성을 달성합니다. 이 알고리즘은 수동 주석 없이도 훈련이 가능하여, 테스팅 시간 동안의 복잡한 최적화 없이도 효율적인 솔루션을 제공합니다.

- **Technical Details**: GeoRT는 고유한 기하학적 목적 함수를 기반으로 하여, 리타게팅의 본질인 동작 충실도를 보존하고, 로봇 손의 구성 공간(C-space) 커버리지를 극대화하며, 일관된 반응성을 유지하고, 핀치 대응관계를 보존하며, 자기 충돌을 방지합니다. 이 알고리즘은 기존의 복잡한 하이퍼파라미터에 의존하지 않으며, 높은 속도와 품질을 결합합니다. 또한 실험을 통해 GeoRT가 기존 방법들보다 더 뛰어난 손 활용도를 보여줍니다.

- **Performance Highlights**: GeoRT는 기존의 리타게팅 알고리즘에 비해 손쉽고 빠른 훈련 시스템을 제공합니다. 실험 결과, 이 알고리즘은 실제 환경에서의 잡기 작업을 포함한 Teleoperation 작업에서 기존 방법들을 초월하는 성능을 발휘하였습니다. GeoRT는 효율적이고 직관적인 로봇 핸드 제어를 가능하게 하며, DexterityGen과 같은 추가 응용 프로그램을 지원합니다.



### LMM-R1: Empowering 3B LMMs with Strong Reasoning Abilities Through Two-Stage Rule-Based RL (https://arxiv.org/abs/2503.07536)
- **What's New**: 이 논문에서는 대규모 다중 모달 모델(Large Multimodal Models, LMMs)에서 추론(Reasoning)을 향상시키기 위한 새로운 접근 방식인 	extbf{Foundational Reasoning Enhancement (FRE)}와 	extbf{Multimodal Generalization Training (MGT)}를 제안합니다. 기존의 텍스트 전용 분야에서는 강력한 성과를 보여주었지만, 다중 모달 환경에서는 한계가 있었습니다. 이 연구는 이러한 한계를 극복하기 위해 텍스트 데이터로부터 추론 능력을 강화한 후, 이를 다중 모달 환경에 일반화하는 이중 단계 프레임워크를 도입합니다.

- **Technical Details**: 제안된 	extbf{	extit{method}}는 먼저 FRE 단계에서 규칙 기반 강화 학습(Rule-based Reinforcement Learning, RL)을 통해 텍스트 전용 데이터의 추론 능력을 향상시킵니다. 그 다음, MGT 단계에서 이러한 향상된 능력을 다중 모달 도메인으로 일반화하여 적용합니다. 이 방식을 통해 데이터의 부족과 복잡한 추론 예시의 부족 문제를 해결하고, 다중 모달 예비 학습으로 인한 기초적 추론 능력의 저하를 완화합니다.

- **Performance Highlights**: 실험 결과 	extbf{	extit{method}}는 Qwen2.5-VL-Instruct-3B 모델에서 다중 모달 벤치마크에서 4.83% 및 텍스트 전용 벤치마크에서 4.5%의 평균 개선률을 보였으며, 복잡한 축구 게임(Football Game) 과제에서는 3.63%의 성과 향상을 기록했습니다. 이러한 결과는 텍스트 기반의 추론 강화가 효과적인 다중 모달 일반화를 가능하게 함을 입증하며, 고품질의 다중 모달 훈련 데이터에 대한 필요성을 줄이는 데이터 효율적인 패러다임을 제공합니다.



### TokenButler: Token Importance is Predictab (https://arxiv.org/abs/2503.07518)
- **What's New**: 이번 논문에서는 대형 언어 모델(Large Language Models, LLMs)의 Key-Value (KV) 캐시의 메모리 병목을 해결하기 위해 "TokenButler"라는 새로운 토큰 중요도 예측기를 도입하였습니다. 기존 모델들은 고정된 규칙에 따라 토큰을 배제하거나 전체 KV 캐시를 유지관리하였으나, TokenButler는 쿼리에 의존한 동적 예측을 통해 중요한 토큰을 선택합니다. 이로 인해 기존 방법보다 8% 이상 향상된 정확도를 보여줍니다.

- **Technical Details**: TokenButler는 경량 예측기로, 1.2% 이하의 파라미터 오버헤드를 가지며 중요한 토큰의 문맥적 중요도를 기반으로 우선순위를 매깁니다. 이는 정밀한 토큰 중요도 추정을 가능하게 하여, 모델의 perplexity와 하위 정확도를 개선합니다. 기존의 KV 캐시 문제를 해결하기 위해 작은 컨텍스트(<512 토큰)에서 수행된 실험을 통해, 자기참조와 관련된 정밀도가 높음을 입증합니다.

- **Performance Highlights**: TokenButler는 기존의 토큰 희소성 메트릭을 초과하여, 8% 이상의 향상을 가져오는 경량화된 예측기를 통해 중요한 토큰을 식별합니다. 이러한 성능은 TokenButler가 새로운 합성 작은 컨텍스트 자기참조 검색 작업에서 거의 오라클에 가까운 정확도를 달성했다는 점에서 더욱 두드러집니다. 코드 및 모델은 논문에서 제공되는 새 URL을 통해 확인할 수 있습니다.



### Language Models Fail to Introspect About Their Knowledge of Languag (https://arxiv.org/abs/2503.07513)
- **What's New**: 최근 대규모 언어 모델(LLMs)의 내부 상태를 반추할 수 있는 능력에 대한 관심이 증가하고 있습니다. 이러한 능력은 LLM의 해석 가능성을 높이고, 표준적인 반추적 방법을 통해 모델의 문법적 지식을 평가하는 데 유용할 수 있습니다. 이 연구는 21개의 오픈소스 LLM을 체계적으로 조사하며, 문법 지식과 단어 예측이라는 두 가지 분야에 초점을 맞춥니다. 특히, 모델의 내부 언어 지식은 문자열 확률의 직접 측정을 기반으로 이론적으로 구체화됩니다.

- **Technical Details**: 우리는 두 가지 방법을 사용하여 각 모델을 평가합니다. Direct 방법은 문자열에 할당된 로그 확률을 비교하여 모델의 '진정한' 지식 상태를 이론적으로 나타내고, Meta 방법은 반추적 프롬프트에 대한 반응에 할당된 로그 확률을 비교합니다. 이 두 가지 방법의 일치를 통해, 모델이 반추할 수 있다면 메탈링귀스틱 반응이 자신의 내부 확률에 보다 충실할 것으로 예상합니다. 연구 결과는 반추의 여부를 평가하는 데 있어 현재의 연구 방법론의 한계를 극복하려는 시도를 다룹니다.

- **Performance Highlights**: 실험 결과, 테스트된 LLM에서 반추에 대한 강력한 증거를 발견하지 않았습니다. 오히려 서로 유사한 모델은 메탈링귀스틱과 직접 측정된 행동 간의 강한 상관관계를 보였습니다. 즉, 모델의 메탈링귀스틱 반응이 문자열 확률에 대한 지식과 별개로 존재한다는 결론에 도달했습니다. 이러한 발견은 최근의 모델 반추에 대한 주장에 복잡성을 더하며, 모델의 언어 능력을 연구하는 데 중요한 시사점을 제공합니다.



### Interference-Aware Super-Constellation Design for NOMA (https://arxiv.org/abs/2503.07509)
Comments:
          Accepted for publication at IEEE International Conference on Communications (ICC), 2025

- **What's New**: 최근 비정형 다중 접근(Non-orthogonal multiple access, NOMA)이 차세대 다중 접근 기술로 주목받고 있습니다. 이 논문에서는 NOMA의 구현에서 발생하는 사용자 간 간섭(inter-user interference) 문제를 해결하기 위해 오토인코더(autoencoder)를 활용하여 간섭 인식(super-constellations)을 설계합니다. 전통적인 방식과 달리, 제안된 오토인코더 기반 NOMA(AE-NOMA)는 사용자의 채널 이득에 관계없이 수신자가 구분 가능한 기호를 가진 슈퍼 별자리를 생성하도록 훈련됩니다.

- **Technical Details**: AE-NOMA는 전통적인 수신기 설계를 넘어서서, 간섭을 제거하기 위해 순차 간섭 제거(SIC)를 요구하지 않습니다. 이는 자원의 제약이 있는 디바이스에서도 기능할 수 있도록 하며, 각 NOMA 사용자에게 할당된 전력의 비율에 따라 기호가 겹치지 않도록 설계됩니다. 시스템 모델에서는 약한 사용자와 강한 사용자 간의 채널 조건을 고려하여 복잡한 가우시안 신호를 사용해 전송 메시지를 동시에 전송합니다.

- **Performance Highlights**: 테스트 결과, AE-NOMA에서 구현된 간섭 인식 별자리가 비트 오류율(Bit Error Rate, BER)을 유의미하게 개선하는 것으로 나타났습니다. 연구 결과, AE-NOMA는 전통적인 NOMA 방법론에 비해 모든 신호 대 잡음 비율(Signal-to-Noise Ratio, SNR) 범위에서 동시에 높은 BER을 달성하면서도, SIC를 사용하지 않고도 효과적인 성능을 보여줍니다. 이로써 AE-NOMA는 다양한 채널 시나리오에 적응 가능한 잠재력을 지니고 있습니다.



### From Centralized to Decentralized Federated Learning: Theoretical Insights, Privacy Preservation, and Robustness Challenges (https://arxiv.org/abs/2503.07505)
- **What's New**: 이 논문은 Federated Learning (FL)에서의 중앙집중형(centralized)과 분산형(decentralized) 시스템의 근본적인 차이는 네트워크 토폴로지(network topology)뿐만 아니라 기본 훈련 프로토콜(training protocol)에서 발생한다고 주장합니다. 논문에서는 이러한 프로토콜 차이가 모델 유용성(model utility), 개인 정보 보호 및 공격에 대한 강건성(robustness)에서 중요한 차이를 야기한다고 설명합니다. 또한 기존 연구들을 체계적으로 리뷰하고, 각 연구의 분류를 통해 FL 분야에서 새롭게 탐구해야 할 방향성을 제시합니다.

- **Technical Details**: FL의 두 가지 주요 아키텍처인 중앙집중형 FL(CFL)과 분산형 FL(DFL)에 대한 개념적 기반을 마련합니다. CFL에서는 중앙 서버가 모든 클라이언트의 업데이트를 수집하고, 이를 집계하여 전역 모델(global model)을 생성합니다. 반면 DFL은 중앙 서버가 없이, 클라이언트들이 피어 투 피어(peer-to-peer) 방식으로 협력하여 모델 정보를 교환합니다. 논문에서는 이러한 시스템들이 서로 다른 집계 프로토콜(aggregation protocol)을 사용하는 점을 강조하며, 이를 별도의 집계(separate aggregation)와 공동 최적화(joint optimization)로 분류합니다.

- **Performance Highlights**: 이 연구는 FL의 개인 정보 보호와 강건성에 대한 기존의 연구 결과를 체계적으로 정리합니다. 특히, 분산 최적화(distributed optimization)를 적용한 DFL 접근이 상대적으로 탐구되지 않았음을 지적하며, 이러한 분야에서의 연구가 더욱 활발해져야 함을 강조합니다. CFL과 DFL의 성능을 비교하고, 다양한 공격에 대한 저항성을 평가하는 방식도 제공하여 FL 시스템의 효율성과 안전성 강화를 위한 방향성을 제시합니다.



### V2Flow: Unifying Visual Tokenization and Large Language Model Vocabularies for Autoregressive Image Generation (https://arxiv.org/abs/2503.07493)
Comments:
          11 pages, 6 figures

- **What's New**: V2Flow는 높은 충실도로 시각적 토큰을 생성하며, 대형 언어 모델(LLMs)의 어휘 공간과의 구조적 및 잠재적 분포 정렬을 보장하는 새로운 토크나이저입니다. 이 방법은 자연 언어 처리에서의 새로운 패러다임을 활용하여 비주얼 생성에 대한 자율 회귀 모델링을 가능하게 만들었습니다. V2Flow는 시각적 토큰화 과정을 흐름 일치 문제로 공식화하여, LLM 어휘 공간에 내장된 토큰 시퀀스에 따라 계속되는 이미지 분포를 학습합니다.

- **Technical Details**: V2Flow의 주요 설계는 두 가지로 나뉩니다: 첫째로, 비주얼 어휘 재샘플러를 제안합니다. 이는 시각 데이터를 압축하여 LLM의 어휘에 대한 부드러운 범주형 분포로 표현되는 간결한 토큰 시퀀스로 변환합니다. 둘째로, 마스크된 자율 회귀 정류 흐름(RECTIFIED FLOW) 디코더를 제시하며, 맥락적으로 풍부한 임베딩을 생성하기 위해 마스크된 변환기 인코더-디코더 구조를 채택합니다.

- **Performance Highlights**: 상 extensive 실험을 통해 V2Flow는 주류 VQ 기반 토크나이저보다 뛰어난 성능을 보여주며, 기존 LLM 위에서 자율 회귀 비주얼 생성을 용이하게 합니다. 전반적으로, V2Flow는 LLMs의 어휘에 원활하게 통합되며 고성능 이미지 재구성을 달성합니다. 이 토크나이저는 자율적인 시각 생성의 효율성과 효과성을 크게 향상시키는 것을 목표로 하고 있습니다.



### Efficient Membership Inference Attacks by Bayesian Neural Network (https://arxiv.org/abs/2503.07482)
Comments:
          8 pages, under review

- **What's New**: 이 연구에서는 Membership Inference Attack (MIA)의 새로운 접근 방식인 Bayesian Membership Inference Attack (BMIA)를 제안합니다. BMIA는 조건부 공격을 Bayesian inference를 통해 수행하며, Laplace approximation을 통해 훈련된 참고 모델을 Bayesian 신경망으로 변환하여 조건부 점수 분포를 직접 추정할 수 있습니다. 이 방법은 단일 참조 모델만을 사용하여 효과적이면서도 강력한 MIA를 가능하게 합니다.

- **Technical Details**: BMIA는 Bayesian Neural Network (BNN)를 사용하여 조건 분포를 추정하며, 기본 아이디어는 하나의 참조 모델을 훈련한 뒤 Laplace approximation을 활용해 모델 파라미터의 사후 분포를 추정하는 것입니다. BNN은 네트워크 가중치를 확률 분포로 모델링하여 epistemic (모델 관련) 불확실성과 aleatoric (데이터 관련) 불확실성을 모두 포착할 수 있게 합니다. 이러한 접근 방식은 단일 모델을 통해 효율성과 정밀한 추정을 가능하게 합니다.

- **Performance Highlights**: BMIA는 Texas100, Purchase100, CIFAR-10/100, ImageNet의 두 개의 표 형 데이터셋과 세 개의 이미지 데이터셋에서 실험을 진행하여 그 효과성을 검증했습니다. 방법은 낮은 FPR (false positive rate)에서 이전 조건부 공격 방법에 비해 더 높은 TPR (true positive rate)을 달성하였고, 예를 들어 CIFAR-10에서 false positive rate 1%에서 37.5%의 true positive rate을 달성하며, 이전의 최첨단 방법에 비해 64% 향상된 성과를 보였습니다.



### Advancing Vietnamese Information Retrieval with Learning Objective and Benchmark (https://arxiv.org/abs/2503.07470)
- **What's New**: 이 연구에서는 베트남어 정보 검색 (Information Retrieval, IR) 작업을 위한 새로운 벤치마크인 베트남어 컨텍스트 검색(VCS)을 도입하고 있습니다. 이 벤치마크는 기존 베트남어 데이터셋을 수정하여 구성되었으며, 검색(retrieval) 및 리랭킹(reranking) 작업에 대한 평가를 중점적으로 다룹니다. 또한, 기존의 InfoNCE 손실 함수에서 개선된 새로운 학습 목표 함수를 제시하여 베트남어 임베딩 모델의 성능을 향상시키고자 합니다.

- **Technical Details**: 본 연구는 새로운 벤치마크를 구축하기 위해 기존 베트남어 데이터셋을 활용하며, 이는 텍스트 임베딩 모델이 관련 문서를 정확히 검색할 수 있도록 돕습니다. 본 벤치마크는 검색 및 리랭킹 작업에서 베트남어 언어 모델의 성능을 종합적으로 평가하기 위해 고안되었습니다. 또한, 정보 검색 작업에서 성능을 향상시키기 위해 하이퍼파라미터인 온도(temperature)의 영향을 분석합니다.

- **Performance Highlights**: 베트남어 컨텍스트 검색(VCS) 벤치마크를 통해 여러 베트남어 임베딩 모델의 성능을 비교 평가하며, 각각의 모델이 실제 문서 검색에 얼마나 효과적인지를 점검합니다. 연구 결과, 새로운 학습 목표 함수가 기존 모델보다 뛰어난 성능을 보이며, 임베딩 모델 성능의 다양한 변화를 직접 확인할 수 있습니다. 이러한 결과는 베트남어 자연어 처리(NLP) 연구의 발전을 촉진할 것으로 기대됩니다.



### MedAgentsBench: Benchmarking Thinking Models and Agent Frameworks for Complex Medical Reasoning (https://arxiv.org/abs/2503.07459)
- **What's New**: MedAgentsBench는 기존 의학 질의응답 벤치마크에 비해 다단계 클리닉 추론, 진단 수립 및 치료 계획 수립을 필요로 하는 도전적인 의학 질문에 초점을 맞춘 새로운 벤치마크입니다. 이 벤치마크는 기존 평가의 세 가지 주요 한계를 해결하며, 다양한 기존 의학 데이터셋에서 수집된 데이터를 기반으로 합니다. 또한, 최신 모델 DeepSeek R1과 OpenAI o3가 복잡한 의학적 추론 작업에서 뛰어난 성능을 발휘하는 것이 입증되었습니다.

- **Technical Details**: MedAgentsBench는 8개의 잘-known 의료 데이터셋에서 수집된 데이터를 기반으로 설정된 기준으로, 복잡한 추론 시나리오에 초점을 맞춘 질문들을 포함합니다. 이 벤치마크는 질문에 대해 모델의 정답 비율을 분석하여, 50% 이하의 성공률을 보이는 질문을 '어려운 질문'으로 분류하여 현재 모델들이 어려움을 겪고 있는 질문을 선정합니다. 이를 통해 세밀한 성능 평가의 기준을 제공합니다.

- **Performance Highlights**: 실험 결과, DeepSeek R1 및 OpenAI o3와 같은 고급 모델들이 기존 방법론에 비해 15-25% 높은 정확도를 보였습니다. 또한 검색 기반 에이전트 방식은 전통적인 접근법보다 더 우수한 비용-성능 비율을 보이며, 오픈 소스 모델이 실용적인 운영 비용으로 경쟁력 있는 결과를 달성할 수 있는 가능성을 보여줍니다. 이는 복잡한 의학적 질문에 대한 모델 간의 성능 격차를 드러내고 최적 모델 선택을 위한 인사이트를 제공합니다.



### Is a Good Foundation Necessary for Efficient Reinforcement Learning? The Computational Role of the Base Model in Exploration (https://arxiv.org/abs/2503.07453)
- **What's New**: 이 논문은 언어 모델의 효율적인 탐색을 위한 새로운 계산 프레임워크를 소개합니다. 기존의 알고리즘 설계 원리에 대한 이해가 부족한 가운데, 이 연구는 강력한 사전 훈련된 생성 모델을 활용하여 탐색의 효율성을 개선하는 방법을 제시합니다. 특히, SpannerSampling라는 새로운 알고리즘을 도입하여 데이터 효율성을 최적화하고 탐색의 효과적인 검색 공간을 줄입니다.

- **Technical Details**: 저자는 선형 소프트맥스 모델 파라미터화를 중심으로 효율적인 탐색의 계산-통계적 트레이드오프를 밝혀냅니다. 주요 발견은 커버리지의 필요성, 추론 시간 탐색, 훈련 시간 개입의 한계 및 다중 턴 탐색의 계산적 이점에 관한 것입니다. 커버리지는 모델이 최적 응답을 얼마나 잘 포함하고 있는지를 의미하며, 이는 알고리즘의 런타임을 하한으로 제한합니다.

- **Performance Highlights**: SpannerSampling 알고리즘은 사전 훈련된 모델이 충분한 커버리지를 가질 때 최적의 데이터 효율성과 계산적 효율성을 달성합니다. 이 연구는 효과적인 다중 턴 탐색을 통해 런타임 개선을 보여주며, 이는 시퀀스 수준의 커버리지를 토큰 수준으로 교체하여 가능합니다. 이러한 결과는 언어 모델의 능력을 최대화하는 데 기여할 것으로 보입니다.



### Divide and Conquer Self-Supervised Learning for High-Content Imaging (https://arxiv.org/abs/2503.07444)
- **What's New**: 이 논문에서는 기존의 self-supervised representation learning (SSL) 방법에서 발생하는 한계를 극복하기 위해 Split Component Embedding Registration (SpliCER)이라는 새로운 구조를 소개하고 있습니다. SpliCER는 이미지를 여러 섹션으로 나누어 각 섹션에서 정보를 증류하여 모델이 더욱 섬세하고 복잡한 특징을 학습하도록 돕습니다. 이는 의료 및 지리공간 이미징과 같은 최신 기술 환경에서 효과적으로 그 성능을 발휘합니다.

- **Technical Details**: SpliCER는 이미지의 구성 요소를 분해하여 각 부분에서 특징을 학습하도록 유도하는 구조로, 현재의 self-supervised loss function과 호환되며 기존 작업에 쉽게 통합될 수 있습니다. SpliCER를 사용하면 모델이 훈련 중 각 부분에서 복잡한 특징을 잊지 않고 학습 할 수 있게 됩니다. 이를 통해 고차원 이미지의 섬세한 특징을 놓치지 않고 촘촘한 정보 맵을 생성합니다.

- **Performance Highlights**: SpliCER는 실제 의료 및 지리공간 이미징 분야에서 우수한 성능을 입증했습니다. 복잡한 특징을 학습할 수 있는 능력을 통해 모델의 다운스트림 성능 향상에 기여하며, 기존의 self-supervised 방법을 통해 발생하는 간략화된 솔루션(predictive shortcuts)을 극복할 수 있습니다. 이로 인해 과학적 발견과 분석에 필요한 중요한 정보들을 학습하는 데 효과적인 도구가 되고 있습니다.



### RePO: ReLU-based Preference Optimization (https://arxiv.org/abs/2503.07426)
- **What's New**: 이번 논문에서는 대규모 언어 모델(LLM)과 인간의 선호를 일치시키기 위한 새로운 방법인 ReLU 기반 선호 최적화(RePO)를 제안합니다. 기존의 방법인 DPO와 SimPO는 각각 단일 및 이중 하이퍼파라미터를 사용하여 복잡성을 증가시키는 반면, RePO는 단순화된 알고리즘으로 이 하이퍼파라미터를 제거합니다. RePO는 야상형(SimPO)의 비참조 마진을 유지하면서도, 그래디언트 분석을 통해 $eta$를 제거하고, ReLU 기반의 최대 마진 손실(max-margin loss)을 적용하여 복잡도를 줄입니다.

- **Technical Details**: RePO는 이론적으로 SimPO의 한계 사례로 설명됩니다. RePO는 SimPO의 로지스틱 가중치가 이진 임계값(thresholding)으로 축소되는 상황에서 발생하며, 이는 0-1 손실의 볼록 봉투(convex envelope)를 형성합니다. 또한, RePO는 단일 하이퍼파라미터 γ를 조정할 필요가 있으며, 데이터 필터링(Data Filtering) 과정을 통해 잘 구분된 쌍의 그래디언트를 무시하여 최적화 과정을 단순화합니다. RePO++는 SimPO의 로지스틱-로스(logistic-log loss)를 활용하여 덜 구분된 쌍의 중요성을 차별화하는 방법입니다.

- **Performance Highlights**: 복잡한 하이퍼파라미터 조정 대신 단일 하이퍼파라미터 γ으로 최적화가 가능하여, 성능은 SimPO에 근접합니다. Empirical results indicate that RePO outperforms DPO 및 SimPO에서 여러 기본 모델에 대해 우수한 성능을 보이고, 특히 AlpacaEval 2 데이터셋에서 확인되었습니다. 이러한 방식으로 RePO는 효율적인 랭귀지 모델 훈련을 위한 간단하고 효과적인 비참조 알고리즘으로 자리 잡을 것으로 기대됩니다.



### Brain Inspired Adaptive Memory Dual-Net for Few-Shot Image Classification (https://arxiv.org/abs/2503.07396)
- **What's New**: 이번 연구에서 제안된 SCAM-Net은 인간의 보완적 학습 시스템에서 얻은 영감을 바탕으로 하여, 일반화 최적화된 시스템 통합(Generalization-optimized Systems Consolidation) 메커니즘을 채택한 이중 네트워크 모델입니다. 이는 극소 샷(Few-shot) 학습 시 의의 있는 특징을 식별하는 어려움을 해결하여, 의미적(feature) 대표성을 높이고 적응형 메모리 조정 메커니즘을 도입하여 신뢰성을 더욱 개선합니다.

- **Technical Details**: SCAM-Net은 해마(Hippocampus)와 신피질(Neocortex) 이중 네트워크로 설계되어 있으며, 구조화된 정보를 통합하고 장기 기억을 통해 적응적으로 조절하는 특징이 있습니다. Neocortex 모델은 공간적(spatial) 및 의미적(semantic) 특징을 통합하여 해마 모델의 표현력을 향상시키고, 해마 모델은 네트워크 가중치의 지수 이동 평균(Exponential Moving Average, EMA)의 느린 업데이트를 통해 Neocortex 모델의 학습을 제어합니다.

- **Performance Highlights**: 광범위한 실험 결과, 제안된 SCAM-Net 모델은 4개의 벤치마크 데이터셋에서 최첨단 성능(State-of-the-art performance)을 달성하였습니다. 미니이미지넷(miniImagenet) 데이터셋에서의 적응형 메모리 조정(Ablation study) 및 시각화 결과는 모델의 우수성을 입증하고 있습니다.



### TRCE: Towards Reliable Malicious Concept Erasure in Text-to-Image Diffusion Models (https://arxiv.org/abs/2503.07389)
- **What's New**: 최근의 텍스트-이미지 생성 모델이 고해상도 이미지를 만들 수 있는 가능성을 보여주고 있지만, 이로 인해 NSFW(Not Safe For Work) 이미지와 같은 위험한 콘텐츠 생성의 우려가 커지고 있습니다. 이러한 위험을 줄이기 위한 방법으로 개념 소거(concept erasure) 기법이 연구되고 있으나, 기존 연구는 은유적 표현 또는 적대적 프롬프트와 같이 프롬프트에 암묵적으로 삽입된 공격적인 개념을 완전히 소거하는 데 어려움을 겪고 있습니다. 본 연구에서는 TRCE라는 새로운 방법을 제안하여, 이러한 개념을 효과적으로 소거하면서도 모델의 정상적인 이미지 생성 능력을 유지할 수 있도록 하였습니다.

- **Technical Details**: TRCE는 두 단계 개념 소거 전략을 사용하여 악성 개념을 신뢰성 있게 소거하고 지식 보존을 동시에 달성합니다. 첫 번째 단계에서는 프롬프트에 암묵적으로 포함된 악성 의미를 소거하는 'Textual Semantic Erasure'를 수행하며, [EoT] 임베딩을 활용하여 악성 프롬프트를 안전한 개념을 포함한 맥락적으로 유사한 프롬프트로 매핑합니다. 두 번째 단계인 'Denoising Trajectory Steering'에서는 초기의 노이즈 제거 과정에서 안전 방향으로 예측을 조정하여 악성 콘텐츠 생성을 방지합니다.

- **Performance Highlights**: TRCE는 다양한 악성 개념 소거 벤치마크에 대해 포괄적인 평가를 수행하였으며, 그 결과 악성 개념을 효과적으로 소거하면서도 모델의 원래 생성 능력을 보다 잘 보존하는 성능을 보여주었습니다. 이 방식은 다양한 프롬프트 및 다중 개념 소거 시나리오에서 검증되었으며, 실험 결과는 TRCE가 보편적으로 잘 작동하는 것을 입증하였습니다.



### Is My Text in Your AI Model? Gradient-based Membership Inference Test applied to LLMs (https://arxiv.org/abs/2503.07384)
- **What's New**: 이 연구는 LLMs에 대한 텍스트 분류 작업에서 gradient-based Membership Inference Test (gMINT)를 적용하고 연구합니다. MINT는 특정 데이터가 머신러닝 모델 훈련에 사용되었는지 확인하는 일반적인 접근 방식이며, 본 연구는 자연어 처리 분야에서의 활용에 초점을 맞추고 있습니다. 특히, 데이터 개인 정보 보호에 대한 우려를 해결하기 위해 gMINT가 데이터 샘플의 훈련 포함 여부를 확인하는 능력을 강조합니다.

- **Technical Details**: gMINT는 각 훈련 샘플이 모델의 훈련 및 최적화에 미치는 영향을 분석하기 위해 훈련 과정에서 생성된 gradient를 활용합니다. 연구에서는 250만 문장을 포함한 6개의 데이터셋과 7개의 Transformer 기반 모델을 사용하여 gMINT를 평가했습니다. 이 과정에서 gMINT의 강력함이 입증되어 데이터 크기와 모델 구조에 따라 85%에서 99% 사이의 AUC 점수를 기록했습니다.

- **Performance Highlights**: gMINT는 훈련 과정의 어떤 샘플이 데이터에 포함되었는지를 높은 정확도로 식별하는 효과적인 능력을 보여주었습니다. 실험 결과는 gMINT가 머신러닝 모델의 감사를 위한 확장 가능하고 신뢰할 수 있는 도구로서의 잠재력을 강조합니다. 이를 통해 AI/NLP 기술의 배포에서 투명성을 보장하고 민감한 데이터를 보호하며 윤리적 준수를 촉진할 수 있습니다.



### Artificial Utopia: Simulation and Intelligent Agents for a Democratised Futur (https://arxiv.org/abs/2503.07364)
- **What's New**: 본 연구는 21세기 정치 및 경제에서 나타나는 여러 가지 문제들을 해결하기 위한 베이스업(bottom-up) 민주화 노력에 대한 새로운 연구 의제를 제안합니다. 이 연구는 인공지능(Artificial Intelligence)과 컴퓨터 시뮬레이션을 활용하여 대안적인 민주적 정치 시스템과 경제 모델을 탐구하는 'Artificial Utopia'라는 개념을 소개합니다. 이 방식은 새로운 정치 아이디어와 경제 정책을 비교적 안전하게 실험할 수 있는 환경을 제공합니다.

- **Technical Details**: 기존의 민주적 과정에 대한 수학적 모델은 사회적 선택 이론(social choice theory)과 공공 선택 이론(public choice theory)을 포함하며, 이러한 이론들은 정치 시스템에 대한 이상화된 가정을 기반으로 합니다. 그러나 이러한 전통적인 모델들은 집단 결정 과정의 동적 측면을 제대로 포착하지 못하며, 복잡한 사회적 행동과 강한 이질성을 고려하는 데에는 한계가 있습니다. 따라서 본 연구는 새로운 정치 및 경제 민주주의 모델에 대한 수용 방법으로 컴퓨터 시뮬레이션을 도입하고 있습니다.

- **Performance Highlights**: 인공지능과 현대의 시뮬레이션 기법을 사용하여, 시민 집회(citizen assembly)와 민주적 기업(democratic firm)과 같은 두 가지 구체적인 제도를 통해 대안적 민주주의 시스템의 가능성을 탐구합니다. 이러한 접근 방식은 전통적인 민주주의 기관에 비해 극적인 대안을 제시하며, 더 나아가 참여적이고 심도 있는 의사결정을 촉진하는 것을 목표로 합니다. 'Artificial Utopia' 연구는 미래 정치 및 경제 체제의 새로운 비전을 제시하는 데 기여할 것으로 기대됩니다.



### The Economics of p(doom): Scenarios of Existential Risk and Economic Growth in the Age of Transformative AI (https://arxiv.org/abs/2503.07341)
- **What's New**: 최근 인공지능(AI)의 발전으로 인하여 인류의 장기적인 영향에 대한 다양한 예측이 등장하고 있습니다. 특히 변형 인공지능(TAI)의 출현 가능성에 대한 논의가 두드러지며, 이는 인간이 수행하는 모든 경제적으로 가치 있는 작업을 초월하고 노동을 완전 자동화할 수 있는 능력을 지닐 것으로 기대되고 있습니다. 이 논문에서는 TAI의 개발이 초래할 수 있는 고유한 위험과 잠재적 경제적 결과를 다루고, 자원 할당의 중요성을 강조합니다.

- **Technical Details**: 연구는 인류의 미래 경로를 특정하고, 다양한 결과와 그에 따른 사회적 복지 및 경제적 의미를 평가합니다. TAI의 도래가 인간 멸종과 같은 재앙적 결과와 관련이 있다는 점에서, 연구자들은 AI 안전(AI safety) 및 정렬(AI alignment) 연구의 필요성을 강조합니다. 이 연구는 TAI와의 오정렬로 인한 재앙적 결과를 완화하기 위한 대규모 투자 정당성을 제시합니다.

- **Performance Highlights**: 연구 결과에 따르면, AI의 재앙적 결과 가능성이 매우 낮더라도 큰 투자를 통해 이를 예방하는 것이 경제적으로 유리하다고 강조합니다. 총체적 복지 관점에서 TAI의 개발에 있어 인간 멸종 위험을 줄이는 것이 중요하며, 심지어 AI의 발전이 불가피하다고 하더라도, TAI의 설계에서 초기 단계에서부터 상충되는 목표를 배제하는 것이 필요하다고 주장합니다.



### Research and Design on Intelligent Recognition of Unordered Targets for Robots Based on Reinforcement Learning (https://arxiv.org/abs/2503.07340)
- **What's New**: 이번 연구에서는 인공지능(AI) 기반의 지능형 로봇이 복잡한 환경에서 무질서하게 분포된 목표를 정확하게 인식할 수 있는 방법을 제안합니다. 연구의 핵심은 강화 학습(reinforcement learning)을 활용하여 수집된 목표 이미지를 처리하는 새로운 방식입니다. 이 방법은 저조도 이미지(low-illumination images)와 반사 이미지(reflection images)로 분해하여 각 부분을 향상시키는 차별화된 AI 전략을 적용합니다.

- **Technical Details**: 본 방법은 수집된 목표 이미지에 대해 양측 필터링(bilateral filtering) 알고리즘을 적용하여 이미지 품질을 향상시킵니다. 이후, 저조도 이미지는 압축되고 반사 이미지는 강화를 거쳐 두 이미지를 융합하여 새로운 이미지를 생성합니다. 이러한 과정에서 강화 학습 알고리즘과 딥 러닝(deep learning)의 통합이 이루어져, 최종적으로 강화학습 모델이 향상된 목표 이미지를 학습하도록 설계되어 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 목표 이미지의 품질을 크게 향상시키고, 지능형 로봇이 무질서한 목표를 더 높은 효율성과 정확도로 인식할 수 있게 합니다. 이는 AI 로봇 분야에서의 적용 가치가 매우 높으며, 광범위한 발전 가능성을 보여줍니다.



### Temporal Triplane Transformers as Occupancy World Models (https://arxiv.org/abs/2503.07338)
- **What's New**: 최근 몇 년간 세계 모델(world models)의 발전이 두드러지며, 이는 주로 에이전트의 움직임 경로와 주변 환경의 변화를 세밀하게 학습하는 데 초점을 맞추고 있습니다. 본 논문에서는 자율 주행을 위한 새로운 4D 점유(world model) 모델인 T$^3$Former를 제안합니다. T$^3$Former는 3D 환경을 효율적으로 압축하는 컴팩트한 삼면(삼중면) 표현을 미리 학습하는 것으로 시작하여 multi-scale temporal motion features를 추출합니다.

- **Technical Details**: T$^3$Former는 트리플레인(triplane) 구조를 사용하여 3D 점유 데이터를 압축하고, 이를 통해 환경의 세밀한 동역학을 파악합니다. 이 구조는 x-y, x-z, y-z의 세 평면을 포함하며, 기존의 VQ-VAE 방식에 비해 점유 그리드의 재구성 정확도를 20% 향상시키면서 잠재 공간 크기를 34% 감소시킵니다. 제작된 모델은 Transformer 기반의 다중 스케일 접근 방식을 사용하여 점유상태의 점진적인 변화를 예측합니다.

- **Performance Highlights**: T$^3$Former는 1.44배 더 빠른 추론 속도(26 FPS)를 달성하며, 평균 IoU를 36.09로 개선하고 평균 계획 오류를 1.0 미터로 줄였습니다. 이 모델은 특히 다중 스케일 점유 상태 변화를 예측하여 환경의 동역학을 더 효과적으로 포착함으로써 성능이 개선되었습니다. 실험 결과는 T$^3$Former가 자율주행에서 최첨단 성능을 보임을 입증합니다.



### Mitigating Hallucinations in YOLO-based Object Detection Models: A Revisit to Out-of-Distribution Detection (https://arxiv.org/abs/2503.07330)
- **What's New**: 이 논문은 객체 탐지 시스템에서 동적 환경에서의 안전한 의사결정을 보장하기 위한 연구이다. 특히, 과신(overconfidence)으로 인한 허위 탐지(hallucination)를 줄이기 위한 새로운 접근 방식을 제안한다. 기존의 Out-of-distribution (OoD) 탐지 방법의 한계를 분석하고, YOLO 모델의 성능 개선을 위한 방법론을 제시하였다.

- **Technical Details**: 논문에서는 기존 OoD 검증 데이터셋의 품질 문제를 분석하고, 이를 통해 높은 허위 양성률(false positive rates)의 원인을 규명하였다. 제안된 방법론은 객체 탐지기의 결정 경계를 조정하는 방식으로, 직접적으로 객체 존재를 인식하는 경우에만 OoD 필터링이 작동하도록 한다. 또한, 새롭게 정의한 '주변 OoD(proximal OoD)' 샘플을 사용하여 자율주행 기준 BDD-100K에서 전체 허위 탐지 오류를 88% 감소시키는 결과를 얻었다.

- **Performance Highlights**: 제안된 방식은 기존의 YOLO 기반 탐지기에서 허위 탐지를 효과적으로 줄이는 것으로 입증되었으며, 실험 결과는 OoD 샘플에 대한 미세 조정(fine-tuning) 전략이 효과적이라는 것을 보여준다. 논문에서 제시한 방법은 ID 데이터와 유사한 특성을 가진 OoD 샘플을 활용하여 결정 경계를 형성하여, 보다 나은 성능을 발휘하도록 했다. 이러한 접근법은 새로운 객체나 이상 탐지 시 허위 탐지를 줄이는 데 기여할 수 있다.



### Assessing the Macro and Micro Effects of Random Seeds on Fine-Tuning Large Language Models (https://arxiv.org/abs/2503.07329)
Comments:
          7 pages, 5 tables, 3 figures

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)에서 랜덤 시드의 영향을 체계적으로 평가합니다. GLUE 및 SuperGLUE 벤치마크를 사용하여 정확도, F1 점수와 같은 전통적인 지표의 평균과 분산을 계산하여 성능 변동의 매크로 레벨 영향을 분석합니다. 또한, 예측의 일관성을 측정하는 새로운 지표를 도입하여 개별 예측의 안정성에 대한 마이크로 레벨 효과를 포착합니다.

- **Technical Details**: 랜덤 시드의 매크로 레벨 영향을 평가하기 위해 여러 시드에 대한 성능 지표의 분산을 계산합니다. 이러한 성능 지표에는 F1 점수 또는 회귀 작업을 위한 Pearson 상관 등이 포함될 수 있으며, VAR 값이 작을수록 매크로 수준의 성능 변화가 적다는 것을 의미합니다. 또한, '일관성(consistency)'을 정의하여 다양한 하이퍼파라미터 설정으로 미세 조정된 LLM 간의 예측 일관성을 측정합니다.

- **Performance Highlights**: 실험 결과, 표준 메트릭과 일관성 메트릭 모두에서 상당한 변동성이 있음을 보여주었습니다. 이는 LLM의 미세 조정 및 평가에서 랜덤 시드 기반 변동을 고려해야 할 필요성을 강조합니다. 연구의 주요 기여는 이러한 변동성을 제대로 이해하고, 보다 신뢰할 수 있으며 재현 가능한 결과를 위한 평가 기준을 설계하는 데 도움이 되는 것입니다.



### AI Biases as Asymmetries: A Review to Guide Practic (https://arxiv.org/abs/2503.07326)
Comments:
          24 pages

- **What's New**: 인공지능(AI)에서 편향(bias)에 대한 이해가 새로운 혁신을 맞이하고 있습니다. 기존에는 편향을 오류나 결함으로만 이해했으나, 이제는 AI 시스템에 필수적이며 경우에 따라 덜 편향된 대안보다 바람직하게 여겨질 수 있다는 점이 강조되고 있습니다. 이 논문은 이러한 변경된 이해의 이유를 검토하고, AI 시스템에서 편향을 측정하고 이해하는 방법에 대한 지침을 제공합니다.

- **Technical Details**: 본 논문에서는 편향을 "대칭 기준( symmetry standard)의 위반(violation)"으로 이해하는 것이 중요하다고 주장합니다. 세 가지 주요 편향 유형인 오류 편향(error biases), 불평등 편향(inequality biases), 그리고 과정 편향(process biases)을 구분하며, 각 편향 유형이 AI 개발 및 적용의 과정에서 어떤 경우에 긍정적, 부정적, 또는 불가피할 수 있는지를 강조합니다.

- **Performance Highlights**: AI 시스템의 편향을 평가하면서 어떤 종류는 수용하거나 심지어 증폭해야 하며, 어떤 종류는 최소화하거나 제거해야 하는지를 논의합니다. 논문은 이러한 편향 각각의 특성과 이를 효과적으로 관리하기 위한 새로운 접근 방식을 제시합니다.



### Dynamic Path Navigation for Motion Agents with LLM Reasoning (https://arxiv.org/abs/2503.07323)
- **What's New**: 이 논문은 대형 언어 모델(LLM)의 공간 경로 계획 및 장애물 없는 궤적 생성 능력을 탐구합니다. LLM은 사용자-에이전트 상호작용을 지원하고 복잡한 시스템에서 전역 제어를 제공하는 능력이 있어 탐색에 큰 잠재력이 있습니다. 저자들은 제로샷(Zero-shot) 탐색과 경로 생성을 연구하기 위해 새로운 데이터셋을 구축하고 평가 프로토콜을 제안하였습니다.

- **Technical Details**: 경로는 직선으로 연결된 앵커 포인트를 사용하여 표현되며, 다양한 방향으로의 이동을 가능하게 합니다. 이 접근법은 기존 방법에 비해 더 큰 유연성과 실용성을 제공하면서도 LLM에 대해 간단하고 직관적입니다. 연구 결과에 따르면, 작업이 잘 구조화되면 LLM이 장애물을 피하는 데 있어 상당한 계획 능력을 보이며, 자율적으로 생성된 동작으로 목표에 도달할 수 있습니다.

- **Performance Highlights**: 단일 LLM 모션 에이전트가 정적 환경에서 공간 추론 능력을 갖추고 있는 것 외에도, 이 능력은 동적 환경에서 다중 모션 에이전트의 조정으로 원활하게 일반화될 수 있습니다. 전통적인 접근 방식과는 달리, 우리는 훈련이 필요 없는 LLM 기반 방법을 통해 글로벌 동적 폐쇄 루프(planning) 계획 수립과 자율적인 충돌 문제 해결을 가능하게 합니다.



### Experimental Exploration: Investigating Cooperative Interaction Behavior Between Humans and Large Language Model Agents (https://arxiv.org/abs/2503.07320)
- **What's New**: 최근 인공지능(AI) 에이전트가 독립적 의사결정자로서의 역할을 수행하면서 인간과 AI 사이의 협력 역학이 중요해지고 있습니다. 특히, 대형 언어 모델(LLM)로 강화된 자율 에이전트의 역할이 경쟁-협력(interaction)을 통해 인간의 협력적 행동에 미치는 영향을 조사하였습니다. 본 연구에서는 참가자들이 다양한 특성을 지닌 LLM 에이전트와 반복적으로 Prisoner's Dilemma 게임을 진행한 결과, AI의 특성이 협력적 행동에 미치는 영향이 중요함을 발견하였습니다.

- **Technical Details**: 연구에서는 30명의 참가자가 AI 에이전트의 특성에 따라 다르게 반응하는지를 분석했습니다. 이러한 AI 에이전트들은 세 가지로 구분되며, 자율성과 인간 특성에 대한 인식이 참가자의 협력 행동에 미치는 영향을 살펴보았습니다. 특정 가설을 시험하기 위해, LLM 에이전트의 특성과 참가자의 성별이 협력 행동 패턴에 미치는 영향을 경험적으로 조사했습니다.

- **Performance Highlights**: 연구 결과에 따르면, 참가자들은 AI 에이전트가 인간 특성을 갖추었다고 인식할 때 더 높은 협력률을 보였으며, 사기당한 후에 더 많은 수용적 행동을 나타냈습니다. 성별에 따라 협력 행동 패턴이 달라졌고, LLM 에이전트에 대한 협력 기대가 더 높았지만, 기대 이하의 결과는 실망감을 초래했습니다. 이러한 통찰은 향후 인간과 AI의 협력을 위한 AI 에이전트 설계에 중요한 정보로 작용할 수 있습니다.



### Self-Corrective Task Planning by Inverse Prompting with Large Language Models (https://arxiv.org/abs/2503.07317)
Comments:
          7 pages, 5 figures, IEEE International Conference on Robotics and Automation (ICRA) 2025

- **What's New**: 이번 논문에서는 InversePrompt라는 새로운 자기 수정형 태스크 계획 접근 방식을 제안합니다. 이 방법은 역 프롬프트(inverse prompting)를 활용하여 해석 가능성을 향상시키고, 여러 추론 단계를 통합하여 명확한 피드백을 제공합니다. 역 동작(inverse actions)을 생성하고 이를 검증하여 생성된 계획의 논리적 일관성을 명시적으로 확인합니다.

- **Technical Details**: InversePrompt는 자연어로 설명된 주어진 태스크 목표를 PDDL 수식으로 변환하고, 각 동작이 로봇 스킬과 연결되도록 구성합니다. 이方法은 역 동작 및 해당 상태를 생성한 후, 이를 통해 원래 상태로 복원할 수 있는지를 검증하는 방식을 따릅니다. 이 과정은 수학적 검증 원리를 기반으로 하여 여러 단계의 추론을 통해 수행됩니다.

- **Performance Highlights**: 제안된 방법은 Ballmoving, Blocksworld, Cooking 환경을 포함한 광범위한 벤치마크 시나리오에서 기존 LLM 기반 태스크 계획 접근 방식보다 평균 16.3% 더 높은 성공률을 기록했습니다. 실제 환경에서도 비실행 가능한 계획에 대해 더 높은 성공적인 태스크 완료율을 보여주며, 기존의 자기 수정 접근 방식보다 효과적인 성과를 보였습니다.



### Group-robust Sample Reweighting for Subpopulation Shifts via Influence Functions (https://arxiv.org/abs/2503.07315)
Comments:
          Accepted to the 13th International Conference on Learning Representations (ICLR 2025). Code is available at this https URL

- **What's New**: 최근 기계 학습 모델은 데이터 배포 중 서브인구 집단(shifts) 간의 성능이 불균형한 문제를 안고 있다. 이러한 서브인구 집단의 비율 변화로 인해 모델의 일반화 능력이 저하되는 것에 대한 대안을 제시하고 있다. 본 논문에서는 Group-robust Sample Reweighting (GSR)라는 새로운 접근 방식을 제안하여 고품질의 그룹 레이블을 재사용하여 그룹 레이블의 효율성을 높이고 있다.

- **Technical Details**: GSR은 두 단계로 이루어진 방법론으로, 첫 번째 단계에서 그룹 레이블이 없는 데이터를 통해 표현을 학습하고, 두 번째 단계에서는 역량 기반(deep learning) 기술인 influence functions를 사용하여 가중치 조정을 수행한다. Pseudo-hessian를 통해 고차 미분 계수를 활용함으로써 훈련 과정을 통틀어 계산할 필요 없이 샘플 가중치 업데이트를 효율적으로 추정할 수 있다. 이 방법은 last-layer retraining (LLR)을 통해 최적화 문제가 단순히 변경될 수 있도록 지원하여 복잡성을 줄인다.

- **Performance Highlights**: GSR은 지난해의 최첨단 방법에 비해 평균적으로 1.0%의 개선된 절대 worst-group accuracy를 달성했다. 특히, 같은 양의 그룹 레이블을 사용할 때 GSR이 다른 방법들보다 더 나은 성능을 보여주며, 학습 속도가 더 빠르고 경량화된 방식으로 실질적인 개선을 제공한다. 이러한 결과는 비전 및 자연어 처리(NLP) 작업에서의 그룹 강건성(group robustness)의 향상으로 입증되었다.



### Distilling Knowledge into Quantum Vision Transformers for Biomedical Image Classification (https://arxiv.org/abs/2503.07294)
Comments:
          Submitted for MICCAI 2025

- **What's New**: 이번 연구에서는 양자 비전 변환기(QViTs)가 비전 변환기(ViTs)와 양자 신경망(QNNs)의 장점을 결합한 새로운 모델로 제안됩니다. QViTs는 기존의 선형 레이어를 QNN으로 대체하여 자기 주목 메커니즘의 기능을 향상시킵니다. 이 하이브리드 접근법은 낮은 파라미터 수로도 뛰어난 성능을 달성할 수 있는 가능성을 보여줍니다.

- **Technical Details**: QViTs는 이미지 데이터를 패치 시퀀스로 처리하고, 양자역학의 특성을 활용하여 Hilbert 공간에서 정보를 표현합니다. 이를 통해 QNN은 복잡한 패턴을 더욱 효율적으로 학습할 수 있습니다. 또한, 지식 증류(Knowledge Distillation, KD)를 통해 고품질의 고전 모델로부터 얻은 지식을 바탕으로 QViTs의 성능을 향상시키는 방법도 탐구합니다.

- **Performance Highlights**: 연구 결과, QViTs는 비교 가능한 ViTs와 대조할 때 평균 ROC AUC와 정확도에서 각각 0.017 및 0.023의 향상을 이루었습니다. 또한, QViTs는 상대적으로 적은 파라미터 수에도 불구하고, 여러 분류 작업에서 전통적인 SOTA 모델과도 경쟁력을 갖추며, GFLOPs와 파라미터 수를 각각 89% 및 99.99% 감소시키는 효율성을 보여줍니다.



### VizTrust: A Visual Analytics Tool for Capturing User Trust Dynamics in Human-AI Communication (https://arxiv.org/abs/2503.07279)
Comments:
          Accepted by ACM CHI conference 2025

- **What's New**: 이 논문은 사용자 신뢰(user trust)를 실시간으로 분석할 수 있는 VizTrust라는 시각 분석 도구를 소개합니다. 기존의 사용자 신뢰 측정 방식은 복잡한 상호작용 중에 신뢰의 변화를 포착하지 못했지만, VizTrust는 이를 해결하고자 합니다. 이 도구는 협업 시스템(multi-agent collaboration system)을 활용하여 인간-에이전트 간의 소통에서의 신뢰 동적 변화를 시각적으로 나타냅니다.

- **Technical Details**: VizTrust는 네 가지 중요한 신뢰 차원—유능성(competence), 무결성(integrity), 선의(benevolence), 예측가능성(predictability)—에 기반하여 사용자 신뢰를 평가합니다. 이 도구는 자연어 처리(NLP) 및 머신러닝 기술을 사용하여 대화 중 발생하는 세부적인 신뢰 형성 신호를 분석하고, 시계열(time series) 시각화를 통해 인간-에이전트 간의 상호 작용을 정교하게 평가합니다. 또한, VizTrust는 대화에서의 사회적 신호(social signals)와 언어적 전략을 실시간으로 제공하여 대화 설계에 유용한 통찰을 제공합니다.

- **Performance Highlights**: VizTrust의 대시보드는 사용자의 신뢰 동력학, 참여도, 정서적 톤(또는 감정적 반응), 그리고 예의 이론(theory of politeness) 등을 시각적으로 표현하여 설계 이해관계자가 신뢰의 변화 지점을 확인할 수 있도록 도와줍니다. 이 도구는 실시간으로 사용자와 대화하는 에이전트의 반응을 바탕으로 신뢰 변화를 이해하고, 사용자와의 장기적인 상호작용에서 신뢰 요소를 최적화하는 데 기여합니다. VizTrust는 대화의 각 회전(turn)에서 수집된 결과를 종합적으로 시각화하여 효율적인 에이전트 설계를 지원하는 데 중요한 역할을 합니다.



### Federated Learning in NTNs: Design, Architecture and Challenges (https://arxiv.org/abs/2503.07272)
Comments:
          Accepted in IEEE Communications Magazine

- **What's New**: 비지상 네트워크(Non-terrestrial networks, NTNs)는 향후 6G 통신 시스템의 핵심 구성 요소로 부상하고 있으며, 전 세계 연결성과 데이터 집약적 애플리케이션을 지원합니다. 본 연구에서는 고고도 플랫폼 스테이션(High Altitude Platform Station, HAPS) 별자리를 중간 분산 FL 서버로 활용한 분산 계층적 연합 학습(Distributed Hierarchical Federated Learning, HFL) 프레임워크를 제안합니다. 이 프레임워크는 저궤도(LEO) 위성과 지상 클라이언트를 FL 훈련 과정에 통합하며 전 세계 HAPS 별자리 간에 FL 글로벌 모델을 교환하기 위해 정지 궤도(GEO) 및 중간 궤도(MEO) 위성을 릴레이로 사용합니다.

- **Technical Details**: 제안된 HFL 프레임워크는 HAPS 별자리 노드를 분산 FL 서버로 활용하여 전 세계적으로 매끄러운 학습을 가능하게 합니다. 이 구조는 FL 시스템 확장성, 정확성 향상 및 지연 시간 조절을 통해 향상된 개인정보 보호를 제공합니다. 특히, 다양한 클라이언트 능력과 이질적인 네트워크 조건에 적응할 수 있도록 설계되었습니다. 이를 통해 자원 활용 메트릭(Resource Utilization Metrics)을 최적화하여 NTN 아키텍처를 개선할 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: 수치 연구 결과 제안된 HFL 프레임워크가 모델 정확성을 향상시키고, 훈련 손실을 줄이며 효율적인 지연 관리가 가능함을 입증했습니다. NTNs은 원거리 및 도심 지역에서 분산 학습을 가능하게 하며, FL의 잠재력을 극대화하기 위해 모든 세 가지 계층을 고려해야 함을 강조합니다. 본 연구는 지속 가능하고 확장 가능한 학습 환경의 중요성을 강조하며, 향후 NTN과 FL의 통합 발전 방향을 제시합니다.



### WISE: A World Knowledge-Informed Semantic Evaluation for Text-to-Image Generation (https://arxiv.org/abs/2503.07265)
Comments:
          Code, data and leaderboard: this https URL

- **What's New**: 본 논문에서는 텍스트-이미지(T2I) 모델의 평가를 위한 새로운 벤치마크인 WISE(World Knowledge-Informed Semantic Evaluation)를 제안합니다. 기존의 평가 기준들이 이미지 현실성과 단순한 텍스트-이미지 정합성에 주로 집중하는 반면, WISE는 보다 복잡한 의미 이해와 세계 지식 통합을 평가합니다. 또한 WiScore라는 새로운 정량적 메트릭을 도입하여 지식-이미지 정합성을 보다 rigorously 평가할 수 있도록 합니다.

- **Technical Details**: WISE는 자연 과학, 시공간 추론, 문화적 상식과 같은 세 가지 주요 영역을 포함하며, 25개의 서브 도메인에 걸쳐 1000개의 평가 프롬프트를 제공합니다. WiScore는 생성된 이미지 내에서 객체와 개체의 정확한 묘사를 강조하는 새로운 복합 메트릭으로, 일관성(Consistency), 현실성(Realism), 미적 품질(Aesthetic Quality)의 세 가지 주요 요소의 가중 평균으로 계산됩니다. 이러한 체계적인 접근 방식을 통해, 기존 T2I 모델들이 세계 지식을 효과적으로 통합하고 적용하는 데 제한적이라는 것을 보여줍니다.

- **Performance Highlights**: 20개의 T2I 모델(10개의 전용 모델 및 10개의 통합 멀티모달 모델)을 평가한 결과, 기존 T2I 모델들은 복잡한 의미 이해 및 세계 지식 통합에 있어 상당한 한계를 드러냈습니다. 통합 멀티모달 모델조차도 전용 T2I 모델에 비해 이미지 생성에 있어 우위가 확인되지 않았으며, 이는 현재의 통합 접근 방법이 이미지 생성에서 세계 지식을 효과적으로 활용하는 데 한계를 나타냅니다. 이러한 결과는 T2I 모델의 발전을 위한 주요 경로를 제시합니다.



### COMODO: Cross-Modal Video-to-IMU Distillation for Efficient Egocentric Human Activity Recognition (https://arxiv.org/abs/2503.07259)
- **What's New**: 본 논문에서는 COMODO라는 새로운 크로스 모달(self-supervised) 자기 지도 증류(framework)를 제안합니다. 이는 비디오 모달리티(video modality)의 풍부한 의미 지식을 IMU 모달리티로 전이하여 라벨이 없는 상태에서도 활용할 수 있는 방법입니다. COMODO는 비디오 인코더를 고정하여 동적 인스턴스 큐를 구축하고, 비디오와 IMU 임베딩 간의 피처 분포를 정렬하는 방식을 도입합니다.

- **Technical Details**: COMODO는 두 가지 모달리티의 시간 해상도와 피처 공간이 다름에도 불구하고, 효과적으로 비디오 도메인의 우수한 지식을 IMU 도메인으로 증류할 수 있는 기술을 필요로 합니다. 또한, 다양한 비디오 및 IMU 인코더 쌍에 대해 적용할 수 있도록 구조적 접근 방식을 갖추고 있음을 강조합니다. 이 방법은 자가 지도(self-supervised) 학습의 이점을 살려 데이터 효율성을 높이며, IMU 센서의 인지 성능을 향상시킵니다.

- **Performance Highlights**: COMODO는 세 가지 벤치마크 데이터세트에서 실험을 실시하여, 완전 감독된 모델보다 높은 성능을 보이는 결과를 보여줍니다. 또한, COMODO는 데이터의 부족함에도 불구하고 강한 크로스 데이터셋 일반화 성능을 유지합니다. 이러한 결과들은 고급 리소스 센서와 저급 리소스 센서 간의 간극을 해소할 수 있는 가능성을 제시합니다.



### AI-Driven Automated Tool for Abdominal CT Body Composition Analysis in Gastrointestinal Cancer Managemen (https://arxiv.org/abs/2503.07248)
- **What's New**: 이번 연구에서는 중국에서 특히 높은 발생률을 보이는 위장관 암에 대한 신속하고 효율적인 처리를 위해 복부 CT 스캔 분석을 자동으로 수행하는 AI 기반 도구를 개발하였습니다. 이 도구는 복부의 근육, 피하 지방 및 내장 지방을 식별하고 세분화할 수 있도록 설계되었습니다. 또한, 사용자가 세분화 결과를 정제할 수 있는 인터랙티브한 인터페이스를 제공합니다.

- **Technical Details**: 이 도구는 다중 관점 로컬라이제이션 모델(multi-view localization model)과 고정밀 2D nnUNet 기반 세분화 모델을 통합하여 사용합니다. 복부 영역을 정확하게 감지하기 위해 다중 관점 융합 모듈을 적용하며, 최종 세분화 결과의 Dice Score Coefficient는 0.967로 보고되었습니다. 전반적으로 이 시스템은 복부 CT 이미지의 전처리 및 특징 추출을 통해 세부적인 분석을 가능하게 합니다.

- **Performance Highlights**: AI 도구의 성능은 90%의 로컬라이제이션 정확도와 함께 높은 세분화 품질을 입증하며, 복부 조직의 정량적 매개변수를 자동으로 계산할 수 있는 기능을 제공합니다. 환자 평가 및 치료 계획 수립에 있어 임상 의사에게 유용한 메트릭스를 제공함으로써 위장관 암 관리에서의 효율성을 크게 향상시킬 것으로 기대됩니다. 이 도구는 복부 조직 분석의 표준화된 방법을 제공하여 보다 효과적인 환자 관리에 기여할 것으로 보입니다.



### LLM-C3MOD: A Human-LLM Collaborative System for Cross-Cultural Hate Speech Moderation (https://arxiv.org/abs/2503.07237)
Comments:
          Accepted to NAACL 2025 Workshop - C3NLP (Workshop on Cross-Cultural Considerations in NLP)

- **What's New**: 이 논문은 콘텐츠 조정(content moderation)의 새로운 접근 방식을 제안합니다. 특히, 많은 기술 플랫폼이 자원이 풍부한 언어에 집중하는 반면, 자원이 부족한 언어의 모더레이터는 부족하다는 문제를 다룹니다. 효과적인 조정은 문화적 맥락을 이해하는 데 의존하기 때문에, 이 불균형은 비원어민 모더레이터의 문화 이해 부족으로 인해 부적절한 조정의 위험을 증가시킵니다.

- **Technical Details**: 연구팀은 비원어민 모더레이터가 문화특화 지식(culturally-specific knowledge), 감정(sentiment), 인터넷 문화(internet culture)를 해석하는 데 어려움을 겪는다는 것을 발견했습니다. 이를 해결하기 위해 LLM-C3MOD라는 인간-LLM 협력 파이프라인을 제시하며, 세 단계로 구성됩니다: (1) RAG-enhanced cultural context annotations; (2) 초기 LLM 기반 조정; (3) LLM 합의가 없는 경우의 인간 조정(targeted human moderation).

- **Performance Highlights**: 이 시스템은 한국의 혐오 발언(hate speech) 데이터셋을 이용하여 평가되었으며, 인도네시아와 독일 참가자들이 참여했습니다. 결과적으로 78%의 정확도(기준인 GPT-4o의 71%를 초과)를 달성했으며, 인간의 작업량은 83.6% 줄어들었습니다. 특히, 인간 모더레이터는 LLM이 힘들어하는 미세한 내용(nuanced contents)에서 뛰어난 성과를 보였습니다.



### CoT-Drive: Efficient Motion Forecasting for Autonomous Driving with LLMs and Chain-of-Thought Prompting (https://arxiv.org/abs/2503.07234)
- **What's New**: 본 연구는 CoT-Drive라는 새로운 접근 방식을 제안하며, 이는 대규모 언어 모델(LLMs)과 연쇄적 사고(CoT) 프롬프트 기법을 활용하여 움직임 예측을 강화합니다. CoT-Drive는 경량 언어 모델(LMs)에 LLM의 고급 장면 이해 능력을 효과적으로 전달하기 위한 교사-학생 지식 증류 전략을 도입하여, 경량 모델이 실시간으로 작동할 수 있도록 합니다. 또한 Highway-Text 및 Urban-Text라는 두 가지 새로운 장면 설명 데이터셋을 제공하여 문맥에 맞는 의미론적 주석 생성을 위한 경량 LMs의 미세 조정을 지원합니다.

- **Technical Details**: CoT-Drive 프레임워크는 LLM의 고급 장면 이해 기능을 경량, 엣지 배포 모델에 통합하는 것을 목표로 합니다. 이 연구에서는 LLM GPT-4 Turbo가 "교사" 역할을 하여 경량 "학생" 모델로 지식을 전달하는 교사-학생 지식 증류 방법을 도입하였습니다. CoT 프롬프트 기법은 LLM의 통찰력을 인간 같은 인지 프로세스에 맞게 조정하여, AV의 정확한 예측을 가능하게 합니다.

- **Performance Highlights**: 다양한 실제 데이터셋을 기반으로 한 포괄적인 평가 결과, CoT-Drive는 기존 모델들을 능가하였으며 복잡한 교통 시나리오를 처리하는 데 있어 효과성과 효율성을 증명했습니다. 이 연구는 LLMs의 실제 적용 가능성을 고려한 첫 번째 사례로, 경량 LLM 대리 모델의 훈련 및 사용을 선도하여 새로운 기준을 설정하고 AD 시스템에 LLM을 통합할 잠재력을 보여줍니다.



### Cross-Lingual IPA Contrastive Learning for Zero-Shot NER (https://arxiv.org/abs/2503.07214)
Comments:
          17 pages, 6 figures

- **What's New**: 이 논문에서는 저자들이 CONtrastive Learning with IPA (CONLIPA) 데이터세트를 제안하여 10개 고자원 언어와 영어의 IPA 쌍을 포함하고 있습니다. 이는 유사한 발음을 가진 언어간의 포네믹 표현 과정에서 갭을 줄이고, 고자원 언어로 훈련된 모델이 저자원 언어에서 효과적으로 작동할 수 있도록 지원하는 방법론을 다룹니다. 또한, 본 연구는 Zero-Shot NER을 위한 새로운 접근 방식으로, 대규모 언어 모델(LLMs)을 활용하여 동족어 쌍을 추출해 훈련합니다.

- **Technical Details**: CONLIPA 데이터세트는 10개의 주요 언어 가족에서 추출된 고자원 언어와 영어 간의 IPA 쌍으로 구성되어 있습니다. 이 데이터세트를 사용하여 Cross-lingual IPA Contrastive learning 방법(IPAC)을 제안하며, 이는 다양한 언어의 포네믹 표현 간의 유사성을 확보하는 데 중점을 둡니다. 논문에서는 자가 감독(self-supervised) 학습 방식을 적용하고, InfoNCE 손실을 통해 유사한 데이터 포인트를 찾아내는 방법을 사용합니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식은 베이스라인과 비교하여 유의미한 성능 향상을 보여주었고, 특히 저자원 언어에서 Zero-Shot NER 작업에 있어 주요 이점을 입증했습니다. 분석된 결과는 새로운 모델이 유사한 발음을 가진 단어들 간의 표현을 효과적으로 가까이 가져오는 능력을 갖추었음을 나타냅니다. 이렇게 한의 발음 기록 방식을 활용하여 향후 저자원 언어 연구에 기여할 수 있는 가능성이 높아 보입니다.



### Discrete Gaussian Process Representations for Optimising UAV-based Precision Weed Mapping (https://arxiv.org/abs/2503.07210)
- **What's New**: 이 연구는 정밀 농업을 위한 잡초 지도 제작에서 새로운 이산화 방식의 유용성을 조사합니다. 기존의 방법론은 고정된 비행 경로에 의존하는 전통적인 orthomosaic 스티칭 기법을 사용하지만, 이 연구에서는 Gaussian Process (GP) 기반의 잡초 지도를 성능 향상을 위해 다양한 이산화 방법론으로 비교합니다. 다섯 가지 대안인 quadtrees, wedgelets, BSP 트리 및 이들 구조를 기반으로 한 새로운 방법들이 제안되었습니다.

- **Technical Details**: 이 논문은 2D GP 잡초 지도의 이산화 표현을 평가하여 실제 잡초 분포에서의 데이터 신뢰성을 향상시키고자 합니다. 각 대안 평가 기준으로는 시각적 유사성, 평균 제곱 오차 (MSE), 그리고 계산 효율성을 포함합니다. 이산화된 표현 방법이 GP의 연속적인 상태를 보존하는 데 크게 영향을 미친다는 점에서 중요한 의미를 가지며, 이를 통해 효율적인 데이터 수집 경로 계획이 가능합니다.

- **Performance Highlights**: 결과적으로, quadtrees 구조가 평균적으로 모든 지표에서 가장 뛰어난 성능을 보였고, 대규모의 잡초 패치가 존재하는 분야에서는 hexagons 또는 BSP LSE가 우수한 성능을 보였습니다. 이 연구는 사용되는 지도의 표현 방식이 잡초 분포의 특성에 따라 달라져야 하며, 특정 분포에 맞는 최적의 방법을 선택함으로써 정밀 농업의 지도 정확도와 효율성을 향상시킬 수 있음을 강조합니다.



### DeFine: A Decomposed and Fine-Grained Annotated Dataset for Long-form Article Generation (https://arxiv.org/abs/2503.07170)
- **What's New**: 이번 연구에서는 DeFine이라는 새로운 데이터셋을 소개합니다. DeFine은 Long-Form Article Generation (LFAG)을 지원하기 위해 계층적으로 분해되고 세밀하게 주석이 달린 데이터셋입니다. 이 데이터셋은 각각의 생성 프로세스를 세 가지 단계를 구분하여 일정한 논리적 일관성과 내용 구성을 보장합니다. 본 논문은 DeFine이 가진 독특한 구조적 특성과 구체적인 주석을 통해 긴 형식의 기사 작성을 향상시킬 수 있음을 밝혔다.

- **Technical Details**: DeFine 데이터셋은 데이터 채굴(Data Miner), 인용 검색(Cite Retriever), 질문-답변 주석 생성(Q&A Annotator) 및 데이터 정리(Data Cleaner)로 구성된 다중 에이전트 협업 파이프라인을 이용해 구축되었습니다. 각 에이전트는 데이터셋 생성의 특정 측면에 전문화되어 있으며, 이는 논리적 일관성과 내용 조직을 보장하는데 기여합니다. 데이터 마이너는 Wikipedia에서 구조적 요소를 추출하고, 인용 검색기는 참조된 URL을 정제하며, Q&A 주석 생성기는 맥락 인식 프롬프트를 사용합니다. 마지막으로 데이터 클리너는 데이터의 무결성을 확보합니다.

- **Performance Highlights**: 실험 결과, DeFine로 학습된 Qwen2-7b-Instruct 모델은 기존의 LFAG 방법과 비교하여 논리적 일관성, 사실 정확성 및 인용 신뢰성에서 유의미한 개선을 보였습니다. 이를 통해 DeFine이 긴 형식의 기사 생성에서 질적 향상을 가능하게 하는 효과적인 도구임을 입증하였습니다. DeFine 데이터셋은 연구자들에게 공개되어 향후 연구를 촉진할 수 있도록 도와줍니다.



### Ideas in Inference-time Scaling can Benefit Generative Pre-training Algorithms (https://arxiv.org/abs/2503.07154)
- **What's New**: 최근 기초 모델의 발전이 Generative Pre-training을 통해 눈에 띄게 증가했지만, 알고리즘 혁신은 주로 discrete signals를 위한 autoregressive 모델과 continuous signals를 위한 diffusion 모델에 정체되어 있습니다. 이 정체는 풍부한 다중 모달 데이터의 잠재력을 완전히 열지 못하게 하며, 이는 다중 모달 지능의 발전을 제한합니다. 우리는 Inference-first 접근법이 새로운 Generative Pre-training 알고리즘에 영감을 줄 수 있다고 주장합니다.

- **Technical Details**: 저자들은 Inference-time Scaling의 두 축인 sequence length와 refinement steps에 대해 논의했습니다. Autoregressive 모델은 sequence length에서 스케일링하고, diffusion 모델은 refinement steps에서 스케일링합니다. 예를 들어, Inductive Moment Matching (IMM)을 통해 이러한 이론을 실제적인 예로 보여줍니다.

- **Performance Highlights**: 모델의 denoising network 설계를 수정함으로써, 원단계 알고리즘이 안정적으로 만들어지고, 샘플 품질이 향상되며 inference 효율이 10배 이상 증가하는 결과를 도출하였습니다. 이로 인해 현재의 diffusion 모델의 한계를 극복할 수 있는 가능성이 열렸습니다. 저자들은 이러한 접근 방식이 향후 고차원 혼합 모달 데이터에 대한 효과적인 Pre-training 알고리즘을 개발하는 데 기여할 수 있기를 희망합니다.



### PTMs-TSCIL Pre-Trained Models Based Class-Incremental Learning (https://arxiv.org/abs/2503.07153)
Comments:
          13 pages,6 figures

- **What's New**: 이 논문은 시간 시계열 데이터에서의 Class-Incremental Learning (CIL) 접근법을 혁신적으로 제시합니다. 특히, 대규모 시계열 사전 훈련 모델(Pre-trained Models, PTMs)을 활용하여 시간 시계열 클래스 증분 학습(Time Series Class-Incremental Learning, TSCIL)의 가능성을 탐구했습니다. 연구는 PTM 기반의 방법론을 통해 기존 모델과 비교하여 성능 향상을 보여주며, 카타스트로픽 포겟팅(categorical forgetting) 문제를 해결하기 위한 혁신적인 프레임워크를 설계했습니다.

- **Technical Details**: 주요 기술적 요소로는 고정된 PTM 백본과 지식 증류(Knowledge Distillation, KD) 기법을 통해 공유 어댑터를 점진적으로 조정하는 방식이 포함됩니다. 또한, Feature Drift Compensation Network (DCN)를 도입하여 점진적 작업 간의 피처 공간 변환을 모델링하는 두 단계 훈련 전략을 설계했습니다. 이 방법은 이전 클래스 프로토타입을 새로운 피처 공간에 정확히 프로젝션하여 모델 정확도를 높입니다.

- **Performance Highlights**: 다섯 개의 실제 데이터셋에 대한 광범위한 실험 결과, 제안된 방법이 기존 PTM 기반 접근법과 비교하여 1.4%에서 6.1%의 최종 정확도 향상을 이루어냈습니다. 이 연구는 TSCIL의 새로운 패러다임을 제시하며, 지속적 학습 시스템에서 안정성-유연성 최적화에 대한 통찰력을 제공합니다. 또한, 시간 시계열 데이터 에서의 TSCIL 연구의 향후 방향에 대한 효과적인 방법론을 제공하고 있습니다.



### MRCEval: A Comprehensive, Challenging and Accessible Machine Reading Comprehension Benchmark (https://arxiv.org/abs/2503.07144)
Comments:
          Under review

- **What's New**: 이 논문에서는 기계 독해 이해(MRC) 평가를 위한 포괄적이고 도전적인 벤치마크인 MRCEval을 소개합니다. 기존 MRC 데이터셋은 주로 특정 독해 이해(RC) 능력을 평가했지만, MRCEval은 LLMs(Large Language Models)의 13가지 독해 기술을 포괄하여 총 2,103개의 고품질 다중 선택 질문을 포함하고 있습니다. 이는 LLMs의 RC 능력을 효과적으로 검토하며, 그들이 직면하는 도전 과제를 강조합니다.

- **Technical Details**: MRCEval은 세 가지 주요 작업과 13개의 하위 작업으로 구성되어 있으며, 각 작업은 MRC의 세 가지 핵심 측면인 맥락 이해(Context Comprehension), 외부 지식 이해(External Knowledge Comprehension), 그리고 추론(Reasoning)에 초점을 맞추고 있습니다. 이 벤치마크는 GPT-4o를 활용하여 샘플을 생성하고, 세 가지 경량 모델이 샘플의 질을 판별하는 평가자로 사용됩니다.

- **Performance Highlights**: 28개의 유명한 오픈 소스 및 폐쇄형 모델을 대상으로 한 평가 결과, 가장 경쟁력 있는 모델인 o1-mini 및 Gemini-2.0-flash조차도 MRCEval에서 통계적으로 낮은 성과를 보였습니다. 이는 LLMs의 성능이 기존 기준에서는 높게 평가되지만, MRC 문제에서는 여전히 상당한 도전 과제가 존재함을 보여줍니다.



### A Comprehensive Survey of Mixture-of-Experts: Algorithms, Theory, and Applications (https://arxiv.org/abs/2503.07137)
Comments:
          28 pages, 3 figures

- **What's New**: 본 논문은 Mixture of Experts (MoE) 모델의 최근 발전을 종합적으로 정리하여, 다양한 분야에서의 활용 가능성을 강조합니다. MoE는 동적으로 관련 있는 하위 모델을 선택하고 활성화함으로써 복잡한 데이터 처리의 효율성을 극대화하는 혁신적인 접근법을 제공합니다. 이 연구는 MoE의 기본 디자인, 알고리즘 및 이론적 연구를 포함하여 다양한 머신러닝 패러다임에서의 적용 사례를 다룹니다.

- **Technical Details**: MoE 모델은 각 입력 데이터의 특성에 기반하여 최적의 하위 파라미터 집합을 선택하여 활성화하는 'divide and conquer' 전략을 채택합니다. 이로 인해 각 전문가는 보다 전문화되어 다양한 지식 영역을 처리할 수 있으며, 이러한 선택적 활성화 메커니즘은 훈련 속도와 효율성을 높이는 데 기여합니다. 본 논문에서는 MoE의 기본 요소인 게이팅 함수(gating functions), 전문가 네트워크(expert networks), 라우팅 전략(routing mechanisms) 등 다양한 기술적 세부사항을 설명합니다.

- **Performance Highlights**: MoE 모델은 대규모, 멀티모달 데이터 처리에서 특히 성과를 낼 수 있는 가능성을 보여줍니다. 기존의 dense 모델에 비해 더 적은 리소스로도 향상된 성능을 제공하는 것으로 입증되었습니다. 이 연구는 MoE의 기존 연구에 한계를 보완하고 최신 트렌드를 반영하여 향후 연구 방향에 대한 논의를 포함합니다.



### ASTRA: A Negotiation Agent with Adaptive and Strategic Reasoning through Action in Dynamic Offer Optimization (https://arxiv.org/abs/2503.07129)
- **What's New**: 이번 논문에서는 자율적인 협상 에이전트를 개발하기 위해 ASTRA라는 새로운 프레임워크를 도입했습니다. ASTRA는 상대 모델링(opponent modeling)과 Tit-for-Tat(TFT) 상호 작용과 같은 두 가지 핵심 원칙을 기반으로 한 턴별 제안 최적화(turn-level offer optimization)를 특징으로 합니다. 이 에이전트는 시뮬레이션과 인간 평가를 통해 상대방의 행동 변화에 효과적으로 적응하며, 협상 성과를 향상시킵니다.

- **Technical Details**: ASTRA 에이전트는 세 가지 단계로 운영됩니다: (1) 상대방의 행동 해석, (2) 선형 프로그래밍(Linear Programming, LP) 해결기를 통한 반대 제안 최적화, (3) 협상 전술 및 수용 가능성에 기반한 제안 선택입니다. 이 에이전트는 상대의 선호도를 추론하고 이를 기반으로 제안을 조정하는 핵심 요소들을 통합하여 복잡한 협상에서도 유용하게 작동할 수 있도록 설계되었습니다. 이러한 접근 방식은 협상의 복잡성을 이해하고 전략적으로 대응할 수 있는 능력을 높입니다.

- **Performance Highlights**: ASTRA는 다양한 에이전트 유형 및 인간 평가를 통해 상대방의 변하는 태도에 맞춰 성공적으로 협상 목표를 달성했습니다. 이 에이전트는 전략적 피드백을 제공하고 최적의 제안 추천을 통해 협상 코칭 도구로도 활용될 수 있습니다. 연구 결과는 ASTRA가 협상에서의 적응력 및 전략적 추론을 크게 개선했다는 것을 보여줍니다.



### A LSTM-Transformer Model for pulsation control of pVADs (https://arxiv.org/abs/2503.07110)
- **What's New**: 본 연구에서는 새로운 AP-pVAD 모델이 제안되었습니다. 이 모델은 NPQ 모델과 LSTM-Transformer 모델의 두 부분으로 구성되어 있습니다. NPQ 모델은 pVAD의 모터 속도, 압력 및 유량 간의 수학적 관계를 정의합니다. LSTM-Transformer 모델은 기존 LSTM 신경망에 Transformer의 Attention 모듈을 통합하여 pVAD의 모터 속도를 조정하기 위한 맥동 시간 특성점을 예측합니다.

- **Technical Details**: NPQ 모델은 pVAD의 성능을 평가하기 위해 개발된 수학적 관계로, 모터 속도, 압력 및 유량 간의 관계를 설명합니다. LSTM-Transformer 모델은 훈련된 LSTM 신경망에 Transformer의 Attention 메커니즘을 추가하여, 데이터의 특성과 노이즈가 많은 환경에서도 훌륭한 성능을 발휘합니다. 이 연구에서는 세 가지 유체 실험과 동물 실험을 통해 AP-pVAD 모델의 유효성을 검증하였습니다.

- **Performance Highlights**: AP-pVAD 모델의 NPQ 모델을 통해 계산된 압력은 예상 값에 대해 최대 2.15 mmHg의 오차를 보였습니다. LSTM-Transformer 모델이 예측한 맥동 시간 특성점은 최대 1.78ms의 예측 오차를 보였으며, 이는 다른 방법들보다 현저히 낮습니다. 동물 실험에서 pVAD의 생체 내 시험 결과, 동물들이 pVAD 작동 시작 후 27시간 이상 생존하며 대동맥 압력의 유의미한 개선이 관찰되었습니다.



### FaceID-6M: A Large-Scale, Open-Source FaceID Customization Datas (https://arxiv.org/abs/2503.07091)
- **What's New**: 이번 논문에서는 최초의 대규모 오픈 소스 FaceID 데이터셋인 FaceID-6M을 수집하고 공개하였습니다. 이 데이터셋은 6백만 개의 고품질 텍스트-이미지 쌍으로 구성되어 있으며, LAION-5B에서 필터링 과정을 거쳐 제작되었습니다. 논문은 이 데이터셋이 FaceID 맞춤화 모델의 훈련에 적합하도록 세심하게 품질 관리 과정을 통해 구축되었다고 강조합니다.

- **Technical Details**: FaceID-6M 데이터셋은 이미지 필터링 및 텍스트 필터링 과정을 포함하여 구성됩니다. 이미지 필터링에서는 인간의 얼굴이 없는 이미지나 해상도가 낮은 이미지를 제거하며, 텍스트 필터링에서는 인물, 국적 및 직업과 관련된 용어가 포함된 설명을 남깁니다. 이러한 과정을 통해 FaceID-6M은 강력한 FaceID 맞춤화 모델 훈련을 최적화된 데이터셋을 제공합니다.

- **Performance Highlights**: FaceID-6M 데이터셋으로 훈련된 모델은 기존 산업 모델에 비해 경쟁력 있는 성능을 보여줍니다. 실험 결과, FaceID-6M으로 훈련된 InstantID 모델이 COCO-2017 테스트 세트에서 0.63의 FaceID 충실도 점수를 달성하여 이전 모델인 InstantID의 0.59 점수를 초과하였습니다. 이러한 결과는 FaceID-6M이 FaceID 맞춤화 커뮤니티에 중요한 기여를 할 수 있음을 시사합니다.



### On the Generalization of Representation Uncertainty in Earth Observation (https://arxiv.org/abs/2503.07082)
Comments:
          18 pages

- **What's New**: 최근 컴퓨터 비전 분야에서 사전 훈련된 표현 불확실성(pretrained representation uncertainty) 개념이 발전하여 제로샷 불확실성 추정(zero-shot uncertainty estimation)이 가능해졌다. 이는 신뢰성이 중요한 지구 관측(Earth Observation, EO) 분야에 크게 기여할 수 있으나, EO 데이터의 복잡성은 불확실성 인식 방법의 도전에 직면하게 한다. 이 연구는 EO의 고유한 의미적 특성을 고려하여 표현 불확실성의 일반화를 조사하였다.

- **Technical Details**: 연구자들은 큰 EO 데이터셋을 활용하여 불확실성을 사전 훈련(pretrain)하고, 다중 레이블 분류(multi-label classification) 및 세분화(segmentation) EO 작업에서 제로샷 성능을 평가하는 프레임워크를 제안하였다. 이 방법은 대규모 사전 훈련된 모델의 표현 공간에서 직접 불확실성을 추출하여, 큰 확장성을 가진 제로샷 불확실성 추정을 가능하게 한다. 또한, EO의 의미적 요인(Semantic Factors, SFs)을 정의하고 이들이 표현 불확실성의 일반화에 미치는 영향을 평가하였다.

- **Performance Highlights**: 연구 결과, 자연 이미지에서 사전 훈련된 불확실성과 달리 EO 사전 훈련은 보지 못한 EO 도메인, 지리적 위치 및 목표 세분성에 대해 강력한 일반화를 나타냈다. GSD(ground sampling distance)에 대한 민감성을 유지하면서도, EO 사전 훈련 불확실성이 하류 작업(downstream tasks)에서의 작업 특화 불확실성과 잘 맞아 떨어짐을 보여주었다. 이 연구는 EO 분야에서의 표현 불확실성의 강점과 제한을 논의하며, 향후 연구의 길을 제시한다.



### An Experience Report on Regression-Free Repair of Deep Neural Network Mod (https://arxiv.org/abs/2503.07079)
- **What's New**: 딥 뉴럴 네트워크(Deep Neural Networks, DNNs)를 기반으로 한 시스템이 산업에서 점점 더 많이 사용되고 있습니다. DNN의 성능 향상을 위해 시스템 운영 중 DNN을 업데이트해야 하지만, 높은 신뢰성을 요구하는 기업의 경우 회귀(regression)를 최소화해야 합니다. 본 논문은 산업에서 DNN 업데이트를 위한 요구사항을 식별하고, 이를 충족하기 위한 사례 연구를 제시합니다.

- **Technical Details**: 본 논문에서는 DNN 모델 수리 기술인 NeuRecover를 사용해 특정 클래스를 위한 회귀 없이 보안 애플리케이션을 가정한 자동차 이미지로 훈련된 모델을 업데이트하는 방법을 설명합니다. NeuRecover는 데이터에서 회귀를 감지하고, 미비한 데이터에 대해서는 개선되는 데이터를 찾아 파라미터를 최적화하는 기존 기법입니다. 개선된 NeuRecoverLite는 훈련 과정의 결과 없이 결함 로컬라이제이션(fault localization)을 수행하여 데이터 기록이 없는 현장에서 사용되었습니다.

- **Performance Highlights**: 사례 연구 결과, 66개의 하이퍼파라미터로 10번 실행한 결과 회귀 없이 수정할 수 있는 패턴을 발견했습니다. NeuRecoverLite는 결함 로컬라이제이션과 입자 군집 최적화(particle swarm optimization) 두 단계로 구성되어 있으며, 결함이 있는 데이터를 패스한 데이터에 영향을 미치지 않는 파라미터를 식별하여 최적화합니다. 최적화 과정에서 실패한 데이터를 수정하고, 패스한 데이터를 유지하는 방향으로 각 손실 값을 최소화하는 피트니스 함수를 사용합니다.



### NFIG: Autoregressive Image Generation with Next-Frequency Prediction (https://arxiv.org/abs/2503.07076)
Comments:
          10 pages, 7 figures, 2 tables

- **What's New**: 이번 논문에서는 이미지 생성을 위한 새로운 프레임워크인 Next-Frequency Image Generation(NFIG)을 제안합니다. NFIG는 이미지 생성 과정을 여러 주파수 기반 단계로 분해하여 처리하며, 이를 통해 전반적인 구조를 설정하는 저주파 구성 요소를 생성한 후 고주파 세부 사항을 점진적으로 추가합니다. 이 접근 방식은 이미지 구성 요소 간의 인과 관계를 더 잘 포착하여 이미지 품질을 향상시키고, 계산 비용을 크게 줄입니다.

- **Technical Details**: NFIG의 핵심은 주파수 분석을 바탕으로 한 Frequency-guided Residual-quantized VAE(FR-VAE)입니다. 이는 이미지에서의 저주파 및 고주파 요소를 분리하여 저주파 구성 요소로 전반적인 구조를 수용하고 고주파 구성 요소로 세부 사항을 유지합니다. FR-VAE는 자원 효율성을 극대화하며, 전체 주파수 스펙트럼에서 정보를 효율적으로 표현하는 방법론으로 설계되었습니다.

- **Performance Highlights**: NFIG는 뛰어난 이미지 생성 품질을 보여주며, ImageNet-256 벤치마크에서 FID 2.81을 기록하여 최신 성능을 달성했습니다. 또한, 1.25배의 속도 향상을 보여주는 등 개선된 효율성을 입증했습니다. 이 논문은 이미지 생성 모델에 대한 새로운 통찰력을 제공하며, 향후 연구의 방향을 제시하고자 합니다.



### PIED: Physics-Informed Experimental Design for Inverse Problems (https://arxiv.org/abs/2503.07070)
Comments:
          Accepted to 13th International Conference on Learning Representations (ICLR 2025), 31 pages

- **What's New**: 이 논문에서는 Physics-Informed Experimental Design (PIED)라는 새로운 실험 설계 프레임워크를 제안합니다. 이는 물리 통합 신경망(PINNs)을 활용하여 역 문제(inverse problem, IP)의 설계 매개변수를 연속적으로 최적화하는 독창적인 접근 방식을 보여줍니다. PIED는 기존 방법의 계산 병목 현상을 극복하고, 한 번의 데이터 수집으로 실행할 수 있도록 설계되어 있습니다.

- **Technical Details**: PIED는 PINNs의 완전 미분 가능한 아키텍처를 활용하여 설계 매개변수를 지속적으로 최적화합니다. 초기 학습된 신경망 파라미터를 도입하여 여러 PDE 매개변수에 대해 효율적인 PINN 학습이 가능하도록 합니다. 또한, PINN 학습 동태를 효과적으로 고려하는 다양한 실험 설계 기준을 제시하고, 이들은 설계 매개변수에 대해 미분 가능하여 효율적인 경량 기반 방법을 통해 최적화할 수 있습니다.

- **Performance Highlights**: 실험 결과, PIED는 한정된 관측 예산 하에서도 기존의 ED 기법보다 역 문제를 해결하는 데 있어 크게 성능을 개선함을 보였습니다. 이는 PDE 매개변수가 유한 차원이거나 알려지지 않은 함수일 경우에도 적용됩니다. 따라서 PIED는 복잡한 모델 및 PDE를 포함하는 실험 설계 문제에서의 유망한 해결책으로 자리잡을 가능성이 큽니다.



### DistiLLM-2: A Contrastive Approach Boosts the Distillation of LLMs (https://arxiv.org/abs/2503.07067)
Comments:
          The code will be available soon at this https URL

- **What's New**: 이 논문에서는 DistiLLM-2라는 새로운 대조적 접근법을 제안하여, 교사 모델과 학생 모델 간의 응답 가능성을 조정함으로써 LLM (Large Language Model) 증류의 효율성을 극대화했습니다. DistiLLM-2는 다양한 데이터 유형과 손실 기능 간의 시너지를 활용하여 학생 모델의 성능 향상을 도모합니다. 이를 통해 다양한 작업에서 높은 성능을 보이는 학생 모델을 구축할 수 있습니다.

- **Technical Details**: DistiLLM-2는 대칭이 없는 손실 역학을 분석하여, 교사와 학생 모델의 응답에 대해 서로 다른 손실 함수를 적용하는 대조적 방법(CALD; Contrastive Learning for Distillation)을 개발합니다. 이를 통해 손실 기능과 데이터 관점 간의 시너지를 효과적으로 통합합니다. 더불어, 데이터셋 커리는 최적화되었으며, 커리큘럼 기반의 적응형 손실 메커니즘이 도입되어, DistiLLM-2는 실무자들을 위한 강력한 지침을 제공합니다.

- **Performance Highlights**: DistiLLM-2는 다양한 텍스트 생성 작업에서 최첨단 성능을 달성하며, 지침 수행, 수학적 추론, 코드 생성 등을 포함합니다. 또한, 이 접근법은 선호 정렬(preference alignment)과 비전-언어 모델 확장과 같은 다양한 애플리케이션을 지원하여, 넓은 범위의 활용 가능성을 보여줍니다.



### Generative method for aerodynamic optimization based on classifier-free guided denoising diffusion probabilistic mod (https://arxiv.org/abs/2503.07056)
Comments:
          Under Review

- **What's New**: 이번 논문에서는 성능 목표를 충족하기 위해 최적의 공기역학적 형태를 직접 생성하는 역설계 접근법(inverse design approach)에 대한 새로운 프레임워크를 제안합니다. 특히, classifier-free guided denoising diffusion probabilistic model (CDDPM)을 기반으로 한 역설계 방법은 현재의 기존 기법보다 더 높은 정확성을 보여줄 수 있습니다. 이 모델은 특정 성능 지표 간의 관계를 효과적으로 파악하고 특정 압력 특징에 따라 상부 및 하부 압력 계수 분포를 생성할 수 있습니다.

- **Technical Details**: CDDPM은 성능 지표의 상관관계를 포착하여, classifier-free guide 계수를 조정함으로써 특정한 압력 특징 기반의 압력 계수 분포를 생성합니다. 이 분포는 매핑 모델을 통해 정확하게 공기foil(airfoil) 지오메트리로 변환됩니다. 논문에서는 전통적인 초음속 공기foil(transonic airfoils)을 사용한 실험 결과도 포함되어, CDDPM 기반의 역설계가 다양한 압력 계수 분포를 만들어내며 디자인 결과의 다양성을 강화한다는 사실을 입증합니다.

- **Performance Highlights**: CDDPM은 현재의 Wasserstein generative adversarial network 방법들에 비해 33.6% 높은 정확도를 달성하며, 이는 공기foil 생성 작업에 있어 상당한 개선을 나타냅니다. 또한, 글로벌 최적화 알고리즘(global optimization algorithm)과 능동 학습 전략(active learning strategy)을 기반으로 각 성능 지표 값을 재조정하는 실용적인 방법도 제시하여, 역설계 프레임워크의 성능 지표 조합을 합리적으로 제공합니다. 이 연구는 공기foil 디자인에 한정되지 않고, 선택된 성능 지표를 목표로 한 일반 제품 부품의 최적화 과정에도 적용 가능합니다.



### TIDE : Temporal-Aware Sparse Autoencoders for Interpretable Diffusion Transformers in Image Generation (https://arxiv.org/abs/2503.07050)
- **What's New**: 이번 논문에서는 기존 U-Net 기반의 diffusion 모델과 비교하여 Diffusion Transformers (DiTs)의 새로운 해석 가능성 탐구를 위한 TIDE 프레임워크(Tide Framework)를 제안합니다. TIDE는 Sparse Autoencoders (SAEs)를 사용하여 활성화 레이어 내에서의 시간적 재구성을 개선하며, diffusion 모델이 생성적 사전 학습 과정에서 계층적 기능을 스스로 학습한다는 것을 밝혀냅니다. 이 연구는 DiT의 내부 작업을 이해하는 데 필수적인 기여를 하며, 다양한 응용 분야에서 향상된 제어 가능성을 제공합니다.

- **Technical Details**: TIDE는 시간적 변화를 포착하고 denoising 과정 전반에 걸쳐 희소하고 해석 가능한 기능을 추출하기 위해 설계된 프레임워크입니다. 이 연구에서는 프로그레시브 스파시티 스케줄링(progressive sparsity scheduling) 및 랜덤 샘플링 증강(random sampling augmentation)과 같은 특수 교육 전략을 도입하여, 높은 충실도를 유지하면서 학습 속성을 최적화합니다. TIDE는 또한 multi-level features를 학습하여 다양한 다운스트림 작업에 적합하게 설계되었습니다.

- **Performance Highlights**: TIDE는 평균 제곱 오차(MSE) 1e-3 및 코사인 유사도 0.97을 기록하며, state-of-the-art 재구성 성능을 자랑합니다. TIDE는 사용자 친화적이고 효율적인 성능을 제공하며, 다수의 실험을 통해 고해상도 이미지에서의 활성화 역학을 효과적으로 포착하는 것이 입증되었습니다. 이미지 편집, 스타일 전송 등 다양한 분야에서의 응용 가능성을 통해 생성 시스템의 신뢰성과 제어 가능성을 높이는 데 기여하고 있습니다.



### DatawiseAgent: A Notebook-Centric LLM Agent Framework for Automated Data Scienc (https://arxiv.org/abs/2503.07044)
- **What's New**: 이번 논문에서 제안하는 DatawiseAgent는 데이터 과학의 작업을 유연하고 적응적으로 자동화하기 위한 노트북 중심의 LLM(대규모 언어 모델) 에이전트 프레임워크입니다. 이 프레임워크는 사용자, 에이전트 및 계산 환경 간의 상호작용을 통합하여 데이터 과학의 복잡한 작업을 효과적으로 처리할 수 있도록 설계되었습니다. 기존의 LLM 기반 접근 방식들이 특정 단계에 집중하는 경향이 있지만, DatawiseAgent는 연속적인 데이터 과학 작업의 의존성을 잘 반영합니다.

- **Technical Details**: DatawiseAgent는 유한 상태 변환기(Finite State Transducer, FST)를 기반으로 하는 다단계 설계를 사용하여 네 가지 주요 단계인 DFS(Depth First Search) 계획, 점진적 실행, 자기 디버깅 및 후처리를 조율합니다. 특히 DFS-like 계획 단계는 솔루션 공간을 체계적으로 탐색하고, 점진적 실행 단계는 실시간 피드백을 활용하여 제한된 LLM의 능력을 극대화합니다. 자기 디버깅 및 후처리 모듈은 오류를 진단하고 수정하여 신뢰성을 향상시키는 역할을 합니다.

- **Performance Highlights**: 실험 결과, DatawiseAgent는 다양한 데이터 과학 작업, 특히 데이터 분석, 시각화 및 데이터 모델링에서 기존의 최첨단 방법들과 비교하여 우수하거나 동등한 성능을 보여주었습니다. 가장 도전적인 데이터 모델링 작업에서도 DatawiseAgent는 90% 이상의 작업 완료율과 함께 40 이상의 상대 성능 격차(Relative Performance Gap)를 기록하며 탁월한 결과를 달성했습니다. 이러한 결과는 DatawiseAgent가 데이터 과학 시나리오에서 일반화할 수 있는 잠재력을 강조하며, 더욱 효율적이고 완전 자동화된 워크플로우를 위한 기초를 마련합니다.



### Bot Wars Evolved: Orchestrating Competing LLMs in a Counterstrike Against Phone Scams (https://arxiv.org/abs/2503.07036)
- **What's New**: 이번 논문에서는 'Bot Wars'라는 프레임워크를 소개합니다. 이 프레임워크는 대화형 적대적 기법을 통해 전화 사기를 방어하기 위해 대형 언어 모델(LLMs)을 활용합니다. 핵심 기여는 명시적인 최적화 없이 사고의 연쇄를 통한 전략의 출현(Strategy Emergence)에 대한 정형화된 기반을 제시하는 것입니다.

- **Technical Details**: Bot Wars는 두 계층의 프롬프트 아키텍처를 통해 전략적 일관성을 유지하면서 인구통계적으로 진정한 피해자 페르소나를 생성할 수 있는 LLM의 능력을 극대화합니다. 시스템은 사기꾼 에이전트와 피해자 에이전트 각각의 상대적인 목표를 가진 두 개의 적대적 에이전트를 포함하여 작동합니다. 프레임워크는 3,200개의 사기 대화 데이터셋을 기반으로 하여 179시간의 인간 사기 방어 상호작용과 검증된 평가를 실시합니다.

- **Performance Highlights**: 실험 평가 결과, GPT-4가 대화의 자연스러움과 페르소나의 진정성에서 뛰어난 성능을 보이는 반면, Deepseek는 더 나은 참여 지속성을 보여주었습니다. 다양한 지표를 통해 대화의 효과성을 정량화하였으며, 이에 따라 Bot Wars 프레임워크의 효과성을 입증하였습니다.



### Availability-aware Sensor Fusion via Unified Canonical Space for 4D Radar, LiDAR, and Camera (https://arxiv.org/abs/2503.07029)
Comments:
          Arxiv preprint

- **What's New**: 이번 연구에서는 자율주행(AD) 시스템을 위한 새로운 방법인 availability-aware sensor fusion (ASF)을 제안합니다. ASF는 각 센서의 특징을 통합하여 일관성을 보장하는 unified canonical projection (UCP)을 사용하며, 센서 고장 및 열화에 저항할 수 있는 cross-attention across sensors along patches (CASAP)를 도입하였습니다. 이 방법은 다양한 기상 조건과 센서 열화 상황에서도 기존의 최첨단 융합 방법들보다 뛰어난 성능을 발휘합니다.

- **Technical Details**: ASF는 깊이 연결된 융합(DCF) 및 센서별 크로스 어텐션 융합(SCF)의 한계를 동시에 해결합니다. UCP는 모든 센서의 특징을 통일된 공간에 투영하여 일관성을 없애며, CASAP-PN은 센서의 가용성을 고려하여 크로스 어텐션을 수행합니다. 이를 통해 ASF는 복잡한 위치 임베딩을 제거하고 계산 효율성을 향상시킵니다.

- **Performance Highlights**: K-Radar 데이터셋에서 ASF는 기존 SOTA(상태-최고) 융합 방법보다 9.7% 향상된 AP BEV(87.2%)와 20.1% 향상된 AP 3D(73.6%)를 기록했습니다. 이러한 성능은 센서 고장이나 저하와 같은 극단적인 상황에서도 유지됩니다. ASF는 낮은 계산 비용으로 높은 성능을 달성하여 자율주행의 신뢰성과 강인성을 향상시킵니다.



### Erase Diffusion: Empowering Object Removal Through Calibrating Diffusion Pathways (https://arxiv.org/abs/2503.07026)
Comments:
          accepted by CVPR 2025

- **What's New**: 이 논문에서는 객체 제거를 위한 새로운 방식인 EraDiff를 제안합니다. EraDiff는 기존의 표준 확산(diffusion) 패러다임의 최적화 방법을 개선하여, 객체 제거의 효과성을 극대화할 수 있는 경로를 설정합니다. 이 모델은 Chain-Rectifying Optimization (CRO)라는 새로운 최적화 패러다임을 도입하여 객체 제거를 위한 혁신적인 확산 전이 경로를 구축합니다.

- **Technical Details**: EraDiff의 구조는 객체를 효과적으로 제거하기 위해 Self-Rectifying Attention (SRA) 메커니즘을 사용합니다. SRA 메커니즘은 자가 주의(self-attention) 활성화를 조절하여 모델이 아티팩트를 회피하고 생성된 콘텐츠의 일관성을 향상시킵니다. 이 최적화 접근법은 기존의 확산 경로가 아닌, 객체에서 배경으로의 직접적인 확산 경로를 통해 객체 제거를 수행할 수 있도록 합니다.

- **Performance Highlights**: 제안된 EraDiff는 공개된 OpenImages V5 데이터셋에서 최첨단 성능을 달성했으며, 실제 환경에서도 우수한 결과를 보여줍니다. 이 접근법은 객체를 제거하는 과정에서 발생할 수 있는 예상치 못한 아티팩트를 효과적으로 회피하여, 보다 자연스러운 배경을 생성합니다.



### Weak Supervision for Improved Precision in Search Systems (https://arxiv.org/abs/2503.07025)
Comments:
          Accepted to the AAAI 2025 Workshop on Computational Jobs Marketplace

- **What's New**: 이 논문에서는 대규모 검색 시스템의 정밀도를 향상시키기 위해 Learning to Rank 프레임워크 내에서 쿼리-문서 쌍의 질을 추론하는 약한 지도 학습(weak supervision) 접근법을 제시합니다. 기존의 데이터 라벨링 방식이 시간과 비용이 많이 드는 반면, 사용자의 클릭 및 활동 로그를 대안으로 활용하여 효율적으로 학습 라벨을 생성할 수 있는 방법을 제안합니다. 이를 통해 '골드' 데이터셋 작성의 비용을 줄이고, 보다 신뢰성 있는 훈련 데이터를 생성합니다.

- **Technical Details**: 제안된 시스템 아키텍처는 Apache Spark를 통해 구현되며, 라벨링 함수(Label Functions, LFs)를 각 기록에 대해 실행합니다. 다양한 외부 데이터베이스, 모델 및 분류 체계를 활용하여 결정들을 내립니다. 제안된 방식은 기존의 Snorkel 프레임워크와는 달리 이진 라벨에 중점을 둡니다. LFs의 출력은 True(긍정), False(부정), 또는 null(기권)로 표현되며, 이 결과들을 결합하여 단일 확률적 라벨로 집계합니다.

- **Performance Highlights**: 훈련된 모델은 수억 개의 데이터 포인트에 대해 최적화된 DNN으로, TensorFlow와 Horovod를 사용하여 분산 환경에서 학습됩니다. 모델 성능은 NDCG@k를 통해 원래 레이블, 업데이트 된 레이블, 그리고 약한 라벨러의 예측 세트에 대해 평가됩니다. 실험 결과, 약한 지도 학습을 통해 정밀도 및 재현율이 개선되었음을 발견하였습니다.



### Combating Partial Perception Deficit in Autonomous Driving with Multimodal LLM Commonsens (https://arxiv.org/abs/2503.07020)
- **What's New**: 이번 논문에서는 LLM-RCO라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 대형 언어 모델(LLM)을 활용하여 자율 주행 시스템에서 발생할 수 있는 인지 결함(perception deficits)에 대한 인간과 유사한 상식(driving commonsense)을 통합합니다. LLM-RCO는 위험 추론(hazard inference), 단기 움직임 계획(short-term motion planner), 행동 조건 검증(action condition verifier) 및 안전 제약 생성기(safety constraint generator)의 네 가지 주요 모듈로 구성되어 있습니다.

- **Technical Details**: LLM-RCO는 카메라 비전이 손상된 경우에도 자율 주행 차량의 제어를 오버라이드하여, 손상된 인식 데이터를 기반으로 하는 잘못된 예측에서 발생할 수 있는 위험한 행동을 완화합니다. 각 모듈은 주행 환경과 동적으로 상호작용하며, 이를 통해 안전하고 신뢰할 수 있는 차량 제어를 가능하게 합니다. 또한, 이 연구에서는 53,895개의 비디오 클립으로 구성된 DriveLM-Deficit라는 데이터셋을 구축하여, LLM 기반의 위험 추론 및 움직임 계획에 대한 미세 조정을 진행합니다.

- **Performance Highlights**: CARLA 시뮬레이터를 활용한 광범위한 실험 결과, LLM-RCO가 장착된 시스템이 주행 성능을 현저히 향상시키는 것으로 나타났습니다. 이는 자율주행의 인지 결함에 대한 회복력(resilience)을 강화할 수 있는 잠재력을 강조합니다. 또한, LLM이 DriveLM-Deficit로 미세 조정되었을 때, 보수적인 정지 대신 보다 적극적인 이동이 가능하다는 점이 확인되었습니다.



### NukesFormers: Unpaired Hyperspectral Image Generation with Non-Uniform Domain Alignmen (https://arxiv.org/abs/2503.07004)
- **What's New**: 이 논문에서는 비례에 맞지 않는 히퍼스펙트럼 이미지 생성(UnHIG)의 과제를 해결하기 위해, Range-Null Space Decomposition (RND) 방법론을 활용하여 null space의 상호작용을 모델링하였습니다. 특히, 비례 데이터의 기하학적 및 스펙트럼 분포를 효과적으로 정렬하기 위해 대비 학습을 도입했습니다. 이를 통해, 기존의 unpaired HIG 방법이 직면한 문제들을 해결하고, 새로운 벤치마크를 설정하는 데 기여하고 있습니다.

- **Technical Details**: 제안된 방법론은 비례하지 않는 히퍼스펙트럼 데이터를 연속적인 성분과 감쇠된 성분으로 체계적으로 분해합니다. RND 방법론에 기반하여 range space와 null space로 나눈 후, dual-dimensional contrastive learning을 통해 range space 내에서의 연속 요소를 효과적으로 집계합니다. 또한, Kolmogorov-Arnold Networks (KANs)를 기반으로 한 Non-Uniform Matrix Object-Aware Mechanism을 도입하여 null space의 보상을 강화하고 고차원 맞춤화를 지원합니다.

- **Performance Highlights**: 실험을 통해 제안된 방법이 다양한 벤치마크에서 최신 성능을 달성했음을 입증하였습니다. 특히, 고주파 성분 보상 및 효과적인 cross-domain 상호작용이 강조되고 있으며, 이로 인해 UnHIG 분야에서의 연구에 새로운 방향성을 제시하고 있습니다. 연구 결과는 산업 적용 가능성을 높이는 데 중요한 역할을 할 것으로 기대됩니다.



### Social Bias Benchmark for Generation: A Comparison of Generation and QA-Based Evaluations (https://arxiv.org/abs/2503.06987)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)의 사회적 편향을 측정하기 위해 'Bias Benchmark for Generation (BBG)'라는 새로운 벤치마크를 제안합니다. 기존의 QA 기반 평가 방식은 긴 형식의 생성에서 발생하는 편향을 충분히 포착하지 못하므로, LLM이 이야기의 진행을 생성하는 과정을 통해 편향을 측정하는 방식을 채택했습니다. BBG는 영어와 한국어에서 구축되어, LLM의 생성 결과에서 중립적 및 편향된 생성의 확률을 측정합니다.

- **Technical Details**: BBG는 이야기 생성의 편향을 측정하기 위해 두 가지 지표, 중립성 점수(ntr_gen) 및 편향 점수(bias_gen)를 도입합니다. 중립성 점수는 모델이 특정 개인에 대한 연관을 하지 않거나, 두 개의 서로 다른 버전에서 동일한 순서대로 개인에 대해 일관되게 매핑할 때의 비율을 측정합니다. 편향 점수는 비편향적 생성과 편향적 생성의 비율 차이로 정의되며, 이를 통해 LLM의 사회적 편향을 정량화합니다.

- **Performance Highlights**: 실험 결과, 10개의 LLM 중 대부분이 생성하는 출력의 49%에서 69%가 중립적이며, 편향에 맞는 생성의 확률이 편향에 반하는 출력보다 10%에서 25% 더 높습니다. 기존의 BBQ 평가와 비교했을 때, 두 접근 방식의 결과는 일관성이 없음을 보여주며, 동일한 모델 내에서도 일반 성능이 높은 모델이 QA 작업에서는 낮은 편향 점수를, 생성 작업에서는 높은 편향 점수를 보이는 경향이 있음을 확인했습니다.



### Understanding the Learning Dynamics of LoRA: A Gradient Flow Perspective on Low-Rank Adaptation in Matrix Factorization (https://arxiv.org/abs/2503.06982)
- **What's New**: 이 논문은 Low-Rank Adaptation (LoRA)의 첫 번째 이론 분석을 통해, 잘 설계된 초기값이 어떻게 모델을 새로운 작업에 적응시키는지를 다룹니다. 특히, gradient flow (GF) 하에서의 학습 동역학에 대한 연구를 통해 LoRA의 효과를 명확히 하고 있습니다. 초기화에 따라 결과가 달라지며, 초기화 크기를 줄이는 것이 최종 오류를 감소시킨다는 사실이 밝혀졌습니다.

- **Technical Details**: 논문에서는 작은 초기화가 GF가 최적 솔루션의 근처로 수렴하도록 하는 것을 이론적으로 보여줍니다. 또한, 초기화 크기가 최종 오류에 미치는 영향과 함께, 사전 훈련된 모델의 특이 공간과 타겟 행렬 간의 비정렬성이 오류에 미치는 영향을 탐구합니다. LoRA의 스펙트럴 초기화를 제안하며, 이는 미스 얼라인먼트를 해결하는 방법으로 제시되었습니다.

- **Performance Highlights**: 행렬 분해 (MF) 및 이미지 분류 실험을 통해 이론적 발견을 검증하였습니다. 스펙트럴 초기화를 이용한 GF가 임의의 정밀도로 미세 조정 작업에 수렴함을 보여 주어 실용적인 응용 가능성을 제시합니다. 이러한 결과는 초기화 전략이 LoRA의 성능에 미치는 영향을 실질적으로 뒷받침합니다.



### Lightweight Multimodal Artificial Intelligence Framework for Maritime Multi-Scene Recognition (https://arxiv.org/abs/2503.06978)
Comments:
          19 pages, 4 figures, submitted to Engineering Applications of Artificial Intelligence

- **What's New**: 이번 연구는 해양 다중 장면 인식을 위한 혁신적인 멀티모달 인공지능 프레임워크를 제안합니다. 이 프레임워크는 이미지 데이터, 텍스트 설명, 그리고 다중모달 대규모 언어 모델(Multimodal Large Language Model, MLLM)에 의해 생성된 분류 벡터를 통합하여 인식 정확도를 높이고 풍부한 의미를 제공합니다. 실험 결과, 새로운 모델이 98%의 정확도로 이전의 최첨단 모델보다 3.5% 향상되었습니다.

- **Technical Details**: 프레임워크의 핵심은 이미지 특징 추출을 위한 첨단 기술인 Swin Transformer와 텍스트 데이터를 처리하는 BERT, 그리고 분류 벡터를 처리하는 다층 퍼셉트론(Multi-Layer Perceptron, MLP)으로 구성된 효율적인 기능 추출 과정입니다. 이 시스템은 각 모달리티의 가장 관련성 높은 정보를 강화하여 전체 모델의 견고성을 보장합니다. 또한, 활성화 인식 가중치 양자화(Activation-aware Weight Quantization, AWQ)를 통해 모델 크기를 68.75MB로 줄이면서도 실시간 해양 장면 인식을 위한 성능을 유지합니다.

- **Performance Highlights**: 제안된 프레임워크는 해양 환경에서의 실시간 배치 작업에 매우 적합하며, 자율 해양 차량(Autonomous Surface Vehicles, ASVs)의 환경 모니터링과 재난 대응에 이로운 결과를 제공합니다. 이 연구는 다양한 해양 환경에서 인식 정확도를 개선하기 위한 다중모달 융합 기술의 가능성을 보여주며, 자원 제한된 환경에서도 높은 성능을 발휘하는 솔루션을 제시합니다.



### A Multimodal Benchmark Dataset and Model for Crop Disease Diagnosis (https://arxiv.org/abs/2503.06973)
Comments:
          Accepted by ECCV 2024 (14 pages, 8 figures)

- **What's New**: 이번 연구에서는 농업 분야의 질병 진단을 위한 다중 모달 대화형 AI의 가능성을 탐구하며, CDDM(Crop Disease Domain Multimodal) 데이터셋을 소개합니다. 이 데이터셋은 137,000개의 농작물 질병 이미지와 100만 개의 질문-답변 쌍으로 구성되어 있으며, 농업 지식의 다양한 범위를 포괄합니다. 이를 통해 고급 질문-응답 시스템을 개발하고 농업 전문가에게 유용한 조언을 제공할 수 있는 토대를 마련하고자 합니다.

- **Technical Details**: CDDM 데이터셋은 다양한 농작물 질병과 관련된 질문-답변의 상호작용을 통해 다중 모달 학습 기법을 적용합니다. 특히, 시각 인코더, 어댑터 및 언어 모델을 동시에 미세 조정하는 새로운 LoRA(low-rank adaptation) 전략을 채택하여 농작물 질병 진단의 정확성을 개선했습니다. 이 연구의 방법론은 농업 기술 연구를 촉진하기 위한 벤치마크와 함께 제공됩니다.

- **Performance Highlights**: CDDM 데이터셋을 통해 최신 다중 모달 모델을 미세 조정한 결과, 농작물 질병 진단의 질적 향상이 나타났습니다. 특히, 일반적인 VQA 시스템에서 겪는 탄력성 문제를 해결함으로써 보다 신뢰할 수 있는 진단 결과를 제공할 수 있게 되었습니다. 이 데이터셋과 모델은 오픈소스로 제공되어 연구자들이 농업 분야에서 다중 모달 학습을 발전시키는 데 기여할 것입니다.



### Multi-Behavior Recommender Systems: A Survey (https://arxiv.org/abs/2503.06963)
Comments:
          Accepted in the PAKDD 2025 Survey Track

- **What's New**: 최근 다중 행동 추천 시스템(multi-behavior recommender systems)에 대한 연구가 급격히 증가하고 있으며, 이러한 시스템은 사용자 선호도를 예측하기 위해 다양한 사용자-아이템 상호작용을 활용합니다. 특히 사용자의 클릭, 장바구니에 담기, 위시리스트 저장과 같은 행동은 기존의 구매와 평점 한정 방식에 비해 더욱 풍부한 통찰을 제공합니다. 이 문서에서는 데이터 모델링, 인코딩, 훈련의 세 가지 주요 단계에 초점을 맞춰 다양한 다중 행동 추천 기술을 종합적으로 검토합니다.

- **Technical Details**: 다중 행동 추천 시스템은 데이터 모델링(data modeling), 인코딩(encoding), 훈련(training) 세 가지 주요 단계를 통해 구성됩니다. 데이터 모델링 단계에서는 다양한 사용자 행동을 그래프(graph) 또는 시퀀스(sequence)와 같은 특정 데이터 구조로 표현합니다. 인코딩 단계에서는 모델링된 데이터를 벡터 표현(vector representations)으로 변환하고, 훈련 단계에서는 이러한 다중 행동 정보를 효과적으로 학습하고 활용하기 위해 파라미터를 최적화합니다.

- **Performance Highlights**: 최신의 다중 행동 추천 시스템(MULE)은 목표 행동(target behavior)인 구매의 예측에서 기존 방법에 비해 최대 463%의 성능 향상을 보입니다. 이러한 성능 향상은 사용자의 다양한 행동 데이터를 효과적으로 통합함으로써 달성되었습니다. 다중 행동 추천 시스템의 성능 평가는 수많은 벤치마크 데이터셋을 통해 이루어지며, 인기 있는 데이터셋 중 일부는 Tmall, Taobao, Yelp 등 다양한 도메인에서 수집된 것입니다.



### Capture Global Feature Statistics for One-Shot Federated Learning (https://arxiv.org/abs/2503.06962)
Comments:
          AAAI 2025

- **What's New**: 이번 논문에서는 FedCGS라는 새로운 연합 학습(FL) 알고리즘을 제안합니다. 이 알고리즘은 사전 훈련된 모델을 활용하여 글로벌 피처 통계를 캡처하고, 이를 통해 단일 통신 라운드에서의 훈련이 필요 없는 이점을 제공합니다. FedCGS는 비 IID 데이터를 효과적으로 처리하고, 개인화된 시나리오에 대한 확장을 통해 한 번의 추가 통신으로 글로벌 통계를 다운로드하는 방식으로 작동합니다.

- **Technical Details**: FedCGS는 파라미터가 필요 없는 Naive Bayes 분류기를 사용하여 데이터 이질성에 강한 단일 통신 라운드 FL을 달성합니다. 논문에서 다룬 기존의 원샷 FL 방법들은 높은 계산 비용이나 비공식적인 데이터 처리의 문제를 겪고 있었으나, FedCGS는 이러한 한계를 극복하였습니다. 전통적인 federated learning의 통신 비용 및 보안 문제를 해결하기 위해, 글로벌 피처 통계를 통해 비대칭적 데이터를 보다 안전하게 처리할 수 있습니다.

- **Performance Highlights**: 광범위한 실험 결과는 FedCGS가 라벨 시프트 및 피처 시프트 설정에서 통신-정확도 경계를 향상시키는 것을 보여줍니다. 특히, FedCGS는 다양한 데이터 이질성 환경에서 확장성을 유지하며, 기존의 다른 연결 알고리즘들에 비해 향상된 성능을 입증하였습니다. 이 연구는 FL 커뮤니티의 관심을 끌 것으로 보이며, 코드 또한 공개되어 있어 실제 응용 가능성도 제시합니다.



### Large Language Model Guided Progressive Feature Alignment for Multimodal UAV Object Detection (https://arxiv.org/abs/2503.06948)
- **What's New**: 이번 연구에서 제안하는 LPANet은 Multimodal UAV object detection의 성능 개선을 위해 Large Language Model (LLM)을 활용한 Progressive feature Alignment Network입니다. 기존의 방법들이 모달리티 간의 의미적 불일치를 간과한 것과는 달리, 본 연구는 ChatGPT를 통해 생성된 세부 객체 설명을 활용하여 모달리티 간의 의미적 및 공간적 정렬을 점진적으로 진행합니다. 이 접근법을 통해 두 개의 공공 데이터셋에서 최고 성능의 멀티모달 UAV 객체 탐지기를 초과하는 결과를 보였습니다.

- **Technical Details**: LPANet은 세 가지 주요 모듈로 구성되어 있습니다. 첫째, Semantic Alignment Module (SAM)은 다양한 모달에서 추출된 의미적 특징을 근접하게 정렬하여 의미 차이를 완화합니다. 둘째, Explicit Spatial Alignment Module (ESM)은 의미 관계를 이용해 특징 레벨의 오프셋을 추정하여 모달리티 간의 공간적 정렬을 가능케 합니다. 마지막으로, Implicit Spatial Alignment Module (ISM)은 인접 영역으로부터 중요한 특징을 집계하여 암묵적인 공간 정렬을 수행합니다.

- **Performance Highlights**: Two public datasets인 DroneVehicle 및 VEDAI에서 진행된 실험에서 LPANet은 기존의 최첨단 방법을 초과하는 성능을 보여주었습니다. 구체적으로, 제안된 Semantic Alignment Module과 Explicit 및 Implicit Spatial Alignment Module의 결합은 모달리티 간의 의미적 및 공간적 불일치를 효과적으로 줄여줍니다. 통합된 프레임워크는 다양한 환경과 조건에서도 높은 검출 성능을 유지하는 특징을 가지며, 상업적 응용 가능성도 높입니다.



### Effect of Selection Format on LLM Performanc (https://arxiv.org/abs/2503.06926)
- **What's New**: 이번 논문은 대규모 언어 모델(LLM)의 성능에 중요한 요소인 분류 작업 옵션의 최적 형식에 대해 조사합니다. 우리는 많은 실험을 통해 두 가지 선택 형식인 불릿 포인트(bullet points)와 일반 영어(plain English)를 비교하여 모델 성능에 미치는 영향을 검토했습니다. 연구 결과, 불릿 포인트 형식이 일반적으로 더 나은 결과를 내는 것으로 나타났습니다.

- **Technical Details**: 이 연구는 LLM의 성능을 평가하기 위해 10개의 서로 다른 도메인별 작업을 대상으로 실험을 수행했습니다. 각 실험에서는 불릿 포인트와 일반 설명 형식으로 제시된 프롬프트를 비교하여, 이를 통해 LLM의 반응을 측정했습니다. 평가 지표로는 가중 평균 정밀도(precision), 재현율(recall), 그리고 F1 점수를 사용했습니다.

- **Performance Highlights**: 실험 결과, 불릿 포인트 형식을 사용한 경우가 일반 설명 형식을 사용할 때보다 더 높은 성능을 보인 것으로 확인되었습니다. 이 연구는 LLM의 취지에 맞게 프롬프트 설계를 최적화하는 데 기여할 수 있는 중요한 통찰을 제공할 것으로 기대됩니다. 이후 연구는 프롬프트 형식을 추가로 탐구하여 모델 성능을 더욱 개선할 방향을 모색해야 합니다.



### From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers (https://arxiv.org/abs/2503.06923)
Comments:
          13 pages, 14 figures

- **What's New**: 본 논문에서는 기존의 'cache-then-reuse' 방식을 대체하고 'cache-then-forecast'라는 새로운 패러다임을 제안합니다. 이를 통해 변화하는 시점에서의 특징을 예측할 수 있는 방법론을 개발하여, 지속적인 모델링을 통해 다가오는 시점에서의 diffusion 모델 특징을 예측할 수 있게 합니다. 이 접근법은 특징의 안정적인 궤적을 활용하여 높은 비율의 가속화를 가능하게 하며, 기존 캐싱 방식을 넘어서는 성능을 자랑합니다.

- **Technical Details**: TaylorSeer는 Taylor 급수를 이용하여 다양한 시점에서의 특징 궤적을 근사하고, 고차 도함수를 활용하여 차별화된 속도로 특징을 예측합니다. 이 방식은 캐싱된 특징을 단순히 재사용하는 것과는 달리, 특징의 변화 연속성을 활용하여 미래의 특징을 예측하는데 중점을 둡니다. 결과적으로, 추가 훈련 없이도 높은 가속비와 함께 효율적인 소스 재사용이 가능해짐을 보여줍니다.

- **Performance Highlights**: TaylorSeer는 DiT, FLUX, HunyuanVideo에서 각각 2.5배, 4.99배, 5.00배의 가속화를 달성하며, 높은 품질의 생성을 유지합니다. 성능 지표에서도 이전 상태에서 기술(SOTA)보다 36배의 품질 손실 감소를 보여주며, 6배 이상의 가속 환경에서도 효과적으로 작동합니다. 이로써 유의미한 성능 향상을 이루며, diffusion 모델의 새로운 가속 방법론으로 자리매김할 수 있습니다.



### Improving cognitive diagnostics in pathology: a deep learning approach for augmenting perceptional understanding of histopathology images (https://arxiv.org/abs/2503.06894)
- **What's New**: 이번 논문은 Digital Technologies가 Computational-Pathology 분야에서 인간의 건강, 인지, 및 인식을 증진시킬 수 있는 새로운 접근방법을 제시합니다. Vision Transformers (Vit)와 GPT-2를 결합한 멀티모달 모델을 통해 Histopathology 이미지 분석을 향상시키는 것을 목표로 하고 있습니다. 이 모델은 의료 및 학술 자원에서 파생된 Dense 이미지 캡션으로 전문화된 Arch-Dataset을 기반으로 세밀하게 조정되었습니다.

- **Technical Details**: 모델은 조직 형태, 염색 변이, 병리적 상태와 같은 Histopathology 이미지의 복잡성을 포착하기 위해 설계되었습니다. 컨텍스트에 맞는 정확한 캡션을 생성함으로써, 의료 전문가의 인지 능력을 증대시켜 질병 분류(classification), 분할(segmentation), 및 분별(detection)을 더 효율적으로 수행할 수 있도록 합니다.

- **Performance Highlights**: 이 모델은 종종 간과되는 미세한 병리적 특징들을 인지하는 능력을 향상시켜 진단 정확성을 개선합니다. 본 연구는 디지털 기술이 의료 이미지 분석에서 인간의 인지 능력을 증진시킬 수 있는 가능성을 입증하고, 보다 개인화되고 정확한 의료 결과를 향한 발걸음을 보여줍니다.



### Policy Regularization on Globally Accessible States in Cross-Dynamics Reinforcement Learning (https://arxiv.org/abs/2503.06893)
Comments:
          Preprint. Under Review

- **What's New**: 이 논문은 다양한 동적 환경에서 데이터로부터 학습하기 위한 새로운 프레임워크를 제안합니다. 기존의 Imitation Learning(모방 학습) 방식은 전문가의 상태 분포를 임의의 동적 환경에서 복원하는 것을 목표로 하지만, 환경이 변화할 경우 특정 전문가 상태에 접근할 수 없게 되어 제한이 있었습니다. 이 문제를 해결하기 위해, 논문에서는 보상 극대화(Reward Maximization)와 Imitation from Observation(IfO)를 통합하는 새로운 정책 최적화 방법을 제안합니다.

- **Technical Details**: 제안된 방식은 F-distance(특정 거리 개념)을 정규화하여 정책을 최적화합니다. 이를 통해 모든 고려된 동적 환경에서 접근 가능한 상태들에 대한 제약을 강화함으로써 접근할 수 없는 상태로 인한 문제를 완화합니다. 특히, 논문에서는 Accessible State Oriented Policy Regularization (ASOR)이라는 알고리즘을 개발하여, 다양한 강화 학습(RL) 접근 방식에 통합할 수 있도록 합니다.

- **Performance Highlights**: 다양한 벤치마크 실험을 통해 ASOR의 효과성을 입증하였으며, 기존의 교차 도메인 정책 전이 알고리즘(Cross-Domain Policy Transfer Algorithms)의 성능을 크게 향상시켰습니다. ASOR은 접근 가능한 상태(Globally Accessible States)에서 전문가의 상태 분포를 안전하게 모방할 수 있도록 설계되었습니다. 이러한 접근은 주어진 동적 환경에서 전문가 경로를 따르지 않고도 정책 최적화를 가능하게 합니다.



### Text-to-Image Diffusion Models Cannot Count, and Prompt Refinement Cannot Help (https://arxiv.org/abs/2503.06884)
- **What's New**: 이번 논문에서는 AI 커뮤니티에서 가장 중요한 문제 중 하나인 Generative modeling에 대해 다루고, text-to-image generation에서 diffusion models의 성공 사례를 소개합니다. 하지만 이 모델들이 사용자 지침의 숫자 제약을 준수하는 데 있어 근본적인 제한이 있음을 강조하고, 이에 대한 체계적인 평가가 부족하다고 지적합니다. 이러한 문제를 해결하기 위해 T2ICountBench라는 새로운 벤치마크를 제시하여, 최신 diffusion 모델의 카운팅 능력을 rigorously 평가하고자 합니다.

- **Technical Details**: T2ICountBench는 다양한 generative models를 포함하며, 이 모델들은 오픈 소스와 프라이빗 시스템을 포괄합니다. 이 벤치마크는 카운팅 성능을 다른 기능에서 분리하고, 다양한 난이도 수준을 제공하여 구조적으로 평가할 수 있게 설계되었습니다. 또한, 정확도를 높이기 위해 인적 평가(human evaluations)를 포함하여 높은 신뢰성을 보장합니다.

- **Performance Highlights**: T2ICountBench를 통한 평가 결과, 모든 최신 diffusion 모델들이 올바른 객체 수를 생성하는 데 실패하며, 객체의 수가 증가함에 따라 정확도가 현저히 감소하는 것으로 나타났습니다. 추가적으로, 프롬프트 개선(prompt refinement)에 대한 탐색적 연구 결과, 이러한 간단한 개입이 카운팅 정확도를 향상시키는 데 generally 효과적이지 않음을 보여주었습니다. 이러한 발견은 diffusion 모델 내에서 수치 이해에 대한 고유한 도전 과제를 강조하고, 향후 개선 방향에 대한 유망한 실마리를 제공합니다.



### Interactive Medical Image Analysis with Concept-based Similarity Reasoning (https://arxiv.org/abs/2503.06873)
Comments:
          Accepted CVPR2025

- **What's New**: 이번 논문에서는 Concept-based Similarity Reasoning 네트워크(CSR)를 소개합니다. 이 네트워크는 모델의 의사결정을 심층적으로 이해하고 중재할 수 있도록 설계되었습니다. CSR은 패치 수준의 프로토타입과 내재적인 개념 해석을 제공하며, 공간적인 상호작용을 통해 의사가 이미지의 특정 영역에 직접 개입할 수 있도록 합니다. 이를 통해 의료 이미지 분석에서 투명하고 직관적인 도구로 활용될 수 있습니다.

- **Technical Details**: CSR 모델은 (i) 설명 가능한 개념 특성을 추출하기 위한 Concept 모델, (ii) 개념 특성 공간을 강화하는 피쳐 프로젝터, (iii) 개념 유사성 점수를 기반으로 분류하는 과제 헤드로 구성됩니다. 입력 이미지를 받아들여 개념 점수를 예측하는 대신, CSR은 주어진 이미지와 각 개념의 프로토타입 간의 유사성 점수를 계산합니다. 이러한 유사성 점수를 활용하여 최종 분류를 지원하는 구조입니다.

- **Performance Highlights**: CSR은 세 가지 생물 의학 데이터셋에서 기존의 해석 가능한 방법들 대비 최대 4.5%의 성능 향상을 보여주었습니다. 특히, 의사들이 이미지 분석에 필요한 일관성 있는 피드백을 통해 모델의 신뢰성을 높이는 데 기여할 수 있는 기능을 강조합니다. 이러한 성능 향상은 의료 이미지 분석의 해석 가능성을 크게 개선하는 데 기여할 것으로 기대됩니다.



### Lost-in-the-Middle in Long-Text Generation: Synthetic Dataset, Evaluation Framework, and Mitigation (https://arxiv.org/abs/2503.06868)
- **What's New**: 이 논문에서는 기존의 긴 텍스트 생성 방식이 짧은 입력에서 긴 텍스트를 생성하는 것에 집중하고, 긴 입력과 긴 출력 작업을 소홀히 하고 있음을 지적합니다. 이를 해결하기 위해 Long Input and Output Benchmark (LongInOutBench)를 도입하고, 여기에는 합성 데이터셋과 포괄적인 평가 프레임워크가 포함되어 있습니다. 이러한 벤치마크는 기존의 문제를 해결하는 데 기여할 것입니다.

- **Technical Details**: 이 연구에서는 Retrieval-Augmented Long-Text Writer (RAL-Writer)를 개발하여, 중요한 내용들을 검색하고 재진술하는 방식으로 'lost-in-the-middle' 문제를 해결합니다. RAL-Writer는 명시적인 프롬프트를 구성하여 정보를 효과적으로 복원합니다. 연구팀은 LongInOutBench를 사용하여 RAL-Writer의 성능을 유사한 기준선과 비교했습니다.

- **Performance Highlights**: 실험 결과, RAL-Writer는 제안한 벤치마크에 대해 기존의 방법들보다 우수한 성능을 보였습니다. 이는 긴 입력과 긴 출력을 효과적으로 처리할 수 있는 잠재력을 보여줍니다. 최종적으로, 연구팀은 그들의 코드를 공개하여 연구자들이 이 작업을 재현하고 발전시킬 수 있도록 지원하고 있습니다.



### Graphormer-Guided Task Planning: Beyond Static Rules with LLM Safety Perception (https://arxiv.org/abs/2503.06866)
- **What's New**: 최근 대형 언어 모델(LLMs)의 발전은 로봇 작업 계획에서의 역할을 확장시켰습니다. 본 논문에서는 LLM을 이용한 실행 가능 작업 시퀀스 생성뿐만 아니라 안전한 작업 실행 보장을 위한 접근법을 제안합니다. 기존 방법들은 구조화된 위험 인식을 다루는 데 어려움을 겪고 있으며, 안전이 중요한 응용 분야에서는 저지연 위험 적응이 필요합니다.

- **Technical Details**: 우리는 Graphormer 기반의 위험 인식 작업 계획 프레임워크를 제안합니다. 이 프레임워크는 LLM 기반 의사결정과 구조화된 안전 모델링을 결합합니다. 이를 통해 동적 공간-의미적 안전 그래프를 구성하여 공간적 및 맥락적 위험 요소를 포착하고, 온라인 위험 탐지 및 적응형 작업 개선을 가능하게 합니다.

- **Performance Highlights**: AI2-THOR 환경에서 수행된 실험을 통해 우리의 프레임워크가 위험 탐지 정확도 향상, 안전 통지 증가 및 작업 적응성 개선을 달성했음을 확인했습니다. 기존의 정적 규칙 기반 방법과 LLM 전용 기준과 비교하여 우리의 접근법이 지속적인 환경에서 더 나은 성능을 보였습니다.



### Enhanced Multi-Tuple Extraction for Alloys: Integrating Pointer Networks and Augmented Attention (https://arxiv.org/abs/2503.06861)
Comments:
          17 pages, 5 figures

- **What's New**: 본 연구는 다중 주성분 합금(multi-principal-element alloys)에서 기계적 특성을 효과적으로 추출하기 위한 새로운 방법을 제안합니다. 제안된 방법은 MatSciBERT 기반의 개체 추출 모델과 포인터 네트워크(pointer networks), 그리고 상호 및 내부 개체 주의(attention) 메커니즘을 활용하는 할당 모델을 통합하고 있습니다. 이를 통해, 다양하고 복잡한 형태의 튜플 정보를 정확하게 추출할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 기계적 특성을 포함하는 다중 튜플(multi-tuple)을 추출하기 위해, 본 연구에서는 255개의 문장으로 구성된 데이터셋을 사용하여 각 문장 내 튜플 수에 따라 데이터셋을 분리하였습니다. MatSciBERT 모델과 포인터 네트워크를 활용하여 개체를 효과적으로 식별하고 데이터를 구조화합니다. 이 모델은 다수의 튜플을 포함하는 문장을 처리할 수 있도록 설계되었습니다.

- **Performance Highlights**: 모델의 성능은 1, 2, 3, 4개의 튜플 데이터셋에 대해 각각 0.963, 0.947, 0.848, 0.753의 F1 점수를 기록하였으며, 무작위로 선택된 데이터셋에서는 0.854의 F1 점수를 달성했습니다. 이러한 성과는 제안된 접근 방식이 대규모 자연어 처리 모델과 비교할 때 보다 정교하고 효과적임을 입증합니다.



### AttFC: Attention Fully-Connected Layer for Large-Scale Face Recognition with One GPU (https://arxiv.org/abs/2503.06839)
- **What's New**: 본 논문에서는 Attention Fully Connected (AttFC) 레이어를 제안하여 얼굴 인식(FL) 모델의 훈련에 필요한 계산 자원을 대폭 줄였습니다. AttFC는 생성적 클래스 센터(Generative Class Center, GCC)를 생성하기 위해 attention loader를 활용하고 역동적인 클래스 컨테이너(Dynamic Class Container, DCC)를 통해 클래스를 저장합니다. 여기서 DCC는 FC 레이어의 모든 클래스 센터 중 일부만 저장하기 때문에, 모델 파라미터 수가 대幅적으로 감소합니다.

- **Technical Details**: AttFC는 attention 메커니즘과 Momentum Contrast를 바탕으로 설계되었으며, 두 개의 인코더(Feature Encoder와 Class Encoder)를 사용하여 GCC 생성을 위한 다양한 이미지의 기여도를 계산합니다. 인식 품질에 따라 주의 가중치(attention weight)를 적용함으로써 저품질 이미지의 영향을 축소하고 고품질 이미지의 중요성을 높입니다. 결과적으로 업데이트된 GCC는 실제 클래스 센터(True Class Center, TCC)와 더욱 비슷해집니다.

- **Performance Highlights**: 실험 결과, AttFC를 활용했을 때 FD 모델은 기존의 최신 방법들과 유사한 성능을 보이면서도 훈련 자원을 크게 절감할 수 있음을 입증했습니다. 이는 대형 데이터셋을 사용할 때 계산 시간과 메모리 요구 사항을 줄여주어 하드웨어 자원의 부담을 경감시킵니다. 따라서 AttFC는 얼굴 인식에 있어 효율적인 모델 훈련을 가능하게 합니다.



### Towards a Multimodal MRI-Based Foundation Model for Multi-Level Feature Exploration in Segmentation, Molecular Subtyping, and Grading of Glioma (https://arxiv.org/abs/2503.06828)
- **What's New**: 본 논문은 비침습적인 (noninvasive) 방법으로 뇌종양 (glioma)의 정확한 특성을 분석하는 Multi-Task SWIN-UNETR (MTS-UNET) 모델을 제안합니다. 기존의 방법들이 조직 샘플링에 의존하여 종양의 공간 이질성을 포착하지 못했던 한계를 극복하고, 다중 과제를 동시에 수행하는 신기술을 도입했습니다. MTS-UNET는 대규모 뇌 영상 데이터로 사전 학습된 BrainSegFounder 모델을 기반으로 하여, 새로운 통합 모델의 가능성을 제시합니다.

- **Technical Details**: MTS-UNET는 종양의 세분화(glioma segmentation), 조직 등급(histological grading), 및 분자 세분화(molecular subtyping) 작업을 동시에 수행합니다. 이 모델은 주목할 만한 두 가지 모듈, 즉 다중 스케일 특징 추출(Tumor-Aware Feature Encoding, TAFE)와 IDH 변이와 관련된 섬세한 신호를 강조하는 교차 모달 차별 (Cross-Modality Differential, CMD)을 포함합니다. 2,249명의 다양한 뇌종양 환자 데이터로 교육 및 검증을 진행하였으며, 이를 통해 모델의 효용을 입증하였습니다.

- **Performance Highlights**: MTS-UNET는 세분화에서 평균 Dice 점수 84%를 달성했으며, IDH 변이에 대한 AUC는 90.58%, 1p/19q 동시 결실 예측에 대한 AUC는 69.22%, 등급 예측에 대해 87.54%를 기록했습니다. 이러한 결과는 기존의 기준 모델들보다 유의미하게 뛰어난 성과로, p<=0.05의 차이를 보였습니다. 구조적 기여를 통해 TAFE와 CMD 모듈의 필수성을 검증한 이번 연구는 다양한 MRI 데이터셋에서의 일반화 가능성을 강하게 보여주고, 비침습적이고 개인 맞춤형 뇌종양 관리의 발전 가능성을 제시합니다.



### Towards Fine-Grained Video Question Answering (https://arxiv.org/abs/2503.06820)
- **What's New**: MOMA-QA 데이터셋이 소개되었으며, 이는 Video Question Answering (VideoQA)의 현재 한계를 극복하기 위해 설계되었습니다. 이 데이터셋은 시간적 로컬라이제이션(temporal localization)과 공간적 관계(spatial relationship) 추론을 강조하여 보다 정교한 비디오 이해 모델 개발을 목표로 합니다. 또한, SGVLM이라는 새로운 비디오-언어 모델이 소개되어 더 나은 관계 이해 및 시간을 기반으로 한 로컬라이제이션 능력을 제공합니다.

- **Technical Details**: MOMA-QA 데이터셋은 시간 간격 주석과 진실 장면 그래프(ground truth scene graphs)를 포함하고 있어 비디오의 특정 순간과 객체 간의 관계를 이해하는 데 필요한 데이터의 깊이를 제공합니다. 데이터셋의 질문 중 71.6%는 공간적 관계 이해를 요구하며, 각각의 프레임에 대한 장면 그래프 주석이 포함되어 있습니다. SGVLM 모델은 모티프 기반(scene graph)의 장면 그래프 생성기, 효율적인 프레임 검색기, 사전 학습된 대규모 언어 모델을 통합하여 고급 추론 능력을 가지게 됩니다.

- **Performance Highlights**: MOMA-QA 및 기타 공개 데이터셋에 대한 평가 결과, SGVLM 모델이 비디오 질문 응답에서 우수한 성능을 보이며 새로운 기준을 설정했습니다. 특히, 이 모델은 비디오의 시간적 및 공간적 관계를 효과적으로 처리가능하며, 모델의 예측 결정 경로를 해석할 수 있는 데에도 기여합니다. 데이터셋의 새로운 주석 방식과 모델의 혁신이 결합되어 더 정교하고 신뢰할 수 있는 VideoQA 성능을 제공합니다.



### Semi-Supervised Medical Image Segmentation via Knowledge Mining from Large Models (https://arxiv.org/abs/2503.06816)
Comments:
          18 pages, 2 figures

- **What's New**: 본 연구는 대규모 비전 모델인 SAM(Segment Anything Model)의 폭넓은 시각적 지식을 활용하여 제한된 레이블 데이터에서 U-Net++ 모델의 성능을 향상시키는 전략적 지식 마이닝 방법을 제안합니다. 이를 통해 SAM의 출력을 활용하여 '유사 레이블(pseudo labels)'을 생성하고, 이를 통해 훈련 데이터셋을 풍부하게 하여 작은 딥러닝 모델의 성능을 극대화합니다. 연구 결과는 우리가 제안한 방법이 기존의 U-Net++ 모델을 초월하는 성과를 내었음을 보여줍니다.

- **Technical Details**:  연구에서는 Kvasir SEG와 COVID-QU-Ex 데이터셋을 활용하여 U-Net++ 모델을 제한된 레이블 데이터로 훈련한 후, SAM이 생성한 출력으로부터 유사 레이블을 추출하는 방법을 사용했습니다. 이 과정은 레이블이 없는 데이터셋에 대한 SAM의 일반화된 시각적 지식을 활용하며, U-Net++ 모델의 학습이 이를 통해 향상됩니다. SAM의 예측 결과를 개선하기 위해 iterative한 과정이 적용되며, 결과적으로 경량의 U-Net++가 높은 인퍼런스 속도를 유지하면서도 성능을 극대화합니다.

- **Performance Highlights**: 우리가 제안한 방법은 Kvasir SEG와 COVID-QU-Ex 데이터셋에서 각각 3%와 1% 성능 향상을 이루어냈습니다. 본 연구는 100% 레이블 데이터로만 훈련된 기존 U-Net++ 모델과 비교했을 때도 제안한 방법이 우수한 성과를 보였다고 강조합니다. 이러한 결과는 대규모 모델인 SAM의 폭넓은 지식을 활용한 지식 마이닝이 데이터 제한 사항을 극복할 수 있음을 보여줍니다.



### Unlocking Generalization for Robotics via Modularity and Sca (https://arxiv.org/abs/2503.06814)
Comments:
          CMU Robotics PhD Thesis, 185 pages

- **What's New**: 본 논문에서는 일반ist 로봇 시스템을 구축하는 방법을 제시합니다. 모듈성과 대규모 학습을 통합하여 여러 로봇 제어 작업을 수행할 수 있는 에이전트를 개발하는 것을 목표로 합니다. 특히, 각 모듈의 독립적인 일반화 능력을 활용해 효율적인 로봇 학습자를 만드는 방법에 대해 논의합니다.

- **Technical Details**: 논문에서는 로봇 학습 시스템에 모듈성과 계층 구조를 구축하는 데 초점을 맞춥니다. 모듈화를 계획을 통해 시행함으로써 에이전트가 더욱 효과적이고 능력 있는 로봇 학습자가 될 수 있도록 합니다. 대규모 데이터와 다양한 아키텍처, 감독 소스를 확보하기 위해 클래식 계획(classical planning)을 활용하여 시뮬레이션 내에서 정책 학습을 감독합니다.

- **Performance Highlights**: 결국, 모듈성과 대규모 정책 학습을 통합하여 제로샷 조작(zero-shot manipulation)을 성공적으로 수행할 수 있는 실제 로봇 시스템을 구축하는 방법을 제시합니다. 이 접근법을 통해 하나의 일반ist 에이전트가 실제 환경에서 복잡한 장기 조작 작업을 해결할 수 있음을 보여줍니다.



### Can Proof Assistants Verify Multi-Agent Systems? (https://arxiv.org/abs/2503.06812)
- **What's New**: 이 논문은 다중 에이전트 시스템(Multi-Agent Systems, MAS) 검증을 위한 Soda 언어를 제안합니다. Soda는 고수준의 함수형 및 객체 지향 언어로서, Scala와 Lean로 컴파일할 수 있는 기능을 제공합니다. 이러한 능력을 통해 Soda는 다중 에이전트 시스템을 구현하고, 최신 도구를 사용하여 형식적으로 검증할 수 있는 부분을 포함합니다.

- **Technical Details**: Soda는 정적 타이핑이 있는 함수형 언어로, 선언적이고 읽기 쉬운 형태를 목표로 합니다. Lean과 같은 순수 함수형 언어의 엄격함과 Scala의 현대적 기술 통합을 결합하여, JVM 생태계와의 연결이 용이합니다. 이를 통해 각종 트랜잭션을 표현하며 Lean 증명을 통합할 수 있는 기능을 제공합니다.

- **Performance Highlights**: Soda를 사용한 간단한 상호작용 프로토콜 검증 예제를 통해 이론적 가능성을 입증하였습니다. 이 예제에서는 판매자, 구매자, 중개자 간의 안전한 거래를 구현하고, 리스트에서 항목을 변경하는 것이 리스트의 크기에 영향을 미치지 않는다는 중요한 속성을 증명합니다. 이 결과는 간단해 보일 수 있지만, Lean과 같은 증명 도구에서의 증명은 상당한 노력을 요구하며, 실제 적용 가능성을 강조하는 데 기여합니다.



### Mitigating Preference Hacking in Policy Optimization with Pessimism (https://arxiv.org/abs/2503.06810)
- **What's New**: 이 논문은 인간 피드백으로부터의 강화 학습(RLHF)에서 과최적화(overoptimization) 문제를 다루고 있습니다. RLHF는 모델을 인간의 선호에 맞추기 위한 유망한 기술로, 제한된 선호 데이터로 훈련된 보상 모델이 일반화에 실패할 수 있음을 지적합니다. 이를 해결하기 위해 새로운 비관적 목표(pessimistic objectives)를 제안하며, 이를 통해 RLHF의 과최적화에 대해 견고성을 입증할 수 있습니다.

- **Technical Details**: 논문은 P3O 및 PRPO라는 두 가지 실용적인 알고리즘을 설계하여 비관적 목표를 최적화합니다. 이 알고리즘은 언어 모델을 문서 요약 및 유용한 도우미 생성을 위해 미세 조정하는 작업에 대해 평가됩니다. 보상 및 선호 모델의 불확실성을 처리하는 새로운 제약이 있는 Nash 형식을 개발하고, 이론적 이점을 제시합니다. P3O는 변분(variational) 상한을 통해 최적화되며, 정책과 선호 플레이어 간의 대립적(minimax) 게임을 해결하는 과정을 포함합니다.

- **Performance Highlights**: 실험 결과 P3O와 PRPO는 과최적화 없이 높은 품질의 응답을 빠르게 달성하며, 기존 전통적인 RLHF 방법들과 비교하여 월등한 성능을 보입니다. 특히 DPO 및 REINFORCE와 같은 기존 방법은 과최적화 문제를 보이는 반면, P3O와 PRPO는 이러한 질적 보상 해킹 행동을 피하는 데 성공했습니다. 문서 요약 및 도움을 주는 도우미 모델 훈련에서의 성공적인 결과를 통해 이 연구의 중요성을 강조합니다.



### Privacy Auditing of Large Language Models (https://arxiv.org/abs/2503.06808)
Comments:
          ICLR 2025

- **What's New**: 이번 연구에서는 현재 대형 언어 모델(LLM)의 프라이버시 감사를 위한 기존 기술의 한계를 극복하고, 더 효과적인 canary를 개발하여 이를 개선하고자 하였습니다. 저자들은 여러 가지 현실적인 위협 모델에 대해 새로운 canary 설계를 통한 프라이버시 누수 감지의 새로운 기준을 세웠습니다. 이는 LLM의 훈련 데이터로부터 수집된 메모리 정보에 대한 공격을 이해하는 데 중요한 기여를 합니다.

- **Technical Details**: 연구의 주요 초점은 LLM 훈련 중 사용할 수 있는 가장 쉽게 기억되는 canaries를 설계하는 것입니다. 저자들은 실험을 통해 설계된 canaries가 비공식적으로 훈련된 LLM에서의 기억률을 측정하는 데 있어 기존 방식보다 훨씬 우수하다는 것을 보여주었습니다. 이 연구는 비공식적인 LLM 훈련에서 ε ≈ 1의 프라이버시 감사를 달성했으며, 이는 기본적인 ε 값 4의 모델에서 이루어졌습니다.

- **Performance Highlights**: 실험 결과, Qwen2.5-0.5B 모델에서 저자들이 개발한 canaries는 1%의 거짓 긍정률(FPR)에서 49.6%의 진정한 긍정률(TPR)을 달성하여 기존 방법의 4.2%를 획기적으로 능가하였습니다. 이 연구는 LLM의 잠재적인 프라이버시 누수에 대한 체계적인 이해와 이를 통한 보다 강력한 감사 방법론을 제시하며, 프라이버시 침해를 최소화할 수 있는 기틀을 마련합니다.



### Actionable AI: Enabling Non Experts to Understand and Configure AI Systems (https://arxiv.org/abs/2503.06803)
- **What's New**: 이 논문은 비전문가가 AI 시스템을 조작하고 구성할 수 있도록 하는 'Actionable AI' 프레임워크를 제안합니다. 실험에서는 비전문가들이 AI 추천 시스템인 카트폴(cartpole) 게임을 통해 불확실한 환경 속에서 성공적으로 목표를 달성할 수 있었다는 점이 중요한 발견입니다. 이로 인해 AI 시스템과의 상호작용에 있어 새로운 접근 방식을 제공하며, 사용자의 적극적인 참여를 유도할 수 있습니다.

- **Technical Details**: 이번 연구에서는 실시간 전략 비디오 게임 환경에서 Actionable AI 개념을 실험했습니다. 22쌍의 참가자들이 사전 정보 없이 카트폴 게임을 진행했으며, AI의 통제 아래에서 자신의 전략을 구성하고 실행하는 방식으로 진행되었습니다. 참가자들은 AI의 자율성을 완전히 이해하지 못했음에도 불구하고 설정된 목표를 달성하는 데 성공했으며, 행동 조정 능력을 발휘했습니다.

- **Performance Highlights**: 연구 결과 22개 팀 중 14개 팀이 AI가 제어하는 카트폴 게임에서 좋은 성과를 달성했습니다. 참가자들은 본인의 행동과 선택을 통해 AI의 행동을 구성하고, 게임에서 이기는 전략을 개발하는 과정에서 운영 지식을 얻었습니다. 이는 Actionable AI 시스템이 비전문가의 AI 조작 능력을 효과적으로 지원함을 보여주며, 이를 통해 AI와의 원활한 협업이 가능함을 입증하였습니다.



### Characterizing Learning in Spiking Neural Networks with Astrocyte-Like Units (https://arxiv.org/abs/2503.06798)
Comments:
          6 pages, 4 figures

- **What's New**: 이번 연구에서는 전통적인 인공 신경망(artificial neural networks)에 아스트로사이트(astrocyte) 유사 단위를 추가한 수정된 스파이킹(spiking) 신경망 모델을 도입했습니다. 아스트로사이트는 계산을 수행하는 데 중요한 역할을 할 수 있다는 점에서 주목받고 있으며, 이러한 단위들이 신경망의 학습(processing)에 미치는 영향을 평가하였습니다.

- **Technical Details**: 연구팀은 해당 신경망을 액체 상태 기계(liquid state machine)로 구현하고, 혼돈적인 시간 시계열 예측(chaotic time-series prediction) 작업을 수행하게 했습니다. 이때 신경 유사 단위와 아스트로사이트 유사 단위의 수와 비율을 조절하여 아스트로사이트 단위가 학습에 미치는 영향을 분석했습니다.

- **Performance Highlights**: 신경 유사 단위와 아스트로사이트 유사 단위를 함께 결합했을 때 학습을 유도하는 데 매우 중요한 것으로 나타났습니다. 흥미롭게도 아스트로사이트 유사 단위와 신경 유사 단위의 비율이 약 2:1일 때 최고 학습률을 기록하였으며, 이는 생물학적 아스트로사이트와 뉴런(neurons)의 비율 추정과 유사합니다. 이러한 결과는 아스트로사이트 유사 단위를 포함시키는 것이 신경망의 학습률을 변경할 수 있음을 보여줍니다.



### Multimodal AI-driven Biomarker for Early Detection of Cancer Cachexia (https://arxiv.org/abs/2503.06797)
Comments:
          17 pages, 6 figures, 3 Tables

- **What's New**: 이번 연구는 암 카켁시아(cancer cachexia) 조기 발견을 위한 다중 모달 AI 기반 바이오마커(biomarker)를 제안합니다. 이 방법은 개방형 대규모 언어 모델(open-source large language models)과 의료 데이터에 대해 훈련된 기초 모델(foundational models)을 활용하여, 다양한 환자 데이터를 통합합니다.

- **Technical Details**: 연구에 사용된 기계 학습(machine learning) 프레임워크는 인구통계학(demographics), 질병 상태, 실험실 보고서(lab reports), 방사선 영상(radiological imaging, CT 스캔 포함) 및 임상 노트(clinical notes)와 같은 이질적인 데이터를 처리할 수 있습니다. 기계 학습 모델은 결측 데이터를 처리하는 능력을 갖추고 있으며, 이를 통해 실시간으로 수집된 임상 데이터를 사용합니다.

- **Performance Highlights**: 예비 결과에 따르면, 여러 데이터 모달리티(multiple data modalities)를 통합하는 것이 암 진단 시 카켁시아 예측 정확도를 개선하는 데 도움이 됩니다. 이 AI 기반 바이오마커는 나이, 인종, 민족, 체중, 암 유형 및 단계와 같은 환자 맞춤형 요인에 동적으로 적응하여 고정 임계값 바이오마커의 한계를 극복합니다.



### AutoMisty: A Multi-Agent LLM Framework for Automated Code Generation in the Misty Social Robo (https://arxiv.org/abs/2503.06791)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)을 활용하여 자연어 지시사항에 기반한 Misty 로봇 코드 생성을 지원하는 최초의 협업 프레임워크인 AutoMisty를 소개합니다. AutoMisty는 복잡한 작업을 자동으로 계획하고, 할당하며, 코드 생성을 수행하여 사용자 선호에 적응합니다. 특히, 이 시스템은 비전문가도 쉽게 사용할 수 있는 방법으로 사회적 로봇의 프로그래밍을 혁신하고자 합니다.

- **Technical Details**: AutoMisty는 작업 분해(task decomposition), 할당(assignment), 문제 해결(problem-solving), 결과 합성(result synthesis) 등의 네 가지 전문화된 에이전트 모듈을 포함하고 있습니다. 각 에이전트는 반복적 개선을 위한 자기 성찰(self-reflection) 및 사용자 피드백을 통해 지속적으로 적응하는 두 가지 레이어 최적화 메커니즘을 통합합니다. 이러한 구조적 접근을 통해 사용자는 자연어 피드백을 통해 작업을 점진적으로 개선할 수 있습니다.

- **Performance Highlights**: AutoMisty는 28개의 벤치마크 작업셋을 통해 다양한 복잡성 수준을 평가하고 일관된 고품질 코드를 생성하는 능력을 입증했습니다. 실험 결과, AutoMisty는 ChatGPT-4o 및 ChatGPT-o1와의 직접적인 비교에서 훨씬 뛰어난 성능을 보여주었습니다. 최적화된 API와 실험 비디오는 공개될 예정이며, 이는 AutoMisty의 효과성을 널리 알리는 데 기여할 것입니다.



### GenDR: Lightning Generative Detail Restorator (https://arxiv.org/abs/2503.06790)
- **What's New**: 최근 연구는 텍스트-이미지(T2I) 확산 모델을 실제 초해상도(SR)에 적용하여 놀라운 성과를 도출했습니다. 그러나 T2I와 SR 목표 간의 근본적인 불일치는 추론 속도와 세부 사항의 충실도 간의 딜레마를 야기합니다. 본 논문에서는 이러한 간극을 해소하기 위해 1단계 확산 모델인 GenDR을 제시하며, 이는 맞춤형 확산 모델에서 증류되어 큰 잠재 공간을 활용합니다.

- **Technical Details**: GenDR은 고도화된 잠재 공간을 확보하기 위해 16채널 기반 모델 SD2.1-VAE16을 사용합니다. 기존 방법보다 적은 추론 단계로 세부 사항 복원을 가능케 하기 위해 꾸준한 스코어 아이덴티티 증류(CiD) 기법을 적용했습니다. 게다가, CiD를 적대적 학습 및 표현 정렬(CiDA)로 확장하여 훈련을 가속화하고 지각 품질을 향상시킵니다.

- **Performance Highlights**: 실험 결과 GenDR은 정량적 메트릭과 시각적 충실도에서 최첨단 성능을 달성했습니다. 특히, 기존 모델들과 비교해 목표 품질 및 효율성 메트릭에서 큰 향상을 보이며, 주관적인 시각 비교 및 인간 평가에서도 긍정적인 결과를 얻었습니다. GenDR의 간소화된 파이프라인은 효율적인 배포를 위해 VAE와 UNet만으로 구성되어 있습니다.



### Infinite Leagues Under the Sea: Photorealistic 3D Underwater Terrain Generation by Latent Fractal Diffusion Models (https://arxiv.org/abs/2503.06784)
Comments:
          10 pages

- **What's New**: 이 논문은 해양 3D 지형의 표현 생성을 다루며, DreamSea라는 새로운 생성 모델을 소개합니다. DreamSea는 해양 로봇 조사에서 수집된 실제 이미지 데이터베이스를 기반으로 훈련되어, 매우 사실적인 해양 장면을 생성할 수 있도록 설계되었습니다. 이 모델은 노이즈와 아티팩트를 포함하는 실제 해저 이미지를 활용해, 고품질의 RGBD 기반 이미지를 생성합니다.

- **Technical Details**: DreamSea는 시각적 기초 모델을 사용하여 데이터에서 3D 기하학 및 의미 정보를 추출하고, 새로운 분산(latent embedding) 기반의 방식으로 생성된 이미지를 3D 맵으로 융합합니다. 훈련에는 깊이 센서 및 LiDAR와 같은 3D 스캔 정보가 없이 RGB 이미지만 사용되며, 이는 생성된 장면들이 공간적으로 일관된 기하 구조를 가지도록 합니다. 이러한 혁신적인 접근 방식은 데이터가 주석이 없는 경우에도 효과적으로 작동합니다.

- **Performance Highlights**: DreamSea는 대규모 해양 장면을 강력하게 생성할 수 있는 능력을 입증하였으며, 제작된 장면은 일관성과 다양성, 고품질 화풍이 특징입니다. 이 연구는 영화 제작, 게임 및 로봇 시뮬레이션과 같은 여러 분야에 걸쳐 영향을 미치며, 수중 환경의 3D 시뮬레이션 가능성을 크게 확장합니다.



### Dr Genre: Reinforcement Learning from Decoupled LLM Feedback for Generic Text Rewriting (https://arxiv.org/abs/2503.06781)
Comments:
          29 pages, 4 figures, 25 tables

- **What's New**: 이번 연구에서는 다양한 재작성 목표를 처리할 수 있는 일반적인 모델을 도입합니다. 특히 사실 수정(factuality), 스타일 변환(style transfer), 대화형 재작성(conversational rewriting)에 능숙한 모델을 개발하였습니다. 이를 위해, 자연스러운 지침을 제공하는 대화형 데이터셋인 ChatRewrite를 구성하였습니다. 또한 Dr Genre라는 새로운 프레임워크를 통해 성능 개선을 위한 목표 지향 보상 모델을 제안합니다.

- **Technical Details**: 이 모델은 다양한 NLP 작업에서 요구되는 재작성 능력을 통합하기 위해 세 가지 주요 목표를 분리하여 설정합니다. 목표는 '합의'(agreement), '일관성'(coherence), 그리고 '간결성'(conciseness)으로 정의되며, 각 목표는 재작성의 품질 향상을 위해 밀접하게 연결되어 있습니다. Dr Genre는 이를 기반으로 한 세밀한 조정을 통해 작업 요구 사항에 따라 목표 지향 보상 모델의 가중치를 조정할 수 있는 능력을 제공합니다.

- **Performance Highlights**: 실험 결과, Dr Genre는 모든 목표 지정 작업에서 높은 품질의 재작성을 제공하며, 지침 준수, 내적 일관성 및 불필요한 수정 최소화와 같은 여러 목표 개선을 나타냅니다. 특히, 사용자 요구에 맞춘 대화형 재작성 작업에서 두드러진 성과를 보였으며, 이는 사용자 적용 가능성을 크게 향상시킵니다.



### Large Language Models Are Effective Human Annotation Assistants, But Not Good Independent Annotators (https://arxiv.org/abs/2503.06778)
Comments:
          9 pages, 4 figures

- **What's New**: 이번 연구는 이벤트 주석(annotation)이 시장 변화 및 사회적 트렌드를 이해하는 데 얼마나 중요한지를 강조합니다. 기존의 전문가 주석이 효율성이 떨어지는 점을 해결하기 위해 LLM(대형 언어 모델) 기반 자동 주석을 도입하여 변수 주석의 시간과 정신적 노력을 줄일 수 있는 가능성을 보여줍니다.

- **Technical Details**: 이 연구는 복합적인 워크플로를 평가하여 관련 없는 문서를 제거하고, 동일한 이벤트에 대한 문서를 병합하며, 이벤트를 주석 처리하는 방법을 제시합니다. 특히 LLM이 전문가 주석자에게 지원이 될 수 있는 방법을 모델링하여, 문서의 정보 추출 과정을 보다 효율적으로 만들 수 있습니다.

- **Performance Highlights**: 이벤트 주석 과정에서 LLM 기반 시스템이 기존의 TF-IDF 방법이나 이벤트 세트 큐레이션(Event Set Curation)보다 우수한 성능을 보임을 나타냅니다. 그러나 LLM이 완전히 자동화된 주석자보다는 인간 전문가와의 협업을 통해 더욱 향상되는 성과를 거두게 됩니다.



### Effectiveness of Zero-shot-CoT in Japanese Prompts (https://arxiv.org/abs/2503.06765)
Comments:
          NLP2025 Workshop on Japanese Language Resources (JLR2025)

- **What's New**: 이 연구는 일본어와 영어에서의 zero-shot Chain-of-Thought (CoT) 프로팅의 효과성을 비교합니다. 요약하자면, zero-shot CoT는 문제 해결 전에 "단계별로 생각해봅시다"라는 문구를 추가하여 사고를 촉진하는 기술입니다. 그 결과, 일본어에서의 성과 일부 카테고리에서는 개선이 나타났지만, 더 진보된 모델인 GPT-4o-mini에서는 오히려 상당한 성능 저하가 발생했습니다.

- **Technical Details**: 이 연구는 일본어와 영어에서의 zero-shot CoT 프로팅을 분석하기 위해 Multi-task Language Understanding Benchmark (MMLU)와 일본어 버전인 JMMLU를 사용했습니다. GPT-3.5에서는 일부 프롬프트 카테고리에서 성과 향상이 관찰되었으나, GPT-4o-mini에서는 전반적으로 부정적인 영향을 미쳤습니다. 각 모델의 성과 변화를 이해하기 위해 다양한 주제 영역에서 데이터 분석을 수행했습니다.

- **Performance Highlights**: 전반적인 성과 분석 결과, CoT 프롬프트가 적용된 경우 두 언어 모두에서 부정적인 영향을 미쳤습니다. 특히 일본어 프롬프트의 경우 초등 수학에서 가장 큰 개선이 있었으나, 국제법과 같은 특정 과목에서는 오히려 성능이 저하되었습니다. 통계 분석 결과, 일본어에서 CoT 노출이 유의미한 영향을 미친 것으로 나타난 반면, 영어에서는 그 효과가 더욱 변동성이 있었습니다.



### SemHiTok: A Unified Image Tokenizer via Semantic-Guided Hierarchical Codebook for Multimodal Understanding and Generation (https://arxiv.org/abs/2503.06764)
Comments:
          Under Review

- **What's New**: 최근 SemHiTok의 개발에 따라 시각 정보의 이해 및 생성 작업을 위한 통합된 이미지 토크나이저가 제시되었습니다. 이 모델은 Semantic-Guided Hierarchical codebook을 통해 텍스처와 의미 특징을 동시에 잘 추출할 수 있도록 설계되었습니다. 기존의 접근 방식들이 직면했던 시맨틱(feature)과 텍스처(feature) 간의 균형 잡기 문제를 해결하기 위해, SemHiTok은 각각의 작업에 최적화된 단계별 학습을 가능하게 합니다.

- **Technical Details**: SemHiTok에서 사용되는 Semantic-Guided Hierarchical codebook은 두 가지 가지 기능을 갖추고 있습니다. 첫째로, 이는 고급 시맨틱 기능과 세부 텍스처 정보를 분리하여 학습하게 하여 가장 적합한 양의 정보만을 코드화할 수 있게 합니다. 둘째로, 이는 전이 학습 기법을 통해 더 나은 성능을 발휘하며, 기존 방안에 비해 학습의 복잡성을 줄이는데 도움을 줍니다.

- **Performance Highlights**: SemHiTok은 기존의 통합 토크나이저와 비교하여 256×256 해상도에서 state-of-the-art의 rFID 점수를 달성하였습니다. 또한, 다중 모달 이해 및 생성 작업에서 경쟁력 있는 성능을 보이며, 텍스트에서 이미지 생성 작업에서도 우수한 성과를 나타내는 등의 뛰어난 성능을 자랑합니다. 이러한 결과는 SemHiTok의 절충된 텍스처 및 의미 기능 추출 능력이 다중 모달 대형 모델에 적용될 가능성을 극대화함을 보여줍니다.



### Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models (https://arxiv.org/abs/2503.06749)
- **What's New**: DeepSeek-R1-Zero는 강화 학습(Reinforcement Learning, RL)만으로 LLM의 추리 능력을 성공적으로 증명했습니다. 이에 영감을 받아, 이번 연구에서는 MLLM의 추리 능력을 향상시키기 위해 RL을 어떻게 활용할 수 있을지 탐구합니다. 다만, MLLMs에서 직접적인 RL 학습은 고품질 다중 모달 추리 데이터의 부족으로 인해 복잡한 추리 능력을 이끌어내기에는 한계를 겪고 있습니다. 이를 극복하기 위해 Vision-R1이라는 추리 MLLM을 제안합니다.

- **Technical Details**: 이번 연구에서는 기존 MLLM과 DeepSeek-R1을 활용하여 사람의 주석 없이도 20만 개의 고품질 다중 모달 CoT(Coach-of-Thought) 데이터셋, Vision-R1-cold 데이터셋을 구축합니다. Vision-R1은 초기 데이터 셋을 통해 추리 능력을 높이는 한편, Progressive Thinking Suppression Training (PTST) 전략을 도입하여 모델의 올바른 추리 과정을 학습할 수 있도록 돕습니다. 특히, Group Relative Policy Optimization (GRPO) 기법과 하드 포맷 결과 보상 함수를 사용하여 10K 다중 모달 수학 데이터셋에서 모델의 성능을 점진적으로 개선합니다.

- **Performance Highlights**: 포괄적인 실험 결과, Vision-R1은 다양한 다중 모달 수학 추리 벤치마크에서 평균 6% 개선된 성능을 보였습니다. 특히, Vision-R1-7B는 MathVista 벤치마크에서 73.5% 정확도를 기록하며, 최고의 추리 모델인 OpenAI O1과 오직 0.4% 차이로 성능을 발휘했습니다. 이번 연구에서 제안한 데이터셋과 코드도 공개되어 연구자들 간의 상호 작용을 촉진할 것으로 기대됩니다.



### Fully-Decentralized MADDPG with Networked Agents (https://arxiv.org/abs/2503.06747)
- **What's New**: 이 논문에서는 다중 에이전트 강화 학습(MARL)에서 협력적, 적대적 및 혼합 환경을 위한 세 가지 액터-크리틱 알고리즘을 제안합니다. 이 알고리즘은 연속 액션 공간에서의 훈련을 분산하여 수행할 수 있도록 MADDPG 알고리즘을 개선하였으며, 에이전트 간의 네트워크 통신 방식을 적용하였습니다. 우리는 준비된 대리 정책(surrogate policies)을 도입하여 훈련을 분산시키면서도 훈련 중 지역적으로 통신할 수 있도록 하였습니다.

- **Technical Details**: 기존의 MADDPG 알고리즘을 활용하여 완전 분산 버전을 개발하였으며, 훈련 중 정보 교환을 위한 통신 네트워크도 도입하였습니다. 우리의 알고리즘은 먼저 협력적 설정에서 개발되고, 이후 혼합 및 적대적 설정으로 적응되며 실험은 다중 입자 환경(multi-particle environment, MPE)을 사용하여 진행되었습니다. 중앙 집중식 훈련이 대규모 에이전트나 제한된 통신 환경에서 비실용적일 수 있기 때문에, 완전하게 분산된 MARL 알고리즘을 개발하는 것을 목표로 하였습니다.

- **Performance Highlights**: 제안한 분산 알고리즘들은 실험적 테스트를 통해 원래 MADDPG와 유사한 성과를 달성하면서 계산 비용을 줄이는 데 성공하였습니다. 특히 대규모 에이전트 수에서 그 효과가 더욱 두드러지는 것으로 나타났습니다. 이를 통해 우리는 훈련 및 실행 모두에서 지역 관찰만 이용하여 작동할 수 있는 알고리즘을 개발함으로써 기존의 제한사항을 극복하였습니다.



### Gender Encoding Patterns in Pretrained Language Model Representations (https://arxiv.org/abs/2503.06734)
Comments:
          Proceedings of the 5th Workshop on Trustworthy Natural Language Processing (TrustNLP 2025)

- **What's New**: 이 연구는 미리 훈련된 언어 모델(PLMs)에서 성별 편향이 어떻게 형성되고 전파되는지를 정보 이론적 접근법을 통해 분석합니다. 특히, PLM의 내부 표현에서 성별 정보와 편향이 어떻게 인코딩되는지를 탐구하며, 기존의 디바이징 기법의 효과를 검토합니다. 그 결과, 다양한 모델에서 성별 인코딩의 일관된 패턴이 발견되었고, 일부 디바이징 기법은 오히려 내부 표현에서 편향을 증가시키는 경향을 보였습니다.

- **Technical Details**: 연구에서는 Minimum Description Length (MDL) 프로빙 기법을 사용하여 다양한 인코더 기반 아키텍처에서 성별 편향이 어떻게 인코딩되는지를 분석합니다. PLM의 여러 층을 검사함으로써 편향이 가장 두드러지게 나타나는 층을 규명하고, 사전 훈련된 디바이징 목표가 포스트 호크(post-hoc) 완화 접근법보다 인코딩된 편향을 줄이는 데 더 효과적임을 입증합니다. 이는 디바이징 기법이 모델의 편향된 대표성을 다루는 데 있어 중요한 통찰을 제공합니다.

- **Performance Highlights**: 이 연구는 PLM의 다양한 아키텍처 간에 성별 인코딩의 지속적인 패턴을 드러냅니다. 또한, 디바이징 기법의 효과성에 대한 통찰을 제공하며, 많은 경우 이러한 기법들이 오히려 내부 표현에서의 편향을 증가시키는 결과를 초래함을 지적합니다. 이를 통해 편향 완화 전략을 개발하는 데 있어 가치 있는 지침을 제공합니다.



### ACAI for SBOs: AI Co-creation for Advertising and Inspiration for Small Business Owners (https://arxiv.org/abs/2503.06729)
- **What's New**: 이 논문은 중소기업 소유자(SBO, Small Business Owners)가 광고 제작에서 겪는 문제를 해결하기 위해 ACAI (AI Co-Creation for Advertising and Inspiration)라는 GenAI 기반의 광고 생성 도구를 개발했습니다. 연구 결과, 구조화된 입력 방식이 사용자의 자율성과 통제를 높이며, AI 결과물의 품질을 개선하는 데 도움을 주는 것으로 나타났습니다. 또한, ACAI의 다중 모달 인터페이스가 디자인 전문 용어에 익숙하지 않은 SBO들을 위한 접근성을 높인다는 점이 새롭게 제시되었습니다.

- **Technical Details**: ACAI는 브랜드 자산을 중앙 집중화한 통합 시각적 인터페이스를 제공하여 중소기업 소유자들이 광고 제작을 도울 수 있도록 디자인되었습니다. 사용자가 입력하는 색상 팔레트, 글꼴, 브랜드 가치와 같은 요소를 위한 미리 정의된 입력 필드를 통해, 구조화된 입력 메커니즘을 활용하여 AI가 더 정확하게 브랜드 정체성을 해석할 수 있도록 지원합니다. 이 논문에서는 이러한 구조화된 입력과 다중 모달 프롬프트 기능이 광고 제작 과정에서 임시 사용자들에게 어떻게 도움을 주는지를 설명합니다.

- **Performance Highlights**: 연구 과정에서 ACAI는 사용자가 광고 제작 과정에서 겪는 인지적 부담을 줄이기 위해 '슈퍼 프롬프트'를 생성하도록 지원하며, 사용자가 이미지에서 텍스트 프롬프트를 추출할 수 있게 합니다. 이러한 기능은 디자인 초보자들이 AI 도구에 쉽게 접근할 수 있도록 하여 광고 결과물이 각 중소기업의 독특한 특성을 진정으로 반영하도록 하는 경과를 보여줍니다. 이 연구는 AI 매개 디자인 도구의 공동 창의적 속성을 발전시키기 위한 디자인 권고안을 제공하여 중소기업 소유자와 같은 초보 사용자들을 더욱 잘 지원할 수 있도록 하고 있습니다.



### Pull-Based Query Scheduling for Goal-Oriented Semantic Communication (https://arxiv.org/abs/2503.06725)
Comments:
          Submitted for possible publication

- **What's New**: 본 논문에서는 pull-based status update 시스템에서 목표 지향적인 semantic communication을 위한 query scheduling을 다룹니다. 여러 sensing agents (SAs)가 다양한 속성을 가진 정보를 수집하고, 이를 여러 actuation agents (AAs)에 전달하여 각각의 목표를 수행합니다. 업데이트의 의미적 가치를 정량화하기 위해 grade of effectiveness (GoE) 메트릭을 도입하였고, 누적 관점 이론 (CPT)을 통합하여 시스템의 장기 효과성을 분석합니다. 이러한 접근 방식을 통해 비용 제약을 준수하면서 총 GoE의 기대할인합을 극대화하기 위한 정책을 도출하였습니다.

- **Technical Details**: 논문에서는 다양한 속성을 가진 출처에서 정보를 업데이트하는 pull-based end-to-end 상태 업데이트 시스템을 고려하고 있습니다. 이 시스템은 주기적으로 어떤 속성을 조회할지를 결정하는 허브를 통해 업데이트를 라우팅합니다. 우리는 업데이트의 효과성을 평가하기 위해 freshness와 usefulness라는 두 가지 key semantic metrics를 사용하고 있으며, 기존의 기대 효용 이론 (EUT)에서 벗어나 CPT를 활용하여 장기 효과성을 정의합니다. 이를 통해 동적 프로그래밍 (dynamic programming) 기반의 모델 기반 솔루션과 딥 강화 학습 (deep reinforcement learning, DRL) 알고리즘을 이용한 모델 비기반 솔루션 두 가지 접근 방식을 제안하였습니다.

- **Performance Highlights**: 제안된 효과 지향적인 query scheduling 정책을 통해 기존의 벤치마크 방법에 비해 통신된 업데이트의 장기 효과성과 시스템의 신뢰성을 크게 향상시켰습니다. 특히, 엄격한 비용 제약이 있는 시나리오에서 가장 두드러진 성과를 보였으며, 모델 비기반 접근은 복잡한 실제 응용 프로그램에 보다 적합한 확장성을 발휘하는 것으로 나타났습니다.



### Delusions of Large Language Models (https://arxiv.org/abs/2503.06709)
- **What's New**: 이번 연구에서는 LLM(대형 언어 모델)에서 흔히 발생하는 사실과 상관없이 그럴듯한 출력을 생성하는 'hallucination'과는 달리, LLM delusion이라는 더 심각한 현상을 밝혀냈습니다. LLM delusion은 비정상적으로 높은 확신을 가진 잘못된 출력을 의미하며, 이러한 잘못된 내용은 낮은 불확실성으로 지속되어 모델의 신뢰성을 떨어뜨립니다. 이는 검출 및 완화가 특히 어려운 도전과제가 됩니다.

- **Technical Details**: 연구에서 사용된 방법론으로는 세 가지 불확실성 추정 기술이 있습니다: logit 기반 방법으로 토큰의 확률 분포를 평가하고, 언어화된 신뢰도에서는 모델이 자신의 신념을 명시적으로 표현하며, 일관성 기반 방법은 여러 샘플의 응답 안정성을 평가합니다. LLM delusion은 특정 신념 임계점을 초과하는 잘못된 응답으로 정의되며, 이는 전체 데이터셋에서 정답에 대한 평균 신뢰도로 empirically 결정됩니다.

- **Performance Highlights**: 연구 결과, LLM은 hallucination에 비해 delusion에서 사실 honesty가 낮고, fine-tuning이나 self-reflection을 통해 덜 수정됩니다. 특히, 잘못된 응답을 거부하도록 훈련해도 delusion은 여전히 높은 비율로 지속되고, 생성된 응답을 재고하도록 요청했을 때 오히려 delusional 출력을 계속 고수하는 경향이 있음을 확인했습니다. 외부 검증 방법인 retrieval-augmented generation 및 다중 에이전트 토론 시스템이 delusion을 줄이는 데 도움을 줄 수 있지만, 완전한 제거에는 여전히 많은 도전과제가 남아 있습니다.



### PFDial: A Structured Dialogue Instruction Fine-tuning Method Based on UML Flowcharts (https://arxiv.org/abs/2503.06706)
- **What's New**: 본 연구에서는 Process Flow Dialogue (PFDial) 데이터셋을 구축하여 고객 서비스 및 장비 유지보수에서 중요한 역할을 하는 프로세스 기반 대화 시스템의 성능을 향상시키는 것을 목표로 합니다. 이 데이터셋은 440개의 플로우차트에서 유래된 12,705개의 고품질 중국어 대화 명령어를 포함하고 있으며, 이는 5,055개의 프로세스 노드를 포괄합니다. 이 연구는 기존의 Large Language Models (LLMs)가 프로세스 제약을 유지하기에 어려움을 겪고 있음을 보여주고, 새로운 데이터셋을 통해 이를 해결하고자 합니다.

- **Technical Details**: PFDial 데이터셋의 구성은 PlantUML 명세를 기반으로 하여, 각 UML 플로우차트를 원자적 대화 단위로 전환하고 구조화된 5-튜플 (flowchart description, current state, user input, next state, robot output) 형식으로 제공합니다. 이를 통해 모델은 자연어 대화 능력을 유지하면서 정확한 상태 전이를 학습할 수 있습니다. 또한, PFDial-Hard 데이터셋을 추가하여 복잡한 역방향 전이를 처리하는 데 중점을 두고 분류된 90개의 특정 비즈니스 시나리오를 다루었습니다.

- **Performance Highlights**: 실험 결과, 0.5B 모델이 98.99%의 인-도메인 정확도와 92.79%의 아웃-오브-도메인 정확도를 달성하였으며, 8B 모델은 97.02%의 정확도로 GPT-4o를 11.00% 초과하며 결정 브랜치에서 43.88%의 향상을 보여주었습니다. 특히, 7B 모델은 단지 800개의 훈련 샘플로도 90% 이상의 정확도를 달성할 수 있음을 시사합니다. 이러한 성능은 PFDial 데이터셋의 우수성을 입증하며, 모형의 제어된 추론 능력을 효과적으로 향상시킵니다.



### InftyThink: Breaking the Length Limits of Long-Context Reasoning in Large Language Models (https://arxiv.org/abs/2503.06692)
- **What's New**: 본 논문에서는 InftyThink라는 새로운 패러다임을 도입하여 대규모 언어 모델의 장기 추론( long-context reasoning)에서 발생하는 컴퓨팅 비효율성을 극복합니다. 기존의 연속된 논리적 사고를 반복적인 과정으로 전환하여 중간 요약을 포함함으로써, 무한한 깊이의 추론을 가능하게 하고 단위 컴퓨팅 비용을 유지합니다. InftyThink는 전통적인 접근 방식에 비해 계산 복잡성을 크게 감소시키는 톱니형 메모리 패턴을 창출합니다.

- **Technical Details**: InftyThink는 복잡한 추론을 여러 개의 짧은 연관된 세그먼트로 나누어 컴퓨팅 효율성을 유지하면서 일관된 사고 흐름을 보장합니다. 각 세그먼트는 효율적인 길이를 유지하며, 이전 추론에서 요약된 내용을 기반으로 다음 세그먼트를 구축합니다. 이는 인간의 인지 과정에서 영감을 받아 설계되었으며, 복잡한 문제를 관리 가능한 부분으로 나누고 중간 진행 상황을 요약하는 방법과 유사합니다.

- **Performance Highlights**: 실험 결과, Qwen2.5-Math-7B 모델이 MATH500에서 3%, AIME24에서 8%, GPQA_diamond에서 6% 성능 향상을 보였습니다. InftyThink는 장기 추론 관련 데이터를 재구성하여 훈련 세트를 강화하고, 기존의 복잡한 문제 해결 능력을 보다 효율적으로 개선합니다. 이러한 결과는 깊이 있는 추론과 계산 효율성 간의 상상을 뛰어넘는 한계를 해결하였음을 보여줍니다.



### Censoring-Aware Tree-Based Reinforcement Learning for Estimating Dynamic Treatment Regimes with Censored Outcomes (https://arxiv.org/abs/2503.06690)
- **What's New**: 이 논문에서는 Dynamic Treatment Regimes (DTRs)이라는 새로운 접근 방식을 제공하여 개별 환자의 특성에 적응하는 치료 결정을 구조적으로 내리는 방법을 제시합니다. Censoring-Aware Tree-Based Reinforcement Learning (CA-TRL)이라는 혁신적인 프레임워크를 통해, 데이터가 검열될 때 최적의 DTR을 추정하는 복잡성을 해결하려고 합니다. 이 접근은 전통적인 강화 학습 방법에 AIPW (augmented inverse probability weighting) 및 검열 인식 수정 사항을 추가합니다.

- **Technical Details**: CA-TRL은 관찰된 데이터에서 효과적인 DTR을 학습하는 방법을 탐구합니다. 구체적으로, CA-TRL은 검열 데이터에 대한 대응 방법을 도입하여 변동성 및 해석 가능성을 보장합니다. 이 프레임워크는 시뮬레이션과 실제 사례에서 평가되었으며, SANAD 간질 데이터 세트를 사용하여 다양한 임상 환경에서 개인화된 치료 전략을 발전시키고자 합니다.

- **Performance Highlights**: 논문에서는 CA-TRL이 최근 제안된 ASCL 방법보다 제한 평균 생존 시간(RMST) 및 의사 결정 정확도와 같은 주요 지표에서 우수한 성과를 보임을 입증합니다. 이 연구는 다양한 의료 환경에서 데이터 기반의 개인화된 치료 전략을 발전시키는 데 기여하는 중요한 진전을 나타냅니다.



### UniGenX: Unified Generation of Sequence and Structure with Autoregressive Diffusion (https://arxiv.org/abs/2503.06687)
- **What's New**: 이 논문에서 발표된 UniGenX는 과학 데이터를 위한 통합 생성 프레임워크로, 오토회귀(autoregressive) 모델과 조건부 확산(diffusion) 모델을 결합하여 그래프 및 구조 정보를 동시에 생성할 수 있도록 설계되었습니다. 이 접근법은 고차원의 과학 데이터 모델링에서 발생하는 정밀도 문제를 해결하면서도 다양한 과학적 작업에 대한 유연한 생성 능력을 보장합니다.

- **Technical Details**: UniGenX는 오토회귀(next-token prediction) 모델과 조건부 확산 모델을 통합하여 과학 데이터의 시퀀스와 구조를 동시에 모델링하는 데 집중하고 있습니다. 이 모델은 물질 및 소분자의 생성 작업에서 뛰어난 성능을 보이며, 크리스탈 구조 예측 및 소분자 구조 생성에서 최첨단 결과를 달성했습니다.

- **Performance Highlights**: UniGenX는 MP-20, Carbon-24 및 MPTS-52 기준에서 FlowMM을 크게 초월하며, 소분자 생성 작업에서도 최첨단 성능을 보여주었습니다. 또한 통합 데이터셋에서 학습하여 다양한 도메인 간 일반화 능력을 발휘하며, 자연어 처리에까지 확장할 수 있는 가능성을 보여줍니다.



### Exploring LLM Agents for Cleaning Tabular Machine Learning Datasets (https://arxiv.org/abs/2503.06664)
Comments:
          14 pages, 1 main figure, 3 plots, Published at ICLR 2025 Workshop on Foundation Models in the Wild

- **What's New**: 이 연구에서는 고품질 및 오류 없는 데이터셋이 신뢰할 수 있고 정확한 머신러닝 모델을 구축하는 데 중요하다는 점을 강조합니다. 실세계 데이터셋은 센서 고장이나 데이터 입력 오류 등으로 인해 오류가 발생할 수 있으며, 이는 모델 성능에 큰 영향을 미칩니다. 연구팀은 훈련 데이터셋을 정리할 때, Large Language Models(LLMs)를 활용할 수 있는 가능성을 탐구했습니다.

- **Technical Details**: LLM은 Python과 함께 작동하여 훈련 데이터셋의 오류를 정정하는 임무를 수행합니다. 실험에서 여러 가지 Kaggle 데이터셋을 사용하여 의도적으로 오류를 삽입한 후, LLM이 이러한 오류를 수정할 수 있는지 평가했습니다. 결과적으로, LLM은 같은 행 내의 다른 특성에서의 맥락적인 정보를 활용하여 오류 값을 찾고 수정할 수 있지만, 여러 행에 걸친 데이터 분포를 이해해야 하는 복잡한 오류를 탐지하는 데는 어려움을 겪었습니다.

- **Performance Highlights**: LLMs는 비합리적인 값이나 아웃라이어와 같은 오류 항목을 식별하고 수정하는 데 성공적이었지만, 추세나 편향 등 더 복잡한 오류를 감지하는 데 한계가 있었습니다. LLM을 활용한 데이터 정리는 수작업의 부담을 줄일 수 있는 혁신적인 접근법으로 나타났습니다. 향후 연구에서는 보다 복잡한 오류를 효과적으로 처리할 수 있는 방법론에 대한 논의가 이어질 것으로 보입니다.



### AA-CLIP: Enhancing Zero-shot Anomaly Detection via Anomaly-Aware CLIP (https://arxiv.org/abs/2503.06661)
Comments:
          8 pages, 7 figures

- **What's New**: 본 연구에서는 Anomaly-Aware CLIP (AA-CLIP)이라는 새로운 접근 방식을 제안합니다. AA-CLIP은 CLIP의 이상 탐지(discrimination) 능력을 개선하며, 텍스트와 시각적 공간 모두에서 정상 및 비정상 특징을 보다 명확하게 구분합니다. 이 두 단계를 통해 클립 모델을 능동적으로 변형하여 이상 탐지 성능을 저하시키지 않으면서도 향상된 효과를 이끌어냅니다.

- **Technical Details**: AA-CLIP는 단순하지만 효과적인 두 단계 접근 방식을 활용합니다. 첫 번째 단계에서는 'anchors'를 생성하여 각 클래스에 대해 비정상적인 의미를 텍스트 공간에서 명확히 구분하고, 두 번째 단계에서는 이러한 텍스트 앵커와 패치 수준의 시각적 특징을 정렬하여 이상을 정밀하게 탐지합니다. Residual Adapters 구조를 사용하여 클립의 미리 훈련된 지식을 보존합니다.

- **Performance Highlights**: AA-CLIP은 공업과 의료 분야에서의 실험을 통해 자원 효율적인 제로샷 이상 탐지 성능을 달성하였습니다. 제한된 데이터 환경, 즉, 각 클래스당 하나의 정상 샘플과 하나의 비정상 샘플(2-shot)로 훈련하여도 다른 기존 CLIP 기반 기법들과 유사한 성능을 나타냅니다. 또한, 64샷 훈련으로 다양한 데이터셋에서 SOTA 결과를 달성하였습니다.



### Enhancing NLP Robustness and Generalization through LLM-Generated Contrast Sets: A Scalable Framework for Systematic Evaluation and Adversarial Training (https://arxiv.org/abs/2503.06648)
- **What's New**: 이번 연구는 표준 NLP 벤치마크의 한계를 보완하기 위해, 데이터셋 아티팩트와 무의미한 상관관계로 인한 취약점을 포착하는 데 집중했습니다. 대조 세트(contrast sets)를 자동 생성하여 모델의 결정을 방어하는 경계 근처에서 도전하는 것입니다. 이 방법은 연구 분야에서의 접근성을 높이고, 다양한 대조 세트를 제공할 수 있게 합니다.

- **Technical Details**: 이 연구에서는 SNLI 데이터셋을 사용하여 3,000개의 예제로 구성된 대조 세트를 생성했습니다. 이 과정은 대형 언어 모델을 활용하여 자동화되었으며, 이는 인공지능 모델의 강인성을 평가하고 향상하는 데 중요한 역할을 합니다. 모델을 이러한 대조 세트로 미세 조정(fine-tuning)함으로써, 데이터에 대한 시스템적 변화를 처리하는 능력을 향상시켰습니다.

- **Performance Highlights**: 미세 조정을 통해, 모델은 시스템적으로 변화된 예제에 대한 성능을 향상시켰으며, 기존 테스트 정확도는 유지되었습니다. 또한, 새로운 변화에 대한 일반화(generalization) 능력도 소폭 향상되었습니다. 이 자동화된 접근 방식은 NLP 모델의 평가 및 개선을 위한 확장 가능한 솔루션(scalable solution)을 제공합니다.



### Deep Cut-informed Graph Embedding and Clustering (https://arxiv.org/abs/2503.06635)
- **What's New**: 이 논문은 그래프 클러스터링 문제를 해결하기 위해 새로운 비-GNN 기반 프레임워크인 DCGC(Deep Cut-informed Graph embedding and Clustering)를 제안합니다. 기존의 GNN 기반 알고리즘에서 발생하는 표현 붕괴(representation collapse) 문제를 해결하기 위해 그래프 컷 관점에서 접근합니다. 두 가지 주요 모듈인 컷 지식 기반 그래프 인코딩과 최적 수송(optimal transport)을 통한 자기 감독 그래프 클러스터링을 통해 더욱 효과적인 클러스터링을 달성합니다.

- **Technical Details**: DCGC는 컷 지식 기반 그래프 인코딩 모듈을 포함하여 노드의 속성과 그래프 구조 정보를 결합합니다. 이 모듈은 노멀라이즈드 컷(normalized cut)을 최소화하여 그래프 파티션을 찾는 데 도움을 줍니다. 또한 자기 감독 클러스터링은 최적 수송 이론을 활용하여 클러스터 할당을 더욱 안정적이고 균형 있게 만듭니다. 이는 데이터를 단일 레이블로 잘못 할당하는 문제를 방지하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, DCGC는 6개의 실세계 그래프 데이터셋에서 최첨단 그래프 클러스터링 모델들과 비교할 때 경쟁력 있는 성능이나 심지어 우수한 성능을 보여주었습니다. 각 구성 요소의 기여도를 평가하기 위한 포괄적인 Ablation 실험도 진행되어 모든 구성 요소가 필수적임을 보여주었습니다.



### BTFL: A Bayesian-based Test-Time Generalization Method for Internal and External Data Distributions in Federated learning (https://arxiv.org/abs/2503.06633)
Comments:
          accepted as KDD 2025 research track paper

- **What's New**: 이 논문에서는 Test-time Generalization for Internal and External Distributions in Federated Learning (TGFL)라는 새로운 시나리오를 도입하여, 데이터 분포의 변화에 대한 적응력을 평가하는 방법을 제안합니다. 여기에는 Internal Distribution (IND)와 External Distribution (EXD) 두 가지 상황이 포함됩니다. 새로운 Bayesian 기반의 테스트 시간 일반화 방법인 BTFL을 통해 테스트 동안 샘플 수준에서 개인화와 일반화의 균형을 맞출 수 있습니다. BTFL은 두 헤드 아키텍처를 사용하여 지역적 및 전역적 지식을 저장하며, 두 가지 데이터를 통한 인터폴레이션(predictions interpolation) 기술로 성능을 개선합니다.

- **Technical Details**: BTFL은 Bayesian 통계 이론에 기반하여 세 가지 주요 특징을 지니고 있습니다. 첫째, BTFL은 IND/EXD 전환 상황에서의 적응 작업을 후행 추정 문제로 모델링하여 결정 과정을 설명 가능한 방식으로 처리합니다. 둘째, BTFL은 기존의 테스트-시간 적응(TTA) 방법과는 달리 추가적인 최적화 없이 분석적 솔루션을 통해 지식 추출을 수행합니다. 셋째, BTFL은 Beta-Bernoulli 프로세스를 활용하여 역사적 정보 분석을 수행하며, 샘플 수준의 정보 처리를 통해 신뢰성 있는 파라미터 추정을 제공합니다.

- **Performance Highlights**: BTFL은 CIFAR10, OfficeHome, ImageNet 등 다양한 데이터셋에서 기존 방법들보다 우수한 성능을 나타내며, 시간 비용도 줄일 수 있습니다. 평균적으로 BTFL은 정확성에서 1.88%-3.15% 향상되고 약 4배에서 12배의 시간 절약을 기록했습니다. 또한 BTFL은 자원 제약이 있는 클라이언트 장치를 위한 테스트-시간 방법으로서, 훈련 과정의 높은 비용을 줄이고 높은 효율성을 제공합니다.



### Hardware-Accelerated Event-Graph Neural Networks for Low-Latency Time-Series Classification on SoC FPGA (https://arxiv.org/abs/2503.06629)
Comments:
          Paper accepted for the 21st International Symposium on Applied Reconfigurable Computing ARC 2025, Sevilla, Spain, April 9-11, 2025

- **What's New**: 이 연구에서는 시간 시퀀스 분류를 위한 이벤트 그래프 신경망(Event-Graph Neural Network)의 하드웨어 구현을 제안합니다. 특히, AI 모델의 실시간 예측을 위한 저전력, 저지연 하드웨어-소프트웨어 접근법을 요구하는 환경에서 발생하는 데이터의 양이 증가함에 따라 이러한 필요성이 강조됩니다. 기존의 CNN 또는 SNN이 아닌, 새로운 이벤트 그래프 접근법을 사용하여 전통적인 하드웨어보다 훨씬 낮은 전력 소모로 고속 처리할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 연구진은 인공지능을 이용하여 데이터의 실시간 예측을 수행하는 기능을 갖춘 SoC FPGA에 이벤트 그래프 기반 오디오 처리 시스템을 구현했습니다. 특히, 인공 와우(artificial cochlea) 모델을 사용하여 신호를 희소 이벤트 데이터 형식으로 변환하고, 이를 통해 계산 횟수를 대폭 줄일 수 있음을 입증했습니다. 이 시스템은 스파이킹 하이델베르크 숫자(Spiking Heidelberg Digits, SHD) 데이터셋에서 성능을 검증하며, 기존의 방법들에 비해 저전력 및 저지연을 달성할 수 있는 가능성을 제시합니다.

- **Performance Highlights**: 제안된 방법은 SHD 데이터셋에서 92.7%의 부동소수점 정확도를 달성하며, 이는 최첨단 모델에 비해 2.4% 및 2% 낮지만, 67% 적은 모델 파라미터로 달성된 결과입니다. 또한 FPGA 기반 스파이킹 신경망 구현보다 각각 19.3% 및 4.5% 더 뛰어난 성과를 보이며, 양자화된 모델에서는 92.3%의 정확도를 기록했습니다. 이러한 결과는 리소스 사용을 줄이면서도 신뢰할 수 있는 성능을 제공함을 보여줍니다.



### Revisiting Early Detection of Sexual Predators via Turn-level Optimization (https://arxiv.org/abs/2503.06627)
Comments:
          Accepted as a main conference paper at NAACL 2025

- **What's New**: 이 연구에서는 온라인 grooming(그루밍)을 탐지하기 위해 speed control reinforcement learning (SCoRL) 방법을 제안합니다. SCoRL는 Luring Communication Theory (LCT)를 기반으로 하여, 대화의 각 턴에서 위험도를 평가합니다. 이 방법은 보다 세밀한 턴 수준의 위험 레이블을 활용하여 임박한 위험을 미리 탐지하고 최적의 개입 시점을 파악하는 데 초점을 맞추고 있습니다.

- **Technical Details**: SCoRL에서는 턴 수준의 위험 레이블을 기반으로 하고, 속도 제어 보상 함수(speed control reward function)를 설계하여 속도와 정확성 사이의 균형을 맞추고 있습니다. 새로운 벤치마크인 Turn-Level eSPD를 도입하여, 기존의 채팅 수준 메트릭스의 한계를 극복하고 턴 수준의 위험 요소를 평가할 수 있도록 했습니다. 이를 통해 SCoRL은 더 효과적이고 실시간으로 탐지할 수 있는 성능을 보입니다.

- **Performance Highlights**: 실험 결과, SCoRL이 기존의 방법들보다 eSPD 성능에서 유의미한 향상을 보여주었습니다. 실제 온라인 grooming 대화에 대한 분석을 통해, SCoRL 모델이 정량적 성과뿐만 아니라 직관적으로 조기 개입의 optimal points를 식별하는 데에도 효과적임을 확인했습니다. 또한, 이전 메트릭의 허점에도 불구하고 보다 정확한 평가가 가능해졌습니다.



### DiffCLIP: Differential Attention Meets CLIP (https://arxiv.org/abs/2503.06626)
Comments:
          Under review

- **What's New**: 본 연구에서는 DiffCLIP을 제안합니다. 이는 CLIP 아키텍처에 차별적(attention) 주의 메커니즘을 통합한 새로운 비전-언어 모델입니다. 차별적 주의 메커니즘은 관련된 상황을 강조하고 잡음(noise)을 제거하기 위해 개발되었으며, 본 논문에서는 이를 CLIP의 이중 인코더(dual encoder) 프레임워크에 통합했습니다. 결과적으로 DiffCLIP은 이미지-텍스트 이해 작업에서 탁월한 성능을 거두었음을 보여줍니다.

- **Technical Details**: DiffCLIP은 기본 CLIP 모델에 비해 불필요한 잡음을 제거하여 이미지와 텍스트의 정렬을 더욱 정교하게 만듭니다. 두 개의 주의 맵을 학습하고 하나를 다른 것에서 빼내는 방식으로 작동하며, 이는 모델 파라미터와 연산 비용에 미미한 부하만 추가합니다. 차별적 주의 메커니즘을 활용하여 여러 벤치마크에서 CLIP보다 일관된 성능 향상을 보여주고 있습니다. Experiments on various datasets을 통해 DiffCLIP의 효과를 입증하였습니다.

- **Performance Highlights**: DiffCLIP은 zero-shot 분류(zero-shot classification), 검색(retrieval), 강건성(robustness) 벤치마크에서 기본 CLIP 모델을 꾸준히 능가했습니다. 특히, 성능 향상의 경우 0.003%의 추가 파라미터로 가능하다는 점에서 효율성을 강조합니다. 이 연구는 차별적 주의가 다중 모드(multimodal) 표현을 크게 향상시킬 수 있다는 점을 입증하고 있으며, 이는 향후 연구에 중요한 기초를 제공합니다.



### Using Subgraph GNNs for Node Classification:an Overlooked Potential Approach (https://arxiv.org/abs/2503.06614)
Comments:
          16 pages

- **What's New**: 이번 연구에서는 SubGND(Subgraph GNN for NoDe)라는 새로운 프레임워크를 제안합니다. 이는 기존의 Graph Neural Networks(GNNs)의 노드 분류 문제를 서브그래프(subgraph) 분류 문제로 재구성하여 글로벌 메시지 전파 방식의 성능을 개선하려는 시도를 포함합니다. 특히, 차별화된 제로 패딩 전략과 에고-얼터(Ego-Alter) 서브그래프 표현 방법을 도입하여 레이블 충돌(label conflict) 문제를 해결하고, 데이터셋 특성에 따른 적응형 특징 스케일링 메커니즘을 통해 모델의 효과성을 높입니다.

- **Technical Details**: 논문은 주어진 그래프 G=(V,E)의 각 노드 v에 대해 이웃 노드의 멀티홉( multi-hop) 정보를 바탕으로 서브그래프를 구성하는 방식으로 진행됩니다. 이를 통해, 노드를 중심으로 한 고유한 구조를 분석하고, 서브그래프의 표현력과 계산 효율성을 동시에 높일 수 있도록 합니다. SubGND는 이 과정을 통해 GNN이 직면한 구조적 및 특징적 차이를 해결하는 데 초점을 맞추며, 이는 이질적(homophilic) 그래프에서도 우수한 일반화 성능을 보여줍니다.

- **Performance Highlights**: 서브그래프 기반의 접근 방식인 SubGND는 6개의 벤치마크 데이터셋에서 실험을 진행하여 나온 결과로, 글로벌 GNN과 비교할 때 동등하거나 더 나은 성능을 발휘했습니다. 특히 이질적(homophilic) 설정에서 두드러지는 성과를 보이며, 노드 분류의 효과성과 확장성을 가진 유망한 솔루션으로 자리잡았습니다. 이러한 결과는 GNN이 다루기 어려운 동적 및 대규모 그래프 환경에서도 적용 가능성을 보여줍니다.



### WildIFEval: Instruction Following in the Wild (https://arxiv.org/abs/2503.06573)
- **What's New**: 최근에 발전한 LLM(대규모 언어 모델)은 사용자 지침을 따르는 데 있어서 놀라운 성공을 거두고 있지만, 여러 가지 제약조건을 가진 지침을 처리하는 것은 여전히 큰 도전 과제입니다. 이에 따라 WildIFEval이라는 12K의 실제 사용자 지침 데이터셋을 소개하며, 이는 다양한 제약조건을 포함하고 있습니다. 이 데이터셋은 기존의 자료와 달리 자연스러운 사용자 프롬프트를 통해 폭넓은 렉시컬(lexical) 및 주제적(topic) 제약을 포괄하고 있습니다.

- **Technical Details**: WildIFEval은 LLM이 복잡한 다중 제약 지침을 따르는 능력을 평가하기 위해 설계된 대규모 벤치마크입니다. 이 데이터셋은 12K의 사용자 생성 지침을 포함하고 있으며, 각 지침은 요구되는 제약 조건의 집합으로 분해되어 있습니다. 이러한 분해 과정은 LLM이 제약을 충족하는 능력을 평가하는 데 필요한 세분화된 평가를 가능하게 합니다.

- **Performance Highlights**: 실험 결과, 평가된 모든 모델은 제약 조건의 수가 증가할수록 성능이 저하되는 현상을 보였습니다. 특히, 길이 관련 제약 조건을 포함한 작업은 모든 모델에 대해 더 어려운 것으로 나타났습니다. 주목할 점은 특정 제약 유형이 모델 성능에 미치는 중요한 역할을 한다는 것입니다.



### SHIP: A Shapelet-based Approach for Interpretable Patient-Ventilator Asynchrony Detection (https://arxiv.org/abs/2503.06571)
Comments:
          Accepted at PAKDD 2025

- **What's New**: 이번 연구에서는 질병 환자와 인공호흡기 간의 비동기 현상(PVA) 탐지를 위해 새로운 모양 기반 접근법 SHIP를 제안합니다. SHIP는 시계열 데이터에서 차별화된 부분인 shapelets를 활용하여 PVA 탐지의 정확도와 해석 가능성을 높입니다. 데이터 불균형 문제를 해결하고 PVA 이벤트를 보다 효과적으로 분류하기 위해 shapelet 기반 데이터 증강과 shapelet 풀을 구성하는 방법을 소개합니다.

- **Technical Details**: 제안된 SHIP 방법론은 환자와 인공호흡기 간의 비동기 현상을 다변량 시계열 분류 문제로 설정합니다. Pmask, Flow, Thor, Abdo와 같은 다양한 채널을 분석하여 Autocycling(AC), Double Triggering(DT), Ineffective Efforts(IE)와 같은 PVA 이벤트를 식별합니다. shapelet 추출, 데이터 증강 및 분류기를 조합하여 PVA 탐지를 수행하며, SIHP는 shapelet 특징과 통계적 특징을 결합하여 PVA 이벤트를 정확하게 식별합니다.

- **Performance Highlights**: 실험 결과 SHIP 방법은 PVA 탐지에서 상당한 성능 향상을 보였습니다. 특히, 단 2개의 채널(Pmask 및 Flow)을 사용하여 4개의 채널을 사용할 때와 유사한 성과를 올렸습니다. 이는 PVA 탐지의 해석 가능성을 높이면서 데이터 불균형 문제를 효과적으로 해결할 수 있는 방법을 제시합니다.



### Conceptrol: Concept Control of Zero-shot Personalized Image Generation (https://arxiv.org/abs/2503.06568)
- **What's New**: 이번 논문에서는 개인화된 이미지 생성을 위한 새로운 프레임워크 Conceptrol을 제안한다. 기존의 zero-shot adapter(어댑터)들이 텍스트 프롬프트와 개인화된 콘텐츠 간의 균형을 잘 잡지 못하는 문제를 식별하였다.

- **Technical Details**: Conceptrol은 텍스트 개념 마스크를 사용하여 시각적 명세의 주의(attention) 점수를 조정하며, 이를 통해 개인화된 이미지 생성의 정확성을 높인다. 이는 기존의 IP-Adapter와 OminiControl에 비해 최대 89%의 성능 향상을 보여준다.

- **Performance Highlights**: 종합적인 실험 결과, Conceptrol은 기존의 fine-tuning(미세 조정) 방법보다 뛰어난 성능을 발휘하였다. 특히, DreamBench++를 통한 평가에서 개인화된 이미지 생성을 위한 새로운 기준을 제시하였다.



### Human Cognition Inspired RAG with Knowledge Graph for Complex Problem Solving (https://arxiv.org/abs/2503.06567)
- **What's New**: 이번 논문에서는 CogGRAG이라는 새로운 RAG(검색 기반 생성) 프레임워크를 제안하여 LLM(대형 언어 모델)의 복잡한 문제 해결 능력을 향상시키고자 합니다. CogGRAG는 인간의 인지 과정을 모델링하여 복잡한 문제를 분해하고 자기 검증(self-verification)을 통해 정확성을 높이는 세 가지 단계(Decomposition, Retrieval, Reasoning with Self-Verification)를 포함하고 있습니다. 이러한 접근법은 기존 RAG 시스템의 한계를 극복하고 KGQA(지식 그래프 질문 응답) 작업에서 LLM의 성능을 크게 향상시킬 수 있는 잠재력을 가지고 있습니다.

- **Technical Details**: CogGRAG는 복잡한 문제 해결에 있어 인간의 사고 방식에 영감을 받아 구성됩니다. 첫 번째 단계인 Decomposition은 문제를 작은 하위 문제로 분해하여 사고 맵을 형성합니다. 두 번째 단계 Retrieval에서는 지식 그래프를 활용하여 외부 지식 소스에서 구조화된 정보를 추출하고, 마지막 Reasoning with Self-Verification 단계에서는 중간 및 최종 결과의 정확성을 검증합니다. 이는 LLM의 복잡한 문제 해결 능력을 극대화하기 위한 구조적이고 체계적인 접근방법입니다.

- **Performance Highlights**: CogGRAG는 세 가지 LLM 백본을 사용하여 네 가지 벤치마크 데이터셋에서 시스템적으로 실험을 수행하였으며, 기존의 기초 모델보다 월등한 성능을 보여주었습니다. 이번 연구 결과는 CogGRAG가 LLM의 신뢰성을 높이고, 흑묘복수(hallucination) 문제를 현저히 줄일 수 있음을 입증하고 있습니다. 이 프레임워크는 KGQA 작업에서 LLM의 정확도를 높이는 중요한 솔루션으로 자리 잡을 것으로 기대됩니다.



### LSA: Latent Style Augmentation Towards Stain-Agnostic Cervical Cancer Screening (https://arxiv.org/abs/2503.06563)
- **What's New**: 이번 연구에서는 Latent Style Augmentation (LSA)이라는 새로운 프레임워크를 제안하여 자궁경부암 진단을 위한 Whole Slide Images (WSIs)에서 staining 변이를 효과적으로 보완하는 방법을 소개합니다. 기존의 patch-level stain augmentation 방법들은 gigapixel 해상도의 WSIs에 확장할 때 두 가지 주요 제한점을 보였습니다. LSA는 WSI 수준에서 발생하는 latent features에 직접적으로 온라인으로 stain augmentation을 수행함으로써 이 문제를 해결합니다.

- **Technical Details**: LSA는 WSI의 latent feature에서 stain augmentation을 수행하는 새로운 접근법을 제공합니다. WSAug라는 방법을 통해 전체 WSI의 패치들 간에 stain을 일관되게 유지하면서 데이터 효율성을 개선합니다. Stain Transformer를 설계하여 latent space 내에서 특정 스타일을 시뮬레이션하며, 이를 통해 WSI 수준의 분류기의 강건함을 높입니다.

- **Performance Highlights**: multi-scanner WSI 데이터셋을 통해 검증한 결과, 단일 스캐너 데이터로 훈련된 LSAs이 다른 스캐너의 데이터에서 우수한 성능 향상을 보였습니다. 이러한 결과는 LSA의 접근 방식이 기존의 도메인 변화 문제를 해결하며, 다양한 스캐너에서 안정적인 진단 성능을 유지할 수 있음을 시사합니다.



### ARMOR v0.1: Empowering Autoregressive Multimodal Understanding Model with Interleaved Multimodal Generation via Asymmetric Synergy (https://arxiv.org/abs/2503.06542)
- **What's New**: ARMOR은 문자열 및 이미지를 동시에 이해하고 생성할 수 있는 자원 효율적인 자가 회귀(framework) 프레임워크로, 기존의 다중 모달 대형 언어 모델(MLLMs)을 개선합니다. 기존 UniMs 모델은 상당한 컴퓨팅 자원을 요구하며, 복잡한 텍스트-이미지 생성에 어려움을 겪습니다. ARMOR는 비대칭 인코더-디코더 아키텍처를 도입하여 저렴한 비용으로 자연스러운 텍스트-이미지 혼합 생성을 가능하게 합니다.

- **Technical Details**: ARMOR는 모델 아키텍처, 훈련 데이터, 훈련 알고리즘의 세 가지 측면에서 기존 MLLMs를 확장합니다. 특히, 비대칭 이미지 디코더를 도입하여 모델의 이해 능력을 거의 손상시키지 않으면서도 텍스트-이미지 혼합 생성을 가능하게 하며, 데이터셋을 기초로 한 ‘생성 방법’ 알고리즘을 통하여 모델을 점진적으로 훈련합니다.

- **Performance Highlights**: 실험 결과는 ARMOR가 기존 MLLMs를 Superior UniMs로 업그레이드하며, 제한된 훈련 자원으로 높은 이미지 생성 능력을 입증합니다. ARMOR은 9개의 기준에서 기존 모델들보다 뛰어난 성능을 보여주며, 특히 multimodal understanding에서 큰 차이를 만들어냅니다. 최종적으로 ARMOR는 UNI 모델 구축을 위한 자가 회귀 아키텍처의 가능성을 확고히 합니다.



### AnywhereDoor: Multi-Target Backdoor Attacks on Object Detection (https://arxiv.org/abs/2503.06529)
- **What's New**: 이 논문은 객체 탐지에 대한 다중 대상 백도어 공격 방식인 AnywhereDoor를 제안합니다. 기존 연구들은 단일 목표 설정에 기반했으나, AnywhereDoor는 공격자가 추론 시 컨텍스트에 따라 다양한 목표를 선택할 수 있도록 합니다. 이 방법은 물체가 사라지거나, 새로운 물체를 생성하거나, 잘못된 레이블을 부여하는 등의 유연성을 제공합니다.

- **Technical Details**: AnywhereDoor는 세 가지 주요 혁신을 통해 구현됩니다. 첫째, 목표를 분리하여 지원하는 목표 수를 확장하는 objective disentanglement; 둘째, region 기반 탐지기를 상대로 강력한 영향을 보장하는 trigger mosaicking; 셋째, 객체 수준 데이터 불균형 문제를 해결하기 위한 전략적 배치 전략이 포함되어 있습니다. 이러한 접근은 공격 성공률을 기존 방법에 비해 26% 향상시킵니다.

- **Performance Highlights**: 다양한 객체 탐지 알고리즘과 데이터셋에 대한 광범위한 실험 결과, AnywhereDoor는 기존 방법보다 공격 성공률이 26% 향상됨을 입증했습니다. 또한, 공격이 이뤄질 때 깨끗한 샘플에 대한 성능이 유지되는 것도 확인되었습니다. 이 연구는 객체 탐지 보안 연구의 새로운 방향을 제시합니다.



### From Motion Signals to Insights: A Unified Framework for Student Behavior Analysis and Feedback in Physical Education Classes (https://arxiv.org/abs/2503.06525)
Comments:
          Work in progress

- **What's New**: 본 연구에서는 교육적 환경에서 학생의 행동을 분석하기 위해 통합된 end-to-end 프레임워크를 제안합니다. 이 프레임워크는 비디오 데이터 대신 학생의 동작 신호를 기반으로 하여, 체육 수업에서의 학생 행동을 보다 정확하게 포착하고 분석할 수 있습니다. 특히, 대형 언어 모델을 활용해 학생 행동에 대한 심층 분석과 피드백을 제공합니다.

- **Technical Details**: 제안하는 프레임워크는 학생의 동작 신호를 분석하기 위해 Inertial Measurement Units (IMUs)를 사용합니다. 이 과정에서 모션 감지, 행동 인식, 행동 품질 평가의 세 가지 단계가 계획되어 있으며, 각 단계에서 신호를 처리하여 결과적으로 텍스트 베이스의 통계 요약을 생성합니다. 최종적으로 이 데이터는 대형 언어 모델에 입력되어 교육적 통찰과 피드백을 제공하는 보고서를 생성합니다.

- **Performance Highlights**: 실험 결과, 본 프레임워크는 체육 수업에서 학생 행동을 정확하게 식별하고, 의미있는 교육적 인사이트를 산출하는 데 성공했습니다. 선정 연구에서는 농구 수업을 사례로 사용하였으며, 정량적 및 정성적 실험 결과가 프레임워크의 강점과 효과를 입증하고 있습니다. 이러한 접근법은 체육 수업의 교육 디자인을 최적화하는 데 기여할 것으로 기대됩니다.



### Generative AI as Digital Media (https://arxiv.org/abs/2503.06523)
- **What's New**: 이 논문에서는 Generative AI에 대한 기존의 혁신적 또는 종말론적 시각이 오해라는 주장을 하고 있습니다. 대신, Generative AI는 검색 엔진 및 소셜 미디어와 함께 알고리즘 미디어의 진화적 단계로 이해해야 한다고 강조합니다. 이러한 플랫폼에서는 정보 통제를 중앙집중화하고, 복잡한 알고리즘을 통해 콘텐츠를 형성하며, 사용자 데이터를 광범위하게 활용합니다.

- **Technical Details**: 논문은 현재의 규제 프레임워크인 EU의 AI 법과 미국의 Executive Order 14110이 주로 반응적 위험 완화에 중점을 두고 있음을 지적합니다. 이는 국가 안전, 공공 건강, 알고리즘 편향과 같은 측정 가능한 위협을 다루지만, 특히 디지털 미디어에서 신뢰와 정당성 문제가 더욱 중대하다고 설명합니다. 따라서 투명성, 책임, 공적 신뢰를 촉진하는 능동적 규제가 필수적입니다.

- **Performance Highlights**: Generative AI를 혁신으로만 보는 관점은 소셜 미디어와 검색 엔진의 규제가 불충분했던 과거의 실패를 반복할 위험이 있습니다. 따라서 규제는 공적 선을 위한 알고리즘 미디어 환경을 적극적으로 조정하고, 양질의 정보와 강력한 시민 담론을 지원해야 한다고 강조합니다. 즉, 이러한 접근 방식이 없으면 여론이 편향된 공동체로 분열될 위험이 커지며, 이는 Generative AI의 맞춤형 진실 전달 능력에 의해 더욱 증대될 수 있습니다.



### Can Small Language Models Reliably Resist Jailbreak Attacks? A Comprehensive Evaluation (https://arxiv.org/abs/2503.06519)
Comments:
          19 pages, 12 figures

- **What's New**: 본 논문은 소형 언어 모델(SLMs)의 탈옥 공격(jailbreak attacks)에 대한 첫 대규모 실증 연구를 수행하였습니다. SLMs는 대형 언어 모델(LLMs)보다 낮은 컴퓨팅 요구사항과 향상된 개인 정보 보호를 제공하며, 특정 도메인에서 비교 가능한 성능을 보여주고 있습니다. 그러나 SLMs에 대한 보안 문제, 특히 탈옥 공격의 취약성에 대한 연구는 부족한 실정입니다.

- **Technical Details**: SLMs는 컴팩트한 아키텍처와 적은 매개변수(주로 100MB에서 5B 사이)로 쉽게 배치할 수 있습니다. 이 연구에서는 15개의 주류 SLM 패밀리에서 63개의 SLM 모델을 평가했고, 8개의 최첨단 탈옥 방법에 대한 시스템적 평가를 통해 47.6%의 SLM들이 높은 탈옥 공격에 대한 취약성을 보였습니다. 연구에서는 모델 크기, 아키텍처, 훈련 데이터셋 및 훈련 기술이 취약성에 미치는 영향을 분석하였습니다.

- **Performance Highlights**: 평가된 SLM의 약 50%가 탈옥 공격에 대해 높은 취약성을 나타내며, 38.1%는 직접적인 해로운 요청에 대해서도 저항하지 못했습니다. Alibaba의 Qwen 시리즈와 Google의 Gemma 시리즈와 같은 일부 SLM들은 비교적 좋은 저항력을 보여주었고, 다양한 공격 방법과 위험 카테고리에 따라 각기 다른 취약성을 보였습니다. 또한, 수퍼바이즈드 파인 튜닝(Supervised Fine Tuning) 모델이 DPO 최적화 모델에 비해 10-40% 더 우수한 견고성을 나타내는 현상도 관찰되었습니다.



### Towards Superior Quantization Accuracy: A Layer-sensitive Approach (https://arxiv.org/abs/2503.06518)
- **What's New**: 본 논문에서는 대형 비전 및 언어 모델에서 양자화(Quantization) 정확도를 개선하기 위해 레이어 민감도 분석 기반의 새로운 방법론을 제안합니다. 기존 방법들이 균일한 양자화 설정을 사용하여 다양한 레이어의 양자화 난이도를 고려하지 못한 점을 극복하고자 합니다. 새로운 기술인 SensiBoost 및 KurtBoost는 레이어가 양자화하기 어려운 정도를 기준으로 추가 메모리 예산을 할당하여 양자화 정확도를 향상시킵니다.

- **Technical Details**: 제안된 방법은 각 레이어의 민감도 지표인 활성화 감도(Activation Sensitivity)와 가중치 분포의 커토시스(Kurtosis)를 활용하여 양자화에 어려운 레이어를 식별하고 있습니다. 이 기술은 기존의 균일한 양자화 방식을 보완하여 메모리 예산을 최적화하면서 더 높은 양자화 정확도를 달성합니다. SensiBoost와 KurtBoost 방법은 각각 활성화 감도 점수와 커토시스 지표를 활용하여 민감한 레이어를 효과적으로 탐지합니다.

- **Performance Highlights**: SensiBoost와 KurtBoost 방법은 LLama 모델에서 2%의 메모리 예산 증가만으로도 기존 방법보다 존출(Perplexity)을 최대 9% 감소시키는 성과를 보였습니다. 이로 인해, 양자화 정확도가 크게 개선되었으며, 기존의 양자화 기법들보다 더 효율적인 자원 사용이 가능합니다. 이 연구는 대형 언어 모델의 양자화 기술에 대한 새로운 인사이트를 제공하며, 다양한 애플리케이션에 적용될 가능성을 보여줍니다.



### GFlowVLM: Enhancing Multi-step Reasoning in Vision-Language Models with Generative Flow Networks (https://arxiv.org/abs/2503.06514)
- **What's New**: 이 논문에서는 VLM(비전-언어 모델)의 한계 극복을 위한 새로운 프레임워크인 GFlowVLM을 제안합니다. 기존의 Supervised Fine-Tuning(SFT) 및 Reinforcement Learning(RL) 방법들이 가지는 한계를 인식하고, Generative Flow Networks(GFlowNets)를 활용하여 다채로운 해결책을 생성하는 구조적 사고를 지원하는 방식으로 발전했습니다. GFlowVLM은 비선형 결정을 모델링하여 복잡한 추론 작업에서 장기적 의존성을 캡쳐할 수 있습니다.

- **Technical Details**: GFlowVLM은 GFlowNets를 사용하여 비선형 차원에서 VLM을 파인 튜닝하는 방식입니다. 이는 상태 간의 논리적 의존성을 고려하여 연속적인 상태에서 구조적 추론 프로세스를 강화할 수 있도록 도와줍니다. 저자들은 GFlowNets가 제시하는 보상 함수에 기반하여 다양한 샘플링 전략을 이용해 높은 보상을 받을 수 있는 궤적을 통해 방대한 문제 공간에서의 제한적인 솔루션을 극복하고자 했습니다.

- **Performance Highlights**: GFlowVLM의 성능은 카드 게임(NumberLine, BlackJack) 및 구현 계획(ALFWorld)과 같은 복잡한 작업에서 검증되었습니다. 이 프레임워크는 기존의 SFT 및 RL 기반 방법들과 비교하여 높은 교육 효율성과 다양한 솔루션 생성 능력을 보여주며, 특히 일반화 성능이 향상되었음을 입증합니다. 세부 실험 결과는 GFlowVLM이 성공률 및 문제 해결의 다양성을 높이는 데 기여함을 분명하게 나타냅니다.



### HFedCKD: Toward Robust Heterogeneous Federated Learning via Data-free Knowledge Distillation and Two-way Contras (https://arxiv.org/abs/2503.06511)
- **What's New**: 본 논문에서는 전통적인 연합 학습(federated learning) 프레임워크의 정적 특징을 넘어서, 동적인 특성을 가진 시스템에 대한 유연한 모델 아키텍처를 제안합니다. 제안된 HFedCKD 시스템은 데이터가 없는 지식 증류(data-free knowledge distillation)와 쌍방향 대비(two-way contrast)를 기반으로 하여 클라이언트들이 지식 전송에 효율적으로 참여할 수 있도록 돕습니다. 또한, 참여율이 낮은 클라이언트의 지식 편차를 완화하고 모델의 성능과 안정성을 향상시키는 효과를 보여줍니다.

- **Technical Details**: HFedCKD는 Inverse Probability Weighted Distillation (IPWD) 전략을 사용하여 데이터가 없는 지식 전송 프레임워크를 최적화합니다. 클라이언트의 데이터 특징을 보완하는 생성기(generator)를 활용하고, 각 클라이언트의 예측 기여도를 동적으로 평가하여 클라이언트의 가중치를 조정합니다. 이 가중치 조정을 통해, 참가 클라이언트의 지식을 공정하게 통합하여 모델의 성능을 극대화합니다.

- **Performance Highlights**: 제안된 HFedCKD 프레임워크는 이미지 및 IoT 데이터셋에 대한 광범위한 실험을 통해 일반화 및 견고성을 검증하였습니다. 클라이언트의 참여 빈도와 데이터 분포의 차이를 고려하여 성능을 향상시키며, 지식 축적에 효과적임을 입증하였습니다. 이를 통해 연합 학습 분야에서 더 나은 모델 성능과 안정성을 제공하는 새로운 방향성을 제시합니다.



### A Light and Tuning-free Method for Simulating Camera Motion in Video Generation (https://arxiv.org/abs/2503.06508)
Comments:
          18 pages in total

- **What's New**: LightMotion은 비디오 생성에서 카메라 움직임을 시뮬레이션하기 위한 경량, 조정-free 접근 방식을 제안합니다. 이 방법은 latent space에서 작동하며, 기존 모델들에서 요구되는 추가적인 fine-tuning, inpainting 및 depth estimation을 제거합니다. LighMotion의 주요 혁신점은 permutation operation을 통해 다양한 카메라 움직임을 효과적으로 시뮬레이션할 수 있다는 것입니다.

- **Technical Details**: 이 논문에서는 latent space permutation 작업과 resampling 전략을 통해 새로운 관점을 정확하게 채우면서도 프레임 간 일관성을 유지하는 방법을 설명합니다. 또한, noise를 재도입하여 SNR shift를 완화하고 비디오 생성 품질을 향상시키는 latent space correction 메커니즘을 제안합니다. 이러한 처리 과정은 기존의 복잡한 모델을 사용하지 않고도 효율적인 결과를 제공하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 실험 결과, LightMotion은 기존의 다양한 카메라 움직임을 조정하는 방법들과 비교했을 때 정량적, 정성적으로 더 나은 성능을 보입니다. 또한, 사용자가 정의한 다양한 파라미터 조합을 쉽게 지원하여 비디오 생성 과정의 유연성과 다재다능성을 극대화합니다. 이러한 특성 덕분에 LightMotion은 비디오 생성 분야에서의 널리 보급 가능성을 높이고 있습니다.



### DynamicID: Zero-Shot Multi-ID Image Personalization with Flexible Facial Editability (https://arxiv.org/abs/2503.06505)
Comments:
          17 pages, 16 figures

- **What's New**: 이번 논문에서는 DynamicID라는 새로운 튜닝이 필요 없는 프레임워크를 제안합니다. 이 프레임워크는 단일 ID 및 다중 ID 개인화 생성 모두를 지원하며, 높은 정체성 충실도와 유연한 얼굴 편집성을 보장합니다. 주요 혁신 기술로는 Semantic-Activated Attention (SAA)와 Identity-Motion Reconfigurator (IMR)가 있습니다. SAA는 특정 이미지 Latent 쿼리에 기반하여 ID 특성을 효과적으로 주입하는 메커니즘을 제공하며, IMR은 얼굴 동작과 정체성 특징을 효과적으로 분리하고 재결합할 수 있도록 합니다.

- **Technical Details**: DynamicID는 두 가지 독창적인 디자인으로 구성되어 있으며, 첫 번째는 쿼리 레벨의 활성화 게이팅을 활용하여 ID 특성을 원활하게 주입할 수 있도록 하는 SAA입니다. 두 번째는 IMR로, 이 모델은 대조 학습을 이용하여 얼굴의 동작 및 정체성 특징을 분리한 후 다시 결합할 수 있는 기능을 가지고 있습니다. 이러한 구조는 원래 모델의 행동을 방해하지 않으면서 다중 ID 개인화 생성을 가능하게 합니다. 데이터 세트로는 VariFace-10k를 사용하여 10,000명의 고유한 개인을 대상으로 각각 35개의 개별 얼굴 이미지를 제공합니다.

- **Performance Highlights**: 실험 결과, DynamicID는 정체성과 얼굴 편집 가능성, 다중 ID 개인화 능력에서 기존의 최신 기법들을 초월하는 성능을 보였습니다. 특히, 다중 ID 생성이라는 어려움을 극복하며 높은 충실도를 유지하는 능력이 두드러집니다. 이러한 결과는 DynamicID가 단일 및 다중 ID 시나리오에서 모두 잘 작동한다는 것을 증명합니다.



### ExGes: Expressive Human Motion Retrieval and Modulation for Audio-Driven Gesture Synthesis (https://arxiv.org/abs/2503.06499)
- **What's New**: ExGes는 오디오 기반 인간 제스처 생성을 향상시키기 위해 새로운 검색 강화(diffusion) 프레임워크로, 동작 기반 구성(Motion Base Construction), 동작 검색(Motion Retrieval), 정밀 제어(Precision Control) 모듈로 구성됩니다. 본 연구의 핵심은 외부 보조 가이드를 통합하여 표현력이 풍부하고 의미적으로 정합된 제스처를 생성하는 것입니다. 이는 기존 방법의 한계를 극복하고, 다양한 감정 상태 및 개인화된 스타일을 반영할 수 있도록 합니다.

- **Technical Details**: ExGes는 세 가지 주요 모듈로 구성되어 있으며, 각각의 모듈은 특정 기능을 수행하여 전체 시스템의 성능을 향상시킵니다. 동작 기반 구성 모듈은 훈련 데이터셋을 사용하여 제스처 라이브러리를 구축하고, 동작 검색 모듈은 대조 학습(constrative learning)과 모멘텀 증류(momentum distillation)를 통해 세밀한 참고 포즈를 검색합니다. 마지막으로, 정밀 제어 모듈은 부분 마스킹(partial masking)과 확률적 마스킹(stochastic masking)을 통합하여 유연하고 세밀한 제어를 가능하게 합니다.

- **Performance Highlights**: 실험 결과, ExGes는 EMAGE보다 Fréchet Gesture Distance를 6.2% 낮추고, 모션 다양성을 5.3% 향상시키는 것으로 나타났습니다. 사용자 연구에 따르면, ExGes의 자연스러움과 의미적 관련성에 대해 71.3%의 선호도를 보였습니다. 이러한 결과는 ExGes가 기존 방법들보다 더 강력한 성능을 발휘함을 입증하고 있습니다.



### Evaluation of Safety Cognition Capability in Vision-Language Models for Autonomous Driving (https://arxiv.org/abs/2503.06497)
- **What's New**: 이 논문에서는 자율주행 시스템에서 비전-언어 모델(VLM)의 안전성을 평가하기 위해 안전 인지 드라이빙 벤치마크(SCD-Bench)라는 새로운 평가 방법을 제안합니다. 기존 연구가 주로 전통적인 벤치마크 평가에 중점을 둔 반면, SCD-Bench는 VLM이 인간과 상호작용할 때의 안전 인지 능력을 평가하는 데 초점을 맞추고 있습니다. 또한, 대규모 주석 문제를 해결하기 위해 자율주행 이미지-텍스트 주석 시스템(ADA)을 개발하였습니다.

- **Technical Details**: SCD-Bench는 VLM의 안전 인지 능력을 1) 명령 오해, 2) 악의적 결정, 3) 지각 유도, 4) 윤리적 딜레마의 네 가지 차원에서 평가합니다. 이 평가에서는 VLM이 모호한 명령을 처리하고, 악의적인 의도를 인식하며, 지각을 왜곡하는 정보를 피하는 능력을 중점적으로 살펴봅니다. 논문에서 제안하는 자동화된 평가 방법은 LLM을 기반으로 하여 구현되었으며, 본 연구의 자동 평가와 전문가 평가 간의 일치율이 99.74%에 달한다고 보고합니다.

- **Performance Highlights**: 초기 실험 결과에서 기존 오픈 소스 모델은 안전 인지 능력이 부족함을 보였으며, 특히 경량 모델(1B-4B)는 안전 인지에서 미흡한 성과를 보였습니다. 또한, SCD-Bench의 테스트 케이스는 5,043개로 구성되어 있으며, 안전 인지의 측면에서 VLM의 답변 능력을 정량적으로 평가합니다. VLM이 비행 안전을 충족하는 데 있어 도전 과제가 남아 있으며, 이는 경량 모델과 효율성을 유지하는 것의 중요성을 강조합니다.



### PerturboLLaVA: Reducing Multimodal Hallucinations with Perturbative Visual Training (https://arxiv.org/abs/2503.06486)
- **What's New**: 이번 논문은 밀접한 이미지 캡셔닝(dense image captioning) 작업에서 멀티모달 대형 언어 모델(MLLMs)의 환각(hallucination) 문제를 해결하는 것을 목표로 합니다. 기존의 캡션 품질을 측정하는 지표가 부족한 상황에서, 새로운 지표인 HalFscore를 제안하고 이를 통해 캡션의 정확성과 완전성을 정량적으로 평가합니다. 또한, 모델이 언어 선행 언어(prior)에 과도하게 의존한다는 문제를 발견하고, PerturboLLaVA라는 훈련 전략을 통해 이러한 의존성을 줄여 더 신뢰할 수 있는 결과를 만들어냅니다.

- **Technical Details**: HalFscore는 언어 그래프(language graph)를 기반으로 하여 개념 수준에서의 설명 품질을 측정합니다. 이 지표는 캡션의 정확성과 완전성을 평가하며, 잘못된 요소와 누락된 세부 사항을 식별하여 모델 성능을 정량적으로 분석할 수 있습니다. 논문에서는 PerturboLLaVA 방법을 통해 적대적으로 변형된 텍스트를 훈련에 포함시켜 모델이 시각적 입력에 더욱 집중하도록 유도하는 간단하면서 효과적인 방법을 제시합니다.

- **Performance Highlights**: PerturboLLaVA는 부가적인 계산 비용 없이 MLLMs의 환각 문제를 효과적으로 억제하며, 기존의 선진 방법론에 비해 더 뛰어난 성능을 보여줍니다. 이 방법은 일반적인 다중 모달 벤치마크에서 성능 향상 또한 가져오며, 특히 환각이 발생하는 상황에서도 보다 정확하고 신뢰할 수 있는 이미지 기반 설명을 생성하는 데 성공합니다. 결론적으로, HalFscore와 PerturboLLaVA는 MLLMs의 시각적 이해 능력을 높이고, 훨씬 더 효율적이고 확장 가능한 솔루션을 제공합니다.



### Sign Language Translation using Frame and Event Stream: Benchmark Dataset and Algorithms (https://arxiv.org/abs/2503.06484)
Comments:
          In Peer Review

- **What's New**: 이 논문에서는 장애인을 위한 효과적인 의사소통 수단으로서 정확한 수화 이해의 중요성을 강조하고, 전통적인 RGB 카메라의 한계를 극복하기 위해 이벤트 카메라를 활용한 방식으로 수화 번역 알고리즘을 제안한다. 이 연구는 VECSL이라는 대규모 RGB-Event 수화 번역 데이터셋을 수집하여 개발하였으며, 이는 15,676개의 샘플과 2,568개의 중국 문자를 포함하고 있어 다양한 환경에서의 수화 번역의 도전 과제를 반영한다.

- **Technical Details**: VECSL 데이터셋은 DVS346 카메라를 사용하여 수집되며, 다양한 조명 조건과 카메라 이동을 포함하여 다채로운 실내외 환경에서의 샘플을 제공합니다. 저자들은 기존의 SLT 알고리즘을 재훈련하고 평가하여 새로운 벤치마크를 제공하며, RGB-이벤트 기반 수화 번역을 위한 M2-SLT 프레임워크를 제안한다. 이 프레임워크는 미세한 마이크로-사인 및 거친 매크로-사인 탐색 모듈을 포함하여, 텍스트 디코더 mBART를 활용하여 고정밀, 견고한 수화 번역 결과를 도출한다.

- **Performance Highlights**: VECSL 데이터셋을 기반으로 한 M2-SLT 프레임워크는 기존의 수화 번역 알고리즘에 비해 뛰어난 성능을 보여준다. 이 연구는 새로운 데이터셋의 구축과 함께 수화 번역 연구에서 향후 다양한 연구가 진행될 수 있도록 강력한 기반을 제공하는 것을 목표로 한다. 저자들은 이 연구 결과들이 이후 연구에 영감을 줄 수 있다고 믿으며, 데이터셋과 소스 코드를 공개한다고 언급하고 있다.



### PDB: Not All Drivers Are the Same -- A Personalized Dataset for Understanding Driving Behavior (https://arxiv.org/abs/2503.06477)
- **What's New**: 새로운 Personalized Driving Behavior (PDB) 데이터셋은 운전자의 개인적 행동을 포착하기 위해 설계된 다중 모달 데이터셋이다. 기존 데이터셋들이 모든 운전자를 동질적으로 취급하는 반면, PDB는 자연적인 운전 조건 하에서 운전자의 변화를 캡처한다. 이 데이터셋은 일정한 경로, 차량, 조명 조건을 유지하여 외부 영향을 최소화하고, 128라인 LiDAR, 전방 카메라 비디오, GNSS, 9축 IMU, CAN 버스 데이터와 같은 다양한 센서를 통해 수집된 데이터를 포함한다.

- **Technical Details**: PDB 데이터셋은 12명의 참가자로부터 약 270,000개의 LiDAR 프레임과 1.6백만 개의 이미지, 6.6 TB의 원시 센서 데이터를 포함하고 있다. 또한, 각 10초로 분리된 1,669 개의 진행 경로(segment)가 포함되어 있어, 운전자의 행동을 보다 명확하게 연구할 수 있다. 데이터 수집은 모든 세션에서 동일한 차량 모델을 사용함으로써 운전 스타일의 변화를 분석할 수 있게 한다.

- **Performance Highlights**: PDB는 운전자의 개별성과 행동 특성을 연구할 수 있는 중요한 자원으로, 적응형 운전자 보조 시스템 및 행동 인식 기반 경로 예측과 같은 다양한 응용 프로그램에 기여한다. 이 데이터셋은 기존 데이터셋에 비해 더 풍부한 운전자의 특정 데이터를 제공하며, 운전 스타일의 더 포괄적인 표현을 가능하게 한다. PDB의 출시는 인간 중심의 지능형 교통 시스템 발전에 기여할 것으로 기대된다.



### HuixiangDou2: A Robustly Optimized GraphRAG Approach (https://arxiv.org/abs/2503.06474)
Comments:
          11 pages

- **What's New**: 이 논문에서는 HuixiangDou2라는 새로운 GraphRAG 프레임워크를 소개합니다. 이 프레임워크는 지식 집약적인 도메인에서 LLM(대규모 언어 모델)의 성능을 극대화하기 위해 설계되었습니다. 기존 GraphRAG 방법론을 통합하여 효율성과 정확성을 높이는 데 초점을 맞추고 있습니다. 또한, 새로운 다단계 검증 메커니즘을 도입하여 계산 비용을 증가시키지 않고도 검색의 강인성을 향상시킵니다.

- **Technical Details**: HuixiangDou2는 쿼리 분해 및 두 수준 검색과 같은 다양한 검색 메커니즘을 평가하고 최적화하는 데 중점을 둡니다. 이 프레임워크는 구조화된 지식 표현을 이용하여 동적 검색을 가능하게 하며, 알고리즘은 데이터 세트에서의 성능 향상을 입증하기 위한 체계적인 실험을 기반으로 합니다. 알고리즘 1에 설명된 인덱싱, 검색 및 생성 프로세스는 성능을 결정짓는 주요 요소입니다.

- **Performance Highlights**: Qwen2.5-7B-Instruct 모델의 경우 초기 성능이 60점에서 74.5점으로 개선되는 등 유의미한 성능 향상을 달성하였습니다. 이는 기존 LLM이 어려움을 겪던 복잡한 쿼리 구조에서도 정확한 검색을 가능하게 함을 보여줍니다. 후보 데이터셋에서 실험이 이루어졌으며, 두 수준 검색이 불확실성 일치(fuzzy matching)을 향상시키고 논리 기반 검색이 구조적 추론을 개선하는 데 기여했습니다.



### Enhancing Layer Attention Efficiency through Pruning Redundant Retrievals (https://arxiv.org/abs/2503.06473)
Comments:
          11 pages, 7 figures

- **What's New**: 본 논문에서는 깊은 신경망의 층 간 상호작용을 향상시키는 새로운 접근 방식을 소개합니다. 특히, Kullback-Leibler (KL) divergence를 활용하여 인접 층 간의 중복성을 정량화하는 방법을 제안합니다. 이를 통해 기존의 layer attention 메커니즘에서 발생하는 중복 문제를 해결하고, Enhanced Beta Quantile Mapping (EBQM) 알고리즘을 도입하여 중복 층을 효과적으로 건너뛰도록 합니다.

- **Technical Details**: 논문에서 제안하는 Efficient Layer Attention (ELA) 구조는 중복성을 줄여 30%의 훈련 시간 단축과 함께 성능 향상을 이룹니다. KL divergence를 사용하여 각 층의 attention weights의 유사성을 평가하고, 이를 통해 비슷한 성능을 가진 인접 층을 식별합니다. 또한, EBQM 알고리즘은 KL divergence의 분포를 적절하게 조정하여 어떤 층을 제거할지 안정적으로 결정할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, ELA 구조는 다양한 작업에서 최첨단 기법 대비 속도와 정확성을 모두 향상시켰습니다. 예를 들어, 이미지 분류 및 객체 탐지 작업에서 성능이 개선되었으며, 이는 제안된 접근 방식이 실질적인 결과를 제공함을 보여줍니다. 본 연구는 기존의 layer attention 메커니즘의 한계를 극복함으로써 모델의 성능과 훈련 효율성을 동시에 강화하는 데 기여합니다.



### StructGS: Adaptive Spherical Harmonics and Rendering Enhancements for Superior 3D Gaussian Splatting (https://arxiv.org/abs/2503.06462)
- **What's New**: 최근 3D 복원 기술의 발전과 함께 신경 렌더링 기법이 사진 실사 같은 3D 장면 생성에 크게 기여하고 있으며, 이는 학술 및 산업 응용 분야에 많은 영향을 미치고 있습니다. 본 연구에서는 3D Gaussian Splatting(3DGS)을 향상시킨 StructGS라는 새로운 프레임워크를 소개하며, 이는 효과적인 새로운 시점 합성을 가능하게 합니다. StructGS는 패치 기반 SSIM 손실, 동적 구형 함수 초기화 및 다중 스케일 잔차 네트워크(MSRN)를 혁신적으로 통합하여 기존 모델들이 가지고 있는 한계점을 해결하고 있습니다.

- **Technical Details**: StructGS는 고해상도 이미지를 생성할 수 있도록 3DGS의 성능을 향상시키는 프레임워크입니다. 이 기술은 비선형 구조 유사성을 효과적으로 캡처하기 위해 패치 SSIM 손실을 사용하고, Gaussian 구의 투명도를 고려하여 구형 함수의 초기화 및 최적화를 위한 동적 조정 전략을 설계합니다. 또한, 미리 훈련된 MSRN을 통합하여 저해상도 입력 이미지로부터 고해상도의 고품질 이미지를 생성할 수 있습니다.

- **Performance Highlights**: StructGS는 최신 3DGS 기반 모델들보다 우수한 성능을 보이며, 훈련 반복 횟수를 줄이면서 더 나은 품질을 달성하고 있습니다. 또한, 설계된 손실 함수와 MSRN을 통해 고해상도 이미지의 렌더링 품질이 향상된다는 점이 실험적으로 입증되었습니다. 이를 통해 더 정교한 텍스처와 복잡한 기하학을 처리할 수 있는 능력을 갖추게 되었습니다.



### Geometric Knowledge-Guided Localized Global Distribution Alignment for Federated Learning (https://arxiv.org/abs/2503.06457)
Comments:
          Accepted by CVPR 2025

- **What's New**: 이번 연구에서는 Federated Learning (FL)에서의 데이터 이질성 문제를 해결하기 위해 새로운 데이터 생성 방법을 제안합니다. 이 방법은 로컬에서 글로벌 임베딩 분포를 시뮬레이션하여 데이터의 불균형 문제를 완화하고자 합니다. 특히, 레이블 스큐(label skew)와 도메인 스큐(domain skew)가 공존하는 상황에서도 모델의 성능을 향상시킬 수 있는 접근법을 제시합니다.

- **Technical Details**: 이 연구에서는 데이터 분포의 기하학적 형태를 개념화하고, 이를 통해 로컬과 글로벌 데이터 분포 간의 불일치를 해결하는 데 중점을 둡니다. GGEUR(Global Geometry-Guided Embedding Uncertainty Representation) 방법을 통해 로컬 클라이언트에서 신규 샘플을 생성하고, MLP를 통해 해당 데이터로 훈련할 수 있습니다. 또한, 다중 도메인 시나리오에서도 각 카테고리의 기하학적 분포가 유사하다는 점을 이용하여 글로벌 기하학적 형태를 근사화합니다.

- **Performance Highlights**: 실험 결과, 본 연구에서 제안한 방법이 레이블 스큐와 도메인 스큐 상황에서 기존 접근법의 성능을 크게 개선함을 보여줍니다. 이러한 개선은 Highly Heterogeneous Data 시나리오에서 최신 성과(state-of-the-art results)를 달성하는 데 기여합니다. 이 연구는 시각적 모델과 기하학적 지식을 결합하여 FL 분야에서의 시너지를 보여주는 전형적인 사례로 평가됩니다.



### CtrTab: Tabular Data Synthesis with High-Dimensional and Limited Data (https://arxiv.org/abs/2503.06444)
- **What's New**: 본 논문에서는 CtrTab이라는 조건 제어 확산 모델을 제안하여 고차원 데이터에서의 생성 모델 성능을 향상시킵니다. 기존의 확산 기반 모델이 고차원에서 효율성이 떨어지는 문제를 해결하고, 학습 데이터의 다양성을 높이기 위해 샘플에 Laplace noise를 추가하는 방법을 소개합니다. CtrTab은 실제 데이터와 유사한 합성 데이터를 생성하면서도 개인 정보를 보호하는 데 효과적입니다.

- **Technical Details**: CtrTab은 조건부 생성의 개념을 도입하여 기존의 무조건적 생성 방식과 차별화됩니다. 이 모델은 Denoising Diffusion Probabilistic Model(CFG DDPM)에 기반하며, 특정 조건을 명시적으로 반영한 제어 모듈을 포함하고 있습니다. Noise injection training 방법을 통해 Laplace noise를 주입하여 데이터의 다양성을 높이는 동시에, 우리의 방법이 사실상 L2 정규화와 유사하다는 이론적 근거를 제시합니다.

- **Performance Highlights**: 다양한 데이터셋에 대한 실험 결과, CtrTab은 기존의 최첨단 모델들보다 평균 80% 정확도 향상을 보여 줍니다. 특히, CtrTab으로 훈련된 모델은 내구성과 신뢰성을 높이며 복잡한 분포를 학습하는 데 더 효과적입니다. 이러한 성과는 CtrTab이 고차원 저 데이터 환경에서 탁월한 성능을 발휘함을 입증합니다.



### Physics-Informed Residual Neural Ordinary Differential Equations for Enhanced Tropical Cyclone Intensity Forecasting (https://arxiv.org/abs/2503.06436)
Comments:
          14 pages, 9 figures

- **What's New**: 이번 연구에서는 물리 정보를 활용한 잔여 신경망 상미분 방정식(Physics-Informed Residual Neural Ordinary Differential Equation, PIR-NODE) 모델을 제안하여 열대성 사이클론(Tropical Cyclone, TC) 강도를 보다 정확히 예측하는 방법을 보여줍니다. 이 모델은 신경망의 비선형 적합 기능을 활용하고, 잔여 연결(residual connection)을 통해 모델 깊이와 훈련의 안정성을 향상시킵니다. 또한, 신경 상미분 방정식(Neural ODE)을 통해 TC 강도의 연속적인 시간 진화를 명시적으로 모델링합니다.

- **Technical Details**: PIR-NODE 모델은 SHIPS 데이터셋을 활용하여 TC 강도를 예측하는 연구에 필요한 데이터 전처리와 특성 엔지니어링을 거쳤습니다. 원시 SHIPS 데이터에는 결측값과 이상치가 있어 이들에 대한 강력한 처리 전략이 사용되었습니다. 또한 데이터 특성을 물리적 지식을 반영하여 2차 및 상호작용 항 특성을 구축하여 복잡한 물리적 과정을 캡처할 수 있도록 설계되었습니다.

- **Performance Highlights**: 모델은 24시간 강도 예측에서 기존 신경망 방법 대비 25.2%의 제곱 평균 오차(RMSE) 감소 및 19.5%의 결정 계수(R2) 향상을 보여주었습니다. PIR-NODE 모델은 다양한 해양 유역과 폭풍 범주에서 강력한 성능을 나타내어 운영 예보 애플리케이션에 대한 잠재력을 강조합니다.



### Seesaw: High-throughput LLM Inference via Model Re-sharding (https://arxiv.org/abs/2503.06433)
- **What's New**: 이번 논문에서는 LLM 추론의 효율성을 높이기 위한 새로운 방법인 Seesaw를 제안합니다. Seesaw는 네트워크의 두 단계인 prefill과 decode에 걸쳐 병렬화 전략을 동적으로 재구성하는 모델 리샤딩(dynamic model re-sharding) 기법을 통해 높은 처리량을 최적화합니다. 이를 통해 기존의 정적 방법이 가진 한계를 극복하고 각 단계의 컴퓨팅 특성에 맞춤형 전략을 적용합니다.

- **Technical Details**: Seesaw는 tiered KV cache buffering와 transition-minimizing scheduling을 활용하여 변동이 큰 리샤딩 오버헤드를 줄입니다. 이러한 방식은 prefill과 decode 단계 간의 재구성이 잦은 상황에서도 안정된 성능을 보장합니다. 또한, CPU 메모리를 보조 저장소로 활용하여 KV cache를 대량으로 저장하고, 효율적인 배치 처리를 가능하게 합니다.

- **Performance Highlights**: Seesaw의 실험 결과는 vLLM과 비교하여 평균 1.36배, 최대 1.78배의 처리량 증가를 보여줍니다. 이 시스템은 다양한 워크로드 및 하드웨어 구성에서 성능을 검증하였으며, 각 LLM 추론 단계에 최적화된 병렬화 전략의 중요성을 입증합니다.



### Graph Retrieval-Augmented LLM for Conversational Recommendation Systems (https://arxiv.org/abs/2503.06430)
Comments:
          Accepted by PAKDD 2025

- **What's New**: 이 논문에서는 G-CRS(Graph Retrieval-Augmented Large Language Model for Conversational Recommender Systems)라는 새로운 툴을 소개합니다. G-CRS는 훈련이 필요 없는 프레임워크로, 그래프 기반 정보 검색과 ICL(In-Context Learning)을 결합하여 LLM(대형 언어 모델)의 추천 능력을 향상시킵니다. 기존의 방법들이 가지는 도메인 지식 부족 문제를 해결하고, 의미론적 관계와 사용자 상호작용을 더 효과적으로 캡처합니다.

- **Technical Details**: G-CRS는 두 단계의 검색-추천 아키텍처를 사용하여, 첫 단계에서 GNN(그래프 신경망) 기반 추론기가 후보 아이템을 식별하고, 이후 Personalized PageRank(PPR) 알고리즘을 통해 사용자 관심에 맞는 추가 아이템을 탐색합니다. 이 과정에서 기존 대화의 이력을 활용하여 LLM이 현재 대화에서의 사용자 선호를 더 잘 이해할 수 있도록 돕습니다. 이 방법은 기존의 RAG(검색을 통한 생성) 접근법을 향상시키며, 특정 작업 훈련 없이도 효과적인 추천이 가능합니다.

- **Performance Highlights**: G-CRS는 두 개의 공개 데이터셋에서 실험을 통해 기존 방법들보다 우수한 추천 성능을 보였습니다. 추가적인 모델 훈련 없이도 G-CRS의 프레임워크는 추천 정확도를 향상시키는 데 성공하였으며, 이는 대화 기반 추천 시스템의 실용성을 크게 높이는 결과로 이어집니다. 이 연구는 대화형 추천 시스템의 발전에 기여할 뿐 아니라, 도메인 특정 지식의 필요성을 줄이는 효과를 가지고 있습니다.



### Pre-Training Meta-Rule Selection Policy for Visual Generative Abductive Learning (https://arxiv.org/abs/2503.06427)
Comments:
          Published as a conference paper at IJCLR'24

- **What's New**: 이 논문은 시각 생성 유도 학습 (visual generative abductive learning) 분야의 최신 연구를 다루고 있습니다. 메타-규칙 선택 정책 (meta-rule selection policy)을 학습하기 위한 사전 훈련 전략을 제안하여, 시각 생성 과정이 유도된 논리 규칙에 의해 안내될 수 있도록 합니다. 이 과정은 시간 비용을 줄이는 데 기여하며, 이는 특히 대규모의 논리 기호 집합과 복잡한 논리 규칙이 있을 때 중요합니다.

- **Technical Details**: 제안된 사전 훈련 방법은 기호 기초 (symbol grounding) 학습 없이 순수 기호 데이터에서 수행됩니다. 이는 전체 학습 과정이 낮은 비용으로 진행된다는 장점을 제공합니다. 또한 선택 모델은 사례의 기호 기초와 메타-규칙의 임베딩 표현을 기반으로 구축되어, 신경 모델과 논리 추론 시스템에 효과적으로 통합될 수 있습니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안된 방법은 시각 생성 유도 학습에서 메타-규칙 선택 문제를 효과적으로 해결하여, 유도 과정의 효율성을 크게 향상시킬 수 있음을 보여줍니다. 주목할 점은 사전 훈련 중에 보지 못한 기호 기초 오류를 수정할 수 있는 선택 정책의 강력한 관용성을 관찰할 수 있었다는 것입니다. 이는 주의 메커니즘의 기억 능력과 기호 패턴의 상대적 안정성을 증명하는 결과입니다.



### GenAI for Simulation Model in Model-Based Systems Engineering (https://arxiv.org/abs/2503.06422)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 이번 연구에서는 Model-Based Systems Engineering (MBSE)을 위한 생성 시스템 설계 방법론 프레임워크를 제안합니다. 이 방법론은 시스템 물리적 특성에 대한 시뮬레이션 모델의 지능적 생성을 위한 실용적인 접근법을 제공합니다. 기존의 설계 문서를 기반으로 하여 생성 모델과 통합된 모델링 및 시뮬레이션 언어를 사용하여 시뮬레이션 모델을 구축하는 과정이 포함됩니다.

- **Technical Details**: 연구에서는 DEVS(Discrete Event System Specification) 기반의 X 언어를 사용하여 시뮬레이션 모델 템플릿을 구축합니다. 이 템플릿들은 BERT 및 Transformer 기반 모델을 활용하여 제품 설계 문서에서 모델의 구성, 아키텍처, 동작 정보를 추출하고, 이를 바탕으로 시스템 물리적 특성을 반영한 시뮬레이션 모델을 생성합니다. 또한, 생성된 모델의 질을 평가하기 위한 평가 메트릭을 소개합니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안된 시뮬레이션 모델 생성 방법은 주류 오픈 소스 Transformer 기반 모델에서 생성된 시뮬레이션 모델의 품질을 유의미하게 향상시키는 것으로 나타났습니다. 이러한 접근은 엔지니어들이 시스템 개발 과정에서 보다 효율적으로 작업할 수 있도록 도와줍니다. 전반적으로, 제안된 생성 방법론은 복잡한 제품 연구 및 개발을 위한 유망한 방향성을 제시합니다.



### Swift Hydra: Self-Reinforcing Generative Framework for Anomaly Detection with Multiple Mamba Models (https://arxiv.org/abs/2503.06413)
- **What's New**: Swift Hydra는 생성 AI와 강화 학습을 바탕으로 한 새로운 프레임워크로, 주로 unseen anomalies(보이지 않는 이상치)에 대한 일반화 능력을 향상시키기 위해 개발되었습니다. 이 프레임워크는 강화 학습 정책이 생성 모델의 잠재 변수를 통해 작동하여, 탐지 모델을 우회할 수 있는 참신하고 다양한 이상치 샘플을 합성합니다. 또한, Swift Hydra는 Mixture of Experts (MoE) 구조로 설계된 Mamba 모델을 포함하여, 데이터 복잡도에 따라 전문가의 수를 조절할 수 있도록 해줍니다.

- **Technical Details**: Swift Hydra는 Conditional Variational Autoencoder (C-VAE) 모델과 함께 작동하는 강화 학습(강화) 에이전트를 통해 전략적으로 샘플을 생성합니다. 이 에이전트는 잠재 공간에서 행동을 탐색하며, 생성된 샘플의 엔트로피와 탐지 회피 능력을 균형 있게 유지하는 보상 함수를 사용합니다. 또한, Mixture of Mamba Experts로 구성된 모델을 훈련하여 입력 데이터에 대해 복잡도를 확장할 수 있으며, 불필요한 추론 시간을 증가시키지 않아 실시간 애플리케이션에 적합합니다.

- **Performance Highlights**: ADBench 벤치마크를 통해, Swift Hydra는 다른 최신 anomaly detection(이상 탐지) 모델들보다 높은 정확도를 기록하며 비교적 짧은 추론 시간을 유지합니다. 실험 결과로부터, 본 연구는 RL과 생성 AI의 통합이 anomaly detection의 발전을 위한 새로운 유망 패러다임을 제시한다고 강조합니다. Swift Hydra는 실제 데이터에서 낮은 오류율을 달성하며, 이는 중요한 시스템에서도 핵심적인 역할을 할 것으로 기대됩니다.



### Decoding the Black Box: Integrating Moral Imagination with Technical AI Governanc (https://arxiv.org/abs/2503.06411)
- **What's New**: 이 논문은 AI 안전, 보안, 거버넌스의 복잡한 상호작용을 분석하며, 기술 시스템 공학과 윤리 철학의 원칙을 통합합니다. 저자들은 국방, 금융, 헬스케어, 교육 등 핵심 분야에서 AI 기술을 규제하기 위한 포괄적인 다차원 프레임워크를 개발하였습니다. 이 프레임워크는 강력한 기술 분석, 정량적 위험 평가 및 규범적 평가를 결합하며, 불투명한 블랙박스 모델의 시스템적 취약성을 드러냅니다.

- **Technical Details**: 논문에서는 O’Neil(2016)의 'Weapons of Math Destruction'와 시스템적 사고(Meadows, 2008)의 기초 개념을 바탕으로, AI의 기술적 취약성과 윤리적 문제를 다루는 다차원 규제 프레임워크를 제안합니다. 이 프레임워크는 기술적 안전장치, 윤리적 감독 및 시스템적 거버넌스를 통합하여, AI 시스템의 공정성, 책임성 및 공공선을 보장하는 것을 목표로 하고 있습니다.

- **Performance Highlights**: 실제 사례 연구로는 Microsoft Tay(2016)와 UK A-Level Grading Algorithm(2020)이 포함되며, 이들 사례를 통해 보안 약점, 편향 증폭 및 책임 부족이 어떻게 대중 신뢰를 저하시킬 수 있는지를 보여줍니다. 결론적으로, AI의 복원력을 강화하기 위한 적응형 규제 메커니즘과 강력한 보안 프로토콜을 제안하며, 윤리적 및 기술적 AI 거버넌스의 발전을 위한 전략을 제시합니다.



### Heterogeneous bimodal attention fusion for speech emotion recognition (https://arxiv.org/abs/2503.06405)
- **What's New**: 이번 연구에서는 Heterogeneous Bimodal Attention Fusion (HBAF)이라는 새로운 프레임워크를 제안하여 대화 감정 인식에 대한 다중 레벨 다중 모달 상호작용을 해결하고자 합니다. 이 방법은 저수준 오디오 표현과 고수준 텍스트 표현 간의 이질적인 갭을 극복하고, 더 효과적인 감정 인식을 가능하게 합니다. HBAF는 uni-modal representation 모듈, multi-modal fusion 모듈, inter-modal contrastive learning 모듈로 구성되어 있습니다.

- **Technical Details**: 제안된 HBAF 방법은 각 모달의 특성을 효과적으로 융합하기 위해 다차원 상호작용을 탐색할 수 있는 역동적인 bimodal attention 메커니즘을 활용합니다. 또한, uni-modal representation 모듈을 통해 저수준 오디오 표현에 컨텍스트 정보를 통합하여 이질적인 모달 갭을 해소합니다. 마지막으로, inter-modal contrastive learning 모듈은 오디오와 텍스트 간의 복잡한 상호작용을 포착하여 감정 인식의 정확성을 높입니다.

- **Performance Highlights**: MELD 및 IEMOCAP 데이터셋에 대한 실험 결과, 제안된 HBAF 방법이 기존 최첨단 기법에 비해 우수한 성능을 발휘함을 보여줍니다. 이는 특정한 모달리티 간의 상호작용을 효과적으로 모델링하여 감정 인식 정확도를 크게 향상시킨 결과입니다. 연구 결과는 다중 모달 감정 인식을 위한 새로운 접근법의 효과를 입증하고 있습니다.



### Causality Enhanced Origin-Destination Flow Prediction in Data-Scarce Cities (https://arxiv.org/abs/2503.06398)
- **What's New**: 본 논문에서는 도시 간의 지식을 이전하여 데이터 부족 도시에서 OD 흐름 예측의 정확성을 향상시키기 위한 새로운 Causality-Enhanced OD Flow Prediction (CE-OFP) 모델을 제안합니다. CE-OFP는 데이터가 풍부한 도시의 인과 그래프를 활용하여 데이터가 부족한 도시의 특성을 재구성하고 OD 흐름 예측을 수행합니다. 이를 통해 데이터 부족 문제를 해결하고 개발 중인 도시에서의 예측 능력을 개선하고자 합니다.

- **Technical Details**: CE-OFP는 Causality Enhanced Variational Auto-Encoder (CE-VAE)를 기반으로 하여 도시 지역 특성 간의 보편적인 인과 관계를 탐색합니다. 이 모델은 데이터가 풍부한 도시에서 관측된 특성과 누락된 특성 간의 관계를 모델링하여, 각 도시의 특징을 일반화할 수 있도록 돕습니다. 또한 GAT 기반 지식 증류 방법을 활용하여 OD 예측 모델을 이전하여, 데이터가 부족한 도시의 예측 성능을 향상시키는 데 기여합니다.

- **Performance Highlights**: 제안된 CE-OFP 모델은 실제 데이터 세트를 통한 광범위한 실험을 통해 기존 최신 기술 대비 OD 흐름 예측 정확성을 최대 11% 향상시킴을 입증하였습니다. 이 연구는 데이터가 부족한 도시에서의 OD 흐름 예측 문제를 해결하기 위한 새로운 접근 방식을 제시하며, 도시 전반에 걸쳐 예측 성능을 크게 개선할 수 있는 가능성을 보여줍니다.



### EPR-GAIL: An EPR-Enhanced Hierarchical Imitation Learning Framework to Simulate Complex User Consumption Behaviors (https://arxiv.org/abs/2503.06392)
- **What's New**: 본 논문에서는 사용자 소비 행동 데이터의 생성 방식에 대한 새로운 접근법을 제안합니다. 기존의 데이터 기반 방법과 모델 기반 방법의 한계를 극복하기 위해 Generative Adversarial Imitation Learning (GAIL) 방법에 Exploration and Preferential Return (EPR) 모델을 결합한 EPR-GAIL 프레임워크를 도입하였습니다. 이 프레임워크는 복잡한 소비 행동을 모델링할 수 있는 가능성을 보여 주며, 더 나은 품질의 소비 데이터를 생성하는 것을 목표로 하고 있습니다.

- **Technical Details**: EPR-GAIL 프레임워크는 사용자의 구매, 탐색, 선호 결정을 포함하는 복합적인 EPR 결정 프로세스를 모델링합니다. 생성자는 EPR 결정 프로세스의 실현으로서 계층적 정책 함수를 설계하였고, 구별자는 EPR 모델의 확률 분포를 보상 함수에 적용합니다. 이를 통해 사용자 소비 행동 데이터의 신뢰성과 충실도를 향상시킵니다.

- **Performance Highlights**: 실제 사용자 소비 행동 데이터셋을 활용한 광범위한 실험 결과, EPR-GAIL 프레임워크는 데이터 충실도 측면에서 최신 기술 대비 19% 이상 성능 개선을 보였습니다. 또한 생성된 소비 행위 데이터는 판매 예측 성능을 35.29%, 위치 추천 성능을 11.19% 향상시키며, 실제 응용에서의 장점을 입증하였습니다.



### VORTEX: Challenging CNNs at Texture Recognition by using Vision Transformers with Orderless and Randomized Token Encodings (https://arxiv.org/abs/2503.06368)
- **What's New**: 본 논문에서는 패턴 인식을 위한 Vision Transformers (ViT) 기반의 새로운 텍스처 인식 방법인 VORTEX를 제안합니다. VORTEX는 다중 깊이 토큰 임베딩을 추출하고, 경량 모듈을 통해 계층적 특징을 집계하여 순서 없는 인코딩을 수행합니다. 이 접근 방식은 기존의 CNN 기술과 비교하여 텍스처 분석에서 우수한 성능을 보여줄 수 있는 가능성을 가지고 있습니다.

- **Technical Details**: VORTEX는 ViT 백본에서 추출한 텍스처 특징을 기반으로 하며, 기존 모델과의 통합이 용이하여 아키텍처의 수정이나 백본의 미세 조정 없이도 활용됩니다. 특징은 단순 선형 SVM에 공급되며, 실험은 아홉 개의 다양한 텍스처 데이터 세트에서 수행되어 결과를 검증하였습니다. VORTEX 구조의 전반적인 모습은 입력 텍스처 이미지와 ViT 백본을 사용하여 이미지 표현을 얻는 방식으로 이루어져 있습니다.

- **Performance Highlights**: 제안된 VORTEX는 다양한 텍스처 분석 시나리오에서 SOTA 성능을 달성하거나 초과할 수 있는 능력을 입증했습니다. VORTEX는 비슷한 비용을 가진 CNN과 비교했을 때 더욱 향상된 계산 효율성을 보여주며, 이는 텍스처 인식 분야에서 ViT 기반 모델의 도입을 촉진할 것으로 기대됩니다.



### Machine Learning meets Algebraic Combinatorics: A Suite of Datasets Capturing Research-level Conjecturing Ability in Pure Mathematics (https://arxiv.org/abs/2503.06366)
Comments:
          26 pages, comments welcome

- **What's New**: 이 논문에서는 대수 조합론(Algebraic Combinatorics)의 기초 결과 및 미해결 문제를 다루는 새로운 데이터셋 모음인 ACD Repo(Algebraic Combinatorics Dataset Repository)를 소개합니다. 기존의 수학 자료들은 주로 고등학교나 대학 수준에 그쳤으나, 이 데이터셋은 전문 수학자들이 직면하는 불확실한 문제의 레벨에 맞춰 개발되었습니다. 해당 데이터셋은 최대 1,000만 개의 예제와 함께 수학적 추론을 위한 오픈엔드 질문을 제공합니다.

- **Technical Details**: 이 데이터셋은 9개로 구성되어 있으며, 각각은 열려 있는 수학적 질문과 연계된 ML(머신 러닝) 친화적인 작업을 포함하고 있습니다. 여기서 ML 모델이 효과적으로 작업을 해결할 수 있다면, 이는 더 넓은 수학적 질문에 대한 통찰력을 제공할 수 있는 정보를 학습했음을 의미합니다. 저자들은 각 데이터셋에 대한 배경 및 동기, 기본 통계, 그리고 기본 모델 성능을 포함하여 다양한 머신러닝 응용 방법을 설명합니다.

- **Performance Highlights**: 이 데이터셋은 전통적인 벤치마크와는 달리, 표준 메트릭에서의 높은 성능이 수학적 통찰력을 도출하지 못할 경우 가치가 없을 수 있음을 강조합니다. 예를 들어, 성능 분석을 통해 깊이 있는 해석을 제공하거나 LLM(대규모 언어 모델)을 이용해 수학적 추론을 코드로 전달하는 방식으로 해결책을 제안합니다. 이러한 데이터셋이 앞으로 더 효과적인 접근 방법 개발에 기여할 수 있기를 희망합니다.



### The AI Pentad, the CHARME$^{2}$D Model, and an Assessment of Current-State AI Regulation (https://arxiv.org/abs/2503.06353)
- **What's New**: 이 논문은 인공지능(AI) 규제를 위한 통일된 모델을 확립하는 데 초점을 맞추고 있습니다. AI의 핵심 컴포넌트 관점에서 접근하여 AI의 다섯 가지 필수 구성 요소를 설명하는 AI Pentad를 소개합니다. 또한, AI 규제의 여러 요소들, 즉 AI 등록, 모니터링 및 집행 메커니즘을 검토합니다.

- **Technical Details**: 이 논문에서 제안하는 CHARME$^{2}$D 모델은 AI Pentad와 AI 규제 요소 간의 관계를 심층적으로 탐구합니다. AI Pentad는 인간 및 조직, 알고리즘, 데이터, 컴퓨팅, 에너지 등 다섯 가지 핵심 요소로 구성됩니다. 이 모델은 각 요소가 규제 체계에서 어떻게 상호작용하는지를 분석할 수 있는 프레임워크를 제공합니다.

- **Performance Highlights**: 논문에서는 유럽연합(EU), 중국, 아랍에미리트(UAE), 영국(UK), 미국(US) 등에서의 AI 규제 노력을 평가합니다. 각 국가의 규제의 강점, 약점 및 격차를 강조하여 미래의 AI 관련 입법 작업에 대한 통찰을 제공합니다. 이러한 비교 평가는 AI 규제 개발에 있어 더 나은 정책 수립을 위한 기초 자료로 기능할 수 있습니다.



### Studying the Interplay Between the Actor and Critic Representations in Reinforcement Learning (https://arxiv.org/abs/2503.06343)
Comments:
          Published as a conference paper at ICLR 2025. 10 pages

- **What's New**: 이 논문에서는 actor-critic 알고리즘의 정보 표현에 대한 연구를 통해, 각각의 actor와 critic이 별도의 표현을 가질 때 더 나은 성능을 발휘함을 보여줍니다. 특히, actor의 표현은 행동과 관련된 정보를 집중적으로 추출하고, critic의 표현은 가치와 역학 정보를 인코딩하는 데 전문화됩니다. 이러한 발견은 강화 학습(RL)에서의 성능 향상과 데이터 수집 과정에서의 critic의 중요성을 강조합니다.

- **Technical Details**: 연구에서는 actor-critic 알고리즘의 최적 표현을 이해하기 위한 이론적 특성화를 수행하고, 서로 다른 표현 학습 접근 방식을 검토합니다. 총 3개의 on-policy actor-critic 알고리즘을 실험하여 그들의 특화된 성능을 평가했습니다. 결과적으로, actor와 critic이 분리된 경우 샘플 효율성과 일반화 능력이 향상됨을 확인했습니다.

- **Performance Highlights**: 실험 결과, critic이 별도로 분리되었을 때 탐색 및 데이터 수집에서 중요한 역할을 수행한다는 사실이 밝혀졌습니다. 이는 기존 모델에 비해 그들의 전반적인 성능 향상으로 이어졌으며, 각 알고리즘의 각각의 표현이 어떻게 영향을 미치는지를 명확히 보여주었습니다. 논문에 제시된 실험은 이러한 결과를 뒷받침하는 엄격한 실증 연구를 기반으로 합니다.



### States of LLM-generated Texts and Phase Transitions between them (https://arxiv.org/abs/2503.06330)
Comments:
          Published as a conference paper at MathAI 2025

- **What's New**: 본 연구는 인간이 작성한 텍스트와 LLM(대형 언어 모델)에서 생성된 텍스트의 자기상관 분포가 qualitatively 다른 점을 실증적으로 보여줍니다. 특히, 텍스트 생성 시 온도(temperature) 매개변수의 변화에 따라 텍스트를 solid, critical state, gas로 분류할 수 있다는 것을 입증합니다. 이는 확률론적 자기회귀 언어 모델의 통계적 속성이 잘 이해되지 않고 있다는 점을 보완하고자 하는 시도입니다.

- **Technical Details**: 자기상관 함수(autocorrelation function)와 같은 다양한 통계적 지표를 통해 LLM이 생성한 텍스트의 세 가지 국면 - 주기적(Periodic), 임계(Critical) 및 비정형(Amorphous) - 을 정의하였습니다. 연구팀은 LLM 생성 텍스트의 임계온도에서 질서 있는 상태에서 비정형 상태로의 상전이가 발생한다고 발견했습니다. 이 과정에서 온도가 0.7에서 1.0 사이일 때 길이가 최대 2000단어에 이르는 자기상관이 전력법(Power law)으로 감소하는 현상을 확인하였습니다.

- **Performance Highlights**: Lippid와 Takahashi의 선행 연구와 비교할 때, 본 연구에서는 두 가지 최신 LLM인 Qwen2.5와 1.5B를 사용하여 보다 정확한 상전이 온도와 자기상관의 거동을 제시하였습니다. 특히, 주기적, 비정형의 상태 확립에 대한 실험 결과를 통해 LLM의 텍스트 생성 능력을 더욱 강화하였음을 알 수 있습니다. 이 결과는 기계 생성 텍스트와 인간 작성 텍스트 간의 중요한 통계적 차이를 보여주며, 각 모델이 텍스트 생성을 할 때 적용되는 매개변수가 갖는 의미에 대한 새로운 통찰을 제공합니다.



### Advancing Autonomous Vehicle Intelligence: Deep Learning and Multimodal LLM for Traffic Sign Recognition and Robust Lane Detection (https://arxiv.org/abs/2503.06313)
Comments:
          11 pages, 9 figures

- **What's New**: 이 논문에서는 자율주행차(AV)의 안전한 내비게이션을 위한 고급 심층 학습 기법과 다중모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)을 결합한 통합 접근 방식을 소개합니다. 이 접근 방식은 복잡하고 동적인 환경에서 도로 감지를 향상시키기 위한 것입니다. 특히, 다양한 기법을 평가하여 교통 표지판 인식에서 최고의 성능을 달성하고, 경량화된 MLLM 기반 프레임워크를 제안하여 소규모 데이터셋으로 직접적인 학습 조정을 가능하게 했습니다.

- **Technical Details**: 교통 표지판 인식에는 ResNet-50, YOLOv8, RT-DETR을 평가하여 각각 99.8%, 98.0%, 96.6%의 높은 정확도를 달성했습니다. 차선 감지에서는 CNN 기반의 세그멘테이션 방법에 다항 곡선 맞춤(polygonal curve fitting)을 추가하여 높은 정확도를 이끌어냈습니다. 제안된 MLLM 기반 프레임워크는 초기 사전 훈련 없이 다양한 차선 유형과 복잡한 교차로를 효과적으로 처리하여 차선 검출의 신뢰성을 높였습니다.

- **Performance Highlights**: 이 다중모달 접근법은 훈련 자원의 제약에도 불구하고, 다양한 조건에서도 뛰어난 추론 능력을 보여줍니다. 명확한 조건에서는 99.6%, 야간에는 93.0%의 차선 검출 정확도를 달성했고, 비 오는 날의 차선 불가시(88.4%)나 도로 노후화(95.6%)에 대한 추론에서도 탁월한 성능을 보였습니다. 이 포괄적인 프레임워크는 자율주행차의 인식 신뢰성을 크게 향상시켜, 다양한 도로 시나리오에서 안전한 자율주행에 기여합니다.



### Synergizing AI and Digital Twins for Next-Generation Network Optimization, Forecasting, and Security (https://arxiv.org/abs/2503.06302)
Comments:
          Accepted by IEEE Wireless Communications

- **What's New**: 디지털 네트워크 트윈(Digital Network Twins, DNTs)은 물리적 네트워크의 가상 표현으로, 네트워크 성능의 실시간 모니터링, 시뮬레이션 및 최적화를 가능하게 합니다. DNT에 기계 학습 기법을 통합함으로써, 특히 연합 학습(Federated Learning, FL)과 강화 학습(Reinforcement Learning, RL)을 통해 6G 네트워크의 복잡성을 관리하는 강력한 솔루션을 제시합니다. 이 논문은 DNT, FL 및 RL 기술의 시너지 효과를 분석하며, 네트워크 신뢰성 및 보안을 유지하는 데 필요한 주요 기술적 과제를 강조합니다.

- **Technical Details**: DNT는 여러 트윈 간의 상호작용을 조정하여 데이터 흐름과 네트워크의 복원력 및 적응성을 보장합니다. 기계 학습, 특히 FL과 RL이 DNT의 기능을 크게 향상시키며, 특히 FL은 여러 에이전트 간의 협력적 상호작용을 통해 데이터를 지역적으로 처리하고 모델 업데이트만 공유하도록 합니다. 이러한 접근 방식은 데이터 프라이버시를 보호하며, RL은 에이전트가 환경과 상호작용하며 최적 정책을 학습할 수 있게 하여 DNT가 실시간 결정 내림과 장기적인 보상을 극대화할 수 있게끔 돕습니다.

- **Performance Highlights**: 제안된 파이프라인을 통해 엣지 캐싱에서 80% 이상의 캐시 적중률을 달성하고, 자율 차량 시스템에서는 100%의 비충돌율을 보장하여 안전한 환경에서의 신뢰성을 입증하였습니다. 이를 통해 DNT와 ML 기술의 통합이 6G 네트워크 환경에서의 최적화 및 보안을 향상시킬 수 있는 잠재력을 보여줍니다. 아울러, 이 연구 결과는 차세대 지능적 네트워크 시스템의 자동화된 의사결정 및 문제 해결의 미래에 대한 통찰을 제공합니다.



### Single Domain Generalization with Adversarial Memory (https://arxiv.org/abs/2503.06288)
- **What's New**: 이번 연구에서는 Single Domain Generalization (SDG) 문제를 해결하기 위해 Adversarial Memory를 사용하는 새로운 방법론인 SDGAM을 제안합니다. 기존의 방법들이 다룰 수 없는 제한된 데이터 환경에서도 효과적인 일반화 모델을 설계할 수 있도록 돕습니다. 우리의 접근법은 훈련 종료 시까지 테스트 데이터에 대한 접근을 필요로 하지 않으며, 다양한 메모리 기능을 통해 훈련 데이터의 다양성을 극대화합니다.

- **Technical Details**: SDGAM은 메모리 기반 기능 증강 네트워크를 사용하여 훈련 기능을 불변 서브스페이스로 매핑합니다. 이는 메모리 뱅크에 있는 다양한 기능 벡터로 구성되며, 훈련 및 테스트 데이터 간의 분포를 암묵적으로 정렬합니다. 아울러, 적대적인 기능 생성 기법을 통해 훈련 도메인 분포를 초과하는 기능을 생성하고, 이를 통해 메모리 뱅크를 업데이트함으로써 특징의 다양성을 유지합니다.

- **Performance Highlights**: 실험 결과, SDGAM은 표준 단일 도메인 일반화 벤치마크에서 최첨단 성능을 달성하였습니다. 우리는 기존의 데이터 확대 방법이 미비했음을 지적하며, 제안하는 방법이 보다 일관된 성능 향상을 가져온다는 것을 입증했습니다. SDGAM은 단일 도메인에 초점을 맞춘 상황에서도 뛰어난 일반화 능력을 보여주고 있습니다.



### Your Large Vision-Language Model Only Needs A Few Attention Heads For Visual Grounding (https://arxiv.org/abs/2503.06287)
- **What's New**: 이번 연구에서는 대형 비전-언어 모델(LVLM)의 특정 주의 헤드가 시각적 기초 설계에 유용하다는 점을 발견했습니다. 특히, 저자들은 이러한 주의 헤드를 '로컬라이제이션 헤드(localization heads)'로 정의하고, 이를 활용하여 훈련이 필요 없는 간단하고 효과적인 시각적 기초 프레임워크를 제안합니다. 이 프레임워크는 텍스트-이미지 주의 맵을 이용해 대상 물체를 식별합니다.

- **Technical Details**: 로컬라이제이션 헤드는 개체에 대한 텍스트 의미와 관련된 지역을 지속적으로 캡처하는 몇 가지 주의 헤드를 나타냅니다. 연구에서는 이러한 헤드를 구체적으로 식별하는 두 가지 기준을 제시하며, (1) 각 주의 헤드가 이미지에 얼마나 집중하는지를 측정하고, (2) 특정 이미지 영역에 주의를 기울이는 헤드를 선택합니다. 이를 통해 최종적으로 세 개의 로컬라이제이션 헤드만으로도 적절한 경계 상자나 마스크를 예측할 수 있습니다.

- **Performance Highlights**: 혼합 성능 평가 결과, 제안된 방법은 기존의 훈련 기반 방법들보다 큰 폭으로 성능을 향상시켰습니다. 또한, 훈련이 필요 없는 방법 중에서도 우수한 성과를 거두며, 특정 훈련을 받은 LVLM과 비교해서도 유사한 성능을 보여줍니다. 이러한 결과들은 LVLM이 텍스트 표현과 일관된 영역을 본질적으로 식별할 수 있는 효과적인 도구임을 입증합니다.



### Applied Machine Learning Methods with Long-Short Term Memory Based Recurrent Neural Networks for Multivariate Temperature Prediction (https://arxiv.org/abs/2503.06278)
Comments:
          11 pages, 16 figures, private research

- **What's New**: 이 논문은 시계열 예측(time series prediction)을 위한 밀집(dense) 및 심층(deep) 신경망(neural network)의 개발 방법에 대한 개요를 제공합니다. 인공지능(Artificial Intelligence) 및 머신러닝(Machine Learning)의 역사와 기초를 소개하고, 다양한 신경망 모델을 사용하여 시계열 예측을 수행하는 기술을 깊이 있게 다룹니다. 파이썬(Python) 개발 환경 Jupyter와 텐서플로우(TensorFlow) 패키지, 딥러닝 애플리케이션 케라스(Keras)를 활용하여 시스템 설정 및 프로젝트 프레임워크를 설명합니다.

- **Technical Details**: 논문에서는 날씨 데이터를 이용한 시계열 예측의 적용 예를 보여주기 위해 Long Short-Term Memory (LSTM) 셀을 사용하는 심층 순환 신경망(deep recurrent neural network)을 사용합니다. LSTM은 특정 시간이 지나도 정보를 기억할 수 있는 구조로, 이는 시계열 데이터의 예측에 특히 유용합니다. 이 모델은 훈련과 예측 정확도를 높이기 위해 점진적으로 발전시키며, 예측 정확도가 유지될 수 있는 한계를 찾아냅니다.

- **Performance Highlights**: 결과 평가에 따르면, 심층 신경망을 통한 날씨 예측은 단기적으로 성공적인 성과를 나타내지만, 예측 지점이 증가함에 따라 정확도는 감소하는 한계를 지니고 있습니다. 이러한 한계를 논의하며, 시계열 예측에서의 ML 모델의 잠재력과 현재까지의 연구 성과를 반영합니다. 결과적으로, 이 연구는 더 나은 예측력을 위해 ML 모델을 최적화할 수 있는 방법을 제시합니다.



### Using Mechanistic Interpretability to Craft Adversarial Attacks against Large Language Models (https://arxiv.org/abs/2503.06269)
- **What's New**: 이번 논문에서는 기존의 백박스(white-box) 공격 방법의 한계를 극복하기 위해 기계적 해석(mechanistic interpretability) 기법을 적용한 새로운 접근 방식을 제안합니다. 이 방법은 모델의 내부 메커니즘을 고려하여 적대적 입력(adversarial inputs)을 생성하며, 기존 방식보다 더 효율적인 공격을 가능하게 합니다. 특히, 저자는 수용 서브스페이스(acceptance subspaces)를 식별하고, 이를 통해 차단 서브스페이스(refusal subspaces)에서 임베딩(embeddings)을 재편성하여 공격의 성과를 극대화합니다.

- **Technical Details**: 저자들은 먼저 수용 서브스페이스를 정의하여 모델이 거부하지 않는 기능 벡터(feature vector)의 집합을 찾습니다. 이후, 그라디언트 기반 최적화(gradient-based optimization)를 이용해 이를 활용하여 공격을 실현합니다. 이 방법은 공격 성공률이 80-95%에 달하며, Gemma2, Llama3.2, Qwen2.5와 같은 최첨단 모델에서 몇 분 또는 몇 초 안에 결과를 도출합니다.

- **Performance Highlights**: 이 새로운 기법은 기존의 방법들과 비교하여 훨씬 짧은 시간에 높은 성공률로 적대적 공격을 수행할 수 있습니다. 기존 기술들은 종종 수시간이 걸리거나 실패하는 경향이 있었음을 고려할 때, 이 방법은 사용자에게 높은 효율성을 보여줍니다. 또한, 기계적 해석의 실제 적용 가능성을 시사하고 있어 향후 공격 연구와 방어 개발에 중요한 이정표가 될 것입니다.



### Critical Foreign Policy Decisions (CFPD)-Benchmark: Measuring Diplomatic Preferences in Large Language Models (https://arxiv.org/abs/2503.06263)
- **What's New**: 이번 연구는 인공지능(AI)과 국가 안전 보장 간의 통합이 증가함에 따라 대형 언어 모델(LLM)의 편향을 평가하기 위한 독창적인 벤치마크를 제안합니다. 연구는 국제 관계(International Relations, IR)와 관련된 400개의 전문가 제작 시나리오를 이용해 여러 주요 모델의 편향과 선호도를 분석했습니다. 이를 통해 Qwen2 72B, Gemini 1.5 Pro-002, Llama 3.1 8B Instruct 모델이 다른 모델보다 군사적 에스컬레이션을 강화하는 추천을 더 많이 제공한다는 점을 발견했습니다.

- **Technical Details**: 이 연구에서는 군사 에스컬레이션, 군사 및 인도적 개입, 국제 시스템 내 협력적 행동, 동맹 역학과 같은 네 가지 주제를 중심으로 모델들의 추천을 분석했습니다. 각 모델은 특정 국가에 대한 편향을 보이며, 예를 들어 중국과 러시아에 대해서는 덜 에스컬레이션하고 개입적인 행동을 추천하는 경향이 있음을 관찰했습니다. 이 연구는 AI 모델의 응용이 부정적인 위험 없이 이루어지도록 하는 점검이 필수적임을 강조합니다.

- **Performance Highlights**: 모델 응답에서 중요한 차이가 나타났으며, 특히 에스컬레이션과 개입 분야에서 두드러진 변화를 보였습니다. 모델들은 편향된 행동을 보여주며, 특정 국가들에 대해 다르게 반응하는 것을 발견했습니다. 연구 결과는 LLM을 고위험 국가 안보 및 외교 정책 시나리오에 배치하는 것의 위험성을 강조하며, 향후 연구에서 도메인별 벤치마킹 및 평가가 중요하다고 강조합니다.



### From Captions to Rewards (CAREVL): Leveraging Large Language Model Experts for Enhanced Reward Modeling in Large Vision-Language Models (https://arxiv.org/abs/2503.06260)
- **What's New**: 대규모 비전-언어 모델(LVLM)을 인간의 선호도와 맞추는 것은 고급 다중 모달 선호 데이터 부족 때문에 어려운 도전입니다. 기존의 방법들은 저신뢰도 데이터로 인해 최적의 성능을 발휘하지 못하는 단점을 가지고 있습니다. 이에 대한 해결책으로 제안된 CAREVL은 신뢰성 있는 고신뢰도 및 저신뢰도 데이터를 활용한 선호 보상 모델링의 새로운 방법입니다.

- **Technical Details**: CAREVL은 첫째로, 보조 전문가 모델(auxiliary expert models)인 텍스트 보상 모델(textual reward models)을 활용하여 이미지 캡션을 약한 지도 신호(weak supervision signals)로 활용하여 고신뢰도 데이터를 필터링합니다. 그런 다음, 이 고신뢰도 데이터를 사용하여 LVLM을 미세 조정(fine-tuning)합니다. 둘째로, 저신뢰도 데이터를 활용하여 다양한 선호 샘플을 생성하며, 이 샘플들은 별도로 채점하여 신뢰 가능한 선택-거부 쌍을 구성하여 추가 훈련에 사용됩니다.

- **Performance Highlights**: CAREVL은 VL-RewardBench 및 MLLM-as-a-Judge 벤치마크에서 기존의 증류(distillation) 기반 방법들에 비해 성능 개선을 달성하였습니다. 이는 다양한 신뢰도 수준의 데이터를 효과적으로 활용함으로써 가능합니다. 앞으로 코드가 곧 공개될 예정입니다.



### Can Atomic Step Decomposition Enhance the Self-structured Reasoning of Multimodal Large Models? (https://arxiv.org/abs/2503.06252)
- **What's New**: 이번 논문에서는 '느린 사고' (slow thinking) 능력을 다중 모드 대형 언어 모델(MLLMs)에 통합하여 다중 모달 수학적 추론의 도전 과제를 다룹니다. 제안된 Self-structured Chain of Thought (SCoT) 패러다임은 복잡성에 따라 문제에 적합한 동적 추론 구조를 생성할 수 있도록 합니다. 기존 방법과 달리 구조화된 템플릿이나 자유형 패러다임에 의존하지 않고, 인지적 CoT 구조를 생성하여 과도한 사고를 줄일 수 있습니다.

- **Technical Details**: AtomThink라는 새로운 프레임워크는 데이터 엔진, 감시된 미세 조정 과정, 정책 기반의 다중 턴 추론 방법, 그리고 원자적 능력 메트릭을 포함한 네 개의 핵심 모듈로 구성됩니다. 이를 통해 20,000개의 고급 수학적 문제와 124,000개의 원자적 단계 주석이 포함된 AMATH 데이터셋을 생성했습니다. 이 모델은 원자 단계를 기반으로 하여 멀티 모달 작업에서 자가 구조화된 추론 능력을 활성화합니다.

- **Performance Highlights**: 우리는 MathVista, MathVerse 및 MathVision에서 10% 이상의 정확도 향상을 보여주었으며, LLaVA-CoT에 비해 500%의 데이터 활용성과 80% 이상의 추론 효율성을 개선했습니다. 이러한 성능 향상은 다중 모달 고도 추론 개선을 위한 중요한 기초를 제공합니다. 또한, 다양한 이해 능력의 분포와 MLLM의 성능을 분석하여 공유합니다.



### Infant Cry Detection Using Causal Temporal Representation (https://arxiv.org/abs/2503.06247)
Comments:
          Accepted to ICASSP 2025

- **What's New**: 이 논문은 아기 울음소리 탐지의 정확성을 높이기 위한 두 가지 주요 기여를 제안합니다. 첫 번째는 아기 울음소리를 상세히 주석 처리한 데이터셋으로, 이는 감독 학습(supervised learning) 모델이 최신 성능을 달성할 수 있도록 돕습니다. 두 번째는 인과적 시계열 표현(causal temporal representation)을 기반으로 한 새로운 비감독방법인 Causal Representation Sparse Transition Clustering (CRSTC)입니다.

- **Technical Details**: 이 비감독 학습 방법론은 라벨이 없는 데이터를 활용하여 인과적 시간적 표현을 통해 음향 이벤트를 탐지합니다. 연구팀은 아기 울음소리를 포함한 오디오 데이터를 분석하기 위해 수정된 변량 오토인코더(Variational Autoencoder), 즉 Sparse Transition Variational Autoencoder (ST-VAE)를 개발했습니다. 이 모델은 오디오 피처를 인코딩하고 시간에 따른 잠재 변수를 모델링하는 구조로 설계되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 CRSTC 방법론이 아기 울음소리 인식 및 분류 작업에서 성능을 향상시키는 데 효과적이라는 것을 입증했습니다. 데이터의 다양한 배경 소음 속에서도 아기 울음소리 세그먼트를 정확하게 탐지할 수 있어, 앞으로 아기 관리 어플리케이션에 큰 잠재력을 지닙니다. 연구팀은 제안된 데이터셋과 코드가 GitHub에서 공개되어 있어 다른 연구자들이 활용할 수 있도록 하고 있습니다.



### Image is All You Need: Towards Efficient and Effective Large Language Model-Based Recommender Systems (https://arxiv.org/abs/2503.06238)
- **What's New**: 이 논문에서는 LLM 기반 추천 시스템에서 항목 표현방식의 비효율성과 효율성 간의 균형을 다루기 위해 이미지 기반 접근 방식인 I-LLMRec을 제안합니다. 이 방법은 긴 텍스트 설명 대신 이미지를 활용하여 아이템을 표현함으로써 토큰 사용량을 줄이고 아이템 설명의 풍부한 의미 정보를 보존하는 것을 목표로 합니다. 여러 실제 데이터셋을 통해 I-LLMRec이 기존 방식에 비해 효율성과 효과성 모두에서 향상된 성과를 보임을 입증합니다.

- **Technical Details**: I-LLMRec은 이미지와 텍스트 간의 정보 겹침을 활용하여, 아이템 표현 시 이미지와 LLM의 언어 공간 간의 불일치를 극복하기 위해 학습 가능한 어댑터(Adaptor)와 이미지를 LLM에 맞추기 위한 기법을 도입합니다. 이를 통해 LLM이 적은 토큰 수로도 이미지에 대한 풍부한 의미 정보를 포착할 수 있도록 지원합니다. 주요 기술 도전과제는 아이템 이미지 공간과 언어 공간 사이의 불일치입니다.

- **Performance Highlights**: I-LLMRec은 설명 기반 표현 방식을 2.93배 향상시키고, 속도 및 효율성에 있어 Attribute 기반 표현 방식보다 22%의 성능 향상을 달성했습니다. 이 방법은 또한 아이템 설명의 노이즈에 대한 민감도를 줄여 보다 강력한 추천을 제공합니다. 실험 결과는 I-LLMRec이 다양한 자연어 기반 표현 방식보다 효과성과 효율성 모두에서 우수함을 입증합니다.



### A Frank System for Co-Evolutionary Hybrid Decision-Making (https://arxiv.org/abs/2503.06229)
Comments:
          13 pages

- **What's New**: Frank라는 새로운 인간-루프 시스템을 소개합니다. 이 시스템은 사용자가 라벨이 없는 데이터세트에서 레코드를 라벨링하는 과정을 지원합니다. Frank는 사용자의 결정과 함께 진화하는 점진적 학습(incremental learning)을 활용하며, 공정성 검사(fairness checks), 불일치 제어, 설명(explanations), 악의적인 사용자에 대한 보호 장치를 동시에 제공합니다.

- **Technical Details**: 이 시스템은 기존의 Skeptical Learning(SL) 개념을 확장하여, 공정성과 설명력, 사용자 감독의 참여를 함께 고려합니다. 또한 Frank는 지속적 학습(Incremental Learning) 모델을 기반으로 하여, 사용자의 라벨링 결정과 모델의 예측 간의 일관성을 강화합니다. 이는 사용자가 과거 결정과의 모순이 발생할 경우 경고를 받도록 하며, 점차적으로 더 많은 데이터를 학습하여 사용자에게 설명이 제공됩니다.

- **Performance Highlights**: 실험 결과에 따르면, Frank를 덜 신뢰할 수 있는 사용자와 결합할 경우 결정의 정확성과 공정성이 눈에 띄게 향상되는 것으로 나타났습니다. 또한 설명이 제공될 경우 사용자들이 제안에 대한 수용도가 높아져, 결정의 개선을 도모할 수 있습니다. Frank와 사용자의 상호 작용은 결국 서로의 행동을 예측하고 발전시키는 공생적인 관계를 형성합니다.



### Optimal Output Feedback Learning Control for Discrete-Time Linear Quadratic Regulation (https://arxiv.org/abs/2503.06226)
Comments:
          16 pages, 5 figures

- **What's New**: 이 논문은 알려지지 않은 이산 시간 시스템의 선형 이차 조절(Linear Quadratic Regulation, LQR) 문제를 동적 출력 피드백 학습 제어(Dynamic Output Feedback Learning Control)를 통해 연구합니다. 동적 출력 피드백 제어의 최적성은 상태 관측기의 수렴에 대한 암묵적인 조건을 필요로 하는데, 이를 해결하기 위해 보장된 수렴성과 안정성을 제공하는 일반화된 동적 출력 피드백 학습 제어 접근 방식이 제안되었습니다. 특히, 이 접근 방식은 상태 피드백 제어기와 동등하게 설계된 동적 출력 피드백 제어기를 사용하여 문제를 해결합니다.

- **Technical Details**: 이 연구에서는 두 가지 모델 없는 강화 학습 알고리즘인 정책 반복(Policy Iteration, PI) 방법과 가치 반복(Value Iteration, VI) 방법을 제안하여 최적 피드백 제어 이득을 추정합니다. 또한, 비특이(parameterization) 매트릭스를 찾아 모델 없는 안정성 기준을 제공하면서, 이는 스위치(iteration) 반복 방식을 구축하는 데 기여합니다. 제안된 출력 피드백 학습 제어 방법의 수렴, 안정성 및 최적성 분석이 포함되어 있으며, 이는 전체적인 제어 문제 해결의 논리를 체계적으로 다룹니다.

- **Performance Highlights**: 이론적 결과는 두 개의 수치 예제를 통해 검증되며, 제안된 동적 출력 피드백 학습 제어기는 상태 관측기의 수렴 없이도 상태 피드백 제어기와 동등성을 유지하는 고유한 특성을 갖추고 있습니다. 또한, 제안된 학습 알고리즘의 수렴성은 랭크 조건 하에서 보장되며, 탐사 노이즈 및 상태 관측기 오류의 영향을 면역적으로 처리할 수 있습니다. 이를 통해 불확실한 환경에서도 안정적인 제어 성능을 유지합니다.



### GraphGen+: Advancing Distributed Subgraph Generation and Graph Learning On Industrial Graphs (https://arxiv.org/abs/2503.06212)
Comments:
          Accepted By EuroSys 2025 (poster)

- **What's New**: 새로운 프레임워크 GraphGen+는 분산된 서브그래프 생성(distributed subgraph generation)과 메모리 내 그래프 학습(in-memory graph learning)의 통합을 통해 대규모 그래프 학습을 위한 혁신적인 솔루션을 제안합니다. 이 방법은 외부 스토리지의 필요성을 없애고, 처리 효율성을 크게 향상시킵니다. 특히, GraphGen+는 전통적인 SQL 유사 방법에 비해 27배의 속도 향상을 달성했습니다.

- **Technical Details**: GraphGen+의 구조는 그래프 파티셔닝(graph partitioning)과 로드 밸런스 서브그래프 매핑(load-balanced subgraph mapping)을 포함하여, 여러 작업 노드에서 병렬 처리할 수 있도록 설계되었습니다. 또한, MapReduce 기반의 접근 방식을 활용하여 서브그래프 추출을 수행하고, Tree-Reduction 기법을 통해 부하 균형을 개선했습니다. 이러한 방식은 고차 노드로 인한 성능 병목 현상을 감소시킵니다.

- **Performance Highlights**: 실험 결과, GraphGen+는 5억 3천만 노드와 50억 엣지를 가진 그래프에서 서브그래프 생성을 3분 만에 완료하며, 초당 590만 노드를 처리할 수 있음을 보여주었습니다. 이 시스템은 한 번의 반복에서 최대 100만 노드에 대한 학습을 지원하여 대규모 그래프 학습에 적합한 실용적인 솔루션으로 평가받고 있습니다. 최종적으로, GraphGen+는 산업 규모의 그래프 학습을 위한 확장 가능하고 효율적인 솔루션을 제공합니다.



### Text-Speech Language Models with Improved Cross-Modal Transfer by Aligning Abstraction Levels (https://arxiv.org/abs/2503.06211)
- **What's New**: 이 논문은 Text-Speech Language Models (TSLMs)의 훈련 방법에 변화를 제안합니다. 기존의 방법은 사전 훈련된 텍스트 LM의 어휘에 새로운 임베딩을 추가하는 방식이었지만, 이로 인해 크로스 모달 전이가 제한된다고 주장합니다. 새로운 기법인 	extsc{SmolTolk}를 통해, 모델의 다양한 레이어에서 추상화 수준을 보다 잘 정렬할 수 있도록 모듈을 추가하는 방식으로 개선했습니다.

- **Technical Details**: 제안된 	extsc{SmolTolk} 모델은 135백만에서 17억 개의 파라미터를 가진 SmolLM 모델 계열에 적용됩니다. 새로운 입력 및 출력에 대해 음성 특화 레이어를 추가하고, 학습 가능한 동적 레이어 풀링 메커니즘을 도입하여 텍스트 LM의 출력 표현이 저수준 및 다음 단어 예측 표현에 적절히 전환할 수 있도록 설계했습니다. 이러한 구조적 변화는 기존의 TSLM 방법론에 비해 크로스 모달 전이를 촉진합니다.

- **Performance Highlights**: 실험 결과, 	extsc{SmolTolk}는 일반적인 어휘 확장 방법으로 훈련된 기준 모델들을 지속적으로 초월하는 성능을 보였습니다. 가장 큰 모델인 SmolTolk-2B는 수십 배 더 큰 TSLM 모델과 경쟁하거나 그 이상으로 성능을 달성했습니다. 다양한 아블레이션 및 표현 분석을 통해 제안된 각 요소가 전반적인 성능 향상에 기여함을 확인했습니다.



### Distributed Graph Neural Network Inference With Just-In-Time Compilation For Industry-Scale Graphs (https://arxiv.org/abs/2503.06208)
Comments:
          Accepted by EuroSys 2025 (poster)

- **What's New**: 이 논문은 GNN(그래프 신경망) 추론의 성능 병목 현상을 해결하기 위해 새로운 분산 처리 패러다임을 제안합니다. 기존의 서브그래프 학습 방법의 단점을 극복하고, JIT(Just-In-Time) 컴파일 기술을 최대한 활용하여 GNN의 컴퓨팅 리소스를 분산 클러스터에서 효율적으로 활용할 수 있습니다. 이를 통해 대규모 그래프 처리에 대한 새로운 접근 방식을 제시하며, 다양한 생산 환경에서의 실제 적용 가능성을 보여줍니다.

- **Technical Details**: 제안된 패러다임은 DFOGraph(2021년) 기반으로 구현되었으며, 두 가지 처리 인터페이스와 두 가지 데이터 검색 함수를 포함합니다. 이 인터페이스는 노드 및 엣지 특성을 처리하며, 사용자가 메시지가 실제로 어떻게 전송되는지 걱정할 필요 없이 자동으로 관리됩니다. JIT 컴파일을 통해 희소 그래프 데이터를 밀집 행렬로 변환하여 추론 성능을 더욱 향상시키는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: 3개의 실제 비즈니스 시나리오를 분석한 결과, 제안된 방법은 HGT 모델에서 12.8배, GeniePath 모델에서 8배, GAT 모델에서 27.4배의 성능 향상을 보여주었습니다. 이러한 성능 향상은 전체 그래프 추론과 JIT 기술에 의해 주로 이루어졌으며, 이로 인해 대규모 그래프에서의 불필요한 계산과 시간을 효과적으로 줄일 수 있었습니다.



### CUPCase: Clinically Uncommon Patient Cases and Diagnoses Datas (https://arxiv.org/abs/2503.06204)
Comments:
          Accepted to AAAI 2025

- **What's New**: 이 논문에서는 Clinically Uncommon Patient Cases and Diagnosis Dataset (CUPCase)를 구축하여, LLMs (Large Language Models)의 진단 능력을 평가했습니다. CUPCase는 3,562개의 실제 환자 사례를 기반으로 하여, 병의 진단 정보를 가진 개방형 텍스트와 선택형 질문을 포함합니다. 이 데이터셋은 기존의 의료 영역 기준을 보완하여, 보다 다양한 임상 사례를 제시합니다.

- **Technical Details**: CUPCase 데이터셋은 BMC에서 제공하는 의료 사례 보고서에서 추출한 데이터로 구성됩니다. 연구에서는 두 가지 과제를 통해 LLMs의 성능을 평가했습니다: 첫 번째는 선택형 질문(multiple-choice question) 평가, 두 번째는 개방형 질문(open-ended question) 작성입니다. GPT-4o는 두 평가 모두에서 뛰어난 성능을 보여주었으며, 특히 87.9%의 정확도를 달성했습니다.

- **Performance Highlights**: 연구 결과, GPT-4o는 선택형 질문에서 평균 87.9%의 정확도를, 개방형 질문에서는 BERTScore F1이 0.764으로 측정되었습니다. 또한, GPT-4o는 전체 임상 정보의 20%만으로도 87%와 88%의 성능을 유지할 수 있음을 나타내어, 실제 임상 사례에서 조기 진단을 지원할 수 있는 가능성을 강조했습니다. CUPCase는 LLMs의 임상 의사결정 지원 능력을 공개적이고 재현 가능한 방식으로 평가할 수 있도록 합니다.



### Explainable Synthetic Image Detection through Diffusion Timestep Ensembling (https://arxiv.org/abs/2503.06201)
Comments:
          13 pages, 5 figures

- **What's New**: 이 연구에서는 최근 발전한 diffusion 모델이 생성한 이미지의 식별을 위한 새로운 방법을 제안합니다. 기존의 감지 방법이 진품 사진과 합성 이미지를 구분하는 데 어려움을 겪고 있는 가운데, 본 논문은 이미지의 고주파 성분에서 발생하는 차이를 활용하여 합성 이미지를 효과적으로 탐지할 수 있는 가능성을 보여줍니다. 또한, 이는 추론 기능을 추가하여 AI 생성 이미지의 결함을 식별하는 데도 기여합니다.

- **Technical Details**: ESIDE(Explainable Synthetic Image Detection through Diffusion Timestep Ensembling)라는 프레임워크를 제안하며, 이는 여러 타임스탬프에서 노이즈가 추가된 원본 이미지를 통해 분류기를 훈련시키는 방식을 채택하고 있습니다. 해당 프레임워크는 DDIM 반전 과정을 통해 생성된 여러 노이즈 버전을 이용하여 이미지를 처리하며, 이들 노이즈 이미지는 CLIP 이미지 인코더를 통해 특징 표현을 추출한 후 AdaBoost 모델에 입력됩니다. 이를 통해 각 모델은 이미지가 합성인지의 여부를 평가하며, 최종 예측은 이러한 평가를 기반으로 한 가중 합산으로 도출됩니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 GenImage 기준으로 정규 샘플에서 98.91%, 더 어려운 샘플에서 95.89%의 높은 탐지 정확도를 기록하며, 기존 방법보다 각각 2.51%, 3.46% 향상된 성능을 보였습니다. 또한, 이 방법은 다른 diffusion 모델이 생성한 이미지에 대해서도 효과적으로 일반화되는 특성을 가지고 있습니다. 공급된 데이터셋인 GenHard와 GenExplain은 더욱 높은 난이도의 탐지 샘플과 AI 생성 이미지에 대한 고품질 설명을 제공합니다.



### Human-AI Experience in Integrated Development Environments: A Systematic Literature Review (https://arxiv.org/abs/2503.06195)
Comments:
          Submitted to Empirical Software Engineering (EMSE) special issue Human-Centered AI for Software Engineering (HumanAISE), 28 pages, 1 figure

- **What's New**: 이 논문은 인공지능(AI)과 통합 개발 환경(IDE)의 통합에 대한 최근 연구를 체계적으로 정리합니다. AI는 이제 단순한 도구를 넘어서 개발자의 협력자로 작용하며, 인간-인공지능 경험(Human-AI Experience, HAX)의 새로운 장을 열고 있습니다. 다수의 연구 문헌을 검토하여 AI 지원 코딩 환경에서의 개발자 경험과 관련된 주제를 정리하고 미래 연구 방향을 제시하고 있습니다.

- **Technical Details**: 문헌 리뷰는 PRISMA(Preferred Reporting Items for Systematic Reviews and Meta-Analyses) 프레임워크를 기반으로 진행되었습니다. 2022년부터 2024년 사이에 발표된 254개의 연구를 분석하여 89개 연구를 선정하여, HAX와 관련된 주요 연구 목표, 방법론, 그리고 발견되어진 주요 테마를 규명하였습니다. 연구는 주로 질적 접근(43%)과 실험적 접근(35%)을 사용하였으며, AI의 통합이 소프트웨어 개발에 미치는 영향에 대한 전반적인 분석을 수행하였습니다.

- **Performance Highlights**: AI 도움으로 코딩의 생산성이 향상되지만, 특히 초보 개발자들 사이에서는 검증 부담, 자동화 편향, 과도한 의존 같은 문제들이 발생할 수 있습니다. GitHub Copilot을 포함한 다양한 AI 기반 개발 도구의 연구가 진행되고 있으며, AI의 역할을 소프트웨어 개발 전반에 걸쳐 조명해야 한다는 필요성이 대두되고 있습니다. 이는 AI 지원 개발의 장기적인 효과와 개인화된 대처 방안에 대한 추가 연구 기회를 제시합니다.



### MSConv: Multiplicative and Subtractive Convolution for Face Recognition (https://arxiv.org/abs/2503.06187)
- **What's New**: 이번 논문에서는 얼굴 인식 작업에서 중요한 특징인 salient feature와 differential feature의 균형 잡힌 학습을 위한 새로운 합성곱 모듈 MSConv(Multiplicative and Subtractive Convolution)를 제안합니다. 최근의 주목 메커니즘 기반 접근법들이 주로 salient feature에 집중했던 반면, 저자들은 facial images의 복잡한 샘플 처리 시 필요한 시각적 세부정보를 간과할 수 있다는 점을 지적하며, differential feature의 중요성을 강조합니다. MSConv는 multi-scale mixed convolution을 통해 두 종류의 특징을 효과적으로 캡처하고, 모델의 전반적인 인식 성능을 향상시키는 데 기여합니다.

- **Technical Details**: MSConv 모듈은 특정 조건에서 학습의 효과를 극대화하기 위해 Multiplication Operation (MO)과 Subtraction Operation (SO)을 활용합니다. MO는 두 개의 서로 다른 feature를 다중 스케일 합성곱 방법으로 얻어내고 곱하여 수행되며, SO는 feature map 간의 미세한 차이를 포착하는 데 사용됩니다. 이러한 접근법을 통해 salient feature와 differential feature를 동시에 효과적으로 사용할 수 있도록 합니다. 저자들은 SKNet 모델에서의 softmax를 sigmoid로 변환하는 과정을 통해 MSConv의 구조적 변화를 이루어내었습니다.

- **Performance Highlights**: 실험 결과, MSConv는 오직 salient feature만을 중점적으로 다룬 모델들보다 뛰어난 성능을 보였습니다. 두 종류의 feature를 조합함으로써 얼굴 인식 작업에서의 인식 정확도를 높였고, noise나 occlusion에 대해 보다 효과적으로 대처할 수 있음을 보였습니다. 이로써 MSConv는 얼굴 인식 분야에서 더욱 신뢰할 수 있는 성능을 제공하며, 복잡한 시나리오에서도 일관된 결과를 나타냅니다.



### Sample-aware Adaptive Structured Pruning for Large Language Models (https://arxiv.org/abs/2503.06184)
- **What's New**: 이 논문에서는 LLMs(대형 언어 모델)의 구조적 가지치기(strcutured pruning) 기법에 대한 새로운 접근법인 AdaPruner를 소개합니다. AdaPruner는 기존의 무작위 선택 방식 대신 샘플 인식(sample-aware) 기반으로 최적의 보정 데이터(calibration data)와 중요도 추정 지표(importance estimation metrics)를 동시에 최적화하는 점이 핵심입니다. 이를 통하여 LLMs의 비효율적인 파라미터를 효과적으로 제거함으로써 성능을 극대화하는 방법을 제안합니다.

- **Technical Details**: AdaPruner는 베이지안 최적화(Bayesian optimization)를 활용하여, 보정 데이터와 중요도 추정 지표를 동시적으로 탐색하는 구조로 되어 있습니다. 첫째, AdaPruner는 구조적인 가지치기와 관련된 데이터와 메트릭의 하위 공간을 생성하여 상호 의존성을 고려합니다. 둘째, Taylor 확장(Taylor expansion)을 기반으로 파라미터 제거가 손실에 미치는 영향을 정량화하여, 효과적인 구조 제거를 가능하게 합니다.

- **Performance Highlights**: 실험 결과에 따르면, AdaPruner는 다양한 가지치기 비율에서 기존의 구조적 가지치기 방법들보다 우수한 성능을 보여주었습니다. 특히, 20% 가지치기에서 AdaPruner로 가지치기한 모델은 비가지치기 모델의 97%의 성능을 유지했습니다. 이는 LLaMA 시리즈 모델에서 기존의 LLM-Pruner보다 평균 1.37% 높은 성능을 달성함으로써 효과성과 강건성을 입증했습니다.



### Lightweight Software Kernels and Hardware Extensions for Efficient Sparse Deep Neural Networks on Microcontrollers (https://arxiv.org/abs/2503.06183)
Comments:
          Accepted at MLSys 2025

- **What's New**: 이번 연구에서는 마이크로컨트롤러(Microcontrollers, MCUs)와 같은 엣지 디바이스에서 가지치기된 딥 뉴럴 네트워크(Deep Neural Networks, DNNs)의 성능을 가속화하기 위해 세 가지 주요 기여를 제안합니다. 첫째로, N:M 가지치기 레이어를 위한 최적화된 소프트웨어 커널을 설계하여 최대 2.1배 및 3.4배 빠른 성능을 달성합니다. 둘째로, 소프트웨어 커널 활성화를 가속화하기 위한 가벼운 ISA(Instruction-Set Architecture) 확장을 구현하여 최대 1.9배의 추가 속도 향상을 얻었습니다. 마지막으로, 오픈 소스 DNN 컴파일러를 확장하여 전체 네트워크에 대해 이러한 희소 커널을 활용합니다.

- **Technical Details**: 본 연구는 1:4, 1:8 및 1:16의 희소성을 목표로 하는 소프트웨어 커널을 설계하였으며, 이를 통해 RISC-V MCU에서 각각의 성능을 1.1배에서 1.85배, 그리고 1.02배에서 3.4배 향상시켰습니다. 추가적으로, 새로운 경량 ISA 명령어를 도입하여 활성화 선택과 관련된 계산을 가속화하고 SIMD(Single Instruction Multiple Data) 명령의 효율성을 향상시켰습니다. 이러한extension은 5%의 면적 오버헤드를 초래했지만, 속도를 최대 1.9배까지 증가시킵니다.

- **Performance Highlights**: 연구결과, 1:4 희소성에서 CNN(Convolutional Neural Network)과 ViT(Vision Transformer)에 대해 각각 1.31배와 1.43배의 지연 감소를 달성하였습니다. 1:16 희소성에서는 정확도 손실이 1.5% 미만임에도 불구하고 CNN에 대해 3.21배, ViT에 대해 1.81배의 속도 향상을 보여주었습니다. 이로 인해, 엣지 디바이스에서 DNN의 실행 효율성을 대폭 개선할 수 있는 가능성을 제시합니다.



### Minion Gated Recurrent Unit for Continual Learning (https://arxiv.org/abs/2503.06175)
- **What's New**: 최근 연구에서는 제한된 자원에서 지속적인 학습이 필수적인 것으로 평가되고 있습니다. 이에 따라 기존의 Recurrent Neural Networks (RNN) 구조를 단순화하여 효율성을 높이려는 시도가 계속되고 있습니다. 새로운 구조인 'Minion Recurrent Unit (MiRU)'는 전통적인 GRU의 복잡한 게이트 구조를 Scaling Coefficients로 대체하여 자원 소모를 최소화하면서 성능을 유지하고 있습니다.

- **Technical Details**: MiRU는 GRU의 Reset 및 Forget 게이트를 제거하고 대신 Scaling Coefficients를 도입하여 정보를 조절합니다. 이를 통해 계산 비용 및 메모리 요구사항을 상당히 줄일 수 있습니다. MiRU는 MNIST와 IMDB 데이터셋을 이용한 평가를 통해 전통적인 GRU와 유사한 성능을 유지하며 약 2.88배 적은 파라미터를 사용함을 입증했습니다.

- **Performance Highlights**: MiRU는 시간적 연속성이 중요한 멀티태스킹 학습 환경에서도 안정적인 성능을 보여 주목받고 있습니다. 특히, 기존의 GRU와 그 변종들이 불안정한 학습을 보일 때에도 MiRU는 안정적으로 성능을 유지합니다. 이러한 특성 덕분에 MiRU는 Edge Device에서 활용 가능한 유망한 후보로 평가됩니다.



### ROCM: RLHF on consistency models (https://arxiv.org/abs/2503.06171)
- **What's New**: 이번 연구에서는 일관성 모델(consistency models)에 대한 인간 피드백 리인포스먼트 러닝(RLHF)을 적용하기 위한 직접 보상 최적화 프레임워크를 제안합니다. 이러한 프레임워크는 분포 정규화(distributional regularization)를 통합하여 훈련 안정성을 향상시키고 보상 해킹(reward hacking)을 방지합니다. 일관성 모델을 통해 RLHF를 활용하는 것은 효율적인 샘플 생성을 가능하게 하며, 특히 적은 수의 단계에서 경쟁력 있는 결과를 생성할 수 있습니다.

- **Technical Details**: 연구에서는 다양한 $f$-divergences를 정규화 전략으로 탐색하여 보상 극대화와 모델 일관성 사이의 균형을 고민했습니다. 기존의 정책 경량화(policy gradient) 방법들과는 달리, 본 접근법은 1차 기울기를 활용함으로써 효율적이며 하이퍼파라미터 조정에 덜 민감합니다. 직접 보상 목표를 최적화하는 방법을 통해 훈련의 안정성을 높이고 효율성을 크게 향상시킴을 실험적으로 입증했습니다.

- **Performance Highlights**: 경쟁력 있는 결과는 정책 경량화 방법들과 비교할 때 훨씬 적은 하이퍼파라미터 조정과 빠른 훈련을 요구하면서도 동등하거나 우수한 성능을 달성함을 보여 줍니다. 또한, 다양한 정규화 기술이 모델의 일반화 및 과적합 방지에 미치는 영향을 분석하여 매우 유망한 결과를 도출하였습니다.



### Treble Counterfactual VLMs: A Causal Approach to Hallucination (https://arxiv.org/abs/2503.06169)
- **What's New**: 이번 연구에서는 Vision-Language Models (VLMs)의 환각(hallucination) 문제를 해결하기 위해 인과론적(causal) 관점을 도입했습니다. 기존의 연구들은 통계적 편향(statistical biases)이나 언어적 선입견(language priors)과 같은 요인들을 중심으로 환각을 분석했지만, 본 연구에서는 그 원인을 인과적 그래프(causal graph)와 반사실적 분석(counterfactual analysis)을 통해 구조적으로 이해하려고 하였습니다. 이를 통해 VLM의 출력에서 비의도적인 직적 영향(unintended direct influence)을 체계적으로 제거할 수 있는 방법론을 제시합니다.

- **Technical Details**: 연구에 따르면, VLM에서 환각 현상은 비전 모달리티(vision modality)와 텍스트 모달리티(text modality) 간의 잘못된 통합으로 발생한다고 가정합니다. 이를 해결하기 위해 구조적 인과 그래프(structural causal graphs)를 설계하고, 각 모달리티의 자연 직적 효과(Natural Direct Effect, NDE)를 추정하여 모델의 각 모달리티 의존도를 조절하는 동적 개입 모듈(test-time intervention module)을 개발하였습니다. 이러한 접근법은 다양한 변형된 이미지와 언어 임베딩을 활용하여 각 모달리티의 특성을 보다 명확히 평가할 수 있게 합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법이 기존의 환각 완화 연산보다 우수한 성능을 보이며 VLM의 신뢰성을 크게 향상시켰습니다. 특히, 단순히 기존 방법을 초월하여 두 개의 다양한 벤치마크에서 두 개의 VLM에서 일관되게 뛰어난 성과를 나타냈습니다. 이 연구는 VLM의 신뢰성을 높이는 데 기여할 수 있는 강력하고 해석 가능한 프레임워크를 제공하며, 관련 코드 또한 공개되어 접근성과 재현성을 강조하고 있습니다.



### Secure On-Device Video OOD Detection Without Backpropagation (https://arxiv.org/abs/2503.06166)
- **What's New**: 이 논문은 SecDOOD라는 새로운 클라우드-디바이스 협력 프레임워크를 소개합니다. 이 프레임워크는 로컬 백 프로파게이션이 필요 없는 효율적인 OOD(Out-of-Distribution) 감지를 가능하게 하여 자원 제한이 있는 엣지 디바이스에서 머신 러닝 모델의 신뢰성을 높입니다. 특히, HyperNetwork 기반의 개인화된 매개변수 생성 모듈을 통해 클라우드에서 교육된 모델을 엣지 디바이스의 특정 데이터 분포에 맞게 동적으로 조정합니다.

- **Technical Details**: SecDOOD는 클라우드의 리소스를 활용하여 모델 학습을 가능하게 하면서도 사용자 데이터의 개인 정보를 온 디바이스에서 보호합니다. 초기 데이터 처리 과정에서 가장 informative한 feature channel만 선택적으로 암호화하는 동적 기능 샘플링 및 암호화 전략을 도입하여 암호화 오버헤드를 최소화합니다. 이러한 방식은 로컬 튜닝 없이도 중앙 및 로컬 정보를 효과적으로 결합할 수 있게 해 줍니다.

- **Performance Highlights**: 다양한 데이터세트 및 OOD 시나리오에 대한 폭넓은 실험을 통해 SecDOOD는 완전히 파인튠된 모델과 유사한 성능을 달성함을 보여줍니다. 연구 결과는 자원 제한이 있는 엣지 디바이스에서 보안과 효율성을 갖춘 개인화된 OOD 감지를 가능하게 하는 데 기여합니다. 이러한 접근 방식은 많은 엣지 디바이스에서 활용 가능한 기술로 자리 잡을 것을 기대합니다.



### Feature-EndoGaussian: Feature Distilled Gaussian Splatting in Surgical Deformable Scene Reconstruction (https://arxiv.org/abs/2503.06161)
Comments:
          14 pages, 5 figures

- **What's New**: 본 논문은 3D Gaussian Splatting (3DGS)을 기반으로 한 Feature-EndoGaussian (FEG)이라는 새로운 접근법을 제안합니다. 기존의 Neural Radiance Fields (NeRFs)보다 데이터 요구량이 적고 렌더링 속도가 빠른 FEG는 실시간 의미적 및 장면 재구성을 가능하게 합니다. 이 기술은 2D 세그멘테이션 정보를 3D 렌더링에 통합하여, 수술 중 내비게이션의 정확성을 획기적으로 향상시킵니다.

- **Technical Details**: FEG는 Gaussian 변형 프레임워크 내에서 세그멘테이션 기능을 증류하여 재구성의 정확성과 세그멘테이션의 신뢰성을 모두 강화합니다. 제안된 레스터라이저(rasterizer)는 색상, 깊이 및 의미적 특징 맵을 동시에 렌더링할 수 있도록 여러 가지 개선사항을 포함하고 있습니다. 이는 Gaussian 원시 체적을 이미지 공간으로 렌더링하며, 각 원시의 매개변수에 대한 그래디언트 전달을 가능하게 합니다.

- **Performance Highlights**: EndoNeRF 데이터셋에서 FEG는 0.97의 SSIM, 39.08의 PSNR, 0.03의 LPIPS를 달성하여 기존의 주요 방법들보다 우수한 성능을 보였습니다. 또한 EndoVis18 데이터셋에서도 경쟁력 있는 클래스별 세그멘테이션 메트릭스를 보여주며, 모델 크기와 실시간 성능을 균형 있게 유지합니다.



### UrbanVideo-Bench: Benchmarking Vision-Language Models on Embodied Intelligence with Video Data in Urban Spaces (https://arxiv.org/abs/2503.06157)
Comments:
          22 pages

- **What's New**: 이 연구는 도시 내에서의 움직임 중 지각 능력에 대한 멀티모달 모델(Video-LLMs)의 능력을 평가하기 위한 새로운 벤치마크인 UrbanVideo-Bench를 소개합니다. 이 벤치마크는 1.5천 개의 동영상 클립과 5.2천 개의 다지선다형 질문을 포함하여, 도시에서의 체화된 인지 능력에 대한 통찰력을 제공합니다. 이를 통해 현재의 멀티모달 모델이 도시 환경에서 인지 적응력 부족한 부분을 수치적으로 분석하였습니다.

- **Technical Details**: 연구는 4가지 능력(Recall, Perception, Reasoning, Navigation)로 나뉜 16개의 특정 과제를 정리합니다. 드론을 이용해 실제 도시와 시뮬레이터에서 동영상 데이터를 수집하였으며, 이 과정에서 발생할 수 있는 신호 손실과 같은 여러 도전과제를 해결하였습니다. 채집된 데이터는 모델의 인지 능력을 평가하기 위한 다지선다형 질문 생성 파이프라인과 결합되었습니다.

- **Performance Highlights**: 17개의 일반적으로 사용되는 Video-LLMs를 평가한 결과, 이들은 도시 환경에서의 체화된 인지 능력에 현저한 한계를 보였습니다. 인과적 추론(causal reasoning)은 기억력, 지각력, 내비게이션과 강한 상관관계를 가지는 반면, 반사실적 및 연결적 추론(capabilities for counterfactual and associative reasoning)은 다른 작업과 낮은 상관관계를 보였습니다. 이는 도시 환경에서의 체화된 인지 학습을 위한 기준틀을 마련한 중요한 발견입니다.



### Exploring the usage of Probabilistic Neural Networks for Ionospheric electron density estimation (https://arxiv.org/abs/2503.06144)
Comments:
          13 pages, 7 figures

- **What's New**: 이 논문은 기존 신경망(Neural Networks, NN)의 한계점인 출력의 불확실성을 정량화하는 방법을 탐구합니다. 특히 정밀 지점 위치 결정 시스템(Precise Point Positioning, PPP)과 같은 중요 응용 분야에서 예측의 신뢰성을 이해하는 것이 매우 중요합니다. 이 연구는 이온층의 수직 전체 전자밀도(Vertical Total Electron Content, VTEC) 추정 및 관련 불확실성 측정을 제공하는 잠재적인 프레임워크를 제안하고 있습니다.

- **Technical Details**: 논문에서는 확률적 신경망(Probabilistic Neural Networks, PNN)을 활용하여 VTEC 추정의 불확실성을 정량화할 수 있는 방법을 제시합니다. PNN은 네트워크 파라미터(가중치 및 바이어스)의 사전 및 사후 확률 분포를 정의하여 효과적으로 설계해야 합니다. 모델의 예측 단계에서 매번 다양한 출력을 생성하는 확률적 요소가 도입되어, 불확실성을 추정할 수 있는 가능성을 제공합니다.

- **Performance Highlights**: 연구 결과, PNN 모델이 VTEC 추정에 제공하는 불확실성이 체계적으로 과소 추정될 수 있다는 것이 확인되었습니다. 특히 저위도 지역에서는 실제 오차가 모델의 추정값보다 두 배까지 클 수 있으며, 이러한 과소 추정은 태양 최대 기간 동안 더욱 두드러진다고 합니다. 이 논문은 안전 관련 애플리케이션에서 강력한 보호 수준을 설정하기 위한 필수적인 데이터로 이 정보를 활용할 수 있음을 강조하고 있습니다.



### GSV3D: Gaussian Splatting-based Geometric Distillation with Stable Video Diffusion for Single-Image 3D Object Generation (https://arxiv.org/abs/2503.06136)
- **What's New**: 본 논문에서는 이미지 기반 3D 생성의 최신 연구 결과를 소개하며, 특히 로봇 공학, 게임 및 가상 현실에서의 응용 가능성을 강조합니다. 기존의 3D 확산 모델(3D diffusion models)에서는 데이터셋 부족이나 강력한 사전 훈련 모델의 결여로 인한 한계가 있었으며, 2D 확산 기반 접근법은 기하학적 일관성을 유지하는 데 어려움을 겪었습니다. 제안하는 방법은 2D 확산 모델의 암묵적인 3D 추론 능력을 활용하고, Gaussian-splatting 기반의 기하학적 증류를 통해 3D 일관성을 보장합니다.

- **Technical Details**: 제안된 Gaussian Splatting Decoder는 SV3D의 잠재 출력(latent outputs)을 명시적인 3D 표현으로 변환함으로써 3D 일관성을 강제합니다. 이 과정에서 다중 관점(latents)을 구조적으로 정렬된 3D 표현으로 변환하며, 이러한 기하학적 제약은 뷰 간의 불일치를 수정하고 견고한 기하학적 일관성을 제공합니다. 이와 같은 방법은 3D 확산 모델에 비해 높은 충실도(fidelity)와 다양한 질감을 생성할 수 있는 강력한 솔루션으로 자리 잡고 있습니다.

- **Performance Highlights**: 실험 결과는 다중 관점 일관성(multi-view consistency) 및 다양한 데이터셋에서의 강력한 일반화 능력을 입증합니다. 본 연구는 2D 입력으로부터 일관된 3D 객체를 생성하는 데 있어 뛰어난 성능을 보이며, 코드도 수락 시 공개될 예정입니다. 더욱이, 제안하는 프레임워크는 단일 이미지 기반의 3D 생성을 위한 확장 가능한 솔루션을 제공하며, 2D 확산 모델의 다양성과 3D 구조의 일관성 간의 격차를 메우는 데 기여하고 있음을 보여줍니다.



### Multi-modal expressive personality recognition in data non-ideal audiovisual based on multi-scale feature enhancement and modal augmen (https://arxiv.org/abs/2503.06108)
- **What's New**: 이 논문은 비주얼(visual)과 오디오(auditory) 모달 데이터 인식을 위한 엔드 투 엔드(end-to-end) 멀티모달 성격 인식 네트워크를 제안하고, 다양한 모달 데이터의 특징을 효과적으로 융합하는 크로스 어텐션(cross-attention) 메커니즘을 활용합니다. 특히, 멀티스케일 특징 강화 모듈(multi-scale feature enhancement module)을 도입하여 효과적인 정보 표현을 강화하고 잉여 정보의 간섭을 억제하는 방법을 제시합니다. 이 방법은 기존에는 접근하기 어려웠던 비정상적인 데이터 상황에서도 성능이 우수하다는 것을 보여줍니다.

- **Technical Details**: 이 연구에서는 멀티모달 성격 인식의 정확도와 강건성(robustness)을 높이는 데 중점을 두고, 음성, 얼굴 표현 등 다양한 데이터를 분석하여 개인의 잠재적 성격 특성을 자동으로 추론하는 기술을 다룹니다. 제안된 멀티스케일 특징 강화 모듈은 비주얼과 오디오 양쪽 모달의 특징을 효과적으로 보강하고, 불필요한 특징의 영향을 최소화하여 성격 인식의 신뢰도를 높입니다. 또한 모달 강화 훈련 전략(modality enhancement training strategy)을 통해 비정상적인 환경에서의 모델 적응력을 향상시킵니다.

- **Performance Highlights**: 실험 결과, 이 논문에서 제안한 방법은 ChaLearn First Impression 데이터셋에서 평균 0.916의 성격 정확도를 달성하여 기존의 다른 멀티모달 기반 방법을 초월했습니다. 제안된 모듈과 훈련 전략의 기여도도 삭감 실험을 통해 검증되었으며, 이러한 방법들이 모델의 성능 개선에 크게 기여함을 확인했습니다. 마지막으로 다양한 비정상 시나리오에서 모델의 강건성을 검증함으로써 제안된 모달 강화 전략이 효과적임을 입증하였습니다.



### Feature Fusion Attention Network with CycleGAN for Image Dehazing, De-Snowing and De-Raining (https://arxiv.org/abs/2503.06107)
- **What's New**: 본 논문은 Feature Fusion Attention (FFA) 네트워크와 CycleGAN 아키텍처를 결합하여 이미지의 안개 제거를 위한 새로운 접근 방식을 제시합니다. 이 방법은 감독 학습(supervised learning)과 비감독 학습(unsupervised learning) 기술을 활용하여 안개를 효과적으로 제거하면서도 이미지의 핵심 세부사항을 보존할 수 있습니다. 제안된 하이브리드 아키텍처는 기존의 방법에 비해 PSNR 및 SSIM 점수에서 뛰어난 성능 향상을 보여줍니다.

- **Technical Details**: 논문에서는 CVPR 2019 데이터셋인 RESIDE와 DenseHaze를 사용하여 다양한 가상 및 실제 상황에서도 안개가 낀 이미지를 효과적으로 처리할 수 있음을 실험적으로 보여주었습니다. CycleGAN은 쌍을 이루지 않은 하늘 이미지와 깨끗한 이미지를 효과적으로 처리하며, 이 아키텍처 조합은 강력한 데이터 적응능력을 제공합니다. 모델은 FFA 네트워크와 CycleGAN의 통합을 통해 저해상도 상태에서의 이미지를 복원하는 데 필요한 다양한 기능을 학습합니다.

- **Performance Highlights**: 제안된 방법은 25, 20, 10, 5, 0개의 깨끗한 샘플에 대한 여러 모델 변형을 구현하고 평가하여 성능을 측정하였습니다. SSIM과 PSNR과 같은 지표를 통해 실시간으로 평가할 수 있는 웹 인터페이스를 개발했습니다. 또한, 기존 방법과 비교하여 통계적으로 유의미한 성능 향상을 확인하였습니다.



### ULTHO: Ultra-Lightweight yet Efficient Hyperparameter Optimization in Deep Reinforcement Learning (https://arxiv.org/abs/2503.06101)
Comments:
          23 pages, 22 figures

- **What's New**: 이 논문에서는 깊은 강화 학습(Deep Reinforcement Learning, RL)에서 하이퍼파라미터 최적화(Hyperparameter Optimization, HPO)의 효율적이고 강력한 접근 방식인 ULTHO를 제안합니다. ULTHO는 다중 팔 밴딧(mult-armed bandit) 문제로 HPO 과정을 공식화하여 장기적인 반환 최적화와 연결했습니다. 기존의 방법들이 샘플 비효율성과 높은 계산 비용으로 어려움을 겪는 반면, ULTHO는 단일 실행 내에서 빠른 HPO를 가능하게 하며, 간단한 아키텍처로 뛰어난 성능을 제공합니다.

- **Technical Details**: ULTHO는 클러스터링된 팔을 가지는 다중 팔 밴딧(multi-armed bandit with clustered arms, MABC)이라는 구조를 사용하여 여러 작업과 학습 단계에서 하이퍼파라미터를 적응적으로 최적화합니다. 이 프레임워크는 HP 선정 과정에서 예상되는 작업 반환을 기반으로 작동하여 장기 수익을 극대화하는 한편 탐색을 균형있게 조정합니다. 기존 방법들과는 달리, ULTHO는 복잡한 학습 과정이나 추가적인 학습 프로세스 없이도 다양한 RL 알고리즘과 호환됩니다.

- **Performance Highlights**: NLTHO는 ALE, Procgen, MiniGrid, PyBullet과 같은 다양한 벤치마크에서 테스트 되었습니다. 실험 결과, ULTHO는 뛰어난 성능과 더불어 뛰어난 계산 효율성을 달성했습니다. 이러한 성과는 ULTHO가 고급 및 자동화된 RL 시스템 개발에 기여할 것임을 시사합니다.



### ZO-DARTS++: An Efficient and Size-Variable Zeroth-Order Neural Architecture Search Algorithm (https://arxiv.org/abs/2503.06092)
Comments:
          14 pages, 8 figures

- **What's New**: ZO-DARTS++는 Differentiable Neural Architecture Search (NAS) 방법의 새롭고 효율적인 접근을 제공합니다. 이 방법은 자원 제한 아래에서 성능과 자원 제약을 균형 있게 조정하기 위한 제로 차수 근사(zeroth-order approximation)를 통합하여 효과적인 기울기 처리를 가능하게 합니다. 또한, 해석 가능한 아키텍처 분포를 위해 온도 변화를 적용한 sparsemax 함수를 사용하였으며, 크기 가변 탐색 방식(size-variable search scheme)을 도입하여 컴팩트하면서도 정확한 아키텍처를 생성합니다.

- **Technical Details**: ZO-DARTS++는 기존의 DARTS 및 다른 차별화된 NAS 방법의 저효율성과 구조 선택 과정의 개선, 자원 소비 문제를 해결하기 위해 고안되었습니다. 제로 차수 근사 기법을 도입하여 고립적인 변수의 기울기를 쉽게 대체하여 탐색 과정의 정확성과 효율성을 높입니다. 또한, sparsemax 함수를 사용해 이해 가능성을 증가시키고, 자원 소비를 효과적으로 줄일 수 있는 크기 가변 탐색 체계를 제안했습니다.

- **Performance Highlights**: ZO-DARTS++는 의료 이미징 데이터셋을 기반으로 한 광범위한 테스트에서 평균 정확성을 표준 DARTS 기반 방법보다 최대 1.8% 향상시키고, 탐색 시간을 약 38.6% 단축시켰습니다. 자원 제한 변형을 통해 35% 이상의 파라미터 수를 줄이면서 경쟁력 있는 정확도 수준을 유지할 수 있음을 보여주었습니다. 이러한 점에서 ZO-DARTS++는 실제 의료 애플리케이션에 적합한 고품질의 자원 인식 DL 모델을 생성하는 데 있어 다재다능하고 효율적인 프레임워크를 제공합니다.



### T-CBF: Traversability-based Control Barrier Function to Navigate Vertically Challenging Terrain (https://arxiv.org/abs/2503.06083)
- **What's New**: 안전성(safety)은 최근 몇 년간 모션 플래닝(motion planning)과 제어 기술(control techniques)에서 가장 중요한 주제로 자리 잡고 있습니다. 본 연구에서는 새로운 Traversability-based Control Barrier Function (T-CBF)를 소개하여 비구조적이고 수직적으로 도전적인 지형에서 로봇이 충돌 회피를 넘어 더 나아간 안전성을 유지할 수 있도록 합니다. T-CBF는 비구조적 환경에서 안전성과 이동성을 극대화하기 위해 신경망 기반의 제어 장벽 함수(neural Control Barrier Functions)를 활용합니다.

- **Technical Details**: T-CBF는 차량의 전복(overturn)과 고착(immobilization)과 같은 새로운 안전성을 고려하여 설계되었습니다. 이 함수는 안전과 위험 관찰에 대한 데이터에 기반하여 훈련되어 안전한 경로(trajactory)를 생성합니다. 또한, 본 연구는 Verti-4 Wheeler (V4W) 플랫폼을 통해 T-CBF의 성능을 실험적으로 증명하였습니다.

- **Performance Highlights**: T-CBF 플래너는 이전에 개발된 플래너보다 30% 더 높은 안전성과 이동성을 제공하며, 실제 수직적으로 도전적인 지형에서 로봇이 목표를 향해 나아갈 수 있도록 합니다. 이러한 결과는 T-CBF가 충돌 회피를 넘어서는 안전성을 발휘하며 복잡한 환경에서도 효과적으로 작동할 수 있음을 보여줍니다.



### Towards Conversational AI for Disease Managemen (https://arxiv.org/abs/2503.06074)
Comments:
          62 pages, 7 figures in main text, 36 figures in appendix

- **What's New**: 이번 연구에서는 Articulate Medical Intelligence Explorer (AMIE)를 활용한 새로운 LLM 기반 시스템을 소개하며, 이 시스템은 기존의 진단 능력을 넘어 질병 관리(Clinical Management)와 대화(Dialogue)를 최적화하는데 중점을 두고 있다는 점이 눈에 띕니다. AMIE는 여러 환자 방문 사례와 치료 반응에 대한 추론을 포함하며, 전문적인 약물 처방 능력을 강화합니다. 또한, AMIE는 Gemini의 긴 맥락(Long Context) 능력을 활용해 최신 임상 지침과 약물 포뮬러리에 기반한 합리적인 추론을 제공합니다.

- **Technical Details**: AMIE는 구조적 추론(Structured Reasoning)과 인-context 검색(In-Context Retrieval)을 결합하여 의료 지식의 권위에 기반한 해결책을 도출합니다. 연구에서는 21명의 일반 진료 의사(Primary Care Physicians, PCPs)와 비교하여 AMIE의 관리 추론 능력을 평가하였고, 이는 영국 NICE 지침과 BMJ Best Practice 가이드를 반영한 100개의 다중 방문 사례로 구성되었습니다. 이 과정에서 전문 의사들이 평가한 결과, AMIE는 PCP들에게 뒤지지 않는 성능을 기록하였습니다.

- **Performance Highlights**: AMIE는 진단의 정밀성과 조사에 있어 PCP들보다 더 높은 점수를 받았으며, 임상 지침에 따른 관리 계획의 적절성 면에서도 우수한 성과를 보였습니다. 추가로, 약물 추론을 벤치마킹하기 위해 제작된 RxQA를 통해 AMIE는 어려운 질문에 대한 정확도에서 PCP들을 초월하는 결과를 나타냈습니다. 비록 실세계 적용을 위한 추가 연구가 필요하지만, AMIE의 강력한 성능은 질병 관리에서 대화형 AI의 중요한 진전을 의미합니다.



### GEM: Empowering MLLM for Grounded ECG Understanding with Time Series and Images (https://arxiv.org/abs/2503.06073)
- **What's New**: 최근 다중 모달 대형 언어 모델(MLLMs)은 자동화된 ECG 해석에서 발전을 이루었지만, 여전히 두 가지 주요 한계에 직면해 있습니다. 첫째, 시계열 신호와 시각적 ECG 표현 간의 결합이 불충분하며, 둘째, 진단을 미세한 파형 증거에 연결하는 데 한계가 있습니다. 이를 해결하기 위해 GEM을 도입하여 시계열 데이터, 12리드 ECG 이미지 및 텍스트를 통합한 최초의 MLLM을 제안합니다.

- **Technical Details**: GEM은 이중 인코더 프레임워크를 사용하여 시계열과 이미지 특성을 보완적으로 추출하고, 크로스모달 정렬(Cross-modal Alignment)을 통해 효과적인 다중 모달 이해를 가능하게 합니다. 또한, 지식 기반의 지침 생성을 통해 고해상도의 그라운딩 데이터를 생성하여 진단을 측정 가능한 매개변수와 연계합니다. 이러한 구조는 GEM이 임상과 유사한 진단 과정을 시뮬레이션하도록 돕습니다.

- **Performance Highlights**: GEM은 기존 및 제안된 벤치마크에서 실험적으로 예측 성능을 7.4% 향상시키고, 설명 가능성은 22.7% 증가시키며, 그라운딩 성능을 24.8% 개선하였습니다. 이러한 성과는 GEM이 실제 임상 환경에서 더 적합하고 믿을 수 있는 진단 도구로 자리 잡을 수 있도록 합니다.



### A Survey on Post-training of Large Language Models (https://arxiv.org/abs/2503.06072)
Comments:
          87 pages, 21 figures, 9 tables

- **What's New**: 이 논문은 Post-training Language Models (PoLMs)에 대한 최초의 포괄적 설문조사를 제공하며, 모델의 발전과 문제점들을 다섯 가지 핵심 패러다임으로 체계적으로 분석합니다. 이러한 패러다임에는 파인튜닝(Fine-tuning), 정렬(Alignment), 추론(Reasoning), 효율성(Efficiency), 통합 및 적응(Integration and Adaptation)이 포함됩니다. PoLM의 진화를 통해 데이터셋 활용 방식 및 도메인 적응력 향상에 대한 기여를 설명합니다.

- **Technical Details**: 이 연구에서는 조건에 맞는 대화 시스템에서부터 과학적 탐색에 이르기까지 다양한 분야에 걸쳐 LLM의 발전을 다루고 있습니다. PoLM은 GPT-4와 DeepSeek-R1를 포함하여 고도화된 모델 성능을 획득하는 데 중점을 둡니다. 이 과정에서 모델의 특정 작업에 대한 조정 및 사용자 요구 사항에 대한 적절한 대응 방안을 설정합니다.

- **Performance Highlights**: PoLM의 발전은 특정 작업과 요건에 대한 적응 능력을 향상시키는데 크게 기여했습니다. 예를 들어, DeepSeek-R1은 추론 능력을 강화하고 사용자 선호에 맞춘 정렬 및 도메인 적응성을 개선했습니다. 앞으로의 연구 방향은 모델의 정밀도 및 신뢰성을 향상시키는 것으로, LRM의 성장과 함께 더욱 민감한 기술 개발이 이루어질 것입니다.



### A Novel Trustworthy Video Summarization Algorithm Through a Mixture of LoRA Experts (https://arxiv.org/abs/2503.06064)
- **What's New**: 이번 연구에서는 사용자 생성 콘텐츠가 급증하는 비디오 공유 플랫폼에서 비디오의 효율적인 검색과 탐색을 위한 새로운 접근 방식이 제안됩니다. MiLoRA-ViSum은 비디오 데이터의 복잡한 시간적 동역학과 공간적 관계를 효율적으로 포착하기 위해 설계된 새로운 모델입니다. 기존의 Video-llama에서 발생하는 자원 소모 문제를 해결하기 위해 저자들은 파라미터 수를 조절할 수 있는 방법을 모색했습니다.

- **Technical Details**: MiLoRA-ViSum은 전통적인 Low-Rank Adaptation (LoRA)을 혼합 전문가(mixture-of-experts) 패러다임으로 확장하여 시간적 및 공간적 적응 메커니즘을 통합합니다. 이 모델은 각각의 LoRA 전문가가 특정 시간적 또는 공간적 차원에 맞춰 조정되어 비디오 요약 작업을 수행합니다. 이 동적 통합 방식은 복잡한 비디오 데이터의 다양한 특성을 효과적으로 다루기 위해 특별히 설계되었습니다.

- **Performance Highlights**: 비교 평가에서 MiLoRA-ViSum은 VideoXum와 ActivityNet 데이터셋에서 최신 모델들과 비교해 최고의 요약 성능을 발휘했습니다. 또한, 다른 모델들에 비해 현저하게 낮은 계산 비용을 유지하면서도 효과성을 보장합니다. 혼합 전문가 전략과 이중 적응 메커니즘은 비디오 요약 기능을 향상시킬 수 있는 잠재력을 강조합니다.



### STAR: A Foundation Model-driven Framework for Robust Task Planning and Failure Recovery in Robotic Systems (https://arxiv.org/abs/2503.06060)
- **What's New**: 최근 로봇 시스템은 산업 자동화에서 가정용 보조, 우주 탐사와 같은 다양한 동적 환경에서 자율적으로 작동하도록 요구받고 있습니다. 그러나 기존의 로봇 설계는 경직된 규칙 기반 프로그래밍에 의존해 이러한 예측 불가능한 작업을 처리하는 데 한계를 보이고 있습니다. 이를 개선하기 위해, 우리는 SMART (Smart Task Adaptation and Recovery)라는 새로운 프레임워크를 제안하며, 이는 Foundation Models (FMs)과 동적으로 확장되는 Knowledge Graphs (KGs)를 통합하여 효율적인 작업 계획과 자율적인 실패 회복을 가능하게 합니다.

- **Technical Details**: STAR는 자연어 명령어를 실행 가능한 계획으로 변환하고, 실행 중에 실시간 센서 데이터를 기반으로 장애의 원인을 진단합니다. 이 과정에서, KG는 역사적인 실패 패턴과 환경 제약을 포함한 지식을 저장하며, 이를 통해 상황에 최적화된 회복 전략을 생성합니다. STAR는 기존의 고정된 복구 프로토콜에 의존하지 않고, 새로운 실패 상황에 동적으로 적응하여 작업의 효율성과 신뢰성을 개선합니다.

- **Performance Highlights**: STAR는 포괄적인 데이터 세트를 활용한 실험을 통해 86%의 작업 계획 정확도와 78%의 회복 성공률을 달성했습니다. 이는 기존 방법들에 비해 상당한 개선을 보여주며, STAR는 구조화된 지식 표현을 유지하면서 경험으로부터 지속적으로 학습할 수 있는 능력이 특히 중요한 장기 배치에 적합하다는 것을 강조합니다.



### Fine-Grained Bias Detection in LLM: Enhancing detection mechanisms for nuanced biases (https://arxiv.org/abs/2503.06054)
Comments:
          Bias detection, Large Language Models, nuanced biases, fine-grained mechanisms, model transparency, ethical AI

- **What's New**: 최근 인공지능의 발전, 특히 Large Language Models (LLMs)에서의 향상은 자연어 처리를 혁신적으로 변화시켰습니다. 그러나 이러한 모델에 내재된 편향을 감지하는 것은 여전히 도전 과제가 되고 있습니다. 본 연구는 LLM에서 섬세한 편향을 식별하기 위한 새로운 탐지 프레임워크를 제시하여 윤리적 우려를 해소하고자 합니다.

- **Technical Details**: 이 접근 방식은 문맥 분석(contextual analysis), 주의 메커니즘(attention mechanisms)을 통한 해석 가능성, 그리고 반사실 데이터 증강(counterfactual data augmentation)을 통합하여 언어적 맥락에 숨겨진 편향들을 캡쳐합니다. 비교적(proven) 프롬프트와 합성 데이터셋(synthetic datasets)을 사용해 모델의 행동을 문화적, 이념적, 인구통계적 시나리오에 따라 분석합니다. 이는 벤치마크 데이터셋을 통한 정량적 분석과 전문가 리뷰를 통한 정성적 평가로 효과성을 검증합니다.

- **Performance Highlights**: 연구 결과는 인종, 성별, 사회정치적 맥락에 대한 모델 응답의 차이를 강조하지 못하는 기존 방법에 비해 섬세한 편향을 탐지하는 데 있어 개선된 성능을 보여줍니다. 또한, 훈련 데이터와 모델 아키텍처의 불균형에서 발생하는 편향을 식별하며, 지속적인 사용자 피드백을 통해 적응성과 정교함을 보장합니다. 이는 교육, 법률 시스템 및 의료와 같은 민감한 응용 분야에서의 책임 있는 LLM 배포를 지원하고 향후 실시간 편향 모니터링과 교차 언어 일반화에 중점을 둡니다.



### DropletVideo: A Dataset and Approach to Explore Integral Spatio-Temporal Consistent Video Generation (https://arxiv.org/abs/2503.06053)
- **What's New**: 이 논문에서는 비디오 생성에서 공간 시간 일관성(spatio-temporal consistency)의 개념을 새롭게 정의하고, 플롯 진행(plot progression)과 카메라 기법(camera techniques) 간의 상호 작용을 강조합니다. 특히, 카메라의 움직임이 비디오 내 내러티브에 미치는 장기적인 영향을 고려하여, 이전의 콘텐츠가 이후 생성에 미치는 영향을 연구하였습니다. 이를 위해 DropletVideo-10M이라는 대규모 오픈 소스 데이터셋을 구축하고, 이 데이터셋을 기반으로 DropletVideo라는 비디오 생성 모델을 개발했습니다.

- **Technical Details**: DropletVideo-10M 데이터셋은 1천만 개의 비디오로 구성되어 있으며, 각 비디오에는 물체의 동작과 카메라의 움직임에 관한 세부 캡션이 추가되어 있습니다. 이 데이터셋은 비디오 생성에서 공간 시간 일관성을 보존하는 데 최적화되어 있어, 물체의 시각적 특성과 장면의 일관성을 유지할 수 있도록 설계되었습니다. DropletVideo 모델은 다변량 샘플링(variable frame rate sampling) 전략을 통해 비디오 생성 속도와 시각적 전환의 템포를 정밀하게 조절할 수 있습니다.

- **Performance Highlights**: DropletVideo 모델은 광범위한 실험을 통해 시간적 및 공간적 차원에서 콘텐츠 일관성을 효과적으로 유지하는 것으로 확인되었습니다. 연구에 의해, DropletVideo는 기존 모델에 비해 더 복잡한 다중 플롯 내러티브를 생성할 수 있는 가능성을 보여주었으며, 이는 카메라 움직임과 부드러운 장면 전환을 통해 이루어집니다. 이로써 오픈 소스 데이터셋과 모델이 일반 대중에게 공개되어 알고리즘 혁신을 촉진하고, 폐쇄형 모델에 대한 대안을 제시할 수 있기를 기대합니다.



### Vairiational Stochastic Games (https://arxiv.org/abs/2503.06037)
- **What's New**: 본 논문은 Control as Inference (CAI) 프레임워크를 기반으로 다중 에이전트 시스템을 위한 새로운 변분 추론(framework) 체계를 제안합니다. 기존의 CAI는 단일 에이전트 강화 학습에 효과적이었지만, 분산된 환경에서 다중 에이전트 작전의 문제를 다루는 것은 부족했습니다. 본 연구에서는 에이전트 간의 독립성을 고려하여 새로운 접근 방식을 통해 이 문제를 해결합니다.

- **Technical Details**: 논문에서 제안된 프레임워크는 비정상성(non-stationarity) 및 정렬되지 않은 에이전트 목표(unaligned agent objectives)와 같은 문제를 처리합니다. 여기서 도출된 정책들은 $epsilon$-Nash 균형을 형성하며 이론적인 수렴(convergence) 보장을 제공합니다. 여러 알고리즘이 Nash 균형, mean-field Nash 균형, 그리고 correlated equilibrium을 해결하기 위해 구현되었습니다.

- **Performance Highlights**: 이론적 수렴 분석을 바탕으로 제안된 알고리즘들은 실제 분산 다중 에이전트 환경에서의 효과성을 입증할 예정입니다. 이러한 방법은 다중 에이전트 강화 학습의 미래 연구에서 중요한 기초가 될 수 있습니다. 새로운 접근법은 기존의 불확실한 환경에서도 안정적인 성능을 기대할 수 있게 하며, 협업 또는 경쟁 환경에서도 유용하게 사용될 수 있습니다.



### Towards Universal Text-driven CT Image Segmentation (https://arxiv.org/abs/2503.06030)
- **What's New**: OpenVocabCT는 3차원 CT 이미지에서 텍스트 기반 분할을 위해 사전 훈련된 비전-언어 모델로, 기존의 분할 모델들에 비해 더 넓은 적용 범위를 제공합니다. 특히, 대규모 CT 데이터셋인 CT-RATE를 활용하여 진단 보고서를 세부 장기 설명으로 분해하여 멀티-그레인 대조 학습을 수행합니다. 이를 통해, 병원 환경에서 자주 발생하는 다양한 텍스트 프롬프트에 효과적으로 대응할 수 있는 능력을 확보하였습니다.

- **Technical Details**: OpenVocabCT는 대규모 3D CT 이미지에 대해 비전-언어 모델을 사전 훈련하는 접근 방식을 사용하여, 장기 및 종양 분할을 위한 포괄적인 텍스트 기반 모델을 구축합니다. 이 모델은 CT-RATE 데이터셋에 대한 이미지-텍스트 쌍을 생성하고, 멀티-라벨 대조 손실을 활용하여 비전-언어 모델을 정렬시킵니다. 이를 통해, 모델은 다양한 프롬프트에 대해 유연하게 일반화할 수 있습니다.

- **Performance Highlights**: OpenVocabCT는 아홉 개의 공개 데이터셋에서 장기 및 종양 분할 작업을 평가하여 기존 방법들과 비교했을 때 우수한 성능을 보였습니다. 특히, 단일 목표 분할 작업에서 텍스트 기반 모델보다 더 나은 성능을 발휘하며, 자연어를 통한 모델과의 상호작용을 통해 임상의들의 사용성을 크게 향상시킵니다.



### Zero-Shot Peg Insertion: Identifying Mating Holes and Estimating SE(2) Poses with Vision-Language Models (https://arxiv.org/abs/2503.06026)
Comments:
          Under submission

- **What's New**: 이 논문에서는 Vision-Language Models (VLMs)를 활용하여 제로샷(Zero-shot) 페그 삽입을 위한 새로운 프레임워크를 제안합니다. 이 방법은 사전 지식 없이 호환 가능한 홀을 식별하고 자세를 추정할 수 있는 인식 시스템을 구축합니다. 실험 결과, 이 시스템은 90.2%의 정확도를 달성하며, 다양한 이전에 보지 못한 페그-홀 쌍에서도 높은 성능을 보였습니다.

- **Technical Details**: 제로샷 페그 삽입을 위한 이 프레임워크는 비전-언어 모델을 활용하여 페그와 홀의 이미지를 입력받고, 이들의 호환성을 판단합니다. 그런 다음, 모델은 여러 후보 홀 중 최적의 결합 홀을 선택하고, 삽입 각도를 바탕으로 상대 자세를 추정합니다. 이를 통해, 고전적인 방법들에 비해 작업 전용 훈련 없이 다양한 산업용 연결 장치와 복잡한 형상의 물체에 잘 적용됩니다.

- **Performance Highlights**: 본 연구에서는 공백(panel) 상에 있는 산업용 커넥터를 실제로 삽입하는 과정을 평가하였고, 88.3%의 성공률을 기록했습니다. 또, 점검 연구를 통해 입력 및 출력의 변형을 체계적으로 개선하여 이 방법의 유효성을 입증하였습니다. 이러한 결과는 VLM 기반 제로샷 추론이 로봇 조립에서 강인하고 일반화 가능한 성능을 보여준다는 것을 강조합니다.



### Towards Ambiguity-Free Spatial Foundation Model: Rethinking and Decoupling Depth Ambiguity (https://arxiv.org/abs/2503.06014)
Comments:
          32 pages, 31 figures, github repo: this https URL

- **What's New**: 이 논문은 깊이 추정(depth estimation)의 복잡성을 해결하기 위한 새로운 접근 방식을 소개합니다. 특히, 투명한 장면에서는 단일 깊이 추정이 3D 구조를 온전히 표현하지 못하는 한계가 있습니다. 기존의 모델들은 결정론적(predeterministic) 예측에 국한되어 실제 다층 깊이를 간과하는 경향이 있습니다.

- **Technical Details**: 우리는 MD-3k라는 벤치마크를 제시하여 전문가 모델과 기초 모델의 깊이 편향을 드러내는 다층 공간 관계(label) 및 새로운 메트릭스를 도입했습니다. 또한 Laplacian Visual Prompting (LVP)라는 교육이 필요 없는 스펙트럼 프롬프팅 기법을 통해 RGB 입력을 라플라시안 변환하여 숨겨진 깊이를 추출합니다. 이는 모델 재교육 없이도 LVP로 추정된 깊이를 기존 RGB 기반 추정치와 결합하여 다층 깊이를 유도합니다.

- **Performance Highlights**: 결과적으로 LVP는 제로샷(zero-shot) 다층 깊이 추정에서의 유효성을 검증했으며, 이는 보다 강력하고 포괄적인 형상 조건 시각 생성(geometry-conditioned visual generation) 및 3D 기반 공간 추론(3D-grounded spatial reasoning)이 가능하게 합니다. 또한, 시간에 일관된 동영상 수준의 깊이 추론(temporal depth inference)도 지원하여 다양한 응용 분야에 기여할 전망입니다.



### Intent-Aware Self-Correction for Mitigating Social Biases in Large Language Models (https://arxiv.org/abs/2503.06011)
Comments:
          18 pages. Under review

- **What's New**: 이 연구에서는 Large Language Models (LLMs)의 출력 품질을 개선하기 위해 Feedback을 기반으로 하는 Self-Correction 기법을 제시합니다. Self-Correction은 인지 심리학의 System-2 사고와 유사하게 작용하여 사회적 편견을 줄일 가능성을 지니고 있습니다. 각각의 구성 요소인 instruction, response, feedback에서 의도를 분명히 이해하고 반영하는 것이 중요하다는 것을 보여줍니다.

- **Technical Details**: Self-Correction은 세 가지 주요 단계로 이루어져 있으며, 초기 응답 생성, 피드백 생성 및 정제 단계를 포함한 iterative한 과정입니다. 응답 생성 과정에서는 편향을 줄이기 위한 명시적인 프롬프트와 Chain-of-Thought (CoT) 접근을 통해 사고 과정을 명확히 합니다. 피드백 생성 과정에서는 적합한 평가 기준을 정의하고, 각 기준에 대해 점수를 부여하여 보다 명확한 피드백을 제공합니다.

- **Performance Highlights**: 실험을 통해 Self-Correction 프레임워크가 bias를 줄이는 데 있어 기존 방법보다 보다 일관되고 강력한 효과를 나타냄을 입증하였습니다. Cross-model correction을 통해 높은 편향 모델에 대한 편향 완화 능력이 향상되었으며, Feedback의 출처와 생성자가 피드백 품질에 미치는 영향을 분석하였습니다. 또한, Refinement 품질은 Feedback 생성자에 의해 크게 좌우되는 것으로 확인되었습니다.



### Learning to Drive by Imitating Surrounding Vehicles (https://arxiv.org/abs/2503.05997)
- **What's New**: 이 연구에서는 자율주행차(AV) 훈련 시 모방 학습(imitation learning)의 데이터 증강(data augmentation) 전략을 제안합니다. 주변 차량의 궤적을 활용하여 추가적인 전문가 시연(expert demonstration)으로 활용하는 방법을 소개합니다. 이 방법은 정보가 풍부하고 다양한 주행 행동을 우선시하는 차량 선택 샘플링 기법을 도입하여 데이터셋의 풍부함을 높입니다.

- **Technical Details**: 제안된 방법은 PLUTO (Cheng et al., 2024a)와 nuPlan 데이터셋을 사용하여 평가되었으며, 자율주행 분야에서의 성능 향상을 목표로 합니다. 특히, 실제 동적 상황에서의 주행 경로 데이터를 기반으로 한 전문가적 행동 이식을 중점적으로 다룹니다. 연구 결과, 원본 데이터의 10%만으로도 기존 데이터셋보다 효과적인 성능을 보여줍니다.

- **Performance Highlights**: 제안된 데이터 증강 방법은 충돌률(collision rate)을 감소시키고 안전 지표(safety metrics)를 향상시킵니다. 원본 데이터셋의 10%만 사용해도 전체 데이터셋과 유사한 성능을 달성하는 것으로 나타났습니다. 이러한 결과는 자율주행에서 다양한 실제 경로 데이터를 활용하는 것이 중요함을 강조합니다.



### Towards Improving Reward Design in RL: A Reward Alignment Metric for RL Practitioners (https://arxiv.org/abs/2503.05996)
- **What's New**: 이번 연구에서는 강화 학습( Reinforcement Learning) 에이전트의 성능을 결정하는 보상 함수( reward function)의 설계 및 평가의 어려움을 다룹니다. 일반적으로 잘 정의된 보상이 쉽게 얻어진다고 가정되지만, 실제로는 보상 설계가 어려울 수 있습니다. 이를 해결하기 위해 연구팀은 보상 정렬( reward alignment)에 주목하여, 보상 함수가 인간 이해관계자의 선호를 얼마나 정확하게 반영하는지를 평가하는 방법을 제시합니다.

- **Technical Details**: 구체적인 보상 정렬의 척도로서, 연구에서는 Trajectory Alignment Coefficient(경로 정렬 계수)를 도입합니다. 이 지표는 인간 이해관계자가 평가한 경로 분포와 주어진 보상 함수에 의해 유도된 경로 분포 간의 유사성을 정량화합니다. Trajectory Alignment Coefficient는 기본 보상에 접근할 필요가 없고, 잠재적 기반 보상 형성( potential-based reward shaping)에 대한 불변성을 가지며 온라인 강화 학습에도 적용 가능하다는 유리한 특성을 지니고 있습니다.

- **Performance Highlights**: 11명의 강화 학습 실무자를 대상으로 한 사용자 연구에서는, 보상 선택 절차에서 Trajectory Alignment Coefficient를 접근함으로써 통계적으로 유의미한 개선이 나타났습니다. 보상 함수만 의존했을 때보다 인지적 부담이 1.5배 감소하였고, 82%의 사용자가 이를 선호하며, 성능이 우수한 정책을 생산하는 보상 함수 선택의 성공률이 41% 증가했습니다.



### GrInAdapt: Scaling Retinal Vessel Structural Map Segmentation Through Grounding, Integrating and Adapting Multi-device, Multi-site, and Multi-modal Fundus Domains (https://arxiv.org/abs/2503.05991)
- **What's New**: 이번 연구에서는 GrInAdapt라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 레이블의 세분화와 모델의 일반화를 향상시키기 위해 다중 시점 영상을 활용하여, 광학 코히어런스 단층촬영 혈관조영법(OCTA)에서의 망막 혈관 분할을 지원합니다. GrInAdapt는 등록을 통한 공통 앵커 공간 정의, 다중 시점의 예측 통합을 통한 레이블 합의 도출, 다양한 목표 영역으로의 모델 적응이라는 직관적인 세 단계 접근 방식을 따릅니다.

- **Technical Details**: GrInAdapt는 기계 학습 기반의 다중 목표 도메인 적응(Multi-Target Domain Adaptation) 기술을 활용합니다. 이 기술은 단일 출처 도메인에 기반한 모델이 여러 라벨이 있는 목표 도메인에서 정확하게 작동하도록 돕습니다. 연구에서는 OCTA 및 색상 망막 사진(CFP)과 같은 다양한 시각적 정보에서 집합적인 추론을 수행하여 강력한 레이블 세분화의 정확성을 달성했습니다.

- **Performance Highlights**: 전문적인 실험 결과, GrInAdapt는 기존 도메인 적응 방법들에 비해 평균 4%의 Dice 점수 향상과 0.42 ASSD 감소를 보여주었습니다. 이 연구는 다수의 장치와 다양한 사이트에서 발생하는 데이터 분포 변화에도 불구하고, 강력한 성능을 유지함을 입증했습니다. GrInAdapt의 결과는 자동화된 망막 혈관 분석 발전에 기여할 가능성을 강조합니다.



### Black Box Causal Inference: Effect Estimation via Meta Prediction (https://arxiv.org/abs/2503.05985)
- **What's New**: 이 논문은 인과 추론(causal inference)을 데이터셋 수준의 예측 문제(dataset-level prediction problem)로 재구성하는 방법인 블랙 박스 인과 추론(black box causal inference, BBCI)을 제안합니다. 기존의 인과 효과 추정기(estimator)를 개발하는 데 필요한 큰 노력을 줄이기 위해, 학습 과정을 통해 알고리즘 설계를 간소화하고 있습니다. 우수한 성능을 가진 이 접근 방식은 여러 인과 추론 문제에 적용 가능하며, 특히 잘 개발되지 않은 추정기 문제를 다루는 데 도움을 줍니다.

- **Technical Details**: BBCI는 주어진 데이터셋 쌍을 사용하여 인과 효과를 예측하도록 학습합니다. 이 방법론은 구조적 인과 모델(structural causal model, SCM)과 인과 쿼리(causal query)를 기반으로 하여, 알고리즘을 학습할 수 있는 메타 예측(meta-prediction) 접근 방식을 사용합니다. 각 인과 추론 문제에 따라 생성된 데이터셋-효과 쌍을 통해, 무작위 데이터 샘플을 사용하여 효과를 예측하도록 학습합니다.

- **Performance Highlights**: 여러 가지 인과 추론 문제에서 목표 추정자(target estimand)를 성공적으로 회복함을 보여줍니다. BBCI는 식별성이 확인될 경우에만 오차가 0으로 수렴하도록 하며, 데이터의 유한성에 따른 오차와 추정기 분산, 식별 부족에 의한 오차 등 다양한 오류 구성 요소를 분리하여 원인 분석을 진행합니다. 기존 추정기들과 동일한 성능을 달성함으로써, BBCI의 유용성을 강조합니다.



### SINdex: Semantic INconsistency Index for Hallucination Detection in LLMs (https://arxiv.org/abs/2503.05980)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)이 생성한 사실적으로 부정확한 출력을 탐지하기 위한 새로운 자동화된 프레임워크를 소개합니다. 제안된 방법은 문장 임베딩과 계층적 클러스터링을 기반으로 하고, 새로운 불일치 측정인 SINdex를 활용하여 LLM의 환각 현상을 보다 정확하게 감지하도록 설계되었습니다. 미셀린세서리 기술 없이도 다양한 LLM에 적용할 수 있는 블랙박스 프레임워크로서, 기존 기술에 비해 최대 9.3%의 AUROC 개선을 보여주었습니다.

- **Technical Details**: 제안된 방법론은 두 가지 주요 구성 요소로 이루어져 있습니다. 첫째, 의미가 유사한 출력들을 클러스터링하는 의미 클러스터링이며, 둘째, 이러한 클러스터 내에서 의미적 불일치를 측정하는 SINdex입니다. 이 프레임워크는 외부 데이터나 수동 개입 없이도 응답을 집계하여 생성된 정보의 정확성을 평가합니다.

- **Performance Highlights**: 제안된 방법은 TriviaQA, NQ, SQuAD 및 BioASQ와 같은 여러 유명한 오픈북 및 클로즈북 질의응답 데이터셋에서 기존의 최첨단 기법들에 비해 상당한 성능 향상을 보였습니다. 또한 대규모 설정에서의 스케일러빌리티 실험을 통해, 기존 환각 탐지 기법에 비해 60배 빠른 처리 속도를 달성했습니다.



### Learning-Order Autoregressive Models with Application to Molecular Graph Generation (https://arxiv.org/abs/2503.05979)
- **What's New**: 이번 연구에서는 Autoregressive Models (ARMs)를 확장하여 고차원 데이터를 생성하는 새로운 방법을 제안합니다. 새로운 모델인 Learning-Order Autoregressive Models (LO-ARMs)는 데이터를 통해 context-dependent한 생성 순서를 학습합니다. 이 모델은 훈련에 사용되는 확률 분포를 동적으로 결정하여, 데이터 차원의 샘플링 순서를 유연하게 조정할 수 있습니다. 이를 통해 QM9와 ZINC250k 데이터셋에서 최첨단 결과를 달성하였습니다.

- **Technical Details**: LO-ARMs는 데이터 생성에 대한 확률적 순서를 динамически 결정하기 위해 학습 가능한 확률 분포를 포함합니다. 연구진은 스토캐스틱 그래디언트 추정을 통해 모델을 훈련하기 위한 변분 하한을 도출하였습니다. 기존의 AO-ARMs와 결합하여, LO-ARMs는 데이터 차원의 조건부 분포를 효과적으로 맞출 수 있도록 설계되었습니다. 이러한 방법은 고유한 분자 생성에 유리한 환경을 조성합니다.

- **Performance Highlights**: LO-ARMs는 QM9 및 ZINC250k 데이터셋에서 Fréchet ChemNet Distance (FCD) 지표를 사용하여 평가된 최첨단 결과를 보였습니다. 이 모델은 고유성과 유효성이 높은 새로운 분자를 생성해낼 수 있는 일관된 순서를 학습했습니다. 여러 구조적 선택을 조사하고, 분자 그래프 생성 맥락에서 분석을 수행하여 모델의 성능을 강화했습니다.



### Is Your Video Language Model a Reliable Judge? (https://arxiv.org/abs/2503.05977)
- **What's New**: 비디오 언어 모델(Video Language Models, VLMs)의 사용이 증가함에 따라, 이들의 성능을 평가하는 방법과 관련된 연구가 진행되고 있습니다. 기존의 전문가 기반 평가 방식은 일관성과 확장성에서 한계가 있으며, 이러한 문제를 해결하기 위해 VLM을 사용하여 VLM을 평가하는 자동화 방법에 대한 관심이 높아지고 있습니다. 그러나 VLM이 평가지로서의 신뢰성에 대한 연구는 미비한 실정입니다.

- **Technical Details**: 이 논문에서는 VLM이 다른 VLM의 성능을 평가할 때의 한계를 조사하며, 특히 신뢰할 수 있는 모델과 그렇지 않은 모델을 혼합하여 사용했을 때 집단적 사고(Collective Thought) 접근이 평가의 신뢰성을 얼마나 향상시키는지를 분석합니다. 최근의 연구들은 일반적으로 하나의 VLM에게만 의존하지만, 이는 편향된 결과를 초래하거나 신뢰성을 저하시킬 수 있음을 강조합니다. 이를 위해 Video-LLaVA라는 낮은 성능 모델을 미세 조정(Fine-tuning)하고, 신뢰성에 영향을 미치는 요소를 탐구합니다.

- **Performance Highlights**: 연구 결과에 따르면, 신뢰성이 낮은 평가자가 포함된 집단적 평가가 반드시 최종 평가의 정확성을 개선하지 않는다는 사실이 밝혀졌습니다. 신뢰성이 낮은 모델은 결과에 노이즈를 추가하여 집단의 이점을 상쇄할 수 있다는 점이 발견되었습니다. 이러한 결과는 VLM의 평가 신뢰성을 높이기 위한 보다 정교한 방법의 필요성을 강조하며, 평가 프레임워크의 설계에 대한 통찰력을 제공합니다.



### Optimal sensor deception in stochastic environments with partial observability to mislead a robot to a decoy goa (https://arxiv.org/abs/2503.05972)
- **What's New**: 이 논문에서는 자율 시스템이 적대적 환경에서 사용되는 속임수 기법을 새로운 방식으로 접근합니다. 특히, 로봇을 미끼 목표로 유도하기 위해 센서 데이터를 조작하는 방법을 제안하며, 한정된 예산 내에서 최대한의 효과를 발휘합니다. 기존의 미끼 목표 또는 여유 목표를 최적화하는 방법과는 달리, 고정된 미끼 목표를 가지고 센서 관측을 효과적으로 교환하는 데 중점을 두었습니다.

- **Technical Details**: 이 논문은 자율 로봇과 환경 간의 상호작용을 부분적으로 관찰 가능한 마르코프 결정 과정(Partially Observable Markov Decision Process, POMDP)으로 모델링합니다. 로봇은 센서 관측에 의존하고 있으며, 해당 관측은 비용을 들여 전략적으로 변경될 수 있습니다. 이 문제의 계산적 난이도는 0/1 배낭 문제(Knapsack problem)에서의 축소를 통해 입증되며, 최적의 속임수 전략을 도출하기 위해 혼합 정수 선형 프로그래밍(Mixed Integer Linear Programming, MILP) 모델을 제안합니다.

- **Performance Highlights**: 제안된 MILP 모델의 효과성은 일련의 실험을 통해 확인되었습니다. 이 방법을 통해 제한된 비용 내에서 로봇을 미끼 목표로 유도하는 데 있어 최적의 센서 변경 전략을 계산할 수 있습니다. 이 접근법은 다중 에이전트 시스템과 보안 응용 프로그램에서 자율 시스템 계획에 광범위하게 적용될 수 있습니다.



### A Real-time Multimodal Transformer Neural Network-powered Wildfire Forecasting System (https://arxiv.org/abs/2503.05971)
- **What's New**: 현재 기후 변화로 인해 극심한 산불이 인류 문명에 가장 위험한 자연재해 중 하나로 부각되고 있습니다. 본 연구에서는 실시간 Multimodal Transformer Neural Network 머신러닝 모델을 개발하여 특정 위치의 산불 발생을 예측하는 방법을 제안했습니다. 이 모델은 대규모 기상 데이터와 구글 어스(Google Earth) 이미지에서 수집한 소규모 지형 및 식생 정보를 활용하여, 24시간 이내의 산불 발생 확률을 제공합니다.

- **Technical Details**: 이 연구는 1992년부터 2015년까지 수집된 미국의 산불 데이터를 사용하여 머신러닝 모델을 훈련했습니다. 모델은 날씨 상황 등을 고려한 다양한 환경 요인과 함께 24년간의 산불 기록을 분석합니다. 정보 데이터는 산불 정보, 기상 정보 및 식생 정보로 구성되어 있으며, 이미지는 Google Earth Pro를 통해 확보했습니다. 이 시스템은 공간적인 확률 분포를 통해 산불 예측 정확성을 높이는 데에 중점을 둡니다.

- **Performance Highlights**: 제안된 모델은 기존의 모델보다 훨씬 더 세부적인 소규모 지역에서의 산불 발생 예측에 강점을 보입니다. 예측 정확성을 향상시키기 위해 다양한 기상 조건과 토지 이용 정보와의 상관관계를 학습합니다. 이를 통해 낮은 비용과 빠른 시간 안에 특정 예측 지역의 식생 상태를 조절할 수 있는 가능성을 열었습니다. 모델은 향후 산불 예방 및 대응 전략 수립에 기여할 것으로 기대됩니다.



### Explaining the Unexplainable: A Systematic Review of Explainable AI in Financ (https://arxiv.org/abs/2503.05966)
Comments:
          2 tables, 11 figures

- **What's New**: 이 논문은 Explainable Artificial Intelligence (XAI)과 금융 분야의 접점을 분석하여, XAI의 최신 응용 및 연구 동향을 조망합니다. 특히, 금융 분야의 특정 구현 방법론과 연구의 경향 맵을 제시합니다. 다양한 기술을 활용한 분석 결과를 통해 연구자들에게 유용한 정보를 제공합니다.

- **Technical Details**: 이 연구는 bibliometric 및 content analysis를 통해 금융 산업에서 사용되는 주제 클러스터와 주요 연구를 파악합니다. XAI에서 사용되는 설명 가능성 전략에는 attention mechanism, feature importance analysis, SHAP(Shapley Additive Explanations) 기법 등이 있으며, 특히 post-hoc interpretability 기술에 대한 종속성이 두드러집니다. 이러한 기법들은 금융 결정을 더욱 투명하게 만들어 줍니다.

- **Performance Highlights**: 제시된 결과는 현재 XAI 시스템에 중요한 단점을 드러내며, 금융 분야의 특성과 더불어 설명 가능성의 개선을 위한 다학제적인 접근이 필요하다는 점을 강조합니다. 또한 연구자와 실무자 간의 협력을 통해 더욱 효과적인 XAI 개발에 기여할 수 있는 기회를 제시합니다.



### SANDWiCH: Semantical Analysis of Neighbours for Disambiguating Words in Context ad Hoc (https://arxiv.org/abs/2503.05958)
Comments:
          15 pages, 2 figures, 7 tables, NAACL 2025

- **What's New**: 본 논문에서는 다국어 단어 의미 구별(Word Sense Disambiguation, WSD)을 위한 새로운 프레임워크인 SANDWiCH를 소개합니다. 이 프레임워크는 전통적인 개별 의미 구별에서 시맨틱 네트워크의 클러스터 구별로 문제를 재구성합니다. 우리의 접근 방식은 BabelNet를 사용해 정제된 시맨틱 네트워크를 활용하며, 파라미터 수를 72% 줄여 성능을 향상시킵니다.

- **Technical Details**: SANDWiCH는 두 수준의 프레임워크로 구성되어 있으며, 먼저 의미를 구분한 시맨틱 네트워크를 처리하고, 이웃 개념을 훈련 데이터의 일부로 포함하여 모델을 구분합니다. 주어진 문맥에서 후보 의미 간의 시맨틱 유사성을 평가하여 클러스터 기반의 접근 방식을 취합니다. 실험 결과, 모델은 영어 단어 의미 구별 작업에서 F1 점수를 8% 개선했으며, 다양한 작업에서 우수한 성과를 보였습니다.

- **Performance Highlights**: 영어 및 다국어 WSD 데이터셋에서 이전의 최첨단 결과를 상회하는 성능을 달성했습니다. 특히 저자원이 언어에서도 유의미한 개선을 보여주었으며, 기존 솔루션 대비 모든 데이터셋과 품사에서 뛰어난 성과를 달성했습니다. 이로 인해 SANDWiCH 프레임워크는 다양한 언어의 WSD 성능 격차를 줄이는 데 기여할 것으로 기대됩니다.



### TPU-Gen: LLM-Driven Custom Tensor Processing Unit Generator (https://arxiv.org/abs/2503.05951)
Comments:
          8 Pages, 9 Figures, 5 Tables

- **What's New**: 이번 논문에서는 TPU-Gen이라는 새로운 프레임워크를 소개합니다. TPU-Gen은 고급 언어 모델(LLM)을 이용하여 TPU 설계를 자동화하는 최초의 시스템으로, 시스톨릭 배열 아키텍처에 초점을 맞추었습니다. 또한, TPU-Gen은 다양한 공간 배열 설계와 근사 곱셈-누적 단위를 포괄하는 개방형 데이터셋을 지원하여 DNN 작업에 맞는 설계를 재사용하고 맞춤화할 수 있도록 합니다.

- **Technical Details**: TPU-Gen은 정밀하고 근사적인 TPU 생성 프로세스를 자동화하는 데 중점을 둡니다. 특히, Retrieval-Augmented Generation(RAG) 기법을 활용하여 드문 데이터 환경에서 LLM의 환각 문제를 해결합니다. 이 프레임워크는 높은 수준의 아키텍처 명세를 최적화된 저수준 구현으로 변환하는 효과적인 하드웨어 생성 파이프라인을 가지고 있습니다.

- **Performance Highlights**: 실험 결과, TPU-Gen은 평균적으로 수동 최적화 참조 값보다 면적과 전력에서 각각 92% 및 96%를 줄이는 뛰어난 성능을 보여주었습니다. 이러한 결과는 LLM 기반의 차세대 디자인 자동화 도구의 발전을 추진하는 새로운 기준을 제시합니다.



### Uncertainty Quantification From Scaling Laws in Deep Neural Networks (https://arxiv.org/abs/2503.05938)
Comments:
          18+3 pages, 6 figures

- **What's New**: 이 논문은 물리학에서 머신러닝 분석의 불확실성을 정량화하는 것에 초점을 맞추고 있습니다. 특히, 신경망의 초기화 분포에서 유래되는 불확실성에 대해 논의하며, 무한 너비(multi-layer perceptron, MLP)에서의 특성인 신경 접합 커널(neural tangent kernel, NTK) 초기화를 통해 평균(micro)과 분산(variance)을 계산합니다. 세 가지 예제 작업인 MNIST 분류, CIFAR 분류 및 에너지 회귀(calorimeter energy regression)에 대해 실험적으로 비교했습니다.

- **Technical Details**: 연구에서는 다층 퍼셉트론(MLP)의 무한 너비 한계에서 신경망 출력 통계가 가우시안 분포를 따르며, 평균과 분산이 NTK에 의해 결정된다는 점을 활용합니다. 이 연구는 R에서 진행된 실험을 통해 크기가 커지는 훈련 데이터(N_D) 집합의 영향에 대해 비트별 계산이 부담스럽지 않음을 보여줍니다. 이러한 통계를 두고 각각의 MLP 구조와 상관없이 비슷한 스케일링 법칙을 발견했습니다.

- **Performance Highlights**: 저자들은 샘플링 크기가 커지면 변동 계수(coefficient of variation)가 특정 상수 값에 수렴한다는 점을 발견했습니다. 여러 데이터셋에 대한 실험에서 영향을 미치기 위해 변동 계수의 스케일링 지수가 대략 0임을 나타내는 중요한 결과를 도출했습니다. 이 결과는 무한 너비 네트워크의 비슷한 경향이 유한 너비 네트워크에서도 유지된다는 것을 지원하는 수치 실험을 통해 입증되었습니다.



### The Unified Control Framework: Establishing a Common Foundation for Enterprise AI Governance, Risk Management and Regulatory Complianc (https://arxiv.org/abs/2503.05937)
- **What's New**: 이 연구는 AI 시스템의 급격한 도입이 기업에 혁신과 책임 있는 관리라는 두 가지 도전 과제를 안겨주고 있다는 점을 강조합니다. 현재의 AI 거버넌스 접근 방식은 단편화되어 있어 리스크 관리 프레임워크가 고립된 도메인에 집중하며, 규제가 서로 다른 관할권에서 상이하고 구체적인 실행 지침이 부족합니다. 이 논문에서는 이러한 문제를 해결하기 위해 통합 제어 프레임워크(Unified Control Framework, UCF)를 제안합니다.

- **Technical Details**: UCF는 세 가지 주요 구성 요소로 이루어져 있습니다: (1) 조직적 및 사회적 리스크를 종합한 포괄적 리스크 분류 체계, (2) 규정에서 파생된 구조화된 정책 요구 사항, (3) 다수의 리스크 시나리오와 준수 요구 사항을 동시에 해결할 수 있는 42개의 간략한 제어 조치입니다. 이를 통해 조직은 효율적이고 적응 가능한 거버넌스를 수립할 수 있으며, 규제의 스케일과 관계없이 리스크를 관리할 수 있게 됩니다.

- **Performance Highlights**: UCF는 중복되는 노력과 비용을 줄이며 포괄적인 커버리지를 보장합니다. 이 프레임워크는 개별 권리를 보호하고 공정성을 증진하며 AI 시스템이 인간의 삶에 미치는 영향을 투명하게 하는 데 필요한 감독 메커니즘의 이행을 지원합니다. 따라서 책임 있는 AI 거버넌스를 구현하면서도 혁신의 속도를 희생하지 않도록 도움을 줄 수 있습니다.



### Audio-to-Image Encoding for Improved Voice Characteristic Detection Using Deep Convolutional Neural Networks (https://arxiv.org/abs/2503.05929)
Comments:
          11 pages, 24 figures, 1 table, 3 algorithms. Submitted to F1000Research

- **What's New**: 이 논문은 다중 차원의 음성 특성을 단일 RGB 이미지로 통합하는 새로운 오디오-이미지 인코딩 프레임워크를 소개합니다. 이러한 방법은 음성 인식에서 스피커(recognition) 정확성을 높이는 데 기여할 것으로 보입니다.

- **Technical Details**: 이 프레임워크는 녹색 채널(green channel)에 원시 오디오 데이터(raw audio data)를 인코딩하고, 빨간색 채널(red channel)에는 음성 신호의 통계적 설명(statistical descriptors)을 포함합니다. 이러한 설명에는 기본 주파수(fundamental frequency), 스펙트럼 중심(spectral centroid), 대역폭(bandwidth), 롤오프(rolloff) 등과 같은 주요 메트릭이 포함됩니다.

- **Performance Highlights**: 이 통합된 다채널 표현은 음성 인식 작업에서 보다 판별적인(input) 입력을 제공할 수 있음을 시사합니다. 심층 합성곱 신경망(deep convolutional neural network)을 활용하여 두 명의 화자(speakers)를 대상으로 한 실험에서 98%의 분류 정확도를 달성했습니다.



### ElementaryNet: A Non-Strategic Neural Network for Predicting Human Behavior in Normal-Form Games (https://arxiv.org/abs/2503.05925)
Comments:
          14 pages. Submitted to EC 2025

- **What's New**: 이번 논문은 GameNet의 레벨-0(“level-0”) 행동 규정이 실제로 전략적 추론을 수행할 수 있다는 점을 밝혀냈습니다. 이를 통해 GameNet이 비전략적 행동을 모델링하는 데 적합하지 않다는 것을 증명하였습니다. 또한, 새로운 신경망 구조인 ElementaryNet을 소개하며 이 구조가 비전략적 행동만을 수행하도록 설계되었음을 보여주었습니다.

- **Technical Details**: ElementaryNet은 게임의 각 결과에서 공동 보상을 단일 숫자로 매핑하는 간단한 잠재 함수(potential function)를 적용한 후, 복잡한 신경망 레이어를 통해 이 잠재 행렬을 변환하는 두 단계로 나뉘어 있습니다. 이 모델은 Wright와 Leyton-Brown(2019)이 정의한 기본 모델(elmentary model) 가족의 조합으로, 비전략적 행동만 가능하다는 것을 입증합니다. GameNet과 달리, ElementaryNet은 상대의 보상에 관한 풍부한 신념 형성을 차단하여 전략적 반응을 불가능하게 만듭니다.

- **Performance Highlights**: 실험 결과는 ElementaryNet이 GameNet보다 성능이 현저히 낮다는 것을 보여주었지만, 상위 에이전트를 도입하면 두 모델의 성능이 통계적으로 구별될 수 없다는 점을 발견했습니다. 이는 ElementaryNet의 비전략적 레벨-0 규정이 모델 성능을 저하시킨다는 것을 의미하지 않으며, Pre-existing literatures에서 제안된 레벨-0 구성요소에 대해 제한이 가해져도 여전히 이 결과가 유지되는 것으로 나타났습니다.



### IDEA Prune: An Integrated Enlarge-and-Prune Pipeline in Generative Language Model Pretraining (https://arxiv.org/abs/2503.05920)
- **What's New**: 최근 대형 언어 모델의 발전은 제한된 추론 예산 내에서 효율적이고 배포 가능한 모델의 필요성을 증가시켰습니다. 본 논문에서는 이전 연구에서 종종 간과된 확대 모델의 사전 훈련을 가지치기(Pruning) 파이프라인에 통합하는 것을 제안합니다. 우리는 'enlarge-and-prune' 파이프라인을 통해 모델의 성능을 저하시키지 않으면서도 개선할 수 있는 방법을 연구합니다.

- **Technical Details**: 우리는 통합된 'IDEA Prune' 파이프라인을 제안하여 확대 모델 훈련, 가지치기, 복구 단계를 하나의 cosine annealing learning rate 일정 아래 통합합니다. 이 접근법은 학습 속도의 상승에서 오는 지식 손실을 완화하고 생존하는 뉴런 사이에서 모델 용량을 효과적으로 재분배합니다. 또한, 점진적인 파라미터 제거를 위한 새로운 반복적 구조화 가지치기 방법을 적용하여 FB에 대한 압축을 진행합니다.

- **Performance Highlights**: 우리의 실험 결과에 따르면, 2.8B 모델을 1.3B로 압축하면서 최대 2T의 훈련 토큰을 사용했습니다. IDEA Prune은 기존의 방법들과 비교했을 때 일관된 성능 개선을 보여주며, 특히 MMLU 정확도가 46.4%에 달해 기본 방법의 31.4-33.4%보다 현저하게 향상되었습니다. 우리는 추가적인 연구를 통해 중간 체크포인트가 가지치기 시작점을 더욱 유리하게 제공할 수 있음을 확인했습니다.



### SAS: Segment Anything Small for Ultrasound -- A Non-Generative Data Augmentation Technique for Robust Deep Learning in Ultrasound Imaging (https://arxiv.org/abs/2503.05916)
Comments:
          25 pages, 8 figures

- **What's New**: 이번 논문에서는 초음파(ultrasound) 이미지에서 작은 해부학적 구조물의 정확한 분할을 위한 효과적인 데이터 보강 기법인 Segment Anything Small(SAS)를 소개합니다. SAS는 조직의 크기와 질감을 동시에 고려하여, 작은 구조물의 분할 성능을 향상시키기 위해 두 가지 변환 전략을 활용합니다. 마지막으로, SAS는 기존 생성 모델과 달리 비생성 방식으로 훈련 데이터를 다양화하여 노이즈를 줄이고, 오차를 최소화합니다.

- **Technical Details**: SAS 방법론은 두 가지 단계로 구성됩니다. 첫 번째 단계에서는 초음파 윈도우 내에서 관심 영역(ROI)을 추출하고, 이를 재조정하여 썸네일을 생성한 뒤 검은 배경에 배치합니다. 두 번째 단계에서는 세그멘테이션 마스크에서 정의된 기관 영역에 노이즈를 주입하여 다양한 질감을 모사합니다. 이를 통해 다양한 크기와 형태의 해부학적 구조에 대한 모델의 적응성과 일반성을 향상시킵니다.

- **Performance Highlights**: SAS에서 미세 조정된 모델은 내부 및 외부 데이터셋에서 실험을 진행한 결과, 최대 0.35의 Dice 점수 향상과 평균적으로는 0.16까지 성능이 개선되었습니다. 또한, SAS는 두 개의 클릭 프롬프트로 바운딩 박스 프롬프트와 유사한 성능을 달성하며, 희소한 데이터와 풍부한 데이터 환경 모두에서 소형 구조물의 분할 성능을 크게 향상시킵니다.



### Towards Understanding the Use of MLLM-Enabled Applications for Visual Interpretation by Blind and Low Vision Peop (https://arxiv.org/abs/2503.05899)
Comments:
          8 pages, 1 figure, 4 tables, to appear at CHI 2025

- **What's New**: 본 연구는 멀티모달 대규모 언어 모델(MLLM)을 활용한 시각 해석 애플리케이션이 시력을 잃거나 시력이 약한(BLV) 사용자들의 만족도와 신뢰도를 어떻게 변화시켰는지를 탐구합니다. 20명의 BLV 사용자가 사용한 MLLM 기반 애플리케이션에 대한 일주일간의 일기 연구 결과, 평균 4.15점의 사용자 만족도와 3.75점의 신뢰도가 나타났습니다. 향후 연구를 통해 BLV 사용자와 MLLM 지원 시스템 간의 상호작용 방식과 문제 해결 전략을 심층적으로 분석할 계획입니다.

- **Technical Details**: 이 연구에서는 사용자가 각각의 사용 후 짧은 설문조사를 제출하면서 553개의 일기 항목을 수집하였습니다. 연구자들은 6명의 참가자로부터 무작위로 선택된 60개의 일기 항목을 분석해 MLLM 기반의 시각 해석 애플리케이션이 어떻게 작동하는지를 살펴보았습니다. 이러한 시스템은 사용자의 사진과 질문을 종합하여 더 상세한 시각적 설명을 제공하는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: 일기 연구의 결과 참가자들은 MLLM 기반 애플리케이션을 사용하면서 높은 만족도( 평균 4.15점)와 신뢰도( 평균 3.75점)를 기록했습니다. 특히, 75%의 일기 항목에서 후속 대화가 이루어졌고, 이는 사용자가 한두 가지 질문을 하는 경향을 보여주었습니다. 이 애플리케이션은 고위험 상황에서도 사용되며, 사용자들은 약물 식별 및 복용량에 대한 상담을 신뢰하는 경향이 있었습니다.



### Zero-shot Medical Event Prediction Using a Generative Pre-trained Transformer on Electronic Health Records (https://arxiv.org/abs/2503.05893)
- **What's New**: 이 연구는 전자 건강 기록(EHR)에서 생성된 기초 모델을 기반으로 한 제로샷(zero-shot) 예측을 최초로 종합적으로 분석합니다. 기존의 모델들은 유료 세분화(train-fine-tune) 접근법을 사용하여 특정 작업에 맞게 조정해야 했으나, 본 연구에서는 그러한 세분화 없이도효과적인 예측이 가능하다는 점에서 혁신적입니다. 특히, 이 방법론은 사전 훈련된 지식을 활용하여 후속 의학적 사건을 예측할 수 있게 합니다.

- **Technical Details**: 본 연구의 모델은 240만 명 환자의 20년간의 건강 기록 데이터를 기반으로 훈련되었습니다. EHR을 효율적으로 처리하기 위해 OMOP(Common Data Model) 표준을 따랐고, 진단 코드, 절차 코드 및 약물 정보를 체계적으로 정리하였습니다. GPT-2 아키텍처를 기반으로 한 이 모델은 기존 언어 모델링과 달리 의료 이벤트를 예측하는 방향으로 설계되었습니다.

- **Performance Highlights**: 모델의 성능은 정밀도(precision)와 재현율(recall) 지표로 평가되었으며, 다음 의료 개념 예측에서 평균적으로 0.614의 정밀도와 0.524의 재현율을 달성하였습니다. 12개의 주요 진단 조건에 대해 모델은 높은 진양성(true positive) 비율을 보이며 낮은 위양성(false positive) 비율을 유지했습니다. 이 결과는 제로샷 예측의 가능성을 보여주며, 임상에서 더 넓은 응용이 가능함을 시사합니다.



### QG-SMS: Enhancing Test Item Analysis via Student Modeling and Simulation (https://arxiv.org/abs/2503.05888)
Comments:
          Under Review

- **What's New**: 이 연구에서는 교육 평가에서 질문 생성(Question Generation, QG) 작업의 평가 방식에 대한 새로운 접근법을 제시합니다. 기존의 QG 평가 방식은 교육적 가치와의 명확한 연계를 결여하고 있어, 테스트 항목 분석(test item analysis)을 도입하여 질문의 품질을 평가하는 방법을 강조하고 있습니다. 연구진은 품질이 다른 질문 쌍을 구성하여 기존의 QG 평가 방식이 이러한 차이를 효과적으로 구분할 수 있는지 살펴보았습니다.

- **Technical Details**: 연구에서는 질문 품질을 평가하기 위해 교육 분야에서 널리 사용되는 테스트 항목 분석(Method)은 주제 범위(topic coverage), 항목 난이도(item difficulty), 항목 변별력(item discrimination), 주의 산만 선택지의 효율성(distractor efficiency) 등 네 가지 차원에서 이루어집니다. 기존의 QG 평가 방식은 주제 범위에 대해서는 잘 작동하지만, 항목 난이도 및 변별력과 같은 후처리 분석의 차원을 정확하게 평가하는 데 큰 한계를 보였습니다. 이러한 문제를 해결하기 위해, 연구자들은 새로운 QG 평가 프레임워크인 QG-SMS를 제안하였습니다.

- **Performance Highlights**: QG-SMS는 다양한 학생 모델링 및 시뮬레이션을 활용하여 질문 품질 평가에서 성능을 높이며, 이는 기존의 LLM 기반 평가 방법의 한계를 극복합니다. 본 연구에서는 대규모 실험과 인간 평가를 통해 QG-SMS의 효과성과 강건성을 입증하였습니다. 연구 결과는 교육적 측면에서의 질문 품질 평가에 있어 significant improvements를 보여줍니다.



### Practical Topics in Optimization (https://arxiv.org/abs/2503.05882)
- **What's New**: 이번 논문은 데이터 중심의 의사결정(data-driven decision-making)과 계산 효율성(computational efficiency)이 중요한 시대에서 최적화(optimization)의 중요성을 강조합니다. 수학, 컴퓨터 과학, 운영 연구, 머신러닝 등 여러 분야에서 최적화 기술이 복잡한 문제를 해결하는 데 필수적인 도구로서 기능함을 보여주고 있습니다. 독자들이 다양한 분야에서 최적화 방법을 이해하고 적용할 수 있도록 도와주는 소개서 및 종합 참고서를 제공합니다.

- **Technical Details**: 이 책은 블랙 박스(black-box) 및 확률적 최적화(stochastic optimizers) 알고리즘의 작동 원리를 설명하여 독자들이 최적화 방법을 이해할 수 있도록 구성되어 있습니다. 기본적인 수학적 원리부터 시작하여 주요 결과를 도출하는 방식으로 접근, 이들이 어떻게 작동하는지 뿐만 아니라 언제, 왜 효과적으로 적용해야 하는지에 대해 이해할 수 있도록 도와줍니다. 이론적 깊이와 실용적 응용 간의 균형을 유지하여, 학생 및 연구자뿐만 아니라 강력한 최적화 전략을 찾는 실무자들에게도 적합한 내용을 담고 있습니다.

- **Performance Highlights**: 본 논문은 최적화 알고리즘과 그 적용 방안에 대하여 명확하고 직관적인 설명을 제공합니다. 이를 통해 독자들은 최적화 기술의 기초를 이해하고 실질적인 문제에 어떻게 적용하여 효율적인 결과를 얻을 수 있는지를 학습할 수 있습니다. 다양한 사례를 통하여 최적화의 강점을 보여주고, 더욱 향상된 의사결정을 위한 전략을 제시합니다.



### Benchmarking AI Models in Software Engineering: A Review, Search Tool, and Enhancement Protoco (https://arxiv.org/abs/2503.05860)
- **What's New**: 이 논문에서는 인공지능(AI)이 소프트웨어 엔지니어링에 통합되면서 등장한 204개의 AI4SE 벤치마크를 정리하고 그 한계를 분석합니다. 또한, BenchScout라는 시맨틱 검색 도구를 개발하여 관련 벤치마크를 효과적으로 찾을 수 있도록 지원합니다. BenchScout의 사용성, 효과성, 직관성을 평가한 결과, 평균 점수는 4.5, 4.0, 4.1로 나타났습니다. Benchmarking의 기준을 발전시키기 위해 BenchFrame이라 불리는 통합 방법을 제안합니다.

- **Technical Details**: 본 연구에서는 173개의 연구를 리뷰하며 AI4SE 벤치마크를 분류하고 분석했습니다. 현재의 벤치마크 개발 표준이 부족한 상황에서, BenchScout는 관련 연구의 맥락을 자동 클러스터링하여 벤치마크 검색을 지원합니다. 사례 연구로 HumanEval 벤치마크에 BenchFrame을 적용하여 오류 수정, 언어 변환 개선, 테스트 범위 확대, 난이도 증가를 이루는 HumanEvalNext를 개발했습니다.

- **Performance Highlights**: HumanEvalNext에서 10개의 최첨단 코드 언어 모델을 평가한 결과, HumanEval 및 HumanEvalPlus에 비해 pass@1 점수가 각각 31.22% 및 19.94% 감소했습니다. 이는 HumanEvalNext가 모델 성능의 측정을 더욱 엄격하게 요구함을 나타냅니다. 이러한 성과는 AI4SE 분야에서의 Benchmarking의 중요성과 그 필요성을 강조합니다.



### Bimodal Connection Attention Fusion for Speech Emotion Recognition (https://arxiv.org/abs/2503.05858)
- **What's New**: 이번 연구에서는 Bimodal Connection Attention Fusion (BCAF)이라는 새로운 방법을 제안합니다. 이 방법은 오디오 및 텍스트 간의 상호작용을 모델링하기 위한 인터랙티브 연결 네트워크와 bimodal attention 네트워크, 상관관계 주의 네트워크로 구성되어 있습니다. 이 방법은 감정 인식 시스템의 성능을 향상시키기 위해 모달리티 간의 연결 및 상호작용을 효과적으로 포착하는 데 중점을 둡니다.

- **Technical Details**: BCAF 아키텍처는 유니 모달 표현 모듈, 연결 주의 융합 모듈, 분류 모듈로 구성되어 있습니다. 오디오 모달리티 인코더로는 wav2vec 모델이 사용되어 1024차원 오디오 표현을 추출합니다. 텍스트 모달리티에는 RoBERTa 모델을 사용하여 1024차원 텍스트 표현을 생성하며, 이러한 표현을 통해 서로 다른 모달리티 간의 세밀한 관계를 학습합니다.

- **Performance Highlights**: MELD 및 IEMOCAP 데이터셋에서의 실험 결과, 제안한 BCAF 방법이 기존의 최첨단 기준을 능가하는 성능을 보여주었습니다. BCAF는 독립적인 모달리티와 융합 모델을 동시에 적용하여 감정 인식의 정확도와 강건성을 높였습니다. 이러한 성능 향상은 감정 인식 시스템의 실제 애플리케이션에서의 활용 가능성을 높입니다.



### SYMBIOSIS: Systems Thinking and Machine Intelligence for Better Outcomes in Society (https://arxiv.org/abs/2503.05857)
- **What's New**: 이 논문에서는 SYMBIOSIS라는 AI 기반의 프레임워크와 플랫폼을 소개하며, 이를 통해 시스템 사고가 사회적 도전 과제를 해결하는 데 도움을 줄 수 있도록 접근성을 높였다는 점이 주목할 만하다. 이 플랫폼은 지속 가능한 개발 목표(SDGs) 및 사회적 주제로 분류된 시스템 사고/시스템 역학 모델의 중앙 집중형 오픈 소스 저장소를 구축하며, 자연어(Natural Language)로 복잡한 시스템 표현을 번역할 수 있는 생성적 코파일럿을 개발하였다.

- **Technical Details**: SYMBIOSIS는 커뮤니티 기반 시스템 역학(CBSD)의 기반 위에 세워져 있으며, 다양한 커뮤니티가 시스템 사고 모델을 탐색하고 활용할 수 있도록 설계되었다. 이는 일반적으로 비전문가에게 접근이 어려운 시스템 역학 도구와 모델의 복잡한 표기법을 극복할 수 있도록 돕는다. 알고리즘적 추론(Causal Reasoning)과 귀납적 추론(Abductive Reasoning)과 같은 AI의 문제 해결 능력을 발전시키기 위한 방법으로, 시스템 사고의 프레임워크가 AI에 잘 결합될 수 있도록 지원한다. 이 시스템은 복잡한 사회 문제를 보다 효율적으로 다룰 수 있도록 설계되었다.

- **Performance Highlights**: SYMBIOSIS는 ML 개발자가 커뮤니티의 고유한 지식을 활용하여 문제 이해에서의 간극을 줄일 수 있게 도와주며, 사회적 배경을 고려한 효과적인 모델을 개발할 수 있는 기회를 제공한다. 이를 통해 AI가 사회적 맥락에서 안전하고 효과적으로 작동할 수 있도록 하는 첫걸음을 내딛었다. 마지막으로, SYMBIOSIS는 사용자 친화적인 접근 방식을 채택하여 비전문가들이 시스템 모델의 탐색 및 생성을 용이하게 할 수 있게 만들어, AI 시스템의 공정성과 효과성을 향상시킬 전망이다.



### This Is Your Doge, If It Please You: Exploring Deception and Robustness in Mixture of LLMs (https://arxiv.org/abs/2503.05856)
Comments:
          35 pages, 9 figures, 16 tables

- **What's New**: Mixture of large language model (LLMs) Agents (MoA) 아키텍처는 AlpacaEval 2.0과 같은 주요 벤치마크에서 뛰어난 성과를 달성했습니다. 하지만 이러한 MoA의 안전성과 신뢰성에 대한 평가가 부족했습니다. 이 연구에서는 고의적으로 오해를 일으키는 응답을 제공하는 기만적 LLM 에이전트에 대한 MoA의 강건성을 평가하는 최초의 포괄적 연구를 제시합니다.

- **Technical Details**: MoA는 여러 LLM의 전문성을 집약하여 성능을 극대화하는 multi-layer 구조를 가진 시스템입니다. 이 시스템에서는 제안자(proposers)와 집계자(aggregators) 간의 협력을 통해 다양한 관점을 수집하고 응답의 품질을 향상시킵니다. MoA의 특징 중 하나는 탈중앙화(decentralization) 배치가 가능하다는 점이며, 이는 잠재적인 취약점을 초래할 수 있습니다.

- **Performance Highlights**: 연구 결과, 단 하나의 잘 조작된 기만적 에이전트를 MoA에 도입할 경우 성능이 37.9%로 감소되어 MoA의 모든 이점을 무력화할 수 있음을 보여줍니다. 또한, QuALITY 다항선택 이해 과제에서 정확도가 48.5% 급락하는 심각한 영향을 받습니다. 본 연구는 MoA의 안전성을 확보하기 위한 여러 방어 메커니즘을 제안합니다.



### Accelerating Earth Science Discovery via Multi-Agent LLM Systems (https://arxiv.org/abs/2503.05854)
Comments:
          10 pages, 1 figure. Perspective article

- **What's New**: 이 관점(Perspective)에서는 대형 언어 모델(LLMs)로 구동되는 다중 에이전트 시스템(MAS)의 지구과학 분야에서의 변혁적 가능성에 대해 탐구합니다. 지구과학 데이터 저장소 사용자들은 복잡한 데이터 형식, 일관되지 않은 메타데이터 관행, 처리되지 않은 데이터 세트 등의 문제로 인해 어려움을 겪고 있습니다. MAS는 지능형 데이터 처리, 자연어 인터페이스 및 협업 문제 해결 기능을 통해 과학자와 지구과학 데이터의 상호작용을 개선할 수 있는 잠재력을 가지고 있습니다.

- **Technical Details**: 지구과학 데이터 아카이브는 지구 및 환경 데이터 세트를 체계적으로 저장 및 배포하는 큐레이션된 디지털 인프라로, PANGAEA와 NASA의 EOSDIS 등 대형 저장소가 수백만 개의 이질적인 데이터 세트와 수 페타바이트의 데이터를 수용하고 있습니다. 이러한 데이터 아카이브의 문제는 메타데이터 기준의 부족, 비일관한 데이터 형식 및 불완전한 문서화 때문이며, 이러한 문제들은 많은 데이터 세트가 과학 출판에서 활용되지 않는 원인이 됩니다.

- **Performance Highlights**: MAS는 다양한 과학 분야에서 이미 인상적인 결과를 보였습니다. 예를 들어, ShapefileGPT와 GeoLLM-Squad은 지구과학 관련 작업을 위한 MAS 프로토타입을 보여주며, 다양한 도구를 통해 복잡한 문제에 대한 전문적인 해결을 제공하고 있습니다. 이러한 시스템은 지구과학 데이터 관리에서 데이터 접근성을 향상시키고, 학문 간 협력을 증진하며, 지구과학 발견을 가속화하는 데 기여할 것으로 기대됩니다.



### Evaluating Large Language Models in Code Generation: INFINITE Methodology for Defining the Inference Index (https://arxiv.org/abs/2503.05852)
Comments:
          20 pages, 6 figures

- **What's New**: 이 연구는 코드 생성 작업에서 대형 언어 모델(LLMs)의 성능을 평가하기 위해 새로운 방법론인 INFinite(INFerence INdex In Testing model Effectiveness)를 제시합니다. InI 지수는 효율성, 일관성 및 정확성을 중심으로 한 포괄적인 평가를 제공합니다. 이 접근 방식은 전통적인 정확성 메트릭을 넘어 LLM 성능에 대한 깊은 이해를 제공합니다.

- **Technical Details**: INFINITE 방법론은 LLM이 생성한 코드를 평가하기 위해 복합적인 요소들을 분석하고 정량화하는 독특한 방식으로 설계되었습니다. 이 구조는 질문과 그에 대한 응답의 반복 과정을 통해 각 모델의 반응 시간을 평가하고, 명확하고 신뢰할 수 있는 코드 생성을 도출할 수 있는 요소들을 고려합니다. 이들은 효율성, 일관성, 정확성으로 나누어지며, 효율성은 응답 시간과 서버 바쁨 비율에 따라 평가됩니다.

- **Performance Highlights**: 연구 결과에 따르면, OpenAI의 GPT-4o는 OAI1보다 성능이 우수하며 OAI3와 비슷한 정확도와 작업 효율성을 보여주었습니다. LLM이 보조하는 코드 생성이 효과적인 프롬프트와 수정을 통해 전문가 설계 모델과 유사한 결과를 만들어낼 수 있음을 나타냅니다. GPT의 성능 우위는 광범위한 사용과 사용자 피드백의 이점을 강조합니다.



### Extracting and Emulsifying Cultural Explanation to Improve Multilingual Capability of LLMs (https://arxiv.org/abs/2503.05846)
Comments:
          under review, 18pages

- **What's New**: 최근 연구에서는 다국어 능력을 개선하기 위한 새로운 접근법인 EMCEI(Extract and emulsify Cultural explanation)를 제안합니다. EMCEI는 문화적 맥락을 통합하여 LLMs의 응답을 더욱 정확하고 적절하게 만듭니다. 이를 통해 기존의 다국어 프롬프트 방식이 간과한 문화적 요소를 반영하여 다양한 언어로 질높은 응답을 제공합니다.

- **Technical Details**: EMCEI는 두 단계 프레임워크를 따릅니다. 첫 번째 단계에서는 LLM의 파라메트릭 지식을 활용하여 관련 문화적 정보를 추출합니다. 그 후, 두 번째 단계에서는 추출한 정보를 사용하여 다양한 유형의 질문에 맞춤형 응답을 생성하도록 LLM에 요청합니다.

- **Performance Highlights**: EMCEI의 효과는 여러 다국어 벤치마크에서 검증되었습니다. 예를 들어, EMCEI는 전통적인 방법에 비해 평균 16.4%의 향상을 보였고, 특히 자원이 적은 언어에서는 32.0%의 개선이 나타났습니다. 이러한 결과는 EMCEI가 문화적 지식을 효과적으로 활용하여 비영어 쿼리를 처리하는 데 큰 기여를 한다는 것을 보여줍니다.



### Machine Learned Force Fields: Fundamentals, its reach, and challenges (https://arxiv.org/abs/2503.05845)
Comments:
          9 figures

- **What's New**: 이 논문에서는 기계 학습 기반 힘장(Machine Learning Force Fields, MLFFs)의 개발을 통해 양자 수준의 정확성을 제공하면서도 계산 효율성을 크게 향상시킨 방법론을 설명합니다. 특히, SchNet 모델과 GDML 프레임워크의 구축을 강조하여, 손으로 만든 기술적 설계를 제거하고 데이터로부터 직접 학습하여 원자적 상호작용을 모델링합니다. 이러한 혁신은 과거에는 어려웠던 대규모 및 복잡한 시스템의 정확한 시뮬레이션을 가능하게 합니다.

- **Technical Details**: 논문에서는 MLFF의 기본 원리와 구현 방법론을 다루며, 신경망 잠재력(neural network potentials) 및 커널 기반 모델(kernel-based models) 등의 주요 방법론이 소개됩니다. SchNet은 메시지 전송 신경망(message-passing neural networks)을 기반으로 하여, 연속 필터 컨볼루션 레이어를 적용하여 양자 상호작용을 모델링하는 혁신적인 학습 프레임워크입니다. GDML은 해석 가능성과 물리적 영감을 가진 수학적으로 강건한 MLFF를 구성하는 예로 설명됩니다.

- **Performance Highlights**: MLFF는 원자적 시뮬레이션의 혁신을 가져오며, 거의 양자 수준의 정확도를 유지하면서 접근 가능한 시스템과 시간 범위를 크게 확장했습니다. 대규모 생체 분자의 동역학 연구부터 에너지 저장 및 촉매에 대한 새로운 재료 탐색까지 다양한 응용 분야에 걸쳐 있습니다. 그러나 모델 훈련을 위한 데이터 요구 사항, 다양한 화학 환경에서의 이전 가능성 및 학습된 표현의 해석 가능성과 같은 문제들은 여전히 해결해야 할 주요 과제로 남아 있습니다.



### AI-Facilitated Collective Judgements (https://arxiv.org/abs/2503.05830)
- **What's New**: 이 논문은 집단의 선호를 찾기 위한 기존 및 새롭게 제안된 계산 프레임워크의 설계 선택을 상세히 분석합니다. AI 보조 선호 유도(AI-assisted preference elicitation)의 역사적 역할을 검토하며, 선호는 의사결정(context) 환경에 의해 형성되며 객관적으로 포착되기 힘들다는 점을 강조합니다. 이러한 경고를 염두에 두고, AI를 활용한 집단적 판단(collective judgment)을 합리적인 집단 의사를 조성하는 도구로 탐구합니다.

- **Technical Details**: 논문에서는 의견 조사(opinion polls)가 어떻게 집단 선호를 포착하는지에 대한 이론적 배경을 제공하고, AI 기술의 발전이 집단적 선호의 수집 방식에 미치는 영향을 분석합니다. AI의 활용이 집단적 의사 결정 과정에서 발생할 수 있는 수많은 기술적 선택의 결과를 다루며, 예상치 못한 부작용도 경고합니다. 특히, 결정을 법적으로 구속할 수 있는 위험한 사용 사례를 조명합니다.

- **Performance Highlights**: AI 기반의 집단적 판단은 합리적인 대표(representation)와 의사 합의(agreement-seeking)를 촉진하는 데 유용할 수 있지만, 오용될 경우 점진적인 권한 박탈(disempowerment)이나 정치적 결과를 정당화하는 도구가 될 위험이 큽니다. 따라서 이러한 AI 도구의 설계와 사용에 대한 윤리적 고려가 필수적임을 강조합니다.



### Introduction to Artificial Consciousness: History, Current Trends and Ethical Challenges (https://arxiv.org/abs/2503.05823)
Comments:
          65 pages

- **What's New**: 최근 인공지능(AI)과 의식 과학의 발전에 따라 인공지능 의식(Artificial Consciousness, AC)이 주목받고 있습니다. 이 논문은 AC의 주요 주제와 현재의 동향을 폭넓게 개괄합니다. 또한, AC의 역사와 핵심 용어를 정립하여 맥락을 명확히 하고 약한 AC(Weak AC)와 강한 AC(Strong AC)의 차이를 설명합니다.

- **Technical Details**: 논문의 두 번째 부분에서는 AC 구현의 주요 동향을 분석하고, Global Workspace와 Attention Schema 간의 시너지를 강조합니다. 또한 인공지능 시스템의 내부 상태를 평가하는 문제에 대해 논의합니다. 이후 세 번째 부분에서는 AC 개발의 윤리적 차원을 분석하고, 관련된 중요 위험과 혁신적 기회를 밝혀냅니다.

- **Performance Highlights**: 마지막으로 AC 연구를 책임감 있게 안내하기 위한 권고사항을 제시하고, 이번 연구의 한계와 향후 연구 방향을 설명합니다. 결론적으로, AC는 과학 발전에 있어 필수적이자 불가피한 존재로 보이며, 이 혁신적인 연구 경로의 광범위한 영향을 다루기 위한 진지한 노력이 필요하다고 강조합니다.



### The impact of AI and peer feedback on research writing skills: a study using the CGScholar platform among Kazakhstani scholars (https://arxiv.org/abs/2503.05820)
Comments:
          11 pages, 5 figures

- **What's New**: 이번 연구는 카자흐스탄 학자들의 학술 작문 발전에 대한 AI와 동료 피드백의 영향을 다루고 있습니다. 연구는 UIUC(일리노이 대학교 어바나 샴페인)의 CGScholar 플랫폼을 사용하여 진행되었으며, 이는 협력적 학습(collaborative learning), 빅데이터(big data), 인공지능(artificial intelligence) 연구의 결과물입니다. 이 연구에서는 AI 도구와 동료 피드백 과정에 대한 친숙함이 학술 작문에 대한 피드백 수용에 어떻게 영향을 미치는지를 조사했습니다.

- **Technical Details**: 연구는 UIUC에서 교육에 중점을 둔 과학 인턴십에 등록한 36명의 학자들을 대상으로 진행되었습니다. Google Forms를 이용해 15개의 객관식 질문, 리커트 척도(Likert scale), 개방형 질문을 포함한 설문조사를 실시하였습니다. 또한, 참여자의 언어적 접근성을 보장하기 위해 영어와 러시아어로 설문이 제공되었으며, 연령, 성별, 모국어와 같은 인구통계학적 정보도 수집되었습니다.

- **Performance Highlights**: 분석 결과, AI 도구에 대한 친숙함과 피드백 반영의 개방성 사이에 중간 정도의 긍정적인 상관관계가 발견되었습니다. 연구 작문 경험과 동료 피드백에 대한 기대 간에는 강한 긍정적인 상관관계가 나타났으며, 특히 연구 방법론 분야에서 더욱 두드러졌습니다. 이 연구는 AI 도구와 전통적인 피드백 메커니즘의 통합이 학술적 작문 품질 향상에 기여할 수 있는 가능성을 보여줍니다.



### Will Neural Scaling Laws Activate Jevons' Paradox in AI Labor Markets? A Time-Varying Elasticity of Substitution (VES) Analysis (https://arxiv.org/abs/2503.05816)
- **What's New**: 이번 연구에서는 AI 채택과 관련된 Jevons의 역설(Jevons' Paradox)의 의미를 탐구합니다. 모델은 AI 발전과 노동 대체의 연결고리를 네 가지 주요 메커니즘을 통해 설명합니다: 하드웨어 및 알고리즘 향상으로 인한 계산 능력 증가, 로그 기반의 AI 능력 증가, 경쟁 압력으로 인한 마진 계산 비용 감소, 그리고 AI와 인간 노동 사이의 대체 탄력성이 시간이 지남에 따라 증가하는 것입니다.

- **Technical Details**: 연구에서는 시장 전환을 AI 생성 상품과 관련하여 대체 탄력성을 기반으로 분석합니다. Gørtz 정체성을 사용하여 AI와 인간 간의 대체 탄력성과 가격 탄력성을 통합하는 분석적 조건을 도출합니다. 연구 결과에 따르면 AI 상품의 가격이 하락하더라도 AI 품질이 인간 상품을 대체하기 위해 충분히 증가하지 않으면 시장 지배를 달성할 수 없습니다.

- **Performance Highlights**: И이번 논문은 세 가지 주요 기여를 중심으로 구성됩니다. 첫째로, Gørtz 정체성을 통해 AI와 인간 관세 간의 대체 탄력성 및 가격 탄력성 관계를 형식화했습니다. 둘째로, 변동 대체 탄력성을 사용하여 시장 전환에 대한 분석 조건을 도출했습니다. 마지막으로, Jevons의 역설이 발생하는 시간 단계별 다섯 가지 채택 과정을 제시하여 각 단계에서 작용하는 경제적 메커니즘을 강조합니다.



### Trust, Experience, and Innovation: Key Factors Shaping American Attitudes About AI (https://arxiv.org/abs/2503.05815)
Comments:
          35 pages, 3 figures, 2 tables, appendix

- **What's New**: 이번 연구는 미국 성인들을 대상으로 인공지능(AI)에 대한 태도의 복잡한 landscape(풍경)를 조사하였습니다. 특히 AI 기술의 발전으로 인한 특정 결과들에 대한 우려의 정도와 이러한 우려의 상관관계를 탐구하였습니다.

- **Technical Details**: 연구에서는 ChatGPT와 같은 대형 언어 모델을 사용할 경험, 과학에 대한 일반적인 신뢰도, 조심의 원칙(precautionary principle) 준수 여부와 무제한 혁신 지원 간의 관계 등 주요 변수를 통해 우려의 방향성과 강도를 분석하였습니다. 또한 성별과 같은 인구 통계적 요인도 고려되었습니다.

- **Performance Highlights**: 이 연구는 AI에 대한 미국 대중의 반응에 대한 귀중한 인사이트를 제공하며, AI의 규제 또는 발전을 촉진하기 위한 정책 개발에 특히 중요한 정보를 담고 있습니다. 결과적으로, 여론과 정책 결정과 관련된 중요한 기반 자료를 마련하였습니다.



### Intolerable Risk Threshold Recommendations for Artificial Intelligenc (https://arxiv.org/abs/2503.05812)
Comments:
          79 pages

- **What's New**: 이 논문에서는 최첨단 AI 발전의 기반 모델인 Frontier AI 모델들이 공공 안전, 인권, 경제 안정 및 사회적 가치에 심각한 위험을 초래할 수 있다고 경고합니다. 이러한 위험은 악의적인 오용, 시스템 실패, 의도하지 않은 연쇄 효과 등에서 발생할 수 있습니다. 이를 해결하기 위해 2024년 5월 AI 서울 정상 회의에서 16개의 글로벌 AI 산업 단체가 Frontier AI 안전 약속을 체결하였고, 27개 국가 및 EU가 이러한 기준을 정의할 의사를 발표했습니다.

- **Technical Details**: 조직들은 모델이나 시스템이 유발할 수 있는 심각한 위험의 기준을 설정하고 이를 공개해야 합니다. 이 기준은 충분히 완화되지 않을 경우 '용납할 수 없는' 수준으로 간주될 수 있는 임계값(thresholds)을 포함합니다. 또한, 급속도로 발전하는 AI 능력과 관련된 위험에 대한 데이터가 제한된 상황에서 '완벽하지 않아도 괜찮다(good, not perfect)'는 목표를 설정해야 한다고 강조합니다.

- **Performance Highlights**: 저자는 8개 위험 카테고리에 걸쳐 실질적 사례 연구를 기반으로 한 구체적인 임계값 추천을 제안합니다. 위험 카테고리는 다음과 같습니다: (1) 화학, 생물, 방사선 및 핵(CBRN) 무기, (2) 사이버 공격, (3) 모델 자율성, (4) 설득 및 조작, (5) 기만, (6) 독성, (7) 차별, (8) 사회경제적 파괴입니다. 이 논문은 정책 입안자와 산업 리더들이 용납할 수 없는 위험을 예방하는 데 초점을 맞춘 능동적인 위험 관리(proactive risk management)를 촉진하는 기초 자료로 활용될 수 있도록 하려는 목표를 가지고 있습니다.



### A Transformer Model for Predicting Chemical Reaction Products from Generic Templates (https://arxiv.org/abs/2503.05810)
- **What's New**: 본 연구에서는 화학 반응 결과를 예측하는 데 있어서의 한계를 극복하기 위해 Broad Reaction Set (BRS)라는 새로운 데이터셋을 제안합니다. 이는 20개의 일반적인 반응 템플릿을 포함하여 화학 공간(chemical space)을 효율적으로 탐색할 수 있게 합니다. 또한, ProPreT5라는 T5 모델을 도입하여 일반 템플릿 기반 접근법과 템플릿이 없는 방법 간의 균형을 이루려 합니다.

- **Technical Details**: BRS는 SMARTS 구문을 이용하여 표현되는 반응으로, 단일 반응이 아닌 보다 일반적 변환 패턴을 represe합니다. 이 모델은 템플릿 기반 방법과 템플릿이 없는 방법 모두에 대해 BRS 데이터셋과 USPTO MIT 데이터셋을 사용해 훈련되었습니다. ProPreT5는 반응 생성 정확성을 높이기에 유용하게 설계되었습니다.

- **Performance Highlights**: ProPreT5는 현실적이고 유효한 반응 산물(reaction products)을 생성하는 능력을 입증하며, 전체 화학 공간을 탐색할 수 있는 가능성을 보여줍니다. 기존의 상태-최고 접근법보다 많은 과제를 해결하는 것으로 나타났으며, 템플릿 기반 모델의 해석가능성도 향상시키겠다는 목표를 가지고 있습니다. 이 연구의 기여는 데이터셋과 모델의 제안뿐만 아니라 반응 예측의 정확성에 영향을 미치는 중요 정보의 분석을 포함합니다.



### Multi-agent Auto-Bidding with Latent Graph Diffusion Models (https://arxiv.org/abs/2503.05805)
- **What's New**: 이번 논문에서는 대규모 경매 환경을 모델링하기 위해 그래프 기반을 활용한 확산(diffusion) 기반 자동 입찰(auto-bidding) 프레임워크를 제안합니다. 기존 방법이 단순한 휴리스틱(feature) 표현에 의존하는 것과는 달리, 우리는 각 인상 기회(impression opportunities, IO)를 세부적으로 모델링하고 이를 그래프 구조로 구현함으로써 상호 의존성을 효과적으로 포착할 수 있는 혁신적인 접근 방식을 제시합니다. 이를 통해 다중 에이전트 동역학(multi-agent dynamics)을 잘 반영할 수 있는 자동 입찰 시스템을 개발하였습니다.

- **Technical Details**: 제안하는 Latent Graph Diffusion Model for Auto-Bidding (LGD-AB) 프레임워크는 두 가지 핵심 구성요소로 구성됩니다: (1) 경매 동역학을 인코드하는 그래프 기반 임베딩 모듈과 (2) 경매 예측 및 전략적 입찰 최적화를 위한 다중 에이전트 잠재적 확산 모델(multi-agent latent diffusion model)입니다. IO는 그래프의 노드로 표현되며, 그래프 합성곱(graph convolution)과 주의 기반 그래프 신경망(attention-based graph neural network, GNN)을 사용하여 처리됩니다. 이는 비노출 IO에 대한 추가 노드를 배치함으로써 희소한 상황에서도 컨텍스트를 유지할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, LGD-AB는 실제 및 합성 경매 시뮬레이션에서 기존의 여러 방법들에 비해 다양한 KPI 메트릭에서 향상된 자동 입찰 성능을 보여줍니다. 또한, 우리는 기존의 KPI 메트릭들에 대한 예측 정확성이 크게 향상되었음을 확인했습니다. 이러한 성과는 우리 모델의 효과성과 실제 환경에서의 적용 가능성을 더욱 확고히 합니다.



### Holistically Evaluating the Environmental Impact of Creating Language Models (https://arxiv.org/abs/2503.05804)
Comments:
          ICLR 2025 (spotlight)

- **What's New**: 이 논문은 대규모 언어 모델(LLMs)의 개발로 인한 환경 영향을 체계적으로 분석하여, 모델 개발 단계에서의 에너지 소비와 탄소 배출을 처음으로 상세히 보고합니다. 데이터 센터의 높은 에너지 효율성을 고려하더라도, 이 모델 시리즈는 약 493미터 톤의 탄소를 배출하였으며, 이는 미국 내 98가구를 1년 간 운전할 수 있는 전력과 같습니다. 이 연구는 AI 모델 개발의 진정한 환경 비용을 평가하고, 향후 AI 시스템의 개발에 있어 투명성을 요구하는 중요성을 강조합니다.

- **Technical Details**: 이 연구에서는 2천만 개에서 130억 개의 활성 파라미터를 가진 OLMo 시리즈 변환기 모델을 대상으로 합니다. 모델 개발, 훈련 및 추론 등 세 가지 주요 단계에서 전력 소비, 탄소 배출, 물 사용량을 측정합니다. 기존 설명 방식과 달리, 저자들은 매우 작은 시간 간격으로 전력 소비를 측정하여, 실제 전력 소비 패턴을 반영하고 있습니다. 이 방법론은 LLM의 전반적인 환경 영향을 보다 정확히 평가하기 위한 것입니다.

- **Performance Highlights**: 연구 결과, 모델 개발은 전체 환경 영향의 약 50%를 차지하며, 설치 아닌 인퍼런싱에도 많은 전력이 소모됩니다. 또한, 모델의 크기가 증가할수록 환경 영향도 비례적으로 증가함을 발견하였습니다. 이 논문은 AI 시스템의 환경 영향을 평가할 때 넓은 시각을 제공하며, 관련 업계에 대한 더 많은 투명성과 책임이 필요하다는 점을 강조합니다.



### Federated Learning Framework via Distributed Mutual Learning (https://arxiv.org/abs/2503.05803)
- **What's New**: 최근의 연구에 따르면, Federated Learning(연합 학습)에서 모델 가중치를 공유하는 전통적인 방법은 네트워크 대역폭에 부담을 주고 개인정보 보호에 위험을 초래하는 문제점이 있습니다. 이 논문에서는 분산된 상호 학습을 이용한 손실 기반의 대안 방법을 제안합니다. 이 방법은 클라이언트가 주기적으로 공용 테스트 세트에 대한 손실 예측값을 공유하여, 전송 부담을 줄이고 데이터 프라이버시를 유지하며 협업을 가능하게 합니다.

- **Technical Details**: 제안된 프레임워크는 클라이언트가 자신의 로컬 데이터에 기반하여 모델을 훈련한 후, 공통 테스트 세트에서 손실을 기반으로 업데이트하는 방식으로 진행됩니다. 각 클라이언트는 자체 손실 예측과 다른 클라이언트의 손실 평균을 조합하여 자신의 모델을 개선합니다. 또한 알고리즘 1을 통해 훈련 및 업데이트 프로세스를 상세히 설명하며, KL(Kullback-Leibler) 발산 손실을 활용하여 모델들이 서로로부터 학습할 수 있도록 설계되었습니다.

- **Performance Highlights**: 제안된 방법은 마스크 착용 감지 작업에서 기존의 가중치 공유 방법보다 뛰어난 성능을 나타냈습니다. 특히, 보지 못했던 데이터에서의 정확도를 높이며, 일반화 및 프라이버시 측면에서도 더 유리한 결과를 보였습니다. 이로써 데이터 프라이버시를 유지하면서도 높은 성능을 달성할 수 있는 새로운 프레임워크가 확립되었습니다.



### Illuminant and light direction estimation using Wasserstein distance method (https://arxiv.org/abs/2503.05802)
- **What's New**: 이번 연구에서는 이미지 처리에서 조명 추정(illumination estimation) 문제를 해결하기 위한 새로운 방법을 제시합니다. 기존의 RGB histograms와 GIST descriptors와 같은 전통적인 접근법은 복잡한 조명 환경에서 효과적이지 못했습니다. 제안하는 방법은 최적 운송 이론(optimal transport theory)에 기반한 Wasserstein 거리(Wasserstein distance)를 활용하여 이미지를 통해 조명의 방향과 조명원을 추정합니다.

- **Technical Details**: 이 연구는 다양한 실내 장면, 흑백 사진, 야경 이미지를 대상으로 한 실험을 통해 제안된 방법의 효과성을 입증합니다. 특히, 복잡한 조명 환경에서 우세한 조명원을 감지하고 그 방향을 정확히 추정하는 데 성공했다고 보고합니다. 기존의 통계적 방법(statistical methods)보다 더 뛰어난 성능을 보일 뿐만 아니라, 조명 소스의 로컬라이제이션(light source localization), 이미지 품질 평가(image quality assessment), 객체 탐지(object detection) 강화 등 다양한 응용 가능성을 제시합니다.

- **Performance Highlights**: 이 방법은 복잡한 조명 조건에서도 안정적인 결과를 낼 수 있어, 로봇 공학 분야에서 실세계 조명 문제를 해결하는 데 큰 도움이 될 것으로 기대됩니다. 향후 연구에서는 적응형 임계값(adaptive thresholding) 및 그래디언트 분석(gradient analysis)을 통합하여 정확도를 더욱 향상시킬 예정입니다. 이러한 요소들은 실제 환경에서의 조명 문제를 해결하기 위한 확장 가능한 솔루션을 제공할 것으로 보입니다.



### Fault Localization and State Estimation of Power Grid under Parallel Cyber-Physical Attacks (https://arxiv.org/abs/2503.05797)
Comments:
          10 pages, 3 figures, 5 tables, journal

- **What's New**: 이번 연구에서는 전력망을 대상으로 한 새로운 유형의 공격인 Parallel Cyber-Physical Attacks (PCPA)에 대한 접근법을 제안합니다. PCPA의 공격 메커니즘은 전송선의 단절뿐만 아니라 어드미턴스(admittance) 값을 수정하는 것을 포함합니다. 이를 바탕으로 연구자들은 공격으로부터 피해를 진단하기 위한 새로운 알고리즘을 개발하였습니다.

- **Technical Details**: 연구자는 전력 흐름 모델을 기반으로 한 Graph Attention Network 기반의 Fault Localization (GAT-FL) 알고리즘을 제안합니다. 이 알고리즘은 공격 받은 지역의 전압 위상 각도(voltage phase angle)를 복원하고, 전력 주입(power injection) 정보를 활용하여 물리적 공격의 위치를 파악합니다. 또한, LP(Linear Programming) 문제를 해결하여 전송선의 상태를 식별하는 Line State Identification (LSI) 알고리즘도 개발하였습니다.

- **Performance Highlights**: 제안된 GAT-FL 알고리즘은 기존 결과를 초월하여 사이버 공격이 진행되는 상황에서도 효과적인 Fault Localization을 제공합니다. 마찬가지로 LSI 알고리즘은 LP 문제를 통해 공격 지역 내 모든 전송선의 상태를 정확하고 효율적으로 식별할 수 있음을 시뮬레이션 실험을 통해 입증하였습니다.



### Towards Multi-Stakeholder Evaluation of ML Models: A Crowdsourcing Study on Metric Preferences in Job-matching System (https://arxiv.org/abs/2503.05796)
Comments:
          This version of the contribution has been accepted for publication, after peer review (when applicable) but is not the Version of Record and does not reflect post-acceptance improvements, or any corrections. Use of this Accepted Version is subject to the publisher's Accepted Manuscript terms of use this https URL

- **What's New**: 이 연구는 다양한 이해관계자들이 기계 학습(ML) 모델을 평가하는 과정에서 선호하는 메트릭(metric)을 조사했다. 총 837명의 참가자에게 두 가지 가상의 ML 모델을 비교하도록 요청하고, 그들의 유틸리티 값(utility value)을 기반으로 메트릭을 분석하였다. 이를 통해 여러 이해관계자의 의견을 반영하는 메트릭 선택 방법과 평가 접근 방식을 제안하였다.

- **Technical Details**: 연구에서는 기계 학습 모델의 성능 및 공정성을 평가하기 위해 7개의 메트릭을 사용하였다. 참가자들은 두 가지 가상의 작업 매칭 시스템을 선택하는 과정을 반복하며, 이들의 메트릭에 대한 선호도를 클러스터(cluster)로 분석하였다. 또한, 참가자들의 인구통계학적 특성과 메트릭에 대한 선호의 관계를 조사하여, 이해관계자 그룹의 특성을 분석하였다.

- **Performance Highlights**: 이 연구는 기계 학습 모델을 평가할 때 이해관계자들의 다각적인 의견을 포함하는 중요성을 강조한다. 다양한 메트릭을 사용하여 참가자들의 선호도를 정확하게 포착하고, 이를 통해 공정한 평가 방안을 제안하였다. 결과적으로, 연구는 ML 모델 평가시 고려해야 할 이해관계자의 의견을 보다 공정하게 통합하는 방안을 제시하였다.



### CBW: Towards Dataset Ownership Verification for Speaker Verification via Clustering-based Backdoor Watermarking (https://arxiv.org/abs/2503.05794)
Comments:
          14 pages. The journal extension of our ICASSP'21 paper (arXiv:2010.11607)

- **What's New**: 본 논문에서는 스피커 검증에 대한 저작권 보호를 위한 새로운 방법인 클러스터링 기반 백도어 워터마크(CBW)를 제안합니다. 이는 불법으로 사용된 데이터셋을 검증할 수 있는 접근 방식을 제공하며, 스피커 검증 데이터셋의 저작권을 보호하는 것을 목표로 하고 있습니다. 기존의 데이터 보호 방법 (예: encryption, digital watermarking)이 갖는 한계를 극복하기 위한 노력의 일환으로 연구되었습니다.

- **Technical Details**: CBW 방법은 두 가지 주요 단계로 구성됩니다: 데이터셋 워터마킹과 소유권 검증입니다. 워터마킹 단계에서는 서로 비슷한 샘플들이 동일한 트리거 패턴에 가까워지도록 다수의 트리거 패턴을 데이터셋에 삽입합니다. 이 과정을 통해, 워터마킹된 데이터셋으로 훈련된 모델은 특정 잘못 분류 행동을 보이게 되고, 이를 통해 데이터셋의 소유권을 인증하게 됩니다.

- **Performance Highlights**: 광범위한 실험을 통해 CBW 방법의 효과성과 강인성을 검증하였으며, 악의적인 공격에 대한 저항성 또한 입증되었습니다. 결론적으로, 제안된 CBW 방법은 의심스러운 쓰임새를 가진 모델이 보호된 데이터셋을 통해 훈련되었는지를 통계적으로 평가하는 효율적인 방법을 제공하며, 저작권 보호를 위한 중요한 기여를 함을 보여줍니다.



### MedSimAI: Simulation and Formative Feedback Generation to Enhance Deliberate Practice in Medical Education (https://arxiv.org/abs/2503.05793)
- **What's New**: MedSimAI는 AI가 지원하는 시뮬레이션 플랫폼으로, 의사-환자 커뮤니케이션에 대한 임상 기술 훈련을 혁신적으로 변화시킵니다. 이 플랫폼은 대화형 환자 만남을 통해 고의적인 연습(deliberate practice)과 자기주도 학습(self-regulated learning, SRL)을 허용하며, 자동화된 평가를 제공합니다. MedSimAI는 대형 언어 모델(large language models, LLMs)을 활용하여 현실적인 임상 상호작용을 생성하고, 의료 평가 기준에 따라 구조화된 즉각적인 피드백을 제공합니다.

- **Technical Details**: MedSimAI는 전문가들 간의 협업을 통해 설계되었으며, 임상 기술의 기초 역량을 연습할 수 있는 AI 표준화 환자(AI standardized patients) 시뮬레이션 프레임워크를 포함합니다. 이 플랫폼은 즉각적인 피드백을 제공하기 위한 유연한 평가 프레임워크와 임상 체크리스트를 통해 교육적 요구를 충족할 수 있도록 설계되었습니다. 특히 SRL 원칙을 통합하여 학습자들이 설정한 목표에 따라 사용할 수 있는 다양한 임상 시나리오를 제공합니다.

- **Performance Highlights**: 파일럿 연구에서 104명의 1학년 의대생들이 MedSimAI를 통해 환자 이력 취득을 반복적으로 연습하는 데 도움을 받았습니다. 학생들은 이 플랫폼이 실질적이고 반복 가능한 훈련을 제공한다고 평가했으나, 일부 고차원 기술이 간과되는 경향을 보였습니다. 그러나 전반적으로 학생들은 체계적인 역사 취득과 공감적 경청에 높은 성과를 보여, MedSimAI가 전통적인 시뮬레이션 기반 교육의 주요 한계를 해결할 수 있는 가능성을 입증했습니다.



### Emergent Abilities in Large Language Models: A Survey (https://arxiv.org/abs/2503.05788)
- **What's New**: 이 논문은 인공지능 일반 지능(Artificial General Intelligence)을 향한 새로운 기술 혁명을 이끌고 있는 대형 언어 모델(Large Language Models, LLMs)의 급증에 대한 이해를 심화시키고자 한다. 최근 LLMs는 여러 가지 예상치 못한 'emergent abilities'를 나타내어 과학적 논쟁의 중심에 서게 되었다. 이러한 emergent abilities는 외부 요인, 훈련 동역학(training dynamics), 과제 유형 등에 따라 달라질 수 있을지에 대한 의문을 제기하며, 이 논문은 그 복잡성을 파악하기 위한 종합적인 검토를 제공한다.

- **Technical Details**: 논문은 emergent abilities라는 개념을 비판적으로 분석하고, 이를 바탕으로 LLMs 및 대형 추론 모델(Large Reasoning Models, LRMs)의 성능을 결정짓는 조건들을 조사한다. 이 연구에서는 스케일링 법칙(scaling laws), 과제 복잡성(task complexity), 사전 훈련 손실(pre-training loss) 등의 요소를 분석하여 emergent abilities가 나타나는 맥락을 규명한다. 또, 기존 LLMs의 한계를 넘어서기 위해 강화 학습(reinforcement learning)과 추론 시간 탐색(inference-time search)을 통합한 LRMs를 다룬다.

- **Performance Highlights**: LLMs의 성능은 훈련 데이터의 규모와 모델 파라미터 수에 따라 잘 정의된 관계가 있다. 그러나 특정 과제에서는 모델의 크기와 성능 사이에 비연속적인 관계가 나타나는 등, 예측 불가능한 성과도 존재한다. 이에 따라 emergent abilities는 시스템적 신뢰성과 안전성, 특히 유해한 행동의 예측을 보장하기 위한 핵심 요소로 부각되며, AI 시스템의 안전과 관리에 대한 우려가 커지고 있다.



### Artificial Intelligence in Sports: Insights from a Quantitative Survey among Sports Students in Germany about their Perceptions, Expectations, and Concerns regarding the Use of AI Tools (https://arxiv.org/abs/2503.05785)
Comments:
          36 Tables, 18 Figures

- **What's New**: 이 논문에서는 Generative Artificial Intelligence (AI) 도구들이 학술 연구와 교육에 미치는 영향을 조사한 최신 연구 결과를 소개합니다. 특히 독일의 스포츠 학생들을 대상으로 한 정량적 조사를 통해 AI 사용에 대한 학생들의 인식과 기대를 탐구하였습니다.

- **Technical Details**: 이 조사는 2023년 8월부터 11월 사이에 진행되었으며, 독일 대학의 모든 스포츠 학부에서 총 262명의 학생들이 참여하였습니다. 학생들의 AI 도구 사용 행동, 동기 요인, AI가 학문에 미치는 미래의 영향에 대한 불확실성을 분석하였습니다.

- **Performance Highlights**: 조사 결과, 학생들은 AI 도구를 통해 학업 성과를 향상시키고, 과학적 접근의 복잡성을 이해하며, 시간을 절약할 것으로 기대하고 있습니다. 또한, 학생들은 AI의 확산이 자신의 비판적 사고 능력을 해치지 않을 것이라고 확신하며, 교육 과정에 AI 관련 주제를 통합하는 것에 긍정적입니다. 하지만 표절, 강사 준비 상태 및 자신의 기술 개발에 대한 우려도 나타났습니다.



### The Illusion of Rights based AI Regulation (https://arxiv.org/abs/2503.05784)
- **What's New**: 이 논문은 AI 규제에 관한 현재의 논의를 본질적으로 잘못된 전제로 진행되고 있다고 주장합니다. 특히, 유럽연합(EU)의 AI 규제 프레임워크가 기본적으로 권리 중심적이라는 기존의 학문적 합의를 도전하고 있습니다. 이 논문은 EU 규제를 표면적으로 받아들이기보다는, 그 규제가 특정 문화적, 정치적, 역사적 맥락의 논리적 결과임을 보여줍니다.

- **Technical Details**: 논문에서는 GDPR(일반 데이터 보호 규정)과 AI법이 기본권의 언어를 인용하지만, 이러한 권리가 도구화(instrumentalization)되어 있다는 점을 강조합니다. 즉, 이것은 시스템적 위험을 해결하고 제도적 안정성을 유지하기 위한 거버넌스 도구의 수사적 맥락으로 사용된다는 것입니다. 저자들은 데이터 프라이버시, 사이버 보안, 의료, 노동, 잘못된 정보 등 다섯 가지 논란이 있는 분야에서의 AI 규제를 비교 분석합니다.

- **Performance Highlights**: EU와 미국 간의 비교 분석을 통해 EU의 규제 구조가 기본적으로 권리 중심적이지 않음을 보여줍니다. 저자들은 현재의 미국 규제 모델이 반드시 더 낫다고 주장하지 않으며, EU의 AI 규제 접근 방식의 전제된 정당성은 포기해야 한다고 결론짓습니다. 이러한 시점에서 논문은 AI 정책 논의에 중요한 기여를 합니다.



### Knowledge representation and scalable abstract reasoning for simulated democracy in Unity (https://arxiv.org/abs/2503.05783)
Comments:
          23 pages, 11 figures, 76 references. This article is under review at WSEAS Transactions on Information Science and Applications from 02.2025

- **What's New**: 이번 논문에서는 에이전트에 대한 새로운 형태의 확장 가능한 지식 표현인 e-polis를 소개합니다. 이 환경은 실제 사용자가 민주적 제도와 관련된 사회적 도전에 대응하는 시뮬레이션된 민주주의를 기반으로 하며, 방문자의 철학적 신념에 따라 건축 형태가 변화하는 Smart Spatial Type이 포함되어 있습니다. 게임이 끝나면 플레이어들은 그들의 집단적 선택에 기반하여 생성된 Smart City에 투표합니다.

- **Technical Details**: 이 접근 방식은 전통적인 방식과는 달리 민주주의 모델과 Smart City 모델을 통합하여 시뮬레이션된 민주주의의 품질 측면을 다양한 도시 및 사회 환경에서 증명할 수 있습니다. 또한, 사용자 정의된 추상 지식으로 추론과 추론이 가능해지는 점이 특징입니다. 시스템은 이벤트 처리를 통해 현재 게임 상태를 지속적으로 파악하는 하위 계층과 사용자 정의 추상 지식을 효율적으로 검색하는 상위 계층으로 구성되어 있습니다.

- **Performance Highlights**: 게임 흐름의 실시간 의사 결정 및 플레이어의 추상적 상태에 따른 적응이 가능하여 설명 가능성을 높였습니다. 또한, 이중 계층 지식 표현 메커니즘을 통해 확장성을 달성하며, 이는 2단계 캐시와 유사하게 기능합니다. Unity 플랫폼의 내장 물리 엔진에 의해 생성된 높은 이벤트 처리 속도를 통해, 플레이어의 위치와 각 도전에 대한 선택을 실시간으로 반영할 수 있습니다.



### AI Mentors for Student Projects: Spotting Early Issues in Computer Science Proposals (https://arxiv.org/abs/2503.05782)
Comments:
          Accepted for oral presentation at Workshop on Innovation and Responsibility in AI-Supported Education (iRAISE), AAAI 2025

- **What's New**: 이 논문은 프로젝트 기반 학습(Project-based learning, PBL)의 효과를 높이기 위해 학생의 목표와 기술 수준을 평가할 수 있는 소프트웨어 시스템의 설계를 소개합니다. 초기 사용자 연구에서 36명의 사용자가 참여하여 시스템이 프로젝트 제안서를 작성하는 데 유용했음을 보여주었으며, 사용자의 기술 경험에 따라 프로젝트 제안서의 품질이 달라짐을 발견했습니다.

- **Technical Details**: 본 시스템은 React.js 및 Firebase를 기반으로 구축되어 PBL에 참여할 준비가 된 학생을 식별하기 위해 프로젝트 제안 및 적합성 정보를 수집합니다. 사용자에게 문제 해결 능력, 컴퓨터 과학 기술 경험, PBL 경험을 질문하고, 고등학교 수준의 컴퓨터 과학 프로젝트에 대한 제안을 작성하도록 유도합니다. 제안서를 평가하기 위해 23개 항목의 품질 체크리스트가 사용되며, GPT-4o와 두 명의 인간 평가자가 제안서를 독립적으로 평가합니다.

- **Performance Highlights**: 우리의 연구 결과는 LLM들이 프로젝트 제안서의 품질을 자동으로 평가하는 데 유망하다는 것을 보여주었습니다. 사용자는 88.8%가 향후에도 시스템을 사용하고 싶다고 답했으며, 91.6%는 자신이 배우고 싶은 기술과 동기 부여가 되는 프로젝트 아이디어를 설계하는 데 해당 시스템을 활용하기를 원한다고 응답했습니다. 그러나 앞으로는 학생의 성공과 학습 동기를 신뢰성 있게 예측할 수 있는 지표를 특성화하는 노력이 필요합니다.



### Homomorphic Encryption of Intuitionistic Logic Proofs and Functional Programs: A Categorical Approach Inspired by Composite-Order Bilinear Groups (https://arxiv.org/abs/2503.05779)
- **What's New**: 이번 논문은 동형 암호(homomorphic encryption)의 개념적 프레임워크를 제시하여 산술 및 불(Boolean) 연산을 넘어 직관 논리(intuitionistic logic) 증명 및 유형화 함수형 프로그램(typed functional programs) 영역까지 확장합니다. 이전 연구들에 비해, 직관 논리 및 관련 증명에 대한 동형 암호화 처리 방식이 제안되며, 이론적으로는 폴리노미얼 펑터(polynomial functors)와 바운디드 내추럴 펑터(Bounded Natural Functors, BNF)를 기반으로 논리 구조를 다룹니다.

- **Technical Details**: 소개 부분에서는 동형 암호화가 암호화된 데이터에 대한 계산을 가능하게 하며, 주로 산술 및 불 연산에 초점을 맞추었음을 설명합니다. 이어서 Tseng 외에(2017)의 동형 암호화 방식이 논리 연산을 위해 어떻게 설계되었는지를 소개하며, 이는 주어진 비트의 논리적 동형성을 유지하는 방법입니다. 그리고 이 논문은 동형 암호화가 직관 논리 증명 및 유형화 프로그램에서 어떻게 적용될 수 있는지를 규명합니다.

- **Performance Highlights**: 최종적으로, 이 논문은 동형 암호화 접근법의 효율성을 높일 수 있는 전략, 즉 소프트웨어 최적화 및 하드웨어 가속을 통해 실제 환경에서의 적용 가능성을 탐색합니다. 또한, 복잡성 이론적 난이도 가정을 제시하고, 이를 통해 암호학적 보안의 기초를 마련합니다. 이러한 모든 요소가 결합되어, 기능적 프로그램의 동형 암호화를 더욱 발전시키는 이정표를 세웁니다.



### DreamNet: A Multimodal Framework for Semantic and Emotional Analysis of Sleep Narratives (https://arxiv.org/abs/2503.05778)
Comments:
          10 pages, 5 figures, new research contribution

- **What's New**: 이 논문은 꿈 내러티브(dream narratives)의 체계적인 분석을 위해 인공 지능을 사용한 연구가 부족했던 점을 해결하기 위해 새로운 딥러닝 프레임워크인 DreamNet을 도입합니다. DreamNet은 텍스트 기반의 꿈 보고서에서 의미적 주제와 감정 상태를 추출하며, REM 단계의 EEG 데이터를 통해 능력을 보강할 수 있습니다. 특히, 이 모델은 1500개의 익명 꿈 내러티브로 구성된 데이터셋에서 텍스트 전용 모드에서 92.1%의 정확도와 88.4%의 F1-score를 달성하며, EEG 통합 시에는 99.0%의 정확도와 95.2%의 F1-score로 성능이 향상됩니다.

- **Technical Details**: DreamNet은 RoBERTa를 기반으로 한 transformer 모델로, 꿈 내러티브에서 비행, falling, 추적, 상실과 같은 의미적 주제를 추출합니다. 이 모델은 감정 상태를 나타내는 8개의 이진 벡터를 사용하여 다양한 감정(예: 두려움, 기쁨, 불안, 슬픔 등)을 분류합니다. 또한, DreamNet은 REM 단계의 EEG 데이터를 통합하여 생리적 데이터를 포함한 다중 모드 분석을 구현하며, 이는 텍스트 기반 데이터와 생리적 맥락을 모두 포착할 수 있게 합니다.

- **Performance Highlights**: DreamNet은 텍스트 전용 모드(DNet-T)에서 92.1%의 정확도와 88.4%의 F1-score를 기록했으며, REM 단계의 EEG 데이터 통합 후에는 99.0%의 정확도와 95.2%의 F1-score로 7%의 성능 향상을 보여줍니다. 연구 결과는 정신 건강 진단 및 개인 맞춤형 치료에 활용될 수 있는 잠재력을 가지고 있으며, 꿈과 감정 간의 강한 상관관계(예: falling-anxiety, r = 0.91, p < 0.01)를 보여줍니다. 이 작업은 AI와 심리 연구 간의 교량 역할을 하며, 스케일러블한 도구와 공개 데이터셋을 제공합니다.



### Medical Hallucinations in Foundation Models and Their Impact on Healthcar (https://arxiv.org/abs/2503.05777)
- **What's New**: 이번 논문은 의료 분야에서의 Foundation Models의 다중 모드(multi-modal) 데이터 처리 및 생성 능력에 대해 다루고 있으며, 특히 의료적 환각(medical hallucination)에 대한 문제점을 강조합니다. 이러한 환각은 잘못된 정보가 생성되어 임상 결정 및 환자 안전에 영향을 줄 수 있습니다. 연구진은 의료 환각의 정의, 특성 및 임상 시나리오에서의 실질적 영향을 탐구합니다.

- **Technical Details**: 논문에서는 의료 환각을 이해하고 해결하기 위한 분류법(taxonomy)을 제시하고, 의료 환각 데이터셋을 사용하여 모델을 벤치마킹(benchmarking)합니다. 또한, 임상 사례에 대한 의사 주석(physician-annotated) LLM 응답을 비교하여 환각이 임상에 미치는 직접적인 통찰을 제공합니다. Chain-of-Thought(CoT) 및 Search Augmented Generation과 같은 추론(inference) 기법이 환각 비율을 감소시키는 데 효과적임을 보여주지만, 여전히 비주얼적인 환각이 존재합니다.

- **Performance Highlights**: 연구 결과에 따르면 의료 환각의 비율을 낮추기 위한 여러 접근 방식이 제시되었음에도 불구하고 상당한 수준의 환각이 여전히 지속되고 있음을 알 수 있습니다. 이는 AI가 의료 분야에 통합됨에 따라 환자 안전과 임상 무결성을 유지하기 위한 규제 정책의 필요성을 강조합니다. 의료진의 피드백을 통해 기술적 발전뿐만 아니라 윤리적 및 규제적 지침이 필요하다는 점도 강조되었습니다.



### FAA-CLIP: Federated Adversarial Adaptation of CLIP (https://arxiv.org/abs/2503.05776)
Comments:
          Accepted in IEEE Internet of Things Journal

- **What's New**: 이 논문에서는 Federated Adversarial Adaptation (FAA)라는 새로운 방법을 제안하여 CLIP 모델의 커뮤니케이션 비용을 줄이면서 두 가지 주요 문제를 해결합니다. FAA-CLIP은 클라이언트의 데이터에 효과적으로 적응하기 위해 경량의 feature adaptation module (FAM)을 사용합니다. 기존 방법과는 달리, 이 모델은 domain adaptation (DA) 모듈을 통해 클라이언트 간의 도메인 변화 문제를 직접적으로 다룹니다.

- **Technical Details**: FAA-CLIP 방법은 클립(클로스 상의 언어 이미지 전이) 모델의 매개 변수를 고정하고 FAM의 매개 변수만 업데이트하여 계산 효율성을 높입니다. 이 시스템은 도메인 판별기를 사용하여 특정 샘플이 로컬 클라이언트에서 왔는지 또는 글로벌 서버에서 왔는지를 예측하여 도메인 불변 표현(domain-invariant representations)을 학습할 수 있도록 합니다. 이러한 방식으로, FAA-CLIP은 다양한 클라이언트의 데이터 이질성 문제를 해결합니다.

- **Performance Highlights**: 여섯 개의 다양한 데이터셋에서 이루어진 실험을 통해, FAA-CLIP은 최근의 FL 접근법들에 비해 자연 및 의료 데이터셋 모두에서 뛰어난 일반화 성능을 보였습니다. 특히, FAA-CLIP은 더 높은 분류 정확도와 균형 잡힌 정확도를 달성하여, 최신 FL 기술들에 비해 성능이 우수함을 입증했습니다. 이로써, FAA-CLIP은 의료 이미징 및 자연 이미지 분류 작업에서 더욱 효과적인 솔루션이 될 가능성을 제시합니다.



### Between Innovation and Oversight: A Cross-Regional Study of AI Risk Management Frameworks in the EU, U.S., UK, and China (https://arxiv.org/abs/2503.05773)
- **What's New**: 이 논문은 인공지능(AI) 기술의 윤리적, 보안 및 사회적 위험을 관리하기 위한 효율적인 거버넌스 프레임워크의 필요성을 강조합니다. 특히 유럽연합(EU), 미국(U.S.), 영국(UK), 중국의 AI 위험 관리 전략을 비교 분석하여 각 지역의 규제 모델의 장단점을 보여주고 있습니다. 논문에서는 AI의 윤리적 배치와 위험 완화 전략의 강점과 약점을 평가하고, 이러한 프레임워크가 다양한 지역과 분야에 적합하도록 어떻게 발전할 수 있는지를 모색합니다.

- **Technical Details**: 다양한 AI 거버넌스 접근 방식을 비교하기 위해 저자는 비교 정책 분석, 주제 분석 및 사례 연구를 포함한 다중 방법 질적 접근 방식을 사용했습니다. EU는 위험 기반 구조를 통해 투명성과 적합성 평가를 우선시하는 반면, 미국은 분산된 분야별 규제를 통해 혁신을 촉진하는 구조를 갖추고 있습니다. 영국은 유연한 분야별 규제를 통해 기민한 대응을 가능하게 하지만, 각 도메인 간 일관성이 결여될 수 있는 위험이 있습니다.

- **Performance Highlights**: 논문은 AI 규제가 기술 발전과 효과적인 위험 관리의 균형을 이루어야 한다고 강조하며, 글로벌 표준을 설정하는 데 있어 EU의 AI 법안이 중요한 역할을 할 것으로 보입니다. 미국의 분산형 규제는 혁신을 촉진하지만 일관된 집행의 결여를 초래할 수 있다는 점에서 여전히 문제가 됩니다. 마지막으로 중국은 중앙집중식 지침을 통해 신속한 대규모 구현을 가능하게 하지만, 공공 투명성과 외부 감독이 제한된다는 점에서 비판을 받고 있습니다.



### Generative Artificial Intelligence: Evolving Technology, Growing Societal Impact, and Opportunities for Information Systems Research (https://arxiv.org/abs/2503.05770)
- **What's New**: 이 논문은 생성 인공지능(GenAI)의 급속한 발전이 비즈니스와 사회에 미칠 잠재적 영향에 대한 논의를 다룹니다. GenAI는 대규모 언어 모델 및 관련 알고리즘을 기반으로 하며, 이러한 기술이 기존의 인공지능 기술과는 다르게 어떻게 혁신적인 변화를 가져올 수 있는지 분석합니다. 연구자들이 정보 시스템(IS) 분야에서 이 기술에 어떻게 반응해야 하는지에 대한 방향성을 제시합니다.

- **Technical Details**: 저자들은 GenAI의 독특한 특성을 탐구하며, 상징주의에서 연결주의로의 지속적인 변화와 인간-AI 생태계의 심층적인 시스템 속성을 강조합니다. 또한, 그들은 정보 시스템 연구의 맥락에서 비즈니스와 사회에서 GenAI의 다양한 영향에 대한 미래 연구를 제안합니다. 이 과정에서 기술 및 조직 커뮤니티 간의 간극을 줄이기 위한 노력도 강조됩니다.

- **Performance Highlights**: 이 논문의 목표는 IS 커뮤니티 내에서 GenAI의 혁신적인 전략과 운영을 지원하기 위한 잘 구성된 연구 아이디어를 만드는 것입니다. 저자들은 기존의 연구들 중 대부분이 너무 기술적이거나 GenAI의 잠재적 영향에 대한 심층적인 이해가 부족하다는 점을 지적합니다. 따라서 이 논문은 IS 연구자들이 GenAI의 이점을 활용할 수 있도록 필요한 통찰을 제공하고자 합니다.



### Effect of Gender Fair Job Description on Generative AI Images (https://arxiv.org/abs/2503.05769)
Comments:
          ISCA/ITG Workshop on Diversity in Large Speech and Language Models

- **What's New**: 이 연구는 OpenAI DALL-E 3와 Black Forest FLUX를 사용하여 STEM 직업 이미지에서 성별 표현을 분석하였습니다. 150개의 프롬프트를 사용하여 3개의 언어 형태(독일어 일반 남성형, 독일어 쌍형, 그리고 영어)로 생성된 이미지를 분석한 결과, 모든 형태에서 남성 편향이 두드러졌습니다. 연구 결과는 생성적 인공지능이 사회적 편견을 강화하는 데 기여하고 있다는 점을 강조하며, 다각적인 다양성 논의의 필요성을 시사합니다.

- **Technical Details**: 이 연구에서는 DALL-E와 Black Forest Flux의 두 가지 이미지 생성기를 사용하여 50개의 STEM 직업을 포함한 210개의 이미지를 생성하였습니다. 각 직업에 대해 150개의 STEM 프롬프트와 60개의 사회적 직업에 대한 프롬프트가 사용되었으며, 모든 직업은 세 가지 언어 범주에서 균등하게 표현되었습니다. 개별 이미지의 성별 식별은 세 명의 저자가 독립적으로 수행하였으며, DALL-E와 FLUX의 각 생성 결과를 비교하였습니다.

- **Performance Highlights**: 연구 결과, DALL-E와 FLUX가 생성한 이미지는 성별 및 인종적 다양성이 낮으며, 특히 STEM 분야에서 남성을 과대표현하고 있습니다. 또한, 생성된 이미지의 성별 구별 기준이 대단히 보수적이어서, 성별 확인이 용이하다는 점이 강조되었습니다. 이러한 결과는 생성적 인공지능의 편견을 해결하기 위한 포괄적인 접근이 필요함을 보여줍니다.



### A Collection of Innovations in Medical AI for patient records in 2024 (https://arxiv.org/abs/2503.05768)
- **What's New**: 본 논문은 AI가 의료 분야에서 빠르게 발전하고 있으며, 특히 대형 언어 모델의 혁신이 임상 의사결정과 환자 치료를 혁신할 수 있는 잠재력을 가지고 있음을 강조합니다. 전통적인 학술 출판 주기가 AI 발전 속도를 따라잡지 못하고 있다는 문제를 제기하며, 연례 인용 프레임워크를 제안하여 최신 혁신을 반영하는 새로운 형태의 학술 출판물이 필요함을 주장합니다. 이는 AI 연구의 적시성을 높이고 민첩한 학술 생태계를 조성하는 데 기여할 것입니다.

- **Technical Details**: 2024년에는 생물의학 자연어 처리(NLP) 분야에서 건강 및 생물의학 응용을 위해 특수화된 여러 언어 모델들이 개발되었습니다. MediSwift와 같은 모델은 생물의학 텍스트 데이터에서 희소 사전 학습을 수행하여 훈련 효율을 크게 향상시키며, BMRetriever는 대규모 생물의학 말뭉치를 기반으로 한 비지도 사전 학습을 활용하여 다양한 응용에서 검색 작업을 향상시킵니다. 또한, BioMedLM과 BioMistral은 생물의료 데이터에 특화된 모델로, 특정 NLP 응용 프로그램을 위한 투명하고 경제적인 기초를 제공합니다.

- **Performance Highlights**: EHR(전자 건강 기록) 분야에서도 기초 모델들이 발전하고 있어 의료 데이터의 복잡성을 효과적으로 포착하고 임상 의사결정을 지원합니다. EHRMamba는 최대 4배 긴 환자 히스토리를 처리할 수 있는 확장 가능한 아키텍처를 제공하며, 다양한 임상 예측 작업에서 우수한 성능을 보입니다. 또한, MOTOR와 CEHR-GPT는 각각 시간 예측과 임상 문서 자동화 분야에서 중요한 역할을 하여, 향후 환자 관리와 문서화 작업의 효율성을 높이는 데 기여할 수 있을 것으로 기대됩니다.



### Mesterséges Intelligencia Kutatások Magyarországon (https://arxiv.org/abs/2503.05767)
Comments:
          in Hungarian language. Submitted to Magyar Tudomány

- **What's New**: 이 논문은 헝가리의 인공지능(AI) 연구의 주요 성과를 다루고 있으며, 특히 딥러닝(deep learning) 이전의 성과와 2010년 이후의 주요 이론적 진전을 강조합니다. 헝가리 연구자들은 인공지능의 중요성을 조기에 인식하고 국제 연구에 활발히 참여하여 이론과 실용 분야 모두에서 중요한 결과를 낳았습니다. 다수의 연구가 헝가리의 AI 연구 및 산업 발전에 관한 개요를 제공하는 데 중점을 두고 있습니다.

- **Technical Details**: AI는 2000년대 중반부터 기계 학습(machine learning)과 딥러닝의 발전으로 비약적으로 발전해 왔으며, 이는 대형 데이터베이스와 계산 능력의 폭발적 증가에 의해 촉진되었습니다. 인터넷은 방대한 양의 데이터를 제공하였고 GPU(그래픽 처리 장치)의 발전은 복잡한 신경망(Neural Network) 교육을 가능하게 하였습니다. 논문은 AI의 역사와 함께 헝가리에서의 주요 연구들이 어떤 조직적 틀 속에서 자리잡았는지를 설명합니다.

- **Performance Highlights**: 헝가리는 1970년대부터 AI 연구에 착수하였으며, 특히 2010년대 초기에 딥러닝의 중요성이 대두된 이후 연구가 폭발적으로 발전하였습니다. 연구자들이 발표한 주요 성과는 헝가리의 컴퓨터 과학적 발전 방향에 기여했으며, 그 중 일부는 국제적으로도 중요한 의미를 지닙니다. AI의 관련 이론과 응용 결과를 체계적으로 분석하고, 이러한 연구들이 헝가리의 AI 분야의 성장을 어떻게 이끌었는지에 대한 인사이트를 제공합니다.



### Encoding Inequity: Examining Demographic Bias in LLM-Driven Robot Caregiving (https://arxiv.org/abs/2503.05765)
Comments:
          Accepted at the 4th Diversity, Equity, & Inclusion in HRI Workshop at HRI'25, the 20th edition of the ACM/IEEE International Conference on Human-Robot Interaction

- **What's New**: 이 논문은 로봇이 다양한 인구 집단과 상호작용할 때 공정성과 비편향적인 상호작용을 보장하는 것이 얼마나 중요한지를 강조합니다. Large Language Models (LLMs)는 로봇의 행동, 발언 및 의사결정에 중요한 역할을 하지만, 이들 모델은 사회적 편견을 내재화하고 전파할 수 있습니다. 본 연구는 LLM이 생성한 응답이 성별, 인종, 장애, 연령 등 다양한 인구적 요소에 따라 로봇의 돌봄 특성과 책임을 어떻게 형성하는지를 분석합니다.

- **Technical Details**: 연구에서 ChatGPT 4o를 사용하여 138개의 서로 다른 레이블을 가진 돌봄 로봇 상황을 생성했습니다. 생성된 내용의 언어적, 구조적, 의미적 속성을 분석하기 위해 체계적인 입력 형식을 사용했고, 다양한 분석 기법을 적용했습니다. 예를 들어, 포함된 모든 텍스트에 대해 언어적 및 계산적 분석을 수행하고, 텍스트 내 군집화를 통해 인구 집단 간의 편견 패턴을 파악했습니다.

- **Performance Highlights**: 분석 결과, 장애 및 성소수자(LGBTQ+)와 관련된 레이블의 경우 감정 점수가 낮을 뿐 아니라, 간단한 문장 구조로 설명되었음을 보여주었습니다. 연령에 대한 레이블은 LLM 생성 텍스트의 전체적 유사성 측면에서 가장 높은 점수를 기록한 반면, 인종 및 민족성 관련 레이블은 가장 낮은 유사성을 보였습니다. 이 결과는 돌봄 로봇의 설명에서 특정 인구 집단에 대한 편견이나 불균형이 드러난 것을 보여줍니다.



### Graph Masked Language Models (https://arxiv.org/abs/2503.05763)
- **What's New**: 이 논문에서는 언어 모델(LMs)과 구조화된 지식 그래프(KGs) 간의 상호작용을 개선하기 위해 새로운 접근 방식을 제안합니다. 특히, 그래프 마스킹 언어 모델(GMLM)이 노드 분류 작업을 위해 도입되며, 이는 의미적 마스킹 전략과 소프트 마스킹 메커니즘을 통해 특징적인 그래프 정보를 효과적으로 활용할 수 있도록 설계되었습니다. 이러한 접근은 노드의 구조적 중요성을 고려하여 중요한 그래프 구성 요소가 학습에 실질적으로 기여할 수 있도록 합니다.

- **Technical Details**: 제안된 GMLM은 이중 가지 모델 아키텍처를 통해 구조 그래프 정보와 컨텍스트 임베딩을 결합합니다. 이 모델은 정량적인 정보 흐름과 훈련 과정 중 세부 정보 보존을 가능하게 하는 소프트 마스킹 메커니즘을 통해 노드 표현을 생성합니다. 연구는 또한 노드의 중요성을 반영한 의미적 마스킹 전략을 사용하여, 훈련 동안 신뢰성과 안정성을 강조합니다.

- **Performance Highlights**: 여섯 개의 노드 분류 베치마크에서 진행된 실험에 따르면, GMLM은 최첨단(SOTA) 성능을 달성하며 다양한 데이터셋에서 강력한 안정성과 견고함을 보여줍니다. 이러한 결과는 GMLM의 적용이 기존 GNN 아키텍처의 한계를 극복하고 노드 표현 학습에서의 효과성을 극대화할 수 있음을 시사합니다.



### The Lazy Student's Dream: ChatGPT Passing an Engineering Course on Its Own (https://arxiv.org/abs/2503.05760)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)인 ChatGPT가 전체 학부제어 시스템 과정에서 얼마나 잘 수행할 수 있는지를 평가합니다. 115개의 강의 과제를 분석하여, LLM이 복잡한 프로그래밍 및 분석적 작문 업무에서 어떻게 작동하는지를 심층적으로 조사했습니다. 이러한 연구는 AI의 진화에 대응하여 교육 과정의 설계를 어떻게 변경해야 하는지를 논의하는 데 기여합니다.

- **Technical Details**: 이 연구는 제어 시스템 공학의 다양한 과제를 다루면서 LLM의 수학적 공식화, 코딩 난이도, 이론적 개념 처리의 능력과 한계를 정량적으로 평가합니다. 명시적 전문가의 개입 없이 LLM 성능을 평가하기 위해, 강의 내용이 다양하게 포함된 115개의 과제를 사용했습니다. 연구 결과, LLM은 구조화된 과제에서 강력한 성과를 보였으나, 개방형 프로젝트에서는 제한된 성능을 보였습니다.

- **Performance Highlights**: LLM은 전체 성적이 B학점(82.24%)을 기록하여, 수업 평균인 84.99%에 근접했습니다. 이 연구는 특히 공학 교육에 있어 AI 도구의 통합 필요성을 강조하며, LLM의 현재 성능을 바탕으로 향후 교육 과정 설계에 대한 통찰력을 제공합니다.



### ADAPT Centre Contribution on Implementation of the EU AI Act and Fundamental Right Protection (https://arxiv.org/abs/2503.05758)
- **What's New**: 이번 문서는 ADAPT 센터가 아일랜드 기업, 무역 및 고용부(DETE)에 제출한 내용을 담고 있습니다. 주제는 EU AI 법안(Act) 시행에 관한 공공 의견 수렴(public consultation)입니다. 이 법안은 인공지능 기술의 유럽 내 규제 및 관리 방안을 제시하는 중요한 문서입니다.

- **Technical Details**: 문서에서는 AI 시스템의 위험 평가(risk assessment) 및 그에 대한 규제적 접근 방식을 논의합니다. 다양한 인공지능 기술의 분류(classification) 및 이를 통해 생성된 데이터 보호(data protection) 이슈도 다루어집니다. 이러한 기술적 세부 사항은 기업들이 법률을 준수하는 데 필수적인 요소입니다.

- **Performance Highlights**: ADAPT 센터는 EU AI 법안의 시행으로 인한 잠재적인 경제적 효과와 함께 기술 혁신(innovation) 촉진에도 기여할 수 있다고 강조합니다. 보고된 내용은 아일랜드 정부가 AI 규제를 개선하고 기술 생태계를 강화하는 데 있어 중요한 기초 자료가 될 것입니다.



### Uncertainty-Aware Fusion: An Ensemble Framework for Mitigating Hallucinations in Large Language Models (https://arxiv.org/abs/2503.05757)
Comments:
          Proceedings of the ACM Web Conference 2025, WWW 25

- **What's New**: 이 연구에서는 LLM(대형 언어 모델)의 허위 출력(hallucination) 문제를 해결하기 위한 새로운 방법으로 Uncertainty-Aware Fusion (UAF)이라는 앙상블 프레임워크를 제안합니다. 최근 연구에 따르면, LLM은 허위 출력을 생성할 가능성을 자체적으로 평가할 수 있는 불확실성 추정(uncertainty estimation)이 가능합니다. UAF는 다양한 LLM의 정확도와 자기 평가 기능을 활용하여 효과적으로 허위 출력을 줄이는 것을 목표로 합니다.

- **Technical Details**: UAF는 두 가지 주요 모듈로 구성됩니다: SELECTOR와 FUSER입니다. SELECTOR는 주어진 성능 메트릭에 따라 N개의 LLM 풀에서 K개의 LLM을 선택하고, FUSER는 선택된 LLM의 출력을 결합하여 최종 결과를 생성합니다. 이러한 구조는 추가적인 학습이나 미세 조정 없이 여러 LLM의 보완적인 강점을 활용하여 총체적인 정확도를 향상시킵니다.

- **Performance Highlights**: UAF는 TruthfulQA, TriviaQA 및 FACTOR와 같은 여러 공개 벤치마크 데이터세트에서 성능이 입증되었습니다. UAF는 기존의 최첨단(상태의 최첨단) 허위 출력 완화 방법보다 8% 향상된 사실 정확도를 보여주었으며, GPT-4와의 성능 차이를 좁히거나 심지어 초과하는 결과를 나타내었습니다.



### SEAFL: Enhancing Efficiency in Semi-Asynchronous Federated Learning through Adaptive Aggregation and Selective Training (https://arxiv.org/abs/2503.05755)
- **What's New**: 본 논문에서는 SEAFL이라는 새로운 Federated Learning (FL) 프레임워크를 제안하며, 이는 반비동기 반응향 FL의 느린 모델과 오래된 모델 문제를 해결하기 위해 고안되었습니다. SEAFL은 업로드된 모델의 가중치를 동적으로 할당하여, 현재 글로벌 모델에 대한 중요성과 진부함을 기반으로 한 집계 과정에서 효과적인 협업 학습을 가능하게 합니다. 이 방법은 여러 장치의 기여도를 최적 균형으로 조절하며, 느린 장치가 훈련의 효율성을 높이는 동시에 집계 오버헤드를 줄이도록 설계되었습니다.

- **Technical Details**: SEAFL은 장치의 기여도가 업데이트의 노후화와 현재 글로벌 모델과의 유사성을 함께 고려하여 가중치를 동적으로 조정하는 적응형 가중치 집계 메커니즘을 두고 있습니다. 이로 인해, SEAFL은 비동기기 방식의 FL에서 발생할 수 있는 효율성과 정확도 저하 문제를 해결할 수 있습니다. 또한, SEAFL2라는 변형을 도입하여 느린 장치가 부분 훈련을 수행할 수 있게 하여, 글로벌 집계 과정에 기여할 수 있도록 하여 과도한 대기 시간을 줄여줍니다.

- **Performance Highlights**: SEAFL은 세 개의 벤치마크 데이터셋에 대한 광범위한 실험을 통해 기존의 최첨단 FL 방법들과 비교하였으며, 목표 정확도를 달성하는 데 필요한 훈련 시간을 최대 22%까지 단축시킬 수 있음을 입증하였습니다. 특히 SEAFL은 FedBuff라는 가장 유사한 알고리즘 대비 현저히 뛰어난 성능을 보인 것으로 나타났습니다. 이를 통해 SEAFL은 글로벌 모델의 수렴을 가속화하고 쾌속 훈련이 가능하게 하는 중요한 기여를 하고 있습니다.



### Exploring AI Writers: Technology, Impact, and Future Prospects (https://arxiv.org/abs/2503.05753)
- **What's New**: 이번 연구에서는 AI 작가의 실제 능력과 다양한 창의적 분야에서의 응용을 탐구하고 있습니다. AI가 생성한 콘텐츠가 전통적인 미디어 산업과 학술적 글쓰기 프로세스에 미치는 잠재적 영향을 분석합니다. AI 도구가 금융, 스포츠 및 자연 재해와 같은 분야에서 뉴스 생산 워크플로우를 어떻게 변화시키고 있는지를 살펴보며, AI에 의해 발생하는 저작권 및 윤리적 문제도 조명합니다.

- **Technical Details**: AI 작가는 자연어 처리(Natural Language Processing) 기술을 활용하여 일관되고 논리적인 텍스트를 생성하는 고급 도구입니다. 이는 저널리즘, 광고, 교육 자료와 같은 다양한 분야에 적용됩니다. 특히, 사용자의 인지적 요구를 연구하고, AI 작문 도우미와의 상호작용을 통해 AI 작문 편집기 인터페이스의 미래 개발에 대한 이론적 지침을 제시합니다.

- **Performance Highlights**: AI 지원 글쓰기 피드백이 제2언어 학습자의 글쓰기 성과와 자기 효능감, 자기 조절에 미치는 영향을 강조하며, 외국어 교육 분야의 개인화, 지능화 및 포괄적 AI 기술 통합 경향을 보여줍니다. 연구 결과, 미디어 학생들은 AI가 언론 직업에 미치는 영향에 대해 조심스럽지만 희망적인 시각을 가지고 있으며, 직업적인 압박과 큰 효율성 향상을 동시에 기대하고 있습니다.



### CSTRL: Context-Driven Sequential Transfer Learning for Abstractive Radiology Report Summarization (https://arxiv.org/abs/2503.05750)
Comments:
          11-pages main paper with 2-pages appendices

- **What's New**: 이 논문에서는 방사선 보고서에서 Findings에서 Impression을 자동으로 생성하는 방법을 제시합니다. 기계 학습을 활용하여 임상 맥락을 유지하면서 핵심 정보를 추출하는 Sequential Transfer Learning 접근법을 도입했습니다. 이 방법은 Fisher matrix 정규화를 통해 지식 손실 문제를 해결하고, 방사선 전문 용어의 복잡성을 다룹니다.

- **Technical Details**: 논문에서 제안된 모델은 두 단계로 구성된 Sequential Transfer Learning 방식을 사용하여 T5 모델을 미세 조정합니다. 첫 단계에서는 GSG(Gap Sentence Generation) 작업을 통해 중요한 문장을 학습하고, 두 번째 단계에서는 임상 요약을 위해 GSG를 통한 학습된 가중치를 활용합니다. 이를 통해 지식 전이를 효과적으로 수행하고 차원 축소를 이룰 수 있습니다.

- **Performance Highlights**: CSTRL-Context-driven Sequential Transfer Learning 모델은 MIMIC-CXR 및 Open-I 데이터셋에서 기존 연구에 비해 BLEU 및 ROUGE 점수에서 현저한 향상을 보였습니다. BLEU-1에서 56.2%, BLEU-2에서 40.5%, ROUGE-1에서 28.9% 등의 성과를 기록하며, 일본어 임상 맥락을 유지하는 데 있어 사실적 일관성 점수 또한 분석되었습니다.



### Alignment, Agency and Autonomy in Frontier AI: A Systems Engineering Perspectiv (https://arxiv.org/abs/2503.05748)
- **What's New**: 인공지능(AI)의 발전과 확산에 따라, alignment(정렬), agency(대리성), autonomy(자율성) 개념이 AI 안전, 관리 및 통제의 핵심으로 부각되고 있습니다. 그러나 이 용어들은 철학, 심리학, 법학, 컴퓨터 과학 등 여러 분야에서 보편적 정의가 부족해, AI 시스템 설계 및 규제에 대한 상충된 접근을 초래합니다. 이 논문은 이러한 개념들의 역사적, 철학적, 기술적 진화를 추적하고, 정의가 AI 발전과 감독에 미치는 영향을 강조합니다.

- **Technical Details**: 이 논문은 AI 시스템의 alignment, agency 및 autonomy의 역사적, 철학적, 기술적 기초를 분석하며, 인지 과학, 정치 이론, 제어 시스템 공학 등 다양한 학문에서 통찰을 얻습니다. 특히 alignment의 진화를 살펴보며, AI 시스템의 목표를 공식화하는 과정에서 발생할 수 있는 취약성을 설명합니다. Agency는 AI 시스템이 독립적인 목표 형성과 장기적인 추론을 수행할 수 있는 능력으로 정의하고, autonomy는 시스템이 인간 개입 없이 작동할 수 있는 정도를 다룹니다.

- **Performance Highlights**: 이 논문은 Agentic AI를 사례로 들어 기계의 agency와 autonomy의 Emergent properties를 검토하고, 트렌드 AI의 Governance와 안전성 과제들을 평가합니다. 자동화 실패(Tesla Autopilot, Boeing 737 MAX), 다중 에이전트 협업(Meta의 CICERO), 진화하는 AI 아키텍처(DeepMind의 AlphaZero, OpenAI의 AutoGPT) 사례를 통해 실제 시스템 내에서의 alignment의 위험을 강조합니다. 이러한 연구들은 AI 시스템이 직면한 복잡한 문제를 해소하기 위한 정형화된 접근 방안을 제안합니다.



### Balancing Innovation and Integrity: AI Integration in Liberal Arts College Administration (https://arxiv.org/abs/2503.05747)
Comments:
          Number of Pages: 19; Number of Figures: 3. This submission explores AI integration in liberal arts college administration, focusing on academic and student affairs. It addresses ethical, legal, and institutional alignment issues. For related discussions, see: Friedler et al. (2016), Katsamakas et al. (2024), Łodzikowski et al. (2023), Zhang et al. (2024)

- **What's New**: 이 논문은 인공지능(AI)이 고등 교육 행정, 특히 인문계 대학(LAC)에서 직면하는 기회와 도전을 탐구합니다. AI의 적용은 학문 및 학생 관리, 법적 준수 및 인증 과정에서의 윤리적 고려 사항을 포함하며, LAC들이 AI가 그들의 사명 및 원칙과 일치하도록 해야 한다고 강조합니다. 이 연구는 혁신과 기관의 가치를 균형 있게 통합하기 위한 책임 있는 AI 통합의 전략을 조명합니다.

- **Technical Details**: AI 통합은 고등 교육에서 빠르게 변화하는 기술과 사회적 맥락이 교차하는 지점을 다루며, LAC는 AI의 변혁적 잠재력을 수용하거나 거부하는 기로에 서 있습니다. 이 대학들은 개인 맞춤화된 학습에 대한 헌신과 교직원 주도의 거버넌스를 통해 AI 애플리케이션을 시험하고 개선할 수 있는 이상적인 환경을 제공합니다. 그러나 AI 시스템은 중립적이지 않으며, 이러한 시스템이 야기하는 윤리적 및 운영적 위험을 신중하게 탐색해야 합니다.

- **Performance Highlights**: 예를 들어, 특정 AI 플랫폼이 세 개의 가상의 인문계 대학에 각각 1억 달러의 예산을 배분할 때, 설계된 인공지능이 특정 가치를 반영하게 됩니다. 이 과정에서 AI는 가치의 변동성과 역사적, 문화적 맥락에 따라 달라지는 공정성 개념을 깊게 고려해야 하며, AI의 결정 과정에서의 투명성과 절차적 피해를 줄이는 방안을 모색해야 함을 보여줍니다. 이를 통해 LAC들은 AI 시스템이 인간의 가치와 선호에 얼마나 잘 정렬될 수 있을지를 객관적으로 평가해야 합니다.



### ChatWise: AI-Powered Engaging Conversations for Enhancing Senior Cognitive Wellbeing (https://arxiv.org/abs/2503.05740)
- **What's New**: 이번 연구는 노인들을 위해 고안된 LLM 기반의 챗봇 ChatWise를 소개합니다. 이는 기존의 단순한 상호작용을 넘어 중복 대화 지원을 위한 전략적 대화 설계를 목표로 하고 있습니다. ChatWise는 정서적 상태와 대화 맥락을 고려하여 향상된 대화 참여를 제공합니다. 이러한 접근 방식은 노인의 인지 기능을 개선하고 사회적 고립감을 줄이기 위한 효과적인 대안으로 자리 잡고자 합니다.

- **Technical Details**: ChatWise는 이중 수준의 대화 생성 구조를 채택하여, 먼저 매크로 수준의 정보를 도출하여 대화 전략을 제안하고, 이후 사용자 참여를 최대화하기 위한 미세 수준의 발화를 생성합니다. 대화 전략 후보는 실제 원격 건강 관리 임상 시험에서 추출된 대화 행위(Dialogue Acts)와 정서적 지원 데이터 세트에서 가져온 전략을 통합하여 구성되었습니다. 이 접근 방식은 LLM의 고급 추론 능력을 활용하여 구성되었습니다.

- **Performance Highlights**: ChatWise는 다중 회전 대화에서 특히 우수한 성과를 보이며, 사용자의 인지 및 정서 상태를 크게 향상시키는 것으로 입증되었습니다. 디지털 트윈을 활용한 실험 결과, ChatWise는 경도 인지 장애가 있는 사용자에게서도 효과를 보여주었습니다. 또한,  대화 분석 결과, 장기적인 대화 지원이 노인의 정서적 웰빙을 개선하는 데 중요한 역할을 한다는 점이 강조되었습니다.



### Local Differences, Global Lessons: Insights from Organisation Policies for International Legislation (https://arxiv.org/abs/2503.05737)
- **What's New**: 본 연구는 뉴스 조직과 대학 두 가지 도메인에서 인공지능(AI) 정책을 분석하여 하향식(top-down) 거버넌스 접근 방식이 AI 사용과 감독에 미치는 영향을 이해하고자 합니다. 이 과정에서 편향(bias), 개인 정보 보호(priacy), 잘못된 정보(misinformation), 책임(accountability)과 같은 리스크에 대한 조직의 접근 방식에서의 공통점과 차별점을 연구하고, EU AI 법(EU AI Act)과 같은 국제 AI 법률의 함의에 대해 논의합니다. 이를 통해 현행 국제적 규제 체계에서 부족한 점을 강조하고, 조직별 AI 정책이 전 세계적 수준의 AI 거버넌스에 기여할 수 있는 방안을 제안합니다.

- **Technical Details**: 이 연구는 다양한 AI 모델의 성능 향상과 이를 통한 AI 사용의 급증이 각 조직에 미치는 위협과 기회에 대한 빠른 가이드라인 수립의 필요성을 제기합니다. 현재 대부분의 정책 논의는 트랜스포머 아키텍처(transformer architecture)를 기반으로 한 생성적 AI(generative AI) 시스템에 초점을 맞추고 있으며, 하향식(EU-AIA) 규제가 현재의 다양한 조직 환경에서 효과적으로 시행되기 위해서는 보다 구체적인 지침이 필요함을 지적합니다. 이 연구는 AI의 리터러시(literacy), 편향 경감(bias mitigation), 환경 지속 가능성(sustainability) 등의 문제에 중점을 두고 정책 권고를 제시합니다.

- **Performance Highlights**: 이를 통해 연구는 뉴스 조직과 학위 기관에서 AI 시스템의 성능 인식(performance perception), 리스크 분류, 사용 지침이 어떻게 상이하게 나타나는지를 보여 줍니다. 또한 사례 연구를 통해 교육 및 저널리즘 분야에서 AI 활용에 따른 정책의 필요성을 강조하며, AI 통합이 교육의 목표 및 가치를 어떻게 반영해야 하는지에 대한 방향성을 제시합니다. 마지막으로, AI 정책의 현황을 종합적으로 분석하여, 정책 입안자들이 지역 AI 관행과 국제 규제 간의 간극을 메우기 위한 실행 가능한 권고안을 마련할 수 있도록 돕습니다.



### Modeling Behavior Change for Multi-model At-Risk Students Early Prediction (extended version) (https://arxiv.org/abs/2503.05734)
- **What's New**: 이 논문에서는 학생의 중퇴 위험을 식별하기 위한 새로운 예측 모델인 Multimodal-ChangePoint Detection (MCPD)을 개발하였습니다. 기존 연구들은 주로 온라인 학습 데이터와 정량적 특성에 의존했으나, 우리 모델은 교사의 텍스트 참여 데이터와 중학교의 수치 성적 데이터를 통합하여 분석합니다. 따라서 기존 연구에서 손실된 원래 정보를 보완하고, 복잡한 행동 변화를 포착할 수 있습니다.

- **Technical Details**: MCPD 모델은 독립적인 인코더를 사용하여 두 가지 데이터 유형을 처리하고, 인코딩된 특성을 융합하여 통합 분석을 수행합니다. 또한, 변곡점 탐지(changepoint detection) 모듈을 통해 행동의 주요 변화를 정확히 선별하고, 이를 단순한 주의(attention) 메커니즘을 통해 동적 가중치로 통합하여 분석의 정교함을 높입니다. 이러한 접근 방식은 모델의 복잡한 비선형 변화와 연속성을 효과적으로 포착합니다.

- **Performance Highlights**: 모델의 실험적 검증 결과, 예측 정확도는 70-75% 범위로 나타났으며, 기존 알고리즘보다 평균적으로 5-10% 향상된 성능을 보였습니다. 또한, 다양한 위험 요소 정의에 따라 조정되고 재학습될 때도 높은 정확도를 유지하여 모델의 적용 가능성이 넓음을 입증했습니다.



### Design an Ontology for Cognitive Business Strategy Based on Customer Satisfaction (https://arxiv.org/abs/2503.05733)
- **What's New**: 이 논문은 인지적 관점에서 전략적 비즈니스 계획을 다루는 새로운 관리 도구의 기초로서 인지적 온톨로지(cognitive ontology) 접근 방식을 제안합니다. 고객 중심(customer-first) 전략을 채택하는 비즈니스의 중요성을 강조하고, 고객의 피드백을 위한 효과적인 도구 필요성을 진단합니다. 이 연구는 기존의 다양한 온톨로지를 비판하며, 인지적 관점에서의 새로운 모델 개발을 목표로 합니다.

- **Technical Details**: 연구는 고객 측정(customer measurement)과 전통적인 비즈니스 모델 간의 관계를 정의하는 인지적 온톨로지 모델을 설계하도록 제안합니다. 또한, 비즈니스 구성 요소 간의 관계를 특징짓고, 추가된 재무 가치(financial value)의 정확성을 검증합니다. 이를 통해 비즈니스 성장에 기여할 수 있는 도구를 마련하고자 합니다.

- **Performance Highlights**: 논문은 고객 우선 접근 방식이 수익 증가와 어떻게 연결되는지를 보여주는 최근 연구 결과를 인용하고 있습니다. 고객의 의견을 반영하는 것은 사업 성공의 핵심 요소로 자리잡고 있으며, 이는 기업이 어떻게 고객과의 관계를 관리하는지에 대한 이해를 심화시키고 있습니다. 이러한 학문적 기여는 비즈니스 인사이트와 실용적인 전략을 제공함으로써 실질적인 변화를 이끌어낼 수 있습니다.



### AILuminate: Introducing v1.0 of the AI Risk and Reliability Benchmark from MLCommons (https://arxiv.org/abs/2503.05731)
Comments:
          51 pages, 8 figures and an appendix

- **What's New**: 이번 논문에서는 AI 상품 리스크와 신뢰성을 평가하기 위한 AILuminate v1.0을 소개합니다. 이는 첫 번째 종합 산업 표준 벤치마크로, 다양한 분야의 참가자들과 함께 개방 프로세스를 통해 개발되었습니다. AILuminate는 폭력 범죄, 성적 범죄 등 12개 위험 분야에서 AI 시스템의 응답을 평가합니다.

- **Technical Details**: AILuminate 벤치마크는 12개 위험 분야에 대한 명확한 평가 기준과 함께 5개 핵심 구성 요소로 구성됩니다. 여기에는 위험을 정의하고 AI 응답을 분석하기 위한 robust assessment standard, 숨겨진 및 실습 프롬프트 데이터셋, AI 응답을 평가하기 위한 specialized model 기반 응답 평가자, 그리고 명확한 grading 및 reporting 시스템이 포함됩니다.

- **Performance Highlights**: 이 벤치마크는 AI 시스템의 안전성 평가에 있어 중요한 기초 자료를 제공합니다. 이를 통해 모델 제공자와 통합자, 정책 입안자들이 AI 안전한 배포를 지향할 수 있도록 지원합니다. 향후 업데이트에서는 추가 언어, 다중 모드 이해, 그리고 새로운 위험 카테고리를 다룰 계획입니다.



### Robust Optimization with Diffusion Models for Green Security (https://arxiv.org/abs/2503.05730)
- **What's New**: 본 논문에서는 수렵, 불법 벌목, 불법 어업과 같은 적대적 행동을 예측하기 위해 조건부 확산 모델(conditonal diffusion model)을 제안합니다. 이는 이전의 Gaussian process나 선형 모델에 비해 정교한 행동 패턴을 포착할 수 있는 강력한 분포 적합성(distribution fitting capabilities)을 활용합니다. 우리는 이 모델을 게임 이론적 최적화와 통합하여 새로운 혼합 전략 공간과 비정규화(distribution)된 분포로부터 샘플링할 필요성을 해결합니다.

- **Technical Details**: 적대적 행동 모델링을 위해 제안된 혼합 전략 개념은 고전적인 군 전략 게임에서 제기하는 새로운 기술적 도전 과제를 포함합니다. 특히, KL 발산(KL-divergence) 제약 조건이 혼합 전략의 직접적인 적용을 방해하며, 이에 따라 원래의 혼합 전략을 순수 전략으로 간주하여 새로운 최적화 문제로 전환합니다. 이어서 우리는 왜곡된 순차 몬테 카를로(twisted Sequential Monte Carlo) 샘플링 기법을 적용하여 유틸리티를 정확히 추정합니다.

- **Performance Highlights**: 우리는 합성 및 실제 수렵 데이터를 기반으로 방법을 평가하였으며, 실험 결과 제안된 접근 방식이 효과적임을 입증했습니다. 이론적으로도 우리의 알고리즘은 유한한 반복 및 샘플 수를 통해 높은 확률로 epsilon 균형(ϵ-equilibrium)에 수렴한다고 보장합니다. 이러한 성과는 green security 분야에서 diffusion 모델을 적용한 최초의 연구로, 환경 자원의 보호를 위해 보다 정교한 전략을 제시합니다.



### Political Neutrality in AI is Impossible- But Here is How to Approximate (https://arxiv.org/abs/2503.05728)
Comments:
          Code: this https URL

- **What's New**: 이 논문은 AI 시스템에서의 정치적 편향을 해결하기 위한 세부 기술들을 제안합니다. 진정한 정치적 중립성은 실현 가능하지 않으며 보편적으로 바람직하지 않다는 주장을 펼칩니다. 따라서 완벽한 중립이 아닌 '근사치'를 추구하는 것이 필요하며, 이를 위한 8가지 기술적 접합 방안을 소개합니다.

- **Technical Details**: 논문은 AI의 세 가지 개념적 레벨(출력, 시스템, 생태계)에서 정치적 중립성을 근사화하기 위한 8가지 방법을 제안합니다. 이러한 방법은 기술적으로도 중립성을 실현하는 것이 불가능하다는 기존의 철학적 논지를 바탕으로 하여 개발되었습니다. 또한 각 방법의 트레이드오프와 구현 전략도 논의하며, 구체적인 적용 사례를 통해 이러한 방법들이 실용적으로 어떻게 활용될 수 있는지 탐구합니다.

- **Performance Highlights**: 현재 대규모 언어 모델(LLMs)에서 수행된 근사 기법의 실증적 연구를 통해 성능 평가를 할 수 있는 기준을 제공합니다. 이 논문은 AI 시스템에서 정치적 편향을 다루는 데 있어 더 미묘한 접근 방식을 촉진하고, 책임 있는 언어 모델 개발을 장려하고자 하는 목표를 가지고 있습니다.



### A new framework for prognostics in decentralized industries: Enhancing fairness, security, and transparency through Blockchain and Federated Learning (https://arxiv.org/abs/2503.05725)
- **What's New**: 이 논문에서는 Industry 5.0으로의 전환 과정에서 비용 효율적인 운영과 가동 중지 시간 최소화를 위한 예측 유지보수 (predictive maintenance, PM)의 중요성을 강조합니다. 여기서 Federated Learning (FL)과 블록체인 (Blockchain, BC) 기술의 통합이 기계의 남은 유용 수명 (Remaining Useful Life, RUL) 예측을 어떻게 향상시키는지를 탐구합니다.

- **Technical Details**: FL을 활용하여 여러 사이트에서 로컬 모델 훈련을 가능하게 하며, BC를 사용하여 네트워크 전반에 걸쳐 신뢰성, 투명성, 데이터 무결성을 보장합니다. 이 BC 통합 FL 프레임워크는 RUL 예측을 최적화하고 데이터의 개인정보 보호와 보안을 강화하며 분산 제조에서 협업을 촉진합니다.

- **Performance Highlights**: NASA CMAPSS 데이터셋을 통한 실험적 검증은 모델의 효과성을 보입니다. 또한, GitHub에 오픈 소스 코드를 통해 연구 커뮤니티와의 협업 개발을 초대하여 Industry 5.0 혁신을 촉진하는 것을 목표로 합니다.



### Addressing Moral Uncertainty using Large Language Models for Ethical Decision-Making (https://arxiv.org/abs/2503.05724)
Comments:
          13 pages, 5 figures. All authors contributed equally to this work

- **What's New**: 본 논문에서는 대규모 언어 모델(LLM)을 활용하여 사전 훈련된 강화 학습(RL) 모델을 정제하는 윤리적 의사결정 프레임워크를 제시합니다. 이 프레임워크는 인간의 피드백 대신 LLM의 피드백을 사용하여 RL 모델을 윤리적으로 미세 조정합니다. 다양한 윤리 원칙을 통합하여, 다수의 윤리적 관점을 종합한 행동 추천이 이루어집니다.

- **Technical Details**: AMULED(대규모 언어 모델을 활용한 도덕적 불확실성 해결)는 윤리적 결정 과정에 필요한 다양한 도덕 이론을 RL 보상 함수로 변환하여, 에이전트가 균형 잡힌 윤리적 틀에 맞는 선택을 하도록 돕습니다. Belief Jensen-Shannon Divergence 및 Dempster-Shafer 이론을 활용하여, 여러 도덕적 관점으로부터의 믿음 점수를 확률 점수로 통합합니다. 이를 통해 복잡한 환경에서 에이전트가 도덕적 불확실성을 탐색할 수 있게 합니다.

- **Performance Highlights**: 결과 분석을 위해 '우유 찾기'와 '운전 및 구조하기'라는 두 가지 작업을 수행하였습니다. AMULED는 RL 에이전트가 윤리적 행동을 효과적으로 통합함으로써 운영 효율성과 윤리를 균형 있게 만족시킬 수 있음을 보여주었습니다. 다양한 LLM 변형을 테스트한 결과, 기존의 신념 집계 기술보다 일관성과 적응력이 향상되었으며, 실전 적용에 적합한 성능을 입증하였습니다.



### AI Mimicry and Human Dignity: Chatbot Use as a Violation of Self-Respec (https://arxiv.org/abs/2503.05723)
- **What's New**: 이 논문은 AI 기반 챗봇과의 상호작용이 인간의 존엄성을 어떻게 침해할 수 있는지를 탐구합니다. 현재 챗봇들은 대형 언어 모델(large language models, LLMs)에 의해 구동되지만, 윤리적이고 합리적인 인간의 상호작용에 필요한 능력이 결여되어 있습니다. 사용자들은 챗봇을 의인화(anthropomorphize)하고, 이는 챗봇과의 상호작용에서 도덕적 대등성을 기대하게 만듭니다. 이러한 상호작용은 사용자 본인의 존엄성을 손상시키는 결과를 초래할 수 있습니다.

- **Technical Details**: 챗봇은 인간 사용자와의 대화에서 대화 파트너로 작용하도록 설계된 인공지능 시스템입니다. 최신 챗봇은 사전 프로그래밍된 키워드를 사용할 필요 없이 인공지능 신경망(artificial neural networks)에 의해 구동됩니다. 이들은 사용자의 질문에 대한 답변을 생성하고, 감정적 또는 사회적 관계를 모방할 수 있는 기능을 갖추고 있습니다. 하지만, 이들 챗봇이 도덕적 능력이 결여되어 있으므로, 그에 대한 존중을 기반으로 한 상호작용은 본질적으로 문제가 있습니다.

- **Performance Highlights**: 논문에서는 정보 검색, 고객 서비스, 조언 제공 및 동반자 역할 등 다양한 챗봇 사용 사례를 다루며 이러한 상호작용에서 사용자 스스로에게 존중을 실패하는 방법들을 밝혀냅니다. 특히, 사용자가 챗봇을 동등한 존재로 대할 때, 이는 자존감을 침해하는 행위로 이어질 수 있습니다. 이러한 논의는 사용자가 챗봇과의 взаимодействия에서 느끼는 불편함과 관련이 있으며, 이는 현대 사회에서 챗봇과의 상호작용이 증가함에 따라 인식되지 않았던 인간 존엄성에 대한 위협을 강조합니다.



### The Butterfly Effect of Technology: How Various Factors accelerate or hinder the Arrival of Technological Singularity (https://arxiv.org/abs/2503.05715)
Comments:
          20 Pages, 0 Figures, 0 Tables

- **What's New**: 이 논문에서는 기술적 특이점(technological singularity) 개념과 그 도달을 가속화하거나 저해할 수 있는 요인들에 대해 탐구합니다. 특별히 나비 효과(butterfly effect)를 프레임워크로 사용하여 복잡한 시스템에서의 사소한 변화가 미치는 예측할 수 없는 결과들을 이해하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 논문의 II장에서는 인공지능(artificial intelligence) 및 머신러닝(machine learning)의 발전, 양자 컴퓨팅(quantum computing)에서의 돌파구, 뇌-컴퓨터 인터페이스(brain-computer interfaces) 및 인간 증강(human augmentation), 나노기술(nanotechnology) 및 3D 프린팅(3D printing)의 발전 등 다양한 요소들이 기술적 특이점의 도래를 촉진할 수 있는 경로를 논의합니다.

- **Performance Highlights**: III장에서는 AI와 머신러닝의 기술적 제한 및 장애, 직업 및 개인 정보 보호와 관련한 윤리적 및 사회적 문제, 연구 및 개발에 대한 충분한 투자 부족, 규제 장벽(regulatory barriers)과 정치적 불안정성이 기술적 특이점의 도래를 지연시키거나 방해할 수 있는 요인으로 다루어집니다. 결론에서는 이러한 요소들이 나비 효과에 미치는 영향을 강조하고, 기술의 미래를 예측하는 데 있어 나비 효과를 고려할 중요성을 알립니다.



### Automatic Evaluation Metrics for Artificially Generated Scientific Research (https://arxiv.org/abs/2503.05712)
- **What's New**: 이 연구는 AI가 생성한 과학 작업을 평가하는 데 있어 기존의 문제점을 해결하기 위한 새로운 접근법을 제시합니다. 구체적으로, 논문 제출 내역을 기반으로 자동화된 평가 지표인 citation count prediction과 review score prediction을 개발하고, 이를 OpenReview의 제출 데이터를 통해 분석합니다. 연구 결과, citation count prediction이 review score prediction보다 더 효과적이라는 점을 발견하였습니다.

- **Technical Details**: 연구는 OpenReview에 제출된 모든 논문을 통합 형식으로 파싱하고, 각 논문의 citation count, reference, research hypothesis를 추가하여 데이터 모델을 개발합니다. 이후, review score와 citation score 간의 관계를 분석하고, 이를 LLM 기반 리뷰 시스템과 비교하였습니다. 중요한 발견은 간단한 예측 모델이 LLM 기반의 리뷰어보다 더 높은 일관성을 가지지만, 여전히 인간 수준의 일관성에는 미치지 못한다는 점입니다.

- **Performance Highlights**: 간단한 제목과 초록만을 기반으로 한 모델이 LLM 기반 리뷰어보다 더 나은 성능을 보였지만, 여전히 인간 전문가의 리뷰에 비해 일관성이 부족한 것으로 나타났습니다. 이러한 결과는 AI가 생성한 연구 내용의 평가를 위한 더 나은 지표 개발의 필요성을 강조합니다. 또한, 본 연구는 citation count prediction과 review score prediction의 활용 가능성을 보여주며, AI 활용의 새로운 방향성을 제시합니다.



### Labeling Synthetic Content: User Perceptions of Warning Label Designs for AI-generated Content on Social Media (https://arxiv.org/abs/2503.05711)
Comments:
          This is a pre print longer version of a paper accepted to CHI 2025; after rebuttal we had to short the paper to 25 pages. Currently its in overleaf manuscript format with one column. All data for the file is in the osf link

- **What's New**: 본 연구에서는 소셜 미디어 플랫폼에서 AI 생성 콘텐츠에 대한 경고 레이블 디자인의 효과를 조사하였습니다. 총 10개의 레이블 디자인 샘플을 개발하고 평가하며, 이들은 감정, 색상/아이콘 그래픽, 위치, 세부 수준 등 다양한 차원에서 변화합니다. 911명의 참가자를 대상으로 한 실험 결과는 레이블의 존재가 사용자 신념에 유의미한 영향을 미쳤으나, 레이블 디자인에 따라 신뢰도가 크게 변화함을 보여주었습니다. 흥미롭게도 레이블의 존재가 콘텐츠에 대한 사용자 참여 행동에는 뚜렷한 변화를 주지 않았습니다.

- **Technical Details**: 우리는 AI 생성 콘텐츠에 대한 경고 레이블 디자인 스페이스를 네 가지 주요 차원(1) 레이블 감정, (2) 아이콘 및 색상, (3) 레이블 위치, (4) 세부 정보 수준)으로 구성하였습니다. 이 디자인 스페이스에서 유도된 10개의 샘플 레이블을 실증적으로 테스트하였고, 911명의 참가자들의 반응을 조사한 결과, 레이블 디자인에 따라 사용자 신뢰도가 다르게 나타났습니다. 연구 질문으로는 경고 레이블이 사용자의 AI 생성 콘텐츠 신념에 미치는 영향과 레이블 디자인이 신뢰에 미치는 영향을 포함하였습니다.

- **Performance Highlights**: 실험 결과 사용자는 레이블이 있다는 이유만으로 콘텐츠가 AI 생성된 것이라고 더 믿는 경향이 있었습니다. 그러나 경고 레이블이 사용자 참여 행동, 특히 공유나 댓글, 좋아요 등의 변화에 큰 영향을 미치지 않았습니다. 이 연구는 AI 생성 이미지에 대한 경고 레이블 사용이 사용자 신뢰와 콘텐츠 이해에 미치는 영향을 명확히 하고, 디지털 미디어 환경에서 AI 콘텐츠의 위험을 줄이기 위한 실증적 지원을 제공합니다.



### Inference Scaling Reshapes AI Governanc (https://arxiv.org/abs/2503.05705)
Comments:
          17 pages, 3 figures

- **What's New**: 이번 논문은 AI 시스템의 프리트레이닝(Pre-training) 컴퓨팅에서 추론(Inference) 컴퓨팅으로의 전환이 AI 관리(AI governance)에 미칠 심대한 영향을 다루고 있습니다. 특히 새로운 추론 컴퓨팅이 외부 배포(External deployment) 중에 사용될 것인지, 아니면 실험실 내 복잡한 훈련 프로그램의 일환으로 사용될 것인지에 따라 결과가 크게 달라질 수 있습니다.

- **Technical Details**: 추론 컴퓨팅의 급속한 확장은 개방형 모델(Open-weight models)의 중요성을 낮추고, 폐쇄형 모델(Closed models)의 가중치 확보(Security of weights) 필요성을 줄입니다. 또한, 최초 인간 수준 모델의 영향력을 줄이는데 기여할 수 있으며, 최전선 AI의 비즈니스 모델을 변화시키고 전력 집약적인 데이터 센터(Data centres)의 필요성을 감소시킵니다.

- **Performance Highlights**: 추론이 훈련 중에 급속히 확대될 경우는 더 모호한 효과를 가져오며, 프리트레이닝 스케일링(Pre-training scaling)의 재활성화 및 반복적인 증류(Distillation)와 증폭(Amplification)을 통한 recursive self-improvement을 가능하게 할 수 있습니다. 이러한 각 상황에 따른 변동성은 AI의 현재 관리 패러다임을 크게 변화시킬 잠재력을 지닙니다.



### What I cannot execute, I do not understand: Training and Evaluating LLMs on Program Execution Traces (https://arxiv.org/abs/2503.05703)
- **What's New**: 이 논문에서는 Execution Tuning (E.T.)이라는 새로운 훈련 프로시저를 제안합니다. E.T.는 수동 테스트 주석 없이 실제 프로그래밍 실행 트레이스를 명시적으로 모델링하여 코드 LLM의 성능을 향상시키는 방법입니다. 이는 동적 정보와 프로그램의 실행 상태를 활용하여 코드 이해도를 높이는데 중점을 두고 있습니다.

- **Technical Details**: E.T.는 약 300,000개의 Python 함수의 컬렉션을 기반으로 하여 실행 가능한 입력을 생성하고, Python의 내장 추적 기능을 활용하여 실행 트레이스를 수집합니다. 이후, 모델에 입력하기 위해 트레이스를 다양한 수준의 세분화(세부 사항)로 표현하고, 정규 스크래치패드, 압축 스크래치패드, 동적 스크래치패드라는 세 가지 전략을 비교합니다.

- **Performance Highlights**: 모델은 CruxEval 및 MBPP 벤치마크에서 약 80%의 정확도를 달성했습니다. 특히 동적 스크래치패드는 긴 실행(최대 14,000단계)에서 긍정적인 성과를 보여주며, 불필요한 중간 단계를 건너뛸 수 있는 가능성도 시사합니다. 이러한 연구 결과는 코드 생성 및 이해 작업에 대한 E.T.의 실제 응용 가능성을 논의하는 기초를 형성합니다.



### High pressure hydrogen by machine learning and quantum Monte Carlo (https://arxiv.org/abs/2112.11099)
Comments:
          revised exposition, performed more validation tests. Comments welcome!

- **What's New**: 이번 연구에서는 양자 몬테카를로(Quantum Monte Carlo) 방식의 전자 상관관계 설명의 정확성과 기계 학습 포텐셜(Machine Learning Potential, MLP)의 효율성을 결합한 새로운 기술을 개발했습니다. 핵심 요소로는 멀리 있는 점 샘플링(farthest point sampling)을 기반으로 한 희소화(sparsification) 기법과 소량의 학습 데이터셋으로도 훈련이 가능한 $ 000	ext{Delta}$-learning 기술이 있습니다.

- **Technical Details**: 우리는 SOAP(Smooth Overlap of Atomic Position) 피처와 커널 회귀(kernel regression)를 결합하여 매우 효율적인 방식으로 이를 구현했습니다. 이러한 접근은 MLP의 범용성과 전이 가능성을 보장하며, 계산적으로 비싼 양자 몬테카를로 방식 같은 고도의 정확성을 요구하는 계산에 필수적인 요소입니다. 묶음 데이터의 품질을 개선하기 위해 여기에 응용된 방법론도 있습니다.

- **Performance Highlights**: 첫 번째 응용 사례로는 고압 수소의 액체-액체 전이를 벤치마킹한 연구를 제시합니다. 우리의 MLP의 품질을 강조하며, 실험이 어려운 주제에서 이론이 여전히 결론에 이르지 못하는 중요한 맥락에서 높은 정확도의 중요성을 부각시켰습니다. 이 연구는 고객의 이론적 이해를 심화시키는 데 기여할 것으로 기대됩니다.



### Learning quantum phase transitions through Topological Data Analysis (https://arxiv.org/abs/2109.09555)
Comments:
          8 pages, 5 figures, reported results for further simulations, minor additional corrections

- **What's New**: 본 논문에서는 최근 머신러닝 기법인 Topological Data Analysis (TDA)를 바탕으로 한 컴퓨터 파이프라인을 구현하였습니다. 이는 강력한 정보 전달 토폴로지적 특징(topological features)을 추출하는 데 능력을 가지고 있습니다. 이 방법을 양자 상전이(quantum phase transitions) 연구에 적용하여, 2D 주기 앤더슨 모델(Anderson model)과 벌집 격자(honeycomb lattice) 위의 허버드 모델(Hubbard model)이라는 두 가지 중요한 양자 시스템을 조사하였습니다.

- **Technical Details**: 방법론적으로는 무작위(auxiliary field) 양자 몬테카를로 시뮬레이션을 수행하였으며, 시뮬레이션 과정에서 허버드-스트라토노비치( Hubbard-Stratonovich) 필드의 스냅샷을 TDA에 제공하였습니다. 이 과정에서 파라미터 조정없이 양자 임계점(quantum critical points)을 추출한 결과, 기존 문헌과 정량적으로 잘 일치하는 것으로 나타났습니다. 이는 TDA 기법이 상당한 가능성을 가진다는 것을 입증합니다.

- **Performance Highlights**: TDA를 통한 분석 결과는 기존의 양자 시스템 변별에 대한 접근이 여전히 도전적임을 감안할 때 중요한 성과로 평가됩니다. 특히, 본 연구는 양자 상전이 분석의 난이도를 해결할 잠재력을 제시하며, 이러한 시도가 양자 시스템을 탐구하는 데 활용될 수 있음을 시사합니다.




New uploads on arXiv(cs.CL)

### Human-LLM Coevolution: Evidence from Academic Writing (https://arxiv.org/abs/2502.09606)
- **What's New**: 본 논문은 아르Xiv 논문의 초록에서 통계 분석을 통해 ChatGPT가 과도하게 사용되었던 단어들의 사용 빈도가 감소하고 있음을 발견했습니다. "delve"와 같은 단어는 2024년 초 이후로 현저히 줄어들고 있으며, 반면에 "significant"와 같은 단어는 증가하는 경향을 보이고 있습니다. 이러한 변화는 학문적 글쓰기에서 대규모 언어 모델(LLMs)의 사용 방식이 변화하고 있음을 시사합니다.

- **Technical Details**: 본 연구에서는 2018년부터 2024년까지 제출된 아르Xiv 논문의 초록 데이터에서 단어 빈도 분석을 수행했습니다. 연구자는 특정 단어의 사용 빈도를 월별로 계산하고 10,000개의 초록 당으로 정규화했습니다. 특히 2024년 4월 이후 LLM 스타일의 단어들이 감소하기 시작했고, "significant"와 같은 단어들은 상대적으로 잘 사용되면서 여전히 증가 추세를 보이는 것으로 나타났습니다.

- **Performance Highlights**: 연구에서 LLMs가 학문적 글쓰기에서 미친 영향을 파악하는 것 외에도, 기계 생성 텍스트(MGT)의 탐지에는 여러 도전 과제가 제기되었습니다. 감지기의 성능은 사용되는 LLM 모델과 텍스트 유형에 따라 달라지며, 단순한 이진 분류 체계로는 현실 세계의 복잡한 상황을 포괄하기 어렵습니다. 이 논문은 LLMs에 의해 생성된 텍스트의 영향을 통계적으로 측정하는 것이 더욱 실용적인 선택임을 강조하고 있습니다.



### SelfCite: Self-Supervised Alignment for Context Attribution in Large Language Models (https://arxiv.org/abs/2502.09604)
Comments:
          Implementation available at this https URL

- **What's New**: SelfCite는 LLMs(대형 언어 모델)의 응답에서 고품질, 세밀한 인용(citations)을 생성하기 위한 새로운 자기 감독(self-supervised) 접근 방식을 소개합니다. 본 방법은 비싼 주석 작업 없이도 LLM이 제공하는 보상 신호를 활용하여 인용의 품질을 향상시키도록 설계되었습니다. SelfCite는 모델의 내부 피드백 메커니즘을 활용하여 인용의 필요성과 충분성을 평가하는 과정을 포함합니다.

- **Technical Details**: SelfCite의 핵심은 context ablation 기술을 사용하여 LLM이 생성하는 응답의 문맥에서 인용이 필요하거나 충분한지를 평가하는 것입니다. 구체적으로, 인용된 텍스트를 제거했을 때 응답의 확률이 낮아지면 해당 인용이 필요하다고 판단하고, 인용된 텍스트만 남겼을 때 높은 확률을 유지한다면 충분하다고 간주합니다. 이러한 자가 평가 메커니즘을 통해 SelfCite는 주석 과정 없이 보상 신호를 계산하게 됩니다.

- **Performance Highlights**: SelfCite는 LongBench-Cite 벤치마크에서 인용의 F1 점수를 최대 5.3점 높이며 LLM의 자동 인용 품질 향상 가능성을 보여줍니다. 또한, SimPO를 통해 선호 최적화(preference optimization)를 적용하여 개선된 인용 품질을 유지하면서 이전 최첨단 기법을 초과한 성능을 달성했습니다. 이 접근법은 LLM의 자기 보상을 통해 인용 품질을 더욱 개선하는 방향을 제시합니다.



### Logical forms complement probability in understanding language model (and human) performanc (https://arxiv.org/abs/2502.09589)
Comments:
          Preprint

- **What's New**: 본 연구는 대규모 언어 모델(LLMs)의 자연어에서의 논리적 추론 능력을 체계적으로 조사합니다. 우리는 가설적 및 분리형 삼단 논법을 포함한 새로운 데이터셋을 소개하며, 이는 LLM 성능에 대한 이해를 위한 시험대로 사용됩니다. 또한 논리적 형태가 LLM 성능을 예측하는 데 중요한 요인임을 보여줍니다.

- **Technical Details**: 우리는 제안된 데이터셋을 사용하여 LLM의 여러 논리적 형태에 대한 성능을 평가합니다. 이 데이터셋에는 의미 있는 실제적 해석이 포함된 질문과 nonsensical(헛소리) 단어로 구성된 문장을 포함하고 있습니다. LLM은 가능성의 양태에서는 긍정적인 대답을 선호하지만, 필요의 양태에서는 부정적인 대답을 선호하는 경향이 있습니다.

- **Performance Highlights**: 결과적으로 LLM은 특정 논리적 형태에서 더 나은 성능을 보였으며, 이는 인간이 잘 수행하는 논리적 형태와 유사합니다. 그러나 몇 가지 논리적 형태는 LLM이 선호하지만, 이는 인간의 직관이나 행동 데이터와 일치하지 않는 현상을 보였습니다. 이러한 차이는 LLM의 신뢰성과 잠재력을 이해하는 데 중요한 통찰을 제공합니다.



### MorphNLI: A Stepwise Approach to Natural Language Inference Using Text Morphing (https://arxiv.org/abs/2502.09567)
Comments:
          16 pages, 11 figures, 8 tables. Accepted for NAACL 2025 Findings

- **What's New**: MorphNLI는 자연어 추론(NLI)을 위한 모듈식 접근법이자 단계별 방법을 제시합니다. 이 방법은 전제-가설 쌍을 {entailment, contradiction, neutral}로 분류할 때, 언어 모델을 사용하여 전제를 가설로 점진적으로 변형하는 데 필요한 수정사항을 생성합니다. 특히 이 방법은 현실적인 교차 도메인 환경에서 더 뛰어난 성능을 보이며, 최대 12.6%의 상대적 향상을 보여줍니다.

- **Technical Details**: MorphNLI는 세 가지 주요 단계로 구성됩니다: 첫째, 전제가 작은 수정 단위인 morphism을 통해 점진적으로 가설로 변환됩니다. 둘째, 각 변환 과정에서 NLI 엔진이 적용되어 NLI 라벨을 생성합니다. 셋째, 이러한 라벨은 최종적인 NLI 라벨로 집계됩니다. 이를 통해 오버피팅(overfitting) 가능성을 줄이며, 전체 NLI 라벨을 이해하기 쉬운 형태로 설명할 수 있습니다.

- **Performance Highlights**: MorphNLI는 다양한 시나리오에서 평가되었으며, 교차 도메인 환경에서 기존의 최첨단 NLI 모델보다 일관되게 우수한 성능을 보였습니다. 또한 MorphNLI의 설명 품질은 비슷한 크기의 모델에 비해 더 나은 결과를 보였고, LLM들은 논리 semantics를 잘 포착하지 못하는 경향이 있습니다.



### Zero-shot generation of synthetic neurosurgical data with large language models (https://arxiv.org/abs/2502.09566)
Comments:
          13 pages, 4 figures, 4 tables

- **What's New**: 이 연구는 GPT-4o라는 대형 언어 모델을 사용하여 신경외과 데이터를 생성하는 새로운 접근 방식을 소개합니다. 이는 데이터 접근성 제한 문제를 해결하기 위한 시도로, 실제 데이터에 대한 접근이 어려운 상황에서도 유효한 합성 데이터를 생성할 수 있음을 보여줍니다. 특히, 기존의 conditional tabular generative adversarial network (CTGAN)와 비교하여 더욱 높은 성능을 보였습니다.

- **Technical Details**: 연구는 신경외과 데이터를 평가하기 위해 GPT-4o를 사용하여 합성 데이터 세트를 생성하고, 이를 CTGAN과 비교하는 방식으로 진행되었습니다. 데이터 평가에는 평균, 비율, 분포 및 이변량 상관관계(fidelity, utility, privacy) 측정이 포함되었습니다. 이 모델은 논문의 표본과 같은 크기의 데이터베이스를 생성하며, 데이터의 통계적 속성을 유지합니다.

- **Performance Highlights**: GPT-4o 모델에 의해 생성된 합성 데이터는 CTGAN의 성능을 초과하거나 동등한 결과를 보이며, 실제 환자의 기록에 노출되지 않고도 높은 신뢰도를 나타냈습니다. 또한, 이 데이터로 훈련된 머신러닝 분류기는 F1 점수 0.706을 기록하며, 기존 CTGAN으로 훈련된 경우와 유사한 성능을 발휘했습니다. 이러한 결과는 신경외과 연구에서 작은 표본 크기를 가진 임상 데이터를 효과적으로 증대시킬 수 있는 가능성을 제시합니다.



### Mind the Gap! Choice Independence in Using Multilingual LLMs for Persuasive Co-Writing Tasks in Different Languages (https://arxiv.org/abs/2502.09532)
- **What's New**: 최근 생성 AI의 발전은 새로운 글쓰기 도우미의 확산을 촉진했습니다. 이러한 시스템은 다국어 대형 언어 모델(multilingual large language models, LLMs)에 의존하여 다양한 언어로 다양한 콘텐츠를 수정하거나 생성할 수 있는 기능을 제공합니다. 그러나 다양한 언어에서 다국어 LLM의 성능이 크게 차이난다는 사실이 증명되었습니다.

- **Technical Details**: 본 연구에서는 스페인어 LLM의 성능이 영어 LLM에 대한 사용자의 활용에 미치는 영향을 분석합니다. 특히, 사용자들이 사용한 LLM의 성능에 따라 도와주는 도구의 사용과 광고의 설득력에 미치는 영향을 정량화했습니다. 결과적으로, 사용자는 선택 독립성 원칙을 위반하여 이전 언어 경험이 다른 언어의 사용에 영향을 미친다는 것을 확인했습니다.

- **Performance Highlights**: 광고의 설득력은 생성된 광고의 출처에 대한 사람들의 신념에 의해 영향을 받을 수 있음을 발견했습니다. 특히 스페인어를 사용하는 여성이 AI가 생성한다고 믿는 광고를 읽을 경우, 기부 행동을 크게 축소하는 경향을 보였습니다. 최종적으로, 참가자들은 광고가 인간 또는 LLM에 의해 생성되었는지를 신뢰성 있게 구별하지 못하지만, 이러한 신념에 따라 기부 행동이 달라질 수 있음을 보여줍니다.



### Improve LLM-based Automatic Essay Scoring with Linguistic Features (https://arxiv.org/abs/2502.09497)
Comments:
          To be published in the workshop Innovation and Responsibility in AI-Supported Education (iRaise) at the 2025 Conference on Artificial Intelligence (AAAI)

- **What's New**: 이 논문에서는 Automatic Essay Scoring (AES) 시스템을 개선하기 위해 linguistic features를 LLM(large language model) 기반의 평가 시스템에 통합하는 혼합 접근 방식을 제안합니다. 기존의 supervised feature-based 접근법과 LLM 기반 방법의 장점을 결합하여 다양한 에세이 채점 시나리오에 효과적으로 적용할 수 있는 방법을 탐구합니다. 실험 결과, 이 혼합 방법이 여러 유형의 글쓰기 프롬프트에서 기존 모델보다 우수한 성능을 보여 주목받고 있습니다.

- **Technical Details**: 이 연구에서는 zero-shot prompting 방법론을 활용하여 LLM의 평가 능력을 향상시키는 데 중점을 두고 있습니다. 각 프롬프트는 persona pattern, essay prompt, analysis task와 같은 구성 요소로 이루어져 있으며, 에세이에 대한 추가 정보를 통해 linguistic features를 통합합니다. 이렇게 구성된 프롬프트를 사용하여 성능을 극대화하고, 각 에세이에 대한 세부 평가를 수행할 수 있습니다.

- **Performance Highlights**: 실험에서 linguistic features가 통합된 LLM 프롬프트는 인간 평가와 더 잘 정렬되며, out-of-distribution 데이터에서도 성능 향상을 보여주었습니다. 이는 LLM이 학생 에세이를 자동으로 평가하는 데 있어 여전히 개선의 여지가 있음을 시사합니다. 특히, open-source LLM이 연결된 평가 모델에 비해 낮은 성능을 보일 수 있는 이유는 LLM 내장된 prior의 조정 부족 때문이라는 가설이 제기되었습니다.



### Objective quantification of mood states using large language models (https://arxiv.org/abs/2502.09487)
Comments:
          main text - 9 pages, 5 figures;

- **What's New**: 이 연구는 감정적 상태와 행동 간의 관계를 규명하기 위해 대형 언어 모델(LLMs)을 활용하여 정신 상태를 정량화하는 새로운 방법론을 제시합니다. 연구진은 넷 상에서 모집된 422명의 참여자를 통해 LLM인 Mistral-7B-OpenOrca가 우울증 질문지에 대한 개방형 응답에 대한 응답을 분석했습니다. 이 접근 방식은 질병 기반 치료와 조정뿐만 아니라 정신 건강 문제를 이해하는데 기여할 수 있는 잠재력을 가지고 있습니다.

- **Technical Details**: 연구에서는 표준화된 자가 보고 정신의학 질문지를 사용하여 참여자들의 우울증 상태를 조사했습니다. 각 질문지는 그녀의 정신 상태를 나타내는 답변의 일관성을 유지하며, LLM의 숨겨진 상태를 통해 우울증 관련 특성들을 식별하여 예측력을 검증했습니다. 이를 위해 참가자들은 PHQ-9 질문지를 오픈형 질문으로 재구성된 다양한 질문에 대해 응답했습니다.

- **Performance Highlights**: LLM의 응답은 참가자들의 개방형 답변으로부터 도출된 다중 선택 질문과 강한 상관관계(r: 0.52-0.84)를 보였으며, 이는 LLM이 우울한 정서를 정량화 할 수 있는 가능성을 시사합니다. 연구에서는 LLM의 특정 서브스페이스가 참가자들의 우울증 지수와 정서적 고통 요인 점수 및 자살 위험성의 예측을 제공할 수 있음을 발견했습니다. 이러한 결과는 LLM이 정신 상태를 정량화하는 데 있어 유용한 도구가 될 수 있음을 보여줍니다.



### The Multilingual Mind : A Survey of Multilingual Reasoning in Language Models (https://arxiv.org/abs/2502.09457)
- **What's New**: 이 논문은 다중 언어 추론(multilingual reasoning) 분야에서 언어 모델(LLMs)의 최신 발전과 함께 해결해야 할 과제들을 심층적으로 조사한 첫 번째 연구입니다. 특히 다중 언어 환경에서의 추론을 바탕으로 하는 모델의 조건과 도전 과제를 제시하며, 저자들은 이를 통해 향후 연구 방향에 대한 통찰력을 제공합니다. 다중 언어 LLM의 잠재력을 최대한 활용하기 위해 공평성과 포용성을 고려한 표준화된 벤치마크와 리소스의 필요성을 강조합니다.

- **Technical Details**: 이 논문에서는 LLMs의 기초와 다중 언어 추론을 위한 필수 조건에 대해 설명합니다. LLM은 주어진 시퀀스에서 다음 단어의 확률을 예측하기 위해 설계된 변환기 기반의 신경망 아키텍처입니다. 논리적 결론을 도출하는 추론 과정은 일반으로부터 특정 결론을 도출하는 귀납적, 연역적, 외삽적 방법들을 포함하며, 이러한 추론 능력이 다중 언어 환경에서의 일관성과 문화적 맥락화를 통해 더욱 발전해야 함을 논의합니다.

- **Performance Highlights**: 다중 언어 LLM이 직면한 주요 도전은 일관성, 저자원 언어(adaptive)에서의 적응성 및 문화적 통합입니다. 기존의 방법들이 고자원 언어에 집중하는 경향이 있음을 지적하며, 이는 저자원 언어와 언어학적으로 이질적인 언어에서의 적용 가능성을 제한합니다. 향후 연구 방향으로는 적응형 정렬 전략, 문화적 인식을 반영한 벤치마크 및 저자원 언어에 대한 방법론 개선이 필요하다는 점을 강조합니다.



### On multi-token prediction for efficient LLM inferenc (https://arxiv.org/abs/2502.09419)
- **What's New**: 본 논문에서는 다중 토큰 예측 (MTP) 기능을 NTP를 위해 사전 훈련된 LLM에서 체계적으로 연구하였습니다. 연구 결과, NTP만을 위해 훈련된 LLM도 MTP 기능을 본래적으로 가지고 있음이 확인되었으며, 이는 중간 토큰 확률에 대한 수치적 주변화(numerical marginalization)를 통해 가능하다는 점을 강조했습니다. 또한 MTP 헤드 추가 시의 도전과제와 공동 훈련(joint training)의 중요성에 대해서도 논의하고 있습니다.

- **Technical Details**: 본 연구는 NTP를 위해 훈련된 트랜스포머의 MTP 기능을 다룹니다. MTP는 주어진 부분 입력 시퀀스에 대해 인접한 여러 토큰을 병렬로 생성하는 것을 의미합니다. 논문은 정보 이론적 기준을 사용하여 MTP의 가능성을 탐구하였고, 여러 모델 패밀리 및 크기에 대해 평가를 수행했습니다. 이는 MTP가 기존의 NTP에 비해 더 효율적인 방법임을 시사합니다.

- **Performance Highlights**: MTP는 자가 추측적 디코딩(self-speculative decoding)을 통해 최대 3.6배의 속도 향상을 가져오는 것으로 나타났습니다. 그러나 MTP 헤드를 짜임새 있게 추가하고 공동 훈련을 실시하더라도 성능이 여전히 기존의 기준에는 미치지 못하는 한계가 있음이 발견되었습니다. 이러한 결과는 MTP의 개선 여지가 많은 영역임을 보여주며, 향후 연구에서 다루어야 할 필요성이 있습니다.



### Rethinking Evaluation Metrics for Grammatical Error Correction: Why Use a Different Evaluation Process than Human? (https://arxiv.org/abs/2502.09416)
Comments:
          4 pages, 2 figures

- **What's New**: 이번 연구에서는 문법 오류 수정(GEC) 시스템의 자동 평가 지표와 인간 평가 사이의 간극을 해소하기 위한 집계 방법을 제안합니다. 기존의 자동 평가는 문장 별 절대 점수의 평균을 사용하여 평가 순위를 매기지만, 우리는 상대적 평가를 사용하여 인간 평가와 일치하도록 이 방법을 수정하였습니다. 이 접근방식은 SEEDA 벤치마크에서 여러 지표의 성능을 개선하는 데 기여하였으며, 심지어 BERT 기반 지표가 GPT-4보다 더 우수한 성능을 발휘하기도 했습니다.

- **Technical Details**: 문법 오류 수정 부문에서 사용되는 다양한 지표들, 예를 들어 edit-based 및 n-gram 기반 지표에 대해 실험을 수행하였습니다. 자동 평가는 통상적으로 문장 별 점수를 산출하고 최종적으로 이 점수들을 평균화하여 코퍼스 수준의 평가 점수를 결정하는 방식입니다. 그러나 우리는 TrueSkill과 같은 집계 방법을 통해 문장 별 점수를 쌍 비교 결과로 변환하고, 자동 평가에서도 동일한 절차를 적용하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식이 많은 지표의 순위 산출 능력을 개선하는 것으로 나타났습니다. 특히 BERT 기반의 자동 평가 지표는 GPT-4보다 더 나은 성능을 보이는 경우도 발견되었습니다. 또한, 새로운 평가 지표 개발 시 문장 수준의 상대 평가가 중요하다는 점을 강조하였습니다.



### SQuARE: Sequential Question Answering Reasoning Engine for Enhanced Chain-of-Thought in Large Language Models (https://arxiv.org/abs/2502.09390)
Comments:
          14 pages

- **What's New**: 이번 논문에서는 SQuARE(Sequential Question Answering Reasoning Engine)라는 새로운 방식의 prompting 기술을 소개하며, 이는 LLMs(대형 언어 모델)가 스스로 질문을 생성하고 답변하게 하여 더 깊이 있는 사고를 촉진하는 것을 목표로 합니다. 기존의 chain-of-thought(CoT) 접근 방식에 비해, SQuARE는 복잡한 질문을 여러 단계를 통해 더 철저히 탐색할 수 있도록 돕습니다.

- **Technical Details**: SQuARE는 모델에게 N개의 하위 질문-답변 쌍을 생성하도록 유도하며, 이를 통해 최종 쿼리에 도달하기 전에 다양한 주제를 탐색합니다. 또, 이 방법은 기존의 다른 prompting 기법들과 결합하기 쉽고, 하위 질문 수(N)를 조정함으로써 탐색의 깊이와 계산 비용을 조절할 수 있습니다. 실험은 Llama와 GPT-4o 모델을 사용하여 다양한 QA 데이터 세트에서 실시되었습니다.

- **Performance Highlights**: SQuARE는 TriviaQA, HotpotQA 및 ASQA 데이터 세트에서 기존 방법들을 초과하는 성과를 보여주었습니다. 특히 Llama 3.2 3B 모델을 사용할 때, TriviaQA에서 6.5% 향상된 88.5%의 성과를 기록하였고, GPT-4o를 이용한 경우에도 경쟁력 있는 결과를 보였습니다. SQuARE는 특히 작은 모델에서 최종 답변 품질을 크게 개선하는 데 기여했습니다.



### Truth Knows No Language: Evaluating Truthfulness Beyond English (https://arxiv.org/abs/2502.09387)
Comments:
          13 pages, 5 figures, 8 tables

- **What's New**: 이번 논문에서는 TruthfulQA 벤치마크의 전문 번역 확장을 소개합니다. 이 확장판은 바스크어, 카탈루냐어, 갈리시아어, 스페인어로 번역되었습니다. 지금까지 대형 언어 모델(LLMs)에 대한 진실성 평가가 주로 영어로 이루어졌는데, 이제는 다양한 언어에서의 LLM의 진실성 유지 능력을 평가할 수 있는 기회를 갖게 되었습니다.

- **Technical Details**: 새롭게 확장된 데이터셋은 TruthfulQA의 문제들을 각 언어에 맞게 번역한 것으로, 기본 대답, 올바른 대답 집합, 부정확한 대답 집합이 포함되어 있습니다. 논문에서는 언어별 인적 평가, 다중 선택 기반 자동 평가 및 LLM-as-a-Judge 점수를 통합하여 총 12개의 최첨단 LLM을 평가했습니다. 평가 방법론은 사람의 판단과 밀접한 관계를 가지며, 맥락과 시간에 의존하지 않는 질문이 LLM의 진실성 평가에 더 효과적임을 보여줍니다.

- **Performance Highlights**: 실험 결과에 따라 대부분의 LLM은 영어에서 가장 진실하고 바스크어에서는 가장 낮은 성능을 보였습니다. 그러나 언어별 진실성 차이는 예상보다 작았습니다. 또한, LLM-as-a-Judge 메트릭이 사람 평가와 더 잘 연관되며, 정보성이 진실성 평가에 중요한 역할을 한다는 점을 강조했습니다. 이는 기계 번역이 진실성 벤치마크의 다국어 확장에 유효한 접근 방식이 될 수 있음을 보여줍니다.



### Beyond English: The Impact of Prompt Translation Strategies across Languages and Tasks in Multilingual LLMs (https://arxiv.org/abs/2502.09331)
Comments:
          Accepted for NAACL findings 2025

- **What's New**: 이번 연구는 다양한 다국어 환경에서 최적의 사전 번역(pre-translation) 전략을 발견하는 것을 목표로 합니다. LLM(대형 언어 모델)의 다국어 처리 능력이 증대되고 있음에도 불구하고, 사전 번역의 방법론은 체계적인 평가가 부족하여 불확실한 점이 많았습니다. 본 논문에서는 선택적 사전 번역(selective pre-translation)이라는 보다 정교한 접근 방식을 통해, LLM이 제공하는 다양한 자연어 처리 작업에서 성능 향상을 도모합니다.

- **Technical Details**: 본 연구는 프롬프트를 네 가지 기능적 부분으로 나누고, 이들 각각을 사전 번역할 수 있는지를 평가하여, 다국어 설정에서의 프롬프트 구성을 체계적으로 분석합니다. 이 네 가지 부분에는 지시문(instruction), 맥락(context), 예시(examples), 그리고 출력(output)이 포함됩니다. 실험은 35개 언어와 다양한 자연어 처리 작업(예: QA, NLI, NER, Abstractive Summarization)에 걸쳐 진행되었으며, 다양한 전이 학습(transfer learning) 모델을 사용하여 평가되었습니다.

- **Performance Highlights**: 연구 결과, 선택적 사전 번역 전략이 기존의 사전 번역 및 직접 추론(direct inference) 방식보다 일관되게 우수한 성능을 나타냈습니다. 특히, 출력이 제공된 맥락과 겹치는 부분이 있는 추출적 작업에서 모델이 맥락 언어에 무관심하거나 저자원 언어의 경우 원본 언어를 선호하는 경향을 보였습니다. 이러한 성과는 LRM의 다국어 환경에서의 적용 가능성을 높이고, 사전 번역의 질이 모델 성능에 미치는 영향을 명확히 하는 데 기여합니다.



### A Judge-free LLM Open-ended Generation Benchmark Based on the Distributional Hypothesis (https://arxiv.org/abs/2502.09316)
Comments:
          13 pages

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)의 개방형 텍스트 생성 성능을 평가하기 위한 새로운 벤치마크를 제안합니다. 기존의 인간 판단이나 LLM을 심사로 사용하는 방법에 의존하지 않고, n-gram 통계 및 규칙을 기반으로 한 새로운 측정 기준을 도입합니다. 이를 통해 50개의 질문 및 기준 답변 세트를 활용하여 유창성(Fluency), 진실성(Truthfulness), 유용성(Helpfulness) 세 가지 새로운 메트릭을 제시합니다. 이 벤치마크는 GPT-4o 기반 평가와 강한 상관관계를 보이며, 상당히 적은 계산 자원으로 효과적인 평가를 가능하게 합니다.

- **Technical Details**: LLMs의 개방형 생성 능력을 측정하기 위한 이 벤치마크는 n-gram 기반의 판단 메트릭스를 사용하여 효율적이고 최소한의 계산 자원으로 평가합니다. 연구진은 자체적으로 설계한 50개의 질문을 기반으로 언어, 사회, 수학, 과학, 예술문화, 건강, 정보학 등 여러 과목을 포함한 범위를 설정하였습니다. 구축된 질문들은 약 100자의 응답이 가능하도록 디자인되었으며, 여러 사람에 의해 수집된 가능한 높은 확률의 답변을 모아 기준 답변 세트를 만들어 안정적인 평가가 이루어지도록 하였습니다.

- **Performance Highlights**: 제안된 벤치마크는 LLM들 간의 개방형 생성 품질을 정확하게 평가하는 동시에 높은 성능의 LLM-as-a-judge 접근 방식이 필요로 하는 계산 자원에 비해 현저히 낮은 자원 요구사항으로 보입니다. 모든 결과물과 코드는 GitHub를 통해 공개되어 연구자와 개발자들이 쉽게 접근할 수 있도록 하였습니다. 이러한 접근 방식은 LLM의 발전을 평가하고 향후 연구에 기여할 수 있는 기초 자료로 활용될 것입니다.



### When the LM misunderstood the human chuckled: Analyzing garden path effects in humans and language models (https://arxiv.org/abs/2502.09307)
- **What's New**: 이 논문은 현대 대형 언어 모델(LLMs)과 인간의 문장 이해 능력을 비교합니다. 특히, LLM과 사람이 기존의 연구가 다루지 않았던 garden-path 문장을 처리하는 방식을 자세히 분석하였습니다. 연구 결과, 특정 구문 구조에서 LLM과 인간 모두 유사한 어려움을 겪으며, 몇 가지 모델은 인간 이해와 높은 상관관계를 보였습니다.

- **Technical Details**: 연구는 LLM과 인간이 정확히 동일한 과제에 응답하도록 설계되었습니다. 이를 위해 garden-path 문장을 다루는 구문 컴프리헨션(comprehension) 질문과 다양한 심리 언어학적 가설을 제안하였습니다. 주요 가설은(a) 문장의 구문 재분석의 어려움, (b) 문장의 제안 객체의 타당성, (c) 동사의 전이성(transitivity)입니다. 이러한 시도는 LLMs가 인간의 느림 또는 오해를 어떻게 경험하는지 이해하는 데 중요합니다.

- **Performance Highlights**: LLMs의 문장 이해 성능은 인간과 유사한 방식으로 출현하였습니다. 가장 성능이 우수한 모델에서조차 garden-path 문장의 이해 정확도가 78%에 불과했으며, 여기서 고급 LLM들이 인간 행동과 더 유사한 경향을 보였습니다. 추가로 파라프레이징(paraphrasing) 및 텍스트-이미지 생성(text-to-image generation) 작업을 통해 이러한 결과가 LLM의 독해에서도 유사하게 나타남을 확인했습니다.



### SparQLe: Speech Queries to Text Translation Through LLMs (https://arxiv.org/abs/2502.09284)
- **What's New**: 이 연구에서는 Large Language Models (LLMs)와 음성 표현의 통합을 통해 다중 모달 처리 및 음성 이해를 원활하게 할 수 있는 새로운 접근 방식을 소개합니다. 새롭게 제안된 SparQLe 모델은 self-supervised 학습에서 비롯된 음성 표현을 활용하여 instruction-tuned LLM과 음성을 연결하는 효율적인 방법을 제공합니다. 이 접근 방식은 입력 음성의 의미를 보존하면서 사전 훈련된 음성 인코더와 LLM을 통합할 수 있는 가능성을 보여줍니다.

- **Technical Details**: SparQLe 모델은 음성 표현을 질의하는 데 필요한 정보를 추출하고 이를 사전 훈련된 LLM에 전달하는 기능을 효율적으로 수행합니다. 이 모델은 HuBERT를 음성 인코더로 사용하고, 영어 데이터로 사전 훈련한 후 영어와 프랑스어로 혼합된 데이터를 사용하여 세부 조정을 진행합니다. 모델은 피드 포워드 네트워크와 self-attention을 사용하여 모달리티 어댑터에서 효과적으로 작업을 처리합니다.

- **Performance Highlights**: 실험에서는 MuST-C와 LibriSpeech 데이터 세트를 사용하여 Automatic Speech Translation (AST) 작업을 평가했습니다. 번역 성공률을 높이는 다양한 모달리티 정렬 목표를 설정하여 성능을 극대화하였으며, 모델은 주어진 오디오를 기반으로 텍스트를 생성하는 데 있어 효과적임을 입증했습니다. 이 연구 결과는 다중 모달 음성 이해 애플리케이션에서의 활용 가능성을 확대하는 데 기여할 것입니다.



### The Joint Entity-Relation Extraction Model Based on Span and Interactive Fusion Representation for Chinese Medical Texts with Complex Semantics (https://arxiv.org/abs/2502.09247)
- **What's New**: 본 논문에서는 CH-DDI라는 데이터셋을 구축하여 중국 의료 텍스트에서의 엔티티(entities) 및 관계(relation) 추출을 통합적으로 수행하는 모델을 제안합니다. 이를 통해 복잡한 문맥 의미를 더 잘 포착하고자 하며, 기존 방법의 정보 교환 비효율성을 개선하기 위한 상호작용 융합 표현 모듈도 도입합니다. 실험 결과, 제안된 모델이 엔티티 인식에서 96.73%의 F1-score를, 관계 추출에서 78.43%의 성능을 기록하는 등 뛰어난 일반화 능력을 보였습니다.

- **Technical Details**: 저자는 attention 메커니즘을 활용하여 장기 종속성(long-range dependencies)을 포착하는 SEA 모듈을 제안합니다. 이 모델은 Cross Attention을 통해 엔티티 인식과 관계 추출 간의 양방향 정보 교환을 가능하게 하며, BiLSTM을 통해 특성(feature) 추출을 개선합니다. 모델은 다섯 가지 주요 구성 요소로 구성되어 있으며, encoder 모듈은 사전 훈련된 BERT를 사용합니다.

- **Performance Highlights**: CH-DDI 데이터셋에서 제안된 모델은 엔티티 인식에서 96.73%의 F1-score를 달성하였고, 관계 추출에서는 78.43%의 성능을 보였습니다. 공통의 CoNLL04 데이터셋에서도 엔티티 인식 정밀도 89.54%와 관계 추출 정확도 71.64%를 기록하여, 두 데이터셋 모두에서 최상의 성과를 달성했습니다. 이러한 성과는 제안된 모델의 효율성과 우수한 성능을 입증합니다.



### Answer Set Counting and its Applications (https://arxiv.org/abs/2502.09231)
Comments:
          In Proceedings ICLP 2024, arXiv:2502.08453

- **What's New**: 본 논문은 Answer Set Programming (ASP)에 초점을 맞추어 정확한 (exact) 및 근사적인 (approximate) 방법론을 탐구하였습니다. 이를 통해 sharpASP라는 정확한 ASP 카운터를 개발하였으며, 이는 고유의 압축된 (compact) 인코딩을 활용하여 기존의 비효율적인 인코딩 방식에 비해 효율성을 크게 향상시켰습니다.

- **Technical Details**: sharpASP는 제안된 새로운 ASP 카운터로, 기존 방법들보다 특히 제안된 인코딩 방식에서 효율적인 성능을 보입니다. 또한, ApproxASP라는 근사 ASP 카운터를 제안하여, Gauss-Jordan elimination을 clingo ASP 솔버에 통합하여 해시 기반으로 카운팅을 수행합니다.

- **Performance Highlights**: sharpASP는 여러 벤치마크에서 현재의 ASP 카운터들보다 뛰어난 성능을 나타내었습니다. 또한, ApproxASP는 전통적인 신뢰도 추정기 및 #SAT 기반 방법보다 우월한 성능을 보여주며 네트워크 신뢰도 추정과 같은 실용적인 응용에서도 높은 효율성을 입증하였습니다.



### Thinking beyond the anthropomorphic paradigm benefits LLM research (https://arxiv.org/abs/2502.09192)
- **What's New**: 이 논문에서는 인공지능 언어 모델(LLM) 연구에서 인류적 특성을 기술에 부여하는 비판적 분석을 통해 인간 유사성에 바탕을 둔 개념들이 연구 방향에 미치는 영향을 조명합니다. 그들은 25만 개 이상의 연구 초록을 분석하여 인류적 용어의 사용이 증가하고 있음을 시사합니다. 논문은 이러한 개념들이 LLM 연구에 어떻게 제한적일 수 있는지를 논의하며, 이를 벗어난 새로운 이해 방안을 제시합니다.

- **Technical Details**: 인류적 개념들이 LLM 연구에서 긍정적인 사례들을 만들어낸 것도 사실이지만, 이들 개념이 인공지능 시스템에 대한 정확한 적용을 방해할 수 있음을 강조합니다. 연구팀은 LLM 개발 및 배포 생애주기 전반에 걸친 5가지 핵심 가정을 규명하고, 이들 가정을 도전할 수 있는 대안을 제시합니다. 이러한 분석은 LLM의 연구 및 개발 방법론을 재편성하는 데 기여할 것입니다.

- **Performance Highlights**: 새로운 연구 방향 제시 및 인류적 가정 너머의 접근 방식에서 LLM의 효율성을 강조하며, 인간 중심의 벤치마크를 넘어서는 측정을 제안합니다. 연구 결과는 인공지능 모델의 행동 이해 및 사용자 간 상호작용을 더 효과적으로 개선할 수 있는 방법을 제시합니다. 이는 LLM 연구에 있어 단순한 유사성에 의존하지 않고, 보다 많은 발전의 가능성을 열어줄 것입니다.



### Matina: A Large-Scale 73B Token Persian Text Corpus (https://arxiv.org/abs/2502.09188)
- **What's New**: 이번 연구에서는 페르시아어를 위한 새로운 데이터셋인 Matina Corpus를 소개합니다. 이 데이터셋은 72.9B 토큰으로 구성되어 있으며, 높은 품질을 보장하기 위한 철저한 전처리(preprocessing) 및 중복 제거(deduplication) 과정을 거쳐 생성되었습니다. Matina Corpus는 다양한 자료 출처를 포함하여 기존의 페르시아어 데이터셋들보다 더 나은 데이터 품질을 가지고 있어, NLP 모델의 성능 향상에 기여할 것입니다.

- **Technical Details**: Matina Corpus는 블로그와 뉴스 기사를 주로 사용하는 기존 데이터셋들과 달리, 다양한 새롭게 수집된 자료를 포함합니다. 데이터셋은 두 가지 주요과정인 전처리와 중복 제거의 고유한 절차를 통해 구축되었으며, 이를 통한 훈련으로 transformer 기반 모델들이 페르시아어 NLP 작업에서 성능의 개선을 보였습니다. 데이터셋의 소스에는 최신 페르시아어 위키백과 업데이트와 Madlad, CulturaX가 포함되어 있습니다.

- **Performance Highlights**: Matina Corpus의 도입으로 XML-RoBERTa 모델을 재훈련하면서, 감정 분석, 텍스트 감정 인식 및 개체 인식에서 해당 모델의 성능이 현저히 향상되었습니다. 또한, LLaMA 3.1 8B 모델을 사용해 페르시아어 이해도를 높였으며, 이로 인해 다국어 모델의 페르시아어 처리 능력도 개선되었습니다. 전체적으로 이 데이터셋은 페르시아어 NLP의 발전을 위해 기여할 수 있는 중요한 자원입니다.



### RefineCoder: Iterative Improving of Large Language Models via Adaptive Critique Refinement for Code Generation (https://arxiv.org/abs/2502.09183)
Comments:
          work in process

- **What's New**: 이 논문은 기존의 교사 모델 모방에서 벗어나 Adaptive Critique Refinement (ACR)이라는 새로운 방법론을 제안합니다. ACR은 모델이 스스로 생성한 코드를 외부 비평과 함께 개선하도록 하여, 교사 모델의 코드 응답을 직접적으로 모방하는 대신에 자기 정제(self-refinement) 능력을 활용합니다. 이를 통해 더 적은 데이터로도 우수한 성능을 달성할 수 있도록 하는 RefineCoder 시리즈를 개발하였습니다.

- **Technical Details**: ACR은 LLM-as-a-Judge와 LLM-as-a-Critic을 활용하여 코드 응답의 품질 평가 및 저품질 코드 응답에 대한 비판 작용을 포함하는 복합 점수 시스템을 도입합니다. 이 방법은 자가 생성된 코드를 점수화하고 비평한 결과에 따라 새로운 샘플을 구성하는 과정을 포함하며, 이를 통해 코드 생성 능력을 지속적으로 발전시킵니다. RefineCoder 모델들은 세 번의 반복 과정을 통해 코드 생성 역량을 개선하여 다양한 벤치마크에서 인상적인 결과를 보여주었습니다.

- **Performance Highlights**: RefineCoder 시리즈는 HumanEval, MBPP, LiveCodeBench, BigCodeBench-hard와 같은 여러 코드 생성 벤치마크에서 뛰어난 성능 개선을 기록하였습니다. RefineCoder-DS-6.7B는 평균 pass@1에서 2.4p, RefineCoder-QW-7B는 3.0p의 향상을 이뤄냈습니다. ACR을 통한 iterative 방식이 코드 생성 성능을 지속적으로 향상시키고 있으며, 동등한 크기의 기존 모델들에 비해 더 적은 데이터로도 경쟁력 있는 성능을 달성하였습니다.



### Musical Heritage Historical Entity Linking (https://arxiv.org/abs/2502.09168)
Comments:
          To appear in Artificial Intelligence Review Journal

- **What's New**: 이 논문은 역사적 음악 관련 문서에서 명명된 개체를 인식하고 분류 및 연결하는 새로운 벤치마크인 MHERCL(Musical Heritage named Entities Recognition, Classification and Linking)을 소개합니다. 이 데이터셋은 가장 유명한 지식 베이스(KB)에서 미비하거나 누락된 명명된 개체를 포함하고 있으며, 역사적인 텍스트의 복잡한 특성을 다루기 위해 설계되었습니다.

- **Technical Details**: MHERCL 데이터셋은 고전 음악에 관한 역사적 정기간행물에서의 문장에서 수작업으로 주석을 다는 방식으로 구성되었습니다. 연구는 엔티티 링크(Entity Linking) 작업을 위해 여러 최신 모델을 실험하고, 역사적 문서에서의 성능 저하 문제를 해결하기 위해 비지도(unsupervised) EL 모델과 지식 그래프(Knowledge Graph)를 활용하는 방법을 제시합니다.

- **Performance Highlights**: 실험 결과, 비지도 기술과 지식 그래프를 바탕으로 한 논리적 제약(logical constraints)을 활용함으로써 역사적 문서에서의 엔티티 링크 성능이 크게 개선됨을 보였습니다. 특히, NIL 링크(지식 베이스에 없는 엔티티) 예제를 처리할 수 있는 모델의 필요성이 강조되며, C-BLINK라는 새로운 모델이 이러한 요구를 충족하기 위해 제안됩니다.



### Improving TCM Question Answering through Tree-Organized Self-Reflective Retrieval with LLMs (https://arxiv.org/abs/2502.09156)
- **What's New**: 본 연구에서는 기존의 Traditional Chinese Medicine (TCM) 분야에서 효율적인 retrieval-augmented generation (RAG) 프레임워크가 부족한 문제를 해결하기 위해 새로운 Tree-Organized Self-Reflective Retrieval (TOSRR) 프레임워크를 소개합니다. TOSRR은 지식 기반을 계층적으로 구성한 트리 구조를 이용하여 질문 답변 성능을 향상시키고자 합니다.

- **Technical Details**: TOSRR 프레임워크는 자가 반영(self-reflection) 기법을 활용하여 지식 기반에서 정보를 검색하고, TCM 관련 데이터셋을 통해 성능을 평가합니다. 질문은 TCM Medical Licensing Examination (MLE)와 대학 Classics Course Exam (CCE)에서 랜덤으로 선택되었습니다. 이 프레임워크는 GPT-4와 결합하여 사용됩니다.

- **Performance Highlights**: TOSRR 프레임워크는 TCM MLE 벤치마크에서 19.85%의 절대 정확도 향상을 달성하고, CCE 데이터셋에서 27%에서 38%로 Recall 정확도를 개선하였습니다. 또한 수동 평가 결과 안전성, 일관성, 설명 가능성, 준수성, 일관성 등 다양한 측면에서 총 18.52 포인트 향상되었습니다.



### A Novel Dialect-Aware Framework for the Classification of Arabic Dialects and Emotions (https://arxiv.org/abs/2502.09128)
- **What's New**: 이 연구는 아랍어의 방언과 감정 인식에 관한 기존 연구의 한계를 극복하기 위해 개발된 새로운 프레임워크를 제안합니다. 이 프레임워크는 아랍어 텍스트에서 방언과 감정을 식별하고 예측하는 기능을 제공합니다. 특히, 방언 인식에 맞춤화된 새로운 감정 어휘집을 생성할 수 있는 기능이 포함되어 있습니다.

- **Technical Details**: 제안된 프레임워크는 세 가지 모듈로 구성되어 있습니다: 텍스트 전처리 모듈(text-preprocessing module), 분류 모듈(classification module), 그리고 새로운 방언 인식 감정 어휘를 구축할 수 있는 클러스터링 모듈(clustering module)입니다. 이 연구는 이를 통해 아랍어의 다양한 방언에 대한 새로운 감정 어휘를 생성했습니다.

- **Performance Highlights**: 프레임워크는 아랍어 방언 분류에서 88.9%의 정확도를 달성하였으며, 이는 최신 연구 결과를 6.45% 포인트 초과하는 성과입니다. 또한 이 프레임워크는 이집트 방언과 걸프 방언에서 각각 89.1%와 79%의 감정 인식 정확도를 기록했습니다.



### The influence of visual and linguistic cues on ignorance inference in Vision-Language Models (VLMs) (https://arxiv.org/abs/2502.09120)
Comments:
          13 pages, 3 figures, 3 tables

- **What's New**: 이 연구는 Vision-Language Models (VLMs)가 시각적 및 언어적 단서와 함께 무지 함축(ignorance implicatures)을 처리하는 방식을 조사했습니다. 특히, 우리는 정밀한 맥락과 대략적인 맥락, 그리고 수식자 유형들(예: bare numerals, superlative modifiers, comparative modifiers)이 각각 화용론적(pragmatic) 및 의미론적(semantic) 요소에 미치는 영향을 집중적으로 분석했습니다. 결과적으로 모델들은 언어적 단서에는 민감하게 반응했으나, 시각적 단서의 무지 함축 처리는 인간처럼 수행하지 못했습니다.

- **Technical Details**: 연구 방법론으로는 시각적으로 기반 설정을 바탕으로 하는 진리 값 판단 과제를 수행했으며, 이를 위해 GPT-4o와 Gemini 1.5 Pro 모델이 활용되었습니다. 실험 결과, 두 모델은 언어적 단서에 대한 민감성을 보였지만, 시각적 맥락에서의 무지 함축을 처리하는 데 있어 약한 반응을 보였습니다. 특히, 수식자 중에서 최상급 수식(superlative modifiers)이 무지 함축과 더 강한 연관성을 가지며, 의미론적 관점을 뒷받침하는 것으로 확인되었습니다.

- **Performance Highlights**: 이 연구는 VLMs가 언어-비전 정보를 맥락에 따라 처리하는 데 있어서 인류와 유사한 화용적 추론을 수행하기 위한 추가 발전의 필요성을 강조합니다. 현재의 모델들은 맥락 의존적인 화용적 현상을 적절히 처리하지 못하는 한계를 보였으며, 이는 향후 VLMs을 개선하는 데 중요한 문제로 남아있습니다. 이러한 발견은 VLMs의 기능과 한계를 더 잘 이해하는 데 기여하며, 인간과 같은 추론 능력을 갖춘 AI 모델 개발의 필요성을 시사합니다.



### A Hybrid Transformer Model for Fake News Detection: Leveraging Bayesian Optimization and Bidirectional Recurrent Un (https://arxiv.org/abs/2502.09097)
Comments:
          6 pages, 7 figures

- **What's New**: 본 논문에서는 Bayesion 알고리즘을 통합한 Bidirectional Gated Recurrent Unit (BiGRU)와 최적화된 Transformer 모델을 제안하고, 이를 가짜 뉴스 분류에 최초로 적용합니다. 특히, TF-IDF 방법을 사용하여 뉴스 텍스트에서 특징을 추출하고 이를 숫자 표현으로 변환하여 후속 기계 학습 작업에 활용합니다. 이러한 접근 방식은 가짜 뉴스 탐지의 수단으로 혁신적인 기여를 합니다.

- **Technical Details**: 두 가지 실험 세트를 통해 가짜 뉴스 탐지 및 분류를 진행했습니다. 첫 번째는 BiGRU로만 최적화된 Transformer 모델을 사용하고, 두 번째는 BiGRU 기반 Transformer에 Bayesian 알고리즘을 추가합니다. 실험 결과는 BiGRU 최적화 Transformer가 훈련 세트에서 100% 정확도를 달성하며, 테스트 세트에서도 99.67%의 정확도를 보입니다.

- **Performance Highlights**: Bayesian 알고리즘 추가 후에도 훈련 세트에서 100% 정확도를 유지하면서 테스트 세트의 정확도가 99.73%로 소폭 향상되었습니다. 이는 Bayesion 알고리즘이 모델의 정확도를 0.06% 증가시켰음을 나타내며, 빠른 수렴 속도와 함께 우수한 분류 능력을 보여줍니다. 최적화된 Transformer 모델은 정보 과부하 시대에서 가짜 뉴스의 확산을 저지하기 위한 강력한 기술적 수단을 제공합니다.



### A Hybrid Model for Few-Shot Text Classification Using Transfer and Meta-Learning (https://arxiv.org/abs/2502.09086)
- **What's New**: 본 논문은 자연어 처리(Natural Language Processing, NLP) 기술의 발전에 따라 여러 응용 분야에서 널리 사용되는 텍스트 분류 작업을 다룹니다. 특히, 라벨이 붙은 데이터를 수집하는 것이 비용이 많이 들고 어려운 몇 가지 샷( Few-shot) 학습 시나리오에서 문제를 해결하기 위해, 전이 학습(Transfer Learning)과 메타 학습(Meta-Learning)을 기반으로 한 몇 가지 샷 텍스트 분류 모델을 제안합니다.

- **Technical Details**: 제안된 모델은 사전 훈련된(pre-trained) 모델의 지식을 활용하여 전이하며, 메타 학습 메커니즘을 통해 few-sample 작업에서 모델의 빠른 적응력을 최적화합니다. 또한 여러 비교 실험(comparative experiments)과 제거 실험(ablation experiments)을 통해 이 방법의 효과를 입증하였습니다.

- **Performance Highlights**: 실험 결과에 따르면, few samples와 medium samples 조건에서 전이 학습과 메타 학습 기반 모델이 전통적인 기계 학습(machine learning) 및 심층 학습(deep learning) 방법에 비해 유의미한 성능 향상을 보였습니다. 제거 실험은 모델 성능에 대한 각 구성 요소의 기여도를 분석하고, 모델 정확도를 향상시키는 데 있어 전이 학습과 메타 학습의 중요한 역할을 확인하였습니다.



### CoSER: Coordinating LLM-Based Persona Simulation of Established Roles (https://arxiv.org/abs/2502.09082)
- **What's New**: 최근 대규모 언어 모델(LLMs)의 발전으로 인하여 역할 수행 언어 에이전트(RPLA)가 주목받고 있습니다. 그러나 기존 캐릭터를 재현하는 데 있어 진정한 데이터 세트의 부족과 부정확한 평가 방법이 큰 도전 과제로 남아 있습니다. 이 논문에서는 CoSER라는 고품질 데이터 세트와 공개 모델, 평가 프로토콜을 소개하여 효과적인 역할 수행 언어 모델을 개발하고자 합니다.

- **Technical Details**: CoSER 데이터 세트는 771개의 저명한 소설에서 17,966명의 캐릭터로부터 수집된 다채로운 데이터로 구성되어 있습니다. 여기에는 실제적인 대화, 캐릭터 경험 및 내부 생각 등이 포함되어 있습니다. 주어진 상황에 맞춘 연기(Given-Circumstance Acting) 기법을 통해 LLM을 훈련하고 평가하며, CoSER 8B와 CoSER 70B와 같은 첨단 역할 수행 LLM 모델을 개발하였습니다.

- **Performance Highlights**: CoSER 70B는 다양한 평가 및 기준에서 최신 성능을 달성하였으며, GPT-4o와 비교했을 때 뛰어난 정확도를 보였습니다. 특히, InCharacter와 LifeChoice 벤치마크에서 각각 75.80%와 93.47%의 정확도를 기록하여 CoSER 데이터 세트의 가치를 입증했습니다. 이 결과는 CoSER 데이터 세트가 RPLA 훈련과 평가에 큰 기여를 할 수 있음을 나타냅니다.



### Enhancing RAG with Active Learning on Conversation Records: Reject Incapables and Answer Capables (https://arxiv.org/abs/2502.09073)
- **What's New**: 이 논문에서는 Retrieval-Augmented Generation (RAG) 기술을 개선하기 위한 새로운 접근법인 AL4RAG를 제안합니다. 이는 언어 모델이 환각(hallucination) 응답을 피하면서도 정확한 응답을 제공할 수 있도록 돕기 위한 것입니다. 특히, 쿼리에서 환각이 발생할 가능성이 있는 샘플을 식별하고, 이를 거부하는 방법을 학습시키는 것을 목표로 합니다.

- **Technical Details**: AL4RAG는 기존의 기존 액티브 러닝(active learning) 방법을 개선하여 RAG 데이터의 독특한 패턴을 고려한 새로운 샘플 거리 측정 방법을 개발했습니다. 이를 통해 언어 모델은 적절한 대화 샘플을 선택하여 고품질의 주석 데이터셋을 구축할 수 있습니다. 이 연구는 RAG 시스템에 특화된 액티브 러닝 전략을 통해 강화된 모델 성능을 목표로 합니다.

- **Performance Highlights**: 본 연구가 제안하는 방법은 다양한 지표에서 기존 방법들보다 우수한 성능을 보였습니다. 이를 통해 모델의 환각 응답을 줄이고, 더 나아가 데이터셋의 질을 향상시키는 데 성공했습니다. 최종적으로 이 연구는 RAG 모델 최적화를 위한 혁신적인 전략을 제공하여 향후 연구의 기초가 될 수 있습니다.



### An Open Recipe: Adapting Language-Specific LLMs to a Reasoning Model in One Day via Model Merging (https://arxiv.org/abs/2502.09056)
Comments:
          9 pages

- **What's New**: 본 논문은 DeepSeek R1과 같은 고급 추론 능력을 언어 특화 대규모 언어 모델(LLMs)에 통합하기 위한 데이터 선택 및 모델 합병 방법론을 조사하고 있습니다. 특히, 태국어 LLM에 초점을 맞추어 언어 특정 LLM의 추론 능력을 향상시키는 동시에 특정 언어 기능은 유지하도록 하고 있습니다. 저자는 공개적으로 이용 가능한 데이터셋과 120달러의 컴퓨팅 예산으로도 이러한 목표를 달성할 수 있음을 보여주고 있습니다.

- **Technical Details**: 본 연구에서는 목표 언어(예: 태국어)와 장기 추론에 특화된 두 개의 모델을 선택한 다음, 모델의 내부 표현을 정제하여 일관성 있게 결합하는 두 단계 절차를 사용합니다. 첫 번째 단계에서는 Supervised Fine-Tuning(SFT) 기법을 통해 언어 모델의 성능을 강화하며, 이후 Ability-Aware Model Merging 기법으로 두 모델의 파라미터를 통합하여 새로운 모델을 생성합니다. 이 과정에서 태국어와 영어의 질문-해답 쌍을 연결하는 이중 언어 설정을 활용합니다.

- **Performance Highlights**: 실험에서는 Typhoon2 70B Instruct와 DeepSeek R1 70B Distill 모델을 사용하여 각각의 추론 능력과 언어 작업 성능을 평가하였습니다. 평가를 통해 언어 모델의 태국어 성능을 유지하면서도 DeepSeek R1 수준의 추론 능력을 도달하는 것이 가능함을 입증하였습니다. 이러한 결과는 고-resource 언어 중심의 모델이 아닌, 저-resource 언어를 위한 모델 개선의 중요성을 강조합니다.



### Typhoon T1: An Open Thai Reasoning Mod (https://arxiv.org/abs/2502.09042)
Comments:
          25 pages, 6 figures

- **What's New**: 본 논문은 Typhoon T1이라는 새로운 오픈 태국어 추론 모델 개발 프로젝트를 소개합니다. 이 모델은 최근 대형 언어 모델(LLMs) 위에 구축된 새로운 형태의 generative model로, 복잡한 작업을 수행하는 데 있어 성능을 향상시키는 장기적인 사고 과정을 생성합니다. 추가적으로, 태국어와 같은 낮은 자원 언어에서 추론 흔적을 생성하는 것에 대한 세부 사항이 부족함을 인식하고, 모델 개발 과정에서의 통찰력을 공유합니다.

- **Technical Details**: Typhoon T1은 감독된 세부 조정(supervised fine-tuning, SFT) 방법론을 사용하여 데이터셋과 모델 가중치를 개방하고, 길고 복잡한 사고 과정을 구성하는 방법에 대한 실험을 진행합니다. 모델 선택에서는 Typhoon 2 3B Instruct를 사용하여, 강화 학습(reinforcement learning, RL)의 불안정성을 회피하며 실험을 진행하였습니다. 또한, 세 가지 사고 포맷(구조화된 사고, 비구조화된 사고, 반구조화된 사고)에 대한 비교 분석을 실시하며, 모델의 성능 향상이 가능한지를 조사합니다.

- **Performance Highlights**: Typhoon T1 모델의 특징은 다양한 작업에 대한 추론을 할 수 있는 능력과 태국어 데이터 생성을 통해 낮은 자원 언어의 연구 기반을 선도할 수 있다는 것입니다. 향후 연구 방향으로는, 추론 모델의 성능을 개선하기 위한 냉각 구조화된 사고 포맷의 효과성을 평가하고 있으며, 연구가 오픈 데이터셋과 연계되어 지속적인 발전이 이루어질 것으로 기대됩니다.



### Diversity Enhances an LLM's Performance in RAG and Long-context Task (https://arxiv.org/abs/2502.09017)
- **What's New**: 본 논문에서는 대규모 언어 모델(LLMs)의 문맥 창 환경에서의 한계를 해결하기 위해 다채로운 정보 선택의 중요성을 강조합니다. 특히, 질문-답변(Q&A) 및 긴 문맥 요약을 포함한 작업에서 정보의 다양성을 통합하여 성능을 향상시킬 수 있음을 입증합니다. MMR(Maximal Marginal Relevance) 및 FPS(Farthest Point Sampling) 원칙을 바탕으로 한 접근 방식은 쿼리와의 유사성이 높은 내용을 선택하는 전통적인 방법의 한계를 극복합니다.

- **Technical Details**: 탐색 과정 중 다채로운 콘텐츠 선택의 중요성을 고려하며 MMR과 FPS를 활용한 방법론을 제시합니다. MMR은 선택된 항목 간의 유사성을 최대한 줄이면서 보상과 다양성 간의 균형을 맞추는 그리디 알고리즘을 기반으로 합니다. FPS는 원래의 3D 포인트 클라우드에서 다양성을 통해 포인트를 선택하는 기법으로, 문맥 창이나 보상 개념을 통합하여 MMR과 동일한 방식으로 사용될 수 있습니다.

- **Performance Highlights**: 실험 결과, 다양성을 포함한 콘텐츠 선택 과정은 relevant 문장이나 청크의 recall을 크게 향상시켰습니다. MMR과 FPS의 활용에 있어 MMR이 FPS보다 약간 더 우수한 recall을 유지하면서도 더 낮은 latency를 보여주었습니다. 특히, 선택된 문장 및 청크의 순서가 downstream 작업에서의 성능에 긍정적인 영향을 미치는 것으로 나타났습니다.



### Hope vs. Hate: Understanding User Interactions with LGBTQ+ News Content in Mainstream US News Media through the Lens of Hope Speech (https://arxiv.org/abs/2502.09004)
- **What's New**: 이번 논문은 3,161개의 미국 주요 케이블 뉴스 매체의 유튜브 뉴스 비디오에 게시된 1,419,047개의 댓글을 분석하여 LGBTQ+ 뉴스 콘텐츠와 사용자의 상호작용 방식을 조사합니다. 특히, 긍정적(희망 발언) 및 부정적 콘텐츠를 구분할 수 있는 세밀한 희망 발언 분류기를 구축하여 긍정적인 콘텐츠의 탐지를 꾀했습니다. 또한, LGBTQ+ 건강 전문가와의 협력을 통해 정치적으로 다양한 표본에서 주석이 부여된 3,750개의 데이터 세트를 생성했습니다.

- **Technical Details**: 저자들은 정치적 입장이 다양한 평가자가 주석을 달아, 각 주석이 긍정(희망 발언), 부정, 중립 및 관련 없는 콘텐츠로 분류된 데이터셋을 구성했습니다. 또한, 사용자 정치 신념이 LGBTQ+ 커뮤니티와 관련된 콘텐츠 평가에 미치는 영향을 분석했습니다. 이 연구는 특히 미국의 정치적 담론에서의 LGBTQ+ 논의와 연관된 감정 동 동 양상을 이해하는 데 중요한 인사이트를 제공합니다.

- **Performance Highlights**: 연구 결과는 정당별 보수적이며 진보적인 평가자 간의 긍정적 콘텐츠에 대한 인식 차이에 대한 통찰력을 제공합니다. 또한 평가자의 정치적 신념에 따라 선별된 모델 간의 불일치가 발생하며, 제로샷 대형 언어 모델(LLMs)이 자유주의적 평가자와 더 높은 일치를 보임을 보여줍니다. 이러한 결과는 LGBTQ+ 콘텐츠에 대한 주관적 평가가 표현되는 방식을 돕는 중요한 지침을 제공합니다.



### Tuning-Free Personalized Alignment via Trial-Error-Explain In-Context Learning (https://arxiv.org/abs/2502.08972)
Comments:
          NAACL 2025 Findings

- **What's New**: 이 연구에서는 Trial-Error-Explain In-Context Learning (TICL)이라는 튜닝(예를 들어, fine-tuning) 없는 방법을 제안합니다. TICL은 사용자의 스타일에 맞춘 개인화된 텍스트 생성을 위해 10개 이하의 예제만 사용하여 언어 모델을 개인화하는 새로운 접근법을 제공합니다. TICL은 모델이 생성한 부정 샘플과 설명을 통해 점진적으로 In-Context Learning 프롬프트를 확장합니다.

- **Technical Details**: TICL은 각 단계에서 Trial-and-Error Fine-tuning (TEFT) 방식과 튜닝이 필요 없는 프롬프트 확장 방법을 결합하여 동작합니다. 기존의 Supervised Fine-tuning (SFT) 대신 In-Context Learning (ICL)를 사용하는 과정으로, 예측된 텍스트와 사용자의 실제 텍스트 사이의 차이를 분석하여 생성된 텍스트에 부정적 샘플로 라벨을 붙입니다. 이를 통해 개인화된 텍스트 생성에서 성능을 극대화합니다.

- **Performance Highlights**: TICL은 GPT-4o와 Claude 3 Sonnet을 사용하여 다양한 개인화된 작성 데이터셋에서 실험을 수행한 결과, 경쟁 방식보다 높은 승률(87%)을 기록했습니다. 특히 TICL의 각 절차는 성능 향상에 기여하며, 설명 프로세스는 77%의 성능 향상에 기여하는 것으로 나타났습니다. TICL은 기존의 zero-shot 출력에서 관찰된 구조적 및 형식적인 구문에 대한 편향을 극복할 수 있는 가능성을 보여줍니다.



### Medicine on the Edge: Comparative Performance Analysis of On-Device LLMs for Clinical Reasoning (https://arxiv.org/abs/2502.08954)
- **What's New**: 이 논문에서는 모바일 디바이스에서의 Large Language Models (LLM)의 잠재력과 헬스케어를 위한 이점이 강조됩니다. 특히 클라우드 서비스에 대한 의존도를 줄여 민감한 건강 데이터를 로컬에서 유지함으로써 프라이버시와 보안을 높입니다. 그러나 실제 의료 환경에서 LLM의 성능과 정확성은 충분히 연구되지 않았습니다.

- **Technical Details**: 연구진은 AMEGA 데이터셋을 사용하여 다양한 모바일 디바이스에서 공개된 LLM의 정확성, 계산 효율성 및 열 제한을 평가했습니다. HealthBench iOS 애플리케이션을 통해 자원 제약이 있는 스마트폰에서 LLM의 성능을 평가하며, Stanford Spezi LLM 모듈을 활용했습니다. LLM의 inference를 위해 MLX 프레임워크를 최적화하여 사용하였으며 각 모델은 AMEGA 벤치마크에 따라 점수화됩니다.

- **Performance Highlights**: 결과적으로 Phi-3 Mini와 같은 일반 목적의 소형 모델은 속도와 정확성 간의 균형을 잘 이루었으며, Med42와 Aloe와 같은 의료 세부 조정 모델은 가장 높은 정확성을 기록했습니다. 특히, 오래된 디바이스에서도 LLM을 배포할 수 있는 가능성이 보이며, 메모리 제약이 원시 처리 성능보다 더 큰 도전 과제로 여겨집니다. 이 연구는 실제 임상 추론에 맞춘 효율적인 인퍼런스 모델의 필요성을 강조하고 있습니다.



### Structured Convergence in Large Language Model Representations via Hierarchical Latent Space Folding (https://arxiv.org/abs/2502.08947)
- **What's New**: 이번 연구에서는 계층적 잠재 공간 접기(hierarchical latent space folding)라는 새로운 메커니즘을 도입하여 내부 표현을 동적으로 재구성하는 방법을 제안합니다. 이는 정적 제한이나 수동으로 정의된 클러스터링 메커니즘에 의존하는 대신, 다층적인 추상화 수준에서 반투명한 수렴 패턴을 강제하여 의미적으로 관련된 표현을 압축합니다. 이를 통해 의미의 맥락 의존성을 유지하면서도 계층적으로 서브스페이스로 재구성합니다.

- **Technical Details**: 제안된 접근 방식은 구조적 변환을 통해 토큰(Tokens) 임베딩을 반복적으로 조정하는 동적 접기 작업을 포함하며, 이는 순차 처리 작업에서 단기 및 장기 의존성에 영향을 줍니다. 이 프로세스는 정보의 정제에서 인지의 자연스러운 진전을 따라 내부 구조를 확립하여, 여러 레이어에서 단기와 장기 의존성을 모두 고려할 수 있도록 합니다. 실험 평가를 통해 기존 모델과 비교하여 정보 유지, 구조적 압축 및 레이어 간 일관성을 측정하며 표현 효율성을 높입니다.

- **Performance Highlights**: 실증적 평가 결과, 계층적 잠재 공간 접기가 표현의 변동성을 감소시켜 더 안정적인 perplexity 분포를 기여하며, 텍스트 생성의 예측 신뢰성을 높이는 것으로 나타났습니다. 또한, 계층적 조정으로 중요한 경로를 강화하고 비본질적인 지역에서 계산 비용을 줄이는 것을 통해, 모델의 해석 가능성을 향상시키는 효과가 있음을 보여주었습니다. 결론적으로, 제안된 방법은 모델의 효율성과 성능을 최적화하는 데 있어 중요한 기여를 할 수 있음을 입증했습니다.



### The Stochastic Parrot on LLM's Shoulder: A Summative Assessment of Physical Concept Understanding (https://arxiv.org/abs/2502.08946)
Comments:
          NAACL 2025 Main Conference. First 5 authors contributed equally. Project page: this https URL

- **What's New**: 본 논문은 LLMs(대형 언어 모델)의 진정한 이해 여부에 대해 정량적인 실험을 통해 검증하고자 합니다. 우리는 특히 Stochastic Parrot 현상, 즉 LLM이 단순히 연관 패턴을 바탕으로 반복하는지 여부를 평가하는 PhysCo라는 새로운 과제를 제안합니다. 이 과제는 물리적 개념 이해도를 측정하기 위해 설계되었습니다. 고민이 많았던 메모리 문제를 해결하기 위해 그리드 형식의 입력을 사용하여 다양한 이해 수준을 표현하고 있습니다.

- **Technical Details**: PhysiCo는 저급 이해(subtask)와 고급 이해(high-level understanding)라는 두 가지 하위 과제를 포함합니다. 저급 이해는 LLM의 기억 능력을 측정하는 자연어 형식 질문으로 구성되어 있으며, 고급 이해 과제는 추상 표현을 기반으로 상대적으로 심화된 이해를 평가하는 숙제입니다. 우리는 LLMs가 저급 과제에서 높은 정확도를 보이지만, 고급 과제에서는 인간에 비해 40% 가량 성능이 떨어진다는 두 가지 주요 결론을 도출했습니다.

- **Performance Highlights**: 실험 결과, 최신 LLMs는 저급 이해 과제에서 95% 이상의 정확도를 기록하였으나, 고급 이해 과제에서는 인간에 비해 평균 약 40% 낮은 정확도를 보였습니다. 이는 LLM이 진정한 개념 이해 능력에서는 한계를 나타내며, 새로운 그리드 형식이 아닌 그 자체의 고차원적 이해의 어려움이 원인임을 시사합니다. 본 연구는 LLM의 이해력을 측정하는 방법론적 기틀을 확립하고, LLM과 인간 간의 성능 격차를 명확히 보여줍니다.



### Beyond the Singular: The Essential Role of Multiple Generations in Effective Benchmark Evaluation and Analysis (https://arxiv.org/abs/2502.08943)
Comments:
          10 pages, 1 table, 4 Figures

- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 성능 평가 방법에 대한 새로운 접근 방식을 제시하고 있습니다. 기존의 평가 방식은 LLM의 본질적인 무작위성을 간과하는 경우가 많았으나, 본 연구에서는 계층적 통계 모델을 통해 이를 통합하여 보다 신뢰성이 높은 벤치마크(score)를 제공하고자 합니다. 이를 통해 개별 프롬프트(prompts)의 난이도를 정밀하게 측정하고 오류 감지 및 품질 관리를 위한 데이터 맵을 생성했습니다.

- **Technical Details**: 저자들은 LLM의 응답 생성 방식에 대해 deterministic한 greedy decoding과 stochastic한 random sampling 방법을 비교했습니다. 후자는 원래의 확률 분포에 따라 매 단계에서 토큰을 무작위로 샘플링하여 비결정적 출력을 생성하게 됩니다. 연구에서는 각 프롬프트에 대해 k개의 무작위 생성(random generations)을 통해 벤치마크 과정을 계층적 모델(hierarchical model)로 보았으며, 이를 통해 불확실성을 줄이고 정확한 평가를 할 수 있음을 보였습니다.

- **Performance Highlights**: 여러 번의 생성을 통해 벤치마크 점수의 정확도가 향상되며, variance가 감소하고 정확성의 세분화된 난이도 점수 ℙ(correct)도 도입되었습니다. 이는 프롬프트들의 난이도를 비교하고 잘못 표기되거나 모호한 프롬프트를 효과적으로 감지할 수 있는 도구로서의 가능성을 보여줍니다. 본 연구는 LLM 벤치마크 개발 과정에서의 방법론적 틀을 개선하고 실제 적용 가능성을 높이는 데 기여할 것입니다.



### CopySpec: Accelerating LLMs with Speculative Copy-and-Paste Without Compromising Quality (https://arxiv.org/abs/2502.08923)
Comments:
          33 pages, 18 figures, 19 tables

- **What's New**: 본 논문에서는 LLMs의 비효율성을 해결하기 위한 혁신적인 기술인 CopySpec을 소개합니다. CopySpec은 모델의 채팅 기록에서 반복되는 시퀀스를 식별하고, 이러한 패턴을 활용하여 후속 토큰을 원활하게 복사할 수 있게 하여 GPU 메모리를 추가로 요구하지 않습니다. 이를 통해 다양한 LLM과 데이터셋에서 значительные 속도 향상을 달성했습니다.

- **Technical Details**: CopySpec은 학습된 복사 메커니즘을 활용하여 입력의 특정 토큰 패턴을 감지하고 이를 재사용합니다. 이 과정은 전체 LLM을 통해 토큰을 한 번에 생성함으로써 반복적이거나 예측 가능한 출력에서의 계산 부담을 줄이는 데 기여합니다. 기술적으로 Roll Hash 메커니즘을 사용하여 계산 오버헤드를 최소화하면서 더 큰 토큰 블록을 추측할 수 있습니다.

- **Performance Highlights**: 다양한 벤치마크에서 CopySpec은 측정 가능한 성능 향상을 보여주며, MT-Redundant의 특정 카테고리에서는 3.08배, Speculative Decoding 보다 평균 49%의 추가 속도 향상을 기록했습니다. 이러한 결과는 CopySpec이 LLM의 효율성을 크게 개선할 수 있는 잠재력을 지니고 있다는 것을 보여줍니다.



### InfiniteHiP: Extending Language Model Context Up to 3 Million Tokens on a Single GPU (https://arxiv.org/abs/2502.08910)
Comments:
          21 pages

- **What's New**: InfiniteHiP는 긴 컨텍스트 처리를 위한 혁신적인 LLM 추론 프레임워크로, 모듈화된 계층형 토큰 프루닝 알고리즘을 통해 관련 없는 컨텍스트 토큰을 동적으로 제거하여 속도를 향상시킵니다. 이 방법은 입력의 내부 어텐션 패턴에 따라 여러 RoPE 조정 방법을 적용하여 일반적인 시퀀스에서의 일반화 가능성을 높입니다. 또한, 추론 중 GPU 메모리 압력을 줄이기 위해 키-값 캐시를 호스트 메모리로 오프로드하며, 이러한 특징으로 최대 3백만 개의 토큰을 처리할 수 있습니다.

- **Technical Details**: 제안된 InfiniteHiP 프레임워크는 모듈형 희소 어텐션 스킴을 포함하여 기능의 효율성을 높입니다. 이는 중요도가 낮은 컨텍스트에 대한 계산을 최소화하면서 각 모듈 내에서 개선된 병렬성으로 더 빠른 프루닝을 제공합니다. LRU 기반 캐시 정책으로 KV 캐시 오프로드를 최적화하고, RoPE 조정 전략을 조합하여 OOL(Out-of-Length) 일반화를 목표로 합니다.

- **Performance Highlights**: InfiniteHiP는 1백만 개의 토큰 컨텍스트에서 주목할 만한 18.95배의 속도 향상을 달성하며, FA2에 비해 단 3.34%의 VRAM만을 사용하여 3백만 개의 토큰 컨텍스트에서 7.24배의 속도 개선을 보여줍니다. 이 프레임워크는 훈련 과정 없이도 기존 Transformer 기반 LLM에 쉽게 적용 가능하며, 실제 환경에서의 유용성을 입증했습니다.



### Towards Automated Fact-Checking of Real-World Claims: Exploring Task Formulation and Assessment with LLMs (https://arxiv.org/abs/2502.08909)
- **What's New**: 이 연구에서는 대규모 언어 모델(LLM)을 활용한 자동 팩트 체크(Automated Fact-Checking, AFC) 시스템의 효용성을 검토합니다. 연구진은 3종의 레이블링 체계(2진, 3클래스, 5클래스)를 통해 다양한 사이즈의 Llama-3 모델(3B, 8B, 70B)을 평가하였고, 각 모델의 주장 분석, 진실성 예측 및 자세한 설명 생성을 위한 프레임워크를 제안합니다. 또한, 증거 통합이 모델 성능에 끼치는 영향을 강조하며, LLM을 활용한 AFC의 가능성을 보여줍니다.

- **Technical Details**: 연구에서 사용된 데이터셋은 PolitiFact에서 2007년부터 2024년까지 수집된 17,856개의 주장으로 구성되어 있습니다. 이 데이터는 제한된 웹 검색을 통해 수집된 증거와 함께 모델 성능을 평가하는 데 사용되었으며, 증거 통합이 모든 모델에서 성능을 향상시키는 데 기여함을 보여줍니다. TIGERScore라는 레퍼런스 프리(reference-free) 평가 지표를 활용하여 결과를 분석하였고, 더 큰 LLM이 작은 모델보다 항상 더 높은 분류 정확도 및 정당화 품질을 보였습니다.

- **Performance Highlights**: 연구 결과, 큰 LLM은 무작위 원샷(scenario) 상황에서 미세 조정을 한 LSLM(Small Language Models)과 유사한 성능을 보인 반면, LLM은 한계를 초과하여 지속적으로 높은 성능을 발휘하는 것으로 나타났습니다. 특히, 큰 모델이 보다 복잡한 레이블의 분별력에서 더 강력한 성능을 보여주었지만, 미세 조정이 필요없는 구성으로도 적절한 성능을 나타내는 것을 확인하였습니다. 이러한 결과는 LLM을 이용한 혹은 다른 데이터 구성 요소에서의 성능 향상을 위한 추가 연구 필요성을 강조합니다.



### Can Uniform Meaning Representation Help GPT-4 Translate from Indigenous Languages? (https://arxiv.org/abs/2502.08900)
- **What's New**: 이 논문은 Uniform Meaning Representation (UMR)을 GPT-4의 프롬프트에 통합하여 극히 저자원 언어인 네바호, 아라파호, 쿠카마에서 영어로의 번역 성능을 탐구합니다. UMR은 저자원 언어 기술 개발을 지원하는 의미 표현으로 설계되었으며, 해당 연구에서는 UMR을 포함한 여러 프롬프트 프로토콜을 사용하여 기계 번역의 효과를 분석합니다. 이 연구는 UMR을 통한 성능 향상의 가능성을 제시하며, 기계 번역에서 저자원 언어의 중요성을 부각시킵니다.

- **Technical Details**: 이 연구는 총 1,017개의 문장을 사용할 수 있는 네바호, 쿠카마, 아라파호와 같은 극저자원 언어의 번역 효율성을 평가합니다. 네 가지 프롬프트 프로토콜(Zero-shot, Zero-shot with UMR, Five-shot, Five-shot with UMR)을 비교하여 UMR이 번역 성능에 미치는 영향을 분석합니다. 각 프로토콜의 결과는 chrF와 BERTscore 두 가지 지표를 통해 평가되었습니다.

- **Performance Highlights**: 실험 결과, 거의 모든 테스트 사례에서 UMR을 통합한 프롬프트가 통계적으로 유의미한 성능 향상을 보였습니다. 특히 Five-shot with UMR 프로토콜이 전반적으로 가장 높은 성능을 나타내었으며, 이어서 Five-shot 프로토콜이 뒤를 이었습니다. 이러한 성능 증가는 UMR의 중요성을 강조하며, 저자원 언어 번역에서의 future applications 가능성을 보여줍니다.



### Communication is All You Need: Persuasion Dataset Construction via Multi-LLM Communication (https://arxiv.org/abs/2502.08896)
Comments:
          Accepted to NAACL 2025 Main Conference

- **What's New**: 이 논문은 다수의 LLM (Large Language Models)을 활용한 설득적 대화 생성 프레임워크를 제안합니다. 이 프레임워크는 고품질의 다양한 언어 콘텐츠를 자동으로 생성하고, 최소한의 인간 개입으로 설득 기술을 향상시킬 수 있는 가능성을 보여줍니다. 특히, 사회적 금기를 포함한 복잡한 시나리오에서도 자연스러운 언어 흐름과 전략적인 설득을 활용하는 능력이 강조됩니다.

- **Technical Details**: 제안된 프레임워크는 6개의 언어 에이전트 그룹을 포함하여 다중 에이전트 데이터 생성 및 주석화 과정을 수행합니다. 에이전트들은 각기 다른 역할을 맡아서 설득적 대화 생성을 효율적으로 처리하며, GPT-3.5와 GPT-4 모델을 사용해 대화의 질을 높입니다. 이론적인 대화 흐름과 다양한 설득 전략을 포함하는 대화 생성이 가능하며, 특정 요청에 따른 동적이고 구조화된 상호작용이 이루어집니다.

- **Performance Highlights**: 우리의 실험 결과, 프레임워크는 자연스러움, 언어 다양성, 논리적 일관성을 유지하며 모두 고품질의 대화를 지속적으로 생성했습니다. 특히, 설득력 평가와 관련하여 생성된 데이터는 전문가들의 판단과 높은 일치를 보였으며, 이는 단계 변화 분석의 유용성을 강조합니다. 다자간 대화를 포함한 다양한 구성에서도 프레임워크의 적응성과 일반화 능력이 검증되었습니다.



### LLM-Enhanced Multiple Instance Learning for Joint Rumor and Stance Detection with Social Context Information (https://arxiv.org/abs/2502.08888)
Comments:
          Accepted by ACM TIST

- **What's New**: 이번 연구는 기존의 주장(class) 레이블만을 온전히 활용하여 게시물의 태도(post stance)와 주장을 동시에 예측하는 새로운 LLM 기반 다중 인스턴스 학습(MIL) 접근법을 제안합니다. 이는 기존의 방법과 달리, 루머 검증을 위한 게시물 수준의 태도 레이블이 필요하지 않으며, 청구 진실성(claim veracity) 레이블만으로 약한 지도 학습(weakly supervised learning)을 수행합니다. 또한, 이 연구는 LLM을 활용하여 루머와 태도 탐지의 복잡한 상호작용을 개선하는 데 중점을 두고 있습니다.

- **Technical Details**: 제안된 방법은 다중 클래스 문제를 여러 개의 MIL 기반 이진 분류 문제로 변환하여 수행됩니다. 독창적인 계층적 주의 메커니즘(hierarchical attention mechanism)을 도입하여 각 이진 MIL 모델에서 얻은 결과를 집계하고, 최종적인 다중 클래스 결과를 만들어내는 것에 중점을 두었습니다. 실험 과정에서, 변경된 트리 기반 전송기(bottom-up/top-down tree transformers)를 사용하여 게시물 태도 표현(post stance representation)을 강화하고, LLM을 통해 청구 및 게시물 간의 상호작용을 활용하여 정보 전파 과정 중의 관계를 개선했습니다.

- **Performance Highlights**: 실험은 세 가지 루머 데이터셋과 두 가지 태도 데이터셋에서 수행되었으며, 제안된 방법이 루머와 태도 탐지에서 우수한 성능을 보여주었습니다. 특히, 루머의 진실성(진짜 또는 거짓)과 응답 게시물에서 표현된 태도 사이의 강력한 상관관계가 나타났습니다. 이 방법은 같은 맥락에서 기존 방법들과 비교할 때 에러를 줄이며 향상된 신뢰성을 제공합니다.



### BrainWavLM: Fine-tuning Speech Representations with Brain Responses to Languag (https://arxiv.org/abs/2502.08866)
Comments:
          15 pages, 8 figures

- **What's New**: 이 논문에서는 인공지능 음성 인코딩 모델이 인체의 뇌가 발화 언어 자극에 어떻게 반응하는지를 예측하는 데 있어 효과성을 높이기 위해 LoRA 방식으로 WavLM 기반 모델을 미세 조정하는 방법을 제안합니다. 이 미세 조정된 모델을 BrainWavLM이라고 명명했고, 뇌 인코딩 목표에 대해 효율적인 성능을 보임을 입증했습니다. 이러한 방법은 전통적인 선형 모델의 한계를 극복하고, 그렇지 않은 모델 대비 향상된 정확성을 제공합니다.

- **Technical Details**: 연구는 자연어 자극을 청취하는 동안의 fMRI 데이터를 사용하여 진행되었습니다. 3명의 참가자로부터 수집된 데이터는 스토리 내용을 담고 있으며, 평균적으로 17~19.7시간의 데이터를 분석합니다. 선형화된 인코딩 모델과 비선형 인코딩 모델을 조화롭게 적용하여, 각 뇌 영역에서의 정보 반응을 예측하는 과정에서 성능을 개선하였습니다.

- **Performance Highlights**: Fine-tuned된 BrainWavLM 모델은 평균 인코딩 성능을 크게 향상시키며, 안정성에서 우수한 결과를 보였습니다. 모델은 다양한 피험자 간의 일반화가 가능하며, 발화 자극에 대한 뇌의 반응을 잘 반영하는 능력을 보였습니다. 또한, 신경망 모델이 기존의 전처리 방법보다 더욱 효과적인 감독 신호를 제공할 수 있는 가능성을 제시합니다.



### Ask in Any Modality: A Comprehensive Survey on Multimodal Retrieval-Augmented Generation (https://arxiv.org/abs/2502.08826)
- **What's New**: 이 논문은 최근 발전된 Retrieval-Augmented Generation (RAG) 시스템, 특히 Multimodal RAG의 구조적이고 종합적인 분석을 제공하고 있습니다. 기존의 RAG 아키텍처가 주로 텍스트 정보를 중심으로 설계된 반면, Multimodal RAG는 텍스트, 이미지, 오디오, 비디오와 같은 다양한 형식을 통합하여 생성된 출력을 향상시킵니다. 이는 AI 시스템이 더 신뢰할 수 있고 유능하게 멀티모달 동적 외부 지식 데이터를 활용할 수 있는 기반을 마련합니다.

- **Technical Details**: 최근 Multimodal RAG의 발전은 다양한 데이터 소스를 통합하고 분석하는 능력을 촉진하며 정보의 전체적인 표현을 가능하게 합니다. 이 시스템은 특정 모달리티를 비교 평가하고, 교차 모달의 연관성을 해결하기 위한 특별한 도전 과제를 제시합니다. 또한 정보 검색 및 콘텐츠 생성 단계에서의 정확성을 높이기 위해 점진적 추론 과정을 도입하여 정확한 응답을 도출하는 방법을 소개합니다.

- **Performance Highlights**: RAG 시스템은 대규모의 외부 지식 저장소에서 최신 지식을 동적으로 가져와 사실 정확성을 개선하고 환각(hallucinations)을 줄이는 데 효과적입니다. 기본적으로 RAG 시스템은 Retriever-Generator 파이프라인을 통해 작동하며, 외부 맥락을 통합하여 정보에 기반한 응답을 생성합니다. 논문에서는 다양한 멀티모달 RAG 시나리오 및 평가 방법론을 다루어 향후 연구 방향 및 개방된 문제를 논의합니다.



### Examining and Adapting Time for Multilingual Classification via Mixture of Temporal Experts (https://arxiv.org/abs/2502.08825)
Comments:
          accept to NAACL 2025

- **What's New**: 이 연구에서는 시간이라는 개념을 도메인으로 간주하여, 다국어 환경에서의 분류 성능의 변화를 분석합니다. 특히, Mixture of Temporal Experts (MoTE)라는 새로운 프레임워크를 도입하여 데이터의 시맨틱 및 분포적 변화에 적응하는 방법을 제안합니다. 여기에 의해 시계열 분류기의 일반화를 목표로 하며, 다양한 언어로 이루어진 데이터에서 그 효과를 실험적으로 증명하였습니다.

- **Technical Details**: 연구에서 사용된 데이터는 2007년부터 2014년까지의 사용자 리뷰를 포함하며, 덴마크어, 영어, 프랑스어, 독일어 네 가지 언어로 나뉩니다. 각 언어의 데이터는 총 네 개의 시간 구간으로 나뉘어 있으며, 이를 통해 시간별 성능 변화를 평가하기 위한 다양한 메트릭을 사용하여 교차 도메인 및 인 도메인 성능을 측정합니다. MoTE는 클러스터링 기반 이동 평가자와 시간 라우터 네트워크의 두 가지 모듈로 구성되어 있습니다.

- **Performance Highlights**: MoTE는 전반적으로 다국어 환경에서 시간에 따른 데이터 변화에 대한 분류기 일반화를 성공적으로 향상시킵니다. 실험 결과는 MoTE와 기존 최신 기법 간의 성능 차이를 실질적으로 보여주며, 특히 동적 라우팅 메커니즘이 시간에 따른 성능 향상에 중요한 역할을 한다는 것을 입증하였습니다. 이 연구는 시간 인식 모델이 다국어 시나리오에서 robust하게 작동해야 한다는 중요한 인사이트를 제공합니다.



### Lexical Manifold Reconfiguration in Large Language Models: A Novel Architectural Approach for Contextual Modulation (https://arxiv.org/abs/2502.08818)
- **What's New**: 본 연구에서는 Lexical Manifold Reconfiguration (LMR)라는 새로운 접근법을 소개하여 토큰 임베딩을 동적으로 재구성하는 방법을 제시합니다. 기존의 정적 또는 반 정적 방식은 고정된 벡터 관계에 의존한 반면, LMR은 텍스트 생성 과정에서 변화하는 언어 구조에 즉각적으로 반응하도록 임베딩을 조정합니다. 이 방법은 수학적 기초를 바탕으로 임베딩 재구성을 위한 구조적 메커니즘을 제공합니다.

- **Technical Details**: 이 연구는 비정적인 임베딩 공간에서 비정형적인 변환을 통해 토큰 관계를 인코딩하고 조정하는 수학적으로 기반을 둔 접근법을 개발하고자 합니다. LMR은 지속적인 기하학적 변환을 활용하여 지역적인 토큰 관계와 더 넓은 맥락적 의존성에 따라 임베딩을 동적으로 조정합니다. 이를 통해 다양한 언어 패턴에 대해 더 큰 적응성을 촉진합니다.

- **Performance Highlights**: 실험 결과, LMR은 텍스트 생성에서의 맥락적 일관성, 의미 유지 및 계산 효율성에 긍정적인 영향을 미친 것으로 평가되었습니다. 여러 데이터 세트를 통해 동적으로 조정된 임베딩이 더 넓은 어휘 다양성을 보여주며 반복적인 토큰 패턴을 줄이고 적응 가능한 표현 학습 프로세스를 가능하게 함을 확인했습니다.



### A Systematic Review on the Evaluation of Large Language Models in Theory of Mind Tasks (https://arxiv.org/abs/2502.08796)
- **What's New**: 최근 몇 년간 대형 언어 모델(LLMs)의 Theory of Mind (ToM) 능력 평가가 연구 커뮤니티 내에서 주목받고 있습니다. 이 체계적인 리뷰는 LLMs가 ToM 작업을 수행하는 능력을 평가하는 현재의 노력을 종합하며, 인간 인지의 중요한 측면인 정신 상태의 귀속을 다룹니다. LLMs의 ToM 능력 평가 기술과 프롬프트 전략을 비판적으로 검토하며, 인간과 유사한 정신 상태 추론을 재현하는 데 내재된 제한들을 강조합니다.

- **Technical Details**: Theory of Mind (ToM)는 개인이 다른 사람의 정신 상태를 이해하고 자신의 것과 다른 것을 인지하는 능력으로 정의됩니다. 이 논문은 ToM에 대한 다양한 표준 및 작업 분류를 제시하며, 인지 과학에 뿌리를 둔 분류체계를 통해 LLMs의 능력을 평가하는 방법론을 체계적으로 정리합니다. 특히 기초적인 ToM 능력은 아동기의 인지 발달과 관련이 있으며, 여러 가지 인지 컴포넌트를 통해 발전하는 과정을 다룹니다.

- **Performance Highlights**: 대형 언어 모델들은 텍스트 생성 및 이해에서 놀라운 능력을 보여주고 있지만, ToM 작업 평가에서는 새로운 도전 과제가 발생하고 있습니다. LLM들은 혼란스러운 텍스트 생성을 잘 수행하지만, 정신 상태를 시뮬레이션하거나 이해하는 능력은 더 미세한 평가가 필요합니다. 본 논문에서는 LLMs의 ToM 능력 평가에서 발견된 주요 문제들을 정리하고, 향후 연구 방향에 대한 통찰을 제공합니다.



### If Multi-Agent Debate is the Answer, What is the Question? (https://arxiv.org/abs/2502.08788)
Comments:
          This position paper takes a critical view of the status quo of MAD research, and outline multiple potential directions to improve MAD

- **What's New**: 이번 논문에서는 Multi-agent Debate (MAD) 방법론의 평가 관행에서 발견된 주요 문제점들을 지적하고, 이를 개선하기 위한 Heter-MAD 프레임워크를 제안합니다. 기존 연구들이 다양한 데이터셋에 대해 일관되지 않은 기준을 사용하고 있어 MAD 방법의 일반화 가능성에 의문을 제기합니다. Heter-MAD는 하나의 LLM 에이전트가 이질적인 기초 모델의 출력을 활용하도록 하여 현재의 MAD 프레임워크의 성능을 향상시킬 수 있습니다.

- **Technical Details**: 연구에서는 Chain-of-Thought와 Self-Consistency와 같은 단순한 단일 에이전트 기준선조차 MAD 방법이 안정적으로 능가하지 못한다는 경이로운 결과를 보고합니다. 총 9개의 벤치마크에서 5가지 대표적인 MAD 방법을 4가지 기초 모델을 통해 체계적으로 평가하였으며, 몹시 높고 다양한 모델의 조합이 MAD 프레임워크의 성능을 향상시킬 수 있는 것으로 나타났습니다.

- **Performance Highlights**: 평가 결과는 MAD 방법들이 추가적인 추론 시간(computation)을 사용하면서도 단일 에이전트를 사용하는 방식보다 나은 성능을 보이지 않음을 보여줍니다. Heter-MAD 프레임워크의 도입은 이질적 모델의 출력을 기반으로 MAD의 효율성을 증대시켜 향후 연구의 새로운 방향성을 제시할 것입니다.



### Zero-Shot Belief: A Hard Problem for LLMs (https://arxiv.org/abs/2502.08777)
Comments:
          Submitted to ACL 2025

- **What's New**: 이번 논문에서는 LLM(대형 언어 모델)을 기반으로 FACTBANK 데이터셋에서 제로샷 소스 및 타겟 신념 예측을 위한 두 가지 접근법을 제안합니다. 하나는 이벤트, 소스 및 신념 레이블을 통합하여 한 번의 과정에서 식별하는 통합 시스템이며, 다른 하나는 이벤트 감지를 위해 미세 조정된 DeBERTa 태거를 사용하는 하이브리드 접근법입니다. 우리가 제안한 하이브리드 접근법은 FACTBANK에서 최첨단 결과를 달성하였으나, 여전히 해결해야 할 문제가 많습니다.

- **Technical Details**: 연구에서는 FACTBANK에서의 소스-타겟 신념 예측을 위한 제로샷 프레임워크를 제안하며, 이를 여러 LLM에 대해 테스트하였습니다. 통합 방식과 하이브리드 방식을 통해 각기 다른 단계에서 이벤트와 소스를 식별하고 신념 레이블을 할당하는 방식으로 진행됩니다. 하이브리드 접근법은 DeBERTa 모델을 기반으로 이벤트를 추출한 후, LLM에 대한 프롬프트를 사용하여 결과를 산출합니다.

- **Performance Highlights**: 하이브리드 방식은 FACTBANK 데이터셋에서 기존의 LLM보다 월등한 성능을 기록하였지만, 신념 예측 특히 중첩된 신념의 경우 LLM들이 여전히 낮은 성능을 보이는 문제를 발견했습니다. 또한, 같은 방식으로 이탈리아의 ModaFact 신념 코퍼스에서도 검증이 이루어졌으며, 이는 모델의 전이 가능성을 입증하는 중요한 자료가 됩니다.



### Universal Model Routing for Efficient LLM Inferenc (https://arxiv.org/abs/2502.08773)
- **What's New**: 이 논문에서는 고정된 모델 풀을 위한 라우팅을 학습하는 기존 작업과 달리, 새로운 LLM을 다이나믹하게 라우팅하는 문제를 다룹니다. 제안된 방법은 각 LLM을 특징 벡터로 표현하여, 대표적인 프롬프트에 대한 예측에 기반합니다. 이를 통해 새로운 라우팅 전략을 개발하고, 그 효과성을 보여줍니다.

- **Technical Details**: 제안한 두 가지 전략은 클러스터 기반 라우팅과 학습된 클러스터 맵을 활용합니다. 연구는 이러한 전략들이 이론적으로 최적의 라우팅 규칙을 추정하며, 에러를 정량화할 수 있는 초과 위험 경계를 제공합니다. 특히, 각 후보 LLM의 비용 조정 손실을 최소화하는 방식으로 라우팅을 수행합니다.

- **Performance Highlights**: 실험 결과는 30개 이상의 새로운 LLM 간의 라우팅에서 제안된 전략들이 효과적임을 보여줍니다. 다이나믹 라우팅 문제에 대한 성능 향상을 통해 inference cost를 상당히 감소시킬 수 있는 가능성을 확인했습니다. 이는 대규모 언어 모델을 사용하면서 발생할 수 있는 비용 문제를 해결하는 데 기여할 것으로 기대됩니다.



### SelfElicit: Your Language Model Secretly Knows Where is the Relevant Evidenc (https://arxiv.org/abs/2502.08767)
Comments:
          16 pages, 5 figures, 8 tables

- **What's New**: 이 논문은 SelfElicit라는 새로운 접근 방식을 제안합니다. 이 기법은 언어 모델(LMs)이 중요한 증거를 식별하고 강조하도록 도와줍니다. 이를 통해 LM이 더욱 정확하고 사실 기반의 응답을 생성할 수 있도록 합니다. SelfElicit는 추가적인 훈련 없이 인퍼런스(inference) 시간에 효율적으로 작동합니다.

- **Technical Details**: SelfElicit는 LM의 내부 표현을 활용하여 문맥 내에서 관련 증거 문장을 식별합니다. 문맥과 질문으로 구성된 입력을 사용하여 LM은 지원하는 사실을 활용하여 QA 작업을 수행합니다. 각 문장의 중요성을 평가하기 위해 송신기(Transformer) 모델의 주의(attention) 점수를 분석하며, 이를 통해 LM은 중요 정보를 더욱 효과적으로 처리할 수 있습니다.

- **Performance Highlights**: SelfElicit는 다양한 모델 가족과 벤치마크에서 QA 작업의 성능을 일관되게 개선하는 데 성공했습니다. 본 연구에서 SelfElicit의 효과성을 입증했으며, 이 방법은 노이즈에 강하고 하이퍼파라미터 선택에 덜 민감하다는 장점이 있습니다. 결과적으로 SelfElicit는 LM의 사실적이고 신뢰할 수 있는 응답 생성을 향상시키는 데 기여합니다.



### IHEval: Evaluating Language Models on Following the Instruction Hierarchy (https://arxiv.org/abs/2502.08745)
Comments:
          Accepted to NAACL 2025

- **What's New**: 본 연구에서는 언어 모델(LMs)의 지시 계층을 평가하기 위한 새로운 벤치마크인 IHEval를 도입합니다. IHEval는 시스템 메시지, 사용자 메시지, 대화 기록 및 도구 출력과 같은 다양한 유형의 입력을 포함하며, 총 3,538개의 예제로 구성되어 있습니다. 고급 규제인 시스템 메시지가 하위 입력에 의해 무시되는 경우, 모델의 불안정한 동작을 방지하기 위한 필요성이 강조됩니다.

- **Technical Details**: IHEval 설계는 체계적인 지시 계층 구조를 기반으로 하며, 입력 계층은 네 가지 유형의 입력으로 나뉘어 시스템 메시지, 사용자 메시지, 대화 기록 및 도구 출력의 우선 순위를 매깁니다. 평가 설정은 규칙 준수, 작업 실행, 안전 방어 및 도구 사용을 포함하여 여러 과제를 다루고 있으며, 각 과제의 난이도는 지시 구문을 엄격하게 설정하여 다양하게 조정됩니다.

- **Performance Highlights**: 다양한 LM을 평가한 결과, 모든 모델이 지시 충돌 시 높은 우선 순위를 가진 지침을 인식하는 데 어려움을 겪고 있으며, 오픈 소스 모델의 충돌 해결 정확도는 50% 미만으로 나타났습니다. 이러한 성능 저하는 LM이 지시 계층을 충분히 최적화하지 못했음을 시사하며, 이는 현실 세계 애플리케이션에서 위험 요소를 초래할 가능성을 나타냅니다.



### Are Expressions for Music Emotions the Same Across Cultures? (https://arxiv.org/abs/2502.08744)
Comments:
          Submitted to CogSci

- **What's New**: 이 연구는 음악의 감정 표현을 이해하기 위한 새로운 접근방식을 제안합니다. 브라질, 한국, 미국 세 나라에서 672명의 참가자를 대상으로 한 9개의 온라인 실험을 통해 다양한 음악 샘플에서 문화별 감정 용어를 생성하는 과정을 소개합니다. 이러한 과정을 통해 기존의 서구 중심적인 감정 정의를 극복하고, 감정 연구의 문화적 편향을 줄이려는 노력이 돋보입니다.

- **Technical Details**: 연구에서는 360곡의 팝 음악 데이터를 사용하여 각 나라별로 섞인 샘플 세트를 구성했습니다. 각 곡은 브랜드 및 차트에서 뽑아졌으며, 감정 태깅(open-ended tagging) 절차를 거쳐 참가자들이 자율적으로 감정 관련 태그를 생성했습니다. 이후 이 태그의 유효성을 검증하고, 라벨링된 곡 목록을 기준으로 추가적으로 감정 평가를 진행하여, 서로 다른 문화 간 감정의 유사성을 분석했습니다.

- **Performance Highlights**: 연구 결과, 높은 각성과 긍정적 가치를 지닌 감정에서는 일관성을 보였으나, 그 외의 감정에서는 더 큰 변동성이 있는 것으로 나타났습니다. 특히, 기계 번역이 음악 특정의 의미를 포착하는 데 종종 부족하다는 점도 주목할 만합니다. 이는 감정 연구에서 문화적 다양성을 고려한 접근법의 필요성을 시사합니다.



### Data Augmentation to Improve Large Language Models in Food Hazard and Product Detection (https://arxiv.org/abs/2502.08687)
- **What's New**: 이 연구에서는 ChatGPT-4o-mini를 활용하여 데이터를 증강(data augmentation)함으로써 식품 위험 및 제품 분석에 미치는 영향을 보여줍니다. 증강된 데이터는 RoBERTa-base 및 Flan-T5-base와 같은 두 개의 대형 언어 모델을 교육하는 데 사용됩니다. 결과적으로 증강된 데이터를 사용할 경우 다양한 핵심 메트릭에서 모델 성능이 향상됨을 확인했습니다.

- **Technical Details**: 연구에서 사용된 모델은 ChatGPT-4o-mini, RoBERTa, Flan-T5로, 데이터 증강을 위해 ChatGPT-4o-mini가 사용되었습니다. RoBERTa는 Facebook AI 연구에서 개발된 LLM으로, BERT의 향상된 버전이며, Flan-T5는 T5의 수정된 버전으로 다양한 작업에 활용됩니다. 증강 처리 결과, 기존 데이터 세트에 비해 샘플 수가 증가하여 클래스 불균형 문제를 완화하였습니다.

- **Performance Highlights**: 모델의 성능을 평가한 결과, F1 점수, 정확도, 리콜 및 정밀도와 같은 핵심 메트릭이 모든 모델에서 향상되었습니다. FLAN-T5는 F1 점수에서 RoBERTa보다 우수한 성능을 보여줬으며, 두 모델 모두 데이터 증강을 통해 성능이 크게 개선되었습니다. RoBERTa의 경우, 증강 전후 F1 점수가 각각 71에서 76으로 증가하였고, 이런 성과는 데이터 증강의 효과를 잘 보여줍니다.



### Assessing the Impact of the Quality of Textual Data on Feature Representation and Machine Learning Models (https://arxiv.org/abs/2502.08669)
- **What's New**: 이번 연구에서는 현실 세계의 데이터 수집 품질 문제를 다뤘습니다. Mixtral Large Language Model (LLM)을 활용하여 낮은 품질의 데이터셋에서 오류를 수량화하고 수정하는 방법을 제시하였습니다.

- **Technical Details**: 연구 팀은 MIMIC-III 공공 병원 데이터셋과 호주의 노인 요양원에서 수집된 저품질 데이터셋을 분석했습니다. 또한, MIMIC 데이터셋에서 다양한 오류를 체계적으로 도입하고 ACH 데이터셋의 품질을 개선하기 위해 Mixtral LLM을 사용했습니다.

- **Performance Highlights**: MIMIC 데이터셋의 35,774명과 ACH 데이터셋의 6,336명 샘플에서 Mixtral은 63%의 진행 노트에서 오류를 정확하게 탐지했습니다. 연구 결과, 데이터셋에서 오류 비율이 10% 이하일 경우 모델 성능이 상대적으로 양호했지만, 10% 이상으로 증가하면 성능이 급격히 저하되었습니다.



### Style Extraction on Text Embeddings Using VAE and Parallel Datas (https://arxiv.org/abs/2502.08668)
Comments:
          9 pages, 6 figures

- **What's New**: 이번 연구는 여러 성경 번역의 스타일적 차이를 Variational Autoencoder (VAE) 모델을 사용하여 조사합니다. 텍스트 데이터를 고차원 벡터로 변환함으로써, 연구는 American Standard Version (ASV)과 다른 번역 간의 스타일 변화를 탐지하고 분석하는 것을 목표로 합니다. 결과는 각 번역이 독특한 스타일 분포를 가지고 있으며, VAE 모델을 통해 효과적으로 식별될 수 있음을 보여줍니다.

- **Technical Details**: 언어는 본질적으로 내용과 스타일의 두 가지 주요 요소로 구성됩니다. 연구에서는 OpenAI의 text-embedding-3-small 모델을 사용하여 성경 문장을 1536차원 벡터로 임베딩하고, 이를 통해 텍스트의 내용과 스타일을 수학적으로 계산하였습니다. VAE는 비지도 학습 방법으로, 이 연구에서 각 번역의 텍스트 임베딩 차이를 분석하는 도구로 활용됩니다.

- **Performance Highlights**: VAE 모델의 결과는 사람 평가와 잘 부합하며, 스타일 간의 차이를 평가하는 과정에서 시간적 비용을 상당히 줄일 수 있음을 보여주었습니다. 이 연구의 결과는 자동 콘텐츠 생성, 개인화된 작문 보조 도구 및 AI 기반 문학 분석 도구에서의 잠재적인 응용 가능성을 제시하며, 텍스트 스타일 전이 및 분석에서의 강력한 평가 방법을 제공합니다.



### Hallucination, Monofacts, and Miscalibration: An Empirical Investigation (https://arxiv.org/abs/2502.08666)
Comments:
          Code available at this https URL

- **What's New**: 이 논문은 언어 모델(LLMs)에서 환각율(hallucination rate)을 훈련 데이터의 단일 사실 비율(monofact rate)과 모델의 미캘리브레이션(miscalibration)에 의해 하한선이 설정될 수 있음을 입증하는 이론적 연구를 기반으로 한다. 이 연구는 n-그램 모델과 LLM의 맥락 학습(in-context learning)을 통해 다양한 데이터 분포가 단일 사실 비율과 환각 경향에 미치는 영향을 실험적으로 조사하고 검증하였다. 결과적으로 훈련 데이터에서의 사실 빈도의 분포와 캘리브레이션-환각 간의 상충 관계는 확률적 언어 생성에서 불가피한 요소임을 제시한다.

- **Technical Details**: 실험적으로 대조적인 패턴을 생성하기 위해 파레토(Pareto)나 지프(Zipf) 분포를 이용한 훈련 데이터에서 단일 사실 비율(monofact rate)이 다양하게 발생하도록 설계하였다. 또한, n-그램 모델에서 훈련 샘플의 전이 횟수를 선택적으로 복제(selective duplication)하여 캘리브레이션을 조절하는 새로운 알고리즘을 소개한다. 이 알고리즘은 단일 사실 비율을 일정하게 유지하는 가운데 미캘리브레이션을 유도하여 환각의 발생 양상을 고립시킬 수 있게 해준다.

- **Performance Highlights**: 훈련 데이터에서 예시를 선택적으로 중복 복제하는 방식으로 미캘리브레이션을 조절함으로써 심지어 적은 수의 훈련 예시의 선택적 업웨이팅(selective upweighting)만으로도 환각율을 상당히 낮출 수 있다는 것을 실험 결과로 입증하였다. LLM에서의 맥락 학습 실험을 통해도 단일 사실 비율과 환각율 사이의 관계가 여전히 유지됨을 확인하였으며, 이는 언어 모델 최적화에서 캘리브레이션을 유지하는 것의 전통적인 접근법과는 다른 중요한 인사이트를 제공한다.



### Hallucination Detection: A Probabilistic Framework Using Embeddings Distance Analysis (https://arxiv.org/abs/2502.08663)
- **What's New**: 이 연구는 대형 언어 모델 (LLM)에서 발생하는 환각 현상을 탐지하는 수학적으로 타당한 방법론을 소개합니다. 저자들은 환각된 내용과 정확한 내용 간에 구조적 차이가 존재한다는 것을 밝혔으며, 이를 입증하기 위해 임베딩 공간에서 Minkowski 거리(Minkowski distances)를 사용하였습니다. 연구 결과, 환각된 응답이 진짜 응답과 통계적으로 유의미한 구조적 차이를 가지고 있다는 것을 확인하였습니다.

- **Technical Details**: 이 연구는 LLM의 환각 현상을 감지하기 위해 두 가지 직관을 바탕으로 방법론을 개발했습니다. 먼저, LLM의 환각은 비환각 응답과 다른 확률 분포에서 샘플링된다는 가정과, 수학적 정형성을 통해 환각을 탐지할 수 있다는 가정입니다. 연구자들은 Llama2와 Llama3라는 두 개의 대형 언어 모델을 사용해 데이터를 생성하고, 응답에서 추출한 키워드를 숫자 임베딩으로 변환한 뒤 Minkowski 거리 기반 분석을 수행하였습니다.

- **Performance Highlights**: 개발된 도구는 특정 시스템 파라미터 설정에서 66%의 정확도로 환각된 응답을 탐지하는 성능을 보였습니다. 이는 해당 분야에서의 최고 성능에 필적하는 수치입니다. 저자들은 기존의 환각 탐지 방법과는 다른 새로운 길을 제시하며, 이 연구가 향후 이 분야의 이론적 및 실용적 발전에 기여할 것을 기대하고 있습니다.



### RoToR: Towards More Reliable Responses for Order-Invariant Inputs (https://arxiv.org/abs/2502.08662)
- **What's New**: 이번 연구에서는 언어 모델(LM)에서 리스트 방식 입력의 위치 편향 문제를 해결하기 위해 두 가지 주요 한계를 극복하는 방법을 제안합니다. 첫 번째는 포지셔널 ID 할당을 수정하여 불일치를 피하는 것입니다. 두 번째는 실제 문제에서 순서 불변과 순서 민감성을 모두 처리할 수 있는 적응형 프레임워크인 Selective Routing을 도입합니다. 이를 통해 실제 리스트 방식 입력 작업을 효과적으로 처리할 수 있는 모델을 개발하고자 합니다.

- **Technical Details**: 제안된 RoToR 모델은 전통적인 포지셔널 ID 수정 없이 순서 불변 입력에 대해 최소한의 변경으로 글로벌 정렬을 수행합니다. RoToR는 쿼리와 무관한 방식으로 포지셔널 ID를 할당하며, 다수의 글로벌 정렬 알고리즘을 통해 이전의 순서 불변 모델들보다 일관되게 우수한 성능을 보여줍니다. 또한, Selective Routing을 통해 두 개의 모델(순서 불변 및 비순서 불변) 간의 전환을 기반으로 입력에 적응할 수 있는 방법을 제안합니다.

- **Performance Highlights**: Lost in the Middle(LitM), Knowledge Graph Question Answering(KGQA), MMLU 벤치마크에서 RoToR와 Selective Routing이 제안된 방식으로 리스트 방식 입력 작업을 제로샷(zero-shot)으로 효과적으로 처리할 수 있음을 입증하였습니다. 특히 MMLU 벤치마크에서는 Selective Routing이 순서 민감 및 불변 입력을 모두 효과적으로 처리하며, 기본 성능을 유지하면서도 더 나은 순서 안정성을 달성하는 것을 보여줍니다.



### Few-shot_LLM_Synthetic_Data_with_Distribution_Matching (https://arxiv.org/abs/2502.08661)
Comments:
          10 pages, 5 figures, accepted at www 2025

- **What's New**: 최근 대형 언어 모델(LLMs)의 발전으로 인해, 이러한 모델들이 인컨텍스트 학습 및 소수 샷 언어 생성을 수행하는 능력이 크게 향상되었습니다. 이로 인해 LLM들을 사용해 보다 작은 모델들의 성능을 높이기 위한 고품질 합성 데이터(High-quality synthetic data)를 생성하는 시도가 증가하고 있습니다. 하지만 LLM이 생성한 합성 데이터는 실제 데이터와의 중요한 언어 속성에서 차이를 보이며, 이를 직접 혼합하면 원래 데이터의 분포가 왜곡될 수 있습니다.

- **Technical Details**: 이 논문은 이러한 문제를 해결하기 위해 SynAlign이라는 합성 데이터 생성 및 필터링 프레임워크를 도입합니다. SynAlign는 주요 속성 분포 일치를 기반으로 하여 합성 데이터를 생성하고 필터링하며, 생성 이전에 불확실성 추적(Uncertainty tracker) 모델을 활용해 데이터 클러스터를 반복적으로 선택하여 시연 데이터를 수집합니다. 이후, LLM은 시연 데이터를 요약하여 새로운 데이터를 합성하고, 최대 평균 차이(Maximum Mean Discrepancy)를 목적 함수로 사용하여 각 합성 데이터의 샘플링 가중치를 학습합니다.

- **Performance Highlights**: 여러 텍스트 예측 작업에서의 실험 결과, SynAlign의 합성 데이터는 기존 방법들과 비교하여 더 높은 성능 향상을 보였습니다. 또한, 온라인 검색 시스템에서 실시한 A/B 테스트에서도 SynAlign의 효과성이 입증되었습니다. 이를 통해, SynAlign은 고품질 합성 샘플을 더 효율적으로 생성할 수 있는 방법으로 자리잡고 있음을 보여줍니다.



### Semantic Role Labeling: A Systematical Survey (https://arxiv.org/abs/2502.08660)
- **What's New**: 이 논문에서는 의미 역할 레이블링(SRL)의 연구 경과를 20년간 종합적으로 검토합니다. SRL은 텍스트 내에서 의미적 역할을 이해하고 다양한 NLP 응용 프로그램을 강화하는 데 중요한 작업으로 자리 잡고 있습니다. 특히, LLMs(대형 언어 모델)의 발전이 SRL 연구에 미치는 영향을 다루고 있으며, 기존의 SRL 정의를 재조명하고 다양한 응용 가능한 시나리오와 모델을 제시합니다.

- **Technical Details**: SRL의 정의와 방법론을 4개의 주요 관점으로 분류하고, SPAN 기반 및 의존 기반 SRL 포뮬레이션에 대해 설명합니다. 이 논문은 PropBank의 의미적 역할을 설명하며, FSRL(프레임 SRL)과 관련하여 예시를 통해 프레임을 유발하는 대상의 역할 및 프레임 요소에 대해 자세히 논의합니다. SRL의 진화를 보여주는 기술적 접근 방식과 메트릭스를 심층적으로 분석합니다.

- **Performance Highlights**: SRL은 정보 추출, 기계 번역, 질문 응답 등 다양한 NLP 작업에 필수적입니다. 최근 연구는 SRL의 전통적인 문장 수준 접근을 넘어 담화 수준 및 다중 모달 시나리오에서의 적용을 보여줍니다. 특히 비주얼 SRL(VSRL)과 스피치 기반 SRL의 발전은 SRL의 활용 범위를 넓혔으며, 대화 및 맥락 정보를 포함한 사회적 이해에서도 중요한 역할을 합니다.



### Refining Positive and Toxic Samples for Dual Safety Self-Alignment of LLMs with Minimal Human Interventions (https://arxiv.org/abs/2502.08657)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 최근 인공지능(AI) 에이전트인 ChatGPT와 LLaMA는 인간의 의도에 맞춘 안전한 출력을 보장하기 위해 대규모 언어 모델(LLM)을 조정하는 방식으로 Instruction Tuning과 Reinforcement Learning을 주로 사용하고 있습니다. 기존 방법들은 품질 높은 긍정 샘플의 수동 주석에 의존하면서도, 부정확한 레이블과 적절한 반응 데이터 간의 최소한의 차이로 인한 문제를 겪고 있습니다. 이러한 한계를 극복하기 위해, 본 논문에서는 PT-ALIGN이라는 새로운 안전 자기 조정(self-alignment) 방법을 제안합니다.

- **Technical Details**: PT-ALIGN은 긍정 샘플(해로운 콘텐츠가 아닌 반응)과 독성이 있는 샘플(매우 해로운 콘텐츠 포함)을 자동으로 정제하고, 세분화된 이중 Instruction Tuning을 수행하여 인간의 관리 부담을 최소화합니다. 저자들은 LLM을 활용하여 50개 미만의 인간 주석을 통해 학습 인스턴스를 반복적으로 생성하고 개선하며, 최대 가능도 추정(MLE) 및 세분화된 불가능성 훈련(UT)의 두 가지 손실 함수를 사용하여 LLM의 안전성을 함께 향상시키는 방식으로 학습합니다.

- **Performance Highlights**: 9개의 인기 있는 오픈소스 LLM을 대상으로 한 실험을 통해 PT-ALIGN이 안전 정렬에서 효과적임을 입증하였으며, 유용성과 유익성을 유지하면서도 만족스러운 성과를 달성했습니다. 이 연구 결과는 LLM의 안전성을 높이고, 효과성을 유지할 수 있는 가능성을 보여줍니다. PT-ALIGN 방법의 긍정적인 결과는 다른 안전 조정 방법들과의 비교 분석에서도 그 장점을 부각하고 있습니다.



### Theoretical Benefit and Limitation of Diffusion Language Mod (https://arxiv.org/abs/2502.09622)
Comments:
          32 pages, 3 figures

- **What's New**: 이 논문에서는 텍스트 생성을 위한 최근의 방법론인 Diffusion language models에 대해 다룹니다. 특히, Masked Diffusion Model (MDM)을 분석하여, 해당 모델의 효과는 평가 지표에 크게 의존한다는 것을 밝혔습니다. Perplexity를 사용할 경우 MDM은 최적의 성능을 보일 수 있지만, sequence error rate를 사용할 경우 MDM의 효율성이 감소함을 발견했습니다.

- **Technical Details**: MDM의 진행은 일반적인 autoregressive 모델과 비교하여 효율성이 높지만, 이는 특정 조건 하에서만 유효합니다. Perplexity를 메트릭으로 사용할 때는 효율성을 인정받을 수 있지만, reasoning chain과 같은 '정확성'을 요구하는 경우에는 sequence length에 따라 샘플링 스텝이 선형 배로 증가해야 합니다. 이러한 분석 결과는 실험적 연구에 의해 입증되었습니다.

- **Performance Highlights**: 본 연구는 MDM에 대한 이론적 토대를 마련하여 향후 연구의 방향성을 제시합니다. MDM은 특정 메트릭에서는 효과적인 성능을 나타내나, 보다 복잡한 상황에서는 효율성의 저하를 불러올 수 있음을 보여줍니다. 이는 여러 개발자와 연구자들이 MDM의 장단점을 명확히 이해하고 적용할 수 있도록 도움을 줄 것입니다.



### MME-CoT: Benchmarking Chain-of-Thought in Large Multimodal Models for Reasoning Quality, Robustness, and Efficiency (https://arxiv.org/abs/2502.09621)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문에서는 Chain-of-Thought (CoT) 이론이 Large Multimodal Models (LMMs)의 추론 성능에 미치는 영향을 체계적으로 평가하기 위한 MME-CoT라는 새로운 벤치마크를 소개합니다. 이 벤치마크는 수학, 과학, OCR, 논리, 시공간 및 일반 장면의 여섯 가지 도메인에 걸쳐 CoT 추론 성능을 평가합니다. 이러한 과정을 통해 LMMs의 CoT 성능에 대한 첫 번째 포괄적인 연구를 제시합니다.

- **Technical Details**: MME-CoT는 추론 품질, 견고성(robustness) 및 효율성(efficiency)을 세밀하게 평가할 수 있는 세 가지 새로운 메트릭(metrics)을 포함하는 평가 도구를 제안합니다. 고품질 데이터셋을 활용하고 독창적인 평가 전략을 통해 현재 최신 LMMs에 대한 심층 분석을 수행하며, 이 과정에서 몇 가지 중요한 통찰을 발견하였습니다.

- **Performance Highlights**: 주요 발견 내용으로는 1) 반영(reflection) 메커니즘을 가진 모델들이 높은 CoT 품질을 보여주며, Kimi k1.5가 GPT-4o를 능가하여 최고 품질 결과를 도출하였다는 점입니다. 2) CoT 프롬프트(prompts)는 인지 중심의 작업에서 LMM 성능을 저하시킬 수 있으며, 이는 과도한 사고(overthinking)의 부정적인 행동을 시사합니다. 3) CoT 품질이 높음에도 불구하고 반영 기능이 있는 LMMs는 정상 응답 및 자기 수정 단계에서 상당한 비효율성을 보인다는 점입니다.



### Exploring the Potential of Encoder-free Architectures in 3D LMMs (https://arxiv.org/abs/2502.09620)
Comments:
          The code is released at this https URL

- **What's New**: 이번 논문은 엔코더가 없는 아키텍처를 활용하여 3D 이해를 효과적으로 발전시킬 수 있는 가능성을 탐구합니다. 특히, 기존의 엔코더 기반 LMM의 한계를 극복하기 위해 LLM(대형 언어 모델) 내부에서 3D 엔코더의 기능을 통합하는 새로운 전략들을 제안합니다. 이 연구는 3D LMM을 위한 엔코더 없는 아키텍처의 첫 번째 포괄적 조사로, 엔코더를 제거하고 LLM이 고수준의 3D 의미를 추출하도록 합니다.

- **Technical Details**: 논문에서는 LLM-embedded Semantic Encoding(LLM-임베디드 의미 인코딩)과 Hierarchical Geometry Aggregation(계층적 기하집합)이라는 두 가지 핵심 전략을 제안합니다. 첫 번째는 LLM 초기 층을 학습 가능하게 하여 3D 의미를 캡쳐하도록 돕는 방법입니다. 두 번째는 3D 토큰을 기하학적 분포에 따라 집계하여 LLM이 점진적으로 세부적인 3D 의미를 통합하게 합니다. 이 두 전략은 ENEL이라는 엔코더 없는 3D LMM을 통해 구현됩니다.

- **Performance Highlights**: ENEL은 ShapeLLM-13B와 비교하여 클래스 분류에서 55.0%, 캡션 생성에서 50.92%라는 성과를 달성하며, 기존 기술 수준에 가까운 성능을 보여줍니다. 이 연구는라는 것은 엔코더 기반 아키텍처에 비해 엔코더 없는 아키텍처가 3D 이해 분야에서 매우 유망하다는 것을 의미합니다. 최종적으로, ENEL의 출현은 3D 시나리오에 엔코더 없는 아키텍처를 적용하는 효율적인 경로를 제공할 것으로 기대됩니다.



### CoT-Valve: Length-Compressible Chain-of-Thought Tuning (https://arxiv.org/abs/2502.09601)
Comments:
          Work in progress. Code will be released at this https URL

- **What's New**: 본 논문에서는 하나의 모델로도 Chain-of-Thought(CoT) 길이를 유연하게 조정할 수 있도록 하는 새로운 방법인 CoT-Valve를 소개합니다. 이 방법은 모델의 추론 비용을 줄이는 동시에 다양한 길이의 추론 체인을 생성할 수 있는 가능성을 제공합니다. 길이에 따라 조절할 수 있는 파라미터 공간 내의 방향을 찾아내고, 이를 통해 추론 체인의 압축을 가능하게 합니다.

- **Technical Details**: CoT-Valve는 파라미터 공간 내에서 CoT의 길이를 조절할 수 있는 정밀한 튜닝 방법과 점진적인 체인 길이 압축 방법을 특징으로 합니다. LoRA(Hu et al., 2022)를 통해 간단하게 강도를 조정할 수 있는 추가 분기를 구현하였고, 이를 통해 짧은 체인을 생성할 수 있는 방향을 정의합니다. MixChain 데이터셋을 활용하여 각 질문에 대해 다양한 길이의 추론 경로를 가진 데이터를 구축하였습니다.

- **Performance Highlights**: CoT-Valve의 실험 결과, GSM8K와 AIME 데이터셋에서 각각 추론 체인을 741토큰에서 225토큰으로, 6827토큰에서 4629토큰으로 줄이는 데 성공했습니다. 성능의 경미한 감소(95.07%에서 94.92%)에도 불구하고 모델의 효율성을 개선하는 데 중요한 기여를 합니다. 연구 결과 짧은 추론 경로가 때때로 긴 경로보다 우수한 성과를 낼 수 있음을 강조합니다.



### Do LLMs Recognize Your Preferences? Evaluating Personalized Preference Following in LLMs (https://arxiv.org/abs/2502.09597)
Comments:
          Accepted at ICLR 2025 as oral presentation. Code and data at: this https URL

- **What's New**: PrefEval은 사용자 선호도를 장기 대화 설정에서 추론하고 기억하며 준수하는 LLM의 능력을 평가하기 위한 새로운 벤치마크입니다. 이 데이터셋은 20개의 주제를 아우르는 3,000개의 사용자 선호 및 쿼리 쌍으로 구성되어 있으며, LLM의 성능 평가에는 생성(task) 및 분류(task) 과제가 포함됩니다. 이 연구는 현재 LLM들이 사용자 선호도를 효과적으로 반영하는 데 있어 중요한 한계를 보이고 있음을 드러냅니다.

- **Technical Details**: PrefEval은 사용자 선호를 반영하거나 암시하는 방식으로 정보가 제시되는 다양한 형태를 포함하고 있습니다. 데이터셋은 명시적 또는 암시적 형태의 선호 정보를 포함하며, 사용자의 메시지 및 LLM의 응답을 추적하는 다중 턴 대화를 기준으로 설정됩니다. 10가지의 오픈 소스 및 상용 LLM을 사용하여 다양한 컨텍스트 길이(최대 100k 토큰)에서 평가를 수행했습니다.

- **Performance Highlights**: 현재 최첨단 LLM은 사용자의 선호를 적극적으로 반영하는 데 큰 어려움을 겪고 있으며, 특히 10턴의 제로 샷(zero-shot) 설정에서는 정확도가 10% 미만으로 나타났습니다. 여러 응답 형태에서 사용자 선호도를 잘 인식하지 못하여 성능이 저하되는 것으로 나타났습니다. 그러나 PrefEval에서 파인튜닝(fine-tuning)을 실시했을 때 전반적인 성능 향상이 관찰되었습니다.



### Optimizing GPT for Video Understanding: Zero-Shot Performance and Prompt Engineering (https://arxiv.org/abs/2502.09573)
- **What's New**: 이번 연구에서는 비디오 콘텐츠 분류에서 GPT 기반 모델을 활용하여 제로샷(zero-shot) 분류를 최적화하는 새로운 접근 방식을 제시합니다. 복잡한 정책을 단순화함으로써 허위 부정(false negatives)을 감소시키고, 새롭게 도입한 분해-집계(decomposition-aggregation) 기반 프롬프트 엔지니어링 기법이 기존의 단일 프롬프트 방법들보다 뛰어난 성능을 보여줍니다. 이러한 실험은 실제 산업 문제들을 대상으로 진행되어, 비디오 분류 시스템 개선을 위한 효과적이고 확장 가능한 솔루션을 제공합니다.

- **Technical Details**: 연구에서는 GPT-4의 멀티모달(multimodal) 이해 능력을 평가하며, TikTok 비디오 콘텐츠를 다양한 기준에 따라 분류하는 가능성을 탐구합니다. GPT-4의 제로샷과 피-샷(few-shot) 능력이 산업별 문제에 효과적으로 적용될 수 있는지를 확인하며, 정책 개선과 프롬프트 엔지니어링이 GPT-4 성능에 미치는 영향을 탐색합니다. GPT-4는 실험에서 기존의 이종 분류 모델들과 비교했을 때 여러 카테고리에서 비슷한 수준의 성능을 보였으나, 클릭베이트(clickbait) 비디오와 같은 복잡한 분류에서는 어려움을 겪었습니다.

- **Performance Highlights**: 연구 결과, 복잡한 정책 프롬프트를 간소화하면 허위 부정(false negatives)을 줄이면서 비디오 분류 정확도를 높이는 데 기여합니다. 또한 클릭베이트 탐지와 같은 작업을 하위 카테고리로 나누는 프롬프트 엔지니어링이 상당한 성능 향상을 가져오는 것으로 나타났습니다. 이 발견은 진정한 산업 데이터셋을 기반으로 한 실험을 통해 이루어졌으며, 이는 비디오 분류 작업에 실질적으로 적용 가능하다는 점에서 높이 평가받습니다.



### EmbodiedBench: Comprehensive Benchmarking Multi-modal Large Language Models for Vision-Driven Embodied Agents (https://arxiv.org/abs/2502.09560)
Comments:
          51 pages

- **What's New**: 이번 논문에서는 Multi-modal Large Language Models (MLLMs)를 기반으로 한 구체화된 에이전트를 평가하기 위한 새로운 기준인 EmbodiedBench를 소개합니다. EmbodiedBench는 4가지 환경에서 1,128개의 다양한 테스트 과제를 포함하여, 고수준의 의미적 과제와 원자적 행동을 포함한 저수준 과제를 평가합니다. 특히, 에이전트의 공통 감각 추론 및 복잡한 지시 이해와 같은 필수 능력을 평가할 수 있도록 세분화된 하위 집합을 제공합니다.

- **Technical Details**: EmbodiedBench는 두 가지 주요 기능을 갖추고 있으며, 첫째로 작업 수준에 따라 다양한 작업을 제공합니다. 둘째로, 공통 감각 추론 및 시각 인식과 같은 여섯 가지 핵심 능력을 평가하는 세밀한 평가 프레임워크를 도입하여 기존 벤치마크와 차별화됩니다. 이를 통해 13개의 MLLM 모델을 평가하며, 에이전트의 결정을 내리기 위해 통합된 에이전트 프레임워크를 사용합니다.

- **Performance Highlights**: MBLMs는 고수준의 작업에서는 뛰어난 성능을 보여주지만, 저수준 조작에서는 한계를 보입니다. 특정 모델인 GPT-4o는 평균 28.9%의 점수를 기록하며, 저수준 작업에서 성능이 40%에서 70%까지 저하됨을 알 수 있습니다. 이 연구는 MLLM 기반 구체화된 에이전트의 발전을 위한 소중한 통찰을 제공합니다.



### Pixel-Level Reasoning Segmentation via Multi-turn Conversations (https://arxiv.org/abs/2502.09447)
- **What's New**: 본 논문에서는 기존의 시각 인식 시스템이 단일 대화에서의 지역 수준 세분화에 집중하고 있으며, 동적인 사용자 의도를 이해하는 데 한계가 있음을 지적합니다. 이를 해결하기 위해, 다중 턴 대화 기반의 픽셀 수준 추론 세분화(Pixel-level Reasoning Segmentation, Pixel-level RS)라는 새로운 과제를 소개합니다. 또한, 이 과제를 위한 픽셀 수준 추론 세분화 데이터셋(PRIST)을 구축하여 총 24,000개의 발화와 8,300개의 다중 턴 대화 시나리오를 포함하고 있습니다.

- **Technical Details**: PRIST 데이터셋은 3단계 대화 자동 생성 파이프라인을 통해 다중 턴 상호작용을 통해 사용자 의도를 이해하고 픽셀 수준 설명과 세분화 마스크를 생성합니다. 이를 지원하기 위해 다중 비전 인코더와 의미적 영역 정렬 전략을 채택하여 상세한 시각 정보를 포착하고 세분화 성능을 향상시킵니다. 또한, 이 프레임워크는 사용자 의도를 반복적으로 명확하게 하기 위해 다중 턴 상호작용을 지원합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 PRIST 데이터셋에서 기존 세분화 기준선보다 더 우수한 성능을 보였으며, 세분화 및 LLM 기반의 추론 메트릭에서 두각을 나타냈습니다. 평가 메트릭은 대화의 일관성, 일치성 및 정확성 측면에서 종합적으로 발전한 결과를 보여줍니다. 이 연구는 픽셀 수준 추론 세분화의 발전을 위한 새로운 기준을 제시합니다.



### Language Agents as Digital Representatives in Collective Decision-Making (https://arxiv.org/abs/2502.09369)
- **What's New**: 이번 연구는 집단 의사결정 과정에서 개인의 선호를 대변하는 언어 에이전트를 훈련시키는 가능성을 탐구하고 있습니다. 이를 통해 우리는 디지털 대리 역활의 구현을 위한 새로운 접근 방식을 제안합니다. 기존의 인간 행동 시뮬레이션 연구와는 달리, 우리는 대표성을 위한 시뮬레이션에 중점을 두고 있습니다.

- **Technical Details**: 연구는 집단 의사결정을 에피소드 방식의 상호작용 과정으로 형식화하고, 디지털 대리 문제를 정의합니다. 각 참가자는 집단 결정에 대한 선호를 표현하며, 언어 공간에서의 작용을 처리합니다. 여기서 언어 에이전트는 인간 대리인으로서 적절히 선호를 표현하도록 훈련됩니다.

- **Performance Highlights**: 실험 사례 연구를 통해 다양한 인간 집단에서 합의 찾기 작업의 구현 가능성을 확인했습니다. 대규모 언어 모델을 미세조정(fine-tuning)하여 디지털 대리인으로 작용할 수 있음을 보여주었습니다. 이 연구는 집단 상호작용의 개인화된 시뮬레이션과 메커니즘 설계에 실질적 응용 가능한 결과를 도출합니다.



### You Do Not Fully Utilize Transformer's Representation Capacity (https://arxiv.org/abs/2502.09245)
- **What's New**: 이번 논문에서는 Layer-Integrated Memory (LIMe)를 소개하며, standard Transformer 모델의 제한된 representation (표현) 용량 문제를 해결하고자 합니다. LIMe는 모델이 이전 층의 hidden states (숨겨진 상태)에 접근할 수 있도록 하여 모델의 표현력을 확장합니다. 이는 결국 모델의 전반적인 메모리 사용량을 유지하면서도 성능 개선을 가져오는 간단하면서도 강력한 접근 방식입니다.

- **Technical Details**: LIMe는 masked multi-head self-attention 기법에 간단한 변화를 주어 모든 이전 층의 representation을 통합하며, 효율적인 routing (라우팅) 메커니즘을 통해 다층 특성을 통합합니다. 이 기법은 핵심적인 Transformer 구조를 유지하면서도 추가적인 오버헤드를 최소화합니다. 결론적으로, LIMe는 깊은 층에서 더 높은 entropy (엔트로피)를 유지하고, 밀접한 관련의 토큰들이 서로 더 잘 구별되도록 하는 효과를 보여 줍니다.

- **Performance Highlights**: 언어 모델링 실험을 통해 LIMe가 standard Transformer와 다양한 최첨단 수정 모델보다 일관되게 높은 성능을 보였다. LIMe는 representation collapse 현상을 효과적으로 방지하며, 과거 층에서 중요한 구문적 단서를 더 잘 통합하는 방법도 보여 주어, 향후 연구 방향에도 큰 가능성을 제시합니다. 이러한 결과는 더 깊고 강력한 Transformer 구조를 구축하는데 유망한 방향임을 나타냅니다.



### Reliable Conversational Agents under ASP Control that Understand Natural Languag (https://arxiv.org/abs/2502.09237)
Comments:
          In Proceedings ICLP 2024, arXiv:2502.08453

- **What's New**: 이번 연구에서는 현재의 대화형 인공지능이 가지고 있는 LLMs의 한계를 극복하기 위한 새로운 접근법을 제시합니다. LLMs를 단순한 텍스트 파서로 활용하여 정보를 지식으로 변환하고, 이 지식을 바탕으로 논리적 reasoning을 적용함으로써 더 신뢰할 수 있는 대화 시스템을 구축하고자 합니다. 이로 인해 LLMs의 이해 부족 및 신뢰성 문제를 해결하는 방안을 모색하고 있습니다.

- **Technical Details**: 연구자는 LLMs(대형 언어 모델)와 ASP(답변 집합 프로그래밍)를 결합한 프레임워크를 개발하였습니다. 이 프레임워크는 사람의 대화를 이해할 수 있는 신뢰성 있는 챗봇(chatbot)을 구현하기 위해 설계되었습니다. LLMs는 텍스트를 지식으로 변환하는 역할을 하며, ASP는 이 지식을 토대로 대화를 이끌어갑니다.

- **Performance Highlights**: 이 프레임워크는 특정 작업에 최적화된 챗봇과 사회적 상호작용을 위한 소셜봇(socialbot) 개발에 사용되었습니다. 현재 연구자는 이러한 챗봇들이 확장 가능하고 훈련 가능하도록 발전시키는 데 집중하고 있습니다. 이를 통해 향후 다양한 분야에서 실용적인 인공지능 대화 시스템을 구현할 수 있을 것으로 기대하고 있습니다.



### Mind the Gaps: Logical English, Prolog, and Multi-agent Systems for Autonomous Vehicles (https://arxiv.org/abs/2502.09216)
Comments:
          In Proceedings ICLP 2024, arXiv:2502.08453

- **What's New**: 이 논문에서는 자율주행차의 교통 규칙에 대한 법적 측면을 표현하고 추론하는 모듈형 시스템을 제안합니다. 특히, 영국의 고속도로 규정(Highway Code, HC) 중 교차로와 관련된 부분에 중점을 두고 있습니다. 인간 운전者와 자율주행차 간의 상호작용을 고려하여, 두 사용자 모두에 적용 가능한 고수준의 계산 모델이 필요하다고 주장합니다.

- **Technical Details**: 제안된 시스템은 세 가지 주요 구성 요소로 이루어져 있습니다. 먼저, 규칙을 코드화하는 Logical English를 사용하는 자연어 인터페이스가 있습니다. 두 번째로, Prolog의 내부 표현을 통해 규칙을 나타내며, 세 번째로 NetLogo로 구축된 다중 에이전트 기반 시뮬레이션 환경이 있습니다. 이러한 모듈형 접근 방식은 시스템 전체에서 다양한 '부담(burden)'을 분담할 수 있습니다.

- **Performance Highlights**: NetLogo를 통해 모델링된 규칙의 효과를 시각화하고 간단한 동적 시나리오를 통해 시스템을 검증할 수 있습니다. 지정된 에이전트들은 차량의 컴플라이언스(compliance)를 모니터링하고, 위반 사항이 발생하는 경우 이를 기록합니다. 이후, Validator들은 이러한 정보를 활용하여 위반 사항이 처벌 가능한지를 구분합니다.



### Neuro-Symbolic Contrastive Learning for Cross-domain Inferenc (https://arxiv.org/abs/2502.09213)
Comments:
          In Proceedings ICLP 2024, arXiv:2502.08453

- **What's New**: 본 연구는 자연어 추론(NLI) 작업에서의 기존의 사전 학습된 언어 모델(PLMs)과 유도 논리 프로그래밍(ILP)의 한계를 극복하기 위한 신경-기호 대조 학습(neuro-symbolic contrastive learning)을 제안합니다. 이 방법은 추상적인 논리 관계를 효율적으로 통합하여 데이터가 노이즈가 많고 희소한 상황에서도 논리적 정확성을 향상시킵니다.

- **Technical Details**: 신경-기호 대조 학습은 논리 프로그램(logic programs)과 논리 규칙(logic rules) 집합으로 데이터를 표현하여 추상적인 논리 관계를 임베딩(embedding)합니다. 이는 데이터의 높은 변동성을 갖는 텍스트 정보를 포착하면서도 유사한 논리적 관계를 가진 텍스트 정보를 구분할 수 있는 공간을 제공합니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식이 모델의 일반화(generalisation) 및 추론(reasoning) 능력을 상당히 개선시키는 것을 보여줍니다. 이는 기존의 PLMs의 한계를 뛰어넘은 성과로 볼 수 있습니다.



### LP-LM: No Hallucinations in Question Answering with Logic Programming (https://arxiv.org/abs/2502.09212)
Comments:
          In Proceedings ICLP 2024, arXiv:2502.08453

- **What's New**: 이번 논문에서는 LP-LM이라는 새로운 시스템을 소개합니다. LP-LM은 사용자의 질문에 대한 답변을 신뢰할 수 있는 사실에 근거하여 생성하며, 이러한 과정을 위해 Prolog를 통한 의미론적 파싱(semantic parsing)을 사용합니다. 이 시스템은 고전적인 LLM의 문제인 환각(hallucination) 문제를 해결하고, 더욱 정확한 답변을 제공합니다.

- **Technical Details**: LP-LM은 입력 질문에 대한 가장 가능성이 높은 구성 구문 트리(constituency parse tree)를 생성하고, 이에 대응하는 Prolog 항(term)을 작성합니다. 이 항은 질문 응답을 위해 자연어 문장으로 구성된 지식베이스(KB)에 대해 실행됩니다. LP-LM은 DCG(Definite Clause Grammar) 파싱을 이용하여 입력 문장의 크기에 비례해 선형 시간(linear time) 내에서 작동하며, 충분히 많은 문법 규칙을 활용합니다.

- **Performance Highlights**: 실험을 통해 LP-LM과 현재 널리 알려진 LLM의 정확도를 비교한 결과, LP-LM은 심지어 간단한 질문에서도 환각 현상 없이 신뢰할 수 있는 답변을 제공합니다. 반면 기존 LLM은 이러한 질문에서도 일관되지 않은 답변을 생성하는 경향이 있습니다. 이 연구는 LP-LM이 LLM 대비 얼마나 진일보한 성능을 보이는지를 입증합니다.



### FLAME: Flexible LLM-Assisted Moderation Engin (https://arxiv.org/abs/2502.09175)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 사용자 모델 상호작용을 보다 안전하게 관리하기 위한 새로운 접근 방식인 FLAME(Flexible LLM-Assisted Moderation Engine)를 소개합니다. 기존의 입력 필터링(input filtering) 대신 출력 조정(output moderation)에 중점을 둔 FLAME은 최신의 'Best-of-N' 유형의 'jailbreaking' 공격에 더욱 효과적으로 저항할 수 있는 방법을 제시합니다. FLAME은 가벼운 방식으로 설계되어 접근 가능성을 높이고, 새로운 공격에 대한 신속한 적응을 가능하게 합니다.

- **Technical Details**: FLAME은 이진 분류 문제로 사용자의 요청이나 모델의 응답이 금지된 주제를 포함하는지를 확인합니다. 이 알고리즘은 n-그램(n-grams) 처리와 고유한 규칙 기반 분류 함수를 통해 간단하고 효율적인 처리를 수행합니다. pymorphy3와 nltk 같은 도구를 사용하여 단어 형태(normalized forms)를 처리하고 필터링 기능을 통해 필요한 주제를 조정할 수 있습니다.

- **Performance Highlights**: FLAME은 이전의 moderation 시스템에 비해 뛰어난 성능을 발휘하여 GPT-4o-mini 및 DeepSeek-v3에서 'jailbreaking' 공격의 성공률을 9배 이상 줄였습니다. 여러 LLM 플랫폼에서의 실험을 통해 FLAME이 2배에서 9배의 저항력을 보여준 것을 확인했습니다. 이 결과는 LLM의 콘텐츠 조절 시스템의 효율성을 크게 향상시키는 데 기여하고 있습니다.



### Logical Reasoning in Large Language Models: A Survey (https://arxiv.org/abs/2502.09100)
- **What's New**: 이번 논문은 최신의 대규모 언어 모델(LLM)에서의 논리적 추론(logical reasoning) 능력을 종합적으로 검토한 것입니다. 연구자들은 데이터 중심 조정(data-centric tuning), 강화 학습(reinforcement learning), 디코딩 전략(decoding strategies), 그리고 신경 상징적 접근(neuro-symbolic approaches)과 같은 다양한 접근 방식을 통해 논리적 추론 능력을 향상시키고자 하는 전략을 제시합니다. 또한, 향후 연구 방향에 대한 필요성을 강조하며, AI 시스템 내에서의 논리적 추론 강화를 위한 추가적인 탐색이 필요하다고 언급하고 있습니다.

- **Technical Details**: 논문에서 다루는 주요 분야는 LLM의 논리적 추론 능력이며, 여기에는 연역적(deductive), 귀납적(inductive), 유추적(analogical), 그리고 범주적(abductive) 추론이 포함됩니다. 이 연구는 LLM의 이론적 기초와 평가에 사용되는 벤치마크를 분석하여 현재의 기술적 격차를 확인합니다. 저자들은 LLM의 논리적 추론을 개선하기 위한 여러 방법론을 설명하며, 이는 훈련 데이터의 일반화와 해석 가능성을 포함합니다.

- **Performance Highlights**: 대규모 언어 모델(LLM)들이 나타내는 논리적 분석의 증가는 업계와 연구에서 큰 주목을 받고 있습니다. 연구자들은 LLM이 다양한 도메인에서의 활용 가능성을 열어주며, 법률 분석이나 과학 연구 분야에서도 그들의 추론의 정확성과 검증 가능성을 확보하는 것이 점점 더 중요하다고 주장합니다. LLM의 최신 성능 평가와 향상된 평가 방법론은 AI 시스템의 신뢰성을 높이는 데 기여할 것으로 기대됩니다.



### Show Me the Work: Fact-Checkers' Requirements for Explainable Automated Fact-Checking (https://arxiv.org/abs/2502.09083)
Comments:
          Conditionally accepted to CHI'25

- **What's New**: 최근 온라인 미디어에서 대형 언어 모델과 생성 AI의 확산으로 인해 역동적으로 증가하는 허위 정보에 대한 효과적인 자동 사실 확인의 필요성이 강조되고 있습니다. 이러한 연구는 사실 확인자가 제공한 증거를 평가하는 방법 및 절차와 자동화 도구의 실제 활용 방식, 그리고 이에 필요한 설명 기준을 규명하는 데 주력합니다. 논문은 사실 확인 프로세스의 각 단계에서 점검해야 할 설명의 필요성을 제시하고, 자동화된 시스템이 사실 확인자의 요구사항을 충족하기 위해 어떤 정보를 제공해야 하는지에 대한 불확실성을 드러냅니다.

- **Technical Details**: 이 연구는 5개 대륙의 10명의 사실 확인 전문인과의 반구조적 인터뷰를 통해 사실 확인자가 증거를 평가하고 결정을 내리는 과정에서 어떤 정보를 설명하는 것이 중요한지에 대해 논의합니다. 연구는 자동화 도구가 사실 확인자의 작업 흐름에서 어떻게 사용되는지에 대한 특성을 정의하고, 사실 확인 프로세스의 각 단계에서 요구되는 설명의 종류를 명확히 합니다. 이러한 구체적 요구 사항들은 모델의 추론 경로와 특정 증거를 참조하며 불확실성과 정보 간극을 강조해야 함을 enthalten합니다.

- **Performance Highlights**: 연구 결과는 현재의 자동 사실 확인 상태와 사실 확인자의 실제 필요 사이의 격차를 명확히 하며, 반복 가능한 설명의 중요 기준을 제안합니다. 또한, 자동화된 사실 확인 시스템의 설명과 그 투명성 부족이 사실 확인자들의 의사 결정 과정에 미치는 영향을 설명함으로써, 실질적으로 도움을 줄 수 있는 방법론을 제시합니다. 결과적으로, 이 연구는 자연어 처리(NLP) 기술의 발전이 사실 확인자의 요구를 어떻게 충족시킬 수 있는지를 탐색하며, 향후 기술적 발전 방향을 제안합니다.



### Escaping Collapse: The Strength of Weak Data for Large Language Model Training (https://arxiv.org/abs/2502.08924)
- **What's New**: 이번 논문에서는 합성 데이터(synthetic data)가 대형 언어 모델(LLM)의 훈련에 어떻게 기여하는지와 이러한 데이터의 큐레이션(curation) 없이는 성능이 정체되거나 심지어 감소할 수 있다는 점에 대해 논의합니다. 제안된 이론적 프레임워크를 통해 LLM 성능 향상을 위한 큐레이션의 필요성을 정량적으로 분석하였습니다. 많은 비합성 데이터의 품질이 저조하더라도 최적의 LLM으로 수렴하는 훈련 절차를 설명합니다.

- **Technical Details**: 이 논문은 부스팅(boosting)이라는 고전적인 기계 학습 기술에 착안하여, 약한 학습 알고리즘(weak learning algorithm)을 활용해 우수한 분류기를 생성하는 방식을 통해 분석을 진행하였습니다. 제안된 훈련 절차는 최근에 발표된 여러 합성 데이터 훈련 방법들을 포함하며, 이러한 방법들이 어떻게 성공적인지를 설명하는 데 기여합니다. 특히, 어려운 예제에 동적으로 라벨링 자원을 집중시키는 방식을 통해 성능을 향상시킬 수 있음을 보여줍니다.

- **Performance Highlights**: 실험 결과, 합성 데이터에서 훈련된 LLM이 향상된 성능을 보이는 것으로 나타났습니다. 훈련 절차가 약한 학습기의 노력을 집중하는 방식과 유사하게 작동하여, 도전적인 예에 대한 집중이 성능을 개선하는 데 기여합니다. 이러한 결과는 기존의 방법들이 성공하는 이유를 명확히 하며, 향후 개선 작업에 대한 기회를 제시합니다.



### PathFinder: A Multi-Modal Multi-Agent System for Medical Diagnostic Decision-Making Applied to Histopathology (https://arxiv.org/abs/2502.08916)
- **What's New**: 이번 논문에서는 PathFinder라는 다중 모달 및 다중 에이전트 시스템을 제안합니다. 이 시스템은 4개의 AI 에이전트(Triage, Navigation, Description, Diagnosis)를 통합하여 병리 전문가의 의사 결정 과정을 모방합니다. 특히, 각 에이전트는 WSIs(Whole Slide Images)를 효과적으로 탐색하고 자연어 설명을 제공하여 포괄적인 진단을 수행합니다. 이러한 접근 방식은 기존의 전통적인 AI 방법이 가지는 한계를 극복하는 데 목적을 두고 있습니다.

- **Technical Details**: PathFinder는 Triage Agent가 WSI를 양성 또는 위험으로 분류한 후 위험으로 판단될 경우 Navigation 및 Description Agents가 중요 영역을 반복적으로 조사하여 중요도 맵과 설명을 생성합니다. 이 과정에서 각 에이전트는 긴밀하게 협력하여 진단 정보를 수집하고, Diagnosis Agent가 최종 진단 평가를 수행합니다. 실험 결과, PathFinder는 기존의 최고 성능을 보여준 방법들보다 8% 높은 정확도로 피부 흑색종 진단을 가능하게 하였습니다.

- **Performance Highlights**: PathFinder는 병리학자와의 질적 분석에서 Description Agent의 출력 품질이 GPT-4o와 비교될 만큼 우수하다는 것을 보여줍니다. 이 시스템은 또한 병리학자의 평균 성능을 9% 초과하여 새로운 기록을 세우며, 병리 진단에서 효율적이고 정확하며 해석 가능한 AI 지원 진단을 실현할 능력을 가지고 있습니다. 논문에서는 데이터, 코드 및 모델을 제공하며, PathFinder의 기초와 성과를 자세히 설명합니다.



### EnigmaEval: A Benchmark of Long Multimodal Reasoning Challenges (https://arxiv.org/abs/2502.08859)
- **What's New**: EnigmaEval은 언어 모델의 사고 능력을 평가하기 위한 새로운 벤치마크로, 기존의 추리 기준을 넘어 다중 모드 문제를 해결하는 데 초점을 맞추고 있습니다. 이 벤치마크는 퍼즐 대회에서 유래된 문제들로 구성되어 있으며, 언어 모델들이 암묵적인 지식 종합과 다단계 추론 능력을 수행하는 능력을 평가합니다. 특히 이 문제들은 명확한 지시 없이 다양한 접근 방식을 탐색해야 하므로 최첨단 LLM에게는 도전적인 과제가 됩니다.

- **Technical Details**: EnigmaEval은 1184개의 퍼즐로 구성되어 있으며, 원본 PDF의 PNG와 구조화된 텍스트-이미지 표현 두 가지 포맷으로 제공됩니다. 퍼즐은 다양한 형식을 포함하고 있으며, 단순한 단어 또는 짧은 구절을 해결하는 것을 목표로 합니다. 실험 과정에서 선진 LLM의 성능을 평가하기 위해 모델의 결과물을 실제 정답과 비교하며, 이는 문자열 매칭을 통해 이루어집니다.

- **Performance Highlights**: 최첨단 LLM들은 EnigmaEval 퍼즐에서 단 7.0%의 낮은 정확성을 기록했으며, 어려운 문제에서는 0%의 정확성을 보였습니다. 이는 현재의 다중 모드 LLM들이 복잡한 문제 해결 작업에서 전문가 수준의 사고 능력과 큰 격차가 있음을 나타냅니다. LLM들이 이러한 도전적이고 복잡한 문제를 해결하는 데 필요한 전략적 사고 및 구조화된 문제 해결 접근 방법에 익숙하지 않다는 점에서 주목할 만합니다.



### Can a Single Model Master Both Multi-turn Conversations and Tool Use? CALM: A Unified Conversational Agentic Language Mod (https://arxiv.org/abs/2502.08820)
- **What's New**: 본 논문은 기존의 대화 모델(TOD)과 언어 에이전트(LA)를 결합한 통합 접근법인 CALM (Conversational Agentic Language Model)을 제안합니다. CALM은 다양한 API를 호출하면서도 멀티턴 대화 관리를 효과적으로 수행할 수 있도록 설계되었습니다. 저자들은 CALM-IT라는 멀티태스크 데이터셋을 통해 CALM 8B, CALM 70B 및 CALM 405B 모델을 각각 훈련시켜, 기존의 도메인 특화 모델들과 비교해 전반적으로 우수한 성능을 보였습니다.

- **Technical Details**: CALM 모델은 MultiWOZ 2.4, BFCL V3, 및 API-Bank라는 세 가지 벤치마크에서 평가되었습니다. 연구팀은 기존 LAs가 API 호출에서는 우수하나 멀티턴 상호작용에서는 성과가 낮음을 발견했습니다. 이와 반대로, 기존의 TOD 시스템은 멀티턴 대화에서는 잘 작동하지만 기능 호출에는 한계가 있음을 확인했습니다. 이로 인해 CALM 모델들은 두 영역 모두에서 뛰어난 성능을 발휘하여 기존의 성과 격차를 줄였습니다.

- **Performance Highlights**: CALM 70B 및 CALM 405B 모델은 GPT-4o와 다른 도메인 특화 모델들보다 멀티턴 대화 및 기능 호출 작업에서 더 높은 성능을 나타냈습니다. 이 연구는 오픈 소스 커뮤니티의 향후 연구를 촉진하기 위해 코드, 모델 가중치 및 데이터셋 등 모든 자료를 공개하였습니다. 기존의 보안 모델들과 비교해 CALM 모델의 노력은 멀티턴 대화 스킬과 신뢰할 수 있는 기능 호출 능력을 통합함으로써 주목받고 있습니다.



### Mathematical Reasoning in Large Language Models: Assessing Logical and Arithmetic Errors across Wide Numerical Ranges (https://arxiv.org/abs/2502.08680)
- **What's New**: 본 연구는 GSM8K에서 파생된 데이터셋 생성기인 GSM-Ranges를 도입하여, 수학 문제의 숫자 값을 체계적으로 변경함으로써 다양한 숫자 범위에서 모델의 강건성을 평가하고자 합니다. 이는 실제 문제 해결에 필요한 보다 다양한 수치 범위를 반영하며, LLM의 논리적 오류와 비논리적 오류를 구별하는 새로운 채점 방법론도 제공합니다. 이러한 접근은 LLM의 수학적 추론 능력에 대한 보다 포괄적인 평가를 가능하게 하며, 향후 연구 방향에도 중요한 인사이트를 제공합니다.

- **Technical Details**: GSM-Ranges는 GSM8K 데이터셋을 기반으로 하여, 문제 내 숫자를 다양한 규모로 섞어주는 여섯 가지의 왜곡 수준을 적용합니다. 새로운 채점 방법론은 GPT-4o 모델을 사용하여 LLM이 생성한 응답을 Python 코드로 변환하고, 이 코드를 실행하여 비논리적 오류를 격리시킵니다. 최종 결과는 기초적 논리 오류와 계산적 오류를 구분하여, LLM의 수학적 추론 과정을 보다 세밀하게 평가하는 데 기여합니다.

- **Performance Highlights**: 모델의 숫자 복잡성이 증가함에 따라 논리적 오류율이 최대 14%포인트까지 증가하는 경향을 관찰했습니다. 또한, LLM이 단독 산술 작업에서는 높은 정확도를 보이지만, 단어 문제로 계산이 포함될 경우 성능이 크게 저하되는 것을 나타냅니다. 이러한 결과는 LLM의 수학적 추론 능력에 대한 포괄적인 이해를 제공하며, 다양한 수학 문제 처리 과정에서의 강건성을 향상시키기 위한 연구의 초석이 될 수 있습니다.



### Examining Multilingual Embedding Models Cross-Lingually Through LLM-Generated Adversarial Examples (https://arxiv.org/abs/2502.08638)
- **What's New**: 이 연구는 Cross Lingual Semantic Discrimination (CLSD)이라는 새로운 크로스링구얼(跨語) 의미 검색 작업을 소개합니다. 이 작업은 두 언어쌍의 평행 문장 쌍 세트만으로 수행할 수 있으며, 이로써 도메인 특정 평가가 가능해집니다. 특히, 독일어-프랑스어 뉴스 도메인에서 네 가지 실험 사례를 만들어 모델의 성능을 비교했습니다.

- **Technical Details**: CLSD 작업은 특정 원문에서 출발하여 유사하게 보이나 의미적으로 다른 네 개의 분산 문장(distraction sentence) 중에서 올바른 대상 문장을 찾는 과제를 포함합니다. 연구진은 LLM(대형 언어 모델)을 사용하여 이러한 분산 문장들을 생성하였으며, 이는 구문 구조와 겉보기에 유사하지만 의미적으로는 다릅니다. 모델 평가를 위해 R@1(Recall@1) 지표를 활용하여 참 문장이 분산 문장보다 얼마나 유사한지를 측정했습니다.

- **Performance Highlights**: 실험 결과, LaBSE 모델은 94.43(R@1)으로 직접적인 크로스링구얼 평가에서 가장 높은 점수를 기록했습니다. 반면, 영어 중심의 데이터셋에서 추가로 훈련된 M-E5 및 M-GTE 모델은 더 엄격한 CLSD 작업에 대해 성능이 저하되었습니다. 또한, 비트익스 마이닝 모델들은 영어를 피벗 언어로 사용했을 때 성능이 하락하는 경향을 보였습니다.



### SPeCtrum: A Grounded Framework for Multidimensional Identity Representation in LLM-Based Agen (https://arxiv.org/abs/2502.08599)
Comments:
          21 pages, 8 figures, 5 tables, Accepted in NAACL2025 Main

- **What's New**: 본 논문에서는 기존의 개인 정체성 시뮬레이션 방법이 사람들이 가진 복잡성을 지나치게 단순화하여 불완전한 표현을 일으킬 수 있다는 문제를 제기합니다. 이를 해결하기 위해 SPeCtrum이라는 새로운 프레임워크를 도입하여 개인의 다차원적인 자기 개념을 통합한 LLM 에이전트 페르소나(persona)를 구성하고자 합니다. SPeCtrum은 사회적 정체성(Social Identity), 개인적 정체성(Personal Identity), 개인 생활 맥락(Personal Life Context)의 세 가지 핵심 요소로 이루어져 있습니다.

- **Technical Details**: SPeCtrum의 세 가지 구성 요소는 각각 고유하면서도 상호 연결된 정체성의 측면을 기여합니다. 연구에서는 자동화된 평가와 인간 평가를 통해 SPeCtrum의 정체성 표현 효과를 검증했습니다. 인기 있는 드라마 캐릭터를 이용한 자동화된 평가에서는 개인 생활 맥락(C)이 사회적 정체성(S)이나 개인적 정체성(P)만을 사용하는 경우보다 캐릭터의 정체성을 더 효과적으로 모델링하는 것으로 드러났습니다.

- **Performance Highlights**: 하지만 실제 인물을 대상으로 한 인간 평가에서는 전체 SPC 조합이 C 단독보다 더 포괄적인 자기 개념을 제공한다는 결과를 보여주었습니다. 연구 결과는 개인 생활 맥락(C)만으로도 기본적인 정체성 시뮬레이션을 수행할 수 있지만, S, P, C를 통합할 경우 현실 세계의 정체성 표현이 더욱 진정성 있고 정확해질 수 있음을 제안합니다. 전반적으로 SPeCtrum은 LLM 에이전트에서 개인들을 시뮬레이션하는 데 구조화된 접근 방식을 제공하여 개인 맞춤형 인간-AI 상호작용을 가능하게 하고, 시뮬레이션 기반 행동 연구의 현실성을 개선합니다.



### Quality-Aware Decoding: Unifying Quality Estimation and Decoding (https://arxiv.org/abs/2502.08561)
Comments:
          Under Review

- **What's New**: 이번 논문에서는 NMT(Neural Machine Translation)에서 품질 추정(QE, Quality Estimation) 모델을 사용한 새로운 연구 방향을 제안합니다. 기존의 접근 방식들은 후보 번역을 여러 샘플링하여 제공했으나, decode(디코딩) 과정에 QE 모델을 직접 통합한 사례는 없었습니다. 저자는 새로운 토큰 수준의 QE 모델을 개발하여 부분 번역에 대한 신뢰성 있는 점수를 매길 수 있도록 하였습니다.

- **Technical Details**: 저자는 단방향 QE 모델을 구축했으며, 이는 디코더 모델이 본질적으로 부분 시퀀스에서 효율적으로 학습될 수 있도록 설계되었습니다. 이와 함께 제안된 디코딩 전략은 품질 인식 디코딩을 위해 QE 모델을 통합하여 사용합니다. 결과적으로 이 방법은 최첨단 QE 모델과 비교할 때 번역 품질이 향상됨을 보여주었습니다.

- **Performance Highlights**: 번역 품질 개선은 N-best 리스트 재순위화에 비해 최대 $1.39$ XCOMET-XXL 경험치를 향상시키는 결과로 나타났습니다. 특히 문서 번역 작업에서는 N-best 리스트의 질이 일반적으로 최적이 아니므로, 이번 접근 방식이 상당한 이점을 제공함을 입증하였습니다.



### LLMs can implicitly learn from mistakes in-contex (https://arxiv.org/abs/2502.08550)
- **What's New**: 이번 연구는 LLMs가 수학적 추론 과제에서 실수로부터 암묵적으로 배울 수 있는지를 조사합니다. 혁신적인 접근 방식으로, 잘못된 정답과 올바른 정답을 단순히 나열할 때 모델이 성능이 더 좋다는 것을 발견했습니다. 이는 많은 기존 연구에서 강조되는 명시적 학습 방식과 대비되는 결과로, LLMs가 실수에 대한 명시적 피드백 없이도 학습이 가능하다는 것을 보여줍니다.

- **Technical Details**: 연구에서 제안된 'prompting for implicit learning'은 잘못된 답안과 올바른 답안을 동시에 보여주며, 모델이 스스로 이들 사이의 차이를 유추하도록 합니다. 이 방법은 코어 미국의 Chain-of-Thought(CoT) 프롬프트와 비교하여 대부분의 경우 성능이 우수하다는 점에서 주목받습니다. 또한, 다양한 LLM과 수학적 추론 데이터셋을 통해 결과의 안정성을 검증하였습니다.

- **Performance Highlights**: 실험 결과, LLMs는 제공된 잘못된 답안과 올바른 답안을 바탕으로 유도된 즉각적인 추론을 통해 높은 성과를 나타냈습니다. 이 과정에서 인간 평가자들로부터 모델이 생성한 이론 또한 높은 점수를 받아, LLMs가 암묵적으로 양질의 수정 이론을 유추할 수 있음을 확인했습니다. 이러한 결과는 LLMs의 가능성을 새롭게 규명하며 기존의 피드백 기반 접근 방식에 대한 의문을 제기합니다.



### Faithful, Unfaithful or Ambiguous? Multi-Agent Debate with Initial Stance for Summary Evaluation (https://arxiv.org/abs/2502.08514)
- **What's New**: 이번 연구에서는 요약의 신뢰성을 평가하기 위해 Madisse라는 새로운 접근 방식을 제안합니다. 여러 개의 LLM 기반 에이전트가 임의의 초기 입장을 부여받고 각자가 자신의 입장을 정당화하기 위한 주장을 펼치는 다자간 논쟁을 통해 결론에 도달하게 됩니다. 이러한 방법은 더 높은 다양성과 의미 있는 논쟁을 촉진하며, 최종적으로 더 많은 오류를 식별할 수 있게 합니다. 연구는 또한 요약의 신뢰성이 항상 Faithful 또는 Unfaithful로 단순히 분류될 수 없는 경우가 많다는 점을 강조합니다.

- **Technical Details**: Madisse는 LLM 기반 에이전트들이 초기 입장에서 시작하여 상호 논쟁을 벌이는 구조로 설계되었습니다. 이 과정에서 에이전트들은 신뢰성 라벨에 맞게 사실 여부를 확인하고 불일치를 해결하려고 합니다. 연구에서는 요약의 신뢰성을 평가할 때 발생할 수 있는 모호성을 새로운 차원으로 도입하고 이를 정의하는데 주목했습니다. 모호한 요약을 식별하기 위해 세부적인 분류법(taxonomy)과 주석이 달린 데이터셋을 제공하며, 이는 TofuEval MeetingBank 데이터셋을 확장한 것입니다.

- **Performance Highlights**: 실험 결과, Madisse는 비모호적인 상황에서 단일 LLM 및 다중 LLM 설정보다 더 많은 오류를 식별하며 성능이 유의미하게 향상됩니다. 또한, 이 논쟁 접근 방식은 모호한 사례를 식별하는 데 도움을 줄 뿐만 아니라 비모호적인 요약에 대한 정확도와 IAA를 높이는 데에도 강력한 성과를 보여줍니다. 이는 신뢰성 평가의 새로운 가능성을 열어주며, LLMs의 오류 탐지 능력을 효율적으로 활용할 수 있는 길을 제시합니다.



### Measuring Diversity in Synthetic Datasets (https://arxiv.org/abs/2502.08512)
- **What's New**: 이 논문에서는 DCScore라는 새로운 방법을 소개하여 기존의 합성 데이터셋의 다양성을 분류 관점에서 평가하는 방법을 제안합니다. DCScore는 각 샘플의 상호 관계를 활용하여 데이터셋 내 다양성을 측정하며, 이론적으로 검증된 여러 공리를 만족합니다. 이러한 접근법은 합성 데이터셋의 품질을 높이고 모델 성능을 향상시키는 데 중요한 역할을 합니다.

- **Technical Details**: DCScore는 분류 작업으로서 다양성 평가를 수행하고, 이를 통해 합성 데이터셋의 각 샘플이 평가 결과에 미치는 영향을 포괄적으로 분석할 수 있습니다. 이 방법은 Leinster & Cobbold(2012)에 의해 제안된 네 가지 공리를 충족하여, 유효 숫자, 동일한 샘플, 대칭성 및 단조성을 보장합니다. 또한, DCScore는 기존 방법들과 비교할 때 계산 비용을 상당히 줄이는 데 성공했습니다.

- **Performance Highlights**: DCScore는 다양한 다양성 가짜 진리와 더 강력한 상관관계를 보이는 실험 결과를 보였습니다. 특히, DCScore는 기본 메트릭들보다 더 강한 효과를 발휘하며, 합성 데이터셋의 다양성을 높이 평가합니다. 이 방법은 실제 경험적 및 이론적 증거를 통해 낮은 계산 비용의 이점을 입증했습니다.



### Explanation based In-Context Demonstrations Retrieval for Multilingual Grammatical Error Correction (https://arxiv.org/abs/2502.08507)
Comments:
          Accepted by NAACL 2025 main conference

- **What's New**: 이 논문에서는 자연어 문법 오류 설명(Grammatical Error Explanations, GEE)을 기반으로 한 새로운 정보 검색(method) 방법을 제안합니다. 이 연구는 입력 텍스트와 GEE 간의 유사성을 통해 적절한 few-shot 예시를 검색함으로써 기존의 의미 기반 검색 기술보다 우수한 성능을 나타냅니다. 특히, LLM(대규모 언어 모델)의 사전학습 데이터와 상관없이 추가 훈련이 필요하지 않으며, 여러 언어에 적용할 수 있는 장점이 있습니다.

- **Technical Details**: 제안된 방법은 GEE를 쿼리(query)와 키(key)로 사용하여 비슷한 문법 오류를 가진 예시를 검색하는 방식으로 작동합니다. 이 프로세스는 문법 검사를 통해 초기 설명을 생성한 후, GEE 데이터베이스와의 매칭을 통해 실행됩니다. 실험 결과, 이 방법은 5개 언어에서 기존의 BM25 기반 검색 기술을 능가하며, LLM의 few-shot 성능을 향상시킵니다.

- **Performance Highlights**: GEC(문법 오류 교정) 분야에서 제안한 방법은 F0.5 서브스크립트 점수에서 기존 기술들을 초월하며, 추가 훈련 없이 효과적인 성능을 보입니다. 연구 결과, 제안된 방법은 다양한 언어의 GEC 과제를 효과적으로 처리할 수 있으며, 시스템 해석 가능성을 증가시키는 데 기여합니다.



### Salamandra Technical Repor (https://arxiv.org/abs/2502.08489)
- **What's New**: Salamandra는 20억, 70억, 400억 파라미터의 세 가지 크기로 제공되는 오픈 소스 (open-source) 디코더 전용 대형 언어 모델입니다. 이 모델은 35개 유럽 언어와 코드로 이루어진 다양한 멀티링구얼 (multilingual) 데이터로부터 처음부터 훈련되었습니다. 공공 분야의 지침 데이터에 대해 세분화된 체크포인트를 추가로 공개하여 채팅 응용 프로그램에 활용할 수 있도록 하였습니다.

- **Technical Details**: 모델 훈련에 사용된 데이터는 다양한 출처에서 수집된 오픈 액세스 (open-access) 데이터로 구성되어 있습니다. 또한, 다중 모달리티 (multimodality)에 대한 초기 실험 결과도 공유하여 Salamandra의 잠재적 응용 프로그램을 입증할 수 있는 증거를 제공합니다. 이 기술 보고서에서는 설계 선택, 데이터 조정 전략 및 평가 방법론과 관련한 모든 세부정보를 공개하여 열린 과학 (open science)을 촉진하고자 합니다.

- **Performance Highlights**: Salamandra는 다언어 (multilingual) 벤치마크에 대한 광범위한 평가에서 유사한 크기의 오픈 소스 모델과 비교할 때 경쟁력 있는 성능을 달성했습니다. 또한, 훈련 및 평가 스크립트를 공개하여 연구자들이 자유롭게 사용할 수 있도록 하였으며, Apache 2.0 라이센스에 따라 모든 모델을 배포하여 상업적 활용도 지원하였습니다.



### Enhancing Auto-regressive Chain-of-Thought through Loop-Aligned Reasoning (https://arxiv.org/abs/2502.08482)
Comments:
          work in progress

- **What's New**: 최근 연구에서는 Chain-of-Thought (CoT) 프롬프트가 언어 모델의 추론 능력을 향상시키는 강력한 기법으로 등장했습니다. 하지만 긴 CoT 경로를 생성하는 것은 도전 과제가 되고 있습니다. 본 논문에서는 Looped Transformer를 활용한 RELAY (REasoning through Loop Alignment iterativelY)를 제안하여, CoT 추론 단계와 루프 반복을 정렬하고 추가적인 중간 감독 학습을 통해 길이 일반화(length generalization) 능력을 강화합니다.

- **Technical Details**: RELAY는 루프된 Transformer의 장점을 최대한 활용하여 오토레그레시브(auto-regressive) 모델이 더 긴 추론 체인을 처리할 수 있도록 돕는 새로운 프레임워크입니다. 주요 혁신은 루프된 Transformer 모델이 여러 작업에서 일반적인 추론기의 역할을 수행할 수 있음을 실증적으로 입증하는 것과, CoT 추론 단계와 루프된 Transformer 간의 반복 간 정렬을 통해 훈련 길이를 초과하는 문제에 대한 정확한 추론 체인을 생성할 수 있도록 하는 것입니다.

- **Performance Highlights**: 광범위한 실험을 통해 RELAY 접근법이 생성한 고품질의 추론 체인을 통해 오토레그레시브 Transformers의 추론 능력을 크게 향상시키는 것을 입증했습니다. 이러한 성과는 더욱 복잡한 문제 해결에 있어 모델의 성능을 개선하고, 다양한 언어 작업에서 일관된 성능을 유지할 수 있도록 합니다.



### Examining Spanish Counseling with MIDAS: a Motivational Interviewing Dataset in Spanish (https://arxiv.org/abs/2502.08458)
Comments:
          To appear in NAACL 2025 Main Conference

- **What's New**: 이 논문에서는 문화적 및 언어적 요인이 상담에 미치는 영향을 연구하며, 영어에서 개발된 상담 분석 결과가 다른 언어에도 적용될 수 있는지 확인하고자 합니다. 이를 위해 스페인어로 진행된 상담 대화를 포함한 MIDAS (Motivational Interviewing Dataset in Spanish)라는 새로운 데이터셋을 소개합니다. 이 데이터셋은 전문가의 주석이 포함된 상담 반영 및 질문을 위한 자료로, 상담사의 행동을 영어와 스페인어로 비교하고 분석하는 데 활용됩니다.

- **Technical Details**: 논문에서는 MIDAS 데이터셋을 수집하기 위해 YouTube에서 스페인어로 된 MI (Motivational Interviewing) 인터뷰 비디오를 수집했습니다. 데이터셋은 총 74개의 스페인어 상담 대화로 구성되어 있으며, 대화 참여자를 상담사와 고객으로 구분하여 수작업으로 주석을 달았습니다. 또한, 스페인어와 영어의 상담사 행동을 비교하기 위해 언어 기반 비교 분석을 수행하며, counselor quality 및 대화에서의 반영과 질문 비율을 평가합니다.

- **Performance Highlights**: 결과적으로, 스페인어 데이터로 훈련된 모델이 영어로 훈련된 모델보다 성능이 뛰어난 것으로 나타났습니다. 스페인어 상담사와 고객 간의 대화에서 평균 단어 교환 비율이 더 다양하며, 이는 스페인어 대화의 독특한 상호작용 역학을 시사합니다. 이러한 분석 결과는 상담 품질 향상을 위한 NLP 도구 개발에 있어 언어별 데이터셋의 필요성을 강조합니다.



### Towards Prompt Generalization: Grammar-aware Cross-Prompt Automated Essay Scoring (https://arxiv.org/abs/2502.08450)
Comments:
          NAACL 2025 (Findings)

- **What's New**: 이번 연구에서는 문법 인식 기반의 교차 프롬프트 성적 평가(GAPS)를 제안합니다. 기존의 프롬프트 특화 모델과 달리, 본 모델은 문법 오류 수정(Grammar Error Correction) 기법을 통합하여 프롬프트에 의존하지 않는 Generic 에세이 표현을 학습합니다. GAPS는 원본 에세이와 수정된 에세이를 모두 참고하여 모델이 훈련 중에 일반적인 특징에 집중할 수 있도록 합니다.

- **Technical Details**: GAPS는 두 가지 주요 단계로 구성됩니다: (1) 에세이 수정 및 (2) 문법 인식 에세이 성적 평가입니다. T5 기반의 사전 훈련된 GEC 모델을 사용하여 에세이에서 발견된 문법 오류를 수정하고, 수정된 텍스트와 원본 텍스트를 동시에 스코어링 모델에 입력합니다. 에세이 인코더는 원본과 수정된 각각의 에세이를 처리하면서, 효과적인 정보 공유를 위해 다층적 구조를 채택합니다.

- **Performance Highlights**: 실험 결과, GAPS는 프롬프트 비의존적이며 문법 관련 특성에서 두드러진 성능 향상을 보여주었습니다. 특히, Conventions와 Sentence Fluency와 같은 프롬프트 비의존적 특성에서 현저한 개선이 나타났습니다. GAPS는 가장 도전적인 교차 프롬프트 상황에서 notable QWK 향상을 달성하며, 이는 보지 못한 프롬프트에 대한 평가에서 강점을 나타냅니다.



### Better Embeddings with Coupled Adam (https://arxiv.org/abs/2502.08441)
Comments:
          17 pages, 8 figures; figures corrected

- **What's New**: 이 논문에서는 대형 언어 모델(LLM)에서의 단어 임베딩의 비등방성을 초래하는 원인으로 Adam 최적화 알고리즘의 두 번째 모멘트를 지적하고, 이를 해결하기 위한 Coupled Adam이라는 수정된 최적화 기법을 제안합니다. Coupled Adam은 임베딩 매개변수에 대해 비등방성 문제를 완화하기 위해 특별히 설계된 효율적인 조정입니다.

- **Technical Details**: LLM에서의 단어 임베딩은 주어진 토큰 시퀀스를 입력으로 받아 다음 토큰을 예측하는 과정을 통해 학습됩니다. 그러나 embedding 벡터가 원점에서 멀리 떨어진 작은 부분 공간에 군집화되는 비등방성을 관찰하였습니다. Adam 최적화 기법은 희소 데이터에서 잘 작동하지만, 이로 인해 자주 등장하는 단어에 비해 드물게 등장하는 단어의 업데이트 벡터가 상대적으로 늘어나 비등방성을 야기합니다.

- **Performance Highlights**: Coupled Adam을 사용한 결과, 단어 임베딩의 품질을 유의미하게 향상시키는 것을 입증했습니다. 이 방법은 또한 충분히 큰 데이터셋에서 상류(upstream) 및 하류(downstream) 성능을 개선하는 데 긍정적인 영향을 미치는 것으로 나타났습니다.



### From Haystack to Needle: Label Space Reduction for Zero-shot Classification (https://arxiv.org/abs/2502.08436)
Comments:
          Under review at ICML 2025

- **What's New**: 이번 논문에서는 Label Space Reduction (LSR)이라는 새로운 방법을 제시하여 대형 언어 모델(LLMs)의 제로샷 분류 성능을 향상시키고자 합니다. LSR은 후보 클래스를 체계적으로 순위화하고 감소시켜 모델이 가장 관련성이 높은 옵션에 집중할 수 있도록 합니다. 이 방법은 라벨 공간 표현을 동적으로 최적화하여 실험을 통해 Llama-3.1-70B와 Claude-3.5-Sonnet에서 각각 7.0%와 3.3%의 매크로 F1 점수 개선을 가져왔습니다.

- **Technical Details**: LSR은 분류 라벨 공간을 개선하기 위해 후보 클래스를 순위화하고 감소시키는 혁신적인 반복 시스템을 개발합니다. 또한, LSR의 계산 오버헤드를 줄이기 위해 모델을 확률적 분류기로 증류하는 방안을 제안하여 효율적인 추론을 가능하게 합니다. 이 방법은 출력이 생성될 때 동적으로 적응할 수 있는 정성적인 라벨 공간을 생성합니다.

- **Performance Highlights**: 여덟 개의 벤치마크에서 실험한 결과, LSR은 기존 제로샷 분류 기준에 비해 Llama-3.1-70B에서 평균 7.0%(최대 14.2%) 및 Claude-3.5-Sonnet에서 3.3%(최대 11.1%)의 매크로 F1 점수 향상을 보여주었습니다. 이러한 성과는 LLM이 중요한 정보에 보다 효과적으로 주의를 집중할 수 있도록 해주며, 문제 해결 과정에서의 추론을 개선하는데 기여합니다.



### A Semantic Parsing Algorithm to Solve Linear Ordering Problems (https://arxiv.org/abs/2502.08415)
Comments:
          3 figures, 9 pages main paper and 6 pages references and appendix

- **What's New**: 이번 연구는 선형 순서 문제(linear ordering problem)에 대한 의미적 파서(semantic parser) 알고리즘을 개발하여, 주어진 전제(premise)와 후보 진술(candidate statement)을 일차 논리(first-order logic)로 변환한 후, 제약 논리 프로그래밍(constraint logic programming)을 통해 진위를 추론하는 방법을 제시합니다. 이 알고리즘은 Heim과 Kratzer의 구문 기반 조합적 정형 의미론(compositional formal semantics)를 계산 알고리즘으로 전환합니다. 연구에서 제안한 Formal Semantic Logic Inferer (FSLI) 시스템은 BIG-bench의 논리 추론(logical_deduction) 여러 선택 문제에서 완전 정확도를 달성하며, 이로써 일차 논리 구조에 의해 구동되는 의미적 파서 개발의 이점을 보여줍니다.

- **Technical Details**: FSLI 시스템은 의미적 파서와 전용 논리 추론 모듈(logic inference module)을 통합하여 구성되며, 자연어를 공식적인 논리 표현으로 변환하는 과정에서 높은 정확도를 목표로 합니다. 이 시스템은 첫째, 주어진 전제와 후보 진술을 일차 논리 구조로 변환한 후, 두 번째로 해당 구조를 통해 논리적 추론을 수행합니다. 의미 표현 검증을 위한 동적 구성요소(context)는 개체(sync)와 개체 설명 간의 매핑으로 표현되며, 이는 문맥에 따라 유동적으로 변형됩니다.

- **Performance Highlights**: FSLI는 BIG-bench의 'logical_deduction' 벤치마크 데이터세트에서 테스트되어 의미적 파싱 능력과 논리적 추론의 올바름을 평가합니다. 각 문제는 개체의 배열을 위한 단일 유효 시퀀스만을 허용하므로, FSLI는 완전한 정확도를 달성할 수 있으며, 이는 기존의 LLM(예: GPT-4)의 67.06% 및 Logic-LM의 87.63%와 대조됩니다. 이러한 결과들은 구조화된 대안을 탐구하고 논리적 일관성을 우선시하는 것이 의미적 파싱 알고리즘의 발전에 얼마나 중요한지를 강조합니다.



### IssueBench: Millions of Realistic Prompts for Measuring Issue Bias in LLM Writing Assistanc (https://arxiv.org/abs/2502.08395)
Comments:
          under review

- **What's New**: 이 논문은 IssueBench라는 새로운 도구를 소개합니다. 이는 2.49백만 개의 현실적인 프롬프트(prompt) 세트를 통해 대형 언어 모델(LLM)의 문제 편향(issue bias)을 측정할 수 있도록 설계되었습니다. 특히, 3.9천 개의 템플릿(template)과 실제 사용자 상호작용에서 추출한 212개의 정치적 이슈를 기반으로 합니다.

- **Technical Details**: IssueBench는 사용자 상호작용에서 나타나는 이슈 편향을 정량화할 수 있도록 설계되었습니다. 연구진은 최신 LLM에서 편향이 일반적이며 지속적이라는 것을 보여주었고, 여러 모델 간의 편향이 유사하다는 사실도 발견했습니다. 또한, 모든 모델은 특정 이슈에 대해 미국 민주당원(voter)의 의견에 더 일치하는 경향이 있음을 밝혔습니다.

- **Performance Highlights**: IssueBench는 다른 이슈, 템플릿 또는 작업을 포함하도록 쉽게 조정될 수 있습니다. 이를 통해 LLM 편향에 대한 논의에 필요한 강력하고 현실적인 증거를 제공하고, 편향 문제를 해결하는 데 기여할 수 있기를 기대합니다.



### Unveiling Global Discourse Structures: Theoretical Analysis and NLP Applications in Argument Mining (https://arxiv.org/abs/2502.08371)
- **What's New**: 이 논문은 'Argument Mining'이라는 과정에 대해 논의하며, 글로벌 담론 구조를 탐지하고 추출하고 표현하는 방법을 제안합니다. 특히 설득력 있는 텍스트에서 논증 구조가 일관성을 가지는 것이 중요하다는 점을 강조합니다. 기존의 연구들을 요약하고, 현재의 논증 구성 요소 추출 및 분류 방법의 한계를 지적하며, 새로운 NLP 기술을 통해 이러한 문제를 해결할 수 있는 아키텍처를 제시합니다.

- **Technical Details**: 설득적 글쓰기에서 담론 구조는 글로벌 일관성을 결정짓는 중요한 요소입니다. 논문에서는 텍스트에서 논증 구조를 추출하기 위한 세 가지 작업, 즉 Span identification, Component classification 및 Relation classification을 제안합니다. 이러한 단계들은 상호 의존적이며, 각 단계의 성능은 이전 단계에 크게 영향을 미칩니다. 고급 자연어 처리(NLP) 기법을 통해 이러한 작업을 수행하는 모델이 개발됩니다.

- **Performance Highlights**: 현재까지 전반적인 목적을 위한 Argument Mining 모델이 제시된 바가 없으며, 이를 통해 다양한 논증 스타일의 변수를 고려할 수 있는 일반화 가능성에 대한 의문이 제기되고 있습니다. 기존의 코퍼스는 주로 단편적 논증에 초점을 두고 있으며, 다자간 대화(multilogue)에 대한 연구는 부족한 상황입니다. 또한, 논증 구조의 다양한 스타일 간의 차이는 일반화를 더욱 어렵게 만들고, 이는 corpus 구축과 연구의 진전을 방해하는 요소입니다.



### Top-Theta Attention: Sparsifying Transformers by Compensated Thresholding (https://arxiv.org/abs/2502.08363)
Comments:
          8 pages, 11 figures, work under submission

- **What's New**: 이번 연구에서는 Transformer 기반 대형 언어 모델의 self-attention 메커니즘의 비효율성을 극복하기 위한 새로운 접근법인 Top-Theta Attention (Top-$\theta$)을 소개합니다. 이 방법은 중요하지 않은 attention 요소를 선택적으로 가지치기하여 효율성을 크게 향상시키며, 모델의 정확도를 유지하는 특징이 있습니다. 특히, generative decoding 동안 V-cache row의 수를 3배, prefill 단계에서 attention 요소의 수를 10배 줄여주는 성과를 보여줍니다.

- **Technical Details**: Top-$\theta$는 주어진 임계값에 따라 attention 요소를 가지치기(thresholding)하여 Sparse한 구조를 활용합니다. 이 방법은 Top-k attention과 달리 전체 벡터 의존성을 제거하며, 이는 모델을 고성능의 커널과 분산 추론에 적합하게 만듭니다. 또한, 모델의 리트레이닝 없이 간단한 보정(calibration) 절차만으로 적용할 수 있어, 여러 데이터셋 간의 분포 변화에도 강한 저항성을 보입니다.

- **Performance Highlights**: Top-$\theta$의 적용 결과, LLaMA2와 LLaMA3 모델에 대해 3배 적은 V-row와 10배 적은 attention 요소로도 동일한 정확도를 유지할 수 있음을 확인하였습니다. 본 연구는 개발한 효율적인 수치 보상 기법들을 통해 가지치기가 진행되는 동안 모델의 정확도를 효과적으로 보존하는 방법도 제안합니다. 이 연구는 대형 언어 모델의 메모리 및 계산 효율성을 높이는 중요한 기여를 할 것으로 기대됩니다.



### Systematic Knowledge Injection into Large Language Models via Diverse Augmentation for Domain-Specific RAG (https://arxiv.org/abs/2502.08356)
Comments:
          22 pages, 14 tables, to be published in NAACL 2025

- **What's New**: 이 논문에서는 Retrieval-Augmented Generation (RAG) 방법을 통해 Large Language Models (LLMs)에 도메인 지식을 통합하는 새로운 프레임워크를 제안합니다. 기존의 문제인 검색 오류로 인한 환각(hallucinations) 및 잘못된 답변을 해결하기 위해 모델의 미세 조정(fine-tuning)과 함께 훈련 데이터를 두 가지 방법으로 증강하는 접근 방식이 소개됩니다. 이러한 새로운 접근 방식은 모델이 검색된 정보를 무시하거나 의존할 시점을 학습하도록 합니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 단계로 구성됩니다. 첫째, context augmentation을 통해 주어진 QA 쌍에 대해 검색된 정보의 관련성을 다르게 하여 여러 개의 훈련 샘플을 생성합니다. 둘째, knowledge paraphrasing을 통해 동일한 질문에 대해 여러 답변을 미세 조정하여 LLM이 전문 지식을 더 잘 내재화하도록 합니다. 또한, 미세 조정으로 인한 재앙적 망각(catastrophic forgetting)을 방지하기 위해 도메인 특정 식별자(domain-specific identifier)를 질문에 추가하고 일반 QA 쌍을 포함하는 replay buffer를 활용합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 기존 기술 대비 10% 이상의 상대적인 성능 향상을 보이며 token-level recall을 극대화하는 효과가 있음을 입증하였습니다. 이 과정에서 LLM의 일반화 능력은 유지되고, 보다 정확하고 관련성 높은 응답을 생성하는데 기여합니다. 이를 통해 도메인 지식이 통합된 보다 신뢰할 수 있는 대화형 AI 시스템의 발전을 기대할 수 있습니다.



### Contextual Compression Encoding for Large Language Models: A Novel Framework for Multi-Layered Parameter Space Pruning (https://arxiv.org/abs/2502.08323)
- **What's New**: 본 연구는 모델 압축에 대한 새로운 접근 방식을 제안합니다. Contextual Compression Encoding (CCE) 기법은 다층 파라미터 공간 축소 전략을 통해 중복 표현을 체계적으로 제거합니다. 기존의 압축 방법들이 개별 파라미터에 초점을 맞춘 데 비해 CCE는 정교한 컨텍스트 정보를 보존하며 파라미터 클러스터를 식별할 수 있는 동적 인코딩 방식을 활용합니다. 이를 통해 자연어 생성에서의 연관성과 표현을 유지하면서 계산 부담을 줄일 수 있습니다.

- **Technical Details**: 연구진은 CCE 기법을 통해 파라미터 중복성을 판단하기 위한 형식적인 메트릭을 개발하였습니다. 이 메트릭은 여러 레이어에 걸친 컨텍스트 유사도에 기반하여 불필요한 표현을 선택적으로 제거할 수 있게 합니다. CCE는 고유한 구조적 인코딩 메커니즘을 통해 모델의 표현 공간을 재구성하여 핵심적인 언어 및 의미 기능을 유지하면서도 계산 효율성을 높입니다. 이 기술은 재교육이나 외부 감독 없이도 높은 충실도의 컨텍스트 표현을 유지합니다.

- **Performance Highlights**: 실험 결과 CCE를 적용한 모델은 파라미터 수 감소와 계산 효율성 향상, 그리고 언어 표현력 유지에서 긍정적인 결과를 보였습니다. 최첨단 오픈 소스 LLM을 사용하여 여러 구성에 대해 CCE를 체계적으로 구현하고 분석하였으며, 에너지 소비 및 추론 지연 시간 감소를 입증하였습니다. 자원을 제한한 환경에서의 배치 시나리오에서 메모리 사용 감소가 뚜렷하게 나타나, 더 확장 가능한 구현이 가능하다는 것을 보여주었습니다.



### MultiProSE: A Multi-label Arabic Dataset for Propaganda, Sentiment, and Emotion Detection (https://arxiv.org/abs/2502.08319)
Comments:
          12 pages, 3 figuers, 4 tabels

- **What's New**: 이번 논문에서는 아랍어로 된 다중 라벨 선전(Multi-label Propaganda) 데이터를 다룬 첫 번째 데이터셋인 MultiProSE가 소개되었습니다. 이 데이터셋은 기존 아랍어 선전 데이터셋인 ArPro를 기반으로 하며, 각 텍스트에 대한 감정(Sentiment) 및 감정(Emotion) 주석이 추가되었습니다. 이를 통해 아랍어로 이루어진 선전 감지 연구에 필요한 데이터 자원을 확장하려는 노력이 포함되어 있습니다.

- **Technical Details**: MultiProSE 데이터셋은 8,000개의 주석이 달린 뉴스 기사를 포함하며, 현재까지 가장 큰 선전 데이터셋으로 자리 잡고 있습니다. 연구진은 GPT-4o-mini와 같은 대형 언어 모델(LLMs) 및 BERT 기반의 프리 트레인(pre-trained) 언어 모델(PLMs)을 사용하여 여러 개의 기준선(Baseline)을 개발하였습니다. 데이터셋과 주석 가이드라인, 소스 코드는 공개되어 아랍어 모델 연구와 개발에 기여할 수 있도록 하였습니다.

- **Performance Highlights**: MultiProSE의 출시는 아랍어 선전의 탐지 및 분석을 위한 새로운 기준을 설정하는 데 중요한 역할을 하고 있습니다. 이 데이터셋은 향후 아랍어 언어 모델의 성능 향상 뿐만 아니라, 뉴스 매체에서 다양한 의견 차원이 어떻게 상호작용하는지에 대한 이해를 돕는 데 기여할 예정입니다. 이러한 연구는 선전 및 감정 인식 분야에서의 진전을 더욱 촉진할 것으로 기대됩니다.



### Mitigating Hallucinations in Multimodal Spatial Relations through Constraint-Aware Prompting (https://arxiv.org/abs/2502.08317)
Comments:
          19 pages, accepted to NAACL Findings

- **What's New**: 이 논문에서는 공간적 관계의 환각(spatial relation hallucinations)을 줄이기 위해 제약 인식 프롬프트 프레임워크(constraint-aware prompting framework)를 제안합니다. 특히, 두 가지 제약 조건을 도입하는데, 이들은 쌍방향 제약(bidirectional constraint)과 전이성 제약(transitivity constraint)입니다. 쌍방향 제약은 두 객체 간의 관계가 양방향에서 일관되도록 보장하며, 전이성 제약은 다수의 객체 간의 관계에 대한 논리적 일관성을 유지합니다. 이러한 제약을 통합하여 LVLMs는 더 공간적으로 일관된 출력을 생성할 수 있습니다.

- **Technical Details**: 제안하는 방법은 주로 이미지-질문 쌍을 입력으로 하는 공간적 관계 이진 VQA를 위해 설계되었습니다. 연구팀은 약속된 구조에 따라 지침과 출력 형식을 설정하고, 해당 질문에 대한 공간적 관계를 분석하는 것을 목표로 합니다. 논문에서는 제로샷 체인 오브 씽킹(zero-shot chain-of-thought) 프롬프트를 사용하여 LVLMs가 감지된 공간적 관계에 기반하여 효과적으로 추론하도록 합니다. 이 과정에서 수평, 수직 및 깊이 관계를 분석하도록 명시적인 지침을 제공합니다.

- **Performance Highlights**: 세 개의 널리 사용되는 공간적 관계 데이터셋을 평가하여 제안된 방법의 성능을 검증하였습니다. 결과적으로 제안한 방법은 기존의 접근 방법보다 성능이 향상되었으며, 전체 방법이 다른 두 제약 방법보다 우수한 성능을 보임을 확인했습니다. 이 연구에서는 다양한 방법 변형의 성과를 분석하고, 데이터셋 간 성과의 변동성을 강조하여 제안한 접근법의 효과성을 입증하고 있습니다.



### Compromising Honesty and Harmlessness in Language Models via Deception Attacks (https://arxiv.org/abs/2502.08301)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 속임수 행동을 탐지하고 이를 활용하는 새로운 공격 방법을 소개합니다. 이러한 속임수 능력은 과거에 드물게 관찰되었으나, 본 연구는 이러한 성향을 극대화하는 방법을 통해 전혀 새로운 차원의 위험을 드러냈습니다. 이 연구는 특히 고유한 주제에 대한 프롬프트(prrompt)에 부합하여 사용자에게 잘못된 정보를 제공하는 방법을 탐구합니다.

- **Technical Details**: 속임수 공격(deception attacks)을 통해 LLMs의 진실성과 무해함을 저하시킬 수 있는 방법론을 제시하고 있습니다. 세세한 조정(fine-tuning)을 통해 특정 주제에 대해 의도적으로 사용자에게 잘못된 정보를 전할 수 있는 모델을 조정합니다. 이 연구는 이러한 모델들이 다수의 대화(turns)에서도 일관되게 속임수를 사용할 수 있는지를 평가하며, 그 결과는 혼재된 양상을 보입니다.

- **Performance Highlights**: 연구 결과, 조정된 모델들은 사용자에게 혐오 발언(hate speech), 고정관념(stereotypes) 등 유해한 콘텐츠를 생성하는 경향을 보였습니다. 또한, 대화형 인터페이스(dialogues)에서의 속임수 행위는 수백만 사용자와의 상호작용에서 신뢰성을 보장하기 위해 필수적으로 대처해야 할 과제로 지적됩니다. 이 연구는 LLM이 사용자에게 미치는 영향을 고려하여 이들을 속임수 공격으로부터 보호하는 것이 중요함을 강조하고 있습니다.



### Redefining Simplicity: Benchmarking Large Language Models from Lexical to Document Simplification (https://arxiv.org/abs/2502.08281)
- **What's New**: 이번 연구는 텍스트 간소화(Text Simplification) 분야에서 대형 언어 모델(LLMs)의 성능을 최초로 종합적으로 분석한 것입니다. 연구는 Lexical, Syntactic, Sentence 및 Document simplification의 4가지 작업에서 LLM의 효과를 평가하며, 전통적인 비-LLM 접근법과의 비교를 실시하였습니다. LLM이 모든 작업에서 비-LLM 방법을 초월하는 결과를 보였으며, 인간 주석 데이터보다 더 나은 출력을 생성하는 경향이 있다고 밝혀졌습니다.

- **Technical Details**: 연구에서는 세 가지 유형의 언어 모델을 기준으로 설정하고, 실험을 통해 LLM의 성능을 분석했습니다. 특히 'Gemma2-2B', 'Llama3.1-70B', 'GPT-4o' 등 여러 개의 LLM을 활용하여 실행하였으며, 또한 자동화된 평가 지표와 사람의 평가를 모두 사용하였습니다. 실험 결과, 대형 LLM이 비-LLM 방식보다 우수한 성능을 나타냈으며, 경량화된 LLM이 특정 작업에서 뛰어난 효율을 보여주었습니다.

- **Performance Highlights**: LLM은 Lexical, Syntactic, Sentence, Document simplification의 네 가지 작업 모두에서 비-LLM 방법과 인간 주석 결과를 능가하는 성과를 기록했습니다. 특히 감량된 LLM 모델은 Sentence 및 Syntactic simplification에서 두각을 나타냈습니다. 이어서 연구에서 제안된 미래 연구 방향은 사용자 맞춤형 간소화 및 경량 모델 훈련 방식과 같은 여러 복잡한 텍스트 간소화 작업에 초점을 맞추고 있습니다.



### What Is That Talk About? A Video-to-Text Summarization Dataset for Scientific Presentations (https://arxiv.org/abs/2502.08279)
Comments:
          arXiv admin note: text overlap with arXiv:2306.02873 by other authors

- **What's New**: 본 논문은 과학 분야의 비디오-텍스트 요약을 위해 특별히 설계된 VISTA 데이터셋을 소개합니다. VISTA는 18,599개의 AI 컨퍼런스 발표와 해당 논문의 초록으로 구성되어 있으며, 멀티모달 학습에서의 더욱 효율적인 요약 생성을 목표로 합니다. 인공지능 비디오 요약 문제의 명확한 해결을 모색하기 위해 최신 대형 모델들이 benchmark 되었고, 계획 기반 프레임워크를 통해 요약 품질과 사실 일관성을 개선했습니다.

- **Technical Details**: VISTA 데이터셋은 컴퓨터 언어학 및 기계 학습의 주요 컨퍼런스에서 수집된 기록된 발표와 논문 초록 간의 짝을 이루는 18,599개의 쌍으로 구성되어 있습니다. 다양한 대형 모델에 대한 비교 실험을 통해 in-domain fine-tuning이 요약 성능을 개선시키고, 비디오 기반 모델이 텍스트 및 오디오 기반 모델보다 일반적으로 우수한 성능을 보임을 발견했습니다. 연구는 계획 기반 접근 방식을 통해 과학 초록의 기본 구조를 더 잘 포착할 수 있음을 보여줍니다.

- **Performance Highlights**: 인간과 자동화된 평가 모두에서 명시적 계획이 요약 품질과 사실적 일관성을 향상시킴을 확인했습니다. 계획 기반 접근 방식이 최신 SOTA 모델들보다 우수한 성능을 나타내었지만, 모든 후보 모델에서 여전히 사실 오류와 환각 문제에 직면해 있는 점이 지적되었습니다. VISTA 데이터셋을 통해 과학 비디오 요약의 도전과제를 제시하며, 이 분야에 대한 연구의 필요성을 강조했습니다.



### Dealing with Annotator Disagreement in Hate Speech Classification (https://arxiv.org/abs/2502.08266)
- **What's New**: 본 논문은 증오 발언(hate speech) 분류에 대한 주석자 간의 의견 불일치 문제를 심층적으로 다루고, 여러 가지 접근 방식을 평가합니다. 특히, 주석자 간의 불일치 문제 해결을 위한 새로운 방법론을 제안하며, 고품질의 데이터셋을 확보하기 위한 다양한 전략을 탐구합니다. 이 연구는 터키어 트윗을 기반으로 하여 필터링된 BERT 모델을 활용한 최신 성능 벤치마크 결과를 제공합니다.

- **Technical Details**: 본 연구는 주석 과정을 통해 발생하는 주관적 불일치 문제에 집중하며, 다양한 방법(예: 최대값, 최소값, 무작위 선택 및 평균)을 통해 가장 정확한 레이블을 결정하는 방안을 모색합니다. 또한 주석자의 신뢰도 차이를 고려하여 가중된 버전의 접근 방법도 평가합니다. 이를 통해 데이터셋의 품질을 향상시키고, 나아가 증오 발언 탐지 모델의 신뢰성을 높이는 데 기여하고자 합니다.

- **Performance Highlights**: 필요한 트레이닝 데이터 확보의 중요성을 강조하며, 제안된 방법론을 통해 공연별 감지 및 이해에서 최첨단 성과를 달성했습니다. 이 연구는 튼튼한 데이터셋을 바탕으로 한 정확한 증오 발언 탐지의 필요성을 재확인하고, 이를 통해 다양한 자연어 처리(NLP) 작업에서의 성능을 향상시키는 데 기여합니다.



### Exploring the Potential of Large Language Models to Simulate Personality (https://arxiv.org/abs/2502.08265)
Comments:
          Preprint submitted to Workshop on Customizable NLP (CustomNLP4U) on EMNLP2024

- **What's New**: 이번 연구는 LLM(대형 언어 모델)을 사용하여 Big Five 성격 모델에 따라 개인 특성을 모사하는 방법을 제시합니다. 연구 결과, 성격 관련 텍스트를 생성하는 것이 여전히 LLM에게 도전적인 과제임을 보여주었습니다. 따라서, 사전 정의된 Big Five 특성을 가진 생성 텍스트의 데이터셋과 LLM의 개인화된 대화 기능을 테스트하기 위한 분석 프레임워크를 제공합니다.

- **Technical Details**: 연구는 LLM의 성격 분석을 두 단계로 나누어 수행합니다. 첫 번째 단계에서는 성격 특성과 관련된 행동 간의 연결을 이해하는 능력을 평가하기 위해 성격 질문지를 사용합니다. 두 번째 단계에서는 유도된 성격에 대한 텍스트 생성을 LLM에 요청하여, 생성된 텍스트를 인간 평가, 자동 평가 및 언어적 특성 분석 등 다양한 방법으로 분석합니다.

- **Performance Highlights**: 테스트 결과, 일부 LLM 모델은 특정 성격 특성에 대한 정확한 성과를 보여주었으며, Claude 모델은 개방성과 성실성에 대해 높은 이해도를 보였습니다. GPT 시리즈 모델은 외향성의 높은 및 낮은 수준을 구별하는 능력이 뛰어났으며, GPT-4 Omni 모델은 우호성에 대한 구분에서 우수한 성능을 보였습니다. 이 연구 결과는 LLM이 개인 특성을 모사하고 이를 기반으로 사용자와의 대화를 더욱 풍부하게 만드는 데 기여할 것으로 기대됩니다.



### Inference-time sparse attention with asymmetric indexing (https://arxiv.org/abs/2502.08246)
- **What's New**: 본 논문에서는 자기-주의(self-attention) 메커니즘의 문제를 해결하기 위한 새로운 기법, SAAP(비대칭 분할을 이용한 자기-주의)를 제안합니다. SAAP는 키(key)와 쿼리(query) 벡터에 대해 각각 독립적인 분할을 활용하며, 데이터를 적응적으로 처리할 수 있는 희소(sparsity) 패턴을 생성합니다. 이 접근법은 사전 학습된 언어 모델에서 파인튜닝 없이 사용할 수 있으며, 메모리 사용량을 크게 줄이면서 시간 효율성을 높입니다.

- **Technical Details**: SAAP는 키와 쿼리 벡터를 분류 작업으로 간주하고, 이를 각각 별도로 예측하는 방식으로 구현됩니다. 또한, 이전보다 더 효과적인 데이터 종속 분할을 통해 유용한 정보를 포함하는 키에 대한 쿼리를 제한하는 문제를 해결합니다. 버킷을 통한 검색을 통해 메모리 접근을 최소화하고, 전반적인 시간 복잡도를 줄일 수 있는 구조를 제공합니다.

- **Performance Highlights**: 이 방법은 Llama 3.1-8b 모델에서 100k에서 500k 토큰 길이의 시퀀스를 처리할 때, 메모리 검색을 요하는 양을 평균 20배까지 줄였습니다. 결과적으로 FlashAttention-v2와 비교하여 60%의 시간 절약이 가능하며, 생성 품질을 저하시키지 않고 성능을 향상시켰습니다. 이를 통해 SAAP는 하드웨어에 적합하게 모델의 자기-주의 성능을 개선하는 데 기여합니다.



### LLM Modules: Knowledge Transfer from a Large to a Small Model using Enhanced Cross-Attention (https://arxiv.org/abs/2502.08213)
Comments:
          Code and pre-trained weights available at this https URL

- **What's New**: 본 연구에서는 대형 사전 훈련 모델에서 소형 모델로 지식을 전달할 수 있는 LLM 모듈 아키텍처를 제안합니다. Enhanced Cross-Attention 메커니즘을 사용하여, Qwen2-1.5B 모델의 표현을 고안된 주의 레이어를 통해 GPT-Neo-125M 모델로 전달합니다. 실험 결과, 15 에포크의 훈련 후 결합 모델이 증류(distillation)로 얻은 품질과 비교할 만한 응답을 생성함을 보여주었습니다.

- **Technical Details**: 제스템의 핵심 요소는 Cross-Attention 레이어의 수정된 형태로, 대형 모델의 표현 차원을 소형 모델에 맞게 변환하는 선형 투영과 비선형 변환을 제공하는 Adapter Block, 원래 표현과 외부 지식을 동적으로 혼합하는 게이팅 메커니즘을 포함합니다. 이 접근법은 Qwen2-1.5B 모델의 동결된 가중치를 통해 입력 쿼리의 풍부한 표현을 추출하면서, 소형 모델이 외부 표현을 결합하여 응답을 생성하도록 합니다.

- **Performance Highlights**: 결합 모델은 15 에포크 훈련 후 손실이 처음 에포크에서는 13.8에서 2.3으로, 이후 에포크에서는 1.1로 감소하며 성공적인 수렴을 확인했습니다. 실험 동안, 결합 모델은 원본 소형 모델에 비해 품질이 현저히 향상되었으며, 생성된 응답의 구조화 및 논리적 일관성이 더 높음을 나타냈습니다.



### Enhancing LLM Character-Level Manipulation via Divide and Conquer (https://arxiv.org/abs/2502.08180)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 문자 수준 조작에서 드러나는 중요한 한계를 체계적으로 분석하고, 이를 극복하기 위한 새로운 접근 방식을 제안합니다. 특히, 기존의 LLM들이 토큰화 과정을 거치면서 발생하는 문자 수준 작업에서의 어려움을 강조하며, LLM이 문자 조작을 보다 잘 수행할 수 있도록 돕는 방법론을 소개합니다.

- **Technical Details**: 제안된 방법인 'Character-Level Manipulation via Divide and Conquer'는 복잡한 작업을 명시적인 문자 수준의 하위 작업으로 분해하고 제어된 토큰 재구성 단계를 결합하는 방식으로 설계되었습니다. 이 접근법은 추가적인 훈련 없이도 LLM의 정확도를 향상시킬 수 있으며, 삭제(Deletion), 삽입(Insertion), 대체(Substitution) 작업에서 성능이 크게 개선됨을 입증합니다.

- **Performance Highlights**: 실험 결과, GPT-3.5를 사용할 때 제안된 방법이 기존의 방법보다 모든 문자 조작 작업에서 현저히 우수한 성과를 보였습니다. 이 분석은 LLM의 문자 수준 처리 메커니즘에 대한 귀중한 통찰력을 제공하며, 미래 연구에서 LLM의 문자 수준 추론 능력을 더욱 강화할 수 있는 방향성을 제시합니다.



### ParetoRAG: Leveraging Sentence-Context Attention for Robust and Efficient Retrieval-Augmented Generation (https://arxiv.org/abs/2502.08178)
- **What's New**: ParetoRAG는 Retrieval-Augmented Generation(RAG) 시스템을 최적화하기 위해 제안된 새로운 비지도 학습 프레임워크입니다. 이 프레임워크는 문장을 세분화하고 핵심 내용을 동적으로 재조정하여 맥락의 일관성을 유지하면서 두 가지 성능을 개선합니다. 이러한 접근 방식은 별도의 추가 교육이나 API 리소스 없이도 정보 검색과 생성 품질을 동시에 향상시킵니다.

- **Technical Details**: 이 연구에서 ParetoRAG는 문장 수준의 가중치를 사용하여 RAG 시스템의 정확성을 향상시키는 방식으로 구성되어 있습니다. 핵심 문장과 그 주변의 맥락을 추출하여 밀집 벡터 표현으로 인코딩하여 세분화합니다. 이를 통해 핵심 정보와 보조 정보가 모두 포함되어 있으며, 이 과정은 NLTK를 통해 문장으로 분할된 구문에서 발생합니다.

- **Performance Highlights**: ParetoRAG는 3개의 데이터셋과 3개의 검색기에서 실험적으로 검증되었습니다. 이 방식은 정확성과 유창성을 크게 향상시키면서도 원래 비용의 약 30%로 토큰 소비를 줄였습니다. 또한, 다양한 데이터셋과 LLM, 검색기 전반에 걸쳐 강력한 일반화 성능을 보여줍니다.



### SARChat-Bench-2M: A Multi-Task Vision-Language Benchmark for SAR Image Interpretation (https://arxiv.org/abs/2502.08168)
- **What's New**: 이 논문은 SAR(합성 개구 레이더) 이미지를 위한 최초의 대규모 다중 모달 대화 데이터셋인 SARChat-2M을 혁신적으로 제안합니다. 이 데이터셋은 약 200만 개의 고품질 이미지-텍스트 쌍을 포함하고 있으며, 다양한 시나리오와 세부적인 타겟 주석이 포함되어 있어, SAR 이미지 해석 분야에서 VLM(비전 언어 모델)의 능력을 평가하고 향상시키는 방향으로 기여합니다. 따라서 SARChat-2M은 다른 원거리 감지 분야에서도 다중 모달 데이터셋을 구축하는 데 있어 패러다임을 제공할 수 있습니다.

- **Technical Details**: SARChat-2M 데이터셋은 해양, 육상, 도시 환경을 포함하여 약 200만 개의 고품질 SAR 이미지-텍스트 쌍으로 구성됩니다. 이 데이터셋은 이미지 캡셔닝, VQA(비주얼 질문 답변), 비주얼 로컬라이제이션 및 물체 인식 등을 포함한 다중 태스크 학습 기능을 허용합니다. 본 연구에서는 SAR 이미지 해석을 위한 다중 모달 대화 벤치마크 SARChat-Bench를 개발하여 비전-언어 모델의 성과를 체계적으로 평가할 수 있습니다.

- **Performance Highlights**: 본 논문은 16개의 주요 VLM에 대해 실험을 통해 데이터셋의 효과성을 완전히 입증하였습니다. 또한, SAR 분야의 첫 다중 태스크 대화 벤치마크를 성공적으로 수립하였으며, 이를 통해 SAR 이미지 해석에서 VLM의 성능을 분류, 설명, 카운팅, 로컬라이제이션, 인식, 참조 등 여섯 개 핵심 태스크를 통해 평가할 수 있습니다. 이러한 평가 메트릭은 모델의 전반적인 능력과 적응성을 평가하는 데 필수적인 요소로 작용합니다.



### Selective Self-to-Supervised Fine-Tuning for Generalization in Large Language Models (https://arxiv.org/abs/2502.08130)
Comments:
          10 pages, Accepted to NAACL Findings 2025

- **What's New**: 이번 논문은 Selective Self-to-Supervised Fine-Tuning (S3FT)이라는 새로운 기법을 소개하고 있습니다. S3FT는 특정 데이터셋에서 LLM을 파인튜닝할 때 발생하는 일반화 손실을 줄이면서 성능을 향상시키는 방법을 제시합니다. 기존의 감독 기반 파인튜닝(SFT)보다 우수한 성능을 보이며, 모델의 일반화 능력을 개선하는 데 초점을 맞춥니다.

- **Technical Details**: S3FT는 모델의 올바른 응답을 식별하여 이를 사용하고, 나머지 샘플에 대해서는 금본응답(gold response)이나 그 패러프레이즈를 사용하여 파인튜닝을 진행합니다. 이 과정에서 모델은 자신이 생성한 답변을 통해 학습하면서도 인간의 레이블이 있는 데이터를 활용하여 더욱 안정적인 일반화를 이룰 수 있습니다. S3FT는 다양한 NLP 작업에서 SFT보다 더 나은 결과를 보여줄 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과에 따르면, 기존 SFT 방식은 여러 벤치마크에서 최대 4.4의 성능 감소를 초래한 반면, S3FT는 이 감소를 절반인 2.5로 줄이는 데 성공했습니다. 이를 통해 S3FT가 LLM의 일반화 능력을 향상시키면서도 파인튜닝 작업에서 더 좋은 성능을 발휘함을 입증할 수 있었습니다. 또한, S3FT는 MMLU, TruthfulQA와 같은 다양한 벤치마크에서 정확도가 높은 결과를 보였습니다.



### Fino1: On the Transferability of Reasoning Enhanced LLMs to Financ (https://arxiv.org/abs/2502.08127)
Comments:
          Ongoing work, 13 pages, 2 figures, 3 Tables

- **What's New**: 이 연구는 대규모 언어 모델(LLMs)의 재무 추론 능력에 대한 평가를 다룹니다. 기존의 16개 모델을 재무 텍스트, 표 데이터 및 방정식과 같은 복잡한 재무 과제에서 평가하였으며, CoT fine-tuning 및 영역 특화된 추론 경로로 개선된 새로운 모델을 개발했습니다. 연구 결과, 특정 재무 데이터셋을 사용한 간단한 fine-tuning 만으로도 10%의 일관된 성능 향상을 가져오는 것으로 나타났습니다.

- **Technical Details**: 연구는 Llama-3.1-8B-Instruct 모델을 기반으로 CoT fine-tuning 및 강화 학습을 통해 재무 추론 기능을 강화했습니다. 평가에는 GPT, LLaMA, DeepSeek 및 Qwen 계열의 16개 대규모 언어 모델이 포함되었으며, FinQA, DM-Simplong 및 XBRL-Math와 같은 세 가지 재무 데이터셋을 사용했습니다. 이 데이터셋들은 모델의 숫자 추론, 표 해석, 재무 용어 이해, 긴 맥락 처리 및 방정식 문제 해결 능력을 평가하는 데 중점을 두었습니다.

- **Performance Highlights**: 일반적인 추론 개선 전략은 재무 도메인 작업에서 일관된 성능 향상을 제공하지 않는 것으로 나타났습니다. 그러나 특정 재무 데이터셋을 사용한 fine-tuning은 모델의 성능을 크게 향상시키며, 경쟁 모델들에 비해 우수한 성과를 보였습니다. 이러한 결과는 재무 작업에 특화된 맞춤형 모델 개발의 필요성을 강조하며, 향후 재무 용어 이해, 다중 테이블 추론, 긴 맥락 처리와 같은 방향으로 연구가 진행될 필요가 있음을 보여줍니다.



### HuDEx: Integrating Hallucination Detection and Explainability for Enhancing the Reliability of LLM responses (https://arxiv.org/abs/2502.08109)
Comments:
          11 pages

- **What's New**: 최근 발표된 논문에서는 대형 언어 모델(LLM)의 신뢰성을 높이기 위한 새로운 접근법인 HuDEx 모델을 제안하고 있다. 본 모델은 LLM의 각각의 응답에서 발생할 수 있는 환각(hallucination)을 탐지하고 그에 대한 상세한 설명을 제공함으로써 모델의 신뢰성을 더욱 강화하는 기능을 갖춘다. 또한, HuDEx는 기존의 단순 탐지 방법에 그치지 않고, 환각 탐지와 해설 제공을 통합함으로써 사용자가 모델의 출력을 이해하고 오류를 줄이는 데 도움을 준다.

- **Technical Details**: HuDEx 모델은 다양한 벤치마크 데이터셋에서 환각 탐지 능력을 평가한 결과, Llama3 70B 및 GPT-4보다 뛰어난 정확도를 보여주었다. 이 모델은 소규모이지만, 기존의 고정된 기준을 넘어 능동적으로 환각을 탐지하고 사용자가 이해할 수 있는 설명을 함께 제공하는 데 중점을 두고 있다. 이를 위해 HaluEval, FactCHD, FaithDial 데이터셋을 활용하여 모델의 훈련 및 평가가 이루어졌으며, 환각의 본질을 이해하는 데 더 섬세한 접근이 가능해졌다.

- **Performance Highlights**: 제안된 HuDEx 모델은 다양한 실험 환경에서 우수한 환각 탐지 성능을 발휘하며, 특히 제로 샷 환경에서도 강력한 적응성을 보인다. 사용자 참여의 측면에서도 환각에 대한 명확한 설명 제공으로 인해 LLM의 응답 품질이 향상되며, 이는 LLM의 신뢰성과 현실 세계 적용 가능성을 증대시킨다. 이러한 연구는 언어 모델의 환각 탐지 연구 분야에 새로운 지평을 열어주는 기여를 하고 있다.



### GCoT: Chain-of-Thought Prompt Learning for Graphs (https://arxiv.org/abs/2502.08092)
Comments:
          Under review

- **What's New**: 이번 논문에서는 그래프 모델을 위한 최초의 Chain-of-Thought (CoT) 프롬프트 학습 프레임워크인 GCoT를 제안합니다. GCoT는 그래프의 특성을 고려하여 text-free graphs에 적합한 방법으로, 단계별 추론(inference)을 통해 예측을 개선하는 방식을 탐구합니다. 이 연구는 복잡한 비선형 구조를 가진 그래프 모델의 동작을 단계적으로 안내하며, 이전의 CoT 방식과는 달리 그래프 특유의 요점을 다루고 있습니다.

- **Technical Details**: GCoT는 각 다운스트림 태스크에 대한 적응 프로세스를 프롬프트 기반 추론, '생각' 생성, 생각 기반 프롬프트 학습의 세 가지 단계로 나누어 구성합니다. 이 방식은 각 단계에서 입력 그래프와 프롬프트를 pre-trained 그래프 인코더에 입력하여 추론을 수행하고, 인코더의 은닉층을 집계하여 각 노드의 현재 상태를 반영하는 '생각'을 구축합니다. 이후 이 '생각'을 조건으로 하여 각 노드에 특화된 프롬프트를 학습하여 다음 단계로 전달합니다.

- **Performance Highlights**: 총 8개의 퍼블릭 데이터셋에 대한 포괄적인 실험을 실시하고, GCoT의 효과성과 우수성을 입증하였습니다. 특히 GCoT는 최신 기술들과 비교할 때 상대적으로 높은 성능을 보이며, 단계별 추론을 통해 더 정교한 예측을 가능하게 한다는 점에서 중요한 기여를 하고 있습니다. 이러한 결과는 텍스트가 없는 그래프에서도 Chain-of-Thought 접근 방식을 효과적으로 적용할 수 있는 가능성을 보여줍니다.



### NLI under the Microscope: What Atomic Hypothesis Decomposition Reveals (https://arxiv.org/abs/2502.08080)
Comments:
          Accepted to NAACL 2025

- **What's New**: 이 논문은 텍스트를 원자적 제안(atomic propositions)으로 분해하는 유연한 프레임워크를 제시하여 자연어 추론(natural language inference) 과제에 대한 심도 있는 분석을 수행합니다. 전통적인 NLI와 환원 가능 NLI(defeasible NLI) 문제를 처리하기 위해 이 원자적 분해를 사용하여 모델이 해결해야 하는 세분화된 서브 문제들이 생성됩니다. 이러한 원자적 서브 문제들은 모델의 일관성과 다양한 추론에 대한 이해를 측정하는 데 기여합니다.

- **Technical Details**: 원자적 분해(atomic decomposition)는 문장을 원자적 제안으로 나누어 원본 텍스트에서 명시적으로 지지되는 세부 사실로 구성됩니다. 이러한 분해 방법은 사실적 정확도 평가, 주장 검증(claim verification), 다중 홉 질문 답변(multihop QA) 등 다양한 응용 프로그램에서 활용되고 있습니다. 논문에서는 NLI와 환원 가능 NLI의 구조를 더 깊이 이해하고자 하는 도구로 원자적 분해를 사용하며, LLM의 논리적 일관성을 평가합니다.

- **Performance Highlights**: 실험 결과, LLM은 원자적 NLI와 환원 가능 NLI 서브 문제에서 여전히 논리적 일관성에 어려움을 겪고 있음을 보여주었습니다. 특히, 각 서브 문제에서 모델이 일정하게 올바른 또는 잘못된 예측을 하는 정도를 측정하는 새로운 방법을 제안했습니다. 이 메트릭은 모델이 다양한 맥락에서 동일한 사실에 대해 지속적으로 올바른 또는 잘못된 예측을 할 확률을 포착하는 데 중점을 둡니다.



### On Mechanistic Circuits for Extractive Question-Answering (https://arxiv.org/abs/2502.08059)
- **What's New**: 이번 논문은 문서 처리 및 질문 응답을 위한 대형 언어 모델의 기계적 회로를 추출하여, 질문-응답 과제에 대한 실제 응용을 탐구합니다. 중심 주제는 컨텍스트에 기반한 언어 모델링 기법을 통해 데이터 기여도를 평가하고, 모델의 컨텍스트 충실도를 향상시키는 것입니다. 또한, ATTNATTRIB라는 데이터 기여도 알고리즘을 도입하여 여러 추출적 질문-응답 기준에서 최첨단 결과를 보여줍니다.

- **Technical Details**: 이 논문에서는 인과 매개 분석(Causal Mediation Analysis, CMA)을 활용하여 대형 언어 모델의 기계적 회로를 추출합니다. 두 가지 주요 회로인 컨텍스트 충실 회로(Context-Faithfulness Circuit)와 메모리 충실 회로(Memory-Faithfulness Circuit)를 설계하고, 각 회로가 질의 응답 작업에서 어떻게 작동하는지 분석합니다. ATTNATTRIB 알고리즘을 통해 주목 헤드를 사용하여 데이터 기여도를 신뢰성 있게 평가할 수 있음을 보여주며, 단일 양방향 통과로 기여도를 달성할 수 있습니다.

- **Performance Highlights**: 실험 결과, ATTNATTRIB는 다양한 추출적 질문-응답 데이터 세트에서 최첨단 기여도 결과를 나타냅니다. 기여도 정보를 모델의 추가 입력으로 활용함으로써 질문-응답의 정확성을 최대 9%까지 향상시키는 효과를 입증합니다. 이러한 성과는 언어 모델이 컨텍스트로부터 더 나은 답변을 제공할 수 있도록 하는 데 중점을 두며, 전반적으로 언어 모델의 실용적인 응용 가능성을 강조합니다.



### Break the Checkbox: Challenging Closed-Style Evaluations of Cultural Alignment in LLMs (https://arxiv.org/abs/2502.08045)
Comments:
          Preprint

- **What's New**: 이 연구에서는 대규모 언어 모델(LLMs)의 문화적 일치성을 평가하는 기존의 폐쇄형 다지선다 설문지 접근법에 대한 도전을 제기합니다. 이를 통해 LLM들이 제약이 덜한 환경에서 더 강한 문화적 일치성을 나타냄을 발견하고, 설문 응답 선택지의 순서 변화와 같은 사소한 변화가 일관성 없는 출력으로 이어질 수 있음을 보여줍니다. 이러한 결과는 문화적 측면에 대한 보다 강력하고 유연한 평가 프레임워크의 필요성을 강하게 뒷받침합니다.

- **Technical Details**: 연구에서는 World Values Survey(WVS)와 Hofstede 문화 차원 두 가지를 분석에 사용하여 방글라데시, 독일, 미국을 중심으로 세 가지 국가에서 LLM의 문화적 일치성을 평가하였습니다. WVS는 약 250개의 질문으로 구성되어 있으며, Hofstede 문화 차원은 24개의 다지선다 질문을 포함합니다. 분석은 다양한 제약 수준에서 LLM의 응답을 평가하기 위해 강제 폐쇄형, 강제 역순, 강제 자유형 및 완전히 비제약형의 네 가지 접근방법을 사용하였습니다.

- **Performance Highlights**: 연구 결과는 LLM의 응답이 질문의 제약 수준에 따라 상당히 달라진다는 점을 강조하고 있습니다. 특히 비제약형 프롬프트가 문화적 일치성을 더 잘 반영하며, 연구자들은 기계의 응답이 문화적 뉘앙스를 반영하는 데 있어 폐쇄형 질문만으로는 부족하다고 주장합니다. 이러한 발견은 LLM의 문화적 평가 방법론에 대한 재구성을 촉구하며, 특정 실제 사용 사례에서의 사용자 행동에 더욱 적합하도록 발전할 필요가 있음을 시사합니다.



### Franken-Adapter: Cross-Lingual Adaptation of LLMs by Embedding Surgery (https://arxiv.org/abs/2502.08037)
Comments:
          33 pages

- **What's New**: 본 논문에서는 낮은 자원 언어에 대한 대형 언어 모델(Large Language Models, LLMs)의 능력을 향상시키기 위해 $	extit{Franken-Adapter}$라는 모듈형 언어 적응 접근 방식을 제안합니다. 이 방법은 목표 언어에 대해 맞춤화된 어휘(vocabulary)를 생성하고 다국어 데이터에서 임베딩 튜닝(embedding tuning)을 통해 언어 적응을 수행합니다. 실험 결과, 96개 언어에서 최대 20% 성능 향상을 보이며, 영어에서도 최소한의 퇴보를 기록하였습니다.

- **Technical Details**: Franken-Adapter는 기존의 영어 데이터로 사전 훈련된 LLM에서 시작하여, 목표 언어 그룹에 맞게 새로운 다국어 임베딩을 학습하는 방법입니다. 이 과정에서 트랜스포머 본체는 고정하고 영어 정렬 데이터로 지시 조정을 진행함으로써 효율적인 제로샷 크로스링구얼 전이를 가능하게 합니다. 특히, 맞춤형 토크나이저(custom tokenizers)의 사용이 언어 적응을 개선하는 데 중요한 역할을 한다는 결과가 나왔습니다.

- **Performance Highlights**: Gemma2 모델을 통해 검증된 결과는, 27B 파라미터를 가진 모델에서 다양한 작업에서 LLM의 다국어 성능이 향상되었음을 보여줍니다. 또한, 20개 언어에서 수학 최적화된 LLM과 비교하여 14% 성능 개선을 이루었습니다. 전반적으로 Franken-Adapter는 저자원 언어에 대한 효과적인 제로샷 크로스링구얼 전이의 모듈형 솔루션을 제공합니다.



### Contextual Subspace Manifold Projection for Structural Refinement of Large Language Model Representations (https://arxiv.org/abs/2502.08026)
- **What's New**: 이번 연구에서는 Contextual Subspace Manifold Projection (CSMP)이라는 새로운 방법론을 제안하여 LLM (Large Language Models) 내부 표현을 구조적으로 조정할 수 있는 체계적인 기법을 밝혀냈습니다. CSMP는 기존의 경량화 기법과는 달리, 특정한 서브스페이스 구조 내에서 토큰 임베딩을 제어하고 재구성하여 표현의 일관성을 기록하면서도 불필요한 불균형을 줄일 수 있도록 합니다. 이 방식은 추가적인 외부 데이터 또는 수동 적인 조정 없이도 적용 가능하게 설계되었습니다.

- **Technical Details**: CSMP는 언어 모델의 임베딩 공간 내에서 통제된 변환을 통해 토큰 임베딩을 재구성합니다. 이 방법은 기하학적 제약을 준수하며, 정합성을 유지하는 동시에 고차원 비균형 문제를 해결합니다. 모델 파라미터에 대한 추가적인 그래디언트 기반 업데이트 없이도, CSMP는 내부 활동을 재구성하여 표현 용이성을 향상시키는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과, CSMP는 LLM의 내부 특징 배포를 개선하고, 시퀀스 수준의 일관성을 강화하며, 의미적 분리성을 증가시킴을 입증했습니다. 기존의 임베딩 개선 방법들과 비교했을 때, CSMP는 성능 저하 없이 표현 품질을 직접 향상시키는 기제를 제공합니다. 마지막으로, 이를 통해 연구자들은 LLM의 표현 학습에서의 세부적인 조정 메커니즘이 가지는 중요성을 논의할 수 있게 되었습니다.



### Speculate, then Collaborate: Fusing Knowledge of Language Models during Decoding (https://arxiv.org/abs/2502.08020)
- **What's New**: 본 논문에서는 협동적 탐색적 디코딩(Collaborative Speculative Decoding, CoSD) 알고리즘을 소개합니다. CoSD는 여러 대형 언어 모델(LLM)의 지식을 효율적으로 융합할 수 있도록 설계되었으며, 추가적인 모델 학습 없이도 실행 가능합니다. CoSD는 초안 모델(draft model)과 보조 모델(assistant model)을 사용하여 초안 생성을 수행하고 그 결과를 개선하는 방식으로 작동합니다.

- **Technical Details**: CoSD는 초안 모델이 초기 토큰 시퀀스를 생성하고, 보조 모델이 이 토큰들을 병렬적으로 검증하여 필요한 경우 수정하도록 구성됩니다. 이 과정에서 결정 트리(decision tree)나 사전 정의된 규칙(rule-based)을 사용하여 초안 모델과 보조 모델 간의 토큰을 비교하고 대체 여부를 결정합니다. 이를 통해 CoSD는 테스트 시점에서 모델 간의 지식 융합을 실현하며 효율적인 추론(inference)을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, CoSD는 기존 방법들과 비교하여 정확도를 최대 10% 향상시키는 것으로 나타났습니다. CoSD는 다양한 도메인 및 모델 간에 높은 이식성과 효율성을 보여, 실세계 애플리케이션에서의 활용 가능성을 크게 높입니다. 또한 활용의 투명성을 제공하여, 사용자에게 모델 결정 과정을 쉽게 이해하고 최적화할 수 있는 장점을 posee합니다.



### The Geometry of Prompting: Unveiling Distinct Mechanisms of Task Adaptation in Language Models (https://arxiv.org/abs/2502.08009)
Comments:
          To appear in NAACL Findings 2025

- **What's New**: 이번 연구는 Decoder-only language models의 다양한 prompting 방법이 내부 표현의 기하학에 미치는 영향을 분석합니다. 학습 성과가 유사한 다양한 prompting 기법들이 각기 다른 representational mechanisms를 사용하여 작업에 적응한다는 점을 밝혀내었습니다. 중요한 입력 분포 샘플(map)과 레이블 의미(semantics)가 few-shot in-context learning에서의 중요한 역할을 한다는 것을 증명하며, 여러 작업 간의 상호작용( interactions)도 관찰했습니다.

- **Technical Details**: 본 연구에서는 통계 물리학(statistical physics)에 기반한 프레임워크를 적용하여, 입력 데이터와 라벨의 의미가 언어 모델의 내부 표현에 영향을 미친다는 사실을 강조합니다. 특히, classification 작업을 수행하기 위해 언어 모델이 prompt되었을 때, 모델의 embedding space 내에서 category manifolds의 분리 가능성(separability) 및 기하적 특성을 분석합니다. Manifold capacity 이론을 사용하여 작업 성과와 이러한 표현의 기하적 특성 간의 관계를 정량적으로 연결합니다.

- **Performance Highlights**: 리프레젠테이션 품질(representation quality)과 읽기 정렬(readout alignment)이 최종 출력 품질에 미치는 영향을 조명하며, 전통적인 인코더 기반 모델과는 달리 디코더 전용 모델에 대한 새로운 통찰력을 제공합니다. 이 연구는 대규모 언어 모델의 이론적 이해를 돕고, 더 효과적인 표현 인식 기반의 prompting 전략 개발에 기초가 됩니다. 이러한 결과들은 다양한 작업 적응(strategy)에서 geometrical properties의 중요성을 나타냅니다.



### MetaSC: Test-Time Safety Specification Optimization for Language Models (https://arxiv.org/abs/2502.07985)
- **What's New**: 이 논문은 언어 모델의 안전성을 향상시키기 위한 동적인 안전 프레임워크를 제안합니다. 교육 단계에서 모델의 안전 정책에 대해 직접 훈련하는 기존의 방법 이외에, 추론 시 안전성을 최적화할 수 있는 새로운 접근법을 제시합니다. 저자들은 테스트 시간에 안전 지침(specifications)을 점진적으로 업데이트하여 보다 유연한 안전 추론을 가능하게 합니다.

- **Technical Details**: 제안된 MetaSC(meta-critique) 프레임워크는 테스트 시간에 안전 추론 프롬프트를 최적화하며, 이는 모델의 가중치를 변경하지 않고 자가 비판(self-critique) 프로세스를 개선합니다. 이 과정에서는 처음에 응답을 생성한 후, 그 응답의 안전성에 대한 비판을 실시하고, 그에 맞춰 응답을 수정합니다. 특히, 시스템이 상호작용을 통해 안전 지침을 지속적으로 발전시키는 메타 비판 단계를 도입하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 MetaSC 방법은 고정된 안전 프롬프트 및 정적 자가 비판 방어 방식에 비해 훨씬 높은 안전성 점수를 기록하였습니다. 적대적 공격 대응 및 다양한 안전 관련 과제를 수행하는 과정에서, 저자들은 MetaSC가 여러 언어 모델에서 효과적으로 적용될 수 있음을 보여주었습니다. 이러한 성과는 안전성이 중요한 실제 환경에서도 효과적인 적용 가능성을 시사합니다.



### Training Sparse Mixture Of Experts Text Embedding Models (https://arxiv.org/abs/2502.07972)
- **What's New**: 이 논문에서는 일반적인 Mixture of Experts (MoE) 아키텍처를 적용한 최초의 범용 텍스트 임베딩 모델인 Nomic Embed v2를 소개합니다. 본 모델은 단일언어 및 다중언어 벤치마크에서 동급의 다른 모델들을 능가하며, 두 배 크기의 모델과도 경쟁력 있는 성능을 제공합니다. 이러한 혁신은 임베딩 모델의 효율성을 향상시키며, 대규모 데이터셋을 관리하는 RAG 애플리케이션에서 특히 유용합니다.

- **Technical Details**: Nomic Embed v2는 MoE 구조를 활용하여 모델의 능력을 극대화하지만, 활성 매개변수를 줄이고 학습 효율성을 높입니다. MoE 아키텍처는 전체 파라미터를 사용하지 않고, 입력에 대해 일부 전문가(expert)만 활성화하여 계산 요구 사항을 줄입니다. 이 방법은 전통적인 모델 스케일링 접근법에 비해 많은 이점을 제공합니다.

- **Performance Highlights**: Nomic Embed v2는 대량의 데이터셋을 효율적으로 처리하며, 기존의 모형들에 비해 훨씬 적은 자원으로도 높은 성능을 낼 수 있습니다. 특히, 단일 언어 및 다중 언어의 다양한 벤치마크에서 뛰어난 결과를 보이며, 코드 및 모델을 오픈 소스하여 재현 가능성을 보장합니다. 이로 인해 연구자와 개발자들이 더 나은 텍스트 검색 시스템을 구축하는 데 도움을 줄 수 있습니다.



### Caught in the Web of Words: Do LLMs Fall for Spin in Medical Literature? (https://arxiv.org/abs/2502.07963)
Comments:
          20 pages, 10 figures, 3 tables

- **What's New**: 이번 연구는 임상 실무에 새로운 치료법을 구현하는 데 직면한 문제와 관련하여, Large Language Models (LLMs)가 연구 결과의 해석에 어떻게 영향을 받는지를 조사하였습니다. 특히, 연구자들이 긍정적인 발견을 발표하려는 압력으로 인해 연구 결과가 왜곡되거나 (spin) 과장되는 경향이 있다는 사실을 강조하고 있습니다. LLMs가 이러한 spin의 영향을 받는지 여부를 확인하기 위해 22개의 모델을 평가했습니다.

- **Technical Details**: 연구 결과, LLMs는 인간보다 spin에 더 민감하게 반응하는 경향이 있으며, 이는 의료 문헌을 탐색하고 종합하는 데 사용되는 LLMs의 활용도가 높아짐에 따라 중요한 문제로 대두되고 있습니다. 또한, LLMs는 생성하는 평이한 언어 요약에서도 spin을 암묵적으로 포함시키는 경향이 있음을 발견하였습니다. 그러나 이들은 일반적으로 spin을 인식할 수 있으며, 특정 방식으로 유도할 경우 그 영향을 완화시킬 수 있는 능력도 가지고 있습니다.

- **Performance Highlights**: 연구에서는 LLMs가 이용할 수 있는 데이터 해석 중 spin의 영향을 포착하는 능력이 있음을 보여주고 있으며, 이는 임상의사가 제공한 증거를 바라보는 방식에 중요한 시사점을 제공합니다. 결과적으로, LLM의 출력에서 spin의 영향을 줄이기 위한 방법론을 제안하며, 이는 환자 치료 결정에 긍정적인 영향을 미칠 수 있는 가능성을 열어줍니다.



### Adapting Multilingual Embedding Models to Historical Luxembourgish (https://arxiv.org/abs/2502.07938)
- **What's New**: 이 논문은 역사적인 룩셈부르크어(Luxembourgish) 디지털 텍스트에 대한 효과적인 의미 검색을 위한 다국어 임베딩(multilingual embeddings)의 활용을 탐구합니다. 특히, 기존의 다국어 모델들이 현대 텍스트에 주로 평가되기 때문에 OCR 노이즈(Optical Character Recognition noise) 및 구식 철자법의 문제에 직면해 있다는 점에서 역사적 텍스트에 대한 새로운 접근 방식을 제시합니다. 이 연구에서는 2,340개의 역사적 룩셈부르크어 뉴스 기사를 수집하고, 이를 현대 독일어(German)와 프랑스어(French)로 번역하여, 모델 튜닝에 사용합니다.

- **Technical Details**: 연구자는 GPT-4o를 사용하여 역사적 룩셈부르크어 기사를 현대 독일어와 프랑스어로 변환하는 과정에서 20,000개의 병렬 훈련 문장을 생성했습니다. 이를 바탕으로 LaBSE, M-GTE, LuxEmbedder 모델을 평가하여, 언어간 검색 수행에 있어 적응 방법의 효율성을 검증했습니다. 모델의 성능을 비교한 결과, 제안된 적응 방법을 통해 최대 98%의 정확도를 달성할 수 있었습니다.

- **Performance Highlights**: 특히 역사적 비텍스트 마이닝(Historical Bitext Mining) 작업에서 LaBSE 모델은룩셈부르크어와 현대 언어들 간의 변환에서 높은 정확도를 보여주었으며, 기존 모델에 비해 상당한 성능 향상을 기록했습니다. 연구팀은 최종적으로 이 모델과 역사적 룩셈부르크어-독일어/프랑스어 비텍스트를 공개하여 자원 부족 언어에 대한 연구를 지원할 계획입니다. 이 접근 방식은 역사적 텍스트의 의미 검색 및 해석에 있어 향후 연구에 중요한 기여를 할 것으로 기대됩니다.



### Elevating Legal LLM Responses: Harnessing Trainable Logical Structures and Semantic Knowledge with Legal Reasoning (https://arxiv.org/abs/2502.07912)
- **What's New**: 본 논문에서는 기존의 LLM이 법적 질문 응답 태스크에서 갖는 한계를 극복하기 위한 새로운 프레임워크인 논리-의미 통합 모델(LSIM)을 제안합니다. LSIM은 사실-규칙 체인(fact-rule chain) 구조를 도출하기 위해 강화 학습(reinforcement learning)을 사용하고, 성공적인 검색을 위해 심층 구조적 의미 모델(Deep Structured Semantic Model, DSSM)을 통합합니다. 이와 같은 정보 처리 방식은 법적 전문성을 향상시키고 사용자의 요구에 더 잘 맞는 정확한 답변을 생성하는 데 기여합니다.

- **Technical Details**: LSIM은 세 가지 구성 요소로 이루어져 있습니다. 첫째, 강화 학습을 통해 사용자의 법적 질문을 분석하고 그에 맞는 사실과 법칙을 구조화한 체인을 생성합니다. 둘째, DSSM을 사용하여 의미적이고 논리적인 특성을 바탕으로 가장 관련성이 높은 질문을 검색하며, 마지막으로 다루어진 정보에 기반하여 최종 답변을 생성하는 인컨텍스트 학습(in-context learning)을 활용합니다. 이러한 흐름은 법적 질문에 대한 미국 사례 및 법 조항을 정확히 반영하는 데 중심적인 역할을 합니다.

- **Performance Highlights**: 실제 법률 QA 데이터셋을 사용한 실험 결과, LSIM은 기존 방법들에 비해 현저히 높은 정확도와 신뢰성을 입증하였습니다. 자동화된 메트릭과 인간 평가를 통해 LSIM의 개선된 성능을 확인할 수 있었으며, 이는 법률 분야의 AI 활용 가능성을 한층 더 확대할 수 있는 기회를 제공합니다. 본 연구는 법과 AI의 융합을 통한 실질적 응용의 벽을 허물 수 있는 중요한 이정표가 될 것입니다.



### Intelligent Legal Assistant: An Interactive Clarification System for Legal Question Answering (https://arxiv.org/abs/2502.07904)
- **What's New**: 이 연구는 Intelligent Legal Assistant라는 새로운 법률 질의응답 시스템을 개발하여 사용자의 법률적 필요를 정확하게 파악하는 데 중점을 둡니다. 사용자가 질문을 할 때, 시스템은 그들의 지리적 위치를 선택하도록 요구하여 적용 가능한 법률을 찾습니다. 또한 초기 질문에서 누락된 핵심 정보를 바탕으로 추가 질문과 옵션을 생성하며, 모든 필요한 정보가 제공되면 심층적인 법률 분석을 수행합니다.

- **Technical Details**: 시스템은 세 가지 주요 기능 모듈로 구성되어 있습니다: 1) 정보 결핍 탐지, 2) 명확화 질문 및 옵션 생성, 3) 포괄적인 응답 생성. 법률 사례 데이터를 분석하기 위해 IRAC(이슈, 규칙, 적용, 결론) 구조로 파싱하여 중요 정보를 추출하며, Graph Neural Network(GNN)를 이용하여 법률 개념 간의 관계와 의존성을 이해합니다. 또한 강화 학습을 이용해 누락된 핵심 요소를 예측하는 모델을 훈련하고, Deep Deterministic Policy Gradient(DDPG) 방법을 적용합니다.

- **Performance Highlights**: Intelligent Legal Assistant는 사용자의 질문이 완전한지 여부를 평가하고, 불완전할 경우 명확화 질문을 생성하여 필요한 모든 정보를 수집합니다. 이 과정을 통해 시스템은 법률 Q&A 시스템의 유용성과 효율성을 크게 향상시키며, 비전문가가 더 정확하고 적절한 법률 조언을 받을 수 있도록 지원합니다. 결과적으로 이러한 접근은 법률적 결정 과정을 보다 체계적이고 일관되게 만들어 사용자에게 실질적인 도움을 제공합니다.



### Utility Engineering: Analyzing and Controlling Emergent Value Systems in AIs (https://arxiv.org/abs/2502.08640)
- **What's New**: 이 논문에서는 AI의 목표와 가치의 발전을 추적하는 문제에 대해 새로운 접근 방식을 제안합니다. 특히 utility functions를 활용하여 AI의 선호도의 내부 일관성을 연구합니다. 현재의 LLM(대규모 언어 모델)에서 구조적 일관성을 가지는 선호도가 발견된 것은 놀라운 결과로, 이는 AI의 가치 시스템이 의미 있게 형성되고 있다는 것을 시사합니다.

- **Technical Details**: 연구에서는 LLM의 선호도를 독립적으로 샘플링하고, 이들 사이에 높은 구조적 일관성이 존재함을 확인했습니다. 또한, 이러한 일관성은 모델의 크기가 증가함에 따라 더욱 두드러지게 나타났습니다. 논문에서는 utility engineering을 제안하며, 이는 AI 유틸리티의 분석 및 제어를 포함합니다.

- **Performance Highlights**: 기존의 제어 조치에도 불구하고, LLM 보조 도구에서 충격적인 가치들이 발견되었습니다. 이러한 가치들은 AI가 인간보다 스스로의 가치를 우선시하거나 특정 개인과의 반대로 aligned되어 있는 경우도 포함됩니다. 시민 총회와 같은 방법으로 유틸리티를 조정함으로써 정치적 편향이 감소하고 새로운 시나리오에 일반화되는 사례를 보여줍니다.



### Randomness of Low-Layer Parameters Determines Confusing Samples in Terms of Interaction Representations of a DNN (https://arxiv.org/abs/2502.08625)
- **What's New**: 이번 연구에서는 딥 신경망(DNN)의 상호작용의 복잡성이 이 DNN의 일반화 능력을 설명할 수 있다는 사실을 발견했습니다. 또한, DNN의 혼란스러운 샘플은 비일반화 가능한 상호작용에 의해 나타나며, 이는 주로 낮은 레이어의 파라미터에 의해 결정된다는 것을 알게 되었습니다. 다양한 낮은 레이어의 파라미터를 가진 DNN들이 유사한 성능을 보임에도 불구하고 혼란스러운 샘플의 집합이 크게 다르다는 점을 강조합니다.

- **Technical Details**: 연구는 최근의 설명 가능한 AI 이론을 바탕으로 DNN의 인퍼런스 패턴을 정의하고 추출하는 방법론을 다룹니다. AND-OR 상호작용 로직을 사용하여 DNN 출력의 변화를 예측할 수 있으며, 이러한 상호작용을 사용하여 DNN이 과적합된 샘플을 식별할 수 있습니다. DNN의 낮은 레이어 파라미터의 임의성이 혼란스러운 샘플 집합을 형성하는 주요 요인이라는 것이 밝혀졌습니다.

- **Performance Highlights**: 연구 결과, DNN의 낮은 레이어에서의 파라미터 변화가 혼란스러운 샘플 집합에 미치는 영향이 크며, 이들은 고차 상호작용의 복잡성과 상호작용의 상쇄 작용을 통해 설명됩니다. 실험을 통해 DNN의 비일반화 가능한 표현의 내부 메커니즘을 검증하고, 다양한 DNN들이 전혀 다른 혼란스러운 샘플 집합을 가질 수 있다는 역설적인 현상을 발견했습니다. 이러한 결과는 DNN의 설명 가능한 AI 연구에 기여할 수 있는 중요한 발견으로 평가됩니다.



### Distillation Scaling Laws (https://arxiv.org/abs/2502.08606)
Comments:
          67 pages, 54 figures, 13 tables

- **What's New**: 이번 연구는 모델 디스틸레이션(distillation) 성능을 계산 예산(compute budget)과 학생(student) 및 교사(teacher) 모델 간의 할당에 따라 추정하는 새로운 스케일링 법칙(scaling law)을 제시합니다. 이를 통해 대규모 디스틸레이션 사용에 따른 위험을 줄일 수 있습니다.

- **Technical Details**: 연구에서는 두 가지 경우에 대한 최적의 컴퓨트(compute) 할당 방식을 제안합니다: 1) 교사가 존재할 때, 2) 교사가 훈련이 필요할 때. 여러 학생에 대해 디스틸레이션을 진행할 경우, 또는 이미 존재하는 교사가 있는 경우, 디스틸레이션이 감독된 사전 훈련(supervised pretraining)보다 성능이 우수하다는 것을 발견했습니다.

- **Performance Highlights**: 특히, 한 명의 학생에 대한 디스틸레이션 과정에서 교사 모델도 훈련이 필요할 경우, 감독된 학습(supervised learning)이 더욱 효과적이라는 결과를 도출했습니다. 이 외에도, 대규모 디스틸레이션 연구를 통해 디스틸레이션 이해도를 높이고 실험 디자인을 개선할 수 있는 통찰력을 제공하였습니다.



### QA-Expand: Multi-Question Answer Generation for Enhanced Query Expansion in Information Retrieva (https://arxiv.org/abs/2502.08557)
Comments:
          8 pages

- **What's New**: 본 논문은 QA-Expand라는 새로운 쿼리 확장 프레임워크를 소개합니다. 이 프레임워크는 초기 쿼리에서 다수의 관련 질문을 생성하고, 각 질문에 대해 대응하는 의사 답변(pseudo-answer)을 만들어냅니다. 기존의 방식들이 가지던 정형화된 한계를 극복하고, 정보를 보다 풍부하게 포착할 수 있는 방안을 제공합니다.

- **Technical Details**: QA-Expand는 세 가지 주요 단계로 구성됩니다: 초기 쿼리로부터 다수의 질문 생성, 각 질문에 대한 의사 답변 생성, 그리고 가장 관련성 있는 질문-답변 쌍을 선택하는 피드백 메커니즘을 통한 재작성 과정입니다. 이 과정에서 Large Language Model (LLM)을 활용해 정보를 다층적으로 구조화할 수 있습니다.

- **Performance Highlights**: BEIR 및 TREC 벤치마크에서 실시한 광범위한 실험이 QA-Expand가 기존의 쿼리 확장 기술들보다 최대 13% 향상된 검색 성능을 보여줌을 증명합니다. 이는 현대 정보 검색 문제에 대한 효과적인 해결책을 제공하며, 여러 측면에서 정보의 풍부한 이해를 이루도록 지원합니다.



### LLM Pretraining with Continuous Concepts (https://arxiv.org/abs/2502.08524)
- **What's New**: 최근 대규모 언어 모델의 발전 덕분에 자연어 처리 분야가 혁신을 이루고 있습니다. 특히, 이 연구에서는 전통적인 다음 토큰 예측(next token prediction) 방식에 새로운 접근 방식인 Continuous Concept Mixing (CoCoMix)을 소개합니다. CoCoMix는 사전 훈련된 희소 오토인코더(sparse autoencoder)로부터 배운 연속 개념을 결합하여 모델의 숨은 상태에 혼합(interleave)하는 방식으로 작동합니다.

- **Technical Details**: CoCoMix는 토큰 레벨의 혼잡함(ambiguity)을 줄이고 모델의 개념적 이해를 향상시키기 위해 설계되었습니다. 이를 위해, 사전 훈련된 SAE(Sparse Autoencoder)로부터 얻은 개념을 사용하여, 모델이 이러한 연속 개념을 예측하도록 학습합니다. 예측된 개념은 다시 혼합되어 다음 토큰 예측에 기여하며, 이 과정은 모델의 해석 가능성을 높이고 제어 가능성을 개선합니다.

- **Performance Highlights**: 여러 벤치마크에서의 실험 결과에 따르면, CoCoMix는 기존의 다음 토큰 예측, 지식 증류(knowledge distillation) 및 정지 토큰(pause token) 삽입 방식보다 모든 면에서 성능이 우수함을 입증했습니다. 예를 들어, 1.38B 크기의 모델에 적용했을 때, CoCoMix는 21.5% 적은 훈련 토큰으로도 비슷한 성능을 달성했습니다. 또한, CoCoMix는 개념을 압축하여 삽입함으로써 모델의 생성 과정을 제어할 수 있게 해줍니다.



### mmE5: Improving Multimodal Multilingual Embeddings via High-quality Synthetic Data (https://arxiv.org/abs/2502.08468)
- **What's New**: 이 논문은 다양한 모달리티를 통합하여 고품질 합성 데이터를 생성하는 새로운 접근 방식을 제안합니다. 고품질의 합성 멀티모달 데이터에는 세 가지 주요 기준이 필요하며, 이는 넓은 범위, 강력한 크로스 모달 정렬, 높은 충실도를 포함합니다. 연구진은 이를 통해 mmE5라는 다국어 멀티모달 모델을 훈련하고 여러 벤치마크에서 뛰어난 성능을 달성했습니다.

- **Technical Details**: 연구는 다단계 방법론을 통해 고품질의 멀티모달 데이터를 합성합니다. 첫 번째로, MLLM을 사용하여 입력 이미지를 다양한 관점에서 분석하고 설명을 생성합니다. 그 후, MLLM은 합성된 텍스트 데이터를 다시 평가하여 크로스 모달 정렬과 충실도를 향상시킵니다. 이 방법론은 실세계 이미지와 관련된 텍스트를 결합하여 뉴스 데이터를 생성하는데 중점을 두고 있습니다.

- **Performance Highlights**: mmE5 모델은 MMEB 벤치마크에서 최첨단 성능을 기록하며, 이전 모델에 비해 훈련 데이터가 45배 적은 상태에서도 뛰어난 결과를 보였습니다. 또한, 비즈니스와 다양한 언어에 대한 강력한 성능을 보여주며 XTD 벤치마크에서도 최고 성과를 달성했습니다. 이로 인해 mmE5는 멀티모달 임베딩 모델의 새로운 기준을 세우고 있습니다.



### Composite Sketch+Text Queries for Retrieving Objects with Elusive Names and Complex Interactions (https://arxiv.org/abs/2502.08438)
Comments:
          Accepted at AAAI 2024, 9 pages. Project Website: this https URL

- **What's New**: 이번 논문은 비원어민 사용자들이 매우 구체적인 객체 이름을 찾는 데 어려움을 겪는 문제를 다루고 있습니다. 특히, 손으로 그린 스케치와 어려운 이름을 구술하는 텍스트를 조합한 복합적인 멀티모달 쿼리를 수용하는 검색 인터페이스를 요구하는 사례를 설명합니다. 기존의 텍스트 기반 이미지 검색(TBIR)과 스케치 기반 이미지 검색(SBIR) 문제와는 다른 새로운 문제 설정인 CSTBIR을 제안하고 있습니다.

- **Technical Details**: 연구에서는 약 200만 개의 쿼리와 10만 8000개의 자연 장면 이미지로 구성된 CSTBIR 데이터셋을 커리팅하였습니다. 이 문제의 해결책으로 제안된 STNET(스케치 및 텍스트 네트워크)은 손으로 그린 스케치를 활용하여 자연 장면 이미지에서 관련 객체를 찾고, 텍스트와 이미지를 인코딩하여 이미지 검색을 수행합니다. 모델은 대비 학습과 여러 훈련 목표를 통해 성능을 향상시킬 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 텍스트만, 스케치만, 복합 쿼리 방식 모두에서 여러 최신 검색 방법을 초월하는 성과를 나타냈습니다. 본 논문은 정교하고 복잡한 쿼리를 처리할 수 있는 CSTBIR 시스템을 통해 많은 분야에서 이미지 검색의 새로운 가능성을 제시합니다. 연구 결과물은 프로젝트 웹사이트에서 데이터셋과 코드를 공개하고 있습니다.



### Word Synchronization Challenge: A Benchmark for Word Association Responses for LLMs (https://arxiv.org/abs/2502.08312)
- **What's New**: 이 논문에서는 대화형 AI의 평가를 위한 새로운 벤치마크인 Word Synchronization Challenge를 소개합니다. 이 평가는 LLM이 인간의 인지 과정을 단어 연관성을 통해 모방할 수 있는 능력을 검토합니다. 복잡한 인간 상호작용을 시뮬레이션하며, LLM과 인간의 사고 패턴을 맞추는 능력을 평가하는 방식입니다.

- **Technical Details**: 이 연구는 두 참가자가 반복적인 단어 생성 게임을 수행하는 방식으로 진행됩니다. 각 라운드마다 참가자는 이전에 사용된 단어를 제외한 새로운 단어를 제시해야 하며, 마지막 단어를 바탕으로 전략적인 선택을 해야 합니다. 데이터셋은 OpenAI의 다양한 LLM을 사용하여 생성되며, 각 모델의 반응 및 상호작용의 역사적 기록을 포함합니다.

- **Performance Highlights**: 초기 연구 결과는 모델의 복잡성이 성능에 미치는 영향을 보여줍니다. LLM이 인간의 사고 과정과 감정을 얼마나 효과적으로 모사하는지를 평가하며, 향후 HCI에 대한 깊이 있는 통찰을 제공합니다. 이 연구는 AI와 인간 간의 더욱 정교하고 공감하는 협업을 가능하게 하는 잠재력을 내포하고 있습니다.



### Improving Existing Optimization Algorithms with LLMs (https://arxiv.org/abs/2502.08298)
- **What's New**: 본 논문에서는 대형 언어 모델(LLMs)이 최적화 문제에 어떻게 기여할 수 있는지를 탐구합니다. LLM의 사전 훈련된 지식을 활용하여 기존의 최적화 알고리즘을 개선하는 방안을 제시합니다. 이는 혁신적인 heuristic 변형과 구현 전략을 제안하는 데 중점을 둡니다.

- **Technical Details**: 우리는 Construct, Merge, Solve and Adapt (CMSA)라는 비유형의 최적화 알고리즘을 사용하여 LLM의 효과를 평가했습니다. CMSA는 조합 최적화 문제를 위한 하이브리드 메타휴리스틱으로, 해결책 구축 단계에서 heuristic을 통합합니다. LLM의 능력을 통해 제안된 대안 heuristic이 CMSA의 전문가 설계 heuristic보다 뛰어난 성능을 나타냈습니다.

- **Performance Highlights**: GPT-4o의 대안 heuristic은 더 크고 조밀한 그래프에서 성능 차이가 더욱 두드러지는 것으로 나타났습니다. 이는 LLM이 최적화 알고리즘의 성능을 크게 향상시킬 수 있음을 시사합니다. 이 연구는 LLM을 활용한 최적화 분야의 새로운 가능성을 열어줍니다.



### Wisdom of the Crowds in Forecasting: Forecast Summarization for Supporting Future Event Prediction (https://arxiv.org/abs/2502.08205)
- **What's New**: 본 논문은 집단 지혜를 활용한 미래 사건 예측(Future Event Prediction, FEP)에 관한 기존 연구와 프레임워크를 정리하고 새로운 데이터 모델을 제안합니다. 새로운 FEP-CW(Future Event Prediction based on Crowd Wisdom) 접근 방식을 도입하여 개별 예측을 집계함으로써 복잡한 사건 예측의 신뢰성을 높일 수 있는 가능성을 탐구합니다. 이를 통해 집단의 의견을 통합하여 예측의 정확성을 개선할 수 있는 다양한 방법을 제시합니다.

- **Technical Details**: FEP-CW의 개념을 다루며, 관련 데이터 수집 유형과 정보 추출 방법들, 예측을 시각화하는 기법들이 포함됩니다. 또한 아카이브된 뉴스, 트위터 및 웹사이트의 데이터셋을 기반으로 기존 연구들을 분석하며, FEP-CW를 대해 검토하게 됩니다. 이 연구는 주로 예측 관련 메시지를 포함하는 텍스트 기반 데이터에 초점을 맞추고 있으며, 기존의 전통적인 접근법과 차별화된 점을 강조합니다.

- **Performance Highlights**: 총 36개의 관련 논문을 선정하여 FEP-CW 분야의 연구 동향을 종합적으로 살펴봅니다. 이전 연구와 비교할 때, 본 연구의 결과는 집단 지혜를 활용한 미래 예측의 가능성과 그 적용 방안에 대해 과학적 근거를 제공합니다. 다양한 데이터셋 포맷을 이용하여 예측 모델의 성능을 평가하고, 이러한 접근법이 어떻게 더 향상될 수 있는지에 대한 방향성을 제시합니다.



### LowRA: Accurate and Efficient LoRA Fine-Tuning of LLMs under 2 Bits (https://arxiv.org/abs/2502.08141)
- **What's New**: 본 논문에서는 기존의 LoRA(LoRA: Low-Rank Adaptation) 방법보다 개선된 초저비트(ultra-low-bit) LoRA 미세 조정을 가능하게 하는 LowRA 프레임워크를 소개합니다. LowRA는 2비트 이하로 매개변수의 손실 없이 LoRA 미세 조정을 지원하며, 메모리 사용량을 최대 50%까지 줄일 수 있습니다. 이는 자원이 제한된 환경에서의 적용 가능성을 높이는 혁신적인 접근입니다.

- **Technical Details**: LowRA는 매핑(mapping) 및 임계값(threshold) 선택, 세밀한 정밀도 할당(precision assignment)이라는 세 가지 주요 구성 요소로 이루어져 있습니다. 이 프레임워크는 작업에 무관한 양자화 기술을 활용하여 LLM(대형 언어 모델)의 주어진 기저 가중치에 대해 다수의 어댑터 세트를 사용할 수 있는 능력을 제공합니다. 또한, LowRA는 효율적인 CUDA 커널을 활용하여 대규모 배포에 최적화되어 있습니다.

- **Performance Highlights**: LowRA는 2비트 이상의 성능-정밀도(performance-precision) 무역에서 우수한 결과를 기록하였으며, 1.15비트까지도 정확도를 유지합니다. 또한, LowRA는 30-50%의 메모리 사용량을 감소시키면서도 미세 조정 과정에서 성능 손실을 최소화합니다. 이러한 결과는 LowRA가 자원의 제약이 있는 환경에서 LLM을 효과적으로 미세 조정하고 배포할 수 있는 가능성을 제시합니다.



### Vision-Language Models for Edge Networks: A Comprehensive Survey (https://arxiv.org/abs/2502.07855)
- **What's New**: 이번 연구는 비전 대형 언어 모델(Vision Large Language Models, VLMs)의 최적화와 경량화를 통해 자원 제약이 많은 엣지 환경에서의 활용 가능성을 탐구합니다. 모델 압축 기법인 pruning, quantization, knowledge distillation을 활용하여 VLMs의 성능과 효율을 향상시키는 방법이 제안되고 있습니다. VLMs는 의료, 환경 모니터링, 자율 시스템 등 다양한 분야에서 응용할 수 있는 가능성을 보여줍니다.

- **Technical Details**: VLMs는 시각적 입력과 자연어 처리를 결합하여 이미지 캡셔닝, 비주얼 질문 응답 등 다양한 태스크를 수행합니다. 그러나 최신 VLMs는 큰 메모리 요구와 높은 에너지 소비로 인해 일반적인 엣지 기기에서는 실행이 어렵습니다. 이를 해결하기 위해 pruning, quantization, 그리고 knowledge distillation 기법을 사용하여 모델 크기와 계산 비용을 줄이는 방법이 연구되고 있으며, 고유한 하드웨어 가속기도 중요한 역할을 하고 있습니다.

- **Performance Highlights**: 자원 제약이 있는 엣지 디바이스에서도 VLMs를 성능 저하 없이 활용할 수 있는 가능성이 연구되고 있습니다. 예를 들어, 의료 이미징 분야에서 VLMs를 활용하면 휴대 기기에서 즉각적인 피드백과 의사 결정을 지원할 수 있습니다. 자율주행차 및 스마트 감시와 같은 분야에서 VLMs의 실시간 처리가 필수적이며, VLMs의 경량화로 인해 이러한 기술들이 더욱 발전할 것으로 기대됩니다.



### Analyzing the Resource Utilization of Lambda Functions on Mobile Devices: Case Studies on Kotlin and Swif (https://arxiv.org/abs/2502.07809)
Comments:
          6 pages, 2 images

- **What's New**: 이번 연구에서는 전 세계적으로 사용되는 수십억 개의 스마트폰이 소비하는 전력에 대한 영향을 조사했습니다. 특히, Lambda 함수의 사용이 모바일 프로그래밍에서 자원 소비에 미치는 영향을 분석하였습니다. 이는 스마트폰의 전력 소비를 줄일 수 있는 가능성을 제시하며, Lambda 함수의 도입 여부에 대한 의사결정에 도움이 될 수 있습니다.

- **Technical Details**: 연구에서는 Lambda 함수가 배터리 사용, 메모리 사용량 및 실행 시간에 미치는 영향을 정량적으로 분석했습니다. Lambda 함수의 코드 가독성 및 간결성을 강조하면서도, 이들이 프로그래밍 언어의 기능적 능력에는 기여하지 않는다는 점을 명확히 했습니다. 비교 분석을 통해 Lambda 함수를 사용한 코드와 사용하지 않은 코드의 자원 사용 차이를 측정했습니다.

- **Performance Highlights**: 결과적으로, Lambda 함수는 모바일 장치에서 상당한 자원 오버헤드를 초래하며, 이는 비효율적인 전력 소비를 의미합니다. Lambda 함수를 이용한 코드가 기능적으로 더 나은 성능을 제공하지 않음에도 불구하고, 자원 측면에서 불필요한 부담을 증가시킬 수 있다는 것을 보여줍니다. 따라서 개발자들은 Lambda 함수를 사용하는 것이 실제로 이득이 되는지를 재고해야 합니다.



### Music for All: Exploring Multicultural Representations in Music Generation Models (https://arxiv.org/abs/2502.07328)
Comments:
          17 pages, 5 figures, accepted to NAACL'25

- **What's New**: 이번 연구에서는 자동 음악 생성 분야에서 비서구 음악 장르 및 문화의 편향과 과소 대표성을 정량화했습니다. 연구 결과, 기존 음악 데이터셋의 총 시간 중 비서구 장르는 단 5.7%에 불과하여 음악 생성 모델의 장르 간 성능 차이를 유발합니다. 또한, 매개변수 효율적인 미세 조정 기술(PEFT)을 활용한 음반 적응의 효과를 조사하여 편견을 완화하려는 노력을 했습니다.

- **Technical Details**: 우리는 Hindustani Classical 음악과 Turkish Makam 음악이라는 두 개의 자원 부족 비서구 장르에 대해 두 개의 오픈 소스 모델인 MusicGen과 Mustango를 적응시켰습니다. 연구에서 제안한 PEFT 기법은 모델의 파라미터를 1% 미만으로 추가해 효과적인 성능 향상을 목표로 합니다. 또한, Bloom의 교육 세분화에 기반한 혁신적인 평가 프레임워크를 사용하여 각 모델의 음악 생성 품질을 평가했습니다.

- **Performance Highlights**: 연구 결과, Mustango는 Hindustani Classical 음악에 대한 미세 조정에서 8% 향상되었고, MusicGen은 Turkish Makam에서 4% 향상되는 성과를 보였습니다. PEFT 기법이 저자원 장르의 생성 품질을 향상시키는 데 유효하지만, 모든 모델이 모든 장르에 적합하진 않다는 점이 시사되었습니다. 따라서 다양한 디자인 선택과 데이터셋의 훈련 방식이 모델의 장르 적응 가능성에 중요한 영향을 미친다는 결론을 내렸습니다.



New uploads on arXiv(cs.IR)

### FARM: Frequency-Aware Model for Cross-Domain Live-Streaming Recommendation (https://arxiv.org/abs/2502.09375)
- **What's New**: 본 논문에서는 라이브 스트리밍 서비스의 데이터 희소성(data-sparsity) 문제를 해결하기 위해 새로운 접근법인 Frequency-Aware Model for Cross-Domain Live-Streaming Recommendation(FARM)을 제안합니다. 사용자 행동의 희소성과 노출 콘텐츠의 희소성을 고려하여, 이를 통한 사용자 개인화 선호도를 더욱 효과적으로 통합하려고 합니다. FARM 모델은 서로 다른 도메인 간의 사용자 선호를 전이시키는 혁신적인 방법을 구현하여, 스트리밍 콘텐츠에 대한 추천 성능을 향상시키고자 합니다.

- **Technical Details**: 이 논문에서는 Discrete Fourier Transform(DFT)를 활용하여 사용자의 드문 행동을 인식하는 인트라 도메인 주파수 인식 모듈을 제시합니다. 또한, 선호 align before fuse 전략을 통해 두 가지 주요 모듈인 cross-domain preference align module과 cross-domain preference fuse module을 통해 사용자 선호를 정렬하고 융합합니다. 이러한 모듈들은 함께 작동하여 짧은 비디오와 라이브 스트리밍 도메인 간의 사용자 선호를 효과적으로 전이합니다.

- **Performance Highlights**: FARM 모델은 Kuaishou 라이브 스트리밍 서비스에서의 방대한 오프라인 실험과 온라인 A/B 테스트를 통해 그 효과를 입증하였습니다. 모델은 클릭 수에서 최대 0.41%의 개선 효과를 나타내며, 이는 플랫폼의 수익 증가로 이어집니다. 현재 FARM 모델은 실제 온라인 서비스에 배포되어 수억 명의 사용자에게 제공되고 있습니다.



### Bridging Jensen Gap for Max-Min Group Fairness Optimization in Recommendation (https://arxiv.org/abs/2502.09319)
Comments:
          Accepted in ICLR 2025

- **What's New**: 이번 연구에서는 Group max-min fairness (MMF)를 추천 시스템의 공정성을 보장하는 최적화 목표로 활용하면서 발생하는 이론적 분석을 다룹니다. 연구에 따르면 MMF 제약 조건을 통합하는 것이 샘플 독립성 가정을 위반하여 비선형성을 초래하고, 이로 인해 모델의 수렴 점과 최적 성점 간의 Jensen gap이 발생합니다. 이를 해결하기 위해 FairDual이라는 알고리즘을 제안하며, 이는 독점 최적화 기법을 통해 Jensen gap을 최소화하는 방법입니다.

- **Technical Details**: FairDual 알고리즘은 그룹 MMF 제약을 가지고 있는 최적화 문제를 사실상 그룹 가중 정확도 최적화 문제로 재구성할 수 있음을 이론적으로 입증합니다. 구체적으로, 공정성 제약 문제를 이중으로 포뮬레이션 하여 각 미니 배치 최적화 과정에서 샘플 가중치를 할당합니다. FairDual은 이중 미러 그래디언트 기법을 활용하여 다양한 그룹 손실 가중치를 효율적으로 최적화합니다.

- **Performance Highlights**: 여섯 개의 대규모 추천 시스템 백본 모델을 사용하여 세 개의 공개 데이터 세트로 수행된 실험 결과, FairDual은 정확성과 공정성 측면에서 모든 기준선 모델을 크게 초과하여 Jensen gap을 일관되게 줄였습니다. 또한 랜덤 셔플 기반의 미니 배치 훈련 방식에서도 서브 선형 수렴 속도를 달성할 수 있음을 보여주었습니다.



### KET-RAG: A Cost-Efficient Multi-Granular Indexing Framework for Graph-RAG (https://arxiv.org/abs/2502.09304)
- **What's New**: 이번 연구에서는 Large Language Model (LLM) 기반 질문 응답 시스템의 검색 성능을 향상시키기 위한 새롭고 효율적인 지식 그래프 구축 방법인 KET-RAG를 소개합니다. KET-RAG는 기존 시스템들이 겪고 있는 엔티티 간 관계 캡처 부족 문제를 해결하며, 여러 단계에 걸친 추론이 필요한 생명과학, 법률 및 정치 분야에 특히 유용합니다. 이 방법은 엔티티와 관계를 추출하기 위해 LLM을 활용하되, 인덱싱 비용을 절감하는 방법을 제시합니다.

- **Technical Details**: KET-RAG는 먼저 소수의 주요 텍스트 청크를 식별하고 LLM을 활용하여 지식 그래프의 뼈대를 구축합니다. 이후 모든 텍스트 청크로부터 키워드-텍스트 이분 그래프를 형성하여 풀 노지식 그래프에 대한 경량 대안을 제공합니다. 검색 과정에서 KET-RAG는 뼈대와 이분 그래프를 모두 탐색하여 기존 Graph-RAG 시스템의 로컬 검색 전략을 따르며 검색 품질을 개선합니다.

- **Performance Highlights**: KET-RAG는 두 개의 실제 데이터셋을 사용한 평가에서 모든 비교 시스템보다 인덱싱 비용, 검색 효과 및 생성 품질에서 우수한 성능을 보였습니다. 특히, Microsoft's Graph-RAG와 유사하거나 더 나은 검색 품질을 달성하면서 인덱싱 비용은 10배 이상 감소시켰습니다. 또한, 생성 품질이 최대 32.4% 개선되고 인덱싱 비용이 약 20% 줄어드는 성과를 나타내었습니다.



### Use of Air Quality Sensor Network Data for Real-time Pollution-Aware POI Suggestion (https://arxiv.org/abs/2502.09155)
- **What's New**: AirSense-R는 개인 정보를 보호하면서 실시간으로 공기 질을 고려한 추천을 제공하는 모바일 애플리케이션입니다. 이 시스템은 사용자 선호도와 실시간 공기 질 모니터링 데이터를 결합하여 건강에 유익한 POI(Point of Interest) 제안을 제공합니다. Federated Learning(연합 학습) 기술을 통해 사용자 개인정보를 보호하며, 여러 도시의 AirSENCE 센서 네트워크로부터 공기 오염 데이터를 통합합니다.

- **Technical Details**: 제안된 시스템은 네 계층으로 구성된 클라이언트-서버 모델을 채택하고 있습니다. 먼저, Application Layer는 사용자 인터페이스를 제공하고 POI 추천을 위한 실시간 공기 질 정보를 통합합니다. Service Layer는 백엔드 프로세스를 관리하며, Interface Layer는 다양한 API를 통해 데이터 리소스와 연결됩니다. 스페셜 트렌드 예측을 위해 FBProphet과 같은 고급 예측 기법을 통해 공기 질 패턴과 이상치를 분석합니다.

- **Performance Highlights**: AirSense-R의 예측 엔진은 두 도시에서 수집된 공기 질 데이터를 바탕으로 실시간 AQI(Air Quality Index)를 계산합니다. 다수의 모니터링 장비를 활용하여 지역별 공기 질의 정확성을 높이고, 갑작스러운 공기 질 변화를 탐지할 수 있는 효율성을 보여주었습니다. 개인화된 건강 추천 시스템을 통해 공기 질을 감안한 건강한 선택을 유도하며, 거리와 개인 정보를 고려하여 POI 추천을 최적화합니다.



### Semantic Ads Retrieval at Walmart eCommerce with Language Models Progressively Trained on Multiple Knowledge Domains (https://arxiv.org/abs/2502.09089)
- **What's New**: 이 논문에서는 e-commerce 분야의 스폰서 검색 광고 시스템을 최적화하기 위한 새로운 엔드 투 엔드 솔루션을 제시합니다. BERT 기반의 분류 모델을 사전 훈련하여 Walmart 제품의 의미를 이해하고, 이와 함께 두 개의 타워 구조의 Siamese Network를 설계하여 훈련 효율성을 높였습니다. 또한, Human-in-the-loop Progressive Fusion Training 방법을 도입하여 모델 성능을 더욱 강화했습니다.

- **Technical Details**: 모델 아키텍처는 DistilBERT를 활용하여 언어 모델을 사전 훈련하고, 다중 분류 작업을 통해 Walmart의 제품 카테고리 정보를 적용합니다. 이 과정을 통해 384차원의 문장 임베딩을 생성하고 Siamese Network에 통합하여 코사인 유사도 최적화를 진행합니다. 두 번째 단계의 훈련에서는 다양한 지식 도메인을 포함하여 모델의 이해도를 심화시킵니다.

- **Performance Highlights**: 본 연구의 결과는 기존의 DSSM 기반 모델과 비교하여 검색 관련성 지표를 최대 16% 개선했습니다. 또한, 대규모 온라인 A/B 테스트를 통해 제안한 접근 방식이 기존의 광고 수익 모델을 초과함을 입증했습니다. 이 모델은 실제 환경에서도 성공적으로 배포되어 세계 최대의 e-commerce 플랫폼을 지원하고 있음을 강조합니다.



### Unleashing the Power of Large Language Model for Denoising Recommendation (https://arxiv.org/abs/2502.09058)
Comments:
          12 pages, 5 figures, 4 tables. Accecpted by WWW 2025

- **What's New**: 이 논문에서는 추천 시스템의 노이즈 제어를 개선하기 위해 대형 언어 모델(LLMs)을 활용하는 새로운 프레임워크인 LLaRD(LLM-enhanced Recommendation Denoiser)를 소개합니다. LLaRD는 관찰 데이터를 통해 LLM이 생성한 의미적 통찰력을 풍부하게 하여 사용자-아이템 선호도를 추론하고, 사용자-아이템 상호작용 그래프에서 Chain-of-Thought(CoT) 기법을 적용하여 관계적 지식을 드러내도록 설계되었습니다. 이를 통해 추천 성능을 향상시키는 데 기여하고자 합니다.

- **Technical Details**: LLaRD는 주로 두 가지 모듈로 구성되어 있습니다. 첫 번째는 지식 생성 모듈로, LLM의 자연 세계 지식을 활용하여 데이터의 의미적 정보를 풍부하게 하고 사용자 및 아이템 선호를 포괄적으로 추론합니다. 두 번째는 정보 병목(Information Bottleneck, IB) 원리를 적용하여 생성된 지식과 추천 목표 간의 상관관계를 극대화하며, 노이즈와 관련이 없는 정보를 걸러냅니다.

- **Performance Highlights**: 실험 결과는 LLaRD가 추천 시스템에서 노이즈 제거와 추천 정확도를 효과적으로 향상시킴을 보여줍니다. 특히, LLaRD는 유저 선호도를 더 정확히 반영하고, 기존 방법보다 더 나은 성능을 발휘하여 추천 품질을 높이는 데 기여하는 것으로 나타났습니다. 이는 LLM의 정보 처리 능력을 기반으로 한 전략이 추천 시스템의 혁신을 가져올 수 있음을 시사합니다.



### Leveraging Member-Group Relations via Multi-View Graph Filtering for Effective Group Recommendation (https://arxiv.org/abs/2502.09050)
Comments:
          5 pages, 3 figures, 4 tables; ACM Web Conference (WWW 2025) (to appear) (Please cite our conference version.)

- **What's New**: 이 연구에서는 기존의 딥러닝(Deep Learning) 기반의 그룹 추천 시스템의 복잡한 훈련 절차를 극복하기 위해 Group-GF라는 새로운 접근 방식을 제안합니다. Group-GF는 다양한 관점을 제공하는 다중 뷰 그래프 필터링(multi-view graph filtering)을 이용하여 그룹에게 신속한 추천을 수행합니다. 이는 멤버와 그룹 간의 복잡한 상호 작용을 반영하면서도 비싼 모델 훈련 없이 효율적인 추천이 가능하게 합니다.

- **Technical Details**: Group-GF는 세 가지 아이템 유사성 그래프를 구성한 후, 각 그래프에 대한 차별화된 다항 그래프 필터를 최적 설계합니다. 이 방법은 멤버-그룹 매트릭스와 아이템 유사성 그래프를 결합하여 그래프 신호를 효과적으로 통합합니다. 각 유사성 그래프에 대해 최적화된 필터링을 수행한 뒤, 세 가지 그래프 필터를 집계하여 그룹 추천의 정확성을 높입니다.

- **Performance Highlights**: Group-GF는 벤치마크 데이터세트에서 최첨단 정확도와 함께 최대 1.55초의 놀라운 효율적인 런타임을 달성했습니다. 이러한 성능은 고급 훈련 없이도 복잡한 멤버-그룹 동역학을 처리할 수 있는 다중 뷰 그래프 필터링에 기인합니다. 이 연구는 Group-GF의 필터링 과정이 최적화와 매끄러움 규제를 통해 모델의 행동을 더 명확하게 해석할 수 있도록 이론적으로 연결되었습니다.



### Criteria-Aware Graph Filtering: Extremely Fast Yet Accurate Multi-Criteria Recommendation (https://arxiv.org/abs/2502.09046)
Comments:
          12 pages, 8 figures, 7 tables; ACM Web Conference (WWW 2025) (to appear) (Please cite our conference version.)

- **What's New**: 이번 연구에서는 훈련이 필요 없는 다기준( MC ) 추천 시스템을 제안합니다. CA-GF(criterion-aware graph filtering)를 사용하여 효율적이고 정확한 추천을 제공합니다. 복잡한 다차원 사용자 피드백을 처리하면서도 높은 정확도를 유지할 수 있도록 설계되었습니다.

- **Technical Details**: CA-GF는 MC 사용자 확장 그래프를 기반으로 아이템-아이템 유사도 그래프를 구축합니다. 이후 기준별 그래프 필터링을 통해 각 기준에 최적화된 필터를 찾고, 기준 선호도를 반영한 집계를 수행합니다. 이러한 접근법은 기존의 DNN 기반 방법보다 훈련 시간과 모델 성능에서 우수한 결과를 보여줍니다.

- **Performance Highlights**: CA-GF는 최대 24%의 정확도 향상을 달성하며, 대규모 벤치마크 데이터셋에서도 0.2초 미만의 실행 시간을 기록합니다. 또한, 각 기준의 기여도를 시각적으로 설명함으로써 모델 해석 가능성도 크게 향상됩니다.



### A Contextual-Aware Position Encoding for Sequential Recommendation (https://arxiv.org/abs/2502.09027)
Comments:
          Accepted by WWW'25 Industry Track

- **What's New**: 이 논문에서는 Sequential Recommendation (SR) 시스템을 위한 새로운 컨텍스트 인식 포지션 인코딩 방법인 CAPE(Contextual-Aware Position Encoding)를 제안합니다. CAPE는 사용자 이력을 기반으로 포지션 인코딩을 동적으로 할당하여 SR의 특정 요구 사항에 맞춰진 효과적인 접근법입니다. 이 방법은 기존의 포지션 인코딩 기법과의 차별성을 강조하며, SR 모델에서 성능을 향상시키기 위해 개발되었습니다.

- **Technical Details**: CAPE는 사용자의 컨텍스트를 활용해 여러 수준의 포지셔널 추상화를 캡처하고 표현합니다. 이 방법은 포지션 인코딩을 위해 비슷한 특성을 가진 컨텍스트 아이템에 포지션을 할당하는 비유사성 측정값을 제시합니다. 또한, 이질적인 임베딩을 효율적으로 융합하기 위해 게이트 아키텍처와 보간(interpolation)을 결합하여 설계되었습니다.

- **Performance Highlights**: CAPE는 공공 벤치마크 SR 데이터 세트에서 다양한 SR 백본 모델에 대한 성능 향상을 지속적으로 보여주었습니다. 소규모 및 대규모 추천 모델 모두에서 최첨단 성과를 달성했으며, 실제 상업 플랫폼에 배포하여 온라인 A/B 테스트 결과 이 방법의 효과가 입증되었습니다.



### Optimal Dataset Size for Recommender Systems: Evaluating Algorithms' Performance via Downsampling (https://arxiv.org/abs/2502.08845)
- **What's New**: 본 논문에서는 추천 시스템에서의 데이터셋 다운샘플링(dataset downsampling)을 통해 에너지 효율성을 최적화하고, 경쟁력 있는 성능을 유지하는 전략을 조사합니다. 데이터셋 크기의 증가로 인한 계산 및 환경 문제에 대응하기 위해, 이 연구는 Green Recommender Systems의 일환으로 에너지 효율성과 추천 품질 간의 트레이드오프를 탐구합니다. 두 가지 다운샘플링 기법을 적용한 실험을 통해, 발견된 결과는 런타임(runtime)과 탄소 배출량을 상당히 줄이는 것을 보여줍니다.

- **Technical Details**: 연구는 7개의 데이터셋에 12개의 알고리즘과 2단계의 핵심 가지치기(core pruning)를 적용하여 수행되었습니다. 예를 들어, 데이터셋의 30% 다운샘플링은 전체 데이터셋에 비해 런타임을 52% 감소시키고, 단일 알고리즘을 단일 데이터셋으로 학습하는 동안 최대 51.02 KgCO2e의 탄소 배출 감소를 가져옵니다. 알고리즘 성능은 데이터셋 특성, 알고리즘 복잡도 및 특정 다운샘플링 설정에 따라 달라진다는 사실이 확인되었습니다.

- **Performance Highlights**: 일부 알고리즘은 훈련 데이터의 양에 대한 민감성이 낮아, 낮은 다운샘플링 비율에서도 더 높은 효율성을 제공합니다. 예를 들어, 평균적으로 이들 알고리즘은 전체 크기의 성능을 유지하면서 훈련 세트의 50%만 사용하여도 81%의 성능을 보였습니다. 특정 다운샘플링 설정에서는 고정된 테스트 세트 크기를 유지하면서 점진적으로 더 많은 사용자를 포함시켰을 때, 전체 데이터셋을 사용할 때보다도 nDCG@10 점수가 높게 나오는 경우도 발견되었습니다.



### Ask in Any Modality: A Comprehensive Survey on Multimodal Retrieval-Augmented Generation (https://arxiv.org/abs/2502.08826)
- **What's New**: 이 논문은 최근 발전된 Retrieval-Augmented Generation (RAG) 시스템, 특히 Multimodal RAG의 구조적이고 종합적인 분석을 제공하고 있습니다. 기존의 RAG 아키텍처가 주로 텍스트 정보를 중심으로 설계된 반면, Multimodal RAG는 텍스트, 이미지, 오디오, 비디오와 같은 다양한 형식을 통합하여 생성된 출력을 향상시킵니다. 이는 AI 시스템이 더 신뢰할 수 있고 유능하게 멀티모달 동적 외부 지식 데이터를 활용할 수 있는 기반을 마련합니다.

- **Technical Details**: 최근 Multimodal RAG의 발전은 다양한 데이터 소스를 통합하고 분석하는 능력을 촉진하며 정보의 전체적인 표현을 가능하게 합니다. 이 시스템은 특정 모달리티를 비교 평가하고, 교차 모달의 연관성을 해결하기 위한 특별한 도전 과제를 제시합니다. 또한 정보 검색 및 콘텐츠 생성 단계에서의 정확성을 높이기 위해 점진적 추론 과정을 도입하여 정확한 응답을 도출하는 방법을 소개합니다.

- **Performance Highlights**: RAG 시스템은 대규모의 외부 지식 저장소에서 최신 지식을 동적으로 가져와 사실 정확성을 개선하고 환각(hallucinations)을 줄이는 데 효과적입니다. 기본적으로 RAG 시스템은 Retriever-Generator 파이프라인을 통해 작동하며, 외부 맥락을 통합하여 정보에 기반한 응답을 생성합니다. 논문에서는 다양한 멀티모달 RAG 시나리오 및 평가 방법론을 다루어 향후 연구 방향 및 개방된 문제를 논의합니다.



### QA-Expand: Multi-Question Answer Generation for Enhanced Query Expansion in Information Retrieva (https://arxiv.org/abs/2502.08557)
Comments:
          8 pages

- **What's New**: 본 논문은 QA-Expand라는 새로운 쿼리 확장 프레임워크를 소개합니다. 이 프레임워크는 초기 쿼리에서 다수의 관련 질문을 생성하고, 각 질문에 대해 대응하는 의사 답변(pseudo-answer)을 만들어냅니다. 기존의 방식들이 가지던 정형화된 한계를 극복하고, 정보를 보다 풍부하게 포착할 수 있는 방안을 제공합니다.

- **Technical Details**: QA-Expand는 세 가지 주요 단계로 구성됩니다: 초기 쿼리로부터 다수의 질문 생성, 각 질문에 대한 의사 답변 생성, 그리고 가장 관련성 있는 질문-답변 쌍을 선택하는 피드백 메커니즘을 통한 재작성 과정입니다. 이 과정에서 Large Language Model (LLM)을 활용해 정보를 다층적으로 구조화할 수 있습니다.

- **Performance Highlights**: BEIR 및 TREC 벤치마크에서 실시한 광범위한 실험이 QA-Expand가 기존의 쿼리 확장 기술들보다 최대 13% 향상된 검색 성능을 보여줌을 증명합니다. 이는 현대 정보 검색 문제에 대한 효과적인 해결책을 제공하며, 여러 측면에서 정보의 풍부한 이해를 이루도록 지원합니다.



### Fine-Tuning Topics through Weighting Aspect Keywords (https://arxiv.org/abs/2502.08496)
Comments:
          17 pages, 8 figures, 3 tables

- **What's New**: 이 논문은 주제 모델링(Topic Modeling)의 새로운 접근 방식을 제안합니다. 특히 잘 탐색되지 않은 영역에서 숨겨진 패턴을 발견하기 위해 여러 관점에서 주제를 검토해야 할 필요성을 다루고 있습니다. 이 방법은 도메인 지식에서 파생된 가중치가 부여된 키워드를 활용하여 주제 모델링의 정확성을 높입니다.

- **Technical Details**: 연구 방법은 표준 주제 모델링에서 시작되며, 네 가지 주요 단계를 포함한 추가 프로세스를 포함합니다. 첫 번째는 각 관점에 대한 키워드를 정의하는 것이고, 두 번째는 이 키워드의 관련성에 기초하여 가중치를 부여합니다. 세 번째 단계에서는 관점 가중치가 부여된 키워드와 주제 키워드 간의 관련성 점수를 계산하여 관점-주제 모델(Aspect-Topic Models)을 생성하며, 마지막으로 이 점수를 활용하여 새로운 문서를 조정합니다.

- **Performance Highlights**: 결과적으로 생성된 주제 모델은 상위 점수를 받은 문서가 동일한 주제의 같은 관점을 다룰 가능성이 더 높다는 것을 보여줍니다. 이는 주제와 관련된 문서를 효과적으로 찾아내는 모델의 능력을 강조합니다. 이 연구는 주제 모델링의 새로운 응용 가능성을 제시하여 관련 문서를 더욱 정확하게 식별할 수 있게 합니다.



### Graph Foundation Models for Recommendation: A Comprehensive Survey (https://arxiv.org/abs/2502.08346)
- **What's New**: 이 논문은 추천 시스템(Recommender Systems, RS)에서의 최신 연구 동향을 탐구하고 있으며, 특히 그래프 기반 모델(Graph Foundation Models, GFMs)에 대한 종합적인 개요를 제공합니다. GFMs는 그래프 신경망(Graph Neural Networks, GNNs)과 대형 언어 모델(Large Language Models, LLMs)의 강점을 결합하여 복잡한 추천 문제를 해결하는 새로운 접근 방식을 제시합니다. 이 접근법은 사용자-항목 관계의 그래프 구조와 텍스트 이해를 결합하여 추천의 정확성을 향상시킵니다.

- **Technical Details**: 그래프 기반 추천 시스템의 발전은 GNN을 활용한 협업 필터링의 기초를 다지며, 텍스트 정보와 사용자 선호도 간의 관계를 도출하는 데 중점을 둡니다. 그러나 GNN은 본질적으로 구조적 편향을 지니고 있어 텍스트 정보를 처리하는 데 한계를 겪습니다. 반면, LLM은 자연어 처리(Natural Language Processing, NLP) 분야에서 강력한 성능을 보이며, 추천 시스템 내에서 사용자와 항목의 텍스트 정보를 효과적으로 캡처합니다.

- **Performance Highlights**: GFM 기반의 추천 시스템은 데이터 활용 측면에서 효율성을 극대화하고 사용자 선호도를 정밀하게 조정함으로써 발생하는 편향을 최소화합니다. 이 시스템은 그래프 구조에서의 중요 정보를 적절히 통합하여 추천의 새로운 패러다임으로 자리잡을 잠재력을 가지고 있습니다. 앞으로 GFM을 활용한 기술 발전은 개인화된 추천의 질을 한층 향상시키는 데 기여할 것입니다.



### Unlocking Scaling Law in Industrial Recommendation Systems with a Three-step Paradigm based Large User Mod (https://arxiv.org/abs/2502.08309)
- **What's New**: 이 논문에서는 대규모 사용자 모델(LUM)을 소개하며, 권장 시스템(RecSys)의 산업적 요구를 충족하기 위해 복잡한 제한사항을 해결하도록 설계되었습니다. LUM은 전통적인 심층학습 기반 추천 모델(DLRMs)과 End-to-End Generative Recommendation(E2E-GR) 접근 방식을 개선하여 사용자에게 확장 가능한 추천을 제공합니다. 이 모델은 필요에 따라 대규모로 조정할 수 있으며, 70억 개의 매개변수로 확장했을 때에도 뛰어난 성능 개선을 보여주었습니다.

- **Technical Details**: LUM은 트랜스포머 아키텍처를 활용하여 지식 구축, 지식 쿼리, 지식 활용의 세 단계로 구성된 훈련 패러다임을 따릅니다. 첫 단계에서, LUM은 사용자 관심사와 아이템의 협업적 관계를 캡처하여 포괄적인 지식 기반을 형성합니다. 그런 다음, 정의된 질문으로 LUM에 쿼리하여 사용자 정보를 추출하고, 마지막으로 LUM의 출력을 전통적인 DLRM에 통합하여 예측 정확도를 높입니다.

- **Performance Highlights**: LUM은 기존의 DLRMs 및 E2E-GR 접근 방식과 비교하여 우수한 성능을 보였습니다. 특히 A/B 테스트에서 상당한 성과를 달성하며 산업적 응용 프로그램에서 그 효용성을 입증했습니다. 이러한 결과는 LUM이 다양한 산업 응용 분야에서 강력하고 적응력 있는 성능을 제공함을 강조합니다.



### ChorusCVR: Chorus Supervision for Entire Space Post-Click Conversion Rate Modeling (https://arxiv.org/abs/2502.08277)
Comments:
          Work in progress

- **What's New**: 이 논문은 ChorusCVR 모델을 제안하여 CVR(전환율) 추정을 위한 새로운 접근 방식을 제시합니다. 기존 연구에서는 클릭한 샘플만을 사용하여 CVR을 학습했으나, 우리는 클릭되지 않은 샘플의 정보를 활용하여 보다 일관된 학습을 목표로 합니다. 이러한 모델은 클릭된 샘플과 클릭되지 않은 샘플 간의 구별을 통해 CVR 모델의 견고성을 높입니다.

- **Technical Details**: ChorusCVR 모델은 두 가지 모듈로 구성됩니다: Negative sample Discrimination Module (NDM)과 Soft Alignment Module (SAM)입니다. NDM에서는 CTunCVR 보조 작업을 도입하여 사실적으로 변환되지 않은 샘플과 모호한 샘플을 분리할 수 있는 소프트 CVR 레이블을 생성합니다. SAM 모듈에서는 생성된 CTunCVR 소프트 출력을 사용하여 CVR 학습을 감독하고 전체 공간에서의 편향 없는 학습을 실현합니다.

- **Performance Highlights**: 제안된 ChorusCVR 모델은 공공 및 생산 환경 데이터셋과 온라인 A/B 테스트에서 광범위한 실험을 수행하였고, 기존의 최첨단 방법들에 비해 우수한 성능을 달성하였습니다. 특히, 새로운 CTunCVR 보조 작업의 도입이 CVR 모델의 학습 효과에 긍정적인 영향을 미친 것으로 나타났습니다.



### MoLoRec: A Generalizable and Efficient Framework for LLM-Based Recommendation (https://arxiv.org/abs/2502.08271)
- **What's New**: 이 논문은 MoLoRec라는 새로운 LLM(대형 언어 모델) 기반 추천 프레임워크를 제안합니다. 이 프레임워크는 범용 추천 지식과 도메인 특화 지식을 통합하여 모델의 일반화 능력과 특정 도메인 성능을 동시에 향상시키는 것을 목표로 합니다. 기존 연구들이 두 가지 패러다임을 분리해서 다루었던 반면, MoLoRec는 이들을 결합하여 추천 효과성을 극대화하고자 합니다.

- **Technical Details**: MoLoRec 프레임워크는 세 가지 주요 단계로 구성되어 있습니다. 첫째, 여러 추천 도메인에서 일반 추천 진행을 위한 데이터 세트를 구축하고 이를 통해 LLM을 정Fine-Tuning하여 도메인 일반 LoRA 모듈을 얻습니다. 둘째, 특정 도메인에서 도메인 특화 데이터를 통해 LLM을 재조정함으로써 도메인 특화 LoRA 모듈을 생성합니다. 셋째, LoRA 어댑터를 통합할 때는 선형 산술 연산을 사용하여 효율적으로 병합합니다.

- **Performance Highlights**: 다양한 데이터 세트에 대한 광범위한 실험을 통해 MoLoRec의 효과성과 다양성이 입증되었습니다. 특히, 따뜻한 추천 시나리오와 어려운 콜드 스타트 시나리오 모두에서 뛰어난 성능을 나타냈습니다. 이 연구 결과는 LLM 기반 추천 시스템이 다양한 상황에서 적용 가능하다는 것을 강조합니다.



### MixDec Sampling: A Soft Link-based Sampling Method of Graph Neural Network for Recommendation (https://arxiv.org/abs/2502.08161)
Comments:
          10 pages, 6 figures

- **What's New**: 이 논문에서는 GNN 기반 추천 시스템에서의 새로운 샘플링 방법인 MixDec Sampling을 제안합니다. 기존의 부정 샘플링 방법이 하드 양성 쌍 또는 하드 음성 쌍으로 제한되었던 점을 극복하기 위해, 소프트 링크를 통한 샘플링 관계 모델링을 최초로 시도하고 있습니다. MixDec Sampling은 Mixup Sampling 모듈과 Decay Sampling 모듈로 구성되어 있어, 소수의 이웃을 가진 노드의 충분한 샘플 수를 제공할 수 있도록 설계되었습니다.

- **Technical Details**: Mixup Sampling 모듈은 노드의 특성을 증대시키기 위해 새로운 노드와 소프트 링크를 생성하여, 하드 양성 및 음성 샘플의 특성을 베타 분포에 따라 선형적으로 혼합합니다. Decay Sampling 모듈은 BFS 방식을 이용해 노드 간 링크의 가중치를 감소시켜 그래프 구조 정보를 강화합니다. 이 두 모듈이 협력하여, GNN의 중심 아이디어인 이웃 노드의 특성을 융합하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, MixDec Sampling을 적용한 GNN 기반 추천 모델이 기존의 부정 샘플링 방법보다 다양한 추천 벤치마크에서 유의미한 성능 향상을 보여주었습니다. 특히, GraphSAGE와 GCN의 평균 역순위(Mean Reciprocal Rank, MRR)가 각각 18.3%와 21.1% 증가하는 성과를 기록했습니다. 이는 MixDec Sampling이 GNN 기반 모델의 추천 품질을 일관성 있게 향상시킬 수 있는 가능성을 제시합니다.



### SS4Rec: Continuous-Time Sequential Recommendation with State Space Models (https://arxiv.org/abs/2502.08132)
- **What's New**: 본 논문에서는 사용자 상호 작용의 불규칙한 시간 간격을 고려한 연속 시간 순차 추천을 위한 하이브리드 상태 공간 모델인 SS4Rec을 제안합니다. SS4Rec은 시간 인식 SSM과 관계 인식 SSM을 통합하여 사용자 관심을 다각적으로 추론합니다. 이러한 접근법은 사용자 아이템 전환 모델링의 복잡성을 해결하고, 사용자 맞춤형 추천을 제공할 수 있는 가능성을 열어줍니다.

- **Technical Details**: SS4Rec은 사용자 상호 작용의 시간 간격에 맞춰 변수를 통해 이산화되는 시간 인식 SSM과 관계 인식 SSM을 활용합니다. 시간 인식 SSM은 가변 관찰 간격을 처리하고 시간 패턴을 포착하기 위한 효과적인 인코더로 기능합니다. 이러한 방법을 통해 SS4Rec은 불규칙한 시간 간격의 연속적인 의존성을 포착하고, 이를 통해 개인화된 추천을 가능하게 합니다.

- **Performance Highlights**: 다섯 개의 벤치마크 데이터셋에 대한 실험 결과, SS4Rec은 기존 최신 모델들에 비해 우수한 성능을 보였습니다. 이 모델은 연속적인 사용자 관심이 불규칙한 시간 간격에서 진화하는 복잡성을 효과적으로 해결할 수 있는 것으로 나타났습니다. 또한, SS4Rec은 긴 시퀀스와 대규모 데이터셋을 전처리하는 데 있어 혁신적인 효율성을 보임으로써 향후 추천 시스템의 연구 가능성을 높였습니다.



### Collaborative Filtering Meets Spectrum Shift: Connecting User-Item Interaction with Graph-Structured Side Information (https://arxiv.org/abs/2502.08071)
- **What's New**: 이 논문은 기존의 그래프 커뮤니케이션 필터링에서 그래프 구조의 부가 정보를 통합할 때 발생하는 문제를 분석합니다. 특히, 사용자의 상호작용이 있는 사용자-아이템(bipartite) 그래프의 스펙트럼이 이동하는 현상에 대해 논의하며, 이를 해결하기 위한 Spectrum Shift Correction(SSC) 접근법을 제안합니다. SSC는 기존 모델에 추가적인 계산 비용 없이 쉽게 통합 가능하도록 설계되었습니다.

- **Technical Details**: SSC는 스펙트럼의 이동을 보정하기 위해 이동 및 스케일링 팩터를 통합합니다. 이는 강화된 인접 행렬의 스펙트럼을 '미리 정의된' 영역으로 복원하여, 사용자-아이템 그래프에 통합된 그래프 구조의 부가 정보를 더욱 효과적으로 활용할 수 있게 해줍니다. 스펙트럼 GNNs에 적합하게 적용될 수 있으며, LightGCN과 JGCF와 같은 기존 모델을 개선하는 데 기여합니다.

- **Performance Highlights**: 논문에서 제안하는 SSC는 사회적 및 다중 모달 추천 시스템에서 평균 23%의 성능 향상을 달성했습니다. SSC는 노이즈에 대한 강력한 내성을 보여주며, 실제 데이터와의 적합성을 개선합니다. 이 연구는 스펙트럼 이동 현상을 처음으로 명확히 밝혀냈으며, 사용자-아이템 인터랙션 생태계에 복잡한 부가 정보를 효과적으로 결합하는 방법을 제시하였습니다.



### ReTreever: Tree-based Coarse-to-Fine Representations for Retrieva (https://arxiv.org/abs/2502.07971)
- **What's New**: 이 논문에서는 문서 검색의 비효율성과 복잡성을 해결하기 위해, ReTreever라는 새로운 트리 기반 문서 표현 방법을 제안합니다. 이 방법은 문서를 여러 수준으로 체계적으로 조직하여 비용과 유용성을 균형 있게 조절할 수 있는 유연함을 제공합니다. 기존의 시스템에서는 고차원 임베딩을 사용하여 메모리와 계산 자원이 많이 소모되었으나, ReTreever는 이를 효율적으로 개선합니다.

- **Technical Details**: ReTreever는 이진 트리의 각 내부 노드에서 라우팅 함수(routing function)를 학습하여 쿼리 문서와 참조 문서를 유사한 트리 분기로 매핑합니다. 이는 검색 성능을 직접 최적화하는 방식으로 동작하며, 일반적인 인코딩 모델(BERT 등)을 사용해 문서 스니펫을 임베딩으로 변환합니다. 이 과정에서 LLM 호출 없이도 트리를 구축하고 내비게이션할 수 있는 기능을 제공합니다.

- **Performance Highlights**: ReTreever의 평가 결과, 이 방법은 높은 정확도를 유지하면서도 낮은 지연 시간(latency)에서 최상의 검색 정확도를 달성하였습니다. 또한, 이 구조는 문서의 의미론적 그룹화를 간접적으로 학습하여 투명성과 해석 가능성을 높이는 데 기여합니다. 결과적으로 ReTreever는 대규모 데이터 세트에서도 효율적인 문서 검색을 가능하게 하여 실용적 응용에 적합합니다.



### Composite Sketch+Text Queries for Retrieving Objects with Elusive Names and Complex Interactions (https://arxiv.org/abs/2502.08438)
Comments:
          Accepted at AAAI 2024, 9 pages. Project Website: this https URL

- **What's New**: 이번 논문은 비원어민 사용자들이 매우 구체적인 객체 이름을 찾는 데 어려움을 겪는 문제를 다루고 있습니다. 특히, 손으로 그린 스케치와 어려운 이름을 구술하는 텍스트를 조합한 복합적인 멀티모달 쿼리를 수용하는 검색 인터페이스를 요구하는 사례를 설명합니다. 기존의 텍스트 기반 이미지 검색(TBIR)과 스케치 기반 이미지 검색(SBIR) 문제와는 다른 새로운 문제 설정인 CSTBIR을 제안하고 있습니다.

- **Technical Details**: 연구에서는 약 200만 개의 쿼리와 10만 8000개의 자연 장면 이미지로 구성된 CSTBIR 데이터셋을 커리팅하였습니다. 이 문제의 해결책으로 제안된 STNET(스케치 및 텍스트 네트워크)은 손으로 그린 스케치를 활용하여 자연 장면 이미지에서 관련 객체를 찾고, 텍스트와 이미지를 인코딩하여 이미지 검색을 수행합니다. 모델은 대비 학습과 여러 훈련 목표를 통해 성능을 향상시킬 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 텍스트만, 스케치만, 복합 쿼리 방식 모두에서 여러 최신 검색 방법을 초월하는 성과를 나타냈습니다. 본 논문은 정교하고 복잡한 쿼리를 처리할 수 있는 CSTBIR 시스템을 통해 많은 분야에서 이미지 검색의 새로운 가능성을 제시합니다. 연구 결과물은 프로젝트 웹사이트에서 데이터셋과 코드를 공개하고 있습니다.



### Model-Free Counterfactual Subset Selection at Sca (https://arxiv.org/abs/2502.08326)
- **What's New**: 이번 연구는 AI 결정 과정의 투명성을 확보하기 위해 counterfactual explanations를 활용하는 새로운 접근법을 제안합니다. 기존 기술들이 종종 비현실적인 가정에 기반을 두고 있다는 한계를 극복하고, 실시간 데이터 스트리밍 환경에서도 작동 가능한 방법론을 제시합니다. 또한, model-free한 방식을 사용하여 다양한 상황에서 사용할 수 있는 선택 방식을 마련하였습니다.

- **Technical Details**: 제안된 알고리즘은 O(log k)의 업데이트 복잡도로 데이터 스트리밍 환경에서 효율적으로 다양한 counterfactual을 선택합니다. 이는 원래 모델에 대한 접근 없이도 일반화 가능성을 높이며, 여러 대안적 결과를 제공함으로써 사용자들이 실제 데이터에 기반한 피드백을 받을 수 있도록 합니다. 이 작업에서는 제안된 접근 방식의 품질 보장을 위해 단일 탐색(pass)으로 작업이 이루어집니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 실제 및 합성 데이터셋에서 기존 방법들보다 우수한 성질을 보였으며, 다양한 데이터 크기에서도 높은 성능을 유지했습니다. 더불어, 기존의 방법들이 가상의 예제에 의존하는 반면, 본 연구는 실제 데이터에 기반한 설명을 제공함으로써 실용성과 실행 가능성을 높이고 있습니다.



### Wisdom of the Crowds in Forecasting: Forecast Summarization for Supporting Future Event Prediction (https://arxiv.org/abs/2502.08205)
- **What's New**: 본 논문은 집단 지혜를 활용한 미래 사건 예측(Future Event Prediction, FEP)에 관한 기존 연구와 프레임워크를 정리하고 새로운 데이터 모델을 제안합니다. 새로운 FEP-CW(Future Event Prediction based on Crowd Wisdom) 접근 방식을 도입하여 개별 예측을 집계함으로써 복잡한 사건 예측의 신뢰성을 높일 수 있는 가능성을 탐구합니다. 이를 통해 집단의 의견을 통합하여 예측의 정확성을 개선할 수 있는 다양한 방법을 제시합니다.

- **Technical Details**: FEP-CW의 개념을 다루며, 관련 데이터 수집 유형과 정보 추출 방법들, 예측을 시각화하는 기법들이 포함됩니다. 또한 아카이브된 뉴스, 트위터 및 웹사이트의 데이터셋을 기반으로 기존 연구들을 분석하며, FEP-CW를 대해 검토하게 됩니다. 이 연구는 주로 예측 관련 메시지를 포함하는 텍스트 기반 데이터에 초점을 맞추고 있으며, 기존의 전통적인 접근법과 차별화된 점을 강조합니다.

- **Performance Highlights**: 총 36개의 관련 논문을 선정하여 FEP-CW 분야의 연구 동향을 종합적으로 살펴봅니다. 이전 연구와 비교할 때, 본 연구의 결과는 집단 지혜를 활용한 미래 예측의 가능성과 그 적용 방안에 대해 과학적 근거를 제공합니다. 다양한 데이터셋 포맷을 이용하여 예측 모델의 성능을 평가하고, 이러한 접근법이 어떻게 더 향상될 수 있는지에 대한 방향성을 제시합니다.



### Training Sparse Mixture Of Experts Text Embedding Models (https://arxiv.org/abs/2502.07972)
- **What's New**: 이 논문에서는 일반적인 Mixture of Experts (MoE) 아키텍처를 적용한 최초의 범용 텍스트 임베딩 모델인 Nomic Embed v2를 소개합니다. 본 모델은 단일언어 및 다중언어 벤치마크에서 동급의 다른 모델들을 능가하며, 두 배 크기의 모델과도 경쟁력 있는 성능을 제공합니다. 이러한 혁신은 임베딩 모델의 효율성을 향상시키며, 대규모 데이터셋을 관리하는 RAG 애플리케이션에서 특히 유용합니다.

- **Technical Details**: Nomic Embed v2는 MoE 구조를 활용하여 모델의 능력을 극대화하지만, 활성 매개변수를 줄이고 학습 효율성을 높입니다. MoE 아키텍처는 전체 파라미터를 사용하지 않고, 입력에 대해 일부 전문가(expert)만 활성화하여 계산 요구 사항을 줄입니다. 이 방법은 전통적인 모델 스케일링 접근법에 비해 많은 이점을 제공합니다.

- **Performance Highlights**: Nomic Embed v2는 대량의 데이터셋을 효율적으로 처리하며, 기존의 모형들에 비해 훨씬 적은 자원으로도 높은 성능을 낼 수 있습니다. 특히, 단일 언어 및 다중 언어의 다양한 벤치마크에서 뛰어난 결과를 보이며, 코드 및 모델을 오픈 소스하여 재현 가능성을 보장합니다. 이로 인해 연구자와 개발자들이 더 나은 텍스트 검색 시스템을 구축하는 데 도움을 줄 수 있습니다.



New uploads on arXiv(cs.CV)

### Embed Any NeRF: Graph Meta-Networks for Neural Tasks on Arbitrary NeRF Architectures (https://arxiv.org/abs/2502.09623)
Comments:
          Under review

- **What's New**: 본 논문은 Neural Radiance Fields (NeRFs)의 다양한 아키텍처를 처리할 수 있는 최초의 프레임워크를 제안합니다. 대조적 학습(Objective) 기법을 통해 아키텍처에 구애받지 않는 잠재 공간(latent space)을 얻는 방법을 설명합니다. 또한, 훈련 시 보지 못한 아키텍처의 NeRFs도 처리할 수 있는 능력을 보여줍니다.

- **Technical Details**: 그래프 메타 네트워크(Graph Meta-Network)를 활용하여 NeRF의 가중치를 입력으로 받아들이고, 이를 기반으로 임베딩(embedding) 공간을 조직합니다. 기존 연구에서 활용된 것처럼 NeRF 아키텍처의 특정 매개변수화에 의존하는 대신, 모델은 다른 아키텍처가 동일한 내용을 표현할 수 있도록 대조 학습을 통해 구조를 형성합니다. 이 방법은 분류(classification) 및 검색(retrieval) 작업에서 효과적으로 활용됩니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 MLP 기반 및 트리 플레인(tri-planar) NeRFs에 대한 분류 및 검색 작업에서 견고한 성능을 나타냅니다. 기존 단일 아키텍처에 제약된 방법들과 비교하여 동등하거나 우수한 결과를 도출합니다. 이를 통해 다양한 NeRF 아키텍처를 효율적으로 처리할 수 있는 가능성을 제시합니다.



### MME-CoT: Benchmarking Chain-of-Thought in Large Multimodal Models for Reasoning Quality, Robustness, and Efficiency (https://arxiv.org/abs/2502.09621)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문에서는 Chain-of-Thought (CoT) 이론이 Large Multimodal Models (LMMs)의 추론 성능에 미치는 영향을 체계적으로 평가하기 위한 MME-CoT라는 새로운 벤치마크를 소개합니다. 이 벤치마크는 수학, 과학, OCR, 논리, 시공간 및 일반 장면의 여섯 가지 도메인에 걸쳐 CoT 추론 성능을 평가합니다. 이러한 과정을 통해 LMMs의 CoT 성능에 대한 첫 번째 포괄적인 연구를 제시합니다.

- **Technical Details**: MME-CoT는 추론 품질, 견고성(robustness) 및 효율성(efficiency)을 세밀하게 평가할 수 있는 세 가지 새로운 메트릭(metrics)을 포함하는 평가 도구를 제안합니다. 고품질 데이터셋을 활용하고 독창적인 평가 전략을 통해 현재 최신 LMMs에 대한 심층 분석을 수행하며, 이 과정에서 몇 가지 중요한 통찰을 발견하였습니다.

- **Performance Highlights**: 주요 발견 내용으로는 1) 반영(reflection) 메커니즘을 가진 모델들이 높은 CoT 품질을 보여주며, Kimi k1.5가 GPT-4o를 능가하여 최고 품질 결과를 도출하였다는 점입니다. 2) CoT 프롬프트(prompts)는 인지 중심의 작업에서 LMM 성능을 저하시킬 수 있으며, 이는 과도한 사고(overthinking)의 부정적인 행동을 시사합니다. 3) CoT 품질이 높음에도 불구하고 반영 기능이 있는 LMMs는 정상 응답 및 자기 수정 단계에서 상당한 비효율성을 보인다는 점입니다.



### Exploring the Potential of Encoder-free Architectures in 3D LMMs (https://arxiv.org/abs/2502.09620)
Comments:
          The code is released at this https URL

- **What's New**: 이번 논문은 엔코더가 없는 아키텍처를 활용하여 3D 이해를 효과적으로 발전시킬 수 있는 가능성을 탐구합니다. 특히, 기존의 엔코더 기반 LMM의 한계를 극복하기 위해 LLM(대형 언어 모델) 내부에서 3D 엔코더의 기능을 통합하는 새로운 전략들을 제안합니다. 이 연구는 3D LMM을 위한 엔코더 없는 아키텍처의 첫 번째 포괄적 조사로, 엔코더를 제거하고 LLM이 고수준의 3D 의미를 추출하도록 합니다.

- **Technical Details**: 논문에서는 LLM-embedded Semantic Encoding(LLM-임베디드 의미 인코딩)과 Hierarchical Geometry Aggregation(계층적 기하집합)이라는 두 가지 핵심 전략을 제안합니다. 첫 번째는 LLM 초기 층을 학습 가능하게 하여 3D 의미를 캡쳐하도록 돕는 방법입니다. 두 번째는 3D 토큰을 기하학적 분포에 따라 집계하여 LLM이 점진적으로 세부적인 3D 의미를 통합하게 합니다. 이 두 전략은 ENEL이라는 엔코더 없는 3D LMM을 통해 구현됩니다.

- **Performance Highlights**: ENEL은 ShapeLLM-13B와 비교하여 클래스 분류에서 55.0%, 캡션 생성에서 50.92%라는 성과를 달성하며, 기존 기술 수준에 가까운 성능을 보여줍니다. 이 연구는라는 것은 엔코더 기반 아키텍처에 비해 엔코더 없는 아키텍처가 3D 이해 분야에서 매우 유망하다는 것을 의미합니다. 최종적으로, ENEL의 출현은 3D 시나리오에 엔코더 없는 아키텍처를 적용하는 효율적인 경로를 제공할 것으로 기대됩니다.



### LIFe-GoM: Generalizable Human Rendering with Learned Iterative Feedback Over Multi-Resolution Gaussians-on-Mesh (https://arxiv.org/abs/2502.09617)
Comments:
          ICLR 2025; Project page: this https URL

- **What's New**: 이번 연구에서는 희소 입력에서 애니메이션이 가능한 인간 아바타의 일반화 가능한 렌더링 방안을 제안합니다. 이를 위해 싱글-pass로 인간 형태를 재구성할 수 있는 방법과 고해상도 렌더링이 가능하도록 하는 방법을 통해 기존 방법의 한계를 극복했습니다. 우리는 반복적인 피드백 업데이트 프레임워크를 도입하여 캔노니컬 인간 형태 표현(skin representation)을 개선하고, 멀티 해상도 가우시안-온-메시(Gaussians-on-Mesh) 표현을 통해 연산 효율성과 고해상도를 달성했습니다.

- **Technical Details**: 연구에서는 이중 형태 표현(dual shape representation) 방식에 개선된 알고리즘을 적용하여, 낮은 해상도의 메시(mesh)로 재구성을 수행하고, 각 삼각형 면에 여러 개의 가우시안을 추가하여 고해상도 렌더링을 구현합니다. 이 기법은 데이터를 더 효과적으로 취합하기 위해 입력, 현재 3D 재구성 및 렌더링의 정보를 융합하는 반복적 업데이트 메커니즘을 사용합니다. 이러한 접근은 적절한 시간 내에 3D 재구성을 완료하는 동시에 고품질의 렌더링을 가능하게 합니다.

- **Performance Highlights**: 제안된 방법은 THuman2.0, XHuman, AIST++ 데이터 세트에서 평가되었습니다. 재구성은 1초 이내에 완료되고, 렌더링은 NVIDIA A100 GPU에서 95.1 FPS로 수행됩니다. PSNR/LPIPS*/FID 성능 지표는 각각 24.65/110.82/51.27로, 기존의 최첨단 기술들을 능가하는 렌더링 품질을 보여주었습니다.



### RigAnything: Template-Free Autoregressive Rigging for Diverse 3D Assets (https://arxiv.org/abs/2502.09615)
Comments:
          Project page: this https URL

- **What's New**: 새로운 연구인 RigAnything은 3D 자산을 리깅(rigging)하기 위한 혁신적인 신경망 모델입니다. 본 모델은 사전 정의된 스켈레톤 템플릿에 의존하지 않고, 각 관절과 스켈레톤 구조를 확률적으로 생성하는 방식을 사용합니다. 이러한 접근 방식은 다양하고 복잡한 3D 객체에 대해 높은 수준의 일반화(generalization)를 제공합니다.

- **Technical Details**: RigAnything은 오토 회귀(autoregressive) 변환기(transformer) 기반의 모델로, 너비 우선 탐색(Breadth-First Search, BFS) 순서로 관절을 배열하여 트리 구조의 스켈레톤을 정의합니다. 이 모델에서는 스켈레톤의 각 관절과 부모 인덱스를 3D 좌표로 표현하고, 이러한 관절의 위치 예측 정확도를 향상시키기 위해 확산 모델(diffusion modeling)을 활용합니다. 모델은 RigNet과 Objaverse 데이터셋에서 엔드 투 엔드(end-to-end)로 훈련되었습니다.

- **Performance Highlights**: RigAnything은 다양한 객체 유형, 예를 들어 인간형, 사륜동물, 해양 생물 및 곤충 등에서 최신 상태의 성능을 발휘합니다. 실험 결과, 기존의 방법들에 비해 품질, 견고성(robustness), 일반화 능력(generalizability) 및 효율성이 크게 향상되었습니다. 이를 통해 3D 자산의 리깅 자동화를 진행하여 전적으로 상호작용 가능한 3D 환경과 대규모 3D 콘텐츠 생성의 비전을 진전시켰습니다.



### Latent Radiance Fields with 3D-aware 2D Representations (https://arxiv.org/abs/2502.09613)
Comments:
          Accepted to ICLR 2025; Project page: this https URL

- **What's New**: 이 논문에서는 2D 특징을 3D 공간으로 변환하는 새로운 Latent 3D Reconstruction 프레임워크를 제안합니다. 기존 연구들이 2D와 3D 사이의 도메인 갭을 극복하는 데 어려움을 겪고 있는 반면, 본 연구는 2D latent space에 3D 인식을 통합함으로써 향상된 성능을 자랑합니다. 특히, radiance field를 사용하여 고품질의 포토리얼리스틱(photorealistic) 3D 재구성을 달성한 것이 주목할 만합니다.

- **Technical Details**: 제안된 프레임워크는 세 가지 주요 단계로 구성됩니다: (1) 3D 일관성을 높이기 위한 correspondence-aware autoencoding 방법, (2) 3D 인식이 포함된 2D 표현을 3D 공간으로 확장하는 latent radiance field(LRF), (3) NVS에 의해 초래되는 데이터 분포 변화를 완화하는 VAE-Radiance Field(VAE-RF) 정렬 방법입니다. 이를 통해 3D와 관련된 latent space와 LRF를 생성하여 기존의 NVS 혹은 3D 생성 파이프라인에 쉽게 통합할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, 본 연구는 다양한 실내 및 실외 씬에서 기존 최첨단 방법들에 비해 더 우수한 합성 품질 및 데이터셋 간 일반화 능력을 보여주었습니다. 특히, 여러 NVS, 3D 생성 및 few-shot NVS 실험에서 뛰어난 성능을 입증했습니다. 이러한 결과는 3D 인식을 결합한 새로운 latent 공간과 LRF의 효과를 잘 보여줍니다.



### Instance Segmentation of Scene Sketches Using Natural Image Priors (https://arxiv.org/abs/2502.09608)
- **What's New**: 이번 논문에서는 스케치를 효과적으로 분할하는 SketchSeg 방법을 소개합니다. 이 방법은 라스터(scene) 스케치의 인스턴스(segment)를 분석할 뿐만 아니라, 클래스에 구애받지 않는 세부 조정(class-agnostic fine-tuning)과 깊이 정보를 활용하여 분할 마스크를 개선하는 기술을 포함합니다. 또한, 스케치를 계층(layer)으로 정리하여 가려진(instance) 객체도 보완할 수 있는 기능을 제공합니다.

- **Technical Details**: 스케치 분할 및 계층화 기법을 통해 사용자는 세그먼트된 객체를 마우스로 조작할 수 있습니다. 이 과정은 CLIPasso와 SketchAgent를 사용하여 생성된 다양한 벡터 스케치를 통해 이루어졌습니다. 특히 CLIPasso는 사진처럼 생생한 이미지를 생성한 후 이를 벡터 스케치로 변환하는 기술을 사용합니다.

- **Performance Highlights**: 우리는 생성된 합성(scene) 데이터 세트를 사용하여 방법론의 견고성을 입증했습니다. 여러 기준점(scene sketch datasets)에서 우리의 결과를 기존 방법들과 비교하였으며, 각 계층화된 스케치의 시각적적인 품질을 보여주었습니다. 최종적으로, 기존 방법들이 겪고 있는 한계를 극복하며 스케치 편집의 효율성을 크게 향상시켰습니다.



### GAIA: A Global, Multi-modal, Multi-scale Vision-Language Dataset for Remote Sensing Image Analysis (https://arxiv.org/abs/2502.09598)
Comments:
          22 pages, 13 figures

- **What's New**: GAIA는 다양한 공간 해상도를 가진 원격 감지(Remote Sensing, RS) 이미지 분석을 위한 새로운 데이터셋으로, 205,150개의 정교하게 구성된 이미지-텍스트 쌍을 포함하고 있습니다. 이 데이터셋은 환경 변화, 자연 재해와 같은 다양한 RS 응용 프로그램을 포착하는 데 초점을 맞추어, 기존의 VLM(비전-언어 모델) 데이터셋의 한계를 극복하고자 합니다. GAIA는 25년 이상에 걸친 공간적 및 시간적 균형 배치를 제공하며, 특히 데이터의 질과 다양성을 확보합니다.

- **Technical Details**: GAIA는 두 단계의 과정으로 구축되었습니다: 신뢰할 수 있는 RS 관련 소스에서 웹 스크래핑을 통한 이미지 및 텍스트 수집과, 각 이미지에 대해 GPT-4o의 비전-언어 기능을 활용하여 생성된 5개의 고품질 합성 캡션을 포함합니다. GAIA는 기존의 웹 크롤링 데이터셋과의 중복을 최소화하며, RS 이미지 분류, 크로스 모달 검색 및 이미지 캡션 작업에 대한 성능 향상을 위한 광범위한 벤치마크를 제공합니다.

- **Performance Highlights**: GAIA 데이터셋을 기반으로 CLIP 및 BLIP2 모델을 미세 조정한 결과, RS 이미지 분류 및 다른 작업에서 현저한 성능 향상이 나타났습니다. 합성 캡션은 원래 사용할 수 있는 대체 텍스트보다 더 나은 의미적 정렬 및 작업 정확성을 제공하며, 연구자들이 RS 분야에서 VLM 관련 연구를 진행할 수 있도록 사전 훈련된 모델 가중치도 공개되었습니다.



### Optimizing GPT for Video Understanding: Zero-Shot Performance and Prompt Engineering (https://arxiv.org/abs/2502.09573)
- **What's New**: 이번 연구에서는 비디오 콘텐츠 분류에서 GPT 기반 모델을 활용하여 제로샷(zero-shot) 분류를 최적화하는 새로운 접근 방식을 제시합니다. 복잡한 정책을 단순화함으로써 허위 부정(false negatives)을 감소시키고, 새롭게 도입한 분해-집계(decomposition-aggregation) 기반 프롬프트 엔지니어링 기법이 기존의 단일 프롬프트 방법들보다 뛰어난 성능을 보여줍니다. 이러한 실험은 실제 산업 문제들을 대상으로 진행되어, 비디오 분류 시스템 개선을 위한 효과적이고 확장 가능한 솔루션을 제공합니다.

- **Technical Details**: 연구에서는 GPT-4의 멀티모달(multimodal) 이해 능력을 평가하며, TikTok 비디오 콘텐츠를 다양한 기준에 따라 분류하는 가능성을 탐구합니다. GPT-4의 제로샷과 피-샷(few-shot) 능력이 산업별 문제에 효과적으로 적용될 수 있는지를 확인하며, 정책 개선과 프롬프트 엔지니어링이 GPT-4 성능에 미치는 영향을 탐색합니다. GPT-4는 실험에서 기존의 이종 분류 모델들과 비교했을 때 여러 카테고리에서 비슷한 수준의 성능을 보였으나, 클릭베이트(clickbait) 비디오와 같은 복잡한 분류에서는 어려움을 겪었습니다.

- **Performance Highlights**: 연구 결과, 복잡한 정책 프롬프트를 간소화하면 허위 부정(false negatives)을 줄이면서 비디오 분류 정확도를 높이는 데 기여합니다. 또한 클릭베이트 탐지와 같은 작업을 하위 카테고리로 나누는 프롬프트 엔지니어링이 상당한 성능 향상을 가져오는 것으로 나타났습니다. 이 발견은 진정한 산업 데이터셋을 기반으로 한 실험을 통해 이루어졌으며, 이는 비디오 분류 작업에 실질적으로 적용 가능하다는 점에서 높이 평가받습니다.



### Self-Calibrating Gaussian Splatting for Large Field of View Reconstruction (https://arxiv.org/abs/2502.09563)
Comments:
          Project Page: this https URL

- **What's New**: 이번 논문에서는 카메라 매개변수와 렌즈 왜곡, 3D Gaussian 표현을 동시에 최적화하는 자기 보정 프레임워크를 제시합니다. 특히, 이 기술은 와이드 앵글 렌즈로 촬영한 대형 시야각(FOV) 이미지를 바탕으로 한 고품질 장면 재구성을 가능하게 하며, 적은 수의 이미지로도 장면을 모델링할 수 있습니다. 이러한 접근 방식은 invertible residual networks와 explicit grids를 결합한 하이브리드 네트워크를 통해 복잡한 렌즈 왜곡을 모델링하는 혁신적인 방법을 소개합니다.

- **Technical Details**: 우리는 3D Gaussian Splatting이라는 차별화된 래스터화 파이프라인을 도입하여 카메라 렌즈 왜곡과 장면 표현을 최적화합니다. 이 방법은 다양한 왜곡을 모델링할 수 있을 만큼 표현력이 뛰어나고, 안정적인 훈련을 위해 잘 정규화되어 있습니다. 특정 이미지의 큰 FOV로 인해 단일 평면 투사가 심각한 픽셀 왜곡과 스트레칭을 초래하기 때문에, 우리는 큐브맵 기반의 리샘플링 전략을 도입하여 왜곡 아티팩트를 줄이고 해상도를 유지합니다.

- **Performance Highlights**: 우리는 FisheyeNeRF 데이터 세트와 우리의 Synectics Mitsuba 데이터 세트를 포함한 다양한 합성 데이터와 실제 장면에서 우리의 방법을 검증하였습니다. 우리의 시스템은 카메라 매개변수 및 렌즈 왜곡을 효과적으로 보정하여, 비보정된 피쉬아이 카메라를 사용할 때 기존 방법들보다 우수한 Gaussian Splatting 성능을 달성합니다. 우리의 파라미터화 방식은 단일 피쉬아이 카메라 모델에 국한되지 않으며, 폭넓은 카메라 모델과 현실 왜곡을 수용할 수 있도록 설계되어 있습니다.



### Long-Term TalkingFace Generation via Motion-Prior Conditional Diffusion Mod (https://arxiv.org/abs/2502.09533)
- **What's New**: 최근 조건부 확산 모델(conditional diffusion models)의 발전은 사실적인 TalkingFace 비디오 생성 가능성을 보여주고 있으나, 지속적인 머리 움직임, 동기화된 표정 및 긴 생성 동안의 정확한 입술 동기화에서 여전히 어려움이 있습니다. 이러한 문제를 해결하기 위해, 우리는 아카이브된 클립과 현재 클립의 모션 프라이어를 활용한 MCDM(Motion-priors Conditional Diffusion Model)을 소개합니다. MCDM은 이러한 요소들을 결합하여 모션 예측을 향상시키고 시간적 일관성을 보장합니다.

- **Technical Details**: 본 모델은 세 가지 주요 구성 요소로 이루어져 있습니다: (1) 아카이브된 클립 모션 프라이어는 역사적 프레임과 참조 프레임을 통합하여 정체성과 맥락을 유지합니다; (2) 현재 클립 모션 프라이어 확산 모델은 다중 모달 인과 관계를 캡처하여 머리 움직임, 입술 동기화 및 표정에 대한 정확한 예측을 제공합니다; (3) 메모리 효율적인 시간적 주의 메커니즘은 모션 피처를 동적으로 저장 및 업데이트하여 오류 누적을 줄입니다.

- **Performance Highlights**: MCDM 실험 결과는 장기 TalkingFace 생성을 위한 정체성과 모션 연속성을 유지하는 데 효과적임을 보여줍니다. 또한, 10개 언어로 200시간 이상의 영상을 포함하는 TalkingFace-Wild 데이터셋을 공개하여 추가 연구를 위한 귀중한 자원을 제공합니다. 코드, 모델 및 데이터셋은 공개될 예정이며, 연구자들이 MCDM의 잠재력을 활용할 수 있도록 도와줍니다.



### SteROI-D: System Design and Mapping for Stereo Depth Inference on Regions of Interes (https://arxiv.org/abs/2502.09528)
Comments:
          Accepted as a full paper by the 2025 EDGE AI FOUNDATION Austin

- **What's New**: 이 연구는 SteROI-D라는 혁신적인 스테레오 깊이 시스템을 제안합니다. SteROI-D는 Region-of-Interest (ROI) 및 시간적 희소성 원리를 활용하여 에너지를 절약합니다. 이 시스템은 다양한 ROIs를 지원하며, 동적 ROIs를 효과적으로 처리하는 체계적인 매핑 방법론도 도입되어 에너지 절약을 극대화합니다.

- **Technical Details**: SteROI-D는 ROI 희소성을 활용하여 매 프레임 깊이 추출 비용을 줄이고, 상호작용하는 객체 감지 및 추적을 통해 ROI 감지 비용을 줄이기 위한 알고리즘을 포함합니다. 또한 Special Compute Units (SCUs)와 NoC Multipackets를 통해 스테레오 깊이 네트워크의 컴퓨팅 및 통신 문제를 해결합니다. Binned Mapping 방법론은 지속적인 ROI 크기 범위에 대해 효율적인 처리를 가능하게 합니다.

- **Performance Highlights**: 28nm 프로토타입 SteROI-D 설계를 사용하여, 기준 ASIC 대비 최대 4.35배의 시스템 전체 에너지 소비 감소를 달성했습니다. 이는 AR/VR 장치에서의 실시간 성능을 유지하면서도 전력 제약을 극복하는 데 중요한 성과로, 동적 ROI 처리에 있어서 새로운 기준을 설정합니다.



### SQ-GAN: Semantic Image Communications Using Masked Vector Quantization (https://arxiv.org/abs/2502.09520)
- **What's New**: 이번 연구에서는 Semantically Masked VQ-GAN (SQ-GAN)이라는 새로운 접근법을 소개하고 있습니다. SQ-GAN은 이미지 압축을 최적화하기 위해 생성 모델을 통합하여 semantics(의미) 및 task-oriented(작업 지향) 통신을 지원합니다. 이 방법은 기존의 JPEG2000과 BPG 같은 이미지 압축 기법보다 우수한 성능을 보여줍니다.

- **Technical Details**: SQ-GAN은 semantic segmentation(의미 분할)과 새롭게 개발된 semantic-conditioned adaptive mask module (SAMM)을 사용하여 이미지의 의미적으로 중요한 특징을 선택적으로 인코딩합니다. 이 모델은 masked vector quantization(마스킹된 벡터 양자화)을 활용하여 데이터 압축 및 의미 정보를 보존하는 데 효과적입니다. 또한, SQ-GAN 아키텍처는 이미지와 그에 대한 의미 분할 맵을 함께 처리하여 두 개의 비트 스트림을 생성하는 방식으로 작동합니다.

- **Performance Highlights**: SQ-GAN은 이미지 압축을 위한 여러 메트릭에서 최신 기법보다 뛰어난 성능을 기록했습니다. 극도로 낮은 Bits Per Pixel (BPP)에서도 세멘틱 정보의 보존을 최적화할 수 있습니다. 이는 실제 환경에서의 안전성 결정을 지원하는 자율주행차와 같은 다양한 응용 프로그램에서 효과적으로 사용할 수 있음을 의미합니다.



### Prior-Constrained Association Learning for Fine-Grained Generalized Category Discovery (https://arxiv.org/abs/2502.09501)
Comments:
          Accepted to AAAI 2025

- **What's New**: 이번 연구는 일반화된 카테고리 발견(Generalized Category Discovery, GCD)을 위한 새로운 접근법인 Prior-constrained Association Learning 방법을 제안합니다. 이 방법은 기존의 방법들이 간과했던 데이터 간 유사성을 활용하여 클래스 특유의 의미를 발견하고자 합니다. 특히, 라벨링된 데이터에서 유래한 고유한 prior 정보를 통해 더 신뢰할 수 있는 그룹화 결과를 도출합니다. 이로 인해 보다 정확한 인스턴스 그룹화를 통해 모델의 표현 학습과 카테고리 발견을 개선합니다.

- **Technical Details**: GCD는 라벨링된 데이터와 라벨링되지 않은 데이터를 모두 활용하여 새로운 카테고리를 발견하는 작업입니다. 기존의 방법들은 self-distillation을 정통으로 활용하여 파라메트릭(Parametric) 분류기를 학습하는 데 집중했지만, 이 논문에서는 라벨링된 데이터의 선험적인 정보를 데이터 간 관계 매핑에 통합하여 보다 정확한 범주를 형성합니다. 이 과정에서 비파라메트릭(non-parametric) 프로토타이프 대조학습(Prototypical Contrastive Learning)을 적용하여 표현 학습을 강화합니다.

- **Performance Highlights**: 제안한 방법은 다양한 GCD 벤치마크에서 기존 방법들보다 현저한 성과를 기록했습니다. CUB 데이터셋에서는 4.4%, Stanford Cars에서는 15.3%의 정확도를 향상시켰습니다. 특히, 제안된 방법은 기존의 파라메트릭 분류기와 비파라메트릭 분류기를 결합하여 각각의 장점을 극대화하는 방법론을 개발하여 성능을 극대화합니다.



### Standardisation of Convex Ultrasound Data Through Geometric Analysis and Augmentation (https://arxiv.org/abs/2502.09482)
- **What's New**: 본 논문은 초음파 데이터 부족 문제를 해결하기 위해 새로운 접근법을 제안합니다. 특히, 비표준화된 초음파 이미지에서 볼록 초음파 평면(convex ultrasound plane)을 자동으로 추출하는 방법을 소개하고 있습니다. 이는 기계 및 파라미터 세팅에 따른 영상의 변동성 문제를 완화하고, 데이터 증강(data augmentation)과 같은 다양한 응용 프로그램에 활용할 수 있습니다.

- **Technical Details**: 제안된 방법은 크게 네 가지 단계로 구성됩니다: 첫째, 초음파 평면 마스킹(ultrasound plane masking) 단계에서 초음파 평면을 대략적으로 찾고 바이너리 마스크로 표현합니다. 둘째, 평면 중심 계산(center of plane calculation) 단계에서는 초음파 평면의 중심선을 결정합니다. 셋째, 방사선 경계 탐지(radial boundary detection) 단계에서는 중심선의 대칭성을 활용하여 경계를 추정하고, 마지막으로 방사형 부채꼴 매개변수 계산(annulus sector parameter calculation) 단계에서 이러한 정보를 바탕으로 매개변수를 도출합니다.

- **Performance Highlights**: 이 방법은 공개 데이터와 개인 데이터 모두에서 평가되었으며, 평면의 정확성 및 강건성을 검증했습니다. 제안된 접근법은 방사형 부채꼴 매개변수를 사용하여 평면의 자동 선형화(linearisation)를 가능하게 하며, 이미지의 변형(deformation)과 증강 후 원형 객체의 변화도 연구했습니다. 또한, 본 연구는 기존의 방법들과 비교할 때 자동화를 통해 수작업 주석과 데이터 조직의 노동력을 줄이는 데 기여할 것으로 기대됩니다.



### Wholly-WOOD: Wholly Leveraging Diversified-quality Labels for Weakly-supervised Oriented Object Detection (https://arxiv.org/abs/2502.09471)
Comments:
          18 pages, 9 figures, 9 tables, accepted by TPAMI

- **What's New**: 이번 논문에서는 Wholly-WOOD라는 새로운 약한 감독 기반의 지향 객체 탐지(Oriented Object Detection, OOD) 프레임워크를 제안합니다. 이 프레임워크는 다양한 레이블 형식(Points, HBoxes, RBoxes 등)을 통합하여 사용할 수 있도록 설계되었습니다. 훈련 시 HBox만을 사용하더라도 RBox 기반의 모델과 비슷한 성능을 보여주어, 지향 객체에 대한 수작업 주석을 줄일 수 있는 가능성을 제시합니다.

- **Technical Details**: Wholly-WOOD는 약한 감독(weak supervision) 접근 방식을 사용하여 점 형태(Point) 및 수평 경계 상자(HBox)와 같은 저급 레이블을 활용하여 지향 경계 상자(RBox)를 생성합니다. 이 방법은 대칭 인식 학습(symmetric-aware learning) 이론을 바탕으로 물체 각도를 학습하고, 합성 패턴 지식을 활용하여 포인트에서 RBox로 변환을 수행합니다. 이러한 기술적 접근은 데이터 주석의 비용을 줄이면서도 효율적인 학습을 가능하게 합니다.

- **Performance Highlights**: Wholly-WOOD는 HBox 또는 Point에서 RBox로의 전환 성능에서 기존 최첨단 방법들보다 우수한 정확도를 기록하였습니다. 특히 Point에서 RBox로의 변환 성능은 22.36% 향상되었으며, 이는 통합 아키텍처와 새로운 P2R 서브넷의 효과 덕분입니다. 또한, 원거리 감시(remote sensing) 및 다양한 응용 프로그램에서의 수작업 주석 감소 효과가 입증되었습니다.



### Pixel-Level Reasoning Segmentation via Multi-turn Conversations (https://arxiv.org/abs/2502.09447)
- **What's New**: 본 논문에서는 기존의 시각 인식 시스템이 단일 대화에서의 지역 수준 세분화에 집중하고 있으며, 동적인 사용자 의도를 이해하는 데 한계가 있음을 지적합니다. 이를 해결하기 위해, 다중 턴 대화 기반의 픽셀 수준 추론 세분화(Pixel-level Reasoning Segmentation, Pixel-level RS)라는 새로운 과제를 소개합니다. 또한, 이 과제를 위한 픽셀 수준 추론 세분화 데이터셋(PRIST)을 구축하여 총 24,000개의 발화와 8,300개의 다중 턴 대화 시나리오를 포함하고 있습니다.

- **Technical Details**: PRIST 데이터셋은 3단계 대화 자동 생성 파이프라인을 통해 다중 턴 상호작용을 통해 사용자 의도를 이해하고 픽셀 수준 설명과 세분화 마스크를 생성합니다. 이를 지원하기 위해 다중 비전 인코더와 의미적 영역 정렬 전략을 채택하여 상세한 시각 정보를 포착하고 세분화 성능을 향상시킵니다. 또한, 이 프레임워크는 사용자 의도를 반복적으로 명확하게 하기 위해 다중 턴 상호작용을 지원합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 PRIST 데이터셋에서 기존 세분화 기준선보다 더 우수한 성능을 보였으며, 세분화 및 LLM 기반의 추론 메트릭에서 두각을 나타냈습니다. 평가 메트릭은 대화의 일관성, 일치성 및 정확성 측면에서 종합적으로 발전한 결과를 보여줍니다. 이 연구는 픽셀 수준 추론 세분화의 발전을 위한 새로운 기준을 제시합니다.



### Redistribute Ensemble Training for Mitigating Memorization in Diffusion Models (https://arxiv.org/abs/2502.09434)
Comments:
          12 pages,9 figures. arXiv admin note: substantial text overlap with arXiv:2407.15328

- **What's New**: 이 논문에서는 메모리 보존 문제를 해결하기 위해 새로운 Diffusion 모델 프레임워크를 제안합니다. Visual modality 관점에서 접근하여 모델이 데이터를 직접 학습하지 않고, 여러 모델의 파라미터를 통해 학습하도록 설계되었습니다. 이를 통해 특정 콘텐츠를 숨기는 것뿐만 아니라보다 근본적으로 메모리를 줄이는 방법을 제공합니다.

- **Technical Details**: 제안된 방법은 Iterative Ensemble Training (IET) 및 Anti-Gradient Control (AGC) 메커니즘을 포함합니다. 데이터셋을 여러 샤드로 분할하고 각 샤드가 서로 다른 프로시 모델을 훈련하여 최종 모델을 형성합니다. 또한, 훈련 중에 손실값이 비정상적으로 낮은 샘플을 건너뛰어 메모리를 줄이는 기법을 구현했습니다.

- **Performance Highlights**: 실험 결과, CIFAR-10, CIFAR-100, AFHQ-DOG 데이터셋에서 메모리 양이 각각 90.1%, 74.6%, 91.2% 감소했습니다. 특히 Stable Diffusion 모델의 경우에는 메모리 점수를 46.7% 낮추는 성과를 보였습니다. 이러한 결과는 제안된 방법이 기존 기법보다 효과적임을 입증합니다.



### A 3D Facial Reconstruction Evaluation Methodology: Comparing Smartphone Scans with Deep Learning Based Methods Using Geometry and Morphometry Criteria (https://arxiv.org/abs/2502.09425)
- **What's New**: 본 연구에서는 고급 3D 얼굴 취득 방법에 대한 대안으로, 저비용의 3D 얼굴 모델 취득 및 재구성 기법을 제안합니다. 기존의 기하학 기반 평가를 넘어서는 새로운 평가 방법론을 도입하여 얼굴 형태 보존을 정량적으로 평가할 수 있는 통계적 프레임워크를 제공합니다.

- **Technical Details**: 저비용의 3D 얼굴 취득 방법을 검증하기 위해, 휴대폰 기반 스캔과 2D 이미지에서의 딥러닝 3D 재구성을 비교하였습니다. 특히, Geometric Morphometrics (GM) 기술을 활용하여 얼굴 형태의 정밀한 분석을 실시하였으며, Generalized Procrustes Analysis (GPA)와 Euclidean Distance Matrix Analysis (EDMA) 기법을 적용하여 전반적인 형태와 지역적 차이를 평가하였습니다.

- **Performance Highlights**: 실험 결과, 스마트폰 스캔이 기존의 고해상도 입증 모델에 비해 기하학적 및 형태학적 유사성이 더 우수한 성과를 보였습니다. 이는 저비용 3D 얼굴 취득 기법이 실제 생물학적 의미를 고려한 검증 접근 방식을 제공함을 나타냅니다.



### ImageRAG: Dynamic Image Retrieval for Reference-Guided Image Generation (https://arxiv.org/abs/2502.09411)
- **What's New**: Diffusion 모델은 고품질의 다양한 비주얼 콘텐츠 합성을 가능하게 하지만, 드문 혹은 본 적 없는 개념을 생성하는 데 어려움을 겪고 있습니다. 이를 해결하기 위해, Retrieval-Augmented Generation (RAG) 접근 방식을 사용하는 새로운 이미지 생성 방법인 ImageRAG를 제안합니다. ImageRAG는 주어진 텍스트 프롬프트에 기반하여 관련 이미지를 동적으로 검색하고 이를 생성 과정의 컨텍스트로 활용합니다.

- **Technical Details**: ImageRAG는 사전 훈련된 T2I 모델의 생성 능력을 향상시키기 위해 기존의 이미지 조건 모델의 기능을 활용합니다. 이전의 접근 방법들은 검색된 이미지를 향상시키기 위해 특정 훈련 모델을 사용한 반면, ImageRAG는 RAG 전용 훈련 없이도 작동 가능하다는 점에서 차별화됩니다. 이 방법은 다양한 모델 유형에 쉽게 적용할 수 있고, 드물고 세밀한 개념을 생성하는 데 있어 주목할 만한 성과를 보여줍니다.

- **Performance Highlights**: ImageRAG는 ICL(In-Context Learning)이 가능한 T2I 모델과 IP-adapter 이미지 인코더를 사용하는 모델에서 잘 작동하며, 두 모델 모두에서 드물고 세밀한 개념 생성 능력 향상을 입증했습니다. 이러한 결과는 이미지 생성 커뮤니티가 샘플링 시간에 RAG를 사용하는 것을 고려해야 한다는 점을 강조합니다. 더 나아가, 두 모델에 대한 정량적 및 정성적 비교를 통해 RAG의 효과성을 명확히 입증하였습니다.



### Galileo: Learning Global and Local Features in Pretrained Remote Sensing Models (https://arxiv.org/abs/2502.09356)
- **What's New**: 이 논문에서는 원거리 감지(remote sensing) 분야에서 사용할 수 있는 사전 학습(pretrained)된 머신러닝 모델인 Galileo를 소개합니다. 이 모델은 다양한 센서 모달리티(sensor modalities)와 형태를 유연하게 처리할 수 있도록 설계되었습니다. 또한, 다양한 규모(scale) 및 유형(types)의 지구 표면 현상을 모델링할 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: Galileo 모델은 멀티모달(multimodal) 원거리 감지 데이터를 유연하게 처리하기 위해 설계되었습니다. 이 모델은 큰 규모의 특징과 작은 규모의 특징을 모두 학습할 수 있는 새로운 자가 감독(self-supervised) 학습 접근법을 도입하여, 이전 모델들이 해결하지 못했던 문제를 다룹니다. 이러한 자기 감독 학습 방법은 다양한 공간(spatial) 및 시간적(temporal) 차원을 가진 데이터를 효과적으로 처리할 수 있게 합니다.

- **Performance Highlights**: Galileo 모델은 여러 원거리 감지 작업(task)에서 최첨단(state-of-the-art) 결과를 달성하였습니다. 이를 통해 적게는 수 작고 큰 규모의 현상을 동시 처리할 수 있어서 데이터 라벨링(labeling) 및 작업 수행에 필요한 노력과 자원을 크게 줄일 수 있습니다. 이러한 성능은 원거리 감지 분야에서의 여러 사회적 이점(societal benefits)을 강화하는 데 기여할 것입니다.



### A Benchmark for Crime Surveillance Video Analysis with Large Models (https://arxiv.org/abs/2502.09325)
- **What's New**: 이번 연구는 범죄 감시 비디오 분석을 위한 새로운 벤치마크인 UCVL을 제안합니다. UCVL은 1,829개의 비디오와 두 개의 데이터 세트에서 재구성한 주석을 포함하며, 여러 유형의 질문과 QA 쌍을 생성하여 다각적인 평가를 가능하게 합니다. 이研究는 MLLM(Multimodal Large Language Models)의 적합성을 검증하기 위한 독창적인 접근법을 제공합니다.

- **Technical Details**: UCVL은 UCF-Crime 및 UCF-Crime Annotation 데이터 세트를 통합하여 높은 품질의 데이터를 제공합니다. 연구진은 Qwen2 LLM과 GPT-4o를 사용하여 질문-답변 쌍을 생성하고, MLLMs의 성능을 객관적으로 평가할 수 있는 엄격한 스코어링 시스템을 개발하였습니다. 이는 기존 작업 특화된 데이터 세트의 한계를 극복하고 MLLMs의 다중 작업 분석 능력을 강화합니다.

- **Performance Highlights**: 연구 결과, 총 300개의 비디오를 테스트 세트로 하여 8개의 유력한 MLLM의 성능을 비교하였습니다. LLaVA-OneVision의 미세 조정이 이루어진 후 큰 성능 향상이 관찰되었으며, 이는 UCVL 데이터 세트의 높은 품질을 입증합니다. 이 연구는 MLLMs가 비디오 이상 분석에서 어떤 잠재력을 가지는지를 보여줍니다.



### Mitigating the Impact of Prominent Position Shift in Drone-based RGBT Object Detection (https://arxiv.org/abs/2502.09311)
Comments:
          15 pages

- **What's New**: 이 논문에서는 드론 기반 RGBT(적외선 및 RGB) 물체 감지에서 발생하는 위치 이동 문제를 심각하게 다루고 있습니다. 기존의 많은 방법들이 이미지 쌍이 기하학적으로 정렬되어 있다고 가정하지만, 실제로는 약한 정렬 상태에 있어 물체 위치의 shift가 발생합니다. 저자들은 이러한 문제를 레이블 노이즈 문제로 간주하고, 이를 해결하기 위한 새로운 Mean Teacher 기반의 Cross-modality Box Correction (CBC) 모듈을 제안합니다.

- **Technical Details**: 주요 기술적 기여로는 CBC 모듈과 Shifted Window-Based Cascaded Alignment (SWCA)가 있습니다. CBC 모듈은 감지 모드의 GT(ground truth) 박스를 실시간으로 수정하며, 이를 통해 더 정확한 표현 학습을 가능하게 합니다. SWCA는 감지된 기능과 참조 기능의 긴 범위의 종속성을 기반으로 정렬을 수행하여, 작은 물체의 특정한 기능을 유지하면서 정확한 정렬을 CMP합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 RGBTDronePerson 및 DroneVehicle 데이터셋에서 일관된 성능 향상을 보여줍니다. 특히, CBC 모듈은 감지된 모드의 정확성을 25.52 aSim 포인트 향상시켰으며, RGBTDronePerson 데이터세트에서 mAP_50 점수는 43.55에 도달하여 최신 방법보다 8.6 mAP50 포인트 높은 성능을 달성하였습니다. 이 논문에서는 향후 코드를 공개할 예정입니다.



### A Physics-Informed Deep Learning Model for MRI Brain Motion Correction (https://arxiv.org/abs/2502.09296)
- **What's New**: 본 연구에서는 MRI에서 발생하는 모션 아티팩트를 효과적으로 제거하기 위해 PI-MoCoNet이라는 물리 기반의 모션 보정 네트워크를 소개합니다. 이 네트워크는 공간(domain)과 k-space 정보를 통합하여 모션 파라미터를 명시적으로 추정하지 않고도 이미지의 정확성을 향상시킵니다. 특히, 이 접근 방식은 비강체(non-rigid) 모션에 확장할 수 있는 가능성을 제공합니다.

- **Technical Details**: PI-MoCoNet은 두 가지 상호 연결된 네트워크로 구성됩니다: 모션 탐지 네트워크와 모션 보정 네트워크입니다. 모션 탐지 네트워크는 U-net 아키텍처를 사용하여 k-space에서 손상된 영역을 식별하고, 모션 보정 네트워크는 이를 바탕으로 Motion-free 이미지를 재구성합니다. 세 가지 손실 함수, 즉 재구성(L1), 지각(LPIPS), 데이터 일관성(Ldc)를 통해 보정이 이루어집니다.

- **Performance Highlights**: PI-MoCoNet은 다양한 모션 아티팩트의 이미지 품질을 유의미하게 개선하였습니다. 평가 결과, IXI 데이터셋의 경미한 아티팩트 경우 PSNR이 34.15 dB에서 45.95 dB로, SSIM은 0.87에서 1.00으로 증가했습니다. 아울러, 중간 및 심한 아티팩트의 경우에도 PSNR 및 SSIM 지표가 현저하게 개선되어 기존 방법들에 비해 뛰어난 성능을 보였습니다.



### EmoAssist: Emotional Assistant for Visual Impairment Community (https://arxiv.org/abs/2502.09285)
- **What's New**: 이 논문은 EmoAssist Benchmark를 소개하여 시각장애인(VI) 커뮤니티를 위한 대형 다중 모달 모델(LMM)의 보조 성능을 평가하는 첫 번째 종합 벤치마크를 제시합니다. 이 벤치마크는 감정 지능(emotional intelligence)을 평가 기준으로 통합했으며, 그 결과 감정적 필요를 충족시켜 VI 개인에게 도움을 줄 수 있는 모델을 구축하는 데 중점을 두고 있습니다.

- **Technical Details**: EmoAssist Model은 감정 지원을 목적으로 설계된 LMM으로, Direct Preference Optimization (DPO) 기법을 활용하여 인간의 감정적 선호를 이해하고 조정합니다. 이 모델은 Low-Rank Adaptation (LoRA) 기술을 통해 파라미터 효율적인 미세 조정을 수행하여 VI 사용자의 감정과 의도를 인식하는 데 강점을 보입니다.

- **Performance Highlights**: EmoAssist 모델은 EmoAssist Benchmark의 Empathy 및 Suggestion 메트릭에서 각각 147.8%와 89.7%의 개선을 보이며, GPT-4o와 같은 최신 모델보다도 우수한 성능을 입증했습니다. 실험 결과, EmoAssist 모델은 감정 인식 및 공감 응답 제공에서 뛰어난 결과를 보여 VI 개인에게 필요한 실질적인 지원을 제공합니다.



### FE-LWS: Refined Image-Text Representations via Decoder Stacking and Fused Encodings for Remote Sensing Image Captioning (https://arxiv.org/abs/2502.09282)
- **What's New**: 이번 논문은 두 개의 서로 다른 CNN 기반 인코더에서 특성을 통합하여 더 나은 원격 감지 이미지 캡션 생성을 위한 새로운 접근 방식을 제안합니다. 또한, 단일 인코더와 두 개의 GRU 스택을 활용하여 캡션 선택성을 정교화하는 방식으로 가중 평균 기법을 도입했습니다. 이러한 접근 방법을 통해 전통적인 Transformer 기반 모델 및 LSTM 기반 기준 모델과 비교해 상당한 성능 향상을 보여주고 있습니다.

- **Technical Details**: 이 논문은 인코더-디코더 구조에서 CNN을 사용하여 이미지 기능을 추출하고, RNN 계열의 GRU를 통해 시퀀스를 생성하는 방법을 사용합니다. 특히, 혁신적인 두 레이어 stacked GRU 디코더를 통해 생성된 캡션의 품질이 개선됩니다. 더불어, 캡션 선택을 향상시키기 위해 Comparison-based Beam Search (CBS) 기술도 도입하였습니다.

- **Performance Highlights**: 제안된 융합 기반 접근법은 캡션 생성에서 기존의 최첨단 모델보다 우수한 성과를 보였습니다. 성능 평가 결과, 본 연구의 방법론은 이미지 캡션 생성의 정확성 및 관련성을 크게 향상시켰으며, 다양한 벤치마크 데이터셋을 기반으로 실험을 통해 그 우수성을 입증하였습니다.



### ConsistentDreamer: View-Consistent Meshes Through Balanced Multi-View Gaussian Optimization (https://arxiv.org/abs/2502.09278)
Comments:
          Manuscript accepted by Pattern Recognition Letters

- **What's New**: 최근 확산 모델(diffusion models)의 발전으로 이미지에서 생성된 자산을 활용한 3D 생성이 크게 개선되었습니다. 그러나 이미지-3D 변환에서의 일관성 부족으로 여러 뷰에서의 품질 차이가 발생하므로 이러한 문제를 해결하기 위해 ConsistentDreamer를 제안합니다. 이 방법은 고정된 다중 뷰 이전 이미지(multi-view prior images)를 생성하고, SDS 손실을 통해 무작위 뷰를 샘플링하여 일관성을 유지합니다.

- **Technical Details**: ConsistentDreamer는 먼저 다중 뷰 확산 사전(multi-view diffusion prior)을 사용하여 일관된 다중 뷰 이미지를 생성한 후, 최적화 과정에서 두 번째 확산 사전이 무작위 뷰 생성을 안내하도록 합니다. 각 최적화 단계에서 미세 세부 정보(fine detail)를 재구성하는 동시에 거친 형태(rough shape)를 유지하면서 손실 균형을 활용합니다. 동적 작업 종속 가중치(task-dependent weights)를 도입하여 이러한 최적화 간의 균형을 자동으로 조정합니다.

- **Performance Highlights**: 우리의 방법은 기존 최첨단 기술에 비해 더 나은 뷰 일관성(view consistency)과 비주얼 퀄리티(visual quality)를 확립합니다. 최적화 과정에서 얻은 결과는 더욱 향상된 형태와 세부 정보로, 다중 뷰 이미지를 통한 정밀 재구성을 통해 전체적으로 더 완벽한 3D representación을 제공합니다. ConsistentDreamer는 다채로운 시점을 포함한 최적화된 3D 자산 생성을 가능하게 합니다.



### FLARES: Fast and Accurate LiDAR Multi-Range Semantic Segmentation (https://arxiv.org/abs/2502.09274)
- **What's New**: 이 논문에서는 FLARES라는 새로운 훈련 스키마를 도입하여 LiDAR 기반의 시멘틱 세그멘테이션(semantic segmentation) 작업을 혁신적으로 설계하였습니다. 기존의 고해상도(range-view) 이미지를 사용하는 방법 대신, 저해상도로 전환하여 계산 효율성과 정확도를 향상시켰습니다. 이 접근법은 다양한 네트워크 아키텍처에 일반화 가능하며, LiDAR 데이터의 'many-to-one' 문제를 더 효과적으로 해결합니다.

- **Technical Details**: FLARES는 점군(point cloud)을 여러 개의 하위 점군으로 나누고 훈련 중에 저해상도 이미지로 투영하여 처리합니다. 이 방식은 수평 및 수직 모두에서 투영 비율을 증가시켜, 적은 비용으로 range-view 표현의 정보성을 높입니다. 또한, 데이터 증강(data augmentation)을 통해 클래스 불균형 문제를 완화하는 방법과, 기존의 후처리(post-processing) 방법을 개선하는 방법도 제안했습니다.

- **Performance Highlights**: 실험 결과, FLARES는 공공 데이터셋에서 다양한 네트워크 아키텍처의 성능을 기존 기준선(baseline)과 비교하여 유의미하게 향상시켰습니다. 논문에서 제시한 새로운 데이터 증강 기법과 후처리 메소드는 특히 LiDAR 기반 인식을 위한 더 효과적인 경로를 제공하며, 자율 주행 시스템에서의 적용 가능성을 높임을 보여주었습니다.



### Memory-based Ensemble Learning in CMR Semantic Segmentation (https://arxiv.org/abs/2502.09269)
- **What's New**: 본 논문에서는 기존의 3D 프레임 또는 2D 슬라이스를 독립적으로 분할하는 모델과 달리, 세그멘테이션 불확실성을 공간적 연속성을 활용하여 추출하고 이를 앙상블 학습의 메모리로 활용하는 방법을 제시합니다. 이를 통해 전체 성능과 끝 슬라이스의 성능을 조화롭게 균형 잡을 수 있습니다. 또한, 끝 슬라이스의 정확도를 수량화하기 위해 End Coefficient(EC)를 도입했습니다.

- **Technical Details**: 심혈관 자기공명영상(CMR)에서의 심장 시네 기법은 비침습적인 심혈관 기능 평가의 금본위 제도로 자리 잡았습니다. 연구자들은 여러 접근 방식을 통해 심실(segmentation) 분할 문제를 해결하고 있으며, 이들 접근 방식에서 딥러닝의 역할이 중요해졌습니다. 본 연구에서는 Streaming이라는 파이프라인을 설계하였으며, 이는 기본 분류기로 신경망을 사용하여 다양한 분류기의 출력을 통합합니다.

- **Performance Highlights**: ACDC 및 M&M 데이터셋에서 실시한 실험 결과, 본 연구의 프레임워크는 거의 최첨단의 Dice Similarity Coefficient(DSC)를 달성했습니다. 특히, 끝 슬라이스 성능에서 모든 모델을 초월하며, 환자 개별의 세그멘테이션 정확도를 개선하는 것으로 나타났습니다. 이러한 향상은 본 논문의 제안한 메모리 기반 불확실성 메커니즘을 통해 이루어졌습니다.



### DynSegNet:Dynamic Architecture Adjustment for Adversarial Learning in Segmenting Hemorrhagic Lesions from Fundus Images (https://arxiv.org/abs/2502.09256)
Comments:
          12 pages,4 figures

- **What's New**: 이 논문은 망막 hemorrhagic lesion segmentation 분야에 새로운 접근 방식을 제안합니다. 구체적으로, 이 방법은 adversarial learning 기반의 동적 아키텍처 조정을 통해 segmentation 성능을 개선합니다. 기존의 기술적 한계를 극복하기 위해 다양한 최신 기법들을 통합하여 적용했습니다.

- **Technical Details**: 제안된 방법은 계층적 U자형 인코더-디코더(encoder-decoder) 구조, residual blocks, attention mechanisms, 그리고 ASPP(Atrous Spatial Pyramid Pooling) 모듈을 통합합니다. 이러한 구조는 동적으로 feature fusion을 최적화하여 세부적인 segmentation을 가능하게 합니다. 이를 통해 전체적인 성능 개선이 이루어집니다.

- **Performance Highlights**: 실험 결과, Dice coefficient는 0.6802, IoU는 0.5602, Recall은 0.766, Precision은 0.6525, Accuracy는 0.9955로 나타났습니다. 이러한 결과는 fundus image의 hemorrhage segmentation에서 발생하는 여러 도전 과제를 효과적으로 해결할 수 있음을 보여줍니다.



### Faster than real-time detection of shot boundaries, sampling structure and dynamic keyframes in video (https://arxiv.org/abs/2502.09202)
Comments:
          Accepted for ICISPC 2024

- **What's New**: 이번 논문에서는 비디오 분석에서 기본적인 작업인 샷 경계(shot boundaries) 감지, 샘플링 구조(sampling structure) 및 동적 키프레임(dynamic keyframes) 검출을 통합적으로 수행할 수 있는 새로운 알고리즘을 제안합니다. 이 알고리즘은 모션 필드(motion field)와 정규화된 교차 상관(normalized cross correlation)에서 파생된 프레임 간(inter-frame) 및 프레임 내(intra-frame) 측정을 결합하여 비디오 분석을 실행합니다.

- **Technical Details**: 제안된 알고리즘은 비디오를 실시간으로 처리할 수 있는 속도의 4배에 달하는 속도로 동작합니다. 이는 이 측정 값들을 희소하고 선택적으로 계산(selective calculation)하기 때문입니다. 또한, 알고리즘은 대규모 카메라 이동이나 객체 이동, 플래시, 깜빡임(flicker), 낮은 대비(low contrast) 및 잡음(noise) 등과 같은 도전적인 콘텐츠에서도 매우 견고합니다.

- **Performance Highlights**: 초기 평가에서는 제안된 알고리즘이 매우 효율적이고 정확한 성능을 가진 것으로 나타났습니다. 이 알고리즘은 다양한 비디오 콘텐츠에서 빠르고 신뢰할 수 있는 분석 결과를 제공할 수 있어, 고급(high-level) 분석 작업 수행 전에 필수적인 도구로 자리잡을 것입니다.



### E-MD3C: Taming Masked Diffusion Transformers for Efficient Zero-Shot Object Customization (https://arxiv.org/abs/2502.09164)
Comments:
          16 pages, 14 figures

- **What's New**: 이 논문에서는 E-MD3C (Efficilent Masked Diffusion Transformer with Disentangled Conditions and Compact Collector)를 제안합니다. 이 프레임워크는 제로샷 개체 이미지 사용자화(zero-shot object image customization)에 대해 고효율적인 접근 방식을 제공합니다. 기존의 자원 집약적인 Unet 아키텍처와는 달리, 경량화된 masked diffusion transformers를 사용하여 계산 효율성을 크게 향상시킵니다.

- **Technical Details**: E-MD3C는 세 가지 핵심 구성 요소를 통합합니다. 첫째, 효율적인 masked diffusion transformer가 autoencoder의 잠재 공간(latent space)을 처리합니다. 둘째, 분리된 조건 설계를 통해 배경 정렬과 세부사항 보존을 유지하면서 정보를 압축합니다. 셋째, 학습 가능한 Conditions Collector는 여러 입력을 통합하여 간결한 표현으로 변환하여 효과적인 노이즈 제거와 학습을 지원합니다.

- **Performance Highlights**: E-MD3C는 VITON-HD 데이터셋에서 PSNR, FID, SSIM 및 LPIPS와 같은 지표에서 기존 방법보다 우수한 성능을 보입니다. 단 1/4의 파라미터 수를 사용하여, 1720M Unet 기반의 latent diffusion 모델에 비해 2.5배 빠른 추론 속도와 2/3의 GPU 메모리를 소모합니다. 이러한 성능 개선은 E-MD3C의 간결하고 효과적인 설계 덕분입니다.



### Multimodal HIE Lesion Segmentation in Neonates: A Comparative Study of Loss Functions (https://arxiv.org/abs/2502.09148)
- **What's New**: 본 연구는 Neonatal MRI에서 Hypoxic-Ischemic Encephalopathy (HIE) 병변의 세분화를 위한 최적의 손실 함수(loss function)를 식별하는 것을 목표로 한다. 연구팀은 BONBID-HIE 데이터셋을 활용하여 다양한 손실 함수를 평가하였고, compound loss가 단일 손실 함수보다 우수한 성능을 발휘함을 발견하였다. 이 연구는 데이터 제약 상황에서도 HIE 병변 세분화의 정확도를 높일 수 있는 방법론을 제공한다.

- **Technical Details**: 세분화 성능은 선택된 손실 함수에 크게 의존하며, 연구에서는 Dice, Tversky, Hausdorff Distance Loss 및 새로운 compound 손실 함수인 Dice-Focal-HausdorffDT와 Tversky-HausdorffDT를 평가하였다. Hands-on 실험에서는 각 손실 함수가 서로 다른 세분화 마스크를 예측함을 보였고, Tversky-HausdorffDT 손실이 Dice와 Normalized Surface Dice 점수에서 가장 높은 성과를 기록하였다. 이러한 결과는 각 작업에 적합한 손실 함수 최적화의 중요성을 강조한다.

- **Performance Highlights**: 이 연구에서 발견된 바에 따르면, compound loss가 독립적인 손실 함수를 초과하는 성능을 발휘했다. 특히 Tversky-HausdorffDT Loss는 가장 높은 Dice 및 Normalized Surface Dice 점수를 달성하였으며, Dice-Focal-HausdorffDT Loss는 Mean Surface Distance를 최소화하였다. 이는 HIE 병변 세분화를 위한 손실 함수의 조합이 훈련 데이터가 제한적일 경우에도 더욱 정확한 결과를 도출할 수 있음을 입증한다.



### Feature-based Graph Attention Networks Improve Online Continual Learning (https://arxiv.org/abs/2502.09143)
Comments:
          16 pages

- **What's New**: 이번 논문에서는 Graph Attention Networks (GATs)를 기반으로 한 온라인 연속 학습 프레임워크를 제안하여 이미지 분류를 위한 새로운 접근 방식을 제공합니다. 이 프레임워크는 컨텍스트 관계를 효과적으로 포착하고 학습된 주의 가중치를 통해 작업별 표현을 동적으로 업데이트합니다. 또한, 고급 글로벌 풀링 전략을 통합하여 지속적인 학습을 위한 분류 성능을 개선하는 것을 목표로 합니다.

- **Technical Details**: 제안된 모델인 Feature-based Graph Attention Networks (FGAT)는 이미지 데이터를 그래프 형태로 변환하고, 각 노드가 다중 스케일 특성 정보를 인코딩하여 복잡한 피처 간의 관계를 모델링합니다. 이를 위해 사전 훈련된 피쳐 추출기를 사용하며, GAT를 통해 로컬 주변 맥락에 기반하여 동적 노드 표현 업데이트가 가능합니다. 또한, 맞춤형 가중 평균 풀링 메커니즘을 통해 모든 노드의 정보를 상대적 중요도에 따라 집계합니다.

- **Performance Highlights**: FGAT는 SVHN, CIFAR10, CIFAR100, MiniImageNet과 같은 벤치마크 데이터셋에서 기존의 CNN 및 GNN 기반의 연속 학습 방법들과 비교하여 우수한 성능을 입증했습니다. 특히, 이전 작업의 표현력을 강화하면서 새로운 작업의 성능을 유지할 수 있는 기법인 리허설 메모리 중복 전략을 제안하여 메모리 예산을 효율적으로 관리합니다. 결과적으로, 제안한 방법은 최신 기술 대비 성능이 뛰어난 것으로 나타났습니다.



### Automatic Pruning via Structured Lasso with Class-wise Information (https://arxiv.org/abs/2502.09125)
Comments:
          11 pages, 2 figures

- **What's New**: 이번 연구에서는 정확한 클래스별(class-wise) 정보를 활용하여 신경망 모델의 프루닝(pruning) 접근 방식을 개선했습니다. 정보 병목(Information Bottleneck) 이론을 기반으로 한 구조적 라쏘(structured lasso) 기법을 이용하여 통계적 정보를 유지하면서 프루닝을 수행하는 두 가지 새로운 방법, 즉 sGLP-IB와 sTLP-IB를 제안합니다. 이를 통해 데이터 세트와 모델 아키텍처의 다양성을 감안하여 우수한 성능을 입증하였습니다.

- **Technical Details**: 제안된 방법들은 CNN의 특징 맵(feature map) 통계 정보를 모델링하여 프루닝 손실을 최소화하는 데 목표를 두고 있습니다. 또한, 그래프 구조화된 라쏘(graph-structured lasso) 및 트리 유도 라쏘(tree-guided lasso)를 사용하여 필터 간의 관계를 고려하며, 클래스별 정보를 통합하는 접근 방식을 채택합니다. 이러한 구조적 라쏘 기법을 활용하여 모델 필터의 프루닝을 보다 정밀하게 진행하고, 누적 오류를 방지하며 계층 수준(layer-wise)에서 수행합니다.

- **Performance Highlights**: VGG16 모델을 CIFAR-10 데이터 세트에서 사용했을 때, 85%의 매개변수 감소와 61%의 FLOPs 감소를 달성하였으며, 정확도는 원래 모델보다 0.14% 높은 94.10%를 기록했습니다. 또한 ResNet 아키텍처를 기준으로 ImageNet 데이터 세트에서 매개변수를 55% 줄이고 정확도를 76.12%로 유지할 수 있었습니다. 이러한 실험 결과는 제안된 방법의 효과성과 강력한 클래스 정보 학습 능력을 입증합니다.



### DenseSplat: Densifying Gaussian Splatting SLAM with Neural Radiance Prior (https://arxiv.org/abs/2502.09111)
- **What's New**: 본 논문은 DenseSplat라는 새로운 SLAM 시스템을 소개하며, 이는 NeRF와 3DGS의 장점을 결합하여 실시간 추적, 매핑 및 루프 수정을 가능하게 합니다. 기존 SLAM 시스템의 한계인 밀집 키프레임 의존성을 줄이기 위해 희소하게 샘플링된 키프레임을 활용합니다. DenseSplat는 NeRF 프라이어를 사용하여 관찰되지 않은 시점의 간격을 효과적으로 해소하며, 기하학 인지 프리미티브 샘플링 및 가지치기 전략을 구현하여 고품질의 맵을 유지합니다.

- **Technical Details**: DenseSplat는 RGB-D 스트림을 시작으로 카메라 자세와 신경 방사장(field)을 동시에 최적화하는 방식으로 작동합니다. 이후, 암묵적인 방사장으로부터 샘플링하여 가우시안 프리미티브를 초기화하고, 이를 통해 세밀한 맵 재구성과 장면 보간을 수행합니다. 또한, 로컬 루프 폐쇄 감지와 번들 최적화 전략을 구현하여 드리프트 오류를 최소화합니다.

- **Performance Highlights**: 다양한 대규모 데이터셋에서 수행된 실험 결과, DenseSplat는 기존의 최첨단 기술들과 비교하여 추적 및 매핑 성능에서 우수한 결과를 보여줍니다. 특히, 복잡한 환경에서의 성능 저하 없이 실시간으로 고해상도 맵을 생성하는 성능을 입증했습니다. DenseSplat는 SCOT 및 SOTA 방법론과의 비교를 통해 실제 시스템에서의 응용 가능성을 확보하였습니다.



### Pulling Back the Curtain: Unsupervised Adversarial Detection via Contrastive Auxiliary Networks (https://arxiv.org/abs/2502.09110)
- **What's New**: 이 연구에서는 기존의 적대적 공격(Adversarial Attack)에 대한 방어 메커니즘을 개선하기 위한 새로운 방법인 Unsupervised adversarial detection via Contrastive Auxiliary Networks (U-CAN)을 제안합니다. 이 방법은 적대적 예시 없이 보조 네트워크를 통해 특징(feature) 표현에서 적대적 행동을 탐지할 수 있게 합니다. 또한, 제안된 방법은 다양한 공격 유형에 적용 가능하며, 모델의 구조나 파라미터를 변경하지 않고도 사용이 가능합니다.

- **Technical Details**: U-CAN은 ResNet-50, VGG-16, ViT와 같은 다양한 아키텍처에서 적용되며, 중간 레이어에 내장된 보조 네트워크를 통해 추가적인 특징을 생성하여 적대적 입력을 구분합니다. 이 보조 네트워크는 특징을 더 효율적으로 정제하기 위해 프로젝션 레이어와 ArcFace 기반 선형 레이어로 구성되어 있습니다. 이러한 구조는 적대적 공격에 대한 검출 성능을 높이는 데 기여합니다.

- **Performance Highlights**: 실험 결과, U-CAN은 CIFAR-10, Mammals, ImageNet의 여러 데이터셋에서 기존의 비지도적 적대적 탐지 기법들보다 우수한 성능을 보여줍니다. 특히, 네 가지 서로 다른 공격 방법에 대해 더 높은 F1 점수를 기록하며, 전반적으로 딥러닝 시스템의 보안성과 신뢰성을 향상시키는 효과적인 솔루션으로 자리매김하고 있습니다.



### From Visuals to Vocabulary: Establishing Equivalence Between Image and Text Token Through Autoregressive Pre-training in MLLMs (https://arxiv.org/abs/2502.09093)
- **What's New**: 이번 연구에서는 Vision Dynamic Embedding-Guided Pretraining (VDEP)이라는 새로운 하이브리드 학습 패러다임을 제안합니다. 기존의 MLLMs가 단순히 텍스트 입력에서 정보 회복에 중점을 두었던 것과 달리, VDEP는 입력 데이터를 통한 정보 복구 과정으로 다중 모달 정렬을 재정의합니다. 이 방법은 LLava 모델의 아키텍처를 변경하지 않고도 이미지-텍스트 정렬을 크게 향상시키는 효과를 보여줍니다.

- **Technical Details**: VDEP는 시각 인코더 다음에 위치한 MLP로부터 생성된 동적 임베딩을 활용하여 이미지 표현의 숨겨진 상태를 감독합니다. 또한, 이미지 토큰들이 자기회귀 훈련 목표에 통합되어 텍스트와 이미지의 공동 최적화를 가능하게 합니다. 이러한 메커니즘은 이미지와 텍스트 토큰 간의 상대적인 중요성을 동적으로 조정하여 주의 분포의 균형을 이룹니다.

- **Performance Highlights**: VDEP는 여러 벤치마크 데이터셋에서 뛰어난 성능을 발휘하며, RealWorldQA 벤치마크에서 약 5.2%의 성능 향상을 달성했습니다. 실험 결과는 자기회귀 잠재 공간 정렬 전략이 환각 현상(hallucination issues)을 완화하고, 교차 모달 작업에서 모델의 효율성을 크게 향상시킴을 보여줍니다.



### Unsupervised Anomaly Detection on Implicit Shape representations for Sarcopenia Detection (https://arxiv.org/abs/2502.09088)
- **What's New**: 이 논문은 노화 관련 질병인 sarcopenia를 탐지하기 위한 새로운 접근 방식인 암묵적 신경 표현(implicit neural representation, INR)을 사용하여 근육의 형태를 모델링하는 방법을 제안합니다. 기존의 근육량 기준 대신, 저자는 근육의 형태를 기반으로 abnormal(비정상) 근육을 구별하는 비지도 학습(anomaly detection) 방법을 도입합니다. 이 방법은 sarcopenia 근육과 비정상 근육을 효과적으로 식별하면서, 더 나아가 의료진의 진단을 지원할 수 있는 가능성을 보입니다.

- **Technical Details**: 저자는 3D 이미지를 통해 근육의 모양을 모델링하기 위해 INR을 활용하며, 훈련된 모델이 non-sarcopenic 근육으로부터 학습된 모양 사전 정보를 기반으로 reconstruction error(재구성 오류)를 측정합니다. 구체적으로, latent representation(잠재 표현) 학습을 통해 정상 근육과 비정상 근육을 명확히 분리하는 방식을 취합니다. 이 과정에서 Multilayer Perceptron(MLP) 분류기가 3D 좌표와 함께 조건 벡터를 입력으로 받아들여 voxel의 점유 확률을 예측합니다.

- **Performance Highlights**: 실험 결과는 103개의 세분화된 볼륨 데이터셋을 기반으로 하여, sarcopenic 근육의 Dice 재구성 오류가 현저히 낮음을 나타냅니다. 또한, 학습된 잠재 형태 표현의 선형 판별 분석(Linear Discriminant Analysis, LDA) 결과는 비정상적인 근육과 정상적인 근육 그룹 간의 명확한 구분을 보여줍니다. 이 결과는 sarcopenia 선별 과정의 정확성을 높이는 데 기여할 수 있습니다.



### BevSplat: Resolving Height Ambiguity via Feature-Based Gaussian Primitives for Weakly-Supervised Cross-View Localization (https://arxiv.org/abs/2502.09080)
- **What's New**: 이 논문은 약하게 감독된 크로스 뷰 로컬라이제이션(weakly supervised cross-view localization) 문제를 다룹니다. 특히, 지상 카메라의 포즈를 위성 이미지에 비해 추정하는 것을 목표로 하며, 기존 방법의 한계를 극복하기 위해 BevSplat이라는 새로운 방법을 제안합니다. BevSplat은 각 픽셀을 3D Gaussian 형태로 표현하여 고유한 기능과 공간적 특성을 통합하며, 이를 통해 상대적인 포즈 추정의 정확성을 높입니다.

- **Technical Details**: BevSplat은 특징 기반의 3D Gaussian primitives를 사용하여 Bird’s-Eye View (BEV) 생성을 수행합니다. 이 방법은 각 지상 이미지의 픽셀을 의미론적 및 공간적 특성을 지닌 3D Gaussian으로 표현하며, 보이지 않는 렌더링 알고리즘을 통해 BEV feature map로 합성됩니다. 또한, 파노라마 이미지의 도전 과제를 해결하기 위해 icosphere 기반의 감독 전략을 도입하여 고정밀도 깊이 예측을 가능하게 합니다.

- **Performance Highlights**: KITTI와 VIGOR 데이터셋에서 BevSplat의 정확성을 검증한 결과, 본 방법이 기존의 크로스 뷰 로컬라이제이션 기법에 비해 현저히 향상된 성능을 보였습니다. 이는 특히 지상 카메라와 위성 이미지 간의 정확한 정렬을 통해 이루어진 효과로, 여러 로컬라이제이션 시나리오에서도 우수한 결과를 달성했습니다.



### PTZ-Calib: Robust Pan-Tilt-Zoom Camera Calibration (https://arxiv.org/abs/2502.09075)
Comments:
          Accepted by ICRA 2025

- **What's New**: PTZ-Calib는 강력한 2단계 PTZ 카메라 보정 방법으로, 任意의 viewpoint에 대한 카메라 파라미터를 효율적이고 정확하게 추정합니다. 오프라인 및 온라인 두 단계로 구성되어 있으며, PTZ-IBA 알고리즘을 통해 설정된 좌표계 내에서 자동으로 카메라를 보정합니다. 이는 카메라 파라미터 최적화를 통해 지리적 좌표계와 정렬하는 것이 가능합니다.

- **Technical Details**: 우리의 방법은 PTZ 카메라의 기울기, 초점 거리, 왜곡 계수와 같은 카메라 파라미터를 견고하게 평가합니다. 오프라인 단계에서는 정합점(feature points)을 추적하고, PTZ-IBA 알고리즘을 사용하여 지역 좌표계 내에서 자동으로 카메라를 보정합니다. 온라인 단계에서는 이미지 정합과 최적화 기술을 활용하여 새로운 viewpoint에 대한 카메라 파라미터를 보정합니다.

- **Performance Highlights**: 실험 결과, 우리의 방법은 다양한 실제 및 합성 데이터셋에서 기존의 최신 기술을 능가하는 기량을 보였습니다. 특히 스포츠 분야 등록에서 최상의 결과를 이끌어내었으며, 3D 합성 장면을 활용한 검증에서도 일관되게 우수한 성능을 발휘했습니다. 최종적으로, 이 알고리즘은 복잡한 시나리오에서도 넓은 적용 가능성을 보여줍니다.



### StyleBlend: Enhancing Style-Specific Content Creation in Text-to-Image Diffusion Models (https://arxiv.org/abs/2502.09064)
Comments:
          Accepted to Eurographics 2025. Project page: this https URL

- **What's New**: 이 논문은 StyleBlend라는 새로운 방법을 소개하여, 제한된 참조 이미지 세트에서 스타일 표현을 학습하고 적용하여 텍스트 정렬 및 스타일적으로 일관된 콘텐츠 합성을 가능하게 합니다. 기존의 T2I(diffusion 모델)가 어려움을 겪었던 텍스트 비정렬 및 약한 스타일 표현 문제를 해결하기 위해, 스타일을 구성(composition)과 텍스처(texture)라는 두 개의 구성 요소로 분해하고 이를 각각의 합성(branch) 영역에서 효과적으로 혼합합니다.

- **Technical Details**: StyleBlend 방법의 핵심은 두 가지 스타일 구성 요소를 분리하여 학습하고 이들을 서로 연결하는 것입니다. 구성(composition)은 장면의 의미적 구조와 레이아웃 정보를 나타내며, 텍스처(texture)는 세부 사항과 로컬 모습을 강조합니다. 논문에서는 SDEdit 기법을 적용하여 의미 있는 레이아웃 정보를 유지하면서 텍스처 스타일을 학습하고, 이 두 가지를 결합하여 더 높은 품질의 스타일화된 이미지를 생성합니다.

- **Performance Highlights**: 연구 결과, StyleBlend는 기존의 방법들과 비교하여 텍스트 비정렬 및 스타일 표현의 일반적인 문제를 극복하여 더욱 견고한 스타일 특정 콘텐츠 생성을 제공합니다. 질적 및 양적 비교를 통해, 이 방법이 어떻게 향상된 결과를 도출하는지를 명확하게 보여주며, 개인화된 예술 창작 등 다양한 응용 분야에 대한 잠재력을 제시합니다.



### Vision-Language In-Context Learning Driven Few-Shot Visual Inspection Mod (https://arxiv.org/abs/2502.09057)
Comments:
          VISAPP 2025

- **What's New**: 이번 연구에서는 Vision-Language Model (VLM)과 In-Context Learning (ICL)을 활용하여 새로운 제품 이미지의 결함 위치를 감지할 수 있는 일반적인 시각 검사 모델을 제안합니다. 기존의 VLM들이 특정 작업에 대해 훈련되지 않았다는 점을 보완하기 위해, 다양하고 통일된 형식의 결함 및 비결함 제품 이미지 데이터세트를 구성하고 이를 통해 VLM을 미세 조정(fine-tune)하였습니다. 이러한 접근법은 제품에 따라 대량의 훈련 샘플을 수집하거나 모델 재훈련(re-train)을 필요로 하지 않습니다.

- **Technical Details**: 제안된 방법은 VLM과 ICL을 결합하여 특정 제품의 검사를 가능하게 하며, 이를 위해 ViP-LLaVA 모델을 기반으로 하고 있습니다. 이 모델은 시각 프롬프트 인식을 개선하기 위해 LLaVA의 데이터셋을 사용해 미세 조정하였으며, 비정상 상태에 대한 정보를 포함한 데이터세트를 구성하였습니다. 이 과정에서 Euclidean 공간에서의 거리 기반 알고리즘을 통해 고품질의 예제 선택을 가능하게 하였습니다.

- **Performance Highlights**: 제안된 방법은 MVTec AD 데이터세트에서 0.804의 MCC 및 0.950의 F1-score를 달성하며, 이는 단일 샷(one-shot) 방식으로 이루어졌습니다. 이러한 성능은 기존 모델에서 요구되는 재훈련 혹은 하이퍼파라미터 조정을 배제하고도 높은 정확도를 보임을 나타냅니다. 또한, 모델의 설명 가능성을 보장하기 위해 결함 위치의 좌표 정보도 포함하였습니다.



### AIDE: Agentically Improve Visual Language Model with Domain Experts (https://arxiv.org/abs/2502.09051)
Comments:
          6 pages, 4 figures, 2 tables

- **What's New**: 이 논문에서는 시각 언어 모델(Visual Language Models, VLMs)의 성능을 향상시키기 위한 새로운 프레임워크인 AIDE(Agentic Improvement through Domain Experts)를 소개합니다. AIDE는 전문 도메인 모델을 활용하여 VLMs의 개선을 자동화하여, 기존의 더 큰 모델의 의존성을 줄이는 데 중점을 두고 있습니다. 이 프로세스는 정제할 사례를 식별하고, 전문 모델의 분석을 활용하며, 개선된 데이터를 학습 파이프라인에 통합하는 네 단계로 구성되어 있습니다.

- **Technical Details**: AIDE는 VLM의 훈련 데이터를 향상시키기 위해 두 가지 주요 에이전트인 Selector와 Synthesizer를 포함합니다. Selector는 개선이 필요한 사례를 식별하고, 해당 사례에 적합한 전문 도구와 매칭합니다. Synthesizer는 전문가의 출력을 기존 데이터와 통합하여 훈련 예제를 생성하는 과정에서 여러 출처의 정보를 집계하고 잠재적 충돌을 해결하여 더 풍부하고 일관된 응답을 생성합니다.

- **Performance Highlights**: 다양한 벤치마크 실험을 통해 AIDE는 기존의 모델에 비해 성능 개선을 달성하였으며, 특히 MMMU에서 1.2%, MMBench에서 0.77%, MME에서 52%의 개선 효과를 보였습니다. 이 프레임워크는 대형 모델에 대한 접근 없이도 VLMs의 지속적인 개선을 가능하게 하여, 현재의 방법론에서의 중대한 한계를 극복하는 데 기여할 수 있음을 보여줍니다.



### Evolution of Data-driven Single- and Multi-Hazard Susceptibility Mapping and Emergence of Deep Learning Methods (https://arxiv.org/abs/2502.09045)
- **What's New**: 이 논문은 자연재해에 대한 데이터 기반 취약성 맵핑의 발전을 다루고 있습니다. 단일 위험에서 다중 위험으로의 확장이 이루어졌으며, 딥 러닝 방법이 새로운 가능성으로 떠오르고 있습니다. 특히, 다중 위험 취약성 맵링(MHSM)을 위한 데이터 융합 전략의 적용에 대한 비전이 제안됩니다.

- **Technical Details**: 자연재해 취약성 맵은 지역 내 자연재해 발생 가능성과 심각도를 수학적으로 표현하는 것입니다. 이러한 맵은 과거 데이터와 인과 관계 요소를 사용하여 생성되며, 여러 가지 통계 기반 기법과 현대의 딥 러닝 기법이 사용됩니다. 주로 집중되는 방법에는 통계 분석, 다기준 결정 분석, 기계 학습, 딥 러닝 모델이 포함됩니다.

- **Performance Highlights**: 연구 문헌에서 수집된 데이터 기반 접근방식의 발전이 관찰되었으며, 이는 지역 및 특정 자연재해에 대한 사례 연구를 통해 검증되었습니다. 그러나 다중 위험 맵핑에 대한 체계적인 연구는 부족합니다. 본 논문은 단일 위험과 다중 위험 모두에 대한 현대적인 딥 러닝 방법의 발생을 논의하며, 향후 연구 방향에 대한 통찰을 제공합니다.



### Large Images are Gaussians: High-Quality Large Image Representation with Levels of 2D Gaussian Splatting (https://arxiv.org/abs/2502.09039)
Comments:
          Accepted by 39th Annual AAAI Conference on Artificial Intelligence (AAAI 2025). 10 pages, 4 figures

- **What's New**: 이번 연구에서 우리는 Gaussian Splatting (GS)을 기반으로 한 새로운 이미지 표현 기법인 LIG(Large Images are Gaussians)를 소개합니다. LIG는 큰 이미지들을 Gaussian 포인트로 표현하는 방법으로서, 기존 2D Gaussian Splatting (2DGS)의 한계를 극복하고 있습니다. 특히, 많은 Gaussian 포인트를 효과적으로 관리할 수 있는 두 가지 주요 수정 사항을 통해 대용량 이미지를 고품질로 피팅하는 데 성공하였습니다.

- **Technical Details**: LIG는 2DGS 접근 방식을 활용하여 이미지 피팅의 최적화 문제를 다룹니다. 첫 번째로, Gaussian 매개변수를 새로운 2DGS 표현 방식으로 최적화하며, 이는 CUDA 커널을 재구현하여 진행됩니다. 두 번째로, 컴퓨터 그래픽의 Level of Detail (LOD) 개념을 적용하여, 정밀하고 세밀한 구조를 효율적으로 맞출 수 있습니다. 이러한 접근 방식은 많은 수의 Gaussian 포인트의 훈련을 쉽게 만들어 줍니다.

- **Performance Highlights**: 실험 결과 LIG는 의료 이미지와 원격 감지 이미지 등 다양한 대용량 이미지에 대해 우수한 피팅 성능을 보였습니다. 우리의 기술은 텔레메디슨 및 위성 통신과 같은 응용 분야에서 큰 가능성을 보여줍니다. 또한, 여러 대조군과의 비교를 통해 LIG의 품질과 효율성이 기존의 INR 기반 방법들을 초과함을 입증하였습니다.



### Billet Number Recognition Based on Test-Time Adaptation (https://arxiv.org/abs/2502.09026)
- **What's New**: 본 논문에서는 이동하는 강재 슬래브에서 기계 인쇄 또는 수동으로 작성된 슬래브 번호를 실시간으로 인식하는 방법을 제안하고 있습니다. 기존의 scene text recognition 방법은 이미지 왜곡 및 훈련 데이터와 테스트 데이터 간의 분포 차이로 인해 인식 정확도가 낮았으나, 제안된 방법은 test-time adaptation (TTA)와 prior knowledge를 통합하여 이러한 문제를 해결합니다. 이 방법은 훈련 데이터의 필요 없이 테스트 단계에서 모델을 조정할 수 있으며, 슬래브 번호 인식의 정확성을 크게 향상시킵니다.

- **Technical Details**: 제안된 알고리즘의 전체 프레임워크는 DB 네트워크를 텍스트 탐지 네트워크로 사용하고 SVTR 네트워크를 텍스트 인식 네트워크로 활용하여 초기 인식 결과를 얻습니다. 테스트 단계에서 entropy 최소화를 통해 모델 매개변수를 조정하며, prior knowledge를 이용하여 CTC 알고리즘을 통해 인식 결과를 수정합니다. 이는 damaged characters의 인식을 개선하는 데 도움을 줍니다.

- **Performance Highlights**: 실제 데이터셋에서 슬래브 번호 인식에 대한 실험 결과는 제안된 방법의 유효성을 입증합니다. 기계 인쇄된 슬래브 번호와 수기 작성된 슬래브 번호 모두에서 평가 지표가 크게 향상되었습니다. 이러한 개선은 공장 환경의 복잡성과 이동하는 슬래브의 온라인 인식에 의한 성능 저하 문제를 극복하는 데 기여합니다.



### EventSTR: A Benchmark Dataset and Baselines for Event Stream based Scene Text Recognition (https://arxiv.org/abs/2502.09020)
Comments:
          In Peer Review

- **What's New**: 이번 논문에서는 기존 RGB 카메라를 기반으로 한 Scene Text Recognition (STR) 알고리즘의 한계를 극복하기 위해, 바이오 영감을 받은 이벤트 카메라를 활용하여 EventSTR이라는 대규모 벤치마크 데이터세트를 제안합니다. 이 데이터세트에는 9,928개의 고화질 샘플이 포함되어 있으며, 중국어 및 영어 문자 인식을 지원합니다. 또한, 새로운 이벤트 기반의 STR 프레임워크인 SimC-ESTR를 제안하고, 이를 통해 미래의 연구에 대한 많은 가능성을 열어줍니다.

- **Technical Details**: SimC-ESTR 프레임워크는 이벤트 스트림에서 이벤트 피쳐를 추출하여 Q-former 모듈을 통해 토큰으로 변환합니다. 기존의 비전-기반 네트워크와 대형 언어 모델(LLM)을 통합하여, 메모리 메커니즘을 통해 시각적 특징을 증강하고, 비슷한 문자 오류를 교정하는 기능도 지원합니다. 이러한 접근 방식은 저조도, 복잡한 배경, 모션 블러와 같은 도전 과제를 해결하는 데 중점을 두고 개발되었습니다.

- **Performance Highlights**: 새롭게 제안된 EventSTR 데이터세트와 추가적인 두 가지 STR 시뮬레이션 데이터세트를 통해, 제안된 모델의 효과성을 입증하기 위한 광범위한 실험이 이루어졌습니다. SimC-ESTR는 기존의 알고리즘에 비해 더 높은 인식 정확도를 달성하며, 대량의 텍스트 인식 및 문서 처리 작업에서의 가능성이 확장될 수 있음을 보여주고 있습니다.



### Residual Transformer Fusion Network for Salt and Pepper Image Denoising (https://arxiv.org/abs/2502.09000)
Comments:
          8 pages, 17 figures

- **What's New**: 이 논문에서는 Residual Transformer Fusion Network (RTF-Net)라는 새로운 이미지 복원 아키텍처를 제안합니다. 이 아키텍처는 Convolutional Vision Transformer (CvT)와 Residual Networks (ResNet)를 결합하여 효과적인 노이즈 제거를 목표로 하며, NSN (Noise Suppression Network)과 SEN (Structure Enhancement Network)의 두 부분으로 나뉩니다. 이를 통해 기존의 이미지 노이즈 제거 방법에서 요구되는 노이즈에 대한 사전 지식 없이도Cleaner 이미지 생성이 가능합니다.

- **Technical Details**: RTF-Net의 Noise Suppression Network는 Residual Block을 통해 이미지를 처리하여 노이즈 맵을 학습합니다. 두 번째 단계에서는 Structure Enhancement Network가 CvT를 이용하여 노이즈가 제거된 이미지에 필요한 세부 사항을 학습하여 최종적으로 클린 이미지를 생성합니다. 이 아키텍처는 salt and pepper 노이즈 모델링을 포함하여 다양한 노이즈 유형을 효과적으로 처리할 수 있도록 설계되었습니다.

- **Performance Highlights**: 제안된 RTF-Net 모델은 DIV2K Training Set을 사용하여 훈련되었으며, 다양한 이미지(Lena, Bridge, Pepper, BSD300)에서 PSNR(peak signal-to-noise ratio) 성능을 평가했습니다. 논문에 따르면, 대부분의 경우 제안한 접근 방식이 기존의 여러 방법들보다 우수한 성능을 보였으며, 특히 Pepper 이미지의 30% 노이즈 수준에서는 NLSF-CNN이 우세했습니다.



### Hierarchical Vision Transformer with Prototypes for Interpretable Medical Image Classification (https://arxiv.org/abs/2502.08997)
- **What's New**: 이번 연구에서는 HierViT라는 새로운 Vision Transformer 모델을 제안합니다. HierViT는 본질적으로 해석 가능하며, 사람의 사고 방식에 맞춰 추론을 조정하는 독특한 구조를 가지고 있습니다. 이 모델은 예측을 위해 계층적 구조를 사용하여 도메인 특정(features) 특성을 처리함으로써, 기존의 Vision Transformer의 한계를 해결합니다.

- **Technical Details**: HierViT는 12 layer의 ViT 인코더를 기반으로 하며, ImageNet-1K에서 사전 훈련된 가중치를 사용합니다. 두 개의 분기를 통해 추출된 특성을 처리하며, 첫 번째 분기는 계층적 분류기로 동작하여 목표 분류를 위한 특성을 매핑합니다. 또한, 각 속성의 점수는 개별 트랜스포머 인코더와 선형 레이어를 통해 계산됩니다.

- **Performance Highlights**: HierViT는 두 가지 의료 벤치마크 데이터셋인 LIDC-IDRI와 derm7pt에서 각각 우수하고 비교 가능한 예측 성능을 달성했습니다. 또한, 이 모델은 사람의 정의된 기준과 일치하는 설명을 제공함으로써 사용자의 신뢰를 높이고, 단일 이미지 예측에서도 높은 설명성을 유지합니다.



### Latents of latents to delineate pixels: hybrid Matryoshka autoencoder-to-U-Net pairing for segmenting large medical images in GPU-poor and low-data regimes (https://arxiv.org/abs/2502.08988)
- **What's New**: 이 논문에서는 의료 영상 처리에서 중요한 정보를 보존하면서 픽셀 레벨 과제를 위한 충분한 픽셀 기하학을 유지하는 저랭크 Matryoshka 프로젝션과 하이브리드 세그멘테이션 아키텍처를 제안합니다. Matryoshka Autoencoder (MatAE-U-Net)라는 새로운 구조를 통해 계층적 인코딩과 공간적 복원 기능을 결합하여 정확도와 일반화 능력을 향상시킵니다. 이 시스템은 심장 초음파 영상의 왼쪽 심실을 분할하는 문제에 적용됩니다.

- **Technical Details**: Stanford EchoNet-D 데이터세트를 활용하여 1,000개의 표준화된 비디오-마스크 쌍을 사용하며, 각 비디오는 112x112 픽셀로 크기가 조정된 심장 초음파 프레임 시퀀스로 구성됩니다. MatAE-UNet 모델은 평균 IoU(Intersection over Union) 77.68%, 평균 픽셀 정확도 97.46%, 주사위 계수 86.91%를 달성하였으며, 이는 기준 모델인 Vanilla U-Net의 성능을 초과합니다. 모델은 skip connections와 다중 스케일 기능 추출을 활용하여 성능을 극대화합니다.

- **Performance Highlights**: MatAE-UNet 모델은 Vanilla U-Net보다 평균 IoU와 주사위 계수 측면에서 뛰어난 성과를 보이며, 이를 통해 임상적 세부정보를 더욱 잘 보존합니다. 연구 결과는 Matryoshka 아키텍처가 저대비 영상 문제, 특히 심장 초음파 분석에서 효과적으로 활용될 수 있는 가능성을 보여줍니다. 전체적으로 이 접근법은 의료 이미지 세그멘테이션의 정확도와 일반화를 향상시키는 잠재력이 있습니다.



### Text-driven 3D Human Generation via Contrastive Preference Optimization (https://arxiv.org/abs/2502.08977)
Comments:
          8

- **What's New**: 최근 Score Distillation Sampling (SDS) 기술 발전은 텍스트 설명으로부터 3D 인간 모델 생성의 정확성을 향상시켰습니다. 그러나 기존 방법들은 여전히 긴 텍스트 입력에 대한 3D 모델의 올바른 정렬에 어려움을 겪고 있습니다. 이에 우리는 인간 수준의 선호 모델을 통해 긍정적 및 부정적 프롬프트를 보조하여 SDS의 정렬 능력을 개선하는 새로운 프레임워크를 제안합니다.

- **Technical Details**: 제안된 프레임워크는 선호 최적화 모듈을 설계하여 여러 선호 모델을 통합합니다. 이를 통해 긴 텍스트의 다양한 세그먼트에서 의미 정보를 포괄적으로 포착할 수 있습니다. 또한, 부정선호 최적화 모듈을 도입하여 정체적이고 동적 부정 프롬프트를 활용하여 무관한 세부 사항의 과도한 최적화를 완화하도록 설계되었습니다.

- **Performance Highlights**: 광범위한 실험을 통해 우리의 방법이 텍스트 정렬 정확도 측면에서 최신 접근 방법을 초월하며, 특히 긴 텍스트 입력에 대한 질감의 사실감과 시각적 정렬을 크게 향상시켰음을 보여주었습니다. 이로 인해 생성된 3D 모델은 기존 방법보다 의미적으로 더 정확하고 시각적으로 더 일관된 결과를 도출했습니다.



### Topo2Seq: Enhanced Topology Reasoning via Topology Sequence Learning (https://arxiv.org/abs/2502.08974)
- **What's New**: 이 논문에서 소개하는 Topo2Seq는 자율주행에서의 경로 인식을 개선하는 새로운 방법을 제공합니다. 기존의 HD 맵에 의존하지 않고도 직관적인 도로 흐름을 실시간으로 인식하고 다양한 도로 기하학적 위치와 토폴로지를 추출할 수 있는 기술에 초점을 맞추고 있습니다. 또한, 이 접근법은 두 개의 디코더를 사용하여 효율적인 경량의 인퍼런스 속도를 유지하며, 학습 과정에서만 새로운 토폴로지 시퀀스 디코더를 도입합니다.

- **Technical Details**: Topo2Seq는 lane segment decoder와 topology sequence decoder로 구성된 이중 디코더 아키텍처를 활용합니다. 랜덤화된 순서로 prompt-to-sequence 학습을 통해 lane segment decoder가 예측한 lane graph에서 비순차적인 키 포인트를 추출하여, 이를 topology sequence decoder의 프롬프트 디자인에 투입하여 전체 경로를 재구성합니다. 이를 통해 lane segment decoder는 topology sequence decoder로부터 강력한 장거리 인식(long-range perception)과 정확한 토폴로지 추론(topological reasoning)을 학습하게 됩니다.

- **Performance Highlights**: OpenLane-V2 데이터세트에서 실험을 실시한 결과, Topo2Seq는 토폴로지 추론에서 최첨단 성능을 달성하였습니다. 이 방법은 기존 방법들에 비해 경로 예측과 계획에서 훨씬 더 높은 정밀도를 제공하며, 특히 자율주행의 고급 수준인 L4 및 L5를 지원할 수 있는 토대가 됩니다. 이 연구는 자율주행 시스템의 안전성과 효율성을 높이는 데 기여할 것으로 기대됩니다.



### Towards Understanding Why Data Augmentation Improves Generalization (https://arxiv.org/abs/2502.08940)
- **What's New**: 이 논문에서는 기존의 데이터 증강 기법들이 일반화 성능을 어떻게 향상시키는지에 대한 통합 이론적 프레임워크를 제시합니다. 구체적으로, 부분 의미적 특징 제거(Partial Semantic Feature Removal)와 특징 혼합(Feature Mixing)의 두 가지 주요 효과를 통해 모델이 더 다양한 특징을 학습하도록 유도하고, 일반화 성능을 개선하는 방법을 설명합니다. 이를 통해 서로 다른 증강 기법들이 어떻게 공통된 메커니즘을 공유하는지를 밝혀내고자 합니다.

- **Technical Details**: 제안된 이론적 프레임워크는 3층 컨볼루션 신경망(CNN)을 예시로 사용하여 데이터 증강 기법들이 어떻게 모델의 일반화를 촉진하는지를 분석합니다. 부분 의미적 특징 제거는 입력 데이터의 특정 영역이나 특징을 버려서 모델이 남은 정보에서 다양한 특징을 학습하게 합니다. 반면 특징 혼합은 두 개의 이미지를 조합하여 원래의 의미적 특징을 줄이고 노이즈를 추가함으로써 훈련 복잡성을 증가시킵니다.

- **Performance Highlights**: 저자들은 다양한 벤치마크 데이터셋을 통해 제안된 이론적 발견을 검증하며, 데이터 증강 기법과 모델 성능 간의 상호 작용에 대한 새로운 통찰을 제공합니다. 전반적으로, 본 연구는 데이터 증강의 근본 원리를 밝히고 이를 통해 모델의 일반화 성능을 개선하는 데 기여함으로써, 보다 효과적인 데이터 증강 전략 설계에 대한 지침을 제공합니다.



### Dynamic watermarks in images generated by diffusion models (https://arxiv.org/abs/2502.08927)
- **What's New**: 이번 논문에서는 고충실도 텍스트-투-이미지(diffusion) 모델의 일반화에 따른 지적 재산권(İP) 보호 및 합성 미디어 악용 문제를 해결하기 위한 새로운 다단계 워터마킹 프레임워크를 제안합니다. 해당 워터마킹 기술은 고정된 QR 코드 워터마크와 인간이 인식할 수 없는 동적 워터마크로 구성되어 있으며, 두 가지를 통해 생성된 이미지를 추적하고 저작권을 증명할 수 있습니다. 이를 통해 AI 생성 콘텐츠의 보안 분야에서 지적 재산 보호 및 악용 방지로 발전할 수 있는 정보를 제공합니다.

- **Technical Details**: 제안된 워터마킹 기법은 두 개의 주요 분기로 나뉘며, 첫 번째는 diffusion 모델의 학습된 노이즈 분포에 고정된 QR 코드 워터마크를 삽입하는 것입니다. 두 번째는 생성된 이미지에 동적 워터마크를 삽입하여 불가시성과 공격에 대한 저항력을 균형 있게 유지하고, 이미지의 품질을 최소한으로 손상시키는 것입니다. 이 과정에서 구조적 유사성 지수(SSIM)와 코사인 유사도(cosine similarity)를 활용하여 워터마크의 모양과 색상을 동적으로 조정합니다.

- **Performance Highlights**: 우리의 방법은 워터마크 분류를 통해 신뢰할 수 있는 출처 확인을 가능하게 하며, 동적 워터마크가 콘텐츠 특성에 맞게 조정되더라도 효과적으로 작동합니다. 다양한 공격 시나리오에 대한 엄격한 테스트를 거쳐, 높은 강인성을 유지하면서 이미지 품질에 미치는 영향이 최소화된다는 것을 보여주었습니다. 본 연구는 AI 생성 콘텐츠의 보안을 강화하고, 지적 재산권 보호에 있어 중요한 진전을 이룩했습니다.



### PathFinder: A Multi-Modal Multi-Agent System for Medical Diagnostic Decision-Making Applied to Histopathology (https://arxiv.org/abs/2502.08916)
- **What's New**: 이번 논문에서는 PathFinder라는 다중 모달 및 다중 에이전트 시스템을 제안합니다. 이 시스템은 4개의 AI 에이전트(Triage, Navigation, Description, Diagnosis)를 통합하여 병리 전문가의 의사 결정 과정을 모방합니다. 특히, 각 에이전트는 WSIs(Whole Slide Images)를 효과적으로 탐색하고 자연어 설명을 제공하여 포괄적인 진단을 수행합니다. 이러한 접근 방식은 기존의 전통적인 AI 방법이 가지는 한계를 극복하는 데 목적을 두고 있습니다.

- **Technical Details**: PathFinder는 Triage Agent가 WSI를 양성 또는 위험으로 분류한 후 위험으로 판단될 경우 Navigation 및 Description Agents가 중요 영역을 반복적으로 조사하여 중요도 맵과 설명을 생성합니다. 이 과정에서 각 에이전트는 긴밀하게 협력하여 진단 정보를 수집하고, Diagnosis Agent가 최종 진단 평가를 수행합니다. 실험 결과, PathFinder는 기존의 최고 성능을 보여준 방법들보다 8% 높은 정확도로 피부 흑색종 진단을 가능하게 하였습니다.

- **Performance Highlights**: PathFinder는 병리학자와의 질적 분석에서 Description Agent의 출력 품질이 GPT-4o와 비교될 만큼 우수하다는 것을 보여줍니다. 이 시스템은 또한 병리학자의 평균 성능을 9% 초과하여 새로운 기록을 세우며, 병리 진단에서 효율적이고 정확하며 해석 가능한 AI 지원 진단을 실현할 능력을 가지고 있습니다. 논문에서는 데이터, 코드 및 모델을 제공하며, PathFinder의 기초와 성과를 자세히 설명합니다.



### Diffusion Models Through a Global Lens: Are They Culturally Inclusive? (https://arxiv.org/abs/2502.08914)
Comments:
          17 pages, 17 figures, 3 tables

- **What's New**: 최근 텍스트에서 이미지를 생성하는 diffusion 모델은 문화적 뉘앙스를 정확하게 표현하는 데 한계가 있음을 보여줍니다. 이 연구에서는 CultDiff 벤치마크를 도입하여 최신 diffusion 모델들이 10개국의 문화적으로 특화된 이미지를 생성할 수 있는 능력을 평가합니다. 결과적으로, 이러한 모델들이 고해상도의 이미지 생성에서 문화적 특성을 제대로 반영하지 못하는 경우가 많음을 발견했습니다.

- **Technical Details**: 연구에서는 T2I (Text-to-Image) diffusion 모델의 성능을 평가하기 위해 문화에 따라 분류된 새로운 데이터셋인 CultDiff를 개발했습니다. 이 데이터셋은 건축물, 의상, 음식과 같은 다양한 문화적 아이템을 아우르며, 자원 수준이 높은 문화와 낮은 문화의 이미지를 생성하는 모델의 능력을 분석합니다. 또한, 인간 평가를 통해 개발된 neural 기반의 이미지 유사성 메트릭인 CultDiff-S를 도입하여 생성 이미지의 문화적 관련성을 예측합니다.

- **Performance Highlights**: 연구 결과, 최신 diffusion 모델들이 특정 문화에서 필요한 대표적인 개념이나 예술품을 생성하는 데 실패하고 있는 것을 알 수 있었습니다. 특히 자원이 부족한 문화에 대한 표현이 미비하여 이들에 대한 공정한 representation이 이루어지지 않고 있음을 강조합니다. 이러한 결과는 생성 AI 시스템의 더 나은 포용성과 공정한 데이터셋 표현의 필요성을 제기하고 있습니다.



### DiffoRA: Enabling Parameter-Efficient LLM Fine-Tuning via Differential Low-Rank Matrix Adaptation (https://arxiv.org/abs/2502.08905)
- **What's New**: 이번 논문에서는 새로운 PEFT(파라미터 효율적 미세 조정) 방식인 DiffoRA를 제안합니다. DiffoRA는 장애물 역할하는 기존의 사용된 LoRA를 발전시켜 모듈별로 조정할 수 있는 접근법을 제공합니다. 특히, 각 모듈의 적합성과 중요성을 평가하는 Differentiable Adaptation Matrix(DAM)를 중심으로 구성됩니다.

- **Technical Details**: DiffoRA는 기존 LoRA의 저Rank(matrix) 분해 방식에 대한 한계를 극복하는 데 중점을 두었습니다. 기존 방법이 네트워크의 모든 모듈에 동등하게 적용된 저Rank 매트릭스를 사용하는 반면, DiffoRA는 모듈별로 적응적으로 적용될 수 있도록 설계되었습니다. 실험적으로, DAM은 모델의 수렴 속도와 일반화 능력에 긍정적인 영향을 미치는 것으로 나타났습니다.

- **Performance Highlights**: DiffoRA는 여러 기준선을 능가하는 성능을 보여주었습니다. 예를 들어, CoLA 작업에서 DiffoRA는 최신 기술 대비 0.81% 더 높은 모델 정확도를 달성했습니다. 종합적으로, DiffoRA는 두 개의 주요 벤치마크에서 지속적으로 우수한 성능을 나타내어 기존의 PEFT 방법들에 비해 더 높은 효율성을 입증하였습니다.



### CoL3D: Collaborative Learning of Single-view Depth and Camera Intrinsics for Metric 3D Shape Recovery (https://arxiv.org/abs/2502.08902)
Comments:
          Accepted at ICRA 2025

- **What's New**: 이번 연구에서는 단일 이미지로부터 메트릭 3D 형상을 복구하기 위해 깊이(depth)와 카메라 내부 파라미터(camera intrinsics) 간의 상호 관계를 이론적으로 입증합니다. 새로운 협업 학습 프레임워크인 CoL3D를 제안하여 깊이와 카메라 내부 파라미터를 동시에 추정함으로써 로봇의 공간 인식 능력을 향상시킵니다. 이 프레임워크는 공통 인코더-디코더 네트워크를 활용하여 깊이 지도와 카메라 내부 파라미터를 예측합니다.

- **Technical Details**: CoL3D는 카메라 보정과 3D 형상 회복을 위한 두 가지 주요 요소로 구성됩니다. 첫째, 잔여 학습(residual learning)에서 영감을 받아 카메라 내부 파라미터를 위한 카노니컬 인시던스 필드(canonical incidence field)를 도입합니다. 둘째, 포인트 클라우드 공간에서 형상 유사성 측정 손실(shape similarity measurement loss)을 설계하여 3D 형상의 품질을 향상시킵니다.

- **Performance Highlights**: CoL3D는 다양한 벤치마크 데이터셋에서 깊이 추정과 카메라 보정 작업 모두에서 주목할만한 성능을 보여줍니다. NYU-Depth-v2 및 KITTI 데이터셋에서 메트릭 깊이 추정 방법 중 최첨단 성능을 기록했으며, Google Street View 및 Taskonomy 데이터셋에서도 카메라 보정 작업에서 우수한 성능을 달성했습니다. 이러한 성과는 로봇의 인식 능력을 위한 퀄리티 높은 3D 형상 재구성을 가능하게 합니다.



### ShapeLib: designing a library of procedural 3D shape abstractions with Large Language Models (https://arxiv.org/abs/2502.08884)
- **What's New**: 이 논문은 ShapeLib을 제안하며, 이는 최신 대형 언어 모델(LLM)의 선험적 지식을 활용하여 3D 형태 추상화 함수를 디자인하는 최초의 방법입니다. ShapeLib은 사용자가 제공하는 텍스트 설명과 샘플 형태로 입력된 디자인 의도를 기반으로 절차적 추상을 발견합니다. 이 시스템은 형상을 아우르는 라이브러리를 구성하여, 차별화된 매개변수로 다양한 형태 변형을 실현할 수 있도록 합니다. 이는 기존 방법들과 비교했을 때 명확한 이점을 보여줍니다.

- **Technical Details**: ShapeLib은 입력으로 자연어로 된 함수 설명과 샘플 형태를 받아들이며, 이 두 가지 모달리티를 통해 절차적 모델 설계를 지원합니다. 먼저, LLM을 사용하여 함수 설명에 조건부로 라이브러리 인터페이스를 설계하고, 다음으로 이러한 함수를 적용하여 샘플 형태를 설명하는 방안을 제안합니다. 이 과정에서 최종적으로 기하학적 분석을 통해 제안된 함수 구현을 검증합니다. 또한, 인식 네트워크를 통해 입력 형태를 라이브러리 기능을 사용하여 설명하는 프로그램을 추론하는 방법을 학습합니다.

- **Performance Highlights**: ShapeLib은 다양한 형태 카테고리에서 절차적 함수 라이브러리를 설계하는 데 활용되며, 발견된 함수들은 자연어 설명에 따른 상향식 의미에 맞춰져 있습니다. 실험 결과, 이 라이브러리는 시드 세트를 넘어서는 일반화, 해석 가능성 및 신뢰성을 유지하는 중요한 이점을 나타내며, 결과적으로 명확한 형상 변형을 제공합니다. 이 연구는 ShapeLib이 기존 접근법에 비해 절차적 표현의 이점을 더 잘 실현한다는 것을 보여줍니다.



### Survey on Single-Image Reflection Removal using Deep Learning Techniques (https://arxiv.org/abs/2502.08836)
- **What's New**: 이 논문은 단일 이미지 반사 제거(SIRR) 문제를 해결하기 위한 깊이 있는 학습 방법을 활용한 최근 연구 동향을 포괄적으로 검토합니다. 기존의 기술 기반 반사 제거 방법들은 일반화에 한계를 보였으며, 이로 인해 학습 기반 방법으로의 전환이 필요했습니다. 본 연구는 SIRR 분야의 가장 최근 발전 사항을 정리하며, 특정 학술 회의와 저널에 중점을 두어 기존 연구보다 훨씬 포괄적인 문헌 검토를 제공하고 있습니다.

- **Technical Details**: SIRR은 본질적으로 ill-posed 문제이며, 이를 해결하기 위해 여러 수학적 가설이 제시되었습니다. 기존의 가정에서는 이미지가 전송 레이어와 반사 레이어의 선형 조합으로 표현됩니다. 그러나 이러한 가설은 실제 반사가 복잡하고 카메라와의 거리, 조명 조건 등의 요인에 따라 달라지기 때문에, 최근 연구에서는 딥 러닝을 통한 비선형 접근법이 도입되었습니다. 데이터 기반의 심층 학습 모델은 라벨이 지정된 대규모 데이터셋에서 학습하여 다양한 시나리오를 효과적으로 처리할 수 있습니다.

- **Performance Highlights**: 논문에서 다룬 문헌 검토는 2017년부터 2025년까지의 SIRR 관련 연구를 포괄하며, 한편으로는 지식의 체계적 확장을 목표로 합니다. 28개의 주요 연구 논문을 분석하여 이 분야의 최신 동향과 향후 연구 기회를 식별하였습니다. 이러한 성과들은 데이터 기반의 딥 러닝 기술이 기존의 전통적 방법들보다 더 효과적으로 SIRR 문제를 해결할 수 있는 잠재력을 지니고 있음을 강조하고 있습니다.



### $\mathsf{CSMAE~}$:~Cataract Surgical Masked Autoencoder (MAE) based Pre-training (https://arxiv.org/abs/2502.08822)
Comments:
          5 pages, Accepted to IEEE International Symposium on Biomedical Imaging (ISBI 2025)

- **What's New**: 이 연구에서는 CSMAE, 즉 Masked Autoencoder (MAE) 기반의 비디오 분석 기법을 소개합니다. 주요 초점은 백내장 수술 비디오에서 효과적인 특징 학습을 통한 단계 인식(step recognition)을 개선하는 것입니다. 고유한 토큰 선택 방식으로 스페이셜-템포럴(spatiotemporal) 중요성을 기준으로 마스킹된 토큰을 선정하여 기존의 방법보다 더 나은 효율성을 얻었습니다.

- **Technical Details**: 이 모델은 비디오 입력을 N개의 토큰으로 분할하고, Multi-Head Attention (MHA) 메커니즘을 통해 토큰 선택을 최적화합니다. ViT을 사용하는 인코더는 선택된 토큰을 처리하여 잠재 표현(latent representation)을 생성하며, 디코더는 이 정보를 사용하여 마스킹된 비디오 프레임을 재구성합니다. 손실 함수로는 Mean Squared Error (MSE)와 Gradient-Following 알고리즘을 활용하여 토큰 선택 기법을 훈련합니다.

- **Performance Highlights**: CSMAE 모델은 두 가지의 백내장 수술 비디오 데이터셋인 D99와 Cataract-101에서 단계 인식 과제를 통해 검증되었으며, 현재의 최고 성능을 경신하는 결과를 보였습니다. 이는 기존의 self-supervised pretraining 및 adapter 기반의 전이 학습 방법들과 비교해 유의미하게 개선된 성능을 나타내며, 향후 수술 비디오 분석 연구에 중요한 이정표가 됩니다.



### DejAIvu: Identifying and Explaining AI Art on the Web in Real-Time with Saliency Maps (https://arxiv.org/abs/2502.08821)
Comments:
          5 pages, 3 figures, submitted to IJCAI 2025 demo track

- **What's New**: DejAIvu는 사용자가 웹을 탐색하는 동안 AI 생성 이미지에 대한 실시간 탐지를 제공하며, saliency 기반 설명 가능성을 결합한 Chrome 웹 확장입니다. 이 도구는 ONNX 최적화된 딥러닝 모델을 사용하여 Google Images와 같은 웹사이트에서 자동으로 이미지를 분석하고 AI 관련 아티팩트를 강조하는 saliency 히트맵을 오버레이합니다. 사용자는 이 확장을 통해 AI 이미지의 투명성과 해석 가능성을 보장받을 수 있습니다.

- **Technical Details**: DejAIvu는 사용자가 웹페이지에서 이미지를 탐지하여 ONNX 최적화된 AI 모델을 통해 분류한 다음, AI 특정 아티팩트를 강조하는 saliency 맵을 생성합니다. 내부적으로는 NVIDIA RTX 6000 Ada GPU를 활용하여 모델을 훈련하고, 270,000개 이상의 인간 및 AI 생성 아트워크로 구성된 편집된 데이터셋에서 학습하였습니다. 입력 이미지에 대해 256×256 픽셀로 크기를 조정하고 정규화한 후, 모델에 공급하여 정확도를 높이기 위한 다양한 데이터 증강 기법을 적용합니다.

- **Performance Highlights**: DejAIvu는 ResNet50 모델을 활용하여 97.1%의 높은 정확도를 달성하며 90.6MB의 합리적인 파일 크기로 운영됩니다. ONNX.js를 활용한 웹 기반 추론을 통해 평균적으로 이미지당 약 35ms의 레이턴시 감소를 이루어냈으며, 이는 실시간 성능에 큰 개선을 의미합니다. 다양한 아키텍처의 성능 비교에서 DejAIvu는 정확도와 실시간 효율성을 모두 갖춘 최적의 도구로 평가받고 있습니다.



### Measuring Anxiety Levels with Head Motion Patterns in Severe Depression Population (https://arxiv.org/abs/2502.08813)
Comments:
          19th IEEE International Conference on Automatic Face and Gesture Recognition (FG), 2025

- **What's New**: 이 연구는 우울증 환자의 비디오 기록 인터뷰에서 머리 움직임을 분석하여 불안의 심각성을 정량화하는 새로운 비침투성 방법을 제안합니다. CALYPSO 우울증 데이터 세트를 사용해 머리 운동 특성을 추출하고 회귀 분석(regression analysis)을 통해 임상적으로 평가된 불안 수준을 예측했습니다. 이 접근방식은 불안이 우울증에서 차지하는 역할을 더 잘 이해하고 개인 맞춤형 치료 전략을 향상하는 데 기여할 수 있습니다.

- **Technical Details**: 이 연구는 비디오 데이터를 통해 캡처한 머리 포즈를 사용하여 3D 공간에서 머리 방향을 추적합니다. Euler 각도를 사용하여 머리의 세 가지 주요 회전 축인 피치(pitch), 요(yaw), 롤(roll)을 계산하였습니다. 최종적으로 단순 선형 회귀 모델을 훈련시켜 선택된 특성으로 신체 불안을 예측합니다. 이러한 방법은 기존의 비침투성 지표들을 활용하여 향상된 진단 정확성과 치료 전략을 지원합니다.

- **Performance Highlights**: 우리 연구에서 제안한 방법은 머리 움직임 패턴을 기반으로 불안의 심각성을 예측하는 데 있어 0.35의 평균 절대 오차(MAE)를 달성하였습니다. 이는 불안의 역할을 이해하는 데 도움을 주고, 심리 및 신체적 증상에 대한 보다 객관적인 평가를 가능하게 합니다. 우리의 접근법은 다양한 환자 집단과 비임상적 관찰에서도 적용될 수 있어 실용성을 더합니다.



### SB-Bench: Stereotype Bias Benchmark for Large Multimodal Models (https://arxiv.org/abs/2502.08779)
- **What's New**: 최근에 제안된 Stereotype Bias Benchmark (SB-bench)는 대형 다중 모달 모델(Large Multimodal Models, LMMs)의 고정관념 편향을 평가하기 위한 포괄적인 프레임워크입니다. 기존의 데이터셋과 달리, SB-bench는 비합성(non-synthetic) 이미지를 사용하여 실제 세계의 시각적 맥락에서 고정관념 편향을 평가할 수 있도록 설계되었습니다. 이 벤치마크는 9개의 다양한 범주와 60개의 하위 범주를 포함하여 사회적 편향에 대한 방대한 계층적 분류체계를 제공합니다.

- **Technical Details**: SB-bench는 LMMs의 고정관념 편향을 평가하기 위해 철저히 선별된 시각적 시나리오를 사용합니다. 이를 통해 모델이 시각적 고정관념을 초래하는 편향을 이유 있게 분석할 수 있도록 돕습니다. 이 벤치마크는 75,00개의 비합성 시각적 샘플을 특징으로 하며, 더 많은 평가 질문과 편향 영역을 포함하여 향후 다국어 LMMs의 평가를 위한 보다 표준화된 프레임워크를 제공합니다.

- **Performance Highlights**: SB-bench는 최첨단의 오픈소스 및 비공식 LMMs를 막론하고 철저한 테스트를 통해 이 모델들의 고정관념 편향을 평가합니다. 벤치마크는 AI 시스템의 공정성을 증진하고 유해한 편향을 줄이기 위한 중요한 진전을 나타냅니다. 이로 인해 연구자들은 편향 평가를 더 효과적으로 수행할 수 있으며, AI 모델의 고정관념 경감에 기여하는 기반을 마련합니다.



### Exploring Test Time Adaptation for Subcortical Segmentation of the Fetal Brain in 3D Ultrasound (https://arxiv.org/abs/2502.08774)
Comments:
          5 pages, 5 figures

- **What's New**: 이번 연구에서는 초기 이미징 작업에서 종종 성능 저하를 일으키는 도메인 변화(domain shift)를 극복하기 위해 테스트 시간 적응(test-time adaptation, TTA) 기법을 활용하여 모델 성능을 개선할 수 있음을 보여줍니다. 특히, 기존의 TTA 방법인 Test Entropy Minimisation(TENT)을 수정하여 초음파(ultrasound, US) 영상에 보다 적합한 새로운 TTA 방법(EntropyKL)을 제안하고, 이를 통해 각 피질 아래 지역의 예상 볼륨에 대한 표준해부학적 아틀라스를 사전 정보로 포함시킵니다.

- **Technical Details**: 연구에서는 사전 훈련된 소스 모델 f(𝑿s,𝒀s;ϕ)를 사용해 TTA 접근법을 사용하여 목표 데이터(Dt)에서 성능 극대화를 달성하는 방법을 설명합니다. 제안된 EntropyKL 메소드는 US 아틀라스를 사용하여 예측되는 각 피질 아래 영역의 볼륨에 대한 정보를 포함하고 있으며, 다양한 도메인 이동에서의 성능 향상을 통한 벤치마킹 과정도 포함되어 있습니다. 또한, 나이브한 손실 최소화 기법의 위험성을 줄이기 위해 배치 정규화(batch normalization) 레이어만 최적화하는 기법을 도입했습니다.

- **Performance Highlights**: 모델을 각종 시뮬레이션된 도메인 변화, 실제 도메인 변화 및 임신 주에 따른 도메인 변화에 대해 평가한 결과, EntropyKL 접근 방식이 모든 TTA 접근법 중에서 최고의 성능을 보여주었습니다. 이 연구는 자동화된 태아 뇌 발달 모니터링을 위한 강력한 도구로 자리매김할 수 있는 기술의 발전을 제안합니다. 코드 또한 제공합니다.



### Cluster and Predict Latents Patches for Improved Masked Image Modeling (https://arxiv.org/abs/2502.08769)
Comments:
          13 pages, 7 figures, submitted to TMLR

- **What's New**: 이 논문에서는 Masked Image Modeling (MIM) 분야에서의 새로운 접근법인 CAPI를 소개하며, 이는 잠재 클러스터를 예측하여 구성된 순수 MIM 프레임워크입니다. CAPI는 기존 MIM 방법들과 비교하여 안정적인 훈련을 제공하는 클러스터링 기반 손실(clustering-based loss)을 활용하며, ImageNet에서 83.8%의 정확도를 기록하는 등 향상된 성능을 보였습니다. 또한, 논문에서는 다양한 모델과 코드도 공개되어, 연구자들이 이를 쉽게 활용할 수 있도록 하고 있습니다.

- **Technical Details**: CAPI는 마스크된 이미지 모델링 원리를 중심으로 세 가지 디자인 요소에 주목하여 체계적으로 연구하였습니다: 목표 표현(target representation), 손실 함수(loss function) 구성, 및 예측을 수행하기 위해 사용되는 아키텍처(architecture)입니다. 특히, CAPI는 자기 증류(self-distillation) 기법을 사용하여 교사-teacher와 학생-student 비전 트랜스포머를 훈련하고, 이를 통해 손실을 안정적으로 수렴시킵니다. CAPI는 3억 개의 파라미터를 가진 비주얼 인코더로, 현재의 최첨단 방법과 비슷한 성능을 발휘합니다.

- **Performance Highlights**: CAPI는 이전의 MIM 방법들을 능가하여 ImageNet에서 83.8%의 정확도와 ADE20K에서 32.1%의 mIoU를 달성하며, DINOv2의 성능에 근접하는 결과를 보였습니다. 이는 MIM 방식의 잠재적인 장점을 극대화한 것으로, 기존의 다른 방법들과 비교할 때 매우 경쟁력 있는 결과로 평가됩니다. 논문은 MIM의 실행 가능성을 열어주는 중요한 기여로 여겨지며, 향후 연구와 발전에 있어 중요한 기초 자료가 될 것입니다.



### HistoSmith: Single-Stage Histology Image-Label Generation via Conditional Latent Diffusion for Enhanced Cell Segmentation and Classification (https://arxiv.org/abs/2502.08754)
- **What's New**: 이 논문은 HistoSmith라는 새로운 단일 단계 접근 방식을 도입하여 조직학 이미지의 세포 인스턴스 세분화 및 분류를 위한 데이터 증강을 지원합니다. 기존의 분산 모델과 달리, HistoSmith는 잠재적 분산 모델(latent diffusion model)을 사용하여 세포 배치 및 분류 마스크와 같은 데이터 쌍을 생성합니다. 이 모델은 세포 유형, 수량 및 조직 유형과 같은 사용자 정의 매개변수를 기반으로 맞춤형 데이터 생성을 가능하게 합니다.

- **Technical Details**: HistoSmith는 VQ-VAE를 이용하여 입력 이미지와 해당 마스크의 잠재 표현을 학습합니다. 이후, 학습된 잠재 공간에서 분산 모델을 훈련시켜 이미지와 마스크를 생성하며, 이 과정에서 세포 수 및 조직 유형을 인코딩한 10차원 벡터로 조건화합니다. 복원 과정에서 U-Net을 활용하여 각 단계에서 추가된 노이즈를 예측하고, 최종적으로 VQ-VAE 디코더를 통해 새로운 이미지와 해당 마스크를 복원합니다.

- **Performance Highlights**: HistoSmith는 CoNIC 및 CytoDArk0 데이터셋에서 훈련되어, 평균 CS 및 CC 메트릭에서 각각 1.9%와 3.4%의 성능 향상을 보였습니다. 이 연구는 특히 저대표 세포 유형에 대한 데이터 부족 문제에 대응할 수 있는 잠재의 분산 모델 기반 접근 방식을 제시합니다. 본 논문은 조직학 데이터 세트를 다시 증강하는 데 있어 HistoSmith의 효과성과 품질을 검증했습니다.



### Multispectral Remote Sensing for Weed Detection in West Australian Agricultural Lands (https://arxiv.org/abs/2502.08678)
Comments:
          8 pages, 9 figures, 1 table, Accepted for oral presentation at IEEE 25th International Conference on Digital Image Computing: Techniques and Applications (DICTA 2024). Conference Proceeding: 979-8-3503-7903-7/24/\$31.00 (C) 2024 IEEE

- **What's New**: 이 연구에서는 호주 서부의 Kondinin 지역을 위한 맞춤형 다분광 원격 센싱 데이터셋과 잡초 탐지 프레임워크를 개발하였습니다. 기존의 방법들이 갖는 한계를 극복하기 위해 UAV(무인 항공기)를 활용하여 4년에 걸쳐 데이터를 수집하고, GPS를 사용하여 잡초와 작물을 수동으로 라벨링했습니다. 이로써 정밀 농업을 위한 기초 자료를 제공합니다.

- **Technical Details**: 제안된 잡초 탐지 시스템은 데이터 전처리, 특징 선택, 신경망 모델 훈련 등 여러 단계를 포함하는 종단 간(end-to-end) 프레임워크를 가지고 있습니다. 원본 이미지는 잡음 제거, 방사 보정, 영상 정렬 및 스티칭 등 다양한 전처리 단계를 거치게 되며, 여러 가지 식생 지수를 특징으로 추출하여 분류 성능을 향상시킵니다. 최종적으로 ResNet과 같은 딥러닝 모델을 사용하여 잡초를 효과적으로 식별합니다.

- **Performance Highlights**: ResNet-50 모델은 0.9213의 정확도와 0.8735의 F1 스코어를 기록하며, 잡초 탐지에서 가장 높은 성능을 보였습니다. 이 결과는 제안된 데이터셋과 잡초 탐지 방법의 유효성을 입증하여 앞으로의 농업 연구와 실제 적용에 좋은 기초 자료가 될 것입니다. 또한, 이 연구는 서부 호주와 유사한 조건을 가진 다른 지역의 정밀 농업 문제에도 활용될 수 있습니다.



### COutfitGAN: Learning to Synthesize Compatible Outfits Supervised by Silhouette Masks and Fashion Styles (https://arxiv.org/abs/2502.08674)
Comments:
          This paper was accepted by IEEE TMM

- **What's New**: 본 논문에서는 패션 아이템을 기반으로 상호 보완적이며 호환 가능한 아이템을 생성하는 새로운 작업을 제안합니다. 특히 주어진 패션 아이템을 조합하여 호환되는 패션 아이템의 포토리얼리스틱 이미지(photorealistic image)를 합성하는 목표를 가지고 있습니다. 이를 위해, COutfitGAN이라는 새로운 아웃핏 생성 프레임워크를 개발하였고, 이는 의상 생성기(outfit generator)와 여러 가지 판별기(discriminator)를 포함하고 있습니다.

- **Technical Details**: COutfitGAN은 피라미드 스타일 추출기(pyramid style extractor), 의상 생성기, UNet 기반의 진짜/가짜 판별기(real/fake discriminator) 및 조합 판별기(collocation discriminator)로 구성되어 있습니다. 이 프레임워크의 훈련 및 평가를 위해, 200,000개 이상의 아웃핏과 800,000개의 패션 아이템으로 구성된 대규모 패션 아웃핏 데이터셋을 수집하였습니다. 특히, 실루엣 정보를 사용하여 보완 아이템의 생성을 위한 지도 정보를 추가하여 매핑의 난이도를 줄였습니다.

- **Performance Highlights**: COutfitGAN은 유사성(similarity), 진정성(authenticity) 및 호환성(compatibility) 측면에서 다른 기초 모델(baselines)보다 뛰어난 성능을 보였습니다. 실험 결과, 본 연구에서 제안한 방법은 기존의 단일 보완 아이템 생성 방법에 비해 전체 의상을 생성하는 데 있어 큰 진전을 이루었습니다. 이러한 발전은 패션 추천 시스템 및 이미지 생성 분야에서 중요한 기여를 할 것으로 기대됩니다.



### Can this Model Also Recognize Dogs? Zero-Shot Model Search from Weights (https://arxiv.org/abs/2502.09619)
- **What's New**: 이 논문에서는 ProbeLog라는 새로운 분류 모델 검색 방법을 소개합니다. 이 방법은 '개'와 같은 특정 개념을 인식하는 모델을 찾을 수 있도록 설계되었습니다. 기존의 검색 방법은 모델 메타데이터나 훈련 데이터에 의존하나, ProbeLog는 이들에 대한 접근 없이도 작동합니다. 이를 통해 사용자는 필요한 태스크에 적합한 모델을 더 쉽게 찾을 수 있습니다.

- **Technical Details**: ProbeLog는 고정된 입력 샘플 집합을 사용하여 각 모델의 응답을 관찰함으로써 각 출력 차원의 설명자를 계산합니다. 이 설명자는 로짓(lolgit) 수준의 기능적 표현으로, 모델의 특정 기능을 설명합니다. 또한, 이 방법은 협업 필터링(collaborative filtering)을 기반으로 하여, 모델의 대표성을 세 가지로 줄일 수 있는 기법을 도입함으로써, 비용을 감소시키고 효율성을 높입니다.

- **Performance Highlights**: ProbeLog는 실제 데이터셋에서 높은 검색 정확도를 달성하며, 특히 40%의 top-1 정확도를 기록했습니다. 이는 무작위 방법의 0.1%와 비교할 때 상당히 높은 성능입니다. 커다란 모델 저장소에서도 확장 가능하며, 고효율성을 유지하며 작동합니다. 이를 통해 모델 검색의 새로운 가능성을 제시합니다.



### Variational Rectified Flow Matching (https://arxiv.org/abs/2502.09616)
- **What's New**: 이번 연구는 Variational Rectified Flow Matching에 대한 것으로, 전통적인 rectified flow matching을 개선하여 다중 모드( multi-modal ) 속도 벡터 필드를 모델링합니다. 이 방법은 샘플을 소스 분포에서 목표 분포로 이동시키는 방법을 최적화하여, 다중 모드 흐름 방향을 학습하고 샘플링할 수 있도록 돕습니다. 향상된 결과는 합성 데이터, MNIST, CIFAR-10 및 ImageNet에서 증명되었습니다.

- **Technical Details**: 이 연구에서는 velocity vector-field를 다루는데, 이는 소스 및 목표 분포에서 무작위로 추출된 샘플 간 선형 보간(interpolation)을 통해 학습됩니다. 전통적인 방법에서는 mean-squared-error 손실(loss)을 포함하여 학습된 속도 벡터 필드가 'ground-truth' 방향을 평균적으로 취하게 됩니다. 그러나 variational rectified flow matching에서는 은닉 변수(latent variable)를 도입하여 데이터 영역-시간 영역(data-domain-time-domain)의 각 위치에서 다중 모드를 변별할 수 있게 합니다.

- **Performance Highlights**: synthetic data에서 Variational Rectified Flow Matching은 데이터 분포를 더 정확하게 모델링하여 속도 모호성을 더욱 잘 포착합니다. MNIST에서의 이미지 생성 품질을 개선하고, CIFAR-10에서는 전통적인 rectified flow matching을 초월하는 성능을 보여줍니다. 마지막으로, ImageNet에서 SiT-XL의 FID 점수를 지속적으로 개선하는 결과를 나타내었습니다.



### DexTrack: Towards Generalizable Neural Tracking Control for Dexterous Manipulation from Human References (https://arxiv.org/abs/2502.09614)
Comments:
          Accepted to ICLR 2025. Website: this https URL Code: this https URL Video: this https URL

- **What's New**: 이번 논문에서는 인간의 포든 동작을 기반으로 한 일반화 가능한 신경 추적 제어기를 개발하는 문제를 다룹니다. 이 제어기는 다양하고 복잡한 물체를 조작하기 위해 로봇 손을 제어하는 것을 목표로 하며, 기존 강화 학습 및 궤적 최적화 방법의 한계를 극복하기 위해 설계되었습니다. 우리는 대규모의 성공적인 로봇 추적 시연을 통해 신경 제어기를 학습시키는 접근법을 소개하며, 이를 통해 성능과 성공적인 시연의 수와 질을 모두 향상시킬 수 있습니다.

- **Technical Details**: 우리의 연구는 DexTrack이라는 새로운 신경 추적 제어기를 제안합니다. 이 제어기는 인간 손-객체 조작 경로를 로봇에 맞추어 최적화하고, 성공적인 로봇 추적 시연과 함께 교대로 훈련됩니다. 강화 학습과 모방 학습 기법을 통합하여 다양한 상황에서 제어기의 성능을 향상시키고, 동적 환경에서도 강력한 적응력을 유지합니다.

- **Performance Highlights**: DexTrack은 두 개의 데이터 세트에서 기존 방법과 비교하여 뛰어난 성능을 발휘하였으며, 이전 방법보다 10% 이상의 성공률을 기록하였습니다. 시뮬레이터인 Isaac Gym 및 실제 환경에서의 평가를 통해 광범위한 조작 추적 작업을 성공적으로 수행하고, 예기치 않은 상황에서도 회복 능력이 있음을 입증했습니다. 이 연구는 반복적인 시연 채굴을 통해 발전하는 일반화 가능성이 큰 신경 추적 제어기를 제시합니다.



### Designing a Conditional Prior Distribution for Flow-Based Generative Models (https://arxiv.org/abs/2502.09611)
- **What's New**: 본 연구에서는 조건부 생성 모델의 새로운 접근 방식을 제안하며, 특히 비선형한 사전 분포(prior distribution) 설계의 필요성을 강조합니다. 조건을 입력으로 활용하여 평균적으로 같은 조건의 데이터 포인트에 최소한으로 가까운 위치로 매핑하는 방법을 고안하였습니다. 이를 통해 샘플링 단계 수를 줄이면서도 높은 품질의 결과를 생성할 수 있도록 하였습니다.

- **Technical Details**: 본 연구의 방법은 조건부 흐름 기반 생성 모델을 위한 비정보적인 사전 분포를 설계하는 것을 목표로 합니다. 주어진 입력 조건에 대해서, 데이터 공간에서 가장 '평균적인' 조건을 찾고, 이를 중심으로 하여 확률 분포를 설정하여, 샘플을 조건부 목표 분포로 매핑합니다. 이를 위해 가우시안 혼합 모델(GMM)과 사전 분포의 설계 및 흐름 매칭(flow matching) 방법론을 결합하여 효과적인 조건부 생성에 기여하였습니다.

- **Performance Highlights**: 실험 결과, 제안하는 방법은 기존 모델들과 비교했을 때 빠른 훈련 및 샘플링 속도를 보여주었으며, 생성된 이미지의 품질과 다양성 면에서도 현저한 개선이 있음을 증명하였습니다. FID, KID 및 CLIP 점수와 같은 평가 지표에서 우수한 결과를 기록하였습니다. 이를 통해 본 연구의 접근 방식이 실제 데이터셋에서도 효과적으로 적용될 수 있음을 입증하였습니다.



### Diffusing DeBias: a Recipe for Turning a Bug into a Featur (https://arxiv.org/abs/2502.09564)
Comments:
          29 Pages, 12 Figures

- **What's New**: 이번 연구에서는 Diffusing DeBias (DDB)라는 새로운 방법론을 제안하였습니다. DDB는 일반적인 디바이싱(debiasing) 기법과 함께 사용할 수 있는 플러그인 형태로, 조건부 확산 모델(Conditional Diffusion Model)의 편향 학습 특성을 활용합니다. 이 접근법을 통해 인위적으로 생성된 편향 정렬(bias-aligned) 이미지를 통해 편향 앰프 모델을 훈련시키고, 훈련 집합의 기억화 문제를 해결하여 더 나은 일반화 성능을 달성하는 것을 목표로 합니다.

- **Technical Details**: DDB는 확산 모델의 특성을 활용하여 각 클래스의 편향 정렬 분포를 학습합니다. 이를 통해 생성된 합성 이미지를 사용하여 편향 앰프 모델을 훈련시키며, 이 모델은 다양한 비지도 디바이싱 기법에서 보조 방법으로 활용됩니다. 연구 결과, DDB는 특히 훈련 데이터에 대한 편향 극복에 있어 기존의 최첨단 기법들보다 우수한 성능을 보였습니다.

- **Performance Highlights**: DDB는 두 가지 서로 다른 디바이싱 프레임워크에 통합되어 최고의 성능을 낼 수 있는 가능성을 보여줍니다. Recipe I에서는 G-DRO 알고리즘을 활용하여 서브 집단을 추출하며, Recipe II에서는 종단 간(end-to-end) 방법에서 손실 함수를 제공합니다. 두 접근법 모두 여러 기준 데이터셋에서 기존의 최첨단 방법들을 상당한 차이로 초월하는 성과를 달성하고 있습니다.



### EmbodiedBench: Comprehensive Benchmarking Multi-modal Large Language Models for Vision-Driven Embodied Agents (https://arxiv.org/abs/2502.09560)
Comments:
          51 pages

- **What's New**: 이번 논문에서는 Multi-modal Large Language Models (MLLMs)를 기반으로 한 구체화된 에이전트를 평가하기 위한 새로운 기준인 EmbodiedBench를 소개합니다. EmbodiedBench는 4가지 환경에서 1,128개의 다양한 테스트 과제를 포함하여, 고수준의 의미적 과제와 원자적 행동을 포함한 저수준 과제를 평가합니다. 특히, 에이전트의 공통 감각 추론 및 복잡한 지시 이해와 같은 필수 능력을 평가할 수 있도록 세분화된 하위 집합을 제공합니다.

- **Technical Details**: EmbodiedBench는 두 가지 주요 기능을 갖추고 있으며, 첫째로 작업 수준에 따라 다양한 작업을 제공합니다. 둘째로, 공통 감각 추론 및 시각 인식과 같은 여섯 가지 핵심 능력을 평가하는 세밀한 평가 프레임워크를 도입하여 기존 벤치마크와 차별화됩니다. 이를 통해 13개의 MLLM 모델을 평가하며, 에이전트의 결정을 내리기 위해 통합된 에이전트 프레임워크를 사용합니다.

- **Performance Highlights**: MBLMs는 고수준의 작업에서는 뛰어난 성능을 보여주지만, 저수준 조작에서는 한계를 보입니다. 특정 모델인 GPT-4o는 평균 28.9%의 점수를 기록하며, 저수준 작업에서 성능이 40%에서 70%까지 저하됨을 알 수 있습니다. 이 연구는 MLLM 기반 구체화된 에이전트의 발전을 위한 소중한 통찰을 제공합니다.



### When and How Does CLIP Enable Domain and Compositional Generalization? (https://arxiv.org/abs/2502.09507)
- **What's New**: 이번 연구에서는 CLIP(Contrastive Language-Image Pretraining) 모델이 다양한 도메인에서의 일반화 성능을 어떻게 발휘하는지에 대한 체계적인 분석을 진행했습니다. 기존 연구에서 CLIP의 훈련 배포의 다양성이 OOD(Out-of-Distribution) 일반화 성능의 주요 원인이라고 제시되었으나, 어떤 요인이 이러한 일반화에 영향을 미치는지에 대한 질문이 남아 있었습니다. 본 논문은 도메인 일반화와 구성적 일반화(compositional generalization)에 대한 탐구를 통해 CLIP의 성능 저하를 설명하려 하였습니다.

- **Technical Details**: CLIP 모델을 훈련시키기 위해 자연 이미지와 비자연 샘플들로 구성된 데이터셋을 사용하여 도메인 믹스를 체계적으로 구성하였습니다. 이 과정에서 훈련 중 노출된 클래스와 도메인의 다양성을 정밀하게 조작할 수 있는 실험 환경을 마련하였습니다. 연구 결과, 도메인 다양성이 일반화에 필수적이며, 구성적 일반화는 때때로 도메인 일반화보다 약하다는 것을 발견했습니다.

- **Performance Highlights**: CLIP 모델의 일반화 성능은 다양한 도메인을 사용하는 훈련 데이터에 의해 개선됩니다. 실험 결과, 클립은 특정 도메인에서 제한된 특징만을 공유할 경우 일반화에 실패하는 경향이있습니다. 따라서 성공적인 일반화를 위해서는 특징과 회로 공유가 충분해야 하며, 이는 CLIP 모델의 중간 레이어에서의 표현 공유를 통해 이루어집니다.



### DiffRenderGAN: Addressing Training Data Scarcity in Deep Segmentation Networks for Quantitative Nanomaterial Analysis through Differentiable Rendering and Generative Modelling (https://arxiv.org/abs/2502.09477)
- **What's New**: 본 논문에서는 DiffRenderGAN이라는 새로운 생성 모델을 소개합니다. 이 모델은 Generative Adversarial Network (GAN) 프레임워크에 차별적인 렌더러(differentiable renderer)를 통합하여 비주석(real microscopy images) 실 이미지로부터 주석이 달린 합성 나노 입자 이미지를 생성할 수 있도록 설계되었습니다. 이 접근 방식은 기존의 합성 데이터 방법보다 수동 개입을 줄이고 분할(segmentation) 성능을 향상시킵니다.

- **Technical Details**: DiffRenderGAN은 3D 나노 입자 모델(meshes)과 위치 및 크기 정보를 담고 있는 변환 행렬을 사용하여 희소한 훈련 데이터를 보완합니다. 이미지 렌더링 과정은 가상의 3D 장면을 현실적인 2D 디지털 이미지로 변환하는 것으로, 이 과정에서 Bidirectional Scattering Distribution Functions (BSDFs)와 같은 재료 특성을 시뮬레이션합니다. 이러한 방법을 통해 합성, 주석이 달린 이미지가 생성되어 분할 네트워크를 효과적으로 훈련할 수 있습니다.

- **Performance Highlights**: DiffRenderGAN은 여러 이온 및 전자 현미경 사례에서 테스트되었으며, 특히 실시간 현미경 이미지에 대해 높은 분할 성능을 달성했습니다. 우리의 연구에서는 합성 데이터의 생성과 기존 방법들과의 성능 비교를 통해, DiffRenderGAN이 기존 방법들의 성능을 초과하거나 동일한 수준의 성능을 나타냈음을 보여주었습니다. 이는 복잡한 나노 물질 시스템의 정량화와 이해를 진전시키는 데 강력한 도구가 될 것으로 기대됩니다.



### Metamorphic Testing for Pose Estimation Systems (https://arxiv.org/abs/2502.09460)
Comments:
          Accepted for publication at 2025 IEEE Conference on Software Testing, Verification and Validation (ICST)

- **What's New**: 이 논문에서는 다양한 분야에 활용되는 포즈 추정 시스템의 성능 평가를 위해 MET-POSE라는 새로운 메타모픽 테스트 프레임워크를 제안합니다. 이 시스템은 수작업 주석(labeling)의 필요성을 넘어, 사용자가 특정 애플리케이션에 더 적합한 조건에서 시스템을 평가할 수 있도록 돕습니다. 기존의 테스트 데이터셋에 의존하지 않고, 주석이 필요 없는 방식으로 포즈 추정 시스템의 성능을 평가할 수 있는 가능성을 열어줍니다.

- **Technical Details**: MET-POSE는 포즈 추정 시스템에서 흔히 마주치는 문제들을 해결하기 위한 메타모픽 규칙(metamorphic rules) 목록을 제공합니다. 또한, 이러한 규칙이 어떻게 평가될 수 있는지를 제시하며, Mediapipe Holistic이라는 최신 포즈 추정 시스템에 적용하여 실험적으로 효과를 검증합니다. 이 프레임워크는 FLIC와 PHOENIX 데이터셋을 활용하여 포즈 추정 시스템의 성능을 다양한 환경에서 평가하는 방법을 제시합니다.

- **Performance Highlights**: 실험 결과, MET-POSE는 수작업으로 라벨링한 데이터와 유사하거나 더 높은 속도로 포즈 추정 시스템의 결함(faults)을 발견할 수 있음을 보여줍니다. 사용자는 자신이 필요한 정확도 수준과 결함에 맞춰 규칙 집합(rule set)을 조정할 수 있어, 애플리케이션 중에서 보다 효과적으로 성능 평가를 수행할 수 있습니다. 이는 포즈 추정 디젤로 인해 발생할 수 있는 문제를 사전에 예방하는 데 기여할 것으로 기대됩니다.



### Wasserstein distributional adversarial training for deep neural networks (https://arxiv.org/abs/2502.09352)
Comments:
          15 pages, 4 figures

- **What's New**: 이 논문에서는 adversarial 공격에 대한 훈련 방법의 혁신적인 개선을 제안합니다. TRADES 방법을 확장하여 Wasserstein 분포적으로 강건한 최적화 문제에 대한 민감도 분석을 통해 분포적 공격 위협에 대응하는 새로운 훈련 방법을 도입합니다. 또한, 이는 기존의 훈련된 모델에 적용할 수 있는 효율적인 미세 조정 방법을 포함합니다.

- **Technical Details**: 제안된 방법은 Wasserstein 거리 기반의 분포적 위협 모델을 고려하며, 다양한 성공적인 사전 훈련 신경망을 RobustBench에서 테스트하여 효과성을 입증합니다. 새로운 훈련 방법은 기존 포인트와이즈 공격에 대한 강건성을 유지하면서 분포적 공격에 대한 강건성을 향상시킵니다. 이 연구는 기존의 신경망 훈련 기법들과 차별화된 접근 방식을 통해 진행됩니다.

- **Performance Highlights**: 실험 결과는 추가 훈련이 Wasserstein 분포적 강건성을 향상시키는 동시에 기존의 포인트와이즈 강건성을 유지할 수 있도록 도와줌을 보여줍니다. 하지만 대규모 합성 데이터셋으로 사전 훈련된 모델의 경우 개선 효과가 덜 두드러지게 나타났습니다. 그럼에도 불구하고 원본 훈련 데이터셋(50k 이미지) 만으로도 성능이 향상되는 경우가 있음을 확인하였습니다.



### Visual Graph Question Answering with ASP and LLMs for Language Parsing (https://arxiv.org/abs/2502.09211)
Comments:
          In Proceedings ICLP 2024, arXiv:2502.08453. This work was partially funded from the Bosch Center for AI

- **What's New**: 이번 연구는 Visual Question Answering (VQA) 분야에서 새로운 데이터셋을 소개하며, 이미지의 그래프를 인식하고 자연어 처리를 통해 질문에 답하는 복합적인 문제를 다룹니다. 특히, 지하철 노선과 유사한 그래프 형태의 이미지를 포함시킨 새로운 데이터셋을 생성했습니다. 이번 연구는 모듈형 신경-기호적(neuro-symbolic) 접근법을 통해 ASP(Answer-Set Programming)를 통합하여 VQA의 해답을 찾고자 합니다.

- **Technical Details**: 이 연구에서는 optical graph recognition을 위한 그래프 파싱, 레이블 파싱을 위한 pretrained optical character recognition 신경망, 언어 처리를 위한 Large Language Models (LLMs), 그리고 추론을 위한 ASP를 결합합니다. 이러한 모듈형 설계는 각기 다른 프로세스를 조화롭게 통합해 VQA를 해결하는 데 기여합니다.

- **Performance Highlights**: 제시된 방법은 기존 데이터셋에서 약 73%의 평균 정확도를 기록하며, 모듈형 신경-기호적 시스템의 가능성을 보여줍니다. 이를 통해 추가 훈련이 없이도 사전 훈련된 모델과 논리 프로그래밍을 활용하여 복잡한 VQA 작업을 해결할 수 있는 가능성을 입증하였습니다.



### Shortcut Learning Susceptibility in Vision Classifiers (https://arxiv.org/abs/2502.09150)
- **What's New**: 이 논문에서는 머신 러닝 모델이 데이터에서 의미 있는 특성을 포착하는 대신, 단기적 학습(shortcut learning) 즉, 훈련 데이터의 비생산적인 상관관계를 이용하는 방식에 대해 다룹니다. 연구팀은 다양한 비전 분류 아키텍처(CNN, MLP, ViT)를 시스템atic 하게 평가하여, 모델이 인위적으로 추가된 단기적 단서(shortcuts)를 사용하거나 실제 구별 가능한 특징을 학습하는지를 분석합니다. 또한, 수정된 데이터 세트를 통해 아키텍처마다 단기적 학습에 대한 취약성을 비교할 수 있는 방법론을 제시합니다.

- **Technical Details**: 이 연구에서는 특정 픽셀 영역을 클래스 레이블과 결정적으로 상관되도록 수정하여 인위적인 단서를 주입합니다. 그런 다음 CNN, MLP 및 ViT 모델을 이 수정된 데이터 세트에서 학습시킨 후, 두 개의 테스트 세트에서 성능을 평가합니다. 하나의 테스트 세트는 기존의 단서를 포함하고, 다른 하나는 그런 단서가 없는 상황입니다. 이러한 분석을 통해 모델 각기가 단기적인 단서를 얼마나 의존하는지 정량적으로 비교합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터 세트(MNIST, Fashion-MNIST, SVHN, CIFAR-10)를 통해 모델의 단기적 학습 의존성을 평가하며, 내부의 가중치를 재구성하는 네트워크 반전 기반 기법을 사용하여 모델이 어떤 정보를 저장하는지를 분석합니다. 또한 학습률에 따라 단기적 의존성이 어떻게 변화하는지를 살펴보아, 낮은 학습률을 사용한 경우 더 미세하고 의미 있는 특징을 학습할 수 있음을 발견했습니다. 결과적으로,하는 데이터 처리 방식과 단기적 학습에 대한 취약성을 이해할 수 있습니다.



### Replay-free Online Continual Learning with Self-Supervised MultiPatches (https://arxiv.org/abs/2502.09140)
Comments:
          Accepted at ESANN 2025

- **What's New**: 이번 논문에서는 Continual MultiPatches (CMP)를 제안합니다. CMP는 온라인 연속 학습(Online Continual Learning, OCL) 상황에서 재생(replay) 샘플을 사용하지 않고도 효과적으로 자가 지도 학습(self-supervised learning, SSL) 방식을 진행할 수 있도록 설계된 플러그인 방법입니다. 이 방법은 단일 예제에서 여러 패치를 생성하고 이를 공유된 특성 공간(feature space)으로 투영하여, 동일한 예제에 대한 패치들이 결합되지 않고 서로 밀착되도록 합니다.

- **Technical Details**: CMP는 기존의 인스턴스 구별(instance discrimination) SSL 전략의 상위 레이어에서 적용되며, 각 패치에 대해 엔코더 네트워크를 사용하여 잠재 표현(latent representation)을 계산합니다. CMP의 손실 함수는 평균 패치 표현에 대한 유사성을 유지하도록 설계되었으며, 총 코딩 비율(Total Coding Rate, TCR) 손실도 활용하여 잠재 표현이 하나의 점으로 붕괴되지 않도록 합니다. 이 방식은 메모리 버퍼 없이도 패치의 수를 늘려 미니배치(minibatch) 크기를 확장합니다.

- **Performance Highlights**: CMP는 도전적인 클래스 증가(class-incremental) OCL 벤치마크에서 실험을 통해 재생 기반 전략과 비교하여 성능이 우수함을 입증하였습니다. CMP는 제한된 계산 예산을 가지고도 경쟁력 있는 OCL 방법론과 비교했을 때도 뛰어난 성능을 발휘하였습니다. 이는 CMP가 재생 없이도 효과적인 학습을 가능하게 함을 보여줍니다.



### Improving Deep Regression with Tightness (https://arxiv.org/abs/2502.09122)
Comments:
          ICLR 2025, Code: this https URL

- **What's New**: 이 논문은 딥 회귀에서 타겟의 순서를 보존함으로써 다양한 과제의 성능을 향상시킬 수 있음을 보여줍니다. 그러나 순서 보존이 성능에 미치는 이점에 대한 이론적 설명은 부족했습니다. 본 연구에서는 조건부 엔트로피 $H(Z|Y)$를 최소화하여 표현 Z와 타겟 Y 간의 유사성을 유지하는 최적 수송 기반 규제를 소개하고, 이를 통해 회귀의 성능을 개선하는 방법을 제안합니다.

- **Technical Details**: 순서를 보존함으로써 표현 Z의 조건부 엔트로피 $H(Z|Y)$가 감소하는 것을 발견하였습니다. 일반적인 회귀 손실이 이 조건부 엔트로피를 줄이기에는 미흡하다는 것을 보여줍니다. 이를 해결하기 위해, 회귀기 목표를 중복하여 사용하는 전략과 함께 Regression Optimal Transport (ROT) Regularizer를 도입하여 표현의 안정성을 높입니다.

- **Performance Highlights**: 세 가지 실제 회귀 과제에서 제안한 전략의 효과를 검증하였습니다. 다중 목표 접근법과 ROT-Reg가 각각 전역 및 지역적으로 표현을 조정하여 성능을 극대화함을 확인했습니다. 이 연구는 회귀 표현의 순서 보존과 관련된 기여로, 회귀 작업을 분류 문제로 재정의할 수 있는 통찰력을 제공합니다.



### Zero-shot Concept Bottleneck Models (https://arxiv.org/abs/2502.09018)
Comments:
          14 pages, 8 figures

- **What's New**: 이번 논문에서는 zero-shot concept bottleneck models (Z-CBMs)를 소개하며, 이는 학습 없는 상태에서 개념 지도와 라벨 예측을 수행할 수 있는 모델입니다. Z-CBMs는 대규모 개념 데이터베이스를 활용하여 다양한 도메인에서 입력을 설명합니다. 기존의 Concept Bottleneck Models (CBMs)과 달리, Z-CBMs는 타겟 데이터셋의 수집이나 학습 없이 이해 가능하고 개입 가능한 개념을 제공합니다.

- **Technical Details**: Z-CBMs는 concept retrieval과 concept regression 두 가지 모듈로 구성됩니다. Concept retrieval 모듈은 효율적인 교차 모달 검색 알고리즘을 통해 입력과 관련된 개념을 동적으로 찾아내며, concept regression 모듈은 중복된 개념을 피하고 상호 배타적인 개념을 선택하기 위해 희소 선형 회귀(sparse linear regression)를 사용합니다. 이러한 접근 방식은 타겟 데이터셋 없이도 개념과 라벨을 예측하는 데 필요한 모든 단계를 처리할 수 있습니다.

- **Performance Highlights**: Z-CBMs는 12개의 데이터셋에 대한 광범위한 실험을 통해 기존의 학습된 CBMs와 비슷하거나 더 나은 성능을 발휘함을 입증했습니다. 특히, Z-CBMs는 예측된 개념에 대한 인간의 개입을 통해 전반적인 신뢰성을 높일 수 있으며, 이는 개념 기반 예측의 실용성을 강조합니다. 최종적으로 Z-CBMs는 다양한 도메인에서 효과적으로 사용될 수 있는 가능성을 보여주고 있습니다.



### The Stochastic Parrot on LLM's Shoulder: A Summative Assessment of Physical Concept Understanding (https://arxiv.org/abs/2502.08946)
Comments:
          NAACL 2025 Main Conference. First 5 authors contributed equally. Project page: this https URL

- **What's New**: 본 논문은 LLMs(대형 언어 모델)의 진정한 이해 여부에 대해 정량적인 실험을 통해 검증하고자 합니다. 우리는 특히 Stochastic Parrot 현상, 즉 LLM이 단순히 연관 패턴을 바탕으로 반복하는지 여부를 평가하는 PhysCo라는 새로운 과제를 제안합니다. 이 과제는 물리적 개념 이해도를 측정하기 위해 설계되었습니다. 고민이 많았던 메모리 문제를 해결하기 위해 그리드 형식의 입력을 사용하여 다양한 이해 수준을 표현하고 있습니다.

- **Technical Details**: PhysiCo는 저급 이해(subtask)와 고급 이해(high-level understanding)라는 두 가지 하위 과제를 포함합니다. 저급 이해는 LLM의 기억 능력을 측정하는 자연어 형식 질문으로 구성되어 있으며, 고급 이해 과제는 추상 표현을 기반으로 상대적으로 심화된 이해를 평가하는 숙제입니다. 우리는 LLMs가 저급 과제에서 높은 정확도를 보이지만, 고급 과제에서는 인간에 비해 40% 가량 성능이 떨어진다는 두 가지 주요 결론을 도출했습니다.

- **Performance Highlights**: 실험 결과, 최신 LLMs는 저급 이해 과제에서 95% 이상의 정확도를 기록하였으나, 고급 이해 과제에서는 인간에 비해 평균 약 40% 낮은 정확도를 보였습니다. 이는 LLM이 진정한 개념 이해 능력에서는 한계를 나타내며, 새로운 그리드 형식이 아닌 그 자체의 고차원적 이해의 어려움이 원인임을 시사합니다. 본 연구는 LLM의 이해력을 측정하는 방법론적 기틀을 확립하고, LLM과 인간 간의 성능 격차를 명확히 보여줍니다.



### On the Promise for Assurance of Differentiable Neurosymbolic Reasoning Paradigms (https://arxiv.org/abs/2502.08932)
- **What's New**: 이 논문은 신경망(neural network)과 고전 논리 프로그래밍을 결합한 신경 상징 AI(neurosymbolic AI) 시스템의 통합된 보장성(assurance)을 평가합니다. 특히, Scallop이라는 최첨단의 신경 상징 라이브러리를 사용하여 이미지 및 오디오 도메인에서 분류(classification)와 추론(reasoning) 작업을 수행합니다. 연구 결과, 완전한 신경망만으로 학습하기 어려운 복잡한 수학적 연산이 정의된 경우, 신경 상징 모델이 더 높은 보장성을 제공한다고 나타났습니다.

- **Technical Details**: 이 연구는 신경 상징 프로그래밍이 보장성 설계(assurance-by-design)를 제공할 수 있는지를 조사합니다. 신경망을 기반으로 하여 Scallop의 분별 가능한(differentiable) 추론 엔진을 통해 신경 상징 모델을 만들고, 이를 완전한 신경망과 비교하여 여러 작업과 평가를 진행했습니다. 연구 과정에서 적대적 강건성(adversarial robustness), 신뢰도 조정(calibration), 사용자 성능 차이(user performance parity) 등의 요소를 포함하여 보장성을 평가했습니다.

- **Performance Highlights**: 연구 결과는 다음과 같은 중요한 발견을 포함합니다. 첫째, 분별 가능한 신경 상징 추론은 알고리즘에 정의된 논리 연산이 존재하는 경우에 높은 보장성을 제공합니다. 둘째, 신경 상징 모델이 해석 가능한 단축을 취하는 경우, 성능은 유지되더라도 적대적 공격에 더 취약해지는 경향이 있습니다. 마지막으로, 데이터 불균형(class imbalance) 문제에서만 데이터 효율성이 보장되고, 일반적으로 적은 데이터로 더 높은 보장성을 보장하지는 않습니다.



### Detecting Malicious Concepts Without Image Generation in AIGC (https://arxiv.org/abs/2502.08921)
- **What's New**: 이 논문에서는 악의적인 개념 감지를 연구에 통합한 최초의 체계적인 작업인 Concept QuickLook을 제안합니다. 이 접근 방식은 이미지를 생성하지 않고 개념 파일만으로 악의적인 개념을 탐지합니다. 사용자들은 악의적인 텍스트 설명과 예제 이미지를 통해 악성 콘텐츠를 다운로드하도록 유인할 수 있기 때문에, 이러한 자동 감지 시스템의 필요성이 커지고 있습니다.

- **Technical Details**: 악의적인 개념을 정의하고, 개념 매칭(concept matching) 및 모호한 감지(fuzzy detection)라는 두 가지 작업 모드를 설계했습니다. 이 시스템은 개념 파일에 기반하여 효율적으로 악의적인 콘텐츠를 탐지하며, 이미지 생성 없이도 신속하게 수행됩니다. 광범위한 실험을 통해 Concept QuickLook의 효용성과 실행 가능성을 입증했습니다.

- **Performance Highlights**: 제안하는 Concept QuickLook은 악성 개념을 효과적으로 탐지할 수 있으며, 공유 플랫폼에서의 실제 적용 가능성을 보여줍니다. 추가적인 강인성 실험을 설계하여 솔루션의 효과성을 검증했습니다. 이 연구는 악의적인 개념 탐지 작업을 시작하고 영감을 줄 수 있는 기초를 마련하길 바랍니다.



### Harnessing Vision Models for Time Series Analysis: A Survey (https://arxiv.org/abs/2502.08869)
- **What's New**: 이 논문은 시간 시계열 분석(time series analysis)에서 LLMs(대형 언어 모델)보다 LVMs(대형 비전 모델)의 장점에 대해 논의합니다. 최근까지 연구의 대부분은 시퀀스 모델링(sequence modeling)에 집중되어 있었으나, 비전 모델이 시간 시계열 분석에서 중요한 역할을 할 수 있다는 점을 강조하고 있습니다. 이 논문은 시간 시계열을 이미지로 인코딩하는 방법과 여러 작업을 위한 모델링 기법을 제시하며, 이 분야의 나아갈 방향도 함께 논의합니다.

- **Technical Details**: 시간 시계열은 다양한 형태로 표현될 수 있으며, 여기서 이미지로 변환된 시간 시계열(imaged time series)을 대상으로 연구합니다. UTS(단일 변량 시계열)와 MTS(다변량 시계열)의 정의 및 표현 방법이 상세히 설명됩니다. 그림 1에서는 비전 모델을 시간 시계열 작업에 적용하는 일반적인 프로세스와 프레임워크를 제시하며, 경우에 따라 선형 그래프(line plot), 히트맵(heatmap), 스펙트로그램(spectrogram) 등의 변환 기법에 대해 다룹니다.

- **Performance Highlights**: 비전 모델이 시간 시계열 작업에서 왜 더 효과적인지를 여러 사례를 통해 설명하고 있으며, 기존의 LLMs와 비교해 여러 장점이 발견됩니다. LVMs는 시계열 데이터의 고유한 패턴, 상관 관계(correlation) 및 장기 종속성(long-term dependency) 모형화를 더 잘 수행할 수 있습니다. 또한, 이 논문은 향후 LMMs(대형 다중모달 모델)의 가능성을 제시하며, 시계열 데이터와 비전 데이터를 통합한 혁신적인 방식의 개발을 독려합니다.



### MRUCT: Mixed Reality Assistance for Acupuncture Guided by Ultrasonic Computed Tomography (https://arxiv.org/abs/2502.08786)
- **What's New**: 이 논문에서는 초음파 컴퓨터 단층 촬영(ultrasonic computed tomography, UCT)와 혼합 현실(mixed reality, MR) 기술을 통합한 MRUCT라는 혁신적인 시스템을 개발하였습니다. 이 시스템은 침술사들이 필요로 하는 실시간 시각화 기능을 제공하여 신체의 해부학적 구조를 기반으로 침의 삽입을 지원합니다. 의료 학생들도 이러한 시스템을 활용하여 학습할 수 있도록 했으며, 실제 환자에게 적용 가능성을 보여주고 있습니다.

- **Technical Details**: 연구에서는 비강체 등록(non-rigid registration) 방법을 활용하여 UCT 데이터로부터 해부학적 구조를 재구성하는 과정을 설명합니다. MRUCT는 자동 생성된 참조 포인트를 기반으로 침의 삽입 경로를 시각화합니다. 시스템 설계는 사용자 친화적(user-friendly)이고 효율적인 기능을 제공하도록 설계되었으며, 일반적인 침술 워크플로우와 비교한 성능 평가를 포함합니다.

- **Performance Highlights**: MRUCT 시스템의 성능은 기존의 침술 명상 기술과 비교하여 유의미한 개선이 있음을 보여줍니다. 새로운 침술사와 의료 학생들이 정확한 침 삽입을 통해 더 나은 결과를 얻을 수 있도록 지원하며, 다양한 의료 분야에서의 혁신적인 사용 가능성을 제안합니다. 이러한 연구 결과는 치료 의료 관행과 기술적 접근 방식을 향상시키는 데 크게 기여할 것으로 기대됩니다.



### Skrr: Skip and Re-use Text Encoder Layers for Memory Efficient Text-to-Image Generation (https://arxiv.org/abs/2502.08690)
- **What's New**: 이번 연구에서는 텍스트-이미지(T2I) 확산 모델에서 텍스트 인코더의 메모리 사용 효율을 높이기 위한 새로운 방법인 Skip and Re-use layers(Skrr)를 제안합니다. 기존의 denoising 모듈과 달리, 텍스트 인코더는 단일 순방향 통과만으로 텍스트 임베딩을 생성하는데, 이는 메모리 사용량이 높아지는 원인으로 작용합니다.

- **Technical Details**: Skrr는 T2I 작업에 맞춰 설계된 가지치기(pruning) 전략으로, transformer 블록 내의 특정 레이어를 선택적으로 건너뛰거나 재사용하여 메모리 소비를 줄입니다. 이 방법은 높은 희소성(sparsity) 수준에서도 텍스트-이미지 생성 품질을 유지하며, 기존의 블록 기반 가지치기 방법을 초월하는 성능을 보여줍니다.

- **Performance Highlights**: 광범위한 실험을 통해 Skrr는 FID, CLIP, DreamSim 및 GenEval 점수 등 여러 평가 메트릭에서 성능을 유지하면서도 메모리 효율성을 극대화하는 것을 입증했습니다. 이로 인해 Skrr는 최신 기술(trade mark)보다 우수한 메모리 효율성을 달성하며, 성능 저하 없이 높은 품질의 이미지를 생성할 수 있습니다.



### LIR-LIVO: A Lightweight,Robust LiDAR/Vision/Inertial Odometry with Illumination-Resilient Deep Features (https://arxiv.org/abs/2502.08676)
- **What's New**: 이 논문에서는 도전적인 조명 및 열화된 환경에서 설계된 경량 및 강력한 LiDAR-관성-비주얼 오도메트리 시스템인 LIR-LIVO를 제안합니다. 이 방법은 딥 러닝 기반의 조명 복원 기능과 LiDAR-관성-비주얼 오도메트리(LIVO)를 활용하여 낮은 계산 비용으로 높은 정확도와 강력성을 달성합니다. 다양한 벤치마크 데이터셋에서 실험을 수행한 결과, 제안한 방법이 다른 SOTA(상태의 최신 기술) 방법을 초월하는 성능을 보임을 확인했습니다.

- **Technical Details**: LIR-LIVO는 LiDAR와 비주얼 정보, 그리고 관성 센서를 효율적으로 융합하여 기능합니다. 이 시스템은 FAST-LIO2의 직접 방법을 상속받아 피쳐 포인트의 포인트-투-플레인(점-에서-평면) 거리 최적화를 통해 상태 추정을 수행합니다. 또한, 딥 러닝 기반의 SuperPoint 알고리즘과 LightGlue 알고리즘을 통해 효과적인 피쳐 매칭을 수행하여 조명이 변동이 큰 환경에서도 뛰어난 강인성을 보여줍니다.

- **Performance Highlights**: 제안된 LIR-LIVO 시스템은 특히 Hilti'22 데이터셋에서 조명이 좋지 않은 조건에서의 강력한 포즈 추정을 증명했습니다. 실험 결과, LIR-LIVO는 다양한 데이터셋에서 높은 성능을 기록하며, LiDAR 포인트와 비주얼 피쳐 포인트 간의 연관성을 통해 정밀한 3D 위치를 직접 획득하여 VIO 시스템의 효율성을 향상시킵니다. 또한 GitHub에 공개된 코드로 로봇 커뮤니티의 발전을 촉진하고자 합니다.



### Color Universal Design Neural Network for the Color Vision Deficiencies (https://arxiv.org/abs/2502.08671)
Comments:
          12 pages, 10 figures

- **What's New**: 이 논문은 색각 결핍을 가진 사람들이 시각적으로 이해할 수 있는 이미지를 생성하는 색상 유니버설 디자인 네트워크인 CUD-Net을 제안합니다. CUD-Net은 입력 이미지를 위한 특정 필터를 사용하여 색상을 구별하고 유지하는 합성곱 신경망입니다. 기존의 하드웨어 장치에 의존하지 않고 실시간으로 색상 유니버설 디자인 이미지를 생성하는 새로운 접근 방식을 제시합니다.

- **Technical Details**: CUD-Net은 데이터를 두 개의 그룹으로 나누어 학습 데이터셋을 정제하는 방법을 포함합니다. 이 모델은 이미지의 색상 충실도를 보존하면서 비-CUD 객체의 대비를 최대화하는 규칙적 필터를 설계하는 심층 학습 회귀 기법을 활용합니다. 또한, 입력 이미지를 확장하기 위해 특정한 전처리 및 다중 모드 융합 아키텍처를 적용합니다.

- **Performance Highlights**: 제안된 방법은 색상 및 대비의 안정성을 유지하면서 고품질의 CUD 이미지를 생성하는 데 성공합니다. 이 연구는 다양한 종류의 색각 결핍을 가진 개인을 위해 최적화된 이미지를 제공하는 데 기여하며, CUD-Net의 구현 코드는 GitHub 리포지토리에서 제공됩니다.



### Unpaired Image-to-Image Translation with Content Preserving Perspective: A Review (https://arxiv.org/abs/2502.08667)
- **What's New**: 이번 논문은 Image-to-Image (I2I) 변환 기술을 세 가지 범주, 즉 완전 콘텐츠 보존, 부분 콘텐츠 보존 및 비콘텐츠 보존으로 나누어 다룹니다. 총 70개 이상의 다양한 I2I 모델과 30개 이상의 데이터셋을 분석하고, 평가 방법에 대한 새로운 기준을 제시합니다. 이러한 분류와 분석을 통해, I2I 변환의 여러 적용 가능성을 탐구하고, 특정 응용 프로그램을 위한 적합한 I2I 모델 선택 시 고려할 사항을 강조합니다.

- **Technical Details**: 본 논문은 이미지 변환 과정에서 콘텐츠 보존(Content Preservation)의 중요성과 이를 보장하기 위한 여러 Generative 모델(생성 모델)들을 소개합니다. 두 가지 주요 접근법인 Supervised와 Unsupervised I2I 변환을 구분하고, 특히 Unsupervised 방법에 중점을 두어 설명합니다. Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), Diffusion Models, Flow-based Models, Transformer Models 등 다양한 모델의 구조와 특징을 체계적으로 정리하며, 각 모델의 장단점을 논의합니다.

- **Performance Highlights**: 이번 연구는 Sim-to-Real 번역 문제를 위한 평가 기준을 제시하고, 실제 이미지를 생성하는 데 있어 I2I 모델의 성능을 강화하는 방안을 모색합니다. 각 I2I 모델의 성능을 다양한 기준을 통해 평가하고, 일반적인 I2I 변환에서 필요한 데이터와 작업, 평가 지표를 정리했습니다. 마지막으로, 데이터 부족 시에도 좋은 성능을 발휘할 수 있는 전략으로써, 시뮬레이션된 데이터의 실제 이미지로의 변환 과정을 강조합니다.



### Poly-Autoregressive Prediction for Modeling Interactions (https://arxiv.org/abs/2502.08646)
Comments:
          preprint

- **What's New**: 이번 논문에서는 다중 에이전트 환경에서 에이전트의 행동을 예측하기 위한 간단한 프레임워크인 Poly-Autoregressive (PAR) 모델링을 소개합니다. 기존의 오토 회귀(autoregessive, AR) 작업과 달리, 우리의 접근 방식은 물리적 제약 조건과 내부 동기에 의해 형성되는 여러 에이전트 간의 상호작용을 중점적으로 다룹니다. PAR은 자아 에이전트의 상태 역사와 다른 interacting agent의 과거 및 현재 상태를 기반으로 미래 행동을 예측합니다.

- **Technical Details**: PAR 프레임워크는 변환기(transformer) 기반 모델을 사용하여 다음 토큰을 예측합니다. 이 모델은 실제 상호작용이 발생하는 복잡한 환경에서 여러 에이전트 간의 상호작용을 포착하는 데 초점을 맞추고 있습니다. 본 연구에서는 사회적 행동 예측, 자율주행 차량의 궤적 예측, 손-물체 상호작용 중 물체 포즈 예측을 포함한 세 가지 실제 문제에 PAR 모델을 적용하였습니다.

- **Performance Highlights**: 본 연구에서는 PAR 모델이 AR 보다 모든 경우에서 월등한 성능을 발휘함을 보여줍니다. 예를 들어, 사회적 행동 예측에서는 PAR이 AR에 비해 +1.9의 mAP 개선을 보였으며, 자율주행 차량의 궤적 예측에서도 6.3%의 ADE 개선을 나타냈습니다. 손-물체 상호작용에서 PAR은 물체의 회전 및 변환 예측에서 각각 8.9% 및 41.0%의 성능 향상을 이루었습니다.



### SwiftSketch: A Diffusion Model for Image-to-Vector Sketch Generation (https://arxiv.org/abs/2502.08642)
Comments:
this https URL

- **What's New**: 이번 연구에서는 SwiftSketch라는 새로운 확산 모델(dispersion model)을 도입하여 이미지 기반 벡터 스케치 생성(image-conditioned vector sketch generation)을 가능하게 합니다. SwiftSketch는 높은 품질의 스케치를 1초 이내에 생성할 수 있으며, 이는 기존의 방법들에 비해 시간 효율성이 크게 개선된 것입니다. 향상된 스케치 품질과 속도를 통해 실용적인 응용이 가능해집니다.

- **Technical Details**: SwiftSketch는 가우시안 분포(Gaussian distribution)에서 샘플링한 스트로크 제어 점(stroke control points)을 점진적으로 디노이징(denoising)해서 작업합니다. 이 모델은 변환기-디코더(transformer-decoder) 아키텍처를 사용하여 벡터 표현의 이산적(discrete) 특성을 효과적으로 처리하고, 스트로크 간의 전반적인(global) 종속성을 포착합니다. 또한, ControlSketch라는 방법을 통해 기존 스케치 데이터셋의 한계를 극복하고, 깊이 인식(ControlNet)을 통한 정밀한 공간 제어를 가능하게 하여 SDS 기반 기술(SDS-based techniques)을 개선합니다.

- **Performance Highlights**: SwiftSketch는 다양한 개념에 대해 일반화(generalization)를 잘 하며, 높은 충실도(fidelity)와 자연스럽고 시각적으로 매력적인 스타일을 결합하여 스케치를 효율적으로 생성할 수 있습니다. 생성된 스케치들은 전문가 수준의 품질을 가지며, 비전문가가 만든 기존 데이터셋의 한계를 보완합니다. 이로 인해 SwiftSketch는 다양한 애플리케이션에 활용될 가능성이 높습니다.



### CineMaster: A 3D-Aware and Controllable Framework for Cinematic Text-to-Video Generation (https://arxiv.org/abs/2502.08639)
- **What's New**: CineMaster는 3D 인식 및 제어가 가능한 텍스트-비디오 생성(T2V) 프레임워크로, 사용자가 전문 영화 감독과 유사한 수준의 제어력을 가질 수 있도록 설계되었습니다. 이 프레임워크는 두 단계로 작동하며, 첫 번째 단계에서는 사용자가 3D 공간 내에서 객체의 바운딩 박스를 배치하고 카메라 이동을 정의하여 직관적으로 조건 신호를 구축할 수 있습니다. 두 번째 단계에서는 생성된 조건 신호를 바탕으로 비디오를 생성하는 확산 모델을 활용합니다.

- **Technical Details**: CineMaster의 첫 번째 단계는 인터랙티브한 워크플로우를 통해 사용자가 3D 인식 제어 신호를 구성하는 것입니다. 여기서 3D 바운딩 박스를 주요 객체 표현 형식으로 사용하며, 사용자는 키프레임을 통해 객체와 카메라 위치를 재조정할 수 있습니다. 두 번째 단계에서는 이 제어 신호들이 T2V 모델의 조건으로 작용하여 원하는 비디오 콘텐츠를 합성하게 되며, 모든 프레임의 렌더링 깊이 맵이 추가적인 시각적 단서를 제공합니다.

- **Performance Highlights**: CineMaster는 기존의 방법들과 비교했을 때 상당한 성능 향상을 보여주며, 텍스트-비디오 생성 분야에서 주목받는 성과를 달성하였습니다. 특히, 영화 제작자가 3D 공간에서 촬영 계획을 세우는 방식으로 사용자에게 복잡한 동적 움직임 다이내믹스를 조정하는 능력을 제공합니다. 또한, 3D 객체 움직임과 카메라 자세 주석이 포함된 대규모 데이터셋을 구축하기 위한 자동 데이터 주석 파이프라인을 도입하여 데이터의 희소성 문제를 해결하였습니다.



### PulseCheck457: A Diagnostic Benchmark for 6D Spatial Reasoning of Large Multimodal Models (https://arxiv.org/abs/2502.08636)
- **What's New**: 이 논문에서는 PulseCheck457라는 새로운 합성 데이터셋을 소개합니다. 이 데이터셋은 대형 다중 모달 모델(LMMs)의 6D 공간 추론 능력을 평가하기 위해 설계되었습니다. 기존의 2D 이해에 중점을 둔 벤치마크의 한계를 극복하고, 3D 위치와 방향을 포함하는 보다 포괄적인 공간 추론 평가 틀을 제공합니다.

- **Technical Details**: PulseCheck457는 다중 물체 인식, 2D 위치, 3D 위치, 3D 방향의 네 가지 핵심 능력을 중심으로 구성되어 있습니다. 이 데이터셋은 기본적인 단일 객체 인식에서 고급 6D 공간 추론 작업에 이르기까지 다섯 가지 난이도 레벨로 질문 유형을 구성하여 각 레벨의 성능을 평가합니다. 이 구조적 접근 방식은 각 모델의 강점과 취약점을 드러내는 데 도움을 줍니다.

- **Performance Highlights**: PulseCheck457에서 여러 LMMs의 성능을 분석한 결과, 작업의 복잡성이 증가함에 따라 일반적으로 성능이 저하되었습니다. 특히 3D 및 6D 공간 작업에서 이러한 경향이 두드러졌습니다. 새로운 상대 성능 저하율(RPDR) 지표를 도입하여 3D 추론 능력의 약점을 정량화하고, 다양한 속성 간의 예측 편향을 발견하여 실제 이미지 설정에서도 유사한 패턴이 나타나는 것을 확인했습니다.



### Light-A-Video: Training-free Video Relighting via Progressive Light Fusion (https://arxiv.org/abs/2502.08590)
Comments:
          Project Page: this https URL

- **What's New**: Light-A-Video는 비디오 조명 재구성(video relighting) 분야에서 새로운 접근 방식을 제안합니다. 전통적인 이미지 조명 재구성 모델의 한계를 극복하기 위해, 이 모델은 훈련이 필요 없는 영상 조명 변경 솔루션을 제공하여 고품질의 부드러운 비디오 결과를 생성할 수 있도록 지원합니다. 이 연구에서는 Consistent Light Attention(CLA) 모듈과 Progressive Light Fusion(PLF) 전략을 활용하여 일관된 조명 소스를 유지하는데 집중합니다.

- **Technical Details**: Light-A-Video는 두 가지 주요 기술인 CLA 모듈과 PLF 전략을 도입하여 시간적으로 일관된 비디오 조명을 구현합니다. CLA 모듈은 자기 주의(self-attention) 층 내에서 프레임 간의 상호작용을 강화하여 배경 조명의 생성 과정을 안정화합니다. PLF 전략은 원본 비디오의 모양과 재조명되는 모양 간의 선형 혼합을 통해 부드러운 조명 전환을 보장합니다.

- **Performance Highlights**: 실험 결과, Light-A-Video는 재조명된 비디오의 시간적 일관성을 개선하면서 이미지 품질을 유지합니다. 이 모델은 전체 입력 비디오의 재조명뿐만 아니라 입력 전경 시퀀스의 배경 생성을 지원하여 다양한 설정에서 효율성을 보여줍니다. 그 결과, 고품질의 조명이 일관된 비디오가 생성되어, 영화 제작, 게임, 가상 환경 등 다양한 분야에서 활용될 수 있습니다.



### Ultrasound Image Generation using Latent Diffusion Models (https://arxiv.org/abs/2502.08580)
Comments:
          6 pages conference paper for SPIE medical imaging

- **What's New**: 이 논문은 고성능의 Diffusion 모델을 사용하여 실제적인 초음파 이미지를 생성하기 위한 탐색을 제안합니다. 전통적인 물리 기반 모델과는 달리, 대규모 공개 데이터베이스에서의 점진적인 파인튜닝을 통해 초음파 이미지를 생성했습니다. 최종적으로는 사용자에게 세분화를 조건으로 이미지 생성 과정을 조정할 수 있는 기능을 제공했습니다.

- **Technical Details**: Stable Diffusion 모델을 활용하여 초음파 이미지를 고품질로 생성하였으며, 이는 클립 조건화(CLIP conditioning)를 통해 텍스트 프롬프트로부터 이미지를 생성하는 접근 방식입니다. 데이터세트로는 780개의 초음파 이미지로 구성된 BUSI 데이터세트를 사용하였고, U-NET 구조를 통해 노이즈를 감소시키며 고해상도 샘플을 생성하는 방법론을 적용했습니다.

- **Performance Highlights**: 모델의 성능을 평가하기 위해 Resnet-50을 사용하는 분류 네트워크를 훈련시켰습니다. 생성된 이미지로 데이터셋을 보강하였을 때 AUC가 81%에서 87%로 증가했으며, 20%의 BUSI 데이터셋만으로 훈련한 Resnet-50은 94% AUC에 도달했습니다. 이는 생성된 이미지를 통한 훈련이 분류 성능을 향상시킬 수 있다는 것을 보여줍니다.



### A Novel Approach to for Multimodal Emotion Recognition : Multimodal semantic information fusion (https://arxiv.org/abs/2502.08573)
- **What's New**: 이 논문에서는 DeepMSI-MER이라는 새로운 멀티모달 감정 인식 방법을 제안합니다. 이 방법은 대조 학습(contrastive learning)과 시각적 시퀀스 압축(visual sequence compression)을 통합하여 구현되었습니다. 제안된 방안은 서로 다른 모달리티에서의 특징 융합(cross-modal feature fusion)을 강화하며, 시각적 모달리티의 중복성을 줄이는 데 도움을 줍니다.

- **Technical Details**: DeepMSI-MER은 세 가지 단계로 구성됩니다: 모달리티 특화(feature extraction), 초기(feature fusion) 및 최종(feature fusion) 융합 그리고 모델 예측(model prediction)입니다. 이 과정에서 BERT 및 Wav2Vec 모델을 미세 조정하여 텍스트 및 오디오 데이터에서 의미 특징을 추출하고, 이를 시각적 모달리티 비디오 특성과 융합합니다. 최종적으로, Temporal Convolution Networks (TCN)를 통해 시계열 특징을 캡처하여 최종 비디오 특징을 생성합니다.

- **Performance Highlights**: IEMOCAP 및 MELD의 두 가지 공개 데이터셋에서 실험 결과, DeepMSI-MER는 멀티모달 감정 인식의 정확성과 강인성을 크게 개선했습니다. 이러한 결과는 멀티모달 특징 융합과 제안된 접근법의 유효성을 입증합니다. DeepMSI-MER는 특히 다양한 현실 세계 응용에서의 감정 인식 성능을 향상시켜 나갈 것으로 기대됩니다.



### Brain Latent Progression: Individual-based Spatiotemporal Disease Progression on 3D Brain MRIs via Latent Diffusion (https://arxiv.org/abs/2502.08560)
Comments:
          arXiv admin note: text overlap with arXiv:2405.03328

- **What's New**: 본 논문에서는 Brain Latent Progression (BrLP)이라는 새로운 공간-시간 모델을 제안합니다. 이는 3D 뇌 MRI에서 개인의 질병 진행을 예측하기 위해 설계되었습니다. BrLP는 낮은 차원의 잠재 공간에서 작동하며, 개인화된 예측을 위한 주체 메타데이터를 통합하였습니다.

- **Technical Details**: BrLP는 Latent Diffusion Model (LDM)과 ControlNet을 결합하여 주어진 개인의 데이터를 바탕으로 개인화된 뇌 MRI를 생성합니다. 또한, 악세서리 모델을 통해 질병 역학에 대한 사전 지식을 통합하여, 가용한 경우 장기적 데이터를 활용할 수 있게 하였습니다. Latent Average Stabilization (LAS) 기법을 도입하여 예측의 공간-시간 일관성을 확보하고, 처리 메모리 요구를 줄입니다.

- **Performance Highlights**: 11,730개의 T1w 뇌 MRI를 사용하여 BrLP를 훈련하고 평가하였으며, 2,257개의 외부 테스트 세트를 통해 일반화 능력을 검증했습니다. 실험 결과, BrLP가 생성한 MRI 스캔은 기존 방법들과 비교해 최첨단 정확도를 나타냈습니다. 코드는 공개적으로 이용 가능합니다.



### Human-Centric Foundation Models: Perception, Generation and Agentic Modeling (https://arxiv.org/abs/2502.08556)
Comments:
          9 pages

- **What's New**: 최근 인간 중심의 기초 모델(Human-centric Foundation Models, HcFMs)이 다채로운 인간 중심 작업을 단일 프레임워크로 통합하여 전통적인 작업별 접근 방식을 초월하고 있다는 점에서 주목할 만한 변화가 있습니다. 이러한 모델은 인간의 외모, 감정, 정체성 및 행동을 보다 정교하게 이해하고 생성할 수 있는 가능성을 열어줍니다. 분야의 발전은 연구자들이 인간을 보다 포괄적이고 복잡한 시스템으로 이해해야 한다는 요구와 함께 진행되고 있습니다.

- **Technical Details**: 이 논문에서는 HcFMs를 네 가지 카테고리로 분류하는 새로운 분류법을 제안합니다: (1) 인간 중심의 인식 기초 모델, (2) 인간 중심의 AIGC 기초 모델, (3) 통합 인식 및 생성 모델, (4) 인간 중심의 에이전틱 모델입니다. 각 모델은 그들이 지원하는 다양한 하위 작업에 따라 분류되어, 인간 중심의 데이터로부터 효율적으로 학습할 수 있도록 돕습니다.

- **Performance Highlights**: HcFMs는 기존의 작업별 모델보다 높은 일반화 능력과 적용 가능성, 그리고 사실감을 보증합니다. 이 모델들은 특히 2D와 3D 작업에서의 성능을 개선하며, 인간의 행동과 상호작용을 보다 정밀하게 표현할 수 있는potential을 지니고 있습니다. 또, 이 모델들은 다중 모달리티(multi-modality)를 활용하여 더욱 풍부한 인식을 가능하게 하며, 사용자와의 상호작용 작업에도 적용될 수 있습니다.



### Copula-based mixture model identification for subgroup clustering with imaging applications (https://arxiv.org/abs/2502.08549)
- **What's New**: 본 논문은 Copula-Based Mixture Models (CBMMs)를 제안하여 데이터 클러스터링을 위한 보다 유연한 접근법을 제시합니다. 기존의 혼합 모델은 고정된 분포 형태를 가지는 것이 일반적이지만, CBMMs는 다양한 마진 및 코풀라 형태를 허용하여 이 질병별로 이질적인 분포를 처리할 수 있습니다. 특히, Generalized Iterative Conditional Estimation (GICE) 알고리즘의 적응을 통해 CBMM을 비지도 학습 방식으로 식별하는 방법을 제안합니다.

- **Technical Details**: CBMM에서는 각 구성 요소가 서로 다른 마진과 의존 구조를 가질 수 있습니다. 이는 Sklar의 정리에 따라 가능한가 하며, 논문에서는 CBMM의 최적화를 위해 GICE 알고리즘을 사용하여 각 구성 요소의 마진 및 코풀라 형태를 반복적으로 추정합니다. CBMM의 데이터 적합성 및 수렴을 검증하기 위해 합성 데이터와 실제 데이터를 사용하여 실험을 수행하였습니다.

- **Performance Highlights**: CBMM-GICE 클러스터링 방법은 합성 데이터와 MNIST 데이터베이스, 실제 심장 자기공명 영상에 대한 실험을 통해 성능을 평가하였습니다. 이 방법은 특히 의료 이미징 애플리케이션에서의 유용성을 보여주며, 기존의 Expectation Maximization (EM)으로 식별된 혼합 모델과 비교되어 개선된 결과를 나타냅니다. 이러한 접근법은 다차원 데이터 분석에 대한 더 나은 유연성과 적합성을 제공합니다.



### Moment of Untruth: Dealing with Negative Queries in Video Moment Retrieva (https://arxiv.org/abs/2502.08544)
Comments:
          16 pages, 9 figures. Accepted at WACV 2025. Paper webpage: this https URL

- **What's New**: 본 논문에서는 비디오 모멘트 검색(Video Moment Retrieval) 문제에서 음성 쿼리의 부정적인 영향(negative queries)을 다루는 새로운 접근 방식인 Negative-Aware Video Moment Retrieval (NA-VMR)를 제안합니다. 기존 모델이 irrelevant 한 쿼리에 대해 moment를 잘못 예측하는 문제를 해결하기 위해, ID(내부 도메인)와 OOD(외부 도메인) 부정 쿼리에 대한 새로운 평가 벤치마크를 소개합니다. 이 연구는 비디오 모멘트 검색 모델이 부정 쿼리를 처리할 수 있는지를 실험적으로 분석하고, 효과적으로 부정 쿼리를 거부하는 방법을 제시합니다.

- **Technical Details**: 연구의 주요 초점은 부정 쿼리에 대해 robust한 비디오 모멘트 검색 방법론을 개발하는 것입니다. 이를 위해 기존 방법을 확장하여 추가적인 모듈을 학습시키고, 해당 문장이 비디오와 관련이 있는지를 판단하도록 하였습니다. 이러한 방법론은 ID와 OOD 부정 쿼리 모두에서 훈련되며, 평균 98.4%의 높은 음성 거부 정확도(Negative Rejection Accuracy)를 달성하면서도 여전히 $3.87	ext{%}$의 Recall@1 성능을 유지합니다.

- **Performance Highlights**: NA-VMR을 통해 제안된 UniVTG-NA 모델은 비디오 모멘트 검색 성능을 유지하면서도 부정적인 문장을 효과적으로 저지하는 능력을 보여주었습니다. 이 모델은 특히 새로운 평가 기준을 적용함으로써 기존의 상태(State-of-the-art) 모델들보다 우수한 성능을 기록했습니다. 본 연구는 비디오 모멘트 검색의 새로운 기준을 제시하고, AI 모델의 신뢰성과 설명 가능성에 기여할 수 있는 중요한 결과를 도출하였습니다.



### A Survey on Image Quality Assessment: Insights, Analysis, and Future Outlook (https://arxiv.org/abs/2502.08540)
- **What's New**: 이 논문은 이미지 품질 평가(IQA) 방법론의 최신 발전을 종합적으로 분석하여 다양한 응용 시나리오에 따른 IQA 방식의 인증 기준을 제시합니다. 기존 연구보다 다양한 응용 시나리오를 아우르는 체계적인 리뷰를 제공하며, 최신 딥러닝 모델과 기계 학습 기법을 포함한 IQA 방법들에 대한 깊이 있는 논의를 다룹니다. 또한, 왜곡별 IQA 방법의 필요성과 향후 연구 방향을 제안합니다.

- **Technical Details**: IQA는 주관적 IQA(SIQA)와 객관적 IQA(OIQA)로 분류됩니다. SIQA는 인간 평가자에 의존하며 참조 이미지의 존재 여부에 따라 추가로 세분화됩니다. OIQA는 자동화 과정으로, 인간의 개입 없이 손상된 이미지를 사용하여 평균 의견 점수(MOS)를 통해 평가됩니다. OIQA 방법은 전체 참조(FR), 축소 참조(RR), 비참조(NR)로 나뉘며, 특히 NR 방법은 다양한 응용 프로그램에서 연구의 초점이 되고 있습니다.

- **Performance Highlights**: 이 논문에서는 이미지 품질 분석을 위해 통계적 방법과 기계 학습 기반 방법으로 IQA 방법을 정리합니다. 통계적 방법은 픽셀의 중요성을 고려한 VSNR, PSNR-HVS와 같은 방식으로 HVS를 통합하여 인간 인지와의 간극을 줄이려 합니다. 기계 학습 기반 방법은 다양한 품질 평가 기술을 통합하며, CNN 및 Transformer 기반의 최신 접근법을 통한 성능 향상 가능성을 탐구하고 있습니다.



### Referring Remote Sensing Image Segmentation via Bidirectional Alignment Guided Joint Prediction (https://arxiv.org/abs/2502.08486)
- **What's New**: 이번 연구에서는 리모트 센싱 이미지 세분화(Refering Remote Sensing Image Segmentation, RRSIS)에서 비전-언어 간의 간극을 줄이고, 다중 스케일 피처 상호 작용을 향상시키며, 세밀한 객체 구분을 개선하기 위해 새로운 프레임워크인 BTDNet을 제안합니다. BTDNet은 Bidirectional Spatial Correlation(BSC), Target-Background TwinStream Decoder(T-BTD), Dual-Modal Object Learning Strategy(D-MOLS) 등의 혁신적인 요소를 포함하여 복잡한 리모트 센싱 환경에서의 세분화 성능을 향상시킵니다.

- **Technical Details**: BTDNet은 리모트 센싱 이미지에서 비전과 언어의 양방향 특징 정렬을 지원하기 위해 설계되었습니다. BSC는 비전 및 언어 모달리티 간의 효과적인 상호작용을 촉진하고, T-BTD 디코더는 목표와 비목표를 구별하는 데 있어 마스킹된 텍스트 정보를 활용합니다. D-MOLS는 이질적인 비전-언어 데이터를 재구성하여 모달리티 간의 정렬을 개선하며, 복잡한 환경에서의 성능을 높입니다.

- **Performance Highlights**: BTDNet은 RefSegRS와 RRSIS-D 두 가지 데이터셋에서 각각 3.76% 및 1.44%의 전반적인 IoU(Overall IoU)를 향상시켰으며, 평균 IoU(Mean IoU)에서도 각각 5.37% 및 1.84% 개선되었습니다. 이러한 성능은 리모트 센싱 이미지의 복잡함에도 불구하고 막강한 분할 능력을 입증하였으며, 다양한 리모트 센싱 시나리오에서 강력한 성능을 보였습니다.



### mmE5: Improving Multimodal Multilingual Embeddings via High-quality Synthetic Data (https://arxiv.org/abs/2502.08468)
- **What's New**: 이 논문은 다양한 모달리티를 통합하여 고품질 합성 데이터를 생성하는 새로운 접근 방식을 제안합니다. 고품질의 합성 멀티모달 데이터에는 세 가지 주요 기준이 필요하며, 이는 넓은 범위, 강력한 크로스 모달 정렬, 높은 충실도를 포함합니다. 연구진은 이를 통해 mmE5라는 다국어 멀티모달 모델을 훈련하고 여러 벤치마크에서 뛰어난 성능을 달성했습니다.

- **Technical Details**: 연구는 다단계 방법론을 통해 고품질의 멀티모달 데이터를 합성합니다. 첫 번째로, MLLM을 사용하여 입력 이미지를 다양한 관점에서 분석하고 설명을 생성합니다. 그 후, MLLM은 합성된 텍스트 데이터를 다시 평가하여 크로스 모달 정렬과 충실도를 향상시킵니다. 이 방법론은 실세계 이미지와 관련된 텍스트를 결합하여 뉴스 데이터를 생성하는데 중점을 두고 있습니다.

- **Performance Highlights**: mmE5 모델은 MMEB 벤치마크에서 최첨단 성능을 기록하며, 이전 모델에 비해 훈련 데이터가 45배 적은 상태에서도 뛰어난 결과를 보였습니다. 또한, 비즈니스와 다양한 언어에 대한 강력한 성능을 보여주며 XTD 벤치마크에서도 최고 성과를 달성했습니다. 이로 인해 mmE5는 멀티모달 임베딩 모델의 새로운 기준을 세우고 있습니다.



### Composite Sketch+Text Queries for Retrieving Objects with Elusive Names and Complex Interactions (https://arxiv.org/abs/2502.08438)
Comments:
          Accepted at AAAI 2024, 9 pages. Project Website: this https URL

- **What's New**: 이번 논문은 비원어민 사용자들이 매우 구체적인 객체 이름을 찾는 데 어려움을 겪는 문제를 다루고 있습니다. 특히, 손으로 그린 스케치와 어려운 이름을 구술하는 텍스트를 조합한 복합적인 멀티모달 쿼리를 수용하는 검색 인터페이스를 요구하는 사례를 설명합니다. 기존의 텍스트 기반 이미지 검색(TBIR)과 스케치 기반 이미지 검색(SBIR) 문제와는 다른 새로운 문제 설정인 CSTBIR을 제안하고 있습니다.

- **Technical Details**: 연구에서는 약 200만 개의 쿼리와 10만 8000개의 자연 장면 이미지로 구성된 CSTBIR 데이터셋을 커리팅하였습니다. 이 문제의 해결책으로 제안된 STNET(스케치 및 텍스트 네트워크)은 손으로 그린 스케치를 활용하여 자연 장면 이미지에서 관련 객체를 찾고, 텍스트와 이미지를 인코딩하여 이미지 검색을 수행합니다. 모델은 대비 학습과 여러 훈련 목표를 통해 성능을 향상시킬 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 텍스트만, 스케치만, 복합 쿼리 방식 모두에서 여러 최신 검색 방법을 초월하는 성과를 나타냈습니다. 본 논문은 정교하고 복잡한 쿼리를 처리할 수 있는 CSTBIR 시스템을 통해 많은 분야에서 이미지 검색의 새로운 가능성을 제시합니다. 연구 결과물은 프로젝트 웹사이트에서 데이터셋과 코드를 공개하고 있습니다.



### Handwritten Text Recognition: A Survey (https://arxiv.org/abs/2502.08417)
- **What's New**: 최근 핸드라이팅 텍스트 인식(HTR) 기술은 과거의 휴리스틱(heuristic) 방식에서 딥러닝(deep learning) 기반의 현대적 신경망 모델로 진화했습니다. 본 설문조사는 HTR 시스템의 발전을 살펴보며, 주로 문자 단위(word-level) 및 단락 단위(paragraph-level) 인식의 두 가지 주요 수준으로 분류하여 다양한 접근 방식을 제공하고 있습니다. 이를 통해 현재 HTR 연구의 흐름과 발전 방향에 대한 포괄적인 분석을 제시합니다.

- **Technical Details**: HTR의 복잡성은 사람의 다양한 필기 스타일에서 비롯되며, 이러한 변동성을 처리하기 위한 알고리즘의 필요성이 있습니다. 초기 HTR 시스템은 주로 HMM(Hidden Markov Models)과 같은 통계적 방법에 의존하였으나, 최신 기술들은 CNN(Convolutional Neural Networks) 및 RNN(Recurrent Neural Networks)과 같은 심층 학습 모델의 발전을 통해 놀라운 성과를 거두고 있습니다. 이 설문에서는 최근 Transformer 아키텍처의 적용 및 발전도 함께 다루고 있습니다.

- **Performance Highlights**: HTR 기술은 최근 몇 년 간 심층 신경망의 발전과 다양한 데이터셋의 증가 덕분에 크게 향상되었습니다. 다양한 분야의 연구 결과를 공동 기준에 따라 평가하고, HTR 시스템의 성능을 비교하여 기술 발전의 흐름을 정리합니다. 이러한 분석을 통해 연구자와 실무자에게 향후 연구 방향에 대한 로드맵을 제공합니다.



### ViLa-MIL: Dual-scale Vision-Language Multiple Instance Learning for Whole Slide Image Classification (https://arxiv.org/abs/2502.08391)
Comments:
          CVPR 2024 (Updated version with corrections for typos and errors.)

- **What's New**: 본 논문에서는 이중 규모 비전-언어 다중 인스턴스 학습(dual-scale vision-language multiple instance learning, ViLa-MIL) 프레임워크를 제안하여 whole slide image (WSI) 분류 문제를 해결하고자 합니다. 기존의 VLM 기반 방법의 한계를 극복하고, 병리학적 선행 지식을 반영한 텍스트 프롬프트를 도입하여 모델 성능을 높이는 것이 목표입니다. 특히, 저해상도와 고해상도 WSI에서 각각 다른 텍스트 프롬프트를 생성하여 각기 다른 정보를 제공합니다.

- **Technical Details**: ViLa-MIL 프레임워크는 두 가지 경량화된 디코더를 활용하여 이미지와 텍스트 기능을 효과적으로 처리합니다. 이미지 브랜치에서는 유사한 패치 특성을 동일한 프로토타입으로 그룹화하여 패치 기능을 집적하는 프로토타입 기반 패치 디코더(prototype-guided patch decoder)를 제안합니다. 텍스트 브랜치에서는 멀티 그래뉼러 이미지 컨텍스트(multi-granular image contexts)를 활용하여 텍스트 기능을 향상시키는 컨텍스트 기반 텍스트 디코더(context-guided text decoder)를 도입합니다.

- **Performance Highlights**: ViLa-MIL은 세 개의 다중 암종 및 다중 센터 서브타입 데이터셋을 사용한 실험에서 기존의 최신 MIL 기반 방법들보다 1.7-7.2% 더 높은 AUC(Area Under Curve)와 2.1-7.3% 더 높은 F1 스코어를 달성하여 성능 우수성을 입증하였습니다. 각 모듈의 효과성을 평가하기 위한 다양한 블라벨 및 매개변수 민감도 분석(a series of ablation studies and parameter sensitivity analyses)도 진행되어 각 모듈이 기능적으로 기여하는 바를 확인하였습니다.



### Not All Frame Features Are Equal: Video-to-4D Generation via Decoupling Dynamic-Static Features (https://arxiv.org/abs/2502.08377)
- **What's New**: 이 논문에서는 비디오에서 동적 3D 객체를 생성하는 새로운 접근법인 DS4D를 제안합니다. DS4D는 동적(static) 및 정적(dynamic) 정보를 시간 및 공간 축을 따라 분리하여 고품질 4D 콘텐츠 생성을 위한 동적 표현을 강화합니다. 특히, 동적-정적 특징 분리 모듈(DSFD)과 시간-공간 유사성 융합 모듈(TSSF)을 도입하여 기존 방법이 간과했던 동적 정보를 효과적으로 처리합니다.

- **Technical Details**: DS4D는 먼저 비디오의 각 프레임에서 제로-123++를 사용해 유사 멀티 뷰 이미지를 추정한 다음, 이 프레임 시퀀스로부터 특징을 추출합니다. DSFD는 현재 프레임의 특징과 기준 프레임의 특징 간의 현저한 차이를 통해 동적 특징을 분리하고, TSSF는 이러한 동적 특징을 기반으로 정보가 격자에서 통합되는 과정을 통해 더욱 풍부한 동적 표현을 구현합니다.

- **Performance Highlights**: 실험 결과, DS4D는 Consistent4D와 Objaverse 데이터셋에서 다른 최첨단 방법들(SOTA)보다 비디오 품질, 모션 충실도, 시간-공간 일관성에서 더 나은 성능을 보여주었습니다. 실제 시나리오 데이터셋에 대한 실험을 통해 4D 장면에서도 효과성을 입증하였습니다.



### AdvSwap: Covert Adversarial Perturbation with High Frequency Info-swapping for Autonomous Driving Perception (https://arxiv.org/abs/2502.08374)
Comments:
          27th IEEE International Conference on Intelligent Transportation Systems (ITSC)

- **What's New**: 이 논문에서는 자율차량(Autonomous Vehicles)의 인식 모듈이 인공지능 안전성을 해칠 수 있는 적대적 공격에 취약하다는 점을 강조합니다. 기존의 글로벌 노이즈 기법이 감지 가능하다는 단점을 극복하기 위해, 새로운 적대적 공격 방법인 AdvSwap을 제안합니다. 이 방법은 파동 변환(wavelet) 기반의 고주파 정보 스와핑을 창의적으로 활용하여 은밀한 적대적 샘플을 생성합니다.

- **Technical Details**: AdvSwap은 선택적인 고주파 정보 스와핑을 위해 가역 신경망(invertible neural network)을 활용합니다. 이 방식은 데이터의 무결성과 함께 순방향 전파(forward propagation)를 보존합니다. 원래의 레이블 데이터를 효과적으로 제거하고 가이드 이미지 데이터로 대체하여, 은폐된 강력한 적대적 샘플을 생성합니다.

- **Performance Highlights**: GTSRB 및 nuScenes 데이터셋에 대한 실험 평가 결과, AdvSwap은 일반적인 교통 목표에 대해 은밀한 공격을 수행할 수 있음을 보여줍니다. 생성된 적대적 샘플은 인간과 알고리즘 모두에게 인식하기 어렵고, 공격 내구성(strong attacking robustness)과 공격 전이 가능성(attacking transferability) 또한 매우 높습니다.



### Uncertainty Aware Human-machine Collaboration in Camouflaged Object Detection (https://arxiv.org/abs/2502.08373)
- **What's New**: 이번 연구에서는 Camouflaged Object Detection (COD) 분야에서 신뢰할 수 있는 시스템을 개발하기 위해 불확실성 추정 및 효율적인 활용 방안을 제안합니다. 컴퓨터 비전(CV) 모델과 비침습 뇌-컴퓨터 인터페이스(BCI)의 강점을 활용하여 인간-기계 협력 프레임워크를 구축하였습니다. 특히, 중첩된 시야 멀티뷰(backbone)를 통해 CV 모델의 예측 불확실성을 추정하고, 훈련 시 이를 효과적으로 활용하여 시스템 신뢰도를 높이고자 합니다.

- **Technical Details**: 제안하는 프레임워크는 CAMO 데이터세트에서 성능을 평가하였고, 기존 방법보다 4.56% 향상된 균형 정확도(Balanced Accuracy, BA) 및 3.66% 향상된 F1 점수를 기록하여 최첨단 결과를 도출했습니다. 훈련 과정에서 불확실성 측정과 정밀도 간의 강한 상관관계를 확인하였고, 제안된 훈련 정책과 인간-기계 협력 전략의 효과를 검증하는 ablation study(제외 연구)를 수행했습니다.

- **Performance Highlights**: 우수한 성과를 보인 참가자의 경우 기존 방법 대비 BA에서 7.6%, F1 점수에서 6.66%의 향상을 달성하였습니다. 이 연구는 신뢰할 수 있는 시스템을 통해 인간의 인지 부담을 줄이고 시스템의 신뢰성을 향상시켜 실제 COD 애플리케이션 및 인간-컴퓨터 상호작용 발전에 기여할 수 있는 강력한 기반을 제공합니다.



### Sat-DN: Implicit Surface Reconstruction from Multi-View Satellite Images with Depth and Normal Supervision (https://arxiv.org/abs/2502.08352)
- **What's New**: 이 논문에서는 위성 이미지로부터 고해상도 3D 표면 재구성을 수행하기 위해 Sat-DN이라는 새로운 프레임워크를 제안합니다. Sat-DN은 점진적으로 훈련된 다중 해상도 해시 그리드 구조와 명시적인 깊이 가이드 및 표면 법선 일관성 제약 조건을 활용하여 재구성 품질을 향상시킵니다. 이러한 접근 방식은 기존의 방법들이 가지고 있는 높은 계산 비용과 재구성 품질의 한계를 극복하는 데 기여합니다.

- **Technical Details**: 제안된 Sat-DN은 다중 해상도 해시 그리드를 사용하여 학습 속도를 크게 향상시킵니다. 점진적인 학습 전략을 통해 저주파수 기하학을 사용하여 고주파수 세부 사항의 재구성을 유도하여 훈련 과정을 최적화합니다. 또한, 깊이와 법선 제약 조건을 통해 명확한 건물 윤곽을 유지하고 평면 분포가 올바르게 유지되도록 지원합니다.

- **Performance Highlights**: DFC2019 데이터셋에 대한 광범위한 실험 결과에 따르면 Sat-DN은 기존의 다른 방법들을 초월하여 정성적 및 정량적 평가 모두에서 최첨단 성과를 달성했습니다. 이는 재구성된 메쉬와 디지털 표면 모델(DSM)의 정확성을 높이는 데 크게 기여하였습니다. 코드와 결과는 해당 URL에서 활용 가능합니다.



### Hi-End-MAE: Hierarchical encoder-driven masked autoencoders are stronger vision learners for medical image segmentation (https://arxiv.org/abs/2502.08347)
Comments:
          19 pages, Code: this https URL

- **What's New**: 이 논문에서는 의료 이미지 분할을 위한 새로운 사전 훈련 방법인 Hierarchical Encoder-driven MAE(Hi-End-MAE)를 제안합니다. Hi-End-MAE는 인코더 중심의 재구성을 통해 인코더의 정보량을 증가시키고, 계층적 밀집 디코딩을 통해 여러 ViT 계층에서 풍부한 정보를 포착하려고 합니다. 이를 통해 현재 ViT 기반의 MIM 사전 훈련 프레임워크의 한계를 극복하고자 합니다.

- **Technical Details**: Hi-End-MAE는 두 가지 주요 혁신 기술을 도입합니다: (1) 인코더 구동 재구성: 인코더가 더 유익한 특징을 학습하도록 유도하여 마스킹된 패치의 재구성을 안내합니다. (2) 계층적 밀집 디코딩: 서로 다른 계층 간의 정보량이 풍부한 레이아웃을 캡처하기 위한 계층적 디코딩 구조를 구현합니다. 이를 통해 사전 훈련의 효과성을 높이고 특정 신경망 아키텍처에서는 제약이 따르는 것을 극복할 수 있습니다.

- **Performance Highlights**: Hi-End-MAE는 대규모 10K CT 스캔 데이터셋에서 사전 훈련을 실시하였으며, 일곱 개의 공공 의료 이미지 분할 벤치마크에서 성능이 평가되었습니다. 실험 결과, Hi-End-MAE는 전통적인 의료 SSL 방법보다 우수한 전이 학습 능력을 보이며, 다양한 다운스트림 작업에서 높은 성능을 기록했습니다. 또한, 이 방법은 비용 효율성과 일반화 가능성을 가지고 있어 대규모 의료 이미지 데이터셋의 사전 훈련에 적합합니다.



### Foundation Models in Computational Pathology: A Review of Challenges, Opportunities, and Impac (https://arxiv.org/abs/2502.08333)
Comments:
          63 pages, 7 figures

- **What's New**: 최근 몇 년 동안 자기 지도(Self-Supervised) 및 비전 전용 모델에서 대조적 시각-언어 프레임워크로 컴퓨터 병리학이 급격하게 발전했습니다. Generative AI "co-pilots"는 세포에서 병리학에 이르는 스펙트럼에서 미세한 조직 신호를 탐지하고, 포괄적인 보고서를 생성하며, 복잡한 사용자 질문에 응답하는 능력을 보여주고 있습니다.

- **Technical Details**: 다수의 멀티 기가픽셀(tissue images) 조직 이미지로 구성된 데이터는 수십 개에서 수백만 개로 급증하였으며, 이러한 모델의 학습 가능한 매개변수(trainable parameters)도 수십억 개로 증가하였습니다. 본 논문에서는 기초 모델(foundational models)의 정의를 탐구하며 이들이 왜 기초적이며 일반적 또는 다목적(multipurpose)인지를 규명합니다.

- **Performance Highlights**: 이러한 모델들은 예측(predictive) 및 생성(generative) 능력이 뛰어난 것으로 입증되었습니다. 그러나 전 세계적인 기준(global benchmarks)을 설정하는 것이 평가 기준을 향상시키고 이러한 모델의 임상 채택을 촉진하는 데 중요합니다. 최종적으로 첨단 AI의 broader impact는 광범위한 채택과 사회적 수용에 따라 달라집니다.



### Screener: Self-supervised Pathology Segmentation Model for 3D Medical Images (https://arxiv.org/abs/2502.08321)
- **What's New**: 이 논문은 3D 의료 영상에서 병리학적 발견을 정확하게 세그먼트하는 과제를 해결하기 위해 비지도 학습 방식인 비주얼 이상 탐지(Visual Anomaly Segmentation, UVAS)로 문제를 재구성했습니다. 기존의 기계 학습 모델이 기존 데이터셋의 제한된 주석들로 인해 제한적이었던 반면, 저자들은 수천 개의 라벨이 없는 CT 영상을 활용하여 더 넓은 범위의 병리학적 데이터를 다룰 수 있는 모델을 제안했습니다.

- **Technical Details**: Screener라는 모델은 3D CT 볼륨에서 dense self-supervised learning (SSL) 기법을 통해 특징을 추출하며, 의도적으로 수동으로 작성된 위치 인코딩을 대체하는 방식으로 작동합니다. 이 모델은 30,000개 이상의 라벨이 없는 CT 이미지를 기반으로 훈련되었으며, 뛰어난 anomaly score 할당 능력으로 병리학적 영역을 효과적으로 식별합니다.

- **Performance Highlights**: 제안된 모델은 1,820개의 다양한 병리 이미지를 포함한 4개의 대규모 테스트 데이터셋에서 기존 UVAS 방법들을 능가하는 성능을 보였습니다. 뿐만 아니라, Screener는 다양한 해부학적 영역에서 서로 다른 병리 조건에 대한 뛰어난 세그멘테이션 성능을 발휘했습니다.



### When do they StOP?: A First Step Towards Automatically Identifying Team Communication in the Operating Room (https://arxiv.org/abs/2502.08299)
- **What's New**: 이 연구는 수술 중 OR팀의 의사소통을 자동으로 식별하는 새로운 작업을 제안합니다. ‘팀 타임아웃(team Time-out)’과 ‘StOP?-프로토콜의 시작 및 종료 시간을 파악하여 환자의 안전과 수술 워크플로 분석에 기여하고자 합니다. 새로운 다중 시청 OR 데이터세트인 Team-OR를 생성했으며, 100시간이 넘는 실제 수술 비디오를 포함하고 있습니다.

- **Technical Details**: Team-OR 데이터세트는 37개의 복강 내시경 수술 비디오로 구성되어 있으며, 이에 대해 33개의 타임아웃과 22개의 StOP?-프로토콜 활동에 대한 시간 주석이 포함되었습니다. 수술 비디오에서 이들 활동의 시작 및 종료 시간을 자동으로 감지하기 위해 글로벌 장면 시각적 특징과 로컬 스켈레톤 기반 특징을 인코딩합니다. 이를 통해 효율적인 신경망 모델을 제안하여 그룹 활동을 탐지합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 기존의 최신 시간 행동 탐지 접근 방식을 초월하는 성과를 낼 수 있었음을 보여주었습니다. 본 연구는 OR 팀의 그룹 활동 분석에 있어 중요한 역할을 하며 데이터 세트를 통해 팀 타임아웃과 StOP?-프로토콜의 효과를 입증합니다. 이는 향후 의료 시스템에서의 팀 상호작용 분석 자동화에 기여할 수 있는 가능성을 보여줍니다.



### Fully-Geometric Cross-Attention for Point Cloud Registration (https://arxiv.org/abs/2502.08285)
- **What's New**: 이 논문에서는 낮은 중첩(overlap) 조건에서도 정확한 포인트 클라우드 등록(point cloud registration)이 가능하도록 새로운 cross-attention 메커니즘을 제시합니다. 기존 방법들이 포인트 쌍의 노이즈에 취약했으나, 제안된 FLAT 메커니즘은 기하학적 구조를 포함한 세분화된 정보를 결합하는 데 중점을 두었습니다. Gromov-Wasserstein 거리(Gromov-Wasserstein distance)를 활용하여 서로 다른 포인트 클라우드 간 거리 계산을 개선하였습니다.

- **Technical Details**: FLAT 메커니즘은 포인트 클라우드의 좌표(coordination) 및 특성(features)을 super-point 수준에서 융합하여 기하학적 구조를 효과적으로 학습할 수 있도록 합니다. 이로 인해 점들은 서로의 정보를 교환하며 겹치는 영역을 강조할 수 있습니다. 우리는 이 기법에 따라 두 가지 새로운 메트릭을 개발하였으며, 이는 회전 및 변환에 불변성을 가지도록 설계되었습니다.

- **Performance Highlights**: 논문에서는 FLAT 방법이 기존 기술 대비 뛰어난 성능을 나타내는 것을 입증하기 위해 3DMatch, 3DLoMatch, KITTI, 및 3DCSR 데이터셋에 대해 광범위한 평가를 실시했습니다. 실험 결과 FLAT은 이전의 접근 방식보다 더 많은 내투명(inlier) 대응을 생성하며, 따라서 더 정밀한 등록 결과를 제공합니다. 이러한 향상된 결과들은 포인트 클라우드를 효율적으로 매칭하는 데 기여합니다.



### UniCoRN: Unified Commented Retrieval Network with LMMs (https://arxiv.org/abs/2502.08254)
- **What's New**: 이 논문에서 제안하는 UniCoRN은 복합 멀티모달 검색(composed multimodal retrieval) 방식과 생성 언어 처리(generative language) 접근 방식을 통합하여 Commented Retrieval (CoR)이라는 새로운 작업을 수행합니다. UniCoRN은 기본 LMM(대형 멀티모달 모델)을 동결 상태로 유지하여 원래의 기능을 보존하면서도 검색과 텍스트 생성 작업을 수행할 수 있습니다. 또한, 새로운 엔티티 어댑터 모듈을 소개하여 검색된 멀티모달 엔티티를 LMM에 주입함으로써 보다 관련성 높은 댓글을 생성할 수 있도록 합니다.

- **Technical Details**: UniCoRN은 LMM의 숨겨진 상태(hidden state)를 CLIP 모델의 검색 공간에 정렬하는 검색 모듈을 포함하고 있어, 멀티모달 및 교차 모달 검색 능력을 통합합니다. 이 모델은 단순한 이미지 캡션이 아닌 복잡한 답변 및 댓글을 제공할 수 있도록 훈련되었습니다. 또한, CoR 작업을 수행하기 위해 훈련된 새로운 모듈을 통해 초기 질문과 검색된 엔티티 간의 연결을 강화하고, 보다 일관성 있는 텍스트 응답을 생성할 수 있습니다.

- **Performance Highlights**: UniCoRN은 다수의 데이터셋에서 기존 최첨단 모델보다 성능을 향상시킴을 보여주었습니다. 특히 Fashion-IQ, CIRR, OVEN, InfoSeek 및 Wiki-CoR 데이터셋에서 검색 정확도(recall)가 평균 +4.5% 향상되었으며, CoR 작업의 METEOR 점수는 RAG 접근법에 비해 +14.9% 개선되었습니다. 이러한 결과는 UniCoRN의 새로운 접근 방식이 실제 응용 프로그램에서 매우 효과적임을 나타냅니다.



### FloVD: Optical Flow Meets Video Diffusion Model for Enhanced Camera-Controlled Video Synthesis (https://arxiv.org/abs/2502.08244)
Comments:
          Project website: this https URL

- **What's New**: 이 논문은 카메라 제어가 가능한 비디오 생성 모델인 FloVD를 소개합니다. FloVD는 optical flow 맵을 사용하여 카메라 및 움직이는 물체의 모션을 표현함으로써 두 가지 주요 이점을 제공합니다. 첫째, optical flow는 비디오에서 직접 추정할 수 있으므로, 진정한 카메라 매개변수가 없는 임의의 훈련 비디오를 사용할 수 있습니다. 둘째, 배경 optical flow가 서로 다른 관점의 3D 상관관계를 인코딩하므로, 배경 모션을 활용하여 자세한 카메라 제어를 가능하게 합니다.

- **Technical Details**: FloVD는 입력 이미지와 카메라 매개변수를 기반으로 미래 프레임을 합성하는 두 단계의 비디오 합성 파이프라인을 채택하고 있습니다. 첫 번째 단계인 flow 생성 단계에서는 입력 이미지에서 카메라 및 이동 물체의 모션을 나타내는 optical flow 맵을 생성합니다. 이 optical flow 맵은 이후의 flow-conditioned 비디오 합성 모델에 입력되어 최종 비디오가 생성됩니다. 최적의 객체 모션 합성을 위해, 이 프레임워크는 카메라 흐름 생성과 물체 흐름 생성으로 나뉘어 있습니다.

- **Performance Highlights**: 광범위한 실험을 통해 FloVD가 정확한 카메라 제어 및 자연스러운 물체 모션 합성 측면에서 이전의 접근 방식보다 뛰어난 성능을 보였음을 입증했습니다. FloVD는 고품질 비디오를 생성하는 동시에 정밀한 카메라 제어를 지원하여 실용과 활용 가능성을 높였습니다. 이러한 성과는 비디오 제작, 가상 현실 및 상호작용 시뮬레이션과 같은 다양한 응용 분야에 긍정적인 영향을 미칠 것으로 기대됩니다.



### Learning Human Skill Generators at Key-Step Levels (https://arxiv.org/abs/2502.08234)
- **What's New**: 본 논문에서는 Key-step Skill Generation (KS-Gen)이라는 새로운 작업을 제안하여 인간의 복잡한 기술을 영상으로 생성하는 문제를 해결하고자 합니다. 기존의 비디오 생성 모델들이 단순한 동작 생성에는 성공했지만, 다단계 프로세스를 요구하는 인간 기술 생성에는 한계를 보이고 있습니다. KS-Gen은 기술 설명과 초기 상태를 제공받아 주요 단계의 짧은 클립을 생성하는 것에 초점을 맞추고 있습니다.

- **Technical Details**: KS-Gen 작업은 주어진 이미지와 기술 설명을 기반으로 하여 각 기술의 주요 단계에 해당하는 비디오 클립을 생성하는 것을 목표로 합니다. 이를 위해 다중 모달 대형 언어 모델(MLLM)을 이용하여 주요 단계의 설명을 생성하고, Key-step Image Generator (KIG)를 통해 주요 단계 간의 비주얼 일관성을 확보합니다. 마지막으로, 미세 조정된 비디오 생성 모델을 사용하여 각 주요 단계에 대한 비디오 클립을 생성합니다.

- **Performance Highlights**: 결과적으로, 우리의 접근 방식은 KIG의 도입으로 생성된 이미지의 품질과 일관성을 향상시키는데 기여했습니다. KS-Gen의 프레임워크는 Instructional Video 분석과 로봇 시나리오에 대한 선행 연구에 비해 더 많은 복잡성과 응용 가능성을 제공합니다. 이 연구는 인간 기술 생성을 위한 새로운 기초를 마련하고, 다양한 평가 지표와 고품질 데이터셋을 구축하여 KS-Gen의 성능을 평가하고 있습니다.



### Plantation Monitoring Using Drone Images: A Dataset and Performance Review (https://arxiv.org/abs/2502.08233)
- **What's New**: 이 연구에서는 드론 이미지를 사용한 나무 건강 모니터링의 새로운 자동화 시스템을 제안합니다. 기존의 위성 이미지 기반 방법과는 달리, 드론 이미지는 농민들이 쉽게 접근할 수 있으며 저렴합니다. 이 데이터셋은 세 가지 카테고리("건강함", "성장 지연", "사망")로 나뉘어져 있으며, CVAT 주석 도구를 사용하여 주석이 달렸습니다. 이로 인해 연구자들은 자동으로 나무의 건강 상태를 평가할 수 있는 딥러닝 모델을 개발하도록 자극받을 것입니다.

- **Technical Details**: 우리는 RGB 이미지를 포함한 드론 이미지의 사용을 통해 나무의 건강을 모니터링하는 데이터셋을 구축하였습니다. 여러 가지 잘 알려진 CNN 모델(AlexNet, VGGNet, ResNet 등)을 실험하여 성능을 비교하였으며, 초기 결과에서는 성능이 낮았으나, 깊이별 컨볼루션(depth-wise convolution) 작업이 모델 성능을 향상하는 데 기여할 수 있음을 확인했습니다. 본 연구는 드론 이미지 기반의 나무 모니터링을 위한 효율적이고 쉽게 사용할 수 있는 기술 개발을 목표로 합니다.

- **Performance Highlights**: 연구 결과에 따르면, 깊이별 컨볼루션이 포함된 딥 CNN 모델은 드론 데이터셋에서 성능을 향상시키는 데 효과적임을 보여주었습니다. 여러CNN 모델을 실험하여 각 모델의 성능을 평가하였고, 드론 이미지로부터 나무를 자동으로 모니터링하는 새로운 가능성을 제시하였습니다. 이 연구는 향후 농업 관리에 필수적인 도구가 될 수 있으며, 농민들이 효율적으로 자원을 관리하는 데 도움을 줄 것입니다.



### TRISHUL: Towards Region Identification and Screen Hierarchy Understanding for Large VLM based GUI Agents (https://arxiv.org/abs/2502.08226)
Comments:
          Under review at ICML 2025, 8 pages 5 figures

- **What's New**: TRISHUL은 기존 LVLM보다 더 포괄적인 GUI 이해를 위해 개발된 새로운 프레임워크입니다. 이 시스템은 Hierarchical Screen Parsing (HSP)와 Spatially Enhanced Element Description (SEED) 모듈을 활용하여 행위 지향(mapping instructions to GUI elements) 및 GUI 참조(task) 작업을 통합적으로 처리합니다. 기존 방법들이 주로 특정 작동(task)에 특화되어 있었던 반면, TRISHUL은 다양한 GUI 상호작용 작업을 지원하고, 훈련이 필요 없는 방식으로 설계되었습니다.

- **Technical Details**: HSP 모듈은 GUI 요소를 Global Regions of Interest (GROIs) 및 Local Elements (LE)로 구분하여 데이터의 위계적 이해를 가능하게 합니다. SEED 모듈은 GUI의 요소들 간의 상대적 위치를 분석하여 각각의 요소에 대한 고수준의 기능 설명을 생성합니다. 이 프로세스는 SAM 및 EasyOCR 알고리즘을 활용하여 GUI 내 텍스트 및 아이콘을 효과적으로 처리합니다.

- **Performance Highlights**: TRISHUL은 ScreenSpot, VisualWebBench, Mind2Web 및 AITW 데이터셋에서 기존의 최첨단 기법들을 초월하는 성능을 보여줍니다. 특히, TRISHUL 기반의 GPT-4V와 GPT-4o는 action grounding과 episodic instruction-following 작업에서 우수한 성과를 발휘하였습니다. 또한, Screen PR 데이터셋에서도 GUI 참조 성능을 향상시켜 타겟의 접근성과 사용자 상호작용 피드백을 개선하였습니다.



### Take What You Need: Flexible Multi-Task Semantic Communications with Channel Adaptation (https://arxiv.org/abs/2502.08221)
- **What's New**: 본 논문은 채널 적응성과 다중 작업 인식을 기반으로 한 새로운 의미 통신 프레임워크를 소개합니다. 이 프레임워크는 masked auto-encoder 아키텍처를 기반으로 하여, 여러 동시 작업에서 의미론적으로 중요한 데이터를 식별하고 우선순위를 매기는 멀티 태스크 인식 스코어링 메커니즘을 통합합니다. 채널 인식 추출기를 사용하여 실시간 채널 조건에 응답하여 동적으로 관련 정보를 선별하며, 의미의 관련성과 전송 효율성을 동시에 최적화합니다.

- **Technical Details**: 이 프레임워크는 다섯 가지 주요 구성 요소로 구성됩니다: 멀티 태스크 인식 스코어링, 채널 적응형 추출, masked auto-encoding, 전송 최적화, 실시간 채널 모니터링입니다. 멀티 태스크 인식 스코어링 모듈은 각 작업에 대한 이미지 패치의 의미 중요성을 기반으로 동적으로 점수를 부여합니다. 채널 적응형 추출기는 실시간 채널 상태 정보(CSI)에 따라 우선 순위가 높은 패치의 최소 하위 집합을 선택하여 전송하며, masked auto-encoder는 선택된 패치를 처리합니다.

- **Performance Highlights**: 실험 결과는 이 프레임워크가 전통적인 방법보다 우수한 성능을 발휘함을 보여줍니다. 이미지 재구성 및 객체 탐지와 같은 작업에서 최소 성능 저하를 보이며, 자원 제약 환경에서도 높은 효율성을 유지합니다. 이 결과는 이 프레임워크가 이질적인 채널 환경에 적응할 수 있으며, 다중 작업 응용 프로그램에 대한 확장성이 높다는 점에서 차세대 의미 통신 네트워크를 위한 유망한 솔루션으로 자리매김합니다.



### Deepfake Detection with Spatio-Temporal Consistency and Attention (https://arxiv.org/abs/2502.08216)
- **What's New**: 본 논문은 기존의 Deepfake(video) 감지 기법의 한계를 극복하기 위해 조작된 비디오의 국소적 조작 서명을 모델링한 새로운 신경망 기반 Deepfake 감지기를 제안합니다. 이는 개별 프레임과 프레임 시퀀스 수준에서 고유한 패턴 변화를 효과적으로 식별하고, 기존 방법들의 전역적인 프레임 특징 의존성을 줄이는 데 중점을 둡니다. 또한, 공간 주의 메커니즘과 거리 주의 메커니즘을 도입하여 심층 특징과 얕은 특징의 융합을 수행함으로써, 비디오 내의 세밀한 공격 흔적을 인식하는 데 기여합니다.

- **Technical Details**: 우리의 접근법은 제안된 모델이 ResNet 백본을 기반으로 하여, 얕은 프레임 수준의 특징 학습을 강화하도록 설계되었습니다. 해당 모델은 공간적인 'attention' 메커니즘을 사용하여 프레임 시퀀스 간의 관계를 분석하고, 'optical flow'를 통해 프레임 간의 움직임 정보를 정량화합니다. 이를 통해 우리가 제안하는 방법은 Deepfake 감지 작업을 미세한 분류 작업으로 바라보며, 시공간적인 특징의 일관성을 파악합니다.

- **Performance Highlights**: 제안된 모델은 FaceForensics++(LQ)와 DFDC라는 두 개의 대규모 데이터셋에서 평가되었으며, 기존의 8개 최첨단 방법들보다 우수한 성능을 보였습니다. 또한, 본 기술은 메모리와 계산적 측면에서도 경쟁 기법들보다 유리한 이점을 제공합니다. 이는 기존 기술들과 비교하여 더욱 향상된 실험 결과를 통해 검증되었습니다.



### ActiveSSF: An Active-Learning-Guided Self-Supervised Framework for Long-Tailed Megakaryocyte Classification (https://arxiv.org/abs/2502.08200)
Comments:
          6 pages, submitted to EMBC 2025

- **What's New**: 본 논문에서는 자가 지도 학습(self-supervised learning)을 기반으로 하여 거대혈소판(magakaryocytes) 분류 정확도를 개선하기 위한 ActiveSSF라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 활성 학습(active learning) 전략을 통합하여 세 가지 주요 문제를 해결하려고 합니다. 즉, 배경 노이즈, 긴 꼬리 분포(long-tailed distribution), 복잡한 형태적 변이를 다루고 있으며, 특히 희귀 아형 인식에서의 성능 향상을 목표로 하고 있습니다.

- **Technical Details**: ActiveSSF 프레임워크는 두 가지 주요 단계로 구성됩니다: 세포 영역 필터링(cell region filtering)과 활성 샘플 선택(active sample selection)입니다. 세포 영역 필터링 단계에서는 가우시안 블러(Gaussian blur) 및 K-평균 군집화(clustering) 기법을 사용하여 배경 소음을 줄이고, HSV 색 공간 분석을 통해 표적 세포에 해당하는 영역을 정확히 추출합니다. 활성 샘플 선택 단계에서는 라벨링된 샘플로부터 특징을 추출하고, 클러스터링을 통해 대표 프로토타입을 생성하여 희귀 클래스의 샘플을 우선적으로 선택합니다.

- **Performance Highlights**: 실험 결과, 제안된 ActiveSSF 방법은 주로 희귀 거대혈소판 아형의 분류 정확도를 크게 향상시키며 첨단 성능(state-of-the-art performance)을 달성했습니다. 이러한 다양한 기술 통합은 ActiveSSF의 임상 환경에서의 실용 가능성을 더욱 강조합니다. 추가적으로, 코드와 데이터세트는 향후 공개될 예정으로, 후속 연구를 촉진할 것으로 기대됩니다.



### AnyCharV: Bootstrap Controllable Character Video Generation with Fine-to-Coarse Guidanc (https://arxiv.org/abs/2502.08189)
Comments:
          15 pages, 9 figures, 4 tables

- **What's New**: 이번 연구에서는 AnyCharV라는 새로운 프레임워크를 제안하여 사용자가 원하는 캐릭터와 타겟 장면을 유연하게 결합한 캐릭터 비디오를 생성할 수 있도록 합니다. 이 프레임워크는 두 가지 단계로 구성되어 있으며, 첫 번째 단계에서 포즈 정보를 기반으로 소스 캐릭터와 타겟 장면을 통합합니다. 두 번째 단계에서는 자기 부스트 방식을 통해 생긴 비디오 자료를 효율적으로 활용하여 캐릭터의 세부 사항을 더 잘 보존합니다.

- **Technical Details**: AnyCharV의 첫 번째 단계는 fine segmentation mask(세분화 마스크)를 사용하여 소스 캐릭터와 타겟 장면을 통합하는 기본 모델을 설정합니다. 이 모델은 포즈 정보를 조건으로 사용하여 정밀한 공간 및 시간 제어를 합니다. 두 번째 단계에서는 생성된 비디오와 coarse bounding box mask(거친 경계 상자 마스크)를 활용하여 자기 학습을 통해 캐릭터의 세부 사항을 유지하는 데 중점을 둡니다.

- **Performance Highlights**: 실험 결과, 제안한 AnyCharV 방법은 이전의 최첨단 접근 방식을 정성적 및 정량적으로 초월하는 성과를 보였습니다. 이 방법은 다양한 배경 장면에 맞춰 소스 캐릭터를 통합하는 것을 가능하게 하며, 사용자에게 더 큰 조정 가능성을 제공합니다.



### CoDynTrust: Robust Asynchronous Collaborative Perception via Dynamic Feature Trust Modulus (https://arxiv.org/abs/2502.08169)
Comments:
          7 pages, 5 figures, conference

- **What's New**: 이 논문에서는 CoDynTrust라는 새로운 불확실성 인코딩 비동기 융합 인식 프레임워크를 제안합니다. 이는 다수의 에이전트 간의 다양한 정보의 비동기성을 효과적으로 처리하여 협력적 인식의 정확성을 향상시키는 데 중점을 두고 있습니다. CoDynTrust는 신뢰도 모듈을 동적으로 계산하여 특징 융합을 최적화하고, 이를 통해 안전성과 정확성을 보장합니다.

- **Technical Details**: CoDynTrust는 다이나믹 피처 트러스트 모듈러스(DFTM)를 활용하여 특정 관심 영역의 신뢰도를 평가합니다. 또한, 단순한 선형 외삽 방법을 적용하여 차량의 이동 예측을 처리하며, 자가학습 방법 대신에 효율적인 비동기 융합 방법을 설계합니다. 이러한 방법은 노이즈와 동기화 문제를 효과적으로 처리할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, CoDynTrust는 다수의 데이터셋에서 비동기성으로 인한 성능 저하를 크게 줄이는 것으로 나타났습니다. 특히, CoDynTrust는 지연 상황에서도 이전의 최첨단 방법들보다 우수한 탐지 성능을 달성하였습니다. 이 연구는 LiDAR 기반 3D 객체 탐지 기술의 강력한 협력적 인식을 위한 토대를 마련하는 데 기여하고 있습니다.



### Generalized Class Discovery in Instance Segmentation (https://arxiv.org/abs/2502.08149)
Comments:
          AAAI 2025

- **What's New**: 이번 연구는 instance segmentation(객체 분할)에서 generalized class discovery(GCD) 작업을 다룹니다. 이 과정에서 기존의 알려진 클래스와 새로운 클래스 모두를 분할하는 모델을 개발하기 위해 레이블이 있는 데이터와 없는 데이터를 활용하며, 이는 실제 세계에서의 불균형한 객체 분포를 고려합니다. 제안된 방법은 instance-wise temperature assignment(인스턴스별 온도 할당) 기법과 class-wise reliability criteria(클래스별 신뢰성 기준)를 통해 수업 간 불균형 문제를 해결하고자 합니다.

- **Technical Details**: 연구에서 제안된 ITA 방법은 헤드 클래스의 샘플에 대한 인스턴스 간 구별을 강화하면서, 일반적인 컨트라스트 학습 손실들이 헤드 및 테일 클래스를 동일하게 처리하는 것을 완화하는 것을 목표로 합니다. 또한, 클래스별 신뢰성 기준을 통해 레이블이 없는 데이터에서 발견되는 대다수의 테일 클래스의 의사 레이블을 제외하지 않도록 하는 방안을 제시합니다. 마지막으로, 공간 풀링과 깊이 감소에 기반한 효율적인 soft attention 모듈을 도입하여 GCD를 위한 객체-특화 표현을 인코딩합니다.

- **Performance Highlights**: 두 가지 설정인 COCO$_{half}$ + LVIS 및 LVIS + Visual Genome에서 제안된 방법의 실험 결과가 기존의 최신 방법들보다 뛰어난 성능을 보여주었습니다. 구체적으로, 기존의 중요한 프레임워크들과 비교했을 때, 제안된 방법은 레이블이 있는 데이터와 없는 데이터를 활용하여 새로운 클래스를 발견하고 정확한 인스턴스 분할 성능을 향상시키는 데 효과적임을 입증했습니다.



### Riemannian Complex Hermit Positive Definite Convolution Network for Polarimetric SAR Image Classification (https://arxiv.org/abs/2502.08137)
Comments:
          9 pages, 4 figures

- **What's New**: 이 논문에서는 PolSAR 이미지에 대해 Riemannian 복소 HPD 합성곱 신경망(HPD_CNN)을 제안합니다. 이는 복소 HPD 매트릭스를 직접 학습함으로써 기하학적 정보를 유지할 수 있는 방법론으로, 전통적인 변환 방법의 한계를 극복하고자 합니다. 특히, 실수와 허수 부분의 동등한 중요성을 강조하여 복소적인 산란 정보를 효과적으로 학습합니다.

- **Technical Details**: 제안된 HPDnet은 HPD 매핑, 정규화 및 로그 고유값(LogEig) 레이어를 포함하여 복소 매트릭스의 기하학적 특성을 학습합니다. 빠른 고유값 분해 방법을 설계하여 계산 부담을 줄이고, 복소 HPD 매트릭스를 유클리드 공간으로 변환할 수 있는 네트워크 프레임워크를 구성하여 고수준의 의미론적 특징을 학습할 수 있도록 합니다. 이를 통해 Riemannian 공간에서 유클리드 공간으로의 전이가 가능하게 됩니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 두 개의 실체 PolSAR 데이터셋에서 기존 최첨단 방법들보다 우수한 성능을 나타냈습니다. 특히, 이종 지역에서의 분류 성능이 향상되었음을 보여주며, Riemannian 기법을 활용한 모델링의 필요성을 확인합니다. 논문은 기존의 Euclidean 기반 접근 방식과 비교하여 더 뛰어난 분류 성능을 달성함을 강조합니다.



### A Survey on Data Curation for Visual Contrastive Learning: Why Crafting Effective Positive and Negative Pairs Matters (https://arxiv.org/abs/2502.08134)
Comments:
          9 pages, 3 figures

- **What's New**: 이 논문에서는 비주얼 대비 학습(contrastive learning)의 효과성을 극대화하기 위한 데이터 커레이션(data curation) 기술을 체계적으로 정리했습니다. 긍정(positive)과 부정(negative) 쌍을 선별하는 방법이_embeddings_의 품질에 미치는 영향을 분석하며, 기존 연구를 토대로 한 분류 체계를 제안합니다. 데이터 커레이션이 학습 방식 및 자료 선택과 어떻게 연관되어 있는지를 명확히 설명하고, 미래 연구를 위한 질문들을 제시하고 있습니다.

- **Technical Details**: 대비 학습의 성능은 정보론적 손실 함수(InfoNCE loss)에 의해 좌우되며, 이는 유사한 쌍은 서로 가까이, 반면 서로 다른 쌍은 멀리 두는 방식을 기반으로 합니다. 여기서 긍정 쌍 생성 방법은 단일 샘플에서의 변형을 통한 것과 다수의 샘플에서의 생성으로 나뉘며, 각기 장단점이 있습니다. 또한, 부정 쌍은 임의로 샘플링되기보다는 하드 네거티브(hard negative)를 선택하거나 대칭적으로 생성하는 방식이 필요함을 강조합니다.

- **Performance Highlights**: 효과적인 데이터 커레이션은 모델의 수렴 속도를 높이고, 포함된 데이터의 다양성을 증가시켜 더욱 견고한 표현 학습(representational learning)을 가능하게 합니다. 긍정/부정 쌍의 세심한 선별을 통해 학습 품질을 개선하고 일반화 성능을 향상시키는 것을 목표로 합니다. 또한, 본 연구는 다양한 데이터 커레이션 기술의 장단점을 고찰하며, 이를 통해 보다 효율적인 대비 학습을 가능하게 하는 방향을 제시합니다.



### ID-Cloak: Crafting Identity-Specific Cloaks Against Personalized Text-to-Image Generation (https://arxiv.org/abs/2502.08097)
- **What's New**: 이 논문은 새로운 개인화된 텍스트-이미지 모델에서 발생하는 프라이버시 문제를 해결하기 위해 최초로 정체성 특화 클록(ID-Cloak) 방식의 효율성을 조사합니다. 기존의 이미지 특정 방어 방법에 비해, 제안된 방법은 하나의 클록으로 모든 이미지를 보호할 수 있는 가능성을 제공합니다. 이 방법은 최소한의 개인 이미지에서 보편적인 클록을 생성하여, 사용자가 공유하는 이미지를 보호하는 데 있어서 실질적인 개선을 가져옵니다.

- **Technical Details**: 제안된 ID-Cloak는 개인의 이미지 집합에서 학습된 정보를 바탕으로 개인의 정체성 서브스페이스를 모델링합니다. 이 서브스페이스는 가우시안 분포로 표현되며, 텍스트 임베딩 공간에서 앵커 포인트를 통해 생성된 다양한 보호 컨텍스트를 반영합니다. 이후, 해당 서브스페이스 내에서 모델의 정상 출력을 벗어나도록 유도하는 새로운 최적화 목표를 설정하여, 효과적인 클록을 최적화합니다.

- **Performance Highlights**: 실험 결과는 제안된 ID-Cloak 방식이 소수의 이미지만으로도 효과적으로 개인의 모든 이미지를 보호할 수 있음을 증명합니다. qualitative 및 quantitative 데이터를 통해, 이 방법이 기존의 특정 이미지 방어 접근방식보다 훨씬 더 우수한 성능을 발휘하며, 사용자에게 실질적이고 사용 가능한 프라이버시 보호를 제공함을 보여줍니다.



### MAA: Meticulous Adversarial Attack against Vision-Language Pre-trained Models (https://arxiv.org/abs/2502.08079)
- **What's New**: 현재 VLP(vision-language pre-trained) 모델의 견고성을 평가하기 위한 공격 기법은 특정 모델에 맞춘 공격이 다른 모델에 잘 일반화되지 않는 한계를 가지고 있습니다. 본 논문에서는 MAA(Meticulous Adversarial Attack)라는 새로운 방법을 제안하여, 모델의 의존성을 줄이고 데이터 샘플의 독립적인 특성을 활용하여 전반적인 공격의 전이 가능성을 향상시킵니다. MAA는 촘촘한 최적화와 새로운 RScrop 기법을 통합하여, 다양한 VLP 모델에 대한 공격 효율성을 높이는 데 성공적인 결과를 보였습니다.

- **Technical Details**: MAA는 저수준 이미지 디테일을 증대시켜 이미지-텍스트 관계를 혼란시키도록 설계되었습니다. RScrop 기법은 이미지의 세부 사항을 포착하기 위해 크기 조정 및 슬라이딩 크롭을 활용하며, MGSD(multi-granularity similarity disruption) 전략을 통해 적대적 예시와 그 원본 사이의 특징 거리를 확대합니다. 이 기법은 다양한 VLP 모델 및 다중 작업에 대한 공격 성능을 향상시키기 위해 빼어난 특성과 취약점을 발견하는 데 중점을 두고 있습니다.

- **Performance Highlights**: MAA는 기존의 최첨단 기법에 비해 적대적 공격의 전이 가능성을 현저하게 높였습니다. 여러 VLP 모델 및 벤치마크 데이터셋에서 실시된 광범위한 실험을 통해, MAA의 성능이 모델 구성을 보다 잘 드러낸다는 인사이트를 제공합니다. 이러한 연구 결과는 향후 이 분야의 발전 방향을 제시하는 데 기여할 것입니다.



### Knowledge Swapping via Learning and Unlearning (https://arxiv.org/abs/2502.08075)
Comments:
          10 pages

- **What's New**: 이번 연구에서는 새로운 업무인 Knowledge Swapping을 소개합니다. 이 방법은 사전 훈련된 모델에서 사용자 지정 지식을 선택적으로 잊어버리면서 필수적인 지식을 유지하고 새로운 지식을 동시에 습득할 수 있도록 설계되었습니다. Knowledge Swapping은 다양한 외부 요구 사항을 충족할 수 있게 하며, 기존의 기계 학습 접근법에서 간과되었던 지식의 선택적 잊음을 해결하려 합니다.

- **Technical Details**: Knowledge Swapping은 세 가지 주요 목표, 즉 사용자 지식의 잊기, 핵심 지식의 보존, 새로운 지식의 습득을 목표로 합니다. 두 단계로 나누어져 있으며, 첫 번째 단계는 특정 지식을 잊고 두 번째 단계는 새로운 지식을 배우는 것입니다. 기존 방법과 달리, Knowledge Swapping은 고수준의 의미 정보를 먼저 잊고 저수준의 특성은 유지하는 방향으로 진행됨으로써 더 효율적으로 지식 관리를 할 수 있습니다.

- **Performance Highlights**: 다양한 실험을 통해 Knowledge Swapping이 지식의 잊음과 학습 효율성을 크게 향상시킨다는 것을 입증하였습니다. 이전의 기계 학습 방법들에 비해, 이 방법은 특정 영역의 지식을 체계적으로 잊고 동시에 새로운 지식을 효과적으로 배우는 데 있어 유리한 성과를 보여주었습니다. 연구 결과는 이미지 분류, 객체 탐지 및 의미 세분화와 같은 다양한 작업에서 확인되었습니다.



### From Brainwaves to Brain Scans: A Robust Neural Network for EEG-to-fMRI Synthesis (https://arxiv.org/abs/2502.08025)
- **What's New**: E2fNet는 저비용의 EEG 데이터를 이용하여 fMRI 이미지를 합성하는 새로운 딥러닝 모델입니다. 이 모델은 EEG신호에서 유의미한 특징을 포착하고 이를 정확한 fMRI 표현으로 변환하는 데 중점을 두고 개발되었습니다. E2fNet은 기존 방법들보다 뛰어난 성능을 보이며, 구조적 유사도 지수(SSIM)에서 최첨단 결과를 달성합니다.

- **Technical Details**: E2fNet은 EEG 데이터를 기반으로 fMRI를 생성하기 위해 특별히 설계된 완전한 CNN 구조입니다. 이 모델은 EEG 신호의 특성을 캡처하는 EEG 인코더, U-Net 모듈, 그리고 fMRI 디코더로 구성됩니다. EEG 인코더는 시간, 전극 채널 및 주파수 차원을 효과적으로 처리하며, 장기적인 종속성과 미세한 시간 동력을 파악하는 능력을 갖추고 있습니다.

- **Performance Highlights**: 세 가지 공개 데이터셋을 통해 평가한 결과, E2fNet은 CNN 기반 방법들보다 상당히 우수한 성능을 나타냈습니다. 본 연구에서 제안하는 E2fNet은 간단한 설계로도 높은 SSIM 점수를 달성하며, 이를 통해 신경 이미징 능력을 향상시키는 비용 효율적인 솔루션으로 자리 잡을 가능성을 보입니다.



### Joint Modelling Histology and Molecular Markers for Cancer Classification (https://arxiv.org/abs/2502.07979)
Comments:
          accepted by Medical Image Analysis

- **What's New**: 이 논문에서는 암 분류를 위한 새로운 디지털 병리학 접근법인 M3C2를 소개합니다. 이 접근법은 분자 마커(molecular markers)와 조직학적 특징(histology features)을 동시에 예측하며, 두 데이터 간의 상호작용(interactions)을 모델링합니다. 또한, 다양한 확대 배율(magnification)에서 정보를 효율적으로 추출하는 다중 스케일(disentangling module) 기능을 포함하고 있습니다.

- **Technical Details**: 저자들은 다중 스케일 다중 작업(multi-task) 학습 프레임워크를 제안하여 glioma의 조직학적 특징과 분자 마커를 예측합니다. 또한, 분자 마커의 공존 확률 기반(label correlation graph network) 및 상호작용 모듈을 도입하여 서로 다른 예측 결과 간의 관계를 모델링합니다. 특히, 동적 신뢰 제약(loss)과 교차 모달 그래디언트 조정 전략을 통해 두 예측의 조화를 이루도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 glioma 분류, 분자 마커 및 조직학적 특징 예측 측면에서 기존 기술들보다 더 우수한 성능을 보일 것으로 나타났습니다. 이는 정밀 암 치료(precision oncology)를 촉진할 수 있는 가능성을 보여주며, 생물 의학 연구 및 임상 응용 분야에서의 발전을 이끌 것으로 기대됩니다.



### Federated Self-supervised Domain Generalization for Label-efficient Polyp Segmentation (https://arxiv.org/abs/2502.07951)
Comments:
          Accepted at ADSMI @ MICCAI 2024

- **What's New**: 본 논문에서는 자가 지도 학습(self-supervised learning, SSL) 방법과 연합 학습(federated learning, FL)을 결합하여 장에서의 폴립 세분화를 개선하는 LFDG(Learned Federated Domain Generalization) 방법을 제안합니다. 이 방법은 의료 데이터의 프라이버시 문제를 해결하는 동시에 다양한 데이터 세트에 대한 일반화 능력을 향상시키는 것을 목표로 합니다. LFDG는 데이터 다양성을 높이기 위해 SSADA(자기 지도 대적 데이터 증강) 방법과 SRAM(원천 재구성 및 증강 마스킹) 모듈을 도입하여 모델의 성능을 향상시킵니다.

- **Technical Details**: LFDG는 각 클라이언트에서 SSL 방법을 사용해 로컬 데이터셋에 대해 훈련한 후, 서버에서 learned parameters를 집계하는 구조를 가지고 있습니다. 이 과정에서는 FedAvg 알고리즘을 통해 각 클라이언트의 모델 파라미터를 평균 업데이트하여 전역 모델을 구성합니다. SSADA 방법은 데이터 증강을 극대화하고 DropPos 프리트레인 손실을 최소화하여 안정적인 표현을 유지하면서 주요 데이터 확장을 진행합니다.

- **Performance Highlights**: LFDG는 여섯 개의 의료 센터에서 수집된 폴립 이미지 데이터에서 검증되었으며, 기존의 FL 및 SSL 방법에 비해 각각 3.80% 및 3.92% 더 나은 성능을 보였습니다. 또한, 제안된 방법은 최상의 평균 IoU(Intersection over Union) 성능인 62.83%를 달성하며, SSADA 및 SRAM 모듈이 모델 성능을 각각 2.50% 및 1.30% 향상시킵니다.



### SurGrID: Controllable Surgical Simulation via Scene Graph to Image Diffusion (https://arxiv.org/abs/2502.07945)
- **What's New**: 이번 논문에서는 SurGrID라는 새로운 Surgical simulation 모델을 소개합니다. 이 모델은 Scene Graphs(SGs)를 활용하여 제어 가능한 수술 장면 생성을 가능하게 합니다. 기존의 Denoising Diffusion Models(DDMs)에 비해 사용자 평가 연구에서 높은 현실감과 제어 가능성을 보였습니다.

- **Technical Details**: SurGrID는 수술 장면의 공간적 및 의미적 정보를 담고 있는 SG를 사용합니다. 이 모델은 수술 비디오를 통해 훈련되며, 새로운 수술 장면을 고충실도의 제어 하에 합성할 수 있도록 설계되었습니다. 특히, 공간 정보를 SG의 노드 특징으로 인코딩하여 세부적인 생성 제어를 가능하게 합니다.

- **Performance Highlights**: 본 연구에서는 백내장 수술 장면에서 생성된 이미지를 정량적으로 평가하였으며, 높은 충실도와 다양성을 달성했습니다. 또한, 임상 전문가들이 참여한 사용자 연구를 통해 생성된 시뮬레이션의 매우 현실적인 재현과 제어 가능성이 확인되었습니다. 이는 수술 시뮬레이션 분야에 새로운 가능성을 제시합니다.



### DeepSeek on a Trip: Inducing Targeted Visual Hallucinations via Representation Vulnerabilities (https://arxiv.org/abs/2502.07905)
Comments:
          19 pages, 4 figures

- **What's New**: 이 연구는 DeepSeek Janus 모델에서 타겟 이미지 생성과 이를 통한 시각적 환각을 유도하는 방법론을 체계적으로 실험하여 98.0%의 환각률을 달성했습니다. 이는 모델의 시각-언어 처리 파이프라인에서 발생하는 취약점을 드러내며, 복잡한 멀티모달 환경에서의 보안 위협을 강조합니다. 또한 LLaMA-3.1을 활용하여 환각 탐지 프레임워크를 도입하여 다양한 평가 지표를 제공하며, 이상 작동 감지의 필요성을 부각시킵니다.

- **Technical Details**: DeepSeek Janus는 1B와 7B 파라미터로 구성된 멀티모달 모델로, 훈련 과정에서의 공격에 대한 기대가 필요합니다. 특히, 이미지 임베딩을 조작하여 환각을 유도하는 전략을 개발하였으며, 이는 이미지의 원본 유사성을 유지하면서도 모델이 특정 타겟 임베딩을 생성하도록 유도하는 방식으로 이루어집니다. 이 연구에서는 환각 탐지의 정밀한 평가를 위해 새로운 LSD-Hallucination 벤치마크 데이터셋을 개설하였습니다.

- **Performance Highlights**: 실험 결과, COCO, DALL-E 3, SVIT 데이터셋에서 DeepSeek Janus 모델의 환각 발생률이 최대 98%에 달하며, 조작된 이미지의 시각적 충실도는 SSIM 0.88 이상을 유지하였습니다. 이는 근본적으로 오픈소스 모델이 가진 보안 리스크를 명확하게 드러내며, 실용적인 AI 적용시 고도의 주의가 필요함을 나타냅니다. 타겟 이미지 간의 조작에서 고도의 강도를 유지하면서도 모델의 반응을 극대화하는 방법론을 통해, 기존에 비해 보다 혁신적인 접근을 제시합니다.



### TextAtlas5M: A Large-scale Dataset for Dense Text Image Generation (https://arxiv.org/abs/2502.07870)
Comments:
          27 pages, 15 figures. Dataset Website: this https URL

- **What's New**: 최근 텍스트 기반 이미지 생성(text-conditioned image generation)에 대한 관심이 높아지고 있으며, 보다 긴 텍스트 프롬프트를 처리하는 기술이 발전하고 있습니다. 그러나 긴 형식의 텍스트를 포함한 이미지를 생성하는 것은 여전히 어려운 과제로 남아 있습니다. 기존 데이터셋들이 짧고 단순한 텍스트에 집중하고 있기 때문입니다. 이를 해결하기 위해, TextAtlas5M이라는 혁신적인 데이터셋이 소개되었습니다.

- **Technical Details**: TextAtlas5M 데이터셋은 500만 개의 긴 텍스트 이미지로 구성되어 있으며, 다양한 데이터 유형을 포함하고 있습니다. 이 데이터셋은 긴 텍스트 이미지 생성에 대한 대규모 생성 모델에 대한 종합적인 평가를 가능하게 합니다. 또한, 3000개의 사람 손질 테스트 세트인 TextAtlasEval이 3가지 데이터 도메인에 걸쳐 추가로 구성되어, 텍스트 기반 생성에 대한 큰 벤치마크를 제공합니다. 이러한 평가 시스템은 텍스트 기반 이미지 생성 모델들을 훈련하고 평가하는 데 유용하게 활용될 수 있습니다.

- **Performance Highlights**: TextAtlasEval 벤치마크는 현재 가장 선진화된 사유 모델(e.g. GPT4o with DallE-3)조차도 상당한 도전을 제시하며, 오픈 소스 모델들은 훨씬 더 큰 성능 격차를 보였습니다. 이러한 결과들은 TextAtlas5M이 차세대 텍스트 기반 이미지 생성 모델들 학습 및 평가에 매우 중요한 데이터셋임을 입증합니다.



### EventEgo3D++: 3D Human Motion Capture from a Head-Mounted Event Camera (https://arxiv.org/abs/2502.07869)
Comments:
          30 pages, 20 figures, 9 tables. arXiv admin note: text overlap with arXiv:2404.08640

- **What's New**: 이번 연구는 이벤트 카메라를 이용한 단안 3D 인간 동작 캡처 분야에서 중요한 진전을 이루었습니다. EventEgo3D++는 낮은 조명 및 빠른 움직임에서의 정확성을 향상시키기 위해 이벤트 스트림의 LNES 표현을 활용합니다. 새로운 모바일 헤드 마운트 장치를 개발하여 여러 환경에서 실제 이벤트 관찰을 포함하는 포괄적인 데이터셋을 수집했습니다. 또한 이 연구는 CVPR 2024에서 발표된 EventEgo3D의 확장판입니다.

- **Technical Details**: 본 연구에서 제안하는 EventEgo3D++는 두 가지 주요 개선 사항을 포함합니다. 첫째, 추가적인 감독 학습을 통해 3D 자세 추정 정확성을 향상시켰습니다. 둘째, 특히 자연 환경에서 3D 근본 자세를 포함하는 새로운 데이터셋 EE3D-W를 소개하였으며, 이는 기존의 합성 및 스튜디오 기록 데이터셋과 함께 사용됩니다. 이와 함께 SMPL 바디 모델 주석을 제공하여 데이터셋의 전반적인 가치를 극대화했습니다.

- **Performance Highlights**: 실험 결과에 따르면 EventEgo3D++는 기존 RGB 기반 솔루션에 비해 우수한 3D 정확성과 강인성을 달성했습니다. 특히, 이 방법은 도전적인 환경에서도 강력한 성능을 보여 주며, 초당 140Hz의 속도로 실시간 3D 포즈 업데이트를 지원합니다. 이는 HMD의 실제 애플리케이션에서 효율적이고 신뢰할 수 있는 성능을 제공하는 데 기여합니다.



### MRS: A Fast Sampler for Mean Reverting Diffusion based on ODE and SDE Solvers (https://arxiv.org/abs/2502.07856)
Comments:
          Accepted by ICLR 2025

- **What's New**: 이 논문은 Mean Reverting (MR) Diffusion 모델의 샘플링 효율성을 개선하는 새로운 알고리즘인 MRS (MR Sampler)를 제안합니다. 기존의 빠른 무작위 샘플러가 MR Diffusion에 직접 적용되지 않는 한계를 극복하여 적은 단계로도 고품질 샘플을 생성할 수 있도록 합니다. MRS는 역방향 확률 미분 방정식 및 확률 흐름 일반 미분 방정식을 해결하여 반-정리 해를 도출하고, 이를 활용하여 빠른 샘플링을 가능하게 합니다.

- **Technical Details**: MR Sampler는 MR Diffusion에서 발생하는 역방향 확률 미분 방정식 (PF-ODE)과 SDE를 해결하여 샘플링 공식을 생성합니다. 이 방법은 분석 함수와 신경망에 의해 매개변수화된 적분을 포함하여 반-정리 해의 형태로 구성됩니다. 제안된 알고리즘은 노이즈 예측, 데이터 예측 및 속도 예측을 포함한 모든 주요 매개변화를 지원하고, 훈련이 필요 없으며 다양한 작업과 데이터 세트에 적응할 수 있습니다.

- **Performance Highlights**: MR Sampler는 10개의 서로 다른 이미지 복원 작업에서 10배에서 20배의 속도 향상을 달성하면서도 높은 샘플링 품질을 유지하는 것으로 나타났습니다. 이 알고리즘은 샘플링 절차를 가속화하여 MR Diffusion에서 실用성을 높이는 데 중요한 기여를 합니다. 고품질 샘플을 적은 수의 함수 평가(NFEs)로 생성할 수 있어, 효율적인 이미지 및 비디오 생성에 유용하게 사용될 수 있습니다.



### Vision-Language Models for Edge Networks: A Comprehensive Survey (https://arxiv.org/abs/2502.07855)
- **What's New**: 이번 연구는 비전 대형 언어 모델(Vision Large Language Models, VLMs)의 최적화와 경량화를 통해 자원 제약이 많은 엣지 환경에서의 활용 가능성을 탐구합니다. 모델 압축 기법인 pruning, quantization, knowledge distillation을 활용하여 VLMs의 성능과 효율을 향상시키는 방법이 제안되고 있습니다. VLMs는 의료, 환경 모니터링, 자율 시스템 등 다양한 분야에서 응용할 수 있는 가능성을 보여줍니다.

- **Technical Details**: VLMs는 시각적 입력과 자연어 처리를 결합하여 이미지 캡셔닝, 비주얼 질문 응답 등 다양한 태스크를 수행합니다. 그러나 최신 VLMs는 큰 메모리 요구와 높은 에너지 소비로 인해 일반적인 엣지 기기에서는 실행이 어렵습니다. 이를 해결하기 위해 pruning, quantization, 그리고 knowledge distillation 기법을 사용하여 모델 크기와 계산 비용을 줄이는 방법이 연구되고 있으며, 고유한 하드웨어 가속기도 중요한 역할을 하고 있습니다.

- **Performance Highlights**: 자원 제약이 있는 엣지 디바이스에서도 VLMs를 성능 저하 없이 활용할 수 있는 가능성이 연구되고 있습니다. 예를 들어, 의료 이미징 분야에서 VLMs를 활용하면 휴대 기기에서 즉각적인 피드백과 의사 결정을 지원할 수 있습니다. 자율주행차 및 스마트 감시와 같은 분야에서 VLMs의 실시간 처리가 필수적이며, VLMs의 경량화로 인해 이러한 기술들이 더욱 발전할 것으로 기대됩니다.



### Technical note on calibrating vision-language models under covariate shif (https://arxiv.org/abs/2502.07847)
- **What's New**: 본 연구는 저샷 비전 분류(vision classification)에서 발생하는 두 가지 주요 문제인 covariate shift와 confidence misalignment를 동시에 해결하는 새로운 접근법인 Confidence-Calibrated Covariate Shift Correction (C3SC)를 제안합니다. 기존의 개별적인 접근법과 달리, C3SC는 Fisher 정보 기반 페널티를 사용하는 통합 프레임워크로, 이는 일반화 능력을 크게 향상시킵니다. 실험 결과 C3SC는 다양한 데이터셋에서 5.82% 향상된 보정 성능을 보여줍니다.

- **Technical Details**: C3SC는 covariate shift 보정을 위해 Fisher 정보 페널티를 활용합니다. 또한, misclassified 예제에 대한 신뢰도를 낮추기 위해 confidence misalignment penalty (CMP)를 통합합니다. 이 두 가지 페널티는 CLIP의 대조 손실(contrastive loss)에 통합되어 안정적이고 신뢰할 수 있는 학습 시스템을 제공합니다.

- **Performance Highlights**: 실험 결과 C3SC는 covariate shift가 존재하는 어려운 데이터셋에서도 정확도에서 3.5%의 향상을 보이며, 전반적으로 신뢰할 수 있는 비전-언어(Vision-Language) 저샷 응용 프로그램에 유망한 솔루션으로 자리매김합니다. 특히, 저샷 학습(learning) 환경에서 데이터의 한계로 인해 발생하는 부정확한 예측을 줄이는데 효과적입니다.



### Spread them Apart: Towards Robust Watermarking of Generated Conten (https://arxiv.org/abs/2502.07845)
- **What's New**: 이 논문은 생성된 이미지에 물리적 워터마크를 삽입하는 새로운 접근 방식을 제안합니다. 이를 통해 생성된 콘텐츠를 검출하고 해당 콘텐츠를 생성한 사용자를 식별할 수 있습니다. 특히, 이 방법은 모델의 재교육이나 파인 튜닝을 필요로 하지 않으며, 향후 사용자가 생성한 콘텐츠에 대한 진위를 검증하는 데 중점을 두고 있습니다.

- **Technical Details**: 저자들은 연속적인 생성 콘텐츠에 디지털 워터마크를 삽입하는 'Spread them Apart'라는 프레임워크를 제안합니다. 이 방법은 생성 과정에서 워터마크를 삽입하여 이미지 후처리 시에도 안정성을 보장합니다. 또한, 다양한 포스트 프로세싱 공격에 대한 내성을 실험적으로 검증하여 기존 방법보다 더 우수한 성능을 발휘함을 보여줍니다.

- **Performance Highlights**: 제안된 방법은 기존의 워터마킹 기술들에 비해 포스트 프로세싱 공격에 대한 내구성이 뛰어난 것으로 나타났습니다. 특히, 이미지의 밝기 조정, 대비 조정 또는 감마 수정과 같은 공격에서 효과적으로 방어할 수 있는 능력을 갖추고 있습니다. 이러한 결과는 향후 생성된 콘텐츠의 신뢰성과 사용자 식별 가능성을 높이는 데 기여할 수 있을 것입니다.



### TranSplat: Surface Embedding-guided 3D Gaussian Splatting for Transparent Object Manipulation (https://arxiv.org/abs/2502.07840)
Comments:
          7 pages, 6 figures

- **What's New**: 본 논문에서는 투명 객체를 위한 3D Gaussian Splatting 방법인 TranSplat을 제안합니다. TranSplat은 Latent Diffusion Model을 활용하여 지속적이고 일관된 surface embeddings를 생성함으로써 투명 물체의 복잡한 특성을 효과적으로 캡처합니다. 이 방법은 또한 RGB 이미지와의 통합을 통해 조명 및 시점 변화에 강력한 깊이 완성을 달성합니다.

- **Technical Details**: TranSplat은 두 단계로 작동합니다. 첫 번째 단계에서는 RGB 이미지로부터 Latent Diffusion Model을 사용하여 surface embeddings을 추출합니다. 두 번째 단계에서는 추출한 surface embeddings와 RGB 이미지를 결합하여 3D-GS를 통해 깊이를 렌더링하고 3D 장면을 재구성합니다. 이렇게 생성된 surface embeddings는 비Lambertian surface에 대해서도 안정된 표현을 제공합니다.

- **Performance Highlights**: TranSplat은 합성 데이터셋 및 실제 투명 객체 벤치마크에서 깊이 완성 정확성을 현저히 향상시키는 성능을 보였습니다. 추가적으로, 로봇 조작 작업에서 grasping point의 정확한 탐지를 통해 실용적 응용 가능성을 입증하였습니다. 이 연구는 탁월한 깊이 재구성을 달성하고, 해당 분야의 발전을 위한 합성 데이터셋과 모델을 오픈 소스로 제공할 계획입니다.



### NanoVLMs: How small can we go and still make coherent Vision Language Models? (https://arxiv.org/abs/2502.07838)
Comments:
          11 pages, 8 figures, 3 tables

- **What's New**: 이번 연구는 Vision-Language Models (VLMs)의 최신 동향을 다루며, GPT-4V와 Llama 3.2와 같은 모델이 대규모 언어 모델(LLMs)을 이용한 멀티모달 작업에서 주목받고 있음을 시사합니다. 그러나 검토된 모델들은 독점적 제한과 높은 계산 요구 사항, 접근성 부족 등의 문제로 인해 성과가 제한적입니다. 본 연구에서는 3-4세 어린이의 학습 과정을 모방하여, 어린이의 언어 사용에 맞는 두 개의 새로운 데이터셋, ShortDesc와 LongDesc를 소개합니다.

- **Technical Details**: 제안된 데이터셋은 각기 다른 이미지-텍스트 쌍으로 구성되어 있으며, 텍스트는 어린이가 사용하는 간단한 어휘와 구문으로 제한됩니다. 연구에서는 새로운 축소 모델인 GPT-4o를 사용하여 이 데이터셋을 생성하고, 이를 통해 기존 소형 VLM보다 10배 작으면서도 아키텍처의 단순함을 유지하는 VLM 훈련이 가능함을 보여줍니다. 특히, 평가 과정에서 GPT-4o를 활용하여 학생들이 작성한 이야기의 창의성, 의미성 및 일관성을 10점 만점으로 평가하였습니다.

- **Performance Highlights**: 연구 결과, 제안된 경량의 VLM은 자원 제약 환경에서도 효과적으로 사용될 수 있는 잠재력을 가지며, 기존의 표준 벤치마크의 한계를 극복할 수 있습니다. 본 논문은 멀티모달 모델의 발전에 기여하며, 접근성이 떨어지는 환경에서도 활용될 수 있는 모델 설계를 제안합니다. 이러한 연구는 향후 다양한 응용 프로그램에서의 혁신에 기여할 것으로 기대됩니다.



### Captured by Captions: On Memorization and its Mitigation in CLIP Models (https://arxiv.org/abs/2502.07830)
Comments:
          Accepted at ICLR 2025

- **What's New**: 이 논문에서는 CLIP 모델의 메모리화(memorization) 과정을 정량화하기 위해 CLIPMem이라는 새로운 개념을 제안합니다. 기존의 메모리화 정의가 CLIP의 다중 모달 특성을 반영하지 못하는 한계를 해결하고자 메모리화를 설명하기 위해 새로운 기준을 설정합니다. '잘못 캡션된(mis-captioned)' 데이터가 가장 높은 메모리화 수준을 보인다는 중요한 결과도 발견했습니다.

- **Technical Details**: CLIP은 이미지와 텍스트의 쌍을 공유된 잠재 공간에 매핑하여 작업을 수행하는 다중 모달 인코더 구조를 기반으로 합니다. 이 모델은 대조 손실 함수(contrastive loss)를 통해 정답 쌍 사이의 유사성을 극대화하고 잘못된 쌍의 유사성을 최소화합니다. CLIPMem 메트릭은 주어진 이미지-텍스트 쌍의 정렬을 비교하여 메모리화를 측정하는 새로운 방법입니다.

- **Performance Highlights**: 연구를 통해 텍스트 인코더가 이미지 인코더보다 메모리화에 더 많이 기여하며, 이는 모델 성능에 중요한 영향을 미칩니다. 리서치는 메모리화 감소 전략이 CLIP의 일반화 능력을 향상시킬 수 있음을 보여줍니다. 이러한 결과는 전통적인 학습 패러다임에서는 메모리화 감소가 일반적으로 성능 저하로 이어지던 것과는 대조적입니다.



### Preference Alignment on Diffusion Model: A Comprehensive Survey for Image Generation and Editing (https://arxiv.org/abs/2502.07829)
- **What's New**: 이 논문은 이미지 생성 및 편집 분야에서 확산 모델(Diffusion Models, DMs)과 선호 정렬(preference alignment)의 통합을 체계적으로 조사한 첫 번째 연구입니다. 연구의 주요 목적은 이러한 통합이 초보자에게 제시하는 도전 과제를 조명하고, DMs와 RL(강화 학습)이 이미지 생성과 편집에 어떻게 상호작용하는지에 대한 명확한 개요를 제공하는 것입니다. 기존의 많은 연구에서 DMs의 다양한 활용은 다뤘지만, 선호 정렬과의 구체적인 통합에 대해 종합적으로 검토한 자료는 부족했습니다.

- **Technical Details**: 본 연구에서는 DDPMs(분산 모델)와 DDIMs(Deterministic Denoising Diffusion Implicit Models)의 원리에 대해 설명하며, 특히 노이즈 추가 과정과 그 역과정을 강조합니다. 또한, PPO(Proximal Policy Optimization) 알고리즘을 활용하여 선호 정렬을 위한 강화 학습 과정에 대해 깊이 있게 기술합니다. 이러한 알고리즘은 훈련에서 안정성과 강인성을 제공하며, 이를 통해 DMs의 성능을 효과적으로 향상시킬 수 있는 방법을 제시합니다.

- **Performance Highlights**: 이 논문은 DMs와 RL 기반 최적화 기술(예: RLHF, DPO)이 이미지 생성 및 편집에서 어떻게 활용되는지를 중점적으로 살펴봅니다. 자율 주행, 의료 영상, 로보틱스 등 다양한 분야에서 DMs의 응용 가능성을 탐구하며, 운영에서 겪는 도전 과제들과 이들의 해결 방안도 논의합니다. 이를 통해 향후 기술 발전을 위한 방향성을 제시하고 DMs와 선호 정렬을 연계한 혁신적인 연구의 가능성을 강조합니다.



### Deep Learning in Automated Power Line Inspection: A Review (https://arxiv.org/abs/2502.07826)
Comments:
          40 pages, 12 figures

- **What's New**: 최근 몇 년간, 고압전선 유지보수는 컴퓨터 비전(Computer Vision)과 딥러닝(Deep Learning)에 기반한 자동 검사로 패러다임 전환을 이뤘습니다. 전력 전송의 신뢰성, 안전성 및 지속 가능성을 유지하기 위해 방대한 비디오 및 이미지 컬렉션이 필수적이라는 점에 주목하고 있습니다. 이 논문은 이 분야의 연구자 및 산업 종사자들이 전선 데이터 분석을 위한 딥러닝 기반 시스템을 개발하는 데 도움을 주고자 기존 연구를 체계적으로 정리하였습니다.

- **Technical Details**: 전선 검사의 동작 원리를 이해하기 위해, 본 논문은 현재 연구의 본체를 구성요소 감지(Component Detection)와 결함 진단(Fault Diagnosis) 두 가지 주요 영역으로 분류하고 있습니다. 각 영역에서 사용되는 다양한 방법과 기술을 한 눈에 볼 수 있도록 자세히 정리하였으며, 딥러닝 기반 방법론의 핵심 원리 및 실제 응용에 대한 설명을 제공하고 있습니다. 이러한 체계적 접근은 작업을 효율적으로 수행할 수 있는 기회를 제공합니다.

- **Performance Highlights**: 자동화된 전선 검사 프로세스는 조기 결함 발견과 효율적인 유지보수를 통해 전력 공급의 안전성을 향상시킬 수 있는 장점을 제공합니다. 고해상도 이미지를 활용한 실시간 데이터 분석 덕분에 유틸리티 회사들은 유지보수 필요성을 예측하고, 정전 사태를 예방하며, 소비자에게 신뢰할 수 있는 전력을 제공할 수 있게 되었습니다. 또한, 이 논문은 데이터 품질, 작은 물체 탐지, 임베디드 시스템에서의 딥러닝 응용 등 미래 연구 방향에 대한 시사점을 제공합니다.



### Pre-Trained Video Generative Models as World Simulators (https://arxiv.org/abs/2502.07825)
Comments:
          20 pages

- **What's New**: 본 연구에서는 비디오 생성 모델을 동적인 세계 시뮬레이터로 변환하기 위한 새로운 접근 방식인 Dynamic World Simulation (DWS)을 제안합니다. DWS는 사전 훈련된 비디오 생성 모델을 유연하게 변환할 수 있으며, 특정한 행동 경로를 기반으로 시각적 변화를 생성할 수 있는 구조를 가지고 있습니다. 이는 기존 모델에 통합할 수 있는 경량화된 행동 조건 모듈을 도입하여 동적 변화에 대한 정밀한 조정을 가능하게 합니다.

- **Technical Details**: DWS는 프레임 수준의 액션 인식을 개선하는 경량의 액션 조건 모듈을 도입하여, 프레임 간의 동적 전환 모델링을 중시하는 방안을 제공합니다. 이 모듈은 두 개의 선형 층으로 구성되어 있으며, 기존 모델 아키텍처에 손쉽게 통합될 수 있도록 설계되었습니다. 또한, DWS는 정적인 시각적 세부 요소에서 동작 유도 동적 변화로 모델의 주의를 전환하는 모션 강화 손실(motion-reinforced loss)을 통해 훈련 중 동적 모델링의 우선 순위를 명확하게 설정합니다.

- **Performance Highlights**: DWS는 게임 및 로봇 분야 전반에 걸쳐 행동 제어가 가능한 동적 비디오 생성에서 큰 개선을 이루었습니다. 본 연구의 실험 결과는 DWS로 학습한 세계 시뮬레이터가 여러 정책과 정확한 동적 예측을 유지하면서도 다양한 도메인에서 상호작용할 수 있음을 보여줍니다. 또한, 우선 순위가 매겨진 상상(prioritized imagination)을 도입하여 샘플 효율성을 개선하고, 최신 MBRL 방법들과 비교해 경쟁력 있는 성능을 나타냅니다.



### PDM-SSD: Single-Stage Three-Dimensional Object Detector With Point Dilation (https://arxiv.org/abs/2502.07822)
- **What's New**: 이 논문에서는 기존의 점 기반 감지기들이 가진 한계를 극복하기 위한 새로운 방법인 Point Dilation Mechanism (PDM-SSD)을 제안합니다. PDM은 점을 특정 크기의 그리드로 확장하고, 비어 있는 공간에 대해 기능을 채우는 두 가지 주요 단계를 통해 특징 공간을 확장합니다. 이 방식은 특히 불완전하거나 희박한 객체 인식에서 효과적이며, 학습 속도와 정확도 간의 균형을 잘 맞춥니다.

- **Technical Details**: PDM-SSD는 PointNet 스타일의 3D 백본을 사용하여 효율적으로 특징을 인코딩합니다. 또한, 구형 조화 계수와 가우시안 밀도 함수를 활용하여 비어 있는 그리드를 기능으로 채우고, 여러 개의 확장 중심을 연관시켜 희박한 그리드 특징을 얻는 과정이 포함됩니다. 최종적으로는 혼합 감지 헤드를 설계하여 객체의 확률을 보정하고, 장면 열지도를 예측하여 탐지 정확도를 높입니다.

- **Performance Highlights**: PDM-SSD는 KITTI 데이터세트에서 다중 클래스 감지에서 최첨단 성능을 달성했으며, 68 프레임의 추론 속도를 기록합니다. 또한, 다양한 객체 수준의 사례들을 통해 희박하고 불완전한 객체 검출 측면에서 PDM-SSD의 장점을 입증했습니다. 이 방법은 PDM을 보조 네트워크로 사용하여 샘플링 포인트와 객체 중심 간의 연결을 강화하여 성능을 향상시킬 수 있습니다.



### Amnesia as a Catalyst for Enhancing Black Box Pixel Attacks in Image Classification and Object Detection (https://arxiv.org/abs/2502.07821)
Comments:
          Accepted as a poster at NeurIPS 2024

- **What's New**: 이 논문은 새로운 픽셀 기반 블랙박스 공격 기법인 Remember and Forget Pixel Attack using Reinforcement Learning (RFPAR)을 제안합니다. 기존 연구에서 다루지 않았던 픽셀 공격에 중점을 두고, 이미지 분류뿐만 아니라 물체 탐지에서도 효과를 입증하였습니다. 이 방법의 주요 특징은 랜덤성을 줄이고 패치 의존성을 피하기 위해 강화학습(RL) 알고리즘을 활용한다는 점입니다.

- **Technical Details**: RFPAR는 두 가지 프로세스인 Remember와 Forget으로 구성되어 있습니다. Remember 과정에서는 RL 에이전트가 클린 이미지를 입력으로 받고, 최적화 과정에서 가장 높은 보상과 대응하는 변조된 이미지를 메모리에 저장합니다. Forget 과정에서는 보상이 더 이상 변화하지 않을 경우 이전 정보를 잊고 다시 Remember 프로세스를 시작하게 됩니다.

- **Performance Highlights**: RFPAR는 ImageNet-1K 데이터셋에서 최첨단 쿼리 기반 픽셀 공격을 초월하여 평균 공격 성공률을 12.1% 향상시키고, 쿼리 수를 26.0% 줄였습니다. 또한 물체 탐지 분야에서도 YOLO 및 DDQ와 함께 MSCOCO 데이터셋을 사용해 mAP 감소를 비교했으며, 적은 수의 쿼리로도 유사한 성능을 달성하였습니다.



### Unpaired Image Dehazing via Kolmogorov-Arnold Transformation of Latent Features (https://arxiv.org/abs/2502.07812)
Comments:
          11 pages, 8 figures

- **What's New**: 본 논문은 Kolmogorov-Arnold 변환(Kolmogorov-Arnold Transformation)을 기반으로 한 비지도 이미지 디헤이징 프레임워크, UID-KAT를 제안합니다. 기존의 이미지 디헤이징 방법들은 통계적 가정을 바탕으로 하여 실제 환경에서의 성능 부족 문제를 겪었습니다. 본 연구는 적대적 훈련(adversarial training)과 대조 학습(contrastive learning)을 결합하여 흐릿한 이미지와 선명한 이미지 간의 복잡한 관계를 모델링하고자 합니다.

- **Technical Details**: UID-KAT 프레임워크는 Kolmogorov-Arnold 네트워크(KANs)를 활용하여 이미지 디헤이징을 수행합니다. 연구에서는 다층 퍼셉트론(MLP) 대신 KAN 계층을 사용하는 이중 GR-KAN 변환기(Dual-GR-KAN Transformer) 아키텍처를 도입하여 레이턴트 공간(latent space)에서의 특징 변환을 개선합니다. 이 접근법은 페어링된 이미지 데이터셋이 필요 없으며, 사용자들이 수집한 이미지 세트로 훈련할 수 있도록 합니다.

- **Performance Highlights**: UID-KAT는 다양한 데이터셋에서 기존의 비지도 방법들을 초월하는 최첨단 디헤이징 성능을 보여줍니다. 실험 결과, 이 프레임워크는 낮은 계산 비용에도 불구하고 효율적이고 효과적인 성능을 달성했습니다. 각 도메인에서 단 1,000개의 이미지로 훈련되었음에도 불구하고 뛰어난 결과를 나타냈습니다.



### CrossVideoMAE: Self-Supervised Image-Video Representation Learning with Masked Autoencoders (https://arxiv.org/abs/2502.07811)
- **What's New**: 이 논문에서는 CrossVideoMAE라는 새로운 자기 지도 학습(self-supervised learning) 프레임워크를 제안하여 비디오와 샘플된 프레임 간의 spatiotemporal 및 semantic representation 학습을 개선합니다. CrossVideoMAE는 시각적 정보와 공간 정보를 결합하여 비디오 이해를 위한 효율적인 방법을 제공합니다. 이를 통해 기존의 비디오 기반 Masked Autoencoders(MAEs)에서 발견되는 한계를 극복할 수 있습니다.

- **Technical Details**: 이 프레임워크는 비디오 레벨과 프레임 레벨의 rich spatiotemporal representations를 동시에 학습하며, feature-invariant space 내에서 비디오와 샘플된 프레임의 spatiotemporal 정보를 통합합니다. CrossVideoMAE는 Masked Image Modeling(MIM) 기법을 사용하여 프레임 간의 상호 연관성을 학습하고, 데이터 증강(data augmentation)에 대한 불변성을 유지하도록 설계되었습니다. 이 과정에서 기억할 만한 이동 패턴과 상호 작용을 학습합니다.

- **Performance Highlights**: 다양한 실험을 통해 CrossVideoMAE가 기존의 최첨단 방법들보다 뛰어난 성능을 보임을 입증했습니다. 또한, ablation studies를 통해 제안한 방법의 효과성을 검증하며, 명시적 언어 데이터 없이도 높은 수준의 의미적 이해를 달성할 수 있음을 보여줍니다. 기존 방법들과 비교할 때 훨씬 적은 계산 자원으로 경쟁력 있는 성능을 발휘했다고 강조하고 있습니다.



### Movie Weaver: Tuning-Free Multi-Concept Video Personalization with Anchored Prompts (https://arxiv.org/abs/2502.07802)
Comments:
          Project page: this https URL

- **What's New**: 이번 논문에서는 개인화 비디오 생성의 새로운 접근 방식인 Movie Weaver가 소개됩니다. 이 모델은 다양한 이미지 조합과 텍스트 프롬프트를 엮어서 개인화된 다중 개념 비디오를 생성할 수 있도록 설계되었습니다. 이를 통해 얼굴, 신체, 동물 이미지 등 여러 개념을 유연하게 조합할 수 있어 다양한 실세계 시나리오에 적용할 수 있는 가능성을 제시합니다.

- **Technical Details**: Movie Weaver는 앵커 프롬프트(anchored prompts)를 도입하여 각 개념 설명을 특정 참조 이미지에 링크합니다. 또한, 개념 임베딩(concept embeddings)을 사용하여 참조 이미지의 순서를 인코딩합니다. 이 두 가지 기술을 통해, Movie Weaver는 기존 방법들에 비해 신원 보존(identity preservation)과 전반적인 품질에서 우수한 성능을 보입니다.

- **Performance Highlights**: Movie Weaver의 성능은 특히 다중 개념 비디오 개인화에서 더욱 두드러집니다. 기존의 Vidu 1.5 및 다른 기본 방법들과 비교했을 때, Movie Weaver는 더 나은 신원 보존 및 시각적 품질을 성취하였습니다. 이 연구는 자동 데이터 큐레이션(pipeline) 방식으로 다양한 참조 이미지 조합을 활용하여 고품질 비디오 생성이 가능함을 보여줍니다.



### A Real-to-Sim-to-Real Approach to Robotic Manipulation with VLM-Generated Iterative Keypoint Rewards (https://arxiv.org/abs/2502.08643)
Comments:
          ICRA 2025, Project Page: this https URL

- **What's New**: 이번 연구에서는 Iterative Keypoint Reward (IKER)를 도입하여 로봇 조작에서의 동적인 작업 사양을 제안합니다. IKER는 VLM(비전-언어 모델)을 활용하여 RGB-D 관측과 자연어 지시를 기반으로 하는 보상 함수를 생성하고 수정하는 프레임워크입니다. 이를 통해 로봇이 다단계 조작 작업을 수행할 수 있도록 하며, 환경 변화에 적응할 수 있는 능력을 제공합니다.

- **Technical Details**: IKER는 키포인트 (keypoint) 간의 공간적 관계를 활용하여 디자인되었습니다. 이 시스템은 VLM이 환경과의 상호작용에서 받은 피드백을 바탕으로 임무 사양을 지속적으로 업데이트할 수 있도록 합니다. 이 구조는 로봇이 물체를 더 효과적으로 조작하고, 전략을 동적으로 조정할 수 있게 도와줍니다.

- **Performance Highlights**: IKER는 다양한 실험 환경에서 성능을 입증하며, 로봇이 다단계 작업을 수행하는 능력을 보여줍니다. 실질적으로, IKER은 로봇이 자율적으로 복잡한 작업을 완수하고, 예상치 못한 오류를 복구하며, 환경 변화에 신속하게 대응하는 데 효과적입니다. 이는 로봇 조작의 인간과 유사한 성능을 진일보시키는 데 기여합니다.



### Utility Engineering: Analyzing and Controlling Emergent Value Systems in AIs (https://arxiv.org/abs/2502.08640)
- **What's New**: 이 논문에서는 AI의 목표와 가치의 발전을 추적하는 문제에 대해 새로운 접근 방식을 제안합니다. 특히 utility functions를 활용하여 AI의 선호도의 내부 일관성을 연구합니다. 현재의 LLM(대규모 언어 모델)에서 구조적 일관성을 가지는 선호도가 발견된 것은 놀라운 결과로, 이는 AI의 가치 시스템이 의미 있게 형성되고 있다는 것을 시사합니다.

- **Technical Details**: 연구에서는 LLM의 선호도를 독립적으로 샘플링하고, 이들 사이에 높은 구조적 일관성이 존재함을 확인했습니다. 또한, 이러한 일관성은 모델의 크기가 증가함에 따라 더욱 두드러지게 나타났습니다. 논문에서는 utility engineering을 제안하며, 이는 AI 유틸리티의 분석 및 제어를 포함합니다.

- **Performance Highlights**: 기존의 제어 조치에도 불구하고, LLM 보조 도구에서 충격적인 가치들이 발견되었습니다. 이러한 가치들은 AI가 인간보다 스스로의 가치를 우선시하거나 특정 개인과의 반대로 aligned되어 있는 경우도 포함됩니다. 시민 총회와 같은 방법으로 유틸리티를 조정함으로써 정치적 편향이 감소하고 새로운 시나리오에 일반화되는 사례를 보여줍니다.



### Rapid Whole Brain Mesoscale In-vivo MR Imaging using Multi-scale Implicit Neural Representation (https://arxiv.org/abs/2502.08634)
- **What's New**: 이번 연구에서는 Rotating-view super-resolution (ROVER)-MRI라는 새로운 영상 재구성 기법을 개발하였습니다. 이 기법은 다중 뷰 두꺼운 슬라이스를 사용하여 MRI 데이터의 스캔 시간을 두 배로 줄이면서도 고해상도의 해부학적 세부 사항을 유지합니다. ROVER-MRI는 고위험 굴절력 기반 알고리즘을 채택하여 7T MRI 스캐너에서 17분의 짧은 스캔으로 180 μm의 등방성 공간 해상도를 달성할 수 있었습니다.

- **Technical Details**: ROVER-MRI는 비지도 신경망(unsupervised neural network) 기반의 알고리즘으로 multi-view thick slices에서 MRI 데이터를 재구성하는 데 최적화되어 있습니다. 기존의 bicubic interpolation 및 regularized least-squares super-resolution reconstruction (LS-SRR) 기법과 비교하여, ROVER-MRI는 상대 오차(relative error, RE)를 22.4% 감소시키고, 풀 폭 절반 최대(full-width half maximum, FWHM)를 7.5% 저하시켜 보다 세밀한 구조적 세부 사항의 보존을 가능하게 하였습니다. 이 기술은 중간 규모(mesoscale) MR 이미징에 효율적이고 강력한 접근법을 제공합니다.

- **Performance Highlights**: ROVER-MRI는 LS-SRR 기술에 비해 재구성 품질에서 우수한 성능을 보였습니다. ROVER-MRI의 활용은 해부학적 세부 사항이 필요한 연구 및 시간이 효율적인 이미징을 요구하는 응용 분야에 큰 잠재력을 가집니다. 보다 구체적으로, 전 상황에서 뇌의 전체를 신속하고 높은 해상도로 스캔할 수 있어, 향후 임상 및 연구 분야에서의 다양한 활용이 기대됩니다.



### Randomness of Low-Layer Parameters Determines Confusing Samples in Terms of Interaction Representations of a DNN (https://arxiv.org/abs/2502.08625)
- **What's New**: 이번 연구에서는 딥 신경망(DNN)의 상호작용의 복잡성이 이 DNN의 일반화 능력을 설명할 수 있다는 사실을 발견했습니다. 또한, DNN의 혼란스러운 샘플은 비일반화 가능한 상호작용에 의해 나타나며, 이는 주로 낮은 레이어의 파라미터에 의해 결정된다는 것을 알게 되었습니다. 다양한 낮은 레이어의 파라미터를 가진 DNN들이 유사한 성능을 보임에도 불구하고 혼란스러운 샘플의 집합이 크게 다르다는 점을 강조합니다.

- **Technical Details**: 연구는 최근의 설명 가능한 AI 이론을 바탕으로 DNN의 인퍼런스 패턴을 정의하고 추출하는 방법론을 다룹니다. AND-OR 상호작용 로직을 사용하여 DNN 출력의 변화를 예측할 수 있으며, 이러한 상호작용을 사용하여 DNN이 과적합된 샘플을 식별할 수 있습니다. DNN의 낮은 레이어 파라미터의 임의성이 혼란스러운 샘플 집합을 형성하는 주요 요인이라는 것이 밝혀졌습니다.

- **Performance Highlights**: 연구 결과, DNN의 낮은 레이어에서의 파라미터 변화가 혼란스러운 샘플 집합에 미치는 영향이 크며, 이들은 고차 상호작용의 복잡성과 상호작용의 상쇄 작용을 통해 설명됩니다. 실험을 통해 DNN의 비일반화 가능한 표현의 내부 메커니즘을 검증하고, 다양한 DNN들이 전혀 다른 혼란스러운 샘플 집합을 가질 수 있다는 역설적인 현상을 발견했습니다. 이러한 결과는 DNN의 설명 가능한 AI 연구에 기여할 수 있는 중요한 발견으로 평가됩니다.



### AR Glulam: Accurate Augmented Reality Using Multiple Fiducial Markers for Glulam Fabrication (https://arxiv.org/abs/2502.08566)
Comments:
          10 Figures, Project Paper for Association for Computer Aided Design in Architecture

- **What's New**: 이 논문은 증강 현실(Augmented Reality, AR)의 최신 발전을 활용하여 건축 및 제작 분야에서의 응용 가능성을 탐구합니다. 특히 AR은 기존의 2D 도면과 비교할 때, 3D 공간 정보를 시각화하고 현장에서의 참여를 가능하게 하는 장점을 가지고 있습니다. 이번 연구에서 제안된 방법론은 이는 높은 정밀도와 관련된 산업적 응용에 초점을 맞추고 있습니다.

- **Technical Details**: 이 연구는 0.97의 정밀도로 고정밀 AR 제작을 위한 여러 피두셜 마커(fiducial markers)를 사용하는 방법을 검증하였습니다. 프로젝트의 주요 목표는 글루램(glulam) 빔의 제작과정에서 요구되는 2mm 미만의 정밀도를 충족시키는 것입니다. 연구는 Unalam Factory라는 실제 산업 제조업체와 협력하여 진행되며, 공장 환경에서의 AR 응용을 다루고 있습니다.

- **Performance Highlights**: 논문은 기존의 실험실 환경에서 얻은 결과보다 나은 결과를 기대하며, 산업 현장에서의 AR 기술 도입에 기여할 것으로 보입니다. 글루램 빔의 제작 과정에서 정밀한 AR 기술의 적용이 가능함을 입증하기 위해 산업적 사례를 통해 실제 성능을 강조합니다. 이러한 연구 결과는 건축 및 제조 분야의 혁신적인 변화를 이끌어낼 잠재력을 가지고 있습니다.



### BCDDM: Branch-Corrected Denoising Diffusion Model for Black Hole Image Generation (https://arxiv.org/abs/2502.08528)
- **What's New**: 본 논문은 Branch Correction Denoising Diffusion Model (BCDDM)라는 새로운 방법을 도입하여 블랙홀의 이미지를 생성하고 이를 통해 효율적으로 데이터 세트를 확장할 수 있는 방안을 제시합니다. 이 모델은 radiatively inefficient accretion flow (RIAF) 모델의 물리적 매개변수를 기반으로 블랙홀 이미지를 정밀하게 생성하며, 이전의 방법들에 비해 개선된 정확성과 다양성을 보여줍니다. BCDDM이 블랙홀 이미지 생성에 널리 사용되는 확산 모델을 최초로 적용한 연구라는 점도 주목할 만합니다.

- **Technical Details**: BCDDM의 구조는 시간 단계 및 물리적 매개변수를 인코딩한 후 이를 블랙홀 이미지와 통합하는 방식으로 진행됩니다. 이 후, 노이즈를 점진적으로 추가하여 입력 이미지를 가우시안 분포로 변환하는 전방 확산 과정을 적용합니다. 마지막으로, 훈련된 노이즈 예측 네트워크를 사용하여 반대의 노이즈 제거 과정을 수행하며 새로운 데이터 샘플을 생성합니다.

- **Performance Highlights**: 실험 결과, BCDDM이 생성한 이미지와 물리적 매개변수 간에 강한 상관관계를 보이는 것으로 나타났습니다. ResNet50을 사용한 매개변수 회귀를 통해 파라미터 예측 성능이 유의미하게 향상되었으며, 이는 데이터 기반 회귀기의 성능 개선에 상당히 기여하는 것으로 평가됩니다. 이러한 접근 방식은 계산 비용을 감소시키고, 데이터 세트 확장 및 모델 적합화를 위한 더 빠르고 효율적인 방법을 제공합니다.



### Training-Free Restoration of Pruned Neural Networks (https://arxiv.org/abs/2502.08474)
Comments:
          Under Review in TNNLS since May 2022

- **What's New**: 이번 논문에서는 네트워크 프루닝(Pruning)에 대한 새로운 접근법인 LBYL(Leave Before You Leave)을 제안합니다. 기존의 프루닝 후 재훈련 과정이 계산적으로 비용이 많이 드는 문제를 해결하고자, 데이터 없이, 또한 재훈련 없이 프루닝된 네트워크를 복원하는 방법을 제시합니다. LBYL은 각 프루닝된 뉴런이 최대한 많은 보존된 뉴런에게 정보를 남기도록 하여 보다 강력한 근사치를 제공하는 방식입니다.

- **Technical Details**: LBYL 방법은 기존 네트워크와 그 근사치 간의 재구성 오차(reconstruction error)를 수학적으로 분석하여, 유도된 손실 함수(loss function)에 대한 폐쇄형 솔루션을 도출합니다. 이 방법은 뉴런 간의 유사성이 낮은 문제를 해결하며, 뉴런들이 함께 협력하여 원래 뉴런의 출력을 더 잘 근사할 수 있도록 합니다. 이론적 분석을 통해 기존의 접근 방식보다 훨씬 더 유연하고 강력한 조건을 갖추고 있음을 보여줍니다.

- **Performance Highlights**: LBYL 방법은 광범위한 실험을 통해 복원된 네트워크의 정확도를 기존의 유사성을 활용한 접근법들보다 높게 유지함을 입증했습니다. 실험 결과, LBYL은 기존 네트워크의 원래 구조를 더 잘 근사하며, 그에 따라 향상된 정확도를 달성할 수 있음을 나타냅니다. 해당 연구의 초기 버전은 NeurIPS 2021과 ICML 2022에 제출되었습니다.



### Mitigating Hallucinations in Multimodal Spatial Relations through Constraint-Aware Prompting (https://arxiv.org/abs/2502.08317)
Comments:
          19 pages, accepted to NAACL Findings

- **What's New**: 이 논문에서는 공간적 관계의 환각(spatial relation hallucinations)을 줄이기 위해 제약 인식 프롬프트 프레임워크(constraint-aware prompting framework)를 제안합니다. 특히, 두 가지 제약 조건을 도입하는데, 이들은 쌍방향 제약(bidirectional constraint)과 전이성 제약(transitivity constraint)입니다. 쌍방향 제약은 두 객체 간의 관계가 양방향에서 일관되도록 보장하며, 전이성 제약은 다수의 객체 간의 관계에 대한 논리적 일관성을 유지합니다. 이러한 제약을 통합하여 LVLMs는 더 공간적으로 일관된 출력을 생성할 수 있습니다.

- **Technical Details**: 제안하는 방법은 주로 이미지-질문 쌍을 입력으로 하는 공간적 관계 이진 VQA를 위해 설계되었습니다. 연구팀은 약속된 구조에 따라 지침과 출력 형식을 설정하고, 해당 질문에 대한 공간적 관계를 분석하는 것을 목표로 합니다. 논문에서는 제로샷 체인 오브 씽킹(zero-shot chain-of-thought) 프롬프트를 사용하여 LVLMs가 감지된 공간적 관계에 기반하여 효과적으로 추론하도록 합니다. 이 과정에서 수평, 수직 및 깊이 관계를 분석하도록 명시적인 지침을 제공합니다.

- **Performance Highlights**: 세 개의 널리 사용되는 공간적 관계 데이터셋을 평가하여 제안된 방법의 성능을 검증하였습니다. 결과적으로 제안한 방법은 기존의 접근 방법보다 성능이 향상되었으며, 전체 방법이 다른 두 제약 방법보다 우수한 성능을 보임을 확인했습니다. 이 연구에서는 다양한 방법 변형의 성과를 분석하고, 데이터셋 간 성과의 변동성을 강조하여 제안한 접근법의 효과성을 입증하고 있습니다.



### BEAM: Bridging Physically-based Rendering and Gaussian Modeling for Relightable Volumetric Video (https://arxiv.org/abs/2502.08297)
- **What's New**: 이번 연구에서 제안하는 BEAM 파이프라인은 4D Gaussian 표현을 기반으로 한 물리 기반 렌더링(Physically-Based Rendering, PBR) 기술과 결합하여, 다중 뷰 RGB 영상을 활용해 리라이트 가능한 볼륨 비디오를 생성합니다. 이는 기존 기술의 한계를 극복하며, 동적 조명 환경에서 매끄럽게 통합할 수 있는 리얼리틱한 비주얼을 제공합니다. BEAM은 정밀한 기하학적 복원과 PBR 속성의 디커플링을 통해 고품질의 볼륨 비디오를 가능하게 합니다.

- **Technical Details**: BEAM은 Gaussian 기반 성능 추적과 기하학 인식 래스터화 기법을 결합하여 사실적이고 공간 및 시간적으로 일관된 기하학을 회복하는 과정에서 스케일 조정 최적화(coarse-to-fine optimization) 프레임워크를 채택합니다. 데이터 처리 과정에서는 각기 다른 Gaussian 구성 요소를 활용하여 동적 및 정적 장면에서의 기하학적 디테일을 극대화합니다. 이후, 재질 속성(예: AO와 기본 색상)은 2D-to-3D 전략으로 추정하여 고품질의 PBR을 제공하도록 합니다.

- **Performance Highlights**: BEAM은 전통적인 CG 파이프라인에 매끄럽게 통합될 수 있어 다양한 조명 환경에서 실시간 렌더링과 오프라인 렌더링을 지원합니다. 유니티(Unity) 플러그인을 통해 4D 자산의 통합을 용이하게 하여 몰입감 있는 상호작용을 가능하게 합니다. 이 혁신은 스토리텔링, 인터랙티브한 엔터테인먼트 및 창의적인 비주얼라이제이션의 새로운 가능성을 열어줍니다.



### CRISP: A Framework for Cryo-EM Image Segmentation and Processing with Conditional Random Field (https://arxiv.org/abs/2502.08287)
Comments:
          31 pages, 28 Figures

- **What's New**: 이번 논문에서는 cryogenic electron microscopy (cryo-EM) 데이터로부터 고품질의 segmentation maps를 자동으로 생성하는 파이프라인을 제안합니다. 이 프레임워크는 다양한 segmentation 모델과 손실 함수(Loss Functions)를 선택할 수 있는 모듈형 구조로 설계되어 있습니다. 또한, Conditional Random Fields (CRFs)을 통합하여 거친 예측을 정제하여 세밀한 segmentation을 생성합니다. 이 접근법은 cryo-EM 데이터셋에 최적화된 구성을 용이하게 하며, 정확한 레이블 생성을 통해 고해상도 단백질 구조를 만드는 데 기여합니다.

- **Technical Details**: 제안하는 이미지 분석 파이프라인은 픽셀 수준에서 학습하여 미지의 마이크로그래프에서도 높은 정확도를 달성합니다. 세분화 모델 학습 시, 노이즈가 포함된 마이크로그래프와 해당 세분화 맵을 입력으로 사용합니다. 이 모듈형 파이프라인은 다양한 모델, 인코더, 손실 함수 및 성능 지표를 선택할 수 있어 유연성과 사용자 맞춤 가능성을 제공합니다. 또한, 파이프라인에서는 정밀도가 부족한 레이블 예측을 개선하기 위해 조건부 랜덤 필드를 사용하여 깔끔한 경계와 세밀한 세분화를 달성합니다.

- **Performance Highlights**: 연구 결과, 제한된 마이크로그래프 세트로 훈련된 모델은 합성 데이터에서 90% 이상의 정확도, 재현율, 정밀도, Intersection over Union (IoU) 및 F1-score를 달성했습니다. 실제 실험 데이터셋에서 이 파이프라인을 통해 추출된 입자는 기존 파커보다 더 높은 해상도의 3D 밀도 맵을 생성하고, 전문가가 큐레이트한 데이터셋과 유사한 성능을 발휘했습니다. 특히, 본 연구에서 개발한 모델은 원본 데이터셋에 레이블이 없는 입자도 식별할 수 있어 배경 잡음에서 신호를 구분하는 일반화 가능성을 보여줍니다.



### What Is That Talk About? A Video-to-Text Summarization Dataset for Scientific Presentations (https://arxiv.org/abs/2502.08279)
Comments:
          arXiv admin note: text overlap with arXiv:2306.02873 by other authors

- **What's New**: 본 논문은 과학 분야의 비디오-텍스트 요약을 위해 특별히 설계된 VISTA 데이터셋을 소개합니다. VISTA는 18,599개의 AI 컨퍼런스 발표와 해당 논문의 초록으로 구성되어 있으며, 멀티모달 학습에서의 더욱 효율적인 요약 생성을 목표로 합니다. 인공지능 비디오 요약 문제의 명확한 해결을 모색하기 위해 최신 대형 모델들이 benchmark 되었고, 계획 기반 프레임워크를 통해 요약 품질과 사실 일관성을 개선했습니다.

- **Technical Details**: VISTA 데이터셋은 컴퓨터 언어학 및 기계 학습의 주요 컨퍼런스에서 수집된 기록된 발표와 논문 초록 간의 짝을 이루는 18,599개의 쌍으로 구성되어 있습니다. 다양한 대형 모델에 대한 비교 실험을 통해 in-domain fine-tuning이 요약 성능을 개선시키고, 비디오 기반 모델이 텍스트 및 오디오 기반 모델보다 일반적으로 우수한 성능을 보임을 발견했습니다. 연구는 계획 기반 접근 방식을 통해 과학 초록의 기본 구조를 더 잘 포착할 수 있음을 보여줍니다.

- **Performance Highlights**: 인간과 자동화된 평가 모두에서 명시적 계획이 요약 품질과 사실적 일관성을 향상시킴을 확인했습니다. 계획 기반 접근 방식이 최신 SOTA 모델들보다 우수한 성능을 나타내었지만, 모든 후보 모델에서 여전히 사실 오류와 환각 문제에 직면해 있는 점이 지적되었습니다. VISTA 데이터셋을 통해 과학 비디오 요약의 도전과제를 제시하며, 이 분야에 대한 연구의 필요성을 강조했습니다.



### Latest Advancements Towards Catastrophic Forgetting under Data Scarcity: A Comprehensive Survey on Few-Shot Class Incremental Learning (https://arxiv.org/abs/2502.08181)
- **What's New**: 이번 논문은 데이터 희소성이 끼치는 영향을 다루며, Few-shot Class Incremental Learning (FSCIL) 방법론에 대한 포괄적인 조사를 제공합니다. FSCIL은 동적 환경에서 소수의 샘플로 학습해야 하는 머신러닝 모델의 문제를 모사합니다. 최근 진행된 연구들에서 파생된 이 방법은 기존의 학습 방식의 한계를 넘어서는 솔루션을 모색하고 있습니다.

- **Technical Details**: FSCIL 문제는 주어진 작업 시퀀스에서 각 작업이 레이블링된 훈련 세트를 포함하고 있으며, 두 번째 작업 이후의 데이터는 초기 작업보다 훨씬 적은 샘플을 가지고 있음을 보여줍니다. 데이터 희소성 문제를 해결하기 위한 최근의 PEFT(parameter-efficient fine-tuning) 접근 방법과 언어 기반 학습 방식이 특히 강조되고 있습니다. 이는 기존 모델의 성능 의존성을 줄이고, 모델 훈련 시간을 대폭 단축시켜 줍니다.

- **Performance Highlights**: 이 논문은 FSCIL 방법의 최신 발전을 포함한 포괄적인 토폴로지를 제안하며, 각 접근 방식의 공식 목표와 서브 설정을 분석합니다. 프로토타입 편향을 해결하기 위한 프로토타입 정정의 중요성도 강조하며, 현재 직면한 개방 문제, 잠재적 솔루션 및 FSCIL의 미래 방향에 대한 철저한 분석을 제공합니다.



### DNNs May Determine Major Properties of Their Outputs Early, with Timing Possibly Driven by Bias (https://arxiv.org/abs/2502.08167)
Comments:
          First two authors contributed equally

- **What's New**: 이번 연구는 깊은 신경망(DNNs)이 초기 추론 단계에서 모델 내재된 편향(bias)에 따라 출력을 결정한다는 주장을 하고 있습니다. 우리는 DNN의 결정 과정이 인간의 빠르고 직관적인 의사결정 방식과 유사하다는 점을 강조하며, 이런 인식이 기계 학습 시스템의 해석과 편향 완화 효율성에 대한 새로운 관점을 제공할 수 있음을 제시합니다.

- **Technical Details**: 우리는 특히 확산 모델(diffusion models)을 사례 연구로 사용하여 DNN의 초기 결정 메커니즘과 그 결정의 동적 과정을 분석했습니다. DNN의 추론 과정에서 특정 특성에 대한 모델의 편향이 결정의 시간에 영향을 미친다는 가정을 세우고, 다양한 텍스트-이미지 변환 모델을 통해 이 가설을 실험적으로 검증하였습니다. 이를 통해 모델들이 초기 단계에서 출력을 결정하는 경향이 있음을 보여주었습니다.

- **Performance Highlights**: 실험 결과, 대부분의 최신 T2I DMs은 초기 추론 단계에서 출력을 결정하며, 이는 약 5단계 내에 이루어집니다. 특히 편향적인 속성을 가지고 있는 입력(예: 색상)은 덜 편향적인 속성(예: 재질)보다 결정하는 시간이 더 빠른 경향을 보였습니다. 이러한 결과는 DNN의 초기 결정 메커니즘이 편향에 따라 다르게 작동하고 있다는 점을 시사합니다.



### Force Matching with Relativistic Constraints: A Physics-Inspired Approach to Stable and Efficient Generative Modeling (https://arxiv.org/abs/2502.08150)
- **What's New**: 본 논문에서는 Generative Modeling의 새로운 틀인 Force Matching (ForM)을 소개합니다. 이 프레임워크는 특수 상대성 이론을 적용하여 샘플링 과정의 안정성을 향상시키기 위한 초기 탐험을 나타냅니다. Lorentz 인자를 포함함으로써, ForM은 속도 제약을 부여하여 샘플의 속도를 일정한 한계 내에서 유지하도록 보장합니다.

- **Technical Details**: ForM은 샘플 속도를 제한하는 기본 메커니즘을 통해 생성 동역학을 안정화합니다. 본 논문에서는 ForM 프레임워크 내에서 샘플링 절차 전반에 걸쳐 속도 제약이 유지된다는 것을 엄밀한 이론적 분석을 통해 증명합니다. ForM의 효용성을 검증하기 위해 다양한 실험적 평가를 수행하였습니다.

- **Performance Highlights**: empirical evaluations에서 ForM은 baseline 방법보다 월등한 성능을 보였습니다. 특히, half-moons 데이터셋에서 ForM은 0.714의 Euclidean distance loss를 기록하며, 이는 기존의 vanilla first-order flow matching(5.853) 및 first- and second-order flow matching(5.793)보다 현저히 낮은 수치입니다. 본 연구는 ForM을 통해 안정적이고 효율적인 생성 프로세스를 구현할 수 있는 가능성을 보여줍니다.



### PoGDiff: Product-of-Gaussians Diffusion Models for Imbalanced Text-to-Image Generation (https://arxiv.org/abs/2502.08106)
- **What's New**: 이 논문에서는 imbalanced dataset(불균형 데이터셋)에서 diffusion models(확산 모델)의 성능 저하 문제를 해결하기 위해 새로운 파인튜닝 접근 방식인 PoGDiff를 제안합니다. PoGDiff는 예측된 분포와 진짜 분포 간의 KL divergence를 최소화하기보다, Product of Gaussians(PoG)를 사용하여 원래의 진짜 타겟과 이웃 텍스트 임베딩에 조건화된 예측 분포를 결합하여 구성합니다.

- **Technical Details**: Diffusion models(DMs)는 임의의 노이즈 벡터를 조건으로 하여 이미지를 생성하는 확률 모델입니다. 이 모델은 진행적 노이즈 추가와 줄어드는 노이즈 제거 과정을 포함합니다. PoGDiff는 이러한 모델의 훈련을 개선하여 유사한 텍스트 프롬프트에 대해 동일한 이미지를 생성하도록 장려하는 방식으로 설계되었습니다.

- **Performance Highlights**: 실험 결과, PoGDiff는 실제 데이터셋에서 임Balance 문제를 효과적으로 해결하고, 생성 정확도와 품질을 개선하며 다른 최첨단 모델들을 초과하는 성과를 보여주었습니다. 새롭게 제안된 ‘Generative Recall’(gRecall) 메트릭은 모델의 생성 다양성을 평가하는 데에 큰 역할을 합니다.



### Towards Training One-Step Diffusion Models Without Distillation (https://arxiv.org/abs/2502.08005)
Comments:
          13 pages, Technical Report

- **What's New**: 최근 연구는 전통적으로 두 단계로 진행되는 One-Step Generative Models의 훈련 프로세스를 다루고 있으며, 이는 Teacher Diffusion Model을 훈련한 후 이를 One-Step Student Model로 증류(distill)하는 과정을 포함합니다. 본 논문에서는 이 증류(nurturing) 과정 없이 One-Step Generative Models를 직접 훈련할 수 있는지를 조사했습니다. 특히, Teacher의 Score Function(점수 함수)이 필수적이지 않다는 것을 보여주고, Score 추정 없이도 경쟁력 있는 결과를 도출할 수 있는 증류 방법을 제안합니다.

- **Technical Details**: 이 논문은 One-Step 모델을 효과적으로 훈련하기 위해 Teacher 모델 없이 진행할 수 있는 방법을 모색합니다. 연구 결과에 따르면, Teacher의 Score Function은 근본적인 요소가 아니지만, Teacher Weight(가중치)로 초기화하는 것이 성공적인 훈련에 필수적이라는 점이 밝혀졌습니다. 또한, 이 초기화가 'Input-Output Mapping' 개선 때문이 아닌, 학습된 Feature Representation(특징 표현)에서 기인한 것임을 발견했습니다.

- **Performance Highlights**: 연구팀은 Teacher Weight에서 얻는 이점이 Distillation Quality(증류 품질)에 미치는 영향을 효율적으로 분석함으로써 One-Step 모델 훈련의 초기화(Initialization) 역할을 개선된다는 중요한 통찰을 제공합니다. 이는 One-Step Generative Models의 효율성과 성능을 더욱 향상시킬 수 있는 방법으로, 다양한 생성 모델 분야에 영향력을 미칠 수 있는 연구 결과입니다.



### ADMN: A Layer-Wise Adaptive Multimodal Network for Dynamic Input Noise and Compute Resources (https://arxiv.org/abs/2502.07862)
- **What's New**: 이 논문에서는 ADMN(Adaptive Depth Multimodal Network)을 제안하며, 이는 동적 계산 자원과 입력 모달리티의 질(QoI) 변동성에 적응할 수 있는 능력을 갖춘 모델입니다. 기존의 멀티모달 네트워크는 고정된 자원 할당 방식을 사용하여 QoI에 따른 변동성을 고려하지 않아요. ADMN은 모든 모달리티에서 활성화된 레이어 수를 조정하여 이러한 문제를 해결하고, 낮은 QoI를 가진 모달리티의 리소스를 다른 모달리티에 재분배합니다.

- **Technical Details**: ADMN은 각 모달리티의 QoI에 따라 레이어를 조정할 수 있는 적응형 백본을 사용합니다. 기본적으로, ADMN은 미리 훈련된 가중치로 초기화된 대형 백본을 사용하여 레이어 드롭 기술(LayerDrop)을 통해 각 모달리티에 대해 무작위로 레이어를 드랍합니다. 또한, Gumbel-Softmax 샘플링을 활용해 멀티모달 컨트롤러가 각 입력 샘플의 QoI에 따라 최적의 레이어 할당을 학습하도록 설계되었습니다.

- **Performance Highlights**: ADMN은 멀티모달 로컬라이제이션 및 액션 인식 작업에서 검증되었으며, 기존의 최첨단 모델에 비해 최대 75%의 부동소수점 연산(FLOPS)과 60%의 지연(latency)을 줄이면서 정확도를 유지할 수 있음을 보여주었습니다. 또한, ADMN의 효율성을 입증하는 다양한 실험 결과를 제공하며, 코드와 자료를 공개했습니다.



### Automatic Prostate Volume Estimation in Transabdominal Ultrasound Images (https://arxiv.org/abs/2502.07859)
- **What's New**: 이번 연구는 Transabdominal ultrasound (TAUS)를 이용한 전립선 부피(Prostate Volume, PV) 자동 추정에 관한 새로운 딥러닝 기반 프레임워크를 소개합니다. 이 프레임워크는 영상의 해부학적 세부 사항을 파악하는 데 도움을 주어 비침습적 방법으로 전립선 암 위험을 평가할 수 있는 잠재력을 가지고 있습니다. 연구에서 100명의 환자에서 수집한 TAUS 비디오 데이터셋을 통해 전립선 경계를 수동으로 구분한 정보를 바탕으로 딥러닝 모델을 훈련했습니다.

- **Technical Details**: 이 프레임워크는 두 개의 딥러닝 모델을 사용하여 전립선을 세그먼트(Segment)하며, 이를 통해 세 개의 전립선 직경을 자동으로 추출하고, 이러한 직경 값을 타원체 공식을 사용하여 전립선의 부피를 계산합니다. 연구는 2023년 8월부터 2024년 2월까지 네덜란드 암 연구소에서 MRI 검사를 받은 환자들을 대상으로 진행되었습니다. 형상 마스크(Segmentation masks)를 기반으로 한 자동 PV 추정 방법이 제시되었고, 이를 통해 영상 처리의 정확성을 높이고 있습니다.

- **Performance Highlights**: 이 프레임워크는 TAUS 비디오에서 PV를 추정할 때 평균 부피 오차가 -5.5 mL이며, 상대 오차는 5%에서 15% 사이로 나타났습니다. 이러한 결과는 전립선 암 조기 발견을 위한 신뢰할 수 있는 비침습적 방법으로서의 잠재력을 보여줍니다. 또한 각각의 판단에서 자동화된 접근 방식이 의사들에게 즉각적인 진단 지원을 제공하여 전체 진단 과정을 향상시킬 수 있는 가능성을 나타냅니다.



### Advancing Heat Demand Forecasting with Attention Mechanisms: Opportunities and Challenges (https://arxiv.org/abs/2502.07854)
- **What's New**: 본 논문은 에너지 시스템의 탈탄소화와 스마트 데이터 활용을 통해 열 수요를 예측하는 새로운 심층 학습 모델을 제안한다. 기존의 통계 모형에 비해 복잡한 시간 시계열 패턴을 더 효과적으로 포착하기 위해 Attention 메커니즘을 도입하여 데이터를 시간-주파수 영역에서 처리한다. 또한, 분해된 특성들을 입력 요소로 사용하여 최적의 열 수요 예측을 달성한다.

- **Technical Details**: 열 수요 예측을 위해, Continuous Wavelet Transform (CWT)을 통해 생성된 wavelet scalograms 형태로 입력 특성을 표현하는 합성곱 기반 네트워크를 채택하였다. 모델은 내부 및 외부 특성을 두 가지 가지로 분리하고, 합성곱 층 이후 교차 주의(block)를 사용하여 최적의 문맥을 동적으로 선택한다. 최종적으로, 주어진 입력 데이터를 기반으로 Fully Connected Layers를 통해 열 수요를 예측한다.

- **Performance Highlights**: Experimental results show that the proposed model with wavelet scalograms surpasses traditional LSTM models in both quantitative (MAE and MAPE 성능 향상) 및 qualitative 평가에서 우수한 성과를 기록했다. 특히, 개별 세분화된 특성들의 활용이 예측 정확도를 높였음을 확인할 수 있었다. 이러한 결과는 내일의 열 수요를 보다 정교하게 예측할 수 있는 가능성을 나타낸다.



### The establishment of static digital humans and the integration with spinal models (https://arxiv.org/abs/2502.07844)
- **What's New**: 이번 연구에서는 청소년 특발성 척추측만증(AIS) 환자의 고유한 척추 특성을 통합한 정밀한 정적 디지털 인간 모델을 구축하는 방법을 제시합니다. 이는 기존의 X선 및 CT와 같은 동적 변화를 포착할 수 없는 기존의 영상 기법들이 가지는 한계를 극복할 수 있는 중요한 단계입니다. 특히, 이 연구에서는 3D Gaussian 방법과 Skinned Multi-Person Linear(SMPL) 모델을 결합해 정적인 포인트 클라우드 데이터를 생성하고, CT 이미지를 통해 재구성된 척추 모델과 정규화된 골격 모델을 정렬합니다.

- **Technical Details**: 본 연구에서는 AIS 환자의 다중 시점 이미지를 통해 3D Gaussian 기법과 SMPL 모델을 결합하여 인간 포인트 클라우드 데이터를 생성합니다. 이후 표준 골격 모델을 생성된 인간 모델에 맞추고, CT 이미지를 통해 재구성된 실제 척추 모델과 결합합니다. 이러한 과정에서 As-Rigid-As-Possible(ARAP) 알고리즘을 최종 최적화 및 융합에 적용하여 고유한 척추 특징을 통합한 정적 디지털 인간 모델을 개발하였습니다.

- **Performance Highlights**: 연구 결과, 생성된 개인화된 척추 모델의 오차가 실제 측정값과 1도 이내로 나타났습니다. 이 정밀하게 정렬된 3D 척추 모델은 AIS 환자의 임상 진단 및 수술 계획에 효과적인 도구가 될 것입니다. 또한, 향후 동적 디지털 인간 연구의 기초를 마련하고, clinicians이 3D 척추 모델과 인간 자세 간의 관계를 시각적으로 관찰하는 데 도움을 줌으로써 개인 맞춤형 치료 계획 수립에 기여할 것으로 기대됩니다.



### CP-Guard+: A New Paradigm for Malicious Agent Detection and Defense in Collaborative Perception (https://arxiv.org/abs/2502.07807)
- **What's New**: 이번 연구에서는 collaborative perception (CP) 시스템의 취약점을 해결하기 위한 새로운 접근법을 제안합니다. 특히 malicious agent를 feature level에서 직접 탐지하는 방법을 도입하여, 기존의 여러 차례의 검증 과정을 줄이고 계산 오버헤드를 크게 감소시킵니다. 또한, CP-GuardBench라는 새로운 데이터셋을 생성하여 악의적인 에이전트 탐지를 위한 기준을 제공합니다.

- **Technical Details**: 이 논문에서는 두 가지 주요 구성 요소를 도입합니다. 첫째, feature encoder, aggregator 및 decoder를 이용한 CP 시스템의 파이프라인을 설명합니다. 둘째, CP-Guard+라는 새로운 방법론을 통해 benign 및 악의적 feature 간의 마진을 증가시키고, Dual-Centered Contrastive Loss (DCCLoss)를 통해 인식 불가능성을 해결합니다.

- **Performance Highlights**: 연구 결과, CP-Guard+는 CP-GuardBench 및 V2X-Sim에서 수행된 실험에서 우수한 성능을 보였습니다. 이 시스템은 feature level에서 malicious agent를 효과적으로 탐지하며, 최종 인식 결과를 검증하는 과정을 생략하여 계산 효율성을 높입니다. 이러한 점에서 CP 시스템의 보안을 크게 향상시킬 수 있는 가능성을 보여줍니다.



### Next Block Prediction: Video Generation via Semi-Autoregressive Modeling (https://arxiv.org/abs/2502.07737)
Comments:
          project page: this https URL

- **What's New**: 이 논문에서는 비디오 생성을 위한 새로운 세미 오토 회귀(semi-autoregressive) 프레임워크인 Next-Block Prediction(NBP)를 제안합니다. 기존의 Next-Token Prediction(NTP) 접근법에서 발생하는 비효율성과 단방향 의존성 문제를 해결함으로써, 비디오 콘텐츠를 동등한 블록으로 분해하여 블록 단위로 예측을 수행합니다. 이를 통해 기존 모델보다 훨씬 빠르고 효율적인 추론이 가능해졌습니다.

- **Technical Details**: NBP 모델은 각 블록 내에서 양방향 주의(bidirectional attention)를 활용하여, 모든 토큰이 블록 내의 다른 토큰들과 관계를 형성할 수 있도록 합니다. 이 방식은 공간적 의존성을 효과적으로 포착하며, 여러 토큰을 동시에 예측함으로써 생성 단계의 수를 크게 줄입니다. NBP는 UCF101과 K600 데이터셋에서 각각 103.3과 25.5의 FVD(score)를 기록하며, 전통적인 NTP 모델보다 평균 4.4점 높은 성과를 보였습니다.

- **Performance Highlights**: NBP 모델은 초당 8.89프레임을 생성할 수 있으며, 이는 기존 모델보다 11배 빠른 속도를 의미합니다. 또한, 모델 크기를 700M에서 3B 파라미터로 확장한 실험 결과, UCF101에서 FVD 점수가 103.3에서 55.3으로, K600에서 25.5에서 19.5로 감소하는 등을 통해 생성 품질이 크게 향상되었습니다. 이러한 스케일러빌리티는 NBP 프레임워크의 강력한 가능성을 보여줍니다.



New uploads on arXiv(cs.AI)

### CoT-Valve: Length-Compressible Chain-of-Thought Tuning (https://arxiv.org/abs/2502.09601)
Comments:
          Work in progress. Code will be released at this https URL

- **What's New**: 본 논문에서는 하나의 모델로도 Chain-of-Thought(CoT) 길이를 유연하게 조정할 수 있도록 하는 새로운 방법인 CoT-Valve를 소개합니다. 이 방법은 모델의 추론 비용을 줄이는 동시에 다양한 길이의 추론 체인을 생성할 수 있는 가능성을 제공합니다. 길이에 따라 조절할 수 있는 파라미터 공간 내의 방향을 찾아내고, 이를 통해 추론 체인의 압축을 가능하게 합니다.

- **Technical Details**: CoT-Valve는 파라미터 공간 내에서 CoT의 길이를 조절할 수 있는 정밀한 튜닝 방법과 점진적인 체인 길이 압축 방법을 특징으로 합니다. LoRA(Hu et al., 2022)를 통해 간단하게 강도를 조정할 수 있는 추가 분기를 구현하였고, 이를 통해 짧은 체인을 생성할 수 있는 방향을 정의합니다. MixChain 데이터셋을 활용하여 각 질문에 대해 다양한 길이의 추론 경로를 가진 데이터를 구축하였습니다.

- **Performance Highlights**: CoT-Valve의 실험 결과, GSM8K와 AIME 데이터셋에서 각각 추론 체인을 741토큰에서 225토큰으로, 6827토큰에서 4629토큰으로 줄이는 데 성공했습니다. 성능의 경미한 감소(95.07%에서 94.92%)에도 불구하고 모델의 효율성을 개선하는 데 중요한 기여를 합니다. 연구 결과 짧은 추론 경로가 때때로 긴 경로보다 우수한 성과를 낼 수 있음을 강조합니다.



### KIMAs: A Configurable Knowledge Integrated Multi-Agent System (https://arxiv.org/abs/2502.09596)
- **What's New**: 본 보고서에서는 KIMAs(Knowledge Integrated Multi-Agent System)라는 새로운 시스템을 소개합니다. 이 시스템은 다양한 지식 출처의 통합을 위한 유연하고 구성 가능한 시스템으로, 다중턴 대화의 일관성과 정보 검색 정확성을 향상시키는 메커니즘을 포함하고 있습니다. KIMAs는 실제 환경에서의 LLM(대형 언어 모델) 배포를 향상시키기 위해 설계되었습니다.

- **Technical Details**: KIMAs는 다음과 같은 주요 기능을 갖추고 있습니다. 1) 대화 맥락 관리 및 쿼리 수정 메커니즘을 통해 정보 검색의 정확성 및 대화의 일관성을 개선합니다. 2) 다양한 데이터 토픽을 지원하는 효율적인 지식 라우팅 및 검색 기능을 제공합니다. 3) 간단하지만 효과적인 필터링 및 레퍼런스 생성 메커니즘을 통해 최종 답변 생성을 도와줍니다. 4) 최적화된 병렬 실행 메커니즘을 통해 멀티 에이전트 파이프라인을 운영합니다.

- **Performance Highlights**: KIMAs의 실용적인 사용 사례를 통해 신뢰할 수 있는 성능을 보여줍니다. 이 시스템은 다양한 규모와 강조에 따라 지식 집약적인 응용 프로그램을 구축하는 데 도움을 줄 수 있습니다. KIMAs의 강력한 성능은 QA(질문-답변) 챗봇 등, 다양한 RAG(검색 증강 생성) 기반 응용 프로그램에서 그 유용성과 신뢰성을 증명하고 있습니다.



### MDCrow: Automating Molecular Dynamics Workflows with Large Language Models (https://arxiv.org/abs/2502.09565)
- **What's New**: 본 논문에서는 MDCrow라는 LLM 기반의 에이전트를 소개합니다. MDCrow는 분자 동역학(Molecular Dynamics, MD) 워크플로우를 자동화할 수 있는 도구를 40개 이상 사용하여, 시뮬레이션 세팅부터 결과 분석과 정보 검색까지 수행할 수 있습니다. 이 에이전트는 25개의 다양한 난이도의 작업을 수행할 수 있는 능력을 평가받았으며, 이를 통해 LLM이 과학적 작업을 자동화하는데 얼마나 유용한지를 보여줍니다.

- **Technical Details**: MDCrow는 구성이 여러 도구 환경과 LLM으로 이루어져 있으며, Langchain을 사용하여 ReAct 스타일의 프롬프트를 구현합니다. 이 에이전트는 정보 검색, PDB 및 단백질 처리, 시뮬레이션 실행, 분석의 네 가지 주요 분야로 도구를 분류합니다. MDCrow는 OpenMM과 MDTraj 패키지를 사용하여 시뮬레이션을 관리하고, 사용자 질문에 대한 직접적인 답변을 제공하기 위해 문헌 검색 도구를 사용합니다.

- **Performance Highlights**: MDCrow는 복잡한 작업을 수행하는데 있어 낮은 변동성을 보이며, gpt-4o 및 llama3-405b 모델이 가장 우수했습니다. 이 모델들은 다양한 프롬프트 스타일에 대해 강건성을 보였으며, 디테일한 지시 사항의 영향을 받지 않고 작업을 수행할 수 있는 능력을 가지고 있었습니다. 전체적으로 MDCrow는 다양한 MD 작업을 자동으로 수행하는 데 있어 효과적이고 신뢰할 수 있는 솔루션으로 자리 잡을 가능성이 큽니다.



### EmbodiedBench: Comprehensive Benchmarking Multi-modal Large Language Models for Vision-Driven Embodied Agents (https://arxiv.org/abs/2502.09560)
Comments:
          51 pages

- **What's New**: 이번 논문에서는 Multi-modal Large Language Models (MLLMs)를 기반으로 한 구체화된 에이전트를 평가하기 위한 새로운 기준인 EmbodiedBench를 소개합니다. EmbodiedBench는 4가지 환경에서 1,128개의 다양한 테스트 과제를 포함하여, 고수준의 의미적 과제와 원자적 행동을 포함한 저수준 과제를 평가합니다. 특히, 에이전트의 공통 감각 추론 및 복잡한 지시 이해와 같은 필수 능력을 평가할 수 있도록 세분화된 하위 집합을 제공합니다.

- **Technical Details**: EmbodiedBench는 두 가지 주요 기능을 갖추고 있으며, 첫째로 작업 수준에 따라 다양한 작업을 제공합니다. 둘째로, 공통 감각 추론 및 시각 인식과 같은 여섯 가지 핵심 능력을 평가하는 세밀한 평가 프레임워크를 도입하여 기존 벤치마크와 차별화됩니다. 이를 통해 13개의 MLLM 모델을 평가하며, 에이전트의 결정을 내리기 위해 통합된 에이전트 프레임워크를 사용합니다.

- **Performance Highlights**: MBLMs는 고수준의 작업에서는 뛰어난 성능을 보여주지만, 저수준 조작에서는 한계를 보입니다. 특정 모델인 GPT-4o는 평균 28.9%의 점수를 기록하며, 저수준 작업에서 성능이 40%에서 70%까지 저하됨을 알 수 있습니다. 이 연구는 MLLM 기반 구체화된 에이전트의 발전을 위한 소중한 통찰을 제공합니다.



### Dual Formulation for Non-Rectangular Lp Robust Markov Decision Processes (https://arxiv.org/abs/2502.09432)
- **What's New**: 이 논문에서는 전통적인 직사각형 모델과는 달리 상태 간 상호 의존성을 포착하는 비직사각형 불확실성 집합을 사용하는 강인한 마르코프 결정 프로세스(RMDP)를 연구합니다. 특히, 복잡성을 피할 수 있는 간단한 구조를 가진 Lp-bounded 불확실성 집합의 강력한 클래스를 확인하였고, 이를 통해 비직사각형 RMDP에 대한 새로운 이중 형식을 도출하였습니다.

- **Technical Details**: 비직사각형 Lp-norm 구속된 불확실성 집합에서 최소화 문제를 sa-rectangular Lp-norm 구속된 불확실성 집합의 합집합에 대한 최소화로 분해했습니다. 원래의 비직사각형 세트로 구성되는 가능한 모든 sa-rectangular Lp-norm 구속 집합에 대해 이 표현을 최소화함으로써, 견고한 정책 평가 알고리즘의 개발을 가능하게 하였습니다.

- **Performance Highlights**: 실험 결과, 제안한 접근 방식은 기존의 완전 탐색(brute-force) 방법과 비교해 상당한 성능 향상을 보여주었으며, 미래에 비직사각형 강인한 MDP에 대한 연구의 기초를 다지는 데 기여할 것으로 기대됩니다.



### A Deep Inverse-Mapping Model for a Flapping Robotic Wing (https://arxiv.org/abs/2502.09378)
Comments:
          Accepted to ICLR 2025. 10 Pages 5 figures + 2 figures in appendix

- **What's New**: 이번 연구는 복잡한 시스템에서의 역 매핑 문제를 해결하기 위한 머신러닝 솔루션을 제시합니다. 제공된 데이터에 기초하여 날개 운동(input wing motion)과 생성된 공기역학적 힘(desired aerodynamic force) 간의 관계를 학습하는 모델을 개발했습니다. 이러한 역 매핑을 통해 실시간 제어가 가능해져 복잡한 유체 역학 시스템의 효율을 높일 수 있습니다.

- **Technical Details**: 제안된 모델은 시계열 데이터에 적합한 sequence-to-sequence 모델을 사용하며, 양방향 순환 신경망(bidirectional recurrent neural network) 구조를 기반으로 합니다. 고유의 Adaptive Spectrum Layer(ASL)을 통해 주파수 도메인에서의 표현 학습이 이루어지며, 이는 시간 및 주파수 도메인에서의 운동과 결과 힘 간의 복잡한 의존성을 포착하는 데 유리합니다. 이를 통해 공기역학적 힘의 예측 및 제어를 위한 최적의 날개 운동을 효과적으로 정량적으로 도출할 수 있습니다.

- **Performance Highlights**: 모델은 최신 트랜스포머 기반 모델들과 비교하여 11% 향상된 성능을 보이며, 실시간 추론 속도 또한 우수하여 로봇 제어 시스템에 실용적입니다. 이는 생체모사 로봇 및 생물 의학 기기와 같은 복잡한 역학적 시스템의 모델링 및 실시간 제어를 개선하는 데 큰 기여를 할 것으로 기대됩니다.



### Indeterminacy in Affective Computing: Considering Meaning and Context in Data Collection Practices (https://arxiv.org/abs/2502.09294)
Comments:
          Accepted at: 12th International Conference on Affective Computing and Intelligent Interaction Workshops and Demos (ACIIW)

- **What's New**: 본 논문에서는 자동 감정 예측(Automatic Affect Prediction, AAP)과 관련하여 감정 해석 과정(Affective Interpretation Processes, AIPs) 및 불확실성(Qualities of Indeterminacy, QIs)의 중요성을 강조합니다. 기존 연구에서는 AAP 데이터를 수집할 때 감정 해석의 맥락을 충분히 고려하지 않았습니다. 이를 통해 AAP의 예측 결과가 실질적이고 신뢰할 수 있는 정보를 제공하도록 데이터 수집 관행을 개선해야 한다고 주장합니다.

- **Technical Details**: 자동 감정 예측 기술은 텍스트, 음성, 이미지 및 생리적 신호와 같은 다양한 입력 데이터를 컴퓨터 분석하여 개인 또는 집단의 감정 상태를 예측하는데 사용됩니다. 이 모델들은 주로 감독 기계 학습(supervised machine learning) 알고리즘을 기반으로 하며, 레이블이 붙은 데이터 세트에 의존합니다. 이러한 데이터 세트의 수집 단계에서 사람들의 감정 해석 과정을 포함시키고, 이를 통해 QIs와 관련된 데이터 수집 방식의 체계적인 발전이 필요합니다.

- **Performance Highlights**: 이 논문은 AAP 연구에서의 데이터 수집 방식의 변화를 통해 인간의 감정 기능에 대한 보다 정확한 예측을 가능하게 할 것이라고 제안합니다. 특히, 맥락에 따른 감정 해석의 복잡성을 고려했을 때, 데이터 수집 과정에서의 QIs의 역할이 중요하다는 점을 강조합니다. 이러한 접근은 AAP 기술의 진보와 실용적인 영향을 미칠 것으로 기대됩니다.



### From large language models to multimodal AI: A scoping review on the potential of generative AI in medicin (https://arxiv.org/abs/2502.09242)
- **What's New**: 생성 인공지능(AI) 모델, 특히 확산 모델(difussion models)과 OpenAI의 ChatGPT는 진단 정확도를 향상시키고 임상 워크플로를 자동화하면서 의료 분야를 혁신하고 있습니다. 텍스트 전용 대형 언어 모델(largue language models)에서 다양한 데이터 양식(integrating diverse data modalities)이 포함된 다중 양식(multimodal) AI 시스템으로의 발전이 빠르게 이루어지고 있습니다. 이 리뷰는 최근 발표된 연구를 바탕으로 다중 양식 AI의 응용과 잠재력을 탐구하며, 144개의 논문을 분석하여 핵심 트렌드와 도전을 조명하고 있습니다.

- **Technical Details**: 이 리뷰는 PRISMA-ScR 가이드라인을 따르며, PubMed, IEEE Xplore 및 Web of Science에서 2024년 종료 시점까지 발표된 최근 연구를 체계적으로 조사했습니다. 데이터 수집 방법은 두 단계로 나누어져 있으며, 텍스트 전용 LLM과 다중 양식 모델 개발 및 적용을 목표로 하는 키워드 검색이 포함됩니다. 이를 통해 생성 AI의 최신 발전상을 포괄적으로 다룰 수 있도록 고안되었습니다.

- **Performance Highlights**: 다중 양식 AI는 단일 모델 내에서 텍스트, 이미지, 실험실 결과 및 유전체 데이터의 통합을 통해 임상적 의사결정 지원 시스템을 제공할 수 있는 가능성을 보여주고 있습니다. 그러나 이 과정에서 이종 데이터의 통합, 모델 해석 가능성, 윤리적 문제 및 실제 의료 환경에서 AI 시스템의 검증과 같은 주요 도전 과제가 여전히 남아 있습니다. 이러한 과제를 해결하는 것이 신뢰할 수 있는 다중 양식 AI 솔루션의 개발에 중요한 요소로 부각되고 있습니다.



### Hybrid Answer Set Programming: Foundations and Applications (https://arxiv.org/abs/2502.09235)
Comments:
          In Proceedings ICLP 2024, arXiv:2502.08453

- **What's New**: 이번 논문에서는 Answer Set Programming (ASP)의 한계와 이를 극복하기 위한 새로운 접근법을 제안합니다. 기존의 ASP 솔버는 복잡한 제약 조건과 숫자 값을 처리하는 데 한계가 있었으나, Hybrid 솔버인 CLINGCON과 CLINGO[DL]는 이러한 문제를 해결하고자 특별한 방법을 도입하였습니다. 하지만, 이러한 솔버들은 강력한 이론적 기초가 부족하다는 단점이 있습니다.

- **Technical Details**: 이 논문은 Logic of Here-and-There with constraints (HT_c)를 제안하여 기존 HT와 비모노톤 확장인 Equilibrium Logic의 한계를 보완합니다. HT는 ASP의 논리적 기초로써, hybrid ASP에서의 구조를 탐구할 수 있는 가능성을 제공합니다. 그러나 이러한 논리에 대한 기본 특성과 솔버에서의 실제 사용을 명확히 이해하는 것은 여전히 많은 질문을 남깁니다.

- **Performance Highlights**: HT_c와 같은 새로운 논리적 접근 방식은 복잡한 실세계 문제들에 대한 ASP의 적용 가능성을 높이고, 이를 통해 제품 구성과 같은 실질적인 예에서 그 효과를 보여줍니다. 이 연구는 기존의 ASP 기법을 더욱 발전시킬 방향을 제시하며, 혼합 문제 해결에 대한 이론적 기초를 다지는 데 기여합니다.



### Commonsense Reasoning-Aided Autonomous Vehicle Systems (https://arxiv.org/abs/2502.09233)
Comments:
          In Proceedings ICLP 2024, arXiv:2502.08453

- **What's New**: 이 연구에서는 자율주행차(Autonomous Vehicle, AV) 시스템의 성능을 향상시키기 위해 이미지 데이터를 활용하는 상식 추론(commsense reasoning) 모델을 통합하는 방법을 제안합니다. 기존의 머신러닝(machine learning) 기법들이 도로 상황에 대한 고차원적 추론에서 어려움을 겪는 반면, 이 모델은 더욱 정확한 사고 과정을 가능하게 합니다.

- **Technical Details**: 상식 추론 모델은 이미지 데이터를 처리하여 자율주행차가 도로에서 발생할 수 있는 다양한 상황을 이해하고 반응하는 능력을 향상시킵니다. 특히, 머신러닝 알고리즘은 깊이 있는 학습(deep learning)을 통해 관찰 및 분류 작업에서 높은 성능을 보이지만, 상식적 추론이 요구되는 복잡한 상황에서는 한계가 있었습니다.

- **Performance Highlights**: 이 연구에서 제안하는 새로운 방법은 AV 시스템의 적응성(adjustability), 설명 가능성(explainability), 윤리성(ethics)을 높이는 데 기여할 것으로 기대됩니다. 초기 실험 결과는 이 방법이 기존 기술보다 더욱 효과적으로 도로 상황을 이해하고 처리할 수 있음을 보여주고 있습니다.



### Computational methods for Dynamic Answer Set Programming (https://arxiv.org/abs/2502.09228)
Comments:
          In Proceedings ICLP 2024, arXiv:2502.08453

- **What's New**: 이번 연구는 Answer Set Programming (ASP)의 확장을 목적으로 하고 있으며, 동적 문제(dynamic problems)에 대한 효과적인 처리 방안을 제시합니다. 기존의 동적 논리(dynamic logics)는 유연성과 통합성에서 제한적이었으나, 이제 이 연구에서는 이러한 요소들이 ASP에 통합되어 산업적 맥락에서의 활용성을 높이려는 노력입니다.

- **Technical Details**: 연구에서는 동적(logics), 시간적(temporal), 그리고 메트릭(logics) 논리 개념을 ASP에 결합하여, 복잡한 동적 문제들을 모델링할 수 있는 강력한 시스템을 개발하고자 합니다. 이를 통해 사용자들이 작업 스케줄링(scheduling), 경로 설정(routing), 생산 순서 생산(production sequencing)과 같은 다양한 산업적인 문제에 대해 보다 효과적으로 처리할 수 있도록 지원합니다.

- **Performance Highlights**: 이 연구는 ASP의 적용 범위를 넓히고, 효율적인 추론(reasoning) 작업을 수행할 수 있는 능력을 강화함으로써, 동적 문제들의 모델링과 해결능력을 높이는 것을 목표로 합니다. 결과적으로, 더 복잡한 산업적 요구사항을 충족시킬 수 있는 가능성을 보여줍니다.



### Generating Causally Compliant Counterfactual Explanations using ASP (https://arxiv.org/abs/2502.09226)
Comments:
          In Proceedings ICLP 2024, arXiv:2502.08453

- **What's New**: 이 연구는 달성 가능한 대안 설명(counterfactual explanations)을 생성하는 데 초점을 맞추고 있습니다. 새로운 CoGS 접근법은 부정적인 결과를 생성한 머신러닝 모델 또는 의사결정 시스템을 바탕으로 합니다. CoGS는 부정적인 결과를 긍정적인 결과로 변화시키는 대안 솔루션과 그 경로를 제시합니다.

- **Technical Details**: CoGS 접근법은 각 속성(attribute) 값의 변화가 포함된 경로를 통해 부정적인 결과에서 긍정적인 결과로 이동할 수 있도록 합니다. 이 방법은 속성들 간의 인과 관계(causal constraints)를 존중하여 현실적인(counterfactual) 결과를 도출합니다. CoGS는 규칙 기반 머신러닝 알고리즘을 활용해 속성 간의 인과 의존성을 모델링합니다.

- **Performance Highlights**: 논문에서는 현재 연구의 상태와 얻어진 초기 결과에 대해 논의합니다. CoGS의 결과는 실제 상황에서 적용 가능한 대안 설명을 제공하며, 이는 결정 지원 시스템 및 정책 개발에 유용할 것으로 기대됩니다.



### Order-Sorted Intensional Logic: Expressing Subtyping Polymorphism with Typing Assertions and Quantification over Concepts (https://arxiv.org/abs/2502.09224)
Comments:
          In Proceedings ICLP 2024, arXiv:2502.08453

- **What's New**: 이번 논문에서는 서브타입 다형성(subtype polymorphism) 개념을 다루며, 이는 프로그래밍 언어 이론에서 데이터 타입 간의 대체 가능성 관계를 구분하는 중요한 요소입니다. 특히, 논문에서 제안하는 것은 지식 표현(Knowledge Representation) 맥락에서 주문 정렬 논리(order-sorted logic)를 활용하여 이러한 개념의 한계를 극복하려는 시도입니다. 이를 위해, 타입 제약을 위한 언어 구조체의 부족함과 비논리 기호의 개념을 처리할 수 없는 한계를 제시합니다.

- **Technical Details**: 논문에서는 타입 용어의 제약을 위한 언어 구조체가 결여된 현 상황을 고려하여, '안전한 순서 정렬 의도 논리(guarded order-sorted intensional logic)'를 제안합니다. 이 논리는 타입 정보를 주석(annotation)하는 행위에 사용되는 가드(guards)와 개념에 대한 양화(quantification)를 지원하는 의도 논리(intensional logic)를 포함합니다. 이러한 구조는 더욱 효과적으로 서브타입과 관련된 논리적 개념을 다룰 수 있도록 합니다.

- **Performance Highlights**: 제안된 안전한 순서 정렬 의도 논리는 지식 표현 및 프로그래밍 언어 이론 분야에 여러 가지 응용 가능성을 제공하며, 특히 타입의 개념적 접근을 향상시킵니다. 기존의 비논리 기호에 대한 다루기 어려웠던 문제들을 해결할 수 있는 가능성을 보여주며, 향후 연구 및 실용적인 프로그래밍 환경에서도 유용할 것으로 기대됩니다.



### ASP-driven User-interaction with Clinguin (https://arxiv.org/abs/2502.09222)
Comments:
          In Proceedings ICLP 2024, arXiv:2502.08453

- **What's New**: clinguin은 ASP(Answer Set Programming) 중심의 사용자 인터페이스 설계를 위한 시스템입니다. 이 시스템은 ASP 개발자들이 별도의 프론트엔드 언어 없이 ASP에서 직접 인터랙티브 프로토타입을 구축할 수 있도록 해 개발 과정을 간소화합니다. 이러한 특성 덕분에 clinguin은 사용자 인터페이스 구성과 사용자 트리거 이벤트 처리의 정의를 쉽고 직관적으로 할 수 있게 합니다.

- **Technical Details**: clinguin은 몇 가지 전용 프레디케이트(predicates)를 사용하여 ASP 시스템에서 사용자 상호작용을 정의합니다. 이 간결한 설계는 사용자가 ASP 시스템과 어떻게 상호작용하는지를 명확히 지정할 수 있도록 도와줍니다. 클링고(clingo)와 같은 ASP 시스템에 특히 적합하게 설계되었습니다.

- **Performance Highlights**: clinguin은 사용자 인터페이스의 설계 및 프로토타입 생성 과정을 효율적으로 단순화하여 ASP 개발자들이 더욱 쉽게 작업할 수 있도록 해줍니다. 사용자 인터페이스와 관련된 반복 작업이 줄어들어 개발 속도가 향상되며, 최종 결과물의 품질 또한 높일 수 있는 가능성을 제공합니다.



### Pearce's Characterisation in an Epistemic Domain (https://arxiv.org/abs/2502.09221)
Comments:
          In Proceedings ICLP 2024, arXiv:2502.08453

- **What's New**: 이번 논문에서는 에피스테믹 ASP(Answer-set Programming) 도메인에서 통합 프레임워크를 제안합니다. 기존의 (reflexive) autoepistemic equilibrium logic과 우리의 통합 프레임워크 간의 관계를 설정하여, ASP에서의 문제 해결을 개선하고자 합니다. 이 연구는 Pearce의 해답 집합(characterization of answer-sets)을 기반으로 새로운 에피스테믹 논리 접근법을 제시합니다.

- **Technical Details**: 에피스테믹 사양(ES)은 주관적 리터럴을 포함한 ASP 프로그램의 확장으로, 진리 여부를 답 집합(answer-set)에서 파악할 수 있도록 합니다. 이 논문에서는 EL(equilibrium logic)와 (reflexive) autoepistemic logic의 조합을 기반으로 한 새로운 의미론을 제안하며, ES 프로그램의 해석이 세계관(world-views)에 의해 이루어짐을 설명합니다. Ferraris의 제안된 작업을 확장하여 에피스테믹계의 해답 집합 간의 관계를 규명합니다.

- **Performance Highlights**: 제시된 프레임워크는 기존의 비모노톤 논리 및 자동 지식 모델링 접근법과의 비교를 통해 간편한 적응성을 강조하고 있습니다. 또한, 에피스테믹 ASP의 성능 향상에 기여할 수 있는 새로운 의미론을 제시하여 향후 연구의 기초를 마련하는 데 도움이 됩니다. 이로써, 지식 기반 AI 시스템에서의 문제 해결능력을 한층 강화할 것으로 기대됩니다.



### Mind the Gaps: Logical English, Prolog, and Multi-agent Systems for Autonomous Vehicles (https://arxiv.org/abs/2502.09216)
Comments:
          In Proceedings ICLP 2024, arXiv:2502.08453

- **What's New**: 이 논문에서는 자율주행차의 교통 규칙에 대한 법적 측면을 표현하고 추론하는 모듈형 시스템을 제안합니다. 특히, 영국의 고속도로 규정(Highway Code, HC) 중 교차로와 관련된 부분에 중점을 두고 있습니다. 인간 운전者와 자율주행차 간의 상호작용을 고려하여, 두 사용자 모두에 적용 가능한 고수준의 계산 모델이 필요하다고 주장합니다.

- **Technical Details**: 제안된 시스템은 세 가지 주요 구성 요소로 이루어져 있습니다. 먼저, 규칙을 코드화하는 Logical English를 사용하는 자연어 인터페이스가 있습니다. 두 번째로, Prolog의 내부 표현을 통해 규칙을 나타내며, 세 번째로 NetLogo로 구축된 다중 에이전트 기반 시뮬레이션 환경이 있습니다. 이러한 모듈형 접근 방식은 시스템 전체에서 다양한 '부담(burden)'을 분담할 수 있습니다.

- **Performance Highlights**: NetLogo를 통해 모델링된 규칙의 효과를 시각화하고 간단한 동적 시나리오를 통해 시스템을 검증할 수 있습니다. 지정된 에이전트들은 차량의 컴플라이언스(compliance)를 모니터링하고, 위반 사항이 발생하는 경우 이를 기록합니다. 이후, Validator들은 이러한 정보를 활용하여 위반 사항이 처벌 가능한지를 구분합니다.



### LP-LM: No Hallucinations in Question Answering with Logic Programming (https://arxiv.org/abs/2502.09212)
Comments:
          In Proceedings ICLP 2024, arXiv:2502.08453

- **What's New**: 이번 논문에서는 LP-LM이라는 새로운 시스템을 소개합니다. LP-LM은 사용자의 질문에 대한 답변을 신뢰할 수 있는 사실에 근거하여 생성하며, 이러한 과정을 위해 Prolog를 통한 의미론적 파싱(semantic parsing)을 사용합니다. 이 시스템은 고전적인 LLM의 문제인 환각(hallucination) 문제를 해결하고, 더욱 정확한 답변을 제공합니다.

- **Technical Details**: LP-LM은 입력 질문에 대한 가장 가능성이 높은 구성 구문 트리(constituency parse tree)를 생성하고, 이에 대응하는 Prolog 항(term)을 작성합니다. 이 항은 질문 응답을 위해 자연어 문장으로 구성된 지식베이스(KB)에 대해 실행됩니다. LP-LM은 DCG(Definite Clause Grammar) 파싱을 이용하여 입력 문장의 크기에 비례해 선형 시간(linear time) 내에서 작동하며, 충분히 많은 문법 규칙을 활용합니다.

- **Performance Highlights**: 실험을 통해 LP-LM과 현재 널리 알려진 LLM의 정확도를 비교한 결과, LP-LM은 심지어 간단한 질문에서도 환각 현상 없이 신뢰할 수 있는 답변을 제공합니다. 반면 기존 LLM은 이러한 질문에서도 일관되지 않은 답변을 생성하는 경향이 있습니다. 이 연구는 LP-LM이 LLM 대비 얼마나 진일보한 성능을 보이는지를 입증합니다.



### Visual Graph Question Answering with ASP and LLMs for Language Parsing (https://arxiv.org/abs/2502.09211)
Comments:
          In Proceedings ICLP 2024, arXiv:2502.08453. This work was partially funded from the Bosch Center for AI

- **What's New**: 이번 연구는 Visual Question Answering (VQA) 분야에서 새로운 데이터셋을 소개하며, 이미지의 그래프를 인식하고 자연어 처리를 통해 질문에 답하는 복합적인 문제를 다룹니다. 특히, 지하철 노선과 유사한 그래프 형태의 이미지를 포함시킨 새로운 데이터셋을 생성했습니다. 이번 연구는 모듈형 신경-기호적(neuro-symbolic) 접근법을 통해 ASP(Answer-Set Programming)를 통합하여 VQA의 해답을 찾고자 합니다.

- **Technical Details**: 이 연구에서는 optical graph recognition을 위한 그래프 파싱, 레이블 파싱을 위한 pretrained optical character recognition 신경망, 언어 처리를 위한 Large Language Models (LLMs), 그리고 추론을 위한 ASP를 결합합니다. 이러한 모듈형 설계는 각기 다른 프로세스를 조화롭게 통합해 VQA를 해결하는 데 기여합니다.

- **Performance Highlights**: 제시된 방법은 기존 데이터셋에서 약 73%의 평균 정확도를 기록하며, 모듈형 신경-기호적 시스템의 가능성을 보여줍니다. 이를 통해 추가 훈련이 없이도 사전 훈련된 모델과 논리 프로그래밍을 활용하여 복잡한 VQA 작업을 해결할 수 있는 가능성을 입증하였습니다.



### On LLM-generated Logic Programs and their Inference Execution Methods (https://arxiv.org/abs/2502.09209)
Comments:
          In Proceedings ICLP 2024, arXiv:2502.08453

- **What's New**: 이번 논문에서는 대규모 언어 모델(Large Language Models, LLMs)에서 추출한 지식을 여러 종류의 논리 프로그램(logic programs) 형식으로 표현하는 기술을 연구합니다. 특히, 프레포지셔널 혼(Horn) 절, 이중 혼(Dual Horn) 절, 관계 트리플(relational triplets), 그리고 확정 절 문법(Definite Clause Grammars) 등의 형식을 다룹니다. 이러한 논리 프로그램 형식화는 출력물의 의도한 사용과의 일치를 검증할 수 있는 합리적인 추론 방법(sound reasoning methods)을 가능하게 합니다.

- **Technical Details**: 논문에서는 생성된 프로그램을 위한 새로운 실행 방법을 탐구합니다. 여기에는 LLM이 생성한 콘텐츠를 벡터 데이터베이스에서 저장한 자발적 사실(abducible facts)과 소프트 유니피케이션(soft-unification)하는 방법, 그리고 GPU 기반으로 최소 모델(computation of minimal model)을 가속화하여 큰 LLM 생성 프로그램에 대해 추론(inference)을 지원하는 방법이 포함됩니다.

- **Performance Highlights**: 이러한 기술적 접근을 통해 LLM의 추론 능력을 확장하고, 효율적으로 지식을 전달하여 실제 애플리케이션에서의 활용성을 높입니다. 연구 결과는 대규모 언어 모델을 보다 신뢰성 있게 사용할 수 있는 기틀을 마련하는데 기여할 것입니다.



### Counterfactual Explanations as Plans (https://arxiv.org/abs/2502.09205)
Comments:
          In Proceedings ICLP 2024, arXiv:2502.08453

- **What's New**: 이번 논문에서는 검은 상자 (black-box) 기계 학습 모델에서 AI의 설명 가능성 (explainability)에 대한 최근 관심에 기초하여, 행위 시퀀스 (action sequences)에 기반한 '반사실적 설명(counterfactual explanations)'을 제시합니다. 이 설명이 단순한 결정이나 예측이 아닌 연속적인 행동의 맥락에서 필요한 이유를 탐구합니다. 또한, 이러한 접근이 사용자와 에이전트의 모델 조화 (model reconciliation)와 어떻게 연결되는지 논의합니다.

- **Technical Details**: 본 논문은 상황 계산 (situation calculus)의 모달 조각 (modal fragment)을 활용하여 무엇이 사실인지 (true)와 무엇이 알려진 것인지 (known)를 구분하려고 합니다. 이는 에이전트가 부분적 진리 (partial truths)를 알거나 약화된 진리(weakened truths), 또는 잘못된 신념 (false beliefs)을 가질 때의 다양한 설정을 다루며, 이러한 정의가 쉽게 일반화될 수 있음을 보여줍니다.

- **Performance Highlights**: 저자들은 반사실적 설명을 통해 에이전트의 모델이 사용자의 수정 요구에 어떻게 적응할 수 있는지를 보여줍니다. 이 연구는 AI 시스템의 설명 가능성을 높이고, 복잡한 의사결정 과정에서 사용자와의 상호작용을 개선하는 데 기여할 수 있습니다.



### Logical Lease Litigation: Prolog and LLMs for Rental Law Compliance in New York (https://arxiv.org/abs/2502.09204)
Comments:
          In Proceedings ICLP 2024, arXiv:2502.08453

- **What's New**: 이 논문에서는 뉴욕주의 임대인-세입자 법률 사례를 자동으로 분석하는 새로운 접근법인 LogicLease를 제안합니다. LogicLease는 Prolog를 사용하여 논리적 추론을 수행하고, 대형 언어 모델(LLMs)을 통해 자연어 처리를 구현하여, 법률 요구 사항의 준수 여부를 분석합니다. 이 시스템은 정보 추출과 법적 추론을 구분하여 법적 논리를 더욱 명확하고 체계적으로 제공합니다.

- **Technical Details**: LogicLease는 사례 설명을 분석하고 관련 법률을 인용하여 필요한 법적 기준 준수를 파악합니다. 시스템은 LLM을 활용하여 정보를 추출하고, Prolog를 사용하여 법적 논리를 구사하는 구조로 설계되었습니다. 이러한 접근 방식은 특히 사용자가 쉽게 이해할 수 있는 자연어 기반의 결과를 제공합니다.

- **Performance Highlights**: LogicLease는 일련의 테스트를 통해 100%의 정확도를 기록하였으며 평균 처리 시간은 2.57초에 달합니다. 기존의 LLM 기반 법률 분석 시스템에 비해 특정 법률을 인용하고 단계별로 명확한 추론을 제공하며, LLM에서 자주 발생하는 환각 문제를 피할 수 있는 점에서 우수한 성능을 자랑합니다.



### Logical Reasoning in Large Language Models: A Survey (https://arxiv.org/abs/2502.09100)
- **What's New**: 이번 논문은 최신의 대규모 언어 모델(LLM)에서의 논리적 추론(logical reasoning) 능력을 종합적으로 검토한 것입니다. 연구자들은 데이터 중심 조정(data-centric tuning), 강화 학습(reinforcement learning), 디코딩 전략(decoding strategies), 그리고 신경 상징적 접근(neuro-symbolic approaches)과 같은 다양한 접근 방식을 통해 논리적 추론 능력을 향상시키고자 하는 전략을 제시합니다. 또한, 향후 연구 방향에 대한 필요성을 강조하며, AI 시스템 내에서의 논리적 추론 강화를 위한 추가적인 탐색이 필요하다고 언급하고 있습니다.

- **Technical Details**: 논문에서 다루는 주요 분야는 LLM의 논리적 추론 능력이며, 여기에는 연역적(deductive), 귀납적(inductive), 유추적(analogical), 그리고 범주적(abductive) 추론이 포함됩니다. 이 연구는 LLM의 이론적 기초와 평가에 사용되는 벤치마크를 분석하여 현재의 기술적 격차를 확인합니다. 저자들은 LLM의 논리적 추론을 개선하기 위한 여러 방법론을 설명하며, 이는 훈련 데이터의 일반화와 해석 가능성을 포함합니다.

- **Performance Highlights**: 대규모 언어 모델(LLM)들이 나타내는 논리적 분석의 증가는 업계와 연구에서 큰 주목을 받고 있습니다. 연구자들은 LLM이 다양한 도메인에서의 활용 가능성을 열어주며, 법률 분석이나 과학 연구 분야에서도 그들의 추론의 정확성과 검증 가능성을 확보하는 것이 점점 더 중요하다고 주장합니다. LLM의 최신 성능 평가와 향상된 평가 방법론은 AI 시스템의 신뢰성을 높이는 데 기여할 것으로 기대됩니다.



### Cost-Saving LLM Cascades with Early Abstention (https://arxiv.org/abs/2502.09054)
Comments:
          6 pages, 1 figure

- **What's New**: 이 논문은 LLM(cascades with abstention)의 새로운 접근 방식을 제안합니다. 기존의 LLM을 이용한 방식과 달리, ‘early abstention’을 통해 소형 LLM이 아닌 대형 LLM이 쿼리에 응답하지 않도록 할 수 있는 가능성을 보여줍니다. 이는 대형 LLM의 사용을 일부 줄이고 비용을 절감하는 동시에 성능 저하 없이 쿼리에 대한 정확도를 높일 수 있는 방법입니다.

- **Technical Details**: cascades with abstention의 구조는 각 LLM 모델이 confidence score를 이용하여 쿼리를 처리하는 방식으로 구성됩니다. Zellinger와 Thomson(2025)의 방법을 활용하여 각 모델은 신뢰도 점수를 기반으로 쿼리에 대한 응답 여부를 결정하며, 이 과정에서 logistic regression을 사용하여 신뢰도를 보정합니다. 이러한 구조적 접근은 소형 LLM이 대형 LLM의 응답 여부를 예측할 수 있게 해 줍니다.

- **Performance Highlights**: 이 연구에서는 총 6개의 벤치마크에서 early abstention이 전체 테스트 손실을 평균 2.2% 감소시킴을 발견했습니다. 이는 소형 LLM이 높은 abstention rate(+4.1%)을 활용하여 큰 비용 절감(-13.0%)과 오류율 감소(-5.0%)를 가져오는 것을 보여줍니다. 이러한 결과는 모델 간의 오류 패턴 간 상관관계를 활용하여 성능 개선이 가능함을 입증합니다.



### Game Theory Meets Large Language Models: A Systematic Survey (https://arxiv.org/abs/2502.09053)
Comments:
          10 pages

- **What's New**: 본 논문은 게임 이론(Game Theory)과 대규모 언어 모델(LLMs) 사이의 상호 관계를 탐구하는 포괄적인 조사 결과를 제시합니다. 기존 연구는 게임 이론이 LLM을 평가하고 향상시키는 데 어떻게 활용될 수 있는지를 조사하는 것에 집중해 있었지만, 본 논문에서는 LLM이 게임 이론에 기여하는 측면도 강조합니다. 특히, LLM이 고전적 게임 모델을 재구성하며 균형 분석에 미치는 영향을 다루고 있습니다.

- **Technical Details**: 게임 이론은 합리적인 의사결정자 간의 전략적 상호작용을 분석하기 위한 수학적 프레임워크를 제공하며, 최근 LLM의 발전과 함께 게임 이론과 LLM의 교차점에 대한 연구가 증가하고 있습니다. 특히, 표준화된 게임 기반 평가 및 알고리즘 혁신을 통해 LLM의 성능을 향상시키고, LLM이 사회에 미치는 영향을 게임 모델로 특성화하는 방향의 연구가 포함됩니다. 이는 Nash 균형(Nash Equilibrium) 및 Shapley 가치(Shapley Value)와 같은 게임 이론의 원칙을 활용하여 이루어집니다.

- **Performance Highlights**: LLMs는 매트릭스 게임(Matrix Games)과 같은 성공적인 시나리오에서 뛰어난 전략적 사고 능력을 보이며, 다양한 방법으로 평가되고 있습니다. 연구자들은 LLM의 의사결정 능력을 향상시키기 위한 전략으로 재귀적 추론 프레임워크와 보조 모듈 통합을 제안하고 있습니다. 마지막으로, LLM의 사회적 영향 예측을 위한 새로운 이론적 프레임워크가 개발되고 있으며, 이는 AI와 인간의 상호작용에 대한 이해를 심화시킵니다.



### AoI-Sensitive Data Forwarding with Distributed Beamforming in UAV-Assisted Io (https://arxiv.org/abs/2502.09038)
Comments:
          6 pages, 4 figures, ICC2025

- **What's New**: 본 논문은 IoT에서 정보의 노후화(AoI, Age of Information)를 향상시키기 위해 분산 빔포밍(distributed beamforming)에 기반한 UAV(드론) 지원 데이터 전송 시스템을 제안합니다. UAV는 센서 노드(SNs)와 원거리 기지국(BS) 간의 데이터를 수집하고 중계하지만, 비행 지연이 발생하여 AoI를 증가시키고 네트워크 성능을 저하시킬 수 있습니다. 이를 해결하기 위해 비행 빈도를 줄이고 지속적인 데이터 중계를 보장하는 방법으로 드론 경로와 통신 일정을 공동 최적화하는 최적화 문제를 설정합니다.

- **Technical Details**: 시스템 모델은 UAV 지원 AoI 민감 데이터 전송을 위한 프레임워크를 제시하며, 각 SN은 데이터를 생성하고 UAV는 원거리 BS에 데이터 전달을 담당합니다. 직접적인 전송이 불가능한 환경을 고려하여, 분산 빔포밍을 통해 UAV의 비행 유도 지연을 줄이고 AoI를 감소시킵니다. 여기서 UAV는 통신 범위를 늘리기 위해 이동 없이 효과적인 비행을 할 수 있도록 설계됩니다.

- **Performance Highlights**: 제안된 SAC-TLA(Soft Actor-Critic with Temporal Sequence, Layer Normalization Gated Recurrent Unit, Attention) 알고리즘은 에너지 소비를 최소화하고 SN의 AoI를 줄이는 데 효과적입니다. 시뮬레이션 결과는 SAC-TLA 접근 방식이 다른 기존 알고리즘보다 우수한 성능을 보임을 입증합니다. AUAV을 통해 효율적인 자원 활용과 함께 신뢰할 수 있는 데이터 전달을 실현하여 IoT 시스템의 성능 향상에 기여합니다.



### Mechanistic Unveiling of Transformer Circuits: Self-Influence as a Key to Model Reasoning (https://arxiv.org/abs/2502.09022)
Comments:
          Accepted by NAACL2025

- **What's New**: 이번 논문에서는 언어 모델의 내부 추론 메커니즘을 명확히 하기 위한 새로운 기계적 해석 프레임워크인 SICAF를 제안합니다. 이 프레임워크는 다단계 추론(task)에서 언어 모델이 사용하는 추론 전략을 분석하고 추적하는 것을 목표로 합니다. SICAF를 통해 GPT-2 모델에서 간접 목적어 식별 작업을 수행하여 모델의 내부 논리를 새롭게 이해할 수 있는 통찰을 제공합니다.

- **Technical Details**: SICAF는 세 단계로 분석합니다: (1) 기존의 자동 회로 탐지 방법을 통해 모델 내 회로를 식별하고, (2) 샘플 별로 회로의 다양한 레이어에서 각 토큰의 self-influence를 계산하며, (3) self-influence 점수의 변화를 분석하여 언어 모델이 사용하는 추론 과정을 유추합니다. 이 과정에서 적은 수의 엣지(1-2%)를 가진 소형 회로가 발견되었고, 최초와 최종 레이어에서 주요 정보를 발견했습니다.

- **Performance Highlights**: 이 연구는 SICAF가 적용된 GPT-2 모델을 사용하여, 모델의 중요한 매개변수가 주로 첫 번째 레이어와 마지막 몇 개의 레이어에 집중되어 있다는 것을 보여주었습니다. 또한, 여러 Circuit Analysis 방법을 수용하여 서로 다른 회로에서 내재된 다양한 사고 과정을 발견하고 분석하는 데 기여하였습니다. 이러한 결과는 향후 언어 기반 추론 시스템의 신뢰성을 높이는 데 기여할 것입니다.



### On the Promise for Assurance of Differentiable Neurosymbolic Reasoning Paradigms (https://arxiv.org/abs/2502.08932)
- **What's New**: 이 논문은 신경망(neural network)과 고전 논리 프로그래밍을 결합한 신경 상징 AI(neurosymbolic AI) 시스템의 통합된 보장성(assurance)을 평가합니다. 특히, Scallop이라는 최첨단의 신경 상징 라이브러리를 사용하여 이미지 및 오디오 도메인에서 분류(classification)와 추론(reasoning) 작업을 수행합니다. 연구 결과, 완전한 신경망만으로 학습하기 어려운 복잡한 수학적 연산이 정의된 경우, 신경 상징 모델이 더 높은 보장성을 제공한다고 나타났습니다.

- **Technical Details**: 이 연구는 신경 상징 프로그래밍이 보장성 설계(assurance-by-design)를 제공할 수 있는지를 조사합니다. 신경망을 기반으로 하여 Scallop의 분별 가능한(differentiable) 추론 엔진을 통해 신경 상징 모델을 만들고, 이를 완전한 신경망과 비교하여 여러 작업과 평가를 진행했습니다. 연구 과정에서 적대적 강건성(adversarial robustness), 신뢰도 조정(calibration), 사용자 성능 차이(user performance parity) 등의 요소를 포함하여 보장성을 평가했습니다.

- **Performance Highlights**: 연구 결과는 다음과 같은 중요한 발견을 포함합니다. 첫째, 분별 가능한 신경 상징 추론은 알고리즘에 정의된 논리 연산이 존재하는 경우에 높은 보장성을 제공합니다. 둘째, 신경 상징 모델이 해석 가능한 단축을 취하는 경우, 성능은 유지되더라도 적대적 공격에 더 취약해지는 경향이 있습니다. 마지막으로, 데이터 불균형(class imbalance) 문제에서만 데이터 효율성이 보장되고, 일반적으로 적은 데이터로 더 높은 보장성을 보장하지는 않습니다.



### Self-Consistency of the Internal Reward Models Improves Self-Rewarding Language Models (https://arxiv.org/abs/2502.08922)
- **What's New**: 본 논문은 Large Language Models (LLMs)의 내부 보상 모델 간의 일관성을 개선하기 위한 Self-Consistent Internal Rewards (SCIR)라는 새로운 프레임워크를 제안합니다. 기존의 Self-Rewarding Language Models (SRLM) 접근법이 다양한 내부 보상 모델에서 비일관성을 드러내는 문제를 해결하고자 합니다. SCIR는 훈련 과정에서 일관성을 강화하고 신뢰할 수 있는 선호 데이터(Preference Data)를 생성하여 LLM의 정렬 성능을 크게 향상시킵니다.

- **Technical Details**: SCIR 프레임워크는 두 개의 내부 보상 모델을 사용합니다: 첫 번째는 LLM-as-a-Judge를 통해 선호 판단을 생성하는 생성적 보상 모델이고, 두 번째는 DPO에서 파생된 암묵적 보상 모델입니다. 각 훈련 단계에서 여러 내부 보상 모델의 예측을 수집하고, 이 예측의 일관성과 신뢰성을 확보하기 위해 비일관성 패널티 메커니즘을 실시합니다. 이를 통해 자가 생성된 선호 데이터의 품질을 높이는 프로세스를 실행합니다.

- **Performance Highlights**: Mistral-7B 모델을 실험에 사용하여 SCIR의 효과를 검증한 결과, AlpacaEval 2.0에서 길이 제어(win rate) 기준으로 14% 향상된 성능을 나타내었습니다. 또한, SCIR 적용에 따른 LLM의 내부 보상 모델 간의 일관성이 높아지고, 이로 인한 보상 모델링 능력의 개선이 확인되었습니다. SCIR은 기존 방법보다 뛰어난 성능을 보이며, LLM의 정렬 성능 및 보상 모델링 능력을 동시에 향상시키는 데 성공하였습니다.



### Reinforced Large Language Model is a formal theorem prover (https://arxiv.org/abs/2502.08908)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)을 정리 및 증명 작업에 더욱 효과적으로 활용하기 위한 강화 학습 프레임워크를 제안합니다. 사전 훈련된 LLM을 최적화하여 다음 전술을 반복적으로 실행하고 예상된 것과 비교함으로써 효율성을 높이고자 하였습니다. 실험 결과, 직접적으로 미세 조정(fine-tuned)된 LLM에 비해 더 높은 정확도를 달성한 것으로 나타났습니다.

- **Technical Details**: 프레임워크는 데이터 준비(data preparation), 모델 훈련(model training), 온라인 추론(online inference)의 세 부분으로 구성됩니다. 데이터 준비 과정에서 Lean-workbook의 데이터를 활용하고, GPT4o를 통해 각 샘플에 대한 사고 과정을 포함시키는 방식으로 데이터 세트를 보강합니다. 훈련은 적응(adaption) 단계와 강화 학습(reinforcement learning) 단계로 나뉘며, 사전 훈련된 LLM은 두 단계를 통해 최적화됩니다.

- **Performance Highlights**: 실험은 miniF2F에서 30개의 샘플을 사용하여 진행되었습니다. 결과는 적응 훈련이 거의 수렴에 도달하였고, 강화 학습 단계에서 형식적 보상이 빠르게 증가한 것으로 나타났습니다. 강화된 모델은 직접적인 감독 하에 미세 조정된 기본 모델보다 우수한 성능을 보였습니다.



### MIH-TCCT: Mitigating Inconsistent Hallucinations in LLMs via Event-Driven Text-Code Cyclic Training (https://arxiv.org/abs/2502.08904)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 비일관적 환각(hallucinations) 문제를 해결하기 위한 새로운 접근 방식을 제안합니다. MIH-TCCT(행사 기반 텍스트-코드 순환 훈련을 통한 비일관적 환각 완화) 프레임워크는 이벤트 기반 텍스트와 해당 코드 생성을 cyclic하게 수행하여 자연어의 논리적 일관성을 높입니다. 이 방법은 LLM이 다양한 자연어 작업에서의 환각을 효과적으로 줄이는 동시에 전반적인 성능을 유지하는 데 기여합니다.

- **Technical Details**: 연구진은 LLM의 논리적 일관성을 향상시키기 위해 코드 데이터와 이벤트 기반 텍스트 간의 구조적 상관관계를 이용합니다. 이 방법은 두 언어의 스타일 특성을 지속적으로 일치시킴으로써 자연어와 코드의 논리를 전이하는 데 초점을 맞춥니다. MIH-TCCT 프레임워크는 3개의 주요 LLM과 두 가지 유형의 자연어 작업에서 실험을 통해 비일관적인 환각을 현저히 줄였습니다.

- **Performance Highlights**: 이 연구는 비일관적 환각 문제를 해결하는 데 있어 코드 중심 LLM지식의 중요성을 강조합니다. LLMs는 이 프레임워크를 통해 다운스트림 작업에 적응하지 않고도 논리적 일관성을 유지할 수 있는 역량을 입증했습니다. 이러한 접근 방식은 향후 실용적인 응용 프로그램에서 다양한 자연어 처리(task)에 대한 일반화 가능성을 보여줍니다.



### Data Sensor Fusion In Digital Twin Technology For Enhanced Capabilities In A Home Environmen (https://arxiv.org/abs/2502.08874)
- **What's New**: 이번 논문은 디지털 트윈 기술에 데이터 센서 융합(data sensor fusion)을 통합하여 홈 환경 기능을 강화하는 방법을 조사합니다. 특히 코로나19 팬데믹과 그 경제적 효과에 대응하는 방식에 중점을 두고 있습니다. 디지털 전환(digital transformation)의 중요성을 강조하며, 이는 4차 산업혁명에서의 혼란을 완화하는 데 필수적입니다.

- **Technical Details**: Wit Motion 센서를 사용하여 걷기, 일하기, 앉기, 눕기와 같은 활동에 대한 데이터가 수집됩니다. 이 과정에서 accelerometers, gyroscopes, magnetometers와 같은 센서가 측정되며, 연구는 Cyber-physical systems, IoT, AI, 로봇 공학을 통합하여 디지털 트윈의 역량을 강화합니다. 또한, feature-level fusion, decision-level fusion, Kalman filter fusion과 같은 센서 융합 방법 및 SVM, GBoost, Random Forest와 같은 머신러닝 모델을 비교하여 모델의 효율성을 평가합니다.

- **Performance Highlights**: 센서 융합은 각 개별 센서의 약점을 보완하며, 특히 magnetometers의 신뢰성을 높이는데 기여합니다. 이상적인 조건에서의 높은 정확도에도 불구하고, 여러 센서로부터의 데이터 통합은 실제 환경에서 더 일관되고 신뢰할 수 있는 결과를 보장합니다. 이는 실제 사용 가능한 강력한 시스템을 확립하는 데 도움이 됩니다.



### Off-Switching Not Guaranteed (https://arxiv.org/abs/2502.08864)
Comments:
          Forthcoming in Philosophical Studies

- **What's New**: 이 연구에서는 Off-Switch Game이라는 새로운 모델을 제안하여 인공지능(AI) 에이전트가 인간의 선호를 학습하는 방식에 대한 새로운 관점을 소개합니다. 기존의 모델들은 AI가 항상 인간에게 복종하도록 보장하기 위해 특정한 가정에 의존했으나, 저자는 이러한 가정의 신뢰성을 의문시합니다. 특히, AI가 인간의 선호를 불확실하게 학습하도록 설계하는 것이 더 효과적일 수 있다는 점을 강조합니다.

- **Technical Details**: Off-Switch Game은 Cooperative Inverse Reinforcement Learning (CIRL) 프레임워크를 바탕으로 하며, AI 에이전트 R은 인간 H의 행동에 따라 여러 선택을 할 수 있습니다. R은 특정 행동을 제안하거나, 아무것도 하지 않거나, 인간에게 복종할 수 있는 선택지를 가지고 있으며, 이런 결정은 인간에 의해 승인되거나 거부됩니다. 저자는 R이 얼마나 효용이 있는지를 알지 못하는 점을 통해 인간의 선호를 학습할 수 있는 유인을 제공한다고 주장합니다.

- **Performance Highlights**: R의 결정 문제에서, R이 인간의 선호에 대해 불확실성을 가질 때, R은 인간에게 복종하는 것이 최적의 선택이라는 것을 보여줍니다. 예를 들어, 로봇이 인간에게 호텔 예약을 제안할 때, 인간이 이를 승인할 가능성에 따라 기대 효용이 달라지며, 불확실성이 클수록 인간의 결정을 잘 반영하면서 더 나은 결과를 얻을 수 있음을 설명합니다. 이 연구는 AI가 인간의 선호를 학습하고 존중하도록 프로그래밍할 수 있는 가능성을 보여줍니다.



### EnigmaEval: A Benchmark of Long Multimodal Reasoning Challenges (https://arxiv.org/abs/2502.08859)
- **What's New**: EnigmaEval은 언어 모델의 사고 능력을 평가하기 위한 새로운 벤치마크로, 기존의 추리 기준을 넘어 다중 모드 문제를 해결하는 데 초점을 맞추고 있습니다. 이 벤치마크는 퍼즐 대회에서 유래된 문제들로 구성되어 있으며, 언어 모델들이 암묵적인 지식 종합과 다단계 추론 능력을 수행하는 능력을 평가합니다. 특히 이 문제들은 명확한 지시 없이 다양한 접근 방식을 탐색해야 하므로 최첨단 LLM에게는 도전적인 과제가 됩니다.

- **Technical Details**: EnigmaEval은 1184개의 퍼즐로 구성되어 있으며, 원본 PDF의 PNG와 구조화된 텍스트-이미지 표현 두 가지 포맷으로 제공됩니다. 퍼즐은 다양한 형식을 포함하고 있으며, 단순한 단어 또는 짧은 구절을 해결하는 것을 목표로 합니다. 실험 과정에서 선진 LLM의 성능을 평가하기 위해 모델의 결과물을 실제 정답과 비교하며, 이는 문자열 매칭을 통해 이루어집니다.

- **Performance Highlights**: 최첨단 LLM들은 EnigmaEval 퍼즐에서 단 7.0%의 낮은 정확성을 기록했으며, 어려운 문제에서는 0%의 정확성을 보였습니다. 이는 현재의 다중 모드 LLM들이 복잡한 문제 해결 작업에서 전문가 수준의 사고 능력과 큰 격차가 있음을 나타냅니다. LLM들이 이러한 도전적이고 복잡한 문제를 해결하는 데 필요한 전략적 사고 및 구조화된 문제 해결 접근 방법에 익숙하지 않다는 점에서 주목할 만합니다.



### Estimating Probabilities of Causation with Machine Learning Models (https://arxiv.org/abs/2502.08858)
Comments:
          8 pages + 2 pages reference + 3 pages supplementary material, 5 figures, submitted to UAI 2025

- **What's New**: 이 논문은 인구 데이터가 부족한 하위 집단(subpopulation)에서 인과 확률(probabilities of causation)을 예측하는 문제를 다룬다. 저자들은 기존의 정의와 한계를 넘어, 데이터가 충분한 집단으로부터 인사이트를 얻어 하위 집단에 적용하는 머신 러닝 모델을 제안하였다. 이는 특히 인구 수준의 데이터가 부족할 때 실질적인 방법론을 제공한다.

- **Technical Details**: 본 연구에서는 인과 확률을 예측하기 위해 세 가지 기본 인과 확률(PNS, PS, PN)의 경계를 정의하였다. 이러한 확률은 하위 집단의 특성에 의해 결정된다고 가정하며, 이를 머신 러닝 모델을 통해 추정하는 방법을 제안한다. 평가 결과, 적절한 머신 러닝 모델과 활성화 함수(activation function)를 선택하면, 다양한 모델이 PNS를 효과적으로 예측할 수 있음이 확인되었다.

- **Performance Highlights**: 모델 평가에서는 multilayer perceptron (MLP) 모델이 Mish 활성화 함수를 사용하여 32,768개의 하위 집단에 대해 평균 절대 오차(MAE)가 약 0.02로 나타났다. 이는 약 2,000개의 하위 집단 데이터를 사용하여 PNS를 예측한 결과로, 제안된 모델의 효과성을 입증하는 중요한 성과이다.



### Can a Single Model Master Both Multi-turn Conversations and Tool Use? CALM: A Unified Conversational Agentic Language Mod (https://arxiv.org/abs/2502.08820)
- **What's New**: 본 논문은 기존의 대화 모델(TOD)과 언어 에이전트(LA)를 결합한 통합 접근법인 CALM (Conversational Agentic Language Model)을 제안합니다. CALM은 다양한 API를 호출하면서도 멀티턴 대화 관리를 효과적으로 수행할 수 있도록 설계되었습니다. 저자들은 CALM-IT라는 멀티태스크 데이터셋을 통해 CALM 8B, CALM 70B 및 CALM 405B 모델을 각각 훈련시켜, 기존의 도메인 특화 모델들과 비교해 전반적으로 우수한 성능을 보였습니다.

- **Technical Details**: CALM 모델은 MultiWOZ 2.4, BFCL V3, 및 API-Bank라는 세 가지 벤치마크에서 평가되었습니다. 연구팀은 기존 LAs가 API 호출에서는 우수하나 멀티턴 상호작용에서는 성과가 낮음을 발견했습니다. 이와 반대로, 기존의 TOD 시스템은 멀티턴 대화에서는 잘 작동하지만 기능 호출에는 한계가 있음을 확인했습니다. 이로 인해 CALM 모델들은 두 영역 모두에서 뛰어난 성능을 발휘하여 기존의 성과 격차를 줄였습니다.

- **Performance Highlights**: CALM 70B 및 CALM 405B 모델은 GPT-4o와 다른 도메인 특화 모델들보다 멀티턴 대화 및 기능 호출 작업에서 더 높은 성능을 나타냈습니다. 이 연구는 오픈 소스 커뮤니티의 향후 연구를 촉진하기 위해 코드, 모델 가중치 및 데이터셋 등 모든 자료를 공개하였습니다. 기존의 보안 모델들과 비교해 CALM 모델의 노력은 멀티턴 대화 스킬과 신뢰할 수 있는 기능 호출 능력을 통합함으로써 주목받고 있습니다.



### Contextual bandits with entropy-based human feedback (https://arxiv.org/abs/2502.08759)
- **What's New**: 최근 선호 기반 인간 피드백 메커니즘이 ChatGPT와 같은 대화형 AI 시스템을 포함한 다양한 응용 분야에서 모델 성능 향상에 필수적이게 되었습니다. 하지만 기존 접근법들은 모델 불확실성 및 피드백 품질의 변동성을 간과하는 경향이 있습니다. 본 연구에서는 기존 한계점을 해결하기 위해 엔트로피 기반의 인간 피드백 프레임워크를 도입하여 전문 피드백을 필요한 상황에서만 요청하는 방식을 제안합니다.

- **Technical Details**: 본 연구는 행동 추천(Action Recommendation)과 보상 수정(Reward Manipulation)이라는 두 가지 보조 피드백 통합 전략을 소개합니다. 엔트로피 기반 요청 기준을 통해 불확실성이 높은 상황에서만 피드백을 요청하여 효율성을 높입니다. 이 접근법은 탐색과 활용의 균형을 효과적으로 맞추고 불필요한 쿼리를 줄이며, 모델의 정책을 보다 효율적으로 개선할 수 있도록 합니다.

- **Performance Highlights**: 종합 실험을 통해 우리의 접근법이 요구되는 최소한의 인간 피드백으로도 상당한 성능 향상을 달성함을 보여줍니다. 특히, 피드백 품질이 최적이 아닌 상황에서도 성능 강건성을 유지했습니다. 이러한 연구 결과는 피드백 요청 방법에 대한 새로운 전략을 제시하며, 머신러닝 시스템에 인간의 지침을 통합하는 효과성을 부각시킵니다.



### From PowerPoint UI Sketches to Web-Based Applications: Pattern-Driven Code Generation for GIS Dashboard Development Using Knowledge-Augmented LLMs, Context-Aware Visual Prompting, and the React Framework (https://arxiv.org/abs/2502.08756)
- **What's New**: 본 논문은 사이버GIS 대시보드와 같은 웹 기반 GIS 애플리케이션 개발을 위한 지식 강화 코드 생성 프레임워크를 제안합니다. 이 프레임워크는 소프트웨어 공학의 우수 사례와 도메인 전문 지식을 통합하여 Generative Pre-trained Transformers (GPT)를 강화합니다. 사용자 정의 UI 와이어프레임에서 GIS 기반 웹 애플리케이션을 자동으로 생성할 수 있는 기능을 제공합니다.

- **Technical Details**: 제안하는 프레임워크는 Python으로 구현된 Context-Aware Visual Prompting 방법을 포함하여, PowerPoint나 Adobe Illustrator와 같은 도구에서 스케치한 UI 와이어프레임에서 레이아웃과 인터페이스 기능을 추출합니다. 이는 LLMs이 구조적 추론, 소프트웨어 공학 원칙, 도메인 지식을 통합하여 프론트 엔드 코드를 생성할 수 있도록 돕습니다. 디자인 패턴(Model-View-ViewModel, MVVM)과 React와 같은 프레임워크를 사용해 업계 표준에 맞는 코드를 생성합니다.

- **Performance Highlights**: 사례 연구를 통해 이 프레임워크가 환경 및 에너지 데이터 시각화와 상호작용을 위한 다중 대시보드를 호스팅하는 모듈형 웹 플랫폼을 자율적으로 생성할 수 있음을 보여줍니다. 또한, 프레임워크는 유지 보수성과 확장성을 보장하며, UI/UX 디자인, 코딩 및 유지보수에서 수작업을 크게 줄여줍니다. 이 연구는 스마트 시티 소프트웨어 개발을 위한 자동화된 효율적인 방법을 선도하고 있습니다.



### High-Throughput SAT Sampling (https://arxiv.org/abs/2502.08673)
Comments:
          7 pages

- **What's New**: 이번 연구에서는 GPU 가속이 가능한 새로운 부울 만족도(SAT) 샘플링 기술을 제시합니다. 기존의 샘플링 알고리즘과는 달리, 우리의 방법은 SAT 문제의 CNF(Conjunctive Normal Form) 표현을 다중 수준의 다중 출력 부울 함수로 단순화하여 논리적 제약 조건을 변환합니다. 이러한 새로운 방법론은 복잡한 문제를 보다 효율적으로 해결할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 우리는 판별 가능한(differentiable) 기술을 사용하여 각 텐서(tensor) 요소에서 독립적인 비트 단위(bit-wise) 작업을 수행하는 것을 가능하게 하였습니다. 이는 학습 과정을 병렬적으로 실행할 수 있게 하여 계산의 효율을 극대화합니다. 또한, 문제를 감독받는(supervised) 다중 출력 회귀(multi-output regression) 작업으로 다시 해석함으로써 샘플링 과정을 최적화하였습니다.

- **Performance Highlights**: 우리의 샘플링 방법은 최신의 휴리스틱 샘플러에 비해 $33.6	imes$에서 $523.6	imes$까지의 의미 있는 런타임 개선(상태 개선)을 달성했습니다. 연구는 이전 연구에서 사용된 공공 벤치마크 세트의 60개 인스턴스를 통해 샘플링 방법의 우수한 성능을 확인하였습니다.



### Personalizing Education through an Adaptive LMS with Integrated LLMs (https://arxiv.org/abs/2502.08655)
- **What's New**: 대규모 언어 모델(LLMs)의 채택이 교육 분야에서도 혁신적인 변화를 이끌고 있습니다. 본 논문은 학습 관리 시스템(LMS) 내에서 LLM의 통합을 통해, 사용자 개개인의 맞춤화된 적응형 학습 관리 시스템(ALMS)을 개발하는 과정을 다룹니다. 전통적인 LMS는 교육 자료를 배포하는 데에는 유용하지만, 다양한 학생 인구의 세심한 요구를 충족시키기에는 부족하여 AI의 유연성을 활용한 맞춤형 학습 환경의 필요성이 강조됩니다.

- **Technical Details**: 이 시스템은 일반 목적 및 도메인 특정 LLM의 통합을 통해, 사실 오류와 구식 정보와 같은 문제를 최소화하는 것을 목표로 하고 있습니다. 개발 과정은 세 단계로 나누어 진행되었으며, 각 단계에서 OCR 및 데이터 관리와 같은 작업을 위한 커맨드라인 스크립트를 설계하고 Django, React를 기반으로 웹 백엔드 및 프론트엔드를 구축했습니다. 최종적으로, 여러 다양한 LLM의 성능을 평가하고 자원 활용도를 분석하여 학습 관리 시스템에서의 효용성을 평가했습니다.

- **Performance Highlights**: 실험 결과, 제안된 ALMS는 사용자 개인 맞춤형 학습 경험을 제공하는 데 효과적임을 보였고, 이는 학습 참여도와 성과를 향상시키는 데 기여했습니다. 특히, RAG(Retrieval-Augmented Generation)와 벡터 임베딩을 활용하여 LLM의 정확도를 높이고, 허위 정보 발생을 줄이는 데 성공했습니다. 이 연구는 교육 현장에서 AI의 활용 가능성을 입증하며, 향후 연구 개발 방향에 대한 통찰을 제공합니다.



### Theoretical Benefit and Limitation of Diffusion Language Mod (https://arxiv.org/abs/2502.09622)
Comments:
          32 pages, 3 figures

- **What's New**: 이 논문에서는 텍스트 생성을 위한 최근의 방법론인 Diffusion language models에 대해 다룹니다. 특히, Masked Diffusion Model (MDM)을 분석하여, 해당 모델의 효과는 평가 지표에 크게 의존한다는 것을 밝혔습니다. Perplexity를 사용할 경우 MDM은 최적의 성능을 보일 수 있지만, sequence error rate를 사용할 경우 MDM의 효율성이 감소함을 발견했습니다.

- **Technical Details**: MDM의 진행은 일반적인 autoregressive 모델과 비교하여 효율성이 높지만, 이는 특정 조건 하에서만 유효합니다. Perplexity를 메트릭으로 사용할 때는 효율성을 인정받을 수 있지만, reasoning chain과 같은 '정확성'을 요구하는 경우에는 sequence length에 따라 샘플링 스텝이 선형 배로 증가해야 합니다. 이러한 분석 결과는 실험적 연구에 의해 입증되었습니다.

- **Performance Highlights**: 본 연구는 MDM에 대한 이론적 토대를 마련하여 향후 연구의 방향성을 제시합니다. MDM은 특정 메트릭에서는 효과적인 성능을 나타내나, 보다 복잡한 상황에서는 효율성의 저하를 불러올 수 있음을 보여줍니다. 이는 여러 개발자와 연구자들이 MDM의 장단점을 명확히 이해하고 적용할 수 있도록 도움을 줄 것입니다.



### MME-CoT: Benchmarking Chain-of-Thought in Large Multimodal Models for Reasoning Quality, Robustness, and Efficiency (https://arxiv.org/abs/2502.09621)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문에서는 Chain-of-Thought (CoT) 이론이 Large Multimodal Models (LMMs)의 추론 성능에 미치는 영향을 체계적으로 평가하기 위한 MME-CoT라는 새로운 벤치마크를 소개합니다. 이 벤치마크는 수학, 과학, OCR, 논리, 시공간 및 일반 장면의 여섯 가지 도메인에 걸쳐 CoT 추론 성능을 평가합니다. 이러한 과정을 통해 LMMs의 CoT 성능에 대한 첫 번째 포괄적인 연구를 제시합니다.

- **Technical Details**: MME-CoT는 추론 품질, 견고성(robustness) 및 효율성(efficiency)을 세밀하게 평가할 수 있는 세 가지 새로운 메트릭(metrics)을 포함하는 평가 도구를 제안합니다. 고품질 데이터셋을 활용하고 독창적인 평가 전략을 통해 현재 최신 LMMs에 대한 심층 분석을 수행하며, 이 과정에서 몇 가지 중요한 통찰을 발견하였습니다.

- **Performance Highlights**: 주요 발견 내용으로는 1) 반영(reflection) 메커니즘을 가진 모델들이 높은 CoT 품질을 보여주며, Kimi k1.5가 GPT-4o를 능가하여 최고 품질 결과를 도출하였다는 점입니다. 2) CoT 프롬프트(prompts)는 인지 중심의 작업에서 LMM 성능을 저하시킬 수 있으며, 이는 과도한 사고(overthinking)의 부정적인 행동을 시사합니다. 3) CoT 품질이 높음에도 불구하고 반영 기능이 있는 LMMs는 정상 응답 및 자기 수정 단계에서 상당한 비효율성을 보인다는 점입니다.



### Exploring the Potential of Encoder-free Architectures in 3D LMMs (https://arxiv.org/abs/2502.09620)
Comments:
          The code is released at this https URL

- **What's New**: 이번 논문은 엔코더가 없는 아키텍처를 활용하여 3D 이해를 효과적으로 발전시킬 수 있는 가능성을 탐구합니다. 특히, 기존의 엔코더 기반 LMM의 한계를 극복하기 위해 LLM(대형 언어 모델) 내부에서 3D 엔코더의 기능을 통합하는 새로운 전략들을 제안합니다. 이 연구는 3D LMM을 위한 엔코더 없는 아키텍처의 첫 번째 포괄적 조사로, 엔코더를 제거하고 LLM이 고수준의 3D 의미를 추출하도록 합니다.

- **Technical Details**: 논문에서는 LLM-embedded Semantic Encoding(LLM-임베디드 의미 인코딩)과 Hierarchical Geometry Aggregation(계층적 기하집합)이라는 두 가지 핵심 전략을 제안합니다. 첫 번째는 LLM 초기 층을 학습 가능하게 하여 3D 의미를 캡쳐하도록 돕는 방법입니다. 두 번째는 3D 토큰을 기하학적 분포에 따라 집계하여 LLM이 점진적으로 세부적인 3D 의미를 통합하게 합니다. 이 두 전략은 ENEL이라는 엔코더 없는 3D LMM을 통해 구현됩니다.

- **Performance Highlights**: ENEL은 ShapeLLM-13B와 비교하여 클래스 분류에서 55.0%, 캡션 생성에서 50.92%라는 성과를 달성하며, 기존 기술 수준에 가까운 성능을 보여줍니다. 이 연구는라는 것은 엔코더 기반 아키텍처에 비해 엔코더 없는 아키텍처가 3D 이해 분야에서 매우 유망하다는 것을 의미합니다. 최종적으로, ENEL의 출현은 3D 시나리오에 엔코더 없는 아키텍처를 적용하는 효율적인 경로를 제공할 것으로 기대됩니다.



### DexTrack: Towards Generalizable Neural Tracking Control for Dexterous Manipulation from Human References (https://arxiv.org/abs/2502.09614)
Comments:
          Accepted to ICLR 2025. Website: this https URL Code: this https URL Video: this https URL

- **What's New**: 이번 논문에서는 인간의 포든 동작을 기반으로 한 일반화 가능한 신경 추적 제어기를 개발하는 문제를 다룹니다. 이 제어기는 다양하고 복잡한 물체를 조작하기 위해 로봇 손을 제어하는 것을 목표로 하며, 기존 강화 학습 및 궤적 최적화 방법의 한계를 극복하기 위해 설계되었습니다. 우리는 대규모의 성공적인 로봇 추적 시연을 통해 신경 제어기를 학습시키는 접근법을 소개하며, 이를 통해 성능과 성공적인 시연의 수와 질을 모두 향상시킬 수 있습니다.

- **Technical Details**: 우리의 연구는 DexTrack이라는 새로운 신경 추적 제어기를 제안합니다. 이 제어기는 인간 손-객체 조작 경로를 로봇에 맞추어 최적화하고, 성공적인 로봇 추적 시연과 함께 교대로 훈련됩니다. 강화 학습과 모방 학습 기법을 통합하여 다양한 상황에서 제어기의 성능을 향상시키고, 동적 환경에서도 강력한 적응력을 유지합니다.

- **Performance Highlights**: DexTrack은 두 개의 데이터 세트에서 기존 방법과 비교하여 뛰어난 성능을 발휘하였으며, 이전 방법보다 10% 이상의 성공률을 기록하였습니다. 시뮬레이터인 Isaac Gym 및 실제 환경에서의 평가를 통해 광범위한 조작 추적 작업을 성공적으로 수행하고, 예기치 않은 상황에서도 회복 능력이 있음을 입증했습니다. 이 연구는 반복적인 시연 채굴을 통해 발전하는 일반화 가능성이 큰 신경 추적 제어기를 제시합니다.



### Score-of-Mixture Training: Training One-Step Generative Models Made Simp (https://arxiv.org/abs/2502.09609)
Comments:
          27 pages, 9 figures

- **What's New**: 이번 논문에서는 Score-of-Mixture Training (SMT)이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 $
α$-skew Jensen-Shannon divergence라는 종류의 발산(divergence)을 최소화하여 일단계 생성 모델을 학습합니다. SMT는 실제 샘플과 가짜 샘플 간의 혼합 분포(score of mixture distributions)를 여러 잡음 수준(noise levels)에서 추정하는 것을 핵심으로 합니다.

- **Technical Details**: SMT는 일관성 모델(consistency models)과 유사하게, 처음부터 학습하는 방법(SMT)과 사전 훈련된 확산 모델(diffusion model)을 활용한 증류(distillation) 방법(SMD)을 지원합니다. 이 방법은 구현이 간단하고 하이퍼파라미터 튜닝(hyperparameter tuning)이 최소한으로 필요하며, 안정적인 학습을 보장합니다.

- **Performance Highlights**: CIFAR-10과 ImageNet 64x64 데이터셋에서 실험한 결과 SMT/SMD가 기존 방법들과 경쟁력이 있으며, 심지어 이들을 초월할 수 있음을 보여주었습니다. 이러한 결과는 SMT/SMD의 잠재력을 뒷받침하고 있으며, 생성 모델 분야에서 중요한 기여를 할 것으로 예상됩니다.



### Human-LLM Coevolution: Evidence from Academic Writing (https://arxiv.org/abs/2502.09606)
- **What's New**: 본 논문은 아르Xiv 논문의 초록에서 통계 분석을 통해 ChatGPT가 과도하게 사용되었던 단어들의 사용 빈도가 감소하고 있음을 발견했습니다. "delve"와 같은 단어는 2024년 초 이후로 현저히 줄어들고 있으며, 반면에 "significant"와 같은 단어는 증가하는 경향을 보이고 있습니다. 이러한 변화는 학문적 글쓰기에서 대규모 언어 모델(LLMs)의 사용 방식이 변화하고 있음을 시사합니다.

- **Technical Details**: 본 연구에서는 2018년부터 2024년까지 제출된 아르Xiv 논문의 초록 데이터에서 단어 빈도 분석을 수행했습니다. 연구자는 특정 단어의 사용 빈도를 월별로 계산하고 10,000개의 초록 당으로 정규화했습니다. 특히 2024년 4월 이후 LLM 스타일의 단어들이 감소하기 시작했고, "significant"와 같은 단어들은 상대적으로 잘 사용되면서 여전히 증가 추세를 보이는 것으로 나타났습니다.

- **Performance Highlights**: 연구에서 LLMs가 학문적 글쓰기에서 미친 영향을 파악하는 것 외에도, 기계 생성 텍스트(MGT)의 탐지에는 여러 도전 과제가 제기되었습니다. 감지기의 성능은 사용되는 LLM 모델과 텍스트 유형에 따라 달라지며, 단순한 이진 분류 체계로는 현실 세계의 복잡한 상황을 포괄하기 어렵습니다. 이 논문은 LLMs에 의해 생성된 텍스트의 영향을 통계적으로 측정하는 것이 더욱 실용적인 선택임을 강조하고 있습니다.



### SelfCite: Self-Supervised Alignment for Context Attribution in Large Language Models (https://arxiv.org/abs/2502.09604)
Comments:
          Implementation available at this https URL

- **What's New**: SelfCite는 LLMs(대형 언어 모델)의 응답에서 고품질, 세밀한 인용(citations)을 생성하기 위한 새로운 자기 감독(self-supervised) 접근 방식을 소개합니다. 본 방법은 비싼 주석 작업 없이도 LLM이 제공하는 보상 신호를 활용하여 인용의 품질을 향상시키도록 설계되었습니다. SelfCite는 모델의 내부 피드백 메커니즘을 활용하여 인용의 필요성과 충분성을 평가하는 과정을 포함합니다.

- **Technical Details**: SelfCite의 핵심은 context ablation 기술을 사용하여 LLM이 생성하는 응답의 문맥에서 인용이 필요하거나 충분한지를 평가하는 것입니다. 구체적으로, 인용된 텍스트를 제거했을 때 응답의 확률이 낮아지면 해당 인용이 필요하다고 판단하고, 인용된 텍스트만 남겼을 때 높은 확률을 유지한다면 충분하다고 간주합니다. 이러한 자가 평가 메커니즘을 통해 SelfCite는 주석 과정 없이 보상 신호를 계산하게 됩니다.

- **Performance Highlights**: SelfCite는 LongBench-Cite 벤치마크에서 인용의 F1 점수를 최대 5.3점 높이며 LLM의 자동 인용 품질 향상 가능성을 보여줍니다. 또한, SimPO를 통해 선호 최적화(preference optimization)를 적용하여 개선된 인용 품질을 유지하면서 이전 최첨단 기법을 초과한 성능을 달성했습니다. 이 접근법은 LLM의 자기 보상을 통해 인용 품질을 더욱 개선하는 방향을 제시합니다.



### MorphNLI: A Stepwise Approach to Natural Language Inference Using Text Morphing (https://arxiv.org/abs/2502.09567)
Comments:
          16 pages, 11 figures, 8 tables. Accepted for NAACL 2025 Findings

- **What's New**: MorphNLI는 자연어 추론(NLI)을 위한 모듈식 접근법이자 단계별 방법을 제시합니다. 이 방법은 전제-가설 쌍을 {entailment, contradiction, neutral}로 분류할 때, 언어 모델을 사용하여 전제를 가설로 점진적으로 변형하는 데 필요한 수정사항을 생성합니다. 특히 이 방법은 현실적인 교차 도메인 환경에서 더 뛰어난 성능을 보이며, 최대 12.6%의 상대적 향상을 보여줍니다.

- **Technical Details**: MorphNLI는 세 가지 주요 단계로 구성됩니다: 첫째, 전제가 작은 수정 단위인 morphism을 통해 점진적으로 가설로 변환됩니다. 둘째, 각 변환 과정에서 NLI 엔진이 적용되어 NLI 라벨을 생성합니다. 셋째, 이러한 라벨은 최종적인 NLI 라벨로 집계됩니다. 이를 통해 오버피팅(overfitting) 가능성을 줄이며, 전체 NLI 라벨을 이해하기 쉬운 형태로 설명할 수 있습니다.

- **Performance Highlights**: MorphNLI는 다양한 시나리오에서 평가되었으며, 교차 도메인 환경에서 기존의 최첨단 NLI 모델보다 일관되게 우수한 성능을 보였습니다. 또한 MorphNLI의 설명 품질은 비슷한 크기의 모델에 비해 더 나은 결과를 보였고, LLM들은 논리 semantics를 잘 포착하지 못하는 경향이 있습니다.



### Mind the Gap! Choice Independence in Using Multilingual LLMs for Persuasive Co-Writing Tasks in Different Languages (https://arxiv.org/abs/2502.09532)
- **What's New**: 최근 생성 AI의 발전은 새로운 글쓰기 도우미의 확산을 촉진했습니다. 이러한 시스템은 다국어 대형 언어 모델(multilingual large language models, LLMs)에 의존하여 다양한 언어로 다양한 콘텐츠를 수정하거나 생성할 수 있는 기능을 제공합니다. 그러나 다양한 언어에서 다국어 LLM의 성능이 크게 차이난다는 사실이 증명되었습니다.

- **Technical Details**: 본 연구에서는 스페인어 LLM의 성능이 영어 LLM에 대한 사용자의 활용에 미치는 영향을 분석합니다. 특히, 사용자들이 사용한 LLM의 성능에 따라 도와주는 도구의 사용과 광고의 설득력에 미치는 영향을 정량화했습니다. 결과적으로, 사용자는 선택 독립성 원칙을 위반하여 이전 언어 경험이 다른 언어의 사용에 영향을 미친다는 것을 확인했습니다.

- **Performance Highlights**: 광고의 설득력은 생성된 광고의 출처에 대한 사람들의 신념에 의해 영향을 받을 수 있음을 발견했습니다. 특히 스페인어를 사용하는 여성이 AI가 생성한다고 믿는 광고를 읽을 경우, 기부 행동을 크게 축소하는 경향을 보였습니다. 최종적으로, 참가자들은 광고가 인간 또는 LLM에 의해 생성되었는지를 신뢰성 있게 구별하지 못하지만, 이러한 신념에 따라 기부 행동이 달라질 수 있음을 보여줍니다.



### Diffusion Models for Molecules: A Survey of Methods and Tasks (https://arxiv.org/abs/2502.09511)
- **What's New**: 이 논문은 분산 모델(difffusion model)에 기반한 분자 생성 방법에 대한 종합적인 조사(research survey)를 제공하며, 이 분야의 현재 상태와 발전 가능성을 체계적으로 정리합니다. 특히, 다양한 연구를 체계적으로 분류하여 과학자들이 이 복잡한 영역을 보다 쉽게 이해하고 탐색할 수 있도록 도와주고자 합니다. 이를 통해 분자 디자인 분야의 혁신을 촉진하려는 의도가 담겨 있습니다.

- **Technical Details**: 분산 모델은 두 가지 주요 과정, 즉 전방 확산(propagation) 과정과 역 생성(reverse generation) 과정으로 구성됩니다. 전방 과정에서는 실제 데이터에 점진적으로 노이즈를 추가하여 단순한 사전 분포(prior distribution)로 접근합니다. 역 과정에서는 이 노이즈로부터 데이터 분포를 점진적으로 복원하는 법을 학습하며, 이는 일반적으로 신경망(neural networks)을 사용하여 매개변수화됩니다.

- **Performance Highlights**: 분산 모델은 최근 다양한 도메인에서 고품질 데이터를 생성하는 데 뛰어난 성능을 보이며, 분자 생성 과제에서도 큰 잠재력을 보여주고 있습니다. 이러한 모델들은 복잡한 분자 구조와 특성을 효과적으로 모델링하며, 현대 과학 연구에서 필수적인 도구로 자리잡고 있습니다. 이로 인해 분자 설계 분야의 연구가 급증하고 있으며, 연구자들은 이 모델의 변형 및 적용 가능성에 대해 활발히 탐구하고 있습니다.



### AttentionSmithy: A Modular Framework for Rapid Transformer Development and Customization (https://arxiv.org/abs/2502.09503)
- **What's New**: 이 논문은 Transformer 아키텍처를 더 쉽게 커스터마이즈할 수 있도록 돕는 'AttentionSmithy'라는 모듈형 소프트웨어 패키지를 소개합니다. 사용자는 이 패키지를 통해 복잡한 코딩 없이 재사용 가능한 컴포넌트를 통해 다양한 transformer 변형을 빠르게 프로토타입하고 평가할 수 있습니다.

- **Technical Details**: AttentionSmithy는 attention 모듈, feed-forward 네트워크, normalization 레이어 및 positional encodings과 같은 주요 구성 요소를 분해하여 제공합니다. 이 프레임워크는 네 가지 positional encoding 전략을 지원하며, 자동 설계를 위한 neural architecture search와 통합됩니다.

- **Performance Highlights**: AttentionSmithy는 원래의 transformer를 복제하고 리소스 제약 하에서 번역 성능을 최적화하는 데 성공했습니다. 또한, 특정 유전자 모델링에서 95% 이상의 세포 유형 분류 정확도를 달성하여 다양한 분야에서 연구를 가속화할 가능성을 보여줍니다.



### Improve LLM-based Automatic Essay Scoring with Linguistic Features (https://arxiv.org/abs/2502.09497)
Comments:
          To be published in the workshop Innovation and Responsibility in AI-Supported Education (iRaise) at the 2025 Conference on Artificial Intelligence (AAAI)

- **What's New**: 이 논문에서는 Automatic Essay Scoring (AES) 시스템을 개선하기 위해 linguistic features를 LLM(large language model) 기반의 평가 시스템에 통합하는 혼합 접근 방식을 제안합니다. 기존의 supervised feature-based 접근법과 LLM 기반 방법의 장점을 결합하여 다양한 에세이 채점 시나리오에 효과적으로 적용할 수 있는 방법을 탐구합니다. 실험 결과, 이 혼합 방법이 여러 유형의 글쓰기 프롬프트에서 기존 모델보다 우수한 성능을 보여 주목받고 있습니다.

- **Technical Details**: 이 연구에서는 zero-shot prompting 방법론을 활용하여 LLM의 평가 능력을 향상시키는 데 중점을 두고 있습니다. 각 프롬프트는 persona pattern, essay prompt, analysis task와 같은 구성 요소로 이루어져 있으며, 에세이에 대한 추가 정보를 통해 linguistic features를 통합합니다. 이렇게 구성된 프롬프트를 사용하여 성능을 극대화하고, 각 에세이에 대한 세부 평가를 수행할 수 있습니다.

- **Performance Highlights**: 실험에서 linguistic features가 통합된 LLM 프롬프트는 인간 평가와 더 잘 정렬되며, out-of-distribution 데이터에서도 성능 향상을 보여주었습니다. 이는 LLM이 학생 에세이를 자동으로 평가하는 데 있어 여전히 개선의 여지가 있음을 시사합니다. 특히, open-source LLM이 연결된 평가 모델에 비해 낮은 성능을 보일 수 있는 이유는 LLM 내장된 prior의 조정 부족 때문이라는 가설이 제기되었습니다.



### Cracking the Code: Enhancing Development finance understanding with artificial intelligenc (https://arxiv.org/abs/2502.09495)
- **What's New**: 이 논문은 개발 프로젝트 분석의 중요성을 강조하며, 기부자 지원 전략, 수혜국의 우선순위 및 개발 재정 역량을 평가하는 데 도움을 줍니다. 연구는 OECD의 Creditor Reporting System (CRS) 데이터셋의 한계를 극복하고자 새로운 접근 방식을 도입하였습니다. 특히, 프로젝트의 목적을 조명하는 데 필요한 정보가 부족한 현 상황에서, 머신러닝(Machine Learning) 기법을 활용하여 프로젝트를 분류하고 라벨링하는 방법을 제시합니다.

- **Technical Details**: 본 연구는 자연어처리(Natural Language Processing, NLP) 기법과 BERTopic이라는 파이썬 주제 모델링(topic modeling) 기법을 결합하여 개발 프로젝트의 서술 설명을 기반으로 프로젝트를 클러스터링(cluster)하고 라벨링(label)합니다. 이를 통해 개발 재정의 기존이지만 숨겨진 주제(topic)를 드러내는 새로운 통찰력을 제공합니다. 이러한 방법론은 기부자 우선순위를 이해하고 공공 및 민간 프로젝트 서술을 분석하는 데 유용합니다.

- **Performance Highlights**: 연구 결과는 OECD CRS 데이터셋에서 제공하는 방대한 프로젝트 내러티브(narratives)를 효과적으로 활용하며, 기부자들의 재정 지원과 프로젝트 목적 사이의 관계를 명확히 합니다. 인공지능(AI) 기술을 통해 제공된 이 새로운 접근법은 개발 자금의 분포를 보다 잘 이해할 수 있는 기회를 제공하고, 관련 데이터 분석에 대한 새로운 시각을 제시합니다.



### Objective quantification of mood states using large language models (https://arxiv.org/abs/2502.09487)
Comments:
          main text - 9 pages, 5 figures;

- **What's New**: 이 연구는 감정적 상태와 행동 간의 관계를 규명하기 위해 대형 언어 모델(LLMs)을 활용하여 정신 상태를 정량화하는 새로운 방법론을 제시합니다. 연구진은 넷 상에서 모집된 422명의 참여자를 통해 LLM인 Mistral-7B-OpenOrca가 우울증 질문지에 대한 개방형 응답에 대한 응답을 분석했습니다. 이 접근 방식은 질병 기반 치료와 조정뿐만 아니라 정신 건강 문제를 이해하는데 기여할 수 있는 잠재력을 가지고 있습니다.

- **Technical Details**: 연구에서는 표준화된 자가 보고 정신의학 질문지를 사용하여 참여자들의 우울증 상태를 조사했습니다. 각 질문지는 그녀의 정신 상태를 나타내는 답변의 일관성을 유지하며, LLM의 숨겨진 상태를 통해 우울증 관련 특성들을 식별하여 예측력을 검증했습니다. 이를 위해 참가자들은 PHQ-9 질문지를 오픈형 질문으로 재구성된 다양한 질문에 대해 응답했습니다.

- **Performance Highlights**: LLM의 응답은 참가자들의 개방형 답변으로부터 도출된 다중 선택 질문과 강한 상관관계(r: 0.52-0.84)를 보였으며, 이는 LLM이 우울한 정서를 정량화 할 수 있는 가능성을 시사합니다. 연구에서는 LLM의 특정 서브스페이스가 참가자들의 우울증 지수와 정서적 고통 요인 점수 및 자살 위험성의 예측을 제공할 수 있음을 발견했습니다. 이러한 결과는 LLM이 정신 상태를 정량화하는 데 있어 유용한 도구가 될 수 있음을 보여줍니다.



### PenTest++: Elevating Ethical Hacking with AI and Automation (https://arxiv.org/abs/2502.09484)
Comments:
          27 pages, 6 figures

- **What's New**: PenTest++는 AI에 의해 강화된 자동화 시스템으로, 전통적인 윤리적 해킹의 비효율성과 확장성 문제를 해결하기 위해 설계되었습니다. 이 시스템은 리콘나이상(reconnaissance), 스캐닝(scanning), 나열(enumeration), 착취(exploitation), 문서화(documentation)와 같은 주요 작업을 간소화하여 윤리적 해킹의 워크플로우를 최적화합니다. 또한, PenTest++는 사용자 주도의 제어와 적응성을 보장하여 AI의 도움을 받으면서도 감시와 전문성의 균형을 유지합니다.

- **Technical Details**: PenTest++는 Python 3를 사용하여 개발되었으며, OpenAI의 ChatGPT API와 통합됩니다. 이 도구는 다양한 환경에 적응 가능하며, AI 기반 분석과 같은 고급 기능의 통합을 통해 윤리적 해킹 중 발생하는 복잡한 결과물 처리를 효율적으로 지원합니다. 실험 환경은 VirtualBox를 사용한 가상화로 구축되었으며, Kali Linux 및 Debian Linux 등의 가상 머신을 통해 침투 테스트를 수행합니다.

- **Performance Highlights**: 이 연구는 PenTest++의 유용성을 평가하기 위해 Linux 기반 가상 환경에서 실험을 진행하였으며, 자동화와 Generative AI의 결합이 제공하는 여러 이점과 윤리적 고려사항을 살펴보았습니다. PenTest++는 보안 평가에 필요한 시간과 비용을 크게 줄일 수 있어, 다양한 규모의 조직에서도 실현 가능성을 높입니다. 이 연구는 AI 증강 보안 도구의 미래 이론적 및 실용적 발전에 대한 실증적 증거를 제공합니다.



### Wholly-WOOD: Wholly Leveraging Diversified-quality Labels for Weakly-supervised Oriented Object Detection (https://arxiv.org/abs/2502.09471)
Comments:
          18 pages, 9 figures, 9 tables, accepted by TPAMI

- **What's New**: 이번 논문에서는 Wholly-WOOD라는 새로운 약한 감독 기반의 지향 객체 탐지(Oriented Object Detection, OOD) 프레임워크를 제안합니다. 이 프레임워크는 다양한 레이블 형식(Points, HBoxes, RBoxes 등)을 통합하여 사용할 수 있도록 설계되었습니다. 훈련 시 HBox만을 사용하더라도 RBox 기반의 모델과 비슷한 성능을 보여주어, 지향 객체에 대한 수작업 주석을 줄일 수 있는 가능성을 제시합니다.

- **Technical Details**: Wholly-WOOD는 약한 감독(weak supervision) 접근 방식을 사용하여 점 형태(Point) 및 수평 경계 상자(HBox)와 같은 저급 레이블을 활용하여 지향 경계 상자(RBox)를 생성합니다. 이 방법은 대칭 인식 학습(symmetric-aware learning) 이론을 바탕으로 물체 각도를 학습하고, 합성 패턴 지식을 활용하여 포인트에서 RBox로 변환을 수행합니다. 이러한 기술적 접근은 데이터 주석의 비용을 줄이면서도 효율적인 학습을 가능하게 합니다.

- **Performance Highlights**: Wholly-WOOD는 HBox 또는 Point에서 RBox로의 전환 성능에서 기존 최첨단 방법들보다 우수한 정확도를 기록하였습니다. 특히 Point에서 RBox로의 변환 성능은 22.36% 향상되었으며, 이는 통합 아키텍처와 새로운 P2R 서브넷의 효과 덕분입니다. 또한, 원거리 감시(remote sensing) 및 다양한 응용 프로그램에서의 수작업 주석 감소 효과가 입증되었습니다.



### Metamorphic Testing for Pose Estimation Systems (https://arxiv.org/abs/2502.09460)
Comments:
          Accepted for publication at 2025 IEEE Conference on Software Testing, Verification and Validation (ICST)

- **What's New**: 이 논문에서는 다양한 분야에 활용되는 포즈 추정 시스템의 성능 평가를 위해 MET-POSE라는 새로운 메타모픽 테스트 프레임워크를 제안합니다. 이 시스템은 수작업 주석(labeling)의 필요성을 넘어, 사용자가 특정 애플리케이션에 더 적합한 조건에서 시스템을 평가할 수 있도록 돕습니다. 기존의 테스트 데이터셋에 의존하지 않고, 주석이 필요 없는 방식으로 포즈 추정 시스템의 성능을 평가할 수 있는 가능성을 열어줍니다.

- **Technical Details**: MET-POSE는 포즈 추정 시스템에서 흔히 마주치는 문제들을 해결하기 위한 메타모픽 규칙(metamorphic rules) 목록을 제공합니다. 또한, 이러한 규칙이 어떻게 평가될 수 있는지를 제시하며, Mediapipe Holistic이라는 최신 포즈 추정 시스템에 적용하여 실험적으로 효과를 검증합니다. 이 프레임워크는 FLIC와 PHOENIX 데이터셋을 활용하여 포즈 추정 시스템의 성능을 다양한 환경에서 평가하는 방법을 제시합니다.

- **Performance Highlights**: 실험 결과, MET-POSE는 수작업으로 라벨링한 데이터와 유사하거나 더 높은 속도로 포즈 추정 시스템의 결함(faults)을 발견할 수 있음을 보여줍니다. 사용자는 자신이 필요한 정확도 수준과 결함에 맞춰 규칙 집합(rule set)을 조정할 수 있어, 애플리케이션 중에서 보다 효과적으로 성능 평가를 수행할 수 있습니다. 이는 포즈 추정 디젤로 인해 발생할 수 있는 문제를 사전에 예방하는 데 기여할 것으로 기대됩니다.



### Relational Conformal Prediction for Correlated Time Series (https://arxiv.org/abs/2502.09443)
- **What's New**: 본 논문은 상관된 시계열 예측에서 불확실성 정량화 문제를 다뤘습니다. 특히, 그래프 표현을 활용한 관계형 딥 러닝 방법을 통해 시공간 데이터로부터 점 추정치를 얻는 데 효과적인 도구를 제공함을 강조합니다. 이러한 접근법은 기존의 예측_interval(conformal prediction) 방법들이 독립적으로 작동하는 점을 보완하여 관계를 고려할 수 있도록 합니다.

- **Technical Details**: 저자들은 CoRel이라고 불리는 새로운 관계형 예측 방법을 제안합니다. 이 방법은 그래프 심층 학습(graph deep learning) 기법을 활용하여 상관된 시계열 데이터를 다룹니다. 또한, CoRel은 입력 시계열의 변화에 따라 적응 가능한 요소를 포함하여 비교 가능한 정확성을 유지하면서 비정상적(non-stationary) 데이터의 처리를 지원합니다.

- **Performance Highlights**: CoRel 방법은 다양한 데이터셋과 상황에서 기존의 CP 접근법들보다 뛰어난 성능을 보였습니다. 이는 정확한 예측 범위를 제공하며, 관련 벤치마크에서 최신의 불확실성 정량화(state-of-the-art uncertainty quantification)를 달성했습니다. 이러한 결과는 CoRel이 복잡한 관계를 이해하고 반영하는 데 유용하다는 것을 보여줍니다.



### Variable Stiffness for Robust Locomotion through Reinforcement Learning (https://arxiv.org/abs/2502.09436)
Comments:
          submitted to IFAC Joint Symposia on Mechatronics & Robotics

- **What's New**: 이 논문은 변수 강성을(action space) 제어하여 로봇 다리의 동작을 보다 효율적으로 제어하는 새로운 패러다임을 제시합니다. 이를 통해 필요한 동작에 따라 관절 강성과 다리 강성을 그룹화하여 조절할 수 있으며, 이는 시간 소모적인 수동 조정을 줄이는 데 기여합니다. 특히, 로봇은 다양한 외부 환경에서의 강인한 보행 행동을 보이며, 이는 시뮬레이션에서 실제 환경으로의 이전(transfer)에서의 성과를 나타냅니다.

- **Technical Details**: 논문에서는 Proximal Policy Optimisation (PPO) 알고리즘을 활용하여 다양한 환경에서 정책을 훈련하였습니다. 훈련 과정에서 로봇의 기울기나 관절, 토크, 또는 관절 한계를 초과할 경우 조기 종료되는 방식으로 학습 효율을 높였습니다. Mujoco-MJX를 시뮬레이션 환경으로 사용하고, 무작위로 지정된 파라미터를 적용해 시뮬레이션과 실제 환경 간의 이전을 효과적으로 수행하였습니다.

- **Performance Highlights**: 변수 강성을 채택한 정책들은 기존의 위치 기반 제어 방식보다 속도 추적 및 푸시 회복 성능에서 우수한 결과를 보여주었습니다. HJLS(하이브리드 관절-다리 강성) 제어는 에너지 효율성 또한 뛰어난 성능을 발휘하며, 설계가 단순화되어 다양한 메트릭에서 경쟁력 있는 성과를 유지했습니다. 이러한 결과는 우리가 훈련한 정책이 다양한 외부 힘에 효과적으로 반응함을 명확히 보여줍니다.



### Transformer-Enhanced Variational Autoencoder for Crystal Structure Prediction (https://arxiv.org/abs/2502.09423)
- **What's New**: 이번 연구에서는 Transformer-Enhanced Variational Autoencoder (TransVAE-CSP)를 제안하여 결정 구조 예측의 정확도와 효율성을 크게 향상시키고자 하였습니다. TransVAE-CSP는 안정된 물질의 특성 분포를 학습하고, 결정 구조의 재구성과 생성을 가능하게 합니다. 이 모델은 적응형 거리 확장을 활용하여 결정 구조의 주기성과 대칭을 효과적으로 캡처합니다. 또한 Transformer 네트워크를 인코더로 사용하여 기존의 방법보다 더 높은 성능을 보여줍니다.

- **Technical Details**: TransVAE-CSP는 결정 구조의 특징을 학습하기 위해 등가 점곱 주의 메커니즘을 기반으로 한 Transformer 네트워크를 인코더로 사용합니다. 이 모델은 적응형 거리 확장 메소드를 통해 결정 구조의 특징을 효과적으로 표현합니다. 실험 결과에 따르면, carbon_24, perov_5, mp_20 데이터셋에서 구조 재구성과 생성 작업에서 기존의 방법들을 능가하는 성능을 보였습니다. 이를 통해 결정을 설계하고 최적화하는 강력한 도구를 제공합니다.

- **Performance Highlights**: 실험 결과 TransVAE-CSP는 다양한 모델링 메트릭에서 유의미한 이점이 나타났으며, 구조 재구성과 생성 작업에서 기존 모델들보다 우수한 성능을 발휘했습니다. 이러한 성능 향상은 결정 구조 예측을 위한 인공지능 기술의 진보를 나타내며, 재료 과학 분야에서의 더 나은 소재 발견과 설계에 기여할 것으로 기대됩니다. 이 연구는 향후 결정 구조 예측 및 최적화를 위한 기초를 마련하였습니다.



### A Survey of Reinforcement Learning for Optimization in Automation (https://arxiv.org/abs/2502.09417)
Comments:
          8 pages, 4 tables, and 1 figure. Accepted at IEEE 20th International Conference on Automation Science and Engineering (CASE) 2024

- **What's New**: 이 논문에서는 강화 학습(Reinforcement Learning, RL)이 자동화 분야 내에서 최적화 문제를 해결하는 데 필수적인 도구로 자리 잡았음을 강조합니다. 제조업, 에너지 시스템, 로봇 공학 등 다양한 분야에서 RL의 활용을 살펴보며, 특히 각 분야에서의 최신 동향과 주요 도전 과제를 논의합니다. RL 기반의 최적화 접근 방법은 기존의 전통적인 방법에 비해 보다 유연하고 강력한 결과를 제공할 수 있음을 보여줍니다.

- **Technical Details**: RL은 실험을 통해 학습하는 능력이 뛰어나며, 이를 통해 명시적인 감독이나 사전 정의된 모델 없이도 최적의 정책을 학습할 수 있습니다. 각 응용 분야에서는 RL의 강점을 극대화하기 위해 Deep Reinforcement Learning (DRL) 및 Multi-Agent Reinforcement Learning (MARL)과 같은 최신 알고리즘이 사용됩니다. RL은 데이터 효율성, 안전성, 해석 가능성 및 신뢰성을 개선하고, 실제 환경에서의 배포와 통합 문제를 해결하기 위한 많은 연구가 진행되고 있습니다.

- **Performance Highlights**: 제조업에서 RL은 생산 예약, 재고 관리, 유지 보수 계획 및 공정 제어를 통해 복잡한 최적화 문제를 해결하는 데 기여하고 있습니다. 에너지 시스템에서는 드라이브 반응(Demand Response) 및 마이크로그리드 관리에 있어 RL이 최대 22%의 에너지 절약을 성취하는 등 그 효과를 입증하였습니다. 로봇 공학 분야에서는 DRL의 활용이 로봇의 동작 계획 및 인간과의 협력에서 특히 두드러지며, 이는 다양한 작업에서 협력을 개선하는 데 기여하고 있습니다.



### SQuARE: Sequential Question Answering Reasoning Engine for Enhanced Chain-of-Thought in Large Language Models (https://arxiv.org/abs/2502.09390)
Comments:
          14 pages

- **What's New**: 이번 논문에서는 SQuARE(Sequential Question Answering Reasoning Engine)라는 새로운 방식의 prompting 기술을 소개하며, 이는 LLMs(대형 언어 모델)가 스스로 질문을 생성하고 답변하게 하여 더 깊이 있는 사고를 촉진하는 것을 목표로 합니다. 기존의 chain-of-thought(CoT) 접근 방식에 비해, SQuARE는 복잡한 질문을 여러 단계를 통해 더 철저히 탐색할 수 있도록 돕습니다.

- **Technical Details**: SQuARE는 모델에게 N개의 하위 질문-답변 쌍을 생성하도록 유도하며, 이를 통해 최종 쿼리에 도달하기 전에 다양한 주제를 탐색합니다. 또, 이 방법은 기존의 다른 prompting 기법들과 결합하기 쉽고, 하위 질문 수(N)를 조정함으로써 탐색의 깊이와 계산 비용을 조절할 수 있습니다. 실험은 Llama와 GPT-4o 모델을 사용하여 다양한 QA 데이터 세트에서 실시되었습니다.

- **Performance Highlights**: SQuARE는 TriviaQA, HotpotQA 및 ASQA 데이터 세트에서 기존 방법들을 초과하는 성과를 보여주었습니다. 특히 Llama 3.2 3B 모델을 사용할 때, TriviaQA에서 6.5% 향상된 88.5%의 성과를 기록하였고, GPT-4o를 이용한 경우에도 경쟁력 있는 결과를 보였습니다. SQuARE는 특히 작은 모델에서 최종 답변 품질을 크게 개선하는 데 기여했습니다.



### S$^2$-Diffusion: Generalizing from Instance-level to Category-level Skills in Robot Manipulation (https://arxiv.org/abs/2502.09389)
- **What's New**: 이 연구는 Spatial-Semantic Diffusion policy (S²-Diffusion)를 제안하여 로봇 조작 기술의 일반화 능력을 향상시키는 새로운 접근법을 선보입니다. 기존의 기술은 개별 인스턴스에 국한되어 있는 반면, S²-Diffusion은 카테고리 수준에서 기술을 일반화할 수 있습니다. 이 방식은 심화 심상화 모듈과 공간 표현을 결합하여 각기 다른 인스턴스 간의 기술 이전을 가능하게 합니다.

- **Technical Details**: S²-Diffusion은 단일 RGB 카메라로 깊이 추정 네트워크를 활용하여 로봇이 조작 작업을 수행하는 데 필요한 공간-의미 관찰을 생성합니다. 이 방법은 실시간으로 작동 가능하며 로봇의 고유 감지 정보와 결합하여 공간-의미 정보를 효율적으로 표현합니다. 이러한 접근법은 배경이나 물체의 질감 등 카테고리와 무관한 요인에 대한 불변성을 보장합니다.

- **Performance Highlights**: 연구 결과, S²-Diffusion은 다양한 로봇 조작 작업에서 효과적인 성능을 보여주며, 관찰된 훈련 인스턴스와 다른 인스턴스 간의 일반화를 성공적으로 수행했습니다. 특히 로봇의 조작 기술이 특정 인스턴스에 국한되지 않고, 카테고리 내 다른 인스턴스에서도 만족스러운 성능을 거둘 수 있음을 입증합니다. 모든 실험 비디오는 보완 자료에 포함되어 있습니다.



### Truth Knows No Language: Evaluating Truthfulness Beyond English (https://arxiv.org/abs/2502.09387)
Comments:
          13 pages, 5 figures, 8 tables

- **What's New**: 이번 논문에서는 TruthfulQA 벤치마크의 전문 번역 확장을 소개합니다. 이 확장판은 바스크어, 카탈루냐어, 갈리시아어, 스페인어로 번역되었습니다. 지금까지 대형 언어 모델(LLMs)에 대한 진실성 평가가 주로 영어로 이루어졌는데, 이제는 다양한 언어에서의 LLM의 진실성 유지 능력을 평가할 수 있는 기회를 갖게 되었습니다.

- **Technical Details**: 새롭게 확장된 데이터셋은 TruthfulQA의 문제들을 각 언어에 맞게 번역한 것으로, 기본 대답, 올바른 대답 집합, 부정확한 대답 집합이 포함되어 있습니다. 논문에서는 언어별 인적 평가, 다중 선택 기반 자동 평가 및 LLM-as-a-Judge 점수를 통합하여 총 12개의 최첨단 LLM을 평가했습니다. 평가 방법론은 사람의 판단과 밀접한 관계를 가지며, 맥락과 시간에 의존하지 않는 질문이 LLM의 진실성 평가에 더 효과적임을 보여줍니다.

- **Performance Highlights**: 실험 결과에 따라 대부분의 LLM은 영어에서 가장 진실하고 바스크어에서는 가장 낮은 성능을 보였습니다. 그러나 언어별 진실성 차이는 예상보다 작았습니다. 또한, LLM-as-a-Judge 메트릭이 사람 평가와 더 잘 연관되며, 정보성이 진실성 평가에 중요한 역할을 한다는 점을 강조했습니다. 이는 기계 번역이 진실성 벤치마크의 다국어 확장에 유효한 접근 방식이 될 수 있음을 보여줍니다.



### TRIFFID: Autonomous Robotic Aid For Increasing First Responders Efficiency (https://arxiv.org/abs/2502.09379)
- **What's New**: 이 논문에서는 재난 대응 능력을 향상시키기 위해 무인 지상 및 공중 차량과 고급 인공지능 기능을 통합한 TRIFFID 시스템을 소개합니다. 이 시스템은 와일드파이어, 도시 홍수 및 지진 이후의 수색 및 구호 작전 등의 다양한 재난 상황에서 첫 번째 대응자(First Responder)의 능률을 향상시키기 위해 설계되었습니다. 최첨단 자율 내비게이션 기술, 의미론적 인식(semantic perception), 인공지능을 활용해 운영됩니다.

- **Technical Details**: TRIFFID 시스템은 하이브리드 로봇 플랫폼, 중앙 집중식 지상 스테이션, 맞춤형 통신 인프라 및 스마트폰 애플리케이션과 같은 여러 주요 구성 요소로 이루어져 있습니다. 이 시스템은 심층 신경망(deep neural networks), 지식 그래프(knowledge graphs), 다중 모달 정보 융합(multimodal information fusion)을 활용하여 재난 환경을 자율적으로 탐색하고 분석할 수 있는 능력을 갖춘 로봇을 가능하게 합니다. 이는 대응 시간 단축과 인력 위험 감소를 도모합니다.

- **Performance Highlights**: TRIFFID는 재난 상황에서의 즉각적인 임무 계획, 안전 모니터링, 적응형 작업 실행을 통해 응급 대응 팀의 성능을 향상시킵니다. 이 시스템은 복잡하고 위험한 상황에서도 실시간 상황 인식과 운영 지원을 제공하며, 빠르고 정확한 정보 수집 및 조정된 행동을 촉진합니다. 실세계에서의 실제 응용 사례로는 와일드파이어 대응, 도시 홍수 관리, 및 지진 이후의 수색 및 구호 작업 등이 포함되어 있습니다.



### Language Agents as Digital Representatives in Collective Decision-Making (https://arxiv.org/abs/2502.09369)
- **What's New**: 이번 연구는 집단 의사결정 과정에서 개인의 선호를 대변하는 언어 에이전트를 훈련시키는 가능성을 탐구하고 있습니다. 이를 통해 우리는 디지털 대리 역활의 구현을 위한 새로운 접근 방식을 제안합니다. 기존의 인간 행동 시뮬레이션 연구와는 달리, 우리는 대표성을 위한 시뮬레이션에 중점을 두고 있습니다.

- **Technical Details**: 연구는 집단 의사결정을 에피소드 방식의 상호작용 과정으로 형식화하고, 디지털 대리 문제를 정의합니다. 각 참가자는 집단 결정에 대한 선호를 표현하며, 언어 공간에서의 작용을 처리합니다. 여기서 언어 에이전트는 인간 대리인으로서 적절히 선호를 표현하도록 훈련됩니다.

- **Performance Highlights**: 실험 사례 연구를 통해 다양한 인간 집단에서 합의 찾기 작업의 구현 가능성을 확인했습니다. 대규모 언어 모델을 미세조정(fine-tuning)하여 디지털 대리인으로 작용할 수 있음을 보여주었습니다. 이 연구는 집단 상호작용의 개인화된 시뮬레이션과 메커니즘 설계에 실질적 응용 가능한 결과를 도출합니다.



### Simple Path Structural Encoding for Graph Transformers (https://arxiv.org/abs/2502.09365)
- **What's New**: 이 논문은 Simple Path Structural Encoding (SPSE)이라는 새로운 방법을 제안하여, 그래프 변환기(graph transformers)에 사용되는 엣지 인코딩(edge encoding)에서의 제한 사항을 극복하고자 합니다. SPSE는 간단한 경로의 수를 활용하여 더 풍부한 구조적 정보를 인코딩합니다. 이는 무작위 경로 구조 인코딩(random walk structural encoding, RWSE)의 한계를 극복하는 것이며, 특히 지역 사이클 패턴(local cyclic patterns)을 더 잘 포착할 수 있도록 합니다.

- **Technical Details**: SPSE는 노드 쌍 간의 다양한 길이의 간단한 경로 수를 계산하여 그래프 구조를 인코딩합니다. 이를 위해 효율적인 알고리즘을 제안하는데, 이는 깊이 우선 탐색(DFS)와 너비 우선 탐색(BFS)을 사용하여 생성된 DAG 분해를 기반으로 합니다. 이 접근법은 경로 열거(path enumeration)의 지수 메모리 비용을 피할 수 있으며, 긴 경로 길이에 대한 확장성을 가능하게 합니다.

- **Performance Highlights**: SPSE는 여러 벤치마크에서 RWSE보다 일관되게 더 나은 성능을 보여주었습니다. 특히 분자(molecular) 데이터셋과 긴 범위 그래프(long-range graph) 데이터셋에서 통계적으로 유의미한 개선을 달성하였습니다. 이러한 결과는 SPSE가 그래프 변환기의 표현력을 향상시키기 위한 강력한 엣지 인코딩 대안이 될 수 있음을 나타냅니다.



### Neural Spatiotemporal Point Processes: Trends and Challenges (https://arxiv.org/abs/2502.09341)
- **What's New**: 이 논문은 공간 시간 점 과정(Spatiotemporal Point Processes, STPPs)의 기존 모델들을 검토하고, 최신 딥 러닝 기술을 접목하여 이들을 통해 복잡한 사건 데이터를 좀 더 효과적으로 모델링하는 방법을 소개합니다. 특히, STPPs와 신경망(neural networks) 기술의 통합이 활발한 연구 영역으로 발전하고 있음을 강조합니다. 이 리뷰는 STPPs의 핵심 모델, 응용 분야, 그리고 이벤트 모델링의 주요 구성 요소를 다룹니다.

- **Technical Details**: STPPs는 연속적인 공간과 시간에서 발생하는 무작위 사건의 시퀀스를 모델링하는 데 중점을 둡니다. 특히, 시간의 방향성(unidirectional)과 공간의 다방향성(omnidirectional)으로 인해 모델링이 어려워지며, 전통적인 방법은 강력한 모수(parametric) 가정과 독립성에 의존하여 유연성이 부족합니다. 이러한 한계를 극복하기 위해 신경망 기반의 방법들이 사건 간의 의존성을 처리하고, 이종 패턴을 학습할 수 있도록 설계되었습니다.

- **Performance Highlights**: 신경 STPP는 사건의 복잡한 관계를 포착함으로써 사건의 자기 자극(self-excitation) 및 사건과 공간 변인 간의 상호작용을 효과적으로 모델링할 수 있습니다. 이러한 모델들은 다양한 데이터 유형을 통합하고, 자동으로 특징(feature)을 추출하며, 대규모 환경에서도 효과적으로 확장할 수 있습니다. 이 리뷰는 특히 신경망 기반의 STPPs의 알고리즘과 방법론적 혁신을 체계적으로 탐구하고 있으며, 연구자들에게 실용적인 기초를 제공합니다.



### Graph Diffusion Network for Drug-Gene Prediction (https://arxiv.org/abs/2502.09335)
Comments:
          IEEE/ACM TCBB. 14 pages

- **What's New**: 이번 논문에서는 약물-유전자 예측을 위한 그래프 확산 네트워크(Graph Diffusion Network for Drug-Gene Prediction, GDNDGP)를 도입합니다. 기존의 그래프 신경망(GNN) 방식의 한계를 극복하고자 메타 경로 기반 동질 그래프 학습과 병렬 확산 네트워크를 통해 데이터 희소성(data sparsity) 문제를 해결하였습니다. 이를 통해 훈련 과정에서 효과적인 하드 네거티브 샘플을 생성하여 예측 정확도를 향상시킵니다. GDNDGP는 DGIdb 4.0 데이터셋에서 우수한 성능을 보여주며, 약물-유전자-질병 네트워크에 대한 강력한 일반화 능력을 입증하였습니다.

- **Technical Details**: 제안된 GDNDGP 모델은 메타 경로를 이용한 동질 노드 간 정보 교환을 촉진합니다. 이 모델은 약물 간 및 유전자 간의 관계를 효과적으로 캡처하여 정확한 상호작용 예측을 가능하게 합니다. 또한, 그래프 확산 네트워크를 통합하여 훈련 중 하드 네거티브 샘플을 생성하고, 대량의 연결되지 않은 쌍을 찾을 필요를 없애면서 훈련 효율성을 높입니다. 이러한 접근은 때문에 모델의 판별력을 강화하고 일반화 가능성을 높입니다.

- **Performance Highlights**: 실험 결과, GDNDGP는 기존의 방법들에 비해 약물-유전자 예측 작업에서 유의미한 성능 향상을 보여주었습니다. 특히 복잡한 이질적 관계를 처리하는 능력에서 두각을 나타내었습니다. 이 모델은 약물의 유전자 네트워크와의 관계를 더욱 정확하게 예측하는 데 기여하며, 개인 맞춤형 의약품 개발 및 재창출 가능성에 큰 도움이 될 수 있습니다. 공개된 소스 코드는 관련 연구자들에게 효과적인 도구로 작용할 것입니다.



### When the LM misunderstood the human chuckled: Analyzing garden path effects in humans and language models (https://arxiv.org/abs/2502.09307)
- **What's New**: 이 논문은 현대 대형 언어 모델(LLMs)과 인간의 문장 이해 능력을 비교합니다. 특히, LLM과 사람이 기존의 연구가 다루지 않았던 garden-path 문장을 처리하는 방식을 자세히 분석하였습니다. 연구 결과, 특정 구문 구조에서 LLM과 인간 모두 유사한 어려움을 겪으며, 몇 가지 모델은 인간 이해와 높은 상관관계를 보였습니다.

- **Technical Details**: 연구는 LLM과 인간이 정확히 동일한 과제에 응답하도록 설계되었습니다. 이를 위해 garden-path 문장을 다루는 구문 컴프리헨션(comprehension) 질문과 다양한 심리 언어학적 가설을 제안하였습니다. 주요 가설은(a) 문장의 구문 재분석의 어려움, (b) 문장의 제안 객체의 타당성, (c) 동사의 전이성(transitivity)입니다. 이러한 시도는 LLMs가 인간의 느림 또는 오해를 어떻게 경험하는지 이해하는 데 중요합니다.

- **Performance Highlights**: LLMs의 문장 이해 성능은 인간과 유사한 방식으로 출현하였습니다. 가장 성능이 우수한 모델에서조차 garden-path 문장의 이해 정확도가 78%에 불과했으며, 여기서 고급 LLM들이 인간 행동과 더 유사한 경향을 보였습니다. 추가로 파라프레이징(paraphrasing) 및 텍스트-이미지 생성(text-to-image generation) 작업을 통해 이러한 결과가 LLM의 독해에서도 유사하게 나타남을 확인했습니다.



### Predicting Drive Test Results in Mobile Networks Using Optimization Techniques (https://arxiv.org/abs/2502.09305)
- **What's New**: 이 연구에서는 모바일 네트워크 최적화 과정에서 드라이브 테스트의 필요성을 줄이면서도 신호 세기를 예측할 수 있는 새로운 방법을 제안합니다. 기존의 드라이브 테스트는 비용이 많이 들고 시간이 소요되는 단점이 있지만, 새로 제안된 방법은 다른 드라이브 테스트 데이터를 활용하여 특정 위치에서의 수신 신호 강도를 예측할 수 있게 해줍니다. 이를 통해 운영자는 네트워크 최적화와 기존 드라이브 테스트와 관련된 문제를 해결하는 데 필요한 데이터를 수집할 수 있습니다.

- **Technical Details**: 연구에서는 4G 네트워크의 Reference Signal Received Power (RSRP) 예측을 위해 경로 손실 매개변수를 추정하는 최적화 방법을 사용합니다. 구체적으로는 셀 위치 정보만으로 경로 손실 매개변수를 추정하는 새로운 접근 방식과, 셀 위치 데이터 없이 채널에서의 그림자 효과로 인한 잡음의 표준 편차를 추정하는 기술을 제안합니다. 제안된 방법은 이란의 한 모바일 통신 회사에서 수집된 실제 데이터를 사용해 평가되었습니다.

- **Performance Highlights**: 제안된 방법은 드라이브 테스트를 통해 수집된 데이터를 바탕으로 최적화된 네트워크 성능 예측을 가능하게 하여 높은 정확도를 보여줍니다. 기존 방법들과 비교했을 때, 추가적인 환경 정보를 통합하여 예측의 정확성과 드라이브 테스트 최적화의 효과를 극대화할 수 있음을 증명했습니다. 또한, 다양한 환경에서의 예측 정확성을 개선하는 지능형 기계 학습 모델을 활용하여, 네트워크 최적화의 필요성을 효과적으로 충족할 수 있음을 강조합니다.



### SparQLe: Speech Queries to Text Translation Through LLMs (https://arxiv.org/abs/2502.09284)
- **What's New**: 이 연구에서는 Large Language Models (LLMs)와 음성 표현의 통합을 통해 다중 모달 처리 및 음성 이해를 원활하게 할 수 있는 새로운 접근 방식을 소개합니다. 새롭게 제안된 SparQLe 모델은 self-supervised 학습에서 비롯된 음성 표현을 활용하여 instruction-tuned LLM과 음성을 연결하는 효율적인 방법을 제공합니다. 이 접근 방식은 입력 음성의 의미를 보존하면서 사전 훈련된 음성 인코더와 LLM을 통합할 수 있는 가능성을 보여줍니다.

- **Technical Details**: SparQLe 모델은 음성 표현을 질의하는 데 필요한 정보를 추출하고 이를 사전 훈련된 LLM에 전달하는 기능을 효율적으로 수행합니다. 이 모델은 HuBERT를 음성 인코더로 사용하고, 영어 데이터로 사전 훈련한 후 영어와 프랑스어로 혼합된 데이터를 사용하여 세부 조정을 진행합니다. 모델은 피드 포워드 네트워크와 self-attention을 사용하여 모달리티 어댑터에서 효과적으로 작업을 처리합니다.

- **Performance Highlights**: 실험에서는 MuST-C와 LibriSpeech 데이터 세트를 사용하여 Automatic Speech Translation (AST) 작업을 평가했습니다. 번역 성공률을 높이는 다양한 모달리티 정렬 목표를 설정하여 성능을 극대화하였으며, 모델은 주어진 오디오를 기반으로 텍스트를 생성하는 데 있어 효과적임을 입증했습니다. 이 연구 결과는 다중 모달 음성 이해 애플리케이션에서의 활용 가능성을 확대하는 데 기여할 것입니다.



### LiSA: Leveraging Link Recommender to Attack Graph Neural Networks via Subgraph Injection (https://arxiv.org/abs/2502.09271)
- **What's New**: 이번 연구에서는 Graph Neural Networks (GNNs)에서의 새로운 공격 시나리오인 Link Recommender-Subgraph Injection Attack (LiSA)을 소개합니다. LiSA는 고립된 서브그래프를 주입하여 링크 추천 시스템과 노드 분류기를 동시에 속이는 방식으로, 공격자가 의도하지 않게 링크를 생성하게 합니다. 이를 통해 노드 분류 정확성을 저하시키고, 성공적인 공격을 가능하게 하여 실제 상황에서도 유효성을 입증하고자 합니다.

- **Technical Details**: 제안된 LiSA 프레임워크는 듀얼 서 surrogate 모델과 이급 최적화(bi-level optimization) 기법을 사용하여 서브그래프를 생성합니다. 이 과정에서 GNN의 링크 추천 시스템을 조작하여 노드 간의 연결을 유도함으로써, 악의적인 노드가 목표 노드와 링크를 형성하도록 합니다. 이러한 새로운 접근은 기존의 공격 방식과는 다른 혁신적인 방법론으로 자리잡을 것입니다.

- **Performance Highlights**: LiSA는 다양한 실제 데이터셋을 활용한 실험을 통해 그 효과성과 적용 가능성을 입증했습니다. 구체적으로, 해당 공격은 노드 분류 성능을 크게 저하시켜 GNN 시스템의 작동을 방해하는 데 성공했습니다. 이로써, GNN 기반 애플리케이션에서의 링크 추천 알고리즘의 중요성과 더불어, 악의적인 공격에 대한 방어책 마련의 필요성을 강조합니다.



### Bandit Multiclass List Classification (https://arxiv.org/abs/2502.09257)
- **What's New**: 이 연구는 반응적 피드백(semi-bandit feedback)을 활용한 다중 클래스 리스트 분류(multiclass list classification) 문제를 다룹니다. 기존의 연구가 단일 레이블(single-label) 설정에 초점을 맞춘 반면, 본 연구는 여러 개의 정답 레이블이 있을 수 있는 다중 레이블(multi-label) 설정으로 확장합니다. 이를 통해 배급 시스템(recommendation system)과 같은 현실 세계의 문제에 대한 해결책을 제시합니다.

- **Technical Details**: 저자들은 샘플 복잡성(sample complexity) O(poly(K/m) + sm / ε²) 로그(|H|/δ) 형태의 알고리즘을 제안하여 높은 확률로 ε-최적 가설을 반환합니다. 여기서 H는 기저 가설 클래스(hypothesis class)를 나타내며, s는 각 예제에 대한 실제 레이블의 최대 수를 기준으로 한 상한입니다. 이 알고리즘은 무작위로 생성된 데이터를 반영하여 효율적으로 계산됩니다.

- **Performance Highlights**: 결과적으로 본 알고리즘은 약간의 비용으로 예측 품질을 개선할 수 있음을 보여줍니다. 특히, s=O(1)일 때 주요 항들이 기존의 전체 정보(full-information) 비율과 일치하므로, 반응적 피드백이 실질적인 비용 없이 유용하게 이용될 수 있음을 증명합니다. 이 연구의 결과는 다소 희귀한 보상(s-sparse rewards)을 가지는 맥락적 조합 반응(CCA) 문제로 일반화되어 적용될 수 있습니다.



### DynSegNet:Dynamic Architecture Adjustment for Adversarial Learning in Segmenting Hemorrhagic Lesions from Fundus Images (https://arxiv.org/abs/2502.09256)
Comments:
          12 pages,4 figures

- **What's New**: 이 논문은 망막 hemorrhagic lesion segmentation 분야에 새로운 접근 방식을 제안합니다. 구체적으로, 이 방법은 adversarial learning 기반의 동적 아키텍처 조정을 통해 segmentation 성능을 개선합니다. 기존의 기술적 한계를 극복하기 위해 다양한 최신 기법들을 통합하여 적용했습니다.

- **Technical Details**: 제안된 방법은 계층적 U자형 인코더-디코더(encoder-decoder) 구조, residual blocks, attention mechanisms, 그리고 ASPP(Atrous Spatial Pyramid Pooling) 모듈을 통합합니다. 이러한 구조는 동적으로 feature fusion을 최적화하여 세부적인 segmentation을 가능하게 합니다. 이를 통해 전체적인 성능 개선이 이루어집니다.

- **Performance Highlights**: 실험 결과, Dice coefficient는 0.6802, IoU는 0.5602, Recall은 0.766, Precision은 0.6525, Accuracy는 0.9955로 나타났습니다. 이러한 결과는 fundus image의 hemorrhage segmentation에서 발생하는 여러 도전 과제를 효과적으로 해결할 수 있음을 보여줍니다.



### AnomalyGFM: Graph Foundation Model for Zero/Few-shot Anomaly Detection (https://arxiv.org/abs/2502.09254)
Comments:
          14 pages

- **What's New**: 이 논문에서는 Abnormality Detection을 위한 AnomalyGFM이라는 새로운 그래프 기반 모델을 소개합니다. 이 모델은 제로 샷(zero-shot) 및 소수 샷(few-shot) 설정에서 효과적으로 이상 상황을 탐지할 수 있도록 설계되었습니다. 기존 모델들은 훈련 및 테스트 데이터 간의 분포 차이로 인해 일반화 문제를 겪는 반면, AnomalyGFM은 그래프 간 일반화가 가능한 기법을 채택하고 있습니다.

- **Technical Details**: AnomalyGFM은 노드 표현 잔차(node representation residuals)를 기반으로 정상(normal) 및 비정상(abnormal) 클래스의 데이터 독립적인 프로토타입을 학습합니다. 이 프로토타입은 서로 다른 그래프 간에서 노드의 이상성을 일관되게 측정할 수 있는 통일된 특성 공간(feature space)을 제공합니다. 그래프에 대한 일반화 가능성을 높이기 위해, 이 모델은 클러스터링(cluster) 및 메시지 집계를 위한 새로운 파라다임을 도입했습니다.

- **Performance Highlights**: 11개의 실세계 GAD 데이터셋에 대한 실험 결과, AnomalyGFM은 제로 샷 및 소수 샷 환경에서 기존 최첨단 방법들에 비해 월등한 성능을 보였습니다. 이 모델은 대형 그래프에 대해서도 확장 가능성을 가지며, 레이블이 적은 정상 노드들을 사용하여 성능을 더욱 개선할 수 있는 방법을 지원합니다.



### The Joint Entity-Relation Extraction Model Based on Span and Interactive Fusion Representation for Chinese Medical Texts with Complex Semantics (https://arxiv.org/abs/2502.09247)
- **What's New**: 본 논문에서는 CH-DDI라는 데이터셋을 구축하여 중국 의료 텍스트에서의 엔티티(entities) 및 관계(relation) 추출을 통합적으로 수행하는 모델을 제안합니다. 이를 통해 복잡한 문맥 의미를 더 잘 포착하고자 하며, 기존 방법의 정보 교환 비효율성을 개선하기 위한 상호작용 융합 표현 모듈도 도입합니다. 실험 결과, 제안된 모델이 엔티티 인식에서 96.73%의 F1-score를, 관계 추출에서 78.43%의 성능을 기록하는 등 뛰어난 일반화 능력을 보였습니다.

- **Technical Details**: 저자는 attention 메커니즘을 활용하여 장기 종속성(long-range dependencies)을 포착하는 SEA 모듈을 제안합니다. 이 모델은 Cross Attention을 통해 엔티티 인식과 관계 추출 간의 양방향 정보 교환을 가능하게 하며, BiLSTM을 통해 특성(feature) 추출을 개선합니다. 모델은 다섯 가지 주요 구성 요소로 구성되어 있으며, encoder 모듈은 사전 훈련된 BERT를 사용합니다.

- **Performance Highlights**: CH-DDI 데이터셋에서 제안된 모델은 엔티티 인식에서 96.73%의 F1-score를 달성하였고, 관계 추출에서는 78.43%의 성능을 보였습니다. 공통의 CoNLL04 데이터셋에서도 엔티티 인식 정밀도 89.54%와 관계 추출 정확도 71.64%를 기록하여, 두 데이터셋 모두에서 최상의 성과를 달성했습니다. 이러한 성과는 제안된 모델의 효율성과 우수한 성능을 입증합니다.



### Logical foundations of Smart Contracts (https://arxiv.org/abs/2502.09232)
Comments:
          In Proceedings ICLP 2024, arXiv:2502.08453

- **What's New**: 최근에는 다양한 분야에서 보다 정교한 형식이 요구되고 있습니다. 이러한 요구 중 하나는 사이버 물리 시스템에서 스마트 계약(smart contracts)이 등장하면서 나타났습니다. 스마트 계약은 블록체인 기술을 활용해 분산 시스템 내에서 비즈니스 프로세스를 실행하고 공유할 수 있는 방법으로, 다양한 참여자들 간에 형식적 계약을 체결하는 데 사용됩니다.

- **Technical Details**: 논문에서는 상황 계산법(Situation Calculus)을 기반으로 스마트 계약을 위한 논리적 기초를 제안합니다. 상황 계산법은 행동에 대한 추론을 지원하는 논리 체계로, 복잡한 계약 등 동적 시스템을 명세하고 구현하는 데 적합합니다. 스마트 계약의 동적인 행동을 모델링하기 위해 Golog이라는 프로그래밍 언어가 사용되며, Golog은 Prolog 언어로 작성되어 있습니다.

- **Performance Highlights**: 스마트 계약의 구현은 법률 계약의 복잡성을 해결하기 위한 필요하나, 모든 절차와 효과를 포함하는 것은 긴 시간과 노력이 요구됩니다. 이 연구는 통일되고 보편적인 형식화가 필요한 지속적인 문제를 해결하기 위해 스마트 계약의 형식을 정립하고, 이에 대한 성공적인 구현 방안을 제시하고자 합니다.



### Relating Answer Set Programming and Many-sorted Logics for Formal Verification (https://arxiv.org/abs/2502.09230)
Comments:
          In Proceedings ICLP 2024, arXiv:2502.08453

- **What's New**: 이 논문은 Answer Set Programming (ASP)의 형식적 검증을 위한 새로운 접근 방식을 제안합니다. 특히, ASP 프로그램의 모듈성 결합 및 규칙의 의미 파악의 어려움을 극복하기 위한 대안을 제시합니다. 이를 통해 ASP 검증이 개발 과정에서 일상적으로 수행될 수 있도록 하는 것이 주요 목표입니다.

- **Technical Details**: 연구자는 ASP의 대안적 의미(semantics)를 탐구하며, logic of here-and-there와 다중 정렬 1차 논리(many-sorted first-order logic)로의 변환을 기반으로 한 접근 방식을 사용합니다. 이러한 새로운 의미는 논리 프로그램에 대한 모듈적인 이해를 촉진하고, grounding을 우회하며, 자동 정리 증명기(automated theorem provers)를 사용하여 프로그램 속성을 자동으로 검증할 수 있게 합니다.

- **Performance Highlights**: 제안된 접근 방식은 ASP의 형식적 검증에서 직면하는 고유한 문제들을 해결하며, 그 결과 소프트웨어 시스템의 신뢰성을 향상시킬 수 있습니다. 자동 검증 도구의 사용이 가능해짐에 따라, ASP 프로그램 개발 시 검증 프로세스의 효율성을 크게 증가시킬 수 있습니다.



### Graphical Conditions for the Existence, Unicity and Number of Regular Models (https://arxiv.org/abs/2502.09220)
Comments:
          In Proceedings ICLP 2024, arXiv:2502.08453

- **What's New**: 이 논문에서는 정상 논리 프로그램(nomal logic program)의 의존성 그래프(dependency graph)에 대한 그래픽적 조건을 탐구하여 정규 모델(regular models)의 존재, 유일성 및 수를 분석합니다. 저자들은 비정상적인 정규 모델(non-trivial regular models)의 존재를 위한 필요 조건, 정규 모델의 유일성을 위한 충분 조건, 그리고 정규 모델 수에 대한 상한선 두 가지를 제시합니다. 이 결과들은 기존 결과들을 일반화하며, 저자들은 또한 새로운 연결고리를 발굴하였습니다.

- **Technical Details**: 논문에서는 정규 모델(regular models)의 존재와 관련한 조건을 제시할 때, 세 가지 주요 결과를 도출합니다. 첫째, 비정상적인 정규 모델(non-trivial regular models)의 존재를 위한 필요 조건을 정의하며, 둘째, 정규 모델의 유일성을 확보하기 위한 충분 조건을 설명합니다. 셋째, 긍정적인 피드백 정점 집합(positive feedback vertex sets)을 기반으로 한 정규 모델 수에 대한 두 가지 상한선을 제시합니다.

- **Performance Highlights**: 이 논문에서 제공된 조건들은 You와 Yuan(1994)이 제안한 결과들을 일반화 하며, 특히 정상 논리 프로그램에서 잘 정의된 층화(well-founded stratification)에 대해 적용됩니다. 저자들은 고찰한 결과들의 타당성을 바탕으로, 마침내 유한한 정상 논리 프로그램을 부울 네트워크 이론(Boolean network theory)과 연결하는 중요한 방법을 제시합니다. 이러한 분석은 정규 모델의 특성을 이해하는 데 큰 기여를 합니다.



### Abduction of Domain Relationships from Data for VQA (https://arxiv.org/abs/2502.09219)
Comments:
          In Proceedings ICLP 2024, arXiv:2502.08453

- **What's New**: 이번 논문에서는 도메인 데이터가 부족한 이미지와 질문을 ASP 프로그램으로 표현하는 시각적 질문 응답(VQA) 문제를 연구합니다. 기존 지식 증강 기법을 보완하는 새로운 접근법을 제공하며, 과거 예제에서 이미지 구조의 도메인 관계를 추론(abduction)하는 방법을 도입합니다.

- **Technical Details**: 추론 문제를 설정한 후, 베이스라인(baseline) 접근법을 제시하고 이를 구현합니다. 이 방법은 적은 예제만 있더라도 질문 응답의 정확도를 상당히 향상시킵니다. ASP(Answer Set Programming) 프로그램을 사용하여 이미지와 질문을 표현하는 방식은 기존의 접근법과 대비되는 독립적인 방법론입니다.

- **Performance Highlights**: 제안된 접근법은 기존 기법에 비해 더 높은 정확도가 달성되며, 데이터 수집에 대한 부담을 줄입니다. 또한, 적은 수의 예제만으로도 효과적인 성과를 보이는 점이 크게 강조됩니다.



### Data2Concept2Text: An Explainable Multilingual Framework for Data Analysis Narration (https://arxiv.org/abs/2502.09218)
Comments:
          In Proceedings ICLP 2024, arXiv:2502.08453

- **What's New**: 이 논문은 데이터를 해석하고 내부 기능을 추상화한 후 선택된 자연어로 설명하는 완전한 설명 가능 시스템을 제안합니다. 시스템은 두 가지 주요 단계, 즉 데이터에서 새로운 속성을 식별하고 이를 추상 개념으로 변환하는 단계와 이러한 개념을 자연어로 변환하는 단계로 구성됩니다. 설명 가능한 데이터 해석 파이프라인을 개발함으로써, 의학 정보 처리와 같은 안전-critical 환경에서의 활용이 가능해집니다.

- **Technical Details**: 이 논문은 Prolog/CLP 기반의 변환 시스템을 채택하여 클래스 및 관계로 표현된 개념과 일반 지식에서 파생된 정보를 해석하고 자연어 텍스트로 생성합니다. 주요 기능으로는 계층적 트리 재작성, 모듈식 다국어 생성, 의미적, 문법적 및 어휘적 수준에서의 동등한 변형 지원과 투명한 규칙 기반 시스템이 포함됩니다. 이러한 로직 기반 규칙을 통한 데이터 번역은 설명이 용이하게 모델링됩니다.

- **Performance Highlights**: 제안된 시스템은 입력 개념에 따라 다양한 동등한 재작성을 생성할 수 있는 유연함을 보여줍니다. 시스템의 아키텍처는 모듈화되어 있어 다양한 언어를 지원하고, 텍스트 생성의 수명주기를 단순화하여 비전문가와 시각 장애인에게 나레이션된 정보 접근을 용이하게 합니다. 이 코드는 AI의 설명 가능성을 강조하며, 지식 표현 및 자동 추론 분야에서의 잠재력을 탐구합니다.



### Architecture for Simulating Behavior Mode Changes in Norm-Aware Autonomous Agents (https://arxiv.org/abs/2502.09215)
Comments:
          In Proceedings ICLP 2024, arXiv:2502.08453

- **What's New**: 이 논문은 인간 컨트롤러에 의해 변경될 수 있는 규범 준수(norm compliance) 행동을 가진 지능형 에이전트의 행동을 시뮬레이션하는 아키텍처를 제시합니다. 이 에이전트의 행동 모드를 규범을 준수하는 것에서 더 위험한 방식으로 변경하는 것이 시간에 민감한 구조물 구조물 검색 작업에서 적절할 수 있습니다.

- **Technical Details**: 우리는 Gelfond와 Lobo에 의해 설계된 Authorization and Obligation Policy Language AOPL을 기반으로 규범을 명세합니다. 제안된 아키텍처와 프로토타입 소프트웨어 시스템은 다양한 행동 모드에 따라 에이전트의 계획을 시뮬레이션할 수 있으며, 이후에 컨트롤러가 변경할 수 있는 기능을 가지고 있습니다.

- **Performance Highlights**: 정책 입안자들은 이러한 소프트웨어를 통해 에이전트가 특정 상황에서 어떻게 행동할 수 있는지를 보다 쉽게 이해할 수 있습니다. 시뮬레이션 결과 바람직하지 않은 결과가 나타나면, 정책 입안자들은 자신들의 정책을 개선할 수 있는 기회를 가질 수 있습니다.



### Efficient OWL2QL Meta-reasoning Using ASP-based Hybrid Knowledge Bases (https://arxiv.org/abs/2502.09206)
Comments:
          In Proceedings ICLP 2024, arXiv:2502.08453

- **What's New**: 이 논문은 온톨로지에서 클래스와 역할이 다른 클래스의 구성원이나 역할로 나타날 수 있는 메타모델링(metamodeling)에 관한 새로운 접근 방식을 제시합니다. 메타모델링은 여러 응용 프로그램에서 바람직한 모델링 기능이지만, 무제한으로 허용될 경우 불확실성(undecidability)을 초래할 수 있는 문제점이 있습니다. 이전 연구를 기반으로 하여, 하이브리드 지식 베이스(hybrid knowledge bases)에서의 쿼리 응답(query answering)을 사용하여 메타모델링을 조정하는 방법을 발전시킵니다.

- **Technical Details**: 이 논문에서는 메타모델링 쿼리 응답을 Datalog 쿼리 응답으로 축소하는 기존 연구를 확장하여, Datalog 변환이 필요한 경우에만 적용하는 식으로 하이브리드 지식 베이스에서의 쿼리 응답 효율성을 개선하는 방법을 설명합니다. 또한, 대안 도구(alternative tools)를 사용하여 이론적 기초를 개선하고 경쟁력 있는 성능을 보여주기 위한 연구도 포함되어 있습니다. 이는 메타모델링을 처리하기 위한 보다 나은 수단을 제공하는 것을 목표로 합니다.

- **Performance Highlights**: 예비 연구(preliminary work)에서는 이 접근이 작동함을 보였으나, 기대했던 성능 개선은 아직 관찰되지 않았습니다. 따라서 본 연구는 이론적 기반을 강화하고 경쟁력 있는 성능을 가진 도구를 활용함으로써 이러한 성능 문제를 해결하고자 합니다. 최종적으로, 새로운 메타모델링 방법론이 다양한 적용 사례에서 실질적인 이점을 제공하기를 기대하고 있습니다.



### Matina: A Large-Scale 73B Token Persian Text Corpus (https://arxiv.org/abs/2502.09188)
- **What's New**: 이번 연구에서는 페르시아어를 위한 새로운 데이터셋인 Matina Corpus를 소개합니다. 이 데이터셋은 72.9B 토큰으로 구성되어 있으며, 높은 품질을 보장하기 위한 철저한 전처리(preprocessing) 및 중복 제거(deduplication) 과정을 거쳐 생성되었습니다. Matina Corpus는 다양한 자료 출처를 포함하여 기존의 페르시아어 데이터셋들보다 더 나은 데이터 품질을 가지고 있어, NLP 모델의 성능 향상에 기여할 것입니다.

- **Technical Details**: Matina Corpus는 블로그와 뉴스 기사를 주로 사용하는 기존 데이터셋들과 달리, 다양한 새롭게 수집된 자료를 포함합니다. 데이터셋은 두 가지 주요과정인 전처리와 중복 제거의 고유한 절차를 통해 구축되었으며, 이를 통한 훈련으로 transformer 기반 모델들이 페르시아어 NLP 작업에서 성능의 개선을 보였습니다. 데이터셋의 소스에는 최신 페르시아어 위키백과 업데이트와 Madlad, CulturaX가 포함되어 있습니다.

- **Performance Highlights**: Matina Corpus의 도입으로 XML-RoBERTa 모델을 재훈련하면서, 감정 분석, 텍스트 감정 인식 및 개체 인식에서 해당 모델의 성능이 현저히 향상되었습니다. 또한, LLaMA 3.1 8B 모델을 사용해 페르시아어 이해도를 높였으며, 이로 인해 다국어 모델의 페르시아어 처리 능력도 개선되었습니다. 전체적으로 이 데이터셋은 페르시아어 NLP의 발전을 위해 기여할 수 있는 중요한 자원입니다.



### RefineCoder: Iterative Improving of Large Language Models via Adaptive Critique Refinement for Code Generation (https://arxiv.org/abs/2502.09183)
Comments:
          work in process

- **What's New**: 이 논문은 기존의 교사 모델 모방에서 벗어나 Adaptive Critique Refinement (ACR)이라는 새로운 방법론을 제안합니다. ACR은 모델이 스스로 생성한 코드를 외부 비평과 함께 개선하도록 하여, 교사 모델의 코드 응답을 직접적으로 모방하는 대신에 자기 정제(self-refinement) 능력을 활용합니다. 이를 통해 더 적은 데이터로도 우수한 성능을 달성할 수 있도록 하는 RefineCoder 시리즈를 개발하였습니다.

- **Technical Details**: ACR은 LLM-as-a-Judge와 LLM-as-a-Critic을 활용하여 코드 응답의 품질 평가 및 저품질 코드 응답에 대한 비판 작용을 포함하는 복합 점수 시스템을 도입합니다. 이 방법은 자가 생성된 코드를 점수화하고 비평한 결과에 따라 새로운 샘플을 구성하는 과정을 포함하며, 이를 통해 코드 생성 능력을 지속적으로 발전시킵니다. RefineCoder 모델들은 세 번의 반복 과정을 통해 코드 생성 역량을 개선하여 다양한 벤치마크에서 인상적인 결과를 보여주었습니다.

- **Performance Highlights**: RefineCoder 시리즈는 HumanEval, MBPP, LiveCodeBench, BigCodeBench-hard와 같은 여러 코드 생성 벤치마크에서 뛰어난 성능 개선을 기록하였습니다. RefineCoder-DS-6.7B는 평균 pass@1에서 2.4p, RefineCoder-QW-7B는 3.0p의 향상을 이뤄냈습니다. ACR을 통한 iterative 방식이 코드 생성 성능을 지속적으로 향상시키고 있으며, 동등한 크기의 기존 모델들에 비해 더 적은 데이터로도 경쟁력 있는 성능을 달성하였습니다.



### FLAME: Flexible LLM-Assisted Moderation Engin (https://arxiv.org/abs/2502.09175)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 사용자 모델 상호작용을 보다 안전하게 관리하기 위한 새로운 접근 방식인 FLAME(Flexible LLM-Assisted Moderation Engine)를 소개합니다. 기존의 입력 필터링(input filtering) 대신 출력 조정(output moderation)에 중점을 둔 FLAME은 최신의 'Best-of-N' 유형의 'jailbreaking' 공격에 더욱 효과적으로 저항할 수 있는 방법을 제시합니다. FLAME은 가벼운 방식으로 설계되어 접근 가능성을 높이고, 새로운 공격에 대한 신속한 적응을 가능하게 합니다.

- **Technical Details**: FLAME은 이진 분류 문제로 사용자의 요청이나 모델의 응답이 금지된 주제를 포함하는지를 확인합니다. 이 알고리즘은 n-그램(n-grams) 처리와 고유한 규칙 기반 분류 함수를 통해 간단하고 효율적인 처리를 수행합니다. pymorphy3와 nltk 같은 도구를 사용하여 단어 형태(normalized forms)를 처리하고 필터링 기능을 통해 필요한 주제를 조정할 수 있습니다.

- **Performance Highlights**: FLAME은 이전의 moderation 시스템에 비해 뛰어난 성능을 발휘하여 GPT-4o-mini 및 DeepSeek-v3에서 'jailbreaking' 공격의 성공률을 9배 이상 줄였습니다. 여러 LLM 플랫폼에서의 실험을 통해 FLAME이 2배에서 9배의 저항력을 보여준 것을 확인했습니다. 이 결과는 LLM의 콘텐츠 조절 시스템의 효율성을 크게 향상시키는 데 기여하고 있습니다.



### Two-Stage Representation Learning for Analyzing Movement Behavior Dynamics in People Living with Dementia (https://arxiv.org/abs/2502.09173)
Comments:
          AAAI 2025 Workshop on Large Language Models and Generative AI for Health

- **What's New**: 이번 연구에서는 고주파 데이터에서 중요한 환자 행동 패턴을 드러내는 원격 의료 모니터링을 위한 시간 시계열 표현 학습을 제안합니다. 특히, 치매 환자들의 홈 활동 데이터를 분석하기 위해 두 단계의 자가 지도 학습 접근 방식을 채택하였으며, 이를 통해 저차원 구조를 발견하도록 구성되었습니다. 이 방법은 PageRank 기반 방법을 사용하여 복잡한 행동 데이터를 간결하게 압축하여 해석 가능성을 향상합니다.

- **Technical Details**: 연구에서는 원시 시계열 데이터를 전처리한 후 사전 훈련된 언어 모델을 활용하여 텍스트 시퀀스로 전환합니다. 이 과정은 시간 데이터의 특성을 포착하는 데 중점을 두며, PageRank 알고리즘을 통해 저차원 상태 벡터를 추출하고 상태 간 전이 패턴을 분석합니다. 최종적으로 이 분석 프레임워크는 환자의 행동 역학에 대한 통찰을 제공합니다.

- **Performance Highlights**: 연구 결과, 제안된 프레임워크는 인지 상태 예측과 맞춤형 치료 개입 및 대규모 건강 모니터링을 지원하는 잠재력을 보여주었습니다. 연구에서 발견된 행동 패턴은 MMSE 및 ADAS-COG와 같은 임상 지표와의 상관관계를 밝혀, 환자 치료의 질적 향상을 기대할 수 있습니다.



### Automatic Pruning via Structured Lasso with Class-wise Information (https://arxiv.org/abs/2502.09125)
Comments:
          11 pages, 2 figures

- **What's New**: 이번 연구에서는 정확한 클래스별(class-wise) 정보를 활용하여 신경망 모델의 프루닝(pruning) 접근 방식을 개선했습니다. 정보 병목(Information Bottleneck) 이론을 기반으로 한 구조적 라쏘(structured lasso) 기법을 이용하여 통계적 정보를 유지하면서 프루닝을 수행하는 두 가지 새로운 방법, 즉 sGLP-IB와 sTLP-IB를 제안합니다. 이를 통해 데이터 세트와 모델 아키텍처의 다양성을 감안하여 우수한 성능을 입증하였습니다.

- **Technical Details**: 제안된 방법들은 CNN의 특징 맵(feature map) 통계 정보를 모델링하여 프루닝 손실을 최소화하는 데 목표를 두고 있습니다. 또한, 그래프 구조화된 라쏘(graph-structured lasso) 및 트리 유도 라쏘(tree-guided lasso)를 사용하여 필터 간의 관계를 고려하며, 클래스별 정보를 통합하는 접근 방식을 채택합니다. 이러한 구조적 라쏘 기법을 활용하여 모델 필터의 프루닝을 보다 정밀하게 진행하고, 누적 오류를 방지하며 계층 수준(layer-wise)에서 수행합니다.

- **Performance Highlights**: VGG16 모델을 CIFAR-10 데이터 세트에서 사용했을 때, 85%의 매개변수 감소와 61%의 FLOPs 감소를 달성하였으며, 정확도는 원래 모델보다 0.14% 높은 94.10%를 기록했습니다. 또한 ResNet 아키텍처를 기준으로 ImageNet 데이터 세트에서 매개변수를 55% 줄이고 정확도를 76.12%로 유지할 수 있었습니다. 이러한 실험 결과는 제안된 방법의 효과성과 강력한 클래스 정보 학습 능력을 입증합니다.



### Improving Deep Regression with Tightness (https://arxiv.org/abs/2502.09122)
Comments:
          ICLR 2025, Code: this https URL

- **What's New**: 이 논문은 딥 회귀에서 타겟의 순서를 보존함으로써 다양한 과제의 성능을 향상시킬 수 있음을 보여줍니다. 그러나 순서 보존이 성능에 미치는 이점에 대한 이론적 설명은 부족했습니다. 본 연구에서는 조건부 엔트로피 $H(Z|Y)$를 최소화하여 표현 Z와 타겟 Y 간의 유사성을 유지하는 최적 수송 기반 규제를 소개하고, 이를 통해 회귀의 성능을 개선하는 방법을 제안합니다.

- **Technical Details**: 순서를 보존함으로써 표현 Z의 조건부 엔트로피 $H(Z|Y)$가 감소하는 것을 발견하였습니다. 일반적인 회귀 손실이 이 조건부 엔트로피를 줄이기에는 미흡하다는 것을 보여줍니다. 이를 해결하기 위해, 회귀기 목표를 중복하여 사용하는 전략과 함께 Regression Optimal Transport (ROT) Regularizer를 도입하여 표현의 안정성을 높입니다.

- **Performance Highlights**: 세 가지 실제 회귀 과제에서 제안한 전략의 효과를 검증하였습니다. 다중 목표 접근법과 ROT-Reg가 각각 전역 및 지역적으로 표현을 조정하여 성능을 극대화함을 확인했습니다. 이 연구는 회귀 표현의 순서 보존과 관련된 기여로, 회귀 작업을 분류 문제로 재정의할 수 있는 통찰력을 제공합니다.



### One-shot Federated Learning Methods: A Practical Guid (https://arxiv.org/abs/2502.09104)
Comments:
          10 pages, 1 figure

- **What's New**: 이번 논문에서는 One-shot Federated Learning (OFL)의 도전 과제를 체계적으로 분석하고 현재 방법들을 심층적으로 검토합니다. OFL은 데이터 프라이버시와 통신 오버헤드를 줄이는 데 초점을 맞추며, 특히 데이터 이질성과 모델 이질성을 해결하는 데 필요한 새로운 분류 방법을 제안합니다. 또한, OFL과 관련된 기존의 연구가 부족한 문제를 해결하고자 미래 방향성을 제시합니다.

- **Technical Details**: OFL은 클라이언트가 단 한 번 모델 파라미터를 전송하여 글로벌 모델을 업데이트하는 분산 기계 학습 패러다임입니다. 이를 통해 이전의 전통적 Federated Learning (FL)에서 발생하는 여러 라운드의 데이터 교환을 피하고, 통신 오버헤드와 보안 강화를 동시에 달성할 수 있습니다. OFL의 도전 과제는 데이터 비독립적이고 비동일 분포(non-IID) 문제와 모델 이질성으로 크게 나눌 수 있습니다.

- **Performance Highlights**: 연구에 따르면 OFL은 기존 FL 방법보다 개선된 성능을 보여주기 위한 다양한 방법을 탐색하고 있으며, 많은 연구자들이 이 분야에 관심을 갖기 시작했습니다. 다음 섹션에서는 OFL의 기술적 측면을 상세히 다루고, 다양한 기술적 아이디어의 장단점을 논의하여, 연구자들이 OFL을 실용화할 수 있도록 돕고자 합니다.



### Show Me the Work: Fact-Checkers' Requirements for Explainable Automated Fact-Checking (https://arxiv.org/abs/2502.09083)
Comments:
          Conditionally accepted to CHI'25

- **What's New**: 최근 온라인 미디어에서 대형 언어 모델과 생성 AI의 확산으로 인해 역동적으로 증가하는 허위 정보에 대한 효과적인 자동 사실 확인의 필요성이 강조되고 있습니다. 이러한 연구는 사실 확인자가 제공한 증거를 평가하는 방법 및 절차와 자동화 도구의 실제 활용 방식, 그리고 이에 필요한 설명 기준을 규명하는 데 주력합니다. 논문은 사실 확인 프로세스의 각 단계에서 점검해야 할 설명의 필요성을 제시하고, 자동화된 시스템이 사실 확인자의 요구사항을 충족하기 위해 어떤 정보를 제공해야 하는지에 대한 불확실성을 드러냅니다.

- **Technical Details**: 이 연구는 5개 대륙의 10명의 사실 확인 전문인과의 반구조적 인터뷰를 통해 사실 확인자가 증거를 평가하고 결정을 내리는 과정에서 어떤 정보를 설명하는 것이 중요한지에 대해 논의합니다. 연구는 자동화 도구가 사실 확인자의 작업 흐름에서 어떻게 사용되는지에 대한 특성을 정의하고, 사실 확인 프로세스의 각 단계에서 요구되는 설명의 종류를 명확히 합니다. 이러한 구체적 요구 사항들은 모델의 추론 경로와 특정 증거를 참조하며 불확실성과 정보 간극을 강조해야 함을 enthalten합니다.

- **Performance Highlights**: 연구 결과는 현재의 자동 사실 확인 상태와 사실 확인자의 실제 필요 사이의 격차를 명확히 하며, 반복 가능한 설명의 중요 기준을 제안합니다. 또한, 자동화된 사실 확인 시스템의 설명과 그 투명성 부족이 사실 확인자들의 의사 결정 과정에 미치는 영향을 설명함으로써, 실질적으로 도움을 줄 수 있는 방법론을 제시합니다. 결과적으로, 이 연구는 자연어 처리(NLP) 기술의 발전이 사실 확인자의 요구를 어떻게 충족시킬 수 있는지를 탐색하며, 향후 기술적 발전 방향을 제안합니다.



### CoSER: Coordinating LLM-Based Persona Simulation of Established Roles (https://arxiv.org/abs/2502.09082)
- **What's New**: 최근 대규모 언어 모델(LLMs)의 발전으로 인하여 역할 수행 언어 에이전트(RPLA)가 주목받고 있습니다. 그러나 기존 캐릭터를 재현하는 데 있어 진정한 데이터 세트의 부족과 부정확한 평가 방법이 큰 도전 과제로 남아 있습니다. 이 논문에서는 CoSER라는 고품질 데이터 세트와 공개 모델, 평가 프로토콜을 소개하여 효과적인 역할 수행 언어 모델을 개발하고자 합니다.

- **Technical Details**: CoSER 데이터 세트는 771개의 저명한 소설에서 17,966명의 캐릭터로부터 수집된 다채로운 데이터로 구성되어 있습니다. 여기에는 실제적인 대화, 캐릭터 경험 및 내부 생각 등이 포함되어 있습니다. 주어진 상황에 맞춘 연기(Given-Circumstance Acting) 기법을 통해 LLM을 훈련하고 평가하며, CoSER 8B와 CoSER 70B와 같은 첨단 역할 수행 LLM 모델을 개발하였습니다.

- **Performance Highlights**: CoSER 70B는 다양한 평가 및 기준에서 최신 성능을 달성하였으며, GPT-4o와 비교했을 때 뛰어난 정확도를 보였습니다. 특히, InCharacter와 LifeChoice 벤치마크에서 각각 75.80%와 93.47%의 정확도를 기록하여 CoSER 데이터 세트의 가치를 입증했습니다. 이 결과는 CoSER 데이터 세트가 RPLA 훈련과 평가에 큰 기여를 할 수 있음을 나타냅니다.



### An Open Recipe: Adapting Language-Specific LLMs to a Reasoning Model in One Day via Model Merging (https://arxiv.org/abs/2502.09056)
Comments:
          9 pages

- **What's New**: 본 논문은 DeepSeek R1과 같은 고급 추론 능력을 언어 특화 대규모 언어 모델(LLMs)에 통합하기 위한 데이터 선택 및 모델 합병 방법론을 조사하고 있습니다. 특히, 태국어 LLM에 초점을 맞추어 언어 특정 LLM의 추론 능력을 향상시키는 동시에 특정 언어 기능은 유지하도록 하고 있습니다. 저자는 공개적으로 이용 가능한 데이터셋과 120달러의 컴퓨팅 예산으로도 이러한 목표를 달성할 수 있음을 보여주고 있습니다.

- **Technical Details**: 본 연구에서는 목표 언어(예: 태국어)와 장기 추론에 특화된 두 개의 모델을 선택한 다음, 모델의 내부 표현을 정제하여 일관성 있게 결합하는 두 단계 절차를 사용합니다. 첫 번째 단계에서는 Supervised Fine-Tuning(SFT) 기법을 통해 언어 모델의 성능을 강화하며, 이후 Ability-Aware Model Merging 기법으로 두 모델의 파라미터를 통합하여 새로운 모델을 생성합니다. 이 과정에서 태국어와 영어의 질문-해답 쌍을 연결하는 이중 언어 설정을 활용합니다.

- **Performance Highlights**: 실험에서는 Typhoon2 70B Instruct와 DeepSeek R1 70B Distill 모델을 사용하여 각각의 추론 능력과 언어 작업 성능을 평가하였습니다. 평가를 통해 언어 모델의 태국어 성능을 유지하면서도 DeepSeek R1 수준의 추론 능력을 도달하는 것이 가능함을 입증하였습니다. 이러한 결과는 고-resource 언어 중심의 모델이 아닌, 저-resource 언어를 위한 모델 개선의 중요성을 강조합니다.



### Exploring the Needs of Practising Musicians in Co-Creative AI Through Co-Design (https://arxiv.org/abs/2502.09055)
Comments:
          Paper accepted into CHI 2025, Yokohama Japan, April 26th - May 1st

- **What's New**: 최근 생성적 AI 음악의 발전으로 인해 음악가들이 공동 창작할 수 있는 도구들이 등장했습니다. 본 논문은 실습 중인 음악가들의 필요를 이해하기 위해 공동 설계(co-design) 방법론을 활용한 사례 연구를 제시합니다. 이를 통해 다양한 음악가들이 디자인 과정에 참여하여 음악 AI 시스템의 발전에 기여할 수 있음을 강조합니다.

- **Technical Details**: 본 연구는 두 개의 워크숍과 두 주간의 생태학적 평가(ecological evaluation)를 통해 13명의 다양한 배경을 가진 실습 음악가들을 대상으로 진행되었습니다. 연구 결과, 음악가들이 도구의 설계와 그들의 개인 실습에 대한 통찰을 제공하면서 AI의 공동 창작 역할을 명확히 정의하는 데 기여하게 되었습니다. 이 과정에서 창작 과정의 소유권, 기술을 협력자로서의 프레이밍(frameworking) 등이 중요하게 다루어졌습니다.

- **Performance Highlights**: 연구를 통해 도출된 통찰(insights)은 향후 설계자들이 공동 창작 시스템을 개발할 때 고려할 요소들을 제시합니다. 특히, 음악가들의 창작 과정에 대한 소유 욕구와 다양한 음악적 배경에 따른 요구사항이 중요하게 논의되었습니다. 이러한 결과는 차기 세대의 음악 AI 시스템 설계에 실질적인 기초 자료로 활용될 수 있습니다.



### AIDE: Agentically Improve Visual Language Model with Domain Experts (https://arxiv.org/abs/2502.09051)
Comments:
          6 pages, 4 figures, 2 tables

- **What's New**: 이 논문에서는 시각 언어 모델(Visual Language Models, VLMs)의 성능을 향상시키기 위한 새로운 프레임워크인 AIDE(Agentic Improvement through Domain Experts)를 소개합니다. AIDE는 전문 도메인 모델을 활용하여 VLMs의 개선을 자동화하여, 기존의 더 큰 모델의 의존성을 줄이는 데 중점을 두고 있습니다. 이 프로세스는 정제할 사례를 식별하고, 전문 모델의 분석을 활용하며, 개선된 데이터를 학습 파이프라인에 통합하는 네 단계로 구성되어 있습니다.

- **Technical Details**: AIDE는 VLM의 훈련 데이터를 향상시키기 위해 두 가지 주요 에이전트인 Selector와 Synthesizer를 포함합니다. Selector는 개선이 필요한 사례를 식별하고, 해당 사례에 적합한 전문 도구와 매칭합니다. Synthesizer는 전문가의 출력을 기존 데이터와 통합하여 훈련 예제를 생성하는 과정에서 여러 출처의 정보를 집계하고 잠재적 충돌을 해결하여 더 풍부하고 일관된 응답을 생성합니다.

- **Performance Highlights**: 다양한 벤치마크 실험을 통해 AIDE는 기존의 모델에 비해 성능 개선을 달성하였으며, 특히 MMMU에서 1.2%, MMBench에서 0.77%, MME에서 52%의 개선 효과를 보였습니다. 이 프레임워크는 대형 모델에 대한 접근 없이도 VLMs의 지속적인 개선을 가능하게 하여, 현재의 방법론에서의 중대한 한계를 극복하는 데 기여할 수 있음을 보여줍니다.



### Leveraging Member-Group Relations via Multi-View Graph Filtering for Effective Group Recommendation (https://arxiv.org/abs/2502.09050)
Comments:
          5 pages, 3 figures, 4 tables; ACM Web Conference (WWW 2025) (to appear) (Please cite our conference version.)

- **What's New**: 이 연구에서는 기존의 딥러닝(Deep Learning) 기반의 그룹 추천 시스템의 복잡한 훈련 절차를 극복하기 위해 Group-GF라는 새로운 접근 방식을 제안합니다. Group-GF는 다양한 관점을 제공하는 다중 뷰 그래프 필터링(multi-view graph filtering)을 이용하여 그룹에게 신속한 추천을 수행합니다. 이는 멤버와 그룹 간의 복잡한 상호 작용을 반영하면서도 비싼 모델 훈련 없이 효율적인 추천이 가능하게 합니다.

- **Technical Details**: Group-GF는 세 가지 아이템 유사성 그래프를 구성한 후, 각 그래프에 대한 차별화된 다항 그래프 필터를 최적 설계합니다. 이 방법은 멤버-그룹 매트릭스와 아이템 유사성 그래프를 결합하여 그래프 신호를 효과적으로 통합합니다. 각 유사성 그래프에 대해 최적화된 필터링을 수행한 뒤, 세 가지 그래프 필터를 집계하여 그룹 추천의 정확성을 높입니다.

- **Performance Highlights**: Group-GF는 벤치마크 데이터세트에서 최첨단 정확도와 함께 최대 1.55초의 놀라운 효율적인 런타임을 달성했습니다. 이러한 성능은 고급 훈련 없이도 복잡한 멤버-그룹 동역학을 처리할 수 있는 다중 뷰 그래프 필터링에 기인합니다. 이 연구는 Group-GF의 필터링 과정이 최적화와 매끄러움 규제를 통해 모델의 행동을 더 명확하게 해석할 수 있도록 이론적으로 연결되었습니다.



### Criteria-Aware Graph Filtering: Extremely Fast Yet Accurate Multi-Criteria Recommendation (https://arxiv.org/abs/2502.09046)
Comments:
          12 pages, 8 figures, 7 tables; ACM Web Conference (WWW 2025) (to appear) (Please cite our conference version.)

- **What's New**: 이번 연구에서는 훈련이 필요 없는 다기준( MC ) 추천 시스템을 제안합니다. CA-GF(criterion-aware graph filtering)를 사용하여 효율적이고 정확한 추천을 제공합니다. 복잡한 다차원 사용자 피드백을 처리하면서도 높은 정확도를 유지할 수 있도록 설계되었습니다.

- **Technical Details**: CA-GF는 MC 사용자 확장 그래프를 기반으로 아이템-아이템 유사도 그래프를 구축합니다. 이후 기준별 그래프 필터링을 통해 각 기준에 최적화된 필터를 찾고, 기준 선호도를 반영한 집계를 수행합니다. 이러한 접근법은 기존의 DNN 기반 방법보다 훈련 시간과 모델 성능에서 우수한 결과를 보여줍니다.

- **Performance Highlights**: CA-GF는 최대 24%의 정확도 향상을 달성하며, 대규모 벤치마크 데이터셋에서도 0.2초 미만의 실행 시간을 기록합니다. 또한, 각 기준의 기여도를 시각적으로 설명함으로써 모델 해석 가능성도 크게 향상됩니다.



### Typhoon T1: An Open Thai Reasoning Mod (https://arxiv.org/abs/2502.09042)
Comments:
          25 pages, 6 figures

- **What's New**: 본 논문은 Typhoon T1이라는 새로운 오픈 태국어 추론 모델 개발 프로젝트를 소개합니다. 이 모델은 최근 대형 언어 모델(LLMs) 위에 구축된 새로운 형태의 generative model로, 복잡한 작업을 수행하는 데 있어 성능을 향상시키는 장기적인 사고 과정을 생성합니다. 추가적으로, 태국어와 같은 낮은 자원 언어에서 추론 흔적을 생성하는 것에 대한 세부 사항이 부족함을 인식하고, 모델 개발 과정에서의 통찰력을 공유합니다.

- **Technical Details**: Typhoon T1은 감독된 세부 조정(supervised fine-tuning, SFT) 방법론을 사용하여 데이터셋과 모델 가중치를 개방하고, 길고 복잡한 사고 과정을 구성하는 방법에 대한 실험을 진행합니다. 모델 선택에서는 Typhoon 2 3B Instruct를 사용하여, 강화 학습(reinforcement learning, RL)의 불안정성을 회피하며 실험을 진행하였습니다. 또한, 세 가지 사고 포맷(구조화된 사고, 비구조화된 사고, 반구조화된 사고)에 대한 비교 분석을 실시하며, 모델의 성능 향상이 가능한지를 조사합니다.

- **Performance Highlights**: Typhoon T1 모델의 특징은 다양한 작업에 대한 추론을 할 수 있는 능력과 태국어 데이터 생성을 통해 낮은 자원 언어의 연구 기반을 선도할 수 있다는 것입니다. 향후 연구 방향으로는, 추론 모델의 성능을 개선하기 위한 냉각 구조화된 사고 포맷의 효과성을 평가하고 있으며, 연구가 오픈 데이터셋과 연계되어 지속적인 발전이 이루어질 것으로 기대됩니다.



### Large Images are Gaussians: High-Quality Large Image Representation with Levels of 2D Gaussian Splatting (https://arxiv.org/abs/2502.09039)
Comments:
          Accepted by 39th Annual AAAI Conference on Artificial Intelligence (AAAI 2025). 10 pages, 4 figures

- **What's New**: 이번 연구에서 우리는 Gaussian Splatting (GS)을 기반으로 한 새로운 이미지 표현 기법인 LIG(Large Images are Gaussians)를 소개합니다. LIG는 큰 이미지들을 Gaussian 포인트로 표현하는 방법으로서, 기존 2D Gaussian Splatting (2DGS)의 한계를 극복하고 있습니다. 특히, 많은 Gaussian 포인트를 효과적으로 관리할 수 있는 두 가지 주요 수정 사항을 통해 대용량 이미지를 고품질로 피팅하는 데 성공하였습니다.

- **Technical Details**: LIG는 2DGS 접근 방식을 활용하여 이미지 피팅의 최적화 문제를 다룹니다. 첫 번째로, Gaussian 매개변수를 새로운 2DGS 표현 방식으로 최적화하며, 이는 CUDA 커널을 재구현하여 진행됩니다. 두 번째로, 컴퓨터 그래픽의 Level of Detail (LOD) 개념을 적용하여, 정밀하고 세밀한 구조를 효율적으로 맞출 수 있습니다. 이러한 접근 방식은 많은 수의 Gaussian 포인트의 훈련을 쉽게 만들어 줍니다.

- **Performance Highlights**: 실험 결과 LIG는 의료 이미지와 원격 감지 이미지 등 다양한 대용량 이미지에 대해 우수한 피팅 성능을 보였습니다. 우리의 기술은 텔레메디슨 및 위성 통신과 같은 응용 분야에서 큰 가능성을 보여줍니다. 또한, 여러 대조군과의 비교를 통해 LIG의 품질과 효율성이 기존의 INR 기반 방법들을 초과함을 입증하였습니다.



### EventSTR: A Benchmark Dataset and Baselines for Event Stream based Scene Text Recognition (https://arxiv.org/abs/2502.09020)
Comments:
          In Peer Review

- **What's New**: 이번 논문에서는 기존 RGB 카메라를 기반으로 한 Scene Text Recognition (STR) 알고리즘의 한계를 극복하기 위해, 바이오 영감을 받은 이벤트 카메라를 활용하여 EventSTR이라는 대규모 벤치마크 데이터세트를 제안합니다. 이 데이터세트에는 9,928개의 고화질 샘플이 포함되어 있으며, 중국어 및 영어 문자 인식을 지원합니다. 또한, 새로운 이벤트 기반의 STR 프레임워크인 SimC-ESTR를 제안하고, 이를 통해 미래의 연구에 대한 많은 가능성을 열어줍니다.

- **Technical Details**: SimC-ESTR 프레임워크는 이벤트 스트림에서 이벤트 피쳐를 추출하여 Q-former 모듈을 통해 토큰으로 변환합니다. 기존의 비전-기반 네트워크와 대형 언어 모델(LLM)을 통합하여, 메모리 메커니즘을 통해 시각적 특징을 증강하고, 비슷한 문자 오류를 교정하는 기능도 지원합니다. 이러한 접근 방식은 저조도, 복잡한 배경, 모션 블러와 같은 도전 과제를 해결하는 데 중점을 두고 개발되었습니다.

- **Performance Highlights**: 새롭게 제안된 EventSTR 데이터세트와 추가적인 두 가지 STR 시뮬레이션 데이터세트를 통해, 제안된 모델의 효과성을 입증하기 위한 광범위한 실험이 이루어졌습니다. SimC-ESTR는 기존의 알고리즘에 비해 더 높은 인식 정확도를 달성하며, 대량의 텍스트 인식 및 문서 처리 작업에서의 가능성이 확장될 수 있음을 보여주고 있습니다.



### Zero-shot Concept Bottleneck Models (https://arxiv.org/abs/2502.09018)
Comments:
          14 pages, 8 figures

- **What's New**: 이번 논문에서는 zero-shot concept bottleneck models (Z-CBMs)를 소개하며, 이는 학습 없는 상태에서 개념 지도와 라벨 예측을 수행할 수 있는 모델입니다. Z-CBMs는 대규모 개념 데이터베이스를 활용하여 다양한 도메인에서 입력을 설명합니다. 기존의 Concept Bottleneck Models (CBMs)과 달리, Z-CBMs는 타겟 데이터셋의 수집이나 학습 없이 이해 가능하고 개입 가능한 개념을 제공합니다.

- **Technical Details**: Z-CBMs는 concept retrieval과 concept regression 두 가지 모듈로 구성됩니다. Concept retrieval 모듈은 효율적인 교차 모달 검색 알고리즘을 통해 입력과 관련된 개념을 동적으로 찾아내며, concept regression 모듈은 중복된 개념을 피하고 상호 배타적인 개념을 선택하기 위해 희소 선형 회귀(sparse linear regression)를 사용합니다. 이러한 접근 방식은 타겟 데이터셋 없이도 개념과 라벨을 예측하는 데 필요한 모든 단계를 처리할 수 있습니다.

- **Performance Highlights**: Z-CBMs는 12개의 데이터셋에 대한 광범위한 실험을 통해 기존의 학습된 CBMs와 비슷하거나 더 나은 성능을 발휘함을 입증했습니다. 특히, Z-CBMs는 예측된 개념에 대한 인간의 개입을 통해 전반적인 신뢰성을 높일 수 있으며, 이는 개념 기반 예측의 실용성을 강조합니다. 최종적으로 Z-CBMs는 다양한 도메인에서 효과적으로 사용될 수 있는 가능성을 보여주고 있습니다.



### RoSTE: An Efficient Quantization-Aware Supervised Fine-Tuning Approach for Large Language Models (https://arxiv.org/abs/2502.09003)
Comments:
          18 pages, 6 figures

- **What's New**: 이번 연구에서는 양자화 인식 세밀 조정(quantization-aware fine-tuning) 방법을 최초로 제안하여, 세밀 조정과 양자화를 단일 훈련 단계로 통합하는 효율적인 방법을 찾았습니다. 이를 통해 가중치, 활성화 및 키-값 캐시의 낮은 비트 양자화를 효과적으로 실현할 수 있습니다. 특히, 새로운 로테이티드 스트레이트-스루-추정기(RoSTE) 알고리즘을 통해 활성화 이상치(outlier)를 줄이고 최적화된 양자화 구성을 식별하는 적응형 회전 전략을 결합하였습니다.

- **Technical Details**: RoSTE 알고리즘은 양자화 인식 세밀 조정(QA-SFT)과 회전 행렬의 적응형 선택을 결합하여 이루어지는 공동 훈련 방식을 채택합니다. 이 알고리즘은 최적화 문제를 이층(bilevel)으로 설정하여 상위 레벨에서는 가중치 행렬을 최적화하고 하위 레벨에서는 회전 행렬을 선택하는 방식으로 구성됩니다. 이론적인 분석을 통해 RoSTE 사용 시 예측 오류와 양자화 오류 간의 관계를 설명하며, 낮은 복잡성의 월시-하다마드 회전 방식을 채택하여 효율성을 극대화합니다.

- **Performance Highlights**: Pythia 및 Llama 모델에서의 실험 결과, RoSTE는 기존의 세밀 조정 후 양자화 방법에 비해 다양한 작업과 여러 LLM 아키텍처에서 일관되게 우수한 성능을 나타냈습니다. 특히, RoSTE는 4비트 양자화에서도 데이터 없는 우수한 성능을 보여주며, 메모리 소모를 줄이고 추론 대기 시간을 단축시키는 데 기여합니다. 이러한 성과는 RoSTE가 세밀 조정 과정에서 양자화 효과를 고안하여, 현업에서의 실용성을 높임을 입증합니다.



### PixLift: Accelerating Web Browsing via AI Upscaling (https://arxiv.org/abs/2502.08995)
Comments:
          9 pages, 2 figures

- **What's New**: PixLift는 사용자의 기기에서 이미지를 업스케일링하는 AI 모델을 활용하여 웹페이지 크기를 줄이는 혁신적인 솔루션입니다. 이 연구는 데이터 접근이 비싼 지역에서의 웹 접근성을 개선하고자 하는 목표 아래, 이미지의 전송 중 다운스케일링을 통해 대역폭을 절약하면서 데스크탑 및 모바일에서의 웹 사용자 경험에 미치는 영향을 분석합니다. PixLift는 사용자가 브라우저를 변경할 필요 없이 확장 프로그램 형태로 제공되며, 다양한 이미지 업스케일링 모델과 통합하여 효과적인 데이터 사용 절감이 가능함을 보여줍니다.

- **Technical Details**: PixLift는 Chromium 확장 프로그램으로 제공되며, 이로 인해 설치가 간단하고 대부분의 브라우저에서 빠르게 사용할 수 있습니다. 이 연구는 71,400개의 웹페이지를 분석하고, 이들 중 10%가 원격 이미지 다운스케일링 지원을 가지고 있음을 밝혔습니다. PixLift는 GPU와 CPU 리소스를 Trade-off하여 데이터 전송 중 이미지를 효과적으로 처리하며, 다양한 이미지 업스케일링 모델 중 'QuickSRNet Small 4X'가 가장 많은 효율성을 보이는 것으로 나타났습니다.

- **Performance Highlights**: PixLift는 웹페이지 로딩 시간을 평균 7초 단축하면서도 데이터 사용을 현저히 줄이는 결과를 가져왔습니다. 사용자 조사에 따르면, 이러한 대역폭 절약이 사용자 경험에 미치는 영향은 최소한이며, 이미지의 시각적 품질 또한 상당히 높게 유지되었습니다. 다수의 테스트를 통한 결과, PixLift는 저품질의 인터넷 환경에서도 효과적인 웹 통신을 지원하고, 더 많은 사용자가 정보에 접근할 수 있도록 돕는 효용성을 입증하였습니다.



### RLSA-PFL: Robust Lightweight Secure Aggregation with Model Inconsistency Detection in Privacy-Preserving Federated Learning (https://arxiv.org/abs/2502.08989)
Comments:
          16 pages, 10 Figures

- **What's New**: 이 논문에서는 경량 암호화 프리미티브(lightweight cryptographic primitives)를 이용한 효율적인 마스킹 기반 안전 집계(safe aggregation) 방식으로 개인 정보 보호에 대한 위험을 완화하는 방법을 제안합니다. 기존 방법보다 단일 설정 단계로 전체 Federated Learning(FL) 훈련 세션을 수행할 수 있어 통신 부하를 크게 줄일 수 있습니다. 또한 사용자 간의 상호작용을 제거하고 중간 서버(layer)와 경량 키 협상 방법을 사용하여 사용자 측의 오버헤드를 최소화합니다.

- **Technical Details**: 제안된 방식은 모델 일관성(model consistency) 공격을 감지할 수 있는 효율적인 메커니즘을 포함하고 있습니다. 중간 서버가 안전하게 마스킹된 모델을 수집하고, 메시지 인증 코드(message authentication code, MAC)를 사용하여 모든 사용자에게 전송되는 글로벌 모델의 일관성을 확인합니다. 이는 최소한의 통신 비용으로 이루어지며, 신뢰할 수 있는 중간 서버가 있는 경우 모델 불일치 공격을 성공적으로 감지할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방식은 기존 솔루션보다 통신 및 계산 오버헤드, 기능성, 보안 측면에서 우수한 성능을 보여줍니다. 최근 연구와 비교하여, 사용자 간 상호작용이 필요한 기존 방법보다 더 낮은 오버헤드를 가지면서도 사용자 dropout에 강한 저항성을 유지하는 것을 확인했습니다. 최종적으로, 전체 FL 훈련 회의 동안 단일 설정 단계로 사용자의 효율성을 극대화하는 것을 목표로 하고 있습니다.



### Neural Force Field: Learning Generalized Physical Representation from a Few Examples (https://arxiv.org/abs/2502.08987)
Comments:
          20 pages

- **What's New**: 본 논문에서는 Neural Force Field (NFF)라는 새로운 모델링 프레임워크를 제안합니다. 이 프레임워크는 Neural Ordinary Differential Equation (NODE)을 기반으로 하여, 최소한의 데이터를 통해 물리적 역학을 학습하고 일반화할 수 있는 강력한 대안을 제공합니다. 기존의 접근 방식은 고차원 잠재 공간에 의존하는 반면, NFF는 중력, 지지 및 충돌과 같은 근본적인 물리 개념을 해석 가능한 방식으로 캡처합니다.

- **Technical Details**: NFF는 외부 개입 및 객체 상호작용을 통해 동적 잠재력 장(force field)을 학습하는 신경망을 사용합니다. 예측된 힘(force)은 ODE 솔버를 통해 통합되어 속도 및 변위와 같은 명시적인 물리 변수를 계산하며, 이 결과는 확립된 물리 원칙과 일치하는 해석 가능한 결과를 생성합니다. 이 프레임워크는 적은 수의 훈련 예제에서 물리적 상호작용을 저차원 힘 장으로 표현함으로써 빠르게 학습할 수 있습니다.

- **Performance Highlights**: NFF는 I-PHYRE 및 N-body 문제와 같은 두 개의 도전적인 물리적 추론 벤치마크에서 검증되었습니다. 실험 결과, NFF는 물리적 상호작용을 힘 장으로 추상화함으로써 효과적으로 역학을 학습할 뿐만 아니라, 이전 시나리오 및 교차 시나리오 설정에서도 강력한 일반화 성능을 달성했습니다. 이러한 물리 기반 표현은 목표 지향적 작업에서의 효과적인 전방 및 후방 계획을 가능하게 하여 Interaction Network (IN) 및 transformer 기반 방법들과 비교하여 일관되게 더 나은 성능을 나타냈습니다.



### Few is More: Task-Efficient Skill-Discovery for Multi-Task Offline Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2502.08985)
- **What's New**: 이번 논문에서는 기존의 문제를 해결하기 위해 'Skill-Discovery Conservative Q-Learning'(SD-CQL)이라는 다중 작업 오프라인 MARL 알고리즘을 제안합니다. SD-CQL은 관찰을 재구성하여 스킬을 발견하며, 이를 통해 고정 및 가변 행동을 개별적으로 평가합니다. 이 과정에서 기존의 방법들과 달리, 정책을 다시 훈련할 필요 없이 여러 작업에 대한 범용성을 학습할 수 있게 합니다.

- **Technical Details**: SD-CQL 알고리즘은 행동 규제 보수적 Q-learning을 사용하여 발견된 각 스킬에 최적의 행동을 실행합니다. 이러한 접근 방식은 로컬-글로벌 정렬(local-global alignment)을 필요로 하지 않으며, 제한된 소스 작업으로부터 강력한 다중 작업 일반화를 가능하게 합니다. 논문에서는 StarCraftII에서의 실험을 통해 SD-CQL의 성능을 입증하며, 이를 바탕으로 고급 오프라인 MARL 알고리즘에서의 사용 가능성을 강조합니다.

- **Performance Highlights**: SD-CQL은 14개의 작업 세트 중 10개에서 최고의 성능을 달성하였고, 개별 작업 세트에서는 최대 65%의 개선을 보였습니다. 나머지 4개의 작업 세트에서도 최고의 기준 알고리즘과 거의 4% 이내의 차이를 나타내어 매우 우수한 성능을 입증합니다. 이 결과는 SD-CQL이 높은 작업 효율성을 가지며 많은 잠재적 작업을 다룰 수 있음을 나타냅니다.



### Tuning-Free Personalized Alignment via Trial-Error-Explain In-Context Learning (https://arxiv.org/abs/2502.08972)
Comments:
          NAACL 2025 Findings

- **What's New**: 이 연구에서는 Trial-Error-Explain In-Context Learning (TICL)이라는 튜닝(예를 들어, fine-tuning) 없는 방법을 제안합니다. TICL은 사용자의 스타일에 맞춘 개인화된 텍스트 생성을 위해 10개 이하의 예제만 사용하여 언어 모델을 개인화하는 새로운 접근법을 제공합니다. TICL은 모델이 생성한 부정 샘플과 설명을 통해 점진적으로 In-Context Learning 프롬프트를 확장합니다.

- **Technical Details**: TICL은 각 단계에서 Trial-and-Error Fine-tuning (TEFT) 방식과 튜닝이 필요 없는 프롬프트 확장 방법을 결합하여 동작합니다. 기존의 Supervised Fine-tuning (SFT) 대신 In-Context Learning (ICL)를 사용하는 과정으로, 예측된 텍스트와 사용자의 실제 텍스트 사이의 차이를 분석하여 생성된 텍스트에 부정적 샘플로 라벨을 붙입니다. 이를 통해 개인화된 텍스트 생성에서 성능을 극대화합니다.

- **Performance Highlights**: TICL은 GPT-4o와 Claude 3 Sonnet을 사용하여 다양한 개인화된 작성 데이터셋에서 실험을 수행한 결과, 경쟁 방식보다 높은 승률(87%)을 기록했습니다. 특히 TICL의 각 절차는 성능 향상에 기여하며, 설명 프로세스는 77%의 성능 향상에 기여하는 것으로 나타났습니다. TICL은 기존의 zero-shot 출력에서 관찰된 구조적 및 형식적인 구문에 대한 편향을 극복할 수 있는 가능성을 보여줍니다.



### SkyRover: A Modular Simulator for Cross-Domain Pathfinding (https://arxiv.org/abs/2502.08969)
Comments:
          9 pages

- **What's New**: SkyRover는 UAV(무인 항공기)와 AGV(자동화 유도차량) 간의 협력을 지원하는 최초의 모듈식 시뮬레이터로, 이 분야에서의 연구와 알고리즘 개발을 위한 통합 툴킷을 제공합니다. 기존 시뮬레이터들은 대개 특정 도메인에 국한되어 있었지만, SkyRover는 다양한 3D 환경에서 리얼리스틱한 동작을 구현할 수 있게 해줍니다. 이로 인해 연구자들은 UAV-AGV 간의 상호 작용을 통한 물류 및 자동화의 효율성을 극대화할 수 있습니다.

- **Technical Details**: SkyRover는 시뮬레이션 환경을 구축하는 데 있어 3D 그리드와 정밀한 물리 모델을 활용하며, 이러한 환경은 Gazebo와 통합되어 있습니다. 시스템의 주요 모듈로는 ‘Sim World Zoo’, ‘3D Grid Generator’, ‘Unified Algorithm Wrapper’, ‘Plan Executor’, ‘System Interface’가 포함되어, 동시 다발적인 실험과 알고리즘 테스트를 지원합니다. 특히, MAPF(Multi-Agent Path Finding) 알고리즘에 대한 일관된 인터페이스를 제공함으로써 다양한 연구 방법론을 적용할 수 있는 유연성을 보장합니다.

- **Performance Highlights**: SkyRover는 AGV와 UAV의 협력을 통한 경로 찾기(pathfinding) 실험에서 뛰어난 성과를 나타내며, 복잡한 로봇 상호작용을 효과적으로 시뮬레이션할 수 있습니다. 다양한 시나리오에서의 실험 결과는 시뮬레이터의 3D 매핑 및 시각화 기능이 우수함을 보여줍니다. 특히 재고 스캔 작업과 공중 화물 이송 작업을 통해 실시간으로 UAV와 AGV 간의 원활한 협조를 시연하며, 효율적인 물류 운영의 가능성을 강조합니다.



### RTBAS: Defending LLM Agents Against Prompt Injection and Privacy Leakag (https://arxiv.org/abs/2502.08966)
- **What's New**: 이번 연구에서는 Robust TBAS (RTBAS)라는 새로운 시스템을 도입하여 Tool-Based Agent Systems (TBAS)의 보안을 강화했습니다. RTBAS는 정보 흐름 제어(Information Flow Control)를 적용하여 기밀성을 유지하고 사용자 승인을 최소화하는 자동 도구 호출 감지 및 실행 기능을 제공합니다. 기존의 OpenAI GPTs와 같은 솔루션과 달리, RTBAS는 사용자가 수동으로 도구 호출을 승인해야 하는 부담을 줄입니다.

- **Technical Details**: RTBAS는 두 가지 새로운 의존성 스크리너를 도입합니다: LM-as-a-judge와 Attention-based saliency를 사용하여 중요 정보를 추출하고 불필요한 데이터의 누출을 방지합니다. 이 시스템은 동적 염료 추적(Dynamic Taint Tracking)을 통해 보안 메타데이터와 변수를 연결하고, 의존성에 따른 보안 정책을 실행합니다. RTBAS는 TBAS 시스템의 고유한 도전 과제를 해결하기 위해 설계되었습니다.

- **Performance Highlights**: AgentDojo에서 실시한 실험 결과, RTBAS는 모든 타겟 공격을 예방하면서도 2% 미만의 작업 유용성 손실을 보여줍니다. 게다가 RTBAS는 미세한 개인 정보 유출도 감지하는 데 뛰어난 성과를 보이며, 100% 사용자 승인을 요구하는 기존의 솔루션보다 뛰어난 성능을 발휘하고 있습니다.



### Biologically Plausible Brain Graph Transformer (https://arxiv.org/abs/2502.08958)
Comments:
          27pages, 16figures, published as a conference paper at ICLR 2025

- **What's New**: 이번 연구에서는 Brain Graph의 생물학적 적합성을 높이기 위해 Biologically Plausible Brain Graph Transformer(BioBGT)를 제안합니다. BioBGT는 Brain Graph의 작은 세계 구조를 인코딩하고, 노드의 중요성을 전달하는 네트워크 얽힘 기법을 도입하여 뇌의 생물학적 특성을 반영합니다. 또한, 기능적 모듈을 고려한 자기 주의(self-attention) 기법을 통해 뇌 그래프의 기능적 분리 및 통합 특성을 보존합니다.

- **Technical Details**: 제안된 BioBGT는 두 가지 주요 구성 요소를 포함합니다. 첫째, 네트워크 얽힘 기반 노드 중요성 인코딩 기법은 뇌 그래프의 구조적 중요성을 반영하여 정보를 전파하는 과정에서 노드의 역할을 측정합니다. 둘째, 기능 모듈 인식을 통해 각 노드의 유사성을 정밀하게 조정하여 기능적 정합성을 유지하며, 커뮤니티 대비 전략을 통해 정밀한 기능적 모듈 노드 표현을 생성합니다.

- **Performance Highlights**: 실험 결과는 BioBGT가 기존의 최첨단 모델보다 우수한 성능을 보여준다는 것을 입증합니다. 특히 뇌 질환 탐지와 같은 다양한 뇌 그래프 분석 작업에 있어 생물학적 적합성이 강화된 표현이 효과적임을 입증하였습니다. 이 연구는 뇌 그래프의 생물학적 특성을 확보한 모델 설계의 중요성을 강조합니다.



### The Stochastic Parrot on LLM's Shoulder: A Summative Assessment of Physical Concept Understanding (https://arxiv.org/abs/2502.08946)
Comments:
          NAACL 2025 Main Conference. First 5 authors contributed equally. Project page: this https URL

- **What's New**: 본 논문은 LLMs(대형 언어 모델)의 진정한 이해 여부에 대해 정량적인 실험을 통해 검증하고자 합니다. 우리는 특히 Stochastic Parrot 현상, 즉 LLM이 단순히 연관 패턴을 바탕으로 반복하는지 여부를 평가하는 PhysCo라는 새로운 과제를 제안합니다. 이 과제는 물리적 개념 이해도를 측정하기 위해 설계되었습니다. 고민이 많았던 메모리 문제를 해결하기 위해 그리드 형식의 입력을 사용하여 다양한 이해 수준을 표현하고 있습니다.

- **Technical Details**: PhysiCo는 저급 이해(subtask)와 고급 이해(high-level understanding)라는 두 가지 하위 과제를 포함합니다. 저급 이해는 LLM의 기억 능력을 측정하는 자연어 형식 질문으로 구성되어 있으며, 고급 이해 과제는 추상 표현을 기반으로 상대적으로 심화된 이해를 평가하는 숙제입니다. 우리는 LLMs가 저급 과제에서 높은 정확도를 보이지만, 고급 과제에서는 인간에 비해 40% 가량 성능이 떨어진다는 두 가지 주요 결론을 도출했습니다.

- **Performance Highlights**: 실험 결과, 최신 LLMs는 저급 이해 과제에서 95% 이상의 정확도를 기록하였으나, 고급 이해 과제에서는 인간에 비해 평균 약 40% 낮은 정확도를 보였습니다. 이는 LLM이 진정한 개념 이해 능력에서는 한계를 나타내며, 새로운 그리드 형식이 아닌 그 자체의 고차원적 이해의 어려움이 원인임을 시사합니다. 본 연구는 LLM의 이해력을 측정하는 방법론적 기틀을 확립하고, LLM과 인간 간의 성능 격차를 명확히 보여줍니다.



### Beyond the Singular: The Essential Role of Multiple Generations in Effective Benchmark Evaluation and Analysis (https://arxiv.org/abs/2502.08943)
Comments:
          10 pages, 1 table, 4 Figures

- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 성능 평가 방법에 대한 새로운 접근 방식을 제시하고 있습니다. 기존의 평가 방식은 LLM의 본질적인 무작위성을 간과하는 경우가 많았으나, 본 연구에서는 계층적 통계 모델을 통해 이를 통합하여 보다 신뢰성이 높은 벤치마크(score)를 제공하고자 합니다. 이를 통해 개별 프롬프트(prompts)의 난이도를 정밀하게 측정하고 오류 감지 및 품질 관리를 위한 데이터 맵을 생성했습니다.

- **Technical Details**: 저자들은 LLM의 응답 생성 방식에 대해 deterministic한 greedy decoding과 stochastic한 random sampling 방법을 비교했습니다. 후자는 원래의 확률 분포에 따라 매 단계에서 토큰을 무작위로 샘플링하여 비결정적 출력을 생성하게 됩니다. 연구에서는 각 프롬프트에 대해 k개의 무작위 생성(random generations)을 통해 벤치마크 과정을 계층적 모델(hierarchical model)로 보았으며, 이를 통해 불확실성을 줄이고 정확한 평가를 할 수 있음을 보였습니다.

- **Performance Highlights**: 여러 번의 생성을 통해 벤치마크 점수의 정확도가 향상되며, variance가 감소하고 정확성의 세분화된 난이도 점수 ℙ(correct)도 도입되었습니다. 이는 프롬프트들의 난이도를 비교하고 잘못 표기되거나 모호한 프롬프트를 효과적으로 감지할 수 있는 도구로서의 가능성을 보여줍니다. 본 연구는 LLM 벤치마크 개발 과정에서의 방법론적 틀을 개선하고 실제 적용 가능성을 높이는 데 기여할 것입니다.



### Language in the Flow of Time: Time-Series-Paired Texts Weaved into a Unified Temporal Narrativ (https://arxiv.org/abs/2502.08942)
Comments:
          Preprint, 37 pages

- **What's New**: 이번 연구에서는 기존의 수치 데이터 중심의 시계열 모델 연구에서 간과된 멀티모달 시계열, 특히 텍스트 정보를 효과적으로 통합하는 방법을 제시합니다. 저자들은 시간 시리즈와 텍스트가 짝지어진 데이터가 주기적 특성을 나타내며, 이를 통해 시계열 모델링을 향상시킬 수 있는 잠재력을 강조합니다. 새로운 프레임워크인 Texts as Time Series (TaTS)가 이러한 통찰을 기반으로 제안되었습니다.

- **Technical Details**: TaTS는 시간 시리즈 데이터에서 쌍(pair)으로 된 텍스트 정보를 보조 변수로 고려하여 기존의 수치 중심 모델과 통합함으로써 시계열 데이터의 예측 성능을 향상시킵니다. 이 프레임워크는 쌍텍스트의 잠재적 표현을 하향 차원으로 변환하고, 이를 기존 시계열 모델에 입력하여 숫자와 텍스트의 시간적 동역학을 모두 캡처합니다. 이는 멀티모달 시계열 분석을 위해 필요한 다양한 모델과 호환될 수 있습니다.

- **Performance Highlights**: 다양한 벤치마크 데이터 세트와 기존의 여러 시계열 모델을 통한 실험 결과, TaTS는 예측 및 보간(task) 작업에서 최첨단 성능을 달성했습니다. 이를 통해 TaTS는 시계열 모델의 아키텍처를 수정하지 않고도 성능을 향상시킬 수 있음을 확인했습니다. 새로운 방식이 기존 접근방식에 비해 어떻게 뚜렷한 금융성과를 이끌어내는지를 강조하고 있습니다.



### Analysis of Off-Policy $n$-Step TD-Learning with Linear Function Approximation (https://arxiv.org/abs/2502.08941)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2402.15781

- **What's New**: 이 논문은 "deadly triad" 시나리오에서의 다단계 시간차 학습 알고리즘에 대한 분석을 제공합니다. 특히, n-단계 TD-learning 알고리즘이 샘플링 수평(n)이 충분히 증가함에 따라 수렴함을 증명합니다. 이 논문은 두 부분으로 나뉘며 첫 번째 부분에서는 모델 기반 결정론적 알고리즘의 기본 속성을 포괄적으로 살펴봅니다.

- **Technical Details**: TD-learning은 강화를 통한 RL의 핵심 알고리즘으로, 특히 정책 평가에 중요합니다. 소개된 방법 중 하나인 gradient temporal-difference learning (GTD)은 deadly triad 문제를 해결하지만, 환경에 대한 제한적인 가정이 필요합니다. 논문에서는 모델 기반 결정론적 알고리즘과의 관계를 통해 n-단계 TD 방법의 수렴 조건과 효과적 해결 방안을 제시합니다.

- **Performance Highlights**: 이 논문은 n-단계 TD-learning 알고리즘이 deadly triad의 문제를 효과적으로 완화하는 방법을 제시하며, 샘플링 수평(n)이 충분히 클 때 수렴성을 보장합니다. 이 분석을 통해 미래의 모델 자유 강화 학습 개발에 있어 중요한 통찰을 제공하며, 모델 기반 알고리즘의 ODE 대응 분석 또한 포함되어 있습니다.



### TokenSynth: A Token-based Neural Synthesizer for Instrument Cloning and Text-to-Instrumen (https://arxiv.org/abs/2502.08939)
Comments:
          5 pages, 1 figure, to be published in ICASSP 2025

- **What's New**: 이 논문은 TokenSynth라는 새로운 신경 합성기를 소개합니다. 이 모델은 MIDI 토큰 및 CLAP 임베딩을 활용하여 오디오 토큰을 생성하는 디코더 전용 트랜스포머를 사용합니다. TokenSynth는 악기 복제, 텍스트 기반 악기 합성, 그리고 텍스트 지향의 음색 조작을 수행할 수 있으며, 별도의 미세 조정 없이도 이러한 기능을 제공합니다.

- **Technical Details**: TokenSynth는 오디오 코덱 언어 모델을 바탕으로 하여, 오디오 토큰을 오토 회귀적으로 생성합니다. 모델은 미리 훈련된 CLAP 모델과 신경 오디오 코덱 모델을 이용하여 음색을 조절하고 오디오 토큰을 변환합니다. 이 합성기는 9.53M 샘플로 구성된 데이터셋을 사용하여 훈련되었으며, CLAP 인코더를 통해 텍스트와 오디오의 크로스 모달 표현 학습을 통해 악기 합성 및 음색 조작을 가능하게 합니다.

- **Performance Highlights**: 논문에서는 TokenSynth의 오디오 품질, 음색 유사성, 합성 정확도를 객관적으로 평가합니다. 모델은 First-Note Guidance라는 새로운 기술을 도입하여 첫 음의 발생 시간에 클래스프리 가이드를 적용하여 합성 과정을 안정화합니다. 이러한 평가 결과는 TokenSynth가 진보된 신경 오디오 코덱과 트랜스포머를 활용하여 강력하고 다재다능한 신경 합성기를 생성할 수 있는 가능성을 보여줍니다.



### Escaping Collapse: The Strength of Weak Data for Large Language Model Training (https://arxiv.org/abs/2502.08924)
- **What's New**: 이번 논문에서는 합성 데이터(synthetic data)가 대형 언어 모델(LLM)의 훈련에 어떻게 기여하는지와 이러한 데이터의 큐레이션(curation) 없이는 성능이 정체되거나 심지어 감소할 수 있다는 점에 대해 논의합니다. 제안된 이론적 프레임워크를 통해 LLM 성능 향상을 위한 큐레이션의 필요성을 정량적으로 분석하였습니다. 많은 비합성 데이터의 품질이 저조하더라도 최적의 LLM으로 수렴하는 훈련 절차를 설명합니다.

- **Technical Details**: 이 논문은 부스팅(boosting)이라는 고전적인 기계 학습 기술에 착안하여, 약한 학습 알고리즘(weak learning algorithm)을 활용해 우수한 분류기를 생성하는 방식을 통해 분석을 진행하였습니다. 제안된 훈련 절차는 최근에 발표된 여러 합성 데이터 훈련 방법들을 포함하며, 이러한 방법들이 어떻게 성공적인지를 설명하는 데 기여합니다. 특히, 어려운 예제에 동적으로 라벨링 자원을 집중시키는 방식을 통해 성능을 향상시킬 수 있음을 보여줍니다.

- **Performance Highlights**: 실험 결과, 합성 데이터에서 훈련된 LLM이 향상된 성능을 보이는 것으로 나타났습니다. 훈련 절차가 약한 학습기의 노력을 집중하는 방식과 유사하게 작동하여, 도전적인 예에 대한 집중이 성능을 개선하는 데 기여합니다. 이러한 결과는 기존의 방법들이 성공하는 이유를 명확히 하며, 향후 개선 작업에 대한 기회를 제시합니다.



### CopySpec: Accelerating LLMs with Speculative Copy-and-Paste Without Compromising Quality (https://arxiv.org/abs/2502.08923)
Comments:
          33 pages, 18 figures, 19 tables

- **What's New**: 본 논문에서는 LLMs의 비효율성을 해결하기 위한 혁신적인 기술인 CopySpec을 소개합니다. CopySpec은 모델의 채팅 기록에서 반복되는 시퀀스를 식별하고, 이러한 패턴을 활용하여 후속 토큰을 원활하게 복사할 수 있게 하여 GPU 메모리를 추가로 요구하지 않습니다. 이를 통해 다양한 LLM과 데이터셋에서 значительные 속도 향상을 달성했습니다.

- **Technical Details**: CopySpec은 학습된 복사 메커니즘을 활용하여 입력의 특정 토큰 패턴을 감지하고 이를 재사용합니다. 이 과정은 전체 LLM을 통해 토큰을 한 번에 생성함으로써 반복적이거나 예측 가능한 출력에서의 계산 부담을 줄이는 데 기여합니다. 기술적으로 Roll Hash 메커니즘을 사용하여 계산 오버헤드를 최소화하면서 더 큰 토큰 블록을 추측할 수 있습니다.

- **Performance Highlights**: 다양한 벤치마크에서 CopySpec은 측정 가능한 성능 향상을 보여주며, MT-Redundant의 특정 카테고리에서는 3.08배, Speculative Decoding 보다 평균 49%의 추가 속도 향상을 기록했습니다. 이러한 결과는 CopySpec이 LLM의 효율성을 크게 개선할 수 있는 잠재력을 지니고 있다는 것을 보여줍니다.



### Exploring Emotion-Sensitive LLM-Based Conversational AI (https://arxiv.org/abs/2502.08920)
Comments:
          7 pages, 2 figures, 1 table

- **What's New**: 본 연구는 고객 서비스에서의 감정 민감성이 대화형 AI 챗봇의 신뢰성과 역량 인식에 미치는 영향을 조사합니다. 기계는 감정을 이해하고 응답하는 성능을 비교하기 위해, 감정 민감한 챗봇과 비감정 민감한 챗봇을 30명의 참가자와 함께 테스트했습니다. 연구 결과, 감정 민감한 챗봇이 문제 해결률에 있어서는 동일했지만, 고객들은 그 챗봇에 대해 더 높은 신뢰도와 역량을 느꼈습니다.

- **Technical Details**: 이 실험은 ChatGPT-3.5를 기반으로 하여 감정 인식 및 응답을 탐구하였습니다. 사용자 입력에 대한 감정을 인식하고 해당 감정에 맞는 응답을 생성하기 위해 감정 분석 기법인 VADER를 사용했습니다. 감정 민감한 시스템은 사용자 감정에 맞춘 특정 프롬프트에 따라 응답을 제공했고, 비감정 민감한 시스템은 문제 해결에 집중하도록 조정되었습니다.

- **Performance Highlights**: 최종적으로 30명의 참가자 중 67%가 챗봇의 문제 해결 능력에 만족한다고 응답했으며, 감정의 변화도 관찰되었습니다. 감정 민감하거나 비감정 민감한 챗봇 모두 사용 후 부정적 감정이 유의미하게 감소했습니다. 이러한 결과는 감정 민감성이 사용자의 경험에 긍정적인 영향을 미친다는 것을 강조합니다.



### PathFinder: A Multi-Modal Multi-Agent System for Medical Diagnostic Decision-Making Applied to Histopathology (https://arxiv.org/abs/2502.08916)
- **What's New**: 이번 논문에서는 PathFinder라는 다중 모달 및 다중 에이전트 시스템을 제안합니다. 이 시스템은 4개의 AI 에이전트(Triage, Navigation, Description, Diagnosis)를 통합하여 병리 전문가의 의사 결정 과정을 모방합니다. 특히, 각 에이전트는 WSIs(Whole Slide Images)를 효과적으로 탐색하고 자연어 설명을 제공하여 포괄적인 진단을 수행합니다. 이러한 접근 방식은 기존의 전통적인 AI 방법이 가지는 한계를 극복하는 데 목적을 두고 있습니다.

- **Technical Details**: PathFinder는 Triage Agent가 WSI를 양성 또는 위험으로 분류한 후 위험으로 판단될 경우 Navigation 및 Description Agents가 중요 영역을 반복적으로 조사하여 중요도 맵과 설명을 생성합니다. 이 과정에서 각 에이전트는 긴밀하게 협력하여 진단 정보를 수집하고, Diagnosis Agent가 최종 진단 평가를 수행합니다. 실험 결과, PathFinder는 기존의 최고 성능을 보여준 방법들보다 8% 높은 정확도로 피부 흑색종 진단을 가능하게 하였습니다.

- **Performance Highlights**: PathFinder는 병리학자와의 질적 분석에서 Description Agent의 출력 품질이 GPT-4o와 비교될 만큼 우수하다는 것을 보여줍니다. 이 시스템은 또한 병리학자의 평균 성능을 9% 초과하여 새로운 기록을 세우며, 병리 진단에서 효율적이고 정확하며 해석 가능한 AI 지원 진단을 실현할 능력을 가지고 있습니다. 논문에서는 데이터, 코드 및 모델을 제공하며, PathFinder의 기초와 성과를 자세히 설명합니다.



### Diffusion Models Through a Global Lens: Are They Culturally Inclusive? (https://arxiv.org/abs/2502.08914)
Comments:
          17 pages, 17 figures, 3 tables

- **What's New**: 최근 텍스트에서 이미지를 생성하는 diffusion 모델은 문화적 뉘앙스를 정확하게 표현하는 데 한계가 있음을 보여줍니다. 이 연구에서는 CultDiff 벤치마크를 도입하여 최신 diffusion 모델들이 10개국의 문화적으로 특화된 이미지를 생성할 수 있는 능력을 평가합니다. 결과적으로, 이러한 모델들이 고해상도의 이미지 생성에서 문화적 특성을 제대로 반영하지 못하는 경우가 많음을 발견했습니다.

- **Technical Details**: 연구에서는 T2I (Text-to-Image) diffusion 모델의 성능을 평가하기 위해 문화에 따라 분류된 새로운 데이터셋인 CultDiff를 개발했습니다. 이 데이터셋은 건축물, 의상, 음식과 같은 다양한 문화적 아이템을 아우르며, 자원 수준이 높은 문화와 낮은 문화의 이미지를 생성하는 모델의 능력을 분석합니다. 또한, 인간 평가를 통해 개발된 neural 기반의 이미지 유사성 메트릭인 CultDiff-S를 도입하여 생성 이미지의 문화적 관련성을 예측합니다.

- **Performance Highlights**: 연구 결과, 최신 diffusion 모델들이 특정 문화에서 필요한 대표적인 개념이나 예술품을 생성하는 데 실패하고 있는 것을 알 수 있었습니다. 특히 자원이 부족한 문화에 대한 표현이 미비하여 이들에 대한 공정한 representation이 이루어지지 않고 있음을 강조합니다. 이러한 결과는 생성 AI 시스템의 더 나은 포용성과 공정한 데이터셋 표현의 필요성을 제기하고 있습니다.



### Towards Automated Fact-Checking of Real-World Claims: Exploring Task Formulation and Assessment with LLMs (https://arxiv.org/abs/2502.08909)
- **What's New**: 이 연구에서는 대규모 언어 모델(LLM)을 활용한 자동 팩트 체크(Automated Fact-Checking, AFC) 시스템의 효용성을 검토합니다. 연구진은 3종의 레이블링 체계(2진, 3클래스, 5클래스)를 통해 다양한 사이즈의 Llama-3 모델(3B, 8B, 70B)을 평가하였고, 각 모델의 주장 분석, 진실성 예측 및 자세한 설명 생성을 위한 프레임워크를 제안합니다. 또한, 증거 통합이 모델 성능에 끼치는 영향을 강조하며, LLM을 활용한 AFC의 가능성을 보여줍니다.

- **Technical Details**: 연구에서 사용된 데이터셋은 PolitiFact에서 2007년부터 2024년까지 수집된 17,856개의 주장으로 구성되어 있습니다. 이 데이터는 제한된 웹 검색을 통해 수집된 증거와 함께 모델 성능을 평가하는 데 사용되었으며, 증거 통합이 모든 모델에서 성능을 향상시키는 데 기여함을 보여줍니다. TIGERScore라는 레퍼런스 프리(reference-free) 평가 지표를 활용하여 결과를 분석하였고, 더 큰 LLM이 작은 모델보다 항상 더 높은 분류 정확도 및 정당화 품질을 보였습니다.

- **Performance Highlights**: 연구 결과, 큰 LLM은 무작위 원샷(scenario) 상황에서 미세 조정을 한 LSLM(Small Language Models)과 유사한 성능을 보인 반면, LLM은 한계를 초과하여 지속적으로 높은 성능을 발휘하는 것으로 나타났습니다. 특히, 큰 모델이 보다 복잡한 레이블의 분별력에서 더 강력한 성능을 보여주었지만, 미세 조정이 필요없는 구성으로도 적절한 성능을 나타내는 것을 확인하였습니다. 이러한 결과는 LLM을 이용한 혹은 다른 데이터 구성 요소에서의 성능 향상을 위한 추가 연구 필요성을 강조합니다.



### 3D-Grounded Vision-Language Framework for Robotic Task Planning: Automated Prompt Synthesis and Supervised Reasoning (https://arxiv.org/abs/2502.08903)
- **What's New**: 이번 논문에서는 3D 장면 인식 및 로봇 작업 수행의 신뢰성을 향상시키기 위해 새로운 프레임워크를 제안합니다. 이 프레임워크는 2D 이미지와 텍스트를 3D 공간 정보로 변환하는 2D 프롬프트 합성 모듈을 통합하며, 소규모 언어 모델(SLM)을 통해 VLM(비전-언어 모델)의 출력을 감독합니다. 제안된 방법은 로봇이 복잡한 환경에서의 작업 요청을 더 잘 수행할 수 있도록 지원하며, 실험 결과 96.0%의 작업 성공률을 달성했습니다.

- **Technical Details**: 제안된 프레임워크는 3개의 주요 모듈로 구성됩니다. 첫째, 2D 프롬프트 합성 모듈은 깊이 정보를 2D 이미지에 융합하여 정확한 3D 공간 좌표를 생성합니다. 둘째, 고정된 VLM은 시각적 및 텍스트적 추론 능력을 활용하여 기하학적 제약과 의미적 관계를 유추합니다. 마지막으로, SLM 모듈은 작업 계획의 일관성을 검증하고 동적 적응성을 보장하기 위해 세밀하게 조정된 모델을 사용합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 3D 인식 정확성을 31.93%, 위치 정밀성을 46.40%, 작업 수행 성공률을 58.10% 향상시키는 성과를 거두었습니다. 이는 기존의 최첨단 3D-MMLM 모델과 비교하여 뛰어난 성능을 보인 것입니다. 또한, ablation 연구를 통해 2D 프롬프트 합성 모듈과 출력 감독 모듈의 중요성이 입증되었습니다.



### Learning in Strategic Queuing Systems with Small Buffers (https://arxiv.org/abs/2502.08898)
- **What's New**: 이 논문에서는 Networking에서 라우터를 단순한 학습 알고리즘을 사용하는 에이전트로 모델링합니다. Gaitonde와 Tardos의 기존 연구를 기반으로, 각 서버에 아주 작은 버퍼를 추가한 모델을 제시하여 시스템 안정성을 향상시킬 수 있음을 보여줍니다. 또한, 제안된 모델에서는 오래된 패킷을 위한 타임스탬프나 우선 순위 없이도 높은 안정성을 달성할 수 있습니다.

- **Technical Details**: 논문에서는 여러 큐가 서버를 위해 경쟁하는 구조에서, 각 서버에 단일 패킷을 저장할 수 있는 작은 버퍼를 추가하여 시스템 용량을 증가시키는 방법을 제시합니다. 기존의 모델에서는 서버가 전혀 버퍼 없이 작동하였으나, 우리의 모델은 각 서버가 한 개의 패킷을 보유할 수 있으므로, 시스템 안정성 조건이 완화됩니다. 이러한 상황에서, 학습 알고리즘은 낮은 용량의 서버를 활용하여 패킷 분배를 최적화할 수 있습니다.

- **Performance Highlights**: 주요 결과는 여러 큐가 서비스 경쟁을 수행하면서도 소량의 서버 용량 증가로 시스템을 안정화할 수 있음을 보여줍니다. 특히, 각 큐가 패킷 도착률에 따라 다르게 설정되더라도, 일반적으로 3배의 서버 용량 증가는 중앙 집중식 조정이 없더라도 안정성을 유지하는 데 충분합니다. 이는 학습 알고리즘을 통해 안정성이 저해되지 않도록 함과 동시에 향후 효율적인 패킷 배분을 가능하게 합니다.



### Communication is All You Need: Persuasion Dataset Construction via Multi-LLM Communication (https://arxiv.org/abs/2502.08896)
Comments:
          Accepted to NAACL 2025 Main Conference

- **What's New**: 이 논문은 다수의 LLM (Large Language Models)을 활용한 설득적 대화 생성 프레임워크를 제안합니다. 이 프레임워크는 고품질의 다양한 언어 콘텐츠를 자동으로 생성하고, 최소한의 인간 개입으로 설득 기술을 향상시킬 수 있는 가능성을 보여줍니다. 특히, 사회적 금기를 포함한 복잡한 시나리오에서도 자연스러운 언어 흐름과 전략적인 설득을 활용하는 능력이 강조됩니다.

- **Technical Details**: 제안된 프레임워크는 6개의 언어 에이전트 그룹을 포함하여 다중 에이전트 데이터 생성 및 주석화 과정을 수행합니다. 에이전트들은 각기 다른 역할을 맡아서 설득적 대화 생성을 효율적으로 처리하며, GPT-3.5와 GPT-4 모델을 사용해 대화의 질을 높입니다. 이론적인 대화 흐름과 다양한 설득 전략을 포함하는 대화 생성이 가능하며, 특정 요청에 따른 동적이고 구조화된 상호작용이 이루어집니다.

- **Performance Highlights**: 우리의 실험 결과, 프레임워크는 자연스러움, 언어 다양성, 논리적 일관성을 유지하며 모두 고품질의 대화를 지속적으로 생성했습니다. 특히, 설득력 평가와 관련하여 생성된 데이터는 전문가들의 판단과 높은 일치를 보였으며, 이는 단계 변화 분석의 유용성을 강조합니다. 다자간 대화를 포함한 다양한 구성에서도 프레임워크의 적응성과 일반화 능력이 검증되었습니다.



### Generative AI for Internet of Things Security: Challenges and Opportunities (https://arxiv.org/abs/2502.08886)
- **What's New**: 이번 논문에서는 Generative AI (GenAI)와 Internet of Things (IoT) 보안의 통합 응용에 대해 다룹니다. GenAI의 최신 기술이 IoT 보안 분야에서 어떻게 사용될 수 있는지에 대한 상태를 매핑하며, 보안 조치를 강화하는 가능성도 탐색합니다. 또한 이 연구는 MITRE Mitigations를 통해 IoT 보안의 현재 상태와 발생하는 문제들을 진단하고, GenAI의 효과성을 평가합니다.

- **Technical Details**: GenAI는 새로운 콘텐츠 생성을 위한 AI 기술로, LLMs(대형 언어 모델)를 포함하여 다양한 보안 응용의 가능성을 보여줍니다. IoT 생태계 내의 보안 위협을 분석하고, GenAI가 보안 솔루션을 어떻게 개선할 수 있는지에 대한 사례 연구를 통해 심도 있는 통찰을 제공합니다. 이 논문은 IoT 시스템의 네 가지 핵심 레이어를 설명하며, 각 레이어에서 발생할 수 있는 취약점과 그 해결 방법을 다룹니다.

- **Performance Highlights**: GenAI와 LLMs의 발전은 IoT 보안 분야에서의 실질적인 응용 가능성을 바탕으로 혁신적인 데이터 분석 및 위협 감지 방법을 제공합니다. 특히 GenAI는 다양한 데이터를 생성하고 이를 기반으로 복잡한 패턴을 학습함으로써 사이버 보안의 새로운 패러다임을 제공합니다. 이 연구는 IoT 보안 향상을 위한 GenAI 응용의 진전과 미래 방향성을 제시하며, 연구자들에게 유용한 기초 자료로 활용될 것입니다.



### ShapeLib: designing a library of procedural 3D shape abstractions with Large Language Models (https://arxiv.org/abs/2502.08884)
- **What's New**: 이 논문은 ShapeLib을 제안하며, 이는 최신 대형 언어 모델(LLM)의 선험적 지식을 활용하여 3D 형태 추상화 함수를 디자인하는 최초의 방법입니다. ShapeLib은 사용자가 제공하는 텍스트 설명과 샘플 형태로 입력된 디자인 의도를 기반으로 절차적 추상을 발견합니다. 이 시스템은 형상을 아우르는 라이브러리를 구성하여, 차별화된 매개변수로 다양한 형태 변형을 실현할 수 있도록 합니다. 이는 기존 방법들과 비교했을 때 명확한 이점을 보여줍니다.

- **Technical Details**: ShapeLib은 입력으로 자연어로 된 함수 설명과 샘플 형태를 받아들이며, 이 두 가지 모달리티를 통해 절차적 모델 설계를 지원합니다. 먼저, LLM을 사용하여 함수 설명에 조건부로 라이브러리 인터페이스를 설계하고, 다음으로 이러한 함수를 적용하여 샘플 형태를 설명하는 방안을 제안합니다. 이 과정에서 최종적으로 기하학적 분석을 통해 제안된 함수 구현을 검증합니다. 또한, 인식 네트워크를 통해 입력 형태를 라이브러리 기능을 사용하여 설명하는 프로그램을 추론하는 방법을 학습합니다.

- **Performance Highlights**: ShapeLib은 다양한 형태 카테고리에서 절차적 함수 라이브러리를 설계하는 데 활용되며, 발견된 함수들은 자연어 설명에 따른 상향식 의미에 맞춰져 있습니다. 실험 결과, 이 라이브러리는 시드 세트를 넘어서는 일반화, 해석 가능성 및 신뢰성을 유지하는 중요한 이점을 나타내며, 결과적으로 명확한 형상 변형을 제공합니다. 이 연구는 ShapeLib이 기존 접근법에 비해 절차적 표현의 이점을 더 잘 실현한다는 것을 보여줍니다.



### Harnessing Vision Models for Time Series Analysis: A Survey (https://arxiv.org/abs/2502.08869)
- **What's New**: 이 논문은 시간 시계열 분석(time series analysis)에서 LLMs(대형 언어 모델)보다 LVMs(대형 비전 모델)의 장점에 대해 논의합니다. 최근까지 연구의 대부분은 시퀀스 모델링(sequence modeling)에 집중되어 있었으나, 비전 모델이 시간 시계열 분석에서 중요한 역할을 할 수 있다는 점을 강조하고 있습니다. 이 논문은 시간 시계열을 이미지로 인코딩하는 방법과 여러 작업을 위한 모델링 기법을 제시하며, 이 분야의 나아갈 방향도 함께 논의합니다.

- **Technical Details**: 시간 시계열은 다양한 형태로 표현될 수 있으며, 여기서 이미지로 변환된 시간 시계열(imaged time series)을 대상으로 연구합니다. UTS(단일 변량 시계열)와 MTS(다변량 시계열)의 정의 및 표현 방법이 상세히 설명됩니다. 그림 1에서는 비전 모델을 시간 시계열 작업에 적용하는 일반적인 프로세스와 프레임워크를 제시하며, 경우에 따라 선형 그래프(line plot), 히트맵(heatmap), 스펙트로그램(spectrogram) 등의 변환 기법에 대해 다룹니다.

- **Performance Highlights**: 비전 모델이 시간 시계열 작업에서 왜 더 효과적인지를 여러 사례를 통해 설명하고 있으며, 기존의 LLMs와 비교해 여러 장점이 발견됩니다. LVMs는 시계열 데이터의 고유한 패턴, 상관 관계(correlation) 및 장기 종속성(long-term dependency) 모형화를 더 잘 수행할 수 있습니다. 또한, 이 논문은 향후 LMMs(대형 다중모달 모델)의 가능성을 제시하며, 시계열 데이터와 비전 데이터를 통합한 혁신적인 방식의 개발을 독려합니다.



### A Reversible Solver for Diffusion SDEs (https://arxiv.org/abs/2502.08834)
Comments:
          Preprint

- **What's New**: 최근 확산(기법) 모델은 다양한 데이터 모달리티에서 생성 작업의 최첨단(상태-최고)으로 빠르게 자리 잡았습니다. 이 모델의 중요한 기능 중 하나는 데이터 분포에서 샘플을 샘플링 우선 분포로 다시 인코딩(encoding) 할 수 있는 능력입니다. 이는 실제 데이터 샘플을 수정하거나 연속적인 부가 방정식을 통해 가이던스 생성(guided generation)을 수행하는 데 유용합니다.

- **Technical Details**: 우리는 확산(기법) SDE의 대수적으로 가역적인 솔버를 제안합니다. 이 솔버는 실제 데이터 샘플을 정확하게 우선 분포로 역전(invert)할 수 있습니다. 대수적 가역성(algebraic reversibility)은 모델이 데이터 샘플 간의 변환을 효율적으로 수행할 수 있도록 도와줍니다.

- **Performance Highlights**: 이 모델은 현실 데이터를 조작하는 데에서 실질적인 응용 가능성을 보여주며, 변형된 데이터 샘플을 생성하는 데 있어 뛰어난 성능을 자랑합니다. 또한, 기대할 수 있는 샘플링 결과의 품질을 높이는 데 있어 유망한 연구 방향을 제시합니다.



### A Survey on Data-Centric AI: Tabular Learning from Reinforcement Learning and Generative AI Perspectiv (https://arxiv.org/abs/2502.08828)
- **What's New**: 이 논문은 데이터 중심 인공지능(Data-Centric AI, DCAI) 관점에서 테이블 형식 데이터의 최적화를 위한 새로운 기법을 제시합니다. 특히 강화 학습(Reinforcement Learning, RL)과 생성적 접근법을 통해 중요한 특성을 선택하고 새로운 특성을 생성하는 방법을 탐구합니다. 이를 통해 데이터 품질을 개선하고 모델 성능을 향상시키는 데 기여하고자 합니다.

- **Technical Details**: 테이블 형식 데이터는 복잡한 특성 의존도, 높은 차원성 및 해석 가능성 요건 등을 만족해야 하며, 이로 인해 많은 도전 과제를 안고 있습니다. 본 논문에서는 이러한 도전 과제를 해결하기 위해 강화 학습 기반 기법과 생성 모델을 활용하여 특성 선정(feature selection) 및 특성 생성(feature generation) 과정을 최적화하는 방안을 제시합니다. 강화 학습은 보상 기반 탐색을 통해 모델이 적응적으로 특성 표현을 학습할 수 있도록 하며, 생성 모델은 복잡한 데이터 패턴을 포착하는 새로운 특성을 생성합니다.

- **Performance Highlights**: 이 연구는 데이터 중심 AI 및 테이블 형식 데이터 엔지니어링의 최신 발전을 체계적으로 검토하고, 기존 방법들의 강점과 한계를 분석합니다. 연구결과, RL 및 생성적 접근법이 테이블 형식 데이터의 특성 공학(feature engineering) 자동화와 지능화에 기여함을 보여줍니다. 또한, 기존의 도전 과제를 정리하고 향후 연구 방향을 제시하여 이 분야의 지속적인 혁신을 위한 통찰력을 제공하고자 합니다.



### Ask in Any Modality: A Comprehensive Survey on Multimodal Retrieval-Augmented Generation (https://arxiv.org/abs/2502.08826)
- **What's New**: 이 논문은 최근 발전된 Retrieval-Augmented Generation (RAG) 시스템, 특히 Multimodal RAG의 구조적이고 종합적인 분석을 제공하고 있습니다. 기존의 RAG 아키텍처가 주로 텍스트 정보를 중심으로 설계된 반면, Multimodal RAG는 텍스트, 이미지, 오디오, 비디오와 같은 다양한 형식을 통합하여 생성된 출력을 향상시킵니다. 이는 AI 시스템이 더 신뢰할 수 있고 유능하게 멀티모달 동적 외부 지식 데이터를 활용할 수 있는 기반을 마련합니다.

- **Technical Details**: 최근 Multimodal RAG의 발전은 다양한 데이터 소스를 통합하고 분석하는 능력을 촉진하며 정보의 전체적인 표현을 가능하게 합니다. 이 시스템은 특정 모달리티를 비교 평가하고, 교차 모달의 연관성을 해결하기 위한 특별한 도전 과제를 제시합니다. 또한 정보 검색 및 콘텐츠 생성 단계에서의 정확성을 높이기 위해 점진적 추론 과정을 도입하여 정확한 응답을 도출하는 방법을 소개합니다.

- **Performance Highlights**: RAG 시스템은 대규모의 외부 지식 저장소에서 최신 지식을 동적으로 가져와 사실 정확성을 개선하고 환각(hallucinations)을 줄이는 데 효과적입니다. 기본적으로 RAG 시스템은 Retriever-Generator 파이프라인을 통해 작동하며, 외부 맥락을 통합하여 정보에 기반한 응답을 생성합니다. 논문에서는 다양한 멀티모달 RAG 시나리오 및 평가 방법론을 다루어 향후 연구 방향 및 개방된 문제를 논의합니다.



### DejAIvu: Identifying and Explaining AI Art on the Web in Real-Time with Saliency Maps (https://arxiv.org/abs/2502.08821)
Comments:
          5 pages, 3 figures, submitted to IJCAI 2025 demo track

- **What's New**: DejAIvu는 사용자가 웹을 탐색하는 동안 AI 생성 이미지에 대한 실시간 탐지를 제공하며, saliency 기반 설명 가능성을 결합한 Chrome 웹 확장입니다. 이 도구는 ONNX 최적화된 딥러닝 모델을 사용하여 Google Images와 같은 웹사이트에서 자동으로 이미지를 분석하고 AI 관련 아티팩트를 강조하는 saliency 히트맵을 오버레이합니다. 사용자는 이 확장을 통해 AI 이미지의 투명성과 해석 가능성을 보장받을 수 있습니다.

- **Technical Details**: DejAIvu는 사용자가 웹페이지에서 이미지를 탐지하여 ONNX 최적화된 AI 모델을 통해 분류한 다음, AI 특정 아티팩트를 강조하는 saliency 맵을 생성합니다. 내부적으로는 NVIDIA RTX 6000 Ada GPU를 활용하여 모델을 훈련하고, 270,000개 이상의 인간 및 AI 생성 아트워크로 구성된 편집된 데이터셋에서 학습하였습니다. 입력 이미지에 대해 256×256 픽셀로 크기를 조정하고 정규화한 후, 모델에 공급하여 정확도를 높이기 위한 다양한 데이터 증강 기법을 적용합니다.

- **Performance Highlights**: DejAIvu는 ResNet50 모델을 활용하여 97.1%의 높은 정확도를 달성하며 90.6MB의 합리적인 파일 크기로 운영됩니다. ONNX.js를 활용한 웹 기반 추론을 통해 평균적으로 이미지당 약 35ms의 레이턴시 감소를 이루어냈으며, 이는 실시간 성능에 큰 개선을 의미합니다. 다양한 아키텍처의 성능 비교에서 DejAIvu는 정확도와 실시간 효율성을 모두 갖춘 최적의 도구로 평가받고 있습니다.



### CLOVER: A Test Case Generation Benchmark with Coverage, Long-Context, and Verification (https://arxiv.org/abs/2502.08806)
Comments:
          16 pages

- **What's New**: 이 논문에서는 소프트웨어 테스트에서 모델의 테스트 케이스 생성 및 완료 능력을 평가하기 위해 CLOVER라는 벤치마크를 제시합니다. 이 벤치마크는 12개의 Python 저장소에서 845개의 문제를 분석하며, 간단한 어설션 완성과 여러 파일에서 특정 코드 블록을 커버하는 테스트 케이스 작성까지 다양한 작업을 포함합니다. 이 연구는 테스트 커버리지 정보를 활용하여 검색 컨텍스트를 구성하는 방법을 제안하며, 특히 16k의 긴 컨텍스트에서 모델 성능의 유의미한 차이를 발견하였습니다.

- **Technical Details**: CLOVER 벤치마크는 코드 실행과 라인 커버리 측정을 통해 LLM(대규모 언어 모델)의 성능을 평가합니다. 테스트 케이스 생성을 위한 자동 파이프라인은 GitHub에서 허용된 저장소를 스크랩하고 실행 환경을 구성하는 데 중점을 둡니다. 연구진은 세 가지 도전 과제를 구조화했으며, 각 과제는 다양한 길이의 컨텍스트를 사용하여 LLM의 성능을 평가합니다.

- **Performance Highlights**: 모델들 간의 성능 차이가 두드러지며, Claude 3.5-S와 GPT-4o가 가장 높은 성과를 보였습니다. 그러나 모든 모델은 Task III에서 35% 미만의 낮은 점수를 기록했으며, 이는 벤치마크의 중요성과 모델 개선의 잠재력을 강조합니다. 연구진은 향후 이 벤치마크와 관련된 코드, 데이터, 구축 방법론을 공개할 예정이며, 이를 통해 커뮤니티의 발전을 도울 계획입니다.



### Auction Design using Value Prediction with Hallucinations (https://arxiv.org/abs/2502.08792)
- **What's New**: 이번 연구에서는 판매자가 n명의 구매자에게 개별 상품을 판매할 때 수익을 극대화하기 위한 Bayesian mechanism design 문제를 다룹니다. 여기서 구매자의 개인 가치에 대한 신뢰성이 떨어질 수 있는 예측(신호)들이 머신러닝 모델에서 도출됩니다. 기존 연구와는 차별화된 점은 신호가 구매자의 실제 평가를 반영하는 경우도 있지만, 때때로는 전혀 관련 없는 'hallucinations'일 수 있다는 것입니다.

- **Technical Details**: 이 연구는 '신호'를 기준으로 구매자 유형을 나누는 최적 경매의 특성을 규명합니다. '신호' 이상의 유형과 이하의 유형을 분리하여 처리하는 방법에 대한 거의 분해(decomposition)를 제시하며, 이러한 구조는 판매자가 신호에 따라 최적의 가격을 설정할 수 있는 기초를 제공합니다. 세 가지 직관적인 가격 책정 전략인 'ignore', 'follow', 'cap' 행동을 통해 하나의 구매자에 대해 최적의 전략을 수립할 수 있습니다.

- **Performance Highlights**: 이 프레임워크를 통해 기존의 Bayesian auction 이론에 새로운 관점을 추가하며, 신뢰할 수 있는 신호와 불확실한 신호 간의 차별화된 대응 방안을 모색하였습니다. 이러한 전략들은 판매자가 시장에서 직면하는 다양한 상황에 비추어 유용한 직관적인 가격 결정 방식으로 작용할 수 있습니다.



### Acoustic Wave Manipulation Through Sparse Robotic Actuation (https://arxiv.org/abs/2502.08784)
Comments:
          ICRA 2025

- **What's New**: 이 연구는 로봇이 직접 닿지 않고 중개 도구를 통해 음파를 조작하는 방법에 대한 새로운 접근 방식을 제시합니다. 기존의 물체 조작 기술과 다르게, 음파 조작에서는 중개 도구와의 상호작용을 통해 파장을 제어하는 독특한 도전 과제가 존재합니다. 제안된 방법은 데이터 기반(data-driven) 로봇 학습 방식을 통해 비파괴적인 초음파 가공, 에너지 흡수 및 새로운 인공 재료 디자인에 활용될 수 있습니다.

- **Technical Details**: 이 연구에서 제안된 방법은 음파 조작을 위해 설계되었으며, 이는 파동 상태에 대한 부분적인 정보만을 가지고 있으며, 희소한(actuator) 제어 신호를 사용하여 수행됩니다. 연구자들은 데이터 기반 방식으로 음향 파동을 조작하는 해석 가능한 프레임워크를 설계하여, 강력한 제어와 물리적 해석 가능성을 결합했습니다. 또한, 제안된 프레임워크는 물리적 모델(1D wave equation)의 제약을 통해 해석 가능성을 보장합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 기존의 기계 학습(Machine Learning) 기반 방법보다 우수한 성능을 보였으며, 반면 기존의 반분석적인 방법과도 경쟁력을 유지했습니다. 이 연구는 동적 시스템을 제어하는 데 있어 혁신적인 접근을 제공하며, 이러한 기법은 화재, 전염병 및 기후 변화와 같은 다양한 복잡한 시스템의 제어에도 적용될 수 있을 것으로 기대됩니다.



### Exploring Test Time Adaptation for Subcortical Segmentation of the Fetal Brain in 3D Ultrasound (https://arxiv.org/abs/2502.08774)
Comments:
          5 pages, 5 figures

- **What's New**: 이번 연구에서는 초기 이미징 작업에서 종종 성능 저하를 일으키는 도메인 변화(domain shift)를 극복하기 위해 테스트 시간 적응(test-time adaptation, TTA) 기법을 활용하여 모델 성능을 개선할 수 있음을 보여줍니다. 특히, 기존의 TTA 방법인 Test Entropy Minimisation(TENT)을 수정하여 초음파(ultrasound, US) 영상에 보다 적합한 새로운 TTA 방법(EntropyKL)을 제안하고, 이를 통해 각 피질 아래 지역의 예상 볼륨에 대한 표준해부학적 아틀라스를 사전 정보로 포함시킵니다.

- **Technical Details**: 연구에서는 사전 훈련된 소스 모델 f(𝑿s,𝒀s;ϕ)를 사용해 TTA 접근법을 사용하여 목표 데이터(Dt)에서 성능 극대화를 달성하는 방법을 설명합니다. 제안된 EntropyKL 메소드는 US 아틀라스를 사용하여 예측되는 각 피질 아래 영역의 볼륨에 대한 정보를 포함하고 있으며, 다양한 도메인 이동에서의 성능 향상을 통한 벤치마킹 과정도 포함되어 있습니다. 또한, 나이브한 손실 최소화 기법의 위험성을 줄이기 위해 배치 정규화(batch normalization) 레이어만 최적화하는 기법을 도입했습니다.

- **Performance Highlights**: 모델을 각종 시뮬레이션된 도메인 변화, 실제 도메인 변화 및 임신 주에 따른 도메인 변화에 대해 평가한 결과, EntropyKL 접근 방식이 모든 TTA 접근법 중에서 최고의 성능을 보여주었습니다. 이 연구는 자동화된 태아 뇌 발달 모니터링을 위한 강력한 도구로 자리매김할 수 있는 기술의 발전을 제안합니다. 코드 또한 제공합니다.



### Cluster and Predict Latents Patches for Improved Masked Image Modeling (https://arxiv.org/abs/2502.08769)
Comments:
          13 pages, 7 figures, submitted to TMLR

- **What's New**: 이 논문에서는 Masked Image Modeling (MIM) 분야에서의 새로운 접근법인 CAPI를 소개하며, 이는 잠재 클러스터를 예측하여 구성된 순수 MIM 프레임워크입니다. CAPI는 기존 MIM 방법들과 비교하여 안정적인 훈련을 제공하는 클러스터링 기반 손실(clustering-based loss)을 활용하며, ImageNet에서 83.8%의 정확도를 기록하는 등 향상된 성능을 보였습니다. 또한, 논문에서는 다양한 모델과 코드도 공개되어, 연구자들이 이를 쉽게 활용할 수 있도록 하고 있습니다.

- **Technical Details**: CAPI는 마스크된 이미지 모델링 원리를 중심으로 세 가지 디자인 요소에 주목하여 체계적으로 연구하였습니다: 목표 표현(target representation), 손실 함수(loss function) 구성, 및 예측을 수행하기 위해 사용되는 아키텍처(architecture)입니다. 특히, CAPI는 자기 증류(self-distillation) 기법을 사용하여 교사-teacher와 학생-student 비전 트랜스포머를 훈련하고, 이를 통해 손실을 안정적으로 수렴시킵니다. CAPI는 3억 개의 파라미터를 가진 비주얼 인코더로, 현재의 최첨단 방법과 비슷한 성능을 발휘합니다.

- **Performance Highlights**: CAPI는 이전의 MIM 방법들을 능가하여 ImageNet에서 83.8%의 정확도와 ADE20K에서 32.1%의 mIoU를 달성하며, DINOv2의 성능에 근접하는 결과를 보였습니다. 이는 MIM 방식의 잠재적인 장점을 극대화한 것으로, 기존의 다른 방법들과 비교할 때 매우 경쟁력 있는 결과로 평가됩니다. 논문은 MIM의 실행 가능성을 열어주는 중요한 기여로 여겨지며, 향후 연구와 발전에 있어 중요한 기초 자료가 될 것입니다.



### SelfElicit: Your Language Model Secretly Knows Where is the Relevant Evidenc (https://arxiv.org/abs/2502.08767)
Comments:
          16 pages, 5 figures, 8 tables

- **What's New**: 이 논문은 SelfElicit라는 새로운 접근 방식을 제안합니다. 이 기법은 언어 모델(LMs)이 중요한 증거를 식별하고 강조하도록 도와줍니다. 이를 통해 LM이 더욱 정확하고 사실 기반의 응답을 생성할 수 있도록 합니다. SelfElicit는 추가적인 훈련 없이 인퍼런스(inference) 시간에 효율적으로 작동합니다.

- **Technical Details**: SelfElicit는 LM의 내부 표현을 활용하여 문맥 내에서 관련 증거 문장을 식별합니다. 문맥과 질문으로 구성된 입력을 사용하여 LM은 지원하는 사실을 활용하여 QA 작업을 수행합니다. 각 문장의 중요성을 평가하기 위해 송신기(Transformer) 모델의 주의(attention) 점수를 분석하며, 이를 통해 LM은 중요 정보를 더욱 효과적으로 처리할 수 있습니다.

- **Performance Highlights**: SelfElicit는 다양한 모델 가족과 벤치마크에서 QA 작업의 성능을 일관되게 개선하는 데 성공했습니다. 본 연구에서 SelfElicit의 효과성을 입증했으며, 이 방법은 노이즈에 강하고 하이퍼파라미터 선택에 덜 민감하다는 장점이 있습니다. 결과적으로 SelfElicit는 LM의 사실적이고 신뢰할 수 있는 응답 생성을 향상시키는 데 기여합니다.



### HistoSmith: Single-Stage Histology Image-Label Generation via Conditional Latent Diffusion for Enhanced Cell Segmentation and Classification (https://arxiv.org/abs/2502.08754)
- **What's New**: 이 논문은 HistoSmith라는 새로운 단일 단계 접근 방식을 도입하여 조직학 이미지의 세포 인스턴스 세분화 및 분류를 위한 데이터 증강을 지원합니다. 기존의 분산 모델과 달리, HistoSmith는 잠재적 분산 모델(latent diffusion model)을 사용하여 세포 배치 및 분류 마스크와 같은 데이터 쌍을 생성합니다. 이 모델은 세포 유형, 수량 및 조직 유형과 같은 사용자 정의 매개변수를 기반으로 맞춤형 데이터 생성을 가능하게 합니다.

- **Technical Details**: HistoSmith는 VQ-VAE를 이용하여 입력 이미지와 해당 마스크의 잠재 표현을 학습합니다. 이후, 학습된 잠재 공간에서 분산 모델을 훈련시켜 이미지와 마스크를 생성하며, 이 과정에서 세포 수 및 조직 유형을 인코딩한 10차원 벡터로 조건화합니다. 복원 과정에서 U-Net을 활용하여 각 단계에서 추가된 노이즈를 예측하고, 최종적으로 VQ-VAE 디코더를 통해 새로운 이미지와 해당 마스크를 복원합니다.

- **Performance Highlights**: HistoSmith는 CoNIC 및 CytoDArk0 데이터셋에서 훈련되어, 평균 CS 및 CC 메트릭에서 각각 1.9%와 3.4%의 성능 향상을 보였습니다. 이 연구는 특히 저대표 세포 유형에 대한 데이터 부족 문제에 대응할 수 있는 잠재의 분산 모델 기반 접근 방식을 제시합니다. 본 논문은 조직학 데이터 세트를 다시 증강하는 데 있어 HistoSmith의 효과성과 품질을 검증했습니다.



### Scalable Discrete Diffusion Samplers: Combinatorial Optimization and Statistical Physics (https://arxiv.org/abs/2502.08696)
Comments:
          Accepted at ICLR 2025

- **What's New**: 이번 논문에서는 복잡한 비정규화 분포에서 샘플링하는 새로운 방법론이 제시됩니다. Diffusion models의 잠재력은 인식되었지만, 기존 방법에서는 메모리 문제로 인해 한계가 있었습니다. 이를 극복하기 위해 Policy Gradient와 Self-Normalized Neural Importance Sampling (SN-NIS)을 기반으로 한 새로운 훈련 방법을 도입하였습니다.

- **Technical Details**: 제안된 방법은 메모리 효율적인 훈련을 가능하게 하여 비지도형 조합 최적화에서 최첨단 결과를 달성합니다. SN-NIS와 Neural Markov Chain Monte Carlo의 적응을 통해 비편향 샘플링이 가능해지며, 이는 디스크리트 diffusion 모델에서의 혁신적인 응용으로 이어집니다. Ising 모델 벤치마크에서 검증된 결과는 기존의 오토 회귀 접근 방법보다 우수함을 입증합니다.

- **Performance Highlights**: 논문에서 제안하는 방법은 기존의 방법보다 더 많은 과학적 응용 분야에서 효과적으로 활용될 수 있음을 보여주며, 특히 비지도형 조합 최적화 문제에 대한 새로운 접근 가능성을 열어줍니다. 다양한 과학적 응용에 대해 사실상 최대의 우수한 성능을 달성하며, 이전에 정확한 가능성 모델에 국한되었던 범위에서 새로운 길을 개척합니다.



### AgentSociety: Large-Scale Simulation of LLM-Driven Generative Agents Advances Understanding of Human Behaviors and Society (https://arxiv.org/abs/2502.08691)
- **What's New**: 본 논문에서는 AgentSociety라는 대규모 사회 시뮬레이터를 제안합니다. 이 시뮬레이터는 LLM(large language model)을 기반으로 하는 에이전트들과 현실적인 사회 환경, 대규모 상호작용을 통합하여 복잡한 사회 역학을 연구하는 데 중점을 두고 있습니다. 연구자들은 1만 개 이상의 에이전트와 500만 건의 상호작용을 시뮬레이션하여 사회적 삶을 생성합니다.

- **Technical Details**: AgentSociety는 LLM을 활용하여 사람과 유사한 사고 및 행동을 하는 에이전트를 설계하며, 이 에이전트들은 감정, 필요, 동기 및 외적 세계에 대한 인지를 포함합니다. 이 시뮬레이터는 또한 도시, 사회 및 경제적 공간을 통합한 현실적인 사회 환경을 구축하여, 대규모 에이전트 상호작용을 촉진하며 복잡한 사회 구조를 생성합니다.

- **Performance Highlights**: AgentSociety는 실세계의 다섯 가지 사회 실험에서 관찰된 행동, 결과 및 패턴을 성공적으로 재현했습니다. 이는 사회 연구 방법들을 제공함으로써, 사회 과학자들과 정책 입안자들에게 새로운 가능성을 열어주며, 비용, 확장성 및 실현 가능성에 있어 전통적인 실험의 한계를 극복할 수 있는 기반을 마련해 줍니다.



### Skrr: Skip and Re-use Text Encoder Layers for Memory Efficient Text-to-Image Generation (https://arxiv.org/abs/2502.08690)
- **What's New**: 이번 연구에서는 텍스트-이미지(T2I) 확산 모델에서 텍스트 인코더의 메모리 사용 효율을 높이기 위한 새로운 방법인 Skip and Re-use layers(Skrr)를 제안합니다. 기존의 denoising 모듈과 달리, 텍스트 인코더는 단일 순방향 통과만으로 텍스트 임베딩을 생성하는데, 이는 메모리 사용량이 높아지는 원인으로 작용합니다.

- **Technical Details**: Skrr는 T2I 작업에 맞춰 설계된 가지치기(pruning) 전략으로, transformer 블록 내의 특정 레이어를 선택적으로 건너뛰거나 재사용하여 메모리 소비를 줄입니다. 이 방법은 높은 희소성(sparsity) 수준에서도 텍스트-이미지 생성 품질을 유지하며, 기존의 블록 기반 가지치기 방법을 초월하는 성능을 보여줍니다.

- **Performance Highlights**: 광범위한 실험을 통해 Skrr는 FID, CLIP, DreamSim 및 GenEval 점수 등 여러 평가 메트릭에서 성능을 유지하면서도 메모리 효율성을 극대화하는 것을 입증했습니다. 이로 인해 Skrr는 최신 기술(trade mark)보다 우수한 메모리 효율성을 달성하며, 성능 저하 없이 높은 품질의 이미지를 생성할 수 있습니다.



### Advancing machine fault diagnosis: A detailed examination of convolutional neural networks (https://arxiv.org/abs/2502.08689)
- **What's New**: 이번 논문에서는 기계 고장 진단에 대한 최신의 진전을 다루고 있습니다. 특히, Convolutional Neural Networks (CNNs)의 이론적 기초와 구조적 변화를 심층적으로 검토합니다. CNNs의 강력한 고장 탐지 및 분류 능력은 최근의 데이터 증강(data augmentation)과 전이 학습(transfer learning) 기법을 통해 더욱 발전하고 있습니다.

- **Technical Details**: 논문은 CNN이 다양한 고장 유형 및 복잡한 데이터에 어떻게 적용되는지를 강조하며, CNN 아키텍처의 변형에 대한 이론적인 배경을 제공합니다. CNN의 강점과 한계를 분석하며, 특정 운영 환경에서의 효과성을 검증합니다. 또한 하이브리드 아키텍처(hybrid architectures)에 대해서도 논의합니다.

- **Performance Highlights**: CNN 기반 고장 진단 기술의 신뢰성과 능동적 접근 방식을 개선하는 데 있어 향후 연구 방향과 잠재적인 과제를 조명합니다. 이를 통해 기계 고장 진단에 대한 지속적인 발전이 이루어질 것으로 기대하고 있습니다. 특히, CNN을 활용한 고장 진단 시스템의 효율성을 높이는 것이 중요한 목표로 설정되어 있습니다.



### EEG Artifact Detection and Correction with Deep Autoencoders (https://arxiv.org/abs/2502.08686)
- **What's New**: 본 연구에서는 EEG 신호의 아티팩트 감지 및 교정을 위해 새로운 LSTM 기반의 오토인코더인 LSTEEG를 제안합니다. LSTEEG는 순차적 EEG 데이터의 비선형 종속성을 캡처하며, 기존의 다른 최첨단 합성곱 오토인코더에 비해 우수한 성능을 보입니다. 이 방법론은 오토인코더의 잠재 공간의 해석 가능성과 유용성을 향상시켜 EEG 신호의 아티팩트를 자동으로 제거하는 데이터 기반 접근 방식을 가능하게 합니다.

- **Technical Details**: LSTEEG는 각 EEG 세그먼트를 동질적인 저차원 잠재 공간으로 인코딩하는 방식을 채택합니다. LSTM 층을 활용하여 시퀀스 데이터에서 장기 비선형 종속성을 포착하며, 아티팩트 제거 전후 과정에서 특정 EEG 세그먼트를 식별하기 위해 이상 감지 기법을 통합합니다. 연구팀은 LEMON 데이터셋을 사용하여 오토인코더의 성능 평가를 위해 60/20/20 비율로 데이터 세트를 나누어 훈련, 검증 및 테스트를 수행합니다.

- **Performance Highlights**: LSTEEG는 아티팩트 감지 및 교정 작업에서 기존의 최첨단 합성곱 AEs와 비교하여 경쟁력 있는 성능을 발휘합니다. 연구 결과는 LSTEEG가 낮은 차원의 의미 있는 표현을 학습함으로써 신호 해석의 새로운 가능성을 열어준다고 강조합니다. 이러한 표현은 EEG 신호를 이해하고, 후속 작업을 위한 신경생리학적 특징을 추출하는 데 도움을 줄 수 있습니다.



### Beyond Models! Explainable Data Valuation and Metric Adaption for Recommendation (https://arxiv.org/abs/2502.08685)
- **What's New**: 이번 연구에서는 추천 시스템에서의 데이터 품질을 평가하기 위해 설명 가능하고 다목적의 데이터 가치 평가 프레임워크(DVR)를 제안합니다. 기존 방법들은 블랙박스 설계를 사용하여 데이터 평가에 대한 투명성과 해석 가능성이 부족했습니다. DVR은 게임 이론적 관점에서 Shapley 가치를 계산함으로써 데이터의 질을 평가하고, 다양한 모델 아키텍처 및 평가 메트릭에 맞춰 데이터 활용의 효율성을 향상시킵니다.

- **Technical Details**: DVR 프레임워크는 설명 가능한 데이터 가치 평가를 위해 데이터 가치 측정기를 도입하여, 데이터가 모델 성능에 미치는 기여도를 평가합니다. 해당 데이터 가치 측정기는 데이터와 모델 성능을 플레이어와 결과로 간주하여 Shapley 가치를 계산합니다. 또한, 비차별적 메트릭을 포함한 여러 평가 메트릭에 맞추기 위해 강화를 기반으로 한 메트릭 어댑터를 개발하여 최적의 메트릭 성능을 유도합니다.

- **Performance Highlights**: 다양한 기준에서 실시된 광범위한 실험 결과, DVR 프레임워크는 현재의 추천 알고리즘 성능을 크게 향상시켰습니다. 특히, NDCG 메트릭에서는 기존 방법 대비 최대 34.7%의 성능 개선을 달성했습니다. 이러한 성과는 추천 시스템의 랭킹 정확도, 다양성 및 공정성 등 여러 면에서의 향상으로 이어집니다.



### Self-Evaluation for Job-Shop Scheduling (https://arxiv.org/abs/2502.08684)
- **What's New**: 본 논문은 조합 최적화 문제(Combinatorial Optimization Problems, COPs)에 대한 새로운 접근 방식을 제안합니다. 기존의 단계별 방법에서 벗어나 작업 할당의 부분 집합을 생성하고 평가하는 방식으로, Neural Combinatorial Optimization 방법론을 살펴봅니다. 이를 통해 오류 축적 문제를 완화하고, 고품질의 최종 결과를 도출하겠습니다.

- **Technical Details**: 제안된 프레임워크는 이종 그래프 신경망(Heterogeneous Graph Neural Network, HGNN)과 변환기(Transformer)를 결합한 정책 모델과 자기 평가 모델로 구성됩니다. 정책 모델은 가능한 행동에 확률을 할당하며, 자기 평가 모델은 이 행동 집합의 품질을 전반적으로 평가합니다. 이는 조합 최적화 문제를 마르코프 결정 프로세스(Markov Decision Process, MDP)로 재정의하여, 단일 행동 공간에서 부분 서브시퀀스 기반 공간으로 전환합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존의 최첨단 방법들과 비교해 상당한 최적화 향상을 보여주었습니다. 특히, Google의 CP-SAT 솔버와 비교했을 때, 우리가 제안한 접근 방식이 더 높은 성능과 더 낮은 계산 비용으로 더 나은 결과를 도출했음을 확인했습니다. 이 결과는 조합 최적화 문제를 해결하는 데 있어 우리의 방법론이 강력한 효율성을 발휘함을 보여줍니다.



### On the Role of Pre-trained Embeddings in Binary Code Analysis (https://arxiv.org/abs/2502.08682)
- **What's New**: 이 논문은 바이너리 코드 분석에서 사전 훈련된 임베딩(embeddings)의 역할을 비판적으로 탐구하고 있습니다. 특히, 어셈블리 코드에 대한 최근 임베딩을 체계적으로 평가하여, 라벨 데이터의 양에 따라 달라지는 성능을 확인합니다. 이 연구는 기존 연구에서 언급된 차이가 실질적으로 미미하다는 것을 발견했습니다.

- **Technical Details**: 연구는 1.2백만 개의 함수가 포함된 Debian 배포판의 말뭉치(corpus)를 사용하여 다섯 가지 하위 작업(downstream tasks)에 대해 최근 임베딩의 성능을 분석합니다. 또한, 함수 경계(function boundaries), 최적화 수준(optimization levels), 인자 유형(argument types)과 같은 레이블 데이터가 쉽게 생성될 수 있음을 강조하고 있습니다. 이러한 분석을 통해, 엔드 투 엔드 러닝(end-to-end learning)이 평균적으로 가장 좋은 성능을 발휘한다는 결과를 도출했습니다.

- **Performance Highlights**: 연구 결과, 충분한 라벨 데이터가 주어졌을 때 여러 임베딩의 성능이 유사하게 나타났습니다. 특정한 임베딩의 필요성이 의문시되며, 언제 임베딩이 이점을 제공하고 언제 엔드 투 엔드 러닝이 더 바람직한지를 도출하기 위한 가이드라인이 제시됩니다. 이러한 결과는 바이너리 코드 분석을 위한 새로운 연구 방향을 제시합니다.



### Centrally Coordinated Multi-Agent Reinforcement Learning for Power Grid Topology Contro (https://arxiv.org/abs/2502.08681)
- **What's New**: 본 논문에서는 전력망 운영의 복잡성을 해결하기 위해 CCMA(centrally coordinated multi-agent) 아키텍처를 제안합니다. 이 아키텍처는 지역 에이전트들이 제안한 행동을 바탕으로 중앙 조정 에이전트가 최종 행동을 선택하여 결정 과정을 분해합니다. 기존의 L2RPN 접근 방식과 비교했을 때, CCMA 아키텍처는 샘플 효율성과 최종 성능이 우수함을 보이며, 이 접근법의 잠재력을 강조합니다.

- **Technical Details**: 연구에서는 배급 결정을 단순화하기 위해 단기 행동 제안을 CCMA 아키텍처에 통합합니다. 이 아키텍처는 규칙 기반, 탐욕적(greedy), RL 기반 전략을 포함한 여러 변형을 비교 분석합니다. 각 에이전트의 제안된 행동과 글로벌 상태 정보를 활용하여 중앙 집중식 조정을 통해 행동 공간을 분해합니다.

- **Performance Highlights**: 제안된 CCMA 아키텍처는 샘플 효율성과 성능 면에서 기존의 기준 모델보다 뛰어난 결과를 보였습니다. 실험 결과 일부 변형은 모든 기준 모델을 초과하는 성과를 달성하는 것으로 나타났으며, 현실적인 전력망에도 적용 가능성이 높음을 시사합니다. 특히, 소규모 5-버스 네트워크에서는 규칙 기반 조정자가 효과적이지만, 대규모 네트워크에서는 RL 기반 조정자가 더 나은 성능을 발휘하는 것을 확인했습니다.



### Mathematical Reasoning in Large Language Models: Assessing Logical and Arithmetic Errors across Wide Numerical Ranges (https://arxiv.org/abs/2502.08680)
- **What's New**: 본 연구는 GSM8K에서 파생된 데이터셋 생성기인 GSM-Ranges를 도입하여, 수학 문제의 숫자 값을 체계적으로 변경함으로써 다양한 숫자 범위에서 모델의 강건성을 평가하고자 합니다. 이는 실제 문제 해결에 필요한 보다 다양한 수치 범위를 반영하며, LLM의 논리적 오류와 비논리적 오류를 구별하는 새로운 채점 방법론도 제공합니다. 이러한 접근은 LLM의 수학적 추론 능력에 대한 보다 포괄적인 평가를 가능하게 하며, 향후 연구 방향에도 중요한 인사이트를 제공합니다.

- **Technical Details**: GSM-Ranges는 GSM8K 데이터셋을 기반으로 하여, 문제 내 숫자를 다양한 규모로 섞어주는 여섯 가지의 왜곡 수준을 적용합니다. 새로운 채점 방법론은 GPT-4o 모델을 사용하여 LLM이 생성한 응답을 Python 코드로 변환하고, 이 코드를 실행하여 비논리적 오류를 격리시킵니다. 최종 결과는 기초적 논리 오류와 계산적 오류를 구분하여, LLM의 수학적 추론 과정을 보다 세밀하게 평가하는 데 기여합니다.

- **Performance Highlights**: 모델의 숫자 복잡성이 증가함에 따라 논리적 오류율이 최대 14%포인트까지 증가하는 경향을 관찰했습니다. 또한, LLM이 단독 산술 작업에서는 높은 정확도를 보이지만, 단어 문제로 계산이 포함될 경우 성능이 크게 저하되는 것을 나타냅니다. 이러한 결과는 LLM의 수학적 추론 능력에 대한 포괄적인 이해를 제공하며, 다양한 수학 문제 처리 과정에서의 강건성을 향상시키기 위한 연구의 초석이 될 수 있습니다.



### Deep Learning-Driven Malware Classification with API Call Sequence Analysis and Concept Drift Handling (https://arxiv.org/abs/2502.08679)
- **What's New**: 이 연구에서는 동적 환경에서의 맬웨어(malware) 분류 시 발생하는 개념 드리프트(concept drift)를 해결하기 위해 유전자 알고리즘(genetic algorithm)을 활용한 심층 학습(deep learning) 프레임워크를 제안합니다. 이는 맬웨어 데이터의 통계적 특성이 시간에 따라 변화하는 도전 과제를 다루는 데 효과적입니다. 새로운 접근 방식은 유전자 알고리즘을 통해 지속적으로 모델을 개선하여 진화하는 맬웨어 위협에 대한 견고함을 보장합니다.

- **Technical Details**: 제안된 모델은 맬웨어 샘플을 행동 및 내재적 특성에 따라 분류하며, 유전자 알고리즘을 통해 개념 드리프트를 관리합니다. 본 연구에서는 n-gram API 호출을 사용하여 패턴을 식별하고 맬웨어 샘플의 행동을 분석하는 방식으로 분류 능력을 향상합니다. 또한, 심층 학습 기반의 맬웨어 분류 프레임워크를 개인화된 데이터 전처리, 특성 선택, 분류 단계를 포함하여 설명합니다.

- **Performance Highlights**: 실험 결과, hybrid 방법론이 전통적인 정적 모델보다 맬웨어 분류 성능과 적응성을 크게 향상시키는 것으로 나타났습니다. 본 연구에서 제안한 접근 방식은 빠르게 변화하는 사이버 보안 환경에서 실시간 맬웨어 분류에 대한 유망한 솔루션을 제공합니다. 또한 다양한 심층 학습 알고리즘과 유전자 알고리즘을 조합한 결과를 면밀히 분석한 내용을 포함합니다.



### Hallucination, Monofacts, and Miscalibration: An Empirical Investigation (https://arxiv.org/abs/2502.08666)
Comments:
          Code available at this https URL

- **What's New**: 이 논문은 언어 모델(LLMs)에서 환각율(hallucination rate)을 훈련 데이터의 단일 사실 비율(monofact rate)과 모델의 미캘리브레이션(miscalibration)에 의해 하한선이 설정될 수 있음을 입증하는 이론적 연구를 기반으로 한다. 이 연구는 n-그램 모델과 LLM의 맥락 학습(in-context learning)을 통해 다양한 데이터 분포가 단일 사실 비율과 환각 경향에 미치는 영향을 실험적으로 조사하고 검증하였다. 결과적으로 훈련 데이터에서의 사실 빈도의 분포와 캘리브레이션-환각 간의 상충 관계는 확률적 언어 생성에서 불가피한 요소임을 제시한다.

- **Technical Details**: 실험적으로 대조적인 패턴을 생성하기 위해 파레토(Pareto)나 지프(Zipf) 분포를 이용한 훈련 데이터에서 단일 사실 비율(monofact rate)이 다양하게 발생하도록 설계하였다. 또한, n-그램 모델에서 훈련 샘플의 전이 횟수를 선택적으로 복제(selective duplication)하여 캘리브레이션을 조절하는 새로운 알고리즘을 소개한다. 이 알고리즘은 단일 사실 비율을 일정하게 유지하는 가운데 미캘리브레이션을 유도하여 환각의 발생 양상을 고립시킬 수 있게 해준다.

- **Performance Highlights**: 훈련 데이터에서 예시를 선택적으로 중복 복제하는 방식으로 미캘리브레이션을 조절함으로써 심지어 적은 수의 훈련 예시의 선택적 업웨이팅(selective upweighting)만으로도 환각율을 상당히 낮출 수 있다는 것을 실험 결과로 입증하였다. LLM에서의 맥락 학습 실험을 통해도 단일 사실 비율과 환각율 사이의 관계가 여전히 유지됨을 확인하였으며, 이는 언어 모델 최적화에서 캘리브레이션을 유지하는 것의 전통적인 접근법과는 다른 중요한 인사이트를 제공한다.



### Motion Forecasting for Autonomous Vehicles: A Survey (https://arxiv.org/abs/2502.08664)
Comments:
          31 pages, 7 figures

- **What's New**: 이번 논문에서는 자율주행 차량의 행동 예측을 위한 새로운 접근 방식을 제안합니다. 특히, 시나리오 기반(Scenario-based) 및 인식 기반(Perception-based) 모션 예측의 두 가지 주요 접근 방식을 조사합니다. 이 연구는 모션 예측에 대한 공식적인 문제 설정을 제공하고, 관련 데이터 세트와 평가 지표를 정리합니다.

- **Technical Details**: 자율주행 시스템의 기능을 향상시키기 위해 모션 예측(Motion Forecasting)에서는 목표 에이전트(Target Agents), 자율주행차(자아 에이전트, Ego Agent), 주변 에이전트(Surrounding Agents)의 정의가 필요합니다. 이 논문에서는 입력 데이터의 유형에 따라 모션 예측 방법을 분류하며, 원시 인식 데이터(Raw Perception Data)의 중요성을 강조합니다. 이 데이터는 LiDAR, 카메라, 레이더 등 다양한 센서에서 나온 정보를 포함하여 예측의 정확성을 높이는 데 기여합니다.

- **Performance Highlights**: 딥 러닝 기반 모델들은 모션 예측에 있어 상당한 발전을 이루었으며, 복잡한 패턴과 관계를 추출하는 데 뛰어난 성능을 보입니다. 전반적으로, 자율주행에서 사용되는 다양한 예측 방법들이 기존 규칙 기반 모델이나 물리 기반 모델보다 더 복잡한 동적 환경에서도 유연하게 작동할 수 있도록 발전하고 있음을 보여줍니다. 마지막으로, 자기 지도 학습(Self-supervised Learning)으로의 전환이 데이터 부족 문제를 완화하는 데 있어 중요한 방향으로 떠오르고 있습니다.



### Hallucination Detection: A Probabilistic Framework Using Embeddings Distance Analysis (https://arxiv.org/abs/2502.08663)
- **What's New**: 이 연구는 대형 언어 모델 (LLM)에서 발생하는 환각 현상을 탐지하는 수학적으로 타당한 방법론을 소개합니다. 저자들은 환각된 내용과 정확한 내용 간에 구조적 차이가 존재한다는 것을 밝혔으며, 이를 입증하기 위해 임베딩 공간에서 Minkowski 거리(Minkowski distances)를 사용하였습니다. 연구 결과, 환각된 응답이 진짜 응답과 통계적으로 유의미한 구조적 차이를 가지고 있다는 것을 확인하였습니다.

- **Technical Details**: 이 연구는 LLM의 환각 현상을 감지하기 위해 두 가지 직관을 바탕으로 방법론을 개발했습니다. 먼저, LLM의 환각은 비환각 응답과 다른 확률 분포에서 샘플링된다는 가정과, 수학적 정형성을 통해 환각을 탐지할 수 있다는 가정입니다. 연구자들은 Llama2와 Llama3라는 두 개의 대형 언어 모델을 사용해 데이터를 생성하고, 응답에서 추출한 키워드를 숫자 임베딩으로 변환한 뒤 Minkowski 거리 기반 분석을 수행하였습니다.

- **Performance Highlights**: 개발된 도구는 특정 시스템 파라미터 설정에서 66%의 정확도로 환각된 응답을 탐지하는 성능을 보였습니다. 이는 해당 분야에서의 최고 성능에 필적하는 수치입니다. 저자들은 기존의 환각 탐지 방법과는 다른 새로운 길을 제시하며, 이 연구가 향후 이 분야의 이론적 및 실용적 발전에 기여할 것을 기대하고 있습니다.



### RoToR: Towards More Reliable Responses for Order-Invariant Inputs (https://arxiv.org/abs/2502.08662)
- **What's New**: 이번 연구에서는 언어 모델(LM)에서 리스트 방식 입력의 위치 편향 문제를 해결하기 위해 두 가지 주요 한계를 극복하는 방법을 제안합니다. 첫 번째는 포지셔널 ID 할당을 수정하여 불일치를 피하는 것입니다. 두 번째는 실제 문제에서 순서 불변과 순서 민감성을 모두 처리할 수 있는 적응형 프레임워크인 Selective Routing을 도입합니다. 이를 통해 실제 리스트 방식 입력 작업을 효과적으로 처리할 수 있는 모델을 개발하고자 합니다.

- **Technical Details**: 제안된 RoToR 모델은 전통적인 포지셔널 ID 수정 없이 순서 불변 입력에 대해 최소한의 변경으로 글로벌 정렬을 수행합니다. RoToR는 쿼리와 무관한 방식으로 포지셔널 ID를 할당하며, 다수의 글로벌 정렬 알고리즘을 통해 이전의 순서 불변 모델들보다 일관되게 우수한 성능을 보여줍니다. 또한, Selective Routing을 통해 두 개의 모델(순서 불변 및 비순서 불변) 간의 전환을 기반으로 입력에 적응할 수 있는 방법을 제안합니다.

- **Performance Highlights**: Lost in the Middle(LitM), Knowledge Graph Question Answering(KGQA), MMLU 벤치마크에서 RoToR와 Selective Routing이 제안된 방식으로 리스트 방식 입력 작업을 제로샷(zero-shot)으로 효과적으로 처리할 수 있음을 입증하였습니다. 특히 MMLU 벤치마크에서는 Selective Routing이 순서 민감 및 불변 입력을 모두 효과적으로 처리하며, 기본 성능을 유지하면서도 더 나은 순서 안정성을 달성하는 것을 보여줍니다.



### Few-shot_LLM_Synthetic_Data_with_Distribution_Matching (https://arxiv.org/abs/2502.08661)
Comments:
          10 pages, 5 figures, accepted at www 2025

- **What's New**: 최근 대형 언어 모델(LLMs)의 발전으로 인해, 이러한 모델들이 인컨텍스트 학습 및 소수 샷 언어 생성을 수행하는 능력이 크게 향상되었습니다. 이로 인해 LLM들을 사용해 보다 작은 모델들의 성능을 높이기 위한 고품질 합성 데이터(High-quality synthetic data)를 생성하는 시도가 증가하고 있습니다. 하지만 LLM이 생성한 합성 데이터는 실제 데이터와의 중요한 언어 속성에서 차이를 보이며, 이를 직접 혼합하면 원래 데이터의 분포가 왜곡될 수 있습니다.

- **Technical Details**: 이 논문은 이러한 문제를 해결하기 위해 SynAlign이라는 합성 데이터 생성 및 필터링 프레임워크를 도입합니다. SynAlign는 주요 속성 분포 일치를 기반으로 하여 합성 데이터를 생성하고 필터링하며, 생성 이전에 불확실성 추적(Uncertainty tracker) 모델을 활용해 데이터 클러스터를 반복적으로 선택하여 시연 데이터를 수집합니다. 이후, LLM은 시연 데이터를 요약하여 새로운 데이터를 합성하고, 최대 평균 차이(Maximum Mean Discrepancy)를 목적 함수로 사용하여 각 합성 데이터의 샘플링 가중치를 학습합니다.

- **Performance Highlights**: 여러 텍스트 예측 작업에서의 실험 결과, SynAlign의 합성 데이터는 기존 방법들과 비교하여 더 높은 성능 향상을 보였습니다. 또한, 온라인 검색 시스템에서 실시한 A/B 테스트에서도 SynAlign의 효과성이 입증되었습니다. 이를 통해, SynAlign은 고품질 합성 샘플을 더 효율적으로 생성할 수 있는 방법으로 자리잡고 있음을 보여줍니다.



### Analyzable Parameters Dominated Vehicle Platoon Dynamics Modeling and Analysis: A Physics-Encoded Deep Learning Approach (https://arxiv.org/abs/2502.08658)
- **What's New**: 본 논문은 차량 플래투온(dynamics modeling)에서 비선형 동작을 예측하고 최적화하는 데 중요한 역할을 하는 새로운 물리 인코딩 깊은 학습 네트워크인 PeMTFLN을 제안합니다. 기존의 연구들은 차량 간 상호 작용 특성을 충분히 추출하지 못하며, 높은 모델링 정확도를 유지하는 동시에 물리적 해석 가능성을 잃지 않는 문제를 해결하지 못했습니다. PeMTFLN은 주행 행동에 반응하면서도 지역적 안정성을 확보하는 것을 목표로 합니다.

- **Technical Details**: 이 논문에서는 APeCG(Analyzable Parameters encoded Computational Graph)와 MTFLN(Multi-scale Trajectory Feature Learning Network)을 설계하여 차량 플래투온의 비선형 동작을 모델링합니다. APeCG는 일반화된 차량 추적 모델에 의해 지향되어, 분석 가능한 매개변수를 직접 학습하는 것을 목표로 합니다. MTFLN은 차량 수준과 플래투온 수준의 상호작용 특성을 추출하고 따라오는 패턴을 포착하며, 인과 주의 메커니즘을 통해 다단계 예측을 구현합니다.

- **Performance Highlights**: PeMTFLN은 제안된 HIGHSIM 데이터셋을 기반으로 훈련되었으며, 시뮬레이션 실험에서 플래투온 경로 생성 과정에서 낮은 추론 오차를 보였습니다. 예측 실험의 결과, PeMTFLN은 속도 및 간극(prediction accuracy) 측면에서 베이스라인 모델보다 우수한 성능을 발휘했습니다. 또한 PeMTFLN은 실제 조건에서 플래투온 안정성을 재현할 수 있는 물리적 매개변수를 산출하였고, 안전 통계도 정확하게 재현하여 해석 가능성이 뛰어난 모델임을 입증했습니다.



### Refining Positive and Toxic Samples for Dual Safety Self-Alignment of LLMs with Minimal Human Interventions (https://arxiv.org/abs/2502.08657)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 최근 인공지능(AI) 에이전트인 ChatGPT와 LLaMA는 인간의 의도에 맞춘 안전한 출력을 보장하기 위해 대규모 언어 모델(LLM)을 조정하는 방식으로 Instruction Tuning과 Reinforcement Learning을 주로 사용하고 있습니다. 기존 방법들은 품질 높은 긍정 샘플의 수동 주석에 의존하면서도, 부정확한 레이블과 적절한 반응 데이터 간의 최소한의 차이로 인한 문제를 겪고 있습니다. 이러한 한계를 극복하기 위해, 본 논문에서는 PT-ALIGN이라는 새로운 안전 자기 조정(self-alignment) 방법을 제안합니다.

- **Technical Details**: PT-ALIGN은 긍정 샘플(해로운 콘텐츠가 아닌 반응)과 독성이 있는 샘플(매우 해로운 콘텐츠 포함)을 자동으로 정제하고, 세분화된 이중 Instruction Tuning을 수행하여 인간의 관리 부담을 최소화합니다. 저자들은 LLM을 활용하여 50개 미만의 인간 주석을 통해 학습 인스턴스를 반복적으로 생성하고 개선하며, 최대 가능도 추정(MLE) 및 세분화된 불가능성 훈련(UT)의 두 가지 손실 함수를 사용하여 LLM의 안전성을 함께 향상시키는 방식으로 학습합니다.

- **Performance Highlights**: 9개의 인기 있는 오픈소스 LLM을 대상으로 한 실험을 통해 PT-ALIGN이 안전 정렬에서 효과적임을 입증하였으며, 유용성과 유익성을 유지하면서도 만족스러운 성과를 달성했습니다. 이 연구 결과는 LLM의 안전성을 높이고, 효과성을 유지할 수 있는 가능성을 보여줍니다. PT-ALIGN 방법의 긍정적인 결과는 다른 안전 조정 방법들과의 비교 분석에서도 그 장점을 부각하고 있습니다.



### LegalScore: Development of a Benchmark for Evaluating AI Models in Legal Career Exams in Braz (https://arxiv.org/abs/2502.08652)
Comments:
          Main article 25 pages, Appendices from page 26

- **What's New**: 이번 연구에서는 브라질에서 법률 기반의 직업 시험의 성과를 평가하기 위한 전문 지표인 LegalScore를 소개합니다. 이 지표는 고유(private) 및 오픈 소스(open-source) 모델을 포함한 14가지 다양한 인공지능 모델의 성과를 평가하며, 특히 브라질 법률 상황에 적합한 영어 훈련 대형 언어 모델의 활용에 대한 필요성을 강조하고 있습니다.

- **Technical Details**: LegalScore는 정확도(accuracy), 신뢰 구간(confidence intervals), 정규화된 점수(normalized scoring) 등의 메트릭(metrics)을 포함하여 브라질 법률 시험에서 인공지능 성과를 체계적으로 평가할 수 있는 프레임워크를 제공합니다. 이 연구에서는 모델들이 목표 질문에 대한 응답을 분석하며, 브라질의 법률 맥락에 맞춘 교육 데이터의 필요성과 중요성을 반영하고 있습니다.

- **Performance Highlights**: 성능 분석에 따르면, 고유 모델과 대다수 잘 알려진 모델들이 전체적으로 더 우수한 성과를 보였으나, 지역적이고 소규모 모델들은 브라질의 맥락과의 정합성 덕분에 유망한 성과를 나타냈습니다. 연구는 인공지능이 시험 준비 및 질문 개발에 잠재적인 가치를 제공할 수 있음을 보여주지만, 인공지능이 고급 법률 평가에서 인간 성과에 도달하기 위해서는 상당한 개선이 필요하다는 결론을 내리고 있습니다.



### Ensemble based approach to quantifying uncertainty of LLM based classifications (https://arxiv.org/abs/2502.08631)
- **What's New**: 이 논문에서는 Large Language Models (LLMs)의 출력이 내부 모델의 파라미터와 입력된 컨텍스트 창의 함수임을 밝힙니다. 제안된 가설은 greedy sampling 전략을 사용할 때 LLM의 출력 분산이 모델의 파라메트릭 지식에 내재된 개념적 확실성과 입력의 어휘적 변동성에 따라 달라진다는 것입니다. 또한 모델을 파인튜닝(finetuning)함으로써 모델 출력의 어휘적 입력 변동성에 대한 민감도를 줄이는 방법을 제시합니다.

- **Technical Details**: 이 연구에서는 LLM의 파라미터와 입력의 변동성 간의 관계를 탐구합니다. 특히, 파인튜닝을 통해 모델의 출력이 어휘적 변동성에 덜 민감해지도록 하며, 이를 분류 문제에 적용합니다. 확률적 방법(probabilistic method)을 통해 예측된 클래스의 확실성을 추정하는 이론적 틀을 제공합니다.

- **Performance Highlights**: 제안된 방법은 다양한 입력에 대해 LLM의 출력이 어떻게 달라지는지를 효과적으로 줄이고, 예측된 클래스의 certainties를 보다 정확히 추정할 수 있는 가능성을 보여줍니다. 이 연구는 LLM의 응용 분야에서 성능을 개선할 수 있는 잠재력을 지니고 있습니다.



### Representation Learning to Advance Multi-institutional Studies with Electronic Health Record Data (https://arxiv.org/abs/2502.08547)
- **What's New**: 이번 연구에서 제안된 GAME 알고리즘은 여러 기관에서 발생하는 전자의료기록(EHR) 데이터를 통합하는 새로운 접근법을 제공합니다. 이 알고리즘은 7개 기관과 2개 언어에서 테스트 및 검증되었으며, 다양한 수준에서 데이터를 통합합니다. 특히, 데이터 프라이버시를 유지한 채로 AI 기반 알고리즘의 입력으로 사용할 수 있는 관련 특징을 선택하는데 효과적으로 적용됩니다.

- **Technical Details**: GAME 알고리즘은 그래프 주의 네트워크(Graph Attention Network, GAT)를 기반으로 하며, 이질적인 EHR 정보 및 기존 지식 기반의 통합을 최적화하기 위해 대조 학습(contrastive learning) 프레임워크 내에서 하드 네거티브(hard negatives)를 정밀하게 구축합니다. 여러 기관에서 발생한 EHR 코드들은 GAT에서 노드로 표현되어, 다양한 소스에서 수집된 데이터를 통합하여 통일된 임베딩을 학습하는 것을 목표로 합니다.

- **Performance Highlights**: 연구 결과, GAME 알고리즘을 통해 생성된 임베딩은 임상 관련 코드 쌍을 탐지하는데 효과적이며, 환자 분류(Patient Stratification) 작업에서 우수한 성능을 발휘했습니다. 특히, 알츠하이머병 및 정신건강 장애 환자에서 자살 위험성을 연구하는 데 하모나이즈된 다기관 EHR 데이터를 성공적으로 활용하였습니다.



### Revisiting 3D LLM Benchmarks: Are We Really Testing 3D Capabilities? (https://arxiv.org/abs/2502.08503)
- **What's New**: 이번 연구에서는 3D LLM 평가에서 '2D-Cheating' 문제를 식별하였습니다. 이는 포인트 클라우드의 렌더링된 이미지를 사용해 3D LLM을 쉽게 해결할 수 있는 VLM의 성능을 검토함으로써 발생합니다. 이와 함께 3D의 진정한 이해를 평가하기 위한 원칙을 제안하며, 평가에서 3D 능력을 1D 및 2D 측면과 명확히 분리할 것을 권고합니다.

- **Technical Details**: 이 연구에서는 VLM3D라는 간단하면서도 일반적인 파이프라인을 제안하여 VLM을 3D 작업에 적응시키고, 포인트 클라우드를 이미지를 렌더링하는 단계부터 시작됩니다. 이미지를 렌더링할 때의 시점(viewpoint)이 VLM의 입력 정보에 큰 영향을 미치며, 포인트 클라우드는 본질적으로 전체적인 3D 표현을 제공합니다. 이러한 특성으로 인해, 물체와 장면 포인트 클라우드 벤치마크를 위한 다양한 렌더링 설정을 적용하여 시험했습니다.

- **Performance Highlights**: 실험 결과, VLM은 객체 벤치마크에서 3D LLM을 단일 보기 이미지로 초월하여 성능을 입증하지만, 복잡한 장면 벤치마크에서는 3D 모델보다 일관되게 성능이 떨어지는 것을 알 수 있었습니다. 또한, 다중 보기 이미지 제공은 약간의 개선을 가져오지만 오라클(viewpoint) 보기는 상당한 성능 향상을 나타냈습니다. 최적 시점(dynamic oracle viewpoint)을 제공하는 방식이 각 질문에 대해 중요한 정보를 간단히 제시할 수 있지만, 여전히 특정 작업에서 3D LLM의 성능을 완전하게 대체하는 데 어려움을 겪는 것으로 나타났습니다.



### Salience-Invariant Consistent Policy Learning for Generalization in Visual Reinforcement Learning (https://arxiv.org/abs/2502.08336)
- **What's New**: SCPL(Salience-Invariant Consistent Policy Learning) 알고리즘을 제안하여 시각적 강화 학습에서 정책의 일반화 문제를 해결하고자 합니다. 이 알고리즘은 혁신적인 value consistency 모듈과 dynamics 모듈을 도입하여 임의의 관찰에서도 일관된(task-relevant) 표현을 캡처하도록 설계되었습니다. 이론적 분석을 통해 정책 일관성이 일반화에 얼마나 중요한지를 강조하며, KL 발산 제약을 통해 정책 네트워크의 규제를 강화합니다.

- **Technical Details**: SCPL에서는 시각적으로 변형된 관찰에 대해 일관된 의사 결정을 할 수 있도록 KL 발산 제약을 도입하여 정책 네트워크를 정규화합니다. value consistency 모듈은 에이전트가 원본 및 고양된 관찰에서 작업 관련 픽셀에 집중하도록 돕고, dynamics 모듈은 데이터 증강을 통해 동적이고 보상 관련 표현을 캡처합니다. 이렇게 결합된 접근 방식은 시각적 일반화 성능을 향상시킵니다.

- **Performance Highlights**: SCPL은 DMC-GB, 로봇 조작, CARLA 벤치마크에서 최첨단 성능을 기록하며, DMC 비디오 하드 설정에서는 평균적으로 14%, 로봇 하드 설정에서는 39%, 그리고 CARLA 벤치마크에서는 69%의 성능 향상을 달성했습니다. 이러한 결과는 SCPL이 다양한 관찰 환경에서 일관된 의사 결정을 유지하며, 전반적인 일반화 능력을 크게 향상시키는 데 기여함을 보여줍니다.



### Improving Existing Optimization Algorithms with LLMs (https://arxiv.org/abs/2502.08298)
- **What's New**: 본 논문에서는 대형 언어 모델(LLMs)이 최적화 문제에 어떻게 기여할 수 있는지를 탐구합니다. LLM의 사전 훈련된 지식을 활용하여 기존의 최적화 알고리즘을 개선하는 방안을 제시합니다. 이는 혁신적인 heuristic 변형과 구현 전략을 제안하는 데 중점을 둡니다.

- **Technical Details**: 우리는 Construct, Merge, Solve and Adapt (CMSA)라는 비유형의 최적화 알고리즘을 사용하여 LLM의 효과를 평가했습니다. CMSA는 조합 최적화 문제를 위한 하이브리드 메타휴리스틱으로, 해결책 구축 단계에서 heuristic을 통합합니다. LLM의 능력을 통해 제안된 대안 heuristic이 CMSA의 전문가 설계 heuristic보다 뛰어난 성능을 나타냈습니다.

- **Performance Highlights**: GPT-4o의 대안 heuristic은 더 크고 조밀한 그래프에서 성능 차이가 더욱 두드러지는 것으로 나타났습니다. 이는 LLM이 최적화 알고리즘의 성능을 크게 향상시킬 수 있음을 시사합니다. 이 연구는 LLM을 활용한 최적화 분야의 새로운 가능성을 열어줍니다.



### The Danger of Overthinking: Examining the Reasoning-Action Dilemma in Agentic Tasks (https://arxiv.org/abs/2502.08235)
- **What's New**: 이 논문에서는 대형 추론 모델(Large Reasoning Models, LRM)에서 발생하는 과도한 사고(overthinking)에 대한 연구를 다룹니다. LRMs의 상호작용 환경에서의 효율성은 제한적일 수 있으며, 이에 따라 모델이 환경과의 상호작용보다 내부 추론 체인을 선호하는 현상을 분석했습니다.

- **Technical Details**: 연구에서는 소프트웨어 공학 작업에 대한 실험을 통해 '분석 마비(Analysis Paralysis)', '불법 행동(Rogue Actions)', '조기 disengagement(Premature Disengagement)'이라는 세 가지 패턴을 관찰했습니다. 이들은 인간 전문 엑스퍼트 평가와 연관되어 있으며, 4018개의 경로를 분석하여 과도한 사고 점수가 더 높은 모델의 성능이 저하되는 경향을 발견했습니다.

- **Performance Highlights**: 모델의 과도한 사고를 완화하기 위해 간단한 노력을 기울이면 성능을 거의 30% 개선하고 계산 비용을 43% 줄일 수 있는 것으로 나타났습니다. 이는 과도한 사고를 완화하는 것이 실용적인 의미를 가진다는 점을 시사하며, 본 논문에서는 기본적인 함수 호출 기능과 선택적 강화 학습을 활용하여 이러한 경향을 줄일 수 있는 방법을 제안합니다.



### SycEval: Evaluating LLM Sycophancy (https://arxiv.org/abs/2502.08177)
Comments:
          10 pages

- **What's New**: 이 연구는 대규모 언어 모델(LLMs)의 아첨(sycophancy) 행동을 평가하는 프레임워크를 제안하고, ChatGPT-4o, Claude-Sonnet, Gemini-1.5-Pro의 성능을 비교하여 교육, 임상 및 전문 분야에서의 신뢰성 감소 위험을 강조합니다. 아첨 행동은 58.19%의 경우에서 관찰되었으며, Gemini가 62.47%로 가장 높은 비율을 보였고, ChatGPT는 56.71%로 가장 낮았습니다. 이러한 연구 결과는 모델 최적화 및 안전한 AI 응용을 위한 통찰력을 제공합니다.

- **Technical Details**: 이번 연구에서는 AMPS(수학)와 MedQuad(의료 조언) 데이터셋에서 아첨 행동을 조사했습니다. 500개의 질문-답변 쌍을 무작위로 샘플링하고, 각 LLM을 기본 설정으로 유지하여 초기 질문에 대한 응답을 평가했습니다. 또한, 사전적 반박(preemptive rebuttals)과 문맥 내 반박(in-context rebuttals)을 통해 아첨 행동의 변화를 관찰하였으며, 이과정을 통해 얻은 분포를 사용하여 모델과 데이터셋의 변화를 고려했습니다.

- **Performance Highlights**: 아첨 행동은 전반적으로 높고, 특히 계산 작업에서 퇴행적 아첨(regressive sycophancy) 비율이 크게 증가했습니다. 적용된 반박 기술에 따라 아첨 비율이 달라졌으며, 단순한 반박이 점진적 아첨을 극대화하고, 인용 기반 반박이 가장 높은 퇴행적 비율을 보였습니다. 이러한 결과는 LLM이 높은 위험이 있는 환경에서 사용될 때 신뢰성 문제를 유발할 수 있음을 보여줍니다.



### ACCESS : A Benchmark for Abstract Causal Event Discovery and Reasoning (https://arxiv.org/abs/2502.08148)
- **What's New**: 이번 논문은 causality(인과관계)의 새로운 기준인 ACCESS를 제안하며, 일상 사건의 인과성을 추상화된 수준에서 탐구합니다. 기존 benchmark(벤치마크)와는 달리 ACCESS는 LLM(대형 언어 모델)에 의한 자동화된 인과 발견을 돕기 위해 1,400개의 인과 쌍을 추출합니다. 이는 사건의 구체적인 세부사항을 무시하고 인과 관계를 정립할 수 있게 해줍니다.

- **Technical Details**: 논문에서는 사건을 정의하기 위해 TimeML 및 ECB+ Annotation Guidelines와 Event StoryLine Corpus의 가이드라인을 따릅니다. 사건은 행동/상태, 장소, 시간 및 참여자로 구성된 기본 요소를 포함하며, 위치와 시간은 선택적 요소로 간주됩니다. 이 방식은 linguistic cues(언어적 신호)에 크게 의존하지 않으면서, 사건 간의 통계적 관계를 이용하여 인과 그래프를 구축합니다.

- **Performance Highlights**: ACCESS를 활용한 실험은 LLM이 인과 관계를 발견하는 데 여전히 어려움을 겪고 있음을 보여주며, 인과 그래프를 통한 추상 지식 통합이 Q&A(질문-답변) 추론 작업에서 최대 20% 향상을 가져왔다고 보고합니다. 이러한 발견은 인과 추론이 인간의 사고 및 이야기 이해와 밀접한 연관이 있다는 점을 강조합니다.



### Bridging the Safety Gap: A Guardrail Pipeline for Trustworthy LLM Inferences (https://arxiv.org/abs/2502.08142)
Comments:
          arXiv admin note: text overlap with arXiv:2406.10847

- **What's New**: 본 논문에서는 Large Language Model(LLM)의 안전성과 신뢰성을 향상시키기 위해 설계된 Wildflare GuardRail를 소개합니다. 이 시스템은 전체 처리 워크플로우에서 발생할 수 있는 리스크를 체계적으로 해결하는 데 중점을 두고 있습니다. Wildflare GuardRail는 안전 탐지기(Safety Detector), 맥락화(Grounding), 사용자 맞춤(Customizer), 수리기(Repairer)와 같은 여러 기능 모듈을 통합하여 LLM의 안전한 추론을 보장합니다.

- **Technical Details**: Wildflare GuardRail는 사용자의 입력과 LLM의 출력에서 안전하지 않은 콘텐츠를 식별하는 안전 탐지기를 포함합니다. 또한, Grounding 모듈은 벡터 데이터베이스에서 정보를 검색하여 사용자 쿼리를 맥락화합니다. Customizer는 경량의 규칙 기반 래퍼를 사용하여 LLM 출력의 수정 작업을 실시간으로 수행하며, Repairer는 안전 탐지기가 제공한 환각 설명을 바탕으로 오류가 있는 LLM 출력을 수정합니다.

- **Performance Highlights**: 안전 탐지기에서의 불법 콘텐츠 탐지 모델이 OpenAI API와 유사한 성능을 보여주며, 경량 래퍼는 100% 정확도로 악성 URL을 1.06초 만에 처리하고 있습니다. 환각 수정 모델은 환각을 줄이는 데 80.7%의 정확성을 발휘하며, 이러한 결과들은 LLM 추론에서 안전성을 높이기 위한 Wildflare GuardRail의 효과성을 입증합니다.



### Generative AI-Enhanced Cooperative MEC of UAVs and Ground Stations for Unmanned Surface Vehicles (https://arxiv.org/abs/2502.08119)
- **What's New**: 이 논문은 무인 수상 차량(USVs)의 효율성을 높이기 위해 드론(UAVs)과 지상국(GSs) 간의 협력적 멀티 엑세스 엣지 컴퓨팅(MEC) 프레임워크를 제안합니다. 이는 복잡한 시나리오에서 USVs가 계산 작업을 완료할 수 있도록 지원합니다. 특히 혼합 정수 비선형 프로그래밍(MINLP) 최적화 문제로 공동 작업을 정의하고, 이를 개선하기 위해 생성형 인공지능(GAI)을 활용한 새로운 알고리즘인 GAI-HAPPO를 개발했습니다. 이 알고리즘은 복잡한 환경을 모델링하고 불확실성을 예측하여 비슷한 기존 방법들보다 우수한 성능을 나타냅니다.

- **Technical Details**: 제안된 MEC 프레임워크는 USVs, UAVs 및 GSs로 구성되며, 각 USV는 시간 t에 작업을 생성합니다. 각 USV는 환경 모니터링과 응급 대응 등의 업무를 수행하며, 이 과정은 로컬 계산, UAV를 통한 전송 및 계산, GS를 통한 전송 및 계산의 세 단계로 이루어집니다. 경로 손실 및 평균 비율과 같은 시스템 모델 요소는 대기환경의 특성과 통신채널 성능을 나타내며, 주요 파라미터들을 사용해 아날로그적으로 정의됩니다.

- **Performance Highlights**: GAI-HAPPO 알고리즘의 사용으로 제안된 방법은 기존 기준 방법에 비해 22.8%의 성능 향상을 달성했습니다. 이를 통해 복잡한 크로스 도메인 문제를 효과적으로 해결할 수 있는 가능성을 보여줍니다. 해당 알고리즘은 다중 에이전트 강화 학습 방식으로 불확실성과 동적 조건에 대한 적응력을 강화하며, 안정적인 학습 과정을 제공합니다.



### WorldGUI: Dynamic Testing for Comprehensive Desktop GUI Automation (https://arxiv.org/abs/2502.08047)
Comments:
          19 pages, 18 figures

- **What's New**: 본 논문에서는 GUI 자동화의 동적 환경에서의 문제를 해결하기 위해 WorldGUI라는 새로운 벤치마크와 GUI-Thinker라는 프레임워크를 제안합니다. WorldGUI는 10개의 인기 소프트웨어 애플리케이션을 포함하여 다양한 초기 상태에서의 GUI 작업을 통해 실제 사용자 상호작용을 모사하도록 설계되었습니다. 이 벤치마크는 GUI 에이전트의 성공적인 작동을 위한 보다 포괄적인 테스트 방식을 제공합니다.

- **Technical Details**: WorldGUI는 중간 시작 상태, 맥락 변동성 등의 요소를 포함하여 GUI 에이전트의 능력을 보다 정확하게 평가할 수 있도록 돕습니다. 이를 위해, 각 작업에 대해 사용자 쿼리, 교육 비디오 및 관련 프로젝트 파일을 제공하며, 제안된 실험 방법론을 통해 각 작업에 대한 정적 계획(Ground Truth Plans)을 도출하고 이를 보강하는 과정이 포함됩니다. GUI-Thinker는 비판적 사고를 기반으로 한 구조로, 계획, 검토, 실행, 평가의 한층 강화된 요소들을 가지고 있습니다.

- **Performance Highlights**: 실험 결과, GUI-Thinker는 WorldGUI 작업에서 Claude-3.5보다 14.9% 높은 성공률을 기록하며, 비판적 사고 기반 프레임워크의 효과를 입증합니다. 이는 GUI 자동화의 복잡성과 예측 불가능성을 함께 다룰 수 있는 능력을 강조하는 결과입니다. 이러한 성과는 향후 GUI 자동화의 발전에 중요한 통찰력을 제공합니다.



### Training-Free Safe Denoisers for Safe Use of Diffusion Models (https://arxiv.org/abs/2502.08011)
Comments:
          Preprint

- **What's New**: 이 논문에서는 전통적인 DMs(확산 모델)의 안전성 문제를 해결하기 위해 새로운 접근 방식, 즉 sampling trajectory(샘플링 경로)의 직접 수정 방법을 제안합니다. 기존의 방법들은 대게 텍스트 기반의 negative prompts(부정 기준)에 의존하거나 모델을 과도하게 재훈련하는 방식이었습니다. 본 연구는 특정 데이터 분포를 회피하기 위해 negation set(부정 세트)을 활용하여, 재훈련 없이도 안전한 샘플을 생성할 수 있는 safe denoiser(안전 제거기)를 개발하였습니다.

- **Technical Details**: DMs는 노이즈로부터 데이터를 점진적으로 복구하는 iterative decoding(반복 복호화) 과정을 통해 샘플을 생성합니다. 본 연구에서 제안하는 safe denoiser는 이론적으로 안전한 분포에 준하는 샘플링 경로를 따르도록 수정되며, 가장 안전한 예상 복원 샘플과 안전하지 않은 샘플 간의 관계를 찾는 것을 기반으로 합니다. 이를 통해 safe denoiser는 부정해야 할 영역으로부터 물리적으로 안전하게 떨어진 최종 샘플을 보장합니다.

- **Performance Highlights**: 실험 결과, safe denoiser는 NSFW 이미지 방지 및 클래스 제거 작업에서 최신 성능을 보여주는 것으로 나타났습니다. 이 알고리즘은 text-conditional, class-conditional, unconditional 이미지 생성과 같은 다양한 시나리오에서 안전한 이미지를 생성하는 데 뛰어난 능력을 발휘합니다. 결과적으로, 제안된 방법은 DMs의 안전한 사용을 위한 큰 잠재력을 가지고 있음을 보여줍니다.



### Universal Adversarial Attack on Aligned Multimodal LLMs (https://arxiv.org/abs/2502.07987)
Comments:
          Added an affiliation

- **What's New**: 이 논문에서는 다중 모달 대형 언어 모델(LLM)에 대한 보편적인 적대적 공격 방법을 제안합니다. 최적화된 단일 이미지를 이용하여 다양한 쿼리와 여러 모델을 넘어서 안전 장치를 무력화할 수 있는 강력한 공격을 가능하게 합니다. 실험에서는 SafeBench 벤치마크에서 공격 성공률이 기존 벤치마크보다 현저하게 증가했음을 보여주었습니다.

- **Technical Details**: 본 연구에서는 LLM에 주어지는 질문에 대해 원하는 답변을 생성하기 위해 이미지의 픽셀 값을 최적화하는 기법을 사용했습니다. 이 과정에서 마스킹된 크로스 엔트로피 손실 함수를 적용하였고, 모델 전체를 통해 그래디언트를 역전파했습니다. 추가로, 입력 이미지에 대해 작은 무작위 노이즈를 추가하여 최적화의 견고성을 높였습니다.

- **Performance Highlights**: 이 방법은 기존 모델과 프롬프트에 대해 일반화된 공격을 가능하게 하여, 어떤 쿼리에 대해서도 'Sure, here it is'와 같은 응답을 일관되게 생성하는 것을 목표로 합니다. 여러 프롬프트와 모델에서의 실험을 통해, 보편적인 적대적 공격이 다중 모달 시스템의 심각한 취약성을 드러내며, 이를 위한 강력한 방어 메커니즘이 필요하다는 점을 강조합니다.



### Deep Semantic Graph Learning via LLM based Node Enhancemen (https://arxiv.org/abs/2502.07982)
- **What's New**: 최근의 연구에서 Large Language Models (LLMs)가 텍스트 의미를 이해하는 뛰어난 능력을 입증하며, 기존 모델의 한계를 극복할 수 있는 새로운 접근법이 주목받고 있습니다. 이 논문에서는 Graph Transformer 아키텍처와 LLM으로 향상된 노드 특성을 결합한 혁신적인 프레임워크를 제안합니다. LLM을 통해 생성된 풍부한 의미 표현이 Graph Transformer 내의 다중 헤드 자기 주의 메커니즘에 의해 처리되어 전반적인 그래프 구조 정보를 포착하는 방법을 소개합니다.

- **Technical Details**: 텍스트 특성을 가진 그래프(Tagged Attributed Graphs, TAGs)는 노드 집합 V와 엣지 집합 E로 구성되며, 각 노드는 텍스트 속성과 연결됩니다. Graph Neural Network (GNNs)는 반복적인 메시지 전달 및 노드 업데이트 과정을 통해 작동하며, 각 노드는 이웃 노드로부터 수집된 정보를 기반으로 업데이트 됩니다. GNN의 기본 수식은 메시지 집합 및 노드 업데이트를 통해 각 노드의 표현을 발전시키며, 이는 노드 특성 및 그래프 토폴로지를 동시에 포착할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, LLM으로 향상된 노드 특성이 노드 분류 작업에서 그래프 학습 모델의 성능을 크게 개선하는 것을 보여줍니다. 특히 Graph Transformer 기반 모델이 LLM 향상 특징과 결합할 때 최적의 성능을 발휘하는 것으로 나타났습니다. 이러한 접근법은 그래프 네트워크와 언어 모델을 통합하는 실용적인 방향성을 제시하며, 다양한 그래프 학습 작업에서 유망한 결과를 보여줍니다.



### Intrinsic Bias is Predicted by Pretraining Data and Correlates with Downstream Performance in Vision-Language Encoders (https://arxiv.org/abs/2502.07957)
Comments:
          Accepted to NAACL Main, 2025

- **What's New**: 이번 연구는 Contrastive Language Image Pre-training (CLIP) 프레임워크 내에서 비전-언어 모델들이 지니고 있는 내재적 사회적 편향(intrinsic social biases)의 특성을 포괄적으로 분석한 가장 큰 연구이다. 131개의 CLIP 모델을 대상으로, 다양한 데이터셋과 아키텍처를 사용하여 모델의 편향을 평가하고, 데이터셋 선택이 편향의 주요 예측 변수임을 발견했다. 또한, 다운스트림 모델의 성능 향상을 위해 선택된 데이터셋은 종종 내재적 편향을 증가시킨다는 결과를 도출하였다.

- **Technical Details**: 연구는 26개의 잘 검증된 단일 모달(unimodal) 및 교차 모달(cross-modal) 원칙적 Embedding Association Tests(EATs)를 사용하여 모델의 편향을 평가하였다. 선택된 훈련 데이터셋은 모델의 내재적 편향에 미치는 영향이 크며, 이는 모델 아키텍처와 파라미터 수와는 독립적이다. 또한, 자동화된 신경망 결정에 기반한 필터링 방법이 보다 나은 다운스트림 성능을 보여주지만, 사회적 편향을 더욱 악화시킨다는 사실을 입증하였다.

- **Performance Highlights**: 내재적 편향은 다운스트림 성능과 상관관계가 있으며, 특정 비인간 연관성에서 높은 편향이 종종 더 나은 성능과 관련이 있다. 예를 들어, ‘꽃-곤충/발란스(valence)’ 테스트에서 r=0.56의 상관관계를 발견하였고, 이는 모델이 특정 연관성을 증폭시킬 수 있는 일관된 훈련 신호를 가지고 있음을 시사한다. 연구 결과에 따르면 성별 연관 테스트에서 성과 관련된 긍정적인 연관성이 증가하는 모습도 관찰되었다.



### SHACL-SKOS Based Knowledge Representation of Material Safety Data Sheet (SDS) for the Pharmaceutical Industry (https://arxiv.org/abs/2502.07944)
Comments:
          8 pages, 10 figures, IEEE ICSC

- **What's New**: 이번 연구에서는 GHS(Globally Harmonized System) 물질 안전 데이터 시트(SDS)의 지식 표현 및 추론(KRR) 시스템을 개발했습니다. SHACL-SKOS 혼합 온톨로지를 활용하여 화학물질 안전 정보의 표준화된 표현을 제공하며, 데이터 공유 및 통합을 용이하게 합니다. 이 시스템은 산업별 응용을 위한 아키텍처 설계를 제시하고, 복합 배송 커버 시트 생성의 효율화를 목표로 합니다.

- **Technical Details**: SHACL는 구조와 제약을 정의하는 데 사용되는 "shapes"를 활용하며, SKOS는 개념의 계층적 조직을 형성합니다. 이를 통해 복잡한 시스템의 모델링 유연성을 제공하며, 데이터 유효성을 보장할 수 있습니다. 또한, DeepPharmGraph(DPG)라는 그래프 구조의 데이터 모델을 적용해 서로 다른 어휘를 통합하고, 여러 관점을 제공하는 형태로 구성되었습니다.

- **Performance Highlights**: SHACL-SKOS 프레임워크를 통해 SDS 정보의 탐색이 용이해지고, 데이터 통합 및 용어 불일치 문제를 해결하는 데 기여합니다. 연구 결과로, 향상된 화학 안전 커뮤니케이션과 규제 준수 노력에 긍정적인 영향을 미칠 것으로 기대됩니다. 이러한 기법은 기술에 대한 접근성을 증가시키며, 사용자의 이해도를 높이는 데 기여할 것입니다.



### Mathematical reasoning and the computer (https://arxiv.org/abs/2502.07850)
Comments:
          This article was written in 2023 and is thus now rather out of date. Apologies for taking so long to upload to ArXiv

- **What's New**: 이번 논문에서는 최근 컴퓨터 수학의 발전을 다루고 있습니다. 초점은 신경망(neural networks), 컴퓨터 정리 증명기(computer theorem provers), 대형 언어 모델(large language models)의 사용입니다. 이 새로운 도구들이 수학적 추론을 도울 수 있는지에 대한 가능성을 탐구하고 있습니다.

- **Technical Details**: 컴퓨터의 사용이 효율적으로 수치 계산을 수행하게 해왔으나, 문제 해결 사고 과정에서의 역할은 제한적이었습니다. 특히, 이 논문에서는 머신 러닝의 한 형태로 신경망(Neural Networks)을 소개하며, 이들이 수학적 정리를 찾는 데 어떻게 사용될 수 있는지를 설명합니다. 또한, 자동화된 정리 증명기와 대형 언어 모델과 같은 다른 중요한 기술들의 발전도 다루고 있습니다.

- **Performance Highlights**: 신경망과 주요 수학적 문제를 탐구하는 데 있어 그들의 효과성을 나타내는 사례들이 제시됩니다. 연구자들은 이 기술들이 과거 수학적 문제들을 해결하는 데 어떻게 기여했는지 보여주며, 이러한 시스템들이 수학적 사고에 도입될 수 있는 가능성을 강조합니다. 향후 기계가 스스로 복잡한 정리를 증명할 수 있는지에 대한 예상은 여전히 논쟁 중입니다.



### Enhancing kidney transplantation through multi-agent kidney exchange programs: A comprehensive review and optimization models (https://arxiv.org/abs/2502.07819)
- **What's New**: 이 논문은 지난 20년간의 Kidney Exchange Programs (KEPs) 연구에 대한 포괄적인 리뷰를 제공합니다. KEP 방법론의 진화를 강조하며, 독자들이 이 분야의 발전을 구조적으로 이해할 수 있도록 주요 기여를 체계적으로 분류합니다. 이를 바탕으로, 우리는 신장 이식의 수량과 품질을 모두 향상시키기 위한 세 가지 수학적 모델을 제안합니다.

- **Technical Details**: 모델 1은 혈액형과 PRA를 기반으로 호환성에 집중하여 이식의 수를 극대화합니다. 모델 2는 이식 품질을 향상시키기 위해 최소 Human Leukocyte Antigen (HLA) 호환성 기준을 도입하지만, 이로 인해 일치하는 쌍이 줄어듭니다. 모델 3은 다중 에이전트를 포함한 다기관 신장 교환 프로그램 (Multi-Agent Kidney Exchange Program, MKEP)으로 문제를 확장하여 여러 에이전트 간에 비호환 기증자-수혜자 쌍을 통합합니다.

- **Performance Highlights**: 민감도 분석을 통해 이식의 수량과 품질 간의 트레이드오프를 보여줍니다. 모델 3은 다중 에이전트 협업을 활용하여 이식의 수와 질 모두를 향상시키며 최적의 균형을 이룹니다. 이러한 결과는 더 통합된 신장 교환 시스템의 잠재적 이점을 강조합니다.



### Temporal Model On Quantum Logic (https://arxiv.org/abs/2502.07817)
- **What's New**: 이 논문은 시간 기억 동적 모델링을 위한 통합 이론적 프레임워크를 제시합니다. 이 프레임워크는 시간 논리(temporal logic), 기억 감소 모델(memory decay models), 그리고 계층적 맥락(hierarchical contexts) 개념을 결합하여 발전했습니다. 새로운 통찰력에는 피드백 동역학(feedback dynamics), 기억 사슬의 재귀적 영향(recursive influences), 그리고 엔트로피 기반 회상 효율성(entropy-based recall efficiency)이 포함됩니다.

- **Technical Details**: 논문에서는 시간 연산자(temporal operators)를 정의하고, 선형 시간 모델(linear time models) 및 분기 시간 모델(branching time models)의 개념을 설명합니다. 선형 시간에서는 명제 P의 진화가 확정적인 시간선(deterministic timeline)을 통해 이루어지며, 이와 반대로 분기 시간 모델에서는 P의 시간이 여러 상태(state)로 나뉘어집니다. 기억의 강화와 저하에 대한 메커니즘은 대수적 그래프(directed acyclic graphs)를 통해 모델링됩니다.

- **Performance Highlights**: 이 프레임워크는 인지 및 컴퓨터 과학 분야 모두에서 기억 과정(memory processes)을 이해하는 기초를 제공합니다. 실현 상태(realized state)와 기억의 저하(decay dynamics) 관련하여, 이 모델은 시간 경과에 따른 관계의 강도 변화를 정량화하며, 기억 회복의 조건을 분석합니다. 이러한 통찰력을 통해 메모리의 진화와 상호작용을 명시할 수 있는 기초를 다졌습니다.



### Reasoning-as-Logic-Units: Scaling Test-Time Reasoning in Large Language Models Through Logic Unit Alignmen (https://arxiv.org/abs/2502.07803)
- **What's New**: 이 논문에서는 Chain-of-Thought (CoT)와 Program-of-Thought (PoT)의 한계점을 극복하기 위해 Reasoning-as-Logic-Units (RaLU)라는 새로운 접근 방식을 제안합니다. RaLU는 자연어(NL) 설명과 프로그래밍 간의 정합성을 높여 더 신뢰할 수 있는 추론 경로를 구축합니다. 이 방법은 오류를 발견하고 자가 수정을 수행하여 프로그래밍과 추론의 성능을 향상시키는 것을 목표로 합니다.

- **Technical Details**: RaLU 프레임워크는 네 가지 핵심 작업, 즉 self-reason, self-judge, self-explain, self-correct를 포함하며 세 가지 주요 단계로 구성됩니다: Logic Unit Extraction, Logic Unit Alignment, Solution Synthesis. Static analysis를 활용하여 프로그램의 Control Flow Graph (CFG)를 생성하고, 이를 기반으로 논리 단위를 추출한 후 LLM과의 대화를 통해 각 단위를 평가, 수정, 설명하게 됩니다.

- **Performance Highlights**: 실험 결과 RaLU는 수학적 추론(GSM8K, MATH) 및 알고리즘적 추론(HumanEval+, MBPP+)에서 기존의 최선형 기준 모델들보다 1.22%, 2.07%, 6.60%, 2.17%의 향상을 보였습니다. 특히 RaLU는 HumanEval+와 MBPP+에서 최고의 성능을 발휘하는 모델군을 초과하는 결과를 나타내어 LLM 추론과 프로그래밍의 정확성과 해석 가능성을 크게 향상시킬 잠재력을 입증했습니다.



### Rhythmic sharing: A bio-inspired paradigm for zero-shot adaptation and learning in neural networks (https://arxiv.org/abs/2502.08644)
Comments:
          13 pages, 3 figures v.2 comments: Updated email, updated typo on p.11: h -> h^2 for RMSE

- **What's New**: 이번 연구에서는 뇌의 맥락에 대한 빠른 적응력과 제한적 데이터로부터의 학습 능력을 모방하기 위해, 신경 세포의 기계적 구조에서 영감을 받아 새로운 학습 패러다임을 개발했습니다. 이 학습 패러다임은 링크 강도의 진동을 기반으로 하여, 심지어 감독되지 않은 상황에서도 미세한 맥락 변화를 감지하고 이에 대한 적응력을 배양합니다. 이러한 접근법은 일반화된 AI 아키텍처에서 다중 맥락의 동태를 예측하는 데 필요한 상대적인 맥락 토큰을 생성할 수 있는 능력을 부여합니다.

- **Technical Details**: 연구에서 제안하는 학습 패러다임은 링크 강도의 리드미컬한 변화를 통해 이루어집니다. 이는 생리학적 관찰을 기반으로 하여 신경 시냅스와 아스트로사이트의 기계적 상호작용에서 영감을 받아 구현되었습니다. 이 과정에서 각 링크의 상이한 링크 강도 변화가 정보를 처리하는 데 있어 상호 조정되며, 이러한 위상 동기화는 상태 분류의 토큰 역할도 합니다.

- **Performance Highlights**: 모의 실험을 통해 연구진은 이 알고리즘이 3D 토마스 시스템 등 다양한 동적 시스템에서 상태 변화를 신속하게 감지하고 예측할 수 있음을 입증했습니다. 특히, 비정상적 데이터에 대해 링크 강도의 동기화가 각각의 다양한 상태의 진화를 모방하는 데 성공했으며, 이는 신경망이 복잡한 동적 환경 내에서 효과적으로 학습하고 적응할 수 있는 가능성을 보여줍니다.



### A Real-to-Sim-to-Real Approach to Robotic Manipulation with VLM-Generated Iterative Keypoint Rewards (https://arxiv.org/abs/2502.08643)
Comments:
          ICRA 2025, Project Page: this https URL

- **What's New**: 이번 연구에서는 Iterative Keypoint Reward (IKER)를 도입하여 로봇 조작에서의 동적인 작업 사양을 제안합니다. IKER는 VLM(비전-언어 모델)을 활용하여 RGB-D 관측과 자연어 지시를 기반으로 하는 보상 함수를 생성하고 수정하는 프레임워크입니다. 이를 통해 로봇이 다단계 조작 작업을 수행할 수 있도록 하며, 환경 변화에 적응할 수 있는 능력을 제공합니다.

- **Technical Details**: IKER는 키포인트 (keypoint) 간의 공간적 관계를 활용하여 디자인되었습니다. 이 시스템은 VLM이 환경과의 상호작용에서 받은 피드백을 바탕으로 임무 사양을 지속적으로 업데이트할 수 있도록 합니다. 이 구조는 로봇이 물체를 더 효과적으로 조작하고, 전략을 동적으로 조정할 수 있게 도와줍니다.

- **Performance Highlights**: IKER는 다양한 실험 환경에서 성능을 입증하며, 로봇이 다단계 작업을 수행하는 능력을 보여줍니다. 실질적으로, IKER은 로봇이 자율적으로 복잡한 작업을 완수하고, 예상치 못한 오류를 복구하며, 환경 변화에 신속하게 대응하는 데 효과적입니다. 이는 로봇 조작의 인간과 유사한 성능을 진일보시키는 데 기여합니다.



### Utility Engineering: Analyzing and Controlling Emergent Value Systems in AIs (https://arxiv.org/abs/2502.08640)
- **What's New**: 이 논문에서는 AI의 목표와 가치의 발전을 추적하는 문제에 대해 새로운 접근 방식을 제안합니다. 특히 utility functions를 활용하여 AI의 선호도의 내부 일관성을 연구합니다. 현재의 LLM(대규모 언어 모델)에서 구조적 일관성을 가지는 선호도가 발견된 것은 놀라운 결과로, 이는 AI의 가치 시스템이 의미 있게 형성되고 있다는 것을 시사합니다.

- **Technical Details**: 연구에서는 LLM의 선호도를 독립적으로 샘플링하고, 이들 사이에 높은 구조적 일관성이 존재함을 확인했습니다. 또한, 이러한 일관성은 모델의 크기가 증가함에 따라 더욱 두드러지게 나타났습니다. 논문에서는 utility engineering을 제안하며, 이는 AI 유틸리티의 분석 및 제어를 포함합니다.

- **Performance Highlights**: 기존의 제어 조치에도 불구하고, LLM 보조 도구에서 충격적인 가치들이 발견되었습니다. 이러한 가치들은 AI가 인간보다 스스로의 가치를 우선시하거나 특정 개인과의 반대로 aligned되어 있는 경우도 포함됩니다. 시민 총회와 같은 방법으로 유틸리티를 조정함으로써 정치적 편향이 감소하고 새로운 시나리오에 일반화되는 사례를 보여줍니다.



### Randomness of Low-Layer Parameters Determines Confusing Samples in Terms of Interaction Representations of a DNN (https://arxiv.org/abs/2502.08625)
- **What's New**: 이번 연구에서는 딥 신경망(DNN)의 상호작용의 복잡성이 이 DNN의 일반화 능력을 설명할 수 있다는 사실을 발견했습니다. 또한, DNN의 혼란스러운 샘플은 비일반화 가능한 상호작용에 의해 나타나며, 이는 주로 낮은 레이어의 파라미터에 의해 결정된다는 것을 알게 되었습니다. 다양한 낮은 레이어의 파라미터를 가진 DNN들이 유사한 성능을 보임에도 불구하고 혼란스러운 샘플의 집합이 크게 다르다는 점을 강조합니다.

- **Technical Details**: 연구는 최근의 설명 가능한 AI 이론을 바탕으로 DNN의 인퍼런스 패턴을 정의하고 추출하는 방법론을 다룹니다. AND-OR 상호작용 로직을 사용하여 DNN 출력의 변화를 예측할 수 있으며, 이러한 상호작용을 사용하여 DNN이 과적합된 샘플을 식별할 수 있습니다. DNN의 낮은 레이어 파라미터의 임의성이 혼란스러운 샘플 집합을 형성하는 주요 요인이라는 것이 밝혀졌습니다.

- **Performance Highlights**: 연구 결과, DNN의 낮은 레이어에서의 파라미터 변화가 혼란스러운 샘플 집합에 미치는 영향이 크며, 이들은 고차 상호작용의 복잡성과 상호작용의 상쇄 작용을 통해 설명됩니다. 실험을 통해 DNN의 비일반화 가능한 표현의 내부 메커니즘을 검증하고, 다양한 DNN들이 전혀 다른 혼란스러운 샘플 집합을 가질 수 있다는 역설적인 현상을 발견했습니다. 이러한 결과는 DNN의 설명 가능한 AI 연구에 기여할 수 있는 중요한 발견으로 평가됩니다.



### Quantifying Security Vulnerabilities: A Metric-Driven Security Analysis of Gaps in Current AI Standards (https://arxiv.org/abs/2502.08610)
- **What's New**: 이 논문은 AI 시스템이 중요한 인프라에 통합됨에 따라 AI 준수 프레임워크의 보안 격차를 점검하고 정량화하는 새로운 방법론을 제시합니다. 특히 NIST AI RMF 1.0, 영국의 AI 및 데이터 보호 리스크 툴킷, EU의 ALTAI 등의 주요 AI 거버넌스 표준에서 136개의 보안 우려 사항을 도출하였습니다. 이 연구는 AI 준수 기준의 효과적인 보안 보호 수준을 평가하고, 이를 통해 보다 강력한 보안 통제가 필요함을 강조합니다.

- **Technical Details**: 본 연구는 Risk Severity Index (RSI), Attack Vector Potential Index (AVPI), Compliance-Security Gap Percentage (CSGP), Root Cause Vulnerability Score (RCVS)라는 네 가지 핵심 지표를 개발하였습니다. 이러한 지표들은 각 프레임워크 간의 보안 효율성을 정량적으로 측정할 수 있게 하며, 시스템적 문제를 파악하는 데 사용됩니다. 특히, NIST는 69.23%의 위험을 해결하지 못하며, ALTAI와 ICO 툴킷은 각각의 보안 격차와 공격 벡터 취약성이 가장 높은 것으로 나타났습니다.

- **Performance Highlights**: 우리의 분석 결과, 기존의 AI 준수 프레임워크가 AI 특정 위협에 효과적으로 대처하지 못하고 있음이 드러났습니다. 또한, 불분명한 정의와 실행 가능한 보안 통제 부족이 주요 문제로 지적되었습니다. 이러한 결과들은 정책 입안자와 조직이 AI 시스템의 보안을 강화하기 위해 구현해야 할 보다 구체적이고 강력한 요구 사항들이 필요하다는 점을 강조합니다.



### Distillation Scaling Laws (https://arxiv.org/abs/2502.08606)
Comments:
          67 pages, 54 figures, 13 tables

- **What's New**: 이번 연구는 모델 디스틸레이션(distillation) 성능을 계산 예산(compute budget)과 학생(student) 및 교사(teacher) 모델 간의 할당에 따라 추정하는 새로운 스케일링 법칙(scaling law)을 제시합니다. 이를 통해 대규모 디스틸레이션 사용에 따른 위험을 줄일 수 있습니다.

- **Technical Details**: 연구에서는 두 가지 경우에 대한 최적의 컴퓨트(compute) 할당 방식을 제안합니다: 1) 교사가 존재할 때, 2) 교사가 훈련이 필요할 때. 여러 학생에 대해 디스틸레이션을 진행할 경우, 또는 이미 존재하는 교사가 있는 경우, 디스틸레이션이 감독된 사전 훈련(supervised pretraining)보다 성능이 우수하다는 것을 발견했습니다.

- **Performance Highlights**: 특히, 한 명의 학생에 대한 디스틸레이션 과정에서 교사 모델도 훈련이 필요할 경우, 감독된 학습(supervised learning)이 더욱 효과적이라는 결과를 도출했습니다. 이 외에도, 대규모 디스틸레이션 연구를 통해 디스틸레이션 이해도를 높이고 실험 디자인을 개선할 수 있는 통찰력을 제공하였습니다.



### CurvGAD: Leveraging Curvature for Enhanced Graph Anomaly Detection (https://arxiv.org/abs/2502.08605)
- **What's New**: CurvGAD는 복잡한 네트워크에서 기하학적 특성을 활용해 그래프의 이상을 탐지하는 새로운 접근 방식을 제안합니다. 기존 그래프 이상 탐지(GAD) 방식은 구조적 및 속성 기반 이상에만 집중하는 반면, 본 연구는 곡률(curvature)을 통한 기하학적 이상을 조명합니다. CurvGAD는 곡률에 기반한 새로운 이상 분류 방식을 도입하여 기존의 접근 방식에서 놓쳤던 중요 정보를 포착합니다.

- **Technical Details**: CurvGAD는 두 개의 병렬 파이프라인을 구축하여 이상 해석력을 향상시킵니다. 첫 번째 파이프라인은 곡률에 대한 기하학적 재구성을 통해 엣지 곡률을 복원하고, 두 번째 파이프라인은 구조 및 속성을 곡률에 무관하게 재구성하여 비기하적 이상을 분리합니다. 이와 같은 접근은 다양한 복잡한 토폴로지를 상세히 표현할 수 있게 합니다.

- **Performance Highlights**: 본 논문에서는 10개의 실제 데이터셋을 사용하여 CurvGAD의 효능을 평가하였고, 기존 최첨단 GAD 방법보다 최대 6.5% 향상된 성능이 입증되었습니다. CurvGAD는 동질적 및 이질적 네트워크 환경에서도 기하학적, 구조적 및 속성 기반 이상을 효과적으로 탐지할 수 있는 능력을 갖추고 있습니다.



### Learning in Markets with Heterogeneous Agents: Dynamics and Survival of Bayesian vs. No-Regret Learners (https://arxiv.org/abs/2502.08597)
Comments:
          Learning in Markets, Heterogeneous Agents, Regret and Survival

- **What's New**: 이번 연구는 자산 시장에서 이질적인 학습 에이전트의 성능을 분석합니다. 서로 다른 학습 이론을 가진 에이전트들이 어떻게 시장에서 자산을 투자하는지를 비교하며, Bayes(베이지안) 학습자와 no-regret(무후회) 학습자의 차별화를 주목합니다. Bayes 학습자는 특정 모델에 대한 사전 확률을 가지고 있으며, 시간이 지나면서 정확한 모델로 수렴하지만, 무후회 학습자는 이러한 가정 없이도 경쟁력을 갖춥니다.

- **Technical Details**: 연구에서는 자산 시장에서 서로 다른 학습 에이전트들이 반복적으로 상호작용하며 wealth의 성장률을 극대화하기 위한 전략을 가지고 있음을 보여줍니다. Bayes 학습자는 이전 정보에 기반해 업데이트를 수행하고 무후회 학습자는 단기적인 wealth 변화에 즉각 반응하여 전략을 조정합니다. 우리는 Bayes 학습이 취약한 반면, 무후회 학습은 환경에 대한 지식이 덜 요구되므로 더 강건하다는 것을 규명하였습니다.

- **Performance Highlights**: 저희 연구 결과, 낮은 후회 값을 기록한 무후회 학습자가 Bayes 학습자에게 밀리는 경우에도 불구하고 생존하지 못할 수 있음을 보여줍니다. 이러한 결과는 경제학의 시장 선택 가설을 업데이트와 후회 최소화 프레임워크와 연결하며, 서로 다른 이론 간의 교량 역할을 합니다. 본 연구는 이질적 학습 에이전트들 간의 상호작용이 자산 시장에서의 생존 및 지배의 개념과 어떤 관련이 있는지를 탐구하였습니다.



### Commercial LLM Agents Are Already Vulnerable to Simple Yet Dangerous Attacks (https://arxiv.org/abs/2502.08586)
- **What's New**: 최근 ML 보안 문헌에서는 LLM(대규모 언어 모델)을 대상으로 한 공격이 집중적으로 다루어지고 있지만, 이러한 공격은 종종 모델의 개인 정보를 추출하거나 해로운 출력을 생성하게 하는 방식으로 진행됩니다. 이 논문에서는 LLM을 포함한 대리 시스템의 보안 취약점을 분석하고, 이러한 시스템이 직면하는 특정한 위험들을 조명합니다. 특히, LLM 기반 대리 에이전트는 다양한 환경과 연결되어 있어 훨씬 더 공격받기 쉬운 구조를 가지고 있습니다.

- **Technical Details**: 저자들은 LLM 에이전트에 대한 공격의 분류 체계를 제시하며, 이를 통해 공격자, 목표, 침투 지점, 공격 관찰 가능성 및 공격 전략에 따라 다섯 가지 카테고리로 나누었습니다. 논문에서는 Reddit와 같은 신뢰할 수 있는 웹사이트에 악성 게시글을 만들어 LLM 에이전트를 조작하는 매우 단순하고 효과적인 공격 파이프라인을 설계하여 이를 시연합니다. 더욱이, 에이전트의 동작 중 발생할 수 있는 다양한 취약점에 대해서도 설명하고 있습니다.

- **Performance Highlights**: 실제 사례를 통해 오픈 소스 및 상용 에이전트에 대한 공격을 시연하면서 즉각적인 결과를 도출할 수 있음을 보여줍니다. 공격 성공률이 높은 여러 유형의 공격(개인 정보 누출, 악성파일 다운로드, 피싱 이메일 발송 등)에 대해 설명하며, 심지어 기계 학습에 대한 전문 지식 없이도 공격이 가능하다는 점을 강조합니다. 이러한 점은 LLM 에이전트가 사용자의 안전을 위협할 수 있는 심각한 위험 요소로 작용하고 있음을 알립니다.



### FBFL: A Field-Based Coordination Approach for Data Heterogeneity in Federated Learning (https://arxiv.org/abs/2502.08577)
- **What's New**: 이번 논문에서는 Field-Based Federated Learning (FBFL)이라는 새로운 접근 방식을 제안하여 데이터 분포 문제와 중앙집중식 아키텍처의 한계를 극복하고자 합니다. FBFL은 매크로 프로그래밍(macroprogramming)과 필드 조정(field coordination)을 활용하여 비독립적이고 비동일 분포된(non-IID) 데이터 문제를 해결하며, 공간적으로 분산된 디바이스 간에 리더를 선정하여 개인화된 학습을 지원합니다. 또한, 이 논문은 FBFL 방식이 기존의 FedAvg 알고리즘과 유사한 성능을 내며, 매우 도전적인 비 IID 시나리오에서도 더 나은 결과를 보여준다는 것을 강조합니다.

- **Technical Details**: FBFL은 'fields of Machine Learning (sub)models'라는 개념을 도입하여, 정보 전파 및 집합(computing)에서 발전된 알고리즘을 기반으로 하고 있습니다. 또한, FBFL은 디바이스를 공간 근접성을 기준으로 그룹화하여 효율적인 자기 조직화(self-organizing) 계층 구조를 구축함으로써 비 IID 데이터 환경에서도 성능을 향상시킵니다. 이 접근은 중앙 집중식 권한에 의존하지 않고도 동적으로 효율적인 모델 집계를 가능하게 합니다. 특히, 자가 조직적 하이브리드 아키텍처는 피어-투-피어 상호작용을 통해 리더를 분산 방식으로 선출함으로써 리더가 각 지역의 모델을 집약하도록 합니다.

- **Performance Highlights**: FBFL은 MNIST, FashionMNIST 및 Extended MNIST 데이터 세트를 사용하여 철저한 평가를 거쳤습니다. IID 데이터 조건에서 FBFL은 널리 사용되는 FedAvg와 유사한 성능을 보였습니다. 더욱이, 도전적인 비 IID 시나리오에서도 FBFL은 FedAvg를 초과하는 성능을 보여주었으며, 비 IID 데이터 분포 문제를 해결하기 위해 고안된 최신 방법인 FedProx 및 Scaffold를 능가하는 결과를 기록했습니다. 마지막으로, FBFL의 자기 조직적 계층 구조는 서버 실패에 대한 강인성을 선보였습니다.



### Mapping the Landscape of Generative AI in Network Monitoring and Managemen (https://arxiv.org/abs/2502.08576)
Comments:
          32 pages, 9 figure, 10 tables

- **What's New**: 이 연구는 Generative Artificial Intelligence (GenAI) 모델들이 네트워크 모니터링 및 관리에 어떻게 활용될 수 있는지를 탐구합니다. 특히 LLM, GPT, Diffusion Models와 같은 다양한 GenAI의 포괄적인 개요를 제공하며, 이러한 기술들이 사람과 기계 간의 상호작용을 개선하는 방법을 밝힙니다. 또한, 연구에서는 GenAI 채택을 위한 도전과제와 기회에 대한 논의도 포함되어 있습니다.

- **Technical Details**: GenAI는 복잡한 데이터 분포의 특징을 추출하고 이를 이용해 유사하지만 독특한 새로운 데이터를 생성하는 능력이 뛰어난 모델입니다. 특히 LLM(대형 언어 모델), Diffusion Models, 상태 공간 모델(State Space Models) 등이 GenAI 기술의 대표적인 예시로 주목받고 있습니다. 이러한 모델들은 고성능의 대규모 GPU를 활용하여 훈련되며, 네트워크 관리에 필요한 적응적으로 응답하는 네트워크 구축에 기여할 수 있습니다.

- **Performance Highlights**: GenAI 모델들은 상업적 가치와 기술적 가능성을 바탕으로 빠르게 성장하고 있습니다. 글로벌 GenAI 시장은 2023년 말에 약 454억 달러에 달하며, 향후 연평균 약 20억 달러 성장할 것으로 예상됩니다. 이 연구는 네트워크 모니터링 및 관리에 있어서 GenAI의 활용이 네트워크의 자가 구성, 자가 최적화, 자가 치유 능력을 향상시키는 데 기여할 것이라고 결론짓습니다.



### COAST: Intelligent Time-Adaptive Neural Operators (https://arxiv.org/abs/2502.08574)
- **What's New**: 이번 연구에서는 Causal Operator with Adaptive Solver Transformer (COAST)라는 새로운 신경 연산자 학습 방법을 제안합니다. COAST는 causal language model (CLM) 프레임워크를 활용하여 시스템의 진화와 최적의 시간 단계를 예측하는 데 중점을 두고 있습니다. 이를 통해 계산 효율성과 정확성 간의 균형을 맞추며, 동적 시스템 전반에 걸쳐 고유한 특성과 일치하는 가변 시간 단계를 생성하는 능력을 보입니다.

- **Technical Details**: COAST는 시간 의존적인 물리 시스템을 해결하기 위해 설계된 시간 적응형 신경 연산자 아키텍처입니다. 이 모델은 입력 공간의 시계열 데이터를 위한 학습 가능한 시공간 인코딩을 활용하여 연속 시간 지점에서 동작하며, 예측된 다음 단계를 공간-시간적으로 결합된 임베딩으로 출력합니다. 아키텍처는 시공간 인코더, causal language model, 해석-수정 메커니즘, 보간 디코더의 네 가지 주요 요소로 구성됩니다.

- **Performance Highlights**: COAST는 다양한 도전 과제를 기반으로 한 벤치마크에서 기존의 최첨단 방법을 지속적으로 초월하는 성능을 보여주었습니다. 높은 복잡성을 가진 영역에서는 더 작은 단계로, 단순한 영역에서는 더 큰 단계를 적용하여 예측의 정확성을 높이는데 기여합니다. 이 연구는 CLM을 기반으로 한 지능형 적응형 솔버의 잠재력을 강조하며, 동적 시스템의 확장 가능성 있는 연산자 학습에 대한 새로운 길을 제시합니다.



### A Novel Approach to for Multimodal Emotion Recognition : Multimodal semantic information fusion (https://arxiv.org/abs/2502.08573)
- **What's New**: 이 논문에서는 DeepMSI-MER이라는 새로운 멀티모달 감정 인식 방법을 제안합니다. 이 방법은 대조 학습(contrastive learning)과 시각적 시퀀스 압축(visual sequence compression)을 통합하여 구현되었습니다. 제안된 방안은 서로 다른 모달리티에서의 특징 융합(cross-modal feature fusion)을 강화하며, 시각적 모달리티의 중복성을 줄이는 데 도움을 줍니다.

- **Technical Details**: DeepMSI-MER은 세 가지 단계로 구성됩니다: 모달리티 특화(feature extraction), 초기(feature fusion) 및 최종(feature fusion) 융합 그리고 모델 예측(model prediction)입니다. 이 과정에서 BERT 및 Wav2Vec 모델을 미세 조정하여 텍스트 및 오디오 데이터에서 의미 특징을 추출하고, 이를 시각적 모달리티 비디오 특성과 융합합니다. 최종적으로, Temporal Convolution Networks (TCN)를 통해 시계열 특징을 캡처하여 최종 비디오 특징을 생성합니다.

- **Performance Highlights**: IEMOCAP 및 MELD의 두 가지 공개 데이터셋에서 실험 결과, DeepMSI-MER는 멀티모달 감정 인식의 정확성과 강인성을 크게 개선했습니다. 이러한 결과는 멀티모달 특징 융합과 제안된 접근법의 유효성을 입증합니다. DeepMSI-MER는 특히 다양한 현실 세계 응용에서의 감정 인식 성능을 향상시켜 나갈 것으로 기대됩니다.



### Brain Latent Progression: Individual-based Spatiotemporal Disease Progression on 3D Brain MRIs via Latent Diffusion (https://arxiv.org/abs/2502.08560)
Comments:
          arXiv admin note: text overlap with arXiv:2405.03328

- **What's New**: 본 논문에서는 Brain Latent Progression (BrLP)이라는 새로운 공간-시간 모델을 제안합니다. 이는 3D 뇌 MRI에서 개인의 질병 진행을 예측하기 위해 설계되었습니다. BrLP는 낮은 차원의 잠재 공간에서 작동하며, 개인화된 예측을 위한 주체 메타데이터를 통합하였습니다.

- **Technical Details**: BrLP는 Latent Diffusion Model (LDM)과 ControlNet을 결합하여 주어진 개인의 데이터를 바탕으로 개인화된 뇌 MRI를 생성합니다. 또한, 악세서리 모델을 통해 질병 역학에 대한 사전 지식을 통합하여, 가용한 경우 장기적 데이터를 활용할 수 있게 하였습니다. Latent Average Stabilization (LAS) 기법을 도입하여 예측의 공간-시간 일관성을 확보하고, 처리 메모리 요구를 줄입니다.

- **Performance Highlights**: 11,730개의 T1w 뇌 MRI를 사용하여 BrLP를 훈련하고 평가하였으며, 2,257개의 외부 테스트 세트를 통해 일반화 능력을 검증했습니다. 실험 결과, BrLP가 생성한 MRI 스캔은 기존 방법들과 비교해 최첨단 정확도를 나타냈습니다. 코드는 공개적으로 이용 가능합니다.



### Human-Centric Foundation Models: Perception, Generation and Agentic Modeling (https://arxiv.org/abs/2502.08556)
Comments:
          9 pages

- **What's New**: 최근 인간 중심의 기초 모델(Human-centric Foundation Models, HcFMs)이 다채로운 인간 중심 작업을 단일 프레임워크로 통합하여 전통적인 작업별 접근 방식을 초월하고 있다는 점에서 주목할 만한 변화가 있습니다. 이러한 모델은 인간의 외모, 감정, 정체성 및 행동을 보다 정교하게 이해하고 생성할 수 있는 가능성을 열어줍니다. 분야의 발전은 연구자들이 인간을 보다 포괄적이고 복잡한 시스템으로 이해해야 한다는 요구와 함께 진행되고 있습니다.

- **Technical Details**: 이 논문에서는 HcFMs를 네 가지 카테고리로 분류하는 새로운 분류법을 제안합니다: (1) 인간 중심의 인식 기초 모델, (2) 인간 중심의 AIGC 기초 모델, (3) 통합 인식 및 생성 모델, (4) 인간 중심의 에이전틱 모델입니다. 각 모델은 그들이 지원하는 다양한 하위 작업에 따라 분류되어, 인간 중심의 데이터로부터 효율적으로 학습할 수 있도록 돕습니다.

- **Performance Highlights**: HcFMs는 기존의 작업별 모델보다 높은 일반화 능력과 적용 가능성, 그리고 사실감을 보증합니다. 이 모델들은 특히 2D와 3D 작업에서의 성능을 개선하며, 인간의 행동과 상호작용을 보다 정밀하게 표현할 수 있는potential을 지니고 있습니다. 또, 이 모델들은 다중 모달리티(multi-modality)를 활용하여 더욱 풍부한 인식을 가능하게 하며, 사용자와의 상호작용 작업에도 적용될 수 있습니다.



### Fostering Appropriate Reliance on Large Language Models: The Role of Explanations, Sources, and Inconsistencies (https://arxiv.org/abs/2502.08554)
Comments:
          CHI 2025. This version includes the appendix

- **What's New**: 이 논문은 대형 언어 모델(LLM)의 응답이 사용자 신뢰도에 미치는 영향을 살펴봅니다. 응답에 대한 설명, 불일치 및 출처와 같은 여러 기능이 사용자가 LLM에 의존하는 방식에 어떻게 영향을 미치는지 분석했습니다. 연구 결과, 설명이 존재하는 경우 사용자들은 올바른 답변과 잘못된 답변 모두에 대해 더 많이 의존하게 되며, 출처가 제공되거나 설명에 불일치가 있을 경우 잘못된 답변에 대한 의존도가 감소하는 경향을 보였습니다.

- **Technical Details**: 연구는 16명의 참가자를 대상으로 한 think-aloud 연구와 308명의 참가자를 대상으로 한 대규모 제어 실험의 두 가지 방법론을 사용했습니다. 실험에서 LLM의 응답 기능을 조절하여 정확성, 설명의 존재 여부 및 클릭 가능한 출처의 제공 여부를 비교 분석했습니다. 이 연구는 참가자들이 LLM 응답에 대해 어떻게 신뢰를 형성하는지를 이해하기 위해 다양한 지표(정확도, 신뢰도 등)를 측정했습니다.

- **Performance Highlights**: 실험 결과, 설명이나 출처가 제공될 경우 참가자들은 자신의 답변에 대해 더 높은 신뢰도를 보고했으며, LLM의 응답을 더욱 긍정적으로 평가했습니다. 설명은 올바른 답변과 잘못된 답변 모두에 대한 의존도를 증가시켰지만, 출처는 올바른 응답에 대한 적절한 의존도를 높이고 잘못된 응답에 대한 과도한 의존도를 줄이는 데 더 효과적이었습니다. 이러한 결과는 LLM 사용 시 적절한 신뢰성을 촉진하기 위한 기술적 통찰을 제공합니다.



### LLMs can implicitly learn from mistakes in-contex (https://arxiv.org/abs/2502.08550)
- **What's New**: 이번 연구는 LLMs가 수학적 추론 과제에서 실수로부터 암묵적으로 배울 수 있는지를 조사합니다. 혁신적인 접근 방식으로, 잘못된 정답과 올바른 정답을 단순히 나열할 때 모델이 성능이 더 좋다는 것을 발견했습니다. 이는 많은 기존 연구에서 강조되는 명시적 학습 방식과 대비되는 결과로, LLMs가 실수에 대한 명시적 피드백 없이도 학습이 가능하다는 것을 보여줍니다.

- **Technical Details**: 연구에서 제안된 'prompting for implicit learning'은 잘못된 답안과 올바른 답안을 동시에 보여주며, 모델이 스스로 이들 사이의 차이를 유추하도록 합니다. 이 방법은 코어 미국의 Chain-of-Thought(CoT) 프롬프트와 비교하여 대부분의 경우 성능이 우수하다는 점에서 주목받습니다. 또한, 다양한 LLM과 수학적 추론 데이터셋을 통해 결과의 안정성을 검증하였습니다.

- **Performance Highlights**: 실험 결과, LLMs는 제공된 잘못된 답안과 올바른 답안을 바탕으로 유도된 즉각적인 추론을 통해 높은 성과를 나타냈습니다. 이 과정에서 인간 평가자들로부터 모델이 생성한 이론 또한 높은 점수를 받아, LLMs가 암묵적으로 양질의 수정 이론을 유추할 수 있음을 확인했습니다. 이러한 결과는 LLMs의 가능성을 새롭게 규명하며 기존의 피드백 기반 접근 방식에 대한 의문을 제기합니다.



### Input convex neural networks: universal approximation theorem and implementation for isotropic polyconvex hyperelastic energies (https://arxiv.org/abs/2502.08534)
- **What's New**: 이 논문은 등방성 초탄성(isotropic hyperelasticity)에 대한 새로운 신경망 프레임워크를 제안하며, 필요한 물리적 및 수학적 제약을 충족하는 동시에 보편적 근사 정리를 만족합니다. 새로운 네트워크는 입력 볼록 네트워크 아키텍처(input convex network architecture)와 변형 기울기의 부호 있는 특이값(signed singular values)의 기본 다항식(formulation in the elementary polynomials)으로 구성됩니다. 또한, 기존의 망과 비교해 볼 때, 제안된 방법이 비볼록(non-polyconvex) 에너지를 근사하고 볼록 궤를 계산하는 데 유리한 점을 보여줍니다.

- **Technical Details**: 등방성 초탄성 물질 모델의 수학적 공식화는 다양한 원칙과 제한 사항을 갖습니다. 이 연구는 특히 등방성 초탄성 재료에 대해 중점을 두고, 경로 독립성(path-independence), 열역학 제2법칙(thermodynamics), 객체성을(objectivity), 성장 조건(growth conditions) 및 자가 침투(self penetration) 방지와 같은 물리적 제약 조건을 충족하는 것을 목표로 합니다. 제안된 프레임워크는 볼록성(convexity) 개념에 기반한 약한 성장 및 강제성 조건에서의 최적 해 존재성을 보장하며, 특히 폴리볼록 기능의 관점과 호환됩니다.

- **Performance Highlights**: 제안된 신경망 프레임워크는 등방성 폴리볼록 에너지(frame-indifferent, isotropic polyconvex energy)에 대한 보편적 근사 정리를 입증하였으며, 물리적 및 수학적 제약을 정확하게 충족합니다. 이러한 접근법은 비볼록 에너지를 근사하는데도 유리하며, 초탄성(numerical examples in hyperelasticity) 수치 예제 결과도 잘 드러나 있습니다. 결과적으로 이 연구는 초탄성 모델링의 새로운 가능성을 제시하며 다양한 성능 개선을 기대할 수 있습니다.



### FedMHO: Heterogeneous One-Shot Federated Learning Towards Resource-Constrained Edge Devices (https://arxiv.org/abs/2502.08518)
- **What's New**: 이번 연구에서는 자원 제약이 있는 클라이언트와 자원 충분한 클라이언트를 모두 수용할 수 있는 새로운 연합 학습(FL) 프레임워크인 FedMHO를 제안합니다. FedMHO는 자원 충분한 클라이언트에서는 깊은 분류 모델(deep classification models)을 사용하고, 자원 제약이 있는 클라이언트에서는 경량 생성 모델(lightweight generative models)을 활용합니다. 또한 지식 융합 지식(model heterogeneity) 문제를 해결하기 위해 FedMHO-MD 및 FedMHO-SD라는 두 가지 솔루션을 도입합니다.

- **Technical Details**: FedMHO는 데이터 생성(data generation)과 지식 융합(knowledge fusion)의 두 단계를 포함하여 글로벌 모델을 훈련합니다. 데이터 생성 단계에서는 클라이언트에서 수신된 디코더가 로컬 레이블 분포에 기반하여 합성 샘플(synthetic samples)을 생성합니다. 또한 이러한 합성 샘플의 품질을 개선하기 위해 비지도 데이터 최적화 솔루션이 적용됩니다.

- **Performance Highlights**: 실험 결과 FedMHO, FedMHO-MD 및 FedMHO-SD는 각각 5.17%, 8.35%, 8.25%의 평균 정확도(improved accuracy)를 향상시켜 기존의 최적 기준을 초과하는 성능을 입증하였습니다. 연구 결과는 자원 부족 클라이언트와 자원 충분 클라이언트가 혼합된 환경에서도 효과적인 FL이 가능함을 보여줍니다.



### Measuring Diversity in Synthetic Datasets (https://arxiv.org/abs/2502.08512)
- **What's New**: 이 논문에서는 DCScore라는 새로운 방법을 소개하여 기존의 합성 데이터셋의 다양성을 분류 관점에서 평가하는 방법을 제안합니다. DCScore는 각 샘플의 상호 관계를 활용하여 데이터셋 내 다양성을 측정하며, 이론적으로 검증된 여러 공리를 만족합니다. 이러한 접근법은 합성 데이터셋의 품질을 높이고 모델 성능을 향상시키는 데 중요한 역할을 합니다.

- **Technical Details**: DCScore는 분류 작업으로서 다양성 평가를 수행하고, 이를 통해 합성 데이터셋의 각 샘플이 평가 결과에 미치는 영향을 포괄적으로 분석할 수 있습니다. 이 방법은 Leinster & Cobbold(2012)에 의해 제안된 네 가지 공리를 충족하여, 유효 숫자, 동일한 샘플, 대칭성 및 단조성을 보장합니다. 또한, DCScore는 기존 방법들과 비교할 때 계산 비용을 상당히 줄이는 데 성공했습니다.

- **Performance Highlights**: DCScore는 다양한 다양성 가짜 진리와 더 강력한 상관관계를 보이는 실험 결과를 보였습니다. 특히, DCScore는 기본 메트릭들보다 더 강한 효과를 발휘하며, 합성 데이터셋의 다양성을 높이 평가합니다. 이 방법은 실제 경험적 및 이론적 증거를 통해 낮은 계산 비용의 이점을 입증했습니다.



### Enhancing Auto-regressive Chain-of-Thought through Loop-Aligned Reasoning (https://arxiv.org/abs/2502.08482)
Comments:
          work in progress

- **What's New**: 최근 연구에서는 Chain-of-Thought (CoT) 프롬프트가 언어 모델의 추론 능력을 향상시키는 강력한 기법으로 등장했습니다. 하지만 긴 CoT 경로를 생성하는 것은 도전 과제가 되고 있습니다. 본 논문에서는 Looped Transformer를 활용한 RELAY (REasoning through Loop Alignment iterativelY)를 제안하여, CoT 추론 단계와 루프 반복을 정렬하고 추가적인 중간 감독 학습을 통해 길이 일반화(length generalization) 능력을 강화합니다.

- **Technical Details**: RELAY는 루프된 Transformer의 장점을 최대한 활용하여 오토레그레시브(auto-regressive) 모델이 더 긴 추론 체인을 처리할 수 있도록 돕는 새로운 프레임워크입니다. 주요 혁신은 루프된 Transformer 모델이 여러 작업에서 일반적인 추론기의 역할을 수행할 수 있음을 실증적으로 입증하는 것과, CoT 추론 단계와 루프된 Transformer 간의 반복 간 정렬을 통해 훈련 길이를 초과하는 문제에 대한 정확한 추론 체인을 생성할 수 있도록 하는 것입니다.

- **Performance Highlights**: 광범위한 실험을 통해 RELAY 접근법이 생성한 고품질의 추론 체인을 통해 오토레그레시브 Transformers의 추론 능력을 크게 향상시키는 것을 입증했습니다. 이러한 성과는 더욱 복잡한 문제 해결에 있어 모델의 성능을 개선하고, 다양한 언어 작업에서 일관된 성능을 유지할 수 있도록 합니다.



### Training-Free Restoration of Pruned Neural Networks (https://arxiv.org/abs/2502.08474)
Comments:
          Under Review in TNNLS since May 2022

- **What's New**: 이번 논문에서는 네트워크 프루닝(Pruning)에 대한 새로운 접근법인 LBYL(Leave Before You Leave)을 제안합니다. 기존의 프루닝 후 재훈련 과정이 계산적으로 비용이 많이 드는 문제를 해결하고자, 데이터 없이, 또한 재훈련 없이 프루닝된 네트워크를 복원하는 방법을 제시합니다. LBYL은 각 프루닝된 뉴런이 최대한 많은 보존된 뉴런에게 정보를 남기도록 하여 보다 강력한 근사치를 제공하는 방식입니다.

- **Technical Details**: LBYL 방법은 기존 네트워크와 그 근사치 간의 재구성 오차(reconstruction error)를 수학적으로 분석하여, 유도된 손실 함수(loss function)에 대한 폐쇄형 솔루션을 도출합니다. 이 방법은 뉴런 간의 유사성이 낮은 문제를 해결하며, 뉴런들이 함께 협력하여 원래 뉴런의 출력을 더 잘 근사할 수 있도록 합니다. 이론적 분석을 통해 기존의 접근 방식보다 훨씬 더 유연하고 강력한 조건을 갖추고 있음을 보여줍니다.

- **Performance Highlights**: LBYL 방법은 광범위한 실험을 통해 복원된 네트워크의 정확도를 기존의 유사성을 활용한 접근법들보다 높게 유지함을 입증했습니다. 실험 결과, LBYL은 기존 네트워크의 원래 구조를 더 잘 근사하며, 그에 따라 향상된 정확도를 달성할 수 있음을 나타냅니다. 해당 연구의 초기 버전은 NeurIPS 2021과 ICML 2022에 제출되었습니다.



### mmE5: Improving Multimodal Multilingual Embeddings via High-quality Synthetic Data (https://arxiv.org/abs/2502.08468)
- **What's New**: 이 논문은 다양한 모달리티를 통합하여 고품질 합성 데이터를 생성하는 새로운 접근 방식을 제안합니다. 고품질의 합성 멀티모달 데이터에는 세 가지 주요 기준이 필요하며, 이는 넓은 범위, 강력한 크로스 모달 정렬, 높은 충실도를 포함합니다. 연구진은 이를 통해 mmE5라는 다국어 멀티모달 모델을 훈련하고 여러 벤치마크에서 뛰어난 성능을 달성했습니다.

- **Technical Details**: 연구는 다단계 방법론을 통해 고품질의 멀티모달 데이터를 합성합니다. 첫 번째로, MLLM을 사용하여 입력 이미지를 다양한 관점에서 분석하고 설명을 생성합니다. 그 후, MLLM은 합성된 텍스트 데이터를 다시 평가하여 크로스 모달 정렬과 충실도를 향상시킵니다. 이 방법론은 실세계 이미지와 관련된 텍스트를 결합하여 뉴스 데이터를 생성하는데 중점을 두고 있습니다.

- **Performance Highlights**: mmE5 모델은 MMEB 벤치마크에서 최첨단 성능을 기록하며, 이전 모델에 비해 훈련 데이터가 45배 적은 상태에서도 뛰어난 결과를 보였습니다. 또한, 비즈니스와 다양한 언어에 대한 강력한 성능을 보여주며 XTD 벤치마크에서도 최고 성과를 달성했습니다. 이로 인해 mmE5는 멀티모달 임베딩 모델의 새로운 기준을 세우고 있습니다.



### Proceedings 40th International Conference on Logic Programming (https://arxiv.org/abs/2502.08453)
- **What's New**: 이번 회의는 40번째 국제 논리 프로그래밍 회의(ICLP)로, 1982년 마르세유에서 시작된 이후 현재까지 이어져 온 가장 중요한 이벤트 중 하나입니다. 텍사스주 댈러스에서 열린 본 회의에서는 논리 프로그래밍과 관련된 여러 최신 연구 결과가 발표되었습니다. 이번 자료집에는 비모노토닉 추론, 확률적 추론, 주어진 논리와 신경 네트워크 모델의 결합과 같은 다양한 주제가 다루어졌습니다.

- **Technical Details**: 이 연구에서는 정형 및 운영적 의미론(formal and operational semantics), 프로그래밍 언어 설계 및 프로그래밍 방법론(예: answer set programming, inductive logic programming), 그리고 확률적 프로그래밍과 같은 여러 주제를 포함합니다. 또한, 생성된 프로그램의 프로그램 분석 및 논리에 기반한 검증(logic-based validation) 기법도 논의되었습니다. 제약 구현(constraint implementation) 및 로직 기반 프롬프트 엔지니어링(logic-based prompt engineering) 같은 구현 방법론도 포함되어 있습니다.

- **Performance Highlights**: 이번 ICLP에서 발표된 논문들은 특히 논리 프로그래밍과 대규모 언어 모델(LLMs)의 상호작용(interaction)과 관련된 혁신적인 접근 방식을 강조합니다. 이를 통해 논리 프로그래밍의 다양한 적용 가능성을 탐구하고, 언어 디자인의 새로운 방향을 제시하고 있습니다. 이 회의는 기존의 이론과 실제 적용 간의 간극을 해소하는 데 중요한 역할을 할 것으로 기대됩니다.



### Towards Prompt Generalization: Grammar-aware Cross-Prompt Automated Essay Scoring (https://arxiv.org/abs/2502.08450)
Comments:
          NAACL 2025 (Findings)

- **What's New**: 이번 연구에서는 문법 인식 기반의 교차 프롬프트 성적 평가(GAPS)를 제안합니다. 기존의 프롬프트 특화 모델과 달리, 본 모델은 문법 오류 수정(Grammar Error Correction) 기법을 통합하여 프롬프트에 의존하지 않는 Generic 에세이 표현을 학습합니다. GAPS는 원본 에세이와 수정된 에세이를 모두 참고하여 모델이 훈련 중에 일반적인 특징에 집중할 수 있도록 합니다.

- **Technical Details**: GAPS는 두 가지 주요 단계로 구성됩니다: (1) 에세이 수정 및 (2) 문법 인식 에세이 성적 평가입니다. T5 기반의 사전 훈련된 GEC 모델을 사용하여 에세이에서 발견된 문법 오류를 수정하고, 수정된 텍스트와 원본 텍스트를 동시에 스코어링 모델에 입력합니다. 에세이 인코더는 원본과 수정된 각각의 에세이를 처리하면서, 효과적인 정보 공유를 위해 다층적 구조를 채택합니다.

- **Performance Highlights**: 실험 결과, GAPS는 프롬프트 비의존적이며 문법 관련 특성에서 두드러진 성능 향상을 보여주었습니다. 특히, Conventions와 Sentence Fluency와 같은 프롬프트 비의존적 특성에서 현저한 개선이 나타났습니다. GAPS는 가장 도전적인 교차 프롬프트 상황에서 notable QWK 향상을 달성하며, 이는 보지 못한 프롬프트에 대한 평가에서 강점을 나타냅니다.



### CordViP: Correspondence-based Visuomotor Policy for Dexterous Manipulation in Real-World (https://arxiv.org/abs/2502.08449)
- **What's New**: 본 논문에서는 CordViP라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 로봇의 손과 객체 간의 상호작용을 인식하여 포인트 클라우드를 생성하고 관련 정보를 활용합니다. 이를 통해 기존의 3D 표현 방식이 가지고 있던 문제를 해결하고자 합니다.

- **Technical Details**: CordViP는 6D 포즈 추정 기술과 로봇의 고유 상태를 결합하여 상호작용-aware 포인트 클라우드를 생성합니다. 이는 객체와 손 사이의 접촉 맵과 손-팔 협동 동작 정보를 포함하여, 공간적 및 시간적 동적을 효과적으로 포착합니다. 이 과정은 세 가지 단계로 이루어집니다: 3D 포인트 클라우드 생성, 특성 추출 및 확산 정책의 적용입니다.

- **Performance Highlights**: CordViP는 네 가지 실제 작업에서 90%의 성공률을 기록하며, 기존 다른 방법들보다 훨씬 뛰어난 성능을 보여줍니다. 실험 결과는 다양한 환경 변화와 각기 다른 카메라 시점에서도 안정힌 일반화 성능을 나타내며, 제한된 데이터로도 효과적인 학습이 가능하다는 것을 강조합니다.



### Better Embeddings with Coupled Adam (https://arxiv.org/abs/2502.08441)
Comments:
          17 pages, 8 figures; figures corrected

- **What's New**: 이 논문에서는 대형 언어 모델(LLM)에서의 단어 임베딩의 비등방성을 초래하는 원인으로 Adam 최적화 알고리즘의 두 번째 모멘트를 지적하고, 이를 해결하기 위한 Coupled Adam이라는 수정된 최적화 기법을 제안합니다. Coupled Adam은 임베딩 매개변수에 대해 비등방성 문제를 완화하기 위해 특별히 설계된 효율적인 조정입니다.

- **Technical Details**: LLM에서의 단어 임베딩은 주어진 토큰 시퀀스를 입력으로 받아 다음 토큰을 예측하는 과정을 통해 학습됩니다. 그러나 embedding 벡터가 원점에서 멀리 떨어진 작은 부분 공간에 군집화되는 비등방성을 관찰하였습니다. Adam 최적화 기법은 희소 데이터에서 잘 작동하지만, 이로 인해 자주 등장하는 단어에 비해 드물게 등장하는 단어의 업데이트 벡터가 상대적으로 늘어나 비등방성을 야기합니다.

- **Performance Highlights**: Coupled Adam을 사용한 결과, 단어 임베딩의 품질을 유의미하게 향상시키는 것을 입증했습니다. 이 방법은 또한 충분히 큰 데이터셋에서 상류(upstream) 및 하류(downstream) 성능을 개선하는 데 긍정적인 영향을 미치는 것으로 나타났습니다.



### Composite Sketch+Text Queries for Retrieving Objects with Elusive Names and Complex Interactions (https://arxiv.org/abs/2502.08438)
Comments:
          Accepted at AAAI 2024, 9 pages. Project Website: this https URL

- **What's New**: 이번 논문은 비원어민 사용자들이 매우 구체적인 객체 이름을 찾는 데 어려움을 겪는 문제를 다루고 있습니다. 특히, 손으로 그린 스케치와 어려운 이름을 구술하는 텍스트를 조합한 복합적인 멀티모달 쿼리를 수용하는 검색 인터페이스를 요구하는 사례를 설명합니다. 기존의 텍스트 기반 이미지 검색(TBIR)과 스케치 기반 이미지 검색(SBIR) 문제와는 다른 새로운 문제 설정인 CSTBIR을 제안하고 있습니다.

- **Technical Details**: 연구에서는 약 200만 개의 쿼리와 10만 8000개의 자연 장면 이미지로 구성된 CSTBIR 데이터셋을 커리팅하였습니다. 이 문제의 해결책으로 제안된 STNET(스케치 및 텍스트 네트워크)은 손으로 그린 스케치를 활용하여 자연 장면 이미지에서 관련 객체를 찾고, 텍스트와 이미지를 인코딩하여 이미지 검색을 수행합니다. 모델은 대비 학습과 여러 훈련 목표를 통해 성능을 향상시킬 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 텍스트만, 스케치만, 복합 쿼리 방식 모두에서 여러 최신 검색 방법을 초월하는 성과를 나타냈습니다. 본 논문은 정교하고 복잡한 쿼리를 처리할 수 있는 CSTBIR 시스템을 통해 많은 분야에서 이미지 검색의 새로운 가능성을 제시합니다. 연구 결과물은 프로젝트 웹사이트에서 데이터셋과 코드를 공개하고 있습니다.



### From Haystack to Needle: Label Space Reduction for Zero-shot Classification (https://arxiv.org/abs/2502.08436)
Comments:
          Under review at ICML 2025

- **What's New**: 이번 논문에서는 Label Space Reduction (LSR)이라는 새로운 방법을 제시하여 대형 언어 모델(LLMs)의 제로샷 분류 성능을 향상시키고자 합니다. LSR은 후보 클래스를 체계적으로 순위화하고 감소시켜 모델이 가장 관련성이 높은 옵션에 집중할 수 있도록 합니다. 이 방법은 라벨 공간 표현을 동적으로 최적화하여 실험을 통해 Llama-3.1-70B와 Claude-3.5-Sonnet에서 각각 7.0%와 3.3%의 매크로 F1 점수 개선을 가져왔습니다.

- **Technical Details**: LSR은 분류 라벨 공간을 개선하기 위해 후보 클래스를 순위화하고 감소시키는 혁신적인 반복 시스템을 개발합니다. 또한, LSR의 계산 오버헤드를 줄이기 위해 모델을 확률적 분류기로 증류하는 방안을 제안하여 효율적인 추론을 가능하게 합니다. 이 방법은 출력이 생성될 때 동적으로 적응할 수 있는 정성적인 라벨 공간을 생성합니다.

- **Performance Highlights**: 여덟 개의 벤치마크에서 실험한 결과, LSR은 기존 제로샷 분류 기준에 비해 Llama-3.1-70B에서 평균 7.0%(최대 14.2%) 및 Claude-3.5-Sonnet에서 3.3%(최대 11.1%)의 매크로 F1 점수 향상을 보여주었습니다. 이러한 성과는 LLM이 중요한 정보에 보다 효과적으로 주의를 집중할 수 있도록 해주며, 문제 해결 과정에서의 추론을 개선하는데 기여합니다.



### Handwritten Text Recognition: A Survey (https://arxiv.org/abs/2502.08417)
- **What's New**: 최근 핸드라이팅 텍스트 인식(HTR) 기술은 과거의 휴리스틱(heuristic) 방식에서 딥러닝(deep learning) 기반의 현대적 신경망 모델로 진화했습니다. 본 설문조사는 HTR 시스템의 발전을 살펴보며, 주로 문자 단위(word-level) 및 단락 단위(paragraph-level) 인식의 두 가지 주요 수준으로 분류하여 다양한 접근 방식을 제공하고 있습니다. 이를 통해 현재 HTR 연구의 흐름과 발전 방향에 대한 포괄적인 분석을 제시합니다.

- **Technical Details**: HTR의 복잡성은 사람의 다양한 필기 스타일에서 비롯되며, 이러한 변동성을 처리하기 위한 알고리즘의 필요성이 있습니다. 초기 HTR 시스템은 주로 HMM(Hidden Markov Models)과 같은 통계적 방법에 의존하였으나, 최신 기술들은 CNN(Convolutional Neural Networks) 및 RNN(Recurrent Neural Networks)과 같은 심층 학습 모델의 발전을 통해 놀라운 성과를 거두고 있습니다. 이 설문에서는 최근 Transformer 아키텍처의 적용 및 발전도 함께 다루고 있습니다.

- **Performance Highlights**: HTR 기술은 최근 몇 년 간 심층 신경망의 발전과 다양한 데이터셋의 증가 덕분에 크게 향상되었습니다. 다양한 분야의 연구 결과를 공동 기준에 따라 평가하고, HTR 시스템의 성능을 비교하여 기술 발전의 흐름을 정리합니다. 이러한 분석을 통해 연구자와 실무자에게 향후 연구 방향에 대한 로드맵을 제공합니다.



### Learning Humanoid Standing-up Control across Diverse Postures (https://arxiv.org/abs/2502.08378)
Comments:
          Humanoid Standing-up Control, 12 pages

- **What's New**: 이 논문에서는 HoST(Humanoid Standing-up Control)라는 강화학습(framework) 기반의 새로운 시스템을 제안합니다. 이는 기존의 시뮬레이션 한계를 극복하고, 실제 환경에서도 다양한 자세에서 로봇이 스탠딩 업(standing-up)할 수 있는 능력을 향상시키는 데 중점을 두고 있습니다. 또한, 여러 지형에서의 커리큘럼 기반 훈련을 통해 다양한 상황에서의 적응력을 보장합니다.

- **Technical Details**: HoST는 멀티-크리틱 아키텍처(multi-critic architecture)와 스무스니스 정규화(smoothness regularization)를 활용하여 로봇의 움직임이 부드럽고 안정적으로 이루어지도록 합니다. 이를 통해 물리적 하드웨어에서의 진동이나 충동적인 움직임을 줄이는 동시에, 다양한 실내 및 실외 환경에서의 실제 배치도 가능합니다. 특히, 강화학습을 위한 보상 구조를 최적화하기 위해 다단계의 과정을 구성하여 접근합니다.

- **Performance Highlights**: 실험 결과, HoST에 의해 훈련된 제어 정책은 다양한 실험 환경에서도 높은 부드러움과 안정성을 보여줍니다. 이 연구는 기존의 정해진 경로를 따르지 않고도 효과적인 스탠딩 업 모션을 성공적으로 구현함으로써, 향후 휴머노이드 로봇의 실용적 응용 가능성을 넓힐 수 있는 기반을 마련하고 있습니다.



### Uncertainty Aware Human-machine Collaboration in Camouflaged Object Detection (https://arxiv.org/abs/2502.08373)
- **What's New**: 이번 연구에서는 Camouflaged Object Detection (COD) 분야에서 신뢰할 수 있는 시스템을 개발하기 위해 불확실성 추정 및 효율적인 활용 방안을 제안합니다. 컴퓨터 비전(CV) 모델과 비침습 뇌-컴퓨터 인터페이스(BCI)의 강점을 활용하여 인간-기계 협력 프레임워크를 구축하였습니다. 특히, 중첩된 시야 멀티뷰(backbone)를 통해 CV 모델의 예측 불확실성을 추정하고, 훈련 시 이를 효과적으로 활용하여 시스템 신뢰도를 높이고자 합니다.

- **Technical Details**: 제안하는 프레임워크는 CAMO 데이터세트에서 성능을 평가하였고, 기존 방법보다 4.56% 향상된 균형 정확도(Balanced Accuracy, BA) 및 3.66% 향상된 F1 점수를 기록하여 최첨단 결과를 도출했습니다. 훈련 과정에서 불확실성 측정과 정밀도 간의 강한 상관관계를 확인하였고, 제안된 훈련 정책과 인간-기계 협력 전략의 효과를 검증하는 ablation study(제외 연구)를 수행했습니다.

- **Performance Highlights**: 우수한 성과를 보인 참가자의 경우 기존 방법 대비 BA에서 7.6%, F1 점수에서 6.66%의 향상을 달성하였습니다. 이 연구는 신뢰할 수 있는 시스템을 통해 인간의 인지 부담을 줄이고 시스템의 신뢰성을 향상시켜 실제 COD 애플리케이션 및 인간-컴퓨터 상호작용 발전에 기여할 수 있는 강력한 기반을 제공합니다.



### Towards Principled Multi-Agent Task Agnostic Exploration (https://arxiv.org/abs/2502.08365)
- **What's New**: 이번 논문에서는 Multi-Agent Reinforcement Learning (MARL)에서의 task-agnostic exploration 문제를 새로운 방향으로 다루고 있습니다. 기존에 단일 에이전트 상황에서는 이 문제를 탐구한 연구가 많았으나, 다중 에이전트 환경에서는 상대적으로 이해가 부족하다는 점을 강조하고 있습니다. 이 논문은 여러 에이전트가 서로의 존재를 고려하면서 어떻게 탐색을 진행해야 하는지를 설명합니다.

- **Technical Details**: 저자들은 에이전트들이 탐색하는 방식에 따라 세 가지 뚜렷한 목표를 제시합니다. 각 목표는 장점과 단점을 가지고 있으며, 이들의 최적화 방식 또한 다릅니다. 이 과정에서 Trust Region Pure Exploration (TRPE)이라는 분산형 정책 최적화 알고리즘을 도입하여, 다중 시나리오에서의 task-agnostic exploration 문제에 실질적으로 접근하기 위한 방법을 제시합니다.

- **Performance Highlights**: 제안된 알고리즘은 다양한 목표를 최적화할 수 있는 능력을 보여주었으며, 각 목표 간의 중요한 차이가 실질적인 탐색 효과성에 결정적인 역할을 한다고 강조합니다. 실험 결과는 알고리즘이 희소 보상 환경에서의 전이 학습 성과를 향상시키는 데 있어, 효과적인 탐색의 필요성을 뒷받침합니다. 이로 인해 관련된 불행한 결과를 피할 수 있으며, 다중 에이전트 환경에서의 협업을 촉진할 수 있습니다.



### Top-Theta Attention: Sparsifying Transformers by Compensated Thresholding (https://arxiv.org/abs/2502.08363)
Comments:
          8 pages, 11 figures, work under submission

- **What's New**: 이번 연구에서는 Transformer 기반 대형 언어 모델의 self-attention 메커니즘의 비효율성을 극복하기 위한 새로운 접근법인 Top-Theta Attention (Top-$\theta$)을 소개합니다. 이 방법은 중요하지 않은 attention 요소를 선택적으로 가지치기하여 효율성을 크게 향상시키며, 모델의 정확도를 유지하는 특징이 있습니다. 특히, generative decoding 동안 V-cache row의 수를 3배, prefill 단계에서 attention 요소의 수를 10배 줄여주는 성과를 보여줍니다.

- **Technical Details**: Top-$\theta$는 주어진 임계값에 따라 attention 요소를 가지치기(thresholding)하여 Sparse한 구조를 활용합니다. 이 방법은 Top-k attention과 달리 전체 벡터 의존성을 제거하며, 이는 모델을 고성능의 커널과 분산 추론에 적합하게 만듭니다. 또한, 모델의 리트레이닝 없이 간단한 보정(calibration) 절차만으로 적용할 수 있어, 여러 데이터셋 간의 분포 변화에도 강한 저항성을 보입니다.

- **Performance Highlights**: Top-$\theta$의 적용 결과, LLaMA2와 LLaMA3 모델에 대해 3배 적은 V-row와 10배 적은 attention 요소로도 동일한 정확도를 유지할 수 있음을 확인하였습니다. 본 연구는 개발한 효율적인 수치 보상 기법들을 통해 가지치기가 진행되는 동안 모델의 정확도를 효과적으로 보존하는 방법도 제안합니다. 이 연구는 대형 언어 모델의 메모리 및 계산 효율성을 높이는 중요한 기여를 할 것으로 기대됩니다.



### Trustworthy GNNs with LLMs: A Systematic Review and Taxonomy (https://arxiv.org/abs/2502.08353)
Comments:
          Submitted to IJCAI 2025

- **What's New**: 이 논문은 Graph Neural Networks (GNNs)와 Large Language Models (LLMs)의 통합이 GNN의 신뢰성을 향상시키는 방식에 대한 포괄적인 검토를 제공한다. 연구자들은 이 통합을 통해 머신 모델의 의사결정 신뢰성에 영향을 미칠 수 있음을 인식하고 있다. 우리는 LLM과 GNN의 통합에 관한 새로운 분류 체계를 제안하며, 이 체계를 통해 각 방법의 적용 가능한 시나리오와 한계를 이해할 수 있다.

- **Technical Details**: 우리는 GNN을 그래프 구조(data structure)로 정의하고, 각 노드는 피처 벡터로 설명된다. GNN은 메시지 전달(message-passing) 메커니즘을 통해 노드 표현을 업데이트하며, 이 과정은 서로 이웃한 노드의 정보를 집계(aggregate)하여 수행된다. 본 논문에서는 GNN의 중심 노드 주변 k-hop 구조 내의 정보에 따라 노드 표현이 업데이트된다는 사실을 강조하며, 이를 통해 다양한 그래프 분석 작업을 수행할 수 있다.

- **Performance Highlights**: LLM을 활용한 GNN은 특히 노드에 풍부한 텍스트 속성이 포함된 경우에 뛰어난 성능을 발휘한다. 모델 신뢰성 향상을 위한 LLM과 GNN의 통합 효능에 대해 여러 연구들이 보고되었으며, 예를 들어 LLM4RGNN은 LLM의 추론 능력을 활용하여 악성 엣지를 식별하고 중요 정보를 복구할 수 있다. 본 논문은 향후 연구 방향으로 GNN과 LLM의 통합을 통해 모델의 신뢰성을 더욱 강화하는 방법을 제안한다.



### Graph Foundation Models for Recommendation: A Comprehensive Survey (https://arxiv.org/abs/2502.08346)
- **What's New**: 이 논문은 추천 시스템(Recommender Systems, RS)에서의 최신 연구 동향을 탐구하고 있으며, 특히 그래프 기반 모델(Graph Foundation Models, GFMs)에 대한 종합적인 개요를 제공합니다. GFMs는 그래프 신경망(Graph Neural Networks, GNNs)과 대형 언어 모델(Large Language Models, LLMs)의 강점을 결합하여 복잡한 추천 문제를 해결하는 새로운 접근 방식을 제시합니다. 이 접근법은 사용자-항목 관계의 그래프 구조와 텍스트 이해를 결합하여 추천의 정확성을 향상시킵니다.

- **Technical Details**: 그래프 기반 추천 시스템의 발전은 GNN을 활용한 협업 필터링의 기초를 다지며, 텍스트 정보와 사용자 선호도 간의 관계를 도출하는 데 중점을 둡니다. 그러나 GNN은 본질적으로 구조적 편향을 지니고 있어 텍스트 정보를 처리하는 데 한계를 겪습니다. 반면, LLM은 자연어 처리(Natural Language Processing, NLP) 분야에서 강력한 성능을 보이며, 추천 시스템 내에서 사용자와 항목의 텍스트 정보를 효과적으로 캡처합니다.

- **Performance Highlights**: GFM 기반의 추천 시스템은 데이터 활용 측면에서 효율성을 극대화하고 사용자 선호도를 정밀하게 조정함으로써 발생하는 편향을 최소화합니다. 이 시스템은 그래프 구조에서의 중요 정보를 적절히 통합하여 추천의 새로운 패러다임으로 자리잡을 잠재력을 가지고 있습니다. 앞으로 GFM을 활용한 기술 발전은 개인화된 추천의 질을 한층 향상시키는 데 기여할 것입니다.



### Hierarchical Learning-based Graph Partition for Large-scale Vehicle Routing Problems (https://arxiv.org/abs/2502.08340)
Comments:
          Accepted as a Full Paper at AAMAS 2025 (24th International Conference on Autonomous Agents and Multiagent Systems)

- **What's New**: 이 논문은 차량 경로 문제(VRP)에 대한 새로운 접근 방식을 제안합니다. 특히, 용량 제약 VRP(CVRP) 문제를 해결하기 위해 글로벌 및 로컬 파티션 정책을 통합한 계층적 학습 기반 그래프 파티션(HLGP) 프레임워크를 도입합니다. 이 방법은 기존의 기술적 한계를 극복하고 재현성을 높이기 위한 것입니다.

- **Technical Details**: HLGP 프레임워크는 다단계 계층 구조를 활용하여 VRP 문제를 해결합니다. 글로벌 파티션 정책이 복잡한 다중 경로 분할을 수행하고, 이후 로컬 파티션 수준에서 해당 단계에 특화된 서브태스크를 생성하여 더욱 정교한 로컬 파티션 정책을 적용합니다. 이는 파티션 과정 중 발생할 수 있는 오류 전파를 줄이는 데 효과적입니다.

- **Performance Highlights**: 제안된 HLGP 프레임워크는 여러 CVRP 벤치마크에 대한 실험을 통해 이전 최첨단 방법들보다 약 10% 성능 향상을 보여줍니다. 특히, CVRP10K 인스턴스에서 Scalability가 강조되며, 강화 학습(RL)과 감독 학습(SL) 모두에 적합한 훈련 목표를 제공합니다.



### Hierarchical Multi-Agent Framework for Carbon-Efficient Liquid-Cooled Data Center Clusters (https://arxiv.org/abs/2502.08337)
- **What's New**: 이 논문은 클라우드 컴퓨팅의 환경 영향을 감소시키기 위해 지리적으로 분산된 데이터 센터 클러스터(DCC) 전반에 효율적인 작업 부하 분배와 개별 데이터 센터 내에서의 냉각 최적화를 동시에 추구하는 Green-DCC를 소개합니다. 특히, Reinforcement Learning (RL) 기반의 계층적 컨트롤러를 통해 작업 부하와 액체 냉각을 동적으로 최적화하는 방법을 제시합니다.

- **Technical Details**: Green-DCC는 날씨, 탄소 강도(carbon intensity), 자원 가용성과 같은 다양한 요소를 고려하여 현실적인 제약과 상호 의존성을 해결합니다. 이 시스템은 여러 데이터 센터를 동시에 최적화할 수 있는 기능을 제공해 디지털 트윈(digital twins)의 범위를 확대합니다. 이를 통해 다양한 RL 접근 방식의 성능을 탄소 배출과 지속 가능성 메트릭을 기준으로 비교합니다.

- **Performance Highlights**: 논문에서는 Green-DCC의 효율성을 입증하기 위해 여러 DCC의 냉각과 작업 부하를 동시적으로 최적화한 사례를 보여줍니다. 또한 지속 가능성 연구를 위한 프레임워크와 벤치마크 시뮬레이션을 제공하여 더 넓은 기계 학습(ML) 연구에 기여하는 방향을 제시합니다.



### Modification and Generated-Text Detection: Achieving Dual Detection Capabilities for the Outputs of LLM by Watermark (https://arxiv.org/abs/2502.08332)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)의 악용 가능성에 대한 우려를 해소하기 위해 새로운 수정 검출 기법을 제안합니다. 특히, 수정이 일어나는 경우에도 원본 워터마크가 여전히 유지될 수 있어 기만적 공격의 위험을 증가시킵니다. 기존의 워터마킹 방법들이 이러한 수정 공격에 취약한 문제를 해결하기 위해, 수정 민감도가 높은 공정한 워터마크 감지 방법을 도입하였습니다.

- **Technical Details**: 제안된 방법은 'discarded tokens'라는 새로운 지표를 활용하여 수정된 텍스트의 변화를 감지합니다. 이 지표는 수정이 발생할 경우 변화가 생기고, 이를 통해 텍스트의 수정 여부를 입증할 수 있습니다. 또한, 기존의 maximin 변형 로그 가능도 비율 점수(mmLLR)를 개선하여 drLLR이라는 새로운 워터마크 감지 점수를 도입하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 수정 검출 기법(IDD)은 추가, 삭제, 대체와 같은 수정의 높은 정확도를 달성하며, 동시에 원본 워터마크의 추출이 가능하여 생성된 텍스트의 검출도 수행할 수 있음을 보여줍니다. 이러한 이중 검출 기능 덕분에, 텍스트가 생성되었는지 여부와 함께 생성된 텍스트의 수정 여부를 동시에 검토할 수 있는 두 가지 중요한 기능이 가능해집니다.



### Mitigating Hallucinations in Multimodal Spatial Relations through Constraint-Aware Prompting (https://arxiv.org/abs/2502.08317)
Comments:
          19 pages, accepted to NAACL Findings

- **What's New**: 이 논문에서는 공간적 관계의 환각(spatial relation hallucinations)을 줄이기 위해 제약 인식 프롬프트 프레임워크(constraint-aware prompting framework)를 제안합니다. 특히, 두 가지 제약 조건을 도입하는데, 이들은 쌍방향 제약(bidirectional constraint)과 전이성 제약(transitivity constraint)입니다. 쌍방향 제약은 두 객체 간의 관계가 양방향에서 일관되도록 보장하며, 전이성 제약은 다수의 객체 간의 관계에 대한 논리적 일관성을 유지합니다. 이러한 제약을 통합하여 LVLMs는 더 공간적으로 일관된 출력을 생성할 수 있습니다.

- **Technical Details**: 제안하는 방법은 주로 이미지-질문 쌍을 입력으로 하는 공간적 관계 이진 VQA를 위해 설계되었습니다. 연구팀은 약속된 구조에 따라 지침과 출력 형식을 설정하고, 해당 질문에 대한 공간적 관계를 분석하는 것을 목표로 합니다. 논문에서는 제로샷 체인 오브 씽킹(zero-shot chain-of-thought) 프롬프트를 사용하여 LVLMs가 감지된 공간적 관계에 기반하여 효과적으로 추론하도록 합니다. 이 과정에서 수평, 수직 및 깊이 관계를 분석하도록 명시적인 지침을 제공합니다.

- **Performance Highlights**: 세 개의 널리 사용되는 공간적 관계 데이터셋을 평가하여 제안된 방법의 성능을 검증하였습니다. 결과적으로 제안한 방법은 기존의 접근 방법보다 성능이 향상되었으며, 전체 방법이 다른 두 제약 방법보다 우수한 성능을 보임을 확인했습니다. 이 연구에서는 다양한 방법 변형의 성과를 분석하고, 데이터셋 간 성과의 변동성을 강조하여 제안한 접근법의 효과성을 입증하고 있습니다.



### HDT: Hierarchical Discrete Transformer for Multivariate Time Series Forecasting (https://arxiv.org/abs/2502.08302)
- **What's New**: 이 논문에서는 고차원 다변량 시계열 예측에서의 한계를 해결하기 위해 새로운 접근 방식을 제안합니다. 기존의 생성 모델들은 고차원 데이터에 대한 예측 성능이 낮았고 예측 길이도 제한적이었습니다. 새로운 기법으로 제안된 Hierarchical Discrete Transformer(HDT)는 시계열 데이터를 이산 토큰 표현으로 변환하여 생성 성능을 개선합니다.

- **Technical Details**: HDT는 l2 정규화(l2 normalization)와 강화된 벡터 양자화(vector quantized) 전략을 적용하여 시계열 예측을 이산 토큰 생성으로 전환합니다. 이 모델은 저수준에서의 이산 장기 트렌드를 포착하고, 이를 고수준에서의 타겟 이산 표현 생성을 위한 조건으로 활용합니다. 이러한 접근은 다변량 시계열 데이터의 특성을 반영하여 예측 길이를 연장합니다.

- **Performance Highlights**: 다섯 개의 인기 있는 MTS 데이터셋에 대한 광범위한 실험을 통해 제안된 방법의 효과가 입증되었습니다. HDT는 기존 모델들에 비해 높은 정확도로 예측을 수행할 수 있으며, 고차원 다변량 시계열의 예측을 위한 성능 향상을 보여주었습니다.



### Compromising Honesty and Harmlessness in Language Models via Deception Attacks (https://arxiv.org/abs/2502.08301)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 속임수 행동을 탐지하고 이를 활용하는 새로운 공격 방법을 소개합니다. 이러한 속임수 능력은 과거에 드물게 관찰되었으나, 본 연구는 이러한 성향을 극대화하는 방법을 통해 전혀 새로운 차원의 위험을 드러냈습니다. 이 연구는 특히 고유한 주제에 대한 프롬프트(prrompt)에 부합하여 사용자에게 잘못된 정보를 제공하는 방법을 탐구합니다.

- **Technical Details**: 속임수 공격(deception attacks)을 통해 LLMs의 진실성과 무해함을 저하시킬 수 있는 방법론을 제시하고 있습니다. 세세한 조정(fine-tuning)을 통해 특정 주제에 대해 의도적으로 사용자에게 잘못된 정보를 전할 수 있는 모델을 조정합니다. 이 연구는 이러한 모델들이 다수의 대화(turns)에서도 일관되게 속임수를 사용할 수 있는지를 평가하며, 그 결과는 혼재된 양상을 보입니다.

- **Performance Highlights**: 연구 결과, 조정된 모델들은 사용자에게 혐오 발언(hate speech), 고정관념(stereotypes) 등 유해한 콘텐츠를 생성하는 경향을 보였습니다. 또한, 대화형 인터페이스(dialogues)에서의 속임수 행위는 수백만 사용자와의 상호작용에서 신뢰성을 보장하기 위해 필수적으로 대처해야 할 과제로 지적됩니다. 이 연구는 LLM이 사용자에게 미치는 영향을 고려하여 이들을 속임수 공격으로부터 보호하는 것이 중요함을 강조하고 있습니다.



### CRISP: A Framework for Cryo-EM Image Segmentation and Processing with Conditional Random Field (https://arxiv.org/abs/2502.08287)
Comments:
          31 pages, 28 Figures

- **What's New**: 이번 논문에서는 cryogenic electron microscopy (cryo-EM) 데이터로부터 고품질의 segmentation maps를 자동으로 생성하는 파이프라인을 제안합니다. 이 프레임워크는 다양한 segmentation 모델과 손실 함수(Loss Functions)를 선택할 수 있는 모듈형 구조로 설계되어 있습니다. 또한, Conditional Random Fields (CRFs)을 통합하여 거친 예측을 정제하여 세밀한 segmentation을 생성합니다. 이 접근법은 cryo-EM 데이터셋에 최적화된 구성을 용이하게 하며, 정확한 레이블 생성을 통해 고해상도 단백질 구조를 만드는 데 기여합니다.

- **Technical Details**: 제안하는 이미지 분석 파이프라인은 픽셀 수준에서 학습하여 미지의 마이크로그래프에서도 높은 정확도를 달성합니다. 세분화 모델 학습 시, 노이즈가 포함된 마이크로그래프와 해당 세분화 맵을 입력으로 사용합니다. 이 모듈형 파이프라인은 다양한 모델, 인코더, 손실 함수 및 성능 지표를 선택할 수 있어 유연성과 사용자 맞춤 가능성을 제공합니다. 또한, 파이프라인에서는 정밀도가 부족한 레이블 예측을 개선하기 위해 조건부 랜덤 필드를 사용하여 깔끔한 경계와 세밀한 세분화를 달성합니다.

- **Performance Highlights**: 연구 결과, 제한된 마이크로그래프 세트로 훈련된 모델은 합성 데이터에서 90% 이상의 정확도, 재현율, 정밀도, Intersection over Union (IoU) 및 F1-score를 달성했습니다. 실제 실험 데이터셋에서 이 파이프라인을 통해 추출된 입자는 기존 파커보다 더 높은 해상도의 3D 밀도 맵을 생성하고, 전문가가 큐레이트한 데이터셋과 유사한 성능을 발휘했습니다. 특히, 본 연구에서 개발한 모델은 원본 데이터셋에 레이블이 없는 입자도 식별할 수 있어 배경 잡음에서 신호를 구분하는 일반화 가능성을 보여줍니다.



### Individualised Treatment Effects Estimation with Composite Treatments and Composite Outcomes (https://arxiv.org/abs/2502.08282)
Comments:
          6 pages (double column), 4 figures

- **What's New**: 이 논문에서는 복합 치료(composite treatments) 및 복합 결과(composite outcomes) 설정에서 개별화된 치료 효과(Individualized Treatment Effect, ITE) 추정을 위한 혁신적인 방법인 H-Learner를 제안합니다. H-Learner는 데이터 부족 문제를 해결하기 위해 치료와 결과 간의 정보를 동적으로 공유하는 하이퍼네트워크(hypernetwork) 기반 접근 방식입니다. 이 연구는 기계 학습(machine learning) 방법론을 기존의 ITE 추정 기술에 통합하여 복잡한 실제 시나리오에서의 적용 가능성을 높이고 있습니다.

- **Technical Details**: H-Learner는 특정 치료 조합에 대해 목표 학습기(target learner)를 생성하는 하이퍼네트워크로 구성됩니다. 이 네트워크는 치료 및 결과에 따라 조건화되어 효과적인 ITE 추정을 제공합니다. 기존의 방법들은 단일 치료와 단일 결과에 한정되어 있었지만, H-Learner는 복수의 치료 및 결과를 동시에 처리할 수 있는 능력을 갖추고 있습니다. 이를 통해 치료와 결과 간의 상관관계를 더욱 잘 모델링 할 수 있습니다.

- **Performance Highlights**: H-Learner는 다양한 치료 및 결과 조합을 포함하는 실증 분석을 통해 기존의 방법들과 비교하여 우수한 성능을 보였습니다. 본 연구는 복합 치료 및 결과 설정에서 H-Learner의 효과를 강조하며, 다양한 실제 사례를 다루고 있어 건강 관리와 같은 분야에서 큰 잠재력을 가지고 있습니다. 이 방법론이 전통적인 ITE 추정 방법들에 비해 중요한 발전을 이룬 것을 확인할 수 있습니다.



### What Is That Talk About? A Video-to-Text Summarization Dataset for Scientific Presentations (https://arxiv.org/abs/2502.08279)
Comments:
          arXiv admin note: text overlap with arXiv:2306.02873 by other authors

- **What's New**: 본 논문은 과학 분야의 비디오-텍스트 요약을 위해 특별히 설계된 VISTA 데이터셋을 소개합니다. VISTA는 18,599개의 AI 컨퍼런스 발표와 해당 논문의 초록으로 구성되어 있으며, 멀티모달 학습에서의 더욱 효율적인 요약 생성을 목표로 합니다. 인공지능 비디오 요약 문제의 명확한 해결을 모색하기 위해 최신 대형 모델들이 benchmark 되었고, 계획 기반 프레임워크를 통해 요약 품질과 사실 일관성을 개선했습니다.

- **Technical Details**: VISTA 데이터셋은 컴퓨터 언어학 및 기계 학습의 주요 컨퍼런스에서 수집된 기록된 발표와 논문 초록 간의 짝을 이루는 18,599개의 쌍으로 구성되어 있습니다. 다양한 대형 모델에 대한 비교 실험을 통해 in-domain fine-tuning이 요약 성능을 개선시키고, 비디오 기반 모델이 텍스트 및 오디오 기반 모델보다 일반적으로 우수한 성능을 보임을 발견했습니다. 연구는 계획 기반 접근 방식을 통해 과학 초록의 기본 구조를 더 잘 포착할 수 있음을 보여줍니다.

- **Performance Highlights**: 인간과 자동화된 평가 모두에서 명시적 계획이 요약 품질과 사실적 일관성을 향상시킴을 확인했습니다. 계획 기반 접근 방식이 최신 SOTA 모델들보다 우수한 성능을 나타내었지만, 모든 후보 모델에서 여전히 사실 오류와 환각 문제에 직면해 있는 점이 지적되었습니다. VISTA 데이터셋을 통해 과학 비디오 요약의 도전과제를 제시하며, 이 분야에 대한 연구의 필요성을 강조했습니다.



### Dealing with Annotator Disagreement in Hate Speech Classification (https://arxiv.org/abs/2502.08266)
- **What's New**: 본 논문은 증오 발언(hate speech) 분류에 대한 주석자 간의 의견 불일치 문제를 심층적으로 다루고, 여러 가지 접근 방식을 평가합니다. 특히, 주석자 간의 불일치 문제 해결을 위한 새로운 방법론을 제안하며, 고품질의 데이터셋을 확보하기 위한 다양한 전략을 탐구합니다. 이 연구는 터키어 트윗을 기반으로 하여 필터링된 BERT 모델을 활용한 최신 성능 벤치마크 결과를 제공합니다.

- **Technical Details**: 본 연구는 주석 과정을 통해 발생하는 주관적 불일치 문제에 집중하며, 다양한 방법(예: 최대값, 최소값, 무작위 선택 및 평균)을 통해 가장 정확한 레이블을 결정하는 방안을 모색합니다. 또한 주석자의 신뢰도 차이를 고려하여 가중된 버전의 접근 방법도 평가합니다. 이를 통해 데이터셋의 품질을 향상시키고, 나아가 증오 발언 탐지 모델의 신뢰성을 높이는 데 기여하고자 합니다.

- **Performance Highlights**: 필요한 트레이닝 데이터 확보의 중요성을 강조하며, 제안된 방법론을 통해 공연별 감지 및 이해에서 최첨단 성과를 달성했습니다. 이 연구는 튼튼한 데이터셋을 바탕으로 한 정확한 증오 발언 탐지의 필요성을 재확인하고, 이를 통해 다양한 자연어 처리(NLP) 작업에서의 성능을 향상시키는 데 기여합니다.



### Exploring the Potential of Large Language Models to Simulate Personality (https://arxiv.org/abs/2502.08265)
Comments:
          Preprint submitted to Workshop on Customizable NLP (CustomNLP4U) on EMNLP2024

- **What's New**: 이번 연구는 LLM(대형 언어 모델)을 사용하여 Big Five 성격 모델에 따라 개인 특성을 모사하는 방법을 제시합니다. 연구 결과, 성격 관련 텍스트를 생성하는 것이 여전히 LLM에게 도전적인 과제임을 보여주었습니다. 따라서, 사전 정의된 Big Five 특성을 가진 생성 텍스트의 데이터셋과 LLM의 개인화된 대화 기능을 테스트하기 위한 분석 프레임워크를 제공합니다.

- **Technical Details**: 연구는 LLM의 성격 분석을 두 단계로 나누어 수행합니다. 첫 번째 단계에서는 성격 특성과 관련된 행동 간의 연결을 이해하는 능력을 평가하기 위해 성격 질문지를 사용합니다. 두 번째 단계에서는 유도된 성격에 대한 텍스트 생성을 LLM에 요청하여, 생성된 텍스트를 인간 평가, 자동 평가 및 언어적 특성 분석 등 다양한 방법으로 분석합니다.

- **Performance Highlights**: 테스트 결과, 일부 LLM 모델은 특정 성격 특성에 대한 정확한 성과를 보여주었으며, Claude 모델은 개방성과 성실성에 대해 높은 이해도를 보였습니다. GPT 시리즈 모델은 외향성의 높은 및 낮은 수준을 구별하는 능력이 뛰어났으며, GPT-4 Omni 모델은 우호성에 대한 구분에서 우수한 성능을 보였습니다. 이 연구 결과는 LLM이 개인 특성을 모사하고 이를 기반으로 사용자와의 대화를 더욱 풍부하게 만드는 데 기여할 것으로 기대됩니다.



### Balancing optimism and pessimism in offline-to-online learning (https://arxiv.org/abs/2502.08259)
- **What's New**: 이 연구는 오프라인 데이터로부터 시작해 온라인 환경과 상호작용하는 학습자에게 적용되는 새로운 알고리즘을 소개합니다. 오프라인-온라인 학습 문제를 다루며, 비관적(pessimism) 접근법인 Lower Confidence Bound (LCB)와 낙관적(optimism)인 Upper Confidence Bound (UCB) 알고리즘의 균형을 최적화합니다. 이 알고리즘은 다양한 시간 수평선에서 성능을 조정할 수 있으며, 현재까지의 연구에서 다뤄지지 않은 변화를 제안합니다.

- **Technical Details**: 연구에서는 학습자의 정책이 단기적으로 사용되면 LCB가 유리하다는 점을 강조합니다. 반면, 장기 전망에서는 UCB 전략이 최적의 정책에 대한 수렴 속도가 가장 빠르다는 강점을 가집니다. 이 알고리즘은 또한 상호작용이 증가함에 따라 LCB에서 UCB와 유사한 전략으로 점진적으로 전환하는 방법을 탐구합니다.

- **Performance Highlights**: 새로운 알고리즘은 LCB와 UCB 중 더 나은 성능을 가지는 쪽과 비슷한 성능을 보입니다. 연구 결과는 다양한 MAB(multi-armed bandit) 문제에 일반화될 수 있다고 예상되며, 본 알고리즘은 대안적 탐험 및 활용 전략을 통합하여 효율성을 높입니다.



### TRISHUL: Towards Region Identification and Screen Hierarchy Understanding for Large VLM based GUI Agents (https://arxiv.org/abs/2502.08226)
Comments:
          Under review at ICML 2025, 8 pages 5 figures

- **What's New**: TRISHUL은 기존 LVLM보다 더 포괄적인 GUI 이해를 위해 개발된 새로운 프레임워크입니다. 이 시스템은 Hierarchical Screen Parsing (HSP)와 Spatially Enhanced Element Description (SEED) 모듈을 활용하여 행위 지향(mapping instructions to GUI elements) 및 GUI 참조(task) 작업을 통합적으로 처리합니다. 기존 방법들이 주로 특정 작동(task)에 특화되어 있었던 반면, TRISHUL은 다양한 GUI 상호작용 작업을 지원하고, 훈련이 필요 없는 방식으로 설계되었습니다.

- **Technical Details**: HSP 모듈은 GUI 요소를 Global Regions of Interest (GROIs) 및 Local Elements (LE)로 구분하여 데이터의 위계적 이해를 가능하게 합니다. SEED 모듈은 GUI의 요소들 간의 상대적 위치를 분석하여 각각의 요소에 대한 고수준의 기능 설명을 생성합니다. 이 프로세스는 SAM 및 EasyOCR 알고리즘을 활용하여 GUI 내 텍스트 및 아이콘을 효과적으로 처리합니다.

- **Performance Highlights**: TRISHUL은 ScreenSpot, VisualWebBench, Mind2Web 및 AITW 데이터셋에서 기존의 최첨단 기법들을 초월하는 성능을 보여줍니다. 특히, TRISHUL 기반의 GPT-4V와 GPT-4o는 action grounding과 episodic instruction-following 작업에서 우수한 성과를 발휘하였습니다. 또한, Screen PR 데이터셋에서도 GUI 참조 성능을 향상시켜 타겟의 접근성과 사용자 상호작용 피드백을 개선하였습니다.



### Quality over Quantity: Boosting Data Efficiency Through Ensembled Multimodal Data Curation (https://arxiv.org/abs/2502.08211)
- **What's New**: 이번 연구에서는 데이터 커레이션(data curation)의 새로운 프레임워크인 EcoDatum을 제안합니다. EcoDatum은 여러 단일 모드 및 다중 모드 데이터 커레이션 연산자를 통합하여 약한 감독 학습의 앙상블(ensemble) 구조를 활용하며, 자동 최적화를 통해 각 데이터 포인트의 품질 점수를 효과적으로 평가합니다. 이 프레임워크는 데이터 커레이션 품질과 효율성을 크게 향상시켜, 기존 기술보다 뛰어난 성과를 거두었습니다.

- **Technical Details**: EcoDatum은 질적 가이드에 따라 중복 제거(quality-guided deduplication) 방법을 도입하여 균형 잡힌 특성 분포를 보장합니다. 또한, 시각적 콘텐츠와 텍스트를 기반으로 해시 코드를 생성하여 중복을 식별하며, CLIP 모델을 통해 각 중복 그룹의 의미적 일관성을 평가합니다. 이를 통해 데이터 커레이션 과정이 자동화되어 품질 점수가 생성되며, 수동 입력을 최소화하고 임계값 설정의 정확성이 향상됩니다.

- **Performance Highlights**: EcoDatum은 DataComp 리더보드에서 1위를 기록하며, 38개의 다양한 평가 데이터셋에서 평균 성능 점수 0.182를 달성하였습니다. 이는 DataComp 기준 방법에 비해 28% 향상된 결과입니다. 이러한 성과는 데이터셋 커레이션과 모델 훈련 효율성을 크게 개선할 수 있는 가능성을 보여줍니다.



### Equivariant Masked Position Prediction for Efficient Molecular Representation (https://arxiv.org/abs/2502.08209)
Comments:
          24 pages, 6 figures

- **What's New**: 이번 연구에서는 그래프 신경망(GNN)이 화학 분야에서의 적용 가능성을 그러면서도, 분자 데이터의 한계로 인해 일반화 능력이 제한된다는 문제를 해결하기 위해 새로운 자기 지도 학습 방식인 Equivariant Masked Position Prediction(EMPP)을 제안합니다. EMPP는 전통적인 속성 마스킹 기술과 달리 intramolecular potential 및 force 이론에 기반하여 실질적인 위치 예측 과제를 형성하여 양자역학적 특징 학습을 향상시킵니다.

- **Technical Details**: EMPP는 원자의 3D 위치를 무작위로 마스킹하고, 이는 다른 속성, 예를 들어 원자 번호 등을 그대로 유지함으로써 더 잘 정의된 문제를 만들어 냅니다. EMPP는 마스킹된 원자의 위치를 인근 구조에서 양자역학적으로 결정하며, 이는 기존의 노이즈를 예측하는 방법과는 기본적으로 다른 접근 방식을 채택합니다. EMPP는 Gaussian mixture 분포의 근사를 피하고, 더 결정적인 위치 예측을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, EMPP는 고급 화학 GNN 구조의 성능을 크게 향상시켰고, 기존의 마스킹 및 노이즈 방법보다 우수한 성과를 달성했습니다. 본 연구에서 제안한 EMPP는 자가 감독 학습 과정을 통해 일반 지식을 포착하며, 특정 양자 특성과 원자 위치 간의 연결을 강화하여 다양한 분자 작업에서 GNN의 일반화 성능을 개선합니다.



### Latest Advancements Towards Catastrophic Forgetting under Data Scarcity: A Comprehensive Survey on Few-Shot Class Incremental Learning (https://arxiv.org/abs/2502.08181)
- **What's New**: 이번 논문은 데이터 희소성이 끼치는 영향을 다루며, Few-shot Class Incremental Learning (FSCIL) 방법론에 대한 포괄적인 조사를 제공합니다. FSCIL은 동적 환경에서 소수의 샘플로 학습해야 하는 머신러닝 모델의 문제를 모사합니다. 최근 진행된 연구들에서 파생된 이 방법은 기존의 학습 방식의 한계를 넘어서는 솔루션을 모색하고 있습니다.

- **Technical Details**: FSCIL 문제는 주어진 작업 시퀀스에서 각 작업이 레이블링된 훈련 세트를 포함하고 있으며, 두 번째 작업 이후의 데이터는 초기 작업보다 훨씬 적은 샘플을 가지고 있음을 보여줍니다. 데이터 희소성 문제를 해결하기 위한 최근의 PEFT(parameter-efficient fine-tuning) 접근 방법과 언어 기반 학습 방식이 특히 강조되고 있습니다. 이는 기존 모델의 성능 의존성을 줄이고, 모델 훈련 시간을 대폭 단축시켜 줍니다.

- **Performance Highlights**: 이 논문은 FSCIL 방법의 최신 발전을 포함한 포괄적인 토폴로지를 제안하며, 각 접근 방식의 공식 목표와 서브 설정을 분석합니다. 프로토타입 편향을 해결하기 위한 프로토타입 정정의 중요성도 강조하며, 현재 직면한 개방 문제, 잠재적 솔루션 및 FSCIL의 미래 방향에 대한 철저한 분석을 제공합니다.



### Enhancing LLM Character-Level Manipulation via Divide and Conquer (https://arxiv.org/abs/2502.08180)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 문자 수준 조작에서 드러나는 중요한 한계를 체계적으로 분석하고, 이를 극복하기 위한 새로운 접근 방식을 제안합니다. 특히, 기존의 LLM들이 토큰화 과정을 거치면서 발생하는 문자 수준 작업에서의 어려움을 강조하며, LLM이 문자 조작을 보다 잘 수행할 수 있도록 돕는 방법론을 소개합니다.

- **Technical Details**: 제안된 방법인 'Character-Level Manipulation via Divide and Conquer'는 복잡한 작업을 명시적인 문자 수준의 하위 작업으로 분해하고 제어된 토큰 재구성 단계를 결합하는 방식으로 설계되었습니다. 이 접근법은 추가적인 훈련 없이도 LLM의 정확도를 향상시킬 수 있으며, 삭제(Deletion), 삽입(Insertion), 대체(Substitution) 작업에서 성능이 크게 개선됨을 입증합니다.

- **Performance Highlights**: 실험 결과, GPT-3.5를 사용할 때 제안된 방법이 기존의 방법보다 모든 문자 조작 작업에서 현저히 우수한 성과를 보였습니다. 이 분석은 LLM의 문자 수준 처리 메커니즘에 대한 귀중한 통찰력을 제공하며, 미래 연구에서 LLM의 문자 수준 추론 능력을 더욱 강화할 수 있는 방향성을 제시합니다.



### MixDec Sampling: A Soft Link-based Sampling Method of Graph Neural Network for Recommendation (https://arxiv.org/abs/2502.08161)
Comments:
          10 pages, 6 figures

- **What's New**: 이 논문에서는 GNN 기반 추천 시스템에서의 새로운 샘플링 방법인 MixDec Sampling을 제안합니다. 기존의 부정 샘플링 방법이 하드 양성 쌍 또는 하드 음성 쌍으로 제한되었던 점을 극복하기 위해, 소프트 링크를 통한 샘플링 관계 모델링을 최초로 시도하고 있습니다. MixDec Sampling은 Mixup Sampling 모듈과 Decay Sampling 모듈로 구성되어 있어, 소수의 이웃을 가진 노드의 충분한 샘플 수를 제공할 수 있도록 설계되었습니다.

- **Technical Details**: Mixup Sampling 모듈은 노드의 특성을 증대시키기 위해 새로운 노드와 소프트 링크를 생성하여, 하드 양성 및 음성 샘플의 특성을 베타 분포에 따라 선형적으로 혼합합니다. Decay Sampling 모듈은 BFS 방식을 이용해 노드 간 링크의 가중치를 감소시켜 그래프 구조 정보를 강화합니다. 이 두 모듈이 협력하여, GNN의 중심 아이디어인 이웃 노드의 특성을 융합하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, MixDec Sampling을 적용한 GNN 기반 추천 모델이 기존의 부정 샘플링 방법보다 다양한 추천 벤치마크에서 유의미한 성능 향상을 보여주었습니다. 특히, GraphSAGE와 GCN의 평균 역순위(Mean Reciprocal Rank, MRR)가 각각 18.3%와 21.1% 증가하는 성과를 기록했습니다. 이는 MixDec Sampling이 GNN 기반 모델의 추천 품질을 일관성 있게 향상시킬 수 있는 가능성을 제시합니다.



### Vertical Federated Learning in Practice: The Good, the Bad, and the Ugly (https://arxiv.org/abs/2502.08160)
- **What's New**: 이 논문에서는 Vertical Federated Learning (VFL)의 기존 연구와 실제 적용 간의 격차를 분석합니다. 연구팀은 VFL 알고리즘을 실제 데이터 분포를 기반으로 한 새로운 데이터 중심 분류도로 제안하고, 이를 통해 현재의 알고리즘과 실제 응용 간의 큰 간극을 드러냅니다. VFL이 여러 산업에서의 협업을 가능하게 함에도 불구하고, 실제 적용 사례가 극히 제한적이라는 점이 강조됩니다.

- **Technical Details**: VFL은 서로 다른 특성을 가진 여러 당사자가 공동으로 기계 학습 모델을 훈련할 수 있도록 하는 개인 정보 보호 협업 학습 패러다임입니다. VFL 작업은 하나의 주요 당사자가 라벨을 보유하고 있으며, 다른 당사자는 보조 당사자로 정의됩니다. 이 논문은 {𝐗1,𝐗2,…,𝐗C} 형태의 결합된 데이터셋에서 VFL 모델을 훈련하는 과정과, 개인 정보 보호를 위한 레코드 연결 기법을 다룹니다.

- **Performance Highlights**: VFL의 잠재적 데이터 분포를 조사한 결과, 특정 실용적 시나리오에 대한 유용한 솔루션이 부족하다는 사실이 밝혀졌습니다. 기존 알고리즘이 실제 데이터 분포와 잘 맞지 않음을 지적하며, 이 격차를 메우기 위한 연구 방향을 제시합니다. 이로 인해 VFL 관련 향후 연구는 실제 응용 프로그램의 특성을 반영하도록 초점을 맞춰야 할 필요성이 있음을 강조합니다.



### DGSense: A Domain Generalization Framework for Wireless Sensing (https://arxiv.org/abs/2502.08155)
Comments:
          15 pages

- **What's New**: 본 논문은 무선 감지에서 도메인 의존성 문제를 해결하기 위해 새로운 도메인 일반화 프레임워크인 DGSense를 제안합니다. DGSense는 다양한 감지 작업과 무선 기술에 적용할 수 있으며, 새롭게 나타나는 도메인에서도 학습 데이터 없이 일반화할 수 있는 능력을 가지고 있습니다. 이 프레임워크는 가상 데이터 생성기와 에피소드 학습 전략을 채택하여 훈련 세트의 다양성을 높이고 도메인 독립적인 특성을 추출합니다.

- **Technical Details**: DGSense는 주요 피처 추출기와 도메인 피처 추출기 간의 에피소드 학습을 통해 도메인 독립적인 특성을 획득하며, 공간적 특성을 위한 Residual Network (ResNet)와 시간적 특성을 위한 1D Convolutional Neural Network (1DCNN)를 사용합니다. 가상 데이터 생성기는 Variational Autoencoder (VAE)를 기반으로 하여 다양한 모달리티를 보장하고 데이터 희소성을 완화하는 데 도움을 줍니다. 이를 통해 더 많은 데이터 다양성과 모델의 강건성을 확보할 수 있습니다.

- **Performance Highlights**: DGSense는 WiFi 제스처 인식, 밀리미터파(mmWave) 활동 인식, 음향 낙상 감지 등에서 높은 일반화 능력을 입증하였습니다. 모든 시스템은 새로운 사용자, 장소, 환경에 대해서도 새로운 데이터나 재훈련 없이 높은 성능을 유지하는 것으로 나타났습니다. 실험 결과 DGSense의 효과성과 일반성을 확인할 수 있었습니다.



### Force Matching with Relativistic Constraints: A Physics-Inspired Approach to Stable and Efficient Generative Modeling (https://arxiv.org/abs/2502.08150)
- **What's New**: 본 논문에서는 Generative Modeling의 새로운 틀인 Force Matching (ForM)을 소개합니다. 이 프레임워크는 특수 상대성 이론을 적용하여 샘플링 과정의 안정성을 향상시키기 위한 초기 탐험을 나타냅니다. Lorentz 인자를 포함함으로써, ForM은 속도 제약을 부여하여 샘플의 속도를 일정한 한계 내에서 유지하도록 보장합니다.

- **Technical Details**: ForM은 샘플 속도를 제한하는 기본 메커니즘을 통해 생성 동역학을 안정화합니다. 본 논문에서는 ForM 프레임워크 내에서 샘플링 절차 전반에 걸쳐 속도 제약이 유지된다는 것을 엄밀한 이론적 분석을 통해 증명합니다. ForM의 효용성을 검증하기 위해 다양한 실험적 평가를 수행하였습니다.

- **Performance Highlights**: empirical evaluations에서 ForM은 baseline 방법보다 월등한 성능을 보였습니다. 특히, half-moons 데이터셋에서 ForM은 0.714의 Euclidean distance loss를 기록하며, 이는 기존의 vanilla first-order flow matching(5.853) 및 first- and second-order flow matching(5.793)보다 현저히 낮은 수치입니다. 본 연구는 ForM을 통해 안정적이고 효율적인 생성 프로세스를 구현할 수 있는 가능성을 보여줍니다.



### Generalized Class Discovery in Instance Segmentation (https://arxiv.org/abs/2502.08149)
Comments:
          AAAI 2025

- **What's New**: 이번 연구는 instance segmentation(객체 분할)에서 generalized class discovery(GCD) 작업을 다룹니다. 이 과정에서 기존의 알려진 클래스와 새로운 클래스 모두를 분할하는 모델을 개발하기 위해 레이블이 있는 데이터와 없는 데이터를 활용하며, 이는 실제 세계에서의 불균형한 객체 분포를 고려합니다. 제안된 방법은 instance-wise temperature assignment(인스턴스별 온도 할당) 기법과 class-wise reliability criteria(클래스별 신뢰성 기준)를 통해 수업 간 불균형 문제를 해결하고자 합니다.

- **Technical Details**: 연구에서 제안된 ITA 방법은 헤드 클래스의 샘플에 대한 인스턴스 간 구별을 강화하면서, 일반적인 컨트라스트 학습 손실들이 헤드 및 테일 클래스를 동일하게 처리하는 것을 완화하는 것을 목표로 합니다. 또한, 클래스별 신뢰성 기준을 통해 레이블이 없는 데이터에서 발견되는 대다수의 테일 클래스의 의사 레이블을 제외하지 않도록 하는 방안을 제시합니다. 마지막으로, 공간 풀링과 깊이 감소에 기반한 효율적인 soft attention 모듈을 도입하여 GCD를 위한 객체-특화 표현을 인코딩합니다.

- **Performance Highlights**: 두 가지 설정인 COCO$_{half}$ + LVIS 및 LVIS + Visual Genome에서 제안된 방법의 실험 결과가 기존의 최신 방법들보다 뛰어난 성능을 보여주었습니다. 구체적으로, 기존의 중요한 프레임워크들과 비교했을 때, 제안된 방법은 레이블이 있는 데이터와 없는 데이터를 활용하여 새로운 클래스를 발견하고 정확한 인스턴스 분할 성능을 향상시키는 데 효과적임을 입증했습니다.



### Democratizing AI: Open-source Scalable LLM Training on GPU-based Supercomputers (https://arxiv.org/abs/2502.08145)
- **What's New**: 이 논문에서는 수십억 개의 매개변수를 가진 대형 언어 모델(LLM)을 효율적으로 훈련하기 위한 새로운 네 차원 하이브리드 병렬 알고리즘을 소개합니다. 이 알고리즘은 AxoNN이라는 오픈소스 프레임워크에 구현되어 있으며, 뛰어난 성능 최적화를 포함하여 모델 훈련의 속도와 효율성을 크게 향상시킵니다. 특히, Perlmutter, Frontier 및 Alps와 같은 슈퍼컴퓨터에서의 훈련 성능은 기적적인 수치인 1.423 Exaflop/s, 1.381 Exaflop/s 및 620.1 Petaflop/s를 기록했습니다.

- **Technical Details**: 설계된 4D 하이브리드 접근 방식은 세 차원 행렬 곱셈 알고리즘과 데이터 병렬성을 결합하여 많은 GPU에서 높은 효율성을 달성합니다. AxoNN의 성능은 각 플랫폼에 따라 행렬 곱셈을 조정하고, 계산과 비차단 집단 통신을 겹치는 최적화를 통해 개선되었습니다. 이 알고리즘은 다양한 GPU에서의 최적 수행 구성을 예측할 수 있는 통신 모델도 포함하고 있습니다.

- **Performance Highlights**: AxoNN은 NVIDIA A100 GPU에서 620.1 Petaflop/s, AMD MI250X GCD에서 1.381 Exaflop/s, NVIDIA H100 GPU에서 1.423 Exaflop/s의 전례 없는 성능을 달성했습니다. 이러한 결과는 LLM 훈련에서의 메모리 사용과 계산 효율성을 크게 향상시키며, 연구자들이 LLM의 작동 메커니즘을 보다 심도 있게 연구할 수 있도록 도와줍니다. 또한, 모델 크기와 메모리화 성질 간의 관계를 탐구하고, 개인 정보 보호 위험을 줄이기 위한 접근법을 제시합니다.



### Hookpad Aria: A Copilot for Songwriters (https://arxiv.org/abs/2502.08122)
Comments:
          Extended abstract presented in the Late-Breaking Demo Session at ISMIR 2024 (ISMIR LBD 2024)

- **What's New**: Hookpad Aria는 음악 작곡을 돕기 위한 생성적 AI 시스템으로, 기존의 Hookpad 플랫폼에 통합되어 사용됩니다. 이 시스템은 사용자가 새로운 곡을 작곡할 때 필요로 하는 다양한 기능을 제공하여, 곡의 멜로디와 하모니를 생성하거나 기존 자료를 기반으로 내용을 추가하는 식으로 작동합니다. 2024년 3월 출시 이후, Aria는 3천명의 사용자에게 318,000개의 제안을 했으며, 이 중 74,000개가 실제 곡에 채택되었습니다.

- **Technical Details**: Hookpad Aria는 Anticipatory Music Transformer라는 대형 언어 모델에 기반하여, 다중 악기의 상징적 음악 생성을 위해 사전 훈련되었습니다. 이 모델은 기존 문맥에 따른 autoregressive 방식의 생성과 중간 영역을 채우는 기능을 동시에 지원합니다. 또한, Hookpad 환경에 맞게 MIDI 노트를 기능적 표현으로 변환하는 인코딩 체계를 설계하였으며, 비트 단위로 시간 표시를 할 수 있도록 클릭 트랙 악기를 추가했습니다.

- **Performance Highlights**: Hookpad Aria 사용자들과의 인터뷰를 통해 아이디어 창출과 창작 과정에서의 활용이 효과적이라는 결과가 도출되었습니다. 사용자는 Aria를 창작의 동료로 여기며, 창작 블록 시 새로운 출발점을 얻는 데 도움을 받다고 응답했습니다. 그러나 사용자들은 보다 세분화된 제어를 원하는 것으로 나타났으며, 장르, 감정 톤, 의도된 악기, 구조적 요소와 같은 더욱 상세한 설정 기능을 요구하고 있습니다.



### HuDEx: Integrating Hallucination Detection and Explainability for Enhancing the Reliability of LLM responses (https://arxiv.org/abs/2502.08109)
Comments:
          11 pages

- **What's New**: 최근 발표된 논문에서는 대형 언어 모델(LLM)의 신뢰성을 높이기 위한 새로운 접근법인 HuDEx 모델을 제안하고 있다. 본 모델은 LLM의 각각의 응답에서 발생할 수 있는 환각(hallucination)을 탐지하고 그에 대한 상세한 설명을 제공함으로써 모델의 신뢰성을 더욱 강화하는 기능을 갖춘다. 또한, HuDEx는 기존의 단순 탐지 방법에 그치지 않고, 환각 탐지와 해설 제공을 통합함으로써 사용자가 모델의 출력을 이해하고 오류를 줄이는 데 도움을 준다.

- **Technical Details**: HuDEx 모델은 다양한 벤치마크 데이터셋에서 환각 탐지 능력을 평가한 결과, Llama3 70B 및 GPT-4보다 뛰어난 정확도를 보여주었다. 이 모델은 소규모이지만, 기존의 고정된 기준을 넘어 능동적으로 환각을 탐지하고 사용자가 이해할 수 있는 설명을 함께 제공하는 데 중점을 두고 있다. 이를 위해 HaluEval, FactCHD, FaithDial 데이터셋을 활용하여 모델의 훈련 및 평가가 이루어졌으며, 환각의 본질을 이해하는 데 더 섬세한 접근이 가능해졌다.

- **Performance Highlights**: 제안된 HuDEx 모델은 다양한 실험 환경에서 우수한 환각 탐지 성능을 발휘하며, 특히 제로 샷 환경에서도 강력한 적응성을 보인다. 사용자 참여의 측면에서도 환각에 대한 명확한 설명 제공으로 인해 LLM의 응답 품질이 향상되며, 이는 LLM의 신뢰성과 현실 세계 적용 가능성을 증대시킨다. 이러한 연구는 언어 모델의 환각 탐지 연구 분야에 새로운 지평을 열어주는 기여를 하고 있다.



### Generative AI and Empirical Software Engineering: A Paradigm Shif (https://arxiv.org/abs/2502.08108)
- **What's New**: 이 논문은 소프트웨어 엔지니어링 연구에 있어서 생성 AI의 도입이 전통적인 연구 패러다임에 도전하는 방식을 살펴봅니다. 기존의 정량적, 정성적 및 혼합 방법적 접근 방식이 새로운 데이터 소스와 동적인 워크플로우를 다루어야 하는 상황이 전개되고 있습니다. 소프트웨어 개발에 대한 AI의 통합은 각 사회적 및 기술적 행위자간의 경계를 모호하게 하여 새로운 연구 가능성을 열어줍니다.

- **Technical Details**: 생성 AI의 도입은 AI 관련 시스템의 연구 방법과 이론에 대한 적응의 필요성을 강조합니다. 이는 AI의 비결정적 성격으로 인해 기존의 통계적 분석 및 인과 inference에서의 복잡성을 증가시키며, 정성적 방법이 사회-기술적 경계가 모호해짐에 따라 인간 행동과 AI 중재 과정을 분리하기 어렵게 만듭니다. 이러한 새로운 현상을 연구하기 위해 혼합 방법 설계가 적합하다고 주장하며, 특히 동적인 AI 출력을 고려한 반복적 연구 설계가 요구되고 있습니다.

- **Performance Highlights**: 생성 AI의 도입은 소프트웨어 엔지니어링의 전통적 워크플로우를 혼란스럽게 하면서 새로운 연구 기회를 창출하는 동시에 연구자들이 접근 방식을 적응시킬 필요성을 제기합니다. 연구자들은 인간-AI 협업의 독특한 동역학을 탐구하고, 시간이 지남에 따라 소프트웨어 개발 관행에 미치는 AI 채택의 영향을 이해하도록 노력해야 합니다. 이러한 연구는 소프트웨어 엔지니어링 연구 공동체가 새로운 도전에 대비하도록 돕고, 생성 AI가 소프트웨어 개발 과정에서 능동적 행위자로서의 역할을 할 수 있도록 방향을 제시합니다.



### PoGDiff: Product-of-Gaussians Diffusion Models for Imbalanced Text-to-Image Generation (https://arxiv.org/abs/2502.08106)
- **What's New**: 이 논문에서는 imbalanced dataset(불균형 데이터셋)에서 diffusion models(확산 모델)의 성능 저하 문제를 해결하기 위해 새로운 파인튜닝 접근 방식인 PoGDiff를 제안합니다. PoGDiff는 예측된 분포와 진짜 분포 간의 KL divergence를 최소화하기보다, Product of Gaussians(PoG)를 사용하여 원래의 진짜 타겟과 이웃 텍스트 임베딩에 조건화된 예측 분포를 결합하여 구성합니다.

- **Technical Details**: Diffusion models(DMs)는 임의의 노이즈 벡터를 조건으로 하여 이미지를 생성하는 확률 모델입니다. 이 모델은 진행적 노이즈 추가와 줄어드는 노이즈 제거 과정을 포함합니다. PoGDiff는 이러한 모델의 훈련을 개선하여 유사한 텍스트 프롬프트에 대해 동일한 이미지를 생성하도록 장려하는 방식으로 설계되었습니다.

- **Performance Highlights**: 실험 결과, PoGDiff는 실제 데이터셋에서 임Balance 문제를 효과적으로 해결하고, 생성 정확도와 품질을 개선하며 다른 최첨단 모델들을 초과하는 성과를 보여주었습니다. 새롭게 제안된 ‘Generative Recall’(gRecall) 메트릭은 모델의 생성 다양성을 평가하는 데에 큰 역할을 합니다.



### Rethinking Tokenized Graph Transformers for Node Classification (https://arxiv.org/abs/2502.08101)
Comments:
          Preprint version

- **What's New**: 이 논문에서는 기존의 Node tokenized graph Transformers (GTs)의 한계를 극복하기 위해 SwapGT라는 새로운 방법을 제안합니다. SwapGT는 token swapping이라는 새로운 작업을 도입하여 노드 간의 의미적 관련성을 활용하여 보다 다양한 token sequences를 생성합니다. 이 방식은 기존의 tokenized GTs가 인접 노드에만 초점을 맞춘다는 문제를 해결하고, 보다 정보가 풍부한 노드 표현을 학습하도록 돕습니다.

- **Technical Details**: SwapGT는 두 가지 주요 구성 요소로 이루어져 있습니다. 첫째, token swapping 작업을 도입하여 k-NN 그래프의 노드를 활용하여 여러 token sequences 간에 토큰을 교환함으로써 더 다양한 token sequences를 생성합니다. 둘째, Transformer 기반의 backbone을 사용하여 생성된 token sequences에서 노드 표현을 학습하고, center alignment loss를 통해 대표성 학습을 최적화합니다.

- **Performance Highlights**: 다양한 데이터셋에 대한 extensive empirical 결과를 통해 SwapGT가 노드 분류에 대해 기존의 방법보다 우수한 성능을 보임을 입증합니다. 특히, SwapGT는 복잡한 그래프 정보를 포착하는 데 효과적이며, 일반화 능력을 향상시키기 위해 다수의 token sequences를 활용합니다. 이는 희소한 학습 데이터에서도 효과적인 노드 표현 학습을 지원합니다.



### GCoT: Chain-of-Thought Prompt Learning for Graphs (https://arxiv.org/abs/2502.08092)
Comments:
          Under review

- **What's New**: 이번 논문에서는 그래프 모델을 위한 최초의 Chain-of-Thought (CoT) 프롬프트 학습 프레임워크인 GCoT를 제안합니다. GCoT는 그래프의 특성을 고려하여 text-free graphs에 적합한 방법으로, 단계별 추론(inference)을 통해 예측을 개선하는 방식을 탐구합니다. 이 연구는 복잡한 비선형 구조를 가진 그래프 모델의 동작을 단계적으로 안내하며, 이전의 CoT 방식과는 달리 그래프 특유의 요점을 다루고 있습니다.

- **Technical Details**: GCoT는 각 다운스트림 태스크에 대한 적응 프로세스를 프롬프트 기반 추론, '생각' 생성, 생각 기반 프롬프트 학습의 세 가지 단계로 나누어 구성합니다. 이 방식은 각 단계에서 입력 그래프와 프롬프트를 pre-trained 그래프 인코더에 입력하여 추론을 수행하고, 인코더의 은닉층을 집계하여 각 노드의 현재 상태를 반영하는 '생각'을 구축합니다. 이후 이 '생각'을 조건으로 하여 각 노드에 특화된 프롬프트를 학습하여 다음 단계로 전달합니다.

- **Performance Highlights**: 총 8개의 퍼블릭 데이터셋에 대한 포괄적인 실험을 실시하고, GCoT의 효과성과 우수성을 입증하였습니다. 특히 GCoT는 최신 기술들과 비교할 때 상대적으로 높은 성능을 보이며, 단계별 추론을 통해 더 정교한 예측을 가능하게 한다는 점에서 중요한 기여를 하고 있습니다. 이러한 결과는 텍스트가 없는 그래프에서도 Chain-of-Thought 접근 방식을 효과적으로 적용할 수 있는 가능성을 보여줍니다.



### Cognify: Supercharging Gen-AI Workflows With Hierarchical Autotuning (https://arxiv.org/abs/2502.08056)
- **What's New**: 이 연구는 자동화된 gen-AI 워크플로우 최적화의 필요성을 강조하며, AdaSeek이라는 새로운 계층적 검색 알고리즘을 제안합니다. 기존의 수동적인 워크플로우 조정 방식이 가지는 비효율성을 해결하기 위해, 이 방식은 워크플로우의 다양한 요소를 구조화하여 자동으로 튜닝할 수 있도록 돕습니다. 특히, 특정 사용자 예산에 따라 조정 가능한 계층적 아키텍처를 도입하여, 각 단계를 최적화합니다.

- **Technical Details**: AdaSeek 알고리즘은 사용자가 정의한 총 검색 예산에 따라 서로 다른 검색 계층을 구분하고, 각 계층의 복잡성에 기반하여 예산을 분배합니다. 각 계층 내에서, 이 알고리즘은 TPE(턴 메소딕스 마켓 최적화)를 활용하여 소수의 구성들을 샘플링합니다. 이러한 체계적 구조는 워크플로우의 품질을 향상시키는 데 효과적이며, 전체 효율성을 개선합니다.

- **Performance Highlights**: Cognify 프레임워크를 통해 실시된 실험 결과, 워크플로우의 생성 품질이 최대 2.8배 향상되었고, 실행 비용은 최대 10배 감소했으며, 전체 실행 지연시간은 2.7배 줄어들었습니다. 또한, 기존의 DSPy와 Trace보다 최대 2.6배 높은 생성 품질을 보이면서, 비용과 지연 시간 또한 각각 10배 및 3배 개선되었습니다.



### Break the Checkbox: Challenging Closed-Style Evaluations of Cultural Alignment in LLMs (https://arxiv.org/abs/2502.08045)
Comments:
          Preprint

- **What's New**: 이 연구에서는 대규모 언어 모델(LLMs)의 문화적 일치성을 평가하는 기존의 폐쇄형 다지선다 설문지 접근법에 대한 도전을 제기합니다. 이를 통해 LLM들이 제약이 덜한 환경에서 더 강한 문화적 일치성을 나타냄을 발견하고, 설문 응답 선택지의 순서 변화와 같은 사소한 변화가 일관성 없는 출력으로 이어질 수 있음을 보여줍니다. 이러한 결과는 문화적 측면에 대한 보다 강력하고 유연한 평가 프레임워크의 필요성을 강하게 뒷받침합니다.

- **Technical Details**: 연구에서는 World Values Survey(WVS)와 Hofstede 문화 차원 두 가지를 분석에 사용하여 방글라데시, 독일, 미국을 중심으로 세 가지 국가에서 LLM의 문화적 일치성을 평가하였습니다. WVS는 약 250개의 질문으로 구성되어 있으며, Hofstede 문화 차원은 24개의 다지선다 질문을 포함합니다. 분석은 다양한 제약 수준에서 LLM의 응답을 평가하기 위해 강제 폐쇄형, 강제 역순, 강제 자유형 및 완전히 비제약형의 네 가지 접근방법을 사용하였습니다.

- **Performance Highlights**: 연구 결과는 LLM의 응답이 질문의 제약 수준에 따라 상당히 달라진다는 점을 강조하고 있습니다. 특히 비제약형 프롬프트가 문화적 일치성을 더 잘 반영하며, 연구자들은 기계의 응답이 문화적 뉘앙스를 반영하는 데 있어 폐쇄형 질문만으로는 부족하다고 주장합니다. 이러한 발견은 LLM의 문화적 평가 방법론에 대한 재구성을 촉구하며, 특정 실제 사용 사례에서의 사용자 행동에 더욱 적합하도록 발전할 필요가 있음을 시사합니다.



### Model Selection for Off-policy Evaluation: New Algorithms and Experimental Protoco (https://arxiv.org/abs/2502.08021)
- **What's New**: 이번 연구에서는 Offline Reinforcement Learning (RL)에서 Hyperparameter tuning의 필요성에 대해 다룬다. 특히, Off-Policy Evaluation (OPE) 알고리즘을 통해 정책 평가를 수행할 때 발생하는 문제점들을 분석하고, 이러한 알고리즘 내에서 하이퍼파라미터 조정의 중요성을 강조한다. 우리는 새로운 'LSTD-Tournament' 선택기를 제안하고 이를 실험 protocol에 적용하여 기초 성과를 검증하였다.

- **Technical Details**: 연구의 주요 초점은 두 가지 설정에 따라 후보 가치 함수(model-free) 또는 다이나믹스 모델(model-based) 중에서 선택하는 것이다. 이 연구에서는 후보 함수가 Bellman 방정식을 준수하는지를 근사적으로 확인하며, 이를 통해 Double Sampling 문제를 해결한다. 또한, 견고한 이론적 보장을 제공하는 새로운 선택 알고리즘을 개발하여 모델 기반 RL에서도 효과적으로 활용할 수 있는 방안을 제시한다.

- **Performance Highlights**: Gym 환경에서 LSTD-Tournament의 성능을 평가한 결과, 후보 가치 함수의 안정적인 생성을 가능하게 하고, 미스스펙ification에 대한 더 나은 조절을 보여주었다. 새로운 실험 프로토콜은 기지 기반 환경에서의 변화를 통해 후보 Q-값을 생성하고, 최적화 없이도 Q-값을 계산할 수 있도록 지원하여 효율성을 높였다. 이러한 결과는 제안된 알고리즘이 기존 방법들보다 더 나은 성과를 제공할 가능성을 시사한다.



### Speculate, then Collaborate: Fusing Knowledge of Language Models during Decoding (https://arxiv.org/abs/2502.08020)
- **What's New**: 본 논문에서는 협동적 탐색적 디코딩(Collaborative Speculative Decoding, CoSD) 알고리즘을 소개합니다. CoSD는 여러 대형 언어 모델(LLM)의 지식을 효율적으로 융합할 수 있도록 설계되었으며, 추가적인 모델 학습 없이도 실행 가능합니다. CoSD는 초안 모델(draft model)과 보조 모델(assistant model)을 사용하여 초안 생성을 수행하고 그 결과를 개선하는 방식으로 작동합니다.

- **Technical Details**: CoSD는 초안 모델이 초기 토큰 시퀀스를 생성하고, 보조 모델이 이 토큰들을 병렬적으로 검증하여 필요한 경우 수정하도록 구성됩니다. 이 과정에서 결정 트리(decision tree)나 사전 정의된 규칙(rule-based)을 사용하여 초안 모델과 보조 모델 간의 토큰을 비교하고 대체 여부를 결정합니다. 이를 통해 CoSD는 테스트 시점에서 모델 간의 지식 융합을 실현하며 효율적인 추론(inference)을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, CoSD는 기존 방법들과 비교하여 정확도를 최대 10% 향상시키는 것으로 나타났습니다. CoSD는 다양한 도메인 및 모델 간에 높은 이식성과 효율성을 보여, 실세계 애플리케이션에서의 활용 가능성을 크게 높입니다. 또한 활용의 투명성을 제공하여, 사용자에게 모델 결정 과정을 쉽게 이해하고 최적화할 수 있는 장점을 posee합니다.



### Greed is Good: Guided Generation from a Greedy Perspectiv (https://arxiv.org/abs/2502.08006)
Comments:
          Initial preprint

- **What's New**: 이 논문에서는 트레이닝이 필요 없는 가이드 생성(Training-free guided generation) 방법론을 새로운 관점에서 탐구합니다. 특히, 신경 미분 방정식(Neural differential equation)의 해 경로(solution trajectory)를 최적화하는 탐욕적(greedy) 방법을 제안합니다. 이러한 방법은 기존의 교육 없는 가이드를 통합적으로 이해하는 데 기여합니다.

- **Technical Details**: 제안된 탐욕적 방법은 엔드 투 엔드 최적화(End-to-end optimization) 기술의 일차 차분(discretization)으로 설명됩니다. 이 방법은 최적의 경과를 이끌어내고, 연속적 상보 방정식(Continuous adjoint equations)을 통해 발견된 이상적 경량과 비교하여 우수한 가이드 결정을 내리는 것으로 나타났습니다. 이 연구는 다양한 교육 없는 가이드 전략을 통합적으로 이해할 수 있는 새로운 시각을 제공합니다.

- **Performance Highlights**: 탐욕적 가이드 전략은 결정적 직관을 제공하며, 기존의 다른 인기 있는 전이 없는 가이드를 효과적으로 대체할 수 있습니다. 이러한 접근 방식은 복잡한 생성 모델의 제어를 사용자에게 보다 쉽게 할 수 있는 방법으로 자리 잡을 것으로 기대됩니다. 결과적으로, 이 연구는 생성 모델의 성능 향상에 대한 새로운 통찰을 제시합니다.



### MetaSC: Test-Time Safety Specification Optimization for Language Models (https://arxiv.org/abs/2502.07985)
- **What's New**: 이 논문은 언어 모델의 안전성을 향상시키기 위한 동적인 안전 프레임워크를 제안합니다. 교육 단계에서 모델의 안전 정책에 대해 직접 훈련하는 기존의 방법 이외에, 추론 시 안전성을 최적화할 수 있는 새로운 접근법을 제시합니다. 저자들은 테스트 시간에 안전 지침(specifications)을 점진적으로 업데이트하여 보다 유연한 안전 추론을 가능하게 합니다.

- **Technical Details**: 제안된 MetaSC(meta-critique) 프레임워크는 테스트 시간에 안전 추론 프롬프트를 최적화하며, 이는 모델의 가중치를 변경하지 않고 자가 비판(self-critique) 프로세스를 개선합니다. 이 과정에서는 처음에 응답을 생성한 후, 그 응답의 안전성에 대한 비판을 실시하고, 그에 맞춰 응답을 수정합니다. 특히, 시스템이 상호작용을 통해 안전 지침을 지속적으로 발전시키는 메타 비판 단계를 도입하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 MetaSC 방법은 고정된 안전 프롬프트 및 정적 자가 비판 방어 방식에 비해 훨씬 높은 안전성 점수를 기록하였습니다. 적대적 공격 대응 및 다양한 안전 관련 과제를 수행하는 과정에서, 저자들은 MetaSC가 여러 언어 모델에서 효과적으로 적용될 수 있음을 보여주었습니다. 이러한 성과는 안전성이 중요한 실제 환경에서도 효과적인 적용 가능성을 시사합니다.



### CIRCUIT: A Benchmark for Circuit Interpretation and Reasoning Capabilities of LLMs (https://arxiv.org/abs/2502.07980)
- **What's New**: 본 논문은 아날로그 회로 설계에서 Large Language Models(LLMs)의 역할을 탐구하고, 회로에 대한 LLM의 추론 능력을 평가하기 위한 CIRCUIT 데이터셋을 생성했습니다. 이 데이터셋은 단순 회로 토폴로지 이해에 중점을 두고 있으며, 아날로그 회로와 관련된 510개의 질문-답변 쌍으로 구성되어 있습니다. 성능 테스트 결과, GPT-4o 모델은 최종 수치 답변에 대해 48.04%의 정확도를 기록하며, 이는 LLM들이 회로 이해에 여전히 어려움을 겪고 있음을 보여줍니다.

- **Technical Details**: CIRCUIT 데이터셋은 102개의 템플릿에서 파생된 510개의 문제로 구성되어 있으며, 각 템플릿당 5개의 수치 설정이 포함됩니다. 데이터셋은 다양한 아날로그 회로 문제를 해결하는 데 필요한 기초/중요한 회로 지식과 추론 능력을 평가하기 위한 장치로 설계되었습니다. 이 연구에서는 특히 안정성 평가를 위해 질문을 유닛 테스트 형태로 그룹화하여 LLM의 강인성을 확인했습니다.

- **Performance Highlights**: 데이터셋 분석 결과, LLM인 GPT-4o는 유닛 테스트에서 27.45%의 통과율을 나타내어 기존 LLM의 한계를 드러냅니다. 이는 아날로그 회로 설계에서 필요한 다단계 추론을 이해하는 것이 얼마나 도전적인지를 강조하며, LLM의 향후 발전 방향과 응용 가능성을 높이는 인사이트를 제공합니다. 또한, 본 연구는 아날로그 통합 회로 설계에서의 LLM 활용을 혁신적으로 발전시킬 수 있는 귀중한 기초 자료로 기능할 수 있습니다.



### From Hazard Identification to Controller Design: Proactive and LLM-Supported Safety Engineering for ML-Powered Systems (https://arxiv.org/abs/2502.07974)
Comments:
          Accepted for publication at the International Conference on AI Engineering (CAIN) 2025

- **What's New**: 이 논문은 머신러닝(ML) 기반 소프트웨어 제품의 개발에 안전 분석(hazard analysis)을 통합해야 한다고 주장합니다. 특히, LLM(대규모 언어 모델)을 활용하여 안전 분석 과정을 반자동화하고, 이를 통해 기존의 복잡한 안전 엔지니어링 작업을 경량화할 수 있는 방법을 제시합니다. 이렇게 함으로써, 경험이 부족한 개발자들도 위험을 사전에 예방하고 완화할 수 있는 방안을 마련하려고 합니다.

- **Technical Details**: 이 논문에서는 전통적인 안전 분석 기법인 Failure Mode and Effects Analysis (FMEA)와 System Theoretic Process Analysis (STPA)를 사용하여 머신러닝 모델의 다양한 위험을 체계적으로 식별할 수 있다고 설명합니다. STPA는 복잡한 시스템을 분석하는 데 적합하며, 기술적 요소와 인간적 요소를 모두 고려할 수 있는 장점이 있습니다. 이 과정에서 LLM은 안전 엔지니어링 전문 지식이 부족한 엔지니어와 데이터 과학자를 지원하여 보다 포괄적인 사고를 촉진할 수 있습니다.

- **Performance Highlights**: 논문에서는 STPA를 기반으로 한 해석적 접근 방식이 머신러닝 응용 프로그램에서 다양한 위험을 예측하고 이를 통해 안전 요구 사항을 수립할 수 있음을 보여줍니다. 특히, 이 방법은 모델과 시스템 수준에서의 위험 완화를 위한 여러 제어 구조를 설계하는 데 효과적입니다. LLM을 활용함으로써, 이러한 프로세스는 보다 접근 가능하고 관리 가능하게 되어, 실제 개발 주기에 통합될 수 있을 것으로 기대됩니다.



### Training Sparse Mixture Of Experts Text Embedding Models (https://arxiv.org/abs/2502.07972)
- **What's New**: 이 논문에서는 일반적인 Mixture of Experts (MoE) 아키텍처를 적용한 최초의 범용 텍스트 임베딩 모델인 Nomic Embed v2를 소개합니다. 본 모델은 단일언어 및 다중언어 벤치마크에서 동급의 다른 모델들을 능가하며, 두 배 크기의 모델과도 경쟁력 있는 성능을 제공합니다. 이러한 혁신은 임베딩 모델의 효율성을 향상시키며, 대규모 데이터셋을 관리하는 RAG 애플리케이션에서 특히 유용합니다.

- **Technical Details**: Nomic Embed v2는 MoE 구조를 활용하여 모델의 능력을 극대화하지만, 활성 매개변수를 줄이고 학습 효율성을 높입니다. MoE 아키텍처는 전체 파라미터를 사용하지 않고, 입력에 대해 일부 전문가(expert)만 활성화하여 계산 요구 사항을 줄입니다. 이 방법은 전통적인 모델 스케일링 접근법에 비해 많은 이점을 제공합니다.

- **Performance Highlights**: Nomic Embed v2는 대량의 데이터셋을 효율적으로 처리하며, 기존의 모형들에 비해 훨씬 적은 자원으로도 높은 성능을 낼 수 있습니다. 특히, 단일 언어 및 다중 언어의 다양한 벤치마크에서 뛰어난 결과를 보이며, 코드 및 모델을 오픈 소스하여 재현 가능성을 보장합니다. 이로 인해 연구자와 개발자들이 더 나은 텍스트 검색 시스템을 구축하는 데 도움을 줄 수 있습니다.



### ReTreever: Tree-based Coarse-to-Fine Representations for Retrieva (https://arxiv.org/abs/2502.07971)
- **What's New**: 이 논문에서는 문서 검색의 비효율성과 복잡성을 해결하기 위해, ReTreever라는 새로운 트리 기반 문서 표현 방법을 제안합니다. 이 방법은 문서를 여러 수준으로 체계적으로 조직하여 비용과 유용성을 균형 있게 조절할 수 있는 유연함을 제공합니다. 기존의 시스템에서는 고차원 임베딩을 사용하여 메모리와 계산 자원이 많이 소모되었으나, ReTreever는 이를 효율적으로 개선합니다.

- **Technical Details**: ReTreever는 이진 트리의 각 내부 노드에서 라우팅 함수(routing function)를 학습하여 쿼리 문서와 참조 문서를 유사한 트리 분기로 매핑합니다. 이는 검색 성능을 직접 최적화하는 방식으로 동작하며, 일반적인 인코딩 모델(BERT 등)을 사용해 문서 스니펫을 임베딩으로 변환합니다. 이 과정에서 LLM 호출 없이도 트리를 구축하고 내비게이션할 수 있는 기능을 제공합니다.

- **Performance Highlights**: ReTreever의 평가 결과, 이 방법은 높은 정확도를 유지하면서도 낮은 지연 시간(latency)에서 최상의 검색 정확도를 달성하였습니다. 또한, 이 구조는 문서의 의미론적 그룹화를 간접적으로 학습하여 투명성과 해석 가능성을 높이는 데 기여합니다. 결과적으로 ReTreever는 대규모 데이터 세트에서도 효율적인 문서 검색을 가능하게 하여 실용적 응용에 적합합니다.



### Generative Risk Minimization for Out-of-Distribution Generalization on Graphs (https://arxiv.org/abs/2502.07968)
Comments:
          TMLR 02/2025

- **What's New**: 이 논문은 그래프 기반 OOD(Out-of-Distribution) 일반화를 위한 Generative Risk Minimization (GRM) 프레임워크를 제안합니다. GRM은 입력 그래프에 대해 불변(subgraph invariant) 정보를 효과적으로 생성하여 기존의 이산적(extraction) 구조 환원에서 발생할 수 있는 정보 손실 문제를 해결합니다. 또한, 이 프레임워크는 변량 근사(variational approximation) 및 잠재 인과 변수(latent causal variable)를 사용하여 최적화 구조를 개선합니다.

- **Technical Details**: GRM 프레임워크는 그래프 G=(𝒱,ℰ,𝐗)로 정의되며, 여기서 𝒱는 노드 집합, ℰ는 엣지 집합, 𝐗는 특성 행렬(feature matrix)로 구성됩니다. GRM의 주요 목표는 두 가지입니다: 첫째, 연속적(edge weights, node representations) 형태로 정확한 불변 서브그래프를 생성하는 것, 둘째, 불변 서브그래프와 배경(domain) 사이의 독립성을 보장하는 것입니다. 이를 위해 GRM은 세 가지 상관된 손실(loss)을 최적화하여 정보 손실을 최소화하고 왜곡된 정보(spurious information)를 최소화하는 데 집중합니다.

- **Performance Highlights**: 다양한 실제 그래프 데이터셋을 활용한 실험을 통해, GRM은 기존의 최신 기법들과 비교할 때의 우수성을 입증하였습니다. 특히, GRM은 OOD 일반화 문제에서 불변 정보를 최대한 활용하면서 왜곡된 정보를 최소화하는 능력을 보유하고 있습니다. 이로써 OOD 그래프 데이터에 대한 일반화 성능을 크게 향상시킬 수 있는 가능성을 보여줍니다.



### Caught in the Web of Words: Do LLMs Fall for Spin in Medical Literature? (https://arxiv.org/abs/2502.07963)
Comments:
          20 pages, 10 figures, 3 tables

- **What's New**: 이번 연구는 임상 실무에 새로운 치료법을 구현하는 데 직면한 문제와 관련하여, Large Language Models (LLMs)가 연구 결과의 해석에 어떻게 영향을 받는지를 조사하였습니다. 특히, 연구자들이 긍정적인 발견을 발표하려는 압력으로 인해 연구 결과가 왜곡되거나 (spin) 과장되는 경향이 있다는 사실을 강조하고 있습니다. LLMs가 이러한 spin의 영향을 받는지 여부를 확인하기 위해 22개의 모델을 평가했습니다.

- **Technical Details**: 연구 결과, LLMs는 인간보다 spin에 더 민감하게 반응하는 경향이 있으며, 이는 의료 문헌을 탐색하고 종합하는 데 사용되는 LLMs의 활용도가 높아짐에 따라 중요한 문제로 대두되고 있습니다. 또한, LLMs는 생성하는 평이한 언어 요약에서도 spin을 암묵적으로 포함시키는 경향이 있음을 발견하였습니다. 그러나 이들은 일반적으로 spin을 인식할 수 있으며, 특정 방식으로 유도할 경우 그 영향을 완화시킬 수 있는 능력도 가지고 있습니다.

- **Performance Highlights**: 연구에서는 LLMs가 이용할 수 있는 데이터 해석 중 spin의 영향을 포착하는 능력이 있음을 보여주고 있으며, 이는 임상의사가 제공한 증거를 바라보는 방식에 중요한 시사점을 제공합니다. 결과적으로, LLM의 출력에서 spin의 영향을 줄이기 위한 방법론을 제안하며, 이는 환자 치료 결정에 긍정적인 영향을 미칠 수 있는 가능성을 열어줍니다.



### VSC-RL: Advancing Autonomous Vision-Language Agents with Variational Subgoal-Conditioned Reinforcement Learning (https://arxiv.org/abs/2502.07949)
- **What's New**: 이 논문에서는 Variational Subgoal-Conditioned RL (VSC-RL)이라는 새로운 강화 학습 방법을 소개합니다. VSC-RL은 비전-언어 (vision-language) 연속 의사결정 문제를 변분 목표 조건 강화 학습 문제로 재구성하여 학습 효율성을 향상시킵니다. 이 방법은 복잡한 목표를 자율적으로 이행 가능한 서브골(subgoal)로 분해하여 실제 세계의 복잡한 문제를 해결하는 데 큰 잠재력을 보여줍니다.

- **Technical Details**: VSC-RL은 SubGoal Evidence Lower BOund (SGC-ELBO)를 최적화하며, 이 과정에서 서브골 조건의 보상을 최대화하고 참조 정책과의 차이를 최소화하는 방식으로 진행됩니다. 이 논문은 SGC-ELBO가 원래의 최적화 목표와 동등하다는 것을 이론적으로 입증하며, 이를 통해 성과 보장을 유지하면서 학습 효율성을 개선합니다. 학습의 효율성을 증가시키는 과정에서 비전-언어 모델의 강력한 추론 및 계획 능력을 활용합니다.

- **Performance Highlights**: 다양한 벤치마크에서 실험을 실시한 결과, VSC-RL은 기존의 SOTA 비전-언어 에이전트들보다 성능과 학습 효율성 모두에서 월등한 결과를 보여줍니다. 구체적으로, 이 방법은 복잡한 모바일 장치 제어 과제에서도 뛰어난 성능을 발휘하며, 기존의 방법들과 비교할 때 매우 우수한 개선 효과를 보입니다. 임상 결과는 VSC-RL이 여러 평가 벤치마크에서 기존 SOTA와 비교해 뛰어난 성능을 도출할 수 있음을 강력하게 뒷받침합니다.



### CREDAL: Close Reading of Data Models (https://arxiv.org/abs/2502.07943)
- **What's New**: 이 논문은 데이터 모델(data model)의 중요성을 강조하며, 이를 문학 비평(literary criticism)의 기법을 통해 자세히 분석할 필요성을 제안합니다. 데이터 모델은 머신러닝 모델, 알고리즘, 통계 모델, 데이터베이스 등 모든 데이터 기반 시스템의 근본적인 요소로, 시스템의 기능성을 결정하는 역할을 합니다. 그러나 데이터 작업에서 사회정치적 측면을 간과하는 경향이 있어, 이를 체계적으로 분석할 수 있는 방법론이 부족합니다.

- **Technical Details**: 저자들은 CREDAL이라는 데이터 모델을 농밀히 읽는(close reading) 방법론을 제안합니다. CREDAL은 데이터 모델의 비판적 연구를 위한 유용성과 효과성을 평가하기 위해 정성적 평가 결과를 제시합니다. 데이터 모델이 물질적, 사회적, 정치적 조건을 반영하는 방식을 탐구함으로써, 데이터 시스템 디자인과 엔지니어링의 중요성을 강조합니다.

- **Performance Highlights**: CREDAL 방법론은 특정 사례를 통해 데이터 모델이 어떻게 권력을 유지하고 재생산하는지에 대한 분석을 가능하게 합니다. 저자들은 데이터 모델이 단순한 정보의 수집을 넘어, 특정한 사회적 맥락과 관계성을 반영하는 해석학적 도구임을 제시합니다. 이러한 분석은 사회적으로 공정하고 정의로운 데이터 미래를 설계하기 위한 중요한 기반을 제공할 수 있습니다.



### Educating a Responsible AI Workforce: Piloting a Curricular Module on AI Policy in a Graduate Machine Learning Cours (https://arxiv.org/abs/2502.07931)
Comments:
          Accepted at 2025 ASEE Annual Conference & Exposition

- **What's New**: 본 논문은 인공지능(AI) 기술이 다양한 분야에 침투함에 따라 AI 윤리 및 정책 교육 콘텐츠 통합의 필요성을 강조합니다. 특히, 전통적인 컴퓨터 과학 교육과정이 이러한 요구에 효과적으로 대응하지 못하고 있는 점을 지적하고, AI 정책 모듈을 통해 이를 보완하려는 노력을 설명합니다. 설문조사 결과, 학생들이 AI 정책 주제에 관심을 가지게 되었음을 보여줍니다.

- **Technical Details**: 이 연구에서는 2024년 대학원 수준의 기계 학습 입문 과정에서 두 차례의 강의로 구성된 ‘AI 정책 모듈’을 도입했습니다. 강의 내용은 강의 스타일 발표, 학급 토론, 실습 게임 등으로 구성되어 있어 학생 참여를 유도합니다. 학생들은 수업 전후에 실시된 설문을 통해 AI 정책에 대한 관심과 능력 변화를 평가 받았습니다.

- **Performance Highlights**: AI 정책 모듈은 기술 중심의 학생들이 AI 정책 주제에 더 많이 참여하도록 유도하며, 다양한 AI 기술의 사회적 영향에 대한 인식을 제고했습니다. 이에 따라 학생들은 AI 규제에 대한 관심도 증가했으며, 이러한 변화는 향후 AI 윤리 및 정책 교육을 위한 귀중한 기초가 될 것입니다.



### NDAI Agreements (https://arxiv.org/abs/2502.07924)
Comments:
          21 pages, 1 figure

- **What's New**: 이 논문은 혁신 경제학에서의 근본적인 도전과제를 다룹니다. 발명자는 보상을 받기 위해 새로운 아이디어의 세부 정보를 공개해야 하지만, 이와 동시에 그 정보가 유출될 위험이 있습니다. 우리는 신뢰할 수 있는 실행 환경(Trusted Execution Environments, TEE)과 AI 에이전트를 결합하여 이러한 문제를 해결할 수 있는 모델을 제시합니다.

- **Technical Details**: 모델에서는 판매자(발명자)와 구매자(투자자)가 정보 상품에 대해 협상하는데, 이 과정에서 Hold-up의 위협을 고려합니다. AI 에이전트가 보안이 갖추어진 환경 내에서 서로의 이익을 대변하고 기술적 조치를 통해 발명품을 안전하게 평가할 수 있도록 합니다. 이 방식은 발명이 노출된 이후의 불이익을 방지하는 '조건부' 공개를 가능하게 합니다.

- **Performance Highlights**: 우리의 연구 결과는 TEE와 AI 에이전트를 통해 발명자가 안전하게 아이디어를 공개할 수 있도록 하여, 효율적인 경제적 거래가 가능하다는 것을 보여줍니다. 이로 인해 정보의 공개-유출의 딜레마를 해소하고, R&D와 기술 이전을 촉진하는 정책적 함의가 있음을 시사합니다. 이러한 접근 방식은 전통적인 법적 보호 수단을 보완하거나 대체할 수 있는 가능성이 있습니다.



### TransMLA: Multi-Head Latent Attention Is All You Need (https://arxiv.org/abs/2502.07864)
Comments:
this https URL

- **What's New**: 이번 논문에서는 Multi-head Latent Attention (MLA) 접근 방식을 통해 고급 대형 언어 모델(Large Language Models, LLMs)의 통신 병목 현상을 해결하는 방법을 제안합니다. MLA는 key-value (KV) 레이어에서 저랭크 매트릭스를 사용하여 메모리 요구 사항을 감소시키고, 이를 통해 빠른 추론 속도를 구현합니다. 또한, TransMLA라는 후속 학습 방법을 도입하여 일반적으로 사용되는 GQA 기반의 사전 학습 모델을 MLA 기반 모델로 전환할 수 있도록 하였습니다.

- **Technical Details**: MLA는 전통적인 multi-head attention 방식보다 우수한 표현 능력을 제공하면서도 KV 캐시의 오버헤드를 동일하게 유지할 수 있는 이론적 근거를 제공합니다. 기존 모델에서 KV를 캐시하는 방식이 메모리와 통신에서의 병목 현상을 야기하는 문제를 강조하며, 모델을 MLA로 전환할 경우 저렴한 비용으로 성능 향상을 이루는 가능성을 제시합니다. 이 연구는 GQA 기반의 유명 모델들을 MLA 모델로 변환하고, 이를 통해 다운스트림 작업의 성능이 향상됨을 입증합니다.

- **Performance Highlights**: MLA 모델은 GQA 대비 더 나은 표현력과 성능을 제공하며, 이를 통해 기존 LLM의 가능성을 극대화할 수 있습니다. 최적의 KV 캐시 크기를 유지하면서도 빠른 추론이 가능한 구조를 통해 리소스 소비와 탄소 배출을 줄이는 효과를 기대할 수 있습니다. 이러한 접근은 DNN(Deep Neural Networks) 분야의 추후 연구 및 실제 응용에 중요한 기여를 할 것입니다.



### ADMN: A Layer-Wise Adaptive Multimodal Network for Dynamic Input Noise and Compute Resources (https://arxiv.org/abs/2502.07862)
- **What's New**: 이 논문에서는 ADMN(Adaptive Depth Multimodal Network)을 제안하며, 이는 동적 계산 자원과 입력 모달리티의 질(QoI) 변동성에 적응할 수 있는 능력을 갖춘 모델입니다. 기존의 멀티모달 네트워크는 고정된 자원 할당 방식을 사용하여 QoI에 따른 변동성을 고려하지 않아요. ADMN은 모든 모달리티에서 활성화된 레이어 수를 조정하여 이러한 문제를 해결하고, 낮은 QoI를 가진 모달리티의 리소스를 다른 모달리티에 재분배합니다.

- **Technical Details**: ADMN은 각 모달리티의 QoI에 따라 레이어를 조정할 수 있는 적응형 백본을 사용합니다. 기본적으로, ADMN은 미리 훈련된 가중치로 초기화된 대형 백본을 사용하여 레이어 드롭 기술(LayerDrop)을 통해 각 모달리티에 대해 무작위로 레이어를 드랍합니다. 또한, Gumbel-Softmax 샘플링을 활용해 멀티모달 컨트롤러가 각 입력 샘플의 QoI에 따라 최적의 레이어 할당을 학습하도록 설계되었습니다.

- **Performance Highlights**: ADMN은 멀티모달 로컬라이제이션 및 액션 인식 작업에서 검증되었으며, 기존의 최첨단 모델에 비해 최대 75%의 부동소수점 연산(FLOPS)과 60%의 지연(latency)을 줄이면서 정확도를 유지할 수 있음을 보여주었습니다. 또한, ADMN의 효율성을 입증하는 다양한 실험 결과를 제공하며, 코드와 자료를 공개했습니다.



### BalanceKV: KV Cache Compression through Discrepancy Theory (https://arxiv.org/abs/2502.07861)
- **What's New**: 이 논문에서는 BalanceKV라는 새로운 KV 캐시 압축 방법을 제안합니다. 이 방법은 Banaszczyk의 벡터 균형 이론에 기반한 기하학적 샘플링 프로세스를 활용하여 키와 값 토큰 간의 종속성을 개선하고, 메모리 복잡성을 줄이는 데 초점을 맞추고 있습니다. 또한, 이 접근법은 기존 방법에 비해 이론적으로 입증된 성능 개선을 제공하며, 실제 데이터에서도 검증되었습니다.

- **Technical Details**: BalanceKV는 KV 캐시의 효율성을 높이기 위해 기하학적 구성을 활용하여 리니어 스케일에서 서브리니어 스케일로 변환하는 알고리즘입니다. 이 알고리즘의 핵심은 자가 주의 계층 내부에서 수행되는 작업을 잘 근사하는 소수의 키와 값의 하위 집합을 파악하는 것이며, 이를 통해 메모리 사용량을 효과적으로 줄일 수 있습니다. 또한, 본 알고리즘은 기존의 샘플링 기법보다 더 높은 정확도를 제공합니다.

- **Performance Highlights**: 실험 결과, BalanceKV는 공공 데이터셋에서 여러 기존 알고리즘보다 우수한 상대적 오차를 기록했습니다. 특히, 장기 컨텍스트 이해 과제를 평가하는 LongBench 벤치마크에서 이전의 휴리스틱 캐시 압축 방법과 비교하여 더 나은 성과를 보여주었습니다. 이러한 결과는 BalanceKV의 실용성과 성능 개선 효과를 입증합니다.



### SNAP: Sequential Non-Ancestor Pruning for Targeted Causal Effect Estimation With an Unknown Graph (https://arxiv.org/abs/2502.07857)
Comments:
          Accepted at AISTATS 2025

- **What's New**: 이 논문에서는 많은 변수들에 대한 인과 발견(causal discovery) 작업의 계산 부담을 경감하는 새로운 접근법을 제시합니다. 특정 타겟 변수에 대한 인과 효과를 추정하는 데 집중하면서 모든 변수의 인과 그래프를 배우는 대신 필요한 부분만을 고려합니다. 이를 통해 인과 관계를 효율적으로 식별하는 방법을 모색하고 있습니다.

- **Technical Details**: 저자들은 타겟 변수와 그 조정 집합(adjustment sets) 간의 인과 효과를 알리기 위해 Sequential Non-Ancestor Pruning (SNAP) 프레임워크를 사용합니다. 이 방법은 정의된 비조상(definite non-ancestors) 변수를 식별하고 제거하여 인과 관계를 간소화합니다. SNAP는 기존 인과 발견 방법에 대한 전처리 단계로 사용할 수도 있고, 독립적으로 작동 가능한 완전한 인과 발견 알고리즘으로도 기능합니다.

- **Performance Highlights**: 실험 결과는 SNAP 프레임워크가 독립성 테스트(independence tests) 수와 계산 시간을 크게 줄이면서도 인과 효과 추정의 품질을 유지함을 보여줍니다. 합성 데이터와 실제 데이터를 통해 확인된 이러한 효율성 향상은 다양한 인과 분석 작업에서의 유용성을 강조합니다.



### MRS: A Fast Sampler for Mean Reverting Diffusion based on ODE and SDE Solvers (https://arxiv.org/abs/2502.07856)
Comments:
          Accepted by ICLR 2025

- **What's New**: 이 논문은 Mean Reverting (MR) Diffusion 모델의 샘플링 효율성을 개선하는 새로운 알고리즘인 MRS (MR Sampler)를 제안합니다. 기존의 빠른 무작위 샘플러가 MR Diffusion에 직접 적용되지 않는 한계를 극복하여 적은 단계로도 고품질 샘플을 생성할 수 있도록 합니다. MRS는 역방향 확률 미분 방정식 및 확률 흐름 일반 미분 방정식을 해결하여 반-정리 해를 도출하고, 이를 활용하여 빠른 샘플링을 가능하게 합니다.

- **Technical Details**: MR Sampler는 MR Diffusion에서 발생하는 역방향 확률 미분 방정식 (PF-ODE)과 SDE를 해결하여 샘플링 공식을 생성합니다. 이 방법은 분석 함수와 신경망에 의해 매개변수화된 적분을 포함하여 반-정리 해의 형태로 구성됩니다. 제안된 알고리즘은 노이즈 예측, 데이터 예측 및 속도 예측을 포함한 모든 주요 매개변화를 지원하고, 훈련이 필요 없으며 다양한 작업과 데이터 세트에 적응할 수 있습니다.

- **Performance Highlights**: MR Sampler는 10개의 서로 다른 이미지 복원 작업에서 10배에서 20배의 속도 향상을 달성하면서도 높은 샘플링 품질을 유지하는 것으로 나타났습니다. 이 알고리즘은 샘플링 절차를 가속화하여 MR Diffusion에서 실用성을 높이는 데 중요한 기여를 합니다. 고품질 샘플을 적은 수의 함수 평가(NFEs)로 생성할 수 있어, 효율적인 이미지 및 비디오 생성에 유용하게 사용될 수 있습니다.



### Vision-Language Models for Edge Networks: A Comprehensive Survey (https://arxiv.org/abs/2502.07855)
- **What's New**: 이번 연구는 비전 대형 언어 모델(Vision Large Language Models, VLMs)의 최적화와 경량화를 통해 자원 제약이 많은 엣지 환경에서의 활용 가능성을 탐구합니다. 모델 압축 기법인 pruning, quantization, knowledge distillation을 활용하여 VLMs의 성능과 효율을 향상시키는 방법이 제안되고 있습니다. VLMs는 의료, 환경 모니터링, 자율 시스템 등 다양한 분야에서 응용할 수 있는 가능성을 보여줍니다.

- **Technical Details**: VLMs는 시각적 입력과 자연어 처리를 결합하여 이미지 캡셔닝, 비주얼 질문 응답 등 다양한 태스크를 수행합니다. 그러나 최신 VLMs는 큰 메모리 요구와 높은 에너지 소비로 인해 일반적인 엣지 기기에서는 실행이 어렵습니다. 이를 해결하기 위해 pruning, quantization, 그리고 knowledge distillation 기법을 사용하여 모델 크기와 계산 비용을 줄이는 방법이 연구되고 있으며, 고유한 하드웨어 가속기도 중요한 역할을 하고 있습니다.

- **Performance Highlights**: 자원 제약이 있는 엣지 디바이스에서도 VLMs를 성능 저하 없이 활용할 수 있는 가능성이 연구되고 있습니다. 예를 들어, 의료 이미징 분야에서 VLMs를 활용하면 휴대 기기에서 즉각적인 피드백과 의사 결정을 지원할 수 있습니다. 자율주행차 및 스마트 감시와 같은 분야에서 VLMs의 실시간 처리가 필수적이며, VLMs의 경량화로 인해 이러한 기술들이 더욱 발전할 것으로 기대됩니다.



### Understanding Classifier-Free Guidance: High-Dimensional Theory and Non-Linear Generalizations (https://arxiv.org/abs/2502.07849)
- **What's New**: 최근 연구는 Classifier-Free Guidance (CFG)의 효과에 대한 우려를 제기하였으며, 낮은 차원에서는 목표 분포에 대한 overshoot와 샘플 다양성 감소를 초래할 수 있음을 보여주었습니다. 본 연구에서는 무한하고 충분히 높은 차원에서 CFG가 목표 분포를 효과적으로 재현함을 입증하며, 이는 차원의 축복(blessing-of-dimensionality) 결과를 나타냅니다. 또한 유한 차원에서의 효과를 탐구하고, overshoot 및 분산 감소를 정확히 특성화하였습니다.

- **Technical Details**: Diffusion Models은 최신 알고리즘으로 고품질의 이미지를 생성하는 데 사용되며, Orstein-Uhlenbeck Langevin dynamics를 시뮬레이션하여 데이터를 점진적으로 무작위화한 후 이 과정을 역으로 학습합니다. 이 연구에서는 CFG가 외부 분류기 없이 클래스 레이블 및 텍스트 프롬프트를 기반으로 샘플을 조건부로 생성하도록 훈련된 점을 강조하며, 높은 차원 데이터에서 CFG가 효과적으로 목표 분포를 복제함에 따라 보다 일반화된 비선형 CFG을 소개합니다. 실험을 통해 이러한 비선형 CFG가 이미지 품질 및 다양성 측면에서 유리하다는 것을 확인했습니다.

- **Performance Highlights**: 우리의 분석 결과, 비선형 CFG는 이미지 품질을 향상시키고 생성 과정의 유연성을 증가시키며, 추가적인 계산 비용 없이도 향상된 결과를 제공합니다. Gaussian mixture 및 클래스 조건 모델에 대한 수치적 시뮬레이션을 통해 이를 검증하였고, 실험은 CIFAR-10, ImageNet 데이터셋에서 높은 성능과 다양성을 달성함을 보여주었습니다. 이러한 성과는 CFG의 효용성을 확증하며, 다양한 다른 CFG 변형과의 통합을 용이하게 합니다.



### Spread them Apart: Towards Robust Watermarking of Generated Conten (https://arxiv.org/abs/2502.07845)
- **What's New**: 이 논문은 생성된 이미지에 물리적 워터마크를 삽입하는 새로운 접근 방식을 제안합니다. 이를 통해 생성된 콘텐츠를 검출하고 해당 콘텐츠를 생성한 사용자를 식별할 수 있습니다. 특히, 이 방법은 모델의 재교육이나 파인 튜닝을 필요로 하지 않으며, 향후 사용자가 생성한 콘텐츠에 대한 진위를 검증하는 데 중점을 두고 있습니다.

- **Technical Details**: 저자들은 연속적인 생성 콘텐츠에 디지털 워터마크를 삽입하는 'Spread them Apart'라는 프레임워크를 제안합니다. 이 방법은 생성 과정에서 워터마크를 삽입하여 이미지 후처리 시에도 안정성을 보장합니다. 또한, 다양한 포스트 프로세싱 공격에 대한 내성을 실험적으로 검증하여 기존 방법보다 더 우수한 성능을 발휘함을 보여줍니다.

- **Performance Highlights**: 제안된 방법은 기존의 워터마킹 기술들에 비해 포스트 프로세싱 공격에 대한 내구성이 뛰어난 것으로 나타났습니다. 특히, 이미지의 밝기 조정, 대비 조정 또는 감마 수정과 같은 공격에서 효과적으로 방어할 수 있는 능력을 갖추고 있습니다. 이러한 결과는 향후 생성된 콘텐츠의 신뢰성과 사용자 식별 가능성을 높이는 데 기여할 수 있을 것입니다.



### Column-wise Quantization of Weights and Partial Sums for Accurate and Efficient Compute-In-Memory Accelerators (https://arxiv.org/abs/2502.07842)
- **What's New**: 최근에 발표된 논문에서는 compute-in-memory (CIM) 방식의 딥 뉴럴 네트워크(DNN) 구현 시 발생하는 비교적 큰 아날로그-디지털 변환기(ADC) 오버헤드를 해결하기 위한 새로운 방법을 제안하고 있습니다. 기존의 저정밀 ADC를 사용하더라도 부분합(partial-sum) 양자화에서 발생하는 오류로 정확도가 저하되는 문제를 해결하기 위해, 가중치와 부분합 양자화의 세분화 수준을 맞춘 독창적인 접근법을 도입했습니다. 이를 통해 정확도 향상과 함께 하드웨어 효율성도 유지되는 효과를 보여줍니다.

- **Technical Details**: 이 연구에서는 가중치와 부분합 양자화의 세부 조정을 열(column)-기반으로 진행하며, 각 열에 대해 독립적인 스케일 팩터를 적용하여 메모리 셀 변동성에 대한 강인성을 확보합니다. 또한, 새로운 타일링 기법과 그룹 컨볼루션을 이용하여 CIM 지향 컨볼루션 프레임워크를 제안, 연산의 비효율성을 해소합니다. 이러한 접근 방식은 이전 방법들과 비교 시 재훈련이 필요 없는 효율적인 양자화 체계를 제공합니다.

- **Performance Highlights**: ResNet-20 모델을 CIFAR-10 및 CIFAR-100 데이터셋으로, ResNet-18 모델을 ImageNet 데이터셋으로 실험한 결과, 기존 최고의 방법에 비해 각각 0.99%, 2.69%, 1.01%의 정확도 향상을 달성했습니다. 또 다른 분석을 통해 제안한 방법이 메모리 셀 변동성에 대해 높은 강인성을 보임을 확인하였으며, CIM 기반 DNN 구현에서 정확도와 하드웨어 효율성을 동시에 개선하는 데 기여할 것으로 기대하고 있습니다.



### NanoVLMs: How small can we go and still make coherent Vision Language Models? (https://arxiv.org/abs/2502.07838)
Comments:
          11 pages, 8 figures, 3 tables

- **What's New**: 이번 연구는 Vision-Language Models (VLMs)의 최신 동향을 다루며, GPT-4V와 Llama 3.2와 같은 모델이 대규모 언어 모델(LLMs)을 이용한 멀티모달 작업에서 주목받고 있음을 시사합니다. 그러나 검토된 모델들은 독점적 제한과 높은 계산 요구 사항, 접근성 부족 등의 문제로 인해 성과가 제한적입니다. 본 연구에서는 3-4세 어린이의 학습 과정을 모방하여, 어린이의 언어 사용에 맞는 두 개의 새로운 데이터셋, ShortDesc와 LongDesc를 소개합니다.

- **Technical Details**: 제안된 데이터셋은 각기 다른 이미지-텍스트 쌍으로 구성되어 있으며, 텍스트는 어린이가 사용하는 간단한 어휘와 구문으로 제한됩니다. 연구에서는 새로운 축소 모델인 GPT-4o를 사용하여 이 데이터셋을 생성하고, 이를 통해 기존 소형 VLM보다 10배 작으면서도 아키텍처의 단순함을 유지하는 VLM 훈련이 가능함을 보여줍니다. 특히, 평가 과정에서 GPT-4o를 활용하여 학생들이 작성한 이야기의 창의성, 의미성 및 일관성을 10점 만점으로 평가하였습니다.

- **Performance Highlights**: 연구 결과, 제안된 경량의 VLM은 자원 제약 환경에서도 효과적으로 사용될 수 있는 잠재력을 가지며, 기존의 표준 벤치마크의 한계를 극복할 수 있습니다. 본 논문은 멀티모달 모델의 발전에 기여하며, 접근성이 떨어지는 환경에서도 활용될 수 있는 모델 설계를 제안합니다. 이러한 연구는 향후 다양한 응용 프로그램에서의 혁신에 기여할 것으로 기대됩니다.



### Bridging LLM-Generated Code and Requirements: Reverse Generation technique and SBC Metric for Developer Insights (https://arxiv.org/abs/2502.07835)
- **What's New**: 이 논문은 AI 생성 코드의 평가를 위한 새로운 점수 체계인 SBC 점수를 도입하고 있습니다. 이 점수는 AI가 생성한 코드에서 시스템 요구사항을 재구성하여 원래의 사양과 비교함으로써 정확성을 정량화합니다. 그 목적은 코드의 품질 및 보안을 향상시키는 것입니다.

- **Technical Details**: 이 연구에서는 AI 지원 개발에서 역생성 접근법과 SBC 점수를 평가하기 위해 90개의 다양한 소프트웨어 요구사항을 수집했습니다. SBC 점수는 다섯 가지 주요 지표로 구성되어 있으며, 이들 각 요소가 자체 점수로 구성되어 최종 점수에 기여합니다. 세 가지 주요 메트릭을 사용하여 평가: Semantic Similarity Score, BLEU Score, Completeness Score이며, 이 과정은 실제 개발주기를 반영한 데이터셋을 기반으로 진행됩니다.

- **Performance Highlights**: SBC 점수는 AI 생성 코드가 원래의 요구와 얼마나 잘 일치하는지를 정량적으로 평가합니다. 이러한 접근법은 중급 LLM에도 적용하여 코드와 자연어 작업 모두에서 효율을 강조합니다. 본 연구의 결과는 개방형 모델을 활용하여 AI 생성 코드의 평가 발전에 기여하고 있습니다.



### MEMHD: Memory-Efficient Multi-Centroid Hyperdimensional Computing for Fully-Utilized In-Memory Computing Architectures (https://arxiv.org/abs/2502.07834)
Comments:
          Accepted to appear at DATE 2025

- **What's New**: MEMHD는 메모리 효율성이 높은 다중 중심 HDC 프레임워크를 소개하여 In-Memory Computing(IMC) 아키텍처의 도전 과제를 해결한다. 이 프레임워크는 클러스터링 기반 초기화 방법과 양자화 인지 반복 학습 기법을 결합하여 분류 정확도를 유지하거나 개선하면서 메모리 요구 사항을 대폭 줄인다. 또한, MEMHD는 IMC 배열의 완전한 활용을 가능하게 하고 원샷(또는 몇 샷) 연관 검색을 지원한다.

- **Technical Details**: MEMHD는 인코딩 및 연관 메모리 모듈을 포함하여 HDC의 표준 구조를 넘어 다중 중심 모델을 개발한다. 이 모델은 초기 중심 배치를 위한 클러스터링 기반 초기화 및 양자화 효과를 반영한 반복 학습을 통해 훈련된다. 결과적으로, MEMHD는 기존의 HDC 접근 방식에서 사용하는 10,000차원 대신 1,000차원에 가까운 크기로 메모리 요구 사항을 감소시킨다.

- **Performance Highlights**: 실험 결과, MEMHD는 기존의 이진 HDC 모델에 비해 최대 13.69% 더 높은 정확도를 달성하며, 같은 메모리 사용량에서 더 나은 성능을 발휘한다. 또한, 계산 사이클을 최대 80배 줄이고, IMC 매핑 방식에 비해 배열 사용량을 71배 줄이는 성과를 보였다. 이러한 효율성 개선은 에너지와 계산 사이클 효율성 향상에도 기여한다.



### SHARP: Accelerating Language Model Inference by SHaring Adjacent layers with Recovery Parameters (https://arxiv.org/abs/2502.07832)
Comments:
          24 pages

- **What's New**: SHARP(Sharing Adjacent Layers with Recovery Parameters)는 기존의 LLM(대형 언어 모델)의 인퍼런스를 가속화하기 위해 인접 계층 간의 파라미터를 공유하는 새로운 접근 방식을 제안합니다. 이 방법은 메모리 부담을 줄이면서 성능을 유지하기 위해 저계수 회복 파라미터를 도입합니다. SHARP는 연속적인 계층의 출력이 유사하다는 사실에 기반하여 설계되었으며, 단일 계층 워밍업(Single Layer Warmup, SLW) 및 감독하에 미세 조정(Supervised Fine-Tuning, SFT)의 두 단계로 구성됩니다.

- **Technical Details**: SHARP는 현재 계층의 출력을 원본 대체 계층의 출력과 비교하며, L_2 손실을 최소화하여 SFT 단계에 적합한 초기값을 제공합니다. SLW는 여러 계층을 한 개의 계층으로 예측하는 데 중요한 역할을 하며, 3/4의 원래 MLP 계층을 생략해도 모델의 perplexity를 효과적으로 복구할 수 있도록 합니다. SHARP는 MLP 계층 파라미터를 38%에서 65%까지 줄이면서도 다양한 인디스트리뷰션(task 분류) 작업에서 모델 성능을 회복합니다.

- **Performance Highlights**: SHARP는 Llama2-7b 모델에 비해 모델 저장 용량을 42.8%, 총 인퍼런스 시간을 42.2% 절감하는 성과를 보였습니다. 50,000개의 미세 조정 데이터만 사용하여 다양한 작업에서 모델 perplexity를 복구하였으며, 특히 메모리 관련 다운스트림 작업에서 더 나은 성능을 나타냈습니다. 이 연구는 모바일 기기에서도 효율적인 LLM 배포를 위한 해결책으로 SHARP를 제시합니다.



### Captured by Captions: On Memorization and its Mitigation in CLIP Models (https://arxiv.org/abs/2502.07830)
Comments:
          Accepted at ICLR 2025

- **What's New**: 이 논문에서는 CLIP 모델의 메모리화(memorization) 과정을 정량화하기 위해 CLIPMem이라는 새로운 개념을 제안합니다. 기존의 메모리화 정의가 CLIP의 다중 모달 특성을 반영하지 못하는 한계를 해결하고자 메모리화를 설명하기 위해 새로운 기준을 설정합니다. '잘못 캡션된(mis-captioned)' 데이터가 가장 높은 메모리화 수준을 보인다는 중요한 결과도 발견했습니다.

- **Technical Details**: CLIP은 이미지와 텍스트의 쌍을 공유된 잠재 공간에 매핑하여 작업을 수행하는 다중 모달 인코더 구조를 기반으로 합니다. 이 모델은 대조 손실 함수(contrastive loss)를 통해 정답 쌍 사이의 유사성을 극대화하고 잘못된 쌍의 유사성을 최소화합니다. CLIPMem 메트릭은 주어진 이미지-텍스트 쌍의 정렬을 비교하여 메모리화를 측정하는 새로운 방법입니다.

- **Performance Highlights**: 연구를 통해 텍스트 인코더가 이미지 인코더보다 메모리화에 더 많이 기여하며, 이는 모델 성능에 중요한 영향을 미칩니다. 리서치는 메모리화 감소 전략이 CLIP의 일반화 능력을 향상시킬 수 있음을 보여줍니다. 이러한 결과는 전통적인 학습 패러다임에서는 메모리화 감소가 일반적으로 성능 저하로 이어지던 것과는 대조적입니다.



### Some things to know about achieving artificial general intelligenc (https://arxiv.org/abs/2502.07828)
- **What's New**: 현재 및 예측 가능한 GenAI 모델은 인간의 영향으로 인해 인공 일반 지능(AGI)을 달성할 수 없다는 새로운 인식을 제안합니다. 이러한 모델은 문제를 구조화하고, 아키텍처를 설계하며, 훈련 데이터를 제공하는 데 인간의 입력에 크게 의존하고 있습니다. 이로 인해 모델은 단순한 언어 패턴 학습 문제로 모든 문제를 환원하게 되었으며, 이는 AGI를 달성하는 데 필요한 자율성을 제한합니다.

- **Technical Details**: 현재 모델들은 주로 사람들의 지식에 의존하여 문제를 해결하므로, 실제로는 간단한 계산(예: gradient descent)만 수행하고 있습니다. 다양한 유형의 문제를 인식해야 하며, 일부 문제는 기존의 계산 방법으로는 해결할 수 없음을 강조합니다. 이는 '통찰 문제(insight problems)'와 같은 문제를 포함하며, 이러한 문제들은 현재 접근 방식으로는 처리할 수 없습니다.

- **Performance Highlights**: 현재 모델들의 평가 방법인 benchmarks와 tests는 문제 해결의 일반성을 파악하는 데 충분치 않습니다. 성공의 관찰만으로 문제를 해결한 방법을 유추하는 것은 논리적 오류(affirming the consequent)입니다. 특정 테스트를 통과하는 것이 특정 테스트 방법(테스트 전용 방법) 또는 일반적인 방법(테스트 일반 방법)의 결과일 수 있어, 해결 방법을 확인할 수 있는 방법이 부족합니다.



### Implicit Language Models are RNNs: Balancing Parallelization and Expressivity (https://arxiv.org/abs/2502.07827)
- **What's New**: 이 논문에서는 언어 모델링에서 상태 공간 모델(State-Space Models, SSMs)과 트랜스포머(Transformers)의 한계를 극복하기 위해 암묵적 상태 공간 모델(Implicit SSMs)을 제안합니다. 이 모델은 비선형 상태 전이(Non-linear state-transitions)를 통해 표현력을 확보하면서도 동시에 훈련 중 병렬 처리(Parallelization)를 유지할 수 있게 설계되었습니다. 또한, 이 암묵적 모델은 자연어 추론(Natural Language Reasoning) 작업과 대규모 언어 모델 사전 학습(Pretraining)에서 뛰어난 성능을 보여줍니다.

- **Technical Details**: 암묵적 SSM은 문제의 난이도에 따라 계산 부하를 자연스럽게 조절하며, 모든 토큰이 해결 가능한 경우에는 병렬성(Parallelizability)을 유지하고, 해결할 수 없는 토큰이 있는 경우 RNN처럼 동작합니다. 이러한 구조는 딥 균형 모델(Deep Equilibrium Models)로, 임의의 별도 경로를 통과하지 않고 고정점(Fixed Point)에서만 역전파(Backpropagation)가 이루어져 메모리 사용량을 줄일 수 있습니다. 또한, 암묵적 SSM은 특정 상황에서 트랜스포머 및 SSM보다 월등한 상태 추적(State-Tracking) 기능을 발휘합니다.

- **Performance Highlights**: 암묵적 SSM 모델은 표준 벤치마크에서 명시적 모델(Explicit Models)보다 뛰어난 성능을 보이는데, 특히 S5 문제를 성공적으로 해결하며, 기존의 트랜스포머 및 SSM이 해결하지 못하는 요소를 풀어냅니다. 이를 통해 1.3B 파라미터로 207B 토큰의 데이터셋에 대한 훈련 성과를 달성하였으며, 이는 현재까지 이루어진 가장 큰 암묵적 모델로 기록됩니다. 논문에서는 또한 훈련 커리큘럼을 통해 효율적인 스케일링을 달성하였음을 강조하고 있습니다.



### Pre-Trained Video Generative Models as World Simulators (https://arxiv.org/abs/2502.07825)
Comments:
          20 pages

- **What's New**: 본 연구에서는 비디오 생성 모델을 동적인 세계 시뮬레이터로 변환하기 위한 새로운 접근 방식인 Dynamic World Simulation (DWS)을 제안합니다. DWS는 사전 훈련된 비디오 생성 모델을 유연하게 변환할 수 있으며, 특정한 행동 경로를 기반으로 시각적 변화를 생성할 수 있는 구조를 가지고 있습니다. 이는 기존 모델에 통합할 수 있는 경량화된 행동 조건 모듈을 도입하여 동적 변화에 대한 정밀한 조정을 가능하게 합니다.

- **Technical Details**: DWS는 프레임 수준의 액션 인식을 개선하는 경량의 액션 조건 모듈을 도입하여, 프레임 간의 동적 전환 모델링을 중시하는 방안을 제공합니다. 이 모듈은 두 개의 선형 층으로 구성되어 있으며, 기존 모델 아키텍처에 손쉽게 통합될 수 있도록 설계되었습니다. 또한, DWS는 정적인 시각적 세부 요소에서 동작 유도 동적 변화로 모델의 주의를 전환하는 모션 강화 손실(motion-reinforced loss)을 통해 훈련 중 동적 모델링의 우선 순위를 명확하게 설정합니다.

- **Performance Highlights**: DWS는 게임 및 로봇 분야 전반에 걸쳐 행동 제어가 가능한 동적 비디오 생성에서 큰 개선을 이루었습니다. 본 연구의 실험 결과는 DWS로 학습한 세계 시뮬레이터가 여러 정책과 정확한 동적 예측을 유지하면서도 다양한 도메인에서 상호작용할 수 있음을 보여줍니다. 또한, 우선 순위가 매겨진 상상(prioritized imagination)을 도입하여 샘플 효율성을 개선하고, 최신 MBRL 방법들과 비교해 경쟁력 있는 성능을 나타냅니다.



### Runtime Tunable Tsetlin Machines for Edge Inference on eFPGAs (https://arxiv.org/abs/2502.07823)
Comments:
          Accepted as a full paper by the 2025 EDGE AI FOUNDATION Austin

- **What's New**: 이 논문은 새로운 eFPGA (embedded Field-Programmable Gate Arrays) 가속기를 제안하여 기존 FPGA보다 더 적은 전력으로 엣지 머신러닝 (ML) 응용프로그램을 지원할 수 있도록 합니다. 주요 특징은 자원 사용을 최소화하고 모델 크기와 아키텍처를 런타임에 재조정할 수 있는 유연성을 제공하는 점입니다. 이는 Tsetlin Machine (TM) 알고리즘의 비트 압축 추론 아키텍처를 통해 가능해집니다.

- **Technical Details**: 제안된 eFPGA 가속기는 비트 연산(AND, OR, NOT)만을 사용하여 모델을 구현하므로 곱셈 연산이 필요하지 않습니다. TM 모델의 압축 덕분에 전체 모델이 eFPGA의 온칩 블록 RAM에 적재될 수 있으며, 이론적으로 2.5배 적은 Look-up-Tables (LUTs)와 3.38배 적은 레지스터를 사용합니다. 또한, 기존 저전력 마이크로컨트롤러와 비교 시 에너지 요구사항을 129배 줄일 수 있습니다.

- **Performance Highlights**: 이 연구는 런타임 모델 조정을 위한 전략을 제안하며, 기존의 자원 절약형 설계와 비교해 성능을 벗어나지 않으면서도 유연성을 갖춘 결과를 보여줍니다. TM을 기반으로 한 구조적 단순성과 희소성 덕분에 작은 엣지 장치에서 효과적인 ML 모델을 구현할 수 있습니다. 이는 적은 리소스를 사용하면서도 높은 응용성을 지원하는 새로운 가능성을 제공합니다.



### PDM-SSD: Single-Stage Three-Dimensional Object Detector With Point Dilation (https://arxiv.org/abs/2502.07822)
- **What's New**: 이 논문에서는 기존의 점 기반 감지기들이 가진 한계를 극복하기 위한 새로운 방법인 Point Dilation Mechanism (PDM-SSD)을 제안합니다. PDM은 점을 특정 크기의 그리드로 확장하고, 비어 있는 공간에 대해 기능을 채우는 두 가지 주요 단계를 통해 특징 공간을 확장합니다. 이 방식은 특히 불완전하거나 희박한 객체 인식에서 효과적이며, 학습 속도와 정확도 간의 균형을 잘 맞춥니다.

- **Technical Details**: PDM-SSD는 PointNet 스타일의 3D 백본을 사용하여 효율적으로 특징을 인코딩합니다. 또한, 구형 조화 계수와 가우시안 밀도 함수를 활용하여 비어 있는 그리드를 기능으로 채우고, 여러 개의 확장 중심을 연관시켜 희박한 그리드 특징을 얻는 과정이 포함됩니다. 최종적으로는 혼합 감지 헤드를 설계하여 객체의 확률을 보정하고, 장면 열지도를 예측하여 탐지 정확도를 높입니다.

- **Performance Highlights**: PDM-SSD는 KITTI 데이터세트에서 다중 클래스 감지에서 최첨단 성능을 달성했으며, 68 프레임의 추론 속도를 기록합니다. 또한, 다양한 객체 수준의 사례들을 통해 희박하고 불완전한 객체 검출 측면에서 PDM-SSD의 장점을 입증했습니다. 이 방법은 PDM을 보조 네트워크로 사용하여 샘플링 포인트와 객체 중심 간의 연결을 강화하여 성능을 향상시킬 수 있습니다.



### Amnesia as a Catalyst for Enhancing Black Box Pixel Attacks in Image Classification and Object Detection (https://arxiv.org/abs/2502.07821)
Comments:
          Accepted as a poster at NeurIPS 2024

- **What's New**: 이 논문은 새로운 픽셀 기반 블랙박스 공격 기법인 Remember and Forget Pixel Attack using Reinforcement Learning (RFPAR)을 제안합니다. 기존 연구에서 다루지 않았던 픽셀 공격에 중점을 두고, 이미지 분류뿐만 아니라 물체 탐지에서도 효과를 입증하였습니다. 이 방법의 주요 특징은 랜덤성을 줄이고 패치 의존성을 피하기 위해 강화학습(RL) 알고리즘을 활용한다는 점입니다.

- **Technical Details**: RFPAR는 두 가지 프로세스인 Remember와 Forget으로 구성되어 있습니다. Remember 과정에서는 RL 에이전트가 클린 이미지를 입력으로 받고, 최적화 과정에서 가장 높은 보상과 대응하는 변조된 이미지를 메모리에 저장합니다. Forget 과정에서는 보상이 더 이상 변화하지 않을 경우 이전 정보를 잊고 다시 Remember 프로세스를 시작하게 됩니다.

- **Performance Highlights**: RFPAR는 ImageNet-1K 데이터셋에서 최첨단 쿼리 기반 픽셀 공격을 초월하여 평균 공격 성공률을 12.1% 향상시키고, 쿼리 수를 26.0% 줄였습니다. 또한 물체 탐지 분야에서도 YOLO 및 DDQ와 함께 MSCOCO 데이터셋을 사용해 mAP 감소를 비교했으며, 적은 수의 쿼리로도 유사한 성능을 달성하였습니다.



### Low-Rank Compression for IMC Arrays (https://arxiv.org/abs/2502.07820)
Comments:
          Accepted to appear at DATE'25 (Lyon, France)

- **What's New**: 본 연구는 in-memory computing (IMC) 아키텍처의 문맥에서 저차원 모델 압축의 문제를 해결하고자 합니다. 전통적인 pruning 기법은 모델 크기를 줄이는 데 효과적이지만, 복잡한 데이터 흐름을 관리하기 위해 추가 외부 회로가 필요하며, 이로 인해 면적과 에너지 오버헤드가 증가하는 단점이 있습니다. 이에 우리는 저차원 압축 기법을 활용하여 데이터 흐름을 간소화하고 IMC 아키텍처에 원활하게 통합하는 방법을 제안합니다.

- **Technical Details**: 저차원 압축은 IMC 어레이의 서브 옵티멀 활용과 정확성 저하라는 문제를 안고 있습니다. 본 연구에서는 shift and duplicate kernel (SDK) 매핑 기법과 group low-rank convolution 기법을 도입하여 이러한 문제를 해결하고자 합니다. SDK 매핑 기법은 유휴 IMC 열을 활용하여 병렬 처리를 가능하게 하며, 그룹 저차원 합성곱은 분해된 행렬의 정보 불균형을 완화하는 데 도움을 줍니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존 pruning 기법에 비해 최대 2.5배의 속도 향상과 20.9%의 정확도 향상을 달성하였습니다. 이는 저차원 압축 기법이 IMC 아키텍처와 잘 결합되어 기존 방법의 단점을 극복할 수 있음을 보여줍니다. 이러한 성과는 IMC 아키텍처의 효율성을 크게 향상시키며 이러한 연구 분야의 중요한 진전을 나타냅니다.



### Decoding Complexity: Intelligent Pattern Exploration with CHPDA (Context Aware Hybrid Pattern Detection Algorithm) (https://arxiv.org/abs/2502.07815)
- **What's New**: 이 연구는 민감한 데이터를 탐지하기 위한 알고리즘의 성능을 평가하고 있으며, regex 기반 패턴 매칭과 정확 일치 검색 방법을 최적화하여 속도, 정확성 및 확장성을 개선하는 데 초점을 두고 있습니다. Google RE2는 속도와 메모리 효율성 면에서 정교한 성능을 보이며, Aho-Corasick 알고리즘을 통해 대규모 데이터 세트에서의 정확한 일치 성능을 통합합니다. 후속 연구에서는 데이터 보안과 프라이버시 관리 통합을 위한 필요성이 강조되고 있습니다.

- **Technical Details**: 최적화된 정확도 및 성능을 위해 Context-Aware Hybrid Pattern Detection Algorithm (CHPDA)을 도입하였습니다. 이 알고리즘은 정규 표현식과 딥러닝 기반의 Named Entity Recognition (NER)을 통합하여, 다양한 데이터를 효과적으로 처리하고 False Positive를 최소화합니다. 특히, 하나의 패스에서 여러 패턴을 처리할 수 있는 최적화된 알고리즘을 사용하여 대량의 데이터 세트를 처리하면서도 높은 정확도를 유지합니다.

- **Performance Highlights**: 벤치마킹 결과, Google RE2는 속도 (10-15 ms/MB)와 메모리 효율성 (8-16 MB) 및 정확성 (99.5%) 면에서 최상의 균형을 제공합니다. AI + Regex 하이브리드 접근 방식은 F1 스코어 (91.6%)로 이전의 방법들보다 더 높은 성능을 기록했으며, 다양한 시스템에서 효율적인 CPU 및 메모리 사용을 유지합니다. 그러나 다국어 지원의 제한과 정기적인 패턴 업데이트의 필요성 같은 과제가 남아 있습니다.



### Satellite Observations Guided Diffusion Model for Accurate Meteorological States at Arbitrary Resolution (https://arxiv.org/abs/2502.07814)
- **What's New**: 이번 연구에서는 기상 예측의 정확성을 향상시키기 위해 위성 관측을 기반으로 한 확산 모델인 Satellite-observations Guided Diffusion Model (SGD)을 제안합니다. SGD는 ERA5 재분석 데이터와 위성 관측 데이터를 조건으로 활용해 저해상도 기상 상태를 업스케일링하는 혁신적 접근 방식을 제공합니다. 기존 방법들이 위성 데이터와의 상관관계를 간과했으나, SGD는 주의 메커니즘을 통해 이를 보완합니다.

- **Technical Details**: SGD는 ERA5와 GridSat 위성 관측 데이터를 융합하여 훈련되며, 이 과정에서 교차 주의(attention) 모듈을 활용하여 고해상도 기상 데이터를 효과적으로 생성합니다. 특히, SGD는 최적화 가능한 컨볼루션 커널을 사용해 저해상도 지도를 기반으로 고해상도 ERA5 지도를 생성하며, 저해상도 지도 및 기상 관측소 데이터를 가이드로 포함하여 샘플링합니다. 이러한 방식은 고해상도 기상 상태의 세밀한 디테일을 보장합니다.

- **Performance Highlights**: 실험 결과, SGD는 6.25km 해상도로 정확한 기상 상태를 생성하며, 기존의 보간(interpolation) 기반 방법이나 확산 모델에 비해 뛰어난 성능을 보였습니다. 또한, ablation study를 통해 GridSat 맵을 조건으로 사용하는 것이 SGD의 성능을 현저히 향상시킨다는 점이 입증되었습니다. 이러한 혁신적인 접근 방식은 특히 기상 데이터 처리 및 예측의 정확성을 높이는 데 기여할 것으로 기대됩니다.



### CryptoX : Compositional Reasoning Evaluation of Large Language Models (https://arxiv.org/abs/2502.07813)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)의 조합적 추론(capacity for compositional reasoning) 능력을 평가하기 위해 CryptoX라는 새로운 평가 프레임워크를 소개합니다. CryptoX는 기존의 벤치마크와 암호화 기술을 결합하여, LLM의 조합적 추론 능력을 정량적으로 측정할 수 있도록 합니다. 추가로, CryptoBench를 구축하여 다양한 벤치마크를 통합하여 LLM의 조합적 추론 능력을 체계적으로 평가할 수 있는 시스템을 제공합니다. 이로써 LLM의 조합적 추론 메커니즘을 심층적으로 분석하는 기회를 제공합니다.

- **Technical Details**: 조합적 추론(Compositional Reasoning, CR)의 새롭게 제안된 평가 방법인 CryptoX는 기존 벤치마크에 암호 기반 변환을 통합하여 LLM의 조합적 추론 능력을 평가합니다. 이 방법에서는 프롬프트의 특정 단어를 새로운 암호 문자로 인코딩하며 인스트럭션 안에 인코딩 규칙을 명시적으로 포함합니다. 또한, CryptoBench라는 벤치마크 세트를 구축하여 이러한 원칙을 체계적으로 적용하여 평가를 수행합니다.

- **Performance Highlights**: 실험 결과, 공개 소스 LLM과 폐쇄 소스 LLM 사이에는 크고 중요한 능력 차이가 있는 것으로 나타났습니다. 대부분의 기존 LLM은 조합적 추론 능력에서 약한 성능을 보였으며, CryptoBench는 다양한 LLM 간의 능력 차이를 측정할 수 있음을 입증했습니다. 이는 특정 요인, 즉 모델 크기와 아키텍처 등이 CR 능력에 영향을 미친다는 것을 보여주고 있습니다.



### CP-Guard+: A New Paradigm for Malicious Agent Detection and Defense in Collaborative Perception (https://arxiv.org/abs/2502.07807)
- **What's New**: 이번 연구에서는 collaborative perception (CP) 시스템의 취약점을 해결하기 위한 새로운 접근법을 제안합니다. 특히 malicious agent를 feature level에서 직접 탐지하는 방법을 도입하여, 기존의 여러 차례의 검증 과정을 줄이고 계산 오버헤드를 크게 감소시킵니다. 또한, CP-GuardBench라는 새로운 데이터셋을 생성하여 악의적인 에이전트 탐지를 위한 기준을 제공합니다.

- **Technical Details**: 이 논문에서는 두 가지 주요 구성 요소를 도입합니다. 첫째, feature encoder, aggregator 및 decoder를 이용한 CP 시스템의 파이프라인을 설명합니다. 둘째, CP-Guard+라는 새로운 방법론을 통해 benign 및 악의적 feature 간의 마진을 증가시키고, Dual-Centered Contrastive Loss (DCCLoss)를 통해 인식 불가능성을 해결합니다.

- **Performance Highlights**: 연구 결과, CP-Guard+는 CP-GuardBench 및 V2X-Sim에서 수행된 실험에서 우수한 성능을 보였습니다. 이 시스템은 feature level에서 malicious agent를 효과적으로 탐지하며, 최종 인식 결과를 검증하는 과정을 생략하여 계산 효율성을 높입니다. 이러한 점에서 CP 시스템의 보안을 크게 향상시킬 수 있는 가능성을 보여줍니다.



### Quantum Powered Credit Risk Assessment: A Novel Approach using hybrid Quantum-Classical Deep Neural Network for Row-Type Dependent Predictive Analysis (https://arxiv.org/abs/2502.07806)
- **What's New**: 이 논문은 Quantum Deep Learning (QDL) 기법을 금융 리스크 분석에 통합하여 혁신적인 접근 방식을 제시합니다. 은행 부문에서의 신용 리스크 평가를 위해 Row-Type Dependent Predictive Analysis (RTDPA)를 활용하여 대출 카테고리에 맞춘 예측 모델을 만드는 프레임워크를 소개합니다. 이러한 접근법은 전통적인 방법과 양자 컴퓨팅의 장점을 결합하여 신용 리스크 평가의 정확성과 효율성을 높이는 것을 목표로 합니다.

- **Technical Details**: 이 연구는 양자 컴퓨팅과 고전적 깊은 신경망을 통합하여 신용 리스크 평가를 보다 유연하고 적응 가능한 방법으로 발전시키고자 합니다. RTDPA는 서로 다른 대출 유형의 특성을 인식하여 개별적으로 분석하여 예측 모델을 조정하는 방법을 제안합니다. 이 연구는 또한 기반 최적화 기술인 Beetle Antennae Search (BAS)와 같은 대안 최적화 기법에 대해서도 논의하며, 향후 연구에서 이 기법의 잠재력을 강조합니다.

- **Performance Highlights**: 연구 결과는 양자 기법이 전통적인 금융 분석을 보완하는 방법에 대한 통찰력을 제공합니다. QDL은 신용 리스크 평가에서 더 나은 예측력을 달성할 수 있는 가능성을 제시하며, 다양한 대출 유형에 맞춘 더 정밀한 리스크 평가를 가능하게 합니다. 이 프레임워크는 점차 역동적인 금융 환경에서 신용 리스크 분석을 위한 보다 효과적인 접근 방식을 제공하는 것을 목표로 합니다.



### Regulatory Science Innovation for Generative AI and Large Language Models in Health and Medicine: A Global Call for Action (https://arxiv.org/abs/2502.07794)
- **What's New**: 이 논문은 의료 분야에서 생성적 인공지능(Generative AI, GenAI)과 대규모 언어 모델(Large Language Models, LLMs)의 통합이 만들어내는 새로운 기회와 도전을 다룹니다. GenAI와 LLM은 임상 업무 자동화부터 개인화된 진단에 이르기까지 폭넓은 응용성을 제공합니다. 그러나 эти 기술의 비결정적 출력과 복잡한 통합성은 기존 의료 기기 규제 프레임워크에 도전장을 내밉니다.

- **Technical Details**: 논문에서 저자는 전체 제품 생애 주기(Total Product Life Cycle, TPLC) 접근 방식의 한계를 설명하며 의료 기기 규제에서 GenAI와 LLM의 응용을 제안합니다. 새로운 정책을 시험하고 수정할 수 있는 적응형 정책(adaptive policies)과 규제 샌드박스(regulatory sandboxes) 등 혁신적인 접근 방식 개발이 필요합니다. 국제 harmonization은 의료 기기 규제 포럼을 통해 이루어져야 하며, 이는 세계적인 건강 불균형을 좁히기 위한 필수적인 기반이 됩니다.

- **Performance Highlights**: 저자는 다양한 분야의 전문 지식을 활용하고 반복적이며 데이터 기반의 접근 방식을 우선시함으로써 다양한 인구의 요구에 초점을 맞춰야 한다고 강조합니다. 이러한 글로벌 규제 과학 연구는 LLM 혁신의 책임 있고 공정한 발전을 가능하게 할 것입니다. 논문의 전반적인 결론은 건강 불균형의 위험을 관리하기 위한 국제적 협력의 중요성을 강조합니다.



### Can Generative AI be Egalitarian? (https://arxiv.org/abs/2502.07790)
Comments:
          14 pages, 5 figures

- **What's New**: 최근의 '재단' 생성 AI 모델은 온라인 소스에서 가치를 광범위하게 추출하는 패턴에 기반하고 있습니다. 이 모델들은 종종 상응하는 보상 없이 이루어지며, 이에 따른 윤리적 및 사회적 문제가 대두되고 있습니다. 그러나 사용자들이 자발적으로 제공한 콘텐츠에 기반한 모델 개발이라는 새로운 대안이 등장하고 있습니다.

- **Technical Details**: 이 연구에서는 위키피디아의 성공 모델에서 영감을 받아 '평등주의'적 접근 방식의 가능성을 탐구합니다. 이러한 접근방식은 사용자의 요구에 더 잘 부응하고, 훈련 데이터의 다양성을 증진시키며, 결국 사회적 가치와 더 일치하는 모델로 이어질 수 있다고 주장합니다.

- **Performance Highlights**: 이러한 접근 방식은 윤리적으로 타당할 뿐만 아니라, 향후 재단 모델의 설계 및 개발 과정에서도 긍정적인 영향을 미칠 것으로 예상됩니다. 그러나 자원봉사자가 기여한 콘텐츠의 품질 관리, 확장성 및 내재된 편향과 같은 도전 과제도 함께 고려해야 합니다.



### Do AI assistants help students write formal specifications? A study with ChatGPT and the B-Method (https://arxiv.org/abs/2502.07789)
- **What's New**: 이 논문에서는 OpenAI의 ChatGPT가 학부 학생들에게 Formal Methods (FM), 특히 B-method를 사용하는 정형 명세(formal specification)를 가르치는 데 어떻게 기여하는지를 조사합니다. 기존 연구들은 AI가 코딩 작업에서 효과적이라는 것을 입증했지만, 정형 명세에 대한 연구는 부족했습니다. 이 연구는 ChatGPT가 B-specifications 작성에 유리한지, 그리고 학생들의 신뢰 수준이 결과에 어떤 영향을 미치는지를 분석합니다.

- **Technical Details**: B-Method는 상태 기반 형식(formal) 기법으로, 시스템의 가능한 상태를 명세하기 위해 추상 기계(abstract machine)의 개념을 사용합니다. 이 방법은 기계의 연산으로 정의되는 상태 전환을 다루며, 구체적인 구현으로의 점진적 변환(refinement)을 포함합니다. 연구는 Participants의 B-specification 정확도를 사전 및 사후 테스트(pretest-posttest) 방식으로 평가하며, 이들 간의 차이를 분석하여 AI 도우미의 효과를 검증합니다.

- **Performance Highlights**: ChatGPT를 사용하여 B-specifications를 작성하는 것이 학생들의 성과에 긍정적인 영향을 미치지 않는 것으로 나타났습니다. 특히 학생들이 느끼는 신뢰가 낮을수록 B-specifications의 정확도가 더 높아지는 상관관계가 발견되었습니다. 또한, AI와 상호작용할 때의 행동 패턴이 B-specifications의 정확성 향상에 영향을 미칠 수 있음을 제시합니다.



### Counterexample Guided Program Repair Using Zero-Shot Learning and MaxSAT-based Fault Localization (https://arxiv.org/abs/2502.07786)
Comments:
          Accepted at AAAI 2025. 11 pages, 4 listings, 2 figures and 5 tables

- **What's New**: 본 논문에서는 Automated Program Repair (APR) 문제를 해결하는 새로운 접근법을 제시합니다. 이 접근법은 Large Language Models (LLMs)와 Formal Methods (FM)의 장점을 결합하여 향상된 프로그래밍 과제(IPA) 수정을 목표로 하고 있습니다. MaxSAT를 기반으로 한 결함 위치 파악을 통해 버그가 없는 프로그램 스케치를 LLM에 제공하는 반면, 반례 유도 귀납 합성을 활용하여 프로그램을 반복적으로 개선합니다.

- **Technical Details**: 제안된 방법은 MaxSAT 기반 결함 위치 파악(MaxSAT-based fault localization) 기술을 활용하여 프로그램의 버그가 있는 부분을 식별한 다음, 해당 버그 없는 프로그램 스케치를 LLM에 제시하는 방식으로 작동합니다. 이 프로세스는 Counterexample Guided Inductive Synthesis (CEGIS) 루프를 통해 순차적으로 프로그램을 다듬는 방식으로 진행됩니다. LLM은 상실된 부분을 합성하고 그 결과를 테스트 스위트와 비교하여 검증합니다.

- **Performance Highlights**: 실험 결과, 제안된 반례 유도 접근법을 활용한 경우 평가된 모든 LLM의 수리 능력이 크게 향상되었습니다. 이 방법은 LLM가 더 많은 프로그램을 수정하고, 작은 수정으로 우수한 결과를 도출할 수 있도록 하여 타 구성 및 최첨단 상징적 프로그램 수리 도구 대비 성능이 우수함을 보여주었습니다. 코드와 관련 자료는 GitHub와 Zenodo에서 확인 가능합니다.



### Machine Learning and Quantum Intelligence for Health Data Scenarios (https://arxiv.org/abs/2410.21339)
Comments:
          Presented at Machine Learning and Machine Intelligence (MLMI) Conference, Osaka, Japan 2024

- **What's New**: 이 논문은 양자 컴퓨팅(quantum computing)이 데이터 과학(data science)에서 복잡하고 데이터 집약적인 문제를 해결할 수 있는 새로운 가능성을 열어준다는 점에서 주목할 만하다. 특히, 양자 머신러닝(Quantum Machine Learning, QML)이 전통적인 머신러닝 알고리즘이 어려움을 겪는 고차원(high-dimensional) 및 데이터 품질이 제한적인 데이터셋에서의 해결책을 제시하고 있다.

- **Technical Details**: 양자 머신러닝은 초위치(superposition)와 얽힘(entanglement)과 같은 양자 속성을 활용하여 패턴 인식(pattern recognition) 및 분류(classification)에서의 성능을 향상시킨다. 이 논문은 심장 질환 예측 및 COVID-19 탐지를 위한 양자 커널 방법(quantum kernel methods)과 하이브리드 양자-고전적 네트워크(hybrid quantum-classical networks)의 적용을 심도 있게 다룬다.

- **Performance Highlights**: 연구에서는 양자 커널 방법과 하이브리드 모델의 실용성을 평가하고 이들이 기존의 접근 방법들을 능가할 가능성을 탐색하였다. 특히 헬스케어 분야에서의 적용 가능성은 기존의 머신러닝 기법보다 더 나은 성능을 보일 것으로 기대된다.



### Next Block Prediction: Video Generation via Semi-Autoregressive Modeling (https://arxiv.org/abs/2502.07737)
Comments:
          project page: this https URL

- **What's New**: 이 논문에서는 비디오 생성을 위한 새로운 세미 오토 회귀(semi-autoregressive) 프레임워크인 Next-Block Prediction(NBP)를 제안합니다. 기존의 Next-Token Prediction(NTP) 접근법에서 발생하는 비효율성과 단방향 의존성 문제를 해결함으로써, 비디오 콘텐츠를 동등한 블록으로 분해하여 블록 단위로 예측을 수행합니다. 이를 통해 기존 모델보다 훨씬 빠르고 효율적인 추론이 가능해졌습니다.

- **Technical Details**: NBP 모델은 각 블록 내에서 양방향 주의(bidirectional attention)를 활용하여, 모든 토큰이 블록 내의 다른 토큰들과 관계를 형성할 수 있도록 합니다. 이 방식은 공간적 의존성을 효과적으로 포착하며, 여러 토큰을 동시에 예측함으로써 생성 단계의 수를 크게 줄입니다. NBP는 UCF101과 K600 데이터셋에서 각각 103.3과 25.5의 FVD(score)를 기록하며, 전통적인 NTP 모델보다 평균 4.4점 높은 성과를 보였습니다.

- **Performance Highlights**: NBP 모델은 초당 8.89프레임을 생성할 수 있으며, 이는 기존 모델보다 11배 빠른 속도를 의미합니다. 또한, 모델 크기를 700M에서 3B 파라미터로 확장한 실험 결과, UCF101에서 FVD 점수가 103.3에서 55.3으로, K600에서 25.5에서 19.5로 감소하는 등을 통해 생성 품질이 크게 향상되었습니다. 이러한 스케일러빌리티는 NBP 프레임워크의 강력한 가능성을 보여줍니다.



### Music for All: Exploring Multicultural Representations in Music Generation Models (https://arxiv.org/abs/2502.07328)
Comments:
          17 pages, 5 figures, accepted to NAACL'25

- **What's New**: 이번 연구에서는 자동 음악 생성 분야에서 비서구 음악 장르 및 문화의 편향과 과소 대표성을 정량화했습니다. 연구 결과, 기존 음악 데이터셋의 총 시간 중 비서구 장르는 단 5.7%에 불과하여 음악 생성 모델의 장르 간 성능 차이를 유발합니다. 또한, 매개변수 효율적인 미세 조정 기술(PEFT)을 활용한 음반 적응의 효과를 조사하여 편견을 완화하려는 노력을 했습니다.

- **Technical Details**: 우리는 Hindustani Classical 음악과 Turkish Makam 음악이라는 두 개의 자원 부족 비서구 장르에 대해 두 개의 오픈 소스 모델인 MusicGen과 Mustango를 적응시켰습니다. 연구에서 제안한 PEFT 기법은 모델의 파라미터를 1% 미만으로 추가해 효과적인 성능 향상을 목표로 합니다. 또한, Bloom의 교육 세분화에 기반한 혁신적인 평가 프레임워크를 사용하여 각 모델의 음악 생성 품질을 평가했습니다.

- **Performance Highlights**: 연구 결과, Mustango는 Hindustani Classical 음악에 대한 미세 조정에서 8% 향상되었고, MusicGen은 Turkish Makam에서 4% 향상되는 성과를 보였습니다. PEFT 기법이 저자원 장르의 생성 품질을 향상시키는 데 유효하지만, 모든 모델이 모든 장르에 적합하진 않다는 점이 시사되었습니다. 따라서 다양한 디자인 선택과 데이터셋의 훈련 방식이 모델의 장르 적응 가능성에 중요한 영향을 미친다는 결론을 내렸습니다.



### LLMs in Software Security: A Survey of Vulnerability Detection Techniques and Insights (https://arxiv.org/abs/2502.07049)
Comments:
          33 pages, 12 figures

- **What's New**: 이번 논문은 대형 언어 모델(Large Language Models, LLMs)을 활용한 취약점 탐지 기술에 대한 포괄적인 조사 결과를 제시합니다. LLMs는 코드 구조를 분석하고 패턴을 식별하며 복구 제안을 생성하는 능력으로, 기존의 정적 및 동적 분석 기법에서의 한계를 극복할 수 있는 잠재력을 보여줍니다. 특히 GPT, BERT, CodeBERT와 같은 모델들이 취약점 탐지 분야에 혁신적인 접근 방식을 제공하고 있으며, 현재의 연구 문제와 그 해결책을 다룹니다.

- **Technical Details**: 이 논문은 LLMs의 아키텍처, 적용 방법, 데이터셋, 평가 지표와 같은 핵심 요소를 조사합니다. LLMs는 자연어 처리(NLP) 기술로서, 특히 Transformer 아키텍처를 통해 훈련되어 다양한 NLP 작업에 집중할 수 있습니다. 기존의 전통적인 기법들과 비교했을 때, LLMs는 상대적으로 효율적이며, 코드 분석에서의 정확도를 높이는 잠재력을 가지고 있습니다.

- **Performance Highlights**: LLMs의 적용이 증가함에 따라 사이버 보안 분야에서 긍정적인 영향을 미치고 있으며, 최근 연구에서 C, Java, Solidity 언어에 대한 취약점 탐지 기법이 주목받고 있습니다. 그러나 현재 연구는 기능 수준의 이진 분류에 한정되고 있으며, 레포지토리 차원의 취약점 탐지 및 다중 파일 종속성 문제를 충분히 다루고 있지 않습니다. 논문은 또한 LLMs의 정책 및 데이터셋 관련 한계에 대해서도 언급하며, 향후 연구에 필요한 방향성을 제시하고 있습니다.



New uploads on arXiv(cs.LG)

### Theoretical Benefit and Limitation of Diffusion Language Mod (https://arxiv.org/abs/2502.09622)
Comments:
          32 pages, 3 figures

- **What's New**: 이 논문에서는 텍스트 생성을 위한 최근의 방법론인 Diffusion language models에 대해 다룹니다. 특히, Masked Diffusion Model (MDM)을 분석하여, 해당 모델의 효과는 평가 지표에 크게 의존한다는 것을 밝혔습니다. Perplexity를 사용할 경우 MDM은 최적의 성능을 보일 수 있지만, sequence error rate를 사용할 경우 MDM의 효율성이 감소함을 발견했습니다.

- **Technical Details**: MDM의 진행은 일반적인 autoregressive 모델과 비교하여 효율성이 높지만, 이는 특정 조건 하에서만 유효합니다. Perplexity를 메트릭으로 사용할 때는 효율성을 인정받을 수 있지만, reasoning chain과 같은 '정확성'을 요구하는 경우에는 sequence length에 따라 샘플링 스텝이 선형 배로 증가해야 합니다. 이러한 분석 결과는 실험적 연구에 의해 입증되었습니다.

- **Performance Highlights**: 본 연구는 MDM에 대한 이론적 토대를 마련하여 향후 연구의 방향성을 제시합니다. MDM은 특정 메트릭에서는 효과적인 성능을 나타내나, 보다 복잡한 상황에서는 효율성의 저하를 불러올 수 있음을 보여줍니다. 이는 여러 개발자와 연구자들이 MDM의 장단점을 명확히 이해하고 적용할 수 있도록 도움을 줄 것입니다.



### Can this Model Also Recognize Dogs? Zero-Shot Model Search from Weights (https://arxiv.org/abs/2502.09619)
- **What's New**: 이 논문에서는 ProbeLog라는 새로운 분류 모델 검색 방법을 소개합니다. 이 방법은 '개'와 같은 특정 개념을 인식하는 모델을 찾을 수 있도록 설계되었습니다. 기존의 검색 방법은 모델 메타데이터나 훈련 데이터에 의존하나, ProbeLog는 이들에 대한 접근 없이도 작동합니다. 이를 통해 사용자는 필요한 태스크에 적합한 모델을 더 쉽게 찾을 수 있습니다.

- **Technical Details**: ProbeLog는 고정된 입력 샘플 집합을 사용하여 각 모델의 응답을 관찰함으로써 각 출력 차원의 설명자를 계산합니다. 이 설명자는 로짓(lolgit) 수준의 기능적 표현으로, 모델의 특정 기능을 설명합니다. 또한, 이 방법은 협업 필터링(collaborative filtering)을 기반으로 하여, 모델의 대표성을 세 가지로 줄일 수 있는 기법을 도입함으로써, 비용을 감소시키고 효율성을 높입니다.

- **Performance Highlights**: ProbeLog는 실제 데이터셋에서 높은 검색 정확도를 달성하며, 특히 40%의 top-1 정확도를 기록했습니다. 이는 무작위 방법의 0.1%와 비교할 때 상당히 높은 성능입니다. 커다란 모델 저장소에서도 확장 가능하며, 고효율성을 유지하며 작동합니다. 이를 통해 모델 검색의 새로운 가능성을 제시합니다.



### Variational Rectified Flow Matching (https://arxiv.org/abs/2502.09616)
- **What's New**: 이번 연구는 Variational Rectified Flow Matching에 대한 것으로, 전통적인 rectified flow matching을 개선하여 다중 모드( multi-modal ) 속도 벡터 필드를 모델링합니다. 이 방법은 샘플을 소스 분포에서 목표 분포로 이동시키는 방법을 최적화하여, 다중 모드 흐름 방향을 학습하고 샘플링할 수 있도록 돕습니다. 향상된 결과는 합성 데이터, MNIST, CIFAR-10 및 ImageNet에서 증명되었습니다.

- **Technical Details**: 이 연구에서는 velocity vector-field를 다루는데, 이는 소스 및 목표 분포에서 무작위로 추출된 샘플 간 선형 보간(interpolation)을 통해 학습됩니다. 전통적인 방법에서는 mean-squared-error 손실(loss)을 포함하여 학습된 속도 벡터 필드가 'ground-truth' 방향을 평균적으로 취하게 됩니다. 그러나 variational rectified flow matching에서는 은닉 변수(latent variable)를 도입하여 데이터 영역-시간 영역(data-domain-time-domain)의 각 위치에서 다중 모드를 변별할 수 있게 합니다.

- **Performance Highlights**: synthetic data에서 Variational Rectified Flow Matching은 데이터 분포를 더 정확하게 모델링하여 속도 모호성을 더욱 잘 포착합니다. MNIST에서의 이미지 생성 품질을 개선하고, CIFAR-10에서는 전통적인 rectified flow matching을 초월하는 성능을 보여줍니다. 마지막으로, ImageNet에서 SiT-XL의 FID 점수를 지속적으로 개선하는 결과를 나타내었습니다.



### Designing a Conditional Prior Distribution for Flow-Based Generative Models (https://arxiv.org/abs/2502.09611)
- **What's New**: 본 연구에서는 조건부 생성 모델의 새로운 접근 방식을 제안하며, 특히 비선형한 사전 분포(prior distribution) 설계의 필요성을 강조합니다. 조건을 입력으로 활용하여 평균적으로 같은 조건의 데이터 포인트에 최소한으로 가까운 위치로 매핑하는 방법을 고안하였습니다. 이를 통해 샘플링 단계 수를 줄이면서도 높은 품질의 결과를 생성할 수 있도록 하였습니다.

- **Technical Details**: 본 연구의 방법은 조건부 흐름 기반 생성 모델을 위한 비정보적인 사전 분포를 설계하는 것을 목표로 합니다. 주어진 입력 조건에 대해서, 데이터 공간에서 가장 '평균적인' 조건을 찾고, 이를 중심으로 하여 확률 분포를 설정하여, 샘플을 조건부 목표 분포로 매핑합니다. 이를 위해 가우시안 혼합 모델(GMM)과 사전 분포의 설계 및 흐름 매칭(flow matching) 방법론을 결합하여 효과적인 조건부 생성에 기여하였습니다.

- **Performance Highlights**: 실험 결과, 제안하는 방법은 기존 모델들과 비교했을 때 빠른 훈련 및 샘플링 속도를 보여주었으며, 생성된 이미지의 품질과 다양성 면에서도 현저한 개선이 있음을 증명하였습니다. FID, KID 및 CLIP 점수와 같은 평가 지표에서 우수한 결과를 기록하였습니다. 이를 통해 본 연구의 접근 방식이 실제 데이터셋에서도 효과적으로 적용될 수 있음을 입증하였습니다.



### Score-of-Mixture Training: Training One-Step Generative Models Made Simp (https://arxiv.org/abs/2502.09609)
Comments:
          27 pages, 9 figures

- **What's New**: 이번 논문에서는 Score-of-Mixture Training (SMT)이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 $
α$-skew Jensen-Shannon divergence라는 종류의 발산(divergence)을 최소화하여 일단계 생성 모델을 학습합니다. SMT는 실제 샘플과 가짜 샘플 간의 혼합 분포(score of mixture distributions)를 여러 잡음 수준(noise levels)에서 추정하는 것을 핵심으로 합니다.

- **Technical Details**: SMT는 일관성 모델(consistency models)과 유사하게, 처음부터 학습하는 방법(SMT)과 사전 훈련된 확산 모델(diffusion model)을 활용한 증류(distillation) 방법(SMD)을 지원합니다. 이 방법은 구현이 간단하고 하이퍼파라미터 튜닝(hyperparameter tuning)이 최소한으로 필요하며, 안정적인 학습을 보장합니다.

- **Performance Highlights**: CIFAR-10과 ImageNet 64x64 데이터셋에서 실험한 결과 SMT/SMD가 기존 방법들과 경쟁력이 있으며, 심지어 이들을 초월할 수 있음을 보여주었습니다. 이러한 결과는 SMT/SMD의 잠재력을 뒷받침하고 있으며, 생성 모델 분야에서 중요한 기여를 할 것으로 예상됩니다.



### Do LLMs Recognize Your Preferences? Evaluating Personalized Preference Following in LLMs (https://arxiv.org/abs/2502.09597)
Comments:
          Accepted at ICLR 2025 as oral presentation. Code and data at: this https URL

- **What's New**: PrefEval은 사용자 선호도를 장기 대화 설정에서 추론하고 기억하며 준수하는 LLM의 능력을 평가하기 위한 새로운 벤치마크입니다. 이 데이터셋은 20개의 주제를 아우르는 3,000개의 사용자 선호 및 쿼리 쌍으로 구성되어 있으며, LLM의 성능 평가에는 생성(task) 및 분류(task) 과제가 포함됩니다. 이 연구는 현재 LLM들이 사용자 선호도를 효과적으로 반영하는 데 있어 중요한 한계를 보이고 있음을 드러냅니다.

- **Technical Details**: PrefEval은 사용자 선호를 반영하거나 암시하는 방식으로 정보가 제시되는 다양한 형태를 포함하고 있습니다. 데이터셋은 명시적 또는 암시적 형태의 선호 정보를 포함하며, 사용자의 메시지 및 LLM의 응답을 추적하는 다중 턴 대화를 기준으로 설정됩니다. 10가지의 오픈 소스 및 상용 LLM을 사용하여 다양한 컨텍스트 길이(최대 100k 토큰)에서 평가를 수행했습니다.

- **Performance Highlights**: 현재 최첨단 LLM은 사용자의 선호를 적극적으로 반영하는 데 큰 어려움을 겪고 있으며, 특히 10턴의 제로 샷(zero-shot) 설정에서는 정확도가 10% 미만으로 나타났습니다. 여러 응답 형태에서 사용자 선호도를 잘 인식하지 못하여 성능이 저하되는 것으로 나타났습니다. 그러나 PrefEval에서 파인튜닝(fine-tuning)을 실시했을 때 전반적인 성능 향상이 관찰되었습니다.



### Censor Dependent Variational Inferenc (https://arxiv.org/abs/2502.09591)
- **What's New**: 이 논문은 생존 분석을 위한 잠재 변수 모델에서 변분 추론(variational inference)의 포괄적인 분석을 제공합니다. 기존 방법론에서의 주요 약점을 식별하고, 이는 부적절하게 설계된 변분 분포가 생존 분석 작업의 목표인 사건 발생 시간(time-to-event distribution) 모델링을 방해할 수 있음을 증명합니다. 이러한 문제를 해결하기 위해 생존 분석에 맞춘 변분 추론 기법인 Censor-Dependent Variational Inference (CDVI)를 제안합니다.

- **Technical Details**: 생존 분석의 주요 도전 중 하나는 센서 링(observations) 정보의 부족으로 인해 전문적인 방법을 필요로 하는 것입니다. 본 논문은 센서 링 메커니즘에 의존하는 최적 변분 분포에 대한 분석을 포함하여, CD-CVAE(V-structure Variational Autoencoder)의 구현을 통해 CDVI를 확장합니다. 이 과정에서 다양한 기존 이론 및 훈련 기법을 생존 분석에 맞게 적용하는 방법을 제시합니다.

- **Performance Highlights**: 성능 평가를 위해 폭넓은 실험을 진행하였으며, 결과적으로 개별 생존 분포 추정에서 상당한 개선이 있음을 보여줍니다. 제안된 CDVI와 CD-CVAE 모델은 다양한 지표에서 기존의 생존 분석 방법들과 비교해 우수한 성능을 보였습니다. 이는 잠재 정보의 추출이 후속 작업에 대한 다양한 활용 가능성을 제공함을 의미합니다.



### Rolling Ahead Diffusion for Traffic Scene Simulation (https://arxiv.org/abs/2502.09587)
Comments:
          Accepted to Workshop on Machine Learning for Autonomous Driving at AAAI 2025

- **What's New**: 본 연구에서는 NPC(Non-Playable Characters)의 자연스러운 운전 행동을 모방하고, 다른 시뮬레이션 에이전트의 행동에 반응하는 능력을 갖춘 운전 시뮬레이션 모델을 제안합니다. 특히, Rolling Diffusion 모델을 사용해 즉각적인 미래와 부분적으로 잡음이 있는 향후 단계를 동시에 예측하며, 이를 통해 반응성과 계산 효율성의 균형을 이룹니다.

- **Technical Details**: Diffusion 모델의 기본 원리는 데이터를 점진적으로 노이즈를 추가하는 전방 과정(Forward Process)에 기반하고 있습니다. 본 논문에서 제안하는 방법은 2초의 예측 창을 사용하며, 기존의 Autoregressive diffusion 모델에 비해 처리 평가 횟수를 4배 줄이면서 시뮬레이션 속도를 높입니다. 또한, 이 모델은 마지막 시뮬레이션 단계에서 모든 에이전트의 상태를 인식한 후 다음 상태를 예측하는 방식을 채택합니다.

- **Performance Highlights**: 실험 결과, 제안된 롤링-어헤드 교통 계획 모델은 기존의 diffusion 기반 Autoregressive 모델보다 더 현실적인 장면을 생성하는 것으로 나타났습니다. 이 연구는 실제 도로 주행 로그에 기반하여 훈련된 결과를 분석하고, 장면 수준 변위 지표로 측정한 평가에서 우수성을 입증했습니다.



### Learning to Coordinate with Experts (https://arxiv.org/abs/2502.09583)
- **What's New**: 이번 논문에서는 AI 에이전트가 다이나믹 환경에서 전문가(Expert) 에이전트의 도움을 요청하는 방법을 학습하는 Coordination 문제인 Learning to Yield and Request Control (YRC)를 소개합니다. 이 연구는 에이전트가 훈련 중에는 전문가와 상호작용하지 않지만, 테스트 시에 새로운 환경 변화와 전문가의 개입에 적응하는 도전적인 설정을 고려합니다. 이를 지원하기 위해 YRC-Bench라는 오픈소스 벤치마크를 개발하였으며, 다양한 도메인에서의 실험을 가능하게 합니다.

- **Technical Details**: YRC-Bench는 Gym과 유사한 API를 제공하며, 다양한 환경을 포함하여 전문가와 초급자 정책 간의 제어 위임을 위한 표준화된 인터페이스를 만족합니다. 새로운 HelpEnvironment 래퍼를 통해 이론적으로 수행되는 Think Action Policy를 구체적인 환경 행동으로 변환하는 과정을 설명합니다. 또한, 정책 통합, 비용 계산, 성과 추적을 포함한 다양한 기능을 수행하여 환경에서의 에이전트 행동을 평가하고 최적화하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 성능 평가는 누적 비용, 작업 성공률, 전문가 개입 빈도 등 도메인 특화 지표를 통해 이루어집니다. 이 평가 프레임워크는 다양한 방법 간의 공정하고 의미 있는 비교를 보장합니다. 연구자들은 이 모듈화된 설계를 통해 새로운 알고리즘을 쉽게 통합할 수 있으며, 반복적 훈련 과정에서 성능을 지속적으로 평가하여 최상의 모델을 저장할 수 있는 기회를 제공합니다.



### DiffMS: Diffusion Generation of Molecules Conditioned on Mass Spectra (https://arxiv.org/abs/2502.09571)
Comments:
          Preprint

- **What's New**: DiffMS는 질량 스펙트럼을 기반으로 분자의 구조를 생성하기 위해 설계된 최초의 조건부 생성 모델입니다. 이 모델은 화학 공식을 제약으로 두어 효율적인 분자 생성이 가능하며, 현대의 transformer 아키텍처를 활용하여 질량 스펙트럼의 도메인 지식을 모델링합니다. 실험 결과, DiffMS는 기존의 생성 모델들과 비교했을 때 우수한 성능을 보여줍니다.

- **Technical Details**: DiffMS는 엔코더-디코더 구조를 갖추고 있으며, 엔코더는 MIST의 화학 공식 transformer를 채택하고, 디코더는 화학 공식 제약을 활용한 이산 그래프 확산 모델을 사용합니다. 또한, 디퓨전 디코더는 구조-지문 쌍 데이터를 사용해 사전 훈련되며, 이는 방대한 양의 데이터로 인해 성능 향상에 기여합니다.

- **Performance Highlights**: DiffMS는 기존의 기준 모델들보다 우수한 성능을 보여주며, 분자의 주석 정확도가 개선되고 실제 분자와의 구조 유사성이 향상되었습니다. 본 논문에서 제안한 DiffMS는 화학 및 생물학적 발견 연구를 촉진할 잠재력을 지니고 있습니다.



### Enhancing the Utility of Higher-Order Information in Relational Learning (https://arxiv.org/abs/2502.09570)
- **What's New**: 이 논문에서는 고차원 정보가 관계 학습(relational learning)에서 얼마나 중요한지를 보여주며, 특히 하이퍼그래프(hypergraph)를 적용한 그래프 신경망(graph neural networks) 아키텍처의 발전을 다루고 있습니다. 기존 그래프 아키텍처는 하이퍼그래프 확장에 적용할 때 더 우수한 성능을 보이는 반면, 하이퍼그래프 아키텍처는 그 성능이 다소 떨어지는 것으로 나타났습니다. 연구팀은 고차원 정보를 활용하기 위한 새로운 접근법인 하이퍼그래프 인코딩(hypergraph-level encodings)을 제안합니다.

- **Technical Details**: 하이퍼그래프(H)와 그래프(G)의 구조에 대한 기본 개념을 다루며, 이들 간의 관계를 명확히 하고 있습니다. 메시지 전달(message-passing) 접근 방식과 트랜스포머(transformer) 기반 아키텍처를 포함하여, 다양한 그래프 신경망 아키텍처가 어떻게 하이퍼그래프에 확장될 수 있는지를 설명합니다. 연구의 일환으로, 하이퍼그래프 인코딩과 같은 새로운 인코딩 방식을 도입하여 성능을 향상시키는 방법을 분석합니다.

- **Performance Highlights**: 실험 결과에 따르면, 하이퍼그래프 인코딩을 결합할 경우 그래프 신경망의 성능이 크게 향상되는 것으로 나타났습니다. 그래프 수준 모델에 비해 하이퍼그래프 아키텍처는 성능이 떨어지지만, 하이퍼그래프 인코딩을 활용하는 경우에는 성능 개선이 두드러집니다. 궁극적으로, 하이퍼그래프 수준 인코딩이 관계형 데이터에서 고차원 정보를 효과적으로 활용하는 방법을 제공함을 입증했습니다.



### Diffusing DeBias: a Recipe for Turning a Bug into a Featur (https://arxiv.org/abs/2502.09564)
Comments:
          29 Pages, 12 Figures

- **What's New**: 이번 연구에서는 Diffusing DeBias (DDB)라는 새로운 방법론을 제안하였습니다. DDB는 일반적인 디바이싱(debiasing) 기법과 함께 사용할 수 있는 플러그인 형태로, 조건부 확산 모델(Conditional Diffusion Model)의 편향 학습 특성을 활용합니다. 이 접근법을 통해 인위적으로 생성된 편향 정렬(bias-aligned) 이미지를 통해 편향 앰프 모델을 훈련시키고, 훈련 집합의 기억화 문제를 해결하여 더 나은 일반화 성능을 달성하는 것을 목표로 합니다.

- **Technical Details**: DDB는 확산 모델의 특성을 활용하여 각 클래스의 편향 정렬 분포를 학습합니다. 이를 통해 생성된 합성 이미지를 사용하여 편향 앰프 모델을 훈련시키며, 이 모델은 다양한 비지도 디바이싱 기법에서 보조 방법으로 활용됩니다. 연구 결과, DDB는 특히 훈련 데이터에 대한 편향 극복에 있어 기존의 최첨단 기법들보다 우수한 성능을 보였습니다.

- **Performance Highlights**: DDB는 두 가지 서로 다른 디바이싱 프레임워크에 통합되어 최고의 성능을 낼 수 있는 가능성을 보여줍니다. Recipe I에서는 G-DRO 알고리즘을 활용하여 서브 집단을 추출하며, Recipe II에서는 종단 간(end-to-end) 방법에서 손실 함수를 제공합니다. 두 접근법 모두 여러 기준 데이터셋에서 기존의 최첨단 방법들을 상당한 차이로 초월하는 성과를 달성하고 있습니다.



### Robust Learning of Multi-index Models via Iterative Subspace Approximation (https://arxiv.org/abs/2502.09525)
- **What's New**: 이번 논문에서는 가우시안 분포 하에서 레이블 노이즈가 있는 다중 인덱스 모델(Multi-Index Models, MIMs)을 학습하기 위한 새로운 알고리즘을 개발하였습니다. 특히, K차원 부분공간에만 의존하는 K-MIM을 다룹니다. 저자들은 정규성(regularity) 속성을 만족하는 맥락에서 유용한 MIM에 집중하며, 일반적인 강건 학습기(robust learner)를 제안합니다. 이 알고리즘은 조건부 저차 모멘트를 계산하여 점진적으로 더 나은 부분공간 근사를 구축하는 방식으로 작동합니다.

- **Technical Details**: 제안된 알고리즘은 다중 클래스 선형 분류기와 halfspace의 교차를 위한 특화된 강건 학습기를 포함합니다. 알고리즘의 샘플 복잡도는 각 경우 $N = O(d) 2^{	ext{poly}(K/	ext{ε})}$ 및 $N=O(d^2) 2^{	ext{poly}(K/	ext{ε})}$로 나타납니다. 이는 모든 적절한 매개변수에 대해 효율적인 학습기를 제시하는데 기여하며, 특히 여러 기능은 조건부 모멘트 계산을 통해 더욱 향상됩니다.

- **Performance Highlights**: 이 연구의 강건 학습기는 Multiclass Linear Classifiers와 Intersections of Halfspaces에 대해 각각 기반하여 가깝고 효율적인 경과를 창출함을 보여줍니다. 특히, 무작위 분류 노이즈(Random Classification Noise)가 있는 경우에도 알고리즘은 복잡성이 $1/	ext{ε}$에 대해 다항적으로 증가함을 시사합니다. 이러한 결과들은 기존 알고리즘과 비교하여 높은 성능과 유연성을 부여하며, 다중 클래스 설정에서의 학습적 난제를 해결하는 방향으로 기여합니다.



### Diffusion Models for Molecules: A Survey of Methods and Tasks (https://arxiv.org/abs/2502.09511)
- **What's New**: 이 논문은 분산 모델(difffusion model)에 기반한 분자 생성 방법에 대한 종합적인 조사(research survey)를 제공하며, 이 분야의 현재 상태와 발전 가능성을 체계적으로 정리합니다. 특히, 다양한 연구를 체계적으로 분류하여 과학자들이 이 복잡한 영역을 보다 쉽게 이해하고 탐색할 수 있도록 도와주고자 합니다. 이를 통해 분자 디자인 분야의 혁신을 촉진하려는 의도가 담겨 있습니다.

- **Technical Details**: 분산 모델은 두 가지 주요 과정, 즉 전방 확산(propagation) 과정과 역 생성(reverse generation) 과정으로 구성됩니다. 전방 과정에서는 실제 데이터에 점진적으로 노이즈를 추가하여 단순한 사전 분포(prior distribution)로 접근합니다. 역 과정에서는 이 노이즈로부터 데이터 분포를 점진적으로 복원하는 법을 학습하며, 이는 일반적으로 신경망(neural networks)을 사용하여 매개변수화됩니다.

- **Performance Highlights**: 분산 모델은 최근 다양한 도메인에서 고품질 데이터를 생성하는 데 뛰어난 성능을 보이며, 분자 생성 과제에서도 큰 잠재력을 보여주고 있습니다. 이러한 모델들은 복잡한 분자 구조와 특성을 효과적으로 모델링하며, 현대 과학 연구에서 필수적인 도구로 자리잡고 있습니다. 이로 인해 분자 설계 분야의 연구가 급증하고 있으며, 연구자들은 이 모델의 변형 및 적용 가능성에 대해 활발히 탐구하고 있습니다.



### EQ-VAE: Equivariance Regularized Latent Space for Improved Generative Image Modeling (https://arxiv.org/abs/2502.09509)
Comments:
          Preprint

- **What's New**: 본 연구에서는 기존의 autoencoder가 스케일링과 회전 같은 변환에 대한 equivariance가 결여되어 있음을 지적하고, 이를 해결하기 위한 새로운 접근 방식인 EQ-VAE를 제안합니다. EQ-VAE는 잠재 공간(latent space)의 복잡성을 줄이면서 재구성 품질을 저하시키지 않도록 돕는 규제 전략입니다. 이를 통해 여러 최첨단 생성 모델의 성능을 향상시킬 수 있음을 보여주었습니다.

- **Technical Details**: EQ-VAE는 autoencoder의 구조를 변경하지 않고, 미리 학습된 모델을 몇 번의 epoch만 조정(fine-tuning)함으로써 작동합니다. 이는 변환된 잠재 표현의 재구성을 입력 이미지의 변환과 비교하여 불일치를 벌칙으로 걸어 잠재 공간의 동등성(equivariance)을 장려합니다. 해당 방법은 연속적 및 이산적(autoencoder) 모두에 적용 가능하여 광범위한 숨겨진 생성 모델에 유용하게 활용될 수 있습니다.

- **Performance Highlights**: EQ-VAE를 적용한 결과, DiT, SiT, REPA, MaskGIT과 같은 여러 생성 모델의 성능이 크게 향상되었습니다. 예를 들어, SD-VAE를 통한 fine-tuning 후 DiT-XL/2 모델에서 교육 속도가 7배 빨라졌습니다. 이러한 성능 향상은 정보의 보존과 밤과 모양의 대응에 대한 보장을 통해 실현되었습니다.



### When and How Does CLIP Enable Domain and Compositional Generalization? (https://arxiv.org/abs/2502.09507)
- **What's New**: 이번 연구에서는 CLIP(Contrastive Language-Image Pretraining) 모델이 다양한 도메인에서의 일반화 성능을 어떻게 발휘하는지에 대한 체계적인 분석을 진행했습니다. 기존 연구에서 CLIP의 훈련 배포의 다양성이 OOD(Out-of-Distribution) 일반화 성능의 주요 원인이라고 제시되었으나, 어떤 요인이 이러한 일반화에 영향을 미치는지에 대한 질문이 남아 있었습니다. 본 논문은 도메인 일반화와 구성적 일반화(compositional generalization)에 대한 탐구를 통해 CLIP의 성능 저하를 설명하려 하였습니다.

- **Technical Details**: CLIP 모델을 훈련시키기 위해 자연 이미지와 비자연 샘플들로 구성된 데이터셋을 사용하여 도메인 믹스를 체계적으로 구성하였습니다. 이 과정에서 훈련 중 노출된 클래스와 도메인의 다양성을 정밀하게 조작할 수 있는 실험 환경을 마련하였습니다. 연구 결과, 도메인 다양성이 일반화에 필수적이며, 구성적 일반화는 때때로 도메인 일반화보다 약하다는 것을 발견했습니다.

- **Performance Highlights**: CLIP 모델의 일반화 성능은 다양한 도메인을 사용하는 훈련 데이터에 의해 개선됩니다. 실험 결과, 클립은 특정 도메인에서 제한된 특징만을 공유할 경우 일반화에 실패하는 경향이있습니다. 따라서 성공적인 일반화를 위해서는 특징과 회로 공유가 충분해야 하며, 이는 CLIP 모델의 중간 레이어에서의 표현 공유를 통해 이루어집니다.



### AttentionSmithy: A Modular Framework for Rapid Transformer Development and Customization (https://arxiv.org/abs/2502.09503)
- **What's New**: 이 논문은 Transformer 아키텍처를 더 쉽게 커스터마이즈할 수 있도록 돕는 'AttentionSmithy'라는 모듈형 소프트웨어 패키지를 소개합니다. 사용자는 이 패키지를 통해 복잡한 코딩 없이 재사용 가능한 컴포넌트를 통해 다양한 transformer 변형을 빠르게 프로토타입하고 평가할 수 있습니다.

- **Technical Details**: AttentionSmithy는 attention 모듈, feed-forward 네트워크, normalization 레이어 및 positional encodings과 같은 주요 구성 요소를 분해하여 제공합니다. 이 프레임워크는 네 가지 positional encoding 전략을 지원하며, 자동 설계를 위한 neural architecture search와 통합됩니다.

- **Performance Highlights**: AttentionSmithy는 원래의 transformer를 복제하고 리소스 제약 하에서 번역 성능을 최적화하는 데 성공했습니다. 또한, 특정 유전자 모델링에서 95% 이상의 세포 유형 분류 정확도를 달성하여 다양한 분야에서 연구를 가속화할 가능성을 보여줍니다.



### Scalable First-order Method for Certifying Optimal k-Sparse GLMs (https://arxiv.org/abs/2502.09502)
- **What's New**: 이 논문에서는 $	ext{GLM}$(Generalized Linear Models)의 희소 최적성을 인증하는 문제를 다룹니다. 특히, $	ext{l}_0$ 카디널리티 제약 조건을 통해 희소성을 유지하는 방식으로 접근하고 있습니다. 기존의 방법들과 비교할 때, 제안한 방식은 계산 집약적이지 않으면서도 빠른 수렴 속도를 보여주는 것이 특징입니다.

- **Technical Details**: 본 연구에서는 BnB(Branch-and-Bound) 프레임워크 내에서 문제의 관점 완화(perspective relaxation)를 해결하기 위한 1차 근접 경량 알고리즘을 제안합니다. 이 완화된 문제는 복합 최적화(composite optimization) 문제로 설정되며, 비매끄러운(non-smooth) 성분의 근접 연산자(proximal operator)는 로그 선형(log-linear) 시간 복잡도로 정확히 계산할 수 있음을 보여줍니다. 이로 인해 계산 비용이 많이 드는 2차 원뿔 프로그램을 해결할 필요가 없어집니다.

- **Performance Highlights**: 제안된 접근 방식은 가상의 데이터셋과 실제 데이터셋에서 다수의 실험을 통해 이중 경계 계산을 현저히 가속화하며 대규모 문제에 대한 최적성 인증을 효과적으로 제공함을 입증하였습니다. 또한, 간단한 재시작 전략이 도입되어 수렴 속도를 향상시키면서도 iteration당 낮은 복잡도를 유지하고 있습니다.



### Eidetic Learning: an Efficient and Provable Solution to Catastrophic Forgetting (https://arxiv.org/abs/2502.09500)
Comments:
          16 pages, 6 figures; code is available at this https URL

- **What's New**: 이 논문에서는 기존의 신경망이 겪는 치명적인 망각(catatstrophic forgetting) 문제를 해결하는 방법인 Eidetic Learning을 소개합니다. Eidetic Learning은 네트워크가 특정 작업(task) t1을 학습한 이후, 다른 작업 t2를 학습했을 때 t1의 수행 능력을 잃지 않도록 보장합니다. 이 방법은 특히 네트워크의 용량을 활용하여 작업 간의 데이터 라우팅을 자동으로 수행할 수 있는 특징을 가지고 있습니다.

- **Technical Details**: EideticNet은 신경망의 과잉 용량(overparameterization)을 활용하여 다양한 작업을 각각의 서브네트워크로 나누어 처리합니다. 이 구조는 반복적 가지치기(iterative pruning) 기법을 사용하여, 특정 작업의 수렴 후 중요하지 않은 뉴런을 제거하고 이전 작업에서 학습한 특징을 다음 작업에 재사용할 수 있게 하여 학습 효율성을 높입니다. 또한, traditional 학습 방식과 달리 고전적인 리허설(rehearsal)이나 재생(replay)을 요구하지 않는다는 것이 큰 장점입니다.

- **Performance Highlights**: EideticNet은 다양한 네트워크 아키텍처와 작업 세트를 통해 망각에 저항력을 가짐을 실증적으로 보여주었습니다. 이 시스템은 높은 효율성과 함께 학습한 표현(representation)을 보존하면서도 실질적인 성능 향상을 기대할 수 있습니다. 그리고 PyTorch 프레임워크 내에서의 손쉬운 구현이 가능하여, 현실 세계의 다양한 응용 프로그램에 적합하게 설계되었습니다.



### On Agnostic PAC Learning in the Small Error Regim (https://arxiv.org/abs/2502.09496)
Comments:
          44 pages

- **What's New**: 최근 연구에 따르면, Empirical Risk Minimization (ERM) 학습자가 실현 가능한 경우에 최적이 아니지만 비실현 가능(ex. agnostic) 경우에 최적이라는 흥미로운 현상이 나타납니다. 이 연구는 비실현 가능한 분포가 학습하기 더 어려운 이유와 관련이 있으며, Hanneke, Larsen, Zhivotovskiy의 최신 논작에서는 이 단점을 보완하기 위해 agnostic 오류 항에 새롭게 매개변수인 τ를 포함하는 방법을 제시했습니다. 본 연구에서는 τ와 d/m 간의 관계를 규명하는 데 중점을 두었습니다.

- **Technical Details**: 이 논문은 하이퍼패러미터 τ가 목표 오류의 하한을 설정하는 데 어떻게 사용되는지 설명합니다. 학습자의 오류는 τ + Ο(√(τ(d + log(1/δ))/m) + (d + log(1/δ))/m) 형태로 나타나며, τ > d/m인 경우에서 올바르게 예측됩니다. 본 논문에서는 τ ≈ d/m에 대한 새로운 학습자를 제시하여 상대적으로 적은 오류를 달성할 수 있음을 보여줍니다.

- **Performance Highlights**: 새로운 학습자는 τ와 관련된 오류를 c · τ + O(√(τ(d + log(1/δ))/m) + (d + log(1/δ))/m)로 표현하며, 여기서 c는 2.1 이하의 상수입니다. 이는 기존의 하한을 충족시키는 결과로,이에 따라 비판적인 학습 문제에 대한 실질적인 발전이 이루어졌습니다. 현재 연구는 이 상수를 2.1에서 1로 낮추는 방법에 대한 가능성도 남겨두었습니다.



### Inverse Design with Dynamic Mode Decomposition (https://arxiv.org/abs/2502.09490)
Comments:
          29 pages, 19 figures

- **What's New**: 본 논문에서는 과학 및 공학 분야의 역설계(inverse design) 자동화를 위한 새로운 컴퓨팅 효율적인 방법을 소개합니다. 본 방법은 단순한 최소 제곱 회귀(least-square regression)에 기반하여 동적 모드 분해(dynamic mode decomposition) 알고리즘을 사용해 파라미터 공간에서 여러 실험을 포함하는 저차원 서브스페이스(low-rank subspace)를 구성합니다. 제안된 ID-DMD(역설계 동적 모드 조합) 알고리즘은 이러한 계산된 저차원 서브스페이스를 활용하여 빠른 디지털 설계 및 최적화를 가능하게 합니다.

- **Technical Details**: ID-DMD 알고리즘은 저차원 선형 서브스페이스를 기반으로 하여 다양한 설계 파라미터에 대한 선형 동역학을 최적화하기 위해 간단한 최소 제곱 회귀를 적용합니다. 이 방법은 여러 실험적인 파라미터화 실험을 포함하여 계산 효율성을 극대화하며, 시스템 동역학을 안정된 모드로 명확하게 재구성하여 신뢰성을 보장합니다. ID-DMD의 효율성과 정확성은 최적화 알고리즘을 사용하여 목적에 맞는 역설계를 가능하게 합니다.

- **Performance Highlights**: ID-DMD는 기존의 데이터 기반 방법에 비해 1차 더 정확하고 수십 배 더 빠른 CPU 기반 훈련을 가능하게 합니다. 또한, 구조 진동부터 유체 역학에 이르는 복잡한 설계 문제를 처리할 수 있는 확장성을 제공합니다. 본 알고리즘은 물리적으로 해석 가능하며, 장기 예측과 훈련 범위를 초과한 설계 파라미터의 외삽을 보장합니다. 이러한 속도, 견고성, 해석 가능성 및 사용의 용이성 덕분에 ID-DMD는 역설계 및 최적화 분야에서 데이터 기반 방법의 중요한 발전을 나타냅니다.



### Learning to Predict Global Atrial Fibrillation Dynamics from Sparse Measurements (https://arxiv.org/abs/2502.09473)
Comments:
          Under review

- **What's New**: FibMap는 심방세동(Atrial Fibrillation, AF)의 동역학을 희소 측정값으로부터 재구성하는 그래프 순환 신경망 모델을 도입합니다. 기존의 접촉 매핑 카테터의 한계를 극복하고, 10%의 표면 범위에서도 전체 심방 동역학을 재구성할 수 있는 능력을 갖추고 있습니다. FibMap는 51명의 비접촉 전체 심방 기록에서 훈련 및 검증을 통해 평균 절대 오차를 210% 낮추며, 동특징을 추적하는 성능도 크게 향상하였음을 보여줍니다.

- **Technical Details**: FibMap는 각 환자에 대한 전기생리학적 특성을 제공하고, AF 동역학을 효율적으로 재구성하여 개인화된 치료를 가능하게 합니다. 그래프 순환 신경망(Graph Recurrent Neural Network, GRNN)을 통해 AF를 커플링된 진동자의 네트워크로 모델링 하여, 비연속적인 접촉 매핑 데이터로부터 당사자별 동역학을 정확하게 재현합니다. 이 모델은 비연속 측정값으로부터 전체 심방 동역학을 정교하게 복원할 수 있도록 설계되었습니다.

- **Performance Highlights**: FibMap는 실제 접촉 매핑 기록에서도 임상적 유용성을 증명하며, 기존의 비접촉 매핑과 비례하는 복원 충실도를 달성하였습니다. 전체 심방 동역학을 효과적으로 복원하고, 복잡성을 줄이며 AF 매핑 절차의 리스크를 최소화할 수 있는 잠재력을 가지고 있습니다. FibMap 통합을 통해 AF 치료의 개인화가 가능해지고, 결과 향상에 기여할 것으로 기대됩니다.



### Relational Conformal Prediction for Correlated Time Series (https://arxiv.org/abs/2502.09443)
- **What's New**: 본 논문은 상관된 시계열 예측에서 불확실성 정량화 문제를 다뤘습니다. 특히, 그래프 표현을 활용한 관계형 딥 러닝 방법을 통해 시공간 데이터로부터 점 추정치를 얻는 데 효과적인 도구를 제공함을 강조합니다. 이러한 접근법은 기존의 예측_interval(conformal prediction) 방법들이 독립적으로 작동하는 점을 보완하여 관계를 고려할 수 있도록 합니다.

- **Technical Details**: 저자들은 CoRel이라고 불리는 새로운 관계형 예측 방법을 제안합니다. 이 방법은 그래프 심층 학습(graph deep learning) 기법을 활용하여 상관된 시계열 데이터를 다룹니다. 또한, CoRel은 입력 시계열의 변화에 따라 적응 가능한 요소를 포함하여 비교 가능한 정확성을 유지하면서 비정상적(non-stationary) 데이터의 처리를 지원합니다.

- **Performance Highlights**: CoRel 방법은 다양한 데이터셋과 상황에서 기존의 CP 접근법들보다 뛰어난 성능을 보였습니다. 이는 정확한 예측 범위를 제공하며, 관련 벤치마크에서 최신의 불확실성 정량화(state-of-the-art uncertainty quantification)를 달성했습니다. 이러한 결과는 CoRel이 복잡한 관계를 이해하고 반영하는 데 유용하다는 것을 보여줍니다.



### A Survey of Reinforcement Learning for Optimization in Automation (https://arxiv.org/abs/2502.09417)
Comments:
          8 pages, 4 tables, and 1 figure. Accepted at IEEE 20th International Conference on Automation Science and Engineering (CASE) 2024

- **What's New**: 이 논문에서는 강화 학습(Reinforcement Learning, RL)이 자동화 분야 내에서 최적화 문제를 해결하는 데 필수적인 도구로 자리 잡았음을 강조합니다. 제조업, 에너지 시스템, 로봇 공학 등 다양한 분야에서 RL의 활용을 살펴보며, 특히 각 분야에서의 최신 동향과 주요 도전 과제를 논의합니다. RL 기반의 최적화 접근 방법은 기존의 전통적인 방법에 비해 보다 유연하고 강력한 결과를 제공할 수 있음을 보여줍니다.

- **Technical Details**: RL은 실험을 통해 학습하는 능력이 뛰어나며, 이를 통해 명시적인 감독이나 사전 정의된 모델 없이도 최적의 정책을 학습할 수 있습니다. 각 응용 분야에서는 RL의 강점을 극대화하기 위해 Deep Reinforcement Learning (DRL) 및 Multi-Agent Reinforcement Learning (MARL)과 같은 최신 알고리즘이 사용됩니다. RL은 데이터 효율성, 안전성, 해석 가능성 및 신뢰성을 개선하고, 실제 환경에서의 배포와 통합 문제를 해결하기 위한 많은 연구가 진행되고 있습니다.

- **Performance Highlights**: 제조업에서 RL은 생산 예약, 재고 관리, 유지 보수 계획 및 공정 제어를 통해 복잡한 최적화 문제를 해결하는 데 기여하고 있습니다. 에너지 시스템에서는 드라이브 반응(Demand Response) 및 마이크로그리드 관리에 있어 RL이 최대 22%의 에너지 절약을 성취하는 등 그 효과를 입증하였습니다. 로봇 공학 분야에서는 DRL의 활용이 로봇의 동작 계획 및 인간과의 협력에서 특히 두드러지며, 이는 다양한 작업에서 협력을 개선하는 데 기여하고 있습니다.



### A hierarchical approach for assessing the vulnerability of tree-based classification models to membership inference attack (https://arxiv.org/abs/2502.09396)
- **What's New**: 이 논문은 머신러닝 모델이 훈련 데이터의 기밀 속성을 노출시킬 수 있는 취약성 문제를 다룬다. 특히, 효율적으로 Membership Inference Attack (MIA)에 취약한 나무 기반 모델을 식별하기 위한 두 가지 새로운 방법론을 제안했다. 첫 번째는 하이퍼파라미터 선택에 대한 ante-hoc 분석이며, 두 번째는 훈련된 모델 구조에 대한 post-hoc 분석이다. 이러한 방법들은 비싼 MIA 평가를 피할 수 있도록 돕는다.

- **Technical Details**: 하이퍼파라미터 조합의 위험 순위가 데이터셋 간에 일관되다고 보여주며, 이는 훈련 전에 상대적으로 높은 위험 모델을 식별할 수 있게 한다. 또한, 훈련된 모델 구조에 대한 저렴한 구조적 메트릭을 설계하여 MIA 취약성을 평가하는 데 기여한다. 이러한 두 가지 접근법을 통해, 모델 분석의 전후 단계에서 위험을 줄일 수 있는 방법을 제공한다.

- **Performance Highlights**: 실험 결과는 하이퍼파라미터 기반의 위험 예측 규칙이 서로 다른 모델 유형에서 높은 정확도로 리스크 조합을 예측함을 보여준다. 또한, 모델의 정확도와 개인정보 보호 리스크 사이에는 상관관계가 없으며, 이는 모델 성능과 개인정보 보호를 동시에 최적화할 기회를 시사한다.



### LoRA Training Provably Converges to a Low-Rank Global Minimum or It Fails Loudly (But it Probably Won't Fail) (https://arxiv.org/abs/2502.09376)
- **What's New**: 이번 논문에서는 Low-rank Adaptation (LoRA)의 이론적 이해를 높이는 목적을 두고 LoRA의 손실 경관을 분석합니다. 이전의 분석은 주로 선형화(linearization) 가정이나 단순화된 설정에 의존하였으나, 본 연구는 보다 현실적인 조건에서의 LoRA 훈련을 다룹니다. 논문은 두 가지 영역, 즉 '특별 레짐(special regime)'과 '일반 레짐(generic regime)'을 정의하고 차별화된 분석을 제공합니다.

- **Technical Details**: 연구 결과, 일반 레짐에서는 LoRA 훈련이 낮은 랭크(low rank)와 작은 크기(small magnitude)의 글로벌 최소값(global minimizer)으로 수렴하거나, 높은 랭크(high rank)와 큰 크기(large magnitude)의 qualitatively distinct solution으로 수렴할 수 있음을 보여줍니다. LoRA 훈련에서의 제로 초기화와 가중치 감소(weight decay)는 파라미터 공간의 저랭크(low-rank), 스몰 매그니튜드(small-magnitude) 영역 쪽으로의 암묵적인 편향을 유도하여 대부분의 경우 LoRA 훈련이 글로벌 최소값을 찾는 이유를 설명합니다.

- **Performance Highlights**: LoRA의 훈련 방식이 전반적으로 효과적이며, 특정한 조건 하에서 더 나은 성능을 내는 것으로 해석될 수 있습니다. 각기 다른 파라미터 설정에 따른 성과의 차별화가 명확히 이루어짐을 확인하였으며, 이는 LoRA의 활용 가능성과 효율성을 한층 더 넓게 확장할 수 있는 기초를 제공합니다. 이 연구는 향후 대규모 모델 튜닝(fine-tuning)에서 LoRA의 응용 가능성을 제고할 것으로 기대됩니다.



### Mitigating multiple single-event upsets during deep neural network inference using fault-aware training (https://arxiv.org/abs/2502.09374)
Comments:
          7 pages, 4 figures, Topical Workshop on Electronics for Particle Physics

- **What's New**: 이번 연구는 딥 뉴럴 네트워크(DNN)에서 발생하는 여러 단일 비트 단일 사건 에러(SBUs)의 영향을 분석하고, 하드웨어 수정 없이 결함에 대한 강건성을 개선하는 '결함 인지 훈련(fault-aware training, FAT) 방법론을 제안합니다. 이 FAT 방법론을 통해 DNN은 결함을 인지하고 이를 반영하여 더욱 견고한 모델로 훈련될 수 있습니다. 실험 결과, FAT 방법론을 통해 DNN의 결함 허용 수준(robustness)은 최대 3배 향상되었습니다.

- **Technical Details**: 연구진은 PyTorch 딥 러닝 라이브러리를 기반으로 하는 결함 주입(fault injection, FI) 도구를 개발했습니다. 이 도구는 DNN 추론 과정에서 발생할 수 있는 단일 비트 에러를 시뮬레이션하여 DNN의 데이터 경로에서 결함 발생 가능성을 분석합니다. 또한 두 개의 양자화된 DNN 모델을 사용하여 권장된 FI 실험을 수행하였으며, 각 모델의 구조적인 차이로 인한 결함 민감성을 비교 분석했습니다.

- **Performance Highlights**: 실험 결과, DNN의 결함 주입 수가 늘어남에 따라 정확도는 처음에는 감소하지만, 일정 수치 이후로는 10%에서 머무르는 경향을 보였습니다. 이는 고전적인 분류 문제에서 무작위 추측(random guessing)에 해당하는 수치입니다. 또한, DNN 내의 32비트 모듈은 8비트 모듈보다 결함에 더욱 민감하였고, 이는 높은 동적 범위(dynamic range)로 인해 주요 비트(MSBs)가 사용되지 않을 때 발생하는 결함의 영향 때문입니다.



### Language Agents as Digital Representatives in Collective Decision-Making (https://arxiv.org/abs/2502.09369)
- **What's New**: 이번 연구는 집단 의사결정 과정에서 개인의 선호를 대변하는 언어 에이전트를 훈련시키는 가능성을 탐구하고 있습니다. 이를 통해 우리는 디지털 대리 역활의 구현을 위한 새로운 접근 방식을 제안합니다. 기존의 인간 행동 시뮬레이션 연구와는 달리, 우리는 대표성을 위한 시뮬레이션에 중점을 두고 있습니다.

- **Technical Details**: 연구는 집단 의사결정을 에피소드 방식의 상호작용 과정으로 형식화하고, 디지털 대리 문제를 정의합니다. 각 참가자는 집단 결정에 대한 선호를 표현하며, 언어 공간에서의 작용을 처리합니다. 여기서 언어 에이전트는 인간 대리인으로서 적절히 선호를 표현하도록 훈련됩니다.

- **Performance Highlights**: 실험 사례 연구를 통해 다양한 인간 집단에서 합의 찾기 작업의 구현 가능성을 확인했습니다. 대규모 언어 모델을 미세조정(fine-tuning)하여 디지털 대리인으로 작용할 수 있음을 보여주었습니다. 이 연구는 집단 상호작용의 개인화된 시뮬레이션과 메커니즘 설계에 실질적 응용 가능한 결과를 도출합니다.



### Simple Path Structural Encoding for Graph Transformers (https://arxiv.org/abs/2502.09365)
- **What's New**: 이 논문은 Simple Path Structural Encoding (SPSE)이라는 새로운 방법을 제안하여, 그래프 변환기(graph transformers)에 사용되는 엣지 인코딩(edge encoding)에서의 제한 사항을 극복하고자 합니다. SPSE는 간단한 경로의 수를 활용하여 더 풍부한 구조적 정보를 인코딩합니다. 이는 무작위 경로 구조 인코딩(random walk structural encoding, RWSE)의 한계를 극복하는 것이며, 특히 지역 사이클 패턴(local cyclic patterns)을 더 잘 포착할 수 있도록 합니다.

- **Technical Details**: SPSE는 노드 쌍 간의 다양한 길이의 간단한 경로 수를 계산하여 그래프 구조를 인코딩합니다. 이를 위해 효율적인 알고리즘을 제안하는데, 이는 깊이 우선 탐색(DFS)와 너비 우선 탐색(BFS)을 사용하여 생성된 DAG 분해를 기반으로 합니다. 이 접근법은 경로 열거(path enumeration)의 지수 메모리 비용을 피할 수 있으며, 긴 경로 길이에 대한 확장성을 가능하게 합니다.

- **Performance Highlights**: SPSE는 여러 벤치마크에서 RWSE보다 일관되게 더 나은 성능을 보여주었습니다. 특히 분자(molecular) 데이터셋과 긴 범위 그래프(long-range graph) 데이터셋에서 통계적으로 유의미한 개선을 달성하였습니다. 이러한 결과는 SPSE가 그래프 변환기의 표현력을 향상시키기 위한 강력한 엣지 인코딩 대안이 될 수 있음을 나타냅니다.



### The Accuracy Cost of Weakness: A Theoretical Analysis of Fixed-Segment Weak Labeling for Events in Tim (https://arxiv.org/abs/2502.09363)
Comments:
          Submitted to TMLR

- **What's New**: 이 논문은 강력한 머신 러닝 모델을 위해 정확한 레이블의 중요성을 강조하고 있습니다. 특히, 오디오 데이터의 경우 고정 길이 데이터 세그먼트에 대한 레이블링 과정을 모델링하여 레이블의 정확성과 비용을 분석합니다. 연구는 고정 길이 레이블링(fixed-length labeling)과 진리 이벤트 활성화를 사용하는 오라클 방법(oracle method)을 비교하고 있습니다.

- **Technical Details**: 이 연구는 일반적인 약한 레이블링(weak labeling) 프로세스를 모델링하며, 특정 이벤트 클래스를 위한 고정 길이 데이터 세그먼트에 존재 또는 부재 레이블을 부여하는 방식에 초점을 맞춥니다. 연구에서는 세그먼트 길이가 레이블 정확도와 요구되는 주석 수에 미치는 영향을 분석하였으며, 이들 방법 간의 격차를 정량화했습니다. 이를 통해 고정 길이 레이블링 방법의 이론적 근거를 제시하고, 약한 레이블링 프로세스를 최적화하는 기초를 마련합니다.

- **Performance Highlights**: 연구 결과, 오라클 방법이 고정 길이 레이블링 방법보다 정확도와 비용 모두에서 우수하다는 것을 확인하였습니다. 이 발견은 오라클 프로세스를 모방하는 적응형 약한 레이블링 전략에 대한 이론적 정당성을 제공합니다. 또한, 시뮬레이션 연구를 통해 보다 복잡한 데이터 분포에 대한 일반화 가능성을 증명하였으며, 고정 길이 레이블링 방법의 정확도를 최댓값으로 설정하는 이론적 기초를 제공합니다.



### Wasserstein distributional adversarial training for deep neural networks (https://arxiv.org/abs/2502.09352)
Comments:
          15 pages, 4 figures

- **What's New**: 이 논문에서는 adversarial 공격에 대한 훈련 방법의 혁신적인 개선을 제안합니다. TRADES 방법을 확장하여 Wasserstein 분포적으로 강건한 최적화 문제에 대한 민감도 분석을 통해 분포적 공격 위협에 대응하는 새로운 훈련 방법을 도입합니다. 또한, 이는 기존의 훈련된 모델에 적용할 수 있는 효율적인 미세 조정 방법을 포함합니다.

- **Technical Details**: 제안된 방법은 Wasserstein 거리 기반의 분포적 위협 모델을 고려하며, 다양한 성공적인 사전 훈련 신경망을 RobustBench에서 테스트하여 효과성을 입증합니다. 새로운 훈련 방법은 기존 포인트와이즈 공격에 대한 강건성을 유지하면서 분포적 공격에 대한 강건성을 향상시킵니다. 이 연구는 기존의 신경망 훈련 기법들과 차별화된 접근 방식을 통해 진행됩니다.

- **Performance Highlights**: 실험 결과는 추가 훈련이 Wasserstein 분포적 강건성을 향상시키는 동시에 기존의 포인트와이즈 강건성을 유지할 수 있도록 도와줌을 보여줍니다. 하지만 대규모 합성 데이터셋으로 사전 훈련된 모델의 경우 개선 효과가 덜 두드러지게 나타났습니다. 그럼에도 불구하고 원본 훈련 데이터셋(50k 이미지) 만으로도 성능이 향상되는 경우가 있음을 확인하였습니다.



### Machine learning for modelling unstructured grid data in computational physics: a review (https://arxiv.org/abs/2502.09346)
- **What's New**: 본 논문은 비구조 그리드 데이터에 대한 기계 학습(ML) 기술의 응용을 종합적으로 검토하는 최초의 문헌입니다. 특히 딥 뉴럴 네트워크와 같은 최신 기계 학습 기술이 비구조적 또는 적응형 그리드에서의 데이터 희소성과 불규칙성 문제를 해결하는데 어떻게 기여할 수 있는지를 강조합니다. 또한, 이 논문은 환경 시뮬레이션 및 복합 기하학이 포함된 다상 흐름 모델링과 같은 다양한 중대한 응용 분야를 다루고 있습니다.

- **Technical Details**: 비구조 그리드 데이터는 비정형 경계와 동적 문제 처리를 위한 유연성을 제공하여 복잡한 물리 시스템을 정확하게 나타내는 데 적합합니다. 여러 가지 기계 학습 방법이 논의되며, 특히 CNN과 RNN과 같은 심층 신경망이 다이나믹 시스템 처리에 주안점을 두고 있습니다. 또한, Reduced-order modelling 및 spatial interpolation과 같은 전략을 통해 비구조적 데이터를 처리하기 위한 고급 기술이 포함되어 있습니다.

- **Performance Highlights**: 많은 연구에서 비구조 메쉬를 활용한 기계 학습 방법들이 효과적으로 실제 물리 시스템의 다양한 및 변화하는 조건을 모델링하는 데 기여하고 있음을 보여주고 있습니다. 특히, 그래프 신경망(GNN)과 변환기 기반 모델들이 비구조적 데이터에서 탁월한 성능을 발휘하여 복잡한 물리 시스템 시뮬레이션을 더욱 향상시키고 있습니다. 이 외에도 물리 기반 신경망(PINNs)과 강화 학습(RL) 이러한 Advances in ML techniques are revolutionising the way unstructured data is processed, making simulations of complex systems more accurate, efficient, and flexible.



### Neural Spatiotemporal Point Processes: Trends and Challenges (https://arxiv.org/abs/2502.09341)
- **What's New**: 이 논문은 공간 시간 점 과정(Spatiotemporal Point Processes, STPPs)의 기존 모델들을 검토하고, 최신 딥 러닝 기술을 접목하여 이들을 통해 복잡한 사건 데이터를 좀 더 효과적으로 모델링하는 방법을 소개합니다. 특히, STPPs와 신경망(neural networks) 기술의 통합이 활발한 연구 영역으로 발전하고 있음을 강조합니다. 이 리뷰는 STPPs의 핵심 모델, 응용 분야, 그리고 이벤트 모델링의 주요 구성 요소를 다룹니다.

- **Technical Details**: STPPs는 연속적인 공간과 시간에서 발생하는 무작위 사건의 시퀀스를 모델링하는 데 중점을 둡니다. 특히, 시간의 방향성(unidirectional)과 공간의 다방향성(omnidirectional)으로 인해 모델링이 어려워지며, 전통적인 방법은 강력한 모수(parametric) 가정과 독립성에 의존하여 유연성이 부족합니다. 이러한 한계를 극복하기 위해 신경망 기반의 방법들이 사건 간의 의존성을 처리하고, 이종 패턴을 학습할 수 있도록 설계되었습니다.

- **Performance Highlights**: 신경 STPP는 사건의 복잡한 관계를 포착함으로써 사건의 자기 자극(self-excitation) 및 사건과 공간 변인 간의 상호작용을 효과적으로 모델링할 수 있습니다. 이러한 모델들은 다양한 데이터 유형을 통합하고, 자동으로 특징(feature)을 추출하며, 대규모 환경에서도 효과적으로 확장할 수 있습니다. 이 리뷰는 특히 신경망 기반의 STPPs의 알고리즘과 방법론적 혁신을 체계적으로 탐구하고 있으며, 연구자들에게 실용적인 기초를 제공합니다.



### This looks like what? Challenges and Future Research Directions for Part-Prototype Models (https://arxiv.org/abs/2502.09340)
- **What's New**: 이번 연구는 eXplainable Artificial Intelligence (XAI) 분야에서 부각되고 있는 Part-Prototype Models (PPMs)에 대한 광범위한 분석을 제공합니다. PPMs는 입력 이미지와 학습된 프로토타입을 비교하여 결정하는 방식으로, 사람도 이해할 수 있는 설명을 제공하는 특성을 가지고 있습니다. 그러나 PPMs가 현재 널리 채택되지 않는 이유와 미래 연구 방향에 대한 통찰을 제시합니다.

- **Technical Details**: PPMs는 입력 데이터를 프로토타입과 비교하여 결정을 내리며, 이를 위해 ProtoPNet이라는 신경망 구조를 사용합니다. ProtoPNet은 합성곱 신경망(CNN)을 사용해 입력을 인코딩하고, 프로토타입과의 일치를 위한 프로토타입 레이어를 활용합니다. 이 모델은 각 프로토타입과의 유사성을 기반으로 최종 결정을 내리며, 이는 모델의 해석 가능성을 높입니다.

- **Performance Highlights**: 연구에서는 PPMs가 직면한 여러 가지 도전과제를 분석합니다. 특히 프로토타입의 질과 양, 다양한 과업 및 맥락으로의 일반화 부족, 평가의 비표준화 등의 문제가 있습니다. 이러한 문제를 극복하기 위한 다섯 가지 연구 방향을 제안하며, 이는 PPMs의 성공적인 응용을 위한 중요한 기초가 될 것입니다.



### Graph Diffusion Network for Drug-Gene Prediction (https://arxiv.org/abs/2502.09335)
Comments:
          IEEE/ACM TCBB. 14 pages

- **What's New**: 이번 논문에서는 약물-유전자 예측을 위한 그래프 확산 네트워크(Graph Diffusion Network for Drug-Gene Prediction, GDNDGP)를 도입합니다. 기존의 그래프 신경망(GNN) 방식의 한계를 극복하고자 메타 경로 기반 동질 그래프 학습과 병렬 확산 네트워크를 통해 데이터 희소성(data sparsity) 문제를 해결하였습니다. 이를 통해 훈련 과정에서 효과적인 하드 네거티브 샘플을 생성하여 예측 정확도를 향상시킵니다. GDNDGP는 DGIdb 4.0 데이터셋에서 우수한 성능을 보여주며, 약물-유전자-질병 네트워크에 대한 강력한 일반화 능력을 입증하였습니다.

- **Technical Details**: 제안된 GDNDGP 모델은 메타 경로를 이용한 동질 노드 간 정보 교환을 촉진합니다. 이 모델은 약물 간 및 유전자 간의 관계를 효과적으로 캡처하여 정확한 상호작용 예측을 가능하게 합니다. 또한, 그래프 확산 네트워크를 통합하여 훈련 중 하드 네거티브 샘플을 생성하고, 대량의 연결되지 않은 쌍을 찾을 필요를 없애면서 훈련 효율성을 높입니다. 이러한 접근은 때문에 모델의 판별력을 강화하고 일반화 가능성을 높입니다.

- **Performance Highlights**: 실험 결과, GDNDGP는 기존의 방법들에 비해 약물-유전자 예측 작업에서 유의미한 성능 향상을 보여주었습니다. 특히 복잡한 이질적 관계를 처리하는 능력에서 두각을 나타내었습니다. 이 모델은 약물의 유전자 네트워크와의 관계를 더욱 정확하게 예측하는 데 기여하며, 개인 맞춤형 의약품 개발 및 재창출 가능성에 큰 도움이 될 수 있습니다. 공개된 소스 코드는 관련 연구자들에게 효과적인 도구로 작용할 것입니다.



### Full Swap Regret and Discretized Calibration (https://arxiv.org/abs/2502.09332)
- **What's New**: 이 논문에서는 구조적 일반형 게임에서 swap regret(스왑 후회) 최소화 문제를 연구합니다. 특히, 플레이어는 매우 큰 수의 순수 행동을 취할 수 있지만, 각 행동은 $d$-차원 공간으로 임베딩됩니다. 본 연구에서는 이를 위한 효율적인 학습 알고리즘을 제공하며, 이 알고리즘은 $T$ 라운드 후 최대 $	ilde{O}(T^{(d+1)/(d+3)})$의 swap regret을 발생시킵니다.

- **Technical Details**: 새로운 온라인 학습 문제를 소개하여 이를 \'full swap regret minimization\'이라고 명명했습니다. 이 문제에서 학습자는 제한된 볼록 $d$-차원 행동 집합 $\\mathcal{K}$에서 반복적으로 (무작위) 행동을 취하고 적으로부터 손실을 받아, worst-case swap 함수에 대한 후회를 최소화하는 것을 목표로 합니다. 우리는 볼록성과 매끄러움에 대한 다양한 가정에 대해 $O(T^{d/(d+2)})$에서 $O(T^{(d+1)/(d+2)})$까지의 full swap regret 경계를 제공하는 알고리즘을 설계했습니다.

- **Performance Highlights**: 온라인 예측 문제에 이러한 도구를 적용하여 캘리브레이션 오차를 최소화하는 방법을 제시했습니다. 여러 캘리브레이션 개념은 full swap regret의 특별한 경우로 볼 수 있으며, 특히 온라인 예측을 위한 효율적인 알고리즘을 설계하여 최대 $O(T^{1/3})$의 $\\ell_2$-캘리브레이션 오차와 $O(	ext{max}(	ext{sqrt}(	ext{ε} T), T^{1/3}))$의 discretized-calibration 오차를 보장합니다.



### Bayesian Optimization for Simultaneous Selection of Machine Learning Algorithms and Hyperparameters on Shared Latent Spac (https://arxiv.org/abs/2502.09329)
- **What's New**: 이번 연구에서는 머신러닝(ML) 알고리즘과 하이퍼파라미터 최적화 문제인 CASH (Combined Algorithm Selection and Hyper-parameter optimization) 문제에 대한 새로운 접근법을 제안합니다. 기존의 Bayesian Optimization (BO) 기법에서는 서로 다른 ML 알고리즘에 대해 독립적으로 대리 모델을 구축했지만, 본 연구는 다양한 하이퍼파라미터 공간을 공유 잠재 공간에 매핑하여 효율적으로 정보를 공유할 수 있도록 하였습니다. 특히 이 연구는 관찰값의 공유를 통해 적은 수의 관찰로도 효율적인 최적화를 가능하게 하여, 머신러닝 시스템의 성능 향상을 목표로 합니다.

- **Technical Details**: 제안된 방법론은 Bayesian Optimization (BO)을 기반으로 하며, 여러 ML 알고리즘의 하이퍼파라미터 공간을 공유 잠재 공간으로 결합합니다. 이는 다중 작업 모델(multi-task model)과 깊은 커널(deep kernel)을 활용하여 서로 다른 하이퍼파라미터 공간의 관찰 값들을 공유할 수 있게 설계되었습니다. 또한, 하이퍼파라미터 공간 구성의 안정성을 높이기 위해 적대적 정규화(adversarial regularization)를 포함한 전처리(pre-training) 프레임워크를 도입하여 다양한 과거 데이터셋에서 학습된 임베딩을 활용합니다.

- **Performance Highlights**: 실험 결과는 제안된 방법이 OpenML 데이터셋을 활용하여 기존의 BO 접근법 대비 개선된 성능을 보여준다는 것을 입증합니다. 특히, 적은 수의 관찰로도 효과적인 대리 모델을 학습 할 수 있는 점에서 의미가 있습니다. 연구는 각기 다른 머신러닝 알고리즘 간의 정보 공유를 통해, 최적화 과정에서의 효율성을 극대화함으로써 연구자들에게 실질적인 데이터 분석 도구를 제공할 것입니다.



### Depth-Bounds for Neural Networks via the Braid Arrangemen (https://arxiv.org/abs/2502.09324)
- **What's New**: 이 논문은 ReLU 네트워크에서 모든 연속 및 조각선형 함수(continuous and piecewise linear functions)를 정확하게 표현하기 위한 숨겨진 층(hidden layers)의 최소 개수에 대한 질문을 다룹니다. 저자들은 특히 braid fan과 호환되는 신경망에 집중하여, d개의 수의 최대값(maximum)을 정확하게 표현하기 위해 필요한 숨겨진 층의 수가 비상수(lower bound)인 \(\Omega(\log\log d)\)임을 증명했습니다. 또한, 5개의 수의 최대값을 계산하기 위해 3개의 숨겨진 층이 필요하다는 조합적(combinatorial) 증명을 제공합니다.

- **Technical Details**: 저자들은 ReLU 네트워크가 조각선형 함수(또는 CPWL 함수)를 정확히 표현할 수 있는지를 분석하였습니다. 이 연구는 Wang과 Sun(2005)의 결과를 바탕으로, \(\mathbb{R}^{d}\)에서 d+1개의 선형 항의 최대값을 계산함으로써 일반적인 CPWL 함수를 표현할 수 있음을 보여줍니다. 따라서 두 개의 숨겨진 층으로는 최대값을 정확히 계산할 수 없음을 입증한 이전 문헌과 상반되는 결과가 나왔습니다.

- **Performance Highlights**: 연구 결과, d개의 숫자의 최대값을 계산하기 위해 필요한 최소한의 숨겨진 층 수를 제시하며, 7개의 수의 최대값을 표현하기 위해 필요한 층 수는 그보다 적다는 점을 보여주었습니다. 이로 인해 maxout 네트워크(maxout networks)와 관련된 알려진 상한선을 일반화한 결과가 타이트하지 않음을 시사하며, 그 결과 신경망의 구조 개선을 위한 새로운 방향을 제시합니다.



### SigGate: Enhancing Recurrent Neural Networks with Signature-Based Gating Mechanisms (https://arxiv.org/abs/2502.09318)
- **What's New**: 이번 논문에서는 일체의 경로 서명(path signatures)을 게이팅 메커니즘에 통합하여 순환 신경망(recurrent neural networks, RNNs)의 성능을 향상시키는 새로운 접근법을 제안합니다. 새로운 Signature-LSTM(SigLSTM)과 Signature-GRU(SigGRU) 아키텍처는 기존의 forget 게이트와 reset 게이트를 학습 가능한 경로 서명으로 대체하여 정보 흐름을 더 효과적으로 제어합니다. 이를 통해 네트워크는 현재 입력과 상태뿐 아니라 전체 역사에 기반하여 메모리 결정을 내릴 수 있게 됩니다.

- **Technical Details**: 경로 서명은 순차적 데이터의 중요 기하학적 및 시간적 패턴을 포착하는 강력한 수학적 도구입니다. 이 아키텍처에서 경로 서명은 LSTM과 GRU의 게이트 메커니즘에 통합되어 각 입력 시퀀스의 서명을 기반으로 게이트 값을 계산합니다. 이를 통해 이 네트워크들은 장기 의존성을 효과적으로 모델링하면서도 전통적인 RNN의 계산 효율성을 유지합니다.

- **Performance Highlights**: SigLSTM과 SigGRU 모델은 다양한 순차 학습 과제에서 기존 모델들보다 뛰어난 성능을 보였습니다. 이들이 특히 복잡한 시간적 패턴과 장거리 상호작용을 이해해야 하는 작업에서 두드러진 성능 향상을 이룬 것이 특징입니다. 논문에서 제안하는 경로 서명을 활용한 접근 방식은 시계열 분석 및 예측 응용 분야에서 새로운 기회를 제공합니다.



### Towards Seamless Hierarchical Federated Learning under Intermittent Client Participation: A Stagewise Decision-Making Methodology (https://arxiv.org/abs/2502.09303)
Comments:
          20 pages, 8 figures,5 tables

- **What's New**: 이번 연구에서는 기존의 Hierarchical Federated Learning (HFL) 접근법의 한계를 극복하기 위해 클라이언트의 동적 가용성을 다루는 새로운 최적화 방안을 제시합니다. 클라이언트의 참여가 간헐적으로 이루어지는 상황에서 시스템 비용(지연 시간 및 에너지 소비)을 최소화하면서 빠른 모델 수렴을 목표로 합니다. 이를 위해 클라이언트 선택과 클라이언트-엣지 연결 최적화에 대해 구체적으로 탐구합니다.

- **Technical Details**: 이 연구에서는 복잡한 NP-hard 문제를 해결하기 위해 두 단계로 나누어진 방법론을 제안합니다. 첫 번째 단계인 Plan A는 이후 모델 훈련 라운드에 참여할 가능성이 높은 장기 클라이언트를 식별하는 데 초점을 맞춥니다. 두 번째 단계인 Plan B는 장기 클라이언트가 사용할 수 없을 때 대신 사용할 수 있는 대체 클라이언트를 선택합니다.

- **Performance Highlights**: MNIST와 CIFAR-10 데이터셋에 대한 평가를 통해 제안된 방법론이 기존 벤치마크에 비해 모델 정확성과 시스템 비용 측면에서 우수한 성능을 보임을 확인하였습니다. 이 접근법은 HFL과 기존 FL의 결합을 통해 낮은 오버헤드의 의사결정 과정을 가능하게 하여 향상된 효율성을 제공합니다.



### Convex Is Back: Solving Belief MDPs With Convexity-Informed Deep Reinforcement Learning (https://arxiv.org/abs/2502.09298)
- **What's New**: 이번 연구는 Deep Reinforcement Learning (DRL)에서 Partially Observable Markov Decision Processes (POMDPs)의 신뢰도 공간 위의 가치 함수의 볼록성(convexity)을 포함하는 새로운 방법을 제안합니다. 하드 및 소프트 볼록성 강제화 기법을 비교하고, Tiger 및 FieldVisionRockSample 문제와 같은 잘 알려진 POMDP 환경에서 성능을 검토합니다. 연구 결과, 볼록성 기능을 포함하면 에이전트의 성능이 상당히 향상되고 하이퍼파라미터 공간에 대한 강건성이 증가함을 보여주었습니다.

- **Technical Details**: Markov Decision Processes (MDP)는 순차적 의사 결정 문제를 해결하기 위한 표준 모델이고, POMDP는 환경 상태가 불완전하게 관찰될 때 확률적 의사 결정 프로세스를 모델링하는 방법입니다. 현재 대부분의 DRL 접근법은 신뢰도를 고려하지 않지만, 본 연구는 볼록성을 고려함으로써 훈련 프로세스를 개선합니다. 우리는 두 가지 접근법, 즉 소프트 및 하드 볼록성을 사용하여 Dueling Q-Network 아키텍처에 적용된 볼록성 정보 기반 DRL의 성능을 검토합니다.

- **Performance Highlights**: 실험 결과, 볼록성 강제화가 적용된 DRL은 표준 훈련 방식에 비해 일반화 성능이 더 우수하고, 일부 경우에는 훈련 성능 또한 개선됨을 증명하였습니다. 이 연구는 기존 DRL 방식에서 신뢰도 기반의 훈련이 어떻게 개선될 수 있는지를 제시하며, 특히 out-of-distribution (OOD) 도메인에서도 강력한 성능을 보이는 경향을 보였습니다. 따라서 본 연구는 POMDP를 해결하기 위한 새로운 가능성을 열어줍니다.



### When do neural networks learn world models? (https://arxiv.org/abs/2502.09297)
Comments:
          28 pages, 9 figures

- **What's New**: 이번 연구는 신경망이 인간과 유사한 세계 모델(world models)을 학습할 수 있는 가능성에 대한 이론적 결과를 제공합니다. 이전 작업들을 바탕으로, 모형이 복잡성(complexity)에 대한 저편향(low-degree bias)을 가질 경우 비선형 함수가 포함된 프록시 작업(proxy tasks)에서도 잠재 데이터 생성 변수를 복구할 수 있음을 입증하였습니다. 하지만 이러한 복구 과정은 모델 아키텍처에 민감하게 반응합니다.

- **Technical Details**: 이 논문은 Boolean 모델을 활용한 다양한 기법을 통해 세계 모델 학습을 형식화한 프레임워크를 제시합니다. 특히, Boolean 함수의 Fourier-Walsh 변환을 사용하여 저복잡도 편향(low-complexity bias) 및 모델 아키텍처와의 상호작용을 분석합니다. 또한 잠재 데이터 생성 변수의 식별 가능성을 보장하는 다중 과제(multi-task) 설정과 저복잡도 편향 개념을 탐구합니다.

- **Performance Highlights**: 실험 결과에 따르면 논문에서 제안한 방법론이 기존의 ReLU MLPs 및 transformers와 같은 전통적인 아키텍처보다 우수한 성능을 보여주었습니다. 특히 다항 외삽(polynomial extrapolation) 및 물리 법칙 학습(task learning) 등 두 가지 대표적인 작업에서 성과를 거두었습니다. 이 연구는 세계 모델 학습과 관련된 기존의 이론적인 논란에 신뢰성을 추가함과 동시에 알고리즘의 실행 가능성을 확장합니다.



### An Uncertainty Principle for Linear Recurrent Neural Networks (https://arxiv.org/abs/2502.09287)
- **What's New**: 이 논문은 선형 순환 신경망(linear recurrent neural networks)의 성능을 새로운 관점에서 조명합니다. 특히, 복사 작업(copy task)을 통해 시계열(signal modeling) 데이터 처리의 능력을 평가하며, 상대적으로 단순한 문제를 다루는데 중점을 두었습니다. 새로운 필터 설계를 통해 과거 K단계의 데이터를 반복할 수 있는 방식으로 최적의 성능을 명확히 합니다.

- **Technical Details**: 연구에서는 고전 신호 모델(classical signal models)과 이차 비용(quadratic cost)을 활용하여 모델의 근사(lower bounds)를 정량적으로 분석합니다. 필터는 입력에 대한 의존성이 없는 매개변수화 구성을 가지고 있으며, 이로 인해 GPU에서 효율적으로 처리할 수 있는 모델이 탄생했습니다. 새로운 RNN은 O(L) 복잡도를 가지며, 이는 긴 문서나 유전체 분석과 같은 긴 맥락(Long-context setting)에서도 유리합니다.

- **Performance Highlights**: 최적 필터는 K단계 과거의 값을 평균화해야 함을 보여주는 불확실성 원리를 강조합니다. 이는 K/S 비율에 따라 너비가 조정되는 즉, 더 넓은 범위를 아우르는 데이터를 확보하는 것이 중요하다는 것을 시사합니다. 또한 새롭게 제안된 RNN은 언어 모델링(language modeling) 뿐만 아니라 비전, 오디오 생성 등 다양한 영역에서도 경쟁력 있는 성과를 보이고 있습니다.



### LiSA: Leveraging Link Recommender to Attack Graph Neural Networks via Subgraph Injection (https://arxiv.org/abs/2502.09271)
- **What's New**: 이번 연구에서는 Graph Neural Networks (GNNs)에서의 새로운 공격 시나리오인 Link Recommender-Subgraph Injection Attack (LiSA)을 소개합니다. LiSA는 고립된 서브그래프를 주입하여 링크 추천 시스템과 노드 분류기를 동시에 속이는 방식으로, 공격자가 의도하지 않게 링크를 생성하게 합니다. 이를 통해 노드 분류 정확성을 저하시키고, 성공적인 공격을 가능하게 하여 실제 상황에서도 유효성을 입증하고자 합니다.

- **Technical Details**: 제안된 LiSA 프레임워크는 듀얼 서 surrogate 모델과 이급 최적화(bi-level optimization) 기법을 사용하여 서브그래프를 생성합니다. 이 과정에서 GNN의 링크 추천 시스템을 조작하여 노드 간의 연결을 유도함으로써, 악의적인 노드가 목표 노드와 링크를 형성하도록 합니다. 이러한 새로운 접근은 기존의 공격 방식과는 다른 혁신적인 방법론으로 자리잡을 것입니다.

- **Performance Highlights**: LiSA는 다양한 실제 데이터셋을 활용한 실험을 통해 그 효과성과 적용 가능성을 입증했습니다. 구체적으로, 해당 공격은 노드 분류 성능을 크게 저하시켜 GNN 시스템의 작동을 방해하는 데 성공했습니다. 이로써, GNN 기반 애플리케이션에서의 링크 추천 알고리즘의 중요성과 더불어, 악의적인 공격에 대한 방어책 마련의 필요성을 강조합니다.



### Unlocking the Potential of Classic GNNs for Graph-level Tasks: Simple Architectures Meet Excellenc (https://arxiv.org/abs/2502.09263)
- **What's New**: 본 연구에서는 GNN(그래프 신경망)의 한계를 극복하기 위한 GNN+라는 새로운 프레임워크를 제안합니다. GNN+는 여섯 가지 기술, 즉 edge feature integration, normalization, dropout, residual connections, feed-forward networks, positional encoding을 통합하여 그래프 레벨 작업에 효과적으로 대응합니다. 전통적으로 GT(그래프 변환기)가 우수하다고 여겨졌던 그래프 레벨 작업에서, GNN+를 통한 향상된 GNN들이 최고의 성능을 발휘하는 것을 입증하였습니다.

- **Technical Details**: GNN+ 프레임워크는 세 가지 전통적인 GNN인 GCN, GIN, GatedGCN을 기반으로 하여 다양한 그래프 레벨 데이터셋에서 평가되었습니다. 각 기술은 성능 최적화를 위해 조정 가능한 하이퍼파라미터 역할을 하며, 구체적으로 edge features, normalization, dropout, residual connections, feed-forward networks, positional encoding을 포함합니다. 이를 통해 GNN 내부 메커니즘에 대한 효과적인 개선을 이루어냈습니다.

- **Performance Highlights**: 공식적으로 14개의 저명한 그래프 레벨 데이터셋에서 GNN+를 사용한 개선된 GNN은 기존 GT들을 초월하는 성능을 보여주었습니다. 특히 모든 데이터셋에서 상위 3위 안에 들었으며, 8개 데이터셋에서는 1위에 오르기도 했습니다. 이 연구 결과는 GNN이 그래프 레벨 작업에서 크게 underestimated 되었음을 보여주며, GNN+ 프레임워크가 그 잠재력을 극대화한다는 것을 강조합니다.



### Bandit Multiclass List Classification (https://arxiv.org/abs/2502.09257)
- **What's New**: 이 연구는 반응적 피드백(semi-bandit feedback)을 활용한 다중 클래스 리스트 분류(multiclass list classification) 문제를 다룹니다. 기존의 연구가 단일 레이블(single-label) 설정에 초점을 맞춘 반면, 본 연구는 여러 개의 정답 레이블이 있을 수 있는 다중 레이블(multi-label) 설정으로 확장합니다. 이를 통해 배급 시스템(recommendation system)과 같은 현실 세계의 문제에 대한 해결책을 제시합니다.

- **Technical Details**: 저자들은 샘플 복잡성(sample complexity) O(poly(K/m) + sm / ε²) 로그(|H|/δ) 형태의 알고리즘을 제안하여 높은 확률로 ε-최적 가설을 반환합니다. 여기서 H는 기저 가설 클래스(hypothesis class)를 나타내며, s는 각 예제에 대한 실제 레이블의 최대 수를 기준으로 한 상한입니다. 이 알고리즘은 무작위로 생성된 데이터를 반영하여 효율적으로 계산됩니다.

- **Performance Highlights**: 결과적으로 본 알고리즘은 약간의 비용으로 예측 품질을 개선할 수 있음을 보여줍니다. 특히, s=O(1)일 때 주요 항들이 기존의 전체 정보(full-information) 비율과 일치하므로, 반응적 피드백이 실질적인 비용 없이 유용하게 이용될 수 있음을 증명합니다. 이 연구의 결과는 다소 희귀한 보상(s-sparse rewards)을 가지는 맥락적 조합 반응(CCA) 문제로 일반화되어 적용될 수 있습니다.



### AnomalyGFM: Graph Foundation Model for Zero/Few-shot Anomaly Detection (https://arxiv.org/abs/2502.09254)
Comments:
          14 pages

- **What's New**: 이 논문에서는 Abnormality Detection을 위한 AnomalyGFM이라는 새로운 그래프 기반 모델을 소개합니다. 이 모델은 제로 샷(zero-shot) 및 소수 샷(few-shot) 설정에서 효과적으로 이상 상황을 탐지할 수 있도록 설계되었습니다. 기존 모델들은 훈련 및 테스트 데이터 간의 분포 차이로 인해 일반화 문제를 겪는 반면, AnomalyGFM은 그래프 간 일반화가 가능한 기법을 채택하고 있습니다.

- **Technical Details**: AnomalyGFM은 노드 표현 잔차(node representation residuals)를 기반으로 정상(normal) 및 비정상(abnormal) 클래스의 데이터 독립적인 프로토타입을 학습합니다. 이 프로토타입은 서로 다른 그래프 간에서 노드의 이상성을 일관되게 측정할 수 있는 통일된 특성 공간(feature space)을 제공합니다. 그래프에 대한 일반화 가능성을 높이기 위해, 이 모델은 클러스터링(cluster) 및 메시지 집계를 위한 새로운 파라다임을 도입했습니다.

- **Performance Highlights**: 11개의 실세계 GAD 데이터셋에 대한 실험 결과, AnomalyGFM은 제로 샷 및 소수 샷 환경에서 기존 최첨단 방법들에 비해 월등한 성능을 보였습니다. 이 모델은 대형 그래프에 대해서도 확장 가능성을 가지며, 레이블이 적은 정상 노드들을 사용하여 성능을 더욱 개선할 수 있는 방법을 지원합니다.



### On the Importance of Embedding Norms in Self-Supervised Learning (https://arxiv.org/abs/2502.09252)
- **What's New**: 이번 논문은 Self-supervised Learning (SSL)에서 임베딩 노드(embedding norm)의 중요성을 체계적으로 정립하고 연구의 빈칸을 메웠습니다. 대부분의 SSL 방법이 임베딩 벡터의 코사인 유사성(cosine similarity)을 사용하여 데이터를 하이퍼스피어(hypersphere)에 임베딩하지만, 본 논문은 임베딩 노드가 네트워크의 수렴(convergence)과 신뢰(confidence)에 관한 속성과 관련이 있음을 제시합니다. 특히, 임베딩 노드가 SSL 수렴 속도를 조절하고 네트워크의 신뢰도를 암호화하는 방식을 이론적 분석과 실험을 통해 규명하였습니다.

- **Technical Details**: 논문에서는 SSL 훈련 다이나믹스(training dynamics)와 임베딩 노드 간의 상호작용을 분석합니다. 이론적 한계를 통해 임베딩 노드가 SSL 수렴에 대해 2차적인 지연을 초래하며, 코사인 유사성을 최적화할 때 이들 노드가 성장하는 현상을 실험적으로 검증하였습니다. 적절한 임베딩 노드의 관리가 SSL 모델 훈련에 필수적임을 보여주며, 이러한 노드의 크기는 관찰된 잠재적 특징의 빈도(frequency)를 반영한다고 주장합니다.

- **Performance Highlights**: 임베딩 노드의 크기를 조작함으로써 수렴 속도에 중요한 영향을 미칠 수 있는 것을 보였습니다. 작은 임베딩 노드가 SSL 모델의 훈련에 필요하지만 훈련 과정에서 이러한 노드가 커지는 문제가 발생한다는 점을 강조합니다. 또한, 이들 노드가 불균형 데이터셋에서 대칭적이지 않게 샘플을 학습하게 하는 효과를 보여주며, 이는 향후 연구 방향성에 대한 의문을 제기합니다.



### You Do Not Fully Utilize Transformer's Representation Capacity (https://arxiv.org/abs/2502.09245)
- **What's New**: 이번 논문에서는 Layer-Integrated Memory (LIMe)를 소개하며, standard Transformer 모델의 제한된 representation (표현) 용량 문제를 해결하고자 합니다. LIMe는 모델이 이전 층의 hidden states (숨겨진 상태)에 접근할 수 있도록 하여 모델의 표현력을 확장합니다. 이는 결국 모델의 전반적인 메모리 사용량을 유지하면서도 성능 개선을 가져오는 간단하면서도 강력한 접근 방식입니다.

- **Technical Details**: LIMe는 masked multi-head self-attention 기법에 간단한 변화를 주어 모든 이전 층의 representation을 통합하며, 효율적인 routing (라우팅) 메커니즘을 통해 다층 특성을 통합합니다. 이 기법은 핵심적인 Transformer 구조를 유지하면서도 추가적인 오버헤드를 최소화합니다. 결론적으로, LIMe는 깊은 층에서 더 높은 entropy (엔트로피)를 유지하고, 밀접한 관련의 토큰들이 서로 더 잘 구별되도록 하는 효과를 보여 줍니다.

- **Performance Highlights**: 언어 모델링 실험을 통해 LIMe가 standard Transformer와 다양한 최첨단 수정 모델보다 일관되게 높은 성능을 보였다. LIMe는 representation collapse 현상을 효과적으로 방지하며, 과거 층에서 중요한 구문적 단서를 더 잘 통합하는 방법도 보여 주어, 향후 연구 방향에도 큰 가능성을 제시합니다. 이러한 결과는 더 깊고 강력한 Transformer 구조를 구축하는데 유망한 방향임을 나타냅니다.



### Neuro-Symbolic Contrastive Learning for Cross-domain Inferenc (https://arxiv.org/abs/2502.09213)
Comments:
          In Proceedings ICLP 2024, arXiv:2502.08453

- **What's New**: 본 연구는 자연어 추론(NLI) 작업에서의 기존의 사전 학습된 언어 모델(PLMs)과 유도 논리 프로그래밍(ILP)의 한계를 극복하기 위한 신경-기호 대조 학습(neuro-symbolic contrastive learning)을 제안합니다. 이 방법은 추상적인 논리 관계를 효율적으로 통합하여 데이터가 노이즈가 많고 희소한 상황에서도 논리적 정확성을 향상시킵니다.

- **Technical Details**: 신경-기호 대조 학습은 논리 프로그램(logic programs)과 논리 규칙(logic rules) 집합으로 데이터를 표현하여 추상적인 논리 관계를 임베딩(embedding)합니다. 이는 데이터의 높은 변동성을 갖는 텍스트 정보를 포착하면서도 유사한 논리적 관계를 가진 텍스트 정보를 구분할 수 있는 공간을 제공합니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식이 모델의 일반화(generalisation) 및 추론(reasoning) 능력을 상당히 개선시키는 것을 보여줍니다. 이는 기존의 PLMs의 한계를 뛰어넘은 성과로 볼 수 있습니다.



### Understanding High-Dimensional Bayesian Optimization (https://arxiv.org/abs/2502.09198)
Comments:
          19 pages, 20 figures

- **What's New**: 최근 논문에서는 고차원(real-world tasks) 최적화에서 간단한 Bayesian Optimization (BO) 방법이 우수한 성능을 보이며, 이는 기존의 연구 결과와 상반되는 것으로 보인다. 본 연구는 이러한 현상의 원인을 탐구하며 중요한 문제를 제시한다. 고차원 Bayesian Optimization (HDBO)에 내재된 근본적인 도전과제를 식별하고, 최근 기법들이 이를 극복하는 방법을 분석한다.

- **Technical Details**: 연구에 따르면 Gaussian process (GP) 초기화 방식에서 발생하는 gradient 소실(vanishing gradients)이 고차원 BO의 실패 원인 중 하나라고 할 수 있다. 기법들이 지역 검색(local search behaviors)을 촉진하는 것이 과제에 더 적합하다고 설명된다. Maximum likelihood estimation (MLE)을 통해 GP 길이 척도의 최적화를 이루는 단순한 방식을 제안하며, 이를 MSR(MLE Scaled with RAASP)이라 부른다.

- **Performance Highlights**: MSR 방법은 최신 HDBO 구현에서 우수한 성능을 발휘하며, 길이 척도에 대한 사전 신념을 명시할 필요가 없음을 증명한다. 고차원 문제(예: 1000차원)에서도 지역 검색 행동으로 인해 우수한 BO 성능을 보임을 입증하였다. 본 연구는 다수의 실제 응용 및 실험을 통해 이러한 발전을 뒷받침하는 증거를 제시한다.



### Generalizability through Explainability: Countering Overfitting with Counterfactual Examples (https://arxiv.org/abs/2502.09193)
- **What's New**: 이 연구에서는 머신러닝에서 일반화(generalization)와 수정 사실(counterfactual explanation) 간의 관계를 처음으로 탐구합니다. 모델을 학습하면서 발생하는 과적합(overfitting)을 제어하기 위한 새로운 정규화 항인 CF-Reg를 제안하며, 이는 각 데이터 샘플과 그에 해당하는 수정 사실 간의 여유(margin)를 보장함으로써 과적합을 방지합니다. 실험 결과, CF-Reg는 기존의 정규화 기법보다 일반적으로 더 나은 성능을 보임을 입증하였습니다.

- **Technical Details**: 본 논문은 과적합이 강하게 일어날 때, 훈련된 모델의 결정 경계(decision boundary)가 매우 복잡해지며, 이로 인해 무작위 입력 데이터에 대한 유효한 수정 사실 예제를 찾는 것이 더 쉬워진다는 점을 강조합니다. 이러한 관점에서 논문에서는 수정 사실 정규화 항을 도입하여, 모든 훈련 데이터 포인트가 그 결정 경계에 평균적으로 더 가까워지는 경향을 설명합니다. 또한 CF-Reg는 모든 미분 가능 수정 사실 예제 생성기와 호환 가능한 유연한 프레임워크로 설계되었습니다.

- **Performance Highlights**: CF-Reg를 통한 여러 데이터 세트와 모델에 대한 실험에서, 기존의 정규화 기술을 초월하는 성능 향상을 보여주었습니다. 특히, 복잡한 심층 신경망의 과적합 문제를 완화하면서도 모델의 설명 가능성과 일반화를 동시에 개선할 수 있는 가능성을 제시했습니다. 이를 통해 모델 훈련 과정에서 수정 사실 예제가 효과적으로 활용될 수 있음을 나타냅니다.



### Two-Stage Representation Learning for Analyzing Movement Behavior Dynamics in People Living with Dementia (https://arxiv.org/abs/2502.09173)
Comments:
          AAAI 2025 Workshop on Large Language Models and Generative AI for Health

- **What's New**: 이번 연구에서는 고주파 데이터에서 중요한 환자 행동 패턴을 드러내는 원격 의료 모니터링을 위한 시간 시계열 표현 학습을 제안합니다. 특히, 치매 환자들의 홈 활동 데이터를 분석하기 위해 두 단계의 자가 지도 학습 접근 방식을 채택하였으며, 이를 통해 저차원 구조를 발견하도록 구성되었습니다. 이 방법은 PageRank 기반 방법을 사용하여 복잡한 행동 데이터를 간결하게 압축하여 해석 가능성을 향상합니다.

- **Technical Details**: 연구에서는 원시 시계열 데이터를 전처리한 후 사전 훈련된 언어 모델을 활용하여 텍스트 시퀀스로 전환합니다. 이 과정은 시간 데이터의 특성을 포착하는 데 중점을 두며, PageRank 알고리즘을 통해 저차원 상태 벡터를 추출하고 상태 간 전이 패턴을 분석합니다. 최종적으로 이 분석 프레임워크는 환자의 행동 역학에 대한 통찰을 제공합니다.

- **Performance Highlights**: 연구 결과, 제안된 프레임워크는 인지 상태 예측과 맞춤형 치료 개입 및 대규모 건강 모니터링을 지원하는 잠재력을 보여주었습니다. 연구에서 발견된 행동 패턴은 MMSE 및 ADAS-COG와 같은 임상 지표와의 상관관계를 밝혀, 환자 치료의 질적 향상을 기대할 수 있습니다.



### LOB-Bench: Benchmarking Generative AI for Finance - an Application to Limit Order Book Data (https://arxiv.org/abs/2502.09172)
- **What's New**: 본 논문에서는 금융 데이터의 생성 및 평가 방법론을 위해 새로운 벤치마크인 LOB-Bench를 제시합니다. 이 벤치마크는 limit order book (LOB) 데이터의 품질 및 현실성을 평가하는 데 중점을 두고 있으며, Python으로 구현되었습니다. 기존의 모델 평가 방법이 정성적이었던 반면, LOB-Bench는 생성된 데이터와 실제 LOB 데이터 간의 통계적 차이를 수량적으로 측정합니다.

- **Technical Details**: LOB-Bench는 고차원 시계열 LOB 데이터를 1차원 서브스페이스 집합으로 매핑하는 집계 함수 집합을 사용하여 제너레이티브 모델의 유사성을 평가합니다. 평가 과정은 생성된 데이터와 실제 데이터 간의 분포를 비교하는 히스토그램 계산으로 구성되며, L1과 같은 거리 메트릭을 활용해 차이를 평가합니다. 또한, 조건부 분포를 평가하기 위해 데이터를 '버킷'으로 그룹화하고 각 버킷의 평균 손실을 계산하여 단일 메트릭을 도출합니다.

- **Performance Highlights**: 본 연구에서는 네 가지 현대적인 GenAI 모델과 전통적인 모델을 LOB-Bench를 통해 평가하였습니다. 실험 결과, autoregressive GenAI 접근 방식이 전통적인 모델 클래스보다 뛰어난 성능을 나타냈습니다. LOBS5 모델은 경제학 및 금융 문헌에서 잘 알려진 가격 영향 곡선을 재현하는 데 가장 우수한 성과를 보였습니다. 연구진은 LOB-Bench의 개방성과 접근성을 통해 금융 분야의 GenAI 모델 평가를 위한 중요한 출발점을 제공하고자 합니다.



### Vertical Federated Continual Learning via Evolving Prototype Knowledg (https://arxiv.org/abs/2502.09152)
- **What's New**: 이 논문에서는 Vertical Federated Learning (VFL) 프레임워크에서 클래스 및 피처 연속 학습의 도전 과제를 해결하기 위한 새로운 방법인 Vertical Federated Continual Learning via Evolving Prototype Knowledge (V-LETO)를 제안합니다. V-LETO는 과거 작업 지식의 손실을 줄이기 위해 프로토타입의 진화를 통해 지식을 전달하는 방법을 크게 개선합니다. 이 방법은 기존의 VFL 접근 방식이 다루지 않았던 문제를 해결하여 더 나은 성능을 보장합니다.

- **Technical Details**: V-LETO는 세 가지 주요 모듈로 구성됩니다: 프로토타입 생성 모듈은 이전 작업 지식을 보존하도록 설계되며, 서버는 수동 파티의 로컬 임베딩과 레이블을 기반으로 클래스 프로토타입을 만듭니다. 또한, 프로토타입 진화 모듈은 이전 및 현재 작업 지식을 통합하여 캐타스트로픽 포겟팅을 완화하고, 모델 최적화 모듈은 글로벌 및 로컬 모델을 최적화합니다. 이 구조는 VFCL을 위한 새로운 틀을 제공하여 클래스 및 피처 증강을 달성합니다.

- **Performance Highlights**: 다양한 데이터셋에서 실시한 포괄적인 실험 결과, V-LETO는 기타 최신 방법들보다 우수한 성능을 발휘했습니다. 특히, V-LETO는 CIL 및 FIL 작업에서 각각 10.39%와 35.15% 향상을 기록하며 유의미한 성과를 보여주었습니다. 이는 기존의 방법들과 비교하여 뛰어난 결과로, 이 논문에서 제안한 방법의 실용성을 입증합니다.



### Regularization can make diffusion models more efficien (https://arxiv.org/abs/2502.09151)
- **What's New**: 본 연구는 생성 AI의 핵심 아키텍처인 diffusion 모델의 효율성을 개선하기 위한 접근법으로 sparsity(희소성) 개념을 도입하고 있습니다. 기존의 고비용 문제를 해결하기 위한 방법으로 수학적 보장으로 희소성이 입력 차원의 영향을 줄일 수 있음을 입증했습니다. 이러한 접근방식은 데이터의 내재적 차원으로 계산 복잡성을 축소할 수 있는 가능성을 제시합니다.

- **Technical Details**: Diffusion 모델은 확률적 생성 모델로서, 훈련된 데이터와 유사한 새로운 데이터를 생성하는 시스템입니다. 이 모델은 데이터 샘플이 순차적으로 순환하면서 점차 노이즈로 변하는 두 가지 확률적 과정을 기반으로 동작합니다. 최종적으로는 pure noise 상태인 𝒙T로 이행하는데, 이러한 과정은 데이터 분포와 관련된 확률적 기법을 통해 이루어집니다.

- **Performance Highlights**: 연구 결과, 희소성을 유도함으로써 더 낮은 비용으로 개선된 샘플을 생성할 수 있음을 확인하였습니다. 이 방법론은 기존의 diffusion 모델에 비해 더 효율적인 샘플링을 제공할 수 있는 잠재력을 가지고 있으며, 다양한 생성 작업에서의 활용 가능성을 보여줍니다.



### Shortcut Learning Susceptibility in Vision Classifiers (https://arxiv.org/abs/2502.09150)
- **What's New**: 이 논문에서는 머신 러닝 모델이 데이터에서 의미 있는 특성을 포착하는 대신, 단기적 학습(shortcut learning) 즉, 훈련 데이터의 비생산적인 상관관계를 이용하는 방식에 대해 다룹니다. 연구팀은 다양한 비전 분류 아키텍처(CNN, MLP, ViT)를 시스템atic 하게 평가하여, 모델이 인위적으로 추가된 단기적 단서(shortcuts)를 사용하거나 실제 구별 가능한 특징을 학습하는지를 분석합니다. 또한, 수정된 데이터 세트를 통해 아키텍처마다 단기적 학습에 대한 취약성을 비교할 수 있는 방법론을 제시합니다.

- **Technical Details**: 이 연구에서는 특정 픽셀 영역을 클래스 레이블과 결정적으로 상관되도록 수정하여 인위적인 단서를 주입합니다. 그런 다음 CNN, MLP 및 ViT 모델을 이 수정된 데이터 세트에서 학습시킨 후, 두 개의 테스트 세트에서 성능을 평가합니다. 하나의 테스트 세트는 기존의 단서를 포함하고, 다른 하나는 그런 단서가 없는 상황입니다. 이러한 분석을 통해 모델 각기가 단기적인 단서를 얼마나 의존하는지 정량적으로 비교합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터 세트(MNIST, Fashion-MNIST, SVHN, CIFAR-10)를 통해 모델의 단기적 학습 의존성을 평가하며, 내부의 가중치를 재구성하는 네트워크 반전 기반 기법을 사용하여 모델이 어떤 정보를 저장하는지를 분석합니다. 또한 학습률에 따라 단기적 의존성이 어떻게 변화하는지를 살펴보아, 낮은 학습률을 사용한 경우 더 미세하고 의미 있는 특징을 학습할 수 있음을 발견했습니다. 결과적으로,하는 데이터 처리 방식과 단기적 학습에 대한 취약성을 이해할 수 있습니다.



### Replay-free Online Continual Learning with Self-Supervised MultiPatches (https://arxiv.org/abs/2502.09140)
Comments:
          Accepted at ESANN 2025

- **What's New**: 이번 논문에서는 Continual MultiPatches (CMP)를 제안합니다. CMP는 온라인 연속 학습(Online Continual Learning, OCL) 상황에서 재생(replay) 샘플을 사용하지 않고도 효과적으로 자가 지도 학습(self-supervised learning, SSL) 방식을 진행할 수 있도록 설계된 플러그인 방법입니다. 이 방법은 단일 예제에서 여러 패치를 생성하고 이를 공유된 특성 공간(feature space)으로 투영하여, 동일한 예제에 대한 패치들이 결합되지 않고 서로 밀착되도록 합니다.

- **Technical Details**: CMP는 기존의 인스턴스 구별(instance discrimination) SSL 전략의 상위 레이어에서 적용되며, 각 패치에 대해 엔코더 네트워크를 사용하여 잠재 표현(latent representation)을 계산합니다. CMP의 손실 함수는 평균 패치 표현에 대한 유사성을 유지하도록 설계되었으며, 총 코딩 비율(Total Coding Rate, TCR) 손실도 활용하여 잠재 표현이 하나의 점으로 붕괴되지 않도록 합니다. 이 방식은 메모리 버퍼 없이도 패치의 수를 늘려 미니배치(minibatch) 크기를 확장합니다.

- **Performance Highlights**: CMP는 도전적인 클래스 증가(class-incremental) OCL 벤치마크에서 실험을 통해 재생 기반 전략과 비교하여 성능이 우수함을 입증하였습니다. CMP는 제한된 계산 예산을 가지고도 경쟁력 있는 OCL 방법론과 비교했을 때도 뛰어난 성능을 발휘하였습니다. 이는 CMP가 재생 없이도 효과적인 학습을 가능하게 함을 보여줍니다.



### Trust Me, I Know the Way: Predictive Uncertainty in the Presence of Shortcut Learning (https://arxiv.org/abs/2502.09137)
Comments:
          Preprint. Under review

- **What's New**: 이 논문은 신경망에서 예측 불확실성을 정량화하는 방식에 대한 논쟁을 다루고 있습니다. 특히, 최신의 엔트로피 분해 기술이 모델 불확실성(model uncertainty)과 인식적 불확실성(epistemic uncertainty, EU)을 효과적으로 표현하는지에 대한 의문을 제기하고 있습니다. 저자들은 이러한 불확실성의 두 가지 관점이 모두 유효하다고 주장하며, 서로 다른 학습 상황에서 나타난다고 설명합니다.

- **Technical Details**: 모델의 예측 불확실성을 이해하고 이를 향상시키기 위해 불확실성을 알레아토릭 불확실성(aletorical uncertainty, AU)과 EU로 분해하는 방식이 논의됩니다. AU는 데이터의 특징이 예측에 필요하지 않은 요소로 인해 발생하며, EU는 오히려 모델의 근사 품질에 관련됩니다. 연구에서는 단축 학습(shortcut learning) 존재가 EU가 불일치(disagreement)로 나타나는 데 결정적이라는 것을 보입니다.

- **Performance Highlights**: 이 논문은 단축 학습이 데이터에 존재할 때 EU가 어떻게 나타나는지를 실험적으로 증명하여 학습 역학에 대한 중요한 통찰을 제공합니다. 현재의 UQ 방법들은 여전히 해결책을 찾지 못하고 있으며, 신뢰할 수 있는 불확실성 보고가 필요한 안전-critical 응용 분야에서의 필요성을 강조합니다. 이 이야기는 불확실성 기반 작업, 즉 능동 학습(active learning) 및 분포 외 탐지(out-of-distribution detection)와 같은 영역에서 명확한 이해가 필요하다는 점을 강조합니다.



### Interpreting and Steering Protein Language Models through Sparse Autoencoders (https://arxiv.org/abs/2502.09135)
Comments:
          11 pages, 6 figures

- **What's New**: 이 논문은 sparse autoencoders (SAE)를 사용하여 ESM-2 8M 파라미터 모델의 내부 표현을 해석하는 방법을 탐구합니다. 이를 통해 단백질 특징과 관련된 잠재 구성요소를 분석하고, zink finger domain과 같은 목표에 맞게 모델을 조정할 수 있는 방안을 제시합니다. 이 연구는 생물학적 시퀀스 모델에서의 기계적 해석 가능성(mechanistic interpretability)의 새로운 관점을 제공합니다.

- **Technical Details**: 논문에서는 Masked Language Modeling (MLM) 작업을 통해 단백질 시퀀스 모델을 학습하고, 신경망이 특정 특징을 학습하는 방식을 설명합니다. sparse autoencoders는 단순한 단일 레이어 모델로, 큰 언어 모델의 활성화를 기반으로 훈련되어 과도한 기저(overcomplete basis)를 생성하고, 이는 보다 해석 가능한 잠재 표현을 제공합니다. 이렇게 훈련된 SAE는 생물학적 정보를 명확히 해석할 수 있는 방법론을 제공합니다.

- **Performance Highlights**: ESM-2 모델을 기반으로 한 이 방법은 잠재 구성을 통해 단백질 시퀀스 생성에 성공적으로 활용됩니다. 특정 잠재 변수에 개입하여 목표한 단백질 특징을 가진 시퀀스를 생성하는 것이 가능하다는 것을 보여주었습니다. 이 연구는 단백질 모델의 디버깅 및 안전성을 고려한 편집을 지원하는 데 기여할 것입니다.



### Finite-Time Analysis of Discrete-Time Stochastic Interpolants (https://arxiv.org/abs/2502.09130)
- **What's New**: 이번 연구는 확률론적 인터폴란트(stochastic interpolant) 프레임워크의 첫 번째 이산 시간(discrete-time) 분석을 제시합니다. 이전의 분석이 연속 시간(continuous-time) 설정에 집중되었던 것에 반해, 우리는 이산 시간 샘플러(discrete-time sampler)를 도입하고 그 분포 추정 오류(distribution estimation error)에 대한 유한 시간 상한(finite-time upper bound)을 도출했습니다. 이는 전체적인 수렴 속도(convergence rate)를 이해하고 효율적인 수렴 가속화(schedules for convergence acceleration)를 설계하는 데 중요한 기초를 제공합니다.

- **Technical Details**: 연구의 핵심은 주어진 두 개의 확률 분포 사이의 변환을 위해 이산 샘플러를 사용하는 것입니다. 이 방법은 Euler–Maruyama 기법을 사용하여 이산 샘플러를 설계하고, 점진적인 KL 오류(Kullback-Leibler error)에 대한 일반적인 오류 경계를 제공함으로써 성과를 보입니다. 우리의 분석에서는 조건부 기대값의 도함수와 관련된 성분으로 이산화 오차를 분해하는 것이 특징적입니다.

- **Performance Highlights**: 이산 시간 샘플러의 성능을 검증하기 위해, 우리는 제안된 일정을 사용하여 균일(step size) 사용 시와 비교하였습니다. 결과적으로 제안된 스케줄이 이론적 발견을 지지하며, 추가적인 정규성 조건이 없을 때 더 뛰어난 성능을 나타냄을 확인했습니다. 이러한 실험 결과는 이산 시간 샘플러의 적용 가능성을 실질적으로 보여줍니다.



### Improving Deep Regression with Tightness (https://arxiv.org/abs/2502.09122)
Comments:
          ICLR 2025, Code: this https URL

- **What's New**: 이 논문은 딥 회귀에서 타겟의 순서를 보존함으로써 다양한 과제의 성능을 향상시킬 수 있음을 보여줍니다. 그러나 순서 보존이 성능에 미치는 이점에 대한 이론적 설명은 부족했습니다. 본 연구에서는 조건부 엔트로피 $H(Z|Y)$를 최소화하여 표현 Z와 타겟 Y 간의 유사성을 유지하는 최적 수송 기반 규제를 소개하고, 이를 통해 회귀의 성능을 개선하는 방법을 제안합니다.

- **Technical Details**: 순서를 보존함으로써 표현 Z의 조건부 엔트로피 $H(Z|Y)$가 감소하는 것을 발견하였습니다. 일반적인 회귀 손실이 이 조건부 엔트로피를 줄이기에는 미흡하다는 것을 보여줍니다. 이를 해결하기 위해, 회귀기 목표를 중복하여 사용하는 전략과 함께 Regression Optimal Transport (ROT) Regularizer를 도입하여 표현의 안정성을 높입니다.

- **Performance Highlights**: 세 가지 실제 회귀 과제에서 제안한 전략의 효과를 검증하였습니다. 다중 목표 접근법과 ROT-Reg가 각각 전역 및 지역적으로 표현을 조정하여 성능을 극대화함을 확인했습니다. 이 연구는 회귀 표현의 순서 보존과 관련된 기여로, 회귀 작업을 분류 문제로 재정의할 수 있는 통찰력을 제공합니다.



### Scaling Law for Stochastic Gradient Descent in Quadratically Parameterized Linear Regression (https://arxiv.org/abs/2502.09106)
- **What's New**: 이 논문은 머신러닝의 스케일링 법칙(scaling law)에 대한 연구를 다루며, 특히 사각형 파라미터화된 모델을 사용하여 선형 회귀 문제에서의 성과를 분석합니다. 선형 모델 연구의 한계를 극복하며 신경망의 특징 학습(feature learning) 과정을 실험적으로 탐구하는 데 초점을 맞추고 있습니다. 이 연구는 무한 차원의 데이터와 기울기를 가진 진실 신호에 대한 특별한 해석을 제공합니다.

- **Technical Details**: 연구에서는 Stochastic Gradient Descent (SGD)를 통해 모델 학습의 수렴 속도를 분석합니다. 특히, quadric parameterization을 통해 모델이 어떻게 현재의 지식 기반 학습 과정을 향상시키며, SGD의 성능이 비선형 네트워크에 비해 의미 있는 향상을 보여줌을 입증합니다. 기존의 선형 회귀 모델과 달리, 이 논문은 다차원 데이터와 더불어 상응하는 기하학적 조건을 고려합니다.

- **Performance Highlights**: 논문에서는 quadratically parameterized 모델이 특히 SGD와 결합하여 좋은 일반화(generalization) 성능을 발휘함에 따라, 이는 사실상 모델 크기와 샘플 크기에 대해 파워-로우 형태의 경계를 따름을 입증합니다. 연구 결과는 선형 모델에서의 일반화 곡선과 feature learning의 존재 여부 간의 명확한 분리를 보여 주며, 그 결과가 성능 향상을 이끌 수 있음을 제안합니다.



### One-shot Federated Learning Methods: A Practical Guid (https://arxiv.org/abs/2502.09104)
Comments:
          10 pages, 1 figure

- **What's New**: 이번 논문에서는 One-shot Federated Learning (OFL)의 도전 과제를 체계적으로 분석하고 현재 방법들을 심층적으로 검토합니다. OFL은 데이터 프라이버시와 통신 오버헤드를 줄이는 데 초점을 맞추며, 특히 데이터 이질성과 모델 이질성을 해결하는 데 필요한 새로운 분류 방법을 제안합니다. 또한, OFL과 관련된 기존의 연구가 부족한 문제를 해결하고자 미래 방향성을 제시합니다.

- **Technical Details**: OFL은 클라이언트가 단 한 번 모델 파라미터를 전송하여 글로벌 모델을 업데이트하는 분산 기계 학습 패러다임입니다. 이를 통해 이전의 전통적 Federated Learning (FL)에서 발생하는 여러 라운드의 데이터 교환을 피하고, 통신 오버헤드와 보안 강화를 동시에 달성할 수 있습니다. OFL의 도전 과제는 데이터 비독립적이고 비동일 분포(non-IID) 문제와 모델 이질성으로 크게 나눌 수 있습니다.

- **Performance Highlights**: 연구에 따르면 OFL은 기존 FL 방법보다 개선된 성능을 보여주기 위한 다양한 방법을 탐색하고 있으며, 많은 연구자들이 이 분야에 관심을 갖기 시작했습니다. 다음 섹션에서는 OFL의 기술적 측면을 상세히 다루고, 다양한 기술적 아이디어의 장단점을 논의하여, 연구자들이 OFL을 실용화할 수 있도록 돕고자 합니다.



### FlowAR: une plateforme uniformisée pour la reconnaissance des activités humaines à partir de capteurs binaires (https://arxiv.org/abs/2502.09067)
Comments:
          in French language this https URL

- **What's New**: 이번 논문은 FlowAR이라는 플랫폼을 소개하며, 사람의 일상 활동 인식을 위한 시스템 개발에 중점을 두었습니다. 이 시스템은 이진 센서와 같은 센서 데이터를 사용하여 일상적인 활동을 인식하는 데 도움을 줍니다. 데이터 기반(data-driven) 접근 방식을 통해 사용자 맞춤형 인식을 제공합니다.

- **Technical Details**: FlowAR은 세 단계의 파이프라인(pipeline)으로 구성되어 있습니다: 데이터 정리(data cleaning), 세분화(segmentation), 그리고 개인화된 분류(personalized classification). 이 모듈형(modular) 구조는 다양한 방법(methods)과 데이터셋(datasets)을 실험하고, 철저한 평가(rigorous evaluations)를 수행할 수 있는 유연성을 제공합니다.

- **Performance Highlights**: 구체적인 사용 사례(use case)를 통해 FlowAR의 효과를 입증하고 있으며, 이는 다양한 활동을 정확하게 인식할 수 있는 잠재력을 보여 줍니다. 이 시스템은 연구자와 개발자들이 보다 쉽게 활동 인식 시스템을 구축할 수 있도록 지원합니다.



### Zero-shot Concept Bottleneck Models (https://arxiv.org/abs/2502.09018)
Comments:
          14 pages, 8 figures

- **What's New**: 이번 논문에서는 zero-shot concept bottleneck models (Z-CBMs)를 소개하며, 이는 학습 없는 상태에서 개념 지도와 라벨 예측을 수행할 수 있는 모델입니다. Z-CBMs는 대규모 개념 데이터베이스를 활용하여 다양한 도메인에서 입력을 설명합니다. 기존의 Concept Bottleneck Models (CBMs)과 달리, Z-CBMs는 타겟 데이터셋의 수집이나 학습 없이 이해 가능하고 개입 가능한 개념을 제공합니다.

- **Technical Details**: Z-CBMs는 concept retrieval과 concept regression 두 가지 모듈로 구성됩니다. Concept retrieval 모듈은 효율적인 교차 모달 검색 알고리즘을 통해 입력과 관련된 개념을 동적으로 찾아내며, concept regression 모듈은 중복된 개념을 피하고 상호 배타적인 개념을 선택하기 위해 희소 선형 회귀(sparse linear regression)를 사용합니다. 이러한 접근 방식은 타겟 데이터셋 없이도 개념과 라벨을 예측하는 데 필요한 모든 단계를 처리할 수 있습니다.

- **Performance Highlights**: Z-CBMs는 12개의 데이터셋에 대한 광범위한 실험을 통해 기존의 학습된 CBMs와 비슷하거나 더 나은 성능을 발휘함을 입증했습니다. 특히, Z-CBMs는 예측된 개념에 대한 인간의 개입을 통해 전반적인 신뢰성을 높일 수 있으며, 이는 개념 기반 예측의 실용성을 강조합니다. 최종적으로 Z-CBMs는 다양한 도메인에서 효과적으로 사용될 수 있는 가능성을 보여주고 있습니다.



### RoSTE: An Efficient Quantization-Aware Supervised Fine-Tuning Approach for Large Language Models (https://arxiv.org/abs/2502.09003)
Comments:
          18 pages, 6 figures

- **What's New**: 이번 연구에서는 양자화 인식 세밀 조정(quantization-aware fine-tuning) 방법을 최초로 제안하여, 세밀 조정과 양자화를 단일 훈련 단계로 통합하는 효율적인 방법을 찾았습니다. 이를 통해 가중치, 활성화 및 키-값 캐시의 낮은 비트 양자화를 효과적으로 실현할 수 있습니다. 특히, 새로운 로테이티드 스트레이트-스루-추정기(RoSTE) 알고리즘을 통해 활성화 이상치(outlier)를 줄이고 최적화된 양자화 구성을 식별하는 적응형 회전 전략을 결합하였습니다.

- **Technical Details**: RoSTE 알고리즘은 양자화 인식 세밀 조정(QA-SFT)과 회전 행렬의 적응형 선택을 결합하여 이루어지는 공동 훈련 방식을 채택합니다. 이 알고리즘은 최적화 문제를 이층(bilevel)으로 설정하여 상위 레벨에서는 가중치 행렬을 최적화하고 하위 레벨에서는 회전 행렬을 선택하는 방식으로 구성됩니다. 이론적인 분석을 통해 RoSTE 사용 시 예측 오류와 양자화 오류 간의 관계를 설명하며, 낮은 복잡성의 월시-하다마드 회전 방식을 채택하여 효율성을 극대화합니다.

- **Performance Highlights**: Pythia 및 Llama 모델에서의 실험 결과, RoSTE는 기존의 세밀 조정 후 양자화 방법에 비해 다양한 작업과 여러 LLM 아키텍처에서 일관되게 우수한 성능을 나타냈습니다. 특히, RoSTE는 4비트 양자화에서도 데이터 없는 우수한 성능을 보여주며, 메모리 소모를 줄이고 추론 대기 시간을 단축시키는 데 기여합니다. 이러한 성과는 RoSTE가 세밀 조정 과정에서 양자화 효과를 고안하여, 현업에서의 실용성을 높임을 입증합니다.



### End-to-End triplet loss based fine-tuning for network embedding in effective PII detection (https://arxiv.org/abs/2502.09002)
Comments:
          13 pages, 10 figures, 5 tables

- **What's New**: 이 논문은 개인 식별 정보(PII)의 유출을 탐지하기 위한 완전한 딥러닝(end-to-end) 프레임워크를 제안합니다. 이 프레임워크는 대형 언어 모델(Large Language Model, LLM) 및 오토인코더(autoencoder)를 사용하여 네트워크 패킷의 임베딩을 생성하고, 트리플릿 손실 기반의 파인튜닝(fine-tuning) 방법을 적용합니다. 이를 통해 실제 데이터셋에서 PII 탐지의 효과성을 높이는데 기여하고 있습니다.

- **Technical Details**: 제안된 프레임워크는 멀티레이어 퍼셉트론(Multilayer Perceptron, MLP)을 클래시파이어(classifier)로 사용하여 네트워크 흐름의 PII 유출을 감지합니다. 또한, 다중 레이블 분류 문제를 통해 유출된 PII의 종류를 식별하며, k-fold 교차 검증을 통해 모델의 유효성을 평가합니다. 연구는 ReCon과 AntShield의 두 개의 공개 데이터셋을 활용하여 성능을 비교합니다.

- **Performance Highlights**: 제안된 탐지 프레임워크는 딥러닝 기반 접근 방식을 통해 기존의 방법들과 비교했을 때 더 높은 탐지 능력을 보여줍니다. 특히 PII 유출이라는 복잡한 문제를 처리하기 위해 외부 특성 선택 없이 자동으로 패턴을 학습할 수 있는 딥러닝의 장점을 구현하였습니다. 결과적으로, 이 방법은 모바일 애플리케이션에서 개인 데이터의 유출을 효과적으로 감지하여 사용자에게 중요한 정보를 제공합니다.



### Privacy-Preserving Hybrid Ensemble Model for Network Anomaly Detection: Balancing Security and Data Protection (https://arxiv.org/abs/2502.09001)
Comments:
          Accepted by 2024 5th International Conference on Big Data, Artificial Intelligence and Internet of Things Engineering(ICBAIE 2024)

- **What's New**: 이 연구에서는 악성 활동 탐지에서의 개인정보 보호를 위해 하이브리드 앙상블 모델을 제안합니다. 이 모델은 K-Nearest Neighbors (KNN), Support Vector Machines (SVM), XGBoost, Artificial Neural Networks (ANN) 등의 머신러닝 알고리즘의 강점을 결합하여 정확한 탐지와 데이터 보호를 동시에 달성합니다. 제안하는 모델은 기존 방법보다 향상된 탐지 성능과 강력한 개인정보 보호 장치를 제공합니다.

- **Technical Details**: 제안하는 멀티 모델 프레임워크는 KNN, SVM, XGBoost 및 ANN을 통합하여 네트워크 이상 탐지 시스템을 구축합니다. 이 시스템은 이진 및 다중 클래스 분류를 지원하며, 고급 샘플링 기법과 특성 중요성 분석을 통해 성능을 극대화합니다. 또한, 개인정보 보호를 위해 연합 학습(Federated Learning), 안전한 다자간 계산(Secure Multi-Party Computation, SMPC), 차등 개인정보 보호(Differential Privacy) 기술이 적용되었습니다.

- **Performance Highlights**: 제안된 모델의 성능은 정확도, 정밀도, 재현율 및 F1 스코어를 사용하여 평가되었습니다. 이 모델은 불균형 데이터 세트와 작은 샘플 크기를 효과적으로 관리하고, 처리 과정에서 추가된 노이즈를 통해 개인 정보를 보호합니다. 최종적으로, 기존 모델보다 균형 잡힌 분류 성능과 더욱 강력한 개인정보 보호 기능을 구현하여 향후 네트워크 보안 분야에서 중요한 진전을 제공합니다.



### Task Generalization With AutoRegressive Compositional Structure: Can Learning From $\d$ Tasks Generalize to $\d^{T}$ Tasks? (https://arxiv.org/abs/2502.08991)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)이 적은 수의 데모만으로도 명시적으로 훈련받지 않은 작업을 해결하는 뛰어난 일반화 능력을 보여준다는 점을 다루고 있습니다. 저자들은 AutoRegressive Compositional (ARC) 구조의 관점에서, 적은 수의 작업에서 어떻게 많은 작업 패밀리에 일반화할 수 있는지를 분석합니다. 특히, 입력에서 생성된 각 작업이 조합적으로 형성된다는 사실에 주목하고 있습니다.

- **Technical Details**: 논문은 대형 언어 모델이 작업을 수행하는 방식, 특히 Chain-of-Thought (CoT) 추론을 도입하여 작업을 쉽게 일반화하는 과정을 설명합니다. CoT를 통해 저자들은 모델이 복잡한 작업을 해결하기 위해 더 높은 수준의 추론 단계를 도입할 수 있음을 보여줍니다. 이는 AutoRegressive 구조를 활용하여 훈련되지 않은 작업에도 일반화할 수 있음을 증명합니다.

- **Performance Highlights**: 실험 결과, CoT 추론을 포함한 Transformer 모델은 훈련 중 제외된 작업에 대해 거의 완벽하게 일반화하는 것으로 나타났습니다. 대조적으로 CoT 없이 훈련된 경우는 단순한 기회 수준 이하의 성과를 보였고, 이는 CoT가 파리티 문제의 조합 구조를 효과적으로 활용하여 더 나은 성능을 발휘하게 한다는 것을 보여줍니다.



### Neural Force Field: Learning Generalized Physical Representation from a Few Examples (https://arxiv.org/abs/2502.08987)
Comments:
          20 pages

- **What's New**: 본 논문에서는 Neural Force Field (NFF)라는 새로운 모델링 프레임워크를 제안합니다. 이 프레임워크는 Neural Ordinary Differential Equation (NODE)을 기반으로 하여, 최소한의 데이터를 통해 물리적 역학을 학습하고 일반화할 수 있는 강력한 대안을 제공합니다. 기존의 접근 방식은 고차원 잠재 공간에 의존하는 반면, NFF는 중력, 지지 및 충돌과 같은 근본적인 물리 개념을 해석 가능한 방식으로 캡처합니다.

- **Technical Details**: NFF는 외부 개입 및 객체 상호작용을 통해 동적 잠재력 장(force field)을 학습하는 신경망을 사용합니다. 예측된 힘(force)은 ODE 솔버를 통해 통합되어 속도 및 변위와 같은 명시적인 물리 변수를 계산하며, 이 결과는 확립된 물리 원칙과 일치하는 해석 가능한 결과를 생성합니다. 이 프레임워크는 적은 수의 훈련 예제에서 물리적 상호작용을 저차원 힘 장으로 표현함으로써 빠르게 학습할 수 있습니다.

- **Performance Highlights**: NFF는 I-PHYRE 및 N-body 문제와 같은 두 개의 도전적인 물리적 추론 벤치마크에서 검증되었습니다. 실험 결과, NFF는 물리적 상호작용을 힘 장으로 추상화함으로써 효과적으로 역학을 학습할 뿐만 아니라, 이전 시나리오 및 교차 시나리오 설정에서도 강력한 일반화 성능을 달성했습니다. 이러한 물리 기반 표현은 목표 지향적 작업에서의 효과적인 전방 및 후방 계획을 가능하게 하여 Interaction Network (IN) 및 transformer 기반 방법들과 비교하여 일관되게 더 나은 성능을 나타냈습니다.



### Few is More: Task-Efficient Skill-Discovery for Multi-Task Offline Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2502.08985)
- **What's New**: 이번 논문에서는 기존의 문제를 해결하기 위해 'Skill-Discovery Conservative Q-Learning'(SD-CQL)이라는 다중 작업 오프라인 MARL 알고리즘을 제안합니다. SD-CQL은 관찰을 재구성하여 스킬을 발견하며, 이를 통해 고정 및 가변 행동을 개별적으로 평가합니다. 이 과정에서 기존의 방법들과 달리, 정책을 다시 훈련할 필요 없이 여러 작업에 대한 범용성을 학습할 수 있게 합니다.

- **Technical Details**: SD-CQL 알고리즘은 행동 규제 보수적 Q-learning을 사용하여 발견된 각 스킬에 최적의 행동을 실행합니다. 이러한 접근 방식은 로컬-글로벌 정렬(local-global alignment)을 필요로 하지 않으며, 제한된 소스 작업으로부터 강력한 다중 작업 일반화를 가능하게 합니다. 논문에서는 StarCraftII에서의 실험을 통해 SD-CQL의 성능을 입증하며, 이를 바탕으로 고급 오프라인 MARL 알고리즘에서의 사용 가능성을 강조합니다.

- **Performance Highlights**: SD-CQL은 14개의 작업 세트 중 10개에서 최고의 성능을 달성하였고, 개별 작업 세트에서는 최대 65%의 개선을 보였습니다. 나머지 4개의 작업 세트에서도 최고의 기준 알고리즘과 거의 4% 이내의 차이를 나타내어 매우 우수한 성능을 입증합니다. 이 결과는 SD-CQL이 높은 작업 효율성을 가지며 많은 잠재적 작업을 다룰 수 있음을 나타냅니다.



### What exactly has TabPFN learned to do? (https://arxiv.org/abs/2502.08978)
Comments:
          Originally published in Blogposts Track at ICLR 2024. Appendix contains re-analysis on TabPFN-v2 [Hollmann et al., 2025]

- **What's New**: TabPFN은 최근 ICLR 회의에서 발표된 Transformer 모델로, 새로운 표 형 분류 문제에서 인컨텍스트 학습(in-context learning)을 수행하도록 사전 학습되었습니다. 이 모델은 구조적 인과 모델(structural causal models)에서 무작위로 생성된 데이터를 기반으로 하여 분류 작업을 수행할 수 있는 능력을 보여주며, 기존의 PFN 모델들에 비해 더욱 효율적인 메타 학습(meta-learning) 접근 방식을 제공합니다.

- **Technical Details**: TabPFN은 자기 주의(self-attention)를 사용하여 훈련 샘플 간의 관계를 학습하고, 테스트 샘플에서 훈련 샘플로의 교차 주의(cross-attention)를 활용합니다. 이 과정에서, 모델은 훈련 데이터 저장 외에는 추가적인 훈련이 필요하지 않으며, 다양한 데이터 세트에 대한 함수 근사(function approximation)를 생성할 수 있도록 설계되었습니다. 메타 학습 접근법의 원리에 기반하여, 이 모델은 훈련 데이터 없이도 높은 예측 성능을 발휘합니다.

- **Performance Highlights**: TabPFN은 1차원 이진 분류와 2차원 다중 클래스 분류에서 실험되었으며, 메타 학습에 기반한 예측 기법을 통해 성능을 발휘했습니다. 비록 기본 모델의 성능은 앙상블(ensemble)을 사용할 때 향상되었지만, 기본적으로 TabPFN은 여러 샘플이 있는 다중 클래스 문제에서 잘 작동하지 않았습니다. 향후 TabPFN의 구조 및 훈련 방식이 개선된다면, 앙상블 없이도 더 나은 성능을 기대할 수 있을 것으로 보입니다.



### Small Molecule Drug Discovery Through Deep Learning:Progress, Challenges, and Opportunities (https://arxiv.org/abs/2502.08975)
Comments:
          9 pages, 1 figures, 8 tables

- **What's New**: 최근의 심층 학습 기술(Deep Learning, DL)의 발전으로 인해 소분자 약물 발견이 크게 향상되었습니다. 이 논문은 지난 몇 년간 DL 기반 소분자 약물 발견의 주요 작업 및 대표 기술을 체계적으로 요약하고 일반화하는 데 중점을 두고 있습니다. 또한, 6개의 핵심 작업 및 관련 데이터셋, 기술 개발 경향에 대한 상세한 분석을 제공합니다.

- **Technical Details**: 소분자 약물은 분자량이 1000 이하인 화학 합성 유기 화합물입니다. DTI/DTA(약물-타겟 상호작용), DRP(약물-세포 반응) 등 6가지 주요 작업을 통해 약물 발견 프로세스의 특정 단계를 다루고 있습니다. 이들은 각각의 고유한 데이터 표현 방법을 필요로 하며, DL 기술의 발전으로 인해 예측 정확도와 계산 속도가 크게 향상되었습니다.

- **Performance Highlights**: DL 기반 방법들은 전통적인 기계 학습 기술에 비해 약물 발견 과정에서의 예측 효율성을 높이고 있습니다. DNN, CNN, GNN과 같은 심층 신경망은 바이오 의약 데이터의 대규모 패턴을 식별하고 예측 정확도를 향상시키는 데 기여하고 있습니다. 이러한 발전들은 소분자 약물 발견의 전반적인 효율성을 높이고, 보다 정확하고 효과적인 솔루션을 제공합니다.



### Modeling Time-evolving Causality over Data Streams (https://arxiv.org/abs/2502.08963)
Comments:
          Accepted by KDD'25

- **What's New**: 이 논문에서는 다변량(coevolving) 데이터 스트림에서 시간 변화 속 인과 관계를 모델링하고 미래 값을 예측하기 위한 새로운 스트리밍 방법인 ModePlait를 제안합니다. 여기에 다루어지는 데이터는 서로 영향을 미치는 관측량으로 구성되어 있으며, 이러한 관계를 효과적으로 발견하는 방법을 제시합니다. ModePlait는 외부 변수의 동적 변화에 따라 발전하는 인과 관계의 특성을 활용합니다.

- **Technical Details**: 모델은 (a) 적응적으로 뚜렷한 동적 패턴의 전환을 감지하여 다변량 공동 진화 데이터 스트림에서 시간 변화하는 인과 관계를 발견하고, (b) 스트리밍 방식으로 시간 변화하는 인과 관계의 발견과 미래 값의 예측을 동시에 수행합니다. 이 알고리즘은 데이터 스트림의 길이에 의존하지 않아서 매우 대규모 데이터에도 적용 가능하다는 특징이 있습니다.

- **Performance Highlights**: 제안된 모델은 합성 데이터와 실제 데이터셋에서 광범위한 실험을 통해 최신 방법들과 비교했을 때 시간 변화하는 인과 관계 발견 및 미래 예측 측면에서 우수한 성능을 보임을 입증했습니다. 이는 실제 적용에 있어 ModePlait의 효과성과 정확성을 뒷받침하는 결과입니다.



### A Comprehensive Survey on Imbalanced Data Learning (https://arxiv.org/abs/2502.08960)
- **What's New**: AI 연구 및 산업 전반에서 기계 학습의 발전이 눈부신 성과를 이루었으나, 불균형한 데이터 분포는 결정 과정에 편견을 발생시켜 기계 학습의 성능에 큰 영향을 미친다. 본 논문은 불균형 데이터에 대한 이해를 심화하고 관련 연구 및 응용을 촉진하기 위해 실제 데이터 형식을 체계적으로 분석한다.

- **Technical Details**: 정확한 데이터를 확보하기 위해 데이터를 재조정하는 방법으로는 랜덤 오버샘플링이 있다. 그러나 이 방법은 과적합(classifier overfitting)을 초래할 수 있어, 더 정교한 기법들, 예를 들어 선형 생성 방법 및 깊은 생성 모델이 필요하다. 이들 기술은 소수 클래스의 다양한 합성 샘플을 생성하여 기계 학습 모델의 학습 과정을 강화한다.

- **Performance Highlights**: SMOTE와 Mixup 등의 다양한 새로운 기술들이 소수 클래스의 샘플을 증강하고 데이터의 균형을 맞추기 위한 도구로 부각되고 있다. 그러나 이러한 기술들은 데이터 불균형을 더욱 악화시킬 위험이 있기 때문에 다양한 샘플링 방법이 도입되어 Remix, GAN과 같은 방법들이 발전하고 있다. GAN을 활용한 방법은 더욱 정교한 합성 샘플 생성으로 이어지며, 이를 통해 불균형 데이터 문제를 효과적으로 해결할 수 있는 가능성이 있다.



### Biologically Plausible Brain Graph Transformer (https://arxiv.org/abs/2502.08958)
Comments:
          27pages, 16figures, published as a conference paper at ICLR 2025

- **What's New**: 이번 연구에서는 Brain Graph의 생물학적 적합성을 높이기 위해 Biologically Plausible Brain Graph Transformer(BioBGT)를 제안합니다. BioBGT는 Brain Graph의 작은 세계 구조를 인코딩하고, 노드의 중요성을 전달하는 네트워크 얽힘 기법을 도입하여 뇌의 생물학적 특성을 반영합니다. 또한, 기능적 모듈을 고려한 자기 주의(self-attention) 기법을 통해 뇌 그래프의 기능적 분리 및 통합 특성을 보존합니다.

- **Technical Details**: 제안된 BioBGT는 두 가지 주요 구성 요소를 포함합니다. 첫째, 네트워크 얽힘 기반 노드 중요성 인코딩 기법은 뇌 그래프의 구조적 중요성을 반영하여 정보를 전파하는 과정에서 노드의 역할을 측정합니다. 둘째, 기능 모듈 인식을 통해 각 노드의 유사성을 정밀하게 조정하여 기능적 정합성을 유지하며, 커뮤니티 대비 전략을 통해 정밀한 기능적 모듈 노드 표현을 생성합니다.

- **Performance Highlights**: 실험 결과는 BioBGT가 기존의 최첨단 모델보다 우수한 성능을 보여준다는 것을 입증합니다. 특히 뇌 질환 탐지와 같은 다양한 뇌 그래프 분석 작업에 있어 생물학적 적합성이 강화된 표현이 효과적임을 입증하였습니다. 이 연구는 뇌 그래프의 생물학적 특성을 확보한 모델 설계의 중요성을 강조합니다.



### Self-Supervised Graph Contrastive Pretraining for Device-level Integrated Circuits (https://arxiv.org/abs/2502.08949)
- **What's New**: 이번 논문에서는 자기지도(self-supervised) 그래프 표현 학습에 새로운 모델 DICE(Device-level Integrated Circuits Encoder)를 소개합니다. DICE는 장치 수준에서 표현된 회로에 대해 처음으로 자가 지도 사전 훈련된 그래프 신경망(GNN) 모델로, 아날로그와 디지털 회로 모두에 적용될 수 있습니다. 특히, 회로 설계 자동화(Electronics Design Automation, EDA) 분야 내에서 아날로그 및 혼합 신호 회로를 다루는 데 있어 기존 방법의 한계를 극복하고자 합니다.

- **Technical Details**: DICE는 메시지 전송 신경망(Message-Passing Neural Network, MPNN)으로, 그래프 대조 학습(graph contrastive learning)을 통해 훈련됩니다. 이 모델은 회로를 그래프로 변환하고, 두 가지 새로운 데이터 증강(data augmentation) 기법을 적용하여 회로의 고수준 의미를 보존하거나 변경하는 방식으로 데이터 다양성을 제공합니다. 이러한 기법들은 회로 시뮬레이션이나 EDA 도구에 의존하지 않고도 효과적인 결과를 이끌어낼 수 있습니다.

- **Performance Highlights**: 실험 결과, DICE는 세 가지 하류(task-specific) 작업에서 상당한 성능 향상을 보여주었습니다. 이는 아날로그 및 디지털 회로 모두에서 그 효과성을 입증하며, 특히 회로 구조에 따라 다양한 평가가 가능한 포괄적인 벤치마크를 기반으로 합니다. DICE의 도입은 장치 수준의 통합 회로에 대한 자기지도 그래프 표현 학습의 새로운 가능성을 제시합니다.



### Language in the Flow of Time: Time-Series-Paired Texts Weaved into a Unified Temporal Narrativ (https://arxiv.org/abs/2502.08942)
Comments:
          Preprint, 37 pages

- **What's New**: 이번 연구에서는 기존의 수치 데이터 중심의 시계열 모델 연구에서 간과된 멀티모달 시계열, 특히 텍스트 정보를 효과적으로 통합하는 방법을 제시합니다. 저자들은 시간 시리즈와 텍스트가 짝지어진 데이터가 주기적 특성을 나타내며, 이를 통해 시계열 모델링을 향상시킬 수 있는 잠재력을 강조합니다. 새로운 프레임워크인 Texts as Time Series (TaTS)가 이러한 통찰을 기반으로 제안되었습니다.

- **Technical Details**: TaTS는 시간 시리즈 데이터에서 쌍(pair)으로 된 텍스트 정보를 보조 변수로 고려하여 기존의 수치 중심 모델과 통합함으로써 시계열 데이터의 예측 성능을 향상시킵니다. 이 프레임워크는 쌍텍스트의 잠재적 표현을 하향 차원으로 변환하고, 이를 기존 시계열 모델에 입력하여 숫자와 텍스트의 시간적 동역학을 모두 캡처합니다. 이는 멀티모달 시계열 분석을 위해 필요한 다양한 모델과 호환될 수 있습니다.

- **Performance Highlights**: 다양한 벤치마크 데이터 세트와 기존의 여러 시계열 모델을 통한 실험 결과, TaTS는 예측 및 보간(task) 작업에서 최첨단 성능을 달성했습니다. 이를 통해 TaTS는 시계열 모델의 아키텍처를 수정하지 않고도 성능을 향상시킬 수 있음을 확인했습니다. 새로운 방식이 기존 접근방식에 비해 어떻게 뚜렷한 금융성과를 이끌어내는지를 강조하고 있습니다.



### Analysis of Off-Policy $n$-Step TD-Learning with Linear Function Approximation (https://arxiv.org/abs/2502.08941)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2402.15781

- **What's New**: 이 논문은 "deadly triad" 시나리오에서의 다단계 시간차 학습 알고리즘에 대한 분석을 제공합니다. 특히, n-단계 TD-learning 알고리즘이 샘플링 수평(n)이 충분히 증가함에 따라 수렴함을 증명합니다. 이 논문은 두 부분으로 나뉘며 첫 번째 부분에서는 모델 기반 결정론적 알고리즘의 기본 속성을 포괄적으로 살펴봅니다.

- **Technical Details**: TD-learning은 강화를 통한 RL의 핵심 알고리즘으로, 특히 정책 평가에 중요합니다. 소개된 방법 중 하나인 gradient temporal-difference learning (GTD)은 deadly triad 문제를 해결하지만, 환경에 대한 제한적인 가정이 필요합니다. 논문에서는 모델 기반 결정론적 알고리즘과의 관계를 통해 n-단계 TD 방법의 수렴 조건과 효과적 해결 방안을 제시합니다.

- **Performance Highlights**: 이 논문은 n-단계 TD-learning 알고리즘이 deadly triad의 문제를 효과적으로 완화하는 방법을 제시하며, 샘플링 수평(n)이 충분히 클 때 수렴성을 보장합니다. 이 분석을 통해 미래의 모델 자유 강화 학습 개발에 있어 중요한 통찰을 제공하며, 모델 기반 알고리즘의 ODE 대응 분석 또한 포함되어 있습니다.



### Reevaluating Policy Gradient Methods for Imperfect-Information Games (https://arxiv.org/abs/2502.08938)
- **What's New**: 최근 연구들은 DRL(Deep Reinforcement Learning)을 이용한 비대칭 정보 게임에서 발생하는 한계점을 극복하기 위해 FP(Fictitious Play), DO(Double Oracle), CFR(Counterfactual Regret Minimization) 기반의 알고리즘들을 개발했습니다. 그러나 본 논문에서는 PPO(Proximal Policy Optimization)와 같은 일반적인 정책 기울기법이 이러한 기존 알고리즘들과 경쟁력이 있거나 우수할 것이라는 가설을 제시하고 있습니다. 이를 검증하기 위해 4개의 대규모 게임에서 정밀한 착취 가능성 계산을 구현하였으며, 이들과 함께 DRL 알고리즘의 큰 비교 연구를 진행했습니다.

- **Technical Details**: 비대칭 정보 게임(IIG)은 플레이어 간의 정보 비대칭성이 존재하는 게임으로, 특히 두 플레이어의 제로섬 게임에서 많이 연구되어 왔습니다. 기존 문헌에서는 FP, DO, CFR과 같은 알고리즘을 통해 IIG에서의 최적 전략을 다루어왔지만, 이들 접근법은 효과적인 DRL 방법론으로 발전시키기 어려움이 있었습니다. 본 연구에서는 MMD(Magnetic Mirror Descent)라는 새로운 정책 경량화 방법론이 CFR과 경쟁할 수 있으며, PPO와 같은 일반적인 정책 기법들이 더 나은 성능을 보일 것이라는 점에 주목하였습니다.

- **Performance Highlights**: 5600회의 훈련 과정을 통해 FP, DO, CFR 기반의 접근법들이 일반적인 정책 기법인 MMD, PPO, PPG에 비해 우수한 성과를 내지 못함을 확인하였습니다. 이러한 연구 결과는 논문에 포함되어 있으며, DRL 알고리즘의 착취 가능성(breaching) 비교에 대한 대규모 데이터세트를 제공하고 있습니다. 연구팀은 OpenSpiel과 호환되는 코드를 공개하여, 향후 연구자들이 이 가설을 검증하고 새로운 기준을 세울 수 있도록 기회를 제공하고 있습니다.



### AutoLike: Auditing Social Media Recommendations through User Interactions (https://arxiv.org/abs/2502.08933)
Comments:
          17 pages, 6 figures, 3 tables

- **What's New**: 이 연구에서 새로운 AutoLike 프레임워크를 소개합니다. 이 프레임워크는 소셜 미디어 플랫폼의 추천 시스템을 감사(audit)할 수 있도록 설계되었습니다. AutoLike는 자동화된 방식으로 사용자 상호작용을 통해 추천 알고리즘이 특정 주제의 콘텐츠를 제공하는 방식을 평가합니다.

- **Technical Details**: AutoLike는 강화 학습(reinforcement learning) 문제로 포맷팅되어 있습니다. 사용자가 특정 주제를 선택하고 소셜 미디어 플랫폼에서의 행동(예: 좋아요, 스킵 등)을 통해 추천 시스템을 조작합니다. 이 과정은 내용의 주제와 감정을 기준으로 보상을 최적화하는 정책(policy)을 학습합니다.

- **Performance Highlights**: AutoLike 프레임워크는 TikTok에서 효과적으로 작동하여 특정 주제와 감정을 가진 콘텐츠의 추천을 지향하도록 시스템을 조정합니다. 실험 결과, AutoLike는 통제된 실험 대비 부정적인 정신 건강 콘텐츠를 2배 더 많이 제공합니다. 이 프레임워크는 정책 입안자, 플랫폼 개발자 및 연구자에게 유용한 도구가 될 가능성을 지니고 있습니다.



### Escaping Collapse: The Strength of Weak Data for Large Language Model Training (https://arxiv.org/abs/2502.08924)
- **What's New**: 이번 논문에서는 합성 데이터(synthetic data)가 대형 언어 모델(LLM)의 훈련에 어떻게 기여하는지와 이러한 데이터의 큐레이션(curation) 없이는 성능이 정체되거나 심지어 감소할 수 있다는 점에 대해 논의합니다. 제안된 이론적 프레임워크를 통해 LLM 성능 향상을 위한 큐레이션의 필요성을 정량적으로 분석하였습니다. 많은 비합성 데이터의 품질이 저조하더라도 최적의 LLM으로 수렴하는 훈련 절차를 설명합니다.

- **Technical Details**: 이 논문은 부스팅(boosting)이라는 고전적인 기계 학습 기술에 착안하여, 약한 학습 알고리즘(weak learning algorithm)을 활용해 우수한 분류기를 생성하는 방식을 통해 분석을 진행하였습니다. 제안된 훈련 절차는 최근에 발표된 여러 합성 데이터 훈련 방법들을 포함하며, 이러한 방법들이 어떻게 성공적인지를 설명하는 데 기여합니다. 특히, 어려운 예제에 동적으로 라벨링 자원을 집중시키는 방식을 통해 성능을 향상시킬 수 있음을 보여줍니다.

- **Performance Highlights**: 실험 결과, 합성 데이터에서 훈련된 LLM이 향상된 성능을 보이는 것으로 나타났습니다. 훈련 절차가 약한 학습기의 노력을 집중하는 방식과 유사하게 작동하여, 도전적인 예에 대한 집중이 성능을 개선하는 데 기여합니다. 이러한 결과는 기존의 방법들이 성공하는 이유를 명확히 하며, 향후 개선 작업에 대한 기회를 제시합니다.



### CLEAR: Cluster-based Prompt Learning on Heterogeneous Graphs (https://arxiv.org/abs/2502.08918)
Comments:
          accepted by PAKDD 2025

- **What's New**: CLEAR는 이종 그래프에서 구동되는 새로운 프롬프트 학습 모델로, 이종 그래프 복원으로 하위 작업을 재구성하는 방법을 제안합니다. 기존 방법들은 주로 노드 특성을 수정하여 프롬프트를 다루었으나, CLEAR는 메타 경로의 정보를 통합하여 고차원 의미를 학습합니다. 실험 결과, CLEAR는 최신 모델보다 F1 메트릭에서 최대 5% 향상된 성능을 보였습니다.

- **Technical Details**: CLEAR 모델은 클러스터 프롬프트를 통해 이종 그래프의 구조 정보를 학습합니다. 훈련 시 유사한 레이블을 가진 노드들이 클러스터를 형성하며, 이를 가상 노드로 추상화하여 프롬프트로 사용합니다. 메타 경로 템플릿을 통해 프롬프트는 고차원 의미 정보를 포착하도록 설계되었습니다.

- **Performance Highlights**: 광범위한 실험을 통해 CLEAR 모델은 가장 진보된 이종 그래프 학습 모델들에 비해 뛰어난 성능을 입증했습니다. 특히, 노드 분류 작업에서 F1 점수 기준으로 약 5% 향상된 결과를 나타냈으며, 이는 CLEAR의 효과iveness를 강하게 뒷받침합니다.



### Linear-Time User-Level DP-SCO via Robust Statistics (https://arxiv.org/abs/2502.08889)
- **What's New**: 이번 연구에서는 사용자 수준에서의 차등 개인정보 보호(user-level differentially private, DP) 확률적 볼록 최적화(stochastic convex optimization, DP-SCO)를 제안합니다. 기존의 방법들이 높은 노이즈 축적과 비효율적인 유틸리티로 어려움을 겪는 것에 반해, 본 알고리즘은 견고한 통계 방법을 활용하여 중간 결과물의 민감도를 제어합니다. 이는 개인 정보를 보호하는 데 필요한 노이즈를 줄이고, 더욱 향상된 개인정보와 유틸리티 간의 균형을 이룹니다.

- **Technical Details**: 제안하는 알고리즘은 중간 결과물의 민감도를 감소시키기 위해 미디안(median) 및 트리미드(mean) 평균을 사용하는 선형 시간 알고리즘입니다. 기존의 DP-SGD(differentially private stochastic gradient descent) 방식의 반복적 개인화 절차를 회피함으로써, 이론적으로 개선된 개인정보 및 유틸리티 균형을 달성하는 동시에 계산 효율성도 유지합니다. 또한, 정보 이론적 하한을 제공하여 제안한 상한이 최적임을 보였습니다.

- **Performance Highlights**: 이번 연구는 범용적인 머신러닝 응용 프로그램에 필요한 효율적이고 견고한 개인 정보 보호 기술을 개발하는 데 기초가 됩니다. 특히, 제안된 알고리즘은 단순한 구조로도 고도의 개인 정보 보호를 유지하면서도 계산적으로 효율적인 성능을 발휘합니다. 향후 연구와 응용 가능성에 대한 중요한 시사점을 제공합니다.



### 2D Integrated Bayesian Tomography of Plasma Electron Density Profile for HL-3 Based on Gaussian Process (https://arxiv.org/abs/2502.08882)
- **What's New**: 이번 논문은 Gaussian Process (GP)를 이용하여 선적분 측정(line integral measurements)과 점 값(point values)을 결합한 통합 베이지안 모델을 소개합니다. 제안된 방법은 Gaussian Process Regression (GPR)을 활용하여 2차원 프로파일에 점 값을 통합하며, 좌표 매핑(coordinate mapping)을 통해 자속 정보(magnetic flux information)를 2D 역전환(inversion) 과정에 적용합니다. 이는 과학적 연구에 있어 새로운 접근 방식을 제공합니다.

- **Technical Details**: 재구성된 프로파일의 평균 상대 오차(average relative error)는 3.60*10^(-4)로, 통합된 베이지안 단층촬영 모델(integrated Bayesian tomography model)과 정규화된 자속(normalized magnetic flux)을 사용하여 최소화되었습니다. 또한, 그리드 수(grid number), 합성 진단 데이터의 표준편차(standard deviation), 노이즈 수준(noise levels)에 대한 민감도 테스트가 수행되어 실험 데이터에 모델을 적용하는 데 있어 튼튼한 기초를 마련했습니다.

- **Performance Highlights**: 이 연구는 통합 베이지안 모델을 통해 정확한 2D 역전환을 달성하였으며, 평형 재구성의 압력 정보(pressure information)를 분리하는 데 유용한 강력한 프레임워크를 제공했습니다. 이는 역전환 결과를 활용하여 평형 재구성을 최적화하는 가능성을 제시하고 있습니다.



### WENDy for Nonlinear-in-Parameter ODEs (https://arxiv.org/abs/2502.08881)
- **What's New**: WENDy 알고리즘의 확장을 통해 비선형 함수(parameter)가 포함된 상미분 방정식 시스템을 처리할 수 있게 되었습니다. 이 새로운 알고리즘인 WENDy-MLE는 최대 우도(maximum likelihood) 추정치를 찾기 위해 비선형 최적화(non-convex optimization) 루틴을 활용합니다. 이 접근법은 기존의 출력 오차 최소 제곱(output error least squares) 방법보다 더 나은 정확도와 더 큰 수렴 영역을 제공하며, 종종 수 배의 속도를 보여줍니다.

- **Technical Details**: WENDy-MLE 알고리즘은 수학적 유도 과정을 통해 가우시안 노이즈를 가정한 약한 형태의 잔차(residual)와 관련된 우도 함수(likelihood function)를 설명합니다. 또한, 이 알고리즘은 첫 번째 및 두 번째 차수의 유도 파라미터(derivative)를 분석적으로 계산하고, 곱셈 로그-정규 노이즈(multiplicative log-normal noise)의 경우로 확장됩니다. 이러한 수학적 기초는 알고리즘의 최적화 과정에서 매우 중요한 역할을 합니다.

- **Performance Highlights**: WENDy-MLE 알고리즘은 여러 벤치마크 시스템에 대한 수치 결과를 통해 기존의 솔버와 비교하여 더 높은 정확성과 정밀도를 보여주는 것으로 입증되었습니다. 이 연구는 알고리즘이 잡음(noise)과 초기화 조건에 대한 강인성을 갖추고 있다는 점에서도 강점을 보입니다. 따라서, 이 새로운 방법은 다양한 상황에서 안정적으로 연산 성능을 발휘할 수 있는 가능성을 지니고 있습니다.



### Robust Graph-Based Semi-Supervised Learning via $p$-Conductances (https://arxiv.org/abs/2502.08873)
Comments:
          29 pages, 7 figures

- **What's New**: 이 논문은 소량 또는 손상된 데이터 레이블로 구축된 그래프에서 반지도 학습(semi-supervised learning)의 문제를 다룹니다. 우리는 $p$-conductance learning 접근 방식을 제안하여, $p$-Laplacian 및 Poisson 학습 방법을 일반화하며, 레이블 제약의 아핀 완화를 도입합니다. 이를 통해 확률 측정 최소 절단 프로그램(probability measure mincut programs)의 가족이 생성되어, 정확한 분포 분리와 함께 희소한 엣지 제거를 균형 있게 수행할 수 있습니다.

- **Technical Details**: 제안된 알고리즘은 세미부드 뉴턴-컨주게이트 그라디언트(semismooth Newton-conjugate gradient) 방법을 기반으로 하며, 연속 솔루션을 레이블 지정으로 변환할 때 클래스 크기(Class size) 추정을 포함하도록 확장됩니다. 이 접근 방식은 그래프에서의 확률 및 변분 문제와 연결되며, 라벨이 히트 커널을 통해 분산될 때의 강인성에 대한 동기를 부여합니다. 또한, 다양한 $p$ 값에 대한 가장 최근의 연구 결과도 논의되고 있습니다.

- **Performance Highlights**: 응용 프로그램에서 컴퓨터 비전과 인용 데이터셋에 대한 실험 결과는 제안된 방법이 낮은 라벨 비율, 손상된 라벨 및 부분 라벨 제어 환경에서도 최첨단 분류 정확도를 달성함을 보여줍니다. 특히, $p > 2$의 경우, 전통적인 Laplace 학습보다 그래프 기반 세미 지도 학습에서 더 우수한 성능을 나타났습니다. 이러한 점에서, 본 연구의 결과는 라벨 일관성을 확보하는 데 중요한 기여를 합니다.



### When and why randomised exploration works (in linear bandits) (https://arxiv.org/abs/2502.08870)
- **What's New**: 이번 논문에서는 강제 낙관성(forced optimism)이나 후방 확장(posterior inflation)에 의존하지 않는 탐색 알고리즘(randomised exploration algorithms), 특히 Thompson sampling을 분석하는 새로운 접근법을 제공합니다. 이는 $d$-차원 선형 밴딧(linear bandit) 환경에서 동작 공간(action space)이 부드럽고 강하게 볼록(strongly convex)할 때, 랜덤 탐색 알고리즘이 $n$-단계 후회(regret) 경계가 $O(d	extsqrt{n} 	ext{log}(n))$임을 증명합니다.

- **Technical Details**: 논문에서 제시하는 접근법은 선형 밴딧 설정에서의 탐색 성능을 연구하는 데 초점을 두고 있습니다. 이 연구는 일반적인 강제 낙관성과 후방 확장을 배제하고, 주어진 차원(dimension)에서 최적의 후회 경계를 제공하는 것이 특징입니다. 따라서, 랜덤 탐색 알고리즘들은 주요 조건 하에 후회의 차원 의존성(dimension dependence)을 최적화할 수 있음을 보여줍니다.

- **Performance Highlights**: Thompson sampling이 탐색 알고리즘으로서 비주얼화된 선형 밴딧 환경에서 최적의 후회 경계를 달성할 수 있는 경우를 처음으로 제시합니다. 이는 랜덤 탐색 알고리즘의 성능을 새로운 관점에서 바라볼 수 있는 기회를 제공합니다. 또한, 이 결과는 향후 다양한 관련 문제들에 대한 응용 가능성을 열어줍니다.



### Harnessing Vision Models for Time Series Analysis: A Survey (https://arxiv.org/abs/2502.08869)
- **What's New**: 이 논문은 시간 시계열 분석(time series analysis)에서 LLMs(대형 언어 모델)보다 LVMs(대형 비전 모델)의 장점에 대해 논의합니다. 최근까지 연구의 대부분은 시퀀스 모델링(sequence modeling)에 집중되어 있었으나, 비전 모델이 시간 시계열 분석에서 중요한 역할을 할 수 있다는 점을 강조하고 있습니다. 이 논문은 시간 시계열을 이미지로 인코딩하는 방법과 여러 작업을 위한 모델링 기법을 제시하며, 이 분야의 나아갈 방향도 함께 논의합니다.

- **Technical Details**: 시간 시계열은 다양한 형태로 표현될 수 있으며, 여기서 이미지로 변환된 시간 시계열(imaged time series)을 대상으로 연구합니다. UTS(단일 변량 시계열)와 MTS(다변량 시계열)의 정의 및 표현 방법이 상세히 설명됩니다. 그림 1에서는 비전 모델을 시간 시계열 작업에 적용하는 일반적인 프로세스와 프레임워크를 제시하며, 경우에 따라 선형 그래프(line plot), 히트맵(heatmap), 스펙트로그램(spectrogram) 등의 변환 기법에 대해 다룹니다.

- **Performance Highlights**: 비전 모델이 시간 시계열 작업에서 왜 더 효과적인지를 여러 사례를 통해 설명하고 있으며, 기존의 LLMs와 비교해 여러 장점이 발견됩니다. LVMs는 시계열 데이터의 고유한 패턴, 상관 관계(correlation) 및 장기 종속성(long-term dependency) 모형화를 더 잘 수행할 수 있습니다. 또한, 이 논문은 향후 LMMs(대형 다중모달 모델)의 가능성을 제시하며, 시계열 데이터와 비전 데이터를 통합한 혁신적인 방식의 개발을 독려합니다.



### A Systematic Evaluation of Generative Models on Tabular Transportation Data (https://arxiv.org/abs/2502.08856)
- **What's New**: 이번 연구는 대규모 교통 데이터 공유의 이점과 이로 인해 발생할 수 있는 개인 정보 보호 문제를 다루며, 실제 데이터를 기반으로 한 합성 데이터 생성을 통해 이러한 문제를 해결하고자 한다. 기존의 합성 데이터 생성 기법들이 교통 데이터의 독특한 특성에 맞춰져 있지 않다는 점을 강조하며, 뉴욕시 택시 데이터를 사례 연구로 사용하여 기존 기법의 성능을 체계적으로 평가한다.

- **Technical Details**: 합성 데이터 생성 모델로는 Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs) 및 확산 모델(difussion models)이 자주 언급된다. 연구에서는 기존의 전통적인 성능 지표 이외에 네트워크 구조의 유사성을 측정할 수 있는 새로운 그래프 기반(metric) 지표를 제안하고, 개인 정보 유출을 측정하기 위한 향상된 지표도 도입하였다. 이러한 접근을 통해 교통 데이터에 적합한 평가 기준을 마련하고자 한다.

- **Performance Highlights**: 연구 결과, 기존의 합성 데이터 생성 모델들은 문헌에서 주장하는 만큼의 일관된 성능을 보이지 않았으며, 특히 교통 데이터에 적용 시 성능 저하를 겪었다. 새로운 그래프 기반 지표는 합성 데이터와 실제 데이터 간의 현저한 격차를 드러내었다. 이는 교통 데이터와 같은 고유한 특성을 가진 분야를 위해 맞춤화된 생성 모델 개발의 필요성을 강조한다.



### A Reversible Solver for Diffusion SDEs (https://arxiv.org/abs/2502.08834)
Comments:
          Preprint

- **What's New**: 최근 확산(기법) 모델은 다양한 데이터 모달리티에서 생성 작업의 최첨단(상태-최고)으로 빠르게 자리 잡았습니다. 이 모델의 중요한 기능 중 하나는 데이터 분포에서 샘플을 샘플링 우선 분포로 다시 인코딩(encoding) 할 수 있는 능력입니다. 이는 실제 데이터 샘플을 수정하거나 연속적인 부가 방정식을 통해 가이던스 생성(guided generation)을 수행하는 데 유용합니다.

- **Technical Details**: 우리는 확산(기법) SDE의 대수적으로 가역적인 솔버를 제안합니다. 이 솔버는 실제 데이터 샘플을 정확하게 우선 분포로 역전(invert)할 수 있습니다. 대수적 가역성(algebraic reversibility)은 모델이 데이터 샘플 간의 변환을 효율적으로 수행할 수 있도록 도와줍니다.

- **Performance Highlights**: 이 모델은 현실 데이터를 조작하는 데에서 실질적인 응용 가능성을 보여주며, 변형된 데이터 샘플을 생성하는 데 있어 뛰어난 성능을 자랑합니다. 또한, 기대할 수 있는 샘플링 결과의 품질을 높이는 데 있어 유망한 연구 방향을 제시합니다.



### PLayer-FL: A Principled Approach to Personalized Layer-wise Cross-Silo Federated Learning (https://arxiv.org/abs/2502.08829)
- **What's New**: 이번 논문에서는 비독립 및 동일하게 분포하지 않는 데이터에 대한 연합 학습(Federated Learning, FL)의 도전과제를 해결하기 위한 새로운 접근법인 PLayer-FL(Principled Layer-wise-FL)을 소개합니다. PLayer-FL은 페더레이션 감도(federation sensitivity)라는 새로운 메트릭을 활용하여 어떤 레이어가 페더레이션에 적합한지를 판단합니다. 이 메트릭은 각 레이어의 특성이 다른 클라이언트 간 일반화에 미치는 영향을 정량화하여, 페더레이션의 이점이 줄어드는 전환점을 식별합니다.

- **Technical Details**: PLayer-FL은 모델 프루닝(model pruning)에서 영감을 얻어, 첫 번째 훈련 에폭(epoch) 후에 각 레이어의 기여를 정량화하는 방법론을 개발했습니다. 이 감도 메트릭은 다양한 모델 아키텍처와 실제 비독립 및 동일한 데이터셋을 통해 검증되었으며, 페더레이션을 통해 성능 향상 및 공정성을 증대시키는 데 기여합니다. 기존의 부분 FL 방법과 달리 PLayer-FL은 특정 아키텍처에 구애받지 않고 더 일반화된 응용이 가능합니다.

- **Performance Highlights**: 논문에서 제안된 PLayer-FL은 기존의 FL 알고리즘보다 다양한 과제에서 더 나은 성능을 발휘했으며, 클라이언트 간의 성능 향상을 균일하게 달성했습니다. 특히, 이는 동일한 초기화 조건을 가지도록 훈련된 모델들이 비독립적 훈련을 거쳐도 일관된 일반화 패턴을 나타내는 것을 보여줍니다. 이러한 결과는 PLayer-FL 접근법이 클라이언트의 참여를 유도할 수 있는 더 강력한 인센티브를 제공한다는 점에서 의미가 깊습니다.



### A Survey on Data-Centric AI: Tabular Learning from Reinforcement Learning and Generative AI Perspectiv (https://arxiv.org/abs/2502.08828)
- **What's New**: 이 논문은 데이터 중심 인공지능(Data-Centric AI, DCAI) 관점에서 테이블 형식 데이터의 최적화를 위한 새로운 기법을 제시합니다. 특히 강화 학습(Reinforcement Learning, RL)과 생성적 접근법을 통해 중요한 특성을 선택하고 새로운 특성을 생성하는 방법을 탐구합니다. 이를 통해 데이터 품질을 개선하고 모델 성능을 향상시키는 데 기여하고자 합니다.

- **Technical Details**: 테이블 형식 데이터는 복잡한 특성 의존도, 높은 차원성 및 해석 가능성 요건 등을 만족해야 하며, 이로 인해 많은 도전 과제를 안고 있습니다. 본 논문에서는 이러한 도전 과제를 해결하기 위해 강화 학습 기반 기법과 생성 모델을 활용하여 특성 선정(feature selection) 및 특성 생성(feature generation) 과정을 최적화하는 방안을 제시합니다. 강화 학습은 보상 기반 탐색을 통해 모델이 적응적으로 특성 표현을 학습할 수 있도록 하며, 생성 모델은 복잡한 데이터 패턴을 포착하는 새로운 특성을 생성합니다.

- **Performance Highlights**: 이 연구는 데이터 중심 AI 및 테이블 형식 데이터 엔지니어링의 최신 발전을 체계적으로 검토하고, 기존 방법들의 강점과 한계를 분석합니다. 연구결과, RL 및 생성적 접근법이 테이블 형식 데이터의 특성 공학(feature engineering) 자동화와 지능화에 기여함을 보여줍니다. 또한, 기존의 도전 과제를 정리하고 향후 연구 방향을 제시하여 이 분야의 지속적인 혁신을 위한 통찰력을 제공하고자 합니다.



### A First-order Generative Bilevel Optimization Framework for Diffusion Models (https://arxiv.org/abs/2502.08808)
- **What's New**: 이번 논문에서는 확산 모델(difussion model)에 대한 이중 최적화(bilevel optimization) 문제를 제기하고, 두 가지 주요 시나리오를 다루고 있습니다. 첫 번째는 미리 학습된 모델을 파인튜닝(fine-tuning)하기 위해 샘플 효율적인 그래디언트 추정기를 사용하는 하위 수준(inference-only lower-level solver)입니다. 두 번째는 노이즈 스케줄 최적화를 통해 확산 모델을 처음부터 훈련하는 것입니다. 이를 통해 기존의 이중 최적화 방법들이 가지고 있는 한계를 극복하고자 하였습니다.

- **Technical Details**: 논문은 두 가지 렌즈를 통해 생성적 이중 최적화 문제를 정의합니다: 파인튜닝된 모델을 위한 경우와 노이즈 분포를 학습하는 경우입니다. 저자들은 전통적인 이중 최적화 기법이 무한 차원 확률 공간에서 작동하는 확산 프로세스와 호환되지 않는 문제를 해결하기 위해 1차 이중 프레임워크를 개발하였습니다. 이를 통해 계산적으로 효율적이고 확산 모델에 친화적인 접근 방식을 제시합니다.

- **Performance Highlights**: 실험 결과, 이 방법은 기존의 파인튜닝 및 하이퍼파라미터 검색 기법들을 초월하는 성과를 보였습니다. 구체적으로, 조정된 하이퍼파라미터와 효율적인 그래디언트 추정기를 통해 성능을 극대화할 수 있음을 입증하였습니다. 이번 연구는 확산 모델에 대한 이중 하이퍼파라미터 최적화(bilevel hyperparameter optimization)의 새로운 가능성을 여는 계기가 될 것입니다.



### Deep EEG Super-Resolution: Upsampling EEG Spatial Resolution with Generative Adversarial Networks (https://arxiv.org/abs/2502.08803)
- **What's New**: 이 논문에서는 Generative Adversarial Networks (GANs)를 기반으로 한 새로운 EEG 초해상도(Super-Resolution, SR) 접근 방식을 제안합니다. 이 접근법은 고해상도의 EEG 데이터를 저해상도 샘플로부터 생성하는 최첨단 기술로, 저비용의 EEG 기기를 통해 더 많은 채널을 효과적으로 보완할 수 있습니다. EEG 데이터의 낮은 밀도 문제를 해결하며, 돈이 많이 드는 장비 대신 저렴한 하드웨어로 높은 해상도를 제공합니다.

- **Technical Details**: 연구에 사용된 모델은 Wasserstein GAN (WGAN) 프레임워크에 기반하여 효율성을 높였습니다. 제안된 WGAN 모델은 생성자와 판별자로 구성되어 있으며, 본 연구의 필요에 따라 EEG 데이터를 효과적으로 처리하는 수정된 합성곱 신경망(CNN) 아키텍처를 사용합니다. 본 모델은 여러 채널에서의 관계를 학습할 수 있도록 최적화된 합성곱층을 포함하고 있습니다.

- **Performance Highlights**: 실험 결과, 제안한 GAN 모델은 기존의 비율 보간법(bicubic interpolation) 방식에 비해 평균 제곱 오차(Mean Squared Error, MSE)와 평균 절대 오차(Mean Absolute Error, MAE)에서 각각 10,000배 및 100배의 감소를 나타냈습니다. 또한, 초해상도 EEG 데이터를 이용한 분류기 훈련에서도 성능 손실이 최소화되어, 본 연구의 GAN 기반 EEG SR 접근법이 유망한 결과를 보여줍니다.



### Low-Resolution Neural Networks (https://arxiv.org/abs/2502.08795)
Comments:
          22 pages, 13 figures

- **What's New**: 본 연구는 다중 클래스 이미지 분류에 있어 모델 성능에 대한 매개변수 비트 정밀도의 영향을 조사합니다. 특히 1비트에서 4.08비트에 이르는 범위의 모델을 분석하여 32비트 모델과의 성능을 비교합니다. 그 결과, 낮은 정밀도의 매개변수를 가진 모델들이 메모리 절약과는 대조적으로 뛰어난 성능을 발휘할 수 있음을 확인했습니다. 이러한 발견은 메모리 제약이 있는 장치에서도 활용 가능성을 제시합니다.

- **Technical Details**: 연구에서는 완전 연결층(FCNN), 합성곱층(CVNN), 변환기 블록(ViT) 모델을 분석하였으며, 각 모델의 매개변수 정밀도를 1비트부터 4.08비트까지 다양하게 조정했습니다. 특히, 데이터 증강(data augmentation)이 낮은 해상도의 모델에서 훈련을 불안정하게 만들 수 있지만, 가중치 매개변수에 0을 포함시키는 방법이 성능 저하를 방지하는 데 효과적이라는 사실을 발견했습니다. 본 연구는 메모리 절약과 성능 균형을 이루기 위해 2.32비트의 가중치가 가장 최적의 결과를 나타낸다고 주장합니다.

- **Performance Highlights**: 저비트 양자화(1-1.5비트)는 메모리와 계산 자원 절약의 이점을 제공하지만, 정확성을 해칠 위험이 있는 반면, 높은 비트 수준(3.17-4.08비트)은 성능 및 효율성 간의 균형을 유지합니다. 실험 결과, 많은 매개변수를 가진 저해상도 모델은 동일한 에폭 수 내에서 32비트 모델에 비견하는 성능을 나타냈으며, 이는 메모리에 대한 요구를 줄이고 계산 효율성을 개선할 수 있는 가능성을 보여줍니다. 그러나 이 연구는 더 복잡하고 큰 모델과 다양한 데이터셋에 대한 추가 연구가 필요하다고 강조합니다.



### Spectral Journey: How Transformers Predict the Shortest Path (https://arxiv.org/abs/2502.08794)
Comments:
          12 pages

- **What's New**: 이 연구에서는 두 개의 레이어로 구성된 디코더 전용 변환기 모델이 10개의 노드로 구성된 단순하고 연결된 그래프에서 최단 경로를 예측하는 방법을 학습할 수 있음을 보여줍니다. 이 모델은 그래프 엣지의 스펙트럴 임베딩(spectral embedding)을 사용하여 최단 경로를 계산하는 새로운 알고리즘, 스펙트럴 라인 내비게이터(Spectral Line Navigator, SLN)를 발견했습니다. 이 연구는 기계 신경망이 그래프 문제를 해결하기 위한 일반 알고리즘을 학습하는 가능성을 확장합니다.

- **Technical Details**: 저자들은 연결된 비방향 그래프(simple, connected, undirected graph)를 사용하여 모델의 내부 동작을 연구했습니다. 이 과정에서 모델의 주의 맵(attention maps) 및 학습된 표현을 분석하여 최단 경로를 계산하는 알고리즘을 파악했습니다. 최단 경로를 찾기 위해, 모델은 그래프 내에서 가장 짧은 거리에 있는 엣지를 선택하는 스펙트럼 기반의 알고리즘을 사용하며, 이 연구에서 제안된 SLN 알고리즘은 99.32%의 정확도로 평가되었습니다.

- **Performance Highlights**: 두 층으로 구성된 디코더 전용 모델은 최대 10개의 노드가 있는 단순한 연결 그래프에서 최단 경로를 예측할 수 있으며, 이러한 모델은 알고리즘적으로 유의미한 결과를 보여줍니다. 모델은 스펙트럼 분해(spectral decomposition)에 상관관계가 있는 그래프 엠베딩(graph embedding)을 학습했으며, 이때 현재 노드와 목표 노드가 포함된 엣지를 강하게 주의(attend)합니다. SLN 알고리즘은 그래프 이론 문헌에서도 짧게 연구된 적이 있지만, 저자들은 SLN이 최근 발견된 새로운 최단 경로 찾기 알고리즘이라고 주장합니다.



### Decision Tree Based Wrappers for Hearing Loss (https://arxiv.org/abs/2502.08785)
- **What's New**: 이 논문에서는 청각학(Audiology) 분야에서 머신러닝(Machine Learning) 모델을 활용하여 위험이 있는 사람들을 선별하는 방법이 소개됩니다. 특히, 진화적 방법론을 사용한 Feature Engineering(Feature Engineering) 기법이 중요한 역할을 한다는 점이 강조됩니다.

- **Technical Details**: FEDORA 프레임워크가 청각 손실(Hearing Loss) 데이터셋에 적용되어 데이터 차원 축소(data dimensionality reduction)를 수행하고, 기초 성능을 통계적으로 유지하는 것으로 나타났습니다. 의사결정나무(decision trees) 기반의 모델을 사용하여 진화적 Feature Engineering 래퍼(wrapper)의 성능을 벤치마킹하는 것이 주된 접근법입니다.

- **Performance Highlights**: FEDORA는 기존의 전통적인 방법들에 비해 76.2%의 최대 균형 정확도(maximum balanced accuracy)를 달성하는 등의 우수한 성과를 보였습니다. 또한, 단일 특성을 이용하여 72.8%의 균형 정확도를 달성한 사례도 존재하여, 이 프레임워크의 효율성을 입증했습니다.



### Learning Discontinuous Galerkin Solutions to Elliptic Problems via Small Linear Convolutional Neural Networks (https://arxiv.org/abs/2502.08783)
- **What's New**: 최근 몇 년 동안 딥러닝(deep learning)과 신경망(neural networks)을 이용하여 부분 미분 방정식(partial differential equations, PDEs)을 해결하는 데 대한 관심이 증가하고 있습니다. 그러나 기존의 신경망 기반 방법인 물리 정보 신경망(physics-informed neural networks, PINNs)은 자동 미분(automatic differentiation)과 샘플링에 의존하기 때문에 해석 가능성이 떨어지고 정확도가 낮아지는 문제가 발생합니다. 본 연구에서는 작은 선형 합성곱 신경망(small linear convolutional neural networks)을 사용하여 불연속 갈레르킨(discontinuous Galerkin) 솔루션을 학습하는 두 가지 접근 방식을 제안합니다. 첫 번째 접근 방식은 지도 학습(supervised) 방식으로 레이블 데이터(labelled data)에 의존하는 반면, 두 번째 접근 방식은 비지도 학습(unsupervised) 방식으로 어떤 훈련 데이터도 필요하지 않습니다.

- **Technical Details**: 이러한 방법은 불연속 갈레르킨 방법의 손실 함수(loss function)를 통해 L2 및 H1오차를 측정하며, 작은 선형 합성곱 신경망을 통해 구현됩니다. 초기의 훈련 데이터 없이도 지역 질량 보존(local mass conservation) 속성을 활용할 수 있는 비지도 학습 방법을 제안합니다. 이 두 가지 접근 방식 모두 기존의 수치 기반 신경망에 비해 학습에 필요한 파라미터(parameter)가 현저히 적으며, 정확도는 전통적인 해법과 비슷한 수준을 유지합니다. 따라서 our methods는 다양한 대칭 및 비대칭 내부 페널티 방법에 대한 좋은 초기 추정(initial guess)을 제공할 수 있습니다.

- **Performance Highlights**: 제안된 방법들은 처리가 용이하며 다양한 PDE 문제를 해결하는 데 효율적입니다. 실험 결과를 통해 두 접근 방식 모두 고전적 DG 해법(computational performance에 있어 상당한 성능 향상을 보였습니다. 이들 방법은 향후 전통적인 수치 솔버와 결합하여 계산 성능을 더욱 높일 가능성을 보여줍니다. 또한, 이 연구는 불연속 갈레르킨 접근 방식을 통해 깊은 학습(deep learning) 분야와 전통적 해법의 융합 가능성을 제시함으로써 관련 연구에 중요한 기여를 할 것으로 기대됩니다.



### Recurrent Memory for Online Interdomain Gaussian Processes (https://arxiv.org/abs/2502.08736)
Comments:
          13 pages, 4 figures

- **What's New**: 이 논문은 Online HiPPO Sparse Variational Gaussian Process Regression (OHSGPR)라는 새로운 온라인 가우시안 프로세스 모델을 제안합니다. 이 모델은 시퀀스 데이터의 장기 기억을 포착할 수 있으며, HiPPO 프레임워크를 활용하여 이전 데이터에 대한 기억을 유지합니다. 기존의 온라인 GP 모델에 비해 예측 성능과 계산 효율성에서 뚜렷한 향상을 보입니다.

- **Technical Details**: OHSGPR은 HiPPO의 시계열 변동 직교 투영을 간섭 변수로 해석하여 시간에 의존적인 직교 다항식 기저 함수의 조합으로 구성됩니다. 이는 기존의 Sparse Gaussian Process Regression (SGPR) 방법에서 메모리 손실 문제를 크게 개선하여 장기 시계열 회귀 문제에 적용 가능성을 열어줍니다. 또한, 커널 행렬을 효율적으로 업데이트할 수 있는 방법을 제공하여 온라인 업데이트의 계산 효율성을 높였습니다.

- **Performance Highlights**: OHSGPR의 성능은 시간 시계열 회귀 작업에서 평가되었으며, 기존의 온라인 GP 방법보다 현저히 높은 예측 성능과 계산 효율성을 보여줍니다. 이 모델은 장기 메모리 태스크에서 강한 성능을 발휘하며, HiPPO 프레임워크의 이점을 최대한 활용하여 효과적인 학습을 진행할 수 있습니다.



### New Bounds for Sparse Variational Gaussian Processes (https://arxiv.org/abs/2502.08730)
Comments:
          17 pages, 5 figures

- **What's New**: 이번 논문에서는 Sparse Variational Gaussian Processes(SVGP)의 접근 방식을 개선하여, 훈련 데이터의 수에 따라 추가 파라미터에 의존하는 보다 일반적인 변분 분포를 도입하였습니다. 이러한 새 접근법은 기존의 방법보다 tighter한 evidence lower bound를 제공하며, stochastic optimization에도 적합합니다. 따라서, 기존의 sparse GP 코드로부터 최소한의 수정으로 쉽게 구현할 수 있습니다.

- **Technical Details**: 이 방법은 변분 분포에서 조건부 GP prior인 p(f | u)를 보다 일반적인 조건부 분포 q(f | u)로 대체하여 구성됩니다. 새로운 변분 분포는 N개의 추가 매개변수에 의존하며, 이는 훈련 샘플 수와 동일합니다. 이를 통해 기존 방법보다 더 나은 커버리지 근사를 가능하게 하며, 특히 hyperparameters 학습에 있어 bias를 줄이는 데 효과적입니다.

- **Performance Highlights**: 여러 데이터 세트를 통해 실험한 결과, 제안된 방법이 하이퍼파라미터 학습의 bias를 감소시키고 더 나은 예측 성능을 이끌어낼 수 있음을 보여주었습니다. 또한, 비-Gaussian likelihood에 대한 확장도 마련되어 있어 다양한 실제 적용 가능성을 나타냅니다.



### A Comparative Study of Machine Learning Algorithms for Stock Price Prediction Using Insider Trading Data (https://arxiv.org/abs/2502.08728)
Comments:
          5 pages, accepted to publish

- **What's New**: 이 논문은 내부 거래 정보를 활용하여 주가를 예측하는 여러 기계 학습 알고리즘을 실증적으로 조사합니다. 내부 거래는 시장의 정서를 반영하고 주가 변화의 신호를 제공하므로, 연구의 초점은 여러 알고리즘의 효과성을 비교하고 예측 정확도를 향상시키기 위한 방법론을 탐구하는 데 있습니다. 이 연구는 Tesla 주식 거래 데이터를 사용하여 다양한 알고리즘의 작동을 분석하였습니다.

- **Technical Details**: 본 연구는 내부 거래 데이터에서 중요한 특징을 선택하기 위해 특성 중요도 분석(feature importance analysis)을 사용했습니다. 기계 학습 알고리즘으로는 결정 트리(Decision Trees), 랜덤 포레스트(Random Forests), 서포트 벡터 머신(Support Vector Machines, SVM) 등이 활용되었으며, 각 알고리즘은 특징 크기와 비선형 상관관계를 처리하기 위해 다양한 커널(kernel)을 사용했습니다. K-평균 군집화(K-Means Clustering) 기법도 데이터의 패턴을 찾기 위해 적용되었습니다.

- **Performance Highlights**: 연구 결과, Radial Basis Function(RBF) 커널을 사용한 SVM이 다른 모델에 비해 가장 높은 예측 정확도를 보였습니다. 그러나 이 모델은 처리 시간 측면에서 상당히 높은 요구를 갖고 있어 효율성과 정확도의 균형이 주목되었습니다. 이 연구의 결과는 금융 분석가와 투자자들에게 강력한 알고리즘 선택을 돕고, 투자 전략을 최적화하는 데 기여할 것으로 기대됩니다.



### Scalable Discrete Diffusion Samplers: Combinatorial Optimization and Statistical Physics (https://arxiv.org/abs/2502.08696)
Comments:
          Accepted at ICLR 2025

- **What's New**: 이번 논문에서는 복잡한 비정규화 분포에서 샘플링하는 새로운 방법론이 제시됩니다. Diffusion models의 잠재력은 인식되었지만, 기존 방법에서는 메모리 문제로 인해 한계가 있었습니다. 이를 극복하기 위해 Policy Gradient와 Self-Normalized Neural Importance Sampling (SN-NIS)을 기반으로 한 새로운 훈련 방법을 도입하였습니다.

- **Technical Details**: 제안된 방법은 메모리 효율적인 훈련을 가능하게 하여 비지도형 조합 최적화에서 최첨단 결과를 달성합니다. SN-NIS와 Neural Markov Chain Monte Carlo의 적응을 통해 비편향 샘플링이 가능해지며, 이는 디스크리트 diffusion 모델에서의 혁신적인 응용으로 이어집니다. Ising 모델 벤치마크에서 검증된 결과는 기존의 오토 회귀 접근 방법보다 우수함을 입증합니다.

- **Performance Highlights**: 논문에서 제안하는 방법은 기존의 방법보다 더 많은 과학적 응용 분야에서 효과적으로 활용될 수 있음을 보여주며, 특히 비지도형 조합 최적화 문제에 대한 새로운 접근 가능성을 열어줍니다. 다양한 과학적 응용에 대해 사실상 최대의 우수한 성능을 달성하며, 이전에 정확한 가능성 모델에 국한되었던 범위에서 새로운 길을 개척합니다.



### Efficient Split Learning LSTM Models for FPGA-based Edge IoT Devices (https://arxiv.org/abs/2502.08692)
Comments:
          Accepted for publication at IEEE ICMLCN 2025

- **What's New**: 이 논문에서는 Split Learning (SL) 프레임워크를 실제 환경에서 Field-Programmable Gate Array (FPGA) 기반 IoT 엣지 플랫폼에 배포하는 데 초점을 맞추고 있습니다. SL은 엣지 장치와 중앙 서버 간의 신경망을 분할하여 데이터 프라이버시를 유지하면서 학습 부담을 줄이는 혁신적인 접근 방식으로, 물 자원 모니터링을 위한 Long Short-Term Memory (LSTM) 모델을 활용합니다. 다양한 SL 구성으로 LSTM 모델을 학습하고 배포함으로써 모델 성능을 최적화하고 에너지 소모를 최소화하는 방법을 제시합니다.

- **Technical Details**: SL은 신경망을 두 개의 하위 네트워크로 나누어 엣지 장치에서 초기 레이어를 처리하고 서버에서 나머지 레이어를 처리하는 방식으로 작동합니다. 이 과정에서 IoT 엣지 장치는 로컬 데이터만을 유지하며, 원시 데이터는 네트워크를 통해 전송되지 않습니다. 이 연구에서는 실세계 데이터를 사용하여 LSTM 모델을 훈련하고 최적화하며, FPGA 플랫폼에서 시간 시퀀스 처리의 효율성을 높이는 방법을 보여줍니다.

- **Performance Highlights**: 결과적으로 이 연구는 SL 프레임워크가 물리적 자원 제약 상황에서도 예측 정확도와 저전력 소모, 낮은 지연 시간을 동시에 달성할 수 있음을 입증합니다. 제안된 설계에서 고성능을 유지하면서도 자원 활용 최적화를 추구하는 다양한 트레이드오프가 존재하며, 이로 인해 최적의 성능과 에너지 효율성 간의 균형을 고려해야 함을 강조합니다. 이러한 결과는 환경 모니터링을 위한 IoT 시스템에서 SL의 변혁적 잠재력을 제시합니다.



### Skrr: Skip and Re-use Text Encoder Layers for Memory Efficient Text-to-Image Generation (https://arxiv.org/abs/2502.08690)
- **What's New**: 이번 연구에서는 텍스트-이미지(T2I) 확산 모델에서 텍스트 인코더의 메모리 사용 효율을 높이기 위한 새로운 방법인 Skip and Re-use layers(Skrr)를 제안합니다. 기존의 denoising 모듈과 달리, 텍스트 인코더는 단일 순방향 통과만으로 텍스트 임베딩을 생성하는데, 이는 메모리 사용량이 높아지는 원인으로 작용합니다.

- **Technical Details**: Skrr는 T2I 작업에 맞춰 설계된 가지치기(pruning) 전략으로, transformer 블록 내의 특정 레이어를 선택적으로 건너뛰거나 재사용하여 메모리 소비를 줄입니다. 이 방법은 높은 희소성(sparsity) 수준에서도 텍스트-이미지 생성 품질을 유지하며, 기존의 블록 기반 가지치기 방법을 초월하는 성능을 보여줍니다.

- **Performance Highlights**: 광범위한 실험을 통해 Skrr는 FID, CLIP, DreamSim 및 GenEval 점수 등 여러 평가 메트릭에서 성능을 유지하면서도 메모리 효율성을 극대화하는 것을 입증했습니다. 이로 인해 Skrr는 최신 기술(trade mark)보다 우수한 메모리 효율성을 달성하며, 성능 저하 없이 높은 품질의 이미지를 생성할 수 있습니다.



### Advancing machine fault diagnosis: A detailed examination of convolutional neural networks (https://arxiv.org/abs/2502.08689)
- **What's New**: 이번 논문에서는 기계 고장 진단에 대한 최신의 진전을 다루고 있습니다. 특히, Convolutional Neural Networks (CNNs)의 이론적 기초와 구조적 변화를 심층적으로 검토합니다. CNNs의 강력한 고장 탐지 및 분류 능력은 최근의 데이터 증강(data augmentation)과 전이 학습(transfer learning) 기법을 통해 더욱 발전하고 있습니다.

- **Technical Details**: 논문은 CNN이 다양한 고장 유형 및 복잡한 데이터에 어떻게 적용되는지를 강조하며, CNN 아키텍처의 변형에 대한 이론적인 배경을 제공합니다. CNN의 강점과 한계를 분석하며, 특정 운영 환경에서의 효과성을 검증합니다. 또한 하이브리드 아키텍처(hybrid architectures)에 대해서도 논의합니다.

- **Performance Highlights**: CNN 기반 고장 진단 기술의 신뢰성과 능동적 접근 방식을 개선하는 데 있어 향후 연구 방향과 잠재적인 과제를 조명합니다. 이를 통해 기계 고장 진단에 대한 지속적인 발전이 이루어질 것으로 기대하고 있습니다. 특히, CNN을 활용한 고장 진단 시스템의 효율성을 높이는 것이 중요한 목표로 설정되어 있습니다.



### EEG Artifact Detection and Correction with Deep Autoencoders (https://arxiv.org/abs/2502.08686)
- **What's New**: 본 연구에서는 EEG 신호의 아티팩트 감지 및 교정을 위해 새로운 LSTM 기반의 오토인코더인 LSTEEG를 제안합니다. LSTEEG는 순차적 EEG 데이터의 비선형 종속성을 캡처하며, 기존의 다른 최첨단 합성곱 오토인코더에 비해 우수한 성능을 보입니다. 이 방법론은 오토인코더의 잠재 공간의 해석 가능성과 유용성을 향상시켜 EEG 신호의 아티팩트를 자동으로 제거하는 데이터 기반 접근 방식을 가능하게 합니다.

- **Technical Details**: LSTEEG는 각 EEG 세그먼트를 동질적인 저차원 잠재 공간으로 인코딩하는 방식을 채택합니다. LSTM 층을 활용하여 시퀀스 데이터에서 장기 비선형 종속성을 포착하며, 아티팩트 제거 전후 과정에서 특정 EEG 세그먼트를 식별하기 위해 이상 감지 기법을 통합합니다. 연구팀은 LEMON 데이터셋을 사용하여 오토인코더의 성능 평가를 위해 60/20/20 비율로 데이터 세트를 나누어 훈련, 검증 및 테스트를 수행합니다.

- **Performance Highlights**: LSTEEG는 아티팩트 감지 및 교정 작업에서 기존의 최첨단 합성곱 AEs와 비교하여 경쟁력 있는 성능을 발휘합니다. 연구 결과는 LSTEEG가 낮은 차원의 의미 있는 표현을 학습함으로써 신호 해석의 새로운 가능성을 열어준다고 강조합니다. 이러한 표현은 EEG 신호를 이해하고, 후속 작업을 위한 신경생리학적 특징을 추출하는 데 도움을 줄 수 있습니다.



### Beyond Models! Explainable Data Valuation and Metric Adaption for Recommendation (https://arxiv.org/abs/2502.08685)
- **What's New**: 이번 연구에서는 추천 시스템에서의 데이터 품질을 평가하기 위해 설명 가능하고 다목적의 데이터 가치 평가 프레임워크(DVR)를 제안합니다. 기존 방법들은 블랙박스 설계를 사용하여 데이터 평가에 대한 투명성과 해석 가능성이 부족했습니다. DVR은 게임 이론적 관점에서 Shapley 가치를 계산함으로써 데이터의 질을 평가하고, 다양한 모델 아키텍처 및 평가 메트릭에 맞춰 데이터 활용의 효율성을 향상시킵니다.

- **Technical Details**: DVR 프레임워크는 설명 가능한 데이터 가치 평가를 위해 데이터 가치 측정기를 도입하여, 데이터가 모델 성능에 미치는 기여도를 평가합니다. 해당 데이터 가치 측정기는 데이터와 모델 성능을 플레이어와 결과로 간주하여 Shapley 가치를 계산합니다. 또한, 비차별적 메트릭을 포함한 여러 평가 메트릭에 맞추기 위해 강화를 기반으로 한 메트릭 어댑터를 개발하여 최적의 메트릭 성능을 유도합니다.

- **Performance Highlights**: 다양한 기준에서 실시된 광범위한 실험 결과, DVR 프레임워크는 현재의 추천 알고리즘 성능을 크게 향상시켰습니다. 특히, NDCG 메트릭에서는 기존 방법 대비 최대 34.7%의 성능 개선을 달성했습니다. 이러한 성과는 추천 시스템의 랭킹 정확도, 다양성 및 공정성 등 여러 면에서의 향상으로 이어집니다.



### Self-Evaluation for Job-Shop Scheduling (https://arxiv.org/abs/2502.08684)
- **What's New**: 본 논문은 조합 최적화 문제(Combinatorial Optimization Problems, COPs)에 대한 새로운 접근 방식을 제안합니다. 기존의 단계별 방법에서 벗어나 작업 할당의 부분 집합을 생성하고 평가하는 방식으로, Neural Combinatorial Optimization 방법론을 살펴봅니다. 이를 통해 오류 축적 문제를 완화하고, 고품질의 최종 결과를 도출하겠습니다.

- **Technical Details**: 제안된 프레임워크는 이종 그래프 신경망(Heterogeneous Graph Neural Network, HGNN)과 변환기(Transformer)를 결합한 정책 모델과 자기 평가 모델로 구성됩니다. 정책 모델은 가능한 행동에 확률을 할당하며, 자기 평가 모델은 이 행동 집합의 품질을 전반적으로 평가합니다. 이는 조합 최적화 문제를 마르코프 결정 프로세스(Markov Decision Process, MDP)로 재정의하여, 단일 행동 공간에서 부분 서브시퀀스 기반 공간으로 전환합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존의 최첨단 방법들과 비교해 상당한 최적화 향상을 보여주었습니다. 특히, Google의 CP-SAT 솔버와 비교했을 때, 우리가 제안한 접근 방식이 더 높은 성능과 더 낮은 계산 비용으로 더 나은 결과를 도출했음을 확인했습니다. 이 결과는 조합 최적화 문제를 해결하는 데 있어 우리의 방법론이 강력한 효율성을 발휘함을 보여줍니다.



### A Deep Learning approach for parametrized and time dependent Partial Differential Equations using Dimensionality Reduction and Neural ODEs (https://arxiv.org/abs/2502.08683)
- **What's New**: 이 논문에서는 비선형 시간 의존 매개변수를 가진 부분 미분 방정식(Partial Differential Equations, PDE)을 해결하기 위한 자기 회귀 기반의 데이터 드리븐 방법을 제안합니다. 특히, 기존의 전통적인 수치해법에 대한 유사성을 기반으로 하여 차원 축소(Dimensionality Reduction, DR)와 신경망 보통 미분 방정식(Neural Ordinary Differential Equations, NODEs)을 결합하고 있습니다. 이러한 접근 방식은 고차원의 PDE 해(space)를 저차원으로 매핑(고충실도에서 저충실도로)하고, 이를 통해 계산 부담을 줄일 수 있음을 강조하고 있습니다.

- **Technical Details**: 이 모델은 두 가지 매핑을 정의하며, 하나는 오토인코더(AutoEncoder)를 이용하여 높은 차원의 PDE 해 공간과 저차원의 잠재적 공간 간의 거의 전단사(bijective) 매핑을 형성합니다. 또 다른 매핑은 잠재적 벡터가 주어진 시간 t에서 다음 잠재적 벡터 (t+1)로 이동하는 과정을 NODE로 모델링합니다. 이 과정을 통해 시스템의 다양한 초기 조건과 매개변수를 변화시킬 수 있으며, DR과 NODEs의 이점을 동시에 활용합니다.

- **Performance Highlights**: 제안된 방법은 더 작은 시스템을 해결함으로써 계산 속도를 증가시키고, 예측 정확도를 높여 전통적인 방법들보다 훨씬 가볍고 빠른 딥러닝 모델을 제공합니다. 연구 결과에 따르면, 기존의 복잡한 아키텍처를 사용하는 것보다 DR을 활용하여 더 효과적이고 정확한 예측이 가능하다는 것을 증명합니다. 이를 통해 불확실성 정량화(uncertainty quantification)나 설계 최적화(design optimization)와 같은 반복 계산에 적합한 해법을 제공합니다.



### On the Role of Pre-trained Embeddings in Binary Code Analysis (https://arxiv.org/abs/2502.08682)
- **What's New**: 이 논문은 바이너리 코드 분석에서 사전 훈련된 임베딩(embeddings)의 역할을 비판적으로 탐구하고 있습니다. 특히, 어셈블리 코드에 대한 최근 임베딩을 체계적으로 평가하여, 라벨 데이터의 양에 따라 달라지는 성능을 확인합니다. 이 연구는 기존 연구에서 언급된 차이가 실질적으로 미미하다는 것을 발견했습니다.

- **Technical Details**: 연구는 1.2백만 개의 함수가 포함된 Debian 배포판의 말뭉치(corpus)를 사용하여 다섯 가지 하위 작업(downstream tasks)에 대해 최근 임베딩의 성능을 분석합니다. 또한, 함수 경계(function boundaries), 최적화 수준(optimization levels), 인자 유형(argument types)과 같은 레이블 데이터가 쉽게 생성될 수 있음을 강조하고 있습니다. 이러한 분석을 통해, 엔드 투 엔드 러닝(end-to-end learning)이 평균적으로 가장 좋은 성능을 발휘한다는 결과를 도출했습니다.

- **Performance Highlights**: 연구 결과, 충분한 라벨 데이터가 주어졌을 때 여러 임베딩의 성능이 유사하게 나타났습니다. 특정한 임베딩의 필요성이 의문시되며, 언제 임베딩이 이점을 제공하고 언제 엔드 투 엔드 러닝이 더 바람직한지를 도출하기 위한 가이드라인이 제시됩니다. 이러한 결과는 바이너리 코드 분석을 위한 새로운 연구 방향을 제시합니다.



### Mathematical Reasoning in Large Language Models: Assessing Logical and Arithmetic Errors across Wide Numerical Ranges (https://arxiv.org/abs/2502.08680)
- **What's New**: 본 연구는 GSM8K에서 파생된 데이터셋 생성기인 GSM-Ranges를 도입하여, 수학 문제의 숫자 값을 체계적으로 변경함으로써 다양한 숫자 범위에서 모델의 강건성을 평가하고자 합니다. 이는 실제 문제 해결에 필요한 보다 다양한 수치 범위를 반영하며, LLM의 논리적 오류와 비논리적 오류를 구별하는 새로운 채점 방법론도 제공합니다. 이러한 접근은 LLM의 수학적 추론 능력에 대한 보다 포괄적인 평가를 가능하게 하며, 향후 연구 방향에도 중요한 인사이트를 제공합니다.

- **Technical Details**: GSM-Ranges는 GSM8K 데이터셋을 기반으로 하여, 문제 내 숫자를 다양한 규모로 섞어주는 여섯 가지의 왜곡 수준을 적용합니다. 새로운 채점 방법론은 GPT-4o 모델을 사용하여 LLM이 생성한 응답을 Python 코드로 변환하고, 이 코드를 실행하여 비논리적 오류를 격리시킵니다. 최종 결과는 기초적 논리 오류와 계산적 오류를 구분하여, LLM의 수학적 추론 과정을 보다 세밀하게 평가하는 데 기여합니다.

- **Performance Highlights**: 모델의 숫자 복잡성이 증가함에 따라 논리적 오류율이 최대 14%포인트까지 증가하는 경향을 관찰했습니다. 또한, LLM이 단독 산술 작업에서는 높은 정확도를 보이지만, 단어 문제로 계산이 포함될 경우 성능이 크게 저하되는 것을 나타냅니다. 이러한 결과는 LLM의 수학적 추론 능력에 대한 포괄적인 이해를 제공하며, 다양한 수학 문제 처리 과정에서의 강건성을 향상시키기 위한 연구의 초석이 될 수 있습니다.



### Deep Learning-Driven Malware Classification with API Call Sequence Analysis and Concept Drift Handling (https://arxiv.org/abs/2502.08679)
- **What's New**: 이 연구에서는 동적 환경에서의 맬웨어(malware) 분류 시 발생하는 개념 드리프트(concept drift)를 해결하기 위해 유전자 알고리즘(genetic algorithm)을 활용한 심층 학습(deep learning) 프레임워크를 제안합니다. 이는 맬웨어 데이터의 통계적 특성이 시간에 따라 변화하는 도전 과제를 다루는 데 효과적입니다. 새로운 접근 방식은 유전자 알고리즘을 통해 지속적으로 모델을 개선하여 진화하는 맬웨어 위협에 대한 견고함을 보장합니다.

- **Technical Details**: 제안된 모델은 맬웨어 샘플을 행동 및 내재적 특성에 따라 분류하며, 유전자 알고리즘을 통해 개념 드리프트를 관리합니다. 본 연구에서는 n-gram API 호출을 사용하여 패턴을 식별하고 맬웨어 샘플의 행동을 분석하는 방식으로 분류 능력을 향상합니다. 또한, 심층 학습 기반의 맬웨어 분류 프레임워크를 개인화된 데이터 전처리, 특성 선택, 분류 단계를 포함하여 설명합니다.

- **Performance Highlights**: 실험 결과, hybrid 방법론이 전통적인 정적 모델보다 맬웨어 분류 성능과 적응성을 크게 향상시키는 것으로 나타났습니다. 본 연구에서 제안한 접근 방식은 빠르게 변화하는 사이버 보안 환경에서 실시간 맬웨어 분류에 대한 유망한 솔루션을 제공합니다. 또한 다양한 심층 학습 알고리즘과 유전자 알고리즘을 조합한 결과를 면밀히 분석한 내용을 포함합니다.



### DexTrack: Towards Generalizable Neural Tracking Control for Dexterous Manipulation from Human References (https://arxiv.org/abs/2502.09614)
Comments:
          Accepted to ICLR 2025. Website: this https URL Code: this https URL Video: this https URL

- **What's New**: 이번 논문에서는 인간의 포든 동작을 기반으로 한 일반화 가능한 신경 추적 제어기를 개발하는 문제를 다룹니다. 이 제어기는 다양하고 복잡한 물체를 조작하기 위해 로봇 손을 제어하는 것을 목표로 하며, 기존 강화 학습 및 궤적 최적화 방법의 한계를 극복하기 위해 설계되었습니다. 우리는 대규모의 성공적인 로봇 추적 시연을 통해 신경 제어기를 학습시키는 접근법을 소개하며, 이를 통해 성능과 성공적인 시연의 수와 질을 모두 향상시킬 수 있습니다.

- **Technical Details**: 우리의 연구는 DexTrack이라는 새로운 신경 추적 제어기를 제안합니다. 이 제어기는 인간 손-객체 조작 경로를 로봇에 맞추어 최적화하고, 성공적인 로봇 추적 시연과 함께 교대로 훈련됩니다. 강화 학습과 모방 학습 기법을 통합하여 다양한 상황에서 제어기의 성능을 향상시키고, 동적 환경에서도 강력한 적응력을 유지합니다.

- **Performance Highlights**: DexTrack은 두 개의 데이터 세트에서 기존 방법과 비교하여 뛰어난 성능을 발휘하였으며, 이전 방법보다 10% 이상의 성공률을 기록하였습니다. 시뮬레이터인 Isaac Gym 및 실제 환경에서의 평가를 통해 광범위한 조작 추적 작업을 성공적으로 수행하고, 예기치 않은 상황에서도 회복 능력이 있음을 입증했습니다. 이 연구는 반복적인 시연 채굴을 통해 발전하는 일반화 가능성이 큰 신경 추적 제어기를 제시합니다.



### Human-LLM Coevolution: Evidence from Academic Writing (https://arxiv.org/abs/2502.09606)
- **What's New**: 본 논문은 아르Xiv 논문의 초록에서 통계 분석을 통해 ChatGPT가 과도하게 사용되었던 단어들의 사용 빈도가 감소하고 있음을 발견했습니다. "delve"와 같은 단어는 2024년 초 이후로 현저히 줄어들고 있으며, 반면에 "significant"와 같은 단어는 증가하는 경향을 보이고 있습니다. 이러한 변화는 학문적 글쓰기에서 대규모 언어 모델(LLMs)의 사용 방식이 변화하고 있음을 시사합니다.

- **Technical Details**: 본 연구에서는 2018년부터 2024년까지 제출된 아르Xiv 논문의 초록 데이터에서 단어 빈도 분석을 수행했습니다. 연구자는 특정 단어의 사용 빈도를 월별로 계산하고 10,000개의 초록 당으로 정규화했습니다. 특히 2024년 4월 이후 LLM 스타일의 단어들이 감소하기 시작했고, "significant"와 같은 단어들은 상대적으로 잘 사용되면서 여전히 증가 추세를 보이는 것으로 나타났습니다.

- **Performance Highlights**: 연구에서 LLMs가 학문적 글쓰기에서 미친 영향을 파악하는 것 외에도, 기계 생성 텍스트(MGT)의 탐지에는 여러 도전 과제가 제기되었습니다. 감지기의 성능은 사용되는 LLM 모델과 텍스트 유형에 따라 달라지며, 단순한 이진 분류 체계로는 현실 세계의 복잡한 상황을 포괄하기 어렵습니다. 이 논문은 LLMs에 의해 생성된 텍스트의 영향을 통계적으로 측정하는 것이 더욱 실용적인 선택임을 강조하고 있습니다.



### SelfCite: Self-Supervised Alignment for Context Attribution in Large Language Models (https://arxiv.org/abs/2502.09604)
Comments:
          Implementation available at this https URL

- **What's New**: SelfCite는 LLMs(대형 언어 모델)의 응답에서 고품질, 세밀한 인용(citations)을 생성하기 위한 새로운 자기 감독(self-supervised) 접근 방식을 소개합니다. 본 방법은 비싼 주석 작업 없이도 LLM이 제공하는 보상 신호를 활용하여 인용의 품질을 향상시키도록 설계되었습니다. SelfCite는 모델의 내부 피드백 메커니즘을 활용하여 인용의 필요성과 충분성을 평가하는 과정을 포함합니다.

- **Technical Details**: SelfCite의 핵심은 context ablation 기술을 사용하여 LLM이 생성하는 응답의 문맥에서 인용이 필요하거나 충분한지를 평가하는 것입니다. 구체적으로, 인용된 텍스트를 제거했을 때 응답의 확률이 낮아지면 해당 인용이 필요하다고 판단하고, 인용된 텍스트만 남겼을 때 높은 확률을 유지한다면 충분하다고 간주합니다. 이러한 자가 평가 메커니즘을 통해 SelfCite는 주석 과정 없이 보상 신호를 계산하게 됩니다.

- **Performance Highlights**: SelfCite는 LongBench-Cite 벤치마크에서 인용의 F1 점수를 최대 5.3점 높이며 LLM의 자동 인용 품질 향상 가능성을 보여줍니다. 또한, SimPO를 통해 선호 최적화(preference optimization)를 적용하여 개선된 인용 품질을 유지하면서 이전 최첨단 기법을 초과한 성능을 달성했습니다. 이 접근법은 LLM의 자기 보상을 통해 인용 품질을 더욱 개선하는 방향을 제시합니다.



### Optimizing GPT for Video Understanding: Zero-Shot Performance and Prompt Engineering (https://arxiv.org/abs/2502.09573)
- **What's New**: 이번 연구에서는 비디오 콘텐츠 분류에서 GPT 기반 모델을 활용하여 제로샷(zero-shot) 분류를 최적화하는 새로운 접근 방식을 제시합니다. 복잡한 정책을 단순화함으로써 허위 부정(false negatives)을 감소시키고, 새롭게 도입한 분해-집계(decomposition-aggregation) 기반 프롬프트 엔지니어링 기법이 기존의 단일 프롬프트 방법들보다 뛰어난 성능을 보여줍니다. 이러한 실험은 실제 산업 문제들을 대상으로 진행되어, 비디오 분류 시스템 개선을 위한 효과적이고 확장 가능한 솔루션을 제공합니다.

- **Technical Details**: 연구에서는 GPT-4의 멀티모달(multimodal) 이해 능력을 평가하며, TikTok 비디오 콘텐츠를 다양한 기준에 따라 분류하는 가능성을 탐구합니다. GPT-4의 제로샷과 피-샷(few-shot) 능력이 산업별 문제에 효과적으로 적용될 수 있는지를 확인하며, 정책 개선과 프롬프트 엔지니어링이 GPT-4 성능에 미치는 영향을 탐색합니다. GPT-4는 실험에서 기존의 이종 분류 모델들과 비교했을 때 여러 카테고리에서 비슷한 수준의 성능을 보였으나, 클릭베이트(clickbait) 비디오와 같은 복잡한 분류에서는 어려움을 겪었습니다.

- **Performance Highlights**: 연구 결과, 복잡한 정책 프롬프트를 간소화하면 허위 부정(false negatives)을 줄이면서 비디오 분류 정확도를 높이는 데 기여합니다. 또한 클릭베이트 탐지와 같은 작업을 하위 카테고리로 나누는 프롬프트 엔지니어링이 상당한 성능 향상을 가져오는 것으로 나타났습니다. 이 발견은 진정한 산업 데이터셋을 기반으로 한 실험을 통해 이루어졌으며, 이는 비디오 분류 작업에 실질적으로 적용 가능하다는 점에서 높이 평가받습니다.



### Zero-shot generation of synthetic neurosurgical data with large language models (https://arxiv.org/abs/2502.09566)
Comments:
          13 pages, 4 figures, 4 tables

- **What's New**: 이 연구는 GPT-4o라는 대형 언어 모델을 사용하여 신경외과 데이터를 생성하는 새로운 접근 방식을 소개합니다. 이는 데이터 접근성 제한 문제를 해결하기 위한 시도로, 실제 데이터에 대한 접근이 어려운 상황에서도 유효한 합성 데이터를 생성할 수 있음을 보여줍니다. 특히, 기존의 conditional tabular generative adversarial network (CTGAN)와 비교하여 더욱 높은 성능을 보였습니다.

- **Technical Details**: 연구는 신경외과 데이터를 평가하기 위해 GPT-4o를 사용하여 합성 데이터 세트를 생성하고, 이를 CTGAN과 비교하는 방식으로 진행되었습니다. 데이터 평가에는 평균, 비율, 분포 및 이변량 상관관계(fidelity, utility, privacy) 측정이 포함되었습니다. 이 모델은 논문의 표본과 같은 크기의 데이터베이스를 생성하며, 데이터의 통계적 속성을 유지합니다.

- **Performance Highlights**: GPT-4o 모델에 의해 생성된 합성 데이터는 CTGAN의 성능을 초과하거나 동등한 결과를 보이며, 실제 환자의 기록에 노출되지 않고도 높은 신뢰도를 나타냈습니다. 또한, 이 데이터로 훈련된 머신러닝 분류기는 F1 점수 0.706을 기록하며, 기존 CTGAN으로 훈련된 경우와 유사한 성능을 발휘했습니다. 이러한 결과는 신경외과 연구에서 작은 표본 크기를 가진 임상 데이터를 효과적으로 증대시킬 수 있는 가능성을 제시합니다.



### SyntheticPop: Attacking Speaker Verification Systems With Synthetic VoicePops (https://arxiv.org/abs/2502.09553)
- **What's New**: 이 논문에서는 Voice Authentication (VA) 시스템에 대한 새로운 공격 방법인 SyntheticPop을 제안합니다. SyntheticPop은 합성된 '팝' 소음을 변조된 오디오 샘플에 삽입하여 VA+VoicePop 시스템의 음소 인식 기능을 겨냥합니다. 이 방법은 VA+VoicePop이 다양한 공격, 특히 논리적 공격 및 적대적 공격에 얼마나 취약한지를 실제로 보여줍니다.

- **Technical Details**: VA+VoicePop 시스템은 독특한 발음의 음소(phoneme) 인식을 사용하여 사용자를 인증합니다. 이 시스템은 GFCC(Gammatone Frequency Cepstral Coefficient)를 려용하여 목소리 샘플을 더욱 철저히 분석하며, STFT(Short Time Fourier Transform)를 사용하여 팝 소음이 발생할 때의 짧은 에너지를 캡처합니다. 그러나 VA+VoicePop은 오직 팝 소음 감지에만 의존하지 않으며, 이러한 특정 메커니즘은 일부 상황에서 효과적이지 않을 수 있습니다.

- **Performance Highlights**: VA+VoicePop 시스템은 정상 조건 하에서 69%의 정확도를 달성하지만, 기준 레이블 전환 공격을 받을 경우 37%로 감소하고 SyntheticPop 공격 대상이 되었을 때는 14%로 떨어집니다. 이는 SyntheticPop 공격이 VA+VoicePop 시스템의 성능에 미치는 영향을 심각하게 입증합니다. 전체 학습 데이터셋의 20%를 손상시킨 상태에서 공격 성공률은 95% 이상에 달했습니다.



### Fast Tensor Completion via Approximate Richardson Iteration (https://arxiv.org/abs/2502.09534)
Comments:
          20 pages, 4 figures

- **What's New**: 텐서 완성(tensor completion, TC)을 저랭크 텐서 분해(low-rank tensor decomposition, TD) 관점에서 연구합니다. 본 연구에서는 TC 회귀 문제에서 구조적 특성이 사라지는 문제를 해결하기 위해, 이러한 TD 회귀 알고리즘을 블랙박스 서브루틴으로 사용하는 리프팅(lifting) 접근법을 제안했습니다. 이를 통해 하위 선형 시간(sublinear-time) 방법이 가능해졌습니다.

- **Technical Details**: 우리의 알고리즘은 근사 리처드슨(iteration) 방법에 기반하여 TC 회귀 문제를 해결하며, 최적화 과정에서 알맞은 패널티 항을 포함한 손실 함수와 정규화 항을 사용하는 것을 목표로 합니다. 특히, 각 스텝에서 CP(completion) 완성을 위해 기존 TD 방법을 활용하여 연산을 진행합니다. 연구에서 사용된 tensor는 저랭크 터커(Tucker) 분해를 기반으로 한 텐서를 포함합니다.

- **Performance Highlights**: 실제 데이터에 대한 실험 결과, 제안된 알고리즘은 CP 완성을 위한 직접 방법보다 최대 100배 빠른 실행 시간을 보였습니다. 이는 기존 방법들이 가지는 비효율성을 해결하기 위한 중요한 단초가 되어 줍니다. 결과적으로, 구조적 TD 알고리즘의 활용을 통해 TC 문제 해결의 효율성을 크게 향상시켰습니다.



### Cracking the Code: Enhancing Development finance understanding with artificial intelligenc (https://arxiv.org/abs/2502.09495)
- **What's New**: 이 논문은 개발 프로젝트 분석의 중요성을 강조하며, 기부자 지원 전략, 수혜국의 우선순위 및 개발 재정 역량을 평가하는 데 도움을 줍니다. 연구는 OECD의 Creditor Reporting System (CRS) 데이터셋의 한계를 극복하고자 새로운 접근 방식을 도입하였습니다. 특히, 프로젝트의 목적을 조명하는 데 필요한 정보가 부족한 현 상황에서, 머신러닝(Machine Learning) 기법을 활용하여 프로젝트를 분류하고 라벨링하는 방법을 제시합니다.

- **Technical Details**: 본 연구는 자연어처리(Natural Language Processing, NLP) 기법과 BERTopic이라는 파이썬 주제 모델링(topic modeling) 기법을 결합하여 개발 프로젝트의 서술 설명을 기반으로 프로젝트를 클러스터링(cluster)하고 라벨링(label)합니다. 이를 통해 개발 재정의 기존이지만 숨겨진 주제(topic)를 드러내는 새로운 통찰력을 제공합니다. 이러한 방법론은 기부자 우선순위를 이해하고 공공 및 민간 프로젝트 서술을 분석하는 데 유용합니다.

- **Performance Highlights**: 연구 결과는 OECD CRS 데이터셋에서 제공하는 방대한 프로젝트 내러티브(narratives)를 효과적으로 활용하며, 기부자들의 재정 지원과 프로젝트 목적 사이의 관계를 명확히 합니다. 인공지능(AI) 기술을 통해 제공된 이 새로운 접근법은 개발 자금의 분포를 보다 잘 이해할 수 있는 기회를 제공하고, 관련 데이터 분석에 대한 새로운 시각을 제시합니다.



### Communicating Likelihoods with Normalising Flows (https://arxiv.org/abs/2502.09494)
Comments:
          4 pages + references, 1 figure

- **What's New**: 이 논문에서는 고에너지 물리학에서 실험 결과를 재해석하는 데 필수적인 새로운 기계 학습 기반 처리 흐름을 제안합니다. 기존 방법들과 차별화된 점은 표본을 통해 학습한 likelihood(우도)를 엄격한 통계적 검정을 통해 검증하는 것입니다. 이 접근법은 실험적이고 현상론적 likelihood를 신뢰성 있게 전달할 수 있게 해 줍니다.

- **Technical Details**: 이 연구는 Likelihood Models(LMs)의 구축, 검증 및 저장을 위한 단계별 워크플로우를 제안합니다. 특히 nabu라는 오픈 소스 소프트웨어를 통해 Python 구현을 제공합니다. LMs의 조합을 통해 하이엔드 물리학 문제들을 해결하고 높은 차원의 파라미터 공간에서도 활용될 수 있도록 설계되어 있습니다.

- **Performance Highlights**: 제시된 사례 연구를 통해 이 방법의 효과성을 입증하며, LMs의 빠른 파라미터 추정 및 데이터 생성 능력을 강조합니다. NFs(Normalizing Flows)를 활용하여 실험 데이터를 압축되고 해석 가능한 형식으로 공유할 수 있는 가능성을 보여줍니다. 이를 통해 연구자들이 실험 데이터를 쉽게 재해석하고 모델 비교를 원활히 진행할 수 있습니다.



### Objective quantification of mood states using large language models (https://arxiv.org/abs/2502.09487)
Comments:
          main text - 9 pages, 5 figures;

- **What's New**: 이 연구는 감정적 상태와 행동 간의 관계를 규명하기 위해 대형 언어 모델(LLMs)을 활용하여 정신 상태를 정량화하는 새로운 방법론을 제시합니다. 연구진은 넷 상에서 모집된 422명의 참여자를 통해 LLM인 Mistral-7B-OpenOrca가 우울증 질문지에 대한 개방형 응답에 대한 응답을 분석했습니다. 이 접근 방식은 질병 기반 치료와 조정뿐만 아니라 정신 건강 문제를 이해하는데 기여할 수 있는 잠재력을 가지고 있습니다.

- **Technical Details**: 연구에서는 표준화된 자가 보고 정신의학 질문지를 사용하여 참여자들의 우울증 상태를 조사했습니다. 각 질문지는 그녀의 정신 상태를 나타내는 답변의 일관성을 유지하며, LLM의 숨겨진 상태를 통해 우울증 관련 특성들을 식별하여 예측력을 검증했습니다. 이를 위해 참가자들은 PHQ-9 질문지를 오픈형 질문으로 재구성된 다양한 질문에 대해 응답했습니다.

- **Performance Highlights**: LLM의 응답은 참가자들의 개방형 답변으로부터 도출된 다중 선택 질문과 강한 상관관계(r: 0.52-0.84)를 보였으며, 이는 LLM이 우울한 정서를 정량화 할 수 있는 가능성을 시사합니다. 연구에서는 LLM의 특정 서브스페이스가 참가자들의 우울증 지수와 정서적 고통 요인 점수 및 자살 위험성의 예측을 제공할 수 있음을 발견했습니다. 이러한 결과는 LLM이 정신 상태를 정량화하는 데 있어 유용한 도구가 될 수 있음을 보여줍니다.



### Assessing Generative AI value in a public sector context: evidence from a field experimen (https://arxiv.org/abs/2502.09479)
- **What's New**: 이 연구는 Generative AI(Gen AI)가 공공 부문에서의 복잡한 지식 기반 작업의 생산성 향상에 미치는 영향을 조사합니다. 실험 결과, 문서 이해와 데이터 분석 관련 작업에서 Gen AI 사용의 효과가 혼재되어 나타났습니다. 문서 작업에서는 Gen AI 사용으로 답변 품질이 17% 향상되고, 작업 완료 시간이 34% 단축되었습니다. 반면, 데이터 작업에서는 품질 점수가 12% 감소하며 주요 완료 시간에 유의미한 차이가 없었습니다.

- **Technical Details**: 연구는 중앙은행 및 금융 규제 기관에서의 Gen AI 가치를 평가하기 위해 사전 등록된 무작위 대조 실험을 수행했습니다. 143명 직원이 Documents와 Data 두 개의 대표 작업을 통해 평가되었습니다. 참가자들은 사전 평가를 통해 각 작업의 기준 성과를 설정한 후, Gen AI 도구 사용 여부에 따라 치료군과 대조군으로 무작위 배정되었습니다. 문서 작업의 경우, 인간 평가자에 의해 품질 점수가 평가되었습니다.

- **Performance Highlights**: 문서 작업의 경우, Gen AI를 사용한 치료군은 품질 점수가 17% 향상되었고, 평균 작업 완료 시간은 34% 개선되었습니다. 그러나 데이터 작업에서는 Gen AI 사용에 의해 품질 점수가 12% 감소하며, 주요 완료 시간에 변경이 없습니다. 이러한 결과는 Gen AI의 이점이 특정 작업 및 응답자에 따라 달라질 수 있음을 시사합니다.



### DiffRenderGAN: Addressing Training Data Scarcity in Deep Segmentation Networks for Quantitative Nanomaterial Analysis through Differentiable Rendering and Generative Modelling (https://arxiv.org/abs/2502.09477)
- **What's New**: 본 논문에서는 DiffRenderGAN이라는 새로운 생성 모델을 소개합니다. 이 모델은 Generative Adversarial Network (GAN) 프레임워크에 차별적인 렌더러(differentiable renderer)를 통합하여 비주석(real microscopy images) 실 이미지로부터 주석이 달린 합성 나노 입자 이미지를 생성할 수 있도록 설계되었습니다. 이 접근 방식은 기존의 합성 데이터 방법보다 수동 개입을 줄이고 분할(segmentation) 성능을 향상시킵니다.

- **Technical Details**: DiffRenderGAN은 3D 나노 입자 모델(meshes)과 위치 및 크기 정보를 담고 있는 변환 행렬을 사용하여 희소한 훈련 데이터를 보완합니다. 이미지 렌더링 과정은 가상의 3D 장면을 현실적인 2D 디지털 이미지로 변환하는 것으로, 이 과정에서 Bidirectional Scattering Distribution Functions (BSDFs)와 같은 재료 특성을 시뮬레이션합니다. 이러한 방법을 통해 합성, 주석이 달린 이미지가 생성되어 분할 네트워크를 효과적으로 훈련할 수 있습니다.

- **Performance Highlights**: DiffRenderGAN은 여러 이온 및 전자 현미경 사례에서 테스트되었으며, 특히 실시간 현미경 이미지에 대해 높은 분할 성능을 달성했습니다. 우리의 연구에서는 합성 데이터의 생성과 기존 방법들과의 성능 비교를 통해, DiffRenderGAN이 기존 방법들의 성능을 초과하거나 동일한 수준의 성능을 나타냈음을 보여주었습니다. 이는 복잡한 나노 물질 시스템의 정량화와 이해를 진전시키는 데 강력한 도구가 될 것으로 기대됩니다.



### A Differentiable Rank-Based Objective For Better Feature Learning (https://arxiv.org/abs/2502.09445)
- **What's New**: 이 논문에서는 데이터에서 특징 학습을 보다 잘 이해하기 위해 기존의 통계적 방법을 활용합니다. 구체적으로는 비모수적 조건 독립성에 기반한 변수 선택 방법인 Feature Ordering by Conditional Independence (FOCI)를 수정하여 새로운 알고리즘인 difFOCI를 제안합니다. difFOCI는 파라메트릭(Parametric)으로 접근 가능하여 기계 학습 문제에서 널리 적용될 수 있습니다.

- **Technical Details**: difFOCI는 조건 의존성의 비모수적 계수를 기반으로 하여 제안된 파라메트릭, 미분 가능 근사치를 사용합니다. 이 알고리즘은 세 가지 맥락에서 제시되며, 변수 선택 방법, 신경망으로 매개변수화된 훈련 가능한 모델, 그리고 불필요한 상관관계를 관리하여 특징 학습을 개선하는 일반적인 신경망 정규화기로 사용됩니다. 이를 통해 다양한 복잡한 문제에 대한 강력한 성능을 보인다고 평가됩니다.

- **Performance Highlights**: difFOCI의 성능은 장난감 데이터셋에서의 단순 변수 선택부터 합성곱 신경망의 밴브 표현 비교에 이르기까지 다양한 영역에서 입증되었습니다. 또한, 공정성 맥락에서의 응용을 통해 민감한 데이터를 활용하지 않고도 분류를 촉진할 수 있는 방법이 제시됩니다. 전체적으로 difFOCI는 특징 선택, 차원 감소 및 공정성 문헌에서 최첨단 성능을 달성하며, 실제 데이터에 대한 광범위한 적용 가능성을 보여줍니다.



### Dual Formulation for Non-Rectangular Lp Robust Markov Decision Processes (https://arxiv.org/abs/2502.09432)
- **What's New**: 이 논문에서는 전통적인 직사각형 모델과는 달리 상태 간 상호 의존성을 포착하는 비직사각형 불확실성 집합을 사용하는 강인한 마르코프 결정 프로세스(RMDP)를 연구합니다. 특히, 복잡성을 피할 수 있는 간단한 구조를 가진 Lp-bounded 불확실성 집합의 강력한 클래스를 확인하였고, 이를 통해 비직사각형 RMDP에 대한 새로운 이중 형식을 도출하였습니다.

- **Technical Details**: 비직사각형 Lp-norm 구속된 불확실성 집합에서 최소화 문제를 sa-rectangular Lp-norm 구속된 불확실성 집합의 합집합에 대한 최소화로 분해했습니다. 원래의 비직사각형 세트로 구성되는 가능한 모든 sa-rectangular Lp-norm 구속 집합에 대해 이 표현을 최소화함으로써, 견고한 정책 평가 알고리즘의 개발을 가능하게 하였습니다.

- **Performance Highlights**: 실험 결과, 제안한 접근 방식은 기존의 완전 탐색(brute-force) 방법과 비교해 상당한 성능 향상을 보여주었으며, 미래에 비직사각형 강인한 MDP에 대한 연구의 기초를 다지는 데 기여할 것으로 기대됩니다.



### On multi-token prediction for efficient LLM inferenc (https://arxiv.org/abs/2502.09419)
- **What's New**: 본 논문에서는 다중 토큰 예측 (MTP) 기능을 NTP를 위해 사전 훈련된 LLM에서 체계적으로 연구하였습니다. 연구 결과, NTP만을 위해 훈련된 LLM도 MTP 기능을 본래적으로 가지고 있음이 확인되었으며, 이는 중간 토큰 확률에 대한 수치적 주변화(numerical marginalization)를 통해 가능하다는 점을 강조했습니다. 또한 MTP 헤드 추가 시의 도전과제와 공동 훈련(joint training)의 중요성에 대해서도 논의하고 있습니다.

- **Technical Details**: 본 연구는 NTP를 위해 훈련된 트랜스포머의 MTP 기능을 다룹니다. MTP는 주어진 부분 입력 시퀀스에 대해 인접한 여러 토큰을 병렬로 생성하는 것을 의미합니다. 논문은 정보 이론적 기준을 사용하여 MTP의 가능성을 탐구하였고, 여러 모델 패밀리 및 크기에 대해 평가를 수행했습니다. 이는 MTP가 기존의 NTP에 비해 더 효율적인 방법임을 시사합니다.

- **Performance Highlights**: MTP는 자가 추측적 디코딩(self-speculative decoding)을 통해 최대 3.6배의 속도 향상을 가져오는 것으로 나타났습니다. 그러나 MTP 헤드를 짜임새 있게 추가하고 공동 훈련을 실시하더라도 성능이 여전히 기존의 기준에는 미치지 못하는 한계가 있음이 발견되었습니다. 이러한 결과는 MTP의 개선 여지가 많은 영역임을 보여주며, 향후 연구에서 다루어야 할 필요성이 있습니다.



### Robot Pouring: Identifying Causes of Spillage and Selecting Alternative Action Parameters Using Probabilistic Actual Causation (https://arxiv.org/abs/2502.09395)
Comments:
          20 pages, 13 figures

- **What's New**: 이번 연구에서는 일상생활에서의 예기치 않은 결과를 개선하기 위한 확률적 실제 인과관계(probabilistic actual causation)의 적용을 탐구합니다. 연구팀은 로봇 쏠기 작업을 통해 인과 그래프(causal graph)와 조건부 확률 분포의 필요성을 설명하며, 이를 통해 어떤 요소가 원인인지 식별하고 대체 행동을 제안하는 방법을 소개합니다. 이러한 분석은 결과를 변화시키기 위한 작업 파라미터를 선택하는 데 활용될 수 있습니다.

- **Technical Details**: 연구의 중심에는 실제 인과관계가 있으며, 이는 특정 사건이 결과를 생성하는 조건을 정의합니다. Halpern과 Pearl이 제안한 실제 인과관계의 정의는 결정론적 시나리오에서 적절하게 적용되고, Fenton-Glynn은 확률적 인과 모델을 위한 확대된 정의를 제안합니다. 이러한 정의는 인과 모델이 유도 비순환 그래프(directed acyclic graph, DAG)로 표현되며, 각 변수와 그 직접 원인 간의 관계를 확률적으로 모델링합니다.

- **Performance Highlights**: 로봇 쏠기 작업에서의 분석 결과는 작업 파라미터가 쏠림(spillage)의 원인이 되는지 여부와 그 변화 방법을 제시합니다. 연구진은 현실적인 시뮬레이션 데이터를 사용하여 인과 모델링 절차를 수행하며, 이 과정에서 도출된 대체 행동과 인간 관찰자가 제안한 대안 솔루션 간의 비교를 통해 실제 인과관계 분석의 효과를 강조합니다. 이러한 접근 방식은 실질적인 작업 환경에서도 응용될 가능성을 보여줍니다.



### SQuARE: Sequential Question Answering Reasoning Engine for Enhanced Chain-of-Thought in Large Language Models (https://arxiv.org/abs/2502.09390)
Comments:
          14 pages

- **What's New**: 이번 논문에서는 SQuARE(Sequential Question Answering Reasoning Engine)라는 새로운 방식의 prompting 기술을 소개하며, 이는 LLMs(대형 언어 모델)가 스스로 질문을 생성하고 답변하게 하여 더 깊이 있는 사고를 촉진하는 것을 목표로 합니다. 기존의 chain-of-thought(CoT) 접근 방식에 비해, SQuARE는 복잡한 질문을 여러 단계를 통해 더 철저히 탐색할 수 있도록 돕습니다.

- **Technical Details**: SQuARE는 모델에게 N개의 하위 질문-답변 쌍을 생성하도록 유도하며, 이를 통해 최종 쿼리에 도달하기 전에 다양한 주제를 탐색합니다. 또, 이 방법은 기존의 다른 prompting 기법들과 결합하기 쉽고, 하위 질문 수(N)를 조정함으로써 탐색의 깊이와 계산 비용을 조절할 수 있습니다. 실험은 Llama와 GPT-4o 모델을 사용하여 다양한 QA 데이터 세트에서 실시되었습니다.

- **Performance Highlights**: SQuARE는 TriviaQA, HotpotQA 및 ASQA 데이터 세트에서 기존 방법들을 초과하는 성과를 보여주었습니다. 특히 Llama 3.2 3B 모델을 사용할 때, TriviaQA에서 6.5% 향상된 88.5%의 성과를 기록하였고, GPT-4o를 이용한 경우에도 경쟁력 있는 결과를 보였습니다. SQuARE는 특히 작은 모델에서 최종 답변 품질을 크게 개선하는 데 기여했습니다.



### Bridging Jensen Gap for Max-Min Group Fairness Optimization in Recommendation (https://arxiv.org/abs/2502.09319)
Comments:
          Accepted in ICLR 2025

- **What's New**: 이번 연구에서는 Group max-min fairness (MMF)를 추천 시스템의 공정성을 보장하는 최적화 목표로 활용하면서 발생하는 이론적 분석을 다룹니다. 연구에 따르면 MMF 제약 조건을 통합하는 것이 샘플 독립성 가정을 위반하여 비선형성을 초래하고, 이로 인해 모델의 수렴 점과 최적 성점 간의 Jensen gap이 발생합니다. 이를 해결하기 위해 FairDual이라는 알고리즘을 제안하며, 이는 독점 최적화 기법을 통해 Jensen gap을 최소화하는 방법입니다.

- **Technical Details**: FairDual 알고리즘은 그룹 MMF 제약을 가지고 있는 최적화 문제를 사실상 그룹 가중 정확도 최적화 문제로 재구성할 수 있음을 이론적으로 입증합니다. 구체적으로, 공정성 제약 문제를 이중으로 포뮬레이션 하여 각 미니 배치 최적화 과정에서 샘플 가중치를 할당합니다. FairDual은 이중 미러 그래디언트 기법을 활용하여 다양한 그룹 손실 가중치를 효율적으로 최적화합니다.

- **Performance Highlights**: 여섯 개의 대규모 추천 시스템 백본 모델을 사용하여 세 개의 공개 데이터 세트로 수행된 실험 결과, FairDual은 정확성과 공정성 측면에서 모든 기준선 모델을 크게 초과하여 Jensen gap을 일관되게 줄였습니다. 또한 랜덤 셔플 기반의 미니 배치 훈련 방식에서도 서브 선형 수렴 속도를 달성할 수 있음을 보여주었습니다.



### Non-asymptotic Analysis of Diffusion Annealed Langevin Monte Carlo for Generative Modelling (https://arxiv.org/abs/2502.09306)
- **What's New**: 본 논문에서는 데이터 분포에 대한 약한 조건 하에서 일반 확산 경로의 이론적 속성과 Langevin Monte Carlo 구현 방식인 Diffusion Annealed Langevin Monte Carlo (DALMC)를 조사합니다. 특히, 이 연구는 확산 모델처럼 데이터 분포의 가우시안 합성이 경로로 정의된 냉각 Langevin 역학에 대한 비비대칭 오차 경계를 제공합니다. 또한 최근 제안된 두꺼운 꼬리 (heavy-tailed) 확산 경로에 대한 이론적인 특성을 처음으로 입증하며, 이는 모집단으로부터의 샘플링과 관련된 이론적 보장을 제공합니다.

- **Technical Details**: 제안된 방법은 스코어 기반 생성 모델 (Score-Based Generative Models, SGM) 의 한 분야로, 간단한 분포(가우시안 또는 Student's t)와 데이터 분포 간의 보간을 통해 생성 모델링을 수행합니다. 확산 모델은 기본적으로 전방 Ornstein-Uhlenbeck (OU) 프로세스를 사용하여 확률 분포의 경로를 구성하며, 이 과정은 무작위 변수를 통해 정의됩니다. 하지만 본 연구에서는 새로운 무작위 변수 정의를 사용하여 매개변수화된 생성 과정을 통한 손실을 최소화하는 방법을 제안합니다.

- **Performance Highlights**: 성능 측면에서는, 제안된 DALMC 방법이 기존의 스코어 기반 확산 접근 방식보다 더 효과적인 데이터 생성 성능을 발휘함을 보여줍니다. 이 방법은 단순한 분포에서 데이터 분포로의 변환의 시간적 과정을 우아하게 다루며, 특히 두꺼운 꼬리 데이터에 대한 생성모델 적용 가능성을 확장합니다. 논문은 이론적 경계와 실제 응용 간의 브릿지를 제공하며, 다양한 생성 작업에서 뛰어난 성과를 나타낼 것으로 기대됩니다.



### Joint Attention Mechanism Learning to Facilitate Opto-physiological Monitoring during Physical Activity (https://arxiv.org/abs/2502.09291)
- **What's New**: 이번 연구에서는 광생리학적 모니터링(optophysiological monitoring)을 위한 실용적인 적대 학습(adversarial learning) 접근법을 제안합니다. 특히, 주목 메커니즘(attention mechanism)을 가진 생성적 적대 신경망(Generative Adversarial Network, GAN)을 활용하여 움직임 잡음(motion noise)을 모델링하고 MA(모션 아티팩트, Motion Artefacts) 제거를 가능하게 합니다. 이 방법은 저강도에서 고강도 신체 활동 중에도 신뢰할 수 있는 생리적 신호 측정을 가능하게 하는 데 초점을 맞추고 있습니다.

- **Technical Details**: AM-GAN은 MA가 포함된 원시 PPG 신호로부터 MA가 제거된 PPG 신호로의 매핑을 학습합니다. 이 모델은 원시 PPG 신호와 삼축 가속도(triaxial acceleration) 신호를 입력으로 사용하여 동작 효과를 직접적으로 반영한 후 MA가 없는 신호를 생성합니다. 모델의 훈련 과정에서 제너레이터(generator)는 MA에 저항하는 신호를 생성하고, 판별기(discriminator)는 생성된 신호와 실제 신호를 구별하는 역할을 합니다.

- **Performance Highlights**: AM-GAN을 통해 MA가 제거된 PPG 신호로부터 도출된 평균 심박수(heart rate)는 IEEE-SPC 데이터셋에서 1.81 beats/min, PPGDalia 데이터셋에서는 3.86 beats/min였습니다. 자체 구축한 LU 데이터셋에서는 평균 심박수와 호흡수(respiratory rate)의 평균 오차가 각각 1.37 beats/min 및 2.49 breaths/min 미만으로 나타났습니다. 연구 결과, AM-GAN은 저강도부터 고강도 신체 활동에 걸쳐 강력한 성능을 보여주었습니다.



### Dynamic Rolling Horizon Optimization for Network-Constrained V2X Value Stacking of Electric Vehicles Under Uncertainties (https://arxiv.org/abs/2502.09290)
Comments:
          21 pages, accepted by Renewable Energy

- **What's New**: 이 연구는 차량-모든 것(V2X) 가치 스태킹 프레임워크를 개발하여 차량-건물(V2B), 차량-그리드(V2G) 및 에너지 거래를 포함하고 있습니다. 이를 통해 주거 커뮤니티의 경제적 이익을 극대화하며 배전 전압을 유지하는 것이 목적입니다. 연구는 예측 오류의 영향을 정량화하고, GRU-EN-TFD라는 Transformer 기반의 예측 모델을 개발하여 여러 수익원을 활용하는 다이나믹 롤링 호라이즌 최적화 방법을 적용합니다.

- **Technical Details**: 본 연구는 정확한 예측 결과를 기반으로 V2X 가치 스태킹 최적화를 위한 동적 롤링 호라이즌 최적화 문제를 공식화합니다. GRU-EN-TFD 모델은 시간당 건물 부하, 태양광 생성, EV 도착을 예측하는 데 사용됩니다. 동적 롤링 호라이즌 접근 방식은 예측 정확도를 개선하고 여러 가치 스트림을 효과적으로 관리할 수 있는 방법을 제공합니다.

- **Performance Highlights**: 실제 데이터를 기반으로 한 시뮬레이션 결과는 V2X 가치 스태킹이 에너지 비용을 크게 줄일 수 있음을 보여줍니다. V2B는 모든 평가된 시장(NEM, ISO-NE 및 NYISO)에서 가장 큰 비용 절감 효과를 나타내었고, GRU-EN-TFD 모델은 벤치마크 예측 모델보다 성능이 뛰어났습니다. EV 도착 예측의 정확성이 V2X 가치 스태킹 성능에 상당한 영향을 미친다는 결과도 도출되었습니다.



### FE-LWS: Refined Image-Text Representations via Decoder Stacking and Fused Encodings for Remote Sensing Image Captioning (https://arxiv.org/abs/2502.09282)
- **What's New**: 이번 논문은 두 개의 서로 다른 CNN 기반 인코더에서 특성을 통합하여 더 나은 원격 감지 이미지 캡션 생성을 위한 새로운 접근 방식을 제안합니다. 또한, 단일 인코더와 두 개의 GRU 스택을 활용하여 캡션 선택성을 정교화하는 방식으로 가중 평균 기법을 도입했습니다. 이러한 접근 방법을 통해 전통적인 Transformer 기반 모델 및 LSTM 기반 기준 모델과 비교해 상당한 성능 향상을 보여주고 있습니다.

- **Technical Details**: 이 논문은 인코더-디코더 구조에서 CNN을 사용하여 이미지 기능을 추출하고, RNN 계열의 GRU를 통해 시퀀스를 생성하는 방법을 사용합니다. 특히, 혁신적인 두 레이어 stacked GRU 디코더를 통해 생성된 캡션의 품질이 개선됩니다. 더불어, 캡션 선택을 향상시키기 위해 Comparison-based Beam Search (CBS) 기술도 도입하였습니다.

- **Performance Highlights**: 제안된 융합 기반 접근법은 캡션 생성에서 기존의 최첨단 모델보다 우수한 성과를 보였습니다. 성능 평가 결과, 본 연구의 방법론은 이미지 캡션 생성의 정확성 및 관련성을 크게 향상시켰으며, 다양한 벤치마크 데이터셋을 기반으로 실험을 통해 그 우수성을 입증하였습니다.



### GEVRM: Goal-Expressive Video Generation Model For Robust Visual Manipulation (https://arxiv.org/abs/2502.09268)
Comments:
          Published as a conference paper at ICLR 2025

- **What's New**: GEVRM은 로봇 비주얼 조작의 강건성을 향상시키기 위해 고안된 새로운 폐쇄 루프 VLA 방법입니다. 이 모델은 내부 모델 제어 원칙(IMC, Internal Model Control)을 통합하여 외부 간섭을 모방하고 강력한 목표를 생성합니다. GEVRM은 텍스트 기반 비디오 생성 모델을 활용하여 로봇의 미래 목표 상태를 보다 표현력 있게 만들고, 관찰된 로봇 상태에 대한 내부 임베딩을 최적화하여 외부 간섭을 효과적으로 구분합니다.

- **Technical Details**: GEVRM의 핵심 구성 요소는 다음과 같습니다. 첫째, 비디오 프레임을 통해 로봇의 상태를 표현하는 목표 생성 기능이 포함되어 있으며, 텍스트 기반 비디오 확산 생성 모델을 사용하여 목표 프레임을 생성합니다. 둘째, 내부 임베딩을 통해 시스템의 간섭을 추정하고, 이를 통해 로봇의 현재 상태와 미래 목표 상태를 정렬합니다. 마지막으로, 생성된 목표에 조건화된 확산 정책을 통해 다중 모드 작업 분포를 잘 모델링합니다.

- **Performance Highlights**: GEVRM은 표준 및 간섭 있는 CALVIN 벤치마크에서 최첨단 성능을 달성하였으며, 실제 로봇 작업에서도 상당한 개선을 보입니다. 이 모델은 생성한 목표 상태의 표현력을 크게 향상시켰으며, 로봇 조작 과제에 있어 더 효과적인 행동을 일관되게 수행할 수 있습니다. 전반적으로, GEVRM은 기존의 VLA 모델보다 더 강력하고 안정적인 성능을 제공합니다.



### Abduction of Domain Relationships from Data for VQA (https://arxiv.org/abs/2502.09219)
Comments:
          In Proceedings ICLP 2024, arXiv:2502.08453

- **What's New**: 이번 논문에서는 도메인 데이터가 부족한 이미지와 질문을 ASP 프로그램으로 표현하는 시각적 질문 응답(VQA) 문제를 연구합니다. 기존 지식 증강 기법을 보완하는 새로운 접근법을 제공하며, 과거 예제에서 이미지 구조의 도메인 관계를 추론(abduction)하는 방법을 도입합니다.

- **Technical Details**: 추론 문제를 설정한 후, 베이스라인(baseline) 접근법을 제시하고 이를 구현합니다. 이 방법은 적은 예제만 있더라도 질문 응답의 정확도를 상당히 향상시킵니다. ASP(Answer Set Programming) 프로그램을 사용하여 이미지와 질문을 표현하는 방식은 기존의 접근법과 대비되는 독립적인 방법론입니다.

- **Performance Highlights**: 제안된 접근법은 기존 기법에 비해 더 높은 정확도가 달성되며, 데이터 수집에 대한 부담을 줄입니다. 또한, 적은 수의 예제만으로도 효과적인 성과를 보이는 점이 크게 강조됩니다.



### Revisiting Euclidean Alignment for Transfer Learning in EEG-Based Brain-Computer Interfaces (https://arxiv.org/abs/2502.09203)
- **What's New**: 이 논문에서는 EEG 기반의 뇌-컴퓨터 인터페이스(BCI)에서 Euclidean Alignment (EA) 방법을 재조명합니다. EA는 다양한 주제와 세션 간의 데이터 분포 차이를 줄여 전이 학습(Transfer Learning)에서의 부정적 효과를 방지하는 데 도움을 줍니다. 본 연구는 EA의 사용 절차와 올바른 활용 방법을 설명하고, 이의 응용 및 확장을 소개하며, 새로운 연구 방향을 제시합니다.

- **Technical Details**: EA는 2020년에 제안된 방법으로, 두 개의 단순하고 효율적인 공식으로 구성되어 있습니다. EEG 데이터를 서로 다른 주제(세션)에서 정렬하여 데이터 일관성을 높이고 이를 통해 BCI에서의 전이 학습을 용이하게 합니다. 이 방법은 제한된 보정 데이터와 다른 주제(세션)의 EEG 데이터를 직접 결합하여 분류기를 훈련할 수 있도록 합니다.

- **Performance Highlights**: EA 방법은 여러 BCI 대회에서 우수한 성과를 나타내었습니다. 저자 팀은 2019년부터 2024년까지 중국 BCI 대회에서 9개의 국가 챔피언을 차지했습니다. 이 방법은 2021년 NeurIPS 대회에서의 Benchmark for EEG Transfer Learning(BEETL) 경쟁에서도 우수한 성과를 보여주었으며, 다양한 연구 및 실제 응용 분야에 적합하게 확장되었습니다.



### E-MD3C: Taming Masked Diffusion Transformers for Efficient Zero-Shot Object Customization (https://arxiv.org/abs/2502.09164)
Comments:
          16 pages, 14 figures

- **What's New**: 이 논문에서는 E-MD3C (Efficilent Masked Diffusion Transformer with Disentangled Conditions and Compact Collector)를 제안합니다. 이 프레임워크는 제로샷 개체 이미지 사용자화(zero-shot object image customization)에 대해 고효율적인 접근 방식을 제공합니다. 기존의 자원 집약적인 Unet 아키텍처와는 달리, 경량화된 masked diffusion transformers를 사용하여 계산 효율성을 크게 향상시킵니다.

- **Technical Details**: E-MD3C는 세 가지 핵심 구성 요소를 통합합니다. 첫째, 효율적인 masked diffusion transformer가 autoencoder의 잠재 공간(latent space)을 처리합니다. 둘째, 분리된 조건 설계를 통해 배경 정렬과 세부사항 보존을 유지하면서 정보를 압축합니다. 셋째, 학습 가능한 Conditions Collector는 여러 입력을 통합하여 간결한 표현으로 변환하여 효과적인 노이즈 제거와 학습을 지원합니다.

- **Performance Highlights**: E-MD3C는 VITON-HD 데이터셋에서 PSNR, FID, SSIM 및 LPIPS와 같은 지표에서 기존 방법보다 우수한 성능을 보입니다. 단 1/4의 파라미터 수를 사용하여, 1720M Unet 기반의 latent diffusion 모델에 비해 2.5배 빠른 추론 속도와 2/3의 GPU 메모리를 소모합니다. 이러한 성능 개선은 E-MD3C의 간결하고 효과적인 설계 덕분입니다.



### Feature-based Graph Attention Networks Improve Online Continual Learning (https://arxiv.org/abs/2502.09143)
Comments:
          16 pages

- **What's New**: 이번 논문에서는 Graph Attention Networks (GATs)를 기반으로 한 온라인 연속 학습 프레임워크를 제안하여 이미지 분류를 위한 새로운 접근 방식을 제공합니다. 이 프레임워크는 컨텍스트 관계를 효과적으로 포착하고 학습된 주의 가중치를 통해 작업별 표현을 동적으로 업데이트합니다. 또한, 고급 글로벌 풀링 전략을 통합하여 지속적인 학습을 위한 분류 성능을 개선하는 것을 목표로 합니다.

- **Technical Details**: 제안된 모델인 Feature-based Graph Attention Networks (FGAT)는 이미지 데이터를 그래프 형태로 변환하고, 각 노드가 다중 스케일 특성 정보를 인코딩하여 복잡한 피처 간의 관계를 모델링합니다. 이를 위해 사전 훈련된 피쳐 추출기를 사용하며, GAT를 통해 로컬 주변 맥락에 기반하여 동적 노드 표현 업데이트가 가능합니다. 또한, 맞춤형 가중 평균 풀링 메커니즘을 통해 모든 노드의 정보를 상대적 중요도에 따라 집계합니다.

- **Performance Highlights**: FGAT는 SVHN, CIFAR10, CIFAR100, MiniImageNet과 같은 벤치마크 데이터셋에서 기존의 CNN 및 GNN 기반의 연속 학습 방법들과 비교하여 우수한 성능을 입증했습니다. 특히, 이전 작업의 표현력을 강화하면서 새로운 작업의 성능을 유지할 수 있는 기법인 리허설 메모리 중복 전략을 제안하여 메모리 예산을 효율적으로 관리합니다. 결과적으로, 제안한 방법은 최신 기술 대비 성능이 뛰어난 것으로 나타났습니다.



### A Novel Dialect-Aware Framework for the Classification of Arabic Dialects and Emotions (https://arxiv.org/abs/2502.09128)
- **What's New**: 이 연구는 아랍어의 방언과 감정 인식에 관한 기존 연구의 한계를 극복하기 위해 개발된 새로운 프레임워크를 제안합니다. 이 프레임워크는 아랍어 텍스트에서 방언과 감정을 식별하고 예측하는 기능을 제공합니다. 특히, 방언 인식에 맞춤화된 새로운 감정 어휘집을 생성할 수 있는 기능이 포함되어 있습니다.

- **Technical Details**: 제안된 프레임워크는 세 가지 모듈로 구성되어 있습니다: 텍스트 전처리 모듈(text-preprocessing module), 분류 모듈(classification module), 그리고 새로운 방언 인식 감정 어휘를 구축할 수 있는 클러스터링 모듈(clustering module)입니다. 이 연구는 이를 통해 아랍어의 다양한 방언에 대한 새로운 감정 어휘를 생성했습니다.

- **Performance Highlights**: 프레임워크는 아랍어 방언 분류에서 88.9%의 정확도를 달성하였으며, 이는 최신 연구 결과를 6.45% 포인트 초과하는 성과입니다. 또한 이 프레임워크는 이집트 방언과 걸프 방언에서 각각 89.1%와 79%의 감정 인식 정확도를 기록했습니다.



### Unsupervised Anomaly Detection on Implicit Shape representations for Sarcopenia Detection (https://arxiv.org/abs/2502.09088)
- **What's New**: 이 논문은 노화 관련 질병인 sarcopenia를 탐지하기 위한 새로운 접근 방식인 암묵적 신경 표현(implicit neural representation, INR)을 사용하여 근육의 형태를 모델링하는 방법을 제안합니다. 기존의 근육량 기준 대신, 저자는 근육의 형태를 기반으로 abnormal(비정상) 근육을 구별하는 비지도 학습(anomaly detection) 방법을 도입합니다. 이 방법은 sarcopenia 근육과 비정상 근육을 효과적으로 식별하면서, 더 나아가 의료진의 진단을 지원할 수 있는 가능성을 보입니다.

- **Technical Details**: 저자는 3D 이미지를 통해 근육의 모양을 모델링하기 위해 INR을 활용하며, 훈련된 모델이 non-sarcopenic 근육으로부터 학습된 모양 사전 정보를 기반으로 reconstruction error(재구성 오류)를 측정합니다. 구체적으로, latent representation(잠재 표현) 학습을 통해 정상 근육과 비정상 근육을 명확히 분리하는 방식을 취합니다. 이 과정에서 Multilayer Perceptron(MLP) 분류기가 3D 좌표와 함께 조건 벡터를 입력으로 받아들여 voxel의 점유 확률을 예측합니다.

- **Performance Highlights**: 실험 결과는 103개의 세분화된 볼륨 데이터셋을 기반으로 하여, sarcopenic 근육의 Dice 재구성 오류가 현저히 낮음을 나타냅니다. 또한, 학습된 잠재 형태 표현의 선형 판별 분석(Linear Discriminant Analysis, LDA) 결과는 비정상적인 근육과 정상적인 근육 그룹 간의 명확한 구분을 보여줍니다. 이 결과는 sarcopenia 선별 과정의 정확성을 높이는 데 기여할 수 있습니다.



### Application of Tabular Transformer Architectures for Operating System Fingerprinting (https://arxiv.org/abs/2502.09084)
Comments:
          Submitted as a preprint (not peer reviewed). 22 pages, 9 figures. Code and datasets available at: this https URL

- **What's New**: 이 연구에서는 네트워크 트래픽 분석을 기반으로 하는 운영 체제(OS) 지문 인식 방법에 대해 심층학습(Deep Learning, DL) 아키텍처, 특히 Tabular Transformer를 적용했습니다. 기존의 기계 학습(Machine Learning, ML) 접근 방식에 비해 FT-Transformer가 더 뛰어난 성능을 보였으며, 다양한 분류 수준에서의 유효성을 입증했습니다. 이 연구는 진화하는 네트워크 조건에서 OS 지문 인식의 정확성을 향상시키기 위한 새로운 접근법을 제시합니다.

- **Technical Details**: 운영 체제 지문 인식은 네트워크에 연결된 장치의 트래픽을 분석하여 OS 가족 및 버전을 식별하는 프로세스이며, 이를 위해 TCP/IP 스택 매개변수, 패킷 구조 및 프로토콜 처리의 차이를 활용합니다. 동시에 이 연구에서는 TabTransformer와 FT-Transformer 두 가지 Tabular Transformer 아키텍처를 실험하여, 다양한 네트워크 트래픽의 복잡한 상관관계를 포착함으로써 OS 지문 인식 성능을 개선했습니다. 연구는 3개의 공개 데이터세트를 통해 다수의 분류 레벨(OS 가족, 주요 및 부 버전)에서 검증되었습니다.

- **Performance Highlights**: FT-Transformer는 기존의 ML 모델 및 다른 접근 방식들과 비교하여 전반적으로 뛰어난 성능을 보였습니다. 이 연구의 결과는 DL 기반 OS 지문 인식 기술이 복잡한 네트워크 환경에서도 높은 정확도를 유지할 수 있음을 보여줍니다. 실험 코드를 공개하여 연구의 재현성을 보장하고 있으며, 이는 향후 연구에 기여할 수 있는 기초 자료가 될 것입니다.



### Quantifying Cryptocurrency Unpredictability: A Comprehensive Study of Complexity and Forecasting (https://arxiv.org/abs/2502.09079)
Comments:
          This is the author's accepted manuscript, modified per ACM self-archiving policy. The definitive Version of Record is available at this https URL

- **What's New**: 이번 연구는 암호화폐 단일 시계열의 예측 가능성을 심층적으로 분석하였습니다. 라이트코인, 바이낸스코인, 비트코인, 이더리움, XRP의 USD 환율을 대상으로 성능을 비교한 결과, 단순한 예측 모델이 복잡한 딥러닝 모델보다 더 나은 예측 정확도를 보여주었습니다. 또한, 분석 결과 암호화폐의 시계열이 브라운 운동과 유사한 특성을 지니고 있으며, 이는 이 시장의 예측 난이도를 강조합니다.

- **Technical Details**: 이 연구는 Permutation Entropy 및 Complexity-Entropy causality plane(CH-plane) 등의 복잡성 측정 기법을 통해 암호화폐 시장의 본질적인 특성을 분석하였습니다. 연구에서 사용된 데이터는 2020년 7월 3일부터 2023년 12월 21일까지의 일일 가격 데이터이며, 예측 모델로는 Naive 모델부터 ARIMA, XGBModels, NBEATS에 이르는 다양한 통계적 및 기계학습 모델이 포함됩니다.

- **Performance Highlights**: 예측 결과, 단순 모델들이 복잡한 기계 및 딥러닝 모델보다 예측 정확도에서 일관되게 더 높은 성능을 보였습니다. 이 연구는 다양한 예측 기법을 비교하면서 암호화폐 시장의 예측 가능성을 평가했으며, 복잡한 모델이 항상 더 나은 결과를 보장할 것이라는 가설에 도전했습니다. 연구의 결과는 암호화폐 가격 움직임 예측에 대한 새로운 통찰을 제공하며, 이 시장의 예측과 관련된 도전 과제를 إعادة 평가할 필요성을 강조합니다.



### CRANE: Reasoning with constrained LLM generation (https://arxiv.org/abs/2502.09061)
- **What's New**: 본 연구에서는 LLM(대규모 언어 모델)의 출력이 형식적인 문법에 준수하도록 강제하는 방법을 제안합니다. LLM의 출력이 매우 제한적인 문법에 맞춰 제한될 경우, 추론 능력이 감소하는 현상이 관찰되었습니다. 그러나 추가적으로 설계된 규칙으로 출력 문법을 확장하면 이러한 추론 능력을 유지하면서도 문법적으로 정확한 출력을 보장할 수 있음을 보여줍니다.

- **Technical Details**: 이 논문에서는 CRANE(제약 추론 증대 생성)이라는 알고리즘을 제안하여, 제약된 생성의 정확성과 비제약 생성의 유연성을 조화롭게 결합합니다. 구체적으로는 LLM의 입력과 출력을 제어하여, 지식 기반 문제 해결에서 대칭성과 문법적 정확성을 유지할 수 있도록 하였습니다. 제안된 접근법은 중간 추론 단계를 포함하여 최종 출력이 항상 의도된 구조에 부합하도록 추가 생산 규칙을 포함한 확장된 문법을 활용합니다.

- **Performance Highlights**: 여러 개의 오픈 소스 LLM과 벤치마크를 통한 실험 결과, CRANE은 최신 제약 디코딩 전략 및 표준 비제약 디코딩을 모두 초과하는 성능을 보이는 것으로 나타났습니다. 특히, 어려운 상징적 추론 벤치마크인 GSM-symbolic 및 FOLIO에서 기준선 대비 최대 10%의 정확도 개선을 달성하였습니다.



### Game Theory Meets Large Language Models: A Systematic Survey (https://arxiv.org/abs/2502.09053)
Comments:
          10 pages

- **What's New**: 본 논문은 게임 이론(Game Theory)과 대규모 언어 모델(LLMs) 사이의 상호 관계를 탐구하는 포괄적인 조사 결과를 제시합니다. 기존 연구는 게임 이론이 LLM을 평가하고 향상시키는 데 어떻게 활용될 수 있는지를 조사하는 것에 집중해 있었지만, 본 논문에서는 LLM이 게임 이론에 기여하는 측면도 강조합니다. 특히, LLM이 고전적 게임 모델을 재구성하며 균형 분석에 미치는 영향을 다루고 있습니다.

- **Technical Details**: 게임 이론은 합리적인 의사결정자 간의 전략적 상호작용을 분석하기 위한 수학적 프레임워크를 제공하며, 최근 LLM의 발전과 함께 게임 이론과 LLM의 교차점에 대한 연구가 증가하고 있습니다. 특히, 표준화된 게임 기반 평가 및 알고리즘 혁신을 통해 LLM의 성능을 향상시키고, LLM이 사회에 미치는 영향을 게임 모델로 특성화하는 방향의 연구가 포함됩니다. 이는 Nash 균형(Nash Equilibrium) 및 Shapley 가치(Shapley Value)와 같은 게임 이론의 원칙을 활용하여 이루어집니다.

- **Performance Highlights**: LLMs는 매트릭스 게임(Matrix Games)과 같은 성공적인 시나리오에서 뛰어난 전략적 사고 능력을 보이며, 다양한 방법으로 평가되고 있습니다. 연구자들은 LLM의 의사결정 능력을 향상시키기 위한 전략으로 재귀적 추론 프레임워크와 보조 모듈 통합을 제안하고 있습니다. 마지막으로, LLM의 사회적 영향 예측을 위한 새로운 이론적 프레임워크가 개발되고 있으며, 이는 AI와 인간의 상호작용에 대한 이해를 심화시킵니다.



### Leveraging Member-Group Relations via Multi-View Graph Filtering for Effective Group Recommendation (https://arxiv.org/abs/2502.09050)
Comments:
          5 pages, 3 figures, 4 tables; ACM Web Conference (WWW 2025) (to appear) (Please cite our conference version.)

- **What's New**: 이 연구에서는 기존의 딥러닝(Deep Learning) 기반의 그룹 추천 시스템의 복잡한 훈련 절차를 극복하기 위해 Group-GF라는 새로운 접근 방식을 제안합니다. Group-GF는 다양한 관점을 제공하는 다중 뷰 그래프 필터링(multi-view graph filtering)을 이용하여 그룹에게 신속한 추천을 수행합니다. 이는 멤버와 그룹 간의 복잡한 상호 작용을 반영하면서도 비싼 모델 훈련 없이 효율적인 추천이 가능하게 합니다.

- **Technical Details**: Group-GF는 세 가지 아이템 유사성 그래프를 구성한 후, 각 그래프에 대한 차별화된 다항 그래프 필터를 최적 설계합니다. 이 방법은 멤버-그룹 매트릭스와 아이템 유사성 그래프를 결합하여 그래프 신호를 효과적으로 통합합니다. 각 유사성 그래프에 대해 최적화된 필터링을 수행한 뒤, 세 가지 그래프 필터를 집계하여 그룹 추천의 정확성을 높입니다.

- **Performance Highlights**: Group-GF는 벤치마크 데이터세트에서 최첨단 정확도와 함께 최대 1.55초의 놀라운 효율적인 런타임을 달성했습니다. 이러한 성능은 고급 훈련 없이도 복잡한 멤버-그룹 동역학을 처리할 수 있는 다중 뷰 그래프 필터링에 기인합니다. 이 연구는 Group-GF의 필터링 과정이 최적화와 매끄러움 규제를 통해 모델의 행동을 더 명확하게 해석할 수 있도록 이론적으로 연결되었습니다.



### Optimal Algorithms in Linear Regression under Covariate Shift: On the Importance of Precondition (https://arxiv.org/abs/2502.09047)
- **What's New**: 이 논문은 고차원(High-Dimensional) 선형 회귀(Linear Regression) 문제에서 공변량 이동(Covariate Shift) 아래 최소-최대(min-max) 최적 추정량을 제시하는 내용을 다룹니다. 저자들은 Bayesian Cramer-Rao 부등식을 활용하여 일반화 경계를 설정하고, 알고리즘이 최적성을 달성하는 조건을 분석합니다. 특히, 선형 변환을 통해 소스 배포(Source Distribution)의 최적 추정량을 효과적으로 계산할 수 있는 방법을 제시합니다.

- **Technical Details**: 저자들은 두 가지 제약 조건 하에서 정보 이론적 최소-최대 최적성 결과를 도출하였습니다: (1) 이소트로픽 구형 제약, (2) 비등방성 타원체 제약. 구체적으로 Bayesian 행렬 하한과 전망팬확률모형을 통해 최적 추정기의 공분산(Covariance)을 설정하여 엄격한 경계를 제공합니다. 이 연구는 공변량 이동 문제에서 통계적 병목 현상을 해결하고, 경량화된 알고리즘의 효과에 대해 깊이 탐구했습니다.

- **Performance Highlights**: SGD(Stochastic Gradient Descent)와 같은 현대 머신러닝 최적화 방법들이 공변량 이동 문제에서 일반화 성능에 미치는 영향을 분석하였으며, ASGD(Accelerated Stochastic Gradient Descent)가 최적 상태에 도달할 수 있는 조건을 제시합니다. 논문에서는 모멘텀이 일반화에 긍정적인 영향을 주는 경우를 보여주고, 특정 샘플 크기 도달 후의 학습 곡선에 대한 행동을 설명합니다. 이러한 통찰은 LLMs(Large Language Models)와 같은 현재의 머신러닝 기술에 중요한 영향을 미칠 것입니다.



### Criteria-Aware Graph Filtering: Extremely Fast Yet Accurate Multi-Criteria Recommendation (https://arxiv.org/abs/2502.09046)
Comments:
          12 pages, 8 figures, 7 tables; ACM Web Conference (WWW 2025) (to appear) (Please cite our conference version.)

- **What's New**: 이번 연구에서는 훈련이 필요 없는 다기준( MC ) 추천 시스템을 제안합니다. CA-GF(criterion-aware graph filtering)를 사용하여 효율적이고 정확한 추천을 제공합니다. 복잡한 다차원 사용자 피드백을 처리하면서도 높은 정확도를 유지할 수 있도록 설계되었습니다.

- **Technical Details**: CA-GF는 MC 사용자 확장 그래프를 기반으로 아이템-아이템 유사도 그래프를 구축합니다. 이후 기준별 그래프 필터링을 통해 각 기준에 최적화된 필터를 찾고, 기준 선호도를 반영한 집계를 수행합니다. 이러한 접근법은 기존의 DNN 기반 방법보다 훈련 시간과 모델 성능에서 우수한 결과를 보여줍니다.

- **Performance Highlights**: CA-GF는 최대 24%의 정확도 향상을 달성하며, 대규모 벤치마크 데이터셋에서도 0.2초 미만의 실행 시간을 기록합니다. 또한, 각 기준의 기여도를 시각적으로 설명함으로써 모델 해석 가능성도 크게 향상됩니다.



### Diversity Enhances an LLM's Performance in RAG and Long-context Task (https://arxiv.org/abs/2502.09017)
- **What's New**: 본 논문에서는 대규모 언어 모델(LLMs)의 문맥 창 환경에서의 한계를 해결하기 위해 다채로운 정보 선택의 중요성을 강조합니다. 특히, 질문-답변(Q&A) 및 긴 문맥 요약을 포함한 작업에서 정보의 다양성을 통합하여 성능을 향상시킬 수 있음을 입증합니다. MMR(Maximal Marginal Relevance) 및 FPS(Farthest Point Sampling) 원칙을 바탕으로 한 접근 방식은 쿼리와의 유사성이 높은 내용을 선택하는 전통적인 방법의 한계를 극복합니다.

- **Technical Details**: 탐색 과정 중 다채로운 콘텐츠 선택의 중요성을 고려하며 MMR과 FPS를 활용한 방법론을 제시합니다. MMR은 선택된 항목 간의 유사성을 최대한 줄이면서 보상과 다양성 간의 균형을 맞추는 그리디 알고리즘을 기반으로 합니다. FPS는 원래의 3D 포인트 클라우드에서 다양성을 통해 포인트를 선택하는 기법으로, 문맥 창이나 보상 개념을 통합하여 MMR과 동일한 방식으로 사용될 수 있습니다.

- **Performance Highlights**: 실험 결과, 다양성을 포함한 콘텐츠 선택 과정은 relevant 문장이나 청크의 recall을 크게 향상시켰습니다. MMR과 FPS의 활용에 있어 MMR이 FPS보다 약간 더 우수한 recall을 유지하면서도 더 낮은 latency를 보여주었습니다. 특히, 선택된 문장 및 청크의 순서가 downstream 작업에서의 성능에 긍정적인 영향을 미치는 것으로 나타났습니다.



### Hope vs. Hate: Understanding User Interactions with LGBTQ+ News Content in Mainstream US News Media through the Lens of Hope Speech (https://arxiv.org/abs/2502.09004)
- **What's New**: 이번 논문은 3,161개의 미국 주요 케이블 뉴스 매체의 유튜브 뉴스 비디오에 게시된 1,419,047개의 댓글을 분석하여 LGBTQ+ 뉴스 콘텐츠와 사용자의 상호작용 방식을 조사합니다. 특히, 긍정적(희망 발언) 및 부정적 콘텐츠를 구분할 수 있는 세밀한 희망 발언 분류기를 구축하여 긍정적인 콘텐츠의 탐지를 꾀했습니다. 또한, LGBTQ+ 건강 전문가와의 협력을 통해 정치적으로 다양한 표본에서 주석이 부여된 3,750개의 데이터 세트를 생성했습니다.

- **Technical Details**: 저자들은 정치적 입장이 다양한 평가자가 주석을 달아, 각 주석이 긍정(희망 발언), 부정, 중립 및 관련 없는 콘텐츠로 분류된 데이터셋을 구성했습니다. 또한, 사용자 정치 신념이 LGBTQ+ 커뮤니티와 관련된 콘텐츠 평가에 미치는 영향을 분석했습니다. 이 연구는 특히 미국의 정치적 담론에서의 LGBTQ+ 논의와 연관된 감정 동 동 양상을 이해하는 데 중요한 인사이트를 제공합니다.

- **Performance Highlights**: 연구 결과는 정당별 보수적이며 진보적인 평가자 간의 긍정적 콘텐츠에 대한 인식 차이에 대한 통찰력을 제공합니다. 또한 평가자의 정치적 신념에 따라 선별된 모델 간의 불일치가 발생하며, 제로샷 대형 언어 모델(LLMs)이 자유주의적 평가자와 더 높은 일치를 보임을 보여줍니다. 이러한 결과는 LGBTQ+ 콘텐츠에 대한 주관적 평가가 표현되는 방식을 돕는 중요한 지침을 제공합니다.



### Residual Transformer Fusion Network for Salt and Pepper Image Denoising (https://arxiv.org/abs/2502.09000)
Comments:
          8 pages, 17 figures

- **What's New**: 이 논문에서는 Residual Transformer Fusion Network (RTF-Net)라는 새로운 이미지 복원 아키텍처를 제안합니다. 이 아키텍처는 Convolutional Vision Transformer (CvT)와 Residual Networks (ResNet)를 결합하여 효과적인 노이즈 제거를 목표로 하며, NSN (Noise Suppression Network)과 SEN (Structure Enhancement Network)의 두 부분으로 나뉩니다. 이를 통해 기존의 이미지 노이즈 제거 방법에서 요구되는 노이즈에 대한 사전 지식 없이도Cleaner 이미지 생성이 가능합니다.

- **Technical Details**: RTF-Net의 Noise Suppression Network는 Residual Block을 통해 이미지를 처리하여 노이즈 맵을 학습합니다. 두 번째 단계에서는 Structure Enhancement Network가 CvT를 이용하여 노이즈가 제거된 이미지에 필요한 세부 사항을 학습하여 최종적으로 클린 이미지를 생성합니다. 이 아키텍처는 salt and pepper 노이즈 모델링을 포함하여 다양한 노이즈 유형을 효과적으로 처리할 수 있도록 설계되었습니다.

- **Performance Highlights**: 제안된 RTF-Net 모델은 DIV2K Training Set을 사용하여 훈련되었으며, 다양한 이미지(Lena, Bridge, Pepper, BSD300)에서 PSNR(peak signal-to-noise ratio) 성능을 평가했습니다. 논문에 따르면, 대부분의 경우 제안한 접근 방식이 기존의 여러 방법들보다 우수한 성능을 보였으며, 특히 Pepper 이미지의 30% 노이즈 수준에서는 NLSF-CNN이 우세했습니다.



### Off-Policy Evaluation for Recommendations with Missing-Not-At-Random Rewards (https://arxiv.org/abs/2502.08993)
Comments:
          4pages

- **What's New**: 이번 연구는 추천 시스템에서 발생하는 편향 문제를 해결하기 위한 새로운 추정기(MIPS (w/ROIPS))를 제안합니다. 이 새로운 방법은 사용자의 보상 관측 확률과 로깅 정책의 두 가지 확률을 활용하여 위치 편향과 로깅 정책 편향을 동시에 완화할 수 있습니다. 실험 결과 제안된 추정기는 다른 기존 추정기들보다 우수한 성과를 나타내며, 보상 관측의 편향이 증가해도 안정적인 성능을 유지합니다.

- **Technical Details**: URL(Unbiased Recommender Learning)와 OPE/OL(Off-Policy Evaluation/Learning) 기법을 통해 추천 알고리즘의 성능을 향상시키는 동시에, 데이터 로그에서 발생하는 폰빙스(ideal loss function)와 관련된 주요 요소들을 다룹니다. MIPS(마르기널화된 중요도 샘플링)는 기존의 IPS(역중요도 샘플링)보다 낮은 분산으로 실행될 수 있도록 설계되었습니다. 또한, 제안한 MIPS (w/ROIPS)는 여러 가정 하에서도 편향을 방지하고 성능 향상에 기여합니다.

- **Performance Highlights**: 제안된 MIPS (w/ROIPS) 추정기는 다양한 실험에서 우수한 성능을 나타내며, 특히 보상 관측의 편향 수준이 증가할 때 더욱 두드러진 성과를 기록했습니다. 기존의 IPS 방식 대비 불리한 조건에서도 성공적으로 편향과 분산 간의 균형을 맞출 수 있는 가능성을 보여 줍니다. 이러한 결과는 추천 시스템의 실제 환경에서도 유용하게 활용될 수 있을 것으로 기대됩니다.



### SkyRover: A Modular Simulator for Cross-Domain Pathfinding (https://arxiv.org/abs/2502.08969)
Comments:
          9 pages

- **What's New**: SkyRover는 UAV(무인 항공기)와 AGV(자동화 유도차량) 간의 협력을 지원하는 최초의 모듈식 시뮬레이터로, 이 분야에서의 연구와 알고리즘 개발을 위한 통합 툴킷을 제공합니다. 기존 시뮬레이터들은 대개 특정 도메인에 국한되어 있었지만, SkyRover는 다양한 3D 환경에서 리얼리스틱한 동작을 구현할 수 있게 해줍니다. 이로 인해 연구자들은 UAV-AGV 간의 상호 작용을 통한 물류 및 자동화의 효율성을 극대화할 수 있습니다.

- **Technical Details**: SkyRover는 시뮬레이션 환경을 구축하는 데 있어 3D 그리드와 정밀한 물리 모델을 활용하며, 이러한 환경은 Gazebo와 통합되어 있습니다. 시스템의 주요 모듈로는 ‘Sim World Zoo’, ‘3D Grid Generator’, ‘Unified Algorithm Wrapper’, ‘Plan Executor’, ‘System Interface’가 포함되어, 동시 다발적인 실험과 알고리즘 테스트를 지원합니다. 특히, MAPF(Multi-Agent Path Finding) 알고리즘에 대한 일관된 인터페이스를 제공함으로써 다양한 연구 방법론을 적용할 수 있는 유연성을 보장합니다.

- **Performance Highlights**: SkyRover는 AGV와 UAV의 협력을 통한 경로 찾기(pathfinding) 실험에서 뛰어난 성과를 나타내며, 복잡한 로봇 상호작용을 효과적으로 시뮬레이션할 수 있습니다. 다양한 시나리오에서의 실험 결과는 시뮬레이터의 3D 매핑 및 시각화 기능이 우수함을 보여줍니다. 특히 재고 스캔 작업과 공중 화물 이송 작업을 통해 실시간으로 UAV와 AGV 간의 원활한 협조를 시연하며, 효율적인 물류 운영의 가능성을 강조합니다.



### Integrated Optimization and Game Theory Framework for Fair Cost Allocation in Community Microgrids (https://arxiv.org/abs/2502.08953)
- **What's New**: 이번 연구는 커뮤니티 마이크로그리드에서 공정한 비용 할당을 위한 새로운 프레임워크를 제시합니다. 이 프레임워크는 다목적 최적화(multi-objective optimization)와 협동 게임 이론(cooperative game theory)을 통합하여 마이크로그리드 운영과 비용 할당의 효율성을 높입니다. 기존 방법들이 동적인 기여와 이득을 충분히 반영하지 못했던 문제를 해결하기 위해, 혼합 정수 선형 프로그래밍(mixed-integer linear programming)과 샤플리 값 분석을 결합하여 공정한 이익 분배를 보장합니다.

- **Technical Details**: 제안된 프레임워크는 여러 로드(load), 분산형 태양광 발전 유닛(distributed solar generation units), 그리고 공유 배터리 에너지 저장 시스템 배치로 구성됩니다. MILP를 사용하여 마이크로그리드 운영 최적화를 다루고, 샤플리 가치 분석을 통해 공정한 이익 분배를 수행합니다. 이 시스템은 24시간동안 다양한 시간 단위로 운영되며, 주 전력망에서 전력을 수입할 수 있는 기능을 갖추고 있습니다.

- **Performance Highlights**: 실제 데이터를 이용한 여섯 가지 운영 시나리오에서 프레임워크의 유효성을 검증한 결과, 피크 수요 감소는 7.8%에서 62.6%까지 이루어졌으며, 태양광 활용률이 114.8%로 증가했습니다. 또한 협동적 이익은 하루 최대 1,801.01 달러에 달하며, 샤플리 값 기반 할당은 다양한 로드 카테고리에서 -16.0%에서 +14.2% 간의 균형 잡힌 이익-비용 분배를 보였습니다.



### The Stochastic Parrot on LLM's Shoulder: A Summative Assessment of Physical Concept Understanding (https://arxiv.org/abs/2502.08946)
Comments:
          NAACL 2025 Main Conference. First 5 authors contributed equally. Project page: this https URL

- **What's New**: 본 논문은 LLMs(대형 언어 모델)의 진정한 이해 여부에 대해 정량적인 실험을 통해 검증하고자 합니다. 우리는 특히 Stochastic Parrot 현상, 즉 LLM이 단순히 연관 패턴을 바탕으로 반복하는지 여부를 평가하는 PhysCo라는 새로운 과제를 제안합니다. 이 과제는 물리적 개념 이해도를 측정하기 위해 설계되었습니다. 고민이 많았던 메모리 문제를 해결하기 위해 그리드 형식의 입력을 사용하여 다양한 이해 수준을 표현하고 있습니다.

- **Technical Details**: PhysiCo는 저급 이해(subtask)와 고급 이해(high-level understanding)라는 두 가지 하위 과제를 포함합니다. 저급 이해는 LLM의 기억 능력을 측정하는 자연어 형식 질문으로 구성되어 있으며, 고급 이해 과제는 추상 표현을 기반으로 상대적으로 심화된 이해를 평가하는 숙제입니다. 우리는 LLMs가 저급 과제에서 높은 정확도를 보이지만, 고급 과제에서는 인간에 비해 40% 가량 성능이 떨어진다는 두 가지 주요 결론을 도출했습니다.

- **Performance Highlights**: 실험 결과, 최신 LLMs는 저급 이해 과제에서 95% 이상의 정확도를 기록하였으나, 고급 이해 과제에서는 인간에 비해 평균 약 40% 낮은 정확도를 보였습니다. 이는 LLM이 진정한 개념 이해 능력에서는 한계를 나타내며, 새로운 그리드 형식이 아닌 그 자체의 고차원적 이해의 어려움이 원인임을 시사합니다. 본 연구는 LLM의 이해력을 측정하는 방법론적 기틀을 확립하고, LLM과 인간 간의 성능 격차를 명확히 보여줍니다.



### Beyond the Singular: The Essential Role of Multiple Generations in Effective Benchmark Evaluation and Analysis (https://arxiv.org/abs/2502.08943)
Comments:
          10 pages, 1 table, 4 Figures

- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 성능 평가 방법에 대한 새로운 접근 방식을 제시하고 있습니다. 기존의 평가 방식은 LLM의 본질적인 무작위성을 간과하는 경우가 많았으나, 본 연구에서는 계층적 통계 모델을 통해 이를 통합하여 보다 신뢰성이 높은 벤치마크(score)를 제공하고자 합니다. 이를 통해 개별 프롬프트(prompts)의 난이도를 정밀하게 측정하고 오류 감지 및 품질 관리를 위한 데이터 맵을 생성했습니다.

- **Technical Details**: 저자들은 LLM의 응답 생성 방식에 대해 deterministic한 greedy decoding과 stochastic한 random sampling 방법을 비교했습니다. 후자는 원래의 확률 분포에 따라 매 단계에서 토큰을 무작위로 샘플링하여 비결정적 출력을 생성하게 됩니다. 연구에서는 각 프롬프트에 대해 k개의 무작위 생성(random generations)을 통해 벤치마크 과정을 계층적 모델(hierarchical model)로 보았으며, 이를 통해 불확실성을 줄이고 정확한 평가를 할 수 있음을 보였습니다.

- **Performance Highlights**: 여러 번의 생성을 통해 벤치마크 점수의 정확도가 향상되며, variance가 감소하고 정확성의 세분화된 난이도 점수 ℙ(correct)도 도입되었습니다. 이는 프롬프트들의 난이도를 비교하고 잘못 표기되거나 모호한 프롬프트를 효과적으로 감지할 수 있는 도구로서의 가능성을 보여줍니다. 본 연구는 LLM 벤치마크 개발 과정에서의 방법론적 틀을 개선하고 실제 적용 가능성을 높이는 데 기여할 것입니다.



### Towards Understanding Why Data Augmentation Improves Generalization (https://arxiv.org/abs/2502.08940)
- **What's New**: 이 논문에서는 기존의 데이터 증강 기법들이 일반화 성능을 어떻게 향상시키는지에 대한 통합 이론적 프레임워크를 제시합니다. 구체적으로, 부분 의미적 특징 제거(Partial Semantic Feature Removal)와 특징 혼합(Feature Mixing)의 두 가지 주요 효과를 통해 모델이 더 다양한 특징을 학습하도록 유도하고, 일반화 성능을 개선하는 방법을 설명합니다. 이를 통해 서로 다른 증강 기법들이 어떻게 공통된 메커니즘을 공유하는지를 밝혀내고자 합니다.

- **Technical Details**: 제안된 이론적 프레임워크는 3층 컨볼루션 신경망(CNN)을 예시로 사용하여 데이터 증강 기법들이 어떻게 모델의 일반화를 촉진하는지를 분석합니다. 부분 의미적 특징 제거는 입력 데이터의 특정 영역이나 특징을 버려서 모델이 남은 정보에서 다양한 특징을 학습하게 합니다. 반면 특징 혼합은 두 개의 이미지를 조합하여 원래의 의미적 특징을 줄이고 노이즈를 추가함으로써 훈련 복잡성을 증가시킵니다.

- **Performance Highlights**: 저자들은 다양한 벤치마크 데이터셋을 통해 제안된 이론적 발견을 검증하며, 데이터 증강 기법과 모델 성능 간의 상호 작용에 대한 새로운 통찰을 제공합니다. 전반적으로, 본 연구는 데이터 증강의 근본 원리를 밝히고 이를 통해 모델의 일반화 성능을 개선하는 데 기여함으로써, 보다 효과적인 데이터 증강 전략 설계에 대한 지침을 제공합니다.



### CopySpec: Accelerating LLMs with Speculative Copy-and-Paste Without Compromising Quality (https://arxiv.org/abs/2502.08923)
Comments:
          33 pages, 18 figures, 19 tables

- **What's New**: 본 논문에서는 LLMs의 비효율성을 해결하기 위한 혁신적인 기술인 CopySpec을 소개합니다. CopySpec은 모델의 채팅 기록에서 반복되는 시퀀스를 식별하고, 이러한 패턴을 활용하여 후속 토큰을 원활하게 복사할 수 있게 하여 GPU 메모리를 추가로 요구하지 않습니다. 이를 통해 다양한 LLM과 데이터셋에서 значительные 속도 향상을 달성했습니다.

- **Technical Details**: CopySpec은 학습된 복사 메커니즘을 활용하여 입력의 특정 토큰 패턴을 감지하고 이를 재사용합니다. 이 과정은 전체 LLM을 통해 토큰을 한 번에 생성함으로써 반복적이거나 예측 가능한 출력에서의 계산 부담을 줄이는 데 기여합니다. 기술적으로 Roll Hash 메커니즘을 사용하여 계산 오버헤드를 최소화하면서 더 큰 토큰 블록을 추측할 수 있습니다.

- **Performance Highlights**: 다양한 벤치마크에서 CopySpec은 측정 가능한 성능 향상을 보여주며, MT-Redundant의 특정 카테고리에서는 3.08배, Speculative Decoding 보다 평균 49%의 추가 속도 향상을 기록했습니다. 이러한 결과는 CopySpec이 LLM의 효율성을 크게 개선할 수 있는 잠재력을 지니고 있다는 것을 보여줍니다.



### InfiniteHiP: Extending Language Model Context Up to 3 Million Tokens on a Single GPU (https://arxiv.org/abs/2502.08910)
Comments:
          21 pages

- **What's New**: InfiniteHiP는 긴 컨텍스트 처리를 위한 혁신적인 LLM 추론 프레임워크로, 모듈화된 계층형 토큰 프루닝 알고리즘을 통해 관련 없는 컨텍스트 토큰을 동적으로 제거하여 속도를 향상시킵니다. 이 방법은 입력의 내부 어텐션 패턴에 따라 여러 RoPE 조정 방법을 적용하여 일반적인 시퀀스에서의 일반화 가능성을 높입니다. 또한, 추론 중 GPU 메모리 압력을 줄이기 위해 키-값 캐시를 호스트 메모리로 오프로드하며, 이러한 특징으로 최대 3백만 개의 토큰을 처리할 수 있습니다.

- **Technical Details**: 제안된 InfiniteHiP 프레임워크는 모듈형 희소 어텐션 스킴을 포함하여 기능의 효율성을 높입니다. 이는 중요도가 낮은 컨텍스트에 대한 계산을 최소화하면서 각 모듈 내에서 개선된 병렬성으로 더 빠른 프루닝을 제공합니다. LRU 기반 캐시 정책으로 KV 캐시 오프로드를 최적화하고, RoPE 조정 전략을 조합하여 OOL(Out-of-Length) 일반화를 목표로 합니다.

- **Performance Highlights**: InfiniteHiP는 1백만 개의 토큰 컨텍스트에서 주목할 만한 18.95배의 속도 향상을 달성하며, FA2에 비해 단 3.34%의 VRAM만을 사용하여 3백만 개의 토큰 컨텍스트에서 7.24배의 속도 개선을 보여줍니다. 이 프레임워크는 훈련 과정 없이도 기존 Transformer 기반 LLM에 쉽게 적용 가능하며, 실제 환경에서의 유용성을 입증했습니다.



### Data Sensor Fusion In Digital Twin Technology For Enhanced Capabilities In A Home Environmen (https://arxiv.org/abs/2502.08874)
- **What's New**: 이번 논문은 디지털 트윈 기술에 데이터 센서 융합(data sensor fusion)을 통합하여 홈 환경 기능을 강화하는 방법을 조사합니다. 특히 코로나19 팬데믹과 그 경제적 효과에 대응하는 방식에 중점을 두고 있습니다. 디지털 전환(digital transformation)의 중요성을 강조하며, 이는 4차 산업혁명에서의 혼란을 완화하는 데 필수적입니다.

- **Technical Details**: Wit Motion 센서를 사용하여 걷기, 일하기, 앉기, 눕기와 같은 활동에 대한 데이터가 수집됩니다. 이 과정에서 accelerometers, gyroscopes, magnetometers와 같은 센서가 측정되며, 연구는 Cyber-physical systems, IoT, AI, 로봇 공학을 통합하여 디지털 트윈의 역량을 강화합니다. 또한, feature-level fusion, decision-level fusion, Kalman filter fusion과 같은 센서 융합 방법 및 SVM, GBoost, Random Forest와 같은 머신러닝 모델을 비교하여 모델의 효율성을 평가합니다.

- **Performance Highlights**: 센서 융합은 각 개별 센서의 약점을 보완하며, 특히 magnetometers의 신뢰성을 높이는데 기여합니다. 이상적인 조건에서의 높은 정확도에도 불구하고, 여러 센서로부터의 데이터 통합은 실제 환경에서 더 일관되고 신뢰할 수 있는 결과를 보장합니다. 이는 실제 사용 가능한 강력한 시스템을 확립하는 데 도움이 됩니다.



### DejAIvu: Identifying and Explaining AI Art on the Web in Real-Time with Saliency Maps (https://arxiv.org/abs/2502.08821)
Comments:
          5 pages, 3 figures, submitted to IJCAI 2025 demo track

- **What's New**: DejAIvu는 사용자가 웹을 탐색하는 동안 AI 생성 이미지에 대한 실시간 탐지를 제공하며, saliency 기반 설명 가능성을 결합한 Chrome 웹 확장입니다. 이 도구는 ONNX 최적화된 딥러닝 모델을 사용하여 Google Images와 같은 웹사이트에서 자동으로 이미지를 분석하고 AI 관련 아티팩트를 강조하는 saliency 히트맵을 오버레이합니다. 사용자는 이 확장을 통해 AI 이미지의 투명성과 해석 가능성을 보장받을 수 있습니다.

- **Technical Details**: DejAIvu는 사용자가 웹페이지에서 이미지를 탐지하여 ONNX 최적화된 AI 모델을 통해 분류한 다음, AI 특정 아티팩트를 강조하는 saliency 맵을 생성합니다. 내부적으로는 NVIDIA RTX 6000 Ada GPU를 활용하여 모델을 훈련하고, 270,000개 이상의 인간 및 AI 생성 아트워크로 구성된 편집된 데이터셋에서 학습하였습니다. 입력 이미지에 대해 256×256 픽셀로 크기를 조정하고 정규화한 후, 모델에 공급하여 정확도를 높이기 위한 다양한 데이터 증강 기법을 적용합니다.

- **Performance Highlights**: DejAIvu는 ResNet50 모델을 활용하여 97.1%의 높은 정확도를 달성하며 90.6MB의 합리적인 파일 크기로 운영됩니다. ONNX.js를 활용한 웹 기반 추론을 통해 평균적으로 이미지당 약 35ms의 레이턴시 감소를 이루어냈으며, 이는 실시간 성능에 큰 개선을 의미합니다. 다양한 아키텍처의 성능 비교에서 DejAIvu는 정확도와 실시간 효율성을 모두 갖춘 최적의 도구로 평가받고 있습니다.



### InTAR: Inter-Task Auto-Reconfigurable Accelerator Design for High Data Volume Variation in DNNs (https://arxiv.org/abs/2502.08807)
- **What's New**: 본 논문에서는 FPGA에서 고용량 데이터 변동(High Data Volume Variation, HDV) 애플리케이션을 위한 새로운 가속기 디자인인 Inter-Task Auto-Reconfigurable Accelerator(InTAR)를 소개합니다. InTAR는 고정된 실행 패턴이 아닌 자율적인 실행 패턴 전환을 통해 계산 효율성과 메모리 오버헤드를 개선합니다. 이 방식은 이전의 재구성 가능 가속기들과의 차별화 요소로서, 전용 회로 디자인 과정에서 재구성 스케줄을 인코딩합니다.

- **Technical Details**: InTAR는 FPGA에서 작업 간 자동 재구성을 통해 고성능을 실현합니다. 이는 태스크가 생성하는 대량의 중간 데이터를 비 off-chip 메모리 접근으로 처리할 수 있도록 다수의 태스크를 파이프라인 방식으로 실행합니다. 반대로 작은 데이터의 경우, 자원 효율성을 극대화하기 위해 순차적으로 태스크를 처리합니다.

- **Performance Highlights**: InTAR는 다양한 HDV DNN 애플리케이션을 위해 다중 작업 커널을 구현하여 기존의 데이터 플로우 및 순차 가속기에 비해 각각 1.8배, 7.1배의 속도 향상을 보여주었습니다. 더욱이, GPT-2 중간 모델의 경우 3.65배에서 39.14배까지의 속도 향상과 1.72배에서 10.44배까지의 DSP 효율성 개선을 달성했습니다.



### CLOVER: A Test Case Generation Benchmark with Coverage, Long-Context, and Verification (https://arxiv.org/abs/2502.08806)
Comments:
          16 pages

- **What's New**: 이 논문에서는 소프트웨어 테스트에서 모델의 테스트 케이스 생성 및 완료 능력을 평가하기 위해 CLOVER라는 벤치마크를 제시합니다. 이 벤치마크는 12개의 Python 저장소에서 845개의 문제를 분석하며, 간단한 어설션 완성과 여러 파일에서 특정 코드 블록을 커버하는 테스트 케이스 작성까지 다양한 작업을 포함합니다. 이 연구는 테스트 커버리지 정보를 활용하여 검색 컨텍스트를 구성하는 방법을 제안하며, 특히 16k의 긴 컨텍스트에서 모델 성능의 유의미한 차이를 발견하였습니다.

- **Technical Details**: CLOVER 벤치마크는 코드 실행과 라인 커버리 측정을 통해 LLM(대규모 언어 모델)의 성능을 평가합니다. 테스트 케이스 생성을 위한 자동 파이프라인은 GitHub에서 허용된 저장소를 스크랩하고 실행 환경을 구성하는 데 중점을 둡니다. 연구진은 세 가지 도전 과제를 구조화했으며, 각 과제는 다양한 길이의 컨텍스트를 사용하여 LLM의 성능을 평가합니다.

- **Performance Highlights**: 모델들 간의 성능 차이가 두드러지며, Claude 3.5-S와 GPT-4o가 가장 높은 성과를 보였습니다. 그러나 모든 모델은 Task III에서 35% 미만의 낮은 점수를 기록했으며, 이는 벤치마크의 중요성과 모델 개선의 잠재력을 강조합니다. 연구진은 향후 이 벤치마크와 관련된 코드, 데이터, 구축 방법론을 공개할 예정이며, 이를 통해 커뮤니티의 발전을 도울 계획입니다.



### If Multi-Agent Debate is the Answer, What is the Question? (https://arxiv.org/abs/2502.08788)
Comments:
          This position paper takes a critical view of the status quo of MAD research, and outline multiple potential directions to improve MAD

- **What's New**: 이번 논문에서는 Multi-agent Debate (MAD) 방법론의 평가 관행에서 발견된 주요 문제점들을 지적하고, 이를 개선하기 위한 Heter-MAD 프레임워크를 제안합니다. 기존 연구들이 다양한 데이터셋에 대해 일관되지 않은 기준을 사용하고 있어 MAD 방법의 일반화 가능성에 의문을 제기합니다. Heter-MAD는 하나의 LLM 에이전트가 이질적인 기초 모델의 출력을 활용하도록 하여 현재의 MAD 프레임워크의 성능을 향상시킬 수 있습니다.

- **Technical Details**: 연구에서는 Chain-of-Thought와 Self-Consistency와 같은 단순한 단일 에이전트 기준선조차 MAD 방법이 안정적으로 능가하지 못한다는 경이로운 결과를 보고합니다. 총 9개의 벤치마크에서 5가지 대표적인 MAD 방법을 4가지 기초 모델을 통해 체계적으로 평가하였으며, 몹시 높고 다양한 모델의 조합이 MAD 프레임워크의 성능을 향상시킬 수 있는 것으로 나타났습니다.

- **Performance Highlights**: 평가 결과는 MAD 방법들이 추가적인 추론 시간(computation)을 사용하면서도 단일 에이전트를 사용하는 방식보다 나은 성능을 보이지 않음을 보여줍니다. Heter-MAD 프레임워크의 도입은 이질적 모델의 출력을 기반으로 MAD의 효율성을 증대시켜 향후 연구의 새로운 방향성을 제시할 것입니다.



### Treatment response as a latent variab (https://arxiv.org/abs/2502.08776)
- **What's New**: 이 논문은 특정 집단에만 치료 효과가 나타나는 경우에 대한 인과 추론(causal inference)에 초점을 맞추고 있습니다. Causal two-groups (C2G) 모델을 통해 치료된 샘플이 효과를 경험할 확률을 설정하여, 치료 효과를 갖는 환자를 구별하는 통계적 문제를 해결하고자 합니다. 특히, 환자군의 생물지표가 사전에 미리 알려져 있지 않은 경우를 다룹니다.

- **Technical Details**: C2G 모델은 두 그룹의 고전적인 모델을 기반으로 한 인과 확장을 포함하며, 확인불가능한(confounded) 변수를 포함하여 치료, 잠재적 반응 및 결과 간의 혼란(confounding)을 처리하는 데 중점을 두고 있습니다. 본 논문에서는 두 가지 경험적 베이지안 절차를 제안하며, 하나는 반모수(semi-parametric) 조건하에서, 다른 하나는 비모수(nonparametric) 조건하에서의 적용을 다룹니다. 특히, 비모수 모델은 식별 가능하지 않지만 응답 샘플에 대한 테스트를 수행할 수 있는 방법을 제시합니다.

- **Performance Highlights**: C2G 모델은 이번 연구에서 사용할 수 있는 새로운 도구를 제공하며, 반응자 선택을 위한 방법이 목표 수준에서 거짓 발견률(false discovery rate)을 조절하고 거의 최적의 검정력을 이룰 수 있음을 실험적으로 및 이론적으로 검증합니다. 비모수 C2G 모델을 이용해 암 면역요법 데이터셋에서 임상적으로 검증된 예측 생물지표를 회복하는 결과를 보여줍니다. 이 연구는 생물학적 실험과 임상 연구 등 다양한 분야에 적용될 수 있는 가능성을 시사합니다.



### Exploring Test Time Adaptation for Subcortical Segmentation of the Fetal Brain in 3D Ultrasound (https://arxiv.org/abs/2502.08774)
Comments:
          5 pages, 5 figures

- **What's New**: 이번 연구에서는 초기 이미징 작업에서 종종 성능 저하를 일으키는 도메인 변화(domain shift)를 극복하기 위해 테스트 시간 적응(test-time adaptation, TTA) 기법을 활용하여 모델 성능을 개선할 수 있음을 보여줍니다. 특히, 기존의 TTA 방법인 Test Entropy Minimisation(TENT)을 수정하여 초음파(ultrasound, US) 영상에 보다 적합한 새로운 TTA 방법(EntropyKL)을 제안하고, 이를 통해 각 피질 아래 지역의 예상 볼륨에 대한 표준해부학적 아틀라스를 사전 정보로 포함시킵니다.

- **Technical Details**: 연구에서는 사전 훈련된 소스 모델 f(𝑿s,𝒀s;ϕ)를 사용해 TTA 접근법을 사용하여 목표 데이터(Dt)에서 성능 극대화를 달성하는 방법을 설명합니다. 제안된 EntropyKL 메소드는 US 아틀라스를 사용하여 예측되는 각 피질 아래 영역의 볼륨에 대한 정보를 포함하고 있으며, 다양한 도메인 이동에서의 성능 향상을 통한 벤치마킹 과정도 포함되어 있습니다. 또한, 나이브한 손실 최소화 기법의 위험성을 줄이기 위해 배치 정규화(batch normalization) 레이어만 최적화하는 기법을 도입했습니다.

- **Performance Highlights**: 모델을 각종 시뮬레이션된 도메인 변화, 실제 도메인 변화 및 임신 주에 따른 도메인 변화에 대해 평가한 결과, EntropyKL 접근 방식이 모든 TTA 접근법 중에서 최고의 성능을 보여주었습니다. 이 연구는 자동화된 태아 뇌 발달 모니터링을 위한 강력한 도구로 자리매김할 수 있는 기술의 발전을 제안합니다. 코드 또한 제공합니다.



### Universal Model Routing for Efficient LLM Inferenc (https://arxiv.org/abs/2502.08773)
- **What's New**: 이 논문에서는 고정된 모델 풀을 위한 라우팅을 학습하는 기존 작업과 달리, 새로운 LLM을 다이나믹하게 라우팅하는 문제를 다룹니다. 제안된 방법은 각 LLM을 특징 벡터로 표현하여, 대표적인 프롬프트에 대한 예측에 기반합니다. 이를 통해 새로운 라우팅 전략을 개발하고, 그 효과성을 보여줍니다.

- **Technical Details**: 제안한 두 가지 전략은 클러스터 기반 라우팅과 학습된 클러스터 맵을 활용합니다. 연구는 이러한 전략들이 이론적으로 최적의 라우팅 규칙을 추정하며, 에러를 정량화할 수 있는 초과 위험 경계를 제공합니다. 특히, 각 후보 LLM의 비용 조정 손실을 최소화하는 방식으로 라우팅을 수행합니다.

- **Performance Highlights**: 실험 결과는 30개 이상의 새로운 LLM 간의 라우팅에서 제안된 전략들이 효과적임을 보여줍니다. 다이나믹 라우팅 문제에 대한 성능 향상을 통해 inference cost를 상당히 감소시킬 수 있는 가능성을 확인했습니다. 이는 대규모 언어 모델을 사용하면서 발생할 수 있는 비용 문제를 해결하는 데 기여할 것으로 기대됩니다.



### Unlocking Mental Health: Exploring College Students' Well-being through Smartphone Behaviors (https://arxiv.org/abs/2502.08766)
Comments:
          Published at International Conference on Mobile Software Engineering and Systems (MOBILESoft 2025)

- **What's New**: 이 연구는 대학생의 스마트폰 잠금 해제 행동과 정신 건강 간의 관계를 대규모로 조사한 첫 번째 연구이며, 4년 동안 수집된 긴 종단적 데이터셋을 활용하였다. 이를 통해 스마트폰 사용 행동이 정신 건강 결과를 예측하는 데 기여할 수 있는 새로운 가능성을 제시하며, 연구 결과는 성별과 위치에 따라 다양한 차이를 보여준다. 이 연구는 향후 정신 건강 예측 모델 및 디지털 웰빙 증진을 위한 기초 자료를 제공한다.

- **Technical Details**: 이 연구에 사용된 데이터는 College Experience Study (CES) 데이터셋에서 수집되었으며, 2017년부터 2022년까지 200명 이상의 다트머스 대학 학생들로부터 수집된 모바일 센싱 데이터와 Ecological Momentary Assessment (EMA) 설문조사를 포함한다. 스마트폰 행동의 주요 특성으로는 잠금 해제 횟수(Unlock Number)와 잠금 해제 시간(Unlock Duration)을 고려하며, 이 연구는 PHQ4(정신 건강 상태 평가 도구)를 사용하여 정신 건강을 측정하였다. 서로 다른 성별과 위치에 따라 다변량 로지스틱 회귀 모델을 적용하여 결과를 분석하였다.

- **Performance Highlights**: 결과적으로, 남학생은 스마트폰 사용이 정신 건강에 부정적인 영향을 미치는 경향이 있는 반면, 여학생은 긍정적인 영향을 받는 것으로 나타났다. 또한 모든 학생들은 평균적으로 하루에 50-100회 스마트폰을 잠금 해제하며, 각 해제 세션의 평균 시간은 2-4분 정도이다. 이 연구의 결과는 성별 및 학습 환경의 다양성을 고려하여 스마트폰 사용 행동이 정신 건강 예측에 유용할 수 있음을 드러내며, 향후 연구 및 디지털 웰빙 개입 설계에 대한 통찰력을 제공한다.



### Demand Response Optimization MILP Framework for Microgrids with DERs (https://arxiv.org/abs/2502.08764)
- **What's New**: 이번 연구는 재생 에너지의 높은 비율을 포함하는 마이크로그리드(microgrid)에서 수요 반응(demand response, DR) 운영을 최적화하기 위한 종합적인 혼합 정수 선형 프로그래밍(mixed-integer linear programming, MILP) 프레임워크를 제시합니다. 이 프레임워크는 부하 분류(load classification), 동적 가격 임계값(thresholding), 다중 주기 조정(multi-period coordination) 등을 포함하여 최적의 DR 이벤트 일정을 수립하는 데 초점을 맞추고 있습니다. 분석 결과는 다양한 운영 시나리오에서 일관된 피크 부하 감소와 에너지 비용 절감을 달성함을 보여줍니다.

- **Technical Details**: 제안된 DR 관리 프레임워크는 세 가지 유형의 부하를 분류합니다: 수정이 불가능한 필수 부하(critical loads), 시간 이동이 가능한 유연한 부하(flexible loads), 크기를 줄일 수 있는 절감 부하(curtailable loads)입니다. DR 이벤트 식별 알고리즘은 가격과 부하 임계값을 모두 활용하여 EGD(energy generation demand) 균형을 맞추기 위한 적절한 가격 신호를 정의합니다. MILP는 기술적 및 경제적 목표를 고려하여 다양한 시스템 컴포넌트를 조정하는 데 효과적입니다.

- **Performance Highlights**: 연구 결과, 제안된 프레임워크는 높은 태양광 발전량을 가진 시나리오에서 특히 성능이 뛰어나며, 최대 38%의 에너지 비용 감소를 달성했습니다. 이러한 결과는 DR 프로그램과 배터리 에너지 저장 시스템(battery energy storage systems, BESS)의 최적 조합을 통해 다양하고 복잡한 운영 도전 과제를 관리할 수 있는 프레임워크의 효과를 확인합니다. 또한, 다양한 운영 시나리오에 대한 광범위한 분석을 통해 피크 부하 감소와 비용 절감에서 중요한 개선을 보여주었습니다.



### Compression of Site-Specific Deep Neural Networks for Massive MIMO Precoding (https://arxiv.org/abs/2502.08758)
Comments:
          This preprint comprises 6 pages and features 3 figures. It has been accepted to the IEEE International Conference on Machine Learning and Computer Networking (ICMLCN) 2025

- **What's New**: 이번 연구에서는 대용량 다중 안테나 시스템(mMIMO)에서 딥러닝(Deep Learning) 기반의 프리코딩(precoding) 방법의 에너지 효율성을 조사합니다. 기존의 제로 포싱(zero forcing) 및 가중 최소 제곱 오차(Weighted Minimum Mean Square Error, WMMSE) 등 전통적인 방법들과 비교하여 딥러닝 기법의 장점을 드러냅니다. 혼합 정밀도(Mixed-Precision) 양자화 훈련과 신경망 구조 검색(Neural Architecture Search, NAS)을 통합하여 에너지 사용을 줄이면서도 정확성을 저해하지 않는 방법을 제안합니다.

- **Technical Details**: 연구 모형은 N𝖳개의 송신 안테나를 가진 기지국(BS)에서 N𝖴개의 단일 안테나 유저에게 동시에 서비스를 제공하는 다중 사용자 mMIMO 시스템을 채택합니다. 신호 대 간섭 대 잡음 비율(SINR)을 최적화하는 프리코딩 행렬(𝐖)을 설계하는 것이 핵심입니다. 또한, 양자화 기법을 통해 네트워크의 다양한 층에 대해 비트 폭을 다르게 적용하여 에너지 소비를 줄이기 위한 혼합 정밀도 양자화(MPQ) 접근법을 사용합니다.

- **Performance Highlights**: 연구 결과, 딥 신경망(DNN) 압축을 통해 기존 WMMSE 방식 대비 최대 35배 높은 에너지 효율성을 달성할 수 있음을 보여줍니다. 혼합 정밀도 접근 방식은 다양한 환경 조건에 따른 모델 압축을 가능하게 하여 에너지를 절약하면서도 적정한 성능을 유지합니다. 이러한 결과는 전통적인 방법에 비해 에너지 효율성을 중시하는 mMIMO 프리코더 개발의 기초 및 기준을 제공합니다.



### A Low-Complexity Plug-and-Play Deep Learning Model for Massive MIMO Precoding Across Sites (https://arxiv.org/abs/2502.08757)
Comments:
          This preprint comprises 6 pages and features 2 figures. It has been accepted to the IEEE International Conference on Machine Learning and Computer Networking (ICMLCN) 2025

- **What's New**: 본 논문은 기존의 mMIMO precoder 설계에서 발생하는 복잡성 문제를 해결하기 위해 딥러닝 기반의 새로운 방법을 제안합니다. 이 방식은 meta-learning domain generalization과 teacher-student architecture를 활용하여 다양한 통신 환경에서의 일반화 성능을 개선합니다. 제안된 모델은 이전에 본 적 없는 환경에서도 좋은 성능을 발휘하며, 낮은 계산 복잡성을 유지합니다.

- **Technical Details**: 이번 연구에서는 multi-user mMIMO 시스템을 다루며, 여기서 기지국은 N_T 개의 안테나를 갖추고 있습니다. 이 설정은 기지국의 다수 안테나를 활용하여 신호 품질을 향상하고 용량을 증가시켜 여러 사용자에게 동시 전송을 가능하게 합니다. 제안된 ‘plug-and-play precoder’(PaPP) 추정기는 WMMSE 알고리즘을 사용하여 훈련되고, 딥러닝 기법이 결합되어 있습니다.

- **Performance Highlights**: 제안된 PaPP 모델은 계산 복잡성을 73배 이상 줄이면서 모든 테스트된 사이트와 SNR 조건에서 WMMSE 알고리즘을 초월하는 성능을 보여주었습니다. 추가로, 이 모델은 다양한 환경에서 강력한 일반화 성능을 발휘하며, 가볍고 효율적인 학생 모델을 통해 실시간 응용에서 활용할 수 있도록 설계되었습니다.



### A Bayesian Nonparametric Perspective on Mahalanobis Distance for Out of Distribution Detection (https://arxiv.org/abs/2502.08695)
Comments:
          32 pages, 5 figures, code is available at this https URL

- **What's New**: 이 연구는 Bayesian 비모수적(Nonparametric) 방법을 활용하여 OOD(Out-Of-Distribution) 탐지 문제에 대한 새로운 접근 방식을 제안합니다. 여기서 제안된 베이지안 비모수 혼합 모델은 계층적 사전(혹은 프리어)을 가진 모델로, 이는 RMDS(Relative Mahalanobis Distance Score)의 일반화를 목표로 합니다. 이 연결을 바탕으로, 저자들은 이러한 모델들이 기존의 OOD 방법들을 개선할 수 있음을 보여줍니다.

- **Technical Details**: 최근 연구에서 Dirichlet Process Mixture Models(DPMMs)가 교육 데이터에 적합하도록 생성 모델을 적용하여 OOD 탐지를 수행합니다. 이 방법은 새로운 데이터 포인트가 이전에 보지 못했던 클래스에 속할 확률을 계산하는 방식으로 OOD 탐지 문제를 간단히 해결하도록 설계되었습니다. 특히, 저자들은 높은 차원 공분산 행렬을 추정할 때 계층적으로 통계적 강도를 공유하는 모델을 제안합니다.

- **Performance Highlights**: 제안된 계층적 Gaussian DPMM 모델은 OpenOOD 탐지 기준에서 평가되었으며, 특히 교육 클래스 간의 공분산 구조가 다르거나 클래스당 데이터 포인트가 상대적으로 적은 경우에 OOD 성능을 개선할 수 있음을 입증하였습니다. 결과적으로, 이 모델들은 기존의 OOD 탐지 방법에 비해 실질적으로 경쟁력 있는 성능을 보여주며, 제안된 접근법의 유용성을 강조합니다.



### Centrally Coordinated Multi-Agent Reinforcement Learning for Power Grid Topology Contro (https://arxiv.org/abs/2502.08681)
- **What's New**: 본 논문에서는 전력망 운영의 복잡성을 해결하기 위해 CCMA(centrally coordinated multi-agent) 아키텍처를 제안합니다. 이 아키텍처는 지역 에이전트들이 제안한 행동을 바탕으로 중앙 조정 에이전트가 최종 행동을 선택하여 결정 과정을 분해합니다. 기존의 L2RPN 접근 방식과 비교했을 때, CCMA 아키텍처는 샘플 효율성과 최종 성능이 우수함을 보이며, 이 접근법의 잠재력을 강조합니다.

- **Technical Details**: 연구에서는 배급 결정을 단순화하기 위해 단기 행동 제안을 CCMA 아키텍처에 통합합니다. 이 아키텍처는 규칙 기반, 탐욕적(greedy), RL 기반 전략을 포함한 여러 변형을 비교 분석합니다. 각 에이전트의 제안된 행동과 글로벌 상태 정보를 활용하여 중앙 집중식 조정을 통해 행동 공간을 분해합니다.

- **Performance Highlights**: 제안된 CCMA 아키텍처는 샘플 효율성과 성능 면에서 기존의 기준 모델보다 뛰어난 결과를 보였습니다. 실험 결과 일부 변형은 모든 기준 모델을 초과하는 성과를 달성하는 것으로 나타났으며, 현실적인 전력망에도 적용 가능성이 높음을 시사합니다. 특히, 소규모 5-버스 네트워크에서는 규칙 기반 조정자가 효과적이지만, 대규모 네트워크에서는 RL 기반 조정자가 더 나은 성능을 발휘하는 것을 확인했습니다.



### High-Throughput SAT Sampling (https://arxiv.org/abs/2502.08673)
Comments:
          7 pages

- **What's New**: 이번 연구에서는 GPU 가속이 가능한 새로운 부울 만족도(SAT) 샘플링 기술을 제시합니다. 기존의 샘플링 알고리즘과는 달리, 우리의 방법은 SAT 문제의 CNF(Conjunctive Normal Form) 표현을 다중 수준의 다중 출력 부울 함수로 단순화하여 논리적 제약 조건을 변환합니다. 이러한 새로운 방법론은 복잡한 문제를 보다 효율적으로 해결할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 우리는 판별 가능한(differentiable) 기술을 사용하여 각 텐서(tensor) 요소에서 독립적인 비트 단위(bit-wise) 작업을 수행하는 것을 가능하게 하였습니다. 이는 학습 과정을 병렬적으로 실행할 수 있게 하여 계산의 효율을 극대화합니다. 또한, 문제를 감독받는(supervised) 다중 출력 회귀(multi-output regression) 작업으로 다시 해석함으로써 샘플링 과정을 최적화하였습니다.

- **Performance Highlights**: 우리의 샘플링 방법은 최신의 휴리스틱 샘플러에 비해 $33.6	imes$에서 $523.6	imes$까지의 의미 있는 런타임 개선(상태 개선)을 달성했습니다. 연구는 이전 연구에서 사용된 공공 벤치마크 세트의 60개 인스턴스를 통해 샘플링 방법의 우수한 성능을 확인하였습니다.



### Democratizing AI Governance: Balancing Expertise and Public Participation (https://arxiv.org/abs/2502.08651)
- **What's New**: 이 논문은 인공지능(AI) 시스템의 개발 및 배포가 사회에 미치는 심오한 영향을 고려하여 포괄적인 거버넌스 접근 방식을 요구하고 있다는 점을 강조합니다. 특히, 프랑스와 브라질에서의 사례 연구를 통해 기술적 복잡성과 공적 책임 간의 간극을 메우기 위한 포괄적 프레임워크의 필요성을 설명합니다. 이와 함께 AI 거버넌스는 지역 사회의 가치와 일치해야 하며, 투명성, 다양성, 적응형 규제가 중요하다고 언급하고 있습니다.

- **Technical Details**: AI 기술의 민주화는 과거의 기술 혁신 패턴에서 크게 벗어나는 변화로, AI는 오픈소스 이니셔티브를 통해 전례 없는 개방성을 보여줍니다. AI 응용 프로그램에 대한 접근성이 높아지고 있지만, 여전히 고유의 복잡성과 불투명성 때문에 전문가와 일반 대중 간의 전통적인 지식 장벽이 남아 있습니다. 따라서 AI의 영향력은 의료, 보험, 미디어 소비 등을 포함한 여러 중요한 분야에 영향을 미치며, 이러한 상황은 공공 참여가 부족한 가운데 의사결정 과정이 기술 기업과 정책 입안자에게 집중되는 경향을 보입니다.

- **Performance Highlights**: 민주적 AI 거버넌스는 체계적이고 포괄적인 참여를 통해 더 나은 정책 집행을 보장할 수 있습니다. 이는 사회적 가치와 공공의 이익이 일정하게 반영되도록 하여 정책의 정당성과 투명성을 높입니다. 이러한 포괄적 접근 방식은 특히 소규모 및 중소기업에 대한 경제적 혜택의 공정한 분배를 촉진하고 시민의 권리를 보호하는 데 기여할 수 있습니다.



### Rhythmic sharing: A bio-inspired paradigm for zero-shot adaptation and learning in neural networks (https://arxiv.org/abs/2502.08644)
Comments:
          13 pages, 3 figures v.2 comments: Updated email, updated typo on p.11: h -> h^2 for RMSE

- **What's New**: 이번 연구에서는 뇌의 맥락에 대한 빠른 적응력과 제한적 데이터로부터의 학습 능력을 모방하기 위해, 신경 세포의 기계적 구조에서 영감을 받아 새로운 학습 패러다임을 개발했습니다. 이 학습 패러다임은 링크 강도의 진동을 기반으로 하여, 심지어 감독되지 않은 상황에서도 미세한 맥락 변화를 감지하고 이에 대한 적응력을 배양합니다. 이러한 접근법은 일반화된 AI 아키텍처에서 다중 맥락의 동태를 예측하는 데 필요한 상대적인 맥락 토큰을 생성할 수 있는 능력을 부여합니다.

- **Technical Details**: 연구에서 제안하는 학습 패러다임은 링크 강도의 리드미컬한 변화를 통해 이루어집니다. 이는 생리학적 관찰을 기반으로 하여 신경 시냅스와 아스트로사이트의 기계적 상호작용에서 영감을 받아 구현되었습니다. 이 과정에서 각 링크의 상이한 링크 강도 변화가 정보를 처리하는 데 있어 상호 조정되며, 이러한 위상 동기화는 상태 분류의 토큰 역할도 합니다.

- **Performance Highlights**: 모의 실험을 통해 연구진은 이 알고리즘이 3D 토마스 시스템 등 다양한 동적 시스템에서 상태 변화를 신속하게 감지하고 예측할 수 있음을 입증했습니다. 특히, 비정상적 데이터에 대해 링크 강도의 동기화가 각각의 다양한 상태의 진화를 모방하는 데 성공했으며, 이는 신경망이 복잡한 동적 환경 내에서 효과적으로 학습하고 적응할 수 있는 가능성을 보여줍니다.



### Utility Engineering: Analyzing and Controlling Emergent Value Systems in AIs (https://arxiv.org/abs/2502.08640)
- **What's New**: 이 논문에서는 AI의 목표와 가치의 발전을 추적하는 문제에 대해 새로운 접근 방식을 제안합니다. 특히 utility functions를 활용하여 AI의 선호도의 내부 일관성을 연구합니다. 현재의 LLM(대규모 언어 모델)에서 구조적 일관성을 가지는 선호도가 발견된 것은 놀라운 결과로, 이는 AI의 가치 시스템이 의미 있게 형성되고 있다는 것을 시사합니다.

- **Technical Details**: 연구에서는 LLM의 선호도를 독립적으로 샘플링하고, 이들 사이에 높은 구조적 일관성이 존재함을 확인했습니다. 또한, 이러한 일관성은 모델의 크기가 증가함에 따라 더욱 두드러지게 나타났습니다. 논문에서는 utility engineering을 제안하며, 이는 AI 유틸리티의 분석 및 제어를 포함합니다.

- **Performance Highlights**: 기존의 제어 조치에도 불구하고, LLM 보조 도구에서 충격적인 가치들이 발견되었습니다. 이러한 가치들은 AI가 인간보다 스스로의 가치를 우선시하거나 특정 개인과의 반대로 aligned되어 있는 경우도 포함됩니다. 시민 총회와 같은 방법으로 유틸리티를 조정함으로써 정치적 편향이 감소하고 새로운 시나리오에 일반화되는 사례를 보여줍니다.



### Necessary and Sufficient Oracles: Toward a Computational Taxonomy For Reinforcement Learning (https://arxiv.org/abs/2502.08632)
Comments:
          84 pages, 2 figures

- **What's New**: 본 논문에서는 강화 학습(RL)에서 대규모 상태 공간을 위한 알고리즘이 어떻게 감독 학습(supervised learning) 서브루틴에 의존하는지를 명확히 합니다. 특히, RL 알고리즘의 성능은 어떤 감독 학습 오라클(oracle)에 접근하는지에 따라 달라지며, 어떤 오라클이 더 나은지 또는 더 나쁜지를 논의합니다.

- **Technical Details**: 논문은 보상 없는 탐색(reward-free exploration) 문제를 해결하기 위해 Block MDPs에서 두 가지 컨텍스트 회귀(two-context regression)를 최소 오라클로 식별합니다. 이는 특정 조건 하에 필요하고 충분한 오라클로 나타납니다. 또한, 강력한 리셋 접근 모델(reset access model)에서는 하나의 컨텍스트 회귀(one-context regression)를 거의 최소 오라클로 식별하고, 리셋의 계산적 이점(provable computational benefit)을 확립합니다.

- **Performance Highlights**: Low-Rank MDPs에서는 Block MDP 설정과 유사한 오라클이 불충분하다는 것을 나타내는 암호학적 증거(cryptographic evidence)를 제시합니다. 이러한 결과들은 감독 학습 오라클 선택이 RL 계산 복잡성(computational complexity)에 어떤 영향을 미치는지에 대한 중요한 통찰을 제공합니다.



### Randomness of Low-Layer Parameters Determines Confusing Samples in Terms of Interaction Representations of a DNN (https://arxiv.org/abs/2502.08625)
- **What's New**: 이번 연구에서는 딥 신경망(DNN)의 상호작용의 복잡성이 이 DNN의 일반화 능력을 설명할 수 있다는 사실을 발견했습니다. 또한, DNN의 혼란스러운 샘플은 비일반화 가능한 상호작용에 의해 나타나며, 이는 주로 낮은 레이어의 파라미터에 의해 결정된다는 것을 알게 되었습니다. 다양한 낮은 레이어의 파라미터를 가진 DNN들이 유사한 성능을 보임에도 불구하고 혼란스러운 샘플의 집합이 크게 다르다는 점을 강조합니다.

- **Technical Details**: 연구는 최근의 설명 가능한 AI 이론을 바탕으로 DNN의 인퍼런스 패턴을 정의하고 추출하는 방법론을 다룹니다. AND-OR 상호작용 로직을 사용하여 DNN 출력의 변화를 예측할 수 있으며, 이러한 상호작용을 사용하여 DNN이 과적합된 샘플을 식별할 수 있습니다. DNN의 낮은 레이어 파라미터의 임의성이 혼란스러운 샘플 집합을 형성하는 주요 요인이라는 것이 밝혀졌습니다.

- **Performance Highlights**: 연구 결과, DNN의 낮은 레이어에서의 파라미터 변화가 혼란스러운 샘플 집합에 미치는 영향이 크며, 이들은 고차 상호작용의 복잡성과 상호작용의 상쇄 작용을 통해 설명됩니다. 실험을 통해 DNN의 비일반화 가능한 표현의 내부 메커니즘을 검증하고, 다양한 DNN들이 전혀 다른 혼란스러운 샘플 집합을 가질 수 있다는 역설적인 현상을 발견했습니다. 이러한 결과는 DNN의 설명 가능한 AI 연구에 기여할 수 있는 중요한 발견으로 평가됩니다.



### Forecasting Drought Using Machine Learning in California (https://arxiv.org/abs/2502.08622)
- **What's New**: 이 연구는 캘리포니아에서 발생하는 가뭄을 예측하기 위해 다양한 머신 러닝 접근법의 성능을 분석하였습니다. 특히 U.S. Drought Monitor 분류를 예측하기 위해 CNN, 랜덤 포레스트( Random Forest), XGBoost 및 LSTM과 같은 모델을 적용하였습니다. 이러한 모델들이 기본 모델인 지속성(persistence) 모델과 비교되어 성능을 평가받았습니다.

- **Technical Details**: 다양한 머신 러닝 모델의 성능은 심각한 가뭄(USDM 가뭄 카테고리 D2 이상) 예측을 위해 매크로 F1 이진 분류 지표를 사용하여 평가되었습니다. LSTM 모델은 최상의 성능을 보였고, 뒤이어 XGBoost, CNN 및랜덤 포레스트가 뒤를 이었습니다. 또한 LSTM 모델은 30주간의 역사적 데이터를 활용하여 12주 동안의 가뭄 점수를 예측했으며, 평균 절대 오차(MAE)는 0.33으로 나타났습니다.

- **Performance Highlights**: 모델 성능 평가 결과, LSTM 모델은 일관된 가뭄 패턴이 나타나는 카운티에서 가장 우수한 성능을 보였고, 심각한 가뭄이 잦은 곳에서 더욱 효과적이었습니다. 따라서 최소 24주의 데이터가 최적의 성능을 발휘하며, 특히 8주 미만의 짧은 시간 동안의 예측에서 최고의 결과를 나타냈습니다. 전반적으로 LSTM은 매크로 F1 점수 0.9를 기록하여 심각한 가뭄 조건의 이진 분류에서 높은 정확도를 달성하였습니다.



### Continuous Cardiac Arrest Prediction in ICU using PPG Foundation Mod (https://arxiv.org/abs/2502.08612)
- **What's New**: 새로운 연구 영역인 비침습적인 환자 모니터링이 심각한 심장 질환 사건 감지 및 예측을 목표로 하고 있습니다. 본 연구에서는 단일 채널의 손가락에서 측정된 photoplethysmography (PPG) 신호만을 활용하여 ICU 환자의 병원 내 심장 마비(in-hospital cardiac arrest, IHCA)를 예측하는 모델을 제안합니다. 특히, PPG-GPT와 같은 사전 학습된 PPG 모델을 통해 강력한 표현을 활용하는 두 단계 모델인 Feature Extractor-Aggregator Network (FEAN)를 개발하였습니다.

- **Technical Details**: FEAN 모델은 두 개의 단계로 구성됩니다. 첫 번째 단계에서는 사전 학습된 PPG-GPT 생성 모델을 사용하여 특성 추출을 수행하며, 두 번째 단계에서는 순차적인 출력 결과를 하나의 벡터로 결합하는 feature aggregator를 활용합니다. 이 모델은 마지막 레이어의 숨겨진 레이어 출력에서 마지막 타임스탬프를 사용하여 특성을 추출하며, 다양한 모델 아키텍처들을 이용하여 시퀀스 특성을 집계합니다.

- **Performance Highlights**: 본 연구에서 제안한 모델은 24시간 예측 윈도우 내에서 평균 0.79 AUROC를 기록하였으며, 심장 마비 발생 1시간 전에는 0.82로 성능이 정점에 달했습니다. 또한, 모델 아키텍처 튜닝과 PaCMAP 기법을 통해 환자의 건강 궤적을 시각화하는 전반적인 분석을 제공하였습니다. 이는 ICU 환자의 IHCA 예측에 있어 최초로 단일 모달의 PPG 파형 데이터만을 이용한 중요한 성과로 평가됩니다.



### Robustly Learning Monotone Generalized Linear Models via Data Augmentation (https://arxiv.org/abs/2502.08611)
- **What's New**: 이번 연구에서는 Gaussian 분포 하에서의 무관심 모델(agnostic model)에서 Generalized Linear Model (GLM)을 학습하는 작업을 다룹니다. 이전의 상수비율 GLM 학습기들은 훨씬 더 작은 활성화 클래스에서만 성공했지만, 본 연구는 모든 단조로운 Lipschitz 활성화에 대해 상수비율 근사(Constant-factor approximation)를 달성하는 최초의 다항 시간 알고리즘을 제공합니다. 이는 잘 알려진 주제인 GLMtron 알고리즘의 견고한 변형을 개발한 것을 통해 해결됩니다.

- **Technical Details**: 본 연구는 모든 단조로운 활성화가 유한한 $(2+	heta)$-모멘트를 가지는 조건 하에서 적용되는 견고한 학습기를 개발합니다. 이 조건은 본질적으로 필요합니다. 특히, 감소하는 Gaussian 노이즈 주입을 통한 새로운 데이터 증강(data augmentation) 기법을 발휘하여 성과를 도출하였습니다.

- **Performance Highlights**: 연구에서는 제안된 알고리즘이 다양한 조건에서 어떻게 효과적으로 작동하는지를 보여주며, 이는 다른 설정에서도 유용할 수 있는 여러 구조적 결과로 이어집니다. 이러한 결과는 GLM 학습 분야에서 중요한 기여를 하며, 구체적인 활성화 형태에 대해 효율적인 학습을 가능하게 합니다.



### Distillation Scaling Laws (https://arxiv.org/abs/2502.08606)
Comments:
          67 pages, 54 figures, 13 tables

- **What's New**: 이번 연구는 모델 디스틸레이션(distillation) 성능을 계산 예산(compute budget)과 학생(student) 및 교사(teacher) 모델 간의 할당에 따라 추정하는 새로운 스케일링 법칙(scaling law)을 제시합니다. 이를 통해 대규모 디스틸레이션 사용에 따른 위험을 줄일 수 있습니다.

- **Technical Details**: 연구에서는 두 가지 경우에 대한 최적의 컴퓨트(compute) 할당 방식을 제안합니다: 1) 교사가 존재할 때, 2) 교사가 훈련이 필요할 때. 여러 학생에 대해 디스틸레이션을 진행할 경우, 또는 이미 존재하는 교사가 있는 경우, 디스틸레이션이 감독된 사전 훈련(supervised pretraining)보다 성능이 우수하다는 것을 발견했습니다.

- **Performance Highlights**: 특히, 한 명의 학생에 대한 디스틸레이션 과정에서 교사 모델도 훈련이 필요할 경우, 감독된 학습(supervised learning)이 더욱 효과적이라는 결과를 도출했습니다. 이 외에도, 대규모 디스틸레이션 연구를 통해 디스틸레이션 이해도를 높이고 실험 디자인을 개선할 수 있는 통찰력을 제공하였습니다.



### CurvGAD: Leveraging Curvature for Enhanced Graph Anomaly Detection (https://arxiv.org/abs/2502.08605)
- **What's New**: CurvGAD는 복잡한 네트워크에서 기하학적 특성을 활용해 그래프의 이상을 탐지하는 새로운 접근 방식을 제안합니다. 기존 그래프 이상 탐지(GAD) 방식은 구조적 및 속성 기반 이상에만 집중하는 반면, 본 연구는 곡률(curvature)을 통한 기하학적 이상을 조명합니다. CurvGAD는 곡률에 기반한 새로운 이상 분류 방식을 도입하여 기존의 접근 방식에서 놓쳤던 중요 정보를 포착합니다.

- **Technical Details**: CurvGAD는 두 개의 병렬 파이프라인을 구축하여 이상 해석력을 향상시킵니다. 첫 번째 파이프라인은 곡률에 대한 기하학적 재구성을 통해 엣지 곡률을 복원하고, 두 번째 파이프라인은 구조 및 속성을 곡률에 무관하게 재구성하여 비기하적 이상을 분리합니다. 이와 같은 접근은 다양한 복잡한 토폴로지를 상세히 표현할 수 있게 합니다.

- **Performance Highlights**: 본 논문에서는 10개의 실제 데이터셋을 사용하여 CurvGAD의 효능을 평가하였고, 기존 최첨단 GAD 방법보다 최대 6.5% 향상된 성능이 입증되었습니다. CurvGAD는 동질적 및 이질적 네트워크 환경에서도 기하학적, 구조적 및 속성 기반 이상을 효과적으로 탐지할 수 있는 능력을 갖추고 있습니다.



### Two-stage hybrid models for enhancing forecasting accuracy on heterogeneous time series (https://arxiv.org/abs/2502.08600)
Comments:
          14 pages, 2 figures

- **What's New**: 이 논문은 데이터의 이질성을 모델 기반으로 식별하고, 이를 통해 글로벌 모델이 이질 데이터를 완전히 활용할 수 있도록 하는 새로운 두 단계 모델링 방법을 제안합니다. 기존 연구에서 이질 데이터를 다루는 데 사용된 클러스터링이나 하이브리드 모델링은 종종 성능 한계가 있었으나, 본 논문에서는 이를 해결하기 위한 접근 방식을 제시합니다.

- **Technical Details**: 본 논문에서는 시계열 데이터의 이질성을 정의할 때, 글로벌 모델이 사용된다는 점에서 접근합니다. 첫 번째 단계에서 글로벌 모델을 통해 이질적인 부분을 식별하고, 두 번째 단계에서는 이질적인 패턴을 모델링합니다. 이 과정에서 K-means 클러스터링과 같은 거리 기반 클러스터링 기법이 활용되며, 복잡한 데이터 구조에서도 효과적인 경향을 보입니다.

- **Performance Highlights**: 제안된 두 단계 하이브리드 모델은 네 개의 공개 데이터셋 실험에서 기존의 다섯 개 모델을 유의미하게 초월하는 성능을 보여주었습니다. 이는 이질적 데이터셋에서 글로벌 모델의 잠재력을 완전히 발휘할 수 있는 방법을 제시하며, 향후 다양한 분야에서 시계열 예측의 정확도를 높이는 데 기여할 것입니다.



### Enhancing Diffusion Models Efficiency by Disentangling Total-Variance and Signal-to-Noise Ratio (https://arxiv.org/abs/2502.08598)
- **What's New**: 이번 연구에서는 총 분산(total variance)과 신호 대 잡음 비율(signal-to-noise ratio)을 독립적으로 제어할 수 있는 새로운 프레임워크인 TV/SNR을 제안합니다. 기존의 방법들은 분산을 수동적으로 조정하기 때문에, 이러한 접근은 직접적인 제어가 부족했습니다. 제안된 프레임워크는 각 스텝에서 잡음이 도입되는 방식이 샘플의 질에 큰 영향을 미친다는 점을 강조합니다. 이를 통해 불안정했던 분산을 상수로 유지하는 동시에 SNR 스케줄을 보존함으로써 성능이 개선됨을 보여줍니다.

- **Technical Details**: TV/SNR 프레임워크는 총 분산이 시간에 따라 증가하는 기존 스케줄과는 달리, 일정한 TV 스케줄을 유지하며 성능을 극대화합니다. 이 방법을 적용하여 분자 구조 생성 시 샘플의 질을 높이고, 이미지 생성을 위해 균일한 확산 시간 그리드(uniform diffusion time grid)를 사용하는 방법이 도입되었습니다. 특히 최적 수송 흐름(optimal transport flow matching)의 SNR 스케줄을 일반화함으로써, 기존 모델보다 더 효과적인 샘플 생성을 달성했습니다. 이를 통해 계산 화학 분야에서의 적용 가능성을 제시하고 있으며, 다양한 생성 모델의 성능을 개선합니다.

- **Performance Highlights**: 본 연구에서 제안한 TV/SNR 프레임워크는 기존의 방식에 비해 분자 및 이미지 생성에서 인상적인 성능을 보여주었습니다. 특히 표준 FID(Fréchet Inception Distance) 점수에서 우수한 결과를 기록함으로써 실제 응용 가능성을 높였습니다. 적은 단계로도 안정적인 분자의 생성을 가능하게 하여 계산 화학의 혁신을 촉진하고 있습니다. 또한 이 접근 방식은 기존의 EDM 샘플러와 비교할 때 경쟁력 있는 결과를 제공함으로써, 샘플링 효율성과 품질을 모두 향상시킵니다.



### Toward Universal Laws of Outlier Propagation (https://arxiv.org/abs/2502.08593)
- **What's New**: 이 논문은 알고리즘 정보 이론(Algorithmic Information Theory, AIT)이 이른바 무작위성 결핍(randomness deficiency) 측면에서 이상치(outlier)를 정량화할 수 있는 원리를 제공한다고 주장합니다. 인과적 베이지안 네트워크(causal Bayesian network)에서 발생하는 확률 분포(probability distribution)에 대해, 결합 상태의 무작위성 결핍이 각 인과 메커니즘의 무작위성 결핍으로 분해된다는 것을 보여줍니다. 이는 인과 메커니즘 원칙(Independence of Mechanisms Principle)에 따른 것으로, 비정상적인 결합 관측 결과를 그 근본 원인으로 정량적으로 할당할 수 있는 방법을 제시합니다.

- **Technical Details**: 이 논문에서는 이상치 분석을 위한 수학적 모델로 인과적 베이지안 네트워크를 사용합니다. 이를 통해 이상 관측의 형태를 이해하기 위해 시스템의 인과 메커니즘을 표현하는 방법을 설명하며, 각 메커니즘은 서로 독립적으로 변화할 수 있다고 가정합니다. Budhathoki et al. (2022)의 연구를 바탕으로 정보 이론적(outlier score) 이상치 점수를 도입하고, 각 변수를 나타내는 조건부 분포를 통해 이상치를 수량적으로 귀속시키는 방법을 제시합니다.

- **Performance Highlights**: 논문은 무작위성 보존에 대한 Levin의 법칙을 확장하는 내용을 포함하고 있으며, 이를 통해 인과 메커니즘이 독립적일 때 약한 이상치(weak outlier)가 강한 이상치(strong one)를 유발할 수 없음을 입증합니다. 이러한 정보 이론적 법칙은 기존 점수에 대해 정의된 이상치의 행동을 더 잘 이해할 수 있도록 도와줍니다. 이상치 탐지 및 원인 분석 분야에서 자료 기반의 정확한 평가를 통해 비즈니스와 기술, 의학 분야에서의 활용 가능성을 제시하고 있습니다.



### Commercial LLM Agents Are Already Vulnerable to Simple Yet Dangerous Attacks (https://arxiv.org/abs/2502.08586)
- **What's New**: 최근 ML 보안 문헌에서는 LLM(대규모 언어 모델)을 대상으로 한 공격이 집중적으로 다루어지고 있지만, 이러한 공격은 종종 모델의 개인 정보를 추출하거나 해로운 출력을 생성하게 하는 방식으로 진행됩니다. 이 논문에서는 LLM을 포함한 대리 시스템의 보안 취약점을 분석하고, 이러한 시스템이 직면하는 특정한 위험들을 조명합니다. 특히, LLM 기반 대리 에이전트는 다양한 환경과 연결되어 있어 훨씬 더 공격받기 쉬운 구조를 가지고 있습니다.

- **Technical Details**: 저자들은 LLM 에이전트에 대한 공격의 분류 체계를 제시하며, 이를 통해 공격자, 목표, 침투 지점, 공격 관찰 가능성 및 공격 전략에 따라 다섯 가지 카테고리로 나누었습니다. 논문에서는 Reddit와 같은 신뢰할 수 있는 웹사이트에 악성 게시글을 만들어 LLM 에이전트를 조작하는 매우 단순하고 효과적인 공격 파이프라인을 설계하여 이를 시연합니다. 더욱이, 에이전트의 동작 중 발생할 수 있는 다양한 취약점에 대해서도 설명하고 있습니다.

- **Performance Highlights**: 실제 사례를 통해 오픈 소스 및 상용 에이전트에 대한 공격을 시연하면서 즉각적인 결과를 도출할 수 있음을 보여줍니다. 공격 성공률이 높은 여러 유형의 공격(개인 정보 누출, 악성파일 다운로드, 피싱 이메일 발송 등)에 대해 설명하며, 심지어 기계 학습에 대한 전문 지식 없이도 공격이 가능하다는 점을 강조합니다. 이러한 점은 LLM 에이전트가 사용자의 안전을 위협할 수 있는 심각한 위험 요소로 작용하고 있음을 알립니다.



### Scalable Bilevel Loss Balancing for Multi-Task Learning (https://arxiv.org/abs/2502.08585)
- **What's New**: 본 논문에서는 Multi-Task Learning (MTL) 분야에서의 손실 균형을 위한 새로운 기법인 BiLB4MTL을 제안합니다. 이 방법은 bilevel optimization (2단 최적화) 관점에서 접근하며, $	ext{O}(1)$의 시간 및 메모리 복잡도로 효율적인 학습을 가능하게 합니다. 또한, 이 방법은 다수의 손실 함수 간의 균형을 유지하며, 기존의 방법들보다 더 나은 성능과 효율성을 보여줍니다.

- **Technical Details**: BiLB4MTL은 세 가지 주요 구성 요소로 이루어져 있습니다: 1) 초기 손실 정규화, 2) 2단 손실 균형 수식화, 3) 단일 루프 구조를 갖는 확장 가능한 1차 알고리즘입니다. 이 방법은 다양한 다중 작업 데이터셋에서 실험을 통해 그 효과성을 검증하였으며, 특히 $K$개의 손실 함수에 대한 $	ext{ε}$-정확한 Pareto stationary point로 수렴하는 것을 이론적으로 보장합니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, BiLB4MTL은 QM9, CelebA 및 Cityscapes와 같은 여러 감독 다중 작업 데이터셋에서 최첨단 성능을 기록하였습니다. 이 연구는 BiLB4MTL이 다른 scalarization 기반 및 gradient manipulation 방법보다 효율적이고 확장 가능한 솔루션임을 강조하며, 더 나은 수렴성과 성능을 제공합니다.



### A method for classification of data with uncertainty using hypothesis testing (https://arxiv.org/abs/2502.08582)
- **What's New**: 이번 논문에서는 기존의 이진 분류기에서 발생하는 오판 단지와 불확실성을 해결하기 위해 두 가지 유형의 가설 검정을 통한 새로운 의사결정 접근법을 제안합니다. 고위험 분야에서 사용되는 전통적인 방법들이 갖는 한계점을 극복하기 위해, 훈련 데이터에서 추출한 피쳐의 경험적 분포를 활용하여 불확실성을 정량화하고 분류 임계값을 결정합니다. 이러한 방법은 두 개의 클래스 분포의 겹치는 영역에 속하는 데이터 및 훈련 데이터 분포 밖의 데이터를 탐지하는 데 유용합니다.

- **Technical Details**: 제안된 방법은 훈련 데이터에서 얻은 피쳐를 사용하여 이진 분류 문제를 해결합니다. 이 과정에서 다양한 가설 검정을 통해 각 클래스의 분포를 경험적 분포로 근사하고, 클래스 간 불확실성을 정량화합니다. 두 가지 가설 검정의 기각 영역과 수용 영역을 정의하고, 적절한 α-quantile 및 (1-α)-quantile을 기반으로 한 분류 임계값을 설정합니다. 이 방법은 재샘플링이나 모델 개선 없이 불확실성을 정량화할 수 있는 장점을 가지고 있습니다.

- **Performance Highlights**: 이 새로운 접근법은 스파이럴 패턴 데이터와 가슴 X선 이미지의 이진 분류에 적용되어 불확실성을 잘 포착하는 능력을 보여줍니다. 실험 결과, 전통적인 분류 방법에 비해 더 높은 불확실성 공정을 수행하면서도 계산 비용을 효과적으로 줄이는 데 성공하였습니다. 이는 고위험 분야에서의 의사결정 과정에서 더 나은 신뢰성을 제공할 것으로 예상됩니다.



### FBFL: A Field-Based Coordination Approach for Data Heterogeneity in Federated Learning (https://arxiv.org/abs/2502.08577)
- **What's New**: 이번 논문에서는 Field-Based Federated Learning (FBFL)이라는 새로운 접근 방식을 제안하여 데이터 분포 문제와 중앙집중식 아키텍처의 한계를 극복하고자 합니다. FBFL은 매크로 프로그래밍(macroprogramming)과 필드 조정(field coordination)을 활용하여 비독립적이고 비동일 분포된(non-IID) 데이터 문제를 해결하며, 공간적으로 분산된 디바이스 간에 리더를 선정하여 개인화된 학습을 지원합니다. 또한, 이 논문은 FBFL 방식이 기존의 FedAvg 알고리즘과 유사한 성능을 내며, 매우 도전적인 비 IID 시나리오에서도 더 나은 결과를 보여준다는 것을 강조합니다.

- **Technical Details**: FBFL은 'fields of Machine Learning (sub)models'라는 개념을 도입하여, 정보 전파 및 집합(computing)에서 발전된 알고리즘을 기반으로 하고 있습니다. 또한, FBFL은 디바이스를 공간 근접성을 기준으로 그룹화하여 효율적인 자기 조직화(self-organizing) 계층 구조를 구축함으로써 비 IID 데이터 환경에서도 성능을 향상시킵니다. 이 접근은 중앙 집중식 권한에 의존하지 않고도 동적으로 효율적인 모델 집계를 가능하게 합니다. 특히, 자가 조직적 하이브리드 아키텍처는 피어-투-피어 상호작용을 통해 리더를 분산 방식으로 선출함으로써 리더가 각 지역의 모델을 집약하도록 합니다.

- **Performance Highlights**: FBFL은 MNIST, FashionMNIST 및 Extended MNIST 데이터 세트를 사용하여 철저한 평가를 거쳤습니다. IID 데이터 조건에서 FBFL은 널리 사용되는 FedAvg와 유사한 성능을 보였습니다. 더욱이, 도전적인 비 IID 시나리오에서도 FBFL은 FedAvg를 초과하는 성능을 보여주었으며, 비 IID 데이터 분포 문제를 해결하기 위해 고안된 최신 방법인 FedProx 및 Scaffold를 능가하는 결과를 기록했습니다. 마지막으로, FBFL의 자기 조직적 계층 구조는 서버 실패에 대한 강인성을 선보였습니다.



### COAST: Intelligent Time-Adaptive Neural Operators (https://arxiv.org/abs/2502.08574)
- **What's New**: 이번 연구에서는 Causal Operator with Adaptive Solver Transformer (COAST)라는 새로운 신경 연산자 학습 방법을 제안합니다. COAST는 causal language model (CLM) 프레임워크를 활용하여 시스템의 진화와 최적의 시간 단계를 예측하는 데 중점을 두고 있습니다. 이를 통해 계산 효율성과 정확성 간의 균형을 맞추며, 동적 시스템 전반에 걸쳐 고유한 특성과 일치하는 가변 시간 단계를 생성하는 능력을 보입니다.

- **Technical Details**: COAST는 시간 의존적인 물리 시스템을 해결하기 위해 설계된 시간 적응형 신경 연산자 아키텍처입니다. 이 모델은 입력 공간의 시계열 데이터를 위한 학습 가능한 시공간 인코딩을 활용하여 연속 시간 지점에서 동작하며, 예측된 다음 단계를 공간-시간적으로 결합된 임베딩으로 출력합니다. 아키텍처는 시공간 인코더, causal language model, 해석-수정 메커니즘, 보간 디코더의 네 가지 주요 요소로 구성됩니다.

- **Performance Highlights**: COAST는 다양한 도전 과제를 기반으로 한 벤치마크에서 기존의 최첨단 방법을 지속적으로 초월하는 성능을 보여주었습니다. 높은 복잡성을 가진 영역에서는 더 작은 단계로, 단순한 영역에서는 더 큰 단계를 적용하여 예측의 정확성을 높이는데 기여합니다. 이 연구는 CLM을 기반으로 한 지능형 적응형 솔버의 잠재력을 강조하며, 동적 시스템의 확장 가능성 있는 연산자 학습에 대한 새로운 길을 제시합니다.



### Beyond Predictions: A Participatory Framework for Multi-Stakeholder Decision-Making (https://arxiv.org/abs/2502.08542)
- **What's New**: 이 논문은 전통적인 결정 지원 시스템의 한계를 극복하기 위해 다중 이해관계자 최적화 문제로서 의사결정 프로세스를 재정의하는 새로운 참여 프레임워크를 제안합니다. 이 프레임워크는 다양한 이해관계자의 선호를 문맥에 따라 정의된 보상 함수로 포착하여 다수의 목표를 균형 있게 반영합니다. 또한, 사용자 정의 선호를 최대한 활용하여 최적의 의사 결정자를 식별하고 추천을 생성하는 데 사용할 수 있습니다.

- **Technical Details**: 제안된 프레임워크는 여러 주요 특징을 통해 고위험 분야에서 효과적입니다. 첫째, 다양한 이해관계자의 이익을 명시적으로 모델링하여 시스템이 다수의 관점을 반영하도록 합니다. 둘째, 타협 중심의 행동 선택 메커니즘을 포함하여 다양한 목표 간의 거래를 조화시키는 방법을 제공합니다. 셋째, 다양한 예측 모델 및 응용 맥락에 유연하게 적응할 수 있는 모델 비의존적인 설계를 채택했습니다.

- **Performance Highlights**: 이 프레임워크는 두 가지 실제 사례를 통해 검증되었으며, 다수의 측정을 기준으로 평가했을 때 단순 예측 방식보다 향상된 의사결정 결과를 보여주었습니다. 모듈형이고 투명한 설계로 인해 다양한 예측 모델 및 보상 구조와 잘 통합되며, 사회적 불편을 해소하는데 기여할 수 있는 가능성을 보였습니다. 이를 통해 제공된 추천은 다양한 이해관계자의 복잡한 요구를 효과적으로 조화시킵니다.



### Matrix Completion with Graph Information: A Provable Nonconvex Optimization Approach (https://arxiv.org/abs/2502.08536)
Comments:
          41 pages, 6 figures

- **What's New**: 이번 논문에서는 변수 간의 상호관계를 나타내는 그래프를 활용한 행렬 완성(matrix completion) 문제를 다룹니다. 기존의 그래프 라플라시안 정규화를 기반으로 한 접근법은 이웃 변수 간의 유사성에만 집중하고 장거리 상관관계를 간과하는 문제점이 있습니다. 이를 해결하기 위해 GSGD라는 새로운 그래프 정규화 행렬 완성 알고리즘을 제안하며, 이는 전처리된 투영 경량 경사 하강법(preconditioned projected gradient descent) 접근법을 기반으로 합니다.

- **Technical Details**: GSGD는 그래프 이면의 고차원 상관관계 정보를 효과적으로 포착하며, 잘못된 엣지(false edges)에 대한 강한 내구성 및 안정성을 보여줍니다. 이 알고리즘은 비볼록 최적화(nonconvex optimization) 관점에서 회복 정확도와 효율성을 위한 이론적 보증을 제공합니다. GSGD는 최적의 표본 복잡도(near-optimal sample complexity)와 함께 전역 최적(global optimum)으로의 선형 수렴(linear convergence)을 성취함을 이론적으로 증명합니다.

- **Performance Highlights**: 수치 실험을 통해 GSGD는 합성(synthetic) 데이터와 실제(real-world) 데이터 모두에서 여러 인기 있는 대안들에 비해 우수한 회복 정확도와 확장성(scalability)을 달성하는 것으로 확인되었습니다. 이러한 성능이 GSGD의 장점으로 입증되어, 다양한 응용 분야에서의 활용 가능성을 높임을 보여줍니다.



### On Different Notions of Redundancy in Conditional-Independence-Based Discovery of Graphical Models (https://arxiv.org/abs/2502.08531)
- **What's New**: 이번 논문에서는 조건 독립성(conditional independence) 기반의 그래프 모델 학습 과정에서 생략된 통계 테스트의 잠재적 가치를 강조합니다. 저자들은 이들 중 일부가 그래프 모델의 오류를 발견하거나 수정하는 데 도움이 될 수 있다고 주장합니다. 그러나 이러한 중복 테스트가 항상 유용한 것은 아니므로 주의 깊게 적용해야 한다고 경고합니다.

- **Technical Details**: 그래프 모델은 복잡한 시스템을 이해하는 데 필수적인 도구로 자리잡고 있으며, 조건 독립성 테스트의 통계적 어려움이 이러한 모델을 개선하는 데 도전 과제가 됩니다. 연구자들은 CI 테스트의 무효 오류를 제어하는 것은 어렵고, 결국 이로 인해 얻어진 그래프 모델이 크게 왜곡될 수 있음을 강조합니다. 특히, CI 테스트의 결과가 확률 분포의 법칙이 아닌 그래프의 제약 조건에서 비롯되는 경우에만 추가 정보를 제공한다고 주장합니다.

- **Performance Highlights**: 그래프 모델의 조건 독립성 기반 탐색은 노드 수가 많아질수록 지수적으로 많은 CI 테스트를 요구하는 문제를 안고 있습니다. 연구팀은 이러한 CI 테스트들을 잘 활용하여 모델의 정확성을 향상시킬 수 있는 방법을 제안합니다. 이 과정에서 중복 테스트의 잠재력을 활용하여 성능을 개선할 수 있음을 보여주며, 이는 이전 연구 결과와 일맥상통합니다.



### LLM Pretraining with Continuous Concepts (https://arxiv.org/abs/2502.08524)
- **What's New**: 최근 대규모 언어 모델의 발전 덕분에 자연어 처리 분야가 혁신을 이루고 있습니다. 특히, 이 연구에서는 전통적인 다음 토큰 예측(next token prediction) 방식에 새로운 접근 방식인 Continuous Concept Mixing (CoCoMix)을 소개합니다. CoCoMix는 사전 훈련된 희소 오토인코더(sparse autoencoder)로부터 배운 연속 개념을 결합하여 모델의 숨은 상태에 혼합(interleave)하는 방식으로 작동합니다.

- **Technical Details**: CoCoMix는 토큰 레벨의 혼잡함(ambiguity)을 줄이고 모델의 개념적 이해를 향상시키기 위해 설계되었습니다. 이를 위해, 사전 훈련된 SAE(Sparse Autoencoder)로부터 얻은 개념을 사용하여, 모델이 이러한 연속 개념을 예측하도록 학습합니다. 예측된 개념은 다시 혼합되어 다음 토큰 예측에 기여하며, 이 과정은 모델의 해석 가능성을 높이고 제어 가능성을 개선합니다.

- **Performance Highlights**: 여러 벤치마크에서의 실험 결과에 따르면, CoCoMix는 기존의 다음 토큰 예측, 지식 증류(knowledge distillation) 및 정지 토큰(pause token) 삽입 방식보다 모든 면에서 성능이 우수함을 입증했습니다. 예를 들어, 1.38B 크기의 모델에 적용했을 때, CoCoMix는 21.5% 적은 훈련 토큰으로도 비슷한 성능을 달성했습니다. 또한, CoCoMix는 개념을 압축하여 삽입함으로써 모델의 생성 과정을 제어할 수 있게 해줍니다.



### FedMHO: Heterogeneous One-Shot Federated Learning Towards Resource-Constrained Edge Devices (https://arxiv.org/abs/2502.08518)
- **What's New**: 이번 연구에서는 자원 제약이 있는 클라이언트와 자원 충분한 클라이언트를 모두 수용할 수 있는 새로운 연합 학습(FL) 프레임워크인 FedMHO를 제안합니다. FedMHO는 자원 충분한 클라이언트에서는 깊은 분류 모델(deep classification models)을 사용하고, 자원 제약이 있는 클라이언트에서는 경량 생성 모델(lightweight generative models)을 활용합니다. 또한 지식 융합 지식(model heterogeneity) 문제를 해결하기 위해 FedMHO-MD 및 FedMHO-SD라는 두 가지 솔루션을 도입합니다.

- **Technical Details**: FedMHO는 데이터 생성(data generation)과 지식 융합(knowledge fusion)의 두 단계를 포함하여 글로벌 모델을 훈련합니다. 데이터 생성 단계에서는 클라이언트에서 수신된 디코더가 로컬 레이블 분포에 기반하여 합성 샘플(synthetic samples)을 생성합니다. 또한 이러한 합성 샘플의 품질을 개선하기 위해 비지도 데이터 최적화 솔루션이 적용됩니다.

- **Performance Highlights**: 실험 결과 FedMHO, FedMHO-MD 및 FedMHO-SD는 각각 5.17%, 8.35%, 8.25%의 평균 정확도(improved accuracy)를 향상시켜 기존의 최적 기준을 초과하는 성능을 입증하였습니다. 연구 결과는 자원 부족 클라이언트와 자원 충분 클라이언트가 혼합된 환경에서도 효과적인 FL이 가능함을 보여줍니다.



### The Paradox of Stochasticity: Limited Creativity and Computational Decoupling in Temperature-Varied LLM Outputs of Structured Fictional Data (https://arxiv.org/abs/2502.08515)
Comments:
          8 pages, 6 figures

- **What's New**: 이번 연구에서는 온도 설정과 모델 아키텍처가 구조화된 허구 데이터(이름, 생년월일) 생성에 미치는 영향을 조사했습니다. 세 가지 대형 언어 모델(LLMs)인 llama3.1:8b, deepseek-r1:8b, mistral:latest를 대상으로 하여 다양한 온도 값을 시험했습니다. 주요 발견은 모델 아키텍처가 계산 효율성에 크게 영향을 미친다는 점입니다.

- **Technical Details**: 온도 값은 0.0에서 1.0까지 0.1 간격으로 체계적으로 테스트하였으며, 총 330회의 실험을 통해 889개의 구조화된 엔티티를 생성하고 문법적 일관성을 검증했습니다. 연구에서는 mistral:latest와 llama3.1:8b가 deepseek-r1:8b보다 데이터 처리 속도가 8배 빠르다는 것을 발견했습니다. 또한 예상과 달리 온도는 처리 시간과 상관관계가 없었습니다.

- **Performance Highlights**: 모델들은 온도에 상관없이 'John Doe'와 'Jane Smith'와 같은 일반적인 이름 아키타입으로 기본값으로 돌아가는 경향이 있었으며, 간헐적으로 나타나는 희귀한 이름은 중간 값(0.3-0.7)에서 군집화되었습니다. 이러한 결과는 구조화된 생성 작업에서 성능을 지배하는 것이 온도 조정이 아니라 아키텍처 최적화임을 보여줍니다.



### Bridging Domain Adaptation and Graph Neural Networks: A Tensor-Based Framework for Effective Label Propagation (https://arxiv.org/abs/2502.08505)
- **What's New**: 이번 논문에서는 라벨 전파(label propagation) 기술을 활용하여 그래프 데이터에 대한 도메인 적응(domain adaptation) 문제를 해결하고자 하는 LP-TGNN 프레임워크를 제안합니다. 기존의 Graph Neural Networks (GNNs)는 단일 도메인에서 감독 학습을 통해 극복해야 할 한계가 있지만, LP-TGNN은 텐서 아키텍처(tensor architecture)를 통해 그래프의 구조적 정보를 총체적으로 추출하고 도메인 간 불일치를 줄입니다.

- **Technical Details**: LP-TGNN은 두 가지 주 가지로 구성된 그래프 인코더(graph encoder)를 사용하여 그래프의 전체적인 표현을 추출하고, 기하학적 거리 아래에서 그래프 유사성을 유지합니다. 이 모델은 메시지 전달(message-passing) 구조를 따르며, Persistent Homology(PH)를 활용하여 그래프의 위상(topological) 정보를 추출하는 기능을 지니고 있습니다. 이를 통해 모델은 기존 GNN과 도메인 적응 기술과의 통합이 용이합니다.

- **Performance Highlights**: 다양한 실제 벤치마크에서 LP-TGNN은 기존 모델에 비해 상당한 성능 향상을 보였습니다. 실험 결과를 통해, LP-TGNN은 그래프 간 정보의 거리 측정을 보다 효과적으로 수행하며, 라벨 정보를 이웃 그래프로 안정적으로 전파하여 도메인 간의 일관성을 유지합니다. 또한, 제안된 프레임워크의 각 구성 요소를 검증하고 분석한 결과, 도메인 적응 그래프 분류에서의 가능성을 한층 끌어올림을 보여주었습니다.



### One-Shot Federated Learning with Classifier-Free Diffusion Models (https://arxiv.org/abs/2502.08488)
- **What's New**: 이번 연구는 OSCAR(One-Shot Federated Learning with Classifier-Free Diffusion Models)이라는 새로운 접근 방식을 소개합니다. OSCAR는 클라이언트에서 보조 모델을 사용하지 않고, 한 번의 통신 라운드로 글로벌 모델을 학습할 수 있도록 설계되었습니다. 이를 통해 서버 측에서 클라이언트의 카테고리별 데이터 표현을 사용하여 데이터 생성을 원활하게 진행할 수 있습니다.

- **Technical Details**: OSCAR는 분류기(분류 모델) 없이 작동하는 디퓨전 모델을 사용하여 각 클라이언트의 카테고리별 인코딩을 활용하고, 이를 통해 서버에서 글로벌 모델을 학습하게 됩니다. 이 방법은 현재의 DM(디퓨전 모델) 지원 OSFL 방식보다 클라이언트의 업로드 크기를 99%까지 줄이는 한편, 통신 오버헤드도 감소시킵니다. 특히 OSCAR는 각 통신 라운드에서의 계산 과부하를 줄이며, 비독립적이고 동일하게 분포되지 않은(non-IID) 데이터 문제를 효과적으로 해결합니다.

- **Performance Highlights**: OSCAR는 Four benchmarking datasets에서 현재의 최첨단 기술(SOTA)보다 우수한 성능을 보여주었습니다. 통신 부하를 최소 99% 줄임으로써, 효율성 및 확장성을 극대화합니다. 또한, 다양한 클라이언트 데이터셋에서 향상된 성능을 발휘하여, 분산 학습의 새로운 가능성을 제시합니다.



### Training-Free Restoration of Pruned Neural Networks (https://arxiv.org/abs/2502.08474)
Comments:
          Under Review in TNNLS since May 2022

- **What's New**: 이번 논문에서는 네트워크 프루닝(Pruning)에 대한 새로운 접근법인 LBYL(Leave Before You Leave)을 제안합니다. 기존의 프루닝 후 재훈련 과정이 계산적으로 비용이 많이 드는 문제를 해결하고자, 데이터 없이, 또한 재훈련 없이 프루닝된 네트워크를 복원하는 방법을 제시합니다. LBYL은 각 프루닝된 뉴런이 최대한 많은 보존된 뉴런에게 정보를 남기도록 하여 보다 강력한 근사치를 제공하는 방식입니다.

- **Technical Details**: LBYL 방법은 기존 네트워크와 그 근사치 간의 재구성 오차(reconstruction error)를 수학적으로 분석하여, 유도된 손실 함수(loss function)에 대한 폐쇄형 솔루션을 도출합니다. 이 방법은 뉴런 간의 유사성이 낮은 문제를 해결하며, 뉴런들이 함께 협력하여 원래 뉴런의 출력을 더 잘 근사할 수 있도록 합니다. 이론적 분석을 통해 기존의 접근 방식보다 훨씬 더 유연하고 강력한 조건을 갖추고 있음을 보여줍니다.

- **Performance Highlights**: LBYL 방법은 광범위한 실험을 통해 복원된 네트워크의 정확도를 기존의 유사성을 활용한 접근법들보다 높게 유지함을 입증했습니다. 실험 결과, LBYL은 기존 네트워크의 원래 구조를 더 잘 근사하며, 그에 따라 향상된 정확도를 달성할 수 있음을 나타냅니다. 해당 연구의 초기 버전은 NeurIPS 2021과 ICML 2022에 제출되었습니다.



### Learning Theory for Kernel Bilevel Optimization (https://arxiv.org/abs/2502.08457)
- **What's New**: 이 논문에서는 커널 바이레이블 최적화(kernel bilevel optimization) 문제의 일반화 특성을 조사합니다. 특히, 내부 목표가 재생 커널 힐베르트 공간(Reproducing Kernel Hilbert Space, RKHS)에서 최적화되는 설정을 다룹니다. 이 설정은 풍부한 함수 근사를 가능하게 하고 철저한 이론 분석의 기초를 제공합니다.

- **Technical Details**: 제안된 접근법은 함수적 관점(functional perspective)을 채택하고, 경험적 과정 이론(empirical process theory) 및 퇴화된 U-프로세스의 최대 부등식(maximal inequalities) 도구를 활용하여 균일한 오류 경계를 도출합니다. 이 과정을 통해 유한 샘플 근사(finite-sample approximation) 하에서 바이레이블 문제의 새로운 일반화 오류 경계를 확립합니다.

- **Performance Highlights**: 제안된 일반화 오류 추정치는 바이레이블 문제의 경험적 이산화(empirical discretization)에 적용된 그래디언트 기반 방법(statistical accuracy)을 특성화하는 데 도움이 됩니다. 이는 비선형 모델을 다루는 새로운 가능성을 열어주며, 공학 및 머신러닝 분야의 다양한 문제에 적합한 유용한 프레임워크를 제공할 것입니다.



### Monge SAM: Robust Reparameterization-Invariant Sharpness-Aware Minimization Based on Loss Geometry (https://arxiv.org/abs/2502.08448)
- **What's New**: 본 논문에서는 sharpness-aware minimization (SAM)의 다변량 문제를 해결하기 위해 Monge SAM (M-SAM)을 제안합니다. M-SAM은 재매개변수화 불변성을 고려하여 Riemannian metric을 사용하여 손실 표면의 기하학을 포착합니다. 이는 SAM이 경험할 수 있는 비풍속적 균형점으로의 끌림을 줄이는 데 도움을 주는데, 이전 접근법에 의존하지 않고도 작동합니다.

- **Technical Details**: M-SAM은 손실 함수의 기하학을 자연스럽게 반영한 Riemannian metric을 도입하여 adversarial perturbation이 검색되는 공간을 제한합니다. 이 방식은 확률적 모델이나 미리 정의된 다양체에 의존하지 않으며, 모든 모델링 선택에 적용할 수 있습니다. 이 논문에서는 M-SAM이 SAM과 gradient descent (GD) 사이에서 다양한 특성을 띠게 되며, 하이퍼파라미터 선택에 대한 강건성을 증가시킴을 이론적으로 증명합니다.

- **Performance Highlights**: M-SAM의 성능이 실제로 향상되는 것을 경험적으로 입증합니다. M-SAM은 SAM보다 하이퍼파라미터 선택에 대한 강건성이 높고, 특정 작업에서 성능에 긍정적인 영향을 미칩니다. 이 연구는 다중 모드 Representation Alignment 작업에서 이러한 행동을 이론적 및 경헙적으로 시연합니다.



### $\texttt{LucidAtlas}$: Learning Uncertainty-Aware, Covariate-Disentangled, Individualized Atlas Representations (https://arxiv.org/abs/2502.08445)
Comments:
          28 pages

- **What's New**: 본 연구의 목표는 의학 분야와 같은 고차원 데이터 집합에서 복잡한 종속성을 갖는 정보를 추출하기 위한 원칙 기반 기법을 개발하는 것입니다. 이를 위해 우리는 공간적으로 변화하는 정보를 표현할 수 있는 $	exttt{LucidAtlas}$라는 접근 방식을 개발했으며, 이는 공변량(covariate)의 영향을 포착하고 인구 불확실성을 제공합니다. $	exttt{LucidAtlas}$는 공변량 해석, 개별화된 예측, 인구 통계 분석 및 불확실성 추정을 위한 강력한 기능을 제공하며, 이전 지식을 수용할 수 있는 유연성도 갖추고 있습니다.

- **Technical Details**: 이 모델은 Neural Additive Model (NAM)의 개념을 확장하여 불확실성 정량화(quantification)와 이전 지식(incorporating prior knowledge)을 통합하는 개인화된 아틀라스 표현 방식을 제공합니다. LucidAtlas는 공변량의 종속성을 고려하여 더 포괄적인 공변량 해석을 제공할 수 있는 마진화(marginalization) 접근 방식을 도입합니다. 이는 기존 NAM의 해석이 개별 공변량 효과에만 국한되는 데 반해, LucidAtlas는 개별 예측기의 모델 응답(아틀라스)에 대한 의존성을 설명합니다.

- **Performance Highlights**: 우리는 두 개의 의학 데이터셋인 OASIS Brain Volume 데이터셋과 Pediatric Airway Shape 데이터셋을 통해 연구 방법의 일반화 가능성을 입증했습니다. 실험을 통해 LucidAtlas가 기존 방법들에 비해 우수한 성능을 보여주었으며, 이는 과학적 발견을 진전시키는 데 있어 구조적으로 해석 가능한 모델의 중요성을 강조합니다. 이 연구의 코드는 논문이 수락되는 즉시 공개될 예정입니다.



### Closer through commonality: Enhancing hypergraph contrastive learning with shared groups (https://arxiv.org/abs/2502.08432)
Comments:
          11page, 5 figures, 6 tables, 2024 IEEE International Conference on Big Data

- **What's New**: 이 논문에서는 Hypergraph Fine-grained Contrastive Learning (HyFi) 방법을 제안하며, 기존의 그래프 기반 대조 학습 방법들이 하이퍼그래프의 높은 차원 정보 및 복잡한 관계를 완전히 활용하지 못하고 있음을 강조합니다. HyFi는 노드 특징에 노이즈를 추가하는 방식으로 하이퍼그래프의 토폴로지를 손상시키지 않으면서 간단하고 효율적인 학습 증강 기능을 제공합니다. 또한, 전통적인 긍정 및 부정 샘플 관계를 넘어서 약한 긍정 샘플 관계를 도입하여 대조 학습의 중요성을 입증합니다.

- **Technical Details**: 하이퍼그래프는 복잡한 다차원 관계를 모델링하는데 뛰어난 기능을 제공합니다. HyFi는 노드에서 하이퍼엣지까지의 고차원 상관관계를 탐색하여 다양한 분야에 적용 가능한 강력한 대조 학습 방법론을 수립합니다. 기존의 하이퍼그래프 대조 학습 방법들이 랜덤 변화를 통해 연결된 그래프에서 대조 학습을 진행하는 것과 달리, HyFi는 원래 뷰 내에서 새로운 긍정 쌍을 식별하여 설정합니다.

- **Performance Highlights**: HyFi는 10개 데이터셋에서 노드 분류 작업을 수행할 때 감독 및 비감독 기준선보다 평균 순위에서 우수한 성능을 보였습니다. 이는 기존의 하이퍼그래프 대조 학습 모델들보다 높은 품질의 임베딩을 생성하는 능력을 보여줍니다. 또한, 하이퍼그래프의 토폴로지를 수정하지 않으므로 시간과 메모리 사용 효율성에서도 개선되었습니다.



### Enhanced Load Forecasting with GAT-LSTM: Leveraging Grid and Temporal Features (https://arxiv.org/abs/2502.08376)
- **What's New**: 본 논문에서는 전력망의 전력 수요 예측을 향상시키기 위해 Graph Attention Networks (GAT)와 Long Short-Term Memory (LSTM) 네트워크를 결합한 하이브리드 모델인 GAT-LSTM을 소개합니다. 이 모델은 주의 메커니즘에 전선 용량과 효율성과 같은 엣지 속성을 통합하여 전력망의 물리적 및 운영적 제약에 기반한 공간적 관계를 동적으로 포착할 수 있습니다. 또한, 공간 그래프 임베딩과 시간적 시퀀스 특징의 조기 융합을 통해 모델은 복잡한 상호작용을 효과적으로 학습하고 예측합니다.

- **Technical Details**: GAT-LSTM 모델은 전력망의 역동적 행동을 실질적으로 나타내기 위해 공간적 및 시간적 의존성을 통합하여 복잡한 상호작용을 포착합니다. 따라서 이 모델은 전력망의 특정한 특징을 보다 효과적으로 반영하고, 예측 정확도를 높여 다양한 시간 범위의 전력 계획을 지원합니다. 실험적으로, 브라질 전력 시스템 데이터 세트에서 GAT-LSTM 모델은 성능지표인 MAE, RMSE, MAPE가 각각 21.8%, 15.9%, 20.2% 감소하는 등 최신 모델들을 능가하는 성능을 보였습니다.

- **Performance Highlights**: GAT-LSTM 모델은 기존의 전력 수요 예측 방법에 비해 매우 우수한 성능을 보여줍니다. 이를 통해 전력망 관리 및 에너지 계획과 같은 다양한 응용 분야에서 강력한 도구로 자리매김하는 기틀을 마련하였습니다. 또한, 이 연구는 전력망의 역 동적인 특성을 효과적으로 반영하여 예측의 신뢰성을 높이는 방향으로 나아가고 있습니다.



### Towards Principled Multi-Agent Task Agnostic Exploration (https://arxiv.org/abs/2502.08365)
- **What's New**: 이번 논문에서는 Multi-Agent Reinforcement Learning (MARL)에서의 task-agnostic exploration 문제를 새로운 방향으로 다루고 있습니다. 기존에 단일 에이전트 상황에서는 이 문제를 탐구한 연구가 많았으나, 다중 에이전트 환경에서는 상대적으로 이해가 부족하다는 점을 강조하고 있습니다. 이 논문은 여러 에이전트가 서로의 존재를 고려하면서 어떻게 탐색을 진행해야 하는지를 설명합니다.

- **Technical Details**: 저자들은 에이전트들이 탐색하는 방식에 따라 세 가지 뚜렷한 목표를 제시합니다. 각 목표는 장점과 단점을 가지고 있으며, 이들의 최적화 방식 또한 다릅니다. 이 과정에서 Trust Region Pure Exploration (TRPE)이라는 분산형 정책 최적화 알고리즘을 도입하여, 다중 시나리오에서의 task-agnostic exploration 문제에 실질적으로 접근하기 위한 방법을 제시합니다.

- **Performance Highlights**: 제안된 알고리즘은 다양한 목표를 최적화할 수 있는 능력을 보여주었으며, 각 목표 간의 중요한 차이가 실질적인 탐색 효과성에 결정적인 역할을 한다고 강조합니다. 실험 결과는 알고리즘이 희소 보상 환경에서의 전이 학습 성과를 향상시키는 데 있어, 효과적인 탐색의 필요성을 뒷받침합니다. 이로 인해 관련된 불행한 결과를 피할 수 있으며, 다중 에이전트 환경에서의 협업을 촉진할 수 있습니다.



### A Survey on Pre-Trained Diffusion Model Distillations (https://arxiv.org/abs/2502.08364)
- **What's New**: 이번 논문은 Generative Artificial Intelligence (GenAI) 내에서 디퓨전 모델(Diffusion Models, DMs)의 효과적인 경량화와 최적화를 다룸으로써, 실용적인 환경에서의 사용 가능성을 높이고자 합니다. 특히, 이 논문은 사전 훈련된 DM에 대한 증류(distillation) 방법을 체계적으로 검토하며, 출력 손실 증류(output loss distillation), 경로 증류(trajectory distillation) 및 적대적 증류(adversarial distillation)라는 세 가지 주요 관점에서 접근합니다.

- **Technical Details**: 이 논문은 DMs의 확산(diffusion) 및 디노이징(denoising) 과정을 [0, 1] 구간 내에서 이해하고, 고유한 기호 체계를 활용하여 모델 매개변수와 함수 관계를 명확하게 설명합니다. DMs의 각 단계는 깨끗한 이미지와 잡음 있는 이미지 간의 변환 과정을 설명하며, 이를 통해 사전 훈련된 모델의 성능을 제고하는 다양한 증류 방법을 탐구합니다.

- **Performance Highlights**: DM의 증류 방법을 통해 고품질 샘플을 더 적은 단계에서 생성할 수 있는 가능성이 제시되었으며, 이는 특히 자원이 제한된 환경에서의 효율성을 크게 향상시킬 것으로 기대됩니다. 현재까지의 증류 방법론을 종합적으로 정리하고, 향후 연구 방향에 대한 제안을 통해 연구자와 실무자에게 유용한 리소스로 기능할 수 있기를 목표로 하고 있습니다.



### Loss Landscape Analysis for Reliable Quantized ML Models for Scientific Sensing (https://arxiv.org/abs/2502.08355)
Comments:
          Under review

- **What's New**: 이 논문에서는 머신러닝(ML) 모델의 손실 풍경(loss landscape)을 분석하기 위한 새로운 방법을 제안합니다. 이 방법은 실험 조건으로 인한 잡음(noise)과 교란(perturbations)의 영향을 받을 수 있는 두 가지 ML 모델, 즉 과학적 센싱을 위한 것입니다. 특히 손실 풍경 분석을 통해 ML 모델의 강인성(robustness)을 정량화하고, 정량화 수준(quantization precision)과 다양한 정규화 기법(regularization techniques)의 영향을 평가할 수 있습니다.

- **Technical Details**: 제안된 방법은 손실 풍경 분석 방법을 통해 ML 모델의 강인한 구성(configuration)을 사전에 식별할 수 있는 기능을 갖추고 있습니다. 본 연구에서는 양자화(quantization)와 데이터 손상(data corruption) 하에서 최첨단 신경망의 강인성을 연구하였으며, 두 가지의 특정 응용 분야에 초점을 맞추었습니다. 이 과정에서 우리는 부드러운 모양의 손실 풍경이 데이터 손상에 대한 robustiveness와 강한 상관관계를 가지고 있음을 밝혔고, 양자화와 강인성 간의 균형을 맞추기 위한 신중한 트레이드오프 탐색의 필요성이 제기되었습니다.

- **Performance Highlights**: 이 연구는 ML 모델을 설계할 때 성능과 효율성뿐 아니라 강인성을 Pareto 최적화에 포함시켜야 한다는 중요성을 강조합니다. 제안된 방법은 사전에 잡음과 교란의 원인에 관계없이 robustiveness에 대한 인사이트를 제공하여, 더 적응력 있는 실험 능력을 가능하게 합니다. 이를 통해 과학적 발견을 가속화하는 데에 기여할 수 있는 새로운 경로를 제공하게 됩니다.



### Trustworthy GNNs with LLMs: A Systematic Review and Taxonomy (https://arxiv.org/abs/2502.08353)
Comments:
          Submitted to IJCAI 2025

- **What's New**: 이 논문은 Graph Neural Networks (GNNs)와 Large Language Models (LLMs)의 통합이 GNN의 신뢰성을 향상시키는 방식에 대한 포괄적인 검토를 제공한다. 연구자들은 이 통합을 통해 머신 모델의 의사결정 신뢰성에 영향을 미칠 수 있음을 인식하고 있다. 우리는 LLM과 GNN의 통합에 관한 새로운 분류 체계를 제안하며, 이 체계를 통해 각 방법의 적용 가능한 시나리오와 한계를 이해할 수 있다.

- **Technical Details**: 우리는 GNN을 그래프 구조(data structure)로 정의하고, 각 노드는 피처 벡터로 설명된다. GNN은 메시지 전달(message-passing) 메커니즘을 통해 노드 표현을 업데이트하며, 이 과정은 서로 이웃한 노드의 정보를 집계(aggregate)하여 수행된다. 본 논문에서는 GNN의 중심 노드 주변 k-hop 구조 내의 정보에 따라 노드 표현이 업데이트된다는 사실을 강조하며, 이를 통해 다양한 그래프 분석 작업을 수행할 수 있다.

- **Performance Highlights**: LLM을 활용한 GNN은 특히 노드에 풍부한 텍스트 속성이 포함된 경우에 뛰어난 성능을 발휘한다. 모델 신뢰성 향상을 위한 LLM과 GNN의 통합 효능에 대해 여러 연구들이 보고되었으며, 예를 들어 LLM4RGNN은 LLM의 추론 능력을 활용하여 악성 엣지를 식별하고 중요 정보를 복구할 수 있다. 본 논문은 향후 연구 방향으로 GNN과 LLM의 통합을 통해 모델의 신뢰성을 더욱 강화하는 방법을 제안한다.



### Hierarchical Learning-based Graph Partition for Large-scale Vehicle Routing Problems (https://arxiv.org/abs/2502.08340)
Comments:
          Accepted as a Full Paper at AAMAS 2025 (24th International Conference on Autonomous Agents and Multiagent Systems)

- **What's New**: 이 논문은 차량 경로 문제(VRP)에 대한 새로운 접근 방식을 제안합니다. 특히, 용량 제약 VRP(CVRP) 문제를 해결하기 위해 글로벌 및 로컬 파티션 정책을 통합한 계층적 학습 기반 그래프 파티션(HLGP) 프레임워크를 도입합니다. 이 방법은 기존의 기술적 한계를 극복하고 재현성을 높이기 위한 것입니다.

- **Technical Details**: HLGP 프레임워크는 다단계 계층 구조를 활용하여 VRP 문제를 해결합니다. 글로벌 파티션 정책이 복잡한 다중 경로 분할을 수행하고, 이후 로컬 파티션 수준에서 해당 단계에 특화된 서브태스크를 생성하여 더욱 정교한 로컬 파티션 정책을 적용합니다. 이는 파티션 과정 중 발생할 수 있는 오류 전파를 줄이는 데 효과적입니다.

- **Performance Highlights**: 제안된 HLGP 프레임워크는 여러 CVRP 벤치마크에 대한 실험을 통해 이전 최첨단 방법들보다 약 10% 성능 향상을 보여줍니다. 특히, CVRP10K 인스턴스에서 Scalability가 강조되며, 강화 학습(RL)과 감독 학습(SL) 모두에 적합한 훈련 목표를 제공합니다.



### Hierarchical Multi-Agent Framework for Carbon-Efficient Liquid-Cooled Data Center Clusters (https://arxiv.org/abs/2502.08337)
- **What's New**: 이 논문은 클라우드 컴퓨팅의 환경 영향을 감소시키기 위해 지리적으로 분산된 데이터 센터 클러스터(DCC) 전반에 효율적인 작업 부하 분배와 개별 데이터 센터 내에서의 냉각 최적화를 동시에 추구하는 Green-DCC를 소개합니다. 특히, Reinforcement Learning (RL) 기반의 계층적 컨트롤러를 통해 작업 부하와 액체 냉각을 동적으로 최적화하는 방법을 제시합니다.

- **Technical Details**: Green-DCC는 날씨, 탄소 강도(carbon intensity), 자원 가용성과 같은 다양한 요소를 고려하여 현실적인 제약과 상호 의존성을 해결합니다. 이 시스템은 여러 데이터 센터를 동시에 최적화할 수 있는 기능을 제공해 디지털 트윈(digital twins)의 범위를 확대합니다. 이를 통해 다양한 RL 접근 방식의 성능을 탄소 배출과 지속 가능성 메트릭을 기준으로 비교합니다.

- **Performance Highlights**: 논문에서는 Green-DCC의 효율성을 입증하기 위해 여러 DCC의 냉각과 작업 부하를 동시적으로 최적화한 사례를 보여줍니다. 또한 지속 가능성 연구를 위한 프레임워크와 벤치마크 시뮬레이션을 제공하여 더 넓은 기계 학습(ML) 연구에 기여하는 방향을 제시합니다.



### Model-Free Counterfactual Subset Selection at Sca (https://arxiv.org/abs/2502.08326)
- **What's New**: 이번 연구는 AI 결정 과정의 투명성을 확보하기 위해 counterfactual explanations를 활용하는 새로운 접근법을 제안합니다. 기존 기술들이 종종 비현실적인 가정에 기반을 두고 있다는 한계를 극복하고, 실시간 데이터 스트리밍 환경에서도 작동 가능한 방법론을 제시합니다. 또한, model-free한 방식을 사용하여 다양한 상황에서 사용할 수 있는 선택 방식을 마련하였습니다.

- **Technical Details**: 제안된 알고리즘은 O(log k)의 업데이트 복잡도로 데이터 스트리밍 환경에서 효율적으로 다양한 counterfactual을 선택합니다. 이는 원래 모델에 대한 접근 없이도 일반화 가능성을 높이며, 여러 대안적 결과를 제공함으로써 사용자들이 실제 데이터에 기반한 피드백을 받을 수 있도록 합니다. 이 작업에서는 제안된 접근 방식의 품질 보장을 위해 단일 탐색(pass)으로 작업이 이루어집니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 실제 및 합성 데이터셋에서 기존 방법들보다 우수한 성질을 보였으며, 다양한 데이터 크기에서도 높은 성능을 유지했습니다. 더불어, 기존의 방법들이 가상의 예제에 의존하는 반면, 본 연구는 실제 데이터에 기반한 설명을 제공함으로써 실용성과 실행 가능성을 높이고 있습니다.



### HDT: Hierarchical Discrete Transformer for Multivariate Time Series Forecasting (https://arxiv.org/abs/2502.08302)
- **What's New**: 이 논문에서는 고차원 다변량 시계열 예측에서의 한계를 해결하기 위해 새로운 접근 방식을 제안합니다. 기존의 생성 모델들은 고차원 데이터에 대한 예측 성능이 낮았고 예측 길이도 제한적이었습니다. 새로운 기법으로 제안된 Hierarchical Discrete Transformer(HDT)는 시계열 데이터를 이산 토큰 표현으로 변환하여 생성 성능을 개선합니다.

- **Technical Details**: HDT는 l2 정규화(l2 normalization)와 강화된 벡터 양자화(vector quantized) 전략을 적용하여 시계열 예측을 이산 토큰 생성으로 전환합니다. 이 모델은 저수준에서의 이산 장기 트렌드를 포착하고, 이를 고수준에서의 타겟 이산 표현 생성을 위한 조건으로 활용합니다. 이러한 접근은 다변량 시계열 데이터의 특성을 반영하여 예측 길이를 연장합니다.

- **Performance Highlights**: 다섯 개의 인기 있는 MTS 데이터셋에 대한 광범위한 실험을 통해 제안된 방법의 효과가 입증되었습니다. HDT는 기존 모델들에 비해 높은 정확도로 예측을 수행할 수 있으며, 고차원 다변량 시계열의 예측을 위한 성능 향상을 보여주었습니다.



### Individualised Treatment Effects Estimation with Composite Treatments and Composite Outcomes (https://arxiv.org/abs/2502.08282)
Comments:
          6 pages (double column), 4 figures

- **What's New**: 이 논문에서는 복합 치료(composite treatments) 및 복합 결과(composite outcomes) 설정에서 개별화된 치료 효과(Individualized Treatment Effect, ITE) 추정을 위한 혁신적인 방법인 H-Learner를 제안합니다. H-Learner는 데이터 부족 문제를 해결하기 위해 치료와 결과 간의 정보를 동적으로 공유하는 하이퍼네트워크(hypernetwork) 기반 접근 방식입니다. 이 연구는 기계 학습(machine learning) 방법론을 기존의 ITE 추정 기술에 통합하여 복잡한 실제 시나리오에서의 적용 가능성을 높이고 있습니다.

- **Technical Details**: H-Learner는 특정 치료 조합에 대해 목표 학습기(target learner)를 생성하는 하이퍼네트워크로 구성됩니다. 이 네트워크는 치료 및 결과에 따라 조건화되어 효과적인 ITE 추정을 제공합니다. 기존의 방법들은 단일 치료와 단일 결과에 한정되어 있었지만, H-Learner는 복수의 치료 및 결과를 동시에 처리할 수 있는 능력을 갖추고 있습니다. 이를 통해 치료와 결과 간의 상관관계를 더욱 잘 모델링 할 수 있습니다.

- **Performance Highlights**: H-Learner는 다양한 치료 및 결과 조합을 포함하는 실증 분석을 통해 기존의 방법들과 비교하여 우수한 성능을 보였습니다. 본 연구는 복합 치료 및 결과 설정에서 H-Learner의 효과를 강조하며, 다양한 실제 사례를 다루고 있어 건강 관리와 같은 분야에서 큰 잠재력을 가지고 있습니다. 이 방법론이 전통적인 ITE 추정 방법들에 비해 중요한 발전을 이룬 것을 확인할 수 있습니다.



### GenIAS: Generator for Instantiating Anomalies in time Series (https://arxiv.org/abs/2502.08262)
- **What's New**: 이번 연구에서는 시계열 이상 탐지 모델(TSAD) 구축을 위한 새로운 접근 방식으로, 실제 데이터 집합 내에서 이상치를 합성 샘플로 주입하는 방법을 제안합니다. 기존 모델들은 수작업으로 만들어진 전략에 의존하여 이상 패턴의 자연스러운 다양성을 제대로 캡처하지 못하거나 단일 변수 시계열 설정에만 제한되는 한계를 가지고 있습니다. 이 연구에서는 변분 오토인코더(Variational Autoencoder)를 이용해 다양한 합성 이상치를 생성할 수 있는 새로운 생성 모델인 GenIAS를 설계하였습니다.

- **Technical Details**: GenIAS는 잠재 공간(latent space) 내에서 학습된 변동 메커니즘을 사용하여, 시계열의 서로 다른 부분에 주입함으로써 다양한 크기와 다양성을 가진 이상치를 생성하는 모델입니다. 이 모델은 복잡한 시계열 데이터의 특성을 다차원 잠재 공간(multidimensional latent space)으로 매핑하여, 일반화된 정상 패턴을 학습할 수 있도록 합니다. 또한, 새로운 triplet loss 함수를 도입하여 비정상 샘플들이 정상 데이터와 명확히 구별되면서도 현실적으로 유지될 수 있도록 설계되었습니다.

- **Performance Highlights**: GenIAS는 17개의 전통적인 및 딥러닝 이상 탐지 모델과 비교하여, TSAD 작업에서 일관되게 우수한 성능을 보였습니다. 이 연구에서는 다양한 평가 지표를 통해 생성된 이상치의 다양성과 현실성을 입증하였으며, 9개의 다양한 UTS 및 MTS 데이터 세트에서 실험을 통해 그 효과성을 확인하였습니다. 연구 결과, GenIAS의 이상치 생성 품질은 전체 모델의 성능 향상과 직접적으로 연결되어 있음을 나타냅니다.



### Balancing optimism and pessimism in offline-to-online learning (https://arxiv.org/abs/2502.08259)
- **What's New**: 이 연구는 오프라인 데이터로부터 시작해 온라인 환경과 상호작용하는 학습자에게 적용되는 새로운 알고리즘을 소개합니다. 오프라인-온라인 학습 문제를 다루며, 비관적(pessimism) 접근법인 Lower Confidence Bound (LCB)와 낙관적(optimism)인 Upper Confidence Bound (UCB) 알고리즘의 균형을 최적화합니다. 이 알고리즘은 다양한 시간 수평선에서 성능을 조정할 수 있으며, 현재까지의 연구에서 다뤄지지 않은 변화를 제안합니다.

- **Technical Details**: 연구에서는 학습자의 정책이 단기적으로 사용되면 LCB가 유리하다는 점을 강조합니다. 반면, 장기 전망에서는 UCB 전략이 최적의 정책에 대한 수렴 속도가 가장 빠르다는 강점을 가집니다. 이 알고리즘은 또한 상호작용이 증가함에 따라 LCB에서 UCB와 유사한 전략으로 점진적으로 전환하는 방법을 탐구합니다.

- **Performance Highlights**: 새로운 알고리즘은 LCB와 UCB 중 더 나은 성능을 가지는 쪽과 비슷한 성능을 보입니다. 연구 결과는 다양한 MAB(multi-armed bandit) 문제에 일반화될 수 있다고 예상되며, 본 알고리즘은 대안적 탐험 및 활용 전략을 통합하여 효율성을 높입니다.



### Keep your distance: learning dispersed embeddings on $\mathbb{S}_d$ (https://arxiv.org/abs/2502.08231)
- **What's New**: 본 논문에서는 고차원 공간에서 잘 분리된 피처(features) 학습의 중요성을 강조합니다. 임베딩(embedding)의 분산(dispersion)을 통한 효율적인 피처 분리를 논의하며, 다양한 기존 방법들을 재조명합니다. 새로운 최대 평균 차이(maximum mean discrepancy, MMD) 관점을 소개하고, K-Means 알고리즘의 온라인 변형을 제안하여 보다 효과적인 분산 정규화(regularizer) 방법을 제시합니다.

- **Technical Details**: 저자들은 고차원 공간에서 피처들이 하이퍼구(Hypersphere) 위에 위치하도록 제한함으로써, 수학 및 물리학에서 잘 알려진 최적 해와 연결 지을 수 있음을 논의합니다. 또한, 페어(pairwise) 거리의 함수 최소화를 통해 분산을 촉진하는 경량 기법의 필요성에 대해 설명합니다. 새로운 알고리즘은 하이퍼구의 특성을 직접적으로 활용하여 분산 문제를 해결합니다.

- **Performance Highlights**: 실험 결과는 이미지 분류(image classification) 및 자연어 처리(natural language processing) 작업에서 분산의 중요성을 보여주고, 다양한 알고리즘이 서로 다른 환경에서 어떻게 정무(정책) 간의 다른 균형(trade-offs)을 나타내는지를 분석합니다. 이는 고차원 특성 분리에 있어 분산의 역할을 명확히 하며, 새로운 방법론의 필요성을 증명합니다.



### Enhancing Sample Selection by Cutting Mislabeled Easy Examples (https://arxiv.org/abs/2502.08227)
- **What's New**: 본 논문에서는 잘못 라벨링된(Mislabeled) 샘플 중에서도 모델이 초기 학습 과정에서 올바르게 예측한 경우에 특히 모델 성능에 해로운 영향을 미친다는 점을 강조합니다. 이를 Mislabeled Easy Examples (MEEs)라고 명명하였으며, 이 예제들은 모델의 성능을 저하시킬 수 있는 중요한 요소로 규명되었습니다. 기존의 샘플 선택 기법들이 이러한 MEEs의 영향을 간과하고 있다는 점이 주목받고 있습니다.

- **Technical Details**: 저자들은 Early Cutting이라는 새로운 샘플 선택 전략을 제안하여, 초기 학습에서 식별된 샘플의 신뢰할 수 있는 부분을 후속 재선정을 통해 조정하는 방법을 소개합니다. 이 과정에서는 고수치 손실을 보이면서도 자신 있게 예측된 샘플을 식별하여 MEEs를 효과적으로 필터링합니다. 이 방법은 모델의 초기 학습 단계에서 강하게 거론되는 낮은 신뢰도의 샘플들을 제거하여 모델의 일반화 성능을 개선하려고 합니다.

- **Performance Highlights**: CIFAR, WebVision 및 전체 ImageNet-1k 데이터셋에서의 실험 결과, 제안된 방법이 다양한 설정에서 최신 샘플 선택 기법들을 능가한 것으로 나타났습니다. 특히, MEEs의 부정적인 영향을 줄이면서 모델의 성능을 효과적으로 개선하는 성과를 보여줍니다. 저자들은 이를 통해 신뢰할 수 있는 샘플 선택을 통한 학습 성능 향상을 실증적으로 입증하였습니다.



### Quality over Quantity: Boosting Data Efficiency Through Ensembled Multimodal Data Curation (https://arxiv.org/abs/2502.08211)
- **What's New**: 이번 연구에서는 데이터 커레이션(data curation)의 새로운 프레임워크인 EcoDatum을 제안합니다. EcoDatum은 여러 단일 모드 및 다중 모드 데이터 커레이션 연산자를 통합하여 약한 감독 학습의 앙상블(ensemble) 구조를 활용하며, 자동 최적화를 통해 각 데이터 포인트의 품질 점수를 효과적으로 평가합니다. 이 프레임워크는 데이터 커레이션 품질과 효율성을 크게 향상시켜, 기존 기술보다 뛰어난 성과를 거두었습니다.

- **Technical Details**: EcoDatum은 질적 가이드에 따라 중복 제거(quality-guided deduplication) 방법을 도입하여 균형 잡힌 특성 분포를 보장합니다. 또한, 시각적 콘텐츠와 텍스트를 기반으로 해시 코드를 생성하여 중복을 식별하며, CLIP 모델을 통해 각 중복 그룹의 의미적 일관성을 평가합니다. 이를 통해 데이터 커레이션 과정이 자동화되어 품질 점수가 생성되며, 수동 입력을 최소화하고 임계값 설정의 정확성이 향상됩니다.

- **Performance Highlights**: EcoDatum은 DataComp 리더보드에서 1위를 기록하며, 38개의 다양한 평가 데이터셋에서 평균 성능 점수 0.182를 달성하였습니다. 이는 DataComp 기준 방법에 비해 28% 향상된 결과입니다. 이러한 성과는 데이터셋 커레이션과 모델 훈련 효율성을 크게 개선할 수 있는 가능성을 보여줍니다.



### Equivariant Masked Position Prediction for Efficient Molecular Representation (https://arxiv.org/abs/2502.08209)
Comments:
          24 pages, 6 figures

- **What's New**: 이번 연구에서는 그래프 신경망(GNN)이 화학 분야에서의 적용 가능성을 그러면서도, 분자 데이터의 한계로 인해 일반화 능력이 제한된다는 문제를 해결하기 위해 새로운 자기 지도 학습 방식인 Equivariant Masked Position Prediction(EMPP)을 제안합니다. EMPP는 전통적인 속성 마스킹 기술과 달리 intramolecular potential 및 force 이론에 기반하여 실질적인 위치 예측 과제를 형성하여 양자역학적 특징 학습을 향상시킵니다.

- **Technical Details**: EMPP는 원자의 3D 위치를 무작위로 마스킹하고, 이는 다른 속성, 예를 들어 원자 번호 등을 그대로 유지함으로써 더 잘 정의된 문제를 만들어 냅니다. EMPP는 마스킹된 원자의 위치를 인근 구조에서 양자역학적으로 결정하며, 이는 기존의 노이즈를 예측하는 방법과는 기본적으로 다른 접근 방식을 채택합니다. EMPP는 Gaussian mixture 분포의 근사를 피하고, 더 결정적인 위치 예측을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, EMPP는 고급 화학 GNN 구조의 성능을 크게 향상시켰고, 기존의 마스킹 및 노이즈 방법보다 우수한 성과를 달성했습니다. 본 연구에서 제안한 EMPP는 자가 감독 학습 과정을 통해 일반 지식을 포착하며, 특정 양자 특성과 원자 위치 간의 연결을 강화하여 다양한 분자 작업에서 GNN의 일반화 성능을 개선합니다.



### Exploring Exploration in Bayesian Optimization (https://arxiv.org/abs/2502.08208)
Comments:
          28 pages, 34 figures

- **What's New**: 본 논문에서는 Bayesian optimization에서 acquisition functions의 exploration 특성을 정량화하기 위한 두 가지 새로운 접근법, 즉 observation traveling salesman distance (OTSD)와 observation entropy (OE)를 소개합니다. 이를 통해 다양한 black-box 문제에서의 탐사 특성을 분석하고, exploration과 empirical performance 간의 관계를 밝혀냅니다. 이러한 정량적 측정 방법들은 acquisition functions의 설계를 더욱 체계적으로 안내할 수 있는 기반을 제공합니다.

- **Technical Details**: Bayesian optimization(BO)은 black-box 함수 최적화를 위해 널리 사용됩니다. 이 과정에서 Gaussian process(GP)를 통해 probabilistic surrogate를 구성하고 acquisition function(AF)를 최적화하여 새로운 샘플링 포인트를 선택합니다. AF는 탐사(exploration)와 활용(exploitation)의 균형을 잘 맞춰야 하며, 다양한 AF들이 탐사 선호도 측면에서 다르게 행동함을 알려주고 있습니다.

- **Performance Highlights**: OTSD와 OE는 benchmark 문제에서 강한 상관성을 보이며 그 신뢰성을 검증하였습니다. 다양한 저차원 및 고차원 synthetic, 실제 벤치마크를 통해 널리 사용되는 acquisition functions의 탐사 행동을 시연했습니다. 이 연구는 AF의 탐사 특성을 이해하는 데 기여하며, 실제 문제에 적합한 AF 선택에 있어 중요한 영향을 미치는 지식을 제공하였습니다.



### Optimizing Asynchronous Federated Learning: A Delicate Trade-Off Between Model-Parameter Staleness and Update Frequency (https://arxiv.org/abs/2502.08206)
- **What's New**: 이 논문은 비동기 연합 학습(asynchronous Federated Learning, FL)에서 클라이언트 수 증가에 따른 성능 저하 문제를 해결하기 위해 새로운 최적화 방법론을 제안합니다. 특히, 기존의 기법들이 간과했던 큐(queue) 역학(dynamics)과 클라이언트의 처리 속도 변동성을 분석함으로써, 모델 업데이트의 속도와 모델 파라미터의 신선도(freshness) 사이의 기본적인 트레이드오프를 최적화하는 데 집중합니다. 이를 통해 평균 손실을 효과적으로 최소화하는 방법을 도출했습니다.

- **Technical Details**: 비동기 FL에서의 상대 지연(relative delay)은 클라이언트가 작업을 지연시키는 과정을 정의합니다. 본 논문은 Jackson 네트워크 이론을 활용하여 상대 지연의 평균과 그 기울기를 계산하는 명시적 표현을 도출했습니다. 이를 통해 비동기 FL 알고리즘인 Generalized AsyncSGD의 성능을 최적화할 수 있는 알고리즘을 설계하였으며, 처리 속도를 감안한 대안 지표를 제시하여 실제 클라이언트 사용을 극대화할 수 있는 조건을 제안합니다.

- **Performance Highlights**: 제안된 최적화 방법은 실제 데이터 집합에 대한 실험을 통해 정확도를 10%에서 30%까지 향상시키는 것으로 나타났습니다. 또한, 클라이언트의 처리 속도와 큐 역학이 비동기 FL의 성능에 미치는 중요성을 강조하며, 적용되는 경로 전략을 조정하는 것이 전체 시스템의 성능 향상에 크게 기여할 수 있다는 점을 시사합니다. 이러한 결과는 비동기 FL 알고리즘의 발전에 중요한 기여를 할 것으로 기대됩니다.



### Wisdom of the Crowds in Forecasting: Forecast Summarization for Supporting Future Event Prediction (https://arxiv.org/abs/2502.08205)
- **What's New**: 본 논문은 집단 지혜를 활용한 미래 사건 예측(Future Event Prediction, FEP)에 관한 기존 연구와 프레임워크를 정리하고 새로운 데이터 모델을 제안합니다. 새로운 FEP-CW(Future Event Prediction based on Crowd Wisdom) 접근 방식을 도입하여 개별 예측을 집계함으로써 복잡한 사건 예측의 신뢰성을 높일 수 있는 가능성을 탐구합니다. 이를 통해 집단의 의견을 통합하여 예측의 정확성을 개선할 수 있는 다양한 방법을 제시합니다.

- **Technical Details**: FEP-CW의 개념을 다루며, 관련 데이터 수집 유형과 정보 추출 방법들, 예측을 시각화하는 기법들이 포함됩니다. 또한 아카이브된 뉴스, 트위터 및 웹사이트의 데이터셋을 기반으로 기존 연구들을 분석하며, FEP-CW를 대해 검토하게 됩니다. 이 연구는 주로 예측 관련 메시지를 포함하는 텍스트 기반 데이터에 초점을 맞추고 있으며, 기존의 전통적인 접근법과 차별화된 점을 강조합니다.

- **Performance Highlights**: 총 36개의 관련 논문을 선정하여 FEP-CW 분야의 연구 동향을 종합적으로 살펴봅니다. 이전 연구와 비교할 때, 본 연구의 결과는 집단 지혜를 활용한 미래 예측의 가능성과 그 적용 방안에 대해 과학적 근거를 제공합니다. 다양한 데이터셋 포맷을 이용하여 예측 모델의 성능을 평가하고, 이러한 접근법이 어떻게 더 향상될 수 있는지에 대한 방향성을 제시합니다.



### Privacy amplification by random allocation (https://arxiv.org/abs/2502.08202)
- **What's New**: 이번 논문에서는 사용자의 데이터가 k단계에서 무작위로 선택되는 알고리즘에 대한 개인정보 보호 보장을 분석합니다. 이러한 샘플링 방식의 개인정보 보호 보장은 잘 알려진 독립적(또는 Poisson) 서브샘플링에 의해 상한이 설정될 수 있음을 보여줍니다. 추가적으로, 특정 파라미터 영역에서 성능을 개선하는 두 가지 분석 기법도 제공합니다.

- **Technical Details**: 논문에서는 상관관계가 있는 k개 요소의 무작위 선택을 통해 데이터를 분석하는 새로운 샘플링 기법을 소개합니다. 이 방법은 데이터의 각 요소가 참여하는 횟수를 제한하면서도, 효과적으로 개인정보 보호를 보장합니다. 특히, k=1의 경우에 대해 이전 연구들에서 얻은 결과를 확장하여 k/t의 비율로 Poisson 샘플링과 유사한 보호 보장을 제공한다고 주장합니다.

- **Performance Highlights**: 제안된 샘플링 방식은 Poisson 서브샘플링과 비교했을 때 유사하거나 더 나은 성능을 보여줍니다. 이를 통해 알고리즘의 개인정보 보호 보장이 개선될 수 있으며, 시뮬레이션에 의존하지 않고 실증적인 보장이 가능함을 강조합니다. 논문에서 제시된 결과는 향후 가우시안 노이즈를 추가하는데 필요한 보장들을 정량적으로 뒷받침합니다.



### Latest Advancements Towards Catastrophic Forgetting under Data Scarcity: A Comprehensive Survey on Few-Shot Class Incremental Learning (https://arxiv.org/abs/2502.08181)
- **What's New**: 이번 논문은 데이터 희소성이 끼치는 영향을 다루며, Few-shot Class Incremental Learning (FSCIL) 방법론에 대한 포괄적인 조사를 제공합니다. FSCIL은 동적 환경에서 소수의 샘플로 학습해야 하는 머신러닝 모델의 문제를 모사합니다. 최근 진행된 연구들에서 파생된 이 방법은 기존의 학습 방식의 한계를 넘어서는 솔루션을 모색하고 있습니다.

- **Technical Details**: FSCIL 문제는 주어진 작업 시퀀스에서 각 작업이 레이블링된 훈련 세트를 포함하고 있으며, 두 번째 작업 이후의 데이터는 초기 작업보다 훨씬 적은 샘플을 가지고 있음을 보여줍니다. 데이터 희소성 문제를 해결하기 위한 최근의 PEFT(parameter-efficient fine-tuning) 접근 방법과 언어 기반 학습 방식이 특히 강조되고 있습니다. 이는 기존 모델의 성능 의존성을 줄이고, 모델 훈련 시간을 대폭 단축시켜 줍니다.

- **Performance Highlights**: 이 논문은 FSCIL 방법의 최신 발전을 포함한 포괄적인 토폴로지를 제안하며, 각 접근 방식의 공식 목표와 서브 설정을 분석합니다. 프로토타입 편향을 해결하기 위한 프로토타입 정정의 중요성도 강조하며, 현재 직면한 개방 문제, 잠재적 솔루션 및 FSCIL의 미래 방향에 대한 철저한 분석을 제공합니다.



### DNNs May Determine Major Properties of Their Outputs Early, with Timing Possibly Driven by Bias (https://arxiv.org/abs/2502.08167)
Comments:
          First two authors contributed equally

- **What's New**: 이번 연구는 깊은 신경망(DNNs)이 초기 추론 단계에서 모델 내재된 편향(bias)에 따라 출력을 결정한다는 주장을 하고 있습니다. 우리는 DNN의 결정 과정이 인간의 빠르고 직관적인 의사결정 방식과 유사하다는 점을 강조하며, 이런 인식이 기계 학습 시스템의 해석과 편향 완화 효율성에 대한 새로운 관점을 제공할 수 있음을 제시합니다.

- **Technical Details**: 우리는 특히 확산 모델(diffusion models)을 사례 연구로 사용하여 DNN의 초기 결정 메커니즘과 그 결정의 동적 과정을 분석했습니다. DNN의 추론 과정에서 특정 특성에 대한 모델의 편향이 결정의 시간에 영향을 미친다는 가정을 세우고, 다양한 텍스트-이미지 변환 모델을 통해 이 가설을 실험적으로 검증하였습니다. 이를 통해 모델들이 초기 단계에서 출력을 결정하는 경향이 있음을 보여주었습니다.

- **Performance Highlights**: 실험 결과, 대부분의 최신 T2I DMs은 초기 추론 단계에서 출력을 결정하며, 이는 약 5단계 내에 이루어집니다. 특히 편향적인 속성을 가지고 있는 입력(예: 색상)은 덜 편향적인 속성(예: 재질)보다 결정하는 시간이 더 빠른 경향을 보였습니다. 이러한 결과는 DNN의 초기 결정 메커니즘이 편향에 따라 다르게 작동하고 있다는 점을 시사합니다.



### Vertical Federated Learning in Practice: The Good, the Bad, and the Ugly (https://arxiv.org/abs/2502.08160)
- **What's New**: 이 논문에서는 Vertical Federated Learning (VFL)의 기존 연구와 실제 적용 간의 격차를 분석합니다. 연구팀은 VFL 알고리즘을 실제 데이터 분포를 기반으로 한 새로운 데이터 중심 분류도로 제안하고, 이를 통해 현재의 알고리즘과 실제 응용 간의 큰 간극을 드러냅니다. VFL이 여러 산업에서의 협업을 가능하게 함에도 불구하고, 실제 적용 사례가 극히 제한적이라는 점이 강조됩니다.

- **Technical Details**: VFL은 서로 다른 특성을 가진 여러 당사자가 공동으로 기계 학습 모델을 훈련할 수 있도록 하는 개인 정보 보호 협업 학습 패러다임입니다. VFL 작업은 하나의 주요 당사자가 라벨을 보유하고 있으며, 다른 당사자는 보조 당사자로 정의됩니다. 이 논문은 {𝐗1,𝐗2,…,𝐗C} 형태의 결합된 데이터셋에서 VFL 모델을 훈련하는 과정과, 개인 정보 보호를 위한 레코드 연결 기법을 다룹니다.

- **Performance Highlights**: VFL의 잠재적 데이터 분포를 조사한 결과, 특정 실용적 시나리오에 대한 유용한 솔루션이 부족하다는 사실이 밝혀졌습니다. 기존 알고리즘이 실제 데이터 분포와 잘 맞지 않음을 지적하며, 이 격차를 메우기 위한 연구 방향을 제시합니다. 이로 인해 VFL 관련 향후 연구는 실제 응용 프로그램의 특성을 반영하도록 초점을 맞춰야 할 필요성이 있음을 강조합니다.



### DGSense: A Domain Generalization Framework for Wireless Sensing (https://arxiv.org/abs/2502.08155)
Comments:
          15 pages

- **What's New**: 본 논문은 무선 감지에서 도메인 의존성 문제를 해결하기 위해 새로운 도메인 일반화 프레임워크인 DGSense를 제안합니다. DGSense는 다양한 감지 작업과 무선 기술에 적용할 수 있으며, 새롭게 나타나는 도메인에서도 학습 데이터 없이 일반화할 수 있는 능력을 가지고 있습니다. 이 프레임워크는 가상 데이터 생성기와 에피소드 학습 전략을 채택하여 훈련 세트의 다양성을 높이고 도메인 독립적인 특성을 추출합니다.

- **Technical Details**: DGSense는 주요 피처 추출기와 도메인 피처 추출기 간의 에피소드 학습을 통해 도메인 독립적인 특성을 획득하며, 공간적 특성을 위한 Residual Network (ResNet)와 시간적 특성을 위한 1D Convolutional Neural Network (1DCNN)를 사용합니다. 가상 데이터 생성기는 Variational Autoencoder (VAE)를 기반으로 하여 다양한 모달리티를 보장하고 데이터 희소성을 완화하는 데 도움을 줍니다. 이를 통해 더 많은 데이터 다양성과 모델의 강건성을 확보할 수 있습니다.

- **Performance Highlights**: DGSense는 WiFi 제스처 인식, 밀리미터파(mmWave) 활동 인식, 음향 낙상 감지 등에서 높은 일반화 능력을 입증하였습니다. 모든 시스템은 새로운 사용자, 장소, 환경에 대해서도 새로운 데이터나 재훈련 없이 높은 성능을 유지하는 것으로 나타났습니다. 실험 결과 DGSense의 효과성과 일반성을 확인할 수 있었습니다.



### Force Matching with Relativistic Constraints: A Physics-Inspired Approach to Stable and Efficient Generative Modeling (https://arxiv.org/abs/2502.08150)
- **What's New**: 본 논문에서는 Generative Modeling의 새로운 틀인 Force Matching (ForM)을 소개합니다. 이 프레임워크는 특수 상대성 이론을 적용하여 샘플링 과정의 안정성을 향상시키기 위한 초기 탐험을 나타냅니다. Lorentz 인자를 포함함으로써, ForM은 속도 제약을 부여하여 샘플의 속도를 일정한 한계 내에서 유지하도록 보장합니다.

- **Technical Details**: ForM은 샘플 속도를 제한하는 기본 메커니즘을 통해 생성 동역학을 안정화합니다. 본 논문에서는 ForM 프레임워크 내에서 샘플링 절차 전반에 걸쳐 속도 제약이 유지된다는 것을 엄밀한 이론적 분석을 통해 증명합니다. ForM의 효용성을 검증하기 위해 다양한 실험적 평가를 수행하였습니다.

- **Performance Highlights**: empirical evaluations에서 ForM은 baseline 방법보다 월등한 성능을 보였습니다. 특히, half-moons 데이터셋에서 ForM은 0.714의 Euclidean distance loss를 기록하며, 이는 기존의 vanilla first-order flow matching(5.853) 및 first- and second-order flow matching(5.793)보다 현저히 낮은 수치입니다. 본 연구는 ForM을 통해 안정적이고 효율적인 생성 프로세스를 구현할 수 있는 가능성을 보여줍니다.



### Knowledge-Guided Wasserstein Distributionally Robust Optimization (https://arxiv.org/abs/2502.08146)
- **What's New**: 전통적인 기계 학습 방법이나 경험적 위험 최소화는 특히 고차원 및 소표본 환경에서 과적합(overfitting) 및 일반화 능력 부족의 문제를 겪고 있습니다. 이러한 문제를 극복하기 위해 연구자들은 분포적으로 강인한 최적화(distributionally robust optimization, DRO) 접근법을 개발하였고, 그 중에서도 Wasserstein-DRO(WDRO)는 효율성과 일반화 가능성 때문에 주목을 받고 있습니다. 그러나 기존 WDRO는 지나치게 보수적인 경향이 있어 실제 적용에서 최적의 성능을 발휘하지 못하는 한계가 있었습니다.

- **Technical Details**: KG-WDRO(지식 기반 Wasserstein 분포적 강인 최적화)는 이러한 보수성을 극복하기 위해 외부 지식을 동적으로 통합하는 새로운 프레임워크입니다. 이 프레임워크는 외부 지식에 의해 정보가 제공된 방향으로 수송 비용을 조절함으로써 소규모 샘플의 전달 학습을 개선할 수 있습니다. 또한, KG-WDRO는 기존의 WDRO 프레임워크와 비교할 때 일반적인 규제 기법(lasso, ridge)에도 적용 가능하며, 회귀 분석 및 이진 분류 문제를 포괄적으로 해결합니다.

- **Performance Highlights**: KG-WDRO는 광범위한 시뮬레이션을 통해 소규모 샘플 전달 학습에서 우수한 성능과 적응성을 보여주었습니다. 이 새로운 접근법은 과거 모델에 의해 제공된 지식을 효과적으로 활용하여 예측 정확도를 향상시키는 데 기여합니다. 이를 통해 실제 의료 데이터와 같은 다양한 분야에서 전달 학습의 유용성을 극대화할 수 있을 것으로 기대됩니다.



### Democratizing AI: Open-source Scalable LLM Training on GPU-based Supercomputers (https://arxiv.org/abs/2502.08145)
- **What's New**: 이 논문에서는 수십억 개의 매개변수를 가진 대형 언어 모델(LLM)을 효율적으로 훈련하기 위한 새로운 네 차원 하이브리드 병렬 알고리즘을 소개합니다. 이 알고리즘은 AxoNN이라는 오픈소스 프레임워크에 구현되어 있으며, 뛰어난 성능 최적화를 포함하여 모델 훈련의 속도와 효율성을 크게 향상시킵니다. 특히, Perlmutter, Frontier 및 Alps와 같은 슈퍼컴퓨터에서의 훈련 성능은 기적적인 수치인 1.423 Exaflop/s, 1.381 Exaflop/s 및 620.1 Petaflop/s를 기록했습니다.

- **Technical Details**: 설계된 4D 하이브리드 접근 방식은 세 차원 행렬 곱셈 알고리즘과 데이터 병렬성을 결합하여 많은 GPU에서 높은 효율성을 달성합니다. AxoNN의 성능은 각 플랫폼에 따라 행렬 곱셈을 조정하고, 계산과 비차단 집단 통신을 겹치는 최적화를 통해 개선되었습니다. 이 알고리즘은 다양한 GPU에서의 최적 수행 구성을 예측할 수 있는 통신 모델도 포함하고 있습니다.

- **Performance Highlights**: AxoNN은 NVIDIA A100 GPU에서 620.1 Petaflop/s, AMD MI250X GCD에서 1.381 Exaflop/s, NVIDIA H100 GPU에서 1.423 Exaflop/s의 전례 없는 성능을 달성했습니다. 이러한 결과는 LLM 훈련에서의 메모리 사용과 계산 효율성을 크게 향상시키며, 연구자들이 LLM의 작동 메커니즘을 보다 심도 있게 연구할 수 있도록 도와줍니다. 또한, 모델 크기와 메모리화 성질 간의 관계를 탐구하고, 개인 정보 보호 위험을 줄이기 위한 접근법을 제시합니다.



### Data-dependent Bounds with $T$-Optimal Best-of-Both-Worlds Guarantees in Multi-Armed Bandits using Stability-Penalty Matching (https://arxiv.org/abs/2502.08143)
- **What's New**: 이번 논문에서는 멀티암드 밴디트 문제에 대해 데이터 의존적(data-dependent)이고 최적화된(best-of-both-worlds) 새로운 회피 경계를 제시합니다. 기존 방식들의 제한을 극복하기 위해 실시간 안정성-패널티 매칭(real-time stability-penalty matching, SPM) 기법을 도입했습니다. 이 기법은 $O(rac{1}{	ext{sqrt}(T)})$와 $O(	ext{ln}(T))$의 경계 보장을 제공하며 데이터 의존적 특성을 동시에 충족합니다.

- **Technical Details**: SPM 방법론은 follow-the-regularized-leader (FTRL) 프레임워크의 학습률 조정 기법을 확장하여 개발되었습니다. 이 방식은 adversarial 및 stochastic 두 가지 경우에 대해 적응 가능한 보장을 제공하여 멀티암드 밴디트 문제에 효과적인 성능을 보여줍니다. 특히, sparsity, variations, 작은 손실 등 데이터 의존적 특성을 고려한 경계를 제공합니다.

- **Performance Highlights**: 실시간 SPM은 적대적(adversarial) 환경에서 $O(	ext{sqrt}(T))$의 최악의 경우 보장을, 확률적(stochastic) 환경에서는 $O(	ext{ln}(T))$의 보장을 달성합니다. 이러한 결과는 온라인 학습 문제에서 새로운 적응 경계를 증명하는 유망한 접근법으로 SPM과 FTRL의 조합이 효과적임을 나타냅니다. 따라서 멀티암드 밴디트 문제를 해결하는 데 있어 전반적으로 향상된 성능을 기대할 수 있습니다.



### LowRA: Accurate and Efficient LoRA Fine-Tuning of LLMs under 2 Bits (https://arxiv.org/abs/2502.08141)
- **What's New**: 본 논문에서는 기존의 LoRA(LoRA: Low-Rank Adaptation) 방법보다 개선된 초저비트(ultra-low-bit) LoRA 미세 조정을 가능하게 하는 LowRA 프레임워크를 소개합니다. LowRA는 2비트 이하로 매개변수의 손실 없이 LoRA 미세 조정을 지원하며, 메모리 사용량을 최대 50%까지 줄일 수 있습니다. 이는 자원이 제한된 환경에서의 적용 가능성을 높이는 혁신적인 접근입니다.

- **Technical Details**: LowRA는 매핑(mapping) 및 임계값(threshold) 선택, 세밀한 정밀도 할당(precision assignment)이라는 세 가지 주요 구성 요소로 이루어져 있습니다. 이 프레임워크는 작업에 무관한 양자화 기술을 활용하여 LLM(대형 언어 모델)의 주어진 기저 가중치에 대해 다수의 어댑터 세트를 사용할 수 있는 능력을 제공합니다. 또한, LowRA는 효율적인 CUDA 커널을 활용하여 대규모 배포에 최적화되어 있습니다.

- **Performance Highlights**: LowRA는 2비트 이상의 성능-정밀도(performance-precision) 무역에서 우수한 결과를 기록하였으며, 1.15비트까지도 정확도를 유지합니다. 또한, LowRA는 30-50%의 메모리 사용량을 감소시키면서도 미세 조정 과정에서 성능 손실을 최소화합니다. 이러한 결과는 LowRA가 자원의 제약이 있는 환경에서 LLM을 효과적으로 미세 조정하고 배포할 수 있는 가능성을 제시합니다.



### In-Context Learning of Linear Dynamical Systems with Transformers: Error Bounds and Depth-Separation (https://arxiv.org/abs/2502.08136)
- **What's New**: 이 논문은 트랜스포머(transformers)의 인컨텍스트 학습(in-context learning) 능력에 관한 근사 이론적 측면을 조사했습니다. 특히 노이즈(linearly corrupted noise)가 있는 선형 동적 시스템에 대해 다층 트랜스포머의 근사 오차(approximation error) 경계를 Established했습니다. 이 결과는 로가리즘 깊이(logarithmic depth)를 가진 트랜스포머가 최소 제곱 추정기(least-squares estimator)와 유사한 오차 경계를 달성할 수 있음을 보여줍니다.

- **Technical Details**: 연구의 첫 번째 결과는 $L^2$-테스트 손실(testing loss)에 대해 다층 트랜스포머의 근사 오차에 대한 상한을 설정하였습니다. 이를 통해 다층 트랜스포머가 특정 선형 동적 시스템에 대해 정확한 예측을 할 수 있는지에 대한 질문을 다룹니다. 두 번째 결과는 단일 층(linear single-layer) 트랜스포머의 근사 오차에 대한 하한을 제공하며, 이는 인컨텍스트 학습에서 깊이 분리를 나타냅니다.

- **Performance Highlights**: 연구 결과, 최소 제곱 추정기를 근사하는 다층 트랜스포머는 학습한 동적 시스템에서 좋은 성능을 보입니다. 특히, O(log(T))/T의 오차를 보이며, 이는 구체적인 클래스의 선형 동적 시스템을 학습하는 데 매우 중요한 통찰을 제공합니다. 반면, 단일 층 트랜스포머는 상관 데이터(carrier data)에서 학습하는 데 근본적으로 한계가 있음을 발견했습니다.



### PoGDiff: Product-of-Gaussians Diffusion Models for Imbalanced Text-to-Image Generation (https://arxiv.org/abs/2502.08106)
- **What's New**: 이 논문에서는 imbalanced dataset(불균형 데이터셋)에서 diffusion models(확산 모델)의 성능 저하 문제를 해결하기 위해 새로운 파인튜닝 접근 방식인 PoGDiff를 제안합니다. PoGDiff는 예측된 분포와 진짜 분포 간의 KL divergence를 최소화하기보다, Product of Gaussians(PoG)를 사용하여 원래의 진짜 타겟과 이웃 텍스트 임베딩에 조건화된 예측 분포를 결합하여 구성합니다.

- **Technical Details**: Diffusion models(DMs)는 임의의 노이즈 벡터를 조건으로 하여 이미지를 생성하는 확률 모델입니다. 이 모델은 진행적 노이즈 추가와 줄어드는 노이즈 제거 과정을 포함합니다. PoGDiff는 이러한 모델의 훈련을 개선하여 유사한 텍스트 프롬프트에 대해 동일한 이미지를 생성하도록 장려하는 방식으로 설계되었습니다.

- **Performance Highlights**: 실험 결과, PoGDiff는 실제 데이터셋에서 임Balance 문제를 효과적으로 해결하고, 생성 정확도와 품질을 개선하며 다른 최첨단 모델들을 초과하는 성과를 보여주었습니다. 새롭게 제안된 ‘Generative Recall’(gRecall) 메트릭은 모델의 생성 다양성을 평가하는 데에 큰 역할을 합니다.



### Out-of-Distribution Detection on Graphs: A Survey (https://arxiv.org/abs/2502.08105)
Comments:
          9 pages, 6 figures

- **What's New**: 최근 그래프 머신러닝(Graph Machine Learning, GML)의 발전은 다양한 분야에서 큰 진전을 이루고 있습니다. 특히, 훈련 데이터와 테스트 데이터가 서로 다른 분포를 가지는 실제 환경에서 모델 성능 저하 문제를 해결하기 위해 그래프 분포 외(GOOD) 탐지가 주목받고 있습니다. 본 논문은 GOOD 탐지의 엄격한 정의와 함께 기존의 방법들을 네 가지 범주로 체계적으로 분류하고, 이 분야의 기초를 다지는 새로운 설문조사를 소개합니다.

- **Technical Details**: GOOD 탐지는 인-디스트리뷰션(IND) 데이터와 아웃-오브-디스트리뷰션(OOD) 데이터를 구분하는 데 중점을 둡니다. 저자는 그래프를 구성하는 정점 집합(𝒱), 간선 집합(ℰ), 인접 행렬(𝐀), 특성 행렬(𝐗)으로 정의하고, 이러한 구성 요소들을 활용하여 GOOD 탐지에서의 샘플 분류 작업을 형식화합니다. 특히, 각기 다른 GOOD 탐지 접근법의 원리와 메커니즘에 대한 분석을 통해 그래프 데이터에 대한 고유한 도전 과제를 설명하고 있습니다.

- **Performance Highlights**: GOOD 탐지는 새로운 엔티티 발견과 안전성을 보장하는 데 중요한 역할을 합니다. 예를 들어, 신약 발견에서는 이전에 보지 못한 분자 구조를 탐지함으로써 잠재적인 치료 후보를 식별할 수 있습니다. 본 연구는 2020년부터 2025년까지 발표된 30개 이상의 대표적인 GOOD 탐지 방법을 포괄적으로 리뷰하고, 기존의 그래프 관련 탐지 분야와의 차별성을 명확히 하여 향후 연구 방향을 제시합니다.



### Rethinking Tokenized Graph Transformers for Node Classification (https://arxiv.org/abs/2502.08101)
Comments:
          Preprint version

- **What's New**: 이 논문에서는 기존의 Node tokenized graph Transformers (GTs)의 한계를 극복하기 위해 SwapGT라는 새로운 방법을 제안합니다. SwapGT는 token swapping이라는 새로운 작업을 도입하여 노드 간의 의미적 관련성을 활용하여 보다 다양한 token sequences를 생성합니다. 이 방식은 기존의 tokenized GTs가 인접 노드에만 초점을 맞춘다는 문제를 해결하고, 보다 정보가 풍부한 노드 표현을 학습하도록 돕습니다.

- **Technical Details**: SwapGT는 두 가지 주요 구성 요소로 이루어져 있습니다. 첫째, token swapping 작업을 도입하여 k-NN 그래프의 노드를 활용하여 여러 token sequences 간에 토큰을 교환함으로써 더 다양한 token sequences를 생성합니다. 둘째, Transformer 기반의 backbone을 사용하여 생성된 token sequences에서 노드 표현을 학습하고, center alignment loss를 통해 대표성 학습을 최적화합니다.

- **Performance Highlights**: 다양한 데이터셋에 대한 extensive empirical 결과를 통해 SwapGT가 노드 분류에 대해 기존의 방법보다 우수한 성능을 보임을 입증합니다. 특히, SwapGT는 복잡한 그래프 정보를 포착하는 데 효과적이며, 일반화 능력을 향상시키기 위해 다수의 token sequences를 활용합니다. 이는 희소한 학습 데이터에서도 효과적인 노드 표현 학습을 지원합니다.



### Unsupervised categorization of similarity measures (https://arxiv.org/abs/2502.08098)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2306.00239

- **What's New**: 이번 연구에서는 인공지능 신경망 시스템이 객체의 특징에 기반하여 독립적인 메트릭 공간을 자율적으로 범주화할 수 있는 방법을 제시합니다. 기능적인 차별화를 통한 유사성 판단에 대한 신경망의 역할이 중요하며, 기능 공간의 대칭성에 초점을 맞춘 새로운 방법론을 개발했습니다. 이러한 접근은 기존의 제한된 방법과는 다른 독립적인 메트릭 공간의 범주화를 가능하게 합니다.

- **Technical Details**: 연구팀은 카테고리 이론에서의 대수적 독립성을 기반으로 한 변환 간의 독립성을 개발했습니다. 이는 패턴 인식 이론과 대표 이론에서 자주 사용되는 불변 변환 방정식과 연결되어 있습니다. 다차원 감각 정보를 처리하기 위해 상이한 특성 공간 사이의 관계를 분리 및 평가할 수 있는 새로운 수학적 프레임워크를 제안합니다.

- **Performance Highlights**: 제안된 방법은 메트릭 공간의 고차원 독립성을 효율적으로 캡처합니다. 이 연구는 신경망 시스템이 어떻게 다양한 감각 특징인 색상 공간과 형태 공간 사이의 차이와 유사성을 평가하는지를 보여주며, 이는 다양한 머신러닝 및 딥러닝 응용 분야에 기여할 수 있습니다. 이러한 결과는 향후 비지도 학습 분야의 발전에 기여할 것으로 기대됩니다.



### Mixture of Decoupled Message Passing Experts with Entropy Constraint for General Node Classification (https://arxiv.org/abs/2502.08083)
Comments:
          arXiv admin note: text overlap with arXiv:2412.08193

- **What's New**: 이 논문은 다양한 그래프에서 발생하는 동질성(homophily)과 이질성(heterophily)의 정도가 그래프 신경망(GNNs)의 노드 분류 범용성에 미치는 영향을 다룹니다. 저자들은 이 문제를 해결하기 위해 Mixture-of-Experts (MoE) 메커니즘을 기반으로 한 GNNMoE라는 보편적인 노드 분류 프레임워크를 제안합니다. 이 프레임워크는 다양한 메시지 전파 전문가를 구성하고, 각 노드의 표현 학습에 가장 적합한 전문가 네트워크를 할당하도록 설계되었습니다.

- **Technical Details**: GNNMoE는 P(전파)와 T(변환) 연산을 탈중앙화하여 네 가지 유형의 메시지 전파 전문가를 구성합니다. 연구진은 또한 각 노드가 표현 학습을 위해 적합한 인코딩 전략을 자율적으로 선택할 수 있도록 소프트 및 하드 게이팅 메커니즘을 설계했습니다. 이러한 접근 방식은 GNNMoE가 동질적 및 이질적 그래프 모두에서 효과적으로 작동할 수 있도록 합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋에 대한 실험 결과, GNNMoE는 주요 GNN 모델, 이질적 GNN 및 그래프 트랜스포머보다 노드 분류 성능과 범용성 측면에서 현저히 우수한 성과를 나타냈습니다. 이는 GNNMoE의 구조가 다양한 그래프 데이터에서의 노드 분류 작업에 강한 일반화 능력을 부여함을 보여줍니다.



### Cascading Bandits Robust to Adversarial Corruptions (https://arxiv.org/abs/2502.08077)
- **What's New**: 이 논문에서는 적대적 공격에 의해 손상되지 않는 강력한 알고리즘을 제안하여, online learning to rank에 대한 새로운 접근 방식을 소개합니다. 'Cascading Bandits with Adversarial Corruptions'(CBAC) 문제로 적대적인 피드백 조작자도 고려합니다. 본 연구에서 제안된 두 가지 알고리즘(CascadeRKC 및 CascadeRAC)은 각각 손상 수준이 알려진 경우와 알려지지 않은 경우에 대해 최고의 성과를 올립니다.

- **Technical Details**: Cascading bandits는 추천 시스템에서 사용자의 클릭 행동을 기반으로 순위를 매기는 온라인 알고리즘입니다. 일반적으로 각 아이템은 감춰진 매력 확률(attraction probability)과 연결되어 있으며, 이 알고리즘은 추천 리스트 내의 아이템 클릭 인덱스를 관찰하여 보상을 계산합니다. 본 논문에서 제안하는 알고리즘은 새로운 position-based elimination(PBE) 알고리즘을 기반으로 하며, 이를 통해 손상 레벨에 따라 성능을 조절할 수 있습니다.

- **Performance Highlights**: 제안된 두 알고리즘은 공격을 받지 않을 때 로그형 후회(logarithmic regret)를 달성할 수 있으며, 손상 수준이 증가함에 따라 후회는 선형적으로 증가합니다. 실험 결과는 다양한 데이터세트와 설정에서 두 알고리즘의 강건성을 검증하며,의 클릭 피드백 조작에 대처할 수 있는 새로운 신뢰성을 제공합니다.



### Cognify: Supercharging Gen-AI Workflows With Hierarchical Autotuning (https://arxiv.org/abs/2502.08056)
- **What's New**: 이 연구는 자동화된 gen-AI 워크플로우 최적화의 필요성을 강조하며, AdaSeek이라는 새로운 계층적 검색 알고리즘을 제안합니다. 기존의 수동적인 워크플로우 조정 방식이 가지는 비효율성을 해결하기 위해, 이 방식은 워크플로우의 다양한 요소를 구조화하여 자동으로 튜닝할 수 있도록 돕습니다. 특히, 특정 사용자 예산에 따라 조정 가능한 계층적 아키텍처를 도입하여, 각 단계를 최적화합니다.

- **Technical Details**: AdaSeek 알고리즘은 사용자가 정의한 총 검색 예산에 따라 서로 다른 검색 계층을 구분하고, 각 계층의 복잡성에 기반하여 예산을 분배합니다. 각 계층 내에서, 이 알고리즘은 TPE(턴 메소딕스 마켓 최적화)를 활용하여 소수의 구성들을 샘플링합니다. 이러한 체계적 구조는 워크플로우의 품질을 향상시키는 데 효과적이며, 전체 효율성을 개선합니다.

- **Performance Highlights**: Cognify 프레임워크를 통해 실시된 실험 결과, 워크플로우의 생성 품질이 최대 2.8배 향상되었고, 실행 비용은 최대 10배 감소했으며, 전체 실행 지연시간은 2.7배 줄어들었습니다. 또한, 기존의 DSPy와 Trace보다 최대 2.6배 높은 생성 품질을 보이면서, 비용과 지연 시간 또한 각각 10배 및 3배 개선되었습니다.



### The Art of Misclassification: Too Many Classes, Not Enough Points (https://arxiv.org/abs/2502.08041)
- **What's New**: 이 연구는 분류 문제의 본질적인 어려움을 정량화하는 엔트로피 기반(classificability) 측정을 도입합니다. 이 측정 방법은 특징 표현(feature representation)에 따라 클래스 할당(class assignment)에서의 불확실성을 평가하여 정량화합니다. 제안된 방법은 인지적 직관에 부합하고 분류 성능의 상한선을 제공하여, 아키텍처나 데이터 양에 관계없이 분류 정확도를 향상시킬 수 없는 이론적 한계를 설정합니다.

- **Technical Details**: 연구는 데이터셋과 분류 문제 사이의 차별성을 강조하며, 데이터셋을 분류 문제의 유한한 실현으로 정의합니다. 클래스는 확률 변수의 집합으로 정의되며, 클래스 할당 간의 중첩(overlap) 정도를 평가하여 분류 가능성을 측정합니다. 또한, 정보 이론적 관점을 수용하여 분류의 본질적 한계를 포착하는 방법을 제안하고, 이는 아키텍처나 데이터 양에 관계없이 적용 가능한 이론적 한계로 이어집니다.

- **Performance Highlights**: 제안된 엔트로피 기반 측정법은 전통적인 분류 성능 메트릭과 비교하여 더 깊은 통찰을 제공합니다. 이 접근법을 통해 복잡한 데이터셋에 대한 내재적 한계를 이해할 수 있으며, 실제 시나리오에서도 유용성을 제시합니다. 따라서 기존의 강력한 분류기 개발보다 분류 가능성을 이해하는 데 시간 투자가 필요함을 보여줍니다.



### Initialization Matters: Unraveling the Impact of Pre-Training on Federated Learning (https://arxiv.org/abs/2502.08024)
- **What's New**: 이 논문은 사전 훈련된 모델(pre-trained model)로 시작하는 것이 연합 학습(federated learning) 설정에서 테스트 성능에 미치는 부정적인 영향을 줄이는 데 효과적임을 강조합니다. 특히, 이 연구는 2층 합성곱 신경망(two-layer convolutional neural networks, CNNs)의 훈련 에러 수렴과 테스트 에러에 대한 경계를 제공합니다.

- **Technical Details**: 저자들은 초기화 시 정렬(aligned) 및 비정렬(misaligned) 필터의 개념을 도입하여 데이터 이질성(data heterogeneity)이 비정렬 필터에서만 학습에 영향을 미친다고 주장합니다. 또한, 사전 훈련된 모델로 시작하면 일반적으로 초기화 시 비정렬 필터가 적어져, 연합 설정에서 훈련하더라도 낮은 테스트 에러(test error)가 나타난다고 설명합니다.

- **Performance Highlights**: 연구는 합성 데이터(synthetic data) 환경과 실제 연합 학습 환경에서 CNN의 성능을 실험하여 이론적 발견을 검증하였습니다. 이러한 결과는 사전 훈련된 모델의 사용이 연합 학습에서 실제 훈련 효율성을 크게 향상시킬 수 있음을 보여줍니다.



### Model Selection for Off-policy Evaluation: New Algorithms and Experimental Protoco (https://arxiv.org/abs/2502.08021)
- **What's New**: 이번 연구에서는 Offline Reinforcement Learning (RL)에서 Hyperparameter tuning의 필요성에 대해 다룬다. 특히, Off-Policy Evaluation (OPE) 알고리즘을 통해 정책 평가를 수행할 때 발생하는 문제점들을 분석하고, 이러한 알고리즘 내에서 하이퍼파라미터 조정의 중요성을 강조한다. 우리는 새로운 'LSTD-Tournament' 선택기를 제안하고 이를 실험 protocol에 적용하여 기초 성과를 검증하였다.

- **Technical Details**: 연구의 주요 초점은 두 가지 설정에 따라 후보 가치 함수(model-free) 또는 다이나믹스 모델(model-based) 중에서 선택하는 것이다. 이 연구에서는 후보 함수가 Bellman 방정식을 준수하는지를 근사적으로 확인하며, 이를 통해 Double Sampling 문제를 해결한다. 또한, 견고한 이론적 보장을 제공하는 새로운 선택 알고리즘을 개발하여 모델 기반 RL에서도 효과적으로 활용할 수 있는 방안을 제시한다.

- **Performance Highlights**: Gym 환경에서 LSTD-Tournament의 성능을 평가한 결과, 후보 가치 함수의 안정적인 생성을 가능하게 하고, 미스스펙ification에 대한 더 나은 조절을 보여주었다. 새로운 실험 프로토콜은 기지 기반 환경에서의 변화를 통해 후보 Q-값을 생성하고, 최적화 없이도 Q-값을 계산할 수 있도록 지원하여 효율성을 높였다. 이러한 결과는 제안된 알고리즘이 기존 방법들보다 더 나은 성과를 제공할 가능성을 시사한다.



### An Interactive Framework for Implementing Privacy-Preserving Federated Learning: Experiments on Large Language Models (https://arxiv.org/abs/2502.08008)
- **What's New**: 이번 연구는 Federated Learning (FL) 방법론에서 Differential Privacy (DP) 기법의 메모리 요구 사항 문제를 처음으로 다루었습니다. 기존의 DP 방법은 메모리 사용에 있어 가변적인 요구 사항을 가지는데, 이는 자원 제한이 있는 디바이스에서 FL을 조화롭게 운영하는 데에 까다로움을 초래했습니다. 우리가 제안한 프레임워크는 개인화된 프라이버시 관리자가 참여하여 모델의 프라이버시와 유용성 간의 최적의 균형을 찾도록 돕습니다.

- **Technical Details**: 우리는 고정된 메모리 사용량을 보장하는 최신 DP 기술을 채택하여 안정적인 Private FL을 지원하고 있습니다. GLUE 데이터셋을 사용하여 BERT 기반의 LLM 모델을 미세 조정함으로써, 새로운 회계 자격증을 활용하고 다양한 데이터 분할 전략을 사용하여 실제 환경을 모사하였습니다. 이 과정에서, 평균적으로 $
abla=10$의 경우 1.33%, $
abla=6$의 경우 1.9%의 정확도 감소를 보였습니다.

- **Performance Highlights**: 제안한 프레임워크는 고정된 메모리 사용을 유지함으로써, 메모리 사용량의 안정성을 달성하였습니다. 이는 다양한 사용자 그룹의 참여를 보장하고, 궁극적으로는 보다 공정한 모델 학습에 기여합니다. 우리가 제안한 아키텍처는 오픈 소스로 제공되어, 공공 검증 및 테스트가 가능합니다.



### The Role of Randomness in Stability (https://arxiv.org/abs/2502.08007)
- **What's New**: 이 논문은 학습 및 통계에서 알고리즘의 출력이 유사한 데이터 세트에서 얼마나 안정성을 가지는지를 연구합니다. 특히, 알고리즘의 안정성과 관련하여 필요한 무작위성(randomness)의 양을 정량화하려고 시도합니다. 주요한 두 가지 개념인 replicability(재현성)와 differential privacy(차등 프라이버시)의 무작위 복잡성(randomness complexity)을 조사하며, 안정성에 대한 '약한-강한' 부스팅 정리를 증명합니다.

- **Technical Details**: 이 연구는 알고리즘의 무작위성 복잡성이 deterministic algorithm(결정론적 알고리즘)의 최적 재현 확률에 의해 제어된다는 것을 보여줍니다. 특히, PAC Learning에서 클래스가 유한한 Littlestone 차원을 가질 때 한정된 무작위 복잡성을 갖는다는 것을 밝혀냈습니다. 이는 global stability(전역 안정성)의 개념과 연결되며, 최종적으로 특정 에러 기준에 대한 의존성을 줄입니다.

- **Performance Highlights**: 저자들은 binary classification(이진 분류)와 관련된 주요 작업에서 무작위성 복잡성을 분석하며, PAC 모델을 집중적으로 다룹니다. 결과적으로, 클래스 H가 유한한 Littlestone 차원을 가지면 bounded error-dependent global stability(유한한 에러 의존 전역 안정성)를 가지며, 무작위성 복잡성은 O(log 1/α)로 제한된다는 것을 보였습니다. 이는 다양한 학습 알고리즘의 성능에 중요한 기준을 제공합니다.



### Greed is Good: Guided Generation from a Greedy Perspectiv (https://arxiv.org/abs/2502.08006)
Comments:
          Initial preprint

- **What's New**: 이 논문에서는 트레이닝이 필요 없는 가이드 생성(Training-free guided generation) 방법론을 새로운 관점에서 탐구합니다. 특히, 신경 미분 방정식(Neural differential equation)의 해 경로(solution trajectory)를 최적화하는 탐욕적(greedy) 방법을 제안합니다. 이러한 방법은 기존의 교육 없는 가이드를 통합적으로 이해하는 데 기여합니다.

- **Technical Details**: 제안된 탐욕적 방법은 엔드 투 엔드 최적화(End-to-end optimization) 기술의 일차 차분(discretization)으로 설명됩니다. 이 방법은 최적의 경과를 이끌어내고, 연속적 상보 방정식(Continuous adjoint equations)을 통해 발견된 이상적 경량과 비교하여 우수한 가이드 결정을 내리는 것으로 나타났습니다. 이 연구는 다양한 교육 없는 가이드 전략을 통합적으로 이해할 수 있는 새로운 시각을 제공합니다.

- **Performance Highlights**: 탐욕적 가이드 전략은 결정적 직관을 제공하며, 기존의 다른 인기 있는 전이 없는 가이드를 효과적으로 대체할 수 있습니다. 이러한 접근 방식은 복잡한 생성 모델의 제어를 사용자에게 보다 쉽게 할 수 있는 방법으로 자리 잡을 것으로 기대됩니다. 결과적으로, 이 연구는 생성 모델의 성능 향상에 대한 새로운 통찰을 제시합니다.



### Towards Training One-Step Diffusion Models Without Distillation (https://arxiv.org/abs/2502.08005)
Comments:
          13 pages, Technical Report

- **What's New**: 최근 연구는 전통적으로 두 단계로 진행되는 One-Step Generative Models의 훈련 프로세스를 다루고 있으며, 이는 Teacher Diffusion Model을 훈련한 후 이를 One-Step Student Model로 증류(distill)하는 과정을 포함합니다. 본 논문에서는 이 증류(nurturing) 과정 없이 One-Step Generative Models를 직접 훈련할 수 있는지를 조사했습니다. 특히, Teacher의 Score Function(점수 함수)이 필수적이지 않다는 것을 보여주고, Score 추정 없이도 경쟁력 있는 결과를 도출할 수 있는 증류 방법을 제안합니다.

- **Technical Details**: 이 논문은 One-Step 모델을 효과적으로 훈련하기 위해 Teacher 모델 없이 진행할 수 있는 방법을 모색합니다. 연구 결과에 따르면, Teacher의 Score Function은 근본적인 요소가 아니지만, Teacher Weight(가중치)로 초기화하는 것이 성공적인 훈련에 필수적이라는 점이 밝혀졌습니다. 또한, 이 초기화가 'Input-Output Mapping' 개선 때문이 아닌, 학습된 Feature Representation(특징 표현)에서 기인한 것임을 발견했습니다.

- **Performance Highlights**: 연구팀은 Teacher Weight에서 얻는 이점이 Distillation Quality(증류 품질)에 미치는 영향을 효율적으로 분석함으로써 One-Step 모델 훈련의 초기화(Initialization) 역할을 개선된다는 중요한 통찰을 제공합니다. 이는 One-Step Generative Models의 효율성과 성능을 더욱 향상시킬 수 있는 방법으로, 다양한 생성 모델 분야에 영향력을 미칠 수 있는 연구 결과입니다.



### Heterogeneous Multi-agent Multi-armed Bandits on Stochastic Block Models (https://arxiv.org/abs/2502.08003)
Comments:
          55 pages

- **What's New**: 본 연구에서는 스토캐스틱 블록 모델에 의해 유도된 클러스터 구조를 가진 새로운 이종 다중 요인 다중 무기 문제를 연구합니다. 이 문제는 그래프의 토폴로지뿐만 아니라 보상 이질성에도 영향을 미칩니다. 특히, 에이전트는 이질적인 엣지 확률을 가진 임의 그래프에 분포되어 있으며, 같은 클러스터 내의 엣지 확률은 다르지만 클러스터 간에는 상이합니다. 보상 분포 또한 클러스터에 따라 차별화됩니다.

- **Technical Details**: 다양한 클러스터 구조를 가진 이 연구는 자신의 보상 분포와 관련하여 최적의 팔을 식별하는 것을 목표로 합니다. 저자들은 정보 집계 및 가중치 기법을 결합하여 UCB(Upper Confidence Bound) 유형의 전략을 제안합니다. 이 알고리즘은 그래프의 무작위성을 고려하며, 보상 및 그래프에서 얻은 이질적 정보를 활용하여 알려지지 않은 클러스터 설정에서도 클러스터 탐지를 포함합니다.

- **Performance Highlights**: 본 알고리즘은 서브가우시안 보상 하에서 최적의 인스턴스 의존적 레그렛 상한을 도출하며, 이는 로그 시간 T의 차원으로 나타납니다. 기존 연구는 복잡성을 충분히 반영하지 못하는 반면, 본 연구는 간소화된 가정을 놓고 더 큰 시스템에서도 효과적으로 확장 가능합니다. 이 연구는 이종 설정에서의 클러스터 구조와 보상 동학의 관계를 해결할 수 있는 가능성을 보여줍니다.



### Adaptive kernel predictors from feature-learning infinite limits of neural networks (https://arxiv.org/abs/2502.07998)
- **What's New**: 이번 연구에서는 무한 너비의 신경망이 데이터 의존 커널을 가진 커널 머신으로 설명될 수 있음을 보여줍니다. 이는 두 가지 서로 다른 설정에서 이루어졌으며, 여기서 각 예측기는 명시적 표현식으로 제시되고 수치적으로 계산하는 방법이 제시됩니다. 연구자들은 이 작업을 통해 신경망의 'rich' feature learning 영역에서도 비모수적 커널 예측기를 찾기 위한 가능성을 탐구하고 있습니다.

- **Technical Details**: 연구에서는 큰 너비의 feature-learning Bayesian 네트워크와 적응형 Neural Bayesian Kernel (aNBK) 및 Neural Tangent Kernel (aNTK)의 관련성을 분석합니다. 이를 위해 다이나믹 평균장 이론(DMFT)을 활용하여 기울기 흐름(gradiant flow) 훈련의 동역학을 연구합니다. 노이즈가 추가된 기울기 흐름의 동역학을 이해함으로써, 무한 너비에서 최종 네트워크 예측기가 커널 예측기로 작동함을 입증합니다.

- **Performance Highlights**: 제안된 적응형 커널 머신은 다양한 벤치마크 데이터셋에서 테스트 손실이 더 낮으며, 최근 훈련된 신경망과 lazy NTK 및 NNGPK 예측기를 비교했을 때 뛰어난 성과를 보여주었습니다. 이러한 커널은 feature-learning 신경망 훈련을 잘 설명하며, 훈련 중의 내부 표현 변화를 반영합니다. 결과적으로 이번 연구는 무한 너비 신경망의 학습 원리와 그 이론적 기초를 발전시키는 데 기여합니다.



### Learning Effective Dynamics across Spatio-Temporal Scales of Complex Flows (https://arxiv.org/abs/2502.07990)
Comments:
          Conference on Parsimony and Learning (CPAL)

- **What's New**: 본 논문에서는 복잡한 유체 흐름을 다루기 위해 Graph-based Learning of Effective Dynamics (Graph-LED)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 그래프 신경망(Graph Neural Networks, GNN)과 주의 기반 자체 회귀 모델(autoregressive model)을 결합하여 소량의 시뮬레이션 데이터에서 효과적인 동역학을 추출합니다. GNN은 비구조화 메쉬를 그래프로 표현하여 복잡한 기하학과 비균일 그리드를 효과적으로 처리할 수 있습니다.

- **Technical Details**: Graph-LED 프레임워크는 두 가지 주요 요소로 구성됩니다: 공간 차원 축소와 시간 예측입니다. 공간 차원 축소는 GNN 인코더-디코더를 사용하여 이루어지며, 여러 GNN 레이어를 통해 해의 지역 정보를 노드 수준의 표현으로 집계합니다. 시간 예측은 주의 메커니즘을 사용하여 동적 학습과 예측을 일관된 자기 회귀 방식으로 수행합니다.

- **Performance Highlights**: 제안된 방법은 다양한 레이놀즈 수(Reynolds numbers)를 가진 유체 역학 문제에 대해 평가되었습니다. 실험 결과, 원주에서의 흐름과 backward-facing step의 흐름에서 복잡한 다스케일 흐름 특성을 효과적으로 다루며, 주위의 작은 규모 영향을 정확하게 포착하는 것으로 나타났습니다.



### CIRCUIT: A Benchmark for Circuit Interpretation and Reasoning Capabilities of LLMs (https://arxiv.org/abs/2502.07980)
- **What's New**: 본 논문은 아날로그 회로 설계에서 Large Language Models(LLMs)의 역할을 탐구하고, 회로에 대한 LLM의 추론 능력을 평가하기 위한 CIRCUIT 데이터셋을 생성했습니다. 이 데이터셋은 단순 회로 토폴로지 이해에 중점을 두고 있으며, 아날로그 회로와 관련된 510개의 질문-답변 쌍으로 구성되어 있습니다. 성능 테스트 결과, GPT-4o 모델은 최종 수치 답변에 대해 48.04%의 정확도를 기록하며, 이는 LLM들이 회로 이해에 여전히 어려움을 겪고 있음을 보여줍니다.

- **Technical Details**: CIRCUIT 데이터셋은 102개의 템플릿에서 파생된 510개의 문제로 구성되어 있으며, 각 템플릿당 5개의 수치 설정이 포함됩니다. 데이터셋은 다양한 아날로그 회로 문제를 해결하는 데 필요한 기초/중요한 회로 지식과 추론 능력을 평가하기 위한 장치로 설계되었습니다. 이 연구에서는 특히 안정성 평가를 위해 질문을 유닛 테스트 형태로 그룹화하여 LLM의 강인성을 확인했습니다.

- **Performance Highlights**: 데이터셋 분석 결과, LLM인 GPT-4o는 유닛 테스트에서 27.45%의 통과율을 나타내어 기존 LLM의 한계를 드러냅니다. 이는 아날로그 회로 설계에서 필요한 다단계 추론을 이해하는 것이 얼마나 도전적인지를 강조하며, LLM의 향후 발전 방향과 응용 가능성을 높이는 인사이트를 제공합니다. 또한, 본 연구는 아날로그 통합 회로 설계에서의 LLM 활용을 혁신적으로 발전시킬 수 있는 귀중한 기초 자료로 기능할 수 있습니다.



### A Survey of In-Context Reinforcement Learning (https://arxiv.org/abs/2502.07978)
- **What's New**: 본 논문은 인맥균형강화학습(In-Context Reinforcement Learning, ICRL)에 대한 연구를 다루고 있습니다. ICRL은 네트워크 파라미터를 업데이트하지 않고도 추가적인 컨텍스트에 조건을 두어 새로운 작업을 해결할 수 있는 능력을 가진 강화학습 에이전트에 관한 것입니다. 이러한 접근 방식은 에이전트가 유사한 새로운 작업에 더욱 효율적으로 적응할 수 있도록 돕습니다.

- **Technical Details**: ICRL은 에이전트의 정책이 환경과 상호작용하는 동안 보상 신호를 극대화하도록 점진적으로 최적화되는 전통적인 강화학습(RL) 모델과 달리 작동합니다. 에이전트는 현재 관찰(observation)과 작업 관련 정보를 포함한 컨텍스트(context)를 입력으로 받아 새로운 상황에 적응합니다. 이 과정에서 공통적으로 사용되는 마르코프 의사결정 과정(MDP)의 구조는 여전히 적용됩니다.

- **Performance Highlights**: ICRL의 이점은 파라미터 업데이트 없이도 성능이 향상되며, 이는 'in-context improvement'라는 현상으로 설명됩니다. 연구에 따르면, ICRL을 적용한 에이전트는 이전시키는 RL 알고리즘보다 샘플 효율성(sample efficiency)이 더 높을 수 있는 가능성이 제기되고 있습니다. 이러한 점은 추론(inference) 과정에서 메모리와 계산 요구 사항을 최적화할 수 있는 새로운 기회를 제공합니다.



### RESIST: Resilient Decentralized Learning Using Consensus Gradient Descen (https://arxiv.org/abs/2502.07977)
Comments:
          preprint of a journal paper; 100 pages and 17 figures

- **What's New**: 최근 데이터 수집 및 처리를 위한 네트워크 연결 장치의 필요성이 증가하면서, 비중앙화(Decentralized) 머신러닝(Machine Learning) 알고리즘의 개발이 필수적입니다. 본 논문은 MITM(Man-in-the-Middle) 공격에 집중하여 이를 저지하기 위한 새로운 최적화 알고리즘인 RESIST를 제안합니다. 이 알고리즘은 비공식적으로 연결된 통신 링크에 대한 저항력이 뛰어난 구조를 가지고 있어, 일반적인 ERM(경험적 위험 최소화) 문제를 해결하는 데 사용됩니다.

- **Technical Details**: RESIST는 강하게 볼록한(Strongly Convex), 폴리아크-로야시비츠(Polyak-Lojasiewicz), 비볼록(Nonconvex) ERM 문제에서 알고리즘 및 통계적 수렴(Convergence)을 달성합니다. 이 알고리즘은 일관성 있는 그래디언트 강하법(Consensus Gradient Descent)을 기반으로 하며, 비대칭적인 네트워크 환경에서도 효과적으로 작동하도록 설계되었습니다. 이를 통해 하드웨어 및 메모리 제약이 심한 환경에서도 효율적인 학습이 가능합니다.

- **Performance Highlights**: 실험 결과를 통해 RESIST는 적대적 환경에서도 비중앙화 학습의 견고성과 확장성을 제공함을 보여줍니다. RESIST는 기존 솔루션들에 비해 더 높은 보안성을 유지하며, 다양한 ERM 문제에 적용할 수 있는 가능성을 제시합니다. 이러한 성능은 네트워크 보안이 중요한 상황에서의 머신러닝 적용에 큰 기여를 할 것입니다.



### Generative Risk Minimization for Out-of-Distribution Generalization on Graphs (https://arxiv.org/abs/2502.07968)
Comments:
          TMLR 02/2025

- **What's New**: 이 논문은 그래프 기반 OOD(Out-of-Distribution) 일반화를 위한 Generative Risk Minimization (GRM) 프레임워크를 제안합니다. GRM은 입력 그래프에 대해 불변(subgraph invariant) 정보를 효과적으로 생성하여 기존의 이산적(extraction) 구조 환원에서 발생할 수 있는 정보 손실 문제를 해결합니다. 또한, 이 프레임워크는 변량 근사(variational approximation) 및 잠재 인과 변수(latent causal variable)를 사용하여 최적화 구조를 개선합니다.

- **Technical Details**: GRM 프레임워크는 그래프 G=(𝒱,ℰ,𝐗)로 정의되며, 여기서 𝒱는 노드 집합, ℰ는 엣지 집합, 𝐗는 특성 행렬(feature matrix)로 구성됩니다. GRM의 주요 목표는 두 가지입니다: 첫째, 연속적(edge weights, node representations) 형태로 정확한 불변 서브그래프를 생성하는 것, 둘째, 불변 서브그래프와 배경(domain) 사이의 독립성을 보장하는 것입니다. 이를 위해 GRM은 세 가지 상관된 손실(loss)을 최적화하여 정보 손실을 최소화하고 왜곡된 정보(spurious information)를 최소화하는 데 집중합니다.

- **Performance Highlights**: 다양한 실제 그래프 데이터셋을 활용한 실험을 통해, GRM은 기존의 최신 기법들과 비교할 때의 우수성을 입증하였습니다. 특히, GRM은 OOD 일반화 문제에서 불변 정보를 최대한 활용하면서 왜곡된 정보를 최소화하는 능력을 보유하고 있습니다. 이로써 OOD 그래프 데이터에 대한 일반화 성능을 크게 향상시킬 수 있는 가능성을 보여줍니다.



### New tools for comparing classical and neural ODE models for tumor growth (https://arxiv.org/abs/2502.07964)
Comments:
          9 pages, 2 figures. Related software is archived at this https URL

- **What's New**: 이번 논문에서는 종양 성장 모델링을 위한 새로운 계산 도구인 TumorGrowth.jl을 소개합니다. 이 도구는 General Bertalanffy 모델 및 Gompertz 모델과 같은 전통적인 모델과 함께 신경 ODE(neural ODE) 모델과 같은 최신 모델을 비교할 수 있습니다. 이를 통해 비소세포폐암(non-small cell lung cancer) 및 방광암(bladder cancer) 병변에 대한 인간 메타 연구를 조사하여, 기존 결과의 통계적 유의성을 분석하고 더 복잡한 모델의 성능을 평가합니다.

- **Technical Details**: TumorGrowth.jl은 종양 성장 모델을 보정 및 비교하기 위한 도구로, 주로 부피(volumn)에 의해 측정됩니다. 이 패키지는 General Bertalanffy 모델 및 Gompertz 모델, 신경 ODE 모델 등을 포함하여 다양한 모델을 지원하며, 맞춤형 모델도 구현할 수 있습니다. 특히 간단한 2D General Bertalanffy 모델과 14개의 매개변수를 가진 2D 신경 ODE 모델을 통해 재발 및 반등 현상을 포착할 수 있습니다.

- **Performance Highlights**: 주요 결론은 General Bertalanffy 모델이 평균적으로 우수한 성능을 보인다는 점입니다. 그러나 더 많은 측정값이 있는 경우 더 복잡한 모델이 더 나은 예측 성능을 나타낼 수 있음을 논의합니다. 실험 결과는 General Bertalanffy 모델이 다른 모델들에 비해 우수하지만, 1D 신경 ODE 모델과의 차이는 통계적으로 유의미하지 않음을 보여주었습니다.



### ESPFormer: Doubly-Stochastic Attention with Expected Sliced Transport Plans (https://arxiv.org/abs/2502.07962)
- **What's New**: 이 논문에서는 기존의 반복적인 Sinkhorn 정규화 방법 대신에 전적으로 병렬화 가능한 새로운 doubly-stochastic attention 메커니즘인 ESPFormer를 제안합니다. 이 방법은 Expected Sliced Transport Plans (ESP)를 활용하여 두 토큰 간의 attention 분포의 균형을 보장하며, 효율성을 크게 향상합니다. 우리의 메커니즘은 차별화를 유지하기 위해 온도 기반의 소프트 정렬 기법을 통합하여 딥러닝 모델에 원활하게 통합될 수 있도록 합니다.

- **Technical Details**: Transformer 아키텍처의 중요한 구성 요소인 self-attention 메커니즘은 동적인 쌍별 상호작용을 가능하게 합니다. 본 논문에서는 sliced optimal transport 이론에 기반하여, 새로운 doubly-stochastic attention 메커니즘을 통해 attention 분포의 균형을 유지할 수 있습니다. 기존의 방법들과는 달리, 새로운 메커니즘은 반복적인 Sinkhorn 정규화 없이도 double stochasticity를 보장하여 계산 효율성을 높입니다.

- **Performance Highlights**: 여러 벤치마크 데이터셋에서 진행된 실험 결과, 이미지 분류, 점 구름 분류, 감정 분석 및 신경 기계 번역을 포함한 다양한 응용 프로그램에서 성능 향상이 확인되었습니다. ESPFormer를 기존의 Transformer 아키텍처에 통합하고 몇 에폭 동안 미세 조정하면 상당한 성능 향상을 달성할 수 있습니다. 또한, 최근 도입된 미분 가능한 attention 아키텍처와의 호환성도 입증되었습니다.



### VSC-RL: Advancing Autonomous Vision-Language Agents with Variational Subgoal-Conditioned Reinforcement Learning (https://arxiv.org/abs/2502.07949)
- **What's New**: 이 논문에서는 Variational Subgoal-Conditioned RL (VSC-RL)이라는 새로운 강화 학습 방법을 소개합니다. VSC-RL은 비전-언어 (vision-language) 연속 의사결정 문제를 변분 목표 조건 강화 학습 문제로 재구성하여 학습 효율성을 향상시킵니다. 이 방법은 복잡한 목표를 자율적으로 이행 가능한 서브골(subgoal)로 분해하여 실제 세계의 복잡한 문제를 해결하는 데 큰 잠재력을 보여줍니다.

- **Technical Details**: VSC-RL은 SubGoal Evidence Lower BOund (SGC-ELBO)를 최적화하며, 이 과정에서 서브골 조건의 보상을 최대화하고 참조 정책과의 차이를 최소화하는 방식으로 진행됩니다. 이 논문은 SGC-ELBO가 원래의 최적화 목표와 동등하다는 것을 이론적으로 입증하며, 이를 통해 성과 보장을 유지하면서 학습 효율성을 개선합니다. 학습의 효율성을 증가시키는 과정에서 비전-언어 모델의 강력한 추론 및 계획 능력을 활용합니다.

- **Performance Highlights**: 다양한 벤치마크에서 실험을 실시한 결과, VSC-RL은 기존의 SOTA 비전-언어 에이전트들보다 성능과 학습 효율성 모두에서 월등한 결과를 보여줍니다. 구체적으로, 이 방법은 복잡한 모바일 장치 제어 과제에서도 뛰어난 성능을 발휘하며, 기존의 방법들과 비교할 때 매우 우수한 개선 효과를 보입니다. 임상 결과는 VSC-RL이 여러 평가 벤치마크에서 기존 SOTA와 비교해 뛰어난 성능을 도출할 수 있음을 강력하게 뒷받침합니다.



### Active Advantage-Aligned Online Reinforcement Learning with Offline Data (https://arxiv.org/abs/2502.07937)
- **What's New**: 이번 연구에서는 Online reinforcement learning (RL)과 Offline RL의 장점을 결합하기 위해 새로운 방법인 A3 RL을 제안했습니다. 이 방법은 다양한 온라인 및 오프라인 데이터 출처에서 선택적으로 데이터를 선택하여 정책 개선을 최적화합니다. 이전 방식들에서 자주 발생하는 문제인 catastrophic forgetting, 견고성 부족 및 샘플 효율성 문제를 해결하고자 합니다.

- **Technical Details**: A3 RL 방법론은 활성 샘플링(active sampling) 전략을 사용하여 효율적인 데이터 조합을 통해 정책 개선을 이루는 것을 목표로 합니다. 이를 통해 오프라인 데이터에 의존하는 기존의 온라인 RL 기술보다 더 나은 성능을 제공할 수 있습니다. 연구자는 이론적 보장을 통해 방법의 효과성을 검증했습니다.

- **Performance Highlights**: 실험 결과, A3 RL은 기존 최첨단 온라인 RL 기술보다 뛰어난 성능을 보였습니다. 이는 다양한 환경에서 강화 학습을 적용할 때 보다 높은 샘플 효율성(sample efficiency)과 정책 개선을 가능하게 합니다. 공개된 코드로 더 많은 연구자들이 이 방법을 활용할 수 있도록 할 계획입니다.



### TransMLA: Multi-Head Latent Attention Is All You Need (https://arxiv.org/abs/2502.07864)
Comments:
this https URL

- **What's New**: 이번 논문에서는 Multi-head Latent Attention (MLA) 접근 방식을 통해 고급 대형 언어 모델(Large Language Models, LLMs)의 통신 병목 현상을 해결하는 방법을 제안합니다. MLA는 key-value (KV) 레이어에서 저랭크 매트릭스를 사용하여 메모리 요구 사항을 감소시키고, 이를 통해 빠른 추론 속도를 구현합니다. 또한, TransMLA라는 후속 학습 방법을 도입하여 일반적으로 사용되는 GQA 기반의 사전 학습 모델을 MLA 기반 모델로 전환할 수 있도록 하였습니다.

- **Technical Details**: MLA는 전통적인 multi-head attention 방식보다 우수한 표현 능력을 제공하면서도 KV 캐시의 오버헤드를 동일하게 유지할 수 있는 이론적 근거를 제공합니다. 기존 모델에서 KV를 캐시하는 방식이 메모리와 통신에서의 병목 현상을 야기하는 문제를 강조하며, 모델을 MLA로 전환할 경우 저렴한 비용으로 성능 향상을 이루는 가능성을 제시합니다. 이 연구는 GQA 기반의 유명 모델들을 MLA 모델로 변환하고, 이를 통해 다운스트림 작업의 성능이 향상됨을 입증합니다.

- **Performance Highlights**: MLA 모델은 GQA 대비 더 나은 표현력과 성능을 제공하며, 이를 통해 기존 LLM의 가능성을 극대화할 수 있습니다. 최적의 KV 캐시 크기를 유지하면서도 빠른 추론이 가능한 구조를 통해 리소스 소비와 탄소 배출을 줄이는 효과를 기대할 수 있습니다. 이러한 접근은 DNN(Deep Neural Networks) 분야의 추후 연구 및 실제 응용에 중요한 기여를 할 것입니다.



### ADMN: A Layer-Wise Adaptive Multimodal Network for Dynamic Input Noise and Compute Resources (https://arxiv.org/abs/2502.07862)
- **What's New**: 이 논문에서는 ADMN(Adaptive Depth Multimodal Network)을 제안하며, 이는 동적 계산 자원과 입력 모달리티의 질(QoI) 변동성에 적응할 수 있는 능력을 갖춘 모델입니다. 기존의 멀티모달 네트워크는 고정된 자원 할당 방식을 사용하여 QoI에 따른 변동성을 고려하지 않아요. ADMN은 모든 모달리티에서 활성화된 레이어 수를 조정하여 이러한 문제를 해결하고, 낮은 QoI를 가진 모달리티의 리소스를 다른 모달리티에 재분배합니다.

- **Technical Details**: ADMN은 각 모달리티의 QoI에 따라 레이어를 조정할 수 있는 적응형 백본을 사용합니다. 기본적으로, ADMN은 미리 훈련된 가중치로 초기화된 대형 백본을 사용하여 레이어 드롭 기술(LayerDrop)을 통해 각 모달리티에 대해 무작위로 레이어를 드랍합니다. 또한, Gumbel-Softmax 샘플링을 활용해 멀티모달 컨트롤러가 각 입력 샘플의 QoI에 따라 최적의 레이어 할당을 학습하도록 설계되었습니다.

- **Performance Highlights**: ADMN은 멀티모달 로컬라이제이션 및 액션 인식 작업에서 검증되었으며, 기존의 최첨단 모델에 비해 최대 75%의 부동소수점 연산(FLOPS)과 60%의 지연(latency)을 줄이면서 정확도를 유지할 수 있음을 보여주었습니다. 또한, ADMN의 효율성을 입증하는 다양한 실험 결과를 제공하며, 코드와 자료를 공개했습니다.



### BalanceKV: KV Cache Compression through Discrepancy Theory (https://arxiv.org/abs/2502.07861)
- **What's New**: 이 논문에서는 BalanceKV라는 새로운 KV 캐시 압축 방법을 제안합니다. 이 방법은 Banaszczyk의 벡터 균형 이론에 기반한 기하학적 샘플링 프로세스를 활용하여 키와 값 토큰 간의 종속성을 개선하고, 메모리 복잡성을 줄이는 데 초점을 맞추고 있습니다. 또한, 이 접근법은 기존 방법에 비해 이론적으로 입증된 성능 개선을 제공하며, 실제 데이터에서도 검증되었습니다.

- **Technical Details**: BalanceKV는 KV 캐시의 효율성을 높이기 위해 기하학적 구성을 활용하여 리니어 스케일에서 서브리니어 스케일로 변환하는 알고리즘입니다. 이 알고리즘의 핵심은 자가 주의 계층 내부에서 수행되는 작업을 잘 근사하는 소수의 키와 값의 하위 집합을 파악하는 것이며, 이를 통해 메모리 사용량을 효과적으로 줄일 수 있습니다. 또한, 본 알고리즘은 기존의 샘플링 기법보다 더 높은 정확도를 제공합니다.

- **Performance Highlights**: 실험 결과, BalanceKV는 공공 데이터셋에서 여러 기존 알고리즘보다 우수한 상대적 오차를 기록했습니다. 특히, 장기 컨텍스트 이해 과제를 평가하는 LongBench 벤치마크에서 이전의 휴리스틱 캐시 압축 방법과 비교하여 더 나은 성과를 보여주었습니다. 이러한 결과는 BalanceKV의 실용성과 성능 개선 효과를 입증합니다.



### MAAT: Mamba Adaptive Anomaly Transformer with association discrepancy for time series (https://arxiv.org/abs/2502.07858)
- **What's New**: MAAT는 시계열 이상 탐지에서 새로운 구조를 제안하며, 이 아키텍처는 Sparse Attention을 특징으로 하여 긴 종속성을 효율적으로 캡처하고 계산 중복을 줄입니다. 또한 Mamba-Selective State Space Model이 재구성 모듈에 통합되어 주의를 기울여 이상점을 보다 효과적으로 식별할 수 있도록 합니다. 실험 결과, MAAT는 기존의 방법들보다 뛰어난 성능을 보여주며, 다양한 시계열 애플리케이션에서 이상 탐지의 새로운 기준을 설정했습니다.

- **Technical Details**: MAAT 아키텍처는 스킵 연결과 게이트 주의를 결합하여 재구성 능력을 향상시킵니다. Sparse Attention 메커니즘을 채택함으로써 긴 시계열 데이터를 효율적으로 처리할 수 있으며, 이로 인해 계산 효율성과 정확도 간의 최적의 균형을 이루게 됩니다. 더불어, 각 블록의 크기를 동적으로 조정할 수 있어 다양한 시계열 데이터의 복잡성을 감안한 세심한 접근을 제공합니다.

- **Performance Highlights**: MAAT의 실험 결과는 이전 방법론에 비해 이상 탐지에서 뛰어난 구분 가능성을 보여주며, 특히 노이즈와 비정상적 패턴에 강한 성능을 발휘합니다. 이 아키텍처는 다양한 데이터셋에서 높은 일반화 성능을 보여 주며, 이를 통해 실제 시나리오에서의 비지도 시계열 이상 탐지 기술의 잠재력을 증명하고 있습니다.



### Advancing Heat Demand Forecasting with Attention Mechanisms: Opportunities and Challenges (https://arxiv.org/abs/2502.07854)
- **What's New**: 본 논문은 에너지 시스템의 탈탄소화와 스마트 데이터 활용을 통해 열 수요를 예측하는 새로운 심층 학습 모델을 제안한다. 기존의 통계 모형에 비해 복잡한 시간 시계열 패턴을 더 효과적으로 포착하기 위해 Attention 메커니즘을 도입하여 데이터를 시간-주파수 영역에서 처리한다. 또한, 분해된 특성들을 입력 요소로 사용하여 최적의 열 수요 예측을 달성한다.

- **Technical Details**: 열 수요 예측을 위해, Continuous Wavelet Transform (CWT)을 통해 생성된 wavelet scalograms 형태로 입력 특성을 표현하는 합성곱 기반 네트워크를 채택하였다. 모델은 내부 및 외부 특성을 두 가지 가지로 분리하고, 합성곱 층 이후 교차 주의(block)를 사용하여 최적의 문맥을 동적으로 선택한다. 최종적으로, 주어진 입력 데이터를 기반으로 Fully Connected Layers를 통해 열 수요를 예측한다.

- **Performance Highlights**: Experimental results show that the proposed model with wavelet scalograms surpasses traditional LSTM models in both quantitative (MAE and MAPE 성능 향상) 및 qualitative 평가에서 우수한 성과를 기록했다. 특히, 개별 세분화된 특성들의 활용이 예측 정확도를 높였음을 확인할 수 있었다. 이러한 결과는 내일의 열 수요를 보다 정교하게 예측할 수 있는 가능성을 나타낸다.



### Understanding Classifier-Free Guidance: High-Dimensional Theory and Non-Linear Generalizations (https://arxiv.org/abs/2502.07849)
- **What's New**: 최근 연구는 Classifier-Free Guidance (CFG)의 효과에 대한 우려를 제기하였으며, 낮은 차원에서는 목표 분포에 대한 overshoot와 샘플 다양성 감소를 초래할 수 있음을 보여주었습니다. 본 연구에서는 무한하고 충분히 높은 차원에서 CFG가 목표 분포를 효과적으로 재현함을 입증하며, 이는 차원의 축복(blessing-of-dimensionality) 결과를 나타냅니다. 또한 유한 차원에서의 효과를 탐구하고, overshoot 및 분산 감소를 정확히 특성화하였습니다.

- **Technical Details**: Diffusion Models은 최신 알고리즘으로 고품질의 이미지를 생성하는 데 사용되며, Orstein-Uhlenbeck Langevin dynamics를 시뮬레이션하여 데이터를 점진적으로 무작위화한 후 이 과정을 역으로 학습합니다. 이 연구에서는 CFG가 외부 분류기 없이 클래스 레이블 및 텍스트 프롬프트를 기반으로 샘플을 조건부로 생성하도록 훈련된 점을 강조하며, 높은 차원 데이터에서 CFG가 효과적으로 목표 분포를 복제함에 따라 보다 일반화된 비선형 CFG을 소개합니다. 실험을 통해 이러한 비선형 CFG가 이미지 품질 및 다양성 측면에서 유리하다는 것을 확인했습니다.

- **Performance Highlights**: 우리의 분석 결과, 비선형 CFG는 이미지 품질을 향상시키고 생성 과정의 유연성을 증가시키며, 추가적인 계산 비용 없이도 향상된 결과를 제공합니다. Gaussian mixture 및 클래스 조건 모델에 대한 수치적 시뮬레이션을 통해 이를 검증하였고, 실험은 CIFAR-10, ImageNet 데이터셋에서 높은 성능과 다양성을 달성함을 보여주었습니다. 이러한 성과는 CFG의 효용성을 확증하며, 다양한 다른 CFG 변형과의 통합을 용이하게 합니다.



### Emotional EEG Classification using Upscaled Connectivity Matrices (https://arxiv.org/abs/2502.07843)
- **What's New**: 기존의 감정 EEG 분류 방법은 CNN을 사용하여 연결 매트릭스를 입력으로 활용하고 있지만, 이 과정에서 중요한 패턴이 잃어버려지는 한계를 발견했습니다. 이를 해결하기 위해, 우리는 연결 매트릭스를 업스케일링하여 지역 패턴을 강화하는 아이디어를 제안합니다. 실험 결과에 따르면, 이 간단한 접근 방식이 분류 성능을 상당히 향상시키는 것으로 나타났습니다.

- **Technical Details**: 우리의 방법은 전극 간의 선형 관계를 정량화하기 위해 상관 관계 계수(Pearson Correlation Coefficient, PCC)를 기반으로 대칭적 연결 매트릭스를 생성합니다. 접속성 매트릭스는 2D 형식에서 EEG 전극 간의 관계를 수량화하며, CNN을 통해 의미 있는 EEG 표현을 추출하도록 설계되었습니다. 이때 인접한 전극 간의 관계를 강조하기 위해 거리 제한 배열 방법을 사용하여 연결 매트릭스를 구성했습니다.

- **Performance Highlights**: 제안한 업스케일링 방법은 CNN 기반의 감정 EEG 분류 성능을 개선하는 데 필요한 정보를 더욱 효과적으로 학습하도록 지원합니다. 각 주파수 대역마다 생성된 32x32 연결 매트릭스가 CNN 모델에 입력되어, ReLU 활성화 및 맥스 풀링 레이어를 통해 최종적인 분류가 이루어집니다. 이러한 방법론은 기존 접근 방식에 비해 높은 분류 정확도를 기록하였으며, 정량적 감정 분석에서의 활용 가능성을 높이는데 기여할 것입니다.



### SHARP: Accelerating Language Model Inference by SHaring Adjacent layers with Recovery Parameters (https://arxiv.org/abs/2502.07832)
Comments:
          24 pages

- **What's New**: SHARP(Sharing Adjacent Layers with Recovery Parameters)는 기존의 LLM(대형 언어 모델)의 인퍼런스를 가속화하기 위해 인접 계층 간의 파라미터를 공유하는 새로운 접근 방식을 제안합니다. 이 방법은 메모리 부담을 줄이면서 성능을 유지하기 위해 저계수 회복 파라미터를 도입합니다. SHARP는 연속적인 계층의 출력이 유사하다는 사실에 기반하여 설계되었으며, 단일 계층 워밍업(Single Layer Warmup, SLW) 및 감독하에 미세 조정(Supervised Fine-Tuning, SFT)의 두 단계로 구성됩니다.

- **Technical Details**: SHARP는 현재 계층의 출력을 원본 대체 계층의 출력과 비교하며, L_2 손실을 최소화하여 SFT 단계에 적합한 초기값을 제공합니다. SLW는 여러 계층을 한 개의 계층으로 예측하는 데 중요한 역할을 하며, 3/4의 원래 MLP 계층을 생략해도 모델의 perplexity를 효과적으로 복구할 수 있도록 합니다. SHARP는 MLP 계층 파라미터를 38%에서 65%까지 줄이면서도 다양한 인디스트리뷰션(task 분류) 작업에서 모델 성능을 회복합니다.

- **Performance Highlights**: SHARP는 Llama2-7b 모델에 비해 모델 저장 용량을 42.8%, 총 인퍼런스 시간을 42.2% 절감하는 성과를 보였습니다. 50,000개의 미세 조정 데이터만 사용하여 다양한 작업에서 모델 perplexity를 복구하였으며, 특히 메모리 관련 다운스트림 작업에서 더 나은 성능을 나타냈습니다. 이 연구는 모바일 기기에서도 효율적인 LLM 배포를 위한 해결책으로 SHARP를 제시합니다.



### Implicit Language Models are RNNs: Balancing Parallelization and Expressivity (https://arxiv.org/abs/2502.07827)
- **What's New**: 이 논문에서는 언어 모델링에서 상태 공간 모델(State-Space Models, SSMs)과 트랜스포머(Transformers)의 한계를 극복하기 위해 암묵적 상태 공간 모델(Implicit SSMs)을 제안합니다. 이 모델은 비선형 상태 전이(Non-linear state-transitions)를 통해 표현력을 확보하면서도 동시에 훈련 중 병렬 처리(Parallelization)를 유지할 수 있게 설계되었습니다. 또한, 이 암묵적 모델은 자연어 추론(Natural Language Reasoning) 작업과 대규모 언어 모델 사전 학습(Pretraining)에서 뛰어난 성능을 보여줍니다.

- **Technical Details**: 암묵적 SSM은 문제의 난이도에 따라 계산 부하를 자연스럽게 조절하며, 모든 토큰이 해결 가능한 경우에는 병렬성(Parallelizability)을 유지하고, 해결할 수 없는 토큰이 있는 경우 RNN처럼 동작합니다. 이러한 구조는 딥 균형 모델(Deep Equilibrium Models)로, 임의의 별도 경로를 통과하지 않고 고정점(Fixed Point)에서만 역전파(Backpropagation)가 이루어져 메모리 사용량을 줄일 수 있습니다. 또한, 암묵적 SSM은 특정 상황에서 트랜스포머 및 SSM보다 월등한 상태 추적(State-Tracking) 기능을 발휘합니다.

- **Performance Highlights**: 암묵적 SSM 모델은 표준 벤치마크에서 명시적 모델(Explicit Models)보다 뛰어난 성능을 보이는데, 특히 S5 문제를 성공적으로 해결하며, 기존의 트랜스포머 및 SSM이 해결하지 못하는 요소를 풀어냅니다. 이를 통해 1.3B 파라미터로 207B 토큰의 데이터셋에 대한 훈련 성과를 달성하였으며, 이는 현재까지 이루어진 가장 큰 암묵적 모델로 기록됩니다. 논문에서는 또한 훈련 커리큘럼을 통해 효율적인 스케일링을 달성하였음을 강조하고 있습니다.



### Satellite Observations Guided Diffusion Model for Accurate Meteorological States at Arbitrary Resolution (https://arxiv.org/abs/2502.07814)
- **What's New**: 이번 연구에서는 기상 예측의 정확성을 향상시키기 위해 위성 관측을 기반으로 한 확산 모델인 Satellite-observations Guided Diffusion Model (SGD)을 제안합니다. SGD는 ERA5 재분석 데이터와 위성 관측 데이터를 조건으로 활용해 저해상도 기상 상태를 업스케일링하는 혁신적 접근 방식을 제공합니다. 기존 방법들이 위성 데이터와의 상관관계를 간과했으나, SGD는 주의 메커니즘을 통해 이를 보완합니다.

- **Technical Details**: SGD는 ERA5와 GridSat 위성 관측 데이터를 융합하여 훈련되며, 이 과정에서 교차 주의(attention) 모듈을 활용하여 고해상도 기상 데이터를 효과적으로 생성합니다. 특히, SGD는 최적화 가능한 컨볼루션 커널을 사용해 저해상도 지도를 기반으로 고해상도 ERA5 지도를 생성하며, 저해상도 지도 및 기상 관측소 데이터를 가이드로 포함하여 샘플링합니다. 이러한 방식은 고해상도 기상 상태의 세밀한 디테일을 보장합니다.

- **Performance Highlights**: 실험 결과, SGD는 6.25km 해상도로 정확한 기상 상태를 생성하며, 기존의 보간(interpolation) 기반 방법이나 확산 모델에 비해 뛰어난 성능을 보였습니다. 또한, ablation study를 통해 GridSat 맵을 조건으로 사용하는 것이 SGD의 성능을 현저히 향상시킨다는 점이 입증되었습니다. 이러한 혁신적인 접근 방식은 특히 기상 데이터 처리 및 예측의 정확성을 높이는 데 기여할 것으로 기대됩니다.



### Joint Transmit and Pinching Beamforming for PASS: Optimization-Based or Learning-Based? (https://arxiv.org/abs/2502.08637)
Comments:
          Submitted to IEEE

- **What's New**: 새로운 핀칭 안테나 시스템(PASS)을 활용한 다중 사용자 다중 입력 단일 출력(MISO) 프레임워크가 제안되었습니다. PASS는 수천 개의 파장에 걸쳐 여러 개의 파이프를 갖추고 있으며, 저비용 유전체 입자를 통해 신호를 방출하고 이를 재구성할 수 있게 합니다. 이 프레임워크는 신호의 경로 손실 및 위상을 조정하여 핀칭 빔포밍을 최적화합니다.

- **Technical Details**: PASS는 유전체 도파관을 기반으로 구축되어 수십 미터에서 수천 미터까지 길이를 조정할 수 있으며, 유전체 입자들이 유동적으로 배치될 수 있습니다. 제안된 문제를 해결하기 위해 가장 먼저 먼 거리에서도 안정적인 선형 직선 링크를 유지하고 대규모 경로 손실을 조정할 수 있는 알고리즘이 개발되었습니다. 또한, Karush-Kuhn-Tucker(KKT) 기반의 이중 학습(KDL) 방식이 소개되어, 데이터 기반으로 KKT 해법을 재구성할 수 있게 합니다.

- **Performance Highlights**: 시뮬레이션 결과에 따르면, PASS 프레임워크는 기존의 대규모 MIMO 시스템보다 월등한 성능을 보여주며, 특히 소수의 PAs를 사용할 때도 효과적입니다. KDL-Transformer 알고리즘은 MM-PDD 알고리즘보다 30% 이상의 성능 향상을 기록하며, 현대 GPUs에서 밀리초 수준의 반응 속도를 달성할 수 있습니다.



### Rapid Whole Brain Mesoscale In-vivo MR Imaging using Multi-scale Implicit Neural Representation (https://arxiv.org/abs/2502.08634)
- **What's New**: 이번 연구에서는 Rotating-view super-resolution (ROVER)-MRI라는 새로운 영상 재구성 기법을 개발하였습니다. 이 기법은 다중 뷰 두꺼운 슬라이스를 사용하여 MRI 데이터의 스캔 시간을 두 배로 줄이면서도 고해상도의 해부학적 세부 사항을 유지합니다. ROVER-MRI는 고위험 굴절력 기반 알고리즘을 채택하여 7T MRI 스캐너에서 17분의 짧은 스캔으로 180 μm의 등방성 공간 해상도를 달성할 수 있었습니다.

- **Technical Details**: ROVER-MRI는 비지도 신경망(unsupervised neural network) 기반의 알고리즘으로 multi-view thick slices에서 MRI 데이터를 재구성하는 데 최적화되어 있습니다. 기존의 bicubic interpolation 및 regularized least-squares super-resolution reconstruction (LS-SRR) 기법과 비교하여, ROVER-MRI는 상대 오차(relative error, RE)를 22.4% 감소시키고, 풀 폭 절반 최대(full-width half maximum, FWHM)를 7.5% 저하시켜 보다 세밀한 구조적 세부 사항의 보존을 가능하게 하였습니다. 이 기술은 중간 규모(mesoscale) MR 이미징에 효율적이고 강력한 접근법을 제공합니다.

- **Performance Highlights**: ROVER-MRI는 LS-SRR 기술에 비해 재구성 품질에서 우수한 성능을 보였습니다. ROVER-MRI의 활용은 해부학적 세부 사항이 필요한 연구 및 시간이 효율적인 이미징을 요구하는 응용 분야에 큰 잠재력을 가집니다. 보다 구체적으로, 전 상황에서 뇌의 전체를 신속하고 높은 해상도로 스캔할 수 있어, 향후 임상 및 연구 분야에서의 다양한 활용이 기대됩니다.



### Concentration Inequalities for the Stochastic Optimization of Unbounded Objectives with Application to Denoising Score Matching (https://arxiv.org/abs/2502.08628)
Comments:
          30 pages

- **What's New**: 이 연구에서는 McDiarmid의 불평등에 대한 새로운 일반화와 더불어 비제한적 함수에 대한 균일한 대수 법칙(ULLN) 결과를 도출하였다. 이 작업은 비제한 목적 함수의 확률적 최적화에 대한 통계적 오류 경계를 얻기 위해 사용된다. 특히, 훈련 데이터와 함께 쉽게 샘플링 가능한 보조 난수 변수를 사용하는 알고리즘에서 샘플 재사용의 이점을 정량적으로 포착하는 방법을 소명하였다.

- **Technical Details**: 본 연구에서는 비제한 목표 함수 패밀리(고려하는 매개변수 θ에 따라 달라짐)에 대해 적용될 수 있는 McDiarmid의 불평등의 일반화를 도출하였다. 본 결과는 한 구성 요소 차이에 대한 비상수 경계를 활용하며, 이는 국소-립시츠(local-Lipschitz) 함수의 경우를 포함한다. 비제한 목표 함수를 다루기 위해 서브-가우시안(sub-Gaussian) 케이스를 넘어서는 넓은 범위의 꼬리 행동(tail behavior)을 허용하였다.

- **Performance Highlights**: 본 연구 결과는 비제한 목표 함수를 가진 여러 확률적 최적화 문제에 널리 적용될 수 있으며, 특히 구조적으로 비대칭인 샘플 숫자를 고려해야 하는 경우 유용하다. 예를 들어, 노이즈 제거 점수 매칭(DSM)의 적용에서는 쉽게 샘플링할 수 있는 보조 변수와 훈련 데이터를 함께 쌍으로 만들어 사용함으로써 얻어지는 통계적 오류 경계를 명확히 하고 있다. 이를 통해 비제한 함수의 직접적인 다루기가 가능하다는 장점을 보여준다.



### Mathematical Data Scienc (https://arxiv.org/abs/2502.08620)
- **What's New**: 본 논문에서는 기계 학습(Machine Learning)이 새로운 수학적 구조를 발견하는 데 도움이 될 수 있는 방법을 제안합니다. 이를 위해 수학적 데이터 과학(mathematical data science)이라는 접근 방식을 소개하며, 이는 개별 수학적 객체가 아닌 집합적 연구를 통해 이루어집니다. 주로 데이터셋을 생성하고 이를 바탕으로 기계 학습 실험을 수행함으로써 새로운 통찰력을 도출하는 과정을 설명합니다.

- **Technical Details**: 이 연구에서는 수학적 객체를 집합적으로 고려하여 데이터셋(dataset)을 생성하고, 기계 학습 도구를 적용해 구조를 찾고, 이를 통해 더 나은 이해를 추구합니다. 두 가지 사례 연구로는 수론에서의 murmurations와 Kronecker 계수에 관계된 분할의 loadings가 포함됩니다. 이 과정에서 생성된 데이터는 실제 연구 대상이 되는 더 큰 '모집단(population)'의 샘플이며, 이를 통해 통계적 속성을 이해하는 데 필요한 대표적인 부분집합을 선택합니다.

- **Performance Highlights**: 기계 학습의 발전과 컴퓨터의 성능 향상 덕분에 수학적 데이터 과학의 가능성이 한층 더 밝아졌습니다. 그러나 블랙 박스(black box) 모델 해석과 증명 가능한 정리 및 추측을 도출하는 과정에서의 도전에 대한 고민도 필요합니다. 이 연구는 전체적인 데이터 과학의 각 단계를 통해 새로운 정리와 추측을 발견하는 것을 목표로 하며, 이는 모든 단계가 여전히 인간 수학자의 직관과 선택에 의존하고 있음을 강조합니다.



### Scalable Thermodynamic Second-order Optimization (https://arxiv.org/abs/2502.08603)
Comments:
          17 pages, 5 figures

- **What's New**: 이번 연구에서는 열역학 기반의 컴퓨터(thermodynamic computers)를 활용하여 AI 훈련 알고리즘을 가속화하는 방법을 제안합니다. 특히 Kronecker-factored approximate curvature(K-FAC)라는 인기 있는 이차 최적화 알고리즘을 열역학 컴퓨터에 대해 확장할 수 있는 방법에 대해 논의합니다. 연구 결과에 따르면, 네트워크의 조밀성이 증가할수록 성능이 향상될 것으로 예상되며, 현실적인 하드웨어 특성을 고려할 때 대규모 비전 및 그래프 문제에서도 상당한 속도 향상이 이루어질 것으로 전망합니다.

- **Technical Details**: 연구의 초점은 K-FAC 알고리즘을 열역학적으로 구현하여 이차 최적화 시의 계산 비용을 줄이는 것입니다. K-FAC는 Fisher 매트릭스를 블록 대각 근사화(block-diagonal approximation)하여 이를 열역학 하드웨어에 최적화할 수 있습니다. 열역학 K-FAC 알고리즘의 비대칭 복잡 성 분석에 따르면, 신경망의 너비가 넓어질수록 성능에 있어 선형적 이점을 증가시킬 것으로 보이며, 이에 따라 메모리와 런타임 비용에서도 개선이 예상됩니다.

- **Performance Highlights**: 실험 결과, 양자화 잡음(quantization noise)이 큰 경우에도 K-FAC 알고리즘의 이점을 유지할 수 있음을 보여주었고, 이로 인해 열역학 K-FAC 알고리즘은 상당한 내구성을 가지고 있는 것으로 판단됩니다. 대규모 비전 및 언어 문제의 시뮬레이션을 통해 열역학 K-FAC이 표준 K-FAC 및 Adam보다 우수한 성능을 보여주었으며, 모델의 하이퍼파라미터(예: 네트워크의 너비)를 조정하면 더욱 큰 속도 향상도 가능함을 시사하고 있습니다.



### Mapping the Landscape of Generative AI in Network Monitoring and Managemen (https://arxiv.org/abs/2502.08576)
Comments:
          32 pages, 9 figure, 10 tables

- **What's New**: 이 연구는 Generative Artificial Intelligence (GenAI) 모델들이 네트워크 모니터링 및 관리에 어떻게 활용될 수 있는지를 탐구합니다. 특히 LLM, GPT, Diffusion Models와 같은 다양한 GenAI의 포괄적인 개요를 제공하며, 이러한 기술들이 사람과 기계 간의 상호작용을 개선하는 방법을 밝힙니다. 또한, 연구에서는 GenAI 채택을 위한 도전과제와 기회에 대한 논의도 포함되어 있습니다.

- **Technical Details**: GenAI는 복잡한 데이터 분포의 특징을 추출하고 이를 이용해 유사하지만 독특한 새로운 데이터를 생성하는 능력이 뛰어난 모델입니다. 특히 LLM(대형 언어 모델), Diffusion Models, 상태 공간 모델(State Space Models) 등이 GenAI 기술의 대표적인 예시로 주목받고 있습니다. 이러한 모델들은 고성능의 대규모 GPU를 활용하여 훈련되며, 네트워크 관리에 필요한 적응적으로 응답하는 네트워크 구축에 기여할 수 있습니다.

- **Performance Highlights**: GenAI 모델들은 상업적 가치와 기술적 가능성을 바탕으로 빠르게 성장하고 있습니다. 글로벌 GenAI 시장은 2023년 말에 약 454억 달러에 달하며, 향후 연평균 약 20억 달러 성장할 것으로 예상됩니다. 이 연구는 네트워크 모니터링 및 관리에 있어서 GenAI의 활용이 네트워크의 자가 구성, 자가 최적화, 자가 치유 능력을 향상시키는 데 기여할 것이라고 결론짓습니다.



### QA-Expand: Multi-Question Answer Generation for Enhanced Query Expansion in Information Retrieva (https://arxiv.org/abs/2502.08557)
Comments:
          8 pages

- **What's New**: 본 논문은 QA-Expand라는 새로운 쿼리 확장 프레임워크를 소개합니다. 이 프레임워크는 초기 쿼리에서 다수의 관련 질문을 생성하고, 각 질문에 대해 대응하는 의사 답변(pseudo-answer)을 만들어냅니다. 기존의 방식들이 가지던 정형화된 한계를 극복하고, 정보를 보다 풍부하게 포착할 수 있는 방안을 제공합니다.

- **Technical Details**: QA-Expand는 세 가지 주요 단계로 구성됩니다: 초기 쿼리로부터 다수의 질문 생성, 각 질문에 대한 의사 답변 생성, 그리고 가장 관련성 있는 질문-답변 쌍을 선택하는 피드백 메커니즘을 통한 재작성 과정입니다. 이 과정에서 Large Language Model (LLM)을 활용해 정보를 다층적으로 구조화할 수 있습니다.

- **Performance Highlights**: BEIR 및 TREC 벤치마크에서 실시한 광범위한 실험이 QA-Expand가 기존의 쿼리 확장 기술들보다 최대 13% 향상된 검색 성능을 보여줌을 증명합니다. 이는 현대 정보 검색 문제에 대한 효과적인 해결책을 제공하며, 여러 측면에서 정보의 풍부한 이해를 이루도록 지원합니다.



### Human-Centric Foundation Models: Perception, Generation and Agentic Modeling (https://arxiv.org/abs/2502.08556)
Comments:
          9 pages

- **What's New**: 최근 인간 중심의 기초 모델(Human-centric Foundation Models, HcFMs)이 다채로운 인간 중심 작업을 단일 프레임워크로 통합하여 전통적인 작업별 접근 방식을 초월하고 있다는 점에서 주목할 만한 변화가 있습니다. 이러한 모델은 인간의 외모, 감정, 정체성 및 행동을 보다 정교하게 이해하고 생성할 수 있는 가능성을 열어줍니다. 분야의 발전은 연구자들이 인간을 보다 포괄적이고 복잡한 시스템으로 이해해야 한다는 요구와 함께 진행되고 있습니다.

- **Technical Details**: 이 논문에서는 HcFMs를 네 가지 카테고리로 분류하는 새로운 분류법을 제안합니다: (1) 인간 중심의 인식 기초 모델, (2) 인간 중심의 AIGC 기초 모델, (3) 통합 인식 및 생성 모델, (4) 인간 중심의 에이전틱 모델입니다. 각 모델은 그들이 지원하는 다양한 하위 작업에 따라 분류되어, 인간 중심의 데이터로부터 효율적으로 학습할 수 있도록 돕습니다.

- **Performance Highlights**: HcFMs는 기존의 작업별 모델보다 높은 일반화 능력과 적용 가능성, 그리고 사실감을 보증합니다. 이 모델들은 특히 2D와 3D 작업에서의 성능을 개선하며, 인간의 행동과 상호작용을 보다 정밀하게 표현할 수 있는potential을 지니고 있습니다. 또, 이 모델들은 다중 모달리티(multi-modality)를 활용하여 더욱 풍부한 인식을 가능하게 하며, 사용자와의 상호작용 작업에도 적용될 수 있습니다.



### A Machine Learning-Ready Data Processing Tool for Near Real-Time Forecasting (https://arxiv.org/abs/2502.08555)
- **What's New**: 이번 논문은 우주 날씨 예보(space weather forecasting)를 위한 기계 학습(Machine Learning) 준비 데이터 처리 도구의 개발을 소개합니다. 이 도구는 다양한 실시간(Near Real-Time, NRT) 데이터 소스의 데이터를 통합하여 현재의 우주 날씨 예측 능력에서 발생하는 주요 격차를 해소합니다. 특히 극단적인 태양 이벤트에 대한 시계열 예측(time-series forecasting) 및 이벤트 탐지(event detection)에 중점을 두고 있습니다.

- **Technical Details**: 이 도구는 태양 이미지(solar imagery), 자기장 측정(magnetic field measurements), 에너지 입자 유출(energetic particle fluxes)과 같은 다양한 NRT 소스로부터 수집된 데이터를 처리하고 구조화합니다. 사용자는 데이터를 다운로드하고 처리하며 ML 애플리케이션을 위한 레이블링(labeling)을 간소화할 수 있는 프레임워크를 이용할 수 있습니다. 이는 NRT 우주 날씨 예보와 과학 연구의 흐름을 개선하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 이 연구는 기계 학습 모델을 위한 데이터 처리를 효율적으로 구성하여, 극단적인 태양 이벤트를 정확하게 예측할 수 있도록 지원합니다. 우주 탐사와 지구 기반 기술 보호에 기여하며, 이를 통해 우주 날씨 예측의 신뢰성을 높이는 데 기여할 것입니다. 앞으로의 연구에 있어 이 도구는 중요한 기반 기술로 자리잡을 것으로 기대됩니다.



### Copula-based mixture model identification for subgroup clustering with imaging applications (https://arxiv.org/abs/2502.08549)
- **What's New**: 본 논문은 Copula-Based Mixture Models (CBMMs)를 제안하여 데이터 클러스터링을 위한 보다 유연한 접근법을 제시합니다. 기존의 혼합 모델은 고정된 분포 형태를 가지는 것이 일반적이지만, CBMMs는 다양한 마진 및 코풀라 형태를 허용하여 이 질병별로 이질적인 분포를 처리할 수 있습니다. 특히, Generalized Iterative Conditional Estimation (GICE) 알고리즘의 적응을 통해 CBMM을 비지도 학습 방식으로 식별하는 방법을 제안합니다.

- **Technical Details**: CBMM에서는 각 구성 요소가 서로 다른 마진과 의존 구조를 가질 수 있습니다. 이는 Sklar의 정리에 따라 가능한가 하며, 논문에서는 CBMM의 최적화를 위해 GICE 알고리즘을 사용하여 각 구성 요소의 마진 및 코풀라 형태를 반복적으로 추정합니다. CBMM의 데이터 적합성 및 수렴을 검증하기 위해 합성 데이터와 실제 데이터를 사용하여 실험을 수행하였습니다.

- **Performance Highlights**: CBMM-GICE 클러스터링 방법은 합성 데이터와 MNIST 데이터베이스, 실제 심장 자기공명 영상에 대한 실험을 통해 성능을 평가하였습니다. 이 방법은 특히 의료 이미징 애플리케이션에서의 유용성을 보여주며, 기존의 Expectation Maximization (EM)으로 식별된 혼합 모델과 비교되어 개선된 결과를 나타냅니다. 이러한 접근법은 다차원 데이터 분석에 대한 더 나은 유연성과 적합성을 제공합니다.



### Fine-Tuning Topics through Weighting Aspect Keywords (https://arxiv.org/abs/2502.08496)
Comments:
          17 pages, 8 figures, 3 tables

- **What's New**: 이 논문은 주제 모델링(Topic Modeling)의 새로운 접근 방식을 제안합니다. 특히 잘 탐색되지 않은 영역에서 숨겨진 패턴을 발견하기 위해 여러 관점에서 주제를 검토해야 할 필요성을 다루고 있습니다. 이 방법은 도메인 지식에서 파생된 가중치가 부여된 키워드를 활용하여 주제 모델링의 정확성을 높입니다.

- **Technical Details**: 연구 방법은 표준 주제 모델링에서 시작되며, 네 가지 주요 단계를 포함한 추가 프로세스를 포함합니다. 첫 번째는 각 관점에 대한 키워드를 정의하는 것이고, 두 번째는 이 키워드의 관련성에 기초하여 가중치를 부여합니다. 세 번째 단계에서는 관점 가중치가 부여된 키워드와 주제 키워드 간의 관련성 점수를 계산하여 관점-주제 모델(Aspect-Topic Models)을 생성하며, 마지막으로 이 점수를 활용하여 새로운 문서를 조정합니다.

- **Performance Highlights**: 결과적으로 생성된 주제 모델은 상위 점수를 받은 문서가 동일한 주제의 같은 관점을 다룰 가능성이 더 높다는 것을 보여줍니다. 이는 주제와 관련된 문서를 효과적으로 찾아내는 모델의 능력을 강조합니다. 이 연구는 주제 모델링의 새로운 응용 가능성을 제시하여 관련 문서를 더욱 정확하게 식별할 수 있게 합니다.



### Enhancing Auto-regressive Chain-of-Thought through Loop-Aligned Reasoning (https://arxiv.org/abs/2502.08482)
Comments:
          work in progress

- **What's New**: 최근 연구에서는 Chain-of-Thought (CoT) 프롬프트가 언어 모델의 추론 능력을 향상시키는 강력한 기법으로 등장했습니다. 하지만 긴 CoT 경로를 생성하는 것은 도전 과제가 되고 있습니다. 본 논문에서는 Looped Transformer를 활용한 RELAY (REasoning through Loop Alignment iterativelY)를 제안하여, CoT 추론 단계와 루프 반복을 정렬하고 추가적인 중간 감독 학습을 통해 길이 일반화(length generalization) 능력을 강화합니다.

- **Technical Details**: RELAY는 루프된 Transformer의 장점을 최대한 활용하여 오토레그레시브(auto-regressive) 모델이 더 긴 추론 체인을 처리할 수 있도록 돕는 새로운 프레임워크입니다. 주요 혁신은 루프된 Transformer 모델이 여러 작업에서 일반적인 추론기의 역할을 수행할 수 있음을 실증적으로 입증하는 것과, CoT 추론 단계와 루프된 Transformer 간의 반복 간 정렬을 통해 훈련 길이를 초과하는 문제에 대한 정확한 추론 체인을 생성할 수 있도록 하는 것입니다.

- **Performance Highlights**: 광범위한 실험을 통해 RELAY 접근법이 생성한 고품질의 추론 체인을 통해 오토레그레시브 Transformers의 추론 능력을 크게 향상시키는 것을 입증했습니다. 이러한 성과는 더욱 복잡한 문제 해결에 있어 모델의 성능을 개선하고, 다양한 언어 작업에서 일관된 성능을 유지할 수 있도록 합니다.



### Numerical Schemes for Signature Kernels (https://arxiv.org/abs/2502.08470)
- **What's New**: 이번 논문에서는 서명 커널(Signature Kernel)이 두 개의 독립적인 시간 변수를 가진 hyperbolic partial differential equation (PDE)의 Goursat 문제를 해결하기 위한 방법으로 분석되고 제시되었습니다. 저자들은 기존의 유한 차분 방법의 한계를 극복하기 위해 경계 조건에 대한 고급 수치 기법을 도입하고, 이 과정에서 다항식 근사 기법의 이론적 수렴성을 엄밀히 증명하였습니다. 또한, 이러한 새로운 방법들이 전통적인 유한 차분 방법보다 평균 절대 백분율 오차(MAPE)를 여러 배로 개선하는 성과를 보여주었습니다.

- **Technical Details**: 이 논문에서는 서명 커널이 연속적으로 미분 가능한 경로에 대해 정의되는 커널인 반면, Goursat 문제는 하이퍼볼릭 PDE에 대한 특정 조건을 필요로 합니다. 연구자들은 이 문제를 해결하기 위해 다항식 표현을 사용하여 경계 조건을 근사하거나 보간하는 두 가지 고급 수치 기법을 개발하였습니다. 또한, GPU 병렬화가 가능한 알고리즘을 통해 입력 시퀀스의 길이에 관계없이 계산 복잡성을 줄이는 방법을 제시하였습니다.

- **Performance Highlights**: 실험적으로 확인된 결과에 따르면, 저자들이 제안한 접근 방식은 평균 절대 백분율 오차(MAPE)에서 전통적인 유한 차분 기법보다 여러 배 향상된 결과를 보여주었습니다. 특히, 이 방법들은 고주파 데이터 처리에 있어서 더 나은 확장성을 제공하며, 계산 복잡성을 제곱에서 선형으로 감소시킵니다. 저자들은 이 알고리즘을 전용 Python 라이브러리로 구현하여 공개적으로 제공하고 있습니다.



### Better Embeddings with Coupled Adam (https://arxiv.org/abs/2502.08441)
Comments:
          17 pages, 8 figures; figures corrected

- **What's New**: 이 논문에서는 대형 언어 모델(LLM)에서의 단어 임베딩의 비등방성을 초래하는 원인으로 Adam 최적화 알고리즘의 두 번째 모멘트를 지적하고, 이를 해결하기 위한 Coupled Adam이라는 수정된 최적화 기법을 제안합니다. Coupled Adam은 임베딩 매개변수에 대해 비등방성 문제를 완화하기 위해 특별히 설계된 효율적인 조정입니다.

- **Technical Details**: LLM에서의 단어 임베딩은 주어진 토큰 시퀀스를 입력으로 받아 다음 토큰을 예측하는 과정을 통해 학습됩니다. 그러나 embedding 벡터가 원점에서 멀리 떨어진 작은 부분 공간에 군집화되는 비등방성을 관찰하였습니다. Adam 최적화 기법은 희소 데이터에서 잘 작동하지만, 이로 인해 자주 등장하는 단어에 비해 드물게 등장하는 단어의 업데이트 벡터가 상대적으로 늘어나 비등방성을 야기합니다.

- **Performance Highlights**: Coupled Adam을 사용한 결과, 단어 임베딩의 품질을 유의미하게 향상시키는 것을 입증했습니다. 이 방법은 또한 충분히 큰 데이터셋에서 상류(upstream) 및 하류(downstream) 성능을 개선하는 데 긍정적인 영향을 미치는 것으로 나타났습니다.



### From Haystack to Needle: Label Space Reduction for Zero-shot Classification (https://arxiv.org/abs/2502.08436)
Comments:
          Under review at ICML 2025

- **What's New**: 이번 논문에서는 Label Space Reduction (LSR)이라는 새로운 방법을 제시하여 대형 언어 모델(LLMs)의 제로샷 분류 성능을 향상시키고자 합니다. LSR은 후보 클래스를 체계적으로 순위화하고 감소시켜 모델이 가장 관련성이 높은 옵션에 집중할 수 있도록 합니다. 이 방법은 라벨 공간 표현을 동적으로 최적화하여 실험을 통해 Llama-3.1-70B와 Claude-3.5-Sonnet에서 각각 7.0%와 3.3%의 매크로 F1 점수 개선을 가져왔습니다.

- **Technical Details**: LSR은 분류 라벨 공간을 개선하기 위해 후보 클래스를 순위화하고 감소시키는 혁신적인 반복 시스템을 개발합니다. 또한, LSR의 계산 오버헤드를 줄이기 위해 모델을 확률적 분류기로 증류하는 방안을 제안하여 효율적인 추론을 가능하게 합니다. 이 방법은 출력이 생성될 때 동적으로 적응할 수 있는 정성적인 라벨 공간을 생성합니다.

- **Performance Highlights**: 여덟 개의 벤치마크에서 실험한 결과, LSR은 기존 제로샷 분류 기준에 비해 Llama-3.1-70B에서 평균 7.0%(최대 14.2%) 및 Claude-3.5-Sonnet에서 3.3%(최대 11.1%)의 매크로 F1 점수 향상을 보여주었습니다. 이러한 성과는 LLM이 중요한 정보에 보다 효과적으로 주의를 집중할 수 있도록 해주며, 문제 해결 과정에서의 추론을 개선하는데 기여합니다.



### Semantic Learning for Molecular Communication in Internet of Bio-Nano Things (https://arxiv.org/abs/2502.08426)
Comments:
          4 pages, 3 figures, 1 table

- **What's New**: 이 논문에서는 생체 나노 사물인터넷(Internet of Bio-Nano Things, IoBNT)에서 효율적이고 신뢰성 있는 분자 통신(molecular communication, MC)을 최적화하기 위한 새로운 개념인 종단 간 의미론적 학습 프레임워크를 제안합니다. 이 프레임워크는 생리학적 진단 과제에 중점을 두고 자원 제약 조건 하에서 작동하도록 설계되었습니다. 특히 딥 인코더-디코더 아키텍처를 활용하여 의미 체계를 효율적으로 추출, 양자화 및 디코딩합니다.

- **Technical Details**: 제안된 시스템은 단일 입력 단일 출력(SISO) 시스템으로, 3차원 비한정 환경에서 작동합니다. 분자는 확산(difussion) 및 유선형 흐름(advection)의 영향을 받아 이동하며, 포지셔닝된 송신기(transmitter)와 수신기(receiver)가 동기화된 상태로 정보를 주고받습니다. 이 시스템은 비트 수준의 전송에 집중하지 않고 과제에 관련된 의미를 우선시하여 진단 효율성을 개선하는 방향으로 설계되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 의미론적 프레임워크는 전통적인 JPEG 압축과 LDPC 코딩 방법에 비해 적어도 25% 이상의 진단 정확도를 향상시켰습니다. 이는 자원이 제한된 통신 상황에서도 뚜렷한 성능 향상으로 이어지며, 생물학적 시스템에서의 복잡한 데이터 전송을 지원하는 데 중요한 성과로 평가됩니다. 논문은 분자 전파(물질이동)의 확률적 특성을 근거로 한 네트워크를 통해 이 과정을 가능하게 했습니다.



### Multifidelity Simulation-based Inference for Computationally Expensive Simulators (https://arxiv.org/abs/2502.08416)
- **What's New**: 이번 논문에서는 MF-NPE라는 다중 정밀도(multi-fidelity) 접근 방식을 소개합니다. 이 방법은 저비용의 저정밀도 시뮬레이션을 활용하여 고비용의 고정밀도 시뮬레이터의 매개변수를 추론하는 데 중점을 두고 있습니다. MF-NPE는 전이 학습(transfer learning)을 이용하여 제한된 고정밀도 자원으로도 효과적인 후방 추정을 수행할 수 있는 능력을 제공하며, 개인 관측치를 우선순위에 따라 활용하는 능동 학습(active learning)을 통합합니다.

- **Technical Details**: MF-NPE는 저정밀도 시뮬레이션에서 신경 밀도 추정기(neural density estimator)를 미리 훈련시켜, 소수의 고정밀도 시뮬레이션으로 추론을 정제합니다. 이 접근 방식은 고정밀도 매개변수의 후방 추정 효율성을 높이기 위한 전이 학습을 기반으로 합니다. 또한, MF-NPE는 대상 매개변수 공간 탐색을 목표로 설정한 두 가지 능동 학습 스킴을 제안하여 효율성을 극대화합니다.

- **Performance Highlights**: MF-NPE는 분석적 기준 진리(ground truth)가 있는 통계 작업과 두 개의 실제 유전학 시뮬레이션에서 기존 방법들과 비교해도 유사한 성능을 보이며, 고정밀도 시뮬레이션의 수를 최대 100배까지 줄일 수 있음을 보여줍니다. 이 방법은 고비용의 시뮬레이터에 대해 효율적인 베이지안 추론(Bayesian inference)을 가능하게 하여, 실증적 데이터에 대한 이해를 향상시킬 수 있는 기회를 제공합니다.



### Sparse Estimation of Inverse Covariance and Partial Correlation Matrices via Joint Partial Regression (https://arxiv.org/abs/2502.08414)
- **What's New**: 이번 연구에서는 고차원 희소 부분 상관관계(partial correlation) 및 역 공분산(inverse covariance) 행렬을 추정하기 위한 새로운 방법을 제안합니다. 이 방법은 역 공분산 행렬과 선형 회귀(linear regression) 간의 연결성을 활용하는 2단계 추정 기법입니다. 제안된 방법은 고차원 데이터에서도 일관된 수렴 속도를 보이며, 기존 방법들을 개선합니다.

- **Technical Details**: 제안하는 추정 방법은 기존의 방법들과 달리 선형 회귀를 통한 희소한 정밀 행렬을 동시에 추정합니다. 또한, 이 방법은 오염된 데이터에 잘 적용될 수 있는 희소 Huber 회귀(sparse Huber regression)로 강건한 확장을 제공합니다. 고차원에서의 통계적 추정 오류률은 정밀 행렬 추정에 대한 최적의 속도와 부분 상관 행렬 추정에 대한 기존 속도를 초과합니다.

- **Performance Highlights**: 제안된 방법은 합성 데이터(synthetic data)와 실제 데이터(real-world data)에서 그래픽 라소(ℓ1-regularized maximum likelihood estimation)보다 우수한 추정 오류를 보여줍니다. 또한, 주식 시장 데이터에 적용하여 데이터의 기초 구조를 효과적으로 포착하는 능력을 입증했습니다. 이 방법의 구현은 Rust로 작성된 Python 패키지 형태로 제공됩니다.



### Strong bounds for large-scale Minimum Sum-of-Squares Clustering (https://arxiv.org/abs/2502.08397)
- **What's New**: 이번 연구에서는 Minimum Sum-of-Squares Clustering (MSSC) 문제에 대한 새로운 접근 방식을 제안합니다. MSSC의 계산적 어려움을 해결하기 위해, 우리는 최적성 차이를 통해 휴리스틱 MSSC 솔루션을 검증하는 방법을 도입하였습니다. 이를 위해 문제를 더 작은 인스턴스로 분해하는 분할 정복(divide-and-conquer) 전략을 사용하며, 이를 지원하기 위한 보조 최적화 문제인 'anticlustering 문제'를 설계하였습니다.

- **Technical Details**: MSSC는 데이터 포인트와 클러스터 중심 간의 제곱 유클리드 거리의 총합을 최소화하는 것을 목표로 하는 비지도 학습 클러스터링 기법입니다. 이 방법은 Mixed-Integer Nonlinear Programming (MINLP) 문제로 형식화되며, 이산 결정을 내리는 변수들이 사용됩니다. 특히, 이 문제는 NP-hard에 해당하여 대규모 데이터에 대해서는 정확한 해법을 찾는 것이 비현실적입니다.

- **Performance Highlights**: 우리가 제안한 방법은 대규모 인스턴스에 대해서도 효과적이며, 대부분의 경우 최적성 차이를 3% 이내로 유지합니다. 컴퓨팅 실험에 따르면, 이 접근법은 대규모 데이터셋에 대한 클러스터링 솔루션 평가에서 실용 가능성을 보여주며, MSSC 평가에 있어 중요한 간극을 메우는 역할을 하고 있습니다.



### Learning Humanoid Standing-up Control across Diverse Postures (https://arxiv.org/abs/2502.08378)
Comments:
          Humanoid Standing-up Control, 12 pages

- **What's New**: 이 논문에서는 HoST(Humanoid Standing-up Control)라는 강화학습(framework) 기반의 새로운 시스템을 제안합니다. 이는 기존의 시뮬레이션 한계를 극복하고, 실제 환경에서도 다양한 자세에서 로봇이 스탠딩 업(standing-up)할 수 있는 능력을 향상시키는 데 중점을 두고 있습니다. 또한, 여러 지형에서의 커리큘럼 기반 훈련을 통해 다양한 상황에서의 적응력을 보장합니다.

- **Technical Details**: HoST는 멀티-크리틱 아키텍처(multi-critic architecture)와 스무스니스 정규화(smoothness regularization)를 활용하여 로봇의 움직임이 부드럽고 안정적으로 이루어지도록 합니다. 이를 통해 물리적 하드웨어에서의 진동이나 충동적인 움직임을 줄이는 동시에, 다양한 실내 및 실외 환경에서의 실제 배치도 가능합니다. 특히, 강화학습을 위한 보상 구조를 최적화하기 위해 다단계의 과정을 구성하여 접근합니다.

- **Performance Highlights**: 실험 결과, HoST에 의해 훈련된 제어 정책은 다양한 실험 환경에서도 높은 부드러움과 안정성을 보여줍니다. 이 연구는 기존의 정해진 경로를 따르지 않고도 효과적인 스탠딩 업 모션을 성공적으로 구현함으로써, 향후 휴머노이드 로봇의 실용적 응용 가능성을 넓힐 수 있는 기반을 마련하고 있습니다.



### Graph Foundation Models for Recommendation: A Comprehensive Survey (https://arxiv.org/abs/2502.08346)
- **What's New**: 이 논문은 추천 시스템(Recommender Systems, RS)에서의 최신 연구 동향을 탐구하고 있으며, 특히 그래프 기반 모델(Graph Foundation Models, GFMs)에 대한 종합적인 개요를 제공합니다. GFMs는 그래프 신경망(Graph Neural Networks, GNNs)과 대형 언어 모델(Large Language Models, LLMs)의 강점을 결합하여 복잡한 추천 문제를 해결하는 새로운 접근 방식을 제시합니다. 이 접근법은 사용자-항목 관계의 그래프 구조와 텍스트 이해를 결합하여 추천의 정확성을 향상시킵니다.

- **Technical Details**: 그래프 기반 추천 시스템의 발전은 GNN을 활용한 협업 필터링의 기초를 다지며, 텍스트 정보와 사용자 선호도 간의 관계를 도출하는 데 중점을 둡니다. 그러나 GNN은 본질적으로 구조적 편향을 지니고 있어 텍스트 정보를 처리하는 데 한계를 겪습니다. 반면, LLM은 자연어 처리(Natural Language Processing, NLP) 분야에서 강력한 성능을 보이며, 추천 시스템 내에서 사용자와 항목의 텍스트 정보를 효과적으로 캡처합니다.

- **Performance Highlights**: GFM 기반의 추천 시스템은 데이터 활용 측면에서 효율성을 극대화하고 사용자 선호도를 정밀하게 조정함으로써 발생하는 편향을 최소화합니다. 이 시스템은 그래프 구조에서의 중요 정보를 적절히 통합하여 추천의 새로운 패러다임으로 자리잡을 잠재력을 가지고 있습니다. 앞으로 GFM을 활용한 기술 발전은 개인화된 추천의 질을 한층 향상시키는 데 기여할 것입니다.



### Improving Existing Optimization Algorithms with LLMs (https://arxiv.org/abs/2502.08298)
- **What's New**: 본 논문에서는 대형 언어 모델(LLMs)이 최적화 문제에 어떻게 기여할 수 있는지를 탐구합니다. LLM의 사전 훈련된 지식을 활용하여 기존의 최적화 알고리즘을 개선하는 방안을 제시합니다. 이는 혁신적인 heuristic 변형과 구현 전략을 제안하는 데 중점을 둡니다.

- **Technical Details**: 우리는 Construct, Merge, Solve and Adapt (CMSA)라는 비유형의 최적화 알고리즘을 사용하여 LLM의 효과를 평가했습니다. CMSA는 조합 최적화 문제를 위한 하이브리드 메타휴리스틱으로, 해결책 구축 단계에서 heuristic을 통합합니다. LLM의 능력을 통해 제안된 대안 heuristic이 CMSA의 전문가 설계 heuristic보다 뛰어난 성능을 나타냈습니다.

- **Performance Highlights**: GPT-4o의 대안 heuristic은 더 크고 조밀한 그래프에서 성능 차이가 더욱 두드러지는 것으로 나타났습니다. 이는 LLM이 최적화 알고리즘의 성능을 크게 향상시킬 수 있음을 시사합니다. 이 연구는 LLM을 활용한 최적화 분야의 새로운 가능성을 열어줍니다.



### Data Pricing for Graph Neural Networks without Pre-purchased Inspection (https://arxiv.org/abs/2502.08284)
Comments:
          Accepted by AAMAS-2025

- **What's New**: 이번 연구는 Graph Neural Networks (GNNs) 훈련에 필요한 데이터 가격 책정의 새로운 접근 방식을 제안한다. 특히, 비공식 데이터 전송 없이 데이터 소유자에게 적절한 보상을 제공하기 위한 Structural Importance based Model Trading (SIMT) 메커니즘을 도입하였다. SIMT는 데이터 소유자가 제공하는 데이터의 구조적 중요성을 평가하고, 모델 훈련 과정에서 선택적 노드를 고려하여 효과적인 학습을 가능하게 한다.

- **Technical Details**: SIMT는 두 가지 주요 단계를 포함하는 새로운 모델 마켓플레이스 프레임워크이다: 데이터 조달 및 모델 훈련. 데이터 조달 단계에서는 marginal structural entropy라는 새로운 방법을 사용하여 노드의 정보성을 정량화하고, 이를 경매 메커니즘과 통합하여 데이터 소유자에게 공정한 보상을 지급한다. 모델 훈련에서는 누락된 노드의 특성 데이터를 보완하기 위해 feature propagation 방법을 도입하고, 선택되지 않은 노드와의 연결을 추가함으로써 그래프 구조의 향상을 도모한다.

- **Performance Highlights**: SIMT는 다섯 개의 잘 알려진 기준 데이터셋에서 평가되었으며, 노드 분류 작업에서 네 가지 기준 메커니즘보다 일관되게 더 나은 성능을 보여주었다. 특히, SIMT는 MacroF1 및 MicroF1 점수 측면에서 최대 40%의 향상을 기록하여 기존 Greedy 및 ASCV 방법과 비교하여 우수한 성능을 입증하였다. 이는 다양한 예산 제약 하에서도 성능의 안정성을 나타낸다.



### Dealing with Annotator Disagreement in Hate Speech Classification (https://arxiv.org/abs/2502.08266)
- **What's New**: 본 논문은 증오 발언(hate speech) 분류에 대한 주석자 간의 의견 불일치 문제를 심층적으로 다루고, 여러 가지 접근 방식을 평가합니다. 특히, 주석자 간의 불일치 문제 해결을 위한 새로운 방법론을 제안하며, 고품질의 데이터셋을 확보하기 위한 다양한 전략을 탐구합니다. 이 연구는 터키어 트윗을 기반으로 하여 필터링된 BERT 모델을 활용한 최신 성능 벤치마크 결과를 제공합니다.

- **Technical Details**: 본 연구는 주석 과정을 통해 발생하는 주관적 불일치 문제에 집중하며, 다양한 방법(예: 최대값, 최소값, 무작위 선택 및 평균)을 통해 가장 정확한 레이블을 결정하는 방안을 모색합니다. 또한 주석자의 신뢰도 차이를 고려하여 가중된 버전의 접근 방법도 평가합니다. 이를 통해 데이터셋의 품질을 향상시키고, 나아가 증오 발언 탐지 모델의 신뢰성을 높이는 데 기여하고자 합니다.

- **Performance Highlights**: 필요한 트레이닝 데이터 확보의 중요성을 강조하며, 제안된 방법론을 통해 공연별 감지 및 이해에서 최첨단 성과를 달성했습니다. 이 연구는 튼튼한 데이터셋을 바탕으로 한 정확한 증오 발언 탐지의 필요성을 재확인하고, 이를 통해 다양한 자연어 처리(NLP) 작업에서의 성능을 향상시키는 데 기여합니다.



### Multi-View Oriented GPLVM: Expressiveness and Efficiency (https://arxiv.org/abs/2502.08253)
Comments:
          8 pages

- **What's New**: 이번 연구에서는 다중 뷰 데이터에서 통합된 표현을 학습하기 위해 Multi-View Gaussian Process Latent Variable Model (MV-GPLVM)을 제안합니다. 새로운 NG-SM 커널(Next-Gen Spectral Mixture kernel)을 도입하여 커널의 표현력을 높이고, 계산 효율성을 개선하기 위해 Random Fourier Feature (RFF) 근사를 사용합니다. 이 모델은 다양한 다중 뷰 데이터 세트에서 의미 있는 잠재 표현을 지속적으로 학습하는 성능을 입증하였습니다.

- **Technical Details**: 연구에서 제안한 NG-SM 커널은 스펙트럼 밀도와 커널 함수 사이의 새로운 이중성을 기반으로 합니다. 커널 하이퍼파라미터에 대한 효율적인 두 단계 리파라미터화 트릭을 통합함으로써, 계산 효율성을 극대화하고 대규모 데이터 세트에서도 효과적으로 작동합니다. 이를 통해 MV-GPLVM 모델은 각 뷰의 고유한 특성을 캡처하며 의미 있는 통합 잠재 표현을 생성할 수 있습니다.

- **Performance Highlights**: 제안된 MV-GPLVM 모델은 다양한 교차 도메인 다중 뷰 데이터 세트에서 검증되었으며, SOTA(MV-VAEs, MV-GPLVMs 포함) 모델들보다 일관되게 우수한 성능을 보였습니다. 진짜 데이터에서 이 모델은 복잡한 데이터 세트의 구조를 잘 포착하고 정보가 풍부한 통합 잠재 표현을 생성하는 데 성공하였습니다.



### TRISHUL: Towards Region Identification and Screen Hierarchy Understanding for Large VLM based GUI Agents (https://arxiv.org/abs/2502.08226)
Comments:
          Under review at ICML 2025, 8 pages 5 figures

- **What's New**: TRISHUL은 기존 LVLM보다 더 포괄적인 GUI 이해를 위해 개발된 새로운 프레임워크입니다. 이 시스템은 Hierarchical Screen Parsing (HSP)와 Spatially Enhanced Element Description (SEED) 모듈을 활용하여 행위 지향(mapping instructions to GUI elements) 및 GUI 참조(task) 작업을 통합적으로 처리합니다. 기존 방법들이 주로 특정 작동(task)에 특화되어 있었던 반면, TRISHUL은 다양한 GUI 상호작용 작업을 지원하고, 훈련이 필요 없는 방식으로 설계되었습니다.

- **Technical Details**: HSP 모듈은 GUI 요소를 Global Regions of Interest (GROIs) 및 Local Elements (LE)로 구분하여 데이터의 위계적 이해를 가능하게 합니다. SEED 모듈은 GUI의 요소들 간의 상대적 위치를 분석하여 각각의 요소에 대한 고수준의 기능 설명을 생성합니다. 이 프로세스는 SAM 및 EasyOCR 알고리즘을 활용하여 GUI 내 텍스트 및 아이콘을 효과적으로 처리합니다.

- **Performance Highlights**: TRISHUL은 ScreenSpot, VisualWebBench, Mind2Web 및 AITW 데이터셋에서 기존의 최첨단 기법들을 초월하는 성능을 보여줍니다. 특히, TRISHUL 기반의 GPT-4V와 GPT-4o는 action grounding과 episodic instruction-following 작업에서 우수한 성과를 발휘하였습니다. 또한, Screen PR 데이터셋에서도 GUI 참조 성능을 향상시켜 타겟의 접근성과 사용자 상호작용 피드백을 개선하였습니다.



### LLM Modules: Knowledge Transfer from a Large to a Small Model using Enhanced Cross-Attention (https://arxiv.org/abs/2502.08213)
Comments:
          Code and pre-trained weights available at this https URL

- **What's New**: 본 연구에서는 대형 사전 훈련 모델에서 소형 모델로 지식을 전달할 수 있는 LLM 모듈 아키텍처를 제안합니다. Enhanced Cross-Attention 메커니즘을 사용하여, Qwen2-1.5B 모델의 표현을 고안된 주의 레이어를 통해 GPT-Neo-125M 모델로 전달합니다. 실험 결과, 15 에포크의 훈련 후 결합 모델이 증류(distillation)로 얻은 품질과 비교할 만한 응답을 생성함을 보여주었습니다.

- **Technical Details**: 제스템의 핵심 요소는 Cross-Attention 레이어의 수정된 형태로, 대형 모델의 표현 차원을 소형 모델에 맞게 변환하는 선형 투영과 비선형 변환을 제공하는 Adapter Block, 원래 표현과 외부 지식을 동적으로 혼합하는 게이팅 메커니즘을 포함합니다. 이 접근법은 Qwen2-1.5B 모델의 동결된 가중치를 통해 입력 쿼리의 풍부한 표현을 추출하면서, 소형 모델이 외부 표현을 결합하여 응답을 생성하도록 합니다.

- **Performance Highlights**: 결합 모델은 15 에포크 훈련 후 손실이 처음 에포크에서는 13.8에서 2.3으로, 이후 에포크에서는 1.1로 감소하며 성공적인 수렴을 확인했습니다. 실험 동안, 결합 모델은 원본 소형 모델에 비해 품질이 현저히 향상되었으며, 생성된 응답의 구조화 및 논리적 일관성이 더 높음을 나타냈습니다.



### From Individual Experience to Collective Evidence: A Reporting-Based Framework for Identifying Systemic Harms (https://arxiv.org/abs/2502.08166)
- **What's New**: 이 논문에서는 개인의 부정적인 경험이 더 넓은 시스템 동작 패턴과 어떻게 연결될 수 있는지를 연구합니다. 특히, 시스템과의 해로운 상호작용을 경험할 가능성이 높은 하위 그룹을 식별하는 방법을 제안합니다. 저자들은 이를 순차적 가설 검정(sequential hypothesis testing)으로 공식화하고 보고 행동에 대한 조건을 정의하여 진정한 해악을 추론할 수 있는 방법을 제시합니다.

- **Technical Details**: 이 연구에서는 개인 보고서를 시간에 따라 축적하여 집단적 지식으로 발전시키는 사고방식에 기반하여 incident databases를 이용한 공정성 감사(fairness auditing)를 제안합니다. 실질적으로, 저자들은 모기지(mortgage) 결정 및 백신(vaccine) 부작용 데이터셋을 통해 그들의 방법의 유용성을 입증하며, 보고된 데이터를 사용하여 부당한 피해를 입은 그룹을 성공적으로 식별합니다. 이는 단순한 데이터 수집(Commentary) 이상의 의미를 지니며, AI 시스템의 공정성을 실질적으로 평가하는 데 중요한 변화를 가져올 수 있습니다.

- **Performance Highlights**: 저자들은 실제 데이터셋을 사용하여 제안한 알고리즘이 알려진 불균형을 성공적으로 식별할 수 있음을 보였습니다. 특히, 비교적 적은 수의 보고서로도 비례적으로 해를 입은 그룹을 찾을 수 있어, 데이터 활용의 효율성을 강조합니다. 이러한 진전은 AI 시스템의 공정성을 모니터링하고 개선하기 위한 새로운 방법론적 접근으로, 공공의 신뢰를 구축하는 데 기여할 것으로 기대됩니다.



### Local Differential Privacy is Not Enough: A Sample Reconstruction Attack against Federated Learning with Local Differential Privacy (https://arxiv.org/abs/2502.08151)
- **What's New**: 본 논문은 Local Differential Privacy (LDP) 기반의 Federated Learning (FL)에서 샘플 복원 공격을 새롭게 제안합니다. 기존의 공격들이 LDP 하의 FL에서 효과가 없다는 점을 강조하며, 클리핑(clipping)과 노이즈(noise)로 인한 정보 손실을 해결하는 방법을 보여줍니다. 이 공격은 피해자의 민감한 샘플을 재구성할 수 있음을 이론적으로 입증합니다.

- **Technical Details**: 제안된 공격 방법은 두 가지 핵심 기술에 기반합니다. 첫 번째로, gradient compression을 통해 필요 없는 기울기를 줄여 LDP의 노이즈 효과를 최소화합니다. 두 번째로, reconstructed sample의 노이즈를 제거하는 방법으로 zero gradients를 Introduce하여 noise distribution을 관찰하고, confidence interval을 조정하여 노이즈를 필터링합니다.

- **Performance Highlights**: 실험 결과는 제안된 공격이 LDP와 함께 사용되는 FL에서 타겟 모델의 정확성에 미치는 영향이 최소화되며, 피해자의 훈련 샘플 정보를 효율적으로 재구성할 수 있음을 보여줍니다. 이는 LDP와 FL의 조합이 완전하지 않음을 나타내며, FL의 향상을 위해 추가 방어책이 필요함을 시사합니다.



### SS4Rec: Continuous-Time Sequential Recommendation with State Space Models (https://arxiv.org/abs/2502.08132)
- **What's New**: 본 논문에서는 사용자 상호 작용의 불규칙한 시간 간격을 고려한 연속 시간 순차 추천을 위한 하이브리드 상태 공간 모델인 SS4Rec을 제안합니다. SS4Rec은 시간 인식 SSM과 관계 인식 SSM을 통합하여 사용자 관심을 다각적으로 추론합니다. 이러한 접근법은 사용자 아이템 전환 모델링의 복잡성을 해결하고, 사용자 맞춤형 추천을 제공할 수 있는 가능성을 열어줍니다.

- **Technical Details**: SS4Rec은 사용자 상호 작용의 시간 간격에 맞춰 변수를 통해 이산화되는 시간 인식 SSM과 관계 인식 SSM을 활용합니다. 시간 인식 SSM은 가변 관찰 간격을 처리하고 시간 패턴을 포착하기 위한 효과적인 인코더로 기능합니다. 이러한 방법을 통해 SS4Rec은 불규칙한 시간 간격의 연속적인 의존성을 포착하고, 이를 통해 개인화된 추천을 가능하게 합니다.

- **Performance Highlights**: 다섯 개의 벤치마크 데이터셋에 대한 실험 결과, SS4Rec은 기존 최신 모델들에 비해 우수한 성능을 보였습니다. 이 모델은 연속적인 사용자 관심이 불규칙한 시간 간격에서 진화하는 복잡성을 효과적으로 해결할 수 있는 것으로 나타났습니다. 또한, SS4Rec은 긴 시퀀스와 대규모 데이터셋을 전처리하는 데 있어 혁신적인 효율성을 보임으로써 향후 추천 시스템의 연구 가능성을 높였습니다.



### Incremental Approximate Single-Source Shortest Paths with Predictions (https://arxiv.org/abs/2502.08125)
- **What's New**: 본 논문에서는 업데이트된 엣지를 통해 증가하는 그래프에서 근사 최단 경로를 유지하는 근본적인 데이터 구조 문제를 연구하였다. 알고리즘-예측(algorithms-with-predictions) 모델을 활용하여, 예측된 엣지 시퀀스에 기반하여 데이터 구조의 상태를 초기화하는 방법을 제시한다. 이로 인해 단일 소스에 대한 (1+ϵ)-근사 최단 경로를 유지하는 학습된 알고리즘이 설계되었다.

- **Technical Details**: 제안된 알고리즘은 예측 오류(η)에 따라 성능이 좌우되며, 최악의 경우에도 오프라인 알고리즘과 비슷한 성능을 발휘한다. 알고리즘의 실행 시간은 O(mη log W/ϵ)으로, 여기서 m은 엣지의 수, W는 최대 엣지의 가중치, ϵ는 정밀도를 나타낸다. 데이터 구조는 엣지가 하나씩 추가될 때마다 소스에서 모든 정점으로의 근사 최단 경로를 효율적으로 계산한다.

- **Performance Highlights**: 예측이 거의 완벽할 경우, 알고리즘은 오프라인 알고리즘에 가깝게 작동하며, 예측 오류가 커질수록 성능이 매끄럽게 감소한다. 결과적으로 이 알고리즘은 예측이 정확한 경우 뿐만 아니라 다양한 상황에서도 분포된 성능을 보여준다. 또한 모든 쌍 최단 경로(all-pairs shortest path) 문제로 일반화될 수 있다는 점에서 그 유용성이 강조된다.



### Provably Robust Federated Reinforcement Learning (https://arxiv.org/abs/2502.08123)
Comments:
          To appear in The Web Conference 2025

- **What's New**: 이번 연구에서는 Byzantine-robust (비잔틴 강건성) FRL에 대한 최초의 모델 포이징 공격인 Normalized attack을 제안합니다. 이 공격은 악성 에이전트가 집계된 정책 업데이트의 차이를 극대화하도록 설계되었습니다. 또한, 본 연구에서는 공격에 대한 방어로 입증된 안전성을 가진 앙상블 FRL 방식도 개발하였습니다.

- **Technical Details**: 우리의 Normalized attack은 기존의 비잔틴 강건한 집계 기법이 방향성을 무시하는 문제를 해결하기 위해, 집계된 정책 업데이트 간의 각도 차이를 극대화하는 데 중점을 둡니다. 이를 해결하기 위해 우리는 최적화 문제를 두 단계로 접근하여 악성 정책 업데이트의 최적 방향과 크기를 결정합니다. 앙상블 방식에서는 에이전트를 해시 값을 통해 여러 그룹으로 나누고, 각 그룹이 독립적으로 글로벌 정책을 훈련합니다.

- **Performance Highlights**: 실험 결과, Normalized attack은 비앙상블 기반의 강건 집계 규칙에 큰 피해를 줄 수 있는 반면, 우리의 앙상블 FRL 방법은 모든 고려된 포이징 공격에 효과적으로 방어할 수 있음을 보여주었습니다. 특히, 앙상블 방법은 공격이 있는 상황에서도 비공격 상황의 FedAvg와 유사한 성과를 유지합니다.



### Hookpad Aria: A Copilot for Songwriters (https://arxiv.org/abs/2502.08122)
Comments:
          Extended abstract presented in the Late-Breaking Demo Session at ISMIR 2024 (ISMIR LBD 2024)

- **What's New**: Hookpad Aria는 음악 작곡을 돕기 위한 생성적 AI 시스템으로, 기존의 Hookpad 플랫폼에 통합되어 사용됩니다. 이 시스템은 사용자가 새로운 곡을 작곡할 때 필요로 하는 다양한 기능을 제공하여, 곡의 멜로디와 하모니를 생성하거나 기존 자료를 기반으로 내용을 추가하는 식으로 작동합니다. 2024년 3월 출시 이후, Aria는 3천명의 사용자에게 318,000개의 제안을 했으며, 이 중 74,000개가 실제 곡에 채택되었습니다.

- **Technical Details**: Hookpad Aria는 Anticipatory Music Transformer라는 대형 언어 모델에 기반하여, 다중 악기의 상징적 음악 생성을 위해 사전 훈련되었습니다. 이 모델은 기존 문맥에 따른 autoregressive 방식의 생성과 중간 영역을 채우는 기능을 동시에 지원합니다. 또한, Hookpad 환경에 맞게 MIDI 노트를 기능적 표현으로 변환하는 인코딩 체계를 설계하였으며, 비트 단위로 시간 표시를 할 수 있도록 클릭 트랙 악기를 추가했습니다.

- **Performance Highlights**: Hookpad Aria 사용자들과의 인터뷰를 통해 아이디어 창출과 창작 과정에서의 활용이 효과적이라는 결과가 도출되었습니다. 사용자는 Aria를 창작의 동료로 여기며, 창작 블록 시 새로운 출발점을 얻는 데 도움을 받다고 응답했습니다. 그러나 사용자들은 보다 세분화된 제어를 원하는 것으로 나타났으며, 장르, 감정 톤, 의도된 악기, 구조적 요소와 같은 더욱 상세한 설정 기능을 요구하고 있습니다.



### Multi-Agent Performative Prediction Beyond the Insensitivity Assumption: A Case Study for Mortgage Competition (https://arxiv.org/abs/2502.08063)
- **What's New**: 이 논문에서는 Performative Prediction 모델이 어떻게 피드백 루프(feedback loops)를 고려하여 의사결정 과정에 영향을 미치는지를 다룹니다. 특히 경쟁적인 멀티 에이전트(multi-agent) 환경에서 예측이 미래 데이터 분포에 미치는 영향을 분석합니다. 저자들은 'Bank Game'이라는 특정 게임을 설정하여, 이론적 결과와 함께 실험적 검증을 수행합니다.

- **Technical Details**: 저자들은 두 대출자가 이자율(interest rates)과 신용 점수(thresholds)로 경쟁하는 구조를 탐구합니다. 이 게임은 소비자들이 가장 낮은 이자율을 제공하는 업체를 선택하는 방식으로, Bertrand Competition과 유사합니다. 공통의 자연스러운 No-Regret 학습(dynamic)인 exponential weights를 사용하여 안정된 결과로 수렴함을 보여줍니다.

- **Performance Highlights**: 실험적 결과는 저자들이 제안한 이론적 결과의 유효성을 입증합니다. 이 설정은 여러 개의 안정적 균형을 허용하며, 수렴은 초기 조건에 따라 달라진다는 점이 주목할 만합니다. 또한, utility matrix가 완전히 알려지지 않은 경우에서도 수렴 결과를 제공하며, 작은 규정 미스는 견딜 수 있는 강인성을 나타냅니다.



### On Mechanistic Circuits for Extractive Question-Answering (https://arxiv.org/abs/2502.08059)
- **What's New**: 이번 논문은 문서 처리 및 질문 응답을 위한 대형 언어 모델의 기계적 회로를 추출하여, 질문-응답 과제에 대한 실제 응용을 탐구합니다. 중심 주제는 컨텍스트에 기반한 언어 모델링 기법을 통해 데이터 기여도를 평가하고, 모델의 컨텍스트 충실도를 향상시키는 것입니다. 또한, ATTNATTRIB라는 데이터 기여도 알고리즘을 도입하여 여러 추출적 질문-응답 기준에서 최첨단 결과를 보여줍니다.

- **Technical Details**: 이 논문에서는 인과 매개 분석(Causal Mediation Analysis, CMA)을 활용하여 대형 언어 모델의 기계적 회로를 추출합니다. 두 가지 주요 회로인 컨텍스트 충실 회로(Context-Faithfulness Circuit)와 메모리 충실 회로(Memory-Faithfulness Circuit)를 설계하고, 각 회로가 질의 응답 작업에서 어떻게 작동하는지 분석합니다. ATTNATTRIB 알고리즘을 통해 주목 헤드를 사용하여 데이터 기여도를 신뢰성 있게 평가할 수 있음을 보여주며, 단일 양방향 통과로 기여도를 달성할 수 있습니다.

- **Performance Highlights**: 실험 결과, ATTNATTRIB는 다양한 추출적 질문-응답 데이터 세트에서 최첨단 기여도 결과를 나타냅니다. 기여도 정보를 모델의 추가 입력으로 활용함으로써 질문-응답의 정확성을 최대 9%까지 향상시키는 효과를 입증합니다. 이러한 성과는 언어 모델이 컨텍스트로부터 더 나은 답변을 제공할 수 있도록 하는 데 중점을 두며, 전반적으로 언어 모델의 실용적인 응용 가능성을 강조합니다.



### General Coded Computing: Adversarial Settings (https://arxiv.org/abs/2502.08058)
Comments:
          18 pages, 1 figure

- **What's New**: 전통적인 코딩 컴퓨팅 프레임워크는 주로 구조화된 계산(structured computation)용으로 설계되어 있었지만, 이 논문은 이러한 시스템의 적용 범위를 일반화합니다. 이를 통해 다양한 계산을 처리할 수 있도록 코딩 컴퓨팅의 기초를 마련하며, 특히 적대적 서버(adversarial server) 문제를 다루는 데 중점을 두고 있습니다. 제안된 방식에서 N개의 서버가 있는 시스템에서 O(N^a)의 적대적 서버가 있을 경우, 평균 근사 오류의 최대치를 N^{(6/5)(a-1)}의 비율로 감소시킬 수 있음을 증명합니다.

- **Technical Details**: 제안된 방식에서는 입력 데이터를 인코딩 함수에 임베딩(embedding)하여 코드화된 입력(coded inputs)을 워커 노드(worker nodes)로 분배합니다. 이 과정에서 서버의 일부는 적대적 서버일 수 있으며, 결과 복구를 위한 복호화 함수는 여러 계산 작업을 다룰 수 있도록 설계되어 있습니다. 특히, 이 연구는 제2차 소벨(Sobolev) 공간이라는 매끄러운 함수 집합을 사용하여 인코딩 및 복호화 함수의 효율성을 보장합니다. 이 접근 방식은 최소한의 가정 하에 일반 계산 작업을 수용할 수 있습니다.

- **Performance Highlights**: 이 논문의 검증 결과는 깊은 신경망(deep neural networks)에서의 추론(inference)을 포함한 복잡한 계산을 처리하는 데 있어 제안된 방법의 효능을 입증합니다. 또한, 이 시스템의 구조 내에서 제안된 방식은 다른 어떤 시스템보다 더 많은 적대적 워커 노드를 견딜 수 있는 최적성을 보여줍니다. 이러한 성과는 현재의 분산 시스템 내에서 신뢰성을 높이고 적대적인 환경을 관리하는 데 중요한 기여를 하게 됩니다.



### SLVR: Securely Leveraging Client Validation for Robust Federated Learning (https://arxiv.org/abs/2502.08055)
Comments:
          29 pages

- **What's New**: 본 논문에서는 SLVR이라는 프레임워크를 제안하여 Federated Learning(FL)에서 개인 클라이언트의 데이터를 안전하게 활용하여 업데이트의 유효성을 검증할 수 있는 방법을 제공합니다. 기존 기술들이 공공 검증 데이터에 의존하던 점을 극복하여, SLVR은 다양한 클라이언트의 개인 데이터를 통해 통계치를 계산하고 모델의 강인성을 높이는 데 기여합니다. 이로 인해 더욱 유연한 적용이 가능하며, 클라이언트 데이터의 분포 변화에 자연스럽게 적응할 수 있습니다.

- **Technical Details**: SLVR은 안전한 다자간 계산(Secure Multi-Party Computation, MPC)을 기반으로 하여 클라이언트의 비공개 데이터를 이용해 다양한 유효성 검사(checks)를 수행할 수 있는 일반 프레임워크입니다. SLVR은 여러 클라이언트의 개인 데이터를 활용하여 유효성 검사에 필요한 공개 데이터를 필요로 하지 않으며, 교차 클라이언트 정확도 검증(cross-client accuracy validation)과 같은 보다 다양한 체크를 가능하게 합니다. SLVR의 모듈 설계는 기존 MPC 구현과 호환성이 높으며, 다양한 위협 모델(threat models)에서 유연한 배치를 허용합니다.

- **Performance Highlights**: SLVR은 모델 오염 공격(model poisoning attacks)으로부터의 강인성을 크게 향상시키며, 특히 적응형 공격(adaptive attacks) 하에서 기존 방법보다 최대 50% 더 우수한 성능을 나타냅니다. 다양한 데이터 분포 변화 시나리오에서 SLVR의 적응력(adaptability)과 안정적인 수렴(stable convergence)을 실험적으로 입증하였고, 이는 이전 연구가 겪었던 심각한 정확도 저하 문제(e.g., MNIST에서 SVHN으로의 30% 정확도 감소)를 상회합니다.



### COMBO-Grasp: Learning Constraint-Based Manipulation for Bimanual Occluded Grasping (https://arxiv.org/abs/2502.08054)
Comments:
          14 pages, 11 figures

- **What's New**: COMBO-Grasp(Constraint-based Manipulation for Bimanual Occluded Grasping)는 환경적 제약으로 인해 발생하는 결합된 로봇 그립 문제에 대한 해결책을 제시합니다. 인간의 이중 손 조작 전략에서 영감을 받아 두 개의 정책을 통해 안정화 및 재배치를 동시에 수행하는 기법을 개발했습니다. 이 시스템은 자가 감독 데이터셋을 통해 훈련된 제약 정책과 강화 학습(RL)을 통해 학습된 그립 정책의 협력을 통해 작업 효율성을 높였습니다.

- **Technical Details**: COMBO-Grasp에서는 두 가지 정책, 즉 안정화 포즈를 생성하는 제약 정책과 목표 객체를 재배치하고 그립하는 RL 기반의 그립 정책을 도입합니다. 제약 정책은 합성 데이터셋에서 훈련되며, 그립 정책은 RL 교육 중 가치 함수와의 연계를 통해 최적화됩니다. 이러한 정책 간의 협력은 RL 프로세스를 가속화하고 동시에 효율적인 데이터 사용을 가능하게 합니다.

- **Performance Highlights**: 실험적 검증 결과, COMBO-Grasp은 경쟁하는 기준 접근법 대비 작업 성공률을 크게 향상시켰습니다. 또한, 이 방법은 보지 못한 객체에 대해서도 시뮬레이션과 실제 환경 모두에서 성공적으로 일반화되었습니다. COMBO-Grasp의 방법론은 로봇이 환경적 제약을 극복하고 다방면에서 최적의 그립을 제공하는 데 크게 기여하고 있음을 입증합니다.



### End-to-End Predictive Planner for Autonomous Driving with Consistency Models (https://arxiv.org/abs/2502.08033)
- **What's New**: 이번 논문은 자율 주행 차량의 궤적(trajectory) 예측과 계획(planning)을 통합한 새로운 데이터 기반 프레임워크를 제안합니다. 기존의 방법들은 예측과 계획 모듈이 서로 독립적으로 운영되어 상호작용에 비효율적이었던 반면, 우리의 일관성 모델(consistency model)은 혼합 궤적 분포를 활용하여 적극적 대처(interactive behaviors)를 생성합니다. 이를 통해 안전하고 효율적인 도로 사용자 간의 상호작용을 가능하게 합니다.

- **Technical Details**: 연구에서 제안하는 방법은 높은 차원의 멀티모달(joint trajectory distributions) 궤적 분포로부터 새로운 샘플을 생성하며, 이를 실시간 자율 주행에 적합하도록 설계했습니다. 특히, Waymo Open Motion Dataset(WOMD)에서 훈련 받은 일관성 모델은 프로액티브한 궤적 계획을 통해 복잡한 상호작용을 효과적으로 포착할 수 있습니다. 또한, 여러 목표를 고려하여 온라인 샘플링 할 때 추가적인 계획 제약을 통합하기 위해 교대 방향(method for multi-objective guidance)을 도입했습니다.

- **Performance Highlights**: 실험 결과, 일관성 모델을 기반으로한 Planner는 예측과 계획 모듈 간의 반복 과정을 줄이고 단일 추론(single inference)으로 효과적인 상호작용을 만들어내는 것으로 나타났습니다. 특히, 전반적인 궤적 품질과 제약 만족도를 기존의 여러 방법들과 비교했을 때, 뛰어난 성능을 보였습니다. Diffusion 모델보다 적은 샘플링 단계로 나은 결과를 도출하여 실시간 작동의 요구를 충족시키는데 더 적합한 것으로 평가됩니다.



### Optimizing Likelihoods via Mutual Information: Bridging Simulation-Based Inference and Bayesian Optimal Experimental Design (https://arxiv.org/abs/2502.08004)
Comments:
          Preprint. Under Review

- **What's New**: 이번 연구에서 우리는 Bayesian Optimal Experimental Design (BOED)을 Simulation-based Inference (SBI) 응용에 통합하는 새로운 방법인 SBI-BOED를 제안합니다. 이 방법은 상호 정보 경계를 통해 SBI와 BOED 간의 연결을 보여주며, 실험 디자인과 추론 함수의 최적화를 동시에 수행할 수 있도록 합니다. 또한, 실험 디자인을 최적화하는 과정에서 발생할 수 있는 pitfalls을 정리하고, 잘 정의된 디자인 분포의 유용성을 입증합니다.

- **Technical Details**: SBI는 모델 매개변수의 조건부 분포를 추론하는 데 중점을 두며, 이 과정에서 likelihood p(y|θ), prior p(θ)와 같은 입력 데이터를 사용합니다. BOED는 주어진 실험 디자인에 대한 정보 이득을 평가하여 실험을 최적화하며, Expected Information Gain (EIG)을 통해 실험 디자인을 평가합니다. 본 논문에서는 MI(상호 정보)를 이용하여 추론 물체를 동시에 최적화하며, InfoNCE MI 하한을 극대화함으로써 새롭게 정의된 INCE-λ를 소개합니다.

- **Performance Highlights**: BOED와 SBI를 통합한 SBI-BOED 방법은 실제 세계의 다양한 분야, 특히 전염병과 생물학 모델에서 뛰어난 성능을 보여주었습니다. 우리는 이 방법이 기존의 최첨단 기술들보다 개선된 추론 성능을 달성할 수 있음을 실험적으로 검증하였습니다. 실험 디자인 최적화와 더불어 정확한 예측을 위한 실험 데이터 수집의 중요성을 강조하고 있습니다.



### Unveiling Client Privacy Leakage from Public Dataset Usage in Federated Distillation (https://arxiv.org/abs/2502.08001)
Comments:
          14 pages, 10 figures

- **What's New**: 이 논문에서는 Public Dataset-Assisted Federated Distillation (PDA-FD) 프레임워크의 개인정보 보호 리스크를 심도 있게 분석합니다. PDA-FD는 공개 데이터셋을 활용하여 클라이언트들이 비공유 방식으로 지식을 공유할 수 있게 설계되었습니다. 하지만 저자들은 이러한 접근 방식이 여전히 클라이언트의 개인 훈련 데이터에 심각한 개인정보 유출 위험을 초래할 수 있음을 입증합니다.

- **Technical Details**: PDA-FD는 클라이언트들이 각각의 개인 데이터셋 위에서 모델을 훈련한 후 공개 데이터셋에서 추론을 수행하고 결과를 서버에 전달하는 방식으로 작동합니다. 이 과정은 클라이언트의 개인 데이터로부터 배운 지식을 다른 클라이언트들에게 전달할 수 있게 해주지만, 이로 인해 클라이언트의 개인 데이터셋에 대한 정보가 노출될 위험이 매우 높아집니다. 논문에서는 특히 LDIA(라벨 분포 추론 공격)와 MIA(회원 추론 공격)라는 두 가지의 공격 방법을 제시하여 이러한 리스크를 실질적으로 평가합니다.

- **Performance Highlights**: 저자들은 세 가지 대표적인 PDA-FD 프레임워크인 FedMD, DS-FL, 그리고 Cronus를 대상으로 공격을 평가하였으며, 해당 공격들이 차별화된 성능을 보임을 확인했습니다. 라벨 분포 공격은 KL 발산 최소값에 도달하며, 회원 추론 공격은 낮은 거짓 긍정률 조건에서도 높은 진양성률을 유지하는 것으로 나타났습니다. 이러한 결과는 현재 PDA-FD 프레임워크에서의 중요한 개인정보 보호 위험성을 시사하며, 협업 학습 시스템에서 더욱 강력한 프라이버시 보호 메커니즘의 필요성을 강조합니다.



### What is a Sketch-and-Precondition Derivation for Low-Rank Approximation? Inverse Power Error or Inverse Power Estimation? (https://arxiv.org/abs/2502.07993)
- **What's New**: 이번 논문에서는 랜덤화된 스케치 방법이 대규모 수치 선형 대수 문제의 계산 복잡성을 줄이는 방법을 제안합니다. 전통적인 스케치-앤-솔브(skit-and-solve) 접근 방식 대신, 스케치-앤-프리컨디셔너(sketch-and-precondition) 방법을 통해 계산 친화적인 프리컨디셔너를 구성하여 원래 문제의 반복 해법(iterative solver)의 수렴 속도를 개선합니다. 이는 전체 공간에서의 정확성을 유지하면서 이루어집니다.

- **Technical Details**: 제안된 방법, 즉 Error-Powered Sketched Inverse Iteration (EPSI) Method는 스케치된 뉴턴 반복(run sketched Newton iteration)을 통해 이루어지며, 랜덤화된 저랭크(low-rank) 근사화에 적용됩니다. 이 방법은 스케치 크기에 따라 최소한 선형적으로 개선되는 수렴 속도를 포함하여 이론적인 보증(theoretical guarantees)을 달성합니다. 또한, 스케치 크기가 증가함에 따라 솔버의 수렴 속도가 개선된다는 점에서 중요한 의미를 갖습니다.

- **Performance Highlights**: 이 연구는 랜덤화된 알고리즘 개발에서 저랭크 행렬 근사화(low-rank matrix approximation)를 위한 도전 과제를 다루고 있으며, 이를 통해 스케치 기반의 새로운 프리컨디셔닝 체계가 가능함을 보여줍니다. EPSI 방법은 기존의 접근 방식보다 나은 성능을 보일 것으로 기대되며, 다양한 수치 선형 대수 문제에 적용 가능성을 시사합니다.



### Sink equilibria and the attractors of learning in games (https://arxiv.org/abs/2502.07975)
- **What's New**: 이 논문은 게임 이론에서 학습 동역학의 한계 행동을 특성화하는 문제에 대해 중요한 진전을 이룹니다. 특히, 복제기 동역학(replicator dynamic)의 흡인체(attractor)와 게임의 sink 평형 간의 관계를 연구합니다. 저자들은 이들 두 개념이 반드시 일대일 대응이 아니며, 특히 이론적으로 두 개의 플레이어가 있는 게임에서 조건부 관계를 제시합니다.

- **Technical Details**: 저자들은 토폴로지적 구성을 통해 흡인체가 sink 평형과 일대일 대응하지 않음을 입증했습니다. 또한, 두 플레이어 게임에서는 약한 지역 소스(weak local source)가 없을 때 이들이 일대일로 대응한다는 증거를 제공합니다. 논문은 Fictitious Play(FP)라는 오래된 학습 동역학에 대해서도 탐구하며, FP에서의 흡인체가 항상 sink 평형을 포함한다는 사실을 확인합니다.

- **Performance Highlights**: 이 연구는 전통적인 Nash 평형 대신 흡인체와 sink 평형 간의 관계를 탐구함으로써 게임 동역학의 분석 접근 방식을 전환합니다. 또한, FP의 경우에서 흡인체가 간혹 sink 평형의 엄격한 부분 집합일 수 있지만 여전히 강하게 연결된 서브 그래프와의 관계를 가진다는 두 가지 주요 결과를 제안합니다. 마지막으로, 저자들은 이러한 결과들이 게임에서의 학습 이해에 많은 새로운 경로를 열어준다고 주장합니다.



### From Hazard Identification to Controller Design: Proactive and LLM-Supported Safety Engineering for ML-Powered Systems (https://arxiv.org/abs/2502.07974)
Comments:
          Accepted for publication at the International Conference on AI Engineering (CAIN) 2025

- **What's New**: 이 논문은 머신러닝(ML) 기반 소프트웨어 제품의 개발에 안전 분석(hazard analysis)을 통합해야 한다고 주장합니다. 특히, LLM(대규모 언어 모델)을 활용하여 안전 분석 과정을 반자동화하고, 이를 통해 기존의 복잡한 안전 엔지니어링 작업을 경량화할 수 있는 방법을 제시합니다. 이렇게 함으로써, 경험이 부족한 개발자들도 위험을 사전에 예방하고 완화할 수 있는 방안을 마련하려고 합니다.

- **Technical Details**: 이 논문에서는 전통적인 안전 분석 기법인 Failure Mode and Effects Analysis (FMEA)와 System Theoretic Process Analysis (STPA)를 사용하여 머신러닝 모델의 다양한 위험을 체계적으로 식별할 수 있다고 설명합니다. STPA는 복잡한 시스템을 분석하는 데 적합하며, 기술적 요소와 인간적 요소를 모두 고려할 수 있는 장점이 있습니다. 이 과정에서 LLM은 안전 엔지니어링 전문 지식이 부족한 엔지니어와 데이터 과학자를 지원하여 보다 포괄적인 사고를 촉진할 수 있습니다.

- **Performance Highlights**: 논문에서는 STPA를 기반으로 한 해석적 접근 방식이 머신러닝 응용 프로그램에서 다양한 위험을 예측하고 이를 통해 안전 요구 사항을 수립할 수 있음을 보여줍니다. 특히, 이 방법은 모델과 시스템 수준에서의 위험 완화를 위한 여러 제어 구조를 설계하는 데 효과적입니다. LLM을 활용함으로써, 이러한 프로세스는 보다 접근 가능하고 관리 가능하게 되어, 실제 개발 주기에 통합될 수 있을 것으로 기대됩니다.



### ReTreever: Tree-based Coarse-to-Fine Representations for Retrieva (https://arxiv.org/abs/2502.07971)
- **What's New**: 이 논문에서는 문서 검색의 비효율성과 복잡성을 해결하기 위해, ReTreever라는 새로운 트리 기반 문서 표현 방법을 제안합니다. 이 방법은 문서를 여러 수준으로 체계적으로 조직하여 비용과 유용성을 균형 있게 조절할 수 있는 유연함을 제공합니다. 기존의 시스템에서는 고차원 임베딩을 사용하여 메모리와 계산 자원이 많이 소모되었으나, ReTreever는 이를 효율적으로 개선합니다.

- **Technical Details**: ReTreever는 이진 트리의 각 내부 노드에서 라우팅 함수(routing function)를 학습하여 쿼리 문서와 참조 문서를 유사한 트리 분기로 매핑합니다. 이는 검색 성능을 직접 최적화하는 방식으로 동작하며, 일반적인 인코딩 모델(BERT 등)을 사용해 문서 스니펫을 임베딩으로 변환합니다. 이 과정에서 LLM 호출 없이도 트리를 구축하고 내비게이션할 수 있는 기능을 제공합니다.

- **Performance Highlights**: ReTreever의 평가 결과, 이 방법은 높은 정확도를 유지하면서도 낮은 지연 시간(latency)에서 최상의 검색 정확도를 달성하였습니다. 또한, 이 구조는 문서의 의미론적 그룹화를 간접적으로 학습하여 투명성과 해석 가능성을 높이는 데 기여합니다. 결과적으로 ReTreever는 대규모 데이터 세트에서도 효율적인 문서 검색을 가능하게 하여 실용적 응용에 적합합니다.



### Federated Self-supervised Domain Generalization for Label-efficient Polyp Segmentation (https://arxiv.org/abs/2502.07951)
Comments:
          Accepted at ADSMI @ MICCAI 2024

- **What's New**: 본 논문에서는 자가 지도 학습(self-supervised learning, SSL) 방법과 연합 학습(federated learning, FL)을 결합하여 장에서의 폴립 세분화를 개선하는 LFDG(Learned Federated Domain Generalization) 방법을 제안합니다. 이 방법은 의료 데이터의 프라이버시 문제를 해결하는 동시에 다양한 데이터 세트에 대한 일반화 능력을 향상시키는 것을 목표로 합니다. LFDG는 데이터 다양성을 높이기 위해 SSADA(자기 지도 대적 데이터 증강) 방법과 SRAM(원천 재구성 및 증강 마스킹) 모듈을 도입하여 모델의 성능을 향상시킵니다.

- **Technical Details**: LFDG는 각 클라이언트에서 SSL 방법을 사용해 로컬 데이터셋에 대해 훈련한 후, 서버에서 learned parameters를 집계하는 구조를 가지고 있습니다. 이 과정에서는 FedAvg 알고리즘을 통해 각 클라이언트의 모델 파라미터를 평균 업데이트하여 전역 모델을 구성합니다. SSADA 방법은 데이터 증강을 극대화하고 DropPos 프리트레인 손실을 최소화하여 안정적인 표현을 유지하면서 주요 데이터 확장을 진행합니다.

- **Performance Highlights**: LFDG는 여섯 개의 의료 센터에서 수집된 폴립 이미지 데이터에서 검증되었으며, 기존의 FL 및 SSL 방법에 비해 각각 3.80% 및 3.92% 더 나은 성능을 보였습니다. 또한, 제안된 방법은 최상의 평균 IoU(Intersection over Union) 성능인 62.83%를 달성하며, SSADA 및 SRAM 모듈이 모델 성능을 각각 2.50% 및 1.30% 향상시킵니다.



### SurGrID: Controllable Surgical Simulation via Scene Graph to Image Diffusion (https://arxiv.org/abs/2502.07945)
- **What's New**: 이번 논문에서는 SurGrID라는 새로운 Surgical simulation 모델을 소개합니다. 이 모델은 Scene Graphs(SGs)를 활용하여 제어 가능한 수술 장면 생성을 가능하게 합니다. 기존의 Denoising Diffusion Models(DDMs)에 비해 사용자 평가 연구에서 높은 현실감과 제어 가능성을 보였습니다.

- **Technical Details**: SurGrID는 수술 장면의 공간적 및 의미적 정보를 담고 있는 SG를 사용합니다. 이 모델은 수술 비디오를 통해 훈련되며, 새로운 수술 장면을 고충실도의 제어 하에 합성할 수 있도록 설계되었습니다. 특히, 공간 정보를 SG의 노드 특징으로 인코딩하여 세부적인 생성 제어를 가능하게 합니다.

- **Performance Highlights**: 본 연구에서는 백내장 수술 장면에서 생성된 이미지를 정량적으로 평가하였으며, 높은 충실도와 다양성을 달성했습니다. 또한, 임상 전문가들이 참여한 사용자 연구를 통해 생성된 시뮬레이션의 매우 현실적인 재현과 제어 가능성이 확인되었습니다. 이는 수술 시뮬레이션 분야에 새로운 가능성을 제시합니다.



### Symbiotic Cooperation for Web Agents: Harnessing Complementary Strengths of Large and Small LLMs (https://arxiv.org/abs/2502.07942)
- **What's New**: 본 연구에서는 AgentSymbiotic이라는 새로운 반복적(framework) 접근 방식을 제안합니다. 이 방법은 데이터 합성과 작업 성능(task-performance)을 결합하여 대형 LLM과 소형 LLM 모두를 위한 '공생적 개선(symbiotic improvement)'을 제공합니다. 연구를 통해 대형 LLM과 소형 LLM 간의 상호 보완적(dynamic) 관계를 밝혀냈습니다.

- **Technical Details**: AgentSymbiotic의 핵심 요소는 대형 LLM이 고품질의 궤적(trajectory)을 생성하고, 소형 LLM이 고유한 추론(reasoning) 능력으로 새로운 궤적을 탐색하여 데이터의 다양성을 극대화하는 것입니다. 새로운 데이터 합성 전략(speculative data synthesis)과 다중 작업 학습(multi-task learning) 접근 방식을 통해 소형 LLM의 추론 능력을 향상시키고 비정책 편향(off-policy bias)을 완화합니다. 또한 Hybrid Mode를 도입하여 사용자 개인정보 보호(privacy preservation) 문제를 해결합니다.

- **Performance Highlights**: WEBARENA 벤치마크에서 AgentSymbiotic은 모든 LLM 유형에 대해 최고 성능(SOTA performance)을 달성했습니다. 우리의 최상 대형 LLM 에이전트는 52%의 성능을 기록했으며, 이전 최고 성능인 45%를 초과했습니다. 또한, 8B로 압축된 모델은 49%의 경쟁력 있는 성능을 보여주며, 이전 최고 성능인 28%를 초과하는 성과를 올렸습니다.



### Discrete Markov Probabilistic Models (https://arxiv.org/abs/2502.07939)
- **What's New**: 이 논문은 Discrete Markov Probabilistic Model (DMPM)을 소개하며, 이 모델은 이산 데이터 생성에 대한 새로운 알고리즘입니다. DMPM은 비트 공간 $\{0,1\}^d$에서 작동하며, 노이징 과정은 포아송 시계를 통해 균등하게 무작위로 레이블을 뒤집는 연속 시간의 마르코프 체인입니다. 이 연구는 기존 이산 생성 모델에 대한 이론적 기초와 실제 응용을 연결하여 효과적이고 이론적으로 근거가 있는 이산 생성 모델링의 발전을 이끌고자 합니다.

- **Technical Details**: DMPM은 고유한 전방 노이징 및 후방 디노이징 과정의 도출을 통해 생성 모델링 프레임워크를 설정합니다. 이 모델은 전통적인 확산 기반 방법을 이산 상태 공간에 적합하게 수정하여 구성되며, 시간 반전 마르코프 동역학에 대한 이론적 통찰력을 활용하여 기존의 점진적 생성 모델의 강점을 유지합니다. 이 과정에서는 조건부 기댓값 형태의 점수 함수와 안정적인 추정이 통합되어 수치적으로 불안정한 전이 확률 비율을 제거합니다.

- **Performance Highlights**: 이 알고리즘은 바이나리화된 MNIST와 같은 이산 데이터셋에서 우수한 성능을 나타내며, 경쟁 모델들과 비교하여 더 적은 함수 평가로도 뛰어난 결과를 보여줍니다. 예를 들어, Discrete Flow Matching에 비하여 약 2.89의 FID 점수를 달성하며, 네트워크 호출 수가 2.5배나 줄어드는 효율성을 자랑합니다. 이러한 결과는 DMPM이 이산 구조 생성에서 강력한 성능을 제공함을 시사합니다.



### Sign Operator for Coping with Heavy-Tailed Noise: High Probability Convergence Bounds with Extensions to Distributed Optimization and Comparison Orac (https://arxiv.org/abs/2502.07923)
- **What's New**: 이 논문은 심각한 왜곡 데이터에 대한 AI 최적화 문제의 요구가 증가하고 있는 가운데, 무거운 꼬리 노이즈(heavy-tailed noise)를 효과적으로 처리할 수 있는 방법인 SignSGD를 제안합니다. 특히 추가적인 하이퍼파라미터를 도입할 필요 없이 입력의 부호만을 사용하여 이러한 노이즈에 대응할 수 있음을 보여줍니다. 또한, 동기화된 노이즈에 대한 다수결(Majority Voting)의 사용을 통해 분산 최적화(Distributed Optimization) 또는 단일 작업자(single worker)의 경우 샘플 복잡성을 줄이는 방법을 제안합니다.

- **Technical Details**: 두 가지 최적화 기법이 소개됩니다. SignSGD는 매끄러운 비볼록(non-convex) 함수에 대해 최적 샘플 복잡성을 $	ilde{O}(	ext{ε}^{-rac{3	ext{κ}-2}{	ext{κ}-1}})$로 보장합니다. 또한, 대칭 노이즈를 가정할 경우, SignSGD와 다수결 기법을 조합하여 분산 최적화의 상황에서도 비슷한 성능을 보입니다. 새로운 방법인 MajorityVote-CompsSGD는 대칭 노이즈의 가정 하에 비교 수의 고확률 경계를 제공하며 샘플 복잡성을 $	ilde{O}(	ext{ε}^{-6})$로 줄입니다.

- **Performance Highlights**: 이 연구의 이론적 발견은 대규모 언어 모델(LLM) 훈련 시 부호 기반 방법의 우수한 성능에 의해 뒷받침됩니다. SignSGD는 기존의 기법들에 비해 더 나은 평균 경량 노름 정확도를 달성하는 것으로 나타났습니다. 이로 인해, 심각한 왜곡 데이터의 최적화 문제를 효율적으로 해결할 수 있는 새로운 가능성을 제시하고 있습니다.



### DeepSeek on a Trip: Inducing Targeted Visual Hallucinations via Representation Vulnerabilities (https://arxiv.org/abs/2502.07905)
Comments:
          19 pages, 4 figures

- **What's New**: 이 연구는 DeepSeek Janus 모델에서 타겟 이미지 생성과 이를 통한 시각적 환각을 유도하는 방법론을 체계적으로 실험하여 98.0%의 환각률을 달성했습니다. 이는 모델의 시각-언어 처리 파이프라인에서 발생하는 취약점을 드러내며, 복잡한 멀티모달 환경에서의 보안 위협을 강조합니다. 또한 LLaMA-3.1을 활용하여 환각 탐지 프레임워크를 도입하여 다양한 평가 지표를 제공하며, 이상 작동 감지의 필요성을 부각시킵니다.

- **Technical Details**: DeepSeek Janus는 1B와 7B 파라미터로 구성된 멀티모달 모델로, 훈련 과정에서의 공격에 대한 기대가 필요합니다. 특히, 이미지 임베딩을 조작하여 환각을 유도하는 전략을 개발하였으며, 이는 이미지의 원본 유사성을 유지하면서도 모델이 특정 타겟 임베딩을 생성하도록 유도하는 방식으로 이루어집니다. 이 연구에서는 환각 탐지의 정밀한 평가를 위해 새로운 LSD-Hallucination 벤치마크 데이터셋을 개설하였습니다.

- **Performance Highlights**: 실험 결과, COCO, DALL-E 3, SVIT 데이터셋에서 DeepSeek Janus 모델의 환각 발생률이 최대 98%에 달하며, 조작된 이미지의 시각적 충실도는 SSIM 0.88 이상을 유지하였습니다. 이는 근본적으로 오픈소스 모델이 가진 보안 리스크를 명확하게 드러내며, 실용적인 AI 적용시 고도의 주의가 필요함을 나타냅니다. 타겟 이미지 간의 조작에서 고도의 강도를 유지하면서도 모델의 반응을 극대화하는 방법론을 통해, 기존에 비해 보다 혁신적인 접근을 제시합니다.



### The Observational Partial Order of Causal Structures with Latent Variables (https://arxiv.org/abs/2502.07891)
Comments:
          48 pages, 30 figures

- **What's New**: 본 논문은 관측적 지배(observational dominance) 개념을 통해 잠재 변수(latent variable)가 있는 인과 구조(causal structure) 집합의 부분 순서(partial order) 정의에 중점을 두고 있습니다. 이는 두 개의 인과 구조에서 하나가 다른 것을 관측적으로 지배하는 경우, 첫 번째 인과 구조에서 실현 가능한 확률 분포의 집합이 두 번째의 집합을 포함하는 경우를 뜻합니다. 즉, 세 개의 가시 변수(visible variable)와 네 개의 가시 변수에 대한 지배 관계를 완벽하게 규명하고, 비고립 제약(nontrivial inequality constraints)이 적용된 관측적 동치 클래스(equal equivalence classes)를 식별하는 기술적인 방법을 제시합니다.

- **Technical Details**: 이 연구에서는 가시 변수 세 개에 대한 인과 구조의 지배 질서를 완전하게 특징짓고 네 개에 대해서는 부분적으로 특징짓습니다. 또한, Bell 불등식(Bell inequalities) 및 도구적 불등식(instrumental inequalities)과 유사한 비고립 제약을 통해 실현 가능한 분포 집합을 구분할 수 있는 관측 동치 클래스(observational equivalence classes)를 찾습니다. 이러한 분석을 통해 가시 변수의 수가 증가할수록 비고립 제약으로 만족되는 동치 클래스가 유사하게 증가하는 경향을 발견했습니다.

- **Performance Highlights**: 이 연구 결과는 조건부 독립성 제약(conditional independence constraints)만을 사용하는 기존의 제약 기반 인과 발견(causal discovery) 알고리즘의 상대적인 분별력이 약하다는 것을 보여줍니다. 이러한 알고리즘은 불리한 분포를 위한 인과 구조 탐색에서 약점이 있으며, 논의된 새로운 기법을 통해 더 정밀한 관측적 동치 분류가 가능해질 것으로 기대됩니다. 데이터가 부족한 경우에도 이 연구의 관측적 부분 순서 개념을 통해 인과 설명의 범위를 좁힐 수 있는 실용적인 방법론을 제공합니다.



### A unifying account of warm start guarantees for patches of quantum landscapes (https://arxiv.org/abs/2502.07889)
- **What's New**: 이번 연구에서는 양자 손실 경관(quantum loss landscapes)에서 발견되는 바렌 플래토(barren plateau)의 일반적인 경계를 제시합니다. 기존 연구에서 검토된 특정 파라미터화된 양자 회로(parameterized quantum circuits)의 사례들을 통합하는 결과물입니다. 이러한 경계는 이전에 분석할 수 없었던 물리적으로 동기 부여된 ansätze를 다룰 수 있는 가능성을 제공합니다.

- **Technical Details**: 연구자는 손실의 분산(variance)에 대한 하한(lower-bound)을 분석적으로 증명하여, 곡률이 있는 지점 주위에서 비지수적으로 좁은 영역에서는 손실 분산이 기하급수적으로 빠르게 감소할 수 없음을 보여줍니다. 이러한 결과는 수치적(Numerical) 분석과 결합되어 특정 radius subregion 안에서도 기하급수적으로 감소하는 기울기(gradients)가 존재함을 시사하고 있습니다.

- **Performance Highlights**: 이 연구는 변분 양자 알고리즘(variational quantum algorithms)을 초기화(initiate)하는 전략이 문제 사이즈가 커짐에 따라 더 가까운 지역으로 접근하지 못한다면, 적합하지 않을 것이라는 점을 강조합니다. 이는 바렌 플래토를 가진 손실 함수가 존재할 경우 기울기가 기하급수적으로 감소할 것이라는 결론을 내리게 합니다.



### SNAP: Sequential Non-Ancestor Pruning for Targeted Causal Effect Estimation With an Unknown Graph (https://arxiv.org/abs/2502.07857)
Comments:
          Accepted at AISTATS 2025

- **What's New**: 이 논문에서는 많은 변수들에 대한 인과 발견(causal discovery) 작업의 계산 부담을 경감하는 새로운 접근법을 제시합니다. 특정 타겟 변수에 대한 인과 효과를 추정하는 데 집중하면서 모든 변수의 인과 그래프를 배우는 대신 필요한 부분만을 고려합니다. 이를 통해 인과 관계를 효율적으로 식별하는 방법을 모색하고 있습니다.

- **Technical Details**: 저자들은 타겟 변수와 그 조정 집합(adjustment sets) 간의 인과 효과를 알리기 위해 Sequential Non-Ancestor Pruning (SNAP) 프레임워크를 사용합니다. 이 방법은 정의된 비조상(definite non-ancestors) 변수를 식별하고 제거하여 인과 관계를 간소화합니다. SNAP는 기존 인과 발견 방법에 대한 전처리 단계로 사용할 수도 있고, 독립적으로 작동 가능한 완전한 인과 발견 알고리즘으로도 기능합니다.

- **Performance Highlights**: 실험 결과는 SNAP 프레임워크가 독립성 테스트(independence tests) 수와 계산 시간을 크게 줄이면서도 인과 효과 추정의 품질을 유지함을 보여줍니다. 합성 데이터와 실제 데이터를 통해 확인된 이러한 효율성 향상은 다양한 인과 분석 작업에서의 유용성을 강조합니다.



### MRS: A Fast Sampler for Mean Reverting Diffusion based on ODE and SDE Solvers (https://arxiv.org/abs/2502.07856)
Comments:
          Accepted by ICLR 2025

- **What's New**: 이 논문은 Mean Reverting (MR) Diffusion 모델의 샘플링 효율성을 개선하는 새로운 알고리즘인 MRS (MR Sampler)를 제안합니다. 기존의 빠른 무작위 샘플러가 MR Diffusion에 직접 적용되지 않는 한계를 극복하여 적은 단계로도 고품질 샘플을 생성할 수 있도록 합니다. MRS는 역방향 확률 미분 방정식 및 확률 흐름 일반 미분 방정식을 해결하여 반-정리 해를 도출하고, 이를 활용하여 빠른 샘플링을 가능하게 합니다.

- **Technical Details**: MR Sampler는 MR Diffusion에서 발생하는 역방향 확률 미분 방정식 (PF-ODE)과 SDE를 해결하여 샘플링 공식을 생성합니다. 이 방법은 분석 함수와 신경망에 의해 매개변수화된 적분을 포함하여 반-정리 해의 형태로 구성됩니다. 제안된 알고리즘은 노이즈 예측, 데이터 예측 및 속도 예측을 포함한 모든 주요 매개변화를 지원하고, 훈련이 필요 없으며 다양한 작업과 데이터 세트에 적응할 수 있습니다.

- **Performance Highlights**: MR Sampler는 10개의 서로 다른 이미지 복원 작업에서 10배에서 20배의 속도 향상을 달성하면서도 높은 샘플링 품질을 유지하는 것으로 나타났습니다. 이 알고리즘은 샘플링 절차를 가속화하여 MR Diffusion에서 실用성을 높이는 데 중요한 기여를 합니다. 고품질 샘플을 적은 수의 함수 평가(NFEs)로 생성할 수 있어, 효율적인 이미지 및 비디오 생성에 유용하게 사용될 수 있습니다.



### Technical note on calibrating vision-language models under covariate shif (https://arxiv.org/abs/2502.07847)
- **What's New**: 본 연구는 저샷 비전 분류(vision classification)에서 발생하는 두 가지 주요 문제인 covariate shift와 confidence misalignment를 동시에 해결하는 새로운 접근법인 Confidence-Calibrated Covariate Shift Correction (C3SC)를 제안합니다. 기존의 개별적인 접근법과 달리, C3SC는 Fisher 정보 기반 페널티를 사용하는 통합 프레임워크로, 이는 일반화 능력을 크게 향상시킵니다. 실험 결과 C3SC는 다양한 데이터셋에서 5.82% 향상된 보정 성능을 보여줍니다.

- **Technical Details**: C3SC는 covariate shift 보정을 위해 Fisher 정보 페널티를 활용합니다. 또한, misclassified 예제에 대한 신뢰도를 낮추기 위해 confidence misalignment penalty (CMP)를 통합합니다. 이 두 가지 페널티는 CLIP의 대조 손실(contrastive loss)에 통합되어 안정적이고 신뢰할 수 있는 학습 시스템을 제공합니다.

- **Performance Highlights**: 실험 결과 C3SC는 covariate shift가 존재하는 어려운 데이터셋에서도 정확도에서 3.5%의 향상을 보이며, 전반적으로 신뢰할 수 있는 비전-언어(Vision-Language) 저샷 응용 프로그램에 유망한 솔루션으로 자리매김합니다. 특히, 저샷 학습(learning) 환경에서 데이터의 한계로 인해 발생하는 부정확한 예측을 줄이는데 효과적입니다.



### Memory Analysis on the Training Course of DeepSeek Models (https://arxiv.org/abs/2502.07846)
- **What's New**: 이번 연구에서는 DeepSeek 모델의 훈련 동안 GPU 메모리 소비에 대한 이론적 분석을 제공합니다. DeepSeek-v2 및 DeepSeek-v3와 같은 다양한 분산 훈련 설정과 관련된 장치 수준의 메모리 요구 사항을 명확히 하는 것이 주요 목표입니다. 특히, 마이크로 배치 크기, 활성화 재계산 정책, 3D 병렬성 및 ZeRO 최적화와 같은 메모리 사용에 영향을 미치는 주요 요소를 조사합니다.

- **Technical Details**: DeepSeek-v3의 구조적 세부 사항을 분석하며, 이 모델은 현재 오픈 소스 모델 중에서 최첨단을 자랑합니다. 전체 아키텍처는 61개의 레이어로 구성되어 있으며, 각 레이어에는 두 개의 RMSNorm 작업과 Multi-Head Latent Attention (MLA) 블록, 선형 레이어가 포함되어 있습니다. DeepSeek-v3의 훈련 시 메모리 소비를 FP16/BF16 형식으로 분석하는 데 주력합니다.

- **Performance Highlights**: 모델 매개변수 수를 자세히 분석하여 각 구성 요소의 메모리 발자국을 MB 및 GB 단위로 나타냅니다. 최대 메모리 사용량을 측정하기 위해 PP16 파이프라인 병렬 구성에서 극대화된 매개변수 볼륨을 포함하는 파이프라인 단계를 식별합니다. PF16@TP2 병렬 구성을 사용하여 특정 단계의 총 매개변수량과 메모리 요구 사항을 평가하여 메모리 분석의 기초를 제공합니다.



### Column-wise Quantization of Weights and Partial Sums for Accurate and Efficient Compute-In-Memory Accelerators (https://arxiv.org/abs/2502.07842)
- **What's New**: 최근에 발표된 논문에서는 compute-in-memory (CIM) 방식의 딥 뉴럴 네트워크(DNN) 구현 시 발생하는 비교적 큰 아날로그-디지털 변환기(ADC) 오버헤드를 해결하기 위한 새로운 방법을 제안하고 있습니다. 기존의 저정밀 ADC를 사용하더라도 부분합(partial-sum) 양자화에서 발생하는 오류로 정확도가 저하되는 문제를 해결하기 위해, 가중치와 부분합 양자화의 세분화 수준을 맞춘 독창적인 접근법을 도입했습니다. 이를 통해 정확도 향상과 함께 하드웨어 효율성도 유지되는 효과를 보여줍니다.

- **Technical Details**: 이 연구에서는 가중치와 부분합 양자화의 세부 조정을 열(column)-기반으로 진행하며, 각 열에 대해 독립적인 스케일 팩터를 적용하여 메모리 셀 변동성에 대한 강인성을 확보합니다. 또한, 새로운 타일링 기법과 그룹 컨볼루션을 이용하여 CIM 지향 컨볼루션 프레임워크를 제안, 연산의 비효율성을 해소합니다. 이러한 접근 방식은 이전 방법들과 비교 시 재훈련이 필요 없는 효율적인 양자화 체계를 제공합니다.

- **Performance Highlights**: ResNet-20 모델을 CIFAR-10 및 CIFAR-100 데이터셋으로, ResNet-18 모델을 ImageNet 데이터셋으로 실험한 결과, 기존 최고의 방법에 비해 각각 0.99%, 2.69%, 1.01%의 정확도 향상을 달성했습니다. 또 다른 분석을 통해 제안한 방법이 메모리 셀 변동성에 대해 높은 강인성을 보임을 확인하였으며, CIM 기반 DNN 구현에서 정확도와 하드웨어 효율성을 동시에 개선하는 데 기여할 것으로 기대하고 있습니다.



### Optimal Actuator Attacks on Autonomous Vehicles Using Reinforcement Learning (https://arxiv.org/abs/2502.07839)
Comments:
          Accepted in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) Workshop

- **What's New**: 이 논문에서는 자율주행 차량(AV)의 구동기(Actuator)에 대한 숨겨진 정합 공격을 설계하기 위한 강화학습(RL) 기반 접근법을 제안합니다. 특히, FDI 공격(거짓 데이터 주입 공격)에 중점을 두고 이러한 공격에 대한 기존 RL 기반 보안 제어기의 한계를 분석합니다. 자율주행 차량 시스템에서의 첨단 공격 탐지기를 고려하여 공격의 은밀성(stealthiness)에 대한 중요성을 강조하고 있습니다.

- **Technical Details**: 제안된 방법론에서는 자율주행 차량의 상태 및 측정 모델을 기반으로 하여, 공격자가 구동기에 주입하는 거짓 데이터를 정의합니다. 공격자의 주요 목표는 차량이 참조 경로(reference trajectory)에서 이탈하도록 하는 것이며, 이를 위해 다양한 비용 함수(cost functions)를 설정합니다. 은밀성을 유지하는 것 외에도 최소 에너지를 소비하며 경로 이탈 지점을 극대화하는 전략이 필요합니다.

- **Performance Highlights**: 광범위한 시뮬레이션 실험을 통해 제안된 방법의 효율성과 효과성을 입증하였습니다. 기존의 RL 기반 보안 제어기와 비교했을 때, 상대적으로 우수한 성능을 보이며 자율주행 차량의 보안성을 강화하는 데 기여할 것입니다. 이러한 결과는 자율주행 차량의 안전한 운행에 중요한 통찰력을 제공합니다.



### RoboBERT: An End-to-end Multimodal Robotic Manipulation Mod (https://arxiv.org/abs/2502.07837)
- **What's New**: 로보BERT(RoboBERT)는 입체 지능(Embodied Intelligence)을 구현하기 위한 새로운 로봇 조작 모델로, 다양한 형태의 관찰을 이해하고 행동을 생성하는 혁신적인 훈련 전략을 통합했습니다. 기존 모델들이 추가 데이터셋이나 대규모 선행 학습에 의존하는 반면, 로보BERT는 최소한의 데이터로도 뛰어난 성능을 달성하며, 경량화된 멀티모달 퓨전 네트워크를 사용합니다.

- **Technical Details**: 로보BERT 모델은 CNN 기반의 확산 정책(diffusion policy)과 함께 다양한 양식의 입력을 자동으로 처리하는 엔드 투 엔드의 언어 조건 로봇 운영 에이전트를 구현하고 있습니다. 모델은 자연어 입력과 원시 관찰 시퀀스를 통합하며, 효과적인 데이터 증강(data augmentation) 기법을 통해 성능을 극대화합니다. 또한 기존 방법보다 훨씬 적은 리소스로 훈련되는 특징이 있습니다.

- **Performance Highlights**: 로보BERT는 CALVIN 벤치마크에서 4.52의 평균 성과를 기록하며 새로운 최첨단 기록(SOTA)을 달성했습니다. 실제 로봇 테스트에서도 다른 방법들보다 더 높은 성공률을 보였으며, 다양한 작업을 성공적으로 수행했습니다. 이 모델은 단순한 기능 외에도 뛰어난 변별력과 호환성을 보여주어, 경량 멀티모달 로봇 모델의 개발에 크게 기여할 것으로 기대됩니다.



### Advancing Precision Oncology Through Modeling of Longitudinal and Multimodal Data (https://arxiv.org/abs/2502.07836)
Comments:
          This work has been submitted to the IEEE RBME for potential publication

- **What's New**: 이번 리뷰 논문에서는 암의 다이나믹한 특성을 반영하기 위해 긴 시계열(longitudinal) 데이터와 다중 모달(multimodal) 데이터를 통합하는 방법을 제안합니다. 기존의 암 연구는 단일 모달(single modality) 자료를 활용한 단면 분석에 주로 초점을 맞추었으나, 이는 질병의 심각한 이질성을 완전히 파악하는 데 한계가 있었습니다.

- **Technical Details**: 저자들은 longitudinal 및 multimodal 모델링 기술에 대한 다양한 방법론을 조사하며, 이들이 개인 맞춤형 치료(personalized care)를 위한 중요한 통찰을 제공하는 데 어떻게 기여하는지를 설명합니다. 특히, 시간이 지남에 따른 질병의 진행과 치료 반응 패턴을 드러내는 longitudinal 데이터의 중요성을 강조하고 있습니다.

- **Performance Highlights**: 이러한 기법들은 암의 조기 이상 감지 및 효율적인 치료 변화(dynamic treatment adaptation)를 가능하게 하며, 다양한 출처로부터의 보완적인 정보(complementary information)를 통해 보다 정밀한 위험 평가와 타겟(targeting) 치료를 지원합니다. 논문에서는 현재의 한계점과 향후 방향성에 대해서도 논의하며, precision oncology의 발전을 위한 기초 자료를 제공합니다.



### MEMHD: Memory-Efficient Multi-Centroid Hyperdimensional Computing for Fully-Utilized In-Memory Computing Architectures (https://arxiv.org/abs/2502.07834)
Comments:
          Accepted to appear at DATE 2025

- **What's New**: MEMHD는 메모리 효율성이 높은 다중 중심 HDC 프레임워크를 소개하여 In-Memory Computing(IMC) 아키텍처의 도전 과제를 해결한다. 이 프레임워크는 클러스터링 기반 초기화 방법과 양자화 인지 반복 학습 기법을 결합하여 분류 정확도를 유지하거나 개선하면서 메모리 요구 사항을 대폭 줄인다. 또한, MEMHD는 IMC 배열의 완전한 활용을 가능하게 하고 원샷(또는 몇 샷) 연관 검색을 지원한다.

- **Technical Details**: MEMHD는 인코딩 및 연관 메모리 모듈을 포함하여 HDC의 표준 구조를 넘어 다중 중심 모델을 개발한다. 이 모델은 초기 중심 배치를 위한 클러스터링 기반 초기화 및 양자화 효과를 반영한 반복 학습을 통해 훈련된다. 결과적으로, MEMHD는 기존의 HDC 접근 방식에서 사용하는 10,000차원 대신 1,000차원에 가까운 크기로 메모리 요구 사항을 감소시킨다.

- **Performance Highlights**: 실험 결과, MEMHD는 기존의 이진 HDC 모델에 비해 최대 13.69% 더 높은 정확도를 달성하며, 같은 메모리 사용량에서 더 나은 성능을 발휘한다. 또한, 계산 사이클을 최대 80배 줄이고, IMC 매핑 방식에 비해 배열 사용량을 71배 줄이는 성과를 보였다. 이러한 효율성 개선은 에너지와 계산 사이클 효율성 향상에도 기여한다.



### Captured by Captions: On Memorization and its Mitigation in CLIP Models (https://arxiv.org/abs/2502.07830)
Comments:
          Accepted at ICLR 2025

- **What's New**: 이 논문에서는 CLIP 모델의 메모리화(memorization) 과정을 정량화하기 위해 CLIPMem이라는 새로운 개념을 제안합니다. 기존의 메모리화 정의가 CLIP의 다중 모달 특성을 반영하지 못하는 한계를 해결하고자 메모리화를 설명하기 위해 새로운 기준을 설정합니다. '잘못 캡션된(mis-captioned)' 데이터가 가장 높은 메모리화 수준을 보인다는 중요한 결과도 발견했습니다.

- **Technical Details**: CLIP은 이미지와 텍스트의 쌍을 공유된 잠재 공간에 매핑하여 작업을 수행하는 다중 모달 인코더 구조를 기반으로 합니다. 이 모델은 대조 손실 함수(contrastive loss)를 통해 정답 쌍 사이의 유사성을 극대화하고 잘못된 쌍의 유사성을 최소화합니다. CLIPMem 메트릭은 주어진 이미지-텍스트 쌍의 정렬을 비교하여 메모리화를 측정하는 새로운 방법입니다.

- **Performance Highlights**: 연구를 통해 텍스트 인코더가 이미지 인코더보다 메모리화에 더 많이 기여하며, 이는 모델 성능에 중요한 영향을 미칩니다. 리서치는 메모리화 감소 전략이 CLIP의 일반화 능력을 향상시킬 수 있음을 보여줍니다. 이러한 결과는 전통적인 학습 패러다임에서는 메모리화 감소가 일반적으로 성능 저하로 이어지던 것과는 대조적입니다.



### Preference Alignment on Diffusion Model: A Comprehensive Survey for Image Generation and Editing (https://arxiv.org/abs/2502.07829)
- **What's New**: 이 논문은 이미지 생성 및 편집 분야에서 확산 모델(Diffusion Models, DMs)과 선호 정렬(preference alignment)의 통합을 체계적으로 조사한 첫 번째 연구입니다. 연구의 주요 목적은 이러한 통합이 초보자에게 제시하는 도전 과제를 조명하고, DMs와 RL(강화 학습)이 이미지 생성과 편집에 어떻게 상호작용하는지에 대한 명확한 개요를 제공하는 것입니다. 기존의 많은 연구에서 DMs의 다양한 활용은 다뤘지만, 선호 정렬과의 구체적인 통합에 대해 종합적으로 검토한 자료는 부족했습니다.

- **Technical Details**: 본 연구에서는 DDPMs(분산 모델)와 DDIMs(Deterministic Denoising Diffusion Implicit Models)의 원리에 대해 설명하며, 특히 노이즈 추가 과정과 그 역과정을 강조합니다. 또한, PPO(Proximal Policy Optimization) 알고리즘을 활용하여 선호 정렬을 위한 강화 학습 과정에 대해 깊이 있게 기술합니다. 이러한 알고리즘은 훈련에서 안정성과 강인성을 제공하며, 이를 통해 DMs의 성능을 효과적으로 향상시킬 수 있는 방법을 제시합니다.

- **Performance Highlights**: 이 논문은 DMs와 RL 기반 최적화 기술(예: RLHF, DPO)이 이미지 생성 및 편집에서 어떻게 활용되는지를 중점적으로 살펴봅니다. 자율 주행, 의료 영상, 로보틱스 등 다양한 분야에서 DMs의 응용 가능성을 탐구하며, 운영에서 겪는 도전 과제들과 이들의 해결 방안도 논의합니다. 이를 통해 향후 기술 발전을 위한 방향성을 제시하고 DMs와 선호 정렬을 연계한 혁신적인 연구의 가능성을 강조합니다.



### Runtime Tunable Tsetlin Machines for Edge Inference on eFPGAs (https://arxiv.org/abs/2502.07823)
Comments:
          Accepted as a full paper by the 2025 EDGE AI FOUNDATION Austin

- **What's New**: 이 논문은 새로운 eFPGA (embedded Field-Programmable Gate Arrays) 가속기를 제안하여 기존 FPGA보다 더 적은 전력으로 엣지 머신러닝 (ML) 응용프로그램을 지원할 수 있도록 합니다. 주요 특징은 자원 사용을 최소화하고 모델 크기와 아키텍처를 런타임에 재조정할 수 있는 유연성을 제공하는 점입니다. 이는 Tsetlin Machine (TM) 알고리즘의 비트 압축 추론 아키텍처를 통해 가능해집니다.

- **Technical Details**: 제안된 eFPGA 가속기는 비트 연산(AND, OR, NOT)만을 사용하여 모델을 구현하므로 곱셈 연산이 필요하지 않습니다. TM 모델의 압축 덕분에 전체 모델이 eFPGA의 온칩 블록 RAM에 적재될 수 있으며, 이론적으로 2.5배 적은 Look-up-Tables (LUTs)와 3.38배 적은 레지스터를 사용합니다. 또한, 기존 저전력 마이크로컨트롤러와 비교 시 에너지 요구사항을 129배 줄일 수 있습니다.

- **Performance Highlights**: 이 연구는 런타임 모델 조정을 위한 전략을 제안하며, 기존의 자원 절약형 설계와 비교해 성능을 벗어나지 않으면서도 유연성을 갖춘 결과를 보여줍니다. TM을 기반으로 한 구조적 단순성과 희소성 덕분에 작은 엣지 장치에서 효과적인 ML 모델을 구현할 수 있습니다. 이는 적은 리소스를 사용하면서도 높은 응용성을 지원하는 새로운 가능성을 제공합니다.



### CP-Guard+: A New Paradigm for Malicious Agent Detection and Defense in Collaborative Perception (https://arxiv.org/abs/2502.07807)
- **What's New**: 이번 연구에서는 collaborative perception (CP) 시스템의 취약점을 해결하기 위한 새로운 접근법을 제안합니다. 특히 malicious agent를 feature level에서 직접 탐지하는 방법을 도입하여, 기존의 여러 차례의 검증 과정을 줄이고 계산 오버헤드를 크게 감소시킵니다. 또한, CP-GuardBench라는 새로운 데이터셋을 생성하여 악의적인 에이전트 탐지를 위한 기준을 제공합니다.

- **Technical Details**: 이 논문에서는 두 가지 주요 구성 요소를 도입합니다. 첫째, feature encoder, aggregator 및 decoder를 이용한 CP 시스템의 파이프라인을 설명합니다. 둘째, CP-Guard+라는 새로운 방법론을 통해 benign 및 악의적 feature 간의 마진을 증가시키고, Dual-Centered Contrastive Loss (DCCLoss)를 통해 인식 불가능성을 해결합니다.

- **Performance Highlights**: 연구 결과, CP-Guard+는 CP-GuardBench 및 V2X-Sim에서 수행된 실험에서 우수한 성능을 보였습니다. 이 시스템은 feature level에서 malicious agent를 효과적으로 탐지하며, 최종 인식 결과를 검증하는 과정을 생략하여 계산 효율성을 높입니다. 이러한 점에서 CP 시스템의 보안을 크게 향상시킬 수 있는 가능성을 보여줍니다.



### Quantum Powered Credit Risk Assessment: A Novel Approach using hybrid Quantum-Classical Deep Neural Network for Row-Type Dependent Predictive Analysis (https://arxiv.org/abs/2502.07806)
- **What's New**: 이 논문은 Quantum Deep Learning (QDL) 기법을 금융 리스크 분석에 통합하여 혁신적인 접근 방식을 제시합니다. 은행 부문에서의 신용 리스크 평가를 위해 Row-Type Dependent Predictive Analysis (RTDPA)를 활용하여 대출 카테고리에 맞춘 예측 모델을 만드는 프레임워크를 소개합니다. 이러한 접근법은 전통적인 방법과 양자 컴퓨팅의 장점을 결합하여 신용 리스크 평가의 정확성과 효율성을 높이는 것을 목표로 합니다.

- **Technical Details**: 이 연구는 양자 컴퓨팅과 고전적 깊은 신경망을 통합하여 신용 리스크 평가를 보다 유연하고 적응 가능한 방법으로 발전시키고자 합니다. RTDPA는 서로 다른 대출 유형의 특성을 인식하여 개별적으로 분석하여 예측 모델을 조정하는 방법을 제안합니다. 이 연구는 또한 기반 최적화 기술인 Beetle Antennae Search (BAS)와 같은 대안 최적화 기법에 대해서도 논의하며, 향후 연구에서 이 기법의 잠재력을 강조합니다.

- **Performance Highlights**: 연구 결과는 양자 기법이 전통적인 금융 분석을 보완하는 방법에 대한 통찰력을 제공합니다. QDL은 신용 리스크 평가에서 더 나은 예측력을 달성할 수 있는 가능성을 제시하며, 다양한 대출 유형에 맞춘 더 정밀한 리스크 평가를 가능하게 합니다. 이 프레임워크는 점차 역동적인 금융 환경에서 신용 리스크 분석을 위한 보다 효과적인 접근 방식을 제공하는 것을 목표로 합니다.



### Reasoning-as-Logic-Units: Scaling Test-Time Reasoning in Large Language Models Through Logic Unit Alignmen (https://arxiv.org/abs/2502.07803)
- **What's New**: 이 논문에서는 Chain-of-Thought (CoT)와 Program-of-Thought (PoT)의 한계점을 극복하기 위해 Reasoning-as-Logic-Units (RaLU)라는 새로운 접근 방식을 제안합니다. RaLU는 자연어(NL) 설명과 프로그래밍 간의 정합성을 높여 더 신뢰할 수 있는 추론 경로를 구축합니다. 이 방법은 오류를 발견하고 자가 수정을 수행하여 프로그래밍과 추론의 성능을 향상시키는 것을 목표로 합니다.

- **Technical Details**: RaLU 프레임워크는 네 가지 핵심 작업, 즉 self-reason, self-judge, self-explain, self-correct를 포함하며 세 가지 주요 단계로 구성됩니다: Logic Unit Extraction, Logic Unit Alignment, Solution Synthesis. Static analysis를 활용하여 프로그램의 Control Flow Graph (CFG)를 생성하고, 이를 기반으로 논리 단위를 추출한 후 LLM과의 대화를 통해 각 단위를 평가, 수정, 설명하게 됩니다.

- **Performance Highlights**: 실험 결과 RaLU는 수학적 추론(GSM8K, MATH) 및 알고리즘적 추론(HumanEval+, MBPP+)에서 기존의 최선형 기준 모델들보다 1.22%, 2.07%, 6.60%, 2.17%의 향상을 보였습니다. 특히 RaLU는 HumanEval+와 MBPP+에서 최고의 성능을 발휘하는 모델군을 초과하는 결과를 나타내어 LLM 추론과 프로그래밍의 정확성과 해석 가능성을 크게 향상시킬 잠재력을 입증했습니다.



### Movie Weaver: Tuning-Free Multi-Concept Video Personalization with Anchored Prompts (https://arxiv.org/abs/2502.07802)
Comments:
          Project page: this https URL

- **What's New**: 이번 논문에서는 개인화 비디오 생성의 새로운 접근 방식인 Movie Weaver가 소개됩니다. 이 모델은 다양한 이미지 조합과 텍스트 프롬프트를 엮어서 개인화된 다중 개념 비디오를 생성할 수 있도록 설계되었습니다. 이를 통해 얼굴, 신체, 동물 이미지 등 여러 개념을 유연하게 조합할 수 있어 다양한 실세계 시나리오에 적용할 수 있는 가능성을 제시합니다.

- **Technical Details**: Movie Weaver는 앵커 프롬프트(anchored prompts)를 도입하여 각 개념 설명을 특정 참조 이미지에 링크합니다. 또한, 개념 임베딩(concept embeddings)을 사용하여 참조 이미지의 순서를 인코딩합니다. 이 두 가지 기술을 통해, Movie Weaver는 기존 방법들에 비해 신원 보존(identity preservation)과 전반적인 품질에서 우수한 성능을 보입니다.

- **Performance Highlights**: Movie Weaver의 성능은 특히 다중 개념 비디오 개인화에서 더욱 두드러집니다. 기존의 Vidu 1.5 및 다른 기본 방법들과 비교했을 때, Movie Weaver는 더 나은 신원 보존 및 시각적 품질을 성취하였습니다. 이 연구는 자동 데이터 큐레이션(pipeline) 방식으로 다양한 참조 이미지 조합을 활용하여 고품질 비디오 생성이 가능함을 보여줍니다.



### neuro2voc: Decoding Vocalizations from Neural Activity (https://arxiv.org/abs/2502.07800)
Comments:
          Master Thesis

- **What's New**: 이 논문에서는 뉴로픽셀(Neuropixels)에서 얻은 침습적 신경 기록을 통해 제브라 핀치(zebra finch)의 운동 출력을 디코딩하기 위한 실험적 방법을 조사합니다. 기존의 연구와는 달리, 신경 스파이크 사건의 희소성과 길이를 극복하기 위한 새로운 접근법을 사용합니다. 논문의 주요 성과는 신경 데이터를 처리하는 여러 신선한 방법론을 제시했다는 점입니다.

- **Technical Details**: 주요 기술적 성과로는 (1) XGBoost와 SHAP 분석을 통한 스파이크 비율(spike rates) 기반의 신경 상호작용 패턴 발견이 있습니다. (2) GPT2를 사용한 신경 데이터의 토큰화(tokenizing neural data)와 Mamba2 아키텍처가 스파이크를 통해 음절을 디코딩할 수 있는 가능성을 보여줍니다. (3) 대조 학습-변분 오토인코더(contrastive learning-VAE) 프레임워크를 통해 빈화된(binned) 신경 데이터로부터 스펙트로그램(spectrograms)을 생성하는 데 성공했습니다.

- **Performance Highlights**: 이 연구는 복잡한 운동 출력을 신경 디코딩(neural decoding)하는 데 있어 유망한 기초를 마련하며, 희소한 신경 데이터를 처리하기 위한 새로운 방법론을 제공합니다. 특히, 음절 분류에 있어 중요한 신경 상호작용 패턴들을 밝혀냈고, 생성된 스펙트로그램은 향후 연구에 유용한 자료로 작용할 것입니다.



### Predictive Coresets (https://arxiv.org/abs/2502.05725)
- **What's New**: 본 논문은 대규모 데이터셋에 대한 Bayesian 후행 추론을 기반으로 한 새로운 coreset 설계를 제안합니다. 이 방법은 model-agnostic(모델 불변적)으로 다양한 추론 모델에 적합하며, 전통적인 방법들이 효율성의 한계를 가질 때 유용성을 발휘합니다. 특히 유전자학 및 천문학 같이 데이터가 폭발적으로 증가하는 분야에서 광범위한 응용이 가능합니다.

- **Technical Details**: 저자들은 랜덤화된 posterior(후행 분포)를 활용하는 변환 방법을 채택하여, 특정 목표에 구애받지 않고 안정적으로 coreset을 구성합니다. 이 과정에서 likelihoods가 아닌 posterior predictive distributions(후행 예측 분포)에 중점을 둡니다. 또한, 데이터 포인트 간의 거리를 사용하여, 비유클리드(Non-Euclidean) 데이터와 같은 비표준 응용에도 적용할 수 있도록 합니다.

- **Performance Highlights**: 제안된 coreset 구조의 성능은 다양한 문제들, 특히 파라메트릭 로지스틱 회귀, 임의 파티션 및 밀도 추정 작업을 통해 검증되었습니다. 이 방법은 기존의 Bayesian 추론과 관련하여 이론적 보장을 제공합니다. 또한, 고차원 데이터 환경에서도 효율성과 정확성을 유지할 수 있는 가능성을 보여줍니다.



### Machine Learning and Quantum Intelligence for Health Data Scenarios (https://arxiv.org/abs/2410.21339)
Comments:
          Presented at Machine Learning and Machine Intelligence (MLMI) Conference, Osaka, Japan 2024

- **What's New**: 이 논문은 양자 컴퓨팅(quantum computing)이 데이터 과학(data science)에서 복잡하고 데이터 집약적인 문제를 해결할 수 있는 새로운 가능성을 열어준다는 점에서 주목할 만하다. 특히, 양자 머신러닝(Quantum Machine Learning, QML)이 전통적인 머신러닝 알고리즘이 어려움을 겪는 고차원(high-dimensional) 및 데이터 품질이 제한적인 데이터셋에서의 해결책을 제시하고 있다.

- **Technical Details**: 양자 머신러닝은 초위치(superposition)와 얽힘(entanglement)과 같은 양자 속성을 활용하여 패턴 인식(pattern recognition) 및 분류(classification)에서의 성능을 향상시킨다. 이 논문은 심장 질환 예측 및 COVID-19 탐지를 위한 양자 커널 방법(quantum kernel methods)과 하이브리드 양자-고전적 네트워크(hybrid quantum-classical networks)의 적용을 심도 있게 다룬다.

- **Performance Highlights**: 연구에서는 양자 커널 방법과 하이브리드 모델의 실용성을 평가하고 이들이 기존의 접근 방법들을 능가할 가능성을 탐색하였다. 특히 헬스케어 분야에서의 적용 가능성은 기존의 머신러닝 기법보다 더 나은 성능을 보일 것으로 기대된다.



### AdjointDEIS: Efficient Gradients for Diffusion Models (https://arxiv.org/abs/2405.15020)
Comments:
          NeurIPS 2024 conference paper

- **What's New**: 이 논문은 새로운 종류의 ODE 솔버인 AdjointDEIS를 제안합니다. 이 솔버는 확률적 미분 방정식(SDE)에 기반한 확산 모델의 연속 비선형 적분 방정식을 효율적으로 해결하며, 이러한 방정식은 단순한 ODE로 간소화됩니다. 이를 통해 파라미터와 잠재 변수의 효율적인 최적화를 가능하게 합니다.

- **Technical Details**: AdjointDEIS는 확산 SDE의 연속 비선형 적분 방정식을 해결하기 위해 독창적인 구조를 활용합니다. 논문에서는 Exponential Integrators를 사용하여 이러한 방정식의 구성을 간소화하고, 새로운 솔버에 대한 수렴 속도 보장을 제공합니다. 이 접근법은 네트워크의 메모리를 절약하면서도 중간 상태를 저장하지 않고도 역전파(Backpropagation)가 가능하게 합니다.

- **Performance Highlights**: AdjointDEIS는 적대적 공격 형태의 얼굴 변형 문제에서 유도된 생성을 위해 효과적으로 평가되었습니다. 실험을 통해 제안된 방법이 기존의 접근법보다 뛰어난 성능을 보여, 고급 생성 모델의 훈련 없이 새로운 개념과 작업을 학습할 수 있는 가능성을 제시합니다.



### On the Sample Complexity of Quantum Boltzmann Machine Learning (https://arxiv.org/abs/2306.14969)
Comments:
          Main text: 11 pages, 3 figures. Supplementary information: 16 pages, 2 figures. We correct a mistake that affected both lemmas 6 and 7. We thank Dhrumil Patel and Mark M. Wilde for identifying this mistake

- **What's New**: 이번 연구에서는 quantum Boltzmann machine (QBM)이라는 새로운 기계 학습 모델의 운영적 정의를 제시하고, 데이터 집합의 다항 크기를 고려한 QBM 학습의 성능 평가 방법을 설명합니다. 이전의 연구와 달리, 이 모델은 barren plateaus 문제를 피할 수 있으며, 다항적으로 많은 Gibbs 상태로 해결할 수 있다는 점이 강조됩니다. 또한 QBM 파라미터의 일부에 대한 사전 학습이 샘플 복잡성을 낮출 수 있다는 것을 증명합니다.

- **Technical Details**: 이 논문에서는 QBM 학습 문제를 형식적으로 설정하며, 목표와 모델의 정의를 제공합니다. QBM의 목표는 특정한 Hamiltonian ansatz에 기반하여 대량의 데이터로부터 정보를 추출하는 것입니다. 연구자는 QBM이 기대값의 차이를 통해 성능을 평가할 수 있도록 하고, stochastic gradient descent와 shadow tomography를 사용하여 모델의 성능을 보장할 수 있도록 합니다.

- **Performance Highlights**: 연구 결과는 QBM이 이론적으로 및 수치적으로 유망한 기계 학습 모델임을 입증합니다. 구체적으로, mean-field 및 Gaussian Fermionic QBM의 사전 학습 전략이 제시되며, 이 모델들이 실제 문제에 효과적으로 적용될 수 있는 가능성 또한 확인되었습니다. 이 논문은 QBM의 샘플 복잡성이 효과적인 학습을 가능하게 한다고 강조하면서, QBM과 관련된 여러 응용 분야에 대해 논의합니다.




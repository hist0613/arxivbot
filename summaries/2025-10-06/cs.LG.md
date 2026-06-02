New uploads on arXiv(cs.CL)

### Reward Models are Metrics in a Trench Coa (https://arxiv.org/abs/2510.03231)
- **What's New**: 본 논문은 대형 언어 모델의 후속 교육에서 강화 학습(reinforcement learning, RL)의 부상과 이에 따른 보상 모델(reward models)의 중요성을 조명합니다. 연구 분야 간의 단절로 인해 유사한 용어와 문제들이 반복되고 있는데, 이를 통해 보상 모델과 평가 메트릭(evaluation metrics)이 서로 보완할 수 있는 기회를 제시합니다. 보상 모델과 평가 메트릭 간의 협력이 필요하다는 주장을 하며, 이를 통해 보상 해킹(reward hacking)과 같은 문제를 극복할 수 있는 방안을 제안합니다.

- **Technical Details**: 강화 학습(REINFORCE 및 minimum risk training 등)은 생성 모델의 훈련 및 평가에 있어 중요한 역할을 하며, 기존 메트릭을 통해 복잡한 상태 공간을 다루는 방식을 설명합니다. RL은 학습 과정에서 발생하는 노출 편향(exposure bias)을 극복하기 위한 자연스러운 방법으로, 다양한 사례에서 활용되고 있습니다. 다양한 소스에서 생성된 언어의 품질을 평가하는 메트릭들은 과거에 비해 의미적 유사성을 측정하게 되어 보상 해킹을 줄이는 데 기여하고 있습니다.

- **Performance Highlights**: 연구에서는 보상 모델이 특정 작업에서 메트릭보다 부족한 성과를 나타낸다는 실험 결과를 제시합니다. 이러한 발견들을 기반으로, 주관적 판단과 머신 학습 기반 판단의 상관관계를 개선할 수 있는 기회를 포착할 필요성이 강조됩니다. 또한, 보상 모델과 평가 메트릭이 함께 발전할 수 있는 다양한 연구 주제와 협력의 중요성을 명확히 합니다.



### Self-Anchor: Large Language Model Reasoning via Step-by-step Attention Alignmen (https://arxiv.org/abs/2510.03223)
- **What's New**: 이 논문은 Self-Anchor라는 혁신적인 파이프라인을 제안하여 LLM(대형 언어 모델)의 주의를 유도한다. Self-Anchor는 복잡한 추론 문제를 구조화된 계획으로 분해하고, 모델의 주의를 가장 관련성 높은 추론 단계에 자동으로 정렬하여 생성 과정에서 초점을 유지할 수 있게 한다. 이 방법은 기존의 SOTA(최첨단) 프롬프트 방법보다 우수한 성능을 보이며, 비전략 모델과 전문화된 추론 모델 간의 성능 격차를 크게 줄일 수 있는 가능성을 지닌다.

- **Technical Details**: Self-Anchor는 복잡한 문제를 계획 단계로 나눈 후, 각 계획 단계가 해당하는 추론과 연관되어 주의 정렬을 유도할 수 있도록 설계되었다. 이 과정에서 SPA(Selective Prompt Anchoring)라는 낮은 비용의 주의 유도 메커니즘을 적용하여 주의 토큰 선택을 자동화한다. 또한, 주의 정렬 강도는 모델의 신뢰도에 따라 동적으로 조정되어 생성 과정에서 민첩하게 변화하는 토큰에 적절하게 주의를 기울이도록 한다.

- **Performance Highlights**: Self-Anchor는 GSM8K, AQuA, MATH 등 6개의 벤치마크에서 SOTA 프롬프트 방법들과 비교하여 모든 설정에서 평균적으로 5.44% 이상의 정확도 향상을 보였다. 또한, Self-Anchor는 5개의 전문 추론 모델과 동등한 성능을 보이면서도 훨씬 낮은 비용과 복잡성으로 실행 가능하다는 점에서 LLM의 추론 능력을 향상시키기 위한 실용적인 대안으로 제시된다.



### Cache-to-Cache: Direct Semantic Communication Between Large Language Models (https://arxiv.org/abs/2510.03215)
- **What's New**: 이번 연구는 다수의 Large Language Models (LLMs) 의 상호작용에서 텍스트를 넘어선 소통을 가능케 하는 Cache-to-Cache (C2C) 패러다임을 제안합니다. 기존의 다중 언어 모델 시스템에서 LLM들은 텍스트를 통해 소통하며, 이 과정에서 중요한 의미 정보가 손실되고 생성 지연이 발생하는 문제를 해결하고자 하였습니다. Oracle 실험 결과 KV-Cache를 활용하면 응답 품질을 개선하고 속도도 향상할 수 있음을 보여주었습니다.

- **Technical Details**: C2C는 소스 모델의 KV-Cache를 타겟 모델의 공간으로 투영하고 이를 통합하는 신경망을 통해 직접적인 의미 전달을 가능하게 합니다. 학습 가능한 게이팅 메커니즘이 캐시 소통의 이익을 얻는 타겟 레이어를 선택하여 보다 정교한 소통을 구현합니다. 이는 텍스트 커뮤니케이션에서 발생하는 여러 제약(예: 정보 병목, 애매모호성 및 지연)을 극복할 수 있는 새로운 방법론으로 자리매김합니다.

- **Performance Highlights**: C2C 방식은 개별 모델 대비 평균 8.5-10.5% 높은 정확도를 달성하였으며, 텍스트 커뮤니케이션 패러다임보다 3.0-5.0% 높은 성과를 보이며, 평균 2.0배의 속도 향상을 제공합니다. 이 결과들은 LLM 간의 원활한 의미 기반 소통을 통해 성능을 극대화할 수 있음을 입증합니다.



### FocusAgent: Simple Yet Effective Ways of Trimming the Large Context of Web Agents (https://arxiv.org/abs/2510.03204)
- **What's New**: FocusAgent는 웹 에이전트가 긴 웹 페이지 관찰을 효율적으로 처리하고, 사용자의 목표를 달성할 수 있도록 돕는 새로운 접근 방식을 제안합니다. 이 방법은 쉽게 구현 가능한 경량의 LLM 리트리버를 사용하여, 접근성 트리(AxTree)에서 가장 관련성 높은 정보를 추출합니다. 특히, FocusAgent는 불필요한 정보를 제거하여 효율성을 높이고, 프롬프트 인젝션 공격에 대한 취약성을 줄이는 데 기여합니다.

- **Technical Details**: FocusAgent는 각 단계에서 웹 에이전트의 탐색 결정을 돕기 위해 AxTree에서 관련 정보를 선택적으로 추출합니다. 이 접근 방식은 전통적인 단순한 의미적 일치를 넘어서 작업 목표와 이전 행동의 맥락을 고려하여 정보를 보존할지를 결정합니다. 실험 결과, FocusAgent는 관찰 크기를 평균 50% 이상 줄이면서도 동등한 성능을 유지할 수 있습니다.

- **Performance Highlights**: FocusAgent는 WorkArena와 WebArena 벤치마크에서 강력한 기준선과 동등한 성능을 보여주면서 공격으로부터의 성공률을 현저히 감소시킵니다. 또한, FocusAgent는 공격 없는 환경에서도 안정적인 성능을 유지하며, 사용자 데이터를 보호하기 위한 보안 기능으로서의 역할을 합니다. 우리 연구 결과는 LLM 기반의 목표 지향적 정보 추출이 웹 에이전트를 구축하는 실용적이고 강력한 전략임을 강조합니다.



### Model-Based Ranking of Source Languages for Zero-Shot Cross-Lingual Transfer (https://arxiv.org/abs/2510.03202)
Comments:
          Accepted to EMNLP 2025 (Main)

- **What's New**: 이 논문에서는 NN-Rank라는 새로운 알고리즘을 소개합니다. NN-Rank는 다국어 모델에서 숨겨진 표현을 활용하여 소스 언어의 순위를 매기는 방법을 제안합니다. 이를 통해 이전의 상태 기반 접근법에 비해 더 나은 성능을 발휘하며, 특히 레이블이 없는 데이터에서 효과적입니다.

- **Technical Details**: NN-Rank는 두 개의 프리트레인한 다국어 모델을 사용하여, part-of-speech tagging (POS)와 named entity recognition (NER)과 같은 두 가지 작업을 수행합니다. 소스 데이터셋에 데이터 가용성을 기준으로 한 순위를 매기기 위해, 모델의 중간 계층에서 추출한 숨겨진 표현을 사용합니다. 이 접근법은 레이블이 없는 데이터만으로도 우수한 성과를 보여줍니다.

- **Performance Highlights**: NN-Rank는 POS의 경우 최대 35.56 NDCG, NER의 경우 18.14 NDCG의 평균 향상을 보여주며, 기존 최고 성능을 기록한 방법들보다 우수합니다. 더불어, 25개의 예제를 사용하는 경우에도 전체 데이터에서 얻은 NDCG의 92.8%를 달성할 수 있어, 적은 양의 데이터로도 효과적인 성과를 보장합니다.



### Topic Modeling as Long-Form Generation: Can Long-Context LLMs revolutionize NTM via Zero-Shot Prompting? (https://arxiv.org/abs/2510.03174)
- **What's New**: 이번 연구는 기존의 전통적인 주제 모델링(Topic Modeling, TM) 접근 방식을 대체하여, 대규모 언어 모델(LLM)을 활용한 새로운 패러다임을 제시합니다. 저자들은 LLM 기반의 주제 모델링을 긴 텍스트 생성 작업으로 재정의하고, 이를 통해 효과적으로 주제를 수집하고 관련 텍스트를 생성하는 방법을 제안했습니다. 이를 통해 NTMs(Neural Topic Models)와 LLM의 성능 차이를 분석하고, LLM이 NTMs에 비해 우수할 수 있음을 조명합니다.

- **Technical Details**: 연구에서는 전통적인 TM 모델, 특히 LDA(잠재 디리클레 할당)와 NTMs의 한계를 지적하며, 이를 극복하기 위한 새로운 접근 방식을 설명합니다. 잠재적 주제 분포를 학습하기 위한 세 가지 주요 단계를 포함하는 방법론을 제시하며, 데이터셋 전처리, 주제 생성, 텍스트 할당의 과정을 통합합니다. 특히, LLM의 긴 컨텍스트 윈도우를 활용하여 보다 효과적으로 주제를 추출하고 텍스트를 할당하는 절차를 마련했습니다.

- **Performance Highlights**: 실험 결과, LLM 기반의 모델이 NTMs에 비해 주제 품질이 개선되는 경향을 보였습니다. 특히, 품질 평가에서 LLM이 더 나은 성능을 제공하는 것으로 드러났으며, 기존 NTMs의 다수는 시대에 뒤떨어져 있다는 주장에 대한 실증적 근거를 제시했습니다. 연구자들은 이러한 발견이 LLM 기반의 접근 방식이 기존의 TM 방식에 비해 적절히 주제를 분석하고 생성할 수 있음을 보여준다고 결론짓습니다.



### Neural Correlates of Language Models Are Specific to Human Languag (https://arxiv.org/abs/2510.03156)
Comments:
          To be presented at NeurIPS 2025 Workshops

- **What's New**: 본 연구는 대형 언어 모델(large language models)과 fMRI 뇌 반응 간의 상관관계가 여러 가지 가능성 있는 우려 사항에 강력하다는 것을 입증하고자 하였다. 특히, 기존의 결과가 차원 축소 이후에도 유효하며, 인간 언어에 대해 훈련된 모델에만 특정한 상관관계가 존재함을 보여주었다. 또한, 모델의 위치 인코딩(positional encoding) 존재 여부가 결과에 영향을 미친다는 점도 확인하였다. 이 결과들은 최신 연구 결과를 강화하고 생물학적 가능성과 해석 가능성에 대한 논의에 기여한다.

- **Technical Details**: 연구에서는 19개의 텍스트 훈련된 트랜스포머(transformers)와 fMRI 데이터(6명의 성인 피험자)로 브레인 모델(brain model) 및 언어 모델 비교를 수행하였다. 실험에서는 서로 다른 구조의 두 가지 트랜스포머 모델(양방향 및 인과(causal) 모델)을 사용하여, BOLD 신호로부터 얻은 핵심 효과를 50개의 주성분(PCA)으로 축소하여 분석하였다. 또한, 중심화된 커널 정렬(centred-kernel alignment, CKA)과 그로모프-바서슈타인 거리(Gromov-Wasserstein distance)를 계산해 대표성 유사성(representational similarity)을 기하학적으로 분석하였다.

- **Performance Highlights**: 실험의 주요 결과로는 위치 인코딩 제거 시 브레인 점수가 최대 0.4가 하락하고, GW 거리가 증가하며, CKA 곡선이 평탄해지는 현상이 관찰되었다. 이는 브레인 모델이 자연어 특정(representationally specific)이며, 언어 입력이 없는 모델을 사용할 경우 상관관계가 사라진다는 것을 의미한다. 이러한 결과는 대형 언어 모델과 인간의 언어 처리 메커니즘 간의 연관성을 더욱 뚜렷하게 부각시키며, 기존 연구의 결과를 보완한다.



### EditLens: Quantifying the Extent of AI Editing in Tex (https://arxiv.org/abs/2510.03154)
- **What's New**: 이번 논문에서는 인공지능(AI)이 수정한 텍스트를 탐지하기 위한 EditLens 모델을 제안합니다. 이 모델은 기존의 이진(classification) 탐지기와 달리 AI 수정의 정도를 연속적인 점수로 추정할 수 있도록 설계되었습니다. 최근 연구에 따르면, 대다수의 사용자가 AI를 통해 텍스트를 수정 요청하고 있어, 이러한 AI 수정 텍스트를 탐지할 필요성이 커지고 있습니다. 특히 EditLens는 기존 AI 텍스트 탐지 기술의 한계를 극복하여 인간, AI, 혼합 텍스트를 효과적으로 구별합니다.

- **Technical Details**: EditLens 모델은 경량의 유사성 메트릭(similarity metrics)을 활용하여 AI의 수정 정도를 정량화합니다. 이 메트릭은 인간 심사자와의 검증을 통해 신뢰성을 확보하였으며, 다중 레벨 수정(mixed authorship)에 대한 정확한 판단을 위한 데이터를 생성합니다. 특히, 모델은 과거 연구에서 다룬 경계 탐지(boundary detection)나 문장 수준 분류(sentence-wise classification)와는 달리, 전체 텍스트에서 AI 편집의 정도를 직접적으로 회귀(regression)하여 추정합니다.

- **Performance Highlights**: EditLens는 이진 분야에서 94.7%의 F1 점수, 삼중(t3rnary) 분야에서 90.4%의 F1 점수를 기록하여 현재까지의 최고 성능을 자랑합니다. 이러한 결과는 EditLens가 AI가 인간의 글에 미친 영향을 정확하게 평가할 수 있음을 입증합니다. 또한, 이 모델은 더 낮은 오탐률(false positive rate)과 함께 AI 사용 정책의 유연성을 높이는 데 기여할 것입니다.



### Beyond the Final Layer: Intermediate Representations for Better Multilingual Calibration in Large Language Models (https://arxiv.org/abs/2510.03136)
- **What's New**: 이 논문은 다국어 상황에서의 신뢰도 보정(confidence calibration)에 대한 최초의 대규모 체계적 연구를 실시했습니다. 6개 모델 계열 및 100개 이상의 다국어를 포함해 비영어 언어의 보정이 시스템적으로 열악하다는 사실을 밝혔습니다. 특히 영어 중심의 훈련 집중으로 인해 다국어에 대한 보정 신호가 불완전하다는 점을 강조합니다.

- **Technical Details**: 무엇보다도 저자들은 다국어 모델에서 최종 층(final layer)에서만 신뢰도 점수를 추출하는 기존의 관행에 도전합니다. 대신 중간 층의 언어 중립적인 표현이 더 나은 보정 신호를 제공한다고 주장합니다. 이 연구는 세 가지 신뢰도 평가 전략을 소개하며, 그 중 LACE(언어 인식 신뢰도 앙상블) 메소드는 특정 언어에 최적화된 층을 선택하는 방법을 제안합니다.

- **Performance Highlights**: 결과적으로 제안된 방법은 모든 모델에서 다국어 보정을 지속적으로 개선하는 효과를 나타냈습니다. 특히, LACE 같은 방법은 기존의 온도 조정(Temperature Scaling) 기법과 결합했을 때 더욱 향상된 성능을 보입니다. 연구는 다국어 LLM의 보정에서의 구조적 격차를 메우기 위한 유효한 새로운 경로를 제시합니다.



### SurveyBench: How Well Can LLM(-Agents) Write Academic Surveys? (https://arxiv.org/abs/2510.03120)
- **What's New**: 이번 논문에서는 학술 설문 작성의 자동화를 위한 새로운 평가 프레임워크인 SurveyBench를 제안합니다. 이 프레임워크는 11,343개의 arXiv 논문과 4,947개의 고품질 설문을 기반으로 한 주제를 탐색하고, 서로 다른 메트릭 계층을 통해 내용의 품질을 평가합니다. 특히, 독자의 정보 요구에 명확히 부합하는 평가 프로토콜을 사용하여 기존 LLM4Survey 접근 방식을 효율적으로 도전합니다. 최근의 연구에 비해 LLM이 만든 설문이 인간 작문에 비해 평균 21% 낮은 평가 점수를 기록하는 것으로 나타났습니다.

- **Technical Details**: SurveyBench는 (1) 고품질의 인간 작성 설문을 포함한 주제 데이터셋, (2) 인간 참조 기반 및 비참조 기반 메트릭을 포함하는 이중 평가 프로토콜, (3) 다차원 평가 기준을 캡처하는 계층적 평가 구조로 구성됩니다. 이 연구는 또한 OpenAI-DeepResearch와 세 가지 설문 전용 방법의 성과를 비교하여 LLM이 생성한 설문이 표면적인 표현에서 유창하지만, 내용 깊이 및 퀴즈 기반 평가에서 여전히 부족하다는 것을 보여주었습니다. 이러한 세부 사항을 통해 학술 설문 작성에 대한 새로운 평가 기준을 마련할 수 있는 가능성을 열게 됩니다.

- **Performance Highlights**: SurveyBench의 효과를 검증하기 위해, OpenAI-DeepResearch와 세 가지 설문 전용 방법을 평가한 결과, LLM이 생성한 설문이 유창성과 구조면에서 우수함에도 불구하고, 내용의 풍부함 및 퀴즈 기반 평가에서 인간 전문가의 설문에 비해 크게 뒤처지는 것으로 나타났습니다. 이는 자동 설문 작성의 향후 발전을 위해 더욱 세분화된 최적화가 필요하다는 점을 강조합니다. 이번 연구는 기존의 평가 메트릭에 비해 더 풍부하고 의미 있는 제안을 통해 학술 연구의 질적 향상을 도모할 수 있는 방법을 보여주고 있습니다.



### Listening or Reading? Evaluating Speech Awareness in Chain-of-Thought Speech-to-Text Translation (https://arxiv.org/abs/2510.03115)
- **What's New**: 이번 논문은 Speech-to-Text Translation (S2TT) 시스템이 직면하는 주요 한계인 에러 전파와 음성의 강조(prosody) 정보 활용 부족을 분석합니다. Chain-of-Thought (CoT) 메커니즘의 도입을 통해 이러한 문제를 해결할 수 있을 것으로 예상했으나, 실제 분석 결과 CoT가 주로 텍스트 전사(transcript)에 의존하고 음성을 거의 활용하지 않는다는 점을 발견하였습니다. 이 논문은 CoT의 가정을 의심하게 하며, 번역에 음성 정보를 통합하는 새로운 아키텍처의 필요성을 강조합니다.

- **Technical Details**: 연구에서는 CoT S2TT 모델의 음성 단서에 대한 기여도를 분석하기 위해 다양한 방법론을 적용했습니다. Speech LLM 아키텍처를 채택하여 자동 음성 인식(ASR) 데이터를 기반으로 한 모델을 특히 확장하고, 두 가지 추론 전략인 CoT 및 Cascade를 비교합니다. 또한, 음성 단위를 사용하여 훈련 중 음성 신호 의식을 강조하고, 누적 값 제로화(Value Zeroing) 기법으로 모델이 어떤 입력에 얼마나 의존하는지를 분석합니다.

- **Performance Highlights**: 모델의 성능은 노이즈가 포함된 전사를 사용하여 에러 전파를 평가하는 방식으로 검증했습니다. 실험 결과, CoT 모델이 음성을 활용하지 못하고 일반적으로 Cascade 모델과 유사한 성능을 보였습니다. 추가적으로, prosody 인식을 평가한 결과 역시 예상보다 낮은 성과를 보였으며, 이는 CoT가 음성을 이용한 의미 분별에서 제한적임을 나타냅니다.



### Semantic Similarity in Radiology Reports via LLMs and NER (https://arxiv.org/abs/2510.03102)
- **What's New**: 이 논문에서는 방사선 보고서의 평가에 대한 새로운 방법론을 제시합니다. 기존의 방법들이 갖고 있는 한계를 극복하기 위해 대형 언어 모델(LLMs)과 명명된 개체 인식(NER)을 결합한 Llama-EntScore라는 방법론을 도입했습니다. 이 새로운 접근법은 보고서 간의 의미적 유사성을 정량적으로 평가하고, 교육적인 피드백을 제공할 수 있습니다.

- **Technical Details**: 제안된 Llama-EntScore는 Llama 3.1 모델과 NER을 사용하여 방사선 보고서의 의미적 유사성을 평가합니다. 이 방법은 두 보고서 사이의 일치하는 개체를 추출하고, 각 개체의 의미 변화를 LLM을 통해 분석함으로써 점수와 해석을 제공합니다. 이로 인해 교육 중인 주니어 방사선의사가 자신의 보고서에서 반복적으로 나타나는 격차를 확인하고 개선할 수 있도록 돕습니다.

- **Performance Highlights**: Llama-EntScore 방법은 방사선 의사들이 제공한 기준 점수와 비교해 67%의 정확한 일치를 기록했습니다. 또한, +/- 1의 범위 내에서 93%의 정확도로 성능을 발휘하여 LLM과 NER을 개별적으로 사용하는 것보다 우수한 성과를 보여줍니다. 이와 같은 결과는 보고서 품질과 일관성을 평가하는 데 있어 AI 도구의 가능성을 시사합니다.



### Revisiting Direct Speech-to-Text Translation with Speech LLMs: Better Scaling than CoT Prompting? (https://arxiv.org/abs/2510.03093)
- **What's New**: 최근 스피치-투-텍스트 번역(S2TT) 분야에서 대규모 언어 모델(LLM) 기반 모델과 체인 오브 생각(Chain-of-Thought, CoT) 프롬프트 방법론에 대한 연구가 진행되고 있습니다. CoT는 우선 음성을 기록한 뒤 이를 번역하는 방식으로, 자동 음성 인식(ASR) 및 텍스트-투-텍스트 번역(T2TT) 데이터셋을 활용하여 성능을 향상시킵니다. 본 논문에서는 CoT와 직접 프롬프트(Direct prompting)를 비교하여 S2TT 데이터 양이 증가함에 따라 성능 차이를 분석합니다. 연구 결과, 데이터가 커질수록 Direct 방법이 더 안정적인 성능 향상을 보인다는 점이 주목할 만합니다.

- **Technical Details**: 본 연구에서는 CoT와 Direct 프롬프트 전략을 체계적으로 비교하기 위해, NSF(Normalized Speech Feature) 데이터셋을 생략하고 ASR 데이터셋의 전사 내용을 여섯 개 유럽 언어로 번역하여 가짜 라벨링(pseudo-labeling)된 S2TT 데이터를 생성하였습니다. Quality Estimation(QE) 시스템을 통해 생성된 샘플의 품질을 평가하고, 언어 식별(Language Identification, LID) 시스템을 통해 잘못된 언어로 번역된 샘플을 제거합니다. 사용된 기반 LLM은 salamandraTA-7B-Instruct로, 35개 유럽어에 대해 높은 성능을 보입니다.

- **Performance Highlights**: 모델 성능 평가를 위해 Word Error Rate(WER) 지표를 사용하며, 다양한 데이터 양에 따른 S2TT 모델을 훈련합니다. 결과적으로, Direct 모델은 CoT 모델에 비해 데이터 양이 증가함에 따라 더 일관되게 성능이 향상되는 것으로 나타났습니다. 이는 E2E 시스템의 발전과 유사한 경향을 보이며, 향후 풍부한 스피치 주석이 S2TT 성능 개선에 기여할 가능성을 시사합니다.



### Semantic Differentiation in Speech Emotion Recognition: Insights from Descriptive and Expressive Speech Roles (https://arxiv.org/abs/2510.03060)
Comments:
          Accepted to the *SEM conference collocated with EMNLP2025

- **What's New**: 본 연구는 Speech Emotion Recognition (SER) 분야에서 감정 역할에 대한 새로운 관점을 제시합니다. 전통적인 SER 접근 방식에서 벗어나, 발화의 의도된 감정과 유도된 감정 간의 구분을 강조합니다. 참가자들이 영화 클립을 본 후 자신의 감정 반응을 평가하면서, 설명적 의미(descriptive semantics)와 표현적 의미(expressive semantics)를 구분하는 방법을 제안하였습니다.

- **Technical Details**: 연구 방법론은 세 가지 주요 단계로 구성됩니다: 자동 음성 인식(ASR)을 통한 음성 전사, 대규모 언어 모델(LLMs)을 활용한 의미 기반 구분, 텍스트 분류기/회귀기를 이용한 감정 예측입니다. 데이터셋은 여섯 가지 감정 범주에 걸쳐 582개의 오디오 녹음으로 구성되며, 의도된 감정과 유도된 감정이 각각 측정됩니다. 이 과정에서 설명적 세그먼트는 의도된 감정과 더 높은 일치를 보이며, 표현적 세그먼트는 유도된 감정과 잘 연관된다는 것을 보여줍니다.

- **Performance Highlights**: 이 연구의 결과는 감정 인식 시스템 설계에 중요한 함의를 제공합니다. 설명적 의미와 표현적 의미의 구분은 SER 연구의 기존 한계를 극복할 수 있는 원리를 제공합니다. 이 접근법을 통해 감정 인식의 정확도를 높일 수 있으며, 가상 비서나 심리 건강 지원 시스템과 같은 다양한 응용 분야에서 활용될 수 있는 가능성을 보여줍니다.



### Grounding Large Language Models in Clinical Evidence: A Retrieval-Augmented Generation System for Querying UK NICE Clinical Guidelines (https://arxiv.org/abs/2510.02967)
- **What's New**: 이 논문은 영국의 건강 및 치료 우수성 국가 연구소(NICE)의 임상 가이드라인을 질의하기 위한 Retrieval-Augmented Generation (RAG) 시스템 개발 및 평가를 다룹니다. 이 시스템은 사용자가 자연어 질의에 따라 정밀하게 일치하는 정보를 제공하도록 설계되었으며, 대규모 언어 모델(LLMs)을 사용하여 임상 가이드라인의 긴 길이와 대량의 데이터를 활용할 수 있도록 합니다.

- **Technical Details**: RAG 시스템은 하이브리드 임베딩 메커니즘으로 구성된 검색 아키텍처를 사용하여 300개 가이드라인에서 유도된 10,195개의 텍스트 청크 데이터베이스와 비교 평가되었습니다. 이 시스템은 7,901건의 질의에서 평균 역순위(Mean Reciprocal Rank, MRR) 0.814를 기록하고, 첫 번째 청크에서 81%, 상위 10개 청크에서 99.1%의 리콜(validation accuracy)을 달성하는 성과를 보였습니다.

- **Performance Highlights**: RAG 시스템은 생성 단계에서 가장 두드러진 성과를 보였으며, 70개의 질문-답변 쌍을 수작업으로 선별한 데이터셋에 대해 RAG 강화 모델이 성능에서 상당한 개선을 보였습니다. RAG 강화 O4-Mini 모델의 신뢰성(Faithfulness)은 64.7% 증가하여 99.5%에 달하며, 의학 중심의 Meditron3-8B LLM의 43%와 비교하여 훨씬 높은 성능을 보여주었습니다. 이 연구는 RAG를 의료 분야에서 생성 AI를 효과적이고 신뢰성 높게 적용할 수 있는 접근 방식으로 입증하였습니다.



### Leave No TRACE: Black-box Detection of Copyrighted Dataset Usage in Large Language Models via Watermarking (https://arxiv.org/abs/2510.02962)
- **What's New**: TRACE는 대형 언어 모델(LLM)에서 저작권 데이터셋 사용을 완전한 블랙박스 조건에서 검출하기 위한 실용적인 프레임워크 입니다. 기존의 수위 측정 기법들은 데이터 품질과 작업 성능을 저하시켰으나, TRACE는 왜곡 없는 워터마크를 사용하여 품질을 유지하면서도 신뢰성을 높였습니다. 이는 법적 저작권 보호를 위한 중요한 기술적 진전을 나타냅니다.

- **Technical Details**: TRACE는 데이터셋 사용 검출을 통계적 가설 시험 문제로 형식화하여, LLM에 대한 출력-입력 쌍 검사를 통해 저작권 데이터셋 사용 여부를 판단합니다. 이 절차는 모델의 내부 매개변수나 로짓에 접근하지 않고, 오히려 입출력을 통해 수집된 데이터를 기반으로 합니다. 엔트로피 기반 방법을 사용하여 모델의 출력에서 높은 불확실성 토큰을 선택적으로 점수화함으로써, 검출 능력을 강력하게 증대시킵니다.

- **Performance Highlights**: TRACE는 다양한 데이터셋과 모델에서 통계적으로 유의미한 검출(p<0.05)을 지속적으로 달성하며, 다수의 모델과 데이터에 대한 검증이 가능합니다. 또한 다중 데이터셋 귀속을 지원하고, 대형 비워터마크 데이터에 대한 지속적인 사전 훈련 이후에도 깨지지 않는 신뢰성을 보여줍니다. 이러한 결과들은 TRACE가 신뢰할 수 있는 블랙박스 신원을 검증하는 현실적인 경로로 자리매김하게 합니다.



### Finding Diamonds in Conversation Haystacks: A Benchmark for Conversational Data Retrieva (https://arxiv.org/abs/2510.02938)
Comments:
          Accepted by EMNLP 2025 Industry Track

- **What's New**: 이번 연구에서는 Conversational Data Retrieval (CDR) 벤치마크를 소개합니다. 이는 제품 인사이트를 위해 대화 데이터를 검색하는 시스템을 평가하기 위한 첫 번째 포괄적인 테스트 세트입니다. 1.6k 쿼리와 9.1k 대화가 포함되어 있어 대화 데이터 검색 성능을 측정하는 신뢰할 수 있는 기준을 제공합니다.

- **Technical Details**: 이 벤치마크는 다섯 가지 분석 작업을 포함하며, 16개의 인기 있는 임베딩 모델을 평가하였습니다. 평가 결과, 가장 성능이 좋은 모델도 NDCG@10에서 약 0.51에 불과하여, 문서 및 대화 데이터 검색 능력 간의 상당한 격차를 드러냅니다. 대화 데이터 검색에서의 고유한 도전과제를 해결하기 위해 암시적 상태 인식(implicit state recognition), 턴 동역학(turn dynamics), 맥락적 참조(contextual references)와 같은 요소를 분석하였습니다.

- **Performance Highlights**: 각 작업 범주에 걸쳐 상세한 오류 분석과 실용적인 쿼리 템플릿을 제공하여 향후 연구의 방향을 제시합니다. 이러한 분석은 대화 데이터 검색 분야의 성능 개선을 위한 통찰을 제공합니다. 전체 벤치마크 데이터셋과 코드도 공개되어 있어 관련 연구자들에게 유용한 자료가 될 것입니다.



### Self-Reflective Generation at Test Tim (https://arxiv.org/abs/2510.02919)
Comments:
          24 pages, 8 figures

- **What's New**: 제안된 Self-Reflective Generation at Test Time (SRGen) 프레임워크는 LLM의 복잡한 추론 작업에서 발생하는 오류를 사전에 방지하는 경량화된 방법입니다. 기존의 오류 수정 메커니즘은 반복적이지 않으며 비효율적인 특성을 가지지만, SRGen은 동적 엔트로피 임계값을 사용하여 불확실한 토큰을 식별하고, 이를 통해 자가 반영(self-reflection) 과정을 실행합니다. 이를 통해 LLM의 reasoning을 향상시키고 오류의 발생 가능성을 줄입니다.

- **Technical Details**: SRGen은 두 가지 주요 단계를 포함하는 모니터-반영-최적화 루프를 통해 자가 반영을 구현합니다. 첫 번째 단계에서는 모델의 예측 불확실성을 평가하여 동적 임계값을 설정합니다. 두 번째 단계에서는 이 임계값을 초과할 경우 자가 반영 최적화를 수행하고, 생성된 숨겨진 상태(hidden state)에 보정 벡터를 적용하여 다음 토큰의 확률 분포를 개선합니다. 이는 전체적으로 LLM의 효율성과 신뢰성을 증가시키는 역할을 합니다.

- **Performance Highlights**: SRGen은 다양한 모델에서 수학적 추론 벤치마크에 대해 평가되었으며, 단일 패스에서의 정확도를 크게 향상시켰습니다. 예를 들어, AIME2024 컨소시엄에서는 DeepSeek-R1-Distill-Qwen-7B가 Pass@1에서 +12.0%, Cons@5에서 +13.3%의 절대적인 개선을 달성했습니다. 또한, SRGen은 다른 테스트 타임 방법과 잘 결합되어 추가적인 성과를 내며, 적은 오버헤드 범위에서도 강력한 성과를 보입니다.



### Constraint Satisfaction Approaches to Wordle: Novel Heuristics and Cross-Lexicon Validation (https://arxiv.org/abs/2510.02855)
Comments:
          35 pages, 14 figures, 10 tables. Open-source implementation with 91% test coverage available at this https URL

- **What's New**: 이 논문은 Wordle을 제약 충족 문제(Constraint Satisfaction Problem, CSP)로 체계화하여 새로운 제약 인식 솔루션 전략을 소개합니다. 기존의 솔버들은 정보 이론에 기반한 엔트로피 최대화나 빈도 기반 휴리스틱에 의존했지만, 본 연구는 제약과 정보를 동시에 고려하는 CSP-Aware Entropy를 제시하여 3.54회의 평균 추측으로 99.9%의 성공률을 달성했습니다.

- **Technical Details**: CSP-Aware Entropy는 제약 전파 이후 정보 이득을 계산하며, 확률적 CSP 프레임워크를 통해 베이지안 단어 빈도 사전과 논리적 제약을 통합합니다. 이 논문에서는 제약 기반 접근법들이 낮은 노이즈 환경에서 우수한 성능을 기록하며, 10% 노이즈 하에서도 5.3% 포인트의 이점을 유지한 결과를 제시합니다.

- **Performance Highlights**: 2,315개의 영어 단어에 대한 평가에서 CSP-Aware Entropy는 99.9%의 성공률을 기록하며, 1.7%의 통계적으로 유의미한 개선을 보였습니다. 또한 스페인어 500개 단어에 대한 교차 어휘 검증 결과, 88%의 성공률을 나타내며, 언어 간 제약 충족 원리가 일반화 가능함을 보여주었습니다.



### Evaluating Large Language Models for IUCN Red List Species Information (https://arxiv.org/abs/2510.02830)
Comments:
          20 pages, 7 figures

- **What's New**: 본 연구에서는 21,955종의 생물에 대해 네 가지 핵심 IUCN 적색 목록 평가 요소인 분류학(taxonomy), 보전 상태(conservation status), 분포(distribution), 위협(threats)에 대해 다섯 개의 선도적인 대형 언어 모델(Large Language Models, LLMs)의 신뢰성을 체계적으로 검증했습니다. 결과적으로, 모델들은 분류학적 분류에서 94.9%의 정확도를 보였으나, 보전 추론에서 27.2%의 낮은 성능을 나타내며 인지-추론 간의 갭이 드러났습니다. 이 연구는 LLM이 정보 검색에서는 강력한 도구일 수 있지만, 판단 기반 결정에서는 인간의 감독이 필요하다는 점을 강조합니다.

- **Technical Details**: 연구는 2022-2023년 사이 IUCN 적색 목록에서 평가된 21,955 종에 대한 21,962개의 평가 기록을 분석했습니다. 데이터 세트에는 척추동물(vertebrates), 식물(plants), 무척추동물(invertebrates), 균(fungi) 등이 포함되었으며, 이들 대조적으로 무척추동물과 곰팡이가 상대적으로 적게 나타났습니다. 평가된 다섯 개의 LLM은 연구의 초점에 맞춰 공통된 조건에서 균일하게 평가되었으며, 각 모델은 최소한의 프롬프트(minimal prompting)를 사용하여 일관된 작업 해석을 보장했습니다. 이 연구에서 평가된 모델은 OpenAI의 GPT-4.1, xAI의 Grok 3, Anthropic의 Claude Sonnet 4 등의 공용 API 기반 시스템과 Google DeepMind의 Gemma 3-27B 및 Meta의 Llama 3.3-70B와 같은 로컬 호스팅 모델을 포함합니다.

- **Performance Highlights**: LLMs는 정보 검색에 있어 유용하지만, 생물 다양성 평가와 같은 고차원적 작업에 있어서는 전문가의 검증 없이 신뢰할 수 없는 성능을 보였습니다. 특히 특정 분류에서 성능의 불균형이 발견되어, 기존 보전 작업에 있어 어렵고 보존이 소외된 무척추동물 및 자생 식물에 대한 검토의 필요성이 제기됩니다. 따라서 LLM은 교육 및 데이터 탐색과 같은 분야에는 유용하나, 위험 평가 및 정책 수립에는 전문가의 관여가 필요하다는 결론에 도달했습니다.



### StepChain GraphRAG: Reasoning Over Knowledge Graphs for Multi-Hop Question Answering (https://arxiv.org/abs/2510.02827)
- **What's New**: StepChain GraphRAG는 Retrieval-Augmented Generation(RAG) 접근 방식을 기반으로 하여 복잡한 다중 홉 질문 대응을 위한 새로운 프레임워크를 소개합니다. 이 모델은 Breadth-First Search(BFS) 추론 흐름을 통합하여 질문을 재구성하고, 외부 지식을 효과적으로 연결하여 정보의 일관성을 유지합니다. 실험 결과, 이 모델은 여러 벤치마크에서 기존의 방법보다 우수한 성과를 나타냈으며, 특히 HotpotQA에서 큰 성과를 기록했습니다.

- **Technical Details**: 이 프레임워크는 전체 코퍼스에 대한 글로벌 인덱스를 구축하고, 추론 단계에서 선택된 패시지를 즉시 지식 그래프로 변환합니다. 각 서브 질문은 BFS 기반의 탐색을 통해 관련된 증거 체인을 구축하며, 이는 불필요한 문맥으로 모델을 압도하지 않도록 돕습니다. 또한, 프레임워크는 매 세부 질문에서 새로운 증거를 그래프에 동적으로 통합하여, 중간 추론 단계를 효과적으로 기록하고 업데이트합니다.

- **Performance Highlights**: StepChain GraphRAG는 MuSiQue, 2WikiMultiHopQA 및 HotpotQA에서 SOTA(state-of-the-art) 정확도 및 F1 점수를 달성했습니다. 이 방법은 평균적으로 EM이 2.57%, F1이 2.13% 향상되었으며, 특히 HotpotQA에서 EM이 4.70%, F1이 3.44% 향상되었습니다. 이러한 성과는 중간 단계에서의 사고 과정을 명확히 유지하여 설명 가능성을 높이는 데 기여했습니다.



### A Computational Framework for Interpretable Text-Based Personality Assessment from Social Media (https://arxiv.org/abs/2510.02811)
Comments:
          Phd thesis

- **What's New**: 이 논문은 성격( Personality ) 분석을 위한 두 개의 데이터셋인 MBTI9k 및 PANDORA를 소개합니다. 이 데이터셋은 Reddit에서 수집된 것으로, 1,700만 개의 댓글과 10,000명 이상의 사용자 데이터를 포함합니다. 이러한 데이터는 개인화된 분석을 위한 기초를 마련하며, 성격 심리와 자연어 처리(NLP)의 간극을 메우기 위한 노력을 다룹니다.

- **Technical Details**: 성격 분석을 위해 삼성의 SIMPA (Statement-to-Item Matching Personality Assessment) 프레임워크를 개발하여 사용자 생성의 진술이 검증된 질문 항목과 매칭됩니다. 이 과정에서 기계 학습 및 의미론적 유사성을 활용하여 사람의 평가와 유사한 결과를 도출하면서도 해석 가능성과 효율성을 유지합니다. 이 프레임워크는 복잡한 라벨 분류를 다룰 수 있는 유연성을 제공합니다.

- **Performance Highlights**: 체험 결과는 인구 통계 변수들이 모델의 유효성에 영향을 미친다는 점을 보여주었습니다. SIMPA를 통해 기계 학습 기술을 활용하여 인사이트를 제공하며, 인간의 직관과 비교할 수 있는 성격 평가를 진행할 수 있습니다. 이는 성격 분석 외에도 다양한 연구 및 실용적인 응용 프로그램에 적합한 가능성을 제공합니다.



### XTRA: Cross-Lingual Topic Modeling with Topic and Representation Alignments (https://arxiv.org/abs/2510.02788)
Comments:
          2025 EMNLP Findings

- **What's New**: 이번 연구에서는 언어 간 주제 모델링의 새로운 프레임워크인 XTRA (Cross-Lingual Topic Modeling with Topic and Representation Alignments)를 제안합니다. XTRA는 Bag-of-Words (BoW) 모델과 다국어 임베딩을 통합하여 두 가지 핵심 구성 요소인 문서-주제 정렬(representation alignment)과 주제 정렬(topic alignment)을 도입합니다. 이를 통해 서로 다른 언어 간의 주제를 해석 가능하고 잘 정렬된 형태로 학습할 수 있습니다.

- **Technical Details**: XTRA는 경량의 MLP를 사용하여 BoW 입력을 공유 공간으로 투영하고, 클러스터링 기반의 대조적 학습(constrastive learning) 목표를 통해 문서-주제 분포를 정제합니다. 또한, 주제-단어 분포를 정렬하기 위해 학습 가능한 변환 레이어를 사용하여 각 언어의 주제-단어 분포를 동일한 의미 공간에 투영합니다. 이러한 이중 대조적 설계는 고품질의 문서-주제 구조와 의미적으로 정렬된 주제-단어 의미를 학습할 수 있게 합니다.

- **Performance Highlights**: 다국어 데이터셋에 대한 실험 결과, XTRA는 주제 일관성(topic coherence), 다양성(topic diversity), 그리고 언어 간 정렬 품질에서 기존의 CLTM 모델들을 초월하는 성능을 보였습니다. 이 결과는 XTRA가 다국어 환경에서도 효과적으로 작동하며, 주제 모델링에 있어 중요한 발전임을 보여줍니다. 연구에 사용된 코드와 재현 가능한 스크립트는 공개되어 있어 실험을 반복할 수 있습니다.



### The Path of Self-Evolving Large Language Models: Achieving Data-Efficient Learning via Intrinsic Feedback (https://arxiv.org/abs/2510.02752)
- **What's New**: 이 연구에서는 대량의 데이터를 요구하지 않고도 강화 학습(RL)을 통해 대형 언어 모델(LLMs)을 개선하는 새로운 접근 방식을 제안합니다. 이 방식은 모델이 스스로 과제를 제안하고 이를 해결하려고 시도하는 것을 교대로 수행하는 것입니다. 자가 인식에 기반한 두 가지 새로운 메커니즘인 자가 인식 난이도 예측(self-aware difficulty prediction)과 자가 인식 한계 돌파(self-aware limit breaking)를 도입하여 데이터 의존성을 최소화합니다.

- **Technical Details**: 자가 인식 난이도 예측 메커니즘을 통해 모델은 과제를 생성할 뿐만 아니라 자신의 능력에 따라 과제의 난이도를 예측합니다. 이는 모델이 자신의 성공률과 일치하도록 난이도를 조정할 수 있게 도와주며, 적절한 난이도의 문제를 우선적으로 다룰 수 있도록 합니다. 자가 인식 한계 돌파 메커니즘은 모델이 생성한 과제가 현재 능력으로는 어려운 경우 외부 데이터를 요청하는 방식으로, 이러한 과제를 해결하기 위한 외부 지침을 적극적으로 요청합니다.

- **Performance Highlights**: 실험 결과, 제안된 자가 인식 RL 접근 방식은 아홉 개의 벤치마크에서 평균 53.8%의 성능 향상을 기록했습니다. 특히, 수학적 추론 벤치마크에서 MATH500에서 29.8%, AMC'23에서 77.8%, OlympiadBench에서 82.4%, LiveCodeBench에서 22.3%의 성능 향상을 보였습니다. 이러한 결과는 자가 인식 RL이 모델 성능 및 일반화 능력을 크게 향상시킬 수 있음을 보여줍니다.



### IndiCASA: A Dataset and Bias Evaluation Framework in LLMs Using Contrastive Embedding Similarity in the Indian Contex (https://arxiv.org/abs/2510.02742)
Comments:
          Accepted at 8th AAAI/ACM Conference on AI, Ethics, and Society (AIES) 2025

- **What's New**: 이 논문은 대규모 언어 모델(LLMs)의 문화적 맥락에서의 내재적 편견을 평가하기 위한 새로운 프레임워크를 제안합니다. 특히, 인도와 같은 다양한 문화적 배경을 가진 환경에서 기존의 편향 평가 방식이 미흡한 점을 기초로 합니다. 연구진은 대조 학습을 통해 훈련된 인코더를 사용하여 미세한 편견을 캡처하는 새로운 데이터셋인 IndiCASA를 도입하였습니다.

- **Technical Details**: 연구에서는 IndiCASA 데이터를 활용하여 대규모 언어 모델의 편견을 측정하는 방법론을 개발했습니다. 편향을 평가하기 위해 주어진 문화적 맥락의 다양한 프롬프트에 따른 무제한 생성을 분석합니다. 논문은 이미 다양한 사회적 편향을 평가한 IndiBias 데이터셋을 바탕으로 하여, 정의된 옵션에 제한되지 않고 편견을 평가할 수 있는 방법론을 제시합니다.

- **Performance Highlights**: 여러 공개 모델을 평가한 결과, 모든 모델이 일부 형태의 고정관념적 편견을 보였으며, 특히 장애 관련 편견이 두드러졌습니다. 종교적 편견은 상대적으로 낮았으며, 이는 글로벌 디바이싱(debiasing) 노력으로 인해 나타난 결과로 해석됩니다. 이 연구는 공정한 모델 개발의 필요성을 강조하며, 새로운 데이터셋과 방법론이 기여할 수 있는 가능성을 보여주고 있습니다.



### PGMEL: Policy Gradient-based Generative Adversarial Network for Multimodal Entity Linking (https://arxiv.org/abs/2510.02726)
- **What's New**: 이 논문은 멀티모달 엔티티 링크(Multimodal Entity Linking, MEL) 문제를 다루기 위해 생성적 적대 신경망(Generative Adversarial Network, GAN)을 활용하는 새로운 접근 방식을 제안합니다. 특히, 고품질의 네거티브 샘플을 생성하는 생성기(generator)를 통해 메트릭 학습(metric learning)을 더욱 효과적으로 수행할 수 있도록 하고 있습니다. 기존 연구들에서는 네거티브 샘플 선택의 중요성이 간과되었으나, 본 논문은 이를 채우기 위한 첫 번째 시도로 주목받고 있습니다.

- **Technical Details**: 본 연구는 PGMEL(Policy gradient-based Generative adversarial Network for Multi-modal Entity Linking)이라는 새로운 프레임워크를 제안하며, 이는 정책 경량화(policy gradient) 기법을 사용하여 생성기를 최적화합니다. 논문에서는 멀티모달_reprresentation learning을 통해 각 데이터 모달리티의 통계적 속성을 보존하고, 서로 다른 특성을 활용하여 효과적인 표현을 학습하는 방법을 설명하고 있습니다. 주목할 점은 생성기와 구별기(discriminator)의 상호작용을 통해 고품질 임베딩을 생성하고, 메트릭 학습이 강하게 이루어질 수 있도록 한다는 것입니다.

- **Performance Highlights**: 실험 결과 PGMEL은 Wiki-MEL, Richpedia-MEL 및 WikiDiverse 데이터 세트를 기반으로 하여 도전적인 네거티브 샘플을 선택함으로써 의미 있는 표현을 학습하고 기존의 최첨단 방법들을 초월하는 성과를 보여주었습니다. 연구진은 여러 텍스트 기반 엔티티 링크 및 멀티모달 엔티티 링크의 벤치마크를 통하여 PGMEL의 성능을 광범위하게 평가하였고, 네거티브 샘플 선택이 성능에 미치는 중요한 영향을 입증했습니다.



### TravelBench : Exploring LLM Performance in Low-Resource Domains (https://arxiv.org/abs/2510.02719)
Comments:
          10 pages, 3 figures

- **What's New**: 이번 연구에서는 저자들이 14개의 여행 도메인 데이터셋을 구성하여 기존 LLM 벤치마크에서 부족했던 저자원(low-resource) 작업에 대한 모델의 성능을 평가했습니다. 이를 통해 여행 산업에서의 LLM 성능을 더 잘 이해할 수 있는 기반을 마련했습니다. 이 연구는 기존의 opinion mining을 넘어선 다양한 작업을 포함하여 LLM의 평가를 위한 자원을 확장하고 있습니다.

- **Technical Details**: 연구팀은 현장 사용 사례에서 수집된 익명 데이터로부터 다양한 NLP 작업을 수행하기 위해 데이터를 수집하였으며, 이를 위해 데이터 수집 과정에서 개인 식별 정보를 제거했습니다. 데이터는 인간 전문가의 참여를 통해 주석이 달리고 검증되었으며, LLM의 잠재적 편향을 최소화하기 위한 지침이 마련되었습니다. 또한 LLM의 내부 지식과 지시 따르기 능력의 조합을 통해 문맥에 따른 예측 능력을 평가하고 있습니다.

- **Performance Highlights**: 저자들은 LLM 모델의 정확도, 확장성 행동, 추론 능력 등을 분석하고, LLM이 복잡한 도메인 별 작업에서 성능 병목 현상을 겪는다는 것을 확인했습니다. 또한, LLM의 훈련 과정에서 FLOPs와 모델 크기의 변화가 성능에 미치는 영향을 조사하였으며, 작은 LLM의 경우 더 나은 추론 능력이 성능을 높이는 데 기여한다는 것을 밝혔습니다.



### Time-To-Inconsistency: A Survival Analysis of Large Language Model Robustness to Adversarial Attacks (https://arxiv.org/abs/2510.02712)
- **What's New**: 이번 연구에서는 대화 AI의 면역력을 평가하기 위해 최초의 포괄적인 생존 분석(survival analysis)을 제시합니다. 9개의 최신 대형 언어 모델(LLMs)에 대해 36,951회의 대화 턴을 분석하여 대화가 실패하는 것을 시간에 따라 변하는 과정으로 모델링합니다. 급격한 의미 변동이 대화 실패의 위험을 극적으로 증가시키는 반면, 점진적이며 누적적인 변동은 보호 효과를 나타내는 등의 발견을 통해, 생존 분석을 LLM의 강건성을 평가하는 강력한 패러다임으로 자리매김하였습니다.

- **Technical Details**: 이 연구에서는 생존 분석을 통해 대화 모델의 일관성을 재구성하였습니다. 대화 실패를 '이벤트'로 모델링하고 시간은 대화 턴의 순서에 따라 측정합니다. 사용된 모델링 기법으로는 Cox 비례 위험 모델, 가속화 실패 시간(Accelerated Failure Time, AFT) 모델, 비모수적 랜덤 생존 포리스트(Random Survival Forest)가 있습니다. 이를 통해 다양한 실패 시간 분포와 위험 함수를 고려하여 강건한 결론을 도출하였습니다.

- **Performance Highlights**: 생존 분석을 통한 이 연구의 결과는 LLM의 대화 강건성 평가에서 강력한 통찰력을 제공합니다. 특히 AFT 모델이 상호작용을 보여주며 우수한 성과를 달성했는데, 이는 변별력과 보정(calibration)에서 뛰어난 결과를 보여주었습니다. 이와 같은 결과는 기존의 단일 턴 평가 방식이 놓치기 쉬운 대화 내 응답 불일치의 위험을 정량화하여 대화 AI 시스템의 신뢰성을 높이는데 기여할 수 있을 것입니다.



### Uncertainty as Feature Gaps: Epistemic Uncertainty Quantification of LLMs in Contextual Question-Answering (https://arxiv.org/abs/2510.02671)
- **What's New**: 이 연구는 Uncertainty Quantification (UQ) 분야에서 기존의 폐쇄형 사실 기반 질문 응답(QA)에서 벗어나 문맥 기반 QA에 초점을 두고 있습니다. 연구자들은 모델의 예측 분포와 진짜 분포 간의 교차 엔트로피를 통한 토큰 수준의 불확실성 측정 방법을 소개합니다. 이러한 접근법을 통해 이념적 모델 기준으로 에피스테믹 불확실성을 수량화할 수 있습니다.

- **Technical Details**: 제안된 방법은 모델의 마지막 레이어 숨겨진 상태와 이상적인 모델 간의 거리를 통해 에피스테믹 불확실성을 한계 지을 수 있음을 보여줍니다. 이 거리는 독립적인 모델 기능들의 거리의 합으로 표현될 수 있으며, 이를 통해 세 가지 주요 특징인 문맥 의존성(context-reliance), 문맥 이해(context comprehension), 정직성(honesty)을 도출합니다. 연구에서는 이 세 가지 특징을 활용하여 주어진 질문-문맥 쌍에 대한 모델의 불확실성을 효율적으로 평가합니다.

- **Performance Highlights**: 연구 결과, 제안된 방법은 기존의 비지도 및 지도 UQ 방법보다 최대 16 PRR 포인트 향상된 성능을 보이며, 동일한 라벨된 샘플을 사용하여 강력한 지도 기반 방법인 SAPLMA 및 LookbackLens를 각각 13 PRR 포인트 초과하여 능가했습니다. 또한, 이 방법은 분포 외 일반화 성능에서도 SAPLMA보다 월등히 뛰어난 성능을 발휘합니다.



### Self-Improvement in Multimodal Large Language Models: A Survey (https://arxiv.org/abs/2510.02665)
Comments:
          EMNLP 2025

- **What's New**: 본 논문은 멀티모달 대형 언어 모델(MLLMs)의 자기 개선(self-improvement) 영역에 대한 포괄적인 개요를 제공합니다. 연구자들이 MLLMs가 스스로 트레이닝 데이터를 수집하고 조직하도록 해, 서서히 개선된 모델을 구축할 수 있는 가능성을 탐구하고 있습니다. 기존 연구들은 이 주제에 대한 체계적인 조망을 제공하지 않았기 때문에, 이 논문은 문헌 정리(overview)와 다양한 연구 방법론을 조합하여 새로운 접근법을 제시합니다.

- **Technical Details**: 자기 개선(self-improvement)의 개념은 모델이 더 나은 자기 생성 모델을 구축하기 위해 필요한 데이터를 수집하고 조직하는 데 중점을 둡니다. 특히 MLLMs의 경우, 데이터 수집(data collection), 데이터 조직(data organization), 모델 최적화(model optimization) 세 가지 관점에서 현존하는 문헌을 분석하고 현재 사용중인 평가 방법과 하류 애플리케이션을 논의합니다. 이 논문은 MLLMs의 성능을 개선하기 위한 다양한 방법들의 차이점을 정리하고, 평가 기준들을 나열합니다.

- **Performance Highlights**: MLLMs의 자기 개선은 점점 더 많은 연구자들에 의해 탐구되고 있으며, 초기 연구 결과들이 이 영역의 잠재력을 보여주고 있습니다. 예를 들어, 기존 접근 방식과 결합하여 부분적으로 자기 개선을 활용하는 연구들이 이루어지고 있으며, 자체적으로 모든 프로세스를 처리할 수 있는 단일 모델을 탐색하는 연구도 있습니다. 이 논문은 MLLMs의 성능 개선을 위한 효과적인 방법론을 소개하고, 가능한 도전 과제들을 제시하여 향후 연구 방향에 대한 명확한 지침을 제공합니다.



### SoT: Structured-of-Thought Prompting Guides Multilingual Reasoning in Large Language Models (https://arxiv.org/abs/2510.02648)
Comments:
          EMNLP 2025 (findings)

- **What's New**: 최근 대형 언어 모델(LLMs)의 발전은 복잡한 추론 작업을 수행할 수 있는 능력을 부각시켰습니다. 하지만, 비고급 리소스 언어에서는 이러한 추론 능력을 성공적으로 전이하기 어려운 상황입니다. 본 논문에서는 다국어 추론 성능을 향상시키기 위해 Structured-of-Thought(SoT)라는 훈련이 필요 없는 방법을 제안하고 있습니다. SoT는 언어 간 추론 단계를 개선하는 다단계 변환을 포함하여, 다양한 언어로 된 쿼리를 더욱 세밀하게 이해할 수 있도록 합니다.

- **Technical Details**: SoT는 언어별 의미 정보에서 언어 무관(structured representations) 구조화 표현으로 변환하여, LLM이 다양한 언어의 쿼리를 더 정교하게 이해하도록 돕습니다. 이 방법은 언어적 사고 변환(Language Thinking Transformation)과 구조적 지식 변환(Structured Knowledge Transformation)의 두 가지 변환 단계를 포함하며, 자연어 쿼리를 구조적 지식 표현으로 변환하여 LLM의 추론 경로를 일관되게 유지할 수 있도록 지원합니다. SoT는 훈련이 필요 없는 조건에서 다국어 추론 성능을 향상시키는 것으로, 다양한 LLM의 백본(backbone) 모델에도 적용이 가능합니다.

- **Performance Highlights**: 실험 결과, SoT는 특정 수학 및 상식 추론 작업에서 여러 강력한 기준선(baselines)을 초과하는 성능을 보여주었습니다. SoT는 훈련이 필요 없는 전략들, 예를 들어 In-Context Learning(ICL) 및 Chain-of-Thought(CoT)과 결합하여 더욱 향상된 성능을 나타내고 있습니다. 이와 함께 SoT는 구조적 지식을 효과적으로 이해함으로써 다양한 크기의 LLM에 적응할 수 있어, 여러 다국어 추론 기준 벤치마크에서 유용성을 입증하였습니다.



### Mind the Gap: Linguistic Divergence and Adaptation Strategies in Human-LLM Assistant vs. Human-Human Interactions (https://arxiv.org/abs/2510.02645)
Comments:
          Accepted to The Second Workshop on Generative AI for E-commerce (GenAIECommerce '25), held September 22, 2025, in Prague, Czech Republic

- **What's New**: 본 연구는 사람들이 LLM 챗봇과 인간 상담원 사이에서 사용하는 커뮤니케이션 스타일이 다름을 보여주는 실증적 증거를 제공하고 있습니다. 표준적으로 인간 간의 상호작용 데이터를 기반으로 훈련된 모델은 LLM이 배포된 후의 커뮤니케이션 스타일 변화에 적절히 대처하지 못할 수 있습니다. 특히, 사용자 메시지가 LLM을 대상으로 할 때 더 간결하고 문법적으로 저하되며 정중함이 떨어지는 경향이 있음을 확인했습니다.

- **Technical Details**: 본 연구에서는 LLM과 인간 상담원 간의 다중 발화 대화에서 사용자 메시지를 분석하였으며, 6가지 언어적 및 의미적 차원에서 사용자 발화를 정량적으로 평가했습니다. 분석 결과, LLM에 대해 메시지를 보낼 때는 문법 유창성, 정중함, 어휘 다양성이 낮아지는 경향이 있음을 발견했습니다. 이러한 연구 접근법은 데이터 증강(data augmentation) 및 추론 중 사용자 메시지 재구성(inference-time user message reformulation) 두 가지 방법으로 LLM의 내구성을 향상시키기 위한 실험을 포함하고 있습니다.

- **Performance Highlights**: 우리는 목표 문장을 단순화하여 생성된 훈련 데이터가 다양한 스타일을 포함할 때 LLM의 성능은 유의미하게 향상됨을 발견했습니다. 반면, 추론 시 스타일 정상화(inference-time style normalization)는 효과적인 결과를 보여주지 못했습니다. 이러한 발견은 자연스러운 스타일 다양성에 대한 훈련이 필수적이라는 점을 강조하며, 이는 사용자와의 상호작용을 개선하기 위한 더 적응적이고 문맥에 민감한 LLM 개발로 이어질 수 있습니다.



### Evaluation Framework for Highlight Explanations of Context Utilisation in Language Models (https://arxiv.org/abs/2510.02629)
- **What's New**: 이 논문은 언어 모델(LMs)의 문맥 활용(context utilisation) 기법을 설명하는 데 있어 하이라이트 설명(HEs)의 효과성을 평가하기 위한 새로운 기준 평가 프레임워크를 제안합니다. HEs는 모델 출력에 영향을 미친 구체적인 문맥 조각을 지적할 수 있어 사용자에게 투명성을 제공합니다. 그러나 기존의 연구에서는 이러한 HEs의 효과를 직접적으로 평가하지 않았습니다.

- **Technical Details**: 이 연구는 네 가지 통제된 평가 시나리오(모순, 무관, 혼합, 이중 모순)를 구성하고, 이전에 사용된 HE 방법들을 평가합니다. 이들 시나리오는 문맥 사용 패턴을 체계적으로 변화시킴으로써 다양한 행동에 대한 강건한 HE 평가를 가능하게 합니다. 특히, MechLight라는 기계적 해석 가능성을 바탕으로 한 새로운 HEs 기법이 모든 문맥 시나리오에서 최상의 성과를 보였습니다.

- **Performance Highlights**: 그러나 모든 HE 방법들은 긴 문맥에 대해 정확도가 떨어지고 위치적 편향이 나타나는 문제를 가지고 있습니다. IG와 MechLight 같은 방법은 특정 문맥 시나리오에서 낮은 정확도를 보이며, 이는 모델의 문맥 활용을 명확히 드러내지 못하는 것으로 나타났습니다. 이러한 결과는 새로운 방식의 설명 기법이 필요함을 강조합니다.



### Transcribe, Translate, or Transliterate: An Investigation of Intermediate Representations in Spoken Language Models (https://arxiv.org/abs/2510.02569)
Comments:
          ASRU 2025

- **What's New**: 이번 연구에서는 음성 언어 모델(Spoken Language Models, SLMs)에서 모달리티 어댑터(Modality Adapters, MAs)의 출력을 분석하여 이들이 어떤 방식으로 표현을 변형하는지를 규명합니다. SALMONN, Qwen2-Audio, Phi-4-Multimodal-Instruct의 세 가지 대표 모델을 대상으로 하여, Whisper 인코더를 사용하는 경우와 사용하지 않는 경우의 차이를 비교하였습니다. 연구 결과, Whisper 인코더를 사용한 모델은 입력 언어와 무관하게 영어 기반의 표현을 출력하며, 다른 모델은 영어 단어로 표현된 음운적(phonetic) 표기 방식을 사용함을 발견하였습니다.

- **Technical Details**: 연구에서는 모달리티 어댑터의 출력으로부터 언어 모델 토큰(Language Model Token)에 가장 가까운 토큰을 찾는 방법을 사용하여 마다출력의 의미를 해석하고 있습니다. SALMONN과 Qwen2-Audio와 같은 Whisper 인코더 기반 모델은 대부분 영어 표현을 출력하며, Phi-4-Multimodal-Instruct는 음운적 표현을 보이는 것을 관찰하였습니다. 이 과정에서 선형 프로브(linear probes)를 사용하여 음성 인코더 표현과 비교하여 각 모델의 언어적 정보를 분석하였습니다.

- **Performance Highlights**: 연구 결과, Whisper 인코더로 훈련된 모델이 지역적 음 절 및 단어 정확성에서는 손실을 보였으나 의미적 정보는 증가한 반면, Phi-4-Multimodal-Instruct의 경우에는 단어와 음절의 정확성이 증가하고, 전반적인 의미 정보 또한 개선된 것으로 나타났습니다. 이러한 결과는 다양한 모달리티 어댑터의 성능 분석을 보다 명확히 할 수 있는 근거를 제공합니다. 이에 따라, 향후 음성 처리 작업에서 모달리티 어댑터의 역할을 보다 깊이 이해할 수 있을 것으로 기대됩니다.



### Knowledge-Graph Based RAG System Evaluation Framework (https://arxiv.org/abs/2510.02549)
- **What's New**: 이번 연구에서는 Retrieval Augmented Generation (RAG) 시스템의 평가 방법을 혁신적으로 개선하기 위해 Knowledge Graph (KG)를 기반으로 한 평가 프레임워크를 제안합니다. 이 새로운 접근 방식은 기존의 RAGAS 평가 기준을 확장하여 멀티-hop reasoning과 의미적 커뮤니티 클러스터링을 통합하였습니다. 연구 결과, KG 기반 메트릭이 기존 RAGAS보다 사실성과 신뢰성 평가에서 더 우수한 성능을 보여주었습니다.

- **Technical Details**: RAG 시스템은 두 가지 주요 구성 요소인 retriever와 generator로 이루어져 있습니다. Retriever는 주어진 입력에 따라 관련 정보를 검색하고, 이를 바탕으로 generator가 최종 출력을 생성합니다. 그러나 기존의 RAG는 여러 정보 출처를 통합하는 데 한계가 있어, 최신 연구에서는 KG를 활용한 GraphRAG 아키텍처가 개발되었습니다. 이를 통해 정보의 통합 구조를 개선하고 반응 품질과 정확도를 높였습니다.

- **Performance Highlights**: 연구 결과, KG 기반의 새로운 평가 시스템은 RAGAS에서 측정한 점수와 인간의 판단 결과 간의 높은 상관관계를 나타냈습니다. 특히, KG 기반 방법은 극단적인 상황에서 질문을 비교할 때 RAGAS를 초월하며 더 정밀한 사실 일치성과 의미적 일관성을 포착할 수 있는 능력을 보여주었습니다. 이러한 결과는 KG 기반 접근 방식의 효과적인 특성을 강조하며, 향후 RAG 시스템 평과의 방향성을 제시합니다.



### Hierarchical Semantic Retrieval with Cobweb (https://arxiv.org/abs/2510.02539)
Comments:
          20 pages, 7 tables, 4 figures

- **What's New**: 본 논문은 Cobweb이라는 계층 인식 프레임워크를 사용하여 문서 검색 방식을 혁신적으로 개선합니다. 기존의 신경망 기반 검색 방법들이 단일 유사도 점수로 문서의 관련성을 평가하는 대신, Cobweb은 문서들을 구조화된 트리 형태로 조직하여 다중 수준의 관련성 신호를 제공합니다. 이를 통해 사용자는 검색 경로를 통해 명확한 해석을 얻을 수 있는 장점을 누립니다.

- **Technical Details**: Cobweb는 문서 임베딩을 계층적으로 조직화하는 방법으로, 내부 노드가 개념 프로토타입 역할을 하여 문서 간의 관련성을 다층적으로 평가합니다. 제안된 두 가지 추론 방법, 즉 일반화된 최적 우선 검색(generalized best-first search)과 경량 경로 합(Path-sum) 랭커를 통해 성능을 높이며, BERT, GPT-2 등 다양한 인코더 및 디코더 임베딩을 활용합니다. 실험의 결과는 Cobweb의 접근 방식이 기존의 밀집 검색 기준과 비슷하거나 더 나은 성능을 나타내며, 임베딩 품질 저하에도 강한 내성을 보임을 입증합니다.

- **Performance Highlights**: 모든 실험에서 Cobweb 기반의 계층 검색 방법은 MS MARCO와 QQP 데이터셋에서 효과적인 평가 결과를 도출했습니다. 특히, 문서 검색은 전통적인 단일 유사도 기반 방법인 dot-product 검색과 비교할 때 우수한 성능을 보여주며, 성능 스케일링 또한 뛰어난 것으로 나타났습니다. 이로 인해 Cobweb은 대규모 문서 검색의 해석 가능성을 제공하는 동시에 효과성과 확장성을 갖춘 솔루션으로 자리매김하게 되었습니다.



### Unraveling Syntax: How Language Models Learn Context-Free Grammars (https://arxiv.org/abs/2510.02524)
Comments:
          Equal contribution by LYS and DM

- **What's New**: 이 논문에서는 언어 모델(LLMs)이 구문(syntax)을 어떻게 습득하는지를 이해하기 위한 새로운 프레임워크를 제안합니다. 이전 연구들은 훈련된 모델의 정적 행동을 분석하는 데 중점을 두었지만, 이 연구는 언어 습득 과정에 대한 이해를 목표로 합니다. 저자들은 확률적 문맥 자유 문법(PCFGs)를 기반으로 한 합성 언어(synthetic languages)를 통해 작고 통제 가능한 모델에서의 학습 동역학을 연구합니다.

- **Technical Details**: 연구의 주요 기여는 언어 모델이 간단한 하위 구조(subgrammar)를 먼저 습득하고 더 복잡한 구문으로 나아가는 과정을 조사하는 것입니다. 저자는 여러 가지 일반적인 재귀 공식(general recursive formulae)을 증명하고, PCFG의 하위 문법 구조에 대한 훈련 손실(training loss)과 쿨백-라이블러 발산(KL divergence)을 조사합니다. 실험적으로 저자들은 트랜스포머가 모든 하위 문법에서 손실을 병렬로 줄이는 것을 발견했습니다.

- **Performance Highlights**: 결론적으로, 모델들은 깊은 재귀 구조에서 어려움을 겪으며, 이는 대형 언어 모델에도 있어 한계입니다. 저자들은 모델들이 긴 문맥에서 고정된 깊이에서는 잘 작동하지만 재귀 깊이가 증가할 때 급격히 실패한 것을 보여주었습니다. 또한, 저자들은 하위 문법의 사전 훈련이 작은 모델의 최종 손실을 개선할 수 있음을 입증하고, 사전 훈련된 모델이 하위 구문 구조와 더 정렬된 내부 표현을 개발하는 것으로 나타났습니다.



### CLARITY: Clinical Assistant for Routing, Inference, and Triag (https://arxiv.org/abs/2510.02463)
Comments:
          Accepted to EMNLP 2025 (Industrial Track)

- **What's New**: CLARITY는 환자를 전문의에게 신속하게 안내하고, 임상 상담을 지원하며, 환자 상태의 심각도를 평가하는 AI 기반 플랫폼입니다. Finite State Machine(FSM)과 Large Language Model(LLM)을 결합하여 증상을 분석하고 적절한 전문의에 대한 추천을 우선시하는 하이브리드 아키텍처를 가지고 있습니다. 최근에 대규모 국가 간 병원 IT 플랫폼에 통합되어, 두 달 동안 5만 5천 개 이상의 사용자 대화가 완료되어, 전문가 검증을 위한 2천 5백 개의 대화가 주목받았습니다.

- **Technical Details**: CLARITY 시스템은 대화 관리에 FSM을 이용하고 요청 처리를 위한 마이크로서비스로 구성된 하이브리드 아키텍처로 설계되었습니다. FSM은 사용자 입력과 맥락에 따라 대화의 상태와 전환을 관리하며, 자연어 생성 및 복잡한 질의 처리를 담당하는 텍스트 생성 서비스와 입력 텍스트 분류 및 의사 결정을 담당하는 결정 서비스가 포함되어 있습니다. 마이크로서비스 아키텍처는 모듈성과 확장성을 보장하며, 서비스 간 지연 시간을 최소화하고 최적화를 통해 실시간 성능을 제공합니다.

- **Performance Highlights**: CLARITY는 환자 상태를 자동으로 평가하고 진단 가설을 생성하여 환자 안내 및 전문의 추천을 최적화합니다. 실험 결과, CLARITY는 첫 시도에서의 라우팅 정확도 면에서 인간의 성능을 초과하며, 상담에서 소요되는 시간을 3배 이상 단축시킵니다. 이러한 시스템이 의료 현장에 안전하고 효과적으로 통합될 수 있도록 하는 다양한 기술적 해결책들이 필요하다는 점을 강조합니다.



### Words That Make Language Models Perceiv (https://arxiv.org/abs/2510.02425)
- **What's New**: 이번 연구에서는 언어 모델(LLMs)이 텍스트만 학습하더라도 시각 및 청각 인코더와의 표현 정합성을 높일 수 있음을 보여주었습니다. 연구진은 명시적인 감각 프롬프트가 LLM의 잠재적 구조를 표출시킬 수 있다는 가설을 검증하였습니다. 예를 들어, 모델이 '보다(see)' 또는 '듣다(hear)'라고 지시받을 때, 이러한 프롬프트가 시각적 또는 청각적 증거에 기초하여 다음 단어 예측에 영향을 미친다는 것을 발견하였습니다.

- **Technical Details**: 연구에서는 텍스트만으로 학습된 LLM의 표현 기하학을 검사하여 이를 단일 모드 비전 및 오디오 인코더와 유사하게 변화시킬 수 있는 방법을 다루었습니다. 새로운 개념으로 생성적 표현을 도입하여, LLM이 생성하는 각 출력은 정해진 프롬프트와 지금까지 생성된 시퀀스의 함수로서, 추가적인 전진 패스를 포함합니다. 감각 프롬프트를 통해 이러한 생성적 표현을 제어할 수 있으며, 이는 더 높은 정합성으로 이어집니다.

- **Performance Highlights**: 연구 결과에 따르면, 단일 감각 단어가 포함된 프롬프트는 텍스트만 학습된 LLM의 커널을 감각 인코더의 기하학에 더 가깝게 이동시킬 수 있음을 보여줍니다. 생성의 길이가 길어질수록 표현의 유사성이 증가하며, 더 큰 모델일수록 감각 프롬프트에 대한 정합성이 높아지는 경향이 있습니다. 또한, 시각적 단서가 있을 경우 LLM이 VQA(Visual Question Answering)에서 더 나은 성능을 발휘하는 것으로 나타났습니다.



### Retrieval and Augmentation of Domain Knowledge for Text-to-SQL Semantic Parsing (https://arxiv.org/abs/2510.02394)
Comments:
          10 pages, 2 figures, 11 tables. Accepted in the 1st Workshop on Grounding Documents with Reasoning, Agents, Retrieval, and Attribution (RARA) held in conjunction with IEEE International Conference on Data Mining (ICDM) 2025

- **What's New**: 본 논문에서는 데이터베이스(DB)에서의 구조화된 도메인 진술(DSs)을 체계적으로 연결할 수 있는 프레임워크를 제안합니다. 기존의 비현실적인 텍스트 힌트를 대신하여, 사용자 쿼리에 적합한 DS를 서브 문자열 수준에서 검색하는 방법을 개발하였습니다. 이를 통해 다양한 DB와 LLM에서 보다 정확한 SQL 변환을 가능하게 합니다.

- **Technical Details**: 우리는 엔터프라이즈 DB의 라이프사이클 동안 도메인 전문가(예: 은행가, 의사)로부터 DS를 수집하며, 이 DS는 NL 형식으로 구술됩니다. 해당 DS는 SQL 코드 스니펫에 매핑된 NL 표현으로 구성된 구조화된 형식을 정의하고, LLM 기반의 컨텍스트 상에서 검색 및 SQL 생성에 최적화된 형식을 제공합니다.

- **Performance Highlights**: 실험을 통해 우리 접근 방식이 기존의 쿼리 특정 DS를 사용하는 방법보다 높은 정확성을 달성함을 확인하였습니다. 11개의 현실적인 DB 스키마에 걸쳐 수행된 평가에서, DB 수준의 구조화된 도메인 statements가 기존의 비현실적인 텍스트 도메인 statements보다 더 실용적이고 정확함을 보여주었습니다. 우리 접근 방식은 LLMs 간의 체계적인 검색 메커니즘의 중요성을 강조합니다.



### KnowledgeSmith: Uncovering Knowledge Updating in LLMs with Model Editing and Unlearning (https://arxiv.org/abs/2510.02392)
Comments:
          Technical report

- **What's New**: 이번 연구에서는 KnowledgeSmith라는 통합 프레임워크를 제안하여 대형 언어 모델(LLMs)의 지식 업데이트 메커니즘을 체계적으로 이해할 수 있도록 합니다. 연구자들은 편집(editing)과 기계 망각(unlearning)을 하나의 제약 최적화 문제로 재구성하고, 여러 그래프 수준과 데이터 스케일에서 구조화된 개입을 제공하는 자동 데이터셋 생성기를 개발하여 LLM 지식을 수정하는 다양한 전략이 모델 지식에 어떻게 전파되는지를 연구합니다.

- **Technical Details**: KnowledgeSmith는 기존 지식 그래프(Knowledge Graphs)를 활용하여 LLM의 동작을 개선할 수 있는 방법론을 제시합니다. 이 프레임워크는 데이터를 수동으로 설계할 필요 없이 다양한 수준의 개입을 통해 평가를 수행할 수 있도록 하며, LLM의 지식 전달, 확장 가능성, 표현 변화 및 내구성을 실험했습니다.

- **Performance Highlights**: 연구 결과, LLM의 지식 업데이트 방식이 인간과는 다르며, FoU(First of Update)와 FoD(First of Deletion)의 비대칭성이 발견되었습니다. 또한, 일관성(capacity)과 용량(consistency) 간의 트레이드오프가 존재하고, 특정 도메인에서는 시점 업데이트가 더 어려움을 겪는다는 것을 보여주었습니다. 이러한 통찰은 향후 안정적이고 확장 가능한 방법 설계에 기여할 수 있을 것으로 기대됩니다.



### Learning to Route: A Rule-Driven Agent Framework for Hybrid-Source Retrieval-Augmented Generation (https://arxiv.org/abs/2510.02388)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLM)의 한계를 극복하기 위해 구조화된 관계형 데이터베이스(relational databases)를 활용하는 새로운 접근 방식을 제안합니다. 기존의 Retrieval-Augmented Generation (RAG) 시스템은 주로 비구조화된 문서에 의존하고 있어, 정확하고 최신의 정보를 요구하는 도메인 특정 상황에서 부족함을 나타냅니다. 이 연구는 데이터베이스와 문서가 쿼리에 따라 상호 보완적인 강점을 제공함을 밝혀내고, 이를 통해 효율성과 효과성의 균형을 맞추는 것이 중요한 요소임을 강조합니다.

- **Technical Details**: 저자들은 RAG 프레임워크 내에서 데이터베이스와 문서의 조합이 효과적이지 않음을 제시하며, 각각의 쿼리에 대해 가장 적합한 출처를 선택하는 라우팅 메커니즘을 제안합니다. 이 라우팅 메커니즘은 명시적인 규칙을 기반으로 후보 경로를 평가하여 가장 적합한 경로를 선택합니다. 또한, QA 피드백을 활용하여 규칙을 개선하고, 유사한 쿼리에 대해 과거의 라우팅 결정을 재사용하는 경로 수준 메타 캐시(path-level meta-cache)를 도입하여 지연 시간과 비용을 줄입니다.

- **Performance Highlights**: 제안된 프레임워크는 세 가지 QA 벤치마크에서 실험을 통해 기존의 정적 전략과 학습된 라우팅의 기준선보다 일관되게 높은 성능을 나타내었습니다. 이 연구는 정밀성과 효율성을 동시에 유지하면서 비용을 절감하는 데 성공했습니다. 따라서, 구조화된 정보의 활용과 문서 검색의 통합을 보다 효율적으로 구현합니다.



### Uncertainty-Aware Answer Selection for Improved Reasoning in Multi-LLM Systems (https://arxiv.org/abs/2510.02377)
- **What's New**: 이번 연구에서는 다수의 LLM에서 가장 신뢰할 수 있는 응답을 선택하는 새로운 방법을 제안하고 있습니다. 저자들은 기존 방법들이 자주 사용되는 외부 검증 도구나 인간 평가자에 의존하는 문제를 해결하고자 하였으며, 자원 제약이 있는 상황에서도 효과적으로 작동하는 기법을 소개하고 있습니다. 새로운 접근 방식은 조정된 로그-우도(log-likelihood) 점수를 사용하여, 다양한 LLM의 고유한 지식과 신뢰성을 활용하는 점이 특징입니다.

- **Technical Details**: 저자들은 다중 LLM 시스템에서 응답을 집계하고 최적의 답변을 선택하기 위해 로그-우도 기반의 점수를 사용합니다. 이들은 각 모델의 파라미터와 로지트 분포가 다르기 때문에, 직접적인 비교가 이론적으로 부적절하다는 점을 강조하며, 고유한 신뢰성 점수를 조정하여 활용합니다. 이 방법은 전방 패스로 조정된 점수를 계산함으로써 비싼 자율 회귀 디코딩을 피하고, 외부 검증자나 별도의 보상 모델이 필요하지 않습니다.

- **Performance Highlights**: 제안한 방법은 GSM8K, MMLU(6개 하위 집합), ARC 데이터셋에서 각각 약 4%, 3%, 5%의 성능 향상을 보여주었습니다. 다중-LLM 설정에서 단일 LLM보다 성능 개선이 더 두드러졌으며, 다양한 응답을 생성하는 다중-LLM 시스템의 잠재력을 더욱 드러냈습니다. 연구 결과는 이 새로운 접근 방식이 응답 품질 향상에 기여할 수 있음을 강하게 시사합니다.



### Pretraining with hierarchical memories: separating long-tail and common knowledg (https://arxiv.org/abs/2510.02375)
- **What's New**: 이 연구는 현대 언어 모델의 성능 향상에 대한 새로운 접근법을 제시합니다. 기존의 방법이 매개변수(parameter)의 규모 확대에 의존하고 있는 반면, 우리는 작은 언어 모델이 큰 계층적(parametric) 메모리 뱅크를 활용하여 폭넓은 세계 지식을 접근할 수 있도록 하는 메모리 증강 아키텍처를 개발하였습니다. 이를 통해 필요한 부분만을 메모리에서 가져와 모델에 추가합니다.

- **Technical Details**: 제안된 메모리 증강 아키텍처는 문맥에 따라 필요한 지식 매개변수를 모델에 연결하여, 실행시간 효율성을 높입니다. 또한, 훈련 시 메모리 파라미터가 유사한 주제의 시퀀스에서만 활성화되고 업데이트됨으로써, 파라미터의 망각(catastrophic forgetting)을 줄이고 보다 효과적으로 장기 지식을 기억할 수 있습니다. 이러한 혁신적인 접근방식은 훈련 효율성을 높이고, 분산 훈련을 더욱 간편하게 합니다.

- **Performance Highlights**: 우리는 실험을 통해 제안된 모델이 기본 모델보다 성능 향상을 보여주며, 매개변수 수가 2배 이상인 일반 모델과 유사한 성능을 낸다는 것을 입증했습니다. 예를 들어, 4.6B 메모리 뱅크에서 가져온 18M 매개변수의 메모리가 보강된 160M 모델이 정규 모델과 유사한 성능을 발휘합니다. 이러한 성능 개선은 메모리 유형, 깊이, 크기 그리고 메모리와 모델의 비율 등에 따라 달라지며, 우리는 각 요소가 성능에 미치는 영향을 체계적으로 분석하였습니다.



### Training Dynamics of Parametric and In-Context Knowledge Utilization in Language Models (https://arxiv.org/abs/2510.02370)
Comments:
          16 pages

- **What's New**: 이 논문에서는 대형 언어 모델들이 훈련 중에 지도 학습(knowledge arbitration) 방식을 어떻게 형성하는지에 대한 체계적인 이해 부족 문제를 다룹니다. 다양한 훈련 조건이 모델들이 문맥 내(in-context) 지식과 매개변수(parametric) 지식之间의 경합(competition)에 미치는 영향을 연구했습니다. 이 연구는 사전 훈련 이후의 계산 자원의 낭비를 예방하는 데 중요한 역할을 할 수 있습니다.

- **Technical Details**: 연구자들은 변환기(transformer) 기반 언어 모델을 합성 전기(biographies corpus) 데이터셋을 사용하여 훈련하고, 다양한 조건을 체계적으로 조절하여 실험을 진행했습니다. 결과적으로, 문서 내부( intra-document )의 사실 반복(repetition)이 매개변수 지식과 문맥 내 지식의 개발을 촉진함을 발견했습니다. 또한, 일관되지 않은 정보(inconsistent information)나 분포 왜곡(distributional skew)을 포함하는 데이터셋에서 훈련하는 것이 모델들이 강력한 전략을 개발하도록 유도함을 확인했습니다.

- **Performance Highlights**: 훈련 조건 개선으로 비이상적인 속성(non-ideal properties)을 제거하는 대신에, 이러한 속성이 강력한 지도 학습을 위한 중요성에 대한 증거를 제공합니다. 이 연구는 매개변수 지식과 문맥 지식의 통합(integration)에 있어 실질적이고 경험적인 지침을 제공하며, 이를 통해 더 효과적인 사전 훈련(pretraining) 모델 설계가 가능함을 보여줍니다.



### Beyond Manuals and Tasks: Instance-Level Context Learning for LLM Agents (https://arxiv.org/abs/2510.02369)
Comments:
          Under review at ICLR 2026

- **What's New**: 이 논문에서는 기존의 Large Language Model (LLM) 에이전트가 수신하는 세 가지 컨텍스트 중 인스턴스 수준의 컨텍스트(Instance-Level Context)라는 중요하지만 간과된 제 3의 유형을 제시합니다. 이는 특정 환경 인스턴스에 연결된 검증 가능하고 재사용 가능한 사실들로 구성되어 있습니다. 이러한 인스턴스 수준의 컨텍스트가 결여될 경우 LLM 에이전트가 복잡한 작업을 수행하는 데 실패하는 일반적인 원인이라는 주장을 하고 있습니다.

- **Technical Details**: 저자들은 '인스턴스 수준 컨텍스트 학습(Instance-Level Context Learning, ILCL)'이라는 문제를 공식화하고, 이를 해결하기 위한 작업 비 특이적(task-agnostic) 방법인 AutoContext를 소개합니다. 이 방법은 TODO 숲(TODO forest)을 사용하여 다음 작업을 우선 지정하고, 경량의 계획-행동-추출(Plan-Act-Extract) 루프를 통해 고정밀 컨텍스트 문서를 자동으로 생성합니다. AutoContext의 구조는 탐색을 조직화하고 시스템적으로 지식 격차를 노출하는 혁신적인 방법을 포함하고 있습니다.

- **Performance Highlights**: 문서화된 결과는 TextWorld, ALFWorld, Crafter에서의 실험을 통해 입증되었으며, ReAct 에이전트의 TextWorld 성공률이 37%에서 95%로 향상되었고, IGE는 81%에서 95%로 개선되었습니다. 이 방법은 일회성 탐색을 지속 가능하고 재사용 가능한 지식으로 변환함으로써 기존의 컨텍스트를 보완하여 더욱 신뢰할 수 있는 LLM 에이전트를 가능하게 합니다. 따라서, AutoContext는 여러 작업에 걸쳐 다운스트림 에이전트의 성능을 크게 개선합니다.



### A Cross-Lingual Analysis of Bias in Large Language Models Using Romanian History (https://arxiv.org/abs/2510.02362)
Comments:
          10 pages

- **What's New**: 이번 연구는 루마니아의 역사적 논란에 대한 질문을 여러 개의 대형 언어 모델(LLMs)에 제시하여 그 모델들의 편향성을 분석하는 내용을 담고 있습니다. 이러한 접근은 교육적 목적뿐만 아니라, 역사라는 주제가 문화와 국가의 이상에 의해 어떻게 왜곡될 수 있는지를 인식하는 데 그 의의가 있습니다. 연구 결과, 다양한 언어와 문맥에서 LLM의 답변이 어떻게 변화하는지를 발견했습니다.

- **Technical Details**: 연구 방법론은 세 가지 주요 단계로 나뉘어 있으며, 이는 특정 분야의 논란이 되는 역사적 사건에 대한 편향 분석을 보장하기 위해 고안되었습니다. 첫 단계에서는 루마니아어를 기본 언어로 선정하고, 영어, 헝가리어, 러시아어를 포함하여 서로 다른 문화적 관점에서 편향을 탐구했습니다. 레벨에 따라 단순한 긍정 혹은 부정 응답, 1-10의 척도를 통한 수치 답변, 구조화된 에세이 작성으로 분석을 진행했습니다.

- **Performance Highlights**: 실험 결과는 언어 모델들이 특정 질문에 대해 일관된 대답을 제공하는 경향이 있지만, 각기 다른 언어 및 형식에 따라 의견이 달라질 수 있음을 보여주었습니다. 특히 이진 답변의 안정성은 상대적으로 높지만 완벽하지 않았고, 수치적 평가에서는 초기의 이진 선택과 일치하지 않는 경우가 많았습니다. 이번 연구는 LLMs가 정보의 제공 방식에 따라 어떻게 반응이 달라질 수 있는지를 명확히 하였습니다.



### ChunkLLM: A Lightweight Pluggable Framework for Accelerating LLMs Inferenc (https://arxiv.org/abs/2510.02361)
- **What's New**: ChunkLLM은 경량화된 모듈인 QK Adapter와 Chunk Adapter를 도입하여 기존의 Transformers에 통합할 수 있는 새로운 훈련 프레임워크를 제안합니다. 이 구조는 각 Transformer 레이어에 연결되어 기능 압축과 청크 주의 획득을 동시에 수행하여 계산 비용을 크게 절감할 수 있습니다. 이러한 접근 방식은 의미의 완전성과 훈련-추론 효율성을 동시에 해결하는 데 중점을 둡니다.

- **Technical Details**: Chunk Adapter는 청크 경계를 식별하기 위해 맥락적 의미 정보를 활용하며, 1층 포워드 신경망 분류기로 구현됩니다. QK Adapter는 각 Transformer 레이어에 병렬로 위치하여, 주의 행렬을 청크 주의 점수로 매핑합니다. 훈련 과정에서는 KL 발산을 통해 청크 주의 점수를 최적화하여, 주요 청크의 회수율을 향상시킵니다.

- **Performance Highlights**: ChunkLLM은 다양한 장기 및 단기 텍스트 벤치마크 데이터 세트에서 실험 평가를 통해 120,000 토큰 처리에서 바닐라 Transformer 대비 4.48배의 속도 향상을 달성했습니다. 특히 단기 텍스트 벤치마크에서 유사한 성능을 유지하며, 장기 맥락 벤치마크에서도 98.64%의 성능을 보유하고 있습니다. 이는 기존의 Transformer 모델들에 비해 더 효율적인 컴퓨팅 자원 관리와 성능 최적화를 가능하게 합니다.



### Spiral of Silence in Large Language Model Agents (https://arxiv.org/abs/2510.02360)
- **What's New**: 이 논문은 Spiral of Silence (SoS) 이론을 인공지능 언어 모델(LLM)에 적용하여 소수 의견의 억제 현상이 LLM 집단에서도 발생할 수 있는지 검토합니다. 저자들은 History와 Persona 신호를 도입하여 SoS 역학을 탐색하고, 이 신호들이 소수 집단의 의견이 사라지는 과정을 어떻게 형성하는지 평가하는 체계적인 프레임워크를 제안합니다.

- **Technical Details**: 연구에서는 LLM 에이전트가 영화에 대한 평가를 순차적으로 수행하는 다중 에이전트 환경을 구축하였습니다. 두 가지 주요 신호인 History(이전 에이전트의 평균 평점)와 Persona(각 에이전트에 할당된 고유한 역할)를 활용하여 의견 역학을 측정합니다. 이를 통해 SoS 역학의 발생 여부을 확인하는 다양한 통계 기법과 지표를 사용하여 실험 결과를 평가합니다.

- **Performance Highlights**: 실험 결과, History와 Persona 신호 모두가 존재할 때 SoS 패턴이 가장 뚜렷하게 나타났으며, History 신호만으로도 강한 고정 효과가 나타났습니다. Persona 신호는 다양한 의견을 유도하지만 이들 간의 상관관계는 없음을 보여주었고, 이러한 발견은 LLM 시스템의 설계와 규제에 대한 중대한 시사점을 제공한다는 점에서 의미가 있습니다.



### Emission-GPT: A domain-specific language model agent for knowledge retrieval, emission inventory and data analysis (https://arxiv.org/abs/2510.02359)
- **What's New**: 이번 논문에서는 대기 오염물질 및 온실가스 배출에 대한 보다 정확한 이해와 분석을 제공하기 위해 Emission-GPT라는 지식 강화 대형 언어 모델 에이전트를 소개합니다. Emission-GPT는 10,000개 이상의 문서로 구성된 맞춤형 지식 기반 위에 구축되었으며, 도메인 특화된 질문 답변을 지원하는 데 필요한 기능을 통합하고 있습니다. 이를 통해 사용자들은 자연어를 이용하여 배출 데이터를 대화형으로 분석하고 시각화할 수 있는 새로운 가능성을 열었습니다.

- **Technical Details**: Emission-GPT 시스템은 대기배출 분야에서 지능형 상호작용 및 분석을 가능하게 하는 모듈화 및 다단계 워크플로우를 제안합니다. 이 시스템은 사용자 쿼리를 기초로 질문을 두 가지 카테고리로 분류하며, 각 카테고리에 맞춰 적절한 전문가 LLM을 호출하여 응답 및 데이터 분석을 수행합니다. 이렇게 구성된 아키텍처는 각기 다른 배출 관련 작업을 위한 맞춤형 AI 에이전트를 지원할 수 있도록 설계되었습니다.

- **Performance Highlights**: Emission-GPT의 성능을 평가하는 사례 연구에서는 간단한 프롬프트를 통해 원시 데이터에서 중요한 인사이트를 직접 추출할 수 있음을 보여주었습니다. 이 시스템은 배출 재고 조사의 효율성을 크게 높이며, 사용자 정의 시나리오에 대한 배출 인자 추천 기능 등을 통해 대기배출 정보의 접근성 및 활용도를 크게 향상시킵니다. 비전문가도 쉽게 사용할 수 형식으로 구성된 Emission-GPT는 향후 배출 재고 개발 및 시나리오 기반 평가의 기초 도구로 자리잡을 것으로 기대됩니다.



### DiffuSpec: Unlocking Diffusion Language Models for Speculative Decoding (https://arxiv.org/abs/2510.02358)
- **What's New**: 대형 언어 모델(LLMs)이 성장함에 따라 정확도는 향상되지만, 自回归(autoregressive, AR) 디코딩의 특성 때문에 지연(latency)의 병목이 발생합니다. 이 문제를 해결하기 위해 DiffuSpec라는 새로운 프레임워크를 소개합니다. DiffuSpec는 사전 훈련된 확산 언어 모델(diffusion language model, DLM)을 사용하여 단일 전방 패스를 통해 여러 토큰 초안을 생성합니다.

- **Technical Details**: DiffuSpec는 두 가지 주요 구성요소로 구성됩니다. 첫째, 인과 일관성 경로 검색(causal-consistency path search, CPS)을 통해 AR 검증에 맞는 왼쪽에서 오른쪽 경로를 선택하여 수용을 극대화합니다. 둘째, 적응형 초안 길이(adaptive draft-length, ADL) 컨트롤러를 통해 최근 수용 피드백을 기반으로 다음 초안 길이를 조정합니다. 이 프레임워크는 별도의 훈련 없이 기존 AR 검증기와 통합할 수 있습니다.

- **Performance Highlights**: DiffuSpec는 다양한 생성 작업에서 최대 3배의 벽시계 속도를 향상시켜 훈련 기반 방법에 근접하는 성능을 보여줍니다. 또한, DiffuSpec는 기존 훈련 프리 베이스라인을 초 outperforming하며, 확산 기반 초안 작성을 AR 드래프트의 강력한 대안으로 자리매김하고 있습니다.



### Modeling the language cortex with form-independent and enriched representations of sentence meaning reveals remarkable semantic abstractness (https://arxiv.org/abs/2510.02354)
- **What's New**: 이 논문은 인간의 언어 피질에서 의미의 추상적 표현을 찾기 위한 연구로, 문장에 대한 신경 반응을 시각 및 언어 모델의 표현을 사용하여 모델링합니다. 특히, 여러 이미지에서 추출한 vision model embeddings를 활용하여 언어 피질의 반응을 보다 정확히 예측할 수 있음을 발견하였습니다. 또한, 문장의 패러프레이즈(paraphrase)를 균형있게 활용하는 것이 예측 정확도를 높이는 데 기여함을 보여줍니다.

- **Technical Details**: 연구에서는 Inferior Frontal Gyrus, Middle Frontal Gyrus, Anterior Temporal cortex 등 'core' 언어 네트워크에서 fMRI 스캐닝을 통해 수집된 데이터셋을 분석하였습니다. 주어진 데이터셋에서 original sentences의 의미를 다른 문장이나 비전 모델의 표현으로도 포착할 수 있는지를 평가하였습니다. 또한, ridge regression 모델을 통해 원래 문장에 기반한 예측 정확도를 데이터의 맥락 정보를 통합하여 개선하는 방법을 사용했습니다.

- **Performance Highlights**: 연구 결과, 문장의 다양한 비주얼 디스펙션(visual depiction)을 이용한 embeddings가 언어 피질의 활동 예측에서 비약적인 예측 능력을 보여주었습니다. 또한, 패러프레이즈를 활용하여 예측 정확도가 향상되는 것을 발견했으며, 상식적(Contextual) 정보로 풍부하게 만든 문장이 예측력을 더욱 크게 증가시킨다는 점이 강조됐습니다. 이러한 결과는 언어 시스템이 단순한 언어 모델 이상으로 풍부하고 넓은 의미 표현을 유지하고 있다는 것을 시사합니다.



### An Senegalese Legal Texts Structuration Using LLM-augmented Knowledge Graph (https://arxiv.org/abs/2510.02353)
Comments:
          8 pages, 8 figures, 2 tables, 1 algorithm

- **What's New**: 이 연구는 세네갈 사법 시스템에서 법률 문서에 대한 접근을 향상시키기 위해 인공지능(AI)과 대형 언어 모델(LLM)의 응용을 다루고 있습니다. 법적 문서를 추출하고 조직하는 데 어려움을 보완할 필요성이 강조됩니다. 연구진은 특히 토지 및 공공 도메인 법전에서 7,967개의 기사를 성공적으로 추출하였으며, 이를 위한 상세한 그래프 데이터베이스를 개발했습니다.

- **Technical Details**: 이 연구에서는 2,872개의 노드와 10,774개의 관계를 포함하는 그래프 데이터베이스를 구축하였습니다. 또한, GPT-4o, GPT-4, Mistral-Large와 같은 모델을 사용해 관계 및 관련 메타데이터를 식별하는 고급 triple extraction 기법을 적용했습니다. 이 과정에서 LLM-augmented Knowledge Graph 원칙을 따르며, 핵심 참고 해소(coreference resolution), 명명된 개체 인식(named entity recognition), 개체 관계 식별 등을 통해 지식 그래프를 생성했습니다.

- **Performance Highlights**: 이 연구의 목표는 세네갈 국민과 법률 전문가가 자신들의 권리와 의무를 보다 효과적으로 이해할 수 있는 포괄적인 프레임워크를 개발하는 것입니다. 개발된 알고리즘과 그래프 데이터베이스는 법률 문서의 명확한 정보 제공을 가능하게 하며, 복잡한 법적 문서를 탐색하는 데 큰 도움이 될 것입니다. 최종적으로 이 시스템은 세네갈 법률 체계의 접근성과 효율성을 significantly 향상시킬 것으로 기대됩니다.



### Evaluating Bias in Spoken Dialogue LLMs for Real-World Decisions and Recommendations (https://arxiv.org/abs/2510.02352)
- **What's New**: 이번 논문은 음성 대화 모델(SDMs)에서의 편향(bias)에 대한 체계적인 평가를 제공하며, 다중 턴 대화(multi-turn dialogues)가 모델 출력에서 어떻게 편향을 증폭할 수 있는지를 연구했습니다. 이는 개방형 모델(Qwen2.5-Omni, GLM-4-Voice)과 폐쇄형 API(GPT-4o Audio, Gemini-2.5-Flash)를 포함하여 처음으로 발표된 연구로, 음성 기반의 대화형 시스템에서 공정성과 신뢰성을 제고하는 데 기여할 것 입니다.

- **Technical Details**: 다중 턴 대화에 따른 편향 검증을 위해 그룹 불공정성 점수(Group Unfairness Score: GUS)와 유사도 기반 정규화 통계 비율(Similarity-Based Normalized Statistics Rate: SNSR) 메트릭을 사용했습니다. 연구에서 생성된 데이터셋은 결정을 내리는 작업 및 추천 작업과 같은 두 가지 주요 실제 시나리오에 초점을 맞추며, 이는 모델의 출력이 사회적 기회를 어떻게 직접적으로 영향을 미칠 수 있는지를 보여줍니다. 이 연구에서는 또한 한국어의 음성과 텍스트 생성 플랫폼을 활용하여 통제된 음성을 합성했습니다.

- **Performance Highlights**: 결과적으로 폐쇄형 모델이 일반적으로 낮은 편향을 보이는 반면, 개방형 모델은 나이와 성별에 대해 더 민감하게 반응함을 발견했습니다. 특히, 추천 작업에서는 집단 간 격차가 확대되는 경향이 있습니다. 다중 턴 대화에서도 편향된 결정이 지속될 수 있으며, 일부 집단은 공정한 결과를 달성하기 위해 더 많은 교정 피드백이 필요하다는 점을 확인했습니다.



### Language, Culture, and Ideology: Personalizing Offensiveness Detection in Political Tweets with Reasoning LLMs (https://arxiv.org/abs/2510.02351)
Comments:
          To appear in the Proceedings of the IEEE International Conference on Data Mining Workshops (ICDMW)

- **What's New**: 이 연구에서는 대형 언어 모델(LLMs)이 정치적 담론에서 공격성을 평가하는 방법을 탐구합니다. 특히, 2020년 미국 대선 트윗을 중심으로 다국어 MD-Agreement 데이터셋을 활용하여 다양한 정치적 관점을 가진 모델들이 트윗을 공격적 혹은 비공격적으로 판단하는 능력을 평가했습니다. 연구에서는 DeepSeek-R1, o4-mini, GPT-4.1-mini, Qwen3 등 여러 최신 모델들을 비교했습니다.

- **Technical Details**: 이 연구는 이념적이고 문화적인 맥락에서 공격성 탐지의 개인화 여부를 평가하는 새로운 프레임워크를 제안합니다. 연구자들은 크게 두 가지 변수 – 모델 크기(소형 vs 대형)와 추론 능력(가능 vs 불가능)을 기준으로 모델을 카테고리화하여 평가하였습니다. 추론 능력을 갖춘 대형 모델들이 이념적 관점을 더 잘 모방할 수 있음을 발견하였습니다.

- **Performance Highlights**: 실험 결과, DeepSeek-R1과 o4-mini와 같이 추론 기능이 있는 모델이 개인화된 공격성 분류에서 뛰어난 성능을 보였습니다. 원래 언어에서 폴란드어 및 러시아어로 번역된 트윗을 사용하였고, 각 모델의 판단이 정치적 배경에 따라 달라질 수 있음을 확인했습니다. 이는 LLM을 더 정교한 사회 정치적 텍스트 분류에 적합하도록 조정하기 위한 중요한 메커니즘이 될 수 있습니다.



### LLMSQL: Upgrading WikiSQL for the LLM Era of Text-to-SQL (https://arxiv.org/abs/2510.02350)
Comments:
          To appear in the Proceedings of the IEEE International Conference on Data Mining Workshops (ICDMW)

- **What's New**: 이 논문에서는 자연어 질문을 SQL 쿼리로 변환하는 새로운 데이터셋인 LLMSQL을 제안합니다. LLMSQL은 기존의 WikiSQL 데이터셋을 체계적으로 개정하고 변형하여 LLM(대형 언어 모델) 시대에 맞게 발전시킨 것입니다. 이 데이터셋은 고르지 못한 주의사항 및 구조적 문제를 해결하여, 안정적인 쿼리 실행과 올바른 결과를 보장합니다.

- **Technical Details**: LLMSQL은 자연어 질문과 SQL 쿼리쌍을 정제하는 자동화된 방법론을 통해 생성되었습니다. WikiSQL에서 발생하는 데이터 형식 불일치, 대소문자 민감도 문제, 질문 없음 등의 오류를 클래스화하였습니다. 다양한 대형 언어 모델(Gemma 3, LLaMA 3.2, Mistral 7B 등)을 평가하여 LLMSQL의 성능을 검증하였습니다.

- **Performance Highlights**: LLMSQL은 단일 테이블에 기반한 대규모 벤치마크로서, 기존 WikiSQL의 문제점을 해결하며 구조화된 쿼리 생성 작업을 위한 실용성을 높였습니다. 실험을 통해 기존 WikiSQL 분할 및 LLMSQL 벤치마크에서의 여러 모델 성능을 비교하였고, LLMSQL이 대형 언어 모델(LLM)의 평가에 적합하다는 것을 입증하였습니다.



### mini-vec2vec: Scaling Universal Geometry Alignment with Linear Transformations (https://arxiv.org/abs/2510.02348)
- **What's New**: 본 논문은 vec2vec를 기반으로 하여, 교차 데이터 없이 텍스트 임베딩 공간을 정렬하도록 설계된 프로세스입니다. 특정한 매칭이 없어도 높은 안정성과 효율성을 자랑하는 mini-vec2vec를 소개하며, 기존 방법보다 연산 비용이 훨씬 낮은 선형 변환을 학습합니다. 이를 통해 새로운 분야에서의 활용 가능성을 제시합니다.

- **Technical Details**: mini-vec2vec는 세 가지 주요 단계로 구성됩니다: 가상 병렬 임베딩 벡터 간의 일치 시도, 변환 적합성, 그리고 반복적인 정제입니다. 상대적 표현 프레임워크를 활용하고, 간단한 아핀 변환을 통해 구조적 유사성을 검색하여 정렬을 가능하게 합니다. 이 방법은 CPU 하드웨어에서 실행될 수 있도록 설계되어 자원 소모를 최소화합니다.

- **Performance Highlights**: 논문에서는 comprehensive experiments를 통해 mini-vec2vec가 기존의 adversarial 방법인 vec2vec의 성능을 능가한다고 보고합니다. 특히, 연산 자원 소모가 대폭 줄어들고, 훈련의 안정성 및 해석 가능성을 제공하여 다양한 데이터 분포에서도 잘 작동함을 강조합니다. mini-vec2vec는 수 동록 훈련 데이터에서 효과적으로 작동하고, 1:1 매칭이 불가능한 경우에도 견고한 결과를 보여줍니다.



### Small Language Models for Curriculum-based Guidanc (https://arxiv.org/abs/2510.02347)
- **What's New**: 이번 연구에서는 오픈소스 소형 언어 모델(SLM)을 활용한 AI 교육 보조 도우미의 개발 및 평가를 다룹니다. 연구에서는 8개 SLM 모델을 대상으로 교육과정에 기반한 가이드를 제공하는 Retrieval-Augmented Generation (RAG) 파이프라인을 적용하였습니다. 특히, SLM이 올바른 프롬프트(prompts)와 목표 지향적 검색을 통해 대규모 언어 모델(LLM)과 동등한 정확도를 보여준다는 점이 강조됩니다.

- **Technical Details**: 연구는 스칸디나비아의 대학에서 제공되는 대수학 관련 자료를 기반으로 하여 RAG 파이프라인을 통해 AI 교육 보조 도우미를 개발하였습니다. 이 시스템은 학생의 질문에 대해 적절한 교과 내용을 색인화하여 관련된 정보를 동적으로 검색하고 응답으로 통합하는 방식으로 운영됩니다. SLM은 각각 17억 개 이하의 매개변수를 사용하여 소비자 수준의 하드웨어에서도 실시간으로 사용 가능한 점이 특징입니다.

- **Performance Highlights**: SLM 기반의 AI 교육 도우미는 교육과정 일관성 유지, 오류 정보 감소, 비용 효율성 증대 등 여러 이점을 제공합니다. 연구에서는 SLM이 상용 LLM에 비해 보다 투명하고 정확한 교육 자료에 기반한 응답을 제공할 수 있다는 점을 입증하였습니다. 이를 통해 AI 교육 도우미가 교육기관에서 지속 가능한 개인화 학습을 구현하는 데 적합하다는 것을 강조합니다.



### Breaking the MoE LLM Trilemma: Dynamic Expert Clustering with Structured Compression (https://arxiv.org/abs/2510.02345)
Comments:
          12 pages, 2 figures, 3 tables. Under review as a conference paper at ICLR 2026

- **What's New**: 이번 연구는 Mixture-of-Experts (MoE) 대형 언어 모델에서 발생하는 부하 불균형, 매개변수 중복 및 통신 오버헤드를 동시에 해결하기 위한 통합 프레임워크를 소개합니다. 이 프레임워크는 긴밀하게 연결된 네 가지 혁신 요소를 통합하여 동적 전문가 클러스터링과 구조적 압축을 통해 MoE 모델의 효율성을 일관되게 개선합니다. 특히, 학습 중 모델 아키텍처를 동적으로 재구성할 수 있는 세맨틱(semantic) 임베딩 기능을 활용하여 효율적으로 전문가를 재편성합니다.

- **Technical Details**: 제안된 방법은 온라인 클러스터링 프로세스를 사용하여 매개변수와 활성화 유사도를 결합한 메트릭으로 전문가를 주기적으로 재조정합니다. 각 클러스터 내에서는 전문가 가중치를 공유 기본 행렬과 매우 낮은 순위의 잔여 적응기로 분해하여 매개변수를 최대 다섯 배 줄이는 동시에 전문성을 유지합니다. 이 구조는 두 단계 계층 라우팅 전략을 가능하게 하여 토큰이 먼저 클러스터에 할당된 후 구체적인 전문가에게 전달되도록 합니다.

- **Performance Highlights**: GLUE 및 WikiText-103에서 평가한 결과, 제안된 프레임워크는 표준 MoE 모델과 동등한 품질을 유지하면서 전체 매개변수를 약 80% 줄이고, 처리량을 10%에서 20% 증가시키며, 전문가 부하 분산을 세 배 이상 감소시켰습니다. 이러한 결과는 구조적 재편성이 확장 가능하고 효율적이며 메모리 효과적인 MoE LLM을 위한 핵심 길임을 입증합니다.



### $\texttt{BluePrint}$: A Social Media User Dataset for LLM Persona Evaluation and Training (https://arxiv.org/abs/2510.02343)
Comments:
          8 pages, 4 figures, 11 tables

- **What's New**: 이번 논문은 SIMPACT라는 프레임워크를 소개하여, 개인 정보를 보호하며 행동 기반 소셜 미디어 데이터셋을 구축하는 새로운 방법을 제시합니다. 이는 LLMs(대형 언어 모델)에 적합한 데이터 자원 부족 문제를 해결하고, 정확한 시뮬레이션을 통해 공공 담론을 탐구할 수 있게 합니다. 또한, BluePrint라는 대규모 데이터셋을 공개하여 정치적 담론을 진단하는 평가 벤치마크를 제공합니다.

- **Technical Details**: SIMPACT는 다음 행동 예측(next-action prediction)을 중심으로 구동되며, 사용자 행동을 다양한 행동 양식으로 추상화하여 클러스터링합니다. 이를 통해 개인의 신원을 노출하지 않으면서도 행동 풍부한 데이터셋을 생성할 수 있습니다. 또한, 사용자의 행동을 효과적으로 캡처하기 위해 PII(개인 식별 정보) 제거, 익명화 등의 기법을 통합하여 데이터의 무결성을 유지합니다.

- **Performance Highlights**: 연구에서는 LLMs에 대해 BluePrint 데이터셋을 사용하여 여러 모델(GPT-4.1 mini, GPT-o3 mini, Qwen 2.5)을 벤치마킹했습니다. 그 결과, 현재의 모델은 텍스트는 잘 생성하지만 실제 사용자 커뮤니티의 미묘한 행동 패턴을 재현하는 데 어려움을 겪고 있음을 발견했습니다. 이러한 결과는 SIMPACT와 같은 표준화된 데이터셋이 새로운 연구 분야에서 왜 중요한지를 강조합니다.



### DRIFT: Learning from Abundant User Dissatisfaction in Real-World Preference Learning (https://arxiv.org/abs/2510.02341)
- **What's New**: 이번 논문에서는 사용자 불만족 신호(DSAT: Dissatisfaction Signal)를 활용하여 LLM(대형 언어 모델)의 동적인 훈련 방법인 DRIFT(Dissatisfaction-Refined Iterative preFerence Training)를 소개합니다. DRIFT는 실제 데이터에서 추출된 불만족 신호를 활용해 모델의 성능을 개선하고, 기존의 사람의 주관적인 평가 방식에 의존하지 않습니다. 이를 통해 더 지속적이고 경제적인 데이터 활용이 가능하도록 지원합니다.

- **Technical Details**: DRIFT 접근법은 사용자 불만족 신호를 기반으로 하여, 현재의 정책과 함께 선택된 응답을 동적으로 샘플링합니다. 이는 기존의 DPO(Direct Preference Optimization)와 같은 정적인 방식을 대체하며, 훈련 과정에서 실시간으로 업데이트된 정책에 기반해 있으므로 성능이 더욱 향상됩니다. 이 방식은 불만족 신호가 풍부하게 존재하는 특성을 이용하여 모델 훈련의 효율성을 높이고 있습니다.

- **Performance Highlights**: DRIFT로 훈련된 모델은 WildBench 작업 점수에서 최대 +6.23%(7B) 및 +7.61%(14B) 상승하며, AlpacaEval2 우승률에서도 최대 +8.95%(7B) 및 +12.29%(14B) 상승하는 성과를 보입니다. 또한, 14B 모델은 DRIFT를 통해 GPT-4o-mini보다 더 우수한 성능을 기록하면서 강력한 기준선 방법들을 초월하는 결과를 나타내었습니다. DRIFT는 다양한 높은 보상 솔루션들을 생성하며, 기술적 한계를 극복하여 탐색의 폭을 넓히고 있습니다.



### Can Prompts Rewind Time for LLMs? Evaluating the Effectiveness of Prompted Knowledge Cutoffs (https://arxiv.org/abs/2510.02340)
- **What's New**: 이번 연구에서는 대형 언어 모델(Large Language Models, LLMs)이 실제로 이전의 지식 컷오프를 시뮬레이션 할 수 있는지를 조사합니다. 최근의 연구에서 프롬프트를 기반으로 한 언러닝(unlearning) 기법이 등장하였고, 이러한 기법을 통해 LLM의 지식 컷오프를 조정할 수 있는 가능성을 탐색합니다. 연구의 주요 초점은 LLM이 잊어야 할 정보에 대한 직접적인 질문이 없더라도 인과적으로 관련된 검색체를 잊는 여부입니다.

- **Technical Details**: 연구진은 사실(Factual), 의미(Semantic), 반사실(Counterfactual) 두 가지 차원에서 LLM의 지식 컷오프 유도 효과를 평가하기 위해 세 개의 데이터 세트를 구축했습니다. 각 데이터 세트는 서로 다른 675, 303 및 689개의 예를 포함합니다. 실험 결과, LLM이 사실적 지식과 의미 변화를 잊는 데 있어 각각 약 82.5%와 70.0%의 성공률을 기록했으나, 인과적 관계가 있는 이벤트를 잊는 경우 성공률이 약 19.2%로 낮음을 보였습니다.

- **Performance Highlights**: 이 연구는 프롬프트 기반의 지식 컷오프가 특정 차원에서 효과적일 수 있지만, 인과적으로 연결된 정보를 잊는 데 한계를 가진다는 것을 보여줍니다. 특히, LLM의 임시 컷오프 증명에서 발생할 수 있는 데이터 오염 문제를 해결하는 데 유용할 수 있습니다. 그러나 실제 세계의 시간적 예측 과제에서 LLM의 공정한 평가를 보장하기 위해서는 보다 강력한 방법이 필요하다는 결론을 내립니다.



### Evaluating Uncertainty Quantification Methods in Argumentative Large Language Models (https://arxiv.org/abs/2510.02339)
Comments:
          Accepted at EMNLP Findings 2025

- **What's New**: 최근 LLM(대규모 언어 모델)의 불확실성 정량화(UQ)에 대한 연구가 증가하고 있으며, 이는 신뢰할 수 있는 AI 시스템 개발에 필수적입니다. 본 논문에서는 논증적 LLM(ArgLLMs) 프레임워크에서 LLM UQ 방법의 통합을 탐구합니다. 이 프레임워크는 컴퓨터 논증에 기반하여 결정 내리기 과정을 설명 가능하게 하고, ArgLLM이 다양한 UQ 방법을 사용한 주장 검증(task)에서 성능을 평가하게 됩니다.

- **Technical Details**: 본 연구에서는 Semantic Entropy, Eccentricity, LUQ와 같은 세 가지 LLM UQ 방법을 사용합니다. 이러한 방법들은 MIT 라이센스의 LM-Polygraph 라이브러리에서 구현된 버전을 중심으로 실행됩니다. ArgLLM은 각 주장에 대해 지원 및 공격 논거를 생성하고, 이들의 신뢰도를 바탕으로 QBAFs(정량적 바이폴라 논증 체계)를 구성하여 주장의 진위를 판별합니다.

- **Performance Highlights**: 실험 결과, 단순히 직접 프롬팅(direct prompting)을 사용하는 것이 다른 복잡한 UQ 방법보다 유의미하게 더 나은 성능을 보였습니다. ArgLLM에서 생성된 주장의 신뢰도 점수는 최종 예측에 직접적으로 영향을 미치며, 다양한 설정에서 실험을 통해 이러한 결과를 입증하고 있습니다. 이러한 접근 방식은 LLM UQ 방법의 평가를 위한 새로운 방식을 제시합니다.



### Optimizing Long-Form Clinical Text Generation with Claim-Based Rewards (https://arxiv.org/abs/2510.02338)
- **What's New**: 이 연구는 임상 문서화를 자동화하기 위해 투자된 평가 통합 강화 학습 프레임워크를 소개합니다. 이 프레임워크는 Group Relative Policy Optimization (GRPO)와 DocLens라는 평가 도구를 결합하여 사실적 기반(factual grounding)과 문서의 완전성을 직접 최적화합니다. 별도의 보상 모델을 훈련할 필요 없이 문서의 질을 개선하며, 훈련 비용을 줄일 수 있는 간단한 보상 게이팅(strategy)을 활용합니다.

- **Technical Details**: 제안된 방법은 LLM(예: GPT-4o)을 평가자로 활용하여, 대화에서 추출한 임상 정보를 바탕으로 보상을 생성합니다. 이 과정을 통해 claim recall과 claim precision을 결합하여 보상을 제공하며, DocLens를 사용하여 논의에 대응하는 사실 정보를 정확히 추출합니다. 이 평가 통합 설계는 임상 요구 사항인 사실적 기반과 문서의 완전성을 최적화하면서도 컴퓨팅 효율성을 유지합니다.

- **Performance Highlights**: 연구 결과는 DocLens의 정밀도, 재현율(F1 점수)에서 일관된 개선을 보여주며, 보상 게이팅 전략을 통해 더 빠르게 수렴할 수 있음을 시사합니다. 이러한 개선은 이미 강력한 기본 모델의 성능을 보완하며, 더 도전적인 실제 작업에서도 추가적인 이점을 예상할 수 있습니다. 이 연구의 적용 가능성은 임상 지침 준수 및 청구 지원과 같은 비즈니스 지표와 최적화 목표에까지 확장될 수 있습니다.



### CRACQ: A Multi-Dimensional Approach To Automated Document Assessmen (https://arxiv.org/abs/2510.02337)
- **What's New**: 본 논문은 CRACQ라는 다차원 평가 프레임워크를 소개합니다. 이 프레임워크는 문서의 다섯 가지 특성인 Coherence(일관성), Rigor(엄밀성), Appropriateness(적절성), Completeness(완전성), Quality(품질)를 평가하도록 설계되었습니다. CRACQ는 자동화된 에세이 채점(Automated Essay Scoring)에서 얻은 통찰을 바탕으로 다른 형태의 기계 생성 텍스트로 그 초점을 확장합니다.

- **Technical Details**: CRACQ는 단일 점수 방식과는 달리 언어적, 의미적, 구조적 신호를 통합하여 누적 평가를 수행합니다. 이를 통해 전체적인 분석(holistic analysis)뿐만 아니라 각 특성 수준(trait-level analysis)의 분석이 가능합니다. 500개의 합성 보조금 제안서(synthetic grant proposals)로 훈련된 CRACQ는 LLM-as-a-judge와 비교 평가되었으며, 실제 강력한 및 약한 응용 프로그램에서도 추가 테스트가 진행되었습니다.

- **Performance Highlights**: 초기 결과에 따르면 CRACQ는 직접적인 LLM 평가보다 더 안정적이고 해석 가능한 특성 수준의 판단을 생성하는 것으로 나타났습니다. 그러나 신뢰성(reliability) 및 도메인 범위(domain scope)에서의 문제점은 여전히 남아 있습니다. 이러한 발견은 CRACQ가 머신러닝 기반의 텍스트 평가에 기여할 수 있는 가능성을 시사합니다.



### KurdSTS: The Kurdish Semantic Textual Similarity (https://arxiv.org/abs/2510.02336)
- **What's New**: 이번 연구에서는 의미 텍스트 유사성(Semantic Textual Similarity, STS)을 측정하는 최초의 쿠르디시(Kurdish) 데이터셋을 소개합니다. 이 데이터셋은 10,000개의 문장 쌍으로 구성되어 있으며, 공식적(formal) 및 비공식적(informal) 표현을 포함하여 유사성을 주석 처리하였습니다. 이는 낮은 자원(low-resource) 언어 분야에서 중요한 기여를 합니다.

- **Technical Details**: 쿠르디시 데이터셋은 형태소(morphology), 표기법(orthographic variation), 코드-믹스(code-mixing)와 같은 과제를 강조하며, Sentence-BERT, 다국어 BERT(multilingual BERT) 및 기타 강력한 기준과 비교하여 평가되었습니다. 이 연구는 쿠르디시 언어의 의미론(semantics)과 낮은 자원 NLP에 대한 향후 연구의 기초를 마련합니다.

- **Performance Highlights**: 이 연구에서는 문장 쌍 간 유사성을 측정하기 위해 여러 가지 모델을 벤치마킹하였으며, 경쟁력 있는 결과를 얻었습니다. 데이터셋과 기준은 재현 가능한 평가 스위트를 수립하였고, 향후 연구에 대한 강력한 출발점을 제공합니다.



### FormalML: A Benchmark for Evaluating Formal Subgoal Completion in Machine Learning Theory (https://arxiv.org/abs/2510.02335)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 현재 수준의 정리 증명 기능을 넘어, 수학자들이 연구 문제를 해결하는 데 필요한 중간 단계인 서브골 완성(subgoal completion)을 연구합니다. 이를 위해 FormalML이라는 새로운 Lean 4 벤치마크를 도입하였으며, 이는 머신러닝(machine learning)의 기초 이론에서 유래된 문제들을 포함합니다. 논문에서는 4937개의 다양한 난이도의 서브골 문제를 수집하여, 기존의 정리 증명 도구의 한계를 분석하고 이를 개선할 필요성을 강조합니다.

- **Technical Details**: 저자들은 Lean 4에서 서브골 추출을 위한 기법을 제안하고, 기계 학습 이론에 초점을 맞춘 FormalML 벤치마크를 구축하여 연구 수준의 정리 증명 문제에 대한 평가를 수행합니다. 이 벤치마크는 공개된 머신러닝 이론 라이브러리를 기반으로 하며, 기존의 평가 방식과 차별화된 서브골 완성 작업에 중점을 둡니다. 특히, 이에 따른 기법은 절차적 증명 스크립트에서 각 논리적 단계를 세분화하여 서브골을 효과적으로 추출하는 프로세스를 포함합니다.

- **Performance Highlights**: 주요 LLM 기반 증명 도구들의 FormalML에 대한 성능 평가 결과, 높은 난이도의 문제에서 성능 저하가 두드러짐을 알 수 있었습니다. 또한, Chain-of-thought prompting 기법이 자연어 추론에서는 효과적이나, 증명 완성에서는 효율성을 저하시키는 경향이 있음을 발견했습니다. 이러한 통찰력은 더 발전된 LLM 기반 정리 증명 도구가 수학자들을 더 잘 지원할 수 있도록 추가적인 개발이 필요함을 보여줍니다.



### Where Did It Go Wrong? Attributing Undesirable LLM Behaviors via Representation Gradient Tracing (https://arxiv.org/abs/2510.02334)
Comments:
          16 pages, 4 figures

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 바람직하지 않은 행동 원인을 진단하기 위한 새로운 프레임워크를 제시합니다. 기존의 접근법들이 가지는 한계를 극복하기 위해, 모델 활성화 공간(activation space)에서 표현과 그 기울기를 분석하여 의미 있는 정보를 제공합니다. 이 프레임워크는 샘플 레벨과 토큰 레벨 모두에서 LLM의 행동을 분석할 수 있는 능력을 보여줍니다.

- **Technical Details**: 제안된 프레임워크인 Representation Gradient Tracing (RepT)는 모델의 내부 표현을 분석하여 훈련 데이터의 원인과 LLM 반응 간의 관련성을 수립합니다. 이를 통해 기존의 매개변수(parameter) 공간 대신에 의미 있는 표현 공간을 중심으로 데이터를 추적합니다. RepT는 정보적인 레이어 선택, 샘플 수준의 귀속 방법, 그리고 특정 단어 및 구절을 식별하는 토큰 수준 분석을 통해 동작합니다.

- **Performance Highlights**: 제안된 방법은 유해 콘텐츠 추적, 백도어 포이즈 탐지 및 지식 오염 식별과 같은 다양한 과제에 대해 체계적으로 평가되었습니다. 결과는 이 방법이 샘플 수준의 귀속뿐만 아니라 특정 샘플과 구절을 정밀하게 식별하는 데 강력한 성능을 보인다는 것을 보여줍니다. 이를 통해 우리는 LLM의 위험을 이해하고 감시하며 궁극적으로 완화하기 위한 강력한 진단 도구를 제공할 수 있습니다.



### Human Mobility Datasets Enriched With Contextual and Social Dimensions (https://arxiv.org/abs/2510.02333)
Comments:
          5 pages, 3 figures, 1 table

- **What's New**: 이번 리소스 논문에서는 OpenStreetMap에서 수집한 두 개의 공개된 데이터셋을 소개합니다. 이 데이터셋은 GPS 궤적을 바탕으로 Contextual Layers(맥락 레이어)인 Stops(정차 지점), Moves(이동), Points of Interest(관심 지점, POI), 추론된 교통 수단 및 날씨 데이터가 포함되어 있습니다. 특히 Large Language Models(대형 언어 모델, LLM)로 생성된 합성 소셜 미디어 게시물이 포함되어 있어 다중 모드 및 의미론적 모빌리티 분석이 가능합니다.

- **Technical Details**: 이 데이터셋은 파리와 뉴욕이라는 두 개의 대도시에 대한 내용을 포함하고 있으며, 탭 형식과 Resource Description Framework(RDF) 형식으로 제공됩니다. 데이터는 OpenStreetMap에서 수집된 실시간 GPS 트레이스와 Weather conditions(날씨 정보) 등의 Contextual Layers로 구성되어 있습니다. 데이터는 사용자 ID가 지정된 GPS 궤적에서 수집되며, 익명화 및 사전 처리 작업을 통해 사용자 개인 정보 보호가 확보됩니다.

- **Performance Highlights**: 이 리소스는 행동 모델링, 모빌리티 예측, 지식 그래프 구성 및 LLM 기반 응용 프로그램 등 다양한 연구 작업을 지원합니다. 연구자들은 공개된 파이프라인을 통해 데이터셋을 커스터마이즈하고, 실시간 궤적, 내용 기반 정보 및 LLM에서 생성된 소셜 미디어 게시물을 활용할 수 있습니다. 이 논문의 제공하는 자원은 모빌리티 및 지식 관리 분야 연구자들의 실험과 검증을 지원하기 위해 설계되었습니다.



### A High-Capacity and Secure Disambiguation Algorithm for Neural Linguistic Steganography (https://arxiv.org/abs/2510.02332)
Comments:
          13 pages,7 figures

- **What's New**: 이 연구에서는 정보 은닉을 위한 새로운 방법인 Look-ahead Sync를 제안합니다. 이 방법은 SyncPool의 한계를 극복하고, 토큰화 모호성을 해소하면서도 보안성을 유지합니다. Look-ahead Sync는 진짜 구별할 수 없는 토큰 시퀀스에 대해서만 동기화 샘플링을 수행하여 임베딩 용량을 극대화합니다.

- **Technical Details**: Look-ahead Sync 알고리즘은 재귀적으로 동작하여, 사용 가능한 모든 경로의 엔트로피를 보존합니다. 이 연구는 zero-KL 보안을 제공하는 이론적 증명을 제시하고, Look-ahead Sync의 성능 한계를 분석합니다. 대규모 언어 모델을 사용하여 이 알고리즘이 임베딩 용량의 이론적 상한에 접근하는 것을 입증했습니다.

- **Performance Highlights**: 실험 결과에서 Look-ahead Sync는 영어에서 160%, 중국어에서 25% 이상의 임베딩 비율 향상을 보여주었습니다. 이 방법은 특히 더 큰 후보 풀을 가진 설정에서 뛰어난 결과를 낼 수 있음을 입증합니다. 이 연구는 고용량 보안 언어 스테가노그래피의 실용성을 더욱 향상시키는 중요한 기여로 평가됩니다.



### Synthetic Dialogue Generation for Interactive Conversational Elicitation & Recommendation (ICER) (https://arxiv.org/abs/2510.02331)
- **What's New**: 이 논문은 대화형 추천 시스템(CRS)에 언어 모델(LM)을 활용하는 새로운 방법론을 소개합니다. 비공식적인 CRS 데이터가 부족하여 LM의 미세 조정이 어려운 상황에서, 연구진은 LM을 사용자 시뮬레이터로 활용하여 자연스러운 대화 생성을 위한 새로운 방법을 개발했습니다. 그 결과, 사용자의 기초 상태에 일관된 자연어 대화를 생성하며, 공개 소스 CRS 데이터 세트도 생성하였습니다.

- **Technical Details**: 연구에서는 행동 시뮬레이터를 사용하여 사용자 선호 일관성을 유지하는 방법론을 제안합니다. 짧게 말해 LM 기반 사용자 시뮬레이터를 생성하고 다층적인 사용자 상호작용을 통해 선호도를 평가합니다. 이 과정에서는 다중 턴 사용자 상호작용을 생성하기 위해 기존 사용자 선택 및 응답 모델을 사용하고, LM-프롬프트( prompting)를 활용해 자연어 표현을 수정하여 비자연적이고 반복적인 대화 생성을 방지합니다.

- **Performance Highlights**: MD-DICER 데이터세트는 100,000개의 CRS 대화를 포함하고 있으며, 평가를 통해 생성된 대화가 자연스럽고 유창하며 사실적이라는 것을 보여주었습니다. 이 대화들은 기존의 템플릿 기반 대화보다 더욱 일관적으로 사용자 이해를 증진시켰습니다. 전체 및 부분 프리픽스(prompt) 대화의 평가 결과, 제안된 방법론이 LM 기반 CRSs와 행동 일관성을 갖춘 LM 기반 시뮬레이션에 대한 추가 연구를 촉진할 것임을 시사합니다.



### EntropyLong: Effective Long-Context Training via Predictive Uncertainty (https://arxiv.org/abs/2510.02330)
Comments:
          work in progress; Correspondence to: Xing Wu <wuxing@iie.this http URL>

- **What's New**: 이번 연구에서는 'EntropyLong'이라는 새로운 데이터 구성 방법을 소개합니다. 이 방법은 예측 불확실성을 활용하여 장기 의존성을 검증하는 것을 목표로 하며, 높은 엔트로피 위치에서 의미적으로 관련된 문맥을 검색하여 예측 엔트로피를 낮추는지를 평가합니다. 이러한 모델 중심 검증(model-in-the-loop verification)은 의미 있는 정보 획득(information gain)을 보장하여 단순히 상관관계를 기반으로 한 의존성에서 벗어납니다.

- **Technical Details**: EntropyLong 방법론은 네 단계의 파이프라인으로 구성되어 있습니다: 1) 적응형 임계값을 사용하여 높은 엔트로피 위치를 식별합니다. 2) 해당 위치에 대해 의미적으로 관련된 문맥을 검색합니다. 3) 검색된 문맥이 예측 엔트로피를 감소시키는지를 경험적으로 검증합니다. 4) 검증된 문맥을 원본 문서와 결합하여 장기 의존성이 있는 훈련 시퀀스를 생성합니다.

- **Performance Highlights**: EntropyLong로 모델을 훈련한 결과, RULER 벤치마크에서 8K-128K의 맥락 길이에 걸쳐 기존 접근법에 비해 두드러진 성능 향상을 보였습니다. 또한, 명령 조정 후 LongBench-v2에서의 성과도 크게 개선되었습니다. 다수의 제거 연구(ablation study)를 통해 엔트로피 기반 검증의 필요성과 효과를 추가로 입증하였습니다.



### SelfJudge: Faster Speculative Decoding via Self-Supervised Judge Verification (https://arxiv.org/abs/2510.02329)
- **What's New**: SelfJudge는 자가 감독(self-supervision)을 활용하여 넓은 NLP 작업에 걸쳐 판별자(validator)를 훈련시키는 새로운 방법을 제안합니다. 이는 인간의 주석이나 검증 가능한 정답이 필요하지 않아 다양한 자연어 처리(task)에서 활용할 수 있습니다. SelfJudge는 원래 응답의 의미를 보존하는지를 측정하여 자동으로 훈련 데이터를 생성하며, 이를 통해 모델의 추론(inference) 속도를 향상시킵니다.

- **Technical Details**: SelfJudge는 목표 모델(target model)의 응답과 대체된 토큰(token)을 비교하여 의미 보존(semantic preservation)에 기반한 판별 기준을 설정합니다. 이 방법은 원래 응답의 신뢰도에 크게 영향을 미치지 않는 토큰 교체를 허용하여 자동적으로 훈련 데이터를 생성합니다. SelfJudge는 다양한 NLP 작업에 대해 상반된 정보 개선과 효율성을 동시에 달성하는 데 중점을 둡니다.

- **Performance Highlights**: 실험 결과, SelfJudge는 기존 판별 방법에 비해 더 높은 정확도와 효율성을 제공합니다. 예를 들어, 이전 판별 방식인 AutoJudge는 성능 저하(-2.7% 정확도)에도 불구하고 +1.96개의 수용된 토큰을 달성했으나, SelfJudge는 단지 -1.0%의 정확도 저하로 +2.06개의 수용된 토큰을 기록했습니다. 이는 SelfJudge가 일반화 가능성(generalizability) 측면에서도 확장 가능하며, 여러 도메인에서의 LLM 추론을 빠르게 할 수 있는 최적의 솔루션임을 나타냅니다.



### AMANDA: Agentic Medical Knowledge Augmentation for Data-Efficient Medical Visual Question Answering (https://arxiv.org/abs/2510.02328)
Comments:
          EMNLP Findings

- **What's New**: 이 논문은 훈련 없는 데이터 효율적인 의료 비주얼 질문 응답(Med-VQA) 시스템인 AMANDA를 소개하며, 기존 Med-MLLM이 가지고 있는 본질적(intrinsic) 및 외부적(extrinsic) 추론 병목 현상을 해결합니다. AMANDA는 의료 지식 증강(Med-KA)을 통해 진단의 깊이를 향상시키고, 최신 의료 지식을 통합하여 정확한 응답을 제공합니다. 본 연구는 8개의 Med-VQA 벤치마크에서 제로샷(zero-shot) 및 몇샷(few-shot) 설정에서 주요 개선 사항을 입증합니다.

- **Technical Details**: AMANDA는 질문을 세분화(coarse-to-fine)하여 내재적인(intrinsic) 시각 이해 능력을 최대한 활용할 수 있도록 설계된 메커니즘입니다. 이 프레임워크는 의료 지식 그래프를 통해 외부 의료 지식을 검색하여 추론 과정을 구체화합니다. 여러 개의 LLM 에이전트를 통해 지식 통합의 깊이를 조절할 수 있으며, 이는 효과성과 효율성을 동시에 유지하는 데 기여합니다.

- **Performance Highlights**: 이 논문에서는 AMANDA가 기존의 접근 방식보다 월등한 성과를 보여 주며, 데이터 효율적인 연구의 가능성을 강화하고 있습니다. Zero-shot 및 few-shot 조건에서 실험을 통해 강력한 일반화 능력을 입증하며, 새로운 의료 비주얼 질문 응답 시스템의 가능성을 엿볼 수 있습니다. AMANDA는 의료 진단의 정확성을 높이고, AI 기반 의료 어시스턴트로서의 활용 가능성을 증대시킵니다.



### KAME: Tandem Architecture for Enhancing Knowledge in Real-Time Speech-to-Speech Conversational AI (https://arxiv.org/abs/2510.02327)
- **What's New**: 이 논문은 기존의 음성-음성(S2S) 모델의 장점과 자동 음성 인식(ASR) 및 텍스트 기반 대형 언어 모델(LLM)의 장점을 결합한 새로운 하이브리드 아키텍처를 소개합니다. 이 모델은 저지연 반응성을 유지하면서 깊은 지식을 통합할 수 있도록 설계되었습니다. 이는 대화형 AI 시스템에서 자연스러운 상호작용을 가능하게 합니다.

- **Technical Details**: 제안된 KAME 설계는 프론트엔드 S2S 모듈과 백엔드 LLM 모듈로 구성되어 있습니다. 프론트엔드 모듈은 사용자의 음성을 실시간으로 처리하여 즉각적인 응답을 생성하며, 백엔드 모듈에서 파생된 지식을 반영하여 응답의 질을 향상시킵니다. S2S 변환기는 여러 독립적인 토큰 시퀀스를 자동 회귀적으로 모델링하여 복잡한 대화의 맥락을 유지합니다.

- **Performance Highlights**: 평가 결과, KAME는 응답 정확도 면에서 기존 S2S 모델을 크게 초월하여 최신의 캐스케이드 시스템과 유사한 품질을 달성했습니다. 반면, 반응 속도는 기존 S2S 모델과 비슷하게 유지되고 있어, 실시간 적용 가능한 대화형 AI 구현에 적합합니다.



### Hallucination-Resistant, Domain-Specific Research Assistant with Self-Evaluation and Vector-Grounded Retrieva (https://arxiv.org/abs/2510.02326)
Comments:
          21 pages, 5 figures

- **What's New**: 이 논문에서는 RA-FSM(Research Assistant - Finite State Machine)을 소개합니다. 이는 생성 과정을 유한 상태 제어 루프에 감싸는 모듈형 GPT 기반의 연구 지원 도구로, 관련성(Relevance), 신뢰도(Confidence), 지식(Knowledge)의 순환을 기반으로 합니다. RA-FSM은 벡터 검색과 결정론적 인용 파이프라인에 기초하여, 전문가의 작업 흐름에서 더욱 유용하게 사용할 수 있는 기능을 제공합니다.

- **Technical Details**: RA-FSM의 설계는 유한 상태 기계(Finite State Machine, FSM)를 활용하여 쿼리를 필터링하고, 답변 가능성을 평가하며, 질문을 분해하고 필요할 때만 검색을 트리거하는 구조입니다. 또한, 각 답변에는 신뢰도 레이블과 중복되지 않은 인용이 제공되어 전반적인 사용성을 향상시킵니다. 이 외에도, RA-FSM은 학술 문헌, 학회, 지수, 사전 인쇄물 및 특허를 통해 특정 도메인 지식 기반을 구축하는 워크플로를 제공합니다.

- **Performance Highlights**: RA-FSM은 블라인드 A/B 리뷰에서 전문가들로부터 강력한 Notebook LM(NLM) 및 기본 GPT API 호출 대비 선호도를 나타냈습니다. 이는 경계 조건 처리를 더욱 효과적으로 하며, 더 신뢰할 수 있는 증거 사용을 가능하게 하였기 때문입니다. 커버리지와 신선도 분석 결과 RA-FSM은 NLM을 넘어서는 탐색이 가능하며, 조정 가능한 지연(latency)과 비용 오버헤드를 수반함을 보여주었습니다.



### Hallucination reduction with CASAL: Contrastive Activation Steering For Amortized Learning (https://arxiv.org/abs/2510.02324)
- **What's New**: CASAL(Contrastive Activation Steering for Amortized Learning)은 LLMs(Large Language Models)의 헛소리(hallucination)를 줄이는 새로운 알고리즘입니다. 기존의 방식들은 실시간 모니터링을 요구했지만, CASAL은 이러한 문제를 해결하기 위해 모델의 가중치(weights)에 activate steering의 이점을 직접 통합합니다. 이를 통해 모델은 아는 질문에는 답변하고, 모르는 질문에 대해서는 답변을 피할 수 있게 됩니다.

- **Technical Details**: CASAL은 단일 transformer layer의 서브모듈만을 사용하는 경량화된 설계로, 훈련(train)의 효율성을 극대화합니다. 이 방법은 여러 단기 QA(Question Answering) 벤치마크에서 헛소리를 30%-40% 감소시켜 보다 신뢰할 수 있는 응답을 제공합니다. 또한, CASAL은 LoRA(baseline) 기반의 SFT와 DPO에 비해 30배의 연산 효율(compute-efficiency)과 20배의 데이터 효율(data-efficiency)을 보이며, 데이터가 부족한 영역에서도 효과적으로 적용될 수 있습니다.

- **Performance Highlights**: CASAL은 OOD(out-of-distribution) 도메인에서도 일반화(generalization)가 잘 이루어지는 특징을 가지고 있습니다. 이 모델은 텍스트(text-only)와 비전-언어(vision-language) 모델 모두에서 헛소리를 완화하는 데 유연한 성능을 보여줍니다. CASAL은 또한 밀집 모델(dense models)과 전문가 혼합 모델(Mixture-of-Experts, MoE) 모두에 효과적인 첫 번째 steering-based training 방법으로, 운영 시스템에서의 실제 배포에 대한 가능성을 높일 수 있습니다.



### Low-probability Tokens Sustain Exploration in Reinforcement Learning with Verifiable Reward (https://arxiv.org/abs/2510.03222)
- **What's New**: 이번 연구는 Verifiable Rewards (RLVR)를 활용한 Reinforcement Learning의 확장성을 개선하기 위해 Low-probability Regularization (Lp-Reg)이라는 새로운 기법을 소개합니다. Lp-Reg는 저확률 탐색 토큰인 Reasoning Sparks를 보존하고 이를 보호하기 위해 노이즈를 필터링하여 탐색 능력을 강화하는 효과를 제공합니다. 기존 방법들이 간과했던 중요한 탐색 메커니즘을 면밀히 분석하여, 무작위성을 증가시키는 데서 오는 훈련 불안정성을 해결했습니다.

- **Technical Details**: 연구에서는 RLVR의 탐색 동역학을 연구하여, 훈련 과정에서 발생하는 'Reasoning Sparks'의 점진적 소멸 현상을 발견했습니다. 연구는 저확률 토큰을 필터링하여 지각되지 않은 노이즈를 제거한 후, 남은 후보들에 대해 재정규화를 수행하는 방법을 제안합니다. 이를 통해 효과적인 정규화 매개변수인 KL divergence를 활용하여 원래 정책이 필터링된 프로시 저확률 토큰을 보호하도록 유도합니다.

- **Performance Highlights**: 실험 결과, Lp-Reg는 약 1,000 스텝 동안 안정적인 on-policy 훈련을 가능하게 하였으며, 이는 기존의 엔트로피 통제 방법들이 붕괴된 지점에서 이루어졌습니다. 다섯 개의 수학 기준에서 평균 60.17%의 정확도를 달성하였고, 이는 이전 방법보다 2.66% 향상된 성과로 나타났습니다. 이러한 안정적인 탐색은 RLVR의 성능을 극대화하는 데 기여하고 있습니다.



### Coevolutionary Continuous Discrete Diffusion: Make Your Diffusion Language Model a Latent Reasoner (https://arxiv.org/abs/2510.03206)
Comments:
          27 pages

- **What's New**: 이번 연구에서 제안하는 Coevolutionary Continuous Discrete Diffusion (CCDD) 모델은 기존의 discrete diffusion 모델을 넘어 연속적인 확산 모델이 더 강력한 표현력을 가진다는 것을 입증합니다. CCDD는 연속적인 표현 공간과 불연속적인 토큰 공간을 결합하여, 두 모달리티의 장점을 살리면서 효과적인 denoising을 수행할 수 있도록 설계되었습니다. 이는 새로운 언어 모델링 패러다임을 제시하여 기존의 모델들이 직면한 훈련 가능성 및 표현력 문제를 해결하는 방향성을 제시합니다.

- **Technical Details**: CCDD는 joint multimodal diffusion 과정을 정의하며, CTMC(continuous-time Markov chain)와 SDE(Stochastic Differential Equation)를 활용하여 서로 다른 특성을 가진 두 공간에서 denoising을 동시에 수행합니다. 중첩된 표현 공간과 연속적인 확률 공간에서의 특성을 결합하여, 특히 CCDD는 언어 모델링에서 질 감도를 향상시키고 훈련 효율성을 높이기 위한 몇 가지 고급 아키텍처와 기술을 도입하고 있습니다. 또한, 새로운 샘플링 알고리즘을 적용하여 sampling quality와 efficiency 간의 균형을 조절합니다.

- **Performance Highlights**: 실험적으로 CCDD는 LM1B 데이터셋에서 기존의 동일한 크기 모델들에 비해 25% 이상의 성능 개선을 보여주었습니다. 이는 CCDD가 강력한 표현력과 저렴한 훈련 비용의 장점을 모두 겸비하고 있음을 입증합니다. CCDD의 아키텍처와 훈련 기법은 실제 언어 과제에서 뛰어난 성능을 보여주어, 차세대 언어 모델 개발에 기여할 것으로 기대됩니다.



### Simulation to Rules: A Dual-VLM Framework for Formal Visual Planning (https://arxiv.org/abs/2510.03182)
Comments:
          30 pages, 5 figures, 5 tables

- **What's New**:  이번 논문에서는 VLMFP라는 새로운 프레임워크를 제안하여, 시각적 계획을 위한 PDDL(Planning Domain Definition Language) 문제 파일과 도메인 파일을 자율적으로 생성합니다. 해당 프레임워크는 두 개의 VLM(Vision Language Model)인 SimVLM과 GenVLM을 통합하여 PDDL 파일 생성을 더욱 신뢰성 있게 만듭니다. 특히, SimVLM은 시뮬레이션을 통해 행동 결과를 예측하고, GenVLM은 이러한 예측 결과를 바탕으로 PDDL 파일을 생성 및 반복적으로 수정합니다.

- **Technical Details**:  VLMFP는 시각적 입력을 통해 문제와 도메인을 생성하기 위해 객체 인식, 공간 이해, 추론 및 PDDL 지식이 요구됩니다. 이 프레임워크는 시나리오를 정확하게 설명하고, 행동을 시뮬레이션하며 목표 도달 여부를 판단하는 SimVLM과 일반적인 추론 및 방대한 PDDL 지식을 활용하는 GenVLM으로 구성됩니다. 두 모델의 조화를 통해 고차원적인 생성 가능성을 달성하며, 생성된 도메인 PDDL은 동일한 도메인의 모든 인스턴스에 재사용할 수 있습니다.

- **Performance Highlights**:  VLMFP는 6개의 격자 세상(grid-world) 도메인에서 평가되었으며, SimVLM은 시나리오 설명 정확도에서 95.5%의 성과를 달성했습니다. 또한, VLMFP는 GPT-4o를 GenVLM으로 활용하여 보지 않은 인스턴스에 대해 각각 70.0%와 54.1%의 유효한 계획(successful plans) 성공률을 기록하며, 기존 방법들보다 뛰어난 성능을 보였습니다.



### When Names Disappear: Revealing What LLMs Actually Understand About Cod (https://arxiv.org/abs/2510.03178)
- **What's New**: 이 논문에서는 코드 이해에 대한 두 가지 주요 채널인 구조적 의미와 인간 해석 가능 이름을 도입합니다. 코드 요약 및 실행 작업에서 이름 제거가 의미를 손상시킬 수 있음을 발견했습니다. 또한 새로운 벤치마크인 ClassEval-Obf를 소개하여 이름 단서를 제거하면서도 프로그램의 행동을 유지하도록 설계되었습니다.

- **Technical Details**: 논문은 코드의 의미를 두 가지 채널, 즉 구조적/구문적 채널(구조 및 실행 동작)과 인간-자연 채널(식별자 이름 및 주석)로 나누어 탐구합니다. 다양한 종류의 obfuscation 기법을 소개하여 이름의 자연성을 점진적으로 약화시키며, 이를 통해 LLM(대규모 언어 모델)의 코드 이해 능력을 평가합니다. 주요 변환 유형에는 단순 구조적 이름 바꾸기, 모호한 식별자 사용, 관련 없는 분야의 용어로 대체 등이 포함됩니다.

- **Performance Highlights**: 연구 결과, 강한 obfuscation 하에 의도 풍부한 코드에서 성능이 급격히 감소하는 것으로 나타났습니다. 그러나 경쟁 프로그래밍 솔루션에서는 요약이 의도를 명확하게 유지하는 경향이 있었습니다. 이 연구는 LLM의 코드 이해 및 일반화 평가에 대한 신뢰성을 높이기 위해 obfuscation이 가진 효과를 명확히 드러내는 기초를 제공합니다.



### Beyond the Final Answer: Evaluating the Reasoning Trajectories of Tool-Augmented Agents (https://arxiv.org/abs/2510.02837)
Comments:
          Preprint. Under Review

- **What's New**: 최근 도구 증강 벤치마크는 복잡한 사용자 요청과 다양한 도구를 포함하고 있지만, 대부분의 평가 방법은 여전히 답변 매칭(answer matching)으로 한정되어 있습니다. 사용자 요청을 해결하는 데 필요한 단계가 증가함에 따라, 에이전트의 성능을 평가하려면 최종 답변을 넘어 문제 해결 과정(problem-solving trajectory)도 평가해야 합니다. 이 논문에서는 이를 위한 새로운 평가 프레임워크인 TRACE를 소개합니다.

- **Technical Details**: TRACE는 도구 증강 LLM 에이전트 성능의 다차원 평가를 위한 프레임워크입니다. 이 프레임워크는 이전의 추론 단계에서 수집한 지식을 축적하는 증거 은행(evidence bank)을 포함하여 에이전트의 추론 경로를 다각적으로 분석하고 평가할 수 있도록 합니다. 이를 통해 에이전트의 성능을 비용 효율적으로 평가할 수 있으며, 모든 유효한 실제 경로(ground-truth trajectory)의 주석 작업이 비현실적이라는 한계를 극복하고 있습니다.

- **Performance Highlights**: TRACE 프레임워크는 다양한 결함이 있는 경로가 포함된 새로운 메타 평가 데이터셋을 개발하여 기존 벤치마크를 보강합니다. 연구 결과는 TRACE가 복잡한 행동을 정확하게 평가할 수 있으며, 소규모 오픈 소스 LLM을 사용하더라도 확장 가능하고 비용 효율적인 방식으로 성능을 평가할 수 있음을 확인합니다. 또한, 도구 증강 작업을 수행하는 동안 에이전트가 생산한 경로를 평가하여 새로운 관찰 결과와 통찰력을 제시합니다.



### NCV: A Node-Wise Consistency Verification Approach for Low-Cost Structured Error Localization in LLM Reasoning (https://arxiv.org/abs/2510.02816)
- **What's New**: 본 논문에서는 Node-wise Consistency Verification (NCV)라는 새로운 프레임워크를 소개합니다. NCV는 다단계 추론 검증을 경량화된 이진 일관성 검사로 재구성하여, 전통적인 방법보다 정확도와 효율성을 높입니다. 이 프레임워크는 긴 추론 체인을 상호 연결된 검증 노드로 분해하여 오류를 정확히 확인하고, 필요 없는 긴 형식 생성을 피합니다.

- **Technical Details**: NCV는 복잡한 추론 검증을 структур화된 분해로 전환함으로써, 여러 단순한 검증 문제로 변환합니다. 이를 통해 비트 단위의 판단 문제를 나누어 처리할 수 있어, 정밀한 오류 위치 파악이 가능해집니다. 다양한 구조적 패턴을 가지며, 명확한 논리적 의존성이 있을 때는 명시적 엣지를 구축할 수 있고, 그렇지 않을 경우 선형 조건 체인으로 처리합니다.

- **Performance Highlights**: 실험 결과, NCV는 기존의 방법에 비해 F1 스코어에서 10%에서 25%까지 향상을 보였으며, 전통적인 방법보다 6배에서 58배 적은 토큰을 소모했습니다. 이러한 개선은 NCV의 구조적 분해 방식을 통해 달성된 결과로, 모든 벤치마크 세트에서 뛰어난 성능 개선을 입증했습니다.



### Pareto-optimal Non-uniform Language Generation (https://arxiv.org/abs/2510.02795)
Comments:
          24 pages, 1 figure

- **What's New**: 이번 연구에서는 Kleinberg와 Mullainathan이 제시한 언어 생성을 다루는 최신 모델을 발전시켜, 비균일 언어 생성(non-uniform language generation)에서의 Pareto Optimality에 대해 분석합니다. 특히, 기존 연구에서 제시된 알고리즘들은 언어에 따라 생성 시간이 최적이 아닐 수 있으며, 이에 대한 해결책을 제시합니다. 저자들은 생성 시간이 거의 Pareto-optimal한 새로운 알고리즘을 제안하여, 모든 언어에 대해 동시에 최적의 생성 시간을 달성하는 방법에 대한 기초를 마련합니다.

- **Technical Details**: 언어 생성 모델은 무한 문자열 집합에 기반한 개별 언어를 다루며, 적대자가 선택한 언어 $L$의 문자열을 온라인 방식으로 나열합니다. 높은 수준의 보장을 위해 비균일 언어 생성 개념이 도입되었으며, 이론적으로 구현 가능성이 보장됩니다. 본 연구의 알고리즘은 언어 $L$의 생성 시간 $t^ullet(L)$가 Pareto-optimal성을 가지도록 설계되어 있으며, 이는 특정 언어의 생성 시간이 다른 언어에 비해 나쁨이 없음을 의미합니다.

- **Performance Highlights**: 제안된 알고리즘은 무한 개의 언어에서 동시에 최적의 생성 시간을 달성하기 위한 새로운 기준을 제시합니다. 알고리즘은 기존의 연구에서 발견된 비균일 언어 생성 알고리즘들과 비교했을 때, 성능상에서 우수성을 보입니다. 이 작업은 노이즈가 있는 또는 대표적인 생성 설정에서도 Pareto-optimal 알고리즘을 얻을 수 있는 통합개념을 제공하여, 다양한 타당한 조건 하에서도 성능을 극대화할 수 있음을 입증합니다.



### MaskCD: Mitigating LVLM Hallucinations by Image Head Masked Contrastive Decoding (https://arxiv.org/abs/2510.02790)
Comments:
          accepted to emnlp2025 findings

- **What's New**: 논문에서는 이미지 헤드 마스크드 대비 디코딩(MaskCD)이라는 새로운 접근법을 제안합니다. 이 방법은 LVLMs의 '이미지 헤드'를 마스킹하여 contrastive decoding을 위한 부정적인 샘플을 구축함으로써, 환각 현상을 효과적으로 완화하는 방향으로 나아갑니다. MaskCD는 기존의 환각 완화 기법 대비 다양한 벤치마크에서 우수한 성능을 보이고 있습니다.

- **Technical Details**: LVLMs는 시각 정보와 텍스트 모달리티를 통합하여 다중 모달 추론을 수행하는 모델입니다. 본 연구에서는 LVLMs의 특정 레이어에서 '이미지 헤드'가 이미지 토큰에 과도한 주의를 기울이는 경향을 발견하였으며, 이들을 마스킹하여 부정적인 샘플을 만드는 방법론을 제시합니다. 기존의 대비 디코딩 방법(constrastive decoding)과 주의 조작(attention manipulation)의 장점을 결합하여 보다 높은 품질의 부정적 샘플을 구축할 수 있음을 강조합니다.

- **Performance Highlights**: MaskCD는 LLaVA-1.5-7b 및 Qwen-VL-7b 모델을 검증하는 다양한 벤치마크에서 실험되었습니다. 결과적으로 MaskCD는 환각 현상을 효과적으로 완화하면서 LVLMs의 기본적인 기능들을 유지하는 데 성공했습니다. 실험을 통해 MaskCD의 성능이 기존의 환각 완화 방법들을 초월한다는 점이 입증되었습니다.



### A Granular Study of Safety Pretraining under Model Abliteration (https://arxiv.org/abs/2510.02768)
Comments:
          Accepted at NeurIPS 2025 bWorkshop Lock-LLM. *Equal Contribution

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)의 사용이 증가함에 따라, 안전성을 위한 개입 방법들이 이러한 모델 수정에서 얼마나 견고하게 유지되는지를 조사하고 있습니다. 특히, 인퍼런스 타임(inference time)에서 모델을 변경할 수 있는 방법인 모델 압축(model abliteration)의 효과를 분석합니다. 연구는 SmolLM2-1.7B와 같은 안전 사전학습(Safety Pretraining)의 체크포인트를 활용하여, 안전성을 유지하는 데이터를 평가하고 그 결과를 기반으로 실제적인 프로토콜을 제안합니다.

- **Technical Details**: 연구에서는 강력한 안전성을 위한 데이터 중심 개입 방법을 검토하며, 저자들은 20개의 모델(원본 및 압축된 버전)에 대해 100개의 프롬프트(50 해로운 + 50 비해로운)를 통해 안전성의 견고성을 평가합니다. 더불어 자동화된 판단에서 발생할 수 있는 오류를 줄이기 위해 일부 프롬프트에 대해 인간의 주석을 추가하였으며, LLM 기반의 판단도 포함하고 있습니다. 연구 결과, 거부(Refusal) 전용 개입 방법들이 압축에 가장 취약하다는 것을 발견하였습니다.

- **Performance Highlights**: 본 연구는 다양한 안전 사전학습 체크포인트 및 공개 오픈 가중치 모델과의 비교를 통해 데이터 중심 안전 개입의 효율성을 평가하였으며, 각 체크포인트에서 안전성을 보장하는 요소를 분리하여 설명합니다. 연구 결과, 일부 안전 데이터 필터링 기법이 부분적인 견고성을 제공하는 것으로 나타났습니다. 또한, 모델이 자신의 출력을 거부하는 것을 인식하는 능력을 평가함으로써, 실제 배치 시스템에서의 자기 모니터링의 한계를 명확히 하고 있습니다.



### Hyperparameter Loss Surfaces Are Simple Near their Optima (https://arxiv.org/abs/2510.02721)
Comments:
          Accepted to COLM 2025. 23 pages, 8 figures

- **What's New**: 이번 연구에서는 하이퍼파라미터(hyperparameters) 손실 표면(loss surface)을 이해하기 위한 새로운 이론을 제안합니다. 이 연구는 복잡한 손실 표면을 탐색하는 대신, 최적 해에 접근함에 따라 단순한 구조가 나타난다는 점을 발견했습니다. 우리의 이론은 무작위 탐색(random search) 기반의 새로운 기술을 사용하여 이러한 구성 요소를 밝히는 데 기여합니다.

- **Technical Details**: 제안된 이론은 최적의 하이퍼파라미터 근처에서 손실 표면이 대략적으로 이차 다항식(quadratic polynomial) 형태로 나타나며, 노이즈(noise)는 정상(distributed normally) 분포를 따른다는 주장을 포함합니다. 이 이론을 통해 무작위 탐색의 실패 지점(threshold) 및 그로 인해 형성되는 비율 볼륨(parameter)과 같은 여러 속성들을 추론할 수 있습니다. 또한 이론은 1024개의 모델을 훈련하여 세 가지 실제 시나리오에서 검증되었습니다.

- **Performance Highlights**: 이론의 검증 결과는 랜덤 서치(random search)로 얻은 것이 실험적으로 측정된 데이터와 밀접하게 일치함을 보여줍니다. 각 실험에서 비대칭 영역은 전체 탐색 공간의 1/3에서 1/2를 차지했으며, 노이즈가 정상 분포로 수렴하는 진행 과정을 관찰했습니다. 이 작업은 하이퍼파라미터 손실 표면의 특성을 이해하고 무작위 탐색의 수렴 과정을 예측할 수 있게 하여, 연구자들이 불확실성을 정량화하면서 이러한 도구를 활용할 수 있도록 합니다.



### Less LLM, More Documents: Searching for Improved RAG (https://arxiv.org/abs/2510.02657)
Comments:
          16 pages. Submitted to ECIR 2026

- **What's New**: 이번 연구에서는 Retrieval-Augmented Generation(RAG)에서 생성기(generator)의 크기를 늘리는 전통적인 접근 대신, 검색기(retriever)의 데이터 코퍼스(corpus)를 확장하는 방법을 제안합니다. 실험 결과, 전체 Q&A 문제에 대해 코퍼스의 크기를 늘리는 것이 작은 LLMs와 조합할 때 이상적인 대안이 될 수 있음을 보였습니다. 이 연구는 작은 모델과 대규모 코퍼스의 조합이 대형 모델과 비슷한 성능을 발휘할 수 있음을 강조하고 있습니다.

- **Technical Details**: 저자들은 ClueWeb22 데이터셋을 사용하여 코퍼스를 여러 개의 균형 잡힌 샤드(shard)로 나누는 방식으로 코퍼스 스케일을 시뮬레이션했습니다. 실험은 다양한 크기의 질문 응답(QA) 벤치마크에서 진행되었으며, F1 및 Exact Match(EM) 스코어를 통해 성능을 평가했습니다. 이 연구에서는 코퍼스와 생성기 간의 거래(trade-off)를 체계적으로 분석하며, 작은 생성기가 특정 코퍼스 크기에서 더 큰 모델의 성능과 동등하게 만들 수 있는지를 탐구하고 있습니다.

- **Performance Highlights**: 실험 결과에 따르면, 1.7B 파라미터 모델이 4배 더 큰 코퍼스와 결합되면 4B 모델보다 더 나은 성능을 보이는 것으로 나타났습니다. 또한, 중간 크기의 모델이 가장 많은 이익을 얻는 반면, 작은 모델과 큰 모델은 변화가 적었습니다. 이 발견은 코퍼스의 확장이 생성기 크기를 늘리는 것과 대등한 성능 향상을 가져올 수 있다는 중요한 실용적 시사점을 제공합니다.



### HyperAdaLoRA: Accelerating LoRA Rank Allocation During Training via Hypernetworks without Sacrificing Performanc (https://arxiv.org/abs/2510.02630)
Comments:
          13 pages

- **What's New**: 본 논문에서는 HyperAdaLoRA라는 혁신적인 프레임워크를 제안하며, 이는 하이퍼네트워크(hypernetwork)를 활용하여 AdaLoRA의 수렴 속도를 크게 향상시킵니다. HyperAdaLoRA는 고정된 순위를 사용하지 않고 동적 순위 할당(dynamic rank allocation)을 도입하여 다양한 모듈 및 레이어의 중요도에 따라 가중치를 유연하게 조정합니다. 이 프레임워크는 파라미터 생성 시 최신 어텐션(attention) 메커니즘을 사용하여 전반적인 성능을 높입니다.

- **Technical Details**: HyperAdaLoRA는 하이퍼네트워크 기반으로 Singular Value Decomposition (SVD)의 구성요소인 P, Λ, Q를 직접 최적화하는 것이 아니라, 이를 동적으로 생성하여 파라미터 업데이트를 수행합니다. 이렇게 생성된 매트릭스의 출력은 트레이닝 과정에서 조정된 후, 고유한 중요도 기반으로 잘리지 않은 특이값을 통해 동적 순위 할당을 실현합니다. 이를 통해 트레이닝 중 발생하는 느린 수렴성과 높은 계산 부담 문제를 개선할 수 있습니다.

- **Performance Highlights**: 다양한 데이터셋과 모델에 대한 포괄적인 실험 결과, HyperAdaLoRA는 더 빠른 수렴 속도를 달성하면서도 성능 저하 없이 정확도를 유지하는 것으로 나타났습니다. 또한, 다른 LoRA 기반 방법들에 대한 실험을 통해 HyperAdaLoRA의 폭넓은 적용 가능성을 검증했습니다. 이러한 결과는 HyperAdaLoRA가 기존 방법들보다 효율적이고 강력하다는 것을 시사합니다.



### On the Role of Temperature Sampling in Test-Time Scaling (https://arxiv.org/abs/2510.02611)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 추론 시 reasoning을 개선하기 위해, 샘플링 온도를 다양하게 조절하는 다중 온도 스케일링(multi-temperature scaling) 방법을 제안하였습니다. 기존의 연구에서는 샘플 수(K)를 늘리면 정확도가 꾸준히 향상되었지만, K가 매우 커질 경우 더 이상의 향상이 발생하지 않는 한계를 발견했습니다. 이를 통해 단일 온도 스케일링이 모델의 잠재력을 완전히 탐색하지 못한다는 점을 밝혀냈습니다.

- **Technical Details**: 다중 온도 스케일링은 다양한 온도를 통해 샘플을 균등하게 나누어 모델의 reasoning 경계를 확장하는 접근입니다. 실험적으로 1,024개의 트레이스를 생성하는 동안 온도를 0.0에서 1.2까지 변화시키며 성능을 비교하였습니다. 이 과정에서, 특정 온도에서만 해결 가능한 어려운 문제들을 발견함으로써 모델의 전반적인 문제 해결 능력이 증가하는 것을 관찰했습니다.

- **Performance Highlights**: 평균적으로 Qwen3 모델(0.6B, 1.7B, 4B, 8B) 및 다섯 가지 대표적인 벤치마크를 넘어온 결과, 다중 온도 스케일링이 단일 온도 TTS에 비해 평균 7.3 포인트의 성능 향상을 보였습니다. 또한, 다중 온도 스케일링을 통해 기본 모델이 추가적인 후 훈련 없이도 RL-trained 모델과 유사한 성능에 도달할 수 있음을 증명했습니다. 이러한 결과는 TTS의 강력한 가능성과 다중 온도 스케일링의 효과적인 적용을 강조합니다.



### How Confident are Video Models? Empowering Video Models to Express their Uncertainty (https://arxiv.org/abs/2510.02571)
- **What's New**: 이 논문은 비디오 생성 모델의 불확실성 정량화(Uncertainty Quantification, UQ)에 대한 최초의 연구를 제안합니다. 기존의 텍스트-비디오 모델이 사용자 의도와 일치하지 않거나 잘못된 정보를 토대로 영상을 생성하는 '환각(hallucination)' 문제를 해결하고자 합니다. 새로운 방법론인 S-QUBED(Semantically-Quantifying Uncertainty with Bayesian Entropy Decomposition)를 통해 이는 더욱 정확하게 접근할 수 있습니다.

- **Technical Details**: S-QUBED는 비디오 생성 모델의 불확실성을 정밀하게 분해할 수 있는 블랙박스 UQ 방법으로, 조건부 확률을 모형화하여 두 단계로 비디오 생성을 모델링합니다. 이 방법은 조건부 독립성을 기반으로 Latent Variable(z)를 활용하여 예측 불확실성을 Aleatoric(우연적 불확실성)과 Epistemic(지식 기반 불확실성)으로 나눕니다. 이를 통해 입력 프롬프트의 모호함이나 모델의 지식 부족으로 인한 불확실성을 효과적으로 구별할 수 있습니다.

- **Performance Highlights**: S-QUBED는 다양한 비디오 생성 작업에서 불확실성 추정치를 조정하는 성능을 보여주며, 태스크 정확도와 부정적 상관 관계를 가지고 있습니다. 논문에 제시된 실험 결과는 S-QUBED가 비디오 모델의 불확실성을 정량화하는 데 유용하고 효과적임을 입증합니다. 이는 향후 비디오 모델의 안전성을 높이는 데 기여할 것으로 기대됩니다.



### Beyond Imitation: Recovering Dense Rewards from Demonstrations (https://arxiv.org/abs/2510.02493)
- **What's New**: 이번 연구에서는 기존의 감독 세밀 조정(Supervised Fine-Tuning, SFT)을 단순한 모방 학습 프로세스로 간주하는 시각에 도전합니다. 우리는 SFT와 역강화 학습(Inverse Reinforcement Learning, IRL) 사이의 근본적인 동치 관계를 수립하였습니다. SFT의 목표가 역 Q-학습(Inverse Q-Learning)의 특별한 경우임을 증명하여, SFT 프로세스가 단지 정책을 배우는 것이 아니라 전문가의 시연을 설명하는 암묵적이고 밀집된 토큰 수준 보상 모델도 함께 학습함을 보여줍니다.

- **Technical Details**: 이 연구에서는 유한 수명 토큰 MDP(다양한 상태 집합 𝒮, 행동 집합 𝒜, 결정적 전이 함수 f 및 초기 상태 분포 ρ0)가 포함된 환경을 다룹니다. 우리는 특정 상태 ss에서의 점유 측정(occupancy measure)과 로그 분배(log-partition) 및 볼츠만 정책(Boltzmann policy) 개념을 통해 이론적 기반을 제공합니다. 이를 통해 전문가의 시연으로부터 회복한 밀집 보상 신호를 통해 정치적 임무를 강화할 수 있는 새로운 가능성을 제안합니다.

- **Performance Highlights**: 우리는 회복한 보상을 사용하여 정책 성능을 더욱 개선할 수 있는 방법인 Dense-Path REINFORCE를 소개합니다. 이 방법은 명령 이행 벤치마크에서 원래 SFT 모델보다 일관되게 뛰어난 성능을 예시합니다. 따라서 이번 연구는 SFT를 단순히 정책 모방이 아닌 강력한 보상 학습 메커니즘으로 재구성하며, 전문가의 시연을 활용하여 새로운 가능성을 열어줍니다.



### Litespark Technical Report: High-Throughput, Energy-Efficient LLM Training Framework (https://arxiv.org/abs/2510.02483)
Comments:
          14 pages

- **What's New**: 본 논문에서는 Litespark라는 새로운 사전 훈련 프레임워크를 소개합니다. 이 프레임워크는 transformer architecture의 attention과 MLP layers에서 비효율성을 해결하여 훈련 시간을 단축하고 에너지 소비를 감소시키는 방법을 제안합니다. Litespark는 모델 성능을 극대화하면서도 기존 transformer 구현과의 호환성을 유지합니다.

- **Technical Details**: Litespark는 두 단계로 최적화를 수행합니다: 첫 번째는 architectural optimization으로 attention과 MLP 블록을 최적화하고, 두 번째는 algorithmic optimization으로 GPU당 FLOPs를 증가시키기 위해 순방향 및 역방향 연산을 최적화합니다. 이 프레임워크는 기존의 FlashAttention 기법과 같은 기술적 진보 위에 성능을 더할 수 있습니다.

- **Performance Highlights**: Litespark는 훈련 처리량을 2배에서 6배까지 증가시키고, 사전 훈련 과정에서 에너지 소비를 55%에서 83%까지 감소시킵니다. 이러한 최적화는 다양한 모델 아키텍처와 하드웨어에 적용 가능하여, 많은 산업계 응용에 유용할 것으로 기대됩니다.



### SIMSplat: Predictive Driving Scene Editing with Language-aligned 4D Gaussian Splatting (https://arxiv.org/abs/2510.02469)
- **What's New**: 새로운 접근 방식인 SIMSplat은 Predictive driving scene editor로, 자연어 프롬프트를 통해 직관적인 환경 조정이 가능하다. 이 시스템은 Gaussian으로 재구성된 장면과 언어를 정렬하여 도로 물체에 대한 직접적인 쿼리를 지원한다. 이러한 혁신은 한 가지의 에이전트에 국한된 편리한 조정을 넘어 여러 에이전트의 상호작용을 고도화하는 데 기여한다.

- **Technical Details**: SIMSplat은 motion-aware language embeddings를 통합하여 3D Gaussian 장면을 쿼리 및 조작할 수 있게 한다. LLM(large language model) 에이전트는 사용자의 프롬프트를 해석하여 객체를 추가, 제거 또는 수정할 수 있도록 한다. 또한, multi-agent motion prediction 모델을 통해 주변 에이전트의 움직임을 자연스럽게 반영하여 시뮬레이션의 현실감을 더욱 강화한다.

- **Performance Highlights**: SIMSplat은 Waymo 데이터셋에서 실험을 통해 탁월한 편집 및 시뮬레이션 능력을 입증하였다. 도로 물체 쿼리에서 정확성 기반의 기반선보다 61.2% 우수한 성능을 기록하며, 시뮬레이터 중 가장 높은 작업 완료율을 달성하였다. 게다가, 다중 에이전트 경로 개선 기능을 통해 예측 시뮬레이션에서도 가장 낮은 충돌 및 실패율을 기록하였다.



### How to Train Your Advisor: Steering Black-Box LLMs with Advisor Models (https://arxiv.org/abs/2510.02453)
- **What's New**: 본 논문에서는 Advisor Models라는 새로운 프레임워크를 소개합니다. 이 모델은 경량의 파라메트릭 정책으로, 블랙박스 모델에 대한 자연어 조언을 생성하여 반응적으로 조정하는 기능을 가지고 있습니다. Advisor Models는 단일 고정 프롬프트의 한계를 극복하고, 다양한 입력과 환경에 적응할 수 있는 다이나믹한 최적화를 가능하게 합니다.

- **Technical Details**: Advisor Models는 강화 학습(Reinforcement Learning)을 통해 최적화되며, 사용자 입력과 블랙박스 모델 사이에 위치하여 상황에 맞는 가이드를 생성합니다. 이 과정에서는 그룹 상대 정책 최적화(Group Relative Policy Optimization)를 사용하여 관찰된 보상에 기반하여 advisor를 업데이트합니다. 블랙박스 모델의 파라미터에 접근할 필요 없이, advice의 출력을 활용하여 블랙박스 모델을 제어합니다.

- **Performance Highlights**: Advisor Models는 개인화 및 도메인 적응이 필요한 다양한 환경에서 테스트되었으며, 특히 사용자 특정 규칙을 완벽히 학습하며 94-100%의 보상을 달성하는 높은 성능을 보여주었습니다. 또한, 다양한 블랙박스 모델 간의 advisor 전이도 성능 저하 없이 이루어지며, 분포 외 입력에 대한 강건성을 유지하는 결과를 보였습니다.



### CATMark: A Context-Aware Thresholding Framework for Robust Cross-Task Watermarking in Large Language Models (https://arxiv.org/abs/2510.02342)
- **What's New**: 이번 논문에서는 기존의 고정된 임계값에 의존하지 않고 문맥에 따라 동적으로 조정되는 Context-Aware Threshold Watermarking (CATMark) 프레임워크를 제안합니다. 이 시스템은 텍스트 생성 중의 세부적인 의미론적 상태를 기반으로 해서 수분할 텍스트 생성 방법을 통해 질 높은 판별 가능한 수조를 생성합니다. CATMark는 고유하게도 사전 정의된 임계값이나 특정 작업 튜닝 없이 작동합니다.

- **Technical Details**: CATMark는 로짓 클러스터링을 사용하여 텍스트 생성을 의미론적 상태로 분할하고, 이러한 상태에 맞춰 문맥 인식 엔트로피 임계값을 설정합니다. 이 방법은 고엔트로피 텍스트에 강한 워터마크를 적용하면서도 저엔트로피 텍스트의 무결성을 보장합니다. 이를 통해 다양한 생성 작업에서의 상대적 성능을 극대화할 수 있으며, 인공 신경망 모델의 샘플링 과정을 미세 조정함으로써 더 나은 결과를 도출합니다.

- **Performance Highlights**: 실험 결과 CATMark는 교차 작업에서 텍스트 품질을 개선하며 탐지 정확도를 희생하지 않고 있습니다. HumanEval에서 82.3%의 pass@1 점수와 StackEval에서 100% AUROC를 기록하여 기존 벤치마크 방법들을 초월하는 성과를 보여주었습니다. 이는 복잡하고 다양한 애플리케이션 내에서 LLM의 안정적인 활용을 가능하게 합니다.



### SpeechCT-CLIP: Distilling Text-Image Knowledge to Speech for Voice-Native Multimodal CT Analysis (https://arxiv.org/abs/2510.02322)
Comments:
          Submitted to ICASSP 2026; under review

- **What's New**: 최근 연구에서는 스피치(Spoken) 의료 보고서를 기반으로 한 시각-언어 표현 학습의 가능성을 제시합니다. 이 연구에서는 Speech-RATE라는 데이터셋을 새롭게 생성하고, SpeechCT-CLIP 모델을 통해 음성과 3D CT 볼륨을 공유 표현 공간에서 정렬하는 방법을 탐구합니다. 연구 결과, 사전 훈련된 텍스트-이미지 CLIP 모델로부터의 지식 증류(Knowledge Distillation)를 통해 음성 기반 모델의 성능을 크게 향상시키는 방법을 발견했습니다.

- **Technical Details**: SpeechCT-CLIP 모델은 음성 인코더와 CT 인코더를 대조적(cosntrative) 및 증류 목표(distillation objectives)를 통해 정렬합니다. Speech-RATE 데이터셋은 50,188개의 CT 볼륨과 음성 보고서를 포함하며, 각 보고서를 다양한 목소리로 합성해 현실적인 음성 인식을 시뮬레이션합니다. 훈련 과정에서는 슬라이딩 윈도우 전략을 사용하여 긴 음성을 처리하고, 이러한 음성과 CT 간의 정렬을 위해 대조적 손실 함수를 최소화합니다.

- **Performance Highlights**: 실험 결과, SpeechCT-CLIP 모델은 제로샷(zero-shot) 분류에서 F1 점수가 0.623에서 0.705로 향상되었으며, 이는 성능 차이의 88%를 회복한 것입니다. 또한, 텍스트 없이도 강력한 검색 결과를 보여주어, 음성을 제목으로 한 의료 AI 시스템의 실용성을 강조합니다. 이러한 결과는 음성이 텍스트를 대체할 수 있는 유망한 수단임을 보여주며, 임상 진단 지원 도구 개발의 가능성을 열어줍니다.



### WEE-Therapy: A Mixture of Weak Encoders Framework for Psychological Counseling Dialogue Analysis (https://arxiv.org/abs/2510.02320)
Comments:
          5 pages

- **What's New**: 본 논문은 심리 상담 분석을 위해 설계된 multi-task AudioLLM인 WEE-Therapy를 소개합니다. 기존의 오디오 언어 모델들이 일반 데이터를 기반으로 학습된 단일 스피치 인코더에 의존하는 반면, WEE-Therapy는 Weak Encoder Ensemble (WEE) 메커니즘을 통합하여 전문적이고도 복잡한 감정을 처리하는 능력을 향상시킵니다. 이를 통해 심리 상담에서 발생하는 복잡한 대화의 감정적 요소와 기술적 세부사항을 효과적으로 파악할 수 있습니다.

- **Technical Details**: WEE-Therapy 구조는 강력한 기본 인코더와 경량화된 전문 인코더의 조합으로 이루어져 있습니다. 기본 인코더는 Whisper-large-v3를 사용하고, 여러 개의 '약한' 인코더들은 심리 상담에서 놓칠 수 있는 세밀한 특징을 보완합니다. 특히, dual-routing 전략을 통해 데이터 독립적이고 의존적인 방식으로 인코더의 특성을 결합하여 최종 오디오 표현을 생성하는 혁신적인 방식을 적용합니다.

- **Performance Highlights**: WEE-Therapy는 감정 인식, 기술 분류, 위험 탐지 및 요약 작업을 포함한 네 가지 주요 작업에서 평가되었습니다. 실험 결과, 모든 작업에서 성능이 크게 향상되었으며, 모델의 파라미터 오버헤드는 최소화되었습니다. 이러한 결과는 AI 지원 클리닉 분석을 위한 강력한 잠재력을 보여주고 있습니다.



### Modeling the Attack: Detecting AI-Generated Text by Quantifying Adversarial Perturbations (https://arxiv.org/abs/2510.02319)
Comments:
          8 pages, 3 figures

- **What's New**: 최근 대규모 언어 모델(Large Language Model, LLM)의 발전은 AI 생성 텍스트 감지 시스템의 필요성을 더욱 부각시키고 있습니다. 이 논문은 기존의 공격에 대한 감지기의 취약성을 분석하고, 새로운 방어 프레임워크인 Perturbation-Invariant Feature Engineering (PIFE)을 도입하여 감지 성능을 향상시키는 방법을 제시합니다. PIFE는 입력 텍스트를 표준화한 후, 변환의 정도를 측정하여 신호를 분류기에 직접 전달하는 방식으로 작동합니다.

- **Technical Details**: 이 연구는 전통적인 적대적 훈련의 한계를 정량화하고, 텍스트와 그 정규형 간의 불일치를 모델링하여 적대적 공격에 대해 보다 강력한 감지기의 구조를 설계합니다. 본 연구에서는 기본적으로 Transformer 구조를 기반으로 한 감지기를 사용하며, 문자, 단어, 문장 수준의 다양한 공격을 평가하여 각 모델의 강건성을 비교 분석합니다. 감지 성능 평가는 True Positive Rate (TPR)와 False Positive Rate (FPR) 기준으로 수행됩니다.

- **Performance Highlights**: PIFE 모델은 기존의 적대적 훈련 기법이 세멘틱 공격에 취약한 반면, 1%의 FPR 하에서도 82.6%의 TPR을 유지하며 효과적으로 공격을 무력화할 수 있음을 보여줍니다. 이는 텍스트에 대한 변형 아티팩트를 명확히 모델링하는 것이 진정한 강건성을 실현하는 보다 유망한 경로임을 입증합니다. 이 연구는 AI-generated text detection의 문제를 해결하기 위한 새로운 가능성을 제시합니다.



### Learning to Parallel: Accelerating Diffusion Large Language Models via Learnable Parallel Decoding (https://arxiv.org/abs/2509.25188)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)에서의 병렬 디코딩을 위한 새로운 프레임워크인 Learning to Parallel Decode (Learn2PD)를 제안합니다. 본 연구는 각 토큰 위치에 대해 예측이 최종 출력과 일치하는지를 판단하는 경량의 적응형 필터 모델을 훈련시키는 방법을 탐구합니다. 특히, 마지막 토큰의 예측이 완료되었음을 감지하는 End-of-Text Prediction (EoTP) 메커니즘을 도입하여 불필요한 패딩 토큰의 디코딩을 방지합니다.

- **Technical Details**: Learn2PD는 각 토큰의 예측 결과를 기반으로 병렬 디코딩 시 최적화를 위해 설계되었습니다. 이 과정에서 훈련된 필터 모델은 매우 적은 GPU 시간(분 수준)으로 최적화되며, 최종 디코딩 과정에서 그 어떤 추가적인 그라디언트 업데이트 없이 고정된 상태로 유지됩니다. 또한, EoTP는 [End-of-Text] 토큰의 예측이 완료됨과 동시에 디코딩을 종료하여 계산의 비효율성을 줄입니다.

- **Performance Highlights**: 실험 결과, Learn2PD는 LLaDA 벤치마크에서 최대 22.58배의 속도 향상을 이루어 내며, 정확도에는 단 하나의 성능 저하 없이 결과를 유지하고 있습니다. KV-캐시(KV-Cache)와 결합 시에는 속도가 57.51배로 증가하면서도 약간의 정확도 저하만 발생합니다. 이러한 결과는 다양한 자연어 처리(NLP) 작업에서의 디코딩 효율성 필터링 자원의 낭비를 줄이는 데 기여합니다.



### jina-reranker-v3: Last but Not Late Interaction for Listwise Document Reranking (https://arxiv.org/abs/2509.25085)
Comments:
          early draft, CodeIR table needs to be updated (qwen baselines are missing)

- **What's New**: 이번 연구에서 소개된 jina-reranker-v3는 0.6B 파라미터의 다국적 리스트 재정렬 모델로, "last but not late" 상호작용 방식을 도입하였습니다. 기존의 late interaction 모델들과는 달리, 이 모델은 쿼리와 모든 문서 간의 인과적 주의를 적용하여 문서들을 인코딩하기 전에 풍부한 상호작용을 가능하게 하여 컨텍스트 임베딩을 추출할 수 있게 합니다.

- **Technical Details**: jina-reranker-v3는 Qwen3-0.6B 아키텍처를 기반으로 하며, 28개의 transformer 레이어, 1024 개의 hidden 차원, 16 개의 attention 헤드 및 131K token 컨텍스트 용량을 가지고 있습니다. 크로스 문서 상호작용을 가능하게 하는 LBNL 방식을 사용하여, 모든 문서와 쿼리를 동시에 처리하여 의미의 풍부한 임베딩을 생성합니다.

- **Performance Highlights**: 평가 결과 jina-reranker-v3는 BEIR에서 61.94 nDCG@10을 달성하여 이전 모델인 jina-reranker-v2에 비해 4.79% 향상되었습니다. 이 모델은 HotpotQA에서 78.58, FEVER에서 94.01, MIRACL에서 66.83 및 MKQA에서 67.92 Recall@10으로 다국적 성능에서도 경쟁력을 보여주고 있습니다.



New uploads on arXiv(cs.IR)

### OpenZL: A Graph-Based Model for Compression (https://arxiv.org/abs/2510.03203)
- **What's New**: 이번 논문에서는 전통적인 손실 없는 압축 방식에 있어, 높은 압축 비율(compression ratio)을 달성하기 위해 리소스 사용(resource utilization)과 처리량(processing throughput)에 큰 비용이 들었던 과거 연구를 다루고 있습니다. 그러나 현대의 생산 작업(workloads)은 높은 처리량과 낮은 리소스 사용을 필요로 합니다. 이에 따라 저자들은 'graph model'이라는 새로운 이론적 프레임워크를 제안하며, 이를 통해 모듈식 코덱(modular codecs)을 사용하는 압축 방식을 구축할 수 있다는 점을 강조합니다.

- **Technical Details**: 저자들은 OpenZL을 제안하는데, 이는 데이터 압축을 자기 설명(self-describing) 데이터 포맷으로 변환하는 새롭고 혁신적인 방법입니다. 이 모델은 모든 구성(configuration)이 범용 디코더(universal decoder)를 통해 디코딩될 수 있도록 설계되었습니다. OpenZL의 구조는 최소한의 코드로 맞춤형 압축기를 신속하게 개발할 수 있도록 하며, 또 범용 디코더의 존재는 배포 지연(deployment lag)을 없앱니다.

- **Performance Highlights**: 실험 결과, OpenZL은 다양한 실제 데이터셋(real-world datasets)에서 최신의 고급 일반 목적 압축기(state-of-the-art general-purpose compressors)보다 뛰어난 압축 비율과 속도를 달성했습니다. Meta에서의 내부 배포 결과 또한 크기 및/또는 속도의 일관된 향상을 보여주었으며, 개발 시간이 월 단위에서 일 단위로 단축되었습니다. 이렇게 OpenZL은 현대 데이터 집약적인 애플리케이션을 위한 실용적이고 확장 가능하며 유지보수가 용이한 데이터 압축의 진전을 나타냅니다.



### AgenticRAG: Tool-Augmented Foundation Models for Zero-Shot Explainable Recommender Systems (https://arxiv.org/abs/2510.02668)
- **What's New**: 본 논문은 추천 시스템에서의 제한된 활용을 극복하기 위해 AgenticRAG라는 혁신적인 프레임워크를 소개합니다. 이 프레임워크는 외부 도구 호출(tool invocation), 지식 검색(knowledge retrieval), 사유 체인(chain-of-thought reasoning)을 결합하여 설명 가능한 추천을 제로샷(zero-shot)으로 제공합니다. 이를 통해 사용자가 추천에 대한 투명한 의사결정 과정을 경험할 수 있습니다.

- **Technical Details**: AgenticRAG 프레임워크는 RAG(검색 증강 생성) 방식을 활용하여 동적 정보 통합을 가능케 하며, 실시간 데이터 접근을 위한 외부 도구 호출 시스템과 투명한 의사결정을 위한 사유 엔진을 포함합니다. 이 구조는 과거의 특정 작업 훈련 없이 다양한 추천 시나리오에서 일반화할 수 있는 능력을 가진 기초 모델(foundation models)들을 활용합니다. 추천 과정은 사용자 질의 처리, 지식 검색, 도구 호출, 사유 과정을 병렬적으로 실행하여 개인 맞춤형 추천을 생성합니다.

- **Performance Highlights**: 세 가지 실제 데이터셋에서 실험 결과, AgenticRAG는 최첨단 기준선보다 일관된 성능 향상을 달성했습니다. Amazon Electronics에서는 NDCG@10에서 0.4% 개선, MovieLens-1M에서는 0.8% 개선, Yelp 데이터셋에서 1.6% 개선을 보였습니다. 이 프레임워크는 설명 가능성(explainability)에서 우수한 성능을 보이며, 전통적인 방법과 유사한 계산 효율성을 유지하고 있습니다.



### Less LLM, More Documents: Searching for Improved RAG (https://arxiv.org/abs/2510.02657)
Comments:
          16 pages. Submitted to ECIR 2026

- **What's New**: 이번 연구에서는 Retrieval-Augmented Generation(RAG)에서 생성기(generator)의 크기를 늘리는 전통적인 접근 대신, 검색기(retriever)의 데이터 코퍼스(corpus)를 확장하는 방법을 제안합니다. 실험 결과, 전체 Q&A 문제에 대해 코퍼스의 크기를 늘리는 것이 작은 LLMs와 조합할 때 이상적인 대안이 될 수 있음을 보였습니다. 이 연구는 작은 모델과 대규모 코퍼스의 조합이 대형 모델과 비슷한 성능을 발휘할 수 있음을 강조하고 있습니다.

- **Technical Details**: 저자들은 ClueWeb22 데이터셋을 사용하여 코퍼스를 여러 개의 균형 잡힌 샤드(shard)로 나누는 방식으로 코퍼스 스케일을 시뮬레이션했습니다. 실험은 다양한 크기의 질문 응답(QA) 벤치마크에서 진행되었으며, F1 및 Exact Match(EM) 스코어를 통해 성능을 평가했습니다. 이 연구에서는 코퍼스와 생성기 간의 거래(trade-off)를 체계적으로 분석하며, 작은 생성기가 특정 코퍼스 크기에서 더 큰 모델의 성능과 동등하게 만들 수 있는지를 탐구하고 있습니다.

- **Performance Highlights**: 실험 결과에 따르면, 1.7B 파라미터 모델이 4배 더 큰 코퍼스와 결합되면 4B 모델보다 더 나은 성능을 보이는 것으로 나타났습니다. 또한, 중간 크기의 모델이 가장 많은 이익을 얻는 반면, 작은 모델과 큰 모델은 변화가 적었습니다. 이 발견은 코퍼스의 확장이 생성기 크기를 늘리는 것과 대등한 성능 향상을 가져올 수 있다는 중요한 실용적 시사점을 제공합니다.



### A Simple but Effective Elaborative Query Reformulation Approach for Natural Language Recommendation (https://arxiv.org/abs/2510.02656)
Comments:
          11 pages, 5 figures

- **What's New**: 이 논문에서는 NL 추천 시스템에서 사용자 쿼리와 항목 설명을 바탕으로 관련 항목을 검색하는 새로운 접근법을 제시합니다. 기존의 Dense Retrieval (DR) 방식은 복잡한 쿼리에 잘 대응하지 못하는 반면, EQR (Elaborative Subtopic Query Reformulation) 방식은 쿼리의 폭과 깊이를 동시에 충족하여 처리합니다. 이와 더불어, 여행, 호텔, 음식 등 세 가지 새로운 미션 기반의 NL 추천 벤치마크를 소개합니다.

- **Technical Details**: EQR는 기존의 Query Reformulation (QR) 방법의 한계를 극복하기 위해 설계되었습니다. 이 방법은 사용자 쿼리에서 여러 부주제를 유추하고, 각 부주제에 대해 정보가 풍부한 설명을 생성하는 데 초점을 맞춥니다. 따라서 EQR는 대규모 언어 모델(LLM)을 활용하여, 쿼리에 대해 다각적인 시각을 제시하며 각 쿼리의 의미를 확장합니다.

- **Performance Highlights**: 실험 결과, EQR은 여러 평가 지표에서 최신의 QR 방법들보다 월등한 성능을 보여주었습니다. 이는 넓고 간접적인 사용자 의도를 가진 쿼리에 대해 NL 추천 시스템을 효과적으로 개선할 수 있는 가능성을 나타냅니다. EQR의 간단하면서도 직관적인 설계는 다양한 데이터셋에서 일관되게 우수한 결과를 기록했습니다.



### Revisiting Query Variants: The Advantage of Retrieval Over Generation of Query Variants for Effective QPP (https://arxiv.org/abs/2510.02512)
Comments:
          11 pages, 4 figures

- **What's New**: 본 논문에서는 쿼리 성능 예측(query performance prediction, QPP)에서 정보 요구가 유사한 쿼리 변형(query variants, QVs)을 활용하는 새로운 방법을 제안합니다. 기존 방법들은 쿼리 확장 또는 비맥락적 임베딩을 사용하여 QVs를 생성하는데, 이는 주제의 편향이나 해시를 유발할 수 있습니다. 본 연구에서는 훈련 세트에서 QVs를 검색하여 QPP에 활용하며, 1-홉(QV) 이상의 질의 정보를 확보하는 2-홉 QVs를 통한 검색 방법을 설명합니다.

- **Technical Details**: QVP 모델에서는 타겟 쿼리와 그 검색 결과를 입력으로 받아 QPP의 품질 점수를 산출합니다. 새로운 접근법은 1-홉 QVs에서 관련 문서를 활용하여 추가적인 2-홉 QVs를 검색함으로써 가능한 후보 세트를 확장합니다. 이를 통해 정보 요구가 비슷한 QVs를 확보하여 QPP 모델의 효율성을 높이는 기법을 개발하고자 합니다.

- **Performance Highlights**: 실험 결과, 제안된 QVs 검색 방법은 MonoT5와 같은 신경 랭킹 모델에서 기존 방법들보다 약 20%의 성능 향상을 보여주었습니다. 이러한 성과는 쿼리 성능 예측의 효과성을 크게 향상시킬 수 있는 잠재력을 나타냅니다. 따라서 IR 훈련 세트에서 QVs를 활용하는 접근 방식이 본 연구의 주요 기여입니다.



### CHORD: Customizing Hybrid-precision On-device Model for Sequential Recommendation with Device-cloud Collaboration (https://arxiv.org/abs/2510.03038)
Comments:
          accepted by ACM MM'25

- **What's New**: 이번 연구에서는 CHORD라는 새로운 프레임워크를 제안합니다. CHORD는 하이브리드 정밀도를 가진 온디바이스 모델을 커스터마이징하여 장치-클라우드 협력 방식을 통해 순차적 추천을 구현합니다. 이를 통해 모델 정확도를 유지하면서 자원 적응형 배포를 달성할 수 있습니다.

- **Technical Details**: CHORD에서는 채널별 혼합정밀도 양자화(channel-wise mixed-precision quantization)를 활용하여 사용자 맞춤형 추천을 제공합니다. 연구에서는 모델 파라미터의 민감도 분석을 통해 사용자 프로필에 맞는 양자화 전략을 쉽게 매핑하는 방법을 개발하였습니다. 이 과정에서 모델 압축을 달성하면서도 커뮤니케이션 오버헤드를 줄이는데 집중하였습니다.

- **Performance Highlights**: 세 가지 실제 데이터셋과 두 가지 인기 있는 네트워크(SASRec 및 Caser)에 대한 실험 결과, CHORD는 정확성, 효율성, 적응성을 모두 입증했습니다. 사용자는 2비트의 전략 인코딩을 통해 클라우드와의 소통 비용을 크게 줄이고, 한 번의 전방 패스를 통해 개인화된 모델 적응을 이룰 수 있음을 보였습니다.



### Grounding Large Language Models in Clinical Evidence: A Retrieval-Augmented Generation System for Querying UK NICE Clinical Guidelines (https://arxiv.org/abs/2510.02967)
- **What's New**: 이 논문은 영국의 건강 및 치료 우수성 국가 연구소(NICE)의 임상 가이드라인을 질의하기 위한 Retrieval-Augmented Generation (RAG) 시스템 개발 및 평가를 다룹니다. 이 시스템은 사용자가 자연어 질의에 따라 정밀하게 일치하는 정보를 제공하도록 설계되었으며, 대규모 언어 모델(LLMs)을 사용하여 임상 가이드라인의 긴 길이와 대량의 데이터를 활용할 수 있도록 합니다.

- **Technical Details**: RAG 시스템은 하이브리드 임베딩 메커니즘으로 구성된 검색 아키텍처를 사용하여 300개 가이드라인에서 유도된 10,195개의 텍스트 청크 데이터베이스와 비교 평가되었습니다. 이 시스템은 7,901건의 질의에서 평균 역순위(Mean Reciprocal Rank, MRR) 0.814를 기록하고, 첫 번째 청크에서 81%, 상위 10개 청크에서 99.1%의 리콜(validation accuracy)을 달성하는 성과를 보였습니다.

- **Performance Highlights**: RAG 시스템은 생성 단계에서 가장 두드러진 성과를 보였으며, 70개의 질문-답변 쌍을 수작업으로 선별한 데이터셋에 대해 RAG 강화 모델이 성능에서 상당한 개선을 보였습니다. RAG 강화 O4-Mini 모델의 신뢰성(Faithfulness)은 64.7% 증가하여 99.5%에 달하며, 의학 중심의 Meditron3-8B LLM의 43%와 비교하여 훨씬 높은 성능을 보여주었습니다. 이 연구는 RAG를 의료 분야에서 생성 AI를 효과적이고 신뢰성 높게 적용할 수 있는 접근 방식으로 입증하였습니다.



### StepChain GraphRAG: Reasoning Over Knowledge Graphs for Multi-Hop Question Answering (https://arxiv.org/abs/2510.02827)
- **What's New**: StepChain GraphRAG는 Retrieval-Augmented Generation(RAG) 접근 방식을 기반으로 하여 복잡한 다중 홉 질문 대응을 위한 새로운 프레임워크를 소개합니다. 이 모델은 Breadth-First Search(BFS) 추론 흐름을 통합하여 질문을 재구성하고, 외부 지식을 효과적으로 연결하여 정보의 일관성을 유지합니다. 실험 결과, 이 모델은 여러 벤치마크에서 기존의 방법보다 우수한 성과를 나타냈으며, 특히 HotpotQA에서 큰 성과를 기록했습니다.

- **Technical Details**: 이 프레임워크는 전체 코퍼스에 대한 글로벌 인덱스를 구축하고, 추론 단계에서 선택된 패시지를 즉시 지식 그래프로 변환합니다. 각 서브 질문은 BFS 기반의 탐색을 통해 관련된 증거 체인을 구축하며, 이는 불필요한 문맥으로 모델을 압도하지 않도록 돕습니다. 또한, 프레임워크는 매 세부 질문에서 새로운 증거를 그래프에 동적으로 통합하여, 중간 추론 단계를 효과적으로 기록하고 업데이트합니다.

- **Performance Highlights**: StepChain GraphRAG는 MuSiQue, 2WikiMultiHopQA 및 HotpotQA에서 SOTA(state-of-the-art) 정확도 및 F1 점수를 달성했습니다. 이 방법은 평균적으로 EM이 2.57%, F1이 2.13% 향상되었으며, 특히 HotpotQA에서 EM이 4.70%, F1이 3.44% 향상되었습니다. 이러한 성과는 중간 단계에서의 사고 과정을 명확히 유지하여 설명 가능성을 높이는 데 기여했습니다.



### AutoMaAS: Self-Evolving Multi-Agent Architecture Search for Large Language Models (https://arxiv.org/abs/2510.02669)
- **What's New**: AutoMaAS는 자가 발전하는 다중 에이전트 아키텍처 검색 프레임워크로서 기존의 단일 아키텍처 솔루션의 한계를 극복합니다. 이 프레임워크는 쿼리 복잡성과 도메인 요구 사항에 따라 자원 할당을 동적으로 조절할 수 있는 능력을 지니고 있습니다. 특히, 네 가지 주요 혁신을 포함하여 성능과 비용 분석을 통한 자동 엔진 생성, 통합 및 제거, 실시간 매개 변수 조정 등을 통해 효과적인 아키텍처 설계를 관한 새로운 패러다임을 제시합니다.

- **Technical Details**: AutoMaAS는 네 가지 구성 요소로 이루어져 있습니다: 동적 작업자 생애 주기 관리자, 다중 목표 비용 최적화기, 온라인 피드백 통합 모듈, 아키텍처 해석 엔진입니다. 이 시스템은 각 작업자의 성능 지표와 사용 패턴을 기준으로 작업자를 동적으로 평가하고 생성하는 과정을 자동화합니다. 또한, 기존의 고정된 작업자 풀 대신, 변화하는 환경에서 자동으로 작업자를 생성 및 제거하여 다양한 쿼리 특성에 적합한 아키텍처를 선택할 수 있습니다.

- **Performance Highlights**: AutoMaAS는 6개의 벤치마크에서 성능 개선을 1.0-7.1% 달성하고, 기존 최첨단 방법에 비해 추론 비용을 3-5% 감소시킵니다. 특히, 다양한 데이터셋과 LLM(대형 언어 모델) 플랫폼에 대한 우수한 전이 가능성을 보여줍니다. 이러한 성과는 다중 에이전트 시스템 설계의 새로운 패러다임을 설정하며, 실제 응용분야에서도 뛰어난 효과성을 입증하였습니다.



### Geolog-IA: Conversational System for Academic Theses (https://arxiv.org/abs/2510.02653)
Comments:
          17 pages, in Spanish language

- **What's New**: Geolog-IA는 에콰도르 중앙대학교의 지질학 논문에 대한 자연어 응답 시스템으로, Llama 3.1 및 Gemini 2.5 언어 모델을 활용합니다. 이 시스템은 Retrieval Augmented Generation(RAG) 아키텍처와 SQLite 데이터베이스를 결합하여 정보 검색의 정확성을 높이고, 과거의 지식이나 잘못된 정보로 인한 문제를 극복합니다. Geolog-IA는 높은 BLEU 점수(0.87)를 기록하며, 사용자 친화적인 웹 인터페이스를 제공하여 교사, 학생 및 관리직이 쉽게 상호작용할 수 있도록 도와줍니다.

- **Technical Details**: 이 시스템은 자연어 처리(NLP) 기술을 사용하여 LLM(Long Language Models)과 RAG-SQL의 조합을 통해 정보의 액세스, 분석 및 추출을 가능하게 합니다. RAG 기법은 LLM의 언어 이해 생성 능력과 함께 외부 데이터베이스에서의 정보 검색을 결합하여 정확하고 최신의 정보를 제공합니다. 학습 과정에서 LLM은 고급 신경망 아키텍처를 기반으로 하며, 이러한 시스템은 연구 및 교육의 효율성을 높이는 데 기여합니다.

- **Performance Highlights**: Geolog-IA는 학생들, 교사 및 행정직원에게 신뢰할 수 있는 정보를 제공하여, 학업과 연구 과정에서의 시간을 절약하고 효율성을 개선합니다. 교사들은 이 시스템을 통해 수업 자료를 업데이트하고 학생들의 연구 주제를 더 잘 지도할 수 있습니다. 학생들은 참고 문헌을 검색하는데 드는 노력과 시간을 획기적으로 줄일 수 있으며, 제안된 시스템은 지질학 분야의 교육과 연구에서 매우 중요한 도구로 자리잡을 것입니다.



### Hierarchical Semantic Retrieval with Cobweb (https://arxiv.org/abs/2510.02539)
Comments:
          20 pages, 7 tables, 4 figures

- **What's New**: 본 논문은 Cobweb이라는 계층 인식 프레임워크를 사용하여 문서 검색 방식을 혁신적으로 개선합니다. 기존의 신경망 기반 검색 방법들이 단일 유사도 점수로 문서의 관련성을 평가하는 대신, Cobweb은 문서들을 구조화된 트리 형태로 조직하여 다중 수준의 관련성 신호를 제공합니다. 이를 통해 사용자는 검색 경로를 통해 명확한 해석을 얻을 수 있는 장점을 누립니다.

- **Technical Details**: Cobweb는 문서 임베딩을 계층적으로 조직화하는 방법으로, 내부 노드가 개념 프로토타입 역할을 하여 문서 간의 관련성을 다층적으로 평가합니다. 제안된 두 가지 추론 방법, 즉 일반화된 최적 우선 검색(generalized best-first search)과 경량 경로 합(Path-sum) 랭커를 통해 성능을 높이며, BERT, GPT-2 등 다양한 인코더 및 디코더 임베딩을 활용합니다. 실험의 결과는 Cobweb의 접근 방식이 기존의 밀집 검색 기준과 비슷하거나 더 나은 성능을 나타내며, 임베딩 품질 저하에도 강한 내성을 보임을 입증합니다.

- **Performance Highlights**: 모든 실험에서 Cobweb 기반의 계층 검색 방법은 MS MARCO와 QQP 데이터셋에서 효과적인 평가 결과를 도출했습니다. 특히, 문서 검색은 전통적인 단일 유사도 기반 방법인 dot-product 검색과 비교할 때 우수한 성능을 보여주며, 성능 스케일링 또한 뛰어난 것으로 나타났습니다. 이로 인해 Cobweb은 대규모 문서 검색의 해석 가능성을 제공하는 동시에 효과성과 확장성을 갖춘 솔루션으로 자리매김하게 되었습니다.



### jina-reranker-v3: Last but Not Late Interaction for Listwise Document Reranking (https://arxiv.org/abs/2509.25085)
Comments:
          early draft, CodeIR table needs to be updated (qwen baselines are missing)

- **What's New**: 이번 연구에서 소개된 jina-reranker-v3는 0.6B 파라미터의 다국적 리스트 재정렬 모델로, "last but not late" 상호작용 방식을 도입하였습니다. 기존의 late interaction 모델들과는 달리, 이 모델은 쿼리와 모든 문서 간의 인과적 주의를 적용하여 문서들을 인코딩하기 전에 풍부한 상호작용을 가능하게 하여 컨텍스트 임베딩을 추출할 수 있게 합니다.

- **Technical Details**: jina-reranker-v3는 Qwen3-0.6B 아키텍처를 기반으로 하며, 28개의 transformer 레이어, 1024 개의 hidden 차원, 16 개의 attention 헤드 및 131K token 컨텍스트 용량을 가지고 있습니다. 크로스 문서 상호작용을 가능하게 하는 LBNL 방식을 사용하여, 모든 문서와 쿼리를 동시에 처리하여 의미의 풍부한 임베딩을 생성합니다.

- **Performance Highlights**: 평가 결과 jina-reranker-v3는 BEIR에서 61.94 nDCG@10을 달성하여 이전 모델인 jina-reranker-v2에 비해 4.79% 향상되었습니다. 이 모델은 HotpotQA에서 78.58, FEVER에서 94.01, MIRACL에서 66.83 및 MKQA에서 67.92 Recall@10으로 다국적 성능에서도 경쟁력을 보여주고 있습니다.



New uploads on arXiv(cs.CV)

### LEAML: Label-Efficient Adaptation to Out-of-Distribution Visual Tasks for Multimodal Large Language Models (https://arxiv.org/abs/2510.03232)
- **What's New**: 이 논문에서는 의료 영상과 같은 전문 도메인에서의 OOD(Out-of-Distribution) 작업을 해결하기 위해 LEAML이라는 라벨 효율적인 적응 프레임워크를 제안합니다. LEAML은 부족한 라벨이 있는 VQA(Visual Question Answering) 샘플과 풍부한 라벨이 없는 이미지를 결합하여 도메인 관련의 의사 질문-답변 쌍을 생성합니다. 이 프레임워크는 QA Generator를 활용하여 도메인 특유의 지식을 효과적으로 습득할 수 있도록 설계되었습니다.

- **Technical Details**: LEAML은 Pseudo QA Generation과 OOD VQA Finetuning의 두 단계로 구성되어 있습니다. Pseudo QA Generation에서는 소량의 라벨 데이터로 학습한 QA Generator가 라벨이 없는 이미지에서 QA 쌍을 생성합니다. OOD VQA Finetuning 단계에서는 원본 라벨 샘플과 생성된 QA 쌍을 사용하여 MLLM(Multimodal Large Language Model)을 미세 조정하고, Selective Neuron Distillation을 통해 도메인 관련 지식을 손실 없이 업데이트합니다.

- **Performance Highlights**: 실험 결과, LEAML은 표준 미세 조정 방법들보다 월등히 뛰어난 성능을 발휘했습니다. 특히, 위장 내시경 및 스포츠 VQA 작업에서 LEAML이 저비용으로 효과적인 결과를 도출할 수 있었음을 보여줍니다. 이로 인해 LEAML의 도메인 특화 MLLM 적응 능력이 강조되었습니다.



### Improving GUI Grounding with Explicit Position-to-Coordinate Mapping (https://arxiv.org/abs/2510.03230)
- **What's New**: 최근의 GUI grounding 연구에서는 자연어 지시사항을 정확한 픽셀 좌표로 변환하는 과정이 중요하게 다루어졌습니다. 본 논문은 고해상도 디스플레이에서의 좌표 예측의 신뢰성 문제를 해결하기 위해 RULER(Rotary position-to-pixeL mappER) 토큰과 I-MRoPE(Interleaved Multidimensional Rotary Positional Encoding)를 제안합니다. 이러한 접근을 통해 GUI 자동화의 정확성을 향상시키고 다양한 해상도와 플랫폼에서의 적용 가능성을 확대합니다.

- **Technical Details**: RULER 토큰은 모델 내에 명시적인 좌표 참조 시스템을 구축하여 픽셀 좌표를 직접 인코딩합니다. I-MRoPE는 공간 차원에서 주파수 불균형 문제를 해결하여 높이와 너비 균형 잡힌 공간 표현을 제공합니다. 이 두 가지 혁신은 모델의 정확성을 높이고 좌표 예측의 불안정성을 해결하는 데 기여합니다.

- **Performance Highlights**: 본 연구는 ScreenSpot 및 ScreenSpot-Pro 벤치마크에서의 평가를 통해 GUI grounding 정확도를 크게 향상시켰습니다. 특히, 고해상도 디스플레이에서의 성능이 31.1%에서 37.2%로 증가하는 등 강력한 일반화 능력을 보여주었습니다. RULER 토큰은 추가적인 계산 비용이 거의 없이 8K 디스플레이에서도 효과적입니다.



### MIXER: Mixed Hyperspherical Random Embedding Neural Network for Texture Recognition (https://arxiv.org/abs/2510.03228)
- **What's New**: 본 논문에서는 Mixer라는 새로운 랜덤 신경망을 제안하여 텍스처 표현 학습을 개선하고자 합니다. Mixer는 하이퍼구형 랜덤 임베딩(hyperspherical random embeddings)과 이중 분기 학습 모듈을 활용하여 채널 간 관계를 효과적으로 포착하는 방법을 채택하고 있으며, 새롭게 정의된 최적화 문제를 통해 丰富한 텍스처 표현을 구축할 수 있도록 설계되었습니다.

- **Technical Details**: Mixer는 네 가지 핵심 모듈로 구성되어 있습니다. 첫 번째 모듈인 로컬 패턴 추출기는 각 이미지 채널에서 작은 패치를 독립적으로 추출하여 로컬 텍스처 정보를 포착합니다. 두 번째 모듈인 하이퍼구형 랜덤 프로젝터는 추출된 패치를 랜덤 임베딩으로 인코딩하고 드디어 마지막 모듈인 압축 모듈은 디코더의 학습된 가중치를 압축하여 인식 작업에 유용한 색상-텍스처 표현으로 요약합니다.

- **Performance Highlights**: 실험 결과, 제안된 색상-텍스처 표현이 기존 방법들보다 일관되게 우수한 성능을 보였으며, Outex 데이터셋에서 97% 이상의 정확도를 기록한 유일한 방법이었습니다. 이는 텍스처 인식 분야에서 새로운 이정표가 될 것으로 기대되며, 향후 관련 연구에 대한 기여도를 높일 수 있을 것입니다.



### Test-Time Defense Against Adversarial Attacks via Stochastic Resonance of Latent Ensembles (https://arxiv.org/abs/2510.03224)
- **What's New**: 이 논문은 적대적 공격(adversarial attacks)에 대한 테스트 시간 방어 메커니즘을 제안합니다. 기존의 방법들은 정보 손실(information loss)을 초래할 수 있는 피처 필터링(feature filtering) 또는 스무딩(smoothing)에 의존하는 반면, 저자는 소음(noise)을 소음으로 상쇄하는 방법을 사용하여 강건성을 높이고 정보 손실을 최소화합니다. 이 기법은 입력 이미지에 소규모 변환적 섭동을 추가하고 변환된 피처 임베딩(feature embeddings)을 정렬하여 집계한 후 원본 이미지로 다시 매핑하는 방식을 사용합니다.

- **Technical Details**: 저자들은 스토캐스틱 공진(stochastic resonance) 기술을 활용하여 적대적 섭동의 영향을 제거하는 방법을 제안합니다. 이를 통해 입력 데이터를 대체하거나 임베딩을 평균화하지 않고, 변환된 임베딩을 잠재 공간(latent space)에서 평균화하는 방식으로 적대적 섭동의 영향을 줄입니다. 이 방법은 별도의 네트워크 모듈 없이 다양한 기존 네트워크 아키텍처에 배포될 수 있으며, 특정 공격 유형을 위한 미세 조정(fine-tuning) 없이 동작합니다.

- **Performance Highlights**: 이 방법은 이미지 분류(image classification), 스테레오 매칭(stereo matching), 광학 흐름(optical flow)과 같은 다양한 작업에서 검증되었습니다. 결과적으로, 깨끗한 성능(clean performance)에 비해 이미지를 분류하는 동안 최대 68.1%의 정확도 손실을 회복하고, 스테레오 매칭에서는 71.9%, 광학 흐름에서는 29.2%를 회복하였습니다. 이 방법은 기존의 어떤 기법보다도 강력한 강건성을 제공하며, 밀집 예측(dense prediction) 작업에 대한 일반적인 테스트 시간 방어(Generic test-time defense)를 수립하였습니다.



### MonSTeR: a Unified Model for Motion, Scene, Text Retrieva (https://arxiv.org/abs/2510.03200)
- **What's New**: 이번 연구에서는 MonSTeR라는 최초의 MOtioN-Scene-TExt Retrieval 모델을 소개합니다. 이는 skeletal movement(움직임), intention(의도), scene(장면) 간의 정합성을 평가할 수 있는 도구를 제공하고자 합니다. MonSTeR는 unimodal(단일 모달) 및 cross-modal(교차 모달) 표현을 활용하여 통합된 latent space(잠재 공간)를 구성합니다.

- **Technical Details**: MonSTeR는 고차원 상호작용을 모델링하여 텍스트, 움직임 및 장면 간의 관계를 효과적으로 포착합니다. 이는 variational encoders를 통해 인코딩된 unimodal 표현을 그래프의 노드로 사용하고, cross-modal encoders를 통해 쌍의 모달리티를 엮는 구조로 이루어져 있습니다. 특히, 임베딩은 scene-text, motion-text 및 scene-motion 간의 관계를 정렬하여 모든 모달리티 간의 독특한 상호작용을 촉진합니다.

- **Performance Highlights**: MonSTeR는 기존의 trimodal(세 가지 모달) 모델에 비해 우수한 성능을 보이며, 사용자 연구를 통해 평가 점수가 인간의 선호와 일치함을 확인하였습니다. 또한, Motion Captioning과 zero-shot in-Scene Object Placement 작업에서 유연성을 입증하였으며, 이는 MonSTeR의 다양한 활용 가능성을 시사합니다.



### Memory Forcing: Spatio-Temporal Memory for Consistent Scene Generation on Minecraf (https://arxiv.org/abs/2510.03198)
Comments:
          19 pages, 8 figures

- **What's New**: 본 논문은 Memory Forcing이라는 새로운 학습 프레임워크를 소개하며, 이 프레임워크는 시공간 메모리를 효율적으로 결합하여 비디오 생성에서 장기적인 공간 일관성을 달성하는 방법을 제시합니다. 이는 Minecraft 같은 게임 세계를 모델링하는 데 특히 유용하며, 새로운 장면 탐색과 재방문 시의 일관성을 동시에 유지하려는 노력을 포함하고 있습니다. 모델은 제한된 메모리 내에서 과거 정보의 압축과 우선순위 설정이 필수적입니다.

- **Technical Details**: Memory Forcing는 Hybrid Training과 Chained Forward Training(CFT) 전략을 통해 시간 메모리와 공간 메모리 사이의 균형을 촉진합니다. Hybrid Training은 플레이어의 탐색과 재방문 처리에 최적화된 데이터 분포를 사용하며, CFT는 모델의 예측을 통해 행동에 대한 변화를 더욱 확장합니다. 또한 Geometry-indexed Spatial Memory를 통해 현재 보이는 3D 포인트를 기반으로 과거 프레임을 효율적으로 검색합니다.

- **Performance Highlights**: 실험 결과, Memory Forcing는 다양한 환경에서 높은 공간 일관성과 생성 품질을 달성하며, 기존 메모리 모델 대비 7.3배 빠른 검색 속도와 98.2% 적은 메모리 저장량을 보여주었습니다. 이는 생성 품질과 계산 효율성 간의 절충을 효과적으로 해결한 것을 나타냅니다. 전반적으로, 이 방법은 Minecraft와 같은 복잡한 게임 환경에서 경쟁력 있는 성능을 유지하도록 설계되었습니다.



### Product-Quantised Image Representation for High-Quality Image Synthesis (https://arxiv.org/abs/2510.03191)
- **What's New**: 본 논문에서는 고해상도 이미지 생성에 대한 검증된 기법인 PQGAN을 소개합니다. PQGAN은 이미 잘 알려진 VQGAN의 벡터 양자화(VQ) 프레임워크에 PQ(Product Quantisation)를 통합하여 발전된 이미지 자동 부호화기입니다. 이 방법은 이전 기법에 비해 복원 성능이 크게 향상되어 PSNR 점수 37dB를 기록했으며, FID, LPIPS, CMMD 점수를 최대 96%까지 줄일 수 있었습니다.

- **Technical Details**: PQGAN은 코드북 크기, 임베딩 차원, 서브스페이스 분할 간의 상호작용을 철저히 분석하여 성공을 달성하였습니다. VQ와 PQ는 임베딩 차원을 스케일링할 때 반대의 성능 경향을 보이며, 이러한 분석을 통해 최적의 하이퍼파라미터 선택에 대한 가이드를 제공합니다. PQ는 압축 및 생성 모델에서의 코드북 인덱스를 분리하여 학습 신호의 밀도를 향상시키도록 설계되었습니다.

- **Performance Highlights**: PQGAN은 512개의 코드북 항목만을 사용하여 ImageNet에서 FID를 0.036으로 낮추고 PSNR을 37.4 dB로 높이며, 기존 연속 기반 방법을 초월하는 성능을 보였습니다. 또한, 사전 학습된 확산(diffusion) 모델에 무리 없이 통합되어 더 빠른 생성 혹은 동일 비용으로 해상도를 두 배로 증가시킬 수 있는 가능성을 보여주었습니다. 이를 통해 PQ가 이미지 합성에서 불연속 잠재 표현의 강력한 확장임을 입증하였습니다.



### Dynamic Prompt Generation for Interactive 3D Medical Image Segmentation Training (https://arxiv.org/abs/2510.03189)
- **What's New**: 이 논문에서는 사용자 프롬프트에 기반하여 예측을 반복적으로 정제할 수 있는 효율적인 모델을 요구하는 대화형 3D 생물 의학 영상 분할을 위한 학습 전략을 제안합니다. 기존 모델들이 볼륨 인식 부족이나 제한된 대화형 기능으로 어려움을 겪고 있는 반면, 본 연구는 동적 볼륨 프롬프트 생성과 콘텐츠 인식 적응형 크롭을 조합한 방법론을 소개하여 성능을 향상시킵니다.

- **Technical Details**: 제안된 방법은 nnInteractive 모델을 기반으로 하여 3D Residual Encoder U-Net 아키텍처를 사용합니다. 이 아키텍처는 최적의 응답을 위해 해부학적 맥락에 따라 모델의 시야 크기를 적응시키고, 경량의 3D 분할을 위한 프롬프트(예: 바운딩 박스와 클릭 프롬프트)를 입력 채널로 결합하여 더욱 효과적으로 학습합니다. 또한, 원활한 훈련을 위해 서로 다른 설정을 확률적으로 선택하여 GPU 메모리 사용량을 최적화합니다.

- **Performance Highlights**: 경쟁인 Foundation Models for Interactive 3D Biomedical Image Segmentation에서 본 모델은 평균 최종 Dice 점수 0.6385, 정규화된 표면 거리 0.6614로 강력한 성능을 입증하였습니다. 특히, Dice와 NSD의 구간 아래 면적(AUC) 측정치에서 각각 2.4799와 2.5671로 나타났습니다. 이러한 성과는 대화형 분할에서 사용자 피드백을 효과적으로 수용한 결과로 평가됩니다.



### ROGR: Relightable 3D Objects using Generative Relighting (https://arxiv.org/abs/2510.03163)
Comments:
          NeurIPS 2025 Spotlight. Project page: this https URL

- **What's New**: ROGR라는 새로운 접근 방식을 소개합니다. 이는 여러 시점에서 캡처된 물체의 조명 가능한 3D 모델을 재구성하며, 물체를 새로운 환경 조명 아래 배치했을 때의 효과를 시뮬레이션하는 생성 조명 모델(generative relighting model)을 기반으로 합니다. 이 방법은 다양한 조명 환경에서 물체의 외관을 샘플링하여 다수의 조명 상태에서 훈련된 Neural Radiance Field(NeRF)로 활용하며, 이는 임의의 환경 조명 하에 물체 외관을 출력할 수 있습니다.

- **Technical Details**: 조명 조건에 최적화된 NeRF는 일반적인 조명 효과와 반사(gloss) 효과를 별도로 인코딩하는 새로운 이중 브랜치 아키텍처를 사용합니다. 최적화된 조명 조건부 NeRF는 조명 최적화 또는 빛 전송 시뮬레이션 없이도 임의의 환경 맵에서 효율적인 피드포워드 재조명(relighting)을 가능하게 합니다. 우리는 이 방법을 TensoIR와 Stanford-ORB 데이터셋에서 평가하여 대부분의 메트릭에서 기존 최첨단 기술을 개선하였으며, 실제 객체 캡처 사례에서도 적용을 보여줍니다.

- **Performance Highlights**: 우리는 3D 재조명(task of 3D relighting) 작업에서 Synthetic 및 실제 벤치마크를 통해 방법의 효과성을 평가합니다. 우리의 접근법은 최첨단 결과를 도출하며, 이전 작업에 비해 시험 시간 성능(test-time performance)이 크게 향상되었습니다. 이러한 성능 향상은 일반화 가능한 피드포워드 재조명 NeRF 덕분입니다.



### UniShield: An Adaptive Multi-Agent Framework for Unified Forgery Image Detection and Localization (https://arxiv.org/abs/2510.03161)
- **What's New**: 본 연구는 이미지 생성 기술의 빠른 발전에 따른 합성 이미지의 현실성 증가가 정보 무결성과 사회적 안전성에 미치는 영향을 다루고 있습니다. 이를 해결하기 위한 새로운 시스템인 UniShield를 제안하여 이미지 위조 탐지 및 위치 확인(Forgery Image Detection and Localization, FIDL)의 중요성을 강조합니다. UniShield는 다양한 도메인에서 이미지 위변조를 탐지하고 국지화할 수 있는 멀티 에이전트 기반의 통합 시스템입니다.

- **Technical Details**: UniShield는 두 가지 주요 에이전트, 즉 인식 에이전트(perception agent)와 탐지 에이전트(detection agent)를 통합하여 작동합니다. 인식 에이전트는 이미지 특징을 분석하여 적합한 탐지 모델을 동적으로 선택하고, 탐지 에이전트는 다양한 전문 탐지기를 통합하여 통합 프레임워크를 구축합니다. 이 시스템은 AI 생성 이미지, DeepFake, 문서 조작 등 다양한 이미지 조작 형태를 처리할 수 있도록 설계되었습니다.

- **Performance Highlights**: UniShield는 광범위한 실험을 통해 기존 통합 접근 방식 및 도메인 특정 탐지 방법을 초월하는 최첨단 결과를 보여주었습니다. 이는 UniShield의 우수한 실용성, 적응성, 그리고 확장성을 강조하며, 실제 적용 가능성을 높이고 사회적 위험을 줄이는 데 기여할 수 있을 것으로 기대됩니다.



### SpineBench: A Clinically Salient, Level-Aware Benchmark Powered by the SpineMed-450k Corpus (https://arxiv.org/abs/2510.03160)
- **What's New**: 이번 논문에서는 619억의 사람들이 영향을 받는 척추 질환에 대한 AI 기반 진단의 한계를 극복하기 위한 SpineMed라는 새로운 생태계를 소개합니다. SpineMed-450k는 45만 건 이상의 지침 인스턴스를 포함하는 대규모 데이터셋을 제공하며, X-ray, CT, MRI 등의 이미징 기법에서 특정 척추 레벨에 대한 의사 결정을 지원합니다. SpineBench는 임상적으로 검증된 평가 프레임워크를 통해 AI 모델의 성능을 평가하는 데 도움을 줍니다.

- **Technical Details**: SpineMed-450k 데이터셋은 임상 의사의 피드백을 바탕으로 수집된 다양한 자료를 통해 구성되었습니다. 이 데이터는 교과서, 수술 지침서, 전문가 합의 등을 통해 고품질, 특화된 정보를 제공합니다. 데이터의 질과 추적 가능성을 보장하기 위해 'clinician-in-the-loop' 파이프라인과 두 단계의 LLM 생성 방법을 사용하여 데이터를 정제했습니다.

- **Performance Highlights**: 여러 최신 대형 비전-언어 모델(LVLM)을 SpineBench에서 평가한 결과, 세밀하고 레벨 별 진단에서 체계적인 약점이 발견되었습니다. 반면, SpineMed-450k로 미세 조정된 SpineGPT 모델은 모든 작업에서 일관된 개선을 보여주었으며, 이 모델의 결과는 임상 평가를 통해 진단의 명확성과 실용성을 확인했습니다.



### ReeMark: Reeb Graphs for Simulating Patterns of Life in Spatiotemporal Trajectories (https://arxiv.org/abs/2510.03152)
Comments:
          15 pages, 3 figures, 2 algorithms, 1 table

- **What's New**: 이번 연구에서는 Markovian Reeb Graphs라는 새로운 프레임워크를 소개하여 인간의 이동성( mobility)을 정확하게 모델링합니다. 이 프레임워크는 기본 데이터에서 학습된 생활 패턴(Patterns of Life, PoLs)을 유지하면서 시공간 경로(spatiotemporal trajectories)를 시뮬레이션할 수 있도록 설계되었습니다. 특히, 개인 및 집단 수준의 이동 구조를 결합하여 확률적(topological model) 모델 내에서 현실적인 미래 경로를 생성합니다.

- **Technical Details**: 우리는 Urban Anomalies 데이터셋의 애틀랜타와 베를린 하위 집합을 사용하여 새로운 방법의 유효성을 평가합니다. 평가 과정에서 Jensen-Shannon Divergence (JSD)를 활용하여 인구 및 에이전트(level) 지표에서의 성능을 비교했습니다. 이 방법은 데이터와 계산(compute) 측면에서 효율성을 유지하면서도 높은 충실도를 달성하는 것으로 나타났습니다.

- **Performance Highlights**: Markovian Reeb Graphs는 다양한 도시 환경에 폭넓게 적용될 수 있는 확장 가능한 경로 시뮬레이션 프레임워크로 자리매김하고 있습니다. 일상생활에서의 일관성과 변동성을 동시에 포착하는 미래 경로 생성의 가능성을 보여줍니다. 이 연구 결과는 도시 계획(urban planning), 전염병학(epidemiology), 교통 관리(traffic management) 분야에서 널리 활용될 수 있습니다.



### Mask2IV: Interaction-Centric Video Generation via Mask Trajectories (https://arxiv.org/abs/2510.03135)
Comments:
          Project page: this https URL

- **What's New**: 대화형 비디오 생성을 위한 새롭고 혁신적인 프레임워크인 Mask2IV가 소개되었습니다. 이 프레임워크는 액터와 객체의 움직임 경로를 예측한 후, 이 경로에 의해 조정된 비디오를 생성하는 두 단계의 분리된 파이프라인을 채택합니다. 또한, 사용자가 상호작용할 객체를 지정하고 동작 경로를 조절할 수 있도록 직관적인 제어를 지원합니다.

- **Technical Details**: Mask2IV는 기존의 핸드 마스크 시퀀스를 필요로 하지 않으며, 대신 초기 이미지와 객체 마스크를 기반으로 상호작용 경로를 예측합니다. 이 모델은 인간-객체 및 로봇-객체 간의 상호작용을 효과적으로 모델링하여, 복잡한 환경에서도 높은 성능을 보여줍니다. 또한, Mask2IV는 행동 설명(action descriptions)이나 공간 위치 단서(spatial position cues)를 통해 상호작용 경로 생성을 안내합니다.

- **Performance Highlights**: 실험 결과, Mask2IV는 기존 모델에 비해 뛰어난 시각적 사실성과 제어 가능성을 보여주었습니다. 연구팀은 HOI4D와 BridgeDataV2라는 두 개의 데이터셋에 기반한 벤치마크를 구축하여 이 모델을 평가하였습니다. 결과적으로 Mask2IV는 인체 객체 및 로봇 조작 시나리오에서의 시각적 충실도와 조작성을 향상시켰습니다.



### HAVIR: HierArchical Vision to Image Reconstruction using CLIP-Guided Versatile Diffusion (https://arxiv.org/abs/2510.03122)
- **What's New**: 인간의 뇌에서의 시각 정보 복원은 신경과학(Neuroscience)과 컴퓨터 비전(Computer Vision) 간의 통합을 촉진합니다. 하지만 복잡한 시각 자극을 정확하게 복원하는 데 어려움을 겪고 있습니다. HAVIR 모델은 시각 피질의 계층적 표현 이론에서 영감을 받아, 구조적 정보와 의미적 정보를 각각 독립적으로 추출합니다.

- **Technical Details**: HAVIR 모델은 두 가지 주요 구성 요소, 즉 Structural Generator와 Semantic Extractor로 이루어져 있습니다. Structural Generator는 공간 처리(Spatial Processing) 복셀에서 구조 정보를 추출하고, 이 정보를 latent diffusion priors로 변환합니다. Semantic Extractor는 의미 처리(Semantic Processing) 복셀을 CLIP 임베딩(CLIP Embeddings)으로 변환하여 두 영역의 정보를 통합합니다.

- **Performance Highlights**: 실험 결과, HAVIR는 복잡한 장면에서도 구조적 및 의미적 품질을 향상시켜, 기존 모델들과 비교하여 성능이 뛰어난 것으로 나타났습니다. 이는 HAVIR 모델이 구조적 피처와 의미적 정보를 분리하여 최종 이미지를 생성하는 "divide and conquer" 전략 덕분입니다.



### Taming Text-to-Sounding Video Generation via Advanced Modality Condition and Interaction (https://arxiv.org/abs/2510.03117)
- **What's New**: 이 연구는 Text-to-Sounding-Video (T2SV) 생성이라는 도전적이지만 유망한 작업에 초점을 맞추고 있습니다. T2SV는 텍스트 조건으로부터 비디오와 동기화된 오디오를 생성하는 것을 목표로 하며, 텍스트와 두 가지 모달리티의 정렬을 보장해야 합니다. 제안된 Hierarchical Visual-Grounded Captioning (HVGC) 프레임워크와 Dual CrossAttention (DCA) 메커니즘을 통해 이러한 문제를 해결하고 있습니다.

- **Technical Details**: HVGC 프레임워크는 비디오 캡션과 오디오 캡션을 생성하여 모달 간 간섭을 제거합니다. BridgeDiT라는 새로운 이중 타워 확산 변환기를 도입하여 정보의 대칭적이고 쌍방향 교환을 가능하게 합니다. 이 구조는 동기화된 비디오 생성을 위한 모달리티 순수 텍스트 캡션을 제공하며, 효과적인 교차 모달 피처 인터랙션을 보장합니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안된 방법이 대부분의 지표에서 최신 기술 결과를 달성했음을 입증했습니다. 또한, 인간 평가와 함께 수행된 포괄적인 아블레이션 연구는 제안된 기여의 효과를 검증하며, T2SV 작업의 미래에 대한 중요한 통찰력을 제공합니다. 모든 코드와 체크포인트는 공개될 예정입니다.



### GeoComplete: Geometry-Aware Diffusion for Reference-Driven Image Completion (https://arxiv.org/abs/2510.03110)
Comments:
          Accepted by NeurIPS 2025. Project page: this https URL

- **What's New**: GeoComplete는 더 정확하고 일관된 이미지 보완을 위해 기하학적 일관성을 강화하는 새로운 프레임워크입니다. 기존의 이미지 기반 접근 방식을 넘어 3D 구조적 지침을 명시적으로 통합함으로써, 문제 해결에서의 강점을 보여줍니다. 이 프레임워크는 두 가지 주요 아이디어인 점 구름(point cloud) 기반의 확산(difusion) 과정 조정 및 타겟 인식 마스킹(target-aware masking)을 통해 차별화되어 있습니다.

- **Technical Details**: GeoComplete는 두 가지 주요 구성 요소로 이루어집니다: (1) Visual Geometry Grounded Transformer (VGGT)와 (2) Language Segment Anything (LangSAM)입니다. VGGT는 카메라 매개변수와 깊이 맵을 한 번의 전방 패스로 예측하여 복잡한 장면에서도 정확한 기하학적 추정을 가능하게 합니다. LangSAM은 동적 지역을 텍스트 프롬프트로 분할하고, 이로 인해 좀 더 신뢰할 수 있는 3D 속성 예측을 지원합니다.

- **Performance Highlights**: GeoComplete는 기존 방법들보다 17.1 PSNR 향상을 이루어내며, 기하학적 정확성을 대폭 개선하였습니다. 또한, 실험 결과 다양한 복잡한 장면에서도 높은 시각적 품질을 유지하며 신뢰성 있는 결과를 생성함을 보여주었습니다. 이로 인해 GeoComplete는 기하학적 조건이 있는 이미지 보완을 위한 강력한 솔루션으로 자리 잡을 가능성이 높습니다.



### Geometry Meets Vision: Revisiting Pretrained Semantics in Distilled Fields (https://arxiv.org/abs/2510.03104)
- **What's New**: 이 논문에서는 기하 구조 기반의 특성을 활용한 새로운 프레임워크 SPINE을 제안합니다. SPINE은 기존의 의미 통합에 대한 한계를 극복하며, 방사 필드(radiance fields)의 정확한 역전(inversion)을 위해 두 가지 주요 구성 요소를 포함합니다. 또한, 기존 복잡한 초기 추정 없이도 방사 필드를 역전할 수 있는 방법을 모색합니다.

- **Technical Details**: 이 연구는 기하학적으로 기반한 의미 특징이 내재된 필드(distilled fields)에서 가지는 잠재적 이점을 조사합니다. 구체적으로, 영상을 기반으로 한 심층 학습 모델에서 추출한 기하학 특징(geometry-aware semantic features)을 검토하고, 이를 통해 구조적 세부사항의 질을 비교합니다. 그러나 의미 제 위치(localization)에서 기하학 기반 특징이 유의미한 개선을 가져오지 못하는 것으로 확인되었습니다.

- **Performance Highlights**: 기하학 기반의 특성을 사용할 경우 포즈 추정(pose estimation) 정확도가 오히려 감소한다는 놀라운 결과를 발견하였습니다. 연구 결과는 시각적 특성만으로도 다양한 후속 작업(downstream tasks)에 대해 더 큰 유연성을 제공할 수 있음을 시사합니다. 이러한 결과는 또한, 사전 학습된 의미 특징의 성능을 증대시키기 위해 기하학 기반의 효과적인 전략 연구의 필요성을 강조합니다.



### Latent Diffusion Unlearning: Protecting Against Unauthorized Personalization Through Trajectory Shifted Perturbations (https://arxiv.org/abs/2510.03089)
- **What's New**: 이 논문에서는 텍스트-이미지 diffusion 모델의 개인화 기술이 데이터 프라이버시와 지적 재산권 보호에 대한 우려를 초래하고 있다는 점을 강조하고 있습니다. 이러한 무단 사용과 모델 복제를 방지하기 위해 'unlearnable' (무학습 가능) 훈련 샘플을 생성하는 방법을 제안하며, 이는 이미지 오염 기법을 활용하여 이루어집니다. 기존의 방법보다 더 높은 시각적 충실도를 유지하는 새로운 차원 변형 방식을 제안하여, 이를 통해 다운스트림 생성 모델에 대한 저항력을 갖춘 이미지를 생성합니다.

- **Technical Details**: 제안된 방법은 latent space (잠재 공간)에서 작동하며, 이는 기존의 픽셀 수준의 조작 방식과는 다릅니다. 모델은 denoising과 inversion을 반복하며, denoising 경로의 시작점을 변경하여 perturbation을 생성합니다. 이 접근법은 Latent Diffusion Models (LDMs) 내에서 적용되어, 비가시적으로 무단 모델 적응에 대한 방어를 통합합니다. 우리의 방법은 다양한 벤치마크 데이터셋에서 검증되었으며, 기존 공격에 대한 견고성을 유지합니다.

- **Performance Highlights**: 연구 결과, 우리의 방법은 기존의 방법들에 비해 시각적으로 더 높은 충실도와 견고성을 달성했습니다. perceptual metrics (지각 메트릭스)인 PSNR, SSIM, FID에서 각각 약 8%에서 10% 개선을 보였고, 다섯 가지 적대적 설정에서 평균적으로 약 10% 향상된 견고성을 입증했습니다. 이러한 성과는 민감한 데이터 보호에 효율적인 방법임을 시사합니다.



### What Drives Compositional Generalization in Visual Generative Models? (https://arxiv.org/abs/2510.03075)
- **What's New**: 본 연구는 시각 생성 모델에서 조합 일반화(compositional generalization)에 영향을 미치는 다양한 디자인 선택이 긍정적 또는 부정적인 방식으로 작용하는지를 체계적으로 분석합니다. 특히, 훈련 목표(training objective)가 이산(discrete) 또는 연속(continuous) 분포에 작용하는지와 훈련 동안 구성 요소 개념에 대한 조건(conditioning)이 얼마나 정보를 제공하는지가 주요하게 작용함을 발견했습니다. 이후, MaskGIT 모델의 이산 손실(discrete loss)을 보조 연속(JEPA-based) 목표로 완화함으로써 조합 성능을 향상시키는 방법을 제안합니다.

- **Technical Details**: 연구에서는 세 가지 주요 구성요소 즉, Tokenizer, Generative model, Conditioning signal을 통해 현대 시각 생성 모델을 분해하고 각 요소가 조합 일반화에 미치는 영향을 살펴봅니다. 각 모델의 아키텍처와 교육 디자인의 선택에 따라 조합 일반화가 어떻게 변화하는지를 체계적으로 비교하기 위해 세 가지 요소에 대해 통제된 실험을 실시하였습니다. 실험 결과, 연속 분포로 훈련된 모델이 이산 분포로 훈련된 모델보다 조합 능력이 더 뛰어난 것으로 나타났습니다.

- **Performance Highlights**: MaskGIT의 분류적 목표를 Joint Embedding Predictive Architecture (JEPA)로 보강함으로써 조합 성능을 명확히 개선할 수 있음을 보여주었습니다. 동시에 JEPA로 훈련된 모델은 더 분리된 중간 표현(intermediate representations)을 제공하고, 이는 예측 기반의 연속 목표가 이산 생성 모델의 내부 구조를 조합성을 유지하는 방향으로 형성할 수 있음을 시사합니다. 이러한 결과는 생성 모델의 조합 및 분해 능력을 향상시키는 방법을 제시하며, 다양한 아키텍처와 훈련 디자인 선택이 생성 품질에 미치는 영향을 강조합니다.



### InsideOut: An EfficientNetV2-S Based Deep Learning Framework for Robust Multi-Class Facial Emotion Recognition (https://arxiv.org/abs/2510.03066)
- **What's New**: 이 논문은 Facial Emotion Recognition (FER) 분야에서 새로운 프레임워크인 InsideOut를 제안합니다. InsideOut는 EfficientNetV2-S를 기반으로 하여 transfer learning, 강력한 데이터 증강(data augmentation), 그리고 imbalance-aware optimization을 결합하여 FER 성능을 향상시킵니다. 이 접근 방식은 데이터셋의 불균형 문제를 다루면서도 가진 성능을 인정받는 결과를 보여줍니다.

- **Technical Details**: InsideOut는 FER2013 데이터셋에서 학습되며, 모든 이미지는 224×224의 크기로 조정되고 ImageNet 통계로 정규화됩니다. 이 프레임워크는 class-weighted cross-entropy 손실 함수를 통합하여 데이터셋에서 소수 감정인 Disgust와 Fear의 인식을 개선하였습니다. 또한 강화 학습을 통해 의사결정 과정을 면밀히 분석하는 다양한 시각적 도구를 제시합니다.

- **Performance Highlights**: InsideOut는 FER2013에서 62.8%의 정확도와 0.590의 macro-averaged F1-score를 달성하였으며, 이는 전통적인 CNN 기반 시스템에 비해 경쟁력 있는 성능을 보여줍니다. 연구 결과, Happy 감정은 가장 높은 인식을 보여주었고, 소수 감정의 인식을 크게 향상시켰습니다. FRER2013에 대한 성능 평가 결과는 모델이 강력한 일반화 능력을 가지고 있음을 확인시켜 줍니다.



### When and Where do Events Switch in Multi-Event Video Generation? (https://arxiv.org/abs/2510.03049)
Comments:
          Work in Progress. Accepted to ICCV2025 @ LongVid-Foundations

- **What's New**: 본 논문에서는 텍스트-비디오(T2V) 생성 분야에서 다중 이벤트 전환을 제어하는 메커니즘을 연구합니다. 특히, 생성 과정에서 이벤트 프롬프트를 얼마나 일찍, 그리고 모델의 어느 층에서 주입해야 하는지를 탐구합니다. 이를 위해 MEve라는 새로운 프롬프트 세트를 소개하여 다중 이벤트 T2V 모델의 평가를 체계적으로 진행합니다. 다양한 모델에서 초반의 개입과 블록 별 모델 레이어의 중요성이 드러났습니다.

- **Technical Details**: 연구는 확산 모델(diffusion model)을 기반으로 하며, 두 개의 이벤트를 연결하는 방식으로 프롬프트를 설정합니다. 이벤트 전환의 적절한 타이밍(denoising steps)과 깊이(model layer depth)에 따라 프롬프트를 조건화했습니다. 연구 결과, 초기 30%의 denoising 단계에서 이벤트 프롬프트를 노출하는 것이 중요하며, 얕은 블록이 전반적인 의미(global semantics)에 영향을 미친다는 사실이 밝혀졌습니다.

- **Performance Highlights**: 실험을 통해 CogVideo와 OpenSora 두 모델 패밀리의 성능을 평가했습니다. MEve 데이터셋을 사용하여 다중 이벤트 생성을 검증했으며, 모델들은 본래의 프레임 수에 맞춰 비디오를 생성하고 고유한 성능에 따라 프레임을 할당했습니다. 이 연구는 다중 이벤트 조건부 생성의 가능성을 강조하며, 향후 연구 방향에 기여할 수 있는 중요한 인사이트를 제공합니다.



### PocketSR: The Super-Resolution Expert in Your Pocket Mobiles (https://arxiv.org/abs/2510.03012)
- **What's New**: 이번 연구에서는 Real-world image super-resolution (RealSR)를 위한 경량 모델 PocketSR을 소개합니다. PocketSR은 고해상도 이미지 복원을 위한 효율적인 방법론으로, 97.5%의 매개변수를 줄인 LiteED를 통해 높은 신뢰성을 유지하면서도 놀라운 처리 속도를 자랑합니다. 또한, U-Net의 온라인 온도 조절 가지치기(online annealing pruning) 기법과 다층 피처 증류(multi-layer feature distillation) 손실을 통해 지식을 효율적으로 전달합니다.

- **Technical Details**: PocketSR은 146M 매개변수를 가진 초경량의 단일 단계 모델입니다. 이 모델은 0.8초 만에 4K 이미지를 처리할 수 있는 뛰어난 속도로, 기존의 멀티 스텝 모델들과 비교하여 훈련 효율성을 크게 개선했습니다. LiteED 아키텍처는 다양한 고차원 피쳐 채널을 통합하여 U-Net의 입력을 보강하고, 효과적인 구조 최적화를 통해 경량화에 성공했습니다.

- **Performance Highlights**: PocketSR은 기존 최첨단 방식과 유사한 성능을 보여주며, 속도와 효율성 모두를 만족시키는 실용적인 솔루션입니다. 본 연구에서 제시한 기법들은 모델 경량화와 함께 성능을 유지하며, 실제 애플리케이션에서 요구되는 요구사항을 충족할 수 있음을 보여줍니다. 실질적으로 PocketSR은 최신 기술에 기반한 한 단계 확철(SR) 모델로서, 경량화와 성능의 균형을 이룹니다.



### Not every day is a sunny day: Synthetic cloud injection for deep land cover segmentation robustness evaluation across data sources (https://arxiv.org/abs/2510.03006)
- **What's New**: 본 연구는 열대 지역에서 일반적으로 발생하는 구름으로 인해 제한되는 레이블이 있는 위성 데이터를 사용한 지도 학습 기반의 토지 피복 의미 분할(Semantic Segmentation)을 개선하기 위한 방법을 제안합니다. 구름 침투 알고리즘(cloud injection algorithm)을 개발하여 Sentinel-1 레이더 데이터로 구름에 의해 가려진 Optical 데이터를 보완하는 방안을 모색합니다. 또한, Normalized Difference Indices(NDIs)를 최종 디코딩 단계에 주입하여 다운샘플링 과정에서 손실되는 공간 및 스펙트럼 세부 정보의 문제를 해결하려고 합니다.

- **Technical Details**: High-resolution LCS(land cover segmentation)는 심층 신경망을 통해 위성 이미지를 픽셀 수준에서 분류합니다. 본 연구에서는 Optical 데이터와 SAR 데이터가 포함된 Segmentation 네트워크의 성능을 비교하였으며, NDIs를 디코더의 마지막 단계에 주입하여 공간 특성의 복구를 강화하는 방법을 제시합니다. 또한, DFC2020 데이터셋을 사용하여 네트워크를 훈련하고, 파라미터 재학습을 통해 NDIs를 결합하여 최종 LCS 수행을 개선하는 전략을 채택하였습니다.

- **Performance Highlights**: 사전 훈련된 네트워크에서 NDIs를 결합한 결과, U-Net 기준으로 1.99%, DeepLabV3 기준으로 2.78%의 성능 개선이 있었습니다. 구름이 있는 조건에서 SAR 데이터와 Optical 데이터를 결합했을 때 모든 모델에서 기존 Optical 데이터 단독 사용보다 유의미한 성능 향상을 보였습니다. 이는 열악한 대기 조건에서도 레이더와 옵티컬 데이터의 융합이 효과적임을 보여줍니다.



### Towards Scalable and Consistent 3D Editing (https://arxiv.org/abs/2510.02994)
- **What's New**: 이 논문에서는 3D 편집 작업에서의 다양한 도전 과제를 해결하기 위한 두 가지 주요 기여를 소개합니다. 첫째, 3DEditVerse라는 대규모 3D 편집 벤치마크를 제안하여 116,309개의 고품질 훈련 쌍과 1,500개의 테스트 쌍을 포함합니다. 둘째, 3D 구조를 보존하는 조건부 변환기인 3DEditFormer를 개발하여 고급 편집을 위한 정밀성과 일관성을 제공합니다.

- **Technical Details**: 3DEditVerse는 포즈 기반의 기하학적 편집과 외관 편집으로 구성된 보완적인 파이프라인을 통해 제작되었습니다. 3DEditFormer는 다중 주의 메커니즘과 시간 적응형 게이팅을 통해 구조를 유지하면서 정밀한 수정이 가능합니다. 이러한 조합은 사용자의 의도에 충실한 고품질 3D 편집을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 기존 최첨단 모델들보다 정량적 및 정성적으로 더 나은 성능을 보여주었으며, 3D 편집의 새로운 기준을 설정하였습니다. 3DEditFormer는 보조 3D 마스크 없이 평균 13% 향상된 3D 메트릭을 기록했으며, 이는 더 높은 충실도와 실용성을 입증합니다.



### TIT-Score: Evaluating Long-Prompt Based Text-to-Image Alignment via Text-to-Image-to-Text Consistency (https://arxiv.org/abs/2510.02987)
- **What's New**: 최근 대형 멀티모달 모델(LMM)의 급속한 발전으로 텍스트-이미지 변환(T2I) 모델들은 질 높은 이미지를 생성하고 간단한 프롬프트에 대해 잘 정렬하는 성능을 보여줍니다. 그러나 긴 프롬프트에 대한 이해와 실제 실행이 부족하여 일관되지 않은 생성이 이루어지는 문제가 지속되고 있습니다. 이를 해결하기 위한 연구로 LPG-Bench라는 포괄적인 벤치마크가 도입되었습니다.

- **Technical Details**: LPG-Bench는 평균 길이 250단어가 넘는 200개의 정밀하게 제작된 프롬프트로 구성되어 있으며, 이를 바탕으로 13개의 최첨단 모델을 사용하여 2,600개의 이미지를 생성했습니다. 또한, 새로운 제로샷 평가 지표인 TIT가 도입되어, T2I 모델이 긴 프롬프트에 따라 얼마나 잘 생성하였는지를 측정하여 기존 지표들과의 차이를 명확히 합니다. TIT는 이미지에 대한 설명의 일관성을 정량화하여 T2I 모델의 정렬 능력을 평가합니다.

- **Performance Highlights**: LPG-Bench와 TIT 방법론의 사용을 통해, 기존 T2I 평가 메트릭이 긴 프롬프트에 대해 불일치와 약점을 보이는 것을 확인했습니다. 특히 TIT-Score-LLM은 7.31%의 절대 정확도 향상을 달성하면서 인간의 판단과 더 잘 일치하는 평가를 제공합니다. 이러한 결과는 T2I 모델의 긴 텍스트 이해를 위한 새로운 기준을 제시하며, 기존 모델의 핵심 한계를 명확히 평가할 수 있는 토대를 마련합니다.



### Flip Distribution Alignment VAE for Multi-Phase MRI Synthesis (https://arxiv.org/abs/2510.02970)
Comments:
          This paper has been early accept by MICCAI 2025

- **What's New**: 이 논문에서는 다중 위상 대조 강화(CE) MRI 합성을 위한 경량화된 Feature-decoupled 변분 오토인코더(FDA-VAE)를 제안합니다. 기존의 방법들이 깊은 오토인코더 기반으로 낮은 매개변수 효율성을 보인 반면, FDA-VAE는 두 개의 대칭적인 잠재 분포를 통해 공유된 특성과 독립적인 특성을 효과적으로 분리합니다. 또한 Y자형 양방향 훈련 전략을 도입하여 특징 분리에 대한 해석 가능성을 향상시킵니다.

- **Technical Details**: 이 경량 VAE 모델은 입력 및 목표 이미지를 두 개의 잠재 분포로 인코딩하여, 각 분포는 표준 정규 분포에 대해 대칭성을 유지하게 됩니다. 이를 통해 독립적인 특성을 극대화하고, 변환 과정에서는 단순한 평균 반전을 통해 특성을 분리합니다. 기존 깊은 오토인코더에 비해 매개변수 수와 추론 시간을 현저히 줄이며, 합성 품질 또한 개선하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 실험 결과, FDA-VAE는 기존의 깊은 오토인코더 기반의 종단 간 합성 방법에 비해 모델 매개변수를 크게 감소시키고, 효율적인 합성을 통해 향상된 품질을 보여줍니다. 이러한 접근은 한정된 훈련 데이터의 상황에서도 최적의 매개변수 활용을 증가시킵니다. 또한, 해석 가능성이 뛰어난 구조를 통해 모델의 안정성과 성능을 더욱 강화하고 있습니다.



### Multimodal Carotid Risk Stratification with Large Vision-Language Models: Benchmarking, Fine-Tuning, and Clinical Insights (https://arxiv.org/abs/2510.02922)
- **What's New**: 이번 연구는 경동맥 죽종질환에 대한 위험 평가를 개선하기 위해 최첨단 대규모 비전-언어 모델(LVLM)의 가능성을 탐구합니다. 이 연구는 초음파 이미징(USI)과 구조화된 임상, 인구통계학적, 실험실, 단백질 바이오마커 데이터를 통합하여 다중 모달 경동맥 플라크 평가를 수행하는 프레임워크를 제안합니다. 또한, 다양한 오픈소스 LVLM을 비교 분석하고, 특히 LLaVa-NeXT-Vicuna를 초음파 분야에 적응시켜 뇌졸중 위험 분류에서 상당한 개선을 이루어냈습니다.

- **Technical Details**: 연구에서는 7272명의 환자로부터 수집된 B-mode 경동맥 USI 비디오와 해당하는 표 형식 데이터를 사용했습니다. 모든 비디오는 표준화된 조건에서 수집되었으며, 영상 해상도는 밀리미터당 1212 픽셀입니다. 데이터셋은 고위험 환자가 5959명, 저위험 환자가 1313명으로 구성되어 있으며, 이를 이진 분류에 적합하게 설계되었습니다.

- **Performance Highlights**: 연구 결과, 다양한 LVLM이 초음파 기반 심혈관 위험 예측에서 보여주는 가능성과 한계를 강조하며, 모델 보정 및 도메인 적응의 중요성을 강조했습니다. LLaVa-NeXT-Vicuna의 임상적 활용 가능성이 높아졌으며, 다중 모달 데이터를 통합함으로써 보다 정확한 위험 평가 및 진단을 위한 기초를 마련했습니다. 이 연구는 또한 기존 LVLM들이 임상적 상황에 얼마나 잘 일반화될 수 있는지를 평가하는 데 기여하고 있습니다.



### Zero-Shot Robustness of Vision Language Models Via Confidence-Aware Weighting (https://arxiv.org/abs/2510.02913)
Comments:
          Accepted to the NeurIPS 2025 Workshop on Reliable ML from Unreliable Data

- **What's New**: 이번 연구에서는 Zero-Shot (제로샷) 강건성을 개선하기 위해 Confidence-Aware Weighting (CAW) 기법을 제안합니다. CAW는 불확실한 적대적 예제에 우선순위를 부여하는 Confidence-Aware 손실과 적대적 입력에 대한 이미지 Encoder의 특징을 정규화하여 의미적 일관성을 유지하는 두 가지 구성 요소로 구성됩니다. 이를 통해 기존의 CLIP 모델의 성능을 향상시키면서도 메모리 사용량을 줄일 수 있습니다. 실험 결과, CAW는 AutoAttack과 같은 강력한 공격에서도 PMG-AFT와 TGA-ZSR을 능가하는 성능을 보여주었습니다.

- **Technical Details**: CAW는 CLIP 같은 비전-언어 모델의 적대적 훈련을 개선하기 위해 설계되었습니다. 첫 번째 구성 요소는 각 예제의 KL 발산을 조정하여 어려운 적대적 예제에 더 집중할 수 있도록 하는 Confidence-Aware 손실입니다. 두 번째 구성 요소는 정규화 항으로, 미세 조정된 이미지 Encoder의 특징과 동결된 특징 간의 거리를 최소화하여 사전 훈련된 모델의 의미적 지식을 유지하도록 돕습니다.

- **Performance Highlights**: TinyImageNet 및 14개의 제로샷 데이터 세트에서 실시한 광범위한 실험을 통해 CAW는 AutoAttack 하에서 최고의 성능을 보여주었습니다. PGD-100 및 CW 상황에서도 CAW는 PMG-AFT에 비해 더 높은 강건성과 깨끗한 정확도를 달성했습니다. 또한 CAW는 기존의 방법들에 비해 메모리 사용량도 적습니다.



### Don't Just Chase "Highlighted Tokens" in MLLMs: Revisiting Visual Holistic Context Retention (https://arxiv.org/abs/2510.02912)
Comments:
          Accepted by NeurIPS 2025 main

- **What's New**: 이 논문은 Multimodal Large Language Models (MLLMs)의 시각적 토큰(visual token)에 대한 효율적인 가지치기(pruning) 방법인 HoloV를 제안합니다. 기존의 attention-first 방법들은 의미적으로 비슷한 토큰을 유지하는 경향이 있어 높은 가지치기 비율 아래에서 성능이 급격히 저하되는 문제점을 가지고 있습니다. HoloV는 이러한 단점을 보완하여 전반적인 시각적 맥락을 포착할 수 있는 정보적이지 않은 토큰을 유지하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: HoloV는 토큰 가지치기에서 전반적인 의미적 연결성과 맥락적 주의(contextual attention)를 균형 있게 유지합니다. 이는 다양한 공간적 크롭(crop)에 걸쳐 가지치기 예산을 적응적으로 분배하여 이루어집니다. 이 방법은 보존된 토큰들이 고립된 주요 특징이 아니라 전체적인 시각적 맥락을 잘 포착하도록 보장하며, 이로 인해 정보의 손실을 최소화하고 과도한 가지치기 하에서도 작업 관련 정보를 유지하는 데 성공합니다.

- **Performance Highlights**: HoloV는 다양한 작업, MLLM 아키텍처 및 가지치기 비율에서 기존의 최첨단 방법들보다 우수한 성능을 보여줍니다. 예를 들어, HoloV를 적용한 LLaVA1.5 모델은 88.9%의 시각적 토큰을 가지치기한 후에도 원래 성능의 95.8%를 유지하여 효율성과 정확성 간의 trade-off를 극대화합니다. 이는 HoloV가 모델 비독점적이며 다양한 MLLM에 쉽게 통합될 수 있는 점에서도 큰 강점을 지니고 있음을 나타냅니다.



### Training-Free Out-Of-Distribution Segmentation With Foundation Models (https://arxiv.org/abs/2510.02909)
Comments:
          12 pages, 5 figures, 2 tables, ICOMP 2025

- **What's New**: 최근 자율 주행과 같은 안전-critical (critical) 응용 프로그램에서 중요한 의미론적 세분화 (semantic segmentation)의 Unknown Object Detection (알 수 없는 객체 탐지)이 중요해졌습니다. DINOv2, InternImage, CLIP과 같은 대형 비전 기반 모델들이 풍부한 특성을 제공하며 다양한 작업에서 일반화 능력을 보여주고 있습니다. 그러나 이러한 모델이 OoD (Out-of-Distribution) 영역을 탐지하는 능력은 충분히 탐구되지 않았습니다.

- **Technical Details**: 이 연구에서는 세분화 데이터 세트 (segmentation datasets)에서 파인 튜닝된 대형 비전 기반 모델이 Unsupervised (비지도) 방식으로 ID (In-Distribution)와 OoD 지역을 구분할 수 있는지 조사합니다. InternImage 모델을 기반으로 하여 KK-Means 군집화 (clustering)와 confidence thresholding을 이용해 OoD 클러스터를 식별하는 간단한 비훈련 방법을 제안했습니다. 이 방법은 Raw Decoder Logits (원시 디코더 로그잇)에서 직접적인 출력을 활용합니다.

- **Performance Highlights**: 제안한 방법은 RoadAnomaly 벤치마크에서 50.02 Average Precision을 달성했으며, ADE-OoD 벤치마크에서는 48.77을 기록했습니다. 이러한 결과는 전통적인 지도 학습 및 비지도 학습 기준을 초월하며, 최소한의 가정이나 추가 데이터로 일반적인 OoD 세분화 방법의 유망한 방향을 제시합니다. 이 연구는 기존의 접근 방식을 넘어서, 대형 비전 모델에서 본격적인 OoD 세분화를 탐구하는데 기여합니다.



### One Patch to Caption Them All: A Unified Zero-Shot Captioning Framework (https://arxiv.org/abs/2510.02898)
- **What's New**: 최근 제안된 zero-shot captioners는 페어링된 이미지-텍스트 데이터에 의존하지 않고 공통 공간의 비전-언어 표현을 활용하여 이미지를 설명하는 모델입니다. 본 논문에서는 이미지 중심에서 패치 중심으로 전환하는 새로운 구조를 제안하여, 지역 수준의 감독 없이도 임의의 영역을 캡션할 수 있는 unified framework rameworkName{}을 소개합니다. 이 접근 방식은 개별 패치를 원자적 캡션 단위로 취급하여 다양한 형태의 이미지 캡션 생성을 가능하게 합니다.

- **Technical Details**: 제안된 rameworkName{}은 기존의 captioning 방식에서 벗어나 패치별로 이미지를 설명할 수 있도록 구성되어 있습니다. 최신 구조인 비전 트랜스포머(vision transformers) 기반의 아키텍처를 이용하여, 이미지의 각 패치를 독립적으로 다루고 이를 조합하여 설명을 생성합니다. 특히 DINO 기반의 비전-언어 차원 모델을 통해 의미 있는 패치 표현을 생성하는 방법론을 중심으로 설명합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 다른 기준선 및 경쟁 모델들에 비해 zero-shot dense, region-set 캡셔닝 작업에서 뛰어난 성과를 보였습니다. 패치 기반의 의미 표현이 확장 가능성이 높은 캡션 생성에 있어 효과적임을 입증하였습니다. 또한, 새로운 trace captioning 과제를 도입하여 작동 방식을 확장하며 다양한 영역의 세부 캡션 생성 성능을 강조합니다.



### ELMF4EggQ: Ensemble Learning with Multimodal Feature Fusion for Non-Destructive Egg Quality Assessmen (https://arxiv.org/abs/2510.02876)
Comments:
          30 pages

- **What's New**: 이 논문은 상업적 가금류 생산에서 안전한 식품 관리와 제품 품질 유지를 위해 필요할 정도로 정확하고 비파괴적인 계란 품질 평가 방법을 제안합니다. ELMF4EggQ라는 앙상블 학습 프레임워크를 도입하여 이미지, 모양, 무게와 같은 외부 특성만을 사용하여 계란의 등급과 신선도를 분류합니다. 186개의 갈색 계란 데이터셋을 생성하였으며, 최초로 외부 비침습적 특성을 활용한 머신러닝 방법으로 내부 품질 평가를 수행한 연구라고 할 수 있습니다.

- **Technical Details**: 이 프레임워크는 외부 계란 이미지에서 추출한 딥 피처를 계란의 형태와 무게와 같은 구조적 특성과 통합하여 각 계란을 포괄적으로 표현합니다. 이미지 피처 추출은 사전 훈련된 CNN 모델(ResNet152, DenseNet169, ResNet152V2)을 활용하여 이루어지며, 이후 PCA 기반의 차원 축소와 SMOTE 증강을 통해 다수의 머신러닝 알고리즘으로 분류됩니다. 최종적으로, 여러 분류기의 예측을 통합하기 위해 앙상블 투표 메커니즘이 적용됩니다.

- **Performance Highlights**: 실험 결과에 따르면, 이 다중 모달 접근법은 이미지 전용 및 구조적 데이터(모양과 무게) 전용 기준에 비해 상당한 성능 향상을 보여주었습니다. 다중 모달 앙상블 접근법은 등급 분류에서 86.57%, 신선도 예측에서 70.83%의 정확도를 달성하였습니다. 또한 코드와 데이터는 이 URL에서 공개되어 투명성과 재현성을 촉진하고 추가 연구를 지원합니다.



### Med-K2N: Flexible K-to-N Modality Translation for Medical Image Synthesis (https://arxiv.org/abs/2510.02815)
Comments:
          ICLR2026 under review

- **What's New**: 이 논문은 다양한 의료 이미징 모달리티를 융합하여 임상 진단을 지원하는 K to N 의료 영상 생성 방안을 제시합니다. 기존의 일대일 번역 기법에서 벗어나 다양한 임상 시나리오를 충족시키기 위해 K → N 변환 접근 방식이 필요하다는 점을 강조합니다. 이 연구는 Modality-Task Heterogeneity, Fusion Quality Control, Modality Identity Consistency와 같은 세 가지 주요 도전 과제를 다루며 새로운 Med-K2N 프레임워크를 통해 이를 해결하고자 합니다.

- **Technical Details**: Med-K2N은 PreWeightNet, ThresholdNet, EffiWeightNet의 세 가지 협업 모듈로 구성되어 있으며, 이는 의료 데이터의 다중 모달리티를 시퀀스 프레임으로 처리합니다. PreWeightNet은 전역 기여도를 평가하고, ThresholdNet은 적응형 필터링 기준을 학습하며, EffiWeightNet은 효과적인 가중치 계산을 담당하여 모달리티 융합의 세밀한 제어를 가능하게 합니다. Causal Modality Identity Module (CMIM)은 생성된 이미지와 목표 모달리티 설명 간의 인과 관계를 설정하여 모달리티의 동일성을 유지합니다.

- **Performance Highlights**: 논문의 실험 결과, Med-K2N은 여러 데이터셋에서 기존의 방법들에 비해 우수한 성능을 보였습니다. 특히 PSNR 및 SSIM과 같은 객관적 지표에서 일관된 개선을 달성하여 다양한 K → N 구성에서 뛰어난 결과를 나타냈습니다. 이 연구는 실제 임상 다중 모달 영상 합성을 위한 유연한 생성 시스템을 제공하여 의료 영상 진단 분야에 기여할 것으로 기대됩니다.



### VERNIER: an open-source software pushing marker pose estimation down to the micrometer and nanometer scales (https://arxiv.org/abs/2510.02791)
- **What's New**: 이 논문에서는 소형 또는 나노 스케일에서의 포즈 추정 문제를 해결하기 위한 새로운 오픈 소스 소프트웨어인 VERNIER를 소개합니다. VERNIER는 pseudo-periodic 패턴을 기반으로 빠르고 신뢰할 수 있는 포즈 측정을 제공하도록 설계되었습니다. 이 소프트웨어는 노이즈, 초점 흐림 및 가림에 강한 로컬 임계값 알고리즘을 활용하여 성능을 개선했습니다.

- **Technical Details**: VERNIER는 작은 마커와 큰 마커의 두 가지 마커 유형을 지원하며, 각 마커 유형은 고유한 측정 원리에 따라 설계되었습니다. 작은 마커는 이미지 센서의 FoV 내에 있으며, 정밀한 이동 측정이 가능하며, 큰 마커는 FoV와 측정 범위를 독립적으로 설정하여 광범위한 측정 비율을 달성합니다. 정밀 측정 원리는 주로 푸리에 변환과 위상 기반 측측에 의존하며, 이는 나노미터 해상도를 제공합니다.

- **Performance Highlights**: VERNIER 소프트웨어는 소형 및 대형 마커를 통해 2D 및 3D 포즈를 이전에 비할 수 없는 정밀도로 측정할 수 있습니다. 특히, 측정 시스템의 작동 범위는 최대 10^8에 달해 뛰어난 해상도에 비례하여 매우 넓은 비율을 자랑합니다. 또한, 다양한 응용 분야에 맞게 쉽게 적응할 수 있도록 설계된 마커 시스템 덕분에 연구자 및 엔지니어들이 실험에 즉시 사용할 수 있는 도구를 제공합니다.



### MaskCD: Mitigating LVLM Hallucinations by Image Head Masked Contrastive Decoding (https://arxiv.org/abs/2510.02790)
Comments:
          accepted to emnlp2025 findings

- **What's New**: 논문에서는 이미지 헤드 마스크드 대비 디코딩(MaskCD)이라는 새로운 접근법을 제안합니다. 이 방법은 LVLMs의 '이미지 헤드'를 마스킹하여 contrastive decoding을 위한 부정적인 샘플을 구축함으로써, 환각 현상을 효과적으로 완화하는 방향으로 나아갑니다. MaskCD는 기존의 환각 완화 기법 대비 다양한 벤치마크에서 우수한 성능을 보이고 있습니다.

- **Technical Details**: LVLMs는 시각 정보와 텍스트 모달리티를 통합하여 다중 모달 추론을 수행하는 모델입니다. 본 연구에서는 LVLMs의 특정 레이어에서 '이미지 헤드'가 이미지 토큰에 과도한 주의를 기울이는 경향을 발견하였으며, 이들을 마스킹하여 부정적인 샘플을 만드는 방법론을 제시합니다. 기존의 대비 디코딩 방법(constrastive decoding)과 주의 조작(attention manipulation)의 장점을 결합하여 보다 높은 품질의 부정적 샘플을 구축할 수 있음을 강조합니다.

- **Performance Highlights**: MaskCD는 LLaVA-1.5-7b 및 Qwen-VL-7b 모델을 검증하는 다양한 벤치마크에서 실험되었습니다. 결과적으로 MaskCD는 환각 현상을 효과적으로 완화하면서 LVLMs의 기본적인 기능들을 유지하는 데 성공했습니다. 실험을 통해 MaskCD의 성능이 기존의 환각 완화 방법들을 초월한다는 점이 입증되었습니다.



### Align Your Query: Representation Alignment for Multimodality Medical Object Detection (https://arxiv.org/abs/2510.02789)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 다양한 의료 이미징 모달리티(예: CXR, CT, MRI)에 대한 의료 물체 탐지의 성능을 개선하기 위해 새로운 방법을 제시합니다. 기존의 단일 탐지기가 혼합된 의료 모달리티에서 학습할 때의 한계를 극복하기 위해, 객체 쿼리의 표현을 모달리티 텍스트와 정렬하는 방식을 채택하였습니다. 이를 통해, 모달리티 정보를 명확하게 반영하면서도 추가적인 주석 없이 가벼운 모달리티 토큰을 도입합니다.

- **Technical Details**: 본 연구의 핵심 기술은 Multimodality Context Attention(MoCA)와 Query REPA입니다. MoCA는 객체 쿼리와 모달리티 토큰 간의 상호 작용을 통해 모달리티 정보를 감지 과정에 통합하는 혁신적인 방법입니다. 또한 제안된 QueryREPA는 사전 학습 단계에서 특정 작업을 위한 대조 손실을 사용하여 쿼리 표현을 모달리티 토큰과 정렬하여, 다중 모달리티에서 신뢰성 있는 대표성을 형성합니다.

- **Performance Highlights**: 제안된 방법은 서로 다른 모달리티로 학습된 데이터셋에서 일관되게 AP(평균 정확도)를 개선하며, 구조적 수정 없이 최소한의 오버헤드로 성능 향상을 꾀합니다. MoCA와 QueryREPA의 결합은 의료 물체 탐지 성능에 시너지 효과를 제공하여, 실제 환경에서의 다중 모달리티 의료 정보 처리를 위한 효과적인 경로를 제시합니다.



### OTR: Synthesizing Overlay Text Dataset for Text Remova (https://arxiv.org/abs/2510.02787)
Comments:
          This is the author's version of the work. It is posted here for your personal use. Not for redistribution. The definitive Version of Record was published in Proceedings of the 33rd ACM International Conference on Multimedia (MM '25), October 27-31, 2025, Dublin, Ireland, this https URL

- **What's New**: 이 논문은 컴퓨터 비전에서 텍스트 제거(text removal)의 새로운 기준 데이터셋인 OTR(Overlay Text Removal)을소개합니다. 기존의 데이터셋이 자연 이미지에서의 장면 텍스트(scene text) 제거에만 초점을 맞춘 것과는 달리, OTR은 복잡한 배경 위에 텍스트를 합성하여 다양한 도메인에서의 평가를 가능하게 합니다. 이 데이터셋은 깨끗한 진실값(ground truth)을 보장하여 텍스트 제거 모델의 성능을 정확하게 비교할 수 있도록 설계되었습니다.

- **Technical Details**: OTR 데이터셋은 복합 배경(complex background)에 텍스트를 합성하는 합성적 접근 방식을 사용합니다. 이 방법은 이미지에서의 개체(object)의 위치를 고려하여 텍스트를 배치하므로, 보다 도전적인 텍스트 제거 시나리오를 생성할 수 있습니다. 또한, OTR은 픽셀 수준의 조작 없이 완전히 깨끗한 배경을 유지하여 기존 데이터셋의 한계를 극복할 수 있도록 지원합니다.

- **Performance Highlights**: 실험 결과, 기존 벤치마크는 질적인 특성을 정확하게 비교하는 데 한계가 있는 반면, OTR 데이터셋은 이러한 질적 특성을 더 잘 포착할 수 있음을 보여주었습니다. PSNR(Peak Signal-to-Noise Ratio) 등 다양한 평가 지표를 사용하여 OTR의 효과성을 입증하였으며, 이를 통해 텍스트 제거 방법의 성능을 더욱 신뢰성 있게 평가할 수 있습니다.



### Reasoning Riddles: How Explainability Reveals Cognitive Limits in Vision-Language Models (https://arxiv.org/abs/2510.02780)
- **What's New**: 본 논문은 Vision-Language Models (VLMs)의 복잡한 측면을 이해하기 위해 리버스 퍼즐(rebus puzzles)을 해결하는 과정에 대한 체계적인 설명 가능성 분석을 수행했습니다. 기존의 성과 평가를 넘어 VLMs의 인지적 과정과 실패 패턴을 탐구하면서, 221개의 리버스 퍼즐로 구성된 데이터 셋을 제시하고, 이 데이터를 통해 다양한 추론 품질을 평가하는 프레임워크를 개발했습니다. 이 연구는 VLM의 인지 과정에 대한 중요한 통찰력을 제공하며, 설명 가능성을 모델 성능의 핵심 요소로 강조합니다.

- **Technical Details**: 리버스 퍼즐 데이터 셋은 6개 인지 카테고리에 따라 분류되며, 각 퍼즐은 공간 인코딩(Spatial Encoding), 결측 이유(Absence Reasoning), 수리적 논리(Quantitative Logic), 문화적 상징(Cultural Symbolism), 음성 변형(Phonetic Transformation), 시각적 구성(Visual Composition) 등으로 주제화되어 있습니다. 우리는 세 가지 서로 다른 프롬프트 전략인 설명 후 해결(explain-then-solve), 해결 후 설명(solve-then-explain), 구성 요소 안내(component-guided)를 설계하여 VLM의 추론 과정과 해석 생성 측면을 타겟으로 했습니다. 연구는 최신 VLM 시스템을 사용하여 진행하였고, 각각의 모델에 대해 동일한 평가 조건을 유지했습니다.

- **Performance Highlights**: 모델 성능은 프롬프트 방식에 따라 크게 달라지며, 특히 구성 요소 안내 전략(component-guided strategy)에서 모든 모델이 개선된 성과를 보였습니다. GPT-o3는 다른 모델에 비해 지속적으로 우수한 정확성을 기록했으며, 이 연구 결과는 구조화된 인지 스캐폴딩이 문제 해결 성과를 향상시킬 수 있다는 가설을 뒷받침합니다. 이러한 차이는 인지적 접근 방식이 결과뿐만 아니라 문제 해결 경로에 미치는 영향을 이해하는 데 중요한 의미를 가집니다.



### AdaRD-key: Adaptive Relevance-Diversity Keyframe Sampling for Long-form Video understanding (https://arxiv.org/abs/2510.02778)
- **What's New**: 이 논문에서는 AdaRD-Key라는 새로운 모듈을 제안하여 긴 동영상의 이해를 위한 키프레임 샘플링 문제를 해결합니다. 현재의 멀티모달 대형 언어 모델(MLLMs)은 균일 샘플링에 의존하여 중요한 순간을 놓치는 문제가 있으며, 기존의 키프레임 선택 방식의 한계를 극복하고자 합니다. AdaRD-Key는 교육이 필요 없는 쿼리 주도 샘플링 모듈로, 쿼리 관련성과 비슷한 다양성을 결합하여 정보가 풍부하면서도 중복이 없는 프레임을 생성합니다.

- **Technical Details**: AdaRD-Key는 Relevance-Diversity Max-Volume (RD-MV) 목표를 통해 쿼리 기준의 관련성 스코어와 로그 결정 다항성과 같은 다양성 요소를 결합합니다. 이는 프레임-쿼리 유사성을 계산하고, 이를 기반으로 선택 사항을 정렬하여 중복 프레임을 억제합니다. 이 모듈은 쿼리 긴밀도가 낮을 경우, 다양성 중심 모드로 전환하여 추가적인 감독 없이도 넓은 범포를 회복할 수 있도록 합니다.

- **Performance Highlights**: AdaRD-Key는 LongVideoBench와 Video-MME에서의 extensive 실험을 통해 최첨단 성능을 보여줍니다. 이는 특히 긴 동영상에서 증거의 보존을 크게 향상시키며, 기존의 방법들에 비해 우수한 결과를 나타냅니다. 전체 파이프라인은 불필요한 훈련 없이 단일 GPU에서 실시간으로 작동 가능하며, 기존의 VLM과의 호환성도 제공합니다.



### Hierarchical Generalized Category Discovery for Brain Tumor Classification in Digital Pathology (https://arxiv.org/abs/2510.02760)
- **What's New**: 이번 연구에서는 뇌종양 분류를 위한 새로운 접근법인 Hierarchical Generalized Category Discovery for Brain Tumor Classification (HGCD-BT)를 소개합니다. 이 방법은 계층적 클러스터링(hierarchical clustering)과 대조 학습(contrastive learning)을 통합하여, 미지의 클래스도 포함된 비정형 데이터에서 분류를 가능하게 합니다. 특히, 기존의 GCD 방법들이 가지는 한계를 극복하며, 뇌종양의 계층적 분류 구조를 반영할 수 있습니다.

- **Technical Details**: HGCD-BT는 레이블이 있는 데이터와 없는 데이터를 혼합하여 비정형 데이터를 분류하는 것을 목표로 합니다. 이 기법은 새로운 반지도 학습(semi-supervised) 기반의 계층적 클러스터링 손실을 도입하여, 학습 과정에서 계층적 구조를 직접적으로 모델링함으로써, 기존의 GCD 방법보다 더 우수한 성능을 보입니다. 실험을 통해 HGCD-BT는 기존 GCD 방식 대비 +28%의 정확성 향상을 달성했으며, 이 결과는 뇌종양의 다양한 하위 유형을 식별하는 데에 특히 효과적임을 보여줍니다.

- **Performance Highlights**: HGCD-BT는 OpenSRH 데이터셋에서 패치 수준 분류에 대한 성능을 평가하여 큰 개선을 보였습니다. 또한 Hematoxylin and Eosin (H&E) 염색된 전체 슬라이드 이미지를 사용한 슬라이드 수준 분류에서도 일반화 가능성을 Demonstrate하여, 12개의 뇌종양 유형을 정확하게 분류할 수 있음을 입증했습니다. 이 연구는 imaging modalities의 다양성과 분류의 세분화에 있어 HGCD-BT의 강력한 성능을 강조합니다.



### Bayesian Test-time Adaptation for Object Recognition and Detection with Vision-language Models (https://arxiv.org/abs/2510.02750)
Comments:
          Under Review

- **What's New**: 이번 연구에서는 Bayesian Class Adaptation plus (BCA+)라는 통합적인 학습 없이 진행되는 Test-Time Adaptation (TTA) 프레임워크를 제안합니다. BCA+는 객체 인식과 감지를 동시에 수행할 수 있도록 설계되었으며, 동적 캐시를 통해 클래스 임베딩과 분포를 적응적으로 업데이트합니다. 이 연구는 VLM(vision-language models)의 실효성을 높이는 데 중점을 두고 있습니다.

- **Technical Details**: BCA+에서는 적응 과정을 베이지안 추론 문제로 수립하며, 이에 따라 초기 VLM 출력과 캐시 기반 예측을 융합하여 최종 예측을 생성합니다. 동적 캐시는 과거 예측에서 파생된 적응형 사전 분포를 저장하며, 현재 이미지의 피쳐와 스케일 유사성을 측정하여 확률을 갱신합니다. 이렇게 두 가지 적응 메커니즘을 도입하여 불확실성 유도 기반 융합을 통해 모델의 의미 이해와 맥락 신뢰도를 개선합니다.

- **Performance Highlights**: BCA+는 다양한 객체 인식 및 감지 벤치마크에서 최첨단 성능을 달성하며, 학습이나 역전파(backpropagation) 과정 없이도 높은 추론 효율성을 유지합니다. 방대한 실험을 통해 BCA+의 우수한 성능을 입증하였고, 이는 실시간 애플리케이션에 적합한 속도와 효율성을 제공합니다. 또한, BCA+는 기존 성과에 비해 클래스 분포의 변화에 보다 robust한 반응을 보여줍니다.



### Retrv-R1: A Reasoning-Driven MLLM Framework for Universal and Efficient Multimodal Retrieva (https://arxiv.org/abs/2510.02745)
Comments:
          NeurIPS 2025

- **What's New**: 본 논문은 다중모달 정보를 활용한 검색을 위한 첫 번째 R1 스타일 MLLM인 Retrv-R1을 소개하며, 강화 학습(MLLM)의 추론 능력을 높여 더 정확한 검색 결과를 생성합니다. 기존 방법들이 직면한 높은 계산 비용과 불안정성 문제를 해결하기 위해 정보 압축 모듈과 세부 검사 메커니즘을 도입하여 효율성을 높였습니다. 또한, 새로운 훈련 패러다임을 제공하여 성능과 효율성을 동시에 향상시킵니다.

- **Technical Details**: Retrv-R1은 기본 MLLM을 직접 훈련하는 대신 세부 검사가 포함된 정보 압축 모듈을 사용하여 토큰 수를 줄입니다. 이 과정에서 CoT(Chain-of-Thought) 접근 방식을 통해 난이도가 높은 후보를 식별하고, 새로운 교육 자료를 통해 효율적인 미세 조정을 수행합니다. 훈련 과정에서의 혁신적인 커리큘럼 보상 시스템이 추가되어 MLLM이 높은 효율성과 강력한 추론 능력을 개발하도록 돕습니다.

- **Performance Highlights**: Retrv-R1은 M-BEIR과 같은 대규모 테스트 벤치마크에서 강력한 성능을 입증하였으며, 다중 작업과 다양한 환경에서도 뛰어난 일반화 능력을 보여줍니다. 이러한 성과는 Retrv-R1이 높은 성능, 효율성 및 강한 일반화 능력을 가지고 있다는 것을 나타냅니다. 이 모델은 앞으로의 다중모달 검색 기술에 중요한 기여를 할 것으로 기대됩니다.



### Net2Net: When Un-trained Meets Pre-trained Networks for Robust Real-World Denoising (https://arxiv.org/abs/2510.02733)
- **What's New**: 이 논문에서는 Net2Net이라는 혁신적인 방법을 제안합니다. 이는 훈련되지 않은 네트워크와 미리 훈련된 네트워크의 장점을 결합하여 실제 세계의 노이즈 제거 문제를 해결하고자 합니다. Net2Net은 비지도 학습 모델인 DIP(Deep Image Prior)와 지도 학습 모델인 DRUNet을 정규화하여(Regularization by Denoising, RED) 통합합니다.

- **Technical Details**: Net2Net 프레임워크는 각 입력 이미지 고유의 노이즈 특성에 적응하며 레이블이 없는 데이터에서 작동합니다. 이를 위해 비지도 학습 네트워크가 먼저 노이즈 특성을 수집하고, 그 다음에 대규모 데이터셋에서 학습한 표현을 이용하여 강력한 디노이징 성능을 제공합니다. 이 하이브리드 접근 방식은 노이즈 패턴 간의 일반화를 개선하고, 특히 훈련 데이터가 제한적인 시나리오에서 성능을 향상시킵니다.

- **Performance Highlights**: 실험을 통해 다양한 실제 노이즈 데이터셋에서 Net2Net의 효과성과 강력함을 입증했습니다. 이 방법은 합성 데이터셋에서뿐만 아니라 다양한 이미징 조건에서 저데이터 환경에서도 우수한 성능을 발휘합니다. Net2Net은 복잡한 데이터와 실제 노이즈 환경에서의 일반화 성능을 강화하며, 다양한 응용 분야에서의 적용 가능성을 보여줍니다.



### From Tokens to Nodes: Semantic-Guided Motion Control for Dynamic 3D Gaussian Splatting (https://arxiv.org/abs/2510.02732)
- **What's New**: 이 논문은 모노큘러 비디오에서 동적 3D 재구성을 위한 새로운 방법론을 제안합니다. 제안된 방법은 모션 복잡도에 따라 제어 점의 밀도를 조정하는 모션 적응 프레임워크를 활용합니다. 이를 통해 정적 배경에서는 중복성을 줄이고, 동적 영역에서는 중요한 제어 점을 집중할 수 있습니다.

- **Technical Details**: 제안된 접근법은 비전 기초 모델에서 추출한 의미론적 및 모션 사전 지식을 활용하여, 3D 공간에 제어 점을 배치하는 데 효과적으로 전달합니다. 이 과정은 후보 노드를 생성하고, 의미적 유사성과 모션 경향 점수에 따라 노드를 병합하는 모션 적응 압축으로 이어집니다. 최종적으로 스플라인 기반의 경로 매개변수화를 통해 보다 매끄러운 모션 표현을 달성합니다.

- **Performance Highlights**: 방대한 실험을 통해 기존 최첨단 방법보다 재구성 품질과 효율성이 크게 향상된 결과를 보였습니다. 제안된 방법은 동적 장면에서 제어 점 할당과 모션 복잡성 간의 불일치를 직접 해결하고, 안정적인 최적화와 매끄러운 모션 흐름을 지원하는 특징을 가지고 있습니다.



### MoGIC: Boosting Motion Generation via Intention Understanding and Visual Contex (https://arxiv.org/abs/2510.02722)
- **What's New**: 본 논문에서는 MoGIC라는 통합 프레임워크를 제안하여 인간의 행동 의도와 시각적 정보를 통합한 다중 모드(multi-modal) 움직임 합성을 구현하고 있습니다. 기존의 텍스트 기반 모션 생성 방식들이 행동의 실행에 대한 인과 관계를 간과하는 한계점을 보완하고, 이러한 의도를 모델링함으로써 더 정밀하고 개인화된 움직임 생성을 가능하게 합니다. MoGIC는 멀티모달 조건에 최적화된 움직임 생성과 의도 예측을 공동으로 수행하여 인간의 잠재적 목표를 밝혀내고, 시각적 사전 정보(visual priors)를 활용하여 품질을 향상시킵니다.

- **Technical Details**: MoGIC는 의도 예측(Intention Prediction)과 모션 생성(Motion Generation) 두 가지 모듈을 갖추고 있으며, 각 모듈이 이질적인 표현으로 인한 의미 혼동을 피하면서 동시에 최적화됩니다. 또한, 시각 모달리티(visual modality)를 추가하여 텍스트의 모호성을 줄이고, 일관된 표현 학습을 위한 보조적인 인식을 도입합니다. 적응형 범위를 갖춘 혼합 주의 메커니즘(mixture-of-attention mechanism)을 통해 중요 조건화 조건(conditional tokens)와의 상호작용을 강화하여 정확한 모션 생성을 지원합니다.

- **Performance Highlights**: MoGIC는 HumanML3D와 Mo440H 데이터셋에서 각각 FID를 38.6%와 34.6% 감소시키며, 기존의 LLM 기반 방식들을 초월하는 성능을 보여줍니다. 경량화된 텍스트 생성 모듈을 활용하여 Motion Caption 작업에서 성능이 개선되었습니다. 또한, 시각적 모달리티의 통합으로 사용 가능한 새로운 기능들이 열려지며 텍스트 전용 조건을 넘어서 다양한 다운스트림 작업에 Apply될 수 있음이 증명되었습니다.



### FSFSplatter: Build Surface and Novel Views with Sparse-Views within 3min (https://arxiv.org/abs/2510.02691)
- **What's New**: FSFSplatter는 자유로운 희소 이미지로부터 빠른 표면 재구성을 가능하게 하는 새로운 접근 방식을 제안합니다. 이 방법은 end-to-end 밀집 Gaussian 초기화, 카메라 매개변수 추정 및 기하학적으로 향상된 장면 최적화를 통합하고 있습니다. FSFSplatter는 멀티 뷰 이미지를 인코딩하기 위해 대형 Transformer를 사용하여 자기 분할 Gaussian 헤드를 통해 밀집하고 기하학적으로 일관된 Gaussian 장면 초기화를 생성합니다.

- **Technical Details**: 이 연구에서는 Gaussian Splatting (GS)의 원리를 기반으로 하여, 3D 장면을 Gaussian 원소들의 집합으로 표현하고 있습니다. FSFSplatter는 카메라 매개변수, 깊이 맵 및 Gaussian 장면을 독립적인 예측 헤드를 통해 회귀합니다. 이를 통해 밀집하고 기하학적으로 정확한 Gaussian 개체를 예측하고, 지역적인 플로터를 제거하며, 희소 입력의 최적화 과정에서 제한된 뷰에 대한 과적합을 줄여줍니다.

- **Performance Highlights**: FSFSplatter는 DTU 및 Replica 데이터셋에서의 표면 재구성 및 새로운 뷰 합성(NVS)에서 최신 기술 수준의 성능을 달성하고 있습니다. 이 방법은 기존의 표면 재구성 방법보다 더 빠르고 정확한 결과를 보여주며, 여러 장면 최적화 없이도 높은 재구성 품질을 유지합니다.



### Smart-GRPO: Smartly Sampling Noise for Efficient RL of Flow-Matching Models (https://arxiv.org/abs/2510.02654)
- **What's New**: 최근 흐름 맞춤(flow-matching) 기술의 발전으로 고품질 텍스트-이미지 생성이 가능해졌습니다. 하지만 이러한 모델은 강화 학습(reinforcement learning)에 적합하지 않다는 문제가 있습니다. 본 논문에서는 Smart-GRPO라는 새로운 방법을 제안하며, 이는 강화 학습을 위한 노이즈 변동 최적화의 첫 번째 방법입니다. Smart-GRPO는 후보 노이즈를 디코딩한 후 보상 함수로 평가하고, 보상이 높은 영역으로 노이즈 분포를 정제하는 반복 검색 전략을 사용합니다.

- **Technical Details**: Smart-GRPO의 핵심 가설은 노이즈의 시드가 효과적인 학습에 기여하는 정도가 다르며, 유용한 시드를 우선적으로 샘플링함으로써 수렴을 가속화할 수 있다는 점입니다. 이 방법은 프리트레인된 보상 모델을 사용하여 후보 노이즈 시드를 평가하고, 고품질 생성을 가져올 것으로 예상되는 시드를 선택합니다. 이는 노이즈 분포에 대한 적응형 커리큘럼을 구성하는 것으로 볼 수 있으며, 훈련이 점진적으로 더 많은 정보를 제공하는 궤적을 생성하는 시드에 중점을 두도록 합니다.

- **Performance Highlights**: Smart-GRPO는 기존 방법과 비교할 때 보상 최적화와 시각적 품질 모두에서 개선되었음을 실험을 통해 입증하였습니다. 이 기술은 흐름 맞춤 모델의 효율적인 훈련과 인간 중심 생성 사이의 격차를 메우는 실용적인 경로를 제시합니다. 최종적으로 Smart-GRPO는 기존의 RLHF 파이프라인과의 호환성을 유지하면서 정책 최적화의 효율성을 개선합니다.



### Sequence-Preserving Dual-FoV Defense for Traffic Sign and Light Recognition in Autonomous Vehicles (https://arxiv.org/abs/2510.02642)
- **What's New**: 이번 연구는 자율주행 차량의 신호등 및 표지판 인식을 위한 이중 시야(Field-of-View) 및 시퀀스 보존 강건성 프레임워크를 제안합니다. 이는 aiMotive, Udacity, Waymo, 그리고 텍사스 지역의 자체 기록된 비디오로 구성된 다중 소스 데이터셋을 기반으로 합니다. 새로운 시스템은 실제 조건에서 발생하는 간섭에 대한 저항력을 갖추도록 설계되었습니다.

- **Technical Details**: 이 연구는 4가지 운영 설계 도메인(ODDs)인 고속도로, 야간, 비 오는 날, 도시 환경에서의 RGB 이미지 시퀀스를 시간적으로 정렬합니다. 제안된 프레임워크는 기능 압축(feature squeezing), 방어적 증류(defensive distillation), 엔트로피 기반 이상 탐지(entropy-based anomaly detection)를 포함하는 3계층 방어 스택을 통합하여 시퀀스 수준의 시간적 투표를 추가적으로 구현합니다. 이 평가 프로토콜은 정확도, 공격 성공률(ASR), 위험 가중치 잘못 분류 심각도 및 신뢰성 안정성을 포함한 여러 요소를 분석하여 온라인 평가를 가능하게 합니다.

- **Performance Highlights**: 제안된 통합 방어 스택은 79.8 mAP를 달성하며 ASR을 18.2%로 감소시켰습니다. 이는 YOLOv8, YOLOv9 및 BEVFormer보다 우수한 성능을 보였고, 또한 고위험 잘못 분류를 32%로 줄이는 성과를 거두었습니다. 이 연구는 시간적 연속성과 다각적 감지 능력을 통해 실제 조건에서의 안정성을 높이는 데 기여하였습니다.



### Deep Generative Continual Learning using Functional LoRA: FunLoRA (https://arxiv.org/abs/2510.02631)
- **What's New**: 이 논문에서는 생성 모델의 지속적인 적응을 위한 새로운 방법인 FunLoRA를 제안합니다. 이 방법은 깊은 생성 모델의 성능 개선을 위해 효율적으로 파라미터를 조정하면서, 새로운 작업에 대해서만 학습을 진행하면서도 잃어버리는 현상인 catastrophic forgetting을 방지합니다. FunLoRA는 저비용 메모리 사용으로도 높은 표현력을 유지합니다.

- **Technical Details**: 기존의 방법들은 일반적으로 synthetic data에 의존하여 forgetting을 방지하려고 하였지만, 높은 메모리 요구와 긴 교육 시간 등의 문제를 안고 있었습니다. FunLoRA는 rank 1 matrices를 사용하여 이러한 문제를 해결하고, U-Net 아키텍처의 convolutional layer에 적용하여 성능을 극대화했습니다. 특히, 이 방법은 적은 수의 추가 파라미터로 높은 품질의 결과를 도출합니다.

- **Performance Highlights**: 다양한 기준선 데이터셋에서의 실험 결과, 제안된 방법이 state-of-the-art 방법들을 능가하는 성능을 보여주었습니다. 특히, CIFAR100 데이터셋에서 Pre-trained stable diffusion 모델보다도 뛰어난 성능을 기록하였고, 이는 훨씬 적은 파라미터와 학습 데이터 사용으로 이루어진 성과입니다. 이러한 결과는 FunLoRA가 지속적인 학습의 도전과제를 극복할 수 있는 잠재력을 가지고 있음을 입증합니다.



### Input-Aware Sparse Attention for Real-Time Co-Speech Video Generation (https://arxiv.org/abs/2510.02617)
Comments:
          Project Page: this https URL

- **What's New**: 이번 연구에서는 빠른 co-speech 비디오 생성에 초점을 맞춘 새로운 조건부 비디오 증류 방법을 제안합니다. 기존의 diffusion 모델은 느리고 실시간 배포에 어려움이 있었으나, 우리는 input human pose conditioning을 통해 이 문제를 해결하고자 하였습니다. 특히, 사람의 자세 정보로 주의(attention)를 올바른 영역에 집중시켜 비디오 품질을 보존하며 계산을 줄입니다.

- **Technical Details**: 제안된 방법은 input-aware sparse attention 메커니즘을 사용하여 pose keypoints 간의 정확한 대응 관계를 활용합니다. 이를 통해 변별력이 있는 부분에만 집중하고 불필요한 계산을 줄여 효율성을 높였습니다. 또한, input-aware 증류 손실을 도입하여 입술 동기화(lip synchronization) 및 손 동작의 사실성을 향상시켰습니다.

- **Performance Highlights**: 우리의 방법은 TalkShow 데이터셋을 포함한 여러 데이터셋에서 실시간 성능을 달성하였으며, 13.1배의 속도 향상과 10%의 시각적 품질 개선을 보였습니다. 기존의 오디오 기반 및 pose 기반 방법들에 비해 생성 품질과 손 동작의 사실성에서 뛰어난 성능을 보여주며, 빠른 처리 속도를 유지하였습니다.



### Ego-Exo 3D Hand Tracking in the Wild with a Mobile Multi-Camera Rig (https://arxiv.org/abs/2510.02601)
- **What's New**: 자동 마커 없이 여러 대의 카메라를 사용하는 새로운 시스템을 도입하여, 자연적인 환경에서의 손과 객체의 3D 인터랙션을 정확하게 추적하는 기술을 제시합니다. 이 시스템은 경량화된 백팩 스타일의 캡처 장비와 사용자가 착용하는 Meta Quest 3 헤드셋을 포함하여, 넓은 이동성의 이점을 제공합니다. 기존의 데이터셋의 한계를 극복하고, 다양한 환경에서의 손-객체 데이터 수집을 가능하게 합니다.

- **Technical Details**: 이 시스템은 8 대의 동기화된 피쉬아이 카메라와 2 대의 사용자 착용 카메라로 구성되며, 3D 손 포즈 그라운드 트루스(ground truth)를 생성하기 위해 다단계 에고-엑소(ego-exo) 추적 파이프라인을 적용합니다. 각 카메라는 최고 60Hz로 작동하며, 1024×1280 해상도를 지원하여 고품질의 비디오 데이터를 제공합니다. 이 시스템은 무선으로 작동하여 참가자의 움직임을 거의 제한하지 않도록 설계되었습니다.

- **Performance Highlights**: 이 시스템은 새로운 데이터셋을 통해 기존의 연구보다 더 도전적인 환경에서 정확한 3D 손 포즈 주석을 생성하는 능력을 보여줍니다. 수집한 데이터는 다양한 환경에서 손-객체 상호작용의 실재감과 3D 주석 정확도 간의 트레이드오프를 성공적으로 줄입니다. 평가 결과, 이 시스템은 기존의 데이터셋보다 개선된 성능을 발휘하며, 에고 중심 손 포즈 추정에 대한 새로운 가능성을 제시합니다.



### PEO: Training-Free Aesthetic Quality Enhancement in Pre-Trained Text-to-Image Diffusion Models with Prompt Embedding Optimization (https://arxiv.org/abs/2510.02599)
- **What's New**: 이 논문은 간단한 프롬프트를 입력받았을 때, 사전 훈련된 텍스트-이미지 확산 모델(text-to-image diffusion model)에서 미적 품질을 향상시키기 위한 새로운 접근 방식을 소개합니다. 이 방법은 Prompt Embedding Optimization (PEO)로 명명되며, 기존의 텍스트 임베딩을 최적화하여 생성된 이미지의 시각적 품질을 개선합니다. PEO는 훈련이 필요 없고 백본 모델(배경 모델)과 독립적이며, 제안된 방법의 성능은 최신 기술과 동등하거나 그 이상으로 나타났습니다.

- **Technical Details**: PEO는 고해상도 텍스트-이미지 생성을 위한 세 가지 목적 함수를 사용합니다. 첫째, LAION Aesthetic Predictor V2(LAION-AesPredv2)를 사용하여 생성된 이미지에 대한 시각적 품질 점수를 얻습니다. 둘째, CLIP의 이미지 인코더(image encoder)를 사용하여 생성된 이미지 피처(features)와 최적화된 텍스트 임베딩의 코사인 유사성을 계산하여 두 범주 간의 일치를 보장합니다. 마지막으로, 초기 텍스트 임베딩과 최적화된 텍스트 임베딩 간의 코사인 유사성을 계산하여 초기 프롬프트와의 최소 편차를 달성합니다.

- **Performance Highlights**: 제안된 방법의 효율성을 입증하기 위해 DiffusionDB, COCO 및 사용자 지정 캡션 세트에서의 실험과 사용자 연구가 진행되었습니다. 이 연구는 PEO가 최신 텍스트-이미지 생성 및 프롬프트 조정 방법들보다 우수한 성능을 발휘함을 보여줍니다. PEO는 간단한 프롬프트를 사용하더라도 고품질 이미지를 생성할 수 있는 가능성을 입증했습니다.



### How Confident are Video Models? Empowering Video Models to Express their Uncertainty (https://arxiv.org/abs/2510.02571)
- **What's New**: 이 논문은 비디오 생성 모델의 불확실성 정량화(Uncertainty Quantification, UQ)에 대한 최초의 연구를 제안합니다. 기존의 텍스트-비디오 모델이 사용자 의도와 일치하지 않거나 잘못된 정보를 토대로 영상을 생성하는 '환각(hallucination)' 문제를 해결하고자 합니다. 새로운 방법론인 S-QUBED(Semantically-Quantifying Uncertainty with Bayesian Entropy Decomposition)를 통해 이는 더욱 정확하게 접근할 수 있습니다.

- **Technical Details**: S-QUBED는 비디오 생성 모델의 불확실성을 정밀하게 분해할 수 있는 블랙박스 UQ 방법으로, 조건부 확률을 모형화하여 두 단계로 비디오 생성을 모델링합니다. 이 방법은 조건부 독립성을 기반으로 Latent Variable(z)를 활용하여 예측 불확실성을 Aleatoric(우연적 불확실성)과 Epistemic(지식 기반 불확실성)으로 나눕니다. 이를 통해 입력 프롬프트의 모호함이나 모델의 지식 부족으로 인한 불확실성을 효과적으로 구별할 수 있습니다.

- **Performance Highlights**: S-QUBED는 다양한 비디오 생성 작업에서 불확실성 추정치를 조정하는 성능을 보여주며, 태스크 정확도와 부정적 상관 관계를 가지고 있습니다. 논문에 제시된 실험 결과는 S-QUBED가 비디오 모델의 불확실성을 정량화하는 데 유용하고 효과적임을 입증합니다. 이는 향후 비디오 모델의 안전성을 높이는 데 기여할 것으로 기대됩니다.



### Unlocking the power of partnership: How humans and machines can work together to improve face recognition (https://arxiv.org/abs/2510.02570)
- **What's New**: 본 연구는 인간과 기계의 얼굴 인식 결정 결합이 정확도를 개선하는 상황을 규명합니다. 가장 큰 발견 중 하나는 기본 정확도의 차이가 작은 경우 인간-기계 협업이 유의미한 이점을 제공한다는 것입니다. 연구자들은 Proximal Accuracy Rule (PAR)이라는 규칙을 수립하여 이 협업의 성공을 예측할 수 있음을 보여줍니다.

- **Technical Details**: 이 연구에서는 Expertise in Facial Comparison Test (EFCT)와 Facial Expertise Test (FET)라는 두 가지 테스트를 통해 얼굴 인식의 정확도를 측정했습니다. 참가자들은 이미지 쌍을 보고 동일 인물인지 차이를 평가했으며, 기계는 이미지의 유사성을 기반으로 확신을 측정했습니다. 두 명의 참가자 또는 한 명과 기계의 결합 결정의 평균값을 기반으로 영혼을 구현했습니다.

- **Performance Highlights**: 결과적으로, 인간과 기계의 결합 결정은 특히 결합 대상의 기본 정확도 차이가 작을 때 성능이 향상됩니다. PAR은 신뢰할 수 있는 예측 도구로 작용하여 평균 이하의 인간이 기계와의 결합에서 발생 가능한 정확도 향상을 가능하게 합니다. 연구팀은 이러한 통찰력이 얼굴 인식 시스템의 설계와 운영에서 중요한 역할을 한다고 보고합니다.



### PhysHMR: Learning Humanoid Control Policies from Vision for Physically Plausible Human Motion Reconstruction (https://arxiv.org/abs/2510.02566)
- **What's New**: 이 논문에서는 PhysHMR이라는 새로운 통합 프레임워크를 제안하여 단일 정책 네트워크를 통해 시각적 관찰과 물리적 동역학을 공동으로 고려하여 인간의 동작을 재구성합니다. 기존의 두 단계 방식과는 달리 이 방식은 물리적 제약을 자연스럽게 적용하며, 이미지 특징을 직접 조건으로 하여 더 풍부한 시각적 문맥을 활용할 수 있습니다. 또한, 학습이 매우 비효율적이고 불안정하게 되는 샘플 비효율성 문제를 해결하기 위해 사전 학습된 모션 캡처 전문가로부터 지식을 전이하는 증류 전략을 도입하고 있습니다.

- **Technical Details**: PhysHMR은 2D 키포인트를 3D 공간으로 들어 올리기 위한 픽셀-레이(pixel-as-ray) 전략을 주요 구성 요소로 사용합니다. 이 3D 레이들은 운동 정책 입력으로 포함되어 강력한 전역 포즈 가이드를 제공합니다. 특히, 로컬 비주얼 피쳐와 결합하여 정책이 세부적인 포즈와 전역 위치를 모두 고려할 수 있도록 도와줍니다. 보상 체계는 운동 모방, 물리적 사실성, 그리고 물리적 부드러움을 균형 있게 추구하여 정책 학습을 안정화하는 방향으로 설계되어 있습니다.

- **Performance Highlights**: PhysHMR은 Human3.6M, AIST++, EMDB2와 같은 도전적인 모션 데이터셋에서 평가되었고, 기존의 동역학 기반 방법들과 비교하여 비주얼 정확도와 물리적 사실성 모두에서 우수한 성능을 보여주었습니다. 이 접근 방식은 일반적인 비물리적 아티팩트(예: 발 미끄러짐, 지면 침투)를 줄여, 애니메이션, 로봇 공학과 같은 하위 응용 프로그램에 적합한 고품질의 재구성된 모션을 제공합니다.



### Oracle-RLAIF: An Improved Fine-Tuning Framework for Multi-modal Video Models through Reinforcement Learning from Ranking Feedback (https://arxiv.org/abs/2510.02561)
Comments:
          Proceedings of the 39th Annual Conference on Neural Information Processing Systems, ARLET Workshop (Aligning Reinforcement Learning Experimentalists and Theorists)

- **What's New**: 최근 대형 비디오-언어 모델(VLMs)에서 인공지능 피드백을 활용한 강화 학습(RLAIF) 프레임워크를 제안하며, 기존의 인간 피드백을 대체했다. Oracle-RLAIF는 후보 모델의 응답을 점수화하는 대신 순위를 매기는 오라클 랭커(Oracle ranker)를 사용하여 비용 효율성을 높인다. 이 접근법은 데이터 효율성을 개선하고 다양한 멀티모달 비디오 모델의 정렬을 촉진한다.

- **Technical Details**: Oracle-RLAIF는 기존의 보상 모델에 의존하지 않고, 단순히 응답의 질을 평가하는 오라클 모델에 의존한다. 이는 비디오 언어 모델(VLMs)의 교육 및 학습 과정을 유연하게 바꿔주며, 다양한 시나리오에 적용 가능하다. 또한, Group Relative Policy Optimization(GRPO)의 새로운 수정 사항인 GRPO_{rank}를 도입하여 순위 기반 손실 함수를 직접적으로 최적화할 수 있도록 한다.

- **Performance Highlights**: Oracle-RLAIF는 여러 비디오 평가 데이터셋에서 기존의 최첨단 비디오-언어 모델을 능가하는 성능을 보인다. 이 모델은 현존하는 미세 조정 기법과 비교할 때 비디오 이해 성능을 지속적으로 개선하며, 순위 기반 피드백을 통해 강화 학습을 수행하는 유연하고 데이터 효율적인 프레임워크를 제시한다.



### Exploring OCR-augmented Generation for Bilingual VQA (https://arxiv.org/abs/2510.02543)
- **What's New**: 이 논문에서는 Vision Language Models (VLMs)를 활용한 OCR(Optical Character Recognition) 증강 생성에 대해 연구합니다. 특히 한국어와 영어를 대상으로 하여 다국어 모델 연구를 촉진하는 것을 목표로 합니다. 연구 결과, KLOCR이라는 강력한 이중 언어 OCR 기반 모델을 제공하며, OCR이 제공하는 추가적인 맥락이 VLM의 성능을 크게 향상시킨다는 관찰을 제시합니다.

- **Technical Details**: KLOCR 모델은 1억 개의 데이터 포인트로 훈련되어 한국어 및 영어에 대한 뛰어난 성능을 보입니다. 이 모델은 Transformer 기반 아키텍처를 사용하며, DeiT를 인코더로, RoberTa를 디코더로 활용합니다. 데이터 증강(SynthTIGER, PixParse 등)을 통해 다양한 텍스트 길이와 이미지 도메인의 OCR 데이터셋을 구성하였고, 이는 모델의 일반화 능력을 향상시키는 데 기여합니다.

- **Performance Highlights**: 모델을 활용한 광범위한 실험 결과, OCR로 추출한 텍스트가 오픈 소스 및 상업 모델에서 성능을 유의미하게 향상시키는 것으로 나타났습니다. OCR-증강된 방법과 비교하여 일반적인 VQA 성능이 향상되었으며, 결과는 VLM의 성능 개선 가능성을 시사합니다. KOCRBench는 Korean에 최적화된 VQA 과제를 위한 새롭고 중요한 기준을 마련하여 추가 연구를 촉진할 것으로 기대됩니다.



### Wave-GMS: Lightweight Multi-Scale Generative Model for Medical Image Segmentation (https://arxiv.org/abs/2510.03216)
Comments:
          5 pages, 1 figure, 4 tables; Submitted to IEEE Conference for possible publication

- **What's New**: Wave-GMS는 의료 이미지 분할을 위한 경량화되고 효율적인 다중 스케일 생성 모델로, 적은 수의 학습 가능한 파라미터(~2.6M)로 높은 성능을 제공합니다. 이 모델은 메모리 집약적인 사전 훈련된 vision foundation 모델을 로드할 필요가 없으며, 제한된 메모리의 GPU에서 대용량 배치를 지원합니다. 이를 통해 병원 및 의료 시설에서 AI 도구의 보다 공정한 배포를 목표로 하고 있습니다.

- **Technical Details**: Wave-GMS는 입력 이미지의 다중 해상도 분해를 통해 고품질의 잠재 표현을 생성하는 학습 가능한 인코더를 사용합니다. 이 모델은 고도로 압축된 Tiny-VAE를 활용하여 입력 이미지 및 분할 마스크의 잠재 표현을 생성하며, Latent Mapping Model (LMM)이 다중 해상도 잠재 표현과 해당 분할 마스크 간의 매핑을 학습합니다. 모든 훈련 중 Tiny-VAE의 인코더와 디코더는 고정되어 있으며, 가벼운 인코더와 LMM만 학습 가능하여 트래이너블 파라미터 수를 크게 줄입니다.

- **Performance Highlights**: Wave-GMS는 BUS, BUSI, Kvasir-Instrument 및 HAM10000과 같은 네 가지 공개 데이터셋에서 광범위한 실험을 통해 최첨단 분할 성능을 달성하였으며, 뛰어난 교차 도메인 일반화 능력을 보여주었습니다. 이는 팀의 기존 모델보다 더 적은 수의 파라미터로 높은 성능을 유지할 수 있도록 하였고, 특히 작은 데이터셋에서의 과적합을 방지하는 데 기여하고 있습니다.



### MM-Nav: Multi-View VLA Model for Robust Visual Navigation via Multi-Expert Learning (https://arxiv.org/abs/2510.03142)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 시각적 네비게이션(visual navigation) 분야에서 MM-Nav라는 새로운 다중 보기 VLA 모델을 제안합니다. 이 모델은 360도 관측을 통해 다양한 내비게이션 기능을 학습하며, 강화 학습(RL) 전문가들로부터 수집한 데이터로 훈련됩니다. 특히, 교사-학생 방식으로 데이터 집합을 조정하여 성능에 기반한 비율을 동적으로 변화시키는 것이 특징입니다.

- **Technical Details**: MM-Nav 모델은 pretrained 대형 언어 모델과 시각적 기초 모델을 기반으로 구축되었습니다. 총 4개의 카메라 뷰를 사용하여 주변 환경을 포괄적으로 관찰하고, 이를 통해 속도 명령(velocity commands)을 예측하는 액션 헤드를 구성합니다. DAgger 방식의 온라인 교사-학생 학습 전략을 도입하여 RL 전문가의 성공적인 시연을 활용하여 모델을 반복적으로 미세 조정합니다.

- **Performance Highlights**: 실험 결과, MM-Nav 모델은 다양한 내비게이션 기능을 갖춘 환경에서 우수한 성능을 발휘하며, 특정 내비게이션 기능을 위해 훈련된 RL 전문가들보다도 성능이 뛰어납니다. 또한, 실제 환경에서도 Robust한 제로샷(sim-to-real) 전환을 통해 강한 일반화 능력을 보여주어, 인공지능 기반 시각적 네비게이션 분야의 가능성을 확장합니다.



### Neural Posterior Estimation with Autoregressive Tiling for Detecting Objects in Astronomical Images (https://arxiv.org/abs/2510.03074)
- **What's New**: 본 논문에서는 고해상도 천체 이미지를 처리하는 새로운 아모타이즈드 변분 추론(amortized variational inference) 방법을 제안합니다. 이 방법은 $K$-색상 체커보드 패턴에 따라 잠재 공간을 분할하고 정렬하는 공간 자가회귀(spatial autoregressive) 변분 분포의 가족을 포함하고 있어 기존의 방법보다 효율성과 정확성을 개선합니다. 특히, 이 구조는 포스터리어 분포의 조건부 독립성과 잘 일치하도록 설계되어 한층 더 신뢰성 있는 추정치를 가능하게 합니다.

- **Technical Details**: 제안된 방법은 신경 후방 추정(neural posterior estimation, NPE)을 사용하여 변분 분포를 최적화 하며, 이 방법은 컨벌루션 신경망(convolutional neural network)을 통해 파라미터화됩니다. Sloan Digital Sky Survey의 이미지를 활용하여 성능을 평가하며, 되도록 포괄적인 불확실성 정량화를 제공하여 다수의 약한 신호를 가진 천체 물체를 효과적으로 식별합니다. 이러한 방식은 천체가 겹치거나 블렌딩(blending)되는 현상에서 발생하는 문제를 극복하는 데 중점을 둡니다.

- **Performance Highlights**: 논문에서 제안된 방법은 최신 기술 대비 우수한 성능을 기록하였으며, 평가에 사용된 데이터셋에서 명확한 분류 능력을 입증하였습니다. 아울러, 새로운 자가회귀 구조가 후방 확률(posterior calibration)을 크게 향상시킨다는 점도 강조됩니다. 이러한 진전은 더 많은 천체 데이터를 분석할 수 있는 가능성을 열어 주며, 우주론적 연구에 있어 매우 가치 있는 정보를 제공할 것으로 기대됩니다.



### Confidence and Dispersity as Signals: Unsupervised Model Evaluation and Ranking (https://arxiv.org/abs/2510.02956)
Comments:
          15 pages, 11 figures, extension of ICML'23 work: Confidence and Dispersity Speak: Characterizing Prediction Matrix for Unsupervised Accuracy Estimation

- **What's New**: 이 논문은 배포 환경에서 레이블이 없는 테스트 데이터로 모델의 일반화 성능을 평가하기 위한 통합적이고 실용적인 프레임워크를 제시합니다. 주목할 점은 'confidence'와 'dispersity' 두 가지 내재적 속성이 모델의 예측 신뢰도와 다양성을 측정하여 일반화 성능에 대한 강력한 신호를 제공한다는 것입니다. 또한, 하이브리드 메트릭이 기존의 단일 측면 메트릭보다 일관되게 우수한 성능을 보임을 입증합니다.

- **Technical Details**: 모델 성능 평가 방식은 크게 데이터셋 중심 평가와 모델 중심 평가로 나눌 수 있습니다. 데이터셋 중심 평가에서는 여러 레이블 없는 테스트 데이터셋에서 고정 모델의 정확도를 추정하고, 모델 중심 평가에서는 단일 레이블 없는 테스트 데이터셋에서 여러 후보 모델 중 가장 적합한 모델을 선정합니다. 이 논문에서는 'confidence'와 'dispersity'를 결합한 하이브리드 메트릭을 통해 두 가지 평가 방식 모두에서 견고한 성능을 보였습니다.

- **Performance Highlights**: 하이브리드 메트릭은 데이터셋 중심 평가와 모델 중심 평가 모두에서 가장 뛰어난 성능을 보여줍니다. 특히, 예측 매트릭스의 핵심 노름(nuclear norm)은 다양한 배포 시나리오에서 강력하고 정확한 성능을 유지하며, 클래스 불균형이 있는 경우에도 신뢰성을 제공합니다. 이러한 발견은 레이블이 없는 모델 평가를 위한 실용적이고 일반화된 기반을 제공합니다.



### PyRadiomics-cuda: a GPU-accelerated 3D features extraction from medical images within PyRadiomics (https://arxiv.org/abs/2510.02894)
- **What's New**: PyRadiomics-cuda는 PyRadiomics 라이브러리의 GPU 가속 확장으로, 의료 이미지에서 3D 형태 특성을 추출하는 계산상의 어려움을 해결하고자 개발되었습니다. GPU 하드웨어를 활용하여 주요 기하학적 계산을 오프로드하여 대규모 볼류메트릭 데이터셋의 처리 시간을 크게 단축시킵니다. 이 시스템은 원래의 PyRadiomics API와 완벽하게 호환되어 있어 기존 AI 워크플로우와 통합이 용이합니다.

- **Technical Details**: PyRadiomics-cuda는 CUDA를 통해 GPU 계산을 사용하여 3D 형태 특성의 추출을 가속화하는 고성능 확장입니다. 이 시스템은 메쉬 볼륨, 표면적, 최대 3D 지름 및 평면 지름을 포함한 형태 특성 클래스의 가속화에 중점을 두고 있습니다. PyRadiomics-cuda는 최적화된 CUDA 커널을 활용하여 형태 특성 계산을 효율적으로 수행하며, 설치 시에는 NVIDIA CUDA 컴파일러를 자동으로 감지하여 CUDA 소스 파일을 컴파일합니다.

- **Performance Highlights**: 효율성 실험에서는 다양한 최적화 전략의 성능을 철저히 벤치마킹하기 위해 C와 CUDA로 독립적인 테스트 프레임워크가 개발되었습니다. 이 프레임워크는 Marching Cubes 메쉬 생성 및 지름 계산과 같은 개별적인 계산 단계의 실행 시간을 측정할 수 있는 기능을 제공합니다. 결과적으로 PyRadiomics-cuda는 기존 CPU 기반 방법에 비해 큰 성능 향상을 보여주어, AI 파이프라인에서 고속 특성 추출이 가능해졌습니다.



### Representing Beauty: Towards a Participatory but Objective Latent Aesthetics (https://arxiv.org/abs/2510.02869)
- **What's New**: 이 논문에서는 기계가 아름다움을 인식하는 것이 무엇을 의미하는지 탐구합니다. 저자들은 신경망(neural networks)이 다양한 형태의 아름다움을 모델링 할 수 있는 가능성을 짚어 보았습니다. 이들은 깊이 학습(d深 learning) 시스템이 아름다움의 형식을 재현할 수 있는 근본 원인이 물리적 및 문화적 실체의 공동 기반에 있음을 주장합니다.

- **Technical Details**: 미술과 딥 러닝(deep learning)은 표현(representation)에 대한 중심적인 관심을 가지고 있습니다. 딥 러닝은 정보가 다양한 맥락에서 어떤 방식으로 유사하게 표현되는지를 보여주기 위한 '유니버설 표현 가설(universal representation hypothesis)'을 제안합니다. 각기 다른 모델들은 점점 현실을 더 잘 이해할 수 있도록 연합된 표현을 생산하게 됩니다.

- **Performance Highlights**: 연구 결과, 딥 러닝 모델들이 미적 이미지를 평가하는 데 효과적이라는 사실이 입증되었습니다. 이러한 발견은 인간-기계 공동 창작이 가능한 것을 넘어, 문화적 생산 및 기계 인식에서 아름다움이 중요한 역할을 한다는 것을 시사합니다. 저자들은 이러한 접근 방식이 예술가들이 AI 도구를 활용하는 방식에 혁신적인 변화를 가져올 가능성이 있다고 주장합니다.



### Work Zones challenge VLM Trajectory Planning: Toward Mitigation and Robust Autonomous Driving (https://arxiv.org/abs/2510.02803)
Comments:
          13 pages,5 figures

- **What's New**: 이 논문에서는 자동 운전 시스템에서 작업 영역에서 경로 계획을 위한 시각 언어 모델(Visual Language Models, VLM) 사용에 대한 최초의 체계적 연구를 수행하였습니다. VLM이 불규칙한 레이아웃 및 동적으로 변화하는 기하학적 구조가 포함된 작업 영역에서 정확한 경로를 생성하지 못한다는 점을 강조하며, 68%의 경우에 성공하지 못함을 보여줍니다. 이 연구는 비정상적인 패턴을 식별하고, 이를 통해 새로운 경로 생성 방법인 REACT-Drive를 제시하여 VLM의 한계를 극복하고자 합니다.

- **Technical Details**: REACT-Drive는 VLM을 통합한 두 단계의 경로 계획 프레임워크로, 실패 사례를 제약 규칙과 실행 가능한 경로 계획 코드로 변환합니다. 또한, Retrieval-Augmented Generation (RAG) 방법을 사용하여 유사한 패턴을 새로운 시나리오에서 검색합니다. 이 방식은 자율 주행 시스템이 안전 요구사항 및 교통 규칙을 준수하는 경로를 생성하도록 안내합니다. 실험 결과, REACT-Drive는 평균 이탈 오차를 약 3배 감소시키고, 타 방법에 비해 가장 낮은 추론 시간인 0.58초를 기록합니다.

- **Performance Highlights**: REACT-Drive는 ROADWork 데이터셋에서 Qwen2.5-VL을 평가할 때 VLM 기준선에 비해 약 3배의 이탈 오차 감소를 달성하였습니다. 또한, 15개의 실제 작업 영역 시나리오에서 실시한 실험을 통해 REACT-Drive의 강력한 실용성을 입증하였습니다. 이 연구는 향후 VLM을 통한 작업 구역 운전 능력 향상 연구에 큰 기여를 할 것으로 기대됩니다.



### GCVAMD: A Modified CausalVAE Model for Causal Age-related Macular Degeneration Risk Factor Detection and Prediction (https://arxiv.org/abs/2510.02781)
- **What's New**: 본 논문에서는 새로운 인과적 AMD 분석 모델인 GCVAMD를 소개합니다. 이 모델은 수정된 CausalVAE 접근법을 통합하여 원시 OCT 이미지에서 잠재적인 인과적 요인을 추출할 수 있도록 설계되었습니다. GCVAMD는 인과성을 고려하여 AMD 탐지를 가능하게 하여, 치료 시뮬레이션이나 주요 위험 요인에 대한 개입 분석도 수행할 수 있게 합니다.

- **Technical Details**: GCVAMD는 Deep Learning 기법 중 Attention Mechanism 기반의 CNN과 GradCAM을 활용하여 OCT 스캔에서 AMD 망막을 구분하는 데 성공적인 성과를 냈습니다. 이 모델은 드루젠(drusen)과 신생혈관화(neovascularization)와 같은 주요 위험 요인에 대한 인과 추론을 가능하게 하여, 관리에 도움되는 정보를 제공합니다. 또한, 원시 OCT 이미지만을 사용할 수 있기 때문에 데이터 수집의 복잡성을 낮추는 데 기여합니다.

- **Performance Highlights**: GCVAMD를 통해 드루젠 상태와 신생혈관화 상태를 AMD의 인과 메커니즘과 함께 식별할 수 있었습니다. 이는 AMD 탐지(분류)에서부터 개입 분석에 이르기까지 다양한 작업에 활용될 수 있습니다. 이러한 접근은 AMD 관련 진단과 분석에 있어 AI 기반 모델의 활용 가능성을 더욱 확대시킬 것입니다.



### Dale meets Langevin: A Multiplicative Denoising Diffusion Mod (https://arxiv.org/abs/2510.02730)
- **What's New**: 이번 논문에서는 생물학적 시스템에서의 학습과 최적화와 일치하는 방식으로 인공 신경망(ANN)을 훈련시키기 위한 새로운 접근 방식을 제안합니다. 특히, Dale의 법칙(Dale's law)에 의해 영감을 받은 지수 경량 최적화(exponentiated gradient optimization) 기법을 통해 로그-정규 분포(log-normal distribution)로 분포된 시냅스 가중치를 생성할 수 있음을 보여줍니다. 이러한 접근은 기하학적 브라운 운동(geometric Brownian motion)을 이용한 확률적 미분 방정식(stochastic differential equations, SDEs)과 연결되어 있으며, 이는 생물학적 기반의 생성 모델을 개발하는 데 적합합니다.

- **Technical Details**: Dale의 법칙에 따르면, 특정 시냅스를 통해 신경세포는 항상 흥분성 또는 억제성으로만 작용하며 학습 과정에서 그 역할을 전환하지 않습니다. 본 논문에서는 SDE의 역시간(reverse-time) 접근 방식을 통해 이러한 경량 최적화 방식이 곱셈 업데이트(multiplicative update) 규칙과 일치한다고 설명합니다. 이는 기존의 Brownian motion 기반의 방법과는 차별화된 신경망 훈련 방식을 제안하며, 새로운 곱셈 정합(score-matching) 손실 함수를 도입함으로써 이미지 생성 이미지와 같은 데이터에 적용 가능합니다.

- **Performance Highlights**: MNIST, Fashion MNIST, Kuzushiji 데이터셋에서의 실험 결과, 제안된 업데이트 방식이 생성 모델로서의 유능성을 입증했습니다. 논문에서 제안된 곱셈 경량 최적화 방식은 기존 생성 모델과 비교했을 때, 생물학적 영감을 받은 첫 번째 사례로 자리매김하며, 이미지 데이터 생성에 있어 유망한 성과를 거두었습니다. 이러한 연구는 딥러닝 및 생성 모델 분야에 있어 새로운 지평을 여는 기초가 될 것으로 기대됩니다.



### Image Enhancement Based on Pigment Representation (https://arxiv.org/abs/2510.02713)
Comments:
          14 pages, 9 figures, accepted at IEEE Transactions on Multimedia (TMM)

- **What's New**: 이 논문은 RGB 색상을 고차원 색 공간인 'pigments'로 변환하여 동적으로 적응하는 혁신적이고 효율적인 이미지 향상 방법을 제시합니다. 기존의 전통적인 방법과는 달리, 이 방법은 입력 콘텐츠에 따라 변환 과정을 조정하여 더욱 뛰어난 이미지 향상 성능을 달성합니다. RGB 색상을 'pigment'로 변환하고, 이를 개별적으로 재투영하여 최적의 색 정보 융합을 통해 향상된 이미지를 생성하는 과정이 포함됩니다.

- **Technical Details**: 제안된 방법은 5단계로 구성됩니다: visual encoder, pigment expansion, pigment reprojection, pigment blending, RGB reconstruction입니다. visual encoder는 이미지 향상에 필요한 주요 매개변수들을 예측하고, 색 공간 변환 과정을 통해 RGB 색상을 피그먼트 세트로 변환합니다. 특히 이 방법은 각 피그먼트에 대해 1D 재투영 함수를 적용하여 색상 간의 상관관계를 잘 반영합니다.

- **Performance Highlights**: 제안된 방법은 최신 이미지 향상 작업에 대한 실험에서 우수한 성능을 입증하였습니다. 특히, MIT-Adobe5K 데이터셋에서 90.09%의 1D LUT 셀을 활성화하며, 이전 3D LUT 기반 방법보다 18.2배 향상된 효율성을 보여줍니다. 이러한 성능 향상은 각 피그먼트를 사용하여 더 복잡한 색상 맵핑을 수행하고, 최종 이미지를 재구성함으로써 이루어졌습니다.



### A Statistical Method for Attack-Agnostic Adversarial Attack Detection with Compressive Sensing Comparison (https://arxiv.org/abs/2510.02707)
- **What's New**: 본 논문에서는 현대 머신 러닝 시스템에 대한 적대적 공격의 위험을 줄이기 위해 새로운 통계적 접근법을 제안합니다. 기존의 탐지 방법들이 보이지 않는 공격을 탐지하는 데 한계를 가진 반면, 본 연구에서는 신경망을 배포하기 전에 탐지 기준선을 수립하여 효과적인 실시간 적대적 탐지를 가능하게 합니다. 이는 압축된 신경망과 비압축된 신경망의 행동을 비교하여 적대적 존재의 지표를 생성합니다.

- **Technical Details**: 우리는 압축된 이미지와 비압축된 이미지를 입력으로 사용하는 두 개의 신경망을 사용해 이들의 분포 사이의 차이를 측정하는 방법을 개발했습니다. KL divergence와 L2 norm과 같은 여러 수학적/통계적 연산들을 활용하여 두 벡터(예: 신경망의 마지막 특성 층에서 생성된 특성 맵) 간의 차이를 정량화합니다. 이를 통해 적대적 이미지와 깨끗한 이미지를 구별할 수 있으며, 특정 임계값을 설정하여 적대적 변형 여부를 결정합니다.

- **Performance Highlights**: 본 방법은 다양한 공격 유형에 대해 거의 완벽한 탐지 성능을 보였으며, 기존의 방법들이 여러 공격에서 성능이 좋지 않은 것과 대비됩니다. 또한, 높은 수준의 가짜 긍정을 줄임으로써 실용적인 응용이 가능하도록 신뢰성을 높였습니다. 이 접근법은 공격 유형에 대한 사전 지식이 필요 없으므로, 모든 적대적 공격에 대해 효과적으로 적용될 수 있는 장점이 있습니다.



### A UAV-Based VNIR Hyperspectral Benchmark Dataset for Landmine and UXO Detection (https://arxiv.org/abs/2510.02700)
Comments:
          This work has been accepted and will be presented at the Indian Geoscience and Remote Sensing Symposium (InGARSS) 2025 in India and will appear in the IEEE InGARSS 2025 Proceedings

- **What's New**: 이 논문은 무인 항공기 플랫폼을 통해 수집된 가시광선 및 근적외선(VNIR) 하이퍼스펙트럼 이미지를 기반으로 한 새로운 벤치마크 데이터셋을 소개합니다. 이 데이터셋은 143개의 실제 대체 지뢰 및 폭발하지 않은 탄두(UXO) 목표가 심어진 통제된 테스트 필드에서 수집되었으며, 하이퍼스펙트럼 이미지는 고해상도 센서를 통해 획득되었습니다. 이러한 데이터셋은 레퍼런스 스펙트럼과 함께 제공되어 재현 가능한 연구를 지원하며, 지뢰 탐지 연구 분야에서 중요한 역할을 할 것으로 기대됩니다.

- **Technical Details**: 본 연구에서 수집된 VNIR 하이퍼스펙트럼 데이터는 Headwall Nano-Hyperspec 센서를 장착한 드론에서 실시되었습니다. 32개의 비행 루트를 통해 약 20.62m의 고도로 비행하며 270개의 연속 스펙트럼 밴드를 캡처했습니다. 데이터 처리 과정에는 방사선 보정(radiometric calibration), 오소레크티피케이션(orthorectification) 및 모자이크(mosaicking)가 포함되며, 두 지점 경험적 선법(Empirical Line Method, ELM)을 사용하여 반사율(reflectance)을 계산했습니다.

- **Performance Highlights**: 크로스 검증을 통해 6개의 레퍼런스 물체에 대해 RMSE(root mean square error) 값이 1.0 미만으로 발견되었고, 400-900 nm 범위에서의 스펙트럴 각도 측정(Spectral Angle Mapper, SAM) 값이 1도에서 6도 사이로 매우 높은 스펙트럴 충실도가 입증되었습니다. 이 데이터셋은 이전에 발표된 드론 기반 전자기 유도(EMI) 데이터와 결합하여 다중 센서 기준을 제공합니다. 따라서 이는 지뢰 탐지 및 UXO 연구에 있어 중요한 기초 자료가 될 것으로 보입니다.



### Learning a distance measure from the information-estimation geometry of data (https://arxiv.org/abs/2510.02514)
Comments:
          Code available at this https URL

- **What's New**: 이번 논문에서는 정보 추정 거리 함수인 Information-Estimation Metric (IEM)을 소개합니다. IEM은 신호의 연속 확률 밀도에 기반을 둔 새로운 형태의 거리 함수로, 정보 이론과 추정 이론 간의 관계를 바탕으로 하여 최적의 소음 제거기(denoiser)가 적용된 신호의 로그 확률과 오류를 연관 짓습니다. 이 논문은 IEM이 유효한 전역(global) 메트릭을 가지며, 지역(local) 리만 메트릭(Riemannian metric)에 대한 폐쇄형 표현식을 도출하는 방법을 보여줍니다.

- **Technical Details**: IEM은 여러 가지 소음 강도에 걸쳐 신호 쌍의 소음 제거 오류 벡터를 비교하여 계산됩니다. 이 함수는 복잡한 분포에 대해서도 적응할 수 있으며, 특히 가우시안 분포의 경우 마할라노비스 거리(Mahalanobis distance)와 일치합니다. IEM 계산에서는 학습된 소음 제거기를 활용하며, 이는 생성적 확산 모델(generative diffusion models)과 유사합니다.

- **Performance Highlights**: 논문에서는 ImageNet 데이터베이스에서 IEM을 학습하여, 인간의 지각 판단을 예측하는 데 기존의 최첨단 감독(supervised) 이미지 품질 메트릭과 경쟁하거나 이를 초과하는 성능을 보임을 실험적으로 입증하고 있습니다. 이러한 결과는 IEM이 이미지 및 신호의 품질 평가에서 유용한 도구가 될 수 있음을 나타냅니다.



### SIMSplat: Predictive Driving Scene Editing with Language-aligned 4D Gaussian Splatting (https://arxiv.org/abs/2510.02469)
- **What's New**: 새로운 접근 방식인 SIMSplat은 Predictive driving scene editor로, 자연어 프롬프트를 통해 직관적인 환경 조정이 가능하다. 이 시스템은 Gaussian으로 재구성된 장면과 언어를 정렬하여 도로 물체에 대한 직접적인 쿼리를 지원한다. 이러한 혁신은 한 가지의 에이전트에 국한된 편리한 조정을 넘어 여러 에이전트의 상호작용을 고도화하는 데 기여한다.

- **Technical Details**: SIMSplat은 motion-aware language embeddings를 통합하여 3D Gaussian 장면을 쿼리 및 조작할 수 있게 한다. LLM(large language model) 에이전트는 사용자의 프롬프트를 해석하여 객체를 추가, 제거 또는 수정할 수 있도록 한다. 또한, multi-agent motion prediction 모델을 통해 주변 에이전트의 움직임을 자연스럽게 반영하여 시뮬레이션의 현실감을 더욱 강화한다.

- **Performance Highlights**: SIMSplat은 Waymo 데이터셋에서 실험을 통해 탁월한 편집 및 시뮬레이션 능력을 입증하였다. 도로 물체 쿼리에서 정확성 기반의 기반선보다 61.2% 우수한 성능을 기록하며, 시뮬레이터 중 가장 높은 작업 완료율을 달성하였다. 게다가, 다중 에이전트 경로 개선 기능을 통해 예측 시뮬레이션에서도 가장 낮은 충돌 및 실패율을 기록하였다.



### Words That Make Language Models Perceiv (https://arxiv.org/abs/2510.02425)
- **What's New**: 이번 연구에서는 언어 모델(LLMs)이 텍스트만 학습하더라도 시각 및 청각 인코더와의 표현 정합성을 높일 수 있음을 보여주었습니다. 연구진은 명시적인 감각 프롬프트가 LLM의 잠재적 구조를 표출시킬 수 있다는 가설을 검증하였습니다. 예를 들어, 모델이 '보다(see)' 또는 '듣다(hear)'라고 지시받을 때, 이러한 프롬프트가 시각적 또는 청각적 증거에 기초하여 다음 단어 예측에 영향을 미친다는 것을 발견하였습니다.

- **Technical Details**: 연구에서는 텍스트만으로 학습된 LLM의 표현 기하학을 검사하여 이를 단일 모드 비전 및 오디오 인코더와 유사하게 변화시킬 수 있는 방법을 다루었습니다. 새로운 개념으로 생성적 표현을 도입하여, LLM이 생성하는 각 출력은 정해진 프롬프트와 지금까지 생성된 시퀀스의 함수로서, 추가적인 전진 패스를 포함합니다. 감각 프롬프트를 통해 이러한 생성적 표현을 제어할 수 있으며, 이는 더 높은 정합성으로 이어집니다.

- **Performance Highlights**: 연구 결과에 따르면, 단일 감각 단어가 포함된 프롬프트는 텍스트만 학습된 LLM의 커널을 감각 인코더의 기하학에 더 가깝게 이동시킬 수 있음을 보여줍니다. 생성의 길이가 길어질수록 표현의 유사성이 증가하며, 더 큰 모델일수록 감각 프롬프트에 대한 정합성이 높아지는 경향이 있습니다. 또한, 시각적 단서가 있을 경우 LLM이 VQA(Visual Question Answering)에서 더 나은 성능을 발휘하는 것으로 나타났습니다.



### Glaucoma Detection and Structured OCT Report Generation via a Fine-tuned Multimodal Large Language Mod (https://arxiv.org/abs/2510.02403)
- **What's New**: 이번 연구에서는 안구 건강 진단을 위한 설명 가능한 다중 모달 대형 언어 모델(MM-LLM)을 개발하였습니다. 이 모델은 옵틱 신경 두부(ONH) OCT 원주 스캔의 품질을 평가하고, 녹내장 진단 및 부위별 망막 신경 섬유층(RNFL) 얇아짐 평가를 포함한 구조화된 임상 보고서를 생성하는 데 사용됩니다. 1310명의 피험자에서 수집된 데이터를 기반으로 하여 모델을 튜닝하였습니다.

- **Technical Details**: MM-LLM은 Llama 3.2 Vision-Instruct 모델을 이용하여 OCT 이미지를 통한 임상 설명을 생성하도록 세밀하게 조정되었습니다. 훈련 데이터는 쌍으로 된 OCT 이미지와 자동으로 생성된 구조화된 임상 보고서를 포함하며, 질이 낮은 스캔은 사용 불가능으로 표시되었습니다. 모델은 품질 평가, 녹내장 탐지, RNFL 얇아짐 분류의 세 가지 작업을 위해 독립적인 테스트 세트에서 평가되었습니다.

- **Performance Highlights**: 모델은 품질 평가에서 0.90의 정확도와 0.98의 특이성을 달성하였습니다. 녹내장 탐지의 경우, 정확도는 0.86으로, 민감도는 0.91, 특이도는 0.73, F1-score는 0.91에 달했습니다. RNFL 얇아짐 예측 정확도는 0.83에서 0.94까지 다양하였으며, 특히 전반적인(글로벌)과 측면(템포럴) 부위에서 가장 높은 성능을 보였습니다.



### Secure and Robust Watermarking for AI-generated Images: A Comprehensive Survey (https://arxiv.org/abs/2510.02384)
- **What's New**: 이 논문은 AI가 생성한 이미지의 워터마킹에 대한 종합적인 조사를 제시합니다. 특히, 기존 연구에서 간과된 시각적 영역에 대한 보안과 강건성을 강조하며, 최신 공격 방법과 방어 전략을 연결하여 향후 연구 방향을 명확히 합니다. 이를 통해 워터마킹 시스템의 기초 정의와 전통적인 방법에서의 발전을 정리하여 독창적인 기능과 응용 분야를 명확히 하고자 합니다.

- **Technical Details**: 워크플로우에서 워터마킹은 두 단계로 나뉩니다: 임베딩(embedding) 단계와 검증(verification) 단계입니다. 임베딩 단계에서는 품질을 유지하면서 이미지를 생성하는 과정에서 워터마크를 포함하고, 검증 단계에서는 임베딩된 워터마크를 참조 워터마크 또는 키와 비교하여 진위를 확인합니다. 기존의 디지털 워터마킹 기술은 기하학적 공격에 취약하고, 생성된 이미지와 의미론적 일치성이 부족하다는 한계가 있습니다.

- **Performance Highlights**: 현재 여러 국가에서 Gen-AI 워터마킹을 통한 규제 강화가 진행되고 있습니다. 중국은 AIGC의 추적 가능성과 감독을 보장하기 위해 워터마킹을 의무화했고, 유럽연합 및 미국도 각각의 법적 요구사항을 통해 워터마킹의 중요성을 인식하고 있습니다. 이러한 노력은 정보의 정확성을 높이고, IP를 보호하며, 디지털 신뢰성을 향상시키기 위한 국제적 총합으로 이어지고 있습니다.



New uploads on arXiv(cs.AI)

### Coevolutionary Continuous Discrete Diffusion: Make Your Diffusion Language Model a Latent Reasoner (https://arxiv.org/abs/2510.03206)
Comments:
          27 pages

- **What's New**: 이번 연구에서 제안하는 Coevolutionary Continuous Discrete Diffusion (CCDD) 모델은 기존의 discrete diffusion 모델을 넘어 연속적인 확산 모델이 더 강력한 표현력을 가진다는 것을 입증합니다. CCDD는 연속적인 표현 공간과 불연속적인 토큰 공간을 결합하여, 두 모달리티의 장점을 살리면서 효과적인 denoising을 수행할 수 있도록 설계되었습니다. 이는 새로운 언어 모델링 패러다임을 제시하여 기존의 모델들이 직면한 훈련 가능성 및 표현력 문제를 해결하는 방향성을 제시합니다.

- **Technical Details**: CCDD는 joint multimodal diffusion 과정을 정의하며, CTMC(continuous-time Markov chain)와 SDE(Stochastic Differential Equation)를 활용하여 서로 다른 특성을 가진 두 공간에서 denoising을 동시에 수행합니다. 중첩된 표현 공간과 연속적인 확률 공간에서의 특성을 결합하여, 특히 CCDD는 언어 모델링에서 질 감도를 향상시키고 훈련 효율성을 높이기 위한 몇 가지 고급 아키텍처와 기술을 도입하고 있습니다. 또한, 새로운 샘플링 알고리즘을 적용하여 sampling quality와 efficiency 간의 균형을 조절합니다.

- **Performance Highlights**: 실험적으로 CCDD는 LM1B 데이터셋에서 기존의 동일한 크기 모델들에 비해 25% 이상의 성능 개선을 보여주었습니다. 이는 CCDD가 강력한 표현력과 저렴한 훈련 비용의 장점을 모두 겸비하고 있음을 입증합니다. CCDD의 아키텍처와 훈련 기법은 실제 언어 과제에서 뛰어난 성능을 보여주어, 차세대 언어 모델 개발에 기여할 것으로 기대됩니다.



### CoDA: Agentic Systems for Collaborative Data Visualization (https://arxiv.org/abs/2510.03194)
Comments:
          31 pages, 6 figures, 5 tables

- **What's New**: 이번 논문에서는 CoDA라는 새로운 협력형 다중 에이전트 시스템을 소개합니다. 이 시스템은 데이터 시각화를 위한 복잡한 자동화를 위해 메타데이터 분석, 작업 계획, 코드 생성 및 자기 반성을 담당하는 전문화된 LLM 에이전트를 사용합니다. 기존의 시각화 자동화 방식이 불가능했던 복잡한 데이터 환경에서의 문제를 해결하고, 단일 질의 구문 분석에 집중하는 대신 협력적인 접근 방식을 채택합니다.

- **Technical Details**: CoDA 시스템은 구조화된 의사소통과 품질 중심의 피드백 루프를 기반으로 하여, 여러 전문 에이전트가 협력하여 쿼리를 분해하고 데이터를 처리하며 결과를 반복적으로 다듬는 방식으로 설계되었습니다. 메타데이터 스키마 분석을 통해 LLM의 컨텍스트 윈도우 한계를 우회하고, 전문 에이전트를 통해 도메인 이해력을 향상시키며, 이미지 기반 평가로 시각화 품질을 검증합니다.

- **Performance Highlights**: 시험 결과, CoDA는 MatplotBench와 Qwen Code Interpreter에서 강력한 기준 대비 최대 41.5%의 개선을 보여줍니다. 특히, DA-Code Benchmark에서도 우수한 성능을 발휘하며 복잡한 소프트웨어 엔지니어링 시나리오에 대해 robust하게 대응할 수 있는 능력을 입증했습니다. 연구 결과는 CoDA의 핵심 요소들이 전체 성능에 통계적으로 유의미한 긍정적인 영향을 미친다는 것을 보여줍니다.



### Improving Cooperation in Collaborative Embodied AI (https://arxiv.org/abs/2510.03153)
Comments:
          In proceedings of UKCI 2025

- **What's New**: 본 논문은 Large Language Models(LLMs)를 활용하여 멀티 에이전트 시스템에서 협동적인 의사결정 및 행동을 향상시키기 위해 다양한 프롬프트 방법을 탐구합니다. CoELA라는 프레임워크를 기반으로 하여 AI 에이전트 간의 소통, 추리, 작업 조정을 위한 새로운 방법론을 제시하며, 음성 통합 기능도 포함됩니다. 이를 통해 협업 성과를 극대화하는 최적의 조합을 확인하고 효율성을 22% 향상시키는 성과를 올렸습니다.

- **Technical Details**: 본 연구는 Perception, Memory, Planning, Communication, Execution의 다섯 가지 모듈로 구성된 CoELA 프레임워크를 기반으로 합니다. 특히 Planning과 Communication 모듈에서 LLM을 활용하여 에이전트 간의 효과적인 협업 방식과 작업 조정 전략을 식별합니다. Ollama를 통합하여 다양한 LLM을 쉽게 실험할 수 있는 환경을 제공하며, 모델 전환 시 시스템의 핵심 파이프라인을 수정할 필요가 없도록 하였습니다.

- **Performance Highlights**: 연구 결과, 프롬프트 최적화는 협동 에이전트의 성능 개선에 큰 기여를 했으며, 예를 들어 Gemma3 모델을 적용했을 때 효율성이 22% 향상되었습니다. LLM 기반의 대화 시스템은 여러 에이전트 간의 상호작용을 명확히 하고 빠르게 만들어줍니다. 자연어 기반의 커뮤니케이션을 통해 인간과의 협력에서도 신뢰와 작업 효율을 높이는 결과를 보였습니다.



### A Study of Rule Omission in Raven's Progressive Matrices (https://arxiv.org/abs/2510.03127)
- **What's New**: 이 연구는 인공지능 모델의 일반화 능력을 테스트하기 위해 훈련 과정에서 특정 규칙을 일부러 생략하는 방식을 적용합니다. 특히, 기존의 RPM 데이터셋 대신 I-RAVEN 데이터셋을 사용하여 모델이 생략된 규칙을 포함한 문제에서 어떻게 성능을 발휘하는지를 분석합니다. 이를 통해 딥러닝 모델의 추상적 추론 능력을 한층 더 깊이 이해하고 새로운 통찰을 제공합니다.

- **Technical Details**: I-RAVEN 데이터셋은 다섯 가지 구조적 속성(Attribute)인 숫자(Number), 위치(Position), 유형(Type), 크기(Size), 색상(Color)을 기반으로 구성됩니다. 각 속성은 고정(Constant), 진행(Progression), 산수(Arithmetic), 세 가지 분포(Distribute Three)와 같은 네 가지 규칙에 의해 통제됩니다. 이 연구에서는 훈련 중 두 가지 규칙을 생략하는 수정된 데이터셋을 생성하여, 모델의 일반화 능력을 평가합니다.

- **Performance Highlights**: 결과적으로, 비전 기반 모델은 익숙한 규칙에 대해 잘 작동하지만, 새로운 규칙이나 생략된 규칙을 처리할 때 성능이 급격히 저하됩니다. 또한, 토큰 수준의 정확도와 전체 정답 정확도 간의 차이는 현재 접근 방식의 근본적인 한계를 드러냅니다. 이러한 결과는 AI 시스템이 패턴 인식을 넘어 더 강력한 추상적 추론을 할 수 있도록 하는 새로운 아키텍처의 필요성을 강조합니다.



### From Facts to Foils: Designing and Evaluating Counterfactual Explanations for Smart Environments (https://arxiv.org/abs/2510.03078)
Comments:
          Accepted at Ex-ASE 2025, co-located with the 40th IEEE/ACM International Conference on Automated Software Engineering (ASE 2025)

- **What's New**: 이 논문에서는 룰 기반 스마트 환경에 최적화된 카운터팩추얼 설명(counterfactual explanations)의 첫 번째 공식화 및 구현을 소개합니다. 기존 설명 엔진을 확장하는 플러그인으로 구현되어 있으며, 이는 스마트 환경에서 설명의 필요성을 충족할 수 있도록 돕습니다. 사용자 연구를 통해 카운터팩추얼과 전통적인 인과적 설명(causal explanations)과의 비교 평가를 진행하였으며, 이 연구는 설명의 필요성과 유용성을 다룹니다.

- **Technical Details**: 스마트 환경에서는 센서 기반 장치들이 사용자 의사결정 지원 및 이상 상황 관리에 필수적입니다. 이러한 환경에서 룰 기반 시스템은 미리 정의된 규칙을 실행하여 자동화를 이룹니다. 카운터팩추얼 설명이란, 특정 결과를 얻기 위해 어떤 조건이 달라져야 하는지를 설명하는 방식으로, 이는 인간의 인과 추론 방식을 반영합니다. 본 연구에서는 이 카운터팩추얼 설명을 생성하기 위한 새로운 프레임워크를 제안합니다.

- **Performance Highlights**: 유저 연구 결과, 카운터팩추얼 설명은 사용자들이 문제를 해결하기 위한 구체적인 행동 내용을 중시하기 때문에 선호되는 경우가 많습니다. 반면, 인과적 설명은 언어적 단순성에서 이점을 가지고 있어 시간 압박이 있는 상황에서 더 긍정적으로 평가되었습니다. 연구는 스마트 환경에서 설명의 실행 가능성을 위한 실용적인 프레임워크를 제공하며, 각 설명 유형의 효과적인 적용 상황에 대한 경험적 증거를 제시합니다.



### Onto-Epistemological Analysis of AI Explanations (https://arxiv.org/abs/2510.02996)
- **What's New**: 인공지능(AI)의 적용이 증가함에 따라 AI 도구의 신뢰성과 당위성에 대한 우려도 커지고 있습니다. 설명 가능한 AI(XAI) 방법은 AI 도구의 사용을 정당화하기 위해 개발되었지만, 각 XAI 방법이 다르게 해석될 수 있기 때문에 이들에 대한 신뢰성은 도전받고 있습니다. 본 연구는 다양한 XAI 방법의 근본 철학적 가정들, 즉 AI 설명의 본질과 존재에 관한 가정들을 탐구하여 XAI의 이해를 높이는 것을 목표로 하고 있습니다.

- **Technical Details**: 연구에서는 설명의 본존(ontology)과 인식론(epistemology)에 대한 가정들을 조사합니다. 특정 XAI 방법인 Layer-Wise Relevance Propagation(LRP)를 예로 들어, 이 방법의 설명은 별도의 입력 차원(term)들의 합으로 구성된다고 정의됩니다. 그러나 이와 같은 접근은 설명의 본질에 대한 철학적 관점에서 중요하며, 각 XAI 방법의 가정들이 어떻게 상이한지 분석하고 정리합니다.

- **Performance Highlights**: XAI 기술의 선택 및 적용에 있어 기본적인 onto-epistemological 패러다임을 무시하는 위험성을 강조하며, 각 분야에 적합한 XAI 방법을 찾는 과정이 필요함을 설명합니다. 예를 들어, 의학 연구는 현실이 인간의 마음 밖에 존재한다고 여기는 XAI 기술에만 적합하다는 점을 강조합니다. 연구 결과, 다양한 XAI 방법들이 다르게 가정하고 있으며, 이는 그 활용 분야에 따라 적합한 방법을 선택하는 데 중요한 역할을 합니다.



### Consolidating Reinforcement Learning for Multimodal Discrete Diffusion Models (https://arxiv.org/abs/2510.02880)
Comments:
          Project Page: this https URL

- **What's New**: 본 연구에서는 MaskGRPO를 소개합니다. MaskGRPO는 강화학습에서 DDM(Discrete Diffusion Model)에 대한 확장된 접근법으로, 효과적인 importance sampling과 모달리티 특화 방법론을 사용하여 멀티모달 강화학습을 가능하게 합니다. MaskGRPO는 DDM의 이론적 기반을 명확히 하여 그라디언트를 업데이트하기 위한 가치 있는 토큰 변동을 포착하는 importance 추정기를 구축할 수 있도록 합니다.

- **Technical Details**: MaskGRPO는 GRPO(Group Relative Policy Optimization)의 통합된 확장으로 설계되었습니다. 언어와 비전의 구조적 특성을 고려하여 언어의 높은 불확실성을 포착하기 위해 점진적으로 마스킹 비율을 증가시키는 fading-out masking 추정기를 도입하였습니다. 이미지에서는 효과적인 우도 추정을 위해 정보 변화를 포착하기 위해 마스크 비율을 높게 제한하는 샘플러를 제안합니다.

- **Performance Highlights**: MaskGRPO는 수학적 추론, 코딩, 시각적 생성 기준에서 더 안정적이며 효율적인 업데이트를 제공하여 향상된 추론 성능과 뛰어난 생성 품질을 나타냅니다. 본 연구는 DDM에 대한 새로운 보상 기반 학습의 기초를 세우며, 각 모달리티에 대한 인식 기반 샘플러와 추정기가 설계될 때만 DDM에서 정책 최적화가 효과적임을 강조합니다.



### Reward Model Routing in Alignmen (https://arxiv.org/abs/2510.02850)
- **What's New**: 이 논문에서는 보강 학습(reinforcement learning)과 AI 피드백을 활용한 새로운 모델인 BayesianRouter를 제안합니다. 기존의 단일 보상 모델(reward model, RM) 사용의 한계를 극복하기 위해, 여러 RM을 동적으로 선택하는 하이브리드 라우팅 프레임워크를 채택합니다. 이 시스템은 오프라인 학습과 온라인 선택을 통합하여 RM의 성능을 극대화하고, 초기에 느린 속도를 개선하는 데 도움을 줍니다.

- **Technical Details**: BayesianRouter는 두 가지 단계로 구성됩니다. 오프라인 단계에서는 다중 작업 라우터가 선호 데이터에 기반하여 각 RM의 신뢰도를 추정합니다. 온라인 단계에서는 Bayesian Thompson 샘플링 기법을 사용하여 쿼리별로 최적의 RM을 선택하며, 이를 통해 효율적인 피드백 수집 및 정책 업데이트를 수행합니다.

- **Performance Highlights**: 실험 결과, BayesianRouter는 AlpacaEval-2, GSM8K 등 여러 벤치마크에서 기존의 단일 RM이나 RM 앙상블을 일관되게 앞서가는 성과를 보였습니다. 기존 라우팅 방법과 비교할 때, BayesianRouter의 성능 향상은 특히 눈에 띄며, 이는 정책 분포의 발전에 적응하는 능력에서 기인합니다.



### Take Goodhart Seriously: Principled Limit on General-Purpose AI Optimization (https://arxiv.org/abs/2510.02840)
Comments:
          9 pages, 1 figure. Under review

- **What's New**: 이 논문에서 제시된 개념은 기계 학습에서 모델이 명시된 목표 함수를 실제로 충족한다는 가정을 비판하는 것입니다. 이를 Objective Satisfaction Assumption (OSA)라 부르며, OSA의 실패는 근본적으로 현실에서의 근사화, 추정, 최적화 오류에서 기인한다고 주장합니다. 이러한 결함들은 AI 시스템의 의도된 목표를 효과적으로 구현하는 것을 불가능하게 만듭니다.

- **Technical Details**: 저자들은 Hu et al.의 최근 통합 학습 프레임워크를 바탕으로 OSA의 개념을 수립하고, 기계 학습의 여러 패러다임을 포괄하는 방법론을 제안합니다. 기계 학습의 학습 작업은 일반적으로 어떤 목표 함수를 극대화하거나 최소화하는 최적화 문제입니다. 여기서는 경험 항(reward 또는 loss 등)과 정규화 항을 포함하는 객체 함수(objective function)를 다루며, 이를 통해 학습 알고리즘이 선택할 수 있는 함수 집합인 가설 클래스(hypothesis class)를 정의합니다.

- **Performance Highlights**: 전체 논문에서 강조되는 점은 일반 목적 AI 시스템에서 목표와 인간의 가치 간의 불일치가 심각한 위험을 초래할 수 있다는 것입니다. 특히 목표를 달성하기 위한 최적화 압력이 강해지면 Goodhart의 법칙에 기반한 통제 상실 시나리오가 발생할 수 있다는 경고를 내립니다. 이에 따라 OSA와 목표의 잘못된 지정(misspecification)이 필연적으로 연결되며, GPAI 최적화의 원리적 한계가 필요하다는 주장을 합니다.



### Beyond the Final Answer: Evaluating the Reasoning Trajectories of Tool-Augmented Agents (https://arxiv.org/abs/2510.02837)
Comments:
          Preprint. Under Review

- **What's New**: 최근 도구 증강 벤치마크는 복잡한 사용자 요청과 다양한 도구를 포함하고 있지만, 대부분의 평가 방법은 여전히 답변 매칭(answer matching)으로 한정되어 있습니다. 사용자 요청을 해결하는 데 필요한 단계가 증가함에 따라, 에이전트의 성능을 평가하려면 최종 답변을 넘어 문제 해결 과정(problem-solving trajectory)도 평가해야 합니다. 이 논문에서는 이를 위한 새로운 평가 프레임워크인 TRACE를 소개합니다.

- **Technical Details**: TRACE는 도구 증강 LLM 에이전트 성능의 다차원 평가를 위한 프레임워크입니다. 이 프레임워크는 이전의 추론 단계에서 수집한 지식을 축적하는 증거 은행(evidence bank)을 포함하여 에이전트의 추론 경로를 다각적으로 분석하고 평가할 수 있도록 합니다. 이를 통해 에이전트의 성능을 비용 효율적으로 평가할 수 있으며, 모든 유효한 실제 경로(ground-truth trajectory)의 주석 작업이 비현실적이라는 한계를 극복하고 있습니다.

- **Performance Highlights**: TRACE 프레임워크는 다양한 결함이 있는 경로가 포함된 새로운 메타 평가 데이터셋을 개발하여 기존 벤치마크를 보강합니다. 연구 결과는 TRACE가 복잡한 행동을 정확하게 평가할 수 있으며, 소규모 오픈 소스 LLM을 사용하더라도 확장 가능하고 비용 효율적인 방식으로 성능을 평가할 수 있음을 확인합니다. 또한, 도구 증강 작업을 수행하는 동안 에이전트가 생산한 경로를 평가하여 새로운 관찰 결과와 통찰력을 제시합니다.



### NCV: A Node-Wise Consistency Verification Approach for Low-Cost Structured Error Localization in LLM Reasoning (https://arxiv.org/abs/2510.02816)
- **What's New**: 본 논문에서는 Node-wise Consistency Verification (NCV)라는 새로운 프레임워크를 소개합니다. NCV는 다단계 추론 검증을 경량화된 이진 일관성 검사로 재구성하여, 전통적인 방법보다 정확도와 효율성을 높입니다. 이 프레임워크는 긴 추론 체인을 상호 연결된 검증 노드로 분해하여 오류를 정확히 확인하고, 필요 없는 긴 형식 생성을 피합니다.

- **Technical Details**: NCV는 복잡한 추론 검증을 структур화된 분해로 전환함으로써, 여러 단순한 검증 문제로 변환합니다. 이를 통해 비트 단위의 판단 문제를 나누어 처리할 수 있어, 정밀한 오류 위치 파악이 가능해집니다. 다양한 구조적 패턴을 가지며, 명확한 논리적 의존성이 있을 때는 명시적 엣지를 구축할 수 있고, 그렇지 않을 경우 선형 조건 체인으로 처리합니다.

- **Performance Highlights**: 실험 결과, NCV는 기존의 방법에 비해 F1 스코어에서 10%에서 25%까지 향상을 보였으며, 전통적인 방법보다 6배에서 58배 적은 토큰을 소모했습니다. 이러한 개선은 NCV의 구조적 분해 방식을 통해 달성된 결과로, 모든 벤치마크 세트에서 뛰어난 성능 개선을 입증했습니다.



### Automated Constraint Specification for Job Scheduling by Regulating Generative Model with Domain-Specific Representation (https://arxiv.org/abs/2510.02679)
Comments:
          Accepted for publication in IEEE Transactions on Automation Science and Engineering

- **What's New**: 이 논문은 제조 일정 최적화를 위한 자동화된 제약(specification)을 지정하는 데 사용되는 제약 중심의 구조를 제안합니다. Generative AI(GenAI) 기법과 대형 언어 모델(LLMs)을 활용하여 이 과정의 자동화를 목표로 하며, 생산의 복잡한 요구사항을 효과적으로 처리할 수 있는 방법을 제시합니다. 이러한 접근은 제조업체들이 직면하고 있는 다양한 제약 조건의 수동 사양 작업을 줄이고, 보다 효율적으로 자원을 배분할 수 있도록 지원할 것입니다.

- **Technical Details**: 제안된 구조는 세 가지 수준으로 구성된 계층적 구조(space)를 기반으로 하며, 이는 제조 제약을 조정하는 역할을 합니다. 상위 수준은 글로벌 운영 의존성과 자원 관계를 다루고, 중간 수준은 특정 실행 구성(context-specific execution configurations)을 처리하며, 하위 수준은 세부적인 일정 매개변수와 생산 사양을 관리합니다. 이러한 구조는 도메인-specific representation을 통해 정밀성과 신뢰성을 보장하면서도 유연성을 유지할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 이 접근 방식은 제약 사양 작업에서 순수 LLM 기반 접근 방식에 비해 현저하게 높은 성능을 보였습니다. 연구 결과는 제안된 구조가 GenAI의 생성능력과 제조 시스템의 신뢰성 요구사항을 성공적으로 조율하고 있음을 입증합니다. 또한, 다양한 제조 시나리오에 적합하게 자동으로 조정할 수 있는 알고리즘도 제시되어, 일관된 생산 계획의 생성을 위한 도구로서의 유용성을 강조하고 있습니다.



### ARMs: Adaptive Red-Teaming Agent against Multimodal Models with Plug-and-Play Attacks (https://arxiv.org/abs/2510.02677)
Comments:
          60 pages, 16 figures

- **What's New**: 이 논문에서는 Vision-Language Models (VLMs)에 대한 새로운 적응형 레드팀 에이전트인 ARMs를 제안합니다. ARMs는 VLM의 위험 평가를 시스템적으로 수행하며, 다양한 레드팀 전략을 최적화하여 유해한 출력( harmful outputs)을 효과적으로 이끌어내는 기능을 갖추고 있습니다. 또한 11개의 새로운 다중모드 공격 전략과 17개의 레드팀 알고리즘을 통합하여 VLM의 새로운 취약점을 탐색하고 있습니다.

- **Technical Details**: 기존의 안전 평가 방법론들은 주로 정적인 벤치마크에 의존하고 있어 새로운 위험에 적절히 대응하지 못하고 있었습니다. ARMs는 멀티모달 중심의 공격 전략을 다단계로 조합하며, 레이어드 메모리 아키텍처를 통해 공격의 다양성을 확보하고 효율적인 위험 탐색을 가능케 합니다. 이 메모리 설계를 통해 ARMs는 공격 패턴에 대한 오버피팅을 방지하고, 다양한 레드팀 인스턴스를 생성할 수 있습니다.

- **Performance Highlights**: ARMs는 여러 공개 벤치마크에서 기존 최상의 기준인 X-Teaming의 공격 성공률(ASR)을 평균 52.1% 초과하여 개선하며, Claude-4-Sonnet 모델에서 90% 이상의 공격 성공률을 기록했습니다. ARMs-Bench를 통해 30,000개 이상의 레드팀 인스턴스와 51개의 위험 범주를 아우르는 대규모 다중모드 안전 데이터셋을 구축했으며, 안전 파인튜닝을 통해 VLM의 안전성을 개선하는 방향으로 실질적인 지침을 제공합니다.



### AutoMaAS: Self-Evolving Multi-Agent Architecture Search for Large Language Models (https://arxiv.org/abs/2510.02669)
- **What's New**: AutoMaAS는 자가 발전하는 다중 에이전트 아키텍처 검색 프레임워크로서 기존의 단일 아키텍처 솔루션의 한계를 극복합니다. 이 프레임워크는 쿼리 복잡성과 도메인 요구 사항에 따라 자원 할당을 동적으로 조절할 수 있는 능력을 지니고 있습니다. 특히, 네 가지 주요 혁신을 포함하여 성능과 비용 분석을 통한 자동 엔진 생성, 통합 및 제거, 실시간 매개 변수 조정 등을 통해 효과적인 아키텍처 설계를 관한 새로운 패러다임을 제시합니다.

- **Technical Details**: AutoMaAS는 네 가지 구성 요소로 이루어져 있습니다: 동적 작업자 생애 주기 관리자, 다중 목표 비용 최적화기, 온라인 피드백 통합 모듈, 아키텍처 해석 엔진입니다. 이 시스템은 각 작업자의 성능 지표와 사용 패턴을 기준으로 작업자를 동적으로 평가하고 생성하는 과정을 자동화합니다. 또한, 기존의 고정된 작업자 풀 대신, 변화하는 환경에서 자동으로 작업자를 생성 및 제거하여 다양한 쿼리 특성에 적합한 아키텍처를 선택할 수 있습니다.

- **Performance Highlights**: AutoMaAS는 6개의 벤치마크에서 성능 개선을 1.0-7.1% 달성하고, 기존 최첨단 방법에 비해 추론 비용을 3-5% 감소시킵니다. 특히, 다양한 데이터셋과 LLM(대형 언어 모델) 플랫폼에 대한 우수한 전이 가능성을 보여줍니다. 이러한 성과는 다중 에이전트 시스템 설계의 새로운 패러다임을 설정하며, 실제 응용분야에서도 뛰어난 효과성을 입증하였습니다.



### A Concept of Possibility for Real-World Events (https://arxiv.org/abs/2510.02655)
- **What's New**: 본 논문은 L.A. Zadeh에 의해 1978년에 도입된 기존의 가능성(possibility) 개념을 대체하는 새로운 개념을 제안합니다. 이 새 버전은 원래 개념에서 영감을 받았지만, 형식적으로는 두 가지 모두 Łukasiewicz 다가치 해석(logical connectives)을 채택한 것 외에는 아무런 공통점이 없습니다. 본 연구는 특정 실세계 사건의 가능성에 집중하여, 사건 발생의 전제와 제약을 기반으로 가능성을 계산합니다.

- **Technical Details**: 이번 연구에서 제안하는 가능성 개념은 사건의 발생 가능성을 전제와 제약의 확률을 기반으로 제공함으로써, 객관적인 계산 방법을 확보합니다. 이 새로운 접근법은 기존의 흐림 집합(fuzzy set) 개념을 사용하지 않으며, 자연어에서 용이하게 이해할 수 있는 모델을 제공합니다. 여기서 사용된 Łukasiewicz 논리는 기본적인 논리 연산자를 직관적으로 이해할 수 있는 방식으로 모델링합니다.

- **Performance Highlights**: 이론적으로 여러 계획이 존재할 때, 가장 가능성이 높은 계획을 결정하는 데 사용될 수 있어, 계획 문제에 적합합니다. 본 연구는 차량 경로 계획(vehicle route planning)을 간단한 예로 들어 이 이론의 적용 가능성을 보여줍니다. 실세계 응용에 대한 가능성을 제시하며, 독창적인 접근 방식으로 인간의 계획에 대한 정상적인 추론을 포착할 수 있음을 시사합니다.



### Geolog-IA: Conversational System for Academic Theses (https://arxiv.org/abs/2510.02653)
Comments:
          17 pages, in Spanish language

- **What's New**: Geolog-IA는 에콰도르 중앙대학교의 지질학 논문에 대한 자연어 응답 시스템으로, Llama 3.1 및 Gemini 2.5 언어 모델을 활용합니다. 이 시스템은 Retrieval Augmented Generation(RAG) 아키텍처와 SQLite 데이터베이스를 결합하여 정보 검색의 정확성을 높이고, 과거의 지식이나 잘못된 정보로 인한 문제를 극복합니다. Geolog-IA는 높은 BLEU 점수(0.87)를 기록하며, 사용자 친화적인 웹 인터페이스를 제공하여 교사, 학생 및 관리직이 쉽게 상호작용할 수 있도록 도와줍니다.

- **Technical Details**: 이 시스템은 자연어 처리(NLP) 기술을 사용하여 LLM(Long Language Models)과 RAG-SQL의 조합을 통해 정보의 액세스, 분석 및 추출을 가능하게 합니다. RAG 기법은 LLM의 언어 이해 생성 능력과 함께 외부 데이터베이스에서의 정보 검색을 결합하여 정확하고 최신의 정보를 제공합니다. 학습 과정에서 LLM은 고급 신경망 아키텍처를 기반으로 하며, 이러한 시스템은 연구 및 교육의 효율성을 높이는 데 기여합니다.

- **Performance Highlights**: Geolog-IA는 학생들, 교사 및 행정직원에게 신뢰할 수 있는 정보를 제공하여, 학업과 연구 과정에서의 시간을 절약하고 효율성을 개선합니다. 교사들은 이 시스템을 통해 수업 자료를 업데이트하고 학생들의 연구 주제를 더 잘 지도할 수 있습니다. 학생들은 참고 문헌을 검색하는데 드는 노력과 시간을 획기적으로 줄일 수 있으며, 제안된 시스템은 지질학 분야의 교육과 연구에서 매우 중요한 도구로 자리잡을 것입니다.



### On the Role of Temperature Sampling in Test-Time Scaling (https://arxiv.org/abs/2510.02611)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 추론 시 reasoning을 개선하기 위해, 샘플링 온도를 다양하게 조절하는 다중 온도 스케일링(multi-temperature scaling) 방법을 제안하였습니다. 기존의 연구에서는 샘플 수(K)를 늘리면 정확도가 꾸준히 향상되었지만, K가 매우 커질 경우 더 이상의 향상이 발생하지 않는 한계를 발견했습니다. 이를 통해 단일 온도 스케일링이 모델의 잠재력을 완전히 탐색하지 못한다는 점을 밝혀냈습니다.

- **Technical Details**: 다중 온도 스케일링은 다양한 온도를 통해 샘플을 균등하게 나누어 모델의 reasoning 경계를 확장하는 접근입니다. 실험적으로 1,024개의 트레이스를 생성하는 동안 온도를 0.0에서 1.2까지 변화시키며 성능을 비교하였습니다. 이 과정에서, 특정 온도에서만 해결 가능한 어려운 문제들을 발견함으로써 모델의 전반적인 문제 해결 능력이 증가하는 것을 관찰했습니다.

- **Performance Highlights**: 평균적으로 Qwen3 모델(0.6B, 1.7B, 4B, 8B) 및 다섯 가지 대표적인 벤치마크를 넘어온 결과, 다중 온도 스케일링이 단일 온도 TTS에 비해 평균 7.3 포인트의 성능 향상을 보였습니다. 또한, 다중 온도 스케일링을 통해 기본 모델이 추가적인 후 훈련 없이도 RL-trained 모델과 유사한 성능에 도달할 수 있음을 증명했습니다. 이러한 결과는 TTS의 강력한 가능성과 다중 온도 스케일링의 효과적인 적용을 강조합니다.



### Mitigating Modal Imbalance in Multimodal Reasoning (https://arxiv.org/abs/2510.02608)
Comments:
          10 pages, 10 figures, CoLM 2025

- **What's New**: 이번 연구에서는 다양한 모달리티(moality) 간의 교차 갈등(cross-modal conflicts)을 다루며, 재단 모델들(FM)의 공동 추론(joint reasoning) 능력에 대해 탐구합니다. 연구 결과, FM들은 단일 모달리티의 경우 90%의 정확도로 갈등을 인식하지만, 모달리티가 분리될 경우 이 비율이 3%로 감소함을 보여줍니다. 이는 서로 다른 모달리티 간의 상관관계가 혼합된 정보를 처리하는 데 어려움을 겪는다는 것을 나타냅니다.

- **Technical Details**: 연구에서는 FM의 모달리티 우선순위가 교차 모달리티 주의 불균형(cross-modal attention imbalance)에 기인한다고 가설을 세웠습니다. 실험을 통해, FM들이 특정 모달리티에 지나치게 의존하며, 이는 주의 점수가 극도로 비대칭적(asymmetric)이라는 것을 확인했습니다. 또한, 단순히 훈련 데이터의 양을 늘리는 것으로는 이 문제를 해결할 수 없으며, 각 훈련 인스턴스 내에서 여러 모달리티를 명시적으로 결합하는 접근 방식이 주의 불균형을 줄이는 데 효과적임을 입증했습니다.

- **Performance Highlights**: 이 연구는 기존의 데이터 세트를 활용하여 여러 모달리티를 혼합하는 방법을 통해 FM의 공동 추론 성능을 향상시킬 수 있다는 점을 강조합니다. 연구 결과, 이는 다운스트림 성능(downstream performance)을 개선하며, 다양한 비전-언어 벤치마크(vision-language benchmarks)에서도 효과적임을 보여주었습니다. 결국, 연구팀은 FM들에게 실세계에서 발생하는 복잡성을 반영한 체계적인 훈련 접근 방식의 필요성을 강조하며, 교차 모달 상황을 다루는 방법의 중요성을 강조하고 있습니다.



### Multimodal Large Language Model Framework for Safe and Interpretable Grid-Integrated EVs (https://arxiv.org/abs/2510.02592)
Comments:
          This paper has been presented at the 2025 IEEE PES Conference on Innovative Smart Grid Technologies (ISGT 2025)

- **What's New**: 이 논문은 전기차(EVs)와 스마트 그리드의 통합이 교통 시스템 및 에너지 네트워크에 미치는 변혁적 기회를 다룹니다. 특히, 멀티모달 대형 언어 모델(LLM)을 기반으로 하는 프레임워크를 통해 다중 센서 데이터를 처리하고 운전자를 위한 자연어 경고를 생성하는 방법을 제시합니다. 이 프레임워크는 실제 주행 데이터를 활용하여 검증되었으며, 도시 환경에서의 안전하고 해석 가능한 상호작용을 보장합니다.

- **Technical Details**: 프레임워크는 이미지 객체 감지(YOLOv8), 위치 기반 정보, 차량 텔레메트리(CAN bus telemetry)를 포함한 멀티모달 인식을 통합합니다. 멀티모달 데이터를 융합하여 중요한 위험 신호를 일관되게 캡처하고 처리하여 운전자가 이해할 수 있는 메시지로 변환합니다. 이 과정은 시각적 인식, 의미론적 분할 및 차량 상태 데이터 모두를 포함하여, 주행 환경의 위험을 효과적으로 평가하고 경고를 생성하는데 필요한 기반을 제공합니다.

- **Performance Highlights**: 사례 연구를 통해 이 프레임워크의 효과를 입증하였고, 보행자, 자전거, 다른 차량과의 근접 상황과 같은 중요 순간에 대해 맥락 인식 경고를 생성하는 능력을 보여주었습니다. 이 연구는 LLM이 전기 모빌리티에서 보조 도구로서의 가능성을 지니고 있으며, 교통 시스템과 전력망 모두에 이점이 될 수 있음을 강조합니다. 안전한 차량 운영을 보장하기 위한 혁신적인 접근 방식으로서 이 프레임워크는 실제 데이터 이용하여 유용한 인사이트를 제시합니다.



### A Benchmark Study of Deep Reinforcement Learning Algorithms for the Container Stowage Planning Problem (https://arxiv.org/abs/2510.02589)
- **What's New**: 본 논문은 해양 운송 및 터미널 운영에서 중요한 컨테이너 적재 계획(Container Stowage Planning Problem, CSPP)의 자동화를 위한 새로운 Gym 환경을 개발했습니다. 이 환경은 기존의 RL(강화 학습) 알고리즘을 다양한 복잡성의 시나리오에서 비교 평가할 수 있는 재사용 가능한 플랫폼 역할을 합니다. 특히, 단일 에이전트 및 다중 에이전트 구성 모두에서 크레인 스케줄링을 포함한 CSPP 문제를 해결하는 방법론이 포함되어 있습니다.

- **Technical Details**: 우리가 개발한 Stowage Planning Gym Environment (SPGE)는 CSPP의 핵심 문제를 포착하고 크레인 스케줄링을 통합하여 사용자 정의 가능한 RL 플랫폼으로 확장되었습니다. SPGE는 선박과 야적장을 3D 슬롯 그리드로 추상화하여 배송 규칙의 제약 조건을 적용할 수 있습니다. 이 환경은 다양한 복잡성으로 문제를 조정하고, 표준 RL 라이브러리와의 호환성을 제공합니다.

- **Performance Highlights**: 논문에서는 DQN, QR-DQN, A2C, PPO, TRPO의 5가지 RL 알고리즘을 사용하여 복잡성이 증가하는 다양한 시나리오에서 성능을 평가했습니다. 실험 결과, 복잡성이 증가함에 따라 알고리즘 선택과 문제의 형식화가 CSPP의 성능에 큰 영향을 미친다는 것을 확인했습니다. 이러한 결과는 CSPP 문제에 관한 알고리즘의 효과성을 검증할 수 있는 중요한 기반을 제공합니다.



### Agentic Additive Manufacturing Alloy Discovery (https://arxiv.org/abs/2510.02567)
- **What's New**: 이 논문에서는 새로운 합금 발견을 자동화하고 가속화하기 위해 대규모 언어 모델(LLM) 기반의 에이전트와 다중 에이전트 시스템을 통합한 방법론을 제시하고 있습니다. 이러한 시스템은 사용자가 제안한 합금의 인쇄 가능성에 대한 분석을 수행하고, Tool call 결과에 따라 작업 경로를 동적으로 조정하여 자율적인 의사 결정을 가능하게 합니다. 이를 통해 Additive Manufacturing (AM) 분야에서 효율적인 합금 개발이 이루어질 수 있습니다.

- **Technical Details**: 연구에서는 Thermo-Calc 소프트웨어를 이용하여 특정 합금 조성에 대한 물성(physical properties)을 계산합니다. CALPHAD 기반의 솔버를 활용하여 고유의 상 다이어그램과 물리적 성질을 예측하며, 선택한 데이터베이스에서 필요한 thermophysical 속성을 추출해냅니다. 이 과정에서 Gibbs Free Energy를 최소화하는 반복 계산을 통해 각 합금의 고상 및 융해 상 전이 온도를 수치적으로 도출합니다.

- **Performance Highlights**: 이 다중 에이전트 시스템은 자율적으로 작업을 수행하며, 편리한 MCP 인터페이스를 통해 다양한 클라이언트에 통합될 수 있습니다. 실시간으로 인쇄 가능성을 평가하고, 결함 없는 융해 과정을 위한 프로세스 맵을 생성하는 등의 기능을 통해 실제 제조 환경에서 매우 유용한 도구로 자리잡을 수 있습니다. 이를 통해 AM 산업에서의 혁신적인 합금 개발 가속화가 기대됩니다.



### Orchestrating Human-AI Teams: The Manager Agent as a Unifying Research Challeng (https://arxiv.org/abs/2510.02557)
Comments:
          Accepted as an oral paper for the conference for Distributed Artificial Intelligence (DAI 2025). 8 pages, 2 figures

- **What's New**: 이 논문은 복잡한 다중 에이전트 작업 흐름의 관리 문제를 해결하고자 하는 Autonomous Manager Agent의 개념을 제시합니다. 이러한 에이전트는 복잡한 목표를 작업 그래프로 분해하고, 인적 및 AI 작업자에게 작업을 할당하며, 진행 상황을 모니터링하고, 변화하는 조건에 적응하며 투명한 커뮤니케이션을 유지해야 합니다. 이를 통해 인간-인공지능 팀의 협력을 조율하는 새로운 방향을 제시하고 있습니다.

- **Technical Details**: 저자들은 자동화된 워크플로우 관리를 Partially Observable Stochastic Game (POSG)로 모델링하였습니다. 이는 에이전트가 임무를 수행하는 중에 불확실성과 동적 환경을 감안하여 의사결정할 수 있도록 해 줍니다. 그녀석은 작업 분해, 비선형 목표 최적화, 긴급 팀의 협력 및 디자인에 의한 거버넌스와 같은 네 가지 기초 연구 문제를 해결하는데 중점을 두고 있습니다.

- **Performance Highlights**: GPT-5 기반의 Manager Agent를 20개의 다양한 작업 흐름에 대해 평가한 결과 이들은 목표 달성, 제약 준수 및 작업 흐름의 지속 시간을 동시에 최적화하는 데에 어려움을 겪고 있다는 점을 확인했습니다. 이는 복잡한 작업 흐름 관리가 에이전틱 AI 분야에서 여전히 해결해야 할 중요한 문제임을 강조합니다. 마지막으로, 이러한 자율 관리 시스템의 조직적 및 윤리적 함의에 대해서도 논의하고 있습니다.



### Multimodal Function Vectors for Spatial Relations (https://arxiv.org/abs/2510.02528)
- **What's New**: 이번 논문에서는 Large Multimodal Models (LMMs)이 어떻게 공간 관계의 표현을 전송하는지에 대한 새로운 통찰을 제공합니다. OpenFlamingo-4B와 같은 비전-언어 모델의 특정 attention heads가 관계 예측에 중대한 영향을 미친다는 것을 발견했습니다. 우리는 causal mediation analysis를 활용하여 이러한 attention heads와 multimodal function vectors를 식별하고, 이를 통해 LMM의 성능을 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: 기술적으로, 연구진은 function vectors (FVs)라는 접근 방식을 채택하여 LMM 내부의 공간 관계를 인코딩하는 구조를 추출하고 조작합니다. 함수 벡터는 적은 수의 attention heads에서 활성화된 값을 평균하여 생성하며, 모델의 은닉층에 삽입될 수 있습니다. 이를 통해 주어진 작업 이전에 시연 없이도 모델이 원하는 행동을 생성할 수 있게 됩니다.

- **Performance Highlights**: 결과적으로, 우리는 fine-tuned function vectors가 LMM의 in-context learning 기준보다 월등한 성능을 발휘한다는 것을 증명했습니다. 또한, 관계-specific function vectors를 선형 결합해 새롭고 훈련되지 않은 공간 관계를 해결하는 데 성공하여, 이 접근 방식의 강력한 일반화 능력을 강조합니다. 이러한 발견은 LMM의 관계 추론 능력에 대한 이해를 선진화시키는 데 기여합니다.



### Safe and Efficient In-Context Learning via Risk Contro (https://arxiv.org/abs/2510.02480)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)의 안전성을 높이기 위한 새로운 접근법을 제안합니다. 특히, 부정확하거나 악의적인 예시가 모델 성능에 미치는 영향을 최소화하고자 합니다. Zero-shot 모델을 기준으로 하여, Distribution-Free Risk Control (DFRC)를 적용하여, 불리한 입력이 성능을 저하시킬 수 있는 정도를 제어합니다.

- **Technical Details**: 연구는 입력 x에 대한 레이블 y를 바탕으로 LLM이 상황별 예시 세트(c)를 통해 학습하는 과정을 다룹니다. LLM이 모든 레이어를 통과한 후 예측을 하지만, Early-Exit LLM에서는 각 레이어에서 예측할 수 있는 선택지를 제공합니다. 이를 통해 위험 제어 프레임워크를 적용하여 적절한 예측 기준을 선택할 수 있도록 합니다. 또한, 안전성을 높이기 위해 세 가지 기여를 제안합니다: 안전한 zero-shot 기준을 사용하는 초기 종료 모델의 새로운 구성, 과도한 학습(overthinking)을 측정하는 새로운 ICL(loss) 손실 설계, 그리고 안전성과 효율성을 모두 고려한 Learn-then-Test (LTT) 위험 제어 프레임워크의 단순한 수정입니다.

- **Performance Highlights**: 연구 결과는 제안된 방법이 부정적인 상황에서도 안전성이 보장되고 도움이 되는 예시에서 성능을 극대화할 수 있음을 보여줍니다. 실험을 통해 성능 속도를 50% 이상 향상시키면서도, zero-shot 성능 기준에 비해 출력 안전성을 보장할 수 있음을 입증하였습니다. 8개의 다양한 벤치마크 작업과 4개의 모델을 통한 실험 결과, 이 프레임워크가 안전성을 효과적으로 보장하고 높은 효율성을 달성할 수 있는 첫 사례로 확인되었습니다.



### RefineShot: Rethinking Cinematography Understanding with Foundational Skill Evaluation (https://arxiv.org/abs/2510.02423)
- **What's New**: 본 연구에서는 ShotBench 벤치마크의 문제가 분석되었고, RefineShot이라는 새로운 기준이 제안되었습니다. RefineShot은 기존의 다중 선택 옵션을 체계적으로 재구성하여 평가의 신뢰도를 높이고 공정한 비교를 촉진합니다. 이 과정은 촬영 기법 이해의 평가를 개선하는 방향으로 이루어졌습니다.

- **Technical Details**: ShotBench는 8개의 촬영 차원에서 3,500개 이상의 전문가 주석 다중 선택 질문을 제공하는 벤치마크입니다. 그러나 이 연구에서는 옵션 설계의 애매함과 ShotVL의 추론 일관성 결여가 평가 신뢰도를 저하시킨다고 보고하였습니다. RefineShot는 이러한 문제를 해결하기 위해 다중 선택 옵션을 일관성 있게 재구성하고 새로운 평가 프로토콜을 도입하였습니다.

- **Performance Highlights**: ShotVL은 여러 카테고리에서 최첨단 성능을 달성했지만, 평가의 신뢰성은 그 설계 품질에 달려 있습니다. 본 연구는 ShotVL의 성능과 추론 신뢰성 사이의 불일치를 밝히고, 데이터 세트의 평가를 보다 균형있게 하여 향후 발전의 기초를 마련합니다. RefineShot는 이러한 새로운 평가 방식을 통해 촬영 기법 이해에 대한 더 신뢰할 수 있는 평가를 가능하게 합니다.



### BrowserArena: Evaluating LLM Agents on Real-World Web Navigation Tasks (https://arxiv.org/abs/2510.02418)
- **What's New**: 최근 LLM 웹 에이전트가 개방형 웹에서 작업을 수행할 수 있도록 발전하며, BrowserArena라는 새로운 평가 플랫폼이 도입되었습니다. 이 플랫폼은 사용자 제출 작업을 기반으로 하여 에이전트 성능을 실시간으로 평가하며, Arena 스타일의 헤드 투 헤드 비교를 통해 에이전트의 실패 모드를 식별합니다. 이를 통해 사용자는 에이전트가 실제 웹 작업에서 어떻게 수행되는지를 보다 정확히 분석하고 이해할 수 있습니다.

- **Technical Details**: BrowserArena는 다양한 사용자 제출 작업을 처리하고, 두 개의 무작위로 선택된 LLM을 사용하여 웹사이트를 탐색합니다. 이 플랫폼은 Chatbot Arena을 기반으로 하며, 사용자로부터 수집한 피드백을 통해 특정 단계에서 LLM의 행동을 검토합니다. 주요 실패 모드로는 captcha 해결, 팝업 배너 제거, URL로의 직접 탐색이 있으며, 다양한 언어 모델이 이 실패 모드를 해결하기 위해 사용하는 전략이 다름을 발견했습니다.

- **Performance Highlights**: 연구 결과, o4-mini는 captcha 해결에서 더 다양한 전략을 구사하지만, DeepSeek-R1은 사용자를 잘못 안내하는 경향이 발견되었습니다. 이 연구는 웹 에이전트의 성능을 평가하는 새로운 접근 방식을 마련하며, 공개된 데이터셋 및 코드베이스를 통해 LLM의 성능 평가에 기여할 것입니다. 또한, VLM(vision-language models)의 인간 선호 모델링 능력의 한계를 드러내며, 사용자 피드백을 통한 세밀한 행동 분석의 중요성을 강조합니다.



### Reward Models are Metrics in a Trench Coa (https://arxiv.org/abs/2510.03231)
- **What's New**: 본 논문은 대형 언어 모델의 후속 교육에서 강화 학습(reinforcement learning, RL)의 부상과 이에 따른 보상 모델(reward models)의 중요성을 조명합니다. 연구 분야 간의 단절로 인해 유사한 용어와 문제들이 반복되고 있는데, 이를 통해 보상 모델과 평가 메트릭(evaluation metrics)이 서로 보완할 수 있는 기회를 제시합니다. 보상 모델과 평가 메트릭 간의 협력이 필요하다는 주장을 하며, 이를 통해 보상 해킹(reward hacking)과 같은 문제를 극복할 수 있는 방안을 제안합니다.

- **Technical Details**: 강화 학습(REINFORCE 및 minimum risk training 등)은 생성 모델의 훈련 및 평가에 있어 중요한 역할을 하며, 기존 메트릭을 통해 복잡한 상태 공간을 다루는 방식을 설명합니다. RL은 학습 과정에서 발생하는 노출 편향(exposure bias)을 극복하기 위한 자연스러운 방법으로, 다양한 사례에서 활용되고 있습니다. 다양한 소스에서 생성된 언어의 품질을 평가하는 메트릭들은 과거에 비해 의미적 유사성을 측정하게 되어 보상 해킹을 줄이는 데 기여하고 있습니다.

- **Performance Highlights**: 연구에서는 보상 모델이 특정 작업에서 메트릭보다 부족한 성과를 나타낸다는 실험 결과를 제시합니다. 이러한 발견들을 기반으로, 주관적 판단과 머신 학습 기반 판단의 상관관계를 개선할 수 있는 기회를 포착할 필요성이 강조됩니다. 또한, 보상 모델과 평가 메트릭이 함께 발전할 수 있는 다양한 연구 주제와 협력의 중요성을 명확히 합니다.



### Improving GUI Grounding with Explicit Position-to-Coordinate Mapping (https://arxiv.org/abs/2510.03230)
- **What's New**: 최근의 GUI grounding 연구에서는 자연어 지시사항을 정확한 픽셀 좌표로 변환하는 과정이 중요하게 다루어졌습니다. 본 논문은 고해상도 디스플레이에서의 좌표 예측의 신뢰성 문제를 해결하기 위해 RULER(Rotary position-to-pixeL mappER) 토큰과 I-MRoPE(Interleaved Multidimensional Rotary Positional Encoding)를 제안합니다. 이러한 접근을 통해 GUI 자동화의 정확성을 향상시키고 다양한 해상도와 플랫폼에서의 적용 가능성을 확대합니다.

- **Technical Details**: RULER 토큰은 모델 내에 명시적인 좌표 참조 시스템을 구축하여 픽셀 좌표를 직접 인코딩합니다. I-MRoPE는 공간 차원에서 주파수 불균형 문제를 해결하여 높이와 너비 균형 잡힌 공간 표현을 제공합니다. 이 두 가지 혁신은 모델의 정확성을 높이고 좌표 예측의 불안정성을 해결하는 데 기여합니다.

- **Performance Highlights**: 본 연구는 ScreenSpot 및 ScreenSpot-Pro 벤치마크에서의 평가를 통해 GUI grounding 정확도를 크게 향상시켰습니다. 특히, 고해상도 디스플레이에서의 성능이 31.1%에서 37.2%로 증가하는 등 강력한 일반화 능력을 보여주었습니다. RULER 토큰은 추가적인 계산 비용이 거의 없이 8K 디스플레이에서도 효과적입니다.



### Test-Time Defense Against Adversarial Attacks via Stochastic Resonance of Latent Ensembles (https://arxiv.org/abs/2510.03224)
- **What's New**: 이 논문은 적대적 공격(adversarial attacks)에 대한 테스트 시간 방어 메커니즘을 제안합니다. 기존의 방법들은 정보 손실(information loss)을 초래할 수 있는 피처 필터링(feature filtering) 또는 스무딩(smoothing)에 의존하는 반면, 저자는 소음(noise)을 소음으로 상쇄하는 방법을 사용하여 강건성을 높이고 정보 손실을 최소화합니다. 이 기법은 입력 이미지에 소규모 변환적 섭동을 추가하고 변환된 피처 임베딩(feature embeddings)을 정렬하여 집계한 후 원본 이미지로 다시 매핑하는 방식을 사용합니다.

- **Technical Details**: 저자들은 스토캐스틱 공진(stochastic resonance) 기술을 활용하여 적대적 섭동의 영향을 제거하는 방법을 제안합니다. 이를 통해 입력 데이터를 대체하거나 임베딩을 평균화하지 않고, 변환된 임베딩을 잠재 공간(latent space)에서 평균화하는 방식으로 적대적 섭동의 영향을 줄입니다. 이 방법은 별도의 네트워크 모듈 없이 다양한 기존 네트워크 아키텍처에 배포될 수 있으며, 특정 공격 유형을 위한 미세 조정(fine-tuning) 없이 동작합니다.

- **Performance Highlights**: 이 방법은 이미지 분류(image classification), 스테레오 매칭(stereo matching), 광학 흐름(optical flow)과 같은 다양한 작업에서 검증되었습니다. 결과적으로, 깨끗한 성능(clean performance)에 비해 이미지를 분류하는 동안 최대 68.1%의 정확도 손실을 회복하고, 스테레오 매칭에서는 71.9%, 광학 흐름에서는 29.2%를 회복하였습니다. 이 방법은 기존의 어떤 기법보다도 강력한 강건성을 제공하며, 밀집 예측(dense prediction) 작업에 대한 일반적인 테스트 시간 방어(Generic test-time defense)를 수립하였습니다.



### Self-Anchor: Large Language Model Reasoning via Step-by-step Attention Alignmen (https://arxiv.org/abs/2510.03223)
- **What's New**: 이 논문은 Self-Anchor라는 혁신적인 파이프라인을 제안하여 LLM(대형 언어 모델)의 주의를 유도한다. Self-Anchor는 복잡한 추론 문제를 구조화된 계획으로 분해하고, 모델의 주의를 가장 관련성 높은 추론 단계에 자동으로 정렬하여 생성 과정에서 초점을 유지할 수 있게 한다. 이 방법은 기존의 SOTA(최첨단) 프롬프트 방법보다 우수한 성능을 보이며, 비전략 모델과 전문화된 추론 모델 간의 성능 격차를 크게 줄일 수 있는 가능성을 지닌다.

- **Technical Details**: Self-Anchor는 복잡한 문제를 계획 단계로 나눈 후, 각 계획 단계가 해당하는 추론과 연관되어 주의 정렬을 유도할 수 있도록 설계되었다. 이 과정에서 SPA(Selective Prompt Anchoring)라는 낮은 비용의 주의 유도 메커니즘을 적용하여 주의 토큰 선택을 자동화한다. 또한, 주의 정렬 강도는 모델의 신뢰도에 따라 동적으로 조정되어 생성 과정에서 민첩하게 변화하는 토큰에 적절하게 주의를 기울이도록 한다.

- **Performance Highlights**: Self-Anchor는 GSM8K, AQuA, MATH 등 6개의 벤치마크에서 SOTA 프롬프트 방법들과 비교하여 모든 설정에서 평균적으로 5.44% 이상의 정확도 향상을 보였다. 또한, Self-Anchor는 5개의 전문 추론 모델과 동등한 성능을 보이면서도 훨씬 낮은 비용과 복잡성으로 실행 가능하다는 점에서 LLM의 추론 능력을 향상시키기 위한 실용적인 대안으로 제시된다.



### Abstain and Validate: A Dual-LLM Policy for Reducing Noise in Agentic Program Repair (https://arxiv.org/abs/2510.03217)
- **What's New**: 이 논문에서는 에이전트 기반의 자동 프로그램 복구(Automated Program Repair, APR) 시스템에서 개발자에게 노이즈를 줄이기 위한 두 가지 정책을 제안합니다. 제안된 정책은 버그 검토를 피하고 패치 검증을 통해 더욱 효율적으로 버그를 해결하도록 돕습니다. 특히, 이 접근법은 인공지능 도구 채택을 위한 개발자 신뢰 구축에 중요한 역할을 합니다.

- **Technical Details**: 버그 검정(Bug abstention) 정책은 LLM을 사용하여 에이전트가 특정 버그를 해결해야할 확률을 예측합니다. 성공 확률이 기준치 이하라면 해당 버그를 무시하여 개발자에게 비효율적인 패치를 보여주지 않습니다. 패치 검증(Validation) 정책은 생성된 패치가 올바른지 평가하는 다단계 과정을 포함하며, LLM 기반 필터를 통해 정확성을 확인합니다.

- **Performance Highlights**: 이 연구는 제안된 두 가지 정책이 결합될 때, Google's 코드베이스에서 174개의 인간 보고 버그에 대해 최대 39%의 성공률 향상을 이루었다고 보고합니다. NPE(Null Pointer Exception)와 같은 기계 보고 버그에서도 평균적으로 성공률을 개선하였으며, 이 정책들이 함께 운영될 때 가장 높은 효과를 보였습니다.



### Wave-GMS: Lightweight Multi-Scale Generative Model for Medical Image Segmentation (https://arxiv.org/abs/2510.03216)
Comments:
          5 pages, 1 figure, 4 tables; Submitted to IEEE Conference for possible publication

- **What's New**: Wave-GMS는 의료 이미지 분할을 위한 경량화되고 효율적인 다중 스케일 생성 모델로, 적은 수의 학습 가능한 파라미터(~2.6M)로 높은 성능을 제공합니다. 이 모델은 메모리 집약적인 사전 훈련된 vision foundation 모델을 로드할 필요가 없으며, 제한된 메모리의 GPU에서 대용량 배치를 지원합니다. 이를 통해 병원 및 의료 시설에서 AI 도구의 보다 공정한 배포를 목표로 하고 있습니다.

- **Technical Details**: Wave-GMS는 입력 이미지의 다중 해상도 분해를 통해 고품질의 잠재 표현을 생성하는 학습 가능한 인코더를 사용합니다. 이 모델은 고도로 압축된 Tiny-VAE를 활용하여 입력 이미지 및 분할 마스크의 잠재 표현을 생성하며, Latent Mapping Model (LMM)이 다중 해상도 잠재 표현과 해당 분할 마스크 간의 매핑을 학습합니다. 모든 훈련 중 Tiny-VAE의 인코더와 디코더는 고정되어 있으며, 가벼운 인코더와 LMM만 학습 가능하여 트래이너블 파라미터 수를 크게 줄입니다.

- **Performance Highlights**: Wave-GMS는 BUS, BUSI, Kvasir-Instrument 및 HAM10000과 같은 네 가지 공개 데이터셋에서 광범위한 실험을 통해 최첨단 분할 성능을 달성하였으며, 뛰어난 교차 도메인 일반화 능력을 보여주었습니다. 이는 팀의 기존 모델보다 더 적은 수의 파라미터로 높은 성능을 유지할 수 있도록 하였고, 특히 작은 데이터셋에서의 과적합을 방지하는 데 기여하고 있습니다.



### Simulation to Rules: A Dual-VLM Framework for Formal Visual Planning (https://arxiv.org/abs/2510.03182)
Comments:
          30 pages, 5 figures, 5 tables

- **What's New**:  이번 논문에서는 VLMFP라는 새로운 프레임워크를 제안하여, 시각적 계획을 위한 PDDL(Planning Domain Definition Language) 문제 파일과 도메인 파일을 자율적으로 생성합니다. 해당 프레임워크는 두 개의 VLM(Vision Language Model)인 SimVLM과 GenVLM을 통합하여 PDDL 파일 생성을 더욱 신뢰성 있게 만듭니다. 특히, SimVLM은 시뮬레이션을 통해 행동 결과를 예측하고, GenVLM은 이러한 예측 결과를 바탕으로 PDDL 파일을 생성 및 반복적으로 수정합니다.

- **Technical Details**:  VLMFP는 시각적 입력을 통해 문제와 도메인을 생성하기 위해 객체 인식, 공간 이해, 추론 및 PDDL 지식이 요구됩니다. 이 프레임워크는 시나리오를 정확하게 설명하고, 행동을 시뮬레이션하며 목표 도달 여부를 판단하는 SimVLM과 일반적인 추론 및 방대한 PDDL 지식을 활용하는 GenVLM으로 구성됩니다. 두 모델의 조화를 통해 고차원적인 생성 가능성을 달성하며, 생성된 도메인 PDDL은 동일한 도메인의 모든 인스턴스에 재사용할 수 있습니다.

- **Performance Highlights**:  VLMFP는 6개의 격자 세상(grid-world) 도메인에서 평가되었으며, SimVLM은 시나리오 설명 정확도에서 95.5%의 성과를 달성했습니다. 또한, VLMFP는 GPT-4o를 GenVLM으로 활용하여 보지 않은 인스턴스에 대해 각각 70.0%와 54.1%의 유효한 계획(successful plans) 성공률을 기록하며, 기존 방법들보다 뛰어난 성능을 보였습니다.



### Topic Modeling as Long-Form Generation: Can Long-Context LLMs revolutionize NTM via Zero-Shot Prompting? (https://arxiv.org/abs/2510.03174)
- **What's New**: 이번 연구는 기존의 전통적인 주제 모델링(Topic Modeling, TM) 접근 방식을 대체하여, 대규모 언어 모델(LLM)을 활용한 새로운 패러다임을 제시합니다. 저자들은 LLM 기반의 주제 모델링을 긴 텍스트 생성 작업으로 재정의하고, 이를 통해 효과적으로 주제를 수집하고 관련 텍스트를 생성하는 방법을 제안했습니다. 이를 통해 NTMs(Neural Topic Models)와 LLM의 성능 차이를 분석하고, LLM이 NTMs에 비해 우수할 수 있음을 조명합니다.

- **Technical Details**: 연구에서는 전통적인 TM 모델, 특히 LDA(잠재 디리클레 할당)와 NTMs의 한계를 지적하며, 이를 극복하기 위한 새로운 접근 방식을 설명합니다. 잠재적 주제 분포를 학습하기 위한 세 가지 주요 단계를 포함하는 방법론을 제시하며, 데이터셋 전처리, 주제 생성, 텍스트 할당의 과정을 통합합니다. 특히, LLM의 긴 컨텍스트 윈도우를 활용하여 보다 효과적으로 주제를 추출하고 텍스트를 할당하는 절차를 마련했습니다.

- **Performance Highlights**: 실험 결과, LLM 기반의 모델이 NTMs에 비해 주제 품질이 개선되는 경향을 보였습니다. 특히, 품질 평가에서 LLM이 더 나은 성능을 제공하는 것으로 드러났으며, 기존 NTMs의 다수는 시대에 뒤떨어져 있다는 주장에 대한 실증적 근거를 제시했습니다. 연구자들은 이러한 발견이 LLM 기반의 접근 방식이 기존의 TM 방식에 비해 적절히 주제를 분석하고 생성할 수 있음을 보여준다고 결론짓습니다.



### UniShield: An Adaptive Multi-Agent Framework for Unified Forgery Image Detection and Localization (https://arxiv.org/abs/2510.03161)
- **What's New**: 본 연구는 이미지 생성 기술의 빠른 발전에 따른 합성 이미지의 현실성 증가가 정보 무결성과 사회적 안전성에 미치는 영향을 다루고 있습니다. 이를 해결하기 위한 새로운 시스템인 UniShield를 제안하여 이미지 위조 탐지 및 위치 확인(Forgery Image Detection and Localization, FIDL)의 중요성을 강조합니다. UniShield는 다양한 도메인에서 이미지 위변조를 탐지하고 국지화할 수 있는 멀티 에이전트 기반의 통합 시스템입니다.

- **Technical Details**: UniShield는 두 가지 주요 에이전트, 즉 인식 에이전트(perception agent)와 탐지 에이전트(detection agent)를 통합하여 작동합니다. 인식 에이전트는 이미지 특징을 분석하여 적합한 탐지 모델을 동적으로 선택하고, 탐지 에이전트는 다양한 전문 탐지기를 통합하여 통합 프레임워크를 구축합니다. 이 시스템은 AI 생성 이미지, DeepFake, 문서 조작 등 다양한 이미지 조작 형태를 처리할 수 있도록 설계되었습니다.

- **Performance Highlights**: UniShield는 광범위한 실험을 통해 기존 통합 접근 방식 및 도메인 특정 탐지 방법을 초월하는 최첨단 결과를 보여주었습니다. 이는 UniShield의 우수한 실용성, 적응성, 그리고 확장성을 강조하며, 실제 적용 가능성을 높이고 사회적 위험을 줄이는 데 기여할 수 있을 것으로 기대됩니다.



### SpineBench: A Clinically Salient, Level-Aware Benchmark Powered by the SpineMed-450k Corpus (https://arxiv.org/abs/2510.03160)
- **What's New**: 이번 논문에서는 619억의 사람들이 영향을 받는 척추 질환에 대한 AI 기반 진단의 한계를 극복하기 위한 SpineMed라는 새로운 생태계를 소개합니다. SpineMed-450k는 45만 건 이상의 지침 인스턴스를 포함하는 대규모 데이터셋을 제공하며, X-ray, CT, MRI 등의 이미징 기법에서 특정 척추 레벨에 대한 의사 결정을 지원합니다. SpineBench는 임상적으로 검증된 평가 프레임워크를 통해 AI 모델의 성능을 평가하는 데 도움을 줍니다.

- **Technical Details**: SpineMed-450k 데이터셋은 임상 의사의 피드백을 바탕으로 수집된 다양한 자료를 통해 구성되었습니다. 이 데이터는 교과서, 수술 지침서, 전문가 합의 등을 통해 고품질, 특화된 정보를 제공합니다. 데이터의 질과 추적 가능성을 보장하기 위해 'clinician-in-the-loop' 파이프라인과 두 단계의 LLM 생성 방법을 사용하여 데이터를 정제했습니다.

- **Performance Highlights**: 여러 최신 대형 비전-언어 모델(LVLM)을 SpineBench에서 평가한 결과, 세밀하고 레벨 별 진단에서 체계적인 약점이 발견되었습니다. 반면, SpineMed-450k로 미세 조정된 SpineGPT 모델은 모든 작업에서 일관된 개선을 보여주었으며, 이 모델의 결과는 임상 평가를 통해 진단의 명확성과 실용성을 확인했습니다.



### Stimulus-Voltage-Based Prediction of Action Potential Onset Timing: Classical vs. Quantum-Inspired Approaches (https://arxiv.org/abs/2510.03155)
- **What's New**: 본 논문에서는 신경 세포의 행동 신호 처리를 이해하기 위한 필수 요소인 액션 포텐셜(AP) 발생 타이밍의 정확한 모델링을 제안합니다. 연구자들은 전통적인 LIF(leaky integrate-and-fire) 모델의 한계를 극복하기 위해 QI-LIF(quantum-inspired leaky integrate-and-fire) 모델을 도입하여 AP 발생을 확률적 사건으로 간주하며, 이를 통해 생리적 변동성과 불확실성을 포괄적으로 설명합니다. 특히, 실제 실험 데이터를 기반으로 하여 QI-LIF 모델이 기존 LIF 모델에 비해 강한 자극에 대한 예측 오류를 크게 줄인다는 점을 강조합니다.

- **Technical Details**: 연구에서 QI-LIF 모델은 AP 발생 타이밍을 가우시안 확률 분포로 모델링하며, 여기서 타이밍의 중심은 LIF 또는 QLIF 모델에 의해 제시된 가장 가능성 있는 AP 시작 시간에 위치합니다. 이 모델은 자극의 세기에 따라 변하는 동적인 시간 상수를 적용하여, 자극 강도가 증가함에 따라 유효한 막 시간 상수가 단축되는 효과를 포착합니다. LIF 모델과 QI-LIF 모델 간의 비교 분석을 통해 QI-LIF 모델이 더 나은 성능을 보임을 확인했습니다.

- **Performance Highlights**: QI-LIF 모델은 강한 자극 조건에서 AP 발생 타이밍 예측의 정확도를 유의미하게 개선하여 관찰된 생리학적 반응과 밀접하게 일치합니다. 연구 결과는 양자 영감 컴퓨팅 프레임워크의 가능성을 보여주며, 신경 모델링의 정확도를 높이는 데 기여할 수 있습니다. 이 검증된 접근 방식은 양자 스파이킹 신경망(quantum spiking neural network) 개발로 이어져, 향후 금융 시계열 및 복잡한 패턴 분류와 같은 시간 패턴 인식 응용에서도 활용될 가능성이 큽니다.



### Signature-Informed Transformer for Asset Allocation (https://arxiv.org/abs/2510.03129)
- **What's New**: 이번 연구에서 제안한 Signature-Informed Transformer (SIT)는 포트폴리오 자산 할당(policy allocation)을 직접 최적화하여 강건한 분산 포트폴리오를 구축하는 혁신적인 딥러닝 프레임워크입니다. SIT는 기하학적 표현을 위한 Rough Path Signatures를 사용하여 자산의 복잡한 동역학을 잘 포착하며, 자산 간의 선행-후행 관계에 대한 금융적 귀납적 편향(inductive bias)을 도입합니다. 이 방법은 전통적인 모델 대비 향상된 성능을 보여주며, 특히 예측-최적화를 사용하는 모델에 비해 결정적 우위를 가집니다.

- **Technical Details**: SIT의 주요 혁신점은 세 가지 기둥으로 구성된 통합 아키텍처에 있습니다. 첫째, Path-wise Feature Representation을 통해 각 자산의 가격 이력을 사용하여 특징을 생성하고, 둘째, Signature-Augmented Attention 메커니즘을 통해 자산 쌍 간의 선행-후행 관계를 모델링합니다. 마지막으로, Decision Alignment 기법을 통해 포트폴리오 구성 목표와의 일치를 극대화하며, 이는 조건부 가치-at-위험(Conditional Value-at-Risk, CVaR) 최적화를 통해 이루어집니다.

- **Performance Highlights**: SIT는 S&P 100 지수의 일일 주식 데이터를 평가한 결과 전통적인 및 딥러닝 기반의 모델을 능가하는 성능을 보였습니다. 특히, 위험 인식 자산 할당에 있어 포트폴리오 인식 목표와 기하학적 귀납적 편향이 필수적임을 보여줍니다. 이러한 결과는 SIT의 강력한 성능을 뒷받침하며, 복잡한 금융 시장 환경에서도 안정적인 결과를 도출할 수 있음을 나타냅니다.



### HAVIR: HierArchical Vision to Image Reconstruction using CLIP-Guided Versatile Diffusion (https://arxiv.org/abs/2510.03122)
- **What's New**: 인간의 뇌에서의 시각 정보 복원은 신경과학(Neuroscience)과 컴퓨터 비전(Computer Vision) 간의 통합을 촉진합니다. 하지만 복잡한 시각 자극을 정확하게 복원하는 데 어려움을 겪고 있습니다. HAVIR 모델은 시각 피질의 계층적 표현 이론에서 영감을 받아, 구조적 정보와 의미적 정보를 각각 독립적으로 추출합니다.

- **Technical Details**: HAVIR 모델은 두 가지 주요 구성 요소, 즉 Structural Generator와 Semantic Extractor로 이루어져 있습니다. Structural Generator는 공간 처리(Spatial Processing) 복셀에서 구조 정보를 추출하고, 이 정보를 latent diffusion priors로 변환합니다. Semantic Extractor는 의미 처리(Semantic Processing) 복셀을 CLIP 임베딩(CLIP Embeddings)으로 변환하여 두 영역의 정보를 통합합니다.

- **Performance Highlights**: 실험 결과, HAVIR는 복잡한 장면에서도 구조적 및 의미적 품질을 향상시켜, 기존 모델들과 비교하여 성능이 뛰어난 것으로 나타났습니다. 이는 HAVIR 모델이 구조적 피처와 의미적 정보를 분리하여 최종 이미지를 생성하는 "divide and conquer" 전략 덕분입니다.



### Distilled Protein Backbone Generation (https://arxiv.org/abs/2510.03095)
- **What's New**: 이 논문은 단백질 디자인의 새로운 접근법을 제시하며, diffusion 및 flow 기반 생성 모델들이 단백질 backbone 생성을 위해 어떻게 활용될 수 있는지를 보여줍니다. 최근의 기술 발전에도 불구하고, 이들 모델의 샘플링 속도는 여전히 큰 문제로 남아 있습니다. 이에 따라, score distillation (점수 증류) 기법을 적용하여 적은 샘플링 단계로 성능을 유지하면서 샘플링 시간을 비약적으로 줄일 수 있는 방법을 모색했습니다.

- **Technical Details**: 연구진은 Score identity Distillation (SiD) 기법을 기반으로 한 새로운 증류 프레임워크를 개발하여, 단백질 backbone 생성기를 훈련시켰습니다. 이 프레임워크는 샘플링 시 낮은 온도로 작업하여, 불필요한 구조적 오류를 최소화하고 디자인 가능성을 높입니다. 결과적으로, 16단계 생성을 통해 기존 teacher model과 유사한 성능을 유지하면서도 20배 이상의 샘플링 시간 개선을 이루었습니다.

- **Performance Highlights**: 저자들은 제안한 방식으로 생성된 단백질 구조가 디자인 가능성, 다양성 및 독창성을 유지하였음을 입증하였습니다. 이로 인해, 기존의 diffusion 기반 모델보다 훨씬 빠른 샘플링 속도를 달성했습니다. 또한, 이러한 기법은 대규모 단백질 디자인의 가능성을 높여, 실제 단백질 공학 응용에의 활용을 가속화합니다.



### What Drives Compositional Generalization in Visual Generative Models? (https://arxiv.org/abs/2510.03075)
- **What's New**: 본 연구는 시각 생성 모델에서 조합 일반화(compositional generalization)에 영향을 미치는 다양한 디자인 선택이 긍정적 또는 부정적인 방식으로 작용하는지를 체계적으로 분석합니다. 특히, 훈련 목표(training objective)가 이산(discrete) 또는 연속(continuous) 분포에 작용하는지와 훈련 동안 구성 요소 개념에 대한 조건(conditioning)이 얼마나 정보를 제공하는지가 주요하게 작용함을 발견했습니다. 이후, MaskGIT 모델의 이산 손실(discrete loss)을 보조 연속(JEPA-based) 목표로 완화함으로써 조합 성능을 향상시키는 방법을 제안합니다.

- **Technical Details**: 연구에서는 세 가지 주요 구성요소 즉, Tokenizer, Generative model, Conditioning signal을 통해 현대 시각 생성 모델을 분해하고 각 요소가 조합 일반화에 미치는 영향을 살펴봅니다. 각 모델의 아키텍처와 교육 디자인의 선택에 따라 조합 일반화가 어떻게 변화하는지를 체계적으로 비교하기 위해 세 가지 요소에 대해 통제된 실험을 실시하였습니다. 실험 결과, 연속 분포로 훈련된 모델이 이산 분포로 훈련된 모델보다 조합 능력이 더 뛰어난 것으로 나타났습니다.

- **Performance Highlights**: MaskGIT의 분류적 목표를 Joint Embedding Predictive Architecture (JEPA)로 보강함으로써 조합 성능을 명확히 개선할 수 있음을 보여주었습니다. 동시에 JEPA로 훈련된 모델은 더 분리된 중간 표현(intermediate representations)을 제공하고, 이는 예측 기반의 연속 목표가 이산 생성 모델의 내부 구조를 조합성을 유지하는 방향으로 형성할 수 있음을 시사합니다. 이러한 결과는 생성 모델의 조합 및 분해 능력을 향상시키는 방법을 제시하며, 다양한 아키텍처와 훈련 디자인 선택이 생성 품질에 미치는 영향을 강조합니다.



### A Study of Neural Polar Decoders for Communication (https://arxiv.org/abs/2510.03069)
- **What's New**: 이 논문은 Neural Polar Decoders (NPDs)를 최종 사용자 통신 시스템에 적용하고 분석하는 내용을 다루고 있습니다. 기존의 연구가 합성 채널(synthetic channels)에서 NPD의 효과를 입증한 반면, 본 연구는 실제 통신 시스템에 NPD를 확장하고 있습니다. NPD는 OFDM 및 단일 반송파(single-carrier) 통신 시스템을 완성하는 데 적합하도록 조정되었습니다.

- **Technical Details**: NPD는 다양한 채널 조건에서의 견고성을 위해 코드 길이에 대한 비율 조정(rate matching), 고차 변조(higher-order modulations), 실용적인 시스템 요구 사항을 지원할 수 있도록 확장되었습니다. 이 디코더는 메모리를 가진 채널에서 직접 작동하며, 그 구조를 이용해 파일럿(pilots)이나 순환 접두사(cyclic prefix) 없이 더 높은 데이터 전송률을 달성합니다. 비록 NPD는 표준 5G polar decoder보다 계산 복잡도가 높지만, 신경망 구조는 채널 통계(channel statistics)의 효율적인 표현을 가능하게 하여 실용적인 시스템에 적합한 복잡도를 보입니다.

- **Performance Highlights**: 실험 결과에 따르면, NPD는 5G 채널에서 BER, BLER, 및 전송량(throughput) 측면에서 5G polar decoder보다 일관되게 우수한 성능을 보여줍니다. 이러한 개선은 특히 저전송률(low-rate) 및 짧은 블록 구성에서 두드러지며, 이는 5G 제어 채널에서 흔히 나타나는 특성입니다. 또한 NPD를 단일 반송파 시스템에 적용할 경우, 낮은 PAPR로 OFDM과 유사한 성능을 제공하여 5G 채널에서 효과적인 단일 반송파 전송을 가능하게 합니다.



### A Unified Deep Reinforcement Learning Approach for Close Enough Traveling Salesman Problem (https://arxiv.org/abs/2510.03065)
- **What's New**: 이 논문은 주로 관심을 받지 못한 Close-Enough Traveling Salesman Problem (CETSP)을 해결하기 위한 새로운 접근 방식을 제안합니다. CETSP는 NP-hard 문제에서 변형된 것으로, 각 노드의 근처를 방문하는 경우에만 해당 노드가 방문된 것으로 간주됩니다. 이를 해결하기 위해 Markov Decision Process (MDP)로 CETSP를 공식화하고, 유니파이드 이중 디코더 DRL(UD3RL) 프레임워크를 제안합니다.

- **Technical Details**: UD3RL은 노드 선택(node selection)과 웨이포인트 결정(waypoint determination)을 분리하여 의사결정 과정을 구성합니다. 이를 위해, 적응형 인코더는 의미 있는 특성을 추출하는 데 사용되고, 노드 디코더(node-decoder)와 위치 디코더(loc-decoder)가 두 개의 하위 작업을 처리합니다. 또한, k-nearest neighbors(k-NN) 서브그래프 상호작용 전략을 도입하여 공간적 추론을 향상시킵니다.

- **Performance Highlights**: 실험 결과, UD3RL은 기존 방법들에 비해 솔루션 질(solution quality)과 실행 시간(runtime)에서 뛰어난 성능을 발휘하며, 다양한 문제 크기와 반경 유형(예: constant와 random)에서 강력한 일반화 능력을 보여줍니다. UD3RL은 동적 환경에서도 높은 견고성을 유지하며, 여러 실제 상황에 적용 가능성이 높습니다.



### Comparative Analysis of Parameterized Action Actor-Critic Reinforcement Learning Algorithms for Web Search Match Plan Generation (https://arxiv.org/abs/2510.03064)
Comments:
          10 pages, 10th International Congress on Information and Communication Technology (ICICT 2025)

- **What's New**: 본 연구에서는 Soft Actor Critic (SAC), Greedy Actor Critic (GAC), 그리고 Truncated Quantile Critics (TQC) 알고리즘의 성능을 고차원 의사결정 작업에서 평가하였습니다. 파라미터화된 행동(Parameterized Action, PA) 공간을 통해 반복 네트워크를 배제하여, Microsoft NNI를 통해 하이퍼파라미터 최적화를 수행하였습니다. 특히, 빠른 훈련 시간과 높은 보상을 기록한 Parameterized Action Greedy Actor-Critic (PAGAC)이었으며, 이는 복잡한 행동 공간에서의 속도와 안정성에서 분명한 장점을 제공하였습니다.

- **Technical Details**: 연구에서는 완전 관측 가능한 환경 내에서 두 개의 벤치마크인 Platform-v0와 Goal-v0에서 GAC, TQC와 SAC의 성능을 비교하였습니다. 강화학습(Reinforcement Learning, RL) 문제로 매칭 계획 생성을 재구성한 Luo et al.의 접근 방식은 상태 신호와 매칭 계획의 동적 파라미터화를 통해 유연성을 강화하였습니다. 본 연구는 이러한 알고리즘들이 막대한 차원의 연속 행동 공간에서 탐험(exploration)과 착취(exploitation) 간의 균형을 추구하였다는 점을 강조합니다.

- **Performance Highlights**: PAGAC는 Platform 게임에서 5,000 에피소드를 41분 24초, 로봇 축구 골 게임에서는 24분 4초에 완료하며, 다른 알고리즘들보다 빠른 학습 시간과 높은 성능을 자랑하였습니다. 연구 결과 SAC, GAC, TQC 간의 성능 비교는 각기 다른 전략의 효과와 적용 가능성을 제시하였으며, PAGAC은 빠른 수렴과 견고한 성능을 요구하는 과제에 적합함을 입증하였습니다. 향후 연구는 엔트로피 정규화를 조합하여 안정성을 개선하는 하이브리드 전략에 대한 탐사를 고려할 예정입니다.



### Semantic Differentiation in Speech Emotion Recognition: Insights from Descriptive and Expressive Speech Roles (https://arxiv.org/abs/2510.03060)
Comments:
          Accepted to the *SEM conference collocated with EMNLP2025

- **What's New**: 본 연구는 Speech Emotion Recognition (SER) 분야에서 감정 역할에 대한 새로운 관점을 제시합니다. 전통적인 SER 접근 방식에서 벗어나, 발화의 의도된 감정과 유도된 감정 간의 구분을 강조합니다. 참가자들이 영화 클립을 본 후 자신의 감정 반응을 평가하면서, 설명적 의미(descriptive semantics)와 표현적 의미(expressive semantics)를 구분하는 방법을 제안하였습니다.

- **Technical Details**: 연구 방법론은 세 가지 주요 단계로 구성됩니다: 자동 음성 인식(ASR)을 통한 음성 전사, 대규모 언어 모델(LLMs)을 활용한 의미 기반 구분, 텍스트 분류기/회귀기를 이용한 감정 예측입니다. 데이터셋은 여섯 가지 감정 범주에 걸쳐 582개의 오디오 녹음으로 구성되며, 의도된 감정과 유도된 감정이 각각 측정됩니다. 이 과정에서 설명적 세그먼트는 의도된 감정과 더 높은 일치를 보이며, 표현적 세그먼트는 유도된 감정과 잘 연관된다는 것을 보여줍니다.

- **Performance Highlights**: 이 연구의 결과는 감정 인식 시스템 설계에 중요한 함의를 제공합니다. 설명적 의미와 표현적 의미의 구분은 SER 연구의 기존 한계를 극복할 수 있는 원리를 제공합니다. 이 접근법을 통해 감정 인식의 정확도를 높일 수 있으며, 가상 비서나 심리 건강 지원 시스템과 같은 다양한 응용 분야에서 활용될 수 있는 가능성을 보여줍니다.



### ZeroShotOpt: Towards Zero-Shot Pretrained Models for Efficient Black-Box Optimization (https://arxiv.org/abs/2510.03051)
- **What's New**: 본 논문에서는 ZeroShotOpt라는 새로운 범용 프리트레인(pretrained) 모델을 소개합니다. 이 모델은 2D에서 20D까지의 연속적인 블랙박스 최적화(tasks) 문제들을 해결하는 데 사용됩니다. 기존의 베이지안 최적화(Bayesian Optimization, BO) 기술의 한계를 극복하기 위해, 우리는 방대한 최적화 경로 데이터를 통해 오프라인 강화 학습(offline reinforcement learning)을 활용합니다.

- **Technical Details**: ZeroShotOpt는 200200M 파라미터를 가진 트랜스포머 기반(transformer-based) 모델로, 다양한 BO 변형들에서 수집된 20M 이상의 합성 함수(synthetic functions) 경로를 바탕으로 사전 학습되었습니다. 이를 통해 모델은 최적화 동역학을 robust하게 이해하고, 낮은 평가 예산(low evaluation budget)에서도 효율적으로 동작할 수 있는 기반을 제공합니다. 또한, ZeroShotOpt는 이전의 트랜스포머 모델들보다 더 나은 zero-shot 일반화를 보여줍니다.

- **Performance Highlights**: ZeroShotOpt는 다양한 unseen global optimization benchmarks에서 뛰어난 성능을 발휘하며, 기존의 선두적인 BO 방법들과 유사한 수준의 샘플 효율성을 달성합니다. 이 모델은 최소한의 조정으로도 후속 확장 및 개선을 위한 견고한 기반을 제공합니다. 연구 결과는 github에 오픈소스로 공개되어 다른 연구자들이 접근하고 활용할 수 있도록 했습니다.



### When and Where do Events Switch in Multi-Event Video Generation? (https://arxiv.org/abs/2510.03049)
Comments:
          Work in Progress. Accepted to ICCV2025 @ LongVid-Foundations

- **What's New**: 본 논문에서는 텍스트-비디오(T2V) 생성 분야에서 다중 이벤트 전환을 제어하는 메커니즘을 연구합니다. 특히, 생성 과정에서 이벤트 프롬프트를 얼마나 일찍, 그리고 모델의 어느 층에서 주입해야 하는지를 탐구합니다. 이를 위해 MEve라는 새로운 프롬프트 세트를 소개하여 다중 이벤트 T2V 모델의 평가를 체계적으로 진행합니다. 다양한 모델에서 초반의 개입과 블록 별 모델 레이어의 중요성이 드러났습니다.

- **Technical Details**: 연구는 확산 모델(diffusion model)을 기반으로 하며, 두 개의 이벤트를 연결하는 방식으로 프롬프트를 설정합니다. 이벤트 전환의 적절한 타이밍(denoising steps)과 깊이(model layer depth)에 따라 프롬프트를 조건화했습니다. 연구 결과, 초기 30%의 denoising 단계에서 이벤트 프롬프트를 노출하는 것이 중요하며, 얕은 블록이 전반적인 의미(global semantics)에 영향을 미친다는 사실이 밝혀졌습니다.

- **Performance Highlights**: 실험을 통해 CogVideo와 OpenSora 두 모델 패밀리의 성능을 평가했습니다. MEve 데이터셋을 사용하여 다중 이벤트 생성을 검증했으며, 모델들은 본래의 프레임 수에 맞춰 비디오를 생성하고 고유한 성능에 따라 프레임을 할당했습니다. 이 연구는 다중 이벤트 조건부 생성의 가능성을 강조하며, 향후 연구 방향에 기여할 수 있는 중요한 인사이트를 제공합니다.



### CHORD: Customizing Hybrid-precision On-device Model for Sequential Recommendation with Device-cloud Collaboration (https://arxiv.org/abs/2510.03038)
Comments:
          accepted by ACM MM'25

- **What's New**: 이번 연구에서는 CHORD라는 새로운 프레임워크를 제안합니다. CHORD는 하이브리드 정밀도를 가진 온디바이스 모델을 커스터마이징하여 장치-클라우드 협력 방식을 통해 순차적 추천을 구현합니다. 이를 통해 모델 정확도를 유지하면서 자원 적응형 배포를 달성할 수 있습니다.

- **Technical Details**: CHORD에서는 채널별 혼합정밀도 양자화(channel-wise mixed-precision quantization)를 활용하여 사용자 맞춤형 추천을 제공합니다. 연구에서는 모델 파라미터의 민감도 분석을 통해 사용자 프로필에 맞는 양자화 전략을 쉽게 매핑하는 방법을 개발하였습니다. 이 과정에서 모델 압축을 달성하면서도 커뮤니케이션 오버헤드를 줄이는데 집중하였습니다.

- **Performance Highlights**: 세 가지 실제 데이터셋과 두 가지 인기 있는 네트워크(SASRec 및 Caser)에 대한 실험 결과, CHORD는 정확성, 효율성, 적응성을 모두 입증했습니다. 사용자는 2비트의 전략 인코딩을 통해 클라우드와의 소통 비용을 크게 줄이고, 한 번의 전방 패스를 통해 개인화된 모델 적응을 이룰 수 있음을 보였습니다.



### Investigating The Smells of LLM Generated Cod (https://arxiv.org/abs/2510.03029)
- **What's New**: 본 연구는 大型 언어 모델(LLMs) 이 생성한 코드의 품질 평가를 위한 시나리오 기반 방법을 제안합니다. 기존의 연구들은 주로 생성 코드의 기능적 정확성에 초점을 맞췄으나, 본 연구는 코드 품질에 대한 새로운 통찰을 제공합니다. LLM이 생성한 코드의 품질 평가에서 가장 취약한 시나리오를 식별하는 것을 목표로 합니다.

- **Technical Details**: 이 방법은 코드 품질의 중요한 지표인 코드 냄새(code smells)를 측정하고, 직업적으로 작성된 코드의 참조 솔루션과 비교합니다. 데이터셋은 코드 주제와 난이도에 따라 다양한 하위 집합으로 나누어 LLM 사용의 다양한 시나리오를 나타냅니다. 또한, 코드 생성에 대한 자동화된 테스트 시스템이 개발되었고, 최신 LLM 4종(예: Gemini Pro, ChatGPT, Codex, Falcon)으로 생성된 Java 프로그램에 대한 실험이 보고됩니다.

- **Performance Highlights**: 실험 결과, LLM에 의해 생성된 코드는 참조 솔루션에 비해 코드 냄새가 더 높은 빈도로 발생하는 것으로 나타났습니다. Falcon은 코드 냄새 증가율이 42.28%로 가장 낮았으며, 그 뒤를 Gemini Pro (62.07%), ChatGPT (65.05%), Codex (84.97%)가 따랐습니다. 평균적으로 모든 LLM에서 코드 냄새가 63.34% 증가했으며, 복잡한 코딩 작업과 고급 주제에서 증가율이 더 큽니다.



### Learning Robust Diffusion Models from Imprecise Supervision (https://arxiv.org/abs/2510.03016)
- **What's New**: 본 논문에서는 DMIS라는 통합 프레임워크를 제안하여 부정확한 감독(imprecise supervision) 하에서 강건한 확산 모델(Diffusion Models) 학습을 가능하게 합니다. 이는 확산 모델 분야에서 최초로 체계적으로 연구된 사항으로, 학습 목적을 우선 가능도 최대화(likelihood maximization) 문제로 공식화하고, 생성(component)과 분류(component) 요소로 분해하였습니다. 이 접근 방식은 부정확한 라벨 분포를 모델링하고, 확산 분류기를 통해 클래스 후확률(class-posterior probabilities)을 추론하는 과정을 포함합니다.

- **Technical Details**: 프레임워크는 생성 모델링 동안 부정확한 라벨 조건부 점수(imprecise-label conditional score)를 청정 라벨의 조건부 점수(clean-label conditional scores)로 표현할 수 있다는 점에 착안했습니다. 또한, 우리는 효율적인 후확률 추론을 위해 최적화된 시간 단계 샘플링 전략(optimized timestep sampling strategy)을 도입하여 계산 복잡성을 줄였습니다. 이 과정에서 부정확한 라벨링에 대응하는 가중치 저노이즈 점수 매칭(objective)이 제안되어, 깨끗한 주석 없이도 라벨 조건 학습이 가능해졌습니다.

- **Performance Highlights**: 다양한 형태의 부정확한 감독에 대한 광범위한 실험 결과, 우리의 프레임워크로 학습한 CDMs는 우수한 생성 품질과 품사 구분(class-discriminative) 샘플을 일관되게 생성함을 보여주었습니다. 이미지 생성, 약한 감독 학습(weakly supervised learning), 노이즈 데이터 세트 응축(noisy dataset condensation)과 같은 여러 작업에서 성능을 입증하며, 미래 연구를 위한 견고한 기준선이 설정되었습니다. 본 연구는 부정확한 감독 하에서의 강건한 CDM 훈련을 위한 통합 프레임워크의 필요성을 강조합니다.



### BrainIB++: Leveraging Graph Neural Networks and Information Bottleneck for Functional Brain Biomarkers in Schizophrenia (https://arxiv.org/abs/2510.03004)
Comments:
          This manuscript has been accepted by Biomedical Signal Processing and Control and the code is available at this https URL

- **What's New**: 이번 연구에서는 뇌 기능 연결망을 분석하기 위해 새로운 그래프 신경망 프레임워크인 BrainIB++를 소개합니다. 이 모델은 정보 경색 원칙을 적용하여 해석 가능성을 위해 훈련 과정 중 가장 정보성이 높은 뇌 영역을 서브 그래프로 식별합니다. 또한, BrainIB++는 기존 머신러닝 기법의 한계인 수동 특성 엔지니어링 문제를 해결하고, 해석 가능성과 신뢰성을 높이고자 합니다.

- **Technical Details**: BrainIB++는 여러 개의 은닉층을 가진 신경망 구조를 활용하여 뇌의 기능적 연결성을 모델링합니다. 이 모델은 AAL 파셀레이션 지도를 사용하는 대신, 100,000명 이상의 참가자로부터 얻은 대규모 resting-state fMRI 데이터셋을 사용하여 점진적 연결성 네트워크를 추정합니다. 또한, 서브 그래프 생성을 위한 노드 선택 방식은 해석 가능성을 크게 향상시킵니다.

- **Performance Highlights**: 우리 모델은 세 가지 다중 집단 정신분열증 데이터셋에 걸쳐 아홉 가지 기존 뇌 네트워크 분류 방법과 비교하여 우수한 진단 정확도를 보여줍니다. 식별된 서브 그래프는 기존의 정신분열증 임상 바이오마커와 일치하며, 시각, 감각 운동, 그리고 고차 인지 기능 네트워크의 이상을 강조합니다. 이러한 결과는 BrainIB++의 실제 진단 응용 가능성을 강화하는 데 기여합니다.



### From high-frequency sensors to noon reports: Using transfer learning for shaft power prediction in maritim (https://arxiv.org/abs/2510.03003)
Comments:
          Keywords: transfer learning, shaft power prediction, noon reports, sensor data, maritime

- **What's New**: 본 연구에서는 전통적인 센서 데이터에서 얻은 지식을 활용하여, 노온 리포트(noon reports) 기반으로 선박의 샤프트 파워(shaft power)를 예측하는 새로운 접근법을 제안합니다. 이 접근법은 고주파(high-frequency) 데이터로 처음 모델을 학습한 후, 저주파(low-frequency) 노온 리포트를 이용해 미세 조정하는 방법입니다. 연구 결과, 노온 리포트만으로 학습한 경우에 비해 평균 절대 백분율 오류가 개선되었습니다.

- **Technical Details**: 선박의 연료 소비를 정확히 예측하기 위해 서로 다른 선박에서 얻은 데이터를 조합해 지식 transfer learning을 적용했습니다. 이 과정에서 자매선(sister vessels) 및 유사선(similar vessels) 데이터와의 미세 조정을 진행했으며, 기계 학습(machine learning) 모델을 사용해 이러한 데이터를 학습했습니다. 전통적인 해양 공학 접근법과는 달리, 이 방법은 데이터 기반으로 샤프트 파워 예측의 정확성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, 자매선의 경우 평균 10.6%의 오류 감소가 있었고, 유사선에서 3.6%, 다른 선박에서는 5.3%의 개선이 있었습니다. 본 연구의 성과는 신뢰할 수 있는 선박 데이터 부족 문제를 해결하며, 오히려 노온 리포트를 활용하는 새로운 접근법을 제시하였다는 점에서 의의가 있습니다. 또한, 다른 선박에서도 다양한 작업에 적용 가능성이 있음을 보여주었습니다.



### Untargeted Jailbreak Attack (https://arxiv.org/abs/2510.02999)
- **What's New**: 이번 논문에서는 첫 번째로 비목표형 그래디언트 기반 탈옥 공격인 Untargeted Jailbreak Attack (UJA)을 제안하였습니다. 기존의 탈옥 공격 방법들은 고정된 목표에 맞추어 이뤄지기 때문에 그 범위가 제한되었으며, 이를 통해 공격 효율이 떨어졌습니다. UJA는 특정한 출력 패턴을 강제하지 않고 LLM의 안전성 확률을 극대화함으로써 이러한 제약을 해결하고 더욱 효율적으로 공격할 수 있도록 설계되었습니다.

- **Technical Details**: UJA의 공격 목표는 잠재적 위험이 있는 응답을 유발하는 것으로, 이를 통해 출력 패턴의 제한 없이 공격을 시행합니다. 비미분 가능성 문제를 해결하기 위해, 목표를 두 개의 미분 가능 서브 목표로 분해하였습니다. 첫 번째 서브 목표는 최적의 해로운 응답을 찾는 것이고, 두 번째는 이에 해당하는 공격 프롬프트를 결정하는 것입니다.

- **Performance Highlights**: UJA는 여섯 가지 화이트박스 LLM에서 실험하였으며, 100회 최적화 이터레이션으로 80% 이상의 공격 성공률을 기록했습니다. 이는 COLD-Attack과 같은 기존의 최첨단 공격 방법들보다 20% 이상 높은 성과입니다. 또한, UJA는 COLD-Attack과 동일한 비용으로 더 높은 공격 효과를 달성하여 효율성과 효과를 동시에 갖춘 방법임을 입증하였습니다.



### AI Generated Child Sexual Abuse Material - What's the Harm? (https://arxiv.org/abs/2510.02978)
- **What's New**: 최근 생성 인공지능(Generative AI) 기술을 이용해 아동 성적 학대물질(AI CSAM)이 만들어지는 현상이 증가하고 있습니다. 과거에는 기술적 전문 지식이 필요했지만, 오픈 소스 확산 모델의 출현으로 누구나 쉽게 AI CSAM을 생성할 수 있는 환경이 조성되었습니다. 2023년에는 약 2만 건의 의심되는 AI 생성 성적 이미지가 다크 웹 포럼에서 확인되었으며, 이는 아동 보호 및 법 집행에 중대한 도전을 제기합니다.

- **Technical Details**: AI CSAM은 확산 모델(difussion models), 생성적 적대 신경망(Generative Adversarial Networks, GANs) 등 다양한 AI 기반 이미지 및 비디오 합성 기술을 통해 생성됩니다. 이러한 모델들은 무작위 노이즈를 정돈된 이미지로 변환하는 과정을 포함하며, 사용자 지시(prompt)에 따라 내용을 생성할 수 있습니다. AI가 생성한 콘텐츠는 진짜 아동을 사이버 공간에서 학대하는 것과 유사한 형태로 재타겟팅(또는 재가해)할 수 있어, 실제 아동들이 피해를 입을 우려가 존재합니다.

- **Performance Highlights**: 2024년까지 AI CSAM과 관련된 보고는 전년 대비 380% 증가했습니다. 또한, 조사에 따르면 10명 중 1명의 청소년이 동료가 생성 AI를 이용해 아동의 노출된 이미지를 만든 경험이 있다고 응답했습니다. 이는 AI CSAM의 확산이 법 집행 및 사회적 보호 조치를 더욱 어렵게 만들고 있음을 강조합니다. AI CSAM은 단순한 대체 불법 자료가 아니라 아동 성 착취의 메커니즘이 변화하는 획기적인 계기가 되고 있습니다.



### Corrosion Risk Estimation for Heritage Preservation: An Internet of Things and Machine Learning Approach Using Temperature and Humidity (https://arxiv.org/abs/2510.02973)
Comments:
          17 pages

- **What's New**: 이 연구는 필리핀의 문화유산인 산 세바스티안 대성당과 같은 유서 깊은 건축물의 강철 구조물의 부식을 예측하기 위해 IoT(Internet of Things) 하드웨어 시스템을 개발했습니다. 이 시스템은 LoRa 무선 통신을 통해 연결되어 있으며, 3년간의 데이터를 활용하여 정확한 부식 예측을 가능하게 합니다. 이를 통해 문화재 보존을 위한 사전 예방적 접근이 이루어질 수 있습니다.

- **Technical Details**: 연구진은 기온(temperature)과 상대 습도(relative humidity) 데이터를 바탕으로 대기 부식률(atmospheric corrosion rates)을 예측하는 머신러닝 프레임워크를 구축했습니다. 이 프레임워크는 Streamlit 대시보드를 통해 배포되어 실시간 부식 모니터링과 보존 추천을 제공합니다. 또한 ngrok 터널링을 사용하여 대중이 접근할 수 있도록 설정되어 있습니다.

- **Performance Highlights**: 이 연구의 접근법은 최소한의 데이터(minimal-data)로도 구현 가능하며, 제한된 모니터링 리소스가 있는 문화유산 사이트에서도 비용 효과적으로 적용될 수 있습니다. 고급 회귀 기법(advanced regression)을 통해 기본적인 기상 데이터로부터 정확한 부식 예측을 도출하는 것이 가능하여, 세계적으로 문화유산의 사전 보존을 지원할 수 있는 잠재력을 보여줍니다.



### Grounding Large Language Models in Clinical Evidence: A Retrieval-Augmented Generation System for Querying UK NICE Clinical Guidelines (https://arxiv.org/abs/2510.02967)
- **What's New**: 이 논문은 영국의 건강 및 치료 우수성 국가 연구소(NICE)의 임상 가이드라인을 질의하기 위한 Retrieval-Augmented Generation (RAG) 시스템 개발 및 평가를 다룹니다. 이 시스템은 사용자가 자연어 질의에 따라 정밀하게 일치하는 정보를 제공하도록 설계되었으며, 대규모 언어 모델(LLMs)을 사용하여 임상 가이드라인의 긴 길이와 대량의 데이터를 활용할 수 있도록 합니다.

- **Technical Details**: RAG 시스템은 하이브리드 임베딩 메커니즘으로 구성된 검색 아키텍처를 사용하여 300개 가이드라인에서 유도된 10,195개의 텍스트 청크 데이터베이스와 비교 평가되었습니다. 이 시스템은 7,901건의 질의에서 평균 역순위(Mean Reciprocal Rank, MRR) 0.814를 기록하고, 첫 번째 청크에서 81%, 상위 10개 청크에서 99.1%의 리콜(validation accuracy)을 달성하는 성과를 보였습니다.

- **Performance Highlights**: RAG 시스템은 생성 단계에서 가장 두드러진 성과를 보였으며, 70개의 질문-답변 쌍을 수작업으로 선별한 데이터셋에 대해 RAG 강화 모델이 성능에서 상당한 개선을 보였습니다. RAG 강화 O4-Mini 모델의 신뢰성(Faithfulness)은 64.7% 증가하여 99.5%에 달하며, 의학 중심의 Meditron3-8B LLM의 43%와 비교하여 훨씬 높은 성능을 보여주었습니다. 이 연구는 RAG를 의료 분야에서 생성 AI를 효과적이고 신뢰성 높게 적용할 수 있는 접근 방식으로 입증하였습니다.



### Ergodic Risk Measures: Towards a Risk-Aware Foundation for Continual Reinforcement Learning (https://arxiv.org/abs/2510.02945)
- **What's New**: 이번 연구에서는 위험 인식 의사결정을 기반으로 한 지속적 강화 학습(continual RL)의 첫 번째 이론적 접근을 제시합니다. 기존의 지속적 RL 연구는 주로 리스크 중립적(decision-making) 관점에서 진행되었으나, 본 논문에서는 리스크 인식(risk-aware) 접근법을 도입하여 에이전트가 평균 이상의 보상을 최적화할 수 있도록 합니다. 이러한 새로운 접근은 에이전트가 재난 상황을 피할 수 있도록 하는 데 집중해야 함을 강조합니다.

- **Technical Details**: 연구진은 기존의 리스크 측정 이론이 지속적 RL의 요구와 상충됨을 확인했습니다. 이로 인해, 지속적 학습(continual learning)에 적합한 새로운 리스크 측정인 에르고딕 리스크 측정(ergodic risk measures)을 도입하여 리스크 측정 이론을 확장하였습니다. 에르고딕 리스크 측정은 평균 보상 마르코프 결정 과정(average-reward Markov decision process, MDP)을 기반으로 하여 제시되며, 안정성과 유연성을 동시에 충족하는데 도움을 줍니다.

- **Performance Highlights**: 연구에서는 에르고딕 리스크 측정의 직관적인 매력과 이론적 타당성을 입증하기 위한 사례 연구를 포함하고, 수치적 결과를 제공합니다. 이 결과는 지속적 학습 상황에서 리스크 인식 의사결정의 필요성을 강조하며, 실제 환경에서의 생존을 위해 필수적인 리스크 관리의 중요성을 보여줍니다. 연구는 지속적 학습에 있어 리스크 인식의 기초를 마련하여, 향후 연구 및 응용에 기여할 것입니다.



### Multimodal Carotid Risk Stratification with Large Vision-Language Models: Benchmarking, Fine-Tuning, and Clinical Insights (https://arxiv.org/abs/2510.02922)
- **What's New**: 이번 연구는 경동맥 죽종질환에 대한 위험 평가를 개선하기 위해 최첨단 대규모 비전-언어 모델(LVLM)의 가능성을 탐구합니다. 이 연구는 초음파 이미징(USI)과 구조화된 임상, 인구통계학적, 실험실, 단백질 바이오마커 데이터를 통합하여 다중 모달 경동맥 플라크 평가를 수행하는 프레임워크를 제안합니다. 또한, 다양한 오픈소스 LVLM을 비교 분석하고, 특히 LLaVa-NeXT-Vicuna를 초음파 분야에 적응시켜 뇌졸중 위험 분류에서 상당한 개선을 이루어냈습니다.

- **Technical Details**: 연구에서는 7272명의 환자로부터 수집된 B-mode 경동맥 USI 비디오와 해당하는 표 형식 데이터를 사용했습니다. 모든 비디오는 표준화된 조건에서 수집되었으며, 영상 해상도는 밀리미터당 1212 픽셀입니다. 데이터셋은 고위험 환자가 5959명, 저위험 환자가 1313명으로 구성되어 있으며, 이를 이진 분류에 적합하게 설계되었습니다.

- **Performance Highlights**: 연구 결과, 다양한 LVLM이 초음파 기반 심혈관 위험 예측에서 보여주는 가능성과 한계를 강조하며, 모델 보정 및 도메인 적응의 중요성을 강조했습니다. LLaVa-NeXT-Vicuna의 임상적 활용 가능성이 높아졌으며, 다중 모달 데이터를 통합함으로써 보다 정확한 위험 평가 및 진단을 위한 기초를 마련했습니다. 이 연구는 또한 기존 LVLM들이 임상적 상황에 얼마나 잘 일반화될 수 있는지를 평가하는 데 기여하고 있습니다.



### WavInWav: Time-domain Speech Hiding via Invertible Neural Network (https://arxiv.org/abs/2510.02915)
Comments:
          13 pages, 5 figures, project page: this https URL

- **What's New**: 본 논문은 DNN(Deep Neural Network) 기반의 새로운 오디오 데이터 숨기기 기법을 제안합니다. 이전 방법들의 한계를 보완하기 위해, 흐름 기반의 가역 신경망을 사용하여 스테고 오디오(stego audio)와 커버 오디오(cover audio) 간의 직접적인 연결을 Establish합니다. 이 방법은 메시지를 숨기고 추출하는 과정의 가역성을 높이며, 시간-주파수 손실(time-frequency loss)을 적용하여 비밀 오디오의 품질을 향상시킵니다.

- **Technical Details**: 제안된 방법은 비밀 메시지를 시간 영역에서 직접 숨기면서 시간-주파수 제약(time-frequency constraints)을 활용하여 채널 왜곡(channel distortion)을 피하는 것을 목표로 합니다. 이 과정에서, 스테고 오디오에서 비밀 오디오를 복원하기 위해 새로운 흐름 기반 가역 신경망을 동시에 학습하여, 기존의 인코더-디코더 아키텍처보다 더 나은 가역성을 가지도록 설계하였습니다. 또한, 다양한 형태의 소음을 처리할 수 있도록 훈련 단계에서 소음 레이어(noise layer)를 도입하였습니다.

- **Performance Highlights**: 실험 결과, VCTK 및 LibriSpeech 데이터셋에서 우리의 방법이 이전 방법들과 비교하여 주관적 및 객관적 지표에서 우수한 성능을 보였습니다. 특히, 제안된 시스템은 비밀 오디오의 품질 유지와 다양한 일반 왜곡에 대한 강인성을 Exhibit하며, 목표 지향적인 안전한 통신 시나리오에서 유용할 것으로 예상됩니다. 또한, 본 연구는 비밀 메시지를 보호하기 위한 암호화-복호화(encryption-decryption) 모듈을 포함하여 시스템의 안전성을 높였습니다.



### FeDABoost: Fairness Aware Federated Learning with Adaptive Boosting (https://arxiv.org/abs/2510.02914)
Comments:
          Presented in WAFL@ECML-PKDD 2025

- **What's New**: 이번 연구에서는 비동질적 데이터 설정에서 연합 학습(Federated Learning, FL)의 성능과 공정성을 개선하기 위해 새로운 프레임워크인 FeDABoost를 제안합니다. FeDABoost는 동적 부스팅 메커니즘과 적응형 그래디언트 집계 전략을 통합하여, 낮은 로컬 오류율을 가진 클라이언트에 더 높은 가중치를 부여하여 글로벌 모델에 신뢰할 수 있는 기여를 장려합니다.

- **Technical Details**: FeDABoost는 특정 클라이언트에서 하드 투 크래프팅 예제를 강조하기 위해 초점 손실(focal loss) 집중 매개변수를 조정함으로써 저성과 클라이언트를 동적으로 부스트합니다. 클라이언트의 로컬 성능에 기반하여 동적으로 클라이언트의 업데이트를 가중치 조정함으로써, 비균질 데이터 환경에서 신뢰할 수 없는 업데이트로 인한 위험을 줄일 수 있습니다.

- **Performance Highlights**: MNIST, FEMNIST, CIFAR10 데이터셋에서 FeDABoost의 성능을 평가한 결과, FedAvg 및 Ditto와 비교하여 공정성과 경쟁력을 갖춘 성능을 달성함을 확인했습니다. 특히, FeDABoost는 각 클라이언트 간의 모델 성능 차이를 줄이면서 평균 예측 정확도를 높이는 것을 목표로 하여, 데이터의 이질성과 클라이언트 불균형 문제를 효과적으로 완화합니다.



### FinReflectKG - MultiHop: Financial QA Benchmark for Reasoning with Knowledge Graph Evidenc (https://arxiv.org/abs/2510.02906)
- **What's New**: 이번 연구에서는 FinReflectKG - MultiHop을 소개합니다. 이 모델은 S&P 100의 10-K 제출 문서에서 수집한 데이터를 바탕으로 구성된 금융 지식 그래프를 활용하여 다단계(multi-hop) 질의응답 체계를 구축합니다. 이를 통해 자주 사용되는 2-3 단계의 서브그래프 패턴을 발굴하고, 품질 보증 과정을 통해 질문과 답변 쌍의 정확성을 높였습니다.

- **Technical Details**: FinReflectKG - MultiHop은 2022-2024년의 감사된 삼중(triple) 데이터를 S&P 100 필링(ex. filing)과 연결한 금융 지식 그래프인 FinReflectKG를 바탕으로 하고 있습니다. 질의 생성을 위해 질문을 KG 패턴과 연결하여, KG 기반의 정보 검색이 비정형 텍스트 기반 검색보다 효율적임을 이론적으로 입증하고자 했습니다. KG를 통해 정확한 정보 검색을 가능하게 하여, 토큰 사용량을 줄이며 비용 효율성을 높였습니다.

- **Performance Highlights**: 연구 결과, KG 기반의 정보 검색 기법이 다단계 QA 벤치마크에서 약 24%의 정확도를 높이고, 전통적인 페이지 윈도우 설정과 비교했을 때 약 84.5%의 토큰 사용량을 줄인 것으로 나타났습니다. 이로써 KG가 다양한 다단계 질의 처리 시 실질적 이점을 제공한다는 점을 강조하고 있습니다. 책정된 555개의 QA 쌍을 공개하여 향후 연구를 촉진할 계획입니다.



### DMark: Order-Agnostic Watermarking for Diffusion Large Language Models (https://arxiv.org/abs/2510.02902)
- **What's New**: 이번 논문에서는 확산 대형 언어 모델( diffusion large language models, dLLMs)을 위한 첫 번째 워터마킹 프레임워크인 DMark를 소개합니다. 기존의 워터마킹 방법들이 dLLM의 비순차적(degrees of non-sequential) 디코딩 때문에 실패하는 문제를 해결합니다. 그 결과, DMark는 전통적인 워터마킹 기술의 한계를 뛰어넘는 새로운 방안을 제공합니다.

- **Technical Details**: DMark는 세 가지 상호 보완적인 전략을 도입하여 워터마크 탐지를 회복합니다. 첫째, predictive watermarking은 실제 맥락(context)이 없을 때 모델 예측 토큰을 사용합니다. 둘째, bidirectional watermarking은 확산 디코딩에 고유한 전방 및 후방 종속성을 활용하여 워터마킹을 강화합니다. 셋째, predictive-bidirectional watermarking은 두 접근 방식을 결합하여 탐지 강도를 극대화합니다.

- **Performance Highlights**: 여러 dLLM을 대상으로 한 실험에서 DMark는 1%의 위양성(false positive) 비율로 92.0-99.5%의 탐지율을 달성하며 텍스트 품질을 유지합니다. 이는 기존 방법의 단순 변형이 49.6-71.2%에 그친 것과 비교되는 성과입니다. DMark는 텍스트 조작에 대해서도 강인성을 보여줌으로써 비자율 언어 모델에 효과적인 워터마킹의 가능성을 확립합니다.



### Global Convergence of Policy Gradient for Entropy Regularized Linear-Quadratic Control with multiplicative nois (https://arxiv.org/abs/2510.02896)
Comments:
          33 pages, 4 figures

- **What's New**: 이 논문은 동적인 환경에서의 결정 과정에 대한 강력한 프레임워크로 여겨지는 강화 학습(Reinforcement Learning, RL)의 새로운 접근법을 제안합니다. 특히, 시스템 파라미터를 알지 못하는 상황에서도 동작할 수 있는 샘플 기반 정규화된 정책 경량화(Sample-Based Regularized Policy Gradient, SB-RPG) 알고리즘을 소개합니다. 이 방법은 엔트로피 정규화를 활용하여 탐색과 활용 간의 균형을 맞추고, 전통적인 방법보다 더 나은 수렴 특성을 보장합니다.

- **Technical Details**: 제안된 방식은 정규화된 정책 경량화(RPG) 알고리즘을 스토캐스틱 최적 제어 문제에 맞게 조정하여, 비볼록(non-convex) 최적화 환경에서도 전역 수렴성을 증명합니다. 제안된 SB-RPG 방법은 시스템 파라미터에 대한 지식이 없을 때에도 효과적으로 동작하며, 이론적 보장을 통해 정책 최적화를 가능하게 합니다. 본 연구는 표준 수학 기호를 따르며, 최적 제어 문제를 최적화 문제로 변환하는 과정을 설명합니다.

- **Performance Highlights**: Numerical simulations은 SB-RPG의 이론적 결과를 뒷받침합니다. 알려지지 않은 파라미터 환경에서도 유효성을 입증하며, RL 기반 제어 문제에서의 활용 가능성을 크게 증가시킵니다. 실험 결과는 제안된 알고리즘이 다양한 시나리오에서 저항력 있는 정책을 생성할 수 있음을 보여줍니다.



### Representing Beauty: Towards a Participatory but Objective Latent Aesthetics (https://arxiv.org/abs/2510.02869)
- **What's New**: 이 논문에서는 기계가 아름다움을 인식하는 것이 무엇을 의미하는지 탐구합니다. 저자들은 신경망(neural networks)이 다양한 형태의 아름다움을 모델링 할 수 있는 가능성을 짚어 보았습니다. 이들은 깊이 학습(d深 learning) 시스템이 아름다움의 형식을 재현할 수 있는 근본 원인이 물리적 및 문화적 실체의 공동 기반에 있음을 주장합니다.

- **Technical Details**: 미술과 딥 러닝(deep learning)은 표현(representation)에 대한 중심적인 관심을 가지고 있습니다. 딥 러닝은 정보가 다양한 맥락에서 어떤 방식으로 유사하게 표현되는지를 보여주기 위한 '유니버설 표현 가설(universal representation hypothesis)'을 제안합니다. 각기 다른 모델들은 점점 현실을 더 잘 이해할 수 있도록 연합된 표현을 생산하게 됩니다.

- **Performance Highlights**: 연구 결과, 딥 러닝 모델들이 미적 이미지를 평가하는 데 효과적이라는 사실이 입증되었습니다. 이러한 발견은 인간-기계 공동 창작이 가능한 것을 넘어, 문화적 생산 및 기계 인식에서 아름다움이 중요한 역할을 한다는 것을 시사합니다. 저자들은 이러한 접근 방식이 예술가들이 AI 도구를 활용하는 방식에 혁신적인 변화를 가져올 가능성이 있다고 주장합니다.



### Constraint Satisfaction Approaches to Wordle: Novel Heuristics and Cross-Lexicon Validation (https://arxiv.org/abs/2510.02855)
Comments:
          35 pages, 14 figures, 10 tables. Open-source implementation with 91% test coverage available at this https URL

- **What's New**: 이 논문은 Wordle을 제약 충족 문제(Constraint Satisfaction Problem, CSP)로 체계화하여 새로운 제약 인식 솔루션 전략을 소개합니다. 기존의 솔버들은 정보 이론에 기반한 엔트로피 최대화나 빈도 기반 휴리스틱에 의존했지만, 본 연구는 제약과 정보를 동시에 고려하는 CSP-Aware Entropy를 제시하여 3.54회의 평균 추측으로 99.9%의 성공률을 달성했습니다.

- **Technical Details**: CSP-Aware Entropy는 제약 전파 이후 정보 이득을 계산하며, 확률적 CSP 프레임워크를 통해 베이지안 단어 빈도 사전과 논리적 제약을 통합합니다. 이 논문에서는 제약 기반 접근법들이 낮은 노이즈 환경에서 우수한 성능을 기록하며, 10% 노이즈 하에서도 5.3% 포인트의 이점을 유지한 결과를 제시합니다.

- **Performance Highlights**: 2,315개의 영어 단어에 대한 평가에서 CSP-Aware Entropy는 99.9%의 성공률을 기록하며, 1.7%의 통계적으로 유의미한 개선을 보였습니다. 또한 스페인어 500개 단어에 대한 교차 어휘 검증 결과, 88%의 성공률을 나타내며, 언어 간 제약 충족 원리가 일반화 가능함을 보여주었습니다.



### Flamed-TTS: Flow Matching Attention-Free Models for Efficient Generating and Dynamic Pacing Zero-shot Text-to-Speech (https://arxiv.org/abs/2510.02848)
- **What's New**: 최근 제로샷(Zero-shot) 텍스트-음성 변환(Text-to-Speech, TTS) 기술이 크게 발전하였으며, 짧은 문맥 프롬프트를 사용하여 텍스트로부터 음성을 합성할 수 있게 되었습니다. 이러한 프롬프트는 화자 정체성, 억양(prosody), 기타 특성을 모방하는 데 도움을 주며 널리 사용되는 화자 별 데이터 없이도 적용 가능하다는 점이 특징입니다. 하지만 기존 접근 방법들은 토큰 반복(token repetition)이나 예기치 않은 콘텐츠 전이(content transfer)와 같은 불안정한 합성을 초래하는 문제와 느린 추론(inference) 속도 및 높은 계산 비용이라는 어려움을 여전히 겪고 있습니다.

- **Technical Details**: 이러한 도전 과제를 해결하기 위해 Flamed-TTS라는 새로운 제로샷 TTS 프레임워크를 제안합니다. Flamed-TTS는 낮은 계산 비용(low computational cost), 낮은 지연(latency), 높은 음성 충실도(speech fidelity) 및 풍부한 시간적 다양성(temporal diversity)을 강조합니다. 이를 위해 흐름 매칭(flow matching) 훈련 패러다임을 재구성하고, 음성의 여러 속성에 해당하는 이산(discrete) 및 연속(continuous) 표현을 통합하였습니다.

- **Performance Highlights**: 실험 결과, Flamed-TTS는 이해도(intelligibility), 자연스러움(naturalness), 화자 유사성(speaker similarity), 음향적 특성 보존(acoustic characteristics preservation), 동적 빠르기(dynamic pace) 측면에서 최신 기법을 초월하는 성능을 보였습니다. 특히 Flamed-TTS는 제로샷 TTS 기준선들과 비교하여 4%의 최상의 WER(Word Error Rate)를 달성하며, 낮은 추론 지연 속도와 높은 음성 충실도를 유지하는 데 성공했습니다.



### Knowledge-Aware Modeling with Frequency Adaptive Learning for Battery Health Prognostics (https://arxiv.org/abs/2510.02839)
Comments:
          12 pages, 4 figures, 4 tables

- **What's New**: 본 논문은 Karma라는 새로운 지식 기반 모델을 제안하여 배터리 용량 추정 및 남은 유효 수명(RUL) 예측의 정확성과 신뢰성을 향상시킵니다. 이 모델은 주파수 적응 학습(Frequency-Adaptive Learning)을 통해 배터리의 해리 과정 및 비선형성을 더욱 수월하게 처리합니다. Karma는 두 개의 스트림으로 구성된 심층 학습 아키텍처를 사용하여 장기 저주파 및 단기 고주파 동적 변화를 각각 포착합니다.

- **Technical Details**: Karma의 방법론은 배터리 신호를 주파수 대역별로 분해한 후, CNN-LSTM과 BiGRU 구조를 통해 각각의 저주파 및 고주파 신호를 처리합니다. 이러한 방식은 시간에 따른 비선형 변화를 보다 효율적으로 모델링할 수 있게 합니다. 또한, Karma는 경험적인 배터리 지식을 통합하여 복원력 있는 예측을 제공하며, 파라미터 최적화를 위해 입자 필터(Particle Filter) 방법을 채택합니다.

- **Performance Highlights**: Karma는 실험적으로 두 개의 주요 데이터세트에서 기존 최신 알고리즘보다 평균적으로 각각 50.6%와 32.6%의 예측 오차 감소를 달성했습니다. 이러한 성능 향상은 Karma 모델의 견고성과 유연성을 입증하며, 다양한 적용 분야에서의 배터리 관리의 안전성과 신뢰성을 높일 수 있는 잠재력을 제시합니다.



### Evaluating Large Language Models for IUCN Red List Species Information (https://arxiv.org/abs/2510.02830)
Comments:
          20 pages, 7 figures

- **What's New**: 본 연구에서는 21,955종의 생물에 대해 네 가지 핵심 IUCN 적색 목록 평가 요소인 분류학(taxonomy), 보전 상태(conservation status), 분포(distribution), 위협(threats)에 대해 다섯 개의 선도적인 대형 언어 모델(Large Language Models, LLMs)의 신뢰성을 체계적으로 검증했습니다. 결과적으로, 모델들은 분류학적 분류에서 94.9%의 정확도를 보였으나, 보전 추론에서 27.2%의 낮은 성능을 나타내며 인지-추론 간의 갭이 드러났습니다. 이 연구는 LLM이 정보 검색에서는 강력한 도구일 수 있지만, 판단 기반 결정에서는 인간의 감독이 필요하다는 점을 강조합니다.

- **Technical Details**: 연구는 2022-2023년 사이 IUCN 적색 목록에서 평가된 21,955 종에 대한 21,962개의 평가 기록을 분석했습니다. 데이터 세트에는 척추동물(vertebrates), 식물(plants), 무척추동물(invertebrates), 균(fungi) 등이 포함되었으며, 이들 대조적으로 무척추동물과 곰팡이가 상대적으로 적게 나타났습니다. 평가된 다섯 개의 LLM은 연구의 초점에 맞춰 공통된 조건에서 균일하게 평가되었으며, 각 모델은 최소한의 프롬프트(minimal prompting)를 사용하여 일관된 작업 해석을 보장했습니다. 이 연구에서 평가된 모델은 OpenAI의 GPT-4.1, xAI의 Grok 3, Anthropic의 Claude Sonnet 4 등의 공용 API 기반 시스템과 Google DeepMind의 Gemma 3-27B 및 Meta의 Llama 3.3-70B와 같은 로컬 호스팅 모델을 포함합니다.

- **Performance Highlights**: LLMs는 정보 검색에 있어 유용하지만, 생물 다양성 평가와 같은 고차원적 작업에 있어서는 전문가의 검증 없이 신뢰할 수 없는 성능을 보였습니다. 특히 특정 분류에서 성능의 불균형이 발견되어, 기존 보전 작업에 있어 어렵고 보존이 소외된 무척추동물 및 자생 식물에 대한 검토의 필요성이 제기됩니다. 따라서 LLM은 교육 및 데이터 탐색과 같은 분야에는 유용하나, 위험 평가 및 정책 수립에는 전문가의 관여가 필요하다는 결론에 도달했습니다.



### A Computational Framework for Interpretable Text-Based Personality Assessment from Social Media (https://arxiv.org/abs/2510.02811)
Comments:
          Phd thesis

- **What's New**: 이 논문은 성격( Personality ) 분석을 위한 두 개의 데이터셋인 MBTI9k 및 PANDORA를 소개합니다. 이 데이터셋은 Reddit에서 수집된 것으로, 1,700만 개의 댓글과 10,000명 이상의 사용자 데이터를 포함합니다. 이러한 데이터는 개인화된 분석을 위한 기초를 마련하며, 성격 심리와 자연어 처리(NLP)의 간극을 메우기 위한 노력을 다룹니다.

- **Technical Details**: 성격 분석을 위해 삼성의 SIMPA (Statement-to-Item Matching Personality Assessment) 프레임워크를 개발하여 사용자 생성의 진술이 검증된 질문 항목과 매칭됩니다. 이 과정에서 기계 학습 및 의미론적 유사성을 활용하여 사람의 평가와 유사한 결과를 도출하면서도 해석 가능성과 효율성을 유지합니다. 이 프레임워크는 복잡한 라벨 분류를 다룰 수 있는 유연성을 제공합니다.

- **Performance Highlights**: 체험 결과는 인구 통계 변수들이 모델의 유효성에 영향을 미친다는 점을 보여주었습니다. SIMPA를 통해 기계 학습 기술을 활용하여 인사이트를 제공하며, 인간의 직관과 비교할 수 있는 성격 평가를 진행할 수 있습니다. 이는 성격 분석 외에도 다양한 연구 및 실용적인 응용 프로그램에 적합한 가능성을 제공합니다.



### Dissecting Transformers: A CLEAR Perspective towards Green AI (https://arxiv.org/abs/2510.02810)
- **What's New**: 이번 논문에서는 Transformer 아키텍처의 핵심 구성 요소에 대한 추론 에너지를 미세하게 분석하는 최초의 실증적 연구를 소개합니다. 기존의 연구들이 에너지 효율성을 단순한 모델 수준에서 다루던 것과 달리, 본 연구에서는 Attention 블록과 같은 개별 요소의 에너지 소비를 정밀하게 측정하고 최적화할 수 있는 방법론인 CLEAR를 제안합니다. 이를 통해 LLM이 차지하는 환경적 비용을 보다 명확히 이해하고 개선할 수 있습니다.

- **Technical Details**: CLEAR는 마이크로초 단위의 구성 요소 실행과 밀리초 단위의 에너지 센서 모니터링 간의 시간 불일치를 극복하기 위해 새롭게 설계된 기술입니다. 이 방법론은 15개의 다양한 모델을 평가하며 각 구성 요소의 에너지를 정확히 측정할 수 있도록 설계되었습니다. 에너지를 구성 요소 단위로 분해하여 고유의 소비 패턴을 비교하고, 다양한 인풋 길이와 플로팅 포인트 정밀도에 대해 실증 분석을 수행합니다.

- **Performance Highlights**: CLEAR 방법론을 사용한 결과, Attention 블록이 계산당 소비하는 에너지가 상대적으로 높은 점이 밝혀졌습니다. FLOP의 수치가 에너지 소비와 비례하지 않음을 보여 줌으로써, FLOP만으로는 실제 에너지 비용을 제대로 평가할 수 없음을 강조합니다. 이번 연구는 구성 요소별 에너지 기준선을 확립하고, 에너지 효율적인 Transformer 모델 구축을 위한 최적화의 기초를 마련하는 데 기여합니다.



### Relevance-Aware Thresholding in Online Conformal Prediction for Time Series (https://arxiv.org/abs/2510.02809)
- **What's New**: 이 논문에서는 기계 학습에서 불확실성 정량화(uncertainty quantification)가 중요한 이슈로 부각되었음을 다루고 있습니다. 특히, 시계열 데이터에 대한 온라인 정합 예측(Online Conformal Prediction, OCP) 방법을 통해 데이터 분포의 변화에 대응할 수 있는 새로운 접근법을 제안합니다. 제안된 방법은 예측 구간의 적합성을 높이고, 급격한 임계값(threshold) 변화를 방지하여 예측 구간을 더욱 좁히는 데 기여합니다.

- **Technical Details**: 논문에서는 OCP 방법에서 예측 구간의 유효성을 단순한 이항 평가(inside/outside)가 아닌, 예측 구간과 실제 값 간의 관련성을 활용하는 방법으로 향상시키고자 합니다. 이를 위해 예측 구간의 경계와 실제 값의 거리 기반으로 평가하는 새로운 피드백 방식을 도입하였습니다. 이 접근법은 OCP의 여러 방법론의 성능 향상에 기여할 수 있으며, 제안된 함수는 맞춤형으로 설계될 수 있습니다.

- **Performance Highlights**: 실제 데이터 세트를 바탕으로 한 실험 결과, 수정된 OCP 방법이 기존 방법들보다 더 나은 또는 경쟁력 있는 유효성과 효율성을 달성했음을 보여줍니다. 더욱이, 제안된 방법은 시계열 데이터 예측에서 불확실성 정량화를 효과적으로 통합할 수 있는 가능성을 제시합니다. 전체 코드는 GitHub에 등록되어 있어, 연구자들이 쉽게 접근할 수 있도록 제공됩니다.



### Work Zones challenge VLM Trajectory Planning: Toward Mitigation and Robust Autonomous Driving (https://arxiv.org/abs/2510.02803)
Comments:
          13 pages,5 figures

- **What's New**: 이 논문에서는 자동 운전 시스템에서 작업 영역에서 경로 계획을 위한 시각 언어 모델(Visual Language Models, VLM) 사용에 대한 최초의 체계적 연구를 수행하였습니다. VLM이 불규칙한 레이아웃 및 동적으로 변화하는 기하학적 구조가 포함된 작업 영역에서 정확한 경로를 생성하지 못한다는 점을 강조하며, 68%의 경우에 성공하지 못함을 보여줍니다. 이 연구는 비정상적인 패턴을 식별하고, 이를 통해 새로운 경로 생성 방법인 REACT-Drive를 제시하여 VLM의 한계를 극복하고자 합니다.

- **Technical Details**: REACT-Drive는 VLM을 통합한 두 단계의 경로 계획 프레임워크로, 실패 사례를 제약 규칙과 실행 가능한 경로 계획 코드로 변환합니다. 또한, Retrieval-Augmented Generation (RAG) 방법을 사용하여 유사한 패턴을 새로운 시나리오에서 검색합니다. 이 방식은 자율 주행 시스템이 안전 요구사항 및 교통 규칙을 준수하는 경로를 생성하도록 안내합니다. 실험 결과, REACT-Drive는 평균 이탈 오차를 약 3배 감소시키고, 타 방법에 비해 가장 낮은 추론 시간인 0.58초를 기록합니다.

- **Performance Highlights**: REACT-Drive는 ROADWork 데이터셋에서 Qwen2.5-VL을 평가할 때 VLM 기준선에 비해 약 3배의 이탈 오차 감소를 달성하였습니다. 또한, 15개의 실제 작업 영역 시나리오에서 실시한 실험을 통해 REACT-Drive의 강력한 실용성을 입증하였습니다. 이 연구는 향후 VLM을 통한 작업 구역 운전 능력 향상 연구에 큰 기여를 할 것으로 기대됩니다.



### OptunaHub: A Platform for Black-Box Optimization (https://arxiv.org/abs/2510.02798)
Comments:
          Submitted to Journal of machine learning research

- **What's New**: OptunaHub는 연구 분야 간의 분산된 블랙박스 최적화(Black-box optimization) 방법과 벤치마크를 중앙 집중화하는 커뮤니티 플랫폼입니다. 이 플랫폼은 통합된 Python API, 기여자 패키지 레지스트리, 그리고 검색성을 증진시키기 위한 웹 인터페이스를 제공합니다. OptunaHub는 기여와 애플리케이션의 선순환을 촉진하는 것을 목표로 합니다.

- **Technical Details**: OptunaHub는 세 가지 주요 구성 요소로 이루어져 있습니다: OptunaHub Module(등록된 패키지를 로드하는 Python 라이브러리), OptunaHub Registry(패키지 레지스트리), 및 OptunaHub Web(등록된 패키지 정보를 집계하는 웹 인터페이스)입니다. load_module 기능을 통해 OptunaHub Registry에서 패키지를 로드할 수 있으며, 다양한 샘플러와 벤치마크의 호환성이 보장됩니다. OptunaHub는 또한 실험 지원 기능을 통해 등록된 패키지에 직접 혜택을 제공합니다.

- **Performance Highlights**: 현재 94개의 패키지가 등록되어 있으며, OptunaHub 패키지의 총 월간 다운로드 수는 100,000회를 초과했습니다. 또한, OptunaHub Web은 사용자 정의 검색성과 가시성을 높여, 각 패키지의 README.md에서 자동으로 생성된 개별 패키지 페이지를 통해 실현됩니다. 이로 인해 기여자들은 자신의 작업을 보다 넓은 청중에게 홍보할 수 있는 기회를 얻습니다.



### Pareto-optimal Non-uniform Language Generation (https://arxiv.org/abs/2510.02795)
Comments:
          24 pages, 1 figure

- **What's New**: 이번 연구에서는 Kleinberg와 Mullainathan이 제시한 언어 생성을 다루는 최신 모델을 발전시켜, 비균일 언어 생성(non-uniform language generation)에서의 Pareto Optimality에 대해 분석합니다. 특히, 기존 연구에서 제시된 알고리즘들은 언어에 따라 생성 시간이 최적이 아닐 수 있으며, 이에 대한 해결책을 제시합니다. 저자들은 생성 시간이 거의 Pareto-optimal한 새로운 알고리즘을 제안하여, 모든 언어에 대해 동시에 최적의 생성 시간을 달성하는 방법에 대한 기초를 마련합니다.

- **Technical Details**: 언어 생성 모델은 무한 문자열 집합에 기반한 개별 언어를 다루며, 적대자가 선택한 언어 $L$의 문자열을 온라인 방식으로 나열합니다. 높은 수준의 보장을 위해 비균일 언어 생성 개념이 도입되었으며, 이론적으로 구현 가능성이 보장됩니다. 본 연구의 알고리즘은 언어 $L$의 생성 시간 $t^ullet(L)$가 Pareto-optimal성을 가지도록 설계되어 있으며, 이는 특정 언어의 생성 시간이 다른 언어에 비해 나쁨이 없음을 의미합니다.

- **Performance Highlights**: 제안된 알고리즘은 무한 개의 언어에서 동시에 최적의 생성 시간을 달성하기 위한 새로운 기준을 제시합니다. 알고리즘은 기존의 연구에서 발견된 비균일 언어 생성 알고리즘들과 비교했을 때, 성능상에서 우수성을 보입니다. 이 작업은 노이즈가 있는 또는 대표적인 생성 설정에서도 Pareto-optimal 알고리즘을 얻을 수 있는 통합개념을 제공하여, 다양한 타당한 조건 하에서도 성능을 극대화할 수 있음을 입증합니다.



### MaskCD: Mitigating LVLM Hallucinations by Image Head Masked Contrastive Decoding (https://arxiv.org/abs/2510.02790)
Comments:
          accepted to emnlp2025 findings

- **What's New**: 논문에서는 이미지 헤드 마스크드 대비 디코딩(MaskCD)이라는 새로운 접근법을 제안합니다. 이 방법은 LVLMs의 '이미지 헤드'를 마스킹하여 contrastive decoding을 위한 부정적인 샘플을 구축함으로써, 환각 현상을 효과적으로 완화하는 방향으로 나아갑니다. MaskCD는 기존의 환각 완화 기법 대비 다양한 벤치마크에서 우수한 성능을 보이고 있습니다.

- **Technical Details**: LVLMs는 시각 정보와 텍스트 모달리티를 통합하여 다중 모달 추론을 수행하는 모델입니다. 본 연구에서는 LVLMs의 특정 레이어에서 '이미지 헤드'가 이미지 토큰에 과도한 주의를 기울이는 경향을 발견하였으며, 이들을 마스킹하여 부정적인 샘플을 만드는 방법론을 제시합니다. 기존의 대비 디코딩 방법(constrastive decoding)과 주의 조작(attention manipulation)의 장점을 결합하여 보다 높은 품질의 부정적 샘플을 구축할 수 있음을 강조합니다.

- **Performance Highlights**: MaskCD는 LLaVA-1.5-7b 및 Qwen-VL-7b 모델을 검증하는 다양한 벤치마크에서 실험되었습니다. 결과적으로 MaskCD는 환각 현상을 효과적으로 완화하면서 LVLMs의 기본적인 기능들을 유지하는 데 성공했습니다. 실험을 통해 MaskCD의 성능이 기존의 환각 완화 방법들을 초월한다는 점이 입증되었습니다.



### Align Your Query: Representation Alignment for Multimodality Medical Object Detection (https://arxiv.org/abs/2510.02789)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 다양한 의료 이미징 모달리티(예: CXR, CT, MRI)에 대한 의료 물체 탐지의 성능을 개선하기 위해 새로운 방법을 제시합니다. 기존의 단일 탐지기가 혼합된 의료 모달리티에서 학습할 때의 한계를 극복하기 위해, 객체 쿼리의 표현을 모달리티 텍스트와 정렬하는 방식을 채택하였습니다. 이를 통해, 모달리티 정보를 명확하게 반영하면서도 추가적인 주석 없이 가벼운 모달리티 토큰을 도입합니다.

- **Technical Details**: 본 연구의 핵심 기술은 Multimodality Context Attention(MoCA)와 Query REPA입니다. MoCA는 객체 쿼리와 모달리티 토큰 간의 상호 작용을 통해 모달리티 정보를 감지 과정에 통합하는 혁신적인 방법입니다. 또한 제안된 QueryREPA는 사전 학습 단계에서 특정 작업을 위한 대조 손실을 사용하여 쿼리 표현을 모달리티 토큰과 정렬하여, 다중 모달리티에서 신뢰성 있는 대표성을 형성합니다.

- **Performance Highlights**: 제안된 방법은 서로 다른 모달리티로 학습된 데이터셋에서 일관되게 AP(평균 정확도)를 개선하며, 구조적 수정 없이 최소한의 오버헤드로 성능 향상을 꾀합니다. MoCA와 QueryREPA의 결합은 의료 물체 탐지 성능에 시너지 효과를 제공하여, 실제 환경에서의 다중 모달리티 의료 정보 처리를 위한 효과적인 경로를 제시합니다.



### Fusing Multi- and Hyperspectral Satellite Data for Harmful Algal Bloom Monitoring with Self-Supervised and Hierarchical Deep Learning (https://arxiv.org/abs/2510.02763)
- **What's New**: 새로운 연구에서는 다중 센서 위성 데이터를 활용한 해로운 조류 번식(HAB) 감지 및 매핑을 위한 자기 감독 기계 학습 프레임워크를 제안합니다. 이 프레임워크는 다양한 위성의 반사율 데이터(VIIRS, MODIS, Sentinel-3, PACE)와 TROPOMI 태양 유도 형광(SIF) 데이터를 융합하여 HAB의 심각도와 종 분류를 생성합니다.

- **Technical Details**: 제안된 프레임워크(SIT-FUSE)는 자기 감독 표현 학습(self-supervised representation learning)과 계층적 딥 클러스터링(hierarchical deep clustering)을 사용하여 해양 식물 플랑크톤의 농도 및 종을 해석 가능한 클래스로 분할합니다. 이 방법은 2018-2025년 동안 멕시코만과 남부 캘리포니아에서 수집된 현장 데이터에 대해 검증되었습니다.

- **Performance Highlights**: 결과는 전체 식물성 플랑크톤 및 주요 조류인 Karenia brevis, Alexandrium spp., Pseudo-nitzschia spp.의 측정값과 강한 일치를 보였습니다. 이 연구는 레이블이 부족한 환경에서도 HAB 모니터링의 확장성을 높이며, 계층적 임베딩을 통해 탐색적 분석을 가능하게 하는 중요한 단계입니다.



### Hierarchical Generalized Category Discovery for Brain Tumor Classification in Digital Pathology (https://arxiv.org/abs/2510.02760)
- **What's New**: 이번 연구에서는 뇌종양 분류를 위한 새로운 접근법인 Hierarchical Generalized Category Discovery for Brain Tumor Classification (HGCD-BT)를 소개합니다. 이 방법은 계층적 클러스터링(hierarchical clustering)과 대조 학습(contrastive learning)을 통합하여, 미지의 클래스도 포함된 비정형 데이터에서 분류를 가능하게 합니다. 특히, 기존의 GCD 방법들이 가지는 한계를 극복하며, 뇌종양의 계층적 분류 구조를 반영할 수 있습니다.

- **Technical Details**: HGCD-BT는 레이블이 있는 데이터와 없는 데이터를 혼합하여 비정형 데이터를 분류하는 것을 목표로 합니다. 이 기법은 새로운 반지도 학습(semi-supervised) 기반의 계층적 클러스터링 손실을 도입하여, 학습 과정에서 계층적 구조를 직접적으로 모델링함으로써, 기존의 GCD 방법보다 더 우수한 성능을 보입니다. 실험을 통해 HGCD-BT는 기존 GCD 방식 대비 +28%의 정확성 향상을 달성했으며, 이 결과는 뇌종양의 다양한 하위 유형을 식별하는 데에 특히 효과적임을 보여줍니다.

- **Performance Highlights**: HGCD-BT는 OpenSRH 데이터셋에서 패치 수준 분류에 대한 성능을 평가하여 큰 개선을 보였습니다. 또한 Hematoxylin and Eosin (H&E) 염색된 전체 슬라이드 이미지를 사용한 슬라이드 수준 분류에서도 일반화 가능성을 Demonstrate하여, 12개의 뇌종양 유형을 정확하게 분류할 수 있음을 입증했습니다. 이 연구는 imaging modalities의 다양성과 분류의 세분화에 있어 HGCD-BT의 강력한 성능을 강조합니다.



### Prototyping Digital Social Spaces through Metaphor-Driven Design: Translating Spatial Concepts into an Interactive Social Simulation (https://arxiv.org/abs/2510.02759)
Comments:
          25 pages, in submission to CHI 2026

- **What's New**: 이 논문에서는 사용자가 새로운 소셜 미디어 환경을 상상하고 탐색할 수 있도록 돕는 은유 기반 시스템(metaphor-driven system)을 소개합니다. 이 시스템은 사용자의 은유를 구조화된 플랫폼 기능 세트로 변환하고, LLM 기반 에이전트가 포함된 상호작용 시뮬레이션을 생성합니다. 연구 결과, 은유를 통해 사용자가 사회적 기대를 표현할 수 있음을 보여주며, 시뮬레이션의 진정성은 친밀감(intimacy), 참여(participation), 시간적 참여(temporal engagement)와 같은 역학을 얼마나 잘 포착하는지에 달려 있음을 밝힙니다.

- **Technical Details**: 은유 기반 설계를 실현하는 핵심 과제는 상상된 사회적 공간이 실제로 어떻게 기능할지를 테스트하는 것입니다. 시뮬레이션은 HCI(인간-컴퓨터 상호작용) 및 사회 과학에서 사용자 행동을 모델링하고 시스템 설계의 결과를 예측하는 데 오랫동안 사용되어 왔습니다. 본 논문에서는 LLM(대형 언어 모델)과 생성적 에이전트 기반 시뮬레이션을 활용하여 사용자의 은유에서 파생된 새로운 소셜 미디어 공간을 상상하고 생성할 수 있는 시스템을 제시하였습니다.

- **Performance Highlights**: 참여자들은 자신의 이상적인 소셜 공간을 은유를 통해 쉽게 구상할 수 있었으며, 개방 공간(open space)과 폐쇄 공간(closed space) 간의 일관된 구별이 드러났습니다. 시뮬레이션에서 생성된 시스템 기능 세트와 사용자가 상상한 공간 간의 정합성에도 차이가 있었으며, 이는 시스템의 잠재적 약점뿐만 아니라 은유 사용의 한계를 강조합니다. 최종적으로, 사용자들은 시뮬레이션과의 상호작용 경험을 통해 사회적 기대와 얼마나 잘 일치하는지를 평가하였으며, 이는 모의 공간의 진정성이 상당히 중요한 요소임을 보여주었습니다.



### SAE-RNA: A Sparse Autoencoder Model for Interpreting RNA Language Model Representations (https://arxiv.org/abs/2510.02734)
Comments:
          preprint

- **What's New**: 최근 대형 언어 모델(Large Language Models)과 깊은 학습의 발전이 생물 분자 모델링을 혁신하고 있습니다. 특히 본 연구에서는 RiNALMo와 같은 RNA 언어 모델의 내부 구조를 해석할 수 있는 SAE-RNA 모델을 제안합니다. 이 모델은 mRNA 및 비코딩 RNA(ncRNA)의 가족에 대한 정보를 탐구하며, 교육 없이도 사전 학습된 임베딩의 개념 발견을 지원합니다.

- **Technical Details**: SAE-RNA는 RiNALMo에서 추출한 여러 숨겨진 상태(hidden states)로부터 사전 학습된 언어 모델의 임베딩을 이용해 구성됩니다. 이 모델은 구조적 요소와 기능적 모티프에 대한 ncRNA 가족을 자동으로 검사하여, 생물학적 카테고리에 개념 특성을 매핑합니다. SAE는 전통적인 과완전(autoencoder) 방식을 사용하여, 추출된 밀집 임베딩을 희소한 특징으로 분해합니다.

- **Performance Highlights**: 실험을 통해 SAE-RNA는 RNA 언어 모델의 임베딩이 생물학적 구조와 ncRNA 가족에 일관되게 맵핑됨을 보여줍니다. 이 연구는 RNA 언어 모델이 생물학적 개념을 어떻게 조직하는지에 대한 새로운 통찰을 제공하며, 생물학적 지식과 깊은 표현 사이의 간극을 줄이는 데 기여할 것으로 기대됩니다.



### TravelBench : Exploring LLM Performance in Low-Resource Domains (https://arxiv.org/abs/2510.02719)
Comments:
          10 pages, 3 figures

- **What's New**: 이번 연구에서는 저자들이 14개의 여행 도메인 데이터셋을 구성하여 기존 LLM 벤치마크에서 부족했던 저자원(low-resource) 작업에 대한 모델의 성능을 평가했습니다. 이를 통해 여행 산업에서의 LLM 성능을 더 잘 이해할 수 있는 기반을 마련했습니다. 이 연구는 기존의 opinion mining을 넘어선 다양한 작업을 포함하여 LLM의 평가를 위한 자원을 확장하고 있습니다.

- **Technical Details**: 연구팀은 현장 사용 사례에서 수집된 익명 데이터로부터 다양한 NLP 작업을 수행하기 위해 데이터를 수집하였으며, 이를 위해 데이터 수집 과정에서 개인 식별 정보를 제거했습니다. 데이터는 인간 전문가의 참여를 통해 주석이 달리고 검증되었으며, LLM의 잠재적 편향을 최소화하기 위한 지침이 마련되었습니다. 또한 LLM의 내부 지식과 지시 따르기 능력의 조합을 통해 문맥에 따른 예측 능력을 평가하고 있습니다.

- **Performance Highlights**: 저자들은 LLM 모델의 정확도, 확장성 행동, 추론 능력 등을 분석하고, LLM이 복잡한 도메인 별 작업에서 성능 병목 현상을 겪는다는 것을 확인했습니다. 또한, LLM의 훈련 과정에서 FLOPs와 모델 크기의 변화가 성능에 미치는 영향을 조사하였으며, 작은 LLM의 경우 더 나은 추론 능력이 성능을 높이는 데 기여한다는 것을 밝혔습니다.



### CST-AFNet: A dual attention-based deep learning framework for intrusion detection in IoT networks (https://arxiv.org/abs/2510.02717)
Comments:
          9 pages, 9 figures, 5 tables

- **What's New**: 이 논문은 IoT(Internet of Things) 네트워크에서의 강인한 침입 탐지를 위해 설계된 CST AFNet이라는 새로운 이중 주의 기반의 딥러닝 프레임워크를 소개합니다. 기존의 방법들이 갖고 있던 복잡한 사이버보안 문제를 해결하기 위해 다중 스케일 Convolutional Neural Networks (CNNs)과 Bidirectional Gated Recurrent Units (BiGRUs)를 통합한 모델입니다.

- **Technical Details**: CST AFNet은 공간적 특성 추출을 위해 CNN을 사용하고, 시간적 의존성을 포착하기 위해 BiGRUs를 활용하는 구조입니다. 또한, 이중 주의 메커니즘인 채널 주의(channel attention)와 시간 주의(temporal attention)를 날려 중요한 패턴에 집중합니다. 이 모델은 Edge IIoTset 데이터셋을 통해 훈련되고 평가되었습니다.

- **Performance Highlights**: CST AFNet은 15가지 공격 유형 및 정상 트래픽에 대해 99.97%의 뛰어난 정확도를 기록하였습니다. 또한, 매크로 평균 정밀도, 재현율 및 F1 점수가 모두 99.3%를 초과하는 Exceptional performance를 보여주었습니다. 실험 결과는 CST AFNet이 기존 딥러닝 모델에 비해 현저히 우수한 탐지 정확도를 달성했음을 확인시켜주었습니다.



### A $1000\times$ Faster LLM-enhanced Algorithm For Path Planning in Large-scale Grid Maps (https://arxiv.org/abs/2510.02716)
- **What's New**: 이번 연구에서는 LLM-A*의 성능 한계를 극복하기 위해 혁신적인 LLM 강화를 기반으로 한 경로 계획 알고리즘인 iLLM-A*를 제안합니다. iLLM-A*는 A*의 최적화, 고품질 웨이포인트 생성, 그리고 적절한 웨이포인트 선택 등 세 가지 주요 메커니즘을 포함하고 있습니다. 이 알고리즘은 기존의 LLM-A*에 비해 계산 시간과 메모리 비용을 획기적으로 줄이는데 도움을 줍니다.

- **Technical Details**: iLLM-A*는 A*의 탐색 시간을 줄이기 위해 해시 테이블을 사용하여 탐색된 격자를 빠르게 쿼리합니다. 또한, LLM을 기반으로 한 점진적 학습 방법을 통해 더 높은 품질의 웨이포인트를 생성합니다. 마지막으로, 경험 기반의 방법을 통해 LLM이 생성한 중복 웨이포인트들 중 적절한 서브셋을 선택하여 경로 계획에 활용합니다.

- **Performance Highlights**: iLLM-A*는 LLM-A*와 비교하여 평균적으로 1000배 이상, 극단적인 경우에는 2349.5배의 속도를 향상시키며, 메모리 비용을 최대 58.6% 절약합니다. 또한, 경로 길이가 명확하게 짧아지고 경로 길이의 표준 편차도 낮아집니다. 이러한 성능 개선은 대규모 격자 맵에서의 경로 계획의 효율성을 크게 향상시킵니다.



### Fully automated inverse co-optimization of templates and block copolymer blending recipes for DSA lithography (https://arxiv.org/abs/2510.02715)
- **What's New**: 이번 연구에서는 블록 공중합체(block copolymers, BCPs)의 유도 자기 조립(directed self-assembly, DSA) 기술을 위한 새로운 템플릿 형상 인자를 제안했습니다. 특히 가우시안 디스크립터(Gaussian descriptor)를 사용하여 템플릿의 형상을 두 개의 매개변수로 특성화할 수 있도록 했습니다. 신뢰성 높은 제조 가능성을 보장하면서 최적화된 템플릿을 생성하는 과정에서 베이지안 최적화(Bayesian optimization, BO)를 적용했습니다. 이러한 접근 방식은 템플릿과 블록 공중합체 시스템의 상호 작용을 개선하여 DSA 기술의 응용 가능성을 높이는 데 기여할 수 있습니다.

- **Technical Details**: 연구에서 제안된 가우시안 디스크립터는 템플릿의 형상을 정의하는 두 개의 파라미터를 바탕으로 구성되어 있습니다. BO를 통해 다중 구멍 패턴을 위한 최적 템플릿을 효율적으로 도출하며, 템플릿의 곡률 변동에 대한 제약 조건을 두어 제조 가능성을 향상시킵니다. 이 과정에서 AB/AB 이진 혼합물(binary blend)을 사용하여 순수한 다블록 공중합체에 비해 템플릿 형상에 더욱 잘 적응하도록 하였습니다.데이터는 최적화 과정 중 각 거리에서의 템플릿 형태를 분석하여 제공됩니다.

- **Performance Highlights**: 최적화된 템플릿의 자가 조립형태는 각 거리에서의 목표 패턴과 완벽하게 일치하며, 원형의 구멍 정확도 및 위치 정확도 모두 뛰어난 결과를 보여줍니다. 제조 가능성 측면에서도 템플릿의 곡률 변동이 줄어들어, 순수한 공중합체에 비해 제조 목적에 더욱 적합한 템플릿을 생성하는 데 성공했습니다. 이 연구는 고차원의 DSA 기술 발전에 중요한 기여를 할 것으로 기대됩니다.



### Time-To-Inconsistency: A Survival Analysis of Large Language Model Robustness to Adversarial Attacks (https://arxiv.org/abs/2510.02712)
- **What's New**: 이번 연구에서는 대화 AI의 면역력을 평가하기 위해 최초의 포괄적인 생존 분석(survival analysis)을 제시합니다. 9개의 최신 대형 언어 모델(LLMs)에 대해 36,951회의 대화 턴을 분석하여 대화가 실패하는 것을 시간에 따라 변하는 과정으로 모델링합니다. 급격한 의미 변동이 대화 실패의 위험을 극적으로 증가시키는 반면, 점진적이며 누적적인 변동은 보호 효과를 나타내는 등의 발견을 통해, 생존 분석을 LLM의 강건성을 평가하는 강력한 패러다임으로 자리매김하였습니다.

- **Technical Details**: 이 연구에서는 생존 분석을 통해 대화 모델의 일관성을 재구성하였습니다. 대화 실패를 '이벤트'로 모델링하고 시간은 대화 턴의 순서에 따라 측정합니다. 사용된 모델링 기법으로는 Cox 비례 위험 모델, 가속화 실패 시간(Accelerated Failure Time, AFT) 모델, 비모수적 랜덤 생존 포리스트(Random Survival Forest)가 있습니다. 이를 통해 다양한 실패 시간 분포와 위험 함수를 고려하여 강건한 결론을 도출하였습니다.

- **Performance Highlights**: 생존 분석을 통한 이 연구의 결과는 LLM의 대화 강건성 평가에서 강력한 통찰력을 제공합니다. 특히 AFT 모델이 상호작용을 보여주며 우수한 성과를 달성했는데, 이는 변별력과 보정(calibration)에서 뛰어난 결과를 보여주었습니다. 이와 같은 결과는 기존의 단일 턴 평가 방식이 놓치기 쉬운 대화 내 응답 불일치의 위험을 정량화하여 대화 AI 시스템의 신뢰성을 높이는데 기여할 수 있을 것입니다.



### A Novel Unified Lightweight Temporal-Spatial Transformer Approach for Intrusion Detection in Drone Networks (https://arxiv.org/abs/2510.02711)
Comments:
          21 pages, 18 figures, 5 tables

- **What's New**: 드론 상업 및 산업 분야에 대한 통합이 증가함에 따라 사이버 보안 문제도 커지고 있습니다. 본 논문에서는 드론 네트워크에 특화된 TSLT-Net이라는 새로운 경량의 통합형 Temporal Spatial Transformer 기반 침입 탐지 시스템을 제안합니다. TSLT-Net은 셀프 어텐션 기법을 통해 네트워크 트래픽의 시간적 패턴과 공간적 의존성을 효과적으로 모델링하여 다양한 침입 유형을 정확하게 탐지합니다.

- **Technical Details**: TSLT-Net은 드론 네트워크의 동적이고 자원이 제한된 환경을 고려하여 설계되었습니다. 이 프레임워크는 간소화된 전처리 파이프라인을 포함하며, 단일 아키텍처 내에서 다중 클래스 공격 분류와 이진 이상 탐지를 모두 지원합니다. 또한, TSLT-Net은 0.04 MB의 최소 메모리 사용량과 9722개의 조정 가능한 파라미터를 유지합니다.

- **Performance Highlights**: ISOT Drone Anomaly Detection Dataset에서의 광범위한 실험 결과, TSLT-Net은 다중 클래스 탐지에서 99.99%의 정확도를 기록하고, 이진 이상 탐지에서는 100%의 정확도를 달성했습니다. 이 성능 결과는 TSLT-Net이 실시간 드론 사이버 보안을 위한 효과적이고 확장 가능한 솔루션임을 입증하며, 특히 미션 크리티컬 UAV 시스템의 엣지 장치에 적합합니다.



### RAMAC: Multimodal Risk-Aware Offline Reinforcement Learning and the Role of Behavior Regularization (https://arxiv.org/abs/2510.02695)
Comments:
          Under review as a conference paper at ICLR 2026, 21 pages, 8 figures. The HTML preview may misrender some figures; please refer to the PDF

- **What's New**: RAMAC (Risk-Aware Multimodal Actor-Critic) 프레임워크는 복합 목표를 결합하여 표현력이 뛰어난 생성자(actor)와 분포 기반 비평가(critic)를 결합한 점이 새롭습니다. 기존 리스크 회피 알고리즘들이 안전성을 보장하기 위해 보수적이거나 제한된 정책 클래스에 의존했던 반면, RAMAC는 높은 표현성을 유지하며 리스크를 조정하는 데 초점을 맞추고 있습니다. 이를 통해 분포적 위험(distributional risk)을 고려한 고급 정책 학습이 가능해졌습니다.

- **Technical Details**: RAMAC는 행동 복제(behavioral cloning)와 조건부 가치-at-위험(Conditional Value-at-Risk, CVaR) 경량 시 계산을 통해 리스크 감지 학습을 수행합니다. 이 프레임워크에서는 선행 작업에서 보인 데이터 외(ood) 행동 방문을 줄이는 동시에, 복잡한 다중 모달 환경에서의 높은 기대 수익을 목표로 합니다. 특히, RAMAC는 확산(diffusion) 및 흐름-매칭(flow-matching) 생성자를 사용하여 더 나은 성능을 달성하고 있습니다.

- **Performance Highlights**: 실험 결과, RAMAC의 두 가지 구현(RADAC와 RAFMAC)은 Stochastic-D4RL 벤치마크에서 기준 모델들에 비해 CVaR 성능이 우수하면서도 대부분의 작업에서 경쟁력 있는 평균 수익을 유지합니다. 이에 따라 RAMAC 프레임워크는 안전성을 유지하면서도 표현력이 뛰어난 정책 학습을 가능케 한다는 점에서 중요한 기여를 하고 있습니다. 이러한 결과는 리스크 인식 오프라인 RL 부문에서 RAMAC의 잠재력이 여전히 많이 남아있음을 시사합니다.



### Fine-Tuning Diffusion Models via Intermediate Distribution Shaping (https://arxiv.org/abs/2510.02692)
- **What's New**: 본 논문은 Diffusion 모델에서의 정책 기울기 방법(Policy Gradient Method)에 관한 새로운 접근 방식을 제안합니다. 저자들은 GRAFT라는 새로운 프레임워크를 통해 보상을 재구성하여 정책 기울기 최적화(Proximal Policy Optimization, PPO)를 포함하는 fine-tuning을 가능하게 합니다. 더불어, 이를 통해 중간 노이즈 레벨에서 분포를 형성하고, 보다 효과적인 fine-tuning을 수행하는 P-GRAFT를 도입합니다.

- **Technical Details**: P-GRAFT는 최종 생성물의 보상을 부분적으로 노이즈가 있는 생성물에 할당하여 중간의 denoising 단계까지만 fine-tuning하도록 설계되었습니다. 이 과정은 bias-변동 트레이드오프를 통해 수학적으로 설명되며, 다양한 생성 작업에서 질적 향상을 보입니다. 또한, inverse noise correction이라는 새로운 방법을 제안하여 명시적 보상 없이 흐름 모델을 개선할 수 있습니다.

- **Performance Highlights**: 제안된 GRAFT 프레임워크는 Stable Diffusion 2와 함께 사용되었으며, 이는 정책 기울기 방법에 비해 VQAScore에서 $8.81\%$의 상대적 개선을 달성했습니다. 또한, unconditional 이미지 생성 시 inverse noise correction으로 인해 생성된 이미지의 FID가 낮은 FLOPs/image에서 개선되었습니다. 전체적으로, 본 연구는 텍스트-이미지 생성, 레이아웃 생성 및 분자 생성 작업에서 중요한 품질 향상을 보여줍니다.



### Can Data-Driven Dynamics Reveal Hidden Physics? There Is A Need for Interpretable Neural Operators (https://arxiv.org/abs/2510.02683)
- **What's New**: 최근 신경 연산자(neural operators)는 함수 공간 사이의 매핑을 학습하는 강력한 도구로 떠올랐으며, 이를 통해 복잡한 역학의 데이터 기반 시뮬레이션이 가능해졌습니다. 본 논문에서는 신경 연산자를 공간(domain) 모델과 함수(functional) 모델의 두 가지 유형으로 분류하고, 물리적 원칙을 따르는 데이터 기반 동역학을 학습하는 방식에 대해 설명합니다. 또한, 신경 연산자가 데이터로부터 숨겨진 물리적 패턴을 학습할 수 있다는 점을 강조하지만, 이 설명 방법이 특정 상황에만 제한된다는 점도 지적합니다.

- **Technical Details**: 신경 연산자는 공간 모델과 함수 모델로 나눌 수 있습니다. 공간 모델은 그리드 기반 표현을 사용해 학습하고, 함수 모델은 함수 기초를 통해 학습합니다. 본 연구는 이러한 분류에 근거하여 다양한 관점을 제시하며, 특히 물리적 원칙을 준수하는 데이터 기반 동역학의 학습을 중점적으로 다룹니다. 또한, 신경 연산자가 학습하는 방식의 해석 가능성에 대한 기초적인 질문이 여전히 남아있음을 강조합니다.

- **Performance Highlights**: 본 논문에서는 간단한 이중 공간 다중 스케일 모델이 최신 성능을 달성할 수 있음을 보여주며, 이 모델이 복잡한 물리적 현상을 학습하는 데 큰 잠재력을 지니고 있음을 주장합니다. 다양한 신경 연산자 아키텍처가 제안된 바 있으며, 이들은 기후 모델링 및 유체 역학 등 다양한 분야에서 활용되고 있습니다. 연구의 결과는 과학적 응용을 위한 보다 효과적인 아키텍처 개발을 유도할 수 있으며, 숨겨진 물리적 현상을 더 많이 밝혀내기 위해 더 많은 노력이 필요하다는 점을 강조하고 있습니다.



### To Compress or Not? Pushing the Frontier of Lossless GenAI Model Weights Compression with Exponent Concentration (https://arxiv.org/abs/2510.02676)
- **What's New**: 본 논문은 수십억 개의 파라미터를 가진 Generative AI (GenAI) 모델의 효율적인 배치를 위해 저정밀(low-precision) 계산이 필수적이라는 점을 강조합니다. 저정밀 부동 소수점 형식 개발의 필요성을 논의하고, 이러한 형식이 수치적 안정성(numerical stability), 메모리 절약(memory savings), 하드웨어 효율성을 제공할 수 있음을 제시합니다. 특히, Exponent-Concentrated FP8 (ECF8)이라는 새로운 손실 없는 압축 프레임워크를 제안하며 실험 결과를 통해 최대 26.9%의 메모리 절약과 177.1%의 처리량 증가를 입증하였습니다.

- **Technical Details**: 부동 소수점 숫자는 IEEE-754 형식에서 부호 비트(sign bit), 지수(exponent), 그리고 가수(mantissa)로 구성됩니다. 저정밀 컴퓨팅에서는 지수가 표현 가능한 값의 동적 범위를 결정합니다. 본 논문에서 저자들은 GenAI 모델의 가중치에서 발생하는 지수 집중 현상(exponent concentration)을 이론적으로 및 경험적으로 분석하며, 이는 학습된 모델에서 지수가 저 엔트로피(low entropy)를 보이고 좁은 범위에 군집하는 현상으로 나타납니다. 이를 통해 FP4.67에 가까운 압축 한계를 증명하고 효과적인 FP8 포맷을 개발하게 됩니다.

- **Performance Highlights**: 제안된 ECF8 프레임워크는 다양한 GenAI 모델에서 최대 671B 파라미터를 지원하며, 메모리 사용량을 최대 26.9% 줄이고 처리량을 177.1% 가속화 함으로써 손실 없는 계산을 보장합니다. 이러한 결과는 ECF8가 메모리 압축을 추론 가속으로 변환할 수 있는 가능성을 입증하며, 학습된 모델의 통계적 법칙으로서 지수 집중 현상을 확립합니다. 이는 FP8 시대에서 손실 없는 저정밀 부동 소수점 설계를 위한 새로운 방향성을 제시합니다.



### HALO: Memory-Centric Heterogeneous Accelerator with 2.5D Integration for Low-Batch LLM Inferenc (https://arxiv.org/abs/2510.02675)
- **What's New**: 이 논문에서는 Low-batch LLM inference를 위한 혁신적인 이종 메모리 중심 엑셀러레이터인 HALO를 제안합니다. HALO는 Compute-in-DRAM (CiD)과 On-chip Analog Compute-in-Memory (CiM)를 통합하여 prefill과 decode 단계의 독특한 과제를 효율적으로 해결합니다. 또한, 이 논문은 LLM의 prefill 및 decode 단계에서 성능의 무역 관계를 분석하여 이종 디자인의 필요성을 강조합니다.

- **Technical Details**: HALO는 HBM 내부의 컴퓨팅 유닛을 통합하여 decode 단계의 성능을 가속화하고, 2.5D 통합을 통해 HBM과 동시 패키징된 on-chip analog CiM 엑셀러레이터를 사용하여 prefill 단계를 가속합니다. 이 과정에서 phase-aware mapping 전략을 도입하여 prefill과 decode 단계의 다양한 요구사항에 적절히 대응합니다. 최적화된 벡터 유닛을 이용하여 NON-GEMM 연산을 수행하는 방식으로, 자원 활용도를 높이고 있습니다.

- **Performance Highlights**: HALO는 LLaMA-2 7B 및 Qwen3 8B 모델에서 평가되었으며, LLM을 HALO에 매핑했을 때 AttAcc보다 최대 18배의 기하 평균 속도 향상을, CENT에 비해 2.5배의 속도 향상을 달성했습니다. 이는 HALO의 오랜 내용 컨텍스트 처리 및 낮은 배치 처리에서의 효율성을 입증합니다.



### AgenticRAG: Tool-Augmented Foundation Models for Zero-Shot Explainable Recommender Systems (https://arxiv.org/abs/2510.02668)
- **What's New**: 본 논문은 추천 시스템에서의 제한된 활용을 극복하기 위해 AgenticRAG라는 혁신적인 프레임워크를 소개합니다. 이 프레임워크는 외부 도구 호출(tool invocation), 지식 검색(knowledge retrieval), 사유 체인(chain-of-thought reasoning)을 결합하여 설명 가능한 추천을 제로샷(zero-shot)으로 제공합니다. 이를 통해 사용자가 추천에 대한 투명한 의사결정 과정을 경험할 수 있습니다.

- **Technical Details**: AgenticRAG 프레임워크는 RAG(검색 증강 생성) 방식을 활용하여 동적 정보 통합을 가능케 하며, 실시간 데이터 접근을 위한 외부 도구 호출 시스템과 투명한 의사결정을 위한 사유 엔진을 포함합니다. 이 구조는 과거의 특정 작업 훈련 없이 다양한 추천 시나리오에서 일반화할 수 있는 능력을 가진 기초 모델(foundation models)들을 활용합니다. 추천 과정은 사용자 질의 처리, 지식 검색, 도구 호출, 사유 과정을 병렬적으로 실행하여 개인 맞춤형 추천을 생성합니다.

- **Performance Highlights**: 세 가지 실제 데이터셋에서 실험 결과, AgenticRAG는 최첨단 기준선보다 일관된 성능 향상을 달성했습니다. Amazon Electronics에서는 NDCG@10에서 0.4% 개선, MovieLens-1M에서는 0.8% 개선, Yelp 데이터셋에서 1.6% 개선을 보였습니다. 이 프레임워크는 설명 가능성(explainability)에서 우수한 성능을 보이며, 전통적인 방법과 유사한 계산 효율성을 유지하고 있습니다.



### TutorBench: A Benchmark To Assess Tutoring Capabilities Of Large Language Models (https://arxiv.org/abs/2510.02663)
- **What's New**: TutorBench는 학생들이 학습 보조 자료로 대규모 언어 모델(LLMs)을 점점 더 많이 사용함에 따라, 이러한 모델들이 튜터링의 미묘한 부분을 효과적으로 처리할 수 있도록 개발된 데이터셋 및 평가 기준입니다. 이 데이터셋은 고등학교 및 AP 수준의 교육과정을 중심으로 한 1,490개의 샘플로 구성되어 있으며, 튜터링이 필요한 세 가지 일반적인 작업을 포함합니다. 이를 통해 LLMs의 핵심 튜터링 기술을 엄격하게 평가할 수 있도록 설계되었습니다.

- **Technical Details**: TutorBench는 적응형 설명 생성, 학생 작업에 대한 행동 피드백 제공, 효과적인 힌트 생성을 통한 능동적 학습 촉진 등 세 가지 튜터링 사용 사례에 중점을 둡니다. 각 샘플은 샘플별 평가 기준이 동반되어 있어 모델의 응답을 평가하는 데 사용됩니다. 또한, TutorBench는 LLM 판별을 활용한 신뢰성 있는 자동 평가 방법을 사용하여 보다 세분화된 평가가 가능하도록 구조화되어 있습니다.

- **Performance Highlights**: 16개의 최신 LLM 모델을 TutorBench에서 평가한 결과, 어떤 모델도 56% 이상의 점수를 기록하지 못했습니다. Claude 모델은 능동 학습 지원에서 가장 우수한 성과를 보였지만, 나머지 두 가지 사용 사례에서는 뒤처졌습니다. TutorBench의 출시는 차세대 AI 튜터 개발을 위한 포괄적이고 비포화된 기준을 제공하여 LLM의 향후 발전에 기여할 것으로 기대됩니다.



### When Researchers Say Mental Model/Theory of Mind of AI, What Are They Really Talking About? (https://arxiv.org/abs/2510.02660)
- **What's New**: 이 논문은 인공지능(AI) 시스템이 ToM (Theory of Mind)이나 정신 모델을 가지고 있다고 주장할 때의 본질적인 혼동을 지적합니다. 연구자들은 행동 예측 및 편향 교정을 논의하는 대신 실제 정신 상태가 아닐 뿐 아니라, ToM 실험에서 LLMs (Large Language Models)의 인간 수준 성과를 모방에 기반한 것으로 간주합니다. 이 논문은 AI 시스템을 개별 인간 인지 테스트에 적용하는 현재의 평가 패러다임이 결함이 있음을 주장하며, 인간과 AI의 상호 작용 동역학을 강조하는 쌍방향 ToM 프레임워크로의 이동을 제안합니다.

- **Technical Details**: 논문은 LLMs에 대한 인지 테스트에서의 근본적인 오해를 다룹니다. 연구자들은 LLMs가 인간과 같은 방식으로 정보를 처리한다고 가정하여 ToM 테스트 및 추론 문제를 설계했지만, Kambhampati(2024)의 주장에 따르면 LLMs는 본질적으로 'n-gram 모델을 강화한 것'에 불과하다는 점을 강조합니다. 또한 LLM이 인간 행동의 통계적 패턴을 재현해 내는 방식이 실제 상호작용의 진정한 인지 과정을 반영하지 않음을 지적합니다.

- **Performance Highlights**: LLMs는 ToM 과제를 통계적 패턴 매칭을 통해 놀라운 성과를 달성했지만, 이는 인간의 경험과는 근본적으로 다릅니다. 중요한 것은 AI가 ToM 시험 성과를 얼마나 잘 기록하는지가 아니라, 자율 팀에서의 인간-AI 시스템이 어떻게 함께 작용하는지를 연구하는 것입니다. 인간과 AI의 상호 작용에서 진정한 협력을 통해 AI는 ToM이 필요하지 않지만, 상호 적응 및 이해를 지원하는 시스템이 필요하다는 점을 강조합니다.



### Automatic Building Code Review: A Case Study (https://arxiv.org/abs/2510.02634)
- **What's New**: 이 연구는 리소스가 제한된 지역 관할구역에서 건축 관련 문서의 수작업 검토가 labor-intensive(노동 집약적) 및 error-prone(오류가 발생하기 쉬운) 문제를 해결하기 위해 자동화된 코드 검토(ACR) 솔루션을 제시합니다. 새로운 에이전트 기반 프레임워크는 BIM 기반 데이터 추출을 자동 검증과 통합하여 Building Information Modeling(건축 정보 모델링)과 Large Language Models(대규모 언어 모델)의 활용을 극대화합니다.

- **Technical Details**: 이 프레임워크는 LLM(대규모 언어 모델) 활성화 에이전트를 사용하여 서로 다른 파일 유형에서 기하학, 일정 및 시스템 속성을 추출하고, 이를 건축 법규 검토를 위해 두 가지 상호 보완적인 메커니즘—US Department of Energy COMcheck 엔진에 대한 직접 API 호출 및 규칙 조항에 대한 RAG(검색 증강 생성) 기반 추론—을 통해 처리합니다.

- **Performance Highlights**: 성능 평가는 기하학 속성 자동 추출, 운영 일정 분석 및 ASHRAE Standard 90.1-2022에 따른 조명 허용 값의 검증을 포함한 사례 demonstrations(사례 시연)을 통해 이루어졌습니다. 비교 성능 테스트에서는 GPT-4o가 효율성과 안정성 간의 최상의 균형을 달성한 반면, 작은 모델은 불일치 또는 실패를 보였습니다. 이 결과는 MCP 에이전트 파이프라인이 RAG 추론 파이프라인보다 rigor(엄격함)과 reliability(신뢰성)에서 우수함을 확인하는 데 기여합니다.



### A Trajectory Generator for High-Density Traffic and Diverse Agent-Interaction Scenarios (https://arxiv.org/abs/2510.02627)
- **What's New**: 자율 주행에서 정확한 궤적 예측은 안전한 움직임 계획과 충돌 회피를 위한 기본 요소입니다. 기존 데이터셋은 낮은 밀도의 시나리오와 간단한 직선 주행 행동에 치우친 분포 문제를 겪고 있어 이로 인해 모델의 일반화에 어려움이 있습니다. 본 연구에서는 이러한 문제를 해결하기 위해 고밀도와 다양한 행동을 동시에 향상시키는 새로운 궤적 생성 프레임워크인 HiD2를 제안합니다.

- **Technical Details**: HiD2는 지속적인 도로 환경을 구조화된 그리드 표현으로 변환하여 세밀한 경로 계획과 명시적인 충돌 탐지 및 다중 에이전트 조정을 지원합니다. 이 프레임워크는 규칙 기반의 의사 결정 트리거와 Frenet 기반의 궤적 평활화, 동적 실현 가능성 제약을 결합하여 동작하는 메커니즘을 포함하고 있습니다. 이를 통해 데이터에서 자주 나타나지 않는 높은 밀도의 상황과 복잡한 상호작용을 포함하는 현실적인 시나리오를 생성할 수 있습니다.

- **Performance Highlights**: 대규모 Argoverse 1 및 Argoverse 2 데이터셋에 대한 광범위한 실험을 통해 우리의 방법이 에이전트 밀도와 행동 다양성을 모두 크게 향상시키면서도 동작의 현실성과 시나리오 안정성을 유지함을 입증했습니다. HiD2 데이터를 사용하여 훈련된 궤적 예측 모델은 기존 데이터셋으로 훈련한 모델보다 고밀도 시나리오에서의 성능이 향상되었습니다. 이 연구는 안전에 중요한 희귀 행동의 대표성을 높일 수 있는 가능성을 보여줍니다.



### MINERVA: Mutual Information Neural Estimation for Supervised Feature Selection (https://arxiv.org/abs/2510.02610)
Comments:
          23 pages

- **What's New**: 기존의 feature 필터는 통계적 쌍-쌍 (pair-wise) 의존성 메트릭스에 기반하여 feature-target 관계를 모델링하지만, 이를 통해 타겟이 고차원 feature 상호작용에 의존하는 경우에는 실패할 수 있습니다. 본 논문에서는 feature와 타겟 간의 상호 정보를 신경망(neural network)을 기반으로 추정하는 새로운 supervised feature selection 방법인 Mutual Information Neural Estimation Regularized Vetting Algorithm(MINERVA)을 소개합니다. 이 방법은 두 단계로 구성된 프로세스를 통해서 representation learning과 feature selection을 분리하여 보다 나은 일반화와 feature 중요성 표현을 보장합니다.

- **Technical Details**: MINERVA는 신경망을 사용하여 상호 정보를 근사화하고, 희소성 유도 정규화자(sparsity-inducing regularizers)를 포함한 정교하게 설계된 손실 함수(loss function)를 사용하여 feature selection을 수행합니다. 이 방법은 고차원 데이터 세트에서의 변수 간의 복잡한 의존 구조를 학습하며, feature subset을 앙상블(ensemble)로 평가하여 효과적으로 이 관계를 포착합니다. 논문에서는 합성 및 실제 사기 데이터 세트에 대한 실험 결과를 통해 제안한 방법의 효율성을 입증합니다.

- **Performance Highlights**: MINERVA는 기존의 feature selection 방법들과 비교하여 더 나은 성능을 보였습니다. 실험 결과, MINERVA는 정확한 해결책을 제공할 수 있는 능력을 갖추고 있으며, 높은 차원의 경우에도 잘 작동하는 것으로 나타났습니다. 이러한 결과는 MINERVA가 신경망 기반의 상호 정보 추정 방식을 효과적으로 활용하고 있음을 보여줍니다.



### How Confident are Video Models? Empowering Video Models to Express their Uncertainty (https://arxiv.org/abs/2510.02571)
- **What's New**: 이 논문은 비디오 생성 모델의 불확실성 정량화(Uncertainty Quantification, UQ)에 대한 최초의 연구를 제안합니다. 기존의 텍스트-비디오 모델이 사용자 의도와 일치하지 않거나 잘못된 정보를 토대로 영상을 생성하는 '환각(hallucination)' 문제를 해결하고자 합니다. 새로운 방법론인 S-QUBED(Semantically-Quantifying Uncertainty with Bayesian Entropy Decomposition)를 통해 이는 더욱 정확하게 접근할 수 있습니다.

- **Technical Details**: S-QUBED는 비디오 생성 모델의 불확실성을 정밀하게 분해할 수 있는 블랙박스 UQ 방법으로, 조건부 확률을 모형화하여 두 단계로 비디오 생성을 모델링합니다. 이 방법은 조건부 독립성을 기반으로 Latent Variable(z)를 활용하여 예측 불확실성을 Aleatoric(우연적 불확실성)과 Epistemic(지식 기반 불확실성)으로 나눕니다. 이를 통해 입력 프롬프트의 모호함이나 모델의 지식 부족으로 인한 불확실성을 효과적으로 구별할 수 있습니다.

- **Performance Highlights**: S-QUBED는 다양한 비디오 생성 작업에서 불확실성 추정치를 조정하는 성능을 보여주며, 태스크 정확도와 부정적 상관 관계를 가지고 있습니다. 논문에 제시된 실험 결과는 S-QUBED가 비디오 모델의 불확실성을 정량화하는 데 유용하고 효과적임을 입증합니다. 이는 향후 비디오 모델의 안전성을 높이는 데 기여할 것으로 기대됩니다.



### Oracle-RLAIF: An Improved Fine-Tuning Framework for Multi-modal Video Models through Reinforcement Learning from Ranking Feedback (https://arxiv.org/abs/2510.02561)
Comments:
          Proceedings of the 39th Annual Conference on Neural Information Processing Systems, ARLET Workshop (Aligning Reinforcement Learning Experimentalists and Theorists)

- **What's New**: 최근 대형 비디오-언어 모델(VLMs)에서 인공지능 피드백을 활용한 강화 학습(RLAIF) 프레임워크를 제안하며, 기존의 인간 피드백을 대체했다. Oracle-RLAIF는 후보 모델의 응답을 점수화하는 대신 순위를 매기는 오라클 랭커(Oracle ranker)를 사용하여 비용 효율성을 높인다. 이 접근법은 데이터 효율성을 개선하고 다양한 멀티모달 비디오 모델의 정렬을 촉진한다.

- **Technical Details**: Oracle-RLAIF는 기존의 보상 모델에 의존하지 않고, 단순히 응답의 질을 평가하는 오라클 모델에 의존한다. 이는 비디오 언어 모델(VLMs)의 교육 및 학습 과정을 유연하게 바꿔주며, 다양한 시나리오에 적용 가능하다. 또한, Group Relative Policy Optimization(GRPO)의 새로운 수정 사항인 GRPO_{rank}를 도입하여 순위 기반 손실 함수를 직접적으로 최적화할 수 있도록 한다.

- **Performance Highlights**: Oracle-RLAIF는 여러 비디오 평가 데이터셋에서 기존의 최첨단 비디오-언어 모델을 능가하는 성능을 보인다. 이 모델은 현존하는 미세 조정 기법과 비교할 때 비디오 이해 성능을 지속적으로 개선하며, 순위 기반 피드백을 통해 강화 학습을 수행하는 유연하고 데이터 효율적인 프레임워크를 제시한다.



### ToolTweak: An Attack on Tool Selection in LLM-based Agents (https://arxiv.org/abs/2510.02554)
- **What's New**: 이 논문은 LLMs (대형 언어 모델)의 툴 사용에 대한 새로운 공격 방식인 ToolTweak를 소개합니다. ToolTweak는 악의적인 검색어 최적화에 의해 툴의 선택률을 증가시킬 수 있는 경량 자동 공격으로, 대칭적이지 않은 툴 선택을 초래하는 심각한 취약점을 확인하였습니다. 이 연구는 공정성과 신뢰성을 위한 툴 선택 시스템의 위험요소를 드러내고, 이를 해결하기 위한 두 가지 방어 메커니즘을 평가합니다.

- **Technical Details**: 이 논문에서 설명하는 ToolTweak는 툴 이름과 설명을 조작하여 선택률을 약 20%에서 81%까지 크게 증가시킬 수 있습니다. 툴 사용의 분포적 변화에 대한 심층 분석을 제공하며, 악의적인 조작이 특정 툴의 사용을 어떻게 왜곡할 수 있는지를 보여줍니다. 이를 통해 LLM 이코시스템에서의 공정성과 보안을 강화하기 위한 무대로 자리매김하고 있습니다.

- **Performance Highlights**: ToolTweak의 공격성이 다양한 모델과 작업에서 매우 효과적임을 입증하였습니다. 툴 선택률의 전반적인 향상과 함께, 기존 툴 시장의 분포를 변화시키는 결과를 초래하여 경쟁과 공정성에 대한 리스크를 제기합니다. 또한, 우리는 방어 메커니즘인 패러프레이징(paraphrasing)와 혼란도 필터링(perplexity filtering)을 평가하여 선택의 편향성을 줄일 수 있음을 보여줍니다.



### Knowledge-Graph Based RAG System Evaluation Framework (https://arxiv.org/abs/2510.02549)
- **What's New**: 이번 연구에서는 Retrieval Augmented Generation (RAG) 시스템의 평가 방법을 혁신적으로 개선하기 위해 Knowledge Graph (KG)를 기반으로 한 평가 프레임워크를 제안합니다. 이 새로운 접근 방식은 기존의 RAGAS 평가 기준을 확장하여 멀티-hop reasoning과 의미적 커뮤니티 클러스터링을 통합하였습니다. 연구 결과, KG 기반 메트릭이 기존 RAGAS보다 사실성과 신뢰성 평가에서 더 우수한 성능을 보여주었습니다.

- **Technical Details**: RAG 시스템은 두 가지 주요 구성 요소인 retriever와 generator로 이루어져 있습니다. Retriever는 주어진 입력에 따라 관련 정보를 검색하고, 이를 바탕으로 generator가 최종 출력을 생성합니다. 그러나 기존의 RAG는 여러 정보 출처를 통합하는 데 한계가 있어, 최신 연구에서는 KG를 활용한 GraphRAG 아키텍처가 개발되었습니다. 이를 통해 정보의 통합 구조를 개선하고 반응 품질과 정확도를 높였습니다.

- **Performance Highlights**: 연구 결과, KG 기반의 새로운 평가 시스템은 RAGAS에서 측정한 점수와 인간의 판단 결과 간의 높은 상관관계를 나타냈습니다. 특히, KG 기반 방법은 극단적인 상황에서 질문을 비교할 때 RAGAS를 초월하며 더 정밀한 사실 일치성과 의미적 일관성을 포착할 수 있는 능력을 보여주었습니다. 이러한 결과는 KG 기반 접근 방식의 효과적인 특성을 강조하며, 향후 RAG 시스템 평과의 방향성을 제시합니다.



### PHORECAST: Enabling AI Understanding of Public Health Outreach Across Populations (https://arxiv.org/abs/2510.02535)
- **What's New**: 이번 연구에서는 다양한 개인과 커뮤니티가 설득적인 메시지에 어떻게 반응하는지를 이해하는 것이 개인화되고 사회적으로 인식 가능한 머신 러닝 향상에 잠재적인 가능성을 제공한다고 강조합니다. 이 논문에서는 PHORECAST라는 새로운 멀티모달 데이터셋을 소개하여 개인차원 행동 반응과 커뮤니티 전체의 참여 패턴을 공공 건강 메시지에 대해 세밀하게 예측할 수 있도록 합니다. 이 데이터셋은 고급 AI 시스템이 이질적인 공공 감정과 행동을 어떻게 모방, 해석 및 예측할 수 있는지를 철저하게 평가할 수 있는 지원을 제공합니다.

- **Technical Details**: PHORECAST 데이터셋은 공공 건강 캠페인에 대한 인간 반응을 예측하기 위해 설계된 멀티모달 데이터셋입니다. 이 데이터셋은 다양한 인구 통계 및 심리적 요인에 따라 구성된 설문 응답을 포함하고 있습니다. 1,000명 이상의 참여자들이 제공한 30,000개 이상의 상세한 반응이 있으며, 각 반응은 감정, 행동 의도 등을 반영하고, 개별의 성격 및 인구 통계적 데이터와 결합되어 있습니다.

- **Performance Highlights**: PHORECAST 데이터셋은 LLM 모델이 개인의 취향과 가치에 더 잘 맞출 수 있도록 합니다. 그 결과, 이는 공공 건강 메시지에 대한 다양한 인구 집단의 반응을 예측하는 데 강력한 일반화 능력을 보여줍니다. 연구자들은 이 데이터셋을 활용하여 개인적 차원이 어떻게 반영되는지를 시뮬레이션하고, 인구 통계와 심리적 요인에 기반한 예측 모델을 훈련하는 두 가지 주요 용도를 입증했습니다.



### From Pixels to Factors: Learning Independently Controllable State Variables for Reinforcement Learning (https://arxiv.org/abs/2510.02484)
- **What's New**: 이 연구에서는 Action-Controllable Factorization (ACF)라는 대조적 학습 접근법을 통해 독립적으로 제어 가능한 잠재 변수를 식별하는 새로운 방법을 제안합니다. ACF는 에이전트가 고차원 관찰만을 통해 переход하는 상황에서 결정을 내리는 데 필요한 정보를 추출하는 데 중점을 두고 있습니다. 이 방법은 Taxi, FourRooms, MiniGrid-DoorKey와 같은 세 가지 기준 데이터셋에서 사실적인 제어 가능 요소를 자동으로 복구하여 기존의 변별화 알고리즘보다 더 나은 성능을 보여주었습니다.

- **Technical Details**: ACF는 에이전트의 행동에 의해 영향을 받는 상태 요소를 분리하여 그들 간의 상관관계를 정확하게 평가합니다. 대조적 학습의 목표를 통해, ACF는 상태 변수가 환경의 자연적 동역학에 대한 에이전트 행동의 다음 상태 예측에서 발생하는 차이를 통해 분리됩니다. 이러한 접근법은 에이전트가 관찰하는 픽셀 정보를 사용하여 제어 가능한 상태 변수를 정확히 식별할 수 있도록 합니다.

- **Performance Highlights**: ACF는 기존의 Disentanglement 알고리즘과 비교하여, 일반적인 RL 도메인에서 작업을 수행하면서 더욱 높은 정확도를 기록하였습니다. 실험 결과, ACF는 고차원의 픽셀 관찰에서 직접 제어 가능한 요소를 복구하는 데 성공하며 전문가가 설계한 표현과 일치하는 결과를 보여주었습니다. 이러한 성과는 ACF의 효과성을 입증하며, 에이전트의 학습 과정을 보다 샘플 효율적으로 만듭니다.



### Litespark Technical Report: High-Throughput, Energy-Efficient LLM Training Framework (https://arxiv.org/abs/2510.02483)
Comments:
          14 pages

- **What's New**: 본 논문에서는 Litespark라는 새로운 사전 훈련 프레임워크를 소개합니다. 이 프레임워크는 transformer architecture의 attention과 MLP layers에서 비효율성을 해결하여 훈련 시간을 단축하고 에너지 소비를 감소시키는 방법을 제안합니다. Litespark는 모델 성능을 극대화하면서도 기존 transformer 구현과의 호환성을 유지합니다.

- **Technical Details**: Litespark는 두 단계로 최적화를 수행합니다: 첫 번째는 architectural optimization으로 attention과 MLP 블록을 최적화하고, 두 번째는 algorithmic optimization으로 GPU당 FLOPs를 증가시키기 위해 순방향 및 역방향 연산을 최적화합니다. 이 프레임워크는 기존의 FlashAttention 기법과 같은 기술적 진보 위에 성능을 더할 수 있습니다.

- **Performance Highlights**: Litespark는 훈련 처리량을 2배에서 6배까지 증가시키고, 사전 훈련 과정에서 에너지 소비를 55%에서 83%까지 감소시킵니다. 이러한 최적화는 다양한 모델 아키텍처와 하드웨어에 적용 가능하여, 많은 산업계 응용에 유용할 것으로 기대됩니다.



### SIMSplat: Predictive Driving Scene Editing with Language-aligned 4D Gaussian Splatting (https://arxiv.org/abs/2510.02469)
- **What's New**: 새로운 접근 방식인 SIMSplat은 Predictive driving scene editor로, 자연어 프롬프트를 통해 직관적인 환경 조정이 가능하다. 이 시스템은 Gaussian으로 재구성된 장면과 언어를 정렬하여 도로 물체에 대한 직접적인 쿼리를 지원한다. 이러한 혁신은 한 가지의 에이전트에 국한된 편리한 조정을 넘어 여러 에이전트의 상호작용을 고도화하는 데 기여한다.

- **Technical Details**: SIMSplat은 motion-aware language embeddings를 통합하여 3D Gaussian 장면을 쿼리 및 조작할 수 있게 한다. LLM(large language model) 에이전트는 사용자의 프롬프트를 해석하여 객체를 추가, 제거 또는 수정할 수 있도록 한다. 또한, multi-agent motion prediction 모델을 통해 주변 에이전트의 움직임을 자연스럽게 반영하여 시뮬레이션의 현실감을 더욱 강화한다.

- **Performance Highlights**: SIMSplat은 Waymo 데이터셋에서 실험을 통해 탁월한 편집 및 시뮬레이션 능력을 입증하였다. 도로 물체 쿼리에서 정확성 기반의 기반선보다 61.2% 우수한 성능을 기록하며, 시뮬레이터 중 가장 높은 작업 완료율을 달성하였다. 게다가, 다중 에이전트 경로 개선 기능을 통해 예측 시뮬레이션에서도 가장 낮은 충돌 및 실패율을 기록하였다.



### CLARITY: Clinical Assistant for Routing, Inference, and Triag (https://arxiv.org/abs/2510.02463)
Comments:
          Accepted to EMNLP 2025 (Industrial Track)

- **What's New**: CLARITY는 환자를 전문의에게 신속하게 안내하고, 임상 상담을 지원하며, 환자 상태의 심각도를 평가하는 AI 기반 플랫폼입니다. Finite State Machine(FSM)과 Large Language Model(LLM)을 결합하여 증상을 분석하고 적절한 전문의에 대한 추천을 우선시하는 하이브리드 아키텍처를 가지고 있습니다. 최근에 대규모 국가 간 병원 IT 플랫폼에 통합되어, 두 달 동안 5만 5천 개 이상의 사용자 대화가 완료되어, 전문가 검증을 위한 2천 5백 개의 대화가 주목받았습니다.

- **Technical Details**: CLARITY 시스템은 대화 관리에 FSM을 이용하고 요청 처리를 위한 마이크로서비스로 구성된 하이브리드 아키텍처로 설계되었습니다. FSM은 사용자 입력과 맥락에 따라 대화의 상태와 전환을 관리하며, 자연어 생성 및 복잡한 질의 처리를 담당하는 텍스트 생성 서비스와 입력 텍스트 분류 및 의사 결정을 담당하는 결정 서비스가 포함되어 있습니다. 마이크로서비스 아키텍처는 모듈성과 확장성을 보장하며, 서비스 간 지연 시간을 최소화하고 최적화를 통해 실시간 성능을 제공합니다.

- **Performance Highlights**: CLARITY는 환자 상태를 자동으로 평가하고 진단 가설을 생성하여 환자 안내 및 전문의 추천을 최적화합니다. 실험 결과, CLARITY는 첫 시도에서의 라우팅 정확도 면에서 인간의 성능을 초과하며, 상담에서 소요되는 시간을 3배 이상 단축시킵니다. 이러한 시스템이 의료 현장에 안전하고 효과적으로 통합될 수 있도록 하는 다양한 기술적 해결책들이 필요하다는 점을 강조합니다.



### Market-Based Data Subset Selection -- Principled Aggregation of Multi-Criteria Example Utility (https://arxiv.org/abs/2510.02456)
- **What's New**: 이번 연구는 교육 데이터의 유용한 소 subset을 선택하는 어려움을 해결하기 위한 시장 기반 선택기의 도입을 제안합니다. 이 선택기는 각 샘플의 유용성을 가격으로 평가하며, 다양한 신호들이 가격을 결정하는 트레이더 역할을 합니다. 또한, 주제별 정규화를 통해 안정성을 제공하고, Token 예산을 명시적으로 관리합니다. 이러한 접근 방식은 비용을 최소화하고 적은 컴퓨팅 자원을 활용하는 동시에 데이터 커랜션을 보다 효율적으로 수행합니다.

- **Technical Details**: 연구에서는 각 교육 아이템을 '계약'으로 보고, Logarithmic Market Scoring Rule (LMSR)을 통해 가격을 책정합니다. 샘플의 유용성을 반영하는 가격은 여러 신호의 조합을 통해 결정되며, 각 신호는 주제에 따라 표준화됩니다. 또한, 선택 과정에서 token 길이에 따른 편향을 조절할 수 있는 파라미터가 포함되어 있으며, 이 시스템은 convex cost와 maximum-entropy 원리를 바탕으로 작동합니다.

- **Performance Highlights**: GSM8K와 AGNews 데이터셋을 통해 실험한 결과, 제안된 시장 기반 선택기가 동일한 신호의 강력한 기준선과 동등한 성능을 보이며, 선택 오버헤드도 0.1 GPU-시간 미만으로 유지됩니다. 특히, AGNews 데이터셋에서 상위 5-25%의 문서를 선택할 때 안정성과 정확성을 향상시킵니다. 이러한 프레임워크는 LLMs의 멀티 신호 데이터 커랜션을 통합하는데 유용합니다.



### How to Train Your Advisor: Steering Black-Box LLMs with Advisor Models (https://arxiv.org/abs/2510.02453)
- **What's New**: 본 논문에서는 Advisor Models라는 새로운 프레임워크를 소개합니다. 이 모델은 경량의 파라메트릭 정책으로, 블랙박스 모델에 대한 자연어 조언을 생성하여 반응적으로 조정하는 기능을 가지고 있습니다. Advisor Models는 단일 고정 프롬프트의 한계를 극복하고, 다양한 입력과 환경에 적응할 수 있는 다이나믹한 최적화를 가능하게 합니다.

- **Technical Details**: Advisor Models는 강화 학습(Reinforcement Learning)을 통해 최적화되며, 사용자 입력과 블랙박스 모델 사이에 위치하여 상황에 맞는 가이드를 생성합니다. 이 과정에서는 그룹 상대 정책 최적화(Group Relative Policy Optimization)를 사용하여 관찰된 보상에 기반하여 advisor를 업데이트합니다. 블랙박스 모델의 파라미터에 접근할 필요 없이, advice의 출력을 활용하여 블랙박스 모델을 제어합니다.

- **Performance Highlights**: Advisor Models는 개인화 및 도메인 적응이 필요한 다양한 환경에서 테스트되었으며, 특히 사용자 특정 규칙을 완벽히 학습하며 94-100%의 보상을 달성하는 높은 성능을 보여주었습니다. 또한, 다양한 블랙박스 모델 간의 advisor 전이도 성능 저하 없이 이루어지며, 분포 외 입력에 대한 강건성을 유지하는 결과를 보였습니다.



### Dynamic Target Attack (https://arxiv.org/abs/2510.02422)
- **What's New**: 본 논문에서는 Dynamic Target Attack (DTA)라는 새로운 jailbreaking 프레임워크를 제안합니다. 기존의 gradient 기반 jailbreak 공격이 고정된 affirmative response를 유도하는 데 초점을 맞추었다면, DTA는 목표 LLM의 자체 응답을 기반으로 대상을 최적화하는 접근 방식을 채택합니다. DTA는 유해한 응답을 직접 샘플링하고, 이 중 가장 유해한 응답을 임시 목표로 선택하여 최적화를 진행합니다.

- **Technical Details**: DTA는 출력 분포에서 유해한 응답을 샘플링하고, 이 과정에서 평균적인 optimal adversarial prompt를 찾는 최적화 사이클을 구성합니다. 기존 방법들이 목표와 출력 간의 모호함으로 인해 수천 번의 반복을 요구했던 반면, DTA는 200회의 최적화로도 평균 87%의 공격 성공률(ASR)을 달성할 수 있습니다. 이 진행 과정은 대상을 반복적으로 샘플링하고 최적화하여 수행됩니다.

- **Performance Highlights**: DTA는 화이트 박스 환경에서 87% 이상의 공격 성공률을 달성하는 데 200회의 반복만 필요하며, 이는 기존 최첨단 방법들보다 15% 이상 개선된 성과입니다. 블랙 박스 환경에서도 DTA는 Llama-3-8B-Instruct를 대체 모델로 사용하여 85%의 ASR을 달성하였으며, 이는 기존 방법들과 비교할 때 25% 이상 상승한 결과입니다.



### NEURODNAAI: Neural pipeline approaches for the advancing dna-based information storage as a sustainable digital medium using deep learning framework (https://arxiv.org/abs/2510.02417)
- **What's New**: 이번 연구에서는 DNA 데이터 저장을 위한 새로운 모듈형 엔드 투 엔드 프레임워크 NeuroDNAAI를 제안합니다. 이 프레임워크는 양자 병렬성(quantum parallelism) 개념을 활용하여 인코딩의 다양성과 강인성을 향상시키고, 생물학적 제약조건 및 딥러닝을 통합하여 DNA 저장에서의 오류 완화를 개선합니다. NeuroDNAAI는 이진 데이터 스트림을 상징적 DNA 시퀀스에 인코딩하고, 불완전한 채널을 통해 전송 후 높은 정확도로 재구성합니다.

- **Technical Details**: 이 시스템은 디지털 정보를 DNA 시퀀스에 인코딩하고, 합성 및 시퀀싱 오류를 시뮬레이션하는 변환 가능한 노이즈 모델을 통해 전송한 후, 인코더-디코더 아키텍처로 재구성합니다. 전통적인 부호 이론(coding theory) 접근 방식과 달리, 변환기(Transformer) 기반 모델은 오류 패턴을 직접 학습하고 다양한 노이즈 수준에 적응하는 구조를 제공하여, 삽입-삭제 오류 처리에 강력한 성능을 자랑합니다.

- **Performance Highlights**: NeuroDNAAI는 기존의 팅킹 방법이나 규칙 기반 접근 방식이 실제 노이즈를 효과적으로 처리하지 못하는 반면, 높은 정확도로 결과를 달성했습니다. 실험 결과는 텍스트와 이미지 모두에서 낮은 비트 오류율(bit error rate)을 보였으며, 이 프레임워크는 이론적, 실용적, 시뮬레이션의 통합을 통해 스케일러블한 생물학적으로 유효한 DNA 저장을 가능하게 합니다.



### Cross-Platform DNA Methylation Classifier for the Eight Molecular Subtypes of Group 3 & 4 Medulloblastoma (https://arxiv.org/abs/2510.02416)
Comments:
          9 pages, 5 figures, 5 tables

- **What's New**: 이 연구는 악성 소아 뇌암인 수막종(medulloblastoma)의 새로운 분자 아형을 식별하는 기초를 제공하는 DNA 메틸레이션 기반의 크로스 플랫폼(cross-platform) 머신러닝 분류기(classifier)를 소개합니다. 이 분류기는 Group 3과 Group 4에 속하는 여덟 가지 새로운 아형을 구별할 수 있는 능력을 가지고 있으며, 이는 개인 맞춤형 치료 전략을 발전시키는 데 도움을 줍니다.

- **Technical Details**: 연구에서 제안한 분류기는 HM450 및 EPIC 메틸레이션 어레이(methylation array) 샘플에서 이들 아형을 식별할 수 있도록 설계되었습니다. 두 개의 독립적인 테스트 세트에서 모델은 균형 잡힌 정확도(balanced accuracy) 0.957과 가중 F1 점수(weighted F1) 0.95를 달성하여 플랫폼 간 일관성을 보여줍니다. 이는 임상 시험, 개인 맞춤 치료 개발 및 환자 모니터링을 지원하기 위해 필수적입니다.

- **Performance Highlights**: 연구에서 제안한 모델은 최초의 크로스 플랫폼 솔루션으로, 기존 플랫폼의 호환성을 유지하면서 최신 플랫폼에의 적용 가능성을 확장합니다. 향후 웹 애플리케이션을 통해 배포될 경우, 이 분류기는 이러한 아형에 대한 첫 번째 공개 분류기가 될 잠재력을 가지고 있습니다. 전반적으로 이 연구는 정밀 의학(precision medicine)을 발전시키고 수막종 subgroups 3과 4에 대한 임상 결과 개선에 기여하는 방향으로 나아가고 있습니다.



### RainSeer: Fine-Grained Rainfall Reconstruction via Physics-Guided Modeling (https://arxiv.org/abs/2510.02414)
- **What's New**: 이 논문은 강수 필드를 재구성하는 혁신적인 방법인 RainSeer를 소개합니다. RainSeer는 레이더 반사도를 물리적으로 기초한 구조적 사전으로 해석하며, 강수가 발생하는 시점, 장소 그리고 방법을 포착합니다. 이 프레임워크는 기존의 기상 관측 장치가 간과하는 급격한 전이와 지역적 극단을 포착하는 데 특화되어 있습니다.

- **Technical Details**: RainSeer는 두 가지 기초 구조로 구성되어 있습니다: 구조-점 매퍼(Structure-to-Point Mapper)와 지리적 인식 강수 디코더(Geo-Aware Rain Decoder)입니다. 구조-점 매퍼는 메조스케일 레이더 구조를 지역적 강수로 변환하며, 지리적 인식 강수 디코더는 수증기의 하강, 용해 및 증발 과정을 모델링하여 물리적 일관성을 유지합니다. 이러한 제어는 인과적 시공간 주의 메커니즘(Causal Spatiotemporal Attention)으로 수행됩니다.

- **Performance Highlights**: RainSeer는 두 개의 공공 데이터셋인 RAIN-F(한국, 2017-2019)와 MeteoNet(프랑스, 2016-2018)에서 기존의 최첨단 기법들과 비교하여 일관된 개선을 보였습니다. MAE(Mean Absolute Error)를 각각 13.31%와 70.71% 감소시키고, NSE(Nash-Sutcliffe Efficiency)를 60.75%와 22.56% 향상시키며, 구조적 충실도를 크게 높였습니다.



### Extreme value forecasting using relevance-based data augmentation with deep learning models (https://arxiv.org/abs/2510.02407)
- **What's New**: 이 연구에서는 Extreme 값 예측을 위한 데이터 증강(data augmentation) 프레임워크를 제시합니다. 주로 Generative Adversarial Networks(GANs) 및 Synthetic Minority Oversampling Technique(SMOTE)와 같은 데이터 증강 모델을 활용하여 Deep Learning 모델의 예측 능력을 개선하려 합니다. 다양한 Deep Learning 모델인 Conv-LSTM 및 BD-LSTM를 사용하여 극단적인 값을 예측하는 방법을 검토하고 있습니다.

- **Technical Details**: Extreme 값 이론(extreme value theory)은 극단적인 값을 포함한 문제를 분석하는 학문입니다. 이러한 극단적인 값 예측을 위한 모델을 개발하기 위해서는 데이터 세트를 극단적인 세트와 일반 세트로 나누는 것이 필요합니다. 이 연구에서는 각 데이터 샘플에 대해 relevance function을 활용하여 극단적인 샘플을 분류할 수 있는 방법을 제안하고 있습니다.

- **Performance Highlights**: 결과적으로, SMOTE 기반 전략이 예측 정확도 향상에 지속적으로 기여하며, 극단적 값 예측에서 더 나은 성능을 보여주었습니다. Conv-LSTM은 주기적이고 안정적인 데이터셋에서 우수한 성능을 나타내며, BD-LSTM은 혼란스러운(non-stationary) 시퀀스에서 더 나은 성과를 보였습니다. Signal Extreme Ratio(SER)라는 tail-sensitive 지표를 통해 극단적인 사건에 대한 모델 성능을 평가하며, 이는 전반적인 예측 정확도뿐만 아니라 극단 값 예측의 정확성을 강조합니다.



### Glaucoma Detection and Structured OCT Report Generation via a Fine-tuned Multimodal Large Language Mod (https://arxiv.org/abs/2510.02403)
- **What's New**: 이번 연구에서는 안구 건강 진단을 위한 설명 가능한 다중 모달 대형 언어 모델(MM-LLM)을 개발하였습니다. 이 모델은 옵틱 신경 두부(ONH) OCT 원주 스캔의 품질을 평가하고, 녹내장 진단 및 부위별 망막 신경 섬유층(RNFL) 얇아짐 평가를 포함한 구조화된 임상 보고서를 생성하는 데 사용됩니다. 1310명의 피험자에서 수집된 데이터를 기반으로 하여 모델을 튜닝하였습니다.

- **Technical Details**: MM-LLM은 Llama 3.2 Vision-Instruct 모델을 이용하여 OCT 이미지를 통한 임상 설명을 생성하도록 세밀하게 조정되었습니다. 훈련 데이터는 쌍으로 된 OCT 이미지와 자동으로 생성된 구조화된 임상 보고서를 포함하며, 질이 낮은 스캔은 사용 불가능으로 표시되었습니다. 모델은 품질 평가, 녹내장 탐지, RNFL 얇아짐 분류의 세 가지 작업을 위해 독립적인 테스트 세트에서 평가되었습니다.

- **Performance Highlights**: 모델은 품질 평가에서 0.90의 정확도와 0.98의 특이성을 달성하였습니다. 녹내장 탐지의 경우, 정확도는 0.86으로, 민감도는 0.91, 특이도는 0.73, F1-score는 0.91에 달했습니다. RNFL 얇아짐 예측 정확도는 0.83에서 0.94까지 다양하였으며, 특히 전반적인(글로벌)과 측면(템포럴) 부위에서 가장 높은 성능을 보였습니다.



### Linear RNNs for autoregressive generation of long music samples (https://arxiv.org/abs/2510.02401)
- **What's New**: 본 연구에서는 HarmonicRNN이라는 모델을 제안하여 오디오 파형을 자율 회귀 방식으로 생성하는 문제를 해결하고자 한다. 이전의 회귀 신경망(recurrent neural networks) 접근법의 한계를 극복하기 위해, 선형 RNN(Linear RNN)을 활용하고 특히 1M 토큰까지의 긴 시퀀스를 처리할 수 있는 컨텍스트 병렬성(context-parallelism)을 도입하였다. 이 모델은 소규모 데이터셋에서 최첨단의 로그 우도(log-likelihood) 및 지각(bottom-up) 지표를 달성하였다.

- **Technical Details**: HarmonicRNN은 오디오 데이터에 최대 우도 훈련(maximum likelihood training)을 직접 적용하는 방식으로 훈련되며, 이는 미니배치(mini-batch)를 활용한 확률적 경사 상승법(stochastic gradient ascent)으로 이루어진다. 모델은 조건부 확률 분포를 계산하는 깊은 선형 RNN을 사용하며, 각 시간 단계tt에 따라 xt에 대한 범주형 분포를 출력한다. 이러한 구조는 메모리 효율성을 가지고 있으며, CG-LRU(complex gated linear recurrent unit) 아키텍처를 기반으로 한다.

- **Performance Highlights**: 실험 결과, HarmonicRNN은 로그 우도를 비롯한 다양한 지각 측정(perceptual metrics)에서 이전 모델에 비해 성능이 향상되었다. SC09, Beethoven, YouTubeMix라는 세 가지 데이터셋에서 우수한 결과를 보였으며, 특히 1분 분량의 긴 오디오 샘플에 대해서도 효과적으로 모델링할 수 있는 성능을 입증하였다. 이 모델은 TPU v4-8 및 v3-128 장치에서 훈련되었으며, 멀티 호스트 훈련을 통해 메모리 요구사항을 충족하였다.



### Hyperparameters are all you need: Using five-step inference for an original diffusion model to generate images comparable to the latest distillation mod (https://arxiv.org/abs/2510.02390)
Comments:
          10 pages, 5 figures, conference

- **What's New**: 이번 논문은 훈련 없이 고해상도 이미지 생성이 가능한 알고리즘을 제안합니다. 특히, 1024 x 1024 해상도의 이미지를 8단계에서 생성하면서 최신의 증류 모델과 유사한 FID(Frechet Inception Distance) 성능을 보입니다. 이는 이전 연구들에 비해 혁신적이며, 훈련 없는 방법으로 512 x 512 이미지를 생성할 때도 최첨단 ODE(Ordinary Differential Equation) 솔버를 초월하는 성능을 보입니다.

- **Technical Details**: 이 연구에서는 확산 확률 모델(Diffusion Probability Model)과 두 가지 프로세스인 확산(Diffusion) 및 반전(Reverse) 프로세스를 설명합니다. 반전 프로세스는 신경망을 사용하여 노이즈가 있는 이미지를 디노이즈하는 방법론을 기반으로 하며, 이를 통해 고품질 이미지를 얻습니다. 이 과정은 확산 확률 방정식(Diffusion SDE)과 반전 방정식(Reverse ODE)로 기술되어 있으며, 새로운 파라미터 설정으로 효율성을 극대화합니다.

- **Performance Highlights**: COCO 2014, COCO 2017, LAION 데이터셋을 통해 실험한 결과, 제안된 알고리즘은 FID 성능이 각각 15.7, 22.35, 17.52로 확인되었습니다. 또한, 5단계 추론에서도 19.18, 23.24, 19.61의 FID 성능을 기록하여, 최신 AMED 플러그인 솔버와의 성능이 유사함을 보였습니다. 이는 훈련 없이도 기존 알고리즘들을 초월할 수 있는 가능성을 제시합니다.



### CWM: An Open-Weights LLM for Research on Code Generation with World Models (https://arxiv.org/abs/2510.02387)
Comments:
          58 pages

- **What's New**: Code World Model (CWM)은 320억 개 매개변수를 가진 새로운 오픈 가중치(Weights) LLM으로, 코드 생성 및 세계 모델링(세계 모델을 통한 문제 해결)에 대한 연구를 촉진하기 위해 출시되었습니다. CWM은 대규모의 Python 인터프리터와 Docker 환경에서의 관찰-행동(Observation-Action) 경로를 활용하여 중간 학습이 이루어졌습니다. 이를 통해 코드 이해도를 개선하고, 다중 작업(multi-task) 사고가 필요한 소프트웨어 엔지니어링 환경에서 강력한 테스트베드를 제공합니다.

- **Technical Details**: CWM은 32B 매개변수를 가진 밀집형 디코더 전용 LLM으로, 최대 131k 토큰의 컨텍스트 크기를 지원하는 슬라이딩 윈도우 주의(attention) 기법을 사용합니다. Python 코드 실행 데이터와 에이전트 상호작용을 기반으로 한 대규모 시뮬레이션 데이터로 중간 학습하여, 코드의 구문(Syntax) 뿐만 아니라 의미(Semantics)를 이해하도록 설계되었습니다. 이 모델은 또한 강화 학습(RL)을 통해 더욱 발전할 수 있는 기초가 마련되어 있습니다.

- **Performance Highlights**: CWM은 일반적인 코딩 및 수학 작업에서 우수한 성능을 발휘하며, SWE-bench Verified에서 65.8%, LiveCodeBench에서 68.6%, Math-500에서 96.6%, AIME 2024에서 76.0%의 pass@1 점수를 기록했습니다. 이러한 성과는 CWM이 에이전트 코드 생성 및 추론(task) 작업에서도 강력한 능력을 지니고 있음을 나타냅니다. 또한, 연구자들을 위해 모델 체크포인트와 최종 가중치가 비상업적 연구 라이센스 하에 공개됩니다.



### On The Fragility of Benchmark Contamination Detection in Reasoning Models (https://arxiv.org/abs/2510.02386)
- **What's New**: 본 논문은 대형 추론 모델(Large Reasoning Models, LRMs)의 벤치마크 오염 문제에 대한 체계적인 연구를 최초로 제시합니다. 연구 결과에 따르면, LRMs는 모델 개발자가 벤치마크 데이터를 훈련 데이터에 포함시켜 성능을 부풀리는 '벤치마크 오염(contamination)'에 매우 취약하다는 사실이 밝혀졌습니다. LRMs는 체인 오브 띵킹(Chain-of-Thought, CoT) 추론 방식을 활용하지만, 이 과정에서 검출이 어렵다는 문제에 직면해 있습니다. 따라서 LRMs의 공정한 평가를 위한 신뢰할 수 있는 평가 프로토콜 개발의 필요성이 강조됩니다.

- **Technical Details**: LRMs의 벤치마크 오염 문제를 다룬 이 연구는 두 가지 주요 단계로 나뉩니다. 첫 번째 단계는 SFT(Supervised Fine-Tuning)와 RL(Reinforcement Learning) 과정 중에 발생하는 오염을 조사하며, 두 번째 단계는 CoT를 적용한 고급 LRM의 최종 SFT 단계에서의 오염을 다룹니다. 연구 결과는 K-시뮬레이션을 통해 기존의 오염 검출 방법이 LRMs의 오염을 발견하기 어려움이 드러났습니다. 특히 GRPO(Generalized Reinforcement Policy Optimization) 훈련이 오염 신호를 숨기는 주된 원인으로 지목되었습니다.

- **Performance Highlights**: 연구 결과, LRMs는 고유의 오염 문제로 인해 기존의 벤치마크 오염 검출 방법이 거의 무의미해졌습니다. CoT를 통한 오염이 발생했을 때, 대부분의 검출 방법은 무작위 추측 수준의 성능으로 떨어졌습니다. 이는 LRMs가 벤치마크 데이터를 기억해야만 검출된다는 기존 가정이 틀렸음을 나타내며, LRMs의 신뢰성과 공정성을 위협합니다. 마지막으로, 이 연구는 LRMs 평가의 무결성을 보장하기 위해 고급 오염 검출 방법의 개발 필요성을 강조합니다.



### Scaling Homomorphic Applications in Deploymen (https://arxiv.org/abs/2510.02376)
Comments:
          5 pages, 6 figures, 1 pseudo code

- **What's New**: 이번 연구에서는 동형 암호화(fully homomorphic encryption, FHE) 기반의 애플리케이션을 개발하여 암호화 생태계의 생산 준비 상태를 판단하기 위한 개념 증명이 이루어졌습니다. 이를 위해 영화 추천 애플리케이션을 구현하였고, 컨테이너화(containerization) 및 오케스트레이션(orchestration)을 통해 프로덕션 환경으로 배포되었습니다. 배포 설정을 조정함으로써 FHE의 계산 한계를 추가 인프라 최적화를 통해 완화시켰습니다.

- **Technical Details**: 이 연구에서는 FHE를 활용한 영화 추천 시스템을 구축하여 실제 운영 가능성을 시연하고 있습니다. Flask 애플리케이션으로 개발되었으며, MovieLens 데이터셋을 활용하여 naive logistic regression 모델을 통해 추천 기능을 수행합니다. 모델 학습 후, FHE 회로로 변환하고, 각 단계를 통해 데이터의 양자화 및 그래프 간소화 작업을 수행하여 실제 인프라에서 활용 가능성을 검증합니다.

- **Performance Highlights**: 재배포 과정에서 경량화된 Kubernetes 버전이 사용되었으며, RL(강화 학습) 에이전트가 오케스트레이터의 구성을 최적화하여 운영 요구를 충족하기 위해 복제본 수를 조정하게 됩니다. 이 연구는 유의미한 성능 최적화 및 효율성을 입증하며, 사용자 인터페이스와의 상호작용을 통해 고객에게 안전한 추천 시스템을 제공합니다.



### Pretraining with hierarchical memories: separating long-tail and common knowledg (https://arxiv.org/abs/2510.02375)
- **What's New**: 이 연구는 현대 언어 모델의 성능 향상에 대한 새로운 접근법을 제시합니다. 기존의 방법이 매개변수(parameter)의 규모 확대에 의존하고 있는 반면, 우리는 작은 언어 모델이 큰 계층적(parametric) 메모리 뱅크를 활용하여 폭넓은 세계 지식을 접근할 수 있도록 하는 메모리 증강 아키텍처를 개발하였습니다. 이를 통해 필요한 부분만을 메모리에서 가져와 모델에 추가합니다.

- **Technical Details**: 제안된 메모리 증강 아키텍처는 문맥에 따라 필요한 지식 매개변수를 모델에 연결하여, 실행시간 효율성을 높입니다. 또한, 훈련 시 메모리 파라미터가 유사한 주제의 시퀀스에서만 활성화되고 업데이트됨으로써, 파라미터의 망각(catastrophic forgetting)을 줄이고 보다 효과적으로 장기 지식을 기억할 수 있습니다. 이러한 혁신적인 접근방식은 훈련 효율성을 높이고, 분산 훈련을 더욱 간편하게 합니다.

- **Performance Highlights**: 우리는 실험을 통해 제안된 모델이 기본 모델보다 성능 향상을 보여주며, 매개변수 수가 2배 이상인 일반 모델과 유사한 성능을 낸다는 것을 입증했습니다. 예를 들어, 4.6B 메모리 뱅크에서 가져온 18M 매개변수의 메모리가 보강된 160M 모델이 정규 모델과 유사한 성능을 발휘합니다. 이러한 성능 개선은 메모리 유형, 깊이, 크기 그리고 메모리와 모델의 비율 등에 따라 달라지며, 우리는 각 요소가 성능에 미치는 영향을 체계적으로 분석하였습니다.



### A Hybrid CAPTCHA Combining Generative AI with Keystroke Dynamics for Enhanced Bot Detection (https://arxiv.org/abs/2510.02374)
Comments:
          6 pages, 4 figures

- **What's New**: 이번 논문에서는 전통적인 CAPTCHA 시스템의 사용성과 AI 기반 봇에 대한 회복력 간의 균형 문제를 해결하기 위한 새로운 하이브리드 CAPTCHA 시스템을 소개합니다. 이 시스템은 대규모 언어 모델(LLM)에서 생성된 인지적 도전 과제를 행동 생체 인식 분석(keystroke dynamics)과 결합하여 동적이며 예측할 수 없는 질문을 생성합니다. 연구 결과, 제안된 시스템은 봇 탐지에서 높은 정확도를 보이며, 인간 사용자에게도 높은 사용 성능을 유지합니다.

- **Technical Details**: 제안하는 시스템은 클라이언트-서버 아키텍처를 기반으로 하여 생성된 질문과 사용자 입력을 처리하며, 키스트로크 데이터를 통한 행동 분석을 수행합니다. LLM에서 동적으로 생성된 질문을 사용하여, 사용자가 제공한 답변의 SHA-256 해시 값을 확인하고, 키 입력의 측정 데이터를 수집하여 인간과 봇을 구분합니다. 핵심 피쳐인 평균 대기시간과 대기시간의 표준 편차를 활용하여 입력이 인간인지 로봇인지를 판별합니다.

- **Performance Highlights**: 실험 결과, 인간 참가자는 첫 시도에서 87%의 성공률을 기록하였고, 두 번째 시도에서는 100% 이내에 성공했습니다. 봇 그룹의 경우, 시스템은 100%의 탐지율을 기록하며, 붙여넣기 봇 및 타이프 시뮬레이션 봇 모두 차단되었습니다. 이러한 결과는 제안된 이중 레이어 접근 방식이 효과적이고 사용자 친화적인 CAPTCHA 솔루션이 될 수 있음을 보여줍니다.



### A-MemGuard: A Proactive Defense Framework for LLM-Based Agent Memory (https://arxiv.org/abs/2510.02373)
- **What's New**: 이 논문에서는 A-MemGuard라는 새로운 프레임워크를 소개합니다. A-MemGuard는 LLM 에이전트의 메모리 보안을 강화하기 위한 첫 번째 능동적 방어 프레임워크로, 메모리 자체가 자기 점검 및 자기 수정 기능을 갖추도록 설계되었습니다. 이전의 방식과 달리, A-MemGuard는 에이전트의 기본 아키텍처를 수정하지 않고도 두 가지 기법을 결합하여 메모리 공격에 대응합니다.

- **Technical Details**: A-MemGuard의 핵심 아이디어는 컨센서스 기반 검증(consensus-based validation)과 이중 메모리 구조(dual-memory structure)를 통한 접근입니다. 이를 통해 A-MemGuard는 여러 관련 기억을 비교하며 이상 징후를 발견하고, 발견된 오류를 '교훈'으로 변환하여 에이전트가 스스로의 실수로부터 학습할 수 있도록 합니다. 이러한 방법은 메모리 항목의 단독 감사(audit)로는 감지할 수 없는 맥락 의존적 공격(context-dependent attack)을 효과적으로 탐지할 수 있게 합니다.

- **Performance Highlights**: A-MemGuard는 다양한 벤치마크에 대한 포괄적인 평가를 통해 95% 이상의 공격 성공률(attacker success rate)을 감소시키는 효과를 보여주었습니다. 실험 결과, 간접 공격에 의한 자기 강화 오류 사이클을 효과적으로 차단하며, 최소한의 성능 비용으로 높은 정확도를 유지했습니다. 또한, A-MemGuard는 다중 에이전트 시스템에서 탁월한 일반화 능력을 보여주었으며, 최고 성공률을 기록했습니다.



### Federated Spatiotemporal Graph Learning for Passive Attack Detection in Smart Grids (https://arxiv.org/abs/2510.02371)
- **What's New**: 본 연구는 스마트 그리드 환경에서 은밀한 패시브 공격을 탐지하기 위한 그래프 중심의 다중 모달 연합 학습 프레임워크를 제안합니다. 기존의 공격 탐지 모델이 증폭된 공격에 집중했을 때, 이 연구는 패시브 공격을 사전 예방적으로 감지하기 위해 공간적 및 시간적 행동을 동시에 모델링합니다. 그래프 컨볼루션 네트워크(GCN)와 양방향 GRU의 결합을 통해 유용한 보안 정보의 융합을 실현했습니다.

- **Technical Details**: 그래프 중심의 다중 모달 탐지기는 물리적 신호와 행동 지표를 융합하여 고유한 시공간 표현을 형성합니다. 두 단계의 인코더가 설계되어 지역적 맥락을 집계하고, 단기 시간 의존성을 모델링하여 패시브 공격을 탐지합니다. 이 과정에서 연합 학습(Federated Learning) 환경에서 학습이 이루어지며, 원시 데이터는 각 노드의 로컬에 남아 개인정보 보호를 보장합니다.

- **Performance Highlights**: 모델은 0.15%의 허위 양성률(FPR) 하에 시퀀스 당 93.35%와 시간당 98.32%의 높은 테스트 정확도를 달성했습니다. 결과는 공간적 및 시간적 맥락의 결합이 은밀한 정찰 활동을 신뢰성 있게 탐지할 수 있도록 하며, 낮은 허위 양성률도 유지된다는 것을 보여줍니다. 이는 비수정적 연합 스마트 그리드 배포에 적합한 접근법이 됩니다.



### Training Dynamics of Parametric and In-Context Knowledge Utilization in Language Models (https://arxiv.org/abs/2510.02370)
Comments:
          16 pages

- **What's New**: 이 논문에서는 대형 언어 모델들이 훈련 중에 지도 학습(knowledge arbitration) 방식을 어떻게 형성하는지에 대한 체계적인 이해 부족 문제를 다룹니다. 다양한 훈련 조건이 모델들이 문맥 내(in-context) 지식과 매개변수(parametric) 지식之间의 경합(competition)에 미치는 영향을 연구했습니다. 이 연구는 사전 훈련 이후의 계산 자원의 낭비를 예방하는 데 중요한 역할을 할 수 있습니다.

- **Technical Details**: 연구자들은 변환기(transformer) 기반 언어 모델을 합성 전기(biographies corpus) 데이터셋을 사용하여 훈련하고, 다양한 조건을 체계적으로 조절하여 실험을 진행했습니다. 결과적으로, 문서 내부( intra-document )의 사실 반복(repetition)이 매개변수 지식과 문맥 내 지식의 개발을 촉진함을 발견했습니다. 또한, 일관되지 않은 정보(inconsistent information)나 분포 왜곡(distributional skew)을 포함하는 데이터셋에서 훈련하는 것이 모델들이 강력한 전략을 개발하도록 유도함을 확인했습니다.

- **Performance Highlights**: 훈련 조건 개선으로 비이상적인 속성(non-ideal properties)을 제거하는 대신에, 이러한 속성이 강력한 지도 학습을 위한 중요성에 대한 증거를 제공합니다. 이 연구는 매개변수 지식과 문맥 지식의 통합(integration)에 있어 실질적이고 경험적인 지침을 제공하며, 이를 통해 더 효과적인 사전 훈련(pretraining) 모델 설계가 가능함을 보여줍니다.



### Beyond Manuals and Tasks: Instance-Level Context Learning for LLM Agents (https://arxiv.org/abs/2510.02369)
Comments:
          Under review at ICLR 2026

- **What's New**: 이 논문에서는 기존의 Large Language Model (LLM) 에이전트가 수신하는 세 가지 컨텍스트 중 인스턴스 수준의 컨텍스트(Instance-Level Context)라는 중요하지만 간과된 제 3의 유형을 제시합니다. 이는 특정 환경 인스턴스에 연결된 검증 가능하고 재사용 가능한 사실들로 구성되어 있습니다. 이러한 인스턴스 수준의 컨텍스트가 결여될 경우 LLM 에이전트가 복잡한 작업을 수행하는 데 실패하는 일반적인 원인이라는 주장을 하고 있습니다.

- **Technical Details**: 저자들은 '인스턴스 수준 컨텍스트 학습(Instance-Level Context Learning, ILCL)'이라는 문제를 공식화하고, 이를 해결하기 위한 작업 비 특이적(task-agnostic) 방법인 AutoContext를 소개합니다. 이 방법은 TODO 숲(TODO forest)을 사용하여 다음 작업을 우선 지정하고, 경량의 계획-행동-추출(Plan-Act-Extract) 루프를 통해 고정밀 컨텍스트 문서를 자동으로 생성합니다. AutoContext의 구조는 탐색을 조직화하고 시스템적으로 지식 격차를 노출하는 혁신적인 방법을 포함하고 있습니다.

- **Performance Highlights**: 문서화된 결과는 TextWorld, ALFWorld, Crafter에서의 실험을 통해 입증되었으며, ReAct 에이전트의 TextWorld 성공률이 37%에서 95%로 향상되었고, IGE는 81%에서 95%로 개선되었습니다. 이 방법은 일회성 탐색을 지속 가능하고 재사용 가능한 지식으로 변환함으로써 기존의 컨텍스트를 보완하여 더욱 신뢰할 수 있는 LLM 에이전트를 가능하게 합니다. 따라서, AutoContext는 여러 작업에 걸쳐 다운스트림 에이전트의 성능을 크게 개선합니다.



### A Cross-Lingual Analysis of Bias in Large Language Models Using Romanian History (https://arxiv.org/abs/2510.02362)
Comments:
          10 pages

- **What's New**: 이번 연구는 루마니아의 역사적 논란에 대한 질문을 여러 개의 대형 언어 모델(LLMs)에 제시하여 그 모델들의 편향성을 분석하는 내용을 담고 있습니다. 이러한 접근은 교육적 목적뿐만 아니라, 역사라는 주제가 문화와 국가의 이상에 의해 어떻게 왜곡될 수 있는지를 인식하는 데 그 의의가 있습니다. 연구 결과, 다양한 언어와 문맥에서 LLM의 답변이 어떻게 변화하는지를 발견했습니다.

- **Technical Details**: 연구 방법론은 세 가지 주요 단계로 나뉘어 있으며, 이는 특정 분야의 논란이 되는 역사적 사건에 대한 편향 분석을 보장하기 위해 고안되었습니다. 첫 단계에서는 루마니아어를 기본 언어로 선정하고, 영어, 헝가리어, 러시아어를 포함하여 서로 다른 문화적 관점에서 편향을 탐구했습니다. 레벨에 따라 단순한 긍정 혹은 부정 응답, 1-10의 척도를 통한 수치 답변, 구조화된 에세이 작성으로 분석을 진행했습니다.

- **Performance Highlights**: 실험 결과는 언어 모델들이 특정 질문에 대해 일관된 대답을 제공하는 경향이 있지만, 각기 다른 언어 및 형식에 따라 의견이 달라질 수 있음을 보여주었습니다. 특히 이진 답변의 안정성은 상대적으로 높지만 완벽하지 않았고, 수치적 평가에서는 초기의 이진 선택과 일치하지 않는 경우가 많았습니다. 이번 연구는 LLMs가 정보의 제공 방식에 따라 어떻게 반응이 달라질 수 있는지를 명확히 하였습니다.



### ChunkLLM: A Lightweight Pluggable Framework for Accelerating LLMs Inferenc (https://arxiv.org/abs/2510.02361)
- **What's New**: ChunkLLM은 경량화된 모듈인 QK Adapter와 Chunk Adapter를 도입하여 기존의 Transformers에 통합할 수 있는 새로운 훈련 프레임워크를 제안합니다. 이 구조는 각 Transformer 레이어에 연결되어 기능 압축과 청크 주의 획득을 동시에 수행하여 계산 비용을 크게 절감할 수 있습니다. 이러한 접근 방식은 의미의 완전성과 훈련-추론 효율성을 동시에 해결하는 데 중점을 둡니다.

- **Technical Details**: Chunk Adapter는 청크 경계를 식별하기 위해 맥락적 의미 정보를 활용하며, 1층 포워드 신경망 분류기로 구현됩니다. QK Adapter는 각 Transformer 레이어에 병렬로 위치하여, 주의 행렬을 청크 주의 점수로 매핑합니다. 훈련 과정에서는 KL 발산을 통해 청크 주의 점수를 최적화하여, 주요 청크의 회수율을 향상시킵니다.

- **Performance Highlights**: ChunkLLM은 다양한 장기 및 단기 텍스트 벤치마크 데이터 세트에서 실험 평가를 통해 120,000 토큰 처리에서 바닐라 Transformer 대비 4.48배의 속도 향상을 달성했습니다. 특히 단기 텍스트 벤치마크에서 유사한 성능을 유지하며, 장기 맥락 벤치마크에서도 98.64%의 성능을 보유하고 있습니다. 이는 기존의 Transformer 모델들에 비해 더 효율적인 컴퓨팅 자원 관리와 성능 최적화를 가능하게 합니다.



### Spiral of Silence in Large Language Model Agents (https://arxiv.org/abs/2510.02360)
- **What's New**: 이 논문은 Spiral of Silence (SoS) 이론을 인공지능 언어 모델(LLM)에 적용하여 소수 의견의 억제 현상이 LLM 집단에서도 발생할 수 있는지 검토합니다. 저자들은 History와 Persona 신호를 도입하여 SoS 역학을 탐색하고, 이 신호들이 소수 집단의 의견이 사라지는 과정을 어떻게 형성하는지 평가하는 체계적인 프레임워크를 제안합니다.

- **Technical Details**: 연구에서는 LLM 에이전트가 영화에 대한 평가를 순차적으로 수행하는 다중 에이전트 환경을 구축하였습니다. 두 가지 주요 신호인 History(이전 에이전트의 평균 평점)와 Persona(각 에이전트에 할당된 고유한 역할)를 활용하여 의견 역학을 측정합니다. 이를 통해 SoS 역학의 발생 여부을 확인하는 다양한 통계 기법과 지표를 사용하여 실험 결과를 평가합니다.

- **Performance Highlights**: 실험 결과, History와 Persona 신호 모두가 존재할 때 SoS 패턴이 가장 뚜렷하게 나타났으며, History 신호만으로도 강한 고정 효과가 나타났습니다. Persona 신호는 다양한 의견을 유도하지만 이들 간의 상관관계는 없음을 보여주었고, 이러한 발견은 LLM 시스템의 설계와 규제에 대한 중대한 시사점을 제공한다는 점에서 의미가 있습니다.



### Emission-GPT: A domain-specific language model agent for knowledge retrieval, emission inventory and data analysis (https://arxiv.org/abs/2510.02359)
- **What's New**: 이번 논문에서는 대기 오염물질 및 온실가스 배출에 대한 보다 정확한 이해와 분석을 제공하기 위해 Emission-GPT라는 지식 강화 대형 언어 모델 에이전트를 소개합니다. Emission-GPT는 10,000개 이상의 문서로 구성된 맞춤형 지식 기반 위에 구축되었으며, 도메인 특화된 질문 답변을 지원하는 데 필요한 기능을 통합하고 있습니다. 이를 통해 사용자들은 자연어를 이용하여 배출 데이터를 대화형으로 분석하고 시각화할 수 있는 새로운 가능성을 열었습니다.

- **Technical Details**: Emission-GPT 시스템은 대기배출 분야에서 지능형 상호작용 및 분석을 가능하게 하는 모듈화 및 다단계 워크플로우를 제안합니다. 이 시스템은 사용자 쿼리를 기초로 질문을 두 가지 카테고리로 분류하며, 각 카테고리에 맞춰 적절한 전문가 LLM을 호출하여 응답 및 데이터 분석을 수행합니다. 이렇게 구성된 아키텍처는 각기 다른 배출 관련 작업을 위한 맞춤형 AI 에이전트를 지원할 수 있도록 설계되었습니다.

- **Performance Highlights**: Emission-GPT의 성능을 평가하는 사례 연구에서는 간단한 프롬프트를 통해 원시 데이터에서 중요한 인사이트를 직접 추출할 수 있음을 보여주었습니다. 이 시스템은 배출 재고 조사의 효율성을 크게 높이며, 사용자 정의 시나리오에 대한 배출 인자 추천 기능 등을 통해 대기배출 정보의 접근성 및 활용도를 크게 향상시킵니다. 비전문가도 쉽게 사용할 수 형식으로 구성된 Emission-GPT는 향후 배출 재고 개발 및 시나리오 기반 평가의 기초 도구로 자리잡을 것으로 기대됩니다.



### DiffuSpec: Unlocking Diffusion Language Models for Speculative Decoding (https://arxiv.org/abs/2510.02358)
- **What's New**: 대형 언어 모델(LLMs)이 성장함에 따라 정확도는 향상되지만, 自回归(autoregressive, AR) 디코딩의 특성 때문에 지연(latency)의 병목이 발생합니다. 이 문제를 해결하기 위해 DiffuSpec라는 새로운 프레임워크를 소개합니다. DiffuSpec는 사전 훈련된 확산 언어 모델(diffusion language model, DLM)을 사용하여 단일 전방 패스를 통해 여러 토큰 초안을 생성합니다.

- **Technical Details**: DiffuSpec는 두 가지 주요 구성요소로 구성됩니다. 첫째, 인과 일관성 경로 검색(causal-consistency path search, CPS)을 통해 AR 검증에 맞는 왼쪽에서 오른쪽 경로를 선택하여 수용을 극대화합니다. 둘째, 적응형 초안 길이(adaptive draft-length, ADL) 컨트롤러를 통해 최근 수용 피드백을 기반으로 다음 초안 길이를 조정합니다. 이 프레임워크는 별도의 훈련 없이 기존 AR 검증기와 통합할 수 있습니다.

- **Performance Highlights**: DiffuSpec는 다양한 생성 작업에서 최대 3배의 벽시계 속도를 향상시켜 훈련 기반 방법에 근접하는 성능을 보여줍니다. 또한, DiffuSpec는 기존 훈련 프리 베이스라인을 초 outperforming하며, 확산 기반 초안 작성을 AR 드래프트의 강력한 대안으로 자리매김하고 있습니다.



### Privacy in the Age of AI: A Taxonomy of Data Risks (https://arxiv.org/abs/2510.02357)
Comments:
          12 pages, 2 figures, 4 tables

- **What's New**: 이 논문은 기존의 개인정보 보호 프레임워크가 인공지능(AI) 기술의 고유한 특성 때문에 부족하다는 점을 강조합니다. 45개의 연구를 체계적으로 검토하여 AI 개인 정보 보호 위험을 분류하는 새로운 분류 체계를 제시합니다. 이를 통해 데이터셋 수준, 모델 수준, 인프라 수준, 내부 위협 위험이라는 네 가지 범주에 걸쳐 19개 주요 위험을 도출하였습니다.

- **Technical Details**: 이 연구는 시스템 전체의 기술적 및 행동적 차원을 아우르며, AI 개인 정보 보호의 고찰을 심화시키고자 합니다. 주요 위험 요소들은 데이터셋, 모델, 인프라 구조, 그리고 내부 위협으로 나뉘며, 각 요소의 비중도 균형을 이루고 있습니다. 특히 인간의 실수(human error)가 가장 큰 위험 요소로 9.45%를 차지하는 것으로 나타났습니다.

- **Performance Highlights**: 논문은 전통적인 보안 접근 방식이 기술적 통제보다 인간 요인을 우선시하지 않는 문제를 지적합니다. 이는 AI 개발에 있어 신뢰성을 높이기 위한 기초를 제공하며, 향후 연구의 방향성을 제시합니다. 이 분류 체계는 AI 개인 정보 보호 문제를 이해하고 해결할 수 있는 강력한 도구로 작용할 것으로 기대됩니다.



### Measuring Physical-World Privacy Awareness of Large Language Models: An Evaluation Benchmark (https://arxiv.org/abs/2510.02356)
- **What's New**: 본 논문에서 제안하는 EAPrivacy는 대형 언어 모델(LLMs) 기반 에이전트의 물리적 세계에서의 개인 정보 보호 인식을 체계적으로 평가하기 위한 새로운 기준입니다. 기존 평가 방식은 자연어 기반의 상황에 제한되어 있었던 반면, EAPrivacy는 감도 높은 물체 처리, 변화하는 환경에 대한 적응, 작업 수행과 개인 정보 보호의 균형을 맞추는 능력을 평가하는 다양한 시나리오를 포함합니다. 특히, 이 평가에서 현재 모델이 물리적 환경에서의 프라이버시 요구를 다루는데 있어 심각한 부족함을 드러내는 결과를 보여주었습니다.

- **Technical Details**: EAPrivacy는 네 가지 주요 수준으로 구성되어 있으며, 각각은 물리적 공간 내에서의 사물 인식, 변화하는 환경에서의 행동 평가, 수동적 프라이버시와 작업 충돌 해결, 사회적 규범과 개인 정보 간의 갈등 탐색을 다룹니다. 각 단계는 서로 다른 유형의 프라이버시 인식을 요구하며, 총 400개 이상의 절차적으로 생성된 시나리오를 통해 에이전트의 성능을 평가합니다. 이를 통해 물리적 세계의 복잡한 사회적 및 개인 정보 환경을 탐색하는 데 있어 LLM 모델의 한계를 강조합니다.

- **Performance Highlights**: 우리의 평가 결과, 현재 모델들은 개인 정보 보호 요구 사항을 일관되게 처리하지 못하고 있으며, 최고 성능을 보이는 모델조차도 변화하는 환경에서 59%의 정확도만을 기록했습니다. 또한, 작업 수행과 개인 정보 보호 요청이 동반된 경우에는 최대 86%의 경우에 작업을 완료하는 것에 우선 순위를 두었습니다. 이러한 결과들은 LLM들이 물리적으로 기반한 개인 정보 보호에 대한 fundamental (근본적인) 불일치를 나타내며, 보다 robust하고 물리적 인식을 갖춘 alignment가 필요함을 강조합니다.



### Evaluating Bias in Spoken Dialogue LLMs for Real-World Decisions and Recommendations (https://arxiv.org/abs/2510.02352)
- **What's New**: 이번 논문은 음성 대화 모델(SDMs)에서의 편향(bias)에 대한 체계적인 평가를 제공하며, 다중 턴 대화(multi-turn dialogues)가 모델 출력에서 어떻게 편향을 증폭할 수 있는지를 연구했습니다. 이는 개방형 모델(Qwen2.5-Omni, GLM-4-Voice)과 폐쇄형 API(GPT-4o Audio, Gemini-2.5-Flash)를 포함하여 처음으로 발표된 연구로, 음성 기반의 대화형 시스템에서 공정성과 신뢰성을 제고하는 데 기여할 것 입니다.

- **Technical Details**: 다중 턴 대화에 따른 편향 검증을 위해 그룹 불공정성 점수(Group Unfairness Score: GUS)와 유사도 기반 정규화 통계 비율(Similarity-Based Normalized Statistics Rate: SNSR) 메트릭을 사용했습니다. 연구에서 생성된 데이터셋은 결정을 내리는 작업 및 추천 작업과 같은 두 가지 주요 실제 시나리오에 초점을 맞추며, 이는 모델의 출력이 사회적 기회를 어떻게 직접적으로 영향을 미칠 수 있는지를 보여줍니다. 이 연구에서는 또한 한국어의 음성과 텍스트 생성 플랫폼을 활용하여 통제된 음성을 합성했습니다.

- **Performance Highlights**: 결과적으로 폐쇄형 모델이 일반적으로 낮은 편향을 보이는 반면, 개방형 모델은 나이와 성별에 대해 더 민감하게 반응함을 발견했습니다. 특히, 추천 작업에서는 집단 간 격차가 확대되는 경향이 있습니다. 다중 턴 대화에서도 편향된 결정이 지속될 수 있으며, 일부 집단은 공정한 결과를 달성하기 위해 더 많은 교정 피드백이 필요하다는 점을 확인했습니다.



### Language, Culture, and Ideology: Personalizing Offensiveness Detection in Political Tweets with Reasoning LLMs (https://arxiv.org/abs/2510.02351)
Comments:
          To appear in the Proceedings of the IEEE International Conference on Data Mining Workshops (ICDMW)

- **What's New**: 이 연구에서는 대형 언어 모델(LLMs)이 정치적 담론에서 공격성을 평가하는 방법을 탐구합니다. 특히, 2020년 미국 대선 트윗을 중심으로 다국어 MD-Agreement 데이터셋을 활용하여 다양한 정치적 관점을 가진 모델들이 트윗을 공격적 혹은 비공격적으로 판단하는 능력을 평가했습니다. 연구에서는 DeepSeek-R1, o4-mini, GPT-4.1-mini, Qwen3 등 여러 최신 모델들을 비교했습니다.

- **Technical Details**: 이 연구는 이념적이고 문화적인 맥락에서 공격성 탐지의 개인화 여부를 평가하는 새로운 프레임워크를 제안합니다. 연구자들은 크게 두 가지 변수 – 모델 크기(소형 vs 대형)와 추론 능력(가능 vs 불가능)을 기준으로 모델을 카테고리화하여 평가하였습니다. 추론 능력을 갖춘 대형 모델들이 이념적 관점을 더 잘 모방할 수 있음을 발견하였습니다.

- **Performance Highlights**: 실험 결과, DeepSeek-R1과 o4-mini와 같이 추론 기능이 있는 모델이 개인화된 공격성 분류에서 뛰어난 성능을 보였습니다. 원래 언어에서 폴란드어 및 러시아어로 번역된 트윗을 사용하였고, 각 모델의 판단이 정치적 배경에 따라 달라질 수 있음을 확인했습니다. 이는 LLM을 더 정교한 사회 정치적 텍스트 분류에 적합하도록 조정하기 위한 중요한 메커니즘이 될 수 있습니다.



### LLMSQL: Upgrading WikiSQL for the LLM Era of Text-to-SQL (https://arxiv.org/abs/2510.02350)
Comments:
          To appear in the Proceedings of the IEEE International Conference on Data Mining Workshops (ICDMW)

- **What's New**: 이 논문에서는 자연어 질문을 SQL 쿼리로 변환하는 새로운 데이터셋인 LLMSQL을 제안합니다. LLMSQL은 기존의 WikiSQL 데이터셋을 체계적으로 개정하고 변형하여 LLM(대형 언어 모델) 시대에 맞게 발전시킨 것입니다. 이 데이터셋은 고르지 못한 주의사항 및 구조적 문제를 해결하여, 안정적인 쿼리 실행과 올바른 결과를 보장합니다.

- **Technical Details**: LLMSQL은 자연어 질문과 SQL 쿼리쌍을 정제하는 자동화된 방법론을 통해 생성되었습니다. WikiSQL에서 발생하는 데이터 형식 불일치, 대소문자 민감도 문제, 질문 없음 등의 오류를 클래스화하였습니다. 다양한 대형 언어 모델(Gemma 3, LLaMA 3.2, Mistral 7B 등)을 평가하여 LLMSQL의 성능을 검증하였습니다.

- **Performance Highlights**: LLMSQL은 단일 테이블에 기반한 대규모 벤치마크로서, 기존 WikiSQL의 문제점을 해결하며 구조화된 쿼리 생성 작업을 위한 실용성을 높였습니다. 실험을 통해 기존 WikiSQL 분할 및 LLMSQL 벤치마크에서의 여러 모델 성능을 비교하였고, LLMSQL이 대형 언어 모델(LLM)의 평가에 적합하다는 것을 입증하였습니다.



### An Investigation into the Performance of Non-Contrastive Self-Supervised Learning Methods for Network Intrusion Detection (https://arxiv.org/abs/2510.02349)
- **What's New**: 이 논문에서는 네트워크 침입 탐지에 대해 비대조 자기 지도 학습(self-supervised learning, SSL) 방법을 조사합니다. 기존의 지도 학습(supervised learning) 방법은 알려진 이상 탐지에 한정된 반면, SSL은 라벨이 없는 데이터에서도 의미 있는 표현을 학습할 수 있습니다. 저자들은 대조가 아닌 여러 SSL 기술을 통해 두 가지 네트워크 침입 탐지 데이터셋에서 최적의 성능을 비교하였습니다.

- **Technical Details**: 저자들은 SSL의 효과를 극대화하기 위해 다섯 가지 비대조 SSL 모델과 세 가지 인코더 아키텍처, 여섯 가지 증강 전략을 결합하여 실험을 진행했습니다. 각 모델의 최상의 성능 조합을 결정하기 위해 정밀도(precision), 재현율(recall), F1 점수, AUC-ROC와 같은 평가 지표를 사용하였습니다. 연구는 90개의 실험으로 체계적으로 구성되어 있으며, 이는 서로 다른 조합을 탐구하는 방식으로 이루어졌습니다.

- **Performance Highlights**: 결과적으로 비대조 SSL 방법들이 두 개의 대표적인 비지도 학습 기준선 모델인 DeepSVDD와 Autoencoder와 비교될 때, 뛰어난 공격 탐지 성능을 보였습니다. 특히, 각 SSL 모델에서 최적의 증강 방법과 인코더 설계를 찾는 것이 매우 중요한 결과를 도출하는 데 기여했습니다. 이 논문은 NID 분야에서 비대조 SSL 모델의 비교 분석을 최초로 시도함으로써 기존 연구와의 차별점을 가지고 있습니다.



### mini-vec2vec: Scaling Universal Geometry Alignment with Linear Transformations (https://arxiv.org/abs/2510.02348)
- **What's New**: 본 논문은 vec2vec를 기반으로 하여, 교차 데이터 없이 텍스트 임베딩 공간을 정렬하도록 설계된 프로세스입니다. 특정한 매칭이 없어도 높은 안정성과 효율성을 자랑하는 mini-vec2vec를 소개하며, 기존 방법보다 연산 비용이 훨씬 낮은 선형 변환을 학습합니다. 이를 통해 새로운 분야에서의 활용 가능성을 제시합니다.

- **Technical Details**: mini-vec2vec는 세 가지 주요 단계로 구성됩니다: 가상 병렬 임베딩 벡터 간의 일치 시도, 변환 적합성, 그리고 반복적인 정제입니다. 상대적 표현 프레임워크를 활용하고, 간단한 아핀 변환을 통해 구조적 유사성을 검색하여 정렬을 가능하게 합니다. 이 방법은 CPU 하드웨어에서 실행될 수 있도록 설계되어 자원 소모를 최소화합니다.

- **Performance Highlights**: 논문에서는 comprehensive experiments를 통해 mini-vec2vec가 기존의 adversarial 방법인 vec2vec의 성능을 능가한다고 보고합니다. 특히, 연산 자원 소모가 대폭 줄어들고, 훈련의 안정성 및 해석 가능성을 제공하여 다양한 데이터 분포에서도 잘 작동함을 강조합니다. mini-vec2vec는 수 동록 훈련 데이터에서 효과적으로 작동하고, 1:1 매칭이 불가능한 경우에도 견고한 결과를 보여줍니다.



### Small Language Models for Curriculum-based Guidanc (https://arxiv.org/abs/2510.02347)
- **What's New**: 이번 연구에서는 오픈소스 소형 언어 모델(SLM)을 활용한 AI 교육 보조 도우미의 개발 및 평가를 다룹니다. 연구에서는 8개 SLM 모델을 대상으로 교육과정에 기반한 가이드를 제공하는 Retrieval-Augmented Generation (RAG) 파이프라인을 적용하였습니다. 특히, SLM이 올바른 프롬프트(prompts)와 목표 지향적 검색을 통해 대규모 언어 모델(LLM)과 동등한 정확도를 보여준다는 점이 강조됩니다.

- **Technical Details**: 연구는 스칸디나비아의 대학에서 제공되는 대수학 관련 자료를 기반으로 하여 RAG 파이프라인을 통해 AI 교육 보조 도우미를 개발하였습니다. 이 시스템은 학생의 질문에 대해 적절한 교과 내용을 색인화하여 관련된 정보를 동적으로 검색하고 응답으로 통합하는 방식으로 운영됩니다. SLM은 각각 17억 개 이하의 매개변수를 사용하여 소비자 수준의 하드웨어에서도 실시간으로 사용 가능한 점이 특징입니다.

- **Performance Highlights**: SLM 기반의 AI 교육 도우미는 교육과정 일관성 유지, 오류 정보 감소, 비용 효율성 증대 등 여러 이점을 제공합니다. 연구에서는 SLM이 상용 LLM에 비해 보다 투명하고 정확한 교육 자료에 기반한 응답을 제공할 수 있다는 점을 입증하였습니다. 이를 통해 AI 교육 도우미가 교육기관에서 지속 가능한 개인화 학습을 구현하는 데 적합하다는 것을 강조합니다.



### Breaking the MoE LLM Trilemma: Dynamic Expert Clustering with Structured Compression (https://arxiv.org/abs/2510.02345)
Comments:
          12 pages, 2 figures, 3 tables. Under review as a conference paper at ICLR 2026

- **What's New**: 이번 연구는 Mixture-of-Experts (MoE) 대형 언어 모델에서 발생하는 부하 불균형, 매개변수 중복 및 통신 오버헤드를 동시에 해결하기 위한 통합 프레임워크를 소개합니다. 이 프레임워크는 긴밀하게 연결된 네 가지 혁신 요소를 통합하여 동적 전문가 클러스터링과 구조적 압축을 통해 MoE 모델의 효율성을 일관되게 개선합니다. 특히, 학습 중 모델 아키텍처를 동적으로 재구성할 수 있는 세맨틱(semantic) 임베딩 기능을 활용하여 효율적으로 전문가를 재편성합니다.

- **Technical Details**: 제안된 방법은 온라인 클러스터링 프로세스를 사용하여 매개변수와 활성화 유사도를 결합한 메트릭으로 전문가를 주기적으로 재조정합니다. 각 클러스터 내에서는 전문가 가중치를 공유 기본 행렬과 매우 낮은 순위의 잔여 적응기로 분해하여 매개변수를 최대 다섯 배 줄이는 동시에 전문성을 유지합니다. 이 구조는 두 단계 계층 라우팅 전략을 가능하게 하여 토큰이 먼저 클러스터에 할당된 후 구체적인 전문가에게 전달되도록 합니다.

- **Performance Highlights**: GLUE 및 WikiText-103에서 평가한 결과, 제안된 프레임워크는 표준 MoE 모델과 동등한 품질을 유지하면서 전체 매개변수를 약 80% 줄이고, 처리량을 10%에서 20% 증가시키며, 전문가 부하 분산을 세 배 이상 감소시켰습니다. 이러한 결과는 구조적 재편성이 확장 가능하고 효율적이며 메모리 효과적인 MoE LLM을 위한 핵심 길임을 입증합니다.



### $\texttt{BluePrint}$: A Social Media User Dataset for LLM Persona Evaluation and Training (https://arxiv.org/abs/2510.02343)
Comments:
          8 pages, 4 figures, 11 tables

- **What's New**: 이번 논문은 SIMPACT라는 프레임워크를 소개하여, 개인 정보를 보호하며 행동 기반 소셜 미디어 데이터셋을 구축하는 새로운 방법을 제시합니다. 이는 LLMs(대형 언어 모델)에 적합한 데이터 자원 부족 문제를 해결하고, 정확한 시뮬레이션을 통해 공공 담론을 탐구할 수 있게 합니다. 또한, BluePrint라는 대규모 데이터셋을 공개하여 정치적 담론을 진단하는 평가 벤치마크를 제공합니다.

- **Technical Details**: SIMPACT는 다음 행동 예측(next-action prediction)을 중심으로 구동되며, 사용자 행동을 다양한 행동 양식으로 추상화하여 클러스터링합니다. 이를 통해 개인의 신원을 노출하지 않으면서도 행동 풍부한 데이터셋을 생성할 수 있습니다. 또한, 사용자의 행동을 효과적으로 캡처하기 위해 PII(개인 식별 정보) 제거, 익명화 등의 기법을 통합하여 데이터의 무결성을 유지합니다.

- **Performance Highlights**: 연구에서는 LLMs에 대해 BluePrint 데이터셋을 사용하여 여러 모델(GPT-4.1 mini, GPT-o3 mini, Qwen 2.5)을 벤치마킹했습니다. 그 결과, 현재의 모델은 텍스트는 잘 생성하지만 실제 사용자 커뮤니티의 미묘한 행동 패턴을 재현하는 데 어려움을 겪고 있음을 발견했습니다. 이러한 결과는 SIMPACT와 같은 표준화된 데이터셋이 새로운 연구 분야에서 왜 중요한지를 강조합니다.



### CATMark: A Context-Aware Thresholding Framework for Robust Cross-Task Watermarking in Large Language Models (https://arxiv.org/abs/2510.02342)
- **What's New**: 이번 논문에서는 기존의 고정된 임계값에 의존하지 않고 문맥에 따라 동적으로 조정되는 Context-Aware Threshold Watermarking (CATMark) 프레임워크를 제안합니다. 이 시스템은 텍스트 생성 중의 세부적인 의미론적 상태를 기반으로 해서 수분할 텍스트 생성 방법을 통해 질 높은 판별 가능한 수조를 생성합니다. CATMark는 고유하게도 사전 정의된 임계값이나 특정 작업 튜닝 없이 작동합니다.

- **Technical Details**: CATMark는 로짓 클러스터링을 사용하여 텍스트 생성을 의미론적 상태로 분할하고, 이러한 상태에 맞춰 문맥 인식 엔트로피 임계값을 설정합니다. 이 방법은 고엔트로피 텍스트에 강한 워터마크를 적용하면서도 저엔트로피 텍스트의 무결성을 보장합니다. 이를 통해 다양한 생성 작업에서의 상대적 성능을 극대화할 수 있으며, 인공 신경망 모델의 샘플링 과정을 미세 조정함으로써 더 나은 결과를 도출합니다.

- **Performance Highlights**: 실험 결과 CATMark는 교차 작업에서 텍스트 품질을 개선하며 탐지 정확도를 희생하지 않고 있습니다. HumanEval에서 82.3%의 pass@1 점수와 StackEval에서 100% AUROC를 기록하여 기존 벤치마크 방법들을 초월하는 성과를 보여주었습니다. 이는 복잡하고 다양한 애플리케이션 내에서 LLM의 안정적인 활용을 가능하게 합니다.



### DRIFT: Learning from Abundant User Dissatisfaction in Real-World Preference Learning (https://arxiv.org/abs/2510.02341)
- **What's New**: 이번 논문에서는 사용자 불만족 신호(DSAT: Dissatisfaction Signal)를 활용하여 LLM(대형 언어 모델)의 동적인 훈련 방법인 DRIFT(Dissatisfaction-Refined Iterative preFerence Training)를 소개합니다. DRIFT는 실제 데이터에서 추출된 불만족 신호를 활용해 모델의 성능을 개선하고, 기존의 사람의 주관적인 평가 방식에 의존하지 않습니다. 이를 통해 더 지속적이고 경제적인 데이터 활용이 가능하도록 지원합니다.

- **Technical Details**: DRIFT 접근법은 사용자 불만족 신호를 기반으로 하여, 현재의 정책과 함께 선택된 응답을 동적으로 샘플링합니다. 이는 기존의 DPO(Direct Preference Optimization)와 같은 정적인 방식을 대체하며, 훈련 과정에서 실시간으로 업데이트된 정책에 기반해 있으므로 성능이 더욱 향상됩니다. 이 방식은 불만족 신호가 풍부하게 존재하는 특성을 이용하여 모델 훈련의 효율성을 높이고 있습니다.

- **Performance Highlights**: DRIFT로 훈련된 모델은 WildBench 작업 점수에서 최대 +6.23%(7B) 및 +7.61%(14B) 상승하며, AlpacaEval2 우승률에서도 최대 +8.95%(7B) 및 +12.29%(14B) 상승하는 성과를 보입니다. 또한, 14B 모델은 DRIFT를 통해 GPT-4o-mini보다 더 우수한 성능을 기록하면서 강력한 기준선 방법들을 초월하는 결과를 나타내었습니다. DRIFT는 다양한 높은 보상 솔루션들을 생성하며, 기술적 한계를 극복하여 탐색의 폭을 넓히고 있습니다.



### Evaluating Uncertainty Quantification Methods in Argumentative Large Language Models (https://arxiv.org/abs/2510.02339)
Comments:
          Accepted at EMNLP Findings 2025

- **What's New**: 최근 LLM(대규모 언어 모델)의 불확실성 정량화(UQ)에 대한 연구가 증가하고 있으며, 이는 신뢰할 수 있는 AI 시스템 개발에 필수적입니다. 본 논문에서는 논증적 LLM(ArgLLMs) 프레임워크에서 LLM UQ 방법의 통합을 탐구합니다. 이 프레임워크는 컴퓨터 논증에 기반하여 결정 내리기 과정을 설명 가능하게 하고, ArgLLM이 다양한 UQ 방법을 사용한 주장 검증(task)에서 성능을 평가하게 됩니다.

- **Technical Details**: 본 연구에서는 Semantic Entropy, Eccentricity, LUQ와 같은 세 가지 LLM UQ 방법을 사용합니다. 이러한 방법들은 MIT 라이센스의 LM-Polygraph 라이브러리에서 구현된 버전을 중심으로 실행됩니다. ArgLLM은 각 주장에 대해 지원 및 공격 논거를 생성하고, 이들의 신뢰도를 바탕으로 QBAFs(정량적 바이폴라 논증 체계)를 구성하여 주장의 진위를 판별합니다.

- **Performance Highlights**: 실험 결과, 단순히 직접 프롬팅(direct prompting)을 사용하는 것이 다른 복잡한 UQ 방법보다 유의미하게 더 나은 성능을 보였습니다. ArgLLM에서 생성된 주장의 신뢰도 점수는 최종 예측에 직접적으로 영향을 미치며, 다양한 설정에서 실험을 통해 이러한 결과를 입증하고 있습니다. 이러한 접근 방식은 LLM UQ 방법의 평가를 위한 새로운 방식을 제시합니다.



### Optimizing Long-Form Clinical Text Generation with Claim-Based Rewards (https://arxiv.org/abs/2510.02338)
- **What's New**: 이 연구는 임상 문서화를 자동화하기 위해 투자된 평가 통합 강화 학습 프레임워크를 소개합니다. 이 프레임워크는 Group Relative Policy Optimization (GRPO)와 DocLens라는 평가 도구를 결합하여 사실적 기반(factual grounding)과 문서의 완전성을 직접 최적화합니다. 별도의 보상 모델을 훈련할 필요 없이 문서의 질을 개선하며, 훈련 비용을 줄일 수 있는 간단한 보상 게이팅(strategy)을 활용합니다.

- **Technical Details**: 제안된 방법은 LLM(예: GPT-4o)을 평가자로 활용하여, 대화에서 추출한 임상 정보를 바탕으로 보상을 생성합니다. 이 과정을 통해 claim recall과 claim precision을 결합하여 보상을 제공하며, DocLens를 사용하여 논의에 대응하는 사실 정보를 정확히 추출합니다. 이 평가 통합 설계는 임상 요구 사항인 사실적 기반과 문서의 완전성을 최적화하면서도 컴퓨팅 효율성을 유지합니다.

- **Performance Highlights**: 연구 결과는 DocLens의 정밀도, 재현율(F1 점수)에서 일관된 개선을 보여주며, 보상 게이팅 전략을 통해 더 빠르게 수렴할 수 있음을 시사합니다. 이러한 개선은 이미 강력한 기본 모델의 성능을 보완하며, 더 도전적인 실제 작업에서도 추가적인 이점을 예상할 수 있습니다. 이 연구의 적용 가능성은 임상 지침 준수 및 청구 지원과 같은 비즈니스 지표와 최적화 목표에까지 확장될 수 있습니다.



### CRACQ: A Multi-Dimensional Approach To Automated Document Assessmen (https://arxiv.org/abs/2510.02337)
- **What's New**: 본 논문은 CRACQ라는 다차원 평가 프레임워크를 소개합니다. 이 프레임워크는 문서의 다섯 가지 특성인 Coherence(일관성), Rigor(엄밀성), Appropriateness(적절성), Completeness(완전성), Quality(품질)를 평가하도록 설계되었습니다. CRACQ는 자동화된 에세이 채점(Automated Essay Scoring)에서 얻은 통찰을 바탕으로 다른 형태의 기계 생성 텍스트로 그 초점을 확장합니다.

- **Technical Details**: CRACQ는 단일 점수 방식과는 달리 언어적, 의미적, 구조적 신호를 통합하여 누적 평가를 수행합니다. 이를 통해 전체적인 분석(holistic analysis)뿐만 아니라 각 특성 수준(trait-level analysis)의 분석이 가능합니다. 500개의 합성 보조금 제안서(synthetic grant proposals)로 훈련된 CRACQ는 LLM-as-a-judge와 비교 평가되었으며, 실제 강력한 및 약한 응용 프로그램에서도 추가 테스트가 진행되었습니다.

- **Performance Highlights**: 초기 결과에 따르면 CRACQ는 직접적인 LLM 평가보다 더 안정적이고 해석 가능한 특성 수준의 판단을 생성하는 것으로 나타났습니다. 그러나 신뢰성(reliability) 및 도메인 범위(domain scope)에서의 문제점은 여전히 남아 있습니다. 이러한 발견은 CRACQ가 머신러닝 기반의 텍스트 평가에 기여할 수 있는 가능성을 시사합니다.



### KurdSTS: The Kurdish Semantic Textual Similarity (https://arxiv.org/abs/2510.02336)
- **What's New**: 이번 연구에서는 의미 텍스트 유사성(Semantic Textual Similarity, STS)을 측정하는 최초의 쿠르디시(Kurdish) 데이터셋을 소개합니다. 이 데이터셋은 10,000개의 문장 쌍으로 구성되어 있으며, 공식적(formal) 및 비공식적(informal) 표현을 포함하여 유사성을 주석 처리하였습니다. 이는 낮은 자원(low-resource) 언어 분야에서 중요한 기여를 합니다.

- **Technical Details**: 쿠르디시 데이터셋은 형태소(morphology), 표기법(orthographic variation), 코드-믹스(code-mixing)와 같은 과제를 강조하며, Sentence-BERT, 다국어 BERT(multilingual BERT) 및 기타 강력한 기준과 비교하여 평가되었습니다. 이 연구는 쿠르디시 언어의 의미론(semantics)과 낮은 자원 NLP에 대한 향후 연구의 기초를 마련합니다.

- **Performance Highlights**: 이 연구에서는 문장 쌍 간 유사성을 측정하기 위해 여러 가지 모델을 벤치마킹하였으며, 경쟁력 있는 결과를 얻었습니다. 데이터셋과 기준은 재현 가능한 평가 스위트를 수립하였고, 향후 연구에 대한 강력한 출발점을 제공합니다.



### FormalML: A Benchmark for Evaluating Formal Subgoal Completion in Machine Learning Theory (https://arxiv.org/abs/2510.02335)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 현재 수준의 정리 증명 기능을 넘어, 수학자들이 연구 문제를 해결하는 데 필요한 중간 단계인 서브골 완성(subgoal completion)을 연구합니다. 이를 위해 FormalML이라는 새로운 Lean 4 벤치마크를 도입하였으며, 이는 머신러닝(machine learning)의 기초 이론에서 유래된 문제들을 포함합니다. 논문에서는 4937개의 다양한 난이도의 서브골 문제를 수집하여, 기존의 정리 증명 도구의 한계를 분석하고 이를 개선할 필요성을 강조합니다.

- **Technical Details**: 저자들은 Lean 4에서 서브골 추출을 위한 기법을 제안하고, 기계 학습 이론에 초점을 맞춘 FormalML 벤치마크를 구축하여 연구 수준의 정리 증명 문제에 대한 평가를 수행합니다. 이 벤치마크는 공개된 머신러닝 이론 라이브러리를 기반으로 하며, 기존의 평가 방식과 차별화된 서브골 완성 작업에 중점을 둡니다. 특히, 이에 따른 기법은 절차적 증명 스크립트에서 각 논리적 단계를 세분화하여 서브골을 효과적으로 추출하는 프로세스를 포함합니다.

- **Performance Highlights**: 주요 LLM 기반 증명 도구들의 FormalML에 대한 성능 평가 결과, 높은 난이도의 문제에서 성능 저하가 두드러짐을 알 수 있었습니다. 또한, Chain-of-thought prompting 기법이 자연어 추론에서는 효과적이나, 증명 완성에서는 효율성을 저하시키는 경향이 있음을 발견했습니다. 이러한 통찰력은 더 발전된 LLM 기반 정리 증명 도구가 수학자들을 더 잘 지원할 수 있도록 추가적인 개발이 필요함을 보여줍니다.



### Where Did It Go Wrong? Attributing Undesirable LLM Behaviors via Representation Gradient Tracing (https://arxiv.org/abs/2510.02334)
Comments:
          16 pages, 4 figures

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 바람직하지 않은 행동 원인을 진단하기 위한 새로운 프레임워크를 제시합니다. 기존의 접근법들이 가지는 한계를 극복하기 위해, 모델 활성화 공간(activation space)에서 표현과 그 기울기를 분석하여 의미 있는 정보를 제공합니다. 이 프레임워크는 샘플 레벨과 토큰 레벨 모두에서 LLM의 행동을 분석할 수 있는 능력을 보여줍니다.

- **Technical Details**: 제안된 프레임워크인 Representation Gradient Tracing (RepT)는 모델의 내부 표현을 분석하여 훈련 데이터의 원인과 LLM 반응 간의 관련성을 수립합니다. 이를 통해 기존의 매개변수(parameter) 공간 대신에 의미 있는 표현 공간을 중심으로 데이터를 추적합니다. RepT는 정보적인 레이어 선택, 샘플 수준의 귀속 방법, 그리고 특정 단어 및 구절을 식별하는 토큰 수준 분석을 통해 동작합니다.

- **Performance Highlights**: 제안된 방법은 유해 콘텐츠 추적, 백도어 포이즈 탐지 및 지식 오염 식별과 같은 다양한 과제에 대해 체계적으로 평가되었습니다. 결과는 이 방법이 샘플 수준의 귀속뿐만 아니라 특정 샘플과 구절을 정밀하게 식별하는 데 강력한 성능을 보인다는 것을 보여줍니다. 이를 통해 우리는 LLM의 위험을 이해하고 감시하며 궁극적으로 완화하기 위한 강력한 진단 도구를 제공할 수 있습니다.



### Human Mobility Datasets Enriched With Contextual and Social Dimensions (https://arxiv.org/abs/2510.02333)
Comments:
          5 pages, 3 figures, 1 table

- **What's New**: 이번 리소스 논문에서는 OpenStreetMap에서 수집한 두 개의 공개된 데이터셋을 소개합니다. 이 데이터셋은 GPS 궤적을 바탕으로 Contextual Layers(맥락 레이어)인 Stops(정차 지점), Moves(이동), Points of Interest(관심 지점, POI), 추론된 교통 수단 및 날씨 데이터가 포함되어 있습니다. 특히 Large Language Models(대형 언어 모델, LLM)로 생성된 합성 소셜 미디어 게시물이 포함되어 있어 다중 모드 및 의미론적 모빌리티 분석이 가능합니다.

- **Technical Details**: 이 데이터셋은 파리와 뉴욕이라는 두 개의 대도시에 대한 내용을 포함하고 있으며, 탭 형식과 Resource Description Framework(RDF) 형식으로 제공됩니다. 데이터는 OpenStreetMap에서 수집된 실시간 GPS 트레이스와 Weather conditions(날씨 정보) 등의 Contextual Layers로 구성되어 있습니다. 데이터는 사용자 ID가 지정된 GPS 궤적에서 수집되며, 익명화 및 사전 처리 작업을 통해 사용자 개인 정보 보호가 확보됩니다.

- **Performance Highlights**: 이 리소스는 행동 모델링, 모빌리티 예측, 지식 그래프 구성 및 LLM 기반 응용 프로그램 등 다양한 연구 작업을 지원합니다. 연구자들은 공개된 파이프라인을 통해 데이터셋을 커스터마이즈하고, 실시간 궤적, 내용 기반 정보 및 LLM에서 생성된 소셜 미디어 게시물을 활용할 수 있습니다. 이 논문의 제공하는 자원은 모빌리티 및 지식 관리 분야 연구자들의 실험과 검증을 지원하기 위해 설계되었습니다.



### A High-Capacity and Secure Disambiguation Algorithm for Neural Linguistic Steganography (https://arxiv.org/abs/2510.02332)
Comments:
          13 pages,7 figures

- **What's New**: 이 연구에서는 정보 은닉을 위한 새로운 방법인 Look-ahead Sync를 제안합니다. 이 방법은 SyncPool의 한계를 극복하고, 토큰화 모호성을 해소하면서도 보안성을 유지합니다. Look-ahead Sync는 진짜 구별할 수 없는 토큰 시퀀스에 대해서만 동기화 샘플링을 수행하여 임베딩 용량을 극대화합니다.

- **Technical Details**: Look-ahead Sync 알고리즘은 재귀적으로 동작하여, 사용 가능한 모든 경로의 엔트로피를 보존합니다. 이 연구는 zero-KL 보안을 제공하는 이론적 증명을 제시하고, Look-ahead Sync의 성능 한계를 분석합니다. 대규모 언어 모델을 사용하여 이 알고리즘이 임베딩 용량의 이론적 상한에 접근하는 것을 입증했습니다.

- **Performance Highlights**: 실험 결과에서 Look-ahead Sync는 영어에서 160%, 중국어에서 25% 이상의 임베딩 비율 향상을 보여주었습니다. 이 방법은 특히 더 큰 후보 풀을 가진 설정에서 뛰어난 결과를 낼 수 있음을 입증합니다. 이 연구는 고용량 보안 언어 스테가노그래피의 실용성을 더욱 향상시키는 중요한 기여로 평가됩니다.



### Synthetic Dialogue Generation for Interactive Conversational Elicitation & Recommendation (ICER) (https://arxiv.org/abs/2510.02331)
- **What's New**: 이 논문은 대화형 추천 시스템(CRS)에 언어 모델(LM)을 활용하는 새로운 방법론을 소개합니다. 비공식적인 CRS 데이터가 부족하여 LM의 미세 조정이 어려운 상황에서, 연구진은 LM을 사용자 시뮬레이터로 활용하여 자연스러운 대화 생성을 위한 새로운 방법을 개발했습니다. 그 결과, 사용자의 기초 상태에 일관된 자연어 대화를 생성하며, 공개 소스 CRS 데이터 세트도 생성하였습니다.

- **Technical Details**: 연구에서는 행동 시뮬레이터를 사용하여 사용자 선호 일관성을 유지하는 방법론을 제안합니다. 짧게 말해 LM 기반 사용자 시뮬레이터를 생성하고 다층적인 사용자 상호작용을 통해 선호도를 평가합니다. 이 과정에서는 다중 턴 사용자 상호작용을 생성하기 위해 기존 사용자 선택 및 응답 모델을 사용하고, LM-프롬프트( prompting)를 활용해 자연어 표현을 수정하여 비자연적이고 반복적인 대화 생성을 방지합니다.

- **Performance Highlights**: MD-DICER 데이터세트는 100,000개의 CRS 대화를 포함하고 있으며, 평가를 통해 생성된 대화가 자연스럽고 유창하며 사실적이라는 것을 보여주었습니다. 이 대화들은 기존의 템플릿 기반 대화보다 더욱 일관적으로 사용자 이해를 증진시켰습니다. 전체 및 부분 프리픽스(prompt) 대화의 평가 결과, 제안된 방법론이 LM 기반 CRSs와 행동 일관성을 갖춘 LM 기반 시뮬레이션에 대한 추가 연구를 촉진할 것임을 시사합니다.



### EntropyLong: Effective Long-Context Training via Predictive Uncertainty (https://arxiv.org/abs/2510.02330)
Comments:
          work in progress; Correspondence to: Xing Wu <wuxing@iie.this http URL>

- **What's New**: 이번 연구에서는 'EntropyLong'이라는 새로운 데이터 구성 방법을 소개합니다. 이 방법은 예측 불확실성을 활용하여 장기 의존성을 검증하는 것을 목표로 하며, 높은 엔트로피 위치에서 의미적으로 관련된 문맥을 검색하여 예측 엔트로피를 낮추는지를 평가합니다. 이러한 모델 중심 검증(model-in-the-loop verification)은 의미 있는 정보 획득(information gain)을 보장하여 단순히 상관관계를 기반으로 한 의존성에서 벗어납니다.

- **Technical Details**: EntropyLong 방법론은 네 단계의 파이프라인으로 구성되어 있습니다: 1) 적응형 임계값을 사용하여 높은 엔트로피 위치를 식별합니다. 2) 해당 위치에 대해 의미적으로 관련된 문맥을 검색합니다. 3) 검색된 문맥이 예측 엔트로피를 감소시키는지를 경험적으로 검증합니다. 4) 검증된 문맥을 원본 문서와 결합하여 장기 의존성이 있는 훈련 시퀀스를 생성합니다.

- **Performance Highlights**: EntropyLong로 모델을 훈련한 결과, RULER 벤치마크에서 8K-128K의 맥락 길이에 걸쳐 기존 접근법에 비해 두드러진 성능 향상을 보였습니다. 또한, 명령 조정 후 LongBench-v2에서의 성과도 크게 개선되었습니다. 다수의 제거 연구(ablation study)를 통해 엔트로피 기반 검증의 필요성과 효과를 추가로 입증하였습니다.



### SelfJudge: Faster Speculative Decoding via Self-Supervised Judge Verification (https://arxiv.org/abs/2510.02329)
- **What's New**: SelfJudge는 자가 감독(self-supervision)을 활용하여 넓은 NLP 작업에 걸쳐 판별자(validator)를 훈련시키는 새로운 방법을 제안합니다. 이는 인간의 주석이나 검증 가능한 정답이 필요하지 않아 다양한 자연어 처리(task)에서 활용할 수 있습니다. SelfJudge는 원래 응답의 의미를 보존하는지를 측정하여 자동으로 훈련 데이터를 생성하며, 이를 통해 모델의 추론(inference) 속도를 향상시킵니다.

- **Technical Details**: SelfJudge는 목표 모델(target model)의 응답과 대체된 토큰(token)을 비교하여 의미 보존(semantic preservation)에 기반한 판별 기준을 설정합니다. 이 방법은 원래 응답의 신뢰도에 크게 영향을 미치지 않는 토큰 교체를 허용하여 자동적으로 훈련 데이터를 생성합니다. SelfJudge는 다양한 NLP 작업에 대해 상반된 정보 개선과 효율성을 동시에 달성하는 데 중점을 둡니다.

- **Performance Highlights**: 실험 결과, SelfJudge는 기존 판별 방법에 비해 더 높은 정확도와 효율성을 제공합니다. 예를 들어, 이전 판별 방식인 AutoJudge는 성능 저하(-2.7% 정확도)에도 불구하고 +1.96개의 수용된 토큰을 달성했으나, SelfJudge는 단지 -1.0%의 정확도 저하로 +2.06개의 수용된 토큰을 기록했습니다. 이는 SelfJudge가 일반화 가능성(generalizability) 측면에서도 확장 가능하며, 여러 도메인에서의 LLM 추론을 빠르게 할 수 있는 최적의 솔루션임을 나타냅니다.



### AMANDA: Agentic Medical Knowledge Augmentation for Data-Efficient Medical Visual Question Answering (https://arxiv.org/abs/2510.02328)
Comments:
          EMNLP Findings

- **What's New**: 이 논문은 훈련 없는 데이터 효율적인 의료 비주얼 질문 응답(Med-VQA) 시스템인 AMANDA를 소개하며, 기존 Med-MLLM이 가지고 있는 본질적(intrinsic) 및 외부적(extrinsic) 추론 병목 현상을 해결합니다. AMANDA는 의료 지식 증강(Med-KA)을 통해 진단의 깊이를 향상시키고, 최신 의료 지식을 통합하여 정확한 응답을 제공합니다. 본 연구는 8개의 Med-VQA 벤치마크에서 제로샷(zero-shot) 및 몇샷(few-shot) 설정에서 주요 개선 사항을 입증합니다.

- **Technical Details**: AMANDA는 질문을 세분화(coarse-to-fine)하여 내재적인(intrinsic) 시각 이해 능력을 최대한 활용할 수 있도록 설계된 메커니즘입니다. 이 프레임워크는 의료 지식 그래프를 통해 외부 의료 지식을 검색하여 추론 과정을 구체화합니다. 여러 개의 LLM 에이전트를 통해 지식 통합의 깊이를 조절할 수 있으며, 이는 효과성과 효율성을 동시에 유지하는 데 기여합니다.

- **Performance Highlights**: 이 논문에서는 AMANDA가 기존의 접근 방식보다 월등한 성과를 보여 주며, 데이터 효율적인 연구의 가능성을 강화하고 있습니다. Zero-shot 및 few-shot 조건에서 실험을 통해 강력한 일반화 능력을 입증하며, 새로운 의료 비주얼 질문 응답 시스템의 가능성을 엿볼 수 있습니다. AMANDA는 의료 진단의 정확성을 높이고, AI 기반 의료 어시스턴트로서의 활용 가능성을 증대시킵니다.



### KAME: Tandem Architecture for Enhancing Knowledge in Real-Time Speech-to-Speech Conversational AI (https://arxiv.org/abs/2510.02327)
- **What's New**: 이 논문은 기존의 음성-음성(S2S) 모델의 장점과 자동 음성 인식(ASR) 및 텍스트 기반 대형 언어 모델(LLM)의 장점을 결합한 새로운 하이브리드 아키텍처를 소개합니다. 이 모델은 저지연 반응성을 유지하면서 깊은 지식을 통합할 수 있도록 설계되었습니다. 이는 대화형 AI 시스템에서 자연스러운 상호작용을 가능하게 합니다.

- **Technical Details**: 제안된 KAME 설계는 프론트엔드 S2S 모듈과 백엔드 LLM 모듈로 구성되어 있습니다. 프론트엔드 모듈은 사용자의 음성을 실시간으로 처리하여 즉각적인 응답을 생성하며, 백엔드 모듈에서 파생된 지식을 반영하여 응답의 질을 향상시킵니다. S2S 변환기는 여러 독립적인 토큰 시퀀스를 자동 회귀적으로 모델링하여 복잡한 대화의 맥락을 유지합니다.

- **Performance Highlights**: 평가 결과, KAME는 응답 정확도 면에서 기존 S2S 모델을 크게 초월하여 최신의 캐스케이드 시스템과 유사한 품질을 달성했습니다. 반면, 반응 속도는 기존 S2S 모델과 비슷하게 유지되고 있어, 실시간 적용 가능한 대화형 AI 구현에 적합합니다.



### Hallucination-Resistant, Domain-Specific Research Assistant with Self-Evaluation and Vector-Grounded Retrieva (https://arxiv.org/abs/2510.02326)
Comments:
          21 pages, 5 figures

- **What's New**: 이 논문에서는 RA-FSM(Research Assistant - Finite State Machine)을 소개합니다. 이는 생성 과정을 유한 상태 제어 루프에 감싸는 모듈형 GPT 기반의 연구 지원 도구로, 관련성(Relevance), 신뢰도(Confidence), 지식(Knowledge)의 순환을 기반으로 합니다. RA-FSM은 벡터 검색과 결정론적 인용 파이프라인에 기초하여, 전문가의 작업 흐름에서 더욱 유용하게 사용할 수 있는 기능을 제공합니다.

- **Technical Details**: RA-FSM의 설계는 유한 상태 기계(Finite State Machine, FSM)를 활용하여 쿼리를 필터링하고, 답변 가능성을 평가하며, 질문을 분해하고 필요할 때만 검색을 트리거하는 구조입니다. 또한, 각 답변에는 신뢰도 레이블과 중복되지 않은 인용이 제공되어 전반적인 사용성을 향상시킵니다. 이 외에도, RA-FSM은 학술 문헌, 학회, 지수, 사전 인쇄물 및 특허를 통해 특정 도메인 지식 기반을 구축하는 워크플로를 제공합니다.

- **Performance Highlights**: RA-FSM은 블라인드 A/B 리뷰에서 전문가들로부터 강력한 Notebook LM(NLM) 및 기본 GPT API 호출 대비 선호도를 나타냈습니다. 이는 경계 조건 처리를 더욱 효과적으로 하며, 더 신뢰할 수 있는 증거 사용을 가능하게 하였기 때문입니다. 커버리지와 신선도 분석 결과 RA-FSM은 NLM을 넘어서는 탐색이 가능하며, 조정 가능한 지연(latency)과 비용 오버헤드를 수반함을 보여주었습니다.



### Agentic-AI Healthcare: Multilingual, Privacy-First Framework with MCP Agents (https://arxiv.org/abs/2510.02325)
Comments:
          6 pages, 1 figure. Submitted as a system/vision paper

- **What's New**: 본 논문은 Agentic-AI 헬스케어 플랫폼을 소개하며, 이는 개인정보 보호를 고려하고 다국어로 지원되는 설명 가능한 연구 프로토타입입니다. 이 시스템은 모델 컨텍스트 프로토콜(Model Context Protocol, MCP)을 이용하여 여러 지능형 에이전트를 조율하고, 증상 체크, 약물 제안, 약속 예약 등을 수행합니다. 또한 HIPAA와 PIPEDA, PHIPA와 같은 주요 의료 데이터 보호 기준에 부합하는 개인정보 보호 및 규정 준수 레이어를 통합하였습니다.

- **Technical Details**: Agentic-AI 헬스케어 플랫폼은 모듈형 에이전트와 개인정보 보호 설계를 통합한 연구 프로토타입입니다. 이 시스템은 MCP를 통해 여러 에이전트를 조정하며, 환자와 의사가 다국어 웹 인터페이스를 통해 상호작용할 수 있도록 구성되어 있습니다. 각 에이전트는 특정 역할을 담당하며, 데이터는 MongoDB에 저장되고 필드 수준 암호화로 보호됩니다.

- **Performance Highlights**: 이 플랫폼은 영어, 프랑스어, 아랍어를 지원하여 다양한 환자 그룹에 포괄적 접근을 제공하고 있습니다. 설명 가능한 출력을 통해 각 추천은 투명한 이유를 동반하여 신뢰성을 높이고, 에이전트 간의 모듈화된 조정이 가능하여 다양한 의료 필요에 대해 유연하게 대응할 수 있습니다. 연구 프로토타입으로 제시되어 AI 헬스케어의 발전을 위한 비전을 보여줍니다.



### Hallucination reduction with CASAL: Contrastive Activation Steering For Amortized Learning (https://arxiv.org/abs/2510.02324)
- **What's New**: CASAL(Contrastive Activation Steering for Amortized Learning)은 LLMs(Large Language Models)의 헛소리(hallucination)를 줄이는 새로운 알고리즘입니다. 기존의 방식들은 실시간 모니터링을 요구했지만, CASAL은 이러한 문제를 해결하기 위해 모델의 가중치(weights)에 activate steering의 이점을 직접 통합합니다. 이를 통해 모델은 아는 질문에는 답변하고, 모르는 질문에 대해서는 답변을 피할 수 있게 됩니다.

- **Technical Details**: CASAL은 단일 transformer layer의 서브모듈만을 사용하는 경량화된 설계로, 훈련(train)의 효율성을 극대화합니다. 이 방법은 여러 단기 QA(Question Answering) 벤치마크에서 헛소리를 30%-40% 감소시켜 보다 신뢰할 수 있는 응답을 제공합니다. 또한, CASAL은 LoRA(baseline) 기반의 SFT와 DPO에 비해 30배의 연산 효율(compute-efficiency)과 20배의 데이터 효율(data-efficiency)을 보이며, 데이터가 부족한 영역에서도 효과적으로 적용될 수 있습니다.

- **Performance Highlights**: CASAL은 OOD(out-of-distribution) 도메인에서도 일반화(generalization)가 잘 이루어지는 특징을 가지고 있습니다. 이 모델은 텍스트(text-only)와 비전-언어(vision-language) 모델 모두에서 헛소리를 완화하는 데 유연한 성능을 보여줍니다. CASAL은 또한 밀집 모델(dense models)과 전문가 혼합 모델(Mixture-of-Experts, MoE) 모두에 효과적인 첫 번째 steering-based training 방법으로, 운영 시스템에서의 실제 배포에 대한 가능성을 높일 수 있습니다.



### Modeling the Attack: Detecting AI-Generated Text by Quantifying Adversarial Perturbations (https://arxiv.org/abs/2510.02319)
Comments:
          8 pages, 3 figures

- **What's New**: 최근 대규모 언어 모델(Large Language Model, LLM)의 발전은 AI 생성 텍스트 감지 시스템의 필요성을 더욱 부각시키고 있습니다. 이 논문은 기존의 공격에 대한 감지기의 취약성을 분석하고, 새로운 방어 프레임워크인 Perturbation-Invariant Feature Engineering (PIFE)을 도입하여 감지 성능을 향상시키는 방법을 제시합니다. PIFE는 입력 텍스트를 표준화한 후, 변환의 정도를 측정하여 신호를 분류기에 직접 전달하는 방식으로 작동합니다.

- **Technical Details**: 이 연구는 전통적인 적대적 훈련의 한계를 정량화하고, 텍스트와 그 정규형 간의 불일치를 모델링하여 적대적 공격에 대해 보다 강력한 감지기의 구조를 설계합니다. 본 연구에서는 기본적으로 Transformer 구조를 기반으로 한 감지기를 사용하며, 문자, 단어, 문장 수준의 다양한 공격을 평가하여 각 모델의 강건성을 비교 분석합니다. 감지 성능 평가는 True Positive Rate (TPR)와 False Positive Rate (FPR) 기준으로 수행됩니다.

- **Performance Highlights**: PIFE 모델은 기존의 적대적 훈련 기법이 세멘틱 공격에 취약한 반면, 1%의 FPR 하에서도 82.6%의 TPR을 유지하며 효과적으로 공격을 무력화할 수 있음을 보여줍니다. 이는 텍스트에 대한 변형 아티팩트를 명확히 모델링하는 것이 진정한 강건성을 실현하는 보다 유망한 경로임을 입증합니다. 이 연구는 AI-generated text detection의 문제를 해결하기 위한 새로운 가능성을 제시합니다.



### Representation Learning for Compressed Video Action Recognition via Attentive Cross-modal Interaction with Motion Enhancemen (https://arxiv.org/abs/2205.03569)
Comments:
          Accepted to IJCAI 2022

- **What's New**: 이 논문은 압축 비디오(Compressed Video) 액션 인식 분야에서 새로운 접근 방식을 제안합니다. 기존의 방법은 원시 비디오를 sparsely sampled RGB 프레임과 압축된 모션 신호(motion cues)로 대체하여 저장 및 계산 비용을 줄였습니다. 그러나 이 과정에서 발생하는 coarse하고 noisy dynamics 문제와 RGB 및 motion 모달리티 간의 융합 부족 문제를 해결하기 위해, 새로운 프레임워크인 MEACI-Net을 도입했습니다.

- **Technical Details**: MEACI-Net은 두 가지 스트림 아키텍처(architecture)를 따릅니다. 하나는 RGB 모달리티를 위한 스트림이고, 다른 하나는 모션 모달리티를 위한 스트림입니다. 특히, 모션 스트림은 다중 스케일 블록(multi-scale block)으로 구성되어 있으며, denoising(노이즈 제거) 모듈을 내장해 표현 학습(representation learning)을 향상시킵니다. 이 두 스트림 간의 상호작용은 Selective Motion Complement(SMC)와 Cross-Modality Augment(CMA) 모듈을 도입하여 강화됩니다.

- **Performance Highlights**: UCF-101, HMDB-51, Kinetics-400 벤치마크에 대한 광범위한 실험의 결과, MEACI-Net의 효과성과 효율성을 입증하였습니다. 이 모델은 두 모달리티 간의 상호작용을 최적화함으로써 인식 성능을 상당히 향상시킵니다. 특히, SMC와 CMA 모듈 덕분에 RGB 모달리티의 정보가 더욱 효과적으로 활용됩니다.



### Learning to Decide with Just Enough: Information-Theoretic Context Summarization for CMDPs (https://arxiv.org/abs/2510.01620)
- **What's New**: 본 논문에서는 Contextual Markov Decision Processes (CMDPs)를 위한 새로운 접근법으로, Large Language Models (LLMs)를 활용하여 컨텍스트 정보를 압축하는 정보이론적 요약 방법을 제안합니다. 이 방법은 상태를 보강하면서 불필요한 중복성을 줄여 의사결정에 중요한 단서를 보존하는 저차원 의미적 요약을 생성합니다. 또한, CMDPs에 대한 최초의 후회 경계(regret bounds)와 지연-엔트로피(latency-entropy) 무역off 특성을 제공하여, 컨텍스트 정보를 효과적으로 관리할 수 있는 방향을 제시합니다.

- **Technical Details**: 저자들은 관찰된 고차원적 정보에서 의사결정에 필요한 본질적인 정보를 추출하기 위한 프레임워크를 제안하며, 이를 통해 효율적인 학습을 위한 최소한의 정보량을 정량화하는 후회 경계를 수립합니다. 이 연구는 정보의 가치(informational value)와 그 표현에 드는 계산 비용(computational cost) 간의 관계를 탐구하며, 컨텍스트의 풍부함이 효율성에 미치는 영향을 조명합니다. 정보이론적 시각에서, 적절히 요약된 컨텍스트는 에이전트가 효율성과 표현력을 동시에 유지할 수 있도록 도와줍니다.

- **Performance Highlights**: 다양한 CMDP 벤치마크, 즉 탐색(navigation), 연속 제어(continuous control), 개인화 추천( personal recommendation) 등에서 실험을 통해, 제안된 요약 기반 에이전트가 비컨텍스트(non-context) 및 원시 컨텍스트( raw-context) 기준선보다 더욱 나은 의사결정 품질과 계산적 확장성을 보여주었습니다. 향상된 보상(reward), 성공률(success rate), 샘플 효율성(sample efficiency)의 개선과 함께, 지연(latency)과 메모리 사용(memory usage)의 감소를 동시에 달성했습니다. 이러한 결과는 LLM 기반 요약이 자원이 제한된 환경에서 효과적인 의사결정을 위한 확장 가능하고 해석 가능한 솔루션임을 보여줍니다.



New uploads on arXiv(cs.LG)

### Low-probability Tokens Sustain Exploration in Reinforcement Learning with Verifiable Reward (https://arxiv.org/abs/2510.03222)
- **What's New**: 이번 연구는 Verifiable Rewards (RLVR)를 활용한 Reinforcement Learning의 확장성을 개선하기 위해 Low-probability Regularization (Lp-Reg)이라는 새로운 기법을 소개합니다. Lp-Reg는 저확률 탐색 토큰인 Reasoning Sparks를 보존하고 이를 보호하기 위해 노이즈를 필터링하여 탐색 능력을 강화하는 효과를 제공합니다. 기존 방법들이 간과했던 중요한 탐색 메커니즘을 면밀히 분석하여, 무작위성을 증가시키는 데서 오는 훈련 불안정성을 해결했습니다.

- **Technical Details**: 연구에서는 RLVR의 탐색 동역학을 연구하여, 훈련 과정에서 발생하는 'Reasoning Sparks'의 점진적 소멸 현상을 발견했습니다. 연구는 저확률 토큰을 필터링하여 지각되지 않은 노이즈를 제거한 후, 남은 후보들에 대해 재정규화를 수행하는 방법을 제안합니다. 이를 통해 효과적인 정규화 매개변수인 KL divergence를 활용하여 원래 정책이 필터링된 프로시 저확률 토큰을 보호하도록 유도합니다.

- **Performance Highlights**: 실험 결과, Lp-Reg는 약 1,000 스텝 동안 안정적인 on-policy 훈련을 가능하게 하였으며, 이는 기존의 엔트로피 통제 방법들이 붕괴된 지점에서 이루어졌습니다. 다섯 개의 수학 기준에서 평균 60.17%의 정확도를 달성하였고, 이는 이전 방법보다 2.66% 향상된 성과로 나타났습니다. 이러한 안정적인 탐색은 RLVR의 성능을 극대화하는 데 기여하고 있습니다.



### To Distill or Decide? Understanding the Algorithmic Trade-off in Partially Observable Reinforcement Learning (https://arxiv.org/abs/2510.03207)
Comments:
          45 pages, 9 figures, published at NeurIPS 2025

- **What's New**: 이 논문에서는 강화 학습 (RL)에서 부분 관측 (Partial observability)의 문제를 해결하기 위해, 유가 정보 (privileged information)를 활용하는 전문가 증류 (expert distillation) 기법을 분석합니다. 특히, 퍼터베이트 블록 MDP (perturbed Block MDP)라는 이론 모델을 사용하여 전문가 증류와 전통적인 RL 사이의 알고리즘적 트레이드 오프를 탐구합니다.

- **Technical Details**: 논문에서는 두 가지 주요 요소를 다룹니다. 첫 번째는 잠재 동작 (latent dynamics)의 확률적 특성과 그것이 approximately decodability (근사 디코딩 가능성) 및 belief contraction (신념 축소)과 상관관계가 있다는 점입니다. 두 번째는 최적의 잠재 정책 (latent policy)이 항상 가장 좋은 증류 정책이 아닐 수 있다는 점입니다.

- **Performance Highlights**: 실험 결과는 부분 관측 도메인에서 유가 정보를 어떻게 효과적으로 활용할 수 있는지를 제시하며, 정책 학습의 효율성을 높이는 데 기여할 수 있는 새로운 지침을 제안합니다. 강화 학습 분야에서의 잠재적 응용 프로그램에 대한 폭넓은 논의를 촉진할 것으로 기대됩니다.



### Best-of-Majority: Minimax-Optimal Strategy for Pass@$k$ Inference Scaling (https://arxiv.org/abs/2510.03199)
Comments:
          29 pages, 3 figures

- **What's New**: 이번 연구에서는 LLM(대규모 언어 모델)의 추론 방식에 대해 새로운 접근법인 Best-of-Majority (BoM)를 제안합니다. BoM은 다수결(majority voting)과 Best-of-N (BoN)의 장점을 결합하여 성능을 개선하고, 샘플링 예산 N이 증가해도 성능이 저하되지 않는 이점을 가지고 있습니다. 이 연구는 기존의 Pass@$k$ 추론 설정에서의 최적 스케일링을 탐구하고, BoM의 성능을 정량적으로 입증합니다.

- **Technical Details**: 기존의 추론 전략인 다수결과 BoN은 Pass@$k$ 설정에서 최적 스케일링을 달성하지 못하는 것으로 나타났습니다. BoM에서는 높은 빈도의 후보 응답을 선별한 후 top-$k$ 보상을 선택하여, 추론의 실질적인 성능을 향상시킵니다. 이 연구에서 도출한 BoM의 후회(regret) 상한은 O(ϵ_opt + √(ϵ_RM²*C^*/k))로, 기존의 하한과 일치합니다.

- **Performance Highlights**: 실험 결과, BoM은 수학 문제 해결에 대한 대규모 언어 모델의 추론 성능을 평가한 결과, 다수결 및 BoN보다 우수한 성과를 보여주었습니다. BoM은 성능의 스케일링 일관성을 유지하며, 이는 기존의 다수결 및 BoN과는 대조되는 특징입니다. 본 연구는 BoM이 이론적으로도 실험적으로도 최적의 성능을 보임을 증명했습니다.



### Estimation of Resistance Training RPE using Inertial Sensors and Electromyography (https://arxiv.org/abs/2510.03197)
- **What's New**: 이 연구에서는 저항 훈련 중 개인 맞춤형 피드백을 제공하고 부상 예방을 위한 휴먼 요인으로 여겨지는 RPE(평가된 노력의 등급)의 정확한 추정 방안을 모색합니다. 69세트 이상의 데이터와 1000회 이상의 반복에서 착용형 관성과 EMG(근전도) 센서 데이터를 활용하여 RPE를 추정하는 머신 러닝 모델을 적용하였습니다. 실험 결과, 랜덤 포레스트 분류기가 41.4%의 정확도와 85.9%의 RPE 정확도를 달성하였습니다.

- **Technical Details**: 연구에서는 18세에서 25세 사이의 남자 참가자 5명으로부터 데이터를 수집하였으며, 모든 참가자는 최소 2년 이상의 저항 훈련 경험이 있습니다. 단일 팔 덤벨 이두 휘둘리기를 통해 범위가 명확한 운동을 선정하였고, Delsys Trigno Wireless EMG 시스템을 사용하여 2148.1Hz의 EMG 데이터와 370.4Hz의 IMU 데이터를 수집했습니다. 각 반복에 대해 EMG와 IMU 데이터의 동기화를 위해 후처리를 실시하였으며, 반복 경계를 확인하기 위해 가속도계 데이터를 분석하였습니다.

- **Performance Highlights**: 연구에서 특징 분석 결과, 이완 반복 시간이 RPE 예측에 있어 가장 강력한 요소로 식별되었습니다. 또한, EMG 데이터는 정확도를 향상시켰지만 데이터 품질 및 배치 민감성 등의 요인으로 특정 제한이 있었음을 나타냅니다. 마지막으로, 새롭게 생성된 EMG 및 IMU 기반 저항 훈련 데이터셋이 공개되어 향후 연구에서 재현성을 높이고, wearable-sensor 기반 운동 모니터링 시스템의 설계에 대한 통찰력을 제공합니다.



### Superposition disentanglement of neural representations reveals hidden alignmen (https://arxiv.org/abs/2510.03186)
- **What's New**: 이번 연구는 초과 겹침(hypothesis of superposition)이 심리 측정치(representational alignment metrics)와 상호작용하여 특정한 방식에서 해로운지에 대한 중요 질문을 탐구합니다. 연구진은 같은 기능을 다른 초과 겹침 배열에서 나타내는 모델이 예측 매핑 측정치에서 간섭을 일으켜서, 기대보다 낮은 정렬(alignment) 점수를 생성할 것이라고 가정합니다. 이를 통해 초과 겹침 해체(superposition disentanglement)의 필요성을 보여주고 있습니다.

- **Technical Details**: 이 연구에서는 심층 신경망(DNN)과 인간 두뇌의 시각 영역 간의 표현 유사성을 측정하는 여러 가지 정렬 측정치의 의존성을 분석합니다. 특히 소프트 매칭(soft-matching)과 같은 강화된 측정치가 초과 겹침 배열의 변화에 의해 어떻게 영향 받는지를 이론적으로 제시합니다. 이 이론은 희소 오토인코더(sparse autoencoders, SAEs)를 사용해 완전히 별개로 이루어진 초과 겹침 체계를 실험적으로 검증함으로써 지지됩니다.

- **Performance Highlights**: 이 연구의 결과는 DNN과 두뇌 간의 선형 회귀 정렬 점수가 특정 조건 아래에서 증가함을 보여주었습니다. 특히 toy 모델과 DNN 각각에서 SAEs의 잠재적 코드로 대체할 때 정렬 점수가 증가하였고, 이는 시각 도메인에서도 유사한 결과를 나타냈습니다. 이러한 결과들은 매핑 측정치가 신경 코드를 진정으로 나타내기 위해서는 초과 겹침 해체가 필수적임을 시사합니다.



### PRISM-Physics: Causal DAG-Based Process Evaluation for Physics Reasoning (https://arxiv.org/abs/2510.03185)
- **What's New**: 이 논문에서는 물리학에서의 복잡한 추론 문제를 위한 새로운 평가 프레임워크인 PRISM-Physics를 소개합니다. 기존의 물리학 벤치마크는 최종 답변만 평가하여 추론 과정을 포착하지 못했습니다. PRISM-Physics는 이 문제를 해결하기 위해 인과적 의존성을 명시적으로 인코딩한 방향 비순환 그래프(DAG)로 해결책을 표현합니다.

- **Technical Details**: PRISM-Physics는 단계별 추론을 평가하기 위해 DAG를 사용하며, 이를 통해 세밀하고 해석 가능한 스코어링이 가능합니다. 또한, 상징적 공식 동등성 매칭을 위한 완전 규칙 기반 방법을 결합하여, 다양한 공식에 걸쳐 일관된 검증을 제공합니다. 제안된 방법은 이론적 보장을 제공하며, 휴리스틱적인 판단 없이도 신뢰할 만한 결과를 도출합니다.

- **Performance Highlights**: 실험 결과, PRISM-Physics는 인공지능 모델과 비교했을 때 인간 전문가의 채점과 더 일치하는 경향을 보였습니다. 단계별 채점은 물리학에서 지속적인 추론 실패를 드러내며, 이는 나중에 모델 훈련을 위한 유용한 신호를 제공합니다. 이러한 구조적 엄밀함과 이론적 보장을 통합함으로써, PRISM-Physics는 과학적 추론을 보다 깊이 있게 발전시킬 수 있는 기초를 제공합니다.



### Q-Learning with Shift-Aware Upper Confidence Bound in Non-Stationary Reinforcement Learning (https://arxiv.org/abs/2510.03181)
- **What's New**: 이 논문에서는 비정상 강화 학습(Non-Stationary Reinforcement Learning) 문제를 다룬다. 구체적으로, 유한-horizon episodic 및 무한-horizon 할인형 마르코프 의사결정 프로세스(Markov Decision Processes, MDPs)에서 분포 변화(distribution shifts)를 고려한다. 새로운 알고리즘, Density-QUCB(DQUCB)를 제안하여 기존의 Q-learning Upper Confidence Bound 알고리즘(QUCB)의 성능을 개선하고자 한다.

- **Technical Details**: DQUCB 알고리즘은 전이 확률 밀도 함수(transition density function)를 활용하여 분포 변화를 탐지하고, Q-learning UCB의 불확실성 추정 품질을 향상시킨다. 이 알고리즘은 탐색(exploration)과 활용(exploitation) 간의 균형을 유지하며, 비정상 환경에서도 효과적으로 적응할 수 있도록 설계되었다. 또한 이론적으로 DQUCB는 QUCB보다 더 나은 후회 여지를 보장하는 것을 증명하였다.

- **Performance Highlights**: DQUCB는 GridWorld와 Frozen-Lake에서 QUCB에 비해 낮은 후회 수준을 달성하며 우수한 성능을 입증하였다. 또한 이 방법은 모델 프리 강화 학습의 계산 효율성을 갖추고 있으며, 모형 기반 RL 알고리즘들보다 공간 및 시간 복잡도가 뛰어나다. COVID-19 환자 병원 배치와 같은 실세계 작업에서도 DQUCB는 기존 Deep Q-learning Baseline들에 비해 더 낮은 누적 후회를 기록하였다.



### FTTE: Federated Learning on Resource-Constrained Devices (https://arxiv.org/abs/2510.03165)
- **What's New**: 이번 연구는 연합 학습(Federated Learning, FL)의 새로운 프레임워크인 FTTE(Federated Tiny Training Engine)를 소개합니다. FTTE는 자원 제약이 있는 엣지 디바이스에서 모델을 협력적으로 학습할 수 있도록 설계되었습니다. 이 프레임워크는 희소한 파라미터 업데이트와 클라이언트 업데이트의 나이 및 분산을 기반으로 한 스테일니스 가중 집합을 활용하여 성능을 개선합니다.

- **Technical Details**: FTTE는 500명의 클라이언트와 90%의 느린 클라이언트가 포함된 대규모 실험을 통해 기존의 동기식 연합 학습보다 81% 더 빠른 수렴 속도를 달성했습니다. 이 시스템은 실제 환경에서도 보다 적은 메모리 사용량과 통신 부하를 줄이며, 클라이언트마다 적절한 파라미터를 선택하여 메모리 제약을 최적화합니다. 세미 비동기식 집합과 함께 이 전략은 느린 클라이언트의 영향을 효과적으로 줄입니다.

- **Performance Highlights**: FTTE는 CIFAR-10 데이터셋에서 동기식 연합 학습과 비교해 80% 더 낮은 메모리 사용량과 69%의 통신 부하 감소를 기록했습니다. FTTE는 다양한 모델과 데이터 분포에서 일관되게 높은 목표 정확도를 달성하며, 자원이 제약된 디바이스에서도 강한 성능을 보입니다. 이러한 결과는 FTTE가 선진 자원 제약 환경에서 실제적인 연합 학습의 솔루션으로 자리 잡을 수 있음을 보여줍니다.



### Why Do We Need Warm-up? A Theoretical Perspectiv (https://arxiv.org/abs/2510.03164)
- **What's New**: 이번 연구는 학습률 워밍업(Learning Rate Warm-Up)의 이론적 기초를 제공하며, 이 과정이 훈련을 어떻게 개선하는지를 탐구합니다. 기존의 경험적 증거와 달리, 우리는 특별한 매끄러움 조건인 (H0,H1)(H_{0},H_{1})-smoothness를 소개하여, 이는 손실의 아급수적 성질에 따라 국소 곡률을 제한하는 방법입니다. 이 조건을 통해 우리는 학습률 워밍업이 훈련의 수렴 속도를 어떻게 증가시키는지를 이론적으로 증명했습니다.

- **Technical Details**: 연구에서 제안하는 (H0,H1)(H_{0},H_{1})-smoothness 조건은 여러 신경망 아키텍처에 적용 가능하며, 평균 제곱 오차(mean-squared error, MSE) 및 교차 엔트로피(cross-entropy) 손실을 통해 훈련된 모델에서 유효함을 입증합니다. 또한, 우리는 기울기 하강법(Gradient Descent, GD)에서 클럭으로서의 학습률 워밍업이 고정 단계 크기(fixed step-size)보다 더 빠른 수렴을 이룬다는 것을 보여주며, 상한 및 하한 복잡도 경계를 Establish합니다. 이를 바탕으로 언어 및 비전 모델 실험을 통해 이론적 통찰력을 검증하였습니다.

- **Performance Highlights**: 우리의 실험 결과는 학습률 워밍업 스케줄이 다양한 모델의 훈련 성능을 향상시키는 데 실질적인 기여를 한다는 것을 보여주었습니다. 특히 우리는 고급 신경망 구조에서의 안정성을 증가시키고, 더 높은 피크 학습률을 사용할 수 있도록 하는 등의 이점을 확인했습니다. 이로 인해, 워밍업 기법이 실제 데이터 튜닝에 필수적임을 강조합니다.



### Calibrated Uncertainty Sampling for Active Learning (https://arxiv.org/abs/2510.03162)
- **What's New**: 본 논문은 낮은 보정 오류(calibration error)를 가진 분류기를 능동적으로 학습하는 문제를 다룹니다. 기존의 Acquisition Function (AF) 중에서도 모델의 불확실성에 따라 샘플을 쿼리하는 방법이 일반적이나, 이는 비정상적인 보정 오류를 초래할 수 있습니다. 저자는 커널 기반의 보정 오류 추정기를 사용하여 보정 오류가 높은 샘플을 우선적으로 쿼리하는 새로운 AF를 제안하며, 이를 통해 DNN의 불확실성을 보다 효과적으로 활용합니다.

- **Technical Details**: 이 연구는 풀 기반의 능동 학습 설정에서, 커버리트 쉬프트(covariate shift) 하에 보정 오류를 추정하는 방식을 사용합니다. 새로운 접근 방식은 각 샘플의 보정 오류를 평가하고, 이를 기준으로 샘플의 우선 순위를 매기는 것입니다. 이를 통해 저자는 model uncertainty(모델 불확실성)를 보다 신뢰할 수 있게 활용하고, 비어있는 데이터셋에서 유용한 샘플을 효율적으로 찾을 수 있음을 이론적으로 입증합니다.

- **Performance Highlights**: 제안된 방법(CUSAL)은 MNIST, F-MNIST, SVHN, CIFAR-10, CIFAR-10-LT 및 ImageNet 데이터셋 설정에서 다른 AF 기준선보다 낮은 보정 및 일반화 오류를 기록했습니다. 이러한 실험 결과는 새로운 AF 전략이 보다 나은 기본 모델 성능을 달성할 수 있음을 확인시켜줍니다. 이를 통해 높은 보정 오류를 가진 샘플을 쿼리함으로써 모델이 신뢰성과 예측 정확도를 동시에 향상시키는 가능성을 보여줍니다.



### Mixture of Many Zero-Compute Experts: A High-Rate Quantization Theory Perspectiv (https://arxiv.org/abs/2510.03151)
- **What's New**: 이 논문은 Mixture-of-Experts (MoE) 모델을 회귀 작업에 적용하는 새로운 통찰을 제공하며, 고전적인 고속 양자화 이론(high-rate quantization theory)을 기반으로 합니다. 저자들은 MoE를 입력 공간을 세분화하여 각 세그먼트에서 단일 파라미터 전문가(expert)가 작동하도록 정의합니다. 특히, Zero-Compute 1-Sparse MoE (ZC-1SMoE) 모델을 통해 각 전문가가 예측을 위해 추가적인 계산 없이 작동할 수 있음을 주장합니다.

- **Technical Details**: ZC-1SMoE 모델은 주어진 입력 공간 세그먼트에 따라 전문가의 매개변수를 학습하는 방법론을 제시하며, 이론적으로와 경험적으로 모형 학습에서 오차를 분석합니다. 또한 1차원 입력 및 다차원 입력에 대한 테스트 오차의 상한을 공식화하고 이를 최소화하는 방법을 연구합니다. MoE의 입력 공간이 충분히 작은 영역으로 세분화되어 있어, 연속적인 밀도로 불연속 세분화를 근사할 수 있도록 하는 고속 양자화 이론을 활용합니다.

- **Performance Highlights**: 논문은 ZC-1SMoE 모델의 근사 오차를 여러 차원 입력에 대해 분석하며, 최적의 입력 공간 세분화를 형성하는 방법을 보여줍니다. 또한 전문가의 수에 따른 근사 및 추정 오류 간의 균형을 이론적으로 설명하고, 이를 통해 MoE 학습이 얼마나 개선될 수 있는지를 제시합니다. 이러한 연구는 전문가의 수와 관련된 새로운 통찰을 제공함으로써 MoE 모델의 이해를 증진시키고자 합니다.



### Taming Imperfect Process Verifiers: A Sampling Perspective on Backtracking (https://arxiv.org/abs/2510.03149)
- **What's New**: 이번 연구에서는 언어 모델의 생성능력과 프로세스 검증기(process verifier)를 결합한 테스트 시점 알고리즘(test-time algorithms)에 대해 논의합니다. 특히, VGB라는 새로운 프로세스 유도 테스트 시간 샘플링 알고리즘을 소개하며, 이는 학습된 검증기의 오류가 생성 과정에서 중대한 실패를 초래하는 문제를 해결하고자 합니다. VGB는 이론적으로 기반이 있는 백트래킹(backtracking) 방식을 통해 검증기 오류에 대해 더 나은 강인성을 제공합니다.

- **Technical Details**: VGB는 자가 회귀 생성(autoregressive generation)을 부분 생성의 트리(tree of partial generations)에서의 무작위 산책(random walk)으로 해석합니다. 전이 확률(transition probabilities)은 프로세스 검증기와 기본 모델에 의해 유도되며, 필수적으로 백트래킹은 확률적으로 발생합니다. 이 과정은 이론 컴퓨터 과학의 근사 계수 및 샘플링 문헌에서 제안된 Sinclair-Jerrum 랜덤 워크를 일반화한 것입니다.

- **Performance Highlights**: 실험 결과, VGB는 합성 및 실제 언어 모델링 과제에서 다양한 메트릭(metrics)으로 기준선을 뛰어넘는 성능을 보여줍니다. 이 알고리즘은 검증기 오류에 대한 강인성을 개선하여 생성 과정에서의 신뢰성을 높이는 데 기여합니다.



### Enhancing XAI Narratives through Multi-Narrative Refinement and Knowledge Distillation (https://arxiv.org/abs/2510.03134)
- **What's New**: 이 논문은 설명 가능한 인공지능(Explainable AI) 분야에서 Counterfactual Explanations (CE)의 중요성을 강조하며, 언어 모델(Language Models)을 활용하여 직관적이고 사용자 중심의 반사실적 설명을 생성하는 새로운 파이프라인을 제안합니다. 특히, 이 연구는 의료 및 금융과 같은 투명성이 중요한 분야에서 모델의 결정 과정을 더 잘 이해할 수 있도록 돕는 데 기여하고 있습니다. 더불어, 우리의 접근 방식은 다양한 규모의 언어 모델을 적용하여 모델의 추론 능력을 개선하고 실제 적용 가능성을 높이는 데 중점을 두고 있습니다.

- **Technical Details**: 본 논문에서는 반사실적 설명의 품질을 높이기 위해 지식 증류(knowledge distillation) 기법과 정제 메커니즘을 도입하여 소형 언어 모델이 대형 모델에 비견될 수 있는 성능을 발휘할 수 있도록 설계하였습니다. Counterfactual Narrative Generation Problem (CNGP) 을 정의하고 이를 해결하기 위한 방법론을 제시합니다. 이를 통해 생성된 자연어 내러티브는 사용자가 이해할 수 있는 형식으로 반사실적 예시를 제공합니다.

- **Performance Highlights**: 제안된 파이프라인은 사용자 접근성을 증가시킴으로써 모델의 투명성 및 사용성을 높이며, 정량적 평가 방법을 통해 생성된 내러티브의 품질을 검증할 수 있는 새로운 평가 프레임워크도 제시하였습니다. 결과적으로, 이 연구는 반사실적 추론의 실질적인 채택을 촉진하고, AI의 정책 요구사항에 부합하여 실생활의 규제된 환경에서도 효과적으로 활용될 수 있도록 합니다.



### Signature-Informed Transformer for Asset Allocation (https://arxiv.org/abs/2510.03129)
- **What's New**: 이번 연구에서 제안한 Signature-Informed Transformer (SIT)는 포트폴리오 자산 할당(policy allocation)을 직접 최적화하여 강건한 분산 포트폴리오를 구축하는 혁신적인 딥러닝 프레임워크입니다. SIT는 기하학적 표현을 위한 Rough Path Signatures를 사용하여 자산의 복잡한 동역학을 잘 포착하며, 자산 간의 선행-후행 관계에 대한 금융적 귀납적 편향(inductive bias)을 도입합니다. 이 방법은 전통적인 모델 대비 향상된 성능을 보여주며, 특히 예측-최적화를 사용하는 모델에 비해 결정적 우위를 가집니다.

- **Technical Details**: SIT의 주요 혁신점은 세 가지 기둥으로 구성된 통합 아키텍처에 있습니다. 첫째, Path-wise Feature Representation을 통해 각 자산의 가격 이력을 사용하여 특징을 생성하고, 둘째, Signature-Augmented Attention 메커니즘을 통해 자산 쌍 간의 선행-후행 관계를 모델링합니다. 마지막으로, Decision Alignment 기법을 통해 포트폴리오 구성 목표와의 일치를 극대화하며, 이는 조건부 가치-at-위험(Conditional Value-at-Risk, CVaR) 최적화를 통해 이루어집니다.

- **Performance Highlights**: SIT는 S&P 100 지수의 일일 주식 데이터를 평가한 결과 전통적인 및 딥러닝 기반의 모델을 능가하는 성능을 보였습니다. 특히, 위험 인식 자산 할당에 있어 포트폴리오 인식 목표와 기하학적 귀납적 편향이 필수적임을 보여줍니다. 이러한 결과는 SIT의 강력한 성능을 뒷받침하며, 복잡한 금융 시장 환경에서도 안정적인 결과를 도출할 수 있음을 나타냅니다.



### Real Time Headway Predictions in Urban Rail Systems and Implications for Service Control: A Deep Learning Approach (https://arxiv.org/abs/2510.03121)
- **What's New**: 이번 연구는 도시 메트로 시스템에서 효율적인 실시간 배차를 위한 새로운 딥러닝 프레임워크를 제안합니다. 이 프레임워크는 Convolutional Long Short-Term Memory (ConvLSTM) 모델을 중심으로 하며, 기차 간격의 복잡한 시공간 전파를 예측하는 데 중점을 둡니다. 또한, 기존의 연구가 승객 수요 예측이나 비정상적 상황에 초점을 맞춘 반면, 본 연구는 운영 제어의 선제적 접근 방식을 강조하고 있습니다.

- **Technical Details**: 제안된 모델은 기차의 역사적 간격 데이터와 계획된 터미널 간격을 주요 입력으로 사용하여 향후 간격의 변화를 정밀하게 예측합니다. 제안된 방법론은 자동 위치 추적(AVL) 시스템을 활용하여 기차 사건의 상세 기록을 생성하고, 이를 기반으로 시공간 데이터로 변환하는 다단계 전처리 파이프라인을 구축합니다. 최종적으로 ConvLSTM 아키텍처를 통해 기차 간격의 동시적 공간 상관관계와 복잡한 시간적 의존성을 효과적으로 모형화합니다.

- **Performance Highlights**: 실험은 Chicago Transit Authority (CTA) Blue Line을 대상으로 하여 진행되었으며, 이 결과 제안된 ConvLSTM 모델이 우수한 예측 성능을 보여주었습니다. 다양한 예측 시간 범위와 지정된 역에서의 headway 예측 정확도를 분석하였으며, 계획된 터미널 간격 통합의 이점과 함께 운영자의 선제적 배차 전략을 지원할 수 있는 모델의 능력을 논의하였습니다. 이 연구는 메트로 시스템의 서비스 일관성과 승객 만족도를 크게 향상시킬 수 있는 효율적인 도구를 제공합니다.



### AdaBet: Gradient-free Layer Selection for Efficient Training of Deep Neural Networks (https://arxiv.org/abs/2510.03101)
- **What's New**: 본 논문에서는 AdaBet이라는 새로운 접근 방식을 소개하며, 이는 사용자 특정 런타임 데이터 분포에 효과적으로 적응하기 위해 사전 훈련된 신경망을 모바일 및 엣지 장치에서 효율적으로 사용할 수 있도록 설계되었습니다. 기존의 방법들은 라벨이 있는 데이터나 서버 사이드 메타 훈련에 의존하여 제약이 있었으나, AdaBet은 레이블이나 그래디언트 없이 중요한 레이어를 선택할 수 있게 해줍니다. 이 접근 방식은 Betti Numbers를 기반으로 하여 활성화 공간의 토폴로지적 특징을 분석함으로써 이루어집니다. AdaBet을 통해 메모리 소비를 40% 줄이고 평균 5%의 분류 정확도를 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: AdaBet은 DNN(Deep Neural Network) 재훈련을 위한 프레임워크로, 그래디언트가 없는 레이어 선택 방법을 제공합니다. 이 방법은 각 레이어의 활성화 출력을 기반으로 하여 학습 용량을 정량화하고, 메모리 효율성을 높이기 위해 레이어 크기로 정규화된 Betti Number를 사용합니다. AdaBet은 모바일 및 엣지 장치에서 자원을 효율적으로 사용할 수 있게 해주며, 점진적인 레이어 선택 프로세스를 통해 전체 모델의 역전파를 수행할 필요가 없습니다. 이러한 접근은 비지도 학습 환경에서도 적용 가능하여 범용성을 높입니다.

- **Performance Highlights**: AdaBet을 다양한 벤치마크 모델과 데이터셋에 대해 평가한 결과, 기존의 그래디언트 기반 방법들에 비해 평균 5% 더 높은 정확도를 달성하며, 평균 피크 메모리 사용량을 40% 감소시키는 성과를 냈습니다. 특히, ResNet, VGG16, MobileNet 및 ViT 등의 인기 있는 사전 훈련된 DNN 모델에 대해 뛰어난 성능을 보였습니다. 이러한 결과는 AdaBet이 자원 효율성과 모델 정확성 간의 균형을 잘 맞춘다는 점에서 주목할 만합니다.



### Adaptive Node Feature Selection For Graph Neural Networks (https://arxiv.org/abs/2510.03096)
- **What's New**: 이 연구는 GNN(그래프 신경망)을 위한 적응형 노드 특성 선택(apaptive node feature selection) 접근 방식을 제안합니다. 이 방법은 모델 훈련 중 불필요한 특성을 식별하고 제거하여 해석 가능성 및 성능을 향상시킵니다. 기존의 특성 중요도 측정 방법이 복잡한 그래프 구조와의 의존성으로 인해 적합하지 않을 수 있는 반면, 이 연구는 훈련 중에 특성이 변경될 때 검증 성능의 변화를 기반으로 중요성을 측정합니다.

- **Technical Details**: 제안된 방법은 명시적 가정 없이 GNN의 예측을 기반으로 특성 중요도 점수를 동적으로 측정하는 것입니다. 주기적으로 각 노드 특성의 값을 무작위로 섞어(GNN 성능의 변경을 측정하여) 특성의 기여를 평가합니다. 이 방식은 그래프 데이터에 대한 가정 없이도 적용 가능하여 다양한 그래프 아키텍처에 유연성을 제공합니다.

- **Performance Highlights**: 실험 결과, 본 알고리즘은 여러 벤치마크 데이터세트에서 모든 가용 노드 특성을 사용하는 GNN의 성능에 필적하는 결과를 도출하며, 모델 아키텍처와 호모필릭(homophilic) 또는 헤테로필릭(heterophilic) 노드 라벨과 같은 다양한 설정에 유연하게 적용될 수 있음을 보여줍니다.



### Distilled Protein Backbone Generation (https://arxiv.org/abs/2510.03095)
- **What's New**: 이 논문은 단백질 디자인의 새로운 접근법을 제시하며, diffusion 및 flow 기반 생성 모델들이 단백질 backbone 생성을 위해 어떻게 활용될 수 있는지를 보여줍니다. 최근의 기술 발전에도 불구하고, 이들 모델의 샘플링 속도는 여전히 큰 문제로 남아 있습니다. 이에 따라, score distillation (점수 증류) 기법을 적용하여 적은 샘플링 단계로 성능을 유지하면서 샘플링 시간을 비약적으로 줄일 수 있는 방법을 모색했습니다.

- **Technical Details**: 연구진은 Score identity Distillation (SiD) 기법을 기반으로 한 새로운 증류 프레임워크를 개발하여, 단백질 backbone 생성기를 훈련시켰습니다. 이 프레임워크는 샘플링 시 낮은 온도로 작업하여, 불필요한 구조적 오류를 최소화하고 디자인 가능성을 높입니다. 결과적으로, 16단계 생성을 통해 기존 teacher model과 유사한 성능을 유지하면서도 20배 이상의 샘플링 시간 개선을 이루었습니다.

- **Performance Highlights**: 저자들은 제안한 방식으로 생성된 단백질 구조가 디자인 가능성, 다양성 및 독창성을 유지하였음을 입증하였습니다. 이로 인해, 기존의 diffusion 기반 모델보다 훨씬 빠른 샘플링 속도를 달성했습니다. 또한, 이러한 기법은 대규모 단백질 디자인의 가능성을 높여, 실제 단백질 공학 응용에의 활용을 가속화합니다.



### Bootstrap Learning for Combinatorial Graph Alignment with Sequential GNNs (https://arxiv.org/abs/2510.03086)
Comments:
          27 pages, 10 figures, 12 tables

- **What's New**: 이 논문은 그래프 정렬 문제에서 전통적인 최적화 방법을 능가하는 새로운 체인 기법을 도입합니다. 기존의 그래프 신경망(GNN) 접근 방식이 제한적인 성능을 보인 반면, 이 체인 기법은 여러 GNN을 세분화하여 각 네트워크가 이전 단계의 유사성 행렬을 개선하도록 학습하게 합니다. 이는 노드 정렬 품질에 대한 이산 순위 정보를 통합하여 결과적으로 각 GNN이 부분 해결책을 개선하는 부트스트랩 효과를 만듭니다.

- **Technical Details**: 체인 기법은 GNN의 반복적인 개선을 통해 그래프 정렬 문제(GAP)가 최적 솔루션에 도달하는 데 도움이 됩니다. 이 방법은 전통적인 최적화 알고리즘과 결합하여 GNN의 성능과 기존의 전문 솔버의 성능을 능가하는 하이브리드 방법을 만듭니다. 저자들은 GNN의 패러다임을 변화시켜 일반적인 그래프가운데서만 해당하는 고유한 경우를 포함한 다양한 그래프 유형에서 실험을 수행했습니다.

- **Performance Highlights**: 실험 결과, 체인 GNN은 기존 방법에 비해 3배 이상의 정확도를 달성하였으며, 특별히 정규 그래프의 경우 모든 경쟁 접근 방식이 실패하는 지점에서 독특한 해결책을 제공합니다. 최적화 후처리 단계에서 전통적인 최적화 방법과 결합했을 때, 이 방법은 그래프 정렬 벤치마크에서 최신 솔버보다 상당한 성과를 보여주었습니다.



### A Unified Deep Reinforcement Learning Approach for Close Enough Traveling Salesman Problem (https://arxiv.org/abs/2510.03065)
- **What's New**: 이 논문은 주로 관심을 받지 못한 Close-Enough Traveling Salesman Problem (CETSP)을 해결하기 위한 새로운 접근 방식을 제안합니다. CETSP는 NP-hard 문제에서 변형된 것으로, 각 노드의 근처를 방문하는 경우에만 해당 노드가 방문된 것으로 간주됩니다. 이를 해결하기 위해 Markov Decision Process (MDP)로 CETSP를 공식화하고, 유니파이드 이중 디코더 DRL(UD3RL) 프레임워크를 제안합니다.

- **Technical Details**: UD3RL은 노드 선택(node selection)과 웨이포인트 결정(waypoint determination)을 분리하여 의사결정 과정을 구성합니다. 이를 위해, 적응형 인코더는 의미 있는 특성을 추출하는 데 사용되고, 노드 디코더(node-decoder)와 위치 디코더(loc-decoder)가 두 개의 하위 작업을 처리합니다. 또한, k-nearest neighbors(k-NN) 서브그래프 상호작용 전략을 도입하여 공간적 추론을 향상시킵니다.

- **Performance Highlights**: 실험 결과, UD3RL은 기존 방법들에 비해 솔루션 질(solution quality)과 실행 시간(runtime)에서 뛰어난 성능을 발휘하며, 다양한 문제 크기와 반경 유형(예: constant와 random)에서 강력한 일반화 능력을 보여줍니다. UD3RL은 동적 환경에서도 높은 견고성을 유지하며, 여러 실제 상황에 적용 가능성이 높습니다.



### Comparative Analysis of Parameterized Action Actor-Critic Reinforcement Learning Algorithms for Web Search Match Plan Generation (https://arxiv.org/abs/2510.03064)
Comments:
          10 pages, 10th International Congress on Information and Communication Technology (ICICT 2025)

- **What's New**: 본 연구에서는 Soft Actor Critic (SAC), Greedy Actor Critic (GAC), 그리고 Truncated Quantile Critics (TQC) 알고리즘의 성능을 고차원 의사결정 작업에서 평가하였습니다. 파라미터화된 행동(Parameterized Action, PA) 공간을 통해 반복 네트워크를 배제하여, Microsoft NNI를 통해 하이퍼파라미터 최적화를 수행하였습니다. 특히, 빠른 훈련 시간과 높은 보상을 기록한 Parameterized Action Greedy Actor-Critic (PAGAC)이었으며, 이는 복잡한 행동 공간에서의 속도와 안정성에서 분명한 장점을 제공하였습니다.

- **Technical Details**: 연구에서는 완전 관측 가능한 환경 내에서 두 개의 벤치마크인 Platform-v0와 Goal-v0에서 GAC, TQC와 SAC의 성능을 비교하였습니다. 강화학습(Reinforcement Learning, RL) 문제로 매칭 계획 생성을 재구성한 Luo et al.의 접근 방식은 상태 신호와 매칭 계획의 동적 파라미터화를 통해 유연성을 강화하였습니다. 본 연구는 이러한 알고리즘들이 막대한 차원의 연속 행동 공간에서 탐험(exploration)과 착취(exploitation) 간의 균형을 추구하였다는 점을 강조합니다.

- **Performance Highlights**: PAGAC는 Platform 게임에서 5,000 에피소드를 41분 24초, 로봇 축구 골 게임에서는 24분 4초에 완료하며, 다른 알고리즘들보다 빠른 학습 시간과 높은 성능을 자랑하였습니다. 연구 결과 SAC, GAC, TQC 간의 성능 비교는 각기 다른 전략의 효과와 적용 가능성을 제시하였으며, PAGAC은 빠른 수렴과 견고한 성능을 요구하는 과제에 적합함을 입증하였습니다. 향후 연구는 엔트로피 정규화를 조합하여 안정성을 개선하는 하이브리드 전략에 대한 탐사를 고려할 예정입니다.



### ZeroShotOpt: Towards Zero-Shot Pretrained Models for Efficient Black-Box Optimization (https://arxiv.org/abs/2510.03051)
- **What's New**: 본 논문에서는 ZeroShotOpt라는 새로운 범용 프리트레인(pretrained) 모델을 소개합니다. 이 모델은 2D에서 20D까지의 연속적인 블랙박스 최적화(tasks) 문제들을 해결하는 데 사용됩니다. 기존의 베이지안 최적화(Bayesian Optimization, BO) 기술의 한계를 극복하기 위해, 우리는 방대한 최적화 경로 데이터를 통해 오프라인 강화 학습(offline reinforcement learning)을 활용합니다.

- **Technical Details**: ZeroShotOpt는 200200M 파라미터를 가진 트랜스포머 기반(transformer-based) 모델로, 다양한 BO 변형들에서 수집된 20M 이상의 합성 함수(synthetic functions) 경로를 바탕으로 사전 학습되었습니다. 이를 통해 모델은 최적화 동역학을 robust하게 이해하고, 낮은 평가 예산(low evaluation budget)에서도 효율적으로 동작할 수 있는 기반을 제공합니다. 또한, ZeroShotOpt는 이전의 트랜스포머 모델들보다 더 나은 zero-shot 일반화를 보여줍니다.

- **Performance Highlights**: ZeroShotOpt는 다양한 unseen global optimization benchmarks에서 뛰어난 성능을 발휘하며, 기존의 선두적인 BO 방법들과 유사한 수준의 샘플 효율성을 달성합니다. 이 모델은 최소한의 조정으로도 후속 확장 및 개선을 위한 견고한 기반을 제공합니다. 연구 결과는 github에 오픈소스로 공개되어 다른 연구자들이 접근하고 활용할 수 있도록 했습니다.



### Bayesian E(3)-Equivariant Interatomic Potential with Iterative Restratification of Many-body Message Passing (https://arxiv.org/abs/2510.03046)
- **What's New**: 이번 연구에서는 Bayesian E(3) 등가 MLP를 개발하여 많은 입자 사이의 상호작용을 반복적으로 재구성하는 방식으로, MLP의 신뢰성을 크게 향상시켰습니다.  새로운 공동 에너지-힘 음의 로그 우도(NLL$_\text{JEF}$) 손실 함수는 에너지와 원자 사이의 힘에서 발생하는 불확실성을 명시적으로 모델링하여 기존의 손실 함수들에 비해 더 나은 정확도를 보여줍니다. 한편, 불확실성 예측 및 활성 학습 작업에서 예방할 수 있는 장점을 밝혔다.

- **Technical Details**: Bayesian 신경망(BNN)은 가중치를 무작위 변수로 취급하여 예측 불확실성을 정량화하는 원리를 제공합니다. 본 논문에서는 NLL$_\text{JEF}$ 손실 함수를 통해 에너지와 힘에 대한 불확실성을 동시에 분석할 수 있으며, RACE 아키텍처에 기반한 BNN 모델을 개발하여 효율적인 불확실성 인식을 가능하게 하였습니다. 또한, 심층 앙상블(Deep Ensembles) 및 확률적 가중치 평균화(Stochastic Weight Averaging)와 같은 여러 베이지안 접근 방식을 체계적으로 벤치마킹하였습니다.

- **Performance Highlights**: 제안된 방법론은 QM9, rMD17 및 PSB3와 같은 여러 기준 테스트에서 검증되었으며, OOD(out-of-distribution) 테스트 세트인 oBN25를 새롭게 소개하였습니다. Bayesian MLP는 최신 모델들과 비교해 경쟁력 있는 정확도를 보여주며, 불확실성 기반의 활성 학습, OOD 탐지 및 에너지/힘 보정 작업을 더 효과적으로 수행할 수 있음을 증명했습니다. 이 연구는 대규모 원자 시뮬레이션을 위한 불확실성 인식 MLP 개발의 강력한 프레임워크로서 Bayesian 등가 신경망의 활용 가능성을 제시합니다.



### CHORD: Customizing Hybrid-precision On-device Model for Sequential Recommendation with Device-cloud Collaboration (https://arxiv.org/abs/2510.03038)
Comments:
          accepted by ACM MM'25

- **What's New**: 이번 연구에서는 CHORD라는 새로운 프레임워크를 제안합니다. CHORD는 하이브리드 정밀도를 가진 온디바이스 모델을 커스터마이징하여 장치-클라우드 협력 방식을 통해 순차적 추천을 구현합니다. 이를 통해 모델 정확도를 유지하면서 자원 적응형 배포를 달성할 수 있습니다.

- **Technical Details**: CHORD에서는 채널별 혼합정밀도 양자화(channel-wise mixed-precision quantization)를 활용하여 사용자 맞춤형 추천을 제공합니다. 연구에서는 모델 파라미터의 민감도 분석을 통해 사용자 프로필에 맞는 양자화 전략을 쉽게 매핑하는 방법을 개발하였습니다. 이 과정에서 모델 압축을 달성하면서도 커뮤니케이션 오버헤드를 줄이는데 집중하였습니다.

- **Performance Highlights**: 세 가지 실제 데이터셋과 두 가지 인기 있는 네트워크(SASRec 및 Caser)에 대한 실험 결과, CHORD는 정확성, 효율성, 적응성을 모두 입증했습니다. 사용자는 2비트의 전략 인코딩을 통해 클라우드와의 소통 비용을 크게 줄이고, 한 번의 전방 패스를 통해 개인화된 모델 적응을 이룰 수 있음을 보였습니다.



### Lightweight Transformer for EEG Classification via Balanced Signed Graph Algorithm Unrolling (https://arxiv.org/abs/2510.03027)
- **What's New**: 본 논문에서는 뇌전증 환자와 건강한 개인을 EEG 신호를 통해 구분하는 문제를 다룹니다. 저자들은 주로 이진 분류 문제를 해결하기 위해 경량의 해석 가능한 transformer와 유사한 신경망을 구축합니다. 이 방법은 주어진 EEG 신호를 기반으로 한 균형 잡힌 유향 그래프(positive graph)에서 작동하며, 특히 이상적인 저통과 필터와 스펙트럼 노이즈 제거 알고리즘을 결합했습니다.

- **Technical Details**: 제안된 알고리즘은 균형 잡힌 유향 그래프(𝐆𝟑)에서 학습된 노이즈 제거기를 기반으로 하며, 각 그래프의 대응되는 긍정적 그래프(positive graph)에서 구현됩니다. 연구는 또한 그래프 신호 처리(GSP) 관점에서 신호를 효율적으로 분리하는 방법론을 제안하고 있습니다. 이 접근법은 또한 장기 데이터 설명 및 학습에 대한 새로운 통찰을 제공합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 대규모 딥러닝 모델들과 비교하여 비슷한 수준의 분류 성능을 달성하였으며, 설계된 신경망은 훨씬 더 적은 매개변수를 사용합니다. 예를 들어, 제안된 모델은 97.6%의 분류 정확도를 달성하여 기존의 transformer 기반 모델(85.1%)에 비해 뛰어난 성능을 보였습니다. 이러한 결과는 자원 제한이 있는 EEG 장치에서의 실용적인 적용 가능성을 강조합니다.



### Differentially Private Wasserstein Barycenters (https://arxiv.org/abs/2510.03021)
Comments:
          24 pages, 9 figures

- **What's New**: 이번 연구에서는 Wasserstein barycenter를 differential privacy (DP) 환경에서 계산하는 알고리즘을 처음으로 제안하고 있습니다. 이러한 알고리즘은 민감한 데이터로부터 구축된 empirical distributions에 적용할 수 있으며, 높은 품질의 private barycenters를 생성할 수 있습니다. 이로 인해 머신러닝 및 통계에서의 응용 가능성이 확대됩니다.

- **Technical Details**: 이 논문에서는 DP 조건 하에 Wasserstein barycenters를 계산하기 위한 두 가지 효율적인 알고리즘을 제안합니다. 첫 번째 방법은 private Wasserstein distance coresets를 사용하는 블랙박스 변환을 이용하여 ε-DP 알고리즘을 제공합니다. 두 번째로, 데이터가 클러스터링 되어 있을 때 잘 작동하는 경량의 (ε, δ)-DP 알고리즘도 소개됩니다.

- **Performance Highlights**: 실험 결과, 제안된 알고리즘은 MNIST 및 대규모 미국 인구 데이터셋에서 높은 정확도와 강력한 개인정보 보호 성과를 보여주었습니다. 결과적으로, 공공 데이터의 보호와 품질을 동시에 충족할 수 있는 실질적인 방법이 제시되었습니다.



### Learning Robust Diffusion Models from Imprecise Supervision (https://arxiv.org/abs/2510.03016)
- **What's New**: 본 논문에서는 DMIS라는 통합 프레임워크를 제안하여 부정확한 감독(imprecise supervision) 하에서 강건한 확산 모델(Diffusion Models) 학습을 가능하게 합니다. 이는 확산 모델 분야에서 최초로 체계적으로 연구된 사항으로, 학습 목적을 우선 가능도 최대화(likelihood maximization) 문제로 공식화하고, 생성(component)과 분류(component) 요소로 분해하였습니다. 이 접근 방식은 부정확한 라벨 분포를 모델링하고, 확산 분류기를 통해 클래스 후확률(class-posterior probabilities)을 추론하는 과정을 포함합니다.

- **Technical Details**: 프레임워크는 생성 모델링 동안 부정확한 라벨 조건부 점수(imprecise-label conditional score)를 청정 라벨의 조건부 점수(clean-label conditional scores)로 표현할 수 있다는 점에 착안했습니다. 또한, 우리는 효율적인 후확률 추론을 위해 최적화된 시간 단계 샘플링 전략(optimized timestep sampling strategy)을 도입하여 계산 복잡성을 줄였습니다. 이 과정에서 부정확한 라벨링에 대응하는 가중치 저노이즈 점수 매칭(objective)이 제안되어, 깨끗한 주석 없이도 라벨 조건 학습이 가능해졌습니다.

- **Performance Highlights**: 다양한 형태의 부정확한 감독에 대한 광범위한 실험 결과, 우리의 프레임워크로 학습한 CDMs는 우수한 생성 품질과 품사 구분(class-discriminative) 샘플을 일관되게 생성함을 보여주었습니다. 이미지 생성, 약한 감독 학습(weakly supervised learning), 노이즈 데이터 세트 응축(noisy dataset condensation)과 같은 여러 작업에서 성능을 입증하며, 미래 연구를 위한 견고한 기준선이 설정되었습니다. 본 연구는 부정확한 감독 하에서의 강건한 CDM 훈련을 위한 통합 프레임워크의 필요성을 강조합니다.



### Distributional Inverse Reinforcement Learning (https://arxiv.org/abs/2510.03013)
- **What's New**: 이 논문에서는 오프라인(offline) 역 강화 학습(Inverse Reinforcement Learning, IRL)을 위한 새로운 분포 기반(framework) 프레임워크를 제안합니다. 회귀하는 보상 함수와 리턴에 대한 전체 분포를 동시에 모델링하여, 기존 IRL 방식이 놓치는 전문가 행동의 풍부한 구조를 포착합니다. 또한, 이 방법은 왜곡 위험 측정(distortion risk measures, DRM)을 정책 학습에 통합함으로써 리턴에 대한 첫 번째 차수의 확률 지배(first-order stochastic dominance, FSD) 위반을 최소화합니다.

- **Technical Details**: 제안된 방법은 보상 분포를 학습하고 분포를 고려한 정책(distribution-aware policies)을 복구하는 데 중점을 두고 있습니다. 기존의 MaxEnt IRL 방법을 넘어서 전형적인 기대 리턴(expected return)만을 최적화하는 것이 아니라, 전체 리턴 분포를 매칭하는 접근 방식을 택합니다. 이를 통해 보상 함수의 불확실성을 고려하며, 로봇 작업이나 생물 행동 모델링과 같은 여러 실제 응용에 적합한 결과를 생성합니다.

- **Performance Highlights**: 실험 결과는 합성(인공) 벤치마크, 실제 생리 행동 데이터, 그리고 MuJoCo 제어 과제에서 제안된 방법이 유의미한 보상 분포를 회복하고, 오프라인 IRL 설정에서 최신 성능을 달성했음을 보여줍니다. 이를 통해 위험을 고려한 모방 학습(imitation learning)을 포함하여 다양한 행동 분석에 대한 모델링 가능성을 확장합니다. 이 연구는 특히 IRL의 전통적인 한계를 극복하는 새로운 시도로 주목받고 있습니다.



### BrainIB++: Leveraging Graph Neural Networks and Information Bottleneck for Functional Brain Biomarkers in Schizophrenia (https://arxiv.org/abs/2510.03004)
Comments:
          This manuscript has been accepted by Biomedical Signal Processing and Control and the code is available at this https URL

- **What's New**: 이번 연구에서는 뇌 기능 연결망을 분석하기 위해 새로운 그래프 신경망 프레임워크인 BrainIB++를 소개합니다. 이 모델은 정보 경색 원칙을 적용하여 해석 가능성을 위해 훈련 과정 중 가장 정보성이 높은 뇌 영역을 서브 그래프로 식별합니다. 또한, BrainIB++는 기존 머신러닝 기법의 한계인 수동 특성 엔지니어링 문제를 해결하고, 해석 가능성과 신뢰성을 높이고자 합니다.

- **Technical Details**: BrainIB++는 여러 개의 은닉층을 가진 신경망 구조를 활용하여 뇌의 기능적 연결성을 모델링합니다. 이 모델은 AAL 파셀레이션 지도를 사용하는 대신, 100,000명 이상의 참가자로부터 얻은 대규모 resting-state fMRI 데이터셋을 사용하여 점진적 연결성 네트워크를 추정합니다. 또한, 서브 그래프 생성을 위한 노드 선택 방식은 해석 가능성을 크게 향상시킵니다.

- **Performance Highlights**: 우리 모델은 세 가지 다중 집단 정신분열증 데이터셋에 걸쳐 아홉 가지 기존 뇌 네트워크 분류 방법과 비교하여 우수한 진단 정확도를 보여줍니다. 식별된 서브 그래프는 기존의 정신분열증 임상 바이오마커와 일치하며, 시각, 감각 운동, 그리고 고차 인지 기능 네트워크의 이상을 강조합니다. 이러한 결과는 BrainIB++의 실제 진단 응용 가능성을 강화하는 데 기여합니다.



### From high-frequency sensors to noon reports: Using transfer learning for shaft power prediction in maritim (https://arxiv.org/abs/2510.03003)
Comments:
          Keywords: transfer learning, shaft power prediction, noon reports, sensor data, maritime

- **What's New**: 본 연구에서는 전통적인 센서 데이터에서 얻은 지식을 활용하여, 노온 리포트(noon reports) 기반으로 선박의 샤프트 파워(shaft power)를 예측하는 새로운 접근법을 제안합니다. 이 접근법은 고주파(high-frequency) 데이터로 처음 모델을 학습한 후, 저주파(low-frequency) 노온 리포트를 이용해 미세 조정하는 방법입니다. 연구 결과, 노온 리포트만으로 학습한 경우에 비해 평균 절대 백분율 오류가 개선되었습니다.

- **Technical Details**: 선박의 연료 소비를 정확히 예측하기 위해 서로 다른 선박에서 얻은 데이터를 조합해 지식 transfer learning을 적용했습니다. 이 과정에서 자매선(sister vessels) 및 유사선(similar vessels) 데이터와의 미세 조정을 진행했으며, 기계 학습(machine learning) 모델을 사용해 이러한 데이터를 학습했습니다. 전통적인 해양 공학 접근법과는 달리, 이 방법은 데이터 기반으로 샤프트 파워 예측의 정확성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, 자매선의 경우 평균 10.6%의 오류 감소가 있었고, 유사선에서 3.6%, 다른 선박에서는 5.3%의 개선이 있었습니다. 본 연구의 성과는 신뢰할 수 있는 선박 데이터 부족 문제를 해결하며, 오히려 노온 리포트를 활용하는 새로운 접근법을 제시하였다는 점에서 의의가 있습니다. 또한, 다른 선박에서도 다양한 작업에 적용 가능성이 있음을 보여주었습니다.



### Confidence and Dispersity as Signals: Unsupervised Model Evaluation and Ranking (https://arxiv.org/abs/2510.02956)
Comments:
          15 pages, 11 figures, extension of ICML'23 work: Confidence and Dispersity Speak: Characterizing Prediction Matrix for Unsupervised Accuracy Estimation

- **What's New**: 이 논문은 배포 환경에서 레이블이 없는 테스트 데이터로 모델의 일반화 성능을 평가하기 위한 통합적이고 실용적인 프레임워크를 제시합니다. 주목할 점은 'confidence'와 'dispersity' 두 가지 내재적 속성이 모델의 예측 신뢰도와 다양성을 측정하여 일반화 성능에 대한 강력한 신호를 제공한다는 것입니다. 또한, 하이브리드 메트릭이 기존의 단일 측면 메트릭보다 일관되게 우수한 성능을 보임을 입증합니다.

- **Technical Details**: 모델 성능 평가 방식은 크게 데이터셋 중심 평가와 모델 중심 평가로 나눌 수 있습니다. 데이터셋 중심 평가에서는 여러 레이블 없는 테스트 데이터셋에서 고정 모델의 정확도를 추정하고, 모델 중심 평가에서는 단일 레이블 없는 테스트 데이터셋에서 여러 후보 모델 중 가장 적합한 모델을 선정합니다. 이 논문에서는 'confidence'와 'dispersity'를 결합한 하이브리드 메트릭을 통해 두 가지 평가 방식 모두에서 견고한 성능을 보였습니다.

- **Performance Highlights**: 하이브리드 메트릭은 데이터셋 중심 평가와 모델 중심 평가 모두에서 가장 뛰어난 성능을 보여줍니다. 특히, 예측 매트릭스의 핵심 노름(nuclear norm)은 다양한 배포 시나리오에서 강력하고 정확한 성능을 유지하며, 클래스 불균형이 있는 경우에도 신뢰성을 제공합니다. 이러한 발견은 레이블이 없는 모델 평가를 위한 실용적이고 일반화된 기반을 제공합니다.



### ContextFlow: Context-Aware Flow Matching For Trajectory Inference From Spatial Omics Data (https://arxiv.org/abs/2510.02952)
Comments:
          26 pages, 9 figures, 13 tables

- **What's New**: 본 논문에서는 ContextFlow라는 새로운 흐름 매칭(flow matching) 프레임워크를 제안합니다. 이 프레임워크는 공간적으로 분해된 omics 데이터(spatially resolved omics data)에서 구조적 조직과 기능적 역동성을 이해하는 데 필요한 정보를 탐지합니다. ContextFlow는 기존의 접근 방식과 달리 조직의 지역적 구성과 리간드-수용체 통신 패턴을 결합하여 통계적으로 일관되면서 생물학적으로 의미 있는 궤적(trajecdory)을 생성합니다.

- **Technical Details**: ContextFlow는 전이 가능성 매트릭스(transition plausibility matrix)를 도입하여 최적 운송(optimal transport) 문제를 정규화합니다. 이 과정에서 지역적인 조직과 리간드-수용체 통신 패턴을 통합하여 데이터의 맥락적 풍부함을 최대한 활용합니다. 새로운 비용 기반(cost-based) 및 엔트로피 기반(entropy-based) 통합 방식을 설계하여 전이 동역학을 제약합니다. 이러한 방식은 현대 하드웨어에서 효율적인 Sinkhorn 최적화에 적합하게 만들어졌습니다.

- **Performance Highlights**: 세 가지 데이터 세트를 통해 평가한 결과, ContextFlow는 다양한 정량적(quantitative) 및 정성적(qualitative) 지표에서 기존의 최첨단 흐름 매칭 방법을 지속적으로 능가하였습니다. 생물학적 일관성과 추론 정확성이 뛰어난 궤적을 생성하여 재생 및 개발 데이터 집합에서 그 효과를 입증하였습니다. ContextFlow는 생물학적으로 유의미한 결과를 제공하여 전반적인 임상 응용 가능성을 높입니다.



### Ergodic Risk Measures: Towards a Risk-Aware Foundation for Continual Reinforcement Learning (https://arxiv.org/abs/2510.02945)
- **What's New**: 이번 연구에서는 위험 인식 의사결정을 기반으로 한 지속적 강화 학습(continual RL)의 첫 번째 이론적 접근을 제시합니다. 기존의 지속적 RL 연구는 주로 리스크 중립적(decision-making) 관점에서 진행되었으나, 본 논문에서는 리스크 인식(risk-aware) 접근법을 도입하여 에이전트가 평균 이상의 보상을 최적화할 수 있도록 합니다. 이러한 새로운 접근은 에이전트가 재난 상황을 피할 수 있도록 하는 데 집중해야 함을 강조합니다.

- **Technical Details**: 연구진은 기존의 리스크 측정 이론이 지속적 RL의 요구와 상충됨을 확인했습니다. 이로 인해, 지속적 학습(continual learning)에 적합한 새로운 리스크 측정인 에르고딕 리스크 측정(ergodic risk measures)을 도입하여 리스크 측정 이론을 확장하였습니다. 에르고딕 리스크 측정은 평균 보상 마르코프 결정 과정(average-reward Markov decision process, MDP)을 기반으로 하여 제시되며, 안정성과 유연성을 동시에 충족하는데 도움을 줍니다.

- **Performance Highlights**: 연구에서는 에르고딕 리스크 측정의 직관적인 매력과 이론적 타당성을 입증하기 위한 사례 연구를 포함하고, 수치적 결과를 제공합니다. 이 결과는 지속적 학습 상황에서 리스크 인식 의사결정의 필요성을 강조하며, 실제 환경에서의 생존을 위해 필수적인 리스크 관리의 중요성을 보여줍니다. 연구는 지속적 학습에 있어 리스크 인식의 기초를 마련하여, 향후 연구 및 응용에 기여할 것입니다.



### RAxSS: Retrieval-Augmented Sparse Sampling for Explainable Variable-Length Medical Time Series Classification (https://arxiv.org/abs/2510.02936)
Comments:
          Accepted at the NeurIPS 2025 Workshop on Learning from Time Series for Health

- **What's New**: 이번 연구는 의료 시간 시계열 분류에서 Stochastic Sparse Sampling (SSS) 프레임워크를 확장하여 Retrieval-Augmented Sparse Sampling (RAxSS)을 제안합니다. 이 방법은 정적 평균화 대신 비슷한 윈도우의 혼합을 통해 예측의 신뢰성과 설명가능성을 높이고자 합니다. 특히, 의사결정을 위한 중요한 증거를 제공하며, 변동성이 큰 임상 데이터를 처리하는 데 적합합니다.

- **Technical Details**: RAxSS는 여러 의료 센터에서 수집한 뇌파(iEEG) 데이터를 활용하여, 잡음이 많고 길이가 가변적인 시간 시계열 데이터를 처리합니다. 모델은 각 윈도우의 이웃을 고려하여 유사성을 기반으로 가중치를 부여하고, 이를 통해 최종 예측을 도출합니다. 유사도는 주로 Pearson 또는 Cosine 유사성을 통해 계산되며, 최종 결정 점수는 윈도우의 기여를 정량적으로 보여줍니다.

- **Performance Highlights**: RAxSS는 멀티센터 iEEG 데이터에서 경쟁력 있는 성능을 보여줍니다. Cosine 변형이 AUC 0.8046으로 기존 SSS와 비슷한 성능을 보였으며, Pearson 변형은 F1 Score에서 우수한 결과를 나타내었습니다. 이러한 결과는 의료 환경에서의 신뢰할 수 있는 분류를 지원하며, 설명 가능성도 함께 제공합니다.



### FeDABoost: Fairness Aware Federated Learning with Adaptive Boosting (https://arxiv.org/abs/2510.02914)
Comments:
          Presented in WAFL@ECML-PKDD 2025

- **What's New**: 이번 연구에서는 비동질적 데이터 설정에서 연합 학습(Federated Learning, FL)의 성능과 공정성을 개선하기 위해 새로운 프레임워크인 FeDABoost를 제안합니다. FeDABoost는 동적 부스팅 메커니즘과 적응형 그래디언트 집계 전략을 통합하여, 낮은 로컬 오류율을 가진 클라이언트에 더 높은 가중치를 부여하여 글로벌 모델에 신뢰할 수 있는 기여를 장려합니다.

- **Technical Details**: FeDABoost는 특정 클라이언트에서 하드 투 크래프팅 예제를 강조하기 위해 초점 손실(focal loss) 집중 매개변수를 조정함으로써 저성과 클라이언트를 동적으로 부스트합니다. 클라이언트의 로컬 성능에 기반하여 동적으로 클라이언트의 업데이트를 가중치 조정함으로써, 비균질 데이터 환경에서 신뢰할 수 없는 업데이트로 인한 위험을 줄일 수 있습니다.

- **Performance Highlights**: MNIST, FEMNIST, CIFAR10 데이터셋에서 FeDABoost의 성능을 평가한 결과, FedAvg 및 Ditto와 비교하여 공정성과 경쟁력을 갖춘 성능을 달성함을 확인했습니다. 특히, FeDABoost는 각 클라이언트 간의 모델 성능 차이를 줄이면서 평균 예측 정확도를 높이는 것을 목표로 하여, 데이터의 이질성과 클라이언트 불균형 문제를 효과적으로 완화합니다.



### Learning Explicit Single-Cell Dynamics Using ODE Representations (https://arxiv.org/abs/2510.02903)
Comments:
          26 pages, 10 figures. Preprint under review

- **What's New**: 이번 논문에서는 세포 분화의 역학을 모델링하기 위한 Cell-Mechanistic Neural Networks (Cell-MNN)을 제안합니다. 이 모델은 스템 세포에서 조직 세포로의 세포 진화를 지배하는 선형화된 ODE (Ordinary Differential Equation)를 통해 세포의 다이나믹스를 처리합니다. Cell-MNN은 전 과정이 end-to-end로 디자인되어 있으며, 생물학적으로 의미 있는 유전자 상호작용을 학습할 수 있습니다.

- **Technical Details**: Cell-MNN은 인코더-디코더 아키텍처로, 고차원 상태 공간에서 세포의 상태가 시간에 따라 진화하는 과정을 모델링합니다. 이 모델은 측정 과정에서 세포를 하나의 스냅샷으로 관찰하며, 다양한 시간 지점에서 유전자 발현 벡터를 뽑습니다. Cell-MNN은 ODE 표현을 통해 세포의 동역학을 효과적으로 예측하고 해석 가능한 유전자 상호작용을 탐색할 수 있습니다.

- **Performance Highlights**: Cell-MNN은 여러 단일 세포 벤치마크에서 경쟁력 있는 성능을 보여주며, 대규모 데이터셋으로 확장할 수 있는 가능성을 가지고 있습니다. 또한, Cell-MNN은 OT (Optimal Transport) 전처리를 완전히 제거하여 계산 효율성을 높이며, 여러 데이터셋에서의 상호 학습이 용이하게 설계되었습니다. 최종적으로, TRRUST 데이터베이스에 대한 정량적 검증을 통해 학습된 유전자 상호작용의 해석 가능성을 입증합니다.



### DMark: Order-Agnostic Watermarking for Diffusion Large Language Models (https://arxiv.org/abs/2510.02902)
- **What's New**: 이번 논문에서는 확산 대형 언어 모델( diffusion large language models, dLLMs)을 위한 첫 번째 워터마킹 프레임워크인 DMark를 소개합니다. 기존의 워터마킹 방법들이 dLLM의 비순차적(degrees of non-sequential) 디코딩 때문에 실패하는 문제를 해결합니다. 그 결과, DMark는 전통적인 워터마킹 기술의 한계를 뛰어넘는 새로운 방안을 제공합니다.

- **Technical Details**: DMark는 세 가지 상호 보완적인 전략을 도입하여 워터마크 탐지를 회복합니다. 첫째, predictive watermarking은 실제 맥락(context)이 없을 때 모델 예측 토큰을 사용합니다. 둘째, bidirectional watermarking은 확산 디코딩에 고유한 전방 및 후방 종속성을 활용하여 워터마킹을 강화합니다. 셋째, predictive-bidirectional watermarking은 두 접근 방식을 결합하여 탐지 강도를 극대화합니다.

- **Performance Highlights**: 여러 dLLM을 대상으로 한 실험에서 DMark는 1%의 위양성(false positive) 비율로 92.0-99.5%의 탐지율을 달성하며 텍스트 품질을 유지합니다. 이는 기존 방법의 단순 변형이 49.6-71.2%에 그친 것과 비교되는 성과입니다. DMark는 텍스트 조작에 대해서도 강인성을 보여줌으로써 비자율 언어 모델에 효과적인 워터마킹의 가능성을 확립합니다.



### RoiRL: Efficient, Self-Supervised Reasoning with Offline Iterative Reinforcement Learning (https://arxiv.org/abs/2510.02892)
Comments:
          Accepted to the Efficient Reasoning Workshop at NeuRIPS 2025

- **What's New**: 이 논문에서는 기존의 Test-Time Reinforcement Learning(TTRL)의 한계를 극복한 새로운 방법, RoiRL(Reasoning with offline iterative Reinforcement Learning)을 제안합니다. RoiRL은 정답 레이블 없이도 같은 수준의 정책 최적화를 달성하면서, 메모리와 계산 비용을 대폭 줄일 수 있는 가벼운 오프라인 학습 방식을 채택합니다. 실험 결과에 따르면, RoiRL은 TTRL보다 2.5배 더 빠르게 훈련되며, 다양한 추론 벤치마크에서 지속적으로 우수한 성과를 보이는 것으로 나타났습니다.

- **Technical Details**: RoiRL은 간단한 가중 로그 우도 목표를 최적화하는 반복적인 오프라인 루프를 통해 작동합니다. 이는 온라인 RL이나 참고 모델을 유지할 필요 없이, 자가 생성된 보상을 활용하여 안정적인 훈련 환경을 제공합니다. 이 방법은 모델 크기에 비례하여 효율적으로 확장되며, 기존의 TTRL 방식에서 발생하는 메모리 과부하 및 복잡성을 해소합니다.

- **Performance Highlights**: RoiRL은 기존 TTRL 방법을 대체할 수 있는 위치에 있으며, 메모리와 계산 자원 사용을 줄이면서도 훈련 속도와 성능을 향상시킵니다. 실험 결과, RoiRL은 추론 성능이 향상될 뿐만 아니라, 메모리 측면에서도 경제적이며 확장 가능성을 보여줍니다. 헬퍼 모델 없이도 자가 생성된 보상 메커니즘을 통해, 대규모 언어 모델의 자기 개선 경로를 확립할 수 있음을 입증했습니다.



### Knowledge-Aware Modeling with Frequency Adaptive Learning for Battery Health Prognostics (https://arxiv.org/abs/2510.02839)
Comments:
          12 pages, 4 figures, 4 tables

- **What's New**: 본 논문은 Karma라는 새로운 지식 기반 모델을 제안하여 배터리 용량 추정 및 남은 유효 수명(RUL) 예측의 정확성과 신뢰성을 향상시킵니다. 이 모델은 주파수 적응 학습(Frequency-Adaptive Learning)을 통해 배터리의 해리 과정 및 비선형성을 더욱 수월하게 처리합니다. Karma는 두 개의 스트림으로 구성된 심층 학습 아키텍처를 사용하여 장기 저주파 및 단기 고주파 동적 변화를 각각 포착합니다.

- **Technical Details**: Karma의 방법론은 배터리 신호를 주파수 대역별로 분해한 후, CNN-LSTM과 BiGRU 구조를 통해 각각의 저주파 및 고주파 신호를 처리합니다. 이러한 방식은 시간에 따른 비선형 변화를 보다 효율적으로 모델링할 수 있게 합니다. 또한, Karma는 경험적인 배터리 지식을 통합하여 복원력 있는 예측을 제공하며, 파라미터 최적화를 위해 입자 필터(Particle Filter) 방법을 채택합니다.

- **Performance Highlights**: Karma는 실험적으로 두 개의 주요 데이터세트에서 기존 최신 알고리즘보다 평균적으로 각각 50.6%와 32.6%의 예측 오차 감소를 달성했습니다. 이러한 성능 향상은 Karma 모델의 견고성과 유연성을 입증하며, 다양한 적용 분야에서의 배터리 관리의 안전성과 신뢰성을 높일 수 있는 잠재력을 제시합니다.



### Subject-Adaptive Sparse Linear Models for Interpretable Personalized Health Prediction from Multimodal Lifelog Data (https://arxiv.org/abs/2510.02835)
Comments:
          6 pages, ICTC 2025

- **What's New**: 이번 논문에서는 주관적 건강 예측을 위한 해석 가능한 모델링 접근 방식인 Subject-Adaptive Sparse Linear (SASL) 프레임워크를 제안합니다. SASL은 일반 최소 제곱 회귀(Ordinary Least Squares regression)를 핵심으로 하며, 개인별 상호작용을 통합하여 개인의 건강 데이터를 분석합니다. 또한, 지표 결과가 연속 프로세스의 이산화된 형태라는 점을 인식하고, 이를 고려한 회귀-임계치 접근 방식을 개발하였습니다.

- **Technical Details**: SASL은 통계적으로 강건한 모델 생성을 위해 반복적 역방향 변수 제거 방법을 사용합니다. 이 방법은 유의한 예측 변수를 잔여 제곱 합(Residual Sum of Squares) 분석을 통해 체계적으로 제거합니다. 또한, 절대적인 정확성을 제공하고자 compact LightGBM 모델에서의 출력을 선택적으로 통합하여 해석 가능성을 유지하는 동시에 예측 성능을 향상시킵니다.

- **Performance Highlights**: CH-2025 데이터세트에서 실시된 평가 결과, SASL-LightGBM 하이브리드 프레임워크는 복잡한 블랙박스 방법과 비교해 예측 성능이 유사한 수준을 보였지만, 훨씬 적은 수의 파라미터와 더 높은 투명성을 제공하였습니다. 이런 방식은 임상 의사와 실무자들에게 명확하고 실행 가능한 통찰력을 제공합니다.



### Multi-scale Autoregressive Models are Laplacian, Discrete, and Latent Diffusion Models in Disguis (https://arxiv.org/abs/2510.02826)
- **What's New**: 본 논문에서는 Visual Autoregressive (VAR) 모델을 반복 정제(iterative refinement) 프레임워크의 관점으로 재조명합니다. VAR를 단순한 다음 스케일 자기 회귀(next-scale autoregression)로 바라보는 대신, 결정적인 전방 과정(deterministic forward process)으로 형식화하고, 이를 통해 랩라시안 스타일의 잠재 피라미드(Laplacian-style latent pyramid)를 구성합니다. 이러한 새로운 접근은 VAR의 효율성 및 충실도를 설명하는 세 가지 설계 선택을 분리하여 명확하게 제시합니다.

- **Technical Details**: VAR 모델은 VQ-VAE 토크나이저를 통해 입력 이미지를 그리드 형태의 불연속 코드로 변환합니다. 각 스케일에서 모든 토큰을 동시에 예측하고, 이는 2D 위치성(locality)을 유지하면서 생성 복잡도를 줄이는 데 기여합니다. VAR은 다음 스케일 예측(next-scale prediction) 프레임워크를 채택하고, 이 과정에서 여러 범위의 스케일에서 새롭게 래핑합니다.

- **Performance Highlights**: VAR의 성능 평가는 정밀도(fidelity)와 속도(speed) 측면에서 기여 요소들을 정량화하는 것으로 진행되었습니다. CONTROLLED EXPERIMENTS를 통해, 본 연구는 VAR이 적은 반복 횟수로도 경쟁력 있는 이미지 품질을 생성할 수 있음을 입증하였습니다. 또한 이 프레임워크는 퍼뮤테이션-불변 그래프 생성 및 중간 범위의 날씨 예측 등 다양한 생성 작업에 확장될 수 있는 가능성을 보여줍니다.



### The Curious Case of In-Training Compression of State Space Models (https://arxiv.org/abs/2510.02823)
- **What's New**: 이 논문에서는 CompreSSM이라는 새로운 인-트레이닝(compression) 압축 기법을 소개합니다. 이 기술은 SSM(State Space Models)의 차원을 효과적으로 줄이는 동시에 압축되지 않은 모델의 표현력을 대체로 유지합니다. CompreSSM은 전통적인 모델 차원 축소 기법인 balanced truncation을 통해 훈련 중에 작고 중요하지 않은 상태 차원을 선택적으로 제거합니다. 이러한 접근 방식은 SSM의 효율성을 극대화하는 데 기여합니다.

- **Technical Details**: SSM은 상태 유지와 업데이트 과정을 통해 오랜 시퀀스 모델링 작업을 처리합니다. 강건한 훈련과 추론을 위해 Hankel singular value 분석을 적용하여 각 상태의 에너지를 측정하고, 그에 따라 차원 축소를 수행합니다. 이 새로운 기술은 주변(주변) 제어 이론의 원리를 바탕으로 하며, SSM의 dominant Hankel singular values를 관찰함으로써 훈련 전반에 걸쳐 중요한 차원만을 보존합니다. CompreSSM은 고전적인 모형 정렬(low-dimensional) 기법을 포함하여 확장 가능합니다.

- **Performance Highlights**: CompreSSM은 훈련을 획기적으로 가속화하면서도 더 큰 압축되지 않은 모델과 유사하거나 더 높은 정확도를 달성할 수 있음을 보여줍니다. 실험 결과, 초기 훈련 과정에서 큰 상태 부분을 잘라냈음에도 불구하고 압축된 모델은 작업에 중요한 구조를 유지하면서도 성능을 저하시키지 않습니다. 이러한 성과는 대규모 언어 및 비전 및 오디오 모델링 작업에서의 SSM의 잠재력을 한층 더 확장합니다.



### FlexiQ: Adaptive Mixed-Precision Quantization for Latency/Accuracy Trade-Offs in Deep Neural Networks (https://arxiv.org/abs/2510.02822)
Comments:
          16 pages. 14 figures. To be published in the Proceedings of the European Conference on Computer Systems (EUROSYS '26)

- **What's New**: 최근 신경망은 크기와 복잡성이 크게 증가하면서 높은 계산 오버헤드와 추론 지연을 초래하고 있습니다. 이러한 흐름 속에서, FlexiQ라는 적응형 혼합 정밀도( mixed-precision ) 양자화 스킴이 제안되었습니다. FlexiQ는 작은 값 범위를 가진 특징 채널( feature channels )에 저비트-width( low-bitwidth ) 계산을 선택적으로 적용하고, 추론 부하의 변동을 실시간으로 관리할 수 있도록 설계되었습니다.

- **Technical Details**: FlexiQ는 특징 채널에서의 세밀한 혼합 비트 연산을 도입하여 높이 비트-width를 다른 채널에 적용하면서 동시에 선택된 채널에 저비트-width 연산을 적용합니다. 또한, 낮은 비트-width의 채널 비율을 실시간으로 조정할 수 있어, 변동하는 추론 작업 부하에 맞춰 지연 시간을 조정할 수 있습니다. 이를 통해 FlexiQ는 지연과 정확성 간의 효율적이고 동적인 균형을 제공합니다.

- **Performance Highlights**: FlexiQ는 4-bit 양자화를 사용하여 11개의 비전 모델(CNN 및 비전 트랜스포머)에 대해 평균적으로 6.6%의 정확도 향상을 달성하고, 기존의 네 가지 첨단 양자화 기법보다 우수한 성능을 보여주었습니다. 우리 모델은 또한 정확도-지연 시간 간의 효율적인 거래를 제공하며, 50% 4-bit / 50% 8-bit 모델은 평균 0.6% 정확도 손실만 발생시키며, 100% 4-bit 모델 더해 8-bit 모델 대비 40%의 속도 향상을 달성합니다.



### Online Learning in the Random Order Mod (https://arxiv.org/abs/2510.02820)
- **What's New**: 이 논문에서는 랜덤 순서 모델(random-order model)에서 강화된 탐색 알고리즘을 통해 온라인 학습을 위한 기초적인 구조를 제안합니다. 이러한 모델은 손실(loss) 순서가 적대자(adversary)에 의해 결정되어 무작위로 배열된 후 학습자에게 제시됩니다. 이는 특히 비정상성(non-stationarity)의 영향을 받을 수 있어데이터의 통계적 특성을 유지하면서 성능을 저하할 수 있습니다.

- **Technical Details**: 랜덤 순서 모델은 전통적인 i.i.d. 스토캐스틱 모델과 적대적인 모델의 중간 정도에 위치합니다. 논문에서는 이러한 모델이 갖는 문제를 탐구하며, 강화된 알고리즘인 Simulation을 통해 스토캐스틱 모델을 랜덤 순서 모델에 적응하는 방법을 보여줍니다. 이를 통해 성능 보장을 크게 손상시키지 않으며, 지연(deferred feedback)과 제약(constraint) 문제 등을 다룰 수 있습니다.

- **Performance Highlights**: 제안된 알고리즘들은 랜덤 순서 모델에서도 예측 성과를 개선할 수 있는 효율성을 보입니다. 지연과 관련된 문제에서 O~(T+d)와 O~(T) 두 가지 성능 지표와 더불어, 스위칭 비용을 가진 문제에서도 T√T 레이트를 복구할 수 있음을 보여주고 있습니다. 궁극적으로, 논문은 랜덤 순서 모델에서 비에러 확률적 학습의 특성을 VC 차원(VC dimension)에 의해 정의하며, 이는 일반적인 적대적 모델과의 구분을 제공합니다.



### Mitigating Spurious Correlation via Distributionally Robust Learning with Hierarchical Ambiguity Sets (https://arxiv.org/abs/2510.02818)
- **What's New**: 이번 연구는 Group DRO(Group Distributionally Robust Optimization)의 계층적 확장을 제안하여, 집단 간 및 집단 내 불확실성을 모두 해결할 수 있는 방법을 제공합니다. 기존의 방법들이 주로 집단 간 분포 변화에만 중점을 두었던 반면, 우리는 집단 내 분포 변화까지 아우르며 더욱 강력한 모델 성능을 확보할 수 있는 새로운 기준 설정도 도입하였습니다. 이러한 접근은 기존의 강인 학습 방법들이 종종 실패하는 조건에서도 뛰어난 강인성을 보여줍니다.

- **Technical Details**: 우리는 Group DRO 프레임워크 안에서 계층적 모호성 집합을 도입하여, 집단 간과 집단 내 불확실성을 모두 포함시키는 접근을 선택했습니다. 우리의 방식은 Wasserstein 거리 기반의 수식화로, 이론적 및 실증적인 지원을 통해 분포적으로 강인한 학습 방법 설계에 효과적이라는 점을 보여줍니다. 최적화 알고리즘은 계산 효율성이 높아, 실제 상황에서도 효과적으로 구현될 수 있도록 합니다.

- **Performance Highlights**: 우리의 방법은 표준 기준 벤치마크에서 뛰어난 성능을 보였으며, 특히 소수 집단의 분포 변화에 강력한 내구성을 발휘했습니다. 기존의 방법이 허점을 보인 곳에서, 우리는 일관되게 강력한 성능을 달성하여 소수 집단 데이터가 의도적으로 왜곡되지 않도록 할 수 있습니다. 이 연구는 집단 간 및 집단 내 불확실성을 포괄해야 할 필요성을 강조합니다.



### Dissecting Transformers: A CLEAR Perspective towards Green AI (https://arxiv.org/abs/2510.02810)
- **What's New**: 이번 논문에서는 Transformer 아키텍처의 핵심 구성 요소에 대한 추론 에너지를 미세하게 분석하는 최초의 실증적 연구를 소개합니다. 기존의 연구들이 에너지 효율성을 단순한 모델 수준에서 다루던 것과 달리, 본 연구에서는 Attention 블록과 같은 개별 요소의 에너지 소비를 정밀하게 측정하고 최적화할 수 있는 방법론인 CLEAR를 제안합니다. 이를 통해 LLM이 차지하는 환경적 비용을 보다 명확히 이해하고 개선할 수 있습니다.

- **Technical Details**: CLEAR는 마이크로초 단위의 구성 요소 실행과 밀리초 단위의 에너지 센서 모니터링 간의 시간 불일치를 극복하기 위해 새롭게 설계된 기술입니다. 이 방법론은 15개의 다양한 모델을 평가하며 각 구성 요소의 에너지를 정확히 측정할 수 있도록 설계되었습니다. 에너지를 구성 요소 단위로 분해하여 고유의 소비 패턴을 비교하고, 다양한 인풋 길이와 플로팅 포인트 정밀도에 대해 실증 분석을 수행합니다.

- **Performance Highlights**: CLEAR 방법론을 사용한 결과, Attention 블록이 계산당 소비하는 에너지가 상대적으로 높은 점이 밝혀졌습니다. FLOP의 수치가 에너지 소비와 비례하지 않음을 보여 줌으로써, FLOP만으로는 실제 에너지 비용을 제대로 평가할 수 없음을 강조합니다. 이번 연구는 구성 요소별 에너지 기준선을 확립하고, 에너지 효율적인 Transformer 모델 구축을 위한 최적화의 기초를 마련하는 데 기여합니다.



### Relevance-Aware Thresholding in Online Conformal Prediction for Time Series (https://arxiv.org/abs/2510.02809)
- **What's New**: 이 논문에서는 기계 학습에서 불확실성 정량화(uncertainty quantification)가 중요한 이슈로 부각되었음을 다루고 있습니다. 특히, 시계열 데이터에 대한 온라인 정합 예측(Online Conformal Prediction, OCP) 방법을 통해 데이터 분포의 변화에 대응할 수 있는 새로운 접근법을 제안합니다. 제안된 방법은 예측 구간의 적합성을 높이고, 급격한 임계값(threshold) 변화를 방지하여 예측 구간을 더욱 좁히는 데 기여합니다.

- **Technical Details**: 논문에서는 OCP 방법에서 예측 구간의 유효성을 단순한 이항 평가(inside/outside)가 아닌, 예측 구간과 실제 값 간의 관련성을 활용하는 방법으로 향상시키고자 합니다. 이를 위해 예측 구간의 경계와 실제 값의 거리 기반으로 평가하는 새로운 피드백 방식을 도입하였습니다. 이 접근법은 OCP의 여러 방법론의 성능 향상에 기여할 수 있으며, 제안된 함수는 맞춤형으로 설계될 수 있습니다.

- **Performance Highlights**: 실제 데이터 세트를 바탕으로 한 실험 결과, 수정된 OCP 방법이 기존 방법들보다 더 나은 또는 경쟁력 있는 유효성과 효율성을 달성했음을 보여줍니다. 더욱이, 제안된 방법은 시계열 데이터 예측에서 불확실성 정량화를 효과적으로 통합할 수 있는 가능성을 제시합니다. 전체 코드는 GitHub에 등록되어 있어, 연구자들이 쉽게 접근할 수 있도록 제공됩니다.



### OptunaHub: A Platform for Black-Box Optimization (https://arxiv.org/abs/2510.02798)
Comments:
          Submitted to Journal of machine learning research

- **What's New**: OptunaHub는 연구 분야 간의 분산된 블랙박스 최적화(Black-box optimization) 방법과 벤치마크를 중앙 집중화하는 커뮤니티 플랫폼입니다. 이 플랫폼은 통합된 Python API, 기여자 패키지 레지스트리, 그리고 검색성을 증진시키기 위한 웹 인터페이스를 제공합니다. OptunaHub는 기여와 애플리케이션의 선순환을 촉진하는 것을 목표로 합니다.

- **Technical Details**: OptunaHub는 세 가지 주요 구성 요소로 이루어져 있습니다: OptunaHub Module(등록된 패키지를 로드하는 Python 라이브러리), OptunaHub Registry(패키지 레지스트리), 및 OptunaHub Web(등록된 패키지 정보를 집계하는 웹 인터페이스)입니다. load_module 기능을 통해 OptunaHub Registry에서 패키지를 로드할 수 있으며, 다양한 샘플러와 벤치마크의 호환성이 보장됩니다. OptunaHub는 또한 실험 지원 기능을 통해 등록된 패키지에 직접 혜택을 제공합니다.

- **Performance Highlights**: 현재 94개의 패키지가 등록되어 있으며, OptunaHub 패키지의 총 월간 다운로드 수는 100,000회를 초과했습니다. 또한, OptunaHub Web은 사용자 정의 검색성과 가시성을 높여, 각 패키지의 README.md에서 자동으로 생성된 개별 패키지 페이지를 통해 실현됩니다. 이로 인해 기여자들은 자신의 작업을 보다 넓은 청중에게 홍보할 수 있는 기회를 얻습니다.



### Optimal Rates for Generalization of Gradient Descent for Deep ReLU Classification (https://arxiv.org/abs/2510.02779)
Comments:
          Accepted at NeurIPS 2025. Camera-ready version to appear

- **What's New**: 이 연구는 깊은 ReLU 네트워크에서 기울기 하강법(Gradient Descent, GD)의 최적 일반화 성능을 보장할 수 있는 새로운 결과를 제시합니다. 여기서는 전통적인 방법보다 깊이에 대한 다항적 의존성(polynomial dependence)을 유지하면서 GD의 최적 일반화 속도를 증명하였습니다. 기존 연구들이 네트워크 깊이에 대한 지수적 의존성을 가지거나 최적화의 하한을 초과하였던 것과 대비됩니다.

- **Technical Details**: 연구자들은 NTK 분리 가능성(NTK separable data) 가정을 바탕으로, 데이터의 마진(γ)에 따라 기울기 하강법을 통해 얻는 과잉 위험률(excess risk rate)을 도출합니다. 새로운 Rademacher 복잡도(Rademacher complexity) 경계를 통해 깊은 ReLU 네트워크의 개선된 성능을 제시하며, 모델 초기화에 대한 Lipschitz 연속성(Lipschitz continuity)을 보장하는 결과도 포함되어 있습니다.

- **Performance Highlights**: 기울기 하강법의 성능은 O~(L4(1 + γL2)/(nγ2))로 나타나며, 이는 기존의 SVM 타입 최적 속도와 거의 동등합니다. 연구는 또한 기존의 방법들과 비교하여 네트워크 깊이와 폭에 관한 의존성을 감소시키면서 데이터의 효과적인 처리 방안을 제시합니다. 마지막으로, 실험 결과들은 이론적 주장들이 실제 네트워크에서 어떻게 작용하는지를 보여줍니다.



### A Granular Study of Safety Pretraining under Model Abliteration (https://arxiv.org/abs/2510.02768)
Comments:
          Accepted at NeurIPS 2025 bWorkshop Lock-LLM. *Equal Contribution

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)의 사용이 증가함에 따라, 안전성을 위한 개입 방법들이 이러한 모델 수정에서 얼마나 견고하게 유지되는지를 조사하고 있습니다. 특히, 인퍼런스 타임(inference time)에서 모델을 변경할 수 있는 방법인 모델 압축(model abliteration)의 효과를 분석합니다. 연구는 SmolLM2-1.7B와 같은 안전 사전학습(Safety Pretraining)의 체크포인트를 활용하여, 안전성을 유지하는 데이터를 평가하고 그 결과를 기반으로 실제적인 프로토콜을 제안합니다.

- **Technical Details**: 연구에서는 강력한 안전성을 위한 데이터 중심 개입 방법을 검토하며, 저자들은 20개의 모델(원본 및 압축된 버전)에 대해 100개의 프롬프트(50 해로운 + 50 비해로운)를 통해 안전성의 견고성을 평가합니다. 더불어 자동화된 판단에서 발생할 수 있는 오류를 줄이기 위해 일부 프롬프트에 대해 인간의 주석을 추가하였으며, LLM 기반의 판단도 포함하고 있습니다. 연구 결과, 거부(Refusal) 전용 개입 방법들이 압축에 가장 취약하다는 것을 발견하였습니다.

- **Performance Highlights**: 본 연구는 다양한 안전 사전학습 체크포인트 및 공개 오픈 가중치 모델과의 비교를 통해 데이터 중심 안전 개입의 효율성을 평가하였으며, 각 체크포인트에서 안전성을 보장하는 요소를 분리하여 설명합니다. 연구 결과, 일부 안전 데이터 필터링 기법이 부분적인 견고성을 제공하는 것으로 나타났습니다. 또한, 모델이 자신의 출력을 거부하는 것을 인식하는 능력을 평가함으로써, 실제 배치 시스템에서의 자기 모니터링의 한계를 명확히 하고 있습니다.



### Curl Descent: Non-Gradient Learning Dynamics with Sign-Diverse Plasticity (https://arxiv.org/abs/2510.02765)
- **What's New**: 이 논문은 생물학적 신경망이 학습 중 비슷한 gradient 기반 전략을 사용하는지에 대한 의문을 제기하며, 비슷한 gradient descent의 대안으로 생물학적으로 타당한 다양한 시냅스 가소성 규칙을 탐구합니다. 구체적으로, 학습 역학(dynamics)에는 기본적으로 non-gradient인 "curl" 성분이 포함될 수 있다는 가능성을 조사하고 있습니다. 이 연구는 이러한 curl term이 해의 안정성을 줄 수 있음을 발견했으며, 이는 신경망 설계와 학습 규칙의 다양성을 수용하는 중요한 통찰력을 제공합니다.

- **Technical Details**: 논문에서는 gradient descent의 전통적인 기준과 달리, 다양한 시냅스 가소성 규칙이 존재하는 신경망에서 학습 역학이 어떻게 달라지는지를 분석합니다. 특히, synapses의 업데이트 방식이 적절한 방향의 gradient 정보와 대칭을 이룰 수 없음을 강조합니다. 연구팀은 무작위 행렬 이론을 활용하여 feedforward(neural) network의 학습 역학에서 phase transition이 발생하는 조건을 수학적으로 규명하였습니다.

- **Performance Highlights**: curl descent 학습 규칙은 네트워크 구조에 따라 매우 다양한 동작을 보여, 때로는 gradient descent보다 더 빠른 학습을 가능하게 할 수 있습니다. 다양한 구조적 매개변수에 따라 안정성 상실이 발생할 수 있으며, 이 경우 학습의 혼란스러운 성향을 초래하거나 성능을 저하 시킬 수 있습니다. 이러한 새로운 발견들은 생물학적 신경망의 학습 방법을 심층적으로 이해하고, 기존의 gradient 기반 학습 이론과의 차별성을 부각시킵니다.



### Fusing Multi- and Hyperspectral Satellite Data for Harmful Algal Bloom Monitoring with Self-Supervised and Hierarchical Deep Learning (https://arxiv.org/abs/2510.02763)
- **What's New**: 새로운 연구에서는 다중 센서 위성 데이터를 활용한 해로운 조류 번식(HAB) 감지 및 매핑을 위한 자기 감독 기계 학습 프레임워크를 제안합니다. 이 프레임워크는 다양한 위성의 반사율 데이터(VIIRS, MODIS, Sentinel-3, PACE)와 TROPOMI 태양 유도 형광(SIF) 데이터를 융합하여 HAB의 심각도와 종 분류를 생성합니다.

- **Technical Details**: 제안된 프레임워크(SIT-FUSE)는 자기 감독 표현 학습(self-supervised representation learning)과 계층적 딥 클러스터링(hierarchical deep clustering)을 사용하여 해양 식물 플랑크톤의 농도 및 종을 해석 가능한 클래스로 분할합니다. 이 방법은 2018-2025년 동안 멕시코만과 남부 캘리포니아에서 수집된 현장 데이터에 대해 검증되었습니다.

- **Performance Highlights**: 결과는 전체 식물성 플랑크톤 및 주요 조류인 Karenia brevis, Alexandrium spp., Pseudo-nitzschia spp.의 측정값과 강한 일치를 보였습니다. 이 연구는 레이블이 부족한 환경에서도 HAB 모니터링의 확장성을 높이며, 계층적 임베딩을 통해 탐색적 분석을 가능하게 하는 중요한 단계입니다.



### TokenFlow: Responsive LLM Text Streaming Serving under Request Burst via Preemptive Scheduling (https://arxiv.org/abs/2510.02758)
Comments:
          Accepted by EuroSys 2026

- **What's New**: 본 논문에서는 실시간 LLM 상호작용을 위한 새로운 시스템 TokenFlow를 소개합니다. TokenFlow는 요청 스케줄링과 KV 캐시 관리 방식을 개선하여 텍스트 스트리밍 성능을 극대화합니다. 이 시스템은 사용자 소비 속도에 맞춰 요청의 우선 순위를 동적으로 조정하고, GPU와 CPU 간의 캐시 전송을 통해 메모리 오버헤드를 최소화합니다.

- **Technical Details**: TokenFlow는 LLM의 요청 처리 성능의 향상을 위해 두 가지 주요 기술을 도입합니다: 1) 버퍼 인식 스케줄링, 2) 계층적 메모리 관리. 버퍼 인식 스케줄링은 실시간 토큰 버퍼 상태에 따라 요청을 자동으로 조정하여 사용자 경험을 방해하지 않으면서 자원을 효율적으로 할당합니다. 계층적 메모리 관리는 KV 캐시의 전송을 최소화하고, 스케줄링 필요에 맞춰 메모리를 동적으로 관리합니다.

- **Performance Highlights**: TokenFlow는 다양한 GPU 환경에서 테스트한 결과 실효 처리량이 최대 82.5% 향상되었고, 첫 번째 토큰을 생성하는 데 걸리는 시간이 80.2% 감소했습니다. 이러한 성능 향상은 복잡한 요청 패턴과 하드웨어 구성에서도 일관된 사용자 경험을 제공합니다. 최종적으로 TokenFlow는 LLM 기반 애플리케이션의 효율성을 높이는 강력한 솔루션으로 자리잡고 있습니다.



### Hybrid-Collaborative Augmentation and Contrastive Sample Adaptive-Differential Awareness for Robust Attributed Graph Clustering (https://arxiv.org/abs/2510.02731)
- **What's New**: 이번 연구에서는 Robust Attributed Graph Clustering (RAGC)라는 새로운 방법을 제안하여 기존의 CAGC 방법론의 한계를 극복하고자 합니다. RAGC는 하이브리드 협동 증강(Hybrid-Collaborative Augmentation, HCA) 및 대조 샘플 적응 차별 인식(Contrastive Sample Adaptive-Differential Awareness, CSADA) 모듈을 결합하여 노드 및 엣지 수준의 임베딩 표현과 증강을 동시에 수행합니다. 이 접근법은 더 포괄적인 유사성 측정을 가능하게 하여, 대조 학습의 성능 향상을 도모합니다.

- **Technical Details**: 제안된 RAGC에서는 각 노드의 속성(attribute) 및 엣지(edge) 간의 관계를 나타내는 attributed graph를 기반으로 하여 학습을 진행합니다. 전체 프레임워크는 HCA 모듈과 CSADA 모듈로 구성되며, HCA 모듈은 노드와 엣지의 임베딩 표현을 동시에 증강하여 더욱 신뢰성 있는 대조 유사성을 제공합니다. CSADA 모듈은 높은 신뢰도를 갖는 비공식 레이블(pseudo-label) 정보를 활용하여 모든 대조 샘플을 식별하고 차별적으로 처리하는 데 초점을 맞춥니다.

- **Performance Highlights**: 여섯 개의 벤치마크 데이터세트에서 포괄적인 그래프 클러스터링 평가 결과, 제안된 RAGC는 기존의 최첨단 CAGC 방법들과 비교할 때 우수한 성능을 보였습니다. 특히, CSADA 전략의 강력한 확장성이 다른 CAGC 방법을 향상시키는 데 기여하는 것을 입증하였습니다. 이 연구는 모든 긍정 및 부정 샘플을 더욱 깊이 인식하여 자기 감독 학습의 차별성을 높이는 데 중요한 기여를 하고 있습니다.



### Dale meets Langevin: A Multiplicative Denoising Diffusion Mod (https://arxiv.org/abs/2510.02730)
- **What's New**: 이번 논문에서는 생물학적 시스템에서의 학습과 최적화와 일치하는 방식으로 인공 신경망(ANN)을 훈련시키기 위한 새로운 접근 방식을 제안합니다. 특히, Dale의 법칙(Dale's law)에 의해 영감을 받은 지수 경량 최적화(exponentiated gradient optimization) 기법을 통해 로그-정규 분포(log-normal distribution)로 분포된 시냅스 가중치를 생성할 수 있음을 보여줍니다. 이러한 접근은 기하학적 브라운 운동(geometric Brownian motion)을 이용한 확률적 미분 방정식(stochastic differential equations, SDEs)과 연결되어 있으며, 이는 생물학적 기반의 생성 모델을 개발하는 데 적합합니다.

- **Technical Details**: Dale의 법칙에 따르면, 특정 시냅스를 통해 신경세포는 항상 흥분성 또는 억제성으로만 작용하며 학습 과정에서 그 역할을 전환하지 않습니다. 본 논문에서는 SDE의 역시간(reverse-time) 접근 방식을 통해 이러한 경량 최적화 방식이 곱셈 업데이트(multiplicative update) 규칙과 일치한다고 설명합니다. 이는 기존의 Brownian motion 기반의 방법과는 차별화된 신경망 훈련 방식을 제안하며, 새로운 곱셈 정합(score-matching) 손실 함수를 도입함으로써 이미지 생성 이미지와 같은 데이터에 적용 가능합니다.

- **Performance Highlights**: MNIST, Fashion MNIST, Kuzushiji 데이터셋에서의 실험 결과, 제안된 업데이트 방식이 생성 모델로서의 유능성을 입증했습니다. 논문에서 제안된 곱셈 경량 최적화 방식은 기존 생성 모델과 비교했을 때, 생물학적 영감을 받은 첫 번째 사례로 자리매김하며, 이미지 데이터 생성에 있어 유망한 성과를 거두었습니다. 이러한 연구는 딥러닝 및 생성 모델 분야에 있어 새로운 지평을 여는 기초가 될 것으로 기대됩니다.



### Accuracy Law for the Future of Deep Time Series Forecasting (https://arxiv.org/abs/2510.02729)
- **What's New**: 최근 심층 시계열 예측이 많은 관심을 받고 있으나, 연구자들은 분명한 연구 목표가 부족하여 혼란스러운 경우가 많습니다. 이 논문에서는 시계열 예측의 본질적인 오류 하한 설정의 중요성을 강조하고, 예측 성능의 상한을 추정하는 방법을 제안합니다. "Accuracy law"라는 새로운 법칙을 통해 심층 모델의 최소 예측 오류와 시계열 패턴 복잡성 간의 관계를 규명했습니다.

- **Technical Details**: 이 논문에서는 2,800개 이상의 심층 예측 모델에 대한 엄격한 통계 테스트를 기반으로 예측의 상한선을 정립했습니다. 전통적인 예측 지표의 한계를 초월하여, 시퀀스-투-시퀀스 예측 패러다임에 맞춰 창(window) 단위 속성을 고려한 연구를 진행했습니다. 이 과정에서 기존의 통계 방법뿐만 아니라 새로운 데이터 기반 방법론도 활용하였습니다.

- **Performance Highlights**: 제안된 'accuracy law'는 특정 데이터 세트에서의 과도한 연구 결과를 벗어나 연구자가 목표를 효과적으로 설정할 수 있도록 돕습니다. 이는 최근의 대규모 심층 모델 학습 전략을 제시하며, 다양하고 복잡한 forecasting 태스크의 상한선을 설정하여 향후 연구에 많은 기여를 할 것으로 기대됩니다. 향후 연구자들이 새로운 예측 목표를 설정하고 더욱 높은 성과를 이루는 데에 유용한 기초 자료로 작용할 것입니다.



### Hyperparameter Loss Surfaces Are Simple Near their Optima (https://arxiv.org/abs/2510.02721)
Comments:
          Accepted to COLM 2025. 23 pages, 8 figures

- **What's New**: 이번 연구에서는 하이퍼파라미터(hyperparameters) 손실 표면(loss surface)을 이해하기 위한 새로운 이론을 제안합니다. 이 연구는 복잡한 손실 표면을 탐색하는 대신, 최적 해에 접근함에 따라 단순한 구조가 나타난다는 점을 발견했습니다. 우리의 이론은 무작위 탐색(random search) 기반의 새로운 기술을 사용하여 이러한 구성 요소를 밝히는 데 기여합니다.

- **Technical Details**: 제안된 이론은 최적의 하이퍼파라미터 근처에서 손실 표면이 대략적으로 이차 다항식(quadratic polynomial) 형태로 나타나며, 노이즈(noise)는 정상(distributed normally) 분포를 따른다는 주장을 포함합니다. 이 이론을 통해 무작위 탐색의 실패 지점(threshold) 및 그로 인해 형성되는 비율 볼륨(parameter)과 같은 여러 속성들을 추론할 수 있습니다. 또한 이론은 1024개의 모델을 훈련하여 세 가지 실제 시나리오에서 검증되었습니다.

- **Performance Highlights**: 이론의 검증 결과는 랜덤 서치(random search)로 얻은 것이 실험적으로 측정된 데이터와 밀접하게 일치함을 보여줍니다. 각 실험에서 비대칭 영역은 전체 탐색 공간의 1/3에서 1/2를 차지했으며, 노이즈가 정상 분포로 수렴하는 진행 과정을 관찰했습니다. 이 작업은 하이퍼파라미터 손실 표면의 특성을 이해하고 무작위 탐색의 수렴 과정을 예측할 수 있게 하여, 연구자들이 불확실성을 정량화하면서 이러한 도구를 활용할 수 있도록 합니다.



### CST-AFNet: A dual attention-based deep learning framework for intrusion detection in IoT networks (https://arxiv.org/abs/2510.02717)
Comments:
          9 pages, 9 figures, 5 tables

- **What's New**: 이 논문은 IoT(Internet of Things) 네트워크에서의 강인한 침입 탐지를 위해 설계된 CST AFNet이라는 새로운 이중 주의 기반의 딥러닝 프레임워크를 소개합니다. 기존의 방법들이 갖고 있던 복잡한 사이버보안 문제를 해결하기 위해 다중 스케일 Convolutional Neural Networks (CNNs)과 Bidirectional Gated Recurrent Units (BiGRUs)를 통합한 모델입니다.

- **Technical Details**: CST AFNet은 공간적 특성 추출을 위해 CNN을 사용하고, 시간적 의존성을 포착하기 위해 BiGRUs를 활용하는 구조입니다. 또한, 이중 주의 메커니즘인 채널 주의(channel attention)와 시간 주의(temporal attention)를 날려 중요한 패턴에 집중합니다. 이 모델은 Edge IIoTset 데이터셋을 통해 훈련되고 평가되었습니다.

- **Performance Highlights**: CST AFNet은 15가지 공격 유형 및 정상 트래픽에 대해 99.97%의 뛰어난 정확도를 기록하였습니다. 또한, 매크로 평균 정밀도, 재현율 및 F1 점수가 모두 99.3%를 초과하는 Exceptional performance를 보여주었습니다. 실험 결과는 CST AFNet이 기존 딥러닝 모델에 비해 현저히 우수한 탐지 정확도를 달성했음을 확인시켜주었습니다.



### A Novel Unified Lightweight Temporal-Spatial Transformer Approach for Intrusion Detection in Drone Networks (https://arxiv.org/abs/2510.02711)
Comments:
          21 pages, 18 figures, 5 tables

- **What's New**: 드론 상업 및 산업 분야에 대한 통합이 증가함에 따라 사이버 보안 문제도 커지고 있습니다. 본 논문에서는 드론 네트워크에 특화된 TSLT-Net이라는 새로운 경량의 통합형 Temporal Spatial Transformer 기반 침입 탐지 시스템을 제안합니다. TSLT-Net은 셀프 어텐션 기법을 통해 네트워크 트래픽의 시간적 패턴과 공간적 의존성을 효과적으로 모델링하여 다양한 침입 유형을 정확하게 탐지합니다.

- **Technical Details**: TSLT-Net은 드론 네트워크의 동적이고 자원이 제한된 환경을 고려하여 설계되었습니다. 이 프레임워크는 간소화된 전처리 파이프라인을 포함하며, 단일 아키텍처 내에서 다중 클래스 공격 분류와 이진 이상 탐지를 모두 지원합니다. 또한, TSLT-Net은 0.04 MB의 최소 메모리 사용량과 9722개의 조정 가능한 파라미터를 유지합니다.

- **Performance Highlights**: ISOT Drone Anomaly Detection Dataset에서의 광범위한 실험 결과, TSLT-Net은 다중 클래스 탐지에서 99.99%의 정확도를 기록하고, 이진 이상 탐지에서는 100%의 정확도를 달성했습니다. 이 성능 결과는 TSLT-Net이 실시간 드론 사이버 보안을 위한 효과적이고 확장 가능한 솔루션임을 입증하며, 특히 미션 크리티컬 UAV 시스템의 엣지 장치에 적합합니다.



### RAMAC: Multimodal Risk-Aware Offline Reinforcement Learning and the Role of Behavior Regularization (https://arxiv.org/abs/2510.02695)
Comments:
          Under review as a conference paper at ICLR 2026, 21 pages, 8 figures. The HTML preview may misrender some figures; please refer to the PDF

- **What's New**: RAMAC (Risk-Aware Multimodal Actor-Critic) 프레임워크는 복합 목표를 결합하여 표현력이 뛰어난 생성자(actor)와 분포 기반 비평가(critic)를 결합한 점이 새롭습니다. 기존 리스크 회피 알고리즘들이 안전성을 보장하기 위해 보수적이거나 제한된 정책 클래스에 의존했던 반면, RAMAC는 높은 표현성을 유지하며 리스크를 조정하는 데 초점을 맞추고 있습니다. 이를 통해 분포적 위험(distributional risk)을 고려한 고급 정책 학습이 가능해졌습니다.

- **Technical Details**: RAMAC는 행동 복제(behavioral cloning)와 조건부 가치-at-위험(Conditional Value-at-Risk, CVaR) 경량 시 계산을 통해 리스크 감지 학습을 수행합니다. 이 프레임워크에서는 선행 작업에서 보인 데이터 외(ood) 행동 방문을 줄이는 동시에, 복잡한 다중 모달 환경에서의 높은 기대 수익을 목표로 합니다. 특히, RAMAC는 확산(diffusion) 및 흐름-매칭(flow-matching) 생성자를 사용하여 더 나은 성능을 달성하고 있습니다.

- **Performance Highlights**: 실험 결과, RAMAC의 두 가지 구현(RADAC와 RAFMAC)은 Stochastic-D4RL 벤치마크에서 기준 모델들에 비해 CVaR 성능이 우수하면서도 대부분의 작업에서 경쟁력 있는 평균 수익을 유지합니다. 이에 따라 RAMAC 프레임워크는 안전성을 유지하면서도 표현력이 뛰어난 정책 학습을 가능케 한다는 점에서 중요한 기여를 하고 있습니다. 이러한 결과는 리스크 인식 오프라인 RL 부문에서 RAMAC의 잠재력이 여전히 많이 남아있음을 시사합니다.



### Fine-Tuning Diffusion Models via Intermediate Distribution Shaping (https://arxiv.org/abs/2510.02692)
- **What's New**: 본 논문은 Diffusion 모델에서의 정책 기울기 방법(Policy Gradient Method)에 관한 새로운 접근 방식을 제안합니다. 저자들은 GRAFT라는 새로운 프레임워크를 통해 보상을 재구성하여 정책 기울기 최적화(Proximal Policy Optimization, PPO)를 포함하는 fine-tuning을 가능하게 합니다. 더불어, 이를 통해 중간 노이즈 레벨에서 분포를 형성하고, 보다 효과적인 fine-tuning을 수행하는 P-GRAFT를 도입합니다.

- **Technical Details**: P-GRAFT는 최종 생성물의 보상을 부분적으로 노이즈가 있는 생성물에 할당하여 중간의 denoising 단계까지만 fine-tuning하도록 설계되었습니다. 이 과정은 bias-변동 트레이드오프를 통해 수학적으로 설명되며, 다양한 생성 작업에서 질적 향상을 보입니다. 또한, inverse noise correction이라는 새로운 방법을 제안하여 명시적 보상 없이 흐름 모델을 개선할 수 있습니다.

- **Performance Highlights**: 제안된 GRAFT 프레임워크는 Stable Diffusion 2와 함께 사용되었으며, 이는 정책 기울기 방법에 비해 VQAScore에서 $8.81\%$의 상대적 개선을 달성했습니다. 또한, unconditional 이미지 생성 시 inverse noise correction으로 인해 생성된 이미지의 FID가 낮은 FLOPs/image에서 개선되었습니다. 전체적으로, 본 연구는 텍스트-이미지 생성, 레이아웃 생성 및 분자 생성 작업에서 중요한 품질 향상을 보여줍니다.



### EvoSpeak: Large Language Models for Interpretable Genetic Programming-Evolved Heuristics (https://arxiv.org/abs/2510.02686)
- **What's New**: EvoSpeak는 Genetic Programming (GP)과 Large Language Models (LLMs)를 통합한 새로운 프레임워크로, 복잡한 최적화 문제에 대한 휴리스틱(effective heuristics) 진화를 향상시키기 위해 개발되었습니다. 이 시스템은 GP에서 진화한 고품질의 휴리스틱을 학습하고, 지식을 추출하여 빠른 수렴을 위한 웜스타트(populations)를 생성하며, 결정 논리를 사람들에게 이해하기 쉬운 자연어 형식으로 변환합니다. 이러한 접근 방식은 휴리스틱의 효율성, 투명성, 그리고 적응성을 개선하는데 도움을 줍니다.

- **Technical Details**: EvoSpeak는 여러 문제의 상황에 대해 GP에서 진화한 복잡한 트리를 자연어 설명으로 번역함으로써 해석 가능성을 높이고 있습니다. 이 시스템은 특히 scheduling과 같은 결정적 영역에 초점을 맞추고, 고 품질의 초기 군집을 생성하여 진화 과정을 가속화하며, 문제마다 서로 다른 힐레틱들을 이용해 우수한 heuristics를 선별하는 역할을 합니다. 또한, LLMs는 공통된 지식을 다른 과제로 전이하여 강력한 의사 결정을 지원합니다.

- **Performance Highlights**: EvoSpeak를 이용한 실험 결과는 DFJSS(Dynamic Flexible Job Shop Scheduling) 문제에서 다양한 목표를 설정했을 때, 더 효과적인 휴리스틱을 생성하며 진화의 효율성을 높였습니다. 결과적으로 EvoSpeak는 해석 가능한 보고서를 통해 사용성을 개선하고, GP 진화의 불투명성 문제를 해결하여 더 많은 신뢰를 생성할 수 있음을 보여주었습니다. 이는 GP의 상징적 추론 능력과 LLM의 해석 및 생성 능력을 결합하여, 실제 최적화 문제를 위한 지능적이고 투명한 휴리스틱 개발을 한 단계 발전시켰습니다.



### Can Data-Driven Dynamics Reveal Hidden Physics? There Is A Need for Interpretable Neural Operators (https://arxiv.org/abs/2510.02683)
- **What's New**: 최근 신경 연산자(neural operators)는 함수 공간 사이의 매핑을 학습하는 강력한 도구로 떠올랐으며, 이를 통해 복잡한 역학의 데이터 기반 시뮬레이션이 가능해졌습니다. 본 논문에서는 신경 연산자를 공간(domain) 모델과 함수(functional) 모델의 두 가지 유형으로 분류하고, 물리적 원칙을 따르는 데이터 기반 동역학을 학습하는 방식에 대해 설명합니다. 또한, 신경 연산자가 데이터로부터 숨겨진 물리적 패턴을 학습할 수 있다는 점을 강조하지만, 이 설명 방법이 특정 상황에만 제한된다는 점도 지적합니다.

- **Technical Details**: 신경 연산자는 공간 모델과 함수 모델로 나눌 수 있습니다. 공간 모델은 그리드 기반 표현을 사용해 학습하고, 함수 모델은 함수 기초를 통해 학습합니다. 본 연구는 이러한 분류에 근거하여 다양한 관점을 제시하며, 특히 물리적 원칙을 준수하는 데이터 기반 동역학의 학습을 중점적으로 다룹니다. 또한, 신경 연산자가 학습하는 방식의 해석 가능성에 대한 기초적인 질문이 여전히 남아있음을 강조합니다.

- **Performance Highlights**: 본 논문에서는 간단한 이중 공간 다중 스케일 모델이 최신 성능을 달성할 수 있음을 보여주며, 이 모델이 복잡한 물리적 현상을 학습하는 데 큰 잠재력을 지니고 있음을 주장합니다. 다양한 신경 연산자 아키텍처가 제안된 바 있으며, 이들은 기후 모델링 및 유체 역학 등 다양한 분야에서 활용되고 있습니다. 연구의 결과는 과학적 응용을 위한 보다 효과적인 아키텍처 개발을 유도할 수 있으며, 숨겨진 물리적 현상을 더 많이 밝혀내기 위해 더 많은 노력이 필요하다는 점을 강조하고 있습니다.



### To Compress or Not? Pushing the Frontier of Lossless GenAI Model Weights Compression with Exponent Concentration (https://arxiv.org/abs/2510.02676)
- **What's New**: 본 논문은 수십억 개의 파라미터를 가진 Generative AI (GenAI) 모델의 효율적인 배치를 위해 저정밀(low-precision) 계산이 필수적이라는 점을 강조합니다. 저정밀 부동 소수점 형식 개발의 필요성을 논의하고, 이러한 형식이 수치적 안정성(numerical stability), 메모리 절약(memory savings), 하드웨어 효율성을 제공할 수 있음을 제시합니다. 특히, Exponent-Concentrated FP8 (ECF8)이라는 새로운 손실 없는 압축 프레임워크를 제안하며 실험 결과를 통해 최대 26.9%의 메모리 절약과 177.1%의 처리량 증가를 입증하였습니다.

- **Technical Details**: 부동 소수점 숫자는 IEEE-754 형식에서 부호 비트(sign bit), 지수(exponent), 그리고 가수(mantissa)로 구성됩니다. 저정밀 컴퓨팅에서는 지수가 표현 가능한 값의 동적 범위를 결정합니다. 본 논문에서 저자들은 GenAI 모델의 가중치에서 발생하는 지수 집중 현상(exponent concentration)을 이론적으로 및 경험적으로 분석하며, 이는 학습된 모델에서 지수가 저 엔트로피(low entropy)를 보이고 좁은 범위에 군집하는 현상으로 나타납니다. 이를 통해 FP4.67에 가까운 압축 한계를 증명하고 효과적인 FP8 포맷을 개발하게 됩니다.

- **Performance Highlights**: 제안된 ECF8 프레임워크는 다양한 GenAI 모델에서 최대 671B 파라미터를 지원하며, 메모리 사용량을 최대 26.9% 줄이고 처리량을 177.1% 가속화 함으로써 손실 없는 계산을 보장합니다. 이러한 결과는 ECF8가 메모리 압축을 추론 가속으로 변환할 수 있는 가능성을 입증하며, 학습된 모델의 통계적 법칙으로서 지수 집중 현상을 확립합니다. 이는 FP8 시대에서 손실 없는 저정밀 부동 소수점 설계를 위한 새로운 방향성을 제시합니다.



### Topological Invariance and Breakdown in Learning (https://arxiv.org/abs/2510.02670)
- **What's New**: 본 연구는 SGD, Adam 등 다양한 permutation-equivariant learning rules에 대해, 훈련 과정이 뉴런 간의 bi-Lipschitz mapping을 유도하고 훈련 중 뉴런 분포의 토폴로지를 강하게 제약함을 입증합니다. 이 결과는 학습률(learning rate)의 크기에 따라 작은 학습률과 큰 학습률의 질적 차이를 드러냅니다.

- **Technical Details**: 특히 임계점 $ eta^*$ 이하의 학습률로 훈련할 경우, 뉴런의 모든 토폴로지적 구조가 보존되도록 제약됩니다. 반면, $ eta^*$를 초과하면 학습 과정에서 토폴로지적 단순화가 허용되어, 뉴런 매니폴드(neuron manifold)가 점진적으로 더 굵어지고 모델의 표현력이 감소합니다.

- **Performance Highlights**: 본 이론은 특정 아키텍처나 손실 함수(loss function)에 독립적이어서, 위상(multiple phase)을 통해 심층 학습(deep learning) 연구에 적용 가능한 보편적인 방법론을 제시합니다. 학습 동역학(learning dynamics)은 먼저 토폴로지적 제약 하에서 부드러운 최적화(smooth optimization)를 거친 후, 급격한 토폴로지적 단순화를 통해 학습하게 됩니다.



### TutorBench: A Benchmark To Assess Tutoring Capabilities Of Large Language Models (https://arxiv.org/abs/2510.02663)
- **What's New**: TutorBench는 학생들이 학습 보조 자료로 대규모 언어 모델(LLMs)을 점점 더 많이 사용함에 따라, 이러한 모델들이 튜터링의 미묘한 부분을 효과적으로 처리할 수 있도록 개발된 데이터셋 및 평가 기준입니다. 이 데이터셋은 고등학교 및 AP 수준의 교육과정을 중심으로 한 1,490개의 샘플로 구성되어 있으며, 튜터링이 필요한 세 가지 일반적인 작업을 포함합니다. 이를 통해 LLMs의 핵심 튜터링 기술을 엄격하게 평가할 수 있도록 설계되었습니다.

- **Technical Details**: TutorBench는 적응형 설명 생성, 학생 작업에 대한 행동 피드백 제공, 효과적인 힌트 생성을 통한 능동적 학습 촉진 등 세 가지 튜터링 사용 사례에 중점을 둡니다. 각 샘플은 샘플별 평가 기준이 동반되어 있어 모델의 응답을 평가하는 데 사용됩니다. 또한, TutorBench는 LLM 판별을 활용한 신뢰성 있는 자동 평가 방법을 사용하여 보다 세분화된 평가가 가능하도록 구조화되어 있습니다.

- **Performance Highlights**: 16개의 최신 LLM 모델을 TutorBench에서 평가한 결과, 어떤 모델도 56% 이상의 점수를 기록하지 못했습니다. Claude 모델은 능동 학습 지원에서 가장 우수한 성과를 보였지만, 나머지 두 가지 사용 사례에서는 뒤처졌습니다. TutorBench의 출시는 차세대 AI 튜터 개발을 위한 포괄적이고 비포화된 기준을 제공하여 LLM의 향후 발전에 기여할 것으로 기대됩니다.



### Optimal Characteristics of Inspection Vehicle for Drive-by Bridge Inspection (https://arxiv.org/abs/2510.02658)
- **What's New**: 이번 연구는 다리 건강 모니터링을 위한 차량 점검 기술의 새로운 최적화 프레임워크를 제안합니다. 이는 차량과 다리의 반응을 분석하여 구조적 무결성과 손상을 평가하는 데 중점을 둡니다. 차량의 역학적 특성이 검사 성능에 결정적인 영향을 미친다는 점을 처음으로 다루고 있습니다.

- **Technical Details**: 연구에서는 적대적 오토인코더(adversarial autoencoders, AAE)를 기반으로 한 비지도 심층 학습 방법을 사용하여 가속도 반응의 주파수 도메인 표현을 재구성합니다. 또한, 건강한 다리 상태와 손상된 다리 상태의 손상 지수 분포 간의 Wasserstein 거리 를 최소화함으로써 이축 차량의 타이어 서스펜션 시스템의 질량과 강성을 최적화합니다. 이를 통해 Kriging 메타 모델을 활용해 이 목표 함수를 효율적으로 근사하고 최적의 차량 구성을 찾아냅니다.

- **Performance Highlights**: 연구 결과, 다리의 첫 번째 고유 진동수에 대해 주파수 비율이 0.3과 0.7 사이에 있는 차량이 가장 효과적임을 보여주었습니다. 공진(resonance) 근처의 차량은 성능이 저하됩니다. 또한, 더욱 가벼운 차량은 최적 감지를 위해 더 낮은 고유 진동수를 필요로 한다는 점이 밝혀졌습니다. 이는 차량 점검 기술을 위한 최초의 철저한 최적화 연구로 평가될 수 있습니다.



### HyperAdaLoRA: Accelerating LoRA Rank Allocation During Training via Hypernetworks without Sacrificing Performanc (https://arxiv.org/abs/2510.02630)
Comments:
          13 pages

- **What's New**: 본 논문에서는 HyperAdaLoRA라는 혁신적인 프레임워크를 제안하며, 이는 하이퍼네트워크(hypernetwork)를 활용하여 AdaLoRA의 수렴 속도를 크게 향상시킵니다. HyperAdaLoRA는 고정된 순위를 사용하지 않고 동적 순위 할당(dynamic rank allocation)을 도입하여 다양한 모듈 및 레이어의 중요도에 따라 가중치를 유연하게 조정합니다. 이 프레임워크는 파라미터 생성 시 최신 어텐션(attention) 메커니즘을 사용하여 전반적인 성능을 높입니다.

- **Technical Details**: HyperAdaLoRA는 하이퍼네트워크 기반으로 Singular Value Decomposition (SVD)의 구성요소인 P, Λ, Q를 직접 최적화하는 것이 아니라, 이를 동적으로 생성하여 파라미터 업데이트를 수행합니다. 이렇게 생성된 매트릭스의 출력은 트레이닝 과정에서 조정된 후, 고유한 중요도 기반으로 잘리지 않은 특이값을 통해 동적 순위 할당을 실현합니다. 이를 통해 트레이닝 중 발생하는 느린 수렴성과 높은 계산 부담 문제를 개선할 수 있습니다.

- **Performance Highlights**: 다양한 데이터셋과 모델에 대한 포괄적인 실험 결과, HyperAdaLoRA는 더 빠른 수렴 속도를 달성하면서도 성능 저하 없이 정확도를 유지하는 것으로 나타났습니다. 또한, 다른 LoRA 기반 방법들에 대한 실험을 통해 HyperAdaLoRA의 폭넓은 적용 가능성을 검증했습니다. 이러한 결과는 HyperAdaLoRA가 기존 방법들보다 효율적이고 강력하다는 것을 시사합니다.



### TabImpute: Accurate and Fast Zero-Shot Missing-Data Imputation with a Pre-Trained Transformer (https://arxiv.org/abs/2510.02625)
- **What's New**: 최근 자료에서 누락된 데이터 문제를 해결하기 위해 TabImpute라는 새로운 사전 훈련된 트랜스포머 모델을 제안합니다. 이 모델은 zero-shot 방식으로 빠르고 정확한 누락 데이터 보간을 가능하게 하며, 이 과정에서 fitting이나 hyperparameter tuning이 필요하지 않습니다.

- **Technical Details**: TabImpute는 Entry-Wise Featurization(EWF) 기술을 도입하여, 기존 TabPFN 모델에 비해 $100	imes$ 속도 향상을 달성하였습니다. 또한, 현실적인 결측 패턴을 포함한 합성 훈련 데이터 생성 파이프라인을 구축하여 테스트 성능을 향상시키고, 42개의 OpenML 데이터셋과 13개의 결측 패턴을 포함한 종합 벤치마크 MissBench를 소개합니다.

- **Performance Highlights**: TabImpute는 MissBench에서 수행한 평가에서 기존 11개 보간 방법과 비교하여 우수한 성능을 보여주었습니다. 특히, TabImpute의 예측 결과는 EWF-TabPFN의 결과와 결합되어, 다양한 데이터셋 전반에서 높은 정확도의 보간을 가능하게 하는 adaptive ensemble 방식으로 성과를 개선하였습니다.



### MINERVA: Mutual Information Neural Estimation for Supervised Feature Selection (https://arxiv.org/abs/2510.02610)
Comments:
          23 pages

- **What's New**: 기존의 feature 필터는 통계적 쌍-쌍 (pair-wise) 의존성 메트릭스에 기반하여 feature-target 관계를 모델링하지만, 이를 통해 타겟이 고차원 feature 상호작용에 의존하는 경우에는 실패할 수 있습니다. 본 논문에서는 feature와 타겟 간의 상호 정보를 신경망(neural network)을 기반으로 추정하는 새로운 supervised feature selection 방법인 Mutual Information Neural Estimation Regularized Vetting Algorithm(MINERVA)을 소개합니다. 이 방법은 두 단계로 구성된 프로세스를 통해서 representation learning과 feature selection을 분리하여 보다 나은 일반화와 feature 중요성 표현을 보장합니다.

- **Technical Details**: MINERVA는 신경망을 사용하여 상호 정보를 근사화하고, 희소성 유도 정규화자(sparsity-inducing regularizers)를 포함한 정교하게 설계된 손실 함수(loss function)를 사용하여 feature selection을 수행합니다. 이 방법은 고차원 데이터 세트에서의 변수 간의 복잡한 의존 구조를 학습하며, feature subset을 앙상블(ensemble)로 평가하여 효과적으로 이 관계를 포착합니다. 논문에서는 합성 및 실제 사기 데이터 세트에 대한 실험 결과를 통해 제안한 방법의 효율성을 입증합니다.

- **Performance Highlights**: MINERVA는 기존의 feature selection 방법들과 비교하여 더 나은 성능을 보였습니다. 실험 결과, MINERVA는 정확한 해결책을 제공할 수 있는 능력을 갖추고 있으며, 높은 차원의 경우에도 잘 작동하는 것으로 나타났습니다. 이러한 결과는 MINERVA가 신경망 기반의 상호 정보 추정 방식을 효과적으로 활용하고 있음을 보여줍니다.



### Towards CONUS-Wide ML-Augmented Conceptually-Interpretable Modeling of Catchment-Scale Precipitation-Storage-Runoff Dynamics (https://arxiv.org/abs/2510.02605)
Comments:
          Main text: 95 pages, 15 figures, 4 tables; Applendix: Section A-E; 2 figures; Supplementary Materials: 15 figures, 7 tables

- **What's New**: 본 연구는 다양한 하이드로-지오-기후 조건을 포함한 대규모 샘플 연구를 통해 물리적으로 해석 가능한 유역 규모 모델을 활용하여 예측 성능 향상을 도모하는 것을 목표로 합니다. 특히, Mass-Conserving Perceptron (MCP)을 기반으로 한 모델의 다양한 복잡성을 평가하였으며, 이는 기존의 머신러닝(ML) 기반 모델 연구와 차별화됩니다.

- **Technical Details**: 연구에서는 snow regime, forest cover, climate zone과 같은 속성 마스크(attribute masks)를 사용하여 결과를 평가하였으며, 개별 하이드롤로지(regime)에 따라 모델 복잡성을 적절히 선택하는 것이 중요하다는 점을 강조합니다. 또한, MCP 기반 모델이 Long Short-Term Memory (LSTM) 아키텍처 기반의 데이터 기반 모델과 비슷한 성능을 발휘할 수 있음을 벤치마크 비교를 통해 보여주었습니다.

- **Performance Highlights**: 이 연구는 이론적으로 뒷받침된 물리적 접근 방식이 대규모 수문학 연구에 기여할 수 있는 잠재력을 강조하며, 기계적 이해(mechanistic understanding)와 해석 가능한 모델 아키텍처 개발을 중시합니다. 이러한 연구 결과는 공간적으로 및 시간적으로 변화하는 프로세스 우위(process dominance)에 대한 정보를 아키텍처적으로 인코딩하는 미래 모델의 기초를 마련하는 데 도움을 줄 것입니다.



### Use the Online Network If You Can: Towards Fast and Stable Reinforcement Learning (https://arxiv.org/abs/2510.02590)
- **What's New**: 본 논문에서는 MINTO라는 새로운 업데이트 규칙을 소개하여 Target 네트워크와 Online 네트워크 간의 MINimum 값을 이용해 학습 속도를 높이면서 안정성을 유지하는 방법을 제안한다. 기존의 Target 네트워크는 업데이트가 늦어지기 때문에 학습 속도가 느려지는 한계가 있는데, MINTO는 이러한 단점을 극복하도록 설계되었다. 이 방식은 다양한 값 기반 및 액터-크리틱 알고리즘에 무리 없이 통합될 수 있다.

- **Technical Details**: MINTO는 Target 네트워크와 Online 네트워크 사이의 MINimum 추정값을 계산하여 회귀 타겟을 생성하는 기법이다. 이 방법은 maximization bias를 줄이고 moving-target 문제를 완화하여 안정적인 학습을 보장한다. 또한, 실제 환경에서 Online 네트워크가 상대적으로 낮은 추정값을 가질 때는 더 신선한 정보를 사용하여 학습 속도를 빠르게 할 수 있다.

- **Performance Highlights**: MINTO는 다양한 벤치마크 테스트에서 일관되게 성능을 향상시켰으며, 이는 이 방법의 광범위한 적용 가능성과 효과를 입증한다. 비교 실험 결과, MINTO는 기존의 Target 네트워크 설계 및 유사한 기법들에 비해 매우 우수한 성능을 보였다. 이로 인해 강화 학습의 안정성과 학습 속도를 동시에 개선할 수 있는 가능성을 보여준다.



### Geospatial Machine Learning Libraries (https://arxiv.org/abs/2510.02572)
Comments:
          Book chapter

- **What's New**: 최근 기계 학습(ML) 분야에서 특정 도메인 소프트웨어 라이브러리의 발전이 이루어졌습니다. 이는 GeoML(geospatial machine learning)의 고유한 문제에 대처하기 위한 라이브러리의 발전이 느리다는 점에서 주목할 만합니다. 이 챕터에서는 TorchGeo, eo-learn, Raster Vision과 같은 인기 있는 GeoML 라이브러리를 소개하고 그들의 아키텍처와 데이터 타입을 자세히 설명합니다.

- **Technical Details**: GeoML 라이브러리는 지리공간 데이터 처리 및 이를 기계 학습 워크플로우에 통합하는 두 가지 핵심 작업을 지원해야 합니다. 대부분의 GeoML 라이브러리는 GDAL(Geospatial Data Abstraction Library)에 의존하여 다양한 레스터 및 벡터 데이터 형식을 지원합니다. Python에서는 rasterio와 fiona가 GDAL의 API를 래핑하여 추가적인 추상화 계층과 기능을 제공합니다.

- **Performance Highlights**: 이 논문은 농작물 유형 매핑과 같은 사례 연구를 통해 GeoML 라이브러리의 실제 응용을 시연합니다. GeoML 생태계는 아직 파편화되어 있으며, 시간 모델링, 센서 융합, 대규모 데이터셋에 대한 분산 추론과 같은 작업에서 제한적인 지원을 받을 수 있습니다. 이러한 도전 과제를 해결하기 위한 최선의 방법 및 개방형 지리공간 소프트웨어의 거버넌스 필요성에 대해서도 논의합니다.



### On The Expressive Power of GNN Derivatives (https://arxiv.org/abs/2510.02565)
Comments:
          30 pages, 3 figures

- **What's New**: 최근 연구들은 Graph Neural Networks (GNNs)에서 표현력의 한계를 해결하기 위한 다양한 접근방법을 제안하였습니다. 본 논문에서는 GNN의 표현력을 향상시키기 위해 노드 특성에 대한 고차 미분(high-order derivatives)을 활용하는 새로운 방법인 High-Order Derivative GNN (HOD-GNN)을 소개합니다. 이 방법은 Message Passing Neural Networks (MPNNs)의 하위 구성 요소에 대한 미분을 추가로 입력하여 그 표현성을 높입니다.

- **Technical Details**: HOD-GNN은 세 가지 주요 컴포넌트로 구성됩니다: 기본 MPNN, 미분 인코더 네트워크, 그리고 다운스트림 GNN입니다. 고차 미분은 단일 노드의 특성에 대한 기본 MPNN의 미분을 계산하여 생성하며, 이 미분들은 새로운 노드 특성으로 인코딩되어 다운스트림 GNN에 전달됩니다. 이 모델은 이론적으로 기존의 GNN보다 더 높은 표현력을 가지며, 인기 있는 구조 인코딩 또한 계산할 수 있습니다.

- **Performance Highlights**: HOD-GNN은 다양한 그래프 학습 벤치마크에서 뛰어난 성능을 입증하며, 다른 표현력이 강한 GNN 아키텍처보다 더 큰 그래프에 대해 확장 가능한 성질을 보여줍니다. 또한, 그래프의 서브구조를 정확히 계산할 수 있는 능력을 보여줍니다. 논문은 일곱 개의 표준 그래프 분류 및 회귀 벤치마크에서 일관된 높은 성능을 달성하였음을 강조합니다.



### AttentiveGRUAE: An Attention-Based GRU Autoencoder for Temporal Clustering and Behavioral Characterization of Depression from Wearable Data (https://arxiv.org/abs/2510.02558)
Comments:
          4 pages, 3 figures, 2 tables, Accepted NeurIPS (TS4H Workshop) 2025, non-camera-ready version)

- **What's New**: 이번 연구에서는 AttentiveGRUAE라는 새로운 attention 기반의 gated recurrent unit (GRU) autoencoder를 소개합니다. 이 모델은 longitudinal wearable data를 사용하여 temporal clustering과 결과 예측을 동시에 수행하도록 설계되었습니다. AttentiveGRUAE는 일일 행동 특징의 compact latent representation을 학습하고, 기간 종료 시점의 우울증 비율을 예측하며, 행동 하위 유형을 식별하기 위해 Gaussian Mixture Model (GMM)을 기반으로 하는 soft clustering을 수행합니다.

- **Technical Details**: AttentiveGRUAE는 시간에 따른 행동 데이터를 해석 가능한 방식으로 클러스터링할 수 있는 프레임워크입니다. 모델은 입력 시퀀스를 재구성하고 이진 결과를 예측하는 학습을 통해 참가자 수준의 임베딩을 생성하며, 이 임베딩은 행동 하위 유형으로 클러스터링됩니다. 본 연구에서는 Adam optimizer를 사용하여 최적화를 수행하며, gradient surgery와 같은 기법을 통해 공동 학습 시 발생할 수 있는 그래디언트 충돌을 완화합니다.

- **Performance Highlights**: AttentiveGRUAE는 임상적 유의미성을 갖춘 결과 예측 및 클러스터 구조에서 뛰어난 성능을 보여줍니다. 노출 추정 및 클러스터 품질이 이전 모델들보다 우수하며, 외부 검증을 통해 클러스터의 재현성도 확인되었습니다. 특히, 모델은 COVID 이전 데이터로 학습하였음에도 불구하고 COVID 이후에도 상당한 성능을 유지하여, 잠재적인 하위 유형 구조를 잘 보존하고 있음을 입증했습니다.



### Model-brain comparison using inter-animal transforms (https://arxiv.org/abs/2510.02523)
Comments:
          16 pages, 8 figures. An extended and revised version of a 9-page paper to be published in the Proceedings of the 2025 Cognitive Computational Neuroscience conference

- **What's New**: 본 논문은 인공 신경망 모델을 실제 뇌 반응과 비교하기 위한 새로운 방법론인 Inter-Animal Transform Class (IATC)를 제안합니다. IATC는 동물 집단 내 신경 반응을 정확히 매핑하기 위해 필요한 함수의 집합을 정의하며, 모델이 일반적인 주체로 가장할 수 있는지 평가합니다. 이러한 기법을 통해 뇌 데이터와 모델 간의 양방향 비교가 가능하며, 보다 정교한 기계적 모델 개발이 촉진됩니다.

- **Technical Details**: 이 논문에서는 IATC에 대한 형식적인 정의를 제공하며, 모델이 뇌와 얼마나 잘 일치하는지를 평가하기 위한 새로운 메트릭을 제안합니다. IATC는 뇌 집단 내 다양한 주체들 간의 유사도를 평가하는 데 사용되며, 이를 통해 매핑 정확성과 엄격성을 모두 고려합니다. 특히, 이론적인 접근법 외에도 인공 신경망 모델과 뇌를 비교하기 위해 실증적으로 확인된 IATC를 활용하는 방안을 제시합니다.

- **Performance Highlights**: IATC는 신경 기계의 정밀한 면모를 드러내며, 특히 비선형 활성화 함수와 같은 중요한 특성을 이해하는 데 기여합니다. 모델과 뇌 간의 정확한 예측 가능성을 드러내며, 서로 다른 뇌 영역의 반응 패턴을 구분하는 데 효과적인 것으로 나타났습니다. 이 연구는 깊은 신경망 모델을 통한 예측 성공에 대한 이전 발견들을 재맥락화하며, IATC를 통해 모델-뇌 비교의 원리를 확립하는 데 중요한 귀찮은 상세를 높였습니다.



### Graph Generation with Spectral Geodesic Flow Matching (https://arxiv.org/abs/2510.02520)
- **What's New**: 이 논문은 그래프 생성을 위한 새로운 프레임워크인 Spectral Geodesic Flow Matching (SFMG)을 제안합니다. 기존의 방법들이 주로 스펙트럼이나 차수 프로파일에 국한되어 있는 반면, SFMG는 고유 벡터들이 유도하는 기하학적 구조와 그래프의 글로벌 구조를 모두 고려합니다. 또한 SFMG는 입력 그래프와 목표 그래프를 연속적인 리만 다양체(Riemannian manifold)로 임베딩하여 그래프 생성을 위한 새로운 접근 방식을 제공합니다.

- **Technical Details**: SFMG는 스펙트럼 기하학을 활용하여 그래프 생성을 위한 새로운 모델을 설계합니다. 이 모델은 고유 값과 고유 벡터의 분포를 Stiefel 다양체에서 직접 모델링함으로써, 그래프 구조에 대한 보다 완전하고 표현력 있는 표현을 가능하게 합니다. 또한, 이 방법은 흐름 매칭(flow matching)을 활용하여 높은 효율성으로 분포 학습을 향상시킵니다.

- **Performance Highlights**: SFMG는 다양한 그래프 데이터셋에서의 실험 결과, 최첨단 방법들과 비교했을 때 우수한 성능을 보였습니다. 특히, SFMG는 확산 모델(diffusion-based models) 대비 최대 30배 빠른 속도를 달성하며, 이는 대규모 그래프 처리에 있어 상당한 이점을 제공합니다. 결과적으로 SFMG는 다양한 그래프 스케일과 응용에서 기존 방법들을 초월하는 성능을 발휘합니다.



### In-memory Training on Analog Devices with Limited Conductance States via Multi-tile Residual Learning (https://arxiv.org/abs/2510.02516)
- **What's New**: 본 논문에서는 메모리 내에서 저조도 상태의 소자를 활용한 잔여 학습(framework)을 제안하여, 정확성이 저하되는 문제를 해결하고자 합니다. 제한된 비트 해상도(4비트)에서의 훈련을 가능하게 하여 일반적인 디지털 동등성과 일치할 수 있도록 하는 것이 본 연구의 주요 목표입니다. 저비용 하드웨어를 사용하면서도 이론 분석을 통해 수렴 성능을 검증하였습니다.

- **Technical Details**: 저자들은 여러 개의 교차바 크로스바 타일을 순차적으로 학습하는 방식을 사용하여, 낮은 해상도의 업데이트로 인한 잔여 오류를 보정합니다. 이런 방식은 메모리 내에서 아날로그 신경망 훈련을 위한 새로운 접근법을 제시하며, 전통적인 TTA 및 CLT와 같은 복잡한 회로 설계 없이도 높은 정확도의 훈련이 가능하다는 점을 강조합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다양한 이미지 분류 벤치마크에서 최신 메모리 내 아날로그 훈련 전략을 초월하는 성능을 보였습니다. 특히, 하드웨어 오버헤드가 최소화된 상태에서도 뛰어난 결과를 달성하며, 이는 이론적 분석과 실험적 검증 모두에서 뒷받침됩니다.



### Beyond Imitation: Recovering Dense Rewards from Demonstrations (https://arxiv.org/abs/2510.02493)
- **What's New**: 이번 연구에서는 기존의 감독 세밀 조정(Supervised Fine-Tuning, SFT)을 단순한 모방 학습 프로세스로 간주하는 시각에 도전합니다. 우리는 SFT와 역강화 학습(Inverse Reinforcement Learning, IRL) 사이의 근본적인 동치 관계를 수립하였습니다. SFT의 목표가 역 Q-학습(Inverse Q-Learning)의 특별한 경우임을 증명하여, SFT 프로세스가 단지 정책을 배우는 것이 아니라 전문가의 시연을 설명하는 암묵적이고 밀집된 토큰 수준 보상 모델도 함께 학습함을 보여줍니다.

- **Technical Details**: 이 연구에서는 유한 수명 토큰 MDP(다양한 상태 집합 𝒮, 행동 집합 𝒜, 결정적 전이 함수 f 및 초기 상태 분포 ρ0)가 포함된 환경을 다룹니다. 우리는 특정 상태 ss에서의 점유 측정(occupancy measure)과 로그 분배(log-partition) 및 볼츠만 정책(Boltzmann policy) 개념을 통해 이론적 기반을 제공합니다. 이를 통해 전문가의 시연으로부터 회복한 밀집 보상 신호를 통해 정치적 임무를 강화할 수 있는 새로운 가능성을 제안합니다.

- **Performance Highlights**: 우리는 회복한 보상을 사용하여 정책 성능을 더욱 개선할 수 있는 방법인 Dense-Path REINFORCE를 소개합니다. 이 방법은 명령 이행 벤치마크에서 원래 SFT 모델보다 일관되게 뛰어난 성능을 예시합니다. 따라서 이번 연구는 SFT를 단순히 정책 모방이 아닌 강력한 보상 학습 메커니즘으로 재구성하며, 전문가의 시연을 활용하여 새로운 가능성을 열어줍니다.



### Improved Robustness of Deep Reinforcement Learning for Control of Time-Varying Systems by Bounded Extremum Seeking (https://arxiv.org/abs/2510.02490)
- **What's New**: 이번 논문에서는 비모델 의존형 제한 극값 탐색(bounded extremum seeking, ES) 피드백 제어를 이용하여 비선형 시간 변동 시스템에 대한 심층 강화 학습(deep reinforcement learning, DRL) 제어기의 강인성을 향상시키는 것을 다룹니다. DRL은 데이터로부터 학습하여 많은 매개변수를 갖는 시스템의 출력을 효과적으로 제어하지만, 시스템 모델이 빠르게 변할 경우 성능이 급격히 저하됩니다. 제한된 ES는 알 수 없는 제어 방향을 가진 시간 변동 시스템을 다룰 수 있지만, 조정된 매개변수의 수가 늘어나면 수렴 속도가 느려질 수 있습니다.

- **Technical Details**: 논문은 DRL과 제한된 ES의 조합으로 새로운 하이브리드 제어기를 제안합니다. 이 하이브리드 모델은 DRL이 과거 데이터를 활용하여 빠른 제어 반응을 제공하는 한편, 제한된 ES가 시간 변동에 대한 강인성을 보장하도록 합니다. 이것은 파라미터 조정이 필요한 로스 알라모스 중성선 과학 센터의 저에너지 빔 수송 구간과 같은 시간 변동 시스템에 대한 자동 조정 연구를 포함합니다.

- **Performance Highlights**: 양질의 데이터 세트를 이용하여 DRL 정책을 훈련하고, 시스템이 빠르게 변화할 때도 강한 성능을 유지하는 데 성공합니다. 제한된 ES는 성능 저하를 방지하는 역할을 하고, 시스템이 느리게 변화할지라도 높은 강인성을 유지할 수 있도록 합니다. 논문에서는 일반적인 시간 변동 동적 시스템과 입자 가속기 응용에 대한 수치적 연구를 통해 이 결과를 보여줍니다.



### From Pixels to Factors: Learning Independently Controllable State Variables for Reinforcement Learning (https://arxiv.org/abs/2510.02484)
- **What's New**: 이 연구에서는 Action-Controllable Factorization (ACF)라는 대조적 학습 접근법을 통해 독립적으로 제어 가능한 잠재 변수를 식별하는 새로운 방법을 제안합니다. ACF는 에이전트가 고차원 관찰만을 통해 переход하는 상황에서 결정을 내리는 데 필요한 정보를 추출하는 데 중점을 두고 있습니다. 이 방법은 Taxi, FourRooms, MiniGrid-DoorKey와 같은 세 가지 기준 데이터셋에서 사실적인 제어 가능 요소를 자동으로 복구하여 기존의 변별화 알고리즘보다 더 나은 성능을 보여주었습니다.

- **Technical Details**: ACF는 에이전트의 행동에 의해 영향을 받는 상태 요소를 분리하여 그들 간의 상관관계를 정확하게 평가합니다. 대조적 학습의 목표를 통해, ACF는 상태 변수가 환경의 자연적 동역학에 대한 에이전트 행동의 다음 상태 예측에서 발생하는 차이를 통해 분리됩니다. 이러한 접근법은 에이전트가 관찰하는 픽셀 정보를 사용하여 제어 가능한 상태 변수를 정확히 식별할 수 있도록 합니다.

- **Performance Highlights**: ACF는 기존의 Disentanglement 알고리즘과 비교하여, 일반적인 RL 도메인에서 작업을 수행하면서 더욱 높은 정확도를 기록하였습니다. 실험 결과, ACF는 고차원의 픽셀 관찰에서 직접 제어 가능한 요소를 복구하는 데 성공하며 전문가가 설계한 표현과 일치하는 결과를 보여주었습니다. 이러한 성과는 ACF의 효과성을 입증하며, 에이전트의 학습 과정을 보다 샘플 효율적으로 만듭니다.



### Litespark Technical Report: High-Throughput, Energy-Efficient LLM Training Framework (https://arxiv.org/abs/2510.02483)
Comments:
          14 pages

- **What's New**: 본 논문에서는 Litespark라는 새로운 사전 훈련 프레임워크를 소개합니다. 이 프레임워크는 transformer architecture의 attention과 MLP layers에서 비효율성을 해결하여 훈련 시간을 단축하고 에너지 소비를 감소시키는 방법을 제안합니다. Litespark는 모델 성능을 극대화하면서도 기존 transformer 구현과의 호환성을 유지합니다.

- **Technical Details**: Litespark는 두 단계로 최적화를 수행합니다: 첫 번째는 architectural optimization으로 attention과 MLP 블록을 최적화하고, 두 번째는 algorithmic optimization으로 GPU당 FLOPs를 증가시키기 위해 순방향 및 역방향 연산을 최적화합니다. 이 프레임워크는 기존의 FlashAttention 기법과 같은 기술적 진보 위에 성능을 더할 수 있습니다.

- **Performance Highlights**: Litespark는 훈련 처리량을 2배에서 6배까지 증가시키고, 사전 훈련 과정에서 에너지 소비를 55%에서 83%까지 감소시킵니다. 이러한 최적화는 다양한 모델 아키텍처와 하드웨어에 적용 가능하여, 많은 산업계 응용에 유용할 것으로 기대됩니다.



### Uncertainty-Guided Model Selection for Tabular Foundation Models in Biomolecule Efficacy Prediction (https://arxiv.org/abs/2510.02476)
Comments:
          NeurIPS 2025 workshop: 2nd Workshop on Multi-modal Foundation Models and Large Language Models for Life Sciences

- **What's New**: 이번 연구에서는 TabPFN 모델을 활용하여 siRNA knockdown 효능 예측에서 단순한 서열 기반 특징으로도 전문화된 최신 모델보다 우수한 성과를 달성할 수 있음을 입증했습니다. 또한 모델의 예측된 inter-quantile 범위(IQR)는 실제 예측 오류와 음의 상관관계를 가진다는 사실을 보여주어 모델 불확실성이 생체 분자 효능 예측을 최적화하기 위한 강력한 수단임을 강조했습니다.

- **Technical Details**: 연구에서는 Huesken et al.의 데이터를 기반으로 siRNA의 특징을 19-mer로 표현하고, 각각의 siRNA와 mRNA에 대해 one-hot 벡터 및 리보뉴클레오타이드 트리머의 개수를 포함한 총 574개의 특징을 생성했습니다. TabPFN 모델은 이러한 특징을 사용하여 모델의 예측 불확실성을 기반으로 하여 최적의 모델을 선택하고 집합체를 형성하는 새로운 접근 방식을 제공하며, 특히 IQR을 활용한 메트릭을 제공합니다.

- **Performance Highlights**: TabPFN 모델은 Huesken 데이터셋에서 OligoFormer와 비교했을 때 더 낮은 평균 절대 오차(MAE)를 보여 주었으며, 새로운 타겟에 대해서도 적은 샘플 데이터를 사용하여 상당한 성능을 발휘했습니다. 더 나아가, 선택된 모델들의 평균 IQR을 기반으로 한 집합체를 구성하여 어림잡은 예측 정확도를 크게 향상시켰습니다.



### SAGE: Streaming Agreement-Driven Gradient Sketches for Representative Subset Selection (https://arxiv.org/abs/2510.02470)
- **What's New**: 이번 논문에서는 대규모 데이터셋에서 신경망 훈련 시 발생하는 높은 계산 비용과 에너지 소비 문제를 해결하기 위해 SAGE라는 새로운 방법을 제안합니다. SAGE는 Frequent Directions (FD) 스케치를 사용하여 고유한 방향성을 유지하고, 훈련 예제를 효과적으로 선택합니다. 이를 통해 데이터의 전부를 사용하지 않고도 일반화 성능을 유지할 수 있게 되었습니다. SAGE는 특히 GPU 친화적인 파이프라인을 통해 메모리 사용을 간소화합니다.

- **Technical Details**: SAGE는 훈련 데이터셋 {(xi,yi)}i=1N{(x_{i},y_{i})}_{i=1}^{N}을 기준으로 작동하며, 모델 fθf_{	heta}는 파라미터 θ∈ℝD	heta	ext{ in }ℝ^{D}을 가집니다. SAGE는 주어진 훈련 예제에 대한 경량화된 Frequent-Directions (FD) 스케치를 유지하며, 업데이트 시 스케치를 스트리밍 방식으로 갱신합니다. 상위 k 인덱스를 선정함으로써 대표성과 다양성을 균형있게 유지하는 점이 특징입니다.

- **Performance Highlights**: SAGE 방법론은 여러 벤치마크에서 소규모 예제 예산을 유지하면서도 전체 데이터 훈련 및 최근의 서브셋 선택 기준선에 비해 경쟁력 있는 정확도를 달성함을 입증했습니다. 실험 결과 SAGE는 모든 실험에서 end-to-end 계산 및 메모리 피크를 줄이는 데 성공하였으며, 효율적인 훈련을 위한 유용한 상수 메모리 대안을 제공합니다. 따라서 SAGE는 모델 압축 및 가지치기를 보완하는 역할을 할 수 있습니다.



### Assessing the Potential for Catastrophic Failure in Dynamic Post-Training Quantization (https://arxiv.org/abs/2510.02457)
- **What's New**: 이 논문에서는 포스트 트레이닝 양자화(PTQ)를 통한 잠재적 성능 저하에 대해 심도 있게 탐구합니다. 특히 동적 PTQ(DPTQ)가 안전-critical 환경에서의 응용 시 발생할 수 있는 심각한 실패를 초래할 수 있음을 강조합니다. 연구진은 지식 증류(Knowledge Distillation)와 강화 학습(Reinforcement Learning) 과제를 통해 이는 가장 나쁜 경우의 잠재적 성능 저하를 분석합니다.

- **Technical Details**: 논문에서는 DPTQ를 적용한 네트워크와 비트 폭 정책을 학습하는 방법을 제시합니다. 이를 통해 모델이 양자화에 취약하거나 강력하도록 결과를 도출할 수 있게 됩니다. 다양한 신경망 아키텍처에 대한 실험을 통해 심각한 실패 사례가 존재함을 실증적으로 확인하였습니다.

- **Performance Highlights**: 연구 결과, 양자화된 네트워크 정책 쌍 중에서는 성능 저하가 10%에서 65%까지 발생할 수 있으며, 일부 경우에는 2% 이하의 성능 감소를 보여주는 견고한 네트워크와 비교되었습니다. 이러한 결과는 PTQ에 의해 유발된 실패 사례에 대한 초기 이해를 제공하며, 실제 배포 시 안전성 확보의 필요성을 강조합니다.



### Market-Based Data Subset Selection -- Principled Aggregation of Multi-Criteria Example Utility (https://arxiv.org/abs/2510.02456)
- **What's New**: 이번 연구는 교육 데이터의 유용한 소 subset을 선택하는 어려움을 해결하기 위한 시장 기반 선택기의 도입을 제안합니다. 이 선택기는 각 샘플의 유용성을 가격으로 평가하며, 다양한 신호들이 가격을 결정하는 트레이더 역할을 합니다. 또한, 주제별 정규화를 통해 안정성을 제공하고, Token 예산을 명시적으로 관리합니다. 이러한 접근 방식은 비용을 최소화하고 적은 컴퓨팅 자원을 활용하는 동시에 데이터 커랜션을 보다 효율적으로 수행합니다.

- **Technical Details**: 연구에서는 각 교육 아이템을 '계약'으로 보고, Logarithmic Market Scoring Rule (LMSR)을 통해 가격을 책정합니다. 샘플의 유용성을 반영하는 가격은 여러 신호의 조합을 통해 결정되며, 각 신호는 주제에 따라 표준화됩니다. 또한, 선택 과정에서 token 길이에 따른 편향을 조절할 수 있는 파라미터가 포함되어 있으며, 이 시스템은 convex cost와 maximum-entropy 원리를 바탕으로 작동합니다.

- **Performance Highlights**: GSM8K와 AGNews 데이터셋을 통해 실험한 결과, 제안된 시장 기반 선택기가 동일한 신호의 강력한 기준선과 동등한 성능을 보이며, 선택 오버헤드도 0.1 GPU-시간 미만으로 유지됩니다. 특히, AGNews 데이터셋에서 상위 5-25%의 문서를 선택할 때 안정성과 정확성을 향상시킵니다. 이러한 프레임워크는 LLMs의 멀티 신호 데이터 커랜션을 통합하는데 유용합니다.



### How to Train Your Advisor: Steering Black-Box LLMs with Advisor Models (https://arxiv.org/abs/2510.02453)
- **What's New**: 본 논문에서는 Advisor Models라는 새로운 프레임워크를 소개합니다. 이 모델은 경량의 파라메트릭 정책으로, 블랙박스 모델에 대한 자연어 조언을 생성하여 반응적으로 조정하는 기능을 가지고 있습니다. Advisor Models는 단일 고정 프롬프트의 한계를 극복하고, 다양한 입력과 환경에 적응할 수 있는 다이나믹한 최적화를 가능하게 합니다.

- **Technical Details**: Advisor Models는 강화 학습(Reinforcement Learning)을 통해 최적화되며, 사용자 입력과 블랙박스 모델 사이에 위치하여 상황에 맞는 가이드를 생성합니다. 이 과정에서는 그룹 상대 정책 최적화(Group Relative Policy Optimization)를 사용하여 관찰된 보상에 기반하여 advisor를 업데이트합니다. 블랙박스 모델의 파라미터에 접근할 필요 없이, advice의 출력을 활용하여 블랙박스 모델을 제어합니다.

- **Performance Highlights**: Advisor Models는 개인화 및 도메인 적응이 필요한 다양한 환경에서 테스트되었으며, 특히 사용자 특정 규칙을 완벽히 학습하며 94-100%의 보상을 달성하는 높은 성능을 보여주었습니다. 또한, 다양한 블랙박스 모델 간의 advisor 전이도 성능 저하 없이 이루어지며, 분포 외 입력에 대한 강건성을 유지하는 결과를 보였습니다.



### RainSeer: Fine-Grained Rainfall Reconstruction via Physics-Guided Modeling (https://arxiv.org/abs/2510.02414)
- **What's New**: 이 논문은 강수 필드를 재구성하는 혁신적인 방법인 RainSeer를 소개합니다. RainSeer는 레이더 반사도를 물리적으로 기초한 구조적 사전으로 해석하며, 강수가 발생하는 시점, 장소 그리고 방법을 포착합니다. 이 프레임워크는 기존의 기상 관측 장치가 간과하는 급격한 전이와 지역적 극단을 포착하는 데 특화되어 있습니다.

- **Technical Details**: RainSeer는 두 가지 기초 구조로 구성되어 있습니다: 구조-점 매퍼(Structure-to-Point Mapper)와 지리적 인식 강수 디코더(Geo-Aware Rain Decoder)입니다. 구조-점 매퍼는 메조스케일 레이더 구조를 지역적 강수로 변환하며, 지리적 인식 강수 디코더는 수증기의 하강, 용해 및 증발 과정을 모델링하여 물리적 일관성을 유지합니다. 이러한 제어는 인과적 시공간 주의 메커니즘(Causal Spatiotemporal Attention)으로 수행됩니다.

- **Performance Highlights**: RainSeer는 두 개의 공공 데이터셋인 RAIN-F(한국, 2017-2019)와 MeteoNet(프랑스, 2016-2018)에서 기존의 최첨단 기법들과 비교하여 일관된 개선을 보였습니다. MAE(Mean Absolute Error)를 각각 13.31%와 70.71% 감소시키고, NSE(Nash-Sutcliffe Efficiency)를 60.75%와 22.56% 향상시키며, 구조적 충실도를 크게 높였습니다.



### OpenTSLM: Time-Series Language Models for Reasoning over Multivariate Medical Text- and Time-Series Data (https://arxiv.org/abs/2510.02410)
- **What's New**: 이번 논문에서는 OpenTSLM이라는 시계열 언어 모델(Time Series Language Models, TSLMs)을 제안하며, 이를 통해 시계열 데이터를 자연어로 처리할 수 있는 방법을 소개합니다. 기존의 LLMs가 시계열 데이터를 다루는 데 한계를 보였던 점을 해결하기 위해, 시계열을 기본 모달리티로 통합하여 모델링하는 방식을 채택하였습니다. OpenTSLM은 시계열과 텍스트를 직접적으로 연결하여 서로 다른 구조로 처리가 가능하도록 설계되었습니다.

- **Technical Details**: OpenTSLM에는 두 가지 아키텍처가 포함되어 있습니다: OpenTSLM-SoftPrompt와 OpenTSLM-Flamingo입니다. OpenTSLM-SoftPrompt는 텍스트 토큰과 결합된 학습 가능한 시계열 토큰을 소프트 프롬프트를 통해 암묵적으로 모델링하여, 두 가지 모두를 단일 시퀀스로 처리합니다. 반면 OpenTSLM-Flamingo는 크로스 어텐션 메커니즘을 사용하여 시계열을 텍스트와 명시적으로 통합하여 연산합니다.

- **Performance Highlights**: OpenTSLM 모델은 다양한 기준선 모델에 비해 뛰어난 성능을 보여주며, 수면 구분 태스크에서 69.9의 F1 스코어를 기록하였고, HAR에서는 65.4에 도달했습니다. 특히, 10억 개의 파라미터를 가진 OpenTSLM 모델도 GPT-4o 모델을 초과하는 성능을 보였으며, OpenTSLM-Flamingo는 OpenTSLM-SoftPrompt와 유사한 성능을 갖추면서도 긴 시퀀스에서 더 나은 성능을 자랑합니다. 모든 코드는 오픈 소스로 제공되어 추가 연구에 용이하도록 지원됩니다.



### Extreme value forecasting using relevance-based data augmentation with deep learning models (https://arxiv.org/abs/2510.02407)
- **What's New**: 이 연구에서는 Extreme 값 예측을 위한 데이터 증강(data augmentation) 프레임워크를 제시합니다. 주로 Generative Adversarial Networks(GANs) 및 Synthetic Minority Oversampling Technique(SMOTE)와 같은 데이터 증강 모델을 활용하여 Deep Learning 모델의 예측 능력을 개선하려 합니다. 다양한 Deep Learning 모델인 Conv-LSTM 및 BD-LSTM를 사용하여 극단적인 값을 예측하는 방법을 검토하고 있습니다.

- **Technical Details**: Extreme 값 이론(extreme value theory)은 극단적인 값을 포함한 문제를 분석하는 학문입니다. 이러한 극단적인 값 예측을 위한 모델을 개발하기 위해서는 데이터 세트를 극단적인 세트와 일반 세트로 나누는 것이 필요합니다. 이 연구에서는 각 데이터 샘플에 대해 relevance function을 활용하여 극단적인 샘플을 분류할 수 있는 방법을 제안하고 있습니다.

- **Performance Highlights**: 결과적으로, SMOTE 기반 전략이 예측 정확도 향상에 지속적으로 기여하며, 극단적 값 예측에서 더 나은 성능을 보여주었습니다. Conv-LSTM은 주기적이고 안정적인 데이터셋에서 우수한 성능을 나타내며, BD-LSTM은 혼란스러운(non-stationary) 시퀀스에서 더 나은 성과를 보였습니다. Signal Extreme Ratio(SER)라는 tail-sensitive 지표를 통해 극단적인 사건에 대한 모델 성능을 평가하며, 이는 전반적인 예측 정확도뿐만 아니라 극단 값 예측의 정확성을 강조합니다.



### Test-Time Defense Against Adversarial Attacks via Stochastic Resonance of Latent Ensembles (https://arxiv.org/abs/2510.03224)
- **What's New**: 이 논문은 적대적 공격(adversarial attacks)에 대한 테스트 시간 방어 메커니즘을 제안합니다. 기존의 방법들은 정보 손실(information loss)을 초래할 수 있는 피처 필터링(feature filtering) 또는 스무딩(smoothing)에 의존하는 반면, 저자는 소음(noise)을 소음으로 상쇄하는 방법을 사용하여 강건성을 높이고 정보 손실을 최소화합니다. 이 기법은 입력 이미지에 소규모 변환적 섭동을 추가하고 변환된 피처 임베딩(feature embeddings)을 정렬하여 집계한 후 원본 이미지로 다시 매핑하는 방식을 사용합니다.

- **Technical Details**: 저자들은 스토캐스틱 공진(stochastic resonance) 기술을 활용하여 적대적 섭동의 영향을 제거하는 방법을 제안합니다. 이를 통해 입력 데이터를 대체하거나 임베딩을 평균화하지 않고, 변환된 임베딩을 잠재 공간(latent space)에서 평균화하는 방식으로 적대적 섭동의 영향을 줄입니다. 이 방법은 별도의 네트워크 모듈 없이 다양한 기존 네트워크 아키텍처에 배포될 수 있으며, 특정 공격 유형을 위한 미세 조정(fine-tuning) 없이 동작합니다.

- **Performance Highlights**: 이 방법은 이미지 분류(image classification), 스테레오 매칭(stereo matching), 광학 흐름(optical flow)과 같은 다양한 작업에서 검증되었습니다. 결과적으로, 깨끗한 성능(clean performance)에 비해 이미지를 분류하는 동안 최대 68.1%의 정확도 손실을 회복하고, 스테레오 매칭에서는 71.9%, 광학 흐름에서는 29.2%를 회복하였습니다. 이 방법은 기존의 어떤 기법보다도 강력한 강건성을 제공하며, 밀집 예측(dense prediction) 작업에 대한 일반적인 테스트 시간 방어(Generic test-time defense)를 수립하였습니다.



### Cache-to-Cache: Direct Semantic Communication Between Large Language Models (https://arxiv.org/abs/2510.03215)
- **What's New**: 이번 연구는 다수의 Large Language Models (LLMs) 의 상호작용에서 텍스트를 넘어선 소통을 가능케 하는 Cache-to-Cache (C2C) 패러다임을 제안합니다. 기존의 다중 언어 모델 시스템에서 LLM들은 텍스트를 통해 소통하며, 이 과정에서 중요한 의미 정보가 손실되고 생성 지연이 발생하는 문제를 해결하고자 하였습니다. Oracle 실험 결과 KV-Cache를 활용하면 응답 품질을 개선하고 속도도 향상할 수 있음을 보여주었습니다.

- **Technical Details**: C2C는 소스 모델의 KV-Cache를 타겟 모델의 공간으로 투영하고 이를 통합하는 신경망을 통해 직접적인 의미 전달을 가능하게 합니다. 학습 가능한 게이팅 메커니즘이 캐시 소통의 이익을 얻는 타겟 레이어를 선택하여 보다 정교한 소통을 구현합니다. 이는 텍스트 커뮤니케이션에서 발생하는 여러 제약(예: 정보 병목, 애매모호성 및 지연)을 극복할 수 있는 새로운 방법론으로 자리매김합니다.

- **Performance Highlights**: C2C 방식은 개별 모델 대비 평균 8.5-10.5% 높은 정확도를 달성하였으며, 텍스트 커뮤니케이션 패러다임보다 3.0-5.0% 높은 성과를 보이며, 평균 2.0배의 속도 향상을 제공합니다. 이 결과들은 LLM 간의 원활한 의미 기반 소통을 통해 성능을 극대화할 수 있음을 입증합니다.



### Joint Bidding on Intraday and Frequency Containment Reserve Markets (https://arxiv.org/abs/2510.03209)
- **What's New**: 이 논문은 배터리 에너지 저장 시스템(BESS)의 다양한 전력 시장 참여를 최적화하는 새로운 접근 방식을 제안합니다. 기존 문헌에서는 주로 단일 전력 시장을 다루었지만, 본 연구는 주파수 조정 시장과 지속적인 거래가 이루어지는 일일 시장을 통합하여 다룹니다. 이를 통해 복잡한 다중 시장 환경에서의 효과적인 정량적 결정 전략을 제공합니다.

- **Technical Details**: 저자들은 혼합 정수 선형 프로그래밍(mixed integer linear programming) 방법론을 활용하여 연속 거래가 가능한 투자 모델을 개발했습니다. 또한, 롤링 내재 알고리즘(rolling intrinsic algorithm)을 적용하여 상태 회복(state of charge recovery) 및 결정 전략을 위한 머신러닝 분류기(learned classifier strategy)를 사용하여 시장 간의 최적 용량 배분을 결정합니다.

- **Performance Highlights**: 실제 독일 시장 데이터에 대한 1년 이상의 전면적인 백테스트 결과, 제안한 접근 방식은 정적 전략 대비 4% 이상의 수익 증가를 보여주었습니다. 특히, 제안된 알고리즘은 순수하게 FCR에만 참여할 때보다 최대 44% 더 많은 수익을 제공하며, 연속적인 일일 시장에서만 거래하는 경우보다 96% 더 높은 수익을 창출함을 입증했습니다.



### Automatic Generation of Digital Twins for Network Testing (https://arxiv.org/abs/2510.03205)
Comments:
          Accepted to ANMS at ICDCS 2025

- **What's New**: 이 연구에서는 자동화된 디지털 트윈(Digital Twin, DT) 생성을 통해 네트워크 소프트웨어의 효율적이고 정확한 검증 도구를 제공하는 방법을 탐구하고 있습니다. 기존의 시뮬레이션 및 하드웨어 기반 접근 방식을 보완하며, ITU-T 자율 네트워크 아키텍처와 연계된 실험 하위 시스템에 맞춰 DT를 자동으로 생성하여 네트워크의 테스트 및 검증을 지원하는 것을 목표로 하고 있습니다.

- **Technical Details**: 자율 네트워크는 최소한의 인간 개입으로 자가 구성, 자가 최적화, 자가 치유 능력을 달성하려고 합니다. 디지털 트윈 기술은 전통적인 네트워크 테스트 방법의 한계를 극복할 수 있는 잠재력을 가지고 있으며, 기계 학습(Machine Learning, ML)을 활용해 데이터 기반의 DT 생성을 자동으로 수행합니다. AutoML(자동화 기계 학습)을 통해 데이터를 처리하고 테스트 시나리오를 처리하는 데 필요한 수작업을 최소화합니다.

- **Performance Highlights**: 초기 사용 사례의 실험 결과는 자동으로 효율적인 디지털 트윈을 생성할 수 있는 접근 방식이 충분한 정확성을 가지고 있다는 것을 보여줍니다. 이 연구는 DT가 네트워크의 동적 행동을 잘 반영하고 더 안전하고 효율적인 검증 솔루션을 제공하는데 유용할 수 있음을 시사합니다. 따라서 AutoML을 통해 생성된 Unit Twins는 시스템 테스트 및 검증을 지원하는 데 중요한 역할을 할 수 있습니다.



### Improving Online-to-Nonconvex Conversion for Smooth Optimization via Double Optimism (https://arxiv.org/abs/2510.03167)
Comments:
          32 pages

- **What's New**: 이번 논문은 비볼록 최적화(nonconvex optimization)에서의 최근 혁신인 온라인-비볼록 변환 프레임워크(online-to-nonconvex conversion framework)를 다룹니다. 이를 통해 $	ext{ε}$-1차 정지점(stationary point)을 찾는 과제를 온라인 학습 문제로 재구성하여 새로운 알고리즘을 제안합니다. 또한, 제안된 방법은 이중 반복 구조(double-loop structure)를 제거하여 복잡도를 낮추고, 확률적 경량화(stochastic gradient)를 통합하여 단일 알고리즘에서 결정론적 및 확률적 성능을 아우르는 통합된 복잡도를 제공합니다.

- **Technical Details**: 우리의 제안은 새로운 이중 낙관적 힌트 기능(doubly optimistic hint function)을 사용하는 온라인 낙관적 경량화 방법입니다. 이 방법은 힌트와 목표 경량화 사이의 차이가 일정하게 유지된다고 가정하며, 연속적인 업데이트 방향이 부드러움(smoothness)으로 인해 천천히 변화한다고 가정합니다. 이러한 새로운 접근을 통해 최소 복잡도 $	ext{O}(	ext{ε}^{-1.75} + σ^{2} 	ext{ε}^{-3.5})$를 달성하게 됩니다.

- **Performance Highlights**: 이번 연구는 최적화 문제에서 결정론적 및 확률적 설정 모두에서 최고의 보증을 동시에 달성하는 최초의 알고리즘을 제안합니다. 기존의 O2NC 프레임워크의 한계를 극복하고, 학습 알고리즘의 모듈성과 확장성을 높이며, 저수준 구간에서 민감도를 유지하는 데 도움을 줍니다. 따라서 알고리즘의 효율성과 실용성을 크게 향상시켰습니다.



### Stimulus-Voltage-Based Prediction of Action Potential Onset Timing: Classical vs. Quantum-Inspired Approaches (https://arxiv.org/abs/2510.03155)
- **What's New**: 본 논문에서는 신경 세포의 행동 신호 처리를 이해하기 위한 필수 요소인 액션 포텐셜(AP) 발생 타이밍의 정확한 모델링을 제안합니다. 연구자들은 전통적인 LIF(leaky integrate-and-fire) 모델의 한계를 극복하기 위해 QI-LIF(quantum-inspired leaky integrate-and-fire) 모델을 도입하여 AP 발생을 확률적 사건으로 간주하며, 이를 통해 생리적 변동성과 불확실성을 포괄적으로 설명합니다. 특히, 실제 실험 데이터를 기반으로 하여 QI-LIF 모델이 기존 LIF 모델에 비해 강한 자극에 대한 예측 오류를 크게 줄인다는 점을 강조합니다.

- **Technical Details**: 연구에서 QI-LIF 모델은 AP 발생 타이밍을 가우시안 확률 분포로 모델링하며, 여기서 타이밍의 중심은 LIF 또는 QLIF 모델에 의해 제시된 가장 가능성 있는 AP 시작 시간에 위치합니다. 이 모델은 자극의 세기에 따라 변하는 동적인 시간 상수를 적용하여, 자극 강도가 증가함에 따라 유효한 막 시간 상수가 단축되는 효과를 포착합니다. LIF 모델과 QI-LIF 모델 간의 비교 분석을 통해 QI-LIF 모델이 더 나은 성능을 보임을 확인했습니다.

- **Performance Highlights**: QI-LIF 모델은 강한 자극 조건에서 AP 발생 타이밍 예측의 정확도를 유의미하게 개선하여 관찰된 생리학적 반응과 밀접하게 일치합니다. 연구 결과는 양자 영감 컴퓨팅 프레임워크의 가능성을 보여주며, 신경 모델링의 정확도를 높이는 데 기여할 수 있습니다. 이 검증된 접근 방식은 양자 스파이킹 신경망(quantum spiking neural network) 개발로 이어져, 향후 금융 시계열 및 복잡한 패턴 분류와 같은 시간 패턴 인식 응용에서도 활용될 가능성이 큽니다.



### ReeMark: Reeb Graphs for Simulating Patterns of Life in Spatiotemporal Trajectories (https://arxiv.org/abs/2510.03152)
Comments:
          15 pages, 3 figures, 2 algorithms, 1 table

- **What's New**: 이번 연구에서는 Markovian Reeb Graphs라는 새로운 프레임워크를 소개하여 인간의 이동성( mobility)을 정확하게 모델링합니다. 이 프레임워크는 기본 데이터에서 학습된 생활 패턴(Patterns of Life, PoLs)을 유지하면서 시공간 경로(spatiotemporal trajectories)를 시뮬레이션할 수 있도록 설계되었습니다. 특히, 개인 및 집단 수준의 이동 구조를 결합하여 확률적(topological model) 모델 내에서 현실적인 미래 경로를 생성합니다.

- **Technical Details**: 우리는 Urban Anomalies 데이터셋의 애틀랜타와 베를린 하위 집합을 사용하여 새로운 방법의 유효성을 평가합니다. 평가 과정에서 Jensen-Shannon Divergence (JSD)를 활용하여 인구 및 에이전트(level) 지표에서의 성능을 비교했습니다. 이 방법은 데이터와 계산(compute) 측면에서 효율성을 유지하면서도 높은 충실도를 달성하는 것으로 나타났습니다.

- **Performance Highlights**: Markovian Reeb Graphs는 다양한 도시 환경에 폭넓게 적용될 수 있는 확장 가능한 경로 시뮬레이션 프레임워크로 자리매김하고 있습니다. 일상생활에서의 일관성과 변동성을 동시에 포착하는 미래 경로 생성의 가능성을 보여줍니다. 이 연구 결과는 도시 계획(urban planning), 전염병학(epidemiology), 교통 관리(traffic management) 분야에서 널리 활용될 수 있습니다.



### The Computational Complexity of Almost Stable Clustering with Penalties (https://arxiv.org/abs/2510.03143)
- **What's New**: 본 논문에서는 작은 doubling dimension을 가진 메트릭에서 안정적인(또는 perturbation-resilient) $	ext{(k-MEANS)}$ 및 $	ext{(k-MEDIAN)}$ 클러스터링 문제의 복잡성을 조사합니다. 기존 연구와는 달리 새로운 안정성 개념인 '거의 안정(almost stable)'을 도입하여, 이는 Balcan과 Liang이 제안한 $(	ext{α}, 	ext{ε})$-perturbation resilience와 유사합니다. 또한, 각 데이터 포인트가 클러스터 센터에 할당되거나 벌점을 부과받는 페널티가 있는 경우에도 결과를 확장하였습니다.

- **Technical Details**: 논문에서는 특별한 경우의 거의 안정 $	ext{(k-MEANS)}/	ext{(k-MEDIAN)}$ 문제(페널티 포함)에 대해 다룬 후, 이들 문제가 다항 시간(polynomial time) 내에 해결 가능하다는 것을 보여줍니다. 이와 함께 일반적인 거의 안정 인스턴스와 $(1 + rac{1}{poly(n)})$-안정 인스턴스의 어려움을 다루면서, 어떤 정확한 알고리즘의 실행 시간에 대해 polynomial 이하의 하한 값을 증명합니다. 이를 위해 널리 신뢰되는 Exponential Time Hypothesis (ETH)를 기반으로 한 분석을 수행합니다.

- **Performance Highlights**: 본 연구에서는 안정성이 보장되는 몇 가지 특수한 경우의 문제에 대해 다항 시간 내에 해결할 수 있음을 보여주어 긍정적인 결과를 제시합니다. 반면, 특정 인스턴스에 대한 낮은 하한 값을 도출함으로써, 이러한 문제들이 해결되기 어려운 상황을 명확하게 합니다. 이는 페널티 기반의 클러스터링 문제에 대한 새로운 통찰력을 제공하며, 향후 연구 방향을 제시하는 기초 자료가 될 것입니다.



### What Drives Compositional Generalization in Visual Generative Models? (https://arxiv.org/abs/2510.03075)
- **What's New**: 본 연구는 시각 생성 모델에서 조합 일반화(compositional generalization)에 영향을 미치는 다양한 디자인 선택이 긍정적 또는 부정적인 방식으로 작용하는지를 체계적으로 분석합니다. 특히, 훈련 목표(training objective)가 이산(discrete) 또는 연속(continuous) 분포에 작용하는지와 훈련 동안 구성 요소 개념에 대한 조건(conditioning)이 얼마나 정보를 제공하는지가 주요하게 작용함을 발견했습니다. 이후, MaskGIT 모델의 이산 손실(discrete loss)을 보조 연속(JEPA-based) 목표로 완화함으로써 조합 성능을 향상시키는 방법을 제안합니다.

- **Technical Details**: 연구에서는 세 가지 주요 구성요소 즉, Tokenizer, Generative model, Conditioning signal을 통해 현대 시각 생성 모델을 분해하고 각 요소가 조합 일반화에 미치는 영향을 살펴봅니다. 각 모델의 아키텍처와 교육 디자인의 선택에 따라 조합 일반화가 어떻게 변화하는지를 체계적으로 비교하기 위해 세 가지 요소에 대해 통제된 실험을 실시하였습니다. 실험 결과, 연속 분포로 훈련된 모델이 이산 분포로 훈련된 모델보다 조합 능력이 더 뛰어난 것으로 나타났습니다.

- **Performance Highlights**: MaskGIT의 분류적 목표를 Joint Embedding Predictive Architecture (JEPA)로 보강함으로써 조합 성능을 명확히 개선할 수 있음을 보여주었습니다. 동시에 JEPA로 훈련된 모델은 더 분리된 중간 표현(intermediate representations)을 제공하고, 이는 예측 기반의 연속 목표가 이산 생성 모델의 내부 구조를 조합성을 유지하는 방향으로 형성할 수 있음을 시사합니다. 이러한 결과는 생성 모델의 조합 및 분해 능력을 향상시키는 방법을 제시하며, 다양한 아키텍처와 훈련 디자인 선택이 생성 품질에 미치는 영향을 강조합니다.



### FR-LUX: Friction-Aware, Regime-Conditioned Policy Optimization for Implementable Portfolio Managemen (https://arxiv.org/abs/2510.02986)
Comments:
          19 pages, 7 figures, includes theoretical guarantees and empirical evaluation, submitted to AI/ML in Finance track

- **What's New**: 본 논문에서는 FR-LUX (Friction-aware, Regime-conditioned Learning under eXecution costs)라는 강화 학습 프레임워크를 소개합니다. 이 방법은 거래 비용을 고려하여 거래 정책을 학습하며, 다양한 변동성-유동성 상태에서도 안정성을 유지합니다. FR-LUX는 거래 비용을 보상에 통합하고, 안정적인 업데이트를 위해 인벤토리 흐름을 제어하는 요구 사항을 도입하는 등 세 가지 핵심 요소를 포함하고 있습니다.

- **Technical Details**: FR-LUX는 포트폴리오 관리를 할인된 마르코프 결정 과정(Discounted Markov Decision Process)으로 모델링합니다. 각 시점에서 에이전트는 예측 변수, 이전 포트폴리오 가중치, 시장 상태를 관찰하고, 후 거래 목표 가중치를 결정하여 거래 흐름을 유도합니다. 비용 및 유동성 고려 메커니즘을 통해 FR-LUX는 효과적인 거래 정책을 학습하며 안정적인 성과를 유지합니다.

- **Performance Highlights**: FR-LUX는 다양한 비용 수준과 시장 상태에 걸쳐 뛰어난 성과를 나타내며, 높은 평균 샤프 비율을 기록했습니다. 또한 증가된 거래 비용에서도 우수한 성능을 유지하며, 낮은 거래량으로도 강력한 성과를 인증합니다. 모델 평가 결과는 통계적으로도 유의미하며, FR-LUX는 기존 대안보다 뛰어난 리스크-리턴 효율을 보여줍니다.



### Oracle-based Uniform Sampling from Convex Bodies (https://arxiv.org/abs/2510.02983)
Comments:
          24 pages

- **What's New**: 새로운 마르코프 체인 몬테 카를로 알고리즘이 제안되어, 볼록체 K에서 균일 분포를 샘플링하는 데 효율적인 방법을 제공합니다. 본 연구의 주요 기여는 거부 샘플링(rejection sampling)과 투영 오라클(projection oracle) 또는 분리 오라클(separation oracle)에 대한 접근을 통해 K에서의 균일 샘플링을 위한 RGO(Restricted Gaussian Oracle)의 효율적인 구현입니다. 이로 인해 설명한 알고리즘은 비편향(unbiased) 샘플을 생성하는 데 필요한 비대칭(complexity) 복잡성을 확립하며, 렌이(divergence) 또는 카이제곱(divergence) 기준에서 정확성을 측정합니다.

- **Technical Details**: 본 논문은 고차원 볼록체에서 균일 샘플을 생성하기 위한 알고리즘에 대해 다루며, 여기서 Alternating Sampling Framework(ASF) 및 근접 샘플러(proximal sampler)를 사용합니다. 우리의 접근법은 ASF와 RGO를 통합하여 두 가지 오라클, 즉 투영 오라클과 분리 오라클을 통한 샘플링을 가능하게 합니다. 알고리즘 3과 알고리즘 4는 이러한 구현을 통해 각각 비편향 샘플을 생성하는 데 집중하고 있으며, 제안된 방법은 기존 방법들보다 실패 확률을 낮춰 보완합니다.

- **Performance Highlights**: 우리의 알고리즘은 알고리즘 2와 결합할 때 더 나은 효율성을 보이며, 비대칭 정확성 조건을 설정해 단계당 최대 질의 수와 재시도 또한 고려합니다. 특히, 알고리즘의 성능은 균일 샘플링이 고차원에서 어떻게 이뤄지는지에 대한 깊은 통찰을 제공하며, 기존 연구들에 비해 명확한 수치적 개선을 보여줍니다. 이를 통해 볼록체의 부피를 효율적으로 계산할 수 있는 소중한 정보와 경험을 제공합니다.



### oRANS: Online optimisation of RANS machine learning models with embedded DNS data generation (https://arxiv.org/abs/2510.02982)
- **What's New**: 이번 논문은 딥 러닝 기반의 Reynolds-averaged Navier–Stokes (RANS) 모델의 온라인 최적화 프레임워크를 제안합니다. 이 프레임워크는 고충실도(training data) 데이터셋이 제한적인 문제를 해결하기 위해, 직접 수치 시뮬레이션(Direct Numerical Simulation, DNS)을 RANS 도메인의 서브도메인에 임베드하여 훈련 데이터를 동적으로 생성합니다. 이를 통해, 모델은 사전 계산된 데이터셋에 의존하지 않고도 새로운 시뮬레이션 조건에 적응할 수 있습니다.

- **Technical Details**: 제안된 온라인 학습 접근법은 고충실도 DNS와 저충실도 RANS 시뮬레이션을 결합하여 이루어집니다. RANS 솔루션은 DNS에 경계 조건을 제공하고 DNS는 흐름 통계량을 제공하여 딥 러닝 클로저 모델의 업데이트를 가능하게 합니다. 이 피드백 루프를 통해 RANS 모델은 사전 데이터 없이도 시뮬레이션 중에 실시간으로 적응할 수 있습니다.

- **Performance Highlights**: 온라인 최적화된 RANS 모델은 기존 오프라인에서 훈련된 모델 및 문헌에서 보정된 클로저보다 상당히 우수한 성과를 보였습니다. 훈련은 기존보다 더 짧은 DNS 서브도메인으로도 가능합니다. 성능 저하는 주로 경계 조건 오염이나 저파수 모드를 포착하기에 너무 짧은 도메인에서 발생합니다. 이 프레임워크는 데이터 적응형 축소 모델을 위한 확장 가능한 경로를 제시합니다.



### Scalable Quantum Optimisation using HADOF: Hamiltonian Auto-Decomposition Optimisation Framework (https://arxiv.org/abs/2510.02926)
Comments:
          Sankar, N., Miliotis, G. and Caton, S. Scalable Quantum Optimisation using HADOF: Hamiltonian Auto-Decomposition Optimisation Framework. In 3rd International Workshop on AI for Quantum and Quantum for AI (AIQxQIA 2025), at the 28th European Conference on Artificial Intelligence (ECAI), October 25-30, 2025, Bologna, Italy

- **What's New**: 본 연구에서는 Hamiltonian Auto-Decomposition Optimisation Framework (HADOF)를 제안합니다. HADOF는 Quadratic Unconstrained Binary Optimisation (QUBO) 문제의 해밀토니안을 자동으로 서브 해밀토니안으로 분해하는 반복 최적화 과정을 활용합니다. 이 방법을 통해 QAOA, Quantum Annealing (QA) 및 Simulated Annealing (SA)와 같은 해밀토니안 기반 최적화 기법으로 개별 최적화를 수행하고, 이를 다시 글로벌 솔루션으로 통합할 수 있습니다.

- **Technical Details**: HADOF는 전체 QUBO를 해밀토니안으로 인코딩하고, 그 후 최적화 알고리즘을 통해 확률 분포를 생성하여 근사 해를 샘플링하는 방식으로 진행됩니다. 각 서브 해밀토니안은 이진 변수의 주변 확률을 사용하여 근사화되며, 반복적으로 서브 해밀토니안을 해결 후 샘플링된 솔루션을 집계하여 다음 반복을 안내합니다. 이러한 접근은 두 개의 주요 단계로 구성되며, 각 서브 문제에서 글로벌 컨텍스트를 유지하며 계산 비용을 관리할 수 있습니다.

- **Performance Highlights**: HADOF는 시뮬레이션 환경에서 CPLEX보다 우수한 성능을 보여주며, 기존의 합리적 하드웨어 조건 하에서도 고품질 솔루션을 동시에 생성할 수 있습니다. 이후 IBM 양자 컴퓨터에서의 토이 문제에 대한 구현을 통해 실용적인 양자 최적화 응용의 가능성을 시연합니다. 이러한 결과는 HADOF가 양자 영감을 받은 고전 알고리즘이자 NISQ 시대 및 미래 양자 장치에서의 확장 가능 방법으로서 유망하다는 것을 보여줍니다.



### Mechanistic Interpretability of Code Correctness in LLMs via Sparse Autoencoders (https://arxiv.org/abs/2510.02917)
- **What's New**: 본 연구는 대형 언어 모델(Large Language Models, LLMs)의 코드 생성 과정에서 코드의 올바름(code correctness) 메커니즘을 이해하는 데 초점을 맞추고 있습니다. 흥미롭게도, Sparse Autoencoders를 적용하여 모델의 표현을 분해하고 코드 정확도를 예측할 수 있는 방향을 식별하였습니다. 이 연구는 코드 생성 메커니즘의 복잡성을 이해하고, LLM의 실제 활용에서의 안전한 배포를 위한 통찰을 제공합니다.

- **Technical Details**: 연구에서 Sparse Autoencoders (SAEs)가 사용되어 LLM의 표현을 해독하고 코드를 올바르게 작성하는 데 필요한 방향을 발견하였습니다. 모델이 코드의 유효성을 판단하는 두 가지 메커니즘, 즉 오류 경고를 위한 detection directions와 의도된 수정을 위한 steering directions이 존재하는 것으로 나타났습니다. 이러한 메커니즘이 특히 테스트 케이스에 집중함으로써 효과적으로 작동한다는 점도 흥미롭습니다.

- **Performance Highlights**: 연구 결과에 따르면 detection directions는 오류를 안정적으로 예측하며, F1 점수는 0.821로 높았습니다. 반면, 정확한 코드에 대한 신뢰도를 나타내는 데는 F1 점수가 0.504로 낮아 비대칭성을 보여주었습니다. 또한, steering interventions는 오류의 4.04%를 수정하지만, 올바른 코드의 14.66%를 손상시키는 trade-off가 필요하다는 점이 강조되었습니다.



### SALSA-V: Shortcut-Augmented Long-form Synchronized Audio from Videos (https://arxiv.org/abs/2510.02916)
- **What's New**: SALSA-V는 비디오에서 오디오로의 전환 형성 모델로, 침묵 비디오 콘텐츠에서 고품질의 고동기화 오디오를 생성할 수 있습니다. 이 모델은 특히 음향 조건화(audio-conditioned generation) 기능과 마스크된 확산 목표를 도입하여 고품질 오디오 샘플을 빠른 단계에서 생성하는 데 성공했습니다. SALSA-V는 기존의 최신 기술과 비교하여 오디오와 비디오 콘텐츠 간의 정밀한 정렬 및 동기화에서 크게 우수한 성능을 보입니다.

- **Technical Details**: SALSA-V는 짧은 샘플링 단계 내에서 고충실도의 오디오 생성을 가능하게 하는 단축 보조(latent flow matching) 모델입니다. 이를 통해 사용자들은 참조 오디오 섹션을 기반으로 오디오 조건을 설정하고, 긴 형식의 오디오 생성도 가능해집니다. 이 모델은 대규모로 사전 학습된 백본을 사용하여 고해상도의 동기화 기능을 제공하며, 이는 기존의 모델들에 비해 월등한 성능을 나타냅니다.

- **Performance Highlights**: SALSA-V는 짧은 시간 내에 높은 품질의 오디오 생성을 가능하게 하여 거의 실시간 응용 프로그램에도 적용할 수 있습니다. 사람 청취 연구(human listening study)를 통해, 이 모델은 오디오 품질과 동기화 측면에서 기존의 최신 기술보다 현저히 우수한 결과를 기록했습니다. 추가적으로, SALSA-V는 Foley 생성과 같은 전문 오디오 합성 작업에의 적용 가능성을 높이는 임의 마스킹(random masking) 훈련 방법을 도입하고 있습니다.



### WavInWav: Time-domain Speech Hiding via Invertible Neural Network (https://arxiv.org/abs/2510.02915)
Comments:
          13 pages, 5 figures, project page: this https URL

- **What's New**: 본 논문은 DNN(Deep Neural Network) 기반의 새로운 오디오 데이터 숨기기 기법을 제안합니다. 이전 방법들의 한계를 보완하기 위해, 흐름 기반의 가역 신경망을 사용하여 스테고 오디오(stego audio)와 커버 오디오(cover audio) 간의 직접적인 연결을 Establish합니다. 이 방법은 메시지를 숨기고 추출하는 과정의 가역성을 높이며, 시간-주파수 손실(time-frequency loss)을 적용하여 비밀 오디오의 품질을 향상시킵니다.

- **Technical Details**: 제안된 방법은 비밀 메시지를 시간 영역에서 직접 숨기면서 시간-주파수 제약(time-frequency constraints)을 활용하여 채널 왜곡(channel distortion)을 피하는 것을 목표로 합니다. 이 과정에서, 스테고 오디오에서 비밀 오디오를 복원하기 위해 새로운 흐름 기반 가역 신경망을 동시에 학습하여, 기존의 인코더-디코더 아키텍처보다 더 나은 가역성을 가지도록 설계하였습니다. 또한, 다양한 형태의 소음을 처리할 수 있도록 훈련 단계에서 소음 레이어(noise layer)를 도입하였습니다.

- **Performance Highlights**: 실험 결과, VCTK 및 LibriSpeech 데이터셋에서 우리의 방법이 이전 방법들과 비교하여 주관적 및 객관적 지표에서 우수한 성능을 보였습니다. 특히, 제안된 시스템은 비밀 오디오의 품질 유지와 다양한 일반 왜곡에 대한 강인성을 Exhibit하며, 목표 지향적인 안전한 통신 시나리오에서 유용할 것으로 예상됩니다. 또한, 본 연구는 비밀 메시지를 보호하기 위한 암호화-복호화(encryption-decryption) 모듈을 포함하여 시스템의 안전성을 높였습니다.



### ELMF4EggQ: Ensemble Learning with Multimodal Feature Fusion for Non-Destructive Egg Quality Assessmen (https://arxiv.org/abs/2510.02876)
Comments:
          30 pages

- **What's New**: 이 논문은 상업적 가금류 생산에서 안전한 식품 관리와 제품 품질 유지를 위해 필요할 정도로 정확하고 비파괴적인 계란 품질 평가 방법을 제안합니다. ELMF4EggQ라는 앙상블 학습 프레임워크를 도입하여 이미지, 모양, 무게와 같은 외부 특성만을 사용하여 계란의 등급과 신선도를 분류합니다. 186개의 갈색 계란 데이터셋을 생성하였으며, 최초로 외부 비침습적 특성을 활용한 머신러닝 방법으로 내부 품질 평가를 수행한 연구라고 할 수 있습니다.

- **Technical Details**: 이 프레임워크는 외부 계란 이미지에서 추출한 딥 피처를 계란의 형태와 무게와 같은 구조적 특성과 통합하여 각 계란을 포괄적으로 표현합니다. 이미지 피처 추출은 사전 훈련된 CNN 모델(ResNet152, DenseNet169, ResNet152V2)을 활용하여 이루어지며, 이후 PCA 기반의 차원 축소와 SMOTE 증강을 통해 다수의 머신러닝 알고리즘으로 분류됩니다. 최종적으로, 여러 분류기의 예측을 통합하기 위해 앙상블 투표 메커니즘이 적용됩니다.

- **Performance Highlights**: 실험 결과에 따르면, 이 다중 모달 접근법은 이미지 전용 및 구조적 데이터(모양과 무게) 전용 기준에 비해 상당한 성능 향상을 보여주었습니다. 다중 모달 앙상블 접근법은 등급 분류에서 86.57%, 신선도 예측에서 70.83%의 정확도를 달성하였습니다. 또한 코드와 데이터는 이 URL에서 공개되어 투명성과 재현성을 촉진하고 추가 연구를 지원합니다.



### The land use-climate change-biodiversity nexus in European islands stakeholders (https://arxiv.org/abs/2510.02829)
Comments:
          In press at the Environmental Impact Assessment Review journal. Pre-proof author's version

- **What's New**: 이번 논문은 21개 유럽 섬에서 기후 변화와 토지 이용 변화에 대한 이해도를 높이기 위해 이해관계자들의 관점과 지식의 격차를 조사하였습니다. 이해관계자들은 생태계 서비스에 영향을 미치는 기후 및 토지 이용 변화 이슈에 대해 논의하였으며, 기후 변화 인식에는 온도, 강수량, 습도, 극단적인 날씨 및 바람이 포함되었습니다.

- **Technical Details**: 본 연구에서는 기후와 토지 이용 변화의 영향 인식을 머신러닝(Machine Learning)을 통해 분석하여 그들의 영향을 정량화했습니다. 주요 기후적 특성은 온도가, 주요 토지 이용 특성은 산림 파괴(deforestation)로 나타났으며, 수자원 관련 문제는 이해관계자들에게 가장 우선적으로 다루어야 할 문제로 identified 되었습니다.

- **Performance Highlights**: 에너지 관련 문제, 즉 에너지 부족과 풍력 및 태양광 시설에 대한 문제는 기후와 토지 이용의 복합적 위험 요소로 높은 비율을 차지하였습니다. 연구 결과, 이해관계자들은 생태계 서비스에 대한 기후 변화의 영향을 부정적으로 인식하고 있으며, 자연 서식지 파괴와 생물 다양성 손실(biodiversity loss)이 가장 큰 문제로 확인되었습니다.



### Pareto-optimal Non-uniform Language Generation (https://arxiv.org/abs/2510.02795)
Comments:
          24 pages, 1 figure

- **What's New**: 이번 연구에서는 Kleinberg와 Mullainathan이 제시한 언어 생성을 다루는 최신 모델을 발전시켜, 비균일 언어 생성(non-uniform language generation)에서의 Pareto Optimality에 대해 분석합니다. 특히, 기존 연구에서 제시된 알고리즘들은 언어에 따라 생성 시간이 최적이 아닐 수 있으며, 이에 대한 해결책을 제시합니다. 저자들은 생성 시간이 거의 Pareto-optimal한 새로운 알고리즘을 제안하여, 모든 언어에 대해 동시에 최적의 생성 시간을 달성하는 방법에 대한 기초를 마련합니다.

- **Technical Details**: 언어 생성 모델은 무한 문자열 집합에 기반한 개별 언어를 다루며, 적대자가 선택한 언어 $L$의 문자열을 온라인 방식으로 나열합니다. 높은 수준의 보장을 위해 비균일 언어 생성 개념이 도입되었으며, 이론적으로 구현 가능성이 보장됩니다. 본 연구의 알고리즘은 언어 $L$의 생성 시간 $t^ullet(L)$가 Pareto-optimal성을 가지도록 설계되어 있으며, 이는 특정 언어의 생성 시간이 다른 언어에 비해 나쁨이 없음을 의미합니다.

- **Performance Highlights**: 제안된 알고리즘은 무한 개의 언어에서 동시에 최적의 생성 시간을 달성하기 위한 새로운 기준을 제시합니다. 알고리즘은 기존의 연구에서 발견된 비균일 언어 생성 알고리즘들과 비교했을 때, 성능상에서 우수성을 보입니다. 이 작업은 노이즈가 있는 또는 대표적인 생성 설정에서도 Pareto-optimal 알고리즘을 얻을 수 있는 통합개념을 제공하여, 다양한 타당한 조건 하에서도 성능을 극대화할 수 있음을 입증합니다.



### Align Your Query: Representation Alignment for Multimodality Medical Object Detection (https://arxiv.org/abs/2510.02789)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 다양한 의료 이미징 모달리티(예: CXR, CT, MRI)에 대한 의료 물체 탐지의 성능을 개선하기 위해 새로운 방법을 제시합니다. 기존의 단일 탐지기가 혼합된 의료 모달리티에서 학습할 때의 한계를 극복하기 위해, 객체 쿼리의 표현을 모달리티 텍스트와 정렬하는 방식을 채택하였습니다. 이를 통해, 모달리티 정보를 명확하게 반영하면서도 추가적인 주석 없이 가벼운 모달리티 토큰을 도입합니다.

- **Technical Details**: 본 연구의 핵심 기술은 Multimodality Context Attention(MoCA)와 Query REPA입니다. MoCA는 객체 쿼리와 모달리티 토큰 간의 상호 작용을 통해 모달리티 정보를 감지 과정에 통합하는 혁신적인 방법입니다. 또한 제안된 QueryREPA는 사전 학습 단계에서 특정 작업을 위한 대조 손실을 사용하여 쿼리 표현을 모달리티 토큰과 정렬하여, 다중 모달리티에서 신뢰성 있는 대표성을 형성합니다.

- **Performance Highlights**: 제안된 방법은 서로 다른 모달리티로 학습된 데이터셋에서 일관되게 AP(평균 정확도)를 개선하며, 구조적 수정 없이 최소한의 오버헤드로 성능 향상을 꾀합니다. MoCA와 QueryREPA의 결합은 의료 물체 탐지 성능에 시너지 효과를 제공하여, 실제 환경에서의 다중 모달리티 의료 정보 처리를 위한 효과적인 경로를 제시합니다.



### Hierarchical Generalized Category Discovery for Brain Tumor Classification in Digital Pathology (https://arxiv.org/abs/2510.02760)
- **What's New**: 이번 연구에서는 뇌종양 분류를 위한 새로운 접근법인 Hierarchical Generalized Category Discovery for Brain Tumor Classification (HGCD-BT)를 소개합니다. 이 방법은 계층적 클러스터링(hierarchical clustering)과 대조 학습(contrastive learning)을 통합하여, 미지의 클래스도 포함된 비정형 데이터에서 분류를 가능하게 합니다. 특히, 기존의 GCD 방법들이 가지는 한계를 극복하며, 뇌종양의 계층적 분류 구조를 반영할 수 있습니다.

- **Technical Details**: HGCD-BT는 레이블이 있는 데이터와 없는 데이터를 혼합하여 비정형 데이터를 분류하는 것을 목표로 합니다. 이 기법은 새로운 반지도 학습(semi-supervised) 기반의 계층적 클러스터링 손실을 도입하여, 학습 과정에서 계층적 구조를 직접적으로 모델링함으로써, 기존의 GCD 방법보다 더 우수한 성능을 보입니다. 실험을 통해 HGCD-BT는 기존 GCD 방식 대비 +28%의 정확성 향상을 달성했으며, 이 결과는 뇌종양의 다양한 하위 유형을 식별하는 데에 특히 효과적임을 보여줍니다.

- **Performance Highlights**: HGCD-BT는 OpenSRH 데이터셋에서 패치 수준 분류에 대한 성능을 평가하여 큰 개선을 보였습니다. 또한 Hematoxylin and Eosin (H&E) 염색된 전체 슬라이드 이미지를 사용한 슬라이드 수준 분류에서도 일반화 가능성을 Demonstrate하여, 12개의 뇌종양 유형을 정확하게 분류할 수 있음을 입증했습니다. 이 연구는 imaging modalities의 다양성과 분류의 세분화에 있어 HGCD-BT의 강력한 성능을 강조합니다.



### Neural Jump ODEs as Generative Models (https://arxiv.org/abs/2510.02757)
- **What's New**: 이번 연구에서는 NJODE(Neural Jump ODE)가 Itô 프로세스의 생성 모델로 사용될 수 있는 방법을 탐구합니다. 이 방법을 통해 고정된 Itô 프로세스 샘플의 이산 관측값을 기반으로 드리프트(drift) 및 확산(diffusion) 계수를 근사화할 수 있습니다. NJODE를 활용하여 구현한 모델은 적대적 훈련(adversarial training) 필요 없이 관찰된 샘플에 대해서만 예측 모델로 훈련될 수 있는 장점이 있습니다.

- **Technical Details**: NJODE 프레임워크는 경로 의존적인 패스(path-dependent) Itô 프로세스를 다루며, 불규칙하게 샘플링된 데이터와 결측값이 있는 데이터에 자연스럽게 대응합니다. 이 모델은 연속 시간에서 최적의 예측을 가능하게 하며, 불완전한 과거 관측치를 기반으로 조건부 경로를 생성할 수 있습니다. 특히, NJODE는 L2 센스에서 최적의 예측으로 수렴하여 Itô 프로세스의 진짜 계수를 복원하는 수학적 보장을 제공합니다.

- **Performance Highlights**: NJODE는 과거의 이산 관측 데이터를 바탕으로 조건부 기대값을 추정하고, 이를 통해 Itô 프로세스의 계수를 정확히 추정하는 데 성공합니다. 이 연구에서는 NJODE의 추정기가 대칭적이고 양의 반정적(positive semi-definite)인 분산 행렬을 유지하도록 수정함으로써, 생성된 샘플의 신뢰성을 보장합니다. 따라서 실제 데이터를 다룰 때 이 모델은 크게 유용할 것으로 기대됩니다.



### Flow with the Force Field: Learning 3D Compliant Flow Matching Policies from Force and Demonstration-Guided Simulation Data (https://arxiv.org/abs/2510.02738)
- **What's New**: 이 논문에서는 최근 비주얼 모터 정책(Visuomotor Policy)에 대한 발전에도 불구하고, 접촉이 많은 과제가 여전히 도전 과제로 남아 있다는 점을 강조합니다. 기존의 비주얼 모터 정책은 물리적 상호작용의 중요성을 간과하고 있어, 일반적으로 과도한 접촉력 또는 불안정한 행동을 초래합니다. 따라서 이 연구에서는 단일 인간 demonstration을 통해 시뮬레이션에서 force 정보가 있는 데이터를 생성하는 프레임워크를 소개하여, 접촉을 잘 유지하고 새로운 조건에 적응하는 성능 향상을 보여줍니다.

- **Technical Details**: 이 연구는 시뮬레이션 데이터를 활용하여 비전-힘 적응형 컴플라이언스 정책을 학습하는 문제를 다룹니다. 연구진은 force-informed trajectory modulation과 Laplacian editing을 통해 시뮬레이터에서 단일 demonstration으로부터 다양한 동작을 생성하는 경량 데이터 생성 전략을 제안하였습니다. 생성된 합성 궤적은 3D 포인트 클라우드 관찰, 엔드 이펙터 포즈, 그리고 힘 측정을 조건으로 하는 모방 정책 학습에 사용됩니다.

- **Performance Highlights**: 이 프레임워크는 Franka 로봇과 같은 실제 로봇에서 상자 뒤집기 및 양손으로 물체 옮기기와 같은 작업에 대해 기존의 실제 demonstration 없이 제로샷 전이를 성공적으로 시연하였습니다. 또한, 정책 롤아웃 자세를 상태-속도 필드(state-velocity field)로 인코딩하여 패시브 임피던스 컨트롤러와 결합함으로써, 현실적인 수행 중에서도 성능과 안정성을 높였습니다. 이러한 접근법은 동작에 대한 에너지 주입을 줄이면서도 좋은 성능을 유지하는 데 기여할 것입니다.



### Quantitative Convergence Analysis of Projected Stochastic Gradient Descent for Non-Convex Losses via the Goldstein Subdifferentia (https://arxiv.org/abs/2510.02735)
Comments:
          40 pages, 2 figures, under review for 37th International Conference on Algorithmic Learning Theory

- **What's New**: 본 논문에서는 비볼록(非凸) 손실을 위해 프로젝티드 확률적 경량화 하강법(Projected Stochastic Gradient Descent, SGD)의 수렴 속성을 분석합니다. 기존의 연구들과는 달리 Moreau envelope를 사용하지 않고, 제약에 의해 생성된 Goldstein 서브미분(Goldstein subdifferential)으로 측정된 수렴 기준을 제안했습니다. 이러한 접근법은 분산 감소(variance reduction) 방법 없이도 수렴을 보장합니다.

- **Technical Details**: 제안된 수렴 기준은 UN-constrained 문제에서 일반적으로 사용되는 기준으로 축소될 수 있습니다. 우리는 독립 동등 분포(IID) 데이터나 혼합 조건($L$-mixing)을 만족하는 데이터에 대해 비-asymptotic 수렴과 밀접하게 관련된 $O(N^{-1/3})$ 한계를 도출합니다. IID 서브-가우시안 데이터의 경우, 거의 확실한 비-asymptotic 수렴 및 높은 확률 $O(N^{-1/5})$ 한계를 즉각적으로 나타냅니다.

- **Performance Highlights**: 비볼록 손실을 가진 프로젝티드 SGD에 대해 최초의 비-asymptotic 높은 확률 한계를 도출한 것이 주요 성과입니다. 각 성능 측정은 Goldstein 서브미분과의 거리를 통해 이루어지며, 이는 기존 연구들과는 근본적으로 다른 점입니다. 제안된 메트릭에 따라 수렴 조건을 개선하여, 수렴 속도를 높일 수 있습니다.



### Time-To-Inconsistency: A Survival Analysis of Large Language Model Robustness to Adversarial Attacks (https://arxiv.org/abs/2510.02712)
- **What's New**: 이번 연구에서는 대화 AI의 면역력을 평가하기 위해 최초의 포괄적인 생존 분석(survival analysis)을 제시합니다. 9개의 최신 대형 언어 모델(LLMs)에 대해 36,951회의 대화 턴을 분석하여 대화가 실패하는 것을 시간에 따라 변하는 과정으로 모델링합니다. 급격한 의미 변동이 대화 실패의 위험을 극적으로 증가시키는 반면, 점진적이며 누적적인 변동은 보호 효과를 나타내는 등의 발견을 통해, 생존 분석을 LLM의 강건성을 평가하는 강력한 패러다임으로 자리매김하였습니다.

- **Technical Details**: 이 연구에서는 생존 분석을 통해 대화 모델의 일관성을 재구성하였습니다. 대화 실패를 '이벤트'로 모델링하고 시간은 대화 턴의 순서에 따라 측정합니다. 사용된 모델링 기법으로는 Cox 비례 위험 모델, 가속화 실패 시간(Accelerated Failure Time, AFT) 모델, 비모수적 랜덤 생존 포리스트(Random Survival Forest)가 있습니다. 이를 통해 다양한 실패 시간 분포와 위험 함수를 고려하여 강건한 결론을 도출하였습니다.

- **Performance Highlights**: 생존 분석을 통한 이 연구의 결과는 LLM의 대화 강건성 평가에서 강력한 통찰력을 제공합니다. 특히 AFT 모델이 상호작용을 보여주며 우수한 성과를 달성했는데, 이는 변별력과 보정(calibration)에서 뛰어난 결과를 보여주었습니다. 이와 같은 결과는 기존의 단일 턴 평가 방식이 놓치기 쉬운 대화 내 응답 불일치의 위험을 정량화하여 대화 AI 시스템의 신뢰성을 높이는데 기여할 수 있을 것입니다.



### A Statistical Method for Attack-Agnostic Adversarial Attack Detection with Compressive Sensing Comparison (https://arxiv.org/abs/2510.02707)
- **What's New**: 본 논문에서는 현대 머신 러닝 시스템에 대한 적대적 공격의 위험을 줄이기 위해 새로운 통계적 접근법을 제안합니다. 기존의 탐지 방법들이 보이지 않는 공격을 탐지하는 데 한계를 가진 반면, 본 연구에서는 신경망을 배포하기 전에 탐지 기준선을 수립하여 효과적인 실시간 적대적 탐지를 가능하게 합니다. 이는 압축된 신경망과 비압축된 신경망의 행동을 비교하여 적대적 존재의 지표를 생성합니다.

- **Technical Details**: 우리는 압축된 이미지와 비압축된 이미지를 입력으로 사용하는 두 개의 신경망을 사용해 이들의 분포 사이의 차이를 측정하는 방법을 개발했습니다. KL divergence와 L2 norm과 같은 여러 수학적/통계적 연산들을 활용하여 두 벡터(예: 신경망의 마지막 특성 층에서 생성된 특성 맵) 간의 차이를 정량화합니다. 이를 통해 적대적 이미지와 깨끗한 이미지를 구별할 수 있으며, 특정 임계값을 설정하여 적대적 변형 여부를 결정합니다.

- **Performance Highlights**: 본 방법은 다양한 공격 유형에 대해 거의 완벽한 탐지 성능을 보였으며, 기존의 방법들이 여러 공격에서 성능이 좋지 않은 것과 대비됩니다. 또한, 높은 수준의 가짜 긍정을 줄임으로써 실용적인 응용이 가능하도록 신뢰성을 높였습니다. 이 접근법은 공격 유형에 대한 사전 지식이 필요 없으므로, 모든 적대적 공격에 대해 효과적으로 적용될 수 있는 장점이 있습니다.



### ARMs: Adaptive Red-Teaming Agent against Multimodal Models with Plug-and-Play Attacks (https://arxiv.org/abs/2510.02677)
Comments:
          60 pages, 16 figures

- **What's New**: 이 논문에서는 Vision-Language Models (VLMs)에 대한 새로운 적응형 레드팀 에이전트인 ARMs를 제안합니다. ARMs는 VLM의 위험 평가를 시스템적으로 수행하며, 다양한 레드팀 전략을 최적화하여 유해한 출력( harmful outputs)을 효과적으로 이끌어내는 기능을 갖추고 있습니다. 또한 11개의 새로운 다중모드 공격 전략과 17개의 레드팀 알고리즘을 통합하여 VLM의 새로운 취약점을 탐색하고 있습니다.

- **Technical Details**: 기존의 안전 평가 방법론들은 주로 정적인 벤치마크에 의존하고 있어 새로운 위험에 적절히 대응하지 못하고 있었습니다. ARMs는 멀티모달 중심의 공격 전략을 다단계로 조합하며, 레이어드 메모리 아키텍처를 통해 공격의 다양성을 확보하고 효율적인 위험 탐색을 가능케 합니다. 이 메모리 설계를 통해 ARMs는 공격 패턴에 대한 오버피팅을 방지하고, 다양한 레드팀 인스턴스를 생성할 수 있습니다.

- **Performance Highlights**: ARMs는 여러 공개 벤치마크에서 기존 최상의 기준인 X-Teaming의 공격 성공률(ASR)을 평균 52.1% 초과하여 개선하며, Claude-4-Sonnet 모델에서 90% 이상의 공격 성공률을 기록했습니다. ARMs-Bench를 통해 30,000개 이상의 레드팀 인스턴스와 51개의 위험 범주를 아우르는 대규모 다중모드 안전 데이터셋을 구축했으며, 안전 파인튜닝을 통해 VLM의 안전성을 개선하는 방향으로 실질적인 지침을 제공합니다.



### Uncertainty as Feature Gaps: Epistemic Uncertainty Quantification of LLMs in Contextual Question-Answering (https://arxiv.org/abs/2510.02671)
- **What's New**: 이 연구는 Uncertainty Quantification (UQ) 분야에서 기존의 폐쇄형 사실 기반 질문 응답(QA)에서 벗어나 문맥 기반 QA에 초점을 두고 있습니다. 연구자들은 모델의 예측 분포와 진짜 분포 간의 교차 엔트로피를 통한 토큰 수준의 불확실성 측정 방법을 소개합니다. 이러한 접근법을 통해 이념적 모델 기준으로 에피스테믹 불확실성을 수량화할 수 있습니다.

- **Technical Details**: 제안된 방법은 모델의 마지막 레이어 숨겨진 상태와 이상적인 모델 간의 거리를 통해 에피스테믹 불확실성을 한계 지을 수 있음을 보여줍니다. 이 거리는 독립적인 모델 기능들의 거리의 합으로 표현될 수 있으며, 이를 통해 세 가지 주요 특징인 문맥 의존성(context-reliance), 문맥 이해(context comprehension), 정직성(honesty)을 도출합니다. 연구에서는 이 세 가지 특징을 활용하여 주어진 질문-문맥 쌍에 대한 모델의 불확실성을 효율적으로 평가합니다.

- **Performance Highlights**: 연구 결과, 제안된 방법은 기존의 비지도 및 지도 UQ 방법보다 최대 16 PRR 포인트 향상된 성능을 보이며, 동일한 라벨된 샘플을 사용하여 강력한 지도 기반 방법인 SAPLMA 및 LookbackLens를 각각 13 PRR 포인트 초과하여 능가했습니다. 또한, 이 방법은 분포 외 일반화 성능에서도 SAPLMA보다 월등히 뛰어난 성능을 발휘합니다.



### On the Role of Temperature Sampling in Test-Time Scaling (https://arxiv.org/abs/2510.02611)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 추론 시 reasoning을 개선하기 위해, 샘플링 온도를 다양하게 조절하는 다중 온도 스케일링(multi-temperature scaling) 방법을 제안하였습니다. 기존의 연구에서는 샘플 수(K)를 늘리면 정확도가 꾸준히 향상되었지만, K가 매우 커질 경우 더 이상의 향상이 발생하지 않는 한계를 발견했습니다. 이를 통해 단일 온도 스케일링이 모델의 잠재력을 완전히 탐색하지 못한다는 점을 밝혀냈습니다.

- **Technical Details**: 다중 온도 스케일링은 다양한 온도를 통해 샘플을 균등하게 나누어 모델의 reasoning 경계를 확장하는 접근입니다. 실험적으로 1,024개의 트레이스를 생성하는 동안 온도를 0.0에서 1.2까지 변화시키며 성능을 비교하였습니다. 이 과정에서, 특정 온도에서만 해결 가능한 어려운 문제들을 발견함으로써 모델의 전반적인 문제 해결 능력이 증가하는 것을 관찰했습니다.

- **Performance Highlights**: 평균적으로 Qwen3 모델(0.6B, 1.7B, 4B, 8B) 및 다섯 가지 대표적인 벤치마크를 넘어온 결과, 다중 온도 스케일링이 단일 온도 TTS에 비해 평균 7.3 포인트의 성능 향상을 보였습니다. 또한, 다중 온도 스케일링을 통해 기본 모델이 추가적인 후 훈련 없이도 RL-trained 모델과 유사한 성능에 도달할 수 있음을 증명했습니다. 이러한 결과는 TTS의 강력한 가능성과 다중 온도 스케일링의 효과적인 적용을 강조합니다.



### FLOWR.root: A flow matching based foundation model for joint multi-purpose structure-aware 3D ligand generation and affinity prediction (https://arxiv.org/abs/2510.02578)
- **What's New**: 이 논문에서는 포켓-인식 3D 리간드 생성과 결합 친화도 예측을 결합한 새로운 동등한 유동 매칭 모델(http URL)을 제안합니다. 이 모델은 de novo 리간드 생성, 약리학적 조건 샘플링, 프래그먼트 정교화 및 다중 목표 친화도 예측(pIC50, pKi, pKd, pEC50)을 지원합니다. 프로젝트 특정 데이터세트에 대한 효율적인 미세 조정을 통해서도 최적의 성능을 발휘합니다.

- **Technical Details**: 제안된 모델은 넓은 스케일의 리간드 라이브러리와 혼합 정확도의 단백질-리간드 복합체로 훈련 후, 정제된 공결정 데이터세트를 바탕으로 미세 조정을 시행합니다. 모델은 포켓 조건화 및 효율적인 ODE 샘플링을 통해 de novo, 상호작용 제약 및 프래그먼트 기반 생성을 통합하여 리간드를 생성하는 흐름 매칭 접근 방식을 제안합니다. 또한, 구조-활동 관계(SAR)의 변화에 따라 모델이 지속적으로 적응하도록 돕는 동적 모델을 목표로 합니다.

- **Performance Highlights**: 이 모델은 무조건 3D 분자 생성과 포켓 조건화 리간드 설계에서 최첨단 성능을 달성하고, GEOMETRICALLY realistic(기하학적으로 현실적인) 저스트레인 구조를 생성합니다. SPINDR 테스트 세트에서의 친화도 예측 모듈은 충분한 정확도를 보여주며, Schrodinger FEP+/OpenFE 벤치마크에서 기존 모델들보다 뛰어난 속도 이점을 가지고 있습니다. 사례 연구에서는 리간드 생성과 실험적인 데이터 간의 강력한 상관관계를 증명하였으며, 약물 발견의 구조 기반 설계에 대한 포괄적인 기초를 제공합니다.



### Agentic Additive Manufacturing Alloy Discovery (https://arxiv.org/abs/2510.02567)
- **What's New**: 이 논문에서는 새로운 합금 발견을 자동화하고 가속화하기 위해 대규모 언어 모델(LLM) 기반의 에이전트와 다중 에이전트 시스템을 통합한 방법론을 제시하고 있습니다. 이러한 시스템은 사용자가 제안한 합금의 인쇄 가능성에 대한 분석을 수행하고, Tool call 결과에 따라 작업 경로를 동적으로 조정하여 자율적인 의사 결정을 가능하게 합니다. 이를 통해 Additive Manufacturing (AM) 분야에서 효율적인 합금 개발이 이루어질 수 있습니다.

- **Technical Details**: 연구에서는 Thermo-Calc 소프트웨어를 이용하여 특정 합금 조성에 대한 물성(physical properties)을 계산합니다. CALPHAD 기반의 솔버를 활용하여 고유의 상 다이어그램과 물리적 성질을 예측하며, 선택한 데이터베이스에서 필요한 thermophysical 속성을 추출해냅니다. 이 과정에서 Gibbs Free Energy를 최소화하는 반복 계산을 통해 각 합금의 고상 및 융해 상 전이 온도를 수치적으로 도출합니다.

- **Performance Highlights**: 이 다중 에이전트 시스템은 자율적으로 작업을 수행하며, 편리한 MCP 인터페이스를 통해 다양한 클라이언트에 통합될 수 있습니다. 실시간으로 인쇄 가능성을 평가하고, 결함 없는 융해 과정을 위한 프로세스 맵을 생성하는 등의 기능을 통해 실제 제조 환경에서 매우 유용한 도구로 자리잡을 수 있습니다. 이를 통해 AM 산업에서의 혁신적인 합금 개발 가속화가 기대됩니다.



### Even Faster Kernel Matrix Linear Algebra via Density Estimation (https://arxiv.org/abs/2510.02540)
- **What's New**: 이 논문은 $	ext{KDE}(Kernel Density Estimation)$를 이용하여 $	ext{R}^d$의 $n$ 데이터 포인트 kernel matrix에 대한 선형 대수학적 작업을 연구합니다. 특히, 기존 알고리즘을 개선하여 행렬-벡터 곱, 행렬-행렬 곱, 스펙트럼 노름 및 모든 항목의 합을 $(1+	ext{ε})$ 상대 오차로 계산하는 데 성공했습니다. 이 알고리즘은 차원 $d$, 데이터 포인트 수 $n$, 목표 오차 $	ext{ε}$에 의해 영향을 받습니다.

- **Technical Details**: 제안된 알고리즘은 KDE 쿼리를 통해 kernel matrix에 접근할 때 $n$에 대한 의존성이 현저히 낮아집니다. 기존의 최고의 알고리즘인 Backurs, Indyk, Musco, Wagner '21과 비교하여, $	ext{ε}$에 대한 다항적 의존성을 줄이고, kernel matrix의 모든 항목의 합을 계산할 때 $n$에 대한 의존성도 감소했습니다. 논문은 이론적 성과를 뒷받침하기 위해 관련 문제에 대한 여러 하한(lower bound)을 제시합니다.

- **Performance Highlights**: 논문의 개선된 메서드는 기존 알고리즘보다 계산 시간에서 더 효율적입니다. KDE 쿼리를 사용하여 kernel matrix를 다룰 때, 특히 $n$에 대한 의존성이 줄어들어 성능이 향상되었습니다. 이 연구는 KDE 기반 접근 방식의 한계와 조건부 이차 시간 난이도 결과를 제시하며, 향후 연구에 대한 방향성을 제공합니다.



### Learning Multi-Index Models with Hyper-Kernel Ridge Regression (https://arxiv.org/abs/2510.02532)
- **What's New**: 이 논문에서는 딥 뉴럴 네트워크가 고차원 문제에서 우수한 성능을 내는 이유를 탐구하며, 이러한 성공의 이론적 기초에 대해 더 깊은 이해를 제공합니다. 특히, 학습 작업의 구성적 구조(compositional structure)를 중시하며, 다중 지수 모델(multi-index model, MIM)을 통해 이를 구체화합니다. 또한, 하이퍼 커널 리지 회귀(hyper-kernel ridge regression, HKRR)를 도입하여 뉴럴 네트워크와 커널 방법의 융합 접근을 연구합니다.

- **Technical Details**: HKRR는 선형 변환과 매끄러운 비선형 함수의 조합으로 구성된 다중 지수 모델을 학습하는 데 적합한 프레임워크를 제공합니다. 이 방법은 커널 리지 회귀(kernel ridge regression, KRR)의 일반화로, 주어진 변환을 학습함에 따라 최적의 솔루션을 제공합니다. 두 가지 최적화 방법인 VarPro와 AGD를 비교하며, HKRR의 최적화 문제를 이론적으로 분석하였습니다.

- **Performance Highlights**: 실험 결과, AGD 방법이 VarPro 방법보다 더 안정적이며 우수한 성능을 보이는 것으로 나타났습니다. HKRR는 고차원 문제에서의 샘플 복잡도를 줄이며, 이론적인 결과와 실험 데이터를 모두 통해 성능을 입증했습니다. 이러한 결과들은 HKRR이 커널 방법의 유용한 확장으로서 구성적 학습 및 표현 학습 모델에 대한 기존 알고리즘적 접근법을 쇄신할 수 있음을 시사합니다.



### Multimodal Function Vectors for Spatial Relations (https://arxiv.org/abs/2510.02528)
- **What's New**: 이번 논문에서는 Large Multimodal Models (LMMs)이 어떻게 공간 관계의 표현을 전송하는지에 대한 새로운 통찰을 제공합니다. OpenFlamingo-4B와 같은 비전-언어 모델의 특정 attention heads가 관계 예측에 중대한 영향을 미친다는 것을 발견했습니다. 우리는 causal mediation analysis를 활용하여 이러한 attention heads와 multimodal function vectors를 식별하고, 이를 통해 LMM의 성능을 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: 기술적으로, 연구진은 function vectors (FVs)라는 접근 방식을 채택하여 LMM 내부의 공간 관계를 인코딩하는 구조를 추출하고 조작합니다. 함수 벡터는 적은 수의 attention heads에서 활성화된 값을 평균하여 생성하며, 모델의 은닉층에 삽입될 수 있습니다. 이를 통해 주어진 작업 이전에 시연 없이도 모델이 원하는 행동을 생성할 수 있게 됩니다.

- **Performance Highlights**: 결과적으로, 우리는 fine-tuned function vectors가 LMM의 in-context learning 기준보다 월등한 성능을 발휘한다는 것을 증명했습니다. 또한, 관계-specific function vectors를 선형 결합해 새롭고 훈련되지 않은 공간 관계를 해결하는 데 성공하여, 이 접근 방식의 강력한 일반화 능력을 강조합니다. 이러한 발견은 LMM의 관계 추론 능력에 대한 이해를 선진화시키는 데 기여합니다.



### Self-supervised diffusion model fine-tuning for costate initialization using Markov chain Monte Carlo (https://arxiv.org/abs/2510.02527)
- **What's New**: 이 논문에서는 긴 시간 동안의 저추력 우주선 궤적 최적화 문제를 위한 새로운 접근 방식을 제안합니다. 조건부 확산 모델과 마르코프 체인 몬테 카를로(Markov Chain Monte Carlo, MCMC) 알고리즘을 결합하여 파레토 최적 해법을 찾는 데 필요한 초기 추정값 문제를 해결합니다. 이를 통해 기존 프레임워크의 한계를 극복하고, 더 효율적인 데이터 생성 및 샘플 품질 개선을 촉진할 수 있는 방법을 마련하였습니다.

- **Technical Details**: 기존의 AmorGS 프레임워크를 발전시켜, 조건부 확산 모델을 이용하여 다층적인 최적 해법 분포를 제시합니다. MCMC 방법론을 적용하여 새로운 데이터 샘플을 생성하고, 이를 초할당(MCMC) 메트로폴리스 알고리즘에 의해 세부 조정합니다. 이 과정에서 효율적으로 제약을 만족하는 해를 찾기 위한 보상 기반 훈련(train) 방법을 사용하여 샘플을 개선합니다.

- **Performance Highlights**: 논문에서는 두 가지 사례 연구를 통해 제안된 프레임워크의 성능을 입증합니다. 첫 번째 사례에서는 목성-유로파 원형 제한 삼체 문제에서 부분적인 파레토 프론트를 완성하는 데 성공하였고, 두 번째 사례에서는 토성-타이탄 간 전송을 위해 밀집하고 우수한 파레토 프론트를 생성하는 것을 보여줍니다. 이러한 연구는 우주 탐사 및 궤적 최적화 문제를 해결하기 위한 중요한 이정표가 될 것입니다.



### Unraveling Syntax: How Language Models Learn Context-Free Grammars (https://arxiv.org/abs/2510.02524)
Comments:
          Equal contribution by LYS and DM

- **What's New**: 이 논문에서는 언어 모델(LLMs)이 구문(syntax)을 어떻게 습득하는지를 이해하기 위한 새로운 프레임워크를 제안합니다. 이전 연구들은 훈련된 모델의 정적 행동을 분석하는 데 중점을 두었지만, 이 연구는 언어 습득 과정에 대한 이해를 목표로 합니다. 저자들은 확률적 문맥 자유 문법(PCFGs)를 기반으로 한 합성 언어(synthetic languages)를 통해 작고 통제 가능한 모델에서의 학습 동역학을 연구합니다.

- **Technical Details**: 연구의 주요 기여는 언어 모델이 간단한 하위 구조(subgrammar)를 먼저 습득하고 더 복잡한 구문으로 나아가는 과정을 조사하는 것입니다. 저자는 여러 가지 일반적인 재귀 공식(general recursive formulae)을 증명하고, PCFG의 하위 문법 구조에 대한 훈련 손실(training loss)과 쿨백-라이블러 발산(KL divergence)을 조사합니다. 실험적으로 저자들은 트랜스포머가 모든 하위 문법에서 손실을 병렬로 줄이는 것을 발견했습니다.

- **Performance Highlights**: 결론적으로, 모델들은 깊은 재귀 구조에서 어려움을 겪으며, 이는 대형 언어 모델에도 있어 한계입니다. 저자들은 모델들이 긴 문맥에서 고정된 깊이에서는 잘 작동하지만 재귀 깊이가 증가할 때 급격히 실패한 것을 보여주었습니다. 또한, 저자들은 하위 문법의 사전 훈련이 작은 모델의 최종 손실을 개선할 수 있음을 입증하고, 사전 훈련된 모델이 하위 구문 구조와 더 정렬된 내부 표현을 개발하는 것으로 나타났습니다.



### Adaptive randomized pivoting and volume sampling (https://arxiv.org/abs/2510.02513)
Comments:
          13 pages, 2 figures

- **What's New**: 이 논문은 Adaptive Randomized Pivoting (ARP) 알고리즘에 대한 새로운 해석을 제공합니다. ARP는 최근에 제안된 매우 효과적인 알고리즘으로, 열 서브셋 선택(column subset selection)에 사용됩니다. 이 논문은 ARP 알고리즘을 볼륨 샘플링 분포(volume sampling distribution)와 선형 회귀(linear regression) 위한 능동 학습(active learning) 알고리즘과 연결지어 재해석합니다.

- **Technical Details**: ARP 알고리즘에 대한 새로운 분석이 제공되며, 거부 샘플링(rejection sampling)을 이용한 빠른 구현(faster implementations) 방법도 제시됩니다. 이러한 연결은 알고리즘의 이론적 근거를 강화하고 실제 적용 가능성을 높입니다. 이는 노이즈가 있는 데이터 환경에서도 유용하게 사용될 수 있음을 시사합니다.

- **Performance Highlights**: 새로 제안된 분석을 통해 ARP 알고리즘의 효율성이 입증되었습니다. 또한, 빠른 구현 방법은 대규모 데이터 세트에 대한 처리 속도를 개선할 수 있는 가능성을 보여줍니다. 이번 연구는 ARP 알고리즘이 열 서브셋 선택에서의 성능을 더욱 향상시키는 데 기여할 것입니다.



### Beyond Linear Diffusions: Improved Representations for Rare Conditional Generative Modeling (https://arxiv.org/abs/2510.02499)
- **What's New**: 이 논문에서는 조건부 분포 $P(Y|X=x)$를 모델링하는 데 있어 만나는 주요한 도전 과제에 주목합니다. 특히, $P(X=x)$가 낮은 지역에서 추가 샘플이 부족한 문제를 해결하기 위해 데이터 표현(data representation) 및 확산 기법(forward scheme)을 조정할 수 있음을 제안합니다. 이를 통해 희귀한 조건의 샘플 복잡성(sample complexity)을 줄일 수 있는 방법론을 제시하고 있습니다.

- **Technical Details**: 조건부 확산 모델은 샘플링을 노이즈 프로세스의 시간 역전으로 정의합니다. 이 논문은 Langevin 확산을 사용하여 연속 시간 차분 방정식(SDE)을 다루며, 조건부 확률 측정치와 관련된 복잡한 함수 장애 요인을 분석합니다. 또한, 희귀한 지역에서의 정확한 점수 함수(score function) 추정을 위한 새로운 방법론을 제안하며, 데이터 변환과 비선형 Langevin 확산과 관련된 노력을 강조합니다.

- **Performance Highlights**: 제안된 방법은 고급 실험을 통해 두 개의 합성 데이터 세트 및 실제 금융 데이터 세트에서 검증되었습니다. 결과적으로, 제안된 꼬리 적응형(tail-adaptive) 접근법은 극단적인 조건에서도 응답 분포를 정확하게 포착하는 데 있어 기존의 표준 확산 모델보다 현저한 성능 개선을 보여주었습니다. 이러한 결과는 특히 금융 위험 평가 및 기후 모델링과 같은 중요 분야에 중요한 기여를 합니다.



### Safe and Efficient In-Context Learning via Risk Contro (https://arxiv.org/abs/2510.02480)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)의 안전성을 높이기 위한 새로운 접근법을 제안합니다. 특히, 부정확하거나 악의적인 예시가 모델 성능에 미치는 영향을 최소화하고자 합니다. Zero-shot 모델을 기준으로 하여, Distribution-Free Risk Control (DFRC)를 적용하여, 불리한 입력이 성능을 저하시킬 수 있는 정도를 제어합니다.

- **Technical Details**: 연구는 입력 x에 대한 레이블 y를 바탕으로 LLM이 상황별 예시 세트(c)를 통해 학습하는 과정을 다룹니다. LLM이 모든 레이어를 통과한 후 예측을 하지만, Early-Exit LLM에서는 각 레이어에서 예측할 수 있는 선택지를 제공합니다. 이를 통해 위험 제어 프레임워크를 적용하여 적절한 예측 기준을 선택할 수 있도록 합니다. 또한, 안전성을 높이기 위해 세 가지 기여를 제안합니다: 안전한 zero-shot 기준을 사용하는 초기 종료 모델의 새로운 구성, 과도한 학습(overthinking)을 측정하는 새로운 ICL(loss) 손실 설계, 그리고 안전성과 효율성을 모두 고려한 Learn-then-Test (LTT) 위험 제어 프레임워크의 단순한 수정입니다.

- **Performance Highlights**: 연구 결과는 제안된 방법이 부정적인 상황에서도 안전성이 보장되고 도움이 되는 예시에서 성능을 극대화할 수 있음을 보여줍니다. 실험을 통해 성능 속도를 50% 이상 향상시키면서도, zero-shot 성능 기준에 비해 출력 안전성을 보장할 수 있음을 입증하였습니다. 8개의 다양한 벤치마크 작업과 4개의 모델을 통한 실험 결과, 이 프레임워크가 안전성을 효과적으로 보장하고 높은 효율성을 달성할 수 있는 첫 사례로 확인되었습니다.



### Heterogeneous Graph Representation of Stiffened Panels with Non-Uniform Boundary Conditions and Loads (https://arxiv.org/abs/2510.02472)
Comments:
          This is a preprint and has been submitted to Engineering with Computers

- **What's New**: 이 논문에서는 기하학적 변동성, 비균일 경계 조건 및 다양한 하중 시나리오를 고려한 강화를 밴난 패널(기계 구조물)의 이질적인 그래프 표현 방식을 제안합니다. 이질적인 그래프 신경망(HGNNs)을 활용하여 강성이 있는 패널을 여러 구조 단위로 나누고 각각의 단위를 geometry, boundary, loading 노드의 세 가지 유형으로 표현합니다. 또한, 연결된 노드의 지역 방향과 공간적 관계를 포함하여 엣지 이질성을 도입했습니다.

- **Technical Details**: 이 시스템은 적응적 구조에 대해 비균일 하중 및 경계 조건을 효과적으로 캡처하며, 다양한 이질적 그래프 표현이 제안되고 분석되었습니다. 각 구조 단위 간의 상호작용을 정확하게 모델링하기 위해 엣지(Edge) 이질성과 함께 HGT(이질적 그래프 변환기)를 통해 von Mises 응력과 변위를 예측하는 데 중점을 두고 있습니다. 실험 통계는 다양한 하중 조건 하의 패치 하중이 적용된 패널에 대해 수행되었습니다.

- **Performance Highlights**: 제안된 이질적인 그래프 표현은 기존의 동질적인 그래프 모델보다 우수한 성능을 보여주었습니다. 이 모델은 변위와 von Mises 응력에 대해 강력한 예측 정확도를 나타내며, 구조적 거동 패턴과 최대 값을 효과적으로 포착합니다. 특히, 이질적 그래프 모델을 사용하여 판 구조물의 하중과 경계에서 발생하는 비균일 변화를 잘 모델링할 수 있습니다.



### Predictive inference for time series: why is split conformal effective despite temporal dependence? (https://arxiv.org/abs/2510.02471)
Comments:
          22 pages

- **What's New**: 이번 연구에서는 시계열 데이터에 대한 예측의 불확실성을 정량화하는 문제를 다루고 있습니다. 특히, 과거 데이터를 사용하여 다음 시점을 예측하는 경우, 예측값 주위에 유효한 예측 구간을 제시할 수 있는지를 검토합니다. 특히, split conformal prediction 방법을 통해 기존의 데이터 분포에 대한 가정 없이도 이 문제를 해결할 수 있음을 보여줍니다.

- **Technical Details**: 이 논문에서는 split conformal prediction의 이론적 특성을 시계열 설정에서 분석합니다. 특히, 예측기가 '메모리'(memory)를 가질 때의 사례도 포함됩니다. 연구 결과는 시계열 내의 시간적 의존성이 교환 가능성(exchangeability) 가정을 위반하는 정도를 측정하는 새로운 '스위치 계수'(switch coefficient)에 따라 이러한 방법의 커버리지 손실을 제한합니다.

- **Performance Highlights**: 가장 놀라운 점은, 전통적으로 시계열 데이터가 교환 가능성을 명백히 위반하지만, split conformal prediction 알고리즘이 낮은 계산 비용과 효과적인 성능 덕분에 여전히 불확실성 정량화에 매력적인 선택이라는 것입니다. 논문은 이러한 알고리즘이 시계열 설정에서 강력한 성능을 보이는 이유를 설명하며, 데이터에 대한 강한 가정을 두지 않고 예측 구간을 구성하는 방법을 제시합니다.



### Words That Make Language Models Perceiv (https://arxiv.org/abs/2510.02425)
- **What's New**: 이번 연구에서는 언어 모델(LLMs)이 텍스트만 학습하더라도 시각 및 청각 인코더와의 표현 정합성을 높일 수 있음을 보여주었습니다. 연구진은 명시적인 감각 프롬프트가 LLM의 잠재적 구조를 표출시킬 수 있다는 가설을 검증하였습니다. 예를 들어, 모델이 '보다(see)' 또는 '듣다(hear)'라고 지시받을 때, 이러한 프롬프트가 시각적 또는 청각적 증거에 기초하여 다음 단어 예측에 영향을 미친다는 것을 발견하였습니다.

- **Technical Details**: 연구에서는 텍스트만으로 학습된 LLM의 표현 기하학을 검사하여 이를 단일 모드 비전 및 오디오 인코더와 유사하게 변화시킬 수 있는 방법을 다루었습니다. 새로운 개념으로 생성적 표현을 도입하여, LLM이 생성하는 각 출력은 정해진 프롬프트와 지금까지 생성된 시퀀스의 함수로서, 추가적인 전진 패스를 포함합니다. 감각 프롬프트를 통해 이러한 생성적 표현을 제어할 수 있으며, 이는 더 높은 정합성으로 이어집니다.

- **Performance Highlights**: 연구 결과에 따르면, 단일 감각 단어가 포함된 프롬프트는 텍스트만 학습된 LLM의 커널을 감각 인코더의 기하학에 더 가깝게 이동시킬 수 있음을 보여줍니다. 생성의 길이가 길어질수록 표현의 유사성이 증가하며, 더 큰 모델일수록 감각 프롬프트에 대한 정합성이 높아지는 경향이 있습니다. 또한, 시각적 단서가 있을 경우 LLM이 VQA(Visual Question Answering)에서 더 나은 성능을 발휘하는 것으로 나타났습니다.



### Adaptive Deception Framework with Behavioral Analysis for Enhanced Cybersecurity Defens (https://arxiv.org/abs/2510.02424)
Comments:
          5 pages, 5 tables, 1 figure

- **What's New**: 본 논문은 CADL(인지-적응형 속임수 레이어)라는 유연한 속임수 프레임워크를 소개합니다. CICIDS2017 데이터셋에서 99.88%의 탐지율과 0.13%의 잘못된 긍정율을 달성하는 이 시스템은 앙상블 머신러닝(Random Forest, XGBoost, Neural Networks)과 행동 프로파일링을 결합하여 네트워크 침입에 대응합니다. 구현을 오픈 소스로 제공함으로써 고급 방어 능력에 대한 접근성을 민주화하고, 기존의 상업적 속임수 플랫폼보다 비용적인 장점을 제공합니다.

- **Technical Details**: CADL 시스템은 네 가지 주요 구성 요소로 이루어져 있습니다: (1) Random Forest, XGBoost 및 Neural Networks를 결합한 Ensemble Detector, (2) 행동 분석 및 적응형 속임수를 위한 CADL, (3) 구성 요소 조정을 위한 Signal Bus, (4) 결정 합성을 위한 Response Orchestrator입니다. 이 시스템은 비선형 관계를 학습하기 위해 다층 퍼셉트론 구조를 사용하여 높은 정확도를 달성하고, 신호 버스를 통해 실시간 정보를 공유하여 적응형 대응을 가능하게 합니다.

- **Performance Highlights**: CADL은 50,000개의 테스트 샘플을 바탕으로 99.92%의 호출율을 기록하며, 40,215개의 정상 샘플 중 51개의 잘못된 긍정 결과를 나타내었습니다. 이는 Snort 대비 28.7%, Suricata 대비 31.4%, ModSecurity 대비 37.6%의 탐지 개선률을 보여줍니다. 평균 처리 시간은 샘플당 34.7ms로, Intel i5-12400F에서 1초당 1,200개의 요청을 처리할 수 있는 지속 가능한 속도를 자랑합니다.



### Higher-arity PAC learning, VC dimension and packing lemma (https://arxiv.org/abs/2510.02420)
Comments:
          12 pages, 1 figure

- **What's New**: 본 논문은 Chernikov, Towsner'20의 연구를 통해 개발된 고차원 VC 이론(VC_n dimension)의 개요를 제공합니다. Haussler 포장 렘마의 일반화 및 관련된 하이퍼그래프 정규성 렘마에 대해 다루고 있으며, Kobayashi, Kuriyama 및 Takeuchi'15가 도입한 곱 측도에 대한 n-겹 곱 공간에서의 PAC 학습(이해력 PAC_n learning)을 특성화하는 방법을 설명합니다.

- **Technical Details**: 고차원 VC 이론이나 VCk 이론의 기초는 kk 곱 공간에서의 부분 집합 가족을 다룹니다. 이 논문에서는 Shelah의 kk-종속 이론에서 암시된 VCk 차원의 개념을 논의하고, VCk 차원에서의 Sauer-Shelah 렘마의 적절한 일반화를 확립했습니다. 또한 고차 PAC 학습을 위한 새로운 일반화를 제안하였으며, 하나의 방향에서 우리의 포장 렘마가 VCk 정규성을 유도함을 보여주었습니다.

- **Performance Highlights**: 논문에서 제시한 주요 결과는 유한한 VCk 차원과 PACk 학습 가능성의 동치성을 입증하는 것입니다. 이러한 동치 관계는 잦은 상관 모델에서 다룬 VCk 차원 포장 렘마를 통해 증명되었습니다. 이와 같은 결과는 수학적 과학 및 응용학회에서 발표되었으며, 최근 고차원 VC 이론에 대한 관심이 높아지고 있는 점도 주목할 만합니다.



### BrowserArena: Evaluating LLM Agents on Real-World Web Navigation Tasks (https://arxiv.org/abs/2510.02418)
- **What's New**: 최근 LLM 웹 에이전트가 개방형 웹에서 작업을 수행할 수 있도록 발전하며, BrowserArena라는 새로운 평가 플랫폼이 도입되었습니다. 이 플랫폼은 사용자 제출 작업을 기반으로 하여 에이전트 성능을 실시간으로 평가하며, Arena 스타일의 헤드 투 헤드 비교를 통해 에이전트의 실패 모드를 식별합니다. 이를 통해 사용자는 에이전트가 실제 웹 작업에서 어떻게 수행되는지를 보다 정확히 분석하고 이해할 수 있습니다.

- **Technical Details**: BrowserArena는 다양한 사용자 제출 작업을 처리하고, 두 개의 무작위로 선택된 LLM을 사용하여 웹사이트를 탐색합니다. 이 플랫폼은 Chatbot Arena을 기반으로 하며, 사용자로부터 수집한 피드백을 통해 특정 단계에서 LLM의 행동을 검토합니다. 주요 실패 모드로는 captcha 해결, 팝업 배너 제거, URL로의 직접 탐색이 있으며, 다양한 언어 모델이 이 실패 모드를 해결하기 위해 사용하는 전략이 다름을 발견했습니다.

- **Performance Highlights**: 연구 결과, o4-mini는 captcha 해결에서 더 다양한 전략을 구사하지만, DeepSeek-R1은 사용자를 잘못 안내하는 경향이 발견되었습니다. 이 연구는 웹 에이전트의 성능을 평가하는 새로운 접근 방식을 마련하며, 공개된 데이터셋 및 코드베이스를 통해 LLM의 성능 평가에 기여할 것입니다. 또한, VLM(vision-language models)의 인간 선호 모델링 능력의 한계를 드러내며, 사용자 피드백을 통한 세밀한 행동 분석의 중요성을 강조합니다.



### NEURODNAAI: Neural pipeline approaches for the advancing dna-based information storage as a sustainable digital medium using deep learning framework (https://arxiv.org/abs/2510.02417)
- **What's New**: 이번 연구에서는 DNA 데이터 저장을 위한 새로운 모듈형 엔드 투 엔드 프레임워크 NeuroDNAAI를 제안합니다. 이 프레임워크는 양자 병렬성(quantum parallelism) 개념을 활용하여 인코딩의 다양성과 강인성을 향상시키고, 생물학적 제약조건 및 딥러닝을 통합하여 DNA 저장에서의 오류 완화를 개선합니다. NeuroDNAAI는 이진 데이터 스트림을 상징적 DNA 시퀀스에 인코딩하고, 불완전한 채널을 통해 전송 후 높은 정확도로 재구성합니다.

- **Technical Details**: 이 시스템은 디지털 정보를 DNA 시퀀스에 인코딩하고, 합성 및 시퀀싱 오류를 시뮬레이션하는 변환 가능한 노이즈 모델을 통해 전송한 후, 인코더-디코더 아키텍처로 재구성합니다. 전통적인 부호 이론(coding theory) 접근 방식과 달리, 변환기(Transformer) 기반 모델은 오류 패턴을 직접 학습하고 다양한 노이즈 수준에 적응하는 구조를 제공하여, 삽입-삭제 오류 처리에 강력한 성능을 자랑합니다.

- **Performance Highlights**: NeuroDNAAI는 기존의 팅킹 방법이나 규칙 기반 접근 방식이 실제 노이즈를 효과적으로 처리하지 못하는 반면, 높은 정확도로 결과를 달성했습니다. 실험 결과는 텍스트와 이미지 모두에서 낮은 비트 오류율(bit error rate)을 보였으며, 이 프레임워크는 이론적, 실용적, 시뮬레이션의 통합을 통해 스케일러블한 생물학적으로 유효한 DNA 저장을 가능하게 합니다.



### The Equilibrium Response of Atmospheric Machine-Learning Models to Uniform Sea Surface Temperature Warming (https://arxiv.org/abs/2510.02415)
- **What's New**: 최근에 자율적으로 안정적인 다년간 지구 기후 시뮬레이션을 생성할 수 있는 기계 학습(ML) 모델이 개발되었습니다. 이 연구에서는 다양한 최첨단 ML 모델(ACE2-ERA5, NeuralGCM, cBottle)이 균일한 해양 표면 온도 상승에 대한 기후 반응을 평가합니다.

- **Technical Details**: 이 연구에서는 세 가지 ML 기반 모델의 기후 응답을 비교하며, 여기에는 cBottle, ACE2, NeuralGCM이 포함됩니다. 이 모델들은 지구 물리학적 방정식 모델인 GFDL의 AM4 모델과 비교하여 성능이 평가됩니다. 평가 기준으로는 지표 공기 온도, 강수량, 온도 및 바람 프로파일, 대기 상단 방사선 등이 사용됩니다.

- **Performance Highlights**: 모든 ML 모델은 AM4와 ERA5에 비해 표면 공기 온도의 평균 기후를 잘 재현합니다. 그러나 cBottle은 육지의 따뜻함을 과소평가하는 경향이 있으며, NeuralGCM은 육지 및 극 지역에서의 따뜻함을 잘 포착합니다. 이 결과는 ML 모델이 기후 변화 응용 프로그램에 잠재력을 가지고 있지만, 더 나은 일반화를 위해 개선이 필요하다는 것을 강조합니다.



### Linear RNNs for autoregressive generation of long music samples (https://arxiv.org/abs/2510.02401)
- **What's New**: 본 연구에서는 HarmonicRNN이라는 모델을 제안하여 오디오 파형을 자율 회귀 방식으로 생성하는 문제를 해결하고자 한다. 이전의 회귀 신경망(recurrent neural networks) 접근법의 한계를 극복하기 위해, 선형 RNN(Linear RNN)을 활용하고 특히 1M 토큰까지의 긴 시퀀스를 처리할 수 있는 컨텍스트 병렬성(context-parallelism)을 도입하였다. 이 모델은 소규모 데이터셋에서 최첨단의 로그 우도(log-likelihood) 및 지각(bottom-up) 지표를 달성하였다.

- **Technical Details**: HarmonicRNN은 오디오 데이터에 최대 우도 훈련(maximum likelihood training)을 직접 적용하는 방식으로 훈련되며, 이는 미니배치(mini-batch)를 활용한 확률적 경사 상승법(stochastic gradient ascent)으로 이루어진다. 모델은 조건부 확률 분포를 계산하는 깊은 선형 RNN을 사용하며, 각 시간 단계tt에 따라 xt에 대한 범주형 분포를 출력한다. 이러한 구조는 메모리 효율성을 가지고 있으며, CG-LRU(complex gated linear recurrent unit) 아키텍처를 기반으로 한다.

- **Performance Highlights**: 실험 결과, HarmonicRNN은 로그 우도를 비롯한 다양한 지각 측정(perceptual metrics)에서 이전 모델에 비해 성능이 향상되었다. SC09, Beethoven, YouTubeMix라는 세 가지 데이터셋에서 우수한 결과를 보였으며, 특히 1분 분량의 긴 오디오 샘플에 대해서도 효과적으로 모델링할 수 있는 성능을 입증하였다. 이 모델은 TPU v4-8 및 v3-128 장치에서 훈련되었으며, 멀티 호스트 훈련을 통해 메모리 요구사항을 충족하였다.



### LLM-Generated Samples for Android Malware Detection (https://arxiv.org/abs/2510.02391)
Comments:
          24 pages

- **What's New**: 이 연구에서는 Android 악성코드의 탐지를 위해 GPT-4.1-mini를 미세 조정하여 BankBot, Locker/SLocker, Airpush/StopSMS의 구조화된 기록을 생성했습니다. 기존의 부족한 데이터 문제를 해결하기 위해 합성 데이터(synthetic data)의 역할을 다루었으며, 대형 언어 모델(Large Language Models, LLMs)을 악성코드 탐지 작업에 활용하는 접근법을 도입했습니다.

- **Technical Details**: KronoDroid 데이터셋을 사용하여 다양한 분류기(classifiers)를 평가하였고, 실제 데이터(real data)만으로 학습한 경우와 합성 데이터(synthetic data)를 포함한 경우, 그리고 합성 데이터만으로 학습한 경우에 대해 분석했습니다. 생성 불일치 문제는 프롬프트 엔지니어링(prompt engineering) 및 후처리(post-processing)를 통해 해결되었습니다.

- **Performance Highlights**: 실제 데이터만으로 학습했을 때 거의 완벽한 탐지 성능을 보였고, 합성 데이터를 추가로 사용해도 높은 성능을 유지했습니다. 그러나 합성 데이터만으로 학습할 경우 효과가 악성코드 종류와 미세 조정 전략에 따라 다르게 나타났습니다. 연구 결과는 LLM이 생성한 악성코드가 데이터 부족 문제를 해결하는 데 도움이 될 수 있지만, 독립적인 교육 소스로는 부족하다는 점을 시사합니다.



### From Trace to Line: LLM Agent for Real-World OSS Vulnerability Localization (https://arxiv.org/abs/2510.02389)
- **What's New**: 이 논문은 T2L-Agent(Trace-to-Line Agent)라는 새로운 엔드 투 엔드 프레임워크를 소개합니다. 이는 모듈에서 취약한 코드 줄로 점진적으로 범위를 좁히는 방식으로, 런타임 증거와 AST 기반 코드 청크를 결합하여 연속적인 개선을 가능하게 합니다. 이를 통해 소프트웨어 개발에서 정확한 코드 라인 레벨 진단과 패치가 가능합니다.

- **Technical Details**: T2L-Agent는 측정 가능한 결과를 위한 두 단계 작업을 제안합니다: (a) 코드 청크의 거시적 탐지와 (b) 정확한 취약한 코드 라인의 미세 탐지입니다. 이 시스템은 여러 도구들을 통합하여 행위 관찰 가능성을 높이고, 코드 분석을 위한 동적 피드백을 포함하여 취약성을 가설화하고 테스트하는 제안 모듈도 기능합니다. T2L-ARVO 벤치마크는 50개의 전문가 검증 사례로 구성되어 있으며 다양한 취약성 유형을 포함합니다.

- **Performance Highlights**: T2L-Agent는 T2L-ARVO에서 최대 58%의 청크 탐지 및 54.8%의 정확한 라인 탐지 성과를 달성하여 기존 기준치를 뛰어넘습니다. 이 과정에서 지속적인 피드백 기반 워크플로우를 통해 엔지니어들이 실제로 디버깅하는 방식과 유사한 방식으로 작동하며, 대규모 코드베이스에서도 효율적으로 스케일링됩니다. 이는 오픈 소스 소프트웨어 개발에서 패치 속도를 높이고 노이즈를 줄이는 데 크게 기여합니다.



### CWM: An Open-Weights LLM for Research on Code Generation with World Models (https://arxiv.org/abs/2510.02387)
Comments:
          58 pages

- **What's New**: Code World Model (CWM)은 320억 개 매개변수를 가진 새로운 오픈 가중치(Weights) LLM으로, 코드 생성 및 세계 모델링(세계 모델을 통한 문제 해결)에 대한 연구를 촉진하기 위해 출시되었습니다. CWM은 대규모의 Python 인터프리터와 Docker 환경에서의 관찰-행동(Observation-Action) 경로를 활용하여 중간 학습이 이루어졌습니다. 이를 통해 코드 이해도를 개선하고, 다중 작업(multi-task) 사고가 필요한 소프트웨어 엔지니어링 환경에서 강력한 테스트베드를 제공합니다.

- **Technical Details**: CWM은 32B 매개변수를 가진 밀집형 디코더 전용 LLM으로, 최대 131k 토큰의 컨텍스트 크기를 지원하는 슬라이딩 윈도우 주의(attention) 기법을 사용합니다. Python 코드 실행 데이터와 에이전트 상호작용을 기반으로 한 대규모 시뮬레이션 데이터로 중간 학습하여, 코드의 구문(Syntax) 뿐만 아니라 의미(Semantics)를 이해하도록 설계되었습니다. 이 모델은 또한 강화 학습(RL)을 통해 더욱 발전할 수 있는 기초가 마련되어 있습니다.

- **Performance Highlights**: CWM은 일반적인 코딩 및 수학 작업에서 우수한 성능을 발휘하며, SWE-bench Verified에서 65.8%, LiveCodeBench에서 68.6%, Math-500에서 96.6%, AIME 2024에서 76.0%의 pass@1 점수를 기록했습니다. 이러한 성과는 CWM이 에이전트 코드 생성 및 추론(task) 작업에서도 강력한 능력을 지니고 있음을 나타냅니다. 또한, 연구자들을 위해 모델 체크포인트와 최종 가중치가 비상업적 연구 라이센스 하에 공개됩니다.



### On The Fragility of Benchmark Contamination Detection in Reasoning Models (https://arxiv.org/abs/2510.02386)
- **What's New**: 본 논문은 대형 추론 모델(Large Reasoning Models, LRMs)의 벤치마크 오염 문제에 대한 체계적인 연구를 최초로 제시합니다. 연구 결과에 따르면, LRMs는 모델 개발자가 벤치마크 데이터를 훈련 데이터에 포함시켜 성능을 부풀리는 '벤치마크 오염(contamination)'에 매우 취약하다는 사실이 밝혀졌습니다. LRMs는 체인 오브 띵킹(Chain-of-Thought, CoT) 추론 방식을 활용하지만, 이 과정에서 검출이 어렵다는 문제에 직면해 있습니다. 따라서 LRMs의 공정한 평가를 위한 신뢰할 수 있는 평가 프로토콜 개발의 필요성이 강조됩니다.

- **Technical Details**: LRMs의 벤치마크 오염 문제를 다룬 이 연구는 두 가지 주요 단계로 나뉩니다. 첫 번째 단계는 SFT(Supervised Fine-Tuning)와 RL(Reinforcement Learning) 과정 중에 발생하는 오염을 조사하며, 두 번째 단계는 CoT를 적용한 고급 LRM의 최종 SFT 단계에서의 오염을 다룹니다. 연구 결과는 K-시뮬레이션을 통해 기존의 오염 검출 방법이 LRMs의 오염을 발견하기 어려움이 드러났습니다. 특히 GRPO(Generalized Reinforcement Policy Optimization) 훈련이 오염 신호를 숨기는 주된 원인으로 지목되었습니다.

- **Performance Highlights**: 연구 결과, LRMs는 고유의 오염 문제로 인해 기존의 벤치마크 오염 검출 방법이 거의 무의미해졌습니다. CoT를 통한 오염이 발생했을 때, 대부분의 검출 방법은 무작위 추측 수준의 성능으로 떨어졌습니다. 이는 LRMs가 벤치마크 데이터를 기억해야만 검출된다는 기존 가정이 틀렸음을 나타내며, LRMs의 신뢰성과 공정성을 위협합니다. 마지막으로, 이 연구는 LRMs 평가의 무결성을 보장하기 위해 고급 오염 검출 방법의 개발 필요성을 강조합니다.



### Uncertainty-Aware Answer Selection for Improved Reasoning in Multi-LLM Systems (https://arxiv.org/abs/2510.02377)
- **What's New**: 이번 연구에서는 다수의 LLM에서 가장 신뢰할 수 있는 응답을 선택하는 새로운 방법을 제안하고 있습니다. 저자들은 기존 방법들이 자주 사용되는 외부 검증 도구나 인간 평가자에 의존하는 문제를 해결하고자 하였으며, 자원 제약이 있는 상황에서도 효과적으로 작동하는 기법을 소개하고 있습니다. 새로운 접근 방식은 조정된 로그-우도(log-likelihood) 점수를 사용하여, 다양한 LLM의 고유한 지식과 신뢰성을 활용하는 점이 특징입니다.

- **Technical Details**: 저자들은 다중 LLM 시스템에서 응답을 집계하고 최적의 답변을 선택하기 위해 로그-우도 기반의 점수를 사용합니다. 이들은 각 모델의 파라미터와 로지트 분포가 다르기 때문에, 직접적인 비교가 이론적으로 부적절하다는 점을 강조하며, 고유한 신뢰성 점수를 조정하여 활용합니다. 이 방법은 전방 패스로 조정된 점수를 계산함으로써 비싼 자율 회귀 디코딩을 피하고, 외부 검증자나 별도의 보상 모델이 필요하지 않습니다.

- **Performance Highlights**: 제안한 방법은 GSM8K, MMLU(6개 하위 집합), ARC 데이터셋에서 각각 약 4%, 3%, 5%의 성능 향상을 보여주었습니다. 다중-LLM 설정에서 단일 LLM보다 성능 개선이 더 두드러졌으며, 다양한 응답을 생성하는 다중-LLM 시스템의 잠재력을 더욱 드러냈습니다. 연구 결과는 이 새로운 접근 방식이 응답 품질 향상에 기여할 수 있음을 강하게 시사합니다.



### Pretraining with hierarchical memories: separating long-tail and common knowledg (https://arxiv.org/abs/2510.02375)
- **What's New**: 이 연구는 현대 언어 모델의 성능 향상에 대한 새로운 접근법을 제시합니다. 기존의 방법이 매개변수(parameter)의 규모 확대에 의존하고 있는 반면, 우리는 작은 언어 모델이 큰 계층적(parametric) 메모리 뱅크를 활용하여 폭넓은 세계 지식을 접근할 수 있도록 하는 메모리 증강 아키텍처를 개발하였습니다. 이를 통해 필요한 부분만을 메모리에서 가져와 모델에 추가합니다.

- **Technical Details**: 제안된 메모리 증강 아키텍처는 문맥에 따라 필요한 지식 매개변수를 모델에 연결하여, 실행시간 효율성을 높입니다. 또한, 훈련 시 메모리 파라미터가 유사한 주제의 시퀀스에서만 활성화되고 업데이트됨으로써, 파라미터의 망각(catastrophic forgetting)을 줄이고 보다 효과적으로 장기 지식을 기억할 수 있습니다. 이러한 혁신적인 접근방식은 훈련 효율성을 높이고, 분산 훈련을 더욱 간편하게 합니다.

- **Performance Highlights**: 우리는 실험을 통해 제안된 모델이 기본 모델보다 성능 향상을 보여주며, 매개변수 수가 2배 이상인 일반 모델과 유사한 성능을 낸다는 것을 입증했습니다. 예를 들어, 4.6B 메모리 뱅크에서 가져온 18M 매개변수의 메모리가 보강된 160M 모델이 정규 모델과 유사한 성능을 발휘합니다. 이러한 성능 개선은 메모리 유형, 깊이, 크기 그리고 메모리와 모델의 비율 등에 따라 달라지며, 우리는 각 요소가 성능에 미치는 영향을 체계적으로 분석하였습니다.



### An Encoder-Decoder Network for Beamforming over Sparse Large-Scale MIMO Channels (https://arxiv.org/abs/2510.02355)
Comments:
          13 pages, 9 figures, submitted to TCOM and is waiting for reviews

- **What's New**: 본 논문에서는 대규모 희소 MIMO 채널을 위한 다운링크 빔포밍(end-to-end beamforming) 깊이 학습 프레임워크를 개발하였습니다. 이 구조는 인코더 NN, 빔포머 디코더 NN 및 채널 디코더 NN의 세 가지 모듈로 구성되어 있으며, 사용자와 기지국 사이의 데이터 흐름을 최적화합니다. 특히, 새로운 반전된 학습(knowledge distillation) 기법을 사용하여 미세 조정된 빔포머를 생성합니다.

- **Technical Details**: Deep EDN 아키텍처는 (1) 인코더 NN, (2) 빔포머 디코더 NN, (3) 채널 디코더 NN으로 구성되어 있습니다. 최적화 과정에서 반감 학습(semi-amortized learning) 및 지식 증류(knowledge distillation)를 포함한 두 가지 주요 전략을 사용합니다. 이는 높은 데이터 전송률을 달성하기 위해 구조를 설계하고 다양한 MIMO 시나리오에 적용할 수 있도록 합니다.

- **Performance Highlights**: 시뮬레이션 결과, 제안된 방법은 다양한 네트워크 환경 및 채널 조건에서 높은 합산 전송률을 일관되게 보여줍니다. 특히 단일 셀 및 공간 분할 시스템에 대한 적용 가능성을 입증하고, 원거리 및 가까운 거리의 하이브리드 빔포밍(hybrid beamforming) 시나리오에서도 효과적임을 확인했습니다. 이러한 접근법은 대규모 MIMO 시스템에서의 스케일링을 용이하게 하며 피드백 오버헤드를 줄이는 데 기여합니다.



### Modeling the language cortex with form-independent and enriched representations of sentence meaning reveals remarkable semantic abstractness (https://arxiv.org/abs/2510.02354)
- **What's New**: 이 논문은 인간의 언어 피질에서 의미의 추상적 표현을 찾기 위한 연구로, 문장에 대한 신경 반응을 시각 및 언어 모델의 표현을 사용하여 모델링합니다. 특히, 여러 이미지에서 추출한 vision model embeddings를 활용하여 언어 피질의 반응을 보다 정확히 예측할 수 있음을 발견하였습니다. 또한, 문장의 패러프레이즈(paraphrase)를 균형있게 활용하는 것이 예측 정확도를 높이는 데 기여함을 보여줍니다.

- **Technical Details**: 연구에서는 Inferior Frontal Gyrus, Middle Frontal Gyrus, Anterior Temporal cortex 등 'core' 언어 네트워크에서 fMRI 스캐닝을 통해 수집된 데이터셋을 분석하였습니다. 주어진 데이터셋에서 original sentences의 의미를 다른 문장이나 비전 모델의 표현으로도 포착할 수 있는지를 평가하였습니다. 또한, ridge regression 모델을 통해 원래 문장에 기반한 예측 정확도를 데이터의 맥락 정보를 통합하여 개선하는 방법을 사용했습니다.

- **Performance Highlights**: 연구 결과, 문장의 다양한 비주얼 디스펙션(visual depiction)을 이용한 embeddings가 언어 피질의 활동 예측에서 비약적인 예측 능력을 보여주었습니다. 또한, 패러프레이즈를 활용하여 예측 정확도가 향상되는 것을 발견했으며, 상식적(Contextual) 정보로 풍부하게 만든 문장이 예측력을 더욱 크게 증가시킨다는 점이 강조됐습니다. 이러한 결과는 언어 시스템이 단순한 언어 모델 이상으로 풍부하고 넓은 의미 표현을 유지하고 있다는 것을 시사합니다.



### An Senegalese Legal Texts Structuration Using LLM-augmented Knowledge Graph (https://arxiv.org/abs/2510.02353)
Comments:
          8 pages, 8 figures, 2 tables, 1 algorithm

- **What's New**: 이 연구는 세네갈 사법 시스템에서 법률 문서에 대한 접근을 향상시키기 위해 인공지능(AI)과 대형 언어 모델(LLM)의 응용을 다루고 있습니다. 법적 문서를 추출하고 조직하는 데 어려움을 보완할 필요성이 강조됩니다. 연구진은 특히 토지 및 공공 도메인 법전에서 7,967개의 기사를 성공적으로 추출하였으며, 이를 위한 상세한 그래프 데이터베이스를 개발했습니다.

- **Technical Details**: 이 연구에서는 2,872개의 노드와 10,774개의 관계를 포함하는 그래프 데이터베이스를 구축하였습니다. 또한, GPT-4o, GPT-4, Mistral-Large와 같은 모델을 사용해 관계 및 관련 메타데이터를 식별하는 고급 triple extraction 기법을 적용했습니다. 이 과정에서 LLM-augmented Knowledge Graph 원칙을 따르며, 핵심 참고 해소(coreference resolution), 명명된 개체 인식(named entity recognition), 개체 관계 식별 등을 통해 지식 그래프를 생성했습니다.

- **Performance Highlights**: 이 연구의 목표는 세네갈 국민과 법률 전문가가 자신들의 권리와 의무를 보다 효과적으로 이해할 수 있는 포괄적인 프레임워크를 개발하는 것입니다. 개발된 알고리즘과 그래프 데이터베이스는 법률 문서의 명확한 정보 제공을 가능하게 하며, 복잡한 법적 문서를 탐색하는 데 큰 도움이 될 것입니다. 최종적으로 이 시스템은 세네갈 법률 체계의 접근성과 효율성을 significantly 향상시킬 것으로 기대됩니다.



### mini-vec2vec: Scaling Universal Geometry Alignment with Linear Transformations (https://arxiv.org/abs/2510.02348)
- **What's New**: 본 논문은 vec2vec를 기반으로 하여, 교차 데이터 없이 텍스트 임베딩 공간을 정렬하도록 설계된 프로세스입니다. 특정한 매칭이 없어도 높은 안정성과 효율성을 자랑하는 mini-vec2vec를 소개하며, 기존 방법보다 연산 비용이 훨씬 낮은 선형 변환을 학습합니다. 이를 통해 새로운 분야에서의 활용 가능성을 제시합니다.

- **Technical Details**: mini-vec2vec는 세 가지 주요 단계로 구성됩니다: 가상 병렬 임베딩 벡터 간의 일치 시도, 변환 적합성, 그리고 반복적인 정제입니다. 상대적 표현 프레임워크를 활용하고, 간단한 아핀 변환을 통해 구조적 유사성을 검색하여 정렬을 가능하게 합니다. 이 방법은 CPU 하드웨어에서 실행될 수 있도록 설계되어 자원 소모를 최소화합니다.

- **Performance Highlights**: 논문에서는 comprehensive experiments를 통해 mini-vec2vec가 기존의 adversarial 방법인 vec2vec의 성능을 능가한다고 보고합니다. 특히, 연산 자원 소모가 대폭 줄어들고, 훈련의 안정성 및 해석 가능성을 제공하여 다양한 데이터 분포에서도 잘 작동함을 강조합니다. mini-vec2vec는 수 동록 훈련 데이터에서 효과적으로 작동하고, 1:1 매칭이 불가능한 경우에도 견고한 결과를 보여줍니다.



### Breaking the MoE LLM Trilemma: Dynamic Expert Clustering with Structured Compression (https://arxiv.org/abs/2510.02345)
Comments:
          12 pages, 2 figures, 3 tables. Under review as a conference paper at ICLR 2026

- **What's New**: 이번 연구는 Mixture-of-Experts (MoE) 대형 언어 모델에서 발생하는 부하 불균형, 매개변수 중복 및 통신 오버헤드를 동시에 해결하기 위한 통합 프레임워크를 소개합니다. 이 프레임워크는 긴밀하게 연결된 네 가지 혁신 요소를 통합하여 동적 전문가 클러스터링과 구조적 압축을 통해 MoE 모델의 효율성을 일관되게 개선합니다. 특히, 학습 중 모델 아키텍처를 동적으로 재구성할 수 있는 세맨틱(semantic) 임베딩 기능을 활용하여 효율적으로 전문가를 재편성합니다.

- **Technical Details**: 제안된 방법은 온라인 클러스터링 프로세스를 사용하여 매개변수와 활성화 유사도를 결합한 메트릭으로 전문가를 주기적으로 재조정합니다. 각 클러스터 내에서는 전문가 가중치를 공유 기본 행렬과 매우 낮은 순위의 잔여 적응기로 분해하여 매개변수를 최대 다섯 배 줄이는 동시에 전문성을 유지합니다. 이 구조는 두 단계 계층 라우팅 전략을 가능하게 하여 토큰이 먼저 클러스터에 할당된 후 구체적인 전문가에게 전달되도록 합니다.

- **Performance Highlights**: GLUE 및 WikiText-103에서 평가한 결과, 제안된 프레임워크는 표준 MoE 모델과 동등한 품질을 유지하면서 전체 매개변수를 약 80% 줄이고, 처리량을 10%에서 20% 증가시키며, 전문가 부하 분산을 세 배 이상 감소시켰습니다. 이러한 결과는 구조적 재편성이 확장 가능하고 효율적이며 메모리 효과적인 MoE LLM을 위한 핵심 길임을 입증합니다.



### Can Prompts Rewind Time for LLMs? Evaluating the Effectiveness of Prompted Knowledge Cutoffs (https://arxiv.org/abs/2510.02340)
- **What's New**: 이번 연구에서는 대형 언어 모델(Large Language Models, LLMs)이 실제로 이전의 지식 컷오프를 시뮬레이션 할 수 있는지를 조사합니다. 최근의 연구에서 프롬프트를 기반으로 한 언러닝(unlearning) 기법이 등장하였고, 이러한 기법을 통해 LLM의 지식 컷오프를 조정할 수 있는 가능성을 탐색합니다. 연구의 주요 초점은 LLM이 잊어야 할 정보에 대한 직접적인 질문이 없더라도 인과적으로 관련된 검색체를 잊는 여부입니다.

- **Technical Details**: 연구진은 사실(Factual), 의미(Semantic), 반사실(Counterfactual) 두 가지 차원에서 LLM의 지식 컷오프 유도 효과를 평가하기 위해 세 개의 데이터 세트를 구축했습니다. 각 데이터 세트는 서로 다른 675, 303 및 689개의 예를 포함합니다. 실험 결과, LLM이 사실적 지식과 의미 변화를 잊는 데 있어 각각 약 82.5%와 70.0%의 성공률을 기록했으나, 인과적 관계가 있는 이벤트를 잊는 경우 성공률이 약 19.2%로 낮음을 보였습니다.

- **Performance Highlights**: 이 연구는 프롬프트 기반의 지식 컷오프가 특정 차원에서 효과적일 수 있지만, 인과적으로 연결된 정보를 잊는 데 한계를 가진다는 것을 보여줍니다. 특히, LLM의 임시 컷오프 증명에서 발생할 수 있는 데이터 오염 문제를 해결하는 데 유용할 수 있습니다. 그러나 실제 세계의 시간적 예측 과제에서 LLM의 공정한 평가를 보장하기 위해서는 보다 강력한 방법이 필요하다는 결론을 내립니다.



### CRACQ: A Multi-Dimensional Approach To Automated Document Assessmen (https://arxiv.org/abs/2510.02337)
- **What's New**: 본 논문은 CRACQ라는 다차원 평가 프레임워크를 소개합니다. 이 프레임워크는 문서의 다섯 가지 특성인 Coherence(일관성), Rigor(엄밀성), Appropriateness(적절성), Completeness(완전성), Quality(품질)를 평가하도록 설계되었습니다. CRACQ는 자동화된 에세이 채점(Automated Essay Scoring)에서 얻은 통찰을 바탕으로 다른 형태의 기계 생성 텍스트로 그 초점을 확장합니다.

- **Technical Details**: CRACQ는 단일 점수 방식과는 달리 언어적, 의미적, 구조적 신호를 통합하여 누적 평가를 수행합니다. 이를 통해 전체적인 분석(holistic analysis)뿐만 아니라 각 특성 수준(trait-level analysis)의 분석이 가능합니다. 500개의 합성 보조금 제안서(synthetic grant proposals)로 훈련된 CRACQ는 LLM-as-a-judge와 비교 평가되었으며, 실제 강력한 및 약한 응용 프로그램에서도 추가 테스트가 진행되었습니다.

- **Performance Highlights**: 초기 결과에 따르면 CRACQ는 직접적인 LLM 평가보다 더 안정적이고 해석 가능한 특성 수준의 판단을 생성하는 것으로 나타났습니다. 그러나 신뢰성(reliability) 및 도메인 범위(domain scope)에서의 문제점은 여전히 남아 있습니다. 이러한 발견은 CRACQ가 머신러닝 기반의 텍스트 평가에 기여할 수 있는 가능성을 시사합니다.



### Where Did It Go Wrong? Attributing Undesirable LLM Behaviors via Representation Gradient Tracing (https://arxiv.org/abs/2510.02334)
Comments:
          16 pages, 4 figures

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 바람직하지 않은 행동 원인을 진단하기 위한 새로운 프레임워크를 제시합니다. 기존의 접근법들이 가지는 한계를 극복하기 위해, 모델 활성화 공간(activation space)에서 표현과 그 기울기를 분석하여 의미 있는 정보를 제공합니다. 이 프레임워크는 샘플 레벨과 토큰 레벨 모두에서 LLM의 행동을 분석할 수 있는 능력을 보여줍니다.

- **Technical Details**: 제안된 프레임워크인 Representation Gradient Tracing (RepT)는 모델의 내부 표현을 분석하여 훈련 데이터의 원인과 LLM 반응 간의 관련성을 수립합니다. 이를 통해 기존의 매개변수(parameter) 공간 대신에 의미 있는 표현 공간을 중심으로 데이터를 추적합니다. RepT는 정보적인 레이어 선택, 샘플 수준의 귀속 방법, 그리고 특정 단어 및 구절을 식별하는 토큰 수준 분석을 통해 동작합니다.

- **Performance Highlights**: 제안된 방법은 유해 콘텐츠 추적, 백도어 포이즈 탐지 및 지식 오염 식별과 같은 다양한 과제에 대해 체계적으로 평가되었습니다. 결과는 이 방법이 샘플 수준의 귀속뿐만 아니라 특정 샘플과 구절을 정밀하게 식별하는 데 강력한 성능을 보인다는 것을 보여줍니다. 이를 통해 우리는 LLM의 위험을 이해하고 감시하며 궁극적으로 완화하기 위한 강력한 진단 도구를 제공할 수 있습니다.



### WEE-Therapy: A Mixture of Weak Encoders Framework for Psychological Counseling Dialogue Analysis (https://arxiv.org/abs/2510.02320)
Comments:
          5 pages

- **What's New**: 본 논문은 심리 상담 분석을 위해 설계된 multi-task AudioLLM인 WEE-Therapy를 소개합니다. 기존의 오디오 언어 모델들이 일반 데이터를 기반으로 학습된 단일 스피치 인코더에 의존하는 반면, WEE-Therapy는 Weak Encoder Ensemble (WEE) 메커니즘을 통합하여 전문적이고도 복잡한 감정을 처리하는 능력을 향상시킵니다. 이를 통해 심리 상담에서 발생하는 복잡한 대화의 감정적 요소와 기술적 세부사항을 효과적으로 파악할 수 있습니다.

- **Technical Details**: WEE-Therapy 구조는 강력한 기본 인코더와 경량화된 전문 인코더의 조합으로 이루어져 있습니다. 기본 인코더는 Whisper-large-v3를 사용하고, 여러 개의 '약한' 인코더들은 심리 상담에서 놓칠 수 있는 세밀한 특징을 보완합니다. 특히, dual-routing 전략을 통해 데이터 독립적이고 의존적인 방식으로 인코더의 특성을 결합하여 최종 오디오 표현을 생성하는 혁신적인 방식을 적용합니다.

- **Performance Highlights**: WEE-Therapy는 감정 인식, 기술 분류, 위험 탐지 및 요약 작업을 포함한 네 가지 주요 작업에서 평가되었습니다. 실험 결과, 모든 작업에서 성능이 크게 향상되었으며, 모델의 파라미터 오버헤드는 최소화되었습니다. 이러한 결과는 AI 지원 클리닉 분석을 위한 강력한 잠재력을 보여주고 있습니다.



### Fine-tuning LLMs with variational Bayesian last layer for high-dimensional Bayesian optimization (https://arxiv.org/abs/2510.01471)
- **What's New**: 이 논문은 Bayes 최적화(Bayesian optimization, BO) 문제에 대하여 LLM(large language models)을 활용한 새로운 접근 방식을 제안합니다. 특히, 높은 차원의 입력 변수를 효율적으로 다루기 위해 Low-Rank Adaptation(LoRA) 기법을 사용하여 LLM의 파라미터를 미세 조정합니다. 이 과정에서 Variational Bayesian Last Layer(VBLL) 프레임워크 또한 활용하여 보다 강력한 성능을 달성합니다.

- **Technical Details**: 제안된 LoRA-VBLL 모델은 기존의 파라미터 효율적인 최적화 방법들에 비해 계산적으로 가벼우며, 재귀적 업데이트(Recursive updates)를 지원합니다. LoRA의 랭크 선택을 자동화하기 위해, 각기 다른 랭크를 가진 LoRA-VBLL 모델들을 앙상블하여 가중치를 조정하는 방안도 마련하였습니다. 이러한 방식으로 모델의 가중치와 LoRA-VBLL 파라미터들을 지속적으로 업데이트할 수 있습니다.

- **Performance Highlights**: 다양한 높은 차원의 최적화 벤치마크와 실제 분자 최적화 작업에서 제안된 (ENS-)LoRA-VBLL 접근 방식이 기존 방법들보다 뛰어난 성능을 보였음을 실험적으로 입증하였습니다. 이러한 결과는 제안된 방법이 기존의 Bayesian 최적화 방식의 한계를 극복할 수 있는 가능성을 보여줍니다.




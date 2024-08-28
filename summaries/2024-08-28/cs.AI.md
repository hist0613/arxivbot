New uploads on arXiv(cs.CL)

### Into the Unknown Unknowns: Engaged Human Learning through Participation in Language Model Agent Conversations (https://arxiv.org/abs/2408.15232)
- **What's New**: 본 논문은 사용자가 질문을 던지지 않고도 대화형 상호작용을 통해 정보를 발견할 수 있도록 하는 새로운 시스템, Collaborative STORM (Co-STORM)를 소개합니다.

- **Technical Details**: Co-STORM은 다수의 언어모델(LM) 에이전트 간의 대화를 관찰하고 방향을 제시할 수 있게 해주며, 사용자가 잘 알지 못하는 정보를 자연스럽게 발견할 수 있도록 돕습니다. 또한, Co-STORM은 발견된 정보를 동적인 마인드 맵(mind map)으로 정리하여 사용자가 쉽게 소통할 수 있도록 지원합니다.

- **Performance Highlights**: Co-STORM은 베이스라인(baseline) 방법들과 비교하여 담화 추적(discourse trace) 및 보고서(report) 품질에서 우수한 성능을 보였으며, 인간 평가에서 참가자의 70%가 Co-STORM을 검색 엔진보다 선호하고, 78%가 RAG 챗봇보다 Co-STORM을 선호한다고 응답했습니다.



### Classifying populist language in American presidential and governor speeches using automatic text analysis (https://arxiv.org/abs/2408.15213)
- **What's New**: 이 연구는 자동화된 분류 모델을 개발하여 포퓰리스트(populist) 언어 사용을 추정하는 파이프라인을 구축합니다. 이 파이프라인은 2010년부터 2018년까지 300명의 미국 주지사와 2016년 대통령 후보의 연설을 통해 포퓰리스트 언어를 식별하는 모델을 훈련시키고 검증합니다.

- **Technical Details**: 주요 데이터셋은 2010-2018 미국 주지사 연설 및 2016 대선 후보 연설로 구성되며, 모델은 포퓰리스트 및 다원주의적 언어가 포함된 문장에 기반해 훈련됩니다. 연구 결과, 주지사 연설의 84% 및 대통령 후보 연설의 89%를 올바르게 분류했습니다.

- **Performance Highlights**: 파이프라인은 다양한 시간대(최신 미국 주지사 연설 92% 정확성), 데이터 양(각 카테고리당 70개 문장으로 유사한 결과 도출) 및 정치인 분류에서 효과적으로 작동하며, 이는 포퓰리스트 언어 분류의 효율성을 크게 향상시킵니다.



### Can Unconfident LLM Annotations Be Used for Confident Conclusions? (https://arxiv.org/abs/2408.15204)
- **What's New**: 해당 연구는 대형 언어 모델(LLM)의 주석을 활용하여 인간 주석의 수를 25% 이상 줄일 수 있는 Confidence-Driven Inference (CDI) 방식을 제안합니다. 이 방법은 LLM의 주석과 신뢰도 지표를 조합하여 수집할 인간 주석을 전략적으로 선택하는 방법론을 다룹니다.

- **Technical Details**: Confidence-Driven Inference는 LLM 주석과 LLM의 신뢰 점수를 활용하여 인간 주석 수를 최소화하며, 통계적 추정을 정확하게 수행합니다. 이 방법은 세 가지 사회 과학 연구 설정, 즉 언어의 정중함, 입장, 그리고 편향을 포함한 다양한 문제에 적용됩니다. 이는 LLM의 정확성을 반영하는 신뢰 점수를 이용하여 진행됩니다.

- **Performance Highlights**: 이 연구는 언어 정중함, 입장, 미디어 편향 등 세 가지 설정에서 통계적 추정 과제를 수행했으며, LLM 주석을 통합한 Human Annotations의 효율성 및 커버리지를 유지하면서 데이터 샘플의 효과적인 크기를 증가시켰습니다.



### Measuring text summarization factuality using atomic facts entailment metrics in the context of retrieval augmented generation (https://arxiv.org/abs/2408.15171)
Comments:
          12 pages

- **What's New**: 최근 LLM(대형 언어 모델)의 사용이 급증하고 있으며, 특히 ChatGPT의 도입 이후 그 활용도가 눈에 띄게 증가했습니다. 그러나 LLM의 기업 및 상업적 채택에 있어 가장 큰 도전 과제는 정보의 부정확성, 즉 'hallucination'입니다. 본 연구에서는 LLM이 생성한 요약의 사실성을 평가하는 새로운 방법을 제안하고 있습니다.

- **Technical Details**: 이 연구에서는 'Retrieval Augmented Generation (RAG)' 기술을 사용하여 LLM의 요약 생성이 얼마나 사실적인지를 평가하기 위해 Naive Bayes 분류기를 활용하고 있습니다. 이 접근법은 사용자의 쿼리를 고차원 벡터로 변환하고, 벡터 데이터베이스에서 관련된 벡터를 검색하여 LLM이 최종 응답을 생성하는 과정을 포함합니다.

- **Performance Highlights**: 이 새로운 방법론은 기존의 ROUGE, BLEU, METEOR 같은 전통적인 요약 평가 지표의 한계를 극복하고, LLM의 생성된 요약과 원본 텍스트 간의 의미적 일관성을 더 잘 평가할 수 있는 잠재력을 보여줍니다.



### Relation Also Knows: Rethinking the Recall and Editing of Factual Associations in Auto-Regressive Transformer Language Models (https://arxiv.org/abs/2408.15091)
- **What's New**: 이 연구는 Transformer 언어 모델의 지식 회수를 해석하는 새로운 관계 중심 관점을 도입하고, 이를 바탕으로 지식 편집을 수행하여 과도 일반화(over-generalizing) 문제를 해결하려고 합니다.

- **Technical Details**: 기존의 locate-then-edit 방법들은 주로 주제(subject)에만 집중하고 관계(relation) 정보를 간과하여, 편집 시 비관련 관계들이 예기치 않게 변경되는 문제를 야기했습니다. 이번 연구는 MLP(Feed-Forward Networks)와 Multi-Head Self Attention(MHSA) 계층에서의 관계 표현의 전파를 분석하여 이러한 관계 중심 해석을 제시합니다. 이를 통해 관계 지식이 최종 관계 토큰에서 집계되고, 해당 위치에서 MLP를 수정함으로써 편집이 이루어질 수 있음을 보여줍니다.

- **Performance Highlights**: 제안된 RETS(Relation-focused Editing for auto-regressive Transformer LMs with Subject constraints) 방법은 기존의 locate-then-edit 방법보다 R-Specificity에서 30% 이상 개선된 성과를 보여주며, 다른 기준에서도 경쟁력을 유지했습니다. 이는 지식 편집의 과도 일반화 문제를 완화하면서도 전반적으로 균형 잡힌 성능을 나타냅니다.



### BaichuanSEED: Sharing the Potential of ExtensivE Data Collection and Deduplication by Introducing a Competitive Large Language Model Baselin (https://arxiv.org/abs/2408.15079)
Comments:
          19 pages, 6 figures

- **What's New**: BaichuanSEED 모델은 자체 데이터 처리 파이프라인을 사용하여 7B 파라미터를 갖춘 LLM(대형 언어 모델)로 훈련되었습니다. 이 모델은 공개 소스를 바탕으로 하며, 특정 하위 작업과의 최적화를 배제하고도 경쟁력 있는 성능을 발휘합니다.

- **Technical Details**: BaichuanSEED는 3T의 이중 언어 토큰을 사용하여 훈련되었으며, Transformer 아키텍처를 기반으로 되어 있습니다. 32개의 층과 32개의 attention heads를 가지며, hidden dimension은 4096, feed-forward layer의 크기는 11008입니다.

- **Performance Highlights**: 실험 결과, BaichuanSEED는 Llama3와 Qwen-1.5와 같은 상업용 모델과 비교하여 동등한 성능을 보여주며, 특히 수학 및 코딩 분야에서 추가 개선의 여지가 있습니다.



### Self-supervised Topic Taxonomy Discovery in the Box Embedding Spac (https://arxiv.org/abs/2408.15050)
Comments:
          to be published in TACL

- **What's New**: 이 논문에서는 기존 방법의 한계점을 극복하기 위해 Box embedding 기반의 Topic Model (BoxTM)을 개발하였습니다. BoxTM은 단어와 주제를 상자 형태의 임베딩 공간에 매핑하여 비대칭 metric을 정의하고 주제 간의 위계적 관계를 보다 정확하게 추론할 수 있도록 합니다.

- **Technical Details**: BoxTM은 단어와 주제를 하이퍼사각형(hyperrectangle)으로 표현하며, 이 하이퍼사각형의 부피는 주제의 의미적 범위의 크기에 비례합니다. 이를 통해 사용자들은 더 정확한 위계적 주제 관계를 파악할 수 있습니다. 또한, BoxTM은 낮은 수준의 주제 박스에서 상위 주제를 추출하기 위해 재귀적인 클러스터링(recursive clustering)을 수행합니다.

- **Performance Highlights**: 실험 결과 BoxTM은 기존의 최신 방법들과 비교하여 높은 품질의 주제 분류를 학습하며, 주제 간의 의미적 관계와 위계 구조를 더 잘 포착하는 것으로 입증되었습니다.



### A Survey of Large Language Models for European Languages (https://arxiv.org/abs/2408.15040)
- **What's New**: 본 논문은 Large Language Models(LLMs)의 발전을 다루며, LLaMA, PaLM, GPT, MoE 등 다양한 LLM 패밀리 및 유럽연합(EU) 언어를 위해 LLM을 생성하고 향상시키기 위한 방법들에 대한 종합적인 개요를 제공합니다. 특히, 유럽 언어를 위한 LLM 개발에 관한 기존 연구를 처음으로 포괄적으로 검토하였습니다.

- **Technical Details**: LLMs는 통계적 모델에서 신경망 모델로 발전하며, 특히 transformer 아키텍처를 사용하여 매우 큰 데이터셋과 매개변수를 활용하여 효율적으로 학습합니다. 이전의 언어 모델들은 단어의 조건부 확률을 기반으로 하였으나, LLM은 수십억 개의 매개변수를 통해 언어의 다양성과 변이를 더 잘 캡처할 수 있습니다. BERT, GPT 시리즈 등 다양한 모델이 언급됩니다.

- **Performance Highlights**: LLMs는 NLP 작업에서 기존의 pretrained 모델들보다 뛰어난 성능을 보이며, 대규모 웹 코퍼스를 기반으로 사전 학습된 최신 모델들은 NLP 태스크에서 최첨단 성능을 자랑합니다. 예를 들어, GPT-3는 175B의 매개변수를 가지며, QA, MT 등 복잡한 과제를 처리하는 데 뛰어난 능력을 보입니다.



### Evidence-Enhanced Triplet Generation Framework for Hallucination Alleviation in Generative Question Answering (https://arxiv.org/abs/2408.15037)
- **What's New**: 이 연구는 Generative Question Answering(GQA)에서 발생하는 hallucination 문제를 해결하기 위한 새로운 프레임워크인 EATQA(Evidence-enhanced Triplet Generation Framework)를 제안합니다. 이 프레임워크는 모델이 (질문, 증거, 답변) 삼중항의 모든 조합을 예측하도록 유도하여 그리고 이들 간의 논리적 관계를 이해하도록 합니다.

- **Technical Details**: EATQA는 질문(Q), 증거(E), 답변(A) 세 가지 부분의 예측을 포함하며, 각 두 쌍의 정보를 바탕으로 나머지 하나를 생성합니다. 프레임워크 내에서 분포 간격을 줄여서 증거로부터 지식을 증류하고, 이는 모델이 질의, 증거 및 답변 간의 논리적 관계를 학습하게 합니다.

- **Performance Highlights**: 실험 결과, EATQA는 MultiRC 및 QASPER 두 개의 GQA 벤치마크 데이터셋에서 기존의 LLM 기반 방법들과 hallucination 완화 접근방식에 비해 뛰어난 성능을 보였습니다. 이 방법은 내부 지식을 유지하면서도 hallucination을 완화하고 신뢰할 수 있는 답변 생성을 가능하게 합니다.



### Speech Recognition Transformers: Topological-lingualism Perspectiv (https://arxiv.org/abs/2408.14991)
- **What's New**: 본 논문은 음성 인식(ASR) 분야에 있어서 최근의 트랜스포머 아키텍처(tansformer architecture)와 자가 주의 메커니즘(self-attention mechanism)이 어떻게 발전해왔는지를 종합적으로 검토합니다. 특히 다국어(multi-lingual), 이중 언어(bi-lingual), 교차 언어(cross-lingual) ASR 시스템에서의 응용과 과제를 다루고 있습니다.

- **Technical Details**: 트랜스포머 아키텍처의 도입으로 기존의 수동 설계(feature engineering) 기반 음성 인식 시스템의 한계를 극복할 수 있게 되었습니다. 본 논문은 ASR 분야의 두 가지 차원인 애플리케이션(application)과 도전과제(challenges)를 중심으로 한 두 단계 구조의 분류법(taxonomy)을 제시합니다. 이러한 구조는 연구자들이 관련 애플리케이션을 식별하고, 서로 다른 ASR 도메인 간의 모델 설계를 비교하는 데 도움을 줍니다.

- **Performance Highlights**: 논문의 리뷰 내용을 통해, 다양한 언어 및 음성 데이터셋에서 ASR 시스템 성능을 평가할 수 있는 기준 세트를 제공하며, ASR 기술의 발전이 음성 감정 인식(emotion recognition), 증오 발언 탐지(hate speech detection), 다중 음성 문맥(multi-speech context) 제어 등에 어떻게 활용될 수 있는지를 강조합니다.



### AgentMonitor: A Plug-and-Play Framework for Predictive and Secure Multi-Agent Systems (https://arxiv.org/abs/2408.14972)
- **What's New**: 본 논문에서는 대형 언어 모델(LLM)을 기반으로 한 에이전트 시스템인 AgentMonitor를 소개합니다. 이 프레임워크는 다중 에이전트 시스템(Multi-Agent System, MAS)의 성능을 사전에 예측할 수 있도록 설계되었으며, 보안 위험을 실시간으로 수정하는 기능도 포함하고 있습니다.

- **Technical Details**: AgentMonitor는 각 에이전트 수준에서 통합되어 입력과 출력을 캡처하고, 이 데이터를 통계로 변환하여 회귀 모델(Regression Model)인 XGBoost를 통해 작업 성능을 예측합니다. 이 방법론은 LLM 발전 과정에서 발견된 스케일링 법칙(Scaling Laws)에 기반하여 MAS의 성능 예측 가능성을 탐구합니다.

- **Performance Highlights**: 실험 결과, XGBoost 모델이 내재적 조건에서 0.89의 Spearman 상관관계를 달성했으며, 보다 도전적인 시나리오에서도 평균 0.58의 상관관계를 유지했습니다. AgentMonitor를 사용함으로써 해로운 콘텐츠를 평균 6.2% 줄이고 유용한 콘텐츠는 1.8% 증가시켜 보안과 신뢰성을 향상시켰습니다.



### Multilingual Arbitrage: Optimizing Data Pools to Accelerate Multilingual Progress (https://arxiv.org/abs/2408.14960)
- **What's New**: 이번 연구는 "multilingual arbitrage"라는 개념을 도입하여 여러 모델 간의 성능 차이를 활용하여 합성 데이터 생성의 효율성을 높이는 방법론을 제안하고 있습니다. 이 방법은 단일 teacher 모델에 의존하는 기존 접근 방식의 한계를 극복하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 이 연구는 15개 언어에서 9개의 최첨단 다국어 모델을 사용하여 exhaustive experiments를 수행하였으며, 각각의 언어에 대해 여러 모델을 teacher로 두고 샘플을 전략적으로 라우팅하는 방법을 평가하여 최적의 성능을 달성하는 방법을 모색합니다.

- **Performance Highlights**: 연구 결과, reward-based routing 기법이 평균 56.5%의 generative win-rates 개선을 보여주었고, discriminative tasks에서는 최대 3.27% 개선을 달성하였습니다. 단일 teacher 모델 compared to, reward-based routing으로의 전환이 가장 성과가 좋음을 입증했습니다.



### SpikingSSMs: Learning Long Sequences with Sparse and Parallel Spiking State Space Models (https://arxiv.org/abs/2408.14909)
- **What's New**: 이번 연구에서는 Spiking Neural Networks (SNNs)와 State Space Models (SSMs)를 통합하여 Spiking State Space Models (SpikingSSMs)를 개발하였습니다. SpikingSSMs는 긴 시퀀스 학습용으로 설계되었으며, 계층적으로 통합된 신경 역학을 통해 희소 시냅스 계산을 실현합니다. 또한, 경량 서보적 동적 네트워크를 도입하여 훈련 속도를 크게 향상시켰습니다.

- **Technical Details**: SpikingSSMs는 LIF (Leaky Integrate-and-Fire) 뉴런의 비유동적 리셋 메커니즘을 이용하여 비동기 이벤트 구동 신경 역학과 병렬 계산 간의 갈등을 해결합니다. 이 연구는 SDN (Surrogate Dynamic Network)을 통해 LIF 뉴런 모델의 동적특성을 근사하며, 고속 훈련이 가능하도록 합니다. SpikingSSM은 고도의 희소성을 유지하면서도 SSMs와 경쟁력 있는 성능을 발휘합니다.

- **Performance Highlights**: SpikingSSM은 Long Range Arena (LRA) 벤치마크 작업에서 최첨단 SSMs와 동등한 성능을 달성하였으며, WikiText-103 데이터셋에서 기존 스파이킹 대형 언어 모델들에 비해 약 3배 작은 모델로도 우수한 결과를 보여주었습니다. 이 모델은 낮은 계산 비용을 위한 백본 아키텍처로서의 가능성을 입증했습니다.



### Writing in the Margins: Better Inference Pattern for Long Context Retrieva (https://arxiv.org/abs/2408.14906)
- **What's New**: 본 논문에서는 Retrieval-oriented (검색 지향) 작업에서 긴 입력 시퀀스를 최적화하기 위해 설계된 새로운 inference pattern인 WiM(Writing in the Margins)을 소개합니다.

- **Technical Details**: 이 접근법은 key-value cache의 chunked prefill을 활용하여 segment-wise inference를 수행합니다. 이를 통해 모델이 특정 작업으로 안내하는 중간 정보(‘margins’)의 생성 및 분류와 함께 넓은 맥락을 효율적으로 처리할 수 있게 합니다.

- **Performance Highlights**: WiM은 reasoning skills (추론 능력)에서 평균 7.5%의 정확도 향상(HotpotQA, MultiHop-RAG)을 제공하며, aggregation tasks (집계 작업)에서는 F1-score가 30.0% 이상 증가하는 효과를 보입니다.



### A Functional Trade-off between Prosodic and Semantic Cues in Conveying Sarcasm (https://arxiv.org/abs/2408.14892)
Comments:
          accepted at Interspeech 2024

- **What's New**: 이번 연구는 풍자(Sarcasm)의 음향적 특성과 그 표현을 사전조음적 신호(prosodic cues)와의 상호작용을 분리하여 조사합니다. 이를 통해 다섯 가지 음향적 특징을 종합적으로 분석하여, 의미적으로 구성이 명확한 경우와 그렇지 않은 경우의 차이점을 밝힙니다.

- **Technical Details**: 연구에 사용된 데이터셋은 MUStARD++로, 미국 TV 시트콤에서 추출된 1,202개의 오디오 비주얼 발화로 구성됩니다. 풍자적인 발화는 제안적 풍자(Propositional Sarcasm), 내재적 풍자(Embedded Sarcasm), 발화적 풍자(Illocutionary Sarcasm)로 분류되었습니다. 각 유형의 풍사는 이론적 토대와 함께 실험을 통해 조사되었습니다.

- **Performance Highlights**: 연구 결과, 의미가 뚜렷한 풍자적 표현에서는 사전조음적 신호보다 의미적 신호가 더 중요하게 작용함을 발견했습니다. 이는 풍자의 의도 전달에서 사전조음적 조정의 의존도가 낮아질 수 있음을 시사합니다.



### Inverse-Q*: Token Level Reinforcement Learning for Aligning Large Language Models Without Preference Data (https://arxiv.org/abs/2408.14874)
- **What's New**: 이 논문에서는 전통적인 RL 방법에서 벗어나 토큰 수준에서의 강화 학습을 최적화하는 새로운 프레임워크인 Inverse-Q*를 소개합니다. 이 방법은 보상 모델이나 가치 모델의 추가 없이 직접적인 선호 최적화 기법을 활용합니다.

- **Technical Details**: Inverse-Q*는 PPO(Proximal Policy Optimization) 알고리즘의 목표를 최적화하며, 조건부 최적 정책을 모델의 응답에서 직접 추정하는 방식으로 더 세분화되고 유연한 정책 형성을 가능하게 합니다. 이 방법은 인간 주석이나 외부 감독에 대한 의존도를 줄여 특히 자원이 적은 환경에서 적합합니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, Inverse-Q*는 PPO보다 더 빠른 수렴 속도와 인간의 선호에 맞춘 모델 응답의 정렬 효과성을 보여주며, 기존 RLHF 방법들보다 효과적이고 실용적인 대안을 제공합니다.



### Advancing Adversarial Suffix Transfer Learning on Aligned Large Language Models (https://arxiv.org/abs/2408.14866)
Comments:
          11 pages, 4 figures

- **What's New**: 이 논문은 두 단계 전이 학습 프레임워크인 DeGCG를 제안하여, 언어 모델의 안전성 문제를 해결하고자 합니다. 이는 검색 효율성과 접미사의 전이 가능성을 연결하는 내용을 포함합니다.

- **Technical Details**: DeGCG는 두 단계로 구성됩니다: 행동 비특이적의 사전 검색(behavior-agnostic pre-searching)과 행동 관련의 후 검색(behavior-relevant post-searching)입니다. 첫 번째 단계에서는 사용자 비의존적인 타겟 토큰을 최적화하여 사전 검색을 수행하고, 두 번째 단계에서는 이 초기 값에서 출발해 맥락 관련 검색을 진행합니다.

- **Performance Highlights**: HarmBench에서의 실험 결과, i-DeGCG 버전은 Llama2-chat-7b 모델에서 ASR(Attack Success Rate)이 각각 43.9%와 39.0%로 기준 가입법보다 각각 22.2% 및 19.5% 향상되었습니다. 이는 접미사의 전이 가능성이 검색 효율성에 중요한 기여를 한다는 것을 보여줍니다.



### Detecting AI Flaws: Target-Driven Attacks on Internal Faults in Language Models (https://arxiv.org/abs/2408.14853)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)에서 독성 콘텐츠를 탐지하는 새로운 공격 패러다임인 ToxDet를 제안하여 독성 응답을 유도하는 데 집중합니다.

- **Technical Details**: ToxDet는 강화 학습(reinforcement learning)을 활용하여 타겟 모델과의 상호작용을 통해 훈련되며, 이를 통해 적절한 질문 및 초기 답변을 생성하여 독성 응답을 유도합니다. 이 방식은 프롬프트 엔지니어링(prompt engineering) 대신 직접적으로 모델을 공격하는 접근입니다.

- **Performance Highlights**: AdvBench 및 HH-Harmless 데이터셋에서 실험 결과, ToxDet는 오픈소스와 블랙박스 모델에 대한 효과적인 공격 능력을 입증하며, 기존 LLM의 취약점을 드러내고 연구자들이 모델 강화를 위한 귀중한 자료를 제공합니다.



### Project SHADOW: Symbolic Higher-order Associative Deductive reasoning On Wikidata using LM probing (https://arxiv.org/abs/2408.14849)
Comments:
          6 pages, 1 figure

- **What's New**: SHADOW라는 중간 작업에 대해 조정된 언어 모델을 소개하고, 위키데이터(Wikidata) 삼중 완성을 사용하여 지식 기반 구축 작업에 대한 성능을 측정합니다. SHADOW는 LM-KBC 2024 챌린지에서 기준 솔루션보다 20% 향상된 F1 점수 68.72%로 평가받았습니다.

- **Technical Details**: SHADOW는 지식 기반 삼중 항에 대해 세밀하게 조정된 모델로, 연관 추론(associative deductive reasoning)에서 영감을 받은 방법론을 사용했습니다. 실험 동안, 주어진 데이터에 대해 조건부 생성 모델로 학습하며, 각 주제 및 관계 쌍에 대한 관련 객체를 검색하기 위한 템플릿을 설계합니다. SHADOW는 377개의 학습 샘플, 378개의 검증 샘플 및 378개의 테스트 샘플이 포함된 데이터 세트를 사용하여 훈련되었습니다.

- **Performance Highlights**: SHADOW는 다양한 관계에 대한 템플릿 식별 작업에서 우수한 성능을 보였습니다. 그러나 countryLandBordersCountry 관계에서 더 낮은 성능을 보였으며, 이는 이 관계의 성격 때문입니다. SHADOW는 기준 모델보다 약 20% 더 향상된 성과를 기록했으며, 다른 해결책과 비교했을 때 좋은 성능을 발휘했습니다. 그러나 여전히 최고의 점수에는 미치지 못하는 한계가 있습니다.



### AAVENUE: Detecting LLM Biases on NLU Tasks in AAVE via a Novel Benchmark (https://arxiv.org/abs/2408.14845)
- **What's New**: 본 논문은 아프리카계 미국인의 방언(Black Vernacular English, AAVE)에서 자연어 이해(NLU) 시스템의 편향을 탐지하는 것이 포괄적인 자연어 처리(NLP) 시스템 개발에 필수적임을 강조하며, AAVENUE(AAVE Natural Language Understanding Evaluation)라는 벤치마크를 도입하여 AAVE와 표준 미국 영어(Standard American English, SAE) 작업에서 대규모 언어 모델(LLM)의 성능을 평가하고자 합니다.

- **Technical Details**: AAVENUE는 VALUE 벤치마크를 발전시키고 확장하였으며, 결정론적인 구문(syntactic) 및 형태학적(morphological) 변환을 대체하여 LLM 기반 번역 및 몇 가지 샘플 예시(few-shot prompting)를 활용하는 더 유연한 방법론을 사용합니다. 이 접근법은 GLUE와 SuperGLUE 벤치마크에서 주요 작업을 번역할 때 성능을 향상시킵니다. 논문에서 다루는 주요 작업은 BoolQ, MultiRC, SST-2, COPA, WSC입니다.

- **Performance Highlights**: 우리의 평가 결과, LLM은 AAVE로 번역된 버전보다 SAE 작업에서 일관되게 더 나은 성능을 보였으며, 이는 내재적 편향을 드러냅니다. AAVE 번역의 품질 점수는 다양한 메트릭에서 VALUE 벤치마크보다 우수한 결과를 보였습니다. 예를 들어, BoolQ 작업에서 우리의 번역은 품질 점수 76.57을 기록하였고, 이는 VALUE의 58.21보다 월등히 높은 수치입니다. 이러한 성과는 AAVENUE 벤치마크가 NLP 모델의 포괄성을 촉진하는 데 기여할 필요성을 강조합니다.



### GSIFN: A Graph-Structured and Interlaced-Masked Multimodal Transformer Based Fusion Network for Multimodal Sentiment Analysis (https://arxiv.org/abs/2408.14809)
- **What's New**: 이번 연구에서는 새로운 GSIFN 모델을 제안합니다. GSIFN은 Multimodal Sentiment Analysis (MSA)에서 발생하는 두 가지 주요 문제를 해결하기 위한 구조화된 그래프 및 interlaced-masked 접근 방식을 사용합니다.

- **Technical Details**: GSIFN은 Graph-Structured and Interlaced-Masked Multimodal Transformer를 기반으로 하며, Interlaced Mask (IM) 메커니즘을 통해 robust한 multimodal graph embedding을 생성합니다. 모델은 unimodal label generation을 위한 low computation overhead의 self-supervised learning framework을 통합하여 non-verbal modal feature를 개선합니다.

- **Performance Highlights**: GSIFN은 CMU-MOSI, CMU-MOSEI, CH-SIMS 데이터셋에서 평가되었으며, 기존의 최신 방법들과 비교하여 성능이 뛰어나고, significantly lower computation overhead를 보여주었습니다.



### A global AI community requires language-diverse publishing (https://arxiv.org/abs/2408.14772)
Comments:
          Translations by Michael Hardy (Guarani), Vandana Sarin and Vivek Sarin (Hindi), Roshna Omer Abdulrahman (Soranî Kurdish), Gabriel Poesia (Portuguese), and Matías Grinberg (Spanish). In the proceedings of the Global AI Cultures Workshop at the Twelfth International Conference on Learning Representations (ICLR) 2024, Vienna, Austria, May 7-11, 2024

- **What's New**: 본 논문은 AI 연구 커뮤니티 내에서 영어의 지배적 위치에 대한 이슈를 다루고 있습니다. 저자들은 영어 출판의 요구가 AI 분야에서 더 넓은 추출(regime of extraction) 체계를 유지하고 강화한다고 주장합니다.

- **Technical Details**: 저자들은 대규모 언어 모델(large language models) 및 기계 번역(machine translation)을 장려하지만, 이러한 기술들이 과학자 및 독자들의 언어적 제외(linguistic exclusion) 증상을 방증한다고 보았습니다.

- **Performance Highlights**: 저자는 회의(conferences)를 개최하는 국가의 언어로 진행하고, 동료 심사(peer review)자가 논문의 언어 적합성을 평가하지 않도록 지침을 제공하며, 다양한 언어로 출판 및 발표 기회를 제공하는 건강한 출판 문화의 대안을 제안합니다.



### LyCon: Lyrics Reconstruction from the Bag-of-Words Using Large Language Models (https://arxiv.org/abs/2408.14750)
Comments:
          Dataset downlodable at this https URL

- **What's New**: 본 논문에서는 저작권 문제로 인해 가사 연구에 필요한 직접적인 가사 사용이 제한되는 상황에서 새로운 접근방법을 제시했습니다. 공개된 Bag-of-Words (BoW) 데이터셋을 활용하여 저작권 없는 가사를 생성하고 이를 기반으로 한 새로운 데이터셋 LyCon을 구축하였습니다.

- **Technical Details**: 가사 재구성을 위해 대형 언어 모델을 활용하며, 이를 위해 musiXmatch 데이터셋의 Bag-of-Words에서 각 곡의 어휘 목록을 수집했습니다. 아티스트, 장르, 감정 주석과 같은 메타데이터를 사용해 특정한 스타일과 주제에 맞는 가사를 생성합니다. 재구성된 가사는 7,863곡에 대해 이루어졌습니다.

- **Performance Highlights**: LyCon의 재구성된 가사는 원래 가사와 통계적으로 유사한 특징을 보이며, 장르와 감정에 맞춰 조건부 가사 생성과 같은 다양한 연구에 활용될 수 있습니다. LyCon은 각 곡이 MSD(밀리언 송 데이터셋)와 연동되어 있으며, 데이터셋은 공개적으로 이용 가능합니다.



### Training-Free Activation Sparsity in Large Language Models (https://arxiv.org/abs/2408.14690)
- **What's New**: TEAL (Training-Free Activation Sparsity in LLMs)는 대규모 언어 모델에서 훈련 없이 활성화 희소성을 적용할 수 있는 방법으로, 모델 전반에 걸쳐 40-50%의 희소성을 달성합니다.

- **Technical Details**: TEAL은 크기 기반 활성화 희소성을 적용하며, 레이어에 따라 기준이 다릅니다. LLaMA 아키텍처 모델에서 제로 평균의 단일 모드 분포를 기반으로 하여 비중요 활성화를 다듬어 모델의 희소성을 높입니다.

- **Performance Highlights**: TEAL은 40% 및 50% 모델 전반에서 각각 1.53배 및 1.8배의 속도 향상을 달성하며, 가중치 양자화와 호환되어 추가적인 효율성을 제공합니다.



### What Makes a Good Story and How Can We Measure It? A Comprehensive Survey of Story Evaluation (https://arxiv.org/abs/2408.14622)
- **What's New**: 본 논문은 Large Language Models (LLMs)의 발전과 함께 자동 생성된 이야기의 수량과 품질이 크게 향상됨에 따라, 이야기 평가의 필요성과 그 도전 과제를 다룹니다. 특히, 기계번역과는 달리 이야기 평가는 요구하는 기준이 더 복잡함을 강조합니다.

- **Technical Details**: 존재하는 스토리 생성 작업에는 text-to-text, visual-to-text, text-to-visual이 포함됩니다. 정확한 평가를 위해 인간 기준에 따라 다양한 평가 메트릭스를 제안하고 정리하였습니다.

- **Performance Highlights**: 대부분의 기존 언어 기반 메트릭스는 성능이 낮았으나, 최근 LLM 기반 메트릭스는 인간의 판단과 더 일관된 평가를 제공합니다. 특히, 일반적인 평가에서 다루지 않았던 개인화된 평가와 긴 이야기 평가에 대한 연구가 필요하다고 제안합니다.



### Surprisingly Fragile: Assessing and Addressing Prompt Instability in Multimodal Foundation Models (https://arxiv.org/abs/2408.14595)
Comments:
          in submission

- **What's New**: 이번 연구는 다중모달 기초 모델(Multimodal Foundation Models, MFMs)의 프롬프트 불안정성(prompt instability) 문제를 조명하며, 증강 데이터(augmented data)를 통해 이러한 불안정성을 완화할 수 있음을 보여줍니다.

- **Technical Details**: 다중모달 모델은 이미지, 비디오, 오디오 등의 다양한 모달리티 데이터를 처리할 수 있는 적응형 접근법을 사용합니다. 이 연구는 주어지는 프롬프트의 변형에 따른 모델의 성능 변화에 대한 분석을 진행하였으며, 특히 프롬프트 변화에 따른 일관성(consistency) 문제를 해결하기 위해 '모달리티 기반 프롬프트 변형(modality-grounded prompt perturbation)' 방법론을 제시합니다.

- **Performance Highlights**: 모델을 증강 데이터를 통해 재훈련(train)한 결과, 다양한 실험 조건에서도 성능이 향상되고 일관성이 증가하는 것을 확인하였습니다. 특히 이미지 모달리티 특정 정보를 타겟으로 한 프롬프트에서 일관된 성능 개선이 관찰되었습니다.



### Improving Clinical Note Generation from Complex Doctor-Patient Conversation (https://arxiv.org/abs/2408.14568)
- **What's New**: 본 논문에서는 CliniKnote라는 방대한 의사-환자 대화 데이터셋과 K-SOAP (Keyword, Subjective, Objective, Assessment, and Plan) 노트 형식을 제안합니다. 이를 통해 클리닉 노트 생성에 대한 자동화 시스템의 발전을 목표로 하고 있습니다.

- **Technical Details**: CliniKnote 데이터셋은 1,200개의 복잡한 의사-환자 대화와 이에 대응하는 클리닉 노트를 포함합니다. 전통적인 SOAP 노트에 키워드 섹션을 추가하여 정보를 신속하게 식별할 수 있도록 개선하였으며, 최신 대형 언어 모델(LLMs)을 기반으로 하는 자동 생성 파이프라인을 개발했습니다. 또한, 다양한 LLM의 성능을 벤치마킹하여 효율성과 성능을 획기적으로 향상시켰습니다.

- **Performance Highlights**: 자동 생성된 K-SOAP 노트의 검토 및 수정에 필요한 시간이 기존의 수작업 작성 및 검토 시간보다 현저히 줄어든 것을 보여주었으며, LLM을 세부 도메인 지식으로 파인튜닝하여 최상의 성능을 달성했습니다.



### LLM Defenses Are Not Robust to Multi-Turn Human Jailbreaks Y (https://arxiv.org/abs/2408.15221)
- **What's New**: 최근 대형 언어 모델(LLM) 방어 시스템의 발전이 악의적인 쿼리에 대한 거부 능력을 크게 향상시켰습니다. 하지만 다단계 대화에서 인간의 감옥 탈출(jailbreak)이 드러낸 취약점이 존재하며, 이로 인해 70% 이상의 공격 성공률을 기록했습니다. 이는 단일 턴 공격에서 취약성을 보여준 방어 시스템과의 큰 차이를 보여줍니다.

- **Technical Details**: 이 연구에서는 Multi-Turn Human Jailbreaks (MHJ)라는 2912개의 프롬프트로 구성된 데이터셋을 제작했습니다. 이 데이터셋은 537개의 다단계 감옥 탈출을 포함하며, 기존 자동 공격과 비교해 인간의 다단계 공격이 19%에서 65%까지 더 높은 공격 성공률(ASR)을 보였습니다. 이를 통해 기존 방어 모델이 다단계 대화 상황에서 일반화되지 않는다는 사실을 입증했습니다.

- **Performance Highlights**: MHJ 데이터셋은 인간의 다단계 공격이 기존 자동 공격보다 월등히 높은 효율성을 지닌다는 것을 보여주며, 안전 메커니즘에 대한 취약성을 드러냈습니다. 특히 머신 언러닝 방어 시스템에서도 이중 용도의 생물 보안 지식을 복구하는 데 성공했습니다.



### Infusing Acoustic Pause Context into Text-Based Dementia Assessmen (https://arxiv.org/abs/2408.15188)
Comments:
          Accepted at INTERSPEECH 2024

- **What's New**: 이번 연구는 언어 모델의 기초로 돋보이는 중지 정보(pause information)를 활용하여 인지장애의 다양한 단계를 정확히 식별하는 방법을 조사합니다. 이를 통해, 치매 조기 진단을 위한 비침습적인 생체 마커(biomarker)로서 잠재력을 평가합니다.

- **Technical Details**: 이 연구는 독일어 어학 센터에서 수행된 다기관 연구의 음성 샘플을 기반으로 하여, 말하기 시험인 Verbal Fluency Test (VFT)와 Picture Description Test (PDT)를 사용합니다. 연구 목표는 인지장애의 없거나 경미한 상태(NC), 경미한 인지장애(MCI), 알츠하이머 치매(AD) 간의 명확한 차이를 정량적으로 확인하는 것입니다. 연구는 Transformer 기반의 언어 모델에서 pause 정보를 통합하고, cross-attention을 사용하여 텍스트 시스템에 음성 정보를 통합한 두 가지 접근 방식을 채택합니다.

- **Performance Highlights**: 모델은 VFT를 사용하여 NC와 MCI를 구별하는 데 가장 효과적이며, 여기서는 음성 정보가 중요한 역할을 합니다. 또한, MCI와 AD는 PDT를 통해 더 효과적으로 구별할 수 있으며, 텍스트 기반 시스템에서의 비유창성과 중단 모델링이 충분합니다. NC와 AD는 테스트와 관계없이 신뢰할 수 있게 구분 가능하지만, pause 모델링이 효과적입니다.



### Unlocking Potential in Pre-Trained Music Language Models for Versatile Multi-Track Music Arrangemen (https://arxiv.org/abs/2408.15176)
Comments:
          Submitted to AAAI 2025

- **What's New**: 본 연구에서는 여러 다중 트랙 음악 편곡 작업을 위한 통합된 시퀀스-투-시퀀스 프레임워크를 제안합니다. 이 프레임워크는 기호 음악 언어 모델의 미세 조정을 가능하게 하여 밴드 편곡, 피아노 축소, 드럼 편곡, 보이스 분리와 같은 다양한 작업을 수행할 수 있습니다.

- **Technical Details**: 이 과정에서는 Transformer 디코더를 사용하여 대규모 비표시 데이터셋에서 표준 언어 모델 훈련을 수행한 후, 제안된 목표에 따라 다중 편곡 작업을 위한 시퀀스-투-시퀀스 형식으로 미세 조정합니다. 또한 REMI-z라는 토큰화를 도입하여 음악 조각을 바 시퀀스와 서로 다른 트랙 시퀀스로 표현합니다.

- **Performance Highlights**: 실험 결과 제안된 방법이 밴드 편곡, 피아노 축소, 드럼 편곡, 보이스 분리의 네 가지 작업에서 기존의 작업 특정 모델들보다 일관되게 더 높은 음악 품질을 달성함을 보여주었습니다. 또한, 미세 조정만으로는 효과적으로 배울 수 없는 필수 음악 지식을 전이 학습을 통해 습득함을 입증하였습니다.



### X-Reflect: Cross-Reflection Prompting for Multimodal Recommendation (https://arxiv.org/abs/2408.15172)
- **What's New**: 본 논문에서는 Cross-Reflection Prompting (X-Reflect)이라는 새로운 프레임워크를 도입하여, 텍스트와 이미지 간의 지원 및 상충 정보를 명시적으로 식별하고 조화롭게 통합하는 방식으로 추천 시스템의 성능을 향상시키고자 하였습니다.

- **Technical Details**: X-Reflect 방법은 LLM과 LMM에서 텍스트와 이미지 정보를 동시 처리하여, 두 모달리티 간의 상호 지원 혹은 상충 정보를 파악하고 조화롭게 통합하는 과정을 포함합니다. 이는 별도의 프롬프트 방식과 결합된 프롬프트 방식을 통해 구현됩니다.

- **Performance Highlights**: 두 개의 널리 사용되는 벤치마크에서 수행한 광범위한 실험을 통해, 제안된 방법이 기존의 프롬프트 기본선 대비 추천 정확도에서 일관되게 우수한 성능을 보임을 입증하였습니다.



### How transformers learn structured data: insights from hierarchical filtering (https://arxiv.org/abs/2408.15138)
Comments:
          18 pages, 9 figures

- **What's New**: 우리는 데이터의 위치적 상관관계 범위를 제어할 수 있는 트리 기반 생성 모델에 대한 계층적 필터링 절차를 도입하였습니다. 이 연구는 vanilla encoder-only transformer 아키텍처가 optimal Belief Propagation 알고리즘을 구현할 수 있음을 보여줍니다.

- **Technical Details**: 제안된 모델은 계층적 데이터를 필터링하여 트리 구조의 서열을 생성하는 것으로, 계층적 상관관계를 조정할 수 있는 특징이 있습니다. 이 과정에서, BP 알고리즘을 정확히 해결할 수 있는 데이터 모델을 통해 훈련된 transformer가 최적의 성능을 예상할 수 있습니다.

- **Performance Highlights**: 훈련된 transformer 네트워크가 BP 예측을 정확히 복제하여, 부분 계층에서 훈련되고 전체 계층에서 테스트될 때에도 일관된 성능을 보여줍니다. Masked Language Modeling(MOM)의 경우, 훈련 초기에는 작은 규모의 상관관계가 먼저 학습된 후 더 큰 규모의 상관관계가 학습되는 패턴을 확인할 수 있었습니다.



### MRSE: An Efficient Multi-modality Retrieval System for Large Scale E-commerc (https://arxiv.org/abs/2408.14968)
- **What's New**: 이번 논문에서는 MRSE(Multi-modality Retrieval System for Large Scale E-commerce)를 제안하여 텍스트 쿼리 및 아이템 이미지, 사용자 선호도를 통합하여 품질 높은 아이템 검색을 위한 솔루션을 제공합니다. MRSE는 기존의 단일 모달리티 시스템의 한계를 극복하며, 사용자 선호에 따라 매칭 성능을 향상시킵니다.

- **Technical Details**: MRSE는 Lightweight Mixture-of-Expert (LMoE) 모듈을 사용하여 텍스트와 이미지 간의 농축된 특징을 정렬하고, 사용자 프로필을 생성합니다. 핵심 기술로 VBert와 FtAtt를 이용해 모달리티 간 및 내의 특징을 개선하고, 하이브리드 손실 함수를 사용하여 고속 부정 샘플링을 통해 일관성과 강건성을 향상시킵니다.

- **Performance Highlights**: 대규모의 Shopee 데이터셋에서 실험한 결과, MRSE는 기존의 단일 모달리티 시스템에 비해 오프라인 관련성에서 18.9%, 온라인 핵심 지표에서 3.7%의 성능 향상을 보여주었습니다. 이로 인해 MRSE는 Shopee의 플랫폼 전반에서 기본 모델로 채택되었습니다.



### Tripl\`etoile: Extraction of Knowledge from Microblogging Tex (https://arxiv.org/abs/2408.14908)
Comments:
          42 pages, 6 figures

- **What's New**: 최근 지식 그래프(knowledge graph)를 문서에서 자동으로 추출하는 방법들이 많이 등장했습니다. 그러나 마이크로 블로깅 사이트나 뉴스와 같은 비전형적 텍스트 소스에서 엔티티와 관계를 모델링하는 데 어려움이 있었습니다. 이 논문에서는 소셜 미디어 플랫폼의 마이크로 블로깅 게시물에서 개방 도메인 엔티티를 포함하는 지식 그래프를 추출하기 위한 개선된 정보 추출 파이프라인을 제안합니다.

- **Technical Details**: 이 파이프라인은 의존 구문 분석(dependency parsing)을 활용하고, 단어 임베딩(word embeddings)에 대한 계층적 클러스터링(hierarchical clustering)을 통해 비지도 학습(unsupervised) 방식으로 엔티티 관계를 분류합니다. 연구는 디지털 트랜스포메이션 관련 10만 개의 트윗에서 의미적 삼중(semantic triples)을 추출하는 사례를 제공합니다.

- **Performance Highlights**: 시스템이 생성한 트리플은 95% 이상의 정밀도를 보이고 있으며, 유사한 파이프라인보다 약 5% 더 높은 정밀도를 기록하고, 상대적으로 많은 수의 트리플을 생성했습니다.



### VHAKG: A Multi-modal Knowledge Graph Based on Synchronized Multi-view Videos of Daily Activities (https://arxiv.org/abs/2408.14895)
Comments:
          5 pages,4 figures, accepted by CIKM2024 Resource Track

- **What's New**: 이번 논문에서는 여러 이벤트로 구성된 일상 활동 비디오의 Multi-modal Knowledge Graph (MMKG)을 생성했습니다. 이 MMKG는 비디오의 프레임별로 세밀한 변화를 포함하며, 일상 활동을 이벤트 중심으로 표현합니다.

- **Technical Details**: VirtualHome-AIST-KG (VHAKG)라는 새로운 MMKG를 소개하며, 다양한 원시 행동을 렌더링하고 2D 바운딩 박스를 자동으로 주석 처리하는 기능을 추가했습니다. 이 데이터는 웹에서 지속 가능하게 사용할 수 있도록 압축되어 배포됩니다.

- **Performance Highlights**: VHAKG를 이용하면 시각 언어 모델(LVLMs)의 성능을 평가하기 위한 맞춤형 테스트 데이터 세트를 쉽게 추출할 수 있습니다. 이러한 MMKG를 사용한 비교 Benchmarking이 가능해져, 다양한 비전-언어 작업의 발전에 기여할 것으로 예상됩니다.



### CL4KGE: A Curriculum Learning Method for Knowledge Graph Embedding (https://arxiv.org/abs/2408.14840)
Comments:
          16 pages, 3 figures

- **What's New**: 이 논문에서는 Knowledge Graph Embedding (KGE) 훈련의 난이도를 평가하기 위한 Z-counts라는 새로운 메트릭을 제안하고, 이를 기반으로 하는 CL4KGE라는 효율적인 Curriculum Learning 방법론을 소개합니다.

- **Technical Details**: Z-counts는 KGs의 각 트리플 (head entity, relation, tail entity)의 훈련 난이도를 측정하는 메트릭입니다. CL4KGE는 이 메트릭을 활용하여 트리플의 난이도를 기준으로 훈련을 순차적으로 진행할 수 있게 합니다. 이 방법은 다양한 KGE 모델에 플러그인 형태로 적용 가능하며, 대부분의 KGs에 대해 적응성이 뛰어납니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋에서 수행한 실험을 통해 CL4KGE가 기존 방법에 비해 뛰어난 성능 향상을 보인 것을 확인했습니다. 특히 링크 예측(link prediction) 및 트리플 분류(triple classification) 작업에서 효과가 입증되었습니다.



### PolicyLR: A Logic Representation For Privacy Policies (https://arxiv.org/abs/2408.14830)
- **What's New**: 이번 논문에서는 Privacy 정책의 복잡성을 해결하기 위한 새로운 접근법인 PolicyLR을 제안합니다. 이 시스템은 Privacy 정책을 기계가 읽을 수 있는 형식으로 변환하여 다양한 후속 작업(compliance, consistency, privacy comparison 등)에 활용할 수 있도록 합니다.

- **Technical Details**: PolicyLR은 공통의 자연어 처리 모델인 Large Language Models(LLMs)를 활용하여 비구조화된 정책 텍스트를 원자(formula) 평 평가 값으로 변환하는 컴파일러를 구축합니다. 이 과정은 두 단계의 번역 및 함의 절차로 나뉘어 있으며, Privacy 정책의 전체 맥락을 고려하여 복잡한 공식을 유도합니다.

- **Performance Highlights**: 우리의 컴파일러는 ToS;DR 데이터셋을 사용하여 0.91의 정밀도(precision)와 0.88의 재현율(recall)을 기록하였습니다. 또한, PolicyLR은 정책 준수(Policy Compliance), 불일치 탐지(Inconsistency Detection), 개인 정보 비교 쇼핑(Privacy Comparison Shopping)과 같은 세 가지 주요 Privacy 작업에서의 유용성을 입증하였습니다.



### From Rule-Based Models to Deep Learning Transformers Architectures for Natural Language Processing and Sign Language Translation Systems: Survey, Taxonomy and Performance Evaluation (https://arxiv.org/abs/2408.14825)
- **What's New**: 이번 논문은 전 세계적으로 증가하는 청각 장애인과 난청 인구를 위해 수화 기계 번역 시스템의 필요성을 강조하고 있습니다. 기존의 연구가 수화의 동적이고 연속적인 특성을 충분히 고려하지 못한 점을 지적하며, 새로운 접근 방식을 제공합니다.

- **Technical Details**: 이 논문은 수화 기계 번역 알고리즘의 시간적 진화를 회고적으로 분석하고, 언어 번역에서 가장 많이 사용되는 Transformers 아키텍처의 분류법을 제공합니다. 또한, 정확한 deep learning 알고리즘에 기반한 실시간 Quality-of-Service 수화 기계 번역 시스템의 요구 사항을 제시합니다.

- **Performance Highlights**: 사실상, 수화 기계 번역의 발전을 위한 미래 연구 방향도 제안됩니다. 이 시스템은 언어의 연속성과 동적 특성을 효과적으로 처리할 수 있는 가능성을 보여줍니다.



### Instruct-SkillMix: A Powerful Pipeline for LLM Instruction Tuning (https://arxiv.org/abs/2408.14774)
- **What's New**: 이 논문에서는 다양하고 고품질의 SFT (Supervised Fine-Tuning) 데이터를 생성하기 위한 자동화된 접근법인 Instruct-SkillMix를 소개합니다.

- **Technical Details**: Instruct-SkillMix 파이프라인은 두 단계로 구성됩니다: (1) Skill Extraction: 기존 데이터셋 또는 모델을 직접 프롬프트하여 지도 학습에 필요한 '기술'을 추출합니다. (2) Data Generation: 강력한 LLM을 활용하여 무작위로 선택된 기술 조합을 가진 (instruction, response) 데이터를 생성합니다. 이 방법은 데이터의 다양성과 난이도를 증가시킵니다.

- **Performance Highlights**: Instruct-SkillMix로 생성된 데이터로 SFT를 수행했을 때 AlpacaEval 2.0, MT-Bench, WildBench와 같은 지침 준수 벤치마크에서 강력한 성능 향상을 기록하였습니다. LLaMA-3-8B-Base 모델은 단 4,000개의 예시로 AlpacaEval 2.0에서 42.76%의 길이 제어 승률을 달성하였고, 이는 SFT만 수행한 모델들 중에서 최첨단 성능을 보여줍니다.



### PAT: Pruning-Aware Tuning for Large Language Models (https://arxiv.org/abs/2408.14721)
- **What's New**: 이번 연구에서는 Pruning-Aware Tuning (PAT)이라는 혁신적인 패러다임을 제안하여 구조적 가지치기와 파인튜닝(fine-tuning)을 동시에 수행함으로써 모델 성능을 개선하는 접근 방식을 소개합니다.

- **Technical Details**: PAT는 Hybrid Sparsification Modules (HSMs)을 Attention과 FFN(Feed Forward Network) 구성 요소 사이에 통합하여 파라미터 효율성을 높이며, Hybird-Identity-Operator (HIO)를 통해 학습 가능한 파라미터 수를 줄입니다. 또한, Identity Loss (IL)를 적용하여 훈련의 강건성을 증대시키고 있습니다. 모든 HSM은 Unified Sparsification Mask (USM)로 통해 통합적으로 관리됩니다.

- **Performance Highlights**: Llama2-7b 모델을 기준으로 25% 가지치기를 통해 1.33배의 속도 향상과 함께 LoRA-64 모델보다 최대 1.26% 더 높은 정확도를 기록했습니다.



### Smart Multi-Modal Search: Contextual Sparse and Dense Embedding Integration in Adobe Express (https://arxiv.org/abs/2408.14698)
- **What's New**: 이번 논문은 Adobe Express 템플릿 검색에서 다중모드(multi-modal) 검색 시스템을 위한 새로운 아키텍처를 소개합니다. CLIP과 같은 다중모드 임베딩을 활용하여 텍스트와 이미지 검색을 직접 지원하면서도, 사용자 지리적 특성이나 최근성 같은 컨텍스트(contextual features)를 통합하는 데의 도전 과제를 다룹니다.

- **Technical Details**: 이 논문에서는 클라이언트의 검색 요구를 충족하기 위해 여러 다중모드 모델들을 사용하였으며, AB 테스트를 통해 임베딩 모델 선택, 매칭 및 랭킹의 역할, 밀집(dense)과 희소(sparse) 임베딩 간의 균형을 최적화하였습니다. AdobeCS 기술을 활용한 다중모드 검색 시스템은 약 30만 개의 템플릿 데이터에서 매우 효율적으로 작동하도록 설계되었습니다.

- **Performance Highlights**: 이번 연구는 희소, 밀집, 컨텍스트 특성을 활용하여 짧은 쿼리와 긴 쿼리 검색을 향상시키고, null 비율을 70% 이상 줄이며 클릭률(CTR)을 증가시키는 데 기여했습니다. 이러한 결과는 복잡한 쿼리에 대한 검색 시스템의 효과적인 개발이 가능하다는 통찰을 제공합니다.



### Relationships are Complicated! An Analysis of Relationships Between Datasets on the Web (https://arxiv.org/abs/2408.14636)
- **What's New**: 이번 논문은 웹에서의 데이터셋 간 관계를 탐구하며, 데이터셋 발견 및 사용 과정에서의 사용자 작업 중심으로 이러한 관계의 중요성을 강조합니다. 데이터셋 간의 관계를 이해하는 것은 메타데이터 이해만큼이나 중요하다는 점을 입증하고, 사용자 요구에 맞춘 포괄적인 분류체계를 제공합니다.

- **Technical Details**: 논문에서는 2.7백만 개의 데이터셋을 분석하여 유전자 기반(provenance-based) 관계를 포함한 데이터셋 간 관계의 분류체계를 개발하고, 이를 사용자 작업에 연결합니다. 기계 학습(machine learning) 기반의 방법을 사용하여 데이터셋 메타데이터 분석을 수행하였으며, 90%의 다중 클래스 분류 정확도를 달성하였습니다.

- **Performance Highlights**: 이 연구에서 연구자들은 데이터셋 간 관계의 20%가 적어도 하나의 다른 데이터셋과 연결되어 있음을 발견하였고, 이를 통해 데이터셋 메타데이터 개선의 필요성을 강조하였습니다. 또한, 전체 데이터셋의 관계를 탐구할 수 있는 중요한 계기를 마련함으로써 향후 연구에 있어 기준점을 설정하였습니다.



### MODOC: A Modular Interface for Flexible Interlinking of Text Retrieval and Text Generation Functions (https://arxiv.org/abs/2408.14623)
- **What's New**: 이번 연구에서는 MODOC라는 모듈형 사용자 인터페이스를 새롭게 소개합니다. 이는 대형 언어 모델(LLM) 기능을 활용하여 과학적 글쓰기에서 발생할 수 있는 잘못된 정보(confabulation)를 감지하는 데 도움을 주며, 정보 검색 및 텍스트 생성 기능을 통합하였습니다.

- **Technical Details**: MODOC는 5개의 모듈로 구성되어 있으며, 사용자는 실시간으로 수백만 개의 과학 문서에서 관련 정보를 검색할 수 있습니다. 또한 자유롭게 정의된 문맥화된 프롬프트를 사용하여 LLM 기반의 과학적 텍스트를 생성할 수 있는 기능을 제공합니다. MODOC는 신뢰성과 윤리를 고려하여 정보 검색과 생성 기능을 명확히 분리했습니다.

- **Performance Highlights**: MODOC는 과학적 글쓰기의 생산성을 향상시키기 위한 첫 번째 실용적인 시도로, 사용자에게 쉽고 직관적인 인터페이스를 제공하여 분석 및 작성의 인지 로드를 줄이는 데 기여합니다. 이는 특히 LLM으로 생성된 내용을 보다 윤리적으로 활용할 수 있도록 돕습니다.



### CURLoRA: Stable LLM Continual Fine-Tuning and Catastrophic Forgetting Mitigation (https://arxiv.org/abs/2408.14572)
Comments:
          Code available at this https URL

- **What's New**: CURLoRA는 CUR 행렬 분해를 활용하여 대형 언어 모델을 조정하는 새로운 접근 방식을 소개합니다. 이 방법은 연속 학습 중의 재앙적 망각을 완화하고 학습 가능한 매개변수 수를 줄이는 두 가지 주요 문제를 해결합니다.

- **Technical Details**: CURLoRA는 CUR decomposition을 사용하여 사전 훈련된 가중치 행렬을 분해하고 U 매트릭스를 제로 매트릭스로 초기화한 뒤 이를 조정합니다. 이 과정은 임PLICIT 정규화를 제공하며, 특정 데이터셋에서 실험을 통해 표준 LoRA보다 우수한 성능을 보였습니다.

- **Performance Highlights**: CURLoRA는 지속적인 조정 시에 기본 모델의 perplexity 점수를 유지하면서도 작업 정확도를 매우 높고 안정적으로 유지하는 성능을 입증했습니다. 특히 제한된 데이터 환경에서도 재앙적 망각을 잘 완화하여 안정적인 성능을 보여주었습니다.



### Revisiting Image Captioning Training Paradigm via Direct CLIP-based Optimization (https://arxiv.org/abs/2408.14547)
Comments:
          BMVC 2024

- **What's New**: 이 논문에서는 이미지 캡셔닝(Image Captioning)을 위한 새로운 훈련 패러다임인 Direct CLIP-Based Optimization (DiCO)를 제안합니다. 기존의 SCST(Self-Critical Sequence Training) 방식의 불안정성과 캡션 품질의 저하 문제를 해결하려는 접근입니다.

- **Technical Details**: DiCO는 사람과의 높은 상관관계를 가진 학습 가능한 캡셔닝 평가자로부터 증류된 보상 모델(reward model)을 공동 학습하고 최적화합니다. 이를 통해 원래 모델의 수렴을 방지하고, 캡셔너(Captioner) 내부에서 가중 분류 문제를 해결하여 다양한 캡션 품질 지표에서 동시에 최적화가 가능합니다.

- **Performance Highlights**: DiCO는 COCO 데이터셋에서 광범위한 실험을 하고, 현대적인 측정 지표에서 개선된 품질과 훈련의 안정성을 보여 주었으며, 전통적인 지표에서도 경쟁력 있는 성과를 유지했습니다. DiCO는 앤지 데이터셋 외에도 6개의 다른 이미지 캡셔닝 벤치마크에서 일반화 능력을 입증했습니다.



### LLMs as Zero-shot Graph Learners: Alignment of GNN Representations with LLM Token Embeddings (https://arxiv.org/abs/2408.14512)
- **What's New**: 이번 논문에서는 Token Embedding-Aligned Graph Language Model (TEA-GLM)이라는 새로운 프레임워크를 제안합니다. TEA-GLM은 GNN(Graph Neural Network)의 표현을 대형 언어 모델의 토큰 임베딩과 정렬하여, cross-dataset 및 cross-task 제로샷(zero-shot) 학습을 가능하게 합니다.

- **Technical Details**: TEA-GLM의 기본 구성 요소는 GNN과 LLM(Large Language Model)으로, GNN은 그래프에서 노드 표현을 도출하고 LLM은 노드 분류 및 링크 예측과 같은 제로샷 작업을 수행합니다. 이 프레임워크는 두 가지 주요 단계로 구성됩니다: GNN의 자기지도 학습과 GNN 표현을 고정된 수의 그래프 토큰 임베딩으로 변환하기 위한 선형 프로젝터(training a linear projector)입니다.

- **Performance Highlights**: TEA-GLM은 다양한 그래프 작업에서 통합 지침을 설계하여 모델의 일반화 기능을 향상시키며, 실험 결과에 따르면, TEA-GLM은 보이지 않는 데이터셋과 작업에 대해 최신 방법들보다 우수한 성능을 나타내었습니다.



### Unveiling the Statistical Foundations of Chain-of-Thought Prompting Methods (https://arxiv.org/abs/2408.14511)
Comments:
          150 pages, 18 figures, 3 tables

- **What's New**: 이 연구에서는 Chain-of-Thought (CoT) prompting와 그 변형을 통계적 추정 관점에서 분석하여 샘플 복잡성(sample complexity)에 대한 포괄적인 특성을 제공합니다.

- **Technical Details**: 이 논문에서는 다단계(latent variable) 모델을 도입하여 추론 과정을 캡슐화하며, 잠재 변수(latent variable)가 작업 정보를 인코딩합니다. CoT prompting에 의해 형성된 추정기는 대규모의 사전 훈련(pretraining) 데이터셋에서 베이지안(Bayesian) 추정기와 동등함을 보여줍니다. 또한, CoT 추정기의 통계적 오차는 두 가지 주요 구성 요소로 분해될 수 있음을 증명합니다: (i) CoT 프롬프트를 사용하여 진짜 작업을 추론함으로써 발생하는 프롬프트 오류(prompting error)와 (ii) 사전 훈련된 LLM의 통계적 오류입니다.

- **Performance Highlights**: 실험을 통해 다단계 추론 문제에서 타겟 분포(target distribution)를 근사하는 변환기 모델(transformer model)을 구성하고, 변환기 블록의 수가 증가함에 따라 오류가 기하급수적으로 감소함을 확인했습니다. CoT의 다른 변형, Self-Consistent CoT, Tree-of-Thought, Selection-Inference에 대한 분석도 포함되어 있으며, 이 방법들의 효율성을 넓은 관점에서 다룹니다.



### Empowering Pre-Trained Language Models for Spatio-Temporal Forecasting via Decoupling Enhanced Discrete Reprogramming (https://arxiv.org/abs/2408.14505)
- **What's New**: 본 논문에서는 시공간(time series) 예측을 위한 새로운 프로그래밍 프레임워크인 RePST를 제안합니다. 기존 접근 방식들이 공간적 의존성과 내재적 주파수 구성요소를 처리하는 데 한계를 보이는 반면, RePST는 이러한 문제를 해결하기 위해 주파수 도메인에서 시공간 동역학을 분리하는 접근 방식을 사용합니다.

- **Technical Details**: RePST 프레임워크는 Fourier 분석과 구조적 확산(operator)을 통해 입력된 시공간 데이터를 내재적 및 확산 신호로 분해합니다. 이를 통해 PLM(Pre-trained Language Models)이 더 잘 이해할 수 있는 데이터 표현을 생성하며, 차별화된 방식으로 확대된 어휘 공간에서 관련된 정보만을 선택하는 전략을 도입하여 정보 병목 현상을 방지합니다.

- **Performance Highlights**: 여러 실제 데이터셋 부문에서 수행한 광범위한 실험을 통해 RePST가 기존 최신 기술보다 뛰어난 성능 향상을 demonstrated하였고, 특히 데이터가 부족한 상황에서도 강력한 일반화 능력을 발휘함을 확인했습니다.



### A New Era in Computational Pathology: A Survey on Foundation and Vision-Language Models (https://arxiv.org/abs/2408.14496)
Comments:
          Initial Version

- **What's New**: 최근의 AI(인공지능) 발전은 Computational Pathology (CPath) 분야를 근본적으로 변화시켰습니다. 기초 모델(FMs) 및 비전-언어 모델(VLMs)을 통합하여 병리학자들의 진단 작업 흐름을 혁신하고 있습니다.

- **Technical Details**: 기초 모델(FMs)은 자가 지도 학습(self-supervised learning, SSL) 기법을 통해 다양한 과제에 적응할 수 있는 표현 공간을 학습합니다. 이와 더불어 VLMs는 자연어로 작성된 병리학 보고서를 활용하여 기존 모델의 성능을 개선하고 자연어 형태로 예측을 생성할 수 있습니다. 이 연구에서는 FMs와 VLMs의 구조 및 훈련 방식에 대한 자세한 정보를 제공합니다.

- **Performance Highlights**: FMs와 VLMs의 통합은 AI 병리학자가 되어 다양한 작업을 수행할 수 있는 가능성을 보여주며, 이는 최근 연구에서 나타난 성과들에 기반하고 있습니다. 또한 CPath 분야에 대한 최근 연구 증가를 통해 이들 모델의 중요성이 부각되고 있습니다.



### Agentic Retrieval-Augmented Generation for Time Series Analysis (https://arxiv.org/abs/2408.14484)
Comments:
          Paper was accepted for Undergraduate Consortium at ACM KDD, 2024. Please find the link: this https URL

- **What's New**: 이 논문은 복잡한 시공간 의존성과 분포 변화 문제를 해결하기 위해 에이전트 기반의 Retrieval-Augmented Generation(RAG) 프레임워크를 제안합니다. 이 프레임워크는 마스터 에이전트가 여러 전문 서브 에이전트를 조정하여 사용자 요청을 처리하는 방식으로, 과거 패턴에 대한 정보를 활용하여 예측을 개선합니다.

- **Technical Details**: 제안된 Agentic RAG 프레임워크는 계층적 다중 에이전트 구조를 채택합니다. 최상위 마스터 에이전트가 사용자 요청을 분석하고, 관련 서브 에이전트에게 작업을 위임합니다. 각 서브 에이전트는 특화된 시계열 과제를 위해 조정된 소형 언어 모델(SLMs)을 사용하고, 공유된 프롬프트 풀에서 관련 정보를 검색하여 예측 능력을 향상시킵니다. 이 과정에서 과거의 패턴을 기반으로 시계열 데이터의 복잡성을 다룰 수 있도록 설계되었습니다.

- **Performance Highlights**: 제안된 방안은 단일 고급 모델보다 유연성과 정확성을 제공하며, 여러 시계열 분석 작업에서 최첨단 성능을 기록하였습니다. 특히, 다차원 데이터나 이차원 데이터와 같은 다양한 데이터셋에서도 성능이 뛰어나며, 이는 기존의 작업 특정 맞춤형 방법보다 더 효과적입니다.



### Unboxing Occupational Bias: Grounded Debiasing of LLMs with U.S. Labor Data (https://arxiv.org/abs/2408.11247)
Comments:
          Accepted in AAAI Spring Symposium 2024

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)에서 사회적 편향(bias)이 어떻게 나타나는지를 실증적으로 분석하고, 미국 노동통계국(NBLS) 데이터를 활용한 편향 제거(debiasing) 메커니즘을 제안합니다. 이 연구는 다른 연구들과 달리 편향 감지의 연관성에 대한 전반적인 조사 없이 진행되었습니다.

- **Technical Details**: 우리는 2,500개의 샘플을 바탕으로 Falcon, GPT-Neo, Gemini 1.5 및 GPT-4o를 포함한 여러 LLM을 평가했습니다. 평가 방법으로는 KS 테스트(Kolmogorov-Smirnov test)와 ANOVA 테스트를 사용했으며, 다양한 작업 유형에서 LLM의 편향을 분석했습니다. 또한, Zero-Shot Prompting과 Few-Shot Prompting을 사용하여 LLM의 편향 감지 및 제거를 위한 방법론을 개발했습니다.

- **Performance Highlights**: 제안된 편향 제거 방법은 단 32개의 예를 이용해 평균 65%의 편향 감소를 달성했습니다. 이 연구는 LLM의 출력 결과가 NBLS 데이터와 상당한 차이를 보이는 것을 밝혔으며, 이는 공정성을 높이고 신뢰할 수 있는 LLM을 개발하는 데 중요한 기여를 합니다.



New uploads on arXiv(cs.IR)

### X-Reflect: Cross-Reflection Prompting for Multimodal Recommendation (https://arxiv.org/abs/2408.15172)
- **What's New**: 본 논문에서는 Cross-Reflection Prompting (X-Reflect)이라는 새로운 프레임워크를 도입하여, 텍스트와 이미지 간의 지원 및 상충 정보를 명시적으로 식별하고 조화롭게 통합하는 방식으로 추천 시스템의 성능을 향상시키고자 하였습니다.

- **Technical Details**: X-Reflect 방법은 LLM과 LMM에서 텍스트와 이미지 정보를 동시 처리하여, 두 모달리티 간의 상호 지원 혹은 상충 정보를 파악하고 조화롭게 통합하는 과정을 포함합니다. 이는 별도의 프롬프트 방식과 결합된 프롬프트 방식을 통해 구현됩니다.

- **Performance Highlights**: 두 개의 널리 사용되는 벤치마크에서 수행한 광범위한 실험을 통해, 제안된 방법이 기존의 프롬프트 기본선 대비 추천 정확도에서 일관되게 우수한 성능을 보임을 입증하였습니다.



### Measuring publication relatedness using controlled vocabularies (https://arxiv.org/abs/2408.15004)
Comments:
          Accepted for presentation at the 28th International Conference on Science, Technology and Innovation Indicators, 2024

- **What's New**: 이 논문은 다양한 연구 질문에 대한 정확성과 적합성을 평가하기 위한 종합적이고 직접적인 테스트가 없었던 기존의 controlled vocabulary 기반 관련성 측정을 검토하고, 새로운 측정 방법을 개발한 후 TREC Genomics 데이터를 사용하여 벤치마크 테스트를 수행합니다.

- **Technical Details**: 연구자들은 기존의 관련성 측정 방법을 수정하고 새로운 방법을 제시하며, TREC Genomics 데이터로 주제를 기반으로 한 벤치마크 평가를 통해 각 방법의 강점과 약점을 분석합니다. 특히 Ahlgren et al.(2020)의 방법과의 비교를 통해 다양한 연구 질문에 대해 어떤 방법이 가장 적합한지 논의합니다.

- **Performance Highlights**: 벤치마크 테스트 결과, 새로 제안된 측정 방법과 Ahlgren et al.(2020)의 방법이 각각 다른 강점과 약점을 가지고 있음을 보여줍니다. 이러한 결과는 학제 간 연구, 정보 검색, 과학의 클러스터링, 연구자의 주제 전환을 연구할 때 어떤 방법을 선택해야 하는지에 대한 논의를 제공합니다.



### Knowledge Discovery in Optical Music Recognition: Enhancing Information Retrieval with Instance Segmentation (https://arxiv.org/abs/2408.15002)
Comments:
          8 pages content and one references, accepted version at the International Conference on Knowledge Discovery and Information Retrieval 2024, Porto, Portugal

- **What's New**: 본 연구는 Optical Music Recognition (OMR) 분야에서 Mask R-CNN을 이용한 instance segmentation 기법을 적용하여 음악 기호의 탐지 및 구획을 개선하는 방법을 제안합니다. 이 방법을 통해 복잡한 Common Western Music Notation (CWMN)의 의미를 보다 잘 해석할 수 있습니다.

- **Technical Details**: OMR 시스템에 대한 성능 향상을 위해 instance segmentation과 같은 고급 심층 학습 기법을 도입합니다. 특히, Mask R-CNN을 사용하여 픽셀 수준의 음악 기호 분류를 수행하며, 전통적인 컴퓨터 비전 기법을 통한 staff detection 단계도 추가하여 음고 추정을 지원합니다. 이로써 낮은 연산 비용과 긴밀한 기호 분류를 달성합니다.

- **Performance Highlights**: DoReMi 및 MUSCIMA++ 데이터셋에서 수행한 평가 결과, 제안된 방법은 mAP가 최대 59.70%에 도달하는 등 고밀도 기호 환경에서도 뛰어난 성능을 보여주었습니다. 이러한 개선은 OMR 기술의 발전에 중요한 기여를 하며, 음악 데이터베이스에서의 효과적인 정보 검색 및 지식 발견을 지원합니다.



### MRSE: An Efficient Multi-modality Retrieval System for Large Scale E-commerc (https://arxiv.org/abs/2408.14968)
- **What's New**: 이번 논문에서는 MRSE(Multi-modality Retrieval System for Large Scale E-commerce)를 제안하여 텍스트 쿼리 및 아이템 이미지, 사용자 선호도를 통합하여 품질 높은 아이템 검색을 위한 솔루션을 제공합니다. MRSE는 기존의 단일 모달리티 시스템의 한계를 극복하며, 사용자 선호에 따라 매칭 성능을 향상시킵니다.

- **Technical Details**: MRSE는 Lightweight Mixture-of-Expert (LMoE) 모듈을 사용하여 텍스트와 이미지 간의 농축된 특징을 정렬하고, 사용자 프로필을 생성합니다. 핵심 기술로 VBert와 FtAtt를 이용해 모달리티 간 및 내의 특징을 개선하고, 하이브리드 손실 함수를 사용하여 고속 부정 샘플링을 통해 일관성과 강건성을 향상시킵니다.

- **Performance Highlights**: 대규모의 Shopee 데이터셋에서 실험한 결과, MRSE는 기존의 단일 모달리티 시스템에 비해 오프라인 관련성에서 18.9%, 온라인 핵심 지표에서 3.7%의 성능 향상을 보여주었습니다. 이로 인해 MRSE는 Shopee의 플랫폼 전반에서 기본 모델로 채택되었습니다.



### Tripl\`etoile: Extraction of Knowledge from Microblogging Tex (https://arxiv.org/abs/2408.14908)
Comments:
          42 pages, 6 figures

- **What's New**: 최근 지식 그래프(knowledge graph)를 문서에서 자동으로 추출하는 방법들이 많이 등장했습니다. 그러나 마이크로 블로깅 사이트나 뉴스와 같은 비전형적 텍스트 소스에서 엔티티와 관계를 모델링하는 데 어려움이 있었습니다. 이 논문에서는 소셜 미디어 플랫폼의 마이크로 블로깅 게시물에서 개방 도메인 엔티티를 포함하는 지식 그래프를 추출하기 위한 개선된 정보 추출 파이프라인을 제안합니다.

- **Technical Details**: 이 파이프라인은 의존 구문 분석(dependency parsing)을 활용하고, 단어 임베딩(word embeddings)에 대한 계층적 클러스터링(hierarchical clustering)을 통해 비지도 학습(unsupervised) 방식으로 엔티티 관계를 분류합니다. 연구는 디지털 트랜스포메이션 관련 10만 개의 트윗에서 의미적 삼중(semantic triples)을 추출하는 사례를 제공합니다.

- **Performance Highlights**: 시스템이 생성한 트리플은 95% 이상의 정밀도를 보이고 있으며, 유사한 파이프라인보다 약 5% 더 높은 정밀도를 기록하고, 상대적으로 많은 수의 트리플을 생성했습니다.



### Graph and Sequential Neural Networks in Session-based Recommendation: A Survey (https://arxiv.org/abs/2408.14851)
- **What's New**: 최근 추천 시스템(Recommendation Systems, RSs)의 발전에 따라, 세션 기반 추천(Session-based Recommendation, SR) 분야가 각광받고 있습니다. 이 논문에서는 SR의 정의와 특성을 명확히 하여 기초적인 방법론을 제시하고, SR 분야의 최근 연구 동향을 조망합니다.

- **Technical Details**: 세션 기반 추천은 사용자의 단기 선호도를 포착하는 데 초점을 맞추며, 두 가지 주요 접근법: sequential neural networks와 graph neural networks (GNNs)를 분류하고 분석합니다. 특히, sequential neural networks는 입력 세션을 시퀀스로 모델링하고 아이템 간의 순서 종속성을 캡처하며, GNNs는 세션 기반 그래프 구조를 전제로 정보 전파 및 집계로 추천을 수행합니다.

- **Performance Highlights**: SR 연구는 2019년 이후 급격히 증가하였으며, 다양한 머신러닝 및 딥러닝 기술들이 적용되고 있습니다. 특히, sequential neural networks와 GNNs는 세션 기반 추천에서 중요한 성과를 보이고 있으며, 각각의 방법론에 대한 체계적인 분류와 비교가 이루어졌습니다.



### Smart Multi-Modal Search: Contextual Sparse and Dense Embedding Integration in Adobe Express (https://arxiv.org/abs/2408.14698)
- **What's New**: 이번 논문은 Adobe Express 템플릿 검색에서 다중모드(multi-modal) 검색 시스템을 위한 새로운 아키텍처를 소개합니다. CLIP과 같은 다중모드 임베딩을 활용하여 텍스트와 이미지 검색을 직접 지원하면서도, 사용자 지리적 특성이나 최근성 같은 컨텍스트(contextual features)를 통합하는 데의 도전 과제를 다룹니다.

- **Technical Details**: 이 논문에서는 클라이언트의 검색 요구를 충족하기 위해 여러 다중모드 모델들을 사용하였으며, AB 테스트를 통해 임베딩 모델 선택, 매칭 및 랭킹의 역할, 밀집(dense)과 희소(sparse) 임베딩 간의 균형을 최적화하였습니다. AdobeCS 기술을 활용한 다중모드 검색 시스템은 약 30만 개의 템플릿 데이터에서 매우 효율적으로 작동하도록 설계되었습니다.

- **Performance Highlights**: 이번 연구는 희소, 밀집, 컨텍스트 특성을 활용하여 짧은 쿼리와 긴 쿼리 검색을 향상시키고, null 비율을 70% 이상 줄이며 클릭률(CTR)을 증가시키는 데 기여했습니다. 이러한 결과는 복잡한 쿼리에 대한 검색 시스템의 효과적인 개발이 가능하다는 통찰을 제공합니다.



### Federated User Preference Modeling for Privacy-Preserving Cross-Domain Recommendation (https://arxiv.org/abs/2408.14689)
- **What's New**: 이 논문에서는 개인정보 보호를 고려한 새로운 Cross-Domain Recommendation 프레임워크인 Federated User Preference Modeling (FUPM)을 제안합니다. 이 방법은 사용자와 아이템 간의 상호작용 데이터뿐만 아니라 추가 데이터를 활용하여 사용자 선호를 포괄적으로 탐구합니다.

- **Technical Details**: FUPM은 네 가지 주요 모듈로 구성되어 있습니다: 1) Representation Learning Module - 사용자 및 아이템 ID와 리뷰 텍스트의 임베딩을 학습합니다. 2) Comprehensive Preference Exploration Module - 사용자 선호를 완전히 탐구하며, Contrastive Feature Alignment와 Potential Interest Mining 두 가지 구성 요소로 나뉩니다. 3) Private Preference Transfer Module - 다중 도메인 간 사용자 선호를 이동시키면서 개인 정보 유출을 방지합니다. 4) Prediction Module - 사용자 선호를 아이템에 대해 예측합니다. 각 모듈은 안전하고 효율적인 지식 전이를 지원합니다.

- **Performance Highlights**: Amazon과 Douban 데이터셋을 기반으로 한 실험 결과, FUPM은 기존 SOTA(SOTA는 State of the Art의 약어) 기법에 비해 뛰어난 성능을 보여줍니다.



### Bridging the Gap: Unpacking the Hidden Challenges in Knowledge Distillation for Online Ranking Systems (https://arxiv.org/abs/2408.14678)
- **What's New**: 본 논문에서는 추천 시스템에 대한 Knowledge Distillation (KD)의 독특한 도전 과제를 해결하고, 데이터 분포 변화, 최적의 교사 구성 찾기, 교사 레이블의 효율적인 공유를 통한 모델 개선을 제시합니다.

- **Technical Details**: KD는 일반적으로 하드 레이블과 소프트 레이블을 사용하는 방법으로, 추천 시스템의 데이터는 빠르게 변동하여 이 방식이 최적이 아닐 수 있습니다. 따라서 본 연구에서는 보조 작업 기반의 KD 전략을 통해 교사의 편향을 학생에게 전달하지 않도록 했습니다. 교사를 지속적으로 업데이트하여 학생에게 최신 정보를 제공합니다.

- **Performance Highlights**: 교사 모델과 학생 모델 간의 구성을 최적화하여, E(LTV) 손실을 0.4% 줄이는 성과를 기록했습니다. 학생 모델은 상대적으로 작은 교사(학생의 2배 크기)로부터도 성능이 향상되는 것을 보여주었습니다.



### Relationships are Complicated! An Analysis of Relationships Between Datasets on the Web (https://arxiv.org/abs/2408.14636)
- **What's New**: 이번 논문은 웹에서의 데이터셋 간 관계를 탐구하며, 데이터셋 발견 및 사용 과정에서의 사용자 작업 중심으로 이러한 관계의 중요성을 강조합니다. 데이터셋 간의 관계를 이해하는 것은 메타데이터 이해만큼이나 중요하다는 점을 입증하고, 사용자 요구에 맞춘 포괄적인 분류체계를 제공합니다.

- **Technical Details**: 논문에서는 2.7백만 개의 데이터셋을 분석하여 유전자 기반(provenance-based) 관계를 포함한 데이터셋 간 관계의 분류체계를 개발하고, 이를 사용자 작업에 연결합니다. 기계 학습(machine learning) 기반의 방법을 사용하여 데이터셋 메타데이터 분석을 수행하였으며, 90%의 다중 클래스 분류 정확도를 달성하였습니다.

- **Performance Highlights**: 이 연구에서 연구자들은 데이터셋 간 관계의 20%가 적어도 하나의 다른 데이터셋과 연결되어 있음을 발견하였고, 이를 통해 데이터셋 메타데이터 개선의 필요성을 강조하였습니다. 또한, 전체 데이터셋의 관계를 탐구할 수 있는 중요한 계기를 마련함으로써 향후 연구에 있어 기준점을 설정하였습니다.



### Into the Unknown Unknowns: Engaged Human Learning through Participation in Language Model Agent Conversations (https://arxiv.org/abs/2408.15232)
- **What's New**: 본 논문은 사용자가 질문을 던지지 않고도 대화형 상호작용을 통해 정보를 발견할 수 있도록 하는 새로운 시스템, Collaborative STORM (Co-STORM)를 소개합니다.

- **Technical Details**: Co-STORM은 다수의 언어모델(LM) 에이전트 간의 대화를 관찰하고 방향을 제시할 수 있게 해주며, 사용자가 잘 알지 못하는 정보를 자연스럽게 발견할 수 있도록 돕습니다. 또한, Co-STORM은 발견된 정보를 동적인 마인드 맵(mind map)으로 정리하여 사용자가 쉽게 소통할 수 있도록 지원합니다.

- **Performance Highlights**: Co-STORM은 베이스라인(baseline) 방법들과 비교하여 담화 추적(discourse trace) 및 보고서(report) 품질에서 우수한 성능을 보였으며, 인간 평가에서 참가자의 70%가 Co-STORM을 검색 엔진보다 선호하고, 78%가 RAG 챗봇보다 Co-STORM을 선호한다고 응답했습니다.



### Writing in the Margins: Better Inference Pattern for Long Context Retrieva (https://arxiv.org/abs/2408.14906)
- **What's New**: 본 논문에서는 Retrieval-oriented (검색 지향) 작업에서 긴 입력 시퀀스를 최적화하기 위해 설계된 새로운 inference pattern인 WiM(Writing in the Margins)을 소개합니다.

- **Technical Details**: 이 접근법은 key-value cache의 chunked prefill을 활용하여 segment-wise inference를 수행합니다. 이를 통해 모델이 특정 작업으로 안내하는 중간 정보(‘margins’)의 생성 및 분류와 함께 넓은 맥락을 효율적으로 처리할 수 있게 합니다.

- **Performance Highlights**: WiM은 reasoning skills (추론 능력)에서 평균 7.5%의 정확도 향상(HotpotQA, MultiHop-RAG)을 제공하며, aggregation tasks (집계 작업)에서는 F1-score가 30.0% 이상 증가하는 효과를 보입니다.



### Personalized Video Summarization using Text-Based Queries and Conditional Modeling (https://arxiv.org/abs/2408.14743)
Comments:
          Ph.D. thesis, 137 pages

- **What's New**: 최근 비디오 콘텐츠가 폭발적으로 증가하면서 자동 비디오 요약의 필요성이 커지고 있습니다. 이 논문은 텍스트 기반 쿼리와 조건부 모델링을 결합하여 사용자 맞춤형 요약을 제공하는 방법론을 제안합니다.

- **Technical Details**: 이 연구에서는 다중 모달 딥 러닝(multi-modal deep learning) 접근 방식을 활용하여 시각 정보와 텍스트 쿼리를 통합하고, 문맥화된 단어 임베딩(contextualized word embeddings)과 주의 네트워크(attention networks)를 통해 텍스트 기반 쿼리 표현을 개선합니다.

- **Performance Highlights**: 제안된 방법은 기존의 최첨단 방법에 비해 더 나은 성능을 보여주며, 특히 비디오 요약의 품질을 평가할 때 정확도와 F1 점수를 사용하여 평가합니다.



### Snap and Diagnose: An Advanced Multimodal Retrieval System for Identifying Plant Diseases in the Wild (https://arxiv.org/abs/2408.14723)
- **What's New**: 새로운 연구에서는 Snap’n Diagnose라는 멀티모달 이미지 검색 시스템을 제안하여, 농업 분야에서의 식물 질병 인식의 효율성을 높이고자 합니다. 이 시스템은 질병 이미지를 업로드하거나, 텍스트 설명을 제공함으로써 관련 이미지를 검색할 수 있습니다.

- **Technical Details**: Snap’n Diagnose는 세계 최대 규모의 PlantWild 데이터셋을 바탕으로 하며, 89개의 카테고리에 걸쳐 18,000장 이상의 실제 식물 질병 이미지를 포함하고 있습니다. CLIP 기반의 비전-언어 모델을 활용하여 질병 설명과 이미지를 동일한 잠재 공간(latent space)으로 인코딩하고, 사용자 질의를 통해 관련 이미지를 효과적으로 검색합니다.

- **Performance Highlights**: Snap’n Diagnose는 다양한 평가 지표에서 Zero-shot CLIP 모델을 능가하는 뛰어난 성능을 보이며, Top-1, Top-5, Top-10 정확도 및 평균 정밀도(mAP)에서 지속적으로 우수한 결과를 나타냈습니다. 이는 식물 질병 인식에 있어 실용적인 도구로서의 가능성을 입증합니다.



### KGPrune: a Web Application to Extract Subgraphs of Interest from Wikidata with Analogical Pruning (https://arxiv.org/abs/2408.14658)
Comments:
          Accepted as a demo paper at ECAI 2024

- **What's New**: KGPrune은 사용자가 관심 있는 시드 엔티티와 탐색할 속성을 제공하면, 이웃 서브그래프를 추출할 수 있는 웹 애플리케이션입니다. 이 애플리케이션은 수학적 추론을 기반으로 하는 경량의 가지치기 알고리즘을 채택하여 관련 이웃만 유지하고 관련 없는 이웃은 제거합니다.

- **Technical Details**: KGPrune은 Wikidata를 기반으로 하여 설계되었으며, 사용자는 관심 있는 시드 엔티티와 속성의 QID 및 PID를 포함하는 두 개의 CSV 파일을 업로드해야 합니다. 알고리즘은 주어진 시드 엔티티의 이웃을 탐색하고, 각각의 이웃에 대해 유사성 패턴을 바탕으로 유지 또는 삭제 결정을 내립니다. 이를 통해 사용자는 관련 정보만 포함된 서브그래프를 얻을 수 있습니다.

- **Performance Highlights**: KGPrune은 다양한 도메인에서의 사용을 지원하며, 특히 기업 지식 그래프를 구축하거나 약탈된 예술 작품에 대한 지식을 추출하는 데 유용함을 보여줍니다. 이 애플리케이션은 사용자에게 각 결과를 시각화하거나 JSON 또는 RDF 형식으로 다운로드할 수 있는 기능을 제공합니다.



### MODOC: A Modular Interface for Flexible Interlinking of Text Retrieval and Text Generation Functions (https://arxiv.org/abs/2408.14623)
- **What's New**: 이번 연구에서는 MODOC라는 모듈형 사용자 인터페이스를 새롭게 소개합니다. 이는 대형 언어 모델(LLM) 기능을 활용하여 과학적 글쓰기에서 발생할 수 있는 잘못된 정보(confabulation)를 감지하는 데 도움을 주며, 정보 검색 및 텍스트 생성 기능을 통합하였습니다.

- **Technical Details**: MODOC는 5개의 모듈로 구성되어 있으며, 사용자는 실시간으로 수백만 개의 과학 문서에서 관련 정보를 검색할 수 있습니다. 또한 자유롭게 정의된 문맥화된 프롬프트를 사용하여 LLM 기반의 과학적 텍스트를 생성할 수 있는 기능을 제공합니다. MODOC는 신뢰성과 윤리를 고려하여 정보 검색과 생성 기능을 명확히 분리했습니다.

- **Performance Highlights**: MODOC는 과학적 글쓰기의 생산성을 향상시키기 위한 첫 번째 실용적인 시도로, 사용자에게 쉽고 직관적인 인터페이스를 제공하여 분석 및 작성의 인지 로드를 줄이는 데 기여합니다. 이는 특히 LLM으로 생성된 내용을 보다 윤리적으로 활용할 수 있도록 돕습니다.



New uploads on arXiv(cs.CV)

### Drone-assisted Road Gaussian Splatting with Cross-view Uncertainty (https://arxiv.org/abs/2408.15242)
Comments:
          BMVC2024 Project Page: this https URL Code: this https URL

- **What's New**: 이 논문에서는 자율주행 시뮬레이션을 위한 드론 기반의 도로 장면 합성을 위한 새로운 패러다임을 제시합니다. 특히, 항공 이미지와 지상 이미지를 통합하여 포괄적인 장면 재구성을 가능하게 하고, 불확실성을 인식하는 훈련 방법을 설계하여 지상 이미지의 학습 성능이 저조한 영역에서 항공 이미지가 도움을 줄 수 있도록 하였습니다.

- **Technical Details**: 제안된 방법은 cross-view uncertainty라는 새로운 개념을 도입하여, 3D Gaussian Splatting (3D-GS) 훈련 과정에서 각 픽셀의 기여도를 가중하여 학습하게 합니다. 기존의 naive joint training과는 달리, 항공 이미지를 평가하는 데 있어 무관한 부분을 제외하여, 도로 장면 합성 중 뷰 이동(view shifting) 및 회전(rotation)에 대한 성능을 향상시켰습니다.

- **Performance Highlights**: 광범위한 실험을 통해, 드론 지원 도로 장면 합성을 위한 불확실성 인지 훈련 방법이 기존의 방법보다 양적 및 질적으로 월등한 성능을 보여주었습니다. 이 연구는 자율주행 시뮬레이션과 같은 다양한 응용 분야에 큰 잠재력을 가지고 있습니다.



### GenRec: Unifying Video Generation and Recognition with Diffusion Models (https://arxiv.org/abs/2408.15241)
Comments:
          17 pages, 6 figures, 7 tables

- **What's New**: 본 논문은 비디오 확산 모델을 활용하여 비디오 생성과 인식을 동시에 최적화하는 GenRec이라는 새로운 통합 프레임워크를 소개합니다. 이는 랜덤 프레임 조건화를 통해 일반화된 공간-시간 표현을 학습합니다.

- **Technical Details**: GenRec 모델은 Stable Video Diffusion(SVD) 모델을 기반으로 하며, 주어진 비디오의 임의 프레임에 조건을 부여하고 나머지 프레임을 마스킹하여 두 작업 간의 학습 과정을 효과적으로 연결합니다.

- **Performance Highlights**: GenRec은 SSV2에서 75.8%, K400에서 87.2%의 인식 정확도를 달성하며, 이미지-비디오 생성에서 최상의 클래스 조건 부여 결과를 나타냅니다. 제한된 프레임만 관찰 가능한 상황에서도 뛰어난 강건성을 보여줍니다.



### Generative Inbetweening: Adapting Image-to-Video Models for Keyframe Interpolation (https://arxiv.org/abs/2408.15239)
Comments:
          project page: this https URL

- **What's New**: 본 연구에서는 두 개의 키 프레임 사이에 일관된 동작을 가진 비디오 시퀀스를 생성하는 새로운 방법을 제안합니다. 이를 위해 기존의 대규모 이미지-비디오 Diffusion 모델을 조정하여 키 프레임 보간을 수행합니다.

- **Technical Details**: 이 방법은 경량의 미세 조정(fine-tuning) 기술을 사용하여 기존 모델을 역방향으로 동작을 예측하도록 조정합니다. 모델은 두 개의 키 프레임에서 시작하여 각기 다른 방향으로 동작을 생성한 후, 이를 결합하여 일관된 비디오를 생성합니다.

- **Performance Highlights**: 실험 결과, 본 방법이 기존의 Diffusion 기반 보간 방법과 전통적인 프레임 보간 기술보다 더 뛰어난 성능을 보임을 입증했습니다.



### Learning-based Multi-View Stereo: A Survey (https://arxiv.org/abs/2408.15235)
- **What's New**: 최신 연구에서는 3D 복원(3D reconstruction)의 필수 요소인 Multi-View Stereo (MVS) 알고리즘을 깊이 추정(depth estimation) 기반 방법으로 분류하고, 해당 방법의 성능을 다양한 벤치마크에서 평가하는 포괄적인 리뷰를 제공합니다.

- **Technical Details**: 본 연구에서는 MVS의 주요 구성 요소들인 카메라 보정(camera calibration), 뷰 선택(view selection), 멀티뷰 깊이 추정(multi-view depth estimation), 깊이 융합(depth fusion)에 대해 설명합니다.

- **Performance Highlights**: 딥러닝의 성공 이후, 기존 방법과 비교하여 높은 성능을 보이는 새로운 학습 기반 MVS 방법들이 제안되었습니다. 특히, 깊이 맵 기반 방법이 MVS의 주요 범주로써 유연성과 확장성이 뛰어난 것으로 나타났습니다.



### Leveraging Hallucinations to Reduce Manual Prompt Dependency in Promptable Segmentation (https://arxiv.org/abs/2408.15205)
Comments:
          We propose using hallucinations as prior knowledge to extract and validate task-related information, which helps generate instance-specific prompts for reducing reliance on manual prompts in promptable segmentation

- **What's New**: 이번 논문에서는 기존의 instance-specific manual prompts를 최소화하기 위해, task-generic promptable segmentation을 도입하였습니다. 이는 여러 이미지를 동일한 태스크에서 분할하기 위해 단일 task-generic prompt를 사용하는 방법입니다.

- **Technical Details**: 제안된 ProMaC(Iterative Prompt-Mask Cycle Generation) 프레임워크는 prompt generator와 mask generator를 포함합니다. 프로세스에는 multi-scale chain-of-thought prompting이 포함되며, 이를 통해 hallucinations을 탐색하고 세부적인 instance-specific prompts를 생성합니다.

- **Performance Highlights**: 5개의 벤치마크 실험에서 ProMaC의 효과를 입증하였고, 다양한 12개의 데이터셋과 22개의 기존 모델에 대한 비교 평가를 통해 유의미한 개선을 확인하였습니다.



### An Investigation on The Position Encoding in Vision-Based Dynamics Prediction (https://arxiv.org/abs/2408.15201)
Comments:
          13 pages, 4 tables, and 3 figures. Accepted to ECCV2024 eXCV workshop

- **What's New**: 이 논문은 객체의 동적 예측(dynamics prediction)에서 객체 추상(object abstract)으로서 바운딩 박스(bounding box)의 사용에 대한 탐구를 강화하고, 바운딩 박스가 위치(position) 정보를 암묵적으로 어떻게 인코딩하는지를 연구합니다.

- **Technical Details**: 연구자는 지역 관심 영역(Region of Interest, RoI) 풀링(RoI Pooling) 작업을 통해 바운딩 박스를 사용하여 객체 상태 피쳐를 추출하는 방법을 분석했습니다. RPCIN(Region Proposal Convolutional Internation Network) 모델을 사용하여 다양한 패딩(padding) 설정 하에서 입력 데이터의 환경 맥락에 따른 동적 예측 성능을 조사했습니다.

- **Performance Highlights**: 모델은 환경 컨텍스트가 불변일 때 가장 적절한 패딩 설정으로 객체 위치 정보를 효과적으로 인코딩할 수 있음을 보여주었습니다. 하지만 환경 조건이 변할 경우 단순한 바운딩 박스만으로는 충분한 예측 성능을 나타내지 못하고, 향후 설명 가능한 모델 개발이 필요하다는 점을 강조했습니다.



### PoseWatch: A Transformer-based Architecture for Human-centric Video Anomaly Detection Using Spatio-temporal Pose Tokenization (https://arxiv.org/abs/2408.15185)
- **What's New**: PoseWatch라는 새로운 변형기 기반 아키텍처를 소개하며, 인체 중심의 포즈 기반 비디오 이상 탐지(VAD)를 위한 혁신적인 방법론을 제시합니다.

- **Technical Details**: PoseWatch는 Spatio-Temporal Pose and Relative Pose (ST-PRP) 토크나이제이션 방식을 채택하여 시간을 통한 인간 동작 표현을 개선하고, Unified Encoder Twin Decoders (UETD) 구조를 통해 비디오 데이터에서 이상 행동 탐지를 향상시킵니다.

- **Performance Highlights**: PoseWatch는 여러 벤치마크 데이터셋에서 기존 방법들을 일관되게 초월하며, 80.67%의 평균 수신자 조작 특성 곡선(ROC AUC) 점수를 달성, 평균 동등 오류율(EER)은 0.27로 설정하여 새로운 성과를 기록합니다.



### A Review of Transformer-Based Models for Computer Vision Tasks: Capturing Global Context and Spatial Relationships (https://arxiv.org/abs/2408.15178)
- **What's New**: 이 논문에서는 변형기(Transformer) 모델이 자연어 처리(NLP)에서의 성공을 바탕으로 컴퓨터 비전(Computer Vision) 작업에 적용될 수 있는 가능성을 탐구하고 있습니다. 특히, 다양한 transformer 아키텍처와 그들이 시각적 데이터 분석을 위해 어떻게 발전했는지를 조명합니다.

- **Technical Details**: Vision Transformer(ViT)와 DEtection TRansformer(DETR) 등의 모델들이 포함되어 있으며, transformer 모델은 이미지를 패치(patch)로 나누고 self-attention 메커니즘을 사용하여 전체 이미지를 한 번에 처리합니다. 이렇게 함으로써 GOT(larger context)와 spatial relationships를 효과적으로 포착할 수 있습니다.

- **Performance Highlights**: Transformer 기반 모델은 이미지 분류(image classification), 객체 탐지(object detection), 세분화(segmentation)와 같은 다양한 컴퓨터 비전 작업에서 뛰어난 성능을 보이며, 최근 연구 방향과 응용 가능성을 논의합니다.



### Empowering Sign Language Communication: Integrating Sentiment and Semantics for Facial Expression Synthesis (https://arxiv.org/abs/2408.15159)
- **What's New**: 이 연구는 수화(SL)의 생산 향상을 위해 비언어적 요소인 얼굴 표정의 합성을 중점적으로 다루고 있습니다. 기존 연구들이 수동 제스처에 중점을 두었던 반면, 이번 논문은 감정 정보의 통합을 통해 얼굴 표정 생성을 개선하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 주요기술로는 Generative Latent Optimization (GLO)을 활용하여 감정 및 의미론적 특성을 고려한 의미 있는 표현 공간에서 샘플링하는 방법을 포함합니다. 또한, 새로운 평가 지표인 Frechet Expression Distance (FED)를 통해 생성된 얼굴 표정의 품질을 평가합니다. 모델은 얼굴 표정을 생성하기 위해 그래프 피라미드 구조를 기반으로 설계되었습니다.

- **Performance Highlights**: 다양한 실험 결과, 이 방법은 How2Sign 및 PHOENIX14T 데이터셋에서 기존 최첨단 기술을 초월하는 우수한 성능을 보였습니다. 또한, 이 연구는 텍스트 데이터만을 사용하여 감정 기반 얼굴 표정을 생성할 수 있는 방법을 제안하여 Sign Language Production 분야의 새로운 가능성을 보여줍니다.



### A Preliminary Exploration Towards General Image Restoration (https://arxiv.org/abs/2408.15143)
- **What's New**: 이번 연구에서는 General Image Restoration (GIR)이라는 문제를 제시하며, 다양한 이미지 복원 과제를 통합하여 실제 응용에서 발생할 수 있는 일반화 문제와 복잡한 변형을 해결하고자 합니다. GIR은 이미지 노이즈 제거, 블러 제거, 비 오는 이미지 복원 및 초해상도와 같은 개별 이미지 복원 작업을 포괄합니다.

- **Technical Details**: GIR의 주요 목표는 복잡하고 예측할 수 없는 변형을 포함한 모든 유형의 변형된 이미지를 자연스러운 클리어 이미지로 변환하는 것입니다. GIR 모델은 데이터셋 구축, 평가 프로토콜 설정, 기본 모델 개발 및 모델 해석을 포함하여 전체 구조를 제시합니다. 본 연구는 기존 기법들의 효과 및 실제적 도전과제를 폭넓게 평가하여 GIR의 유효성을 강조합니다.

- **Performance Highlights**: GIR 문제는 심층 모델의 일반화 능력을 시험하는 중요한 기준점이며, 복잡한 현실적 과제를 처리할 수 있는 강력한 모델의 필요성이 대두되고 있습니다. 연구 결과, GIR이 기존 기술과의 차별성을 갖고 있으며, 일반화를 위한 새로운 방향성을 제공할 수 있음을 보여줍니다.



### T-FAKE: Synthesizing Thermal Images for Facial Landmarking (https://arxiv.org/abs/2408.15127)
Comments:
          22 pages, 12 figures, Philipp Flotho and Moritz Piening share equal contribution

- **What's New**: 이 논문은 T-FAKE 데이터셋을 소개하며, 이는 얼굴 분석을 위한 첫 번째 대규모 합성 열 데이터셋으로, 희소(sparse) 및 밀집(dense) 랜드마크가 포함되어 있습니다.

- **Technical Details**: T-FAKE 데이터셋 생성을 위해 RGB2Thermal 손실 함수가 제안되었습니다. 이 함수는 RGB-열 쌍의 작은 하위 집합을 기반으로 열 얼굴 생성을 제어하는 감독 데이터 항목, 생성된 합성 열 이미지와 실제 열 이미지 간의 패치 분포 정렬을 위한 Wasserstein 거리 항목, 다양한 얼굴 영역에 대한 임상 온도 통계에 대한 사전 정보를 포함하는 세 가지 주요 요소로 구성됩니다.

- **Performance Highlights**: 모델은 이전 방법들과 비교하여 열 랜드마크 예측 벤치마크에서 우수한 성능을 보이며, RGB 이미지에서의 성능도 최첨단 RGB 방법과 유사하게 유지합니다. 70점 및 478점 랜드마크 주석에 대한 높은 정확성을 보여줍니다.



### Machine Learning for Methane Detection and Quantification from Space -- A survey (https://arxiv.org/abs/2408.15122)
- **What's New**: 이번 연구는 메탄( methane) 감지 센서에 대한 최신 정보를 확장하며, 기계 학습( ML) 접근 방식을 포함한 전통적인 방법을 검토합니다. 특히, ME 접근법이 전통적인 방법에 비해 높은 성능을 보여주는 것을 강조합니다.

- **Technical Details**: 연구에서는 메탄의 잔여 검출( detection) 및 배출률 추정( emission rate estimation)을 위해 CNN( convolutional neural networks) 및 Transformer 아키텍처에 기반한 ML 모델의 아키텍처와 데이터 사용에 대해 논의합니다. 메탄 감지는 픽셀당 메탄 컬럼 농도를 감지& 계산, 메탄 플룸(segmenting methane plumes) 분할, 배출원 위치 및 배출률 추정 등 세 가지 작업으로 나뉘며, 이러한 작업은 전통적인 방법과 ML 접근방식 간에 큰 차이를 보입니다.

- **Performance Highlights**: ML 모델은 특히 CNN 기반 접근법이 전통적인 방법보다 우수한 성능을 나타냅니다. 메탄 발생의 주요 원인인 농업, 에너지 및 폐기물 분야에서의 데이터 분석을 통해, ML 접근 방식은 유효 전반에 걸쳐 메탄 배출의 감지 및 추정 생성을 위한 효과성을 입증하고 있습니다.



### Urdu Digital Text Word Optical Character Recognition Using Permuted Auto Regressive Sequence Modeling (https://arxiv.org/abs/2408.15119)
- **What's New**: 이번 연구는 디지털 우르두( اردو ) 텍스트 인식을 위한 혁신적인 단어 수준의 Optical Character Recognition (OCR) 모델을 소개합니다.

- **Technical Details**: 이 모델은 transformer 기반 아키텍처와 attention 메커니즘을 활용하여 약 160,000개의 우르두 텍스트 이미지로 훈련되었으며, 문자 오류율(Character Error Rate, CER)은 0.178로 보고되었습니다. 이 모델의 독특한 아키텍처는 permuted autoregressive sequence (PARSeq) 모델을 포함하여, 양방향 맥락 정보를 활용한 맥락 인식 추론 및 반복적 개선을 가능하게 합니다.

- **Performance Highlights**: 다양한 우르두 텍스트 스타일, 글꼴 및 변형을 처리할 수 있는 능력 덕분에 실제 응용에서의 적합성이 향상되었습니다. 그러나 블러 처리된 이미지, 비수평 방향, 패턴이나 선, 다른 텍스트의 겹침 처리에는 한계가 있으며, 이로 인해 때때로 최적의 성능을 발휘하지 못할 수 있습니다.



### Few-Shot Unsupervised Implicit Neural Shape Representation Learning with Spatial Adversaries (https://arxiv.org/abs/2408.15114)
Comments:
          ICML 2024

- **What's New**: 이번 연구에서는 Sparse 3D point clouds에서 Implicit Neural Representations를 이용하여 Neural Signed Distance Functions (SDF)를 개선하기 위한 새로운 정규화 기법을 도입하였습니다. 과거의 방법들은 일반적으로 Smoothness Priors를 활용했으나, 본 연구는 Adversarial Samples를 통해 SDF 학습을 향상시키는 방법을 제안하였습니다.

- **Technical Details**: 이 연구는 Adversarial Training 기법을 활용하여 원래의 질의 포인트 주변에서 다양성을 갖는 Adversarial Samples를 생성합니다. 그리고 이러한 샘플들은 SDF의 최소화 과정에 통합하여 학습을 정규화하고 오버피팅을 방지하는 데 활용됩니다. 본 연구에서는 네트워크가 Sparse 한 입력 데이터에서 발생할 수 있는 오류를 줄이기 위해 Spatial Gradient, Lipschitz Regularization 등의 기존 기법들을 고려하였습니다.

- **Performance Highlights**: 다양한 실험을 통해 제안된 방법이 기존 Baseline 및 최신 기술 대비 우수한 성능을 보이며, 특히 형태 예측이 가장 어려운 세부 구조와 신체 말단 부분에서 성능이 크게 증가한 것을 확인하였습니다. 제안된 방법은 Dense Reconstruction 환경에서도 유용하며, 평가 과정에서의 안정성을 높여줍니다.



### AnomalousPatchCore: Exploring the Use of Anomalous Samples in Industrial Anomaly Detection (https://arxiv.org/abs/2408.15113)
Comments:
          Accepted at the 2nd workshop on Vision-based InduStrial InspectiON (VISION) @ ECCV

- **What's New**: 최신 연구에서는 기존의 이상 탐지 시스템인 AnomalousPatchCore(APC)를 제안하며, 이는 정상 및 비정상 샘플의 정보를 활용하여 더 나은 성능을 발휘합니다. APC는 기존의 모델보다 고급 기능 추출기를 통해 비정상 샘플을 사용하여 개선됩니다.

- **Technical Details**: APC는 정상 및 비정상 샘플로 훈련된 기능 추출기를 기반으로 하고, 이를 통해 이미지에서 비정상 패턴이나 변형을 탐지합니다. 세 가지 보조 작업(분류, 세분화 및 재구성)을 도입하여 학습 중 비정상 샘플의 imbalanced effect를 완화하며, MVTec 데이터셋에서 성능을 평가합니다.

- **Performance Highlights**: APC는 후속 수동 검사와 관련하여 이상 탐지에서 가장 진보한 시스템을 능가하는 뛰어난 성능을 보여주었으며, 특히 비정상 샘플을 활용함으로써 기존 정상 샘플만으로 훈련된 시스템에 비해 많은 잠재력을 입증했습니다.



### Enhancing License Plate Super-Resolution: A Layout-Aware and Character-Driven Approach (https://arxiv.org/abs/2408.15103)
Comments:
          Accepted for presentation at the Conference on Graphics, Patterns and Images (SIBGRAPI) 2024

- **What's New**: 이번 연구는 License Plate Recognition (LPR) 시스템의 저해상도 및 저품질 이미지에서의 성능 향상을 위한 새로운 손실 함수인 Layout and Character Oriented Focal Loss (LCOFL)를 제안합니다. 이 방법은 기존의 연구가 고해상도 이미지에 집중한 점을 보완하며, 실제 교통 감시 환경에서도 효과적으로 작동할 수 있도록 설계되었습니다.

- **Technical Details**: LCOFL은 해상도, 텍스처, 구조적 세부사항을 고려하며, LPR 작업 자체의 성능 또한 반영합니다. 또한, Deformable Convolutions와 Attention Module의 공유 가중치를 활용하여 문자 특징 학습을 강화합니다. GAN 기반 학습 방법을 적용하여 Optical Character Recognition (OCR) 모델의 예측 결과를 이용해 슈퍼 해상도 과정을 안내합니다.

- **Performance Highlights**: 실험 결과, LCOFL을 사용한 모델은 문자 재구성 품질에서 두 가지 최신 방법을 모두 초과하는 성능 향상을 보여주었습니다. 제안된 방법은 저해상도 이미지에서 문자의 혼동이나 레이아웃 불일치 오류를 줄이는 데 매우 효과적입니다.



### MTMamba++: Enhancing Multi-Task Dense Scene Understanding via Mamba-Based Decoders (https://arxiv.org/abs/2408.15101)
Comments:
          arXiv admin note: text overlap with arXiv:2407.02228

- **What's New**: 본 논문에서는 MTMamba++라는 새로운 multi-task dense scene understanding 아키텍처를 제안합니다. 이 아키텍처는 Mamba 기반 디코더를 특징으로 하며, RMS(기본 행동 내역)에 기반한 self-task Mamba SCM과 cross-task Mamba CTM 블록 두 가지 핵심 블록을 포함합니다.

- **Technical Details**: MTMamba++의 디코더는 ECR(expand, concatenate, reduce), STM(self-task Mamba), CTM(cross-task Mamba) 블록으로 구성됩니다. ECR 블록은 태스크 특징을 업스케일링하고 인코더의 고수준 특징과 융합하는 역할을 합니다. STM 블록은 SSM(state space models) 메커니즘을 활용하여 각 태스크의 글로벌 컨텍스트 정보를 효과적으로 캡처하고, CTM 블록은 다양한 태스크 간의 지식 교환을 통해 태스크의 특징을 강화합니다. CTM 블록에는 F-CTM(feature level)와 S-CTM(semantic level)이라는 두 가지 변형이 포함되어 있어, 동적 크로스-태스크 표현과 태스크 간의 관계를 모델링합니다.

- **Performance Highlights**: MTMamba++는 NYUDv2, PASCAL-Context, Cityscapes 데이터셋에서 CNN 기반 및 Transformer 기반 방법들보다 월등한 성능을 보였습니다. 정량적 결과는 MTMamba++의 다중 태스크 밀집 예측 성능이 이전 방법들보다 뛰어난 것을 보여주며, 정성적 연구에서는 보다 우수한 시각적 결과와 세밀한 정확성을 생성하는 것으로 확인되었습니다.



### CLIP-AGIQA: Boosting the Performance of AI-Generated Image Quality Assessment with CLIP (https://arxiv.org/abs/2408.15098)
Comments:
          accepted by ICPR2024

- **What's New**: AI 생성 이미지(AIGI)의 품질 평가에 대한 새로운 모델 CLIP-AGIQA를 제안했습니다. CLIP 기반의 회귀 모델로, 다양한 품질 수준을 반영한 학습 가능한 프롬프트를 도입하여 평가 성능을 향상시킵니다.

- **Technical Details**: CLIP-AGIQA는 CLIP(Contrastive Language-Image Pre-training)에 기반하여 이미지 품질 분석을 수행합니다. 이 모델은 다중 카테고리 학습 가능한 프롬프트를 설계하여 텍스트 지식을 최대한 활용하고, CLIP 특징을 품질 점수로 매핑하는 회귀 네트워크를 구현합니다.

- **Performance Highlights**: 클립-AGIQA는 AGIQA-3K 및 AIGCIQA2023과 같은 여러 분류의 평가 기준에서 기존 IQA 모델보다 우수한 성능을 보였습니다. 특히, 생성된 이미지의 품질 평가에 있어 탁월한 결과를 달성했습니다.



### MMASD+: A Novel Dataset for Privacy-Preserving Behavior Analysis of Children with Autism Spectrum Disorder (https://arxiv.org/abs/2408.15077)
- **What's New**: 이번 연구에서는 자폐 스펙트럼 장애(ASD)에 대한 새로운 치료적 접근으로, MMASD+라는 모델을 소개합니다. 이 모델은 다양한 데이터 모달리티(data modalities)를 활용하여 치료사와 아동을 구분하는 데 있어 중요한 장벽을 해결합니다.

- **Technical Details**: MMASD+는 3D-Skeleton, 3D Body Mesh, Optical Flow 데이터를 포함한 다양한 데이터 모달리티를 통합합니다. Yolov8 및 Deep SORT 알고리즘의 기능을 결합하여 데이터 간 비교 문제를 극복합니다. 또한, 11개의 행동 유형과 ASD 존재를 예측하기 위해 멀티모달 트랜스포머(Multimodal Transformer) 프레임워크를 제안합니다.

- **Performance Highlights**: 이 프레임워크는 행동 유형 예측에서 95.03%의 정확도, ASD 존재 예측에서 96.42%의 정확도를 달성하였으며, 단일 데이터 모달리티로 훈련된 모델과 비교하여 10% 이상의 성능 향상을 보여주었습니다.



### Geometric Artifact Correction for Symmetric Multi-Linear Trajectory CT: Theory, Method, and Generalization (https://arxiv.org/abs/2408.15069)
Comments:
          15 pages, 10 figures

- **What's New**: 본 논문은 Symmetric Multi-Linear trajectory Computed Tomography (SMLCT)에서 발생하는 기하학적 아티팩트(geometric artifacts)를 효율적으로 처리하는 새로운 방법을 제안합니다. 기존의 비효율적인 보정(calibration) 방법을 개선한 점이 두드러집니다.

- **Technical Details**: 연구에서는 민감한 기하학적 매개변수(sensitive geometric parameters)와 아티팩트 특성 사이의 관계를 정리하고, 이러한 매개변수들과 관련된 수학적 관계를 구성합니다. 또한, GCC-PHAT (Generalized Cross-Correlation with Phase Transform) 알고리즘을 이미지 등록(image registration) 작업에 혁신적으로 적용함으로써 고효율적인 강체 변환(rigid translation) 등록 방법을 설계합니다.

- **Performance Highlights**: 모의 실험(simulation)과 물리적 실험을 통해 제안된 방법의 뛰어난 성능이 검증되었으며, 일반적인 회전 CT(rotated CT) 및 SMLCT 변형에 대한 우수한 일반화(generalization) 결과를 보여줍니다.



### Adapting Segment Anything Model to Multi-modal Salient Object Detection with Semantic Feature Fusion Guidanc (https://arxiv.org/abs/2408.15063)
Comments:
          10 pages, 9 figures

- **What's New**: 본 논문에서는 pre-trained Segment Anything Model(SAM)을 활용하여 multi-modal SOD(Salient Object Detection) 문제를 해결하기 위한 새로운 프레임워크인 Sammese를 제안합니다. 이 방법은 SAM의 강력한 feature representation과 zero-shot generalization 능력을 활용하여 멀티모달 데이터를 효과적으로 다룰 수 있게 합니다.

- **Technical Details**: Sammese는 SAM에 semantic feature fusion guidance를 결합하여 multi-modal saliency-specific 지식을 통합합니다. 이 프레임워크는 multi-modal complementary fusion 모듈을 사용하여 시각과 열 또는 깊이 이미지 쌍에서 강력한 multi-modal semantic feature를 추출합니다. SAM의 image encoder에 multi-modal adapter를 추가하고, mask decoder에서 semantic-geometric prompt generation 전략을 통해 saliency 관련 필드의 다양한 특징을 효과적으로 예측할 수 있도록 합니다.

- **Performance Highlights**: Sammese는 RGB-D 및 RGB-T SOD 벤치마크에서 실험을 수행한 결과, 우수한 성능을 보였으며, 기초 모델을 활용한 multi-modal 데이터 처리 및 멀티모달으로의 salient detection 작업 수행에 대한 가능성을 시사합니다.



### DocLayLLM: An Efficient and Effective Multi-modal Extension of Large Language Models for Text-rich Document Understanding (https://arxiv.org/abs/2408.15045)
- **What's New**: 본 논문에서는 텍스트가 풍부한 문서 이해(TDU)를 위해 설계된 효율적이고 효과적인 멀티모달 확장 모델인 DocLayLLM을 소개합니다. 기존의 OCR 의존 방식과는 다르게, LLM의 강력한 텍스트 이해 능력과 문서 레이아웃 인식 능력을 결합하여 성능을 극대화합니다.

- **Technical Details**: DocLayLLM은 이미지 패치 토큰과 2D 위치 토큰을 문서의 텍스트 콘텐츠와 통합하여 LLM을 통한 자연어 표현으로 처리합니다. 또한, 체인 오브 스로트(Chain-of-Thought, CoT)를 모든 훈련 단계에 통합하여 모델의 추론 능력을 강화시켰습니다. CoT Pre-training과 CoT Annealing 전략을 도입하여 데이터 활용 효율성과 전반적인 성능을 향상시킵니다.

- **Performance Highlights**: DocLayLLM은 기존의 OCR 의존 방법을 초월하며, 특정 작업에 대한 미세 조정 없이도 우수한 성능을 발휘합니다. 3M의 프리 트레이닝 데이터와 300K의 감독적 미세 조정 데이터만으로 전체 훈련 프로세스를 36 A100 일 이내에 마치며, 가장 뛰어난 OCR-free 방식과 비교하여 성능에서 우위를 점하고 있습니다.



### Interactive Occlusion Boundary Estimation through Exploitation of Synthetic Data (https://arxiv.org/abs/2408.15038)
- **What's New**: 본 연구에서는 Occlusion Boundary (OB)의 상호작용 추정 방법 DNMMSI를 제안하며, 이를 통해 오토메틱 방법들보다 현저한 성능 향상을 이루어냈습니다. 또한 이 연구는 3D 장면에서 OB를 기하학적으로 정의하고 이를 바탕으로 2D 이미지와 OB의 진실값을 자동 생성하는 Mesh2OB 도구를 개발했습니다.

- **Technical Details**: 진행된 연구는 세 가지 주요 기여를 포함합니다: 첫째, DNMMSI라는 딥 러닝 기반의 스크리블 방법을 통해 인터랙티브 OB 추정의 성능을 향상시켰습니다. 둘째, Mesh2OB 도구를 이용하여 3D-FUTURE 데이터세트에서 19,186개의 합성 샘플로 구성된 OB-FUTURE 벤치마크를 구축했습니다. 셋째, OB-LabName 벤치마크를 통해 120개의 고해상도 이미지와 이들의 진실 OB를 제공합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법들은 데이터 도메인 적응 기법을 사용하지 않고도 우수한 성능을 달성하였으며, 특히 OB 품질은 기존 방법보다 뛰어난 것으로 평가되었습니다.



### Mamba2MIL: State Space Duality Based Multiple Instance Learning for Computational Pathology (https://arxiv.org/abs/2408.15032)
- **What's New**: 본 논문에서는 기존의 MIL 접근 방식의 한계를 극복하기 위해 Mamba2MIL이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 주어진 WSI를 긴 시퀀스로 모델링할 수 있는 state space duality model (SSD)을 활용하며, 가중치 기반의 특성 선택(feature selection)을 통하여 더 다양한 특성을 융합할 수 있도록 지원합니다.

- **Technical Details**: Mamba2MIL은 WSI 패치의 긴 시퀀스를 효과적으로 모델링하기 위해 SSD을 사용하며, 다양한 크기의 WSI에 적합한 시퀀스 변환 방법을 도입합니다. 이 방법은 시퀀스 의존적 특성을 강화하고, 동시에 로컬 시퀀스 정보를 유지하는 데 중점을 두어 시퀀스 정보의 전체 활용도를 향상시킵니다.

- **Performance Highlights**: Mamba2MIL은 여러 데이터셋에서 실험 결과, 모든 성능 지표에서 개선을 보였습니다. 특히, NSCLC 데이터셋에서 AUC 0.9533, 정확도 0.8794를 기록하였고, BRACS 데이터셋에서는 AUC 0.7986, 정확도 0.4981을 달성했습니다.



### Sequence-aware Pre-training for Echocardiography Probe Guidanc (https://arxiv.org/abs/2408.15026)
Comments:
          Tech Report

- **What's New**: 이번 연구에서는 심장 초음파 프로브의 조정을 위한 새로운 방법인 sequence-aware self-supervised pre-training 방법을 제안합니다. 이 방법은 개별 환자의 심장 구조에 대한 개인화된 정보를 기반으로 합니다.

- **Technical Details**: 연구팀은 심장 초음파 영상과 조정 행동을 함께 고려하여 개인의 심장 구조를 학습하는 방식을 채택했습니다. 이를 위해 masked-out된 이미지와 조정 행동을 예측하는 방식으로 개인화된 2D 및 3D 심장 구조 특징을 학습합니다. 또한, 기존 스캔 데이터에 기반하여 개인의 심장 구조 정보를 모델링하는 시퀀스 모델링 접근 방식을 도입했습니다.

- **Performance Highlights**: 대규모 데이터셋에서 실험한 결과, 제안된 방법은 기존 최첨단 기술에 비해 내비게이션 오차를 15.90%에서 36.87%까지 줄이고, 회전 오차는 11.13%에서 20.77%까지 감소시켰습니다.



### Hierarchical Graph Interaction Transformer with Dynamic Token Clustering for Camouflaged Object Detection (https://arxiv.org/abs/2408.15020)
Comments:
          Submitted to IEEE Transactions on Image Processing

- **What's New**: 본 연구에서는 camouflaged object detection (COD) 분야에 기여하기 위해 새로운 모델인 HGINet를 제안합니다. HGINet는 계층적 그래프 상호작용을 통해 imperceptible (식별 불가능한) 객체를 발견할 수 있는 능력을 가지고 있습니다.

- **Technical Details**: HGINet는 Region-Aware Token Focusing Attention (RTFA) 및 Hierarchical Graph Interaction Transformer (HGIT) 모듈을 활용하여 지역적에서 두드러진 토큰을 발굴하고 시각적 의미를 포착하기 위해 양방향 정렬 통신을 구축합니다. 또한, Confidence Aggregated Feature Fusion (CAFF) 디코더 네트워크를 통해 모호한 지역에서 세부 사항을 정제할 수 있도록 설계되었습니다.

- **Performance Highlights**: COD10K, CAMO, NC4K 및 CHAMELEON과 같은 공개 데이터셋에 대한 실험 결과, HGINet는 기존의 최신 방법들에 비해 우수한 성능을 보여줍니다. HGINet는 복잡한 배경에서도 높은 품질의 camouflaged object segmentation을 성공적으로 수행할 수 있습니다.



### Pre-training Everywhere: Parameter-Efficient Fine-Tuning for Medical Image Analysis via Target Parameter Pre-training (https://arxiv.org/abs/2408.15011)
- **What's New**: 새로운 프레임워크인 Target Parameter Pre-training (TPP)을 제안합니다. 이는 파라미터 효율적 미세 조정(PEFT)의 기존 방법에 간단히 추가하여 성능을 개선하는 것을 목표로 합니다.

- **Technical Details**: TPP는 PEFT 방법을 활용하기 전에 새로운 파라미터, 즉 target parameters를 미리 학습하는 추가적인 단계를 포함합니다. 이 과정에서, 이미 학습된 backbone 모델의 파라미터를 고정하고 target parameters만을 학습하여 특정한 표현(representation)을 학습하도록 유도합니다.

- **Performance Highlights**: TPP는 5개의 공개 데이터셋을 통한 평가에서 기존 PEFT 방법들과의 통합이 용이하며, 일관되게 성능을 개선하는 결과를 보였습니다. 특히, 의료 이미지 분류(classification)와 분할(segmentation) 작업에서 효과가 두드러집니다.



### FastTextSpotter: A High-Efficiency Transformer for Multilingual Scene Text Spotting (https://arxiv.org/abs/2408.14998)
Comments:
          Accepted in ICPR 2024

- **What's New**: 이번 논문에서는 FastTextSpotter라는 새로운 텍스트 탐지 프레임워크를 소개합니다. 이 프레임워크는 Swin Transformer를 시각적 백본으로 통합하고, 새로운 SAC2(자기 주의 유닛)를 도입하여 처리 속도를 개선하면서 정확도를 유지하도록 설계되었습니다.

- **Technical Details**: FastTextSpotter는 Transformer 인코더-디코더 구조와 효율적인 자기 주의 메커니즘을 결합하여 멀티랭귀지(다국어) 장면 텍스트 인식의 성능을 높입니다. 특히 SAC2 모듈은 훈련 속도를 높이고 정확도를 유지하는 데 중점을 두고 설계되었습니다.

- **Performance Highlights**: FastTextSpotter는 ICDAR2015, CTW1500, TotalText와 같은 여러 데이터셋에서 기존 SOTA(State-of-the-Art) 모델들과 비교했을 때, 모든 데이터셋에서 우수한 정확도를 달성하며 프레임 속도(FPS) 측면에서도 효율성을 높였습니다.



### MegActor-$\Sigma$: Unlocking Flexible Mixed-Modal Control in Portrait Animation with Diffusion Transformer (https://arxiv.org/abs/2408.14975)
- **What's New**: MegActor-Σ는 오디오와 비주얼 모달리티를 효과적으로 통합한 혼합 모달 조건 확산 변환기(DiT)를 소개하여 초상화 애니메이션의 새로운 가능성을 제시합니다. 이 모델은 기존의 단일 모달리티 접근법의 한계를 극복하고 두 가지 모달리티를 혼합하여 유연하게 제어할 수 있는 능력을 제공합니다.

- **Technical Details**: MegActor-Σ는 강화된 모델 구조의 DiT를 기반으로 하고, 'Modality Decoupling Control' 훈련 전략을 적용하여 비주얼과 오디오 모달리티 간의 제어 강도를 균형 있게 조절합니다. 또한, 'Amplitude Adjustment' 추론 전략을 통해 각 모달리티의 동작 진폭을 자유롭게 조절할 수 있습니다.

- **Performance Highlights**: 광범위한 실험을 통해 MegActor-Σ는 생동감 있는 초상화 애니메이션을 생성하는 데 있어 기존 방법들보다 우수한 성능을 보임을 입증하였습니다. 이를 통해 향후 AI 기반 인간 상호작용 및 디지털 아바타와 같은 다양한 응용 분야에서의 활용 가능성이 높아졌습니다.



### Deep Learning-based Average Shear Wave Velocity Prediction using Accelerometer Records (https://arxiv.org/abs/2408.14962)
Comments:
          12 pages, 14 figures, Accepted by 18th World Conference on Earthquake Engineering WCEE2024

- **What's New**: 이번 연구에서는 지진 공학에서 중요한 Vs30(상위 30미터의 시트로파 동속도) 예측을 위한 딥러닝 기반 접근법을 제시합니다.

- **Technical Details**: 이 연구는 터키에 위치한 700개 이상의 강력한 움직임 기록 스테이션에서 수집한 3채널 지진 기록을 사용하여 Vs30를 예측합니다. Convolutional Neural Networks (CNNs)와 확장된(dilated) 및 인과(causal) 합성곱 레이어를 이용하여 심층 특성을 추출합니다.

- **Performance Highlights**: 제안된 방법은 수작업으로 생성된 특징을 사용하는 기계 학습 모델과 비교했으며, 딥 합성곱 인코더 기반 Vs30 예측 모델이 훨씬 더 우수한 성능을 나타냈습니다.



### CVPT: Cross-Attention help Visual Prompt Tuning adapt visual task (https://arxiv.org/abs/2408.14961)
- **What's New**: 본 논문에서는 Cross Visual Prompt Tuning (CVPT) 방법을 제안하여 기존의 Visual Prompt Tuning (VPT) 방법의 성능과 효율성을 향상시킵니다. CVPT는 프롬프트 토큰과 임베딩 토큰 간의 크로스 어텐션을 계산하여 세분화된 시각적 작업에 맞출 수 있도록 모델을 조정합니다.

- **Technical Details**: CVPT는 프롬프트 토큰과 임베딩 토큰 간의 크로스 어텐션을 활용하여 세멘틱 관계를 계산하고, learnable parameter의 수를 줄여 효율성을 높이는 weight-sharing 메커니즘을 도입합니다. 실험 결과, CVPT는 25개의 데이터셋에서 검증되었으며 VPT에 비해 평균적으로 4% 더 높은 정확도를 보였습니다.

- **Performance Highlights**: CVPT는 VTAB-1K 벤치마크에서 VPT에 비해 평균 4%의 정확도 향상을 이루었으며, FGVC 및 ADE20K 데이터셋에서도 SOTA 방법들과 경쟁할 수 있는 성능을 보여주었습니다. 특히 적은 수의 프롬프트 토큰을 사용할 때 다른 선진 PEFT 방법들과 비슷한 성능을 나타내었습니다.



### Applying ViT in Generalized Few-shot Semantic Segmentation (https://arxiv.org/abs/2408.14957)
Comments:
          7 pages, 4 figures

- **What's New**: 이 논문은 ViT 기반 모델이 일반화된 몇 샷 의미론 분할(Generalized Few-shot Semantic Segmentation, GFSS) 프레임워크에서 보여주는 가능성을 탐구합니다. 다양한 백본(backbone) 모델 조합에 대한 실험이 진행되었으며, DINOv2를 사용한 구조는 PASCAL-$5^i$ 데이터셋에서 뛰어난 성능을 발휘하였습니다.

- **Technical Details**: GFSS는 쿼리 이미지와 지원 이미지의 클래스가 동일할 필요가 없으며, 베이스 클래스와 새로운 클래스 모두에서 평가됩니다. 다양한 ViT 기반 모델과 ResNet 모델을 비교하여 DINO 및 DINO v2가 ResNet보다 우수한 성능을 보임을 확인했습니다. 이 논문은 여러 가지 디코더를 사용하여 ViT 모델의 성능을 비교하고, Mask Transformer는 Linear Classifier보다 더 많은 오버피팅 문제를 나타났습니다.

- **Performance Highlights**: DINOv2와 Linear Classifier 조합은 원샷 시나리오에서 ResNet 구조 대비 116% 성능 향상을 보여주었습니다. GFSS 작업에서 대규모 사전 훈련된 ViT 모델의 가능성을 입증하고 있으며, 추가 테스트 벤치마크에서의 성능 개선도 기대됩니다.



### NeuralOOD: Improving Out-of-Distribution Generalization Performance with Brain-machine Fusion Learning Framework (https://arxiv.org/abs/2408.14950)
- **What's New**: 본 논문에서는 Deep Neural Networks (DNNs)의 OOD(Out-of-Distribution) 일반화 성능을 개선하기 위한 새로운 Brain-machine Fusion Learning (BMFL) 프레임워크를 제안합니다.

- **Technical Details**: BMFL 프레임워크는 cross-attention 메커니즘을 채택하여 CV(Computer Vision) 모델의 시각적 지식과 인간 두뇌의 사전 인지 지식을 융합합니다. 또한, 사전 훈련된 시각 신경 인코딩 모델을 사용하여 fMRI(functional Magnetic Resonance Imaging) 데이터를 예측하며, Pearson 상관 계수 최대화 정규화 방법을 교육 과정에 도입합니다.

- **Performance Highlights**: 우리 모델은 ImageNet-1k 검증 데이터셋과 6개의 OOD 데이터셋에서 DINOv2 및 기준 모델보다 뛰어난 성능을 보여 다양한 시나리오에서 우수성을 입증하였습니다.



### BOX3D: Lightweight Camera-LiDAR Fusion for 3D Object Detection and Localization (https://arxiv.org/abs/2408.14941)
Comments:
          Presented in MED 2024

- **What's New**: 이번 논문에서는 BOX3D라는 새로운 multi-modal(다중 모달) 및 lightweight(경량) 스킴을 제안하여 RGB 카메라와 3D LiDAR의 정보를 융합함으로써 관심 대상 객체를 로컬라이징하는 방법을 소개합니다. BOX3D는 세 개의 계층 구조(three-layered architecture)로 구성되어 있어, 연속 센서 데이터를 통한 로컬 감지(local perception)에서 시작해 아웃라이어(outliers)와 각 객체 관측의 일반적인 일관성(global perception refinement)을 고려합니다.

- **Technical Details**: BOX3D는 첫 번째 계층에서 카메라와 LiDAR 데이터를 저수준에서 융합하여 초기 3D 바운딩 박스를 추출합니다. 두 번째 계층에서는 LiDAR의 스캔 바운딩 박스를 전 세계 좌표 프레임으로 변환하고, 여러 관점에서 관찰된 객체의 유일성을 유지하기 위해 공간적 페어링 및 머징 메커니즘을 적용합니다. 마지막으로, 세 번째 계층은 글로벌 맵에서 결과의 일관성을 반복적으로 감독하며, 객체에 속하는 모든 점들을 식별하기 위해 포인트-투-복셀(point-to-voxel) 비교를 사용합니다.

- **Performance Highlights**: 제안된 BOX3D 프레임워크는 대규모 도시 환경의 공개된 최첨단 데이터셋에서 여러 실험을 통해 신속하고 정확한 객체 감지를 보여주었습니다. 실험 결과는 OBJECT DETECTION(객체 감지) 및 LOCALIZATION(위치 파악) 분야에서 BOX3D의 유용성을 입증했습니다.



### Cross-Modal Temporal Alignment for Event-guided Video Deblurring (https://arxiv.org/abs/2408.14930)
Comments:
          Accepted in ECCV2024

- **What's New**: 이 연구에서는 동적 움직임으로 인해 발생하는 비디오 모션 블러를 효과적으로 제거하기 위해 마이크로초 단위의 시간 해상도를 가진 이벤트 카메라를 활용하는 새로운 비디오 디블러링(video deblurring) 방법론을 제안합니다. 특히, 이 방법은 블러 처리된 단일 프레임의 데이터 부족을 보완하기 위해 인접한 비디오 프레임에서 정보를 효과적으로 수집합니다.

- **Technical Details**: 제안된 접근 방식은 두 가지 모듈로 구성됩니다: 1) Intra-frame feature enhancement는 단일 블러 프레임의 노출 시간 내에서 교차 모드(feature) 기능을 반복적으로 향상시킵니다. 2) Inter-frame temporal feature alignment는 타겟 프레임에 대해 가치 있는 장거리 시간 정보를 수집하여 이벤트의 이점을 활용하여 선명한 특징을 집계합니다. 이러한 방식은 복잡한 계산을 피하면서 이벤트의 시간 정보를 효과적으로 활용합니다.

- **Performance Highlights**: 광범위한 실험을 통해, 제안된 방법이 최신 frame-based 및 event-based 모션 디블러링 방법보다 우수한 성능을 보여주며, 실제 환경에서 적용 가능한 새로운 데이터를 포함한 EVRB 데이터셋도 소개합니다.



### Towards Real-world Event-guided Low-light Video Enhancement and Deblurring (https://arxiv.org/abs/2408.14916)
Comments:
          Accepted in ECCV2024

- **What's New**: 이번 연구는 저조도 환경에서 이벤트 카메라를 사용하여 저조도 개선과 동작 흐림 제거를 동시에 해결하는 첫 번째 시도를 제안합니다. 연구자들은 새로운 혼합 카메라 시스템을 구축하고, 이를 통해 실제 환경에서 수집된 RELED 데이터셋을 제공합니다.

- **Technical Details**: 이 프레임워크는 두 가지 모듈을 포함하고 있습니다: 이벤트 기반 변형 시간 특징 정렬 모듈(ED-TFA)과 스펙트럼 필터링 기반 교차 모달 특징 강화 모듈(SFCM-FE). ED-TFA 모듈은 이벤트 정보를 활용하여 여러 시각적 스케일에서 시간적 정보를 효율적으로 이용합니다. SFCM-FE 모듈은 저조도 상황에서 노이즈를 줄이면서 주요 구조 정보를 강화하는 것을 목표로 합니다.

- **Performance Highlights**: 제안된 방법은 기존 접근 방식에 비해 저조도 개선 및 동작 흐림 제거 문제를 효과적으로 다루어 최신 기술의 성능을 초월하는 결과를 보여주었습니다.



### MeshUp: Multi-Target Mesh Deformation via Blended Score Distillation (https://arxiv.org/abs/2408.14899)
- **What's New**: 본 논문에서는 3D 메쉬를 여러 타겟 개념으로 변형시키는 새로운 기법인 MeshUp을 제안했습니다. 사용자는 텍스트 쿼리나 영감이 되는 이미지를 입력하여 각 개념이 표현되는 지역을 직관적으로 제어할 수 있습니다.

- **Technical Details**: MeshUp은 각 개념에 대한 활성화 맵을 혼합하는 Blended Score Distillation (BSD)이라는 새로운 기법을 사용하여 구현되며, 이를 통해 변형 과정에서의 지역적 표현을 제어합니다. 사용자는 변형할 메쉬의 특정 정점을 선택하여 각 개념을 세부적으로 조정할 수 있습니다.

- **Performance Highlights**: 실험 결과, BSD는 여러 목적에 맞춰 다양한 메쉬를 효과적으로 변형할 수 있음을 보여주며, 사용자가 쉽고 창의적으로 3D 콘텐츠를 생성할 수 있는 도구로서의 가능성을 지니고 있습니다.



### Adversarial Manhole: Challenging Monocular Depth Estimation and Semantic Segmentation Models with Patch Attack (https://arxiv.org/abs/2408.14879)
Comments:
          Accepted for WISA 2024. Code and dataset: this https URL

- **What's New**: 이 논문은 유명한 'Manhole Cover'와 유사한 실用적인 패치를 사용하여 단안 깊이 추정(Monocular Depth Estimation, MDE)과 의미론적 분할(Semantic Segmentation, SS) 모델을 속이는 새로운 적대적 공격(adversarial attack)을 제시합니다. 이는 자율주행 시스템의 안전성을 저해할 수 있는 잠재적 위협을 식별하고 있습니다.

- **Technical Details**: 이 연구에서는 Depth Planar Mapping 기법을 사용하여 도로 표면에 패치를 정확하게 위치시키는 방법을 설명합니다. 논문은 MDE와 SS 모델을 동시에 공격하는 최초의 접근 방식을 제안하며, 패치를 단순한 원근 변환을 넘어 활용하는 방식을 다룹니다. 이를 통해 자율주행 시스템의 취약점을 이용할 수 있는 지속 가능한 방법을 제공합니다.

- **Performance Highlights**: 실험 결과, 적대적 패치가 MDE에 대해 43%의 상대 오류를 유발하고 SS에 대해 96%의 공격 성공률을 달성했습니다. MDE에서 발생한 오류 지역은 패치 크기의 두 배 이상이며, SS에서는 패치 크기와 대등한 것으로 나타났습니다. 이 연구는 패치의 효과성을 실제 시뮬레이션 환경에서도 증명하였습니다.



### ZeroMamba: Exploring Visual State Space Model for Zero-Shot Learning (https://arxiv.org/abs/2408.14868)
- **What's New**: 이번 논문에서는 Zero-shot learning (ZSL)을 위한 새로운 프레임워크 ZeroMamba를 제안합니다. ZeroMamba는 기존의 CNN 및 ViT의 한계를 극복하기 위해 Vision Mamba를 기반으로 하여, 시각적 및 의미적 상호작용을 더 효과적으로 처리합니다.

- **Technical Details**: ZeroMamba는 세 가지 주요 구성 요소인 Semantic-aware Local Projection (SLP), Global Representation Learning (GRL), Semantic Fusion (SeF)을 포함합니다. SLP는 의미적 임베딩을 통합하여 시각적 특징을 지역적 의미 관련 표현으로 매핑하고, GRL은 모델이 글로벌 의미 표현을 학습하도록 유도하며, SeF는 두 가지 의미 표현을 결합하여 의미적 특징의 구별력을 향상시킵니다. 이러한 디자인은 Vision Mamba에 통합되어 최종적으로 ZSL을 위한 엔드-투-엔드 프레임워크를 형성합니다.

- **Performance Highlights**: ZeroMamba는 네 가지 주류 ZSL 벤치마크에서 우수한 성능을 보이며, 기존 최첨단 방법들(CNN 기반 및 ViT 기반)을 뛰어넘는 성과를 달성하였습니다. 특히, 전통적인 ZSL (CZSL) 및 일반화된 ZSL (GZSL) 설정 모두에서 가장 높은 성능을 기록하였습니다.



### DiffSurf: A Transformer-based Diffusion Model for Generating and Reconstructing 3D Surfaces in Pos (https://arxiv.org/abs/2408.14860)
Comments:
          Accepted at ECCV2024

- **What's New**: 논문에서 DiffSurf라는 트랜스포머 기반의 디노이징 확산 모델을 제안합니다. 이 모델은 다양한 형태와 포즈에서 3D 표면을 생성하고 복원하는 데 초점을 맞추고 있습니다.

- **Technical Details**: DiffSurf는 노이즈가 있는 3D 표면 정점과 법선을 이용하여 노이즈를 예측하는 확산 트랜스포머 아키텍처를 설계했습니다. 이 모델은 포인트와 법선으로 표면을 표현하고, 이를 확산 트랜스포머에서 처리하여 다양한 포즈와 객체 유형을 처리할 수 있도록 설계되었습니다.

- **Performance Highlights**: DiffSurf는 이전의 생성 모델보다 더 높은 품질과 다양성을 가진 3D 형태를 생성할 수 있으며, 단일 이미지에서 3D 인간 메쉬 회복 작업에서 유사한 정확도를 제공하면서도 거의 실시간 속도로 동작합니다.



### Diffusion-Occ: 3D Point Cloud Completion via Occupancy Diffusion (https://arxiv.org/abs/2408.14846)
- **What's New**: 이 논문에서는 Diffusion-Occ라는 새로운 프레임워크를 제안하여 점구름(point clouds) 완료 방법을 혁신적으로 개선했습니다. 이 방법은 전통적인 포인트 기반 접근법 대신에, 조밀한 점군을 예측하는 코사인 접근 방식을 사용합니다.

- **Technical Details**: Diffusion-Occ는 두 단계로 이루어진 접근 방식을 활용합니다. 첫 번째 단계에서 Coarse Density Voxel Prediction Network (CDNet)를 통해 부분적인 점을 입력받아 조밀한 밀도 복셀을 예측하여 글로벌 피쳐를 추출합니다. 두 번째 단계에서는 Occupancy Generation Network (OccGen)을 사용하여 변형기를 기반으로 한 조건부 점유 확산(occupancy diffusion) 모델을 구현합니다.

- **Performance Highlights**: 실험 결과, Diffusion-Occ는 ShapeNet-55에서 기존 방법보다 최소 +12%에서 +9%의 성능 개선을 보였습니다. 이는 저비용으로 높은 품질의 점구름을 생성할 수 있게 하는 주요한 우수성을 입증하였습니다.



### From Bias to Balance: Detecting Facial Expression Recognition Biases in Large Multimodal Foundation Models (https://arxiv.org/abs/2408.14842)
- **What's New**: 이 연구는 대규모 다중 모드 기초 모델(Large Multimodal Foundation Models, LMFMs) 내의 얼굴 표현 인식(Facial Expression Recognition, FER) 시스템에서 인종적 편향을 다룹니다. 기존의 FER 시스템이 어두운 피부 톤을 가진 개인에 대해 더 높은 오류율을 보인다는 사실이 강조되었습니다.

- **Technical Details**: 연구팀은 GPT-4o, PaliGemma, Gemini, CLIP 등 4개의 주요 LMFMs을 벤치마킹하였으며, CLIP 임베딩을 기반으로 한 선형 분류기는 RADIATE(95.9%), Tarr(90.3%), Chicago Face(99.5%)에서의 정확도를 기록했다. 또한, 아프리카계 미국인 여성에서 분노가 백인 여성보다 경악으로 잘못 분류되는 비율이 2.1배 더 높다는 사실이 발견되었습니다.

- **Performance Highlights**: CLIP은 모든 인종 카테고리에서 94.2%에서 97.4% 사이의 정확도를 기록하며, 감정 인식에 있어 우수한 성능을 나타냈습니다. 특히, 행복과 중립 감정을 인식하는 데 있어 CLIP은 98.1%에서 100%에 달하는 정확도를 보였습니다.



### Diffusion based Semantic Outlier Generation via Nuisance Awareness for Out-of-Distribution Detection (https://arxiv.org/abs/2408.14841)
- **What's New**: 새로운 방법론인 SONA(Semantic Outlier generation via Nuisance Awareness)를 도입하여 기존의 OOD(out-of-distribution) 탐지 문제를 해결하고, ID(in-distribution) 샘플의 픽셀 공간을 직접 활용해 도전적인 아웃라이어(outlier)를 생성합니다.

- **Technical Details**: SONA는 SONA guidance를 통해 ID 샘플의 의미 영역(semantic region)과 난잡성(nuisance regions) 레벨을 조절할 수 있습니다. 이로 인해 생성된 아웃라이어는 명확한 의미적으로 불일치하는 정보를 제공하며, ID와의 난잡성 유사성을 유지합니다. 또한, OOD 탐지기가 SONA 아웃라이어를 활용하여 의미 구별을 집중적으로 학습하도록 설계되었습니다.

- **Performance Highlights**: SONA 프레임워크는 Near-OOD 데이터셋에서 88%의 인상적인 AUROC(Area Under the Receiver Operating Characteristic) 점수를 달성하였으며, 이는 기존의 베이스라인 방법들보다 약 6% 향상된 성능을 보여줍니다.



### Time-Aware Face Anti-Spoofing with Rotation Invariant Local Binary Patterns and Deep Learning (https://arxiv.org/abs/2408.14829)
- **What's New**: 이 논문은 기존의 생체인증 시스템에 대한 모의 공격(Impersonation attacks) 취약성을 분석하고, 머신러닝을 활용한 새로운 생체 인증 방법인 Color and Texture LSTM (C&T-LSTM)을 제안합니다. 이 시스템은 최소한의 대역폭을 요구하면서도 다중 프레임 분류(multi-frame classification)를 통해 높은 정확도를 보장합니다.

- **Technical Details**: C&T-LSTM 시스템은 0.25초에서 0.5초의 짧은 시간 동안 16개의 연속 프레임을 사용하여 인증을 진행하며, 새로운 롤러 이너리 패턴(Local binary patterns, LBP) 변형 기법을 활용합니다. 이 시스템은 Long Short-Term Memory (LSTM) 모델을 기반으로 하여 깊이 있는 학습 전략을 적용합니다.

- **Performance Highlights**: 제안된 C&T-LSTM은 공격에 대한 저항력이 뛰어나고, 기존 생체 인증 시스템에 비해 사용자 경험을 방해하지 않으며, 모바일 디바이스와의 호환성이 높습니다. 이 시스템은 기존 PAD 시스템들과의 비교를 통해 그 성능을 철저히 평가하였습니다.



### Alfie: Democratising RGBA Image Generation With No $$$ (https://arxiv.org/abs/2408.14826)
Comments:
          Accepted at ECCV AI for Visual Arts Workshop and Challenges

- **What's New**: 본 연구에서는 디자인 및 문서에 쉽게 통합할 수 있는 고품질의 RGBA (Red, Green, Blue, Alpha) 일러스트레이션을 자동으로 생성하는 새로운 파이프라인인 Alfie를 제안합니다. 기존의 모델을 활용하되, 추가적인 컴퓨팅 비용 없이 생성 과정에서의 유연성을 극대화하였습니다.

- **Technical Details**: Alfie는 사전 훈련된 Diffusion Transformer 모델을 수정하여 두 가지 주요 방식을 통해 이미지를 생성합니다. 첫째, 주제와 배경 각각의 라텐트를 마스킹하고 결합하여 이미지 중심에서 날카로운 크롭이 없는 주제를 생성할 수 있도록 합니다. 둘째, 생성 과정에서 계산된 교차 주의(attention) 및 자기 주의 맵을 활용하여 알파 채널 값을 추정합니다.

- **Performance Highlights**: 사용자 연구 결과, 대다수의 사용자가 기존의 이미지 생성 후 매팅 과정보다 Alfie를 통해 생성된 이미지를 선호함을 보여주었고(63%), 생성된 일러스트레이션은 Adobe Stock의 일러스트와 유사한 품질의 장면 합성 파이프라인에 적합하다는 것을 입증하였습니다.



### LapisGS: Layered Progressive 3D Gaussian Splatting for Adaptive Streaming (https://arxiv.org/abs/2408.14823)
- **What's New**: 본 논문은 XR(확장 현실) 응용 프로그램의 증가에 대응하기 위해 대역폭에 민감한 스트리밍을 위한 LapisGS라는 계층적 3D Gaussian Splatting (3DGS) 모델을 제안합니다. 이 모델은 점진적 렌더링과 적응형 스트리밍을 지원합니다.

- **Technical Details**: LapisGS는 누적 표현을 위한 계층 구조를 구성하고, 동적 불투명도 최적화를 통해 시각적 충실도를 유지하며, 점유 맵(occupancy maps)을 활용하여 Gaussian splats를 효율적으로 관리합니다. 이 방법은 기본 레이어와 하나 이상의 강화 레이어로 구성된 여러 세부 수준(levels of detail, LOD)을 제공합니다.

- **Performance Highlights**: 실험 결과, 본 접근법은 SSIM에서 50.71%, LPIPS에서 286.53% 개선을 보여주며, 모델 크기를 318.41% 줄이는 성과를 달성했습니다. 이로 인해 대역폭 최적화를 위한 3D 스트리밍 및 렌더링 응용 프로그램에 적합한 모델임을 입증합니다.



### Build-A-Scene: Interactive 3D Layout Control for Diffusion-Based Image Generation (https://arxiv.org/abs/2408.14819)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문에서는 상호작용 가능한 3D 레이아웃 제어를 통해 텍스트에서 이미지로(Text-to-Image, T2I) 생성하는 확산(diffusion) 기반 접근 방식을 제안합니다. 기존 T2I 모델의 단점을 보완하기 위해 2D가 아닌 3D 레이아웃을 활용하여 사용자에게 보다 직관적인 제어를 제공합니다.

- **Technical Details**: 제안된 방법은 기존의 2D 박스 대신 3D 박스를 사용하여 레이아웃을 구축하며, Dynamic Self-Attention (DSA) 모듈과 일관된 3D 객체 변환 전략을 통해 이미지 생성 프로세스를 다단계 생성 작업으로 재구성합니다. 사용자는 각 단계에서 객체를 추가, 변경, 이동할 수 있으며, 이전 단계의 객체는 유지됩니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식이 복잡한 장면을 3D 레이아웃을 기반으로 생성하며, 기존 깊이 조건의 T2I 방법에 비해 객체 생성 성공률이 2배 높아졌습니다. 또한 기존 방법들과 비교했을 때 레이아웃 변경 시 객체를 더 잘 보존하는 성능을 보여줍니다.



### HPT++: Hierarchically Prompting Vision-Language Models with Multi-Granularity Knowledge Generation and Improved Structure Modeling (https://arxiv.org/abs/2408.14812)
Comments:
          19 pages, 7 figures, 7 tables. arXiv admin note: substantial text overlap with arXiv:2312.06323

- **What's New**: 이 논문에서는 LLMs(대형 언어 모델)를 활용하여 구조적 지식을 강화하여 기존의 자연 언어 설명을 개선하는 Hierarchical Prompt Tuning(HPT)이라는 새로운 접근 방식을 제안합니다.

- **Technical Details**: HPT는 구조적 지식과 기존 언어 지식을 동시에 모델링할 수 있게 하며, 세 가지 수준의 프롬프트(저수준, 고수준, 전역)를 통해 복잡한 세멘틱 관계를 효과적으로 나타냅니다. 또한, 관계에 기반한 주의(attention) 모듈을 도입하여 엔티티와 속성 간의 쌍별 관계를 포착합니다.

- **Performance Highlights**: HPT는 다양한 평가 설정에서 기존의 SOTA(State-of-the-Art) 방법들보다 우수한 성능을 보여 주며, 다면적 지식 생성을 통해 더욱 향상된 결과를 얻었습니다. HPT++는 HPT의 성능을 더욱 개선하며, 더 많은 구조적 정보와 관계 기반 모듈을 통해 일반화 문제를 해결합니다.



### Platypus: A Generalized Specialist Model for Reading Text in Various Forms (https://arxiv.org/abs/2408.14805)
Comments:
          Accepted by ECCV2024

- **What's New**: 이 논문에서는 Platypus라는 보편적인 전문 모델을 제안하고 있습니다. 이 모델은 다양한 형태의 텍스트 인식을 단일 아키텍처로 수행하며, 우수한 정확도와 높은 효율성을 달성하는 점이 특징입니다.

- **Technical Details**: Platypus는 텍스트 읽기 작업을 네 가지 주요 카테고리로 나누어 다룹니다: 자연 이미지, 문서 이미지, 잘린 텍스트, 잘린 수식. 또한, 텍스트 인식의 세분성을 두 가지 레벨로 구분합니다: 단어 레벨과 라인 레벨. 이 아키텍처는 SAM 모델의 원칙을 면밀히 적용한 인코더-디코더 프레임워크입니다.

- **Performance Highlights**: Platypus는 여러 텍스트 읽기 시나리오에서 전문 모델과 MLLM을 능가하는 성능을 보여주며, 최신 상태의 기술적 성과를 설정합니다.



### RAW-Adapter: Adapting Pre-trained Visual Model to Camera RAW Images (https://arxiv.org/abs/2408.14802)
Comments:
          ECCV 2024, code link: this https URL

- **What's New**: 이 논문에서는 RAW 데이터를 카메라 RAW 데이터에 적응시킬 목적으로 sRGB로 사전 훈련된 모델을 최적화하는 새로운 접근 방식인 RAW-Adapter를 제시합니다. 다양한 조명 조건에서 수행된 실험을 통해 우리의 알고리즘이 최첨단(SOTA)의 성능을 달성했음을 보여주었습니다.

- **Technical Details**: RAW-Adapter는 입력 수준 적응기(input-level adapters)와 모델 수준 적응기(model-level adapters)로 이루어져 있습니다. 이들은 각각 RAW 입력을 조정하고 ISP 단계와 후속 고급 네트워크 간의 연결을 구축하는 역할을 합니다. 입력 수준 적응기는 Query Adaptive Learning (QAL) 및 Implicit Neural Representation (INR) 전략을 활용하여 핵심 ISP 매개변수를 조정하고, 모델 수준 적응기는 입력 수준에서 얻은 기능을 후속 네트워크에 통합합니다.

- **Performance Highlights**: RAW-Adapter는 다양한 조명 조건에서의 검출 및 세분화 실험에서 기존의 주류 ISP 및 공동 훈련 방법과 비교하여 뛰어난 성능을 발휘했습니다. 이로 인해 RAW 이미지를 활용한 컴퓨터 비전 작업의 효율성을 높이는 데 기여할 수 있습니다.



### Revisiting Surgical Instrument Segmentation Without Human Intervention: A Graph Partitioning View (https://arxiv.org/abs/2408.14789)
- **What's New**: 본 연구는 수술 기구 분할(Surgical Instrument Segmentation, SIS)을 위한 비지도 학습 방법을 제안합니다. 기존의 점검되고 엄격한 주석이 요구되는 방식 대신, 비디오 프레임의 분할 작업을 그래프 파티셔닝 문제로 재구성하여 이미지 픽셀을 그래프 노드로 간주합니다.

- **Technical Details**: 비지도 전이 학습된 모델을 특징 추출기로 활용하여 높은 수준의 의미적 특징을 캡처하고, 라플라시안 행렬을 계산하여 그래프 파티셔닝을 수행합니다. 이를 통해 그래프의 고유 벡터를 클러스터링하거나 임계값 처리하여 수술 비디오 프레임을 도구와 조직 등의 모듈로 의미 있게 분할할 수 있습니다.

- **Performance Highlights**: 다양한 데이터셋(예: EndoVis2017, EndoVis2018, UCL 등)에서 실험을 통해, 제안된 방법은 비지도 상태-of-the-art(SOTA) 방법들보다 뛰어난 성능과 강건성을 입증하였습니다.



### MROVSeg: Breaking the Resolution Curse of Vision-Language Models in Open-Vocabulary Semantic Segmentation (https://arxiv.org/abs/2408.14776)
Comments:
          Technical report

- **What's New**: MROVSeg라는 다중 해상도 훈련 프레임워크를 제안하여 개방 어휘 의미 분할(Open-Vocabulary Semantic Segmentation) 문제를 해결합니다. 이 방법은 단일 pretrained CLIP 백본을 사용하여 고해상도 입력을 슬라이스(slicing)하고 균일한 패치(patch)로 나누어 처리합니다.

- **Technical Details**: MROVSeg는 Multi-Res Adapter를 포함하여 공간 기하를 복원하고 패치 간의 지역-전역 일치를 포착합니다. Multi-grained Masked Attention 메커니즘을 통해 객체 쿼리와 다중 해상도 CLIP 특성 간의 교차-어텐션(cross-attention)을 수행하여 다중 해상도(feature)를 집계합니다.

- **Performance Highlights**: MROVSeg는 Pascal Context-59, ADE-150, Pascal VOC 등에서 기존 최첨단 방법보다 2.4 mIoU% 향상된 성능을 보여 주며 오픈-어휘 의미 분할의 새로운 기준을 세웠습니다.



### Text-guided Foundation Model Adaptation for Long-Tailed Medical Image Classification (https://arxiv.org/abs/2408.14770)
Comments:
          Accepted by IEEE ISBI 2024

- **What's New**: 이 논문에서는 희귀 질환에 대한 레이블 부족으로 인해 의료 이미지 분류에서 발생하는 클래스 불균형 문제를 해결하기 위해 Text-guided Foundation model Adaptation for Long-Tailed medical image classification (TFA-LT)라는 새로운 접근법을 제안합니다. 이는 두 단계의 훈련 전략을 통해 기본 모델의 표현을 최적화합니다.

- **Technical Details**: 본 연구에서는 고정된 인코더를 사용하는 기초 모델에 대해 리니어 어댑터와 리니어 앙상블러를 활용하여 이미지와 텍스트의 표현을 통합하고, 임의의 데이터셋에서의 다중작업을 통해 균형 잡힌 결과를 도출합니다. 훈련 과정은 두 단계로 구성되어 있으며, 라벨링된 텍스트 대신 의미적으로 더 풍부한 프롬프트를 사용합니다.

- **Performance Highlights**: TFA-LT 방법은 현재 최고의 알고리즘보다 27.1%의 정확도 향상을 달성하면서도 GPU 메모리 사용량은 6.1%에 불과하다는 점에서 매우 효율적인 결과를 보이고 있습니다.



### CrossViewDiff: A Cross-View Diffusion Model for Satellite-to-Street View Synthesis (https://arxiv.org/abs/2408.14765)
Comments:
          21 pages, 11 figures

- **What's New**: 이 논문에서는 위성 이미지에서 도로 뷰 이미지를 생성하는 CrossViewDiff라는 새로운 교차 뷰 확산 모델을 제안합니다. 이 모델은 위성 이미지와 도로 뷰 이미지 간의 큰 차이를 극복하기 위해 위성 장면 구조 추정 및 교차 뷰 질감 매핑 모듈을 설계하였습니다.

- **Technical Details**: CrossViewDiff는 위성 이미지에서 구조 및 질감 제어를 구축하여 생성된 이미지를 더 현실감 있게 만듭니다. 또한, 향상된 교차 뷰 주의(attention) 모듈을 통한 교차 뷰 기반 제어 유도 노이즈 제거 프로세스를 설계하여 질감을 개선합니다. 다양한 데이터 소스(예: 텍스트, 지도 데이터, 건물 높이 데이터)를 탐색합니다.

- **Performance Highlights**: CrossViewDiff는 세 가지 공공 교차 뷰 데이터셋에서 최첨단 기술들을 초월하여 평균 SSIM이 9.0%, FID가 39.0%, GPT 기반 점수가 35.5% 향상된 성능을 보였습니다. 이는 더 높은 품질의 도로 뷰 파노라마를 생성함을 의미합니다.



### SynthDoc: Bilingual Documents Synthesis for Visual Document Understanding (https://arxiv.org/abs/2408.14764)
- **What's New**: SynthDoc은 고품질의 다양한 문서 데이터 세트를 생성하는 새로운 합성 문서 생성 파이프라인을 소개합니다. 이 파이프라인은 텍스트, 이미지, 표 및 차트를 포함하여 VDU(Visual Document Understanding)를 향상시키는 것을 목표로 합니다.

- **Technical Details**: SynthDoc은 공개적으로 이용 가능한 코퍼스를 활용하고, 고급 렌더링 도구를 사용하여 포괄적이고 다재다능한 데이터 세트를 생성합니다. 본 연구는 Donut 모델을 사용하여 수행된 실험으로, SynthDoc에서 생성된 데이터로 훈련된 모델은 텍스트 읽기 작업에서 우수한 성능을 달성하며 하위 작업에서도 강인성을 유지합니다.

- **Performance Highlights**: 생성된 5,000개의 이미지-텍스트 쌍으로 구성된 벤치마크 데이터 세트를 출시하여 VDU 분야의 연구 및 개발을 지원합니다. 모델은 문서 구문 분석 및 문서 VQA(Visual Question Answering)와 같은 다양한 하위 작업에서 높은 성능을 유지했습니다.



### Learning effective pruning at initialization from iterative pruning (https://arxiv.org/abs/2408.14757)
- **What's New**: 본 논문에서는 초기화 시 가지치기(Pruning at Initialization, PaI) 기법을 통해 네트워크의 학습 비용을 줄이는 방법을 제안합니다. 이는 특히 네트워크 크기가 커질수록 중요해집니다. 기존의 PaI 방법과는 다르게, 반복적인 가지치기(iterative pruning)에서 영감을 받아 성능을 개선하는 방법론을 소개합니다.

- **Technical Details**: 우리의 접근법은 초기 모델 특징을 입력으로 받고 그들의 점수를 출력한 후, 학습 전에 최저 점수 파라미터를 가지치기(prune)하는 end-to-end 신경망(	extbf{AutoS}parse)을 사용하는 것입니다. 이 방법은 다양한 모델에 대해 PaI를 수행하여 정확성과 일반화를 검증합니다.

- **Performance Highlights**: 실험 결과, 우리의 방법은 높은 희소성(high sparsity) 조건에서 기존 방법보다 우수한 성능을 보였습니다. 또한, 모델 가지치기의 기본 논리가 다양한 모델에서 일관되기 때문에, 한 모델(예: ResNet-18/CIFAR-10)의 IRP를 한 번 수행하는 것으로 VA GG-16/CIFAR-10, ResNet-18/TinyImageNet 등 다양한 모델에 일반화 할 수 있습니다.



### RSTeller: Scaling Up Visual Language Modeling in Remote Sensing with Rich Linguistic Semantics from Openly Available Data and Large Language Models (https://arxiv.org/abs/2408.14744)
Comments:
          Submitted to ISPRS

- **What's New**: 본 연구에서는 OpenStreetMap (OSM) 데이터를 활용하여 Google Earth Engine (GEE) 플랫폼에서 수집된 이미지에 대해 의미론적으로 풍부한 캡션을 대규모로 생성하는 워크플로우를 제안합니다. 이를 통해 100만 개 이상의 원거리 감지 (RS) 이미지를 포함하는 RSTeller라는 다중모달 데이터셋을 구축하였습니다.

- **Technical Details**: RSTeller 데이터셋은 다중 설명 캡션과 함께 제공되며, 기존의 비전 언어 모델 (VLM)들이 RS 장면 이해 작업에서 성능을 향상시키도록 지속적 학습 (continual pre-training) 방법을 통해 효과를 검증하였습니다. 이 방법론은 최소한의 수작업 노력으로 고품질 주석 데이터에 대한 접근성을 민주화합니다.

- **Performance Highlights**: RSTeller 데이터셋을 통해 여러 기존 VLM들의 성능이 향상됨을 입증하였습니다. 이러한 접근은 RS 이미지를 주석 처리하는 데 필요한 전문 지식과 노력을 크게 줄이며, 환경 지속 가능성 문제도 해결하게 됩니다.



### Personalized Video Summarization using Text-Based Queries and Conditional Modeling (https://arxiv.org/abs/2408.14743)
Comments:
          Ph.D. thesis, 137 pages

- **What's New**: 최근 비디오 콘텐츠가 폭발적으로 증가하면서 자동 비디오 요약의 필요성이 커지고 있습니다. 이 논문은 텍스트 기반 쿼리와 조건부 모델링을 결합하여 사용자 맞춤형 요약을 제공하는 방법론을 제안합니다.

- **Technical Details**: 이 연구에서는 다중 모달 딥 러닝(multi-modal deep learning) 접근 방식을 활용하여 시각 정보와 텍스트 쿼리를 통합하고, 문맥화된 단어 임베딩(contextualized word embeddings)과 주의 네트워크(attention networks)를 통해 텍스트 기반 쿼리 표현을 개선합니다.

- **Performance Highlights**: 제안된 방법은 기존의 최첨단 방법에 비해 더 나은 성능을 보여주며, 특히 비디오 요약의 품질을 평가할 때 정확도와 F1 점수를 사용하여 평가합니다.



### OctFusion: Octree-based Diffusion Models for 3D Shape Generation (https://arxiv.org/abs/2408.14732)
Comments:
          Technical Report

- **What's New**: Diffusion 모델을 기반으로 한 새로운 3D 생성 방법인 OctFusion을 소개합니다. 이 방법은 단일 Nvidia 4090 GPU에서 2.5초 안에 임의 해상도의 3D 형태를 생성할 수 있습니다.

- **Technical Details**: OctFusion의 핵심 구성 요소는 octree 기반의 잠재 표현(latent representation)과 이와 관련된 diffusion 모델입니다. 이 방법은 암시적 신경 표현(implicit neural representations)과 명시적 공간 octree의 장점을 결합하며, octree 기반의 변량 오토인코더(variational autoencoder)를 통해 학습됩니다.

- **Performance Highlights**: ShapeNet 및 Objaverse 데이터셋에서 OctFusion의 효율성과 효과를 검증하였으며, 모양 생성 작업에 대한 최첨단 성능을 달성했습니다. OctFusion은 약 33M의 훈련 가능한 매개변수만으로도 우수한 결과를 보여 주며, 50개의 diffusion 샘플링 단계를 설정했을 때 2.5 초 이내에 메쉬를 예측할 수 있습니다.



### GeoTransfer : Generalizable Few-Shot Multi-View Reconstruction via Transfer Learning (https://arxiv.org/abs/2408.14724)
- **What's New**: 본 연구에서는 Neural Radiance Fields(NeRF) 기반의 효율적인 sparse 3D reconstruction을 위한 새로운 접근법을 제안합니다. NeRF의 특성을 활용하여 정확한 occupancy field를 학습하는 방식으로, 기존의 sparse 입력으로부터의 3D reconstruction의 한계를 극복하고자 합니다.

- **Technical Details**: 제안된 방법은 NeRF에서 학습한 정보를 전이하여 occupancy field 표현을 정교하게 하며, 사전 학습된 NeRF 네트워크를 사용하여 세밀한 장면의 radiance 정보를 캡처합니다. 이 과정을 통해 generalizable implicit occupancy network를 신속하게 훈련할 수 있으며, 훈련 시간은 수일이 아닌 3.5시간으로 단축됩니다. 또한, volumetric rendering 가중치에 대한 새로운 손실 함수를 도입하여 정확한 occupancy field의 학습을 촉진합니다.

- **Performance Highlights**: DTU 데이터셋에서 평가한 결과, 본 방법은 sparse 입력 데이터와 occluded 지역에 대해 뛰어난 reconstruction 정확도를 달성하였으며, 최신 기술들과 비교하여 state-of-the-art의 성능을 입증하였습니다. 추가적으로 Blended MVS 데이터셋에서도 재훈련 없이 일반화된 결과를 성공적으로 보여주었습니다.



### Snap and Diagnose: An Advanced Multimodal Retrieval System for Identifying Plant Diseases in the Wild (https://arxiv.org/abs/2408.14723)
- **What's New**: 새로운 연구에서는 Snap’n Diagnose라는 멀티모달 이미지 검색 시스템을 제안하여, 농업 분야에서의 식물 질병 인식의 효율성을 높이고자 합니다. 이 시스템은 질병 이미지를 업로드하거나, 텍스트 설명을 제공함으로써 관련 이미지를 검색할 수 있습니다.

- **Technical Details**: Snap’n Diagnose는 세계 최대 규모의 PlantWild 데이터셋을 바탕으로 하며, 89개의 카테고리에 걸쳐 18,000장 이상의 실제 식물 질병 이미지를 포함하고 있습니다. CLIP 기반의 비전-언어 모델을 활용하여 질병 설명과 이미지를 동일한 잠재 공간(latent space)으로 인코딩하고, 사용자 질의를 통해 관련 이미지를 효과적으로 검색합니다.

- **Performance Highlights**: Snap’n Diagnose는 다양한 평가 지표에서 Zero-shot CLIP 모델을 능가하는 뛰어난 성능을 보이며, Top-1, Top-5, Top-10 정확도 및 평균 정밀도(mAP)에서 지속적으로 우수한 결과를 나타냈습니다. 이는 식물 질병 인식에 있어 실용적인 도구로서의 가능성을 입증합니다.



### gWaveNet: Classification of Gravity Waves from Noisy Satellite Data using Custom Kernel Integrated Deep Learning Method (https://arxiv.org/abs/2408.14674)
Comments:
          This paper has been accepted at the 27th International Conference on Pattern Recognition (ICPR) 2024

- **What's New**: 본 연구는 위성 이미지에서 중력 파(gravity wave)를 감지하기 위한 새로운 커널 설계를 제시하며, 이는 깊은 컨볼루션 신경망인 gWaveNet에 통합되었습니다. 이를 통해 소음이 있는 위성 데이터에서도 효과적으로 중력 파를 감지할 수 있습니다.

- **Technical Details**: 제안된 gWaveNet 모델은 커스텀 체커보드 커널을 사용하여 중력 파의 패턴 인식을 개선합니다. 이는 DNN(deep neural network)을 기반으로 하여, 잡음 있는 데이터에서도 중력 파의 세부 사항을 포착할 수 있도록 설계되었습니다. 우리 모델은 98% 이상의 훈련 정확도와 94% 이상의 테스트 정확도를 달성하였습니다.

- **Performance Highlights**: 우리는 제안된 방법이 기존의 접근 방식보다 우수한 성능을 보임을 입증했으며, 이는 중력 파 감지에 있어 과거의 최고 정확도로 알려져 있습니다. 우리는 연구 결과를 공개 소스로 제공하고 있습니다.



### Physically Feasible Semantic Segmentation (https://arxiv.org/abs/2408.14672)
- **What's New**: 본 논문은 Physically Feasible Semantic Segmentation (PhyFea)라는 새로운 방법을 제안하여, 기존의 데이터 기반 모델이 가지는 물리적 제약을 위반하는 경우를 탐지하고 이를 수정하는 데 중점을 둡니다. 이 방법은 기존 모델의 예측 과정에 물리적인 사전 지식을 적용하여 분류의 정확성을 향상시키는 것을 목표로 합니다.

- **Technical Details**: PhyFea는 세 가지 주요 데이터셋(ADE20K, Cityscapes, ACDC)에서 기반 모델(SegFormer-B4 및 OCRNet)에서의 예측 결과를 입력으로 받아 두 가지 물리적 이상: 불가능한 포함(infeasible inclusion)과 단절(discontinued segment)을 감지합니다. 이를 통해 이미지 형태학적 작업(morphological operations)을 사용하여 처리합니다. 주의할 점은 PhyFea는 훈련 가능한 파라미터가 없는 비훈련 가능 커스텀 레이어를 통해 물리적 제약을 적용한다는 것입니다.

- **Performance Highlights**: PhyFea를 적용함으로써 ADE20K에서 1.5%, ACDC에서 2.1%의 mIoU(median Intersection over Union) 성능 개선을 이루었습니다. 이는 기존의 최첨단 네트워크와 비교해도 확연한 성과를 보여주며, 모든 세 가지 데이터셋에서 일관된 성능 향상을 확인하였습니다.



### Comparative Analysis: Violence Recognition from Videos using Transfer Learning (https://arxiv.org/abs/2408.14659)
Comments:
          6 pages, 5 figures, The paper will be published in IEEE AICT 2024 Conference

- **What's New**: 이번 연구는 복잡한 행동 인식(Behavior Recognition)에 대한 다양한 딥러닝(deep learning) 기법들을 벤치마킹(benchmarking)하고, 데이터의 양을 증가시킴으로써 성능 향상을 테스트하고 있습니다.

- **Technical Details**: 연구에서는 500개에서 1,600개 비디오로 증가한 대규모 데이터셋을 활용하고, 이를 통해 네 가지 모델에서 평균 6%의 정확도 향상이 나타났습니다. 복잡한 사건(Complex Events), 특히 폭력 탐지(Violence Detection)에 초점을 맞추었습니다.

- **Performance Highlights**: 연구에서 제안한 접근 방식은 복잡한 비디오에서 행동 인식을 위한 기계 학습(Machine Learning) 기술들의 유용성을 보여주며, 데이터 양 증가에 따른 성능 개선 효과를 입증하였습니다.



### 3D Point Cloud Network Pruning: When Some Weights Do not Matter (https://arxiv.org/abs/2408.14601)
Comments:
          Accepted in BMVC 2024

- **What's New**: 이번 연구에서는 3D Shape Classification 네트워크에서 'winning ticket' subnetworks를 발견하여 모델을 대폭 간소화시킬 수 있는 방법을 제시하고 있습니다. 특히, 상위 p%의 최고 가중치를 보존하는 것이 정확도를 유지하는데 중요하다는 것을 강조하고 있습니다.

- **Technical Details**: 기존의 PCNN(Points Cloud Neural Networks) 모델들(PointNet, DGCNN, PointCNN)을 통해 3D 포인트 클라우드 데이터를 처리하는 과정에서 발생하는 메모리와 계산 요구사항을 줄이는 다양한 방법을 모색했습니다. 연구진은 99%의 가중치를 가지면서도 원래 성능에 가까운 정확도를 유지하는 구조를 제안했습니다. 데이터셋으로는 ModelNet40, ScanObjectNN, ShapeNetCore를 활용했습니다.

- **Performance Highlights**: 본 연구에서 제안한 방법은 1%의 중요한 가중치를 보존함으로써, 3D 모델의 정확도를 대폭 낮추지 않으면서도 파라미터 수를 줄이고 계산 요구사항을 상당히 감소시킬 수 있음을 보여주었습니다. 이는 실제 애플리케이션에서의 에너지 소비와 지연(latency)의 문제를 해결하는 데 큰 기여를 할 것으로 기대됩니다.



### PVAFN: Point-Voxel Attention Fusion Network with Multi-Pooling Enhancing for 3D Object Detection (https://arxiv.org/abs/2408.14600)
Comments:
          3D Object Detection

- **What's New**: 최근 LiDAR 기반 3D 객체 탐지에서 포인트(point)와 복셀(voxel) 표현의 통합이 더욱 흔해지고 있습니다. 하지만 이 조합은 효과적으로 의미 정보를 캡처하는 데 어려움이 있습니다. 이를 해결하기 위해 제안된 Point-Voxel Attention Fusion Network (PVAFN) 모델은 주목(attention) 메커니즘을 활용하여 다중 모달 특징 융합을 향상시키며, 다단계 및 지역별 정보를 효과적으로 통합하는 다중 풀링(multi-pooling) 전략을 사용합니다.

- **Technical Details**: PVAFN은 2단계 구조로, 첫 단계에서는 복셀 특징과 Bird's-Eye-View (BEV) 특징을 융합하여 다차원 특징을 생성합니다. 이후 자가 주목(self-attention) 층을 통해 문맥 정보를 강화합니다. 두 번째 단계에서는 RoI 클러스터링 풀링 및 RoI 피라미드 풀링을 통해 지오메트릭 상세 정보와 정밀한 모양 구조를 효율적으로 캡처하여 로컬(local) 및 글로벌(global) 특징의 통합을 향상시킵니다.

- **Performance Highlights**: KITTI 및 Waymo 데이터셋에서의 실험 결과, PVAFN은 경쟁력 있는 성능을 입증하였으며, 차량, 보행자 및 자전거 탐지에서 뛰어난 결과를 보였습니다. 제안된 모델의 코드는 추후 공개될 예정입니다.



### MMR: Evaluating Reading Ability of Large Multimodal Models (https://arxiv.org/abs/2408.14594)
- **What's New**: 이번 논문에서는 기존의 간단한 추출 기반 질문 응답을 넘어 LMM(대형 멀티모달 모델)의 복잡한 추론과 공간 이해 능력을 평가하기 위해 Multi-Modal Reading (MMR) 벤치마크를 제안합니다.

- **Technical Details**: MMR은 11개의 다양한 작업으로 구성되어 있으며, 사람의 주석을 기반으로 한 최초의 텍스트가 풍부한 이미지 벤치마크로, 고급 언어 모델을 활용하여 생성된 질문들이 포함되어 있습니다. MMR은 LAION 데이터셋을 기반으로 구축되었으며, 텍스트가 중요한 존재를 나타내는 이미지가 선택되었습니다. 이는 LMM 성능을 더욱 정교하게 평가할 수 있는 기회를 제공합니다.

- **Performance Highlights**: 여러 최신 LMM을 평가한 결과, 기존 LMM의 한계를 드러내고, 특정 모델에서 더 나은 성능을 나타내는 오픈소스 모델을 포함한 여러 사례를 발견했습니다. 텍스트 기초 작업에서 모든 모델이 낮은 성능을 보였으며, 이는 향후 LMM 개선의 중요한 영역으로 지적되었습니다.



### Global-Local Distillation Network-Based Audio-Visual Speaker Tracking with Incomplete Modalities (https://arxiv.org/abs/2408.14585)
Comments:
          Audio-Visual Speaker Tracking with Incomplete Modalities

- **What's New**: 이번 논문에서는 GLDTracker라는 새로운 오디오-비주얼 (audio-visual) 스피커 추적기를 제안합니다. 이 시스템은 불완전한 모달리티(incomplete modalities) 문제를 해결하기 위해 교사-학생 (teacher-student) 지식 증류 모형을 기반으로 구성되었습니다.

- **Technical Details**: GLDTracker는 전역 신호(global signals)와 지역 신호(local signals)를 처리하는 교사 네트워크(teacher network)와 학생 네트워크(student network)를 포함합니다. 학생 네트워크는 지역적으로 누락된 정보를 다루며 Generative Adversarial Network (GAN)를 활용하여 글로벌 특징을 재구성하는 모듈을 포함합니다.

- **Performance Highlights**: AV16.3 데이터셋에서 광범위한 실험 결과를 통해 GLDTracker가 기존의 최첨단 오디오-비주얼 추적기를 능가하며, 불완전한 모달리티 데이터셋에서도 우수한 성능을 보임을 입증했습니다.



### DIAGen: Diverse Image Augmentation with Generative Models (https://arxiv.org/abs/2408.14584)
Comments:
          Accepted for publication in GCPR 2024

- **What's New**: DIAGen은 최근 제안된 DA-Fusion을 기반으로 하여 고차원 시맨틱 다양성을 높이는 새로운 데이터 증강 방법을 제안합니다. Gaussian noise를 임베딩에 적용하여 다양성을 강화하고, 텍스트-텍스트 생성 모델을 활용하여 이미지 생성을 안내합니다.

- **Technical Details**: DIAGen은 세 가지 주요 요소를 바탕으로 구축되었습니다: 첫째, Gaussian noise를 추가하여 클래스 개념의 임베딩 공간에 변형을 가합니다. 둘째, 텍스트-텍스트 모델 GPT-4를 이용하여 클래스별 텍스트 프롬프트를 통한 생성 과정을 유도합니다. 셋째, 질 낮은 샘플의 영향을 줄이기 위해 가중치 메커니즘을 도입합니다.

- **Performance Highlights**: DIAGen은 다양한 데이터셋에서 실험 결과를 통해 시맨틱 다양성을 증진시킬 뿐만 아니라 후속 분류기의 성능도 개선하는 효과를 보였습니다. 특히, 배포되지 않은 샘플에 대해 DIAGen의 장점이 두드러졌습니다.



### A Survey of Camouflaged Object Detection and Beyond (https://arxiv.org/abs/2408.14562)
Comments:
          26 pages, 10 figures, 8 tables

- **What's New**: 이 논문은 Camouflaged Object Detection (COD)에 대한 가장 포괄적인 리뷰를 제공하며, 기존의 조사에서의 한계를 극복하고 최신 기술 및 방법론을 포함하고 있습니다.

- **Technical Details**: COD는 정적 이미지에서 camouflaged 객체를 탐지하는 이미지 수준 COD와 비디오 시퀀스에서 이러한 객체를 탐지하는 비디오 수준 COD(VCOD)로 나뉘며, 각기 다른 접근 방식으로 분석됩니다. 전통적인 방법과 딥러닝 기반 접근 모두를 포함하여 다양한 방법론이 논의됩니다.

- **Performance Highlights**: 딥러닝 기반 기법의 정량적 및 정성적 성능을 분석하고, 40개의 이미지 수준 모델과 8개의 비디오 수준 모델을 벤치마킹하며 평가 메트릭을 사용하여 결과를 종합적으로 제시합니다.



### Exploring the Potential of Synthetic Data to Replace Real Data (https://arxiv.org/abs/2408.14559)
Comments:
          ICIP 2024

- **What's New**: 이 논문은 synthetic data(합성 데이터)가 데이터 집합을 확장하는 방식에 대한 새로운 통찰력을 제공하며, 테스트 도메인과 다른 도메인에서의 소량의 실제 이미지와 함께 사용할 때의 효과를 실험적으로 평가합니다.

- **Technical Details**: 논문에서는 train2test distance(훈련-테스트 거리) 및 AP_{t2t}(average precision over train to test distances; 훈련에서 테스트까지의 거리 평균 정밀도)라는 두 가지 새로운 메트릭을 도입하여 합성 데이터가 훈련 성과에 미치는 영향을 분석합니다. 이 메트릭은 훈련 세트가 테스트 세트의 인스턴스와 배경 인스턴스를 얼마나 잘 표현하는지를 측정합니다.

- **Performance Highlights**: 실험 결과, 합성 데이터가 실제 데이터를 대체하는 능력은 더 많은 실제 훈련 이미지가 사용될수록 향상되며, 합성 데이터는 중간 신뢰도 감지에 더 큰 영향을 미치는 것으로 나타났습니다. 또한, 합성 데이터의 사용은 테스트 세트에 따라 다르게 나타났으며, 이는 주로 테스트 세트에서 잘못된 긍정(false positive)의 발생에 미치는 영향이 다르기 때문입니다.



### Revisiting Image Captioning Training Paradigm via Direct CLIP-based Optimization (https://arxiv.org/abs/2408.14547)
Comments:
          BMVC 2024

- **What's New**: 이 논문에서는 이미지 캡셔닝(Image Captioning)을 위한 새로운 훈련 패러다임인 Direct CLIP-Based Optimization (DiCO)를 제안합니다. 기존의 SCST(Self-Critical Sequence Training) 방식의 불안정성과 캡션 품질의 저하 문제를 해결하려는 접근입니다.

- **Technical Details**: DiCO는 사람과의 높은 상관관계를 가진 학습 가능한 캡셔닝 평가자로부터 증류된 보상 모델(reward model)을 공동 학습하고 최적화합니다. 이를 통해 원래 모델의 수렴을 방지하고, 캡셔너(Captioner) 내부에서 가중 분류 문제를 해결하여 다양한 캡션 품질 지표에서 동시에 최적화가 가능합니다.

- **Performance Highlights**: DiCO는 COCO 데이터셋에서 광범위한 실험을 하고, 현대적인 측정 지표에서 개선된 품질과 훈련의 안정성을 보여 주었으며, 전통적인 지표에서도 경쟁력 있는 성과를 유지했습니다. DiCO는 앤지 데이터셋 외에도 6개의 다른 이미지 캡셔닝 벤치마크에서 일반화 능력을 입증했습니다.



### DCT-CryptoNets: Scaling Private Inference in the Frequency Domain (https://arxiv.org/abs/2408.15231)
Comments:
          Under Review; 10 pages content, 3 pages appendix, 4 figures, 8 tables; Code TBD

- **What's New**: DCT-CryptoNets라는 새로운 접근법이 소개되었으며, 이는 주파수 도메인 학습(frequency-domain learning)을 활용하여 완전 동형 암호화(fully homomorphic encryption) 기반의 깊은 신경망(deep neural networks)에서 발생하는 높은 계산 비용, 지연(latency) 및 확장성 문제를 해결합니다.

- **Technical Details**: DCT-CryptoNets는 JPEG 압축에서 일반적으로 사용되는 이산 코사인 변환(discrete cosine transform, DCT)을 이용하여 이미지 데이터를 주파수 텐서로 변환합니다. 이 방법은 특히 저주파(low-frequency) 구성 요소에 집중하여 동형 연산의 계산 부담을 줄입니다. 또한, 이 접근법은 원격 컴퓨팅 서비스와 본질적으로 호환됩니다.

- **Performance Highlights**: DCT-CryptoNets는 이미지 분류 작업에서 기존 연구와 비교하여 최대 5.3배의 지연 감소를 보였으며, ImageNet 추론의 경우 2.5시간 내에 처리하며, 이는 과거 12.5시간 소요와 비교되는 큰 개선입니다. 또한, DCT-CryptoNets는 ImageNet에서의 정확도 변동성을 ±2.5%에서 ±1.0%로 줄여 신뢰성을 높였습니다.



### SAM & SAM 2 in 3D Slicer: SegmentWithSAM Extension for Annotating Medical Images (https://arxiv.org/abs/2408.15224)
Comments:
          Future work: support for box and mask inputs for the video predictor of SAM 2

- **What's New**: 이번 논문에서는 3D 의료 영상의 주석을 작성하기 위해 Segment Anything Model 2 (SAM 2) 모델을 3D Slicer라는 인기 있는 주석 소프트웨어에 통합하는 방법을 소개합니다. 사용자는 2D 슬라이스에 포인트 프롬프트를 배치하여 주석 마스크를 생성하고, 이를 전체 볼륨에 걸쳐 전파할 수 있습니다.

- **Technical Details**: 이 연구에서는 의료 이미지의 정확한 분할을 위한 SAM과 SAM 2 모델을 3D Slicer에 통합하여 2D 및 3D 분할 작업을 가능하게 합니다. 사용자는 3D Slicer의 사용자 인터페이스를 통해 필요에 따라 임의의 슬라이스를 세분화할 수 있으며, 모든 체크포인트를 선택할 수 있습니다. 3D 의료 이미지를 비디오처럼 처리하여 특정 슬라이스에서 조건부 프롬프트를 설정함으로써 여러 슬라이스에 분할 마스크를 전파할 수 있습니다.

- **Performance Highlights**: 실험 결과, SAM 2 모델을 사용하여 다양한 의료 데이터 샘플에서 평균적으로 우수한 세분화 성능을 보였습니다. 특히 간 및 종양 세분화 작업에서 점 프롬프트를 통해 효과적인 결과를 도출했으며, 전체 볼륨에서의 3D 마스크 생성 또한 가능함을 입증했습니다.



### Histo-Diffusion: A Diffusion Super-Resolution Method for Digital Pathology with Comprehensive Quality Assessmen (https://arxiv.org/abs/2408.15218)
Comments:
          We have submitted our paper to Medical Image Analysis and are currently awaiting feedback

- **What's New**: 디지털 병리학의 분야에서 큰 진전을 보이고 있으며, Histo-Diffusion이라는 새로운 방법이 슈퍼 해상도(Super-resolution) 이미지를 생성하고 평가하는 데 특화되어 있습니다. 이 방법은 기존의 GAN 기반 접근 방식보다 뛰어난 성능을 발휘합니다.

- **Technical Details**: Histo-Diffusion은 두 가지 주요 모듈을 포함합니다: 하나는 조직 병리학적 정보를 복원하는 모듈이고, 다른 하나는 고품질 이미지를 생성하기 위한 제어 가능한 확산 모듈입니다. 본 연구는 TCGA 데이터베이스를 사용해 병리학 이미지 품질 평가(IQA) 데이터셋을 구축하고, 생성된 이미지의 품질을 평가하기 위한 포괄적인 평가 전략을 제안합니다.

- **Performance Highlights**: Histo-Diffusion은 여러 암 도메인에서 ST-LPIPS 점수가 최적으로 개선되었으며, PRAD에서는 12.93%, LUAD에서는 20.83%, GBM에서는 12.88% 향상되었습니다. 또한, MUSIQ 점수의 향상도 보였으며, 질감과 강도 유사성이 각각 PRAD, GBM에서 17.96% 및 19.74% 개선되었습니다.



### Fundus2Video: Cross-Modal Angiography Video Generation from Static Fundus Photography with Clinical Knowledge Guidanc (https://arxiv.org/abs/2408.15217)
Comments:
          The paper has been accepted by Medical Image Computing and Computer Assisted Intervention Society (MICCAI) 2024

- **What's New**: 이 연구에서는 Color Fundus (CF) 이미지를 기반으로 한 동적 Fundus Fluorescein Angiography (FFA) 비디오 생성 방법을 최초로 제안하고 있습니다. 기존의 방법들이 정적인 이미지 생성을 중심으로 한 반면, 본 연구에서는 프레임별로 부드러운 FFA 비디오를 생성하는 autoregressive GAN 아키텍처인 Fundus2Video를 도입하였습니다.

- **Technical Details**: Fundus2Video 모델은 이미지 간의 종속성을 포착하기 위해 autoregressive GAN 아키텍처를 사용하며, CF 이미지를 입력으로 받아 연속적인 FFA 비디오 프레임을 생성합니다. 이 과정에서 지식 마스크(knowledge mask)를 활용하여 병변 변화가 중요한 영역에 집중할 수 있도록 설계하였습니다. 또한, pixel misalignment 문제를 해결하기 위해 mask-enhanced patchNCE loss를 적용하였습니다.

- **Performance Highlights**: 이 방법은 기존의 비디오 생성 기법들에 비해 FVD(Fréchet Video Distance) 1503.21 및 PSNR(Peak Signal-to-Noise Ratio) 11.81라는 최상의 성능을 기록하였으며, 안과 의사에 의한 인간 평가에서도 높은 생성 품질을 인정받았습니다. 특히, 제안된 지식 마스크는 감독된 병변 세분화 마스크에 비해 더 나은 성능을 보여주어 전통적인 FFA에 대한 비침습적인 대안을 제공합니다.



### X-Reflect: Cross-Reflection Prompting for Multimodal Recommendation (https://arxiv.org/abs/2408.15172)
- **What's New**: 본 논문에서는 Cross-Reflection Prompting (X-Reflect)이라는 새로운 프레임워크를 도입하여, 텍스트와 이미지 간의 지원 및 상충 정보를 명시적으로 식별하고 조화롭게 통합하는 방식으로 추천 시스템의 성능을 향상시키고자 하였습니다.

- **Technical Details**: X-Reflect 방법은 LLM과 LMM에서 텍스트와 이미지 정보를 동시 처리하여, 두 모달리티 간의 상호 지원 혹은 상충 정보를 파악하고 조화롭게 통합하는 과정을 포함합니다. 이는 별도의 프롬프트 방식과 결합된 프롬프트 방식을 통해 구현됩니다.

- **Performance Highlights**: 두 개의 널리 사용되는 벤치마크에서 수행한 광범위한 실험을 통해, 제안된 방법이 기존의 프롬프트 기본선 대비 추천 정확도에서 일관되게 우수한 성능을 보임을 입증하였습니다.



### DIFR3CT: Latent Diffusion for Probabilistic 3D CT Reconstruction from Few Planar X-Rays (https://arxiv.org/abs/2408.15118)
Comments:
          11 pages, 9 figures

- **What's New**: DIFR3CT는 고품질의 극히 드문 CT 재구성을 위한 첫 번째 조건부 잠재 확산 모델입니다. 이 모델은 최소한의 입력 평면 엑스레이 이미지만으로도 가능성 있는 3D CT 이미지를 생성합니다.

- **Technical Details**: DIFR3CT는 2D 엑스레이 내용을 결합해 3D 공간에서 확산(diffusion)을 수행하는 방식으로 작동합니다. 저차원 잠재 공간에서 조건부 확산 모델을 학습해 CT 볼륨을 생성하며, 불확실성 측정을 위한 몬테 카를로 샘플링(Monte Carlo sampling)도 지원합니다.

- **Performance Highlights**: DIFR3CT는 LIDC 및 내부 데이터 세트인 Thoracic post-mastectomy CT에서 다양한 희소 뷰 CT 알고리즘 성능을 초과하며, 5명의 환자를 대상으로 자동화된 유방 방사선 치료 윤곽 작성 및 계획에 유망한 가능성을 보였습니다.



### Constrained Diffusion Models via Dual Training (https://arxiv.org/abs/2408.15094)
Comments:
          41 pages, 4 figures, 2 tables

- **What's New**: 본 논문에서는 제한된 조건을 기반으로 한 확산 모델(constrained diffusion models)을 제안합니다. 기존의 확산 모델들이 편향된 데이터 생성을 초래할 수 있는 문제를 해결하기 위해, 우리는 특정 요구 사항을 반영한 데이터 분포를 따르도록 하는 제약 조건을 두었습니다.

- **Technical Details**: 제한된 생성 모델을 훈련하기 위해, 우리는 Lagrangian 기반의 이중 훈련 알고리즘을 개발했습니다. 이는 무한 차원의 분포 공간에서 제약 최적화(constrained optimization) 관점에서 적절한 데이터 분포의 혼합(mixture data distribution)으로부터 새로운 데이터를 생성하는 것을 목표로 합니다.

- **Performance Highlights**: 우리는 두 가지 제약 조건 생성 작업에서 우리의 제한된 모델의 효과를 실증적으로 입증했습니다. 첫 번째 작업에서는 미소수 집단에 대한 샘플링 공정을 개선하여 모든 클래스에서 공정한 샘플링을 보장했습니다. 두 번째 작업에서는 미세 조정된 모델이 기본 생성 능력을 유지하면서 새로운 데이터를 생성하는 능력을 보여주었습니다.



### Alternating Minimization Schemes for Computing Rate-Distortion-Perception Functions with $f$-Divergence Perception Constraints (https://arxiv.org/abs/2408.15015)
Comments:
          This work has been submitted for possible publication

- **What's New**: 본 논문에서는 단일 문자 평균 왜곡 제약 및 f-divergences 계열에 해당하는 지각 제약을 받는 이산 메모리 없는 소스의 비율-왜곡-지각 기능(RDPF) 계산을 연구합니다. RDPF는 볼록 프로그래밍 문제를 형성하며 최적 매개변수 솔루션을 특성화합니다. 이 결과를 Optimal Alternating Minimization (OAM) 이라는 교대 최소화 스킴에 적용하고 수렴 보장을 제공합니다.

- **Technical Details**: RDPF 문제는 convex program으로 등장하며, 이를 바탕으로 우리는 OAM과 두 가지 대체 최소화 접근 방식(즉, NAM과 RAM)을 제안합니다. NAM은 Newton의 root-finding 방법을 활용하여 최적 반복 솔루션을 근사하는 방식이고, RAM은 OAM 반복 구조의 완화를 통해 전역 최적 솔루션으로의 수렴을 보장합니다. 수렴 보장을 위해 충분한 조건을 설정하고 수치 시뮬레이션을 통해 이 이론적 결과를 지원합니다.

- **Performance Highlights**: NAM과 RAM 스킴은 충분한 조건을 바탕으로 하여 여러 반복 수에서 기하급수적으로 빠르게 수렴합니다. 특히 TV 지각 메트릭의 경우에서 이 점을 강조하며, 논문에서 개발한 부드러운 근사치를 포함한 수치 결과를 제시하고 있습니다.



### Knowledge Discovery in Optical Music Recognition: Enhancing Information Retrieval with Instance Segmentation (https://arxiv.org/abs/2408.15002)
Comments:
          8 pages content and one references, accepted version at the International Conference on Knowledge Discovery and Information Retrieval 2024, Porto, Portugal

- **What's New**: 본 연구는 Optical Music Recognition (OMR) 분야에서 Mask R-CNN을 이용한 instance segmentation 기법을 적용하여 음악 기호의 탐지 및 구획을 개선하는 방법을 제안합니다. 이 방법을 통해 복잡한 Common Western Music Notation (CWMN)의 의미를 보다 잘 해석할 수 있습니다.

- **Technical Details**: OMR 시스템에 대한 성능 향상을 위해 instance segmentation과 같은 고급 심층 학습 기법을 도입합니다. 특히, Mask R-CNN을 사용하여 픽셀 수준의 음악 기호 분류를 수행하며, 전통적인 컴퓨터 비전 기법을 통한 staff detection 단계도 추가하여 음고 추정을 지원합니다. 이로써 낮은 연산 비용과 긴밀한 기호 분류를 달성합니다.

- **Performance Highlights**: DoReMi 및 MUSCIMA++ 데이터셋에서 수행한 평가 결과, 제안된 방법은 mAP가 최대 59.70%에 도달하는 등 고밀도 기호 환경에서도 뛰어난 성능을 보여주었습니다. 이러한 개선은 OMR 기술의 발전에 중요한 기여를 하며, 음악 데이터베이스에서의 효과적인 정보 검색 및 지식 발견을 지원합니다.



### Depth Restoration of Hand-Held Transparent Objects for Human-to-Robot Handover (https://arxiv.org/abs/2408.14997)
Comments:
          7 pages, 7 figures, conference

- **What's New**: 이 논문은 투명 객체 인식을 위한 새로운 방법을 제시하며, 이는 일련의 RGB-D 이미지에서 손 포즈 정보를 통합하여 깊이 복원을 수행합니다. 이 방법은 손 자세를 중요한 가이딩 요소로 사용하여 의미적이고 기하학적인 정보를 활용합니다.

- **Technical Details**: 제안된 Hand-Aware Depth Restoration (HADR) 방법은 단일 RGB-D 이미지로부터 암시적 신경 표현 함수를 생성하고, 이를 통해 손에 의해 가려진 투명 객체의 깊이 정보를 복원합니다. 이 과정에서 TransHand-14K라는 고품질 합성 데이터셋이 사용됩니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존 방법들에 비해 깊이 복원에서 우수한 성능을 보이며, 특히 투명 객체에 대한 일반화 능력이 향상되었습니다. 또한, 실제 환경에서 인간-로봇 간의 물체 인계 시스템을 성공적으로 구축하여 이 방법의 적용 가치를 입증하였습니다.



### LN-Gen: Rectal Lymph Nodes Generation via Anatomical Features (https://arxiv.org/abs/2408.14977)
Comments:
          8 pages

- **What's New**: 이 연구에서는 직장암의 림프절 세분화(segmentation) 문제를 해결하기 위해, 실제와 유사한 다양한 합성 직장 림프절 샘플을 생성하는 새로운 합성 기법을 제안합니다.

- **Technical Details**: 이 기법은 Implicit SDF(Signed Distance Function) 기반의 방법을 사용하여 안정적이고 연속적인 마스크를 생성하며, 신체의 해부학적 구조를 실제와 유사하게 캡처하여 세분화 모델의 학습을 지원합니다. 합성 구조물과 CT 이미지로 이루어진 데이터 풀을 구성하여 생성 과정의 다양성을 높이고, 합성 데이터를 실제 데이터와 결합하여 세분화 모델을 훈련합니다.

- **Performance Highlights**: 실험 결과, 합성된 데이터가 세분화 성능을 상당히 개선함을 보여주었으며, 이는 직장암 진단과 치료 개선에 기여할 것으로 기대됩니다.



### Prior-free Balanced Replay: Uncertainty-guided Reservoir Sampling for Long-Tailed Continual Learning (https://arxiv.org/abs/2408.14976)
- **What's New**: 이 연구에서는 기존 기술에 의존하지 않고 장기적 분포의 데이터를 학습하는 새로운 방법인 Prior-free Balanced Replay (PBR) 프레임워크를 제안합니다. 이는 불균형한 클래스에서 정보의 잊힘을 줄이기 위해 설계되었습니다.

- **Technical Details**: PBR 프레임워크는 불확실성 기반의 저수지 샘플링 전략을 활용해 마이너리티( minority) 데이터를 우선적으로 메모리에 저장하는 방식입니다. 추가적으로, 두 가지 prior-free 구성 요소인 boundary constraint와 prototype constraint를 도입하여 마이너리티 데이터의 잊힘 문제를 완화합니다.

- **Performance Highlights**: 이 방법은 세 가지 장기적 벤치마크에서 평가되어 기존의 Continual Learning(CL) 방법들과 SOTA LTCL 접근 방식보다 우수한 성능을 기록하였습니다. 이는 작업 증가 및 클래스 증가 설정에서 모두 효과적입니다.



### ERX: A Fast Real-Time Anomaly Detection Algorithm for Hyperspectral Line-Scanning (https://arxiv.org/abs/2408.14947)
Comments:
          10 pages, 9 figures, 3 tables, code and datasets accessible at this https URL

- **What's New**: 이 연구는 Hyperspectral line-scan 카메라를 사용한 실시간 이상 탐지에 대한 새로운 접근 방식인 Exponentially moving RX 알고리즘(ERX)을 도입합니다. 이 알고리즘은 기존의 RX 기반 방법과 비교하여 속도와 검출 성능에서 우수한 결과를 나타냈습니다.

- **Technical Details**: ERX 알고리즘은 소규모 컴퓨터(예: 드론 또는 소형 위성)에서 실행할 수 있도록 설계되었습니다. 이 알고리즘은 고차원 데이터에 확장 가능하고 환경 변화에 적응할 수 있으며 기하학적 및 방사선 왜곡에 대해 견고합니다. RX 알고리즘의 단점을 보완하기 위해, ERX는 시간 순서에 따라 과거 및 현재 데이터만을 접근하는 방식을 사용합니다.

- **Performance Highlights**: ERX는 Jetson Xavier NX 컴퓨트 모듈을 사용하여 세 가지 새로운 데이터셋에서 테스트되었으며, 속도와 검출 성능의 최적 조합을 달성했습니다. 또한 Python 코드가 공개되어 있어 연구자들이 쉽게 사용할 수 있도록 제공됩니다.



### Automatic Detection of COVID-19 from Chest X-ray Images Using Deep Learning Mod (https://arxiv.org/abs/2408.14927)
Comments:
          Accepted in AIP Conference Proceedings (Vol. 2424, No. 1)

- **What's New**: 이번 논문에서는 미국 주요 코로나바이러스(2019-nCoV) 진단을 위한 새로운 자동화된 시스템을 제안하고 있습니다. 기존의 RT-PCR 방식으로는 검사 키트의 부족으로 인해 모든 환자를 검사하기 어려운 상황에서 Chest X-ray를 활용한 진단 방법이 소개되었습니다.

- **Technical Details**: 제안하는 모델은 Deep Learning을 기반으로 하여 U-Net 아키텍처를 사용해 Chest X-ray 이미지의 세분화(semantic segmentation)를 수행합니다. 모델은 400x400 픽셀 크기의 흑백 이미지를 입력으로 하며, convolutions와 max-pooling을 통해 다운샘플링 및 업샘플링 과정을 거칩니다. 최종적으로 2클래스(코로나19 vs 정상) 및 3클래스(코로나19 vs 정상 vs 폐렴) 분류 정확도를 99.17%, 97.50%로 달성했습니다.

- **Performance Highlights**: 제안한 모델은 공개된 COVID-19 Chest X-ray 데이터셋에서 2클래스 분류에서 99.17%의 분류 정확도와 100%의 민감도를, 3클래스 분류에서 97.50%의 정확도와 100%의 민감도를 기록했습니다. 이는 기존의 다른 방법들에 비해 우수한 성능으로, 방사선 전문의들이 초기 스크리닝 결과를 검증하는 데 도움을 줄 수 있습니다.



### VHAKG: A Multi-modal Knowledge Graph Based on Synchronized Multi-view Videos of Daily Activities (https://arxiv.org/abs/2408.14895)
Comments:
          5 pages,4 figures, accepted by CIKM2024 Resource Track

- **What's New**: 이번 논문에서는 여러 이벤트로 구성된 일상 활동 비디오의 Multi-modal Knowledge Graph (MMKG)을 생성했습니다. 이 MMKG는 비디오의 프레임별로 세밀한 변화를 포함하며, 일상 활동을 이벤트 중심으로 표현합니다.

- **Technical Details**: VirtualHome-AIST-KG (VHAKG)라는 새로운 MMKG를 소개하며, 다양한 원시 행동을 렌더링하고 2D 바운딩 박스를 자동으로 주석 처리하는 기능을 추가했습니다. 이 데이터는 웹에서 지속 가능하게 사용할 수 있도록 압축되어 배포됩니다.

- **Performance Highlights**: VHAKG를 이용하면 시각 언어 모델(LVLMs)의 성능을 평가하기 위한 맞춤형 테스트 데이터 세트를 쉽게 추출할 수 있습니다. 이러한 MMKG를 사용한 비교 Benchmarking이 가능해져, 다양한 비전-언어 작업의 발전에 기여할 것으로 예상됩니다.



### Intraoperative Glioma Segmentation with YOLO + SAM for Improved Accuracy in Tumor Resection (https://arxiv.org/abs/2408.14847)
- **What's New**: 이번 연구에서는 수술 중 실시간 MRI(ioMRI) 이미지를 개선하기 위한 딥러닝 파이프라인을 제안합니다. 이 모델은 YOLOv8과 SAM(Segment Anything Model) Vision Transformer를 결합하여 피질 종양인 글리오마를 감지하고 세분화하는 데 중점을 두고 있습니다.

- **Technical Details**: 모델은 BraTS 2021 데이터셋을 기반으로 훈련되었으며, Gaussian noise가 추가된 MRI 이미지를 사용하여 ioMRI 이미지를 모의했습니다. 이 과정에서 YOLOv8을 사용하여 종양을 탐지한 후, SAM 모델을 통해 세부적인 세분화 결과를 제공합니다.

- **Performance Highlights**: 모델은 DICE 점수 0.79를 달성하였으며, 이는 기존의 최신 세분화 모델과 유사한 성능을 나타냅니다. 더불어 YOLO + SAM 모델은 15초에서 25초의 빠른 추론 시간을 기록하여 수술 중 실시간 응용에 더 적합합니다.



### Diffusion Models Are Real-Time Game Engines (https://arxiv.org/abs/2408.14837)
Comments:
          Project page: this https URL

- **What's New**: GameNGen은 신경망(neural model)에 의해 완전히 구동되는 최초의 게임 엔진으로, 복잡한 환경에서 실시간 상호작용을 가능하게 합니다. 이 엔진은 인기 게임 DOOM을 단일 TPU에서 초당 20프레임 이상으로 시뮬레이션 할 수 있습니다.

- **Technical Details**: GameNGen은 두 단계로 훈련됩니다: (1) 강화 학습( reinforcement learning, RL) 에이전트가 게임을 플레이하며 훈련 세션을 기록하고, (2) 확산 모델(diffusion model)이 과거 프레임과 동작의 시퀀스에 따라 다음 프레임을 생성하도록 훈련됩니다. 이러한 조건화( conditioning)는 긴 경로를 따라 안정적인 자기 회귀적(auto-regressive) 생성을 가능하게 합니다.

- **Performance Highlights**: 다음 프레임 예측의 PSNR은 29.4로, 손실 JPEG 압축과 유사한 품질을 나타냅니다. 인간 평가자는 게임의 짧은 클립과 시뮬레이션의 클립을 구별하는 데 있어서 거의 무작위 확률에 가까운 성과를 보입니다.



### From Rule-Based Models to Deep Learning Transformers Architectures for Natural Language Processing and Sign Language Translation Systems: Survey, Taxonomy and Performance Evaluation (https://arxiv.org/abs/2408.14825)
- **What's New**: 이번 논문은 전 세계적으로 증가하는 청각 장애인과 난청 인구를 위해 수화 기계 번역 시스템의 필요성을 강조하고 있습니다. 기존의 연구가 수화의 동적이고 연속적인 특성을 충분히 고려하지 못한 점을 지적하며, 새로운 접근 방식을 제공합니다.

- **Technical Details**: 이 논문은 수화 기계 번역 알고리즘의 시간적 진화를 회고적으로 분석하고, 언어 번역에서 가장 많이 사용되는 Transformers 아키텍처의 분류법을 제공합니다. 또한, 정확한 deep learning 알고리즘에 기반한 실시간 Quality-of-Service 수화 기계 번역 시스템의 요구 사항을 제시합니다.

- **Performance Highlights**: 사실상, 수화 기계 번역의 발전을 위한 미래 연구 방향도 제안됩니다. 이 시스템은 언어의 연속성과 동적 특성을 효과적으로 처리할 수 있는 가능성을 보여줍니다.



### Generalist Segmentation Algorithm for Photoreceptors Analysis in Adaptive Optics Imaging (https://arxiv.org/abs/2408.14810)
- **What's New**: 본 연구에서는 Confocal Adaptive Optics Scanning Light Ophthalmoscope (AOSLO) 이미지를 사용하여 cone photoreceptor (콘) 세포의 자동화된 탐지 및 분할을 위한 심층 학습(Deep Learning, DL) 기반 방법을 소개합니다. 이 방법은 기존의 수작업 라벨링 절차를 크게 줄일 수 있는 가능성을 제시합니다.

- **Technical Details**: 연구팀은 18명 참가자의 20개의 AOSLO 이미지 배치로 구성된 반자동 라벨링 데이터셋을 사용하여 모델을 훈련시켰습니다. 이 모델은 foveal center로부터 0도, 1도, 2도 위치에서 각각 F1 점수가 0.968, 0.958, 0.954로, 기존의 DL 접근법들보다 뛰어난 성능을 발휘하였습니다.

- **Performance Highlights**: 본 연구의 새로운 DL 기법은 AOSLO 이미지의 콘 라벨링에서 5%라는 적은 라벨 데이터만 필요로 하며, 이는 특히 라벨 데이터가 제한적인 안과 분야에서 큰 이점으로 작용합니다. 또한, 수동 라벨링의 필요성을 줄이면서 연구자들의 작업 부담도 경감할 수 있습니다.



### Sequential-Scanning Dual-Energy CT Imaging Using High Temporal Resolution Image Reconstruction and Error-Compensated Material Basis Image Generation (https://arxiv.org/abs/2408.14754)
- **What's New**: 이번 논문에서는 전통적인 CT 시스템에 직접 적용 가능하고, 전문 하드웨어 설계가 필요 없는 순차 스캐닝 데이터 수집 방식의 이점을 활용한 이중 에너지 컴퓨터 단층 촬영(dual-energy computed tomography, DECT) 기법을 개선하는 방법을 제안합니다. 이 방법은 ACCELERATION이라는 기법을 기반으로 하여 보다 정확한 물질 농도 정량화를 가능하게 합니다.

- **Technical Details**: ACCELERATION은 고해상도 이미지 재구성과 오류 보완 물질 기초 이미지 생성을 결합하여 순차 스캐닝 DECT의 기술적 문제를 해결합니다. 이 과정에서 두 가지 주요 기술이 사용되며, 첫째로 고Temporal Resolution 이미지 재구성을 통한 시간적 매칭과, 둘째로 오류 보완 물질 기반 이미지 생성을 통한 물질 정량화 기술입니다.

- **Performance Highlights**: ACCELERATION을 사용한 결과, 정량화 정확도와 이미지 품질이 개선되었음을 보여줍니다. 이 접근법은 임상 적용 가능성을 높이고, 보다 넓은 범위의 환자들에게 혜택을 줄 수 있는 저비용의 DECT 기법이 구현되었음을 의미합니다.



### Learning Differentially Private Diffusion Models via Stochastic Adversarial Distillation (https://arxiv.org/abs/2408.14738)
Comments:
          accepted by ECCV 2024

- **What's New**: 이번 논문에서는 DP-SAD라는 새로운 방법론을 제안했습니다. 이는 개인 정보를 보호하면서 데이터 생성의 품질을 향상시키기 위해 확산 모델(difussion models)을 활용한 접근 방식을 사용합니다.

- **Technical Details**: DP-SAD는 교사 모델(teacher model)과 학생 모델(student model) 사이의 스토캐스틱 적대적 증류(stochastic adversarial distillation) 방법을 통해 훈련됩니다. 노이즈를 추가하여 개인 정보를 보호하며, 디스크리미네이터(discriminator)를 도입하여 이미지의 품질을 높이고 훈련 과정을 가속화합니다.

- **Performance Highlights**: 광범위한 실험을 통해 DP-SAD는 기존의 방법들보다 더 나은 이미지 생성을 보여주며, 자원 제약이 있는 상황에서도 효과적입니다. 또한, 적은 배치 사이즈로 훈련할 수 있다는 장점도 제시되었습니다.



### Smart Multi-Modal Search: Contextual Sparse and Dense Embedding Integration in Adobe Express (https://arxiv.org/abs/2408.14698)
- **What's New**: 이번 논문은 Adobe Express 템플릿 검색에서 다중모드(multi-modal) 검색 시스템을 위한 새로운 아키텍처를 소개합니다. CLIP과 같은 다중모드 임베딩을 활용하여 텍스트와 이미지 검색을 직접 지원하면서도, 사용자 지리적 특성이나 최근성 같은 컨텍스트(contextual features)를 통합하는 데의 도전 과제를 다룹니다.

- **Technical Details**: 이 논문에서는 클라이언트의 검색 요구를 충족하기 위해 여러 다중모드 모델들을 사용하였으며, AB 테스트를 통해 임베딩 모델 선택, 매칭 및 랭킹의 역할, 밀집(dense)과 희소(sparse) 임베딩 간의 균형을 최적화하였습니다. AdobeCS 기술을 활용한 다중모드 검색 시스템은 약 30만 개의 템플릿 데이터에서 매우 효율적으로 작동하도록 설계되었습니다.

- **Performance Highlights**: 이번 연구는 희소, 밀집, 컨텍스트 특성을 활용하여 짧은 쿼리와 긴 쿼리 검색을 향상시키고, null 비율을 70% 이상 줄이며 클릭률(CTR)을 증가시키는 데 기여했습니다. 이러한 결과는 복잡한 쿼리에 대한 검색 시스템의 효과적인 개발이 가능하다는 통찰을 제공합니다.



### Enhancing Neural Network Interpretability Through Conductance-Based Information Plane Analysis (https://arxiv.org/abs/2408.14681)
Comments:
          16 pages, 10 figures

- **What's New**: 본 논문에서는 기존의 활성화 기반 방법 대신 layer conductance를 활용하여 Neural Networks의 정보 처리 동역학을 보다 정확하게 분석하는 새로운 접근법을 제안합니다.

- **Technical Details**: 제안된 conductance 기반 Information Plane과 새로운 Information Transformation Efficiency (ITE) 메트릭을 사용하여, pretrained ResNet50 및 VGG16 모델에서 ImageNet 데이터셋으로 실험을 수행했습니다. 이 연구는 gradient 기반 기여도를 통합하여 정보 동역학의 정밀한 특성을 제공합니다.

- **Performance Highlights**: 실험 결과, 모델 성능 및 해석 가능성에 중요한 기여를 하는 숨겨진 층을 식별할 수 있는 능력을 보여주었으며, 정보 압축, 보존 및 활용에 대한 통찰을 제공합니다. 이 접근법은 Neural Networks 내의 의사 결정 과정을 이해하는 데 기여합니다.



### BreakNet: Discontinuity-Resilient Multi-Scale Transformer Segmentation of Retinal Layers (https://arxiv.org/abs/2408.14606)
- **What's New**: BreakNet은 가시광선 광학적 간섭 단층 촬영(visible light optical coherence tomography, vis-OCT)을 사용하는 망막 이미징에서 boundary discontinuities를 해결하기 위해 설계된 다중 스케일 Transformer 기반 분할(segmentation) 모델입니다.

- **Technical Details**: BreakNet은 계층적 Transformer와 convolutional 블록을 활용하여 다중 스케일(global, local) 특징 맵을 추출합니다. 이 모델은 정밀한 분할을 보장하기 위해 디코더 블록을 포함하고 있으며, pathwaproys를 확장하여 세부 사항 및 의미적 정보를 효과적으로 추출합니다.

- **Performance Highlights**: BreakNet은 prototype vis-OCT로 얻은 설치어(rodent) 망막 이미지에서 TCCT-BP 및 U-Net과 같은 최신 분할 모델보다 뛰어난 성능을 보여주어, 제한된 품질의 ground truth 데이터에서도 효과적임을 입증하였습니다.



### Improving Nonlinear Projection Heads using Pretrained Autoencoder Embeddings (https://arxiv.org/abs/2408.14514)
Comments:
          15 pages, 1 figure

- **What's New**: 본 연구는 SimCLR 프레임워크에서 사용하는 표준 2레이어 MLP 프로젝션 헤드의 효과성을 향상시키기 위해 pretrained autoencoder embedding을 활용하는 실증 연구입니다.

- **Technical Details**: 대조 학습(Contrastive Learning) 작업을 위한 데이터셋에서 얕은 autoencoder 아키텍처를 학습하고, 인코더의 embedding 레이어에 있는 압축된 표현을 추출하여 이 pretrained 레이어의 가중치를 고정한 후 SimCLR의 기본 프로젝터의 입력 레이어를 대체합니다. 아울러, 프로젝터의 너비를 줄이고 활성화 함수(Activation Function)를 변경하는 등 아키텍처 변경도 적용합니다.

- **Performance Highlights**: pretrained autoencoder embedding을 사용한 프로젝터가 분류 정확도를 평균 1.7%에서 최대 2.9% 향상시킬 수 있으며, 프로젝션 공간의 차원 수를 크게 감소시키는 것으로 나타났습니다. 또한, sigmoid 및 tanh 활성화 함수를 사용하는 것이 ReLU보다 분류 정확도에서 더 우수한 성능을 보일 수 있음을 발견했습니다.



New uploads on arXiv(cs.AI)

### Aligning XAI with EU Regulations for Smart Biomedical Devices: A Methodology for Compliance Analysis (https://arxiv.org/abs/2408.15121)
Comments:
          Accepted for publication at ECAI 2024, main-track

- **What's New**: 본 연구는 의료 및 헬스케어 애플리케이션에서 Explainable AI (XAI) 기술의 선택 과정을 조사하여 EU 규정의 설명 요건을 준수하기 위한 적절한 XAI 방법을 선택하는 과정을 제시합니다. 이를 통해 의료 분야의 스마트 바이오일렉트로닉스와 EU 규정의 조화를 이룹니다.

- **Technical Details**: 연구 방법론은 스마트 의료 기기의 종류를 제어 메커니즘에 따라 분류하고, 이를 바탕으로 GDPR, AI 법률, MDR의 설명 요건을 분석합니다. 개발된 XAI 방법들을 설명 목표에 따라 분류하고, 각 설명 요건에 맞는 XAI 알고리즘을 매칭하는 과정을 포함하여, 법적 요구사항과 기술적 가능성을 연계합니다.

- **Performance Highlights**: 본 연구는 여러 가지 신경 임플란트 사례를 통해 EU 규정에 부합하는 다양한 XAI 알고리즘의 적합성을 보여주며, 개발자와 연구자들에게 실질적인 프레임워크를 제공하여 AI 혁신이 법적 및 윤리적 기준을 준수하도록 안내합니다.



### Evaluating Stability of Unreflective Alignmen (https://arxiv.org/abs/2408.15116)
- **What's New**: AI 정렬(Alignment)의 미래 세대 연구에 대한 우선 순위를 명확히 하기 위해, Reflective Stability와 관련된 위험 요소인 Counterfactual Priority Change (CPC) 불안정성에 대한 새로운 기제를 제안합니다.

- **Technical Details**: CPC 기반 후퇴(CPC-based stepping back)와 선호 불안정성(preference instability)이라는 두 가지 위험 요소를 정의하고 이를 통해 LLM(대형 언어 모델)에서의 Reflective Stability 문제 발생 가능성을 분석합니다. 실험으로는 CPC 곡선 실험과 다중 팔 쥐 실험(Multi-Armed Bandit)을 진행합니다.

- **Performance Highlights**: 현재의 LLM에서는 확대된 규모와 능력이 CPC 기반 후퇴 및 선호 불안정성과 연관되어 있으며, 이는 미래 LLM에서의 Reflective Stability 문제를 초래할 가능성을 시사합니다.



### Interactive dense pixel visualizations for time series and model attribution explanations (https://arxiv.org/abs/2408.15073)
Comments:
          5 pages, 2 figures, accepted at MLVIS 2023

- **What's New**: 이 논문에서는 시간 시계열 데이터와 같은 비직관적인 데이터에 대한 설명을 탐색할 수 있는 상호작용적 비주얼 분석 방법인 DAVOTS를 제안합니다.

- **Technical Details**: DAVOTS는 원시 시간 시계열 데이터, 신경망의 활성화, 그리고 밀집 픽셀 비주얼라이제이션(dense-pixel visualization)을 통해 데이터, 모델의 결정 및 설명을 시각화합니다. 이 과정에서 클러스터링(clustering) 기법을 적용하여 그룹을 강조하고 각 데이터 탐색 전략을 제시합니다.

- **Performance Highlights**: 연구는 FordA 데이터셋에서 훈련된 CNN을 시각화하며 DAVOTS의 효과적인 패턴 발견 및 대규모 데이터셋 탐색을 보여줍니다.



### Earth Observation Satellite Scheduling with Graph Neural Networks (https://arxiv.org/abs/2408.15041)
Comments:
          Accepted at 17th European Workshop on Reinforcement Learning (EWRL 2024)

- **What's New**: 본 논문에서는 Earth Observation Satellite Planning (EOSP) 문제를 해결하기 위해 Graph Neural Networks (GNN)와 Deep Reinforcement Learning (DRL)을 활용하는 새로운 기법을 제안합니다. 이는 기존의 탐색 방법들이 큰 문제에 있어서는 효율적이지 않았던 점을 개선한 것입니다.

- **Technical Details**: EOSP 문제는 agile Earth observation satellite에서 요청된 관측을 스케줄링하는 최적화 문제입니다. GNN은 EOSP의 인스턴스를 그래프로 표현하고, DRL은 최적 스케줄을 찾아내기 위한 학습을 수행합니다. 시뮬레이션 결과, 이 접근 방식은 소규모 문제에서 학습하고, 실제 대규모 문제에 일반화할 수 있는 능력을 보여주었습니다.

- **Performance Highlights**: 제안된 기법은 기존의 기법들과 비교하여 매우 경쟁력 있는 성능을 보였으며, 소규모 인스턴스에서 학습하여 대규모 인스턴스를 효율적으로 해결할 수 있는 좋은 일반화 능력을 가지고 있습니다.



### Flexible categorization using formal concept analysis and Dempster-Shafer theory (https://arxiv.org/abs/2408.15012)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2210.17330

- **What's New**: 이 논문은 비즈니스 프로세스의 범주화를 위한 Dempster-Shafer 이론 기반의 형식적 프레임워크를 개발합니다. 이 프레임워크는 기계 학습 알고리즘이 낮은 수준과 높은 수준의 이상을 탐지하는 데 사용할 수 있도록 설계되었습니다.

- **Technical Details**: Dempster-Shafer mass functions를 사용하여 다양한 재무 계정에 대한 관심을 나타내는 의제를 모델링하며, 이는 여러 대리인 간의 심사 시나리오를 통해 집계된 의제 및 범주화를 도달하도록 합니다. 또한 Formal Concept Analysis(FCA)를 통해 이러한 범주화를 분석합니다.

- **Performance Highlights**: 이 프레임워크는 전문가와의 상호작용을 용이하게 하여 이상 탐지와 분류를 위한 설명 가능한 기계 학습 알고리즘을 제공합니다. 불규칙한 거래를 더욱 명확하게 탐지하여 감사 프로세스의 효율성을 향상시키는 데 기여할 것으로 보입니다.



### VHAKG: A Multi-modal Knowledge Graph Based on Synchronized Multi-view Videos of Daily Activities (https://arxiv.org/abs/2408.14895)
Comments:
          5 pages,4 figures, accepted by CIKM2024 Resource Track

- **What's New**: 이번 논문에서는 여러 이벤트로 구성된 일상 활동 비디오의 Multi-modal Knowledge Graph (MMKG)을 생성했습니다. 이 MMKG는 비디오의 프레임별로 세밀한 변화를 포함하며, 일상 활동을 이벤트 중심으로 표현합니다.

- **Technical Details**: VirtualHome-AIST-KG (VHAKG)라는 새로운 MMKG를 소개하며, 다양한 원시 행동을 렌더링하고 2D 바운딩 박스를 자동으로 주석 처리하는 기능을 추가했습니다. 이 데이터는 웹에서 지속 가능하게 사용할 수 있도록 압축되어 배포됩니다.

- **Performance Highlights**: VHAKG를 이용하면 시각 언어 모델(LVLMs)의 성능을 평가하기 위한 맞춤형 테스트 데이터 세트를 쉽게 추출할 수 있습니다. 이러한 MMKG를 사용한 비교 Benchmarking이 가능해져, 다양한 비전-언어 작업의 발전에 기여할 것으로 예상됩니다.



### Learning Robust Reward Machines from Noisy Labels (https://arxiv.org/abs/2408.14871)
Comments:
          Preprint accepted for publication to the 21st International Conference on Principles of Knowledge Representation and Reasoning (KR 2024)

- **What's New**: 본 논문은 PROB-IRM을 제안합니다. 이는 노이즈가 있는 실행 트레이스로부터 강건한 보상 기계를 학습하는 방법론으로, 강화 학습(RL) 에이전트에게 원활한 학습을 제공합니다.

- **Technical Details**: PROB-IRM은 노이즈 있는 예제를 처리할 수 있는 최신의 귀납적 논리 프로그래밍 프레임워크를 사용하여 보상 기계(RMs)를 학습합니다. 이는 Bayesian posterior 방식으로 신뢰도를 평가하여 불일치를 극복합니다. 또한, PROB-IRM 알고리즘은 RL과 RM 학습 프로세스를 교차하여 새롭게 학습된 RM을 즉시 활용할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, PROB-IRM은 노이즈가 있는 트레이스에서 RMs을 성공적으로 학습하고 이를 활용하여 RL 에이전트를 훈련하여 작업을 효율적으로 해결함을 증명하였습니다. PROB-IRM을 통해 학습된 에이전트는 전문가가 수작업으로 생성한 RM과 유사한 성능을 보였습니다.



### Enhancing Analogical Reasoning in the Abstraction and Reasoning Corpus via Model-Based RL (https://arxiv.org/abs/2408.14855)
Comments:
          Accepted to IJCAI 2024 IARML Workshop

- **What's New**: 이 연구는 모델 기반 강화 학습(Model-Based RL)이 유사한 작업에서의 유추 추론(Analogical Reasoning) 능력을 획득하는 데 효율적이라는 가설을 세우고 이를 검증하기 위해 DreamerV3와 Proximal Policy Optimization(PPO)를 비교했습니다.

- **Technical Details**: 이 연구에서는 모델 기반 RL과 모델 프리 RL의 성능을 Abstraction and Reasoning Corpus(ARC)에서 비교했습니다. 모델 기반 RL은 환경의 내부 모델을 바탕으로 하여 예측 가능한 시나리오를 통해 의사 결정을 개선할 수 있으며, 이는 유추 추론에 도움이 됩니다. DreamerV3는 입력 데이터에서 중요한 특징을 추출하고 이들을 잠재 표현(Latent Representation)으로 변환하여 환경 동역학의 예측 모델을 학습합니다.

- **Performance Highlights**: 모델 기반 RL인 DreamerV3는 단일 작업에서 더 효율적일 뿐만 아니라, 유사한 작업에 대한 추론에서도 뛰어난 성능을 보였습니다. 실험 결과, 모델 프리 RL인 PPO보다 훨씬 높은 유추 추론 능력을 발휘하였습니다.



### CL4KGE: A Curriculum Learning Method for Knowledge Graph Embedding (https://arxiv.org/abs/2408.14840)
Comments:
          16 pages, 3 figures

- **What's New**: 이 논문에서는 Knowledge Graph Embedding (KGE) 훈련의 난이도를 평가하기 위한 Z-counts라는 새로운 메트릭을 제안하고, 이를 기반으로 하는 CL4KGE라는 효율적인 Curriculum Learning 방법론을 소개합니다.

- **Technical Details**: Z-counts는 KGs의 각 트리플 (head entity, relation, tail entity)의 훈련 난이도를 측정하는 메트릭입니다. CL4KGE는 이 메트릭을 활용하여 트리플의 난이도를 기준으로 훈련을 순차적으로 진행할 수 있게 합니다. 이 방법은 다양한 KGE 모델에 플러그인 형태로 적용 가능하며, 대부분의 KGs에 대해 적응성이 뛰어납니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋에서 수행한 실험을 통해 CL4KGE가 기존 방법에 비해 뛰어난 성능 향상을 보인 것을 확인했습니다. 특히 링크 예측(link prediction) 및 트리플 분류(triple classification) 작업에서 효과가 입증되었습니다.



### From Rule-Based Models to Deep Learning Transformers Architectures for Natural Language Processing and Sign Language Translation Systems: Survey, Taxonomy and Performance Evaluation (https://arxiv.org/abs/2408.14825)
- **What's New**: 이번 논문은 전 세계적으로 증가하는 청각 장애인과 난청 인구를 위해 수화 기계 번역 시스템의 필요성을 강조하고 있습니다. 기존의 연구가 수화의 동적이고 연속적인 특성을 충분히 고려하지 못한 점을 지적하며, 새로운 접근 방식을 제공합니다.

- **Technical Details**: 이 논문은 수화 기계 번역 알고리즘의 시간적 진화를 회고적으로 분석하고, 언어 번역에서 가장 많이 사용되는 Transformers 아키텍처의 분류법을 제공합니다. 또한, 정확한 deep learning 알고리즘에 기반한 실시간 Quality-of-Service 수화 기계 번역 시스템의 요구 사항을 제시합니다.

- **Performance Highlights**: 사실상, 수화 기계 번역의 발전을 위한 미래 연구 방향도 제안됩니다. 이 시스템은 언어의 연속성과 동적 특성을 효과적으로 처리할 수 있는 가능성을 보여줍니다.



### Brain-inspired Artificial Intelligence: A Comprehensive Review (https://arxiv.org/abs/2408.14811)
Comments:
          35 pages, 4 figures

- **What's New**: 본 리뷰 논문은 현대 인공지능(AI) 모델의 설계 원칙에 대한 포괄적인 검토를 제공하고, 뇌에서 영감을 받은 인공지능(BIAI)의 잠재력과 한계를 논의합니다. BIAI 접근법을 물리적 구조와 인간 행동으로 나누어 분류하며, 이 모델들이 실제에서 어떻게 적용되는지 사례를 통해 설명합니다.

- **Technical Details**: BIAI는 생물학적 연결과 과정을 모방함으로써 더 인간 같은 행동을 달성하기 위한 AI 시스템을 지칭합니다. BIAI 모델은 물리적 구조(PS) 영감을 받은 모델과 인간 행동(HB) 영감을 받은 모델로 나눌 수 있으며, 프로세스와 기능에서 높은 생물학적 타당성을 지닌 AI 시스템 설계를 목표로 합니다. 주요 기술적 요소로는 Multi-layer Perceptron(MLP), Artificial Neural Networks(ANNs), Spiking Neural Networks(SNNs), attention mechanism, transfer learning, reinforcement learning 등이 있습니다.

- **Performance Highlights**: BIAI는 적응성, 일반화, 해석 가능성 면에서 전통적인 AI 접근법을 초월할 가능성을 지니고 있으며, 의료 진단, 자율주행차, 챗봇, 사이버 위협 탐지 등 여러 실제 문제 해결에 기여할 수 있습니다. 또한, BIAI는 통신과 정보를 처리할 때 에너지 효율성을 높이고 로봇 시스템의 모션 및 조작 능력을 개선할 수 있는 잠재력을 보여줍니다. 그러나 BIAI 모델의 복잡성 및 불투명성 문제는 AI 시스템의 신뢰성과 해석 가능성에 중요한 도전 과제가 됩니다.



### Optimizing Structured Data Processing through Robotic Process Automation (https://arxiv.org/abs/2408.14791)
Comments:
          This manuscript has been accepted for publication in the journal Revue d'Intelligence Artificielle

- **What's New**: 로봇 프로세스 자동화(RPA)가 데이터 추출의 혁신적인 기술로 부상하면서 문서 처리 방식이 변화하고 있습니다. 본 연구는 RPA의 구조적 데이터 추출을 조사하고 인간의 수동 프로세스와 비교해 이점을 평가합니다.

- **Technical Details**: RPA 소프트웨어 봇과 사람이 수행하는 작업을 비교하여 청구서에서 데이터를 추출하는 데 있어 효율성과 정확도를 측정하였습니다. 네 가지 시나리오에서 청구서 수에 따라 시간을 기준으로 한 효율성 및 수동과 RPA 프로세스의 오류율을 통한 정확도를 분석했습니다.

- **Performance Highlights**: RPA에 의해 달성된 효율성 향상은 주목할 만합니다. 모든 경우에서 봇은 수동 노력에 비해 작업을 훨씬 적은 시간 내에 완료하였고, 모든 경우에서 완벽한 정확도를 달성하여 오류의 위험을 줄이고 프로세스의 신뢰성을 향상시켰습니다.



### Artificial Intelligence in Landscape Architecture: A Survey (https://arxiv.org/abs/2408.14700)
Comments:
          Preprint. 3 figures, 2 tables

- **What's New**: 이 논문은 조경 건축(LA) 분야에서 인공지능(AI) 기술의 응용을 종합적으로 검토하여 이 분야가 직면하고 있는 문제를 해결할 수 있는 방안을 제시하고 있습니다. 설계, 계획, 관리의 다양한 측면에서 AI가 제공하는 잠재적 이점에 대해 소개하고, AI와 인간 전문 지식의 결합 필요성을 강조합니다.

- **Technical Details**: AI는 프로젝트의 초기 조건을 과학적으로 식별하고 분석하며, 설계 논리를 구축하고, 설계 결과를 평가하는 데 중요한 역할을 합니다. 주요 기술로는 Generative Adversarial Network (GAN)와 Machine Learning (ML), 자연어 처리(NLP), 컴퓨터 비전(CV) 기술이 사용되고, 데이터 분석을 통해 실제 설계 결과를 도출합니다.

- **Performance Highlights**: AI는 조경 설계의 효율성을 향상시키고, 스마트 관개 및 유지 관리 시스템을 통해 자원을 절약하며, 방문객에게 개인 맞춤형 경험을 제공합니다. AI의 도움으로 생태계 시뮬레이션 및 평가가 가능해져, 디자인 기획과 관리의 최적화를 도모합니다.



### KGPrune: a Web Application to Extract Subgraphs of Interest from Wikidata with Analogical Pruning (https://arxiv.org/abs/2408.14658)
Comments:
          Accepted as a demo paper at ECAI 2024

- **What's New**: KGPrune은 사용자가 관심 있는 시드 엔티티와 탐색할 속성을 제공하면, 이웃 서브그래프를 추출할 수 있는 웹 애플리케이션입니다. 이 애플리케이션은 수학적 추론을 기반으로 하는 경량의 가지치기 알고리즘을 채택하여 관련 이웃만 유지하고 관련 없는 이웃은 제거합니다.

- **Technical Details**: KGPrune은 Wikidata를 기반으로 하여 설계되었으며, 사용자는 관심 있는 시드 엔티티와 속성의 QID 및 PID를 포함하는 두 개의 CSV 파일을 업로드해야 합니다. 알고리즘은 주어진 시드 엔티티의 이웃을 탐색하고, 각각의 이웃에 대해 유사성 패턴을 바탕으로 유지 또는 삭제 결정을 내립니다. 이를 통해 사용자는 관련 정보만 포함된 서브그래프를 얻을 수 있습니다.

- **Performance Highlights**: KGPrune은 다양한 도메인에서의 사용을 지원하며, 특히 기업 지식 그래프를 구축하거나 약탈된 예술 작품에 대한 지식을 추출하는 데 유용함을 보여줍니다. 이 애플리케이션은 사용자에게 각 결과를 시각화하거나 JSON 또는 RDF 형식으로 다운로드할 수 있는 기능을 제공합니다.



### Emergent Language in Open-Ended Environments (https://arxiv.org/abs/2408.14649)
Comments:
          10 pages, 4 figures, 4 tables, preprint

- **What's New**: 본 논문에서는 보다 복잡한 다중 에이전트 시스템에서의 의사소통의 출현을 탐구하고, 언어 생성의 한계를 극복하기 위해 새로운 코퍼스 환경을 제시합니다. Multi-Agent Pong 및 Collectors라는 두 가지 협력 환경을 도입하여 에이전트 간의 토큰 기반 통신이 어떻게 발생하는지를 분석합니다.

- **Technical Details**: 이 연구에서는 에이전트들이 움직임과 의사소통을 통해 환경과 상호작용하는 개방형 다중 에이전트 환경을 다룹니다. saliency maps, perturbation, diagnostic classifiers와 같은 설명 가능한 AI(Explainable AI) 기법을 활용하여 에이전트의 언어 채널 사용을 추적하고 해석합니다.

- **Performance Highlights**: 발생하는 의사소통은 희소 형태를 띄며, 에이전트들이 조정 없이는 성공할 수 없는 상황에서만 의미 있는 메시지를 생성하고 수신한 메시지에 따라 행동한다는 것을 발견했습니다. 이러한 연구 결과는 MARL(다중 에이전트 강화 학습) 프레임워크가 언어의 발생을 촉진할 수 있는 방법에 대한 이해를 발전시키고 있으며, 향후 다중 에이전트 의사소통 시스템의 동적이고 적응적인 본질을 탐구하는 기초를 마련합니다.



### Effect of Adaptation Rate and Cost Display in a Human-AI Interaction Gam (https://arxiv.org/abs/2408.14640)
- **What's New**: 이번 연구에서는 AI의 적응 알고리즘이 인간 행동 예측에 미치는 영향을 조사하였고, 두 종류의 시각적 피드백 정보를 제공하여 Nash 균형과 Stackelberg 균형 간의 결과 변화를 관찰했습니다.

- **Technical Details**: AI는 서로 다른 적응 속도에서 gradient descent 알고리즘을 사용하여 동작을 조정하고, 인간 참가자는 비용 피드백을 통해 행동을 최적화합니다. 피드백은 현재 공동 행동 벡터의 비용 또는 해당 벡터의 지역 이웃에서의 비용으로 제공되었습니다.

- **Performance Highlights**: AI의 느린 적응 속도는 결과를 Nash 균형으로, 빠른 속도는 인간 주도의 Stackelberg 균형으로 전환시키는 효과가 있었습니다. 또한, 지역화된 비용 정보의 추가 제공은 Nash 균형으로의 결과 전환을 촉진했습니다.



### On Centralized Critics in Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2408.14597)
- **What's New**: 이번 논문은 Multi-Agent Reinforcement Learning (MARL)에서의 Centralized Training과 Decentralized Execution 방식의 분석을 제공하며, 특히 중앙 집중식 비평가(critic)를 사용하는 것이 이론적 및 실증적으로 얼마나 효과적인지를 중점적으로 다루고 있습니다.

- **Technical Details**: 저자들은 상태 기반 비평가(state-based critics)와 역사 기반 비평가(history-based critics)의 차이점을 분석하고, 후자의 사용이 통계적 편향(bias) 및 분산(variance)에 미치는 영향을 논의합니다. 기존의 생각과는 달리, 중앙 집중식 비평가가 항상 유리하지 않다는 점을 지적하며, 상태 값(state values)의 사용이 해로울 수 있음을 보입니다.

- **Performance Highlights**: 실험 결과, 부분 관찰 환경(partially observable environments)에서의 표현 학습(representation learning)의 어려움과 같은 실용적 문제들이 발생한다는 점이 강조됩니다. 이 논문은 MARL 분야에서 중앙 집중식 비평가의 이론적 문제점이 종종 간과된 이유를 명확히 보여줍니다.



### EVINCE: Optimizing Adversarial LLM Dialogues via Conditional Statistics and Information Theory (https://arxiv.org/abs/2408.14575)
Comments:
          19 pages, 7 figures, four tables

- **What's New**: 이 논문은 EVINCE(Entropy and Variation IN Conditional Exchanges)라는 새로운 대화 프레임워크를 소개하며, 이는 인공지능 일반 지능(AGI) 발전에 기여합니다. EVINCE는 적대적 토론을 활용하고, 새로운 이중 엔트로피 이론을 통해 다양한 관점을 탐색하며 강력한 사전 지식 활용을 조화롭게 통합하여 대규모 언어 모델(LLMs)의 예측 정확도와 안정성을 향상시킵니다.

- **Technical Details**: EVINCE 프레임워크는 세 가지 주요 기둥을 기반으로 합니다: 1) Inclusiveness Exploration은 조건부 통계를 활용하여 LLM이 다양한 관점을 자유롭게 탐색할 수 있도록 돕고, 2) Information Flow Dynamics는 정보 흐름의 다양성, 신선함, 상호 설득력을 정량화하여 대화 품질을 향상시키며, 3) Reasoning Quality and Coherence는 다중 에이전트의 추론 논리성과 일관성을 평가합니다.

- **Performance Highlights**: EVINCE는 의료 분야에서 진단 정확도를 향상시키는 등의 성과를 보여주며, 이는 LLM 간의 협업을 통해 얻어진 결과입니다. 이 연구는 AGI 발전을 위한 이론적 기초와 실증적 검증을 제공하며, 다중 에이전트 대화를 통한 성능 개선 가능성을 시사합니다.



### Unveiling the Statistical Foundations of Chain-of-Thought Prompting Methods (https://arxiv.org/abs/2408.14511)
Comments:
          150 pages, 18 figures, 3 tables

- **What's New**: 이 연구에서는 Chain-of-Thought (CoT) prompting와 그 변형을 통계적 추정 관점에서 분석하여 샘플 복잡성(sample complexity)에 대한 포괄적인 특성을 제공합니다.

- **Technical Details**: 이 논문에서는 다단계(latent variable) 모델을 도입하여 추론 과정을 캡슐화하며, 잠재 변수(latent variable)가 작업 정보를 인코딩합니다. CoT prompting에 의해 형성된 추정기는 대규모의 사전 훈련(pretraining) 데이터셋에서 베이지안(Bayesian) 추정기와 동등함을 보여줍니다. 또한, CoT 추정기의 통계적 오차는 두 가지 주요 구성 요소로 분해될 수 있음을 증명합니다: (i) CoT 프롬프트를 사용하여 진짜 작업을 추론함으로써 발생하는 프롬프트 오류(prompting error)와 (ii) 사전 훈련된 LLM의 통계적 오류입니다.

- **Performance Highlights**: 실험을 통해 다단계 추론 문제에서 타겟 분포(target distribution)를 근사하는 변환기 모델(transformer model)을 구성하고, 변환기 블록의 수가 증가함에 따라 오류가 기하급수적으로 감소함을 확인했습니다. CoT의 다른 변형, Self-Consistent CoT, Tree-of-Thought, Selection-Inference에 대한 분석도 포함되어 있으며, 이 방법들의 효율성을 넓은 관점에서 다룹니다.



### Artificial intelligence for science: The easy and hard problems (https://arxiv.org/abs/2408.14508)
Comments:
          16 pages, 3 boxes, 4 figures

- **What's New**: 최근 인공지능의 발전으로 인해 다양한 과학적 발견이 이뤄지고 있지만, 이러한 접근법은 과학 연구의 두 가지 측면 중 하나인 '쉬운 문제'에 해당한다고한다. 본 논문은 문제 정의의 중요성과 '어려운 문제'를 해결하기 위한 필요성을 강조하고 있다. AI가 과학적 발견에서 진정으로 중요한 '어려운 문제'를 해결하기 위해서는 과학자들이 문제를 정의하는 방식에 대한 인식을 고찰해야 하며, 이를 통해 새로운 컴퓨팅 에이전트를 설계할 수 있다.

- **Technical Details**: 본 연구는 과학 탐구에서 '쉬운 문제'와 '어려운 문제'에 대해 다루며, 특히 과학자들이 문제를 정의하는 과정에서 발생하는 개념적 중재를 강조한다. 쉬운 문제는 특정 기능을 최적화하는 것으로, 대규모 데이터 세트와 접목하여 AI 최적화 도구가 활용되고 있다. 반면, 어려운 문제는 기존의 알고리즘으로 접근하기 어려운 영역으로, 개념적 혁신을 필요로 하며 문제의 정의가 중요하다는 것을 보여준다.

- **Performance Highlights**: AI는 다양한 과학 문제에 대해 놀라운 성과를 거두었으나, 과학적 발견과 조사에 있어 이 최적화는 불완전한 모델에 의존하고 있다. 연구는 AI가 발견 문제를 자동으로 생성할 수 있는 능력이 부족하다는 것을 지적하며, 이는 문제 정의 및 목표 함수 설정의 비법칙성에서 기인한다고한다. 궁극적으로 과학자들이 어떻게 문제를 정의하는지를 이해함으로써, AI 시스템이 '어려운 문제'를 해결할 수 있는 새로운 길을 모색할 수 있을 것으로 기대된다.



### Active learning of digenic functions with boolean matrix logic programming (https://arxiv.org/abs/2408.14487)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2405.06724

- **What's New**: 이 연구는 Boolean Matrix Logic Programming (BMLP)이라는 새로운 접근법을 제시합니다. 이 방법은 boolean 매트릭스를 활용하여 대규모 로직 프로그램을 평가하고, 감정 실험을 통해 유용한 유전 가설 영역을 탐색할 수 있도록 합니다.

- **Technical Details**: BMLP 접근법은 genome-scale metabolic network model (GEM)인 iML1515에 적용되어, E. coli의 예를 통해 유전자 쌍 간의 상호작용을 학습하는데 필요한 훈련 샘플 수를 줄입니다. 이 과정은 datalog 논리 프로그램을 사용해 설명 가능한 방식으로 수행됩니다.

- **Performance Highlights**: BMLP_{active}는 무작위 실험보다 적은 수의 훈련 예제로 성공적으로 유전자의 상호작용을 학습할 수 있으며, 대사 모델의 빠른 최적화를 가능하게 하고 미생물 공학을 위한 자율 실험실(self-driving lab)을 현실화하는데 기여합니다.



### Agentic Retrieval-Augmented Generation for Time Series Analysis (https://arxiv.org/abs/2408.14484)
Comments:
          Paper was accepted for Undergraduate Consortium at ACM KDD, 2024. Please find the link: this https URL

- **What's New**: 이 논문은 복잡한 시공간 의존성과 분포 변화 문제를 해결하기 위해 에이전트 기반의 Retrieval-Augmented Generation(RAG) 프레임워크를 제안합니다. 이 프레임워크는 마스터 에이전트가 여러 전문 서브 에이전트를 조정하여 사용자 요청을 처리하는 방식으로, 과거 패턴에 대한 정보를 활용하여 예측을 개선합니다.

- **Technical Details**: 제안된 Agentic RAG 프레임워크는 계층적 다중 에이전트 구조를 채택합니다. 최상위 마스터 에이전트가 사용자 요청을 분석하고, 관련 서브 에이전트에게 작업을 위임합니다. 각 서브 에이전트는 특화된 시계열 과제를 위해 조정된 소형 언어 모델(SLMs)을 사용하고, 공유된 프롬프트 풀에서 관련 정보를 검색하여 예측 능력을 향상시킵니다. 이 과정에서 과거의 패턴을 기반으로 시계열 데이터의 복잡성을 다룰 수 있도록 설계되었습니다.

- **Performance Highlights**: 제안된 방안은 단일 고급 모델보다 유연성과 정확성을 제공하며, 여러 시계열 분석 작업에서 최첨단 성능을 기록하였습니다. 특히, 다차원 데이터나 이차원 데이터와 같은 다양한 데이터셋에서도 성능이 뛰어나며, 이는 기존의 작업 특정 맞춤형 방법보다 더 효과적입니다.



### The Mamba in the Llama: Distilling and Accelerating Hybrid Models (https://arxiv.org/abs/2408.15237)
Comments:
          Code is open-sourced at this https URL

- **What's New**: 이번 연구는 대규모 Transformer 모델을 linear RNN으로 변환할 수 있는 방법을 제시하며, 특히 Mamba 아키텍처를 통해 이를 구현합니다. 연구팀은 attention 레이어의 선형 투영 가중치를 재사용하여 대규모 Transformer를 distillation(증류) 할 수 있음을 보여주었고, 이 hybrid 모델은 기존 Transformer와 유사한 성능을 유지하면서도 많은 attention 레이어를 제거함으로써 효율성을 개선했습니다.

- **Technical Details**: 이 연구의 핵심 기술적 기여는 다음과 같습니다. 1) attention 레이어에서 가중치를 재사용하여 큰 hybrid-linear RNN 모델을 효율적으로 생성할 수 있음. 2) 표준 LLM 파이프라인과 유사한 multistage distillation 접근 방식을 제안하여 perplexity와 downstream 평가에서 더 나은 성능을 보임. 3) 하드웨어에 최적화된 speculative sampling 알고리즘과 빠른 커널을 개발해 Mamba와 hybrid 아키텍처에서 300 tokens/second 이상의 처리 성능을 달성. 4) 대규모 오픈 챗 LLM을 distilling 하여 일반적인 챗 기준에서 teacher 모델과 유사한 성능을 보임.

- **Performance Highlights**: 최고 성능의 모델은 Llama3-8B-Instruct에서 증류된 것으로, GPT-4에 대해 AlpacaEval 2에서 29.61의 길이 제어 승률을 기록하였고, MT-Bench에서는 7.35의 성능을 기록했습니다. 이는 기존 instruction-tuned linear RNN 모델보다 뛰어난 성능입니다.



### Into the Unknown Unknowns: Engaged Human Learning through Participation in Language Model Agent Conversations (https://arxiv.org/abs/2408.15232)
- **What's New**: 본 논문은 사용자가 질문을 던지지 않고도 대화형 상호작용을 통해 정보를 발견할 수 있도록 하는 새로운 시스템, Collaborative STORM (Co-STORM)를 소개합니다.

- **Technical Details**: Co-STORM은 다수의 언어모델(LM) 에이전트 간의 대화를 관찰하고 방향을 제시할 수 있게 해주며, 사용자가 잘 알지 못하는 정보를 자연스럽게 발견할 수 있도록 돕습니다. 또한, Co-STORM은 발견된 정보를 동적인 마인드 맵(mind map)으로 정리하여 사용자가 쉽게 소통할 수 있도록 지원합니다.

- **Performance Highlights**: Co-STORM은 베이스라인(baseline) 방법들과 비교하여 담화 추적(discourse trace) 및 보고서(report) 품질에서 우수한 성능을 보였으며, 인간 평가에서 참가자의 70%가 Co-STORM을 검색 엔진보다 선호하고, 78%가 RAG 챗봇보다 Co-STORM을 선호한다고 응답했습니다.



### Fundus2Video: Cross-Modal Angiography Video Generation from Static Fundus Photography with Clinical Knowledge Guidanc (https://arxiv.org/abs/2408.15217)
Comments:
          The paper has been accepted by Medical Image Computing and Computer Assisted Intervention Society (MICCAI) 2024

- **What's New**: 이 연구에서는 Color Fundus (CF) 이미지를 기반으로 한 동적 Fundus Fluorescein Angiography (FFA) 비디오 생성 방법을 최초로 제안하고 있습니다. 기존의 방법들이 정적인 이미지 생성을 중심으로 한 반면, 본 연구에서는 프레임별로 부드러운 FFA 비디오를 생성하는 autoregressive GAN 아키텍처인 Fundus2Video를 도입하였습니다.

- **Technical Details**: Fundus2Video 모델은 이미지 간의 종속성을 포착하기 위해 autoregressive GAN 아키텍처를 사용하며, CF 이미지를 입력으로 받아 연속적인 FFA 비디오 프레임을 생성합니다. 이 과정에서 지식 마스크(knowledge mask)를 활용하여 병변 변화가 중요한 영역에 집중할 수 있도록 설계하였습니다. 또한, pixel misalignment 문제를 해결하기 위해 mask-enhanced patchNCE loss를 적용하였습니다.

- **Performance Highlights**: 이 방법은 기존의 비디오 생성 기법들에 비해 FVD(Fréchet Video Distance) 1503.21 및 PSNR(Peak Signal-to-Noise Ratio) 11.81라는 최상의 성능을 기록하였으며, 안과 의사에 의한 인간 평가에서도 높은 생성 품질을 인정받았습니다. 특히, 제안된 지식 마스크는 감독된 병변 세분화 마스크에 비해 더 나은 성능을 보여주어 전통적인 FFA에 대한 비침습적인 대안을 제공합니다.



### Can Unconfident LLM Annotations Be Used for Confident Conclusions? (https://arxiv.org/abs/2408.15204)
- **What's New**: 해당 연구는 대형 언어 모델(LLM)의 주석을 활용하여 인간 주석의 수를 25% 이상 줄일 수 있는 Confidence-Driven Inference (CDI) 방식을 제안합니다. 이 방법은 LLM의 주석과 신뢰도 지표를 조합하여 수집할 인간 주석을 전략적으로 선택하는 방법론을 다룹니다.

- **Technical Details**: Confidence-Driven Inference는 LLM 주석과 LLM의 신뢰 점수를 활용하여 인간 주석 수를 최소화하며, 통계적 추정을 정확하게 수행합니다. 이 방법은 세 가지 사회 과학 연구 설정, 즉 언어의 정중함, 입장, 그리고 편향을 포함한 다양한 문제에 적용됩니다. 이는 LLM의 정확성을 반영하는 신뢰 점수를 이용하여 진행됩니다.

- **Performance Highlights**: 이 연구는 언어 정중함, 입장, 미디어 편향 등 세 가지 설정에서 통계적 추정 과제를 수행했으며, LLM 주석을 통합한 Human Annotations의 효율성 및 커버리지를 유지하면서 데이터 샘플의 효과적인 크기를 증가시켰습니다.



### Automatic 8-tissue Segmentation for 6-month Infant Brains (https://arxiv.org/abs/2408.15198)
Comments:
          11 pages, 4 figures, to be published in MICCAI PIPPI workshop

- **What's New**: 이 연구에서는 6개월 아기 뇌를 위한 최초의 8개 조직 세분화(Tissue Segmentation) 파이프라인을 제안하였습니다. 이는 도메인 적응(Domain Adaptation) 기술을 활용하여 신생아 이미지를 포함한 장기 데이터를 이용합니다.

- **Technical Details**: 이 파이프라인은 원시 6개월 이미지(raw 6-month images)를 입력으로 받고 8개 조직 세분화를 출력으로 생성하는 엔드 투 엔드(end-to-end) 세분화 파이프라인입니다. 세분화되는 조직은 백질(White Matter, WM), 회백질(Gray Matter, GM), 뇌척수액(Cerebrospinal Fluid, CSF), 뇌실(Ventricles), 소뇌(Cerebellum), 기저핵(Basal Ganglia), 뇌간(Brainstem), 해마(Hippocampus) 및 편도체(Amygdala)를 포함합니다. Cycle-Consistent Generative Adversarial Network(CycleGAN)와 Attention U-Net을 사용하여 신생아 이미지와 6개월 이미지 간의 이미지 대비 변환을 수행하였고, 합성된 6개월 이미지에 대해 조직 세분화를 수행했습니다.

- **Performance Highlights**: 실제 6개월 이미지를 사용한 평가에서 DICE 점수는 0.92, HD95는 1.6, ASSD는 0.42를 달성했습니다.



### PoseWatch: A Transformer-based Architecture for Human-centric Video Anomaly Detection Using Spatio-temporal Pose Tokenization (https://arxiv.org/abs/2408.15185)
- **What's New**: PoseWatch라는 새로운 변형기 기반 아키텍처를 소개하며, 인체 중심의 포즈 기반 비디오 이상 탐지(VAD)를 위한 혁신적인 방법론을 제시합니다.

- **Technical Details**: PoseWatch는 Spatio-Temporal Pose and Relative Pose (ST-PRP) 토크나이제이션 방식을 채택하여 시간을 통한 인간 동작 표현을 개선하고, Unified Encoder Twin Decoders (UETD) 구조를 통해 비디오 데이터에서 이상 행동 탐지를 향상시킵니다.

- **Performance Highlights**: PoseWatch는 여러 벤치마크 데이터셋에서 기존 방법들을 일관되게 초월하며, 80.67%의 평균 수신자 조작 특성 곡선(ROC AUC) 점수를 달성, 평균 동등 오류율(EER)은 0.27로 설정하여 새로운 성과를 기록합니다.



### Evaluating the Energy Consumption of Machine Learning: Systematic Literature Review and Experiments (https://arxiv.org/abs/2408.15128)
Comments:
          52 pages,

- **What's New**: 기계 학습(Machine Learning) 에너지 소비 평가를 위한 다양한 도구와 방법에 대한 체계적인 문헌 검토 및 실험 프로토콜 개발이 이루어졌습니다. 이는 기계 학습 및 일반 소프트웨어의 에너지 소비를 평가할 수 있는 도구들을 총망라하고 비교하는 작업입니다.

- **Technical Details**: 본 연구는 Systematic Literature Review (SLR) 방식을 통해 에너지 소비 평가 도구 및 방법에 대한 폭넓은 리뷰를 진행했습니다. 에너지를 직접 측정하는 방식, 공급업체 특정 센서 사용, 분석적 추정 모델 기반 등 다양한 접근 방식이 비교되었습니다. 실험은 서로 다른 ML 작업(비전, 언어)에 대해 진행되었으며, Python 스크립트가 GitHub에 공개되어 있습니다.

- **Performance Highlights**: 이 논문은 기계 학습의 에너지 소비 평가를 위한 도구 및 방법에 대한 포괄적인 가이드를 제공하며, 에너지 소비 최적화와 모니터링을 위한 열린 소스(repository)도 제공합니다. 다양한 ML 작업에 대한 정성적 및 정량적 비교가 이루어져, 푸른 녹색 ICT 및 AI의 필요성에 대한 인식을 높이고 기계 학습 알고리즘의 에너지 효율성을 촉진할 기초 자료를 제시합니다.



### Urdu Digital Text Word Optical Character Recognition Using Permuted Auto Regressive Sequence Modeling (https://arxiv.org/abs/2408.15119)
- **What's New**: 이번 연구는 디지털 우르두( اردو ) 텍스트 인식을 위한 혁신적인 단어 수준의 Optical Character Recognition (OCR) 모델을 소개합니다.

- **Technical Details**: 이 모델은 transformer 기반 아키텍처와 attention 메커니즘을 활용하여 약 160,000개의 우르두 텍스트 이미지로 훈련되었으며, 문자 오류율(Character Error Rate, CER)은 0.178로 보고되었습니다. 이 모델의 독특한 아키텍처는 permuted autoregressive sequence (PARSeq) 모델을 포함하여, 양방향 맥락 정보를 활용한 맥락 인식 추론 및 반복적 개선을 가능하게 합니다.

- **Performance Highlights**: 다양한 우르두 텍스트 스타일, 글꼴 및 변형을 처리할 수 있는 능력 덕분에 실제 응용에서의 적합성이 향상되었습니다. 그러나 블러 처리된 이미지, 비수평 방향, 패턴이나 선, 다른 텍스트의 겹침 처리에는 한계가 있으며, 이로 인해 때때로 최적의 성능을 발휘하지 못할 수 있습니다.



### Few-Shot Unsupervised Implicit Neural Shape Representation Learning with Spatial Adversaries (https://arxiv.org/abs/2408.15114)
Comments:
          ICML 2024

- **What's New**: 이번 연구에서는 Sparse 3D point clouds에서 Implicit Neural Representations를 이용하여 Neural Signed Distance Functions (SDF)를 개선하기 위한 새로운 정규화 기법을 도입하였습니다. 과거의 방법들은 일반적으로 Smoothness Priors를 활용했으나, 본 연구는 Adversarial Samples를 통해 SDF 학습을 향상시키는 방법을 제안하였습니다.

- **Technical Details**: 이 연구는 Adversarial Training 기법을 활용하여 원래의 질의 포인트 주변에서 다양성을 갖는 Adversarial Samples를 생성합니다. 그리고 이러한 샘플들은 SDF의 최소화 과정에 통합하여 학습을 정규화하고 오버피팅을 방지하는 데 활용됩니다. 본 연구에서는 네트워크가 Sparse 한 입력 데이터에서 발생할 수 있는 오류를 줄이기 위해 Spatial Gradient, Lipschitz Regularization 등의 기존 기법들을 고려하였습니다.

- **Performance Highlights**: 다양한 실험을 통해 제안된 방법이 기존 Baseline 및 최신 기술 대비 우수한 성능을 보이며, 특히 형태 예측이 가장 어려운 세부 구조와 신체 말단 부분에서 성능이 크게 증가한 것을 확인하였습니다. 제안된 방법은 Dense Reconstruction 환경에서도 유용하며, 평가 과정에서의 안정성을 높여줍니다.



### MTMamba++: Enhancing Multi-Task Dense Scene Understanding via Mamba-Based Decoders (https://arxiv.org/abs/2408.15101)
Comments:
          arXiv admin note: text overlap with arXiv:2407.02228

- **What's New**: 본 논문에서는 MTMamba++라는 새로운 multi-task dense scene understanding 아키텍처를 제안합니다. 이 아키텍처는 Mamba 기반 디코더를 특징으로 하며, RMS(기본 행동 내역)에 기반한 self-task Mamba SCM과 cross-task Mamba CTM 블록 두 가지 핵심 블록을 포함합니다.

- **Technical Details**: MTMamba++의 디코더는 ECR(expand, concatenate, reduce), STM(self-task Mamba), CTM(cross-task Mamba) 블록으로 구성됩니다. ECR 블록은 태스크 특징을 업스케일링하고 인코더의 고수준 특징과 융합하는 역할을 합니다. STM 블록은 SSM(state space models) 메커니즘을 활용하여 각 태스크의 글로벌 컨텍스트 정보를 효과적으로 캡처하고, CTM 블록은 다양한 태스크 간의 지식 교환을 통해 태스크의 특징을 강화합니다. CTM 블록에는 F-CTM(feature level)와 S-CTM(semantic level)이라는 두 가지 변형이 포함되어 있어, 동적 크로스-태스크 표현과 태스크 간의 관계를 모델링합니다.

- **Performance Highlights**: MTMamba++는 NYUDv2, PASCAL-Context, Cityscapes 데이터셋에서 CNN 기반 및 Transformer 기반 방법들보다 월등한 성능을 보였습니다. 정량적 결과는 MTMamba++의 다중 태스크 밀집 예측 성능이 이전 방법들보다 뛰어난 것을 보여주며, 정성적 연구에서는 보다 우수한 시각적 결과와 세밀한 정확성을 생성하는 것으로 확인되었습니다.



### No Regrets: Investigating and Improving Regret Approximations for Curriculum Discovery (https://arxiv.org/abs/2408.15099)
- **What's New**: 이 논문은 Unsupervised Environment Design (UED) 방식이 실제 로봇 문제에 Inspired된 새로운 환경에 적용될 때의 강건성을 살펴봅니다. 연구 결과, 현재의 UED 방법들이 Domain Randomisation (DR)의 단순한 기본선조차 초과하지 못하거나, 과도한 하이퍼파라미터 튜닝을 요구한다는 사실을 발견했습니다.

- **Technical Details**: 이 연구에서는 UED 방법들의 내재된 채점 함수들이 '학습 가능성'을 직관적으로 예측하지 못한다는 점을 분석하고, 따라서 학습 가능한 환경 설정(에이전트가 때때로 해결하지만 항상 그렇지 않은 설정)을 찾아 이를 직접적으로 학습하는 간단한 방법을 제안합니다. 이 방식은 Sampling For Learnability (SFL)이라 불리며, 여러 도전적인 환경에서 UED 방법들과 DR을 초월하는 성능을 보여주었습니다.

- **Performance Highlights**: 새로운 평가 프로토콜을 도입하여, 에이전트의 성능을 가장 나쁜 환경에서 평가함으로써 강건성을 측정합니다. 이러한 방법을 통해 SFL 방식이 현재의 최첨단 UED 접근 방식보다 매우 우수하다는 것을 보여주었으며, 이를 통해 다양한 환경에서 안정성과 일반화 성능의 향상을 입증했습니다.



### Post-processing fairness with minimal changes (https://arxiv.org/abs/2408.15096)
- **What's New**: 본 문서에서는 모델에 독립적이며 시험 시 민감한 특성을 요구하지 않는 새로운 사후 처리 알고리즘을 소개합니다. 이 알고리즘은 편향된 예측과 수정된 예측 간의 최소 변경을 적용하도록 설계되었습니다.

- **Technical Details**: 우리는 사후 처리 작업을 새로운 지도 학습 문제로 정의하고, 편향된 모델의 예측을 재조정하기 위해 비율 기반 모델 디바이징(RBMD) 접근 방식을 제안합니다. 이 방법은 검은 상자 분류기(black-box classifier)의 로짓(logit) 값에 곱셈 인자를 적용합니다.

- **Performance Highlights**: RBMD는 기존의 네 가지 디바이징 알고리즘과 비교하여, 공정성(fairness) 및 정확성(accuracy) 점수에서 경쟁력 있는 결과를 달성하며, 수정된 예측의 수를 최소화합니다.



### BaichuanSEED: Sharing the Potential of ExtensivE Data Collection and Deduplication by Introducing a Competitive Large Language Model Baselin (https://arxiv.org/abs/2408.15079)
Comments:
          19 pages, 6 figures

- **What's New**: BaichuanSEED 모델은 자체 데이터 처리 파이프라인을 사용하여 7B 파라미터를 갖춘 LLM(대형 언어 모델)로 훈련되었습니다. 이 모델은 공개 소스를 바탕으로 하며, 특정 하위 작업과의 최적화를 배제하고도 경쟁력 있는 성능을 발휘합니다.

- **Technical Details**: BaichuanSEED는 3T의 이중 언어 토큰을 사용하여 훈련되었으며, Transformer 아키텍처를 기반으로 되어 있습니다. 32개의 층과 32개의 attention heads를 가지며, hidden dimension은 4096, feed-forward layer의 크기는 11008입니다.

- **Performance Highlights**: 실험 결과, BaichuanSEED는 Llama3와 Qwen-1.5와 같은 상업용 모델과 비교하여 동등한 성능을 보여주며, 특히 수학 및 코딩 분야에서 추가 개선의 여지가 있습니다.



### MMASD+: A Novel Dataset for Privacy-Preserving Behavior Analysis of Children with Autism Spectrum Disorder (https://arxiv.org/abs/2408.15077)
- **What's New**: 이번 연구에서는 자폐 스펙트럼 장애(ASD)에 대한 새로운 치료적 접근으로, MMASD+라는 모델을 소개합니다. 이 모델은 다양한 데이터 모달리티(data modalities)를 활용하여 치료사와 아동을 구분하는 데 있어 중요한 장벽을 해결합니다.

- **Technical Details**: MMASD+는 3D-Skeleton, 3D Body Mesh, Optical Flow 데이터를 포함한 다양한 데이터 모달리티를 통합합니다. Yolov8 및 Deep SORT 알고리즘의 기능을 결합하여 데이터 간 비교 문제를 극복합니다. 또한, 11개의 행동 유형과 ASD 존재를 예측하기 위해 멀티모달 트랜스포머(Multimodal Transformer) 프레임워크를 제안합니다.

- **Performance Highlights**: 이 프레임워크는 행동 유형 예측에서 95.03%의 정확도, ASD 존재 예측에서 96.42%의 정확도를 달성하였으며, 단일 데이터 모달리티로 훈련된 모델과 비교하여 10% 이상의 성능 향상을 보여주었습니다.



### MiWaves Reinforcement Learning Algorithm (https://arxiv.org/abs/2408.15076)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2402.17739

- **What's New**: MiWaves은 Emerging Adults(18-25세) 사이의 대마초 사용을 감소시키기 위해 개인화된 개입 메시지 전달을 최적화하는 Reinforcement Learning(RL) 알고리즘입니다. 미주리주의 대마초 사용이 증가하고 있는 가운데, MiWaves는 데이터와 도메인 전문 지식을 활용하여 효과적인 메시지 전달을 도모합니다.

- **Technical Details**: MiWaves는 참가자의 context를 기반으로 하여 매일 두 번씩 개입 프롬프트를 결정하는 RL 알고리즘을 활용합니다. 상태(state), 행동(action), 보상(reward)을 정의하여 알고리즘 학습을 위한 기초를 마련하며, 온라인 RL 알고리즘으로서 참가자의 반응을 지속적으로 학습하여 최적화합니다.

- **Performance Highlights**: MiWaves는 2024년 3월부터 5월까지 122명의 참가자를 대상으로 임상 시험에 배포되어 효과성을 검증할 예정이며, 참가자의 애플리케이션 사용 참여를 최대화 목표로 합니다.



### Causal Rule Forest: Toward Interpretable and Precise Treatment Effect Estimation (https://arxiv.org/abs/2408.15055)
Comments:
          The 25th IEEE International Conference on Information Reuse and Integration for Data Science (IRI 2024)

- **What's New**: 이번 연구에서는 이질적 치료 효과(Heterogeneous Treatment Effects, HTE)와 조건부 평균 치료 효과(Conditional Average Treatment Effects, CATE)의 이해 및 추론을 위한 새로운 접근 방식인 Causal Rule Forest (CRF)를 제안했습니다. 기존의 방법들에 비해 해석 가능성과 예측 성능의 균형을 이루도록 설계되었습니다.

- **Technical Details**: CRF는 데이터에서 숨겨진 패턴을 학습하고 이를 해석 가능한 다단계 부울 규칙(Boolean rules)으로 변환하는 방법입니다. 이 방식은 다른 해석 가능한 인과 inference 모델들과 함께 사용되어 HTE와 CATE 추정에서 예측 오류를 줄이는 동시에, 치료가 더 효과적인 하위 그룹을 식별하는 데 있어 해석 가능성을 유지합니다.

- **Performance Highlights**: 실험 결과, CRF는 개인 맞춤형 중재 및 정책을 향상시키는 잠재력을 보여주고 있으며, 미래 연구에서 복잡한 인과 inference 도전 과제를 해결하는 데 있어 그 유용성이 더욱 확대될 것으로 기대됩니다.



### Evidence-Enhanced Triplet Generation Framework for Hallucination Alleviation in Generative Question Answering (https://arxiv.org/abs/2408.15037)
- **What's New**: 이 연구는 Generative Question Answering(GQA)에서 발생하는 hallucination 문제를 해결하기 위한 새로운 프레임워크인 EATQA(Evidence-enhanced Triplet Generation Framework)를 제안합니다. 이 프레임워크는 모델이 (질문, 증거, 답변) 삼중항의 모든 조합을 예측하도록 유도하여 그리고 이들 간의 논리적 관계를 이해하도록 합니다.

- **Technical Details**: EATQA는 질문(Q), 증거(E), 답변(A) 세 가지 부분의 예측을 포함하며, 각 두 쌍의 정보를 바탕으로 나머지 하나를 생성합니다. 프레임워크 내에서 분포 간격을 줄여서 증거로부터 지식을 증류하고, 이는 모델이 질의, 증거 및 답변 간의 논리적 관계를 학습하게 합니다.

- **Performance Highlights**: 실험 결과, EATQA는 MultiRC 및 QASPER 두 개의 GQA 벤치마크 데이터셋에서 기존의 LLM 기반 방법들과 hallucination 완화 접근방식에 비해 뛰어난 성능을 보였습니다. 이 방법은 내부 지식을 유지하면서도 hallucination을 완화하고 신뢰할 수 있는 답변 생성을 가능하게 합니다.



### Mamba2MIL: State Space Duality Based Multiple Instance Learning for Computational Pathology (https://arxiv.org/abs/2408.15032)
- **What's New**: 본 논문에서는 기존의 MIL 접근 방식의 한계를 극복하기 위해 Mamba2MIL이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 주어진 WSI를 긴 시퀀스로 모델링할 수 있는 state space duality model (SSD)을 활용하며, 가중치 기반의 특성 선택(feature selection)을 통하여 더 다양한 특성을 융합할 수 있도록 지원합니다.

- **Technical Details**: Mamba2MIL은 WSI 패치의 긴 시퀀스를 효과적으로 모델링하기 위해 SSD을 사용하며, 다양한 크기의 WSI에 적합한 시퀀스 변환 방법을 도입합니다. 이 방법은 시퀀스 의존적 특성을 강화하고, 동시에 로컬 시퀀스 정보를 유지하는 데 중점을 두어 시퀀스 정보의 전체 활용도를 향상시킵니다.

- **Performance Highlights**: Mamba2MIL은 여러 데이터셋에서 실험 결과, 모든 성능 지표에서 개선을 보였습니다. 특히, NSCLC 데이터셋에서 AUC 0.9533, 정확도 0.8794를 기록하였고, BRACS 데이터셋에서는 AUC 0.7986, 정확도 0.4981을 달성했습니다.



### Sequence-aware Pre-training for Echocardiography Probe Guidanc (https://arxiv.org/abs/2408.15026)
Comments:
          Tech Report

- **What's New**: 이번 연구에서는 심장 초음파 프로브의 조정을 위한 새로운 방법인 sequence-aware self-supervised pre-training 방법을 제안합니다. 이 방법은 개별 환자의 심장 구조에 대한 개인화된 정보를 기반으로 합니다.

- **Technical Details**: 연구팀은 심장 초음파 영상과 조정 행동을 함께 고려하여 개인의 심장 구조를 학습하는 방식을 채택했습니다. 이를 위해 masked-out된 이미지와 조정 행동을 예측하는 방식으로 개인화된 2D 및 3D 심장 구조 특징을 학습합니다. 또한, 기존 스캔 데이터에 기반하여 개인의 심장 구조 정보를 모델링하는 시퀀스 모델링 접근 방식을 도입했습니다.

- **Performance Highlights**: 대규모 데이터셋에서 실험한 결과, 제안된 방법은 기존 최첨단 기술에 비해 내비게이션 오차를 15.90%에서 36.87%까지 줄이고, 회전 오차는 11.13%에서 20.77%까지 감소시켰습니다.



### Cross-subject Brain Functional Connectivity Analysis for Multi-task Cognitive State Evaluation (https://arxiv.org/abs/2408.15018)
- **What's New**: 이번 연구는 electroencephalography (EEG) 신호를 이용하여 여러 주체 간의 뇌 기능적 연결성을 평가하여 실시간 인지 상태를 모니터링하고 분석하는 새로운 접근 방식을 제공합니다. 특히, 가상 현실 기반 비행 플랫폼을 통해 세 가지 인지 관련 작업을 수행하며 다차원적인 인지 상태 평가를 제안합니다.

- **Technical Details**: 이 연구는 다양한 난이도의 인지 작업을 설계하고, EEG 신호를 활용하여 내주체 및 교차주체의 인지 관련 뇌 기능적 연결성을 분석합니다. 특히, 다중 클래스 인지 상태 평가를 위해 multi-head attention 기반의 EEGNet을 제안합니다.

- **Performance Highlights**: 연구 결과, 다중 클래스 인지 상태 평가에서 95.83%의 정확도를 달성하여 기존 연구들을 초월한 결과를 도출했습니다. 이로 인해 인지, 결정-making 및 뇌 기능 지역 간의 동적인 관계를 이해하는 데 있어 중요한 기여를 할 것으로 기대됩니다.



### Prior-free Balanced Replay: Uncertainty-guided Reservoir Sampling for Long-Tailed Continual Learning (https://arxiv.org/abs/2408.14976)
- **What's New**: 이 연구에서는 기존 기술에 의존하지 않고 장기적 분포의 데이터를 학습하는 새로운 방법인 Prior-free Balanced Replay (PBR) 프레임워크를 제안합니다. 이는 불균형한 클래스에서 정보의 잊힘을 줄이기 위해 설계되었습니다.

- **Technical Details**: PBR 프레임워크는 불확실성 기반의 저수지 샘플링 전략을 활용해 마이너리티( minority) 데이터를 우선적으로 메모리에 저장하는 방식입니다. 추가적으로, 두 가지 prior-free 구성 요소인 boundary constraint와 prototype constraint를 도입하여 마이너리티 데이터의 잊힘 문제를 완화합니다.

- **Performance Highlights**: 이 방법은 세 가지 장기적 벤치마크에서 평가되어 기존의 Continual Learning(CL) 방법들과 SOTA LTCL 접근 방식보다 우수한 성능을 기록하였습니다. 이는 작업 증가 및 클래스 증가 설정에서 모두 효과적입니다.



### CVPT: Cross-Attention help Visual Prompt Tuning adapt visual task (https://arxiv.org/abs/2408.14961)
- **What's New**: 본 논문에서는 Cross Visual Prompt Tuning (CVPT) 방법을 제안하여 기존의 Visual Prompt Tuning (VPT) 방법의 성능과 효율성을 향상시킵니다. CVPT는 프롬프트 토큰과 임베딩 토큰 간의 크로스 어텐션을 계산하여 세분화된 시각적 작업에 맞출 수 있도록 모델을 조정합니다.

- **Technical Details**: CVPT는 프롬프트 토큰과 임베딩 토큰 간의 크로스 어텐션을 활용하여 세멘틱 관계를 계산하고, learnable parameter의 수를 줄여 효율성을 높이는 weight-sharing 메커니즘을 도입합니다. 실험 결과, CVPT는 25개의 데이터셋에서 검증되었으며 VPT에 비해 평균적으로 4% 더 높은 정확도를 보였습니다.

- **Performance Highlights**: CVPT는 VTAB-1K 벤치마크에서 VPT에 비해 평균 4%의 정확도 향상을 이루었으며, FGVC 및 ADE20K 데이터셋에서도 SOTA 방법들과 경쟁할 수 있는 성능을 보여주었습니다. 특히 적은 수의 프롬프트 토큰을 사용할 때 다른 선진 PEFT 방법들과 비슷한 성능을 나타내었습니다.



### Multilingual Arbitrage: Optimizing Data Pools to Accelerate Multilingual Progress (https://arxiv.org/abs/2408.14960)
- **What's New**: 이번 연구는 "multilingual arbitrage"라는 개념을 도입하여 여러 모델 간의 성능 차이를 활용하여 합성 데이터 생성의 효율성을 높이는 방법론을 제안하고 있습니다. 이 방법은 단일 teacher 모델에 의존하는 기존 접근 방식의 한계를 극복하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 이 연구는 15개 언어에서 9개의 최첨단 다국어 모델을 사용하여 exhaustive experiments를 수행하였으며, 각각의 언어에 대해 여러 모델을 teacher로 두고 샘플을 전략적으로 라우팅하는 방법을 평가하여 최적의 성능을 달성하는 방법을 모색합니다.

- **Performance Highlights**: 연구 결과, reward-based routing 기법이 평균 56.5%의 generative win-rates 개선을 보여주었고, discriminative tasks에서는 최대 3.27% 개선을 달성하였습니다. 단일 teacher 모델 compared to, reward-based routing으로의 전환이 가장 성과가 좋음을 입증했습니다.



### NeuralOOD: Improving Out-of-Distribution Generalization Performance with Brain-machine Fusion Learning Framework (https://arxiv.org/abs/2408.14950)
- **What's New**: 본 논문에서는 Deep Neural Networks (DNNs)의 OOD(Out-of-Distribution) 일반화 성능을 개선하기 위한 새로운 Brain-machine Fusion Learning (BMFL) 프레임워크를 제안합니다.

- **Technical Details**: BMFL 프레임워크는 cross-attention 메커니즘을 채택하여 CV(Computer Vision) 모델의 시각적 지식과 인간 두뇌의 사전 인지 지식을 융합합니다. 또한, 사전 훈련된 시각 신경 인코딩 모델을 사용하여 fMRI(functional Magnetic Resonance Imaging) 데이터를 예측하며, Pearson 상관 계수 최대화 정규화 방법을 교육 과정에 도입합니다.

- **Performance Highlights**: 우리 모델은 ImageNet-1k 검증 데이터셋과 6개의 OOD 데이터셋에서 DINOv2 및 기준 모델보다 뛰어난 성능을 보여 다양한 시나리오에서 우수성을 입증하였습니다.



### Quotient Normalized Maximum Likelihood Criterion for Learning Bayesian Network Structures (https://arxiv.org/abs/2408.14935)
Comments:
          Accepted to AISTATS 2018

- **What's New**: 본 연구에서는 Bayesian 네트워크 구조 학습을 위한 새로운 정보 이론적 기준인 'quotient normalized maximum likelihood (qNML)'를 도입하였다. 기존의 factorized normalized maximum likelihood (fNML) 기준과 비교할 때, qNML는 점수 동등성을 만족하며 조정 가능한 하이퍼파라미터가 전혀 필요하지 않다.

- **Technical Details**: qNML는 BDeu 및 fNML 기준을 기반으로 하며, 더 나아가 예측 정확도를 손상시키지 않으면서도 단순한 모델을 생성하는 데 집중하고 있다. 정보 이론에 기반한 최대 우도(Normalized Maximum Likelihood, NML) 기준의 비용 효율적인 근사값을 제안하고, 이를 통해 계산 효율성을 높이고자 하였다.

- **Performance Highlights**: 인공 데이터 및 실제 데이터를 통해 qNML 기준이 성능이 뛰어난 경제적인 모델을 생성할 수 있음을 입증하였다. 이 기준을 사용하여 생성한 모델은 잘 예측할 수 있는 능력을 갖추고 있으며, 데이터 세트를 통해 qNML의 효과를 강조하였다.



### Distance-Forward Learning: Enhancing the Forward-Forward Algorithm Towards High-Performance On-Chip Learning (https://arxiv.org/abs/2408.14925)
- **What's New**: 이번 연구는 Forward-Forward (FF) 알고리즘을 재구성하여 거리 기반 메트릭 학습을 이용한 Distance-Forward (DF) 알고리즘을 제안하며, 이를 통해 FF의 성능을 개선하고 특히 효율적인 온칩 학습을 위한 경쟁력을 강화합니다.

- **Technical Details**: DF 알고리즘은 centroid-based 메트릭 학습의 관점에서 FF를 해석하고, N-pair margin loss를 통해 차별화된 feature 학습을 지원합니다. 또한, greedy local parameter 업데이트로 인한 정보 손실을 줄이기 위해 layer-collaboration local update 전략을 통합합니다.

- **Performance Highlights**: DF 모델은 MNIST에서 99.7%, CIFAR-10에서 88.2%, CIFAR-100에서 59%, SVHN에서 95.9%, ImageNette에서 82.5%의 정확도를 달성하며, BP 훈련에 비해 40% 미만의 메모리 비용으로 비교 가능한 성능을 보이고, 다양한 하드웨어 노이즈에 대한 견고성을 강화하여 온라인 학습 및 에너지 효율적인 neuromorphic 칩에서의 활용 가능성을 보여줍니다.



### The VoxCeleb Speaker Recognition Challenge: A Retrospectiv (https://arxiv.org/abs/2408.14886)
Comments:
          TASLP 2024

- **What's New**: 이번 VoxCeleb Speaker Recognition Challenges (VoxSRC)는 2019년부터 2023년까지 매년 개최된 일련의 도전 과제입니다. 이 도전 과제들은 스피커 인식 (speaker recognition) 및 다이어리제이션 (diarisation)을 평가하기 위해 여러 설정에서 진행되었고, 새로운 교육 및 평가 데이터 세트가 매년 공개되었습니다.

- **Technical Details**: VoxSRC의 두 가지 주요 작업은 (i) 스피커 검증 (speaker verification)과 (ii) 스피커 다이어리제이션입니다. 스피커 검증 작업은 두 음성 세그먼트가 동일한 스피커로부터 나온 것인지 확인하는 것이고, 스피커 다이어리제이션은 "누가 언제 말했는지" 라벨링하는 것입니다. 참가자들은 VoxCeleb2 dev 세트를 기반으로 훈련하며, 폐쇄형 및 개방형 트랙에서 참가할 수 있습니다.

- **Performance Highlights**: 5년 동안의 VoxSRC 도전 과제를 통해 성능이 지속적으로 향상되었습니다. 특히, 자기 지도 학습(self-supervised learning) 및 반지도 학습(semi-supervised learning)을 통해 스피커 인식 분야의 최신 동향을 반영하고 있으며, 실제로 사용되는 다양한 데이터 세트를 바탕으로 효과적인 방법론들이 제시되었습니다.



### Adversarial Attacks and Defenses in Multivariate Time-Series Forecasting for Smart and Connected Infrastructures (https://arxiv.org/abs/2408.14875)
Comments:
          17 pages, 32 figures

- **What's New**: 본 연구는 다변량 시계열 예측에 대한 적대적 공격의 영향을 탐구하며, 모델을 오염시키는 다양한 공격 방법을 활용하여 예측 정확도를 저하시킬 수 있음을 보여줍니다. 특히, Fast Gradient Sign Method (FGSM)와 Basic Iterative Method (BIM)와 같은 untargeted white-box 공격을 사용하여 적대적 방어 모델을 개발하는 방법에 대한 연구는 주목할 만합니다.

- **Technical Details**: 이 논문에서는 Long Short-Term Memory (LSTM) 모델을 활용하여 과거 데이터를 기반으로 미래 전력 소비 예측을 위한 여러 실험을 수행하였습니다. FGSM과 BIM를 사용하여 모델에 대한 공격을 시뮬레이션하고, 데이터 증강 및 계층별 경량화(layer-wise hardening)를 통해 robust 모델을 개발하는 방법을 제시합니다.

- **Performance Highlights**: 전력 데이터셋에서 RMSE(평균 제곱근 오차)는 72.41% 감소하였으며, 하드 디스크 데이터셋에서는 94.81% 감소하는 결과를 보여주었습니다. 또한, 연구에서 제안한 공격 및 방어 방법은 쉽게 전이 가능함을 입증하였습니다.



### Detecting AI Flaws: Target-Driven Attacks on Internal Faults in Language Models (https://arxiv.org/abs/2408.14853)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)에서 독성 콘텐츠를 탐지하는 새로운 공격 패러다임인 ToxDet를 제안하여 독성 응답을 유도하는 데 집중합니다.

- **Technical Details**: ToxDet는 강화 학습(reinforcement learning)을 활용하여 타겟 모델과의 상호작용을 통해 훈련되며, 이를 통해 적절한 질문 및 초기 답변을 생성하여 독성 응답을 유도합니다. 이 방식은 프롬프트 엔지니어링(prompt engineering) 대신 직접적으로 모델을 공격하는 접근입니다.

- **Performance Highlights**: AdvBench 및 HH-Harmless 데이터셋에서 실험 결과, ToxDet는 오픈소스와 블랙박스 모델에 대한 효과적인 공격 능력을 입증하며, 기존 LLM의 취약점을 드러내고 연구자들이 모델 강화를 위한 귀중한 자료를 제공합니다.



### Project SHADOW: Symbolic Higher-order Associative Deductive reasoning On Wikidata using LM probing (https://arxiv.org/abs/2408.14849)
Comments:
          6 pages, 1 figure

- **What's New**: SHADOW라는 중간 작업에 대해 조정된 언어 모델을 소개하고, 위키데이터(Wikidata) 삼중 완성을 사용하여 지식 기반 구축 작업에 대한 성능을 측정합니다. SHADOW는 LM-KBC 2024 챌린지에서 기준 솔루션보다 20% 향상된 F1 점수 68.72%로 평가받았습니다.

- **Technical Details**: SHADOW는 지식 기반 삼중 항에 대해 세밀하게 조정된 모델로, 연관 추론(associative deductive reasoning)에서 영감을 받은 방법론을 사용했습니다. 실험 동안, 주어진 데이터에 대해 조건부 생성 모델로 학습하며, 각 주제 및 관계 쌍에 대한 관련 객체를 검색하기 위한 템플릿을 설계합니다. SHADOW는 377개의 학습 샘플, 378개의 검증 샘플 및 378개의 테스트 샘플이 포함된 데이터 세트를 사용하여 훈련되었습니다.

- **Performance Highlights**: SHADOW는 다양한 관계에 대한 템플릿 식별 작업에서 우수한 성능을 보였습니다. 그러나 countryLandBordersCountry 관계에서 더 낮은 성능을 보였으며, 이는 이 관계의 성격 때문입니다. SHADOW는 기준 모델보다 약 20% 더 향상된 성과를 기록했으며, 다른 해결책과 비교했을 때 좋은 성능을 발휘했습니다. 그러나 여전히 최고의 점수에는 미치지 못하는 한계가 있습니다.



### Diffusion based Semantic Outlier Generation via Nuisance Awareness for Out-of-Distribution Detection (https://arxiv.org/abs/2408.14841)
- **What's New**: 새로운 방법론인 SONA(Semantic Outlier generation via Nuisance Awareness)를 도입하여 기존의 OOD(out-of-distribution) 탐지 문제를 해결하고, ID(in-distribution) 샘플의 픽셀 공간을 직접 활용해 도전적인 아웃라이어(outlier)를 생성합니다.

- **Technical Details**: SONA는 SONA guidance를 통해 ID 샘플의 의미 영역(semantic region)과 난잡성(nuisance regions) 레벨을 조절할 수 있습니다. 이로 인해 생성된 아웃라이어는 명확한 의미적으로 불일치하는 정보를 제공하며, ID와의 난잡성 유사성을 유지합니다. 또한, OOD 탐지기가 SONA 아웃라이어를 활용하여 의미 구별을 집중적으로 학습하도록 설계되었습니다.

- **Performance Highlights**: SONA 프레임워크는 Near-OOD 데이터셋에서 88%의 인상적인 AUROC(Area Under the Receiver Operating Characteristic) 점수를 달성하였으며, 이는 기존의 베이스라인 방법들보다 약 6% 향상된 성능을 보여줍니다.



### Diffusion Models Are Real-Time Game Engines (https://arxiv.org/abs/2408.14837)
Comments:
          Project page: this https URL

- **What's New**: GameNGen은 신경망(neural model)에 의해 완전히 구동되는 최초의 게임 엔진으로, 복잡한 환경에서 실시간 상호작용을 가능하게 합니다. 이 엔진은 인기 게임 DOOM을 단일 TPU에서 초당 20프레임 이상으로 시뮬레이션 할 수 있습니다.

- **Technical Details**: GameNGen은 두 단계로 훈련됩니다: (1) 강화 학습( reinforcement learning, RL) 에이전트가 게임을 플레이하며 훈련 세션을 기록하고, (2) 확산 모델(diffusion model)이 과거 프레임과 동작의 시퀀스에 따라 다음 프레임을 생성하도록 훈련됩니다. 이러한 조건화( conditioning)는 긴 경로를 따라 안정적인 자기 회귀적(auto-regressive) 생성을 가능하게 합니다.

- **Performance Highlights**: 다음 프레임 예측의 PSNR은 29.4로, 손실 JPEG 압축과 유사한 품질을 나타냅니다. 인간 평가자는 게임의 짧은 클립과 시뮬레이션의 클립을 구별하는 데 있어서 거의 무작위 확률에 가까운 성과를 보입니다.



### Strategic Optimization and Challenges of Large Language Models in Object-Oriented Programming (https://arxiv.org/abs/2408.14834)
Comments:
          10 pages

- **What's New**: 코드 생성(Code Generation) 연구에서 개별 함수의 작성에서 벗어나 클래스 레벨의 메서드 코드를 개발하는 데 초점이 맞춰졌습니다. 이 연구는 Object-Oriented Programming (OOP) 프레임워크 내에서의 메서드 레벨 코드 생성에 관해 다룹니다.

- **Technical Details**: 실험은 메서드 특정(context-specific)부터 프로젝트 범위의 모든 세부정보까지 다양한 맥락 정보가 포함된 프롬프트를 변경하여 진행되었습니다. "Prompt-Token Cost-Effectiveness"라는 혁신적인 메트릭(metric)을 도입하여 추가 맥락 레이어를 포함하는 경제적 실행 가능성을 평가했습니다.

- **Performance Highlights**: 결과는 메서드 호출 세부 정보가 포함된 프롬프트가 가장 높은 비용 효율성을 나타낸다는 것을 보여주었습니다. 또한, 다양한 Large Language Models (LLMs) 간의 오류 유형 분포와 개발자에게 제공하는 지원 수준의 차이를 발견했습니다. 특정 작업의 결합도(coupling)에서 LLM의 성능 차이를 관찰하였고, 개발자는 적절한 LLM을 선택하여 코드 품질을 최적화할 수 있습니다.



### A Comprehensive Benchmark of Machine and Deep Learning Across Diverse Tabular Datasets (https://arxiv.org/abs/2408.14817)
- **What's New**: 이 연구는 Deep Learning (DL) 모델과 전통적인 Machine Learning (ML) 모델의 성능을 비교하기 위해 포괄적인 벤치마크를 도입하였습니다. 기존의 DL 모델이 전통적인 ML 모델에 비해 성능이 떨어진다는 일반화 의미를 도전하고, 다양한 데이터 세트에서 DL 모델이 우수한 성능을 보일 수 있는 조건을 분석하였습니다.

- **Technical Details**: 111개의 데이터 세트를 사용하여 20개의 다양한 모델 구성(회귀와 분류 작업 포함)을 평가하였습니다. DL 모델로는 7개, Tree-based Ensemble 모델 7개, 전통 ML 모델 6개를 포함하였으며, 각 데이터 세트의 특성을 파악하기 위한 메타 학습 접근 방식을 채택했습니다. 데이터 세트의 크기와 특성의 다양성에 중점을 두어 선택하였습니다.

- **Performance Highlights**: 연구 결과 보고서에 따르면, DL 모델은 높은 첨도(kurtosis)와 상대적으로 적은 수의 행(row)에서 우수한 성능을 보였으며, DL 모델이 ML 모델을 초과할 확률을 예측하는 모델을 훈련하여 86.1%의 정확도(AUC 0.78)를 달성했습니다.



### Poly2Vec: Polymorphic Encoding of Geospatial Objects for Spatial Reasoning with Deep Neural Networks (https://arxiv.org/abs/2408.14806)
- **What's New**: 이번 연구에서는 Poly2Vec라는 새로운 인코딩 프레임워크를 소개합니다. Poly2Vec는 2D 포인트, 폴리라인, 폴리곤 등 다양한 지리 공간 객체를 통합하여 모델링할 수 있는 방법을 제시하며, 다양한 다운스트림 작업에 관계없이 효과적으로 활용될 수 있습니다.

- **Technical Details**: Poly2Vec는 2D Fourier transform의 힘을 활용하여 지리 공간 객체의 중요한 공간 속성을 고정 길이 벡터로 인코딩합니다. 이 벡터는 이후 신경망 모델에 입력되어 공간 추론 작업을 수행합니다. Poly2Vec는 위치와 모양 정보를 효과적으로 캡처하며, 지리 공간 객체의 복합 속성을 지원하는 데 적합합니다.

- **Performance Highlights**: Poly2Vec는 합성 데이터셋과 실제 데이터셋 모두에서 다양한 기하학적 타입을 처리하면서 이전 방법들보다 뛰어난 성능을 보였습니다. 이 프레임워크는 공간 관계 분류 및 k-NN 검색과 같은 여러 공간 추론 작업에서 Consistency한 성능을 보여주었습니다.



### GINN-KAN: Interpretability pipelining with applications in Physics Informed Neural Networks (https://arxiv.org/abs/2408.14780)
- **What's New**: 본 논문에서는 해석 가능한 신경망 모델을 위한 해석 가능성 파이프라인(interpretability pipelining) 개념을 제안하며, 이를 통해 기존의 개별 해석 기술들을 통합하여 성능을 향상시키는 방법을 소개합니다.

- **Technical Details**: 특히, Growing Interpretable Neural Network (GINN)와 Kolmogorov Arnold Networks (KAN)이라는 두 가지 최근 모델에 초점을 맞추고, 이들 각각의 한계와 장점을 분석합니다. 새로운 해석 가능한 신경망 GINN-KAN을 도입하여 이 두 모델의 장점을 결합합니다.

- **Performance Highlights**: Feynman symbolic regression benchmark 데이터셋에서 GINN-KAN이 기존의 GINN과 KAN보다 우수한 성능을 보였으며, 일반적인 블랙박스 신경망 대신 물리 정보 신경망(Physics-Informed Neural Networks, PINNs)의 대안으로 자리 잡을 수 있는 가능성을 보여줍니다.



### MROVSeg: Breaking the Resolution Curse of Vision-Language Models in Open-Vocabulary Semantic Segmentation (https://arxiv.org/abs/2408.14776)
Comments:
          Technical report

- **What's New**: MROVSeg라는 다중 해상도 훈련 프레임워크를 제안하여 개방 어휘 의미 분할(Open-Vocabulary Semantic Segmentation) 문제를 해결합니다. 이 방법은 단일 pretrained CLIP 백본을 사용하여 고해상도 입력을 슬라이스(slicing)하고 균일한 패치(patch)로 나누어 처리합니다.

- **Technical Details**: MROVSeg는 Multi-Res Adapter를 포함하여 공간 기하를 복원하고 패치 간의 지역-전역 일치를 포착합니다. Multi-grained Masked Attention 메커니즘을 통해 객체 쿼리와 다중 해상도 CLIP 특성 간의 교차-어텐션(cross-attention)을 수행하여 다중 해상도(feature)를 집계합니다.

- **Performance Highlights**: MROVSeg는 Pascal Context-59, ADE-150, Pascal VOC 등에서 기존 최첨단 방법보다 2.4 mIoU% 향상된 성능을 보여 주며 오픈-어휘 의미 분할의 새로운 기준을 세웠습니다.



### A global AI community requires language-diverse publishing (https://arxiv.org/abs/2408.14772)
Comments:
          Translations by Michael Hardy (Guarani), Vandana Sarin and Vivek Sarin (Hindi), Roshna Omer Abdulrahman (Soranî Kurdish), Gabriel Poesia (Portuguese), and Matías Grinberg (Spanish). In the proceedings of the Global AI Cultures Workshop at the Twelfth International Conference on Learning Representations (ICLR) 2024, Vienna, Austria, May 7-11, 2024

- **What's New**: 본 논문은 AI 연구 커뮤니티 내에서 영어의 지배적 위치에 대한 이슈를 다루고 있습니다. 저자들은 영어 출판의 요구가 AI 분야에서 더 넓은 추출(regime of extraction) 체계를 유지하고 강화한다고 주장합니다.

- **Technical Details**: 저자들은 대규모 언어 모델(large language models) 및 기계 번역(machine translation)을 장려하지만, 이러한 기술들이 과학자 및 독자들의 언어적 제외(linguistic exclusion) 증상을 방증한다고 보았습니다.

- **Performance Highlights**: 저자는 회의(conferences)를 개최하는 국가의 언어로 진행하고, 동료 심사(peer review)자가 논문의 언어 적합성을 평가하지 않도록 지침을 제공하며, 다양한 언어로 출판 및 발표 기회를 제공하는 건강한 출판 문화의 대안을 제안합니다.



### Sequential-Scanning Dual-Energy CT Imaging Using High Temporal Resolution Image Reconstruction and Error-Compensated Material Basis Image Generation (https://arxiv.org/abs/2408.14754)
- **What's New**: 이번 논문에서는 전통적인 CT 시스템에 직접 적용 가능하고, 전문 하드웨어 설계가 필요 없는 순차 스캐닝 데이터 수집 방식의 이점을 활용한 이중 에너지 컴퓨터 단층 촬영(dual-energy computed tomography, DECT) 기법을 개선하는 방법을 제안합니다. 이 방법은 ACCELERATION이라는 기법을 기반으로 하여 보다 정확한 물질 농도 정량화를 가능하게 합니다.

- **Technical Details**: ACCELERATION은 고해상도 이미지 재구성과 오류 보완 물질 기초 이미지 생성을 결합하여 순차 스캐닝 DECT의 기술적 문제를 해결합니다. 이 과정에서 두 가지 주요 기술이 사용되며, 첫째로 고Temporal Resolution 이미지 재구성을 통한 시간적 매칭과, 둘째로 오류 보완 물질 기반 이미지 생성을 통한 물질 정량화 기술입니다.

- **Performance Highlights**: ACCELERATION을 사용한 결과, 정량화 정확도와 이미지 품질이 개선되었음을 보여줍니다. 이 접근법은 임상 적용 가능성을 높이고, 보다 넓은 범위의 환자들에게 혜택을 줄 수 있는 저비용의 DECT 기법이 구현되었음을 의미합니다.



### CoopASD: Cooperative Machine Anomalous Sound Detection with Privacy Concerns (https://arxiv.org/abs/2408.14753)
Comments:
          Accepted by GLOBECOM 2024

- **What's New**: 최근 산업 인터넷(IoT)에서 기계의 비정상 소음 탐지(Anomalous Sound Detection, ASD)에 대한 새로운 프레임워크인 CoopASD가 제안되었습니다. CoopASD는 탈중앙화 환경에서 데이터 프라이버시를 보장하면서 여러 공장이 협력하여 ASD 모델을 공동 개발할 수 있는 방법을 제공합니다.

- **Technical Details**: CoopASD 프레임워크는 각 공장이 로컬 데이터세트에서 ASD 모델을 훈련하고, 중앙 서버가 이러한 로컬 모델을 주기적으로 집계하는 구조를 따릅니다. 이 과정에서 비독립적 비동일분포(non-iid) 데이터와 레이블이 없는 이상치로 인한 과적합 문제를 해결하기 위해 샘플링(sampling), 선택적 업로드(selective upload), 조기 중지(early stop)와 같은 정규화(regularization) 방법이 적용됩니다.

- **Performance Highlights**: CoopASD는 14개 기계 유형에 대해 경쟁력 있는 성능을 보이며, 중앙 집중식 설정에서의 최첨단 모델(state-of-the-art) 대비 오직 0.08%의 성능 저하만을 기록했습니다. 이를 통해 CoopASD는 실제 제작 환경에서도 유용할 수 있는 가능성을 보여줍니다.



### Benchmarking Reinforcement Learning Methods for Dexterous Robotic Manipulation with a Three-Fingered Gripper (https://arxiv.org/abs/2408.14747)
- **What's New**: 본 연구는 RL(강화 학습) 알고리즘을 실제적인 환경에서 직접 훈련하는 새로운 방법을 제안합니다. 여기에 사용된 세 가지 RL 알고리즘은 복잡한 조작 작업에서 비현실적인 시뮬레이션 환경의 한계를 극복하기 위해 실제 상황에서 벤치마킹되었습니다.

- **Technical Details**: RL 알고리즘은 여러 손가락을 가진 로봇 그리퍼를 사용하여 조작 기술을 학습합니다. 연구는 RL 훈련의 효용성을 강조하며, 실세계에서의 활동을 위해 RL을直接 학습하는 방법론을 탐구합니다.

- **Performance Highlights**: 제안된 방법론은 기존의 다른 연구들에 비해 다중 조작 작업에서 더 높은 성과를 보였습니다. 특히, RL의 실제 환경에서의 적용 가능성을 입증하여 로봇공학 연구자들과 실무자들에게 실질적인 인사이트를 제공합니다.



### RSTeller: Scaling Up Visual Language Modeling in Remote Sensing with Rich Linguistic Semantics from Openly Available Data and Large Language Models (https://arxiv.org/abs/2408.14744)
Comments:
          Submitted to ISPRS

- **What's New**: 본 연구에서는 OpenStreetMap (OSM) 데이터를 활용하여 Google Earth Engine (GEE) 플랫폼에서 수집된 이미지에 대해 의미론적으로 풍부한 캡션을 대규모로 생성하는 워크플로우를 제안합니다. 이를 통해 100만 개 이상의 원거리 감지 (RS) 이미지를 포함하는 RSTeller라는 다중모달 데이터셋을 구축하였습니다.

- **Technical Details**: RSTeller 데이터셋은 다중 설명 캡션과 함께 제공되며, 기존의 비전 언어 모델 (VLM)들이 RS 장면 이해 작업에서 성능을 향상시키도록 지속적 학습 (continual pre-training) 방법을 통해 효과를 검증하였습니다. 이 방법론은 최소한의 수작업 노력으로 고품질 주석 데이터에 대한 접근성을 민주화합니다.

- **Performance Highlights**: RSTeller 데이터셋을 통해 여러 기존 VLM들의 성능이 향상됨을 입증하였습니다. 이러한 접근은 RS 이미지를 주석 처리하는 데 필요한 전문 지식과 노력을 크게 줄이며, 환경 지속 가능성 문제도 해결하게 됩니다.



### TART: Boosting Clean Accuracy Through Tangent Direction Guided Adversarial Training (https://arxiv.org/abs/2408.14728)
- **What's New**: 이 논문에서는 Tangent Direction Guided Adversarial Training (TART)라는 새로운 방법을 제안하며, 기존의 적대적 방어 알고리즘을 개선하기 위해 데이터 매니폴드의 접선 공간(tangent space)을 활용합니다. TART는 적대적 예제의 접선 방향(tangential direction)을 추정하여 적응형 섭동 한계(perturbation limit)를 할당함으로써 모델의 청정 정확도(clean accuracy)를 회복할 수 있도록 합니다.

- **Technical Details**: TART는 적대적 예제의 접선 성분(tangential component)과 정상 성분(normal component)을 고려하여 훈련 중 큰 정상 성분을 가진 적대적 예제를 피할 수 있도록 합니다. 이를 통해 중요한 데이터 포인트에 적절한 섭동 한계를 설정하게 됩니다.

- **Performance Highlights**: 실험 결과, TART는 청정 정확도를 일관되게 향상시킴과 동시에 적대적 공격에 대한 높은 강인성을 유지하는 데 성공했습니다. 특히, TART는 다양한 적대적 방어 방법과 결합할 수 있는 범용적인 방법임을 보여주었습니다.



### PAT: Pruning-Aware Tuning for Large Language Models (https://arxiv.org/abs/2408.14721)
- **What's New**: 이번 연구에서는 Pruning-Aware Tuning (PAT)이라는 혁신적인 패러다임을 제안하여 구조적 가지치기와 파인튜닝(fine-tuning)을 동시에 수행함으로써 모델 성능을 개선하는 접근 방식을 소개합니다.

- **Technical Details**: PAT는 Hybrid Sparsification Modules (HSMs)을 Attention과 FFN(Feed Forward Network) 구성 요소 사이에 통합하여 파라미터 효율성을 높이며, Hybird-Identity-Operator (HIO)를 통해 학습 가능한 파라미터 수를 줄입니다. 또한, Identity Loss (IL)를 적용하여 훈련의 강건성을 증대시키고 있습니다. 모든 HSM은 Unified Sparsification Mask (USM)로 통해 통합적으로 관리됩니다.

- **Performance Highlights**: Llama2-7b 모델을 기준으로 25% 가지치기를 통해 1.33배의 속도 향상과 함께 LoRA-64 모델보다 최대 1.26% 더 높은 정확도를 기록했습니다.



### Residual-based Adaptive Huber Loss (RAHL) -- Design of an improved Huber loss for CQI prediction in 5G networks (https://arxiv.org/abs/2408.14718)
Comments:
this https URL

- **What's New**: 본 논문은 5G 네트워크의 Channel Quality Indicator (CQI) 예측을 위해 Residual-based Adaptive Huber Loss (RAHL)라는 새로운 손실 함수를 제안합니다. RAHL은 learnable residual을 도입하여 모델이 데이터의 오류 분포에 따라 적응할 수 있도록 합니다.

- **Technical Details**: RAHL은 MSE (Mean Squared Error)와 MAE (Mean Absolute Error)의 장점을 결합하며, 사용자가 정의하는 hyperparameter인 delta를 통해 두 손실 함수 사이에서 원활하게 전환합니다. LSTM (Long Short-Term Memory) 모델과 함께 사용하여 성능을 극대화합니다.

- **Performance Highlights**: RAHL은 Mean Absolute Percentage Error (MAPE) 기준으로 기존의 MSE, MAE, Huber 손실 함수보다 우수한 성능을 보이며, 5G 네트워크에서 CQI 예측의 정확도를 향상시키는 데 기여하였습니다.



### Text2SQL is Not Enough: Unifying AI and Databases with TAG (https://arxiv.org/abs/2408.14717)
- **What's New**: 이 논문에서는 데이터베이스를 통해 자연어 질문에 답하기 위한 새로운 모델인 Table-Augmented Generation (TAG)을 제안합니다. 기존의 Text2SQL이나 Retrieval-Augmented Generation (RAG) 방법의 한계를 극복하고, LM과 데이터베이스 시스템의 결합된 능력을 활용하여 사용자가 임의의 질문을 할 수 있도록 합니다.

- **Technical Details**: TAG 모델은 세 가지 주요 단계를 포함합니다: 1) 쿼리 합성 단계(syn), 2) 쿼리 실행 단계(exec), 3) 답변 생성 단계(gen). 첫 번째 단계에서는 사용자의 자연어 요청을 실행 가능한 데이터베이스 쿼리로 변환하고, 두 번째 단계에서는 이 쿼리를 실행하여 관련 데이터를 수집합니다. 마지막으로, LM의 기능을 이용하여 자연어로 된 최종 답변을 생성합니다.

- **Performance Highlights**: 기존의 방법들은 20% 이하의 정확도로 쿼리를 처리하지만, TAG 파이프라인은 20%에서 65%까지 높은 정확도를 기록하였습니다. 이는 TAG 시스템의 효율성과 가능성을 보여줍니다.



### StyleSpeech: Parameter-efficient Fine Tuning for Pre-trained Controllable Text-to-Speech (https://arxiv.org/abs/2408.14713)
- **What's New**: 이번 논문에서는 자연스럽고 정확한 합성 음성을 향상시키기 위한 새로운 Text-to-Speech (TTS) 시스템인 StyleSpeech를 소개합니다. 기존 TTS 기술을 기반으로 StyleSpeech는 스타일 및 음소(feature) 특성을 동시에 학습할 수 있는 독특한 Style Decorator 구조를 통합하여 적응성 및 효율성을 개선하는 Lower Rank Adaptation (LoRA) 원칙을 활용합니다.

- **Technical Details**: StyleSpeech는 스타일 특성을 최소한의 변화로 기존 음소 매개변수와 분리하여 학습하는 스타일 장식 구조를 사용합니다. LoRA 기술을 적용하여 사전 훈련된 모델의 효율적인 미세 조정이 가능하며, LLM-Guided Mean Opinion Score (LLM-MOS)라는 새로운 자동 평가 지표를 도입하여 TTS 시스템 성능을 객관적이고 안정적으로 평가합니다.

- **Performance Highlights**: 방대한 기준 데이터 세트를 통한 실험 결과, StyleSpeech는 기존의 최첨단 기준 모델 대비 단어 오류율(Word Error Rate)을 15% 개선하고, 전반적인 평가에서 12% 향상된 성능을 보여주었습니다. 이러한 발전은 TTS 시스템의 적용 가능성을 더욱 넓히며, 인터랙티브 가상 비서, 적응형 오디오북, 게임 맞춤형 음성 등 다양한 분야에서 활용될 수 있습니다.



### Smart Multi-Modal Search: Contextual Sparse and Dense Embedding Integration in Adobe Express (https://arxiv.org/abs/2408.14698)
- **What's New**: 이번 논문은 Adobe Express 템플릿 검색에서 다중모드(multi-modal) 검색 시스템을 위한 새로운 아키텍처를 소개합니다. CLIP과 같은 다중모드 임베딩을 활용하여 텍스트와 이미지 검색을 직접 지원하면서도, 사용자 지리적 특성이나 최근성 같은 컨텍스트(contextual features)를 통합하는 데의 도전 과제를 다룹니다.

- **Technical Details**: 이 논문에서는 클라이언트의 검색 요구를 충족하기 위해 여러 다중모드 모델들을 사용하였으며, AB 테스트를 통해 임베딩 모델 선택, 매칭 및 랭킹의 역할, 밀집(dense)과 희소(sparse) 임베딩 간의 균형을 최적화하였습니다. AdobeCS 기술을 활용한 다중모드 검색 시스템은 약 30만 개의 템플릿 데이터에서 매우 효율적으로 작동하도록 설계되었습니다.

- **Performance Highlights**: 이번 연구는 희소, 밀집, 컨텍스트 특성을 활용하여 짧은 쿼리와 긴 쿼리 검색을 향상시키고, null 비율을 70% 이상 줄이며 클릭률(CTR)을 증가시키는 데 기여했습니다. 이러한 결과는 복잡한 쿼리에 대한 검색 시스템의 효과적인 개발이 가능하다는 통찰을 제공합니다.



### Training-Free Activation Sparsity in Large Language Models (https://arxiv.org/abs/2408.14690)
- **What's New**: TEAL (Training-Free Activation Sparsity in LLMs)는 대규모 언어 모델에서 훈련 없이 활성화 희소성을 적용할 수 있는 방법으로, 모델 전반에 걸쳐 40-50%의 희소성을 달성합니다.

- **Technical Details**: TEAL은 크기 기반 활성화 희소성을 적용하며, 레이어에 따라 기준이 다릅니다. LLaMA 아키텍처 모델에서 제로 평균의 단일 모드 분포를 기반으로 하여 비중요 활성화를 다듬어 모델의 희소성을 높입니다.

- **Performance Highlights**: TEAL은 40% 및 50% 모델 전반에서 각각 1.53배 및 1.8배의 속도 향상을 달성하며, 가중치 양자화와 호환되어 추가적인 효율성을 제공합니다.



### Bridging the Gap: Unpacking the Hidden Challenges in Knowledge Distillation for Online Ranking Systems (https://arxiv.org/abs/2408.14678)
- **What's New**: 본 논문에서는 추천 시스템에 대한 Knowledge Distillation (KD)의 독특한 도전 과제를 해결하고, 데이터 분포 변화, 최적의 교사 구성 찾기, 교사 레이블의 효율적인 공유를 통한 모델 개선을 제시합니다.

- **Technical Details**: KD는 일반적으로 하드 레이블과 소프트 레이블을 사용하는 방법으로, 추천 시스템의 데이터는 빠르게 변동하여 이 방식이 최적이 아닐 수 있습니다. 따라서 본 연구에서는 보조 작업 기반의 KD 전략을 통해 교사의 편향을 학생에게 전달하지 않도록 했습니다. 교사를 지속적으로 업데이트하여 학생에게 최신 정보를 제공합니다.

- **Performance Highlights**: 교사 모델과 학생 모델 간의 구성을 최적화하여, E(LTV) 손실을 0.4% 줄이는 성과를 기록했습니다. 학생 모델은 상대적으로 작은 교사(학생의 2배 크기)로부터도 성능이 향상되는 것을 보여주었습니다.



### Visions of Destruction: Exploring a Potential of Generative AI in Interactive Ar (https://arxiv.org/abs/2408.14644)
- **What's New**: 이 논문은 인터랙티브 아트에 대한 생성 AI의 잠재력을 탐구하며, 'Visions of Destruction'라는 아트워크를 사례 연구로 제공합니다. 이는 gaze 기반 상호작용을 통해 환경에 대한 인간 활동의 영향을 상징적으로 표현한 디지털 풍경 변화를 이루어냅니다.

- **Technical Details**: 생성 AI의 발전, 특히 Deep Learning (DL)과 GANs(Generative Adversarial Networks)의 도입은 창작 방법론을 혁신적으로 변화시켰습니다. StyleGAN3와 Stable Diffusion 등 다양한 모델들이 대규모 데이터셋으로 훈련되어 실시간 이미지 전환 및 상호작용을 가능하게 합니다. 또한, ResNet과 CLIP과 같은 기술들은 참여자의 행동을 더 세밀하게 해석할 수 있도록 지원합니다.

- **Performance Highlights**: 이 논문은 생성 AI가 인터랙티브 아트 분야에 혁신을 가져올 수 있는 잠재력을 강조하며, 예술 작품이 관객의 상호작용에 실시간으로 반응함으로써 새로운 창작 경험을 제공합니다. 여러 사례(예: 'Dreampainter' 및 'Learning to See')는 이러한 기술이 관객의 참여를 어떻게 풍부하게 만드는지를 보여줍니다.



### Hybrid Deep Convolutional Neural Networks Combined with Autoencoders And Augmented Data To Predict The Look-Up Table 2006 (https://arxiv.org/abs/2408.14626)
Comments:
          11 pages, 6 figures

- **What's New**: 본 연구는 autoencoder(오토인코더)와 데이터 증강(data augmentation) 기술로 강화된 혼합 심층 컨볼루션 신경망(DCNN) 모델을 개발하여 비판적인 열 유량(Critical Heat Flux, CHF)을 높은 정확도로 예측하는 방법을 모색합니다.

- **Technical Details**: 이 모델은 7225개의 샘플로 구성된 데이터셋에서 훈련 및 테스트되었으며, 결정계수(R²), Nash-Sutcliffe 효율성(NSE), 평균 절대 오차(MAE), 정규화 루트 평균 제곱 오차(NRMSE) 등의 성능 지표를 사용하여 평가되었습니다. 오토인코더 구성의 차이를 두어 원래 입력 특성을 증강함으로써 모델의 예측 능력이 상당히 향상되었습니다.

- **Performance Highlights**: 테스트된 모델 중 DCNN_3F-A2 구성은 훈련 중 R² 0.9908, 테스트 중 R² 0.9826을 달성하여 기본 모델 및 다른 증강 버전보다 높은 정확도를 보였습니다. 이러한 결과는 심층 학습과 특성 증강의 결합이라 하는 제안된 혼합 접근법이 CHF 예측에 대해 강력한 해결책을 제공하며, 더 넓은 범위의 조건에서 일반화될 수 있음을 시사합니다.



### How to build trust in answers given by Generative AI for specific, and vague, financial questions (https://arxiv.org/abs/2408.14593)
- **What's New**: 이 연구는 재무 문제에 대한 Generative Artificial Intelligence (GenAI)의 조언에 대한 소비자의 신뢰 구축 모델을 제안하고, 구체적인 질문과 모호한 질문의 경우 소비자의 신뢰 구축 방식이 다름을 발견했습니다.

- **Technical Details**: 이 연구는 구조 방정식 모델링(Structural Equation Modelling, SEM)과 다중 그룹 분석(Multi-Group Analysis, MGA)을 사용하여 신뢰 구축 방법을 검증했습니다. MGA는 소비자가 특정 질문을 한 경우와 모호한 질문을 한 경우를 비교합니다.

- **Performance Highlights**: 연구 결과, 특정 재무 질문에 대해 신뢰가 구축되는 방식과 모호한 질문에 대해 신뢰가 구축되는 방식이 다르게 나타났습니다. 특히, 특정 질문에 대하여는 인간 같은 상호작용이 신뢰를 강화하지 않지만, 모호한 질문에 대해서는 인간성이 신뢰를 구축하는 데 기여했습니다. 신뢰 구축의 네 가지 방식은 (1) 인간 감독 및 개입, (2) 투명성과 통제, (3) 정확성과 유용성, (4) 사용 용이성과 지원입니다.



### DIAGen: Diverse Image Augmentation with Generative Models (https://arxiv.org/abs/2408.14584)
Comments:
          Accepted for publication in GCPR 2024

- **What's New**: DIAGen은 최근 제안된 DA-Fusion을 기반으로 하여 고차원 시맨틱 다양성을 높이는 새로운 데이터 증강 방법을 제안합니다. Gaussian noise를 임베딩에 적용하여 다양성을 강화하고, 텍스트-텍스트 생성 모델을 활용하여 이미지 생성을 안내합니다.

- **Technical Details**: DIAGen은 세 가지 주요 요소를 바탕으로 구축되었습니다: 첫째, Gaussian noise를 추가하여 클래스 개념의 임베딩 공간에 변형을 가합니다. 둘째, 텍스트-텍스트 모델 GPT-4를 이용하여 클래스별 텍스트 프롬프트를 통한 생성 과정을 유도합니다. 셋째, 질 낮은 샘플의 영향을 줄이기 위해 가중치 메커니즘을 도입합니다.

- **Performance Highlights**: DIAGen은 다양한 데이터셋에서 실험 결과를 통해 시맨틱 다양성을 증진시킬 뿐만 아니라 후속 분류기의 성능도 개선하는 효과를 보였습니다. 특히, 배포되지 않은 샘플에 대해 DIAGen의 장점이 두드러졌습니다.



### CURLoRA: Stable LLM Continual Fine-Tuning and Catastrophic Forgetting Mitigation (https://arxiv.org/abs/2408.14572)
Comments:
          Code available at this https URL

- **What's New**: CURLoRA는 CUR 행렬 분해를 활용하여 대형 언어 모델을 조정하는 새로운 접근 방식을 소개합니다. 이 방법은 연속 학습 중의 재앙적 망각을 완화하고 학습 가능한 매개변수 수를 줄이는 두 가지 주요 문제를 해결합니다.

- **Technical Details**: CURLoRA는 CUR decomposition을 사용하여 사전 훈련된 가중치 행렬을 분해하고 U 매트릭스를 제로 매트릭스로 초기화한 뒤 이를 조정합니다. 이 과정은 임PLICIT 정규화를 제공하며, 특정 데이터셋에서 실험을 통해 표준 LoRA보다 우수한 성능을 보였습니다.

- **Performance Highlights**: CURLoRA는 지속적인 조정 시에 기본 모델의 perplexity 점수를 유지하면서도 작업 정확도를 매우 높고 안정적으로 유지하는 성능을 입증했습니다. 특히 제한된 데이터 환경에서도 재앙적 망각을 잘 완화하여 안정적인 성능을 보여주었습니다.



### Improving Clinical Note Generation from Complex Doctor-Patient Conversation (https://arxiv.org/abs/2408.14568)
- **What's New**: 본 논문에서는 CliniKnote라는 방대한 의사-환자 대화 데이터셋과 K-SOAP (Keyword, Subjective, Objective, Assessment, and Plan) 노트 형식을 제안합니다. 이를 통해 클리닉 노트 생성에 대한 자동화 시스템의 발전을 목표로 하고 있습니다.

- **Technical Details**: CliniKnote 데이터셋은 1,200개의 복잡한 의사-환자 대화와 이에 대응하는 클리닉 노트를 포함합니다. 전통적인 SOAP 노트에 키워드 섹션을 추가하여 정보를 신속하게 식별할 수 있도록 개선하였으며, 최신 대형 언어 모델(LLMs)을 기반으로 하는 자동 생성 파이프라인을 개발했습니다. 또한, 다양한 LLM의 성능을 벤치마킹하여 효율성과 성능을 획기적으로 향상시켰습니다.

- **Performance Highlights**: 자동 생성된 K-SOAP 노트의 검토 및 수정에 필요한 시간이 기존의 수작업 작성 및 검토 시간보다 현저히 줄어든 것을 보여주었으며, LLM을 세부 도메인 지식으로 파인튜닝하여 최상의 성능을 달성했습니다.



### A Survey of Camouflaged Object Detection and Beyond (https://arxiv.org/abs/2408.14562)
Comments:
          26 pages, 10 figures, 8 tables

- **What's New**: 이 논문은 Camouflaged Object Detection (COD)에 대한 가장 포괄적인 리뷰를 제공하며, 기존의 조사에서의 한계를 극복하고 최신 기술 및 방법론을 포함하고 있습니다.

- **Technical Details**: COD는 정적 이미지에서 camouflaged 객체를 탐지하는 이미지 수준 COD와 비디오 시퀀스에서 이러한 객체를 탐지하는 비디오 수준 COD(VCOD)로 나뉘며, 각기 다른 접근 방식으로 분석됩니다. 전통적인 방법과 딥러닝 기반 접근 모두를 포함하여 다양한 방법론이 논의됩니다.

- **Performance Highlights**: 딥러닝 기반 기법의 정량적 및 정성적 성능을 분석하고, 40개의 이미지 수준 모델과 8개의 비디오 수준 모델을 벤치마킹하며 평가 메트릭을 사용하여 결과를 종합적으로 제시합니다.



### Revisiting Image Captioning Training Paradigm via Direct CLIP-based Optimization (https://arxiv.org/abs/2408.14547)
Comments:
          BMVC 2024

- **What's New**: 이 논문에서는 이미지 캡셔닝(Image Captioning)을 위한 새로운 훈련 패러다임인 Direct CLIP-Based Optimization (DiCO)를 제안합니다. 기존의 SCST(Self-Critical Sequence Training) 방식의 불안정성과 캡션 품질의 저하 문제를 해결하려는 접근입니다.

- **Technical Details**: DiCO는 사람과의 높은 상관관계를 가진 학습 가능한 캡셔닝 평가자로부터 증류된 보상 모델(reward model)을 공동 학습하고 최적화합니다. 이를 통해 원래 모델의 수렴을 방지하고, 캡셔너(Captioner) 내부에서 가중 분류 문제를 해결하여 다양한 캡션 품질 지표에서 동시에 최적화가 가능합니다.

- **Performance Highlights**: DiCO는 COCO 데이터셋에서 광범위한 실험을 하고, 현대적인 측정 지표에서 개선된 품질과 훈련의 안정성을 보여 주었으며, 전통적인 지표에서도 경쟁력 있는 성과를 유지했습니다. DiCO는 앤지 데이터셋 외에도 6개의 다른 이미지 캡셔닝 벤치마크에서 일반화 능력을 입증했습니다.



### Multi-Agent Path Finding with Real Robot Dynamics and Interdependent Tasks for Automated Warehouses (https://arxiv.org/abs/2408.14527)
Comments:
          Accepted to ECAI-2024. For related videos, see this https URL

- **What's New**: 이번 연구에서는 기존의 다중 에이전트 경로 찾기(MAPF) 문제를 새로운 관점에서 접근하여, 창고에서의 온라인 주문 배송 문제를 해결하고자 합니다. 이를 위해 Interleaved Prioritized Planning 알고리즘과 Via-Point Star (VP*) 알고리즘을 제안하였습니다.

- **Technical Details**: 제안된 방법은 상호 의존적인 작업을 처리하기 위해 동적으로 우선 순위를 할당하는 Interleaved Prioritized Planning 알고리즘을 갖추고 있습니다. 또한, 이동하는 장애물을 피하면서 로봇이 목표 지점을 방문하는 최적 동작 경로를 계산하는 VP* 알고리즘을 사용합니다. 로봇의 속도 프로필을 고정함으로써 NP-난해한 문제를 다항시간 문제로 변환하여 효율적인 평가를 가능하게 했습니다.

- **Performance Highlights**: 모의 실험 및 실제 창고에서의 초기 테스트를 통해 제안된 방법의 효과성을 검증하였고, 다양한 환경 구성의 영향을 분석하였습니다. 이를 통해 로봇의 실제 동역학을 반영한 경로 설정의 필요성을 확인했습니다.



### Estimating Uncertainty with Implicit Quantile Network (https://arxiv.org/abs/2408.14525)
Comments:
          This method is simple to implement and offers important information for performance critical applications

- **What's New**: 이 논문은 ensemble learning 및 bayesian neural networks와 같은 기존 접근 방식에 대한 간단한 대안을 제시합니다. Implicit Quantile Network를 사용하여 손실 분포를 직접 모델링함으로써 모델이 예측에 대해 얼마나 불확실한지를 추정합니다.

- **Technical Details**: Implicit Quantile Network(IQN)는 모델 아키텍처를 변경할 필요 없이 예측 모델의 훈련 세트에 대한 손실 분포를 예측하여 테스트 세트에서 모델의 불확실성을 추정합니다. 이 방법은 ensemble 모델이나 dropout을 사용하지 않고도 적용할 수 있으며, 경쟁력 있는 성능을 제공합니다.

- **Performance Highlights**: MNIST 및 CIFAR 데이터셋에 대한 실험 결과, 잘못된 예측에 대한 손실 분포의 평균이 2배 더 높았으며, 높은 불확실성 데이터가 제거되었을 때 모델의 정확도가 최대 10%까지 향상되었습니다.



### Retrieval Augmented Generation for Dynamic Graph Modeling (https://arxiv.org/abs/2408.14523)
Comments:
          Under review

- **What's New**: 이 논문은 동적 그래프 모델링(Dynamic Graph Modeling)을 위한 새로운 프레임워크인 Retrieval-Augmented Generation for Dynamic Graph Modeling(RAG4DyG)을 소개합니다. 기존 방법론의 한계를 극복하고, 각 노드(nodal)의 관점을 확장하기 위해 유사한 패턴을 가진 예시를 검색하여 활용합니다.

- **Technical Details**: RAG4DyG은 두 가지 주요 고난이도 문제를 해결합니다: (1) 동적 그래프 샘플과 관련된 고품질 예시를 식별하고 검색해내는 방법, (2) 이러한 예시를 효과적으로 통합하여 동적 그래프 모델링을 향상시키는 방법. 이를 위해 시간 인식 기반의 대조 학습(time-aware contrastive learning) 모듈과 그래프 융합(graph fusion) 전략을 사용하여 예시를 통합합니다.

- **Performance Highlights**: 다양한 실제 데이터셋을 기반으로 한 실험을 통해 RAG4DyG의 동적 그래프 모델링 성능이 입증되었습니다. 이 접근법은 서로 다른 도메인에서 동적 패턴을 효과적으로 분석하고 예측하는 데 유리한 성능을 보입니다.



### Towards Graph Prompt Learning: A Survey and Beyond (https://arxiv.org/abs/2408.14520)
Comments:
          19 pages, 2 figures

- **What's New**: 그래프 프롬프트 학습(Graph Prompt Learning)은 Prompt Learning의 원칙을 그래프 도메인에 적용한 혁신적인 확장입니다. 이 논문은 이 새로운 분류법을 제안하며, 그래프 프롬프트 학습의 최근 연구를 포괄적으로 정리합니다. 특히, 2023년 이후의 연구를 반영하여 최신 발전을 반영하고 있습니다.

- **Technical Details**: 이 논문에서는 그래프 프롬프트 학습의 여러 방법론을 분류하며, 토큰 설계 원칙, 작업 정렬 메커니즘, 프롬프트 튜닝 방법 등을 포함합니다. 또한 동질 및 이질 그래프 프롬프트의 일반 방법과 이들의 하류 응용 사례를 살펴봅니다. Graph Neural Networks(GNNs)에 기초한 메시지 전달(message passing) 접근을 통해 노드 속성을 통합하는 방법을 논의합니다.

- **Performance Highlights**: 이 논문은 100개 이상의 관련 연구를 정리하며, 그래프 데이터에서의 프롬프트 기반 학습이 실제 문제 해결에 어떻게 기여하는지를 설명합니다. 그래프 프롬프트 학습은 대규모 프리트레인(pre-trained) 모델을 효과적으로 활용하여 적은 양의 라벨링된 데이터로도 신규 작업을 수행할 수 있는 가능성을 보여줍니다. 이 접근 방식은 대규모 데이터셋에서도 강건한 성능을 유지할 수 있도록 돕습니다.



### A Joint Learning Model with Variational Interaction for Multilingual Program Translation (https://arxiv.org/abs/2408.14515)
Comments:
          Accepted by the 39th IEEE/ACM International Conference on Automated Software Engineering (ASE 2024)

- **What's New**: 이 논문에서는 여러 프로그래밍 언어 간의 코드 변환을 위한 통합 모델을 공동 학습하는 것이 기존의 언어 쌍별 데이터로 개별적으로 학습하는 것보다 우수하다고 주장합니다. 이를 위해 'Variational Interaction for Multilingual Program Translation (VIM-PT)'라는 새로운 방법론을 제안합니다.

- **Technical Details**: VIM-PT는 코드의 언어 공유 기능과 언어 특정 기능을 분리하여 처리합니다. 변별적 추론(variational inference)과 상호 정보(interaction information)를 활용하여 새로운 하한을 사용해 코드 변환을 수행하며, 조건부 생성(conditional generation)을 통해 목표 언어의 언어 특정 특징을 샘플링하여 코드 변환을 가능하게 합니다.

- **Performance Highlights**: VIM-PT는 통합 모델로서 여러 가지 구현에서 언어 공유 정보를 더 정확하고 완전하게 포착하며, 비평행 및 부분적으로 누락된 데이터를 효과적으로 활용합니다. 이를 통해 언어 간 의미의 분포 이동(distribution shift) 문제를 해결하고, 다양한 번역 쌍에 걸쳐 총 매개변수를 줄여 배포의 편리성을 높입니다. VIM-PT는 7개 일반 프로그래밍 언어 데이터셋에서 기존 최첨단 접근법보다 성능 개선을 보였습니다.



### Variational autoencoder-based neural network model compression (https://arxiv.org/abs/2408.14513)
- **What's New**: 이 논문은 Variational Autoencoders (VAEs)를 기반으로 한 신경망 모델 압축 방법을 탐구하는 것을 목표로 합니다. 기존의 모델 압축 기술과 비교했을 때, VAE를 통해 잠재 공간(latent space)을 모델 압축의 표현으로 사용함으로써 압축률을 개선할 수 있음을 보여줍니다.

- **Technical Details**: 이 연구에서는 MNIST에서의 인식에 대해 FNN(Feedforward Neural Network), CNN(Convolutional Neural Network), RNN(Recurrent Neural Network), LSTM(Long Short-Term Memory)과 같은 다양한 신경망 모델을 압축 목표로 삼았습니다. VAE는 오토인코더(Autoencoder)의 변형으로, 인코더와 디코더를 구성하고 있으며, 인코더는 모델 파라미터를 잠재 공간(latent space)으로 축소할 수 있습니다.

- **Performance Highlights**: VAEs를 이용한 압축 방법은 전통적인 압축 방법들(예: pruning, quantization)에 비해 더 높은 압축률을 달성하면서도 모델의 정확도가 크게 저하되지 않음을 실험적으로 입증했습니다.



### LLMs as Zero-shot Graph Learners: Alignment of GNN Representations with LLM Token Embeddings (https://arxiv.org/abs/2408.14512)
- **What's New**: 이번 논문에서는 Token Embedding-Aligned Graph Language Model (TEA-GLM)이라는 새로운 프레임워크를 제안합니다. TEA-GLM은 GNN(Graph Neural Network)의 표현을 대형 언어 모델의 토큰 임베딩과 정렬하여, cross-dataset 및 cross-task 제로샷(zero-shot) 학습을 가능하게 합니다.

- **Technical Details**: TEA-GLM의 기본 구성 요소는 GNN과 LLM(Large Language Model)으로, GNN은 그래프에서 노드 표현을 도출하고 LLM은 노드 분류 및 링크 예측과 같은 제로샷 작업을 수행합니다. 이 프레임워크는 두 가지 주요 단계로 구성됩니다: GNN의 자기지도 학습과 GNN 표현을 고정된 수의 그래프 토큰 임베딩으로 변환하기 위한 선형 프로젝터(training a linear projector)입니다.

- **Performance Highlights**: TEA-GLM은 다양한 그래프 작업에서 통합 지침을 설계하여 모델의 일반화 기능을 향상시키며, 실험 결과에 따르면, TEA-GLM은 보이지 않는 데이터셋과 작업에 대해 최신 방법들보다 우수한 성능을 나타내었습니다.



### Cost-Aware Uncertainty Reduction in Schema Matching with GPT-4: The Prompt-Matcher Framework (https://arxiv.org/abs/2408.14507)
- **What's New**: 이 논문은 GPT-4를 사용하여 스키마 매칭에서 발생하는 불확실성을 줄이는 방법을 제안합니다. 기존의 크라우드워커 대신 GPT-4를 활용하여 후보 매칭 세트를 쿼리하고 검증하는 방식을 통해 더 정확한 대응을 확인합니다.

- **Technical Details**: 우리는 GPT-4에 대해 두 가지 맞춤형 프롬프트, 즉 Semantic-match 및 Abbreviation-match를 설계하여 DeepMDatasets와 Fabricated-Datasets에서 최첨단 결과를 달성했습니다. 새로운 프레임워크 Prompt-Matcher를 통해 여러 자동 스키마 매칭 알고리즘의 통합 과정에서 불확실성을 줄이고 복잡한 파라미터 선택을 최적화합니다.

- **Performance Highlights**: GPT-4의 우리의 프롬프트를 사용하여 DeepMDatasets에서 100%의 재현율을 기록하고 Fabricated-Datasets에서 91.8%의 재현율을 달성했습니다. 또한, 우리는 예산 제약 내에서 효과적인 결과를 제공하는 비용 인식 솔루션을 설계했습니다.



### Empowering Pre-Trained Language Models for Spatio-Temporal Forecasting via Decoupling Enhanced Discrete Reprogramming (https://arxiv.org/abs/2408.14505)
- **What's New**: 본 논문에서는 시공간(time series) 예측을 위한 새로운 프로그래밍 프레임워크인 RePST를 제안합니다. 기존 접근 방식들이 공간적 의존성과 내재적 주파수 구성요소를 처리하는 데 한계를 보이는 반면, RePST는 이러한 문제를 해결하기 위해 주파수 도메인에서 시공간 동역학을 분리하는 접근 방식을 사용합니다.

- **Technical Details**: RePST 프레임워크는 Fourier 분석과 구조적 확산(operator)을 통해 입력된 시공간 데이터를 내재적 및 확산 신호로 분해합니다. 이를 통해 PLM(Pre-trained Language Models)이 더 잘 이해할 수 있는 데이터 표현을 생성하며, 차별화된 방식으로 확대된 어휘 공간에서 관련된 정보만을 선택하는 전략을 도입하여 정보 병목 현상을 방지합니다.

- **Performance Highlights**: 여러 실제 데이터셋 부문에서 수행한 광범위한 실험을 통해 RePST가 기존 최신 기술보다 뛰어난 성능 향상을 demonstrated하였고, 특히 데이터가 부족한 상황에서도 강력한 일반화 능력을 발휘함을 확인했습니다.



### Is Functional Correctness Enough to Evaluate Code Language Models? Exploring Diversity of Generated Codes (https://arxiv.org/abs/2408.14504)
Comments:
          15pages, 6 figures, 8 tables

- **What's New**: 본 연구는 코드 생성 기능 평가에서 다양성(diversity)을 중요 기준으로 강조합니다. 기존 연구들은 기능적 정확성에만 초점이 맞춰져 있었으나, 본 논문은 생성된 코드의 다양성을 평가하는 체계적인 접근 방식을 제안합니다.

- **Technical Details**: 저자들은 코드 유사성을 측정하기 위한 새로운 지표인 Sim@K 및 CSim@K을 도입하였습니다. 또한, 기능적 정확성을 평가하기 위한 DPass@K을 제안하여 코드의 다양성과 정확성을 함께 측정할 수 있도록 합니다. 본 연구는 다양한 모델 크기, 온도 매개변수, 훈련 기법, 프롬프트 전략 등 다양한 변수가 코드 품질에 미치는 영향을 심층적으로 분석합니다.

- **Performance Highlights**: 연구 결과, 입력 문제의 난이도가 증가할수록 생성된 코드의 다양성도 증가하는 경향을 보였으며, Instruction-tuned 모델들이 기본 모델보다 코드의 유사성을 더 많이 생성한다는 것을 발견했습니다. 또한, 온도를 감소시켜 몇 가지 사고 과정을 활용하면 코드 간의 유사성을 증가시킬 수 있습니다.



### Applying graph neural network to SupplyGraph for supply chain network (https://arxiv.org/abs/2408.14501)
Comments:
          8 pages, 5 figures

- **What's New**: 이번 연구는 SupplyGraph라는 공급망 데이터셋을 분석하고, 데이터 품질 보증 프로세스 및 머신러닝 모델의 하이퍼파라미터에 대한 명확성을 더했습니다. 또한, GNN이 공급망 예측에서 유용하다는 것을 입증하는 과정이 포함되어 있습니다.

- **Technical Details**: 연구에 사용된 SupplyGraph 데이터셋은 40개의 제품 노드와 여러 에지 유형을 포함하며, 템포럴 피처를 재정의하여 수요 예측을 수행하는데 초점을 맞췄습니다. 세 가지 모델(Multilayer Perceptions, Graph Convolution Network, Graph Attention Network)을 비교 분석하였으며, 각 모델의 성능을 MSE(Mean Squared Error)와 SE(Squared Error)에 대한 통계 분석을 통해 평가했습니다.

- **Performance Highlights**: GAT 모델이 가장 높은 성능을 보였으며, GCN과 MLP 모델보다 통계적으로 유의미한 차이를 보였습니다(α=0.05). GNN 구조를 활용하는 것이 공급망 모델링에서 중요한 효과를 나타냈습니다.



### SHEDAD: SNN-Enhanced District Heating Anomaly Detection for Urban Substations (https://arxiv.org/abs/2408.14499)
Comments:
          12 pages, 5 figures, FMEC2024

- **What's New**: SHEDAD는 District Heating (DH) 시스템의 이상 감지를 개선하기 위해 설계된 새로운 방법입니다. 이 방법은 DH 네트워크의 상대적인 토폴로지를 근사화하고, 민감한 정보(예: 변전소 위치)를 공개하지 않고도 로컬 이상 감지를 가능하게 합니다.

- **Technical Details**: SHEDAD는 multi-adaptive k-Nearest Neighbor (k-NN) 그래프를 활용하여 초기 이웃 생성을 개선하고, 노이즈를 줄이는 병합 기법을 도입합니다. 이 방법은 Median Absolute Deviation (MAD)와 수정된 z 점수를 사용하여 이상 변전소를 플래그합니다. SHEDAD는 두 가지 이상 범주를 식별합니다: 공급 온도와 변전소 성능과 관련된 이상입니다.

- **Performance Highlights**: SHEDAD는 전통적인 클러스터링 방법보다 우수한 성능을 보이며, intra-cluster 분산과 거리를 유의미하게 낮추었습니다. 30개의 이상 변전소를 식별하였으며, 민감도는 약 65%, 특이도는 약 97%에 도달했습니다. 이는 네트워크 내에서 저성능 변전소에 대한 보다 집중적이고 효과적인 유지 보수 작업을 가능하게 합니다.



### A New Era in Computational Pathology: A Survey on Foundation and Vision-Language Models (https://arxiv.org/abs/2408.14496)
Comments:
          Initial Version

- **What's New**: 최근의 AI(인공지능) 발전은 Computational Pathology (CPath) 분야를 근본적으로 변화시켰습니다. 기초 모델(FMs) 및 비전-언어 모델(VLMs)을 통합하여 병리학자들의 진단 작업 흐름을 혁신하고 있습니다.

- **Technical Details**: 기초 모델(FMs)은 자가 지도 학습(self-supervised learning, SSL) 기법을 통해 다양한 과제에 적응할 수 있는 표현 공간을 학습합니다. 이와 더불어 VLMs는 자연어로 작성된 병리학 보고서를 활용하여 기존 모델의 성능을 개선하고 자연어 형태로 예측을 생성할 수 있습니다. 이 연구에서는 FMs와 VLMs의 구조 및 훈련 방식에 대한 자세한 정보를 제공합니다.

- **Performance Highlights**: FMs와 VLMs의 통합은 AI 병리학자가 되어 다양한 작업을 수행할 수 있는 가능성을 보여주며, 이는 최근 연구에서 나타난 성과들에 기반하고 있습니다. 또한 CPath 분야에 대한 최근 연구 증가를 통해 이들 모델의 중요성이 부각되고 있습니다.



### Knowledge Graph Modeling-Driven Large Language Model Operating System (LLM OS) for Task Automation in Process Engineering Problem-Solving (https://arxiv.org/abs/2408.14494)
Comments:
          Accepted for Publication by Association for the Advancement of Artificial Intelligence, Fall Symposium Series

- **What's New**: PEOA(프로세스 엔지니어링 운영 어시스턴트)라는 AI 기반의 프레임워크를 소개하며, 이는 화학 및 프로세스 산업의 복잡한 문제를 해결하는 데 초점을 맞추고 있습니다. 이 프레임워크는 메타 에이전트에 의해 조정되는 모듈형 아키텍처를 가지고 있습니다.

- **Technical Details**: PEOA 프레임워크는 복잡한 문제를 서브 태스크로 분해하고 각 태스크를 수행하기 위한 적절한 전문가 모델을 선택하여 정밀한 해결책을 제공합니다. 주요 기술로는 성능 향상을 위한 속성 그래프(Property Graph)를 사용한 고급 지식 모델링이 포함되어 있습니다. 또한, GPT-4를 활용한 교사-학생 전이 학습 접근 방식을 사용하여 도메인 적응을 위해 액션 생성기와 전문 모델을 미세 조정합니다.

- **Performance Highlights**: PEOA 프레임워크는 계산 자동화, 프로토타입 생성 가속화, 산업 공정에 대한 AI 보강 의사 결정을 제공함으로써 프로세스 엔지니어링의 능력을 획기적으로 향상시킵니다. 다양한 엔지니어링 작업에 대한 선도적인 독점 언어 모델과 비교했을 때 프레임워크의 유효성을 평가했습니다.



### Handling abort commands for household kitchen robots (https://arxiv.org/abs/2408.14480)
- **What's New**: 이번 연구는 로봇에게 주어진 중단(abort) 명령을 처리하는 새로운 솔루션을 제안합니다. 특히 주방 로봇을 예로 들어, 이전에 받은 명령을 우아하게 취소할 수 있는 방식으로 행동 시퀀스를 찾는 계획(Planning) 기법을 사용합니다.

- **Technical Details**: 제안된 시스템은 주방 활동 및 행동을 모델링하기 위해 PDDL(Planning Domain Definition Language)을 사용하며, 온라인 온톨로지와 지식 그래프(DBPedia)로부터 얻은 지식으로 풍부하게 구성됩니다. 이러한 방식으로, 로봇은 중단 명령 수신 시, 환경을 깨끗한 상태로 유지하기 위해 행동을 재구성하고 다시 계획할 수 있습니다.

- **Performance Highlights**: 실험 결과, 효율적인 계획 수립이 가능하며, 로봇이 주방에서 명령을 취소할 때 안정적인 상태를 유지할 수 있는 능력이 향상되었습니다. 이 연구는 로봇의 안전성, 견고성 및 신뢰성을 크게 향상시킬 수 있는 가능성을 보여줍니다.



### Uncertainty Quantification in Alzheimer's Disease Progression Modeling (https://arxiv.org/abs/2408.14478)
Comments:
          This work was done as part of degree requirements for the authors in 2021-2022

- **What's New**: 이번 연구에서는 알츠하이머병(Alzheimer's Disease, AD) 진행 상황을 예측하기 위해 MC Dropout, 변이 추정(Variational Inference), 마르코프 체인 몬테 카를로(Monte Carlo Markov Chain, MCMC), 앙상블 학습(Ensemble Learning) 등 네 가지 모델의 성능을 비교하여 신뢰 구간(confidence bounds)을 제공하는 새로운 예측 모델을 제안합니다.

- **Technical Details**: 연구는 신경영상과 임상 측정값, 그리고 인지 테스트 점수를 포함하는 ADNI(Alzheimer’s Disease Neuroimaging Initiative) 데이터베이스의 TADPOLE 장기 데이터셋을 기반으로 하며, 512명의 환자를 대상으로 4년 간의 MMSE 점수를 예측합니다. 각 모델에 대한 불확실성 정량화 방법(uncertainty quantification, UQ)을 비교하여 각각의 예측 모델의 성능을 검증합니다.

- **Performance Highlights**: MC Dropout과 MCMC 모델은 훈련 데이터에 노이즈가 있는 경우에도 신뢰할 수 있는 예측 결과를 보이며, 올바르게 보정된(predictions well-calibrated) 정밀한 예측을 생산할 수 있음을 보여줍니다. 이는 알츠하이머병의 조기 진단과 관리에 중요한 기여를 할 것으로 기대됩니다.



### Tipta uzmanlik sinavinda (tus) buyuk dil modelleri insanlardan daha mi basarili? (https://arxiv.org/abs/2408.12305)
Comments:
          9 pages, in Turkish language, 8 figures

- **What's New**: 최근 자연어 처리(natural language processing)와 인공지능(ai)의 발전에 힘입어 의료 교육 및 평가에서 인공지능의 잠재력이 드러났습니다. 이 연구는 터키의 Tıpta Uzmanlık Sınavı(이하 TUS)에서 인공지능 모델의 성과를 평가하였습니다.

- **Technical Details**: 연구에서는 Gemini, ChatGPT-4 및 ChatGPT-4o 3가지 인공지능 모델이 2021년 1학기 TUS의 240개 문제에 대한 간단한 질문을 해결하는 성능을 비교했습니다. CMST(Clinical Medical Sciences)와 BMST(Basic Medical Sciences) 섹션에서 각각의 모델이 맞춘 질문 수를 기록하고 평가하였습니다.

- **Performance Highlights**: 상대적으로 ChatGPT-4o가 CMST에서 117문항, BMST에서 107문항을 맞추어 가장 높은 성과를 보였으며, 이는 해당 분야의 인공지능 모델들이 의료 교육 및 평가에서 높은 정확도를 달성할 수 있음을 나타냅니다. 이 연구는 인공지능 모델의 교육 활용 가능성을 강조하며, 교육 및 평가 과정의 질 향상에 기여할 수 있음을 보여주었습니다.




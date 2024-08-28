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




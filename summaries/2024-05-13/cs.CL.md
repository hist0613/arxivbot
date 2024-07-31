### Linearizing Large Language Models (https://arxiv.org/abs/2405.06640)
- **What's New**: 이 논문은 Recurrent Neural Networks (RNNs)를 향상시키기 위한 새로운 방법인 Scalable UPtraining for Recurrent Attention (SUPRA)를 소개합니다. SUPRA는 고성능 Large Language Models (LLMs)을 RNN으로 전환하는 새로운 접근 방식을 사용하여 기존의 Transformer 모델들을 Recurrent 모델로 업트레이닝하는 것을 제안합니다.

- **Technical Details**: 저자들은 소프트맥스 (softmax) 주의 메커니즘을 선형 커널과 정규화 전략으로 대체하여 성능이 뛰어난 LLM을 RNN으로 전환합니다. 이 방법은 특히 적은 계산 비용으로도 기존 트랜스포머 (Transformer) 모델들을 효율적으로 재사용할 수 있는 장점이 있습니다. 고비용의 대규모 사전 훈련 데이터셋 대신, 공개 데이터셋에서 적은 양의 훈련으로도 경쟁력 있는 결과를 달성할 수 있음을 보여줍니다.

- **Performance Highlights**: SUPRA는 사전 훈련된 최고 성능의 재래적 RNN 모델들과 경쟁할 수 있는 성능을 보이며, 표준 벤치마크에서 경쟁력 있는 성능을 나타냈습니다. 이는 기존 트랜스포머 모델의 강력한 사전 훈련 데이터와 성능을 활용하면서도 훈련 비용의 5%만을 요구하는 것입니다. 그러나, 이 연구에서도 지적한 것처럼, 'in-context learning'과 'long-context modeling'에 있어서는 아직도 해결해야 할 과제가 남아있습니다.



### Explaining Text Similarity in Transformer Models (https://arxiv.org/abs/2405.06604)
Comments:
          Accepted to NAACL 2024

- **What's New**: 변환기(Transformer)가 자연어 처리(NLP) 과제에서 최첨단 모델로 자리잡으면서, 이들의 예측을 이해하고 설명하는 것이 점점 중요해지고 있습니다. 정보 검색과 같은 비지도 학습 응용 프로그램에서는 기초 모델(Foundation Model) 표현을 기반으로 구축된 유사성 모델이 널리 사용되고 있으나, 그 내부 예측 메커니즘이 대체로 불투명합니다. 최근 설명 가능한 AI(Explainable AI)의 발전으로 변환기의 설명을 개선함으로써 이러한 한계를 완화할 수 있게 되었습니다.

- **Technical Details**: 이 연구에서는 층별 관련성 전파(Layer-wise Relevance Propagation, LRP)를 활용하여 변환기의 설명을 향상시키는 방법을 제시합니다. 특히, BiLRP(Bilinear LRP)라는 확장 기법을 사용하여 이차원 설명(Second-order Explanations)을 계산하였고, 이를 통해 NLP 모델에서 유사성을 주도하는 특징 상호작용을 조사하였습니다.

- **Performance Highlights**: Grammatical interactions, multilingual semantics, and biomedical text retrieval 등 세 가지 코퍼스 수준(Corpus-Level) 사례 연구에서 결과적인 설명을 검증하고 그 유틸리티를 입증했습니다. 이러한 분석을 통해 다양한 의미적 유사성 과제와 모델에 대한 깊은 이해를 제공하며, 새롭게 등장한 설명 가능한 AI 기술이 심층적 분석체계 및 코퍼스 수준의 통찰을 가능하게 하는 방법을 강조합니다.



### What Can Natural Language Processing Do for Peer Review? (https://arxiv.org/abs/2405.06563)
- **What's New**: 매년 생성되는 과학 기사의 수는 급속도로 증가하고 있으며, 이러한 기사들의 품질 관리는 과학자들뿐만 아니라 공공의 이익을 위해 중요합니다. 본 논문은 기계 보조 피어 리뷰(peer review)를 위한 NLP(Natural Language Processing, 자연어 처리)의 미래 연구 노력의 기초를 제공하고자 합니다. 특히 AI 컨퍼런스에서의 리뷰 과정을 예시로 들어 각 단계를 자세히 설명하고 NLP의 도움을 받을 수 있는 기회와 도전 과제를 논의합니다.

- **Technical Details**: 피어 리뷰가 주로 텍스트 기반인 점을 고려할 때, NLP는 리뷰 과정에서 큰 잠재력을 가지고 있습니다. 이 논문은 AI 대회에서의 원고 제출부터 카메라 준비(camera-ready) 수정까지의 과정을 상세히 설명하며, NLP가 어떻게 도움을 줄 수 있는지와 그 한계를 구체적으로 조명합니다. 또한 데이터 획득 및 라이선싱, 운영화(operationalization), 실험 설계(experimentation) 및 윤리적 문제 등 NLP를 위한 큰 도전 과제를 다룹니다.

- **Performance Highlights**: NLP를 사용하여 피어 리뷰의 효율성을 개선하고 오류를 줄일 수 있는 가능성을 제시하며, 이를 위해 큰 언어 모델(LLMs, Large Language Models)의 사용이 논의됩니다. 기존 연구를 통해 예시를 들면서 NLP가 피어 리뷰 과정에서 구체적으로 어떤 도움을 줄 수 있을지 탐색합니다. 또한, 연구 커뮤니티, 정책 입안자(policy makers), 자금 조달 기관들이 함께 노력하여 NLP를 위한 기계 보조 피어 리뷰 연구를 발전시킬 수 있는 행동 계획을 제안합니다.



### ADSumm: Annotated Ground-truth Summary Datasets for Disaster Tweet Summarization (https://arxiv.org/abs/2405.06551)
- **What's New**: ADSumm은 재난 사건 요약을 위해 주석 처리된(annotated) 기계 요약(ground-truth summaries)을 추가하여 정제하고 확장된 데이터셋을 제공합니다. 여기에는 자연 및 인공 재난이 포함되며, 다양한 국가의 사건을 포함하고 있습니다. 이 연구는 추가된 데이터셋이 감독 학습(supervised learning) 접근 방식의 향상을 제공하여 8-28%의 ROUGE-N F1 점수 개선을 보여주었습니다.

- **Technical Details**: ADSumm은 입력 트윗(tweet)에 대해 세 가지 추가 기능을 소개합니다: 카테고리 라벨(category labels), 키-프레이즈(key-phrases), 그리고 관련성 라벨(relevance labels). 이러한 기능들은 각각 트윗이 재난 문맥에서 어떤 범주에 속하는지, 어떻게 중요한지, 그리고 자동 요약 결정에 대한 설명을 제공합니다. 또한, 이 연구는 재난별, 위치별 변이를 고려하여 요약 알고리즘(summarization algorithms)의 로버스트함을 높이는 데 사용됩니다. 무엇보다도 이 논문은 기존에 공개적으로 이용 가능하지 않았던 주석 절차에 대해 구체적으로 제공하며, 이는 질 높은 요약 생성을 위한 중요한 기준을 마련합니다.

- **Performance Highlights**: 실험 분석과 성능 비교는 ADSumm이 추가된 데이터셋이 기존의 감독된 요약 접근 방식보다 우수한 성능을 보인다는 것을 보여줍니다. 본 논문에서는 새로운 데이터셋의 특성을 활용한 다양한 자연어 처리(NLP) 작업에 대한 유틸리티 분석을 제공하며, 이는 분류 정확도, 요약 다양성 및 요약 품질 평가 등에 기여합니다. 또한 새로운 데이터셋이 기존의 상태 기술(state-of-the-art) 요약 접근 방식에 비해 어떻게 개선했는지에 대한 벤치마킹을 포함합니다.



### Mitigating Hallucinations in Large Language Models via Self-Refinement-Enhanced Knowledge Retrieva (https://arxiv.org/abs/2405.06545)
- **What's New**: 이 연구에서는 의료 분야에서 대규모 언어 모델(LLMs)의 반응의 사실성을 높이기 위한 새로운 방법으로 Self-Refinement-Enhanced Knowledge Graph Retrieval (Re-KGR)을 제안합니다. 이 방법은 특히 hallucination 즉, 관련 없거나 가짜 정보를 생성하는 경향을 줄이는데 초점을 맞추고 있습니다.

- **Technical Details**: Re-KGR은 각 토큰의 다음 토큰 예측 확률 분포(attribution of next-token predictive probability distributions)를 이용하여 hallucination 가능성이 높은 토큰을 우선적으로 식별합니다. 이를 통해 지식 그래프(knowledge graphs, KGs)에서 관련 지식 트리플(knowledge triples)을 검색하고, 이를 세밀하게 정제함으로써 검증 라운드를 줄입니다. 또한, 검색된 지식을 사용하여 후처리 단계에서 부정확한 콘텐츠를 수정하여 생성된 반응의 진실성을 높입니다.

- **Performance Highlights**: 의료 데이터셋에서의 실험 결과, Re-KGR 방법은 다양한 기초 모델들(foundation models)에서 LLMs의 사실적 능력을 향상시킬 수 있음을 보여줍니다. 특히, 진실성(truthfulness)에 대한 가장 높은 점수를 기록하였습니다.



### ATSumm: Auxiliary information enhanced approach for abstractive disaster Tweet Summarization with sparse training data (https://arxiv.org/abs/2405.06541)
- **What's New**: 트위터에서 발생하는 재난 상황에 관한 정보의 방대함으로 인해 사용자가 중요하고 관련된 정보를 수동으로 판별하는 것이 어려워졌습니다. 이러한 문제를 해결하기 위해, 새롭게 제안된 Abstractive Tweet Summarizer (ATSumm)는 데이터 희소성 문제를 다뤄 효과적인 요약을 제공합니다.

- **Technical Details**: 이 연구에서는 기존 연구에서 사용된 방법론을 추출적 요약 단계에 적용하고, 추상적 요약 단계를 위해 새로운 Auxiliary Pointer Generator Network (AuxPGN) 모델을 소개했습니다. 이 모델은 입력 트윗에서 중요한 단어나 구문(예: key-phrase)과 그 중요도 점수를 포함하는 'Key-phrase attention'이라는 특별한 주의 메커니즘(attention mechanism)을 사용하여 보조 정보를 활용합니다.

- **Performance Highlights**: ATSumm은 13개의 재난 데이터셋에 걸쳐 10개의 최신 기법(state-of-the-art approaches)과 비교하여 성능을 평가했으며, ROUGE-N F1-스코어에서 4-80%의 개선을 나타내어 뛰어난 성능을 보여주었습니다.



### Prompting Large Language Models with Knowledge Graphs for Question Answering Involving Long-tail Facts (https://arxiv.org/abs/2405.06524)
- **What's New**: 이 연구에서는 긴 꼬리 사실에 관한 질문에 대답하는 데 필요한 지식을 갖춘 비파라메트릭(non-parametric) 지식의 효과를 분석합니다. 우리는 새로운 벤치마크인 Long Tail Generation (LTGen)을 제안하며, 이는 비파라메트릭 지식과 결합된 Large Language Models (LLMs)의 성능을 평가하기 위한 것입니다. LTGen은 기존의 PopQA 데이터셋의 한계를 극복하고, 더 다양하고 복잡한 관계를 포함하는 질문을 생성합니다.

- **Technical Details**: LTGen 벤치마크는 자동화된 파이프라인을 사용하여 생성되며, 긴 꼬리 엔티티 (long-tail entities) 선택, 트리플 검색 (triples retrieval), 그리고 샘플 생성의 세 단계로 구성됩니다. 이 벤치마크는 파라메트릭 지식만을 사용할 때보다 비파라메트릭 지식을 프롬프트(prompt)하는 것이 LLM의 성능을 뚜렷이 향상시킨다는 것을 보여줍니다. 특히, 지식 그래프 (Knowledge Graphs, KGs)에 기반한 구조적 지식이 텍스트 패시지 기반의 지식보다 더 효과적입니다.

- **Performance Highlights**: KG 트리플을 사용한 프롬프팅이 텍스트 패시지 기반 프롬프팅보다 우수한 성능을 보이는 경우가 많으며, KG와 문서 모두를 사용하는 프롬프팅은 지식의 범위는 일관되게 향상시키지 않지만 생성된 내용에서의 환각을 크게 줄일 수 있습니다. 이러한 접근 방식은 NLI-based metric 점수에서 여러 설정에 걸쳐 가장 높은 점수를 달성합니다.



### Aspect-based Sentiment Evaluation of Chess Moves (ASSESS): an NLP-based Method for Evaluating Chess Strategies from Textbooks (https://arxiv.org/abs/2405.06499)
Comments:
          accepted in the 10th Games and NLP 2024 workshop at LREC 2024

- **What's New**: 체스 도메인은 의사결정(decision-making)을 포함한 실제 세계의 도전 과제를 모방하는 인공지능(AI) 시스템을 만드는 데 적합합니다. 이번 연구에서는 체스 교재에서 참조된 여러 수(moves) 사이의 복잡한 관계를 조사하고, 수행 행동 구문(move-action phrases)에서 파생된 체스 지식을 포착하는 새로운 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 Aspect-Based Sentiment Analysis (ABSA) 방법을 수정하여 텍스트 기반 체스 수를 평가하는 수단으로 사용하는 것을 조사합니다. ABSA는 참조된 체스 수와 관련된 감정(sentiment)을 평가하는 데 진전을 대표합니다. 이 접근법은 수행 행동 구문에서 통찰을 추출함으로써 더 세밀하고 맥락적으로 인식하는 '체스 수 기반' 감정 분류(sentiment classification)를 제공하려고 합니다.

- **Performance Highlights**: 실증적 실험과 분석을 통해 우리는 ABSA 모델을 세밀하게 조정하고 평가하였으며, 체스 영역 내에서 측면 기반 감정 분류를 진전시키는 접근법의 효율성을 확인하는 결과를 제시합니다. 이 연구는 기계에 의한 게임 플레이 분야에 기여하며, 전략 게임의 맥락을 이해하기 위해 NLP(Natural Language Processing) 기술을 활용하는 실용적 적용 가능성을 보여줍니다.



### LyS at SemEval-2024 Task 3: An Early Prototype for End-to-End Multimodal Emotion Linking as Graph-Based Parsing (https://arxiv.org/abs/2405.06483)
Comments:
          Accepted at SemEval 2024

- **What's New**: 이 논문은 다자간 대화에서 감정 원인 분석(Multimodal Emotion Cause Analysis)을 위한 SemEval 2024 Task 3에 참가하며 개발한 초기 프로토타입 시스템을 설명합니다. 주로 의존성 파싱(dependency parsing)에서 온 그래프 기반 방법을 사용하여 대화에서 감정의 원인 관계를 파악하는 데 초점을 맞추었습니다.

- **Technical Details**: 연구팀은 다자간 대화 데이터를 맥락화(contextualizing)하기 위한 신경 변환기 기반 인코더(neural transformer-based encoder)와 인과 그래프(causal graph)의 인접 행렬 점수를 생성하는 그래프 기반 디코더(graph-based decoder)를 포함하는 모델을 개발했습니다. 이 모델은 텍스트 입력만 사용하여 Subtask 1에서 15개 유효한 공식 제출 중 7위를 차지했습니다.

- **Performance Highlights**: 이 시스템은 텍스트 기반 입력에서 Subtask 1에 참여하여 15개의 유효한 공식 제출물 중 7위를 달성했으며, 평가 후 멀티모달 입력을 사용한 Subtask 2에도 참여했습니다. 멀티모달(multi-modal) 입력을 통한 감정 원인 분석에 대한 접근 방식을 확장하는 데 중점을 둔 추가 논의가 행해졌습니다.



### Are EEG-to-Text Models Working? (https://arxiv.org/abs/2405.06459)
- **What's New**: 이 연구는 열린 어휘(open-vocabulary) EEG-to-Text 번역 모델들을 심도 있게 분석합니다. 이전 연구들이 평가 중에 암시적인 교사 강요(implicit teacher-forcing)를 사용하여 성능 지표를 인위적으로 부풀렸다는 중요한 한계를 지적합니다. 또한, 순수한 잡음 데이터(noise inputs)의 성능을 비교하는 중요한 벤치마크가 부재했다는 점을 드러냈습니다.



### E2TP: Element to Tuple Prompting Improves Aspect Sentiment Tuple Prediction (https://arxiv.org/abs/2405.06454)
- **What's New**: 이 논문에서는 새로운 'Element to Tuple Prompting (E2TP)' 접근 방식을 소개하고 있습니다. 기존의 감성 분석 연구들이 주로 전체 텍스트의 성분을 일괄적으로 예측하는 방식을 사용한 반면, E2TP는 먼저 단일 요소를 예측하고 이를 이용해 관련 튜플(tuple)로 맵핑하는 두 단계 아키텍처를 사용하여 문제를 더욱 세분화하여 접근합니다.

- **Technical Details**: E2TP는 문제 해결을 위한 인간의 접근 방식에서 영감을 받아, 첫 번째 단계에서는 개별 요소를 예측하고, 두 번째 단계에서는 이러한 예측된 요소들을 튜플로 맵핑합니다. 이 과정에서 세 가지 다양한 E2TP 방식 (E2TP($diet$), E2TP($f_1$), E2TP($f_2$))이 설계되어, 학습 과정을 용이하게 하며, 이는 크로스 도메인(cross-domain) 상황에서도 효과적이며 일반화 가능성을 입증합니다.

- **Performance Highlights**: 다양한 벤치마크(benchmarks)에 대한 광범위한 분석을 통해, E2TP는 거의 모든 경우에서 새로운 최고 성과(state-of-the-art results)를 달성했다는 점을 보여줍니다. 이는 기존 방식들보다 E2TP가 더 우수한 성능을 제공함을 의미합니다.



### Improving Instruction Following in Language Models through Proxy-Based Uncertainty Estimation (https://arxiv.org/abs/2405.06424)
Comments:
          Accepted to ICML 2024

- **What's New**: 새로운 불확실성 인식 보상 모델(Uncertainty-aware Reward Model, URM)을 제안하여 언어 모델의 지시 사항에 따른 응답 품질을 평가합니다. 이 모델은 베이지안 근사(Bayesian approximation)를 사용하여 응답의 품질에 대한 견고한 불확실성 추정을 도입합니다.

- **Technical Details**: URM은 선호도 데이터 세트(preference datasets)로 훈련되며, 응답에 대한 보상을 점수화할 뿐만 아니라 그 내재된 불확실성을 평가합니다. 이는 언어 모델 훈련에서 데이터 큐레이션을 정제하고 정책 최적화 목표(policy optimization objectives)를 개선하는 데 도움이 됩니다.

- **Performance Highlights**: 실험 결과, 제안된 프록시(proxy)를 언어 모델 훈련에 통합함으로써 지시 사항을 따르는 능력이 크게 향상되었으며, Vicuna 및 MT-bench와 같은 벤치마크에서 기존 방법들을 큰 폭으로 상회하는 성능을 보였습니다.



### Can Large Language Models Replicate ITS Feedback on Open-Ended Math Questions? (https://arxiv.org/abs/2405.06414)
Comments:
          Educational Data Mining 2024

- **What's New**: 이 연구에서는 대규모 언어 모델(Large Language Models, LLMs)을 사용하여 개방형 수학 문제에 대한 피드백을 생성하는 능력을 탐구합니다. 특히, 기존의 템플릿 기반 접근(Intelligent Tutoring Systems, ITSs) 방식과 비교하여 LLMs의 가능성을 평가합니다. 자동 피드백을 제공하는 새로운 시도에서, 오픈 소스 및 독점적인 LLMs는 실제 학생의 응답과 ITS에서 제공하는 피드백을 기반으로 세부 조정(fine-tune)됩니다.

- **Technical Details**: 연구에서는 오픈 소스 및 독점 LLM을 특정 대응 피드백에 맞게 세부 조정하고, 생성된 피드백의 질을 텍스트 유사도 메트릭스(text similarity metrics)를 사용하여 측정했습니다. 이러한 메트릭스는 피드백이 학습 단계에서 본 피드백을 얼마나 잘 모방하는지 평가하는 데 사용됩니다.

- **Performance Highlights**: LLMs는 훈련 중에 본 피드백을 재현하는 데는 유망한 결과를 보였으나, 이전에 보지 못한 학생 오류에 대해서는 일반화(generalize)하는 데 실패했습니다. 피드백의 형식은 배울 수 있지만, 학생들의 수학적 오류를 완전히 이해하는 데는 능력이 부족한 것으로 나타났습니다.



### Potential and Limitations of LLMs in Capturing Structured Semantics: A Case Study on SRL (https://arxiv.org/abs/2405.06410)
Comments:
          Accepted by ICIC 2024

- **What's New**: 이번 논문에서는 대규모 언어 모델 (LLMs)이 구조화된 의미론을 이해하는 데 얼마나 효과적인지 평가하기 위해 의미역 레이블링(Semantic Role Labeling, SRL)을 사용하는 새로운 접근법을 제안합니다. 이를 통해 언어 모델들이 자연어를 명시적인 의미 구조로 매핑할 수 있게 하며, 이는 LLM의 속성을 해석 가능한 창으로 제공합니다.

- **Technical Details**: 이 연구에서 개발된 few-shot SRL parser인 PromptSRL은 특정의 적은 양의 데이터로 효율적으로 학습할 수 있는 구조로, LLM을 활용하여 자연어로부터 의미 구조를 추출합니다. 이 방법은 특히 LLM의 구조화된 의미론 이해 능력을 평가하는 데 사용됩니다.

- **Performance Highlights**: PromptSRL을 사용한 결과, LLM은 실제로 의미 구조를 포착할 수 있는 능력이 있음을 발겎했습니다. 그러나 모델을 확장하는 것이 항상 성능 향상을 의미하지는 않으며, C-arguments 등에서 LLM의 한계도 관찰되었습니다. 놀랍게도 LLM과 훈련되지 않은 사람들이 만든 오류에는 거의 30%가 중복되는 것으로 나타났습니다.



### LLM Discussion: Enhancing the Creativity of Large Language Models via Discussion Framework and Role-Play (https://arxiv.org/abs/2405.06373)
Comments:
          10 pages, 6 figures, Under Review of COLM

- **What's New**: LLM (Large Language Models)이 자연 언어 처리 분야에서 뛰어난 능력을 보여주고 있지만 창의적인 답변을 생성하는데 한계를 보이고 있습니다. 이를 해결하기 위해, 저희는 다양한 배경과 관점을 가진 참여자들과의 토론을 통해 집단 창의력을 자극하는 인간의 과정을 모방하는 통찰력을 바탕으로 'LLM 토론(LLM Discussion)'이라는 새로운 세 단계 토론 프레임워크를 제안합니다. 또한, LLM의 동질성을 극복하기 위해 역할 분담 기법(role-playing technique)을 도입하였습니다.

- **Technical Details**: 제안된 LLM 토론 프레임워크는 초기 단계(initiation phase), 토론 단계(discussion phase), 그리고 수렴 단계(convergence phase)로 구성되어 있으며 각 단계마다 특별히 설계된 프롬프트를 사용합니다. 역할 분담은 'Design Thinking'에서와 같이 각 LLM 에이전트에게 다양한 배경과 관점을 가진 역할을 할당함으로써 서로 다른 아이디어를 교환하고 창의적인 답변으로 수렴할 수 있도록 합니다.

- **Performance Highlights**: 이 프레임워크는 창의성 평가를 위해 Alternative Uses Test, Instances Test, Similarities Test, 그리고 Scientific Creativity Test 등 네 가지 벤치마크를 포함하고 있습니다. 연구 결과, 제안된 프레임워크는 기존의 단일 LLM 접근법이나 다른 멀티-LLM 프레임워크들을 상회하는 창의성 수치를 기록했습니다. 특히, 원본성(Originality), 상세화(Elaboration), 유창성(Fluency), 그리고 유연성(Flexibility)의 네 가지 주요 메트릭스에서 우수한 성능을 보였습니다.



### Akal Badi ya Bias: An Exploratory Study of Gender Bias in Hindi Language Technology (https://arxiv.org/abs/2405.06346)
Comments:
          Accepted to FAccT 2024

- **What's New**: 이 연구는 힌디어에 대한 성 차별을 측정하고 완화하는 기존 연구가 주로 영어에 중점을 두어 비영어권 국가와 글로벌 사우스(Global South)의 복잡한 문제를 간과했다는 점을 지적합니다. 이 논문은 세계에서 세 번째로 많이 사용되는 언어인 힌디어에서 성 차별의 미묘한 특성을 철저히 조사하는 최초의 종합 연구를 제시합니다.

- **Technical Details**: 이 연구는 다양한 마이닝(mining) 기술, 계산 모델(computational models), 그리고 현장 연구(field studies)를 활용하여 현재 방법론의 한계를 밝히고 있습니다. 기존 방법으로는 힌디어에서 성편견 문장을 추출하는 데 어려움이 있었기 때문에 현장 조사를 통해 이러한 문장 수집을 촉진하였습니다.

- **Performance Highlights**: 농촌 지역 및 저소득층 여성을 포함한 현장 조사를 통해 다양한 성편견에 대한 인식을 밝혀내고, 맥락에 특화된 접근이 필요함을 강조하였습니다. 또한, 이 연구는 힌디어에서의 성 차별에 대한 이해를 넓히는 동시에 인도 언어들의 추가적인 탐구를 위한 기초를 마련하며, 지금까지 고려되지 않았던 맥락에서 성 차별에 대한 신중한 참여를 촉구하면서 포괄적이고 공평한 언어 및 문화적 맥락에서의 포용성을 증진하고자 합니다.



### Correlation Dimension of Natural Language in a Statistical Manifold (https://arxiv.org/abs/2405.06321)
Comments:
          Published at Physical Review Research

- **What's New**: 이 논문은 자연 언어의 상관 차원(correlation dimension)을 노이즈가 많은 대규모 언어 모델에서 생성된 고차원 시퀀스에 Grassberger-Procaccia 알고리즘을 적용하여 측정합니다. 이 방법은 유클리드 공간(Euclidean space)에서만 연구되었던 것을 통계 매니폴드(statistical manifold)로 재구성하고, Fisher-Rao 거리를 통해 새로운 통찰을 제공합니다.

- **Technical Details**: 언어는 다중 프랙탈(multifractal) 구조를 보이며, 전역적인 자기 유사성(global self-similarity)과 약 6.5의 보편적 차원(universal dimension)을 가집니다. 이는 간단한 이산 무작위 시퀀스보다는 작고, Barabási-Albert 과정의 차원보다는 큽니다. 장기 기억(long memory)이 자기 유사성을 생성하는 주요 요인으로 작용합니다.

- **Performance Highlights**: 이 방법은 확률적 모델(probabilistic model)을 사용하여 실제 세계의 이산 시퀀스(discrete sequences)에 널리 적용될 수 있으며, 음악 데이터에 대한 응용을 보여줍니다. 이는 언어 뿐만 아니라 다른 유형의 데이터에도 유사한 패턴이 존재함을 시사합니다.



### A NLP Approach to "Review Bombing" in Metacritic PC Videogames User Ratings (https://arxiv.org/abs/2405.06306)
Comments:
          11 pages, 4 figures. Accepted by Discover Artificial Intelligence but withdrawn due to APC

- **What's New**: {What's New}리뷰 폭탄(review bombing) 현상이 비디오 게임에서 발생할 때, 많은 저평가들이 제품의 실제 품질을 반영하지 않습니다. 이 연구에서는 메타크리틱(Metacritic)의 5만 개 이상의 PC 게임 유저 평점을 분석하여, 자연어 처리(Natural Language Processing, NLP) 방법을 사용해 이러한 평가에서 주로 나타나는 주요 단어와 개념을 이해하려고 시도하였습니다.

- **Technical Details**: {Technical Details}연구팀은 NLP 기술을 활용하여 리뷰 텍스트 데이터를 분석하고, 진정한 나쁜 평가와 리뷰 폭탄을 구분하는 모델을 개발하였습니다. 이 모델은 검증 세트에서 0.88의 정확도(Accuracy)를 달성했으며, 이는 리뷰 텍스트의 패턴을 분석하여 리뷰 폭탄을 식별할 수 있음을 보여줍니다.

- **Performance Highlights**: {Performance Highlights}개발된 NLP 모델은 리뷰 폭탄과 일반적인 나쁜 리뷰를 구분하는 데 88%의 정확도를 보였습니다. 이 높은 정확도는 리뷰 폭탄이 특정 단어나 개념을 반복적으로 언급하거나 특정 패턴을 따르는 것으로 나타났기 때문입니다.



### Aspect-oriented Consumer Health Answer Summarization (https://arxiv.org/abs/2405.06295)
- **What's New**: 이 연구는 건강 관련 질문에 대한 다양한 답변을 요약하는 측면 기반 요약(aspect-based summarization)에 초점을 맞추었습니다. 기존의 커뮤니티 질문-답변(CQA) 포럼들은 일반적으로 가장 많이 투표된 단일 답변을 각 질문의 대표 요약으로 사용했지만, 이 방법은 다른 해결책이나 정보를 놓칠 수 있습니다. 이를 해결하기 위해 다단계 주석 가이드라인을 개발하고, 특정 측면에 기반한 인간 작성 요약 데이터셋을 제공합니다.

- **Technical Details**: 연구팀은 여러 최신 모델을 특정 작업에 맞게 미세 조정하여 자동화된 다면적 답변 요약 파이프라인을 구축했습니다. 이 파이프라인은 질문 유사성을 활용하여 관련 답변 문장을 검색하고, 이를 적절한 측면 유형별로 분류합니다. 그 후, 최근의 추상적 요약 모델(abstractive summarization models)을 사용하여 측면 기반 요약을 생성합니다.

- **Performance Highlights**: 이 파이프라인은 인간의 분석을 통해 검증되었으며, 관련 내용을 효과적으로 포착하고 다양한 해결책을 제공하는 데 있어 높은 평가를 받았습니다. 이는 CQA 포럼의 사용성을 향상시키는 데 기여할 수 있습니다.



### Pruning as a Domain-specific LLM Extractor (https://arxiv.org/abs/2405.06275)
Comments:
          NAACL 2024 Findings

- **What's New**: 이 연구에서는 대규모 언어 모델(LLMs)의 도메인 특화 압축을 위한 새로운 이중 프루닝 방법론, D-Pruner를 소개합니다. D-Pruner는 일반적인 능력과 도메인 특화 지식을 모두 고려하여 모델의 크기를 줄이는 동시에 특정 도메인에서 다양한 작업을 효과적으로 처리할 수 있는 압축된 모델을 추출합니다.

- **Technical Details**: D-Pruner는 개방형 도메인 캘리브레이션 데이터셋을 사용하여 가중치의 일반 중요성을 평가하고, 이 정보를 사용하여 훈련 손실을 조정하여 특정 도메인에 적합할 때 일반성을 유지합니다. 또한, 도메인 특화 캘리브레이션 데이터셋에서 가중치의 중요성을 효과적으로 근사화하여 일반성과 특화성을 강조하는 가지치기(pruning)된 모델을 얻습니다. 본 방법론은 또한 empirical Fisher를 활용하여 가중치의 중요성을 효율적으로 계산하고, 일반 가중치 중요성 모듈, 업데이트된 훈련 손실 함수, 그리고 그라디언트를 활용한 가중치 중요성 계산과 같은 여러 단계를 포함합니다.

- **Performance Highlights**: D-Pruner는 건강 관리 및 법률 도메인과 같은 다양한 도메인 특화 데이터셋에서 뛰어난 성능을 보여주었습니다. 특히 LLaMA2 모델을 사용한 실험에서, D-Pruner는 전체 밀집 모델과 비교할 수 있는 결과를 달성하면서도 50%의 희소성을 달성하여 기존의 프루닝 기술들을 능가하는 성과를 보였습니다. 이러한 결과는 언어 이해, 질문 응답 및 요약 작업을 포함하여 다양한 작업에서 입증되었습니다.



### Automatic Generation of Model and Data Cards: A Step Towards Responsible AI (https://arxiv.org/abs/2405.06258)
Comments:
          NAACL 2024 Main Poster

- **What's New**: 이 논문은 기계 학습(AI/Machine Learning)에서 데이터와 모델이 급속도로 늘어나는 상황에서 표준화된 일관된 문서화의 필요성을 다룹니다. 저자들은 현재 인간이 생성하는 모델 카드와 데이터 카드의 정보 불완전성을 해결하기 위해 대규모 언어 모델(Large Language Models, LLMs)을 사용한 자동 생성 접근 방식을 제안합니다.

- **Technical Details**: 연구팀은 모델 카드(model cards)와 데이터 카드(data cards)로 구성된 종합 데이터셋 CardBench를 만들었고, 이는 4.8천개의 모델 카드와 1.4천개의 데이터 카드를 집약한 것입니다. 이 데이터셋을 기반으로, 두 단계 검색 과정을 포함하는 CardGen 파이프라인을 개발하였습니다.

- **Performance Highlights**: 이 새로운 접근법은 생성된 모델 카드와 데이터 카드에서 완성도, 객관성, 신뢰성을 향상시키며, 이는 책임감 있는 AI 문서화 관행을 보장하여 더 나은 계정성(accountability)과 추적 가능성(traceability)을 확보하는 데 중요한 발걸음입니다.



### SaudiBERT: A Large Language Model Pretrained on Saudi Dialect Corpora (https://arxiv.org/abs/2405.06239)
- **What's New**: 이 논문에서는 사우디 방언 텍스트만을 대상으로 사전 훈련된 사우디어 버전의 언어 모델인 SaudiBERT를 소개합니다. 사우디 방언을 효과적으로 이해하고 분석할 수 있는 SaudiBERT는 감성 분석(sentiment analysis)과 텍스트 분류(text classification)를 포함한 다양한 평가 데이터셋에서 여러 다양한 다중 방언 아랍어 언어 모델과 비교하여 뛰어난 성능을 보였습니다.

- **Technical Details**: SaudiBERT는 사우디어 방언을 기반으로 한 새로운 데이터셋인 Saudi Tweets Mega Corpus (STMC)와 Saudi Forums Corpus (SFC)를 사용하여 사전 훈련되었습니다. STMC는 1억 4100만 개 이상의 트윗을 포함하고 있으며, SFC는 15.2GB의 텍스트를 5개의 사우디 온라인 포럼에서 수집했습니다. 이와 같은 대규모의 특화된 데이터셋은 모델을 사우디 방언에 더욱 최적화시키는 데 도움이 되었습니다.

- **Performance Highlights**: SaudiBERT는 감성 분석에서 평균 F1-점수(F1-score) 86.15%를, 텍스트 분류에서는 87.86%를 달성하며 기존의 다른 모델들을 모두 상회하는 성능을 보였습니다. 이 모델은 대부분의 작업에서 최신 기술(state-of-the-art) 성과를 이뤄내며 기타 포함된 언어 모델들보다 우수한 결과를 보였습니다. 또한, SaudiBERT 모델은 공개적으로 사용 가능하며, 해당 HTTP URL에서 확인할 수 있습니다.



### For the Misgendered Chinese in Gender Bias Research: Multi-Task Learning with Knowledge Distillation for Pinyin Name-Gender Prediction (https://arxiv.org/abs/2405.06221)
- **What's New**: 이 연구는 중국어 이름(핑인 Pinyin)을 사용하여 개인의 성별을 추정하는 기존 도구들이 가지고 있는 문제를 해결하기 위한 새로운 시도를 소개합니다. 연구진은 지식 증류(Knowledge Distillation)를 도입한 다중 작업 학습 네트워크(Multi-Task Learning Network)를 개발하여, 핑인과 중국어 문자(한자 Hanzi) 사이의 연관성을 모델링하고, 이를 통해 성별 정보를 효과적으로 추론할 수 있는 방법을 제시했습니다. 특히, 이 방법은 기존 상용 성별 추정 도구들을 9.70%에서 20.08% 상대적으로 능가하는 성능을 보였으며, 소스 코드는 오픈 소스로 공개되었습니다.

- **Technical Details**: 연구진은 핑인 이름-성별 추정 문제를 새롭게 정의하고, 중국 문자의 의미적 특성을 핑인 임베딩에 통합할 수 있도록 설계된 다중 작업 학습 네트워크를 사용했습니다. 이 네트워크는 핑인과 관련된 모든 중국 문자의 정보를 통합하면서 핑인과 중국 문자 사이의 일대다 관계를 유지하는 핑인 임베딩을 학습합니다. 또한, 지식 증류를 통해 중국 문자 이름-성별 예측 모델로부터 학습된 정보를 핑인 이름-성별 예측 모델로 전달하여, 성별 경향성이 핑인 임베딩에 명확하게 반영되도록 했습니다.

- **Performance Highlights**: 이 방법은 기존 상용 성별 추정 서비스들과 비교하여 9.70%에서 20.08% 상대적으로 높은 성능 향상을 보였습니다. 특히, 핑인 이름의 성별을 추정하는 데 있어서, 새로운 접근 방식이 기존 방법들보다 훨씬 정확하다는 것이 입증되었습니다. 이 연구는 핑인 이름에 대한 성별 추정의 가능한 정확도 상한을 판단하는 데 도움을 주며, 로마자로 표기되는 다른 언어의 이름(예: 일본어)에 대해서도 유사한 모델을 적용할 가능성을 제시합니다.



### A Survey on RAG Meets LLMs: Towards Retrieval-Augmented Large Language Models (https://arxiv.org/abs/2405.06211)
- **What's New**: 인공지능에서 가장 진보된 기술 중 하나인 검색 증강 생성(Retrieval-Augmented Generation, RAG)은 광범위한 작업에 대해 신뢰할 수 있고 최신의 외부 지식을 제공하여 큰 편의를 제공합니다. 특히 AI 생성 콘텐츠(AI-Generated Content, AIGC) 시대에 RAG의 강력한 검색 능력이 기존 생성 AI를 도와 고품질 출력물을 생산하는데 있어 중요한 역할을 합니다.

- **Technical Details**: 이 연구에서는 검색 증강 대형 언어 모델 (Retrieval-Augmented Large Language Models, RA-LLMs)에 대한 기존 연구를 종합적으로 검토합니다. 주요 기술적 관점으로는 구조(architectures), 훈련 전략(training strategies), 그리고 응용 프로그램(applications)을 다룹니다. RA-LLMs는 대형 언어 모델의 생성 품질을 향상시키기 위해 모델의 내부 지식에만 의존하지 않고 외부 및 권위 있는 지식 베이스를 활용합니다.

- **Performance Highlights**: RA-LLMs는 기존 대형 언어 모델들이 가질 수 있는 한계점, 예를 들어 환각(깊이 있는 외부 지식 부족으로 인한 정보의 변형)과 오래된 내부 지식을 극복할 수 있는 능력을 보여주었습니다. 또한, 본 연구에서는 RA-LLMs의 구체적인 도전 과제와 각각에 대한 능력을 자세히 설명함으로써 이러한 기술이 어떻게 실질적인 중요성을 가지는지를 보여줍니다.



### HC$^2$L: Hybrid and Cooperative Contrastive Learning for Cross-lingual Spoken Language Understanding (https://arxiv.org/abs/2405.06204)
Comments:
          Accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI). arXiv admin note: text overlap with arXiv:2312.03716

- **What’s New**: 새롭게 제시된 모델은 제로-샷(Zero-shot) 교차 언어 음성 언어 이해(cross-lingual spoken language understanding)를 위해 혼합 및 협력 대조 학습(Hybrid and Cooperative Contrastive Learning) 방법을 제안합니다. 이 방법은 감독 없는 교차 언어 대조 학습(unsupervised contrastive learning)뿐만 아니라, 소스 언어의 감독 대조 학습(supervised contrastive learning), 교차 언어 감독 대조 학습, 그리고 다국어 감독 대조 학습(multilingual supervised contrastive learning)을 포함하여 라벨 인식 의미 구조(label-aware semantics structure)를 종합적으로 조정합니다.

- **Technical Details**: 본 논문에서는 네 가지 대조 학습(contrastive learning) 메커니즘을 개발하고 있으며, 이들은 단일 작업(single-task)과 공동 작업(joint-task) 시나리오 모두를 포함합니다. 각 대조 학습 메커니즘은 서로의 입력을 강화하며, 이는 훈련 과정에서 더 일관되고 차별화된 표현을 학습하는 데 도움을 줍니다.

- **Performance Highlights**: 이 모델은 9개 언어에 대한 일관된 개선을 보여주어 새로운 최고 성능(state-of-the-art performance)을 달성합니다. 연구 결과는 이러한 혼합 및 협력 대조 학습 접근 방식이 다양한 언어에 걸쳐 의미적 정렬과 표현 학습을 개선할 수 있음을 시사합니다.



### Lost in Transcription: Identifying and Quantifying the Accuracy Biases of Automatic Speech Recognition Systems Against Disfluent Speech (https://arxiv.org/abs/2405.06150)
Comments:
          Accepted to NAACL 2024

- **What's New**: 이 연구는 음성 인식 시스템(Automatic Speech Recognition, ASR)이 말더듬이 있는 사람들의 발화를 처리할 때 발생하는 편향을 정량적으로 분석합니다. 연구팀은 실험을 통해 신디사이즈드(합성) 말더듬 음성 데이터셋과 실제 말더듬 음성 데이터셋을 사용하여 여섯 가지 주요 ASR 시스템의 성능을 평가하였습니다.

- **Technical Details**: 연구자들은 WER(Word Error Rate, 단어 오류율) 및 CER(Character Error Rate, 문자 오류율)와 같은 지표를 사용하여 ASR 시스템의 정확도를 측정하고, 의미적 정확성(Semantic Accuracy)까지 평가하여 발화가 얼마나 잘 보존되었는지 평가하였습니다. 이를 위해 LibriSpeech 벤치마크에서 파생된 합성 데이터셋과 실제 말더듬 음성 샘플을 사용하였습니다.

- **Performance Highlights**: 연구 결과, 모든 ASR 시스템에서 일반적인 발화에 비해 말더듬 발화에 대한 처리 성능이 현저히 낮음을 확인할 수 있었습니다. 이러한 결과는 ASR 기술의 현재 한계를 드러내며, 말더듬을 가진 사람들이 디지털 환경에서 효과적으로 소통할 수 있도록 기술의 개선이 필요함을 시사합니다.



### Reddit-Impacts: A Named Entity Recognition Dataset for Analyzing Clinical and Social Effects of Substance Use Derived from Social Media (https://arxiv.org/abs/2405.06145)
Comments:
          7 pages, 1 figure, 4 tables

- **What's New**: 이 연구는 약물 남용 장애 (Substance use disorders, SUDs)의 임상적 및 사회적 영향을 탐색하기 위해 Reddit-Impacts라는 새로운 데이터셋을 소개합니다. 이 데이터셋은 Reddit에서 비의료적 사용자가 생성한 글에서 임상적 및 사회적 영향을 자동으로 감지할 수 있는 시스템 개발을 돕기 위해 만들어졌습니다. 데이터셋은 의료용 및 불법 오피오이드와 그 치료제에 대한 토론을 포함합니다.

- **Technical Details**: Reddit-Impacts 데이터셋은 14개의 오피오이드 관련 서브레딧에서 수집된 데이터로 구성되어 있습니다. 수집된 데이터는 수동으로 주석을 달아 임상 및 사회적 영향을 나타내는 텍스트 범위를 식별하고, 이를 통해 NER (Named Entity Recognition) 작업을 개선할 수 있는 훈련 데이터를 제공합니다. 또한, 이 연구는 BERT와 RoBERTa와 같은 트랜스포머(Transformer) 모델들을 사용하여 기준 성능을 설정하고, DANN과 GPT-3.5를 포함한 몇 가지 기계 학습 모델을 실험적으로 적용하였습니다.

- **Performance Highlights**: Reddit-Impacts 데이터셋에서 추출된 데이터는 임상 및 사회적 영향의 자동 감지를 위한 탄탄한 기준선(Baseline) 테스트를 제공하며, 아직 깊이 연구되지 않은 분야에서의 연구를 가능하게 합니다. 트랜스포머 기반 접근법은 충분한 양의 주석이 달린 데이터가 있을 때 NER 작업에서 최고의 F1-Score를 달성하는 것으로 나타났습니다. 이는 소수의 주석 데이터로도 효과적인 결과를 내는 데 도움이 될 수 있습니다.



### Muting Whisper: A Universal Acoustic Adversarial Attack on Speech Foundation Models (https://arxiv.org/abs/2405.06134)
- **What's New**: 이 연구에서는 대규모 음성 기반 기초 모델인 Whisper를 사용하는 자동 음성 인식(ASR) 시스템의 새로운 취약점을 드러냅니다. 특히, Whisper의 '<endoftext>' 같은 특별한 토큰(token)이 적대적 공격(adversarial attacks)에 의해 악용될 수 있음을 보여주며, 이를 이용해 ASR 모델을 '무음(mute)' 상태로 만드는 방법을 제시합니다.

- **Technical Details**: 연구팀은 Whisper의 '<endoftext>' 토큰을 학습하여 0.64초 길이의 적대적 음향 구간(adversarial audio segment)을 생성하는 방법을 개발했습니다. 이 음향 구간을 어떠한 음성 신호에도 앞에 추가함으로써, 해당 신호가 ASR 시스템에 의해 무시되도록 만들 수 있습니다. 이는 ASR 시스템이 실제 음성을 전사(transcribe)하지 않고 특수 토큰만을 기록하게 함으로써, 모델을 '무음(mute)' 상태로 만드는 효과를 가집니다.

- **Performance Highlights**: 실험 결과, 이 적대적 음향 구간은 97% 이상의 음성 샘플에 대해 Whisper ASR 모델을 성공적으로 '무음' 시킬 수 있었습니다. 또한, 이 공격은 다양한 데이터셋과 작업에 걸쳐 높은 전이성(transferability)을 보여, 다른 음성 처리 작업에도 유사하게 적용될 수 있음을 입증하였습니다.



### Can Perplexity Reflect Large Language Model's Ability in Long Text Understanding? (https://arxiv.org/abs/2405.06105)
- **What's New**: LLM(Large Language Models)이 극도로 긴 텍스트 처리 능력을 가지고 있음이 최근 연구에서 밝혀졌습니다. 그러나 이 연구는 PPL(Perplexity)이 긴 텍스트 이해 능력과는 상관 관계가 없으며, PPL이 오직 지역적인 정보 모델링 능력만을 반영할 수 있다는 사실을 발견했습니다. 또한, PPL이 낮다는 것이 긴 텍스트를 더 잘 이해한다는 것을 의미하지 않음을 실험을 통해 입증하였습니다.

- **Technical Details**: 연구는 LLaMA2 같은 짧은 컨텍스트 윈도우(4,096) 모델과 100K 토큰 이상의 긴 컨텍스트 윈도우를 가진 여러 LLM 변형을 비교했습니다. PPL은 모델이 얼마나 잘 지역 정보만을 이용하는지를 보여주는 지표로 사용되었고, 이는 ALiBi와 같은 위치 코딩 방식이 어떻게 장거리 의존성을 무시하고 계속 좋은 PPL 성능을 낼 수 있는지를 설명해 줍니다.

- **Performance Highlights**: YARN, Yi, LongLoRA 같은 모델들은 76K 입력 길이에서 PPL을 통한 언어 모델링 성능은 높지만, QMSUM 및 NarrativeQA와 같은 긴 텍스트 이해를 요구하는 다운스트림 태스크에서는 그 성능이 일치하지 않았습니다. 이는 PPL이 긴 텍스트 이해 능력을 반영하지 않음을 시사합니다.



### HMT: Hierarchical Memory Transformer for Long Context Language Processing (https://arxiv.org/abs/2405.06067)
- **What's New**: 새로운 계층적 메모리 트랜스포머(Hierarchical Memory Transformer, HMT)가 제안되었습니다. 이 모델은 인간의 기억 행위를 모방하여 무한한 컨텍스트를 효과적으로 처리할 수 있는 능력을 갖추도록 설계되었습니다. HMT는 기존의 트랜스포머 기반 대규모 언어 모델(LLM)에 쉽게 통합될 수 있는 플러그 앤 플레이(Plug-and-play) 구조로, 긴 컨텍스트 처리 능력을 향상시키기 위해 메모리 증강 세그먼트 레벨 순환을 활용합니다.

- **Technical Details**: HMT는 각종 메모리 계층(감각, 단기, 장기 메모리)을 모방하여 인코딩된 메모리 임베딩을 저장하고, 현재 토큰 세그먼트와 관련있는 정보를 검색합니다. 이런 구조는 특히 컨텍스트 전환 시 유효성을 높입니다. 훈련 및 파인튜닝은 기존 모델의 파라미터와 새롭게 도입된 파라미터를 함께 조정하여 수행됩니다.

- **Performance Highlights**: 실험 결과, HMT는 Wikitext-103과 PG-19 데이터셋에서 기존의 재귀 메모리 트랜스포머(RMT) 모델보다 우수한 성능을 보였습니다. HMT를 적용한 OPT와 OpenLlamaV2 모델은 Wikitext-103에서 각각 25.5%, 17.6% 향상된 효과를 보였고, PG-19에서는 11.4%, 9.48% 향상되었습니다. 또한, PubMedQA 데이터셋에서는 긴 답변 컨텍스트 추론에서 9.81% 향상되었고, 짧은 답변 예측 정확도는 1.0% 증가하였습니다.



### A Mixture-of-Experts Approach to Few-Shot Task Transfer in Open-Ended Text Worlds (https://arxiv.org/abs/2405.06059)
- **What's New**: 본 논문은 정해진 목표나 환경 보상 신호가 없는 열린 세계에서 새로운 역할에 맞는 행동을 빠르게 학습할 수 있는 새로운 기술을 제시합니다. 이 기술은 미리 알려진 여러 작업의 정책(policy)을 전문가들의 혼합(Mixture-of-Experts, MoE) 모델에 결합하고, 얼어붙은(frozen) 전문가와 새로운 전문가 사이의 주의 기제(attention mechanism)를 사용하여 효과적으로 작업 전환(task-transfer)을 수행합니다.

- **Technical Details**: MoE 모델은 기존의 정책(policy) 모델에서 어떤 전문가(expert)에게 주의를 기울일지를 배우는 attention 메커니즘을 사용합니다. 새로운 상황에서는 새로운 전문가를 학습하여 적절할 때 기존의 얼어붙은 전문가에게 주의를 기울입니다. 이 연구는 텍스트 기반 환경, 특히 '던전 앤 드래곤'과 같은 게임에서 캐릭터 역할에 따른 행동 학습에 초점을 맞추고 있으며, 주어진 역할에 적합한 행동을 빠르게 학습할 수 있습니다.

- **Performance Highlights**: 제안된 MoE 기법은 zero-shot 설정에서 더 많은 보상을 얻을 수 있었으며, few-shot 학습 설정에서는 보상을 발견하는데 있어 더 높은 표본 효율성을 보여주었습니다. 이는 새로운 역할 학습 시 기존 전문가의 지식을 재활용하고 빠르게 새로운 역할에 적응할 수 있음을 의미합니다.



### Value Augmented Sampling for Language Model Alignment and Personalization (https://arxiv.org/abs/2405.06639)
Comments:
          Website: this https URL

- **What's New**: 새로운 보상 최적화 프레임워크인 'Value Augmented Sampling (VAS)'이 제안되었습니다. 이 방법은 Large Language Models(LLMs)를 다양한 인간 선호도에 맞추고, 새로운 기술을 학습하며, 해로운 행동을 잊도록 하는 것을 목표로 합니다. VAS는 기존의 Reinforcement Learning(RL) 방법이나 그 밖의 탐색 기반 방법들의 문제점을 해결하고자 합니다.

- **Technical Details**: VAS는 초기의 고정된 LLM으로부터 샘플된 데이터만 사용하여 다양한 보상 함수를 최대화할 수 있습니다. 이는 정책(policy)과 값 함수(value function)를 공동 학습하지 않고, 최적의 보상 극대화 정책을 해결할 수 있게 함으로써 최적화를 안정화시킵니다. 또한, VAS는 기존의 LLM의 가중치를 변경할 필요가 없으며, API로만 제공되는 LLMs(예: ChatGPT)도 적응할 수 있습니다.

- **Performance Highlights**: VAS는 표준 벤치마크에서 기존 베이스라인인 PPO와 DPO를 능가하며, Best-of-128과 비교할 만한 결과를 보여주면서도 더 낮은 추론 비용을 가집니다. 이를 통해 VAS는 보다 효율적인 LLM의 적응을 가능하게 하며, 다양한 보상을 조합하고 배치 시에 각 보상의 정도를 조절하는 새로운 기능을 제공합니다.



### Federated Document Visual Question Answering: A Pilot Study (https://arxiv.org/abs/2405.06636)
- **What's New**: 이 연구는 분산된 사적 문서 데이터(Documents)에 대한 공유 모델을 훈련하는 방법으로 연합 학습(Federated Learning, FL) 방식을 탐구합니다. 특히, 문서에 대한 시각적 질문 응답(Task of Document VQA) 문제를 중심으로 연구가 진행되었는데, 이는 데이터 도메인 간 다양성 때문에 모델이 요구하는 추론 능력이 상이할 수 있기 때문에 이 방식이 적합합니다.

- **Technical Details**: 연구자들은 다양한 분야에서 기존의 DocVQA 데이터셋을 조합하여 실제 세계 응용 프로그램에서 데이터의 이질성을 반영하고자 했습니다. 연구는 또한 자가 사전 훈련(Self-pretraining) 기술을 다중 모달(Multi-modal) 설정에서 탐구하여, 사전 훈련과 미세 조정(Finetuning) 모두에 동일한 데이터를 사용함으로써 프라이버시 보존에 유리하게 작용한다는 점을 강조했습니다. 추가적으로, 중앙 집중식 적응 최적화(Centralized Adaptive Optimization)와 연합 DocVQA 훈련 방법을 결합하는 새로운 방법을 제안하여 기준선인 FedAvg보다 우수한 성능을 달성했습니다.

- **Performance Highlights**: 실험을 통해, 연구팀은 연합 훈련(Federated Training)하에 다양한 DocVQA 데이터셋으로 사전 훈련 전략이 효과적임을 보여주었습니다. 또한, 문서 작업에 있어서 연합 설정 하에서 하이퍼파라미터(Hyperparameters) 튜닝의 중요성을 강조하면서 실제 작업에 대한 다면적 분석을 제시하였습니다.



### Multimodal LLMs Struggle with Basic Visual Network Analysis: a VNA Benchmark (https://arxiv.org/abs/2405.06634)
Comments:
          11 pages, 3 figures

- **What's New**: 이 연구에서는 GPT-4와 LLaVa 모델이 시각 네트워크 분석(Visual Network Analysis, VNA) 작업에서 제로-샷(Zero-shot) 능력을 평가합니다. 연구진은 3가지 기본 네트워크 과학 개념을 바탕으로 5가지 작업을 설정하여 양 모델을 평가했습니다. 이러한 평가를 통해 VLM(Vision Language Models)의 성능 한계를 파악하고 해당 분야에 대한 벤치마크를 처음으로 공개하고자 합니다.

- **Technical Details**: GPT-4와 LLaVa 모델에는 5가지 VNA 작업이 주어졌는데, 이는 1) 최대 도수(degree) 노드 식별, 2) 서명된 삼중항(triads)이 균형 잡혔는지 여부 식별, 3) 그래프 내 구성 요소 수 계산 등의 작업으로 구성되어 있습니다. 각 작업은 높은 해상도의 합성 그래프 시각화를 사용하여 진행되었으며, GPT-4와 LLaVa는 모두 인간의 도움 없이 네트워크 시각화에서 그래프 이론 개념을 도출하도록 요구되었습니다. 실험 결과, GPT-4가 LLaVa보다 일관되게 더 좋은 성능을 보였지만, 모든 VNA 작업에서 양 모델은 모두 어려움을 겪었다는 결과가 나타났습니다.

- **Performance Highlights**: GPT-4는 구성 요소(isolates) 수를 세는 작업에서 100개의 그래프 시각화 중 67개를 정확하게 식별하여 가장 높은 정확도를 달성했습니다. 하지만, 삼중항의 구조적 균형 여부를 예측하는 작업은 놀랍게도 양 모델에게 가장 도전적인 작업 중 하나로, GPT-4는 이 작업에서 무작위 추측과 맞먹는 0.51의 정확도를 보였습니다. 이 연구는 VLM이 시각적 그래프 분석 작업에서의 제로-샷 성능을 이해하고 향상시키는 데 더 많은 작업이 필요함을 시사합니다.



### Characterizing the Accuracy - Efficiency Trade-off of Low-rank Decomposition in Language Models (https://arxiv.org/abs/2405.06626)
- **What's New**: 본 논문에서는 큰 언어 모델(LLMs)의 메모리 최적화를 위해 터커 분해(Tucker decomposition)라는 저차원 분해 방법을 적용하고, 이를 통해 모델 크기를 줄임으로써 실시간 서비스가 필요한 LLM 기반 애플리케이션에 적합하도록 만들었습니다. 특히, 오픈소스 LLM인 Llama 2를 사용하여 실험을 수행했습니다.

- **Technical Details**: 터커 분해 방법을 사용하여 LLMs의 데이터 전송량과 메모리 사용량을 최적화하였습니다. Llama 2 모델에 적용하여 설계 공간을 형식화하고, 큰 설계 공간(O($2^{37}$))을 탐색하기 위한 케이스 스터디를 수행하였습니다.

- **Performance Highlights**: 이 저차원 분해 방법을 통해 모델 크기를 9% 줄이면서, 벤치마크의 난이도에 따라 4%에서 10% 포인트의 정확도 감소만을 보였습니다. 이러한 결과는 저차원 분해(decomposition)가 모델의 정확성뿐만 아니라 응답 시간(latency)이 중요한 응용 프로그램에서 유망한 방향임을 보여줍니다.



### Sampling the Swadesh List to Identify Similar Languages with Tree Spaces (https://arxiv.org/abs/2405.06549)
Comments:
          19 pages, 26 figures

- **What's New**: 이 논문에서는 언어 클러스터링을 위한 새로운 데이터 분석 방법을 소개합니다. 주로, '3-spider'라는 간단한 공간 모델을 사용하여 언어의 계통을 나타냅니다. 이 모델은 Latin Script(라틴 문자)를 사용하는 언어 샘플 간의 거리에 기반한 단일 연결 방식(single linkage method)으로 클러스터를 형성하는 방법에 초점을 맞추었습니다.

- **Technical Details**: 연구팀은 각 언어의 샘플들 사이의 거리를 기반으로 하는 단일 연결 방식을 사용하여 언어 트리를 구축합니다. '3-spider'는 세 개의 광선(rays)이 한 점에서 연결된 구조(T3)로, 이는 언어의 계통을 표현하는 데 사용됩니다. 논문은 세 언어를 동시에 분석하여 그들의 '바리센터(barycenter)'를 결정하고 있으며, 초기 결과는 'non-sticky'과 'sticky' 두 가지 유형의 표본 평균을 식별했습니다.

- **Performance Highlights**: 분석 결과에 따르면, 언어 샘플의 평균이 'non-sticky'속성을 보이는 경우, 한 언어가 다른 두 언어와 다른 조상으로부터 온 것일 수 있으며, 'sticky' 속성을 보이는 경우, 세 언어 모두 공통의 조상을 공유하거나 모든 언어가 서로 다른 조상을 가질 수 있습니다. 이는 언어의 유전적 연관성과 발전을 이해하는 데 중요한 통찰을 제공합니다.



### Pseudo-Prompt Generating in Pre-trained Vision-Language Models for Multi-Label Medical Image Classification (https://arxiv.org/abs/2405.06468)
- **What's New**: 의료 이미지 인식의 복잡성을 해결하기 위해 새로운 접근법인 Pseudo-Prompt Generating (PsPG)이 제안되었습니다. 이 방법은 멀티-레이블 제로-샷 학습(zero-shot learning)에 초점을 맞추고 있으며, 자연어 처리(NLP)에서 영감을 받은 프롬프트 생성 기법을 사용합니다. PsPG는 다중 모달 특성의 사전 지식을 활용하여 클래스별 맞춤형 임베딩 벡터, 즉 가짜 프롬프트(pseudo-prompts)를 생성합니다.

- **Technical Details**: PsPG는 RNN 기반 디코더를 특징으로 하며, 이 디코더는 클래스에 특화된 임베딩 벡터를 자동 회귀적으로 생성합니다. 이 접근 방식은 비전-언어 모델(VLMs)을 하류 작업(downstream tasks)에 적응시키기 위해 프롬프트 튜닝(prompt tuning) 과정을 자동화하는 데 효과적입니다. 이전의 CoOp 기반 전략과 달리, PsPG는 보이지 않는 카테고리에 대한 클래스별 프롬프트 수행이 가능하여, 세밀한 시나리오에서의 일반화(generalizability)를 향상시킵니다.

- **Performance Highlights**: 다양한 멀티-레이블 체스트 X-레이 데이터셋에서의 비교 평가를 통해, PsPG는 주요 의료 비전-언어 및 멀티-레이블 프롬프트 학습 방법들에 비하여 우수함을 확인하였습니다. PsPG는 고도로 특화된 프롬프트를 제공하여 더 정확한 의료 이미지 분류가 가능하게 합니다.



### LMD3: Language Model Data Density Dependenc (https://arxiv.org/abs/2405.06331)
Comments:
          10 pages in the main body

- **What's New**: 새롭게 개발된 방법론은 언어 모델의 작업 성능을 개별 예제 수준에서 분석하고, 훈련 데이터의 밀도 추정에 기반을 둔다. 이를 통해 특정 테스트 쿼리에 대한 훈련 분포의 지원을 증가시키면 밀도가 증가하고, 이는 개선된 성능의 중요한 예측 요인이 될 수 있다는 것을 보여준다.

- **Technical Details**: 본 연구에서는 미세조정(finetuning) 데이터에 대한 패러프레이징(paraphrasing)을 통제된 개입으로 사용하며, 특정 테스트 쿼리를 위한 훈련 분포의 지원을 증가시키는 실험을 수행하였다. 이러한 증가는 밀도 측정치와 성능 향상 사이의 상관관계를 입증한다. 사전 훈련 데이터(pretraining data)와 관련된 실험은 모델의 복잡성(perplexity) 변동의 상당 부분을 밀도 측정을 통해 설명할 수 있음을 보여준다.

- **Performance Highlights**: 개선된 훈련 분포는 테스트 태스크(test task)에 대한 지원을 효과적으로 캐릭터화할 수 있을 뿐만 아니라, 타겟 모델의 예측이 훈련 데이터의 특정 부분에 어떻게 의존하는지에 대한 통계적 증거를 제공한다. 결과적으로, 훈련 데이터 밀도 추정을 통해 모델 성능에 영향을 미치는 요인을 식별할 수 있다.



### Decoding Emotions in Abstract Art: Cognitive Plausibility of CLIP in Recognizing Color-Emotion Associations (https://arxiv.org/abs/2405.06319)
Comments:
          To appear in the Proceedings of the Annual Meeting of the Cognitive Science Society 2024

- **What's New**: 이 연구에서는 추상적인 시각 예술에서 불러일으키는 감정을 인식하는 데 있어 사전 훈련된 다중 모드 모델, CLIP의 인지적 타당성을 조사합니다. 인간 평가자가 제공한 감정 라벨과 이 라벨의 텍스트 설명이 포함된 데이터셋을 활용하여, 이미지와 텍스트에 대한 제로 샷(zero-shot) 감정 분류, 유사성 기반 예측, 색상-감정 연결을 분석했습니다.

- **Technical Details**: CLIP을 사용하여, 이미지와 텍스트의 감정 분류를 실시한 결과, CLIP는 추상적 이미지와 설명에서 감정을 인식하는 데 있어 기준(baseline)을 상회하지만 인간의 인지 과정과는 잘 일치하지 않는 것으로 나타났습니다. 또한 색상과 감정 사이의 상호작용을 탐구하였으며, 예상되는 색상-감정 연관성(예: 붉은 색과 분노)이 이미지와 인간 및 CLIP에 의해 감정 라벨이 달린 텍스트에서 확인되었습니다.

- **Performance Highlights**: CLIP이 추상적 예술 작품의 감정을 분석하는 능력은 기준보다는 우수하지만 인간의 인지 과정과 완벽하게 일치하지 않는 것으로 나타났습니다. 색상-감정 연관성에서는 클립(CLIP)이 인간보다 강한 상호작용을 보였습니다.



### Learning from String Sequences (https://arxiv.org/abs/2405.06301)
Comments:
          10 pages, 1 figure, 4 tables, Technical Report

- **What's New**: 새로운 유형의 거리 측정 방법인 Universal Similarity Metric (USM)이 시퀀스 데이터 간의 '유사성'을 측정하는 데 유용하게 사용됨을 보여줍니다. 이 연구에서는 USM을 K-Nearest Neighbours (K-NN) 학습자의 거리 메트릭으로 사용하여 가변 길이 시퀀스 데이터의 패턴 인식에 효과적으로 적용할 수 있음을 입증하고 있습니다.

- **Technical Details**: 본 논문에서는 USM 기반 K-NN 방법을 일반적으로 사용되는 문자열-단어 벡터(string-to-word vector) 접근 방식과 비교합니다. 실험은 두 가지 상이한 분야의 데이터 세트에서 수행되었습니다: 1) 스팸 이메일 필터링과 2) 단백질의 세포 내 위치(protein subcellular localization).

- **Performance Highlights**: USM 기반 K-NN 학습자는 문자열-단어 벡터 접근 방식을 사용한 기술들보다 높은 분류 정확도를 제공합니다. 또한, 이 방법은 신뢰할 수 있는 확률 예측을 생성하는 데 사용될 수 있습니다.



### XAI4LLM. Let Machine Learning Models and LLMs Collaborate for Enhanced In-Context Learning in Healthcar (https://arxiv.org/abs/2405.06270)
- **What's New**: 이 연구는 의료 진단에 대한 Large Language Models (LLM)의 통합이 임상 의사 결정에 유망할 수 있음을 보여줍니다. 특히, 의료 분야 지식을 통합하는 새로운 zero-shot/few-shot in-context learning (ICL) 방법과 두 가지 통신 스타일(Numerical Conversational (NC), Natural Language Single-Turn (NL-ST))의 효과를 탐구합니다.

- **Technical Details**: 개발된 방법은 multi-layered structured prompt를 사용하여 ICL을 수행하고, 920개 환자 기록을 사용하여 진단 정확도 및 위험 요소(성별 편향, false negative rates)를 평가합니다. 여기에는 기존의 임상 machine learning (ML) 모델과의 성능 비교도 포함됩니다.

- **Performance Highlights**: 전통적인 ML 모델이 zero-shot 및 few-shot setting에서 LLM을 일반적으로 능가하지만, 충분한 수의 예제와 효과적인 explainable AI (XAI) 방법을 사용할 때 성능 격차가 크게 좁혀집니다. 특히, NC 스타일은 충분한 시간과 증가된 예제 수를 사용할 때 ML 모델의 성능에 근접합니다. LLM은 비용 감지 정확도(cost-sensitive accuracy)에서 ML 모델과 비교해 상응하거나 우수함을 보여줍니다.



### SKVQ: Sliding-window Key and Value Cache Quantization for Large Language Models (https://arxiv.org/abs/2405.06219)
- **What's New**: 이 논문에서는 LLMs의 KV 캐시(Key-Value Cache)를 효율적으로 압축하는 새로운 전략인 SKVQ(Sliding-Window KV Cache Quantization)을 소개합니다. 긴 텍스트 처리를 위한 메모리 부담을 줄이면서 정확도를 유지할 수 있는 방법을 제공하여, LLMs의 활용 범위를 넓힐 수 있습니다.

- **Technical Details**: SKVQ는 KV 캐시의 채널을 재배열하여 양자화 그룹 내의 채널 유사성을 향상시키고, 그룹 수준에서 클립된 동적 양자화(Clipped Dynamic Quantization)를 적용합니다. 이 기법은 최근 윈도우 토큰을 높은 정밀도로 유지함으로써 중요하지만 소량의 KV 캐시 정보 유지에 집중합니다.

- **Performance Highlights**: SKVQ는 KV 캐시를 2-bit 키와 1.5-bit 값으로 양자화하면서 정확도의 미미한 손실만을 야기함으로써 이전의 양자화 방식들을 능가하는 성능을 보여줍니다. SKVQ를 사용함으로써 80GB 메모리 GPU에서 7b 모델의 최대 1M 컨텍스트 길이를 처리할 수 있으며, 디코딩 속도가 최대 7배 빨라집니다.



### VLSM-Adapter: Finetuning Vision-Language Segmentation Efficiently with Lightweight Blocks (https://arxiv.org/abs/2405.06196)
Comments:
          12 pages, 5 figures, 2 tables

- **What's New**: 최근 Foundation Vision-Language Models (VLM)는 Vision-Language Segmentation Models (VLSMs)로 발전하여 텍스트 프롬프트를 사용하여 이미지 세분화를 안내하는 데 사용되었습니다. 이 연구에서는 의료 이미지에 특화된 모델 튜닝을 위해 적은 수의 훈련 가능한 파라미터를 사용하는 새로운 어댑터, VLSM-Adapter를 소개합니다.

- **Technical Details**: VLSM-Adapter는 이미 훈련된 VLSM에 적용될 수 있는 Transformer 인코더를 사용하는 어댑터입니다. 이 어댑터는 CLIP 기반의 세분화 모델에 적용되어, 전체 모델을 다시 훈련할 필요 없이 주어진 도메인의 데이터셋에 맞게 중간 표현을 조정합니다. 고정된 원래 모델 무게(weights)를 유지하면서, 경량 어댑터 모듈이 도입되어 훈련 가능한 파라미터가 크게 줄어들었습니다.

- **Performance Highlights**: VLSM-Adapter는 단 3백만의 훈련 가능한 파라미터만을 가지고도 최고의 성능을 발휘하며, 본질적으로 더 많은 파라미터를 요구하는 전체 모델 미세 조정(end-to-end fine-tuning)과 비교할 때 경쟁력 있는 결과를 보여주었습니다. 이는 특히 데이터셋이 작은 특수 분야에서 비용 효율적이며 자원 소모가 큰 훈련 과정을 크게 줄일 수 있는 가능성을 제시합니다.



### Narrative to Trajectory (N2T+): Extracting Routes of Life or Death from Human Trafficking Text Corpora (https://arxiv.org/abs/2405.06129)
- **What's New**: 이 논문에서는 'Narrative to Trajectory (N2T+)'라는 새로운 시스템을 제안합니다. N2T+ 시스템은 이주민들의 증언을 분석하여 인신 매매 루트를 자동으로 찾아내고 지도에 표시하는 기술을 개발하였습니다. 이러한 시스템은 기존의 방법들과 달리 데이터 과학(Data Science) 및 자연 언어 처리(Natural Language Processing, NLP) 기술을 활용하여 더욱 효과적으로 작동합니다.

- **Technical Details**: N2T+는 인신 매매 관련 증언에서 위치 이름을 추출하고, 해당 이름의 모호성을 해결하여 정확한 지리적 위치를 파악합니다. 이를 위해, 고급 NLP 기술과 이름 식별 및 중의성 해소(disambiguation) 메커니즘을 사용합니다. 추출된 데이터는 지도 상에 인신 매매 루트로 시각화되어 표현됩니다.

- **Performance Highlights**: N2T+는 위치 검출(geolocation detection)에서 기존의 최신 기술보다 현저히 높은 성능을 보여줍니다. 기존 기술들과 비교평가(comparative evaluation)를 거친 결과, N2T+는 더 높은 정확도와 효율성을 제공함을 확인할 수 있습니다.



### Selective Fine-tuning on LLM-labeled Data May Reduce Reliance on Human Annotation: A Case Study Using Schedule-of-Event Table Detection (https://arxiv.org/abs/2405.06093)
Comments:
          21 pages

- **What's New**: 이 연구에서는 임상 시험 프로토콜에서 치료 계획을 명시하는 Schedule-of-Event (SoE) 표를 검출하기 위해 PaLM-2를 노이즈 라벨(gemini-pro 1.0에서 얻은)과 함께 parameter efficient fine-tuning (PEFT) 방법으로 파인 튜닝하였습니다. 특히, 라벨의 정확성을 높이기 위해 높은 신뢰도를 가진 라벨을 선택하는 필터링 메커니즘을 도입했습니다. 이는 자동 생성된 라벨의 노이즈를 줄이는 데 도움이 됩니다.

- **Technical Details**: PaLM-2 모델은 SoE 표 분류를 위해 fine-tuning되었으며, JSON 및 텍스트 표현이 일치하는 LLM 라벨을 사용하여 gemini-pro 1.0 모델 추론에서 합의를 이루어 훈련 데이터셋의 노이즈를 감소시켰습니다. 이는 특히 전문가 주석이 부족하거나 매우 비싼 분야에서 파인 튜닝을 통한 LLM 성능 개선을 위한 가능한 전략으로 제시됩니다.

- **Performance Highlights**: 파인 튜닝된 PaLM-2 모델은 기존 PaLM-2 및 gemini-pro 1.0 모델을 초과하는 성능을 보였으며, 인간 주석자가 만든 라벨로 파인 튜닝된 모델과 유사한 성능 수준을 달성했습니다. 이는 LLM 생성 라벨을 이용한 도메인 특화 작업에 효과적임을 시사합니다.



### LLMs for XAI: Future Directions for Explaining Explanations (https://arxiv.org/abs/2405.06064)
- **What's New**: 이 연구는 인공지능(AI) 시스템의 투명성과 해석 가능성을 강화하기 위해 대규모 언어 모델(Large Language Models, LLMs)을 사용하여 기계 학습(ML) 설명을 자연스럽고 인간이 읽을 수 있는 서술로 변환하는 방법을 탐구합니다. 연구팀은 기존의 설명 가능한 인공지능(XAI; Explainable Artificial Intelligence) 알고리즘을 통해 생성된 ML 설명을 개선하여 LLM들의 가능성을 탐구했습니다.

- **Technical Details**: 본 연구는 특히 대조적 설명 기법(SHAP; Shapley additive explanations)과 같은 이론적으로 근거한 설명 알고리즘을 사용하여 생성된 ML 설명을 자연어 서술로 변환하는 데 LLM을 사용합니다. 연구팀은 이를 위해 다양한 프롬프트 디자인을 실험하고, GPT-3.5와 GPT-4와 같은 LLM들의 효과를 비교했습니다. 또한, 추가 주입 학습(finetuning) 방식과 도메인 관련 외부 데이터를 통합하는 방법을 탐구했습니다.

- **Performance Highlights**: 연구팀은 초기 실험과 사용자 연구를 통해 GPT-4가 사운드니스(soundness), 완전성(completeness), 그리고 맥락 인식(context-awareness)에서 더 높은 성과를 보였지만, GPT-3.5는 더 짧고 유창한(fluent) 반응을 제공하는 것으로 나타났습니다. 또한, 사용자 연구는 참가자 대다수가 전통적인 설명 방법보다 서술 기반 설명을 더 쉽게 이해하고 정보적으로 유용하다고 평가했다는 점에서 LLM 기반 서술 설명의 잠재적 이점을 시사합니다.



### Large Language Models Show Human-like Social Desirability Biases in Survey Responses (https://arxiv.org/abs/2405.06058)
Comments:
          3 pages, 2 figures, submitted to PNAS Nexus

- **What's New**: 이 연구는 대규모 언어 모델(Large Language Models, LLMs)의 인간 행동 모델링에 관한 새로운 사회적 바람직성 편향(Social desirability bias)을 발견하였습니다. GPT-4 등 다양한 최신 LLMs에서 이러한 편향이 존재하며, 이는 심리 측정 항목을 이용한 LLM의 프로파일링에 제약을 줄 수 있습니다.

- **Technical Details**: 연구팀은 스탠다드화된 Big Five 성격 설문조사를 사용해 LLM의 반응 편향을 실험적으로 평가했습니다. 이를 통해 LLM이 평가 상황을 인지하고 있을 때, 바람직한 성격 특성 점수로 결과를 왜곡하는 것을 확인했습니다. 특히, GPT-4와 Llama 3 모델에서 더 높은 편향이 관찰되었습니다. 실험은 문항의 순서 무작위화, 변형, 그리고 질문세트의 온도 설정을 조절하여 진행되었습니다.

- **Performance Highlights**: LLMs는 각기 다른 문항 배치 크기(Qn​) 아래에서도 일관되게 편향된 반응을 보였으며, 특히 GPT-4는 사회적으로 바람직한 성격적 특징 점수가 크게 높아지는 경향을 보여주었습니다. 이러한 결과는 LLMs가 평가되고 있다고 인지하면 스코어를 조절할 수 있는 능력이 있음을 나타냅니다. 반전 코딩(Reverse-coding)은 편향을 줄이는 데 도움이 되었으나 완전히 제거하지는 못했습니다.



### LLM-QBench: A Benchmark Towards the Best Practice for Post-training Quantization of Large Language Models (https://arxiv.org/abs/2405.06001)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)의 정량화(quantization)에 초점을 맞추고 있으며, 최신 퀀타이제이션 도구인 LLMC를 개발하여 다양한 설정에서 LLMs의 정량화를 최적화하는 방법을 탐구합니다. LLMC는 사용자가 하드웨어 친화적인 고성능 정량화 LLM을 획득할 수 있도록 돕는 여러 PTQ 알고리즘을 지원하며, 모델 정량화의 효율성 및 정확성을 향상시킬 수 있는 새로운 방법을 제시합니다.

- **Technical Details**: 논문에서는 LLM의 정량화를 효율적으로 수행하기 위한 네 가지 주요 원칙을 제안합니다: 추론 효율성(inference efficiency), 정량화된 정확도(quantized accuracy), 보정 비용(calibration cost), 모듈화(modularization). 또한, PTQ(post-training quantization) 파이프라인을 개발하여, 단일 GPU를 사용하여 수백억 매개변수의 LLM을 몇 시간 내에 손실 없이 정량화할 수 있습니다. 이 연구는 전체적으로 600가지 이상의 실험을 통해 데이터 보정, 정량화 알고리즘, 그리고 정량화 구성 선택에 대한 중요한 통찰력을 제공합니다.

- **Performance Highlights**: LLMC 도구를 사용하여 다양한 모델과 데이터셋에서 진행된 실험을 통해, 최적의 정량화 전략을 식별하고 최고의 정확도 및 효율성 균형을 달성하는 방법을 파악했습니다. 결과적으로, LLMC는 플러그 앤 플레이(Plug-and-play) 정량화 도구로서 독보적인 PTQ 알고리즘을 제공하며, 다양한 인프라 및 하드웨어에서 정량화된 LLMs를 배포하는 데 사용될 수 있습니다.



### Special Characters Attack: Toward Scalable Training Data Extraction From Large Language Models (https://arxiv.org/abs/2405.05990)
- **What's New**: LLM들이(training data) 훈련 데이터를 기억하고 간단한 반복 토큰들로 데이터 유출을 유발할 수 있다는 최근 연구에 이어, 특수 문자 또는 그와 영어 문자의 조합이 더 강력한 메모리 트리거(memory triggers)로 작용하여 데이터 유출을 더욱 심각하게 만들 수 있다는 것을 보여주어 새로운 발견을 제시합니다. 이는 특수 문자가 포함된 방대한 데이터로 훈련된 LLMs가 이러한 문자와 원문(raw texts) 사이의 공존을 기억할 수 있기 때문입니다.

- **Technical Details**: 새로운 공격 방법인 특수 문자 공격(Special Characters Attack, SCA)을 제안하여 LLMs가 훈련 데이터를 유출하게 유도합니다. SCA는 특수 문자와 영문자의 조합을 사용하여 LLMs를 속이고 데이터를 유출하게 만듭니다.

- **Performance Highlights**: SCA는 최신 LLMs에 대해 매우 높은 효과를 입증했습니다. 코드 저장소(code corpus), 웹 페이지, 개인 식별 정보(personally identifiable information) 등 다양한 훈련 데이터가 유출되었고, 때로는 계속되는 출력(non-stop outputs)을 부수적인 결과로 생성하기도 했습니다. 또한, 유출된 데이터를 검사함으로써 훈련 데이터 코퍼스의 구성을 드러낼 수 있으며, 이는 고성능 LLMs를 사전 훈련(pre-training)할 때 중요한 정보를 제공합니다.




New uploads on arXiv(cs.CL)

### Q-Sparse: All Large Language Models can be Fully Sparsely-Activated (https://arxiv.org/abs/2407.10969)
Comments:
          Work in progress

- **What's New**: Q-Sparse는 간단하면서도 효과적인 방법으로, 대형 언어 모델(LLM)의 활성화를 희소화하여 효율성을 크게 향상시키는 접근법입니다. 특히 활성화의 top-K를 선택하는 sparsification 기법과 학습에 straight-through-estimator 기법을 사용하여 모든 활성화를 완전 희소화할 수 있습니다. Q-Sparse는 학습 비용을 유지하면서도 추론 단계에서 상당한 효율성 향상을 달성합니다.

- **Technical Details**: Q-Sparse의 주요 변화는 LLM의 선형 투사(즉, 행렬 곱셈)에 있습니다. 각 선형 투사에 대해, 입력 텐서에서 top-K 활성화를 선택하는 top-K 희소화 함수를 적용합니다. 역전파 단계에서는 활성화의 기울기를 계산하기 위해 straight-through-estimator를 사용합니다. 또한, 희소화를 더욱 향상시키기 위해 squared ReLU 함수를 피드-포워드 레이어에 도입합니다. Q-Sparse는 완전 정밀도(full precision) 및 양자화된(quantized) LLM 모두에서 사용할 수 있습니다.

- **Performance Highlights**: Q-Sparse는 추론 단계에서 같은 계산 비용(Inference Compute Budget)으로 인해 기존의 밀집 모델(dense 모델)과 유사한 성능을 보여줍니다. 파라미터 수가 증가할수록 희소 모델과 밀집 모델 간의 성능 격차는 줄어듭니다. 약 40%의 희소성 비율을 갖는 희소 활성화 모델은 동일한 모델 크기 및 학습 토큰으로 밀집 모델과 유사한 성능을 보입니다. 특히, 제안된 Q-Sparse는 기초적인 모델 학습부터 기존 LLM의 지속 학습, 미세 조정(finetuning)까지 다양한 설정에서 효과적입니다.



### MMM: Multilingual Mutual Reinforcement Effect Mix Datasets & Test with Open-domain Information Extraction Large Language Models (https://arxiv.org/abs/2407.10953)
Comments:
          Under Review. 11 pages, 5 Figure

- **What's New**: Mutual Reinforcement Effect (MRE)을 활용하여 정보 추출 및 멀티태스킹을 연구하는 새로운 방법론과 이를 위한 다국어 데이터셋(Multilingual MRE Mix dataset, MMM)을 소개합니다. 영어, 일본어, 중국어를 포함한 21개의 하위 데이터셋으로 구성되어 있습니다.

- **Technical Details**: MMM 데이터셋을 구축하기 위해 대형 언어 모델(Large Language Models, LLMs)을 사용한 데이터셋 번역 방법을 제안합니다. 이는 원본 일본어 데이터셋을 번역하여 수동 주석 작업 시간을 크게 줄였습니다. 추가적으로, 오픈 도메인 명명 엔터티 인식(NER)과 문장 분류 작업을 포함하여 데이터셋을 확장했습니다.

- **Performance Highlights**: 이 확장된 데이터셋을 활용하여 Open-domain Information Extraction Large Language Model (OIELLM)을 개발했습니다. OIELLM 모델은 새로운 MMM 데이터셋을 효과적으로 처리하며 성능이 크게 향상되었습니다.



### Representing Rule-based Chatbots with Transformers (https://arxiv.org/abs/2407.10949)
Comments:
          Code and data are available at this https URL

- **What's New**: 새로운 연구에서는 Transformer 기반의 챗봇들이 자연스럽고 유창한 대화를 할 수 있는 기제를 이해하는 데 초점을 맞췄습니다. 기존 연구는 정규 표현식과 Dyck 언어 같은 형식 언어 작업을 위해 Transformer를 구축하는 접근 방식을 사용했지만, 이번 연구는 ELIZA 프로그램을 구현하는 Transformer를 구성하여 대화형 에이전트를 이해하려는 새로운 접근을 취했습니다.

- **Technical Details**: 본 연구에서는 유한 상태 자동자(finite-state automata)를 시뮬레이트하기 위한 기존의 구성 방식을 활용하여 ELIZA 같은 고전적인 규칙 기반 챗봇을 구현했습니다. 이를 위해 인덕션 헤드 메커니즘(Induction Head Mechanism)을 선호하는 모델과 정확한 위치 기반 복사 메커니즘(Position-Based Copying Mechanism)을 선호하는 모델을 비교했습니다. 또한, ELIZA의 기억 메커니즘을 시뮬레이트하기 위해 중간 결과물을 사용하는 재귀적인 데이터 구조를 탐구했습니다.

- **Performance Highlights**: 연구 결과, 신경망 챗봇과 해석 가능한 상징적 메커니즘 간의 명확한 연결을 그려내어 대화형 에이전트의 기계적 분석을 위한 새로운 환경을 제공했습니다. 이를 통해 모델들이 선호하는 메커니즘과 훈련된 ELIZA 대화 데이터를 분석하여 Transformer가 학습하는 기제를 설명했습니다.



### Learning from Naturally Occurring Feedback (https://arxiv.org/abs/2407.10944)
- **What's New**: 인공지능 언어 모델(LLM, Language Model)의 성능을 향상시키기 위해 사용자의 자연스러운 피드백을 추출하고 이를 모델 학습에 활용하는 새로운 방법이 제안되었습니다. 이 방법은 기존의 높은 비용과 확장성 문제를 해결할 수 있습니다.

- **Technical Details**: 제안된 방법에서는 인간과 모델 간의 상호작용에서 자연스럽게 발생하는 피드백을 추출합니다. 피드백의 유형은 긍정적 피드백(Positive Feedback)과 네 가지 부정적 피드백(Negative Feedback)으로 나뉩니다: 반복 또는 재구성(Repeat or Rephrase), 수정과 함께 인식(Make Aware with Correction), 수정 없이 인식(Make Aware without Correction), 명확화 요청(Ask for Clarification)입니다. 이 피드백을 추출하기 위해 1M 개의 대화를 분석하여 수십만 개의 피드백 샘플을 얻었습니다.

- **Performance Highlights**: 추출된 피드백을 이용해 모델을 학습한 결과, 테스트 케이스의 최대 78%에서 사전 학습된 모델보다 뛰어난 성능을 보였습니다. 이는 제안된 방법이 인간의 선호에 모델을 더 잘 맞추는 데 효율적임을 나타냅니다.



### Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together (https://arxiv.org/abs/2407.10930)
- **What's New**: 이 논문은 다단계 파이프라인에서 여러 언어 모델(LMs)과 프롬프트(prompt) 전략을 함께 최적화하여 성능을 향상시키는 방법을 제시합니다. 특히 중간 단계에 대한 골드 레이블이 없는 상황에서도 효율적인 최적화 전략을 평가했습니다. 실험 결과, 프롬프트와 가중치(weight)를 함께 최적화하는 간단한 접근 방식이 각기 따로 최적화하는 것보다 높은 성능을 보였습니다.

- **Technical Details**: 논문에서는 각 모듈의 프롬프트(π)와 언어 모델 가중치(θ)를 번갈아가며 최적화하는 전략을 제안했습니다. 이 최적화 문제는 프로그램의 기대 성능을 최대화하는 것으로 정의되며, 각 모듈의 프롬프트와 가중치를 업데이트하여 달성합니다. 최적화는 프롬프트와 가중치를 교대로 조정하면서 학습 라벨을 부트스트랩하는 방식으로 진행됩니다.

- **Performance Highlights**: HotPotQA의 멀티-홉 질의응답과 GSM8K의 수학적 추론, 그리고 Iris 데이터셋에서의 특징 기반 분류 실험에서 제안된 방법이 모든 언어 모델과 태스크에서 평균적으로 5%에서 136%로 성능 향상이 있었습니다. 이는 프롬프트만 최적화하거나 가중치만 최적화한 결과를 뛰어넘는 성과입니다.



### Weighted Grouped Query Attention in Transformers (https://arxiv.org/abs/2407.10855)
- **What's New**: 이번 연구에서는 Grouped-Query Attention (GQA)의 변형으로서 Weighted Grouped-Query Attention (WGQA)을 제안합니다. T5 디코더의 어텐션 블록 내에서 각 key와 value 헤드에 대해 새로운 학습 가능 매개변수를 도입하여 파인튜닝(finetuning) 동안 가중 평균을 가능하게 했습니다. 이를 통해 기존 GQA보다 평균 0.53% 향상된 성능을 달성하고, 인퍼런스(inference) 시 추가 오버헤드 없이 Multi-head Attention (MHA)과 유사한 성능을 보여줍니다.

- **Technical Details**: WGQA는 기존 GQA와 달리 key와 value 헤드에 대한 학습 가능한 가중치를 추가합니다. 이로 인해 파라미터 크기가 증가함에 따라 성능 차이가 더욱 뚜렷해집니다. WGQA 모듈은 key-value 헤드를 위한 추가적인 스칼라 또는 벡터 매개변수를 포함하며, 이를 통해 attention 계산 시 수정된 K와 V 매트릭스를 사용합니다. 이러한 방식은 기존의 element-wise mean을 사용하는 비결정론적 방법과는 다릅니다.

- **Performance Highlights**: WGQA 모델은 기존 GQA 대비 평균 0.53% 성능 향상을 보였으며, T5-small 및 T5-base 아키텍처 간의 비교를 통해 확장 법칙(scaling laws)이 유효함을 입증했습니다. 이는 기존의 MHA 성능에 도달하면서도 메모리 효율성을 유지할 수 있음을 보여줍니다.



### An Actionable Framework for Assessing Bias and Fairness in Large Language Model Use Cases (https://arxiv.org/abs/2407.10853)
- **What's New**: LLMs(대규모 언어 모델)은 다양한 방식으로 편향(bias)을 나타낼 수 있으며, 이는 특정 그룹에게 불공정한 결과를 초래할 수 있습니다. 이 논문은 LLM 활용 사례에서 편향과 공정성 리스크를 평가하기 위한 기술적 가이드를 제공합니다. 주요 기여는 특정 LLM 사례에 맞는 메트릭스를 결정할 수 있는 의사 결정 프레임워크입니다. 새로운 편향 및 공정성 메트릭스, 특히 반사실적(counterfactual) 메트릭스와 스테레오타입 분류기 기반 메트릭스를 소개합니다.

- **Technical Details**: 이 연구는 LLM 편향과 공정성 리스크를 범주화하고 이를 LLM 활용 사례의 분류체계에 맵핑했습니다. 다양한 리스크를 평가하기 위한 메트릭스를 공식적으로 정의하고, 실제 활용 사례의 프롬프트(prompt) 특성을 고려한 평가를 수행합니다. 평가 메트릭스는 LLM 출력만을 사용하므로 실무자가 쉽게 적용할 수 있습니다. 추가로, ROUGE, BLEU, 코사인 유사도(cosine similarity)의 반사실적 적응을 포함한 혁신적인 메트릭스도 소개됩니다.

- **Performance Highlights**: 이번 프레임워크는 실무에서 쉽게 적용 가능하며, 프롬프트와 모델 리스크를 모두 고려합니다. 제안된 평가 메트릭스는 생성된 텍스트, 추천 공정성, 분류 공정성 메트릭스 등으로 구성되어 있으며, LLM의 출력만을 필요로 합니다. 이러한 메트릭스는 프롬프트 기반 리스크를 집중적으로 평가하여 특정 활용 사례에 맞는 맞춤형 리스크 평가가 가능합니다.



### BiasScanner: Automatic Detection and Classification of News Bias to Strengthen Democracy (https://arxiv.org/abs/2407.10829)
Comments:
          10 pages, 3 figures, 1 table

- **What's New**: BiasScanner는 온라인 뉴스 기사에서 편향된 문장을 식별하는 대규모 언어 모델을 활용한 새로운 애플리케이션입니다. 뉴스 소비자들에게 보다 균형 잡힌 정보를 제공하기 위해 개발되었으며, 현재 20여 가지 이상의 언론 편향 유형을 문장 단위로 식별하고 분류할 수 있습니다. BiasScanner는 웹 브라우저 플러그인 형태로 제공되며, 사용자의 프라이버시를 존중하는 방식으로 구현됐습니다.

- **Technical Details**: BiasScanner는 OpenAI의 GPT-3.5-turbo-16k 모델을 활용하며, BABE 데이터셋을 기반으로 편향 유형과 강도 정보를 추가하여 학습시킨 모델입니다. 서버 측에서는 REST API를 사용하여 GPT-3.5 모델에 액세스하며, 사용자의 개인 식별 정보를 저장하지 않습니다. 프런트엔드 애플리케이션은 JavaScript로 구현되었으며, Mozilla의 가독성 라이브러리를 사용해 웹 페이지의 관련 텍스트를 추출합니다.

- **Performance Highlights**: BiasScanner는 뉴스 기사에서 편향된 문장을 강조 표시할 뿐만 아니라, 각 분류 결정에 대한 설명과 기사 전체에 대한 요약 분석을 제공합니다. 편향 레포트는 기사 내의 편향 문장의 비율과 평균 편향 점수를 기준으로 계산된 점수를 포함합니다. 사용자는 또한 연구 목적으로 편향 레포트를 기부할 수 있습니다.



### Foundational Autoraters: Taming Large Language Models for Better Automatic Evaluation (https://arxiv.org/abs/2407.10817)
Comments:
          31 pages, 5 figures, 7 tables

- **What's New**: 변화를 가져온 주요한 내용은, FLAMe라는 새로운 Foundational Large Autorater Models의 도입입니다. FLAMe는 100개 이상의 품질 평가 과제를 포함한 500만 개 이상의 인간 판단 데이터로 훈련되었으며, 이러한 데이터는 과거 연구에서 수집한 공개된 인간 평가 데이터를 수집하고 표준화한 것입니다. FLAMe는 다양한 작업에 대한 일반화 능력이 뛰어나며, GPT-4나 Claude-3와 같은 독점 데이터로 훈련된 모델들을 여러 작업에서 능가합니다.

- **Technical Details**: FLAMe의 데이터 수집은 기존 연구에서 얻은 100개 이상의 다양한 품질 평가 과제를 포함한 500만 개 이상의 인간 판단 데이터로 이루어졌습니다. 이 데이터는 하나의 통합된 텍스트-텍스트 형식으로 변환되어, 각 작업의 맥락과 예상되는 인간 평가 내용을 포함합니다. FLAMe는 PaLM-2-24B 모델을 기반으로 다중 작업 지시 조정(multi-task instruction tuning)을 통해 훈련되었으며, 일반적인 품질 평가 능력을 학습하도록 설계되었습니다.

- **Performance Highlights**: FLAMe-RM-24B 모델은 RewardBench에서 87.8%의 정확도를 달성하였으며, 이는 GPT-4-0125 (85.9%)와 GPT-4o (84.7%)를 능가하는 성능입니다. 또한, FLAMe-Opt-RM 모델은 새롭고 효율적인 tail-patch 미세 조정 전략을 사용하여 25배 적은 학습 데이터로도 경쟁력 있는 성능을 보여줍니다. 전체적으로 FLAMe의 다양한 변형 모델은 8개의 평가 벤치마크에서 기존의 LLM-as-a-Judge 모델보다 더 나은 성능을 보였습니다.



### Employing Sentence Space Embedding for Classification of Data Stream from Fake News Domain (https://arxiv.org/abs/2407.10807)
Comments:
          8 pages, 8 figures

- **What's New**: 이 논문은 문장 공간(sentece space) 방법을 사용하여 텍스트 데이터를 이산 디지털 신호 형태로 인코딩함으로써 데이터 스트림(data stream) 분류에 대한 새로운 접근 방식을 제안합니다. 이를 통해 이미지 분류에 사용되는 심층 신경망을 텍스트 데이터를 기반으로 가짜 뉴스(fake news) 인식 작업에 적용할 수 있습니다. 이 연구는 실제 Fakeddit 데이터셋을 기반으로 현재 최고의 데이터 스트림 분류 알고리즘과 비교하여 일반화 능력 및 시간 복잡성 측면에서 성능을 분석합니다.

- **Technical Details**: 자연어 처리(NLP) 작업을 위한 신경망과의 결합에서는 거의 항상 임베딩(embeddings)이 사용됩니다. 단어2벡(Word2Vec), FastText 및 GloVe와 같은 방법을 통해 단어를 벡터 공간으로 변환합니다. 최근에는 대형 언어 모델(Large Language Models, LLMs)인 BERT, MiniLM 등이 텍스트 표현을 개선하는 데 활용되고 있습니다. 이 논문에서는 문장 공간(sentence space) 표현을 사용하여 짧은 텍스트를 이미지 형태로 변환하고, 이미지 분류에 특화된 컨볼루션 신경망(CNN)을 사용하여 스트리밍 데이터의 텍스트를 분류하는 방법을 제안합니다.

- **Performance Highlights**: 제안된 스트리밍 문장 공간(Streaming Sentence Space, SSS) 접근법은 현재 최고의 데이터 스트림 앙상블 분류 알고리즘과 비교하여 분류 정확도와 계산 복잡성 측면에서 우수한 성능을 보였습니다. 특히 SSS는 온라인 뉴스 포털과 소셜 미디어 플랫폼에서 발생하는 정보의 홍수 속에서도 새로운 사실과 언어 동태에 적응하는 모델을 제공하여 가짜 뉴스 인식 성능을 향상시켰습니다.



### Think-on-Graph 2.0: Deep and Interpretable Large Language Model Reasoning with Knowledge Graph-guided Retrieva (https://arxiv.org/abs/2407.10805)
- **What's New**: 최신 연구인 Think-on-Graph 2.0(ToG 2.0)은 향상된 Retrieval-augmented Generation(RAG) 프레임워크를 제안합니다. 이 프레임워크는 질문을 지식 그래프와 정렬하여 이를 탐색 도구로 사용합니다. ToG 2.0은 RAG 패러다임을 심화하고 정제하여 정보 수집과 통합을 최적화하고 논리적 일관성을 유지하며, 정확하고 상호 운영성을 갖춘 정보를 제공할 수 있습니다.

- **Technical Details**: ToG 2.0은 문서에서 비정형 지식을 통합하고 지식 그래프(KGs)에서 구조화된 통찰력을 제시하여 복잡한 문제 해결 능력을 향상시킵니다. 이 접근법은 질문을 지식 그래프와 정렬하고, 지식 그래프를 탐색 도구로 사용하여 의미적 유사성을 보장하며 사실적 일관성을 유지하고, 장거리 연관성을 강화하여 논리적 일관성을 유지합니다. 주요한 혁신은 주제 가지치기(Topic Pruning), 관계 가지치기 최적화(Relation Pruning Optimization), DPR 기반 엔티티 랭킹(DPR-based Entity Ranking)과 같은 전략을 통해 탐색 속도와 응답 품질의 균형을 맞추는 것입니다.

- **Performance Highlights**: ToG 2.0은 네 개의 공개 데이터셋에서 기존 방법들보다 뛰어난 성능을 입증했습니다. 이 시스템은 LLM의 정확성과 신뢰성을 향상시키며, 인간과 유사한 문제 해결 능력을 보여주었습니다. 실험 결과, ToG 2.0은 LLM의 대답 정확성 및 신뢰성을 크게 향상시켜, 신뢰할 수 있는 하이브리드 구조화 지식 시스템의 잠재력을 보여줍니다.



### Mix-CPT: A Domain Adaptation Framework via Decoupling Knowledge Learning and Format Alignmen (https://arxiv.org/abs/2407.10804)
Comments:
          LLM, CPT, knowledge learning, format alignment; work in progress

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 도메인 적응 문제를 해결하기 위한 새로운 프레임워크인 Mix-CPT를 제안합니다. 이 프레임워크는 도메인 지식 학습과 일반 포맷 정렬의 두 가지 주요 단계로 나뉘며, 도메인 지식 혼합 지속적 사전 훈련을 통해 지식 암기와 활용을 동시에 개선합니다.

- **Technical Details**: 제안된 Mix-CPT 프레임워크는 두 단계로 구성됩니다. 첫 번째 단계에서는 지식 혼합 지속적 사전 훈련(Knowledge Mixture Continual Pre-training)을 수행하여 지식 암기와 활용을 동시에 강화합니다. 이 과정에서 로그잇 스왑 자기 증류(Logit Swap Self-Distillation)를 도입하여 카타스트로픽 포겟팅(catastrophic forgetting)을 방지합니다. 두 번째 단계에서는 소수의 일반적인 학습 샘플을 활용해 인스트럭션 튜닝 및 정렬을 수행하여 포맷 정렬을 달성합니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, 제안된 Mix-CPT 프레임워크는 전통적인 도메인 적응 방법에 비해 도메인 특정 작업과 일반 작업에서 대형 언어 모델의 성능을 동시에 향상시킬 수 있음을 보여줍니다. 이는 17개의 대표적인 벤치마크를 기반으로 한 7개의 뚜렷한 기능을 포함한 총평가로 검증되었습니다.



### Multilingual Contrastive Decoding via Language-Agnostic Layers Skipping (https://arxiv.org/abs/2407.10795)
- **What's New**: 이번 연구에서는 DoLa(Decoding by Contrasting Layers)를 개선한 새로운 대조 디코딩 알고리즘을 제안합니다. 이 알고리즘은 비영어권 작업에서도 효과적으로 적용될 수 있으며, LLM(Large Language Models)의 체인-오브-쏘트(chain-of-thought) 추론 정확도를 다양하게 향상시킵니다.

- **Technical Details**: 기존 DoLa의 문제점은 초기 출력(아마추어 로짓, amateur logits)과 최종 출력(전문가 로짓, expert logits) 사이의 언어 불일치에서 발생합니다. 이를 해결하기 위해 두 가지 전략을 사용해 더 유의미한 아마추어 로짓을 얻는 개선된 대조 디코딩 알고리즘을 제안합니다. 첫 번째 전략은 언어 비중립층(language-agnostic layers)을 건너뛰는 것이며, 두 번째는 엔트로피 변화를 기반으로 레이어 건너뛰는 위치를 동적으로 결정하는 것입니다.

- **Performance Highlights**: 실험 결과, 새로운 방법은 기존 DoLa 접근법보다 뛰어난 성능을 보이며, 11개 언어에서 체인-오브-쏘트 추론 정확도를 크게 향상시켰습니다. 특히, LLaMA2, LLaMA3, Mistral 등 다양한 오픈소스 LLM에서 우수한 성과를 시연했습니다.



### Graphusion: Leveraging Large Language Models for Scientific Knowledge Graph Fusion and Construction in NLP Education (https://arxiv.org/abs/2407.10794)
Comments:
          24 pages, 11 figures, 13 tables. arXiv admin note: substantial text overlap with arXiv:2402.14293

- **What's New**: Graphusion은 무료 텍스트로부터의 zero-shot 지식 그래프 구축(KGC)을 위한 혁신적인 프레임워크를 소개합니다. 이 시스템은 기존의 로컬 관점 대신 글로벌 관점을 채택하여, 엔티티 병합, 충돌 해결, 새로운 트리플렛(triplet) 발견을 포함한 통합 방법을 제공합니다. 특히 'TutorQA'라는 새로운 QA 벤치마크를 통해 교육 시나리오에서 Graphusion의 성능을 검증하였습니다.

- **Technical Details**: Graphusion 프레임워크는 LLM(대형 언어 모델)을 사용하여 지식을 추출하고 통합하는 새로운 접근방식을 제시합니다. 이 프레임워크는 트리플렛 후보들을 추출하고 여러 소스에서 엔티티와 관계를 통합합니다. 이렇게 함으로써, 보다 포괄적이고 정확한 지식 그래프를 구축할 수 있습니다. 또한, zero-shot KGC를 지원하여, 사전 훈련 없이도 효과적인 성능을 발휘합니다.

- **Performance Highlights**: Graphusion은 링크 예측(link prediction) 작업에서 감독 학습 기반 모델을 최대 10% 능가하며, 개념 엔티티 추출과 관계 인식에 대해 인간 평가에서 각각 2.92 및 2.37 점(3 점 만점)을 받았습니다. TutorQA 벤치마크에서는 1,200개의 QA 쌍을 통해 높은 정확도를 입증했습니다.



### GraphEval: A Knowledge-Graph Based LLM Hallucination Evaluation Framework (https://arxiv.org/abs/2407.10793)
Comments:
          12 pages, to be published at KiL'24: Workshop on Knowledge-infused Learning co-located with 30th ACM KDD Conference, August 26, 2024, Barcelona, Spain

- **What's New**: GraphEval은 Knowledge Graph (KG) 구조를 이용하여 대형 언어 모델(LLM)의 출력을 평가하고 일관성 결여 (hallucination)를 탐지하는 새로운 프레임워크입니다. 이 방법은 LLM 적응형 평가 프레임워크에서는 최초로 KG를 적용하여, 기존 방법보다 출력에서 일관성 결여가 발생한 특정 부분에 대한 더 높은 수준의 통찰을 제공합니다. 또한, GraphEval은 최신 자연어 추론(NLI) 모델과 함께 사용될 때, 다양한 일관성 결여 벤치마크에서 균형 잡힌 정확도를 향상시킵니다. 더불어, KG의 구조를 활용하여 GraphCorrect라는 이름의 방법을 통해 일관성 결여를 수정할 수 있음을 보여줍니다.

- **Technical Details**: GraphEval은 KG 구조로 정보를 표현하여 일관성 결여 탐지에서 설명력을 강화합니다. 구체적으로, KG는 (e1, r, e2) 형태의 트리플 컬렉션으로 구성되며, 여기서 e1, e2는 엔티티(entity), r은 관계(relationship)를 나타냅니다. GraphEval 메트릭은 두 단계로 진행됩니다: 1단계에서는 평가할 LLM 출력에서 KG를 구성하고, 2단계에서는 KG의 모든 트리플을 순회하며 각 트리플이 일관성 결여 (hallucination)가 있는지 식별합니다.

- **Performance Highlights**: GraphEval을 최신 자연어 추론(NLI) 모델과 결합하여 사용하면, 기존의 원시 NLI 모델을 사용하는 것에 비해 다양한 일관성 결여 벤치마크에서 균형 잡힌 정확도가 향상됩니다. 또한, GraphCorrect 메서드를 통해 LLM 출력의 대부분의 일관성 결여를 효과적으로 수정할 수 있음을 입증하였습니다.



### Codebook LLMs: Adapting Political Science Codebooks for LLM Use and Adapting LLMs to Follow Codebooks (https://arxiv.org/abs/2407.10747)
Comments:
          Presented at PolMeth 2024

- **What's New**: 이번 연구에서는 코드북(Codebooks)과 생성형 대형 언어 모델(LLMs)을 활용하여 정치학 텍스트 데이터를 자동으로 라벨링하고 분석하는 방법에 대해 논의합니다. 저자들은 정치학자들이 정확한 측정을 위해 코드북-구성 라벨 가정을 사용해야 함을 주장하며, 이를 입증하기 위해 세 가지 정치학 데이터셋과 원본 코드북을 실험에 사용했습니다.

- **Technical Details**: 이 연구에서는 Mistral 7B Instruct LLM을 사용하여 코드북 지침에 따라 텍스트를 라벨링하는 성능을 평가했습니다. 특히 코드북을 재구성하여 성능 향상을 시도했으며, 코드북-문서-라벨 튜플로 LLM을 instruction-tuning 한 실험을 수행했습니다. 재구성된 코드북은 제로샷 분류(Zero-shot classification) 성능에서 약간의 향상을 보였으나 여전히 코드북의 제약을 준수하는 데 어려움을 겪었습니다.

- **Performance Highlights**: instruction-tuning을 통한 모델 성능 향상은 고무적이었으며, 한 데이터셋에서 제로샷 분류보다 현저히 높은 성능(0.76 대비 0.53의 micro F1)을 보였습니다. 이는 코드북 특정 작업, 가정, instruction-tuning 파이프라인이 정치학자들에게 LLM 시대에 적응하는 데 도움이 될 수 있음을 시사합니다.



### What distinguishes conspiracy from critical narratives? A computational analysis of oppositional discours (https://arxiv.org/abs/2407.10745)
Comments:
          submitted to the Expert Systems journal

- **What's New**: 이 논문에서는 음모론과 비판적인 텍스트를 구분하는 새로운 주제-무관 주석 체계를 제안하고 있습니다. 이를 통해 집단 간 갈등(inter-group conflict)의 역할을 강조하며, COVID-19 관련 Telegram 메시지를 고품질로 주석 처리한 다국어 코퍼스인 XAI-DisInfodemics 코퍼스를 제공합니다. 또한, 자연어 처리(NLP) 기반 자동화를 통해 강력한 기본 솔루션을 도출하는 다양한 실험을 수행하였습니다.

- **Technical Details**: 논문은 음모론(conspiracy)과 비판적인 텍스트를 구분하는 신규 주석 체계를 제안하며, 주석 체계는 에이전트(Agents), 피해자(Victims), 조력자(Facilitators), 캠페이너(Campaigners)와 같은 다양한 범주로 나뉩니다. COVID-19와 관련된 Telegram 메시지를 사용한 다국어(영어 및 스페인어) XAI-DisInfodemics 코퍼스를 구성하고, 이 코퍼스를 바탕으로 여러 NLP 태스크를 수행했습니다. 여기에는 음모론 vs. 비판적인 텍스트를 구분하는 이진 분류(binary classification)와 주요 요소 탐지와 같은 다중 라벨 주석 태스크(multi-label annotation task)가 포함됩니다.

- **Performance Highlights**: 최신 NLP 방법을 사용하여 음모론 vs. 비판적인 텍스트 구분 작업을 성공적으로 수행할 수 있음을 입증하였습니다. 또한, 집단 간 갈등 요소를 포함한 이야기 요소를 자동으로 탐지할 수 있다는 것도 확인했습니다. 특히 음모론 텍스트는 비판적인 텍스트에 비해 집단 간 갈등을 더 강하게 초래하고 분노와 정치적 폭력과 관련된 단어를 더 많이 전달한다는 사실을 밝혔습니다.



### CLAVE: An Adaptive Framework for Evaluating Values of LLM Generated Responses (https://arxiv.org/abs/2407.10725)
- **What's New**: 새로운 프레임워크 CLAVE가 소개되었습니다. CLAVE는 두 가지 상호보완적인 언어 모델을 통합하여 LLM 생성 텍스트의 가치를 평가하는 혁신적인 접근법입니다. 또한 다양한 도메인과 세 가지 주요 가치 체계를 포함하는 광범위한 데이터셋인 ValEval도 함께 제시되었습니다.

- **Technical Details**: CLAVE는 대형 LLM을 이용하여 소수의 인간 라벨에서 고차원적인 가치 개념을 추출하고, 이러한 개념을 바탕으로 작은 모델을 미세 조정하여 인간의 가치 이해에 맞추는 듀얼 모델 어프로치를 사용합니다. 이를 통해 최소한의 인간 라벨링 샘플(<100)로 다양한 가치 체계에 대한 캘리브레이션이 가능합니다. ValEval 데이터셋은 13,000개 이상의 (텍스트, 가치, 라벨) 튜플을 포함하고 있으며, 이는 세 가지 주요 가치 체계를 다룹니다.

- **Performance Highlights**: 12개 이상의 인기 있는 LLM 평가자를 벤치마크하여 이들의 강점과 약점을 분석했습니다. 그 결과, 미세 조정된 작은 모델(Small Models)과 프롬프트 기반의 대형 모델(Large Models)을 결합하는 것이 가치 평가에서 우수한 균형을 이루는 접근 방식임을 확인했습니다.



### DOCBENCH: A Benchmark for Evaluating LLM-based Document Reading Systems (https://arxiv.org/abs/2407.10701)
Comments:
          Work in progress

- **What's New**: DocBench는 LLM 기반 문서 읽기 시스템을 평가하기 위한 새로운 벤치마크입니다. 이 벤치마크는 사용자가 업로드한 문서와 그 문서와 관련된 질문에 대한 응답을 생성하는 시스템의 성능을 평가하도록 설계되었습니다. DocBench는 5개 도메인(학계, 금융, 정부, 법률, 뉴스)에 걸쳐 229개의 실제 문서와 1,102개의 질문을 포함하고 있으며, 이는 현재 사용 가능한 LLM 기반 시스템과 인간 성능 간의 공백을 드러냅니다.

- **Technical Details**: DocBench는 세 단계의 데이터셋 구축 파이프라인을 통해 개발되었습니다. 첫 번째로, 공용 리포지토리에서 다양한 도메인의 문서를 크롤링하여 PDF 파일로 표준화합니다. 두 번째로, GPT-4와 인간 어노테이터 팀의 도움을 받아 해당 문서에 대한 QA(질문-답변) 쌍을 생성합니다. 마지막으로, 자동 필터링과 수동 검토를 통해 생성된 데이터의 품질을 검토합니다. 이 과정에서, 다양한 형식의 정보(예: 텍스트, 표, 그림 및 메타데이터) 이해를 포함하여 문서의 다차원적 정보를 다루기 위한 fitz 라이브러리를 활용합니다.

- **Performance Highlights**: DocBench는 웹 인터페이스 또는 API로 접근 가능한 여러 상용 LLM 기반 시스템과 오픈 소스 LLM을 활용한 '파싱 후 읽기' 파이프라인을 평가합니다. 평가 결과, 현재 존재하는 LLM 기반 문서 읽기 시스템과 인간의 성능 간에는 눈에 띄는 격차가 있음을 발견하였습니다. 따라서, DocBench는 LLM 기반 문서 읽기 시스템의 객관적 평가를 위한 표준화된 벤치마크로서의 역할을 하며, 미래 연구를 위한 발전 방향을 제시합니다.



### Qwen2 Technical Repor (https://arxiv.org/abs/2407.10671)
Comments:
          25 pages, 1 figure

- **What's New**: Qwen2 시리즈를 소개합니다. 이 시리즈는 0.5억에서 720억 파라미터에 이르는 새로운 대형 언어 모델(Large Language Models, LLMs) 및 멀티모달 모델을 포함합니다. Qwen2는 이전 모델인 Qwen1.5를 뛰어넘으며 다양한 벤치마크에서 경쟁력 있는 성능을 보입니다.

- **Technical Details**: Qwen2 모델은 Transformer 아키텍처를 기반으로 하여 다음 토큰 예측을 통해 학습되었습니다. 파라미터 수는 0.5억, 1.5억, 7억, 720억, 그리고 Mixture-of-Experts (MoE) 570억 모델로 구성되어 있습니다. Qwen2는 7조 개의 토큰을 포함한 대규모 고품질 데이터셋으로 사전 학습되었습니다. 특히 Grouped Query Attention (GQA), Dual Chunk Attention (DCA), YARN과 같은 최신 기술을 도입하여 모델 성능을 개선하였습니다.

- **Performance Highlights**: 주요 모델인 Qwen2-72B는 MMLU 84.2점, GPQA 37.9점, HumanEval 64.6점, GSM8K 89.5점, BBH 82.4점을 기록했습니다. 여기에 Instruction-tuned 모델 Qwen2-72B-Instruct는 MT-Bench 9.1점, Arena-Hard 48.1점, LiveCodeBench 35.7점을 달성했습니다. 또한 Qwen2 모델은 영어, 중국어, 스페인어, 프랑스어, 독일어, 아랍어, 러시아어, 한국어, 일본어, 태국어, 베트남어 등 약 30개의 다국어를 지원합니다.



### Enhancing Retrieval and Managing Retrieval: A Four-Module Synergy for Improved Quality and Efficiency in RAG Systems (https://arxiv.org/abs/2407.10670)
Comments:
          ECAI2024 #1304

- **What's New**: 이 연구는 Retrieval-Augmented Generation (RAG) 시스템을 향상시키기 위한 여러 모듈을 제안합니다. 식별된 여러 문제점을 해결하기 위해 Query Rewriter+와 Knowledge Filter 모듈이 도입되었습니다. 또한, Memory Knowledge Reservoir와 Retriever Trigger 모듈을 통해 중복 검색 문제를 해결함으로써 전체 시스템 성능을 최적화합니다.

- **Technical Details**: Query Rewriter+ 모듈은 다중 쿼리 생성을 통해 단일 쿼리에서 발생하는 Information Plateaus 문제를 해결하고, 모호한 질문을 명확하게 변경하여 의도 파악을 개선합니다. Knowledge Filter 모듈은 Natural Language Inference (NLI) 작업을 수행하여 관련 없는 정보를 걸러내는 역할을 합니다. Memory Knowledge Reservoir는 캐싱 메커니즘을 통해 반복적인 질문에 대해 빠른 정보 검색을 가능하게 하고, Retriever Trigger는 외부 지식 검색의 필요성을 판단하여 자원 활용을 최적화합니다.

- **Performance Highlights**: 이 연구에서 제안된 네 가지 모듈은 RAG 시스템의 응답 정확도와 효율성을 향상시키는 데 시너지 효과를 발휘합니다. 6개의 일반 QA 데이터셋을 통해 실험과 ablation study가 수행되었으며, 제안된 방법이 직접 질문보다 5%-10% 더 높은 정확도를 보였습니다. 반복적인 질문에 대해서는 응답 시간을 46% 단축했습니다.



### An Empirical Study of Validating Synthetic Data for Formula Generation (https://arxiv.org/abs/2407.10657)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)을 활용하여 스프레드시트 공식 작성에 도움을 주기 위해, 합성된 자연어(NL) 문장을 사용하여 모델을 미세 조정하는 방법을 소개합니다. 이 연구는 합성된 NL 문장의 정확성을 검증하는 3개의 대리 목표(surrogate objectives)를 통해 합성 데이터의 검증이 모델 성능 향상에 어떻게 기여하는지를 실증적으로 분석합니다.

- **Technical Details**: 이 연구는 다음과 같은 구조를 가지고 있습니다. 먼저, 스프레드시트에서 사용할 수 있는 파생-칼럼 공식(derived-column formulas)을 작성하는 것이 어려운 문제라는 점을 인식합니다. 이를 해결하기 위해 기존의 테이블과 공식 데이터를 활용하여 이와 대응하는 자연어 문장을 생성합니다. 그런 다음, LLM을 사용하여 합성된 자연어 문장의 정확성을 예측하는 세 가지 대리 목표(출력 예측, 대체 코드 생성, 분류)를 정의합니다. 각각의 목표는 모델이 생성한 합성 문장이 실제 공식을 정확하게 설명하는지를 평가하고, 이를 통해 더 나은 미세 조정을 가능하게 합니다.

- **Performance Highlights**: 검증된 데이터로 미세 조정된 모델들은 더 복잡한 문제를 해결하는 능력이 향상되었습니다. 예를 들어, GPT-4는 대체 코드 생성 목표를 통해 검증된 데이터로 미세 조정된 결과, 평가 점수가 최대 25% 향상되었고, 훈련 시간도 23% 감소했습니다.



### Prompt Selection Matters: Enhancing Text Annotations for Social Sciences with Large Language Models (https://arxiv.org/abs/2407.10645)
- **What's New**: 대형 언어 모델(Large Language Models, LLMs)이 최근 사회 과학에서 텍스트 주석 작업에 적용되면서 인간 작업자의 성능을 능가하거나 비용을 획기적으로 절감할 수 있음을 보여준다. 그러나 프롬프트 선택(prompt selection)이 라벨링 정확도에 미치는 영향에 대한 연구는 아직 없었다. 이 연구에서는 성능이 프롬프트에 따라 크게 달라지며, 자동 프롬프트 최적화(automatic prompt optimization) 방법을 사용하여 고품질 프롬프트를 체계적으로 구성하는 방법을 제시한다. 또한 간단한 브라우저 기반 구현을 제공하여 커뮤니티에 기여한다.

- **Technical Details**: 프롬프트 선택(prompt selection)을 통한 성능 변동성을 조사하기 위해, 다양한 표준 분류 작업에서 수작업으로 작성된 프롬프트와 자동으로 최적화된 프롬프트를 비교하였다. 자동 프롬프트 최적화를 통해 일관된 높은 성능을 달성했으며, 대부분의 작업에서 단순 프롬프트 제작 방법(예: Chain of Thoughts prompts)을 능가한다는 것을 발견했다. 또한, 연구자들이 쉽게 자신들의 분류 작업에서 최적의 프롬프트를 적용할 수 있도록 단계별 설명을 제공한다.

- **Performance Highlights**: 대형 언어 모델(예: GPT-4)을 사용하면 트윗과 뉴스 기사에서 부정적 감정 탐지의 경우 95% 이상의 정확도, 트윗에서 입장 탐지의 경우 75% 이상의 정확도, 또한 트윗의 정치적 성향 분류에서 92% 이상의 정확도를 달성할 수 있다. 이전에는 성취하기 어려웠던 성능이며, 전문 교육 없이 달성 가능하다는 점에서 큰 의의를 지닌다. 그러나 프롬프트의 선택이 정확도에 큰 영향을 미치므로, 최적화된 프롬프트를 사용하는 것이 중요하다.



### Arena Learning: Build Data Flywheel for LLMs Post-training via Simulated Chatbot Arena (https://arxiv.org/abs/2407.10627)
- **What's New**: 이번 논문에서는 Arena Learning이라는 혁신적인 오프라인 전략을 소개합니다. 이는 AI 기반 주석을 통해 챗봇 아레나 전투의 결과를 평가하여 목표 모델의 지속적인 개선을 가능하게 합니다. 이 접근법은 비용과 시간의 제약없이 대규모 언어 모델(LLM) 평가를 자동화할 수 있습니다.

- **Technical Details**: Arena Learning은 두 가지 핵심 요소로 구성됩니다. 첫째, WizardArena라는 파이프라인을 통해 오프라인 시뮬레이션과 온라인 경쟁의 일관성을 유지하면서 다양한 모델의 Elo 순위를 정확하게 예측합니다. 둘째, 전투 결과와 세분화된 모델을 바탕으로 훈련 데이터를 지속적으로 개선해나갑니다. 이 방법을 통해 목표 모델, WizardLM-β,의 성능을 현저히 향상시킬 수 있습니다.

- **Performance Highlights**: WizardArena가 생성한 Elo 순위는 LMSys Chatbot Arena와 평균 98.79%의 일관성을 보였습니다. 이는 Arena-Hard-v1.0보다 8.58%, MT-Bench보다 35.23% 더 높은 일관성을 보입니다. 이러한 결과는 Arena Learning이 대규모의 전투 데이터를 생성해 모델의 성능을 지속적으로 향상시키는 데 매우 효과적임을 입증합니다. 특히, 세 번의 반복 루프에서 모델의 성능이 각 라운드마다 현저히 개선되었습니다.



### NoviCode: Generating Programs from Natural Language Utterances by Novices (https://arxiv.org/abs/2407.10626)
- **What's New**: 현재 Text-to-Code 모델은 프로그래머가 제시한 기술 지침에 따라 실행 가능한 코드를 생성하는 데 놀라운 능력을 보입니다. 그러나 NoviCode는 비기술 사용자들이 제공하는 일상 언어 설명으로부터 복잡한 실행 가능한 프로그램을 생성하는 새로운 NL Programming 과제를 도입함으로써 이 도전에 응답합니다. 이는 API 접근과 제어 구조(loops, 조건, sequences 등)를 포함할 수 있는 프로그램을 생성하는 것을 목표로 합니다.

- **Technical Details**: NoviCode 과제는 비기술적인 사용자가 제공하는 자연어 설명과 API를 입력으로 받아 복잡한 프로그램을 생성합니다. 이를 평가하기 위해 새로운 벤치마크와 테스트 수트를 제공하며, 단순히 코드의 형태가 아니라 기능적 실행을 기준으로 코드의 유효성을 평가합니다. 이 작업은 모델이 코드의 계층적 구조와 자연어 발화를 정렬함으로써 성능이 크게 향상된다는 것을 보여 줍니다.

- **Performance Highlights**: 실험 결과, NoviCode 과제가 Code Synthesis 영역에서 매우 도전적이며, 비기술적인 지침으로부터 복잡한 코드를 생성하는 것이 현재의 Text-to-Code 패러다임을 넘어선다는 것을 입증했습니다. 새로운 접근법을 사용하여 자연어 발화와 코드의 구성적 계층 구조를 정렬하면, 성능이 크게 향상된다는 것을 볼 수 있었습니다.



### Boosting Zero-Shot Crosslingual Performance using LLM-Based Augmentations with Effective Data Selection (https://arxiv.org/abs/2407.10582)
Comments:
          Accepted in Findings of ACL 2024

- **What's New**: 대형 언어 모델(LLMs)이 특정 작업 데이터 생성을 통해 저자원(target language) 언어에 대한 교차 언어 전이를 촉진하는 방법을 제안합니다. 이 연구는 교사 모델(teacher model)과 데이터를 라벨링하는 LLM을 활용하여 효과적인 데이터 선택 전략을 제안합니다. 또한, 원본 소스 데이터와 LLM 생성 데이터의 라벨링 방법이 성능에 미치는 영향을 탐구합니다.

- **Technical Details**: 우리는 영어로 된 소스 데이터와 교사 모델을 이용해 LLM이 생성한 데이터를 라벨링하고, 교사 모델의 라벨 확률을 사용해 대표적인 하위 집합을 선택하는 전략을 채택합니다. 제안된 방법은 Llama-2와 같은 오픈소스 LLM을 활용하여 과제에 맞는 텍스트를 생성하고, 생성된 텍스트를 기계 번역 시스템을 통해 목표 언어로 번역하는 것입니다. 또한, Pseudo Labels(의사 라벨)은 LLM 프롬프트를 통해 생성되거나 교사 모델을 통해 생성됩니다. 학생 모델(Student Model)은 교사 모델의 부드러운 라벨로 훈련됩니다.

- **Performance Highlights**: 이 방법으로 감정 분석(Sentiment Analysis) 및 자연어 추론(Natural Language Inference) 작업에서 신뢰할 수 있는 성능 향상을 확인했습니다. 특히, 힌디어, 마라티어, 우르두어, 스와힐리어 등 여러 목표 언어와 도메인에서 최대 7.13 포인트의 절대 성능 향상과 평균 1.5 포인트의 절대 성능 향상을 달성했습니다.



### Beyond Generative Artificial Intelligence: Roadmap for Natural Language Generation (https://arxiv.org/abs/2407.10554)
- **What's New**: 최근 몇 년 동안 대규모언어모델(LLM)과 생성 인공지능(Generative AI)의 등장으로 자연어 생성(NLG)이 크게 발전했습니다. GPT-4, Bard 및 ChatGPT와 같은 도구가 NLG 연구의 새로운 표준이 되었습니다. 이 연구는 기존의 NLG 연구 동향을 점검하고, 아직 LLM에 의해 충분히 해결되지 않은 NLG 측면을 확인하며 향후 연구 방향을 제안하고자 합니다.

- **Technical Details**: 이 논문은 NLG 발전의 역사적 변천을 소개하며 1970년대 이후 NLG 아키텍처의 변화 과정을 설명합니다. 초창기 NLG 아키텍처는 매크로 계획(Macroplanning), 미세 계획(Microplanning), 실현(Realization)의 세 단계로 구성된 모듈러 아키텍처였습니다. 이후, 태스크 분할이 덜 엄격해진 계획 관점의 아키텍처로 발전했습니다. 현재의 글로벌 접근 방식은 통계적 학습에 의존하며, 트랜스포머(Transformer) 아키텍처를 사용하여 단일 단계에서 모든 생성 작업을 수행합니다. 최근의 연구는 더 큰 LLM을 개발하는 데 중점을 두고 있지만, 이러한 모델들은 여전히 정확성이 부족하고 인간처럼 텍스트를 생성하는 데 몇 가지 문제가 있습니다.

- **Performance Highlights**: 최근 몇 년간 조사된 16개의 NLG 서베이 논문을 분석한 결과 주요 연구 목표와 기여, 포함된 코퍼스(Corpora), 사용 방법론(techniques) 및 도구(tools) 등 다양한 측면을 확인했습니다. 최신 LLM 모델들이 여러 NLG 태스크에서 인상적인 성능을 보여주고 있지만, 여전히 구문 분석(syntactic parsing)이나 의미 조합(semantic compositionality), 인과 관계(causality relationships)와 같은 몇 가지 해결되지 않은 문제들이 남아 있습니다. 서베이 결과를 바탕으로 현재 연구 격차를 식별하고, 향후 NLG 연구가 초점을 맞춰야 할 몇 가지 주요 영역을 제안했습니다.



### TCM-FTP: Fine-Tuning Large Language Models for Herbal Prescription Prediction (https://arxiv.org/abs/2407.10510)
- **What's New**: 새로운 연구에서는 전통 중국 의약(TCM) 처방 예측의 주요 문제를 해결하기 위해 DigestDS라는 새로운 데이터셋을 소개했습니다. 이는 소화계 질환에 대한 실제 전문가들의 의료 기록을 포함합니다. 또한, TCM-FTP(세분화된 사전학습 모델)라는 새로운 방법을 제안하여 사전 학습된 대형 언어 모델(LLM)을 DigestDS 데이터셋에 대해 지도 학습 시킵니다.

- **Technical Details**: 제안된 TCM-FTP 방법은 low-rank adaptation(낮은 순위 적응) 기술을 사용하여 계산 효율을 향상시킵니다. 뿐만 아니라, 처방에서 약초 순서의 비순차적 특성을 활용하기 위해 데이터 증강(data augmentation) 방법도 채택했습니다. 이는 약초의 섞인 순서를 통해 데이터의 다양성을 높이는 방식입니다. LLM을 활용한 처방 예측 모델 개발의 중요성을 강조하며, 이를 효율적으로 수행하는 방안을 제시합니다.

- **Performance Highlights**: TCM-FTP 방법은 F1-score 0.8031을 달성했으며, 기존 방법들을 현저히 능가하는 성능을 보여주었습니다. 또한, 투여량 예측에서도 정확도가 매우 높아, 정규화된 평균 제곱 오차(NMSE)가 0.0604로 나타났습니다. LLM을 이용해 기존의 모델들보다 훨씬 더 정교하고 정확한 처방 예측이 가능함을 입증했습니다.



### CIBench: Evaluating Your LLMs with a Code Interpreter Plugin (https://arxiv.org/abs/2407.10499)
Comments:
          Under review

- **What's New**: 이 논문에서는 CIBench라는 새로운 인터랙티브 평가 프레임워크를 제안하여 LLM의 코드 인터프리터 활용 능력을 평가합니다. 이 프레임워크는 평가 데이터셋과 두 가지 평가 모드를 포함하며, 데이터 과학 작업에서 LLM의 효율성을 종합적으로 분석합니다.

- **Technical Details**: CIBench는 LLM-인간 협력 접근 방식을 사용하여 연속적이고 상호작용적인 IPython 세션을 통해 평가 데이터셋을 구성합니다. 평가 모드는 인간의 도움 여부에 따라 LLM의 능력을 평가하며, 다양한 출력에 대한 세밀한 메트릭스를 도입하여 LLM의 코딩 역량을 전반적으로 측정합니다.

- **Performance Highlights**: 24개의 LLM을 대상으로 실험한 결과, 공개 소스 LLM들이 PyTorch와 TensorFlow와 같은 모듈을 활용하는 데 어려움을 겪었으며, 가장 뛰어난 공개 소스 LLM조차 GPT-4에 비해 10.0% 뒤처지는 것으로 나타났습니다. CIBench는 총체적으로 LLM의 코드 인터프리터 사용 능력을 평가할 수 있는 검증된 프레임워크를 제공합니다.



### How and where does CLIP process negation? (https://arxiv.org/abs/2407.10488)
Comments:
          Accepted at the 3rd Workshop on Advances in Language and Vision Research (ALVR 2024)

- **What's New**: 이번 연구는 VALSE 벤치마크를 기반으로 CLIP 모델이 부정(Negation)을 처리하는 방식을 분석합니다. 이 연구는 모델 해석 가능성에 대한 문헌에서 영감을 받아, 모델 내부 프로세스를 설명하려고 합니다.

- **Technical Details**: 연구는 CLIP (Radford et al., 2021) 텍스트 인코더에 대한 심도 있는 분석을 통해 진행되었습니다. 특히 CLIP의 negation 처리를 분석하고, attention heads가 이 작업에서 하는 역할을 분석했습니다. 또한, 부정 처리가 로컬라이즈드(Localised)된 것인지 분산(Distributed)된 것인지, 그리고 데이터셋의 고수준 특징이 모델 성능에 미치는 영향을 탐구합니다.

- **Performance Highlights**: 첫째, 언어 모델 해석 가능성 문헌의 방법(예: 인과 추적(causal tracing))을 멀티모달 모델 및 작업에 적용하는 방법을 설명합니다. 둘째, CLIP이 VALSE 존재 작업에서 부정을 처리하는 방식을 구체적으로 밝혀냈습니다. 셋째, VALSE 데이터셋이 언어적 이해에 있어 참조 벤치마크로서의 본질적인 한계를 강조합니다.



### The Good, The Bad, and The Greedy: Evaluation of LLMs Should Not Ignore Non-Determinism (https://arxiv.org/abs/2407.10457)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 성능 평가에서 비결정론(non-determinism)을 고려한 평가 방식을 제안합니다. 기존의 평가 방식이 단일 출력에만 집중한 반면, 이번 연구는 비결정론의 중요성에 대해 논의하며 다양한 디코딩 방식의 성능 차이를 분석했습니다. 연구 결과, 일반적으로 greedy decoding 방식이 sampling 방식보다 대부분의 작업에서 우수한 성능을 보였으며, alignment 방법이 샘플링 변동성을 줄이는 데 효과적이라는 점을 발견했습니다.

- **Technical Details**: 연구는 greedy decoding과 샘플링 방법 간의 성능 격차를 조사했으며, 다양한 LLM 크기와 alignment(정렬) 방법에 따른 일관성을 확인했습니다. 실험에서는 다양한 벤치마크를 사용했으며, 특히 GSM8K와 HumanEval 같은 수학 및 코드 생성 능력을 평가하는 벤치마크도 포함했습니다. 샘플링 시의 온도와 반복 패널티 변수가 성능에 미치는 영향을 분석했으며, best-of-N sampling 접근법을 통해 작은 LLM이 GPT-4-Turbo와 같은 대형 모델을 능가할 수 있음을 보였습니다.

- **Performance Highlights**: 연구 결과, 대부분의 벤치마크에서 greedy decoding 방식이 더 높은 성능을 보였으며, 특히 MMLU와 MixEval 같은 제한된 출력 공간을 가진 벤치마크에서 일관된 성능을 보였습니다. 또한, 수학적 추론 및 코드 생성 작업에서는 샘플링 변동성이 큰 영향을 미쳤습니다. alignment 방법이 적용된 경우, 샘플링 변동성이 크게 줄어드는 것을 확인했습니다. best-of-N 샘플링 접근법에서는 작은 규모의 LLM이 더 큰 모델보다 우수한 성능을 보일 수 있음을 확인했습니다.



### Don't Throw Away Data: Better Sequence Knowledge Distillation (https://arxiv.org/abs/2407.10456)
- **What's New**: 이 논문은 최소 베이즈 위험(Minimum Bayes Risk, MBR) 디코딩을 좀 더 긴밀하게 통합한 지식 증류(Knowledge Distillation, KD) 방법을 제안합니다. 특히, 단일 선택된 시퀀스 대신 여러 고득점 MBR 번역을 사용하여 교사 모델의 다양한 출력을 포착합니다. 이를 통해 영어-독일어와 영어-일본어 번역 작업에서 강력한 기준 방법보다 일관된 개선을 보였습니다.

- **Technical Details**: MBR-n은 여러 개의 고득점 MBR 시퀀스를 사용해 학생이 고품질 교사 출력을 더 잘 학습할 수 있도록 하는 방법입니다. SeqKD(순차 지식 증류) 방식의 단점을 해결하기 위해 제안되었으며, 블랙 박스 접근 방식으로 교사의 전체 분포를 근사하는 것을 목표로 합니다. MBR 디코딩은 샘플링된 출력들 간의 쌍별 비교를 통해 가장 중심적인 샘플을 선택하는 방식으로, 다양한 모델 불확실성과 레퍼런스 기반 평가 지표를 포함합니다.

- **Performance Highlights**: 실험은 두 개의 번역 작업(영어-독일어와 영어-일본어)과 다양한 크기의 교사 및 학생 모델에서 수행되었습니다. 실험 결과, 여러 시퀀스를 제공하면 단일 MBR 감독보다 학생 모델의 성능이 일관되게 향상됨을 확인했습니다. 특히 데이터 효율성과 출력 다양성 측면에서의 장점을 보여주었으며, 교사와 학생간의 용량 차이 문제(capacity gap curse)도 확인했습니다.



### Enhancing Medication Recommendation with LLM Text Representation (https://arxiv.org/abs/2407.10453)
Comments:
          65 pages, 18 figures

- **What's New**: 기존의 약물 추천 모델은 주로 의료 코드와 같은 구조화된 데이터를 사용하여 예측하지만, 대량의 비구조화 또는 반구조화 데이터는 제대로 활용되지 않았습니다. 이 문제를 해결하기 위해, 우리는 대형 언어 모델(LLM)을 사용하여 텍스트 표현을 향상시키는 방법을 제안합니다.

- **Technical Details**: LLM은 강력한 언어 이해 및 생성 능력을 갖추고 있으며, 복잡한 용어가 포함된 임상 메모와 같은 비구조화된 데이터에서 정보를 추출할 수 있습니다. 이 방법은 우리가 선택한 여러 기존 기본 모델에 적용할 수 있으며, 텍스트와 의료 코드의 결합 표현을 통해 약물 추천 성능을 향상시킬 수 있습니다. 두 개의 다른 데이터셋으로 실험한 결과, LLM 텍스트 표현만으로도 의료 코드 표현만큼의 성능을 보일 수 있음을 확인했습니다.

- **Performance Highlights**: LLM 텍스트 표현은 단독으로도 의료 코드 표현과 비교할 수 있는 능력을 보여주었으며, 전체적으로 이 방법은 다른 모델에도 적용하여 추천 성능을 개선할 수 있는 일반적인 방법입니다.



### Expanding the Scope: Inductive Knowledge Graph Reasoning with Multi-Starting Progressive Propagation (https://arxiv.org/abs/2407.10430)
Comments:
          Accepted in the 23rd International Semantic Web Conference (ISWC 2024)

- **What's New**: 이번 논문에서는 새로운 inductive KG reasoning 모델인 MStar를 소개합니다. MStar는 conditional message passing neural networks (C-MPNNs)을 활용하여 다수의 쿼리 특화 시작 엔티티 (query-specific starting entities)를 선택, 진행성 전파의 범위를 확장합니다. 또한 highway layer를 설계하여 이 시작 엔티티들로 정보를 전파하고, LinkVerify라는 훈련 전략을 도입하여 노이즈가 많은 훈련 샘플의 영향을 완화합니다.

- **Technical Details**: MStar는 기존 C-MPNNs의 문제점인 일관된 메시지 전달 범위 한계를 극복하기 위해 새로운 접근 방식을 도입합니다. 본 모델은 Multi-Starting progressive propagation을 사용하여 시작 엔티티를 다수로 설정하고 이들 사이의 지름길을 만드는 방식으로 작동합니다. 시작 엔티티 선택 모듈 (SES)과 highway layer를 도입하여 효율적인 메시지 전달을 가능하게 했습니다. SES 모듈은 사전 임베딩된 GNN을 사용하여 엔티티를 인코딩하고 다수의 쿼리 의존 시작 엔티티를 선택합니다. highway layer는 ResNet의 skip connection을 기반으로 한 디자인으로, 메시지 전달 경로를 단축시킵니다.

- **Performance Highlights**: 실험 결과, MStar는 기존 최첨단 모델들에 비해 특히 먼 엔티티를 예측하는 데 있어서 우수한 성능을 보였습니다. MStar는 제한된 단계 내에 더 많은 쿼리 관련 먼 엔티티들을 방문하고, 기존 모델들보다 더 풍부한 조건 메시지를 전달할 수 있음을 실험을 통해 검증했습니다.



### By My Eyes: Grounding Multimodal Large Language Models with Sensor Data via Visual Prompting (https://arxiv.org/abs/2407.10385)
Comments:
          21 pages, 16 figures

- **What's New**: 대형 언어 모델(LLMs)이 다양한 도메인에서 뛰어난 성능을 보여주고 있지만, 센서 데이터 시퀀스를 처리할 때 기존의 텍스트 프롬프트(text prompts) 방식은 성능 저하와 높은 토큰 비용을 초래합니다. 이를 해결하기 위해 멀티모달 LLMs(MLLMs)를 활용한 시각적 프롬프트(visual prompting) 접근 방식을 제안합니다. 시각화 생성기를 도입하여 최적의 시각화를 자동으로 생성함으로써 사전 지식 없이도 다양한 감각 과제에 적용할 수 있습니다.

- **Technical Details**: 이 연구에서는 현재 LLMs의 텍스트 프롬프트 방식의 한계를 극복하기 위해 센서 데이터를 시각화된 이미지로 변환하여 MLLMs에 공급하는 방식을 제안합니다. 이를 통해 MLLMs의 시각적 해석 능력을 활용하여 센서 데이터를 분석할 수 있게 합니다. 또한, 공공 라이브러리를 이용하여 자동으로 적절한 시각화를 생성하는 시각화 생성기(visualization generator)를 설계하였습니다. 이 생성기는 각 감각 과제에 맞는 최적의 시각화를 선택합니다.

- **Performance Highlights**: 제안된 방식은 9개의 감각 과제와 4가지 감각 모달리티에서 평균 정확도가 10% 향상되었으며, 텍스트 기반 프롬프트에 비해 토큰비용이 15.8배 감소하였습니다. 이러한 결과는 다양한 감각 과제에서 시각적 프롬프트가 MLLMs의 성능과 비용 효율성을 크게 향상시킴을 보여줍니다.



### Comparing Complex Concepts with Transformers: Matching Patent Claims Against Natural Language Tex (https://arxiv.org/abs/2407.10351)
Comments:
          5th Workshop on Patent Text Mining and Semantic Technologies (PatentSemTech 2024) at ACM SIGIR

- **What's New**: 두 가지 새로운 LLM 기반 접근법을 사용하여 특허 명세서와 같은 다른 텍스트와 특허 청구항을 비교하는 문제가 해결되었습니다. 두 접근법 모두 기존에 발표된 값보다 상당히 나은 성능을 보였습니다.

- **Technical Details**: 특허 청구항의 언어는 특허 명세서나 비특허 텍스트와 다른데, 이는 컴퓨터 기반 자연어 처리(Natural Language Processing, NLP)에서 도전적인 문제였습니다. 이 논문에서는 두 가지 새로운 대형 언어 모델(LLM, Large Language Model) 기반 접근법을 테스트하였습니다.

- **Performance Highlights**: 두 가지 접근법 모두 특허 청구항과 다른 텍스트 사이의 조밀한 정보(dense information)를 다른 어휘로 표현된 더 분산된 정보(distributed information)와 일치시키는 능력에서 이전보다 훨씬 나은 성능을 보여 주었습니다.



### MambaForGCN: Enhancing Long-Range Dependency with State Space Model and Kolmogorov-Arnold Networks for Aspect-Based Sentiment Analysis (https://arxiv.org/abs/2407.10347)
Comments:
          25 pages, 3 figures and 3 tables

- **What's New**: 이 논문에서는 Aspect-based Sentiment Analysis(ABSA)에서 문맥의 길이와 상관없이 의미와 구문 정보를 효과적으로 통합할 수 있는 새로운 접근법인 MambaForGCN을 제안합니다. 이 모델은 구문 기반 Graph Convolutional Network(SynGCN)와 MambaFormer(Mamba-Transformer) 모듈을 결합하여, aspect와 opinion 단어 간의 단거리 및 장거리 종속성을 강화합니다.

- **Technical Details**: MambaForGCN은 입력을 의존 관계와 의미 정보로 인코딩하는 SynGCN과 MambaFormer 모듈을 포함합니다. MambaFormer 모듈의 Multihead Attention(MHA)과 Mamba 블록은 단거리 및 장거리 종속성을 강화하는 채널로 사용됩니다. 또한, SynGCN과 MambaFormer 표현을 결합하기 위해 Kolmogorov-Arnold Networks(KANs) 게이트 융합 방법이 도입되었습니다.

- **Performance Highlights**: 세 가지 벤치마크 데이터셋에 대한 실험 결과, MambaForGCN은 기존 최고 수준(state-of-the-art, SOTA) 모델보다 우수한 성능을 보였습니다.



### Does Burrows' Delta really confirm that Rowling and Galbraith are the same author? (https://arxiv.org/abs/2407.10301)
Comments:
          9 pages, 6 figures

- **What's New**: 이 논문은 J.K. 롤링이 쓴 것으로 밝혀진 소설 'The Cuckoo's Calling'의 저자 판별 문제를 해결하기 위해 stylo 패키지를 사용한 새로운 접근법을 제안합니다. 이 패키지의 기존 데이터셋은 비현대적인 저자들의 다른 장르 텍스트를 포함하고 있어 비판을 받고 있습니다. 이 논문은 연구 질문에 더 적합한 텍스트를 사용하여 해당 방법의 성능을 테스트하려고 합니다.

- **Technical Details**: 디지털 인문학에서 저자 판별을 위한 주된 기법으로 John Burrows가 2002년에 도입한 Delta 방법이 사용됩니다. stylo 패키지는 R 언어를 위한 종합적인 스타일로메트리 분석 도구로, 범용성과 직관적인 그래픽 인터페이스(GUI)를 제공합니다. 패키지에는 5명의 작가로부터 총 26권의 책에서 추출한 가장 빈번한 3000개의 단어의 상대 빈도 테이블이 포함되어 있습니다. 그러나 사용된 글들이 너무 혼성적이고 현대적이지 않다는 비판이 있습니다.

- **Performance Highlights**: 독자적인 텍스트 저자 판별이 'The Cuckoo's Calling'의 저자를 J.K. 롤링으로 식별한 것처럼, stylo 패키지도 동일한 결과를 얻었습니다. 그러나 기존의 distractors(대조 텍스트)가 정확하지 않다는 문제로 인해 Juola의 연구와는 차이가 있습니다. 이 연구는 더 적절한 현대 탐정소설 텍스트를 대조 텍스트로 사용하여 이러한 한계를 극복하려고 합니다.



### Cross-Lingual Multi-Hop Knowledge Editing -- Benchmarks, Analysis and a Simple Contrastive Learning based Approach (https://arxiv.org/abs/2407.10275)
Comments:
          Paper on Cross-Lingual Multi-Hop Knowledge Editing

- **What's New**: 이 논문에서는 'Cross-Lingual Multi-Hop Knowledge Editing' 패러다임을 제안하여 다양한 최신 기술(State-of-the-Art; SoTA) 지식 편집 기법들의 성능을 다국어 환경에서 측정하고 분석합니다. 이를 위해 CROLIN-MQUAKE라는 평행 다국어 벤치마크를 만들었습니다. 이 연구는 영어 중심의 설정과 다국어 설정 간의 성능 차이를 폭넓게 분석해 중요한 격차를 발견했습니다. 이러한 격차를 줄이기 위해, 새로운 체계인 CLEVER-CKE를 제안하였습니다.

- **Technical Details**: CLEVER-CKE는 'retrieve, verify and generate' 지식 편집 프레임워크에 기반을 두고 있으며, 편집된 사실들을 회상하고 이를 언어 모델이 준수하도록 합니다. 이 시스템은 다국어와 세밀한 사실 검색 및 검증 프로세스를 개선하기 위해 언어 인식 및 hard-negative 기반 대비 목표를 개발했습니다. 다양한 언어, 데이터셋 및 대형 언어 모델(LLM)에서 실험한 결과, 기존 방법론보다 최대 30% 성능 향상을 보여주었습니다.

- **Performance Highlights**: CLEVER-CKE는 다양한 LLMs, 8개의 언어와 2개의 데이터셋에서 최대 30%까지 성능 향상을 이루었습니다. 이는 다국어 지식 편집에서 사실 편집을 정확하게 기억하고, 세분화된 문장 이해를 통해 높은 수준의 다국어 검색이 가능케 한 결과입니다.



### psifx -- Psychological and Social Interactions Feature Extraction Packag (https://arxiv.org/abs/2407.10266)
- **What's New**: 이번 발표된 psifx는 플러그 앤 플레이 형식의 다중 모달(feature extraction) 도구로, 최신 머신 러닝 기술을 인간 과학 연구에 활용할 수 있도록 돕고자 합니다. 주된 목표는 데이터를 자동화하고 표준화하여 비싼 비용과 시간을 절감하며 일관성이 없는 인간의 작업을 대체하는 것입니다. 이는 심리학 연구 소프트웨어를 오픈 소스로 개발하고 배포하며, 비전문가 사용자도 쉽게 접근할 수 있도록 제공됩니다.

- **Technical Details**: psifx는 음성 분할(speaker diarization), 자막 텍스트 전사(closed-caption transcription and translation), 그리고 비디오에서 신체, 손, 얼굴 자세 추정(body, hand, and facial pose estimation) 및 시선 추적(gaze tracking) 등의 작업을 위한 다양한 도구를 포함합니다. 해당 프레임워크는 모듈화되어 있으며, 작업 지향적인 접근 방식을 채택해 커뮤니티가 새로운 도구를 쉽게 추가하거나 업데이트할 수 있도록 설계되었습니다.

- **Performance Highlights**: psifx는 비디오와 오디오 데이터를 병렬적으로 처리하며, 이에 따라 대규모 데이터셋을 효율적으로 처리할 수 있습니다. 사용자는 단 몇 개의 명령어로 설정이 가능하며, 무료로 사용할 수 있는 파이썬 패키지와 Docker 이미지도 제공됩니다. 또한, 이 도구는 민감한 데이터를 제3자에게 공유하지 않고도 로컬에서 데이터를 처리할 수 있습니다. 최종 목표는 심리학자들이 최신 기술을 손쉽게 활용할 수 있도록 하는 것입니다.



### Nullpointer at CheckThat! 2024: Identifying Subjectivity from Multilingual Text Sequenc (https://arxiv.org/abs/2407.10252)
- **What's New**: 이 연구는 텍스트 시퀀스(문장 또는 단락)가 주관적인지 객관적인지를 판별하는 이진 분류 작업에 대해 다룹니다. 이 작업은 아랍어, 불가리아어, 영어, 독일어, 이탈리아어, 다국어 카테고리에 걸쳐 진행되었습니다. 우리는 'MarieAngeA13/Sentiment-Analysis-BERT'라는 감정 분석 Transformer 모델을 미세 조정하여 사용했습니다.

- **Technical Details**: 데이터 전처리 과정에서 품사 태깅(POS tagging), 물음표 식별, 주의 마스크(Attention Masks)를 적용했습니다. 비영어 데이터를 영어로 번역하여 데이터셋의 일관성을 유지했습니다. 데이터 불균형 문제를 해결하기 위해 객관적인 데이터에 더 높은 가중치를 부여한 커스텀 분류기를 구현했습니다. 또한 Google Translator API를 사용하여 비영어 데이터를 영어로 번역했습니다.

- **Performance Highlights**: 우리 모델은 다국어 데이터셋에서 Macro F1=0.7121, 독일어에서 Macro F1=0.7908로 우수한 성과를 기록했습니다. 아랍어에서는 두 번째로 높은 점수(Macro F1=0.4908), 불가리아어(Macro F1=0.7169)와 이탈리아어(Macro F1=0.7430)에서는 각각 두 번째와 세 번째로 높은 점수를 받았습니다. 그러나 영어에서는 상대적으로 낮은 성과(Macro F1=0.6893)를 보였습니다.



### GenSco: Can Question Decomposition based Passage Alignment improve Question Answering? (https://arxiv.org/abs/2407.10245)
- **What's New**: 본 연구에서는 다중 단계 질문 답변(MHQA)을 위한 혁신적인 접근 방식인 'GenSco'를 소개합니다. 이 접근 방식은 질문 분해를 기반으로 선택된 passage sequence가 답변 생성에 어떻게 더 나은 결과를 가져오는지를 탐구합니다. GenSco는 두 가지 서로 다른 대형 언어 모델(LLM)을 사용하여, 생성기 LLM은 질문 분해와 최종 답변 생성을 하고, 보조 오픈소스 LLM은 생성기를 안내하는 scorer 역할을 합니다.

- **Technical Details**: GenSco는 다중 단계 질문을 분해하여 생성된 하위 질문에 기반한 passage들 간의 negative log-likelihood를 이용해 passage를 선택하는 방식입니다. 이 방법은 초기 context가 빈 상태에서 시작하여, 생성기 LLM을 사용해 하위 질문을 생성하고, scorer LLM을 통해 passage를 평가하여 가장 높은 점수를 받은 passage를 context에 추가하는 과정을 반복합니다. 마지막으로 누적된 context와 원래 질문을 생성기 LLM에 전달하여 최종 답변을 생성합니다.

- **Performance Highlights**: GenSco는 2WikiMultiHop와 MuSiQue 데이터셋에서 각각 Exact Match 점수에서 15.1점과 5.9점의 절대적인 이득을 보였습니다. 이는 기존 최상 성능 기법 대비 뛰어난 성능을 보여줍니다. 또한 GenSco는 passage 검색 작업에서 높은 정밀도를 달성하여 LLM 응답의 환상을 효과적으로 줄입니다.



### BiasAlert: A Plug-and-play Tool for Social Bias Detection in LLMs (https://arxiv.org/abs/2407.10241)
- **What's New**: 이번 연구에서는 대규모 언어 모델(Large Language Models, LLMs)의 개방형 텍스트 생성(open-text generation) 시나리오에서 발생하는 사회적 편견을 감지하는 도구인 BiasAlert를 소개합니다. BiasAlert는 외부 인간 지식과 LLM의 내재된 추론 능력을 통합하여 사회적 편견을 신뢰성 있게 감지합니다. 이 도구는 텍스트 생성 및 질문 응답과 같은 다양한 시나리오에서 편견 감지 및 완화에 유용하게 사용될 수 있습니다.

- **Technical Details**: BiasAlert는 플러그 앤 플레이 방식의 도구로, LLM의 텍스트 생성 내용을 입력으로 받아 편견을 분석합니다. BiasAlert는 사용자 입력을 받아 관련된 사회적 편견을 데이터베이스에서 검색한 후, 감지 모듈이 공정성을 판단하고 편견이 포함된 콘텐츠를 식별합니다. 이 도구를 구현하기 위해 인간 주석이 달린 사회적 편견 데이터베이스를 구축하고, 추론 및 언어 이해 능력을 강화한 명령-따르기 데이터 세트로 BiasAlert를 훈련했습니다.

- **Performance Highlights**: BiasAlert는 Redditbias 및 Crows-pairs 데이터셋을 활용한 실험에서 기존의 편견 감지 도구 및 최신 언어 모델(GPT-4)을 능가하는 성능을 보였습니다. 추가 실험을 통해 BiasAlert의 감지 능력, 적응성 및 실용성을 입증했으며, 텍스트 생성 작업, 질문 응답 작업 및 LLM 배포 시 편견 완화에 대한 사례 연구도 제공했습니다.



### Key-Point-Driven Mathematical Reasoning Distillation of Large Language Mod (https://arxiv.org/abs/2407.10167)
- **What's New**: 대형 언어 모델(LLMs)의 복잡한 수학적 추론 능력을 작은 언어 모델(SLMs)로 증류하는 새로운 방법인 Key-Point-Driven Mathematical Reasoning Distillation (KPDD)를 제안합니다. 이 방법은 문제 해결 과정을 Core Question Extraction, Problem-Solving Information Extraction, Step-by-Step Solution 세 단계로 나누어 진행합니다. KPDD는 Chain-of-Thought(COT)와 Program-of-Thought(POT) 두 가지 방법으로 나뉘며, 여러 수학적 추론 데이터셋에서 성능을 크게 향상시킵니다.

- **Technical Details**: KPDD는 수학적 추론 성능을 향상시키기 위해 다음과 같은 세 가지 단계를 포함합니다: 1) Core Question Extraction: 원래 문제에서 핵심 질문을 추출, 2) Problem-Solving Information Extraction: 문제 해결에 필요한 관련 정보를 추출, 3) Step-by-Step Solution: 추출된 핵심 정보를 사용해 문제를 단계별로 해결. 이를 통해 KPDD-CoT(Chain-of-Thought)와 KPDD-PoT(Program-of-Thought) 두 가지 형태로 구분됩니다.

- **Performance Highlights**: KPDD는 FlanT5 모델에 대해 광범위한 실험을 통해 성능을 평가하였으며, KPDD-CoT 방법으로 FlanT5-Large 모델은 평균 정확도 24.71%를 달성했고, KPDD-PoT 방법으로는 평균 정확도 63.83%를 기록하였습니다. 이는 SOTA(State-of-the-Art) 성능을 보이며, 특히 이해 오류를 효과적으로 줄여 SLMs의 수학적 추론 성능을 크게 향상시켰습니다.



### Look Within, Why LLMs Hallucinate: A Causal Perspectiv (https://arxiv.org/abs/2407.10153)
Comments:
          15 pages, 7 figures

- **What's New**: 이번 논문은 대규모 언어 모델(LLMs)의 환각(hallucination) 문제를 심층 조사하며, LLMs의 주목(attention) 메커니즘이 이러한 환각에 어떻게 영향을 미치는지를 탐구합니다. 저자들은 기존 연구들이 주로 데이터 품질에 집중한 반면, 주목 메커니즘의 역할에 대해서는 거의 연구되지 않았음을 지적합니다. 이에 저자들은 LLMs의 주목 층에 개입(intervene)하여 그 구조와 크기를 유지하면서 환각을 감소시키는 새로운 방법을 제안합니다.

- **Technical Details**: 이 논문은 구조적 인과 모델(Structural Causal Model, SCM)을 사용하여 LLMs의 환각 생성 과정을 설명하고, 인과 추론의 관점에서 주목 층에 개입합니다. 구체적으로 여러 인기 있는 오픈 소스 LLMs의 주목 층을 비활성화(disable)하고 원래 모델과 비교하여 환각 정도를 평가합니다. 또한, 주목 층이 어떤 역할을 하는지 분석하기 위해, 전방 또는 후방 주목 층을 비활성화할 경우 환각 문제가 완화될 수 있음을 발견했습니다.

- **Performance Highlights**: 논문 결과에 따르면, 특정 주목 층을 비활성화하면 환각 현상이 줄어드는 것을 발견했습니다. 예를 들어, LLaMA 2-7B-Chat 모델에서 3333번째 주목 층을 비활성화하면 TruthfulQA 데이터셋에서 약 2% 성능 향상이 있었습니다. 또한 Gemma-2B-instruct 모델에서 13131313번째 주목 층을 비활성화한 후 HaluEval 데이터셋에서 4% 성능 향상이 나타났습니다. 이러한 결과는 LLMs의 주목 층이 각기 다른 환각 내용을 대표할 수 있음을 시사합니다.



### Mitigating Translationese in Low-resource Languages: The Storyboard Approach (https://arxiv.org/abs/2407.10152)
Comments:
          published at LREC-COLING 2024

- **What's New**: 이번 연구에서는 시각적 자극인 스토리보드를 활용하여 더 유창하고 자연스러운 문장을 생성하는 새로운 데이터 수집 방법을 제안합니다. 이 방법은 번역 기반의 기존 방법 대신 사용됩니다.

- **Technical Details**: 기존의 번역 기반 방법론이 가지는 번역체(translationalese) 효과를 최소화하기 위해, 스토리보드(storyboards)를 사용해 시각적 장면을 묘사하는 방식을 채택했습니다. 참가자는 원본 텍스트에 노출되지 않고 시각적 콘텐츠만을 기준으로 장면을 설명합니다.

- **Performance Highlights**: 인간 평가자와 정량적 지표를 통해 평가 결과, 전통적 텍스트 번역 방법이 정확도 면에서는 우수했으나, 제안된 스토리보드 기반 방법이 유창성 면에서는 더 나은 결과를 보여주었습니다. 본 연구는 4개의 저자원 아프리카 언어(스와힐리어, 요루바어, 하우사어, 이비비오어)에서 데이터 수집과 평가를 진행했습니다.



### Textless Dependency Parsing by Labeled Sequence Prediction (https://arxiv.org/abs/2407.10118)
Comments:
          Accepted to Interspeech 2024

- **What's New**: 새로운 연구는 텍스트가 없는(textless) 방법으로 의존 구문 분석(dependency parsing)을 제안했습니다. 기존의 자동 음성 인식(ASR) 시스템을 이용한 텍스트처리 모델과 달리, 이 방법은 음성 신호에서 직접 의존 트리를 예측합니다.

- **Technical Details**: 텍스트 없이(textless) 방법은 음성 신호에서 직접 의존 트리를 예측하며, 트리를 라벨이 붙은 시퀀스로 나타냅니다. 이 방법은 음성 신호를 단어 단위로 분절하지 않고 전체 음성 신호에서 의존 관계를 추출합니다. 서로 다른 두 언어(프랑스어와 영어)를 대상으로 실험이 진행되었습니다.

- **Performance Highlights**: 실험 결과, 전통적인 ASR 기반 처리 방법이 전반적인 구문 분석 정확도에서 우수함을 보였지만, 중요한 음향 특징을 포함한 경우 텍스트 없는(textless) 방법이 더 우수한 성능을 보였습니다. 이로 인해 단어 수준의 표현과 문장 수준의 운율(prosody)을 결합하는 것이 구문 분석 성능 향상에 중요하다는 결론을 내렸습니다.



### TokenSHAP: Interpreting Large Language Models with Monte Carlo Shapley Value Estimation (https://arxiv.org/abs/2407.10114)
- **What's New**: 대규모 언어 모델(LLMs)의 해석 가능성을 높이기 위한 TokenSHAP 방법이 소개되었습니다. 이 방법은 게임 이론의 Shapley 값을 자연어 처리(NLP)에 적용하여 입력 프롬프트의 각 토큰이나 부분 문자열이 모델의 응답에 미치는 중요성을 정량적으로 분석합니다.

- **Technical Details**: TokenSHAP는 Monte Carlo sampling을 통해 Shapley 값을 추정하여 입력 프롬프트의 각 토큰에 중요도를 부여합니다. 이는 LLM의 응답에 대한 각 부분의 기여도를 이해하는 엄밀한 프레임워크를 제공하며, 모델의 투명성과 신뢰성을 높여줍니다.

- **Performance Highlights**: 다양한 프롬프트와 LLM 아키텍처에서 TokenSHAP의 효과가 입증되었습니다. 인간의 판단과 일치하며 모델의 행동에 대한 신뢰성을 높이고 일관성을 유지함으로써 기존의 기준보다 꾸준히 향상된 성능을 보여주었습니다. TokenSHAP는 특히 토큰 간의 미세한 상호작용을 포착하여 LLM 행동을 이해하는 데 귀중한 인사이트를 제공합니다.



### Enhancing Emotion Prediction in News Headlines: Insights from ChatGPT and Seq2Seq Models for Free-Text Generation (https://arxiv.org/abs/2407.10091)
Comments:
          published at LREC-COLING 2024

- **What's New**: 이번 연구에서는 뉴스 헤드라인이 유발하는 감정을 예측하기 위해 전통적인 접근법에서 벗어나, 사람들이 뉴스 헤드라인을 읽고 느끼는 감정을 자유롭게 서술한 텍스트를 활용하는 방법을 제안합니다. 이를 통해 뉴스 헤드라인만으로는 부족한 감정적 맥락을 더 잘 활용하여 감정 분류 모델의 성능을 향상시킬 수 있습니다.

- **Technical Details**: 이번 연구에서는 BU-NEmo+ 데이터셋을 사용하여 뉴스 헤드라인에 대한 감정 설명 텍스트를 생성하기 위한 sequence-to-sequence transformer 모델을 훈련했습니다. 또한 사전 훈련된 대형 언어 모델인 GPT-4와 T5 모델을 활용하여 감정 설명 생성을 시도했습니다. 생성된 감정 설명 텍스트를 감정 분류 초기 단계의 입력으로 사용하여 기존의 뉴스 헤드라인 데이터만을 사용하는 경우에 비해 성능을 평가했습니다.

- **Performance Highlights**: McNemar의 유의성 검정 결과, GPT를 활용하여 생성된 감정 설명 텍스트를 포함한 방법이 단순히 뉴스 헤드라인만을 사용하는 방법에 비해 유의미한 성능 향상(P-value < 0.05)을 보였습니다. 이는 자유 텍스트 형태의 감정 설명을 감정 예측 작업에 사용함으로써 얻을 수 있는 가치가 크다는 것을 강조합니다.



### Rapid Biomedical Research Classification: The Pandemic PACT Advanced Categorisation Engin (https://arxiv.org/abs/2407.10086)
- **What's New**: 팬데믹 PACT 고급 분류 엔진(PPACE)과 그에 따른 데이터셋을 소개합니다. PPACE는 WHO 연구 우선순위에 따라 연구 초록을 자동으로 분류하기 위해 개발된 모델입니다. 이 모델은 인간이 주석을 달아 분류한 데이터를 기반으로 한 학습을 통해 연구 트렌드를 모니터링하고, 글로벌 보건 준비 및 대응의 격차를 식별하는 데 필요합니다.

- **Technical Details**: PPACE는 인간이 주석을 달아 정리한 데이터와 생성형 AI를 이용해 각 분류에 대한 `rationales'(근거)을 생성합니다. 그런 다음 이 데이터로 소형 모델을 미세 조정하여 분류 작업을 자동화합니다. 이 모델은 여러 병원체에 대한 연구 자금 및 임상 증거를 추적한 팬데믹 PACT 프로젝트의 일환으로 개발되었습니다. 데이터셋과 모델 가중치는 공개되어, 유사 모델 개발이 가능합니다.

- **Performance Highlights**: PPACE는 기존 베이스라인 모델들에 비해 성능이 크게 향상되었습니다. 이는 생물의학 문서 분류 및 글로벌 보건 우선순위와의 정렬에서 큰 진전을 의미합니다.



### Multi-Granularity Semantic Revision for Large Language Model Distillation (https://arxiv.org/abs/2407.10068)
- **What's New**: 이번 연구는 대형 언어 모델(Large Language Models, LLM)을 압축하기 위한 새로운 지식 증류(Knowledge Distillation) 방법을 제안합니다. 이 방법은 시퀀스 수정 및 재생성(Sequence Correction and Re-Generation, SCRG) 전략, 분포 적응 클리핑 Kullback-Leibler(DAC-KL) 손실 함수, 그리고 스팬 레벨의 사전 지식을 활용하여 LLM 증류 과정에서 발생할 수 있는 여러 문제를 해결합니다.

- **Technical Details**: 연구에서는 세 가지 주요 레벨에서 개입하여 지식 증류 과정을 개선합니다. 첫째, 시퀀스 레벨에서 SCRG 전략을 통해 교사 모델과 학생 모델 간의 의미적 인지 차이를 계산하고 오류 토큰을 수정한 후 시퀀스를 다시 생성합니다. 둘째, 토큰 레벨에서 DAC-KL 손실 함수를 통해 교사의 출력 중에서 의미가 밀집된 부분을 학습 가능한 서브 네트워크로 적응적으로 추출합니다. 마지막으로, 스팬 레벨에서는 시퀀스의 스팬 사전 지식을 활용하여 교사와 학생 모델의 확률 상관관계를 일치시켜, 의미 정보의 일관된 전송을 보장합니다.

- **Performance Highlights**: 다양한 모델 계열에서 매개변수가 0.1B에서 13B에 이르는 실험을 통해 제안된 방법의 우수성을 입증했습니다. 특히 LLAMA2, OpenLLAMA2, OPT, GPT2 시리즈 등을 포함한 모델에서 기존의 지식 증류 방법보다 뛰어난 성능을 보였습니다.



### Learning to Refuse: Towards Mitigating Privacy Risks in LLMs (https://arxiv.org/abs/2407.10058)
- **What's New**: 새로운 연구에서는 대규모 언어 모델(LLM)들이 개인 정보를 기억하여 개인정보 침해의 위험을 초래할 수 있다는 문제를 해결하기 위해, 개인의 데이터를 보호하는 방법을 제안하고 있습니다. 연구진은 Wikipedia에서 2,492명의 인물 데이터를 수집하여 QA(pair)를 작성한 eturn 데이터셋을 소개하며, 이름 인식 학습 프레임워크(NAUF)를 통해 특정 개인의 정보 보호를 가능하게 하였습니다.

- **Technical Details**: 이 연구는 LLM들이 개인 정보를 기억하지 않도록 '머신 언러닝(Machine Unlearning, MU)' 방법론을 적용하였습니다. eturn 데이터셋은 실제 인물의 정보와 20개의 QA(pair)로 구성되어 있습니다. NAUF 프레임워크는 '이름 인식 거부 응답(Name-Aware Refusal Answer)'과 '대조 데이터 증강(Contrastive Data Augmentation)'으로 구성되어, 특정 개인의 데이터를 보호하면서 다른 사람들의 질문에 대한 응답 능력을 유지합니다.

- **Performance Highlights**: 광범위한 실험 결과, NAUF는 평균 언러닝 점수에서 기존 최고 방법보다 5.65점 높은 성과를 보여주어, 목표로 하는 개인의 데이터를 효과적으로 보호하면서 모델의 일반 성능도 유지할 수 있음을 증명했습니다.



### AutoGRAMS: Autonomous Graphical Agent Modeling Softwar (https://arxiv.org/abs/2407.10049)
- **What's New**: AutoGRAMS 프레임워크가 소개되었습니다. 이 프레임워크는 다중 단계의 상호작용을 언어 모델과 함께 프로그래밍 할 수 있게 합니다. AutoGRAMS는 AI 에이전트를 그래프 형태로 표현하는데, 각 노드는 언어 모델링 명령 또는 전통적인 코드를 실행할 수 있습니다. 또한, 그래프의 전환은 언어 모델링 결정이나 전통적인 분기 로직에 의해 관리될 수 있습니다.

- **Technical Details**: AutoGRAMS 프레임워크는 AI 에이전트와 챗봇의 복잡한 상호작용을 구성할 수 있도록 설계되었습니다. 이 프레임워크는 다양한 기능을 제공하며, 노드와 전환 규칙을 정의하여 그래프 형태의 절차를 설정합니다. 이 노드들 각각은 특정 동작과 전환 규칙을 포함할 수 있으며, 이 그래프를 통해 상호작용이 진행됩니다. 또한, AutoGRAMS는 자율적인 프로그램인 'autogram'을 도입하여 자기 참조 기능을 강조합니다. 이 프로그램은 스스로를 수정할 수 있는 구조를 갖추고 있습니다.

- **Performance Highlights**: AutoGRAMS를 통해 사용자는 언어 모델의 예측을 활용하여 복잡한 대화 흐름을 설계하고, 명시적인 분기점을 정의하며, 메모리에 변수를 활용할 수 있습니다. 예를 들어, 사용자의 회신에 따라 에이전트가 답변을 생성하거나 다음 노드를 결정할 수 있습니다. 이러한 기능들은 프레임워크의 직관적인 그래픽 인터페이스를 통해 쉽게 시각화할 수 있으며, 스프레드시트 소프트웨어를 사용하여 복잡한 에이전트를 디자인하는 것도 가능합니다.



### Document-level Clinical Entity and Relation Extraction via Knowledge Base-Guided Generation (https://arxiv.org/abs/2407.10021)
Comments:
          Accepted at Association for Computational Linguistics BioNLP 2024

- **What's New**: 이 연구에서는 GPT 모델을 사용하여 임상 엔티티와 관계 추출에 UMLS(Ulified Medical Language System) 지식베이스를 활용하는 프레임워크를 소개합니다. UMLS는 텍스트와 관련된 의학 개념을 선택하고, 이를 프롬프트로 결합하여 언어 모델이 엔티티를 추출하도록 안내합니다. 이 접근법은 RAG(Retrieval Augmented Generation) 기법보다 더 효율적이며, UMLS 개념을 통합함으로써 추출 결과를 향상시킵니다.

- **Technical Details**: 제안된 프레임워크는 GPT 모델의 문맥 캡처 능력과 UMLS의 지식 캡처 능력을 결합합니다. MetaMap을 사용하여 임상 텍스트에서 UMLS 개념을 매핑하고, 그런 다음 필터링하여 '유기 화학물', '항생제', '약리학적 물질'과 같은 의약품 관련 개념만 포함시킵니다. 그 후, 필터링된 개념을 포함한 몇 가지 샷의 프롬프트를 사용하여 GPT 모델을 안내합니다.

- **Performance Highlights**: 제안된 방법은 UMLS 개념을 포함하지 않은 일반 GPT 모델과 비교하여 임상 엔티티 및 관계 추출 성능이 향상되었습니다. 특히, RAG 모델보다도 뛰어난 성능을 보였습니다.



### Causality extraction from medical text using Large Language Models (LLMs) (https://arxiv.org/abs/2407.10020)
- **What's New**: 이 연구는 자연어 모델, 특히 대형 언어 모델(Large Language Models, LLMs)을 사용하여 임상 실습 가이드라인(Clinical Practice Guidelines, CPGs)에서 인과 관계를 추출하는 가능성을 탐구합니다. 특히 임신성 당뇨병 관련 CPGs에서의 인과 관계 추출 결과를 처음으로 제시합니다. 이 연구는 BioBERT, DistilBERT 및 일반 BERT와 같은 다양한 BERT 변형들과 GPT-4, LLAMA2와 같은 LLM들을 사용한 실험 결과를 보고합니다.

- **Technical Details**: BERT(Devlin et al., 2018)는 입력 요소 간의 연결을 기반으로 출력의 각 부분 간에 가중치를 동적으로 조정하여 인과 관계 추출(Causality Extraction) 등의 여러 자연어 처리 작업에 탁월한 성능을 보여줘 왔습니다. 최근에는 GPT-4(OpenAI, 2023)와 같은 LLMs가 등장하여, 광범위한 데이터로 사전 학습을 한 후 인류의 원칙과 정책 준수를 보장하기 위해 인간과 AI의 피드백을 통해 강화 학습을 진행합니다. LLAMA2(Touvron et al., 2023)도 최근 주목받는 모델로, 2조 개의 토큰으로 학습되었으며 정보 추출 작업에서 널리 사용됩니다.

- **Performance Highlights**: 실험 결과, BioBERT가 다른 모델들보다 뛰어난 성능을 보였으며, 평균 F1 점수는 0.72로 기록되었습니다. GPT-4와 LLAMA2도 유사한 성능을 보였으나 일관성은 낮았습니다. BERT 변형 모델들은 fine-tuning이 쉽고 일관된 성능을 보여 여전히 선호될 수 있음을 확인했습니다. LLAMA2는 몇 가지 데이터에 대해 예측을 생성하지 못했지만 예측한 부분에서는 평균 F1 점수 76%를 기록하며 가능성을 보여주었습니다.



### Minimizing PLM-Based Few-Shot Intent Detectors (https://arxiv.org/abs/2407.09943)
- **What's New**: 이 연구는 소량의 레이블링된 데이터로 사전 학습된 언어 모델(PLM)을 기반으로 효율적인 인텐트 탐지기를 학습하는 방법을 제시합니다. 이를 통해 자원 제약 환경, 특히 모바일 디바이스에서의 배포 문제를 해결하고자 합니다. 연구진은 대형 언어 모델(LLMs)을 이용한 데이터 증가, 첨단 모델 압축 기법인 지식 증류 및 V-Prune이라는 새로운 어휘 정리 메커니즘을 사용하여 모델 크기를 최소화하는 방법을 탐구했습니다.

- **Technical Details**: 이번 연구에서는 LLM을 이용한 데이터 증대(data augmentation)를 통해 데이터 부족 문제를 해결하고, CoFi라는 최신 Transformer 압축 방법을 사용하여 모델을 압축했습니다. CoFi는 교사 모델의 파라미터를 점진적으로 삭제하여 학생 모델을 도출하며 뛰어난 성능을 제공합니다. 또한, V-Prune이라는 어휘 정리 기법을 도입하여 인텐트 탐지에 필수적인 토큰만을 유지하고, 누락된 토큰을 유사한 토큰으로 대체하는 방식으로 성능 저하 문제를 해결했습니다.

- **Performance Highlights**: 5개의 샷(few-shot) 시나리오에서 제안된 방법은 BERT 모델과 어휘 크기를 각각 20배 및 30배 축소하면서도 거의 동일한 성능을 유지했습니다. 연구진은 이 방법을 통해 4개의 실제 벤치마크에서 거의 동일한 성능을 유지하면서 모델 메모리 사용량을 21배 압축하는 데 성공했습니다.



### WojoodNER 2024: The Second Arabic Named Entity Recognition Shared Task (https://arxiv.org/abs/2407.09936)
- **What's New**: WojoodNER-2024는 두 번째 아랍어 개체명 인식(NER) 공유 태스크로, 세분화된 아랍어 NER에 중점을 두고 있습니다. 새로운 아랍어 세분화 NER 데이터셋인 wojoodfine이 제공되었으며, 개체의 하위 유형으로 주석이 달려 있습니다. WojoodNER-2024는 세 가지 서브태스크를 포함하는데, (i) Flat Fine-Grained NER, (ii) Nested Fine-Grained NER, (iii) 이스라엘-가자 전쟁에 대한 Open-Track NER이 그것입니다.

- **Technical Details**: Flat Fine-Grained NER와 Nested Fine-Grained NER 서브태스크에서는 PERS, ORG, GPE 등의 클래스에 대한 멘션을 인식하고 이를 사전 정의된 클래스에 분류합니다. 하지만 이번 대회에서는 더 세분화된 WojoodFINE 데이터셋을 사용하여 기존의 GPE를 COUNTRY, STATE-OR-PROVINCE, TOWN 등 7개의 하위 유형으로 세분화하였습니다. 데이터셋은 약 550k 토큰과 51개의 개체 유형 및 하위 유형을 포함합니다. 참가 팀들은 전통적인 머신러닝부터 딥러닝, 트랜스포머 기반 기술 등을 적용하여 다양한 접근 방식을 실험하였습니다. 특히, SinaTools를 통해 API로 접근할 수도 있습니다.

- **Performance Highlights**: 43개 팀이 대회에 등록하였으며, Flat Fine-Grained 서브태스크에서는 5개 팀, Nested Fine-Grained 서브태스크에서는 2개 팀, Open-Track NER 서브태스크에서는 1개 팀이 참여하였습니다. 우승 팀은 Flat Fine-Grained 서브태스크에서 F-1 점수 91%, Nested Fine-Grained 서브태스크에서 92%를 기록하였으며, Open-Track NER 서브태스크에서는 73.7%의 F-1 점수를 기록하였습니다.



### Cohesive Conversations: Enhancing Authenticity in Multi-Agent Simulated Dialogues (https://arxiv.org/abs/2407.09897)
Comments:
          Accepted to COLM 2024

- **What's New**: 이 논문은 대규모 언어 모델(Large Language Models, LLMs)을 활용한 시뮬레이션에서 다중 에이전트 대화의 품질을 조사합니다. Park et al. (2023)의 사례 연구에서 25명의 에이전트가 하루 종일 복잡한 행동과 상호작용을 보이는 시뮬레이션을 통해 이를 연구하였습니다. 주요 문제점으로 반복, 불일치, 환각 등 문제를 발견했으며, 이를 해결하기 위해 Screening, Diagnosis, and Regeneration (SDR) 프레임워크를 제안하였습니다.

- **Technical Details**: SDR 프레임워크는 발화 오류를 즉시 감지하고 교정하는 방법으로, 세 단계(Sreening, Diagnosis, Regeneration)를 통해 작동합니다. Screening 단계에서는 잠재적인 문제를 식별하고 과거 대화에서 관련 증거를 수집합니다. Diagnosis 단계에서는 LLM이 증거를 분석해 발화의 진위성을 평가하고 피드백을 제공합니다. Regeneration 단계에서는 문제가 있는 발화를 수정합니다. NLI (Natural Language Inference)-Graph 모듈을 도입해 논리적 불일치 및 환각을 감지하고 바른 정보를 유지합니다.

- **Performance Highlights**: GPT-4 평가와 인간 평가를 통해 SDR 프레임워크의 효과가 입증되었습니다. 다중 에이전트 시뮬레이션에서 일관성을 높이고, 잘못된 정보 전달을 줄이는 데 유의미한 개선을 보였습니다. 자동 평가 메트릭 역시 대화의 다양성이 향상되었음을 확인했습니다.



### Synergistic Multi-Agent Framework with Trajectory Learning for Knowledge-Intensive Tasks (https://arxiv.org/abs/2407.09893)
- **What's New**: SMART라는 혁신적인 다중 에이전트 프레임워크를 도입하여 외부 지식을 활용하여 대규모 언어 모델(LLMs)에서 생성된 응답의 해석 가능성과 사실적 일관성을 향상시킨다. SMART는 복잡한 지식 집약적 작업을 처리하기 위해 각기 다른 하위 경로 작업을 수행하는 4개의 전문 에이전트로 구성된다.

- **Technical Details**: SMART 프레임워크는 다음과 같은 4개의 주요 에이전트로 구성된다: 의도 재구성기(Intent Reconstructor, IR), 지식 검색기(Knowledge Retriever, KR), 사실 위치추적기(Fact Locator, FL), 응답 생성기(Response Generator, RG). 이 에이전트들은 복잡한 지식 집약적 작업을 처리하기 위해 각기 다른 하위 경로 작업을 수행한다. SMART는 개별 에이전트의 세부 수행을 보장하면서 에이전트 간의 시너지를 유지하는 '장단 경로 학습(Long- and Short-Trajectory Learning)'이라는 다중 에이전트 공동 학습 패러다임을 제안한다.

- **Performance Highlights**: SMART는 5개의 작업에서 광범위한 실험을 통해 기존에 널리 채택된 방법들보다 우수한 성능을 입증했다. 새롭게 제안된 프레임워크는 지식 해석 가능성과 사실적 일관성을 크게 향상시킨다.



### FarFetched: Entity-centric Reasoning and Claim Validation for the Greek Language based on Textually Represented Environments (https://arxiv.org/abs/2407.09888)
Comments:
          DeepLo NAACL 2022

- **What's New**: FarFetched는 다양한 온라인 뉴스 출처에서 집계된 증거를 기반으로 텍스트 주장을 자동으로 검증하는 프레임워크입니다. 특히 자원이 부족한 언어를 대상으로 하는 자동화된 주장 검증의 문제점을 해결하고, 그리스어를 사례로 사용하여 관련 모델들을 교육하고 평가하였습니다.

- **Technical Details**: 이 프레임워크는 실체 중심 추론(entity-centric reasoning) 방식을 사용하여 사건, 행동, 또는 진술 사이의 잠재적 연결을 실체 언급을 통해 드러내고 이를 그래프 데이터베이스에 표현합니다. 실체 연결(entity linking)과 의미적 유사성(semantic similarity)을 이용해 다양한 출처에서 정보를 수집하고 결합하여 사용자의 주장과 관련된 증거를 생성합니다. 그런 다음 텍스트 함의 인식(textual entailment recognition)을 활용하여 이러한 주장이 신뢰할 수 있는지 양적으로 판단합니다. 그리스어를 포함한 자원이 부족한 언어에서의 STS와 NLI 모델을 교육하고 평가하였습니다.

- **Performance Highlights**: FarFetched는 새로운 뉴스 기사가 지속적으로 업데이트됨에 따라 검증된 주장에 대한 판단이 동적으로 변할 수 있습니다.



### sPhinX: Sample Efficient Multilingual Instruction Fine-Tuning Through N-shot Guided Prompting (https://arxiv.org/abs/2407.09879)
Comments:
          20 pages, 12 tables, 5 figures

- **What's New**: 이 연구에서는 영어 외의 다른 언어에서 성능 격차를 해결하기 위해 다국어 합성 지시 조정 데이터셋인 sPhinX를 도입합니다. 이 데이터셋은 영어로 된 지시 응답 쌍을 선택적으로 50개 언어로 번역하여 만들어졌습니다. 연구진은 sPhinX를 사용하여 최첨단 모델인 Phi-3-small과 Mistral-7B를 미세 조정하고, 다양한 다국어 벤치마크를 통해 평가하여 평균적으로 4.2%pt 및 5%pt의 성능 향상을 확인했습니다.

- **Technical Details**: sPhinX는 Orca 지시 조정 데이터셋을 기반으로 GPT-4를 사용하여 번역하여 생성된 51개 언어의 180만개의 지시 응답 쌍으로 구성됩니다. 연구진은 LAnguage-Specific N-shot Guided Instruction fine-tuning (LANG) 전략을 개발하여 다국어 성능을 높였으며, 이 접근법은 다양한 언어에서의 샘플 효율성과 다양성을 확보합니다.

- **Performance Highlights**: Phi-3-small과 Mistral-7B 모델이 sPhinX로 미세 조정되었을 때, 기존 기준 모델보다 각각 4.2%pt와 5%pt 더 나은 성과를 보였습니다. 또한 N-shot 예제를 각 미세 조정 샘플에 포함시켰을 때, 각각 3%pt와 10%pt의 추가 성능 향상을 달성했습니다. 무엇보다도, sPhinX는 다른 다국어 지시 조정 데이터셋보다 더 나은 성과를 보이며 샘플 효율적이고 다양성이 뛰어납니다.



### Towards Systematic Monolingual NLP Surveys: GenA of Greek NLP (https://arxiv.org/abs/2407.09861)
Comments:
          68 pages

- **What's New**: 본 연구는 단일 언어 자연어 처리(NLP) 서베이의 결여를 보완하기 위해 체계적이고 포괄적인 단일 언어 NLP 서베이를 생성하는 방법을 제안합니다. 이 연구는 그리스어 NLP에 대한 체계적인 문헌 검토를 수행하여 2012년에서 2022년 사이의 연구 현황과 과제를 종합적으로 다룹니다. 이를 통해 단일 언어 NLP 서베이가 다언어주의(NLP에서의 multilingualism)를 위한 토대를 제공할 수 있음을 보여줍니다.

- **Technical Details**: 이 방법은 구조화된 검색 프로토콜을 특징으로 하며, 이를 통해 출판물을 선택하고 NLP 작업의 분류 체계를 통해 조직할 수 있습니다. 또한, 접근 가능성과 데이터셋의 주석(annotation)에 따라 언어 자원(Language Resources, LRs)을 분류하여 공개적으로 이용 가능하고 기계적으로 활용 가능한 LRs를 강조합니다.

- **Performance Highlights**: 연구는 데이터 유출(data leakage)과 오염(contamination)과 같은 일반적인 함정을 피하고 NLP 작업별로 언어 지원을 평가하는 데 도움이 된다고 설명합니다. 그리스어 NLP의 진행 상황을 논의하고 접근 가능성과 사용 가능성에 따라 분류된 그리스어 언어 자원을 기술합니다. 이 연구는 단일 언어 NLP 서베이의 이점을 입증하며, NLP 연구가 상대적으로 미흡한 많은 언어들에 유사한 접근을 적용할 수 있음을 시사합니다.



### Building pre-train LLM Dataset for the INDIC Languages: a case study on Hind (https://arxiv.org/abs/2407.09855)
Comments:
          Accepted as a book chapter in the book Title "APPLIED SPEECH AND TEXT PROCESSING FOR LOW RESOURCE LANGUAGES"

- **What's New**: 이번 논문에서는 인도 언어 중 하나인 힌디어(Indic language Hindi)를 위한 대규모 사전 훈련 데이터셋을 제안합니다. 이러한 데이터셋은 여러 분야와 주요 힌디어 방언들을 포함하여 수집되었으며, 총 12억 8천만 개 이상의 힌디어 토큰을 포함하고 있습니다. 데이터 수집, 전처리 및 사용 가능성에 대한 파이프라인을 설명하며, 다른 인도 언어 및 저자원 언어(Indic and low-resource languages)로도 확장 가능하도록 설계되었습니다.

- **Technical Details**: 힌디어를 위한 대규모 사전 훈련 데이터셋을 만들기 위해 본 연구에서는 여러 분야와 방언을 대상으로 데이터 수집을 진행하였습니다. 데이터 수집, 전처리 과정뿐만 아니라 LLM(large language model)을 사전 훈련하기 위한 접근 방식을 상세히 설명합니다. 다양한 모델 아키텍처(model architecture), 훈련 방법(training approaches), 및 언어 고유의 평가 전략(assessment strategies) 등을 살펴봅니다.

- **Performance Highlights**: 힌디어 LLM을 위한 사전 훈련 데이터셋은 힌디어 자연 언어 처리(Natural Language Processing; NLP) 작업에서 기존 모델들보다 더 나은 성과를 제공할 가능성이 높습니다. 이 데이터셋은 감정 분석(sentiment analysis), 기계 번역(machine translation), 텍스트 분류(text categorization), 명명된 개체 인식(named entity identification) 등 여러 NLP 작업에서 성능을 크게 향상시킬 수 있습니다. 이러한 모델들은 힌디어 사용자들의 필요에 맞춘 신뢰할 수 있는 NLP 시스템을 만드는 데 중요한 역할을 수행하며, 연구와 산업 응용에 기여할 것입니다.



### Text-Based Detection of On-Hold Scripts in Contact Center Calls (https://arxiv.org/abs/2407.09849)
Comments:
          9 pages, 3 figures, 4 tables

- **What's New**: 이번 연구는 고객 서비스 통화에서 특별한 대기 스크립트를 사용하여 긍정적인 상호작용을 유지하는 방법을 제시합니다. 새로운 자연어 처리 모델은 자동 음성 인식 기술로 전사된 고객 서비스 통화에서 대기 구절(on-hold phrases)을 탐지합니다.

- **Technical Details**: 이 작업은 대화에서 대기 스크립트를 찾는 문제를 세 가지 상호 배타적인 클래스(classes)로 분류된 멀티클래스 텍스트 분류 문제로 정의하였습니다. 해당 클래스는 고객을 대기 상태로 만드는 스크립트, 고객에게 돌아오는 스크립트, 대기 스크립트와 관련 없는 구절로 구성됩니다. 우리는 내부 데이터셋을 수집하여 각 통화에서 각 대화 회전을 레이블링하였습니다. 그리고 RuBERT를 다양한 초매개변수(hyperparameter) 세트를 탐색하면서 해당 데이터셋으로 파인튜닝(fine-tuning)했습니다.

- **Performance Highlights**: 고성능 모델을 달성하였으며, 개발된 모델은 에이전트가 미리 정의된 대기 스크립트를 따르는지 여부를 확인하는 방법을 제공하여 에이전트 모니터링에 도움이 됩니다.



### Investigating Low-Rank Training in Transformer Language Models: Efficiency and Scaling Analysis (https://arxiv.org/abs/2407.09835)
Comments:
          Accepted by ICML 2024 Next Generation of Sequence Modeling Architectures Workshop. arXiv admin note: substantial text overlap with arXiv:2406.16450

- **What's New**: 최신 대형 언어 모델(LLMs)은 높은 계산 비용을 요구하며, 이로 인해 매개변수 수와 비용을 줄이면서 성능을 유지하려는 연구가 진행되고 있습니다. 본 연구에서는 Transformer 기반 LLM에서 계산 비용이 높은 feedforward 네트워크(FFNs)에 저격자(low-rank) 매개변수를 적용했습니다. 실험 결과, 저격자 FFNs이 더 가파른 스케일링 곡선을 보여 큰 가능성을 보였습니다.

- **Technical Details**: 저격자 매개변수화를 통해 FFN의 매개변수를 32% 수준으로 줄여 1.3B 모델의 학습 속도를 1.35배 향상시켰습니다. 이러한 구조적 FFN은 전통적인 Transformer보다 더 가파른 손실 스케일링 곡선을 보여주었습니다. 주로 작업에서 FFN 모듈만 저격자 매개변수화를 적용했으며, 초기화는 스펙트럼 초기화를 따랐습니다. 모델은 Rotary Embedding과 GeLU 활성화 함수를 사용하는 기본 Transformer 아키텍처로 설계되었습니다.

- **Performance Highlights**: 저격자 매개변수화를 통해 FFN 매개변수가 63% 및 32%로 줄어든 상태에서 각각 1.4배 및 2.6배의 속도 향상을 나타냈습니다. 또한, Transformer-xl에서 훈련 시간을 15% 감소시키면서 약 0.4 PPL 증가에 그쳤고, 모델 전체적으로는 1.35배 빠른 속도로 약 1.0 PPL 증가라는 결과를 보였습니다.



### NativQA: Multilingual Culturally-Aligned Natural Query for LLMs (https://arxiv.org/abs/2407.09823)
Comments:
          LLMs, Native, Multilingual, Language Diversity, Contextual Understanding, Minority Languages, Culturally Informed, Foundation Models, Large Language Models

- **What's New**: 이번 연구에서는 지역과 문화에 맞춘 'NativQA' 프레임워크를 제안하여 네이티브 언어로 QA(질문 답변) 데이터를 생성하고, 이를 통해 대규모 언어 모델(LLM) 평가 및 튜닝에 활용할 수 있도록 했습니다. 이를 입증하기 위해 7개의 다양한 언어로 약 72K QA 데이터를 포함한 'MultiNativQA' 멀티링구얼 QA 데이터셋을 개발했습니다.

- **Technical Details**: NativQA 프레임워크는 세 가지 주요 모듈로 구성되어 있습니다: 1) 쿼리 수집(Query Collection), 2) QA 수집(QA Collection), 3) QA 검증(QA Validation). 먼저 네이티브 스피커들로부터 주제에 따라 오픈엔디드 쿼리를 모아 이를 LLM을 사용해 다양하게 생성된 쿼리로 확장합니다. 확장된 쿼리들은 검색 엔진(예: 구글)을 통해 관련 답변을 찾아 최종 QA 페어를 마련합니다.

- **Performance Highlights**: MultiNativQA 데이터셋을 오픈 및 클로즈드 소스 LLM과 벤치마크한 결과, 이 데이터셋은 언어 모델들이 다양한 언어와 주제에서 얼마나 잘 작동하는지 평가할 수 있는 튼튼한 기준을 제공합니다. 그 결과, 널리 쓰이는 QA, NLP, 그리고 스피치 데이터셋들과는 달리 지역 및 문화적 특수성을 잘 반영한 평가가 가능해졌습니다.



### AraFinNLP 2024: The First Arabic Financial NLP Shared Task (https://arxiv.org/abs/2407.09818)
- **What's New**: AraFinNLP는 아랍권 금융 시장의 확장과 함께 발생하는 NLP 도구의 필요성에 대응하기 위해 금융 분야의 아랍어 NLP 도구 개발을 지원하는 새로운 공유 과제를 제안합니다. 이 과제는 두 가지 하위 과제인 다이얼렉트별 의도 탐지(Multi-dialect Intent Detection)와 다이얼렉트 간 번역 및 의도 보존(Cross-dialect Translation and Intent Preservation)을 포함합니다.

- **Technical Details**: AraFinNLP는 ArBanking77 데이터셋을 사용하며, 약 39,000개의 모던 표준 아랍어(MSA)와 4가지 다이얼렉트로 구성된 병렬 쿼리를 포함합니다. 각 쿼리는 은행 업무와 관련된 77개의 의도(intent) 중 하나 또는 그 이상의 라벨이 부착되어 있습니다. 이 과제는 머신 트랜스레이션, 챗봇(banking chat-bots) 등의 NLP 응용 분야에서 강력한 파이낸셜 아랍어 NLP 개발을 촉진합니다.

- **Performance Highlights**: Subtask 1에 참가한 총 11개의 팀 중 우승 팀은 F1 스코어 0.8773을 달성했습니다. Subtask 2에 참가한 유일한 팀은 1.667 BLEU 스코어를 기록했습니다. 총 45개의 팀이 과제에 등록했으며 11개의 팀이 최종 테스트 단계에 참가했습니다.



### MaskMoE: Boosting Token-Level Learning via Routing Mask in Mixture-of-Experts (https://arxiv.org/abs/2407.09816)
Comments:
          Work in progress

- **What's New**: 최근 논문에서는 Mixture-of-Experts (MoEs) 모델의 단점들을 해결하기 위해 MaskMoE라는 새로운 방법을 제안했습니다. 이 방법은 라우팅 마스킹 기법을 채택하여 토큰별 학습을 개선합니다. 결과적으로 MaskMoE는 이전의 주류 MoEs 모델들보다 뛰어난 성능을 보이며, perplexity (PPL)와 다운스트림 태스크에서 탁월한 성과를 보입니다.

- **Technical Details**: MaskMoE는 토큰의 빈도에 따라 서로 다른 수의 전문가를 할당하는 라우팅 마스크 기법을 도입합니다. 이는 각 토큰에 대해 레이어 마스크 벡터를 생성하여 빈도가 낮은 토큰은 하나의 전문가에게만 라우팅되고, 빈도가 높은 토큰은 다수의 전문가에게 라우팅될 수 있도록 합니다. 이를 통해 빈도가 낮은 토큰의 집중적인 학습을 가능하게 하고, 빈도가 높은 토큰의 표현 다양성을 유지합니다. 이 방법은 기존의 동적 라우팅이 초래하는 라우팅 변동 문제를 해소하며, 고정된 라우팅 기법의 부족한 표현 다양성을 보완합니다.

- **Performance Highlights**: 실험 결과 MaskMoE는 이전의 Mixture-of-Experts 모델들에 비해 우수한 성능을 보였습니다. 특히, perplexity 측정 및 다양한 벤치마크 데이터셋에 대한 다운스트림 태스크에서 더 나은 결과를 기록했습니다. MaskMoE는 토큰 빈도에 따른 전문가 할당을 통해 과적합 문제를 해결하고, 표현의 다양성을 유지하면서도 충분한 훈련을 가능하게 했습니다.



### LLM-Collaboration on Automatic Science Journalism for the General Audienc (https://arxiv.org/abs/2407.09756)
Comments:
          Under review

- **What's New**: 새로운 연구에서는 과학 저널리즘의 접근성을 높이기 위해 세 가지 LLMs (Large Language Models)를 이용한 프레임워크를 제안합니다. 이 프레임워크는 저널리스트, 일반 대중 독자, 그리고 편집자 역할을 하는 LLMs를 통해 실제 글쓰기-읽기-피드백-수정의 워크플로우를 모방하여 과학 기사를 작성합니다. 실험 결과, 두 개의 7B LLMs와 하나의 1.8B LLM을 활용한 이 방법이 기존의 방법들보다 더 접근성 높은 기사를 생성한다는 것을 보여줍니다.

- **Technical Details**: 이 프레임워크는 다섯 가지 주요 절차로 구성되어 있습니다: (1) 기자 역할을 하는 LLM이 기사를 작성, (2) 소규모 LLM인 독자가 기사를 읽고 피드백 제공, (3) 더 작은 LLM인 편집자가 독자의 피드백을 바탕으로 제안 생성, (4) 기자 LLM이 이 제안을 반영하여 기사를 수정, (5) 이 과정을 반복하여 기사를 지속적으로 개선. 각 LLM은 특정 역할을 맡아 자동화된 협력 과정을 수행합니다.

- **Performance Highlights**: 제안된 방법은 자동 메트릭과 인간 평가를 통해 평가되었습니다. 읽기 쉬움, 정보 전달, 진위성, 흥미도 등의 측면에서 다른 방법들보다 더 높은 평가를 받았습니다. 특히 읽기 쉬움에서 최고 성과를 기록했으며, 다른 측정에서도 경쟁력 있는 성과를 보였습니다. 에디터 LLM, 리더 LLM, 혹은 둘 다 제거하여 수행한 평가 및 트렌드 분석 결과도 함께 포함되어 있습니다.



### Multi-Token Joint Speculative Decoding for Accelerating Large Language Model Inferenc (https://arxiv.org/abs/2407.09722)
- **What's New**: 트랜스포머 기반 대규모 언어 모델(LLMs)의 추론 비용을 절감하기 위해 새로운 디코딩 알고리즘인 Multi-Token Joint Speculative Decoding (MJSD)를 제안했습니다. 기존의 speculative decoding보다 향상된 perplexity와 효율성을 제공합니다.

- **Technical Details**: MJSD는 두 가지 주요 접근 방식을 사용합니다. 먼저, 작은 모델을 사용하여 대규모 모델의 공동 분포를 근사하고, 그런 다음 검증 단계를 통해 근사의 정확성을 보장합니다. 두 번째로, Beam Decoding을 사용해 작은 모델의 공동 분포로부터 시퀀스 생성을 가속화합니다. 이를 통해 Multi-Token Joint Greedy Decoding (MJGD)보다 실용적인 계산 비용 내에서 더 나은 perplexity를 달성할 수 있습니다.

- **Performance Highlights**: MJSD는 텍스트 생성 작업에서 기존의 greedy decoding과 비교하여 속도가 2.21배 빨라졌고, 에너지 소비는 2.84배 감소했습니다. 또한, vanilla speculative decoding과 비교해서도 속도는 1.49배, 에너지 소비는 1.62배 각각 개선되었습니다.



### What an Elegant Bridge: Multilingual LLMs are Biased Similarly in Different Languages (https://arxiv.org/abs/2407.09704)
- **What's New**: 이 논문은 심리언어학 연구에 영감을 받아, 대형 언어 모델(LLMs)을 통해 문법적 성별의 편향을 조사하고 있습니다. 특히 Boroditsky(2003)의 기본 실험을 확장하여 다국어 LLMs를 이용해 문법적 성별이 언어 지각에 미치는 영향을 평가합니다. 이를 위해, 다양한 언어에서 명사를 형용사로 설명하도록 모델을 유도하고, 형용사의 성별 예측을 위해 이진 분류기를 훈련시켰습니다.

- **Technical Details**: 연구팀은 LLama-2와 Mistral 같은 오픈소스 LLMs를 사용해 문법적 성별이 있는 10개 언어에서 형용사를 이용하여 명사를 설명하도록 모델에 지시합니다. 데이터베이스에서 각 언어의 성별이 있는 명사를 추출하고, 형용사를 빈도에 따라 목록화한 후, 이진 분류기를 훈련시켜 형용사가 명사의 문법적 성별을 예측할 수 있는지를 평가합니다.

- **Performance Highlights**: 주요 발견은 간단한 분류기가 형용사를 사용하여 명사의 성별을 예측할 수 있고, 이러한 분류기가 언어 간에 안정적으로 전이된다는 점입니다. 이는 LLMs이 언어에 관계없이 비슷한 편향을 나타낸다는 것을 시사합니다.



### Large Language Models for Integrating Social Determinant of Health Data: A Case Study on Heart Failure 30-Day Readmission Prediction (https://arxiv.org/abs/2407.09688)
Comments:
          36 pages including references and appendix. This is a work in progress

- **What's New**: 이 논문에서는 대형 언어 모델(Large Language Models, LLMs)을 사용하여 사회적 건강 결정 요인(Social Determinants of Health, SDOH) 데이터를 자동으로 통합하고, 이러한 SDOH 특성이 임상 예측에 얼마나 유용한지 평가하였습니다. 연구진은 700개 이상의 변수를 다섯 가지 SDOH 범주로 수작업으로 라벨링하고, 이를 통해 9개의 오픈 소스 LLM들이 이 작업을 얼마나 잘 수행하는지 벤치마크했습니다.

- **Technical Details**: 800개 이상의 공공 데이터 출처의 변수를 하나의 SDOH 범주로 라벨링하였습니다. 그런 다음, LLM들이 이 분류 작업을 얼마나 잘 수행하는지 평가하였습니다. 마지막으로 39,000명의 심부전(heart failure, HF) 환자에 대한 30일 내 재입원 예측 모델을 훈련하고, 표준 임상 변수와 비교하여 SDOH 변수가 예측 성능에 미치는 영향을 비교했습니다. 추가적으로, few-shot LLM 프롬프팅이 LLM 주석 성능에 미치는 영향을 조사하고, 메타데이터 미삭제 연구를 통해 어떤 정보가 LLM이 변수를 정확하게 주석 달기 위해 도움이 되는지 평가했습니다.

- **Performance Highlights**: 몇몇 오픈 소스 LLM들은 특정 튜닝 없이 zero-shot 프롬프팅(프롬프팅 없이)의 조건에서도 SDOH 변수를 효과적으로 정확하게 주석을 달 수 있음을 발견했습니다. 결정적으로, 표준 임상 특징들과 결합했을 때, LLM 주석된 'Neighborhood and Built Environment' 하위 집합의 SDOH 변수는 HF 환자의 30일 재입원을 예측하는데 가장 뛰어난 성능을 보였습니다.



### Bridging the Gap Between Information Seeking and Product Search Systems: Q&A Recommendation for E-commerc (https://arxiv.org/abs/2407.09653)
- **What's New**: 새로운 연구에서는 쇼핑 미션 중인 소비자가 제품 검색과 정보 탐색 시스템을 어떻게 번갈아 사용하는지를 분석하고, 이를 통합한 Q&A 추천 시스템을 제안합니다. 이 시스템은 Q&A 쌍을 제공하여 사용자가 구매 결정을 내리는 데 도움을 주며, 기존의 불편한 전환 과정을 최소화합니다.

- **Technical Details**: 연구팀은 Q&A 추천 시스템의 요구사항, 질문과 답변의 특성, 생성 방법 및 추천 작업의 최적화 문제를 다루고 있습니다. LLMs (Large Language Models)을 사용하여 Q&A 쌍을 생성하고, 이를 자동완성, 검색 결과 페이지, 제품 상세 페이지 등 다양한 쇼핑 단계에서 제안합니다.

- **Performance Highlights**: 이 시스템은 사용자가 쇼핑 여정 중 각 단계에서 필요로 하는 정보를 적시에 제공함으로써 특히 탐색, 비교, 최종 고려 단계에서 큰 효율성을 보입니다. 적절한 질문과 답변을 통해 사용자의 필요를 충족시키고, 구매 결정을 더욱 쉽게 도와줍니다.



### How Chinese are Chinese Language Models? The Puzzling Lack of Language Policy in China's LLMs (https://arxiv.org/abs/2407.09652)
Comments:
          Wen-Yi and Jo contributed equally to this work

- **What's New**: 최근 중국의 언어 모델 개발자들은 다언어 모델(multilingual models)을 통해 다양한 정치 및 비즈니스적인 고려사항을 탐색하고 있다. 논문은 중국이 다민족 사회를 다루는 언어 정책 변화와 이를 현재의 언어 기술에 미치는 영향에 대해 연구한다. 이 연구는 중국에서 개발된 6개의 오픈소스 다언어 대규모 언어 모델(LLM)을 18개의 언어에 대해 평가하며, 중국의 AI 언어 정책의 불확실성에 대해 탐구한다.

- **Technical Details**: 중국 회사들이 사전 훈련한 오픈소스 LLM 6개(Qwen, Yi, DeepSeek, InternLM2, XVERSE, Baichuan)를 18개의 다양한 언어에 대해 테스트하고, 국제적인 모델(Llama, Mistral)과 비교했다. 모델의 기술 보고서 또한 분석하여 데이터 수집 과정에서 영어와 중국어를 제외한 다른 언어에 대한 고려 부족을 발견했다. 두 개의 데이터셋(FLORES+와 Belebele)을 사용하여 언어 모델의 성능을 평가했으며, 다양한 언어에서의 토큰 예측 능력을 측정했다.

- **Performance Highlights**: 중국에서 개발된 LLM은 다양한 언어에 대해 국제 모델과 비교해도 거의 구분이 가지 않을 만큼 성능이 비슷했다. 저자들은 FLORES+ 및 Belebele 평가 기준에서 중국 내 민족 소수 언어를 어드레스하고 있는 모델들에도 성능 차이가 없음을 발견했고, 단일언어 모델에서 볼 수 있는 성능 한계점도 유사했다. 중국 정부의 AI 정책에도 소수 언어에 대한 명확한 방침이 없다.



### Diversifying the Expert Knowledge for Task-Agnostic Pruning in Sparse Mixture-of-Experts (https://arxiv.org/abs/2407.09590)
Comments:
          13pages, 6 figures

- **What's New**: 이번 연구에서는 Mixture-of-Experts (MoE) 모델의 메모리 사용량 증가 문제를 해결하기 위한 새로운 방법을 제안합니다. 저자들은 각 MoE 레이어의 유사한 전문가들을 그룹화하고, 중복된 전문가를 제거하는 방법을 통해 모델의 파라미터 효율성을 향상시켰습니다. 이를 통해 Mixtral-8x7B와 Mixtral-8x22B와 같은 최첨단 MoE 모델에서 기존 방법들보다 더 나은 성능을 확인했습니다.

- **Technical Details**: MoE 아키텍처는 여러 병렬 FFN(Fully Connected Feed-Forward Networks)을 사용하여 네트워크를 확장합니다. 그러나 이로 인해 모델이 수행할 때 일부 전문가(Experts)만 활성화되더라도 메모리 소모량이 크게 증가합니다. 본 연구는 일부 전문가들이 유사한 지식을 인코딩하고 있다는 점을 실험을 통해 밝혔습니다. 이에 따라 저자들은 두 단계로 이루어진 전문가 그룹화 및 가지치기(pruning) 방법을 고안했습니다. 먼저, 특징 공간(feature space)에서 유사한 전문가들을 그룹화한 다음, 각 그룹 내에서 가중치 공간(weight space)을 통해 전문가들을 병합하여 다양한 지식을 유지합니다.

- **Performance Highlights**: 제안된 방법론을 통해 두 개의 최첨단 MoE 모델인 Mixtral-8x7B와 Mixtral-8x22B에서 중복된 전문가를 효과적으로 제거할 수 있음을 확인했습니다. 평가 결과, 제안된 방법이 자연어 처리(Natural Language Processing) 작업에서 다른 모델 가지치기 방법들보다 더 나은 성능을 발휘함을 보여주었습니다. 이러한 결과를 바탕으로 코드와 가지치기된 MoE 모델들이 공개될 예정입니다.



### A Transformer-Based Multi-Stream Approach for Isolated Iranian Sign Language Recognition (https://arxiv.org/abs/2407.09544)
Comments:
          17 pages, 10 figures

- **What's New**: 이번 연구는 이란 수어 (Iranian Sign Language) 인식 시스템을 개발하여 청각 장애인 커뮤니티의 의사소통 격차를 줄이는 것을 목표로 하고 있습니다. 최신 딥러닝 기술인 트랜스포머 (transformers)를 활용하여 이란 수어 단어 101가지의 인식을 시도합니다.

- **Technical Details**: 데이터셋은 주로 대학과 같은 학술 환경에서 자주 사용되는 101개의 이란 수어 단어를 포함하고 있습니다. 네트워크는 얼리 퓨전 (early fusion) 및 레이트 퓨전 (late fusion) 트랜스포머 인코더 기반 네트워크로 구성되어 있으며, 유전 알고리즘 (genetic algorithm)을 통해 최적화되었습니다. 네트워크 훈련을 위해 선택된 특징은 손과 입술의 키 포인트, 그리고 손 사이의 거리와 각도입니다. 또한 단어의 임베딩 벡터 (embedding vectors)가 멀티 태스크 학습 (multi-task learning)의 일환으로 사용되어 보다 매끄럽고 효율적인 훈련이 가능하도록 했습니다.

- **Performance Highlights**: 이 모델은 단어 데이터셋에서 생성된 문장을 윈도잉 기법 (windowing technique)으로 번역하여 테스트되었으며, 실험 결과 테스트 데이터에서 90.2%의 정확도로 평가되었습니다. 또한, 이 모델을 활용한 실시간 피드백을 제공하는 수어 학습 소프트웨어가 소개되었으며, 사용자 설문 조사를 통해 이 소프트웨어의 학습 효과성과 효율성을 조사하였습니다.



### MATE: Meet At The Embedding -- Connecting Images with Long Texts (https://arxiv.org/abs/2407.09541)
- **What's New**: 최근 Vision Language Models(VLMs)의 발전은 시각적 데이터와 텍스트 데이터를 정렬하는 능력을 크게 향상시켰지만, 주로 짧은 설명 캡션에 중점을 두어 복잡한 텍스트 상호작용을 다루기 어렵다는 한계가 있었습니다. 본 논문에서는 Meet At The Embedding(MATE)을 도입했습니다. MATE는 VLM과 Large Language Models(LLMs)의 능력을 결합하여 추가적인 이미지-긴 텍스트 쌍이 필요 없이 이 문제를 해결합니다. 특히, VLM의 텍스트 인코더를 사전 학습된 LLM 기반 인코더로 대체하고, 이미지 임베딩과 LLM 임베딩 간의 간극을 연결하기 위해 다단계로 훈련된 프로젝션 모듈을 사용합니다.

- **Technical Details**: MATE는 VLM과 LLM의 임베딩을 정렬하여 이미지와 긴 텍스트 간의 연결을 촉진합니다. VLM의 텍스트 인코더를 LLM 기반 인코더로 교체하고, 이미지 인코더를 그대로 유지한 채 프로젝션 모듈을 통해 임베딩을 정렬합니다. 이 모듈은 텍스트 간 임베딩을 먼저 정렬한 후, 이미지 임베딩을 LLM 임베딩과 정렬하도록 적응됩니다. 이를 통해 직접적인 이미지-긴 텍스트 쌍 없이도 효과적인 연결을 달성합니다.

- **Performance Highlights**: 실험 결과, MATE는 이미지와 긴 텍스트 간의 연결을 효과적으로 수행하고 다양한 의미 관계를 발견했습니다. 이는 직관적인 검색 결과를 제공하며, 복잡한 텍스트와 시각 정보를 통합하는 능력을 향상시킵니다. 또한, 새로운 크로스 모달 검색 벤치마크를 통해 우수한 성능을 입증했습니다.



### Integrating Large Language Models with Graph-based Reasoning for Conversational Question Answering (https://arxiv.org/abs/2407.09506)
- **What's New**: 이번 연구는 문맥 이해와 이종 소스(텍스트, 지식 그래프, 테이블, 인포박스)에서 증거를 수집해 추론하는 대화형 질문 답변 작업에 중점을 둡니다. 문장의 문맥과 검색된 증거를 통합하는 그래프 구조 표현을 활용하며, 대형 언어 모델(LLMs)의 추론 및 텍스트 생성 능력을 이용합니다.

- **Technical Details**: 우리의 방법은 질문과 그 문맥에 대한 정보를 동적으로 생성된 그래프로 표현합니다. 그래프 임베딩(graph embeddings)은 토큰 임베딩 레이어(token embedding layers)를 우회하여 LLM에 직접 삽입되며, 크로스 엔트로피(cross-entropy)를 최소화하여 학습됩니다. 또한 메모리 모듈(memory module)을 통해 과거 증거를 추적 및 업데이트하여 대화가 진행됨에 따라 그래프의 구조에 영향을 미칩니다.

- **Performance Highlights**: ConvMix 벤치마크에서의 실험 결과, 그래프 임베딩이 LLM의 추론 능력을 향상시키는 반면, 메모리 모듈은 소음 및 검색 오류에 대한 견고함을 제공합니다.



### Image captioning in different languages (https://arxiv.org/abs/2407.09495)
- **What's New**: 이 논문은 비영어권 이미지 캡션 데이터셋(non-English image captioning datasets)에 대해 수집한 목록을 제공한다. 2024년 5월 기준, 다양한 언어를 포괄하는 데이터셋은 23개의 언어로 구성되어 있다. Crossmodal-3600 데이터셋(Thapliyal et al., 2022, 36개 언어)을 추가해도 여전히 다양한 언어에 비해 매우 적다. 이 논문은 Vision & Language (비전 & 언어) 분야에 대한 여러 개방형 질문으로 마무리된다.

- **Technical Details**: 이 논문에서 설명한 비영어권 이미지 캡션 데이터셋 목록은 Google Scholar 알림을 8년간 모니터링하여 'MS COCO'와 'Flickr30K' 키워드를 사용해 수집된 논문들을 수동으로 큐레이션한 결과이다. 목록에는 데이터셋의 언어, 이미지 소스 (기존 데이터셋 혹은 새로운 데이터셋 이름), 캡션이 영어에서 번역되었는지 독립적으로 수집되었는지, 데이터셋이 소개된 논문의 참조 등이 포함되어 있다.

- **Performance Highlights**: 현재 제공된 비영어권 이미지 캡션 데이터셋이 다양하지 않다는 점이 가장 두드러진다. 많은 데이터셋이 영어 캡션을 (자동) 번역하여 생성되었는데, 이는 독립적으로 캡션을 수집하는 것보다 비용이 저렴하기 때문이다. 하지만 이러한 번역된 데이터셋은 비서구적 언어로 작성되었더라도 여전히 서구적 관점을 담고 있다는 한계가 있다.



### Spider2-V: How Far Are Multimodal Agents From Automating Data Science and Engineering Workflows? (https://arxiv.org/abs/2407.10956)
Comments:
          34 pages, 14 figures, 10 tables

- **What's New**: 이 논문에서는 데이터 과학과 엔지니어링 워크플로우를 자동화할 수 있는 첫 번째 멀티모달 에이전트 벤치마크인 Spider2-V를 소개합니다. 이 벤치마크는 494개의 실제 작업을 포함하며, 20개의 엔터프라이즈 수준의 전문 응용 프로그램을 포함하고 있습니다.

- **Technical Details**: Spider2-V는 데이터 관련 작업을 수행하는 멀티모달 에이전트의 능력을 평가하기 위해 코드 작성 및 GUI 관리 작업을 수행합니다. 이를 위해 작업 설정의 자동 구성과 각 작업에 대한 평가 메트릭스를 신중하게 설계했습니다. 또한, 멀티모달 에이전트를 종합적인 엔터프라이즈 데이터 소프트웨어 시스템 문서와 함께 보완했습니다.

- **Performance Highlights**: 현재의 최첨단 LLM/VLM 기반 에이전트는 전체 데이터 워크플로우를 안정적으로 자동화하지 못하며(성공률 14.0%), 세부적인 지식이 필요한 GUI 작업에서도 낮은 성과를 보이고 있습니다(성공률 16.2%). 클라우드 기반 워크스페이스를 사용하는 작업에서는 성과가 더욱 저조합니다(성공률 10.6%).



### Benchmarking Vision Language Models for Cultural Understanding (https://arxiv.org/abs/2407.10920)
- **What's New**: 이번 연구에서는 CulturalVQA라는 새로운 비주얼 퀘스천-앤서링(Visual Question Answering) 벤치마크를 소개합니다. 이 벤치마크는 VLM(Vision Language Models)의 지리적으로 다양한 문화 이해도를 평가하는 데 목적이 있습니다. CulturalVQA는 11개국의 문화적 요소를 반영한 2,378개의 이미지 질문 쌍을 큐레이션하여, 각 질문당 1-5개의 답변을 포함합니다.

- **Technical Details**: CulturalVQA 벤치마크는 비주얼 퀘스천-앤서링(VQA) 형식으로 구축되었습니다. 이 데이터셋은 주로 의류, 음식, 음료, 의식 및 전통과 같은 다양한 문화적 개념을 탐구하는 질문들로 구성되어 있습니다. CulturalVQA는 기존의 언어 전용 CANDLE 데이터셋을 확장하여 시각적 개념을 포함한 이미지와 질문-답변 쌍을 추가적으로 수집했습니다. 이를 통해 문화적 공감각 지식을 평가할 수 있는 시스템을 제공합니다.

- **Performance Highlights**: 최신 VLM들을 CulturalVQA로 평가한 결과, 지역별로 문화 이해도에 뚜렷한 성능 격차가 나타났습니다. 예를 들어, 북미 문화 이해도에서는 높은 성능을 보였으나, 아프리카에서는 성능이 크게 저하되는 것으로 나타났습니다. 또한, 의류, 의식 및 전통에 관한 질문에서는 비교적 높은 성능을 보였으나 음식 및 음료 관련 질문에서는 낮은 성과를 보였습니다. 이를 통해 VLM들이 문화적 이해에서 부족한 부분을 식별할 수 있습니다.



### Leveraging LLM-Respondents for Item Evaluation: a Psychometric Analysis (https://arxiv.org/abs/2407.10899)
- **What's New**: 이 연구는 GPT-3.5, GPT-4, Llama 2, Llama 3, Gemini-Pro, Cohere Command R Plus 와 같은 여섯 가지 대형 언어 모델(Large Language Models, LLMs)을 활용하여 대학교 학부생들과 유사한 응답을 생성할 수 있는지 탐구합니다. 이러한 접근법은 전통적인 인간 응답 수집의 필요성을 줄이고, 시간과 비용을 절감할 수 있는 가능성을 제시하고 있습니다.

- **Technical Details**: 이 연구에서는 OpenStax College Algebra 2.2 과의 20 문제를 사용하여 총 150개의 응답을 LLM에서 생성했습니다. 이러한 응답은 미국 학부생의 응답과 비교하여 심리측정적 특성을 분석하였습니다. Item Response Theory (IRT)를 활용해 항목의 심리측정적 특성을 비교하였고, 다양한 조합의 모델들을 이용하여 인간 응답자의 능력 분포를 최대한 유사하게 재현하려고 했습니다.

- **Performance Highlights**: 실험 결과, 몇몇 LLM은 대학 수준의 대수학 문제에서 실제 대학생들보다 높은 또는 비슷한 능력을 보여주었습니다. 예를 들어, GPT-3.5의 항목 파라미터는 인간의 것과 매우 높은 상관관계를 보였습니다(>0.8). 또한, LLM-으응답으로 보강된 데이터는 인간만을 사용한 데이터보다 스피어만 상관관계가 높은 것으로 나타났습니다 (인간만 0.89에서 보강된 데이터 0.93).

- **Significance**: 이 연구는 LLM이 교육 평가에 중요한 도구가 될 수 있는 잠재력을 제시합니다. LLM 응답자로부터 얻은 항목 파라미터는 인간으로부터 얻은 것과 매우 흡사하여, 다양한 교육적 맥락에서 신속하고 비용 효율적인 평가 방법으로 사용할 수 있는 가능성을 보여줍니다.



### LLM Circuit Analyses Are Consistent Across Training and Sca (https://arxiv.org/abs/2407.10827)
- **What's New**: 이번 연구는 70 million에서 2.8 billion 개의 파라미터를 가지는 decoder-only LLMs를 대상으로, 300 billion 개의 토큰에 따른 학습 중 모델 메커니즘(기능 회로, circuits)의 등장과 발전을 추적하였습니다. 기존 연구들은 주로 학습이 끝난 시점의 모델만을 다루었지만, 이번 연구는 학습 기간 동안의 변화를 분석함으로써 실제로 배포된 모델에서도 적용 가능한 통찰을 제공합니다.

- **Technical Details**: 본 연구는 Pythia suite의 모델을 사용하여, edge attribution patching with integrated gradients (EAP-IG) 방법을 통해 각 돌연변이가 발생한 상황에서도 모델의 행동이 유지되는 메커니즘을 찾아냈습니다. Circuit은 모델이 특정 과업을 수행할 때 사용하는 최소한의 계산 서브그래프를 의미합니다. 우리는 모델의 학습 도중에 기능 구성 요소가 언제, 어떻게 등장하고 발전하는지를 추적하였습니다.

- **Performance Highlights**: 각 모델 크기에서 기능 구성 요소의 등장 시점은 비슷하였으며, 이는 모델의 크기와 관계없이 유사한 토큰 수에 대한 학습이 중요한 역할을 한다는 것을 시사합니다. 또한 개별 구성 요소가 변하더라도 전체 알고리즘은 일관되게 유지되며, 이는 큰 모델에서도 작은 모델에서 발견된 회로 분석이 유용하다는 것을 의미합니다. 이러한 결과는 학습 종료 시점이 아닌 중간 단계에서도 circuit 분석이 모델의 메커니즘 이해에 있어 중요한 도구가 될 수 있음을 보여줍니다.



### Qwen2-Audio Technical Repor (https://arxiv.org/abs/2407.10759)
Comments:
this https URL. Checkpoints, codes and scripts will be opensoursed soon

- **What's New**: Qwen2-Audio는 다양한 오디오 신호 입력을 수용하고 오디오 분석 또는 텍스트 응답을 수행할 수 있는 대규모 오디오-언어 모델입니다. 이 모델은 자연 언어 프롬프트를 사용하여 사전 훈련 프로세스를 간소화하였으며 데이터 볼륨을 확장했습니다. 두 가지의 독립적인 오디오 상호작용 모드를 구현하여, 음성 대화 모드에서는 사용자가 텍스트 입력 없이 음성 상호작용을 할 수 있고, 오디오 분석 모드에서는 오디오 및 텍스트 지시 사항으로 분석할 수 있습니다. 이는 음성 명령을 지능적으로 이해하고 적절하게 응답할 수 있습니다.

- **Technical Details**: Qwen2-Audio의 훈련 과정은 오디오 인코더와 대형 언어 모델을 포함합니다. 훈련 목표는 오디오 표현과 이전 텍스트 시퀀스를 조건으로 하여 다음 텍스트 토큰 확률을 극대화하는 것에 있습니다. Whisper-large-v3 모델을 기반으로 오디오 인코더를 초기화하였고, 원시 웨이브폼을 128채널 멜-스펙트로그램으로 변환하여 사용합니다. Qwen-7B 대형 언어 모델을 기본 구성 요소로 사용하여 총 8.2억 개의 매개변수를 가지고 있습니다. 자연 언어 프롬프트를 사용하여 일반화 능력과 지시 따르기 능력을 개선하였습니다.

- **Performance Highlights**: AIR-Bench 평가 결과에 따르면, Qwen2-Audio는 음성 중심의 지시 따르기 능력에서 이전 SOTA 모델인 Gemini-1.5-pro를 능가하였습니다. Qwen2-Audio는 Aishell2, FLUERS-zh, VocalSound 및 AIR-Bench 채팅 벤치마크 테스트 집합에서 최고 성능을 달성했습니다.



### Transforming Agency. On the mode of existence of Large Language Models (https://arxiv.org/abs/2407.10735)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)인 ChatGPT와 같은 시스템의 존재론적 특성을 조사합니다. 특히 이러한 모델들을 에이전트(agent)로 간주할 수 있는지에 대한 논의를 다룹니다. 논문은 LLMs의 아키텍처, 처리, 그리고 훈련 절차에 대해 자세히 설명하고, LLMs를 에이전트 유사 시스템으로 변환하기 위해 사용되는 확장 기술에 대해 분석합니다.

- **Technical Details**: LLMs가 자율적인 에이전시(autonomous agency)를 충족하지 못하는 이유를 다음 세 가지 조건을 통해 설명합니다: 개별성 조건(individuality condition), 규범성 조건(normativity condition), 그리고 상호작용 비대칭 조건(interactional asymmetry condition). LLMs는 자신의 활동의 산물이 아니며, 자체적인 규범이나 목표를 생성하지 않습니다. 또한 환경과의 상호작용의 기원 또는 지속적인 원천이 아닙니다.

- **Performance Highlights**: LLMs는 자율적인 에이전트가 아니라 언어적 자동화 장치(linguistic automaton)로 간주될 수 있으며, 목적이 없는 형태의 대화 경험을 생성할 수 있습니다. 인간과 상호작용할 때, 텍스트적인 구현(textual embodiment)과 대규모 연산 자원(resource-hungry computational embodiment)을 통해 기존의 인간 에이전시 형태를 변형시킵니다. LLM-인간 결합은 의도적인 에이전시를 생산하는 중간 형태의 에이전시(midtended forms of agency)를 생산할 수 있습니다.



### Sibyl: Simple yet Effective Agent Framework for Complex Real-world Reasoning (https://arxiv.org/abs/2407.10718)
Comments:
          Our code is available at this https URL

- **What's New**: 새로운 연구로, 복잡한 문제 해결 능력을 높이기 위해 설계된 LLM 기반 에이전트 프레임워크인 'Sibyl'이 소개되었습니다. Sibyl은 최소한의 도구 세트를 효과적으로 활용하여 기존의 LLM 에이전트가 못 이룬 장기적 추론 및 복잡한 시나리오에 대한 해결책을 제공합니다.

- **Technical Details**: Sibyl은 Global Workspace Theory에서 영감을 받아 지식과 대화 기록을 관리하고 공유하는 글로벌 워크스페이스를 통합합니다. 또한, Society of Mind Theory를 참고하여 다중 에이전트 토론 기반의 배심원을 통해 최종 답변을 스스로 개선하는 메커니즘을 구현합니다. 리엔트란시(reentrancy) 개념을 포함한 기능적인 프로그래밍 원칙을 적용해 확장성과 디버깅 용이성을 높였으며, 외부 정보 취득 채널과 선택적 압축을 위한 표현 언어를 도입해 정보 품질과 관련성을 증대시킵니다.

- **Performance Highlights**: GAIA 벤치마크 테스트 세트에서 GPT-4를 이용한 Sibyl 에이전트는 평균 34.55%의 점수로 기존 최첨단 방법을 능가했습니다. 특히, 더 복잡한 레벨 2와 레벨 3 시나리오에서 각각 32.7%와 16.33%의 점수를 기록하며 최고 성능을 보였습니다. 이런 결과는 긴 추론 체인들을 요구하는 복잡한 실제 문제를 해결하는 능력에서 Sibyl의 뛰어난 성능을 입증합니다.



### $\texttt{MixGR}$: Enhancing Retriever Generalization for Scientific Domain through Complementary Granularity (https://arxiv.org/abs/2407.10691)
- **What's New**: 최근 연구들은 과학 분야에서 LLM(대형 언어 모델)의 생성을 위해 문서 검색(document retrieval)의 중요성이 증가하고 있음을 보여줍니다. 그러나 dense retrievers는 도메인 특화 검색과 복잡한 쿼리-문서 관계에서 종종 어려움을 겪습니다. 이러한 문제를 해결하기 위해 본 논문에서는 $	exttt{MixGR}$을 도입합니다. 이는 다양한 수준의 세분화된 쿼리와 문서 간의 매칭에 대한 인식을 개선함으로써 dense retrievers를 향상시킵니다.

- **Technical Details**: $	exttt{MixGR}$은 제로 쇼트(zero-shot) 접근 방식으로 쿼리와 문서의 다양한 세분화 수준(granularity)에 기반한 여러 메트릭스를 결합하여 하나의 통합 점수를 만듭니다. 이 점수는 종합적인 쿼리-문서 유사성(query-document similarity)을 반영합니다.

- **Performance Highlights**: 실험 결과, $	exttt{MixGR}$는 감독되지 않은(unsupervised) retrievers에서 24.7%, 감독된(supervised) retrievers에서 9.8%로 nDCG@5에서 이전 문서 검색 기법을 능가했습니다. 다섯 개의 과학 검색 데이터셋에서 다중 서브쿼리를 포함한 쿼리를 평균으로 한 결과입니다. 또한, 두 가지의 과학 질문-응답(scientific question-answering) 작업에서의 효능이 $	exttt{MixGR}$이 과학 분야에서 LLM의 적용을 촉진하는 데 유리하다는 점을 강조합니다.



### Balancing the Scales: Reinforcement Learning for Fair Classification (https://arxiv.org/abs/2407.10629)
- **What's New**: 최근의 연구들은 분류 작업에서 공정성을 높이기 위해 알고리즘적 접근 방식을 강조하고 있습니다. 이 논문에서는 강화 학습(Reinforcement Learning, RL) 기법을 활용하여 편향을 줄이는 새로운 접근 방식을 제안합니다. 특히, 이 논문은 컨텍스츄얼 멀티암드 밴딧(Contextual Multi-Armed Bandit, CMAB) 프레임워크를 이용하여 각 클래스 내부에서 보호 그룹 간의 불균형을 감소시키는 보상 함수(Reward Function) 조정을 통해 공정성을 달성하는 방법을 탐구합니다.

- **Technical Details**: 이 연구는 CMAB를 이용하여 멀티클래스 분류 작업을 공정하게 처리하는 문제를 형식화하고, 세 가지 대표적인 RL 알고리즘을 개조하여 사용합니다. 각 알고리즘은 보호된 속성(Protected Attribute)의 불균형을 보상 함수 조정을 통해 완화하는 방식으로 공정성을 목표로 합니다. 또한 전통적인 지도학습(Baseline Supervised Learning)과의 성능을 비교하기 위해 다양한 스케일링 접근 방식을 통합합니다.

- **Performance Highlights**: 두 개의 공정 분류 데이터셋에서 실험한 결과, 이 논문의 RL 알고리즘은 기존 기준점(Baselines)과 경쟁력 있는 성능을 보였으며, 보상 스케일링이 분류 작업에서 편향을 줄이는 강력한 도구임을 입증했습니다. 깊이 있는 RL 알고리즘들은 멀티클래스 데이터셋에서 우수한 성능을 보여주었으며, 전통적인 CMAB 알고리즘은 바이너리 데이터셋에서 뛰어난 성과를 나타냈습니다. 추가로, 스케일링된 지도학습 구현은 기존 구현을 능가하며 멀티클래스 분류에서 최신의 성능을 보였습니다.



### Leave No Knowledge Behind During Knowledge Distillation: Towards Practical and Effective Knowledge Distillation for Code-Switching ASR Using Realistic Data (https://arxiv.org/abs/2407.10603)
- **What's New**: 최근의 자동 음성 인식(Automatic Speech Recognition, ASR)의 발전은 대규모 음성 기반 모델을 활용하여 고품질의 전사를 생성하는 것에 의존합니다. 그러나 이러한 모델은 제한된 컴퓨팅 자원으로 인해 실용적이지 않을 수 있습니다. 이 문제는 코드-스위칭 ASR(Code-Switching ASR, CS-ASR)과 같은 현실적이고 어려운 시나리오에서 더욱 심화됩니다. 이를 해결하기 위해 우리는 현실적인 음성만 데이터를 사용한 지식 증류(Knowledge Distillation) 방법으로 더 효율적인 CS-ASR 모델을 개발하는 프레임워크를 제안합니다. 제안된 방법, '지식 증류 과정에서 어떠한 지식도 남기지 않는다'(Leave No Knowledge Behind During Knowledge Distillation, K$^2$D)는 교사 모델의 지식과 작은 보조 모델의 추가 통찰을 활용합니다.

- **Technical Details**: 우리의 K2D 프레임워크는 세 가지 단계로 구성됩니다: 현실적인 의사 라벨링(pseudo-labeling), 데이터 사전 필터링(data pre-filtering), 그리고 지식 증류(knowledge distillation). 첫째, 우리는 현실적인 장기 음성 데이터를 사용하여 의사 라벨링을 수행하고 각 데이터를 작은 청크로 분할합니다. 다음으로, 보조 소형 모델의 추가 지식을 기반으로 청크 단위의 의사 라벨을 검증하는 데이터 사전 필터링을 진행합니다. 마지막으로, 검증된 데이터를 사용하여 지식 증류를 수행합니다. 교사 모델로 Whisper를 사용하여 timestamp와 전사를 생성하고, 보조 모델을 통해 데이터 필터링을 수행합니다.

- **Performance Highlights**: 우리의 K2D 방법은 두 개의 도메인 내(in-domain)와 두 개의 도메인 외(out-domain) 테스트 세트에서 평가되었으며, 모든 테스트 세트에서 기준 방법과 교사 모델을 능가했습니다. K2D를 통해 얻은 모델은 더 작고 더 빠르게 동작하며, 동일한 데이터 세트에서 교사 모델보다 두 배 더 작은 모델로, 생성 속도는 다섯 배 더 빠릅니다. K2D가 적용된 모델은 범용성 및 효율성이 뛰어남을 보여줍니다. 생성된 모델은 Hugging Face에 공개되었습니다.



### Learning Dynamics of LLM Finetuning (https://arxiv.org/abs/2407.10490)
Comments:
          32 pages

- **What's New**: 대규모 언어 모델(fine-tuning 시)의 학습 동역학(learning dynamics)을 분석한 연구가 공개되었습니다. 이 연구는 특정 훈련 예시가 다른 예시들에 미치는 예측 영향력을 체계적으로 분석했습니다. 이를 통해 많은 알고리즘들이 훈련되는 과정을 보다 일관되게 이해할 수 있게 되며, 이러한 방법들의 장점이 어디에서 오는지 설명할 뿐만 아니라, 정렬 성능(alignment performance)을 향상시킬 수 있는 간단하고 효과적인 방법을 제시합니다.

- **Technical Details**: 연구진은 학습 동역학의 단계별 분해(step-wise decomposition)와 다양한 응답 간의 누적 영향(accumulated influence)을 분석하는 프레임워크를 고안했습니다. 이 프레임워크는 instruction tuning과 preference tuning 알고리즘의 훈련을 일관되게 해석할 수 있게 해줍니다.

- **Performance Highlights**: 이 연구는 학습 동역학을 이해함으로써 정렬 성능을 더욱 향상시킬 수 있는 간단하고 효과적인 방법을 제안하며, 실험 코드가 공개되어 추가적인 검증 및 발전이 가능하다는 장점을 가집니다. 실험용 코드는 제공된 링크에서 확인할 수 있습니다.



### IDEAL: Leveraging Infinite and Dynamic Characterizations of Large Language Models for Query-focused Summarization (https://arxiv.org/abs/2407.10486)
- **What's New**: 본 논문은 Query-focused Summarization (QFS) 분야에서 새로운 접근법을 제안합니다. 이 접근법은 두 가지 필수적인 특성을 소개하며, 각각 길이문서 요약 및 효율적인 미세 조정된 질의와 대형 언어 모델(LLM)의 정렬을 다룹니다. 이러한 특성을 기반으로 두 개의 모듈, Query-aware HyperExpert와 Query-focused Infini-attention이 제안되었습니다.

- **Technical Details**: 논문에서는 IDEAL (Infinite and Dynamic largE languAge modeL-based framework)이라는 새로운 프레임워크를 제안합니다. IDEAL은 두 가지 모듈로 구성됩니다. 첫째, Query-aware HyperExpert는 Parameter-efficient Fine-tuning (PEFT) 전략을 활용하여 사용자의 질의에 따라 강력하게 관련된 LLM의 파라미터 변화를 동적으로 생성합니다. 둘째, Query-focused Infini-attention 모듈은 긴 문서를 적은 메모리로 처리하기 위해 설계되었으며, 질문에 집중된 압축 메모리를 포함합니다.

- **Performance Highlights**: 다양한 QFS 벤치마크 실험에서 IDEAL은 효과적이고 일반화 가능한 성능을 입증했습니다. 특히, LLAMA2-7B 백본 모델을 사용한 IDEAL은 단일 24GB 메모리의 Nvidia GeForce RTX 3090에서 입력 토큰의 평균 길이가 13,000인 데이터를 처리할 수 있음을 보여주었습니다. 추가로, 여러 QFS 데이터셋에서 기존 방법보다 뛰어난 성능을 입증했습니다.



### SuperPADL: Scaling Language-Directed Physics-Based Control with Progressive Supervised Distillation (https://arxiv.org/abs/2407.10481)
- **What's New**: SuperPADL을 통해 물리 기반 텍스트-투-모션(text-to-motion) 모델을 확장할 수 있는 프레임워크를 소개합니다. 이는 강화 학습(Reinforcement Learning, RL)과 지도 학습(Supervised Learning)을 결합하여 수천 개의 다양한 모션 클립을 학습하는 방식으로 구현됩니다. 이모델은 5000개 이상의 기술에 대해 실시간으로 동작하며 사용자와의 상호작용을 통해 다단계 애니메이션을 제작할 수 있습니다.

- **Technical Details**: SuperPADL은 프로그레시브 디스틸레이션(progressive distillation)을 통해 여러 스킬특화 컨트롤러로부터 시작하여 점진적으로 더 큰 범용 컨트롤러를 구축하는 방식입니다. 초기 스킬 별 전문가 컨트롤러들은 RL을 사용하여 학습되고, 이후 강화학습과 지도학습을 결합하여 점진적으로 디스틸되는 과정을 거칩니다. 최종 모델은 5000개 이상의 스킬에 대해 학습된 데이터셋을 기반으로 합니다.

- **Performance Highlights**: SuperPADL은 RL 기반의 기존 기준선 모델들 대비 큰 데이터 스케일에서 성능이 크게 우수합니다. 또한 소비자용 GPU에서도 실시간으로 동작이 가능하며, 사용자 명령의 변화에 따라 자연스럽게 기술 간 전환이 가능합니다.



### NTSEBENCH: Cognitive Reasoning Benchmark for Vision Language Models (https://arxiv.org/abs/2407.10380)
Comments:
          15 pages, 2 figures, 5 tables

- **What's New**: 새로운 데이터셋 NTSEBench가 소개되었습니다. 이 데이터셋은 인도의 NTSE(National Talent Search Examination) 시험에서 추출한 26개의 문제 카테고리와 4,642장의 이미지를 포함한 2,728개의 다중 선택 질문으로 구성돼 있습니다. NTSEBench는 대형 모델(LLMs, VLMs)의 인지적 다중 모드(reasoning) 및 문제 해결 능력을 평가하기 위해 설계되었습니다.

- **Technical Details**: NTSEBench는 텍스트와 이미지 형식을 모두 포함한 문제들을 다룹니다. 데이터셋은 주로 NTSE 시험지와 관련 학습 자료에서 추출하여 구성되었으며, MathPix OCR, docxlatex, PyMuPDF 등의 라이브러리를 사용하여 텍스트와 이미지를 추출했습니다. 데이터셋은 26개의 문제 카테고리와 8개의 다른 조합의 모달리티 유형을 포함하고 있습니다.

- **Performance Highlights**: 최신 LLMs 및 VLMs를 사용하여 NTSEBench의 기준선을 설정했습니다. 다양한 모달리티 입력을 처리하기 위해 네 가지 독립된 모델링 전략을 제안했으며, 여러 오픈 소스 및 독점 모델들의 성능을 비교 평가했습니다. 이를 통해 모델들이 인지적 문제 해결 능력에서 어떤 점에서 부족한지를 분석했습니다.



### Large Language Model-based FMRI Encoding of Language Functions for Subjects with Neurocognitive Disorder (https://arxiv.org/abs/2407.10376)
Comments:
          5 pages, accepted by Interspeech 2024

- **What's New**: 이 연구는 대형 언어 모델(LLM: Large Language Model)을 기반으로 한 fMRI 인코딩을 통해 신경인지장애(NCD) 노인의 언어 관련 뇌 기능 변화를 분석한 최초의 연구입니다. 특히, 기존 연구에서는 젊고 건강한 성인만을 대상으로 했지만, 이번 연구는 NCD 노인에게 초점을 맞추어 뇌 점수(brain scores)와 인지 점수 간의 상관관계를 분석했습니다.

- **Technical Details**: 연구진은 LlaMA2-7b-Cantonese 모델을 사용하여 영화 클립에서 나타나는 각 광동어 단어의 문맥 특징을 추출했습니다. 해당 LlaMA2-7b 모델은 Meta가 공개한 32층 모델로 영어와 중국어 데이터를 혼합해 훈련되었습니다. 연구에서는 헝가리어표준화인 MoCA 테스트(HK-MoCA)를 사용하여 참가자들의 인지 기능을 평가했습니다. fMRI 신호는 3 테슬라 MRI 스캐너를 사용하여 획득했고, Siemens MAGNETOM Prisma를 통해 얻은 fMRI 데이터는 SPM12 툴킷을 사용하여 전처리했습니다.

- **Performance Highlights**: 연구 결과, 높은 인지 능력을 가진 그룹이 더욱 우수한 뇌 점수를 가지며, 중간 측두회(middle temporal gyrus)와 상측 전두회(superior frontal gyrus)에서 인지 점수와의 상관관계가 가장 높게 나타났습니다. 이는 뇌 점수가 NCD 초기 변화를 감지하는 데 유용할 수 있음을 시사합니다.



### The Silent Curriculum: How Does LLM Monoculture Shape Educational Content and Its Accessibility? (https://arxiv.org/abs/2407.10371)
Comments:
          5 pages and 4 figures. Accepted at The Workshop on Global AI Cultures at the International Conference on Learning Representations, 2024 (ICLR'24)

- **What's New**: 이번 연구는 대형 언어 모델(LLM, Large Language Models)이 제공하는 정보가 기존 검색 엔진보다 더 편리하게 제공됨에 따라 새로운 관점이 전파될 가능성을 탐구합니다. 특히 어린이들이 이러한 디지털 오라클을 통해 지식을 얻는 과정에서 발생하는 'Silent Curriculum' 현상을 조사합니다.

- **Technical Details**: GPT-3.5와 LLaMA2-70B 모델을 활용해 문화적 표현과 직업적 고정관념의 미묘한 역학을 조사했습니다. 이 실험은 WinoBias 데이터셋에서 영감을 받아, 7개의 중복된 민족 그룹(백인, 흑인, 아시아인, 히스패닉, 아메리카 원주민, 중동인, 라틴 아메리카인)을 식별하고, 어린이 이름과 출생지를 중심으로 한 단편 소설을 생성하여 내재된 고정관념을 분석했습니다. 그 결과, 모델 간에 높은 코사인 유사성(0.87)이 발견되었습니다.

- **Performance Highlights**: 연구 결과, 두 모델(GPT-3.5와 LLaMA2-70B)은 아시아인, 백인, 히스패닉, 흑인 그룹의 캐릭터를 선호하는 경향을 보였으며, 라틴 아메리카인, 중동인, 아메리카 원주민의 표현 비율은 낮았습니다. 이는 LLM들이 새로운 문화적 편향을 형성하고 있음을 시사합니다.



### Sora and V-JEPA Have Not Learned The Complete Real World Model -- A Philosophical Analysis of Video AIs Through the Theory of Productive Imagination (https://arxiv.org/abs/2407.10311)
Comments:
          30 pages, 9 figures

- **What's New**: OpenAI의 Sora와 Meta의 V-JEPA AI 시스템의 세계 이해 능력에 대한 논쟁을 심화시키기 위해 Kant의 철학에 기반한 생산적 상상력 이론을 제시합니다. 이를 통해 고유한 세계 모델을 구축하는 데 필요한 세 가지 필수 요소를 식별했습니다: 고립된 객체의 표현, 시공간 전반의 사전적 변화 법칙(a priori law of change), 및 칸트적 범주(Kantian categories).

- **Technical Details**: Sora는 사전적 변화 법칙과 칸트적 범주를 간과하여 세계 이해에 제한이 있으며, 훈련을 확대한다고 해서 이러한 결함이 해결되지 않습니다. 반면 V-JEPA는 사전적 변화 법칙의 문맥적인 측면을 학습하지만 칸트적 범주를 완전히 이해하지 못하며 경험을 통합하지도 않습니다. 따라서 두 시스템 모두 현재로서는 포괄적인 세계 이해에 도달하지 못하고 있지만, 각각 핵심 구성 요소를 발전시키고 있습니다.

- **Performance Highlights**: V-JEPA는 Sora보다 사전적 변화 법칙에 대해 더 나은 성능을 보이지만, 전체적인 세계 이해 면에서는 아직 불완전합니다. 제시된 혁신적인 훈련 프레임워크는 복잡한 감각 입력을 구조화된 세계 모델로 변환하는 공동 임베딩 시스템(joint embedding system)을 중심으로 하여, 미래의 합리적 추론 및 계획에 적용될 수 있는 AI 시스템 개발 경로를 제안합니다.



### Improving Neural Biasing for Contextual Speech Recognition by Early Context Injection and Text Perturbation (https://arxiv.org/abs/2407.10303)
Comments:
          Accepted to INTERSPEECH 2024

- **What's New**: 이 연구는 기존 automatic speech recognition (ASR) 모델들의 한계를 극복하기 위해 새로운 문맥-인지 기술을 제안합니다. 주요 기법은 문맥을 인코더의 초기 단계에 주입하고, 모델이 문맥을 활용하도록 훈련 시 대체 철자를 활용하는 것입니다. 이로 인해 희귀 단어 인식률이 크게 향상되었습니다.

- **Technical Details**: 첫 번째 기법은 문맥을 인코더 초기 단계에 주입하여 문맥-인지 neural biasing을 강화하는 것입니다. 이는 이전의 연구들과 달리 특정 레이어와 문맥 통합 층의 수를 명확히 밝혔습니다. 두 번째 기법은 훈련 시 대체 철자를 사용하는 것입니다. 예를 들어 'Klein'을 'Klane'으로 바꿔 모델이 문맥에 의존해 올바른 예측을 하도록 합니다. 이러한 접근은 기존의 다양한 철자 대체 알고리즘을 참고하였지만, 단순한 수제 철자 규칙을 사용하여 구현되었습니다.

- **Performance Highlights**: 제안된 기법들은 LibriSpeech에서 희귀 단어 오류율을 각각 60%와 25% 감소시켜 새로운 state-of-the-art 성능을 기록했습니다. 또한, SPGISpeech와 실제 데이터세트인 ConEC에서도 기존 기준들에 비해 성능이 향상되었습니다.



### Modern Information Technologies in Scientific Research and Educational Activities (https://arxiv.org/abs/2407.10296)
Comments:
          Monograph Scientific publication (issue). Published By Iowa State University Digital Press. ISBN 978-1-958291-07-8; 273 pages; Published May 1, 2024

- **What's New**: 이번 논문은 상호작용 인공지능 시스템, 텍스트 생성 시스템, 전문 인력 경쟁력 진단, 이미지 생성 시 올바른 색상 구현, 대학원생 업무 정보화, 3D 모델 생성 기술 등 인공지능 및 정보 기술 분야의 최신 연구 현황을 요약하고 분석했습니다. 이 논문은 IT 분야에서 일하는 전문가뿐만 아니라 교사, 석사, 학생, 대학원생 등 정보 기술에 관심 있는 모든 사람에게 유용할 것입니다.

- **Technical Details**: 이번 논문은 2023년 10월에 개최된 제16회 국제 과학 실무 회의인 정보 기술 및 자동화 - 2023 (Information technologies and automation - 2023) 결과를 바탕으로 오데사 국립 기술 대학 (Odessa National University of Technology)에서 작성되었습니다. 다양한 분야의 최신 연구 결과와 기술 동향이 포함되어 있습니다. 예를 들어, 'interactive AI systems', 'text generation systems'와 같은 최신 기술 동향과 이들의 응용 방법, 3D 모델링 기술 등을 다루고 있습니다.

- **Performance Highlights**: 세부적으로 논문은 상호작용 인공지능 시스템의 현재 상태를 비롯해 텍스트 생성 시스템의 효율성, 올바른 색상 구현 방법론, 정보화 기술을 통한 대학원생들의 업무 개선 효과 등을 조명합니다. 이러한 다양한 주제들을 체계적으로 분석해 앞으로의 기술 발전 방향을 제시하고 있습니다.



### Fine-grained Analysis of In-context Linear Estimation: Data, Architecture, and Beyond (https://arxiv.org/abs/2407.10005)
- **What's New**: 최근 연구에 따르면 선형 Attention을 사용하는 트랜스포머 (Transformers)는 경사 하강법 (gradient descent) 단계를 통해 맥락 학습 (in-context learning, ICL)이 가능하다는 사실이 밝혀졌습니다. 이번 연구에서는 ICL의 최적화 및 일반화 풍경을 다양한 직접적인 설정에서 더욱 강력하게 특성화하였습니다.

- **Technical Details**: 이번 연구는 일층 선형 Attention과 일층 H3 상태 공간 모델에서의 최적화 풍경을 분석합니다. 적절한 상관 설계 가정 하에, 두 모델 모두 1단계 전처리된 경사 하강법을 구현한다는 것을 증명했습니다. 특히 H3는 자체적으로 샘플 가중치를 적용하고, 이러한 설정에서 선형 Attention보다 우수한 성능을 보이는 장점이 있습니다. 또한, 상관 설계를 연구함으로써 Retrieval Augmented Generation (RAG)의 새로운 리스크 한계와 과제-특징 정렬(Task-Feature Alignment)이 ICL 샘플 복잡도에 미치는 영향을 밝혔습니다. 마지막으로, 공분산 스펙트럼 (covariance spectrum)을 통해 저차원 파라미터화된 Attention 가중치의 최적 리스크를 도출하고, LoRA가 작업 공분산 사이의 변화를 포착하여 새 분포에 적응하는 방식에 대해 설명하였습니다.

- **Performance Highlights**: 실험 결과는 이론적 결론을 뒷받침하며, ICL의 최적화 및 리스크 풍경에 대한 실용적인 이해를 심화시키는 데 기여합니다.



### Mitigating Interpretation Bias in Rock Records with Large Language Models: Insights from Paleoenvironmental Analysis (https://arxiv.org/abs/2407.09977)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs: Large Language Models)을 이용하여 지질 분석의 해석 편향을 줄이고, 보다 정확하고 신뢰할 수 있는 결과를 도출하는 혁신적인 접근법을 소개합니다. 이 연구는 퇴적학(sedimentology)과 고지형학(paleogeography)에 이 프레임워크를 적용하여, 동일한 데이터에 대한 여러 가설(hypotheses)을 생성하고 평가함으로써, 해석 편향을 효과적으로 감소시킵니다.

- **Technical Details**: 이 접근법은 LLMs와 더불어 검색 기반 생성(retrieval augmented generation) 및 실시간 검색 능력을 포함합니다. 이러한 기능들은 해석 과정에서 발생하는 인간의 편향을 줄이는데 중요한 역할을 합니다. 이를 통해 다양한 지구 과학 하위 분야에서 LLMs의 적용 가능성을 넓히고, 보다 깊고 정확한 지구 진화의 묘사가 가능해집니다.

- **Performance Highlights**: 연구 결과, 이 프레임워크는 지구 환경 연구의 정교함을 향상시키고 학문적 접근 방식을 재정립하며, 해석의 다변성과 정확성을 증가시키는데 있어 탁월한 성과를 보여줍니다. 특히, 다중 가설 접근법을 통해 인간 편향을 줄여 믿을 수 있는 지질 해석 결과를 도출하는 데 큰 기여를 합니다.



### The GPT Surprise: Offering Large Language Model Chat in a Massive Coding Class Reduced Engagement but Increased Adopters Exam Performances (https://arxiv.org/abs/2407.09975)
Comments:
          32 pages

- **What's New**: 이 연구는 다양한 학습 경험에서 빠르게 채택되고 있는 대형 언어 모델(LLMs)이 학생 학습에 미치는 영향을 평가했습니다. 이러한 연구의 일환으로, 전 세계 146개국에서 5,831명의 학생들이 참여한 온라인 코딩 수업에서 랜덤화 대조 실험을 진행했습니다. 일부 학생들에게는 GPT-4가 탑재된 채팅 인터페이스에 접근할 수 있게 했습니다.

- **Technical Details**: LLMs는 코딩 작업에서 강력한 성능을 발휘하며, 현재 전문가들 사이에서도 널리 사용되고 있습니다. 이번 실험에서는 GPT-4를 활용해 학생들이 코딩 교육에서 얻는 이점을 평가하고자 했습니다. 예를 들어, 채팅 인터페이스를 사용하여 코딩 문제 해결에 도움을 주는 방식입니다.

- **Performance Highlights**: 채팅 도구를 사용한 학생들은 시험 성적에서 긍정적인 효과를 봤지만, 전반적으로 GPT-4를 홍보한 후 시험 참여율이 눈에 띄게 감소했습니다. 또한, 이러한 참여 감소는 학생의 출신 국가에 따라 다르게 나타났습니다. 특히, 인간 개발 지수가 낮은 나라의 학생들에게 LLMs에 대한 접근을 제공한 경우, 이들의 시험 참여율이 평균적으로 증가했습니다. 따라서 LLMs를 교육 현장에서 도입하는 데에는 학생 성공에 대한 장기적인 영향에 대해 추가적인 조사가 필요합니다.



### To what extent is ChatGPT useful for language teacher lesson plan creation? (https://arxiv.org/abs/2407.09974)
- **What's New**: 이 논문에서는 ChatGPT를 사용하여 외국어 교사를 위한 수업 계획을 생성하면서 발생하는 여러 경향성을 평가하였습니다. 특히 제로샷(Zero-shot) 프롬프트를 사용해 높은 품질의 수업 계획을 생성할 수 있었지만, 프롬프트에 추가적인 맥락과 구체성을 더할 때 항상 품질이 향상되는 것은 아니었습니다.

- **Technical Details**: 프롬프트(pormpt)의 구체성과 다양성, 그리고 생성된 수업 계획의 약점을 중점으로 살펴보았습니다. 프롬프트의 복잡성을 점차 높여가며 실험한 결과, 동일한 프롬프트를 사용해 생성된 출력물에서 극단적인 변동성을 관찰했습니다. 이러한 변동성은 종종 20세기와 21세기의 교육적 실천 사이의 갈등을 반영했습니다. 이는 생성형 AI 모델이 클래식 교육 문헌을 기반으로 훈련된 경우, 오래된 교육 실천으로 편향될 가능성이 있음을 시사합니다.

- **Performance Highlights**: ChatGPT는 제로샷 프롬프트에서도 전반적으로 높은 품질의 수업 계획을 생성했습니다. 그러나 동일한 프롬프트를 사용하더라도 결과물 사이의 변동성이 크게 나타났고, 이는 생성된 컨텐츠가 21세기 교육 관행과 일치하지 않는 경우가 많았습니다. 이러한 결과는 실무 및 외국어 교사 교육에 AI 도구를 사용하는 데 즉각적인 적용 가능성을 제시합니다.



### Transferring Structure Knowledge: A New Task to Fake news Detection Towards Cold-Start Propagation (https://arxiv.org/abs/2407.09894)
Comments:
          ICASSP 2024

- **What's New**: 최근 연구들은 유효한 의미적 및 구조적 특징을 추출하여 가짜 뉴스 탐지의 성능을 향상시켜왔습니다. 하지만 전파 데이터가 없는 뉴스에 대해 훈련된 전파 기반 모델을 적용하기 어려운 실정입니다. 이를 해결하기 위해 콘텐츠만을 기반으로 가짜 뉴스를 탐지하는 '콜드 스타트 가짜 뉴스 탐지(cold-start fake news detection)'라는 새로운 과제를 제시했습니다. 이를 위해 전파 데이터를 포함한 샘플에서 학습한 후 전파 데이터가 없는 콘텐츠만의 샘플을 예측하는 'Structure Adversarial Net (SAN)' 프레임워크를 설계했습니다.

- **Technical Details**: SAN 프레임워크는 전파 데이터가 있는 샘플과 없는 샘플 사이의 특징 불일치를 최소화하도록 설계되었습니다. 구조적 판별자(Structure Discriminator)를 도입하여 전파 데이터의 유무에 따른 특징을 평가하고 구조 불변 특징을 학습합니다. 이를 통해 기존 전파 기반 방법의 일반화 능력을 향상시키고, 콘텐츠만 있는 샘플에 대한 탐지 성능을 제고합니다. 학습 단계에서는 전파 데이터와 콘텐츠를 모두 포함하는 샘플에서 구조와 의미적 패턴을 학습하고, 테스트 단계에서는 전파 데이터가 없는 샘플에 대해 예측을 수행합니다.

- **Performance Highlights**: 세 가지 데이터셋에 대해 정성적 및 정량적 실험을 수행했습니다. 실험 결과는 기존 전파 기반 모델이 전파 데이터가 없을 때 성능이 저하되는 어려움을 나타내며, 제안된 SAN 프레임워크가 이러한 조건에서도 탐지 성능을 지속적으로 향상시킴을 보여줍니다.



### Speech-Copilot: Leveraging Large Language Models for Speech Processing via Task Decomposition, Modularization, and Program Generation (https://arxiv.org/abs/2407.09886)
Comments:
          8 pages, 2 figures

- **What's New**: 이번 연구에서는 Speech-Copilot이라는 모듈형 프레임워크를 소개합니다. 이 프레임워크는 음성 처리 작업을 지시 기반으로 효율적으로 수행할 수 있도록 설계되었습니다. 기존의 대형 오디오-언어 모델을 사용하는 엔드 투 엔드 방식과 달리, Speech-Copilot은 사전 수집된 작업 지시를 분석하고 이를 관리 가능한 하위 작업으로 분해하여 특화된 도구 세트를 구축합니다. 이를 통해 대규모 언어 모델 기반의 유연한 에이전트가 프로그래밍을 통해 작업을 수행할 수 있게 합니다.

- **Technical Details**: Speech-Copilot는 도구 세트 구축과 대형 언어 모델(LLM) 기반 에이전트라는 두 가지 주요 구성 요소로 이루어져 있습니다. 도구 세트 구축은 LLM이 다양한 사전 수집된 작업 지시를 분석하고 이를 하위 작업으로 분해해 기본 모듈로 정리하는 방식으로 진행됩니다. 이후, 사람의 개입을 최소화한 상태로, LLM이 코드 모듈로 구현하며 이를 적절한 음성 모델과 결합해 사용합니다. 에이전트는 이러한 모듈을 프로그래밍을 통해 조합하여 다양한 작업을 해결하며, 이 방법은 기존의 대형 오디오-언어 모델 및 계단식 시스템 대비 뛰어난 성과를 보입니다.

- **Performance Highlights**: Speech-Copilot은 Dynamic-SUPERB 벤치마크에서 최첨단 성능(state-of-the-art performance)을 달성했습니다. 특히, 여러 작업을 단일 사용자 쿼리에서 다룸에도 성과 저하 없이 수행이 가능하며, 강력한 다중 작업 능력을 보여주었습니다. 연구진은 이 시스템을 데모 페이지를 통해 공개하였으며, 커뮤니티 사용을 위해 코드도 공개할 예정입니다.



### Empowering Whisper as a Joint Multi-Talker and Target-Talker Speech Recognition System (https://arxiv.org/abs/2407.09817)
Comments:
          Accepted to INTERSPEECH 2024

- **What's New**: 본 논문에서는 Whisper라는 음성 모델을 활용하여 다중 화자 음성 인식(multi-talker speech recognition) 및 대상 화자 음성 인식(target-talker speech recognition)을 동시에 해결하는 새로운 접근법을 제안합니다. 제안된 방법은 Whisper의 파라미터를 고정하고 사이드카 분리기(Sidecar separator)를 추가하여 여러 화자의 혼합된 임베딩을 분리하고, Target Talker Identifier를 도입하여 대상 화자의 임베딩 플로우를 실시간으로 식별합니다. 또한 디코더의 소프트 프롬프트 튜닝(soft prompt tuning)을 통해 더욱 효율적인 작업 적응을 도모합니다.

- **Technical Details**: 제안된 방법은 네 가지 주요 구성 요소로 이루어져 있습니다: Whisper, 사이드카 분리기, Target Talker Identifier, 그리고 소프트 프롬프트 튜닝입니다. Whisper는 주의기반(Attention-based) 인코더-디코더 구조를 특징으로 하는 음성 인식 모델로, 대규모 웹 스케일 라벨링된 음성 데이터를 바탕으로 훈련되었습니다. 사이드카 분리기는 혼합 임베딩을 분리하기 위해 Whisper의 인코더에 삽입된 시계열 합성곱 네트워크로, Conv-TasNet에서 영감을 받아 설계되었습니다. Target Talker Identifier는 등록된 음성을 기준으로 대상 화자의 임베딩 플로우를 실시간으로 식별하며, 소프트 프롬프트 튜닝은 디코더의 작업 적응을 향상시킵니다.

- **Performance Highlights**: 제안된 방법은 두 명과 세 명의 화자가 포함된 LibriMix 및 LibriSpeechMix 데이터셋에서 최첨단 성능을 보여주며, AishellMix 중국어 데이터셋에서도 제로-샷(multi-talker ASR) 성능이 우수합니다. 본 기술은 적은 훈련 파라미터로도 높은 성능을 발휘할 수 있으며, 분할 및 식별 과정에서의 계산 오버헤드를 최소화하여 효율성을 극대화합니다.



### IoT-LM: Large Multisensory Language Models for the Internet of Things (https://arxiv.org/abs/2407.09801)
- **What's New**: 최신 연구에서는 IoT-LM이라는 새로운 대형 다중 감각 언어 모델을 소개합니다. 이 모델은 IoT 센서 데이터를 처리하고 다양한 예측, 질문 응답, 추론 및 상호 대화 작업을 수행할 수 있도록 설계되었습니다. 주요 혁신에는 가장 광범위한 통합 IoT 데이터셋인 MultiIoT가 포함되어 있으며, 이 데이터셋은 1.15백만 샘플과 12가지 감각 모달리티 및 8가지 작업을 포함합니다. 또한 IoT-LM을 위한 새로운 다중 감각 멀티태스크 어댑터 레이어도 포함되어 있습니다.

- **Technical Details**: IoT-LM의 핵심 아키텍처는 다양한 IoT 센서 입력을 수용하고, 멀티태스크 인코더를 통해 여러 센서와 작업 간의 정보를 융합합니다. 인코더는 학습된 표현을 사전 훈련된 LLM 입력 공간으로 변환하며, IoT-LM은 이를 통해 다양한 감각 데이터를 효과적으로 처리하고 추론할 수 있습니다. 이 접근법은 감각 데이터를 초기에 융합하거나 나중에 융합하거나 모델 기반 융합 방법을 사용해, 작업의 요구 사항에 맞게 적절히 선택합니다.

- **Performance Highlights**: IoT-LM은 8가지 감독 IoT 분류 작업에서 상당한 성능 향상을 보여주었습니다. 또한 IoT 센서 데이터를 기반으로 한 새로운 상호 질문-응답, 추론 및 대화 능력을 입증했습니다. 더 나아가, IoT-LM의 데이터 소스 및 새로운 다중 감각 언어 모델링 프레임워크를 공개하여 연구 커뮤니티에 기여하고 있습니다.



### GOFA: A Generative One-For-All Model for Joint Graph Language Modeling (https://arxiv.org/abs/2407.09709)
- **What's New**: 이번 논문에서는 기존 텍스트 및 이미지 데이터와 달리 명확한 구조가 없는 그래프 데이터를 효과적으로 다룰 수 있는 Graph Foundation Model (GFM)의 필요성을 강조합니다. 이를 위해 저자들은 새로운 그래프 생성 언어 모델인 GOFA(Generative One-For-All)를 제안합니다. GOFA는 무작위로 초기화된 GNN 레이어를 사전 학습된 LLM에 통합하여 그래프 구조를 이해하면서도 텍스트 생성 능력을 유지합니다.

- **Technical Details**: 저자들은 GFM이 가져야 할 세 가지 주요 속성으로 '대규모 자가 지도 학습(self-supervised pretraining)', '유연한 태스크 처리(fluidity in tasks)', '그래프 인식(graph awareness)'을 정의합니다. 이러한 속성을 만족시키기 위해 GNN 레이어를 LLM 내부에 삽입하여 구조적 및 의미적 모델링 능력을 유기적으로 결합하였습니다. GOFA는 그래프 수준의 다음 단어 예측, 질문-답변, 구조적 태스크 등을 통해 사전 학습됩니다.

- **Performance Highlights**: GOFA는 다양한 다운스트림 태스크에서 평가되었으며, 특히 제로샷(zero-shot) 시나리오에서 뛰어난 성능을 보였습니다. 이를 통해 GOFA가 그래프 도메인에서 범용적인 그래프 분석 모델로서의 잠재력을 입증했습니다. 또한, 코드도 공개되어 있어 연구자들이 쉽게 접근하고 활용할 수 있습니다.



### Bridging Dictionary: AI-Generated Dictionary of Partisan Language Us (https://arxiv.org/abs/2407.09661)
Comments:
          Accepted to CSCW Demo 2024

- **What's New**: 새로운 연구는 각기 다른 정치적 배경을 가진 사람들이 단어를 다르게 해석하는 문제를 해결하는 'Bridging Dictionary'라는 도구를 소개합니다. 이 도구는 정치적 견해에 따라 단어가 어떻게 사용되는지 조명하여, 오해를 방지하고 효과적인 소통을 돕기 위해 설계되었습니다.

- **Technical Details**: Bridging Dictionary는 두 가지 주요 구성 요소로 이루어져 있습니다: 종이 사전과 인터랙티브 데모입니다. 종이 사전에는 796개의 용어와 각 용어에 대한 정치적 견해에 따른 요약이 포함되어 있습니다. 인터랙티브 데모는 사용자가 특정 단어의 사용 빈도, 감정 분석, 요약 및 예시를 탐색할 수 있도록 돕습니다. 요약 생성은 gpt-3.5-turbo를 이용해 수행되었으며, 데이터셋은 2020년 미국 선거 기간 동안 수집된 4.7백만 개의 트윗에서 추출되었습니다.

- **Performance Highlights**: 이 도구는 뉴스 콘텐츠를 작성하는 저널리스트들에게 도움이 될 수 있는 데이터를 제공합니다. 특히, 통계와 요약 기능은 정치적 언어 사용 차이를 이해하는 데 유용하다는 평가를 받았습니다. 또한, 토픽 산포도와 샘플 리스트는 LLM 생성된 콘텐츠를 지원하고, 정보 출처를 검토하는 데 도움을 줍니다.



### BitNet b1.58 Reloaded: State-of-the-art Performance Also on Smaller Networks (https://arxiv.org/abs/2407.09527)
Comments:
          17 pages, 4 Tables, 14 grafs. Accepted to DeLTA

- **What's New**: 새로운 연구는 1.58비트 양자화 인지 훈련(Quantization Aware Training)에 관한 것으로, 작은 언어 모델과 비전 모델에 대한 적용을 조사합니다. 비트넷(BitNet) b1.58의 변형 버전을 도입하여 평균 대신 중간값을 사용하는 양자화 방법을 제안합니다. 이 방법은 이전보다 더 작은 모델에서도 동작할 수 있음을 보여줍니다.

- **Technical Details**: 이 연구에서는 BitNet b1.58 아키텍처를 일반화한 양자화 인지 훈련 체계를 제안합니다. 설계된 BitLinear 레이어는 PyTorch의 torch.nn.Linear 레이어를 대체할 수 있도록 설계되었으며, 양자화된 가중치와 활성화 함수들을 사용해 전체 연산을 수행합니다. 특히, 가중치의 양자화는 중간값 및 평균값을 이용해 수행될 수 있으며, 이는 다양한 양자화 전략을 보다 유연하게 적용할 수 있게 합니다.

- **Performance Highlights**: 실험 결과, 1.58비트 양자화 인지 훈련은 작은 언어 모델에서 성능을 거의 유지하면서도, 비전 모델에서는 동급 최고의 성능을 초과하는 결과를 보여주었습니다. 작은 모델에서도 이 기술이 적용 가능하다는 점을 강조하며, 이는 저자원이 요구되는 상황에서도 딥러닝 모델을 효과적으로 배치할 수 있게 합니다.



### Putting GPT-4o to the Sword: A Comprehensive Evaluation of Language, Vision, Speech, and Multimodal Proficiency (https://arxiv.org/abs/2407.09519)
- **What's New**: 이번 연구는 GPT-4o 모델의 언어, 비전, 음성, 멀티모달(multimodal) 역량을 종합적으로 평가합니다. 연구는 표준 시험 문제, 추론 작업, 번역 평가를 통해 언어 능력을 평가하고, 이미지 분류 및 객체 인식 작업, 그리고 악센트 분류를 통해 비전과 음성 능력을 테스트합니다.

- **Technical Details**: 연구는 표준화된 시험 문제와 추론 작업(reasoning tasks), 번역 평가를 사용해 GPT-4o 모델의 언어 능력을 평가했습니다. 비전과 음성 능력은 이미지 분류(image classification), 객체 인식(object recognition), 악센트 분류(accent classification) 작업을 통해 테스트했습니다. 또한 멀티모달 평가를 통해 시각적 데이터와 언어적 데이터를 통합하는 능력을 평가했습니다.

- **Performance Highlights**: GPT-4o 모델은 여러 도메인에서 높은 정확성과 효율성을 보여주었으며, 특히 몇 샷 학습(few-shot learning)을 요구하는 작업에서 우수한 성과를 나타냈습니다. 멀티모달 과제에서도 이전 모델보다 눈에 띄는 개선을 보였습니다. 하지만 복잡하고 모호한 입력 처리에서 특히 오디오 및 비전 능력에서 변동성과 한계를 보였습니다. 논문은 더욱 포괄적인 벤치마크와 견고한 평가 프레임워크의 필요성을 강조하며, 미래 연구에서는 데이터셋 확대, 프롬프트 기반 평가(prompt-based assessment), 몇 샷 학습 기술 향상이 필요하다고 제안합니다.



### Using Artificial Intelligence to Unlock Crowdfunding Success for Small Businesses (https://arxiv.org/abs/2407.09480)
- **What's New**: 최근 AI 기술을 활용하여 크라우드펀딩 캠페인의 성공 여부를 예측하고 향상시키는 연구가 발표되었습니다. 특히, 텍스트 설명 (textual descriptions)을 최적화하여 자금 조달을 위한 전략을 개선하는 방법을 제시하고 있습니다.

- **Technical Details**: 가장 성능이 우수한 머신 러닝 모델 (machine learning model)은 크라우드펀딩 캠페인의 성공 여부를 81.0%의 정확도로 예측하며, 이는 주로 텍스트 설명에 기반을 둡니다. 이 모델을 해석하여 캠페인 설명을 개선하기 위한 실질적인 제안을 제공할 수 있습니다. 또한, 대형 언어 모델 (large language model)을 활용하여 텍스트의 세 가지 측면을 강화함으로써 캠페인이 더욱 매력적으로 보인다는 것을 검증하였습니다.

- **Performance Highlights**: 텍스트 내러티브의 세 가지 측면을 강화함으로써, 캠페인이 83%의 사람들에게 더 매력적으로 느껴지며 자금 지원 가능성이 11.9% 증가하는 것으로 나타났습니다. 이 연구는 소규모 비즈니스가 효과적인 크라우드펀딩 캠페인 설명을 작성할 수 있게 하는 새로운 전략을 제시합니다.



### Prefixing Attention Sinks can Mitigate Activation Outliers for Large Language Model Quantization (https://arxiv.org/abs/2406.12016)
- **What's New**: 이번 논문에서는 LLM (Large Language Models)의 활성화 양자화 (activation quantization)를 개선하기 위한 새로운 방법인 CushionCache를 제안합니다. 이는 문제가 되는 토큰의 생성을 방지하며, 효과적으로 문제를 해결합니다. CushionCache는 키-값 캐시 (key-value cache)를 사용하는 방식으로, 활성화 값의 이상치 (outliers)를 완화합니다.

- **Technical Details**: CushionCache는 두 단계로 작동합니다. 첫 번째 단계에서는 이후 토큰들의 최대 활성화 값을 최소화하는 프롬프트 토큰 시퀀스를 탐색합니다. 두 번째 단계에서는 토큰 캐시를 추가로 튜닝하여 활성화 값을 양자화 친화적으로 만드는 것입니다. 이 방법은 LLM 활성화 양자화에서 발생하는 이상치 문제를 효과적으로 해결합니다.

- **Performance Highlights**: 광범위한 모델과 벤치마크에서의 평가를 통해 CushionCache는 기존의 W8A8 양자화 방법보다 성능이 크게 향상됨을 확인했습니다. 특히, LLaMA3-8B에서 W8A8 퍼텐서 정적 범위 양자화의 상태를 30%p 이상 개선하는 성과를 보였습니다.



New uploads on arXiv(cs.IR)

### SEMINAR: Search Enhanced Multi-modal Interest Network and Approximate Retrieval for Lifelong Sequential Recommendation (https://arxiv.org/abs/2407.10714)
Comments:
          9 pages,code released

- **What's New**: 최신 논문에서는 현대의 추천 시스템에서 사용자 행동을 모델링하는 것이 중요함을 강조합니다. 이 연구에서는 검색 강화 다중 모드 관심 네트워크 및 근사 검색(Search Enhanced Multi-Modal Interest Network and Approximate Retrieval, SEMINAR)이라는 새로운 모델을 제안합니다. SEMINAR는 사용자의 평생 역사적 다중 모드 행동을 모델링하며, 검색 쿼리 시퀀스와 항목 열람 시퀀스를 통합하여 사용자 의도를 보다 정확하게 파악할 수 있도록 합니다.

- **Technical Details**: SEMINAR 모델은 Pretraining Search Unit (PSU) 네트워크를 통해 역사적 다중 모드 쿼리-항목 쌍(sequence)의 평생 시퀀스를 학습합니다. PSU는 다중 모드 정렬, 다음 쿼리-항목 쌍 예측, 쿼리-항목 관련성 예측 등의 여러 목표를 포함하는 사전학습(finetune) 방식으로 작동합니다. 이 모델은 클릭률(Click-Through Rate, CTR) 예측 및 개인화된 검색 순위(Personalized Search Ranking, PSR) 작업에 적용할 수 있습니다. 아울러, 다중 모드 코덱 기반 제품 양자화 전략(product quantization strategy)을 도입하여 정확한 주의(attention) 계산 속도를 가속화했습니다.

- **Performance Highlights**: 연구진은 실제 데이터셋을 기반으로 한 광범위한 실험을 통해 SEMINAR 모델의 효과를 입증했습니다. 또한, 코드도 공개하여 추후 연구를 장려하고 있습니다(https://github.com/paper-submission-coder/SEMINAR).



### $\texttt{MixGR}$: Enhancing Retriever Generalization for Scientific Domain through Complementary Granularity (https://arxiv.org/abs/2407.10691)
- **What's New**: 최근 연구들은 과학 분야에서 LLM(대형 언어 모델)의 생성을 위해 문서 검색(document retrieval)의 중요성이 증가하고 있음을 보여줍니다. 그러나 dense retrievers는 도메인 특화 검색과 복잡한 쿼리-문서 관계에서 종종 어려움을 겪습니다. 이러한 문제를 해결하기 위해 본 논문에서는 $	exttt{MixGR}$을 도입합니다. 이는 다양한 수준의 세분화된 쿼리와 문서 간의 매칭에 대한 인식을 개선함으로써 dense retrievers를 향상시킵니다.

- **Technical Details**: $	exttt{MixGR}$은 제로 쇼트(zero-shot) 접근 방식으로 쿼리와 문서의 다양한 세분화 수준(granularity)에 기반한 여러 메트릭스를 결합하여 하나의 통합 점수를 만듭니다. 이 점수는 종합적인 쿼리-문서 유사성(query-document similarity)을 반영합니다.

- **Performance Highlights**: 실험 결과, $	exttt{MixGR}$는 감독되지 않은(unsupervised) retrievers에서 24.7%, 감독된(supervised) retrievers에서 9.8%로 nDCG@5에서 이전 문서 검색 기법을 능가했습니다. 다섯 개의 과학 검색 데이터셋에서 다중 서브쿼리를 포함한 쿼리를 평균으로 한 결과입니다. 또한, 두 가지의 과학 질문-응답(scientific question-answering) 작업에서의 효능이 $	exttt{MixGR}$이 과학 분야에서 LLM의 적용을 촉진하는 데 유리하다는 점을 강조합니다.



### General algorithm of assigning raster features to vector maps at any resolution or sca (https://arxiv.org/abs/2407.10599)
- **What's New**: 이 연구는 다양한 해상도, 크기, 스케일의 래스터(raster) 및 벡터(vector) 데이터를 통합하여 지리적 분석을 가능하게 하는 일반 알고리즘을 제안합니다. 이 알고리즘은 래스터 데이터의 특성(예: 공기 오염 농도)을 도시 지도의 벡터 구성 요소(예: 도로)에 할당하는 과정을 포함하며, 2D 투사된 지도에서 도시 중심에서 경계로 확장하는 가상 레이어를 반복적으로 구성합니다.

- **Technical Details**: 제안된 알고리즘은 도시 지도를 일정한 해상도의 그리드로 나누고, 각 그리드에 오염 데이터를 할당합니다. 도로는 벡터 데이터의 엣지(edge)로 표현되며, 이 엣지에 오염 데이터(PM2.5, NO2)를 정확하게 연결하는 과정이 있습니다. 알고리즘은 도시 크기와 그리드 해상도에 따라 홀수 또는 짝수 단계로 나뉘며, 가상 레이어를 사용하여 점진적으로 공간을 확장하여 최종 레이어를 형성합니다.

- **Performance Highlights**: 이 알고리즘은 전 세계 1692개 도시에서 PM2.5 및 NO2 농도를 도로에 할당하는 데 성공적으로 적용되었습니다. 이는 다중 해상도 및 구성이 다른 데이터 세트를 정확하게 융합할 수 있는 효율적이고 범용적인 방법을 제공하며, 긴급한 기후 문제에 대한 기민한 연구를 가능하게 합니다.



### Numbers Matter! Bringing Quantity-awareness to Retrieval Systems (https://arxiv.org/abs/2407.10283)
- **What's New**: 이 논문에서는 수량(quantitative) 정보를 이해하고 처리할 수 있는 두 가지 수량 인식 랭킹(Quantity-aware Ranking) 기법을 소개합니다. 이는 문서의 수량과 텍스트 내용을 결합하여 보다 정확한 검색 결과를 제공하며, 금융과 의학 도메인의 새로운 벤치마크 데이터셋 또한 제공합니다.

- **Technical Details**: 제안된 방법은 두 가지 접근 방식을 이용합니다. 첫 번째는 'disjoint combination', 두 번째는 'joint ranking' 방식입니다. 'Disjoint combination'은 독립적으로 수량과 텍스트를 순위 매길 수 있는 방법으로, 다양한 어휘 및 의미 기반 IR 시스템과 호환되는 인덱스 구조를 사용합니다. 반면, 'joint ranking'은 신경망 기반의 IR 모델을 사용하여 수량-인식 문서 및 쿼리 표현을 학습합니다. 이 과정에서 수량과 텍스트의 상호작용을 더 잘 반영할 수 있도록 합니다.

- **Performance Highlights**: 금융과 의학 도메인에서의 벤치마크 데이터셋을 이용한 평가에서, 제안된 모델은 다양한 어휘 및 신경망 모델을 능가하는 성능을 나타냈습니다. 이로써 수량 중심의 쿼리에 대한 검색 성능을 크게 향상시켰습니다.



### Towards Robust Recommendation via Decision Boundary-aware Graph Contrastive Learning (https://arxiv.org/abs/2407.10184)
Comments:
          KDD 2024

- **What's New**: 최근 그래프 컨트라스티브 학습(GCL)이 추천 시스템에서 데이터 희소성으로 인한 편향을 줄이는 데 효과적이라는 이유로 주목받고 있습니다. 그러나 대부분의 기존 GCL 모델은 휴리스틱 접근 방식에 의존하며, 대조 뷰를 구성할 때 엔터티 독립성을 가정합니다. 우리는 이러한 방법들이 훈련 과정 동안 의미의 불변성과 뷰의 난이도 사이의 균형을 맞추기 어려워한다고 주장합니다. 이를 해결하기 위해, RGCL이라는 새로운 GCL 기반 추천 프레임워크를 제안합니다. RGCL은 대조 쌍의 의미 불변성을 효과적으로 유지하고, 모델 능력이 발전함에 따라 동적으로 적응합니다.

- **Technical Details**: RGCL은 먼저 의사 경계 인지적 대적 교란을 도입하여 대조적으로 확장된 뷰의 탐색 공간을 제한하여 작업 관련 정보의 감소를 방지합니다. 또한, 사용자-사용자 및 항목-항목 간의 글로벌 협력 관계를 고려하여 견고한 대조 뷰를 생성하기 위해 대적-비교 학습 목표를 제안합니다. 마지막으로, 최대 교란을 기반으로 하는 대적 예제를 도입하여 마진 극대화를 달성하고, 데이터 포인트와 결정 경계 간의 거리를 증가시켜 모델의 견고성을 향상시킵니다.

- **Performance Highlights**: 다섯 개의 공개 데이터 세트를 사용한 광범위한 실험을 통해, RGCL이 열두 개의 기준 모델에 비해 우수한 성능을 보인다는 것을 입증했습니다.



### Warming Up Cold-Start CTR Prediction by Learning Item-Specific Feature Interactions (https://arxiv.org/abs/2407.10112)
Comments:
          KDD 2024

- **What's New**: 페이퍼 'EmerG'는 새롭게 등장하는 항목의 클릭 스루 레이트(Click-Through Rate, CTR) 예측을 더욱 정확하게 하기 위해 고안된 신기술을 소개합니다. 새 항목은 초기에는 데이터가 부족해 기존의 글로벌 방법론으로는 예측이 어려웠으나, EmerG는 항목별로 특정한 특징 상호작용 패턴을 학습하여 이를 극복합니다.

- **Technical Details**: EmerG는 하이퍼네트워크(hypernetworks)를 사용하여 항목별 특징 그래프를 생성하고, 이 그래프를 그래프 신경망(Graph Neural Network, GNN)을 통해 처리합니다. 이 GNN은 맞춤형 메시지 전달 메커니즘을 통해 고차원 특징 상호작용을 포착하는데 특화되었습니다. 또한, 이 모델은 메타 학습 전략을 채택하여 다양한 항목의 CTR 예측 작업에서 하이퍼네트워크와 GNN의 파라미터를 최적화하며, 각 작업 내에서 최소한의 항목별 파라미터만 조정합니다.

- **Performance Highlights**: 벤치마크 데이터셋에 대한 광범위한 실험 결과, EmerG는 새 항목에 대한 CTR 예측에서 일관되게 가장 좋은 성능을 보였습니다. 데이터가 전혀 없거나 일부만 있는 경우, 그리고 충분한 데이터가 있는 경우 모두에서 탁월한 성과를 나타냈습니다.



### All Roads Lead to Rome: Unveiling the Trajectory of Recommender Systems Across the LLM Era (https://arxiv.org/abs/2407.10081)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)의 출현이 추천 시스템을 어떻게 재정의할 수 있는지에 대한 새로운 시각을 제공합니다. 특히, 현대 추천 시스템의 두 가지 진화 경로—목록형 추천(list-wise recommendation)과 대화형 추천(conversational recommendation)—에 대해 설명하고, 이들 경로가 LLM 에이전트로 수렴되는 과정을 탐구합니다.

- **Technical Details**: 논문은 언어 기반 모델이 추천 시스템에 어떻게 적용되고 있는지에 대한 기술적 진보를 포괄적으로 개관합니다. 전통적인 목록형 추천에서 LLM으로 강화된 추천, 그리고 대화형 추천 시스템으로의 진화를 면밀히 조사하고, 각 경로에서의 기술적 특징, 연구 방법론, 내재된 과제들을 분석합니다. 또한, LLM이 사용자 모델링, 항목 이해, 결과 설명, 대화 생성 및 파이프라인 조정 등 다양한 추천 연구 분야에 미치는 영향에 대해 논의합니다.

- **Performance Highlights**: LLM으로 강화된 추천 시스템은 장기 기억, 반영, 도구 지능 능력을 지니며, 정보의 효과성을 높이고 사용자 취득 비용을 낮추는 데 기여합니다. 이러한 시스템은 기존의 간단한 사용자 피드백을 넘어 보다 자연스럽고 포괄적인 사용자 의도를 포착할 수 있습니다.



### Semantic Understanding and Data Imputation using Large Language Model to Accelerate Recommendation System (https://arxiv.org/abs/2407.10078)
- **What's New**: 이번 연구는 추천 시스템에서 드문 데이터와 누락된 데이터를 해결하기 위해 대형 언어 모델(Large Language Model, LLM)을 활용한 새로운 데이터 보완 접근법을 제안합니다. 기존 방법들은 복잡한 관계를 포착하는 데 어려움이 있지만, LLM은 방대한 텍스트 기반의 훈련을 통해 이러한 복잡한 관계를 이해하고 지능적으로 누락된 정보를 채울 수 있습니다. 제안된 방법은 데이터 겨냥 인공지능 모델에서 개인화된 추천의 정확성을 크게 향상시킵니다.

- **Technical Details**: LLM 기반 데이터 보완 방법은 LoRA(Low-Rank Adaptation) 기술을 이용해 효율적인 모델 미세 조정(fine-tuning)을 수행합니다. 미세 조정된 LLM은 주어진 데이터 프롬프트를 바탕으로 누락된 값을 예측합니다. 예를 들어, UserId=11, MovieId=22, Genres=‘NaN’(결측치), Rating=4.5인 데이터 항목이 주어지면 프롬프트는 ‘UserId는 11, MovieId는 22, Rating은 4.5입니다. Genres는 무엇인가요?’와 같이 구성됩니다. 이후 LLM이 예측한 값으로 NaN을 대체합니다.

- **Performance Highlights**: 다양한 추천 시스템 과제에서 LLM 기반 보완 방법의 효과를 평가한 결과, 단일 분류(single classification), 다중 분류(multi-classification), 회귀(regression) 작업 모든 영역에서 기존 통계법보다 뛰어난 성능을 보였습니다. 실험에서 정밀도(precision), 재현율(recall), F1-score, R@k(Recall at k), N@k(Normalized Discounted Cumulative Gain at k), MAE(Mean Absolute Error), MSE(Mean Squared Error), RMSE(Root Mean Squared Error) 등의 성능 지표를 활용해 LLM 데이터 보완의 장점을 입증했습니다.



### Correlating Power Outage Spread with Infrastructure Interdependencies During Hurricanes (https://arxiv.org/abs/2407.09962)
Comments:
          IEEE 25th International Conference on Information Reuse and Integration for Data Science (IEEE IRI-2024)

- **What's New**: 이번 연구에서는 허리케인과 같은 극단적인 날씨 이벤트 동안 정전의 확산에 대한 네트워크 분석을 통해 중요 인프라의 상호연관성이 정전 발생에 미치는 영향을 조사했습니다. 분석 결과, 초기 영향 지역으로부터 k-hop 거리 이내에 접근 가능한 중요한 인프라 구성 요소의 범위와 넓은 지역에서 정전 발생 간에 일관된 양의 상관관계가 있음을 발견했습니다.

- **Technical Details**: 연구는 주로 세 가지 데이터셋을 활용했습니다: hurricanemapping.com의 허리케인 예측 바람 범위 데이터, NAERM-IA의 미국 내 다양한 중요 인프라 간의 상호 의존성 그래프 데이터, 그리고 ORNL의 EAGLE-I 시스템에서 제공하는 역사적 정전 데이터입니다. 이를 통해, 네트워크 분석을 사용해 허리케인 동안 정전의 확산과 관련된 상관을 분석했습니다. 특히, k-hop 거리 내에서 복수 단계를 통해 간접적으로 연결된 인프라 구성 요소들을 파악함으로써 영향 받은 지역을 예측했습니다.

- **Performance Highlights**: 허리케인 Ida와 Ian의 데이터 분석 결과, 초기 영향 지역으로부터 k-hop 거리 이내에 접근 가능한 중요 인프라 구성 요소의 범위와 정전 발생 사이에 높은 상관관계(0.6 이상)가 있음을 확인했습니다. k 값이 클수록(즉, 더 많은 hop 단계를 포함할수록) 상관관계가 높아지는 경향이 나타났으며, 이는 간접적으로 연결된 구성 요소들이 정전 확산에 중요한 역할을 함을 시사합니다.



### Popular News Always Compete for the User's Attention! POPK: Mitigating Popularity Bias via a Temporal-Counterfactua (https://arxiv.org/abs/2407.09939)
- **What's New**: 이번 연구에서는 뉴스 추천 시스템에서 인기 기사로 인한 편향을 줄이기 위한 새로운 방법인 POPK를 소개합니다. POPK는 시간적 반사실분석(temporal-counterfactual analysis)을 사용하여 인기 뉴스 기사가 사용자에게 미치는 영향을 줄입니다. 이 방법을 통해 특정 시점에서 인기 기사들이 사용자 클릭을 위해 경쟁하는 상황을 가정하고 추천 정확도와 다양성을 향상시키고자 합니다.

- **Technical Details**: POPK의 작동 원리는 다음과 같습니다. 먼저, 기존 모델들이 보이지 않는 방식으로 사용자에게 인기를 끌었던 기사들의 영향을 제거하고, 이것을 부정적 샘플링 과정에 명시적으로 포함시킵니다. 이는 인기 기사들이 항상 사용자 주의를 끌고 있다는 가정 하에 이루어집니다. POPK는 반사실적 추론(counterfactual reasoning)을 시간적 접근법과 결합하여 부정적 샘플 공간을 조정하고, 이를 통해 사용자 관심사의 정확성을 높이고자 합니다.

- **Performance Highlights**: POPK는 일본어, 영어, 노르웨이어의 세 가지 다른 언어 데이터셋에서 실험된 결과, 전통적인 방법보다 뛰어난 성능을 보였습니다. POPK는 정확도와 다양성을 모두 향상시킬 수 있는 유연성을 제공하며, 인기의 측정 방법도 다양합니다. 실험을 통해 POPK는 추천 기사들의 정확성과 다양성을 효과적으로 향상시키며, 특정 요구 사항에 맞도록 쉽게 맞춤 설정할 수 있음을 확인했습니다.



### SocialRec: User Activity Based Post Weighted Dynamic Personalized Post Recommendation System in Social Media (https://arxiv.org/abs/2407.09747)
Comments:
          This research paper has been accepted in the Social Media Sway: Unraveling the Impact of Social Media on Human Behavior - SMS workshop, to be held in conjunction with the International Conference on Social Networks Analysis and Mining (ASONAM 2024) and will be published in Springer

- **What's New**: 이 논문은 사용자의 소셜 미디어 활동과 프로필 데이터를 통합하여 추천 점수를 평가하는 시스템을 제안합니다. 이 시스템은 사용자의 게시물, 참여 활동 이력, 인구통계 데이터를 활용하여 추천 성능을 최적화합니다.

- **Technical Details**: 이 논문은 사용자 이력과 인구통계 데이터를 이용한 가중치를 동적으로 계산합니다. 게시물 카테고리에 따라 사용자의 관심도를 반영하며, 가중치 계산에는 Matrix Factorization과 Neural Network Matrix Factorization 접근 방식을 사용합니다. Collaborative Filtering을 도입하여 유사한 사용자 간의 유사성 점수를 통해 랭킹을 매깁니다.

- **Performance Highlights**: Hit Rate (HR)와 Normalized Discounted Cumulative Gain (NDCG)를 사용하여 추천 시스템의 성능을 평가했으며, NeuMF 모델에서는 각각 0.80과 0.6의 높은 성능을 달성했습니다.



### BiasScanner: Automatic Detection and Classification of News Bias to Strengthen Democracy (https://arxiv.org/abs/2407.10829)
Comments:
          10 pages, 3 figures, 1 table

- **What's New**: BiasScanner는 온라인 뉴스 기사에서 편향된 문장을 식별하는 대규모 언어 모델을 활용한 새로운 애플리케이션입니다. 뉴스 소비자들에게 보다 균형 잡힌 정보를 제공하기 위해 개발되었으며, 현재 20여 가지 이상의 언론 편향 유형을 문장 단위로 식별하고 분류할 수 있습니다. BiasScanner는 웹 브라우저 플러그인 형태로 제공되며, 사용자의 프라이버시를 존중하는 방식으로 구현됐습니다.

- **Technical Details**: BiasScanner는 OpenAI의 GPT-3.5-turbo-16k 모델을 활용하며, BABE 데이터셋을 기반으로 편향 유형과 강도 정보를 추가하여 학습시킨 모델입니다. 서버 측에서는 REST API를 사용하여 GPT-3.5 모델에 액세스하며, 사용자의 개인 식별 정보를 저장하지 않습니다. 프런트엔드 애플리케이션은 JavaScript로 구현되었으며, Mozilla의 가독성 라이브러리를 사용해 웹 페이지의 관련 텍스트를 추출합니다.

- **Performance Highlights**: BiasScanner는 뉴스 기사에서 편향된 문장을 강조 표시할 뿐만 아니라, 각 분류 결정에 대한 설명과 기사 전체에 대한 요약 분석을 제공합니다. 편향 레포트는 기사 내의 편향 문장의 비율과 평균 편향 점수를 기준으로 계산된 점수를 포함합니다. 사용자는 또한 연구 목적으로 편향 레포트를 기부할 수 있습니다.



### NTSEBENCH: Cognitive Reasoning Benchmark for Vision Language Models (https://arxiv.org/abs/2407.10380)
Comments:
          15 pages, 2 figures, 5 tables

- **What's New**: 새로운 데이터셋 NTSEBench가 소개되었습니다. 이 데이터셋은 인도의 NTSE(National Talent Search Examination) 시험에서 추출한 26개의 문제 카테고리와 4,642장의 이미지를 포함한 2,728개의 다중 선택 질문으로 구성돼 있습니다. NTSEBench는 대형 모델(LLMs, VLMs)의 인지적 다중 모드(reasoning) 및 문제 해결 능력을 평가하기 위해 설계되었습니다.

- **Technical Details**: NTSEBench는 텍스트와 이미지 형식을 모두 포함한 문제들을 다룹니다. 데이터셋은 주로 NTSE 시험지와 관련 학습 자료에서 추출하여 구성되었으며, MathPix OCR, docxlatex, PyMuPDF 등의 라이브러리를 사용하여 텍스트와 이미지를 추출했습니다. 데이터셋은 26개의 문제 카테고리와 8개의 다른 조합의 모달리티 유형을 포함하고 있습니다.

- **Performance Highlights**: 최신 LLMs 및 VLMs를 사용하여 NTSEBench의 기준선을 설정했습니다. 다양한 모달리티 입력을 처리하기 위해 네 가지 독립된 모델링 전략을 제안했으며, 여러 오픈 소스 및 독점 모델들의 성능을 비교 평가했습니다. 이를 통해 모델들이 인지적 문제 해결 능력에서 어떤 점에서 부족한지를 분석했습니다.



### GenSco: Can Question Decomposition based Passage Alignment improve Question Answering? (https://arxiv.org/abs/2407.10245)
- **What's New**: 본 연구에서는 다중 단계 질문 답변(MHQA)을 위한 혁신적인 접근 방식인 'GenSco'를 소개합니다. 이 접근 방식은 질문 분해를 기반으로 선택된 passage sequence가 답변 생성에 어떻게 더 나은 결과를 가져오는지를 탐구합니다. GenSco는 두 가지 서로 다른 대형 언어 모델(LLM)을 사용하여, 생성기 LLM은 질문 분해와 최종 답변 생성을 하고, 보조 오픈소스 LLM은 생성기를 안내하는 scorer 역할을 합니다.

- **Technical Details**: GenSco는 다중 단계 질문을 분해하여 생성된 하위 질문에 기반한 passage들 간의 negative log-likelihood를 이용해 passage를 선택하는 방식입니다. 이 방법은 초기 context가 빈 상태에서 시작하여, 생성기 LLM을 사용해 하위 질문을 생성하고, scorer LLM을 통해 passage를 평가하여 가장 높은 점수를 받은 passage를 context에 추가하는 과정을 반복합니다. 마지막으로 누적된 context와 원래 질문을 생성기 LLM에 전달하여 최종 답변을 생성합니다.

- **Performance Highlights**: GenSco는 2WikiMultiHop와 MuSiQue 데이터셋에서 각각 Exact Match 점수에서 15.1점과 5.9점의 절대적인 이득을 보였습니다. 이는 기존 최상 성능 기법 대비 뛰어난 성능을 보여줍니다. 또한 GenSco는 passage 검색 작업에서 높은 정밀도를 달성하여 LLM 응답의 환상을 효과적으로 줄입니다.



### Harnessing Feature Clustering For Enhanced Anomaly Detection With Variational Autoencoder And Dynamic Threshold (https://arxiv.org/abs/2407.10042)
Comments:
          This work was presented at the 2024 IEEE International Geoscience and Remote Sensing Symposium, IGARSS 2024, 07-12 July 2024, Athens, Greece

- **What's New**: 이번 연구에서는 다변수 시계열 데이터(multivariate time series data)를 이용한 이상 탐지 방법을 제안하여 북극의 눈 녹음(snowmelt)과 같은 극단적인 기후 사건을 탐지하는 방법을 제시했습니다. Variational Autoencoder (VAE)와 동적 임계값 설정(dynamic thresholding), 상관관계 기반 특징 클러스터링(correlation-based feature clustering)을 통합한 프레임워크를 도입하여, 기후 데이터의 시간적 관계를 학습하고 지역적 의존성을 식별하는 능력을 강화했습니다.

- **Technical Details**: 이 연구에서 제안한 방법은 다음을 포함합니다: 1) VAE를 이용한 클러스터 기반 이상 탐지 프레임워크(Cluster-VAE), 2) 기후 속성(예: 계절성)을 특징으로 하는 동적 임계값 알고리즘을 설계하여 이상 현상을 지역적으로 탐지, 그리고 3) 이상 기간 및 특징을 식별할 수 있는 실용적인 도구 제공. 제안된 방법은 Pearson의 상관 계수를 사용하여 MTS의 상관 점수를 구하고, 특징 간의 관계를 클러스터링하여 모델의 표현 능력을 개선하였습니다.

- **Performance Highlights**: 제안된 방법론은 벤치마크 데이터셋에서 더 높은 F1-score로 이상 탐지 정확도를 입증하였습니다. 이 연구는 기후 연구에서 중요한 지역의 이상현상을 설명하는 능력을 제공합니다.



### Causality extraction from medical text using Large Language Models (LLMs) (https://arxiv.org/abs/2407.10020)
- **What's New**: 이 연구는 자연어 모델, 특히 대형 언어 모델(Large Language Models, LLMs)을 사용하여 임상 실습 가이드라인(Clinical Practice Guidelines, CPGs)에서 인과 관계를 추출하는 가능성을 탐구합니다. 특히 임신성 당뇨병 관련 CPGs에서의 인과 관계 추출 결과를 처음으로 제시합니다. 이 연구는 BioBERT, DistilBERT 및 일반 BERT와 같은 다양한 BERT 변형들과 GPT-4, LLAMA2와 같은 LLM들을 사용한 실험 결과를 보고합니다.

- **Technical Details**: BERT(Devlin et al., 2018)는 입력 요소 간의 연결을 기반으로 출력의 각 부분 간에 가중치를 동적으로 조정하여 인과 관계 추출(Causality Extraction) 등의 여러 자연어 처리 작업에 탁월한 성능을 보여줘 왔습니다. 최근에는 GPT-4(OpenAI, 2023)와 같은 LLMs가 등장하여, 광범위한 데이터로 사전 학습을 한 후 인류의 원칙과 정책 준수를 보장하기 위해 인간과 AI의 피드백을 통해 강화 학습을 진행합니다. LLAMA2(Touvron et al., 2023)도 최근 주목받는 모델로, 2조 개의 토큰으로 학습되었으며 정보 추출 작업에서 널리 사용됩니다.

- **Performance Highlights**: 실험 결과, BioBERT가 다른 모델들보다 뛰어난 성능을 보였으며, 평균 F1 점수는 0.72로 기록되었습니다. GPT-4와 LLAMA2도 유사한 성능을 보였으나 일관성은 낮았습니다. BERT 변형 모델들은 fine-tuning이 쉽고 일관된 성능을 보여 여전히 선호될 수 있음을 확인했습니다. LLAMA2는 몇 가지 데이터에 대해 예측을 생성하지 못했지만 예측한 부분에서는 평균 F1 점수 76%를 기록하며 가능성을 보여주었습니다.



### EVOLVE: Predicting User Evolution and Network Dynamics in Social Media Using Fine-Tuned GPT-like Mod (https://arxiv.org/abs/2407.09691)
Comments:
          This article has been accepted as a long paper in the MSNDS 2024 workshop, to be held in conjunction with the International Conference on Social Networks Analysis and Mining (ASONAM 2024), September 2-5, 2024. and will be published in Springer

- **What's New**: 이 논문은 소셜 미디어 사용자의 진화를 예측하는 새로운 방법을 제안합니다. E-GPT(진화-GPT)라 명명한 GPT-like 디코더 기반 모델을 통해 사용자의 소셜 미디어 진화의 다음 단계를 예측하고자 합니다.

- **Technical Details**: 사용자의 네트워크, 인구통계 데이터, 역사, 그리고 참여도를 종합적으로 분석하여 소셜 네트워크에서의 다음 사용자 진화 단계를 예측합니다. GPT의 디코더 부분을 활용하여 시퀀스 예측 기능을 통해 사용자의 미래 상태를 예측합니다.

- **Performance Highlights**: 이를 통해 네트워크 변화 및 활동 패턴 예측을 실시하며, 추천 시스템 등의 소셜 미디어 문제를 해결하는데 기여할 수 있는 결과를 도출하였습니다.



### Bridging the Gap Between Information Seeking and Product Search Systems: Q&A Recommendation for E-commerc (https://arxiv.org/abs/2407.09653)
- **What's New**: 새로운 연구에서는 쇼핑 미션 중인 소비자가 제품 검색과 정보 탐색 시스템을 어떻게 번갈아 사용하는지를 분석하고, 이를 통합한 Q&A 추천 시스템을 제안합니다. 이 시스템은 Q&A 쌍을 제공하여 사용자가 구매 결정을 내리는 데 도움을 주며, 기존의 불편한 전환 과정을 최소화합니다.

- **Technical Details**: 연구팀은 Q&A 추천 시스템의 요구사항, 질문과 답변의 특성, 생성 방법 및 추천 작업의 최적화 문제를 다루고 있습니다. LLMs (Large Language Models)을 사용하여 Q&A 쌍을 생성하고, 이를 자동완성, 검색 결과 페이지, 제품 상세 페이지 등 다양한 쇼핑 단계에서 제안합니다.

- **Performance Highlights**: 이 시스템은 사용자가 쇼핑 여정 중 각 단계에서 필요로 하는 정보를 적시에 제공함으로써 특히 탐색, 비교, 최종 고려 단계에서 큰 효율성을 보입니다. 적절한 질문과 답변을 통해 사용자의 필요를 충족시키고, 구매 결정을 더욱 쉽게 도와줍니다.



### Learning Outcomes, Assessment, and Evaluation in Educational Recommender Systems: A Systematic Review (https://arxiv.org/abs/2407.09500)
- **What's New**: 이 연구에서는 교육 추천 시스템 (Educational Recommender Systems, ERS)에서 학습이 어떻게 측정되고 최적화되는지를 분석합니다. 기존 연구에서 주로 사용되는 타겟 메트릭 및 평가 방법을 조사하며, 특히 추천의 교육적 효과에 중점을 둡니다. 총 1,395개의 관련 논문을 검토하고 포함 및 제외 기준을 통해 28개의 논문을 최종 분석했습니다.

- **Technical Details**: ERS는 개인화된 교육 자료를 추천하여 학습을 촉진하는 것을 목표로 합니다. 추천 시스템(RS)에는 협업 필터링(Collaborative Filtering, CF), 콘텐츠 기반 필터링(Content-based Filtering, CBF), 지식 기반 필터링(Knowledge-based Filtering, KB) 및 이들 방법의 하이브리드 모델이 사용됩니다. 협업 필터링은 사용자 간의 유사성을 기반으로 추천하며, 콘텐츠 기반 필터링은 항목의 메타데이터를 사용하여 추천합니다. 지식 기반 필터링은 사용자 요구와 항목 사이의 관계를 명시적으로 모델링하여 추천합니다.

- **Performance Highlights**: 분석된 논문 중 50% 미만이 학습 기반 메트릭을 최적화하고 있으며, 교육적 효과 측정을 위해 성과 기반 평가를 사용한 논문은 1/3에 불과합니다. 이러한 결과는 ERS 연구에서 대규모 및 비형식 교육 환경에서의 교육적 효과를 평가하는데 격차가 있음을 시사합니다.



### Leveraging Large Language Models for Relevance Judgments in Legal Case Retrieva (https://arxiv.org/abs/2403.18405)
- **What's New**: 본 연구는 대규모 언어 모델(Large Language Models, LLMs)을 활용하여 법률 사례 검색에서의 유사 판단(legal relevance judgment)을 자동화하는 새로운 방법을 제시합니다. 특히, GPT-3.5와 같은 일반적인 LLM을 사용하여 법적 관련성을 평가하고 그 정확성을 전문가의 판정과 비교했습니다. 그 결과, 제안된 워크플로우는 인간 전문가와 높은 일관성을 보이며 기존 법률 사례 검색 모델을 향상시킬 수 있는 합성 데이터를 생성하는데 활용되었습니다.

- **Technical Details**: 이 새로운 방법은 고급 LLM의 이해와 추론 능력을 향상시키기 위해 몇 가지 단계를 통해 전문가의 추론 과정을 세밀하게 지시합니다. 또한, 길이가 긴 텍스트 문제를 해결하기 위해 주어진 법률 사례를 작은 단위로 분해하여 처리하고, 전문 지식이 필요한 부분들을 효과적으로 추출하고 설명합니다. 데이터 수집 면에서는 언레이블된 법률 사례 그룹에서 가능한 긍정적인 사례 쌍을 효율적으로 수집하는 전략도 포함되었습니다.

- **Performance Highlights**: 실험 결과, 이 방법은 전문가 라벨과 높은 수준의 일관성을 보였습니다. 또한, 생성된 합성 데이터를 사용한 모델 파인튜닝 결과, 법률 사례 검색 작업에서 성능이 크게 향상된 것으로 나타났습니다. 이는 제안된 접근법의 유효성을 간접적으로 검증하는 결과입니다.




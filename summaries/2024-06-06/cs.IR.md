### Text-like Encoding of Collaborative Information in Large Language Models for Recommendation (https://arxiv.org/abs/2406.03210)
Comments:
          Accepted by ACL 2024

- **What's New**: 이번 연구에서는 기존 협업 임베딩(colla보araive embeddings)을 텍스트 유사 형식으로 변환하여 LLMRec에 원할하게 통합할 수 있는 BinLLM을 소개합니다. BinLLM은 협업 정보를 직접적으로 활용하기 위해 이진(binary) 시퀀스로 변환하여 대형 언어 모델(LLMs)에서 직접 사용 가능하게 합니다. 또한 이진 시퀀스를 점-10진(dot-decimal) 표기법으로 압축하는 옵션도 제공하여 시퀀스 길이를 줄입니다.

- **Technical Details**: BinLLM의 주요 구성 요소는 두 가지입니다. (1) 협업 모델(colla보araive model): 이 모델은 사용자와 항목의 협업 임베딩을 수치적 잠재 벡터로 생성합니다. (2) 이진화 및 압축 모듈(binarization & compression module): 협업 임베딩을 이진 시퀀스로 변환하거나 더 압축된 형식으로 변환합니다. 협업 임베딩을 이진화하기 위해, 전결합층을 통해 협업 임베딩을 적절한 공간으로 변환한 후, 부호(sign) 함수를 적용하여 이진 결과를 얻습니다.

- **Performance Highlights**: 광범위한 실험을 통해 BinLLM이 LLM에 더 잘 맞춰진 방식으로 협업 정보를 도입하여 성능 향상을 이끌어냄을 확인했습니다. 특히, 이진 시퀀스로 변환된 협업 정보를 이용하여 추천 성능이 크게 향상되었음을 입증했습니다. 우리의 코드는 공개되어 있으며, 추가적인 실험을 통해 접근법의 효과를 더욱 검증할 계획입니다.



### CAPRI-FAIR: Integration of Multi-sided Fairness in Contextual POI Recommendation Framework (https://arxiv.org/abs/2406.03109)
- **What's New**: 이번 연구는 기존 추천 시스템에 공정성을 고려한 포스트 필터링(post-filtering) 방법론을 도입하려는 시도로, 소비자와 제공자 간의 공정성을 개선하고자 합니다. 이를 통해 장소 추천 시스템(Point-of-Interest, POI)에서 제공되는 항목의 노출도(item exposure)와 성능 지표(precision 및 distance)를 동시에 충족시키는 것을 목표로 합니다.

- **Technical Details**: 제안된 모델은 소비자와 제공자의 공정성을 고려해 점수 조정 방식의 여러 매개 변수를 튜닝하여 공정성을 평가합니다. 이를 위해 F_c(u)와 F_p(u) 같은 추상 함수를 사용하여 각각 소비자와 제공자의 공정성을 나타냅니다. 또한, 'CAPRI-FAIR' 프레임워크를 사용해 기존 시스템과의 호환성을 유지하면서 공정성 요소를 추가하는 방식을 적용하였습니다.

- **Performance Highlights**: 실험 결과, 제공자 공정성 요소를 고려한 경우 추천 항목의 재점수화에서는 성능과 롱테일(long-tail) 노출도 간의 가장 좋은 균형을 보여주었습니다. 반면, 비활성 사용자를 위해 더 인기 있는 POI를 추천하므로써 소비자 공정성을 향상시키려는 시도는 일부 추천 모델과 데이터셋에서만 정확도의 향상을 보였습니다. 소비자와 제공자 공정성을 모두 고려한 경우, 이 둘 간의 트레이드오프(trade-off)로 인해 가장 낮은 정확도 값을 얻게 되었습니다.



### A Bi-metric Framework for Fast Similarity Search (https://arxiv.org/abs/2406.02891)
- **What's New**: 새로운 'bi-metric' 프레임워크가 제안되었습니다. 이 프레임워크는 두 가지 비유사도(dissimilarity) 함수를 사용합니다: 정확하지만 계산 비용이 높은 실제 전장거리(ground-truth) 측정 가중치와 저렴하지만 덜 정확한 프록시(proxy) 측정 가중치입니다. 이 프레임워크는 비용 효율적인 방식으로 이 두 가지 함수를 결합하여 근접 이웃 데이터 구조를 설계할 수 있게 합니다.

- **Technical Details**: 본 프레임워크는 두 가지 인기 있는 근접 이웃 검색 알고리즘인 DiskANN과 Cover Tree를 이용해 그 이론적 성능을 입증합니다. 프록시 가중치를 사용하여 데이터 구조를 생성하고, 쿼리 절차에서는 최소한의 실제 전장거리 계산만으로 높은 정확도를 보장할 수 있습니다. 텍스트 검색 문제에 적용한 결과, MTEB 벤치마크의 거의 모든 데이터 세트에서 기존 방법들보다 훨씬 더 우수한 정확도-효율성 절충(tradeoff)을 달성했습니다.

- **Performance Highlights**: 제안된 프레임워크는 DiskANN 및 Cover Tree 알고리즘에 적용하여, 프록시 가중치가 실제 전장거리를 일정 변수로 근사할 수 있다면 임의의 정확도 보장이 가능합니다. 또한, 실험 결과, 이 프레임워크는 기존의 재배열(re-ranking) 접근법에 비해 효율성과 정확도 모두에서 뛰어난 성능을 보였습니다.



### Exploring User Retrieval Integration towards Large Language Models for Cross-Domain Sequential Recommendation (https://arxiv.org/abs/2406.03085)
Comments:
          10 pages, 5 figures

- **What's New**: 새로운 프레임워크인 URLLM이 제안되어 Cross-Domain Sequential Recommendation(CDSR) 문제를 해결하고자 합니다. URLLM은 사용자 검색 접근법과 도메인 구체화(domain grounding)를 통해 대형 언어 모델(LLM)의 성능을 극대화하여, 사용자의 순차적 선호도를 다른 도메인으로 이전해 주는 기능을 효과적으로 수행합니다.

- **Technical Details**: URLLM은 이중 그래프 시퀀스 모델(dual-graph sequence model)을 도입하여 협업(collaborative) 및 구조적-의미적 정보(structural-semantic information)를 캡처합니다. 그런 다음, KNN 사용자 검색기가 LLM에 관련된 사용자 정보를 검색하여 통합합니다. 이와 함께, 도메인 구별 전략과 정제 메커니즘(refinement module)을 통해 입력 및 생성된 응답의 도메인 특정 성격을 유지합니다.

- **Performance Highlights**: Amazon에서 수행한 광범위한 실험에서 URLLM은 최첨단 기준모델들과 비교하여 정보 통합과 도메인 특정 생성 능력에서 우수한 성능을 보여주었습니다. 실험 결과, 통합된 정보의 종류와 모델 성능 간에 긍정적인 상관관계가 확인되었으며, 검색된 사용자의 적중률(hit rate)과 모델 성능 간에도 긍정적인 상관관계가 나타났습니다.



### Path-Specific Causal Reasoning for Fairness-aware Cognitive Diagnosis (https://arxiv.org/abs/2406.03064)
Comments:
          Accpeted by KDD'2024

- **What's New**: 이번 연구에서는 인공지능 교육에서 매우 중요한 역할을 하는 인지 진단(Cognitive Diagnosis)의 공정성과 성능을 동시에 유지하기 위해 새로운 경로 특정 인과 추론 프레임워크(Path-Specific Causal Reasoning Framework, PSCRF)를 제안합니다. 기존 방법들이 민감한 학생 정보를 잘못 다루어 불공정한 결과를 초래할 수 있는 문제를 해결하고자 합니다.

- **Technical Details**: 제안된 PSCRF는 인코더를 사용하여 학생-문제 상호작용 로그에서 특징을 추출하고, 학생의 일반 정보와 민감한 정보에 대한 임베딩을 생성합니다. 새로운 속성 지향 예측기(attribute-oriented predictor)인 Decoupled Predictor(DP)를 설계하여 민감한 속성을 분리하고, 공정성과 관련된 민감한 특징 임베딩을 제거하며 유용한 정보를 유지합니다. 다양한 제약 조건을 적용하여 공정성과 진단 성능을 동시에 향상시키는 다중 요인 제약(multi-factor constraint)을 도입하였습니다.

- **Performance Highlights**: 실제 세계 데이터셋(PISA 등)에서 수행된 광범위한 실험 결과, 제안된 PSCRF는 공정성을 유지함과 동시에 학생 숙련도 모델링의 정확성을 유지하는 데 매우 효과적임을 입증했습니다. 코드는 커뮤니티의 연구 발전을 위해 공개되었습니다.



### Know Your Neighborhood: General and Zero-Shot Capable Binary Function Search Powered by Call Graphlets (https://arxiv.org/abs/2406.02606)
- **What's New**: 이 논문은 악성코드 분석, 취약점 연구, 표절 탐지 등 여러 분야에서 중요한 문제인 바이너리 코드 유사성 탐지(binary code similarity detection)를 다룹니다. 새로운 그래프 데이터 표현 방식인 '콜 그래플릿(call graphlets)'과 새로운 그래프 신경망 아키텍처(graph neural network architecture)를 결합한 접근 방식을 제안합니다. 콜 그래플릿은 바이너리 실행 파일 내 각 함수 주변의 네이버후드를 인코딩하여 통계적 특성을 통해 로컬 및 글로벌 컨텍스트를 캡처합니다.

- **Technical Details**: 콜 그래플릿(call graphlets)은 함수의 종속성과 호출자를 포함하여 구조를 제공하고, 간단한 함수 수준의 특성을 노드 속성으로 사용하는 그래프 데이터 형식을 도입합니다. 제안된 그래프 신경망 모델은 이러한 그래프 표현에서 작동하며, 심층 메트릭 학습(deep metric learning)을 사용하여 이를 기능 벡터로 매핑하여 의미 있는 코드 유사성을 인코딩하도록 학습됩니다. 실험은 다양한 아키텍처, 컴파일러 도구 체인, 최적화 수준을 다루는 네 가지 데이터셋에서 수행되었습니다.

- **Performance Highlights**: 실험 결과, 콜 그래플릿과 새로운 그래프 신경망 아키텍처의 조합은 단일 아키텍처 및 다중 아키텍처 환경에서 최첨단 성능을 달성했음을 보여줍니다. 또한, 제안된 접근 방식은 도메인 외 평가(out-of-domain evaluation)에서도 우수한 성능을 보였습니다. 이는 다양한 소프트웨어 생태계에서 광범위하게 적용 가능한 일반적이고 효과적인 그래프 신경망 기반 솔루션을 제공합니다.



### Pairwise Ranking Loss for Multi-Task Learning in Recommender Systems (https://arxiv.org/abs/2406.02163)
- **What's New**: 이 논문은 다중 작업 학습(Multi-Task Learning, MTL)에서 광고 시스템의 클릭률(Click-Through Rate, CTR) 및 전환율(Conversion Rate, CVR)을 최적화하는 새로운 손실 함수(PWiseR)를 제안합니다. 이 손실 함수는 모델 예측 간 	extbf{p}air	extbf{wise} 	extbf{r}anking을 계산하여, 전환이 발생한 노출에 더 많은 가중치를 부여합니다. 기존 방법들이 클릭과 전환 간의 인과 관계를 명확히 하지 못한 부분을 해결하고 있습니다.

- **Technical Details**: CTR와 CVR은 광고 시스템에서 중요한 두 가지 측정 지표입니다. 기존의 MTL 모델들은 이 두 가지를 별도로 최적화하지만, 이 논문은 전환이 클릭 뒤에 발생하는 인과 관계를 고려한 새로운 손실 함수(PWiseR)를 제안합니다. PWiseR 손실 함수는 전환이 발생한 샘플에 더 높은 점수를 부여함으로써 잡음을 줄이고 모델의 정확성을 높입니다. 이를 통해 광고 추천 시스템의 성능을 향상시킬 수 있습니다.

- **Performance Highlights**: 제안된 PWiseR 손실 함수는 네 가지 공개 MTL 데이터셋(Alibaba FR, NL, US, CCP)과 산업 데이터셋에서 기존의 BCE 손실 함수보다 더 나은 AUC(Area Under Curve) 성능을 보였습니다. 특히 전환율이 높은 샘플에 더 높은 점수를 할당함으로써 모델의 잡음에 대한 강건성을 향상시켰습니다.



### Auto-Encoding or Auto-Regression? A Reality Check on Causality of Self-Attention-Based Sequential Recommenders (https://arxiv.org/abs/2406.02048)
- **What's New**: 이번 논문에서는 Auto-Encoding(AE)와 Auto-Regression(AR) 모델의 성능 비교를 통해, AR 모델이 AE 모델을 초과하는 경우를 체계적으로 재평가하였습니다. SASRec과 BERT4Rec의 구현 세부 사항을 신중히 검토한 결과, 평가 지표 및 손실 함수 선택이 성능에 중요한 영향을 미친다는 것을 발견했습니다. 또한, 커스터마이징된 디자인 스페이스를 활용하여 AR 모델이 AE 모델보다 우수한 성능을 보인다는 결론을 내렸습니다.

- **Technical Details**: SASRec(AR 기반)와 BERT4Rec(AE 기반)을 재평가하기 위해 통제된 환경에서 일련의 실험을 수행했습니다. 평가 지표와 손실 함수 선택의 중요성을 밝혀냈으며, 다양한 특징과 모델링 접근 방식을 포함한 통합 평가 환경(ModSAR)을 제공했습니다. 또한, Sequential Recommendation Tasks에서의 AE/AR 성능 차이를 낮은 차수 근사(Low-Rank Approximation)와 유도 편향(Inductive Bias) 관점에서 분석했습니다.

- **Performance Highlights**: AR 모델이 AE 모델을 능가하는 성능을 보였으며, 이 결과는 SASRec과 BERT4Rec만이 아닌 광범위한 자가 주의(Self-Attentive) 모델 변형에도 적용되었습니다. 평가 환경을 통일함으로써 AR 모델이 주요 추천 시나리오에서 일반적으로 더 나은 성능을 제공함을 입증했습니다.



### Session Context Embedding for Intent Understanding in Product Search (https://arxiv.org/abs/2406.01702)
Comments:
          5 pages, 1 Figure, 5 Tables, SIGIR 2024, LLM for Individuals, Groups, and Society

- **What's New**: 기존의 검색 훈련 방식이 사용자의 의도를 충분히 반영하지 못한다는 문제를 해결하고자, 검색 세션 동안의 일련의 참여 행위(예: 클릭, 장바구니 담기(ATC), 주문)를 활용한 세션 컨텍스트 벡터화 방법(session embedding)을 제안합니다. 실시간(session)에서는 쿼리 임베딩(query embedding)의 대안으로 세션 임베딩을 사용하여 검색과 재순위화에 활용할 수 있습니다. 이 방법을 통해 사용자 의도 이해도를 개선하는 데 성공하였습니다.

- **Technical Details**: 세션 임베딩은 이전 쿼리 및 참여한 항목을 벡터화하는 방식으로, 이를 위해 대형 언어 모델(LLMs)을 사용합니다. 쿼리 임베딩(query embedding)에 세션 컨텍스트(이전 쿼리 및 아이템 참여 정보)를 추가한 상태 벡터(session state vector)가 생성됩니다. 이러한 세션 벡터는 텍스트 입력과 아이템 속성(제목, 성별, 크기, 브랜드, 설명)을 벡터로 변환하며, 이는 분류(classification)를 포함한 다양한 작업에 사용될 수 있습니다. 또한 실시간에서 쿼리와 세션 벡터를 결합하거나 독립적으로 사용할 수 있도록 설계되었습니다. 훈련 단계에서는 경량 언어 모델(예: DeBERTa)을 사용하여 빠르고 효율적인 세션 임베딩을 제공합니다.

- **Performance Highlights**: 세션 데이터를 활용한 훈련과 평가를 통해, 쿼리의 제품 유형 분류(f1 score)에서 상당한 성능 향상을 보여주었습니다. 특히 이전의 쿼리가 넓고 현재의 쿼리가 좁은 경우(sequential broad to narrow queries), 세션 컨텍스트를 포함하면 사용자 의도 이해도에서 큰 성능 향상을 보였습니다.



### An LLM-based Recommender System Environmen (https://arxiv.org/abs/2406.01631)
- **What's New**: 이번 아카이브 논문에서는 강화 학습(RL)을 추천 시스템에 구현하기 위한 새로운 프레임워크인 SUBER를 제안하였습니다. 이 프레임워크는 대형 언어 모델(LLM)의 기능을 활용하여 인간 행동을 시뮬레이션하는 합성 환경을 제공합니다. 특히 영화와 도서 추천 실험을 통해 SUBER의 효과를 입증하였으며, 모든 소프트웨어는 공개 소스로 제공됩니다.

- **Technical Details**: SUBER는 강화 학습 기반의 추천 시스템을 훈련하고 평가하기 위해 가상의 유저 행동을 시뮬레이션하는 환경을 제공합니다. SUBER는 세 가지 모듈(전처리 모듈, LLM 컴포넌트, 후처리 모듈)로 구성되어 있으며, 각 모듈은 개별적으로 변경 및 확장이 가능합니다. 이는 Gymnasium API를 기반으로 구성되어 있어 유연한 설계를 지원합니다. RL 모델은 메모리 모듈에서 사용자를 선택해 행동 관찰을 한 후, 아이템을 추천하고 그 결과를 평가합니다.

- **Performance Highlights**: 실험 결과, SUBER는 영화와 도서 추천 설정에서 효과적인 성능을 보였습니다. 다양한 LLM 구성의 영향을 깊이 있게 분석한 결과, 여러 LLM이 사용자의 선택을 성공적으로 모방할 수 있음을 확인했습니다. 이를 통해 추천 시스템 평가 및 RL 전략 개선에 있어 SUBER의 잠재력을 입증했습니다.



### RecDiff: Diffusion Model for Social Recommendation (https://arxiv.org/abs/2406.01629)
- **What's New**: 이번 연구에서는 추천 시스템의 정확도를 저하시킬 수 있는 불필요한 소셜 연결을 필터링하여 개선된 소셜 추천 프레임워크를 제안합니다. 본 연구에서 제안한 RecDiff는 숨겨진 공간(diffusion-based social denoising framework) 확산 패러다임을 활용해 소셜 연결의 소음을 제거하는 데 주력합니다.

- **Technical Details**: RecDiff는 다단계 노이즈 확산 및 제거 과정을 통해 사용자 임베딩에서 소음을 식별하고 제거하는 능력을 갖추고 있습니다. 이 확산 모듈은 다운스트림 작업에 맞춰 최적화되어 추천 과정을 극대화합니다. RecDiff는 소셜 및 상호작용 그래프에서 구조적 특징을 저차원 임베딩으로 인코딩하여 효율성을 높입니다. 그 후 이 임베딩을 기반으로 확산 기반 소음 제거 과정을 수행하여 다양한 소셜 노이즈를 효과적으로 처리합니다.

- **Performance Highlights**: 제안한 RecDiff 프레임워크는 추천 정확도, 학습 효율성 및 노이즈 제거 효과 측면에서 기존 모델들보다 뛰어남을 보였습니다. 특히, 소셜 정보의 잡음을 효과적으로 제거하여 향상된 사용자 선호도 모델링을 제공합니다.



### FinEmbedDiff: A Cost-Effective Approach of Classifying Financial Documents with Vector Sampling using Multi-modal Embedding Models (https://arxiv.org/abs/2406.01618)
Comments:
          10 pages, 3 figures

- **What's New**: 다중 모달 금융 문서를 정확히 분류하는 것은 중요한 과제입니다. 전통적인 텍스트 기반 접근 방식은 이러한 문서의 복잡한 다중 모달 특성을 포착하는 데 실패하는 경우가 많습니다. 이를 개선하기 위해 FinEmbedDiff라는 비용 효율적인 벡터 샘플링 방법을 제안합니다. FinEmbedDiff는 사전 학습된 다중 모달 임베딩 모델을 활용하여 금융 문서를 분류합니다.

- **Technical Details**: FinEmbedDiff는 문서에 대해 다중 모달 임베딩 벡터를 생성한 후, 벡터 유사성 측정을 사용하여 미리 계산된 클래스 임베딩과 새 문서를 비교합니다. 이 접근 방식은 텍스트, 표, 차트, 이미지 등 다양한 형태의 콘텐츠를 포함하는 문서의 특성을 효과적으로 반영합니다. 이를 통해 문서의 복잡한 속성을 효과적으로 분석합니다.

- **Performance Highlights**: FinEmbedDiff는 대규모 데이터셋에서 평가된 결과, 최신 기준보다 경쟁력 있는 분류 정확도를 달성하면서 계산 비용을 크게 줄였습니다. 또한 이 방법은 강력한 일반화 능력을 보였으며, 실제 금융 애플리케이션에 실용적이고 확장 가능한 해결책을 제공합니다.



### System-2 Recommenders: Disentangling Utility and Engagement in Recommendation Systems via Temporal Point-Processes (https://arxiv.org/abs/2406.01611)
Comments:
          Accepted at FAccT'24

- **What's New**: 최신 연구에서는 사용자 효용(user utility)을 추론하는 기존 추천 시스템 방식 대신 사용자가 플랫폼으로 돌아오는 확률(return probability)을 기반으로 효용을 추론하는 새로운 접근 방식을 제안하고 있습니다. 기존 시스템은 좋아요, 공유, 시청 시간 등의 참여 신호(engagement signals)를 활용해 컨텐츠를 최적화하지만, 이러한 신호는 사용자의 단기적 충동에 의해 영향을 받을 수 있어 진정한 효용을 측정하기 어렵다는 문제가 있습니다.

- **Technical Details**: 연구는 사용자의 과거 컨텐츠 상호작용이 사용자 도착률(arrival rates)에 영향을 미치는 자기 흥분 하우크스 프로세스(Hawkes process)를 기반으로 한 생성 모델(generative model)을 제안합니다. 이 도착률은 시스템-1(System-1)과 시스템-2(System-2) 의사결정 프로세스를 결합합니다. 시스템-1 도착 강도(arrival intensity)는 즉각적 만족에 기반하며 빠르게 사라지지만, 시스템-2 도착 강도는 사용자의 장기적인 효용에 기반한 지속적 영향을 미칩니다.

- **Performance Highlights**: 가상 데이터(synthetic data)를 활용한 실험 결과, 제안된 모델을 통해 관찰된 사용자 상호작용 데이터를 기반으로 시스템-1과 시스템-2 행동을 분리하는 것이 가능하다는 것을 확인했습니다. 또한, 추천을 사용자의 효용을 기반으로 최적화함으로써 참여 기반 시스템보다 사용자 효용을 크게 향상시킬 수 있음을 보여줍니다.



### An Empirical Study of Excitation and Aggregation Design Adaptions in CLIP4Clip for Video-Text Retrieva (https://arxiv.org/abs/2406.01604)
Comments:
          20 pages

- **What's New**: CLIP4Clip 모델을 기반으로 한 새로운 비디오-텍스트 검색 모델을 제안했습니다. 기존 프레임 특징(Features)이 평균 풀링(mean pooling) 방식으로 집계되는 제한점을 극복하고, 비디오 표현 생성을 위한 자극(excitation) 및 집계(aggregation) 모듈을 도입하였습니다.

- **Technical Details**: 제안된 시스템은 두 가지 주요 모듈로 구성됩니다: 1) 자극 모듈(Excitation Module)은 프레임 특징들 사이의 비상호배타적(non-mutually-exclusive) 관계를 포착하고 프레임별 특징 재조정을 수행합니다. 2) 집계 모듈(Aggregation Module)은 독점성(Exclusiveness)을 학습하여 특징을 집계합니다. 또한, 연속형 프레임 표현과 다중 모달 상호작용을 위해 이 모듈들을 순차적으로 결합하여 사용합니다.

- **Performance Highlights**: 제안된 모델은 세 가지 벤치마크 데이터셋인 MSR-VTT, ActivityNet, DiDeMo에서 평가되었습니다. 이 모델은 MSR-VTT에서 43.9 R@1, ActivityNet에서 44.1 R@1, DiDeMo에서 31.0 R@1의 성능을 기록했으며, 각각 CLIP4Clip 결과를 상대적으로 1.2%, 4.5%, 9.5% 초과합니다.



### Privacy-preserving recommender system using the data collaboration analysis for distributed datasets (https://arxiv.org/abs/2406.01603)
- **What's New**: 본 연구에서는 다양한 기관이 보유한 분산 데이터셋을 통합하여 고품질의 추천을 제공하는 개인 정보 보호 추천 시스템을 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 데이터 협력 분석(data collaboration analysis)를 통해 분산 데이터셋의 개인 정보를 보호하면서도 예측 정확성을 향상시킵니다.

- **Technical Details**: 이 연구는 유저-아이템 평점 매트릭스를 플래튼(flattened) 형식으로 변환하여 데이터 협력 분석을 예측 기능에 적용합니다. 이 방법은 수평적, 수직적 데이터셋 통합을 모두 처리할 수 있습니다. 기존의 개인 분석(individual analysis) 및 중앙 집중식 분석(centralized analysis) 방법과 비교하여 제안된 방법의 성능을 평가하기 위해 두 개의 공개 평점 데이터셋을 사용한 수치 실험을 수행했습니다.

- **Performance Highlights**: 실험 결과 제안된 방법은 개인 분석 방법보다 더 높은 예측 정확성을 보여주었으며, 중앙 집중식 분석에 필적하는 성능을 보였습니다. 특히 참여하는 파티의 수가 증가함에 따라 예측 정확도가 향상되는 경향을 확인하였습니다.



### ProGEO: Generating Prompts through Image-Text Contrastive Learning for Visual Geo-localization (https://arxiv.org/abs/2406.01906)
- **What's New**: 이번 연구에서는 Visual Geo-localization (VG) 문제를 해결하기 위해 두 단계의 학습 방식을 제안합니다. 특히 CLIP (Contrastive Language-Image Pretraining) 모델의 멀티모달 특성을 활용하여 모호한 텍스트 설명을 통해 이미지 인코더의 성능을 향상시키는 방법을 소개합니다.

- **Technical Details**: 제안된 ProGEO 모델은 CLIP을 활용한 두 단계의 학습 과정을 통해 이미지 인코더가 더 일반화된 시각적 특징을 학습할 수 있도록 합니다. 첫 번째 단계에서는 학습 가능한 텍스트 프롬프트를 생성하여 지리적 이미지 특징을 모호하게 설명하고, 두 번째 단계에서는 다이나믹 텍스트 프롬프트를 사용하여 이미지 인코더의 학습을 돕습니다. 이를 통해 지역적인 특징의 결함을 보완하고 효과적인 시각적 특징을 추출합니다.

- **Performance Highlights**: 여러 대규모 VG 데이터셋에서 제안된 모델의 유효성을 입증하며, 대부분의 데이터셋에서 경합력 있는 결과를 달성했습니다. Pitts30k와 St Lucia 데이터셋에서의 실험 결과는 제안된 전략의 유효성과 적용 가능성을 확인합니다.



### Large Language Models as Recommender Systems: A Study of Popularity Bias (https://arxiv.org/abs/2406.01285)
Comments:
          Accepted at Gen-IR@SIGIR24 workshop

- **What's New**: 대규모 언어 모델(LLMs)을 추천 시스템에 통합하면서 발생하는 인기도 편향의 문제를 연구한 최신 연구가 발표되었습니다. 연구에서는 LLM 기반 추천 시스템이 인기도 편향을 악화시킬 가능성과 이를 프롬프트 튜닝(prompt tuning)을 통해 완화할 수 있는 기회를 동시에 제시하고 있습니다.

- **Technical Details**: LLM이 추천 시스템에서 인기도 편향을 따르는지 또는 완화할 수 있는지 분석했습니다. 이 연구는 인기도 편향을 측정하기 위한 기존의 지표들을 논의하고, 일련의 원하는 조건을 만족하는 새로운 지표를 제안합니다. 새로운 지표를 기반으로 LLM 기반 추천 시스템과 전통적인 추천 시스템을 영화 추천 작업에 대해 비교했습니다. 결과적으로, LLM 추천 시스템은 명시적인 완화 조치 없이도 덜 인기도 편향을 나타내었습니다.

- **Performance Highlights**: 영화 추천 작업에서, 간단한 LLM 기반 추천 시스템은 전통적인 추천 시스템에 비해 인기도 편향이 적게 나타났습니다. 연구는 또한 프롬프트를 통해 편향을 완화할 수 있는 가능성을 제시하며, 정확도와 인기도-편향 간의 트레이드오프를 연구했습니다.



### Demo: Soccer Information Retrieval via Natural Queries using SoccerRAG (https://arxiv.org/abs/2406.01280)
Comments:
          accepted to CBMI 2024 as a demonstration; this https URL

- **What's New**: SoccerRAG는 Retrieval Augmented Generation (RAG)와 대형 언어 모델(Large Language Models, LLMs)을 활용해 축구 관련 정보를 자연어로 쿼리할 수 있는 혁신적인 프레임워크입니다. 이 시스템은 동적인 쿼리와 자동 데이터 검증을 지원하여, 사용자에게 스포츠 아카이브에 쉽게 접근할 수 있는 인터페이스를 제공합니다. 새로운 상호작용형 사용자 인터페이스(UI)는 Chainlit 프레임워크를 기반으로 구축되어, 사용자들이 챗봇과 비슷한 방식으로 SoccerRAG와 상호작용할 수 있게 합니다.

- **Technical Details**: SoccerRAG 프레임워크는 SoccerNet 데이터를 기반으로 하여 게임 비디오, 이미지 프레임, 오디오, 타임스탬프된 캡션, 경기 이벤트 주석 및 선수 정보를 포함한 확장된 축구 데이터셋을 사용합니다. 사용자는 자연어 쿼리를 통해 원하는 정보를 입력하고, LLM은 쿼리에서 추출된 속성을 기반으로 SQL 쿼리를 생성하여 데이터베이스에서 필요한 정보를 검색합니다. 이 프레임워크는 Python 3.12 이상을 필요로 하며, 명령 줄 인터페이스(CLI)와 Chainlit 기반 UI를 지원합니다.

- **Performance Highlights**: SoccerRAG는 GPU 없이 CPU만으로도 실행 가능하며, 다양한 운영 체제에서 테스트되었습니다. 예를 들어, Windows 11 운영 체제에서 Intel Core i5-9300H CPU @2.40GHz와 16GB 메모리를 사용하여 성공적으로 실행되었습니다. 이 시스템은 OpenAI API 키를 필요로 하며, LangSmith API 키를 통해 각 호출의 비용을 모니터링하고 히스토리를 유지할 수 있습니다. SoccerRAG는 사용자 쿼리에 대한 중간 피드백을 제공하고, 잘못된 정보나 부족한 정보를 보완하기 위해 추가적인 사용자 입력을 요구할 수 있습니다.



### SoccerRAG: Multimodal Soccer Information Retrieval via Natural Queries (https://arxiv.org/abs/2406.01273)
Comments:
          accepted to CBMI 2024 as a regular paper; this https URL

- **What's New**: 디지털 스포츠 미디어의 빠른 발전에 대응하여 SoccerRAG라는 혁신적인 프레임워크가 소개되었습니다. 이 시스템은 Retrieval Augmented Generation (RAG)과 Large Language Models (LLMs)를 활용하여 자연어 쿼리를 통해 축구 관련 정보를 추출하는 데 초점을 맞추고 있습니다. SoccerRAG는 다양한 멀티모달 데이터세트를 지원하며, 동적 쿼리와 자동 데이터 검증 기능을 갖추고 있어 사용자와의 상호 작용을 개선하고 스포츠 기록에 대한 접근성을 높입니다.

- **Technical Details**: SoccerRAG는 SoccerNet 데이터셋을 기반으로 게임 비디오, 이미지 프레임, 오디오, 타임스탬프된 캡션, 게임 이벤트 및 선수 정보와 같은 멀티모달 데이터를 사용하여 자연어 쿼리를 통해 정보를 검색합니다. 프레임워크는 크게 데이터베이스, 특성 추출기, 특성 검증기, SQL 에이전트의 네 가지 주요 구성 요소로 이루어져 있습니다. 사용자의 자연어 쿼리에 대해 LLM이 특성을 추출하고, 이를 데이터베이스와 대조하여 SQL 쿼리를 생성 및 실행합니다.

- **Performance Highlights**: 예비 평가 결과 SoccerRAG는 기존 정보 검색 시스템에 비해 정확성과 사용자 참여도 측면에서 상당한 개선을 보여줬습니다. 특히 복잡한 쿼리를 효과적으로 처리하며, LLM과 RAG 사용의 잠재성을 강조하여 스포츠 데이터 분석의 미래 발전을 위한 가능성을 열었습니다. 오픈 소스 구현체 또한 제공되어 누구나 프레임워크를 실험하고 재현할 수 있습니다.



### FourierKAN-GCF: Fourier Kolmogorov-Arnold Network -- An Effective and Efficient Feature Transformation for Graph Collaborative Filtering (https://arxiv.org/abs/2406.01034)
- **What's New**: 이 연구에서는 그래프 기반의 추천 시스템인 FourierKAN-GCF를 새롭게 제안합니다. Fourier Kolmogorov-Arnold Network(KAN)을 사용하여 전통적인 다층 퍼셉트론(MLP)을 대체하였으며, 이는 GCN에서의 메시지 전달 중 특징 변환을 개선하였습니다.

- **Technical Details**: 제안된 FourierKAN-GCF 모델은 GCN에서의 메시지 전달 중 특징 변환 부분을 Fourier Kolmogorov-Arnold Network(KAN)으로 대체하여 사용자와 아이템의 삽입 벡터를 강화합니다. 또한, 메시지 드롭아웃과 노드 드롭아웃 기법을 적용하여 모델의 표현력과 강건성을 향상시켰습니다.

- **Performance Highlights**: 공개된 두 개의 데이터셋에서 광범위한 실험을 통해 FourierKAN-GCF 모델이 기존의 최첨단 방법들을 능가하는 성능을 보임을 입증했습니다.



### Cold-start Recommendation by Personalized Embedding Region Elicitation (https://arxiv.org/abs/2406.00973)
Comments:
          Accepted at UAI 2024

- **What's New**: 이 연구는 콜드 스타트 문제를 해결하기 위해 2단계의 개인화된 평가 유도를 도입한 새로운 프레임워크를 제안합니다. 첫 번째 단계에서는 사용자에게 인기 있는 아이템 세트를 평가하도록 요청하고, 두 번째 단계에서는 사용자의 선호도와 표현을 정교화하기 위해 적응형 아이템을 평가하도록 sequentially 값을 탐색합니다.

- **Technical Details**: 사용자의 임베딩 값을 포인트 추정치가 아닌 영역 추정치로 표현하여 추천 시스템의 정보를 얻는 가치를 극대화합니다. 또한, DPP(Determinantal Point Process)를 이용해 초기 설문 항목을 다양성과 품질 간의 균형을 맞추도록 선택합니다. 이후에는 사용자 맞춤형 설문을 통해 지속적으로 사용자의 선호도를 업데이트합니다.

- **Performance Highlights**: 인기 있는 데이터셋에서 제안된 방법의 효과를 실험적으로 입증했습니다. 기존의 평가 유도 방법들과 비교하여 더 나은 초기 추천 품질을 제공하는 것을 확인했습니다.



### Maximum-Entropy Regularized Decision Transformer with Reward Relabelling for Dynamic Recommendation (https://arxiv.org/abs/2406.00725)
- **What's New**: 본 연구에서는 성능 향상과 더불어 Recommendation Systems (RS)에서 기존 Decision Transformer(결정 변환기) 기반 방법의 한계를 극복하기 위해 Max-Entropy를 활용한 새로운 방법론, 즉 'EDT4Rec'을 소개합니다. 이 방법론은 특히 온라인 환경에서의 효과적인 탐색을 위해 Max-Entropy 강화 탐색 전략과 보상 재레이블링 기법을 포함하고 있습니다.

- **Technical Details**: EDT4Rec는 Max-Entropy 관점을 도입하여 탐색 전략의 효과를 극대화합니다. 이를 통해 온라인 환경에서의 탐색 능력이 향상됩니다. 또한, 모델이 최적화되지 않은 경로(sub-optimal trajectories)를 'stitching'할 수 있도록 보상 재레이블링(reward relabeling) 기법을 포함하고 있습니다. 이 기술은 사용자의 클릭 시퀀스를 바탕으로 보상 값을 재정의하여, 에이전트가 덜 보상 받는 경로를 피하게 합니다.

- **Performance Highlights**: EDT4Rec의 성능을 검증하기 위해 6개의 실제 오프라인 데이터셋과 온라인 시뮬레이터에서 종합적인 실험을 진행했습니다. 실험 결과, EDT4Rec는 Decision Transformer 기반 기존 방법들에 비해 뛰어난 성능을 보였으며, 온라인 환경에서도 효과적으로 적용될 수 있음을 입증했습니다.



### COS-Mix: Cosine Similarity and Distance Fusion for Improved Information Retrieva (https://arxiv.org/abs/2406.00638)
- **What's New**: 이 연구는 Retrieval-Augmented Generation(RAG)을 개선하기 위해 코사인 유사도와 코사인 거리 측정을 통합한 새로운 하이브리드 검색 전략을 제안합니다. 기존 코사인 유사도 측정이 특정 시나리오에서 임의의 결과를 초래할 수 있는 한계를 극복하고자, 코사인 거리 측정을 통해 벡터 간 비유사성을 정량화하여 보완적인 관점을 제공합니다. 이 접근법은 독점 데이터를 실험 대상으로 사용해 개선된 검색 성능을 보여주었으며, 특히 희소한 데이터에 대해 더 나은 성능을 발휘합니다.

- **Technical Details**: HTML 페이지들을 웹 크롤러로 수집하고 텍스트 파일로 변환한 후, OpenAI의 embedding 모델 text-embedding-ada-002를 사용해 임베딩 벡터로 변환했습니다. 사용자 쿼리에 대한 응답을 생성하기 위해 사전 학습된 GPT-3.5-TURBO 모델을 사용하고, BM25 검색기법과 전통적 벡터 검색기법을 결합한 하이브리드 검색기를 통해 정보를 검색했습니다. 또한, 거리를 기반으로 한 접근법을 사용해 쿼리를 해결했습니다. 저희 알고리즘은 정보 검색에 있어 희소한 정보에 대한 시간 절약을 목표로 했습니다.

- **Performance Highlights**: 프로프라이어터리 데이터셋에서는 거의 모든 지표에서 하이브리드 검색 방법이 더 나은 성능을 발휘했지만, 모든 데이터셋에서 컨텍스추얼 적합성(Contextual Relevancy)에서 낮은 성능을 보였습니다. 예를 들어, LLM이 관련 정보를 가진 상황에서도 사용자의 질문에 답하지 못하는 경우가 많았습니다. 그러나, 거리 기반 접근법을 통해 희소한 정보에 대해서도 정확한 답변을 제공할 수 있었습니다. 이렇게 함으로써 더 고급 LLM을 사용할 필요 없이 비용을 절감할 수 있었습니다.



### Making Recommender Systems More Knowledgeable: A Framework to Incorporate Side Information (https://arxiv.org/abs/2406.00615)
Comments:
          15 pages, 8 figures

- **What's New**: 이번 연구에서는 세션 기반 추천 시스템의 성능을 향상시키기 위해 아이템별 부가 정보를 활용하는 일반적인 프레임워크를 제안했습니다. 이 프레임워크는 기존 모델 아키텍처를 크게 변경하지 않고도 적용 가능합니다. 실험 결과, 부가 정보를 사용한 우리의 추천 시스템이 최신 모델들보다 훨씬 더 우수한 성능을 보였고, 수렴 속도도 훨씬 빨랐습니다. 또한 추천 시스템에 사용되는 어텐션 메커니즘을 정규화하는 새로운 유형의 로스를 제안하고, 이 로스가 모델 성능에 미치는 영향을 평가했습니다.

- **Technical Details**: 이 연구는 세션 기반 추천 시스템에서 부가 정보를 활용하여 성능을 향상시키기 위한 일반적인 프레임워크를 제안합니다. 예를 들어, 아이템의 카테고리나 서브클래스 ID와 같은 부가 정보가 사용됩니다. 우리는 Diginetica, Last.FM, MovieLens, Ta Feng와 같은 다양한 데이터셋에서 실험을 수행했습니다. 데이터 전처리 단계는 각 데이터셋에 맞게 세션을 나누고 세션 길이를 기준으로 데이터를 분할하며, 보다 효과적인 모델 훈련을 위해 부가 정보 매핑을 수행합니다.

- **Performance Highlights**: 부가 정보를 사용함으로써 우리의 추천 시스템은 최신 모델들보다 성능이 월등히 우수했으며, 더 빠르게 수렴했습니다. 예를 들어, Diginetica 데이터셋에서는 기본 정보만 사용한 RepeatNet 모델을 부가 정보를 추가해 성능 개선을 이루었습니다. Last.FM 데이터셋에서는 세션 길이를 50으로 설정하고, 8시간 동안 세션 정보를 수집하여 실험을 수행했으나, 부가 정보가 충분하지 않아서 베이스라인 모델로만 사용되었습니다.



### A Practice-Friendly Two-Stage LLM-Enhanced Paradigm in Sequential Recommendation (https://arxiv.org/abs/2406.00333)
- **What's New**: 이 논문은 대형 언어 모델(LLM)을 통합한 새로운 두 단계 LLM 강화 패러다임(TSLRec)을 제안합니다. 기존의 LLM-강화 방법들은 풍부한 텍스트 정보를 요구하거나 인스턴스 수준의 지도학습(SFT)에 의존하여 비효율적이고 많은 경우에 제한적이었습니다. TSLRec은 사용자 수준 SFT 작업을 도입하여 협업 정보를 LLM에 주입하고, LLM의 추론 능력을 활용한 임베딩을 생성합니다.

- **Technical Details**: TSLRec의 첫 번째 단계에서는 미리 훈련된 SRS 모델의 도움으로 사용자 수준 SFT 작업을 설계하여 협업 정보를 주입합니다. LLM은 각 항목의 잠재 카테고리를 추론하고 사용자의 선호도 분포를 재구성합니다. 두 번째 단계에서는 각 항목을 LLM에 입력하여 협업 정보와 LLM 추론 능력이 결합된 강화 임베딩을 생성합니다. 이 임베딩들은 이후 다양한 SRS 모델 훈련에 사용될 수 있습니다.

- **Performance Highlights**: TSLRec의 효과성과 효율성을 세 가지 SRS 벤치마크 데이터셋에서 검증한 결과, 기존의 방법들에 비해 뛰어난 성능을 보였으며, 특히 협업 정보 주입과 관련된 효율성이 높았습니다.



### BeFA: A General Behavior-driven Feature Adapter for Multimedia Recommendation (https://arxiv.org/abs/2406.00323)
- **What's New**: 이번 연구에서는 Behavior-driven Feature Adapter (BeFA)라는 새로운 기법을 소개합니다. 이는 콘텐츠 특징이 사용자 선호도를 제대로 반영하지 못하는 문제를 해결하기 위해 behavioral 정보로 콘텐츠 특징을 재구성하는 방법입니다. 또한, 각 제품의 콘텐츠 특징을 시각적으로 분석하기 위한 유사성 기반의 attribution 분석 방법도 제안되었습니다.

- **Technical Details**: 일반적인 멀티미디어 추천 시스템에서는 콘텐츠 특징을 추출하기 위해 사전에 학습된 feature encoder를 사용합니다. 그러나 이 방식은 사용자 선호도와 무관한 과도한 정보를 포함할 수 있습니다. 이를 해결하기 위해 BeFA는 behavioral 정보를 활용하여 콘텐츠 특징을 재구성하여 보다 정확한 사용자 선호도 모델링을 가능하게 합니다. 또한, 유사성 기반의 시각적 분석 기법을 통해 콘텐츠 특징이 사용자 선호도를 얼마나 잘 반영하는지 직관적으로 평가할 수 있습니다.

- **Performance Highlights**: 광범위한 실험 결과, BeFA는 다양한 멀티미디어 추천 방법에 대해 유의미한 성능 향상을 보여주었습니다. 정보 부재(information omission)와 정보 이탈(information drift) 문제를 해결함으로써 추천 시스템의 성능을 최적화할 수 있음을 증명했습니다. 이는 특히 콘텐츠와 behavioral 특징이 상호 작용하는 방식에서 현저한 개선을 확인할 수 있음을 시사합니다.



### Large Language Models for Relevance Judgment in Product Search (https://arxiv.org/abs/2406.00247)
Comments:
          10 pages, 1 figure, 11 tables - SIGIR 2024, LLM4Eval

- **What's New**: 본 논문은 대규모 언어 모델(LLM)을 활용하여 제품 검색 시 쿼리-아이템 페어(QIP)의 관련성을 자동으로 평가하는 기법을 제시합니다. 이는 수백만 개의 QIP 데이터셋을 갖춘 사람 평가자의 라벨링 데이터를 이용하여 모델을 최적화하는 방법입니다.

- **Technical Details**: 연구에서는 LLM(QIP)의 관련성 판단을 위해 Low Rank Adaptation (LoRA)와 아이템 속성의 다양한 모드 포함 및 프롬프트 방식을 포함한 여러 기법을 시험하였으며, 모델과 데이터 크기 사이의 트레이드오프가 중요한 역할을 함을 보여주었습니다. Llama2 7B와 Mistral 7B 모델을 LoRA를 통해 미세 조정했으며, 각 모델의 관련성 예측이 기존보다 월등히 개선되었습니다.

- **Performance Highlights**: Llama2 7B 모델은 88.8%, Mistral 7B 모델은 89.6%의 micro f1 점수를 기록하여, 오프더쉘프(pretrained) LLM보다 4.6% 개선된 성능을 보였습니다. 인간 평가자와의 비교 실험에서도 LLM은 높은 일치율(89%)을 보였으며, 이는 LLM이 인간의 관련성 판단을 경제적 및 시간적으로 대체할 수 있음을 시사합니다.



### ImplicitSLIM and How it Improves Embedding-based Collaborative Filtering (https://arxiv.org/abs/2406.00198)
Comments:
          Published as a conference paper at ICLR 2024; authors' version

- **What's New**: 이번 논문에서는 ImplicitSLIM이라는 새로운 비지도 학습 접근법을 소개합니다. 이 접근법은 협업 필터링(collaborative filtering)에 적용되는 고차원 희소 데이터의 처리를 위해 고안되었습니다. 기존의 SLIM 모델이 뛰어난 성능을 보이지만 메모리 집약적이고 확장하기 어려운 점을 개선하기 위해, ImplicitSLIM은 SLIM과 유사한 모델로부터 임베딩(embeddings)을 추출하여 계산 비용 및 메모리를 절약하는 방법을 제안합니다. 이를 통해 최신 및 고전적인 협업 필터링 방법들의 성능을 향상시키고 수렴 속도를 높이는 데 성공했습니다.

- **Technical Details**: ImplicitSLIM은 SLIM-like 모델을 명시적으로 학습하지 않고, 임베딩을 추출하여 메모리와 계산 자원을 절약하는 접근법을 사용합니다. 임베딩 기반 모델의 초기화 및 정규화에 사용할 수 있는 이 임베딩은 행렬 분해(matrix factorization)부터 오토인코더(autoencoder) 기반 접근법, 그래프 합성곱 네트워크(GCN) 기반 모델 및 RNN이나 Transformer 기반 순차 모델들도 포함합니다. 이 방법은 LLE(Locally Linear Embeddings)와 EASE(Embarrassingly Shallow Autoencoders)와 같은 기존 방법들에 기반을 두고, 이들의 단계를 더 최적화된 방식으로 단순화합니다.

- **Performance Highlights**: ImplicitSLIM을 도입함으로써 MovieLens-20M 및 Netflix Prize 데이터셋에서 최신의 RecVAE 및 H+Vamp(Gated) 모델의 성능을 크게 향상시켰습니다. 대부분의 경우, ImplicitSLIM은 최종 추천 결과를 매우 상당히 개선하는 것으로 나타났습니다.



### Extracting Essential and Disentangled Knowledge for Recommendation Enhancemen (https://arxiv.org/abs/2406.00012)
- **What's New**: 이 논문은 추천 시스템(Recommender Systems)에서 자주 발생하는 문제인 'catastrophic forgetting'을 해결하기 위해 제안된 새로운 접근법을 소개합니다. 즉, 빠르게 변화하는 데이터 분포로 인해 발생하는 문제를 해결하기 위해 역사적 데이터에서 필수적이고 분리된 지식을 추출하는 두 가지 제약을 제안합니다. 이는 모델 파라미터를 증가시키지 않고도 추천 시스템의 기억 및 일반화 능력을 향상시킬 수 있도록 합니다.

- **Technical Details**: 논문은 데이터의 본질적 원칙(essential principle)과 분리 원칙(disentanglement principle)을 적용하여 과거 데이터에서 필수적인 지식을 추출하고 저장된 정보의 중복성을 줄이는 방법을 설명합니다. 본질적 원칙은 입력을 대표 벡터로 압축하여 잡음을 걸러내고, 분리 원칙은 정보의 중복을 줄여 분리된 불변 패턴을 캡처하도록 지식 베이스를 유도합니다.

- **Performance Highlights**: 두 개의 데이터셋을 기반으로 한 광범위한 실험을 통해 제안된 방법의 유효성을 입증했습니다. 이러한 방법들은 지식을 합리적으로 압축하여 강력하고 일반화된 지식 표현을 촉진합니다.



### DisCo: Towards Harmonious Disentanglement and Collaboration between Tabular and Semantic Space for Recommendation (https://arxiv.org/abs/2406.00011)
- **What's New**: 새로운 추천 시스템 모델 DisCo가 소개됩니다. DisCo는 탭(표) 형태의 표현 공간과 의미 표현 공간 간의 고유 패턴을 분리하고 협력하여 추천의 정확도를 높입니다. 기존 방식은 두 공간을 단순히 정렬하는 반면, DisCo는 각 공간의 고유 정보를 유지하고 잘 활용합니다.

- **Technical Details**: DisCo는 1) 듀얼 사이드 어텐티브 네트워크(dual-side attentive network)로 도메인 내부 및 도메인 간 패턴을 포착하고, 2) 두 표현 공간 각각의 작업 관련 정보를 유지하며 노이즈를 필터링하는 충분성 제약(sufficiency constraint)을 두며, 3) 각 표현 공간의 고유 정보를 유지하기 위한 분리 제약(disentanglement constraint)을 사용합니다. 이들 모듈은 두 공간의 협력과 분리 균형을 맞추어 정보 전달 벡터를 생성합니다.

- **Performance Highlights**: DisCo는 다양한 추천 백본(예: DeepFM, xDeepFM 등)에서의 성능을 검증받았으며 일관된 우수성을 보였습니다. 여러 세부 연구와 효율성 분석을 통해 각 구성 요소의 효과를 입증했습니다. DisCo는 다양한 백본과 호환 가능한 모델로, 높은 유연성과 일반성을 제공합니다.



### Disentangling Specificity for Abstractive Multi-document Summarization (https://arxiv.org/abs/2406.00005)
Comments:
          The IEEE World Congress on Computational Intelligence (WCCI 2024)

- **What's New**: Multi-document summarization (MDS) 모델인 DisentangleSum이 소개되었습니다. DisentangleSum은 각 문서의 독특한 내용을 분리해내는 새로운 접근 방식을 사용합니다. 이 모델은 문서 간의 고유한 정보를 추출함으로써 종합적인 요약을 생성하는 것을 목표로 합니다. 이 기술은 문서셋 내 각 문서의 고유성을 고려한 최초의 딥러닝 기반 MDS 접근법입니다.

- **Technical Details**: DisentangleSum 모델은 문서 고유성 표현 학습을 통해 문서셋 내 각 문서의 고유 정보를 분리해냅니다. 제안된 모델은 orthogonal constraint(직교 제약)을 활용해 문서 고유성 표현 벡터가 서로 수직으로 정렬되도록 합니다. 이를 통해 의미적 분리를 보장합니다. 문서셋의 고유 표현과 문서셋 자체의 표현을 모두 사용해 요약을 생성하기 위한 목표 함수를 설계했습니다. 이는 손실 증가를 직선화하여 문서셋에 있는 많은 문서를 처리할 수 있게 합니다.

- **Performance Highlights**: 실험 결과, DisentangleSum은 두 개의 MDS 데이터셋에서 그 효과를 입증하였습니다. 다양한 관점에서 포괄적인 분석을 수행하여 DisentangleSum의 메커니즘과 제안된 모델이 잘 작동하는 상황을 조사하였습니다. 이 모델은 기존의 문서 간 관계를 무시하고 개별 문서 고유성을 놓치는 기존 방법론의 한계를 극복합니다.



### Navigating the Future of Federated Recommendation Systems with Foundation Models (https://arxiv.org/abs/2406.00004)
Comments:
          20 pages, position paper

- **What's New**: 최근 연방 학습(Federated Learning, FL)과 추천 시스템(Recommendation Systems, RS)의 통합, 즉 연방 추천 시스템(Federated Recommendation Systems, FRS)이 사용자 데이터를 클라이언트 디바이스에 유지하며 사용자 프라이버시를 보호하는 것으로 주목받고 있습니다. 그러나 FRS는 데이터의 이질성(heterogeneity) 및 희소성(sparsity) 문제로 한계를 겪고 있습니다. 본 논문에서는 대규모 사전 학습(Foundation Models, FMs)이 이러한 문제를 해결하는 데 사용될 수 있음을 제안합니다. FM을 FRS에 통합하는 방법론에 대한 종합적인 리뷰를 수행하며, FRS와 FM이 직면한 도전 과제와 향후 연구 방향을 논의합니다.

- **Technical Details**: 연방 추천 시스템(FRS)은 중앙 서버에 데이터를 업로드하지 않고 클라이언트 측에서 로컬 데이터를 활용하여 모델을 학습하는 시스템으로, FL과 RS가 결합된 형태입니다. FSR은 각 클라이언트가 로컬 데이터로 모델을 학습 후 파라미터를 서버로 전송, 서버는 이를 통합하여 글로벌 모델을 업데이트합니다. 그러나 이는 데이터의 이질성과 희소성 문제를 야기합니다. 이를 해결하기 위해 사전 학습된 대규모 모델(FM)을 FRS에 도입함으로써 데이터 부족 문제를 완화하고 다양한 클라이언트의 요구를 반영할 수 있습니다.

- **Performance Highlights**: 논문에서는 FRS와 FM의 조합이 데이터의 희소성과 이질성 문제를 해결할 수 있음을 강조합니다. FM은 대량의 비라벨 데이터에서 자가 지도 학습(Self-Supervised Learning)을 통해 사전 학습되므로, 다운스트림 작업에 적용될 때 높은 성능을 발휘할 수 있습니다. 이러한 특성은 FRS가 효과적으로 동작하는데 도움이 되며, 사용자 프라이버시를 보호하면서도 정확도와 효율성을 높일 수 있습니다.



### Poisoning Attacks and Defenses in Recommender Systems: A Survey (https://arxiv.org/abs/2406.01022)
Comments:
          22 pages, 8 figures

- **What's New**: 이 논문은 추천 시스템(recommender systems)의 취약성을 공격자의 시각에서 분석한 최초의 종합적 조사 연구입니다. 이는 독특한 관점을 제공하며 포이즈닝 공격(poisoning attacks)의 메커니즘과 영향을 심층적으로 탐구합니다.

- **Technical Details**: 포이즈닝 공격은 4단계로 구성된 체계적인 파이프라인(pipeline)으로 설명됩니다: 공격 목표 설정, 공격자 능력 평가, 피해자 아키텍처 분석, 포이즈닝 전략 구현. 또한 방어 전략은 적대적 데이터 필터링과 견고한 모델 학습으로 분류됩니다.

- **Performance Highlights**: 분석을 통해 기존 연구의 한계와 새로운 연구 방향을 제안합니다. 이는 추천 시스템의 보안을 강화하기 위한 추가 연구와 혁신적 접근을 촉진할 것입니다.




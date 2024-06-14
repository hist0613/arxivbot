### Generating Query Recommendations via LLMs (https://arxiv.org/abs/2405.19749)
Comments:
          Generating Query Recommendations via LLMs

- **What's New**: 이번 논문은 검색 엔진의 쿼리 추천 시스템을 대상으로 하는 새로운 접근 방식인 Generative Query Recommendation (GQR)을 제안합니다. GQR은 LLM(Large Language Model)을 기반으로 하며, 쿼리 로그나 사용자 데이터 없이도 작동할 수 있습니다. 특히, 기존의 복잡한 데이터 수집 파이프라인을 거칠 필요가 없다는 점에서 'cold start' 상황에서도 유리합니다. 추가적으로, Retriever-Augmented GQR(RA-GQR)라는 변형을 통해 쿼리 로그를 활용해 성능을 향상시킵니다.

- **Technical Details**: GQR은 LLM을 활용하여 쿼리 추천을 생성하는 접근 방식으로, 별도의 학습이나 튜닝이 필요 없습니다. 이 방법은 단일 예제를 사용하여 특정 추천 작업을 이해하도록 설계된 프롬프트를 포함합니다. RA-GQR는 쿼리 로그에서 유사한 쿼리를 검색하여 다이나믹하게 프롬프트를 구성함으로써 추천 정확도를 높입니다.

- **Performance Highlights**: GQR 시스템은 NDCG@10 및 명확성 점수에서 기존의 상업용 검색 엔진 및 최신 방법론을 능가하는 성능을 보여주었습니다. Robust04 및 ClueWeb09B 데이터셋에서 이전 최고 경쟁자보다 평균적으로 NDCG@10 성능을 4%까지 향상시켰습니다. RA-GQR은 더욱 발전된 성능을 보이며, 최고 경쟁자를 대상으로 NDCG@10에서 각각 11%와 6%의 향상을 달성했습니다. 사용자 선호도를 평가하는 블라인드 사용자 연구에서도 GQR 시스템이 59%의 선호도를 가져가는 우수한 결과를 보였습니다.



### Keyword-driven Retrieval-Augmented Large Language Models for Cold-start User Recommendations (https://arxiv.org/abs/2405.19612)
Comments:
          10 pages, 10 figures, 4 tables

- **What's New**: KALM4Rec은 'Keyword-driven Retrieval-Augmented Large Language Models for Cold-start User Recommendations'이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 사용자들이 최소한의 키워드만 제공하는 상황에서 레스토랑 추천 문제를 해결하는 것을 목표로 하고 있습니다. 두 주요 단계로 작동하며, 후보군 추출(candidates retrieval)과 LLM을 사용한 후보군 재정렬(LLM-based candidates re-ranking)로 구성됩니다.

- **Technical Details**: KALM4Rec은 키워드 기반 검색 모델을 사용하여 잠재적 후보군을 식별한 후, 다양한 프롬프트 전략(예: zero-shot, few-shot 기법)을 사용하여 LLM이 후보군을 다시 정렬하는 두 단계로 이루어집니다. 이를 통해 많은 토큰을 처리해야 하는 LLM의 한계를 극복하고 잘못된 정보를 생성하는 위험을 줄입니다. Yelp의 레스토랑 데이터셋과 사용자 리뷰를 사용한 평가에서, 프레임워크가 추천의 품질을 크게 개선함을 확인했습니다.

- **Performance Highlights**: 실험 결과, KALM4Rec은 초기 사용자 추천 시스템의 성능을 크게 향상시켰습니다. 특히, 맥락 지시사항을 LLM 프롬프트에 통합함으로써 cold-start 사용자 추천 시스템의 성능이 현저히 향상되었습니다.



### Uncertainty-aware sign language video retrieval with probability distribution modeling (https://arxiv.org/abs/2405.19689)
- **What's New**: 수화 비디오 검색에서 새롭게 제시된 Uncertainty-aware Probability Distribution Retrieval (UPRet) 방법론을 도입했습니다. 이 방법은 수화 비디오와 텍스트 간의 맵핑 과정을 확률 분포의 관점에서 개념화하고, 그들의 잠재적 상호 관계를 탐구하며, 유연한 매핑을 가능하게 합니다.

- **Technical Details**: UPRet는 수화 비디오와 텍스트를 다변량 가우시안 분포로 모델링하여 비디오와 텍스트 간의 대응을 더 넓은 의미 공간에서 탐색합니다. 또한, Monte Carlo 샘플링을 통해 데이터 분포의 구조와 연관성을 심도 있게 탐구하며, Optimal Transport(OT)를 사용하여 분포 간의 거리를 미세하게 측정합니다.

- **Performance Highlights**: 제안된 방법은 세 가지 벤치마크 데이터셋(Retrieval)에 대해 최첨단 성능을 달성했습니다: How2Sign (59.1%), PHOENIX-2014T (72.0%), CSL-Daily (78.4%). 이로 인해 기존의 단일 매핑 방법론들이 가지는 한계를 극복하며, 수화 비디오와 텍스트 간의 불확실성 및 다의성을 더 정확하게 포착할 수 있습니다.



### MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings (https://arxiv.org/abs/2405.19504)
- **What's New**: 이번 연구에서는 MUVERA(무베라)가 소개되었습니다. MUVERA는 다중 벡터 검색을 단일 벡터 유사성 검색으로 줄이는 검색 메커니즘입니다. 이 방법을 통해 기존 MIPS(Maximum Inner Product Search) 알고리즘을 활용할 수 있습니다. MUVERA는 쿼리와 문서의 고정 차원 인코딩(Fixed Dimensional Encodings, FDEs)을 비대칭적으로 생성하여, 그 내적이 다중 벡터 유사성을 근사하도록 변환합니다.

- **Technical Details**: MUVERA는 쿼리와 문서 세트를 고정 차원 벡터로 변환한 뒤, 이 벡터 간의 내적이 다중 벡터 유사성을 근사하도록 합니다. 이는 첫 번째 단일 벡터를 통한 Chamfer Similarity(샴퍼 유사성)를 제공하며, 이 과정은 기존의 MIPS 솔버를 이용하여 수행됩니다. 이러한 방법은 이론적인 보증을 제공하며, Chamfer 유사성 검색에 대한 최초의 단일 벡터 프록시를 제시합니다.

- **Performance Highlights**: MUVERA는 기존의 최신 휴리스틱 기법과 비교하여 2~5배 적은 후보를 검색하면서도 동일한 리콜(Recall)을 달성합니다. 또한, BEIR 데이터셋을 대상으로 한 성능 비교에서 평균 10% 향상된 리콜과 90% 낮아진 지연 시간을 보였습니다. 또한, 제품 양자화(Product Quantization) 기법을 통해 FDE를 32배 압축하여 메모리 사용량을 크게 줄이면서 품질 손실은 거의 없는 결과를 확인했습니다.




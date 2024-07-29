### Modeling Activity-Driven Music Listening with PACE (https://arxiv.org/abs/2405.01417)
Comments: 5 pages, CHIIR'24 conference

- **What's New**: 이 논문에서는 음악 추천 시스템의 주요 주제인 '듣기 맥락(listening context)'을 넘어 사용자의 주기적인 듣기 행동을 고려하는 새로운 프레임워크, PACE (PAttern-based user Consumption Embedding)를 제안합니다. PACE는 사용자의 다중 채널 시계열 소비 패턴을 활용하여 사용자의 반복적인 청취 동태를 보여주는 사용자 벡터를 구축합니다.

- **Technical Details**: PACE는 주간 소비 패턴을 기반으로 사용자 임베딩을 생성하며, 음악 스트리밍 서비스에서 발견되는 다양한 소비 용도를 특성화합니다. 또한, 각 사용자의 소비 이력은 다차원 시계열로 인코딩되고, 이 시계열 데이터로부터 고정된 수의 표준화된 행동, 즉 '아톰(atoms)'을 파생하여 사용자 임베딩을 생성합니다. 이러한 아톰들은 각각이 주간 청취 패턴을 나타내는 시계열로 구성됩니다.

- **Performance Highlights**: PACE의 성능은 사용자가 음악을 듣는 동안 수행하는 활동을 예측하는 태스크를 통해 평가되었습니다. 이 평가는 Deezer 사용자들을 대상으로 한 설문조사를 바탕으로 하며, PACE를 통해 얻은 임베딩이 활동 주도적 청취 행태를 포착할 수 있는지를 검증합니다. 이러한 접근은 추천 시스템의 정확성을 높이고, 사용자의 음악 소비 습관을 더 깊이 이해할 수 있는 통찰을 제공합니다.



### Are We Really Achieving Better Beyond-Accuracy Performance in Next  Basket Recommendation? (https://arxiv.org/abs/2405.01143)
Comments: To appear at SIGIR'24

- **What's New**: 이 연구에서는 다음 바구니 추천(NBR)에 대해 '정확성'뿐만 아니라 '정확성을 넘어선 지표'(beyond-accuracy metrics)를 최적화하는 새로운 접근 방식인 TREx 프레임워크를 제안합니다. 이 프레임워크는 반복(Repeat) 항목과 탐색(Explore) 항목을 분리하여 처리함으로써, 높은 정확성을 유지하면서도 다양성과 품목 공정성과 같은 추가적인 지표들을 향상시킬 수 있는 방법을 탐구합니다.

- **Technical Details**: TREx(두 단계 반복-탐색) 프레임워크는 NBR에서 반복되는 항목과 탐색 항목을 별도로 예측합니다. 간단하지만 효과적인 '반복 모듈'은 정확한 반복 항목 예측을 담당하고, 두 가지 '탐색 모듈'은 다양성과 공정성 같은 정확성을 넘어선 지표들을 최적화하기 위해 설계되었습니다. 이 방식은 기존의 NBR 방법들과 달리 모든 항목에 대한 점수나 확률을 출력하여 상위-k개 항목을 선정하는 방식에서 벗어나, 반복과 탐색의 차이를 명확히 구분합니다.

- **Performance Highlights**: 실험결과 TREx는 높은 정확도를 달성하면서도 다양성과 품목 공정성을 포함한 정확성을 넘어선 지표들을 개선할 수 있음을 확인하였습니다. 사용한 데이터셋은 두 가지 널리 사용되는 데이터셋으로, 5가지 공정성 지표와 3가지 다양성 지표를 바탕으로 실험을 수행했습니다. 이 과정에서, TREx는 단순 반복 모델만으로도 최고 수준의 정확도를 보였으며, 탐색 모델을 통해 추가 지표들의 성능을 끌어올렸습니다.



### Generative Relevance Feedback and Convergence of Adaptive Re-Ranking:  University of Glasgow Terrier Team at TREC DL 2023 (https://arxiv.org/abs/2405.01122)
Comments: 5 pages, 5 figures, TREC Deep Learning 2023 Notebook

- **What's New**: 이 논문에서는 TREC 2023 Deep Learning Track에 참여한 글래스고 대학 테리어(Terrier) 팀의 새로운 접근 방식을 설명합니다. 제안된 방법은 큰 언어 모델을 사용한 생성적 관련성 피드백(generative relevance feedback)과 적응형 재정렬(adaptive re-ranking)를 통합합니다. 이 팀은 BM25와 SPLADE와 같은 두 가지 희소 검색(sparse retrieval) 접근 방식을 적용하고, BM25 코퍼스 그래프에 점수를 매기는 monoELECTRA 교차 인코더(cross-encoder)를 사용하여 재정렬하는 방식을 채택했습니다.

- **Technical Details**: 연구에서는 PyTerrier 정보 검색 도구를 사용하여, 생성적 질의 개혁(Gen-QR)과 생성적 의사 관련 피드백(Gen-PRF)을 포함하여 다양한 질의 유형에 대한 첫 단계 검색의 효과를 조사하였습니다. 또한, 적응형 재정렬을 사용하여 초기 단계의 영향을 받지 않는 그래프 탐색의 지점을 찾기 위해 첫 단계와의 비교도 수행하였습니다. BM25 코퍼스 그래프와 결합된 monoELECTRA 모델을 통해 적응형 재정렬(adaptive re-ranking)이 수행되었습니다.

- **Performance Highlights**: 실험 결과, 생성적 질의 개혁을 적용한 방법은 성능 향상을 보였으며, 특히 생성적 의사 관련 피드백을 적용한 'uogtr_b_grf_e_gb' 실행에서 P@10과 nDCG@10 모두에서 가장 강력한 성능을 나타냈습니다. 연구 결과는 다양한 질의 유형에서 이러한 생성적 접근방식의 유효성을 검증하는 데 기여하며, 충분히 큰 코퍼스 그래프를 사용하는 것이 가벼운 첫 단계 순위 결정자의 성능을 학습된 첫 단계 검색 함수의 성능과 동일하게 만들 수 있음을 시사합니다.



### Faster Learned Sparse Retrieval with Block-Max Pruning (https://arxiv.org/abs/2405.01117)
Comments: SIGIR 2024 (short paper track)

- **What's New**: 이 논문은 학습된 희소 검색 모델을 위해 최적화된 새로운 동적 프루닝 전략인 Block-Max Pruning (BMP)을 소개합니다. BMP는 특정 문서 집합의 잠재적인 관련성에 기반하여 우선 순위를 지정하기 위해 블록 필터링 메커니즘을 활용합니다. 이 전략은 학습된 희소 검색 환경에서 발생하는 인덱스에 특화된 것으로, 더 높은 검색 효율성과 정확성 사이의 균형을 제공합니다.

- **Technical Details**: BMP는 문서 공간을 작고 연속적인 문서 범위로 나누는 블록 필터링 메커니즘을 사용합니다. 이러한 블록은 실시간으로 집계되고 정렬되며, 필요에 따라 전체 처리됩니다. 이 프로세스는 안전한 조기 종료 기준 또는 대략적인 검색 요구에 따라 안내됩니다. BMP는 SPLADE와 uniCOIL과 같은 주요 학습된 희소 모델에서 뛰어난 성능을 보여주며, 전통적인 동적 프루닝 전략과 비교할 때 탁월한 효율성을 제공합니다.

- **Performance Highlights**: 실험 결과에 따르면, BMP는 안전한 결과 검색에서 기존 메소드보다 두 배에서 최대 60배 빠른 성능을 보여줍니다(MS MARCO 데이터셋에서). 또한, 대략적인 질의 처리에서는 효율성과 효과 사이의 훨씬 나은 균형을 달성했습니다.



### "In-Context Learning" or: How I learned to stop worrying and love  "Applied Information Retrieval" (https://arxiv.org/abs/2405.01116)
Comments: 9 Pages, 3 Figures, Accepted as Perspective paper to SIGIR 2024

- **Korean AI Newsletter Summary**: [{"What's New": '이번 연구는 인-컨텍스트 학습(ICL)의 새로운 패러다임을 소개하며, 큰 언어 모델(Large Language Models, LLMs)의 다운스트림 작업(Downstream Task)을 위해 특정 파라미터를 미세 조정(Fine-tuning)하는 대신, 예제를 프롬프트 지시어에 추가하여 디코더 생성 과정을 제어합니다. 이 방법은 ICL을 정보 검색(IR)의 쿼리와 유사한 아날로지로 설명하며, 효율적인 예제 선택을 위한 새로운 접근법을 제시합니다.'}, {'Technical Details': '이 연구는 비감독 순위 모델(Unsupervised Ranking Models)과 감독 순위 모델(Supervised Ranking Models)을 사용하여 학습 세트에서 유용한 예제를 검색하는 과정을 설명합니다. 예제의 효과를 극대화하기 위해 관련성의 개념을 재정의하고, 특정 작업에 대한 유용성을 기반으로 예제를 선택하는 감독된 학습 모델을 훈련할 수 있다고 제안합니다. 이러한 모델은 Bi-encoder나 Cross-encoder와 같은 아키텍처를 활용할 수 있습니다.'}, {'Performance Highlights': 'LLMs는 몇 가지 예제만으로도 높은 수준의 적응성을 보이며, ICL 방법론을 통해 라벨이 지정된 예제에 대한 과적합(Overfitting) 문제 없이 다양한 도메인에서 효과적인 결과를 얻을 수 있음을 보여줍니다. 이 연구는 ICL과 IR 사이의 상호 작용을 통해 ICL 예측의 효율성을 개선할 수 있는 가능성을 시사합니다.'}]



### Fair Recommendations with Limited Sensitive Attributes: A  Distributionally Robust Optimization Approach (https://arxiv.org/abs/2405.01063)
Comments: 8 pages, 5 figures

- **What's New**: 새로운 연구에서는 권장 시스템(Recommender Systems)의 공정성(Fairness)을 증진시키는 새로운 방법인 분포적으로 강인한 공정 최적화(Distributionally Robust Fair Optimization, DRFO)를 제안합니다. 이 방법은 민감한 속성(Sensitive Attributes) 정보의 부족에도 불구하고 권장 시스템의 공정성을 증진시킬 수 있습니다.

- **Technical Details**: 기존의 공정성 향상 방법들은 모든 민감한 속성의 사용 가능성을 전제로 하는 반면, 이 연구에서는 민감한 속성 정보가 제한적일 때의 문제를 해결하고자 합니다. DRFO는 재구성된 민감한 속성들의 잠재적인 확률 분포를 고려하여 최악의 경우의 불공정성을 최소화함으로써 재구성 오류의 영향을 고려합니다.

- **Performance Highlights**: 이론적 및 실증적 증거들은 DRFO가 민감한 속성 정보가 제한적인 상황에서도 권장 시스템의 공정성을 효과적으로 보장할 수 있음을 보여줍니다. 이 방법은 실제 세계의 복잡한 민감한 속성 재구성 문제와 법적 규제로 인한 오류에 강한 방법임을 입증합니다.



### Multi-intent-aware Session-based Recommendation (https://arxiv.org/abs/2405.00986)
Comments: SIGIR 2024. 5 pages

- **What's New**: 본 연구에서는 세션 기반 추천(SBR) 시스템의 한계를 극복하고자 새로운 모델인 Multi-intent-aware Session-based Recommendation Model (MiaSRec)을 제안합니다. MiaSRec 모델은 세션 내에서의 아이템 빈도를 나타내는 빈도 임베딩(frequency embedding) 벡터를 사용하여 반복적인 아이템의 정보를 강화하고, 각 아이템을 중심으로 다양한 사용자 의도를 표현하는 다중 세션 표현을 도입하여 중요한 것을 동적으로 선택합니다.

- **Technical Details**: MiaSRec 모델은 세션 내 아이템과의 관계를 파악하여 세션 표현을 학습하는 데 중점을 둔 기존의 복잡한 신경 기반(neural-based) 인코더의 설계를 초월합니다. 이 모델은 각 세션 아이템을 중심으로 여러 세션 표현을 파생시키고, 이 중에서 중요한 표현을 동적으로 선택하여 사용자의 다양한 의도를 반영합니다.

- **Performance Highlights**: MiaSRec은 여섯 개 데이터셋에서 기존의 최첨단 SBR 모델들을 능가하는 성능을 보여줍니다. 특히 평균 세션 길이가 긴 데이터셋에서 뛰어난 성능을 보여주며, MRR@20 (Mean Reciprocal Rank at 20) 및 Recall@20 지표에서 각각 최대 6.27% 및 24.56%의 성능 향상을 기록했습니다.



### Efficient and Responsible Adaptation of Large Language Models for Robust  Top-k Recommendations (https://arxiv.org/abs/2405.00824)
- **What's New**: 본 논문에서는 기존 추천 시스템(Recommendation Systems, RSs)이 모든 훈련 샘플에 대해 동일하게 성능을 최적화하는 한계를 극복하고 다양한 사용자 집단의 특성에 맞춰 서비스를 제공하기 위해 하이브리드 작업 할당 프레임워크를 제안하였습니다. 이 프레임워크는 대규모 언어 모델(Large Language Models, LLMs)과 전통적인 RSs의 장점을 결합하여 하드 샘플(hard samples)에 대응하며, 특히 약한(weak) 사용자와 비활성(inactive) 사용자를 식별하고 그들에게 최적화된 랭킹 성능을 제공합니다.

- **Technical Details**: 이 연구에서는 두 단계 접근법을 이용해 RSs에 대한 강건성(robustness)을 향상시킵니다. 첫 번째 단계에서는 RSs에 의해 부적합한 랭킹 성능을 받는 약한 및 비활성 사용자를 식별합니다. 다음으로, 이러한 사용자들을 위해 인-콘텍스트 학습(in-context learning) 방법을 적용하며, 각 사용자의 상호작용 이력을 고유한 랭킹 작업으로 구성하여 LLM에 제공합니다. 연구는 협업 필터링(Collaborative Filtering) 및 학습-투-랭크(Learning-to-Rank) 모델 같은 다양한 추천 알고리즘과 오픈 소스 및 폐쇄 소스의 두 가지 LLMs를 포함하여 하이브리드 프레임워크를 테스트하였습니다.

- **Performance Highlights**: 실제 세계 데이터셋 세 개에서의 실험 결과, 이 하이브리드 프레임워크는 약한 사용자의 수를 현저히 감소시켰으며 RSs의 서브-포퓰레이션(sub-population)에 대한 강건성을 약 12% 향상시켰습니다. 이는 전반적인 성능 개선을 이끌었으며 비용 증가를 불균형적으로 야기하지 않는 효과적이고 책임감 있는 LLMs의 적용을 가능하게 하였습니다.




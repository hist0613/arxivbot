New uploads on arXiv(cs.CL)

### ElasticRec: A Microservice-based Model Serving Architecture Enabling Elastic Resource Scaling for Recommendation Models (https://arxiv.org/abs/2406.06955)
- **What's New**: ElasticRec는 추천 시스템(RecSys)을 위한 새로운 모델 서빙 아키텍처로, 리소스 탄력성과 높은 메모리 효율성을 제공합니다. 기존의 모델 단위로 리소스를 할당하는 방식 대신, ElasticRec는 미세한 수준의 리소스 할당을 가능하게 하는 마이크로서비스(microservice) 소프트웨어 아키텍처를 바탕으로 설계되었습니다.

- **Technical Details**: ElasticRec는 두 가지 주요 특징을 가지고 있습니다. 첫째, 마이크로서비스 기반 추론 서버를 사용하여 리소스 탄력성을 극대화합니다. 마이크로서비스는 큰 단일 응용 프로그램을 여러 개의 독립적이고 세밀한 서비스로 나눌 수 있게 해줍니다. 둘째, 유틸리티 기반의 리소스 할당 정책을 통해 높은 메모리 효율성을 달성합니다. 모델은 밀집 DNN 레이어와 희석 임베딩 레이어로 나뉘며, 임베딩 레이어는 다시 '핫'과 '콜드' 임베딩 단위로 나뉩니다.

- **Performance Highlights**: ElasticRec는 메모리 할당 크기를 평균 3.3배 줄이고, 메모리 유틸리티를 8.1배 증가시켜 평균적으로 배포 비용을 1.6배 절감할 수 있습니다. 또한, 고유한 리소스 수요에 맞춘 리소스 할당을 통해 전체적인 QPS(Queries Per Second)를 최대화할 수 있습니다.



### Non-autoregressive Personalized Bundle Generation (https://arxiv.org/abs/2406.06925)
Comments:
          Submitted to Information Processing & Management

- **What's New**: 최근 추천 시스템 연구에서, 사용자의 선호에 맞춰 개인화된 번들을 생성하는 문제에 대한 관심이 증가하고 있습니다. 기존 연구들은 번들의 순서 불변성 특성을 고려하지 않아 순차적 모델링 방법을 채택한 반면, 본 연구에서는 비자기회귀(non-autoregressive) 메커니즘을 활용하여 번들을 생성하는 새로운 인코더-디코더 프레임워크인 BundleNAT을 제안합니다. 이를 통해 본 연구는 번들의 순서에 의존하지 않고 한 번에 목표 번들을 생성할 수 있습니다.

- **Technical Details**: 본 연구에서는 사전 훈련(pre-training) 기술과 그래프 신경망(Graph Neural Network, GNN)을 채택하여 사용자 기반 선호도 및 아이템 간 호환성 정보를 완전하게 내재화합니다. 이후 자기 주의(self-attention) 기반 인코더를 활용하여 글로벌 종속성 패턴을 추출합니다. 이를 기반으로 번들 내의 순열에 고유한 디코딩 아키텍처를 설계하여 직접적으로 원하는 번들을 생성합니다.

- **Performance Highlights**: YouShu와 Netease의 세 가지 실제 데이터셋에서 진행된 실험 결과, BundleNAT은 정밀도(Precision), 확장 정밀도(Precision+), 재현율(Recall)에서 각각 최대 35.92%, 10.97%, 23.67%의 절대적 향상을 보여 현재 최신 기법들을 크게 능가하는 성과를 보였습니다.



### Scaling the Vocabulary of Non-autoregressive Models for Efficient Generative Retrieva (https://arxiv.org/abs/2406.06739)
Comments:
          14 pages, 6 tables, 2 figures

- **What's New**: 이 논문은 정보 검색(Information Retrieval)을 더 효율적으로 수행하기 위해 Non-autoregressive(NAR) 언어 모델을 사용하는 새로운 접근법을 제안합니다. 특히, 다중 단어 엔티티 및 공통 구문(phrases)을 포함한 확장된 어휘(vocabulary)를 사용하여 NAR 모델의 성능을 향상시키는 PIXAR 접근법을 제안합니다. 이 방법은 Autoregressive(AR) 모델에 비해 지연 시간(latency)과 비용은 낮추면서도 검색 성능을 유지하도록 돕습니다.

- **Technical Details**: PIXAR은 NAR 모델에서 다중 단어 구문을 예측할 수 있는 확장된 목표 어휘를 사용하여 검색 성능을 향상시킵니다. 추가된 어휘는 최대 5백만 개의 토큰을 포함하며, 이로 인해 모델은 오토리그레시브 모델만큼의 복잡한 종속성 문제를 해결할 수 있습니다. 또한, PIXAR은 효율적인 추론 최적화 기법을 도입하여 큰 어휘를 사용하더라도 낮은 추론 지연 시간을 유지하기 위해 설계되었습니다.

- **Performance Highlights**: PIXAR은 MS MARCO에서 MRR@10 기준으로 31.0% 상대 성능 향상을, Natural Questions에서 Hits@5 기준으로 23.2% 향상을 보여줍니다. 또한, 대형 상업 검색 엔진에서 진행한 A/B 테스트 결과, 광고 클릭은 5.08%, 수익은 4.02% 증가했습니다.



### Leveraging Large Language Models for Knowledge-free Weak Supervision in Clinical Natural Language Processing (https://arxiv.org/abs/2406.06723)
- **What's New**: 본 논문에서는 임상 도메인에서 라벨된 학습 데이터가 충분하지 않은 상황에서, 약한 감독학습(weak supervision)과 컨텍스트 학습(in-context learning)을 사용하는 새로운 접근 방식을 제안합니다. 이 새로운 방법은 도메인 지식 없이도 Llama2와 같은 대형 언어 모델(LLM)을 활용하여 약한 라벨 데이터를 생성하고, 이를 통해 성능이 우수한 BERT 모델을 학습합니다.

- **Technical Details**: 연구자들은 프롬프트 기반 접근 방식을 사용하여 LLM(Llama2)를 통해 약한 라벨 데이터를 생성하고, 이를 이용해 down-stream BERT 모델을 학습시켰습니다. 이후 소량의 고품질 데이터로 추가 미세조정을 통해 모델의 성능을 더욱 향상시켰습니다. 이 접근법은 n2c2 데이터셋 세 가지를 사용하여 평가되었습니다.

- **Performance Highlights**: 10개의 gold standard 노트만 사용했을 때, Llama2-13B로 약한 감독을 받은 최종 BERT 모델은 기본 제공되는 PubMedBERT의 F1 점수를 4.7%에서 47.9%까지 지속적으로 능가했습니다. 50개의 gold standard 노트만 사용했을 때에도, 이 모델은 완전히 미세 조정된 시스템에 가까운 성능을 보여주었습니다.



### Link Prediction in Bipartite Networks (https://arxiv.org/abs/2406.06658)
Comments:
          28th International Conference on Knowledge-Based and Intelligent Information & Engineering Systems (KES), Sep 2024, Sevilla, Spain

- **What's New**: 이번 연구에서는 이분 그래프(bipartite networks)에서 링크 예측(link prediction)을 위한 19개의 방법을 비교 실험하였습니다. 일부 방법은 기존 문헌에서 가져왔으며, 일부는 단일 네트워크(unipartite networks)를 위해 설계된 기술을 저자들이 이분 그래프로 수정한 것입니다. 추가적으로, 그래프 합성망(Convolutional Networks, GCN)을 기반으로 하는 추천 시스템을 이분 그래프의 새로운 링크 예측 솔루션으로 제안하였습니다.

- **Technical Details**: 이번 연구에서는 다양한 위상구조를 가진 3개의 실제 이분 그래프 데이터셋을 구축하여 실험을 진행하였습니다. 연구에 포함된 19개의 링크 예측 방법에는 기존 연구에서 사용된 방법과, 단일 네트워크를 위해 설계되었지만 이분 그래프로 수정한 방법이 포함되어 있습니다. GCN 기반 개인화 추천 시스템을 통해 링크 예측 성능을 평가하였으며, 또한 학습 프로세스에 의존하지 않는 순수한 휴리스틱 지표(Structural Perturbation Method, SPM) 역시 효과적인 결과를 보였습니다.

- **Performance Highlights**: 결과적으로, GCN 기반 개인화 추천 시스템은 이분 그래프에서 성공적인 링크 예측을 할 수 있음을 보였습니다. 또한 학습 과정이 필요 없는 구조 교란법(SPM)과 같은 순수한 휴리스틱 지표도 성공적으로 링크 예측을 수행했습니다.



### Harnessing AI for efficient analysis of complex policy documents: a case study of Executive Order 14110 (https://arxiv.org/abs/2406.06657)
Comments:
          28 pages, 1 figure

- **What's New**: 정책 문서(legislation, regulations, executive orders)가 사회를 형성하는 데 중요한 역할을 하지만, 그 길이와 복잡성 때문에 해석과 적용이 어렵고 시간이 많이 소요됩니다. 본 연구는 인공지능(AI), 특히 대형 언어 모델(LLMs)이 이러한 문서 분석을 자동화하여 정확성과 효율성을 높일 수 있는 가능성을 평가하는 것을 목적으로 합니다. 특히 정책 문서에서 콘텐츠 추출과 질문 응답 작업에 대한 AI의 잠재력을 조사했습니다. '인공지능의 안전하고, 보안적이고, 신뢰할 수 있는 개발 및 사용'에 관한 행정명령 14110을 사례 연구로 사용하여 네 개의 상업용 AI 시스템이 이 문서를 분석하고 대표적인 정책 질문에 답하도록 했습니다.

- **Technical Details**: 연구는 질문 응답과 콘텐츠 추출 작업에 초점을 맞추어 진행되었습니다. Gemini 1.5 Pro와 Claude 3 Opus 두 AI 시스템이 특히 뛰어난 성능을 보였으며, 복잡한 문서에서 정확하고 신뢰할 수 있는 정보 추출을 제공했습니다. 이들은 인간 분석가와 비교해도 손색이 없었으며, 훨씬 높은 효율성을 나타냈습니다.

- **Performance Highlights**: Gemini 1.5 Pro와 Claude 3 Opus 시스템은 복잡한 정책 문서에서의 정확한 정보 추출 및 분석을 통해 높은 성능을 입증했습니다. 하지만 재현성(reproducibility) 문제는 여전히 해결이 필요하며, 추가적인 연구와 개발이 요구됩니다.



### Anna Karenina Strikes Again: Pre-Trained LLM Embeddings May Favor High-Performing Learners (https://arxiv.org/abs/2406.06599)
Comments:
          9 pages (not including bibliography), Appendix and 10 tables. Accepted to the 19th Workshop on Innovative Use of NLP for Building Educational Applications, Co-located with NAACL 2024

- **What's New**: 학생의 자유 답변을 통해 행동 및 인지 프로파일을 도출하는 머신러닝 기술에서 LLM(대형 언어 모델) 임베딩을 사용한 비지도 클러스터링이 새로운 시도로 연구되고 있습니다. 이 연구는 생물학 수업에서의 학생 답변을 대상으로, 전문 연구자들이 이론 기반의 'Knowledge Profiles(KPs)'로 분류한 결과와 순수한 데이터 기반의 클러스터링 기법의 결과를 비교했습니다. 그 결과, 정답을 포함한 특정 KPs를 제외하고는 대다수의 KPs가 잘 발견되지 않는 '발견 편향(discoverability bias)'이 나타났음을 밝혔습니다.

- **Technical Details**: 학생 답변 데이터를 활용하여 KMeans와 HDBSCAN 같은 일반적인 클러스터링 기법이 이론 기반의 KPs를 발견하는 정도를 평가했습니다. 또한, 'Anna Karenina 원칙'이라는 개념을 도입하여, 답변의 품질(정답 또는 다양한 정도의 오답)과 이들의 임베딩 기반 표현의 형태 및 밀도 사이의 관계를 분석했습니다.

- **Performance Highlights**: 결과적으로, 데이터의 임베딩과 클러스터링 기법이 대부분의 이론 기반 KPs를 발견하는 데 실패했으며, 정답을 포함하는 KPs만이 어느 정도 잘 발견되었습니다. 이는 교육적으로 의미 있는 정보를 유지하는 데 문제가 있을 수 있음을 시사합니다. 중요한 점은, 사전 학습된 LLM 임베딩이 교육적 프로파일 발견의 기초로서 반드시 최적이 아닐 수도 있다는 것입니다.



### Graph Neural Network Enhanced Retrieval for Question Answering of LLMs (https://arxiv.org/abs/2406.06572)
Comments:
          Under review

- **What's New**: 이 논문은 GNN-Ret이라는 새로운 데이터 검색 방법을 제안합니다. GNN-Ret은 그래프 뉴럴 네트워크(GNN)를 활용하여 문단 사이의 관계성을 반영함으로써 검색 성능을 향상시킵니다. 또한, 반복적인 그래프 뉴럴 네트워크(RGNN)를 사용하는 RGNN-Ret을 통해 멀티 홉 추론 질문도 처리할 수 있습니다.

- **Technical Details**: GNN-Ret은 먼저 구조적으로 연관된 문단과 키워드를 공유하는 문단을 연결하여 문단의 그래프를 구성합니다. 그런 다음 GNN을 사용하여 문단 간의 관계를 이용해 검색을 개선합니다. RGNN-Ret은 멀티 홉 질문의 검색을 향상시키기 위해 각 단계에서 이전 단계의 검색 결과를 통합하는 방식으로 동작합니다.

- **Performance Highlights**: 광범위한 실험에서 GNN-Ret은 단일 LLM 쿼리를 통해도 기존 다수 쿼리 기반 방식을 초과하는 높은 정확도를 보여주었습니다. 특히, RGNN-Ret은 2WikiMQA 데이터셋에서 10.4% 이상의 정확도 향상을 달성하며, 최신 성능을 보여주었습니다.



### UMBRELA: UMbrela is the (Open-Source Reproduction of the) Bing RELevance Assessor (https://arxiv.org/abs/2406.06519)
Comments:
          5 pages, 3 figures

- **What's New**: 새로 등장한 UMBRELA 툴킷은 OpenAI의 GPT-4 모델을 사용해 대형 언어 모델(LLM)이 검색 시스템 평가에 필요한 관련성 판단을 자동화하는 오픈 소스 도구입니다. Microsoft Bing의 연구를 재현하고 이를 더욱 확장했습니다. 이 툴킷은 TREC 2024 RAG 트랙에서 사용될 예정입니다.

- **Technical Details**: UMBRELA는 LLM을 이용해 검색 쿼리와 결과 간의 관련성을 평가합니다. 논문에서 제시된 흐름을 재현하기 위해 'zero-shot DNA prompting' 기법을 사용했습니다. TREC 2019-2023의 Deep Learning Tracks 데이터를 사용하여 LLM의 관련성 판단 결과를 인간 평가자와 비교했습니다.

- **Performance Highlights**: UMBRELA의 LLM 기반 관련성 판단은 다단계 검색 시스템의 순위와 높은 상관관계를 보였습니다. 결과적으로 LLM은 인간 평가자와 유사한 수준의 정확성과 신뢰성을 제공함을 확인했습니다. 이러한 성과는 LLM이 더 비용 효율적이고 정확한 대안이 될 수 있음을 뒷받침합니다.



### Survey for Landing Generative AI in Social and E-commerce Recsys -- the Industry Perspectives (https://arxiv.org/abs/2406.06475)
- **What's New**: 최근 생성적 인공지능(GAI)의 등장으로 산업 추천 시스템(Recsys)이 혁신적인 변화를 겪고 있습니다. 이 논문은 사회적 및 전자상거래 플랫폼에서 GAI를 성공적으로 통합한 경험을 바탕으로, GAI와 Recsys의 통합에 대한 실질적인 통찰과 도전과제를 종합적으로 검토합니다. GAI와 Recsys 통합에 관한 실용적인 적용 사례와 해결책을 제시하며 향후 연구 방향을 제시합니다.

- **Technical Details**: 이 논문은 산업 Recsys의 복잡한 인프라, 운영 절차 및 비즈니스 제품 관점을 고려하여 GAI를 통합하는 데 필요한 실제 솔루션 프레임워크를 탐구합니다. 특히, GAI와 LLMOps(Long Language Model Operations) 기초, 맞춤형 추천을 강화를 위한 GAI 유스케이스, 그리고 Recsys 내의 Retrieval-Augmented Generation(RAG) 활용 방법 등을 포괄적으로 다룹니다. Prompt engineering, in-context learning, chain-of-thought와 같은 기법 적용 방법도 상세히 설명됩니다.

- **Performance Highlights**: 논문에서는 사용자 만족도 및 투명성과 신뢰성을 향상시키기 위해 GAI를 활용한 콘텐츠의 재목적화와 외부 지식을 통한 큐레이션이 강조됩니다. 또한, Recsys가 더욱 상호작용적이고 사용자 피드백 루프 기반으로 발전할 수 있는 방향성을 제시합니다. GAI 솔루션의 비용, 지연 시간, 전용 데이터 및 도메인 지식을 효율적으로 사용하기 위한 최적화 방향도 제시됩니다.



### Greedy SLIM: A SLIM-Based Approach For Preference Elicitation (https://arxiv.org/abs/2406.06061)
- **What's New**: 새로운 사용자의 선호도 추정을 위한 방법으로, 최신형 추천 시스템인 SLIM을 기반으로 하는 새로운 접근 방식을 제안합니다. 본 연구는 Greedy SLIM이라는 새로운 학습 기법을 활용해, 새로운 사용자에게 질문할 항목을 선정합니다. 이를 통해 특히 사용자 연구에서 뛰어난 성능을 보인다는 결론을 얻었습니다.

- **Technical Details**: 제안한 Greedy SLIM 방법은 기존 SLIM 학습 방법의 문제를 해결하기 위해 고안되었습니다. SLIM(Scalable Likelihood-based Item Model)은 협업 필터링에서 최적의 상위 N개의 추천을 위해 사용되는 기법입니다. Greedy SLIM은 항목을 하나씩 선택해 SLIM 손실을 최소화하는 방식으로 학습을 진행합니다. 이는 active learning 접근법의 일환으로, 새로운 사용자가 시스템에 입력할 항목을 최적화합니다.

- **Performance Highlights**: 오프라인 실험과 사용자 연구를 통해 Greedy SLIM의 성능을 평가했습니다. 사용자 연구에서는 특히 긍정적인 결과를 보이며, 기존의 잠재 인자 모델(LFM) 기반 방법보다 더 적합한 것으로 나타났습니다. 이는 사용자가 적은 항목만 평가해도 적정한 추천 결과를 얻을 수 있음을 시사합니다.



### Modeling User Retention through Generative Flow Networks (https://arxiv.org/abs/2406.06043)
Comments:
          KDD-ADS 2024

- **What's New**: 이번 연구는 사용자의 재방문 행동(user retention)을 최적화하는 새로운 추천 시스템 프레임워크인 GFN4Retention을 제안합니다. 기존 연구 대부분이 사용자의 즉각적인 피드백을 최대화하는 데 초점을 맞췄다면, 본 연구는 사용자의 세션 간 안정적이고 지속적인 사용을 고려하였습니다.

- **Technical Details**: GFN4Retention은 Generative Flow Networks (GFNs)를 기반으로 한 세션-wise 추천 시스템입니다. 이 프레임워크는 사용자의 세션 종료 시점의 만족도를 추정하고, 이를 기반으로 각 추천 항목에 대해 확률적 흐름(probabilistic flow)을 모델링합니다. 구체적으로, 추천 과정을 조건부 순방향 확률적 흐름으로 간주하고, 각각의 사용자 상태에 대한 흐름 추정기(flow estimator)를 활용합니다.

- **Performance Highlights**: GFN4Retention은 두 개의 공공 데이터셋과 실제 산업 플랫폼에서의 온라인 A/B 테스트를 통해 검증되었습니다. 기존의 강화 학습 기반 추천 모델들과 비교하여 우수한 성능을 보였으며, 각 구성 요소의 효과를 분석한 에이블레이션 연구에서도 높은 안정성을 나타냈습니다.



### A WT-ResNet based fault diagnosis model for the urban rail train transmission system (https://arxiv.org/abs/2406.06031)
Comments:
          12 pages,10 figures

- **What's New**: 새로운 연구는 도시 철도 시스템의 고장 진단을 위한 혁신적인 모델을 제안합니다. 이 모델은 웨이블릿 변환(Wavelet Transform)과 잔차 신경망(Residual Neural Network, ResNet)의 장점을 통합하여 진단 정확도와 강건성을 높였습니다.

- **Technical Details**: 제안된 모델은 웨이블릿 변환을 사용하여 도시 철도의 특징을 추출하고, ResNet을 통해 패턴 인식을 수행합니다. 이는 딥러닝(DL) 알고리즘 가운데 높은 성과를 보이는 ResNet을 활용한 것입니다. 또한, 기존의 CNN(Convolutional Neural Network), RNN(Recurrent Neural Network)와의 비교 및 다양한 데이터 세트에 대한 적응력을 강조합니다.

- **Performance Highlights**: 실험 결과, 제안된 WT-ResNet 모델은 도시 철도 열차의 고장을 식별하는 데 있어 높은 효율성과 정확도를 입증했습니다. 이는 유지보수 전략을 개선하고 가동 중단 시간을 줄이는 데 기여할 수 있습니다.



### Weighted KL-Divergence for Document Ranking Model Refinemen (https://arxiv.org/abs/2406.05977)
- **What's New**: 이 논문은 트랜스포머 기반의 문서 검색 및 랭킹 모델을 위한 새로운 학습 손실 함수, 즉 대조적으로 조정된 KL 다이버전스(contrastively-weighted KL divergence, CKL)를 제안합니다. 기존의 KL 다이버전스 기반 지식 증류(loss)를 개선하여 문서 탐색 모델의 성능을 높이는 것을 목표로 합니다.

- **Technical Details**: CKL은 기존의 KL 다이버전스 대신, 긍정(positive) 문서와 부정(negative) 문서 간의 구분을 명확히 하기 위해 대조 학습(contrastive learning)과 결합된 새로운 방법론입니다. 이 방법은 문서 랭킹의 타이트한 분포 매칭(tight distribution matching)에 따른 과도한 보정(over-calibration) 문제를 해결하는 데 중점을 둡니다. MS MARCO와 BEIR 데이터셋을 사용한 평가에서 CKL의 유효성을 입증했습니다.

- **Performance Highlights**: CKL은 기존 KL 다이버전스 및 최근 BKL 접근법과 비교해 문서 검색의 성능을 효과적으로 향상시켰습니다. 실험 결과, CKL을 적용한 학생 모델은 긍정 문서와 부정 문서의 적절한 분포를 유지하면서도 랭킹의 관련성을 크게 높였습니다.



### Async Learned User Embeddings for Ads Delivery Optimization (https://arxiv.org/abs/2406.05898)
Comments:
          Accepted by workshop on Multimodal Representation and Retrieval at SIGIR 2024, Washington DC

- **What's New**: Meta 플랫폼에서 수십억 명의 사용자들을 대상으로 한 고품질 사용자 임베딩(user embeddings)을 비동기적으로 학습하는 방식을 제안합니다. 이 임베딩은 사용자 유사도 그래프(user similarity graphs)로 변환되어, 실시간 사용자 활동과 결합되어 광고 추천에 활용됩니다.

- **Technical Details**: 사용자 임베딩은 다중 모드(multimodal) 사용자 활동 신호를 기반으로 하는 시퀀스(sequence)에서 학습되며, 이는 Transformer와 유사한 구조로 처리됩니다. 다양한 사용자 활동에는 클릭한 광고, 댓글, 조회한 사진 및 동영상 등이 포함됩니다. 이러한 임베딩은 비동기적으로 업데이트되어, 수십억 명의 사용자 데이터를 처리할 수 있습니다. 또한, 사용자 유사도 그래프는 이 데이터를 기반으로 생성되며, 광고 추천에 더욱 효과적으로 활용됩니다.

- **Performance Highlights**: 제안된 모델은 오프라인 및 온라인 실험 모두에서 유의미한 성능 향상을 보였습니다. 특히, 사용자의 최신 상호작용과 피드백을 기반으로 사용자 임베딩을 지속적으로 업데이트하고 정제할 수 있는 능력을 갖추고 있습니다. 추가로, 사용자 임베딩은 대규모 모델에서 효율적인 스토리지와 컴퓨팅 절약을 위해 압축할 수 있습니다.



### Prioritizing Potential Wetland Areas via Region-to-Region Knowledge Transfer and Adaptive Propagation (https://arxiv.org/abs/2406.05578)
- **What's New**: 새로운 연구는 데이터 부족 문제를 해결하여 습지 식별 및 우선순위 설정을 돕기 위한 두 가지 전략을 제안합니다. 첫 번째 전략은 습지가 풍부한 지역에서 데이터가 희소한 지역으로 지식을 전달하는 것입니다. 두 번째 전략은 적응형 전파 메커니즘을 사용하는 공간 데이터 보강 전략입니다.

- **Technical Details**: 제안하는 접근법은 두 가지 주요 기술을 포함합니다. (1) 도메인 비디오 연습(domain disentanglement) 전략을 사용하여 풍부한 습지 지역의 지식을 데이터를 통해 전달합니다. 이 과정에서는 도메인별로 동일하게 적용 가능한 정보만 선택적으로 전이하고, 도메인 분리기로 도메인별 정보와 공유 가능한 정보를 분리합니다. (2) Graph Neural Networks (GNNs)의 적응형 전파 메커니즘을 도입하여 인접 노드들의 상호 영향을 구분하는 방식입니다. 이는 지역 내 세포 간의 유용한 정보를 차별화하여 전달합니다.

- **Performance Highlights**: 제안된 방법의 효과, 강인성 및 확장성을 입증하기 위해 엄격한 실험을 수행했습니다. 이를 통해 제안된 방법이 기존 최신 기준선을 능가하며, 모듈별 실험을 통해 각 모듈이 기존 습지 식별에 필수적임을 보여줍니다.



### I-SIRch: AI-Powered Concept Annotation Tool For Equitable Extraction And Analysis Of Safety Insights From Maternity Investigations (https://arxiv.org/abs/2406.05505)
- **What's New**: I-SIRch라는 새로운 접근 방식이 소개되었습니다. I-SIRch는 인공지능과 머신러닝 알고리즘을 활용하여 영국의 Healthcare Safety Investigation Branch(HSIB)에서 생산된 조사 보고서에서 발생한 임신 관련 사고에 대한 인간 요인(human factors)을 자동으로 식별하고 레이블을 지정합니다. 이는 생물의학적 개념에만 초점을 맞추는 기존 도구와 차별화되며, 인간 요인을 포함하여 의료 제공 시스템을 더 잘 이해하는 데 기여합니다.

- **Technical Details**: I-SIRch는 SIRch 택소노미를 사용하여 의료 조사로부터 안전 통찰을 추출합니다. 데이터 전처리 단계에서 PDF로 제공된 보고서를 텍스트로 추출하고, 특정 기준(예: 섹션, 페이지, 단락)에 따라 텍스트를 추출합니다. 수동으로 태그된 데이터를 사용해 모델을 학습시키고, 새로운 보고서를 처리하면서 인간 전문가의 계속된 주석을 통해 모델 성능을 향상시키는 'human-in-the-loop' 방식을 채택했습니다.

- **Performance Highlights**: I-SIRch는 실제 및 합성 데이터를 통해 연구되었습니다. 818개의 합성 문장과 97개의 실제 보고서에서 추출한 1960개의 문장을 테스트한 결과, 실제 보고서 문장에서 90%의 정확도로 관련 개념을 올바르게 식별했습니다. 이를 통해 다양한 인종 그룹에 따라 특정 인간 요인이 어떻게 차이 나는지 분석할 수 있었으며, 이는 사회적, 기술적 및 조직적 요인이 산모 안전과 인구 건강 결과에 미치는 복잡한 상호작용을 이해하는 데 새로운 가능성을 열어주었습니다.



### PTF-FSR: A Parameter Transmission-Free Federated Sequential Recommender System (https://arxiv.org/abs/2406.05387)
- **What's New**: 최근 사용자 데이터 프라이버시에 대한 우려가 증가함에 따라 연구자들은 사용자 데이터 프라이버시를 보호하면서 협력 학습을 구현하는 Federated Sequential Recommender Systems(FedSeqRecs)라는 접근 방식을 제안했습니다. 하지만 이는 모델 공유와 높은 통신 비용이라는 두 가지 큰 단점이 존재했죠. 이번에 제안된 연구는 그 단점을 극복하기 위해 모델 파라미터 전송 없이 협력 학습을 가능하게 하는 새로운 프레임워크, 'Parameter Transmission-Free Federated Sequential Recommendation (PTF-FSR)'을 소개합니다.

- **Technical Details**: PTF-FSR은 기존의 파라미터 교환 방식 대신 예측 결과만을 전송하여 모델 크기와 상관없이 통신 비용을 대폭 줄일 수 있습니다. 또한, 사용자 프라이버시 보호를 위해 클라이언트 측에서 사용자의 원본 항목 상호작용 시퀀스에 섭동을 추가하는 지수 기법(exponential mechanism)을 사용합니다. 이 프레임워크는 ID-based와 ID-free 파라다임을 아우르는 여러 순차적 추천 모델에서 효과적으로 적용되었습니다.

- **Performance Highlights**: 세 가지 널리 사용되는 추천 데이터셋에서 다양한 ID-based 및 ID-free 순차적 추천 모델을 테스트한 결과, PTF-FSR은 높은 성능과 일반화 능력을 보였습니다. 이는 보다 복잡하고 큰 순차적 추천 모델도 수용할 수 있는 새로운 연합 학습 구조의 가능성을 보여줍니다.



### Measuring Fairness in Large-Scale Recommendation Systems with Missing Labels (https://arxiv.org/abs/2406.05247)
- **What's New**: 이번 연구에서는 대규모 추천 시스템에서 발생하는 공정성 문제를 다룹니다. 특히, 추천된 항목에 대한 레이블(label)이 없는 상황에서의 공정성 문제를 해결하기 위한 새로운 방법을 제안합니다. 이 연구는 랜덤 트래픽(randomized traffic)을 활용하여 공정성 지표를 더 정확하게 추정하는 방법을 제시하며, 이를 증명하기 위해 TikTok의 실제 데이터를 사용한 실험 결과를 제공합니다. 또한, TikTok의 공정성 관련 데이터셋을 처음으로 공개하며, 추천 시스템의 데이터셋 수집 방법론에 대한 새로운 기준을 제시합니다.

- **Technical Details**: 이 연구는 'Ranking-based Equal Opportunity (REO)'라는 공정성 개념에 기반하여 진행됩니다. REO는 사용자-아이템(user-item) 상호작용이 완전히 관찰된 경우의 공정성 문제를 다루며, 사용자의 진정한 선호도를 알 수 없는 대규모 추천 시스템에서의 공정성 문제를 해결합니다. 연구팀은 랜덤 트래픽 데이터를 사용하여 공정성 메트릭의 오차 한계를 이론적으로 제시하며, 이를 TikTok의 실제 데이터로 검증했습니다.

- **Performance Highlights**: 실험 결과, 제안된 랜덤 트래픽 활용 방법이 기존의 간단한 방법들보다 훨씬 정확하고 효율적임을 보여줍니다. 특히, TikTok의 실제 데이터셋과 합성 데이터(synthetic data)를 통해 제안된 방법의 이론적 타당성과 실효성을 입증했습니다. 또한, 랜덤 트래픽 데이터를 사용하여 대규모 추천 시스템의 공정성 지표를 정확하게 추정하는 것이 필수적임을 확인했습니다.



### Evaluating the Retrieval Component in LLM-Based Question Answering Systems (https://arxiv.org/abs/2406.06458)
- **What's New**: 이 연구는 Retrieval-Augmented Generation (RAG) 기반의 챗봇 시스템에서 리트리버 (retriever) 성능을 평가하는 간단한 기준을 제안합니다. 기존 평가지표들이 대형 언어 모델(LLMs)의 능력을 완전히 포착하지 못하는 상황에서, 새로운 평가 프레임워크는 챗봇의 전체 성능과 더 잘 맞아떨어지는 평가를 제공합니다.

- **Technical Details**: RAG 모델을 활용한 QA 시스템은 두 가지 주요 구성 요소로 나뉘어집니다. 리트리버가 문서 코퍼스에서 관련 정보를 검색하고, 생성기(generator)가 이 문서를 바탕으로 응답을 생성합니다. 기존 평가 방법이 주석된 데이터에만 집중하는 한계를 극복하기 위해, 우리는 이상적인 리트리버와 실제 리트리버의 출력을 비교하여 downsteam 효과까지 고려하는 새로운 평가 방식을 제안합니다. 이를 위해 Exact Match (EM), 토큰 기반 지표(ROUGE, BLEU, METEOR), 임베딩 기반 지표(BERTScore), 그리고 LLM-기반 평가 방법을 사용합니다.

- **Performance Highlights**: 새로운 리트리버 평가 방법론은 기존의 정밀도(Precision), 재현율(Recall), F1 점수보다 LLM 기반 QA 시스템 성능을 더 잘 반영합니다. NQ-open 코퍼스 실험 결과, 새로운 방법론이 리트리버의 유효성을 더 잘 포착하였고, 기존 지표와 높은 상관관계를 나타냈습니다.



### Combining Embeddings and Domain Knowledge for Job Posting Duplicate Detection (https://arxiv.org/abs/2406.06257)
Comments:
          To be published at 9th International Symposium on Language & Knowledge Engineering LKE 2024

- **What's New**: 이 논문은 여러 플랫폼에 게시된 구인 공고에서 중복 되는 공고를 탐지하는 새로운 접근 방식을 제안합니다. 문자 유사성 및 텍스트 임베딩(text embedding), 키워드 매칭(keyword matching) 방법을 결합하여 성능을 향상시켰습니다. 이 접근 방식은 실제로 사용되고 있으며 긍정적인 피드백을 받고 있습니다.

- **Technical Details**: 중복 탐지에는 문자열 비교, 딥 텍스트 임베딩, 특정 기술에 대한 가중치 목록을 사용한 조합 방식이 사용되었습니다. 각 방법을 개별적으로 사용할 때 성능이 만족스럽지 않지만, 이들을 결합하면 높은 성능과 낮은 오탐률(false positives)을 보입니다.

- **Performance Highlights**: 실제 사용 사례에서 새로운 접근 방식이 높은 성능을 보였으며, 수작업으로 진행하던 중복 탐지 작업을 자동화하는 데 성공적이었습니다. 이로 인해 개발 및 운영 비용도 절감되었습니다.



### Black carbon plumes from gas flaring in North Africa identified from multi-spectral imagery with deep learning (https://arxiv.org/abs/2406.06183)
Comments:
          Published at the workshop Tackling Climate Change with Machine Learning at ICLR 2024

- **What's New**: 이 논문에서는 인공지능(deep learning) 프레임워크를 사용하여 인공위성 이미지 (satellite imagery)를 통해 북아프리카 지역에서 가스 플레어링 (gas flaring)으로 인한 블랙 카본(BC) 배출량을 직접 모니터링하는 방법을 소개합니다. 2022년 동안 모니터링된 결과, 이 지역의 BC 배출량은 약 백만 톤 탄소 동등 (tCO$_{2,	ext{eq}}$)에 이르렀습니다.

- **Technical Details**: 유럽 우주국(ESA)의 Sentinel-2 위성 데이터를 사용하여 알제리, 리비아 및 이집트에서 가스 플레어링 사이트를 분석했습니다. ConvLSTM 모델을 사용하여 위성 이미지에서 BC 플룸(plume)을 감지하고 분류했습니다. 이 모델은 두 RGB 이미지 시퀀스를 입력으로 받아 두 번째 이미지에서 BC 플룸을 세그먼트화(segmentation)하는 작업을 수행합니다. 모델 훈련은 Synthetic BC Plumes를 포함한 Sentinel-2 데이터로 이루어졌습니다. 추가적으로, LightGBM 분류기를 사용하여 거짓 양성(false positive)을 필터링했습니다.

- **Performance Highlights**: 2022년 동안 1963개의 개별 플레어(flares)를 감지했으며, 대부분은 짧은 시간 동안 활동했습니다. 가스 플레어링 사이트 중 약 10곳이 전체 BC 배출량의 25% 이상을 차지했습니다. 이는 효율적인 감지 및 완화 정책을 구현하기 위한 중요한 발걸음입니다.



### Thanking the World: Exploring Gender-Based Differences in Acknowledgment Patterns and Support Systems in Theses (https://arxiv.org/abs/2406.06006)
- **What's New**: 이 연구는 전자 논문 및 학위 논문(Electronic Theses and Dissertations, ETDs)에서 지원 시스템을 조사하여 학위 과정 중 연구자들이 어떤 형태의 지원을 받았는지 분석하였습니다. 특히 도서관 및 정보 과학(Library and Information Science) 분야를 대상으로 했습니다. RoBERTa 기반 모델을 사용하여 1252개의 ETD를 분석한 것은 이번 연구의 주요 기여입니다.

- **Technical Details**: 본 연구에서는 RoBERTa 모델을 사용하여 ETD 감사 섹션에서 다양한 지원 형태를 추출했습니다. 이를 통해 연구자들이 인식하는 주요 지원 유형은 학문적 지원(academic support), 도덕적 지원(moral support), 재정적 지원(financial support), 그리고 종교적 지원(religious support)이었음을 확인했습니다.

- **Performance Highlights**: 연구 결과, 종교적 및 재정적 지원에서는 성별 차이가 거의 없었으나, 학문적 지원과 도덕적 지원의 비율에서는 큰 성별 차이가 발견되었습니다. 특히 지도 교수들은 동성 연구자를 선호하는 경향이 있음을 보여주었습니다. 이 연구는 성별 간의 차이를 이해하고 포용적이며 지원적인 학문 환경을 조성하는 데 중요한 시사점을 제공합니다.



### Explainable AI for Mental Disorder Detection via Social Media: A survey and outlook (https://arxiv.org/abs/2406.05984)
- **What's New**: 이 논문은 데이터 과학과 인공지능(AI)을 이용한 정신 건강 관리의 최신 동향을 조사하며, 특히 온라인 소셜 미디어(OSM)를 통한 정신 질환 감지에 중점을 둡니다. 특히 Explainable AI (XAI) 모델의 중요성을 강조하며, 기존의 진단 방법과 최신 인공지능 구동 연구를 종합적으로 리뷰합니다.

- **Technical Details**: 정신 건강 진단에는 전통적으로 DSM-5와 같은 국제 표준을 기반으로 한 대면 인터뷰와 자기보고식 질문지가 사용됩니다. 최근에는 OSM 데이터를 활용한 딥러닝(Deep Learning) 기술을 통한 감지 모델도 연구되고 있는데, 이러한 모델의 설명 가능성(Explainability)을 높이는 작업이 필요합니다. 또, 이 논문은 기존 문헌에서 설명 가능성과 사회적 상호작용의 중요성을 간과하는 문제를 지적합니다.

- **Performance Highlights**: 다양한 최신 머신 러닝 및 딥러닝 모델을 검토한 결과, 크게 발전한 모델로는 DepressionNet과 EDNet 등이 있으며, 이러한 모델은 정신 질환의 조기 진단에 유망한 도구로 평가됩니다. 그러나, 블랙박스 모델의 활용은 의료 결정에서 안전성 문제를 야기할 수 있어, 설명 가능한 AI 모델로의 전환이 필요합니다.



### General Distribution Learning: A theoretical framework for Deep Learning (https://arxiv.org/abs/2406.05666)
Comments:
          arXiv admin note: text overlap with arXiv:2105.04026 by other authors. arXiv admin note: text overlap with arXiv:2105.04026 by other authors

- **What's New**: 새로운 연구 논문에서는 딥 러닝(deep learning, DL)에서 아직 해결되지 않은 여러 중요한 질문들을 다루기 위해 GD Learning이라는 새로운 이론적 학습 프레임워크를 도입합니다. 이 프레임워크는 분류(classification), 회귀(regression), 파라미터 추정(parameter estimation)을 포함한 다양한 머신 러닝 및 통계적 과제를 해결하는 데 초점을 맞추고 있습니다. 특히, GD Learning은 데이터 부족 상황에서 외부 지식을 통합하여 학습 오류를 최소화하는 것을 목표로 합니다.

- **Technical Details**: GD Learning 프레임워크는 학습 오류를 모델 및 학습 알고리즘에 의한 피팅 오류(fitting errors)와 제한된 샘플링 데이터로 인한 샘플링 오류(sampling errors)로 나눌 수 있습니다. 또한, 비정형(non-uniformity) 자코비안 행렬(Jacobian matrix)의 고유값(eigenvalues)를 이용하여 Gradient 구조 제어 알고리즘(Gradient Structure Control Algorithm)을 통해 글로벌 최적 해(global optimal solution)에 접근할 수 있음을 보여줍니다. 이러한 구조는 비콘백스(non-convex) 최적화 문제들, 예를 들어 피팅 오류 최소화에서도 사용될 수 있습니다.

- **Performance Highlights**: GD Learning은 과적합(overparameterization), 비콘백스 최적화(non-convex optimization), 편향-분산 트레이드오프(bias-variance trade-off)와 평평한 최소값(flat minima)의 메커니즘 등 딥 러닝에서의 해결되지 않은 질문들에 대해 새로운 통찰을 제공합니다. 기존의 통계적 학습 이론과는 다르게 진정한 기본 분포(underlying distribution)에 초점을 맞추면서, 성능을 향상시킬 수 있는 실질적인 방법을 제시합니다.



### DomainRAG: A Chinese Benchmark for Evaluating Domain-specific Retrieval-Augmented Generation (https://arxiv.org/abs/2406.05654)
- **What's New**: 최근 발표된 논문에서는 Retrieval-Augmented Generation (RAG) 모델이 대형 언어 모델(LLMs)의 한계를 극복하는 해결책으로 주목받고 있습니다. 특히 전문가와 도메인 별 애플리케이션에서 RAG 모델의 필요성이 강조되었습니다. 해당 논문에서는 대학 입학 시나리오에서 RAG 모델의 성능을 평가하였으며, 여섯 가지 필수 능력을 분석하였습니다.

- **Technical Details**: RAG 시스템을 이해하기 위해 논문에서 제시된 여섯 가지 주요 능력은 다음과 같습니다: 1) 대화형 RAG에서의 능력, 2) 구조적 정보 분석, 3) 외부 지식의 신뢰성, 4) 노이즈 제거(denoising), 5) 시간에 민감한 문제 해결, 6) 다중 문서 상호작용 이해. 이 능력들을 평가하기 위해 각 능력에 대응하는 데이터셋이 제공되었습니다. 평가된 모델은 Llama, Baichuan, ChatGLM, GPT 등입니다.

- **Performance Highlights**: 실험 결과 기존의 'closed-book' LLM는 도메인 별 질문에 대처하는 데 어려움을 겪었으며, 이는 RAG 모델이 전문가 문제를 해결하는 데 필요하다는 것을 강조합니다. 또한, 대화 히스토리 이해, 구조적 정보 분석, 노이즈 제거, 다중 문서 상호작용, 외부 지식의 신뢰성 측면에서 RAG 모델의 향상 가능성이 있는 것으로 나타났습니다.



### Separating the "Chirp" from the "Chat": Self-supervised Visual Grounding of Sound and Languag (https://arxiv.org/abs/2406.05629)
Comments:
          Computer Vision and Pattern Recognition 2024

- **What's New**: DenseAV는 비디오만을 통해 고해상도, 의미적으로 의미 있는, 오디오-비주얼(AV) 정렬 피처(feature)를 학습하는 새로운 듀얼 인코더(dense encoder) 그라운딩 아키텍처를 도입했습니다. 이 시스템은 명확한 위치 지정 감독 없이도 단어의 '의미'와 소리의 '위치'를 발견할 수 있습니다. 또한, 두 가지 유형의 연관성을 자동으로 발견하고 구분합니다.

- **Technical Details**: DenseAV는 새로운 multi-head feature aggregation 연산자를 사용하여, 밀집된 이미지와 오디오 표현을 대조 학습(contrastive learning)을 통해 직접 비교합니다. 이를 통해 DenseAV는 음성과 비주얼 신호 간의 높은 품질의 지역 표현을 학습합니다. 또한, DenseAV는 두 개의 새로운 데이터셋을 도입해 음성과 소리 기반의 의미 분할(semantic segmentation)을 평가합니다. 이 데이터셋은 ADE20K 데이터셋에서 제공하는 고품질 분할 마스크를 기반으로 구축되었습니다.

- **Performance Highlights**: DenseAV는 음성과 소리 기반의 의미 분할에서 이전의 최첨단 기술인 ImageBind를 크게 능가합니다. 또한, DenseAV는 동일한 작업에서 ImageBind의 절반 이하의 매개변수만을 사용하면서 뛰어난 성능을 보입니다. 이로 인해 DenseAV는 새로운 소리와 자원이 적은 언어에 대한 적용 가능성을 보유하게 됩니다.



### Toward Reliable Ad-hoc Scientific Information Extraction: A Case Study on Two Materials Datasets (https://arxiv.org/abs/2406.05348)
- **What's New**: 이번 연구는 GPT-4가 과학 문헌에서 adhoc 스키마 기반 정보 추출을 수행할 수 있는 능력을 탐구합니다. 기존의 수작업으로 추출된 두 가지 재료 과학 데이터셋을 재현할 수 있는지 평가하며, 모델이 희망하는 정보를 정확히 추출하는 데 어려움을 겪는 부분을 구체적으로 분석하였습니다.

- **Technical Details**: 연구는 두 가지 전문가가 수작업으로 추출한 재료 특성 데이터셋을 사용했습니다. 하나는 다중-주요 원소 합금(MPEAs)에 관한 것이고, 다른 하나는 실리케이트 녹아내림의 요소 확산에 관한 것입니다. 모델의 성능을 평가하기 위해 재료 과학자들이 오류 분석을 수행했습니다. GPT-4는 스키마에 따라 데이터를 추출하고, 내러티브나 기존의 표에서 잘 작동했지만, 그래프와 PDF 파싱 이슈에서 많은 오류가 발생했습니다.

- **Performance Highlights**: GPT-4는 내러티브나 표 형식에서 정보를 상당히 잘 추출하는 능력을 보였지만, 그래프와 PDF 파싱 문제에서 상당한 오류를 보였습니다. 추가적으로, 비표준 표 형식, 추출된 값의 후처리 필요성, 그리고 향상된 프롬프트 엔지니어링이 요구되는 진정한 읽기 이해 오류도 주요 오류 원인이었습니다.



### TLEX: An Efficient Method for Extracting Exact Timelines from TimeML Temporal Graphs (https://arxiv.org/abs/2406.05265)
Comments:
          25 pages, 9 figures

- **What's New**: 이번 연구에서는 TimeML 주석(texts)로부터 완전한 이벤트 타임라인을 추출하는 TLEX (TimeLine EXtraction)이라는 새로운 시스템을 개발했습니다. TLEX는 기존의 타임라인 추출 방법들보다 정확하며, 특히 이벤트의 불일치 및 불확정 섹션을 자동으로 식별하는 두 가지 새로운 기능을 추가했습니다.

- **Technical Details**: TLEX는 TimeML 주석을 트렁크와 브랜치 구조로 배열된 타임라인 컬렉션으로 변환합니다. 기존 작업과 마찬가지로, TLEX는 시간 그래프의 일관성을 검사하고 정렬합니다. 또한, 특정 관계가 불일치를 초래하는지 식별하고, 타임라인의 불확정 섹션을 식별할 수 있습니다. 이는 자연어 처리 및 이벤트 정렬 작업에 중요한 정보입니다.

- **Performance Highlights**: TLEX는 네 개의 코퍼스로부터 385개의 TimeML 주석 텍스트에 적용되어 실험적 평가를 거쳤으며, 123개의 텍스트가 불일치 상태였으며, 181개 텍스트는 여러 '실제 세계' 또는 주요 타임라인을 가지고 있고, 총 2,541개의 불확정 섹션이 발견되었습니다. 샘플링 평가 결과 TLEX는 다섯 가지 차원에서 98-100%의 정확도를 가지고 있음이 입증되었습니다: 타임포인트의 정렬, 주요 타임라인 수, 주요 및 부속 타임라인의 타임포인트 배치, 브랜치 타임라인의 연결 포인트, 불확정 섹션의 위치.



### Corpus Poisoning via Approximate Greedy Gradient Descen (https://arxiv.org/abs/2406.05087)
- **What's New**: 새로운 연구는 정보 검색 시스템에서의 코퍼스(poisoning attack)를 효과적으로 실행할 수 있는 새로운 공격 방법인 'Approximate Greedy Gradient Descent (AGGD)'을 제안합니다. 이 연구는 기존 HotFlip 방법의 한계를 극복하고, 더 구조적인 검색을 통해 더 높은 질의 토큰 수준의 변형을 선택할 수 있음을 보입니다.

- **Technical Details**: AGGD는 랜덤하게 토큰을 샘플링하는 대신 모든 토큰 위치에서 최상위 토큰을 선택하여 점진적 경사 하강법(Greedy Gradient Descent)을 사용합니다. 이는 AGGD의 검색 궤적을 결정적(deterministic)으로 만들어 더 구조적인 최선-우선 검색(best-first search)을 가능하게 합니다. 실험 결과, AGGD는 NQ와 MS MARCO 데이터셋에서 기존 HotFlip보다 각각 17.6% 및 13.37% 높은 공격 성공률을 기록했습니다.

- **Performance Highlights**: AGGD는 여러 데이터셋과 검색 모델에서 높은 공격 성공률을 달성했습니다. 특히 ANCE 검색 모델을 공격할 때, NQ 데이터셋 코퍼스의 단 0.00003%, MS MARCO 데이터셋의 0.00001%에 해당하는 하나의 적대적 패세지를 주입함으로써, NQ 데이터셋에서 44.35%, MS MARCO 데이터셋에서 26.16%의 공격 성공률을 보여주었습니다. 또한 AGGD는 다른 도메인의 새로운 질의에 대해서도 82.28%의 공격 성공률을 기록했습니다.



### CHIQ: Contextual History Enhancement for Improving Query Rewriting in Conversational Search (https://arxiv.org/abs/2406.05013)
- **What's New**: 이 논문에서는 오픈소스 대형 언어 모델(LLMs)을 효과적으로 활용하여 대화형 검색에서 모호한 쿼리를 개선하는 방법을 연구합니다. 새로운 'CHIQ' 방법을 도입하여, 대화 기록에서 모호성을 해결한 후 쿼리를 재작성하는 두 단계 방식을 제안합니다. 이는 주로 폐쇄형 LLMs를 사용하는 기존 연구들과는 대조적입니다. 5개의 주요 벤치마크에서 CHIQ가 대부분의 설정에서 최첨단 성능을 보임을 입증했습니다.

- **Technical Details**: CHIQ는 대화 기록의 모호성을 해결하기 위해 NLP 과제 해결 능력을 갖춘 LLM을 사용합니다. 본 연구에서는 LLaMA-2-7B와 같은 오픈소스 LLM을 사용하여, 컨텍스트를 확장하거나 코리퍼런스(coreference) 관계를 해결하고 대화 기록을 개선해 쿼리의 적절성을 높입니다. 이처럼 개선된 대화 기록을 기존 프레임워크에 통합하는 다양한 방법을 조사하였습니다.

- **Performance Highlights**: 광범위한 실험 결과, CHIQ는 밀집(dense) 및 희소(sparse) 검색 설정에서 대부분의 벤치마크에서 최첨단 성능을 달성하였습니다. 폐쇄형 LLM과 비교했을 때, 개선된 대화 기록을 사용할 때 성능 격차가 상당히 좁아졌습니다. 이는 오픈소스 LLM이 상업용 모델과 경쟁할 수 있는 가능성을 보여줍니다.



### QAGCF: Graph Collaborative Filtering for Q&A Recommendation (https://arxiv.org/abs/2406.04828)
- **What's New**: Q&A 플랫폼의 새로운 추천 모델, QAGCF(Graph Collaborative Filtering)가 제안되었습니다. 이 모델은 기존 추천 시스템의 한계를 극복하고 질문과 답변 쌍의 협업 및 의미적 정보를 효과적으로 분리하여 사용자의 클릭 행동을 더 정확하게 예측합니다.

- **Technical Details**: QAGCF는 그래프 신경망(neural network) 모델을 기반으로하여 협업 뷰와 의미 뷰를 분리하여 각각의 협업 및 의미 정보를 분리합니다. 협업 뷰에서는 사용자가 클릭한 질문과 답변을 개별적으로 모델링하며, meaning view에서는 질문과 답변 사이, 그리고 질문-답변 쌍들 간의 의미적 연결을 캡처합니다. 이 두 뷰는 글로벌 그래프로 결합되어 전체적인 협업 및 의미 정보를 통합합니다. 글로벌 그래프에서 고차 동조성(high heterophily) 문제를 해결하기 위해 다항식 기반 그래프 필터(polynomial-based graph filters)를 사용하며, 강건한 임베딩(robust embedding)을 얻기 위해 대조 학습(contrastive learning)도 활용합니다.

- **Performance Highlights**: 산업 및 공개 데이터셋에 대한 광범위한 실험 결과 QAGCF가 지속적으로 기존 방법들을 능가하고 최첨단 성과를 달성함을 입증하였습니다.



### Scaling Automatic Extraction of Pseudocod (https://arxiv.org/abs/2406.04635)
- **What's New**: 본 연구에서는 약 32만 개의 가짜 코드(pseudocode) 예제를 포함한 대규모 컬렉션이 제공되었습니다. 이 컬렉션은 arXiv 논문에서 추출된 것으로, 이는 알고리즘 이해를 높이고 자동 코드 생성 및 Optical Character Recognition (OCR) 등의 작업에 유용할 수 있습니다. arXiv 논문 220만 편을 스캔하였으며, 그 중 1,000편은 수작업으로 점검 및 레이블링되었습니다.

- **Technical Details**: 가짜 코드 추출을 위해 arXiv 논문의 LaTex 파일 및 PDF 파일을 분석하는 메커니즘을 개발했습니다. LaTex 파일에서는 명령어를 통해 상대적으로 쉽게 가짜 코드를 추출할 수 있으나, PDF 파일에서는 텍스트와 그림의 경계를 감지하고 이를 추출하는 것이 복잡한 작업입니다. 이를 위해 머신 러닝 기반의 도구가 사용되었습니다.

- **Performance Highlights**: 통계 분석 결과, arXiv 논문에서 가짜 코드 사용이 지수적 증가를 보이고 있음을 밝혔습니다. 또한, 가짜 코드의 클러스터링과 주제별 분석을 통해 다양한 가짜 코드 구조를 조사했습니다.



### Better Late Than Never: Formulating and Benchmarking Recommendation Editing (https://arxiv.org/abs/2406.04553)
- **What's New**: 이번 논문에서는 'recommendation editing(추천 편집)'이라는 새로운 과제를 제안합니다. 이는 기존의 추천 시스템이 제공하는 부적절한 추천을 수정하는 방법으로, 기존 모델을 재학습(retraining)하거나 원본 학습 데이터를 접근하지 않고도 부적절한 아이템을 제거하는 데 중점을 둡니다.

- **Technical Details**: 추천 편집 문제는 세 가지 주요 목표를 정의합니다: (1) 엄격한 수정(Strict Rectification)은 중대한 문제를 유발하는 부적절한 추천 아이템을 제거하는 것입니다. (2) 협력적 수정(Collaborative Rectification)은 관찰되지 않은 유사한 부적절한 추천 아이템도 제거하는 것입니다. (3) 집중적 수정(Concentrated Rectification)은 적절한 추천이 대부분 유지되도록 하는 것입니다. 이를 위해, 새로운 'Editing Bayesian Personalized Ranking Loss'를 기반으로 하는 간단하지만 효과적인 기준점을 제안합니다.

- **Performance Highlights**: 제안된 방법의 효과를 입증하기 위해 다양한 관련 분야의 방법들을 통합한 포괄적인 벤치마크를 설립했습니다. 이를 통해, 제안된 추천 편집 방법이 부적절한 추천 문제를 완화하는 데 실질적으로 효과적임을 보여주었습니다.



### Innovations in Cover Song Detection: A Lyrics-Based Approach (https://arxiv.org/abs/2406.04384)
Comments:
          6 pages, 3 figures

- **What's New**: 이 논문에서는 커버 곡(cover song)을 자동으로 식별하는 새로운 방법을 제안합니다. 특히, 기존의 오디오 분석에만 의존하는 접근법과 달리, 이 방법은 곡의 가사를 활용합니다. 이를 위해 새로운 데이터셋을 구축했으며, 이 데이터셋에는 5078개의 커버 곡과 2828개의 원곡이 포함되어 있습니다. 모든 곡에는 주석이 달린 가사가 첨부되어 있습니다.

- **Technical Details**: 제안된 방법은 Levenshtein 거리와 단어 오류율(WER)을 사용하여 원곡과 커버 곡 사이의 가사 유사성을 평가합니다. 이를 위해 Levenshtein 거리 및 단어 오류율을 계산하는 기존 구현을 활용합니다. 또한, 텍스트 전처리 및 임베딩 생성을 위해 사전 학습된 XLM-RoBERTa 모델을 사용합니다. 이 임베딩 벡터를 기반으로 커버 곡과 원곡 사이의 유사성을 계산하여 가장 유사한 곡을 예측합니다. 모델 훈련에는 삼중 항 손실(triplet loss) 방법을 사용하여 유사한 샘플 간의 거리를 최소화하고 비유사한 샘플 간의 거리를 최대화합니다.

- **Performance Highlights**: 제안된 방법은 기존의 여러 기준 방법들보다 더 나은 성능을 보여주었습니다. 평가 메트릭으로는 평균 정밀도(mAP), 평균 순위(MR), 정밀도@1(P@1)을 사용하였으며, 이는 커버 곡 탐지 분야에서 널리 사용되는 메트릭입니다. 이러한 결과는 가사를 활용한 커버 곡 식별 방법의 우수성을 입증합니다.



### Dynamic Online Recommendation for Two-Sided Market with Bayesian Incentive Compatibility (https://arxiv.org/abs/2406.04374)
- **What's New**: 이 논문은 인터넷 경제에서 중요한 역할을 하는 추천 시스템의 설계 과정에서 직면하는 두 가지 주요 문제, 즉 (1) 새로운 제품 탐색과 이미 알려진 사용자 선호도 활용 간의 탐색-활용 균형 문제와 (2) 사용자들의 자발적 행동과 이질적 선호도를 고려한 동적인 인센티브 호환성 문제를 공식화했습니다. 이를 해결하기 위해 동적 베이지안 인센티브 호환 추천 프로토콜(DBICRP)을 제안하고, RCB라는 두 단계 알고리즘을 개발했습니다.

- **Technical Details**: RCB 알고리즘은 첫 번째 단계에서 동적 인센티브 호환성을 유지하면서 충분한 샘플 크기를 결정하기 위해 제품을 탐색하고, 두 번째 단계에서는 반비례 샘플링을 사용하여 적은 후회를 보장합니다. 이 알고리즘은 가우시안 사전(gaussian prior) 가정 하에서 베이지안 인센티브 호환성(Bayesian Incentive Compatibility, BIC)을 만족함을 이론적으로 증명했습니다.

- **Performance Highlights**: RCB 알고리즘은 후회(regret)가 $O(\sqrt{KdT})$임을 이론적으로 증명했으며, 시뮬레이션 및 실제 사례(예: 맞춤형 와파린 개인별 투여량)에서 강력한 인센티브 효과와 적은 후회, 높은 강건성을 입증했습니다.



### Multi-Head RAG: Solving Multi-Aspect Problems with LLMs (https://arxiv.org/abs/2406.05085)
- **What's New**: MRAG(다중 헤드 검색 증강 생성 모델)은 대규모 언어 모델(LLM)이 다양한 내용이 담긴 여러 문서를 검색해야 하는 쿼리를 처리할 수 있도록 설계되었습니다. 기존의 RAG(검색 증강 생성) 솔루션은 이러한 쿼리를 처리하는 데 어려움을 겪었으나, MRAG는 변환기(Transformer)의 다중 헤드 주의층(multi-head attention layer) 활성화를 활용해 이 문제를 해결합니다. 다양한 주의 헤드가 여러 데이터 측면을 학습할 수 있도록 함으로써 복잡한 쿼리에 대한 검색 정확도를 향상시킵니다.

- **Technical Details**: MRAG는 기본적인 RAG 설계를 향상시키는 간단하지만 강력한 접근법을 제안합니다. 기존의 마지막 층 디코더(decoder) 블록의 활성화를 키로 사용하지 않고, 다중 헤드 주의층의 활성화를 키로 사용하여 다중 측면 문서를 검색합니다. 이러한 다중 측면 임베딩(embedding)을 데이터 항목과 쿼리 표현 모두에 직접 사용합니다. MRAG는 새로운 평가 방법론과 메트릭, 합성 데이터셋 및 실제 사례를 통해 그 효과를 입증합니다. MRAG의 코드와 관련 자료는 공개되어 있으며, RAGAS와 같은 벤치마킹 도구 및 다양한 데이터 스토어 클래스와 쉽게 통합될 수 있습니다.

- **Performance Highlights**: MRAG는 복잡한 쿼리에 대한 검색 정확도에서 기존 RAG 기반보다 최대 20% 향상된 성능을 보여줍니다. 예를 들어, 다중 측면 위키피디아(Wikipedia) 기사 검색에서 20% 향상을 보였습니다. 이러한 다중 측면 임베딩 아이디어는 추가적인 공간 요구 없이 RAG의 성능을 향상시킵니다.



### Error Bounds of Supervised Classification from Information-Theoretic Perspectiv (https://arxiv.org/abs/2406.04567)
- **What's New**: 이번 연구는 정보이론적 관점에서 심층 신경망(DNN)을 사용한 지도 분류의 이론적 기초를 탐구하며, 오버파라미터화된 신경망의 일반화 능력, 비볼록 최적화 문제에서의 효율적 성능, 플랫 최소값(flat minima)의 메커니즘을 설명하는 새로운 개념을 도입했습니다. 본 논문에서는 fitting error(맞춤 오류)와 model risk(모델 위험)을 소개하여 기존의 generalization error(일반화 오류)와 함께 기대 위험의 상한을 형성합니다.

- **Technical Details**: 일반화 오류가 데이터 분포의 스무스성과 샘플 크기에 의해 영향을 받는 복잡성에 의해 제한됨을 증명했습니다. 우리는 NTK(Neural Tangent Kernel) 및 모델의 파라미터 수와 fitting error의 상관관계를 도출합니다. KL 발산(Kullback-Leibler divergence)을 사용하여 기존 손실 함수의 의존성을 제거하고, 삼각부등식(triangle inequality)을 활용하여 기대 위험의 상한을 설정했습니다.

- **Performance Highlights**: 실증 검증은 도출된 이론적 상한과 실제 기대 위험 사이에 유의미한 양의 상관관계가 있음을 보여, 이론적 발견의 실용성을 확인했습니다. 작은 최대의 eNTK(equal-input Neural Tangent Kernel, λ_max(H(fθ(x))))은 기대 위험을 최소화하는 데 유리한 것으로 증명되었습니다.



### GNNAnatomy: Systematic Generation and Evaluation of Multi-Level Explanations for Graph Neural Networks (https://arxiv.org/abs/2406.04548)
- **What's New**: 새로운 연구는 다양한 하위 구조를 체계적으로 탐색하고 결과를 평가하는 데 어려움을 겪는 기존 방법론의 한계를 극복하기 위해 GNNAnatomy라는 시각적 분석 시스템을 소개합니다. 이는 그래프 수준의 분류 작업에서 GNN의 동작을 설명하기 위해 그래프렛(graphlets)을 사용하며, 가설적 사실(factual) 및 반사실적(counterfactual) 설명을 통해 GNN의 행동을 분석합니다.

- **Technical Details**: GNNAnatomy 시스템은 모델 및 데이터셋에 독립적으로 작동하며, 그래프렛(graphlets)을 사용해 GNN의 예측과 그래프렛 빈도의 상관 관계를 분석하여 설명을 생성합니다. 구체적으로, (1) 그래프렛 빈도와 분류 신뢰도 간의 상관 관계와 (2) 원래 그래프에서 해당 하위 구조를 제거한 후 분류 신뢰도의 변화를 평가하는 두 가지 측정을 도입합니다. 실제로 그래프렛 빈도를 계산하는 것은 NP-hard 문제이므로, GNNAnatomy는 샘플링 방법을 사용하여 3, 4, 5개의 노드를 가진 그래프렛의 빈도를 계산합니다.

- **Performance Highlights**: 실제 데이터셋과 합성 데이터셋을 활용한 사례 연구에서 GNNAnatomy는 효과적인 설명을 제공하는 것으로 입증되었습니다. 또한, 최신 GNN 설명자(state-of-the-art GNN explainer)와 비교하여 그 설계의 유용성과 다용성을 보여주었습니다.



### Negative Feedback for Music Personalization (https://arxiv.org/abs/2406.04488)
Comments:
          6 pages, 4 figures, accepted to ACM UMAP 2024

- **What's New**: 이번 연구에서는 인터넷 라디오의 Next-Song 추천 시스템에서 실제 부정 피드백을 활용하여 학습 속도와 정확성을 크게 개선할 수 있음을 입증했습니다. 또한, 사용자 피드백 시퀀스에 스킵(건너뛰기) 데이터를 추가함으로써 사용자 커버리지와 정확성을 모두 개선하는 방법을 제안했습니다.

- **Technical Details**: 본 연구에서는 SASRec와 BERT4Rec와 같은 기존의 Transformer 아키텍처를 바탕으로 한 추천 시스템을 참고했습니다. 또한, 부정 표본으로 랜덤 샘플을 사용하는 대신 실제 사용자로부터 수집된 명시적 부정 피드백(예: 'thumb-down')을 사용하여 모델을 학습시켰습니다. 이를 통해, 학습 시간을 약 60% 절감하고 테스트 정확도를 약 6% 개선할 수 있음을 확인했습니다.

- **Performance Highlights**: 실험 결과, 명시적 부정 피드백을 포함한 모델이 더 적은 학습 시간으로 더 높은 정확도를 보였으며, 특히 부정 표본으로 사용된 데이터의 양이 적절했을 때 최상의 성능을 발휘했습니다. 또한, 스킵 데이터를 추가로 입력하여 개인화된 추천의 범위를 확대하고 정확도도 약간 향상시켰습니다.



New uploads on arXiv(cs.IR)

### ElasticRec: A Microservice-based Model Serving Architecture Enabling Elastic Resource Scaling for Recommendation Models (https://arxiv.org/abs/2406.06955)
- **What's New**: ElasticRec는 추천 시스템(RecSys)을 위한 새로운 모델 서빙 아키텍처로, 리소스 탄력성과 높은 메모리 효율성을 제공합니다. 기존의 모델 단위로 리소스를 할당하는 방식 대신, ElasticRec는 미세한 수준의 리소스 할당을 가능하게 하는 마이크로서비스(microservice) 소프트웨어 아키텍처를 바탕으로 설계되었습니다.

- **Technical Details**: ElasticRec는 두 가지 주요 특징을 가지고 있습니다. 첫째, 마이크로서비스 기반 추론 서버를 사용하여 리소스 탄력성을 극대화합니다. 마이크로서비스는 큰 단일 응용 프로그램을 여러 개의 독립적이고 세밀한 서비스로 나눌 수 있게 해줍니다. 둘째, 유틸리티 기반의 리소스 할당 정책을 통해 높은 메모리 효율성을 달성합니다. 모델은 밀집 DNN 레이어와 희석 임베딩 레이어로 나뉘며, 임베딩 레이어는 다시 '핫'과 '콜드' 임베딩 단위로 나뉩니다.

- **Performance Highlights**: ElasticRec는 메모리 할당 크기를 평균 3.3배 줄이고, 메모리 유틸리티를 8.1배 증가시켜 평균적으로 배포 비용을 1.6배 절감할 수 있습니다. 또한, 고유한 리소스 수요에 맞춘 리소스 할당을 통해 전체적인 QPS(Queries Per Second)를 최대화할 수 있습니다.



### Non-autoregressive Personalized Bundle Generation (https://arxiv.org/abs/2406.06925)
Comments:
          Submitted to Information Processing & Management

- **What's New**: 최근 추천 시스템 연구에서, 사용자의 선호에 맞춰 개인화된 번들을 생성하는 문제에 대한 관심이 증가하고 있습니다. 기존 연구들은 번들의 순서 불변성 특성을 고려하지 않아 순차적 모델링 방법을 채택한 반면, 본 연구에서는 비자기회귀(non-autoregressive) 메커니즘을 활용하여 번들을 생성하는 새로운 인코더-디코더 프레임워크인 BundleNAT을 제안합니다. 이를 통해 본 연구는 번들의 순서에 의존하지 않고 한 번에 목표 번들을 생성할 수 있습니다.

- **Technical Details**: 본 연구에서는 사전 훈련(pre-training) 기술과 그래프 신경망(Graph Neural Network, GNN)을 채택하여 사용자 기반 선호도 및 아이템 간 호환성 정보를 완전하게 내재화합니다. 이후 자기 주의(self-attention) 기반 인코더를 활용하여 글로벌 종속성 패턴을 추출합니다. 이를 기반으로 번들 내의 순열에 고유한 디코딩 아키텍처를 설계하여 직접적으로 원하는 번들을 생성합니다.

- **Performance Highlights**: YouShu와 Netease의 세 가지 실제 데이터셋에서 진행된 실험 결과, BundleNAT은 정밀도(Precision), 확장 정밀도(Precision+), 재현율(Recall)에서 각각 최대 35.92%, 10.97%, 23.67%의 절대적 향상을 보여 현재 최신 기법들을 크게 능가하는 성과를 보였습니다.



### Scaling the Vocabulary of Non-autoregressive Models for Efficient Generative Retrieva (https://arxiv.org/abs/2406.06739)
Comments:
          14 pages, 6 tables, 2 figures

- **What's New**: 이 논문은 정보 검색(Information Retrieval)을 더 효율적으로 수행하기 위해 Non-autoregressive(NAR) 언어 모델을 사용하는 새로운 접근법을 제안합니다. 특히, 다중 단어 엔티티 및 공통 구문(phrases)을 포함한 확장된 어휘(vocabulary)를 사용하여 NAR 모델의 성능을 향상시키는 PIXAR 접근법을 제안합니다. 이 방법은 Autoregressive(AR) 모델에 비해 지연 시간(latency)과 비용은 낮추면서도 검색 성능을 유지하도록 돕습니다.

- **Technical Details**: PIXAR은 NAR 모델에서 다중 단어 구문을 예측할 수 있는 확장된 목표 어휘를 사용하여 검색 성능을 향상시킵니다. 추가된 어휘는 최대 5백만 개의 토큰을 포함하며, 이로 인해 모델은 오토리그레시브 모델만큼의 복잡한 종속성 문제를 해결할 수 있습니다. 또한, PIXAR은 효율적인 추론 최적화 기법을 도입하여 큰 어휘를 사용하더라도 낮은 추론 지연 시간을 유지하기 위해 설계되었습니다.

- **Performance Highlights**: PIXAR은 MS MARCO에서 MRR@10 기준으로 31.0% 상대 성능 향상을, Natural Questions에서 Hits@5 기준으로 23.2% 향상을 보여줍니다. 또한, 대형 상업 검색 엔진에서 진행한 A/B 테스트 결과, 광고 클릭은 5.08%, 수익은 4.02% 증가했습니다.



### Leveraging Large Language Models for Knowledge-free Weak Supervision in Clinical Natural Language Processing (https://arxiv.org/abs/2406.06723)
- **What's New**: 본 논문에서는 임상 도메인에서 라벨된 학습 데이터가 충분하지 않은 상황에서, 약한 감독학습(weak supervision)과 컨텍스트 학습(in-context learning)을 사용하는 새로운 접근 방식을 제안합니다. 이 새로운 방법은 도메인 지식 없이도 Llama2와 같은 대형 언어 모델(LLM)을 활용하여 약한 라벨 데이터를 생성하고, 이를 통해 성능이 우수한 BERT 모델을 학습합니다.

- **Technical Details**: 연구자들은 프롬프트 기반 접근 방식을 사용하여 LLM(Llama2)를 통해 약한 라벨 데이터를 생성하고, 이를 이용해 down-stream BERT 모델을 학습시켰습니다. 이후 소량의 고품질 데이터로 추가 미세조정을 통해 모델의 성능을 더욱 향상시켰습니다. 이 접근법은 n2c2 데이터셋 세 가지를 사용하여 평가되었습니다.

- **Performance Highlights**: 10개의 gold standard 노트만 사용했을 때, Llama2-13B로 약한 감독을 받은 최종 BERT 모델은 기본 제공되는 PubMedBERT의 F1 점수를 4.7%에서 47.9%까지 지속적으로 능가했습니다. 50개의 gold standard 노트만 사용했을 때에도, 이 모델은 완전히 미세 조정된 시스템에 가까운 성능을 보여주었습니다.



### Link Prediction in Bipartite Networks (https://arxiv.org/abs/2406.06658)
Comments:
          28th International Conference on Knowledge-Based and Intelligent Information & Engineering Systems (KES), Sep 2024, Sevilla, Spain

- **What's New**: 이번 연구에서는 이분 그래프(bipartite networks)에서 링크 예측(link prediction)을 위한 19개의 방법을 비교 실험하였습니다. 일부 방법은 기존 문헌에서 가져왔으며, 일부는 단일 네트워크(unipartite networks)를 위해 설계된 기술을 저자들이 이분 그래프로 수정한 것입니다. 추가적으로, 그래프 합성망(Convolutional Networks, GCN)을 기반으로 하는 추천 시스템을 이분 그래프의 새로운 링크 예측 솔루션으로 제안하였습니다.

- **Technical Details**: 이번 연구에서는 다양한 위상구조를 가진 3개의 실제 이분 그래프 데이터셋을 구축하여 실험을 진행하였습니다. 연구에 포함된 19개의 링크 예측 방법에는 기존 연구에서 사용된 방법과, 단일 네트워크를 위해 설계되었지만 이분 그래프로 수정한 방법이 포함되어 있습니다. GCN 기반 개인화 추천 시스템을 통해 링크 예측 성능을 평가하였으며, 또한 학습 프로세스에 의존하지 않는 순수한 휴리스틱 지표(Structural Perturbation Method, SPM) 역시 효과적인 결과를 보였습니다.

- **Performance Highlights**: 결과적으로, GCN 기반 개인화 추천 시스템은 이분 그래프에서 성공적인 링크 예측을 할 수 있음을 보였습니다. 또한 학습 과정이 필요 없는 구조 교란법(SPM)과 같은 순수한 휴리스틱 지표도 성공적으로 링크 예측을 수행했습니다.



### Harnessing AI for efficient analysis of complex policy documents: a case study of Executive Order 14110 (https://arxiv.org/abs/2406.06657)
Comments:
          28 pages, 1 figure

- **What's New**: 정책 문서(legislation, regulations, executive orders)가 사회를 형성하는 데 중요한 역할을 하지만, 그 길이와 복잡성 때문에 해석과 적용이 어렵고 시간이 많이 소요됩니다. 본 연구는 인공지능(AI), 특히 대형 언어 모델(LLMs)이 이러한 문서 분석을 자동화하여 정확성과 효율성을 높일 수 있는 가능성을 평가하는 것을 목적으로 합니다. 특히 정책 문서에서 콘텐츠 추출과 질문 응답 작업에 대한 AI의 잠재력을 조사했습니다. '인공지능의 안전하고, 보안적이고, 신뢰할 수 있는 개발 및 사용'에 관한 행정명령 14110을 사례 연구로 사용하여 네 개의 상업용 AI 시스템이 이 문서를 분석하고 대표적인 정책 질문에 답하도록 했습니다.

- **Technical Details**: 연구는 질문 응답과 콘텐츠 추출 작업에 초점을 맞추어 진행되었습니다. Gemini 1.5 Pro와 Claude 3 Opus 두 AI 시스템이 특히 뛰어난 성능을 보였으며, 복잡한 문서에서 정확하고 신뢰할 수 있는 정보 추출을 제공했습니다. 이들은 인간 분석가와 비교해도 손색이 없었으며, 훨씬 높은 효율성을 나타냈습니다.

- **Performance Highlights**: Gemini 1.5 Pro와 Claude 3 Opus 시스템은 복잡한 정책 문서에서의 정확한 정보 추출 및 분석을 통해 높은 성능을 입증했습니다. 하지만 재현성(reproducibility) 문제는 여전히 해결이 필요하며, 추가적인 연구와 개발이 요구됩니다.



### Anna Karenina Strikes Again: Pre-Trained LLM Embeddings May Favor High-Performing Learners (https://arxiv.org/abs/2406.06599)
Comments:
          9 pages (not including bibliography), Appendix and 10 tables. Accepted to the 19th Workshop on Innovative Use of NLP for Building Educational Applications, Co-located with NAACL 2024

- **What's New**: 학생의 자유 답변을 통해 행동 및 인지 프로파일을 도출하는 머신러닝 기술에서 LLM(대형 언어 모델) 임베딩을 사용한 비지도 클러스터링이 새로운 시도로 연구되고 있습니다. 이 연구는 생물학 수업에서의 학생 답변을 대상으로, 전문 연구자들이 이론 기반의 'Knowledge Profiles(KPs)'로 분류한 결과와 순수한 데이터 기반의 클러스터링 기법의 결과를 비교했습니다. 그 결과, 정답을 포함한 특정 KPs를 제외하고는 대다수의 KPs가 잘 발견되지 않는 '발견 편향(discoverability bias)'이 나타났음을 밝혔습니다.

- **Technical Details**: 학생 답변 데이터를 활용하여 KMeans와 HDBSCAN 같은 일반적인 클러스터링 기법이 이론 기반의 KPs를 발견하는 정도를 평가했습니다. 또한, 'Anna Karenina 원칙'이라는 개념을 도입하여, 답변의 품질(정답 또는 다양한 정도의 오답)과 이들의 임베딩 기반 표현의 형태 및 밀도 사이의 관계를 분석했습니다.

- **Performance Highlights**: 결과적으로, 데이터의 임베딩과 클러스터링 기법이 대부분의 이론 기반 KPs를 발견하는 데 실패했으며, 정답을 포함하는 KPs만이 어느 정도 잘 발견되었습니다. 이는 교육적으로 의미 있는 정보를 유지하는 데 문제가 있을 수 있음을 시사합니다. 중요한 점은, 사전 학습된 LLM 임베딩이 교육적 프로파일 발견의 기초로서 반드시 최적이 아닐 수도 있다는 것입니다.



### Graph Neural Network Enhanced Retrieval for Question Answering of LLMs (https://arxiv.org/abs/2406.06572)
Comments:
          Under review

- **What's New**: 이 논문은 GNN-Ret이라는 새로운 데이터 검색 방법을 제안합니다. GNN-Ret은 그래프 뉴럴 네트워크(GNN)를 활용하여 문단 사이의 관계성을 반영함으로써 검색 성능을 향상시킵니다. 또한, 반복적인 그래프 뉴럴 네트워크(RGNN)를 사용하는 RGNN-Ret을 통해 멀티 홉 추론 질문도 처리할 수 있습니다.

- **Technical Details**: GNN-Ret은 먼저 구조적으로 연관된 문단과 키워드를 공유하는 문단을 연결하여 문단의 그래프를 구성합니다. 그런 다음 GNN을 사용하여 문단 간의 관계를 이용해 검색을 개선합니다. RGNN-Ret은 멀티 홉 질문의 검색을 향상시키기 위해 각 단계에서 이전 단계의 검색 결과를 통합하는 방식으로 동작합니다.

- **Performance Highlights**: 광범위한 실험에서 GNN-Ret은 단일 LLM 쿼리를 통해도 기존 다수 쿼리 기반 방식을 초과하는 높은 정확도를 보여주었습니다. 특히, RGNN-Ret은 2WikiMQA 데이터셋에서 10.4% 이상의 정확도 향상을 달성하며, 최신 성능을 보여주었습니다.



### UMBRELA: UMbrela is the (Open-Source Reproduction of the) Bing RELevance Assessor (https://arxiv.org/abs/2406.06519)
Comments:
          5 pages, 3 figures

- **What's New**: 새로 등장한 UMBRELA 툴킷은 OpenAI의 GPT-4 모델을 사용해 대형 언어 모델(LLM)이 검색 시스템 평가에 필요한 관련성 판단을 자동화하는 오픈 소스 도구입니다. Microsoft Bing의 연구를 재현하고 이를 더욱 확장했습니다. 이 툴킷은 TREC 2024 RAG 트랙에서 사용될 예정입니다.

- **Technical Details**: UMBRELA는 LLM을 이용해 검색 쿼리와 결과 간의 관련성을 평가합니다. 논문에서 제시된 흐름을 재현하기 위해 'zero-shot DNA prompting' 기법을 사용했습니다. TREC 2019-2023의 Deep Learning Tracks 데이터를 사용하여 LLM의 관련성 판단 결과를 인간 평가자와 비교했습니다.

- **Performance Highlights**: UMBRELA의 LLM 기반 관련성 판단은 다단계 검색 시스템의 순위와 높은 상관관계를 보였습니다. 결과적으로 LLM은 인간 평가자와 유사한 수준의 정확성과 신뢰성을 제공함을 확인했습니다. 이러한 성과는 LLM이 더 비용 효율적이고 정확한 대안이 될 수 있음을 뒷받침합니다.



### Survey for Landing Generative AI in Social and E-commerce Recsys -- the Industry Perspectives (https://arxiv.org/abs/2406.06475)
- **What's New**: 최근 생성적 인공지능(GAI)의 등장으로 산업 추천 시스템(Recsys)이 혁신적인 변화를 겪고 있습니다. 이 논문은 사회적 및 전자상거래 플랫폼에서 GAI를 성공적으로 통합한 경험을 바탕으로, GAI와 Recsys의 통합에 대한 실질적인 통찰과 도전과제를 종합적으로 검토합니다. GAI와 Recsys 통합에 관한 실용적인 적용 사례와 해결책을 제시하며 향후 연구 방향을 제시합니다.

- **Technical Details**: 이 논문은 산업 Recsys의 복잡한 인프라, 운영 절차 및 비즈니스 제품 관점을 고려하여 GAI를 통합하는 데 필요한 실제 솔루션 프레임워크를 탐구합니다. 특히, GAI와 LLMOps(Long Language Model Operations) 기초, 맞춤형 추천을 강화를 위한 GAI 유스케이스, 그리고 Recsys 내의 Retrieval-Augmented Generation(RAG) 활용 방법 등을 포괄적으로 다룹니다. Prompt engineering, in-context learning, chain-of-thought와 같은 기법 적용 방법도 상세히 설명됩니다.

- **Performance Highlights**: 논문에서는 사용자 만족도 및 투명성과 신뢰성을 향상시키기 위해 GAI를 활용한 콘텐츠의 재목적화와 외부 지식을 통한 큐레이션이 강조됩니다. 또한, Recsys가 더욱 상호작용적이고 사용자 피드백 루프 기반으로 발전할 수 있는 방향성을 제시합니다. GAI 솔루션의 비용, 지연 시간, 전용 데이터 및 도메인 지식을 효율적으로 사용하기 위한 최적화 방향도 제시됩니다.



### Greedy SLIM: A SLIM-Based Approach For Preference Elicitation (https://arxiv.org/abs/2406.06061)
- **What's New**: 새로운 사용자의 선호도 추정을 위한 방법으로, 최신형 추천 시스템인 SLIM을 기반으로 하는 새로운 접근 방식을 제안합니다. 본 연구는 Greedy SLIM이라는 새로운 학습 기법을 활용해, 새로운 사용자에게 질문할 항목을 선정합니다. 이를 통해 특히 사용자 연구에서 뛰어난 성능을 보인다는 결론을 얻었습니다.

- **Technical Details**: 제안한 Greedy SLIM 방법은 기존 SLIM 학습 방법의 문제를 해결하기 위해 고안되었습니다. SLIM(Scalable Likelihood-based Item Model)은 협업 필터링에서 최적의 상위 N개의 추천을 위해 사용되는 기법입니다. Greedy SLIM은 항목을 하나씩 선택해 SLIM 손실을 최소화하는 방식으로 학습을 진행합니다. 이는 active learning 접근법의 일환으로, 새로운 사용자가 시스템에 입력할 항목을 최적화합니다.

- **Performance Highlights**: 오프라인 실험과 사용자 연구를 통해 Greedy SLIM의 성능을 평가했습니다. 사용자 연구에서는 특히 긍정적인 결과를 보이며, 기존의 잠재 인자 모델(LFM) 기반 방법보다 더 적합한 것으로 나타났습니다. 이는 사용자가 적은 항목만 평가해도 적정한 추천 결과를 얻을 수 있음을 시사합니다.



### Modeling User Retention through Generative Flow Networks (https://arxiv.org/abs/2406.06043)
Comments:
          KDD-ADS 2024

- **What's New**: 이번 연구는 사용자의 재방문 행동(user retention)을 최적화하는 새로운 추천 시스템 프레임워크인 GFN4Retention을 제안합니다. 기존 연구 대부분이 사용자의 즉각적인 피드백을 최대화하는 데 초점을 맞췄다면, 본 연구는 사용자의 세션 간 안정적이고 지속적인 사용을 고려하였습니다.

- **Technical Details**: GFN4Retention은 Generative Flow Networks (GFNs)를 기반으로 한 세션-wise 추천 시스템입니다. 이 프레임워크는 사용자의 세션 종료 시점의 만족도를 추정하고, 이를 기반으로 각 추천 항목에 대해 확률적 흐름(probabilistic flow)을 모델링합니다. 구체적으로, 추천 과정을 조건부 순방향 확률적 흐름으로 간주하고, 각각의 사용자 상태에 대한 흐름 추정기(flow estimator)를 활용합니다.

- **Performance Highlights**: GFN4Retention은 두 개의 공공 데이터셋과 실제 산업 플랫폼에서의 온라인 A/B 테스트를 통해 검증되었습니다. 기존의 강화 학습 기반 추천 모델들과 비교하여 우수한 성능을 보였으며, 각 구성 요소의 효과를 분석한 에이블레이션 연구에서도 높은 안정성을 나타냈습니다.



### A WT-ResNet based fault diagnosis model for the urban rail train transmission system (https://arxiv.org/abs/2406.06031)
Comments:
          12 pages,10 figures

- **What's New**: 새로운 연구는 도시 철도 시스템의 고장 진단을 위한 혁신적인 모델을 제안합니다. 이 모델은 웨이블릿 변환(Wavelet Transform)과 잔차 신경망(Residual Neural Network, ResNet)의 장점을 통합하여 진단 정확도와 강건성을 높였습니다.

- **Technical Details**: 제안된 모델은 웨이블릿 변환을 사용하여 도시 철도의 특징을 추출하고, ResNet을 통해 패턴 인식을 수행합니다. 이는 딥러닝(DL) 알고리즘 가운데 높은 성과를 보이는 ResNet을 활용한 것입니다. 또한, 기존의 CNN(Convolutional Neural Network), RNN(Recurrent Neural Network)와의 비교 및 다양한 데이터 세트에 대한 적응력을 강조합니다.

- **Performance Highlights**: 실험 결과, 제안된 WT-ResNet 모델은 도시 철도 열차의 고장을 식별하는 데 있어 높은 효율성과 정확도를 입증했습니다. 이는 유지보수 전략을 개선하고 가동 중단 시간을 줄이는 데 기여할 수 있습니다.



### Weighted KL-Divergence for Document Ranking Model Refinemen (https://arxiv.org/abs/2406.05977)
- **What's New**: 이 논문은 트랜스포머 기반의 문서 검색 및 랭킹 모델을 위한 새로운 학습 손실 함수, 즉 대조적으로 조정된 KL 다이버전스(contrastively-weighted KL divergence, CKL)를 제안합니다. 기존의 KL 다이버전스 기반 지식 증류(loss)를 개선하여 문서 탐색 모델의 성능을 높이는 것을 목표로 합니다.

- **Technical Details**: CKL은 기존의 KL 다이버전스 대신, 긍정(positive) 문서와 부정(negative) 문서 간의 구분을 명확히 하기 위해 대조 학습(contrastive learning)과 결합된 새로운 방법론입니다. 이 방법은 문서 랭킹의 타이트한 분포 매칭(tight distribution matching)에 따른 과도한 보정(over-calibration) 문제를 해결하는 데 중점을 둡니다. MS MARCO와 BEIR 데이터셋을 사용한 평가에서 CKL의 유효성을 입증했습니다.

- **Performance Highlights**: CKL은 기존 KL 다이버전스 및 최근 BKL 접근법과 비교해 문서 검색의 성능을 효과적으로 향상시켰습니다. 실험 결과, CKL을 적용한 학생 모델은 긍정 문서와 부정 문서의 적절한 분포를 유지하면서도 랭킹의 관련성을 크게 높였습니다.



### Async Learned User Embeddings for Ads Delivery Optimization (https://arxiv.org/abs/2406.05898)
Comments:
          Accepted by workshop on Multimodal Representation and Retrieval at SIGIR 2024, Washington DC

- **What's New**: Meta 플랫폼에서 수십억 명의 사용자들을 대상으로 한 고품질 사용자 임베딩(user embeddings)을 비동기적으로 학습하는 방식을 제안합니다. 이 임베딩은 사용자 유사도 그래프(user similarity graphs)로 변환되어, 실시간 사용자 활동과 결합되어 광고 추천에 활용됩니다.

- **Technical Details**: 사용자 임베딩은 다중 모드(multimodal) 사용자 활동 신호를 기반으로 하는 시퀀스(sequence)에서 학습되며, 이는 Transformer와 유사한 구조로 처리됩니다. 다양한 사용자 활동에는 클릭한 광고, 댓글, 조회한 사진 및 동영상 등이 포함됩니다. 이러한 임베딩은 비동기적으로 업데이트되어, 수십억 명의 사용자 데이터를 처리할 수 있습니다. 또한, 사용자 유사도 그래프는 이 데이터를 기반으로 생성되며, 광고 추천에 더욱 효과적으로 활용됩니다.

- **Performance Highlights**: 제안된 모델은 오프라인 및 온라인 실험 모두에서 유의미한 성능 향상을 보였습니다. 특히, 사용자의 최신 상호작용과 피드백을 기반으로 사용자 임베딩을 지속적으로 업데이트하고 정제할 수 있는 능력을 갖추고 있습니다. 추가로, 사용자 임베딩은 대규모 모델에서 효율적인 스토리지와 컴퓨팅 절약을 위해 압축할 수 있습니다.



### Prioritizing Potential Wetland Areas via Region-to-Region Knowledge Transfer and Adaptive Propagation (https://arxiv.org/abs/2406.05578)
- **What's New**: 새로운 연구는 데이터 부족 문제를 해결하여 습지 식별 및 우선순위 설정을 돕기 위한 두 가지 전략을 제안합니다. 첫 번째 전략은 습지가 풍부한 지역에서 데이터가 희소한 지역으로 지식을 전달하는 것입니다. 두 번째 전략은 적응형 전파 메커니즘을 사용하는 공간 데이터 보강 전략입니다.

- **Technical Details**: 제안하는 접근법은 두 가지 주요 기술을 포함합니다. (1) 도메인 비디오 연습(domain disentanglement) 전략을 사용하여 풍부한 습지 지역의 지식을 데이터를 통해 전달합니다. 이 과정에서는 도메인별로 동일하게 적용 가능한 정보만 선택적으로 전이하고, 도메인 분리기로 도메인별 정보와 공유 가능한 정보를 분리합니다. (2) Graph Neural Networks (GNNs)의 적응형 전파 메커니즘을 도입하여 인접 노드들의 상호 영향을 구분하는 방식입니다. 이는 지역 내 세포 간의 유용한 정보를 차별화하여 전달합니다.

- **Performance Highlights**: 제안된 방법의 효과, 강인성 및 확장성을 입증하기 위해 엄격한 실험을 수행했습니다. 이를 통해 제안된 방법이 기존 최신 기준선을 능가하며, 모듈별 실험을 통해 각 모듈이 기존 습지 식별에 필수적임을 보여줍니다.



### I-SIRch: AI-Powered Concept Annotation Tool For Equitable Extraction And Analysis Of Safety Insights From Maternity Investigations (https://arxiv.org/abs/2406.05505)
- **What's New**: I-SIRch라는 새로운 접근 방식이 소개되었습니다. I-SIRch는 인공지능과 머신러닝 알고리즘을 활용하여 영국의 Healthcare Safety Investigation Branch(HSIB)에서 생산된 조사 보고서에서 발생한 임신 관련 사고에 대한 인간 요인(human factors)을 자동으로 식별하고 레이블을 지정합니다. 이는 생물의학적 개념에만 초점을 맞추는 기존 도구와 차별화되며, 인간 요인을 포함하여 의료 제공 시스템을 더 잘 이해하는 데 기여합니다.

- **Technical Details**: I-SIRch는 SIRch 택소노미를 사용하여 의료 조사로부터 안전 통찰을 추출합니다. 데이터 전처리 단계에서 PDF로 제공된 보고서를 텍스트로 추출하고, 특정 기준(예: 섹션, 페이지, 단락)에 따라 텍스트를 추출합니다. 수동으로 태그된 데이터를 사용해 모델을 학습시키고, 새로운 보고서를 처리하면서 인간 전문가의 계속된 주석을 통해 모델 성능을 향상시키는 'human-in-the-loop' 방식을 채택했습니다.

- **Performance Highlights**: I-SIRch는 실제 및 합성 데이터를 통해 연구되었습니다. 818개의 합성 문장과 97개의 실제 보고서에서 추출한 1960개의 문장을 테스트한 결과, 실제 보고서 문장에서 90%의 정확도로 관련 개념을 올바르게 식별했습니다. 이를 통해 다양한 인종 그룹에 따라 특정 인간 요인이 어떻게 차이 나는지 분석할 수 있었으며, 이는 사회적, 기술적 및 조직적 요인이 산모 안전과 인구 건강 결과에 미치는 복잡한 상호작용을 이해하는 데 새로운 가능성을 열어주었습니다.



### PTF-FSR: A Parameter Transmission-Free Federated Sequential Recommender System (https://arxiv.org/abs/2406.05387)
- **What's New**: 최근 사용자 데이터 프라이버시에 대한 우려가 증가함에 따라 연구자들은 사용자 데이터 프라이버시를 보호하면서 협력 학습을 구현하는 Federated Sequential Recommender Systems(FedSeqRecs)라는 접근 방식을 제안했습니다. 하지만 이는 모델 공유와 높은 통신 비용이라는 두 가지 큰 단점이 존재했죠. 이번에 제안된 연구는 그 단점을 극복하기 위해 모델 파라미터 전송 없이 협력 학습을 가능하게 하는 새로운 프레임워크, 'Parameter Transmission-Free Federated Sequential Recommendation (PTF-FSR)'을 소개합니다.

- **Technical Details**: PTF-FSR은 기존의 파라미터 교환 방식 대신 예측 결과만을 전송하여 모델 크기와 상관없이 통신 비용을 대폭 줄일 수 있습니다. 또한, 사용자 프라이버시 보호를 위해 클라이언트 측에서 사용자의 원본 항목 상호작용 시퀀스에 섭동을 추가하는 지수 기법(exponential mechanism)을 사용합니다. 이 프레임워크는 ID-based와 ID-free 파라다임을 아우르는 여러 순차적 추천 모델에서 효과적으로 적용되었습니다.

- **Performance Highlights**: 세 가지 널리 사용되는 추천 데이터셋에서 다양한 ID-based 및 ID-free 순차적 추천 모델을 테스트한 결과, PTF-FSR은 높은 성능과 일반화 능력을 보였습니다. 이는 보다 복잡하고 큰 순차적 추천 모델도 수용할 수 있는 새로운 연합 학습 구조의 가능성을 보여줍니다.



### Measuring Fairness in Large-Scale Recommendation Systems with Missing Labels (https://arxiv.org/abs/2406.05247)
- **What's New**: 이번 연구에서는 대규모 추천 시스템에서 발생하는 공정성 문제를 다룹니다. 특히, 추천된 항목에 대한 레이블(label)이 없는 상황에서의 공정성 문제를 해결하기 위한 새로운 방법을 제안합니다. 이 연구는 랜덤 트래픽(randomized traffic)을 활용하여 공정성 지표를 더 정확하게 추정하는 방법을 제시하며, 이를 증명하기 위해 TikTok의 실제 데이터를 사용한 실험 결과를 제공합니다. 또한, TikTok의 공정성 관련 데이터셋을 처음으로 공개하며, 추천 시스템의 데이터셋 수집 방법론에 대한 새로운 기준을 제시합니다.

- **Technical Details**: 이 연구는 'Ranking-based Equal Opportunity (REO)'라는 공정성 개념에 기반하여 진행됩니다. REO는 사용자-아이템(user-item) 상호작용이 완전히 관찰된 경우의 공정성 문제를 다루며, 사용자의 진정한 선호도를 알 수 없는 대규모 추천 시스템에서의 공정성 문제를 해결합니다. 연구팀은 랜덤 트래픽 데이터를 사용하여 공정성 메트릭의 오차 한계를 이론적으로 제시하며, 이를 TikTok의 실제 데이터로 검증했습니다.

- **Performance Highlights**: 실험 결과, 제안된 랜덤 트래픽 활용 방법이 기존의 간단한 방법들보다 훨씬 정확하고 효율적임을 보여줍니다. 특히, TikTok의 실제 데이터셋과 합성 데이터(synthetic data)를 통해 제안된 방법의 이론적 타당성과 실효성을 입증했습니다. 또한, 랜덤 트래픽 데이터를 사용하여 대규모 추천 시스템의 공정성 지표를 정확하게 추정하는 것이 필수적임을 확인했습니다.



### Evaluating the Retrieval Component in LLM-Based Question Answering Systems (https://arxiv.org/abs/2406.06458)
- **What's New**: 이 연구는 Retrieval-Augmented Generation (RAG) 기반의 챗봇 시스템에서 리트리버 (retriever) 성능을 평가하는 간단한 기준을 제안합니다. 기존 평가지표들이 대형 언어 모델(LLMs)의 능력을 완전히 포착하지 못하는 상황에서, 새로운 평가 프레임워크는 챗봇의 전체 성능과 더 잘 맞아떨어지는 평가를 제공합니다.

- **Technical Details**: RAG 모델을 활용한 QA 시스템은 두 가지 주요 구성 요소로 나뉘어집니다. 리트리버가 문서 코퍼스에서 관련 정보를 검색하고, 생성기(generator)가 이 문서를 바탕으로 응답을 생성합니다. 기존 평가 방법이 주석된 데이터에만 집중하는 한계를 극복하기 위해, 우리는 이상적인 리트리버와 실제 리트리버의 출력을 비교하여 downsteam 효과까지 고려하는 새로운 평가 방식을 제안합니다. 이를 위해 Exact Match (EM), 토큰 기반 지표(ROUGE, BLEU, METEOR), 임베딩 기반 지표(BERTScore), 그리고 LLM-기반 평가 방법을 사용합니다.

- **Performance Highlights**: 새로운 리트리버 평가 방법론은 기존의 정밀도(Precision), 재현율(Recall), F1 점수보다 LLM 기반 QA 시스템 성능을 더 잘 반영합니다. NQ-open 코퍼스 실험 결과, 새로운 방법론이 리트리버의 유효성을 더 잘 포착하였고, 기존 지표와 높은 상관관계를 나타냈습니다.



### Combining Embeddings and Domain Knowledge for Job Posting Duplicate Detection (https://arxiv.org/abs/2406.06257)
Comments:
          To be published at 9th International Symposium on Language & Knowledge Engineering LKE 2024

- **What's New**: 이 논문은 여러 플랫폼에 게시된 구인 공고에서 중복 되는 공고를 탐지하는 새로운 접근 방식을 제안합니다. 문자 유사성 및 텍스트 임베딩(text embedding), 키워드 매칭(keyword matching) 방법을 결합하여 성능을 향상시켰습니다. 이 접근 방식은 실제로 사용되고 있으며 긍정적인 피드백을 받고 있습니다.

- **Technical Details**: 중복 탐지에는 문자열 비교, 딥 텍스트 임베딩, 특정 기술에 대한 가중치 목록을 사용한 조합 방식이 사용되었습니다. 각 방법을 개별적으로 사용할 때 성능이 만족스럽지 않지만, 이들을 결합하면 높은 성능과 낮은 오탐률(false positives)을 보입니다.

- **Performance Highlights**: 실제 사용 사례에서 새로운 접근 방식이 높은 성능을 보였으며, 수작업으로 진행하던 중복 탐지 작업을 자동화하는 데 성공적이었습니다. 이로 인해 개발 및 운영 비용도 절감되었습니다.



### Black carbon plumes from gas flaring in North Africa identified from multi-spectral imagery with deep learning (https://arxiv.org/abs/2406.06183)
Comments:
          Published at the workshop Tackling Climate Change with Machine Learning at ICLR 2024

- **What's New**: 이 논문에서는 인공지능(deep learning) 프레임워크를 사용하여 인공위성 이미지 (satellite imagery)를 통해 북아프리카 지역에서 가스 플레어링 (gas flaring)으로 인한 블랙 카본(BC) 배출량을 직접 모니터링하는 방법을 소개합니다. 2022년 동안 모니터링된 결과, 이 지역의 BC 배출량은 약 백만 톤 탄소 동등 (tCO$_{2,	ext{eq}}$)에 이르렀습니다.

- **Technical Details**: 유럽 우주국(ESA)의 Sentinel-2 위성 데이터를 사용하여 알제리, 리비아 및 이집트에서 가스 플레어링 사이트를 분석했습니다. ConvLSTM 모델을 사용하여 위성 이미지에서 BC 플룸(plume)을 감지하고 분류했습니다. 이 모델은 두 RGB 이미지 시퀀스를 입력으로 받아 두 번째 이미지에서 BC 플룸을 세그먼트화(segmentation)하는 작업을 수행합니다. 모델 훈련은 Synthetic BC Plumes를 포함한 Sentinel-2 데이터로 이루어졌습니다. 추가적으로, LightGBM 분류기를 사용하여 거짓 양성(false positive)을 필터링했습니다.

- **Performance Highlights**: 2022년 동안 1963개의 개별 플레어(flares)를 감지했으며, 대부분은 짧은 시간 동안 활동했습니다. 가스 플레어링 사이트 중 약 10곳이 전체 BC 배출량의 25% 이상을 차지했습니다. 이는 효율적인 감지 및 완화 정책을 구현하기 위한 중요한 발걸음입니다.



### Thanking the World: Exploring Gender-Based Differences in Acknowledgment Patterns and Support Systems in Theses (https://arxiv.org/abs/2406.06006)
- **What's New**: 이 연구는 전자 논문 및 학위 논문(Electronic Theses and Dissertations, ETDs)에서 지원 시스템을 조사하여 학위 과정 중 연구자들이 어떤 형태의 지원을 받았는지 분석하였습니다. 특히 도서관 및 정보 과학(Library and Information Science) 분야를 대상으로 했습니다. RoBERTa 기반 모델을 사용하여 1252개의 ETD를 분석한 것은 이번 연구의 주요 기여입니다.

- **Technical Details**: 본 연구에서는 RoBERTa 모델을 사용하여 ETD 감사 섹션에서 다양한 지원 형태를 추출했습니다. 이를 통해 연구자들이 인식하는 주요 지원 유형은 학문적 지원(academic support), 도덕적 지원(moral support), 재정적 지원(financial support), 그리고 종교적 지원(religious support)이었음을 확인했습니다.

- **Performance Highlights**: 연구 결과, 종교적 및 재정적 지원에서는 성별 차이가 거의 없었으나, 학문적 지원과 도덕적 지원의 비율에서는 큰 성별 차이가 발견되었습니다. 특히 지도 교수들은 동성 연구자를 선호하는 경향이 있음을 보여주었습니다. 이 연구는 성별 간의 차이를 이해하고 포용적이며 지원적인 학문 환경을 조성하는 데 중요한 시사점을 제공합니다.



### Explainable AI for Mental Disorder Detection via Social Media: A survey and outlook (https://arxiv.org/abs/2406.05984)
- **What's New**: 이 논문은 데이터 과학과 인공지능(AI)을 이용한 정신 건강 관리의 최신 동향을 조사하며, 특히 온라인 소셜 미디어(OSM)를 통한 정신 질환 감지에 중점을 둡니다. 특히 Explainable AI (XAI) 모델의 중요성을 강조하며, 기존의 진단 방법과 최신 인공지능 구동 연구를 종합적으로 리뷰합니다.

- **Technical Details**: 정신 건강 진단에는 전통적으로 DSM-5와 같은 국제 표준을 기반으로 한 대면 인터뷰와 자기보고식 질문지가 사용됩니다. 최근에는 OSM 데이터를 활용한 딥러닝(Deep Learning) 기술을 통한 감지 모델도 연구되고 있는데, 이러한 모델의 설명 가능성(Explainability)을 높이는 작업이 필요합니다. 또, 이 논문은 기존 문헌에서 설명 가능성과 사회적 상호작용의 중요성을 간과하는 문제를 지적합니다.

- **Performance Highlights**: 다양한 최신 머신 러닝 및 딥러닝 모델을 검토한 결과, 크게 발전한 모델로는 DepressionNet과 EDNet 등이 있으며, 이러한 모델은 정신 질환의 조기 진단에 유망한 도구로 평가됩니다. 그러나, 블랙박스 모델의 활용은 의료 결정에서 안전성 문제를 야기할 수 있어, 설명 가능한 AI 모델로의 전환이 필요합니다.



### General Distribution Learning: A theoretical framework for Deep Learning (https://arxiv.org/abs/2406.05666)
Comments:
          arXiv admin note: text overlap with arXiv:2105.04026 by other authors. arXiv admin note: text overlap with arXiv:2105.04026 by other authors

- **What's New**: 새로운 연구 논문에서는 딥 러닝(deep learning, DL)에서 아직 해결되지 않은 여러 중요한 질문들을 다루기 위해 GD Learning이라는 새로운 이론적 학습 프레임워크를 도입합니다. 이 프레임워크는 분류(classification), 회귀(regression), 파라미터 추정(parameter estimation)을 포함한 다양한 머신 러닝 및 통계적 과제를 해결하는 데 초점을 맞추고 있습니다. 특히, GD Learning은 데이터 부족 상황에서 외부 지식을 통합하여 학습 오류를 최소화하는 것을 목표로 합니다.

- **Technical Details**: GD Learning 프레임워크는 학습 오류를 모델 및 학습 알고리즘에 의한 피팅 오류(fitting errors)와 제한된 샘플링 데이터로 인한 샘플링 오류(sampling errors)로 나눌 수 있습니다. 또한, 비정형(non-uniformity) 자코비안 행렬(Jacobian matrix)의 고유값(eigenvalues)를 이용하여 Gradient 구조 제어 알고리즘(Gradient Structure Control Algorithm)을 통해 글로벌 최적 해(global optimal solution)에 접근할 수 있음을 보여줍니다. 이러한 구조는 비콘백스(non-convex) 최적화 문제들, 예를 들어 피팅 오류 최소화에서도 사용될 수 있습니다.

- **Performance Highlights**: GD Learning은 과적합(overparameterization), 비콘백스 최적화(non-convex optimization), 편향-분산 트레이드오프(bias-variance trade-off)와 평평한 최소값(flat minima)의 메커니즘 등 딥 러닝에서의 해결되지 않은 질문들에 대해 새로운 통찰을 제공합니다. 기존의 통계적 학습 이론과는 다르게 진정한 기본 분포(underlying distribution)에 초점을 맞추면서, 성능을 향상시킬 수 있는 실질적인 방법을 제시합니다.



### DomainRAG: A Chinese Benchmark for Evaluating Domain-specific Retrieval-Augmented Generation (https://arxiv.org/abs/2406.05654)
- **What's New**: 최근 발표된 논문에서는 Retrieval-Augmented Generation (RAG) 모델이 대형 언어 모델(LLMs)의 한계를 극복하는 해결책으로 주목받고 있습니다. 특히 전문가와 도메인 별 애플리케이션에서 RAG 모델의 필요성이 강조되었습니다. 해당 논문에서는 대학 입학 시나리오에서 RAG 모델의 성능을 평가하였으며, 여섯 가지 필수 능력을 분석하였습니다.

- **Technical Details**: RAG 시스템을 이해하기 위해 논문에서 제시된 여섯 가지 주요 능력은 다음과 같습니다: 1) 대화형 RAG에서의 능력, 2) 구조적 정보 분석, 3) 외부 지식의 신뢰성, 4) 노이즈 제거(denoising), 5) 시간에 민감한 문제 해결, 6) 다중 문서 상호작용 이해. 이 능력들을 평가하기 위해 각 능력에 대응하는 데이터셋이 제공되었습니다. 평가된 모델은 Llama, Baichuan, ChatGLM, GPT 등입니다.

- **Performance Highlights**: 실험 결과 기존의 'closed-book' LLM는 도메인 별 질문에 대처하는 데 어려움을 겪었으며, 이는 RAG 모델이 전문가 문제를 해결하는 데 필요하다는 것을 강조합니다. 또한, 대화 히스토리 이해, 구조적 정보 분석, 노이즈 제거, 다중 문서 상호작용, 외부 지식의 신뢰성 측면에서 RAG 모델의 향상 가능성이 있는 것으로 나타났습니다.



### Separating the "Chirp" from the "Chat": Self-supervised Visual Grounding of Sound and Languag (https://arxiv.org/abs/2406.05629)
Comments:
          Computer Vision and Pattern Recognition 2024

- **What's New**: DenseAV는 비디오만을 통해 고해상도, 의미적으로 의미 있는, 오디오-비주얼(AV) 정렬 피처(feature)를 학습하는 새로운 듀얼 인코더(dense encoder) 그라운딩 아키텍처를 도입했습니다. 이 시스템은 명확한 위치 지정 감독 없이도 단어의 '의미'와 소리의 '위치'를 발견할 수 있습니다. 또한, 두 가지 유형의 연관성을 자동으로 발견하고 구분합니다.

- **Technical Details**: DenseAV는 새로운 multi-head feature aggregation 연산자를 사용하여, 밀집된 이미지와 오디오 표현을 대조 학습(contrastive learning)을 통해 직접 비교합니다. 이를 통해 DenseAV는 음성과 비주얼 신호 간의 높은 품질의 지역 표현을 학습합니다. 또한, DenseAV는 두 개의 새로운 데이터셋을 도입해 음성과 소리 기반의 의미 분할(semantic segmentation)을 평가합니다. 이 데이터셋은 ADE20K 데이터셋에서 제공하는 고품질 분할 마스크를 기반으로 구축되었습니다.

- **Performance Highlights**: DenseAV는 음성과 소리 기반의 의미 분할에서 이전의 최첨단 기술인 ImageBind를 크게 능가합니다. 또한, DenseAV는 동일한 작업에서 ImageBind의 절반 이하의 매개변수만을 사용하면서 뛰어난 성능을 보입니다. 이로 인해 DenseAV는 새로운 소리와 자원이 적은 언어에 대한 적용 가능성을 보유하게 됩니다.



### Toward Reliable Ad-hoc Scientific Information Extraction: A Case Study on Two Materials Datasets (https://arxiv.org/abs/2406.05348)
- **What's New**: 이번 연구는 GPT-4가 과학 문헌에서 adhoc 스키마 기반 정보 추출을 수행할 수 있는 능력을 탐구합니다. 기존의 수작업으로 추출된 두 가지 재료 과학 데이터셋을 재현할 수 있는지 평가하며, 모델이 희망하는 정보를 정확히 추출하는 데 어려움을 겪는 부분을 구체적으로 분석하였습니다.

- **Technical Details**: 연구는 두 가지 전문가가 수작업으로 추출한 재료 특성 데이터셋을 사용했습니다. 하나는 다중-주요 원소 합금(MPEAs)에 관한 것이고, 다른 하나는 실리케이트 녹아내림의 요소 확산에 관한 것입니다. 모델의 성능을 평가하기 위해 재료 과학자들이 오류 분석을 수행했습니다. GPT-4는 스키마에 따라 데이터를 추출하고, 내러티브나 기존의 표에서 잘 작동했지만, 그래프와 PDF 파싱 이슈에서 많은 오류가 발생했습니다.

- **Performance Highlights**: GPT-4는 내러티브나 표 형식에서 정보를 상당히 잘 추출하는 능력을 보였지만, 그래프와 PDF 파싱 문제에서 상당한 오류를 보였습니다. 추가적으로, 비표준 표 형식, 추출된 값의 후처리 필요성, 그리고 향상된 프롬프트 엔지니어링이 요구되는 진정한 읽기 이해 오류도 주요 오류 원인이었습니다.



### TLEX: An Efficient Method for Extracting Exact Timelines from TimeML Temporal Graphs (https://arxiv.org/abs/2406.05265)
Comments:
          25 pages, 9 figures

- **What's New**: 이번 연구에서는 TimeML 주석(texts)로부터 완전한 이벤트 타임라인을 추출하는 TLEX (TimeLine EXtraction)이라는 새로운 시스템을 개발했습니다. TLEX는 기존의 타임라인 추출 방법들보다 정확하며, 특히 이벤트의 불일치 및 불확정 섹션을 자동으로 식별하는 두 가지 새로운 기능을 추가했습니다.

- **Technical Details**: TLEX는 TimeML 주석을 트렁크와 브랜치 구조로 배열된 타임라인 컬렉션으로 변환합니다. 기존 작업과 마찬가지로, TLEX는 시간 그래프의 일관성을 검사하고 정렬합니다. 또한, 특정 관계가 불일치를 초래하는지 식별하고, 타임라인의 불확정 섹션을 식별할 수 있습니다. 이는 자연어 처리 및 이벤트 정렬 작업에 중요한 정보입니다.

- **Performance Highlights**: TLEX는 네 개의 코퍼스로부터 385개의 TimeML 주석 텍스트에 적용되어 실험적 평가를 거쳤으며, 123개의 텍스트가 불일치 상태였으며, 181개 텍스트는 여러 '실제 세계' 또는 주요 타임라인을 가지고 있고, 총 2,541개의 불확정 섹션이 발견되었습니다. 샘플링 평가 결과 TLEX는 다섯 가지 차원에서 98-100%의 정확도를 가지고 있음이 입증되었습니다: 타임포인트의 정렬, 주요 타임라인 수, 주요 및 부속 타임라인의 타임포인트 배치, 브랜치 타임라인의 연결 포인트, 불확정 섹션의 위치.



### Corpus Poisoning via Approximate Greedy Gradient Descen (https://arxiv.org/abs/2406.05087)
- **What's New**: 새로운 연구는 정보 검색 시스템에서의 코퍼스(poisoning attack)를 효과적으로 실행할 수 있는 새로운 공격 방법인 'Approximate Greedy Gradient Descent (AGGD)'을 제안합니다. 이 연구는 기존 HotFlip 방법의 한계를 극복하고, 더 구조적인 검색을 통해 더 높은 질의 토큰 수준의 변형을 선택할 수 있음을 보입니다.

- **Technical Details**: AGGD는 랜덤하게 토큰을 샘플링하는 대신 모든 토큰 위치에서 최상위 토큰을 선택하여 점진적 경사 하강법(Greedy Gradient Descent)을 사용합니다. 이는 AGGD의 검색 궤적을 결정적(deterministic)으로 만들어 더 구조적인 최선-우선 검색(best-first search)을 가능하게 합니다. 실험 결과, AGGD는 NQ와 MS MARCO 데이터셋에서 기존 HotFlip보다 각각 17.6% 및 13.37% 높은 공격 성공률을 기록했습니다.

- **Performance Highlights**: AGGD는 여러 데이터셋과 검색 모델에서 높은 공격 성공률을 달성했습니다. 특히 ANCE 검색 모델을 공격할 때, NQ 데이터셋 코퍼스의 단 0.00003%, MS MARCO 데이터셋의 0.00001%에 해당하는 하나의 적대적 패세지를 주입함으로써, NQ 데이터셋에서 44.35%, MS MARCO 데이터셋에서 26.16%의 공격 성공률을 보여주었습니다. 또한 AGGD는 다른 도메인의 새로운 질의에 대해서도 82.28%의 공격 성공률을 기록했습니다.



### CHIQ: Contextual History Enhancement for Improving Query Rewriting in Conversational Search (https://arxiv.org/abs/2406.05013)
- **What's New**: 이 논문에서는 오픈소스 대형 언어 모델(LLMs)을 효과적으로 활용하여 대화형 검색에서 모호한 쿼리를 개선하는 방법을 연구합니다. 새로운 'CHIQ' 방법을 도입하여, 대화 기록에서 모호성을 해결한 후 쿼리를 재작성하는 두 단계 방식을 제안합니다. 이는 주로 폐쇄형 LLMs를 사용하는 기존 연구들과는 대조적입니다. 5개의 주요 벤치마크에서 CHIQ가 대부분의 설정에서 최첨단 성능을 보임을 입증했습니다.

- **Technical Details**: CHIQ는 대화 기록의 모호성을 해결하기 위해 NLP 과제 해결 능력을 갖춘 LLM을 사용합니다. 본 연구에서는 LLaMA-2-7B와 같은 오픈소스 LLM을 사용하여, 컨텍스트를 확장하거나 코리퍼런스(coreference) 관계를 해결하고 대화 기록을 개선해 쿼리의 적절성을 높입니다. 이처럼 개선된 대화 기록을 기존 프레임워크에 통합하는 다양한 방법을 조사하였습니다.

- **Performance Highlights**: 광범위한 실험 결과, CHIQ는 밀집(dense) 및 희소(sparse) 검색 설정에서 대부분의 벤치마크에서 최첨단 성능을 달성하였습니다. 폐쇄형 LLM과 비교했을 때, 개선된 대화 기록을 사용할 때 성능 격차가 상당히 좁아졌습니다. 이는 오픈소스 LLM이 상업용 모델과 경쟁할 수 있는 가능성을 보여줍니다.



### QAGCF: Graph Collaborative Filtering for Q&A Recommendation (https://arxiv.org/abs/2406.04828)
- **What's New**: Q&A 플랫폼의 새로운 추천 모델, QAGCF(Graph Collaborative Filtering)가 제안되었습니다. 이 모델은 기존 추천 시스템의 한계를 극복하고 질문과 답변 쌍의 협업 및 의미적 정보를 효과적으로 분리하여 사용자의 클릭 행동을 더 정확하게 예측합니다.

- **Technical Details**: QAGCF는 그래프 신경망(neural network) 모델을 기반으로하여 협업 뷰와 의미 뷰를 분리하여 각각의 협업 및 의미 정보를 분리합니다. 협업 뷰에서는 사용자가 클릭한 질문과 답변을 개별적으로 모델링하며, meaning view에서는 질문과 답변 사이, 그리고 질문-답변 쌍들 간의 의미적 연결을 캡처합니다. 이 두 뷰는 글로벌 그래프로 결합되어 전체적인 협업 및 의미 정보를 통합합니다. 글로벌 그래프에서 고차 동조성(high heterophily) 문제를 해결하기 위해 다항식 기반 그래프 필터(polynomial-based graph filters)를 사용하며, 강건한 임베딩(robust embedding)을 얻기 위해 대조 학습(contrastive learning)도 활용합니다.

- **Performance Highlights**: 산업 및 공개 데이터셋에 대한 광범위한 실험 결과 QAGCF가 지속적으로 기존 방법들을 능가하고 최첨단 성과를 달성함을 입증하였습니다.



### Scaling Automatic Extraction of Pseudocod (https://arxiv.org/abs/2406.04635)
- **What's New**: 본 연구에서는 약 32만 개의 가짜 코드(pseudocode) 예제를 포함한 대규모 컬렉션이 제공되었습니다. 이 컬렉션은 arXiv 논문에서 추출된 것으로, 이는 알고리즘 이해를 높이고 자동 코드 생성 및 Optical Character Recognition (OCR) 등의 작업에 유용할 수 있습니다. arXiv 논문 220만 편을 스캔하였으며, 그 중 1,000편은 수작업으로 점검 및 레이블링되었습니다.

- **Technical Details**: 가짜 코드 추출을 위해 arXiv 논문의 LaTex 파일 및 PDF 파일을 분석하는 메커니즘을 개발했습니다. LaTex 파일에서는 명령어를 통해 상대적으로 쉽게 가짜 코드를 추출할 수 있으나, PDF 파일에서는 텍스트와 그림의 경계를 감지하고 이를 추출하는 것이 복잡한 작업입니다. 이를 위해 머신 러닝 기반의 도구가 사용되었습니다.

- **Performance Highlights**: 통계 분석 결과, arXiv 논문에서 가짜 코드 사용이 지수적 증가를 보이고 있음을 밝혔습니다. 또한, 가짜 코드의 클러스터링과 주제별 분석을 통해 다양한 가짜 코드 구조를 조사했습니다.



### Better Late Than Never: Formulating and Benchmarking Recommendation Editing (https://arxiv.org/abs/2406.04553)
- **What's New**: 이번 논문에서는 'recommendation editing(추천 편집)'이라는 새로운 과제를 제안합니다. 이는 기존의 추천 시스템이 제공하는 부적절한 추천을 수정하는 방법으로, 기존 모델을 재학습(retraining)하거나 원본 학습 데이터를 접근하지 않고도 부적절한 아이템을 제거하는 데 중점을 둡니다.

- **Technical Details**: 추천 편집 문제는 세 가지 주요 목표를 정의합니다: (1) 엄격한 수정(Strict Rectification)은 중대한 문제를 유발하는 부적절한 추천 아이템을 제거하는 것입니다. (2) 협력적 수정(Collaborative Rectification)은 관찰되지 않은 유사한 부적절한 추천 아이템도 제거하는 것입니다. (3) 집중적 수정(Concentrated Rectification)은 적절한 추천이 대부분 유지되도록 하는 것입니다. 이를 위해, 새로운 'Editing Bayesian Personalized Ranking Loss'를 기반으로 하는 간단하지만 효과적인 기준점을 제안합니다.

- **Performance Highlights**: 제안된 방법의 효과를 입증하기 위해 다양한 관련 분야의 방법들을 통합한 포괄적인 벤치마크를 설립했습니다. 이를 통해, 제안된 추천 편집 방법이 부적절한 추천 문제를 완화하는 데 실질적으로 효과적임을 보여주었습니다.



### Innovations in Cover Song Detection: A Lyrics-Based Approach (https://arxiv.org/abs/2406.04384)
Comments:
          6 pages, 3 figures

- **What's New**: 이 논문에서는 커버 곡(cover song)을 자동으로 식별하는 새로운 방법을 제안합니다. 특히, 기존의 오디오 분석에만 의존하는 접근법과 달리, 이 방법은 곡의 가사를 활용합니다. 이를 위해 새로운 데이터셋을 구축했으며, 이 데이터셋에는 5078개의 커버 곡과 2828개의 원곡이 포함되어 있습니다. 모든 곡에는 주석이 달린 가사가 첨부되어 있습니다.

- **Technical Details**: 제안된 방법은 Levenshtein 거리와 단어 오류율(WER)을 사용하여 원곡과 커버 곡 사이의 가사 유사성을 평가합니다. 이를 위해 Levenshtein 거리 및 단어 오류율을 계산하는 기존 구현을 활용합니다. 또한, 텍스트 전처리 및 임베딩 생성을 위해 사전 학습된 XLM-RoBERTa 모델을 사용합니다. 이 임베딩 벡터를 기반으로 커버 곡과 원곡 사이의 유사성을 계산하여 가장 유사한 곡을 예측합니다. 모델 훈련에는 삼중 항 손실(triplet loss) 방법을 사용하여 유사한 샘플 간의 거리를 최소화하고 비유사한 샘플 간의 거리를 최대화합니다.

- **Performance Highlights**: 제안된 방법은 기존의 여러 기준 방법들보다 더 나은 성능을 보여주었습니다. 평가 메트릭으로는 평균 정밀도(mAP), 평균 순위(MR), 정밀도@1(P@1)을 사용하였으며, 이는 커버 곡 탐지 분야에서 널리 사용되는 메트릭입니다. 이러한 결과는 가사를 활용한 커버 곡 식별 방법의 우수성을 입증합니다.



### Dynamic Online Recommendation for Two-Sided Market with Bayesian Incentive Compatibility (https://arxiv.org/abs/2406.04374)
- **What's New**: 이 논문은 인터넷 경제에서 중요한 역할을 하는 추천 시스템의 설계 과정에서 직면하는 두 가지 주요 문제, 즉 (1) 새로운 제품 탐색과 이미 알려진 사용자 선호도 활용 간의 탐색-활용 균형 문제와 (2) 사용자들의 자발적 행동과 이질적 선호도를 고려한 동적인 인센티브 호환성 문제를 공식화했습니다. 이를 해결하기 위해 동적 베이지안 인센티브 호환 추천 프로토콜(DBICRP)을 제안하고, RCB라는 두 단계 알고리즘을 개발했습니다.

- **Technical Details**: RCB 알고리즘은 첫 번째 단계에서 동적 인센티브 호환성을 유지하면서 충분한 샘플 크기를 결정하기 위해 제품을 탐색하고, 두 번째 단계에서는 반비례 샘플링을 사용하여 적은 후회를 보장합니다. 이 알고리즘은 가우시안 사전(gaussian prior) 가정 하에서 베이지안 인센티브 호환성(Bayesian Incentive Compatibility, BIC)을 만족함을 이론적으로 증명했습니다.

- **Performance Highlights**: RCB 알고리즘은 후회(regret)가 $O(\sqrt{KdT})$임을 이론적으로 증명했으며, 시뮬레이션 및 실제 사례(예: 맞춤형 와파린 개인별 투여량)에서 강력한 인센티브 효과와 적은 후회, 높은 강건성을 입증했습니다.



### Multi-Head RAG: Solving Multi-Aspect Problems with LLMs (https://arxiv.org/abs/2406.05085)
- **What's New**: MRAG(다중 헤드 검색 증강 생성 모델)은 대규모 언어 모델(LLM)이 다양한 내용이 담긴 여러 문서를 검색해야 하는 쿼리를 처리할 수 있도록 설계되었습니다. 기존의 RAG(검색 증강 생성) 솔루션은 이러한 쿼리를 처리하는 데 어려움을 겪었으나, MRAG는 변환기(Transformer)의 다중 헤드 주의층(multi-head attention layer) 활성화를 활용해 이 문제를 해결합니다. 다양한 주의 헤드가 여러 데이터 측면을 학습할 수 있도록 함으로써 복잡한 쿼리에 대한 검색 정확도를 향상시킵니다.

- **Technical Details**: MRAG는 기본적인 RAG 설계를 향상시키는 간단하지만 강력한 접근법을 제안합니다. 기존의 마지막 층 디코더(decoder) 블록의 활성화를 키로 사용하지 않고, 다중 헤드 주의층의 활성화를 키로 사용하여 다중 측면 문서를 검색합니다. 이러한 다중 측면 임베딩(embedding)을 데이터 항목과 쿼리 표현 모두에 직접 사용합니다. MRAG는 새로운 평가 방법론과 메트릭, 합성 데이터셋 및 실제 사례를 통해 그 효과를 입증합니다. MRAG의 코드와 관련 자료는 공개되어 있으며, RAGAS와 같은 벤치마킹 도구 및 다양한 데이터 스토어 클래스와 쉽게 통합될 수 있습니다.

- **Performance Highlights**: MRAG는 복잡한 쿼리에 대한 검색 정확도에서 기존 RAG 기반보다 최대 20% 향상된 성능을 보여줍니다. 예를 들어, 다중 측면 위키피디아(Wikipedia) 기사 검색에서 20% 향상을 보였습니다. 이러한 다중 측면 임베딩 아이디어는 추가적인 공간 요구 없이 RAG의 성능을 향상시킵니다.



### Error Bounds of Supervised Classification from Information-Theoretic Perspectiv (https://arxiv.org/abs/2406.04567)
- **What's New**: 이번 연구는 정보이론적 관점에서 심층 신경망(DNN)을 사용한 지도 분류의 이론적 기초를 탐구하며, 오버파라미터화된 신경망의 일반화 능력, 비볼록 최적화 문제에서의 효율적 성능, 플랫 최소값(flat minima)의 메커니즘을 설명하는 새로운 개념을 도입했습니다. 본 논문에서는 fitting error(맞춤 오류)와 model risk(모델 위험)을 소개하여 기존의 generalization error(일반화 오류)와 함께 기대 위험의 상한을 형성합니다.

- **Technical Details**: 일반화 오류가 데이터 분포의 스무스성과 샘플 크기에 의해 영향을 받는 복잡성에 의해 제한됨을 증명했습니다. 우리는 NTK(Neural Tangent Kernel) 및 모델의 파라미터 수와 fitting error의 상관관계를 도출합니다. KL 발산(Kullback-Leibler divergence)을 사용하여 기존 손실 함수의 의존성을 제거하고, 삼각부등식(triangle inequality)을 활용하여 기대 위험의 상한을 설정했습니다.

- **Performance Highlights**: 실증 검증은 도출된 이론적 상한과 실제 기대 위험 사이에 유의미한 양의 상관관계가 있음을 보여, 이론적 발견의 실용성을 확인했습니다. 작은 최대의 eNTK(equal-input Neural Tangent Kernel, λ_max(H(fθ(x))))은 기대 위험을 최소화하는 데 유리한 것으로 증명되었습니다.



### GNNAnatomy: Systematic Generation and Evaluation of Multi-Level Explanations for Graph Neural Networks (https://arxiv.org/abs/2406.04548)
- **What's New**: 새로운 연구는 다양한 하위 구조를 체계적으로 탐색하고 결과를 평가하는 데 어려움을 겪는 기존 방법론의 한계를 극복하기 위해 GNNAnatomy라는 시각적 분석 시스템을 소개합니다. 이는 그래프 수준의 분류 작업에서 GNN의 동작을 설명하기 위해 그래프렛(graphlets)을 사용하며, 가설적 사실(factual) 및 반사실적(counterfactual) 설명을 통해 GNN의 행동을 분석합니다.

- **Technical Details**: GNNAnatomy 시스템은 모델 및 데이터셋에 독립적으로 작동하며, 그래프렛(graphlets)을 사용해 GNN의 예측과 그래프렛 빈도의 상관 관계를 분석하여 설명을 생성합니다. 구체적으로, (1) 그래프렛 빈도와 분류 신뢰도 간의 상관 관계와 (2) 원래 그래프에서 해당 하위 구조를 제거한 후 분류 신뢰도의 변화를 평가하는 두 가지 측정을 도입합니다. 실제로 그래프렛 빈도를 계산하는 것은 NP-hard 문제이므로, GNNAnatomy는 샘플링 방법을 사용하여 3, 4, 5개의 노드를 가진 그래프렛의 빈도를 계산합니다.

- **Performance Highlights**: 실제 데이터셋과 합성 데이터셋을 활용한 사례 연구에서 GNNAnatomy는 효과적인 설명을 제공하는 것으로 입증되었습니다. 또한, 최신 GNN 설명자(state-of-the-art GNN explainer)와 비교하여 그 설계의 유용성과 다용성을 보여주었습니다.



### Negative Feedback for Music Personalization (https://arxiv.org/abs/2406.04488)
Comments:
          6 pages, 4 figures, accepted to ACM UMAP 2024

- **What's New**: 이번 연구에서는 인터넷 라디오의 Next-Song 추천 시스템에서 실제 부정 피드백을 활용하여 학습 속도와 정확성을 크게 개선할 수 있음을 입증했습니다. 또한, 사용자 피드백 시퀀스에 스킵(건너뛰기) 데이터를 추가함으로써 사용자 커버리지와 정확성을 모두 개선하는 방법을 제안했습니다.

- **Technical Details**: 본 연구에서는 SASRec와 BERT4Rec와 같은 기존의 Transformer 아키텍처를 바탕으로 한 추천 시스템을 참고했습니다. 또한, 부정 표본으로 랜덤 샘플을 사용하는 대신 실제 사용자로부터 수집된 명시적 부정 피드백(예: 'thumb-down')을 사용하여 모델을 학습시켰습니다. 이를 통해, 학습 시간을 약 60% 절감하고 테스트 정확도를 약 6% 개선할 수 있음을 확인했습니다.

- **Performance Highlights**: 실험 결과, 명시적 부정 피드백을 포함한 모델이 더 적은 학습 시간으로 더 높은 정확도를 보였으며, 특히 부정 표본으로 사용된 데이터의 양이 적절했을 때 최상의 성능을 발휘했습니다. 또한, 스킵 데이터를 추가로 입력하여 개인화된 추천의 범위를 확대하고 정확도도 약간 향상시켰습니다.




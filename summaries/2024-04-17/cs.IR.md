### Spiral of Silences: How is Large Language Model Killing Information  Retrieval? -- A Case Study on Open Domain Question Answering (https://arxiv.org/abs/2404.10496)
- **What's New**: 이 연구는 Retrieval-Augmented Generation (RAG) 시스템에 대한 인공지능(AI) 대형 언어 모델(LLM: Large Language Models) 생성 텍스트의 단기 및 장기 효과를 탐구합니다. RAG 시스템을 이용하여 자동응답 분야(ODQA: Open Domain Question Answering)에서의 성능을 테스트하였습니다. 특히, 이 연구는 LLM 생성 텍스트가 인간 작성 내용보다 검색 순위에서 높은 성능을 내며 이로 인해 'Spiral of Silence' 현상을 초래할 수 있는 가능성을 밝혀냈습니다.

- **Technical Details**: 연구 팀은 RAG 시스템에 대한 LLM 생성 텍스트의 영향을 모니터링하기 위해 시뮬레이션 파이프라인을 구축하였습니다. 이 파이프라인은 연속적으로 LLM 생성 텍스트를 웹 데이터셋에 통합하고, 이로 인한 RAG 시스템의 성능 변화를 반복적으로 평가합니다. 주로 사용된 기술은 검색, 재랭킹 방법(re-ranking methods), 그리고 질의 응답 성능 평가 방법이 포함되었습니다.

- **Performance Highlights**: LLM 생성 텍스트는 단기적으로는 검색 성능을 향상시키나, 장기적으로는 검색의 효율성이 감소하고 있습니다. 반복적인 실험을 통해 검출된 'Spiral of Silence' 현상은 LLM 생성 텍스트가 검색 결과에서 인간 작성 텍스트보다 일관되게 높은 순위를 차지하면서 나타났습니다. 이는 실제 정보의 다양성과 정확성이 위협받을 수 있음을 시사합니다.



### Promoting the linguistic diversity of TEI in the Maghreb and the Arab  region (https://arxiv.org/abs/2404.10371)
- **What's New**: 이 프로젝트는 막렙(Maghreb) 지역에서 구술 말뭉치(oral corpus)와 풍부한 문헌 자료(text resources)를 대상으로 합니다. 특히, 12세기 이상 지속되어 온 여전히 활발한 고전 아랍어(classical Arabic)와, 리비아, 로마, 히브리, 오스만의 영향과 최근의 프랑스, 스페인, 이탈리아어로 인한 언어 간섭(linguistic interference)에 의해 지탱되는 방언들의 극단적인 혼합(hybridization)에 중점을 둡니다. 간단히 말해서, 막렙은 매우 풍부하지만 아직 충분히 활용되지 않는(textual studies) 텍스트 연구의 장소입니다.

- **Technical Details**: 이 연구는 막렙 지역의 다양한 언어적 영향을 조명하면서, 고전 아랍어와 그 지역 방언들 사이의 연속성에 초점을 맞춥니다. 여러 외래 언어들의 간섭으로 인해 생성된 혼합 언어의 복잡성을 탐구하고 이를 분석하기 위해 다양한 언어학적 도구와 방법론이 사용될 예정입니다.

- **Performance Highlights**: 이 연구는 막렙 지역의 언어적 복잡성과 그 지역의 언어가 시간에 따라 어떻게 발전하였는지에 대한 깊은 이해를 제공합니다. 또한, 이러한 지역의 언어적 특성을 보존하고, 학문적으로 그 가치를 끌어올릴 수 있는 기회를 제공할 것입니다.



### Exact and Efficient Unlearning for Large Language Model-based  Recommendation (https://arxiv.org/abs/2404.10327)
- **What's New**: 새롭게 제시된 Large Language Model 기반 추천 시스템(Large Language Model-based Recommendation, LLMRec)은 사용자 데이터의 개인정보 보호 문제를 해결하기 위해 데이터를 효율적으로 제거하는 (unlearning) 기능을 갖추었습니다. 특히, 이 연구에서는 Adapter Partition and Aggregation (APA) 프레임워크를 도입하여 데이터 셔드(data shards)에 대한 어댑터(adapter)를 분리하고 필요한 부분만 재학습(retraining)하여 불필요한 데이터를 정확하고 효율적으로 제거할 수 있게 하였습니다.

- **Technical Details**: APA 프레임워크는 각 테스팅 샘플에 대해 샘플적응적 주의기능(sample-adaptive attention)을 이용해 파라미터 수준에서 어댑터의 집합(parameter-level adapter aggregation)을 수행합니다. 이 방법은 추천 성능을 유지하면서도 추론 비용(inference costs)을 크게 줄일 수 있는 핵심 기술입니다. 연구팀은 광범위한 실험을 통해 이 프레임워크의 효과와 효율성을 입증했습니다.

- **Performance Highlights**: APA 프레임워크의 도입은 추천 시스템에서 데이터를 제거하는 과정에서 발생할 수 있는 성능 저하를 최소화하면서 비용 효율적인 운영을 가능하게 합니다. 학습 시 데이터의 정확한 파티셔닝(partitioning)과 어댑터의 효과적인 재학습은 기존 대비 향상된 불필요 데이터 제거와 프라이버시 보호를 가능하게 하며, 이는 추천 시스템의 신뢰성을 크게 증가시킵니다.



### Cluster-based Graph Collaborative Filtering (https://arxiv.org/abs/2404.10321)
Comments: 22 pages, 8 figures

- **What's New**: 이 논문에서는 사용자와 아이템의 다양한 관심사를 반영하여 효과적인 추천 시스템을 제공하는 새로운 GCN(Graph Convolution Networks)-기반 모델인 ClusterGCF(Cluster-based Graph Collaborative Filtering)를 제안합니다. 이 모델은 사용자와 아이템 노드를 여러 클러스터로 분류하는 비지도 학습 및 최적화 가능한 소프트 노드 클러스터링 접근법을 사용합니다.

- **Technical Details**: ClusterGCF는 사용자와 아이템의 다중 관심사를 캡처하고 공통 관심사를 식별하여 클러스터별 그래프를 구성합니다. 소프트 노드 클러스터링 결과와 사용자-아이템 상호작용 그래프의 토폴로지를 기반으로 노드가 다른 클러스터에 속할 확률을 할당하여, 클러스터별 그래프를 구축합니다. 또한, 모델은 높은 순서의 이웃 노드로부터 더 가치 있는 정보를 캡처하면서, 불필요한 정보를 필터링해내는 고차 그래프 컨볼루션을 수행합니다.

- **Performance Highlights**: ClusterGCF는 네 개의 공개 데이터셋에서 광범위한 실험을 통해 검증되었습니다. 실험 결과, 본 모델은 최신 GCN 기반 추천 시스템 대비 우수한 성능을 보여주었습니다. 모델은 특히 클러스터별 그래프에서 고차 이웃 노드의 정보를 통해 노드 임베딩의 고유성을 유지하는 데 도움을 주는 것으로 나타났습니다. 이는 클러스터링 접근법의 효과성을 입증하며, 이 논문의 연구 결과와 함께 출시된 코드 및 설정은 다른 연구자들이 이 작업을 반복할 수 있도록 지원합니다.



### Compressible and Searchable: AI-native Multi-Modal Retrieval System with  Learned Image Compression (https://arxiv.org/abs/2404.10234)
- **What's New**: 이 연구는 AI 기술을 활용하여 다중 모드 검색 기능과 신경 이미지 압축을 통합하는 새로운 프레임워크를 제안합니다. Learned Image Compression (LIC)과 Contrastive Language-Image Pretraining (CLIP) 기능을 결합하여 저장 및 검색 시스템의 효율성을 개선합니다.

- **Technical Details**: 저자들은 이미지의 압축 가능성과 검색 가능성 간의 복잡한 관계를 분석하고, 이 두 요소가 저장 및 검색 시스템의 효율성에 얼마나 중요한지 인식합니다. 실험은 Kodak 데이터셋에서 수행되어 기존 방법론과 비교할 때 압축 효율과 검색 정확도에서 우수한 성능을 보였습니다. 이 프레임워크는 간단한 어댑터를 사용하여 LIC의 특징과 CLIP을 연결하며, 다중 모달 데이터의 의미적 충실도와 검색을 유지합니다.

- **Performance Highlights**: 제안하는 방법은 전통적인 압축 알고리듬과 비교하여 더 높은 압축 비율을 달성하며 본질적인 이미지 구조를 활용합니다. 또한, 대규모 사전 훈련된 모델을 사용하여 텍스트-이미지, 이미지-텍스트, 이미지-이미지 간 검색이 가능하며, 이는 다양한 검색 작업에 걸쳐 풍부한 의미 정보를 활용할 수 있게 합니다.



### LegalPro-BERT: Classification of Legal Provisions by fine-tuning BERT  Large Language Mod (https://arxiv.org/abs/2404.10097)
Comments: 17 pages, 4 figures

- **What's New**: 이 연구에서는 계약 분석을 위한 새로운 AI 기반 시스템, LegalPro-BERT를 제안하고 있습니다. 이 모델은 법률 문서 내의 중요 조항과 내용을 확인하고 분류하는 작업을 자동화하기 위해 설계되었습니다. 특히, 전문 법률 용어를 이해하고 처리할 수 있도록 맞춤형으로 학습된 BERT(transformer architecture) 모델을 사용하여 계약서 검토 프로세스의 효율성을 높이는 것을 목표로 합니다.

- **Technical Details**: LegalPro-BERT는 사전 학습된 대형 언어 모델(pre-trained large language model)을 사용하고, 법률 분야의 특수한 용어와 문법에 적합하도록 조정(calibrated)합니다. 이를 통해 법률 전문 분야에서 부족한 라벨링된 데이터 문제를 해결하고, 법적 문서에서 특별히 사용되는 전문 용어를 정확히 인식할 수 있게 됩니다. 이 모델은 법적 조항의 분류(classification) 작업에 효과적으로 사용될 수 있도록 fine-tune됩니다.

- **Performance Highlights**: LegalPro-BERT는 법률 문서 분석에서 기존 벤치마크와 비교하여 성능이 우수함을 입증했습니다. 모델은 기존의 벤치마크 결과와 비교하여 더 높은 메트릭스(metrics)에서 뛰어난 결과를 보였습니다. 이는 AI 기반 법률 분석 도구의 가능성을 보여주는 중요한 발전입니다.




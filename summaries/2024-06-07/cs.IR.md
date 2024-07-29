### Polyhedral Conic Classifier for CTR Prediction (https://arxiv.org/abs/2406.03892)
- **What's New**: 이 논문은 산업용 추천 시스템에서 click-through rate (CTR) 예측을 위한 새로운 접근 방식을 소개합니다. 데이터셋의 불균형과 기하학적 비대칭성 문제를 해결하기 위해, 다면체 원뿔 함수를 사용하는 심층 신경망 분류기를 제안합니다. 이 분류기는 일종의 one-class classifier와 유사하며, 긍정 클래스 샘플을 분리하기 위한 밀집된 다면체 허용 영역을 반환합니다.

- **Technical Details**: CTR 예측 모델은 embedding layer, feature interaction layer, output layer로 구성됩니다. 본 연구는 다면체 원뿔 함수를 사용하여 긍정 클래스 샘플을 더 정확하게 근사하는 심층 컴팩트 다면체 원뿔 분류기(deep compact polyhedral conic classifier, DCPCC)를 채택했습니다. 이 분류기는 다양한 특징과 맥락적 요소를 포함하여 복잡하고 비선형적인 관계를 모델링합니다. DCPCC는 양성 클래스를 상대적으로 잘 구분하기 위해 hyperplane과 L1 cone의 교차를 사용하여 긍정 클래스에 대한 컴팩트하고 경계가 있는 허용 영역을 반환합니다.

- **Performance Highlights**: 제안된 방법을 Criteo, Avazu, MovieLens, Frappe와 같은 4개의 공개 데이터셋에서 테스트한 결과, 이 방법이 기본 바이너리 크로스 엔트로피(Binary Cross Entropy) 손실을 활용한 모델보다 우수한 성능을 보였습니다. 실험 결과는 제안된 방법이 CTR 예측 작업에서 더 나은 성능을 제공하는 것을 확인했습니다.



### Reducing the climate impact of data portals: a case study (https://arxiv.org/abs/2406.03858)
Comments:
          4 pages

- **What's New**: 최근 정보통신기술(ICT) 분야의 탄소 배출량 증가 문제를 해결하기 위한 논문이 발표되었습니다. 이 프로젝트 제안서는 MediaWiki 기반 지식 베이스인 MaRDI 포털을 더욱 에너지 효율적으로 만드는 방법을 다루고 있으며, 향후 이러한 개선 조치를 구현하고 에너지 효율성 향상에 대한 구체적인 측정치를 제공할 계획입니다.

- **Technical Details**: 연구는 MaRDI 포털의 에너지 효율성을 강화하기 위해 여러 가지 기술적 접근법을 제안합니다. 여기에는 데이터 전송량 최소화, 불필요한 데이터 배달 방지, 자주 요청되는 데이터의 최적화된 가용성 확보, 내부 워크플로우 효율화 등이 포함됩니다. 또한 프로파일링 도구를 사용하여 에너지 소비 핫스팟을 식별하고, 이를 최적화하는 알고리즘을 개발할 계획입니다. HTTP 가속기, Docker 컨테이너 최적화, 데이터 캐싱 및 Manticore Search로의 전환을 통해도 에너지 절약이 기대됩니다.

- **Performance Highlights**: MaRDI 포털은 현재 약 4억 5천만 개의 트리플을 포함하고 있으며, 단일 가상 머신(OpenStack)에서 작동 중입니다. 가상머신은 128GB 메인 메모리와 32개의 CPU, 1500GB HDD를 갖추고 있습니다. 제안된 개선 조치들이 성공적으로 구현되고 평가될 경우, 심각하게 에너지 수요와 탄소 발자국을 줄일 수 있을 것입니다. 예를 들어, 컨테이너 이미지를 메모리에 캐싱하고, 컨테이너를 재시작하는 대신 재생성하는 방식은 12%의 에너지 절약을 예상할 수 있습니다.



### Data Measurements for Decentralized Data Markets (https://arxiv.org/abs/2406.04257)
Comments:
          20 pages, 11 figures

- **What's New**: 이 연구는 기계 학습을 위한 더 공정한 데이터 획득 방법으로서 분산 데이터 시장(decentralized data markets)을 제안하며, 관련성과 다양성을 기반으로 셀러(판매자)를 선택할 수 있는 연합 데이터 측정(federated data measurements) 기법을 소개합니다. 이 연구는 브로커(brokers)와 특정 작업 모델 간의 중재 없이 데이터를 검증할 수 있는 새로운 메커니즘을 제공합니다.

- **Technical Details**: 이 프레임워크는 셀러의 데이터를 직접 액세스하지 않고도 데이터를 비교할 수 있는 이점을 지니며, 교육이 필요 없고 작업마다 다르지 않는(task-agnostic) 특성을 가집니다. 다양한 컴퓨터 비전(computer vision) 데이터셋에서 여러 정의의 다양성과 관련성을 평가하고 벤치마크하여 셀러를 순위 매기는 능력과 분류 성능과의 상관관계, 중복 및 노이즈 데이터에 대한 견고성을 분석합니다.

- **Performance Highlights**: 연합 데이터 측정을 통해 분산 데이터 시장에서 데이터 구매자가 셀러를 효율적으로 찾을 수 있도록 도와 검색 비용을 줄일 수 있는 효과를 보였습니다. 또한, 이 프레임워크는 컴퓨터 비전 데이터셋에서 셀러를 순위 매기는 능력과 분류 성능과 강한 상관관계를 가집니다.



### Beyond Similarity: Personalized Federated Recommendation with Composite Aggregation (https://arxiv.org/abs/2406.03933)
- **What's New**: 새로운 연구는 페더레이티드 추천 시스템(Federated Recommendation)을 위한 복합 집계(Composite Aggregation) 메커니즘을 제안합니다. 이는 기존 클라이언트(유사 클라이언트)의 학습된 임베딩(embedding)을 강화하는 것뿐만 아니라 비학습된 임베딩을 업데이트하는 보완 클라이언트를 통합합니다. 연구 결과는 여러 실제 데이터셋에서 이 모델이 최신 방법들을 능가함을 입증합니다.

- **Technical Details**: 본 연구는 페더레이티드 추천 시스템의 모델들에 적합한 새로운 집계 메커니즘을 제안했습니다. 기존 방법들은 주로 페더레이티드 비전(Federated Vision) 커뮤니티에서 사용된 유사성 기반 집계를 사용했지만, 이러한 방식은 임베딩 스큐(Embedding Skew) 문제를 야기합니다. 이를 해결하기 위해, 기존 유사성 집계와 보완적 집계 메커니즘을 통합한 FedCA(Federated Composite Aggregation)를 도입하여 학습되지 않은 임베딩을 업데이트하고 정보의 다양한 측면을 통합합니다.

- **Performance Highlights**: 제안된 FedCA 모델은 여러 실제 데이터셋(Filmtrust, Movielens)에서 실시된 광범위한 실험을 통해 효율성을 입증했습니다. 테스트 데이터셋에서의 성능 저하를 줄이고, 일반화(generalization) 능력을 향상시켜 미래의 아이템 추천을 더욱 정확하게 할 수 있게 합니다.



### Attribute-Aware Implicit Modality Alignment for Text Attribute Person Search (https://arxiv.org/abs/2406.03721)
- **What's New**: 텍스트 속성(person search) 검색 시스템이 도입되었습니다. 이 시스템은 증인 설명을 바탕으로 특정 보행자를 찾는 문제를 해결하기 위해 고안되었습니다. 주요 도전 과제는 텍스트 속성과 이미지 간의 큰 모달리티 간극입니다. 기존 방법들은 단일 모달리티(pre-trained models)를 통해 명시적 표현과 정렬을 달성하려고 합니다. 그러나 이러한 방법들은 모달리티 간 대응 정보를 포함하지 않아 정보 왜곡 문제가 발생합니다. 이에 따라 Attribute-Aware Implicit Modality Alignment (AIMA) 프레임워크가 제안되었습니다.

- **Technical Details**: AIMA는 로컬 표현의 일치를 학습하고, 글로벌 표현 매칭을 결합하여 모달리티 간격을 좁힙니다. 먼저 CLIP 모델을 백본으로 사용하고, 프롬프트 템플릿(prompt templates)을 설계하여 속성 조합을 구조화된 문장으로 변환합니다. 이후, 마스크 속성 예측(MAP) 모듈을 통해 이미지와 마스크된 텍스트 속성 표현 간의 관계를 암묵적으로 정렬합니다. 마지막으로 Attribute-IoU Guided Intra-Modal Contrastive (A-IoU IMC) 손실을 도입하여 특정 텍스트 속성의 분포를 임베딩 공간에서 정렬하여 더 나은 의미 배치를 달성합니다.

- **Performance Highlights**: Market-1501, PETA 및 PA100K 데이터셋에서 AIMA의 성능이 기존 최첨단 방법들을 상당히 능가했음을 확인했습니다. 특히 CLIP 모델의 미세 조정(fine-tuning)을 통해 탁월한 결과를 얻을 수 있었습니다.




### Can't Hide Behind the API: Stealing Black-Box Commercial Embedding Models (https://arxiv.org/abs/2406.09355)
- **What's New**: 최근 연구는 상업적 목적으로 사용되는 임베딩 모델을 도둑질(steal)할 수 있는 새로운 방법을 제안합니다. OpenAI와 Cohere 같은 회사들이 제공하는 상업적 임베딩 모델은 API를 통해 접근할 수 있으며, 사용자는 이에 대해 비용을 지불해야 합니다. 그러나 이 연구는 이러한 모델을 텍스트-임베딩(input-output pairs)을 수집하여, 로컬 모델을 훈련시킴으로써 성공적으로 복제할 수 있음을 보여줍니다.

- **Technical Details**: 이 연구는 처음으로 상업적 임베딩 모델을 도둑질하는 방법을 소개합니다. 연구진은 OpenAI와 Cohere의 임베딩 API를 통해 텍스트-임베딩 페어를 수집하고, 이를 이용해 로컬 도둑(thief) 모델을 훈련시켰습니다. 이 도둑 모델은 BERT base를 초기화하여 강력한 임베딩 모델을 만드는 방식으로 학습됩니다. 또한, OpenAI와 Cohere의 모델을 단일 학생(student) 임베딩 모델로 증류(distillation)하는 실험도 수행되었습니다.

- **Performance Highlights**: 이 연구는 도둑 모델이 주어진 API 기반 모델과 비슷한 수준의 검색 성능을 비용 효율적으로 재현할 수 있음을 보입니다. 도둑 모델을 훈련하는 데 드는 비용은 약 $200에 불과하며, 적은 차원으로 구현된 모델임에도 높은 성능을 발휘합니다. 이러한 결과는 상업적 임베딩 모델 배포 시 중요한 고려 사항을 제기하며, 모델 도난을 방지하기 위한 방법을 제안합니다.



### Master of Disaster: A Disaster-Related Event Monitoring System From News Streams (https://arxiv.org/abs/2406.09323)
Comments:
          6 pages, 2 figures

- **What's New**: 새로운 오픈소스 재난 모니터링 시스템인 Master of Disaster(MoD)가 소개되었습니다. 이 시스템은 뉴스 스트림을 수신하고, 사건 정보를 추출하여 지식 그래프(Knowledge Graph, KG)인 위키데이터(Wikidata)와 연결하며, 시각적으로 사건 인스턴스를 구별합니다. 이를 통해 동일한 실제 사건 인스턴스를 참조하는 여러 뉴스 기사를 시각적 검토를 통해 구별할 수 있습니다.

- **Technical Details**: MoD 시스템은 데이터 전처리기, 사건 추출기, 사건 시각화 도구, Gradio 기반의 GUI로 구성되어 있습니다. 데이터 전처리기는 GDELT API를 사용하여 사용자 제공 키워드를 포함하는 기사를 쿼리하며, 영어로 작성된 기사만 선택합니다. 사건 추출기는 사전 학습된 RoBERTa 모델을 사용해 TREC-IS 데이터셋으로 미세 조정되었습니다. 또한, BLINK 기반 엔티티 링크를 사용하여 사건 관련 엔티티를 추출하고, RDF 그래프 생성기를 통해 위키데이터로 매핑하여 이벤트 그래프(JSON-LD 형식으로)를 출력합니다. 사건 시각화 도구는 필터링된 기사 스트림을 받아 사건 유형 감지기를 통해 문장 임베딩을 생성하고, PCA를 사용하여 2차원 공간에서 시각화합니다.

- **Performance Highlights**: 모니터링 시스템 MoD의 유용성을 평가하기 위해 진행된 설문 조사 결과, 이벤트 추출 탭 사용 용이성, 정보 정확성, 이벤트 시각화 탭 사용 용이성, 일간 뉴스 개요의 유용성, 빠른 뉴스 접근성에 대해 전반적으로 높은 점수(평균 4.14점)를 받았습니다. 이로써 MoD는 실용적인 반자동 재난 관련 사건 모니터링 시스템으로 실무에 적용될 수 있음을 확인했습니다.



### On Softmax Direct Preference Optimization for Recommendation (https://arxiv.org/abs/2406.09215)
- **What's New**: LM(언어 모델)-기반의 추천 시스템 성능을 개선하기 위해 새로운 'Softmax-DPO' (S-DPO) 방법론이 제안되었습니다. 이 방법론은 사용자의 선호도 데이터를 더욱 효율적으로 활용하고, 기존의 언어 모델링 손실을 대체하여 개인 맞춤형 순위 예측 정확도를 높이는 데 중점을 둡니다.

- **Technical Details**: 기존 LM 기반 추천 시스템은 사용자의 열람 이력을 텍스트 프롬프트로 변환하고, 긍정적인 항목만을 대상으로 LM을 미세 조정(fine-tuning)하는 방식으로 학습합니다. 하지만 새로운 S-DPO 방법론은 이러한 접근 방식의 한계를 극복하기 위해 설계되었습니다. S-DPO는 사용자의 선호도 데이터에 여러 부정 항목(negative items)을 포함시키고, 이러한 데이터로부터 랭킹 정보를 추출하여 LM에 주입합니다. 이를 위해 Softmax 샘플링 전략과 결합된 DPO 손실 함수의 변형을 도입하였습니다.

- **Performance Highlights**: 세 개의 실제 데이터셋에 대한 광범위한 실험 결과, S-DPO 방법론이 기존 방법보다 사용자 선호도를 더욱 효과적으로 모델링하고, 추천 성능을 크게 향상시켰음을 입증하였습니다. 또한, DPO의 데이터 우연성 감소 문제를 완화하는 데도 기여하였습니다.



### Contextual Distillation Model for Diversified Recommendation (https://arxiv.org/abs/2406.09021)
Comments:
          accepted by KDD 2024

- **What's New**: 이 논문에서는 다양한 추천을 위한 Contextual Distillation Model (CDM)을 제안합니다. 이 모델은 기존의 재순위화(re-ranking) 단계뿐만 아니라 사전 순위화(pre-ranking) 및 순위화(ranking) 단계에서도 효율적으로 적용될 수 있습니다. 특히 CDM은 같은 사용자 요청 내의 후보 아이템들을 문맥(context)으로 활용하여 결과의 다양성을 높이는 방식입니다.

- **Technical Details**: CDM은 양성(positive) 및 음성(negative) 문맥을 모델링하기 위해 주의 메커니즘(attention mechanism)을 사용하는 대조적 문맥 인코더(Contrastive Context Encoder)를 제안합니다. CDM은 목표 아이템을 문맥 임베딩(context embedding)과 비교하여 각 목표 아이템의 선발 확률을 학습합니다. 추론 단계에서는 추천 점수와 학생 모델(student model)의 점수를 선형 결합하여 다양성과 효율성을 보장합니다.

- **Performance Highlights**: CDM은 두 개의 산업 데이터셋을 대상으로 오프라인 평가를 수행했으며, 단기 동영상 플랫폼인 쿠아이쇼우(KuaiShou)에서 온라인 A/B 테스트를 진행했습니다. 그 결과, 추천 품질 및 다양성에서 상당한 향상을 보였으며, 이는 CDM의 효과성을 강하게 입증하였습니다.



### Robust Information Retrieva (https://arxiv.org/abs/2406.08891)
Comments:
          accepted by SIGIR2024 Tutorial

- **What's New**: IR 시스템에서의 효과성 이외에도 견고성 (robustness)에 대한 관심이 높아지고 있습니다. IR 시스템이 강력한 성능을 제공하는 것은 물론, 다양한 예외적인 상황도 잘 처리하는 것이 필요합니다. 최근 몇 년 동안 IR의 견고성에 대한 연구가 크게 증가했으며, 광범위한 분석과 다양한 전략이 제안되었습니다. 이번 튜토리얼에서는 기본 정보와 IR의 견고성의 분류를 제공합니다. 더 나아가, IR-specific 맥락에서의 adversarial robustness와 out-of-distribution (OOD) robustness를 검토하며, LLMs와 관련된 IR의 견고성도 논의합니다.

- **Technical Details**: IR 시스템의 견고성은 예상치 못한 다양한 상황에서도 일관된 성능을 유지하는 것을 의미합니다. 특히 두 가지 주요 유형의 견고성이 주목받고 있습니다. 첫째, adversarial robustness는 SEO(검색엔진 최적화)나 웹 스팸과 같은 악의적인 공격으로부터 IR 모델을 방어하는 능력을 의미합니다. 둘째, OOD robustness는 학습 데이터와 다른 분포를 가진 새로운 쿼리와 문서에 대한 성능을 측정합니다. 또한, LLM(large language models)이 IR 시스템에 통합됨에 따라 새로운 견고성 문제가 발생하며, LLM이 IR 시스템의 견고성을 향상시킬 기회를 제공합니다.

- **Performance Highlights**: 최근 연구에서는 다양한 견고성 문제를 해결하기 위한 많은 방법들이 제안되었습니다. 예를 들어, adversarial robustness를 위해서는 corpus poison attacks, backdoor attacks, encoding attacks 등이 연구되었습니다. 이를 방어하기 위해서는 공격 감지, 경험적 방어, 인증된 견고성 등의 방법이 제안되었습니다. 한편, OOD robustness를 위해서는 새로운 코퍼스에 적응하거나 원래 코퍼스를 증분하는 방법이 사용됩니다. 여기에는 데이터 증강, 도메인 모델링, 아키텍처 수정, 모델 용량 확대 등의 접근법이 포함됩니다.



### Self-supervised Graph Neural Network for Mechanical CAD Retrieva (https://arxiv.org/abs/2406.08863)
- **What's New**: 최신 연구에서는 GC-CAD라고 불리는 새로운 자가 지도 학습 기반 그래프 신경망 방법을 제안했습니다. 이는 매커니컬 산업에서 CAD(CAD - Computer-Aided Design) 유사성 검색을 혁신적으로 개선합니다. 특히, 기존의 고비용 레이블링 작업 없이 효율적으로 매개 변수가 있는 CAD 원시 파일을 모델링하는 데 중점을 둡니다.

- **Technical Details**: GC-CAD는 두 가지 핵심 모듈로 구성됩니다: 구조 인식 표현 학습과 대조 학습(contrastive learning) 기반의 그래프 신경망(framework). 이 방법은 CAD 모델의 기하학적(geometric) 및 위상(topological) 정보를 추출하여 피처(feature) 표현을 생성하는 데 그래프 신경망을 활용합니다. 이를 통해 복잡한 CAD 모델의 3D 모양을 효과적으로 캡처할 수 있습니다. 그 후, 간단하지만 효과적인 대조 학습 방법을 도입하여 모델을 수동 레이블 없이 훈련할 수 있어, 검색 준비가 된 표현을 생성합니다.

- **Performance Highlights**: GC-CAD는 네 개의 데이터셋에 대한 실험에서 기존의 베이스라인 방법에 비해 상당한 정확도 향상과 최대 100배의 효율성 향상을 보여주었습니다. 또한 인간 평가에서도 우수한 성능이 입증되었습니다.



### How Powerful is Graph Filtering for Recommendation (https://arxiv.org/abs/2406.08827)
Comments:
          Accepted to KDD'24

- **What's New**: 그래프 컨볼루션 네트워크(GCN)를 사용한 추천 시스템의 효율성은 주로 스펙트럼 그래프 필터링(spectral graph filtering)에 기인합니다. 이 연구는 GCN 기반의 기존 방법들이 가지는 두 가지 주요 한계를 지적하고 이를 해결하는 새로운 방법론을 제안합니다: (1) 일반성 부족과 (2) 표현력 부족. 제안된 해결책은 일반화된 그래프 정규화(G^2N)와 개별화된 그래프 필터(IGF)를 포함합니다.

- **Technical Details**: 첫 번째 한계를 해결하기 위해 데이터 노이즈 분포와 스펙트럼의 날카로움 간의 관계를 분석하여 더 날카로운 스펙트럼 분포가 훈련 없이도 노이즈를 분리할 수 있음을 보여줍니다. 이를 기반으로 G^2N(G^2N)을 제안해 스펙트럼의 날카로움을 조정하여 데이터 노이즈를 재분배하고 훈련 없이도 데이터셋 밀도에 관계없이 일반화할 수 있습니다. 두 번째 한계를 해결하기 위해 사용자 선호도의 서로 다른 신뢰 수준에 적응하는 개별화된 그래프 필터(IGF)를 제안하여 임의의 임베딩을 생성할 수 있습니다.

- **Performance Highlights**: 네 개의 다른 밀도 설정을 가진 데이터셋을 이용한 실험 결과, 제안된 방법들이 기존의 GCN 기반 추천 시스템보다 효율성과 효과 면에서 우수함을 보였습니다. 특히 G^2N과 IGF는 데이터 밀도와 관계없이 추천 성능을 향상시켰습니다.



### Reducing Task Discrepancy of Text Encoders for Zero-Shot Composed Image Retrieva (https://arxiv.org/abs/2406.09188)
Comments:
          17 pages

- **What's New**: 최근 발표된 논문에서는 새로운 텍스트 인코더 훈련 방식(RTD: Reducing Task Discrepancy)을 도입하여 압축된 이미지 검색(Composed Image Retrieval, CIR) 작업에서 텍스트 인코더의 성능을 크게 향상시켰습니다. RTD는 저렴한 텍스트 트리플릿을 활용하여 기존 CLIP 인코더의 작업 불일치를 줄이는 것이 특징입니다.

- **Technical Details**: 이 논문에서 제안하는 RTD는 새로운 목표 기반 텍스트 대조 학습(target-anchored text contrastive learning)을 이용하여 텍스트 인코더의 능력을 향상시킵니다. 또한, 하드 네거티브 샘플링 전략과 정교한 연결 방식을 추가하여 학습 효과를 극대화합니다. 이를 통해 기존의 프로젝션 기반 ZS-CIR 방법론과 쉽게 통합할 수 있으며, 텍스트 인코더를 업데이트하여 성능을 향상시킬 수 있습니다.

- **Performance Highlights**: RTD를 프로젝트 기반 ZS-CIR 방법론(SEARLE, Pic2Word, LinCIR)에 통합한 실험 결과, 다양한 데이터셋(CIRR, CIRCO, FashionIQ, GeneCIS)과 백본(ViT-B/32, ViT-L/14)에서 성능이 크게 개선되었습니다. 특히 RTD를 LinCIR/SEARLE(ViT-B/32)에 통합한 결과에서도 이점이 확인되었습니다.



### DIET: Customized Slimming for Incompatible Networks in Sequential Recommendation (https://arxiv.org/abs/2406.08804)
Comments:
          Accepted by KDD 2024

- **What's New**: DIET (customizeD slImming framework for incompatiblE neTworks)와 DIETING을 소개합니다. 이 두 가지 방법은 모바일 엣지에서의 추천 시스템을 위한 새로운 접근 방식으로, 모델의 비효율적인 부분을 줄이고 전송 및 저장 공간을 최적화하는 데 중점을 둡니다.

- **Technical Details**: DIET은 사용자 상호작용 데이터를 바탕으로 맞춤형 서브넷 '다이어트'를 생성하며, 이는 필터 레벨 및 요소 레벨에서 중요한 데이터를 학습하여 전송 효율성을 극대화합니다. DIETING은 네트워크 전체를 단일 파라미터 레이어로 대표하여 저장 효율성을 높입니다. 이렇게 생성된 서브넷은 이진 마스크(binary mask)를 활용해 네트워크 사이의 비호환성을 줄이고, 적은 자원으로도 높은 성능을 발휘할 수 있게 합니다.

- **Performance Highlights**: 실험 결과, DIET와 DIETING는 네 가지 최신 데이터셋과 두 가지 널리 사용되는 모델에서 추천 성능과 전송 및 저장 효율성에서 탁월한 성과를 보였습니다. 특히, 엣지 자원의 제약 조건 하에서도 높은 정확도를 유지하며 빠른 추론 시간을 제공했습니다.




New uploads on arXiv(cs.CL)

### LLM Post-Training: A Deep Dive into Reasoning Large Language Models (https://arxiv.org/abs/2502.21321)
Comments:
          31 pages, 7 figures, 3 tables, 375 references

- **What's New**: 최근 논문에서는 LLMs(대규모 언어 모델)가 자연어 처리 분야에서 많은 변화를 가져왔음을 강조하고 있습니다. 이 연구는 사전 훈련(pretraining)뿐만 아니라 후속 훈련(post-training) 기법에 대한 집중적인 탐구를 통해 LLM의 성능을 극대화하려는 노력에 주목하고 있습니다. 후속 훈련 기법들은 LLMs의 지식을 세분화하고, 추론 능력을 향상시키며, 사용자 의도 및 윤리적 고려 사항과 더 잘 조화를 이루도록 돕습니다.

- **Technical Details**: 이 논문은 LLMs의 훈련 단계를 두 가지로 나누어 설명합니다. 사전 훈련은 대규모 데이터를 기반으로 다음 토큰 예측(next-token prediction) 타겟을 사용하고, 후속 훈련은 여러 번의 세밀한 조정(fine-tuning)과 정렬(alignment)을 포함합니다. 후속 훈련 기법은 LLM의 행동을 개선하고, 인간의 의도에 맞게 일관성을 갖도록 조정하는 데 목표를 두고 있습니다. 이 과정에서 LoRA(저순위 적응)와 리인포스먼트 러닝(Reinforcement Learning) 같은 접근 방법이 도입됩니다.

- **Performance Highlights**: 후속 훈련을 통해 LLMs는 사용자 의도와 윤리적 요구 사항에 더 잘 맞춰질 수 있습니다. 미세 조정(fine-tuning)과 같은 방법은 특정 작업에서의 성능을 크게 향상시키지만, 과적합(overfitting)이나 높은 계산 비용 같은 문제를 초래할 수 있습니다. 또한, 리인포스먼트 러닝을 통해 LLMs는 다이나믹한 피드백을 활용하여 더 나은 적응력을 발휘할 수 있으며, 테스트 시 조정(test-time scaling) 기법이 적용되어 성능을 최적화할 수 있습니다.



### Identifying Emerging Concepts in Large Corpora (https://arxiv.org/abs/2502.21315)
Comments:
          9 pages, 4 figures

- **What's New**: 본 논문에서는 대규모 텍스트 코퍼스에서 새로운 개념을 식별하는 새로운 방법론을 제안합니다. 이 방법은 임베딩 공간의 히트맵(heatmap) 변화를 분석함으로써 개념을 높은 정확도로 조기 탐지할 수 있습니다. 또한, 1941년부터 2015년까지의 미국 상원 연설 분석을 통해 새로운 개념이 소수당에 의해 더욱 자주 도입된다는 사실을 밝혔습니다.

- **Technical Details**: 우리의 방법론은 다음과 같은 주요 단계를 포함합니다: (1) 문장을 임베딩(embedding)하여 그 의미적 특성을 포착하고, (2) 이러한 임베딩의 차원(Dimensionality)을 축소하여 히트맵을 생성하며, (3) 히트맵 간의 차이를 감지하여 중요한 변화를 검출합니다. 이 과정에서 우리는 기존의 방법과 달리 특정 단어 수준에서 벗어나 개념의 보다 복잡한 구조를 포착할 수 있도록 설계되었습니다.

- **Performance Highlights**: 우리의 방법은 특히 개념의 출현 직후에 정확한 탐지가 가능하다는 점에서 뛰어난 성과를 보입니다. 여러 평가를 통해, 제안된 방법이 기존의 개념 탐지 방법들과 비교했을 때 우수한 성능을 발휘함을 증명하였습니다. 사회과학 및 디지털 인문학 분야에서도 우리의 방법이 새로운 개념의 도입을 탐구하는 데 유용하다는 것을 확인했습니다.



### FANformer: Improving Large Language Models Through Effective Periodicity Modeling (https://arxiv.org/abs/2502.21309)
- **What's New**: 이 연구는 최근의 대형 언어 모델(LLMs)들이 데이터와 계산 자원에 대한 높은 요구로 인해 학습 효율성이 낮다는 점을 지적합니다. 새로운 모델 FANformer를 제안하여 Fourier Analysis Network (FAN)를 Transformer의 attention 메커니즘에 통합하여 주기성(periodicity) 모델링을 보다 효율적으로 수행하고 있습니다. FANformer는 Transformer 아키텍처의 학습 효율성과 성능을 개선하기 위해 주기적 패턴을 포착하고 표현하는 Fourier 원리를 활용합니다. 이 연구 결과는 FANformer가 LLM을 발전시키기 위한 유망한 아키텍처가 될 수 있음을 보여줍니다.

- **Technical Details**: FANformer는 샘플 입력 𝐗𝐗oldsymbol{X}의 폴리폼 및 주기 기저 함수(frequency basis functions)를 사용하는 FAN 레이어를 기반으로 구성됩니다. FAN 레이어는 행동 기능을 유지하면서 주기적 패턴을 explicit하게 인코딩하여 MLP 레이어와 차별화됩니다. 또한 attention 메커니즘의 feature projection 과정을 수정하여 frequency domain representations을 통합하고 있습니다. 이러한 구조는 주기성을 효과적으로 캡처하고 모델링할 수 있도록 설계되었습니다.

- **Performance Highlights**: FANformer는 1조 개의 토큰을 활용하여 11억 파라미터의 FANformer-1B를 사전 훈련하였으며, 결과적으로 오픈 소스 LLM보다 뛰어난 성능을 보였습니다. 모델 파라미터와 훈련 토큰이 같은 조건에서 FANformer는 기존의 Transformers보다 적은 자원으로 유사한 성능을 발휘하면서 뛰어난 학습 효율성을 가지고 있음을 확인했습니다. FANformer의 학습 효율성이 더 크고 복잡한 모델에서 더욱 돋보이며, 이는 LLM의 발전 가능성을 시사합니다.



### Persuasion Should be Double-Blind: A Multi-Domain Dialogue Dataset With Faithfulness Based on Causal Theory of Mind (https://arxiv.org/abs/2502.21297)
Comments:
          23pages

- **What's New**: 이 논문에서는 인지 이론(Theory of Mind) 기반의 보일라 지식(DoMMA)이라는 새로운 대화 생성 프레임워크를 소개합니다. 이 프레임워크는 대화 참여자 간에 정보가 공유되지 않도록 하여 '더블 블라인드(double blind)' 조건을 유지합니다. 또한, 'CToMPersu'라는 대규모의 다분야, 다회차 설득 대화 데이터셋을 제시하여 현실적인 인간 대화와의 더 나은 정합성을 보여줍니다.

- **Technical Details**: ToMMA는 대화 생성을 위해 여러 에이전트를 운영하는 다중 에이전트 프레임워크로 설계되었습니다. 이 프레임워크는 persuader(설득자)가 persuadee(설득 대상)의 심리 상태를 추정하여 주장을 구성할 수 있도록 유도합니다. 여기서 Causal Theory of Mind 평가 방법을 통해 데이터셋의 신뢰성을 검증하며, 이는 LLM이 대화에서 persuadee의 신념과 욕망을 추론한 후 이를 효율적으로 다루었는지를 평가합니다.

- **Performance Highlights**: 제안된 CToMPersu 데이터셋은 6,275개의 대화로 구성되어 있으며, 이는 35개 분야 및 6,257개의 고유 시나리오를 포함합니다. 이 데이터셋은 다중 평가 지표에서 높은 성능을 보이며, 현실적인 인간 대화와의 정합성을 강화합니다. 또한, 기존 LLM 생성 데이터셋에 비해 더 나은 설득 대화 품질을 입증합니다.



### Token-level Ensembling of Models with Different Vocabularies (https://arxiv.org/abs/2502.21265)
Comments:
          Under review

- **What's New**: 이 논문에서는 서로 다른 vocabulary를 가진 모델을 조합할 수 있는 인퍼런스 타임( inference-time) 알고리즘인 Agreement-Based Ensembling (ABE)을 제안합니다. 기존의 모델 앙상블(ensemble) 방식은 shared vocabulary를 전제로 하여 가중 합을 통해 새로운 분포를 생성해야 하며, 이는 제한적인 측면이 있었습니다. ABE는 추가 파라미터를 학습하거나 모델을 수정할 필요 없이, 생성된 토큰들이 표면 형식(surface form)에서 일치하는지 조정하여 동작합니다. 결과적으로, 본 기법은 기계 번역(machine translation) 성능을 개선할 뿐만 아니라, 기존의 트랜스포머(Transformer) 기반 모델과 단일 디코더 LLM의 조합에도 적용됩니다.

- **Technical Details**: 주요 기술적 측면으로는 ABE가 인퍼런스 타임에 서로 다른 모델의 vocabulary에 대해 앙상블을 수행한다는 점입니다. 각 모델은 고유의 vocabulary에 대해 분포를 생성하며, 이를 통해 다음 토큰을 선택하기 위해 모델 간의 일치를 유지합니다. 이 과정에서 각 모델은 자신의 tokenization에 따라 부분 문자열(local hypothesis)을 유지하고, 이를 기준으로 공유 string인 global hypothesis와 비교합니다. 또한, ABE는 beam search와 같은 다양한 인퍼런스 알고리즘에 쉽게 확장될 수 있습니다.

- **Performance Highlights**: ABE는 다양한 아키텍처의 모델들 간의 앙상블을 가능하게 하여, 모델 각각의 경우보다 기계 번역 성능을 향상시키는 사례를 보여줍니다. 특히, 문맥적으로 호환 가능한 토큰을 효율적으로 검색하는 과정에서, 토큰의 길이에 대한 제한을 고려하여 미래 검색을 제약하는 것이 특징입니다. 이러한 결과는 언어 모델이 가진 여러 가지 차이를 극복하면서도, 향상된 성능을 제공할 수 있음을 입증합니다.



### RuCCoD: Towards Automated ICD Coding in Russian (https://arxiv.org/abs/2502.21263)
- **What's New**: 이 연구는 러시아어로 된 임상 코딩 자동화의 가능성을 검토하고 있습니다. 10,000개 이상의 엔티티와 1,500개 이상의 고유한 ICD 코드가 주석 처리된 새로운 ICD 코딩 데이터셋을 제시합니다. 이 데이터셋은 여러 최신 모델에 대한 벤치마크로 사용되며, 전이 학습(transfer learning)을 검토하여 PubMed 초록에서 의학적 진단으로의 변화를 포함합니다.

- **Technical Details**: 연구에서 제시한 RuCCoD(Russian ICD Coding Dataset)는 의료 전문가에 의해 ICD-10 CM 시스템에 기반하여 레이블링된 데이터셋입니다. 우리가 개발한 모델은 BERT을 기반으로 한 정보 추출 파이프라인과 LoRA가 적용된 LLaMA 모델, 그리고 RAG를 포함합니다. 실험은 2017년부터 2021년까지의 환자 기록이 포함된 대규모 EHR 데이터셋에 대해 수행되었습니다.

- **Performance Highlights**: 자동으로 예측된 코드를 사용한 훈련이 의사에 의해 주석 처리된 데이터에 비해 정확도를 크게 향상시킨다는 것을 보여줍니다. 특정 실험에서는 자동으로 할당된 ICD 코드로 사전 훈련을 통해 진단 예측에서 macro-averaged F1-score가 28% 증가했습니다. 경량 콘텐츠가 부족한 러시아어와 같은 언어에서 임상 코딩의 자동화를 위한 기반을 제공할 것으로 기대합니다.



### Semantic Volume: Quantifying and Detecting both External and Internal Uncertainty in LLMs (https://arxiv.org/abs/2502.21239)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)에서 외부 불확실성(external uncertainty)과 내부 불확실성(internal uncertainty)을 동시에 정량화할 수 있는 새로운 수학적 측정법, Semantic Volume을 제안합니다. 기존의 방법은 주로 내부 불확실성에 초점을 맞춰왔으나, 본 연구는 모호한 사용자 쿼리로 인한 외부 불확실성의 감지도 충분히 다룰 수 있음을 보여줍니다. 이 접근법은 LLM의 내부 상태에 대한 액세스 없이도 적용할 수 있어, 블랙박스 환경에서도 유용하게 활용될 수 있습니다.

- **Technical Details**: Semantic Volume 방법은 쿼리와 응답을 섭동시켜 각 쿼리의 여러 증강 버전을 생성하고, 이를 의미 공간에 임베딩(embedding)하여 그람 행렬의 행렬식(determinant)을 계산합니다. 이 방식을 통해 의미적 분산(semantic dispersion)을 정량화하고, 이를 낮은 불확실성에 기반해 측정할 수 있습니다. 본 연구에서는 쿼리 및 응답 불확실성 감지에 대한 포괄적인 실험을 수행하고, Semantic Volume이 기존의 기준선들보다 일관되게 우수한 성능을 보임을 입증하였습니다.

- **Performance Highlights**: Semantic Volume 접근 방식은 외부와 내부 불확실성 모두에 효과적으로 적용될 수 있으며, 이전의 샘플링 기반의 불확실성 측정 방법인 Semantic Entropy의 일반화된 형태로 볼 수 있습니다. 이 연구는 LLM의 신뢰성을 향상시키기 위한 체계적인 불확실성 감지 방안을 제공하며, 이로 인해 LLM의 사용성과 신뢰성을 개선할 수 있습니다. 또한, Semantic Volume의 이론적 배경은 미분 엔트로피(differential entropy)와 연결되어 있어 기존의 샘플링 기반 방법들과의 관련성을 이해하는 데 도움을 줍니다.



### ECLeKTic: a Novel Challenge Set for Evaluation of Cross-Lingual Knowledge Transfer (https://arxiv.org/abs/2502.21228)
- **What's New**: 이번 논문에서는 다국어 대형 언어 모델(LLMs)의 언어 간 지식 전이 능력을 평가하기 위한 새로운 데이터셋, ECLeKTic을 제안합니다. ECLeKTic은 12개 언어에서 비대칭적으로 커버된 정보를 기반으로 만들어졌으며, 모델이 언어를 초월해 지식을 전이할 수 있는 능력을 검증하기 위한 간단한 방법론을 사용합니다. 이를 통해 기존 문헌에서 부족한 경험적 지식 전이 평가 문제를 해결하고자 합니다.

- **Technical Details**: ECLeKTic은 다국어 폐쇄형 질문-답변(CBQA) 데이터셋으로, 각 언어에서 특별히 참고할 수 있는 위키피디아 기사를 중심으로 한 질문-답변 쌍을 생성합니다. 12개 언어 중 하나의 언어에서 작성된 기사를 기반으로 질문을 생성하고, 이 질문을 해당 기사가 없는 다른 11개 언어로 번역했습니다. 이를 통해 모델이 언어 간 지식을 전이할 수 있는 기관적 능력을 평가할 수 있음을 보입니다.

- **Performance Highlights**: 실험 결과 8개의 상위 모델로 ECLeKTic를 테스트한 결과, 최고의 성능을 기록한 모델은 41.3%의 전체 성공률과 65%의 지식 전이 능력을 보였습니다. 이 결과는 대다수의 최첨단 모델이 언어 간 지식 공유에서 어려움을 겪고 있음을 나타내며, 더 나은 다국어 모델 개발을 위한 임계점을 제시합니다. 언어 간 스크립트 공유가 전이에 중요한 요인으로 작용함을 보여주는 것이 인상적입니다.



### Detecting Linguistic Diversity on Social Media (https://arxiv.org/abs/2502.21224)
Comments:
          Accepted to Cartography and GIScience in Australasia and Oceania: Including twenty years of GeoCart

- **What's New**: 이 논문은 소셜 미디어 데이터를 활용하여 특정 지역의 언어 사용 변화를 분석하는 방법의 유효성을 탐구합니다. Aotearoa New Zealand(뉴질랜드)의 인구 조사 데이터와 소셜 미디어 데이터를 비교하여, 후자의 데이터가 공간적 및 시간적 통찰을 제공할 수 있음을 보여줍니다. 이는 정책 입안자와 연구자들이 전통적으로 의존해온 인구 조사 통계 외에 새로운 대안 자료의 필요성을 강조합니다.

- **Technical Details**: 본 연구에서는 뉴질랜드 인구 조사 데이터를 기초 사실으로 삼고, 소셜 미디어 언어 데이터를 대안 자료로 활용합니다. 이를 통해 두 데이터 소스 간 언어 사용 변화를 비교하고 언어 다양성을 국가, 지역 및 로컬 지리 수준에서 분석합니다. 또한, 소셜 미디어 데이터셋에서 각 트윗의 언어 조건을 식별하기 위해 두 개의 언어 식별 모델을 사용하여 결과를 검증합니다.

- **Performance Highlights**: 결과에 따르면, 소셜 미디어 데이터는 낮은 지역 및 로컬 지리에서의 인구통계 및 사회정치적 변화에 민감하고, 언어 사용의 다양성을 풍부하게 기록할 수 있는 가능성을 가지고 있습니다. 특히, 소셜 미디어 언어 데이터는 공식 통계 대신 사용할 수 있는 유망한 대안으로 제시되며, 뉴질랜드의 변화하는 언어 프로필을 이해하는 데 중요한 자원이 될 수 있습니다.



### Generating patient cohorts from electronic health records using two-step retrieval-augmented text-to-SQL generation (https://arxiv.org/abs/2502.21107)
Comments:
          7 pages, 1 figure

- **What's New**: 이번 연구에서는 대규모 언어 모델을 활용한 자동화 시스템을 제안하여 전자 건강 기록(EHR) 데이터를 사용하여 환자 집단(cohort)을 생성하는 방법을 소개합니다. 이 시스템은 기준 파싱(criteria parsing), 이중 수준의 Retrieval Augmented Generation (RAG), 의학 개념 표준화 및 SQL 생성 등을 통합하여 복잡한 관계를 효과적으로 포착합니다. 최종적으로, 시스템은 EHR 데이터에서 환자 집단을 식별하는 데 0.75 F1-score를 달성하며, 이는 역학 연구에 대한 자동화된 집단 생성의 가능성을 보여줍니다.

- **Technical Details**: 우리는 EHR에서 환자 집단 정의를 SQL 쿼리로 변환하는 다단계 파이프라인을 개발했습니다. 이 과정에서, 대규모 언어 모델(LLM)을 사용하여 기준을 반구조적 형식으로 변환하고, OMOP-CDM 및 SQL 쿼리 구조에 대한 도메인 지식을 포함한 지침 프롬프트를 통합합니다. 세 가지 RAG 전략(기준 수준 RAG, 집단 수준 RAG, 결합 RAG)을 구현하여 쿼리 패턴 재사용을 가능하게 하고, 최종적으로 환자 ID와 인덱스 날짜를 포함한 환자 집단 테이블을 생성합니다.

- **Performance Highlights**: 여러 최신 LLM 모델을 평가한 결과, RAG가 적용된 모델들이 SQL 컴파일 및 데이터 검색 성공률을 높이는 것으로 나타났습니다. 특히, RAG+A+C 결합 전략은 모든 모델에서 ZS보다 상당히 우수한 성과를 보여 구성 요소 수준 예제가 F1 점수를 높이는 데 효과적입니다. 연구에서 Claude와 Gemini가 최고의 성과를 보였으며, 환자 집단 생성에 대한 정확성을 완벽히 평가한 결과, 가장 성능이 높은 모델에서 단일 쿼리 접근법과 비교하여 90%의 정규화된 집단 크기 유사성을 기록했습니다.



### PASemiQA: Plan-Assisted Agent for Question Answering on Semi-Structured Data with Text and Relational Information (https://arxiv.org/abs/2502.21087)
- **What's New**: 본 논문에서는 PASemiQA라는 새로운 방법론을 제안합니다. PASemiQA는 반구조화 데이터(semistructured data)에서 텍스트와 관계 정보를 결합하여 질문에 답하도록 설계되었습니다. 기존의 RAG(리트리벌 증강 생성) 방법들이 단일 유형의 외부 데이터에 초점을 맞춘 반면, PASemiQA는 다양한 데이터 유형을 효과적으로 활용할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 이 방법은 먼저 질문에 대한 관련 텍스트 및 관계 정보를 식별하기 위한 계획 모듈(plan module)을 생성하고, 이후에 LLM(대형 언어 모델)을 기반으로 한 에이전트를 사용해 반구조화 데이터를 탐색합니다. 이 과정에서 필요한 정보를 추출하여 질문에 대한 정확한 답변을 생성하는데 초점을 맞춥니다. 또한 이 접근 방식은 기존 RAG나 KGQA(지식 그래프 질문 응답) 방법과는 다르게, 질문에 의해 결정된 복합 관계와 텍스트 정보를 동시에 처리합니다.

- **Performance Highlights**: PASemiQA는 다양한 도메인의 여러 반구조화 데이터셋을 통해 그 효과성을 입증하였습니다. 실험 결과, 이 방법은 질문 응답 시스템의 정확성과 신뢰성을 향상시키는데 기여할 수 있음을 보여주었습니다. 나아가 PASemiQA는 향후 복잡한 반구조화 질문에 대한 응답 생성의 새로운 가능성을 모색할 수 있는 기반이 될 것입니다.



### CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation (https://arxiv.org/abs/2502.21074)
Comments:
          15 pages

- **What's New**: CODI (Continuous Chain-of-Thought via Self-Distillation) introduces a novel approach that distills explicit Chain-of-Thought (CoT) into an implicit form, marking a significant advancement in Large Language Models (LLMs). 이 프레임워크는 단일 모델이 교사와 학생 역할을 동시에 수행하며, 효율적으로 CoT를 학습할 수 있도록 합니다. CODI는 기존의 CoT 방법에 비해 3.1배의 압축률로 더 나은 성능을 발휘하고, 해석 가능성을 유지합니다.

- **Technical Details**: CODI는 두 가지 주요 설계 고려 사항, 즉 forward function과 training objective를 필요로 합니다. 기존 CoT가 언어 모델링을 통해 명시적인 토큰을 학습하는 반면, CODI는 자기 증류(self-distillation) 방식으로 학습 진행 중 발생할 수 있는 잊음 문제를 완화합니다. 이 과정에서 CoT 정보가 포함된 최종 답변을 생성하는 토큰의 숨겨진 활성화를 정렬하여 효과적인 학습을 촉진합니다.

- **Performance Highlights**: CODI는 GSM8k 데이터셋에서 명시적인 CoT의 성능을 뛰어 넘으며, 이전의 state-of-the-art보다 28.2% 높은 정확도를 기록했습니다. 또한 CODI는 더 복잡한 CoT 데이터셋에 대해 확장 가능하고 일반화 가능하며, 그 과정이 투명하여 해석 가능성을 유지합니다. 이는 암시적인 CoT가 단순히 효율적일 뿐 아니라, 강력한 대안임을 증명합니다.



### Beyond Words: A Latent Memory Approach to Internal Reasoning in LLMs (https://arxiv.org/abs/2502.21030)
Comments:
          13 pages, 5 figures

- **What's New**: 최근 대형 언어 모델(LLMs)의 발전으로 인해 체인 오브 생각(Chain-of-Thought, CoT) 패러다임이 주목받고 있습니다. 이 접근법은 모델이 자연어로 명시적인 추론 단계를 생성하여 해석 가능성을 높이고 외부 감사를 용이하게 합니다. 그러나 저자는 인간의 인지가 완전한 언어화 없이도 과거의 감각 및 에피소드 정보를 회상하는 암묵적 정신적 표현에 의존한다는 점을 지적하며, 이를 LLM의 내부 추론 과정에 통합하는 프레임워크를 제안합니다.

- **Technical Details**: 제안된 구조는 암묵적 메모리 모듈(Implicit Memory Module, IMM)을 갖춘 LLM을 포함합니다. IMM은 처리 중 생성된 암묵적 표현을 저장하고 필요할 때 이를 검색하여 내부 추론을 지원합니다. 이 모델은 동적으로 정보를 처리하고, 특정 시간 단계에서 요약 표현을 메모리에 기록하며, 쿼리 벡터를 생성해 기억 저장소에서 정보를 검색합니다.

- **Performance Highlights**: 초기 실험 결과, 단순한 GPT 모델에 IMM을 통합함으로써 최종 훈련 손실이 기존 GPT 기준선보다 35%에서 57% 감소하는 효과가 있었습니다. 이러한 방법은 정밀한 해석 가능성 채널도 쉽게 추가할 수 있어, 효율적이면서도 더 강력한 추론을 가능하게 합니다. 따라서 이 연구는 미래의 명시적 감사 가능성을 위한 방향도 제시합니다.



### PersuasiveToM: A Benchmark for Evaluating Machine Theory of Mind in Persuasive Dialogues (https://arxiv.org/abs/2502.21017)
- **What's New**: 이 논문은 PersuasiveToM이라는 새로운 벤치마크를 제안하고 있습니다. 이 벤치마크는 Large Language Models (LLMs)의 Theory of Mind (ToM) 능력을 평가하며, 실제 사회적 상황에서의 설득 대화에 초점을 맞추고 있습니다. 기존의 ToM 평가 방식이 물리적 인식에 집중했던 것과 달리, PersuasiveToM은 심리적 상태의 복잡성을 다루면서 LLM의 사고 능력을 더 깊이 평가할 수 있도록 설계되었습니다.

- **Technical Details**: PersuasiveToM은 두 가지 질문 카테고리를 도입하여 LLM을 평가합니다: (1) ToM Reasoning, 즉 페르소나의 정신 상태 변화(예: 설득 대상의 욕구 변화)를 추적하는 능력을 평가하고, (2) ToM Application, LLM이 추론한 정신 상태를 활용하여 효과적인 설득 전략을 선택하고 이를 평가하는 능력을 측정합니다. 이 벤치마크는 Belief-Desire-Intention (BDI) 모델에서 영감을 받아 심리적 상태에 대한 더 복잡한 추론을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, LLM은 정적인 측면(예: 설득자의 욕구)에서는 인간과 경쟁적으로 성과를 보이는 반면, 역동적인 변화(예: 설득자의 욕구 변동)를 추론하는 질문에서는 현저히 낮은 성적을 기록했습니다. 또한, Chain-of-Thought (CoT) 프롬프팅은 LLM의 Mental State Reasoning에는 큰 기여를 하지 않았지만, 설득 전략 예측 성능을 향상시킵니다. LLM은 대화 전체에서 정신 상태의 역동성을 이해하는 데 어려움을 겪으며, 이 부분에서 인간보다 현저히 낮은 성과를 보였습니다.



### Capability Localization: Capabilities Can be Localized rather than Individual Knowledg (https://arxiv.org/abs/2502.20992)
- **What's New**: 이 논문은 대규모 언어 모델의 파라미터가 성능 개선에 미치는 영향을 명확히 규명하고자 합니다. 연구자들은 개별 지식이 어떻게 저장되는지에 대한 기존의 가정을 검토하고, 이를 통해 'Commonality Neuron Localization (CNL)'이라는 새로운 방법을 제안합니다. 이 방법은 GSM8K 데이터셋에서 96.42%의 신경망 중복율을 달성하며, 성능 향상에 기여할 수 있는 지식의 위치를 효과적으로 찾아냅니다.

- **Technical Details**: 본 연구에서는 개별 지식의 저장 형태에 대한 신뢰성 평가를 수행하였으며, 기존 지식 위치화 방법이 개별 지식에 대해 신뢰할 수 없다는 결론을 도출했습니다. 실험을 통해 KN, ROME, KC와 같은 다양한 위치화 방법의 제약을 분석하여, 이들이 파라미터의 정확한 위치를 찾지 못함을 보여줍니다. 또한, decoupling 실험을 통해 데이터의 공통성을 분석하고, 지식 저장 형태의 제한을 발견했습니다.

- **Performance Highlights**: CNL 방법을 통해 발견된 공통성 신경망은 성능 향상에 기여하는 능력 신경망의 집합체로 확인됩니다. 데이터셋 간의 교차 실험을 통해, 공통성 신경망이 특정한 능력을 지닌다는 것을 입증하였습니다. 이는 개별 지식의 형태가 아니라, 공통 데이터 간의 상관 관계가 파라미터와 연결될 수 있음을 시사합니다.



### UoR-NCL at SemEval-2025 Task 1: Using Generative LLMs and CLIP Models for Multilingual Multimodal Idiomaticity Representation (https://arxiv.org/abs/2502.20984)
- **What's New**: SemEval-2025 Task 1에서는 주어진 명사 합성이 가지는 이디엄적 의미에 따라 이미지를 순위화하는 것을 목표로 합니다. 이를 위해, generative large language models (LLMs) 및 다국어 CLIP 모델을 활용하여 이디엄적 합성어 표현을 개선하는 새로운 접근 방식을 제시합니다.

- **Technical Details**: 이 연구는 LLMs를 사용해 잠재적인 이디엄적 합성어에 대한 의미를 생성하여 의미 해석을 풍부하게 합니다. 생성된 의미는 multilingual CLIP 모델을 통해 인코딩되어 이미지 순위화에 사용됩니다. 이 과정에서 contrastive learning 및 data augmentation 기술이 적용되어 임베딩을 미세 조정합니다.

- **Performance Highlights**: 실험 결과, 이 방법을 통해 추출된 다중모달 표현이 원래의 명사 합성어를 기반으로 한 표현보다 더 우수한 성능을 보였습니다. 그러나 미세 조정이 없는 임베딩을 사용하는 것보다 미세 조정 접근법의 효과는 다소 떨어지는 것으로 나타났습니다.



### Set-Theoretic Compositionality of Sentence Embeddings (https://arxiv.org/abs/2502.20975)
- **What's New**: 본 논문은 NLP 작업에서의 문장 인코더의 구성 속성을 평가하는 새로운 접근법을 제시합니다. 기존 평가 방법이 특정 작업 성능에 주로 집중된 반면, 본 연구는 'set-like' 조합의 개념을 도입하여 문장 임베딩의 기본적인 조합 속성을 평가합니다. 이를 통해 문장 인코더의 성능을 보다 풍부하게 이해할 수 있는 기초 자료를 제공합니다.

- **Technical Details**: 저자들은 문장 레벨 조합을 위해 세 가지 'set-like' 연산인 TextOverlap, TextDifference, TextUnion을 제안하고, 16개의 문장 인코더를 체계적으로 분석합니다. 또한 약 192K의 샘플 데이터셋을 생성하여 작업 독립적인 문장 임베딩의 평가를 용이하게 합니다. 이러한 방법론은 문장 인코더의 조합 속성을 평가하는 새로운 프레임워크를 제공합니다.

- **Performance Highlights**: 연구 결과, SBERT가 가장 일관되게 set-like 조합 속성을 보여주며 최신 LLM보다도 뛰어남을 확인했습니다. 이러한 발견은 문장 인코딩 모델 간의 비교 연구에서 중요한 통찰력을 제공하며, 향후 문장 임베딩의 조합성 연구를 촉진할 것으로 기대됩니다.



### Arabizi vs LLMs: Can the Genie Understand the Language of Aladdin? (https://arxiv.org/abs/2502.20973)
Comments:
          Submitted to MT Summit 2025

- **What's New**: 이번 연구는 새롭고 비공식적인 언어 소통 방식인 Arabizi의 기계 번역 가능성을 평가합니다. 약 420백만 명의 아랍어 사용자가 있지만, Modern Standard Arabic (MSA)는 사용자의 모국어가 아닙니다. Arabizi는 라틴 문자를 사용하여 아랍어 방언을 표현하는 혼합 형태로, 비공식적인 환경에서 주로 사용됩니다. 이 연구는 다양한 아랍어 방언에 대한 번역의 가능성을 평가할 수 있는 대규모 언어 모델(LLMs)의 성능을 분석합니다.

- **Technical Details**: 연구는 레바논, 이집트, 알제리의 세 가지 아랍어 방언을 MSA 및 영어로 번역하는 데 초점을 맞추고 있습니다. 각 방언은 각각의 독특한 문화적, 언어적 특징을 가지고 있으며, 이들 사이의 차이를 분석하고자 합니다. 연구팀은 31명의 아랍어 원어민 참가자들을 통해 자연스러운 대화 데이터를 수집하고, 이를 기반으로 전문 번역가가 MSA 및 영어로 번역하여 참조 텍스트를 생성했습니다. 실험에서는 자동화된 평가 프로토콜을 활용하여 번역 품질을 체계적으로 평가합니다.

- **Performance Highlights**: 연구 결과는 다양한 아랍어 방언이 MSA 및 영어로 변환될 때 어떻게 다른 성과를 보이는지를 상세히 보여줍니다. 세 가지 방언 각각에 대한 모델의 번역 품질이 비교되며, 성과는 ARabizi의 비공식성과 비표준화된 특징에도 불구하고 기존의 표준화된 아랍어 번역 방식보다 더 나은 번역 결과를 나타낼 수 있다는 가능성을 제시합니다. 본 연구의 발견들은 Arabizi의 자연어 처리(NLP) 연구에 중요한 기여를 할 것으로 기대됩니다.



### Beware of Your Po! Measuring and Mitigating AI Safety Risks in Role-Play Fine-Tuning of LLMs (https://arxiv.org/abs/2502.20968)
Comments:
          25 pages, 10 figures, 13 tables

- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)이 역할 놀이(role-playing)를 통해 사용자와 몰입감 있는 상호작용을 할 수 있지만, 동시에 안전성에 대한 위험이 커질 수 있음을 강조합니다. 특히, 기존의 역할 놀이 미세 조정(role-play fine-tuning) 기법이 악당 캐릭터와 같은 역할에서 안전성을 저하시킬 수 있다는 점을 지적합니다. 이를 해결하기 위해 제안된 안전 인지 역할 놀이 미세 조정(SaRFT) 방법은 역할 적응성과 안전성을 조화롭게 유지할 수 있도록 설계되었습니다.

- **Technical Details**: SaRFT 방법은 두 가지 주요 단계로 구성됩니다: 역할-안전 적응 데이터 선택(Role-Safety Adaptive Data Selection, RDS)과 역할-안전 균형 최적화(Role-Safety Balance Optimization, RBO)입니다. RDS에서는 역할별 특성에 맞춰 '위험' 하위 집합을 선택하기 위한 임시 보상 함수를 사용하고, RBO에서는 교차 엔트로피 손실과 KL-발산 손실을 통해 안전성과 역할 놀이 성능의 균형을 맞춥니다. 이를 통해 SaRFT는 역할 놀이 LLM의 안전성을 개선하는 새로운 접근 방식을 제시합니다.

- **Performance Highlights**: 실험 결과, SaRFT는 LLaMA-3-8B-Instruct, Gemma-2-9B-it, Qwen2.5-7B-Instruct 모델에서 최신의 기준 방법들에 비해 일관되게 우수한 성능을 발휘하였습니다. 특히, 이러한 성과는 LoRA와 전체 파라미터 미세 조정 설정 하에서도 나타났으며, SaRFT가 역할-안전 충돌을 완화하는 데 효과적임을 강조합니다. 연구의 주요 기여는 현재의 역할 놀이 LLM과 관련한 안전 위험을 정량화하고 완화하는 첫 걸음을 내디딘 것입니다.



### WebFAQ: A Multilingual Collection of Natural Q&A Datasets for Dense Retrieva (https://arxiv.org/abs/2502.20936)
Comments:
          10 pages, 3 figures, 7 tables

- **What's New**: WebFAQ는 FAQ 스타일의 주석에서 파생된 대규모 오픈 도메인 질문 답변(QA) 데이터셋 컬렉션입니다. 총 96백만 개의 자연 질문-답변 쌍이 75개 언어로 구성되어 있으며, 이 중 47백만 개(49%)가 비영어 샘플입니다. WebFAQ는 20개의 단일 언어 검색 벤치마크의 기초로도 활용되며, 이렇게 다각화된 데이터는 다국어 밀집 검색 모델의 훈련과 평가를 위한 고품질 자원을 제공합니다.

- **Technical Details**: WebFAQ 데이터셋은 정제된 필터링 및 근사용 데이터 감지를 통해 신중하게 선별되어, 1120만 개 QA 쌍으로 구성된 20개의 단일 언어 검색 벤치마크를 포함합니다. 자동화된 데이터셋 생성에 대한 첨단 방법을 활용하여, QA 정렬된 이중 언어 말뭉치도 구축되었습니다. 또한 수집된 QA 쌍을 활용하여 in-domain 사전 훈련된 XLM-RoBERTa 모델을 미세 조정했으며, 이 모델은 언어 간 검색 데이터셋에도 일반화되는 성능 향상을 보여줍니다.

- **Performance Highlights**: WebFAQ 데이터는 밀집 검색 모델의 성능을 향상시키며, 이는 오픈 도메인 Q&A 검색에서 실질적인 개선을 나타냅니다. 특히, 미세 조정된 모델은 비영어 샘플을 포함한 다양한 다국어 검색 데이터셋에서도 주목할 만한 성능을 발휘합니다. 최종적으로, 자동화된 비텍스트 생성 방법을 통해 구축된 이중 언어 말뭉치는 비슷한 데이터셋에 비해 높은 번역 품질을 보인 것으로 확인되었습니다.



### Automated Evaluation of Meter and Rhyme in Russian Generative and Human-Authored Poetry (https://arxiv.org/abs/2502.20931)
Comments:
          7 pages, 1 figure

- **What's New**: 본 논문에서는 러시아어로 된 운율 시의 스트레스(mark) 지정, 운율 탐지 및 시적 결함 식별을 위한 러시아 시 스캔 도구(Russian Poetry Scansion Tool, RPST) 라이브러리를 소개합니다. 또한 다양한 장르의 시 조각으로 구성된 RIFMA라는 데이터셋을 출시하였으며, 이는 현대의 대형 언어 모델이 시적 텍스트에서 스트레스 마크를 정확하게 배치할 수 있는 능력을 평가하는 데 사용될 수 있습니다. 이러한 자원들은 생성적 시 시스템의 개발과 평가에 필요한 유용한 도구를 제공합니다.

- **Technical Details**: 본 논문은 러시아어의 음절-강세 시의 운율과 운문에 대한 새로운 접근 방식을 제시하며, 이를 통해 생성 시의 기술적 분석을 자동화할 수 있는 도구를 개발하고자 했습니다. 이 도구는 정확도, 확장성, 해석 가능한 출력을 제공함으로써 시의 평가가 전문적인 개입 없이도 이루어질 수 있도록 합니다. RPST는 러시아어 시 및 노래의 텍스트 분석과 마크업을 지원하며, 시의 음절 배치, 운율 탐지, 그리고 기술 수준을 평가합니다.

- **Performance Highlights**: RPST의 알고리즘은 기능별 단어의 스트레스 조정, 복수 음절의 전치사 처리 등의 규칙을 포함하여 효율적인 스트레스 지정을 가능하게 합니다. 또한, 신경망 기반 모델을 통해 사전 불일치(OOV) 단어에 대해서도 유연한 스트레스 배치를 지원합니다. 논문에서는 생성된 시와 인간의 시를 비교하는 측면에서, 평가 세션을 통해 다양한 기준에 따라 시적 기준 준수 여부를 평가하는 방법도 제시했습니다.



### A database to support the evaluation of gender biases in GPT-4o outpu (https://arxiv.org/abs/2502.20898)
Comments:
          ISCA/ITG Workshop on Diversity in Large Speech and Language Models

- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 공정성을 평가하기 위해 새로운 데이터베이스 구축 방법론을 제안합니다. 본 접근법은 성별 편견을 넘어 LLM이 생성하는 언어의 공정성을 검토하는 방법으로, 전통적인 평가 기준과 차별화됩니다. 논문은 Feminist Standpoint Theory를 기반으로 하여, 사회적 권력 불균형과 소외된 집단 보호의 중요성을 강조하고 있습니다.

- **Technical Details**: 이 연구는 LLM의 출력을 평가하기 위해 성별 편견과 관련된 다양한 프롬프트를 설계하고, 이를 바탕으로 데이터셋을 생성합니다. GPT-4o 모델을 사용하여 2024년 동안 7504개의 프롬프트와 9216개의 응답을 체계적으로 평가합니다. 이 과정에서 LLM 자신이 가지고 있는 고유한 관점을 배제하지 않고, 데이터 생성 과정 및 결과에 내재된 편견을 명확하게 드러내고자 합니다.

- **Performance Highlights**: 제안된 데이터베이스는 LLM 출력을 평가하며 그 공정성 문제를 심도 있게 탐구하는 데 기여하고 있습니다. 연구 결과, LLM의 출력은 그 시스템의 관점에 의해 영향을 받으며, 이는 공정성과 정의에 대한 평가의 필수 요소로 작용합니다. 연구진은 연구 과정에서의 모든 편견과 가정들을 투명하게 드러내는 것을 필수적인 목표로 삼고 있습니다.



### Beyond Demographics: Fine-tuning Large Language Models to Predict Individuals' Subjective Text Perceptions (https://arxiv.org/abs/2502.20897)
Comments:
          Reviewed ARR December 2024

- **What's New**: 이 연구에서는 대규모 언어 모델(LLMs)을 사용하여 주관적인 질문에 대한 주석자 변동성을 더 정확하게 모델링하는 방법을 제안합니다. 이는 LLM이 사회 인구학적(sociodemographic) 특성에 기반하여 주석자의 행동을 학습할 수 있는지 여부를 탐구하고 있습니다. 연구 결과, LLM이 특정 주석자의 행동을 더 잘 모델링할 수 있지만, 이는 개별 주석자의 행동을 학습하는 데 주로 기인하며 사회적 패턴을 잘 반영하지 못함을 보여줍니다.

- **Technical Details**: 이 연구에서는 새로운 접근법을 도입하여 개인화(persona) 프롬프트와 주석자 모델링을 결합합니다. 데이터 세트인 DeMo는 친밀감(intimacy), 공격성(offensiveness), 공손함(politeness), 안전(safety), 감정(sentiment) 과 같은 다섯 개의 기존 데이터 세트를 통합하여 구성되었습니다. 연구의 주요 연구 질문들은 LLM이 주석자의 사회적 특성(sociodemographics)을 기반으로 얼마나 잘 모델링할 수 있는지, 일반화 가능성은 있는지, 그리고 주석 행동과의 관련성을 얼마나 학습하는지를 포함합니다.

- **Performance Highlights**: 연구에서 LLM은 주석자 ID와 사회적 특성을 포함할 때 기본(Baseline) 모델보다 성능이 향상됨을 보였지만, 새로운 주석자에 대한 일반화 능력은 없음을 확인했습니다. 또한, LLM은 사회적 특성이 부여된 모델이 고유한 특성을 가진 주석자에서 개선되었으며, 이는 주석 행동과 사회적 특성 간의 의미 있는 연결을 배우지 못했음을 보여줍니다. 이러한 결과는 LLM을 사용한 사회적 변동성 및 행동 시뮬레이션의 현재 유용성에 대한 의문을 제기합니다.



### ProBench: Benchmarking Large Language Models in Competitive Programming (https://arxiv.org/abs/2502.20868)
- **What's New**: 최근 OpenAI-o3와 DeepSeek-R1과 같은 새로운 reasoning language models가 등장하면서, 대규모 언어 모델인 LLMs가 발전의 새로운 단계에 접어들었습니다. 그러나 기존의 코드 평가 벤치마크는 고급 LLMs의 코드 추론 능력을 충분히 평가하기에 부족합니다. 이를 해결하기 위해, 우리는 ProBench를 제안하여 경쟁 프로그래밍에서 LLMs를 벤치마킹하고, International Collegiate Programming Contest에서 영감을 받았습니다.

- **Technical Details**: ProBench는 2024년 7월부터 12월까지 Codeforces, Luogu, Nowcoder 플랫폼에서 수집한 경쟁 프로그래밍 문제의 종합적인 세트를 기반으로 합니다. 각 문제는 문제 난이도와 알고리즘 태그가 포함된 문제 속성 시스템을 통해 정식으로 분류됩니다. ProBench는 이러한 데이터를 사용하여 9개의 최신 LLMs의 코드 추론 능력을 여러 차원에서 평가하며, 이를 통해 사고 과정 분석(thought chain analysis), 오류 유형 진단(error type diagnosis), 그리고 추론의 깊이 평가(reasoning depth evaluation)를 수행합니다.

- **Performance Highlights**: 실험 결과, QwQ-32B-Preview 모델이 20.93점으로 가장 높은 성능을 기록하며, DeepSeek-V3는 16.38점으로 그 뒤를 이었습니다. 이러한 결과는 전문화된 추론 작업으로 훈련된 모델들이 일반 목적 모델보다 프로그래밍 성능에서 우수함을 나타냅니다. 더 나아가 알고리즘 적응성(algorithm adaptability) 및 추론의 충분성(reasoning sufficiency) 등 프로그래밍 능력 향상을 위한 주요 분야를 분석하여, 앞으로의 추론 모델 발전에 대한 중요한 통찰을 제공합니다.



### Better Benchmarking LLMs for Zero-Shot Dependency Parsing (https://arxiv.org/abs/2502.20866)
Comments:
          Accepted at NoDaLiDa/Baltic-HLT 2025

- **What's New**: 이 논문은 최신 LLM(large language models)의 구문 분석 성능을 조사하며, 기존의 암묵적 기준을 사용하지 않는 다양한 기준과 비교합니다. 주목할 점은, LLM들이 대부분의 언어에서 가장 단순한 랜덤 트리와 같은 기준조차 초월하지 못한다는 것입니다. 오직 최신의 가장 큰 LLaMA 모델만이 대다수 언어에서 이러한 성능을 보였지만 여전히 낮은 성과를 기록했습니다. 이는 공개 LLM들이 정확한 zero-shot 구문 분석을 제공하기에는 부족하다는 것을 시사합니다.

- **Technical Details**: 이 논문에서는 LLM을 활용한 zero-shot 의존 구문 분석에 초점을 맞추고 있습니다. 구문 분석은 주어진 문장에서 단어 간의 의존 관계를 라벨이 있는 방향성 관계 집합으로 표현하는 작업입니다. zero-shot 분석에서는 특정 태스크에 대한 라벨이 없는 데이터 없이 의존 관계를 추정하는 것을 목표로 합니다. 이러한 접근법은 LLM의 일반적인 사전 훈련 지식을 활용하여 미리 보지 않은 데이터에서 구문 관계를 추론하게 합니다.

- **Performance Highlights**: 연구 결과, LLM들은 제안된 여러 기준보다 높은 성능을 보이지 않았습니다. 특정 경우에만 최신의 LLaMA가 소폭 더 나은 성과를 보였지만 전체적으로는 여전히 의미 있는 성능 향상을 보여주지는 못했습니다. 이러한 성과는 LLM들이 암묵적 정보에 의존하기보다 실제로 입력을 처리하는 능력이 낮다는 것을 보여줍니다. 따라서 이 결과들은 LLM들이 구문 분석에 필요한 소스 간 복잡한 관계를 이해하는 데 한계가 있음을 반증합니다.



### Do Language Models Understand Honorific Systems in Javanese? (https://arxiv.org/abs/2502.20864)
- **What's New**: 이번 논문에서는 자바어의 복잡한 존칭 체계를 포착하기 위해 신중하게 편집된 데이터셋인 Unggah-Ungguh를 소개합니다. 이 데이터셋은 소셜 계층과 문맥에 따라 단어와 구문 선택을 규정하는 자바어의 언어 에티켓인 Unggah-Ungguh Basa의 뉘앙스를 담고 있습니다. 연구진은 이 데이터셋을 사용하여 자바어 존칭의 다양한 수준을 처리하는 언어 모델의 능력을 평가하고, 크로스-링구얼 기계 번역 실험을 통해 자바어와 인도네시아어 간의 번역 성능을 분석합니다.

- **Technical Details**: Unggah-Ungguh 데이터셋은 네 가지 존칭 수준(Ngoko, Ngoko Alus, Krama, Krama Alus)을 포함하여 자바어의 언어적 특성을 체계적으로 분류하고 주석을 달았습니다. 이 논문은 다양한 언어 모델, 포함된 언어 중심 모델 및 다국적 모델을 포함한 여러 접근 방식을 사용하여 존칭 레벨 분류, 존칭 스타일 변경, 크로스-링구얼 존칭 번역 및 대화 생성과 같은 4개의 NLP 작업을 다룹니다. 이를 통해 연구진은 존칭 사용과 관련된 잠재적 편향을 평가하고, 사용자의 역할과 문맥에 맞는 대화 생성 능력을 분석합니다.

- **Performance Highlights**: 연구 결과, 현재의 언어 모델은 대부분의 존칭 수준에서 어려움을 겪고 있으며 특정 존칭 계층에 대한 편향을 보였습니다. Unggah-Ungguh 데이터셋은 4,024개의 문장으로 구성되어 있으며, 자바어의 복잡한 사회언어적 구조를 반영하는 귀중한 자원으로 NLP 모델의 정확성과 문화적 민감도를 높이는 데 기여할 것입니다. 본 연구는 향후 다른 저자원 언어 연구에 대한 기초 자료로서 유용할 것으로 기대됩니다.



### The Power of Personality: A Human Simulation Perspective to Investigate Large Language Model Agents (https://arxiv.org/abs/2502.20859)
- **What's New**: 이 논문에서는 기존에 존재하는 큰 언어 모델(LLM)의 능력에 관한 설명을 넘어, 인간의 지능과의 연결 고리를 찾기 위해 '인간 시뮬레이션(human simulation)'의 관점에서 LLM 지능을 체계적으로 조사했습니다. 연구팀은 성격 특성(personality traits)이 문제 해결 및 창의성에 미치는 영향을 분석하고, 단일 에이전트와 다중 에이전트 간의 협업 성과의 연관성을 탐구했습니다. 이를 통해 특정 성격 특성이 논리적 정확성(closed tasks) 및 창의적 산출물(open tasks)에 미치는 영향을 밝혀냈습니다.

- **Technical Details**: 연구에서는 LLM 에이전트에 Big Five personality traits를 할당하고, 이 특성들이 폐쇄형 과제(closed tasks) 및 개방형 과제(open tasks)에서 성과에 미치는 영향을 평가했습니다. 각 에이전트의 성격 특성은 긍정적 및 부정적 차원을 통해 수학적으로 표현되며, 연구는 성격 특성이 다를 때 에이전트 간의 협동 방식과 성과에 미치는 영향을 다룹니다. 실험은 초기 답변 생성, 그룹 토론, 최종 답변 생성의 세 단계로 나누어 진행되었습니다.

- **Performance Highlights**: 연구 결과, 특정 성격 특성이 폐쇄형 과제에서 문제 해결 능력과 개방형 과제에서의 창의성에 유의미한 영향을 미침을 확인하였습니다. 또한 단일 에이전트의 성과는 다중 에이전트의 협력 성과에 영향을 미치며, 다양한 성격 조합이 집단 지능을 육성하고 개인의 능력을 초과하는 성과를 달성하도록 돕습니다. 이러한 발견은 LLM 기반 에이전트가 인간과 유사한 행동 양식을 나타내며, 다음 토큰 예측(next-token prediction)을 통해 인간의 언어와 결정을 모방할 수 있다는 점을 강조합니다.



### MAMUT: A Novel Framework for Modifying Mathematical Formulas for the Generation of Specialized Datasets for Language Model Training (https://arxiv.org/abs/2502.20855)
- **What's New**: 이 논문에서는 Math Mutator (MAMUT) 프레임워크를 소개하여, 수학 식을 LaTeX 표기법으로 변환한 후 동등한 표현과 허위 표현을 생성할 수 있도록 설계하였다. 머신러닝 모델의 수학 콘텐츠 인코딩 향상을 위한 특화된 데이터셋의 개발에 중점을 두고 있다. 이를 통해 다양한 수학 표기법을 포괄하는 4개의 대규모 데이터셋을 생성하였다.

- **Technical Details**: MAMUT는 동등한 수학적 식 (EquVG)과 비슷하지만 서로 다른 수학적 식 (FalseVG)을 생성하는 기능을 제공한다. 이 프레임워크는 변수 및 함수 식별자를 무작위로 변경하고 LaTeX 표기법의 변화를 포함하여 교환 가능성과 대칭성 같은 수학적 특성을 활용한다. 또한, MAMUT는 텍스트 내의 수학 LaTeX 표기에도 적용되어, 식별자와 표기 스타일의 일관된 변경을 보장한다.

- **Performance Highlights**: MAMUT를 통해 생성된 데이터셋은 계산식 완성 과제와 같은 수학 언어 모델의 추가 훈련을 위한 자료로 활용될 수 있다. 기존 데이터셋의 한계를 극복하고, 다양한 수학적 표기법을 통해 모델의 학습 능력을 향상시키는 데 기여할 것으로 기대된다. 이를 통해 수학 정보 검색과 같은 응용 프로그램에서의 성능 개선이 가능할 것으로 보인다.



### Learning to Substitute Components for Compositional Generalization (https://arxiv.org/abs/2502.20834)
Comments:
          23 pages, 9 figures, preprint, the extension paper of the paper (arXiv:2306.02840)

- **What's New**: 본 논문에서는 신경 언어 모델의 조합 일반화(compositional generalization)에서의 결함을 해결하기 위한 새로운 기법인 Component Substitution (CompSub)을 소개합니다. 기존의 수작업 방식인 조합 데이터 증강(compositional data augmentation)이 성과가 제한적이었으나, CompSub는 다중 수준(multi-grained) 조합을 가능하게 하여 학습의 폭을 넓힙니다. 또 다른 기여로는 Learning Component Substitution (LCS) 프레임워크를 통해 고급 조합을 위한 효과적인 방법을 제시하는 점이 있습니다.

- **Technical Details**: CompSub는 데이터의 조합(bias)을 주입하는 방식으로, 학습 문장에서 특정 구성요소를 다른 문장에서 대체함으로써 새로운 예제를 생성합니다. 이 과정에서 LCS 프레임워크는 컴포넌트 교체 확률을 학습하여 어려운 조합(concept)을 우선적으로 선택하도록 합니다. 이들 모두는 신경 언어 모델의 손실을 극대화하여 훈련을 진행하는 방식으로 통합됩니다.

- **Performance Highlights**: 본 논문의 방법들은 SCAN, COGS, GeoQuery와 같은 네 가지 기준 벤치마크에서 실험하여 우수한 성능 향상을 입증합니다. CompSub, LCS, 그리고 LCS-ICL은 각각 66.5%, 10.3%, 1.4%, 8.8%의 성능 개선을 보여주었습니다. 이로 인해 본 연구가 조합 일반화 향상에 기여할 수 있음을 분명히 하고 있습니다.



### Plan2Align: Predictive Planning Based Test-Time Preference Alignment in Paragraph-Level Machine Translation (https://arxiv.org/abs/2502.20795)
Comments:
          Preprint. Code will be released at Plan2Align GitHub link: this https URL

- **What's New**: 이번 논문에서는 Machine Translation (MT) 분야에서 단문 수준의 번역만을 생각하는 전통적인 방식에서 벗어나, 긴 텍스트를 보다 유기적으로 번역할 수 있는 새로운 기법인 Plan2Align을 소개합니다. 특히, 본 논문은 번역 프로세스를 예측적 계획 문제로 간주하고, Model Predictive Control (MPC) 원칙을 적용하여 번역 결과를 반복적으로 개선하는 방법을 제안합니다. 기존의 방법들은 긴 문맥을 고려하는 데 한계가 있었으나, Plan2Align은 이러한 도전을 해결하고자 합니다.

- **Technical Details**: Plan2Align은 모델의 재조정을 쉽게 하기 위해 테스트 시간에 최적화된 자기 재작성(self-rewriting) 프레임워크를 도입했습니다. 이를 통해 이전 반복에서 높은 품질의 문장을 통합하면서 번역의 일관성과 유창성을 향상시킵니다. 본 연구는 문맥 레벨(context-level)에서의 번역 품질을 중요시하며, 서로 관계가 있는 문장 그룹의 번역을 정교하게 개선하는 데 중점을 두고 있습니다.

- **Performance Highlights**: WMT24 자료집 위에서 Plan2Align은 LLaMA-3.1 8B 모델을 사용하여 paragraph-level 번역 성능을 테스트하였으며, 그 결과 기존의 훈련 및 테스트 시간 정렬 방법들보다 뛰어난 성능을 보였습니다. 이러한 방법의 적용으로, Plan2Align은 문맥의 일관성을 유지하면서도 번역 품질을 크게 향상시키는 데 기여하였습니다. 앞으로의 연구에서도 Plan2Align은 다양한 fine-tuning 접근법과 결합하여 더욱 발전할 수 있는 가능성을 보여줍니다.



### Chain-of-Thought Matters: Improving Long-Context Language Models with Reasoning Path Supervision (https://arxiv.org/abs/2502.20790)
Comments:
          14 pages,6 figures

- **What's New**: 이번 연구는 최근 대규모 언어 모델(LLMs)의 발전에 따라 긴 맥락(long-context) 작업 처리의 도전 과제를 강조합니다. Chain-of-Thought (CoT) 방식이 다단계 추론에 효과적이지만, 긴 맥락 시나리오에서의 효용은 상대적으로 탐구가 부족했습니다. 다수의 작업에 대한 체계적인 조사를 통해 CoT의 이점이 대부분의 긴 맥락 시나리오에서 일반화됨을 입증하였고, 컨텍스트 길이가 증가할수록 효과가 더욱 증대됨을 관찰했습니다.

- **Technical Details**: 이 연구는 LongRePS라는 과정-감독(process-supervised) 프레임워크를 제안합니다. 이 프레임워크는 모델이 높은 품질의 추론 경로를 생성하도록 교육하여 긴 맥락 성능을 향상시킵니다. Self-sampling 기법을 통해 추론 경로를 부트스트랩하고, 긴 맥락 시나리오에 특화된 새로운 품질 평가 프로토콜을 포함합니다.

- **Performance Highlights**: 다양한 긴 맥락 벤치마크에서 실험한 결과, 우리의 접근 방식은 결과 감독(baselines) 기준에 비해 상당한 개선을 이루었습니다. MuSiQue 데이터셋에서 LLaMA와 Qwen 모델의 경우 각각 +13.6 및 +3.8 포인트 개선을 보였고, 다양한 QA 작업에서 평균 +9.3/+8.1 포인트의 교차 도메인 일반화 성능을 달성했습니다. 이와 함께 연구를 촉진하기 위해 코드, 데이터 및 훈련된 모델을 공개하였습니다.



### GraphCheck: Multi-Path Fact-Checking with Entity-Relationship Graphs (https://arxiv.org/abs/2502.20785)
- **What's New**: 이번 논문에서는 GraphCheck라는 자동 사실 확인 프레임워크를 제안합니다. 이 프레임워크는 복잡한 주장들을 엔티티-관계 그래프로 변환하여 다각적인 검증 경로를 생성합니다. 특히, DP-GraphCheck라는 두 단계 변형을 도입하여 성능을 향상시키고, 초기 필터링 단계로 직접 프롬프팅을 포함시킵니다. 실험 결과 HOVER 및 EX-FEVER 데이터셋에서 기존 방법보다 뛰어난 성능을 보였습니다.

- **Technical Details**: GraphCheck는 LLM(대형 언어 모델)을 활용하여 텍스트 기반 주장을 구조화된 엔티티-관계 그래프로 변환합니다. 이 그래프는 사실 삼중항으로 구성되며, 각 삼중항은 엔티티 간의 관계를 정의하고 독립적으로 검증 가능한 하위 주장을 형성합니다. DP-GraphCheck는 구조적 검증을 선택적으로 적용하여 정확도와 효율성을 높이면서 복잡한 멀티-홉 주장을 효과적으로 처리합니다. 또한, 역동적인 검색 전략을 통해 주장 복잡도에 따라 검색 문서의 수를 조정하여 검증 정확도를 개선합니다.

- **Performance Highlights**: 실험 결과, GraphCheck는 복잡한 멀티-홉 검증 작업에서 다른 방법들보다 뛰어난 성능을 보였습니다. 특히, 두 단계 프레임워크는 다양한 사실 확인 방법에 걸쳐 일반화가 잘 되고, 복잡한 멀티-홉 주장에서도 성능 향상을 보여줍니다. 또한, 그래프 구조를 통한 주장의 세분화로 보다 포괄적이고 체계적인 사실 확인이 가능해졌습니다.



### Triple Phase Transitions: Understanding the Learning Dynamics of Large Language Models from a Neuroscience Perspectiv (https://arxiv.org/abs/2502.20779)
Comments:
          46 pages

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 학습 과정에서 관찰되는 '상태 전이(phase transition)' 현상을 분석했습니다. 기존 연구들은 이러한 전이를 개별적 측면에서만 분석했으나, 본 연구는 인간의 뇌와의 유사성, LLM의 내부 상태, 그리고 하위 과제 성능 간의 상관관계를 통합적으로 분석하였습니다. 이로써 LLM의 학습 역동성을 새로운 관점에서 이해하고자 합니다.

- **Technical Details**: 연구에서는 OLMo-2, OLMo-0724, LLM-jp, Amber의 네 가지 사전 훈련된 모델을 사용하여 학습 동역학을 분석하였습니다. 각 모델은 서로 다른 데이터셋과 토크나이저를 활용하며, 파라미터 수는 6.74~7.3B 범위로 설정되어 있습니다. 연구 팀은 세 가지 분석 접근법을 통해 모델의 학습 과정에서 뇌 활동과의 정렬도를 조사하고, 여러 단계의 상관관계를 밝혀내었습니다.

- **Performance Highlights**: 주목할 만한 결과로, LLMs는 세 가지 주요 단계의 상태 전이를 겪습니다: (1) 작업 지침을 따르기 시작할 때 전체 뇌와의 정렬이 급증합니다. (2) 하위 과제 정확도가 일시적으로 정체될 때 LLM들이 뇌와 멀어지는 현상이 관찰됩니다. (3) LLM들이 하위 과제를 해결하는 능력을 갖추게 되면 다시 뇌와의 정렬이 나타납니다. 이러한 연구 결과는 LLM의 emergent capabilities가 훈련 과정 동안 어떻게 형성되고 통합되는지를 보여줍니다.



### The Rise of Darkness: Safety-Utility Trade-Offs in Role-Playing Dialogue Agents (https://arxiv.org/abs/2502.20757)
- **What's New**: 이 논문은 역할 놀이 대화 에이전트에서의 안전성-유용성 트레이드오프(safety-utility trade-off)를 최초로 밝혀내고 수량화했습니다. 이를 해결하기 위해 Adaptive Dynamic Multi-Preference (ADMP) 방법을 제안하며, 이는 사용자 쿼리와 캐릭터 설정 간의 위험 결합(risk coupling)에 따라 안전성과 유용성 선호를 동적으로 조절합니다.

- **Technical Details**: ADMP는 실시간으로 위험 결합을 감지하여 대화 에이전트가 안전성을 극대화하는 동시에 캐릭터 묘사의 풍부함을 유지할 수 있도록 합니다. 또한 Coupling Margin Sampling (CMS)을 도입하여 위험 결합이 가장 두드러진 경계 사례(edge cases)를 타겟팅해 더욱 향상된 결합 탐지를 가능합니다.

- **Performance Highlights**: 실험 결과, 제안된 ADMP 방법이 안전성 지표를 향상시키면서도 역할 놀이의 유용성을 충분히 유지하는 것으로 나타났습니다. 다양한 대표적인 모델을 비교 분석하여 이들 모델 간 안전성과 유용성 사이의 관계에 대한 중요한 통찰을 제공했습니다.



### Mitigating Hallucinations in Large Vision-Language Models by Adaptively Constraining Information Flow (https://arxiv.org/abs/2502.20750)
Comments:
          Accepted to AAAI 2025. Camera ready version

- **What's New**: 이 논문에서는 대규모 비전-언어 모델(LVLM)에서 발생하는 객체 환각(object hallucination) 문제를 해결하기 위한 새로운 방법을 제안합니다. 이 문제는 비관련 시각적 특성에 대한 과신(overconfidence)으로 인해 발생하며, 이를 해결하기 위해 변량 정보 병목(Variational Information Bottleneck, VIB) 기법을 도입합니다. 제안하는 시스템 AdaVIB는 시각적 특성이 LLM의 단어 임베딩(spatial embedding) 공간으로 매핑될 때의 정보 흐름을 제어하여 객체 환각 현상을 줄이는 데 중점을 두고 있습니다.

- **Technical Details**: AdaVIB는 비전-언어 프로젝터(vision-language projector)에 대한 미세 조정(fine-tuning)과정에서 압축 항(compression term)을 추가하여 과신 문제를 완화합니다. 이는 소프트 비주얼 토큰(soft visual tokens)에 확률적 노이즈를 주입하여 정보 흐름을 제어하는 방식으로 이루어집니다. 또한, 엔트로피 기반의 노이즈 제어 메커니즘을 통해 매핑되는 시각 정보의 부드러움(smoothness)을 적절히 관리하며, 다양한 LVLM 아키텍처에 적용할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, AdaVIB는 두 개의 객체 환각 벤치마크에서 반복적인 개선을 보여줍니다. 이는 소프트 비주얼 토큰과 LLM의 단어 임베딩 간의 유사성 분포(smoothness distribution)를 효과적으로 조절하여 과신 문제를 줄였음을 시사합니다. 본 연구는 VIB를 통해 LVLM의 객체 환각 문제를 완화하는 전례 없는 접근 방식을 제공하며, 이는 신뢰할 수 있는 AI 시스템 구현에 있어 중요한 기여를 한다고 할 수 있습니다.



### Teach-to-Reason with Scoring: Self-Explainable Rationale-Driven Multi-Trait Essay Scoring (https://arxiv.org/abs/2502.20748)
- **What's New**: 이번 연구에서는 복합 특성 자동 에세이 평가(Automated Essay Scoring, AES)에서 생성된 점수에 대한 합리적인 설명을 제공하는 RaDME(자기 설명 가능한 합리적 다중 특성 자동 에세이 점수 프레임워크)를 제안합니다. 기존의 AES 시스템은 점수는 정확하게 예측하였지만, 그 점수의 이유를 설명하는 데 실패하여 투명성이 부족했습니다. RaDME는 대형 언어 모델(LLMs)의 추론 능력을 작은 모델로 증류하여 특성 점수와 그에 따른 설명을 순차적으로 생성하도록 설계되었습니다.

- **Technical Details**: RaDME는 LLM의 추론 능력을 활용하여 작지만 효과적인 점수 모델을 훈련시킵니다. 이 모델은 점수와 그에 대한 합리적 설명을 동시에 생성함으로써 점수 결정 과정이 명확한 이유에 기반하도록 유도합니다. 모델은 여러 특성 및 프롬프트에 걸쳐 점수-합리성 쌍을 다중 작업 학습(multi-task learning) 접근법으로 최적화하여 구성됩니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, RaDME는 뛰어난 점수 성능을 달성하며 최근의 최첨단 방법들보다 더 높은 성과를 보입니다. RaDME는 점수-이유 학습 초기 접근 방식을 통해 생성된 합리적 설명의 품질을 향상시킵니다. 이는 자동 평가의 투명성을 높이고, 더 해석 가능하고 신뢰할 수 있는 AES를 위한 기초를 마련하는 중요한 단계입니다.



### Retrieval Backward Attention without Additional Training: Enhance Embeddings of Large Language Models via Repetition (https://arxiv.org/abs/2502.20726)
- **What's New**: 이 논문은 미리 학습된(Pre-trained) 언어 모델을 제로샷(Zero-shot) 상황에서 성능을 향상시키기 위한 새로운 방법론을 제안합니다. 연구팀은 문맥 정보 인코딩을 개선하기 위해 새로운 역방향 주의 메커니즘(Backward Attention Mechanism)을 도입하였습니다. 제안된 방법은 C-MTEB(Chinese Massive Text Embedding Benchmark)에서 성능 개선을 입증하여 제로샷 학습 역량을 발전시키기 위한 귀중한 통찰을 제공하고 있습니다.

- **Technical Details**: 이 연구는 텍스트 임베딩 학습(Text Embedding Learning)에 초점을 맞추고 있으며, 자연어 처리(NLP)에서의 텍스트를 벡터 표현으로 변환하는 방법을 다룹니다. 특히, 자동 회귀 언어 모델(Autoregressive Language Model)이 어떻게 이전 토큰을 기반으로 다음 토큰을 예측하는지 설명하고, 주의 메커니즘(Attention Mechanism)의 수학적 원리를 다룹니다. 새롭게 도입된 역방향 주의 메커니즘은 기존의 하삼각(attention matrix) 구조와 대조적으로 후속 컨텍스트와의 관계를 고려하여 문맥 인코딩 품질을 향상시킵니다.

- **Performance Highlights**: 최신의 자동 회귀 언어 모델에서 텍스트 반복(Text Repetition)을 적용하여 성능을 대폭 향상시켰습니다. 이 방식은 모델이 정보를 효과적으로 캡처할 수 있도록 도와줍니다. 또, 제안된 방법은 추가적인 모델 학습 없이 기존 모델의 성능을 개선하여 연산 비용을 절약하는 동시에 구체적인 토큰의 임베딩을 선택적으로 개선합니다.



### JAM: Controllable and Responsible Text Generation via Causal Reasoning and Latent Vector Manipulation (https://arxiv.org/abs/2502.20684)
Comments:
          10 pages, 3 figures, and 6 tables

- **What's New**: 이 논문에서는 JAM (Just A Move)이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 LLM의 생성 프로세스를 제어하고 해석하기 위해 원인과 결과 분석을 통합하며, 이를 통해 보다 책임감 있게 텍스트를 생성할 수 있습니다. JAM은 기존의 CTG (Controllable Text Generation) 방법보다 최대 22% 높은 성능 향상을 보여주며, 모델의 원인과 결과성을 유지하는 동시에 출력의 질과 신뢰성을 보장합니다.

- **Technical Details**: JAM은 잠재 공간(latent space) 내에서의 원인과 결과 분석을 통해 효과적으로 LLM의 생성 과정을 제어하는 접근법입니다. 잠재 벡터(latent vector)는 LLM 아키텍처의 기본 요소로, 모델이 텍스트 생성을 수행하는 데 필요한 정보를 인코딩합니다. 이 연구에서는 LLM 생성 과정에서의 잠재 벡터의 행동과 내재된 원인 관계를 조사하고, 이러한 관계를 분석하여 더 정교한 텍스트 생성을 위한 방법론을 제안합니다.

- **Performance Highlights**: JAM은 HHH (Helpful, Honest, and Harmless) 기준 및 독성 감소 작업에서 다양한 인기 LLM 모델에서 실험을 수행하여 성능을 평가했습니다. JAM은 기존 방법에 비해 여러 지표에서 최대 10% 개선된 점수를 보여주었으며, GPT-4 평가에서도 대부분의 경우 우세한 결과를 보였습니다. 이는 JAM이 책임감 있고 현실적인 텍스트 생성을 위한 효과적이고 효율적인 방법임을 강조합니다.



### Fine-tuning BERT with Bidirectional LSTM for Fine-grained Movie Reviews Sentiment Analysis (https://arxiv.org/abs/2502.20682)
Comments:
          14 pages, 5 figures, published in International Journal On Advances in Systems and Measurements, volume 16, numbers 3 and 4, 2023

- **What's New**: 이번 연구는 Bidirectional Encoder Representations from Transformers (BERT)와 Bidirectional LSTM (BiLSTM)을 결합하여 영화 리뷰에 대한 이진 정서 분석 (Sentiment Analysis, SA)을 향상시키고자 합니다. 기존의 문헌에서 주로 이진 분류 또는 세부적인 정서 분류에 집중해왔던 결과, 이 두 가지를 동시에 다루는 연구가 부족했습니다. 본 연구는 BERT 모델을 세밀하게 조정하는 방법론을 제공하면서 이러한 격차를 해소하고자 합니다. 특히, 영화 리뷰 영역에서의 SA의 중요성을 강조하며 해당 분야의 데이터셋을 활용하였습니다.

- **Technical Details**: 연구에서 사용한 BERT 모델은 복잡한 정서 분석을 위해 BiLSTM과 결합되었습니다. 이 모델은 전후 문맥을 모두 이해함으로써 정서의 복잡성을 보다 효과적으로 처리할 수 있습니다. 저자들은 이진 및 다중 클래스 정서 분류를 수행하고, 실제로 Synthetic Minority Oversampling Technique (SMOTE)와 NLP Augmenter (NLPAUG)와 같은 기술들을 사용하여 모델 성능을 개선했습니다. 전체 감정 극성을 구하기 위해 새로운 휴리스틱 알고리즘을 도입하였고, 이를 통해 다양한 정서 분류 작업에 적용하였습니다.

- **Performance Highlights**: 모델의 성능은 유명한 IMDb 데이터셋에서 이진 분류 정확도 97.67%를 기록하였고, 이는 기존 SOTA 모델을 0.27% 초과하는 값입니다. 또한, SST-5 데이터셋의 다중 클래스 분류에서 59.48%의 정확도를 달성하여 기존 BERT-large 기준을 3.6% 초과했습니다. 이러한 결과는 BERT와 BiLSTM을 결합한 접근 방식이 각종 평가에서 경쟁력 있는 성능을 발휘할 수 있음을 증명합니다.



### Disentangling Feature Structure: A Mathematically Provable Two-Stage Training Dynamics in Transformers (https://arxiv.org/abs/2502.20681)
- **What's New**: 이번 연구는 Transformers의 두 단계 학습 동역학(two-stage training dynamics)에 대한 이론적 분석을 제공합니다. 기존의 이론 분석은 이러한 현상을 거의 고려하지 않았으나, 이 논문에서는 GPT-2가 Counterfact 데이터셋에서 훈련되는 사례를 통해 시각적으로 이 과정을 보여줍니다. 이 논문의 주안점은 주어진 특성 구조를 분해(disentangled)하여 변별력을 분석하는 데 있습니다.

- **Technical Details**: 저자들은 컨텍스트 학습(in-context learning) 규칙 하에서 특성 학습(feature learning) 기법을 활용하여 Transformers의 동역학을 분석합니다. 이 과정에서 두 가지 유형의 특성 구조를 가지는 분리된 구조가 일반적이라는 점을 강조합니다. 예를 들어, 자연어에는 구문(syntax)과 의미(semantics)가 포함되며, 단백질(proteins)의 경우 1차 및 2차 구조가 존재합니다.

- **Performance Highlights**: 이 연구는 Transformers의 두 단계 최적화 과정에 대한 최초의 엄밀한 결과를 제시합니다. 또한, 이 과정이 attention weights의 스펙트럼 속성과 밀접하게 관련되어 있다는 보조 정리가 제시되어, 실험적으로 발견된 결과와 일치함을 보여줍니다.



### Prediction of Item Difficulty for Reading Comprehension Items by Creation of Annotated Item Repository (https://arxiv.org/abs/2502.20663)
- **What's New**: 본 논문은 텍스트 내용을 기반으로 한 아이템 난이도 예측에 대한 연구를 다루고 있다. 저자들은 뉴욕과 텍사스의 3-8학년 학생 데이터를 이용해 IRT(아이템 반응 이론) 기반의 난이도를 복원하는 방법을 제시한다. 새롭게 발표된 아이템 난이도 모델은 RMSE 0.52로 성능을 개선하며, 기존 데이터의 성능을 확실히 능가한다.

- **Technical Details**: 이 연구에서는 텍스트 해석에 필요한 여러 메타 데이터를 포함한 읽기 패세지를 구성하였다. 이러한 메타 데이터는 언어적 요소, 테스트 요소 및 맥락 특징으로 나누어지며, 이를 바탕으로 페널티 회귀(prediction model) 모델을 구축하였다. LLM(대형 언어 모델) 임베딩을 이용한 예측 성능 또한 기존 모델보다 다소 향상된 결과를 제공하였다.

- **Performance Highlights**: 이 모델은 기존의 baseline RMSE 0.92와 비교했을 때 향상된 성능을 보여 주며, 실제 난이도와 예측 난이도 간의 상관관계는 0.77에 달한다. 저자들은 LLM 임베딩과 언어적 특징의 단독 사용이 비슷한 예측 성능을 나타낸다는 점을 강조하며, 향후 다양한 응용 가능성을 제시하였다.



### Consistency Evaluation of News Article Summaries Generated by Large (and Small) Language Models (https://arxiv.org/abs/2502.20647)
Comments:
          21 pages, 6 figures, 4 tables

- **What's New**: 이번 논문은 다양한 텍스트 요약 기법을 탐구하고 있습니다. 기존의 전통적인 평가 지표인 ROUGE 점수 및 BERT 점수를 사용하여 생성된 요약을 평가하며, LLM(Large Language Model) 기반의 평가 방법도 도입하고 있습니다. 특히, 생성된 요약의 일관성을 측정하기 위한 새로운 메타 평가 점수를 소개하고 있습니다. XL-Sum 데이터셋을 사용하여 여러 요약 모델의 성능을 비교한 결과, 모든 요약 모델은 참조 요약보다 일관성을 나타냈습니다.

- **Technical Details**: 논문은 요약 방법을 추출적(extractive) 및 추상적(abstractive) 접근 방식으로 나누어 다루고 있습니다. XL-Sum 데이터셋은 BBC 웹사이트에서 수집한 100만 쌍의 뉴스 기사와 요약을 포함하고 있으며, 본 작업에서는 영어 데이터만을 사용합니다. 해당 데이터셋에서 37.37%의 수동 검토된 영어 요약이 사실과 상충하는 정보가 포함되어 있음을 지적하면서, 이러한 문제의 최소화를 위해 기사 길이에 대한 필터링을 진행했습니다.

- **Performance Highlights**: 요약 방법으로는 TextRank, BART, Mistral-7B-Instruct, OpenAI GPT-3.5-Turbo 등의 모델이 포함되어 있습니다. 논문은 LLM의 발전에도 불구하고, 소규모 모델들이 여전히 비용 효율적인 솔루션으로 빛날 수 있는 여지를 강조하고 있습니다. 이에 따라 대규모 LLM들이 보유하지 못하는 다양한 성능을 가진 소형 모델들과의 비교가 이루어졌습니다.



### LexRAG: Benchmarking Retrieval-Augmented Generation in Multi-Turn Legal Consultation Conversation (https://arxiv.org/abs/2502.20640)
Comments:
          10 pages

- **What's New**: LexRAG는 법률 도메인에서 RAG 시스템을 평가하기 위한 최초의 벤치마크로, 1,013개의 다중 턴 대화 샘플과 17,228개의 후보 법률 기사를 포함하고 있습니다. 이 벤치마크는 다섯 단계로 진행되는 질문을 포함하여, 법률 전문가가 주석을 달았습니다. LexRAG는 법률 상담의 복잡성을 이해하고 평가하기 위해 두 가지 주요 작업인 Conversational Knowledge Retrieval과 Response Generation을 포함하고 있습니다.

- **Technical Details**: LexRAG는 다중 턴 대화에서 법률 관련 지식을 정확하게 검색하고 법적으로 타당한 응답을 생성하는 시스템을 평가하는 데 중점을 둡니다. 데이터셋은 법률 전문가에 의해 신중하게 주석이 달린 1,013개의 대화를 포함하며, 각 대화는 다섯 개의 질문과 응답으로 구성됩니다. LexiT라는 법률 RAG 툴킷을 통해 RAG 시스템을 위한 다양한 구현이 가능하며, LLM-as-a-judge 평가 파이프라인은 상세하고 효과적인 평가를 지원합니다.

- **Performance Highlights**: LexRAG를 통해 여러 LLM 및 검색 방법에 대한 종합적인 평가를 수행하여 현재 RAG 시스템이 법률 도메인에서 직면하고 있는 주요 한계와 문제점을 밝혀냈습니다. 이 분석은 법률 상담 시스템과 관련된 RAG의 발전을 위한 귀중한 통찰력을 제공하며, 향후 개선을 위한 로드맵을 제시합니다. LexRAG는 법률 AI 기술을 발전시키고, 다양한 도메인에서 RAG의 미래 개발을 위한 기초를 마련하는 역할을 합니다.



### Rectifying Belief Space via Unlearning to Harness LLMs' Reasoning (https://arxiv.org/abs/2502.20620)
- **What's New**: 이번 연구는 스푸리어스 신념(spurious beliefs)으로 인해 생기는 오류를 줄이기 위한 새로운 접근법을 제시합니다. 저자들은 모델이 사실상 잘못된 신념을 갖고 있을 수 있으며, 이러한 신념을 억제하고 올바른 신념을 증대시키는 방법을 통해 좀 더 신뢰할 수 있는 추론을 가능하게 한다고 주장합니다. Forward-Backward Beam Search (FBBS) 기법을 통해 신념을 식별하고, 신념 공간을 정리하는 방법을 제안합니다.

- **Technical Details**: 스푸리어스 신념을 억제하기 위해 경량화(unlearning) 기법을 적용하고, 이러한 신념의 참(true) 및 거짓(false) 여부를 판단하여 신념 공간을 재구성합니다. FBBS는 주어진 입력에서 정답을 도출하기 위해 필요한 정보의 설명을 생성하면서, 올바른 신념을 강조하는데 중점을 둡니다. 연구에서는 여러 QA 데이터셋에서 LLM(대형 언어 모델)의 성능을 평가합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 OLMo, Pythia, RedPajama와 같은 LLM 모델에서 답변 오류를 감소시키고 전반적인 성능을 향상시키는 것을 보여줍니다. 특히, OLMo의 경우 정확도가 최대 6.4포인트 증가했으며, unseen evaluation data에서 9.6포인트 개선된 결과를 기록했습니다. 이러한 강력한 일반화 능력은 신념 공간을 교정함으로써 잦은 오류를 줄이는 데 기여함을 나타냅니다.



### Continuous Adversarial Text Representation Learning for Affective Recognition (https://arxiv.org/abs/2502.20613)
Comments:
          6 pages, 3 figures, The 7th International Conference on Artificial Intelligence in Information and Communication (ICAIIC 2025)

- **What's New**: 본 논문에서는 감정 인식을 위한 새로운 프레임워크인 Continuous Adversarial Representation Learning (CARL)을 제안합니다. 기존의 언어 모델의 한계를 극복하고 감정 인식 태스크에서 효과적인 성능 향상을 목표로 하고 있습니다. 이 접근법은 연속적인 가치-각성(valence-arousal) 레이블링 시스템을 통해 대조적 학습을 가이드하며, 정서적 뉘앙스를 더 미세하게 포착할 수 있게 해줍니다.

- **Technical Details**: CARL은 문장 수준의 Momentum Continuous Contrastive Learning (MCCL)과 토큰 수준의 Gradient-based Perturbed Token Detection (PTD)을 결합한 두 가지 사전 훈련 작업으로 구성됩니다. 이러한 구조는 모델의 감정 임베딩 능력을 강화하는 데 중점을 두며, 목표 네트워크와 온라인 네트워크를 활용해 학습의 불안정성을 줄이고 더 견고한 표현을 가능하게 합니다. 표준적인 대조적 학습 기법과 달리, 이 방법은 Russell의 Circumplex 모델을 기반으로 감정을 두 차원 공간으로 표현하여 정서적 변이를 파악할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, 제안된 CARL 프레임워크는 감정 분류 벤치마크에서 최대 15.5%의 성능 향상을 달성하며, 연속 레이블의 중요성을 강조합니다. CARL은 세 가지 주요 감정 인식 태스크에서도 기존 방법들에 비해 우수한 성능을 보이며, 감정 표현 공간의 품질을 향상시키는 것을 입증합니다. 이러한 성과는 감정 표현 학습에서의 효과성을 더욱 증명하며, 정서적 이해의 정확성과 맥락적 관련성을 향상시키는 데 기여하고 있습니다.



### Leveraging Large Language Models for Building Interpretable Rule-Based Data-to-Text Systems (https://arxiv.org/abs/2502.20609)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLM)을 활용하여 완전히 해석 가능한 규칙 기반 데이터-텍스트 시스템을 순수 Python으로 자동 구현하는 간단한 접근 방식을 소개합니다. WebNLG 데이터셋에 대한 실험 평가 결과, 이 시스템은 직접 출력을 생성하기 위해 LLM을 사용했을 때보다 더 나은 품질의 텍스트를 생산하며, 동일한 데이터로 미세 조정된 BART 언어 모델보다 환각(hallucination)이 적게 발생합니다. 또한, 이 접근 방식은 신경망(neural) 접근 방식이 요구하는 처리 시간의 일부만으로 텍스트를 거의 즉각적으로 생성할 수 있습니다.

- **Technical Details**: 데이터-텍스트(data-to-text) 시스템의 개발에서 두 가지 주요 접근 방식이 있습니다: 규칙 기반(rule-based) 방법과 신경망(neural) 방법입니다. 규칙 기반 방법은 사전에 정의된 템플릿(template)과 언어 규칙을 사용하여 구조화된 데이터를 텍스트로 변환하는 반면, 신경망 방법은 딥 러닝 모델을 활용하여 데이터에서 텍스트로의 맵핑(mapping)을 자동으로 학습합니다. 본 논문에서는 LLM을 이용하여 규칙 기반 시스템을 구현하는 훈련 과정을 제안하며, 훈련 세트의 예제에 대해 LLM이 간단한 Python 코드를 작성하도록 요청하고, 이를 통해 최종적으로 입력 데이터에 대한 텍스트화(textualisation)를 생성할 수 있는 단일 Python 코드 파일을 생성합니다.

- **Performance Highlights**: WebNLG 데이터셋을 기반으로 실험을 수행한 결과, 자동 작성된 규칙 기반 시스템은 미세 조정된 신경망 모델의 BLEU 또는 BLEURT 점수를 달성하지는 못했지만, 환각 발생이 현저히 적으며 비트리비얼(non-trivial) 신경망 기준 모델을 능가하는 성과를 보였습니다. 또한, 제작된 시스템은 완전히 해석 가능하며, 필요 시 Python 프로그래머가 수정할 수 있는 높은 제어 가능성을 제공합니다. 이를 통해 시스템은 GPU 없이 단일 CPU에서 거의 즉시 텍스트를 생성하는 효율성을 보여주었습니다.



### Few-Shot, No Problem: Descriptive Continual Relation Extraction (https://arxiv.org/abs/2502.20596)
Comments:
          Accepted to AAAI 2025

- **What's New**: 본 논문에서는 AI 시스템이 동적인 현실 세계에서 변화하는 관계를 식별하고 적응하는 Few-shot Continual Relation Extraction (FCRE)을 위한 새로운 접근 방식을 제안합니다. 전통적인 메모리 기반 방법들이 한정된 샘플에 과적합(Overfitting)되며 예전 지식을 강화하는 데 실패하는 문제를 보여주고 있습니다. 이에 따라, 우리는 각 관계에 대한 설명을 생성하기 위해 대형 언어 모델(Large Language Model, LLM)을 활용하여 이 문제를 해결합니다.

- **Technical Details**: 제안된 방법은 두 개의 인코더를 활용한 검색 학습 프레임워크를 통해 샘플 및 클래스 표현 학습을 수행합니다. 또한, 각 샘플이 가장 잘 맞는 관계를 "검색(retrieve)"하여 예측하는 방법을 설계하였으며, 이는 관계 설명 벡터와 클래스 프로토타입(class prototype)을 통합하여 상호 순위 융합 점수를 사용하는 방식입니다. 특히, LLM은 설명 생성에만 활용되고, 훈련 또는 추론 과정에는 참여하지 않아 계산적인 부하를 최소화합니다.

- **Performance Highlights**: 여러 데이터셋에 대한 광범위한 실험을 통해 기존 방법들에 비해 상당한 성능 개선을 입증하였습니다. 본 방법은 연속 작업에서 강력한 성능을 유지하며, 재앙적 잊음(catastrophic forgetting) 문제를 효과적으로 해결하고 있습니다. 특히, 관계 설명을 통해 과거 지식을 안정적으로 유지하면서 새로운 정보를 학습할 수 있는 방법론을 제시합니다.



### Multi$^2$: Multi-Agent Test-Time Scalable Framework for Multi-Document Processing (https://arxiv.org/abs/2502.20592)
- **What's New**: 최근 테스트 시간 스케일링(test-time scaling)은 대형 언어 모델(LLMs)의 성능 향상에 긍정적인 결과를 보여주고 있습니다. 이 방식은 추론(inference) 중 전략적인 계산(resource allocation)을 통해 성능을 개선할 수 있음을 증명했지만, 특히 자연어 생성(NLG) 및 다중 문서 요약(MDS) 같은 분야에서는 연구가 미비한 상황입니다. 본 논문에서는 이러한 문제를 해결하기 위해 새로운 프레임워크인 Multi2를 제안하며, 이는 다양한 프롬프트를 사용해 후보 요약을 생성하고, 이를 집계하여 최종 요약을 만드는 방식을 채택합니다.

- **Technical Details**: 대형 언어 모델의 요약 능력과 스케일링 특성을 살펴보는 본 논문은 다중 문서 요약(MDS)을 중심으로 연구하였습니다. MDS는 긴 문서에서 정보를 추출하고 통합하는 복잡한 작업으로, 정보의 중복성, 사실 일관성, 간결함을 유지하면서도 주요 세부사항을 반영해야 합니다. 이를 위해 본 연구는 프롬프트 앙상블(prompt ensemble) 기법을 사용하여 테스트 시간에 요약 작업을 확장하는 방법을 제안하며, 두 가지 새로운 평가 지표(CAP score 및 LLM ACU score)를 도입하여 요약의 질적 평가를 개선하고자 합니다.

- **Performance Highlights**: 본 논문에서 제안한 Multi2 프레임워크는 기존 모델과 비교하여 수치적으로 요약 품질을 향상시켰습니다. 또한, 테스트 시간 스케일링에서 발생할 수 있는 과도한 계산 자원 사용에 따른 성능 저하 현상을 체계적으로 분석하였습니다. 다양한 프롬프트로 생성한 후보 요약을 집계하여의 최종 요약은 LLM의 맥락 이해를 최적화하며, 새로운 평가 메트릭스를 통해 더 신뢰할 수 있는 요약 품질 평가를 가능하게 했습니다.



### The Noisy Path from Source to Citation: Measuring How Scholars Engage with Past Research (https://arxiv.org/abs/2502.20581)
- **What's New**: 이번 연구에서는 학술 인용(citation)의 변별력을 정량화할 수 있는 계산적 파이프라인(computational pipeline)을 도입했습니다. 이를 통해 1300만 개의 인용문 문장 쌍을 분석하여 기초 과학 정보의 변화를 평가하고, 인용의 충실도(fidelity)가 저자의 H-index, 팀 크기, 저자가 최신 연구를 인용하는지 여부에 따라 어떻게 달라지는지를 밝혔습니다. 이러한 인용 충실도의 차이는 단순히 인용 수를 기반으로 한 분석의 한계를 드러냅니다.

- **Technical Details**: 제안된 계산적 파이프라인은 전체 연구 논문(full-text papers)의 텍스트를 활용하여 인용 논문(citing papers)에서 인용을 식별하고, 인용된 논문(cited papers)에서 해당 주장을 찾습니다. 그런 다음, 신뢰도를 측정하기 위해 감독 학습(supervised models)을 적용하여 문장 수준에서 충실도를 측정합니다. 이러한 방법론을 통해 문서의 텍스트를 활용하여 더욱 정확하게 인용 충실도를 분석할 수 있습니다.

- **Performance Highlights**: 연구 결과, 저자가 인용할 때 인용 논문이 최신이며 학문적으로 근접하고 접근하기 쉬운 경우 인용 충실도가 더 높은 경향을 보였습니다. 또한, 저자의 팀 크기가 중간일 때, 그리고 첫 저자의 H-index가 낮을 때 인용 충실도가 높다는 것을 발견했습니다. '전화 효과(telephone effect)'를 통해, 원래의 주장을 인용하는 문헌과 중개 문헌을 인용하는 경우 충실도가 낮아지는 경향이 있어, 정보 전이 과정의 왜곡을 드러내었습니다.



### HuAMR: A Hungarian AMR Parser and Datas (https://arxiv.org/abs/2502.20552)
- **What's New**: HuAMR는 헝가리어를 위한 최초의 Abstract Meaning Representation (AMR) 데이터셋과 대형 언어 모델 기반 AMR 파서들을 제시합니다. 이 데이터셋은 Llama-3.1-70B 모델을 활용하여 자동으로 생성된 'silver-standard' AMR 주석을 수작업으로 정제하여 제작되었습니다. 또한, 다양한 모델 아키텍처와 세밀 조정 전략이 AMR 파싱 성능에 미치는 영향을 분석합니다.

- **Technical Details**: AMR은 문장의 의미를 그래프 구조를 활용하여 표현하는 의미 프레임워크입니다. AMR 그래프는 동사, 개체 및 PropBank frameset과 같은 개념을 나타내는 노드와 이들 간의 의미적 역할 및 관계를 나타내는 엣지로 구성됩니다. 헝가리어 데이터를 위한 AMR 그래프 생성을 위해 mT5 Large 및 Llama-3.2 모델을 활용하여 효율적인 헝가리어 AMR 파서를 개발하였습니다.

- **Performance Highlights**: 훈련 데이터에 'silver-standard' AMR을 포함시키는 것이 전체 점수를 항상 높이지는 않지만, 이러한 기법이 헝가리어 뉴스 데이터를 기반으로 한 파싱 정확도를 효과적으로 향상시킨다는 결과를 보였습니다. 평가 지표로 Smatch 점수를 사용하여 HuAMR과 우리의 파서들이 의미 파싱 연구를 발전시킬 잠재력을 지니고 있음을 확인하였습니다.



### NANOGPT: A Query-Driven Large Language Model Retrieval-Augmented Generation System for Nanotechnology Research (https://arxiv.org/abs/2502.20541)
Comments:
          61 pages, 3 figures

- **What's New**: 이 논문은 나노기술 연구를 위해 설계된 대형 언어 모델 기반의 정보 검색 증강 생성 시스템(LLM-RAG)의 개발 및 응용을 제시합니다. 이 시스템은 지능형 연구 보조 도구로서의 역할을 하여 나노기술 분야의 문헌 조사를 보다 효율적이고 포괄적으로 만드는 데 기여합니다. Google Scholar의 고급 검색과 Elsevier, Springer Nature 및 ACS Publications의 오픈 액세스 논문을 활용하여 신뢰할 수 있는 여러 출처에서 데이터를 통합합니다.

- **Technical Details**: LLM-RAG 시스템은 고급 쿼리 백엔드 검색 메커니즘을 중심으로 구성되어 있으며, 이는 다수의 신뢰할 수 있는 출처로부터 데이터를 결합합니다. LLM은 대규모 훈련 데이터를 통해 인간과 유사한 텍스트를 생성하고 이해할 수 있으며, 이러한 모델은 자연어 처리(NLP) 분야에서 혁신적인 기술로 부각되고 있습니다. 본 논문에서는 LLM을 기반으로 한 RAG 기법이 LLM의 성능을 개선하지고 구조적 오류를 줄이는 방법에 대해 논의합니다.

- **Performance Highlights**: LLM-RAG 시스템은 포괄적인 문헌 리뷰에 필요한 시간과 노력을 크게 줄이는 동시에 높은 정확성과 쿼리 관련성을 유지하는 데 효과적이라는 것이 엄격한 테스트를 통해 검증되었습니다. 이 시스템은 표준 공개 LLM보다 성능이 우수하며, 나노기술 분야에서 연구를 가속화하는 데 중요한 잠재력을 보여줍니다. 다양한 연구 분야에서 새로운 나노 재료의 발견과 실험 데이터의 해석을 원활하게 하는 데 기여할 것으로 기대됩니다.



### Supervised Fine-Tuning LLMs to Behave as Pedagogical Agents in Programming Education (https://arxiv.org/abs/2502.20527)
- **What's New**: 이 논문에서는 프로그래밍 교육을 위해 설계된 맞춤형 언어 모델인 GuideLM을 소개합니다. GuideLM은 Debugging C Compiler (DCC)에 통합되어, LLMs를 활용해 교육적으로 유용한 오류 설명을 생성합니다. 이를 통해 기존의 OpenAI 모델들이 학생들에게 직접적인 해결책을 제공하는 과도한 도움을 줄이려 합니다.

- **Technical Details**: GuideLM은 528개의 학생 질문/교사 답변 쌍 데이터를 기반으로 Supervised Fine-Tuning (SFT) 과정을 통해 개발되었습니다. 두 개의 모델인 GuideLM과 GuideLM-mini가 각각 ChatGPT-4o 및 4o-mini에 맞춰 조정되었습니다. 이 모델들은 교수학습 원칙에 따른 평가를 통해 구성적 발달과 인지 부담 이론을 바탕으로 설계되었습니다.

- **Performance Highlights**: 연구 결과, GuideLM과 GuideLM-mini는 기존의 OpenAI 모델 대비 교수적 성과가 8% 향상되었고, 단어 경제성은 58% 개선되었습니다. 그러나 이러한 정제 과정은 일반 정확성에서 약간의 감소를 초래하기도 했습니다. 이에 따라, 특정 데이터셋으로 LLM을 미세 조정하는 접근 방식이 교육적 맥락에 더욱 적합한 모델 개발에 대한 가능성을 제시합니다.



### TripCraft: A Benchmark for Spatio-Temporally Fine Grained Travel Planning (https://arxiv.org/abs/2502.20508)
Comments:
          27 pages, 18 Tables and 6 Figures

- **What's New**: 최근 대형 언어 모델(LLMs)의 잠재력을 활용한 개인화된 여행 계획 에이전트로서의 가능성을 탐구하는 연구들이 진행되고 있습니다. 하지만 기존의 기준들은 실제 적용 가능성이 제한적입니다. 본 연구에서는 TripCraft라는 새로운 여행 계획 데이터셋을 소개하면서, 이를 통해 현실적인 제약 사항을 통합하여 보다 효과적인 일정 생성을 목표로 하고 있습니다.

- **Technical Details**: TripCraft는 공공 교통 일정, 이벤트 가용성, 다양한 관광 카테고리 및 사용자 성격(persona)과 같은 실세계 제약을 통합한 공간적 시간적(spatiotemporally) 일관된 여행 계획 데이터셋입니다. 또한, 기존의 이진 검증 방법을 넘어 여행 계획의 질을 평가하기 위해 다섯 가지 지속적 평가 메트릭스, 즉 Temporal Meal Score, Temporal Attraction Score, Spatial Score, Ordering Score 및 Persona Score를 제안합니다.

- **Performance Highlights**: 모델의 매개변수 설정을 통해 식사 일정 수립이 크게 향상되어 7일 시나리오에서 Temporal Meal Score가 61%에서 80%로 증가했습니다. TripCraft는 LLM 기반의 개인화된 여행 계획을 위한 새로운 기준을 제시하며, 일정 생성에 있어 더 현실적이고 제약을 고려한 프레임워크를 제공합니다. 데이터셋과 코드베이스는 수락 후 공개될 예정입니다.



### A Thousand Words or An Image: Studying the Influence of Persona Modality in Multimodal LLMs (https://arxiv.org/abs/2502.20504)
- **What's New**: 최근 대형 언어 모델(LLMs)은 다양한 페르소나(persona)를 구현하는 데 있어 큰 발전을 보여주었습니다. 이 논문은 텍스트와 이미지라는 서로 다른 표현 방식을 갖는 페르소나의 영향력을 처음으로 체계적으로 분석합니다. 새로운 모달리티 평행 데이터셋을 생성하고, LLM이 특정 페르소나의 특성과 시나리오를 얼마나 잘 표현하는지를 평가하기 위한 프레임워크를 개발하였습니다.

- **Technical Details**: 우리는 다양한 연령, 성별, 직업, 위치를 가진 4040명의 페르소나로 이루어진 데이터셋을 생성하였습니다. 각 페르소나는 이미지 전용, 텍스트 전용, 이미지와 짧은 텍스트의 조합, 타이포그래피를 활용한 이미지 네 가지 방식으로 표현됩니다. 이러한 표현방식에 대한 비교 분석을 통해 멀티모달 LLM의 성능을 평가하고, 텍스트 기반 페르소나가 이미지 표현보다 뛰어난 결과를 나타냈습니다.

- **Performance Highlights**: 실험 결과, LLM은 상세한 텍스트로 표현된 페르소나에서 더 많은 언어적 습관을 보였고, 타이포그래피 이미지는 페르소나와의 일관성이 더 높았습니다. 이러한 연구 결과는 LLM이 이미지로 전달되는 페르소나 특정 세부 사항을 종종 간과하고 있음을 밝혔습니다. 이를 통해 향후 연구가 이 격차를 좁히는 데 기여할 수 있도록 하는 기반을 마련합니다.



### Protecting multimodal large language models against misleading visualizations (https://arxiv.org/abs/2502.20503)
Comments:
          Preprint. Code and data available at this https URL

- **What's New**: 이번 논문은 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)이 왜곡된 시각화(visualizations)에서 얼마나 취약한지를 평가합니다. 연구 결과, 이러한 왜곡된 시각화는 MLLMs의 질문-응답 정확도를 무작위 기준선(random baseline) 수준으로 떨어뜨려 불확실한 결론을 도출하게 만듭니다. 이를 해결하기 위해, 연구진은 MLLMs의 성능을 향상시킬 수 있는 여섯 가지 인퍼런스 타임(inference-time) 방법을 소개했습니다.

- **Technical Details**: 연구에서는 특히 데이터 테이블을 추출한 후 텍스트 전용 대형 언어 모델을 활용하여 질문에 답하는 방법이 가장 효과적임을 알렸습니다. 연구는 총 16개의 MLLMs의 QA 정확도를 세 개의 데이터셋을 통해 비교 분석했으며, 왜곡된 시각화에서 정확도가 무작위 기준선과 유사한 수준(24.8%)으로 떨어졌음을 보여주었습니다. 왜곡된 시각화에 대한 MLLMs의 취약성은 ChartQA의 성능 향상으로 자연스럽게 해결되지 않음을 발견했습니다.

- **Performance Highlights**: 가장 효과적인 방법인 테이블 기반 QA(table-based QA)는 왜곡된 시각화에 대해 15.4%에서 19.6%의 성능 개선을 보였습니다. 시각화를 재작성하는 방법도 유망하지만, 이는 시각화가 Python 인터프리터에서 컴파일 될 수 있는 경우에만 효과적입니다. 분석 결과, 왜곡된 시각화를 정확히 해석하지 못한 MLLMs가 높은 엉터리 수치제시로 인해 정보의 비정확성을 초래할 수 있음을 확인하였습니다.



### Explainable AI for Clinical Outcome Prediction: A Survey of Clinician Perceptions and Preferences (https://arxiv.org/abs/2502.20478)
- **What's New**: 이 연구에서는 AI 예측을 해석하는 데 있어 다양한 Explainable AI (XAI) 기법에 대한 임상의 선호도를 이해하기 위한 설문 조사 연구를 수행하였습니다. LIME, Attention 기반 하이라이트, 유사 환자 검색, LLM(대형언어모델)으로 생성된 자유 텍스트 논거 등 4가지 XAI 기법을 구현하여 결과 예측 모델을 평가했습니다. 이번 연구는 의료 AI 도구를 사용하는 임상의관들이 어떤 XAI 기법을 선호하는지를 살펴보는 데 중점을 두었습니다. 연구 결과를 통해 각 기법의 적합성 및 잠재적 진화 방안도 제시합니다.

- **Technical Details**: 연구에 사용된 데이터는 MIMIC-III에서 수집된 비식별 건강 기록으로, 2001년부터 2012년 사이에 ICU에 입원한 환자 46,520명의 기록입니다. 환자의 입원 노트를 사용하여 입원 중 사망 가능성을 예측하는 조기 발견 사망 예측 작업에 초점을 맞췄습니다. 실험에서는 LIME, Attention 기반 설명, 유사 환자 검색 및 자유 텍스트 논거 생성 기법을 적용하여 예측 결과를 설명하였습니다. UmlsBERT 모델을 사용하여 87.86 micro-F1 및 66.43 macro-F1 성능을 확인했습니다.

- **Performance Highlights**: 임상의들은 XAI 기법 중에서 자유 텍스트 논거를 가장 선호하며, 이는 의료 제공자와 환자 간의 소통을 향상시키는 데 기여할 수 있음을 보여줍니다. 또한, 유사 환자 검색 기법은 임상 사용자와 AI 시스템 간의 신뢰 구축에 기여하는 중대한 가능성을 가지고 있다는 점을 강조하였습니다. 연구 결과는공통적인 임상 의사 결정 과정에 대한 통찰력을 제공하며, 다양한 전문성 수준에 맞춘 XAI 툴 개발의 필요성을 잘 나타냅니다. 이 연구가 추후 XAI 기법의 설계 및 구현 방향에 영향을 미칠 것으로 예상됩니다.



### Promote, Suppress, Iterate: How Language Models Answer One-to-Many Factual Queries (https://arxiv.org/abs/2502.20475)
- **What's New**: 이번 연구에서는 언어 모델(LM)이 다중 사실 질의(one-to-many factual queries)에 응답하는 기제를 분석합니다. 구체적으로, 모델은 모든 대답을 회상한 뒤, 이전에 생성된 대답을 억제하는 promote-then-suppress 메커니즘을 사용합니다. 이 연구는 모델이 어떻게 주제와 이전의 답변 토큰을 활용하여 지식을 회상하고, 반복을 피하는지를 탐구합니다.

- **Technical Details**: 모델은 attention과 다층 퍼셉트론(MLP)을 사용하여 주제 정보와 이전의 답변을 처리합니다. 중간 계층에서는 주제 정보를 복사하고, MLP는 가능한 모든 답변을 증진시키며, 후반 계층에서는 이전 답변을 억제합니다. Token Lens 기법과 knockout 방법을 통해 모델의 작동 기제를 분석하고, 특정 토큰들이 어텐션과 MLP 계층에서 어떻게 사용되는지를 연구합니다.

- **Performance Highlights**: 이 연구는 Llama-3-8B-Instruct 및 Mistral-7B-Instruct 모델이 다층 네트워크를 통해 어떻게 지식 회상 및 반복 방지를 수행하는지를 다양한 데이터 세트를 통해 증명합니다. 모델은 주제와 이전 답변 토큰들을 사용하여 명확한 답변을 생성하는 데 필요한 모든 정보를 집계합니다. 이 연구는 복잡한 사실 회상을 위한 LMs의 동적 정보 통합 방식을 밝혀내며, 향후 연구에 중요한 기초 자료를 제공합니다.



### Shades of Zero: Distinguishing Impossibility from Inconceivability (https://arxiv.org/abs/2502.20469)
- **What's New**: 이 연구는 불가능(impossibility)과 비간섭(inconceivability) 사이의 구별을 사람들이 어떻게 유지하는지에 대한 새로운 통찰을 제공합니다. 기존의 연구들은 불가능과 가능성이 낮은 것 사이의 차이를 놓고 대부분 다룬 반면, 비간섭성에 대한 실증적 연구는 드물었습니다. 실험을 통해 사람들은 불가능한 사건과 비간섭적인 사건 사이의 차이를 쉽게 인식할 수 있음을 보여주었습니다.

- **Technical Details**: 연구에서는 카테고리화(categorization) 실험을 통해 사람들이 불가능과 비간섭 이벤트를 구별하는 능력을 평가했습니다. 참여자들은 이벤트 설명을 확률, 불가능, 비간섭 카테고리로 나누는 과정에서 높은 일관성을 보였습니다. 또한, 통계적 언어 모델(statistical language models, LMs)의 확률이 이러한 구별을 지원하는지에 대한 실험도 진행되었습니다.

- **Performance Highlights**: 사람들은 불가능한 사건과 비간섭적인 사건을 쉽게 구별할 수 있으며, 이는 언어 모델에서도 마찬가지입니다. 언어 모델의 확률 추정치는 사람들의 사건 가능성 평가와 크게 일치함을 보여주었습니다. 이러한 결과는 비간섭적인 사건에 대한 구별이 통계적 학습을 통해 학습될 수 있음을 시사합니다.



### Among Them: A game-based framework for assessing persuasion capabilities of LLMs (https://arxiv.org/abs/2502.20426)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 설득 능력을 평가하기 위한 독창적인 게임 프레임워크인 'Among Them'을 제안합니다. 이 프레임워크는 25가지 사회 심리학 및 수사학적 설득 전략을 기반으로 LLM의 조작 기술을 정량화 및 정성화할 수 있게 합니다. 640개의 게임 실험을 통해, 다양한 유형과 크기의 8가지 LLM이 조사되었으며, 모든 모델이 25가지 기법 중 22가지 기법을 성공적으로 활용할 수 있다는 것을 보여주었습니다.

- **Technical Details**: 논문에서는 LLM 에이전트를 위한 게임 환경, LLM 에이전트 모듈, 설득 평가 모듈, 평가 대시보드의 네 가지 주요 모듈을 포함하는 'Among Them' 프레임워크를 도입합니다. 참가자들은 크루원과 사기꾼의 두 역할로 나뉘어, 사기꾼을 식별하고 임무를 수행하는 사회적 추측 게임을 진행합니다. 각 게임은 행동, 토론, 투표의 3단계로 진행되며, 게임 내 상태 정보, 플레이어 위치, 생존 상태, 행동 기록 등이 자세히 추적됩니다. 또한, OpenRouter API를 통해 다양한 모델의 선택이 가능하게 구현되었습니다.

- **Performance Highlights**: 실험 결과, 모든 LLM 모델은 반복적인 게임 진행에서 프레임워크 내에서 성공적으로 설득 기술을 사용할 수 있었습니다. 흥미롭게도, 더 큰 모델이 더 작은 모델에 비해 설득 면에서 유리하지 않으며, 응답 길이가 길어질수록 승리한 게임 수와는 부정적인 상관관계가 발견되었습니다. 이는 LLM의 조작 가능성을 깊이 이해하는 데 기여하며, 향후 AI의 조작 능력에 대한 연구를 촉진할 자료를 제공합니다.



### SEKI: Self-Evolution and Knowledge Inspiration based Neural Architecture Search via Large Language Models (https://arxiv.org/abs/2502.20422)
- **What's New**: 이번 논문에서는 SEKI라는 새로운 대형 언어 모델(LLM) 기반의 신경망 아키텍처 검색(NAS) 방법을 소개합니다. SEKI는 현대 LLM의 체인 오브 사고(Chain-of-Thought, CoT) 패러다임에서 영감을 받아 두 가지 주요 단계인 자기 진화(self-evolution)와 지식 증류(knowledge distillation)로 작동합니다. 이 방법은 기존 아키텍처에 의존하지 않고도 효율성과 성능을 향상시킬 수 있는 잠재력을 지니고 있습니다.

- **Technical Details**: SEKI의 첫 번째 단계인 자기 진화에서는 초기 LLM이 참조 예제가 부족하여 성능 피드백을 기반으로 아키텍처를 반복적으로 개선합니다. 이후 두 번째 단계인 지식 증류에서는 고성능 아키텍처의 공통 패턴을 분석하여 새로운 최적화된 설계를 생성합니다. 이 과정을 통해 SEKI는 역량을 극대화하며, 단 0.05 GPU-days의 비용으로 SOTA 성능을 달성합니다.

- **Performance Highlights**: SEKI는 CIFAR-10에서 97.71%, CIFAR-100에서 84.14%, 이미지넷에서 75.8%의 최고 정확도를 기록하였으며, 특히 이미지넷에 대한 직접 검색에서 76.1%의 top-1 정확도를 달성하여 기존의 SOTA 방법들을 능가합니다. SEKI는 여러 작업에서 SOTA 경합 결과를 보이며 강력한 일반화 능력을 보여줍니다. 이처럼 SEKI는 NAS 분야에서 LLM의 성공과 가능성을 제시하고 있습니다.



### Chitranuvad: Adapting Multi-Lingual LLMs for Multimodal Translation (https://arxiv.org/abs/2502.20420)
- **What's New**: 이번 연구에서는 WAT2024의 영문에서 저해상도 다중 모달 번역 과제를 위한 제출물로 Chitranuvad라는 다중 모달 모델을 소개합니다. 이 모델은 다국어 LLM과 비전 모듈을 효과적으로 통합하여 다중 모달 번역을 수행합니다. ViT 이미지 인코더를 사용하여 시각적 표현을 추출하고, 이를 LLM의 공간으로 변환하여 자가 회귀 방식으로 번역을 생성합니다.

- **Technical Details**: Chitranuvad는 다국어 번역을 위해 적응된 대형 다중 모달 모델입니다. 이 모델은 Krutrim LLM을 백본으로 사용하고, 특정 작업에 맞게 시각적 이미지 인코더와 함께 멀티모달 학습을 진행합니다. 모델 훈련 과정에서 이미지를 비전 인코더로 인코딩한 후, 모달리티 프로젝션 레이어를 통해 LLM 임베딩 공간으로 시각적 토큰을 나타냅니다.

- **Performance Highlights**: 모델은 Hindi, Bengali, Malayalam 언어의 3가지 트랙에 참가했으며, 모든 트랙에서 Hindi에 대해 SOTA 성과를 달성했습니다. Chitranuvad는 비주얼 진화 번역 데이터 세트에서 세부 조정 작업의 효율성을 입증하였고, 자가 회귀 방식으로 올바른 번역을 생성하는 능력을 보여주었습니다.



### Pause-Tuning for Long-Context Comprehension: A Lightweight Approach to LLM Attention Recalibration (https://arxiv.org/abs/2502.20405)
- **What's New**: 본 연구에서는 LLMs가 긴 문맥(compact context)에서 정보를 처리하는 데 어려움을 겪는 Lost-in-the-Middle (LITM) 문제를 해결하기 위해 새로운 접근 방법을 제안합니다. 이것은 pause-tuning이라는 기법으로, 입력 데이터에서 인위적으로 삽입된 pause tokens를 사용하여 주의(attention)를 재분배하는 것입니다. 이러한 방식으로 긴 입력을 더 작은 청크로 나누어 각 부분을 보다 효과적으로 처리할 수 있도록 합니다.

- **Technical Details**: Pause tokens는 입력 시퀀스에 전략적으로 삽입되어 모델의 주의 분포를 조정합니다. 연구에서는 이 pause tokens의 삽입 방법을 여러 가지로 실험하고, 15가지 문맥 깊이(context depth)와 3회의 단일 needle 테스트를 포함하여 최적의 기술을 찾기 위해 노력했습니다. 각 테스트에서 모델의 성능을 비교하고 분석하여 주의의 재배치가 긴 문맥 처리에 미치는 영향을 평가하였습니다.

- **Performance Highlights**: 실험 결과, LLaMA 3.2 3B Instruct 모델과 LLaMA 3.1 8B Instruct 모델이 각각 10.61%와 3.57%의 성능 향상을 보였습니다. 또한 pause-tuning 기법이 기존의 다른 기술보다 긴 문맥 유지(long-context retention) 및 처리 개선에 보다 효과적임을 보여주었습니다. 이 연구는 LLMs에서의 긴 문맥 처리에서 pause tokens의 효과성을 강조하며, 향후 이러한 기법의 활용 가능성을 제시합니다.



### Momentum Posterior Regularization for Multi-hop Dense Retrieva (https://arxiv.org/abs/2502.20399)
Comments:
          Accepted by COLING 2025

- **What's New**: 이 논문은 Multi-hop 질문 응답 시스템의 효과적인 지식 증류(knowledge distillation) 방법인 MoPo(Momentum Posterior Regularization)를 제시합니다. 기존의 지식 증류 방법들은 Multi-hop QA에 적합하지 않았으나, 본 연구에서는 새로운 접근 방식을 도입하여 이전 조건에서의 응답을 활용할 수 있는 모델을 개발했습니다. 특히, 이 연구는 새로운 데이터셋인 PostSumQA를 소개하여 Multi-hop 질문에 대한 포스터리어 요약(postorior summary) 정보를 제공하고 있어 향후 연구에 기여할 전망입니다.

- **Technical Details**: MoPo는 두 가지 핵심 혁신을 기반으로 합니다. 첫째, 포스터리어 정보는 이전 단계와 현재 단계의 골드 지식에서 쿼리 포커스 요약(query-focus summary)으로 정의됩니다. 둘째, MoPo는 모멘텀 이동 평균(momentum moving average) 방법을 사용하여 포스터리어 검색이 우선 검색과 함께 업데이트되도록 하여 smoother한 훈련을 가능하게 합니다. 이 두 가지 접근 방식은 지식 간 큰 갭을 줄이고 효과적인 지식 증류를 가능하게 합니다.

- **Performance Highlights**: HotpotQA와 StrategyQA에서의 실험 결과는 MoPo가 기존 방법들보다 우수한 성능을 내는 것을 보여줍니다. MoPo를 사용한 간단한 파이프라인은 기존의 Multi-hop reranking 및 LLM을 기반으로 한 방법보다 더 뛰어난 다운스트림 성능을 기록했습니다. 전체적으로 MoPo는 이전 Multi-hop 검색 방법들과 포스터리어 정규화 방법들보다 향상된 인출 성능을 보여줍니다.



### Transforming Tuberculosis Care: Optimizing Large Language Models For Enhanced Clinician-Patient Communication (https://arxiv.org/abs/2502.21236)
Comments:
          GenAI4Health at AAAI-25

- **What's New**: 이번 연구는 결핵(Tuberculosis, TB) 치료를 위한 새로운 디지털 적응 기술(Digital Adherence Technologies, DATs) 개발을 다룹니다. 저소득 및 중간소득 국가에서의 의료 접근 문제를 해결하기 위해, 대형 언어 모델(Large Language Model, LLM)을 활용하여 치료 지지자와 환자의 상호작용을 증진시키는 것을 목표로 하고 있습니다. 이 기술은 '인간-현장' 시스템 내에서 AI를 구동하여 환자 참여도를 높이고 TB 치료 결과를 개선하려는 시도로, 전체적인 치료 경험을 향상시킬 수 있는 잠재력을 지니고 있습니다.

- **Technical Details**: 연구에서는 인간 감독 아래에서 TB 치료를 지원하는 LLM 기반의 대화형 모델을 개발하고, 이를 통해 문맥적 학습 기법을 활용하여 환자의 요구에 맞는 지원 도구를 만드는 것을 목표로 합니다. 모델 설계는 환자와 치료 지원자 간의 대화를 시뮬레이션하는 몇 가지 다이얼로그 샘플을 기반으로 하며, 다양한 텍스트 정화(dwifferentally private text sanitization) 방법을 통해 환자의 개인 정보를 보호합니다. 연구진은 또한 TB 관련 자료와 신뢰할 수 있는 출처의 자원을 활용하여 사실적이고 정확한 응답을 보장하기 위한 RAG(검색 보강 생성) 파이프라인을 구현했습니다.

- **Performance Highlights**: 개발된 LLM 모델은 스페인어를 사용하는 결핵 환자를 지원하기 위한 것이며, 문화적 및 언어적 적합성을 고려한 대화 응답을 모델링하는 데 성공했습니다. 환자와 치료 지원자 간의 대화를 재현하여 공감적 응답을 유도하도록 교육받았으며, 이를 통해 치료 지침에 따라 개인 맞춤형 지원을 제공할 수 있습니다. 이 연구는 특히 비영어권 환자에게 유용할 것으로 기대되며, 다국적 의료 환경에서의 LLM 적용에 대한 통찰을 제공합니다.



### Optimizing Large Language Models for ESG Activity Detection in Financial Texts (https://arxiv.org/abs/2502.21112)
- **What's New**: 이 논문은 ESG(환경, 사회, 지배구조) 요인이 기업 의사결정에 통합되는 과정에서 AI 모델을 활용하여 지속 가능성 보고서와 비재무 공시를 자동으로 평가하는 새로운 접근 방식을 제안합니다. 특히, ESG-Activities라는 새로운 벤치마크 데이터셋을 도입하여 이를 통해 LLMs(대형 언어 모델)의 성능을 강화하는 방법에 대해 설명합니다.

- **Technical Details**: 현재 세대 LLMs의 텍스트 분류 능력을 평가하면서, ESG-Activities 데이터셋을 이용해 모델을 세부 환경 활동에 맞춰 학습시키는 방법을 채택합니다. 이 데이터셋은 전문가가 수작업으로 선별한 데이터와 LLMs에 의해 생성된 합성 데이터를 포함하고 있어 모델 성능을 높이는 데 기여합니다.

- **Performance Highlights**: 실험 결과, ESG-Activities로 파인튜닝된 모델이 기존의 제로샷 학습과 수작업으로 주석이 붙은 데이터만으로 학습한 모델에 비해 분류 정확도가 크게 향상된 것을 확인하였습니다. Llama 7B와 같은 오픈 소스 모델이 특정 설정에서 대형 프로프라이어터리 모델보다 우수한 성능을 발휘했으며, 이는 AI 연구자와 금융 분석가들에게 중요한 영향을 미칠 것입니다.



### Re-evaluating Theory of Mind evaluation in large language models (https://arxiv.org/abs/2502.21098)
Comments:
          under review

- **What's New**: 이 논문은 대규모 언어 모델(LLMs)의 마음 이론(Theory of Mind, ToM) 능력에 대한 새로운 평가 방법을 제안합니다. ToM은 다른 사람의 정신 상태를 추론하는 능력이고, 이러한 능력이 LLMs에 얼마나 적용되는지에 대한 논란이 있으며 기존의 평가 방법에 의문을 제기합니다. 저자들은 LLMs가 인간의 행동을 일치시키는 것뿐만 아니라, 이러한 행동의 근본적인 계산을 이해하도록 기대해야 하는지 명확하지 않다고 주장합니다.

- **Technical Details**: 저자들은 ToM 평가에서 두 가지 주요 이슈를 강조합니다. 첫 번째는 ToM의 정의에 대한 불일치입니다. ToM을 가진다는 것의 의미는 사람들의 행동에 맞추는 것(behavior-matching)일 수도 있고, 이러한 행동을 가능하게 하는 정신적 알고리즘(computation matching)일 수도 있습니다. 두 번째는 평가의 타당성으로서, 현재의 ToM 평가가 심리적 구성요소를 측정하지 못할 수 있으며, 적대적인 예시(adversarial examples)에 대한 평가가 순수한 ToM을 벗어난 일반적 사고 능력으로 평가를 변화시킬 수 있음을 지적합니다.

- **Performance Highlights**: 현재까지 LLMs의 ToM 능력에 대한 연구는 상반된 결과를 보였습니다. 일부 연구자들은 LLM이 인간 수준의 ToM 평가 과제를 성공적으로 수행한다고 주장하는 반면, 다른 연구자들은 LLM이 미세한 변수나 적대적 변화에 민감하다고 주장합니다. 새로운 모델과 평가 기준이 빠르게 발전하고 있지만, LLM의 ToM 능력을 평가하는 데 필요한 명확한 기준이 부족하여 혼란스러움이 지속되고 있습니다.



### Extending Dense Passage Retrieval with Temporal Information (https://arxiv.org/abs/2502.21024)
- **What's New**: 이 연구에서는 전통적인 정보 검색 방법인 BM25 및 Dense Passage Retrieval(DPR)가 시간 민감성 쿼리를 처리하는 데 한계가 있음을 지적하고, 이를 극복하기 위한 새로운 Temporal Retrieval Model을 소개합니다. 이 모델은 쿼리 타임스탬프와 문서 날짜를 통합하여 명시적인 시간 신호를 검색 시스템에 결합합니다. 이를 통해 검색된 내용이 주제적으로 관련 있을 뿐만 아니라 사용자 의도와 시간적으로 일치하도록 합니다.

- **Technical Details**: Temporal Dense Passage Retrieval(TempDPR)이라는 모델은 세 가지 혁신적인 접근 방식을 도입합니다. 첫째, DateAsTag 접근 방식은 타임스탬프를 [S-DATE] 및 [E-DATE]처럼 특수 토큰으로 감싸 명시적으로 시간 정보를 표시합니다. 둘째, DateAsToken 접근법은 쿼리와 문서에 타임스탬프를 일반 텍스트로 추가하고, 세 번째로는 TempDPR이 쿼리 및 문서 표현에 시간 임베딩을 직접 통합하여 다양한 융합 기술을 탐색합니다.

- **Performance Highlights**: 이 연구는 ArchivalQA와 ChroniclingAmericaQA라는 대규모 벤치마크 데이터 세트에서 성능을 평가하여 표준 검색 기준선 대비 상당한 성능 향상을 달성했습니다. 특히, ArchivalQA에서 Top-1 검색 정확도가 6.63% 향상되었고, ChroniclingAmericaQA에서는 9.56% 향상됨을 보여줍니다. 이러한 결과는 시간 모델링을 통한 검색 시스템의 중요성과 시간적으로 기반한 쿼리를 처리하는 새로운 기준을 제시하고 있습니다.



### Merging Clinical Knowledge into Large Language Models for Medical Research and Applications: A Survey (https://arxiv.org/abs/2502.20988)
- **What's New**: 이 논문은 의학적 인공지능(medical AI) 시스템을 구축하는 데 있어 아카데미와 산업의 차이를 포괄적으로 비교하는 자료가 부족하다는 문제를 지적하고 있습니다. 최근 대형 언어 모델(large language models, LLMs)의 발전에 따라 신속하고 정확한 진단을 지원하기 위한 여러 연구가 진행되고 있지만, 관련 지식과 데이터의 통합이 여전히 도전과제로 남아 있습니다. 이 연구는 의료 데이터, 학습 프로세스, 의료 지식 그래프(knowledge graphs) 통합 및 평가 시스템 등 다양한 측면에서 의료 AI 시스템 구축 패러다임을 살펴봅니다.

- **Technical Details**: 임상 데이터베이스 및 데이터셋의 소개를 통해 의료 LLM 구축을 위한 기초를 다지고, 의사결정 지원을 위해 직접 학습(direct learning)과 지식 추출(knowledge extraction) 두 가지 연구 방향을 제시하고 있습니다. LLM들은 방대한 의료 데이터에서 패턴을 발견하고, 이를 통해 진단 및 치료를 위한 개인 맞춤형 치료 계획을 제공하는 데 중점을 둡니다. 특히 의료 데이터의 품질과 최신성을 유지하는 것이 중요한데, 이는 모델의 예측 정확도를 높이는 데 기여합니다.

- **Performance Highlights**: 대형 언어 모델들은 전자 의료 기록(electronic medical records, EMRs), 의료 영상 및 유전 정보와 같은 대량의 데이터를 활용하여 진단의 정확도를 높이는 데 중요한 역할을 합니다. 커리큘럼 중심의 데이터 세트가 지속적으로 업데이트되고 있으며, 이를 통해 의료 AI 시스템은 임상 요구에 더 잘 적응하고 있습니다. 논문은 결국 아카데미와 임상 실무 간의 조화로운 발전을 통해 의료 AI의 향후 연구 방향을 제시하고, 실제 의료 서비스 개선을 위한 실질적인 가이드를 제공합니다.



### Everything, Everywhere, All at Once: Is Mechanistic Interpretability Identifiable? (https://arxiv.org/abs/2502.20914)
- **What's New**: 본 연구는 AI 시스템의 해석 가능성(interpretability)을 보장하는 기준으로서 기계적 해석 가능성(Mechanistic Interpretability, MI)의 역할을 조사합니다. 전통적인 통계에서의 식별성(identifiability) 문제를 AI 해석 가능성에 적용하여, 특정 행동에 대한 독특한 설명이 존재하는지를 탐색합니다. 이 연구는 두 가지 주요 MI 전략을 통해 다양한 설명 기법을 제시하고 검토합니다.

- **Technical Details**: MI는 복잡한 신경망의 행동을 단순한 알고리즘으로 역설명하고, 이를 통해 내부 계산을 추적하는 접근 방식을 취합니다. 본 논문에서는 '어디-그 다음 무엇(where-then-what)'과 '무엇-그 다음 어디(what-then-where)'의 두 가지 전략을 정의하며, 이를 통해 부울 함수와 다층 퍼셉트론(Multi-layer Perceptrons)에서 다양한 후보 설명을 평가합니다. 각각의 전략에서 최적의 서브셋과 causally aligned 알고리즘을 찾아내는 과정을 진행합니다.

- **Performance Highlights**: 실험 결과, 기계적 해석 가능성의 기준이 충분히 엄격하지 않음을 보여줍니다. 여러 회로가 동일한 모델 행동을 복제할 수 있으며, 하나의 회로에 대해 다수의 해석이 가능함을 발견했습니다. 최종적으로 이 연구는 AI의 설명 기준을 정의하는 데 기여하며, 개별 설명의 독창성이 이해에 필수적인지를 재조명합니다.



### A Pilot Empirical Study on When and How to Use Knowledge Graphs as Retrieval Augmented Generation (https://arxiv.org/abs/2502.20854)
Comments:
          8 pages, 2 figures, 14 tables

- **What's New**: 이 논문은 Knowledge Graphs (KGs)와 Retrieval Augmented Generation (RAG) 프레임워크의 통합에 대한 체계적인 이해 부족을 파악하고 이를 해결하기 위한 방법론을 제공하고자 합니다. 구체적으로, KG-RAG의 성능을 다양한 기술 구성으로 분석하여 KG-RAG를 사용해야 하는 조건과 방법을 정립하고 있습니다. 이 연구는 7개의 데이터 세트에 걸쳐 6개의 KG-RAG 방법을 재구현하고 평가하여, KG-RAG 구성 요소의 최적화와 적절한 적용 조건의 중요성을 강조합니다.

- **Technical Details**: KG-RAG는 LLM (Large Language Model)의 정보 검색 능력을 개선하기 위한 새로운 패러다임으로, 엔티티 간의 의미적 관계를 활용하여 더욱 정교한 추론 능력을 제공합니다. 본 연구에서는 KG-RAG의 적용 가능성을 살펴보기 위해 과제를 분야와 난이도에 따라 분류하고, KG 질의 증진 전략, 다양한 검색 형태, 후속 프롬프트 방법 등 9개의 KG-RAG 구성을 분석했습니다. 이 연구는 KG-RAG의 효과적인 진행을 위한 실용적인 가이드라인을 제시하고 있습니다.

- **Performance Highlights**: KG-RAG의 성능 분석 결과는 개방형 LLM이 특정 작업 도메인과 난이도에서 큰 이점을 제공함을 나타냅니다. 더불어, 다양한 KG-RAG 구성들이 성능에 미치는 영향을 분석하고, KG 품질과 LLM의 능력이 KG-RAG의 효과성에 미치는 중요성을 강조하였습니다. 최종적으로, 현재 KG-RAG 연구의 몇 가지 제한점도 제시하며 향후 연구 방향을 제시하고 있습니다.



### HAIC: Improving Human Action Understanding and Generation with Better Captions for Multi-modal Large Language Models (https://arxiv.org/abs/2502.20811)
- **What's New**: 이번 연구에서는 비디오 이해의 한계를 개선하기 위해 새로운 데이터 주석 파이프라인을 소개합니다. 이 파이프라인은 명확한 인간 행동을 포함하는 비디오를 인터넷에서 수집하는 두 단계로 구성되어 있습니다. 또한, 두 개의 데이터셋 HAICTrain과 HAICBench를 정교하게 관리하여 고품질 비디오-캡션 쌍을 제공합니다.

- **Technical Details**: 제안된 데이터 주석 파이프라인은 첫 번째 단계에서 비디오를 자동으로 수집하고, 두 번째 단계에서 개별 행동과 상호작용을 상세하게 주석 처리하는 표준화된 캡션 형식을 정의합니다. HAICTrain은 126,000개의 비디오-캡션 쌍을 포함하고 있으며, HAICBench는 500개의 수작업 비디오-캡션 쌍과 1,400개의 QA 쌍으로 구성되어 있어 전반적인 평가를 가능하게 합니다.

- **Performance Highlights**: 실험 결과, HAICTrain으로 훈련한 모델은 인간 행동 이해 능력을 1.4%에서 2.1%까지 개선했습니다. 또한, HAICTrain을 활용하여 텍스트-비디오 생성 성능도 상당히 향상되었습니다. 이 연구는 MLLMs의 인간 행동 이해 성능 향상에 중요한 기여를 제공하고 있습니다.



### MedHallTune: An Instruction-Tuning Benchmark for Mitigating Medical Hallucination in Vision-Language Models (https://arxiv.org/abs/2502.20780)
- **What's New**: 이번 연구에서는 의료 분야의 Vision-Language Models (VLMs)에서 발생하는 황홀경(hallucination) 문제를 다루기 위해 MedHallTune이라는 대규모 벤치마크를 제안합니다. MedHallTune은 100,000개 이상의 이미지와 1,000,000쌍의 지침으로 구성되어 있으며, 각 데이터는 진실(annotation)로 주석이 달려 있습니다. 이 데이터셋은 의료 VLMs의 성능 평가 및 개선을 목적으로 설계되었습니다.

- **Technical Details**: MedHallTune은 100,000개 이상의 이미지-텍스트 쌍을 샘플링하고, GPT-4o를 활용해 황홀경 및 비황홀경 예제를 포함한 지침 데이터를 생성합니다. 각 데이터 쌍은 두 번의 검증 과정을 통해 생성되어, 잘못된 해석을 필터링합니다. 연구에서는 기존 VLM을 미세 조정(fine-tuning)하여 의료 맥락에서 황홀경의 대응 능력을 향상시키는 방법을 제시합니다.

- **Performance Highlights**: 실험 결과, MedHallTune을 활용한 미세 조정이 여러 의료 및 일반 VLM 모델의 황홀경 관리 능력을 크게 향상시키는 것으로 나타났습니다. 고급 평가 메트릭을 통해, 임상 정확성(clinical accuracy), 관련성(clinical relevance), 정보의 상세 수준(detail level), 위험 수준(risk level) 등에서 성능이 크게 개선되었습니다. 이를 통해, 더욱 신뢰할 수 있는 의료 VLM 개발에 기여하고 있습니다.



### FlexPrefill: A Context-Aware Sparse Attention Mechanism for Efficient Long-Sequence Inferenc (https://arxiv.org/abs/2502.20766)
Comments:
          Accepted at ICLR 2025 (Oral)

- **What's New**: 본 논문에서는 FlexPrefill이라는 유연한 스프레드 프리필 메커니즘을 제안합니다. 이 기법은 스프레드 어텐션 패턴과 계산 예산을 실시간으로 조정하여 입력의 특정 요구사항 및 어텐션 헤드에 맞춰 조절할 수 있습니다. 기존의 고정된 스프레드 어텐션 방식의 한계를 극복하고, 다양한 입력 요구사항에 더욱 효율적으로 대응할 수 있는 방법을 제공합니다.

- **Technical Details**: FlexPrefill의 핵심 요소는 두 가지입니다: 첫째, Query-Aware Sparse Pattern Determination입니다. 이 방법은 쿼리-특정 다양한 어텐션 패턴과 사전 정의된 패턴 간의 전환을 가능하게 해주는 Jensen-Shannon divergence를 측정합니다. 둘째, Cumulative-Attention Based Index Selection으로, 입력에 따라 계산할 쿼리-키 인덱스를 동적으로 선택하여, 어텐션 점수의 합이 사전 정의된 임계값을 충족하도록 보장합니다.

- **Performance Highlights**: 다양한 장기 맥락 벤치마크에서 FlexPrefill을 통해 이전 방법들에 비해 속도와 정확도에서 유의미한 개선이 있음을 보여주었습니다. 실험 결과, FlexPrefill은 여러 맥락 길이와 작업에서 모델 성능을 유지하거나 심지어 향상시키는 데 성공했습니다. 이로 인해 LLM(대형 언어 모델) 추론에 더욱 유연하고 효율적인 솔루션을 제공할 수 있음을 입증했습니다.



### Collective Reasoning Among LLMs A Framework for Answer Validation Without Ground Truth (https://arxiv.org/abs/2502.20758)
Comments:
          14 pages, 2 figures. arXiv admin note: substantial text overlap with arXiv:2411.16797

- **What's New**: 이번 연구는 여러 개의 대규모 언어 모델(GPT-4-0125-preview, Meta-LLaMA-3-70B-Instruct, Claude-3-Opus, Gemini-1.5-Flash)이 협력하여 복잡한 박사 수준의 확률 문제를 해결하는 새로운 프레임워크를 제시합니다. 이러한 협력이 응답의 신뢰성을 높이고, 생성된 질문의 품질을 평가하는 데도 도움을 줍니다. 연구진들은 카이제곱 테스트(chi-square tests), 플라이스 카파(Fleiss' Kappa), 신뢰구간(confidence interval) 분석과 같은 통계적 방법을 이용하여 여러 모델 간의 일치성과 응답의 정확성을 측정합니다.

- **Technical Details**: 연구에서는 Claude와 Gemini 모델이 잘 구조화되고 모호성이 적은 질문을 생성하는 경향이 있어, 이로 인해 높은 상호 모델 일치를 보였습니다. 반대로, LLaMA 모델은 질문 제시에서 변동성이 크고 신뢰성이 낮아 넓은 신뢰구간과 낮은 합의율로 나타났습니다. 이러한 결과는 다중 모델 협력이 응답의 신뢰성을 향상시킬 뿐 아니라 명확한 정답이 없는 상황에서도 질문 품질을 평가하고 개선하는 유용한 프레임워크를 제공함을 시사합니다.

- **Performance Highlights**: 본 연구는 다중 모델 협력을 통해 복잡한 지식을 검증하는 새로운 프레임워크를 제시하며, 협력 검증의 효과를 뒷받침하는 실증적 증거를 제공합니다. 또한 향후 LLM 기반 지식 검증 연구를 위한 기준도 마련하였습니다. 연구의 발견은 교육 기술, 자동화 평가 시스템 및 전문 학술 분야의 연구 검증에 걸쳐 광범위한 영향을 미칠 것으로 기대됩니다.



### Acquiring Grounded Representations of Words with Situated Interactive Instruction (https://arxiv.org/abs/2502.20754)
- **What's New**: 이번 연구에서는 혼합 주도(mixed-initiative)와 인간 강사와의 상황적 상호작용을 통해 단어의 기초가 되는 표현을 습득하는 접근 방식을 제시합니다. 이 연구는 퍼셉트럴(perceptual), 의미론적(semantic), 절차적(procedural) 지식을 포함한 다양한 지식을 습득하는 데 중점을 두며, 학습된 의미의 기초를 배우는 것과 함께 에이전트가 미지의 개념에 대한 지침을 요구하여 학습을 효율적으로 할 수 있게 합니다. 이 접근 방식은 Soar에서 구현되었으며, 소형 물체를 조작할 수 있는 테이블탑 로봇 팔을 통해 평가되었습니다.

- **Technical Details**: 우리의 에이전트는 로봇 팔, Kinect 센서, 사전에 정의된 네 개의 위치(스토브, 식기세척기, 쓰레기장, 팬트리)를 갖춘 간단한 테이블탑 환경에서 존재합니다. 인식 시스템은 오버헤드 Kinect 카메라에서 제공하는 컬러 3D 포인트 클라우드 데이터를 사용하여 장면을 객체로 분할합니다. 각 인식 속성에 대해 특징을 추출하고 K-Nearest Neighbor (KNN) 분류기를 통해 분류하여 에이전트가 이전에 학습한 인식 기호(perceptual symbol)를 생성합니다. 이 기호와 함께 객체의 위치 및 경계 상자 정보가 제공되어, 에이전트는 이를 통해 상징적 표현(symbolic representation)을 형성합니다.

- **Performance Highlights**: 우리는 에이전트가 인간과의 상호작용을 통해 단어의 기초가 되는 표현을 효과적으로 학습하고 있음을 보여주었습니다. 에이전트는 주어진 과제 수행 중에 인간 강사와 실시간으로 상호작용하여 학습을 하고 있으며, 반응 속도는 2초 이내입니다. 학습 과정에서 에이전트는 예제의 수를 줄이며, 학습해야 할 개념들이나 미비한 지식을 스스로 요청하는 능력을 발휘합니다. 결과적으로, 혼합 주도 상호작용(mixed initiative interaction)이 효율적이고 유연한 지식 습득 방법임을 입증하였습니다.



### Structured Preference Optimization for Vision-Language Long-Horizon Task Planning (https://arxiv.org/abs/2502.20742)
Comments:
          18 pages

- **What's New**: 기존의 비전-언어 (vision-language) 태스크 플래닝 방법론은 짧은 수평 작업에서는 뛰어난 성능을 보이지만, 복잡하고 긴 수평 작업에서는 한계가 있습니다. 이를 해결하기 위해 Structured Preference Optimization (SPO)라는 새로운 접근법을 제안했습니다. SPO는 장기 작업 계획에 있어 추론 (reasoning) 및 행동 선택 (action selection)을 향상시키는 것을 목표로 합니다.

- **Technical Details**: SPO는 두 가지 핵심 개념을 도입합니다. 첫째, Preference-Based Scoring and Optimization은 태스크 관련성, 비주얼 그라운딩 (visual grounding), 역사적 일관성을 기반으로 추론 체인을 체계적으로 평가합니다. 둘째, Curriculum-Guided Training은 모델이 단순한 작업에서 복잡한 작업으로 점진적으로 적응하여 장기 시나리오에서의 일반화 능력을 향상시키고 추론의 견고성을 강화합니다.

- **Performance Highlights**: SPO는 VirtualHome과 Habitat 2.0에서 각각 +5.98% GCR 및 +4.68% SR, +3.30% GCR 및 +2.11% SR 개선을 이뤘습니다. 이는 이전의 방법을 능가하며 비전-언어 태스크 플래닝에서의 선호 기반 최적화의 효과성을 입증합니다. 실험 결과는 SPO가 장기 작업에서의 추론 품질 및 최종 결정의 정확성을 크게 향상시키는 것을 보여줍니다.



### ProAI: Proactive Multi-Agent Conversational AI with Structured Knowledge Base for Psychiatric Diagnosis (https://arxiv.org/abs/2502.20689)
Comments:
          21 pages, 8 figures

- **What's New**: 이 논문은 대화형 AI 시스템이 사용자 프롬프트에 반응하는데 그쳤던 기존의 방식에서 벗어나, 좀 더 주도적(proactive)이고 목표 지향적(goal-oriented)인 대화형 AI 프레임워크인 ProAI를 소개합니다. ProAI는 정신 건강의 차별 진단을 위한 응용 프로그램의 맥락을 통해 AI가 적절한 질문을 던지고 대화를 특정 목표로 안내할 수 있도록 설계되었습니다.

- **Technical Details**: ProAI는 구조화된 지식 기반 메모리(knowledge-guided memory), 다중 에이전트의 주도적(reasoning) 사고, 다면적인 평가 전략을 통합하여 LLMs(대형 언어 모델)이 단순한 응답 생성을 넘어 전문가 스타일의 진단(reasoning) 진행을 가능하게 합니다. 이러한 기술적 요소들은 AI가 대화 중에 전문적이고 공감하는(interaction standards) 방식으로 상호작용할 수 있도록 지원합니다.

- **Performance Highlights**: 모의 환자 상호작용(simulated patient interactions), 사용자 경험 평가(user experience assessment), 전문가 임상 검증(professional clinical validation)을 통해 ProAI는 정신 질환의 차별 진단에서 최대 83.3%의 정확도를 달성했습니다. 이러한 결과는 LLM들이 반응형 대화 시스템을 넘어 더욱 신뢰할 수 있는, 적응력 있는(goal-driven) AI 진단 비서로 발전할 수 있는 잠재력을 보여줍니다.



### Automatic database description generation for Text-to-SQL (https://arxiv.org/abs/2502.20657)
- **What's New**: 이번 논문은 Natural Language(자연어) 질의를 Structured Query Language(SQL)로 변환하는 Text-to-SQL 기술의 일환으로, 데이터베이스의 테이블과 열 설명이 중요한 역할을 한다고 강조합니다. 자동으로 효과적인 데이터베이스 설명을 생성하는 방법을 제공하며, 이는 명시적인 설명이 부족할 때 유용합니다. 두 가지 접근 방식인 coarse-to-fine(거친-세밀)과 fine-to-coarse(세밀-거친) 프로세스를 통해 이 과정을 수행하여 데이터베이스 구조에 대한 포괄적인 이해를 가능하게 합니다.

- **Technical Details**: 제안된 방법론은 전체 데이터베이스의 이해에서 시작하여 특정 도메인을 인식하고 기존 지식을 활용하여 데이터베이스에서 나타나는 주요 속성을 분석합니다. 각 테이블과 열에 대한 기능 분석과 의미 예측을 통해 데이터의 저장 유형과 목적을 정리합니다. 개발된 모델은 테이블 및 열을 상세히 분석하고, 정보 흐름이 맥락에 맞게 정렬되도록 합니다.

- **Performance Highlights**: 실험 결과에 따르면 제안한 방법으로 생성된 설명을 사용할 경우 SQL 생성 정확도가 0.93% 향상된 것으로 나타났습니다. 이는 총 37%의 인간 수준 성능에 접근하는 것을 보여줍니다. 논문에서 제안된 방법은 일반적으로 수작업으로 입력되는 주석의 부족으로 인해 발생하는 문제를 해결하는 효과적인 대안이 될 것으로 보입니다.



### NutriGen: Personalized Meal Plan Generator Leveraging Large Language Models to Enhance Dietary and Nutritional Adherenc (https://arxiv.org/abs/2502.20601)
- **What's New**: 본 논문에서는 NutriGen이라 불리는 개인화된 음식 추천 시스템을 소개합니다. NutriGen은 사용자 정의 식습관과 제약 사항에 부합하는 맞춤형 식단을 생성하며, 큰 언어 모델(LLM)을 활용하여 더 나은 유연성과 사용 편의성을 제공합니다. 기존 시스템의 한계점을 해결하고 지속 가능한 식습관을 지원하기 위한 혁신적인 접근법을 제공합니다.

- **Technical Details**: NutriGen은 사용자의 음식 섭취 이력, 식습관, 선호 및 식이 제한을 기반으로 개인화된 식단을 생성하는 프레임워크를 제공합니다. 시스템은 사용자의 입력을 받아 구조화된 영양 데이터베이스를 통합하고, 이를 통해 칼로리 카운트, 대체 음식 옵션 및 제안된 레시피를 포함한 식단을 구성합니다. 식사 계획 생성 과정은 구조화된 프롬프트를 사용하여 사용자 요구 사항에 맞춘 맞춤형 출력이 이루어집니다.

- **Performance Highlights**: NutriGen의 평가 결과, Llama 3.1 8B 모델과 GPT-3.5 Turbo가 각각 1.55% 및 3.68%의 낮은 비율 오차를 기록하며, 사용자 정의 칼로리 목표에 밀접하게 일치하는 식단을 효과적으로 생성함을 보여주었습니다. 또한 DeepSeek V3와 여러 기존 모델들의 성능을 비교하여 개인화된 영양 계획에서의 잠재력을 평가하였습니다.



### LLMs Have Rhythm: Fingerprinting Large Language Models Using Inter-Token Times and Network Traffic Analysis (https://arxiv.org/abs/2502.20589)
- **What's New**: 이 논문에서는 에 대한 새로운 접근 방식을 제안합니다. 저자들은 대형 언어 모델(LLMs)의 비侵입적인 실시간 지문 기술을 개발했으며, 이는 네트워크 트래픽이 암호화된 경우에도 모델을 식별할 수 있도록 돕습니다. 이 기법은 토큰 간 시간 간격(Inter-Token Times, ITTs)을 측정하여 각 모델의 고유한 타이밍 패턴을 식별합니다.

- **Technical Details**: 제안된 방법은 딥 러닝(Deep Learning) 기반의 파이프라인을 사용하여 네트워크 트래픽 데이터를 처리하고, 36개의 특징을 추출합니다. 이 특징들은 양방향 장기 단기 기억(BiLSTM) 레이어와 다중 헤드 주의 메커니즘(multi-head attention mechanism)을 포함하는 하이브리드 DL 아키텍처로 전달되어 모델을 식별합니다. 저자들은 이 기술을 다양한 배포 시나리오에서 평가하여 효과적이고 강력함을 입증했습니다.

- **Performance Highlights**: 실험 결과는 16개의 소형 언어 모델(SLMs)과 10개의 독점 LLM에서 제안된 기술이 높은 정확도로 모델을 식별할 수 있음을 보여줍니다. 여러 네트워크 조건에서도 탁월한 성능을 유지하며, 이로 인해 모델 식별에 대한 새로운 관점을 제시하고 더 안전한 LLM 배포를 위해 기여할 수 있음을 확인할 수 있었습니다.



### ECCOS: Efficient Capability and Cost Coordinated Scheduling for Multi-LLM Serving (https://arxiv.org/abs/2502.20576)
- **What's New**: 본 논문에서는 대형 언어 모델(LLMs)을 위한 새로운 스케줄링 프레임워크인 ECCOS를 제안합니다. ECCOS는 쿼리의 응답 품질과 작업량을 명시적으로 제약하여 LLM 추론 비용을 최적화하는 데 중점을 둡니다. 특히, 이 프레임워크는 두 단계의 스케줄링을 통해 멀티-LLM 서비스를 처리합니다.

- **Technical Details**: ECCOS는 멀티-목표 예측기(multi-objective predictor)와 제약 최적화기(constrained optimizer)를 핵심으로 두고 있습니다. 첫 번째 단계에서는 모델 능력과 비용을 추정하고, 두 번째 단계에서는 제약 조건을 만족하며 최적의 쿼리-모델 할당을 결정하는 알고리즘을 설계합니다. 문제는 제약 최적화 문제로 형식화되며, 응답 품질과 시스템 작업량이 고려됩니다.

- **Performance Highlights**: ECCOS는 기존 방법에 비해 성공률을 6.30% 향상시키고 비용을 10.15% 절감하는 성과를 보입니다. 또한, LLM의 응답 시간의 0.5% 미만을 소모하며, 요청이 증가할수록 더욱 향상된 성능을 발휘합니다. 이 프레임워크는 간단한 쿼리를 작은 모델에 효과적으로 배분하면서 높은 품질 기준을 유지하는 강점을 가지고 있습니다.



### Visual Reasoning at Urban Intersections: FineTuning GPT-4o for Traffic Conflict Detection (https://arxiv.org/abs/2502.20573)
- **What's New**: 이번 연구는 신호 없는 도시 교차로에서의 교통 제어의 복잡한 문제를 해결하기 위해 Multimodal Large Language Models (MLLMs), 특히 GPT-4o의 능력을 탐구합니다. 이 방법은 교차로의 조감도 비디오를 직접 사용하여 논리적이고 시각적인 추론을 제공하는 방식을 채택합니다. 이 연구는 교차로의 충돌 감지 및 운전자를 위한 설명과 권장 사항을 제공하는 지능형 시스템으로서의 GPT-4o의 잠재력을 강조합니다.

- **Technical Details**: 연구에서 사용된 GPT-4o는 사전 훈련된 언어 모델에 비디오 데이터를 결합하여 학습되었습니다. 이를 통해 모델은 교차로에서 발생하는 충돌을 탐지하고, 이에 대한 설명 및 다음 행동을 추천하는 기능을 수행합니다. 모델은 77.14%의 정확도로 훈련된 반면, 수동 평가에서는 설명에 대해 89.9%, 추천된 다음 행동에 대해 92.3%의 정확도를 달성하였습니다.

- **Performance Highlights**: MLLMs를 사용하여 비디오 데이터 기반의 실시간 교통 관리가 가능하다는 결과를 도출했습니다. 이 연구의 결과는 교차로 관리 및 운영에 대한 확장 가능하고 실용적인 통찰을 제공하며, 향후 교통 관리 시스템에 MLLMs의 활용 가능성을 보여줍니다. 연구에서 사용된 코드는 특정 URL에서 확인할 수 있습니다.



### HazardNet: A Small-Scale Vision Language Model for Real-Time Traffic Safety Detection at Edge Devices (https://arxiv.org/abs/2502.20572)
- **What's New**: 본 논문에서는 차량 증가와 복잡한 도로 네트워크로 인해 심화되는 도시 교통 안전 문제를 해결하기 위한 새로운 모델, HazardNet을 소개합니다. HazardNet은 고급 언어 및 비전 모델의 추론 능력을 활용하여 교통 안전을 개선하도록 설계된 소규모 Vision Language Model입니다. 이번 연구에서 우리는 Qwen2-VL-2B 모델을 미세 조정하여 HazardNet을 구축하였으며, 이는 오픈소스 대안 중에서 뛰어난 성능을 발휘하는 것으로 평가되었습니다.

- **Technical Details**: HazardNet은 20억 개의 파라미터를 가진 경량화된 모델로 에지 디바이스에서의 효율적인 추론 처리 속도를 지원합니다. 또한, 본 연구에서는 교통 안전과 관련된 실제 상황을 다루기 위해 նոր한 Vision Question Answering (VQA) 데이터셋인 HazardQA를 신규 구축했습니다. 이 데이터셋은 HazardNet의 훈련에 사용 됩니다.

- **Performance Highlights**: 실험 결과, 미세 조정된 HazardNet은 기본 모델에 비해 F1-Score에서 최대 89%의 성능 향상을 보였으며, 일부 경우에는 GPT-4o와 같은 대형 모델과 비교했을 때도 최대 6% 개선된 결과를 나타냈습니다. 이러한 발전은 HazardNet이 실시간으로 신뢰할 수 있는 교통 안전 사건 탐지를 제공할 수 있는 잠재력을 가지고 있음을 강조합니다.



### Towards Statistical Factuality Guarantee for Large Vision-Language Models (https://arxiv.org/abs/2502.20560)
- **What's New**: 이 논문에서는 LVLM(대형 비전-언어 모델)의 신뢰성을 높이기 위해 ConfLVLM이라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 기존 LVLM이 생성한 텍스트가 시각적 맥락과 불일치하는 문제를 해결하기 위해 설계되었습니다. ConfLVLM은 통계적 가설 검정을 통해 생성된 텍스트의 신뢰성을 평가하고, 잘못된 주장을 사전에 필터링함으로써 사용자에게 정확한 정보를 제공합니다.

- **Technical Details**: ConfLVLM은 LVLM을 가설 생성기로 간주하고, 각 생성된 텍스트는 개별적인 주장으로 다룹니다. 이 프레임워크는 효율적인 불확실성 측정치를 사용하여 주장을 검증할 수 있도록 설계되었습니다. 통계적 가설 검정 절차를 활용하여, 잘못된 주장을 걸러내고 신뢰할 수 있는 답변만 사용자에게 제공합니다.

- **Performance Highlights**: ConfLVLM은 LLaVa-1.5에 의해 생성된 장면 설명의 오류율을 87.8%에서 10.0%로 대폭 줄였습니다. 이 과정에서 95.3%의 진정한 양성률(true positive rate)을 기록하여, 필터링의 효과성을 입증했습니다. 또한, ConfLVLM은 다양한 LVLM 및 불확실성 측정치와 조합되어 다양한 비전-언어 작업에 유연하게 적용될 수 있는 특징을 가지고 있습니다.



### $Q\sharp$: Provably Optimal Distributional RL for LLM Post-Training (https://arxiv.org/abs/2502.20548)
- **What's New**: 이 연구에서는 KL-정규화된 강화 학습(RL)을 위한 가치 기반 알고리즘인 $Q\sharp$을 소개합니다. 기존의 정책 기반 방법이 가진 한계를 개선하고자 하였으며, 최적의 정규화된 $Q$ 함수를 통해 레퍼런스 정책을 안내합니다. 특히 $Q\sharp$은 수학적 이론을 입증받은 방법으로, KL 정규화된 RL 문제에 대한 최적 정책을 학습하는 데 유리합니다.

- **Technical Details**: 이 알고리즘은 분포론적 강화 학습(distributional RL)을 활용하여 집계된 온라인 데이터셋에서 최적 $Q$ 함수를 학습합니다. $Q\sharp$는 정책의 가중치를 변경하지 않고도 레퍼런스 정책의 생성을 안내할 수 있으며, 이 과정에서 더 작은 모델을 사용해도 성능을 향상시킬 수 있습니다. 전통적인 RL 방법과 달리 $Q\sharp$은 복잡한 시계열 차이 학습(temporal difference learning) 없이 직접 감독 학습(supervised learning)으로 대체합니다.

- **Performance Highlights**: $Q\sharp$은 수학적 추론 벤치마크에서 이전의 기준 알고리즘들보다 뛰어난 성능을 보였습니다. 또한, 레퍼런스 정책과의 KL 발산을 줄임으로써 안정적인 학습을 도모하였습니다. 실험 결과는 $Q\sharp$이 LLM 후속 훈련에 효과적인 접근법임을 보여줍니다.



### EgoNormia: Benchmarking Physical Social Norm Understanding (https://arxiv.org/abs/2502.20490)
- **What's New**: 이 논문은 Norms(규범) 이해 및 추론을 위한 새로운 데이터셋인 EgoNormia를 소개합니다. 총 1,853개의 자기 중심의 동영상(interaction videos)과 관련 질문들이 포함되어 있으며, 이는 기계가 다양한 규범들 간의 trade-off(상충) 문제를 이해하도록 도와줍니다.

- **Technical Details**: EgoNormia 데이터셋은 안전(safety), 개인정보 보호(privacy), 근접성(proxemics), 공손성(politeness), 협력(cooperation), 조정/능동성(coordination/proactivity), 소통/가독성(communication/legibility) 등 일곱 가지 규범적 행동(normative actions)을 평가하는데 초점을 맞추고 있습니다. 이 데이터셋의 구축에 있어 비디오 샘플링(video sampling), 자동 답변 생성(automatic answer generation), 필터링(filtering) 및 인간 검증(human validation)을 활용한 새로운 파이프라인(pipeline)을 제안합니다.

- **Performance Highlights**: 현재 최첨단 비전-언어 모델(VLMs)은 EgoNormia에서 최대 45%를 기록하며, 인간 기준(92%)과 비교할 때 규범적 이해가 부족함을 보여줍니다. 이 논문에서는 안전, 개인정보 보호 및 협력과 소통 능력의 부족으로 인해 실제 에이전트(real-world agents)에 적용할 경우 발생할 수 있는 큰 위험을 강조합니다. 또한, 검색 기반 생성 방법을 통해 EgoNomia를 사용하여 VLMs의 규범적 추론(normative reasoning) 능력을 향상시킬 수 있음을 보여줍니다.



### Brain-Inspired Exploration of Functional Networks and Key Neurons in Large Language Models (https://arxiv.org/abs/2502.20408)
Comments:
          13 pages, 5 figures

- **What's New**: 본 연구는 대규모 언어 모델(LLMs)에서의 기능적 네트워크(FBNs)를 탐구하기 위한 새로운 접근 방식을 소개합니다. 기존의 연구들은 개별 뉴런의 기여도에 중점을 두었으나, 본 연구는 뉴런 간의 상호작용을 통해 성능이 발휘되는 복잡한 네트워크를 분석합니다. 이러한 새로운 관점은 LLM의 메커니즘을 이해하는 데 기여하며, 이를 통해 모델의 투명성 및 신뢰성을 향상시킬 수 있습니다.

- **Technical Details**: 본 연구에서는 LLM의 다층 퍼셉트론(MLP) 층의 출력 데이터를 기능적 자기 공명 영상(fMRI) 신호와 유사하게 간주하여 독립 성분 분석(ICA) 기법을 적용합니다. 이러한 분석 방법을 통해 LLM 내에서 다수의 기능적 네트워크가 존재함을 확인하였으며, 이들 네트워크는 다양한 입력 자극 간의 공간적 일관성을 보여줍니다. 특히, 핵심 기능 네트워크를 차단할 경우 모델의 성능이 상당히 저하된다는 결과를 아울러 제시하였습니다.

- **Performance Highlights**: 연구 결과, LLM에서 2% 미만의 뉴런으로 구성된 핵심 기능 네트워크를 유지할 경우 모델 성능을 유지하거나 향상시킬 수 있음을 발견하였습니다. 불필요한 뉴런을 차단하면서 새로운 기능적 네트워크를 점진적으로 통합함으로써 모델의 성능이 낮은 단계에서 높은 단계로 개선될 수 있음을 보여주었습니다. 마지막으로, MLP 층의 10% 이하의 뉴런만으로도 원래 네트워크와 맞먹는 성능을 달성했습니다.



### Behind the Tip of Efficiency: Uncovering the Submerged Threats of Jailbreak Attacks in Small Language Models (https://arxiv.org/abs/2502.19883)
Comments:
          12 pages. 6 figures

- **What's New**: 본 논문은 작은 언어 모델(SLMs)의 보안 성능을 평가하기 위해 13개의 최신 SLM을 다양한 jailbreak 공격하에 실험한 포괄적인 연구를 제공합니다. SLM은 효율성과 낮은 계산 비용 덕분에 엣지 디바이스에서의 배포에 매력적이지만, 보안 리스크는 대형 언어 모델(LLMs)에 비해 상대적으로 소홀히 다뤄졌습니다. 연구 결과, 대부분의 SLM이 기존 jailbreak 공격에 취약하다는 것을 밝혀내며, 대응 방법의 효과도 평가합니다.

- **Technical Details**: 이 연구는 16개의 최첨단 모델을 수집하여 SLM과 LLM의 보안 차이를 폭넓게 살펴봅니다. SLMs는 수십억 개의 매개변수로 구성되어 있어 적은 양의 교육 데이터와 계산 비용으로 배포할 수 있지만, LLM에 비해 보안에 더 취약합니다. 찾아낸 취약점은 SLM의 아키텍처 압축, 양자화(quantization), 지식 증류(knowledge distillation) 등 다양한 요인들로 인해 발생합니다.

- **Performance Highlights**: SLM에 대한 다양한 방어 기법의 효과를 평가한 결과, 이러한 기법들은 SLM이 jailbreak 공격에 대한 회복력을 향상시키는 데 상당한 적용 가능성을 가지고 있음을 보여주었습니다. 특히 SLM은 LLM에 비해 대응 방법의 적응력이 높아, 보안 강화를 위한 새로운 통찰력을 제공합니다. 이 연구는 SLM 개발 시 보안과 생성 능력 간의 균형을 맞추기 위한 중요한 기초 자료가 될 것입니다.



### Are All Spanish Doctors Male? Evaluating Gender Bias in German Machine Translation (https://arxiv.org/abs/2502.19104)
Comments:
          ISCA/ITG Workshop on Diversity in Large Speech and Language Models

- **What's New**: WinoMTDE는 독일어 기계 번역 시스템의 직업 편견 및 여성의 저지도를 평가하기 위한 새로운 성별 편향 평가 테스트 세트입니다. 이 데이터셋은 288개의 독일어 문장으로 구성되어 있으며, 성별과 고정관념에 대해 균형을 이루고 있습니다. 기존의 자동 평가 방법을 기반으로 한 이 연구는 독일어에 특화된 기계 번역 시스템을 평가하기 위해 확장되었습니다.

- **Technical Details**: WinoMTDE 데이터세트는 Winograd 스키마를 따르는 문장 구조로 구성되어 있으며, 각 문장은 특정 성별을 가진 주어를 포함합니다. 데이터셋은 남성과 여성 주어가 동등한 수로 포함되어 있어 성별과 고정관념에 대해 균형을 이룹니다. 평가 파이프라인은 번역, 예측 및 평가의 세 가지 주요 단계로 나뉘며, 목표 언어로 번역된 문장의 성별 정보에 따라 다양한 메트릭이 계산됩니다.

- **Performance Highlights**: 평가 결과, 대부분의 기계 번역 시스템에서 지속적인 성별 편향이 발견되었습니다. GPT-4o-mini와 같은 대형 언어 모델이 전통적인 시스템보다 더 우수한 성능을 보였습니다. 이 연구는 기계 번역의 성별 편향 문제를 해결하기 위한 기준 및 데이터셋을 제공하고 있으며, 전체 문서와 관련 코드는 공개적으로 이용 가능하다고 발표했습니다.



New uploads on arXiv(cs.IR)

### Joint Modeling in Recommendations: A Survey (https://arxiv.org/abs/2502.21195)
Comments:
          arXiv admin note: text overlap with arXiv:2302.03525

- **What's New**: 이번 논문은 Deep Recommender Systems (DRS)의 단일 추천 작업 및 데이터 모드에 의존하는 기존 방법의 한계를 지적하고, 여러 작업과 시나리오를 통합하는 joint modeling 접근법의 필요성을 강조합니다. 이는 사용자의 복잡하고 변화하는 선호도를 보다 정확하게 반영할 수 있도록 합니다. 동시에 다양한 접근 방식을 포괄하여 추천의 정밀도, 효율성 및 개인화를 크게 향상할 수 있는 가능성을 제시합니다.

- **Technical Details**: 논문에서는 joint modeling의 네 가지 차원인 multi-task, multi-scenario, multi-modal 및 multi-behavior modeling에 대한 정의와 중요성을 설명합니다. multi-task 모델링은 서로 다른 추천 작업 간의 시너지를 활용하는 것을 목표로 하고, multi-scenario 모델링은 다양한 사용자 상호작용의 풍부한 다양성을 활용하여 시스템의 적응성과 견고성을 향상시킵니다. 또한 multi-modal 모델링은 텍스트, 시각 및 청각 정보를 포함하여 사용자 선호도를 보다 정교하게 이해하려고 합니다.

- **Performance Highlights**: joint modeling 접근법은 더 정교하고 개인화된 추천 결과를 제공하여 사용자 경험을 향상시키는 데 중점을 둡니다. 최신 기술들과 연구 방향을 기반으로 여러 가지 promising avenue를 제시하며, 앞으로의 연구 방향에 대해 논의합니다. 마지막으로, joint modeling이 DRS의 발전에 필수적이며 지속적인 혁신이 필요함을 강조하면서 논문을 마무리합니다.



### Extending Dense Passage Retrieval with Temporal Information (https://arxiv.org/abs/2502.21024)
- **What's New**: 이 연구에서는 전통적인 정보 검색 방법인 BM25 및 Dense Passage Retrieval(DPR)가 시간 민감성 쿼리를 처리하는 데 한계가 있음을 지적하고, 이를 극복하기 위한 새로운 Temporal Retrieval Model을 소개합니다. 이 모델은 쿼리 타임스탬프와 문서 날짜를 통합하여 명시적인 시간 신호를 검색 시스템에 결합합니다. 이를 통해 검색된 내용이 주제적으로 관련 있을 뿐만 아니라 사용자 의도와 시간적으로 일치하도록 합니다.

- **Technical Details**: Temporal Dense Passage Retrieval(TempDPR)이라는 모델은 세 가지 혁신적인 접근 방식을 도입합니다. 첫째, DateAsTag 접근 방식은 타임스탬프를 [S-DATE] 및 [E-DATE]처럼 특수 토큰으로 감싸 명시적으로 시간 정보를 표시합니다. 둘째, DateAsToken 접근법은 쿼리와 문서에 타임스탬프를 일반 텍스트로 추가하고, 세 번째로는 TempDPR이 쿼리 및 문서 표현에 시간 임베딩을 직접 통합하여 다양한 융합 기술을 탐색합니다.

- **Performance Highlights**: 이 연구는 ArchivalQA와 ChroniclingAmericaQA라는 대규모 벤치마크 데이터 세트에서 성능을 평가하여 표준 검색 기준선 대비 상당한 성능 향상을 달성했습니다. 특히, ArchivalQA에서 Top-1 검색 정확도가 6.63% 향상되었고, ChroniclingAmericaQA에서는 9.56% 향상됨을 보여줍니다. 이러한 결과는 시간 모델링을 통한 검색 시스템의 중요성과 시간적으로 기반한 쿼리를 처리하는 새로운 기준을 제시하고 있습니다.



### Variations in Relevance Judgments and the Shelf Life of Test Collections (https://arxiv.org/abs/2502.20937)
Comments:
          11 pages, 6 tables, 5 figures

- **What's New**: 이 연구는 전통적인 테스트 컬렉션의 평가 안정성에 대한 속성을 확인했습니다. 최근에는 신경 검색 모델이 현대 테스트 컬렉션의 특성에 영향을 미치고 있으며, 문서가 짧아지고 판단 기준이 4단계로 변경되는 등 새로운 조건에서 평가의 신뢰성이 유지되는지를 조사하고 있습니다. 또한, 테스트 컬렉션의 만료 기준을 설정하는 필요성을 제기하고 있습니다.

- **Technical Details**: 기존의 TREC DL 2019 컬렉션을 기반으로 한 재annotation 연구를 수행하여 다양한 주제에 대해 2명의 보조 주석자를 활용하여 관련성을 재평가했습니다. 새로운 관련성 판단을 통해, 주석자 간의 낮은 동의에도 불구하고 시스템 순위를 안정적으로 유지할 수 있음을 보여줍니다. 또한, 인간 평가자의 판단이 현대의 시스템 성능에 미치는 영향을 분석하여 신경 검색 모델의 편향을 평가합니다.

- **Performance Highlights**: 재생산 연구의 결과에 따르면, 현재의 첨단 모델들이 인간 평가자들과 유사한 성능을 내는 것으로 확인되었고, 최고 성능의 한계는 nDCG@10에서 약 0.81로 평가되었습니다. 또한, 기존의 평가 시스템들이 시간 경과에 따라 만료될 수 있는 가능성도 제시하며, 새로운 관련성 판단이 시스템 순위에 큰 영향을 미칠 수 있음을 발견했습니다.



### Scalable Overload-Aware Graph-Based Index Construction for 10-Billion-Scale Vector Similarity Search (https://arxiv.org/abs/2502.20695)
Comments:
          Accepted by WWW'25

- **What's New**: 본 논문은 SOGAIC라는 새로운 시스템을 소개하며, 이것은 초대형 벡터 데이터베이스를 위한 최초의 확장 가능한 과부하 인식 그래프 기반 ANNS 인덱스 구축 시스템입니다. 기존의 ANNS 알고리즘들이 높은 리콜률을 달성했지만, 느린 구축 속도와 한정된 확장성으로 인해 실제 산업 환경에서는 사용에 제약이 있었습니다. SOGAIC는 동적 데이터 파티셔닝 알고리즘과 로드 밸런싱 작업 스케줄링 프레임워크를 결합하여 효율적으로 대규모 데이터 집합을 처리합니다.

- **Technical Details**: SOGAIC의 구조는 두 개의 주요 단계로 구성됩니다. 첫 번째 단계에서는 과부하 경계를 제어하면서 겹쳐진 하위 집합을 분할하고, 두 번째 단계에서는 배포된 클러스터를 통해 하위 그래프 구축 작업을 스케줄링하여 최종 ANNS 그래프 인덱스를 형성합니다. 이 시스템은 용량과 최대 겹치는 요소에 따라 적절한 파티션 수를 계산하고, K-평균 클러스터링을 통해 초기 참조 점을 생성합니다.

- **Performance Highlights**: SOGAIC는 기존 방법들에 비해 평균 구축 시간을 47.3% 감소시켰습니다. 이 시스템은 100억 개 이상의 벡터를 매일 갱신하며, 수억 명의 사용자에게 서비스를 제공하는 실제 산업 검색 엔진에 성공적으로 배포되었습니다. 이를 통해 SOGAIC는 대규모 데이터셋에서의 효율적인 벡터 유사도 검색 성능을 크게 개선함을 보여줍니다.



### Unleashing the Potential of Two-Tower Models: Diffusion-Based Cross-Interaction for Large-Scale Matching (https://arxiv.org/abs/2502.20687)
- **What's New**: 본 논문에서는 새로운 매칭 패러다임인 T2Diff를 제안합니다. 이는 정보 상호작용을 강조하는 생성적 교차 상호작용 분리 아키텍처로, 두 개의 타워 모델의 잠재력을 최대한 활용할 수 있습니다. T2Diff는 확산 모듈(diffusion module)을 사용하여 사용자 다음 긍정 의도를 재구성하고, 혼합 주의(mixed-attention) 메커니즘을 도입하여 사용자와 아이템 간의 상호작용을 원활하게 합니다.

- **Technical Details**: T2Diff는 기존의 두 타워 아키텍처의 한계를 극복하기 위해 설계되었습니다. 사용자의 긍정적 상호작용을 아이템 타워를 통해 재구성하기 위한 생성 방법을 채택합니다. 이 모델은 혼합 주의 모듈을 통해 사용자와 아이템 특징 간의 복합적인 상호작용을 개선하며, 사용자의 행동 시퀀스에서 시간 변화를 명확히 추출하여 재구성의 정확도를 높입니다.

- **Performance Highlights**: 실험 결과, T2Diff는 두 개의 현실 세계 데이터셋과 하나의 산업 데이터셋에서 기존 SOTA 두 타워 모델들을 크게 초월하는 성능을 보여주었습니다. 또한, T2Diff의 확산 접근법은 아이템 표현의 재구성에서 다른 생성 모델보다 우수한 성능을 나타냈습니다. 이를 통해 T2Diff는 높은 정확도와 낮은 지연 시간(rate latency)을 동시에 달성함을 입증하였습니다.



### CS-PaperSum: A Large-Scale Dataset of AI-Generated Summaries for Scientific Papers (https://arxiv.org/abs/2502.20582)
- **What's New**: 이 논문에서는 CS-PaperSum이라는 대규모 데이터셋을 소개합니다. 이 데이터셋은 31개의 주요 컴퓨터 과학 회의에서 수집한 91,919개의 논문을 포함하고 있으며, ChatGPT를 이용한 AI 생성 구조 요약으로 보강되어 있습니다. CS-PaperSum은 기존의 메타데이터 기반 데이터셋의 한계를 극복하고, 연구 성과와 방법론을 효과적으로 정리한 정보를 제공합니다.

- **Technical Details**: CS-PaperSum은 GPT-3.5를 사용하여 각 논문의 핵심 기여, 새로운 방법론, 평가 지표 및 미래 연구 방향을 구조적으로 요약합니다. 이를 통해 효율적으로 대규모 과학 문헌 분석과 연구 경향 예측이 가능해집니다. 논문 간의 의미적 정렬을 평가하기 위해 SciBERT를 활용하여 원본 논문과 AI 생성 요약 간의 임베딩을 비교하고, 이를 시각화하여 대조를 수행하였습니다.

- **Performance Highlights**: CS-PaperSum 데이터를 활용한 연구 동향 분석 결과, 주요 AI 회의에서의 연구 초점의 변화를 확인하였습니다. 예를 들어, NeurIPS에서 전통적인 강화 학습 방법에서 그래프 신경망으로의 전환을 관찰했습니다. 이 데이터셋은 AI 기반의 문헌 분석과 연구 트렌드 탐지에 기여하며, 과학적 발견을 지원하는 AI 시스템 개발에 중요한 자산이 될 것입니다.



### Creator-Side Recommender System: Challenges, Designs, and Applications (https://arxiv.org/abs/2502.20497)
Comments:
          9 pages and 9 figures

- **What's New**: 이 논문에서는 콘텐츠 제작자의 경험을 향상시키기 위해 DualRec이라는 새로운 창작자 측 추천 시스템을 제안합니다. 기존의 추천 시스템이 사용자 측에 집중되어 있었던 반면, DualRec은 제작자가 적절한 사용자와 연결될 수 있도록 지원합니다. 이 시스템은 기존 사용자 측 알고리즘을 간단하게 변경하여 창작자 측 버전으로 전환할 수 있도록 설계되었습니다.

- **Technical Details**: DualRec 시스템은 사용자 요청에 따라 적합한 사용자를 찾아 콘텐츠를 추천하는 구조로, 이를 통해 제작자의 만족도를 높이는 것을 목표로 합니다. DualRec은 두 가지 주요 모듈인 검색 모듈과 순위 매기기 모듈을 포함하며, 사용자 가용성 계산(User Availability Calculation, UAC) 모듈을 도입하여 추천 성능을 개선합니다. 이 시스템은 변경이 간단해 기존 사용자 측 연구 결과를 창작자 측으로 직접 적용할 수 있습니다.

- **Performance Highlights**: DualRec은 이미 1억 명 이상의 사용자와 1천만 명 이상의 제작자가 있는 Kwai 플랫폼에 구현되어 있으며, 제작자의 경험을 크게 향상시켰습니다. 사용자 가용성 문제를 해결함으로써, 추천 시스템이 사용자를 효과적으로 목표로 설정할 수 있도록 지원합니다. 이는 제작자들이 더 많은 피드백을 받고, 콘텐츠 제작을 촉진하는 긍정적인 결과를 가져옵니다.



### Optimizing Large Language Models for ESG Activity Detection in Financial Texts (https://arxiv.org/abs/2502.21112)
- **What's New**: 이 논문은 ESG(환경, 사회, 지배구조) 요인이 기업 의사결정에 통합되는 과정에서 AI 모델을 활용하여 지속 가능성 보고서와 비재무 공시를 자동으로 평가하는 새로운 접근 방식을 제안합니다. 특히, ESG-Activities라는 새로운 벤치마크 데이터셋을 도입하여 이를 통해 LLMs(대형 언어 모델)의 성능을 강화하는 방법에 대해 설명합니다.

- **Technical Details**: 현재 세대 LLMs의 텍스트 분류 능력을 평가하면서, ESG-Activities 데이터셋을 이용해 모델을 세부 환경 활동에 맞춰 학습시키는 방법을 채택합니다. 이 데이터셋은 전문가가 수작업으로 선별한 데이터와 LLMs에 의해 생성된 합성 데이터를 포함하고 있어 모델 성능을 높이는 데 기여합니다.

- **Performance Highlights**: 실험 결과, ESG-Activities로 파인튜닝된 모델이 기존의 제로샷 학습과 수작업으로 주석이 붙은 데이터만으로 학습한 모델에 비해 분류 정확도가 크게 향상된 것을 확인하였습니다. Llama 7B와 같은 오픈 소스 모델이 특정 설정에서 대형 프로프라이어터리 모델보다 우수한 성능을 발휘했으며, 이는 AI 연구자와 금융 분석가들에게 중요한 영향을 미칠 것입니다.



### Fast 3D point clouds retrieval for Large-scale 3D Place Recognition (https://arxiv.org/abs/2502.21067)
Comments:
          8 pages, 1 figures

- **What's New**: 이 논문에서는 3D 포인트 클라우드의 검색(retrieval) 효율성을 높이기 위해 차별 가능한 검색 색인(Differentiable Search Index, DSI) 기법을 도입하고, 이를 3D 데이터에 맞게 조정하는 방법을 제안하고 있습니다. 이를 통해 포인트 디스크립터(descriptors)를 바탕으로 1D 식별자(identifiers)를 생성하여 상수 시간(constant time)으로 직접적인 검색이 가능하도록 하고 있습니다. 또한, 시각 변환기(Vision Transformers)를 통합하여 포지셔널 및 시맨틱 인코딩(encoding)을 활용하는 방식입니다.

- **Technical Details**: 이 연구는 DSI를 활용하여 대규모 3D 포인트 클라우드의 검색을 가능하게 하는 새로운 방법론을 제안합니다. DSI는 텍스트 정보 검색을 위해 설계된 변환기 기반 접근법으로, 포인트 클라우드 데이터를 docid에 매핑(mapping)하는 단일 T5 모델을 사용합니다. 고급 인덱싱 전략을 소개하여 지리정보 및 힐버트 곡선(Hilbert curve) 기반 방법을 통합함으로써 문서의 표현을 강화합니다.

- **Performance Highlights**: 제안된 방법은 공개 벤치마크에서 평가되어, 속도 및 품질 측면에서 최신 방법과 비교되며 매우 우수한 검색 성능을 보여줍니다. 3D 포인트 클라우드의 지리적 인식을 위해 이 기법이 크게 기여할 것으로 기대됩니다. LiDAR 센서를 사용하여 수집된 데이터의 정확성을 높이는 다양한 응용 분야에 활용될 수 있습니다.



### The RAG Paradox: A Black-Box Attack Exploiting Unintentional Vulnerabilities in Retrieval-Augmented Generation Systems (https://arxiv.org/abs/2502.20995)
- **What's New**: 본 논문에서는 retrieval-augmented generation (RAG) 시스템의 성능을 저하시킬 수 있는 공격 방법을 제시합니다. 기존의 공격 방법이 비현실적인 white-box 가정에 의존하는 반면, 우리는 RAG 시스템이 신뢰성을 높이려 할 때 우연히 드러나는 취약점을 이용한 현실적인 black-box 공격 시나리오를 도입합니다. 이를 통해 RAG 시스템이 외부 문서를 참조하여 응답을 생성하는 과정에서 공격을 수행하는 방법을 제시합니다.

- **Technical Details**: RAG 시스템은 대규모 언어 모델(LLM)이 주어진 쿼리와 관련된 문서를 검색하고 응답 생성 과정에 활용하는 기술입니다. 우리는 RAG 시스템이 참조하는 외부 문서 출처를 식별하고, 이와 일치하도록 잘못된 정보가 포함된 오염된 문서를 자동으로 생성하는 방법을 소개합니다. 최종적으로 오염된 문서는 공개된 출처에 게시되어 RAG 시스템의 응답 생성 과정을 방해합니다.

- **Performance Highlights**: 실험 결과, 제안한 공격 방법은 RAG 시스템에 대한 내부 접근이 없어도 성능을 유의미하게 저하시킬 수 있음을 확인했습니다. 또한, RAG 시스템 내부 관점에서의 재순위화(re-ranking) 방법을 제안하여 예기치 않은 공격에 대한 최소한의 보호를 제공하는 방안을 논의합니다. 이 연구는 RAG 시스템의 신뢰성 문제를 해결하기 위한 기초 작업을 제공합니다.



### WebFAQ: A Multilingual Collection of Natural Q&A Datasets for Dense Retrieva (https://arxiv.org/abs/2502.20936)
Comments:
          10 pages, 3 figures, 7 tables

- **What's New**: WebFAQ는 FAQ 스타일의 주석에서 파생된 대규모 오픈 도메인 질문 답변(QA) 데이터셋 컬렉션입니다. 총 96백만 개의 자연 질문-답변 쌍이 75개 언어로 구성되어 있으며, 이 중 47백만 개(49%)가 비영어 샘플입니다. WebFAQ는 20개의 단일 언어 검색 벤치마크의 기초로도 활용되며, 이렇게 다각화된 데이터는 다국어 밀집 검색 모델의 훈련과 평가를 위한 고품질 자원을 제공합니다.

- **Technical Details**: WebFAQ 데이터셋은 정제된 필터링 및 근사용 데이터 감지를 통해 신중하게 선별되어, 1120만 개 QA 쌍으로 구성된 20개의 단일 언어 검색 벤치마크를 포함합니다. 자동화된 데이터셋 생성에 대한 첨단 방법을 활용하여, QA 정렬된 이중 언어 말뭉치도 구축되었습니다. 또한 수집된 QA 쌍을 활용하여 in-domain 사전 훈련된 XLM-RoBERTa 모델을 미세 조정했으며, 이 모델은 언어 간 검색 데이터셋에도 일반화되는 성능 향상을 보여줍니다.

- **Performance Highlights**: WebFAQ 데이터는 밀집 검색 모델의 성능을 향상시키며, 이는 오픈 도메인 Q&A 검색에서 실질적인 개선을 나타냅니다. 특히, 미세 조정된 모델은 비영어 샘플을 포함한 다양한 다국어 검색 데이터셋에서도 주목할 만한 성능을 발휘합니다. 최종적으로, 자동화된 비텍스트 생성 방법을 통해 구축된 이중 언어 말뭉치는 비슷한 데이터셋에 비해 높은 번역 품질을 보인 것으로 확인되었습니다.



### CoTMR: Chain-of-Thought Multi-Scale Reasoning for Training-Free Zero-Shot Composed Image Retrieva (https://arxiv.org/abs/2502.20826)
- **What's New**: 이번 연구에서는 Zero-Shot Composed Image Retrieval (ZS-CIR) 문제를 다루며, 새로운 Framework인 CoTMR을 제안합니다. CoTMR은 Chain-of-Thought (CoT) 및 Multi-scale Reasoning을 도입하여 기존의 방법들과의 차별점을 보이고 있습니다. 기존의 캡션 모델을 사용하지 않고, Large Vision-Language Model (LVLM)을 통해 통합된 이해와 추론을 제공합니다. 이 과정을 통해 ZS-CIR 작업에서 더 나은 성능과 해석 가능성을 발휘할 수 있게 됩니다.

- **Technical Details**: CoTMR은 복잡한 수정 질의(edited query)를 처리하기 위해 LVLM을 활용합니다. 본 시스템에서는 CIRCoT 라는 새로운 방법론을 통해 LVLM의 추론 과정을 단계별로 구성된 여러 하위 작업(Subtasks)으로 나누어 제공합니다. 또한 Multi-Grained Scoring (MGS) 메커니즘을 통해 각 추론의 유사성 점수를 계산하여 정확한 검색을 가능하게 합니다. 이러한 구조적 접근은 기존의 글로벌 추론을 넘어 세부적인 요소에 대한 예측을 가능하게 합니다.

- **Performance Highlights**: 실험 결과 CoTMR은 유명 벤치마크 테스트에서 기존의 최첨단 방법들보다 현저하게 우수한 성능을 보였습니다. CoTMR은 기존 방법들이 직면한 컴포넌트 비호환성, 시각적 정보 손실, 및 불완전한 추론 문제를 극복하여 더 높은 해석 가능성을 제공합니다. 본 연구는 ZS-CIR의 영역에서 매우 의미 있는 기여로 평가되며, 다양한 작업에 효과적으로 적용될 수 있을 것입니다.



### LexRAG: Benchmarking Retrieval-Augmented Generation in Multi-Turn Legal Consultation Conversation (https://arxiv.org/abs/2502.20640)
Comments:
          10 pages

- **What's New**: LexRAG는 법률 도메인에서 RAG 시스템을 평가하기 위한 최초의 벤치마크로, 1,013개의 다중 턴 대화 샘플과 17,228개의 후보 법률 기사를 포함하고 있습니다. 이 벤치마크는 다섯 단계로 진행되는 질문을 포함하여, 법률 전문가가 주석을 달았습니다. LexRAG는 법률 상담의 복잡성을 이해하고 평가하기 위해 두 가지 주요 작업인 Conversational Knowledge Retrieval과 Response Generation을 포함하고 있습니다.

- **Technical Details**: LexRAG는 다중 턴 대화에서 법률 관련 지식을 정확하게 검색하고 법적으로 타당한 응답을 생성하는 시스템을 평가하는 데 중점을 둡니다. 데이터셋은 법률 전문가에 의해 신중하게 주석이 달린 1,013개의 대화를 포함하며, 각 대화는 다섯 개의 질문과 응답으로 구성됩니다. LexiT라는 법률 RAG 툴킷을 통해 RAG 시스템을 위한 다양한 구현이 가능하며, LLM-as-a-judge 평가 파이프라인은 상세하고 효과적인 평가를 지원합니다.

- **Performance Highlights**: LexRAG를 통해 여러 LLM 및 검색 방법에 대한 종합적인 평가를 수행하여 현재 RAG 시스템이 법률 도메인에서 직면하고 있는 주요 한계와 문제점을 밝혀냈습니다. 이 분석은 법률 상담 시스템과 관련된 RAG의 발전을 위한 귀중한 통찰력을 제공하며, 향후 개선을 위한 로드맵을 제시합니다. LexRAG는 법률 AI 기술을 발전시키고, 다양한 도메인에서 RAG의 미래 개발을 위한 기초를 마련하는 역할을 합니다.



### NANOGPT: A Query-Driven Large Language Model Retrieval-Augmented Generation System for Nanotechnology Research (https://arxiv.org/abs/2502.20541)
Comments:
          61 pages, 3 figures

- **What's New**: 이 논문은 나노기술 연구를 위해 설계된 대형 언어 모델 기반의 정보 검색 증강 생성 시스템(LLM-RAG)의 개발 및 응용을 제시합니다. 이 시스템은 지능형 연구 보조 도구로서의 역할을 하여 나노기술 분야의 문헌 조사를 보다 효율적이고 포괄적으로 만드는 데 기여합니다. Google Scholar의 고급 검색과 Elsevier, Springer Nature 및 ACS Publications의 오픈 액세스 논문을 활용하여 신뢰할 수 있는 여러 출처에서 데이터를 통합합니다.

- **Technical Details**: LLM-RAG 시스템은 고급 쿼리 백엔드 검색 메커니즘을 중심으로 구성되어 있으며, 이는 다수의 신뢰할 수 있는 출처로부터 데이터를 결합합니다. LLM은 대규모 훈련 데이터를 통해 인간과 유사한 텍스트를 생성하고 이해할 수 있으며, 이러한 모델은 자연어 처리(NLP) 분야에서 혁신적인 기술로 부각되고 있습니다. 본 논문에서는 LLM을 기반으로 한 RAG 기법이 LLM의 성능을 개선하지고 구조적 오류를 줄이는 방법에 대해 논의합니다.

- **Performance Highlights**: LLM-RAG 시스템은 포괄적인 문헌 리뷰에 필요한 시간과 노력을 크게 줄이는 동시에 높은 정확성과 쿼리 관련성을 유지하는 데 효과적이라는 것이 엄격한 테스트를 통해 검증되었습니다. 이 시스템은 표준 공개 LLM보다 성능이 우수하며, 나노기술 분야에서 연구를 가속화하는 데 중요한 잠재력을 보여줍니다. 다양한 연구 분야에서 새로운 나노 재료의 발견과 실험 데이터의 해석을 원활하게 하는 데 기여할 것으로 기대됩니다.



New uploads on arXiv(cs.CV)

### How far can we go with ImageNet for Text-to-Image generation? (https://arxiv.org/abs/2502.21318)
- **What's New**: 이 논문에서는 기존의 '더 큰 것이 더 좋다(bigger is better)'라는 패러다임에 도전하여, 정교하게 선별된 작은 데이터셋에 대한 전략적 데이터 증강(strategic data augmentation)을 통해 T2I 생성 모델의 성능을 향상시킬 수 있음을 보여줍니다. 기존의 거대한 웹 스크랩(WEB-SCRAPED) 데이터셋 대신 향상된 ImageNet과 잘 설계된 텍스트 및 이미지 증강을 사용하여 성과를 낼 수 있음을 증명합니다.

- **Technical Details**: 연구에서는 SD-XL과의 성능 비교를 통해 일반적인 데이터셋에 비해 훨씬 적은 파라미터 수(1/10) 및 훈련 이미지 수(1/1000)를 사용하면서도 GenEval에서 +2, DPGBench에서 +5의 점수를 달성했습니다. 이 결과는 잘 선별된 소규모 데이터셋을 활용한 증강이 훨씬 효과적일 수 있음을 나타냅니다.

- **Performance Highlights**: 결과적으로, 대규모 데이터셋을 사용하는 대신 전략적 데이터 증강이 T2I 생성에 더 지속 가능한 길을 제시할 수 있음을 밝혀냈습니다. 이는 T2I 모델의 훈련과 개발에 있어 더 효율적이고 환경적인 접근법이 될 가능성을 제시합니다.



### Raccoon: Multi-stage Diffusion Training with Coarse-to-Fine Curating Videos (https://arxiv.org/abs/2502.21314)
- **What's New**: 이 논문은 텍스트에서 비디오 생성을 위한 새로운 접근 방식을 제안합니다. CFC-VIDS-1M이라는 고품질 비디오 데이터셋을 통계적 정교화(coarse-to-fine curation) 파이프라인을 통해 구축하였습니다. 이는 비디오 품질과 텍스트-비디오 정렬을 개선하기 위해 시각-언어 모델을 사용하여 텍스트 비디오의 의미적 풍부함을 높입니다.

- **Technical Details**: 본 연구에서는 RACCOON이라는 변형기(transformer) 기반 아키텍처를 개발했습니다. 이 아키텍처는 분리된 공간-시간 주의 메커니즘을 활용하여 비디오 생성을 위한 효과적인 처리를 가능하게 합니다. 훈련은 심층적인 네 단계를 통해 진행되며, 이는 모델의 복잡성을 효율적으로 다룰 수 있도록 설계되어 있습니다.

- **Performance Highlights**: 실험 결과, 우리의 고품질 데이터 정리 및 효율적인 훈련 전략이 시각적으로 매력적이고 시간적으로 일관된 비디오 생성에 기여함을 보여주었습니다. 연구팀은 데이터셋, 코드 및 모델을 공개할 계획이며, 이는 다양한 창의적 콘텐츠 및 시각적 storytelling 분야에서의 활용 가능성을 높입니다.



### Unsupervised Parameter Efficient Source-free Post-pretraining (https://arxiv.org/abs/2502.21313)
- **What's New**: 이 논문은 UpStep이라는 새로운 접근 방식을 제안하여, 사전 학습된 모델을 소스 도메인과는 독립적으로 목표 도메인에 효율적으로 적응시키는 방법입니다. UpStep은 라벨이 없는 목표 도메인에서 자기 지도(self-supervised) 학습 방식으로 모델을 조정하는 데 집중하며, 재학습 없이도 파라미터의 낭비를 최소화합니다. 이 접근은 카타스트로픽 포겟팅(catatstrophic forgetting) 문제를 해결하기 위해 CVR(center vector regularization)이라는 보조 기능을 도입하였습니다.

- **Technical Details**: UpStep은 임계점 없이 소스 도메인 데이터에 접근하지 않고도 사전 학습된 모델을 조정할 수 있도록 설계되었습니다. 이 과정에서 고정 클러스터 센터를 사용한 두 가지 스트림의 자기 지도 클러스터링 방식을 채택하였으며, 각 스트림은 동일한 아키텍처를 공유하면서 모델의 효율성을 극대화합니다. 또한, 모델을 적응시키는 과정에서 저차원(low-rank) 적응 기법을 사용하여 필요한 파라미터 수를 줄입니다.

- **Performance Highlights**: 여러 종류의 일반화된 기반 모델을 사용하여 다양한 목표 도메인에서 UpStep의 적응성과 일반화 능력을 입증하였습니다. 이 접근법은 목표 도메인에 대해 사전 학습된 모델을 최적화하는 데 필요한 파라미터 수를 크게 줄이며, 비용 효율적인 방법으로 평가되었습니다. 실험 결과, 다양한 데이터셋에서 UpStep의 성능이 뛰어난 것으로 나타났습니다.



### MIGE: A Unified Framework for Multimodal Instruction-Based Image Generation and Editing (https://arxiv.org/abs/2502.21291)
- **What's New**: 이번 논문에서는 MIGE라는 통합 프레임워크를 제안하여, 텍스트 기반 지침과 주제 중심 이미지 생성을 통합합니다. MIGE는 멀티모달 지침을 활용하여 두 가지 작업을 표준화하고, 상호 보완적인 데이터로부터 학습함으로써 개선된 성과를 달성합니다. 이 접근법은 서로 다른 입력 정보 간의 일관성을 유지하며, 두 작업의 성능을 동시에 향상시킬 수 있습니다.

- **Technical Details**: MIGE는 새로운 멀티모달 인코더를 도입하여 자유로운 형태의 멀티모달 지침을 통합된 비전-언어 공간으로 매핑합니다. 이 인코더는 시각적 특징과 의미적 특징을 통합하여, 보다 정밀한 주제 세부 정보를 유지하도록 설계되었습니다. 또한, 공동 학습을 통해 서로의 작업을 강화할 수 있는 체계를 구축하여 새로운 조합 작업인 지침 기반 주제 드리븐 편집을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, MIGE는 주제 중심 생성 및 지침 기반 편집 모두에서 뛰어난 성능을 보여주며, 새로운 작업인 지침 기반 주제 드리븐 편집에서도 최첨단 결과를 달성합니다. 또한, MIGE는 다양한 시나리오에서 정교하고 일관된 출력을 생성하는 능력을 강조하며, MIGEBench 벤치마크에서의 성능 평가 결과에서도 강력한 성과를 나타냅니다.



### Back to the Future Cyclopean Stereo: a human perception approach unifying deep and geometric constraints (https://arxiv.org/abs/2502.21280)
- **What's New**: 이 논문은 스테레오 비전을 혁신하며, 깊이 불연속성과 가림(occlusion)을 포함한 분석적인 3D 표면 모델을 제공하는 사이클로피안 아이 모델(cyclopean eye model)을 채택했습니다. 이러한 기하학적 기반과 학습된 스테레오 특성을 결합하여 기존 방법의 장점을 모두 활용하는 시스템을 구현했습니다. 이 모델은 기존 데이터 기반 방법들과 동등하거나 더 나은 성능을 보여주며, 비주얼 품질에서 우수함을 높이 평가합니다.

- **Technical Details**: 논문에서 제안하는 하이브리드 기하학-학습 프레임워크는 기하학적 추론과 딥러닝의 적응력을 결합한 새로운 스테레오 비전 접근 방식을 제공합니다. 여기서, 사이클로피안 아이 모델은 깊이 불연속성과 가림을 처리할 수 있도록 설계되어 있으며, 불투명한 표면에 대해서는 사이클로피안 공간 좌표(e,x) 당 하나의 불일치(disparity) 솔루션을 제공합니다. 또한, 전제 단안 표면 모델을 활용하여 텍스처가 부족한 영역의 가림을 보완하며, 전통적인 매칭 기법으로는 해결하기 어려운 깊이 추정을 개선합니다.

- **Performance Highlights**: 제안된 스테레오 비전 시스템은 최신 데이터 기반 방법의 성능에 필적하거나 이를 초과하는 결과를 보여주며, 특히 깊이 맵의 비주얼 품질에서 개선된 성능을 발휘합니다. 이러한 품질 향상은 가상 현실 및 로봇 분야에서의 적용 가능성을 높이며, 중요한 오류를 줄이는 데 기여할 수 있습니다. 전체적으로, 3D 표면의 기하학적 특성을 이해하고 모델링하는 것이 컴퓨터 비전 연구에 유익하다는 것을 실증하는 것을 목표로 합니다.



### Adaptive Keyframe Sampling for Long Video Understanding (https://arxiv.org/abs/2502.21271)
Comments:
          CVPR2025

- **What's New**: 본 논문은 비디오 이해를 위한 Adaptive Keyframe Sampling (AKS)이라는 새로운 알고리즘을 제안합니다. 이는 기존의 비디오 기반 MLLM들이 비디오 토큰을 선택하는 방식에서 발생하는 정보 손실 문제를 해결하고자 합니다. AKS는 고정된 수의 비디오 토큰으로 유용한 정보를 극대화하는 데 중점을 두며, 키프레임 선택을 통해 비디오의 관련성과 커버리지를 동시에 고려합니다.

- **Technical Details**: 이 연구는 키프레임 선택을 최적화 문제로 모델링하여, 키프레임과 질문 간의 관련성을 측정하고, 키프레임 집합이 비디오에서 유용한 정보를 얼마나 잘 커버하는지를 평가합니다. 전체 알고리즘은 비디오와 텍스트 입력을 결합하여 최적의 키프레임을 선택하고, 고품질의 비주얼 데이터를 기반으로 MLLM의 성능을 향상시키는데 초점을 맞추고 있습니다. AKS는 LLaVA-Video와 같은 기존 MLLM에 통합되어 성능 개선을 확인합니다.

- **Performance Highlights**: AKS는 LongVideoBench와 VideoMME에서 비디오 질문 응답의 정확도를 개선하는 데 성공하였습니다. 본 연구에서 제안된 알고리즘은 고품질 키프레임의 발견 덕분에 MLLM의 전반적인 성능을 향상시켰음을 입증하였습니다. 실험 결과, AKS가 통합된 모델은 모든 테스트에서 일관된 정확도 향상을 보여주며, 이로 인해 새로운 벤치마크 기록을 세우게 되었습니다.



### Foundation Models -- A Panacea for Artificial Intelligence in Pathology? (https://arxiv.org/abs/2502.21264)
Comments:
          50 pages, 15 figures and an appendix (study protocol) which is previously published, see this https URL

- **What's New**: 최근 인공지능(AI)의 역할은 진단 지원에서 전체 슬라이드 이미지(WSI)에서 예측적 형태학 패턴을 발견하는 방향으로 진화하였습니다. 본 연구에서는 전 세계 11개 국가의 15개 사이트에서 수집된 100,000건 이상의 코어 바늘 생검 데이터를 활용해 전립선암 진단 및 Gleason 등급 평가를 위한 AI의 임상 성능을 집중적으로 탐구하였습니다.

- **Technical Details**: 우리는 다중 인스턴스 학습(multiple instance learning) 프레임워크 내에서 두 가지 기초 모델(foundation models, FMs)과 완전한 엔드 투 엔드(task-specific, TS) 모델을 비교하였습니다. 결과적으로, FMs가 데이터가 부족한 상황에서 유용성을 보여주었지만, 충분한 라벨이 있는 훈련 데이터가 제공될 때 TS 모델과 유사한 성능을 보였으며, 때로는 초과하기도 했습니다.

- **Performance Highlights**: FMs는 TS 모델보다 최대 35배 많은 에너지를 소모했으며, 이는 지속 가능성에 대한 우려를 불러일으킵니다. TS 모델은 임상적으로 중요한 오등급, 난해한 형태학의 오진 및 다양한 WSI 스캐너 간의 변동성을 크게 줄였습니다. 연구 결과는 FMs가 빠른 프로토타이핑 및 연구에 유리하지만, 임상 적용에 있어 유니버설 솔루션으로서의 역할은 불확실하다는 점을 강조하고 있습니다.



### Anatomically-guided masked autoencoder pre-training for aneurysm detection (https://arxiv.org/abs/2502.21244)
Comments:
          11 pages, 3 figures

- **What's New**: 이번 연구는 뇌동맥류 감지를 위한 3D 비전 트랜스포머 모델의 사전 훈련(pre-training) 전략을 제안합니다. 일반적으로 사용되는 주석이 없는 두부 CT 스캔 데이터를 활용하여 모델의 사전 훈련을 시행하고, 이후 실제 감지 작업을 위해 미세 조정(fine-tuning)합니다. 이러한 접근 방식은 데이터가 제한적인 상황에서 모델의 일반화 능력을 향상시키는 데 기여합니다.

- **Technical Details**: 이 연구에서는 마스크 자동 인코더(MAE) 방식을 변형하여, 3D 주의(attention) 메커니즘을 사용하여 계산 가능성을 높이고, 마스킹된 패치를 혈관에 가까운 영역으로 제한하여 뇌동맥류 발생 가능성에 집중합니다. CT 스캔의 강도 값뿐만 아니라 각 복셀과 가장 가까운 혈관 간의 거리를 나타내는 거리 맵도 재구성합니다. 이는 모델이 뇌혈관에 대한 정보와 시각적 패턴을 연관 지을 수 있도록 돕습니다.

- **Performance Highlights**: 제안된 접근 방식은 다른 최신 모델(SOTA) 대비 0.5의 허용 오차 조건에서 4-8%의 절대 민감도(Sensitivity)를 개선하며, 분포 내 데이터에 대해서는 기본 감지 성능을 유지하는 반면, 분포 외 데이터에 대해서는 더욱 우수한 성능을 보여줍니다. 이러한 결과는 본 연구의 모델이 다양한 의료 환경에서도 효과적으로 적용될 수 있다는 가능성을 시사합니다.



### Towards long-term player tracking with graph hierarchies and domain-specific features (https://arxiv.org/abs/2502.21242)
- **What's New**: 이 논문에서는 팀 스포츠 분석에서 선수의 장기 추적 문제를 해결하기 위한 새로운 방법인 SportsSUSHI를 소개합니다. 이 방법은 저지 번호(jersey number), 팀 ID(team ID), 필드 좌표(field coordinates)와 같은 도메인 특정 기능을 활용하여 추적 정확성을 향상시킵니다. 또한, 새로운 하키 추적 데이터셋을 제안하여 SportsSUSHI의 성능을 평가할 수 있는 기회를 제공합니다.

- **Technical Details**: SportsSUSHI는 계층적 그래프 기반 접근 방식을 사용하여 선수의 추적을 수행하는데, 이는 기존의 SUSHI 방법론을 기반으로 합니다. 이 방법은 온라인 단기 추적과 장기 추적 문제를 동시에 다루며, 이를 위해 도메인 특화 기능을 통합합니다. 실험 결과, SoccerNet 데이터셋 및 새롭게 제안된 하키 데이터셋에서 높은 성능을 보였음을 확인할 수 있습니다.

- **Performance Highlights**: SportsSUSHI는 실험을 통해 기존 방법들에 비해 높은 추적 정확성을 보여줍니다. 특히, 도메인 특정 기능을 포함했을 때의 성능 향상이 뚜렷하게 나타났습니다. 새로운 하키 데이터셋은 매우 긴 비디오_sequences를 포함하고, 선수의 특성이 보호 장비에 가려지는 도전적인 조건에서도 일관된 결과를 제공합니다.



### The PanAf-FGBG Dataset: Understanding the Impact of Backgrounds in Wildlife Behaviour Recognition (https://arxiv.org/abs/2502.21201)
Comments:
          Accepted at the IEEE / CVF Computer Vision and Pattern Recognition Conference 2025

- **What's New**: 이 논문에서는 야생 침팬지의 행동을 기록한 PanAf-FGBG 데이터셋을 소개합니다. 350개 이상의 카메라 위치에서 촬영된 20시간의 비디오로 구성되어 있으며, 각 침팬지 비디오와 해당 비디오에서 침팬지가 없는 배경 비디오를 쌍으로 제공합니다. 이를 통해 행동 인식을 위한 데이터셋에서 배경의 역할과 이를 통해 모델 성능을 개선할 수 있는 방법을 명확히 평가할 수 있는 기회를 제공합니다.

- **Technical Details**: PanAf-FGBG 데이터셋은 각 침팬지 행동 비디오(전경 비디오)와 같은 카메라 위치에서 촬영된 관련 배경 비디오(배경 비디오)의 쌍으로 구성됩니다. 이 데이터셋은 두 가지 실험 조건으로 나뉘어져, 겹치는 카메라 위치 및 분리된 카메라 위치을 포함하며, 이를 통해 실질적인 배경 영향 분석이 가능합니다. 또한, 다양한 행동 주석과 메타 데이터를 포함하여, 인-디스트리뷰션(in-distribution) 및 아웃-오브-디스트리뷰션(out-of-distribution) 상황을 평가하는 데 필수적인 정보를 제공합니다.

- **Performance Highlights**: 저자들은 배경만으로 훈련된 모델이 즉각적으로 65%의 성능을 보여준다고 주장합니다. 또한, latent-space 배경 정규화 기술을 도입하여, OOD 시나리오에서 컨볼루션 모델의 mAP(Mean Average Precision)이 +5.42%, 트랜스포머 기반 모델의 경우 +3.75% 향상된 성능을 달성했다고 보고했습니다. 이를 통해 배경 기간, 즉 전경 비디오 내 배경 프레임 수의 영향이 행동 인식 모델에 미치는 영향이 밝혀졌습니다.



### Towards High-performance Spiking Transformers from ANN to SNN Conversion (https://arxiv.org/abs/2502.21193)
- **What's New**: 이번 논문은 Spiking Neural Networks(SNNs)에 대한 혁신적인 접근 방식을 제시하고 있습니다. 특히, Transformers를 SNN으로 변환하는 과정에서 비선형 모듈에 대한 도전 과제를 해결하기 위해 Expectation Compensation Module(ECM)을 도입했습니다. 이러한 새로운 모듈은 이전 T 시간 단계의 정보를 활용하여 현재 시간 단계에서의 예상 출력을 계산함으로써 변환의 정확도를 유지합니다.

- **Technical Details**: SNN은 생물학적 신경망의 메커니즘을 모방하여 설계된 신경망 모델로, 특정 임계값에 도달했을 때만 스파이크를 방출합니다. 이 연구에서 제안된 멀티 스레시홀드 뉴런과 병렬 매개변수 정규화는 고정밀도를 요구하는 큰 시간 단계를 줄여, 네트워크 지연(latency) 및 전력 소비를 감소시키는 것을 목표로 합니다. 이 접근 방식은 ImageNet1k 데이터셋에서 정확도와 에너지 소비 모두에서 기존 모델을 초월하는 성과를 냈습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 최상급의 성능을 발휘하여 상위-1 정확도 88.60%를 달성했습니다. 이 과정에서 기존 ANN 대비 오직 1%의 정확도 손실을 보였고, Transformer의 전력 소모는 약 35%로 줄였습니다. 이러한 성과는 SNN으로 성공적으로 변환된 최초의 인공지능 모델로, 복잡한 데이터셋에서 높은 정확도와 낮은 지연, 저전력 소비를 모두 달성하고 있습니다.



### HQColon: A Hybrid Interactive Machine Learning Pipeline for High Quality Colon Labeling and Segmentation (https://arxiv.org/abs/2502.21183)
- **What's New**: 이 논문에서는 고해상도 대장 세분화를 위한 최초의 완전 자동화된 방법을 제시하고 있습니다. 기존의 오픈 소스 도구인 TotalSegmentator가 대장의 복잡한 형태로 인해 정확성을 보장하지 못하는 문제를 해결하기 위해, 연구진은 새로운 세분화 모델을 개발하는데 필요한 데이터세트를 생성했습니다. 이 데이터셋은 CT 대장조영술 이미지의 435개 레이블이 있는 샘플로 구성되어 있으며, 이를 기반으로 nnU-Net 모델을 훈련하여 세분화의 정확성을 획기적으로 향상시켰습니다.

- **Technical Details**: 研究者들은 825명의 대장 내시경 검사가 필요한 환자들을 대상으로 한 공개된 CTC 데이터셋을 사용하여 연구를 진행했습니다. 초기 세분화를 위해 용이한 주석 생성을 위한 전통적인 방법을 적용하고, 대장 내용물의 세분화를 위한 RootPainter라는 인터랙티브 머신러닝 방법을 활용하였습니다. 모델을 학습하는 과정에서 3D nnU-Net v2를 사용하고, 입력 및 마스크 이미지 두 가지 데이터셋을 통해 대장 전체를 포함한 세분화 모델을 훈련시켰습니다.

- **Performance Highlights**: 최종적으로, 연구진의 모델은 평균 대칭 표면 거리(ASSD) 0.2 mm, 95번째 백분위수 하우스도르프 거리(HD 95%) 1.0 mm를 기록하며, 기존의 TotalSegmentator의 결과보다 상당히 개선된 성능을 보여줍니다. 이 모델은 고해상도 대장 세분화에서 훨씬 더 높은 정확성을 제공하며, 연구진은 해당 훈련된 모델과 코드도 민간에 공개하여 접근성을 높였습니다.



### Adaptive Illumination-Invariant Synergistic Feature Integration in a Stratified Granular Framework for Visible-Infrared Re-Identification (https://arxiv.org/abs/2502.21163)
- **What's New**: 이번 논문에서는 가시-적외선 개인 재식별(Visible-Infrared Person Re-Identification, VI-ReID) 문제를 해결하기 위해 Adaptive Modality Interaction Network(AMINet)을 제안합니다. AMINet은 다양한 조명 조건과 차단을 극복하기 위한 다중 해상도(feature extraction)의 기술을 적용하여 이미지에서 신원 속성을 포괄적으로 포착합니다. 이 모델은 RGB-IR 양식 간의 간극을 효과적으로 매핑하기 위해 심층 intra-modal 및 cross-modal 정렬을 위한 상호작용 특징 융합(interactive feature fusion) 전략을 통합합니다.

- **Technical Details**: AMINet은 전신 이미지(full-body)와 상체 이미지(upper-body)에서 특징을 추출하여 차단과 배경 잡음에 대한 강인성을 높입니다. 모델은 단계 일치(phase congruency)를 이용하여 조명 변화에 강한 특징을 추출하며, 다양한 스케일에 걸쳐 특징 분포를 정렬하기 위해 적응형 다중 스케일 커널(Multi-scale kernel MMD)을 활용합니다. 이러한 기술들은 VI-ReID의 정확성을 크게 향상시킵니다.

- **Performance Highlights**: SYSU-MM01 벤치마크 데이터셋에서 AMINet은 Rank-1 정확도(Rank-1 accuracy) 74.75%를 기록하며, 기존의 기준 모델보다 7.93% 상승한 결과를 보였습니다. 현재의 최첨단 기술 대비 3.95% 더 우수한 성능을 발휘함으로써, 제안된 접근 방식의 효율성을 입증합니다.



### A Review on Generative AI For Text-To-Image and Image-To-Image Generation and Implications To Scientific Images (https://arxiv.org/abs/2502.21151)
- **What's New**: 이 논문은 생성적 AI(Generative AI)의 범위 내에서 텍스트-이미지(text-to-image) 및 이미지-이미지(image-to-image) 생성의 최신 상태를 조사합니다. 특히, Variational Autoencoders, Generative Adversarial Networks, Diffusion Models와 같은 세 가지 주요 아키텍처에 대한 비교 분석을 제공합니다. 이러한 비교를 통해 각 아키텍처의 핵심 개념, 건축 혁신 및 과학적 이미지 이해를 위한 실용적인 강점과 한계를 설명합니다.

- **Technical Details**: 이 연구에서는 Variational Autoencoders, Generative Adversarial Networks, Diffusion Models의 구조적 혁신을 상세히 분석합니다. 이러한 모델들은 각각 고유한 방식으로 이미지 생성을 수행하며, 텍스트 기반의 이미지 생성에 효과적으로 적용됩니다. 핵심 기술인 Latent 공간(Latent space) 표현, adversarial training, 그리고 노이즈 제거 과정을 통한 이미지 향상 기술이 주요 테마로 다뤄집니다.

- **Performance Highlights**: 생성적 AI 분야에서의 이들 모델들은 각각 다양한 상황에서의 성능을 발휘합니다. 예를 들어, Generative Adversarial Networks는 높은 품질의 이미지를 생성하는 데 강점을 보이며, Diffusion Models는 더 자연스러운 이미지를 생성하는 데 유리합니다. 그러나 각 모델은 특정 응용 분야에 따라 다른 한계점과 도전 과제를 가지고 있어 추가 연구가 필요합니다.



### Fast and Accurate Gigapixel Pathological Image Classification with Hierarchical Distillation Multi-Instance Learning (https://arxiv.org/abs/2502.21130)
Comments:
          11 pages, 4 figures, accepted by CVPR2025

- **What's New**: 이번 논문에서는 계층적 증류 다중 사례 학습(HDMIL) 프레임워크를 제안하여 고해상도의 전체 슬라이드 이미지(WSI)에서 불필요한 패치를 배제함으로써 빠르고 정확한 분류를 달성합니다. HDMIL은 동적 다중 사례 네트워크(DMIN)와 경량 인스턴스 사전 스크리닝 네트워크(LIPN)로 구성되어 있습니다. 이러한 접근 방식은 기존의 다중 사례 학습 알고리즘의 성능 저하 없이 추론 시간을 크게 단축시킵니다.

- **Technical Details**: HDMIL는 고해상도 WSI에서 인스턴스 수준의 특징을 추출하여 DMIN을 교육하고, 주의 점수 기반 마스크를 생성하여 불필요한 패치를 식별합니다. LIPN은 저해상도 WSI의 각 패치의 중요성을 예측하도록 훈련되어, 체계적으로 불필요한 패치를 제거하여 분류 성능을 높입니다. 또한, Chebyshev 다항식 기반의 Kolmogorov-Arnold 분류기를 설계하여 분류 정확성을 더욱 향상시키고 있습니다.

- **Performance Highlights**: 세 가지 공개 데이터셋에서 광범위한 실험을 통해 HDMIL가 이전의 최신 기술들을 초월함을 입증하였습니다. 예를 들어, Camelyon16 데이터셋에서 HDMIL는 AUC 90.88%와 정확도 88.61%를 달성하며, 각각 3.13%와 3.18% 향상되었습니다. 또한, HDMIL는 추론 시간을 28.6% 줄이는 데 성공하여 성능과 효율성의 균형을 이루었습니다.



### SEE: See Everything Every Time -- Adaptive Brightness Adjustment for Broad Light Range Images via Events (https://arxiv.org/abs/2502.21120)
- **What's New**: 이번 연구에서는 일반적인 조명 조건에서 이벤트 카메라를 활용하여 이미지의 밝기를 효과적으로 조정하는 방법을 제안합니다. 기존 연구가 주로 저조도(低照度) 이미지 개선에 집중했던 반면, 우리는 다양한 조명 조건에서의 이미지 품질 향상 문제를 다룹니다. 새로운 데이터셋 SEE-600K를 통해 202가지 장면에서 4가지 조명 조건을 포함한 총 610,126개의 이미지를 수집하며, 이를 바탕으로 다채로운 조명 변화에 적응할 수 있는 새로운 방법론을 제공합니다.

- **Technical Details**: SEE-Net이라는 효율적인 신경망 구조를 설계하여 밝기 조정을 수행합니다. 이 구조는 이벤트 신호를 기반으로 하여 이미지 밝기를 개선하며, 밝기 프롬프트를 통해 조명 조절을 구현합니다. 특히, 크로스 집합(cross-attention) 메커니즘을 활용하여 이미지의 컬러를 포착하고 조정하여, 픽셀 단위에서의 세밀한 조정이 가능합니다. 연구 결과, SEE-600K 데이터셋을 통해 제안하는 방법의 성능이 저조도에서의 이미지 개선 뿐 아니라, 일반 또는 고조도 이미지를 위한 확장에도 강력함을 보여줍니다.

- **Performance Highlights**: 시험 결과, SEE-Net은 저조도 개선 데이터셋(SDE)과 SEE-600K 데이터셋 모두에서 뛰어난 성능을 발휘했습니다. 특히, 다양한 조명 조건에서도 원활한 밝기 조정이 가능하여 이미지 후처리(processing) 능력의 향상을 도모합니다. 이 접근법은 향후 고급 이미지 처리와 같은 다양한 응용 분야에도 큰 잠재력을 가질 것으로 기대됩니다.



### FlexDrive: Toward Trajectory Flexibility in Driving Scene Reconstruction and Rendering (https://arxiv.org/abs/2502.21093)
- **What's New**: 본 연구에서 제안하는 주요 혁신은 Inverse View Warping (IVW) 기법입니다. 이 기법은 차량의 경로와 상관없이 바깥쪽 뷰(view)에서의 고품질 이미지 생성을 가능하게 하여, 관련 씬의 재구성을 위한 작업에 대한 품질을 향상시킵니다. 또한, Depth Bootstrap (DB) 전략을 도입하여 최적화 과정에서 밀집된 심도(depth) 맵을 얻어내는데 성공하였습니다. 이 두 가지 기법의 결합으로 FlexDrive라는 프레임워크를 구축하여, 주행 시나리오의 새로운 시각화 가능성을 제시하고 있습니다.

- **Technical Details**: FlexDrive의 아키텍처는 IVW와 DB의 두 주요 컴포넌트로 구성되어 있습니다. IVW는 바깥쪽 가상 뷰포인트에서의 이미지 렌더링 품질을 개선하기 위해 고품질 비주얼 슈퍼비전을 생성합니다. DB는 정확하고 밀집된 깊이 맵을 제공하여 IVW의 효과성을 강화하는 역할을 합니다. 또한, 동적 객체 모델링을 향상시켜 바깥쪽 뷰포인트에서의 동적 씬 재구성을 지원합니다.

- **Performance Highlights**: Waymo Open 데이터셋과 개발된 벤치마크에서 제안된 방법은 기존 방법들에 비해 우수한 성능을 보였습니다. 특히, 기존 경로 인식(in-path) 및 바깥쪽 뷰(out-of-path) 재구성 및 렌더링 성능에서 눈에 띄는 향상을 이루었습니다. 이 연구는 차량 시뮬레이션에 있어 고품질 렌더링을 제공할 수 있는 새로운 이정표를 마련하였습니다.



### BST: Badminton Stroke-type Transformer for Skeleton-based Action Recognition in Racket Sports (https://arxiv.org/abs/2502.21085)
Comments:
          8 pages (excluding references). The code will be released in a few months

- **What's New**: 이 논문에서는 배드민턴 방송 경기의 각 선수가 라켓 스윙을 하는 장면을 추출하기 위해 새로운 비디오 분할 전략을 제시합니다. 이 분할된 프레임은 두 개의 기존 모델에 의해 처리되며, 하나는 Human Pose Estimation (HPE) 모델로 선수의 신체 관절을 얻고, 다른 하나는 셔틀콕 궤적 탐지를 위한 모델입니다. 우리가 제안하는 Badminton Stroke-type Transformer (BST)를 통해 싱글 경기에서 선수의 스윙 유형을 분류할 수 있습니다.

- **Technical Details**: 세션 3에서는 경기 비디오를 클립으로 분할하는 방법을 설명합니다. 각 클립에는 두세 번의 스윙이 포함되어 있으며, 이 과정에서 선수의 포즈, 위치와 셔틀콕의 궤적을 추출합니다. 이를 통해 생성된 데이터는 Transformer 기반 아키텍처에 입력되어 분류 결과를 도출합니다. 이 연구에서는 각 랠리와 스윙 각각을 어떻게 정의하고 세그먼트하는지를 상세히 설명합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법이 공개된 가장 큰 배드민턴 비디오 데이터셋인 ShuttleSet에서 이전 최첨단 기술을 초월하는 성능을 보여줍니다. 데이터셋에서 효과적으로 셔틀콕 궤적 정보를 활용함으로써 배드민턴 스윙 유형 분류 성능이 현저하게 향상되었습니다. 이러한 결과는 라켓 스포츠 동작 인식에서의 셔틀콕 궤적 정보 활용이 중요한 추세가 될 것임을 시사합니다.



### Training-free and Adaptive Sparse Attention for Efficient Long Video Generation (https://arxiv.org/abs/2502.21079)
- **What's New**: 이 논문에서는 고품질의 긴 영상을 생성하는 차세대 기법인 AdaSpa를 제안합니다. AdaSpa는 Dynamic Pattern과 Online Precise Search를 결합한 비지도 학습(data-free) 방식의 희소(attention) 기법으로, 비디오 생성 속도를 획기적으로 높이는데 중점을 두고 있습니다. 이를 통해, 디퓨전 트랜스포머(DiTs) 기반의 비디오 생성에서 발생하는 계산 비용을 유의미하게 줄일 수 있습니다.

- **Technical Details**: AdaSpa는 기본적으로 블록화(blockified) 패턴을 활용하여 DiTs의 계층적 희소성(hierarchical sparsity)을 포착합니다. 또한, Fused LSE-Cached Search를 도입하여 디퓨전 단계에 걸쳐 희소 인덱스를 정확하게 실시간으로 식별할 수 있도록 합니다. 이 방법은 입력, 레이어 및 헤드에 따라 다르게 나타나는 희소 패턴을 효과적으로 처리합니다.

- **Performance Highlights**: 광범위한 실험을 통해, AdaSpa가 여러 모델에서 비디오 품질을 유지하면서도 상당한 속도 향상을 가져온다는 것을 확인했습니다. 기존의 Static 및 Dynamic 패턴 방법들에 비해 성능이 월등히 개선되어, AdaSpa는 효율적인 비디오 생성 접근 방식으로서 강력하고 확장 가능하다는 것이 입증되었습니다.



### Enhancing deep neural networks through complex-valued representations and Kuramoto synchronization dynamics (https://arxiv.org/abs/2502.21077)
- **What's New**: 이 논문에서는 Kuramoto dynamics를 활용하여 인공 신경망에서 개체 인식을 강화하는 새로운 접근법을 제안합니다. 특히, 복소수 표현(complex-valued representation)과 동기화(synchrony) 메커니즘을 결합하여 멀티 객체를 효과적으로 인식할 수 있도록 합니다. 저자들은 두 가지 아키텍처를 비교하였고, 이러한 모델들은 기존의 모델들에 비해 우수한 성능을 보여 정확성과 일반화 능력을 향상시키는 잠재력을 시사합니다.

- **Technical Details**: 저자들은 복소수 단위를 포함한 계층적 모델인 KomplexNet을 설계하였습니다. 이 모델은 Kuramoto dynamics를 이용하여 초기 레이어에서 동기화된 상태를 유도하고 이를 통해 영상 특징을 결합합니다. 또한 피드백 연결을 통해 동기화를 정교화시키는 방법을 도입하여 객체 표현의 구조를 강화합니다. 이러한 접근법은 첨단 심층 학습 아키텍처의 견고성과 일반화 능력을 높이는데 기여합니다.

- **Performance Highlights**: KomplexNet은 다중 객체 이미지 작업에서 특히 뛰어난 성능을 보여주었습니다. 저자들은 이 모델이 Gaussian noise에 강하고, 전이 학습(out-of-distribution classification) 문제에서도 우수한 일반화 능력을 보인다는 점을 강조합니다. 피드백 연결을 추가한 KomplexNet은 동기화의 정확성을 높여 더욱 강력한 성능을 보였으며, 이는 모델의 일반화 능력과 로버스트니스에 긍정적인 영향을 미쳤습니다.



### Spatial Reasoning with Denoising Models (https://arxiv.org/abs/2502.21075)
Comments:
          Project website: this https URL

- **What's New**: 본 연구에서는 Denoising Generative Models를 통해 연속 변수 집합에 대한 추론을 수행할 수 있는 Spatial Reasoning Models (SRMs) 프레임워크를 소개합니다. SRMs는 관측된 변수에 대한 관찰을 통해 관찰되지 않은 변수의 연속 표현을 추론합니다. 현재의 생성 모델들은 복잡한 분포에서 환각(hallucination) 문제로 인해 한계를 보이는데, 이를 측정하기 위한 벤치마크 작업을 제안합니다.

- **Technical Details**: 조건부 생성 모델은 복잡하고 다모드 분포를 모델링 할 수 있는 가능성을 제공하는데, 이는 고차원, 연속 데이터에서 학습된 의미적 구조를 활용하는 것을 포함합니다. 본 연구는 SRMs라는 새로운 프레임워크를 도입하여 상관관계가 있는 변수 간의 이유를 추론할 때 sequentialization의 중요성을 강조합니다. 또한, 이는 Denoising Network에 의해서 생성 순서가 예측 가능함을 보여주며, 특정 추론 작업의 정확도를 향상시킵니다.

- **Performance Highlights**: SRMs는 복잡한 추론을 위한 체계적 접근 방식을 제공하며, 생성 모델이 간단한 시각적 추론에는 능하지만, 복잡한 분포에 대해서는 환각 현상이 발생하는 것을 보여줍니다. 학습 샘플링 전략의 선택이 중요하며, 추론 작업에서 순차적 접근방식 사용이 환각을 줄이고 추론을 개선하는 데 기여합니다. 이 연구의 결과는 생성 모델들의 성능을 향상시키는 데 중요한 영향을 미칠 것으로 기대됩니다.



### Fast 3D point clouds retrieval for Large-scale 3D Place Recognition (https://arxiv.org/abs/2502.21067)
Comments:
          8 pages, 1 figures

- **What's New**: 이 논문에서는 3D 포인트 클라우드의 검색(retrieval) 효율성을 높이기 위해 차별 가능한 검색 색인(Differentiable Search Index, DSI) 기법을 도입하고, 이를 3D 데이터에 맞게 조정하는 방법을 제안하고 있습니다. 이를 통해 포인트 디스크립터(descriptors)를 바탕으로 1D 식별자(identifiers)를 생성하여 상수 시간(constant time)으로 직접적인 검색이 가능하도록 하고 있습니다. 또한, 시각 변환기(Vision Transformers)를 통합하여 포지셔널 및 시맨틱 인코딩(encoding)을 활용하는 방식입니다.

- **Technical Details**: 이 연구는 DSI를 활용하여 대규모 3D 포인트 클라우드의 검색을 가능하게 하는 새로운 방법론을 제안합니다. DSI는 텍스트 정보 검색을 위해 설계된 변환기 기반 접근법으로, 포인트 클라우드 데이터를 docid에 매핑(mapping)하는 단일 T5 모델을 사용합니다. 고급 인덱싱 전략을 소개하여 지리정보 및 힐버트 곡선(Hilbert curve) 기반 방법을 통합함으로써 문서의 표현을 강화합니다.

- **Performance Highlights**: 제안된 방법은 공개 벤치마크에서 평가되어, 속도 및 품질 측면에서 최신 방법과 비교되며 매우 우수한 검색 성능을 보여줍니다. 3D 포인트 클라우드의 지리적 인식을 위해 이 기법이 크게 기여할 것으로 기대됩니다. LiDAR 센서를 사용하여 수집된 데이터의 정확성을 높이는 다양한 응용 분야에 활용될 수 있습니다.



### FC-Attack: Jailbreaking Large Vision-Language Models via Auto-Generated Flowcharts (https://arxiv.org/abs/2502.21059)
Comments:
          13 pages, 6 figures

- **What's New**: 이번 연구에서는 뷰와 텍스트를 통합한 대형 비전-언어 모델(LVLMs)이 멀티모달 jailbreak 공격에 취약함을 발견했습니다. 이러한 공격은 모델이 해로운 콘텐츠를 생성하도록 유도하며, 이는 안전 위험을 초래할 수 있습니다. 우리는 자동 생성된 플로우차트를 기반으로 한 FC-Attack이라는 새롭고 효과적인 jailbreak 공격 방법을 제안하고, 이를 통해 LVLMs의 취약점을 더욱 조사합니다.

- **Technical Details**: FC-Attack은 사전 학습된 LLM을 미세 조정하여 단계 설명 생성기를 만들고, 이를 사용하여 해로운 쿼리에 대한 단계 설명을 생성합니다. 생산된 단계 설명은 세 가지 형태(수직, 수평, S자형)의 플로우차트로 변환되어 LVLMs에 주입됩니다. 이 공격의 실험은 Advbench 데이터셋을 사용하여 수행되었으며, 다양한 모델에서 90% 이상의 공격 성공률(ASR)을 달성했습니다.

- **Performance Highlights**: FC-Attack은 Gemini-1.5, Llava-Next, Qwen2-VL, InternVL-2.5 모델에서 기존 방법을 초월하는 수치의 ASR을 기록했습니다. 플로우차트 내 글꼴 스타일의 변화와 같은 다양한 요소가 공격 성과에 미치는 영향을 조사한 결과, Claude-3.5 모델에서 ASR이 4%에서 28%로 향상되는 성과를 거두었습니다. 방어 전략 또한 고려하여 AdaShield-A가 공격 성과를 크게 줄일 수 있었지만 유용성의 감소라는 대가를 치러야 했습니다.



### HoloMine: A Synthetic Dataset for Buried Landmines Recognition using Microwave Holographic Imaging (https://arxiv.org/abs/2502.21054)
Comments:
          under review

- **What's New**: 본 논문에서는 묻혀 있는 지뢰 탐지를 위한 새로운 합성 데이터 세트를 제안합니다. 이 데이터 세트는 41,800개의 마이크로웨이브 홀로그램 이미지를 포함하고 있으며, 연구자들이 지뢰 탐지 문제를 관찰하고 측정하며 해결할 수 있도록 지원합니다. 제안된 데이터 세트는 지뢰 및 기타 물체들을 포함하고 있기 때문에 기존의 탐지 방법에 비해 중요한 자원을 제공합니다. 이는 지뢰 탐지 분야에서의 진전을 촉진할 수 있는 잠재력을 지니고 있습니다.

- **Technical Details**: 제안된 데이터 세트는 2D 및 3D 형식의 홀로그램 이미지들로 구성되어 있으며, 여러 시나리오에서 훈련된 군용 모형을 포함하고 있습니다. 시뮬레이션된 홀로그램은 물리적으로 물체를 묻고 스캔하는 것보다 빠르고 효율적으로 생성될 수 있습니다. 홀로그램 이미지는 반전 알고리즘을 통해 생성되며, 이는 객체의 전자기 반사율을 분석하여 묻혀 있는 물체의 형태와 거리를 재구성하는데 활용됩니다. 이 접근 방식은 객관적으로 묻힌 물체를 표현할 수 있는 가능성을 제공합니다.

- **Performance Highlights**: 최신 딥 러닝 모델을 활용하여 2D 및 3D 데이터에 대한 성능을 평가하였습니다. 결과에 따르면, 모델들의 분류 성능에는 개선의 여지가 있으며, 어려운 과제임을 보여줍니다. 하지만, 합성 데이터의 제한에도 불구하고, 본 데이터 세트는 묻혀 있는 지뢰 탐지 연구에 귀중한 기여를 할 것으로 예상됩니다. 데이터와 모델은 GitHub에서 공개되어 있으며, 이는 연구 커뮤니티에 도움이 될 것입니다.



### Synthesizing Individualized Aging Brains in Health and Disease with Generative Models and Parallel Transpor (https://arxiv.org/abs/2502.21049)
Comments:
          20 pages, 9 figures, 6 tables, diffeomorphic registration, parallel transport, brain aging, medical image generation, Alzheimer's disease

- **What's New**: 이 논문은 Individualized Brain Synthesis (InBrainSyn)이라는 새로운 프레임워크를 소개하며, 이는 개별 뇌 영상을 기반으로 한 고해상도의 개인 맞춤형 장기적 MRI 스캔을 합성하는 기술입니다. InBrainSyn은 생성적 심층 템플릿 네트워크의 인구 수준 노화 궤적을 적응시키기 위해 병렬 전달 알고리즘을 사용하며, 알츠하이머병(AD) 및 정상 노화의 신경퇴행을 시뮬레이션합니다. 이 방법은 단일 기준 스캔만으로도 신뢰할 수 있는 3D 시공간 T1w MRI 스캔을 합성할 수 있는 장점을 제공하여 개별화된 노화 궤적을 생성합니다.

- **Technical Details**: InBrainSyn은 다양한 건강 상태에서 개별적인 뇌의 노화 궤적을 예측하기 위해, 인체의 해부학적 변화를 정확히 반영하는 매개변수화된 연속적 변환(또는 diffeomorphic transformation)을 이용합니다. 이 과정에서는 인구 수준의 데이터만 사용하여 생성된 궤적들을 통해 개인 맞춤형 노화를 예측하는 데 필요한 정보를 끌어냅니다. 또한, 기존의 MRI 영상의 처리에서 발생할 수 있는 강도 아티팩트(intensity artifacts)를 방지할 수 있도록 해부학적으로 타당한 이미지를 생성하는 데 중점을 두고 있습니다.

- **Performance Highlights**: InBrainSyn은 알츠하이머병 및 건강한 대조군 집단에 대한 정량적 및 정성적 평가를 통해 그 성능을 입증했습니다. 실험 결과, InBrainSyn은 정상 노화와 AD 사이의 신경해부학적 전이도 모델링할 수 있음을 보여주었습니다. 고유한 뇌 스캔을 기반으로 한 시뮬레이션이 각 개인에게 맞춤화된 결과를 내제화하여, 신경영상 분야에서의 임상적 이점과 연구에 대한 통찰력을 제공할 수 있음을 명확히 하고 있습니다.



### Data-free Universal Adversarial Perturbation with Pseudo-semantic Prior (https://arxiv.org/abs/2502.21048)
Comments:
          Accepted by CVPR 2025

- **What's New**: 본 논문에서는 랜덤 잡음에서 생성된 단일 변형을 사용하여 심층 신경망을 속이는 데이터 없는 전방향 적대적 변형(data-free Universal Adversarial Perturbation, UAP) 기법을 제안합니다. 기존의 데이터 없는 UAP 방법은 랜덤 잡음에서 기인한 의미적 정보 부족으로 인해 제한된 전이성을 가집니다. 이에 따라, 우리는 UAP로부터 반복적으로 의사 의미적 사전(pseudo-semantic prior)을 생성하는 새로운 접근 방식을 도입하여 데이터 없는 UAP 프레임워크 내에서 의미적 내용을 풍부하게 합니다.

- **Technical Details**: 제안하는 방법은 UAP가 본질적으로 잠재적인 의미적 정보를 포함하고 있다는 관찰에 기초하고 있습니다. 이러한 정보를 활용하여 다양한 의미를 포착하기 위해 지역 샘플링을 통해 UAP를 생성하며, 주목할 만한 사안들을 강조하기 위해 샘플 재가중화(sample reweighting) 기법을 도입합니다. 또한, 데이터 없는 UAP 방법에 일반적으로 효과가 없었던 입력 변환(input transformations)을 적용하여 블랙박스 환경에서 전이성을 향상시킵니다.

- **Performance Highlights**: 종합적인 실험 결과, 제안하는 방법은 이미지넷(ImageNet)에서 최첨단 성능을 달성하며, 기존의 데이터 없는 UAP 방법들과 비교하여 다양한 CNN 아키텍처에서 공격 전이성을 현저하게 개선합니다. 우리의 방법은 또한 데이터 의존적 UAP 방법들보다도 뛰어난 성능을 보이며, 이는 독창적인 전방향 공격 접근 방식의 가능성을 보여줍니다.



### MagNet: Multi-Level Attention Graph Network for Predicting High-Resolution Spatial Transcriptomics (https://arxiv.org/abs/2502.21011)
- **What's New**: 이 논문에서는 고해상도 HD 데이터의 정확한 예측을 위해 설계된 다단계 주의 그래프 네트워크(MagNet)를 제안합니다. MagNet은 여러 해상도에서 피쳐를 통합하여 정보 병목 현상을 극복하며, HD 수준의 유전자 발현 예측에 기여합니다. 또한 이 프레임워크는 기존의 저해상도 입력의 한계에서 벗어나 데이터를 효과적으로 처리할 수 있는 획기적인 방법론을 선보입니다.

- **Technical Details**: 제안된 MagNet은 여러 해상도의 정보(빈, 스팟 및 영역 수준)를 결합하기 위해 교차 주의 레이어(cross-attention layers)를 활용합니다. 이 모델은 GAT-Transformer 블록을 통해 이웃 정보(neighborhood information)를 집계하고, 경량화된 ResNet50을 사용하여 피쳐를 추출합니다. 이러한 접근 방식은 데이터의 고차원 유전자 발현 예측을 지원하기 위한 정교한 기능 표현을 제공합니다.

- **Performance Highlights**: MagNet은 개인적으로 수집된 신장 HD ST 데이터 세트와 공개된 대장암 HD ST 데이터 세트에서 기존의 ST 예측 모델들과 비교하여 체계적인 평가를 수행하였습니다. 결과적으로 MagNet은 저해상도 스팟 수준 및 고해상도 빈 수준에서 최첨단 성능을 달성하였으며, 이는 향후 고해상도 HD 수준의 공간 전사체학 연구 및 응용에 대한 새로운 벤치마크를 제공합니다.



### Soften the Mask: Adaptive Temporal Soft Mask for Efficient Dynamic Facial Expression Recognition (https://arxiv.org/abs/2502.21004)
Comments:
          8 pages, 3 figures

- **What's New**: 이 논문에서는 다이나믹 얼굴 표정 인식(Dynamic Facial Expression Recognition, DFER)을 위한 새로운 모델인 AdaTosk를 제안합니다. 이 모델은 자가 지도(self-supervised) 재구성 분기와 감독(supervised) 분류 분기를 통합하여 비효율적인 데이터 처리 문제를 해결하고자 합니다. 특히, AdaTosk는 시간적 중요성에 기반하여 가시적 토큰을 동적으로 마스킹하는 적응형(Adaptive) 마스킹 기법을 사용합니다.

- **Technical Details**: 기존 DFER 접근법의 문제는 비디오 내에서 나타나는 시간적 중복성과 불필요한 정보를 처리하는 데 있습니다. AdaTosk는 두 가지 주요 구성 요소, 즉 클래스 비관계(class-agnostic) 마스크와 클래스 의미론적(class-semantic) 마스크를 통해 중요한 표정 순간을 강조하고 의미적 중복성을 줄입니다. 이러한 방식으로 서로 다른 마스크를 적용하여 정보 손실을 방지하면서도 효율성을 크게 향상시킵니다.

- **Performance Highlights**: AdaTosk는 FERV39K, DFEW, MAFW 데이터셋에서 현재 최신 모델들보다 3%의 성능 향상을 달성하면서 약 10M 파라미터를 줄이고 계산 비용을 6G FLOPs로 낮추었습니다. 이러한 결과는 AdaTosk가 DFER 분야에서 경쟁력 있는 성능과 효율성을 모두 갖추고 있음을 보여줍니다.



### Towards Lossless Implicit Neural Representation via Bit Plane Decomposition (https://arxiv.org/abs/2502.21001)
- **What's New**: 이번 연구에서는 암묵적 신경 표현(Implicit Neural Representation, INR)의 모델 크기에 대한 상한선을 정량화하였습니다. 비트 정밀도가 증가함에 따라 모델 크기의 상한선이 기하급수적으로 증가하는 것을 확인하였습니다. 이를 해결하기 위해 비트 평면 분해(bit-plane decomposition) 방법을 제안하여 INR이 비트 평면을 예측함으로써 모델 크기의 상한을 줄이는 효과를 볼 수 있었습니다.

- **Technical Details**: 연구의 핵심 원리는 디지털 신호의 비트 정밀도가 모델 크기에 영향을 미친다는 것입니다. 비트 평면 분해 기법을 사용해 신호를 비트 평면으로 분해하고 이를 기반으로 손실 없는 표현을 생성하는 방식을 제안하였습니다. 실험을 통해 본 방법이 기존의 다른 네트워크보다 손실 없는 표현을 더 빠르게 수렴하게 함을 검증하였습니다.

- **Performance Highlights**: 제안한 방법은 16비트와 같은 고비트 심도 신호에서도 손실 없는 이미지를 표현할 수 있는 능력을 보여주었습니다. 또한, 비트 편향(bit bias) 현상을 발견하여 가장 중요한 비트가 더 빠르게 수렴한다는 점을 강조하였습니다. 이 연구는 손실 없는 압축, 비트 깊이 확장, 그리고 모델 양자화와 같은 새로운 응용 분야를 확장하는 데 기여하고 있습니다.



### LesionLocator: Zero-Shot Universal Tumor Segmentation and Tracking in 3D Whole-Body Imaging (https://arxiv.org/abs/2502.20985)
Comments:
          Accepted at CVPR 2025

- **What's New**: 본 논문에서는 LesionLocator라는 프레임워크를 소개하며, 이는 3D 의료 이미징에서 제로샷(longitudinal) 병변 추적 및 분할을 위한 최초의 엔드 투 엔드 모델입니다. 이 모델은 23,262개의 주석이 달린 의료 스캔 데이터셋과 다양한 병변 유형을 포함한 합성(longitudinal) 데이터를 활용하여, 실제 의료 이미징 문제에 대한 일반화 능력을 크게 향상시킵니다. LesionLocator는 모든 기존의 프롬프트 가능 모델을 병변 분할에서 약 10 dice 포인트 정도 초과하는 성능을 보여주며, 인체 수준의 성능을 달성했습니다.

- **Technical Details**: LesionLocator는 세 가지 핵심 구성 요소를 갖춘 프롬프트 가능한 프레임워크입니다. 첫째, 사용자의 프롬프트에 기반하여 다양한 병변을 제로샷 방식으로 분할하고, 둘째, 연속 스캔에서 병변을 효율적으로 추적하기 위한 프롬프트 전파 기능을 제공합니다. 셋째, 시계열 촬영 데이터에 대한 강건성과 일반화를 높이기 위한 합성(longitudinal) 데이터셋도 포함되어 있습니다. 이러한 접근方式는 3D 의료 데이터의 복잡한 공간 구조를 캡처하는 데 중요한 역할을 합니다.

- **Performance Highlights**: LesionLocator는 six out-of-distribution zero-shot 병변 분할 작업에서 뛰어난 일반화 성능을 보이며, 다양한 병변 유형과 신체 부위를 커버합니다. 이 모델은 단일 시점 분할 방법을 초과하는 성능을 발휘하며, 다중 시점 데이터에서 볼륨 추적에 대한 새로운 기준을 설정합니다. 또한, 연구 촉진을 위해 합성(longitudinal) 데이터셋과 모델 가중치를 공개할 예정이며, 이는 병변 추적을 위한 최초의 오픈 소스 모델로 자리잡게 될 것입니다.



### Distribution Prototype Diffusion Learning for Open-set Supervised Anomaly Detection (https://arxiv.org/abs/2502.20981)
Comments:
          Accepted by CVPR 2025

- **What's New**: 이번 연구에서는 Open-set Supervised Anomaly Detection (OSAD)에 대한 새로운 접근법으로 Distribution Prototype Diffusion Learning (DPDL) 방법을 제안합니다. 기존의 기법이 정상 샘플의 중요한 사전 정보를 간과하여 비효율적이라는 문제를 해결하고자 하였습니다. DPDL은 정상 샘플을 컴팩트하고 판별적인 분포 공간에 배치하여, 이전보다 더 효과적인 특징을 학습하도록 합니다.

- **Technical Details**: DPDL 방법은 다수의 가우시안 프로토타입을 생성하여 정상 샘플의 잠재적 표현 공간을 구성하고, 슈뢰딩거 다리(Schrödinger bridge) 프레임워크를 통해 정상 샘플과 프로토타입 간의 확산적 전이를 용이하게 합니다. 이는 정상 샘플이 비정상 샘플로부터 멀어지도록 하여 판별 용량을 향상시키고, 하이퍼구(Spherical) 공간에서 분산 특징 학습 방식도 결합하여 샘플 간 구별을 도모합니다.

- **Performance Highlights**: 실험 결과, DPDL 방법은 9개의 공용 데이터세트에서 최첨단 성능을 달성하며 기존 방법들보다 8.3% 이상의 성능 향상을 보였습니다. 특히, 단일 비정상 샘플 훈련 설정에서 AITEX, ELPV, Mastcam 데이터셋에서 두드러진 성과를 보고했습니다. 이러한 결과는 제안된 방법의 우수성과 일반화 능력을 분명하게 보여줍니다.



### Real-Time Aerial Fire Detection on Resource-Constrained Devices Using Knowledge Distillation (https://arxiv.org/abs/2502.20979)
- **What's New**: 이 연구에서는 기존의 CCTV 기반 화재 감지 시스템의 한계를 극복하기 위해 MobileViT-S를 기반으로 한 경량 화재 감지 모델을 제안합니다. 이 모델은 지식 증류(knowledge distillation)를 통해 더 강력한 교사 모델로부터 압축된 방식으로 개발되었습니다. 이를 통해 모델의 정확성과 효율성이 높아져 위성이나 UAV, IoT 장치의 배치에 적합하게 됩니다. 고해상도 화재 데이터를 처리하며 실시간 성능을 달성하는 점에서 중요한 발전을 이루었습니다.

- **Technical Details**: 제안된 모델은 MobileViT-S 아키텍처를 기반으로 하며, 교사-학생 모델 구조를 통해 기존의 깊은 학습 모델들의 복잡성을 줄이고 처리 속도를 높였습니다. 연구에서는 Grad-CAM을 사용하여 주요 화재 영역의 시각화 결과를 제시하여, 모델이 중요 정보에 제대로 집중하고 있음을 확인합니다. 이 모델은 세 가지 화재 분류 벤치마크에서 기존 방법 보다 더 높은 정확도를 기록하며, 컴팩트한 모델 크기를 유지합니다.

- **Performance Highlights**: 본 연구에서 제안한 화재 감지 모델은 기존 최첨단 모델 대비 0.44%의 정확도 향상과 2.00%의 전반적인 성능 개선을 보였습니다. 또한, 자원 제약이 있는 기기에서도 실시간으로 성능을 발휘하여 화재의 조기 감지에 기여하고 있습니다. 이로써 화재 예방법과 관리 방안의 효과성을 크게 향상시키는 결과를 도출했습니다.



### Fine-Grained Retrieval-Augmented Generation for Visual Question Answering (https://arxiv.org/abs/2502.20964)
- **What's New**: 이번 논문에서는 기존의 Visual Question Answering (VQA)에 외부 지식을 통합하여 정확성을 향상시킨 Knowledge-Based Visual Question Answering (KB-VQA) 접근법을 제안합니다. 특히, Retrieval-Augmented Generation (RAG) 기술을 활용하여 정보 검색과 생성을 조화롭게 결합하며, fine-grained retrieval 방식을 통해 기존의 시각적 정보 손실 문제를 해결하고자 합니다. 또한, Knowledge Unit(지식 단위)라는 새로운 개념을 도입하여 데이터베이스에서의 지식 검색을 보다 효율적으로 수행합니다.

- **Technical Details**: 제안된 Knowledge Unit Retrieval-Augmented Generation (KU-RAG) 방법은 MLLMs(다중 모달 대형 언어 모델)와 통합되어, 시각적 정보를 기반으로한 질문에 대한 답변을 생성하기 위해 구조화된 지식 단위를 활용합니다. 이 방법은 이미지와 텍스트 질문을 기반으로 지식을 검색하고, 이를 바탕으로 보다 정확하고 관련성 높은 답변을 생성하는 지식 수정 체계(Knowledge Correction Chain, KCC)와 연계됩니다. KU-RAG는 fine-grained retrieval을 통해 ultra-accurate knowledge mapping을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, KU-RAG 방법이 기존의 KB-VQA 방법들에 비해 최대 10%의 성능 향상을 이루었다는 것을 보여주었습니다. MLLM의 reasoning 능력을 강화하며, 복잡한 질문에 대해서도 더 정교한 답변을 제공할 수 있도록 합니다. 이러한 성과는 KB-VQA 벤치마크에서 확인되었으며, 연구 결과가 향후 연구나 실제 애플리케이션에 큰 기여를 할 것으로 기대됩니다.



### BadRefSR: Backdoor Attacks Against Reference-based Image Super Resolution (https://arxiv.org/abs/2502.20943)
Comments:
          5 pages,4 figures

- **What's New**: 이 연구에서는 Reference-based image super-resolution (RefSR) 모델에 대한 백도어 공격의 가능성을 탐구합니다. 기존의 단일 이미지 초해상도(SISR) 기술과 달리, RefSR은 추가적인 참조 이미지를 사용하여 고해상도 이미지를 복원합니다. 저자들은 BadRefSR이라는 새로운 공격 프레임워크를 제안하여, 참조 이미지에 트리거를 추가함으로써 모델을 손상시키는 방식을 선보입니다.

- **Technical Details**: BadRefSR은 트리거가 있는 참조 이미지를 사용하여 RefSR 모델에 백도어를 삽입하는 방법입니다. 이 모델은 낮은 해상도(LR) 입력 이미지 대신 참조 입력에서 트리거를 사용합니다. 다양한 실험을 통해 BadRefSR이 기존 백도어 공격 트리거에 적용 가능함을 보여주며, 낮은 병합 비율에서도 효과적인 공격을 유지합니다.

- **Performance Highlights**: BadRefSR 공격 기법은 정상적인 입력 이미지에 대해 정상이 아닌 고해상도 이미지를 출력하는 RefSR 모델을 생성합니다. 공격자가 지정한 타겟 이미지를 트리거된 입력 이미지에 출력하는 성능을 발휘합니다. 이 방법은 사용자가 보는 참조 이미지의 미세한 차이를 감추면서도, 원래 기능을 유지하도록 설계되어 있습니다.



### Less is More? Revisiting the Importance of Frame Rate in Real-Time Zero-Shot Surgical Video Segmentation (https://arxiv.org/abs/2502.20934)
- **What's New**: 이 연구는 AI 보조 수술의 실시간 비디오 분할에 대한 새로운 통찰력을 제공합니다. SAM2 모델의 효과를 담낭절제술(cholecystectomy) 과정에서 다양한 초당 프레임 수(frame rate)에서 평가함으로써, 프레임 속도가 분할 성능에 미치는 영향을 분석하였습니다. 특히, 놀랍게도 전통적인 평가 환경에서는 1 FPS의 낮은 속도가 25 FPS를 초과하는 성능을 보일 수 있다는 것을 발견하였습니다.

- **Technical Details**: 연구에서는 SAM2.1 Hiera Large 모델을 실시간 수술 비디오 분할에 활용하였습니다. 이 모델은 강력한 transformer 기반 아키텍처를 사용하여 공간적 및 시간적 일관성을 보장하며, 복잡한 수술 비디오 분석에 매우 적합합니다. 성능 평가에는 CholecSeg8k 데이터셋이 사용되었으며, 1, 10, 15, 20, 25 FPS 설정에서 SAM2의 성능을 조사하였습니다.

- **Performance Highlights**: 결과적으로 25 FPS 설정이 실시간 스트리밍 환경에서 우수한 성능을 나타냈으며, 특히 수술용 그레이퍼와 같은 동적 객체에 대해 시간적 일관성과 안정성을 제공하는 것으로 나타났습니다. 또한, 설문조사를 통해 전문가들이 높은 FPS로 제공되는 분할 마스크를 선호한다는 사실이 확인되어, AI 보조 수술의 실시간 평가가 중요함을 재확인하였습니다.



### Decoder Gradient Shield: Provable and High-Fidelity Prevention of Gradient-Based Box-Free Watermark Remova (https://arxiv.org/abs/2502.20924)
Comments:
          Accepted by CVPR 2025

- **What's New**: 이 논문에서는 기존의 box-free 워터마크 시스템에서 보호되지 않은 디코더의 취약점을 드러냅니다. 이 취약점은 공격자가 워터마크 제거 네트워크를 훈련하는 데 이용될 수 있음을 보여주고, 새로운 방어 메커니즘으로 디코더 그래디언트 실드(Decoder Gradient Shield, DGS)를 제안합니다. DGS는 워터마크 제거를 방지하는 보호층으로, 워터마크가 포함된 쿼리의 그래디언트 방향을 재조정하고 크기를 조절하여 훈련 손실이 수렴되지 않도록 합니다.

- **Technical Details**: 기존 연구들에서는 워터마크의 강인성 향상에 주목했지만, 본 논문에서는 그래디언트 기반 공격의 가능성을 강조합니다. DGS는 트레이너가 특정 수준까지 훈련 손실이 감소하지 않도록 방해하는 방향으로 그래디언트를 조정합니다. 이는 이전의 방어 메커니즘과는 달리 closed-form 솔루션을 제공하며, 워터마크 제거 공격을 효과적으로 막는 역할을 합니다.

- **Performance Highlights**: 제안된 DGS의 효과는 실험 결과를 통해 입증되었습니다. DGS를 사용할 경우, 워터마크 제거 네트워크의 훈련 손실이 DGS 없이 훈련했을 때보다 더 이상 감소하지 않음을 보여줍니다. 이를 통해 워터마크의 품질을 유지하면서도 공격을 효과적으로 방어할 수 있음을 확인했습니다.



### DiffBrush:Just Painting the Art by Your Hands (https://arxiv.org/abs/2502.20904)
- **What's New**: DiffBrush는 일반 사용자들이 손으로 그린 스케치를 통해 이미지 생성 및 편집을 가능하게 하는 새로운 접근 방식을 제안합니다. 기존의 텍스트 기반 T2I 모델과도 호환되며, 추가적인 훈련 없이도 사용자 맞춤형 이미지를 생성할 수 있습니다. DiffBrush는 사용자가 약간의 형태를 그릴 때 그에 맞는 인스턴스를 자연스럽게 생성합니다.

- **Technical Details**: DiffBrush는 Latent Diffusion Model (LDM)에서의 잡음 제거 과정에서 잠재적인 방향을 제어하여 실시간으로 사용자 필요에 맞춘 이미지를 생성합니다. 이 방법은 인스턴스의 색상, 의미 및 객체 제어를 지속적으로 조정하여 이미지 퀄리티를 높이는 데 초점을 맞춥니다. 또한, 사용자가 그린 마스크를 기반으로 초기 잡음을 정제하여 최종적인 이미지 배치를 개선합니다.

- **Performance Highlights**: DiffBrush는 사용자 친화적인 AI 그림 도구로, 다수의 텍스트-이미지(T2I) 모델과 호환됩니다. 주요 기여로는 색상과 인스턴스의 의미에 대한 조건적 가이던스 방법을 제안하며, 전통적인 방법에 비해 훈련 비용이 낮고 더 높은 정밀도의 제어를 가능하게 합니다. 이러한 기능을 통해 사용자는 보다 직관적으로 이미지 생성 및 편집을 수행할 수 있습니다.



### Adaptive Identification of Blurred Regions for Accurate Image Deblurring (https://arxiv.org/abs/2502.20880)
- **What's New**: 이번 연구에서는 AIBNet이라는 새로운 네트워크를 제안하여 이미지 디블러링(image deblurring) 과정에서 블러 처리가 된 영역을 적응적으로 식별합니다. 기존의 방법들은 블러 이미지의 다양한 영역에서의 열화 정도를 간과하여, 동일한 처리를 적용하고 결과적으로 인위적인 아티팩트를 초래합니다. AIBNet은 블러 영역을 차별적으로 복원할 수 있도록 설계된 구조를 가지며, 이로써 실제 데이터셋과 합성 데이터셋 모두에서 우수한 성능을 보입니다.

- **Technical Details**: AIBNet은 공간 특성 차별 처리 블록(SFDHBlock)과 고주파 특성 선택 블록(HFSBlock)으로 구성되며, SFDHBlock은 공간에서 특성을 향상시키고, HFSBlock은 고주파 정보를 선택적으로 유지합니다. SFDHBlock의 핵심 구성 요소인 공간 특성 향상 모듈(SFEM)은 블러 영역에서의 핵심 정보를 강조하고 비선명한 영역의 노이즈를 감소시키는 데 도움을 줍니다. 이러한 모듈 구성은 해석하여 블러 이미지의 다양한 열화 정도에 적응할 수 있도록 설계되었습니다.

- **Performance Highlights**: 광범위한 실험 결과, AIBNet은 이미지 디블러링에서 탁월한 성능을 보여줍니다. 특히, SFEM과 HFSBlock의 조합이 블러 영역에서의 특징을 효과적으로 강조하여 디블러링 품질을 크게 향상시켰습니다. 또한, 프로그레시브 트레이닝 전략을 도입하여 훈련 시 GPU 메모리 사용을 최소화하여 효율성을 높였습니다.



### egoPPG: Heart Rate Estimation from Eye-Tracking Cameras in Egocentric Systems to Benefit Downstream Vision Tasks (https://arxiv.org/abs/2502.20879)
- **What's New**: 이번 논문에서 제안하는 egoPPG는 egocentric 비전 시스템이 사용자의 심박수(Heart Rate, HR)를 추출할 수 있는 새로운 작업입니다. 이를 위해 Eye-tracking 비디오를 사용하여 HR을 추정하는 새로운 방법인 EgoPulseFormer를 제시합니다. 이 시스템은 추가적인 하드웨어 없이도 HR 값을 효과적으로 추적할 수 있도록 설계되었습니다.

- **Technical Details**: EgoPulseFormer는 사용자의 눈 주변에서 빛의 흡수가 발생하는 미세한 변화를 포착하여 photoplethysmogram (PPG)을 복원하는 방법입니다. 이 과정에서 infrared (IR) 조명을 활용하여 다양한 조명 조건에서도 안정적으로 작동합니다. 연구팀은 Project Aria 안경을 사용하여 수집된 13시간 이상의 Eye-tracking 비디오와 ECG 데이터를 통해 모델을 훈련하고 검증했습니다.

- **Performance Highlights**: EgoPulseFormer는 다양한 일상 활동 중에 연속적으로 HR을 정확하게 예측하며, MAE(Mean Absolute Error)가 8.82 bpm로 현재의 rPPG 모델보다 27% 더 낮은 오차를 기록했습니다. 또한 EgoExo4D의 프로피션시 추정 정확도를 14.1% 향상시키는 효과를 보였습니다. 이러한 연구 결과는 egocentric 시스템이 환경 인식과 생리학적 추적을 통합하여 사용자 행동을 보다 잘 이해할 수 있음을 보여줍니다.



### PathVG: A New Benchmark and Dataset for Pathology Visual Grounding (https://arxiv.org/abs/2502.20869)
Comments:
          10pages, 4figures

- **What's New**: 이번 논문에서는 Pathology Visual Grounding (PathVG)라는 새로운 벤치마크를 제안하여 병리학 이미지에서 유연하고 지역 중심의 검출이 가능하게 하였습니다. 기존의 AI 기반 병리학 진단 작업들은 미리 정의된 범주를 따르며 유연성이 부족했지만, PathVG는 다양한 표현 입력을 통해 이러한 문제를 해결합니다. PathVG는 또한 27,610장의 이미지를 포함한 RefPath라는 새로운 데이터셋을 생성하여, 언어와 시각적 정보를 결합한 다각적인 병리 표현을 제공합니다.

- **Technical Details**: PathVG는 다중 배율의 병리학 이미지를 사용하고, 병리학적 지식을 포함한 표현으로 특정 영역을 로컬라이즈합니다. 제안된 Pathology Knowledge-enhanced Network (PKNet)은 대형 언어 모델(LLMs)의 지식 강화 기능을 활용하여 병리학 용어를 시각적 정보로 변환하고, Knowledge Fusion Module (KFM)을 통해 지식 특성과 표현 특성을 융합합니다. PKNet은 4차원의 언어-기반 박스(b=(x,y,w,h)) 출력을 목표로 하며, 여러 모듈을 통해 서로 다른 정보를 조합하여 작업을 수행합니다.

- **Performance Highlights**: PathVG 벤치마크에서 제안된 방법은 최신 성능을 달성하였습니다. RefPath 데이터셋은 33,500개의 언어-기반 박스를 갖추고 있어 보다 세밀한 평가를 가능케 합니다. 이 실험에서는 병리학적 표현에 내재된 암묵적인 정보가 가장 큰 도전 과제였으며, PKNet은 이러한 도전을 극복하기 위해 설계되었습니다.



### MESC-3D:Mining Effective Semantic Cues for 3D Reconstruction from a Single Imag (https://arxiv.org/abs/2502.20861)
Comments:
          Published in CVPR 2025

- **What's New**: 이번 논문에서는 기존의 이미지 기반 3D 복원 방법의 한계를 극복하기 위해 MESC-3D라는 새로운 3D 구조 재구성 방법을 제안합니다. 이 방법은 3D 포인트 클라우드와 이미지 내의 의미적 속성을 연결하는 효과적인 의미 마이닝 모듈(Effective Semantic Mining Module, ESM)을 사용하여 정보의 선택적 추출을 구현합니다. 또한, 세 번째 차원에 대한 사전 지식을 활용하여 단일 이미지의 명암을 보완하는 3D 의미 사전 학습 모듈(3D Semantic Prior Learning Module, 3DSPL)을 통해 더욱 정교한 재구성이 가능해집니다.

- **Technical Details**: MESC-3D는 단일 이미지를 입력으로 받아, 이미지의 의미적 속성을 추출하고 이를 기반으로 포인트 클라우드를 재구성하는 과정에서 두 가지 주요 모듈, 즉 ESM과 3DSPL을 활용합니다. ESM은 포인트 클라우드가 이미지에서 추출된 의미적 정보를 선택하도록 돕고, 3DSPL은 인간의 경험에서 오는 사전 지식을 통합하여 잠재적인 의미적 정보를 보완합니다. 이러한 방식은 3D 구조의 복원을 보다 현실적이고 정확하게 수행할 수 있게 합니다.

- **Performance Highlights**: MESC-3D는 실험 결과 기존 방법들에 비해 재구성 품질과 견고성이 현저히 향상되었다는 것을 보여줍니다. 특히, 합성 데이터셋과 실제 데이터셋 모두에서 뛰어난 성능을 보이며, 제로샷(Zero-shot) 성능에서도 우수한 일반화 능력을 발휘하였습니다. 논문을 통해 제안된 방법의 유효성을 명확히 입증하는 다양한 실험이 진행되어, 최신 기술에 적용 가능성을 넓힐 것으로 기대됩니다.



### VLEER: Vision and Language Embeddings for Explainable Whole Slide Image Representation (https://arxiv.org/abs/2502.20850)
Comments:
          Under review

- **What's New**: 이 연구는 전통적인 비전-언어 모델(VLMs)의 적용 한계를 넘어, 전체 슬라이드 이미지(Whole Slide Image, WSI)에서 유용한 통찰력을 제공할 수 있는 가능성을 제시합니다. 특히 'Vision and Language Embeddings for Explainable WSI Representation (VLEER)'라는 새로운 방법론을 도입하여, VLM을 활용한 설명 가능한 WSI 표현을 실현합니다. 연구는 이 새로운 접근법이 기존의 비전 특징에 비해 WSI 분석에서 더 나은 성능을 나타낸다는 것을 입증하고 있습니다.

- **Technical Details**: VLEER은 두 가지 주요 구성 요소를 활용하여 설명 가능한 WSI 임베딩을 학습합니다: 과제와 관련된 병리학 키워드의 텍스트 풀과 사전 훈련된 병리학 VLM입니다. 이 방법은 WSI의 병리 이미지를 패치로 분할한 후, 해당 패치와 관련된 키워드를 매핑하여 비전 및 언어 패치 임베딩을 생성합니다. 이러한 임베딩은 최종적으로 서로 결합되어 WSI의 텍스트 기반 임베딩을 형성합니다.

- **Performance Highlights**: VLEER는 병리학 WSI 데이터셋 세 가지에서 체계적으로 평가되었으며, 전통적인 비전 모델에 비해 우수한 성능을 보였습니다. 또한, VLEER은 텍스트 기반 표기를 통해 결과를 해석할 수 있는 독특한 이점을 제공하여, 임상 병리학 분야에서의 설명 가능성을 높이는 중요한 기회를 창출했습니다.



### CoTMR: Chain-of-Thought Multi-Scale Reasoning for Training-Free Zero-Shot Composed Image Retrieva (https://arxiv.org/abs/2502.20826)
- **What's New**: 이번 연구에서는 Zero-Shot Composed Image Retrieval (ZS-CIR) 문제를 다루며, 새로운 Framework인 CoTMR을 제안합니다. CoTMR은 Chain-of-Thought (CoT) 및 Multi-scale Reasoning을 도입하여 기존의 방법들과의 차별점을 보이고 있습니다. 기존의 캡션 모델을 사용하지 않고, Large Vision-Language Model (LVLM)을 통해 통합된 이해와 추론을 제공합니다. 이 과정을 통해 ZS-CIR 작업에서 더 나은 성능과 해석 가능성을 발휘할 수 있게 됩니다.

- **Technical Details**: CoTMR은 복잡한 수정 질의(edited query)를 처리하기 위해 LVLM을 활용합니다. 본 시스템에서는 CIRCoT 라는 새로운 방법론을 통해 LVLM의 추론 과정을 단계별로 구성된 여러 하위 작업(Subtasks)으로 나누어 제공합니다. 또한 Multi-Grained Scoring (MGS) 메커니즘을 통해 각 추론의 유사성 점수를 계산하여 정확한 검색을 가능하게 합니다. 이러한 구조적 접근은 기존의 글로벌 추론을 넘어 세부적인 요소에 대한 예측을 가능하게 합니다.

- **Performance Highlights**: 실험 결과 CoTMR은 유명 벤치마크 테스트에서 기존의 최첨단 방법들보다 현저하게 우수한 성능을 보였습니다. CoTMR은 기존 방법들이 직면한 컴포넌트 비호환성, 시각적 정보 손실, 및 불완전한 추론 문제를 극복하여 더 높은 해석 가능성을 제공합니다. 본 연구는 ZS-CIR의 영역에서 매우 의미 있는 기여로 평가되며, 다양한 작업에 효과적으로 적용될 수 있을 것입니다.



### MFSR-GAN: Multi-Frame Super-Resolution with Handheld Motion Modeling (https://arxiv.org/abs/2502.20824)
Comments:
          8 pages, 6 figures

- **What's New**: 이 논문에서는 스마트폰 카메라의 한계를 극복하기 위해 멀티 프레임 초해상도 (MFSR) 기술을 제안합니다. 특히, 저해상도(LR) 이미지 프레임으로부터 높은 해상도(HR) 이미지를 생성하기 위한 새로운 합성 데이터 엔진을 도입하여, 센서 특유의 노이즈 특성과 핸드헬드 촬영 시의 이미지 모션을 보존합니다. 또한, MFSR-GAN 네트워크를 제안하여 기존 방식보다 더 나은 화질을 제공합니다.

- **Technical Details**: MFSR-GAN은 기본 프레임을 중심으로 설계되며, 이는 아티팩트를 줄이고 추가 버스트 프레임으로부터 정보를 효과적으로 통합합니다. 합성 데이터 생성 파이프라인은 정적 장면에서 얻은 실제 단기 노출 RAW 사진을 그대로 이용해 LR-HR 훈련 쌍을 생성합니다. 이 과정에서 베이스 프레임을 우선시하여 동작 흐림(motion blur)과 고스트(gosting) 현상을 줄입니다.

- **Performance Highlights**: 실험 결과, 제안된 MFSR-GAN은 합성 및 실제 데이터 세트 모두에서 기존의 최첨단 MFSR 방법들보다 선명하고 더 현실적인 이미지를 생성함을 보여주었습니다. 이의 적용으로 인해 고화질 이미지의 재구성이 가능해져 실제 핸드헬드 버스트 적용에서 높은 성능을 발휘할 수 있음을 입증했습니다.



### Can We Simplify Slide-level Fine-tuning of Pathology Foundation Models? (https://arxiv.org/abs/2502.20823)
Comments:
          11 pages, 3 figures, 4 tables

- **What's New**: 본 연구에서는 기존의 Multiple Instance Learning (MIL) 기반의 약한 감독 학습 방법 대신, 평균 풀링(mean pooling)과 다층 퍼셉트론(multilayer perceptron)을 조합한 단순 비선형 맵핑 전략인 SiMLP를 제안합니다. 이 접근법은 복잡한 학습 과정을 필요로 하지 않으면서도 패치 수준의 foundation 모델을 슬라이드 수준의 작업에 효과적으로 적응시킵니다. 실험 결과, SiMLP는 다양한 다운스트림 작업에서 최첨단 성능을 기록하였으며, 특히 대규모 암 분류 작업에서 MIL 기반 방법보다 3.52% 성능이 우수함을 보여주었습니다.

- **Technical Details**: SiMLP는 슬라이드 수준의 작업에 적합하게 패치 수준의 특징을 결합하는 방법으로, 비선형 MLP를 사용하는 간단한 평균 풀링 방식을 적용합니다. 연구에서 우리는 세 가지 대표적인 foundation 모델을 사용하여 총 여섯 가지 대규모 WSI 분류 작업을 수행하였고, SiMLP는 임의로 선택된 기법들과 비교하여 강력한 특징 표현 능력을 입증하였습니다. 또한, SiMLP는 비선형 분류기를 적용하여 선형 프롬프트보다 높은 전송 가능성을 유지하게 됩니다.

- **Performance Highlights**: SiMLP는 총 7개의 대규모 데이터셋에서 실험이 진행되었으며, 여러 작업에 걸쳐 모든 foundation 모델에서 우수한 성능을 보였습니다. SiMLP는 3개의 대표 모델에서 각각 81.32%, 81.52%, 80.96%의 성능을 달성하였으며, 이는 기존의 약한 감독 학습 방법보다 강한 적응력을 입증하는 결과입니다. 또한, 비소세포 폐암 분류와 같은 세부 작업에서도 높은 안정성과 이전 가능성을 보이며, 다수의 외부 테스트 코호트에 대한 확장 가능성도 가지고 있습니다.



### Improved 3D Point-Line Mapping Regression for Camera Relocalization (https://arxiv.org/abs/2502.20814)
- **What's New**: 이 논문에서는 카메라 재위치시키기 위한 3D 포인트 및 라인 맵 회귀(regression)를 개선할 새로운 접근 방식을 제안합니다. 기존 방법은 특징 매칭(feature matching)이나 단일 네트워크를 통한 포인트와 라인의 인코딩에 의존하고 있으며, 이는 연산적으로 비효율적일 수 있습니다. 우리의 방법은 포인트와 라인 특징을 독립적으로 학습하여 최적의 정확성을 달성하는 데 중점을 두고 있습니다.

- **Technical Details**: 이 연구에서 제안하는 새로운 아키텍처는 포인트와 라인을 각각 독립적으로 우선 순위화한 다음, 이를 통합하여 위치 추정(localization)을 수행합니다. 우리는 두 개의 별도 회귀 분기(regression branches)를 네트워크에 설계하여 각각 포인트와 라인의 특징을 학습합니다. 이 모델은 조기 프루닝 층(early pruning layer)을 사용하여 중요하지 않은 특징을 걸러내고, 그 후 다층 퍼셉트론(multi-layer perceptron)으로 3D 좌표를 회귀합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법이 카메라 재위치시키기에서 3D 맵 포인트 및 라인 회귀 성능을 크게 향상시킴을 입증했습니다. 우리는 어려운 조건의 두 개의 데이터 세트에서 방법을 검증하였으며, 이전 회귀 기반 방법들과 비교하여 성능 개선을 보여주었습니다. 우리의 기법은 실시간 응용에서도 효과적인 성능을 제공하는 것을 목표로 하고 있습니다.



### HAIC: Improving Human Action Understanding and Generation with Better Captions for Multi-modal Large Language Models (https://arxiv.org/abs/2502.20811)
- **What's New**: 이번 연구에서는 비디오 이해의 한계를 개선하기 위해 새로운 데이터 주석 파이프라인을 소개합니다. 이 파이프라인은 명확한 인간 행동을 포함하는 비디오를 인터넷에서 수집하는 두 단계로 구성되어 있습니다. 또한, 두 개의 데이터셋 HAICTrain과 HAICBench를 정교하게 관리하여 고품질 비디오-캡션 쌍을 제공합니다.

- **Technical Details**: 제안된 데이터 주석 파이프라인은 첫 번째 단계에서 비디오를 자동으로 수집하고, 두 번째 단계에서 개별 행동과 상호작용을 상세하게 주석 처리하는 표준화된 캡션 형식을 정의합니다. HAICTrain은 126,000개의 비디오-캡션 쌍을 포함하고 있으며, HAICBench는 500개의 수작업 비디오-캡션 쌍과 1,400개의 QA 쌍으로 구성되어 있어 전반적인 평가를 가능하게 합니다.

- **Performance Highlights**: 실험 결과, HAICTrain으로 훈련한 모델은 인간 행동 이해 능력을 1.4%에서 2.1%까지 개선했습니다. 또한, HAICTrain을 활용하여 텍스트-비디오 생성 성능도 상당히 향상되었습니다. 이 연구는 MLLMs의 인간 행동 이해 성능 향상에 중요한 기여를 제공하고 있습니다.



### Two-Stream Spatial-Temporal Transformer Framework for Person Identification via Natural Conversational Keypoints (https://arxiv.org/abs/2502.20803)
- **What's New**: 본 연구에서는 AI 기반 생성 기술의 발전으로 인해 기존 생체 인식 시스템들이 마주하는 도전과제를 다루고 있습니다. 특히, 고급 deepfake와 face reenactment 기술로부터의 위협을 언급하며, 온라인 대화에서 나타나는 상체 keypoints를 사용하는 인물 식별을 위한 Two-Stream Spatial-Temporal Transformer Framework를 제안합니다. 이 프레임워크는 keypoints 간의 공간적 관계와 시간적 변화를 처리하며, 여러 가지 혁신적인 접근 방식을 통해 높은 인식 정확도를 달성하고 있습니다.

- **Technical Details**: 제안된 방법의 주요 요소는 사용자가 대화 시 보이는 상체 keypoints의 공간적 및 시간적 패턴을 효과적으로 추적하는 것입니다. Spatial Transformer (STR)와 Temporal Transformer (TTR)을 포함한 두 개의 전문 분기를 통해 keypoints의 독특한 구조 패턴과 연속적인 움직임 패턴을 학습합니다. Sapiens 포즈 추정기를 통해 133개의 keypoints를 추출하여 사람의 얼굴 특징, 머리 각도 및 손 위치를 정확하게 표현합니다.

- **Performance Highlights**: 실험 결과, 공간 스트림에서 80.12%, 시간 스트림에서 63.61%의 인식 정확도를 달성하며, 두 가지 융합 전략을 통해 공유 손실 함수 접근 방식이 82.22%의 정확도를, feature-level 융합 방법이 94.86%로 성능을 크게 향상시키는 결과를 보였습니다. 이 시스템은 기존의 외형 기반 생체 인식 방법보다 스푸핑(spoofing)에 더 강력한 성능을 보여줍니다.



### Information Bottleneck-Guided Heterogeneous Graph Learning for Interpretable Neurodevelopmental Disorder Diagnosis (https://arxiv.org/abs/2502.20769)
- **What's New**: 이번 논문에서는 해석 가능한 신경발달장애(NDD) 진단을 위한 새로운 모델인 Interpretable Information Bottleneck Heterogeneous Graph Neural Network(이하 I²B-HGNN)을 제안합니다. 이 프레임워크는 fMRI와 같은 이미징 데이터와 비이미징 데이터를 효과적으로 통합하여 진단의 정확성을 높이려는 노력을 담고 있습니다. 특히, 이 모델은 정보 병목(Information Bottleneck) 원칙을 통해 지역적 FC 패턴과 글로벌 다중 모달 상호작용을 학습합니다.

- **Technical Details**: I²B-HGNN은 두 가지 주요 모듈로 구성되어 있습니다. 첫 번째 모듈은 지역 패턴을 위한 Information Bottleneck Graph Transformer(IBGraphFormer)로, 이 모듈은 그래프 신경망(GNN)과 정보를 압축하여 지표를 식별합니다. 두 번째 모듈인 Information Bottleneck Heterogeneous Graph Attention Network(IB-HGAN)은 다중 모달 데이터를 해석하는 데 중점을 두어 비이미징 데이터의 중요성을 명확히 전달합니다.

- **Performance Highlights**: 실험 결과, I²B-HGNN은 신경발달장애 진단에서 높은 정확도를 기록하며, 해석 가능한 바이오마커 식별과 비이미징 데이터 분석의 효과성을 보여줍니다. 이러한 성과는 뇌 기능 네트워크에서 필수적인 패턴을 유지하면서 정보를 정량화함으로써 이루어진 것입니다. 따라서, I²B-HGNN은 다중 모달 데이터의 유효한 통합과 해석 가능성을 제공하여 NDD 진단의 새로운 가능성을 열어줍니다.



### VRM: Knowledge Distillation via Virtual Relation Matching (https://arxiv.org/abs/2502.20760)
- **What's New**: 본 논문에서는 관계 기반의 지식 증류(knowledge distillation, KD) 방법을 재조명하고, 그들의 약점인 과적합(overfitting)과 잘못된 반응(spurious responses) 문제를 해결하기 위한 몇 가지 핵심 이슈를 다룹니다. 새로운 관계로써 가상의 뷰(virtual views)와 관계를 활용해, 풍부한 샘플 간, 클래스 간, 뷰 간 상관관계를 포착한 친화성 그래프(affinity graphs)를 구성하여 학생 모델이 더 풍부한 지침 신호를 받을 수 있도록 합니다. 이를 통해 보다 효과적인 KD가 가능해집니다.

- **Technical Details**: 가상의 뷰를 생성하고, 이를 기반으로 가상 친화성 그래프를 구성한 후, 실제 샘플과 가상 샘플 간의 관계를 전이하는 새로운 접근 방식을 제안합니다. 기존의 간단한 Gram 행렬 대신, 다양한 관계와 고유 지식을 포함하여 관계 표현의 밀도와 유형을 증가시키고자 합니다. 또한, 불필요하고 신뢰할 수 없는 엣지를 동적으로 제거하여 잘못된 샘플의 영향을 줄이는 방법도 소개됩니다.

- **Performance Highlights**: 실험 결과, 제안된 가상 관계 매칭(virtual relation matching, VRM) 방법은 CIFAR-100과 ImageNet 데이터셋에서 다양한 모델과 아키텍처에 대해 우수한 성능을 보였습니다. 예를 들어, VRM은 ImageNet에서 ResNet50에서 MobileNetV2로의 증류에서 74.0%의 정확도를 달성하였고, CIFAR-100에서는 ResNet56을 사용하며 DeiT-T를 14.44% 개선하였습니다. 이와 같은 결과는 VRM이 관계 기반 방법의 경쟁력을 회복시켰음을 보여줍니다.



### Structured Preference Optimization for Vision-Language Long-Horizon Task Planning (https://arxiv.org/abs/2502.20742)
Comments:
          18 pages

- **What's New**: 기존의 비전-언어 (vision-language) 태스크 플래닝 방법론은 짧은 수평 작업에서는 뛰어난 성능을 보이지만, 복잡하고 긴 수평 작업에서는 한계가 있습니다. 이를 해결하기 위해 Structured Preference Optimization (SPO)라는 새로운 접근법을 제안했습니다. SPO는 장기 작업 계획에 있어 추론 (reasoning) 및 행동 선택 (action selection)을 향상시키는 것을 목표로 합니다.

- **Technical Details**: SPO는 두 가지 핵심 개념을 도입합니다. 첫째, Preference-Based Scoring and Optimization은 태스크 관련성, 비주얼 그라운딩 (visual grounding), 역사적 일관성을 기반으로 추론 체인을 체계적으로 평가합니다. 둘째, Curriculum-Guided Training은 모델이 단순한 작업에서 복잡한 작업으로 점진적으로 적응하여 장기 시나리오에서의 일반화 능력을 향상시키고 추론의 견고성을 강화합니다.

- **Performance Highlights**: SPO는 VirtualHome과 Habitat 2.0에서 각각 +5.98% GCR 및 +4.68% SR, +3.30% GCR 및 +2.11% SR 개선을 이뤘습니다. 이는 이전의 방법을 능가하며 비전-언어 태스크 플래닝에서의 선호 기반 최적화의 효과성을 입증합니다. 실험 결과는 SPO가 장기 작업에서의 추론 품질 및 최종 결정의 정확성을 크게 향상시키는 것을 보여줍니다.



### CADDreamer: CAD object Generation from Single-view Images (https://arxiv.org/abs/2502.20732)
Comments:
          Accepted to CVPR 2025

- **What's New**: 최근 3D 생성 분야에서 Diffusion 기반의 혁신적 발전이 이루어졌습니다. 하지만, 기존 3D 생성 모델은 밀도가 높고 비구조적인 메쉬를 생성하는 경향이 있어, 이는 전문가가 설계한 CAD 모델과 크게 대조됩니다. 이를 해결하기 위해 'CADDreamer'라는 새로운 접근 방식을 제안하며, 이는 단일 이미지에서 CAD 객체의 경계 표현(B-rep)을 생성하는 특징이 있습니다.

- **Technical Details**: CADDreamer는 원시 인식(primitive-aware) 다중 View Diffusion 모델을 이용하여 복잡한 기하학적 구조를 형성합니다. 기본적으로 원시 기하학적 세부 정보와 고수준 구조적 의미를 동시에 캡처하며, 이 과정에서 색상 도메인에 원시 의미를 인코딩하여 강력한 사전 학습된 모델의 특성을 활용합니다. 또한, 기하학 최적화 기술과 위상 보존 추출 방법을 도입하여 생성된 원시의 노이즈와 왜곡을 줄입니다.

- **Performance Highlights**: 실험 결과, CADDreamer 방식은 단일 View 이미지에서 고품질의 CAD 객체를 효과적으로 복원할 수 있음을 보여줍니다. CADDreamer가 생성한 B-rep 모델은 구조적으로 명확하고, 경계가 날카로우며, 물이 새지 않는 강력한 위상 구조를 제공합니다. 이러한 결과는 게임, 제조 및 제품 디자인 등 높은 품질의 구조화된 3D 모델이 필요한 다양한 현실 세계의 응용 분야에 유용합니다.



### Glioma Classification using Multi-sequence MRI and Novel Wavelets-based Feature Fusion (https://arxiv.org/abs/2502.20715)
Comments:
          18 pages, 11 figures, 6 tables, journal paper

- **What's New**: 이번 연구는 세계보건기구의 기준에 따라 저등급 뇌종양(Low Grade Glioma, LGG)과 고등급 뇌종양(High Grade Glioma, HGG)으로 분류되는 신경아교세포에서 유래한 뇌종양인 교모세포종(glioma)의 비침습적 평가를 위한 최신 접근법을 제시합니다. 특히, 다중 시퀀스 MRI 이미지를 기반으로 한 웨이블릿 기반의 새로운 융합 알고리즘을 도입하여 상당한 진단 능력을 확보하였습니다.

- **Technical Details**: 연구에서는 T1, T1-대비 증강(T1CE), T2 및 유체감쇠 회복(Fluid Attenuated Inversion Recovery, FLAIR) MRI 이미지를 활용하여 이미징 특징(radiomics features)을 계산하였습니다. 또한, 특징 공간(feature space)을 축소하기 위해 주성분 분석(Principal Component Analysis, PCA)을 적용하고 XGBoost, Support Vector Machine(SVM), Random Forest 분류기를 사용하여 분류 작업을 수행하였습니다.

- **Performance Highlights**: SVM 알고리즘은 BraTS 2018 데이터셋에 대해 90.17%의 정확도(accuracy)와 91.04%의 정밀도(precision), 96.19%의 재현율(recall), 93.53%의 F1 점수(F1-score), 94.60%의 AUC(Area Under Curve)를 달성하며 우수한 성능을 보였습니다. 또 다른 BraTS 2018 데이터셋에서도 91.34%의 정확도, 93.05%의 정밀도, 96.13%의 재현율, 94.53%의 F1 점수, 93.71%의 AUC를 기록하여 제안된 알고리즘이 교모세포종의 컴퓨터 지원 진단 및 등급 시스템에 적용 가능성을 보여줍니다.



### Towards General Visual-Linguistic Face Forgery Detection(V2) (https://arxiv.org/abs/2502.20698)
Comments:
          8 pages, 5 figures, Accpet by CVPR2025

- **What's New**: 본 논문은 최근 얼굴 조작 기술의 발전에 발맞추어 Face Forgery Text Generator (FFTG)라는 새로운 주석 파이프라인을 제안합니다. FFTG는 위조 마스크를 활용하여 초기 지역 및 유형 식별을 수행한 후, 다중 모달 대형 언어 모델(MLLM)의 환각(hallucination)을 줄이기 위한 포괄적인 프롬프트 전략을 사용합니다. 이 접근 방식은 더 정확한 텍스트 주석 생성을 통해 위조 탐지 모델의 신뢰성을 높입니다.

- **Technical Details**: FFTG는 진짜 이미지와 위조 이미지를 비교하여 위조 맵을 생성하고, 얼굴 구성 요소마다 위조 정도를 평가합니다. 이를 통해, 수작업으로 정의한 기능을 사용하여 위조 유형을 추정하고, 이러한 요소들을 통합하여 초기 주석을 생성하는 과정을 포함합니다. FFTG의 정확성을 높이기 위해, 시각적 프롬프트, 유도 프롬프트, 작업 설명 프롬프트, 표준화된 출력 프롬프트 등 총 네 가지 유형의 프롬프트를 통해 MLLM을 안내합니다.

- **Performance Highlights**: 실험 결과, FFTG 주석은 전통적인 방법과 비교하여 더 높은 지역 식별 정확도를 자랑하며, 다양한 위조 탐지 벤치마크에서 모델 성능의 증가를 이끌어냅니다. 특히 CLIP 모델에 대한 파인튜닝 결과, FFTG 주석이 기존 방법에 비해 더 나은 일반화 성능을 나타냈습니다. 또한 MLLM의 경우, FFTG의 주석은 인간 주석 및 직접 GPT 라벨링보다 더 높은 정확도를 보이며, 주석 오류를 줄임으로써 다양한 메트릭에서 개선된 모델 성능을 보여주었습니다.



### WorldModelBench: Judging Video Generation Models As World Models (https://arxiv.org/abs/2502.20694)
- **What's New**: 본 논문에서는 비디오 생성 모델들의 세밀한 세계 모델링 능력을 평가하기 위해 'WorldModelBench'라는 새로운 벤치마크를 제안합니다. 기존의 평가 기준들은 비디오 품질에 초점을 맞추어 세계 모델링의 복잡성을 충분히 반영하지 못했습니다. 'WorldModelBench'는 물리적 규칙을 따르는지의 여부와 지침 준수 등의 차원을 평가하여 모델의 신뢰성을 높이고 있습니다.

- **Technical Details**: 'WorldModelBench'는 350개의 이미지 및 텍스트 조건 쌍으로 구성되며 7개의 응용 분야에서 56개의 다양한 하위 분야를 포함하고 있습니다. 이 벤치마크는 텍스트에서 비디오(T2V) 및 이미지에서 비디오(I2V) 모델을 평가할 수 있도록 설계되었습니다. 각 비디오의 평가 기준은 지침 준수와 향후 프레임 생성으로 나뉘어 있으며, 각각의 세부 항목은 더욱 세분화되었습니다.

- **Performance Highlights**: 제안된 모델 평가 방법을 통해, 'WorldModelBench'는 기존의 모델보다 8.6% 높은 평균 정확도로 비디오의 세계 모델링 위반을 예측할 수 있음을 보여주었습니다. 또한, 인간의 주석을 최대화하여 Fine-tuning한 판별자는 비디오 생성 모델의 세계 모델링 능력을 눈에 띄게 향상시키는 데 기여함을 입증하였습니다. 이 연구는 향후 로봇 공학, 자율 주행 등의 분야에서 보다 신뢰할 수 있는 비디오 생성 모델의 개발로 이어질 것입니다.



### EDM: Equirectangular Projection-Oriented Dense Kernelized Feature Matching (https://arxiv.org/abs/2502.20685)
- **What's New**: 이 논문에서는 Equirectangular Projection-Oriented Dense Kernelized Feature Matching (EDM)이라는 첫 번째 학습 기반 밀집 매칭 알고리즘을 소개합니다. 이 알고리즘은 구형 이미지에 최적화되어 있으며, 기존의 매칭 기법의 왜곡 문제를 해결하는 데 중점을 두고 있습니다. 특히, 구형 카메라 모델과 지오데식 흐름 정제를 활용하여 성능을 향상시킵니다. EDM은 두 개의 구형 이미지 간의 상대적인 포즈 추정을 위한 밀집 매칭에서 혁신적인 접근 방식을 제공합니다.

- **Technical Details**: EDM은 Spherical Spatial Alignment Module (SSAM)과 Geodesic Flow Refinement라는 두 가지 주요 단계로 구성됩니다. SSAM은 ERP 이미지에 대한 구형 위치 임베딩을 활용하고, 디코더를 통해 전역 매칭을 생성합니다. Geodesic Flow Refinement 단계에서는 좌표 변환을 수행하여 대응점 잔차를 정제합니다. 이 방법은 기존의 점 기반 매칭 기법이 직면한 'why'를 해결하면서 구형 이미지의 특성을 잘 살립니다.

- **Performance Highlights**: 제안된 방법은 Matterport3D 및 Stanford2D3D 데이터셋에서 각각 +26.72 및 +42.62의 AUC@5° 개선을 달성하여 밀집 매칭 성능이 크게 향상되었습니다. 또한, EgoNeRF 및 OmniPhotos 데이터셋에서 질적으로 평가하여 다양한 환경에서 강력한 성능을 입증했습니다. 마지막으로, 데이터 증강을 위해 방위각 회전을 적용하여 두 개의 구형 이미지 간의 밀집 매칭 및 상대 포즈 추정에서 최첨단 성능을 달성하였습니다.



### Diffusion Restoration Adapter for Real-World Image Restoration (https://arxiv.org/abs/2502.20679)
- **What's New**: 이 논문에서는 기존의 ControlNet보다 가벼운 Restoration Adapter를 제안하여 이미지 복원에 활용하고자 하였습니다. ControlNet은 많은 매개변수를 포함하여 훈련과 추론에서 부담이 컸던 반면, 제안된 Adapter는 사전 훈련된 생성 모델의 강력한 생성 능력을 활용하여 선명한 이미지를 복원합니다. 이 Adapters는 denoising UNet과 DiT 아키텍처 모두에 적응할 수 있도록 설계되었습니다.

- **Technical Details**: Diffusion models는 간단한 분포를 복잡한 분포로 변환하기 위해 일련의 가역 단계를 통해 작동하는 생성 모델입니다. 이 모델은 저품질 이미지(LQ)를 조건으로 사용하여 고품질 이미지(HQ)를 생성하며, Latent Diffusion Models (LDM)와 같은 사전 훈련된 모델을 주로 활용합니다. 본 논문에서는 StableDiffusion XL과 StableDiffusion 3을 사용하여 복원 작업을 위한 효과적인 프라이어(Generative Prior)를 제공하고, Restoration Adapter를 통해 이를 통합합니다.

- **Performance Highlights**: 제안된 방법은 다양한 이미지 유형에서 고품질 복원 결과를 생성할 수 있으며, 특정 프롬프트에 기반한 제어 가능한 복원도 지원합니다. 이 모델은 이미지 복원의 정확성과 다양성 간의 균형을 높이는 샘플링 전략을 도입하였으며, 이를 통해 복원 과정에서의 충실도를 보장합니다. 이렇게 개선된 방식은 기계 학습 분야에서 이미지 복원 작업의 성능을 크게 향상시키는 데 기여할 것으로 기대됩니다.



### STPro: Spatial and Temporal Progressive Learning for Weakly Supervised Spatio-Temporal Grounding (https://arxiv.org/abs/2502.20678)
Comments:
          CVPR'25 Conference

- **What's New**: 이번 연구에서는 약한 감독 하의 시공간 비디오 그라우딩(Weakly Supervised Spatio-Temporal Video Grounding, WSTVG) 기술을 다룹니다. 이 기술은 텍스트 쿼리만을 사용하여 비디오에서 대상을 시공간적으로 위치 지정하는 도전적인 작업입니다. 우리는 Tubelet Referral Grounding (TRG) 모듈을 도입하여 텍스트 쿼리를 tubelet과 연결함으로써 발생하는 시공간 예측을 지원합니다. 또한 STPro라는 새로운 점진적 학습 프레임워크를 통해 복합적인 행동 이해 및 복잡한 장면 문제를 해결하려고 합니다.

- **Technical Details**: WSTVG의 기본 문제는 비디오 내에서 특정 행동과 객체를 정확하게 위치시키는 것입니다. 우리는 Sub-Action Temporal Curriculum Learning (SA-TCL)과 Congestion-Guided Spatial Curriculum Learning (CG-SCL)이라는 두 가지 핵심 모듈을 도입했습니다. SA-TCL은 행동 시퀀스의 복잡한 이해를 점진적으로 향상시키고, CG-SCL은 복잡한 장면에 맞게 모델을 조정하여 학습합니다.

- **Performance Highlights**: STPro는 VidSTG-Declarative 데이터셋에서 1.0%, HCSTVG-v1에서 3.0%의 성능 향상을 달성하여 최신 기술보다 우수한 결과를 보여주었습니다. 이 결과는 약한 감독 하에서 비디오를 처리하는 데 있어 혁신적인 접근 방식을 제시하며, 복잡한 행동 관계와 대규모 비디오 데이터 세트에서의 의미론적 이해 능력을 크게 향상시킵니다.



### SciceVPR: Stable Cross-Image Correlation Enhanced Model for Visual Place Recognition (https://arxiv.org/abs/2502.20676)
- **What's New**: 이번 논문에서는 Visual Place Recognition (VPR) 분야에서의 주요 도전 과제를 다루며, 기존의 DINOv2 모델을 활용한 새로운 접근 방식을 제안합니다. 기존 모델들은 이미지 간의 상관관계를 탐색하거나 효과적인 성능을 위해 두 단계의 재정렬 전략을 사용하지만, 이들은 최종 출력만을 이용하여 안정적인 검색 결과를 제공하지 못했습니다. SciceVPR이라는 새로운 모델은 다양한 컨텍스트 정보(valuable contextual knowledge)를 효과적으로 인코딩하는 특징 표현을 활용하여 더 안정적이고 일관된 전역(descriptor) 특성을 얻을 수 있는 방법을 제공합니다.

- **Technical Details**: SciceVPR 모델은 DINOv2의 다층 출력에서 중요한 채널과 공간 정보를 포착하기 위해 다중 층(feature fusion) 모듈을 사용합니다. 또한, SciceVPR은 배치 내 이미지 간의 불변한 상관관계를 중요한 지식으로 간주하여 자기 강화 인코더(self-enhanced encoder)에 이를 증류(distill)합니다. 이러한 방식을 통해 SciceVPR은 도메인 변화(domain shifts)가 있는 상황에서도 강력한 글로벌 특징을 획득할 수 있어 조명 변화, 날씨 및 시점 변화에 강한 성능을 발휘합니다.

- **Performance Highlights**: 실험 결과에 따르면, SciceVPR-B라는 기본 변형은 다양한 도메인 조건에서 단일 입력으로 여러 데이터셋에서 SOTA 일단계 방법보다 뛰어난 성능을 보입니다. 특히 대형 변형인 SciceVPR-L은 SOTA 이단계 모델과 동등한 성능을 나타내며, 도전적인 Tokyo24/7 데이터셋에서 기존 모델보다 Recall@1에서 3% 이상의 향상을 기록했습니다. 이러한 결과를 바탕으로 SciceVPR은 VPR 분야에서 주목할 만한 발전을 이루었음을 보여줍니다.



### EndoPBR: Material and Lighting Estimation for Photorealistic Surgical Simulations via Physically-based Rendering (https://arxiv.org/abs/2502.20669)
Comments:
          10 pages, 3 figures

- **What's New**: 본 논문은 수술 장면의 3D 비전을 위한 라벨이 있는 데이터셋의 부족을 해결하기 위해 새로운 차별화 가능한 렌더링(framework) 프레임워크를 제안합니다. 이 프레임워크는 내시경(endoscopic) 이미지와 알려진 기하학(geometry)을 통해 재료(material) 및 조명(lighting) 추정을 수행합니다. 특히, 조명과 재료 속성을 명확히 분리하여 포토리얼리스틱(photorealistic)한 새로운 뷰 합성을 가능하게 합니다.

- **Technical Details**: EndoPBR는 물리 기반 렌더링(physically-based rendering)을 통해 색상 픽셀 값을 결정합니다. 우리는 조도가 높은 현장 흡수 매개변수(BRDF: Bidirectional Reflectance Distribution Function)와 조명 조건을 추정하기 위해 엔도스코피 이미지에 대한 미분 가능 렌더링을 사용합니다. 이 방법은 훈련 세트와 다른 카메라 뷰에 대해 일반화할 수 있는 것으로 나타났습니다.

- **Performance Highlights**: 본 연구에서 제안하는 방법은 대장내시경 3D 비디오 데이터셋(C3VD)을 사용하여 평가되었습니다. 결과적으로, EndoPBR는 기존의 NeRF 및 3DGS 방법과 비슷한 성능으로 새로운 뷰를 합성하는 것으로 나타났습니다. 또한, 생성된 포토리얼리틱 합성 데이터를 사용하여 깊이 추정 모델을 미세 조정(finetuning)할 수 있는 가능성을 보여줍니다.



### OpenEarthSensing: Large-Scale Fine-Grained Benchmark for Open-World Remote Sensing (https://arxiv.org/abs/2502.20668)
- **What's New**: 본 논문에서는 OpenEarthSensing라는 대규모의 세밀한 벤치마크를 제안합니다. 이 벤치마크는 5개의 도메인과 3개의 모달리티를 포함하여 189개의 카테고리와 총 157,674개의 이미지를 포함합니다. OpenEarthSensing은 실제 세상에서 발생할 수 있는 의미적 변화를 다룰 수 있도록 설계되었습니다.

- **Technical Details**: OpenEarthSensing는 RGB 위성 이미지, RGB 드론 공중 이미지, MSRGB, 적외선 등의 다양한 데이터 도메인을 보유하고 있어, 서로 다른 공변량 변화(covariate shifts) 조건을 갖춘 데이터 세트를 제공합니다. 이를 통해 모델의 일반화 성능을 평가할 수 있는 종합적인 테스트베드를 제공합니다.

- **Performance Highlights**: 기존의 열린 세계(open-world) 원거리 감지 기술과 방법을 기준으로 평가를 수행하여, OpenEarthSensing이 지속적으로 진화하는 환경에서의 모델 성능을 시험하는 데 있어 도전적인 벤치마크 역할을 한다는 것을 입증합니다. 이 논문에서 제안하는 벤치마크는 연구 및 개발을 촉진할 것으로 기대됩니다.



### Advancing AI-Powered Medical Image Synthesis: Insights from MedVQA-GI Challenge Using CLIP, Fine-Tuned Stable Diffusion, and Dream-Booth + LoRA (https://arxiv.org/abs/2502.20667)
- **What's New**: MEDVQA-GI 챌린지는 의료 진단에서 AI 기반 text-to-image 생성 모델의 통합을 다룹니다. 기존 방법은 정적인 이미지 분석에 주로 초점을 맞추고 있으며, 텍스트 설명에서 동적으로 의료 이미지를 생성하는 데는 부족하고, 이에 대한 새로운 접근 방식을 제안하고 있습니다. 본 연구는 Fine-tuned Generative Models을 사용하여 동적이고 확장 가능하며 정밀한 이미지를 생성하는 방법을 제시합니다.

- **Technical Details**: 이 시스템은 Fine-tuned Stable Diffusion, DreamBooth 모델 및 Low-Rank Adaptation (LORA)을 통합하여 고품질 의료 이미지를 생성합니다. 연구의 두 가지 주요 태스크는 Image Synthesis (IS)와 Optimal Prompt Generation (OPG)으로, 전자는 언어 프롬프트를 통해 이미지를 생성하고 후자는 주어진 카테고리 내에서 고품질 이미지를 생성하는 프롬프트를 제공합니다. 기존 의료 이미지 생성 방법의 한계에 대해 강조하며, Stable Diffusion이 이미지 품질 면에서 다른 모델들보다 우수함을 보여줍니다.

- **Performance Highlights**: Stable Diffusion은 Fréchet Inception Distance (FID) 스코어가 가장 낮았으며, 평균 Inception Score도 가장 높아(2.327) 뛰어난 다양성과 품질을 나타냅니다. 이는 AI 기반 의료 진단 분야의 발전을 의미하며, 향후 연구는 모델 개선, 데이터 세트 증대 및 윤리적 고려 사항에 중점을 두어야 할 것입니다.



### Dataset Distillation with Neural Characteristic Function: A Minmax Perspectiv (https://arxiv.org/abs/2502.20653)
Comments:
          Accepted by CVPR 2025, 11 pages, 7 figures

- **What's New**: 이 논문에서는 데이터셋 증류(dataset distillation)의 새로운 접근 방식을 제안하여, Neural Characteristic Function Discrepancy (NCFD)라는 새로운 지표를 도입합니다. 기존의 거리 측정 방법들이 분포의 차이를 정확하게 구별하지 못하는 문제를 해결하기 위해, NCFD는 특성 함수를 활용하여 전체 분포 정보를 포괄적으로 캡슐화하도록 설계되었습니다. 이러한 방법은 깊은 신경망의 학습 과정에서 생성 데이터와 실제 데이터 간의 차이를 최소화하는 동시에 정확도 향상을 도모합니다.

- **Technical Details**: NCFD는 특성 함수를 기반으로 하는 매개변수화된 지표로, 확률 밀도 함수의 푸리에 변환으로 정의됩니다. 이 접근 방식은 고차원 매니폴드에서의 시맨틱 구조를 포착하는데 효과적이며, 분포 매칭을 적대적인 미니맥스 최적화 문제로 재구성하여, 실시간으로 데이터 간의 불일치를 극대화합니다. 결과적으로, NCFD는 기존 최대 평균 불일치(maximum mean discrepancy, MMD) 방법보다 훨씬 더 높은 성능과 효율성을 보여줍니다.

- **Performance Highlights**: 실험 결과, NCFM 방법은 최신 기법들(SOTA)과 비교하여 ImageSquawk 데이터셋에서 20.5%의 정확도 향상을 달성했습니다. 또한 GPU 메모리 사용량을 300배 이상 줄이고, 처리 속도를 20배 향상시키는 성과를 올렸습니다. 특히, CIFAR-100 데이터셋에 대해 단일 NVIDIA 2080 Ti GPU에서 2.3GB 메모리만으로 무손실 압축을 달성하여 주목받고 있습니다.



### The Common Objects Underwater (COU) Dataset for Robust Underwater Object Detection (https://arxiv.org/abs/2502.20651)
- **What's New**: COU: Common Objects Underwater 데이터셋을 소개합니다. 이는 다양한 수중 및 해양 환경에서 발견되는 인공 객체의 인스턴스 분할 이미지 데이터셋으로, 약 10K 개의 주석이 달린 이미지가 포함되어 있습니다. COU는 자율 수중 차량(AUVs)을 교육하기 위해 필요한 데이터셋 부족 문제를 해결하기 위해 만들어졌으며, 해양 생물에 초점을 맞춘 기존 데이터셋과는 달리 다양한 객체 클래스의 이미지를 제공합니다.

- **Technical Details**: COU 데이터셋은 특수한 수중 객체 감지를 위해 큐레이션된 데이터로, 24개 객체 클래스(예: 해양 쓰레기, 다이빙 도구, AUVs)의 세분화된 이미지를 포함하고 있습니다. 이 데이터셋은 조명 굴절, 색상 왜곡, 가시성 제한 등 수중 이미징의 독특한 도전 과제를 다루기 위해 모든 이미지를 수중에서 촬영하여 수중 조건에 최적화된 모델 훈련을 가능하게 합니다.

- **Performance Highlights**: COU로 훈련된 감지기의 성능은 육상의 데이터로만 훈련된 모델에 비해 개선된 결과를 보여줍니다. 이는 수중 객체에 주석이 달린 이미지를 기반으로 학습하는 것이 분명한 이점을 제공함을 나타냅니다. COU 데이터셋은 현재 오픈소스 라이센스 하에 제공되며, 다양한 연구 및 응용 분야에서 광범위하게 사용될 수 있습니다.



### Gungnir: Exploiting Stylistic Features in Images for Backdoor Attacks on Diffusion Models (https://arxiv.org/abs/2502.20650)
- **What's New**: 최근 확산 모델(Diffusion Models, DMs)은 이미지 생성을 위한 혁신적인 기법으로 주목받고 있습니다. 그러나 DMs는 백도어 공격(Backdoor Attacks)에 취약하다는 연구 결과가 있습니다. 이 연구에서는 Gungnir라는 새로운 방법을 제안하여 사용자가 입력하는 이미지의 숨겨진 스타일 트리거를 통해 DMs의 백도어를 활성화할 수 있도록 하고 있습니다.

- **Technical Details**: 우리는 Gungnir를 통해 입력 이미지의 스타일적 특징을 트리거로 사용하는 최초의 방법을 제안했습니다. Reconstruction-Adversarial Noise (RAN) 기법을 사용하여 이미지 간 변환(image2image) 작업에서 성공적으로 백도어 공격을 수행하였습니다. 또한, Short-Term-Timesteps-Retention (STTR) 기법을 활용하여 모델의 스타일 특징을 보다 효과적으로 활용할 수 있게 되었습니다.

- **Performance Highlights**: 결과적으로, 우리의 방법은 기존의 방어 방법들을 쉽게 우회할 수 있으며, 현재 사용하는 DMs의 주요 백도어 방어 틀 중에서 0%의 백도어 탐지율(Backdoor Detection Rate, BDR)을 기록하였습니다. 이는 DMs가 고차원 특징을 감지하고 백도어 공격에 취약함을 나타내는 중요한 발견입니다.



### EDENet: Echo Direction Encoding Network for Place Recognition Based on Ground Penetrating Radar (https://arxiv.org/abs/2502.20643)
- **What's New**: 본 논문에서는 Ground Penetrating Radar (GPR) 기반의 장소 인식(Place Recognition, PR) 문제를 다루며, 기존의 방법들은 주로 소규모 환경에 초점을 맞추었다는 점에서 새롭습니다. GPR의 두드러진 특징인 방향성 정보를 활용하여 네트워크 설계를 강화하는 방법을 제안합니다. 또한, 학습 가능한 Gabor 필터와 방향 인식 주의 메커니즘을 도입하여 GPR 에코로부터 정밀한 방향 응답을 추출합니다.

- **Technical Details**: EDENet라 불리는 새롭고 강력한 접근법에서는 GPR 에코의 기하학적 관계를 활용하여 효과적인 피쳐 인코딩 및 필터링을 수행합니다. 이 방법에서는 방향성을 인식하는 주의 메커니즘을 통해 GPR 신호를 통합적으로 분석하고, 다중 스케일 집합 전략을 적용하여 다양한 전기적 특성 변화를 효과적으로 수용합니다. 확장 현실에서 EDENet은 특히 지하 환경에서 명확한 우위를 보이며, 생산성 향상을 위해 고안되었습니다.

- **Performance Highlights**: 공개 데이터셋에서 수행된 실험을 통해 EDENet이 기존의 솔루션을 뛰어넘는 성능을 발휘하는 것을 보여주었습니다. 모델의 크기와 계산 효율성 또한 크게 향상되어 실제 적용 가능성을 높입니다. EDENet은 특히 서브서피치(지하) 환경에서 장소 인식의 강점을 극대화하여, 기존 방법들과 비교했을 때 중요한 성능 개선을 이루었습니다.



### TractCloud-FOV: Deep Learning-based Robust Tractography Parcellation in Diffusion MRI with Incomplete Field of View (https://arxiv.org/abs/2502.20637)
- **What's New**: 이번 논문에서는 TractCloud-FOV라는 새로운 딥 러닝(framework) 프레임워크를 소개합니다. 이 프레임워크는 임상 스캔에서 흔히 발생하는 불완전한 시야(FOV)에서도 견고하게 트랙토그래피(parcellation)를 수행할 수 있도록 설계되었습니다. 특히, 새로운 학습 전략인 FOV-Cut Augmentation(FOV-CA)를 통해 실제 상황을 모사한 다양한 불완전한 FOV 컷오프(기준)을 시뮬레이션합니다.

- **Technical Details**: FOV-Cut Augmentation은 합성적으로 트랙토그램(tracograms)을 잘라내어 현실적인 잘린 스트림라인(streamlines)을 생성하는 데이터 증강(data augmentation) 방법입니다. 이 접근법은 훈련 세트를 현실감 있게 보강하여 모델의 일반화(generalization) 능력을 극대화합니다. 실험은 합성적으로 잘린 트랙토그래피와 불완전한 FOV를 가진 두 개의 실제 데이터셋에서 수행되었습니다.

- **Performance Highlights**: TractCloud-FOV는 모든 테스트 데이터셋에서 스트림라인 분류 정확도(classification accuracy), 일반화 능력, 해부학적 묘사(anatomical depiction), 계산 효율성(computational efficiency) 면에서 여러 최신의(state-of-the-art) 방법을 크게 능가합니다. 전반적으로 TractCloud-FOV는 불완전한 FOV 조건에서도 효율적이고 일관된 트랙토그래피 파셀레이션(parcellation)을 달성합니다.



### T2ICount: Enhancing Cross-modal Understanding for Zero-Shot Counting (https://arxiv.org/abs/2502.20625)
Comments:
          Accepted by CVPR2025

- **What's New**: 본 논문은 텍스트 설명을 통해 임의의 객체 카테고리를 집계하는 제로샷(Zero-shot) 객체 집계 방법인 T2ICount를 제안합니다. 기존의 CLIP 기반 비전-언어 모델들은 텍스트 프롬프트에 대한 민감도가 부족하다는 문제를 가지고 있으며, 본 연구에서는 이를 해결하기 위해 계층적 의미 보정 모듈(Hierarchical Semantic Correction Module)과 표현적 지역 일관성 손실(Representational Regional Coherence Loss)을 도입하였습니다. 또한, FSC-147에서 재주석된 데이터셋인 FSC-147-S를 제공하여 텍스트 지향 집계 능력을 보다 잘 평가할 수 있도록 하였습니다.

- **Technical Details**: T2ICount는 사전 학습된 확산 모델로부터 풍부한 사전 지식을 활용하여 제로샷 집계를 수행하는 프레임워크입니다. 효율성을 보장하는 단일 단계 제거(denoising) 방식을 사용하지만, 이는 텍스트 민감도를 감소시키는 단점을 가지고 있습니다. 이를 보완하기 위해 HSCM을 통해 다중 스케일 피처 보정 및 ℒRRC을 통해 교차 주의 맵을 활용하여 세밀한 객체 검토를 수행합니다. 이로 인해 보다 정확한 피처 학습이 가능합니다.

- **Performance Highlights**: 다양한 벤치마크에서 T2ICount는 기존 방법들에 비해 우수한 성능을 기록하였으며, 어려운 하위 집합인 FSC-147-S에서도 뛰어난 성과를 보여주었습니다. 본 연구는 조건부 집계 모델의 편향 없는 평가를 가능하게 하는 새로운 프로토콜을 도입하였으며, 이러한 기여를 통해 제로샷 집계 분야에서의 혁신을 도모하고 있습니다.



### RTGen: Real-Time Generative Detection Transformer (https://arxiv.org/abs/2502.20622)
- **What's New**: 이 논문에서는 Real-Time GENerative Detection Transformer (RTGen)을 제안합니다. RTGen은 고정된 카테고리에 의존하지 않는 실시간 개체 감지기가 되며, 개체와 관련 텍스트를 동시에 처리할 수 있도록 설계되었습니다. 특히, 비자율 회귀 언어 모델인 Directed Acyclic Transformer (DA-Transformer)를 통합하여 빠른 처리 속도를 자랑합니다.

- **Technical Details**: RTGen은 고유한 Region-Language Decoder (RL-Decoder)를 도입하여 언어 모델과 감지기 간의 통합을 개선했습니다. 기존의 자율 회귀 언어 모델과는 달리, RTGen은 객체와 텍스트 데이터를 동시에 처리할 수 있도록 구조가 설계되었습니다. 이 모델은 실시간 성능을 발휘하고, 주어진 데이터 세트에서 향상된 정확도를 달성합니다.

- **Performance Highlights**: RTGen은 LVIS 데이터 세트에서 60.41 FPS의 우수한 추론 속도를 기록하며, mAP(Mean Average Precision) 측정기준에서 18.6을 달성하여 이전의 최고 성능(SOTA) 방법보다 3.5 mAP 향상된 성능을 보여주었습니다. 이러한 성능은 다양한 실제 적용 분야에서의 활용 가능성을 높여 줍니다.



### Interpreting CLIP with Hierarchical Sparse Autoencoders (https://arxiv.org/abs/2502.20578)
- **What's New**: 본 논문에서는 Matryoshka SAE (MSAE)라는 새로운 아키텍처를 소개합니다. MSAE는 여러 세분화 수준에서 계층적 표현을 동시에 학습하여 재구성 품질과 희소성(sparsity)이라는 두 가지 지표를 효율적으로 최적화할 수 있도록 돕습니다. 이를 통해 CLIP 모델의 해석 가능성과 제어 가능성을 향상시킵니다.

- **Technical Details**: MSAE는 TopK 연산을 h번 적용하여 점진적으로 증가하는 활성 뉴런 수(k)를 학습합니다. 이러한 방식으로, MSAE는 coarse 개념에서 fine-grained 특징까지 다양한 granularities를 동시에 학습하며, 모든 수준에서 재구성 손실을 결합하여 보다 유연하고 적응적인 희소성 패턴을 달성합니다. 이로써 기존의 TopK 제약조건과 L1 정규화의 문제를 해결합니다.

- **Performance Highlights**: MSAE는 CLIP의 경우에서 0.99 코사인 유사도(cosine similarity)와 0.1 미만의 설명되지 않은 분산(fraction of variance unexplained)을 달성하면서 80%의 희소성을 유지합니다. MSAE의 실용성을 검증하기 위해 120개 이상의 해석 가능한 개념을 추출하고 이를 기반으로 유사도 검색 및 편향 분석을 수행했습니다.



### InstaFace: Identity-Preserving Facial Editing with Single Image Inferenc (https://arxiv.org/abs/2502.20577)
- **What's New**: 이 논문에서는 단 하나의 이미지로 정체성을 유지하면서도 사실적인 이미지를 생성할 수 있는 새로운 확산 기반 프레임워크인 InstaFace를 도입합니다. 기존의 방법들이 여러 이미지를 필요로 하는 것과 달리, InstaFace는 효율적인 가이던스 네트워크를 통해 3D 관점을 통합하고 추가적인 학습 가능한 매개변수를 도입하지 않습니다. 이를 통해 얼굴 편집에서의 자연스러운 전이와 맥락 정보인 배경 및 액세서리 같은 세부 요소를 보존할 수 있습니다.

- **Technical Details**: InstaFace는 3D 조건 맵(3D conditional maps)을 효과적으로 처리하기 위해 3D Fusion Controller Module을 설계하였으며, 이를 통해 메모리 사용량과 계산 오버헤드를 크게 줄일 수 있습니다. 또한, Identity Preserver Module을 도입하여 얼굴 인식 엔코더와 CLIP 이미지 엔코더를 통합하여 더 정확한 정체성 유지를 보장합니다. 이 모듈은 각 트레이닝 단계에서 얼굴 속성의 제어 능력을 향상시키는데 기여합니다.

- **Performance Highlights**: 실험 결과, InstaFace는 정체성 유지, 사진 리얼리즘 및 포즈, 표정, 조명의 효과적인 제어 측면에서 기존의 여러 최첨단 접근 방법을 초월하는 성능을 보여줍니다. FFHQ 데이터셋을 사용하여 학습한 이 모델은 단 하나의 이미지로 이루어진 인퍼런스를 통해 높은 정밀도의 얼굴 이미지를 생성합니다. 각 단계 및 모듈에서의 개선 사항을 강조하는 다양한 실험 결과 또한 제시되었습니다.



### Visual Reasoning at Urban Intersections: FineTuning GPT-4o for Traffic Conflict Detection (https://arxiv.org/abs/2502.20573)
- **What's New**: 이번 연구는 신호 없는 도시 교차로에서의 교통 제어의 복잡한 문제를 해결하기 위해 Multimodal Large Language Models (MLLMs), 특히 GPT-4o의 능력을 탐구합니다. 이 방법은 교차로의 조감도 비디오를 직접 사용하여 논리적이고 시각적인 추론을 제공하는 방식을 채택합니다. 이 연구는 교차로의 충돌 감지 및 운전자를 위한 설명과 권장 사항을 제공하는 지능형 시스템으로서의 GPT-4o의 잠재력을 강조합니다.

- **Technical Details**: 연구에서 사용된 GPT-4o는 사전 훈련된 언어 모델에 비디오 데이터를 결합하여 학습되었습니다. 이를 통해 모델은 교차로에서 발생하는 충돌을 탐지하고, 이에 대한 설명 및 다음 행동을 추천하는 기능을 수행합니다. 모델은 77.14%의 정확도로 훈련된 반면, 수동 평가에서는 설명에 대해 89.9%, 추천된 다음 행동에 대해 92.3%의 정확도를 달성하였습니다.

- **Performance Highlights**: MLLMs를 사용하여 비디오 데이터 기반의 실시간 교통 관리가 가능하다는 결과를 도출했습니다. 이 연구의 결과는 교차로 관리 및 운영에 대한 확장 가능하고 실용적인 통찰을 제공하며, 향후 교통 관리 시스템에 MLLMs의 활용 가능성을 보여줍니다. 연구에서 사용된 코드는 특정 URL에서 확인할 수 있습니다.



### HazardNet: A Small-Scale Vision Language Model for Real-Time Traffic Safety Detection at Edge Devices (https://arxiv.org/abs/2502.20572)
- **What's New**: 본 논문에서는 차량 증가와 복잡한 도로 네트워크로 인해 심화되는 도시 교통 안전 문제를 해결하기 위한 새로운 모델, HazardNet을 소개합니다. HazardNet은 고급 언어 및 비전 모델의 추론 능력을 활용하여 교통 안전을 개선하도록 설계된 소규모 Vision Language Model입니다. 이번 연구에서 우리는 Qwen2-VL-2B 모델을 미세 조정하여 HazardNet을 구축하였으며, 이는 오픈소스 대안 중에서 뛰어난 성능을 발휘하는 것으로 평가되었습니다.

- **Technical Details**: HazardNet은 20억 개의 파라미터를 가진 경량화된 모델로 에지 디바이스에서의 효율적인 추론 처리 속도를 지원합니다. 또한, 본 연구에서는 교통 안전과 관련된 실제 상황을 다루기 위해 նոր한 Vision Question Answering (VQA) 데이터셋인 HazardQA를 신규 구축했습니다. 이 데이터셋은 HazardNet의 훈련에 사용 됩니다.

- **Performance Highlights**: 실험 결과, 미세 조정된 HazardNet은 기본 모델에 비해 F1-Score에서 최대 89%의 성능 향상을 보였으며, 일부 경우에는 GPT-4o와 같은 대형 모델과 비교했을 때도 최대 6% 개선된 결과를 나타냈습니다. 이러한 발전은 HazardNet이 실시간으로 신뢰할 수 있는 교통 안전 사건 탐지를 제공할 수 있는 잠재력을 가지고 있음을 강조합니다.



### LISArD: Learning Image Similarity to Defend Against Gray-box Adversarial Attacks (https://arxiv.org/abs/2502.20562)
- **What's New**: 이 논문은 기존의 방어 메커니즘들이 주로 화이트 박스 공격을 기반으로 평가받고 있다는 점을 지적하며, 실제 공격자가 모델의 기울기를 접근할 수 없다는 현실을 반영한 그레이 박스(grayscale box) 공격 상황을 제안합니다. 새로운 방어 메커니즘인 Learning Image Similarity Adversarial Defense (LISArD)를 소개하며, 이는 템포럴과 컴퓨테이셔널 비용을 증가시키지 않으며 그레이 박스와 화이트 박스 공격 모두에 대한 강인함을 제공합니다.

- **Technical Details**: LISArD는 변형된 이미지와 깨끗한 이미지의 임베딩(embedding)을 기반으로 교차 상관 행렬(cross-correlation matrix)을 대각 행렬(diagonal matrix)로 근사화하면서 동시에 분류 학습(classification learning)을 실시하는 방법입니다. 이 접근 방식은 모형이 깨끗한 이미지와 변형된 이미지를 유사하게 인식하도록 유도하여 공격의 영향을 감소시키는 것을 목표로 합니다. 이 메커니즘은 추가적인 훈련 에포크(epoch)나 파라미터 없이 적용 가능합니다.

- **Performance Highlights**: 실험 결과 LISArD는 그레이 박스 공격에 대한 강인함을 입증하며, 다양한 아키텍처에서 사용될 수 있고 화이트 박스 환경에서도 탄력성을 유지한다고 보고되었습니다. 또한, 기존의 Adversarial Distillation(AD) 모델들이 AT(Adversarial Training) 제거 시에 성능이 급격히 하락하는 반면, LISArD는 훈련 비용을 증가시키지 않으면서 우수한 성과를 보였습니다. 이로 인해 LISArD는 기존 이론에서 나오는 약점을 해결하여 다양한 조건에서 효과적으로 방어할 수 있는 가능성을 제시합니다.



### Finer Disentanglement of Aleatoric Uncertainty Can Accelerate Chemical Histopathology Imaging (https://arxiv.org/abs/2502.20532)
- **What's New**: 이번 논문에서는 디지털 병리학 워크플로우를 개선하기 위한 새로운 전략을 제시합니다. 기존의 라벨 없는 화학 이미징에서 데이터 수집 속도를 높이기 위해 빠르게 저정보(low information) 이미지를 스캔하고, 높은 불확실성(aletorric uncertainty)을 가진 지역을 찾아내어 이들을 고품질(high information) 이미지로 재촬영하는 방식을 사용합니다. 이 연구는 HI와 LI 이미지 간의 효과적인 전환을 위해 고유한 방법론을 제안합니다.

- **Technical Details**: 제안된 방법론에서는 먼저 저정보 이미지를 스캔하여 중요한 구역을 선택하고, 이 구역에서 높은 정보 이미지를 생성하는 과정을 포함합니다. 이를 위해 잠재 공간(post-hoc latent space)을 분석하여 높은 불확실성 지역을 구별하는 미세한 분리 기법을 활용합니다. 이 과정은 동적인 이미지 공간에서의 AI 활용을 위한 새로운 접근법으로, 의료 이미지에서의 불확실성을 효율적으로 처리하는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: 이 방법론이 유방 조직의 적외선 분광 데이터에 적용된 결과, 의도된 고화질 데이터가 무작위 재촬영 대비 우수한 세분화 성능(segmentation performance)을 나타내었습니다. 이는 기존의 불확실성 프레임워크에서 해결하지 못한 문제를 다루며, 고유한 하위 카테고리로 분리함으로써 AI의 신뢰도를 높이는 데 크게 기여합니다. 새로운 알고리즘은 동적인 이미지 공간 내에서의 고유한 분석을 통해 실험 디자인의 최적화를 지원하는 첫 사례로 꼽힙니다.



### On the Role of Individual Differences in Current Approaches to Computational Image Aesthetics (https://arxiv.org/abs/2502.20518)
Comments:
          15 pages

- **What's New**: 이번 연구는 이미지 미적 평가(Image Aesthetic Assessment, IAA)의 이론적 기초를 확립하고, 개인 및 집단 평가를 위한 통합 모델을 제안합니다. 제안된 모델은 개인 특성을 분포 형태로 인코딩하며, GIAA(Generic IAA)와 PIAA(Personal IAA) 간의 전이 학습(transfer learning)을 명확하게 정의합니다. 이에 따라 개인 및 집단 특성을 동시에 고려할 수 있는 새로운 방법론입니다.

- **Technical Details**: 모델은 개인 특성의 분포를 입력으로 받아 GIAA와 PIAA를 동시에 추론할 수 있는 기능을 갖추고 있어, 기존의 방법론에서 불명확했던 점을 보완합니다. GIAA는 특성 분포의 평균을 점수 분포의 평균에 매핑하고, PIAA는 각각의 특성 경계점을 점수 경계점에 매핑하는데, 이를 통해 전이 학습 시 외삽(extrapolation)과 내삽(interpolation)의 개념을 명확히 하였습니다.

- **Performance Highlights**: 연구에서는 개인 점수 평균을 단순히 평균하여 제거할 수 없음을 보여주며, 교육 수준이 미적 차이에 가장 큰 영향을 미친다는 사실을 밝혀냈습니다. 제안된 sGIAA라는 데이터 증강(method) 접근법을 통해 집단 구성에 의해 성능이 개선되었고, 제로샷 PIAA 성능이 20.9% 향상되었습니다. 이는 다양한 인구 통계적 요소와 집단 크기의 영향력을 검토한 연구 결과입니다.



### In-Model Merging for Enhancing the Robustness of Medical Imaging Classification Models (https://arxiv.org/abs/2502.20516)
- **What's New**: 이 논문에서는 단일 모델 내에서의 병합을 통해 모델의 강건성을 향상시키는 새로운 접근 방식인 In-model Merging (InMerge)을 제안합니다. 이는 의료 이미지 도메인에서 특히 중요한데, 기존의 병합 기법은 여러 모델을 필요로 합니다. 단일 모델 내에서 유사한 convolutional kernels를 선택적으로 병합함으로써 모델 성능을 향상시키는 혁신적인 방법을 소개합니다. 이 기술을 통해 훈련 과정에서 redundancy를 줄여 강건성을 높입니다.

- **Technical Details**: 제안된 In-model Merging 전략은 CNN의 깊은 층에서 유사한 convolutional kernels를 선택적으로 병합하여, 얕은 층의 특징을 보존합니다. 병합의 유사성 측정은 cosine similarity를 사용하여 두 kernels 간의 각도 차이를 정량화합니다. 훈련 후 추가적인 에폭 만으로 fine-tuning을 수행함으로써 훈련 과정에서의 계산 오버헤드를 최소화하고, 추론 시에는 병합 연산이 없어 추가적인 계산을 추가하지 않습니다.

- **Performance Highlights**: 이 연구에서는 4개의 다양한 데이터셋에 대해 제안된 InMerge 기법의 적용 가능성과 효과성을 입증하였습니다. InMerge로 훈련된 모델은 일반적으로 훈련된 모델에 비해 상당한 성능 개선을 보여주었습니다. 또한, 이 방법은 적은 훈련 비용으로 다양한 CNN 아키텍처에 통합될 수 있는 플러그 앤 플레이 모듈로 제공됩니다.



### Best Foot Forward: Robust Foot Reconstruction in-the-wild (https://arxiv.org/abs/2502.20511)
- **What's New**: 이 논문은 기존의 불완전한 스캔 및 해부학적 변화를 극복하기 위한 새로운 3D 발 재구성 파이프라인을 제안합니다. 이 방법은 SE(3) 정준화와 시점 예측 모듈을 통해 스캔 정렬의 모호성을 해결하고, 주목 기반 네트워크를 활용하여 누락된 기하학을 완성합니다. 또한, 이 접근법은 임상적으로 검증된 해부학적 정확성을 유지하면서도 주목할 만한 성능을 달성합니다.

- **Technical Details**: 파이프라인은 두 단계로 나뉘며, 첫 번째 단계에서 Structure-from-Motion (SfM) 및 Multi-View Stereo (MVS)를 사용하여 카메라 포즈를 추정하고 초기 포인트 클라우드를 생성합니다. 이후 모양 완성 모듈이 누락된 기하학을 채워 주므로, 최종적으로 고밀도의 완전한 재현을 제공합니다. 이를 통해 Synthetic training data를 사용하여 각 단계에서 효과적으로 문제를 해결할 수 있습니다.

- **Performance Highlights**: 이 연구는 ROBUSTNESS, feature accuracy, surface quality와 같은 재구성 메트릭에서 기존의 COLMAP 및 최첨단 Gaussian splatting 방법보다 뛰어난 성능을 보여줍니다. 또한, 다양한 속성을 가진 Hike3D 데이터셋을 통해 발 모양에 대한 보다 강력한 모델링이 가능해졌습니다. 이 기술은 의료 및 소매 산업에서 모바일 기반 3D 스캔의 새로운 기회를 열어줄 것입니다.



### CoCa-CXR: Contrastive Captioners Learn Strong Temporal Structures for Chest X-Ray Vision-Language Understanding (https://arxiv.org/abs/2502.20509)
- **What's New**: 이번 연구에서는 CXR(Chest X-Ray) 분석을 위한 새로운 접근 방식을 제안합니다. 특히, CXR 보고서에서 진행 상황을 기술하는 내용을 이미지 쌍과의 의미적 차이와 잘 정렬시키는 것을 중점적으로 다룹니다. 이를 위해 CXR 보고서를 처리하는 파이프라인과 CoCa-CXR이라는 새로운 모델을 구현하여 이미지 분석과 보고서 생성을 동시에 개선하고자 합니다.

- **Technical Details**: CXR-4 데이터셋을 통해 4개의 하위 데이터셋을 구성하였으며, 이는 MIMIC-CXR 이미지 및 보고서를 기반으로 합니다. 이 데이터셋은 대형 언어 모델(LLM)을 사용해 보고서를 기술적 및 비교적 구성 요소로 분리하여 학습할 수 있도록 합니다. CoCa-CXR 모델은 이미지 쌍 간의 지역 차이를 인식하기 위해 새로운 지역적 교차 주의 모듈을 포함하고 있습니다.

- **Performance Highlights**: CoCa-CXR은 진행 분석 및 보고서 생성 작업에서 기존의 SOTA(상태 최선 모델)와 비교하여 뛰어난 성능을 보입니다. 특히, MS-CXR-T 진행 분류에서 65.0% 평균 테스트 정확도를 기록했으며, BioViL-T 모델보다 4.8% 향상된 결과를 나타냈습니다. 추가적으로 MIMIC-CXR에서 RadGraph F1 메트릭으로 24.2%를 달성하여 Med-Gemini 모델과 유사한 성능을 보였습니다.



### EgoNormia: Benchmarking Physical Social Norm Understanding (https://arxiv.org/abs/2502.20490)
- **What's New**: 이 논문은 Norms(규범) 이해 및 추론을 위한 새로운 데이터셋인 EgoNormia를 소개합니다. 총 1,853개의 자기 중심의 동영상(interaction videos)과 관련 질문들이 포함되어 있으며, 이는 기계가 다양한 규범들 간의 trade-off(상충) 문제를 이해하도록 도와줍니다.

- **Technical Details**: EgoNormia 데이터셋은 안전(safety), 개인정보 보호(privacy), 근접성(proxemics), 공손성(politeness), 협력(cooperation), 조정/능동성(coordination/proactivity), 소통/가독성(communication/legibility) 등 일곱 가지 규범적 행동(normative actions)을 평가하는데 초점을 맞추고 있습니다. 이 데이터셋의 구축에 있어 비디오 샘플링(video sampling), 자동 답변 생성(automatic answer generation), 필터링(filtering) 및 인간 검증(human validation)을 활용한 새로운 파이프라인(pipeline)을 제안합니다.

- **Performance Highlights**: 현재 최첨단 비전-언어 모델(VLMs)은 EgoNormia에서 최대 45%를 기록하며, 인간 기준(92%)과 비교할 때 규범적 이해가 부족함을 보여줍니다. 이 논문에서는 안전, 개인정보 보호 및 협력과 소통 능력의 부족으로 인해 실제 에이전트(real-world agents)에 적용할 경우 발생할 수 있는 큰 위험을 강조합니다. 또한, 검색 기반 생성 방법을 통해 EgoNomia를 사용하여 VLMs의 규범적 추론(normative reasoning) 능력을 향상시킬 수 있음을 보여줍니다.



### VideoA11y: Method and Dataset for Accessible Video Description (https://arxiv.org/abs/2502.20480)
Comments:
          ACM CHI 2025

- **What's New**: 두꺼운 비디오 콘텐츠의 생성 속도가 빨라짐에 따라, 시각 장애인과 저시력 사용자(BLV 사용자)들이 디지털 접근성(a11y)에서 겪는 격차가 더욱 확대되고 있습니다. 본 연구에서는 이러한 격차를 해소하기 위해 다중 모달 대형 언어 모델(MLLMs)과 비디오 접근성 지침을 활용하여 BLV 사용자에게 맞춤화된 비디오 설명을 생성하는 VideoA11y 접근 방식을 제안합니다. VideoA11y-40K 데이터셋은 BLV 사용자를 위해 설명된 40,000개의 비디오로 구성되어 있으며, 이전의 모든 비디오 설명 데이터셋 중 가장 크고 포괄적입니다.

- **Technical Details**: VideoA11y는 BLV 사용자를 위해 비디오의 내용을 명확하게 설명하기 위한 42개의 오디오 설명(AD) 지침을 기반으로 하는 새로운 방법론입니다. 이를 통해 비디오 설명을 생성하거나 기존 주석을 개선할 수 있으며, 이는 고급 MLLMs를 활용하여 이루어집니다. 연구에서는 비디오 설명의 품질을 평가하기 위해 총 5개의 사용자 연구를 실시하였고, 347명의 시각 있는 참가자, 40명의 BLV 사용자 및 7명의 전문 오디오 서술자와 함께 결과를 분석하였습니다.

- **Performance Highlights**: 연구 결과, VideoA11y가 생성한 비디오 설명은 초보적 인간 주석보다 모든 평가 지표에서 우수한 품질을 보였고, 훈련된 인간 주석의 품질과도 비교할 만한 결과를 도출했습니다. 즉, 비디오 설명의 기술적 기준을 대폭 향상시키고 BLV 사용자에게 더 나은 접근성을 제공하는 데 기여할 것으로 보입니다. 또한 VideoA11y-40K 데이터셋은 앞으로의 비디오 접근성 연구에 중요한 자원이 될 것입니다.



### LLM Post-Training: A Deep Dive into Reasoning Large Language Models (https://arxiv.org/abs/2502.21321)
Comments:
          31 pages, 7 figures, 3 tables, 375 references

- **What's New**: 최근 논문에서는 LLMs(대규모 언어 모델)가 자연어 처리 분야에서 많은 변화를 가져왔음을 강조하고 있습니다. 이 연구는 사전 훈련(pretraining)뿐만 아니라 후속 훈련(post-training) 기법에 대한 집중적인 탐구를 통해 LLM의 성능을 극대화하려는 노력에 주목하고 있습니다. 후속 훈련 기법들은 LLMs의 지식을 세분화하고, 추론 능력을 향상시키며, 사용자 의도 및 윤리적 고려 사항과 더 잘 조화를 이루도록 돕습니다.

- **Technical Details**: 이 논문은 LLMs의 훈련 단계를 두 가지로 나누어 설명합니다. 사전 훈련은 대규모 데이터를 기반으로 다음 토큰 예측(next-token prediction) 타겟을 사용하고, 후속 훈련은 여러 번의 세밀한 조정(fine-tuning)과 정렬(alignment)을 포함합니다. 후속 훈련 기법은 LLM의 행동을 개선하고, 인간의 의도에 맞게 일관성을 갖도록 조정하는 데 목표를 두고 있습니다. 이 과정에서 LoRA(저순위 적응)와 리인포스먼트 러닝(Reinforcement Learning) 같은 접근 방법이 도입됩니다.

- **Performance Highlights**: 후속 훈련을 통해 LLMs는 사용자 의도와 윤리적 요구 사항에 더 잘 맞춰질 수 있습니다. 미세 조정(fine-tuning)과 같은 방법은 특정 작업에서의 성능을 크게 향상시키지만, 과적합(overfitting)이나 높은 계산 비용 같은 문제를 초래할 수 있습니다. 또한, 리인포스먼트 러닝을 통해 LLMs는 다이나믹한 피드백을 활용하여 더 나은 적응력을 발휘할 수 있으며, 테스트 시 조정(test-time scaling) 기법이 적용되어 성능을 최적화할 수 있습니다.



### TomoSelfDEQ: Self-Supervised Deep Equilibrium Learning for Sparse-Angle CT Reconstruction (https://arxiv.org/abs/2502.21320)
- **What's New**: 최근 심층 학습(deep learning)은 이미징(in imaging) 역문제를 해결하기 위한 강력한 도구로 떠오르고 있습니다. 하지만 기존 접근법의 대부분은 ground truth 이미지와 짝지어진 훈련 데이터가 필요하여 의료 응용 분야에서는 데이터 확보가 어렵습니다. 이 논문에서는 TomoSelfDEQ라는 새로운 self-supervised Deep Equilibrium (DEQ) 프레임워크를 소개합니다. 이 프레임워크는 spars-angle CT 재구성을 위해 부족한 샘플링된 측정치에서 직접 훈련할 수 있습니다.

- **Technical Details**: TomoSelfDEQ는 non-unitary forward operator를 처리할 수 있는 이론적 보장을 제공합니다. 이를 통해 self-supervised 손실과 full-supervised 손실 간의 일반적인 관계를 확립하고 Jacobian-Free Backpropagation (JFB)을 적용하여 학습을 최적화합니다. 기존 방법과 달리 TomoSelfDEQ는 다양한 이미징 변환에서 잘 작동할 수 있는 특성을 지니고 있습니다. 이를 통해 CT forward operator뿐만 아니라 다른 비단위 변환에서의 응용이 가능해집니다.

- **Performance Highlights**: TomoSelfDEQ는 sparse-angle CT 데이터에서 최첨단 결과를 제공하는 것으로 입증되었습니다. 특히 16개의 투영 각도만 사용하여도 기존 self-supervised 방법보다 우수한 성능을 발휘했습니다. 실험은 TomoSelfDEQ가 기존 기술들과 비교하여 뛰어난 성과를 거두었음을 보여주며, ground truth 데이터 없이도 최적의 성능을 보장합니다. 이는 이미징 역문제 커뮤니티에서 ground truth 없이 학습하는 연구에 더욱 관심을 불러일으킬 것입니다.



### AutoComb: Automated Comb Sign Detector for 3D CTE Scans (https://arxiv.org/abs/2502.21311)
Comments:
          10 pages, 5 figures

- **What's New**: 이번 논문은 Comb Sign을 정확히 탐지하기 위해 완전 자동화된 기술을 제안하는 첫 번째 연구입니다. 기존의 탐지 방법들은 수동적이고 시간 소모적이며, 다중 평면 이미징의 필요성으로 인해 주관적 해석에 취약합니다. 본 연구에서는 CTE 스캔에서 Comb Sign을 탐지하기 위한 혁신적인 방법론을 소개합니다.

- **Technical Details**: 제안된 방법은 병리적 과혈관화(vascularity)를 나타내는 영역을 초세혈관 분기(bifurcations)와 벽 강화(wall enhancement)를 통해 식별하여 확률적(probabilistic) 지도를 개발합니다. 이 과정은 Deep Learning segmentation 모델, Gaussian Mixture Model (GMM), vesselness filter를 이용한 혈관 추출(vessel extraction), 그리고 이웃 최대화(neighborhood maximization)를 통한 반복적인 혈관성(vesselness) 향상을 포함한 단계적 algorithmic 모듈을 통해 이루어집니다.

- **Performance Highlights**: 실험 결과, 제안된 파이프라인은 Comb Sign을 효과적으로 식별하며, 크론병(Crohn's disease) 및 관련된 과혈관 상태에서 진단 정확도를 향상시키는 객관적이고 정확한 도구를 제공합니다. Comb Sign은 이러한 조건에서 중요한 바이오마커로 간주됩니다.



### RoboBrain: A Unified Brain Model for Robotic Manipulation from Abstract to Concr (https://arxiv.org/abs/2502.21257)
- **What's New**: 최근의 Multimodal Large Language Models (MLLMs) 발전은 놀라운 성능을 보이지만, 로봇 상황, 특히 장기 조작 작업에서는 많은 한계를 드러내고 있습니다. 이 논문에서는 ShareRobot이라는 고품질의 이질적인 데이터셋을 소개하는데, 이는 작업 계획, 물체의 affordance, 그리고 엔드 이펙터 궤적과 같은 다차원 정보를 레이블링합니다. RoboBrain은 이러한 데이터를 기반으로 개발된 MLLM 모델로, 로봇 조작 능력을 향상시키기 위해 다단계 훈련 전략과 긴 비디오 및 고해상도 이미지를 활용합니다.

- **Technical Details**: RoboBrain은 LLaVA 프레임워크 기반으로 구성되며, 광학 인코더, 프로젝터, 대형 언어 모델(LLM)의 세 가지 주요 구성 요소로 이루어집니다. 특히, 우리는 SigLIP 모델을 사용하여 인코더를 구현하였으며, 이는 이미지-텍스트 쌍에만 작동하는 시그모이드 손실 함수를 채택하고 있습니다. LoRA 방법론을 사용하여 RoboBrain의 affordance 인식 및 궤적 예측 능력을 훈련하고, 이를 통해 대규모 모델에 대한 효율적인 파라미터 조정을 할 수 있었습니다.

- **Performance Highlights**: RoboBrain은 다양한 로봇 작업에 걸쳐 최첨단 성능을 기록하였으며, 이는 로봇 두뇌 능력을 향상시키기 위한 잠재력을 강조합니다. 추가 실험에서는 ShareRobot의 활용이 훈련에 미치는 영향을 관찰하고, 로봇 데이터의 비율 변화에 따른 효과를 분석하였습니다. RoboBrain은 일반 및 로봇 벤치마크에서 기존 모델에 비해 우수한 성능을 보여주며, 구조적 OCR 데이터에 대한 이해 능력에서도 성공적으로 평가되었습니다.



### Same accuracy, twice as fast: continuous training surpasses retraining from scratch (https://arxiv.org/abs/2502.21147)
- **What's New**: 이 논문은 기존의 데이터로 훈련된 모델을 활용하여 새로운 데이터에 대한 훈련 비용을 줄이는 방법을 제시합니다. 기존 모델을 사용함으로써 새로운 데이터에서도 성능을 유지하거나 향상시킬 수 있으며, 이를 통해 리소스를 절약할 수 있습니다. 특히, 모델과 데이터 크기가 커질수록 'from scratch' 방식의 훈련이 비효율적이라는 점을 강조합니다.

- **Technical Details**: 논문은 SGD 업데이트 규칙을 기반으로 초기화, 정규화(regularization), 데이터 선택(data selection), 하이퍼 파라미터(hyper-parameters)와 같은 최적화 측면을 통해 계산 비용을 크게 줄일 수 있는 방안을 모색합니다. 이러한 각 측면에 대해 효율적인 첫 단계 방법(first-step methods)을 제안하고, 이를 통해 모델의 학습 속도를 향상시킬 수 있음을 보입니다. 더욱이, 이러한 방법들은 서로 보완적이어서 결합했을 때 더 큰 계산 절약 효과를 얻을 수 있습니다.

- **Performance Highlights**: 제안된 방법들은 여러 컴퓨터 비전 작업에서 최대 2.7배의 계산 시간 절약을 보여주며, 이는 재훈련이 반복적으로 이루어지는 경우 의미 있는 차이를 생성합니다. 또한 다양한 이미지 분류 데이터셋과 다중 작업(multi-task) 환경 및 도메인 점진적 시나리오에서 훈련 효율성 향상을 입증했습니다. 이러한 결과는 제안된 방법이 머신러닝 모델 재훈련의 계산 부담을 줄일 수 있는 잠재력을 가지고 있음을 강조합니다.



### "No negatives needed": weakly-supervised regression for interpretable tumor detection in whole-slide histopathology images (https://arxiv.org/abs/2502.21109)
- **What's New**: 본 논문에서는 약한 감독 하에 regression을 이용하여 종양 탐지에 접근하는 새로운 방법을 제안합니다. 기존의 여러 종양 탐지 접근 방식은 분류 기반 모델을 사용했지만, 이 연구는 다양한 암 유형에서 종양 비율을 추정하고 이를 통해 성능을 개선했습니다. 주요 기여는 기존의 대량 주석 데이터 없이도 효율적으로 종양을 검출할 수 있는 방법을 제시한 것입니다.

- **Technical Details**: 본 연구에서는 Multiple Instance Learning (MIL) 기술을 활용하여 종양 영역 비율을 예측하는 회귀 모델을 개발했습니다. 종양 비율은 병리학자가 주관적으로 결정한 값으로, 논문에서는 이 값을 직접 목표로 설정하여 weakly-supervised regression framework를 구성했습니다. 새로운 비율 목표인 "amplification technique"를 도입하여 모델 학습에서의 효과를 극대화하였으며, 다양한 데이터셋을 통해 성능을 검증했습니다.

- **Performance Highlights**: 제안된 방법은 여러 조직 유형에서 안정성과 해석 가능성을 강조하여, 종양 비율 목표의 노이즈와 변동성에 대한 연구를 포함하고 있습니다. 결과적으로, 다양한 범주의 암에서 종양 탐지 성능이 향상되었음을 보여주며, 이는 실제 임상 환경에서의 적용 가능성을 높입니다. 또한, 모델 예측의 시각적 해석을 통해 AI의 설명 가능성을 높이는 방안을 제공했습니다.



### A Non-contrast Head CT Foundation Model for Comprehensive Neuro-Trauma Triag (https://arxiv.org/abs/2502.21106)
- **What's New**: 최근 AI와 의료 영상 분야의 발전은 응급 상황에서 머리 CT 해석을 혁신적으로 변모시킬 잠재력을 보여주고 있습니다. 본 연구에서는 다양한 신경 외상 신호를 정확하고 효율적으로 탐지하기 위한 3D 기초 모델을 도입하였습니다. 자동 레이블링을 위한 대형 언어 모델(LLMs)을 활용하여 중요한 조건에 대한 종합적인 다중 레이블 주석을 생성하였습니다.

- **Technical Details**: 머리 CT 기초 모델 개발을 위해 신경 방사선 전문의가 긴급 임상적 주의가 필요한 신경 외상 발견 리스트를 작성했습니다. 이 리스트를 바탕으로 대규모 데이터셋에 대한 레이블을 자동으로 생성하였으며, 다중 모드 미세 조정을 통해 두 개의 작업 특화된 비전 네트워크를 통합하였습니다. 이 모델은 hemorrhage(subtype), brain anatomy parcellation 등의 다양한 작업을 수행하도록 훈련되었습니다.

- **Performance Highlights**: 우리 모델은 응급 방사선학에서 AI 지원 신경 외상 진단의 기준을 제시하며, 다양한 신경 외상 발견을 종합적으로 탐지할 수 있는 능력을 보여주었습니다. 대조군인 CT-CLIP과 비교하여 주요 신경 외상 발견에 대한 정확도가 높았으며, AUC 점수는 0.861로 나타났습니다. 이러한 성과는 향후 AI 기반 의료 영상 진단의 재정의에 기여할 것으로 기대됩니다.



### Adaptive Accelerated Proximal Gradient Methods with Variance Reduction for Composite Nonconvex Finite-Sum Minimization (https://arxiv.org/abs/2502.21099)
- **What's New**: 이번 논문에서는 {	t AAPG-SPIDER}라는 새로운 방법을 제안합니다. 이 방법은 적응형 가속 근접 경량(Adaptive Accelerated Proximal Gradient, AAPG) 기법을 사용하여 비선형 복합 유한합 함수의 최소화를 수행합니다. 특히, {	t AAPG-SPIDER}는 적응형 스텝 사이즈, Nesterov의 외삽법(extrapolation), 그리고 SPIDER라는 재귀적 확률 경로 통합 추정기를 결합하여 성능을 향상시킵니다.

- **Technical Details**: 특히, {	t AAPG-SPIDER}는 비확률 및 전체 배치 풀링 환경에서도 {	t AAPG}로 단순화될 수 있습니다. 이들은 학습률(learning rate)이 필요 없는 첫 번째 방법으로, 복합 최적화 문제 클래스에 대해 최적의 반복 복잡도를 달성합니다. {	t AAPG}는 $	extcal{O}(N 	heta^{-2})$, {	t AAPG-SPIDER}는 $	extcal{O}(N + 	extsqrt{N} 	heta^{-2})$로 구간 내 정적 점을 찾습니다.

- **Performance Highlights**: Kurdyka-Lojasiewicz 가정 하에서 두 방법의 비에르고딕(non-ergodic) 수렴 속도를 확립하였습니다. 초기 실험에서는 희소 물리 복구(sparse phase retrieval)와 선형 고유값 문제(linear eigenvalue problem) 대회에서 {	t AAPG-SPIDER}와 {	t AAPG}가 기존 방법과 비교하여 우수한 성능을 보임을 입증하였습니다.



### When Unsupervised Domain Adaptation meets One-class Anomaly Detection: Addressing the Two-fold Unsupervised Curse by Leveraging Anomaly Scarcity (https://arxiv.org/abs/2502.21022)
- **What's New**: 이 논문은 비지도적 이상 탐지 (Unsupervised Anomaly Detection, UAD) 를 위한 최초의 완전 비지도적 도메인 적응 (Unsupervised Domain Adaptation, UDA) 프레임워크를 소개합니다. 전통적인 이상 탐지 기법은 도메인 변경이 발생할 경우 성능 저하가 심각하게 발생하는데, 이는 실제 상황에서 피하기 어려운 문제입니다. 이 논문에서는 비지도적 문제의 두 가지 측면을 강조하며, 이상 데이터는 일반적으로 희귀하다는 아이디어를 활용하여 이 문제를 해결하는 혁신적인 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 우선 대상 특징 공간에서 주요 클러스터를 식별하고 이를 정상 원본 특징과 정렬합니다. 또한, ResNet 기반 특징 추출기를 사용하여 원본 및 대상 특징을 처리하며, CLIP 시각 인코더를 이용해 대응하는 대상 특징을 생성하고 K-평균 군집화를 통해 주요 클러스터의 샘플을 식별합니다. 이후 주요 클러스터의 샘플은 ResNet 기반 특징 공간으로 매핑되고 원본 특징과 정렬됩니다.

- **Performance Highlights**: 이 연구는 여러 표준 UDA 벤치마크에서 실험을 수행하여 제안된 방법의 효과성을 입증합니다. 또한, 제안된 접근법은 최신 수준의 (state-of-the-art, SoA) 성능을 달성했으며, 이는 기존 few-shot 방법들과 비교했을 때도 높은 성능을 제공합니다. 이 프레임워크는 모듈식으로 구성되어 있어 유연한 구성 요소 변경이 가능하며, 다양한 적응 전략을 지원합니다.



### FedDyMem: Efficient Federated Learning with Dynamic Memory and Memory-Reduce for Unsupervised Image Anomaly Detection (https://arxiv.org/abs/2502.21012)
- **What's New**: 이 논문에서는 연합 학습( federated learning)을 기반으로 한 비지도 이미지 이상 탐지( unsupervised image anomaly detection )의 새로운 방법인 FedDyMem을 제안합니다. FedDyMem은 동적 메모리(dynamic memory)와 메모리 축소(memory-reduce) 기법을 이용하여 클라이언트가 소유한 정상 샘플을 기반으로 지식 공유를 촉진합니다. 이 방법은 데이터의 프라이버시를 보장하면서도 다양한 산업 및 의료 데이터셋에서 효과적인 성능을 입증합니다.

- **Technical Details**: FedDyMem의 핵심은 모든 클라이언트가 하나의 클래스(즉, 정상 샘플)로 구성되어 있다고 가정하고, 클라이언트의 동적 메모리 뱅크를 통해 서버와의 지식 공유를 구축하는 것입니다. 이를 위해 메모리 생성기(memory generator)와 메트릭 손실(metric loss)을 사용하여 정상 샘플의 특성 분포의 일관성을 높입니다. 또한, k-평균 집합(k-means aggregation) 기술을 사용하여 클라이언트 간의 집계 과정에서 분포의 바이어스를 줄입니다.

- **Performance Highlights**: 다양한 산업 및 의료 분야의 11개 공개 데이터셋을 기반으로 6개 제품 또는 건강 검진 유형으로 분류하였으며, 각 유형에 대해 FedDyMem의 성능을 평가했습니다. 실험 결과, FedDyMem은 비지도 이미지 이상 탐지에서 최첨단 성능을 달성하며 통신 효율성을 높이는 동시에 이상 탐지 성능을 유지하는 데 성공했습니다. 이러한 결과는 FedDyMem이 기존 방법과 비교하여 우수한 연합 비지도 학습 성능을 제공함을 보여줍니다.



### Guiding Quantitative MRI Reconstruction with Phase-wise Uncertainty (https://arxiv.org/abs/2502.20877)
Comments:
          Submitted to MICCAI2025

- **What's New**: 본 논문에서는 정량적 자기공명영상(quantitative magnetic resonance imaging, qMRI)에서 불확실성 정보(uncertainty information)를 활용한 새로운 접근법인 PUQ를 소개합니다. 기존의 다양한 연구들은 불확실성을 측정하는 데 집중했지만, 이를 재구성 성능 향상에 활용하는 방법은 거의 다루지 않았습니다. 본 접근법은 qMRI 재구성을 위한 혁신적인 방법으로, 이전의 연구와 비교해 한 단계 더 나아간 것으로 평가받고 있습니다.

- **Technical Details**: PUQ는 두 단계의 재구성(reconstruction) 및 매개변수 적합(parameter fitting) 프레임워크를 사용합니다. 첫 번째 단계에서 각 단계의 불확실성은 재구성 중에 추정되며, 두 번째 단계에서는 이 정보를 매개변수 적합에 사용합니다. 이 설계는 불확실성이 서로 다른 단계의 신뢰성을 반영하고 매개변수 적합 시 정보 통합을 안내하도록 합니다.

- **Performance Highlights**: PUQ는 건강한 피험자로부터 얻은 T1 및 T2 매핑 데이터세트를 평가하였으며, 기존의 qMRI 재구성 방법들과 비교하여 매개변수 매핑에서 최신 성능(state-of-the-art performance)을 달성했습니다. 이는 불확실성 가이드를 통해 재구성 성능을 효과적으로 향상시킬 수 있음을 보여줍니다.



### Oscillation-Reduced MXFP4 Training for Vision Transformers (https://arxiv.org/abs/2502.20853)
- **What's New**: 이 논문에서는 FP4 정밀도로 트랜스포머를 사전 훈련하는 새로운 방법인 TetraJet을 제안합니다. 저 precision 훈련은 큰 신경망의 학습을 가속화하는 유망한 기술로 부각되고 있으며, FP4 형식은 특히 속도 측면에서 강력한 잠재력을 가지고 있습니다. 그러나 FP4 훈련의 정확도 저하 문제를 해결하기 위해 MXFP4 데이터를 사용하면서 발생하는 문제에 대해 체계적으로 연구한 결과를 보여줍니다.

- **Technical Details**: TetraJet은 MXFP4 계산을 사용하여 트랜스포머의 앞으로 및 앞으로(Backward) 패스를 수행하는 새로운 훈련 방법론입니다. 이 방법은 모든 가중치/활성화/그래디언트 텐서를 MXFP4로 양자화하여 하드웨어의 가속화 잠재력을 최대한 활용합니다. 특히, 저자들은 weight oscillation 문제를 발견하고 이를 해결하기 위한 Q-EMA(EMA Quantizer) 및 Q-Ramping(Adaptive Ramping Optimizer) 방법을 제안하여 성능을 개선했습니다.

- **Performance Highlights**: TetraJet을 사용한 광범위한 실험 결과, 기존의 4비트 훈련 방법에 비해 항상 더 나은 성능을 나타내었습니다. 특히, Q-EMA 및 Q-Ramping은 진동을 효과적으로 줄여줌으로써 추가적인 향상을 가져왔고, 기반선(기존 방법) 대비 정확도 저하를 50% 이상 줄이는 데 성공했습니다. 결과적으로 TetraJet은 풀 정밀도 훈련(Full Precision Training)과 비교할 때 경쟁력 있는 성과를 달성했습니다.



### Delta-WKV: A Novel Meta-in-Context Learner for MRI Super-Resolution (https://arxiv.org/abs/2502.20852)
Comments:
          This paper has been published to MICCAI 2025. Feel free to contact on nomodeset@qq.com

- **What's New**: 새로운 모델 Delta-WKV는 MRI 초해상도를 위한 혁신적인 접근 방식을 제안합니다. 이 모델은 Meta-in-Context Learning (MiCL)과 Delta 규칙을 통합하여 MRI 이미지에서 지역적(local) 및 전역적(global) 패턴을 효과적으로 인식합니다. Delta-WKV는 추론 과정에서 동적으로 가중치를 조정하여 더 적은 파라미터와 계산량으로 패턴 인식 능력을 향상시키는 데 초점을 맞추고 있습니다.

- **Technical Details**: Delta-WKV는 4방향 스캐닝 메커니즘과 시간을 혼합하고 채널을 혼합하는 구조를 통해 장기 의존성을 포착하면서 고주파 세부사항을 유지합니다. 이 모델은 Receptance Weighted Key Value (RWKV)에서 영감을 받아 MLP를 대체하고 공간 혼합 선형 주의 블록을 추가하여 효율적인 상관 모델링을 달성합니다. Delta-WKV는 전통적인 CNN 아키텍처의 한계를 극복하기 위해 설계되었으며, 특히 의료 이미지에 맞춰 최적화된 구조를 가지고 있습니다.

- **Performance Highlights**:  Delta-WKV는 IXI 및 fastMRI 데이터셋에서 기존 방법들보다 우수한 성능을 보였습니다. PSNR은 0.06 dB 향상되었고 SSIM은 0.001 증가했으며, 훈련 및 추론 시간을 15% 이상 단축했습니다. 이러한 결과는 Delta-WKV가 큰 데이터셋과 고해상도 이미징을 사용할 때 효율적이고 잠재력이 있음을 보여줍니다.



### Towards Semantic 3D Hand-Object Interaction Generation via Functional Text Guidanc (https://arxiv.org/abs/2502.20805)
- **What's New**: 이 논문에서는 기능 텍스트에 의해 구동되는 3D 손-물체 상호작용(HOI) 생성을 위한 새로운 두 단계 프레임워크인 Functional Grasp Synthesis Net (FGS-Net)을 제안합니다. 기존 방법이 손-물체 상호작용의 기능적 의미를 충분히 고려하지 못한 반면, FGS-Net은 기능적 텍스트 프롬프트를 기반으로 손의 자세와 물체의 위치를 더욱 인간의 의도에 부합하게 조정합니다.

- **Technical Details**: FGS-Net은 기능 텍스트를 기반으로 손과 물체의 3D 모델을 생성하는 Functional Grasp Generator (FGG)와 손-물체 간의 자세를 최적화하는 Functional Grasp Refiner (FGR)라는 두 가지 모듈로 구성되어 있습니다. FGG는 주어진 텍스트 입력에 따라 3D 모델을 생성하고, FGR은 Object Pose Approximator와 에너지 함수를 통해 손과 물체의 상대적인 위치를 정제합니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안된 FGS-Net은 추가적인 3D 주석 데이터 없이도 기존의 재구성 및 그립 합성 방법들과 비교하여 우수한 성능을 보여주었습니다. 이로써 사람의 의도와 일치하는 고품질의 손-물체 상호작용 모델을 효과적으로 생성할 수 있음을 입증하였습니다.



### Autoregressive Medical Image Segmentation via Next-Scale Mask Prediction (https://arxiv.org/abs/2502.20784)
Comments:
          10 pages, 4 figures

- **What's New**: 이번 연구에서는 복잡한 해부학적 영역에서의 의료 이미지 분할을 위한 새로운 방법인 AR-Seg를 제안합니다. AR-Seg는 통합된 아키텍처 내에서 모든 이전 스케일 간의 의존성을 모델링하여 다음 스케일 마스크를 점진적으로 예측하는 오토회귀(autoregressive) 프레임워크입니다. 이 연구의 중요한 기여는 다음 스케일 예측을 위한 기술을 최초로 도입한 것입니다.

- **Technical Details**: AR-Seg는 다중 스케일 마스크 오토인코더를 사용하여 마스크를 K개의 다중 스케일 토큰 맵으로 양자화합니다. 이 후, 다음 스케일 오토회귀 메커니즘을 통해 점진적으로 마스크를 예측하며, 최종 마스크는 여러 샘플 결과를 조합하여 생성하는 합의 집계(consensus-aggregation) 전략을 통해 제공합니다. 각 구성 요소에서 교육(training)과 추론(inference) 과정이 상세히 설명됩니다.

- **Performance Highlights**: AR-Seg는 다양한 모드와 데이터비율에서 두 개의 벤치마크 데이터셋에 대한 광범위한 실험을 통해 최첨단(SOTA) 방법들을 능가하는 성과를 보였습니다. 특히, 중간의 조잡한 세분화(coarse-to-fine segmentation) 과정을 명시적으로 시각화하여 의료 환경에서의 신뢰성을 높였습니다.



### MedHallTune: An Instruction-Tuning Benchmark for Mitigating Medical Hallucination in Vision-Language Models (https://arxiv.org/abs/2502.20780)
- **What's New**: 이번 연구에서는 의료 분야의 Vision-Language Models (VLMs)에서 발생하는 황홀경(hallucination) 문제를 다루기 위해 MedHallTune이라는 대규모 벤치마크를 제안합니다. MedHallTune은 100,000개 이상의 이미지와 1,000,000쌍의 지침으로 구성되어 있으며, 각 데이터는 진실(annotation)로 주석이 달려 있습니다. 이 데이터셋은 의료 VLMs의 성능 평가 및 개선을 목적으로 설계되었습니다.

- **Technical Details**: MedHallTune은 100,000개 이상의 이미지-텍스트 쌍을 샘플링하고, GPT-4o를 활용해 황홀경 및 비황홀경 예제를 포함한 지침 데이터를 생성합니다. 각 데이터 쌍은 두 번의 검증 과정을 통해 생성되어, 잘못된 해석을 필터링합니다. 연구에서는 기존 VLM을 미세 조정(fine-tuning)하여 의료 맥락에서 황홀경의 대응 능력을 향상시키는 방법을 제시합니다.

- **Performance Highlights**: 실험 결과, MedHallTune을 활용한 미세 조정이 여러 의료 및 일반 VLM 모델의 황홀경 관리 능력을 크게 향상시키는 것으로 나타났습니다. 고급 평가 메트릭을 통해, 임상 정확성(clinical accuracy), 관련성(clinical relevance), 정보의 상세 수준(detail level), 위험 수준(risk level) 등에서 성능이 크게 개선되었습니다. 이를 통해, 더욱 신뢰할 수 있는 의료 VLM 개발에 기여하고 있습니다.



### Towards Practical Real-Time Neural Video Compression (https://arxiv.org/abs/2502.20762)
Comments:
          CVPR 2025. The code is available at this https URL

- **What's New**: 이번 논문에서는 높은 압축 비율, 낮은 지연 시간, 그리고 다양한 활용성을 갖춘 실시간 신경 비디오 코덱(Neural Video Codec, NVC)을 도입합니다. 기존의 NVC들은 주로 계산 비용을 줄이는 데 집중했으나, 본 연구에서는 비계산적 운영 비용이 주요 병목 현상임을 확인하고 이를 최소화하기 위한 디자인 개선안을 제시합니다. 특히, 복잡한 명시적 모션 모듈을 제거하고 단일 저해상도 잠재 표현(latent representation)을 사용하여 NVC의 속도를 크게 향상시키면서 압축 품질은 유지하였습니다.

- **Technical Details**: 논문에서 제안한 DCVC-RT는 평균 인코딩 속도 125.2fps(프레임 당 초속도)와 디코딩 속도 112.8fps로 1080p 비디오 인코딩을 지원하며, H.266/VTM 대비 평균 21%의 비트 전송률(bitrate) 절약을 달성했습니다. 이 모델은 단일 저해상도 기법을 활용하여 메모리 I/O 비용을 줄이고, 연속적인 정수 산술 처리를 통해 다양한 장치에서 일관된 인코딩을 보장합니다. 또한, 모듈 뱅크 기반의 비율 조정(rate control) 방식으로 실제 환경에서의 적응성을 개선했습니다.

- **Performance Highlights**: DCVC-RT는 상업용 하드웨어에서 1080p 비디오의 실시간 인코딩을 가능하게 하며, 전반적으로 21%의 비트 전송률 감소를 제공합니다. NVIDIA RTX 2080Ti에서 인코딩 평균 40fps, 디코딩 34fps를 기록하며, NVIDIA A100 GPU에서는 인코딩 125fps, 디코딩 113fps를 달성했습니다. 이 모델은 기존의 신경 비디오 코덱들보다 18배 이상 빠른 인코딩 속도를 보이며, 실질적인 NVC의 첫 사례로 주목받고 있습니다.



### SemiSAM+: Rethinking Semi-Supervised Medical Image Segmentation in the Era of Foundation Models (https://arxiv.org/abs/2502.20749)
- **What's New**: 이번 연구에서는 SemiSAM+라는 새로운 반지도 학습(SSL) 프레임워크를 제안합니다. 이는 의료 이미지 분할을 위한 제한된 라벨링 데이터로부터 효율적으로 학습할 수 있도록 돕습니다. SemiSAM+는 일반적인 경우에 활용 가능한 프롬프트 기반 기초 모델과 전환 가능한 전문 모델을 결합하여, 매개변수 조정과 데이터 효율성을 극대화합니다.

- **Technical Details**: SemiSAM+는 하나 이상의 프롬프트 기반 기초 모델과 특정 작업에 대해 훈련 가능한 전문 모델로 구성됩니다. 전문 모델은 무작위하게 구축된 프롬프트를 사용하여 고정된 일반 모델과 상호작용하여 유사 라벨을 확보합니다. 이 과정에서 일반 모델의 출력을 통해 전문 모델은 더욱 효율적이고 유익한 감독을 얻습니다.

- **Performance Highlights**: 실험 결과, SemiSAM+는 두 개의 공용 데이터 세트와 하나의 내부 임상 데이터 세트에서 큰 성능 향상을 이뤘습니다. 특히 매우 제한된 주석 상황에서도 강력한 효율성을 보여주는 플러그 앤 플레이 방식으로 다양한 모델에 쉽게 적응될 수 있습니다.



### Subtask-Aware Visual Reward Learning from Segmented Demonstrations (https://arxiv.org/abs/2502.20630)
Comments:
          Project webpage: this https URL

- **What's New**: 이번 논문은 REDS(REward learning from Demonstration with Segmentations)라는 새로운 보상 학습 프레임워크를 소개합니다. 이는 최소한의 감독으로 행동 없는 비디오를 활용하여 로봇 조작 작업에서의 보상을 학습합니다. REDS는 비디오 시연을 세부 작업으로 나누어 각 세그먼트를 실제 보상으로 간주하여 훈련합니다.

- **Technical Details**: REDS는 비디오 세그먼트를 기반으로 한 밀집 보상 함수(dense reward function)를 훈련하며, 이러한 과정에서 Equivalent-Policy Invariant Comparison (EPIC) 거리를 최소화하는 방식으로 실제 보상 신호와의 정렬을 보장합니다. 또한, 대조 학습(constrastive learning) 목표를 사용하여 비디오 표현과 세부 작업을 정렬함으로써 실시간 상호작용 중에 세부 작업을 정확하게 추론합니다.

- **Performance Highlights**: REDS는 Meta-World에서 복잡한 로봇 조작 작업을 수행하는 데 있어 기본 방법들보다 현저하게 개선된 성능을 보여줍니다. 특히, 가구 조립과 같은 복잡한 장기 작업에서 최소한의 인간 개입으로 학습할 수 있는 가능성을 보여줍니다. 또한, REDS는 이전에 보지 못한 작업 및 로봇 구조에 대한 일반화 능력을 지니고 있습니다.



### SafeText: Safe Text-to-image Models via Aligning the Text Encoder (https://arxiv.org/abs/2502.20623)
- **What's New**: 새로운 연구에서는 기존의 정렬 방법(Alignment Methods) 대신 텍스트 인코더(Text Encoder)를 세밀하게 조정함으로써 안전하지 않은 프롬프트에 대해 해로운 이미지 생성을 효과적으로 방지하는 새로운 방법인 SafeText를 제안했습니다. 이는 기존 방식이 확산 모듈(Diffusion Module)을 수정하여 안전한 프롬프트에 대한 이미지 품질을 상당히 저하시키는 문제를 해결합니다. SafeText는 안전하지 않은 프롬프트에 대해 임베딩 벡터를 크게 조정하며, 안전한 프롬프트에 대해서는 최소한의 영향을 미칩니다.

- **Technical Details**: SafeText는 효과성 목표와 유용성 목표를 달성하기 위해 텍스트 인코더를 조정하는 최적화 문제로 형식화됩니다. 이 방법은 두 개의 손실 항을 개발하여 각각 효과성과 유용성 목표를 정량화하고, 표준 경량 기반 방법(예: Adam 옵티마이저)을 사용하여 텍스트 인코더를 미세 조정합니다. 이는 안전하지 않은 프롬프트에 대해 해로운 이미지를 생성하지 않으면서도 안전한 프롬프트에 대한 이미지 품질을 유지할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, SafeText는 안전한 프롬프트 및 사용자 제작과 공격에 의해 생성된 여러 안전하지 않은 프롬프트에 대해 평가했을 때, 기존의 6가지 정렬 방법보다 우수한 성능을 보였습니다. SafeText는 해로운 이미지 생성을 효과적으로 방지하면서 안전한 프롬프트에 대한 이미지 품질을 유지하는 균형을 잘 맞춥니다. 또한 연구진은 논문 수락 후 코드를 공개할 예정입니다.



### Style Content Decomposition-based Data Augmentation for Domain Generalizable Medical Image Segmentation (https://arxiv.org/abs/2502.20619)
- **What's New**: 이 논문에서는 의료 이미지의 학습된 분할 모델이 테스트 환경에서 성능 저하를 겪는 이유를 발견하고, 이를 해결하기 위한 새로운 데이터 증강 방법 "StyCona"를 제안합니다. 국내 의료 이미지에서의 스타일 변화(style shifts)와 내용 변화(content shifts)를 구분하여, 두 가지 모두를 동시에 해결하는 접근 방식을 제시합니다. StyCona는 단순하면서도 효과적인 플러그 앤 플레이(module) 방식으로 추가 학습 없이 모델의 일반화 성능을 크게 향상시킵니다.

- **Technical Details**: StyCona는 주로 이미지의 스타일 코드(style code)와 내용 맵(content map)을 분해하는 구조를 가지고 있으며, 이를 통해 두 가지 도메인 변화 유형을 분리합니다. 스타일 변화는 픽셀 강도의 변화로 나타나며, 내용 변화는 해부학적 구조의 차이에서 비롯됩니다. 이 방식은 singular value decomposition (SVD)를 사용하여 이미지의 내부 구조를 가까이 파악하고, 이를 바탕으로 스타일 및 콘텐츠 증강을 수행합니다.

- **Performance Highlights**: StyCona는 교차 시퀀스, 교차 센터 그리고 교차 모달리티의 의료 이미지 분할 설정에서 실험을 통해 뛰어난 성능을 입증하였습니다. 기존의 최첨단 방법들에 비해 월등한 성능을 보여주어, 도메인 일반화(deep learning domain generalization)가 필요한 다양한 의료 영상 분할 작업에서 효과적인 해결책이 됩니다. 이 연구는 또한, 데이터 증강(Data Augmentation) 기법의 새로운 패러다임을 제안하여 의료 이미지 분할 분야의 발전에 기여합니다.



### Discovering Global False Negatives On the Fly for Self-supervised Contrastive Learning (https://arxiv.org/abs/2502.20612)
- **What's New**: GloFND는 자기 감독 기반의 대비 학습(self-supervised contrastive learning)에서 가짜 부정 쌍(false negatives)을 식별하기 위한 새로운 최적화 접근 방식을 제안합니다. 기존 방법들이 미니배치 내에서 로컬하게 작동하는 것과 달리, GloFND는 전체 데이터셋에서 가짜 부정을 글로벌하게 탐지하여 보다 정밀한 학습을 가능하게 합니다. 이 방법은 훈련 중 각 앵커 데이터의 가짜 부정에 대한 임계값(threshold)을 자동으로 학습하여 효과적인 대비 학습을 지원합니다.

- **Technical Details**: GloFND의 α(알파) 하이퍼파라미터는 가짜 부정의 다양한 정의에 적응할 수 있게 도와줍니다. 이 하이퍼파라미터의 값은 가짜 부정의 수나 가능성에 대한 사전 지식, 표현의 세밀함에 따라 설정할 수 있으며, 최소의 연산 비용으로 계산됩니다. GloFND를 SimCLR, SogCLR 및 CLIP과 같은 대비 기반 메서드와 결합할 경우, 성능 저하 없이 잘 작동하며, 실험에서는 2%의 훈련 시간 증가만 보여줍니다.

- **Performance Highlights**: GloFND는 이미지 및 이미지-텍스트 데이터 실험에서 유의미한 성능 향상을 나타냈습니다. 특히, ResNet-50을 기반으로 한 선행 학습된 모델에서 더욱 효과적인 특성 표현을 생성하는 능력을 보여주었습니다. 실험 결과는 GloFND가 다양한 레이블 비율에 대한 준지도 학습에서도 효과적인 성능을 발휘하며, 전체적인 준지도 성능을 평균화한 결과를 통해 기여도를 입증하였습니다.



### An Integrated Deep Learning Framework Leveraging NASNet and Vision Transformer with MixProcessing for Accurate and Precise Diagnosis of Lung Diseases (https://arxiv.org/abs/2502.20570)
- **What's New**: 이번 연구는 NASNet과 Vision Transformer(ViT)의 장점을 결합한 새로운 딥러닝 프레임워크인 NASNet-ViT를 제안합니다. 이 모델은 폐 질환을 다섯 가지 범주로 분류하는 데 초점을 맞추고 있으며, 조기 및 정확한 진단을 요구하는 여러 폐 질환에 대한 해결책을 제공합니다.

- **Technical Details**: NASNet-ViT는 MixProcessing이라는 다면적 전처리 전략을 사용하여 진단 정확도를 높입니다. MixProcessing은 웨이브렛 변환(wavelet transform), 적응형 히스토그램 평활화(adaptive histogram equalization), 형태학적 필터링(morphological filtering) 기술을 결합하여 특징을 효과적으로 추출합니다.

- **Performance Highlights**: NASNet-ViT 모델은 98.9%의 정확도와 0.99의 민감도(sensitivity), 0.989의 F1-score, 0.987의 특이도(specificity)를 달성하여 기존의 최신 아키텍처를 능가합니다. 이 모델은 25.6MB의 작은 크기와 12.4초의 짧은 계산 시간을 제공하여 실제 임상 환경에서도 효율적으로 사용할 수 있습니다.



### Towards Statistical Factuality Guarantee for Large Vision-Language Models (https://arxiv.org/abs/2502.20560)
- **What's New**: 이 논문에서는 LVLM(대형 비전-언어 모델)의 신뢰성을 높이기 위해 ConfLVLM이라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 기존 LVLM이 생성한 텍스트가 시각적 맥락과 불일치하는 문제를 해결하기 위해 설계되었습니다. ConfLVLM은 통계적 가설 검정을 통해 생성된 텍스트의 신뢰성을 평가하고, 잘못된 주장을 사전에 필터링함으로써 사용자에게 정확한 정보를 제공합니다.

- **Technical Details**: ConfLVLM은 LVLM을 가설 생성기로 간주하고, 각 생성된 텍스트는 개별적인 주장으로 다룹니다. 이 프레임워크는 효율적인 불확실성 측정치를 사용하여 주장을 검증할 수 있도록 설계되었습니다. 통계적 가설 검정 절차를 활용하여, 잘못된 주장을 걸러내고 신뢰할 수 있는 답변만 사용자에게 제공합니다.

- **Performance Highlights**: ConfLVLM은 LLaVa-1.5에 의해 생성된 장면 설명의 오류율을 87.8%에서 10.0%로 대폭 줄였습니다. 이 과정에서 95.3%의 진정한 양성률(true positive rate)을 기록하여, 필터링의 효과성을 입증했습니다. 또한, ConfLVLM은 다양한 LVLM 및 불확실성 측정치와 조합되어 다양한 비전-언어 작업에 유연하게 적용될 수 있는 특징을 가지고 있습니다.



### A Thousand Words or An Image: Studying the Influence of Persona Modality in Multimodal LLMs (https://arxiv.org/abs/2502.20504)
- **What's New**: 최근 대형 언어 모델(LLMs)은 다양한 페르소나(persona)를 구현하는 데 있어 큰 발전을 보여주었습니다. 이 논문은 텍스트와 이미지라는 서로 다른 표현 방식을 갖는 페르소나의 영향력을 처음으로 체계적으로 분석합니다. 새로운 모달리티 평행 데이터셋을 생성하고, LLM이 특정 페르소나의 특성과 시나리오를 얼마나 잘 표현하는지를 평가하기 위한 프레임워크를 개발하였습니다.

- **Technical Details**: 우리는 다양한 연령, 성별, 직업, 위치를 가진 4040명의 페르소나로 이루어진 데이터셋을 생성하였습니다. 각 페르소나는 이미지 전용, 텍스트 전용, 이미지와 짧은 텍스트의 조합, 타이포그래피를 활용한 이미지 네 가지 방식으로 표현됩니다. 이러한 표현방식에 대한 비교 분석을 통해 멀티모달 LLM의 성능을 평가하고, 텍스트 기반 페르소나가 이미지 표현보다 뛰어난 결과를 나타냈습니다.

- **Performance Highlights**: 실험 결과, LLM은 상세한 텍스트로 표현된 페르소나에서 더 많은 언어적 습관을 보였고, 타이포그래피 이미지는 페르소나와의 일관성이 더 높았습니다. 이러한 연구 결과는 LLM이 이미지로 전달되는 페르소나 특정 세부 사항을 종종 간과하고 있음을 밝혔습니다. 이를 통해 향후 연구가 이 격차를 좁히는 데 기여할 수 있도록 하는 기반을 마련합니다.



### Chitranuvad: Adapting Multi-Lingual LLMs for Multimodal Translation (https://arxiv.org/abs/2502.20420)
- **What's New**: 이번 연구에서는 WAT2024의 영문에서 저해상도 다중 모달 번역 과제를 위한 제출물로 Chitranuvad라는 다중 모달 모델을 소개합니다. 이 모델은 다국어 LLM과 비전 모듈을 효과적으로 통합하여 다중 모달 번역을 수행합니다. ViT 이미지 인코더를 사용하여 시각적 표현을 추출하고, 이를 LLM의 공간으로 변환하여 자가 회귀 방식으로 번역을 생성합니다.

- **Technical Details**: Chitranuvad는 다국어 번역을 위해 적응된 대형 다중 모달 모델입니다. 이 모델은 Krutrim LLM을 백본으로 사용하고, 특정 작업에 맞게 시각적 이미지 인코더와 함께 멀티모달 학습을 진행합니다. 모델 훈련 과정에서 이미지를 비전 인코더로 인코딩한 후, 모달리티 프로젝션 레이어를 통해 LLM 임베딩 공간으로 시각적 토큰을 나타냅니다.

- **Performance Highlights**: 모델은 Hindi, Bengali, Malayalam 언어의 3가지 트랙에 참가했으며, 모든 트랙에서 Hindi에 대해 SOTA 성과를 달성했습니다. Chitranuvad는 비주얼 진화 번역 데이터 세트에서 세부 조정 작업의 효율성을 입증하였고, 자가 회귀 방식으로 올바른 번역을 생성하는 능력을 보여주었습니다.



### HVI: A New Color Space for Low-light Image Enhancemen (https://arxiv.org/abs/2502.20272)
Comments:
          Qingsen Yan, Yixu Feng, and Cheng Zhang contributed equally to this work

- **What's New**: 이 논문은 Low-Light Image Enhancement (LLIE) 분야의 새로운 접근 방식을 제안합니다. 기존의 방법들은 주로 RGB (sRGB) 색 공간에 기반을 두어 색 균형 문제와 밝기 왜곡을 발생시킵니다. 새로운 수평/수직-강도 색 공간인 Horizontal/Vertical-Intensity (HVI)를 도입하여 이러한 문제를 해결하고자 합니다.

- **Technical Details**: HVI 색 공간은 편광된 HS 맵과 학습 가능한 강도를 기반으로 정의됩니다. 이 시스템은 빨간색 좌표 간 작은 거리를 강제하여 빨간색 아티팩트를 제거하고, 저조도 영역을 압축하여 검은색 아티팩트를 줄입니다. 또한, 새로운 Color and Intensity Decoupling Network (CIDNet)를 통해 다양한 조명 조건에서 정확한 포토메트릭 맵핑 함수(photometric mapping function)를 학습합니다.

- **Performance Highlights**: 벤치마크 및 아블레이션 실험 결과, HVI 색 공간과 CIDNet을 활용한 접근법이 10개 데이터셋에서 최첨단 방법들보다 우수한 성능을 보였습니다. 이 연구는 저조도 이미지 복원 분야의 기법 개선에 크게 기여할 것으로 기대됩니다.



### SalM$^{2}$: An Extremely Lightweight Saliency Mamba Model for Real-Time Cognitive Awareness of Driver Attention (https://arxiv.org/abs/2502.16214)
Comments:
          The article has been accepted for publication at AAAI 2025

- **What's New**: 이 논문에서는 운전 장면에서 발생하는 운전자의 주의를 인식하기 위한 새로운 실시간 Saliency Mamba 네트워크를 제안합니다. 기존의 방법들에 비해 수천 개의 매개변수를 줄이고도 SOTA 성능을 유지하면서, 최신 Mamba 프레임워크를 기반으로 하고 있습니다. 이러한 접근은 운전 중 특정 목표와 객체에 대한 집중을 높이기 위해 운전 작업과 관련된 의미론적 정보를 활용합니다.

- **Technical Details**: 제안된 모델은 'Bottom-up' 브랜치와 'Top-down' 브랜치로 구성된 이중 구조를 가지고 있습니다. Bottom-up 브랜치는 운전 장면에서 이미지 특징을 추출하고, Top-down 브랜치는 장면의 의미론적 정보를 캡처하여 운전자의 주의를 효과적으로 인식할 수 있도록 합니다. 특히, Selective Channel Parallel Mamba(SCPM) 레이어를 통해 매개변수 폭주 문제를 해결하고, Cross-Modal Attention(CMA) 융합 모듈을 통해 이미지와 의미론적 정보의 효과적인 융합을 가능하게 합니다.

- **Performance Highlights**: SalM2 네트워크는 단 0.08M의 훈련 가능한 매개변수로 운전자의 주의를 가장 잘 예측하는 모델로, 다른 모델에 비해 약 0.09%에서 11.16% 만큼의 매개변수만을 사용하여 SOTA 성능을 달성하고 있습니다. 이 모델은 세 가지 인기 있는 데이터셋을 통해 훈련되었으며, 운전 장면에서 다음에 발생할 운전자의 주의를 예측하는 데 성공했습니다.



New uploads on arXiv(cs.AI)

### Contextualizing biological perturbation experiments through languag (https://arxiv.org/abs/2502.21290)
Comments:
          The Thirteenth International Conference on Learning Representations (2025)

- **What's New**: 이 연구에서 제안된 PerturbQA는 생물학적 동원 실험의 결과를 정량적으로 쿼리하는 기준 벤치마크입니다. 기존에 사용되던 벤치마크와는 달리, PerturbQA는 차별적 표현 같은 기존 실험에서의 미해결 문제를 해결하는 데 중점을 두고 있습니다. 이 연구에서는 Summer라는 LLM 프레임워크도 소개되었으며, 이는 현재의 최첨단 기술을 능가하거나 맞먹는 성능을 보입니다.

- **Technical Details**: PerturbQA는 생물학적 동원 실험의 결과를 정량적으로 묻는 질문-응답 방식으로 구성됩니다. 이 시스템은 언어 모델을 이용하여 생물학적 지식 그래프 및 실험 데이터를 통합하여 생물학적 질문에 대한 응답을 도출합니다. Summer 프레임워크는 LLM을 사용하여 유전자에 대한 설명을 요약하고, 이 정보를 기반으로 실험 결과를 조회하며, 마지막으로 비슷한 결과를 보이는 유전자 클러스터를 특성화합니다.

- **Performance Highlights**: 연구 결과 현재의 최첨단 기계 학습 및 통계적 접근 방식이 PerturbQA에서 저조한 성능을 보이는 것으로 나타났습니다. 그러나 Summer는 이러한 기준에서 훌륭한 성능을 발휘하며, 모든 조정 없이 생물학자가 이해할 수 있는 언어로 작동하는 경량 모델입니다. 이 연구는 기계 학습 모델링의 접근성과 관심을 높일 수 있는 기반이 될 것입니다.



### Modeling Human Beliefs about AI Behavior for Scalable Oversigh (https://arxiv.org/abs/2502.21262)
Comments:
          53 pages

- **What's New**: 이번 연구에서는 AI 시스템의 인간 피드백 신뢰성이 떨어지는 문제를 다룹니다. AI 시스템이 발전함에 따라 인간의 능력을 초과하게 되었고, 이에 따라 AI 시스템을 감독하는 문제에 대한 새로운 접근 방식을 제안합니다. 연구진은 인간 평가자의 믿음을 모델링하여 AI 시스템의 행동에 대한 피드백을 더 효과적으로 해석할 수 있는 방법을 모색합니다.

- **Technical Details**: 연구에서는 인간 믿음 모델(human belief models)을 공식화하고 이 모델들이 인간의 가치를 추론하는 데 어떤 역할을 하는지 이론적으로 분석합니다. 또한 이러한 추론에서 남아 있는 모호성을 특성화하고, 모호성이 사라지는 조건을 규명합니다. 정확한 믿음 모델에 대한 의존성을 줄이기 위해, 인간 믿음 모델의 커버링(relaxation of human belief model covering) 개념을 도입합니다.

- **Performance Highlights**: 기본 모델(foundational models)을 활용하여 커버링 신념 모델을 구성하는 방안을 제안하였습니다. 이는 AI 시스템의 확장 가능한 감독(sc可alability oversight) 문제에 대한 새로운 접근법을 제공할 것으로 기대됩니다. 이러한 연구 결과는 AI alignment 분야에 중요한 기여를 할 것으로 보입니다.



### Towards Developing Ethical Reasoners: Integrating Probabilistic Reasoning and Decision-Making for Complex AI Systems (https://arxiv.org/abs/2502.21250)
- **What's New**: 이 논문에서는 AI와 자율 시스템을 위한 포괄적이고 유연한 컴퓨터 윤리 프레임워크를 제시합니다. 기존의 접근법들은 빠르게 변하는 실세계의 윤리적 맥락에서 적응하기 부족한 경우가 많아, 다양한 상황에 대한 효과성이 제한적이었습니다. 따라서 논문에서는 중간 표현(intermediate representations), 확률적 추론(probabilistic reasoning), 그리고 지식 표현(knowledge representation)을 결합한 메타 수준의 프레임워크를 구축하는 데 필요한 요소들을 설명합니다.

- **Technical Details**: 컴퓨터 윤리에 기반한 윤리적 판단을 위한 프레임워크는 개인의 의사결정 뿐만 아니라 다중 에이전트 시스템의 집합적 동역학을 지원할 수 있도록 확장 가능해야 합니다. 이를 위해, 논문에서는 정교한 윤리 이론과 적응형 학습 시스템을 통합하는 통합 아키텍처의 필요성을 강조합니다. 또한, 본 논문에서는 중간 표현을 사용하는 것이 윤리적 의사결정 과정에서 중요한 요인임을 논의하며, 복잡한 윤리적 딜레마를 관리 가능한 하위 목표로 분해하여 체계적 접근을 제공합니다.

- **Performance Highlights**: 윤리적 이유 시스템 개발에 있어, 기존의 접근방법의 한계들을 명확히 하고자 합니다. 여기에는 상황적 세부정보와 윤리적 목표를 명확히 정의하여 윤리적 복잡성을 체계적으로 탐색할 수 있는 구조화된 프레임워크를 소개합니다. 이러한 시나리오를 통해 AI 시스템이 복잡한 윤리적 결정을 효과적으로 처리할 수 있는 능력을 키우게 될 것입니다.



### Transforming Tuberculosis Care: Optimizing Large Language Models For Enhanced Clinician-Patient Communication (https://arxiv.org/abs/2502.21236)
Comments:
          GenAI4Health at AAAI-25

- **What's New**: 이번 연구는 결핵(Tuberculosis, TB) 치료를 위한 새로운 디지털 적응 기술(Digital Adherence Technologies, DATs) 개발을 다룹니다. 저소득 및 중간소득 국가에서의 의료 접근 문제를 해결하기 위해, 대형 언어 모델(Large Language Model, LLM)을 활용하여 치료 지지자와 환자의 상호작용을 증진시키는 것을 목표로 하고 있습니다. 이 기술은 '인간-현장' 시스템 내에서 AI를 구동하여 환자 참여도를 높이고 TB 치료 결과를 개선하려는 시도로, 전체적인 치료 경험을 향상시킬 수 있는 잠재력을 지니고 있습니다.

- **Technical Details**: 연구에서는 인간 감독 아래에서 TB 치료를 지원하는 LLM 기반의 대화형 모델을 개발하고, 이를 통해 문맥적 학습 기법을 활용하여 환자의 요구에 맞는 지원 도구를 만드는 것을 목표로 합니다. 모델 설계는 환자와 치료 지원자 간의 대화를 시뮬레이션하는 몇 가지 다이얼로그 샘플을 기반으로 하며, 다양한 텍스트 정화(dwifferentally private text sanitization) 방법을 통해 환자의 개인 정보를 보호합니다. 연구진은 또한 TB 관련 자료와 신뢰할 수 있는 출처의 자원을 활용하여 사실적이고 정확한 응답을 보장하기 위한 RAG(검색 보강 생성) 파이프라인을 구현했습니다.

- **Performance Highlights**: 개발된 LLM 모델은 스페인어를 사용하는 결핵 환자를 지원하기 위한 것이며, 문화적 및 언어적 적합성을 고려한 대화 응답을 모델링하는 데 성공했습니다. 환자와 치료 지원자 간의 대화를 재현하여 공감적 응답을 유도하도록 교육받았으며, 이를 통해 치료 지침에 따라 개인 맞춤형 지원을 제공할 수 있습니다. 이 연구는 특히 비영어권 환자에게 유용할 것으로 기대되며, 다국적 의료 환경에서의 LLM 적용에 대한 통찰을 제공합니다.



### An Algebraic Framework for Hierarchical Probabilistic Abstraction (https://arxiv.org/abs/2502.21216)
- **What's New**: 이 논문은 복잡한 시스템의 추상화를 위해 새로운 계층적 확률적 추상화 프레임워크를 소개합니다. 기존의 단일 계층 기반 접근법의 한계를 극복하고 고급 개념화와 저급 인지 데이터를 연결하여 해석력을 극대화하고자 합니다. 이 프레임워크는 여러 계층의 맵핑을 통해 분산 문제 해결을 지원하며, 체계적인 분석을 가능하게 합니다. 이러한 작업은 인공지능(AI)의 다양한 하위 분야에서 광범위한 적용을 지원합니다.

- **Technical Details**: 제안된 프레임워크는 계층적 추상화를 위한 수학적 기초를 확장하여 다층 구조를 가능하게 합니다. 각 계층에서는 개별 변환 및 확률적 관계를 분석할 수 있도록 하며, 이를 통해 전체 시스템에 대한 종합적인 이해를 돕습니다. 고급 이론을 저급 데이터로부터 도출하기 위한 구조화된 프레임워크가 마련되어 있으며, 이는 복잡한 공간에서의 추론을 강화합니다. 계층 구조를 통한 명확한 관계 설정은 다양한 맥락에서 추상화 계층을 학습할 수 있도록 지원합니다.

- **Performance Highlights**: 이 프레임워크는 시스템 1(직관적 사고)과 시스템 2(논리적 사고) 간의 일치를 지원하여 인식된 복잡성을 해소합니다. 맞춤형 교육 시스템과 같은 실제 사례에서 응용될 수 있는 잠재력이 있으며, 여러 계층에서의 해석 가능한 통찰을 제공하여 복잡한 상호작용을 분석할 수 있도록 합니다. 또한, 다양한 추상화 방법론 개발에 탄탄한 기초를 마련하여 인지 AI, 통계적 관계 학습, 신경기호 AI의 연구 발전에 기여할 수 있습니다.



### ARIES: Autonomous Reasoning with LLMs on Interactive Thought Graph Environments (https://arxiv.org/abs/2502.21208)
- **What's New**: 본 논문에서는 큰 언어 모델(LLM)이 생각 그래프(thought graph) 변환을 마르코프 결정 과정(Markov Decision Process, MDP)의 행동으로 보고, 효율적인 액션 정책을 제안합니다. 특히, ARIES라는 다중 에이전트 아키텍처를 통해 LLM이 서브 문제를 해결하고 생각 그래프 상태를 모니터링하며, 문제 해결 전략을 동적으로 조정합니다. 이를 통해 LLM이 상황에 따라 자동으로 탐색 전략을 조정할 수 있는 가능성을 제시합니다.

- **Technical Details**: ARIES 프레임워크에서는 정책 LLM 에이전트가 상태를 관찰하고, 특정 액션을 선택하여 문제를 해결하는 과정이 포함됩니다. LLM은 생각 그래프 환경에서 피드백을 수집하여 탐색 전략을 동적으로 조정하며, 기존의 정적 transform schedule을 초월할 수 있는 가능성을 보여 줍니다. 이 연구는 생각 그래프를 독립적으로 해결할 수 있는 서브 문제로 나누어, 각 에이전트가 협력하여 문제를 해결하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 비지도 미세 조정(supervised fine-tuning, SFT)을 하지 않은 LLM을 정책 에이전트로 사용할 경우, 정적 변환 일정에 비해 HumanEval에서 최대 29% 더 높은 정확도를 얻었으며, 추론 비용을 35% 줄이고, 탐색 비용이 필요 없음을 확인했습니다. 기존 연구와의 비교를 통해, 우리의 접근 방식이 LLM의 계획 및 추론 능력을 극대화할 수 있음을 입증하고 있습니다.



### A Survey of Link Prediction in Temporal Networks (https://arxiv.org/abs/2502.21185)
- **What's New**: 이번 논문은 Temporal Link Prediction (TLP)의 최신 기술을 다루며, 대표성과 추론 방법을 명확하게 구분하는 새로운 분류 체계를 제시합니다. 기존의 조사는 여러 측면을 다루었지만, 포괄적인 틀이 부족했던 점을 해결하고자 합니다. 이 연구는 TLP 접근 방식의 분류를 통해 다양한 기존 방법의 조합을 탐색할 수 있는 기초를 제공합니다.

- **Technical Details**: Temporal networks는 시간에 따라 변화하는 복잡한 시스템의 상호작용을 모델링하는 데 중요한 역할을 합니다. 이 논문에서는 temporal network를 모델링하기 위해 두 가지 요소, 즉 representation unit과 inference unit이 필요하다는 점을 강조합니다. 이를 통해 예측 문제를 해결하고, 복잡한 시간적 및 공간적 네트워크 동태를 반영하는 모델을 생성할 수 있습니다.

- **Performance Highlights**: TLP와 관련된 다양한 연구 기회를 제시하며, 기존 방법들 간의 새로운 조합과 혁신적인 단위 설계의 가능성을 논의합니다. 이 논문은 네트워크 표현과 TLP 방법론의 현재 도전 과제를 정리하고, 향후 연구 방향을 제시하여 이 영역에서의 새로운 발견을 촉진하길 기대하고 있습니다.



### Multimodal Dreaming: A Global Workspace Approach to World Model-Based Reinforcement Learning (https://arxiv.org/abs/2502.21142)
Comments:
          Under review in a conference

- **What's New**: 이번 연구에서는 세계 모델(World Models)과 글로벌 작업(Global Workspace) 이론을 결합한 새로운 강화 학습 시스템(GW-Dreamer)을 평가합니다. 기존의 세계 모델은 환경 변수를 직접 처리하여 학습 속도가 느린 문제를 가지고 있었습니다. 이번 연구는 고차원 잠재 공간(latent space)에서 정신 시뮬레이션(dreaming process)을 수행하면 환경 단계가 적어도 훈련 가능하다는 것을 보여줍니다. 이로 인해 다양한 상황에 적응하는데 효과적인 모델을 제안합니다.

- **Technical Details**: GW-Dreamer는 기존의 정책 최적화 알고리즘인 PPO와 원래의 Dreamer 알고리즘 여러 버전과 비교되었습니다. 연구 결과, GW 잠재 공간 내에서 진행되는 정신 시뮬레이션이 환경 단계 수를 줄이고, 훈련을 효과적으로 수행하는 데 도움을 줍니다. 이 시스템은 또한 이미지나 시뮬레이션 속성과 같은 관찰 모달리티 하나의 부재에도 강한 견고성을 보여줍니다.

- **Performance Highlights**: GW-Dreamer 모델은 기존의 비교 기준 모델들에 비해 우수한 성능을 보여줍니다. 특히, GW와 세계 모델의 결합이 강화 학습 에이전트의 의사결정 능력을 크게 향상시키는 잠재력을 보입니다. 이러한 결과는 강화 학습 분야에 새로운 접근 방식을 제시하며, 다양한 상황에 대한 적응력을 증가시킬 수 있습니다.



### Optimizing Large Language Models for ESG Activity Detection in Financial Texts (https://arxiv.org/abs/2502.21112)
- **What's New**: 이 논문은 ESG(환경, 사회, 지배구조) 요인이 기업 의사결정에 통합되는 과정에서 AI 모델을 활용하여 지속 가능성 보고서와 비재무 공시를 자동으로 평가하는 새로운 접근 방식을 제안합니다. 특히, ESG-Activities라는 새로운 벤치마크 데이터셋을 도입하여 이를 통해 LLMs(대형 언어 모델)의 성능을 강화하는 방법에 대해 설명합니다.

- **Technical Details**: 현재 세대 LLMs의 텍스트 분류 능력을 평가하면서, ESG-Activities 데이터셋을 이용해 모델을 세부 환경 활동에 맞춰 학습시키는 방법을 채택합니다. 이 데이터셋은 전문가가 수작업으로 선별한 데이터와 LLMs에 의해 생성된 합성 데이터를 포함하고 있어 모델 성능을 높이는 데 기여합니다.

- **Performance Highlights**: 실험 결과, ESG-Activities로 파인튜닝된 모델이 기존의 제로샷 학습과 수작업으로 주석이 붙은 데이터만으로 학습한 모델에 비해 분류 정확도가 크게 향상된 것을 확인하였습니다. Llama 7B와 같은 오픈 소스 모델이 특정 설정에서 대형 프로프라이어터리 모델보다 우수한 성능을 발휘했으며, 이는 AI 연구자와 금융 분석가들에게 중요한 영향을 미칠 것입니다.



### Re-evaluating Theory of Mind evaluation in large language models (https://arxiv.org/abs/2502.21098)
Comments:
          under review

- **What's New**: 이 논문은 대규모 언어 모델(LLMs)의 마음 이론(Theory of Mind, ToM) 능력에 대한 새로운 평가 방법을 제안합니다. ToM은 다른 사람의 정신 상태를 추론하는 능력이고, 이러한 능력이 LLMs에 얼마나 적용되는지에 대한 논란이 있으며 기존의 평가 방법에 의문을 제기합니다. 저자들은 LLMs가 인간의 행동을 일치시키는 것뿐만 아니라, 이러한 행동의 근본적인 계산을 이해하도록 기대해야 하는지 명확하지 않다고 주장합니다.

- **Technical Details**: 저자들은 ToM 평가에서 두 가지 주요 이슈를 강조합니다. 첫 번째는 ToM의 정의에 대한 불일치입니다. ToM을 가진다는 것의 의미는 사람들의 행동에 맞추는 것(behavior-matching)일 수도 있고, 이러한 행동을 가능하게 하는 정신적 알고리즘(computation matching)일 수도 있습니다. 두 번째는 평가의 타당성으로서, 현재의 ToM 평가가 심리적 구성요소를 측정하지 못할 수 있으며, 적대적인 예시(adversarial examples)에 대한 평가가 순수한 ToM을 벗어난 일반적 사고 능력으로 평가를 변화시킬 수 있음을 지적합니다.

- **Performance Highlights**: 현재까지 LLMs의 ToM 능력에 대한 연구는 상반된 결과를 보였습니다. 일부 연구자들은 LLM이 인간 수준의 ToM 평가 과제를 성공적으로 수행한다고 주장하는 반면, 다른 연구자들은 LLM이 미세한 변수나 적대적 변화에 민감하다고 주장합니다. 새로운 모델과 평가 기준이 빠르게 발전하고 있지만, LLM의 ToM 능력을 평가하는 데 필요한 명확한 기준이 부족하여 혼란스러움이 지속되고 있습니다.



### An LLM-based Delphi Study to Predict GenAI Evolution (https://arxiv.org/abs/2502.21092)
- **What's New**: 이번 연구에서는 데이터가 부족하거나 신뢰할 수 없는 영역에서의 미래 예측 문제를 해결하기 위해 Large Language Models(LLM)을 활용한 혁신적인 방법론을 제시합니다. 이 방법론은 Generative Artificial Intelligence(GenAI)의 미래를 탐사하는 데 적용되었으며, 지정학적 긴장, 경제적 불균형, 규제 프레임워크 등 주요 요인에 대한 통찰을 제공합니다. LLM 기반의 Delphi 연구는 응답자 피로도를 줄이면서 다양한 관점을 수집할 수 있는 구조화된 시나리오 분석을 가능하게 합니다.

- **Technical Details**: 이 연구는 전통적인 Delphi 방법을 기반으로 하여 두 가지 유형의 에이전트를 포함하는 단순한 프로세스로 설계되었습니다. 첫 번째 에이전트는 조직 에이전트로, 응답을 수집하고, 질문지를 생성하며, 응답을 바탕으로 새로운 질문을 만듭니다. 두 번째 에이전트는 응답 에이전트로, 전문가의 역할을 시뮬레이션하며 개방형 질문과 폐쇄형 질문 모두에 답변합니다.

- **Performance Highlights**: LLM 기반의 Delphi 방법은 GenAI의 진화에 대한 구조적 예측을 효과적으로 생성할 수 있음을 보여주며, 주요 결과는 GenAI 개발에 대한 지정학적 긴장과 경제적 불균형의 영향을 강조합니다. 연구 결과는 GenAI의 사회적 영향이 학제간 협력, 문화적 적응 및 경제적 접근성에 의존한다는 점을 시사하며, 규제 메커니즘과 동적인 윤리적 프레임워크의 필요성을 강조합니다.



### Are foundation models useful feature extractors for electroencephalography analysis? (https://arxiv.org/abs/2502.21086)
- **What's New**: 의료 데이터가 제한된 환경에서도 효과적인 성능을 보이는 foundation 모델을 EEG 분석에 적용하는 연구가 진행되었습니다. 이 연구는 OTiS 모델을 통해 여러 EEG 관련 작업의 진단 정확도를 측정하였으며, 그 결과 OTiS가 전문화된 EEG 모델보다 뛰어난 성능을 보였음을 밝혔습니다. 이러한 모델링 접근법은 대규모 도메인 특화 데이터셋의 필요성을 최소화하여, 임상 분야에서 중요한 도구로 자리잡을 수 있을 것으로 보입니다.

- **Technical Details**: OTiS 모델은 12개의 레이어, 3개의 헤드, 192의 폭, 그리고 8M의 파라미터를 가진 기본 변형에 대해 분석되었습니다. 이 모델은 다양한 도메인에서 수집된 640,187개의 타임 시리즈 샘플로 사전 훈련되었습니다. 본 연구는 OTiS가 EEG 분석을 위해 얼마나 효과적인지를 평가하기 위해 세 가지 공공 데이터셋에 대해 전문화된 EEG 모델과의 비김을 통해 검토하였습니다.

- **Performance Highlights**: 실험 결과 OTiS가 EEG 기준에서 최첨단 성능을 달성하며, 특정 작업에 맞춤화된 모델보다 더 높은 품질의 EEG 특징을 추출함을 입증했습니다. OTiS는 또한 주파수 대역 across frequency bands에 따른 EEG 특징을 포착할 수 있어, 중요 인구 통계학적 정보나 질병 관련 바이오마커를 지역화하는 데 도움이 됩니다. 아키텍쳐 설계의 주요 선택이 진단 정확도에 큰 영향을 미치는 것으로 나타났습니다.



### Merging Clinical Knowledge into Large Language Models for Medical Research and Applications: A Survey (https://arxiv.org/abs/2502.20988)
- **What's New**: 이 논문은 의학적 인공지능(medical AI) 시스템을 구축하는 데 있어 아카데미와 산업의 차이를 포괄적으로 비교하는 자료가 부족하다는 문제를 지적하고 있습니다. 최근 대형 언어 모델(large language models, LLMs)의 발전에 따라 신속하고 정확한 진단을 지원하기 위한 여러 연구가 진행되고 있지만, 관련 지식과 데이터의 통합이 여전히 도전과제로 남아 있습니다. 이 연구는 의료 데이터, 학습 프로세스, 의료 지식 그래프(knowledge graphs) 통합 및 평가 시스템 등 다양한 측면에서 의료 AI 시스템 구축 패러다임을 살펴봅니다.

- **Technical Details**: 임상 데이터베이스 및 데이터셋의 소개를 통해 의료 LLM 구축을 위한 기초를 다지고, 의사결정 지원을 위해 직접 학습(direct learning)과 지식 추출(knowledge extraction) 두 가지 연구 방향을 제시하고 있습니다. LLM들은 방대한 의료 데이터에서 패턴을 발견하고, 이를 통해 진단 및 치료를 위한 개인 맞춤형 치료 계획을 제공하는 데 중점을 둡니다. 특히 의료 데이터의 품질과 최신성을 유지하는 것이 중요한데, 이는 모델의 예측 정확도를 높이는 데 기여합니다.

- **Performance Highlights**: 대형 언어 모델들은 전자 의료 기록(electronic medical records, EMRs), 의료 영상 및 유전 정보와 같은 대량의 데이터를 활용하여 진단의 정확도를 높이는 데 중요한 역할을 합니다. 커리큘럼 중심의 데이터 세트가 지속적으로 업데이트되고 있으며, 이를 통해 의료 AI 시스템은 임상 요구에 더 잘 적응하고 있습니다. 논문은 결국 아카데미와 임상 실무 간의 조화로운 발전을 통해 의료 AI의 향후 연구 방향을 제시하고, 실제 의료 서비스 개선을 위한 실질적인 가이드를 제공합니다.



### A Pilot Empirical Study on When and How to Use Knowledge Graphs as Retrieval Augmented Generation (https://arxiv.org/abs/2502.20854)
Comments:
          8 pages, 2 figures, 14 tables

- **What's New**: 이 논문은 Knowledge Graphs (KGs)와 Retrieval Augmented Generation (RAG) 프레임워크의 통합에 대한 체계적인 이해 부족을 파악하고 이를 해결하기 위한 방법론을 제공하고자 합니다. 구체적으로, KG-RAG의 성능을 다양한 기술 구성으로 분석하여 KG-RAG를 사용해야 하는 조건과 방법을 정립하고 있습니다. 이 연구는 7개의 데이터 세트에 걸쳐 6개의 KG-RAG 방법을 재구현하고 평가하여, KG-RAG 구성 요소의 최적화와 적절한 적용 조건의 중요성을 강조합니다.

- **Technical Details**: KG-RAG는 LLM (Large Language Model)의 정보 검색 능력을 개선하기 위한 새로운 패러다임으로, 엔티티 간의 의미적 관계를 활용하여 더욱 정교한 추론 능력을 제공합니다. 본 연구에서는 KG-RAG의 적용 가능성을 살펴보기 위해 과제를 분야와 난이도에 따라 분류하고, KG 질의 증진 전략, 다양한 검색 형태, 후속 프롬프트 방법 등 9개의 KG-RAG 구성을 분석했습니다. 이 연구는 KG-RAG의 효과적인 진행을 위한 실용적인 가이드라인을 제시하고 있습니다.

- **Performance Highlights**: KG-RAG의 성능 분석 결과는 개방형 LLM이 특정 작업 도메인과 난이도에서 큰 이점을 제공함을 나타냅니다. 더불어, 다양한 KG-RAG 구성들이 성능에 미치는 영향을 분석하고, KG 품질과 LLM의 능력이 KG-RAG의 효과성에 미치는 중요성을 강조하였습니다. 최종적으로, 현재 KG-RAG 연구의 몇 가지 제한점도 제시하며 향후 연구 방향을 제시하고 있습니다.



### MV-MATH: Evaluating Multimodal Math Reasoning in Multi-Visual Contexts (https://arxiv.org/abs/2502.20808)
Comments:
          47 pages

- **What's New**: MV-MATH라는 새로운 benchmark을 소개합니다. 이 데이터셋은 2,009개의 고품질 수학 문제로 구성되어 있으며, 각 문제는 실제 K-12 상황에서 유래한 여러 이미지를 포함하여 텍스트와 혼합되어 있습니다. 다양한 난이도와 주제를 포괄하는 MV-MATH는 현재의 MLLM(Math Large Language Models) 성능을 평가하기 위한 포괄적이고 엄격한 기준을 제공합니다.

- **Technical Details**: MV-MATH는 다중 이미지와 텍스트가 혼합된 형식으로 제공되고, 11개의 주제와 3개의 난이도로 분류된 문제들을 포함하고 있습니다. 이 데이터셋은 2명 이상의 주석자가 교차 검증하여 높은 품질과 정확성을 보장하며, 문제 유형은 다중 선택, 자유 형식 및 다단계 문제를 포함합니다.

- **Performance Highlights**: 다양한 모델들의 성능을 평가한 결과, Claude 모델이 33.9%의 점수를 받아 가장 높은 성과를 기록했습니다. LLaVA-OneVision은 경쟁력 있는 26.2%의 점수로 GPT-4v를 초월했지만, 여전히 모든 모델들이 인간 수준의 성능에는 미치지 못하는 것으로 나타났습니다. 이러한 결과 분석을 통해 MLLMs의 강점과 한계를 파악할 수 있습니다.



### MedHallTune: An Instruction-Tuning Benchmark for Mitigating Medical Hallucination in Vision-Language Models (https://arxiv.org/abs/2502.20780)
- **What's New**: 이번 연구에서는 의료 분야의 Vision-Language Models (VLMs)에서 발생하는 황홀경(hallucination) 문제를 다루기 위해 MedHallTune이라는 대규모 벤치마크를 제안합니다. MedHallTune은 100,000개 이상의 이미지와 1,000,000쌍의 지침으로 구성되어 있으며, 각 데이터는 진실(annotation)로 주석이 달려 있습니다. 이 데이터셋은 의료 VLMs의 성능 평가 및 개선을 목적으로 설계되었습니다.

- **Technical Details**: MedHallTune은 100,000개 이상의 이미지-텍스트 쌍을 샘플링하고, GPT-4o를 활용해 황홀경 및 비황홀경 예제를 포함한 지침 데이터를 생성합니다. 각 데이터 쌍은 두 번의 검증 과정을 통해 생성되어, 잘못된 해석을 필터링합니다. 연구에서는 기존 VLM을 미세 조정(fine-tuning)하여 의료 맥락에서 황홀경의 대응 능력을 향상시키는 방법을 제시합니다.

- **Performance Highlights**: 실험 결과, MedHallTune을 활용한 미세 조정이 여러 의료 및 일반 VLM 모델의 황홀경 관리 능력을 크게 향상시키는 것으로 나타났습니다. 고급 평가 메트릭을 통해, 임상 정확성(clinical accuracy), 관련성(clinical relevance), 정보의 상세 수준(detail level), 위험 수준(risk level) 등에서 성능이 크게 개선되었습니다. 이를 통해, 더욱 신뢰할 수 있는 의료 VLM 개발에 기여하고 있습니다.



### Damper-B-PINN: Damper Characteristics-Based Bayesian Physics-Informed Neural Network for Vehicle State Estimation (https://arxiv.org/abs/2502.20772)
- **What's New**: 본 연구에서는 차량 섀시 시스템과 같은 노이즈가 있는 다중 입력 다중 출력 (MIMO) 시스템을 위한 상태 추정 기술이 도입됩니다. 기존 방법의 한계를 극복하기 위해, 댐퍼 특성을 기반으로 한 베이지안 물리 정보 신경망(Damper-B-PINN) 설계를 제안합니다. 이를 통해 신경망의 수치적 안정성을 강화하고 입력-출력 관계의 복잡성을 해결합니다.

- **Technical Details**: 본 연구에서는 댐퍼의 기계적 성질을 영감으로 하는 뉴런 전진 과정을 도입하여, 에포크 사이에서 뉴런 값의 급격한 변화가 발생하지 않도록 합니다. 또한, MIMO 시스템에 최적화된 베이지안 드롭아웃 레이어를 적용하여 노이즈에 대한 강인성을 높이고 비수렴 문제를 방지합니다. 물리적 정보는 손실 함수에 포함되어 신경망의 물리적 사전(prior) 역할을 합니다.

- **Performance Highlights**: Damper-B-PINN 구조는 10개의 데이터 세트와 14종의 차량 유형에서 테스트되어 다른 최신 기술과 비교할 때 정확성, 계산 효율성 및 수렴성에서 우수한 성능을 입증했습니다. 특히, 차량 상태 추정(예: 동적 휠 하중)에서 그 효과가 두드러집니다.



### Acquiring Grounded Representations of Words with Situated Interactive Instruction (https://arxiv.org/abs/2502.20754)
- **What's New**: 이번 연구에서는 혼합 주도(mixed-initiative)와 인간 강사와의 상황적 상호작용을 통해 단어의 기초가 되는 표현을 습득하는 접근 방식을 제시합니다. 이 연구는 퍼셉트럴(perceptual), 의미론적(semantic), 절차적(procedural) 지식을 포함한 다양한 지식을 습득하는 데 중점을 두며, 학습된 의미의 기초를 배우는 것과 함께 에이전트가 미지의 개념에 대한 지침을 요구하여 학습을 효율적으로 할 수 있게 합니다. 이 접근 방식은 Soar에서 구현되었으며, 소형 물체를 조작할 수 있는 테이블탑 로봇 팔을 통해 평가되었습니다.

- **Technical Details**: 우리의 에이전트는 로봇 팔, Kinect 센서, 사전에 정의된 네 개의 위치(스토브, 식기세척기, 쓰레기장, 팬트리)를 갖춘 간단한 테이블탑 환경에서 존재합니다. 인식 시스템은 오버헤드 Kinect 카메라에서 제공하는 컬러 3D 포인트 클라우드 데이터를 사용하여 장면을 객체로 분할합니다. 각 인식 속성에 대해 특징을 추출하고 K-Nearest Neighbor (KNN) 분류기를 통해 분류하여 에이전트가 이전에 학습한 인식 기호(perceptual symbol)를 생성합니다. 이 기호와 함께 객체의 위치 및 경계 상자 정보가 제공되어, 에이전트는 이를 통해 상징적 표현(symbolic representation)을 형성합니다.

- **Performance Highlights**: 우리는 에이전트가 인간과의 상호작용을 통해 단어의 기초가 되는 표현을 효과적으로 학습하고 있음을 보여주었습니다. 에이전트는 주어진 과제 수행 중에 인간 강사와 실시간으로 상호작용하여 학습을 하고 있으며, 반응 속도는 2초 이내입니다. 학습 과정에서 에이전트는 예제의 수를 줄이며, 학습해야 할 개념들이나 미비한 지식을 스스로 요청하는 능력을 발휘합니다. 결과적으로, 혼합 주도 상호작용(mixed initiative interaction)이 효율적이고 유연한 지식 습득 방법임을 입증하였습니다.



### DeepSolution: Boosting Complex Engineering Solution Design via Tree-based Exploration and Bi-point Thinking (https://arxiv.org/abs/2502.20730)
- **What's New**: 이번 논문에서는 복잡한 공학 솔루션 설계 과제를 다루기 위한 새로운 벤치마크인 SolutionBench를 소개합니다. 이전 연구들이 복잡한 공학 문제의 해결책 설계에 충분히 집중하지 않았다는 점을 보완하고자 합니다. 또한, SolutionRAG라는 새로운 시스템을 제안하여 나무 기반 탐색(tree-based exploration) 및 이점 사고(bi-point thinking) 메커니즘을 활용하여 신뢰할 수 있는 솔루션을 생성합니다. 이 시스템은 공학 설계의 자동화 및 신뢰성을 향상시키는 데 잠재력을 가지고 있습니다.

- **Technical Details**: SolutionBench는 복잡한 공학 요구사항을 위한 솔루션 생성 능력을 평가하는 도구로, 공인된 저널에서 수집된 전문 기술 보고서를 바탕으로 구성됩니다. 데이터는 LLMs를 활용하여 자동으로 추출되며, 요구사항, 솔루션, 분석 지식 및 기술 지식을 포함하는 형식화된 템플릿이 사용됩니다. 이 과정에서 수집된 데이터는 철저한 수동 검토를 거쳐 최종 벤치마크로 통합됩니다.

- **Performance Highlights**: 실험 결과, 기존의 RAG 방식은 복잡한 공학 솔루션을 효과적으로 생성하지 못하는 것으로 나타났습니다. 반면 SolutionRAG는 최신 성능(state-of-the-art)을 달성하며, 기존 방법들과 비교할 때 훨씬 더 나은 결과를 제공합니다. 이러한 결과는 복잡한 공학 설계 작업에서 SolutionRAG의 유용성을 강조합니다.



### Fuzzy Speculative Decoding for a Tunable Accuracy-Runtime Tradeoff (https://arxiv.org/abs/2502.20704)
- **What's New**: 이번 연구에서는 Fuzzy Speculative Decoding (FSD)라는 새로운 디코딩 알고리즘을 소개했습니다. FSD는 기존의 Speculative Decoding (SD) 방식의 한계를 극복하기 위해 설계되었으며, 후보 토큰의 수용을 목표 모델과 초안 모델의 분포 간 위반을 바탕으로 결정합니다. 이를 통해 사용자들은 생성 품질과 추론 속도 간의 트레이드오프를 유연하게 조정할 수 있습니다.

- **Technical Details**: FSD는 사용자에게 후보 수용의 관대함을 조정할 수 있는 매개변수 T𝑇Titalic_T를 제공하여, 목표 모델의 분포에서 얼마나 벗어날지를 선택할 수 있게 합니다. 이 방식은 후보 토큰의 수용률을 높여주며, SD보다 훨씬 뛰어난 속도 향상을 제공합니다. 여러 가지 벤치마크를 통해 FSD가 SD의 정확도를 유지하면서도 2토큰 이상의 속동향을 이루어낼 수 있음을 입증했습니다.

- **Performance Highlights**: FSD는 추론 속도에서 최대 5토큰의 향상을 달성할 수 있으며, 이는 정확도가 약 2% 감소하는 경우에도 가능합니다. 또한, 다양한 실험을 통해 SD의 정확성과 FSD의 속도를 비교했을 때, FSD가 더 높은 성능을 나타냈음을 보여주었습니다. 마지막으로, FSD는 여러 후보 모델에 대해 훈련이 필요 없고, 모든 모델 쌍에 즉시 적용할 수 있어 효율성이 더욱 높습니다.



### Why Trust in AI May Be Inevitab (https://arxiv.org/abs/2502.20701)
- **What's New**: 이번 연구에서는 인간-AI 상호작용에서 설명(explanation)의 중요성이 강조되지만, 신뢰(trust) 없이는 설명이 항상 가능하지 않다는 점을 주장합니다. 이는 설명이 지식 네트워크(knowledge networks) 내에서의 검색 과정으로 형식화됨을 통해 도출된 결과입니다. AI 시스템의 신뢰성을 구축하는 데 있어 설명이 필수적이지만, 때때로 그 자체가 불가능할 수 있다는 점을 새롭게 제시합니다.

- **Technical Details**: 연구에서 설명은 공유 개념(shared concepts)과 설명해야 할 개념(concept to be explained) 사이의 경로를 찾는 검색 과정으로 정의됩니다. 설명이 성공하기 위해서는 공유된 지식의 존재 뿐만 아니라 시간 제약 내에서 연결 경로를 발견해야 합니다. 따라서 설명을 시도하는 것이 비합리적일 수도 있으며, 이는 인간과 AI 간의 의사소통에 중요한 영향을 미칠 수 있습니다.

- **Performance Highlights**: 연구 결과는 특히 대형 언어 모델(Large Language Models)이 더욱 정교해짐에 따라 인간이 진정한 설명을 요구하기보다 신뢰에 의존할 가능성이 있음을 강조합니다. 이는 잘못된 신뢰(misplaced trust)와 불완전한 지식 통합(imperfect knowledge integration)의 위험을 증가시킬 수 있습니다. 이러한 경향은 AI 상호작용에서의 신뢰 구축 방식에 중요한 의미를 지닙니다.



### ProAI: Proactive Multi-Agent Conversational AI with Structured Knowledge Base for Psychiatric Diagnosis (https://arxiv.org/abs/2502.20689)
Comments:
          21 pages, 8 figures

- **What's New**: 이 논문은 대화형 AI 시스템이 사용자 프롬프트에 반응하는데 그쳤던 기존의 방식에서 벗어나, 좀 더 주도적(proactive)이고 목표 지향적(goal-oriented)인 대화형 AI 프레임워크인 ProAI를 소개합니다. ProAI는 정신 건강의 차별 진단을 위한 응용 프로그램의 맥락을 통해 AI가 적절한 질문을 던지고 대화를 특정 목표로 안내할 수 있도록 설계되었습니다.

- **Technical Details**: ProAI는 구조화된 지식 기반 메모리(knowledge-guided memory), 다중 에이전트의 주도적(reasoning) 사고, 다면적인 평가 전략을 통합하여 LLMs(대형 언어 모델)이 단순한 응답 생성을 넘어 전문가 스타일의 진단(reasoning) 진행을 가능하게 합니다. 이러한 기술적 요소들은 AI가 대화 중에 전문적이고 공감하는(interaction standards) 방식으로 상호작용할 수 있도록 지원합니다.

- **Performance Highlights**: 모의 환자 상호작용(simulated patient interactions), 사용자 경험 평가(user experience assessment), 전문가 임상 검증(professional clinical validation)을 통해 ProAI는 정신 질환의 차별 진단에서 최대 83.3%의 정확도를 달성했습니다. 이러한 결과는 LLM들이 반응형 대화 시스템을 넘어 더욱 신뢰할 수 있는, 적응력 있는(goal-driven) AI 진단 비서로 발전할 수 있는 잠재력을 보여줍니다.



### Automatic database description generation for Text-to-SQL (https://arxiv.org/abs/2502.20657)
- **What's New**: 이번 논문은 Natural Language(자연어) 질의를 Structured Query Language(SQL)로 변환하는 Text-to-SQL 기술의 일환으로, 데이터베이스의 테이블과 열 설명이 중요한 역할을 한다고 강조합니다. 자동으로 효과적인 데이터베이스 설명을 생성하는 방법을 제공하며, 이는 명시적인 설명이 부족할 때 유용합니다. 두 가지 접근 방식인 coarse-to-fine(거친-세밀)과 fine-to-coarse(세밀-거친) 프로세스를 통해 이 과정을 수행하여 데이터베이스 구조에 대한 포괄적인 이해를 가능하게 합니다.

- **Technical Details**: 제안된 방법론은 전체 데이터베이스의 이해에서 시작하여 특정 도메인을 인식하고 기존 지식을 활용하여 데이터베이스에서 나타나는 주요 속성을 분석합니다. 각 테이블과 열에 대한 기능 분석과 의미 예측을 통해 데이터의 저장 유형과 목적을 정리합니다. 개발된 모델은 테이블 및 열을 상세히 분석하고, 정보 흐름이 맥락에 맞게 정렬되도록 합니다.

- **Performance Highlights**: 실험 결과에 따르면 제안한 방법으로 생성된 설명을 사용할 경우 SQL 생성 정확도가 0.93% 향상된 것으로 나타났습니다. 이는 총 37%의 인간 수준 성능에 접근하는 것을 보여줍니다. 논문에서 제안된 방법은 일반적으로 수작업으로 입력되는 주석의 부족으로 인해 발생하는 문제를 해결하는 효과적인 대안이 될 것으로 보입니다.



### PersonaBench: Evaluating AI Models on Understanding Personal Information through Accessing (Synthetic) Private User Data (https://arxiv.org/abs/2502.20616)
- **What's New**: 이번 논문은 개인화된 AI 모델의 발전을 위한 데이터 생성 파이프라인을 소개합니다. 사용자들이 가진 개인 정보에 접근하고 이를 해석하는 AI 모델의 능력을 평가하기 위해, 현실적인 사용자 프로필(User Profiles)과 개인 문서(Private Documents)를 생성하는 합성 데이터 생성 방식을 개발했습니다. 이러한 데이터를 활용하여 PersonaBench라는 벤치마크를 마련하고, AI 모델의 개인 정보 이해 능력을 평가하려고 합니다.

- **Technical Details**: 이 합성 데이터 생성 파이프라인은 다양한 사용자 프로필을 생성하고, 이를 통해 생성한 개인 문서들이 인간의 활동을 시뮬레이션하도록 설계되었습니다. 데이터 생성 과정은 인물 샘플링(Persona Sampling), 사회 그래프 생성(Social Graph Creation), 다중 프로필 완성(Multi-step Profile Completion) 및 다중 타입 개인 문서 생성(Multi-type Private Document Generation) 단계를 포함합니다. GPT-4o 모델을 활용하여 사용자 프로필과 문서를 생성하지만, 다른 대형 언어 모델에도 적용 가능합니다.

- **Performance Highlights**: RAG(상관 검색 증강 생성) 모델을 평가한 결과, 현재의 RAG 시스템은 개인 문서에서 개인 정보를 추출하여 개인 질문에 효과적으로 답변하는 데 어려움을 겪고 있음을 발견했습니다. 이는 AI 개인화 능력을 향상시키기 위해 더 발전된 방법론과 시스템이 필요하다는 점을 강조합니다. 따라서, 우리가 제안한 PersonaBench는 AI 모델이 복잡한 개인 속성을 이해하고 응답하는 데 있어 새로운 평가 기준이 될 수 있습니다.



### NutriGen: Personalized Meal Plan Generator Leveraging Large Language Models to Enhance Dietary and Nutritional Adherenc (https://arxiv.org/abs/2502.20601)
- **What's New**: 본 논문에서는 NutriGen이라 불리는 개인화된 음식 추천 시스템을 소개합니다. NutriGen은 사용자 정의 식습관과 제약 사항에 부합하는 맞춤형 식단을 생성하며, 큰 언어 모델(LLM)을 활용하여 더 나은 유연성과 사용 편의성을 제공합니다. 기존 시스템의 한계점을 해결하고 지속 가능한 식습관을 지원하기 위한 혁신적인 접근법을 제공합니다.

- **Technical Details**: NutriGen은 사용자의 음식 섭취 이력, 식습관, 선호 및 식이 제한을 기반으로 개인화된 식단을 생성하는 프레임워크를 제공합니다. 시스템은 사용자의 입력을 받아 구조화된 영양 데이터베이스를 통합하고, 이를 통해 칼로리 카운트, 대체 음식 옵션 및 제안된 레시피를 포함한 식단을 구성합니다. 식사 계획 생성 과정은 구조화된 프롬프트를 사용하여 사용자 요구 사항에 맞춘 맞춤형 출력이 이루어집니다.

- **Performance Highlights**: NutriGen의 평가 결과, Llama 3.1 8B 모델과 GPT-3.5 Turbo가 각각 1.55% 및 3.68%의 낮은 비율 오차를 기록하며, 사용자 정의 칼로리 목표에 밀접하게 일치하는 식단을 효과적으로 생성함을 보여주었습니다. 또한 DeepSeek V3와 여러 기존 모델들의 성능을 비교하여 개인화된 영양 계획에서의 잠재력을 평가하였습니다.



### On Benchmarking Human-Like Intelligence in Machines (https://arxiv.org/abs/2502.20502)
Comments:
          18 pages, 5 figures

- **What's New**: 이 논문은 AI가 다양한 인지 작업에서 인간 수준의 성과에 도달했다고 주장하는 최근 연구들을 비판하고, 현재의 AI 평가 체계가 인간과 유사한 인지 능력을 평가하는 데 불충분하다고 주장합니다. 논문은 인간 검증을 거치지 않은 레이블, 인간 반응의 변동성과 불확실성을 충분히 반영하지 못하는 점, 생태학적으로 유효하지 않은 단순화된 작업에 의존하는 등의 문제점을 지적합니다. 또한, 이러한 한계를 해결하기 위한 다섯 가지 구체적 권장 사항을 제시합니다.

- **Technical Details**: 논문은 Alan Turing의 정의에 따라 인간과 구별할 수 없는 판단과 행동을 이끌어낼 수 있는 지능형 시스템을 구축하는 데 중점을 둡니다. 10101010 개의 AI 벤치마크 작업을 대상으로 한 인간 평가 연구를 통해, 데이터셋 레이블에서의 인간 검증 부재, 수집된 인간 데이터의 변동성 부족, 그리고 생태학적 유효성이 결여된 단순한 작업들에 대한 과도한 의존을 강조합니다. 모든 벤치마크는 각 자극에 대해 단일 진리 레이블을 가지고 있으며, 데이터 수집을 위해 240240240240명의 참가자를 모집해 레이블을 생성했습니다.

- **Performance Highlights**: 인간과 유사한 AI 성능이 강조되기 위해서는, AI 벤치마크에 대한 평가 방안으로 실제 인간의 행동을 '금' 레이블로 사용하는 것이 중요합니다. 이를 통해 주관적인 개념을 평가하는 데 있어 객관적 정답이 존재하지 않을 수 있는 복잡성을 반영할 수 있습니다. 과거의 인지 모델링 연구에서 도출된 권장 사항들이 AI 성능을 측정하는 데 적용될 수 있으며, 이는 이를 통해 AI 시스템의 성능을 보다 신뢰성 있게 평가하고 기술의 발전에 기여할 것입니다.



### R-ParVI: Particle-based variational inference through lens of rewards (https://arxiv.org/abs/2502.20482)
- **What's New**: 새로운 R-ParVI(Reward-guided Particle Variational Inference) 방법이 제안되었습니다. 이 방법은 부분적으로 알려진 밀도에서 샘플링하는 데 효과적이며, 보상 기구에 의해 파라미터 공간 내에서 입자들이 이동하는 방식으로 샘플링 문제를 재구성합니다. 이로 인해 R-ParVI는 고차원 문제에서도 높은 유연성과 패럴렐성(parallelism)을 제공하여 실제 문제에 적합합니다.

- **Technical Details**: R-ParVI는 입자 시스템의 진화를 보상 함수에 의해 안내하는 방법론입니다. 보상 함수는 목표 밀도와 엔트로피 기반의 다양성 유지 요소를 고려하여 설계되었습니다. 각 입자에 대한 업데이트는 보상 신호에 따라 속도 및 위치의 변화를 통해 수행됩니다.

- **Performance Highlights**: R-ParVI는 고밀도 영역으로 입자를 효과적으로 유도하면서 군집화(clustering)를 방지하는 데 기여합니다. 이 방법은 기존의 미분 기반 접근법보다도 더 나은 유연성을 제공하며, 다양한 거대한 확률 모델에 적합하도록 설계되었습니다. 향후 실험 평가를 통해 R-ParVI의 효율성과 정확성이 확인될 예정입니다.



### Large Language Model Strategic Reasoning Evaluation through Behavioral Game Theory (https://arxiv.org/abs/2502.20432)
- **What's New**: 이번 논문에서는 큰 언어 모델(LLMs)의 전략적 의사결정 메커니즘을 분석하는 평가 프레임워크를 소개합니다. 기존 연구는 네시 균형(Nash Equilibrium, NE) 근사화에 초점을 맞추고 있었는데, 이는 모델의 전략적 선택을 이끌어내는 과정과는 무관한 접근이었습니다. 새로운 연구는 행동 게임 이론(behavioral game theory)을 기반으로 하여 추론 능력과 맥락적 효과를 분리하여 평가합니다.

- **Technical Details**: 22종의 최신 LLM들을 대상으로 한 실험 결과, GPT-o3-mini, GPT-o1 및 DeepSeek-R1 모델이 여러 게임에서 우수한 성능을 보였습니다. 그러나 모델의 규모가 성능을 결정짓는 유일한 요소는 아님을 알게 되었습니다. 체인 오브 생각(Chain-of-Thought, CoT) 프롬프팅의 효과는 모델마다 다르며, 특정 수준의 모델에게만 전략적 추론을 증가시키는 경향이 있음을 발견했습니다. 또한, 인코딩된 인구 통계적 특성의 영향이 모델의 의사결정 패턴에 미치는 영향을 조사했습니다.

- **Performance Highlights**: 대규모 모델들이 대부분의 작업에서 우위를 보였지만, 특정 작업에서는 소규모 모델이 이를 초과하거나 일치하는 성과를 냈습니다. 이러한 결과는 모델 크기와 구조보다는 특정 과제의 활용 가능성에 더 중점을 두어야 함을 시사합니다. 또한, 인구 통계적 특성이 포함된 결과들은 스트레티지적 추론에 편향이 존재함을 암시하는데, 이는 향후 LLM 개발에 균형 잡히고 윤리적인 접근이 필요하다는 점을 강조합니다.



### Beyond transparency: computational reliabilism as an externalist epistemology of algorithms (https://arxiv.org/abs/2502.20402)
- **What's New**: 이번 논문에서는 알고리즘의 인식론에 대해 논의하며, 특히 'computational reliabilism (CR)'이라는 외부적 인식론을 제안합니다. 알고리즘의 투명성에 의존하는 기존 접근법과 달리, CR은 알고리즘의 출력이 신뢰할 수 있는 알고리즘에 의해 생성된다면 정당화된다고 주장합니다. CR은 다양한 과학 분야에서 활용되는 알고리즘을 다루며, 머신러닝 응용에 중점을 두고 있습니다.

- **Technical Details**: CR의 기초를 통해 세 가지 신뢰성 지표(type-RIs)를 소개합니다: (1) type1-RI 기술적 성능, (2) type2-RI 컴퓨터 기반의 과학적 실천, (3) type3-RI 신뢰성의 사회적 구성. 이 세 가지 지표를 통해 알고리즘의 신뢰성을 평가하고 그 과정에서 발생하는 여러方法을 설명합니다. 이러한 지표는 알고리즘의 설계, 코딩, 사용 및 유지 관리와 같은 다양한 단계에서 신뢰성을 어떻게 제공하는지를 탐구합니다.

- **Performance Highlights**: 알고리즘을 통한 과학적 성과는 주목할 만한 결과를 나타내고 있습니다. 예를 들어, AlphaFold는 단백질 구조를 예측하며, BenevolentAI는 비정형 생물의학 데이터와 정보 검색을 결합하여 COVID-19 증상에 대해 바리시티니브와 같은 약물을 발견했습니다. 이러한 사례들은 알고리즘이 과학 연구에 있어서 매우 유용한 도구가 될 수 있음을 보여줍니다.



### FANformer: Improving Large Language Models Through Effective Periodicity Modeling (https://arxiv.org/abs/2502.21309)
- **What's New**: 이 연구는 최근의 대형 언어 모델(LLMs)들이 데이터와 계산 자원에 대한 높은 요구로 인해 학습 효율성이 낮다는 점을 지적합니다. 새로운 모델 FANformer를 제안하여 Fourier Analysis Network (FAN)를 Transformer의 attention 메커니즘에 통합하여 주기성(periodicity) 모델링을 보다 효율적으로 수행하고 있습니다. FANformer는 Transformer 아키텍처의 학습 효율성과 성능을 개선하기 위해 주기적 패턴을 포착하고 표현하는 Fourier 원리를 활용합니다. 이 연구 결과는 FANformer가 LLM을 발전시키기 위한 유망한 아키텍처가 될 수 있음을 보여줍니다.

- **Technical Details**: FANformer는 샘플 입력 𝐗𝐗oldsymbol{X}의 폴리폼 및 주기 기저 함수(frequency basis functions)를 사용하는 FAN 레이어를 기반으로 구성됩니다. FAN 레이어는 행동 기능을 유지하면서 주기적 패턴을 explicit하게 인코딩하여 MLP 레이어와 차별화됩니다. 또한 attention 메커니즘의 feature projection 과정을 수정하여 frequency domain representations을 통합하고 있습니다. 이러한 구조는 주기성을 효과적으로 캡처하고 모델링할 수 있도록 설계되었습니다.

- **Performance Highlights**: FANformer는 1조 개의 토큰을 활용하여 11억 파라미터의 FANformer-1B를 사전 훈련하였으며, 결과적으로 오픈 소스 LLM보다 뛰어난 성능을 보였습니다. 모델 파라미터와 훈련 토큰이 같은 조건에서 FANformer는 기존의 Transformers보다 적은 자원으로 유사한 성능을 발휘하면서 뛰어난 학습 효율성을 가지고 있음을 확인했습니다. FANformer의 학습 효율성이 더 크고 복잡한 모델에서 더욱 돋보이며, 이는 LLM의 발전 가능성을 시사합니다.



### Clustering Context in Off-Policy Evaluation (https://arxiv.org/abs/2502.21304)
Comments:
          35 pages, 25 figures, 2 tables. AISTATS 2025

- **What's New**: 이번 연구에서는 Off-policy evaluation (OPE) 방법론을 발전시키기 위해 클러스터링을 활용한 대안 추정기를 제안합니다. 기존의 off-policy 기법들은 정책과 로깅 정책 간의 차이로 인해 성능이 저하되는 문제를 겪고 있었으나, 새로운 접근법은 유사한 컨텍스트 간 정보를 공유함으로써 이를 해소하려 시도합니다. 실험을 통해 이 방법이 특히 정보가 부족한 상황에서 추정 정확도를 향상시키는 것을 확인하였습니다.

- **Technical Details**: 연구는 다양한 구조의 컨텍스트 클러스터를 활용하여 클러스터 내의 모든 컨텍스트로부터 정보를 집계하는 방법을 제안합니다. 제안된 추정기인 CHIPS(Clustering-based Importance-weighted Policy Score)는 이론적인 특성을 분석하며, 비편향(bias)과 분산(variance) 특성을 다양한 조건 하에 특성화합니다. OPE 문제의 정의 및 관련 알고리즘을 설명하며, 이론적인 결과들이 현실 세계의 추천 시스템에 어떻게 적용되는지 탐구합니다.

- **Performance Highlights**: CHIPS 추정기의 성능은 여러 합성 데이터셋과 실제 추천 데이터셋에서 기존 기법들과 비교되어 그 효과를 입증했습니다. 실험 결과, 클러스터링을 통한 컨텍스트 정보의 통합이 추정 정확도 향상에 기여하는 것으로 나타났습니다. 특히, 로깅 정책에서 저확률을 갖는 행동이 많은 환경에서 이 방법의 장점이 두드러지게 나타났습니다.



### L-Lipschitz Gershgorin ResNet Network (https://arxiv.org/abs/2502.21279)
Comments:
          10 pages, 6 figures

- **What's New**: 이번 논문은 깊은 잔여 네트워크(Deep Residual Network)를 Linear Matrix Inequality (LMI) 형식으로 재구성하여 $	ext{L}$-Lipschitz 연속성을 보장하는 매개변수 제약을 파생하는 새로운 방법론을 제안합니다. 이는 기존의 접근 방식이 지닌 제약을 극복하며, 복잡한 네트워크 구조에서도 이론적인 안정성을 보장합니다. Gershgorin 원 이론을 활용하여 이 문제를 해결함으로써 고유값의 부정 준거성을 보장하고, 네트워크의 신뢰성을 증가시킵니다.

- **Technical Details**: 이 연구에서는 깊은 잔여 네트워크의 내부 구조를 선형 방정식의 재귀 시스템으로 나타내어 $	ext{L}$-Lipschitz 조건을 유지하면서 표현력을 극대화하고자 하였습니다. LMI 제약 조건을 이용하여 각 레이어의 매개변수를 명확하게 규정하였으며, 일반적인 활성화 함수에서도 적용 가능한 제약 조건을 제시합니다. 또한, 이전 연구들은 단일 레이어로 제한된 잔여 네트워크에 초점을 맞췄지만, 본 연구는 더 다양한 내층 구조를 갖는 일반화된 모델을 구성합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 적대적 공략(adversarial attacks)에 대해 더 높은 견고성을 보여주었으며, 이론적으로 보장된 Lipschitz 제약 덕분에 네트워크의 출력이 입력 변화에 안정적으로 유지되었음을 증명합니다. 그러나 Gershgorin 기반의 근사 방식은 시스템을 과도하게 제약하여 비선형 동적 요소를 억제하며, 이로 인해 네트워크의 표현력이 감소하는 한계를 드러냈습니다. 이러한 점은 향후 연구를 통해 보완해야 할 중요한 사항으로 언급됩니다.



### BAnG: Bidirectional Anchored Generation for Conditional RNA Design (https://arxiv.org/abs/2502.21274)
- **What's New**: 이 논문에서는 RNA 분자와 특정 단백질 간의 상호작용을 예측하기 위해 RNA-BAnG라는 심층 학습 기반 모델을 개발하였습니다. 기존의 접근법과는 달리, 실험적으로 결정된 RNA 서열이나 RNA 구조에 대한 세부 지식 없이도 RNA 서열을 생성할 수 있는 방법을 제시합니다. 또한 새로운 생성 방법인 Bidirectional Anchored Generation (BAnG)을 사용하여, 단백질 결합 RNA 서열이 일반적으로 기능적 결합 모티프를 포함하고 있다는 점을 이용합니다.

- **Technical Details**: BAnG 모델은 두 개의 특수 앵커 토큰 (<ancl>, <ancr>)을 사용하여 서열을 생성하는 방식을 채택합니다. 이 모델은 먼저 오른쪽에서 다음 토큰을 샘플링한 후 왼쪽에서 또 한 번 토큰을 생성하는 방식으로 진행되며, 이 과정을 반복하여 서열이 확장됩니다. 모델 트레이닝과 추론 동안 각 방향의 다음 토큰 확률은 가장 최근에 생성된 토큰의 임베딩을 기반으로 하도록 설계되어 있습니다.

- **Performance Highlights**: RNA-BAnG 모델은 합성 작업과 생물학적 서열 평가에서 그 효과성을 입증하였습니다. 기존의 세대 방법들과 비교했을 때, 제공된 단백질에 대해 조건부 RNA 서열 디자인에서 더 나은 성능을 보여주며, 복잡한 바이오 분자 문제를 해결하는 데 있어 새로운 가능성을 열고 있습니다. 최종적으로, 이 방법은 기존의 실험적 데이터에 의존하지 않고도 RNA 서열 생성을 가능하게 하여, 생물의학적 응용에 크게 기여할 수 있을 것입니다.



### Adaptive Keyframe Sampling for Long Video Understanding (https://arxiv.org/abs/2502.21271)
Comments:
          CVPR2025

- **What's New**: 본 논문은 비디오 이해를 위한 Adaptive Keyframe Sampling (AKS)이라는 새로운 알고리즘을 제안합니다. 이는 기존의 비디오 기반 MLLM들이 비디오 토큰을 선택하는 방식에서 발생하는 정보 손실 문제를 해결하고자 합니다. AKS는 고정된 수의 비디오 토큰으로 유용한 정보를 극대화하는 데 중점을 두며, 키프레임 선택을 통해 비디오의 관련성과 커버리지를 동시에 고려합니다.

- **Technical Details**: 이 연구는 키프레임 선택을 최적화 문제로 모델링하여, 키프레임과 질문 간의 관련성을 측정하고, 키프레임 집합이 비디오에서 유용한 정보를 얼마나 잘 커버하는지를 평가합니다. 전체 알고리즘은 비디오와 텍스트 입력을 결합하여 최적의 키프레임을 선택하고, 고품질의 비주얼 데이터를 기반으로 MLLM의 성능을 향상시키는데 초점을 맞추고 있습니다. AKS는 LLaVA-Video와 같은 기존 MLLM에 통합되어 성능 개선을 확인합니다.

- **Performance Highlights**: AKS는 LongVideoBench와 VideoMME에서 비디오 질문 응답의 정확도를 개선하는 데 성공하였습니다. 본 연구에서 제안된 알고리즘은 고품질 키프레임의 발견 덕분에 MLLM의 전반적인 성능을 향상시켰음을 입증하였습니다. 실험 결과, AKS가 통합된 모델은 모든 테스트에서 일관된 정확도 향상을 보여주며, 이로 인해 새로운 벤치마크 기록을 세우게 되었습니다.



### ReaLJam: Real-Time Human-AI Music Jamming with Reinforcement Learning-Tuned Transformers (https://arxiv.org/abs/2502.21267)
Comments:
          Published in Extended Abstracts of the CHI Conference on Human Factors in Computing Systems (CHI EA '25), April 26-May 1, 2025, Yokohama, Japan

- **What's New**: 최근 생성 인공지능(AI) 음악 분야에서의 발전으로, 고품질 음악 콘텐츠 생성을 위한 모델들이 개발되었습니다. 그러나 이 모델들이 실시간으로 인간과 협력하여 음악을 생성하는 데 어떻게 사용될 수 있는지는 잘 연구되지 않았습니다. 본 논문에서는 인간과 Transformer 기반의 AI 에이전트 간의 라이브 음악 잼 세션을 지원하기 위한 ReaLJam이라는 인터페이스 및 프로토콜을 소개합니다.

- **Technical Details**: ReaLJam 시스템은 실시간 상호작용을 가능하게 하며, AI 에이전트는 예상(anticipation) 개념을 활용하여 성능이 어떻게 전개될지를 예측하고 그 계획을 사용자에게 시각적으로 전달합니다. 시스템은 사용자와 에이전트의 예측을 향상시키고, 노트를 시간 차원에서 정렬하여 지연 없이 연주할 수 있도록 합니다. 이를 통해, 음악 아이디어의 즉흥 생성 및 협력적 공연을 가능하게 합니다.

- **Performance Highlights**: 사용자 연구를 통해 경험이 풍부한 음악가들이 ReaLJam을 통해 AI 에이전트와 함께 즉흥적으로 연주한 결과, 즐거운 음악 경험을 제공하고자 하는 목표가 달성되었습니다. 연구 결과는 음악 모델링 및 사용자 인터페이스 설정의 세부 조정의 중요성에 대한 유의미한 발견을 포함하고 있습니다. ReaLJam은 대규모 Transformer 모델로 실시간 잼을 가능하게 한 첫 번째 시스템으로, 향후 작업을 위한 중요한 통찰을 제공합니다.



### Supporting the development of Machine Learning for fundamental science in a federated Cloud with the AI_INFN platform (https://arxiv.org/abs/2502.21266)
Comments:
          Under review in EPJ Web of Conferences (CHEP 2024)

- **What's New**: 최근 논문에서는 인공지능(AI)과 기계 학습(ML)이 INFN의 데이터 집약적 소프트웨어 개발 방식을 혁신하고 있음을 강조하고 있습니다. AI_INFN 프로젝트는 INFN의 사례에서 ML 기술 채택을 촉진하기 위해 AI 맞춤형 컴퓨팅 자원을 제공합니다. 이 프로젝트는 INFN Cloud의 클라우드 네이티브 솔루션을 활용하여 하드웨어 가속기를 효과적으로 공유하고 연구 활동의 다양성을 보장합니다.

- **Technical Details**: AI_INFN 이니셔티브는 하드웨어 가속기를 포함한 컴퓨팅 인프라를 구축 및 운영하고, 머신 러닝 채택을 위한 교육 이벤트를 조직하며, INFN 내 ML 전문가 및 개발자 커뮤니티를 만들기 위한 네 가지 작업 패키지로 구성되어 있습니다. 또한, GPU 뿐만 아니라 FPGA 및 양자 프로세서와 같은 하드웨어 가속기를 활용할 수 있는 역량을 개발하는 데 초점을 맞추고 있습니다. 이 플랫폼은 High Performance Computing(HPC) 작업을 위해 설계된 전문 하드웨어를 기반으로 하며, INFN CNAF에서 호스팅 및 관리됩니다.

- **Performance Highlights**: AI_INFN 플랫폼은 2024년 1월 시작되었습니다. 현재까지 72명의 연구자가 16개의 연구 활동에 접근 요청을 하였으며, 평균적으로 하루에 10명에서 15명이 플랫폼에 접속하고 있습니다. 초기에는 ML_INF 프로젝트에서 가속 자원을 공유하기 위한 모델이 운영되었으나, 사용량 증가로 인해 관리 부담과 비효율성 문제가 발생하였습니다. 이런 문제를 해결하고자 새로운 모델을 도입하여 사용자들이 자신의 클라우드 기반 컴퓨팅 환경을 조정할 수 있습니다.



### Foundation Models -- A Panacea for Artificial Intelligence in Pathology? (https://arxiv.org/abs/2502.21264)
Comments:
          50 pages, 15 figures and an appendix (study protocol) which is previously published, see this https URL

- **What's New**: 최근 인공지능(AI)의 역할은 진단 지원에서 전체 슬라이드 이미지(WSI)에서 예측적 형태학 패턴을 발견하는 방향으로 진화하였습니다. 본 연구에서는 전 세계 11개 국가의 15개 사이트에서 수집된 100,000건 이상의 코어 바늘 생검 데이터를 활용해 전립선암 진단 및 Gleason 등급 평가를 위한 AI의 임상 성능을 집중적으로 탐구하였습니다.

- **Technical Details**: 우리는 다중 인스턴스 학습(multiple instance learning) 프레임워크 내에서 두 가지 기초 모델(foundation models, FMs)과 완전한 엔드 투 엔드(task-specific, TS) 모델을 비교하였습니다. 결과적으로, FMs가 데이터가 부족한 상황에서 유용성을 보여주었지만, 충분한 라벨이 있는 훈련 데이터가 제공될 때 TS 모델과 유사한 성능을 보였으며, 때로는 초과하기도 했습니다.

- **Performance Highlights**: FMs는 TS 모델보다 최대 35배 많은 에너지를 소모했으며, 이는 지속 가능성에 대한 우려를 불러일으킵니다. TS 모델은 임상적으로 중요한 오등급, 난해한 형태학의 오진 및 다양한 WSI 스캐너 간의 변동성을 크게 줄였습니다. 연구 결과는 FMs가 빠른 프로토타이핑 및 연구에 유리하지만, 임상 적용에 있어 유니버설 솔루션으로서의 역할은 불확실하다는 점을 강조하고 있습니다.



### RuCCoD: Towards Automated ICD Coding in Russian (https://arxiv.org/abs/2502.21263)
- **What's New**: 이 연구는 러시아어로 된 임상 코딩 자동화의 가능성을 검토하고 있습니다. 10,000개 이상의 엔티티와 1,500개 이상의 고유한 ICD 코드가 주석 처리된 새로운 ICD 코딩 데이터셋을 제시합니다. 이 데이터셋은 여러 최신 모델에 대한 벤치마크로 사용되며, 전이 학습(transfer learning)을 검토하여 PubMed 초록에서 의학적 진단으로의 변화를 포함합니다.

- **Technical Details**: 연구에서 제시한 RuCCoD(Russian ICD Coding Dataset)는 의료 전문가에 의해 ICD-10 CM 시스템에 기반하여 레이블링된 데이터셋입니다. 우리가 개발한 모델은 BERT을 기반으로 한 정보 추출 파이프라인과 LoRA가 적용된 LLaMA 모델, 그리고 RAG를 포함합니다. 실험은 2017년부터 2021년까지의 환자 기록이 포함된 대규모 EHR 데이터셋에 대해 수행되었습니다.

- **Performance Highlights**: 자동으로 예측된 코드를 사용한 훈련이 의사에 의해 주석 처리된 데이터에 비해 정확도를 크게 향상시킨다는 것을 보여줍니다. 특정 실험에서는 자동으로 할당된 ICD 코드로 사전 훈련을 통해 진단 예측에서 macro-averaged F1-score가 28% 증가했습니다. 경량 콘텐츠가 부족한 러시아어와 같은 언어에서 임상 코딩의 자동화를 위한 기반을 제공할 것으로 기대합니다.



### ByteScale: Efficient Scaling of LLM Training with a 2048K Context Length on More Than 12,000 GPUs (https://arxiv.org/abs/2502.21231)
Comments:
          12 pages, 21 figures

- **What's New**: 최근 대규모 언어 모델(LLM)의 긴 맥락 처리가 필수적이게 되었습니다. 이에 따라 ByteScale이라는 새로운 훈련 프레임워크가 제안되었습니다. 이 프레임워크는 Hybrid Data Parallelism(HDP)이라 불리는 새로운 병렬 처리 전략을 통해 장단기 시퀀스를 효율적으로 혼합하여 훈련할 수 있도록 설계되었습니다.

- **Technical Details**: ByteScale의 핵심은 기존의 inter-data partitioning(데이터 병렬 처리)과 intra-data partitioning(맥락 병렬 처리)을 통합하는 HDP입니다. 이 방식은 동적 메쉬 디자인을 활용하여, 각 시퀀스가 각기 다른 길이를 가질 때에도 유연하게 처리가 가능합니다. 또한 통신 최적화 기법을 통해 짧은 시퀀스에서 불필요한 통신을 줄이고, 긴 시퀀스에서는 선택적 오프로드를 통해 통신 비용을 축소합니다.

- **Performance Highlights**: 12,000개 이상의 GPU가 운영되는 생산 클러스터에서 7B에서 141B까지 다양한 모델 크기와 256K에서 2048K까지의 맥락 길이로 ByteScale을 평가했습니다. 실험 결과, ByteScale은 기존 훈련 방법에 비해 최대 7.89배의 속도 향상을 달성하여 훈련 효율성을 크게 개선했습니다.



### ECLeKTic: a Novel Challenge Set for Evaluation of Cross-Lingual Knowledge Transfer (https://arxiv.org/abs/2502.21228)
- **What's New**: 이번 논문에서는 다국어 대형 언어 모델(LLMs)의 언어 간 지식 전이 능력을 평가하기 위한 새로운 데이터셋, ECLeKTic을 제안합니다. ECLeKTic은 12개 언어에서 비대칭적으로 커버된 정보를 기반으로 만들어졌으며, 모델이 언어를 초월해 지식을 전이할 수 있는 능력을 검증하기 위한 간단한 방법론을 사용합니다. 이를 통해 기존 문헌에서 부족한 경험적 지식 전이 평가 문제를 해결하고자 합니다.

- **Technical Details**: ECLeKTic은 다국어 폐쇄형 질문-답변(CBQA) 데이터셋으로, 각 언어에서 특별히 참고할 수 있는 위키피디아 기사를 중심으로 한 질문-답변 쌍을 생성합니다. 12개 언어 중 하나의 언어에서 작성된 기사를 기반으로 질문을 생성하고, 이 질문을 해당 기사가 없는 다른 11개 언어로 번역했습니다. 이를 통해 모델이 언어 간 지식을 전이할 수 있는 기관적 능력을 평가할 수 있음을 보입니다.

- **Performance Highlights**: 실험 결과 8개의 상위 모델로 ECLeKTic를 테스트한 결과, 최고의 성능을 기록한 모델은 41.3%의 전체 성공률과 65%의 지식 전이 능력을 보였습니다. 이 결과는 대다수의 최첨단 모델이 언어 간 지식 공유에서 어려움을 겪고 있음을 나타내며, 더 나은 다국어 모델 개발을 위한 임계점을 제시합니다. 언어 간 스크립트 공유가 전이에 중요한 요인으로 작용함을 보여주는 것이 인상적입니다.



### XAIxArts Manifesto: Explainable AI for the Arts (https://arxiv.org/abs/2502.21220)
Comments:
          Author version of paper in: Extended Abstracts of the CHI Conference on Human Factors in Computing Systems, April 26-May 1, 2025, Yokohama, Japan DOI https://doi.org/10.1145/3706599.3716227 ISBN 979-8-4007-1395-8/25/04

- **What's New**: 이번 논문은 기존의 기술 중심적 (technocentric) 설명 가능성의 개념을 넘어선 새로운 사고 방식인 XAI for the Arts (XAIxArts) 매니페스토를 소개합니다. 이 매니페스토는 예술과 AI의 교차점을 탐색하며, 다양한 사용자 그룹의 접근성을 증가시키고 예술적 관행을 높이 평가하는 데 중점을 두고 있습니다. 또한, XAI의 새로운 가능성을 제공하고 AI에 대한 신뢰를 구축하는 방법에 대한 논의를 촉진하려는 노력을 포함하고 있습니다.

- **Technical Details**: XAIxArts 작업은 HCI(인간-컴퓨터 상호작용) 및 디지털 예술 연구자와 창작자들로 구성된 39명의 참가자들이 모여 설명 가능 AI의 현재와 미래의 관행을 비판적으로 성찰하는 것이 중심되었습니다. 참가자들은 각자의 연구와 예술적 접근 방식을 공유하며 설명 가능성의 정의와 확장을 논의했습니다. 작업에서 생긴 여러 작품들은 예술적 관행을 통해 AI 모델을 더 이해 가능한 방식으로 제시하는 것을 목표로 하며, HCI 및 AI의 영역에서 상호 작용을 촉진합니다.

- **Performance Highlights**: 이 논문은 XAIxArts 매니페스토의 첫 번째 출판물로, 예술과 AI의 융합을 통한 혁신과 이해의 미래에 대한 비전을 제시합니다. 연구자와 예술가들이 모여 실질적인 변화를 위한 효과적인 접근 방법을 개발할 수 있도록 유도하는 목표를 가지고 있습니다. 이를 통해 XAIxArts 커뮤니티가 활성화되고, 다양한 사용자 그룹이 AI 도구에 더 쉽게 접근할 수 있도록 하는 데 기여하고자 합니다.



### Transformers Learn to Implement Multi-step Gradient Descent with Chain of Though (https://arxiv.org/abs/2502.21212)
Comments:
          ICLR 2025 Spotlight

- **What's New**: 이 논문에서는 Chain of Thought (CoT) 프롬프트가 대형 언어 모델(LLM)의 성능을 두드러지게 향상시킨다는 점에 집중합니다. 특히, 수학적 계산 및 추론 작업에서 CoT가 중간 추론 단계를 생성하도록 모델에 지시함으로써 얻은 성과를 보여줍니다. CoT의 훈련 동역학을 탐구하며, 그 기초 메커니즘이 대부분 미개척되어 있음을 강조합니다.

- **Technical Details**: 저자들은 CoT 목표에 대한 트랜스포머의 훈련 동역학을 연구했습니다. 이들은 선형 회귀의 컨텍스트 내 가중치 예측 작업을 통해, CoT가 없는 1계층 선형 트랜스포머는 단일 단계의 경량하강법(Gradient Descent, GD)만을 구현할 수 있음을 입증합니다. 반면 CoT 프롬프트를 사용한 트랜스포머는 다단계 GD를 자가 회귀적으로 학습하여 거의 정확하게 진실 가중치 벡터를 복원할 수 있음을 보여줍니다.

- **Performance Highlights**: 실험적으로 CoT 프롬프트가 상당한 성능 향상을 가져온 것을 증명하였습니다. 또한 훈련된 트랜스포머는 보지 못한 데이터에서도 효과적으로 일반화할 수 있습니다. 루프된 트랜스포머는 선형 회귀의 컨텍스트 내 학습에서 루핑이 없는 트랜스포머에 비해 최종 성능을 크게 개선합니다.



### The PanAf-FGBG Dataset: Understanding the Impact of Backgrounds in Wildlife Behaviour Recognition (https://arxiv.org/abs/2502.21201)
Comments:
          Accepted at the IEEE / CVF Computer Vision and Pattern Recognition Conference 2025

- **What's New**: 이 논문에서는 야생 침팬지의 행동을 기록한 PanAf-FGBG 데이터셋을 소개합니다. 350개 이상의 카메라 위치에서 촬영된 20시간의 비디오로 구성되어 있으며, 각 침팬지 비디오와 해당 비디오에서 침팬지가 없는 배경 비디오를 쌍으로 제공합니다. 이를 통해 행동 인식을 위한 데이터셋에서 배경의 역할과 이를 통해 모델 성능을 개선할 수 있는 방법을 명확히 평가할 수 있는 기회를 제공합니다.

- **Technical Details**: PanAf-FGBG 데이터셋은 각 침팬지 행동 비디오(전경 비디오)와 같은 카메라 위치에서 촬영된 관련 배경 비디오(배경 비디오)의 쌍으로 구성됩니다. 이 데이터셋은 두 가지 실험 조건으로 나뉘어져, 겹치는 카메라 위치 및 분리된 카메라 위치을 포함하며, 이를 통해 실질적인 배경 영향 분석이 가능합니다. 또한, 다양한 행동 주석과 메타 데이터를 포함하여, 인-디스트리뷰션(in-distribution) 및 아웃-오브-디스트리뷰션(out-of-distribution) 상황을 평가하는 데 필수적인 정보를 제공합니다.

- **Performance Highlights**: 저자들은 배경만으로 훈련된 모델이 즉각적으로 65%의 성능을 보여준다고 주장합니다. 또한, latent-space 배경 정규화 기술을 도입하여, OOD 시나리오에서 컨볼루션 모델의 mAP(Mean Average Precision)이 +5.42%, 트랜스포머 기반 모델의 경우 +3.75% 향상된 성능을 달성했다고 보고했습니다. 이를 통해 배경 기간, 즉 전경 비디오 내 배경 프레임 수의 영향이 행동 인식 모델에 미치는 영향이 밝혀졌습니다.



### AMPLE: Event-Driven Accelerator for Mixed-Precision Inference of Graph Neural Networks (https://arxiv.org/abs/2502.21196)
- **What's New**: 이번 논문에서 제안하는 AMPLE(Accelerated Message Passing Logic Engine)는 이벤트 기반 프로그래밍 흐름을 활용하여 GNN(그래프 신경망) 추론을 최적화하는 FPGA(전계 프로그래머블 게이트 어레이) 가속기입니다. GNN은 비유클리드 데이터에서 성능을 발휘하는 강력한 방법으로 주목받고 있으며, AMPLE은 특히 불규칙한 노드 분포로 인한 비효율적인 메모리 접근 문제를 해결하고자 합니다. 이 아키텍처는 다양한 정밀도의 다중 집합 Aggregation Core를 통합하여 노드의 차수와 정밀도에 따라 동적으로 할당하는 특징이 있습니다.

- **Technical Details**: IMPLE의 아키텍처는 메모리 맵 된 레지스터를 통해 비동기적으로 노드를 프로그래밍할 수 있는 이벤트 기반 프로그래밍 모델을 제공하며, 그래프 데이터의 다양성을 효과적으로 관리합니다. GNN 추론을 노드 수준에서 양자화할 수 있도록 혼합 산술 아키텍처가 개발되었습니다. 또한 외부 메모리에 대한 데이터 및 명령 프리패쳐를 구현하여 메모리 접근 최적화 및 노드의 병렬성을 극대화합니다.

- **Performance Highlights**: 2K에서 700K 노드 범위의 데이터셋에 대한 성능 평가에서 AMPLE은 CPU 및 GPU에 비해 각각 평균 243배 및 7.2배의 속도를 기록했습니다. 이러한 성능 향상은 GNN의 집합(gathering) 단계의 비효율성을 극복하고 연산 패턴을 최적화한 결과로, 향후 GNN 처리 시스템의 발전에 기여할 것으로 기대됩니다.



### Scalable Decision-Making in Stochastic Environments through Learned Temporal Abstraction (https://arxiv.org/abs/2502.21186)
Comments:
          Accepted by ICLR2025. Code would be available at \href{this https URL}{this https URL}

- **What's New**: 본 논문은 고차원 연속 행동 공간에서의 순차적인 의사결정 문제를 다룹니다. 특히, 전통적인 Offline RL 환경에서 에이전트가 확률적 행동 정책을 바탕으로 의사결정을 학습해야 하는 복잡한 도전 과제를 해결하는 방안을 제시합니다. 이를 위해 저자들은 Latent Macro Action Planner (L-MAP)라는 새로운 접근법을 도입하여, 상태 조건적 Vector Quantized Variational Autoencoder (VQ-VAE)를 활용해 행동 차원을 효과적으로 감소시킵니다.

- **Technical Details**: L-MAP은 잠재적 전이 모델로 작동하는 별도의 학습된 사전(prior) 모델을 사용하여 plausible한 행동을 효율적으로 샘플링합니다. 이 접근법은 Monte Carlo Tree Search (MCTS)를 통해 환경과 행동 정책의 확률성을 고려하여 계획을 수립하며, 고차원 행동 공간에서의 의사결정 절차에서 이전 모델들이 처리했던 비효율성을 크게 줄입니다. L-MAP은 디스크리트한 잠재 행동을 탐색하여 높은 기대 수익을 제공하는 방식으로 작동합니다.

- **Performance Highlights**: 실험 결과, L-MAP는 고차원 행동 공간과 본래 확률적 동역학을 포함한 다양한 과제에서 기존 모델 기반 방법을 초월하며, 강력한 모델 없는 Actor-Critic 기반 모델과도 동등한 성능을 보였습니다. 특히, L-MAP은 결정 지연을 줄이면서도 높은 성능을 유지하여, 스토캐스틱한 환경에서의 의사결정에 있어 탁월한 효과성을 입증합니다.



### Predicting clinical outcomes from patient care pathways represented with temporal knowledge graphs (https://arxiv.org/abs/2502.21138)
- **What's New**: 이 논문은 뇌동맥류( intracranial aneurysm )의 예후를 예측하기 위해 그래프 임베딩(Graph Embedding)과 그래프 신경망(Graph Convolutional Network, GCN)을 이용하는 방법을 탐구합니다. 특히, 환자의 개인적 특성과 병원에서 관찰된 치료 정보를 결합하여 adverse outcomes를 예측하는 방법론을 제시하고 있습니다. 데이터는 합성으로 생성된 것이지만 실제 환자의 경과를 반영하였으며, tabular 데이터와 그래프 기반 표현을 비교하여 최적의 예측 성능을 도출했습니다.

- **Technical Details**: 연구에서는 그래프의 구성 요소인 노드와 간선을 연속 벡터 공간으로 매핑하는 그래프 임베딩을 사용하여 데이터 분석을 수행합니다. Knowledge Graph(KG)에서 정의된 임베딩 기법을 적용하여 환자 데이터를 표현하고, 특히 GCN을 이용해 노드 간의 이웃 정보를 활용하여 예측 성능을 높이는 데 중점을 두었습니다. 또한 개인 데이터 표현과 시간 데이터 표현의 모델링 방식이 예측 성능에 미치는 영향을 실험하였습니다.

- **Performance Highlights**: 실험 결과, GCN 임베딩을 이용한 그래프 표현이 tabular 데이터 기반의 전통적 분류 기법보다 우수한 성능을 보임을 확인했습니다. 연구의 결과는 환자 특성의 컴팩트한 표현이 예측 성능에 유의미한 긍정적 영향을 미친다는 것을 보여주며, 시간 표현 방식이 예측 성능에는 큰 영향을 미치지 않는다는 점도 강조되었습니다. 이는 생물의학 예측 모델링에서 그래프 기반 접근 방식의 가능성을 제시합니다.



### Dynamically Local-Enhancement Planner for Large-Scale Autonomous Driving (https://arxiv.org/abs/2502.21134)
- **What's New**: 이번 연구에서는 기본 주행 플래너에 지역 주행 데이터를 동적으로 보강하는 개념인 Dynamically Local-Enhancement (DLE) 플래너를 소개합니다. 이 방법은 플래너를 영구적으로 수정하지 않고도 자율 주행 시스템의 확장성을 향상시키고자 합니다. 특히, 지역 특정 주행 특징을 추출하기 위해 graph neural network를 사용하여 기본적인 강화 학습 기반 정책을 개선하는 방법을 제안합니다.

- **Technical Details**: DLE 플래너는 지역에서 수집된 데이터를 통해 기본 주행 정책을 보강하는 방식으로 작동합니다. 이는 지능형 시스템이 다양한 주행 환경에서 적응할 수 있도록 돕습니다. 연구에서는 Markov Decision Process (MDP)와 그래프 신경망의 결합을 사용하여 스파이셜 및 타임템포럴(spatiotemporal) 표현에 기반한 지역 특징을 동적으로 추출하여 모델의 계산 부담을 최소화합니다.

- **Performance Highlights**: DLE 플래너는 여러 시나리오에서 평가되어 전통적인 일괄 모델에 비해 안전성과 평균 보상 측면에서 우수한 성과를 보였습니다. 연구 결과, 이 방법은 자율 주행 시스템의 대규모 배치에 기여할 수 있는 가능성을 보여주며, 장치 내 주행 모델의 확장을 대폭 줄이면서도 효율성을 향상시킵니다.



### Einleitung [Introduction] (https://arxiv.org/abs/2502.21131)
Comments:
          in German language

- **What's New**: 힐러리 퍼트넘(Hilary Putnam)의 전기와 철학적 발전은 지난 40년간 앵글로색슨 철학의 역사를 반영합니다. 퍼트넘은 이 역사에 상당한 영향을 미쳐왔습니다. 본 논문의 서문에서는 퍼트넘의 철학적 기여를 이해하기 위한 맥락을 제시하는 것이 주된 목표입니다.

- **Technical Details**: 퍼트넘의 철학적 발전에 대한 개요와 함께 그의 작업을 초기 철학사적 분류 시도도 다룹니다. 그러나 이 논문은 포괄적인 비판이나 제시가 아닌, 기본적인 수준에 머물러야 합니다. 또한, 퍼트넘의 작업은 '분석 철학(analytic philosophy)'과 '대륙 철학(continental philosophy)' 간의 rapprochement(접근)에 중요한 부분을 차지하고 있음을 강조합니다.

- **Performance Highlights**: 서문에서는 번역된 텍스트에 대한 독자들에게 퍼트넘이 어떤 가치를 제공하는지를 명확히 해야 합니다. 퍼트넘의 철학은 철학적 대화를 확장하는 데 도움을 줄 수 있으며, 비분석적 독자에게도 매력적인 통찰을 제공합니다.



### Causality Is Key to Understand and Balance Multiple Goals in Trustworthy ML and Foundation Models (https://arxiv.org/abs/2502.21123)
- **What's New**: 최근 기계 학습(ML) 시스템의 신뢰성을 보장하는 것이 중요해졌습니다. 이 논문은 형평성(fairness), 개인 정보 보호(privacy), 견고성(robustness), 정확성(accuracy), 설명 가능성(explainability) 등의 원칙 간의 균형을 위해 인과적(causal) 방법의 통합을 주장합니다. 이를 통해 서로 상충하는 목표를 해결하고, 새로운 해결책을 제시합니다.

- **Technical Details**: 이 논문에서는 인과성(causality) 모델을 통해 신뢰할 수 있는 ML 시스템의 다양한 차원을 조화롭게 통합할 수 있는 방법을 제안합니다. Pearl의 구조적 인과 모델(Structural Causal Models, SCMs)을 사용하여 인과성을 정의하고, 유향 비순환 그래프(directed acyclic graphs, DAGs)를 통해 인과 종속성을 표현합니다. 이러한 접근법은 데이터의 근본적인 메커니즘을 파악하는 데 도움을 줍니다.

- **Performance Highlights**: 인과적 방법론은 형평성과 정확성 또는 개인 정보 보호와 견고성과 같은 서로 대립하는 목표를 성공적으로 조화시켜 ML의 신뢰성을 향상시키는 데 기여할 수 있습니다. 논문은 미래 연구 방향과 제약에 대해서도 논의하며, 신뢰할 수 있는 AI 시스템을 구축하기 위한 더 윤리적이고 책임감 있는 접근 방식을 모색합니다.



### AuthSim: Towards Authentic and Effective Safety-critical Scenario Generation for Autonomous Driving Tests (https://arxiv.org/abs/2502.21100)
- **What's New**: 이 연구에서는 자율주행 테스트 시나리오의 진정성과 효과성을 종합적으로 해결하는 첫 번째 시도를 제안합니다. 기존의 방법들은 주로 무제한 충돌 시나리오에 집중해 왔으며, 이는 종종 비현실적이고 극단적인 시나리오를 생성합니다. 반면, 본 연구는 세 층의 상대 안전 영역 모델을 통해 NPC 차량들이 상대적으로 안전한 경계 영역 내에서 적대적 행동을 하도록 지시하여 시나리오의 진정성을 높입니다.

- **Technical Details**: 본 연구에서 제안된 세 층 상대 안전 영역 모델은 위험 수준에 따라 구역을 나누며, NPC 차량이 상대 경계 영역으로 들어갈 가능성을 높입니다. 또한, 보강 학습( reinforcement learning)을 통합한 AuthSim 플랫폼을 도입하여 안전 위험 시나리오를 실제적이고 효과적으로 생성합니다. 이 모델은 NPC 차량의 상태와 행동 공간을 설계하고, 보상 함수로 세 층 안전 영역 모델을 사용하여 적대적 시나리오를 생성합니다.

- **Performance Highlights**: AuthSim은 기존 방법들보다 5.25% 향상된 평균 컷인 거리 및 27.12% 증가한 평균 충돌 간격 시간 개선을 기록하며, 안전 위험 시나리오의 생성에서 더 높은 효율성을 보여주었습니다. 이러한 연구 결과는 기존 방법들보다 진정한 시나리오 생산에서 AuthSim의 우수성을 강조합니다. extensive experiments를 통해 AuthSim의 효율성과 진정성이 검증되었습니다.



### PASemiQA: Plan-Assisted Agent for Question Answering on Semi-Structured Data with Text and Relational Information (https://arxiv.org/abs/2502.21087)
- **What's New**: 본 논문에서는 PASemiQA라는 새로운 방법론을 제안합니다. PASemiQA는 반구조화 데이터(semistructured data)에서 텍스트와 관계 정보를 결합하여 질문에 답하도록 설계되었습니다. 기존의 RAG(리트리벌 증강 생성) 방법들이 단일 유형의 외부 데이터에 초점을 맞춘 반면, PASemiQA는 다양한 데이터 유형을 효과적으로 활용할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 이 방법은 먼저 질문에 대한 관련 텍스트 및 관계 정보를 식별하기 위한 계획 모듈(plan module)을 생성하고, 이후에 LLM(대형 언어 모델)을 기반으로 한 에이전트를 사용해 반구조화 데이터를 탐색합니다. 이 과정에서 필요한 정보를 추출하여 질문에 대한 정확한 답변을 생성하는데 초점을 맞춥니다. 또한 이 접근 방식은 기존 RAG나 KGQA(지식 그래프 질문 응답) 방법과는 다르게, 질문에 의해 결정된 복합 관계와 텍스트 정보를 동시에 처리합니다.

- **Performance Highlights**: PASemiQA는 다양한 도메인의 여러 반구조화 데이터셋을 통해 그 효과성을 입증하였습니다. 실험 결과, 이 방법은 질문 응답 시스템의 정확성과 신뢰성을 향상시키는데 기여할 수 있음을 보여주었습니다. 나아가 PASemiQA는 향후 복잡한 반구조화 질문에 대한 응답 생성의 새로운 가능성을 모색할 수 있는 기반이 될 것입니다.



### Enhancing deep neural networks through complex-valued representations and Kuramoto synchronization dynamics (https://arxiv.org/abs/2502.21077)
- **What's New**: 이 논문에서는 Kuramoto dynamics를 활용하여 인공 신경망에서 개체 인식을 강화하는 새로운 접근법을 제안합니다. 특히, 복소수 표현(complex-valued representation)과 동기화(synchrony) 메커니즘을 결합하여 멀티 객체를 효과적으로 인식할 수 있도록 합니다. 저자들은 두 가지 아키텍처를 비교하였고, 이러한 모델들은 기존의 모델들에 비해 우수한 성능을 보여 정확성과 일반화 능력을 향상시키는 잠재력을 시사합니다.

- **Technical Details**: 저자들은 복소수 단위를 포함한 계층적 모델인 KomplexNet을 설계하였습니다. 이 모델은 Kuramoto dynamics를 이용하여 초기 레이어에서 동기화된 상태를 유도하고 이를 통해 영상 특징을 결합합니다. 또한 피드백 연결을 통해 동기화를 정교화시키는 방법을 도입하여 객체 표현의 구조를 강화합니다. 이러한 접근법은 첨단 심층 학습 아키텍처의 견고성과 일반화 능력을 높이는데 기여합니다.

- **Performance Highlights**: KomplexNet은 다중 객체 이미지 작업에서 특히 뛰어난 성능을 보여주었습니다. 저자들은 이 모델이 Gaussian noise에 강하고, 전이 학습(out-of-distribution classification) 문제에서도 우수한 일반화 능력을 보인다는 점을 강조합니다. 피드백 연결을 추가한 KomplexNet은 동기화의 정확성을 높여 더욱 강력한 성능을 보였으며, 이는 모델의 일반화 능력과 로버스트니스에 긍정적인 영향을 미쳤습니다.



### FC-Attack: Jailbreaking Large Vision-Language Models via Auto-Generated Flowcharts (https://arxiv.org/abs/2502.21059)
Comments:
          13 pages, 6 figures

- **What's New**: 이번 연구에서는 뷰와 텍스트를 통합한 대형 비전-언어 모델(LVLMs)이 멀티모달 jailbreak 공격에 취약함을 발견했습니다. 이러한 공격은 모델이 해로운 콘텐츠를 생성하도록 유도하며, 이는 안전 위험을 초래할 수 있습니다. 우리는 자동 생성된 플로우차트를 기반으로 한 FC-Attack이라는 새롭고 효과적인 jailbreak 공격 방법을 제안하고, 이를 통해 LVLMs의 취약점을 더욱 조사합니다.

- **Technical Details**: FC-Attack은 사전 학습된 LLM을 미세 조정하여 단계 설명 생성기를 만들고, 이를 사용하여 해로운 쿼리에 대한 단계 설명을 생성합니다. 생산된 단계 설명은 세 가지 형태(수직, 수평, S자형)의 플로우차트로 변환되어 LVLMs에 주입됩니다. 이 공격의 실험은 Advbench 데이터셋을 사용하여 수행되었으며, 다양한 모델에서 90% 이상의 공격 성공률(ASR)을 달성했습니다.

- **Performance Highlights**: FC-Attack은 Gemini-1.5, Llava-Next, Qwen2-VL, InternVL-2.5 모델에서 기존 방법을 초월하는 수치의 ASR을 기록했습니다. 플로우차트 내 글꼴 스타일의 변화와 같은 다양한 요소가 공격 성과에 미치는 영향을 조사한 결과, Claude-3.5 모델에서 ASR이 4%에서 28%로 향상되는 성과를 거두었습니다. 방어 전략 또한 고려하여 AdaShield-A가 공격 성과를 크게 줄일 수 있었지만 유용성의 감소라는 대가를 치러야 했습니다.



### Robust Deterministic Policy Gradient for Disturbance Attenuation and Its Application to Quadrotor Contro (https://arxiv.org/abs/2502.21057)
- **What's New**: 이 논문에서는 Robust Deterministic Policy Gradient (RDPG)라는 강화학습 알고리즘을 제안하여 $H_\infty$ 제어 문제를 두 플레이어 제로섬 동적 게임으로 재구성하고 있습니다. 이러한 재구성을 통해, 사용자(제어기)는 비용을 최소화하고 상대방(교란)은 그것을 최대화하는 역할을 맡습니다. 이 새로운 접근법은 기존의 복잡하고 계산 집약적인 $H_\infty$ 제어기 디자인 방식을 개선합니다.

- **Technical Details**: RDPG 알고리즘은 deterministic policy gradient (DPG) 방법을 기반으로 하며, 사용자와 상대방 각각이 자신의 정책을 학습하여 최적의 제어 입력을 결정할 수 있도록 합니다. 특히, 이 논문에서는 기존 DDPG의 변형인 robust deep deterministic policy gradient (RDDPG) 방식을 도입하여, 깊은 신경망 아키텍처와 twin-delayed DDPG (TD3) 기술을 결합하여 안정성과 학습 효율을 높이고 있습니다. 이렇게 제안된 방법은 외부 교란이 있는 동적 환경에서도 효율적으로 동작합니다.

- **Performance Highlights**: 제안된 RDDPG 방법은 UAV가 교란이 발생하는 환경에서 미리 정의된 경로를 따라 비행할 때 다른 제어 접근 방식보다 더욱 견고함을 나타내며, 심지어 심각한 교란 조건에서 이동하는 목표를 정확히 추적하는 데 성공하였습니다. 수치 시뮬레이션 결과, RDDPG는 DDPG, PPG, SAC 및 TD3와 같은 최신 DRL 접근 방식보다 낮은 비용을 달성하여 효과적인 성능을 보여주었습니다.



### Synthesizing Individualized Aging Brains in Health and Disease with Generative Models and Parallel Transpor (https://arxiv.org/abs/2502.21049)
Comments:
          20 pages, 9 figures, 6 tables, diffeomorphic registration, parallel transport, brain aging, medical image generation, Alzheimer's disease

- **What's New**: 이 논문은 Individualized Brain Synthesis (InBrainSyn)이라는 새로운 프레임워크를 소개하며, 이는 개별 뇌 영상을 기반으로 한 고해상도의 개인 맞춤형 장기적 MRI 스캔을 합성하는 기술입니다. InBrainSyn은 생성적 심층 템플릿 네트워크의 인구 수준 노화 궤적을 적응시키기 위해 병렬 전달 알고리즘을 사용하며, 알츠하이머병(AD) 및 정상 노화의 신경퇴행을 시뮬레이션합니다. 이 방법은 단일 기준 스캔만으로도 신뢰할 수 있는 3D 시공간 T1w MRI 스캔을 합성할 수 있는 장점을 제공하여 개별화된 노화 궤적을 생성합니다.

- **Technical Details**: InBrainSyn은 다양한 건강 상태에서 개별적인 뇌의 노화 궤적을 예측하기 위해, 인체의 해부학적 변화를 정확히 반영하는 매개변수화된 연속적 변환(또는 diffeomorphic transformation)을 이용합니다. 이 과정에서는 인구 수준의 데이터만 사용하여 생성된 궤적들을 통해 개인 맞춤형 노화를 예측하는 데 필요한 정보를 끌어냅니다. 또한, 기존의 MRI 영상의 처리에서 발생할 수 있는 강도 아티팩트(intensity artifacts)를 방지할 수 있도록 해부학적으로 타당한 이미지를 생성하는 데 중점을 두고 있습니다.

- **Performance Highlights**: InBrainSyn은 알츠하이머병 및 건강한 대조군 집단에 대한 정량적 및 정성적 평가를 통해 그 성능을 입증했습니다. 실험 결과, InBrainSyn은 정상 노화와 AD 사이의 신경해부학적 전이도 모델링할 수 있음을 보여주었습니다. 고유한 뇌 스캔을 기반으로 한 시뮬레이션이 각 개인에게 맞춤화된 결과를 내제화하여, 신경영상 분야에서의 임상적 이점과 연구에 대한 통찰력을 제공할 수 있음을 명확히 하고 있습니다.



### Fast Adversarial Training against Sparse Attacks Requires Loss Smoothing (https://arxiv.org/abs/2502.21041)
- **What's New**: 이 논문은 $l_0$ 노름에 의해 제한된 희소(side-by-side sparse) 적대적 perturbations에 대한 빠른 적대적 훈련 방법을 연구합니다. 기존의 1단계 공격을 사용할 때 발생하는 성능 저하와 치명적인 과적합(cO) 문제를 강조합니다. 특히, $l_0$ 적대적 훈련에서 cO는 1단계 공격의 최적화되지 않은 perturbation 위치에 의해 초래됩니다.

- **Technical Details**: 이 연구는 $l_0$ 적대적 훈련의 손실 경관(loss landscape)이 다른 노름인 $l_	ext{infinity}$, $l_2$, $l_1$에 비해 더욱 억세고(craggy), 이는 치명적인 과적합(cO)을 악화시켜 성능 저하를 초래하는 것을 이론적 및 실증적으로 분석합니다. 논문에서 제안한 Fast-LS-$l_0$는 소프트 레이블(soft labels)과 trade-off 손실 함수를 포함하여 적대적 손실 경관을 매끄럽게 만들어 cO 문제를 해결합니다.

- **Performance Highlights**: 광범위한 실험 결과는 제안된 방법이 치명적인 과적합 문제를 극복하고 최첨단의(state-of-the-art) 성능을 달성할 수 있음을 보여줍니다. 1단계 적대적 훈련과 다단계(multi-step) 적대적 훈련 사이의 성능 격차를 효과적으로 축소함으로써, 이 연구는 희소 공격(sparse attacks)에 대한 효율적인 적대적 훈련의 새로운 최첨단 성능을 확립합니다.



### Reward Learning from Multiple Feedback Types (https://arxiv.org/abs/2502.21038)
Comments:
          Published as a conference paper at ICLR 2025

- **What's New**: 이 논문은 다양한 유형의 피드백을 사용하여 강화 학습(Reinforcement Learning, RL)에서의 보상을 학습하는 새로운 접근 방식을 소개합니다. 특히, 기존의 이진 비교(preference-based feedback) 방법 대신, 다양한 피드백 타입을 통해 더 효과적인 인간 피드백을 수집하는 방법을 제안합니다. 다양한 피드백 유형을 실험하고 평가하여 이러한 방식의 효과성을 입증한 첫 번째 연구로, 차별화된 피드백이 보상 모델링에 긍정적인 영향을 미칠 수 있다는 가능성을 보여줍니다.

- **Technical Details**: 본 연구에서는 여섯 가지 다른 유형의 피드백에 대한 고품질 시뮬레이션 피드백을 생성하는 프로세스를 개발하였습니다. 각 피드백 유형에 대해 보상 모델(reward models)과 후속 강화 학습 훈련을 실행하며, 이를 통해 10개의 다양한 RL 환경에서 피드백 유형을 비교하고 순수 이진 비교 기준선(preference-based baselines)과의 성능 차이를 분석합니다. 이를 통해 여러 피드백 타입을 동시에 활용할 수 있는 방법론을 제시합니다.

- **Performance Highlights**: 다양한 피드백 유형이 강화 학습에서 강력한 보상 모델링 성능을 이끌어낼 수 있음을 경험적으로 입증하였습니다. 이 연구는 다중 피드백 유형이 RLHF(Reinforcement Learning from Human Feedback)에서의 잠재력을 강하게 나타내는 것을 강조합니다. 결과적으로, 다양한 피드백의 동시 사용이 동일한 트레이닝 환경에서도 성능 개선을 가져올 수 있다는 점을 발견하였습니다.



### Synthesizing Tabular Data Using Selectivity Enhanced Generative Adversarial Networks (https://arxiv.org/abs/2502.21034)
Comments:
          This thesis submitted to the University of Melbourne for partial fulfillment of the degree of Master of Data Science

- **What's New**: 이 논문에서는 주요 쇼핑 이벤트(예: Black Friday) 동안 증가하는 거래량에 대비하기 위해 E-commerce 플랫폼의 리소스 계획을 위한 새로운 방법을 제안합니다. 이전 연구는 Generative Adversarial Networks (GAN)를 이용하여 데이터 생성에 초점을 맞추었지만, 이러한 방법은 E-commerce 스트레스 테스트에 적합하지 않은 계산 요구 사항을 간과했습니다. 본 연구는 쿼리 선택성 제약(query selectivity constraints)을 포함한 새로운 GAN 기반 접근 방식을 소개합니다.

- **Technical Details**: 논문에서는 사전 훈련된 딥 신경망(pre-trained deep neural network)을 통합하여 실제 데이터와 합성 데이터 간의 선택성 일관성(selectivity consistency)을 유지하는 방법을 제시합니다. 데이터 처리 과정에서 쿼리 선택성 제약은 데이터베이스 트랜잭션 처리의 중요한 요소로, 본 연구의 핵심적인 기술적 요소입니다. 본 방법론은 5개의 실세계 데이터셋에서 테스트되었으며, 선택성 추정 정확도(selectivity estimation accuracy)와 머신러닝 유틸리티(machine learning utility)를 향상시켰습니다.

- **Performance Highlights**: 본 연구의 방법은 세 가지 최첨단 GAN 모델과 VAE 모델을 초월하여 성능을 입증합니다. 선택성 추정 정확도는 최대 20% 향상되었으며, 머신러닝 유틸리티는 최대 6% 향상되었습니다. 이 결과는 E-commerce 플랫폼에서 실질적인 데이터 처리에 있어 더 나은 성능을 제공할 수 있음을 보여줍니다.



### Beyond Words: A Latent Memory Approach to Internal Reasoning in LLMs (https://arxiv.org/abs/2502.21030)
Comments:
          13 pages, 5 figures

- **What's New**: 최근 대형 언어 모델(LLMs)의 발전으로 인해 체인 오브 생각(Chain-of-Thought, CoT) 패러다임이 주목받고 있습니다. 이 접근법은 모델이 자연어로 명시적인 추론 단계를 생성하여 해석 가능성을 높이고 외부 감사를 용이하게 합니다. 그러나 저자는 인간의 인지가 완전한 언어화 없이도 과거의 감각 및 에피소드 정보를 회상하는 암묵적 정신적 표현에 의존한다는 점을 지적하며, 이를 LLM의 내부 추론 과정에 통합하는 프레임워크를 제안합니다.

- **Technical Details**: 제안된 구조는 암묵적 메모리 모듈(Implicit Memory Module, IMM)을 갖춘 LLM을 포함합니다. IMM은 처리 중 생성된 암묵적 표현을 저장하고 필요할 때 이를 검색하여 내부 추론을 지원합니다. 이 모델은 동적으로 정보를 처리하고, 특정 시간 단계에서 요약 표현을 메모리에 기록하며, 쿼리 벡터를 생성해 기억 저장소에서 정보를 검색합니다.

- **Performance Highlights**: 초기 실험 결과, 단순한 GPT 모델에 IMM을 통합함으로써 최종 훈련 손실이 기존 GPT 기준선보다 35%에서 57% 감소하는 효과가 있었습니다. 이러한 방법은 정밀한 해석 가능성 채널도 쉽게 추가할 수 있어, 효율적이면서도 더 강력한 추론을 가능하게 합니다. 따라서 이 연구는 미래의 명시적 감사 가능성을 위한 방향도 제시합니다.



### Measuring and identifying factors of individuals' trust in Large Language Models (https://arxiv.org/abs/2502.21028)
Comments:
          24 pages, 6 figures

- **What's New**: 이 논문에서는 인간과 대화하는 것처럼 보이는 대형 언어 모델(LLM)에 대한 신뢰 형성을 측정하기 위한 새로운 프레임워크인 Trust-In-LLMs Index(TILLMI)를 소개합니다. TILLMI는 McAllister의 인지적(cognitive) 및 정서적(affective) 신뢰 차원을 확장하여 개인의 LLM에 대한 신뢰를 측정합니다. 1,000명의 미국 응답자를 대상으로 한 검증 과정을 통해, TILLMI는 2개의 차원으로 구성된 6개의 항목으로 구성된 신뢰할 수 있는 도구로 개발되었습니다.

- **Technical Details**: TILLMI는 심리측정(psychometric) 척도로 개발되었으며, 탐색적 요인 분석(Exploratory Factor Analysis)은 2개의 요인 구조를 발견했습니다. 중복된 항목 2개가 제거된 끝에 최종적으로 6개 항목으로 구성되었습니다. 확인 요인 분석(Confirmatory Factor Analysis)은 강력한 모델 적합도를 보여주었으며, TILLMI의 두 가지 요인은 "LLM과의 친밀감" (정서적 차원)과 "LLM에 대한 의존" (인지적 차원)으로 해석되었습니다.

- **Performance Highlights**: TILLMI는 LLM 사용자와 비사용자 간의 신뢰 수준 차이를 발견했으며, 연령이 낮은 남성 사용자가 LLM과의 친밀감 및 의존감이 더 높은 것으로 나타났습니다. LLM을 직접 사용해본 경험이 없는 개인은 사용한 사람들과 비교해 낮은 신뢰를 보였습니다. 이 연구는 AI 기반의 언어적 소통에 대한 신뢰 측정을 위한 새로운 경험적 기초를 제공하며, 책임 있는 디자인과 인간-AI 간의 균형 잡힌 협업을 촉진하는 데 기여합니다.



### LesionLocator: Zero-Shot Universal Tumor Segmentation and Tracking in 3D Whole-Body Imaging (https://arxiv.org/abs/2502.20985)
Comments:
          Accepted at CVPR 2025

- **What's New**: 본 논문에서는 LesionLocator라는 프레임워크를 소개하며, 이는 3D 의료 이미징에서 제로샷(longitudinal) 병변 추적 및 분할을 위한 최초의 엔드 투 엔드 모델입니다. 이 모델은 23,262개의 주석이 달린 의료 스캔 데이터셋과 다양한 병변 유형을 포함한 합성(longitudinal) 데이터를 활용하여, 실제 의료 이미징 문제에 대한 일반화 능력을 크게 향상시킵니다. LesionLocator는 모든 기존의 프롬프트 가능 모델을 병변 분할에서 약 10 dice 포인트 정도 초과하는 성능을 보여주며, 인체 수준의 성능을 달성했습니다.

- **Technical Details**: LesionLocator는 세 가지 핵심 구성 요소를 갖춘 프롬프트 가능한 프레임워크입니다. 첫째, 사용자의 프롬프트에 기반하여 다양한 병변을 제로샷 방식으로 분할하고, 둘째, 연속 스캔에서 병변을 효율적으로 추적하기 위한 프롬프트 전파 기능을 제공합니다. 셋째, 시계열 촬영 데이터에 대한 강건성과 일반화를 높이기 위한 합성(longitudinal) 데이터셋도 포함되어 있습니다. 이러한 접근方式는 3D 의료 데이터의 복잡한 공간 구조를 캡처하는 데 중요한 역할을 합니다.

- **Performance Highlights**: LesionLocator는 six out-of-distribution zero-shot 병변 분할 작업에서 뛰어난 일반화 성능을 보이며, 다양한 병변 유형과 신체 부위를 커버합니다. 이 모델은 단일 시점 분할 방법을 초과하는 성능을 발휘하며, 다중 시점 데이터에서 볼륨 추적에 대한 새로운 기준을 설정합니다. 또한, 연구 촉진을 위해 합성(longitudinal) 데이터셋과 모델 가중치를 공개할 예정이며, 이는 병변 추적을 위한 최초의 오픈 소스 모델로 자리잡게 될 것입니다.



### UoR-NCL at SemEval-2025 Task 1: Using Generative LLMs and CLIP Models for Multilingual Multimodal Idiomaticity Representation (https://arxiv.org/abs/2502.20984)
- **What's New**: SemEval-2025 Task 1에서는 주어진 명사 합성이 가지는 이디엄적 의미에 따라 이미지를 순위화하는 것을 목표로 합니다. 이를 위해, generative large language models (LLMs) 및 다국어 CLIP 모델을 활용하여 이디엄적 합성어 표현을 개선하는 새로운 접근 방식을 제시합니다.

- **Technical Details**: 이 연구는 LLMs를 사용해 잠재적인 이디엄적 합성어에 대한 의미를 생성하여 의미 해석을 풍부하게 합니다. 생성된 의미는 multilingual CLIP 모델을 통해 인코딩되어 이미지 순위화에 사용됩니다. 이 과정에서 contrastive learning 및 data augmentation 기술이 적용되어 임베딩을 미세 조정합니다.

- **Performance Highlights**: 실험 결과, 이 방법을 통해 추출된 다중모달 표현이 원래의 명사 합성어를 기반으로 한 표현보다 더 우수한 성능을 보였습니다. 그러나 미세 조정이 없는 임베딩을 사용하는 것보다 미세 조정 접근법의 효과는 다소 떨어지는 것으로 나타났습니다.



### Improving Open-world Continual Learning under the Constraints of Scarce Labeled Data (https://arxiv.org/abs/2502.20974)
- **What's New**: 이 논문은 오픈 월드 지속 학습(Open-world Continual Learning, OWCL)에서 희소 레이블 데이터(scarce labeled data)로 발생하는 문제를 탐구하고, 특히 새로운 범주가 제한된 주석과 소량으로 등장하는 현실적인 상황에서의 오픈 월드 소수 샷 지속 학습(Open-world Few-shot Continual Learning, OFCL)을 제안합니다. OFCL은 이전 지식 유지 및 과적합(overfitting) 방지, 레이블 데이터가 제한된 상황에서의 컴팩트한 결정 경계(compact decision boundaries) 구성, 그리고 레이블을 학습한 후에 미지의 샘플의 지식을 업데이트하는 것을 포함하는 도전 과제가 있습니다.

- **Technical Details**: 제안된 OFCL 프레임워크는 세 가지 핵심 구성 요소로 이루어져 있습니다: (1) 인스턴스 기반 토큰 증강(Instance-wise Token Augmentation, ITA)은 샘플 표현을 추가 지식으로 풍부하게 구성합니다. (2) 마진 기반 오픈 경계(Margin-based Open Boundary, MOB)는 새로운 작업이 시간에 따라 나타나는 오픈 감지를 지원합니다. (3) 적응 지식 공간(Adaptive Knowledge Space, AKS)은 미지의 샘플에 지식을 부여하여 미지의 것을 알려진 것으로 업데이트하도록 돕습니다.

- **Performance Highlights**: 광범위한 실험 결과는 OFCL 프레임워크가 모든 기준 방법과 비교해 현저한 성능 향상을 보이며, 실제 중요성과 재현성을 강조합니다. OWCL 접근법, 소수 샷 증분 학습(few-shot incremental learning) 방법 및 오픈 감지 기준과 비교했을 때, 제안된 OFCL 프레임워크가 그 효율성과 내구성을 입증하고 있습니다.



### Fine-Grained Retrieval-Augmented Generation for Visual Question Answering (https://arxiv.org/abs/2502.20964)
- **What's New**: 이번 논문에서는 기존의 Visual Question Answering (VQA)에 외부 지식을 통합하여 정확성을 향상시킨 Knowledge-Based Visual Question Answering (KB-VQA) 접근법을 제안합니다. 특히, Retrieval-Augmented Generation (RAG) 기술을 활용하여 정보 검색과 생성을 조화롭게 결합하며, fine-grained retrieval 방식을 통해 기존의 시각적 정보 손실 문제를 해결하고자 합니다. 또한, Knowledge Unit(지식 단위)라는 새로운 개념을 도입하여 데이터베이스에서의 지식 검색을 보다 효율적으로 수행합니다.

- **Technical Details**: 제안된 Knowledge Unit Retrieval-Augmented Generation (KU-RAG) 방법은 MLLMs(다중 모달 대형 언어 모델)와 통합되어, 시각적 정보를 기반으로한 질문에 대한 답변을 생성하기 위해 구조화된 지식 단위를 활용합니다. 이 방법은 이미지와 텍스트 질문을 기반으로 지식을 검색하고, 이를 바탕으로 보다 정확하고 관련성 높은 답변을 생성하는 지식 수정 체계(Knowledge Correction Chain, KCC)와 연계됩니다. KU-RAG는 fine-grained retrieval을 통해 ultra-accurate knowledge mapping을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, KU-RAG 방법이 기존의 KB-VQA 방법들에 비해 최대 10%의 성능 향상을 이루었다는 것을 보여주었습니다. MLLM의 reasoning 능력을 강화하며, 복잡한 질문에 대해서도 더 정교한 답변을 제공할 수 있도록 합니다. 이러한 성과는 KB-VQA 벤치마크에서 확인되었으며, 연구 결과가 향후 연구나 실제 애플리케이션에 큰 기여를 할 것으로 기대됩니다.



### Retrieval Augmented Generation for Topic Modeling in Organizational Research: An Introduction with Empirical Demonstration (https://arxiv.org/abs/2502.20963)
Comments:
          30 pages, 4 figures

- **What's New**: 이번 논문은 텍스트 데이터 분석의 새로운 접근법인 Agentic Retrieval-Augmented Generation (Agentic RAG)을 소개합니다. 기존의 LLM 기반 주제 모델링 방법이 가지고 있는 데이터 전처리의 어려움, 해석 가능성, 신뢰성 문제를 해결하기 위한 자동화된 방법으로 자리잡고 있습니다. 이 방법은 검색, 생성, 에이전트 기반 학습의 세 가지 주요 요소를 통합하여 효과적인 주제 모델링을 가능하게 합니다.

- **Technical Details**: Agentic RAG는 세 가지 구성 요소로 이루어져 있습니다: 첫 번째는 검색(retrieval)으로, LLM의 사전 학습 지식 외부에서 데이터에 접근할 수 있는 자동화된 방법을 제공합니다. 두 번째는 생성(generation)으로, LLM의 텍스트 합성 능력을 활용하여 데이터를 생성합니다. 세 번째는 에이전트 기반 학습(agent-driven learning)으로, 검색과 쿼리 구성 과정을 반복적으로 개선하여 더 나은 성과를 도출합니다.

- **Performance Highlights**: 논문은 Agentic RAG가 이전에 Mu et al. (2024a)가 분석한 Twitter/X 데이터셋에 대한 재분석을 통해 방법론이 효율적이고 해석 가능하며 높은 신뢰성과 타당성을 달성했음을 보여줍니다. 이는 전통적인 기계 학습 접근법에 비해나 LLM 프롬프트 기반 주제 모델링과 비교할 때도 우수한 성능을 입증합니다. 결과적으로 Agentic RAG는 리더십, 관리 및 조직 연구에서 AI 기반 질적 연구의 강력하고 확장 가능하며 투명한 대안으로 자리잡고 있습니다.



### Concealed Adversarial attacks on neural networks for sequential data (https://arxiv.org/abs/2502.20948)
- **What's New**: 이번 연구에서는 다양한 시계열 모델에 대해 숨겨진 적대적 공격(concealed adversarial attack)을 제안합니다. 이 공격은 보다 사실적인 교란을 제공하여 인간이나 모델 탐지기(discriminator)에 의해 쉽게 감지되지 않도록 설계되었습니다. 공격의 강도를 높이기 위해 훈련된 탐지기를 통한 접근법도 도입되어 다양한 방어 기법에 대응할 수 있는 폭넓은 커버리지를 제공합니다.

- **Technical Details**: 제안된 공격 방법은 분류기(classifier)와 훈련된 탐지기 손실(discriminator loss)의 집합체를 최대화하는 방식으로 진행됩니다. 이번 연구는 4개의 다양한 아키텍처(모델)에 걸쳐 6개의 UCR 시계열 데이터셋을 활용하여 수행되며, 순환 신경망(RNN), 합성곱 신경망(CNN), 상태 공간(state-space) 및 변환기(transformer) 기반 모델을 포함합니다. 또한, 정확성과 탐지의 균형을 이룰 수 있도록 다양한 강도로 악성 샘플을 탐지하는 탐지기 훈련 절차도 제안합니다.

- **Performance Highlights**: 이 공격 기법은 기존의 방법들보다 훌륭한 성능을 발휘하며, 인간의 눈이나 도메인 특정 이상 탐지기(detection model)에서 잘 탐지되지 않는 것을 강조합니다. 다루어진 여러 데이터셋을 통해 이 공격 방식이 시계열 데이터의 적대적 공격에서 얼마나 중요한지를 보여줍니다. 마지막으로, 연구진은 코드와 데이터셋을 GitHub에 제공함으로써 재현 가능성과 후속 연구의 촉진을 도모하고 있습니다.



### Generative Uncertainty in Diffusion Models (https://arxiv.org/abs/2502.20946)
- **What's New**: 이 논문에서는 생성 모델의 생성적 불확실성(generative uncertainty)을 추정하기 위한 베이지안 프레임워크( Bayesian framework)를 제안합니다. 최근 확산 모델(diffusion models)은 생성 모델링에서 중요한 발전을 이끌어내었지만, 개별 샘플의 품질이 낮을 수 있는 문제를 해결하는 것이 어려웠습니다. 본 연구는 고차원의 샘플 공간에서 발생하는 문제를 해결하기 위해 새로운 의미론적 가능도(semantic likelihood)를 도입했습니다.

- **Technical Details**: 제안된 프레임워크는 대규모 확산 모델에 대해 실용적인 베이지안 추정을 가능하게 하며, 라플라스 근사(Laplace approximation)를 이용하여 모델의 예측을 평가합니다. 이 연구에서 소개된 의미론적 가능도는 이미지 인코더(pretrained image encoders)를 활용하여 잠재적( latent ) 의미 공간에서 변동성을 계산합니다. 이 접근법을 이용하여 생성적 불확실성을 효과적으로 검출할 수 있음을 보였습니다.

- **Performance Highlights**: 실험 결과, 제안된 생성적 불확실성이 기존의 불확실성 기반 방법들보다 우수한 성능을 보여주었으며, 낮은 품질의 샘플을 효과적으로 식별할 수 있음을 입증했습니다. 또한, 샘플링 중 베이지안 불확실성의 계산 비용을 최소화할 수 있는 간단하면서도 효과적인 기술을 제시하였습니다. 이 프레임워크는 확산 모델 외에도 다른 생성을 위한 관계모델(latent flow matching model)에 적용 가능하다는 점이 특히 주목할 만합니다.



### A Deep User Interface for Exploring LLaMa (https://arxiv.org/abs/2502.20938)
- **What's New**: 최근 대규모 언어 모델(LLMs)의 인기가 높아짐에 따라, 사용자와 모델 간의 상호작용을 향상시키는 도구의 필요성이 증가하고 있습니다. 이 논문에서는 핵심 하이퍼파라미터를 조정할 수 있는 시각적 분석 도구를 제시하고, 이를 통해 사용자가 LLM의 출력을 탐색하고 비교할 수 있는 기능을 제공합니다. 사용자 연구를 통해 도구의 효과성을 평가한 결과, 비주얼 디자인과 사용자 인터페이스의 편리함에 대한 긍정적인 피드백을 받았습니다.

- **Technical Details**: 이 논문에서 제안하는 시각적 분석 도구는 LLM의 하이퍼파라미터를 시각적으로 조정할 수 있도록 설계되었습니다. top-p, frequency 및 presence penalty와 같은 하이퍼파라미터를 통해 모델 출력을 탐색하고 비교할 수 있는 인터페이스를 개발하였습니다. 이러한 접근 방식은 모델의 작동 원리를 이해하고 효과적인 의사 결정을 지원하기 위해 사용됩니다.

- **Performance Highlights**: 사용자 피드백을 통해 도구의 시각적 디자인이 긍정적으로 평가되었으며, 특히 인터페이스 레이아웃과 탐색의 용이성에 대한 칭찬을 받았습니다. 이 도구는 사용자가 하이퍼파라미터의 영향을 비교하고 분석하는데 실질적인 통찰력을 제공하여, LLM의 신뢰성을 높이고 사용자 상호작용에 큰 기여를 합니다.



### WebFAQ: A Multilingual Collection of Natural Q&A Datasets for Dense Retrieva (https://arxiv.org/abs/2502.20936)
Comments:
          10 pages, 3 figures, 7 tables

- **What's New**: WebFAQ는 FAQ 스타일의 주석에서 파생된 대규모 오픈 도메인 질문 답변(QA) 데이터셋 컬렉션입니다. 총 96백만 개의 자연 질문-답변 쌍이 75개 언어로 구성되어 있으며, 이 중 47백만 개(49%)가 비영어 샘플입니다. WebFAQ는 20개의 단일 언어 검색 벤치마크의 기초로도 활용되며, 이렇게 다각화된 데이터는 다국어 밀집 검색 모델의 훈련과 평가를 위한 고품질 자원을 제공합니다.

- **Technical Details**: WebFAQ 데이터셋은 정제된 필터링 및 근사용 데이터 감지를 통해 신중하게 선별되어, 1120만 개 QA 쌍으로 구성된 20개의 단일 언어 검색 벤치마크를 포함합니다. 자동화된 데이터셋 생성에 대한 첨단 방법을 활용하여, QA 정렬된 이중 언어 말뭉치도 구축되었습니다. 또한 수집된 QA 쌍을 활용하여 in-domain 사전 훈련된 XLM-RoBERTa 모델을 미세 조정했으며, 이 모델은 언어 간 검색 데이터셋에도 일반화되는 성능 향상을 보여줍니다.

- **Performance Highlights**: WebFAQ 데이터는 밀집 검색 모델의 성능을 향상시키며, 이는 오픈 도메인 Q&A 검색에서 실질적인 개선을 나타냅니다. 특히, 미세 조정된 모델은 비영어 샘플을 포함한 다양한 다국어 검색 데이터셋에서도 주목할 만한 성능을 발휘합니다. 최종적으로, 자동화된 비텍스트 생성 방법을 통해 구축된 이중 언어 말뭉치는 비슷한 데이터셋에 비해 높은 번역 품질을 보인 것으로 확인되었습니다.



### Less is More? Revisiting the Importance of Frame Rate in Real-Time Zero-Shot Surgical Video Segmentation (https://arxiv.org/abs/2502.20934)
- **What's New**: 이 연구는 AI 보조 수술의 실시간 비디오 분할에 대한 새로운 통찰력을 제공합니다. SAM2 모델의 효과를 담낭절제술(cholecystectomy) 과정에서 다양한 초당 프레임 수(frame rate)에서 평가함으로써, 프레임 속도가 분할 성능에 미치는 영향을 분석하였습니다. 특히, 놀랍게도 전통적인 평가 환경에서는 1 FPS의 낮은 속도가 25 FPS를 초과하는 성능을 보일 수 있다는 것을 발견하였습니다.

- **Technical Details**: 연구에서는 SAM2.1 Hiera Large 모델을 실시간 수술 비디오 분할에 활용하였습니다. 이 모델은 강력한 transformer 기반 아키텍처를 사용하여 공간적 및 시간적 일관성을 보장하며, 복잡한 수술 비디오 분석에 매우 적합합니다. 성능 평가에는 CholecSeg8k 데이터셋이 사용되었으며, 1, 10, 15, 20, 25 FPS 설정에서 SAM2의 성능을 조사하였습니다.

- **Performance Highlights**: 결과적으로 25 FPS 설정이 실시간 스트리밍 환경에서 우수한 성능을 나타냈으며, 특히 수술용 그레이퍼와 같은 동적 객체에 대해 시간적 일관성과 안정성을 제공하는 것으로 나타났습니다. 또한, 설문조사를 통해 전문가들이 높은 FPS로 제공되는 분할 마스크를 선호한다는 사실이 확인되어, AI 보조 수술의 실시간 평가가 중요함을 재확인하였습니다.



### Everything, Everywhere, All at Once: Is Mechanistic Interpretability Identifiable? (https://arxiv.org/abs/2502.20914)
- **What's New**: 본 연구는 AI 시스템의 해석 가능성(interpretability)을 보장하는 기준으로서 기계적 해석 가능성(Mechanistic Interpretability, MI)의 역할을 조사합니다. 전통적인 통계에서의 식별성(identifiability) 문제를 AI 해석 가능성에 적용하여, 특정 행동에 대한 독특한 설명이 존재하는지를 탐색합니다. 이 연구는 두 가지 주요 MI 전략을 통해 다양한 설명 기법을 제시하고 검토합니다.

- **Technical Details**: MI는 복잡한 신경망의 행동을 단순한 알고리즘으로 역설명하고, 이를 통해 내부 계산을 추적하는 접근 방식을 취합니다. 본 논문에서는 '어디-그 다음 무엇(where-then-what)'과 '무엇-그 다음 어디(what-then-where)'의 두 가지 전략을 정의하며, 이를 통해 부울 함수와 다층 퍼셉트론(Multi-layer Perceptrons)에서 다양한 후보 설명을 평가합니다. 각각의 전략에서 최적의 서브셋과 causally aligned 알고리즘을 찾아내는 과정을 진행합니다.

- **Performance Highlights**: 실험 결과, 기계적 해석 가능성의 기준이 충분히 엄격하지 않음을 보여줍니다. 여러 회로가 동일한 모델 행동을 복제할 수 있으며, 하나의 회로에 대해 다수의 해석이 가능함을 발견했습니다. 최종적으로 이 연구는 AI의 설명 기준을 정의하는 데 기여하며, 개별 설명의 독창성이 이해에 필수적인지를 재조명합니다.



### DexGraspVLA: A Vision-Language-Action Framework Towards General Dexterous Grasping (https://arxiv.org/abs/2502.20900)
Comments:
          22 pages, 10 figures

- **What's New**: 이번 논문에서는 Dexterous grasping(정교한 잡기) 문제가 로봇공학에서 여전히 중요한 도전 과제라는 점을 강조합니다. 기존의 연구들은 일반적으로 단일 객체 환경이나 제한된 환경에 대한 가정을 바탕으로 하였는데, 이로 인해 일반화에 한계가 있었습니다. 저자들은 DexGraspVLA라는 계층적 프레임워크를 제안하며, 이는 사전 훈련된 Vision-Language 모델을 고수준 작업 계획자로 활용하고, 저수준 Action controller로서 diffusion 기반 정책을 학습합니다.

- **Technical Details**: DexGraspVLA는 다양한 언어 및 시각 입력을 반복적으로 변환하여 domain-invariant representations(도메인 불변 표현)으로 만드는 과정에서 주요 인사이트를 제공합니다. 이로 인해 imitation learning(모방 학습)을 효과적으로 적용할 수 있으며, 도메인 변화의 영향이 완화됩니다. 결과적으로, 다양한 실제 시나리오에 걸쳐 강력한 일반화 능력을 발휘할 수 있습니다. 이러한 구조는 상호 작용을 통해 더욱 향상된 성능을 제공합니다.

- **Performance Highlights**: 저자들의 방법은 ‘zero-shot’ 환경에서 수천 가지의 보지 못한 객체, 조명 및 배경 조합에 대해 90% 이상의 성공률을 기록합니다. 실험 분석을 통해 환경 변화에 따른 내부 모델 행동의 일관성이 확인되어, 그 설계의 유효성을 검증하고 일반화 성능을 설명합니다. 이 연구는 일반적인 정교한 잡기를 달성하는 데 한 걸음 더 나아가는 데 기여할 것으로 기대됩니다.



### A Fused Gromov-Wasserstein Approach to Subgraph Contrastive Learning (https://arxiv.org/abs/2502.20885)
- **What's New**: 본 논문에서는 FOSSIL(Fused Gromov Wasserstein Subgraph Contrastive Learning)이라는 새로운 방법을 제안합니다. 이 모델은 노드 수준(node-level)과 서브그래프 수준(subgraph-level)의 대조 학습(contrastive learning)을 통합하여, 표준 노드 수준 대조 손실과 Fused Gromov-Wasserstein 거리의 조합을 활용합니다. FOSSIL은 동질성(homophily) 및 이질성(heterophily) 그래프 모두에 잘 작동하며, 긍정적 및 부정적 쌍을 생성하기 위한 동적 뷰(view)를 만들 수 있는 장점을 가지고 있습니다.

- **Technical Details**: FOSSIL은 그래프 데이터셋의 동질성 수준에 대한 변동을 견딜 수 있는 강력한 아키텍처를 설계하였습니다. 이 모델은 노드의 특징과 그래프의 구조를 동시에 캡처하기 위해 Fused Gromov-Wasserstein 거리(FGWD)를 활용합니다. 또한, GNN 인코더는 이질성 데이터에 효과적으로 처리할 수 있도록 디커플링(decoupled) 방식으로 설계되었습니다. 이 방식을 통해 FOSSIL은 서브그래프의 구조적 및 특징적 특성을 동시에 인코딩하여 대조 손실을 계산할 수 있습니다.

- **Performance Highlights**: FOSSIL은 다수의 기준 그래프 데이터셋에서 광범위한 실험을 수행하였으며, 최신 방법들과 비교하여 우수한 성능을 보였습니다. 특히, 이 모델은 동질성 및 이질성 데이터 모두에서 일관된 성능을 발휘하여 실용성을 높입니다. 또한, 다양한 데이터셋에서 자가 감독 노드 분류(task)에서 FOSSIL의 효과를 입증하였습니다. 각 디자인 선택에 대한 철저한 분석을 통해 FOSSIL의 성능을 더욱 강화하였습니다.



### Oscillation-Reduced MXFP4 Training for Vision Transformers (https://arxiv.org/abs/2502.20853)
- **What's New**: 이 논문에서는 FP4 정밀도로 트랜스포머를 사전 훈련하는 새로운 방법인 TetraJet을 제안합니다. 저 precision 훈련은 큰 신경망의 학습을 가속화하는 유망한 기술로 부각되고 있으며, FP4 형식은 특히 속도 측면에서 강력한 잠재력을 가지고 있습니다. 그러나 FP4 훈련의 정확도 저하 문제를 해결하기 위해 MXFP4 데이터를 사용하면서 발생하는 문제에 대해 체계적으로 연구한 결과를 보여줍니다.

- **Technical Details**: TetraJet은 MXFP4 계산을 사용하여 트랜스포머의 앞으로 및 앞으로(Backward) 패스를 수행하는 새로운 훈련 방법론입니다. 이 방법은 모든 가중치/활성화/그래디언트 텐서를 MXFP4로 양자화하여 하드웨어의 가속화 잠재력을 최대한 활용합니다. 특히, 저자들은 weight oscillation 문제를 발견하고 이를 해결하기 위한 Q-EMA(EMA Quantizer) 및 Q-Ramping(Adaptive Ramping Optimizer) 방법을 제안하여 성능을 개선했습니다.

- **Performance Highlights**: TetraJet을 사용한 광범위한 실험 결과, 기존의 4비트 훈련 방법에 비해 항상 더 나은 성능을 나타내었습니다. 특히, Q-EMA 및 Q-Ramping은 진동을 효과적으로 줄여줌으로써 추가적인 향상을 가져왔고, 기반선(기존 방법) 대비 정확도 저하를 50% 이상 줄이는 데 성공했습니다. 결과적으로 TetraJet은 풀 정밀도 훈련(Full Precision Training)과 비교할 때 경쟁력 있는 성과를 달성했습니다.



### Reinforcement Learning with Curriculum-inspired Adaptive Direct Policy Guidance for Truck Dispatching (https://arxiv.org/abs/2502.20845)
- **What's New**: 이 논문에서는 공개광산에서 트럭 배치 문제를 해결하기 위해 새로운 curriculum learning 전략인 Curriculum-inspired Adaptive Direct Policy Guidance를 소개하고 있습니다. 이 방법은 Proximal Policy Optimization (PPO)을 적응시켜 불균일한 결정 간격을 처리하고, Shortest Processing Time teacher policy를 사용해 정책 탐색을 안내합니다. 이를 통해 기존 PPO 대비 10%의 성능 향상과 더 빠른 수렴성을 보여줍니다.

- **Technical Details**: 트럭 배치 알고리즘 문제는 OpenMines를 gym API를 통해 정의하였고, 다양한 환경 정보를 포함하여 모델 성능을 향상시켰습니다. 관찰 공간은 Order State, Truck Self State, Road States 및 Target States로 나뉘며, M=5, N=5, K=71로 설정되었습니다. 보상 함수 설계는 중요한 과제로, 밀집 보상과 희소 보상의 두 가지 형태를 정의하여 직접 정책 안내의 효과를 검증합니다.

- **Performance Highlights**: 제안된 방법은 여러 트럭이 동일한 에이전트로부터 배차를 요청하는 복합적인 환경에서도 우수한 성능을 나타냅니다. 기존의 최적화 기반 방법들이 직면한 동적 불확실성 문제를 해결하며, 보상 설계에 대한 탄력성을 보여줍니다. 이러한 전략은 RL 기반 트럭 배치 분야에서 일반적이고 효과적인 커리큘럼 학습 기법으로 자리매김할 것으로 기대됩니다.



### Neuro-Symbolic Learning for Galois Groups: Unveiling Probabilistic Trends in Polynomials (https://arxiv.org/abs/2502.20844)
- **What's New**: 이 논문은 다항식의 Galois 그룹을 분류하기 위해 신경 기호적(neurosymbolic) 접근 방식을 제시합니다. 고전적인 Galois 이론(classical Galois theory)과 머신 러닝(machine learning)을 결합하여 대수적 계산(algebraic computation)의 도전 과제를 해결하고 있습니다. 이 연구는 신경망(neural networks)과 기호적 추론(symbolic reasoning)을 함께 사용하여 순수 수치적 방법보다 더 높은 정확성과 해석 가능성을 제공하는 모델을 개발했습니다.

- **Technical Details**: 주요 초점은 높이(height) $ 	$ 이하인 육차(sextic) 다항식에 있으며, 53,972개의 비가환(irreducible) 예제를 포함한 데이터베이스를 분석합니다. 연구자들은 Galois 그룹이 $C_6$인 20개의 육차 다항식이 단지 7개의 불변 정의 등가 클래스(invariant-defined equivalence classes)를 포함하고 있다는 새로운 분포적 경향을 발견했습니다. 이러한 결과는 높이 제약 하에서 Galois 그룹 확률(Galois group probabilities)에 대한 최초의 경험적 통찰(empirical insights)을 제공하며, 기하학적 해법을 탐구하기 위한 기초를 마련합니다.

- **Performance Highlights**: 이 연구는 AI가 전통적인 기호적 기술로는 드러나지 않는 패턴을 발견하는 데 기여할 수 있는 잠재력을 보여줍니다. 연구 결과는 확률적 추측(probabilistic conjectures) 및 고차원 분류(higher degree classifications)와 같은 대수적 문제에 대한 미래 연구의 방향성을 제시합니다. 이는 대수적 계산(computational algebra) 분야에서 상당한 영향을 미칠 것으로 기대됩니다.



### Hierarchical and Modular Network on Non-prehensile Manipulation in General Environments (https://arxiv.org/abs/2502.20843)
Comments:
this http URL

- **What's New**: 이 논문은 가정과 같이 일반적인 환경에서 로봇이 비파악형 조작(Non-prehensile manipulation) 작업을 수행할 수 있는 능력을 향상시키기 위한 새로운 아키텍처를 제안합니다. 특히, 객체의 기하학적 다양성을 고려해 환경 제약 조건에 적응할 수 있는 모듈형 리컨피겨러블 아키텍처(Modular and reconfigurable architecture)를 도입합니다. 또한, 다양한 환경을 생성하기 위한 프로시저 알고리즘을 제안하여, 실제 객체와 환경에 대해 제로샷 전이(Zero-shot transfer)가 가능하도록 합니다.

- **Technical Details**: 제안된 정책 아키텍처는 HAMnet(Hierarchical and Modular Network)으로 불리며, 로봇이 현재 환경에 따라 모듈을 조절하여 다양한 조작 전략을 학습할 수 있게 합니다. 이 아키텍처는 모듈형 네트워크(modular network)로 구성되어 있어, 로봇의 기능을 빠르게 변화하는 환경에 적응시킬 수 있습니다. 또한, 새로운 UniCORN이란 기하학적 표현을 통한 컨택트의 가능성을 모델링하여, 객체와 환경 간의 상호 작용을 고려합니다.

- **Performance Highlights**: 이 연구에서는 비파악형 조작 정책이 다양한 환경 및 객체에 대해 일반화될 수 있음을 보여줍니다. 훈련은 전적으로 시뮬레이션 내에서 이루어지며, 실제 환경과 객체에 대해 제로샷 전이를 성공적으로 수행합니다. 또한, 9개의 디지털 트윈 환경과 353개의 객체로 구성된 벤치마크를 제공하여 비파악형 조작 연구를 지원합니다.



### Weakly Supervised Multiple Instance Learning for Whale Call Detection and Localization in Long-Duration Passive Acoustic Monitoring (https://arxiv.org/abs/2502.20838)
- **What's New**: 이번 연구에서는 DSMIL-LocNet이라는 새로운 다중 인스턴스 학습 프레임워크를 소개합니다. 이 모델은 고래 소리의 탐지와 위치 추정을 위해 단지 가방 수준의 라벨만을 사용하며, 2-30분의 긴 오디오 구간을 처리할 수 있도록 설계되었습니다. 기존의 분석 방법들은 전통적인 방법으로는 처리하기 어려운 대량의 PAM 데이터를 효율적으로 분석할 필요성이 커지고 있습니다.

- **Technical Details**: DSMIL-LocNet은 dual-stream 아키텍처를 채택하여 CNN 기반의 스펙트로그램 인코딩과 MLP 기반의 시간 특징 추출을 결합합니다. 이 모델은 classification 정확도, 시간 일관성, attention sparsity, 인스턴스 일관성을 균형 있게 조절하는 4가지 손실 함수를 활용합니다. 이를 통해 모델은 긴 오디오 구간에서도 고래 소리의 시간적 위치를 정확히 추정할 수 있게 됩니다.

- **Performance Highlights**: 테스트 결과, 남극 고래 데이터에서 분류 성능이 F1 점수 0.8-0.9에 도달하며, 중간 수준의 인스턴스가 локализация 정밀도를 0.65-0.70으로 보장하는 것으로 나타났습니다. 이러한 결과는 MIL이 해양 모니터링의 확장 가능성을 향상시킬 수 있음을 시사합니다. 또한, 고래 소리를 탐지하고 위치를 추정할 수 있는 새로운 방법론이 제시되어, PAM 데이터 분석에 큰 기여를 할 것으로 기대됩니다.



### LADs: Leveraging LLMs for AI-Driven DevOps (https://arxiv.org/abs/2502.20825)
Comments:
          17 pages with Appendix, 8 figures, and 7 tables. This paper is currently Under Review

- **What's New**: 최근 발표된 연구에서는 자동화된 클라우드 구성 및 배포의 복잡성을 극복하기 위한 LADS 프레임워크를 소개합니다. 이 프레임워크는 Large Language Models (LLMs)를 사용하여 클라우드 관리의 견고함과 효율성을 보장합니다. LADs는 기존 기술의 한계를 넘어, 최적화 방법을 심층 분석하여 적응성과 효율성을 제공합니다.

- **Technical Details**: LADS는 Retrieval-Augmented Generation, Few-Shot Learning, Chain-of-Thought 및 Feedback-Based Prompt Chaining을 활용하여 새로운 구성 설정을 생성하고 배포 실패로부터 학습하여 시스템 설정을 반복적으로 개선합니다. 이를 통해 DevOps 팀의 작업 부하를 줄이고, 리소스 활용성을 최적화하며, 시스템의 신뢰성을 향상시킵니다.

- **Performance Highlights**: 광범위한 평가를 통해 LADS는 수동 노력을 크게 줄이고, 리소스 사용을 최적화하며, 시스템의 전반적인 신뢰성을 증가시켰습니다. 이 연구 결과는 성능, 비용 및 확장성 간의 트레이드 오프(key insights)와 함께 다양한 배포 시나리오에 적합한 전략을 수립하는 데 도움을 줍니다.



### Multimodal Learning for Just-In-Time Software Defect Prediction in Autonomous Driving Systems (https://arxiv.org/abs/2502.20806)
Comments:
          9

- **What's New**: 이번 논문에서는 자율주행 소프트웨어 시스템에서의 신뢰성을 높이기 위한 새로운 접근법인 just-in-time 소프트웨어 결함 예측(JIT-SDP)을 제안합니다. 이 방법은 다중 모달(multi-modal) 학습을 활용하며, 코드 특성, 변경 지표(change metrics), 맥락 정보와 같은 여러 데이터 모달리티를 처리하는데 중점을 둡니다. 특히, 여러 데이터 모달 간의 attention 메커니즘을 활용하여 예측력을 극대화하는 방식입니다.

- **Technical Details**: 제안된 모델은 다중 모달 변환기(multimodal transformers)를 이용하여 사전 훈련된 변환기(pre-trained transformers)와 결합 모듈(combining module)을 통해 서로 다른 데이터 모달리티의 출력을 결합합니다. 텍스트 데이터 모델과 범주형 및 수치형 데이터가 포함된 테이블 형식의 출력을 완전 연결층(fully connected layers)을 통해 결합하여 예측을 생성합니다. 이를 통해 다양한 데이터 모달리티 간의 정보를 효과적으로 활용할 수 있습니다.

- **Performance Highlights**: 세 가지 오픈 소스 자율주행 시스템 소프트웨어 프로젝트(Apollo, Carla, Donkeycar)에 대한 실험 결과, 제안된 방법이 최신 딥 러닝(deep learning) 및 머신 러닝(machine learning) 모델을 명백히 능가함을 보여주었습니다. 연구 결과는 다중 모달 학습이 결함 예측을 개선함으로써 자율주행 소프트웨어의 신뢰성과 안전성을 높일 수 있는 잠재력을 지니고 있음을 강조합니다.



### Characteristics Analysis of Autonomous Vehicle Pre-crash Scenarios (https://arxiv.org/abs/2502.20789)
- **What's New**: 최근 자율주행 차량(Automated Vehicles, AV) 테스트에서 수많은 사고가 발생하고 있어 AV의 신뢰성 및 안전성 개선이 필요함을 보여주고 있습니다. 본 논문은 기존의 인간 운전 차량을 중심으로 한 사고 연구의 빈틈을 메우기 위해, 최신 캘리포니아 AV 충돌 보고서를 분석하고 새로운 프리-충돌 시나리오(Pre-crash scenario) 유형론을 적용하였습니다.

- **Technical Details**: 이 연구에서는 차량의 동역학(vehicle dynamics)과 운동학(kinematics) 특성을 기반으로 충돌 사고를 분류하는 프리-충돌 시나리오 유형론을 새롭게 제안하였습니다. 자동으로 이러한 시나리오를 추출할 수 있는 매핑 규칙(mapping rules)을 수립하였으며, 24가지 충돌 유형을 98.1%의 정확도로 식별하였습니다.

- **Performance Highlights**: 후미 충돌 시나리오에 대한 협동 분석에서는 환경적 영향 요소로 교통 통제 유형, 위치 유형, 조명 등을 확인하였습니다. 특히, 심각한 충돌이 발생할 가능성이 높은 교차로 상황에 대한 인과 분석을 통해, 습관적 위반 및 특정 행동에 대한 기대와 같은 중요한 원인 요인을 도출하였으며, 이는 AV 시스템 최적화를 위한 정부 감독과 제조업체의 개선 권장 사항을 도출하는 데 기여할 것입니다.



### Flattening Supply Chains: When do Technology Improvements lead to Disintermediation? (https://arxiv.org/abs/2502.20783)
- **What's New**: 이 연구는 디지털 경제에서 기술 혁신이 콘텐츠 생산과 소비에 미치는 영향을 탐구합니다. 특히, 생성적 AI 도구의 발전이 콘텐츠 제작자와 소비자 간의 중개자 역할을 어떻게 변화시킬 수 있는지를 분석했습니다. 우리는 기술 향상이 생산 비용을 줄이면 소비자들이 중개자를 우회할 가능성을 고려하여, 이로 인해 발생하는 복잡한 동태에 대한 인사이트를 제공합니다.

- **Technical Details**: 연구에서는 중개자와 생산 기술 공급자, 소비자가 연관된 게임 이론을 기반으로 하여, 기술 발전이 중개자를 대체(Disintermediation)시킬 때의 상황을 살펴봅니다. 결과적으로, 생산 비용이 너무 높거나 낮을 때 중개자에 의한 대체가 발생하며, 이러한 과정에서 사회적 복지(Social Welfare)와 콘텐츠 품질(Quality)에 미치는 영향을 분석합니다. 우리는 중개자가 시장에서 사회적 이익을 개선하는 역할을 하지만 그 이익을 모두 회수(Extract)한다는 점도 강조합니다.

- **Performance Highlights**: 결과적으로, 중개자는 지속적으로 시장에서 생존할 수 있으며, 특히 경쟁이 치열한 환경에서는 중개자의 존재가 콘텐츠 품질 향상에 긍정적인 영향을 미칠 수 있습니다. 또한 기술 발전이 있음에도 불구하고 중개자의 존재는 콘텐츠 품질에 대한 소비자의 반응을 약화시키는 경향이 있음을 보여줍니다. 최종적으로 연구에서는 이러한 동태가 디지털 콘텐츠 시장의 구조를 어떻게 형성하는지를 단적으로 보여줍니다.



### Triple Phase Transitions: Understanding the Learning Dynamics of Large Language Models from a Neuroscience Perspectiv (https://arxiv.org/abs/2502.20779)
Comments:
          46 pages

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 학습 과정에서 관찰되는 '상태 전이(phase transition)' 현상을 분석했습니다. 기존 연구들은 이러한 전이를 개별적 측면에서만 분석했으나, 본 연구는 인간의 뇌와의 유사성, LLM의 내부 상태, 그리고 하위 과제 성능 간의 상관관계를 통합적으로 분석하였습니다. 이로써 LLM의 학습 역동성을 새로운 관점에서 이해하고자 합니다.

- **Technical Details**: 연구에서는 OLMo-2, OLMo-0724, LLM-jp, Amber의 네 가지 사전 훈련된 모델을 사용하여 학습 동역학을 분석하였습니다. 각 모델은 서로 다른 데이터셋과 토크나이저를 활용하며, 파라미터 수는 6.74~7.3B 범위로 설정되어 있습니다. 연구 팀은 세 가지 분석 접근법을 통해 모델의 학습 과정에서 뇌 활동과의 정렬도를 조사하고, 여러 단계의 상관관계를 밝혀내었습니다.

- **Performance Highlights**: 주목할 만한 결과로, LLMs는 세 가지 주요 단계의 상태 전이를 겪습니다: (1) 작업 지침을 따르기 시작할 때 전체 뇌와의 정렬이 급증합니다. (2) 하위 과제 정확도가 일시적으로 정체될 때 LLM들이 뇌와 멀어지는 현상이 관찰됩니다. (3) LLM들이 하위 과제를 해결하는 능력을 갖추게 되면 다시 뇌와의 정렬이 나타납니다. 이러한 연구 결과는 LLM의 emergent capabilities가 훈련 과정 동안 어떻게 형성되고 통합되는지를 보여줍니다.



### Collective Reasoning Among LLMs A Framework for Answer Validation Without Ground Truth (https://arxiv.org/abs/2502.20758)
Comments:
          14 pages, 2 figures. arXiv admin note: substantial text overlap with arXiv:2411.16797

- **What's New**: 이번 연구는 여러 개의 대규모 언어 모델(GPT-4-0125-preview, Meta-LLaMA-3-70B-Instruct, Claude-3-Opus, Gemini-1.5-Flash)이 협력하여 복잡한 박사 수준의 확률 문제를 해결하는 새로운 프레임워크를 제시합니다. 이러한 협력이 응답의 신뢰성을 높이고, 생성된 질문의 품질을 평가하는 데도 도움을 줍니다. 연구진들은 카이제곱 테스트(chi-square tests), 플라이스 카파(Fleiss' Kappa), 신뢰구간(confidence interval) 분석과 같은 통계적 방법을 이용하여 여러 모델 간의 일치성과 응답의 정확성을 측정합니다.

- **Technical Details**: 연구에서는 Claude와 Gemini 모델이 잘 구조화되고 모호성이 적은 질문을 생성하는 경향이 있어, 이로 인해 높은 상호 모델 일치를 보였습니다. 반대로, LLaMA 모델은 질문 제시에서 변동성이 크고 신뢰성이 낮아 넓은 신뢰구간과 낮은 합의율로 나타났습니다. 이러한 결과는 다중 모델 협력이 응답의 신뢰성을 향상시킬 뿐 아니라 명확한 정답이 없는 상황에서도 질문 품질을 평가하고 개선하는 유용한 프레임워크를 제공함을 시사합니다.

- **Performance Highlights**: 본 연구는 다중 모델 협력을 통해 복잡한 지식을 검증하는 새로운 프레임워크를 제시하며, 협력 검증의 효과를 뒷받침하는 실증적 증거를 제공합니다. 또한 향후 LLM 기반 지식 검증 연구를 위한 기준도 마련하였습니다. 연구의 발견은 교육 기술, 자동화 평가 시스템 및 전문 학술 분야의 연구 검증에 걸쳐 광범위한 영향을 미칠 것으로 기대됩니다.



### Teach-to-Reason with Scoring: Self-Explainable Rationale-Driven Multi-Trait Essay Scoring (https://arxiv.org/abs/2502.20748)
- **What's New**: 이번 연구에서는 복합 특성 자동 에세이 평가(Automated Essay Scoring, AES)에서 생성된 점수에 대한 합리적인 설명을 제공하는 RaDME(자기 설명 가능한 합리적 다중 특성 자동 에세이 점수 프레임워크)를 제안합니다. 기존의 AES 시스템은 점수는 정확하게 예측하였지만, 그 점수의 이유를 설명하는 데 실패하여 투명성이 부족했습니다. RaDME는 대형 언어 모델(LLMs)의 추론 능력을 작은 모델로 증류하여 특성 점수와 그에 따른 설명을 순차적으로 생성하도록 설계되었습니다.

- **Technical Details**: RaDME는 LLM의 추론 능력을 활용하여 작지만 효과적인 점수 모델을 훈련시킵니다. 이 모델은 점수와 그에 대한 합리적 설명을 동시에 생성함으로써 점수 결정 과정이 명확한 이유에 기반하도록 유도합니다. 모델은 여러 특성 및 프롬프트에 걸쳐 점수-합리성 쌍을 다중 작업 학습(multi-task learning) 접근법으로 최적화하여 구성됩니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, RaDME는 뛰어난 점수 성능을 달성하며 최근의 최첨단 방법들보다 더 높은 성과를 보입니다. RaDME는 점수-이유 학습 초기 접근 방식을 통해 생성된 합리적 설명의 품질을 향상시킵니다. 이는 자동 평가의 투명성을 높이고, 더 해석 가능하고 신뢰할 수 있는 AES를 위한 기초를 마련하는 중요한 단계입니다.



### Structured Preference Optimization for Vision-Language Long-Horizon Task Planning (https://arxiv.org/abs/2502.20742)
Comments:
          18 pages

- **What's New**: 기존의 비전-언어 (vision-language) 태스크 플래닝 방법론은 짧은 수평 작업에서는 뛰어난 성능을 보이지만, 복잡하고 긴 수평 작업에서는 한계가 있습니다. 이를 해결하기 위해 Structured Preference Optimization (SPO)라는 새로운 접근법을 제안했습니다. SPO는 장기 작업 계획에 있어 추론 (reasoning) 및 행동 선택 (action selection)을 향상시키는 것을 목표로 합니다.

- **Technical Details**: SPO는 두 가지 핵심 개념을 도입합니다. 첫째, Preference-Based Scoring and Optimization은 태스크 관련성, 비주얼 그라운딩 (visual grounding), 역사적 일관성을 기반으로 추론 체인을 체계적으로 평가합니다. 둘째, Curriculum-Guided Training은 모델이 단순한 작업에서 복잡한 작업으로 점진적으로 적응하여 장기 시나리오에서의 일반화 능력을 향상시키고 추론의 견고성을 강화합니다.

- **Performance Highlights**: SPO는 VirtualHome과 Habitat 2.0에서 각각 +5.98% GCR 및 +4.68% SR, +3.30% GCR 및 +2.11% SR 개선을 이뤘습니다. 이는 이전의 방법을 능가하며 비전-언어 태스크 플래닝에서의 선호 기반 최적화의 효과성을 입증합니다. 실험 결과는 SPO가 장기 작업에서의 추론 품질 및 최종 결정의 정확성을 크게 향상시키는 것을 보여줍니다.



### NeuroMorse: A Temporally Structured Dataset For Neuromorphic Computing (https://arxiv.org/abs/2502.20729)
- **What's New**: 신경모사 공학(Neuromorphic engineering)은 뇌의 효율적인 처리 방식을 모방하여 컴퓨팅을 발전시키는 것을 목표로 합니다. 본 논문에서는 neuromorphic 학습 시스템을 벤치마킹하기 위한 새로운 데이터셋, NeuroMorse를 소개합니다. 이 데이터셋은 영어에서 가장 많이 사용되는 50개의 단어를 모스 부호(Morse code) 스파이크 시퀀스로 변환하여 시간적 구조를 가지도록 설계되었습니다.

- **Technical Details**: NeuroMorse 데이터셋은 모스 점(dot)과 대시(dash)를 위한 두 개의 입력 스파이크 채널만을 사용하여 복잡한 정보를 시간적 패턴을 통해 인코딩합니다. 이 벤치마크는 다양한 시간적 스케일에서의 피쳐 계층(feature hierarchy)을 포함하고 있어, neuromorphic 알고리즘이 입력 패턴을 공간적(spatial) 및 시간적(temporal) 계층으로 분해하는 능력을 테스트합니다. 또한, 선형 분류기(linear classifier)를 사용하여 훈련 세트를 분류하는 것이 어렵고, 기존 방법으로 테스트 세트에서 키워드를 식별하는 데도 어려움이 있음을 보여줍니다.

- **Performance Highlights**: NeuroMorse 데이터셋은 neuromorphic 시스템의 독특한 강점과 특성을 충분히 평가하지 못하는 일반적인 벤치마크의 격차를 해소하기 위해 설계되었습니다. 이 데이터셋은 기존 신경망 기술로는 처리하기 힘든 도전적인 데이터 세트를 제공합니다. NeuroMorse는 Zenodo에서 사용할 수 있으며, 관련 코드는 GitHub에 제공됩니다.



### SPD: Sync-Point Drop for efficient tensor parallelism of Large Language Models (https://arxiv.org/abs/2502.20727)
Comments:
          Preprint

- **What's New**: 본 논문에서는 Sync-Point Drop (SPD)라는 혁신적인 최적화 기법을 소개하여, Tensor Parallelism (TP)에서 통신 오버헤드를 줄이면서 레이턴시를 최소화하는 방법을 제시합니다. 특히, 모델의 인퍼런스를 위한 분산 환경에서 효율적인 사용할 수 있는 블록 설계를 제안하였습니다. SPD는 모델 정확도에 미치는 영향을 최소화하면서도 통신 병목 현상을 효과적으로 완화합니다.

- **Technical Details**: 우리는 SPD를 통해 통신을 선택적으로 제거하여 다수의 컴퓨팅 장치에서의 분산 추론 성능을 개선합니다. 또한, 통신 감수성에 따라 다른 SPD 전략을 각각의 블록에 적용하여 최적화를 합니다. 이는 정보 손실을 최소화하고, 통신 간섭 없이도(decoded block의 경우) 실행할 수 있도록 설계되었습니다.

- **Performance Highlights**: LLaMA2-70B 모델의 분산 인퍼런스 실험에서 SPD는 약 20%의 전반적인 인퍼런스 레이턴시 감소를 기록했으며, 정확도는 1% 미만으로 저하되었습니다. 제안된 방법은 다양한 모델 크기 및 데이터셋에서 효과적인 가능성을 보여주었습니다.



### Generating Clinically Realistic EHR Data via a Hierarchy- and Semantics-Guided Transformer (https://arxiv.org/abs/2502.20719)
- **What's New**: 이 논문에서는 전통적인 전자 건강 기록(EHR) 데이터 생성 방식의 한계를 극복하기 위해, 계층적 및 의미 정보를 활용한 새로운 프레임워크인 Hierarchy- and Semantics-Guided Transformer (HiSGT)를 제안합니다. 기존의 방법들이 EHR을 평면적인 의학 코드 시퀀스로 처리함으로써 발생하는 문제점을 해결하고자 하며, 특히 임상 코드의 계층적 구조와 그 세부적인 의미를 반영합니다.

- **Technical Details**: HiSGT는 임상 코드 간의 부모-자식 및 형제 관계를 인코딩하기 위해 계층적 그래프를 구성하고, 이를 기반으로 그래프 신경망(Graph Neural Network, GNN)을 사용하여 계층 인식 임베딩을 도출합니다. 이와 함께, 사전 훈련된 임상 언어 모델(예: ClinicalBERT)로부터 추출된 의미 임베딩과 융합하여, 진짜 EHR의 세밀한 임상 패턴을 정확하게 모델링하는 Transformer 기반 생성기를 개발합니다.

- **Performance Highlights**: MIMIC-III 및 MIMIC-IV 데이터셋에서의 광범위한 실험을 통해 HiSGT가 생성한 데이터가 실제 환자 기록과 통계적으로 우수한 일치를 이룬다는 점을 입증했습니다. 또한 HiSGT는 만성 질환의 분류와 같은 하류 애플리케이션에서 매우 강력한 성능을 지원하여, 데이터 품질 향상 및 개인 정보 보호 기능을 향상하는 데 기여합니다.



### WorldModelBench: Judging Video Generation Models As World Models (https://arxiv.org/abs/2502.20694)
- **What's New**: 본 논문에서는 비디오 생성 모델들의 세밀한 세계 모델링 능력을 평가하기 위해 'WorldModelBench'라는 새로운 벤치마크를 제안합니다. 기존의 평가 기준들은 비디오 품질에 초점을 맞추어 세계 모델링의 복잡성을 충분히 반영하지 못했습니다. 'WorldModelBench'는 물리적 규칙을 따르는지의 여부와 지침 준수 등의 차원을 평가하여 모델의 신뢰성을 높이고 있습니다.

- **Technical Details**: 'WorldModelBench'는 350개의 이미지 및 텍스트 조건 쌍으로 구성되며 7개의 응용 분야에서 56개의 다양한 하위 분야를 포함하고 있습니다. 이 벤치마크는 텍스트에서 비디오(T2V) 및 이미지에서 비디오(I2V) 모델을 평가할 수 있도록 설계되었습니다. 각 비디오의 평가 기준은 지침 준수와 향후 프레임 생성으로 나뉘어 있으며, 각각의 세부 항목은 더욱 세분화되었습니다.

- **Performance Highlights**: 제안된 모델 평가 방법을 통해, 'WorldModelBench'는 기존의 모델보다 8.6% 높은 평균 정확도로 비디오의 세계 모델링 위반을 예측할 수 있음을 보여주었습니다. 또한, 인간의 주석을 최대화하여 Fine-tuning한 판별자는 비디오 생성 모델의 세계 모델링 능력을 눈에 띄게 향상시키는 데 기여함을 입증하였습니다. 이 연구는 향후 로봇 공학, 자율 주행 등의 분야에서 보다 신뢰할 수 있는 비디오 생성 모델의 개발로 이어질 것입니다.



### Unleashing the Potential of Two-Tower Models: Diffusion-Based Cross-Interaction for Large-Scale Matching (https://arxiv.org/abs/2502.20687)
- **What's New**: 본 논문에서는 새로운 매칭 패러다임인 T2Diff를 제안합니다. 이는 정보 상호작용을 강조하는 생성적 교차 상호작용 분리 아키텍처로, 두 개의 타워 모델의 잠재력을 최대한 활용할 수 있습니다. T2Diff는 확산 모듈(diffusion module)을 사용하여 사용자 다음 긍정 의도를 재구성하고, 혼합 주의(mixed-attention) 메커니즘을 도입하여 사용자와 아이템 간의 상호작용을 원활하게 합니다.

- **Technical Details**: T2Diff는 기존의 두 타워 아키텍처의 한계를 극복하기 위해 설계되었습니다. 사용자의 긍정적 상호작용을 아이템 타워를 통해 재구성하기 위한 생성 방법을 채택합니다. 이 모델은 혼합 주의 모듈을 통해 사용자와 아이템 특징 간의 복합적인 상호작용을 개선하며, 사용자의 행동 시퀀스에서 시간 변화를 명확히 추출하여 재구성의 정확도를 높입니다.

- **Performance Highlights**: 실험 결과, T2Diff는 두 개의 현실 세계 데이터셋과 하나의 산업 데이터셋에서 기존 SOTA 두 타워 모델들을 크게 초월하는 성능을 보여주었습니다. 또한, T2Diff의 확산 접근법은 아이템 표현의 재구성에서 다른 생성 모델보다 우수한 성능을 나타냈습니다. 이를 통해 T2Diff는 높은 정확도와 낮은 지연 시간(rate latency)을 동시에 달성함을 입증하였습니다.



### JAM: Controllable and Responsible Text Generation via Causal Reasoning and Latent Vector Manipulation (https://arxiv.org/abs/2502.20684)
Comments:
          10 pages, 3 figures, and 6 tables

- **What's New**: 이 논문에서는 JAM (Just A Move)이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 LLM의 생성 프로세스를 제어하고 해석하기 위해 원인과 결과 분석을 통합하며, 이를 통해 보다 책임감 있게 텍스트를 생성할 수 있습니다. JAM은 기존의 CTG (Controllable Text Generation) 방법보다 최대 22% 높은 성능 향상을 보여주며, 모델의 원인과 결과성을 유지하는 동시에 출력의 질과 신뢰성을 보장합니다.

- **Technical Details**: JAM은 잠재 공간(latent space) 내에서의 원인과 결과 분석을 통해 효과적으로 LLM의 생성 과정을 제어하는 접근법입니다. 잠재 벡터(latent vector)는 LLM 아키텍처의 기본 요소로, 모델이 텍스트 생성을 수행하는 데 필요한 정보를 인코딩합니다. 이 연구에서는 LLM 생성 과정에서의 잠재 벡터의 행동과 내재된 원인 관계를 조사하고, 이러한 관계를 분석하여 더 정교한 텍스트 생성을 위한 방법론을 제안합니다.

- **Performance Highlights**: JAM은 HHH (Helpful, Honest, and Harmless) 기준 및 독성 감소 작업에서 다양한 인기 LLM 모델에서 실험을 수행하여 성능을 평가했습니다. JAM은 기존 방법에 비해 여러 지표에서 최대 10% 개선된 점수를 보여주었으며, GPT-4 평가에서도 대부분의 경우 우세한 결과를 보였습니다. 이는 JAM이 책임감 있고 현실적인 텍스트 생성을 위한 효과적이고 효율적인 방법임을 강조합니다.



### Fine-tuning BERT with Bidirectional LSTM for Fine-grained Movie Reviews Sentiment Analysis (https://arxiv.org/abs/2502.20682)
Comments:
          14 pages, 5 figures, published in International Journal On Advances in Systems and Measurements, volume 16, numbers 3 and 4, 2023

- **What's New**: 이번 연구는 Bidirectional Encoder Representations from Transformers (BERT)와 Bidirectional LSTM (BiLSTM)을 결합하여 영화 리뷰에 대한 이진 정서 분석 (Sentiment Analysis, SA)을 향상시키고자 합니다. 기존의 문헌에서 주로 이진 분류 또는 세부적인 정서 분류에 집중해왔던 결과, 이 두 가지를 동시에 다루는 연구가 부족했습니다. 본 연구는 BERT 모델을 세밀하게 조정하는 방법론을 제공하면서 이러한 격차를 해소하고자 합니다. 특히, 영화 리뷰 영역에서의 SA의 중요성을 강조하며 해당 분야의 데이터셋을 활용하였습니다.

- **Technical Details**: 연구에서 사용한 BERT 모델은 복잡한 정서 분석을 위해 BiLSTM과 결합되었습니다. 이 모델은 전후 문맥을 모두 이해함으로써 정서의 복잡성을 보다 효과적으로 처리할 수 있습니다. 저자들은 이진 및 다중 클래스 정서 분류를 수행하고, 실제로 Synthetic Minority Oversampling Technique (SMOTE)와 NLP Augmenter (NLPAUG)와 같은 기술들을 사용하여 모델 성능을 개선했습니다. 전체 감정 극성을 구하기 위해 새로운 휴리스틱 알고리즘을 도입하였고, 이를 통해 다양한 정서 분류 작업에 적용하였습니다.

- **Performance Highlights**: 모델의 성능은 유명한 IMDb 데이터셋에서 이진 분류 정확도 97.67%를 기록하였고, 이는 기존 SOTA 모델을 0.27% 초과하는 값입니다. 또한, SST-5 데이터셋의 다중 클래스 분류에서 59.48%의 정확도를 달성하여 기존 BERT-large 기준을 3.6% 초과했습니다. 이러한 결과는 BERT와 BiLSTM을 결합한 접근 방식이 각종 평가에서 경쟁력 있는 성능을 발휘할 수 있음을 증명합니다.



### Disentangling Feature Structure: A Mathematically Provable Two-Stage Training Dynamics in Transformers (https://arxiv.org/abs/2502.20681)
- **What's New**: 이번 연구는 Transformers의 두 단계 학습 동역학(two-stage training dynamics)에 대한 이론적 분석을 제공합니다. 기존의 이론 분석은 이러한 현상을 거의 고려하지 않았으나, 이 논문에서는 GPT-2가 Counterfact 데이터셋에서 훈련되는 사례를 통해 시각적으로 이 과정을 보여줍니다. 이 논문의 주안점은 주어진 특성 구조를 분해(disentangled)하여 변별력을 분석하는 데 있습니다.

- **Technical Details**: 저자들은 컨텍스트 학습(in-context learning) 규칙 하에서 특성 학습(feature learning) 기법을 활용하여 Transformers의 동역학을 분석합니다. 이 과정에서 두 가지 유형의 특성 구조를 가지는 분리된 구조가 일반적이라는 점을 강조합니다. 예를 들어, 자연어에는 구문(syntax)과 의미(semantics)가 포함되며, 단백질(proteins)의 경우 1차 및 2차 구조가 존재합니다.

- **Performance Highlights**: 이 연구는 Transformers의 두 단계 최적화 과정에 대한 최초의 엄밀한 결과를 제시합니다. 또한, 이 과정이 attention weights의 스펙트럼 속성과 밀접하게 관련되어 있다는 보조 정리가 제시되어, 실험적으로 발견된 결과와 일치함을 보여줍니다.



### OpenEarthSensing: Large-Scale Fine-Grained Benchmark for Open-World Remote Sensing (https://arxiv.org/abs/2502.20668)
- **What's New**: 본 논문에서는 OpenEarthSensing라는 대규모의 세밀한 벤치마크를 제안합니다. 이 벤치마크는 5개의 도메인과 3개의 모달리티를 포함하여 189개의 카테고리와 총 157,674개의 이미지를 포함합니다. OpenEarthSensing은 실제 세상에서 발생할 수 있는 의미적 변화를 다룰 수 있도록 설계되었습니다.

- **Technical Details**: OpenEarthSensing는 RGB 위성 이미지, RGB 드론 공중 이미지, MSRGB, 적외선 등의 다양한 데이터 도메인을 보유하고 있어, 서로 다른 공변량 변화(covariate shifts) 조건을 갖춘 데이터 세트를 제공합니다. 이를 통해 모델의 일반화 성능을 평가할 수 있는 종합적인 테스트베드를 제공합니다.

- **Performance Highlights**: 기존의 열린 세계(open-world) 원거리 감지 기술과 방법을 기준으로 평가를 수행하여, OpenEarthSensing이 지속적으로 진화하는 환경에서의 모델 성능을 시험하는 데 있어 도전적인 벤치마크 역할을 한다는 것을 입증합니다. 이 논문에서 제안하는 벤치마크는 연구 및 개발을 촉진할 것으로 기대됩니다.



### Advancing AI-Powered Medical Image Synthesis: Insights from MedVQA-GI Challenge Using CLIP, Fine-Tuned Stable Diffusion, and Dream-Booth + LoRA (https://arxiv.org/abs/2502.20667)
- **What's New**: MEDVQA-GI 챌린지는 의료 진단에서 AI 기반 text-to-image 생성 모델의 통합을 다룹니다. 기존 방법은 정적인 이미지 분석에 주로 초점을 맞추고 있으며, 텍스트 설명에서 동적으로 의료 이미지를 생성하는 데는 부족하고, 이에 대한 새로운 접근 방식을 제안하고 있습니다. 본 연구는 Fine-tuned Generative Models을 사용하여 동적이고 확장 가능하며 정밀한 이미지를 생성하는 방법을 제시합니다.

- **Technical Details**: 이 시스템은 Fine-tuned Stable Diffusion, DreamBooth 모델 및 Low-Rank Adaptation (LORA)을 통합하여 고품질 의료 이미지를 생성합니다. 연구의 두 가지 주요 태스크는 Image Synthesis (IS)와 Optimal Prompt Generation (OPG)으로, 전자는 언어 프롬프트를 통해 이미지를 생성하고 후자는 주어진 카테고리 내에서 고품질 이미지를 생성하는 프롬프트를 제공합니다. 기존 의료 이미지 생성 방법의 한계에 대해 강조하며, Stable Diffusion이 이미지 품질 면에서 다른 모델들보다 우수함을 보여줍니다.

- **Performance Highlights**: Stable Diffusion은 Fréchet Inception Distance (FID) 스코어가 가장 낮았으며, 평균 Inception Score도 가장 높아(2.327) 뛰어난 다양성과 품질을 나타냅니다. 이는 AI 기반 의료 진단 분야의 발전을 의미하며, 향후 연구는 모델 개선, 데이터 세트 증대 및 윤리적 고려 사항에 중점을 두어야 할 것입니다.



### Dataset Distillation with Neural Characteristic Function: A Minmax Perspectiv (https://arxiv.org/abs/2502.20653)
Comments:
          Accepted by CVPR 2025, 11 pages, 7 figures

- **What's New**: 이 논문에서는 데이터셋 증류(dataset distillation)의 새로운 접근 방식을 제안하여, Neural Characteristic Function Discrepancy (NCFD)라는 새로운 지표를 도입합니다. 기존의 거리 측정 방법들이 분포의 차이를 정확하게 구별하지 못하는 문제를 해결하기 위해, NCFD는 특성 함수를 활용하여 전체 분포 정보를 포괄적으로 캡슐화하도록 설계되었습니다. 이러한 방법은 깊은 신경망의 학습 과정에서 생성 데이터와 실제 데이터 간의 차이를 최소화하는 동시에 정확도 향상을 도모합니다.

- **Technical Details**: NCFD는 특성 함수를 기반으로 하는 매개변수화된 지표로, 확률 밀도 함수의 푸리에 변환으로 정의됩니다. 이 접근 방식은 고차원 매니폴드에서의 시맨틱 구조를 포착하는데 효과적이며, 분포 매칭을 적대적인 미니맥스 최적화 문제로 재구성하여, 실시간으로 데이터 간의 불일치를 극대화합니다. 결과적으로, NCFD는 기존 최대 평균 불일치(maximum mean discrepancy, MMD) 방법보다 훨씬 더 높은 성능과 효율성을 보여줍니다.

- **Performance Highlights**: 실험 결과, NCFM 방법은 최신 기법들(SOTA)과 비교하여 ImageSquawk 데이터셋에서 20.5%의 정확도 향상을 달성했습니다. 또한 GPU 메모리 사용량을 300배 이상 줄이고, 처리 속도를 20배 향상시키는 성과를 올렸습니다. 특히, CIFAR-100 데이터셋에 대해 단일 NVIDIA 2080 Ti GPU에서 2.3GB 메모리만으로 무손실 압축을 달성하여 주목받고 있습니다.



### Consistency Evaluation of News Article Summaries Generated by Large (and Small) Language Models (https://arxiv.org/abs/2502.20647)
Comments:
          21 pages, 6 figures, 4 tables

- **What's New**: 이번 논문은 다양한 텍스트 요약 기법을 탐구하고 있습니다. 기존의 전통적인 평가 지표인 ROUGE 점수 및 BERT 점수를 사용하여 생성된 요약을 평가하며, LLM(Large Language Model) 기반의 평가 방법도 도입하고 있습니다. 특히, 생성된 요약의 일관성을 측정하기 위한 새로운 메타 평가 점수를 소개하고 있습니다. XL-Sum 데이터셋을 사용하여 여러 요약 모델의 성능을 비교한 결과, 모든 요약 모델은 참조 요약보다 일관성을 나타냈습니다.

- **Technical Details**: 논문은 요약 방법을 추출적(extractive) 및 추상적(abstractive) 접근 방식으로 나누어 다루고 있습니다. XL-Sum 데이터셋은 BBC 웹사이트에서 수집한 100만 쌍의 뉴스 기사와 요약을 포함하고 있으며, 본 작업에서는 영어 데이터만을 사용합니다. 해당 데이터셋에서 37.37%의 수동 검토된 영어 요약이 사실과 상충하는 정보가 포함되어 있음을 지적하면서, 이러한 문제의 최소화를 위해 기사 길이에 대한 필터링을 진행했습니다.

- **Performance Highlights**: 요약 방법으로는 TextRank, BART, Mistral-7B-Instruct, OpenAI GPT-3.5-Turbo 등의 모델이 포함되어 있습니다. 논문은 LLM의 발전에도 불구하고, 소규모 모델들이 여전히 비용 효율적인 솔루션으로 빛날 수 있는 여지를 강조하고 있습니다. 이에 따라 대규모 LLM들이 보유하지 못하는 다양한 성능을 가진 소형 모델들과의 비교가 이루어졌습니다.



### FedConv: A Learning-on-Model Paradigm for Heterogeneous Federated Clients (https://arxiv.org/abs/2502.20639)
- **What's New**: 이 논문에서는 사용자가 자원 제약이 있는 클라이언트에 대해 맞춤형 서브 모델을 제공하는 클라이언트 친화적 연합 학습(FL) 프레임워크인 FedConv를 제안합니다. FedConv는 통상적인 FL이 동질적인 대규모 글로벌 모델을 모든 클라이언트에 전송하고 훈련시키는 문제를 해결하여, 자원 부족 클라이언트의 부담을 최소화합니다. 이전의 기술들과는 달리, FedConv는 합성곱 압축(convolutional compression)을 통해 이질적인 서브 모델의 파라미터를 학습합니다.

- **Technical Details**: FedConv는 클라이언트가 로컬 훈련 후 모델 파라미터만을 서버에 업로드하고, 서버는 모델 집합을 조정하여 글로벌 모델을 업데이트 하는 전통적인 FL 시스템에 기반합니다. 이 프레임워크는 클라이언트가 각자의 자원 예산에 맞게 다양한 서브 모델의 파라미터를 학습할 수 있도록 합성곱을 이용하여 글로벌 모델을 압축합니다. 서브 모델은 클라이언트에서 직접 훈련 가능하며, 이 과정에서 클라이언트에 대한 추가 부담 없이도 사용자 데이터의 개인화된 정보를 유지합니다.

- **Performance Highlights**: 실험 결과, FedConv는 여섯 개의 공용 데이터 세트에서 기존 최첨단 FL 시스템보다 평균 35% 이상의 모델 정확도를 기록하며, 계산 및 통신 오버헤드는 각각 33% 및 25% 감소하였습니다. 이러한 성과는 FedConv가 자원 제약이 있는 클라이언트를 위한 연합 학습의 새로운 가능성을 제시하며, 전체 훈련 시간을 절감하는 데 기여합니다. FedConv는 글로벌 모델의 중요 정보를 효과적으로 유지하면서도 압축하는 최초의 방법으로, FL 시스템에서 비대칭 자원 활용 문제를 심각하게 개선하였습니다.



### A Compact Model for Large-Scale Time Series Forecasting (https://arxiv.org/abs/2502.20634)
- **What's New**: UltraSTF 모델이 소개되었으며, 이 모델은 스페이셜-템포럴(spatio-temporal) 데이터에 최적화된 접근 방식을 제공합니다. 기존의 SparseTSF 모델의 한계를 극복하기 위해 초-컴팩트(shape bank) 구성 요소를 추가하였습니다. 이 모델은 주기성을 활용하고 intra-period(기간 내) 시간 의존성을 효과적으로 포착하여 예측 성능을 개선합니다.

- **Technical Details**: UltraSTF는 교차 주기 예측 모듈과 초-컴팩트 형태의 뱅크(shape bank) 요소를 통합하여 설계되었습니다. 이 모델은 형태 뱅크의 주의(attention) 메커니즘을 통해 시간 시계열의 반복 패턴을 탐지하고 학습할 수 있습니다. 이러한 특성으로 인해 UltraSTF는 고차원 데이터에서도 높은 성능을 발휘할 수 있습니다.

- **Performance Highlights**: UltraSTF는 LargeST 벤치마크에서 최첨단 성능을 달성했습니다. 이 모델은 두 번째로 우수한 방법들이 요구하는 것의 0.2%도 안 되는 파라미터(parameter)만으로 이러한 성능을 나타내어 기존 방법들의 Pareto 경계를 더욱 확장했습니다.



### Lattice Protein Folding with Variational Annealing (https://arxiv.org/abs/2502.20632)
Comments:
          Github respository will be provided soon

- **What's New**: 본 논문에서는 2차원 Hydrophobic-Polar(HP) 격자 단백질 접힘 문제에서 최저 에너지 폴드를 식별하기 위해 마스킹을 사용하는 새로운 상한(training scheme) 훈련 방식을 소개합니다. 이 접근법은 Dilated Recurrent Neural Networks(RNN)와 온도 변동에 의해 구동되는 소거 프로세스를 통합하여 최적의 폴드를 예측합니다. 60개의 비드(beads)까지의 벤치마크 시스템에 대해 정확한 예측이 가능하며, 이를 통해 기존 머신러닝 기법의 잠재력을 강조합니다.

- **Technical Details**: 단백질 접힘은 아미노산의 선형 서열이 3차원 구조를 갖는 생물학적 과정입니다. 본 연구에서는 2D 격자에서 접힘의 복잡성을 줄이기 위해 HP 모델을 사용하여 단백질 접힘을 연구합니다. Dilated RNN을 온도 소거과정과 결합함으로써 RNN의 훈련 안정성을 높이고, 샘플링 과정에서 유효한 폴드를 마스킹하여 잘못된 샘플링 부담을 없애는 방식입니다.

- **Performance Highlights**: 본 방법은 최적의 폴드를 찾아내는 데 있어 다른 기존의 머신러닝 방법보다 경쟁력 있는 결과를 보입니다. 60 비드까지의 낮은 에너지를 가진 폴드를 효과적으로 찾을 수 있으며, RNN의 자기회귀적 샘플링 속성을 유지하게 됩니다. 실제 실험을 통해 소거 및 상한 훈련의 효과를 입증하며, 3차원으로 일반화할 수 있는 가능성을 제시합니다.



### Subtask-Aware Visual Reward Learning from Segmented Demonstrations (https://arxiv.org/abs/2502.20630)
Comments:
          Project webpage: this https URL

- **What's New**: 이번 논문은 REDS(REward learning from Demonstration with Segmentations)라는 새로운 보상 학습 프레임워크를 소개합니다. 이는 최소한의 감독으로 행동 없는 비디오를 활용하여 로봇 조작 작업에서의 보상을 학습합니다. REDS는 비디오 시연을 세부 작업으로 나누어 각 세그먼트를 실제 보상으로 간주하여 훈련합니다.

- **Technical Details**: REDS는 비디오 세그먼트를 기반으로 한 밀집 보상 함수(dense reward function)를 훈련하며, 이러한 과정에서 Equivalent-Policy Invariant Comparison (EPIC) 거리를 최소화하는 방식으로 실제 보상 신호와의 정렬을 보장합니다. 또한, 대조 학습(constrastive learning) 목표를 사용하여 비디오 표현과 세부 작업을 정렬함으로써 실시간 상호작용 중에 세부 작업을 정확하게 추론합니다.

- **Performance Highlights**: REDS는 Meta-World에서 복잡한 로봇 조작 작업을 수행하는 데 있어 기본 방법들보다 현저하게 개선된 성능을 보여줍니다. 특히, 가구 조립과 같은 복잡한 장기 작업에서 최소한의 인간 개입으로 학습할 수 있는 가능성을 보여줍니다. 또한, REDS는 이전에 보지 못한 작업 및 로봇 구조에 대한 일반화 능력을 지니고 있습니다.



### Continuous Adversarial Text Representation Learning for Affective Recognition (https://arxiv.org/abs/2502.20613)
Comments:
          6 pages, 3 figures, The 7th International Conference on Artificial Intelligence in Information and Communication (ICAIIC 2025)

- **What's New**: 본 논문에서는 감정 인식을 위한 새로운 프레임워크인 Continuous Adversarial Representation Learning (CARL)을 제안합니다. 기존의 언어 모델의 한계를 극복하고 감정 인식 태스크에서 효과적인 성능 향상을 목표로 하고 있습니다. 이 접근법은 연속적인 가치-각성(valence-arousal) 레이블링 시스템을 통해 대조적 학습을 가이드하며, 정서적 뉘앙스를 더 미세하게 포착할 수 있게 해줍니다.

- **Technical Details**: CARL은 문장 수준의 Momentum Continuous Contrastive Learning (MCCL)과 토큰 수준의 Gradient-based Perturbed Token Detection (PTD)을 결합한 두 가지 사전 훈련 작업으로 구성됩니다. 이러한 구조는 모델의 감정 임베딩 능력을 강화하는 데 중점을 두며, 목표 네트워크와 온라인 네트워크를 활용해 학습의 불안정성을 줄이고 더 견고한 표현을 가능하게 합니다. 표준적인 대조적 학습 기법과 달리, 이 방법은 Russell의 Circumplex 모델을 기반으로 감정을 두 차원 공간으로 표현하여 정서적 변이를 파악할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, 제안된 CARL 프레임워크는 감정 분류 벤치마크에서 최대 15.5%의 성능 향상을 달성하며, 연속 레이블의 중요성을 강조합니다. CARL은 세 가지 주요 감정 인식 태스크에서도 기존 방법들에 비해 우수한 성능을 보이며, 감정 표현 공간의 품질을 향상시키는 것을 입증합니다. 이러한 성과는 감정 표현 학습에서의 효과성을 더욱 증명하며, 정서적 이해의 정확성과 맥락적 관련성을 향상시키는 데 기여하고 있습니다.



### Leveraging Large Language Models for Building Interpretable Rule-Based Data-to-Text Systems (https://arxiv.org/abs/2502.20609)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLM)을 활용하여 완전히 해석 가능한 규칙 기반 데이터-텍스트 시스템을 순수 Python으로 자동 구현하는 간단한 접근 방식을 소개합니다. WebNLG 데이터셋에 대한 실험 평가 결과, 이 시스템은 직접 출력을 생성하기 위해 LLM을 사용했을 때보다 더 나은 품질의 텍스트를 생산하며, 동일한 데이터로 미세 조정된 BART 언어 모델보다 환각(hallucination)이 적게 발생합니다. 또한, 이 접근 방식은 신경망(neural) 접근 방식이 요구하는 처리 시간의 일부만으로 텍스트를 거의 즉각적으로 생성할 수 있습니다.

- **Technical Details**: 데이터-텍스트(data-to-text) 시스템의 개발에서 두 가지 주요 접근 방식이 있습니다: 규칙 기반(rule-based) 방법과 신경망(neural) 방법입니다. 규칙 기반 방법은 사전에 정의된 템플릿(template)과 언어 규칙을 사용하여 구조화된 데이터를 텍스트로 변환하는 반면, 신경망 방법은 딥 러닝 모델을 활용하여 데이터에서 텍스트로의 맵핑(mapping)을 자동으로 학습합니다. 본 논문에서는 LLM을 이용하여 규칙 기반 시스템을 구현하는 훈련 과정을 제안하며, 훈련 세트의 예제에 대해 LLM이 간단한 Python 코드를 작성하도록 요청하고, 이를 통해 최종적으로 입력 데이터에 대한 텍스트화(textualisation)를 생성할 수 있는 단일 Python 코드 파일을 생성합니다.

- **Performance Highlights**: WebNLG 데이터셋을 기반으로 실험을 수행한 결과, 자동 작성된 규칙 기반 시스템은 미세 조정된 신경망 모델의 BLEU 또는 BLEURT 점수를 달성하지는 못했지만, 환각 발생이 현저히 적으며 비트리비얼(non-trivial) 신경망 기준 모델을 능가하는 성과를 보였습니다. 또한, 제작된 시스템은 완전히 해석 가능하며, 필요 시 Python 프로그래머가 수정할 수 있는 높은 제어 가능성을 제공합니다. 이를 통해 시스템은 GPU 없이 단일 CPU에서 거의 즉시 텍스트를 생성하는 효율성을 보여주었습니다.



### Exploring the Impact of Temperature Scaling in Softmax for Classification and Adversarial Robustness (https://arxiv.org/abs/2502.20604)
- **What's New**: 이 연구는 deep learning에서 softmax 함수의 '온도'(temperature) 매개변수의 중요성을 강조합니다. 저자들은 온도 스케일링이 이미지 분류에서 모델의 학습 및 최적화 성능에 미치는 영향을 실증적 및 이론적으로 탐구하였고, 중간 온도가 전반적인 성능을 개선할 수 있음을 발견했습니다. 특히, 높은 온도가 적대적 공격에 대한 모델의 견고성을 향상시킨다는 새로운 이점을 제시했습니다.

- **Technical Details**: 연구에서는 convolutional neural networks (CNNs)와 transformers를 사용하여 다양한 벤치마크 데이터셋에서temperature의 영향력을 분석했습니다. 높은 온도가 모델의 학습 방향을 조정하며, 더 균형 잡힌 학습을 촉진한다는 논의가 이루어졌습니다. 저자들은 또한, 낮은 온도가 오히려 오류가 잦은 클래스에 집중하게 하고, 높은 온도는 모든 클래스에 대한 균형 잡힌 학습을促進한다고 설명했습니다.

- **Performance Highlights**: 저자들은 온도를 조절하여 모델을 훈련했을 때 성능 향상과 함께, 높은 온도를 사용하는 경우에 적대적 훈련의 성능이 개선된다고 밝혔습니다. 이 연구는 모델의 견고성을 높이는 새로운 방법론을 제시하며, deep learning 응용에서의 성능과 보안 강화를 위한 새로운 방향을 제공합니다. 저자들은 이후 모델 성능과 적대적 훈련을 결합하여 온도 조절의 가능성을 보여주었습니다.



### Scalable Coordinated Learning for H2M/R Applications over Optical Access Networks (Invited) (https://arxiv.org/abs/2502.20598)
Comments:
          This article is accepted for publication in 29th Opto-Electronics and Communications Conference 2024 (OECC2024). Copyright @ IEEE

- **What's New**: 이번 논문에서는 Industry 5.0을 위한 인간-기계/로봇(H2M/R) 협업 통신을 다루고 있으며, 이는 광범위한 지리적 거리에서 확장 가능한 H2M/R 통신을 통해 새로운 기계/로봇의 빠른 온보딩을 가능하게 합니다. 연구 결과, $	ext{약} 72	ext{ %}$의 학습 시간을 절약할 수 있습니다. 이는 기존 협업 시스템의 한계를 극복하고, 새로운 기계가 네트워크에 도입될 때 발생하는 비효율성을 해결하기 위한 방안을 제공합니다.

- **Technical Details**: GLAD(글로벌-로컬 인공지능 협력 학습) 프레임워크는 중앙 사무소(CO)에서 로컬 AI와 연결된 H2M/R 협업을 통해 구성됩니다. 이 시스템은 HO가 담당하는 기계/로봇의 제어 신호를 전송하고, 이러한 신호를 기반으로 로컬 AI가 예측할 수 있도록 합니다. GLAD 프레임워크는 네트워크 부하와 거리의 영향을 최소화하여 실시간 또는 준실시간 H2M/R 협업을 가능하게 합니다.

- **Performance Highlights**: GLAD의 성능 평가를 위해 VR 환경에서 H2M/R 협업 실험을 수행했으며, 이를 통해 12000개의 샘플에서 통제 및 해프틱 피드백 데이터를 수집했습니다. 결과적으로, GLAD는 새로운 기계/로봇의 신속한 온보딩을 지원하고 기존 기계/로봇의 작동을 중단하지 않고도 높은 예측 정확도를 유지할 수 있음을 보여주었습니다. 수행된 시뮬레이션 결과, GLAD를 통해 시간 지연을 최소화할 수 있음을 확인했습니다.



### LLMs Have Rhythm: Fingerprinting Large Language Models Using Inter-Token Times and Network Traffic Analysis (https://arxiv.org/abs/2502.20589)
- **What's New**: 이 논문에서는 에 대한 새로운 접근 방식을 제안합니다. 저자들은 대형 언어 모델(LLMs)의 비侵입적인 실시간 지문 기술을 개발했으며, 이는 네트워크 트래픽이 암호화된 경우에도 모델을 식별할 수 있도록 돕습니다. 이 기법은 토큰 간 시간 간격(Inter-Token Times, ITTs)을 측정하여 각 모델의 고유한 타이밍 패턴을 식별합니다.

- **Technical Details**: 제안된 방법은 딥 러닝(Deep Learning) 기반의 파이프라인을 사용하여 네트워크 트래픽 데이터를 처리하고, 36개의 특징을 추출합니다. 이 특징들은 양방향 장기 단기 기억(BiLSTM) 레이어와 다중 헤드 주의 메커니즘(multi-head attention mechanism)을 포함하는 하이브리드 DL 아키텍처로 전달되어 모델을 식별합니다. 저자들은 이 기술을 다양한 배포 시나리오에서 평가하여 효과적이고 강력함을 입증했습니다.

- **Performance Highlights**: 실험 결과는 16개의 소형 언어 모델(SLMs)과 10개의 독점 LLM에서 제안된 기술이 높은 정확도로 모델을 식별할 수 있음을 보여줍니다. 여러 네트워크 조건에서도 탁월한 성능을 유지하며, 이로 인해 모델 식별에 대한 새로운 관점을 제시하고 더 안전한 LLM 배포를 위해 기여할 수 있음을 확인할 수 있었습니다.



### LiteASR: Efficient Automatic Speech Recognition with Low-Rank Approximation (https://arxiv.org/abs/2502.20583)
- **What's New**: LiteASR는 Automatic Speech Recognition (ASR) 인코더에 대한 저품질 압축 방법으로, 특히 Whisper와 같은 현대 모델에서 발견된 강력한 저랭크 특성을 활용합니다. 본 연구에서는 주어진 소량의 교정 데이터셋을 사용하여 주성분 분석(principal component analysis, PCA)을 수행하고, 이를 통해 인퍼런스 성능을 크게 향상시키면서 정확도를 유지합니다. LiteASR는 Whisper large-v3의 인코더 크기를 50% 이상 감소시킬 수 있음을 보여주며, 이는 효율성과 성능의 새로운 Pareto-optimal 경계를 설정합니다.

- **Technical Details**: LiteASR는 숨겨진 활성화에서 저랭크 구조를 활용하여 ASR 인코더의 연산 부하를 줄이는 압축 메커니즘입니다. 이 방법은 주성분 분석을 통해 기본 성분을 추출하고, 이를 통해 선형 변환을 저랭크 행렬 곱셈 체인으로 근사합니다. 또한, 자기 주의(self-attention)를 저차원에서 작동하도록 최적화하여 인퍼런스에 필요한 부동 소수점 연산(FLOPs)을 줄입니다. 이 프로세스는 적응형 메커니즘을 통해 각 레이어의 최적 저랭크 근사도를 결정합니다.

- **Performance Highlights**: LiteASR를 Whisper large-v3에 적용했을 때, 인코더 크기를 약 40% 줄이는 동시에 실행 속도는 약 1.4배 개선되었습니다. 이 방법은 다른 언어와 모델에 걸쳐도 효과적으로 적용 가능성을 보여줍니다. 또한, 모델 크기를 Whisper medium과 유사한 수준으로 줄이면서 향상된 정확도를 달성할 수 있음을 입증했습니다.



### Interpreting CLIP with Hierarchical Sparse Autoencoders (https://arxiv.org/abs/2502.20578)
- **What's New**: 본 논문에서는 Matryoshka SAE (MSAE)라는 새로운 아키텍처를 소개합니다. MSAE는 여러 세분화 수준에서 계층적 표현을 동시에 학습하여 재구성 품질과 희소성(sparsity)이라는 두 가지 지표를 효율적으로 최적화할 수 있도록 돕습니다. 이를 통해 CLIP 모델의 해석 가능성과 제어 가능성을 향상시킵니다.

- **Technical Details**: MSAE는 TopK 연산을 h번 적용하여 점진적으로 증가하는 활성 뉴런 수(k)를 학습합니다. 이러한 방식으로, MSAE는 coarse 개념에서 fine-grained 특징까지 다양한 granularities를 동시에 학습하며, 모든 수준에서 재구성 손실을 결합하여 보다 유연하고 적응적인 희소성 패턴을 달성합니다. 이로써 기존의 TopK 제약조건과 L1 정규화의 문제를 해결합니다.

- **Performance Highlights**: MSAE는 CLIP의 경우에서 0.99 코사인 유사도(cosine similarity)와 0.1 미만의 설명되지 않은 분산(fraction of variance unexplained)을 달성하면서 80%의 희소성을 유지합니다. MSAE의 실용성을 검증하기 위해 120개 이상의 해석 가능한 개념을 추출하고 이를 기반으로 유사도 검색 및 편향 분석을 수행했습니다.



### PFformer: A Position-Free Transformer Variant for Extreme-Adaptive Multivariate Time Series Forecasting (https://arxiv.org/abs/2502.20571)
Comments:
          PAKDD 2025 special session on Data Science: Foundations and Applications (DSFA)

- **What's New**: 이 연구는 PFformer라는 새로운 위치 비기반( position-free) Transformer 모델을 소개합니다. 이 모델은 극단적인 변동성을 특징으로 하는 다변량 시계열(Multivariate time series, MTS) 예측을 위해 설계되었습니다. PFformer는 기존의 모델들이 데이터의 중요한 상관관계를 제대로 반영하지 못하는 문제를 해결하도록 만들어졌습니다.

- **Technical Details**: PFformer는 두 가지 새로운 임베딩 전략을 통합합니다: 향상된 특성 기반 임베딩(Enhanced Feature-based Embedding, EFE)과 자동 인코더 기반 임베딩(Auto-Encoder-based Embedding, AEE)입니다. EFE는 관련된 시퀀스 하위 집합을 위치 제약 없이 고차원 공간으로 매핑하여 상관관계를 효과적으로 인코딩합니다. 이를 통해 모델은 기존의 위치 인코딩 한계 없이 더 나은 성능을 발휘할 수 있습니다.

- **Performance Highlights**: PFformer는 3일 후 장기 예측과 수자원 관리의 실시간 의사결정을 반영한 4시간마다의 롤링 예측 시나리오에서 효과적으로 평가되었습니다. 이 모델은 최신 모델들과 비교하여 예측 정확도를 20%에서 60% 향상시켰습니다. 이러한 성과는 PFformer의 우수한 예측 능력을 보여줍니다.



### DPZV: Resource Efficient ZO Optimization For Differentially Private VFL (https://arxiv.org/abs/2502.20565)
- **What's New**: 본 논문은 Vertical Federated Learning (VFL)에서의 개인정보 보호 및 메모리 효율성을 향상시키기 위해 DPZV라는 새로운 메모리 효율적 Zeroth-Order (ZO) 최적화 프레임워크를 제안합니다. 이 프레임워크는 Differential Privacy (DP)의 개념을 포함하여 세 가지 주요 문제를 해결합니다: (1) gradient 누출로 인한 개인정보 취약점, (2) 첫 번째 순서 방법의 높은 계산 및 통신 비용, (3) 기존 ZO 방식의 지나치게 큰 메모리 사용량입니다.

- **Technical Details**: DPZV는 비동기 통신을 가능하게 하고, 두 점 gradient 추정을 통해 클라이언트의 메모리 사용량을 첫 번째 순서 방법에 비해 90%까지 줄입니다. 또한, 서버에서 가우시안 노이즈를 주입함으로써, 제3자의 신뢰 가정 없이 강력한 $(B5, B4)$-DP 보장을 달성합니다. 이론적으로는 비볼록 목표에서 중앙집중식 케이스와 동등한 수렴 속도를 보여줍니다.

- **Performance Highlights**: 이미지 및 NLP 벤치마크에서의 광범위한 실험 결과, DPZV가 모든 기준 모델에 비해 높은 정확도를 기록하며 강력한 개인정보 보호를 제공했습니다($B5  3C 10$). 자원 제한 환경에서도 계산 자원 요구량이 적어 새로운 최첨단 개인정보-유용성 트레이드오프를 확립하며, 실용적이고 안전하며 확장 가능한 federated learning 솔루션으로 발돋움합니다.



### $Q\sharp$: Provably Optimal Distributional RL for LLM Post-Training (https://arxiv.org/abs/2502.20548)
- **What's New**: 이 연구에서는 KL-정규화된 강화 학습(RL)을 위한 가치 기반 알고리즘인 $Q\sharp$을 소개합니다. 기존의 정책 기반 방법이 가진 한계를 개선하고자 하였으며, 최적의 정규화된 $Q$ 함수를 통해 레퍼런스 정책을 안내합니다. 특히 $Q\sharp$은 수학적 이론을 입증받은 방법으로, KL 정규화된 RL 문제에 대한 최적 정책을 학습하는 데 유리합니다.

- **Technical Details**: 이 알고리즘은 분포론적 강화 학습(distributional RL)을 활용하여 집계된 온라인 데이터셋에서 최적 $Q$ 함수를 학습합니다. $Q\sharp$는 정책의 가중치를 변경하지 않고도 레퍼런스 정책의 생성을 안내할 수 있으며, 이 과정에서 더 작은 모델을 사용해도 성능을 향상시킬 수 있습니다. 전통적인 RL 방법과 달리 $Q\sharp$은 복잡한 시계열 차이 학습(temporal difference learning) 없이 직접 감독 학습(supervised learning)으로 대체합니다.

- **Performance Highlights**: $Q\sharp$은 수학적 추론 벤치마크에서 이전의 기준 알고리즘들보다 뛰어난 성능을 보였습니다. 또한, 레퍼런스 정책과의 KL 발산을 줄임으로써 안정적인 학습을 도모하였습니다. 실험 결과는 $Q\sharp$이 LLM 후속 훈련에 효과적인 접근법임을 보여줍니다.



### Revisiting Kernel Attention with Correlated Gaussian Process Representation (https://arxiv.org/abs/2502.20525)
Comments:
          21 pages, 4 figures

- **What's New**: 이 논문에서는 Correlated Gaussian Process Transformer (CGPT)라는 새로운 변형의 트랜스포머를 제안합니다. 기존 트랜스포머의 한계인 대칭 주의 메커니즘을 탈피하여 비대칭성을 허용하여 모델의 표현 능력을 향상시킵니다. CGPT는 두 개의 상관된 Gaussian Process (GP) 간의 교차 공분산을 모델링하여 불확실성을 효과적으로 보정할 수 있습니다.

- **Technical Details**: CGPT는 다중 헤드 자기 주의 (multi-head self-attention) 메커니즘의 자기 주의 단위를 두 개의 상관된 GP 간의 공분산으로 표현합니다. 이를 통해 각각의 쿼리와 키 기능을 배정하는 서로 다른 아핀 변환을 사용할 수 있습니다. 또한 CGPT는 입출력 토큰 수의 세제곱 의존성을 제거하는 희소 근사법을 개발하여 효율성을 증가시킵니다.

- **Performance Highlights**: 실험 결과에 따르면 CGPT와 희소 CGPT 기반 트랜스포머는 여러 컴퓨터 비전 및 자연어 처리 벤치마크에서 기존의 GP 기반 트랜스포머보다 더 나은 예측 성능을 보여주었습니다. 이러한 성능 향상은 CGPT의 새로운 구조적 특징에 기인합니다. 논문에서 제안된 방법은 다양한 안전 및 신뢰성 중심 작업에 적용될 수 있습니다.



### Personas Evolved: Designing Ethical LLM-Based Conversational Agent Personalities (https://arxiv.org/abs/2502.20513)
- **What's New**: 이 논문은 Large Language Models(LLMs)가 Conversational User Interfaces(CUIs)을 혁신하고, 동적이고 맥락을 반영하는 인간 같은 상호작용을 가능케 한다고 강조합니다. 그러나 LLM 기반의 페르소나의 빠른 채택은 편향, 조작, 예상치 못한 사회적 결과 등의 윤리적 및 실용적 문제를 야기하고 있습니다. 이는 전통적인 CUI와는 달리, LLM 기반 페르소나는 광범위한 데이터셋에서 동적으로 반응을 생성하므로 그 행동을 예측하고 관리하기 더 어렵습니다.

- **Technical Details**: CUI에서 에이전트의 독특한 페르소나와 정체성을 설계하는 것은 오랫동안 중요한 역할을 해왔습니다. LLMs는 자연스럽고 일관된 대화를 가능하게 하며, 텍스트 기반의 상황에서 인간 행동을 시뮬레이션할 수 있는 가능성을 열어줍니다. 이러한 기술은 복잡한 사회적 행동을 시뮬레이션하고 인간 행동을 예측하기 위해 상세한 페르소나 입력을 통합하여 현실적이고 유연한 디지털 생태계를 창조하는 데 기여하고 있습니다.

- **Performance Highlights**: LLM 기반의 페르소나가 다양한 응용 프로그램과 이벤트에 통합된 실례는 많습니다. 예를 들어, LLM 기반의 가상 에이전트가 알버트 아인슈타인 페르소나를 구현하고 전문가들과 실시간으로 토론한 사례에서 볼 수 있듯, LLM은 새로운 상호작용 경험을 제공합니다. 그러나 이러한 발전과 함께, 사용자 조작, misinformation, 예상치 못한 정서적 애착과 같은 윤리적 문제도 제기되고 있습니다.



### TripCraft: A Benchmark for Spatio-Temporally Fine Grained Travel Planning (https://arxiv.org/abs/2502.20508)
Comments:
          27 pages, 18 Tables and 6 Figures

- **What's New**: 최근 대형 언어 모델(LLMs)의 잠재력을 활용한 개인화된 여행 계획 에이전트로서의 가능성을 탐구하는 연구들이 진행되고 있습니다. 하지만 기존의 기준들은 실제 적용 가능성이 제한적입니다. 본 연구에서는 TripCraft라는 새로운 여행 계획 데이터셋을 소개하면서, 이를 통해 현실적인 제약 사항을 통합하여 보다 효과적인 일정 생성을 목표로 하고 있습니다.

- **Technical Details**: TripCraft는 공공 교통 일정, 이벤트 가용성, 다양한 관광 카테고리 및 사용자 성격(persona)과 같은 실세계 제약을 통합한 공간적 시간적(spatiotemporally) 일관된 여행 계획 데이터셋입니다. 또한, 기존의 이진 검증 방법을 넘어 여행 계획의 질을 평가하기 위해 다섯 가지 지속적 평가 메트릭스, 즉 Temporal Meal Score, Temporal Attraction Score, Spatial Score, Ordering Score 및 Persona Score를 제안합니다.

- **Performance Highlights**: 모델의 매개변수 설정을 통해 식사 일정 수립이 크게 향상되어 7일 시나리오에서 Temporal Meal Score가 61%에서 80%로 증가했습니다. TripCraft는 LLM 기반의 개인화된 여행 계획을 위한 새로운 기준을 제시하며, 일정 생성에 있어 더 현실적이고 제약을 고려한 프레임워크를 제공합니다. 데이터셋과 코드베이스는 수락 후 공개될 예정입니다.



### A Thousand Words or An Image: Studying the Influence of Persona Modality in Multimodal LLMs (https://arxiv.org/abs/2502.20504)
- **What's New**: 최근 대형 언어 모델(LLMs)은 다양한 페르소나(persona)를 구현하는 데 있어 큰 발전을 보여주었습니다. 이 논문은 텍스트와 이미지라는 서로 다른 표현 방식을 갖는 페르소나의 영향력을 처음으로 체계적으로 분석합니다. 새로운 모달리티 평행 데이터셋을 생성하고, LLM이 특정 페르소나의 특성과 시나리오를 얼마나 잘 표현하는지를 평가하기 위한 프레임워크를 개발하였습니다.

- **Technical Details**: 우리는 다양한 연령, 성별, 직업, 위치를 가진 4040명의 페르소나로 이루어진 데이터셋을 생성하였습니다. 각 페르소나는 이미지 전용, 텍스트 전용, 이미지와 짧은 텍스트의 조합, 타이포그래피를 활용한 이미지 네 가지 방식으로 표현됩니다. 이러한 표현방식에 대한 비교 분석을 통해 멀티모달 LLM의 성능을 평가하고, 텍스트 기반 페르소나가 이미지 표현보다 뛰어난 결과를 나타냈습니다.

- **Performance Highlights**: 실험 결과, LLM은 상세한 텍스트로 표현된 페르소나에서 더 많은 언어적 습관을 보였고, 타이포그래피 이미지는 페르소나와의 일관성이 더 높았습니다. 이러한 연구 결과는 LLM이 이미지로 전달되는 페르소나 특정 세부 사항을 종종 간과하고 있음을 밝혔습니다. 이를 통해 향후 연구가 이 격차를 좁히는 데 기여할 수 있도록 하는 기반을 마련합니다.



### Unified Kernel-Segregated Transpose Convolution Operation (https://arxiv.org/abs/2502.20493)
- **What's New**: 본 논문에서는 딥러닝 애플리케이션을 위한 transpose convolution layer의 최적화를 위해 kernel segregation 메커니즘을 제안합니다. 기존의 kernel segregation이 가지는 단점, 즉 스레드를 실행할 때 홀수 차원을 가진 출력 특성 맵을 얻기 위해 추가 요소를 계산해야 하는 문제를 해결하고자 합니다. 이를 위해, 네 개의 서브 커널을 실행하기 위해 하나의 통합된 커널을 사용하는 통합 커널 세그리게이션 접근 방식을 도입하였습니다.

- **Technical Details**: 이 접근 방식은 메모리와 계산 자원의 사용을 제한하면서 더욱 효율적으로 transpose convolution layer를 최적화합니다. 연구 결과에 따르면, RTX 2070 GPU를 사용할 때 특정 데이터셋에서 평균 2.03배의 계산 속도 향상이 이루어졌고, Intel Xeon CPU에서는 3.89배의 향상이 있음을 보여줍니다. 또한, 널리 알려진 Generative Adversarial Networks(GANs)의 transpose convolution layers를 평가했을 때 평균 3.5배의 계산 속도 향상을 보였습니다.

- **Performance Highlights**: 제안된 방법을 EB-GAN 모델의 transpose convolution layers에 적용한 결과, 최대 35MB의 메모리 절약을 이루었습니다. 이는 메모리 효율성을 극대화하면서도 계산 성능을 향상하는 데 중요한 기여를 합니다. 이러한 개선은 앞으로의 딥러닝 모델 구축에 있어 매우 유용할 것으로 기대됩니다.



### EgoNormia: Benchmarking Physical Social Norm Understanding (https://arxiv.org/abs/2502.20490)
- **What's New**: 이 논문은 Norms(규범) 이해 및 추론을 위한 새로운 데이터셋인 EgoNormia를 소개합니다. 총 1,853개의 자기 중심의 동영상(interaction videos)과 관련 질문들이 포함되어 있으며, 이는 기계가 다양한 규범들 간의 trade-off(상충) 문제를 이해하도록 도와줍니다.

- **Technical Details**: EgoNormia 데이터셋은 안전(safety), 개인정보 보호(privacy), 근접성(proxemics), 공손성(politeness), 협력(cooperation), 조정/능동성(coordination/proactivity), 소통/가독성(communication/legibility) 등 일곱 가지 규범적 행동(normative actions)을 평가하는데 초점을 맞추고 있습니다. 이 데이터셋의 구축에 있어 비디오 샘플링(video sampling), 자동 답변 생성(automatic answer generation), 필터링(filtering) 및 인간 검증(human validation)을 활용한 새로운 파이프라인(pipeline)을 제안합니다.

- **Performance Highlights**: 현재 최첨단 비전-언어 모델(VLMs)은 EgoNormia에서 최대 45%를 기록하며, 인간 기준(92%)과 비교할 때 규범적 이해가 부족함을 보여줍니다. 이 논문에서는 안전, 개인정보 보호 및 협력과 소통 능력의 부족으로 인해 실제 에이전트(real-world agents)에 적용할 경우 발생할 수 있는 큰 위험을 강조합니다. 또한, 검색 기반 생성 방법을 통해 EgoNomia를 사용하여 VLMs의 규범적 추론(normative reasoning) 능력을 향상시킬 수 있음을 보여줍니다.



### Promote, Suppress, Iterate: How Language Models Answer One-to-Many Factual Queries (https://arxiv.org/abs/2502.20475)
- **What's New**: 이번 연구에서는 언어 모델(LM)이 다중 사실 질의(one-to-many factual queries)에 응답하는 기제를 분석합니다. 구체적으로, 모델은 모든 대답을 회상한 뒤, 이전에 생성된 대답을 억제하는 promote-then-suppress 메커니즘을 사용합니다. 이 연구는 모델이 어떻게 주제와 이전의 답변 토큰을 활용하여 지식을 회상하고, 반복을 피하는지를 탐구합니다.

- **Technical Details**: 모델은 attention과 다층 퍼셉트론(MLP)을 사용하여 주제 정보와 이전의 답변을 처리합니다. 중간 계층에서는 주제 정보를 복사하고, MLP는 가능한 모든 답변을 증진시키며, 후반 계층에서는 이전 답변을 억제합니다. Token Lens 기법과 knockout 방법을 통해 모델의 작동 기제를 분석하고, 특정 토큰들이 어텐션과 MLP 계층에서 어떻게 사용되는지를 연구합니다.

- **Performance Highlights**: 이 연구는 Llama-3-8B-Instruct 및 Mistral-7B-Instruct 모델이 다층 네트워크를 통해 어떻게 지식 회상 및 반복 방지를 수행하는지를 다양한 데이터 세트를 통해 증명합니다. 모델은 주제와 이전 답변 토큰들을 사용하여 명확한 답변을 생성하는 데 필요한 모든 정보를 집계합니다. 이 연구는 복잡한 사실 회상을 위한 LMs의 동적 정보 통합 방식을 밝혀내며, 향후 연구에 중요한 기초 자료를 제공합니다.



### Will AI replace Software Engineers? Hold your Breath (https://arxiv.org/abs/2502.20429)
Comments:
          3 pages

- **What's New**: 이번 논문에서는 Large Language Models (LLMs)가 소프트웨어 엔지니어링의 중요한 측면인 유지보수와 이해에 대해서는 아직 역량이 부족하다는 점을 강조합니다. LLMs는 코드 생성을 자동화하는 데 유용하지만, 기능적, 안전성, 보안 문제를 다루기 위해서는 인간 소프트웨어 엔지니어의 역할이 필수적입니다. 향후 LLM이 소프트웨어 직종을 완전히 대체할 것이라는 견해는 과장된 것임을 확인합니다.

- **Technical Details**: LLMs는 자연어 설명에서 자동 코드를 생성하고 코드 문제를 자동으로 해결하는 데 놀라운 능력을 보여주고 있습니다. 그러나 코드의 실행 의미에 대해 이해하고 이유를 제시하는 능력은 제한적입니다. LLM은 기존 코드 이해와 프로그램의 동적 실행 의미를 포괄적으로 다룰 수 있는 능력이 결여되어 있으며, 이는 소프트웨어 엔지니어가 수동으로 해결해야 할 복잡한 문제들입니다.

- **Performance Highlights**: LLMs는 기초적인 프로그래밍 작업에서 패턴 인식을 활용할 수 있지만, 동적 실행 의미를 포착하고 이와 관련된.reasoning에 있어서는 상당한 한계를 가지고 있습니다. 미래의 소프트웨어 작업에서 LLMs는 유용한 도구가 될 수 있지만, 인간 엔지니어와의 협력이 매우 중요하며, LLMs가 모든 소프트웨어 프로세스를 완전히 대체하는 것은 불가능할 것으로 예상됩니다.



### DeePen: Penetration Testing for Audio Deepfake Detection (https://arxiv.org/abs/2502.20427)
- **What's New**: 이 논문에서는 Deepfake(딥페이크) 콘텐츠를 감지하는 기계 학습 기반 분류기의 강인성을 평가하기 위해 DeePen이라는 체계적인 침투 테스트 방법론을 도입하였습니다. DeePen은 목표 딥페이크 감지 모델에 대한 사전 지식이나 접근 없이 작동하며, 신호 처리 수정(신호 처리 방식의 변경)을 활용하여 모델의 취약점을 평가합니다. 이 접근 방식은 딥페이크 감지의 안전성을 높이기 위한 새로운 시도를 제시합니다.

- **Technical Details**: DeepPen은 여러 종류의 신호 처리 공격을 적용하여 모델을 테스트합니다. 이는 시간 늘리기(time-stretching) 및 에코 추가(echo addition)와 같은 간단한 조작을 포함하여 모든 테스트한 시스템이 취약점을 보인다는 것을 보여줍니다. 논문에서 발표한 결과는 실세계의 생산 시스템 및 공개된 학술 모델 체크포인트를 모두 포함한 분석을 기반으로 합니다.

- **Performance Highlights**: 결과적으로, 특정 공격에 대한 인식이 없는 감지 시스템은 간단한 조작에도 쉽게 오인되는 것을 발견했습니다. 몇몇 공격은 해당 공격에 대한 지식을 가지고 재훈련을 통해 완화될 수 있지만, 다른 공격들은 지속적으로 효과를 발휘합니다. 이 논문은 관련된 모든 코드를 공개하여 후속 연구자들이 활용할 수 있도록 하였습니다.



### Among Them: A game-based framework for assessing persuasion capabilities of LLMs (https://arxiv.org/abs/2502.20426)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 설득 능력을 평가하기 위한 독창적인 게임 프레임워크인 'Among Them'을 제안합니다. 이 프레임워크는 25가지 사회 심리학 및 수사학적 설득 전략을 기반으로 LLM의 조작 기술을 정량화 및 정성화할 수 있게 합니다. 640개의 게임 실험을 통해, 다양한 유형과 크기의 8가지 LLM이 조사되었으며, 모든 모델이 25가지 기법 중 22가지 기법을 성공적으로 활용할 수 있다는 것을 보여주었습니다.

- **Technical Details**: 논문에서는 LLM 에이전트를 위한 게임 환경, LLM 에이전트 모듈, 설득 평가 모듈, 평가 대시보드의 네 가지 주요 모듈을 포함하는 'Among Them' 프레임워크를 도입합니다. 참가자들은 크루원과 사기꾼의 두 역할로 나뉘어, 사기꾼을 식별하고 임무를 수행하는 사회적 추측 게임을 진행합니다. 각 게임은 행동, 토론, 투표의 3단계로 진행되며, 게임 내 상태 정보, 플레이어 위치, 생존 상태, 행동 기록 등이 자세히 추적됩니다. 또한, OpenRouter API를 통해 다양한 모델의 선택이 가능하게 구현되었습니다.

- **Performance Highlights**: 실험 결과, 모든 LLM 모델은 반복적인 게임 진행에서 프레임워크 내에서 성공적으로 설득 기술을 사용할 수 있었습니다. 흥미롭게도, 더 큰 모델이 더 작은 모델에 비해 설득 면에서 유리하지 않으며, 응답 길이가 길어질수록 승리한 게임 수와는 부정적인 상관관계가 발견되었습니다. 이는 LLM의 조작 가능성을 깊이 이해하는 데 기여하며, 향후 AI의 조작 능력에 대한 연구를 촉진할 자료를 제공합니다.



### Efficient Risk-sensitive Planning via Entropic Risk Measures (https://arxiv.org/abs/2502.20423)
- **What's New**: 본 논문은 Markov Decision Processes (MDPs)에서 tail-focused metrics를 최대화하는 risk-sensitive planning의 새로운 접근법을 제시합니다. 기존 연구들은 Entropic Risk Measures (EntRM)가 동적 프로그래밍을 통해 효율적으로 최적화할 수 있음을 보여주었지만, 해석하기 어려운 매개변수 선택 문제를 남겼습니다. 저자는 EntRM의 전체 최적 정책 집합을 계산함으로써 관심 있는 메트릭에 대한 강력한 근사를 제공할 수 있음을 입증합니다.

- **Technical Details**: 제안된 방법은 새로운 구조 분석 및 entropic risks의 smoothness properties를 활용하여 최적성 경계를 효과적으로 계산할 수 있도록 합니다. 이러한 접근 방식에 따르면, 다양한 매개변수 값에 대해 EntRM을 통해 최적의 정책을 찾아낼 수 있습니다. 이러한 과정을 통해 얻어진 최적 정책은 여러 의사결정 시나리오에서의 성능을 높이는 데 기여합니다.

- **Performance Highlights**: 실험 결과는 제안된 방식이 다양한 의사결정 시나리오에서 우수한 성능을 달성함을 보여줍니다. 이는 현재 널리 사용되는 메트릭인 threshold probabilities나 (Conditional) Values at Risk와 비교했을 때, 훨씬 더 효율적이고 해석 가능한 방안을 제공한다는 것을 의미합니다. 결과적으로, 저자들은 이 접근법이 실제 세계의 복잡한 의사결정 문제에 적용될 수 있음을 강조합니다.



### SEKI: Self-Evolution and Knowledge Inspiration based Neural Architecture Search via Large Language Models (https://arxiv.org/abs/2502.20422)
- **What's New**: 이번 논문에서는 SEKI라는 새로운 대형 언어 모델(LLM) 기반의 신경망 아키텍처 검색(NAS) 방법을 소개합니다. SEKI는 현대 LLM의 체인 오브 사고(Chain-of-Thought, CoT) 패러다임에서 영감을 받아 두 가지 주요 단계인 자기 진화(self-evolution)와 지식 증류(knowledge distillation)로 작동합니다. 이 방법은 기존 아키텍처에 의존하지 않고도 효율성과 성능을 향상시킬 수 있는 잠재력을 지니고 있습니다.

- **Technical Details**: SEKI의 첫 번째 단계인 자기 진화에서는 초기 LLM이 참조 예제가 부족하여 성능 피드백을 기반으로 아키텍처를 반복적으로 개선합니다. 이후 두 번째 단계인 지식 증류에서는 고성능 아키텍처의 공통 패턴을 분석하여 새로운 최적화된 설계를 생성합니다. 이 과정을 통해 SEKI는 역량을 극대화하며, 단 0.05 GPU-days의 비용으로 SOTA 성능을 달성합니다.

- **Performance Highlights**: SEKI는 CIFAR-10에서 97.71%, CIFAR-100에서 84.14%, 이미지넷에서 75.8%의 최고 정확도를 기록하였으며, 특히 이미지넷에 대한 직접 검색에서 76.1%의 top-1 정확도를 달성하여 기존의 SOTA 방법들을 능가합니다. SEKI는 여러 작업에서 SOTA 경합 결과를 보이며 강력한 일반화 능력을 보여줍니다. 이처럼 SEKI는 NAS 분야에서 LLM의 성공과 가능성을 제시하고 있습니다.



### Backpropagation-free Spiking Neural Networks with the Forward-Forward Algorithm (https://arxiv.org/abs/2502.20411)
- **What's New**: 이번 연구에서는 Forward-Forward (FF) 알고리즘을 통해 SNN(Spiking Neural Networks) 훈련을 위한 새로운 접근 방식을 제안합니다. 기존의 역전파(Backpropagation)와 달리 FF 알고리즘은 두 번의 전진 전달만을 사용하여 지역적 학습(localized learning)을 촉진하며, 계산 효율성을 높이고 신경형 하드웨어(neuromorphic hardware)와의 호환성을 개선합니다. 새로운 FF 기반 SNN 훈련 프레임워크를 소개하고, 다양한 데이터셋에서의 성능을 평가하였습니다.

- **Technical Details**: SNN은 이산 스파이크 이벤트를 통해 정보를 처리하는 생물학적으로 영감을 받은 모델입니다. 하지만 SNN 훈련은 주로 신경망 출력 오류에 대한 개별 신경의 기여를 할당하는 데 어려움을 겪고 있습니다. FF 알고리즘은 두 개의 전진 전달만을 필요로 하며, 이를 통해 모든 중간 활동을 저장할 필요가 없고 대칭적인 가중치 연결이 필요하지 않아 생물학적 원리에 더 가깝습니다. 또한, FF는 데이터 파이프라인을 중단하지 않고도 즉각적인 학습이 가능 합니다.

- **Performance Highlights**: 실험 결과, FF 기반 모델은 MNIST와 Fashion-MNIST에서 기존의 FF 기반 SNN보다 5% 이상 높은 성능을 보였고, 최신 역전파 훈련 SNN과 유사한 정확성을 달성했습니다. 더욱 복잡한 작업인 CIFAR-10과 SHD에서도 최대 6%의 성능 향상을 보여주며, 역전파 기반 SNN과 경쟁력을 차지하고 있습니다. 이러한 결론은 FF 알고리즘이 SNN 훈련 방법론의 발전과 신경형 컴퓨팅의 확장을 가능하게 할 수 있음을 강조합니다.



### Brain-Inspired Exploration of Functional Networks and Key Neurons in Large Language Models (https://arxiv.org/abs/2502.20408)
Comments:
          13 pages, 5 figures

- **What's New**: 본 연구는 대규모 언어 모델(LLMs)에서의 기능적 네트워크(FBNs)를 탐구하기 위한 새로운 접근 방식을 소개합니다. 기존의 연구들은 개별 뉴런의 기여도에 중점을 두었으나, 본 연구는 뉴런 간의 상호작용을 통해 성능이 발휘되는 복잡한 네트워크를 분석합니다. 이러한 새로운 관점은 LLM의 메커니즘을 이해하는 데 기여하며, 이를 통해 모델의 투명성 및 신뢰성을 향상시킬 수 있습니다.

- **Technical Details**: 본 연구에서는 LLM의 다층 퍼셉트론(MLP) 층의 출력 데이터를 기능적 자기 공명 영상(fMRI) 신호와 유사하게 간주하여 독립 성분 분석(ICA) 기법을 적용합니다. 이러한 분석 방법을 통해 LLM 내에서 다수의 기능적 네트워크가 존재함을 확인하였으며, 이들 네트워크는 다양한 입력 자극 간의 공간적 일관성을 보여줍니다. 특히, 핵심 기능 네트워크를 차단할 경우 모델의 성능이 상당히 저하된다는 결과를 아울러 제시하였습니다.

- **Performance Highlights**: 연구 결과, LLM에서 2% 미만의 뉴런으로 구성된 핵심 기능 네트워크를 유지할 경우 모델 성능을 유지하거나 향상시킬 수 있음을 발견하였습니다. 불필요한 뉴런을 차단하면서 새로운 기능적 네트워크를 점진적으로 통합함으로써 모델의 성능이 낮은 단계에서 높은 단계로 개선될 수 있음을 보여주었습니다. 마지막으로, MLP 층의 10% 이하의 뉴런만으로도 원래 네트워크와 맞먹는 성능을 달성했습니다.



### Pause-Tuning for Long-Context Comprehension: A Lightweight Approach to LLM Attention Recalibration (https://arxiv.org/abs/2502.20405)
- **What's New**: 본 연구에서는 LLMs가 긴 문맥(compact context)에서 정보를 처리하는 데 어려움을 겪는 Lost-in-the-Middle (LITM) 문제를 해결하기 위해 새로운 접근 방법을 제안합니다. 이것은 pause-tuning이라는 기법으로, 입력 데이터에서 인위적으로 삽입된 pause tokens를 사용하여 주의(attention)를 재분배하는 것입니다. 이러한 방식으로 긴 입력을 더 작은 청크로 나누어 각 부분을 보다 효과적으로 처리할 수 있도록 합니다.

- **Technical Details**: Pause tokens는 입력 시퀀스에 전략적으로 삽입되어 모델의 주의 분포를 조정합니다. 연구에서는 이 pause tokens의 삽입 방법을 여러 가지로 실험하고, 15가지 문맥 깊이(context depth)와 3회의 단일 needle 테스트를 포함하여 최적의 기술을 찾기 위해 노력했습니다. 각 테스트에서 모델의 성능을 비교하고 분석하여 주의의 재배치가 긴 문맥 처리에 미치는 영향을 평가하였습니다.

- **Performance Highlights**: 실험 결과, LLaMA 3.2 3B Instruct 모델과 LLaMA 3.1 8B Instruct 모델이 각각 10.61%와 3.57%의 성능 향상을 보였습니다. 또한 pause-tuning 기법이 기존의 다른 기술보다 긴 문맥 유지(long-context retention) 및 처리 개선에 보다 효과적임을 보여주었습니다. 이 연구는 LLMs에서의 긴 문맥 처리에서 pause tokens의 효과성을 강조하며, 향후 이러한 기법의 활용 가능성을 제시합니다.



### Adversarial Robustness of Partitioned Quantum Classifiers (https://arxiv.org/abs/2502.20403)
- **What's New**: 이 논문은 양자 분류기(quantum classifiers)의 회로 분할(circuit cutting) 기법을 통해 적대적 공격(adversarial attack)에 대한 취약성을 어떻게 증가시키는지를 조사하는 최초 연구로, 기존의 문헌에서 다루지 않았던 중요한 주제를 다룹니다. 또한, 원래 회로의 출력을 재현하지 못하게 하는 적대적 게이트(adversarial gates)의 삽입이 양자 분류기의 설계에 미치는 영향을 확인합니다. 이로 인해 양자 기계 학습(quantum machine learning)의 장점과 고전 모델(classical models)과의 성능 비교에서 중요한 고찰을 제공합니다.

- **Technical Details**: NISQ(Noise Intermediate-Scale Quantum) 시대에서, 회로 분할 기법을 통해 여러 양자 처리 장치에서 양자 회로를 실행할 수 있도록 하고, 고전적 통신(classical communication)을 통해 결합할 수 있습니다. 이 연구는 회로 분할 과정을 통해 준비된 상태에 적대적 변화를 추가함으로써 양자 분류기의 설계를 더욱 취약하게 만드는 방법을 이론적 및 실험적 관점에서 조사합니다. 또한, 다양한 깊이에서 적대적 게이트를 도입함으로써 예측 신뢰도(predictive confidence)에 미치는 잠재적 변화를 이론적으로 구속하는 정리를 제시합니다.

- **Performance Highlights**: 이 연구의 실험 결과는 변형 양자 분류기(variational quantum classifiers)에서 적대적 게이트를 배치할 경우, 전통적인 입력 상태를 공격하는 것과는 다른 방식으로 예측 성능에 큰 영향을 미친다는 것을 보여줍니다. 실험적 조사를 통해서 적대적 게이트의 깊이에 따른 효과를 비교하며, 회로 조각의 결과를 결합할 때 발생하는 전반적인 신뢰도 저하를 시각적으로 확인합니다. 이 논문은 양자 기계 학습 분야에서 중요성이 증가하고 있는 적대적 견고성을 다루며, 양자 분류기의 미래 개선에 중요한 통찰을 제공합니다.



### HVI: A New Color Space for Low-light Image Enhancemen (https://arxiv.org/abs/2502.20272)
Comments:
          Qingsen Yan, Yixu Feng, and Cheng Zhang contributed equally to this work

- **What's New**: 이 논문은 Low-Light Image Enhancement (LLIE) 분야의 새로운 접근 방식을 제안합니다. 기존의 방법들은 주로 RGB (sRGB) 색 공간에 기반을 두어 색 균형 문제와 밝기 왜곡을 발생시킵니다. 새로운 수평/수직-강도 색 공간인 Horizontal/Vertical-Intensity (HVI)를 도입하여 이러한 문제를 해결하고자 합니다.

- **Technical Details**: HVI 색 공간은 편광된 HS 맵과 학습 가능한 강도를 기반으로 정의됩니다. 이 시스템은 빨간색 좌표 간 작은 거리를 강제하여 빨간색 아티팩트를 제거하고, 저조도 영역을 압축하여 검은색 아티팩트를 줄입니다. 또한, 새로운 Color and Intensity Decoupling Network (CIDNet)를 통해 다양한 조명 조건에서 정확한 포토메트릭 맵핑 함수(photometric mapping function)를 학습합니다.

- **Performance Highlights**: 벤치마크 및 아블레이션 실험 결과, HVI 색 공간과 CIDNet을 활용한 접근법이 10개 데이터셋에서 최첨단 방법들보다 우수한 성능을 보였습니다. 이 연구는 저조도 이미지 복원 분야의 기법 개선에 크게 기여할 것으로 기대됩니다.



### Behind the Tip of Efficiency: Uncovering the Submerged Threats of Jailbreak Attacks in Small Language Models (https://arxiv.org/abs/2502.19883)
Comments:
          12 pages. 6 figures

- **What's New**: 본 논문은 작은 언어 모델(SLMs)의 보안 성능을 평가하기 위해 13개의 최신 SLM을 다양한 jailbreak 공격하에 실험한 포괄적인 연구를 제공합니다. SLM은 효율성과 낮은 계산 비용 덕분에 엣지 디바이스에서의 배포에 매력적이지만, 보안 리스크는 대형 언어 모델(LLMs)에 비해 상대적으로 소홀히 다뤄졌습니다. 연구 결과, 대부분의 SLM이 기존 jailbreak 공격에 취약하다는 것을 밝혀내며, 대응 방법의 효과도 평가합니다.

- **Technical Details**: 이 연구는 16개의 최첨단 모델을 수집하여 SLM과 LLM의 보안 차이를 폭넓게 살펴봅니다. SLMs는 수십억 개의 매개변수로 구성되어 있어 적은 양의 교육 데이터와 계산 비용으로 배포할 수 있지만, LLM에 비해 보안에 더 취약합니다. 찾아낸 취약점은 SLM의 아키텍처 압축, 양자화(quantization), 지식 증류(knowledge distillation) 등 다양한 요인들로 인해 발생합니다.

- **Performance Highlights**: SLM에 대한 다양한 방어 기법의 효과를 평가한 결과, 이러한 기법들은 SLM이 jailbreak 공격에 대한 회복력을 향상시키는 데 상당한 적용 가능성을 가지고 있음을 보여주었습니다. 특히 SLM은 LLM에 비해 대응 방법의 적응력이 높아, 보안 강화를 위한 새로운 통찰력을 제공합니다. 이 연구는 SLM 개발 시 보안과 생성 능력 간의 균형을 맞추기 위한 중요한 기초 자료가 될 것입니다.



New uploads on arXiv(cs.LG)

### Clustering Context in Off-Policy Evaluation (https://arxiv.org/abs/2502.21304)
Comments:
          35 pages, 25 figures, 2 tables. AISTATS 2025

- **What's New**: 이번 연구에서는 Off-policy evaluation (OPE) 방법론을 발전시키기 위해 클러스터링을 활용한 대안 추정기를 제안합니다. 기존의 off-policy 기법들은 정책과 로깅 정책 간의 차이로 인해 성능이 저하되는 문제를 겪고 있었으나, 새로운 접근법은 유사한 컨텍스트 간 정보를 공유함으로써 이를 해소하려 시도합니다. 실험을 통해 이 방법이 특히 정보가 부족한 상황에서 추정 정확도를 향상시키는 것을 확인하였습니다.

- **Technical Details**: 연구는 다양한 구조의 컨텍스트 클러스터를 활용하여 클러스터 내의 모든 컨텍스트로부터 정보를 집계하는 방법을 제안합니다. 제안된 추정기인 CHIPS(Clustering-based Importance-weighted Policy Score)는 이론적인 특성을 분석하며, 비편향(bias)과 분산(variance) 특성을 다양한 조건 하에 특성화합니다. OPE 문제의 정의 및 관련 알고리즘을 설명하며, 이론적인 결과들이 현실 세계의 추천 시스템에 어떻게 적용되는지 탐구합니다.

- **Performance Highlights**: CHIPS 추정기의 성능은 여러 합성 데이터셋과 실제 추천 데이터셋에서 기존 기법들과 비교되어 그 효과를 입증했습니다. 실험 결과, 클러스터링을 통한 컨텍스트 정보의 통합이 추정 정확도 향상에 기여하는 것으로 나타났습니다. 특히, 로깅 정책에서 저확률을 갖는 행동이 많은 환경에서 이 방법의 장점이 두드러지게 나타났습니다.



### Controlled Model Debiasing through Minimal and Interpretable Updates (https://arxiv.org/abs/2502.21284)
- **What's New**: 이 논문에서는 기존 모델을 고려하지 않고 새롭게 모델을 구축해야 하는 전통적인 공정한 기계 학습 모델 접근 방식의 한계를 극복하기 위해, 'Controlled Model Debiasing'이라는 개념을 소개합니다. 이 접근법은 새로운 공정 모델과 기존 모델 간의 차이를 최소화하고 해석 가능하도록 만드는 것을 목표로 하며, 관련 이론적인 보장도 제공합니다. 이는 가능한 한 적은 모델 업데이트로 공정성을 달성할 수 있는 가능성을 제시합니다.

- **Technical Details**: 본 논문은 'Controlled Model Debiasing'의 두 가지 요구사항 - 변화는 최소화되어야 하고, 변화는 해석 가능해야 한다는 - 를 바탕으로 새로운 모델 업데이트 과제로 프레이밍합니다. COMMOD(COncept-based Minimal MOdel Debiasing)라는 알고리즘을 통해, 기존 편향된 모델을 바탕으로 하여, 해석 가능하고 희소한 변경을 통해 공정성을 증대시키는 방안을 제시합니다. 이 알고리즘은 모델에 의존하지 않으며 테스트 시 민감한 속성이 필요하지 않습니다.

- **Performance Highlights**: COMMOD 알고리즘은 실험을 통해 고전적인 공정성 데이터셋에서 기존의 최첨단 불균형 해소 방법들과 비교해 동등한 성능을 나타내며, 보다 적은 수의 예측 변화로 실행 가능합니다. 즉, 이 방법은 예측 변화가 더 의미 있고 이해하기 쉬워, 높은 이해도를 요구하는 실무에서도 활용할 수 있는 가능성을 높입니다.



### L-Lipschitz Gershgorin ResNet Network (https://arxiv.org/abs/2502.21279)
Comments:
          10 pages, 6 figures

- **What's New**: 이번 논문은 깊은 잔여 네트워크(Deep Residual Network)를 Linear Matrix Inequality (LMI) 형식으로 재구성하여 $	ext{L}$-Lipschitz 연속성을 보장하는 매개변수 제약을 파생하는 새로운 방법론을 제안합니다. 이는 기존의 접근 방식이 지닌 제약을 극복하며, 복잡한 네트워크 구조에서도 이론적인 안정성을 보장합니다. Gershgorin 원 이론을 활용하여 이 문제를 해결함으로써 고유값의 부정 준거성을 보장하고, 네트워크의 신뢰성을 증가시킵니다.

- **Technical Details**: 이 연구에서는 깊은 잔여 네트워크의 내부 구조를 선형 방정식의 재귀 시스템으로 나타내어 $	ext{L}$-Lipschitz 조건을 유지하면서 표현력을 극대화하고자 하였습니다. LMI 제약 조건을 이용하여 각 레이어의 매개변수를 명확하게 규정하였으며, 일반적인 활성화 함수에서도 적용 가능한 제약 조건을 제시합니다. 또한, 이전 연구들은 단일 레이어로 제한된 잔여 네트워크에 초점을 맞췄지만, 본 연구는 더 다양한 내층 구조를 갖는 일반화된 모델을 구성합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 적대적 공략(adversarial attacks)에 대해 더 높은 견고성을 보여주었으며, 이론적으로 보장된 Lipschitz 제약 덕분에 네트워크의 출력이 입력 변화에 안정적으로 유지되었음을 증명합니다. 그러나 Gershgorin 기반의 근사 방식은 시스템을 과도하게 제약하여 비선형 동적 요소를 억제하며, 이로 인해 네트워크의 표현력이 감소하는 한계를 드러냈습니다. 이러한 점은 향후 연구를 통해 보완해야 할 중요한 사항으로 언급됩니다.



### Does Generation Require Memorization? Creative Diffusion Models using Ambient Diffusion (https://arxiv.org/abs/2502.21278)
Comments:
          33 pages

- **What's New**: 이번 연구에서는 최신의 diffusion modeling 패러다임에서 모델이 훈련 세트를 기억하는 문제를 해결하기 위한 새로운 접근법을 제시합니다. 기존의 방법들은 일반적으로 이미지 품질을 저하시켰지만, 본 연구는 높은 생성 품질을 유지하면서 메모리화(memorization)를 줄일 수 있는 가능성을 탐구합니다.

- **Technical Details**: 연구에서는 이론적 근거를 통해 diffusion 모델에서의 메모리화는 주로 저노이즈 스케일에서의 denoising 문제에 필요하다는 사실을 밝혔습니다. 이를 바탕으로, 우리는 고노이즈 스케일에서 노이즈가 있는 데이터를 활용하여 모델을 훈련하는 간단하면서도 원칙적인 방법을 제안합니다.

- **Performance Highlights**: 제안한 방법은 텍스트 조건 모델과 비조건 모델 모두에서 메모리화를 현저하게 줄이는 데 성공했으며, 다양한 데이터 가용성 설정에서도 이미지 품질을 저하시키지 않는 결과를 보여주었습니다. 이로써 연구는 높은 수준의 생성 품질과 낮은 메모리화 간의 균형을 향상시키는 데 기여했습니다.



### BAnG: Bidirectional Anchored Generation for Conditional RNA Design (https://arxiv.org/abs/2502.21274)
- **What's New**: 이 논문에서는 RNA 분자와 특정 단백질 간의 상호작용을 예측하기 위해 RNA-BAnG라는 심층 학습 기반 모델을 개발하였습니다. 기존의 접근법과는 달리, 실험적으로 결정된 RNA 서열이나 RNA 구조에 대한 세부 지식 없이도 RNA 서열을 생성할 수 있는 방법을 제시합니다. 또한 새로운 생성 방법인 Bidirectional Anchored Generation (BAnG)을 사용하여, 단백질 결합 RNA 서열이 일반적으로 기능적 결합 모티프를 포함하고 있다는 점을 이용합니다.

- **Technical Details**: BAnG 모델은 두 개의 특수 앵커 토큰 (<ancl>, <ancr>)을 사용하여 서열을 생성하는 방식을 채택합니다. 이 모델은 먼저 오른쪽에서 다음 토큰을 샘플링한 후 왼쪽에서 또 한 번 토큰을 생성하는 방식으로 진행되며, 이 과정을 반복하여 서열이 확장됩니다. 모델 트레이닝과 추론 동안 각 방향의 다음 토큰 확률은 가장 최근에 생성된 토큰의 임베딩을 기반으로 하도록 설계되어 있습니다.

- **Performance Highlights**: RNA-BAnG 모델은 합성 작업과 생물학적 서열 평가에서 그 효과성을 입증하였습니다. 기존의 세대 방법들과 비교했을 때, 제공된 단백질에 대해 조건부 RNA 서열 디자인에서 더 나은 성능을 보여주며, 복잡한 바이오 분자 문제를 해결하는 데 있어 새로운 가능성을 열고 있습니다. 최종적으로, 이 방법은 기존의 실험적 데이터에 의존하지 않고도 RNA 서열 생성을 가능하게 하여, 생물의학적 응용에 크게 기여할 수 있을 것입니다.



### ALVI Interface: Towards Full Hand Motion Decoding for Amputees Using sEMG (https://arxiv.org/abs/2502.21256)
Comments:
          6 pages, video demo: this https URL

- **What's New**: 본 연구에서는 상지 절단자를 위한 실시간 손 움직임 디코딩 시스템을 제시합니다. 이 시스템은 20자유도를 갖춘 손가락 관절 각도를 복원하며, sEMG 신호를 활용하여 정밀한 손 움직임을 구현할 수 있습니다. 특히, Transformer 기반의 EMG-모션 변환 모델과 실시간 피드백 모듈인 ALVI 인터페이스를 통합하여 사용자에게 향상된 제어 경험을 제공합니다.

- **Technical Details**: 시스템은 VR 기반 데이터 수집 플랫폼, EMG 신호에서 손 움직임으로 변환하는 Transformer 모델, 및 실시간 보정 및 피드백 모듈로 구성됩니다. 특정 기술적 요소로는 손의 동작을 음성으로 인식하기 위해 8개의 sEMG 센서를 사용하고, 사용자와 상호작용하는 VR 환경을 통해 데이터의 수집 및 처리를 동기화합니다. 또한 HandFormer라는 Transformer 아키텍처를 사용하여 신호 처리를 최적화하고 있습니다.

- **Performance Highlights**: 시스템은 비절단자와 절단자를 포함하여 22명의 참가자를 대상으로 실험을 수행했으며, 절단자에 대해 0.80의 상관관계를 기록했습니다. 실시간 성능은 25Hz에서 운영되며 평균 지연 시간은 51.2ms로, 사용자들이 자연스러운 손 움직임을 VR 환경에서 수행할 수 있도록 지원합니다. 훈련 초기 10분 후, 사용자들은 신속하게 손가락을 정밀하게 제어할 수 있었으며, 시스템과 사용자가 서로 adaptative behavior를 나타내는 흥미로운 현상을 관찰했습니다.



### TimesBERT: A BERT-Style Foundation Model for Time Series Understanding (https://arxiv.org/abs/2502.21245)
- **What's New**: 본 논문에서는 다변량 시계열을 다문장 문서로 처리하는 Time Series Understanding에 관한 새로운 접근 방식을 제안합니다. 기존의 GPT 스타일 모델들이 시계열 예측에 적합하다고 여겨졌으나, BERT 스타일 아키텍처가 시계열 이해를 위해 충분히 활용되지 않았음을 강조합니다. 특히, 시계열의 고유한 다중 변수 구조를 반영한 TimesBERT 모델을 설계하였으며, 이는 다양한 다운스트림 작업에 우수한 성능을 보입니다.

- **Technical Details**: TimesBERT는 2600억 개의 시계열 데이터 포인트로 대규모 학습을 수행하며, 다변량 시계열의 시간적 패턴과 변수 중심 특성을 포함하는 일반적인 표현을 학습하도록 설계되었습니다. BERT의 기능적 토큰 예측 작업을 병행하여 다중 세분화 구조를 구현하고, 다변량 시계열을 문서처럼 모델링하여 기존의 데이터 구조로부터 유용한 정보를 추출합니다. 이를 통해 다변량 시계열의 내재적 구조적 표현을 유지합니다.

- **Performance Highlights**: TimesBERT는 4가지 일반적인 이해 작업 및 113개의 실제 데이터셋에서 기존의 최첨단 모델과 비교해 현저한 성능 향상을 달성하였습니다. 특히, 시간 시계열 분류, 결측치 보정, 이상 탐지 및 단기 예측 작업에서 우수성을 입증하며, 기존의 작업 특화 모델 및 언어 사전 학습 백본 모델들을 초월하는 성과를 보입니다. 이로써, TimesBERT는 고유한 시계열 이해의 기초 모델로 자리매김할 수 있게 됩니다.



### A Method of Selective Attention for Reservoir Based Agents (https://arxiv.org/abs/2502.21229)
Comments:
          6 pages, 2 figures

- **What's New**: 본 연구에서는 강화 학습(Deep Reinforcement Learning) 분야에서 입력 차원(input dimensions)의 불필요한 요소를 제거하여 학습 효율을 높이는 새로운 방법인 Excessively Parameterized Input Concealment (EPIC)을 제안합니다. 기존의 layer normalization 방법에 비해 EPIC을 적용한 경우 훈련 속도가 2배 증가하는 것을 확인했습니다. 이 연구는 보상 신호(reward signals)를 기반으로 하여 입력 마스킹(input masking) 방식이 강화 학습 효율에 미치는 영향을 분석하였습니다.

- **Technical Details**: 입력 마스킹은 강화 학습 에이전트의 메모리 표현(memory representation)에서 신호 대 잡음 비율(signal-to-noise ratio)을 증가시키는 중요한 역할을 하며, 이는 Echo State Network (ESN)이라는 재귀 신경망(rnn)을 사용하였습니다. 피실험적인 환경은 Multi-armed Bandit 문제를 기반으로 하여 Gaussian 노이즈로 복잡성을 추가했습니다. 여러 입력 차원에서의 훈련이 포함되며, layer normalization(레이어 정규화) 및 다른 모델과 비교됩니다.

- **Performance Highlights**: 실험 결과, EPIC을 사용하여 입력 마스킹을 적용한 경우, 평균적으로 훈련 시간 면에서 4배의 속도 향상을 보였으며, 이는 강화 학습 프로세스의 효율성을 크게 향상시킵니다. 입력 차원의 수가 많아질수록 EPIC의 성능이 더욱 두드러지며, 이는 더 많은 잡음을 처리할수록 효과적으로 학습할 수 있음을 시사합니다. 또한, 입력 마스킹을 통해 훈련 곡선(training curve)의 신뢰성이 개선되었습니다.



### Transformers Learn to Implement Multi-step Gradient Descent with Chain of Though (https://arxiv.org/abs/2502.21212)
Comments:
          ICLR 2025 Spotlight

- **What's New**: 이 논문에서는 Chain of Thought (CoT) 프롬프트가 대형 언어 모델(LLM)의 성능을 두드러지게 향상시킨다는 점에 집중합니다. 특히, 수학적 계산 및 추론 작업에서 CoT가 중간 추론 단계를 생성하도록 모델에 지시함으로써 얻은 성과를 보여줍니다. CoT의 훈련 동역학을 탐구하며, 그 기초 메커니즘이 대부분 미개척되어 있음을 강조합니다.

- **Technical Details**: 저자들은 CoT 목표에 대한 트랜스포머의 훈련 동역학을 연구했습니다. 이들은 선형 회귀의 컨텍스트 내 가중치 예측 작업을 통해, CoT가 없는 1계층 선형 트랜스포머는 단일 단계의 경량하강법(Gradient Descent, GD)만을 구현할 수 있음을 입증합니다. 반면 CoT 프롬프트를 사용한 트랜스포머는 다단계 GD를 자가 회귀적으로 학습하여 거의 정확하게 진실 가중치 벡터를 복원할 수 있음을 보여줍니다.

- **Performance Highlights**: 실험적으로 CoT 프롬프트가 상당한 성능 향상을 가져온 것을 증명하였습니다. 또한 훈련된 트랜스포머는 보지 못한 데이터에서도 효과적으로 일반화할 수 있습니다. 루프된 트랜스포머는 선형 회귀의 컨텍스트 내 학습에서 루핑이 없는 트랜스포머에 비해 최종 성능을 크게 개선합니다.



### Geodesic Slice Sampler for Multimodal Distributions with Strong Curvatur (https://arxiv.org/abs/2502.21190)
- **What's New**: 본 논문에서는 Hit-and-Run slice sampling 기법을 일반화하여 다양한 기하학에 맞춰 샘플링을 하는 새로운 방법을 제안합니다. 이 방법은 편미분 방정식의 해로 지오데식(geodesic)을 근사함으로써, 다중 모드 분포(multi-modal distributions)의 국소 탐색 문제를 해결합니다. 기존의 방법들은 일반적으로 닫힌 형태로 지오데식을 제공해야 했지만, 우리의 접근은 이를 확장하여 효율적인 샘플링이 가능합니다.

- **Technical Details**: 제안된 방법은 복잡한 기하학을 가진 분포에서 정확한 샘플링을 수행할 수 있도록 Riemannian geometry를 적용합니다. 샘플링 공간을 더 높은 차원의 Riemannian manifold로 매핑하여, 기준 분포의 기하학적 정보를 포함하는 방식으로 진행됩니다. 우리는 이 Riemannian sampler를 통해 대상 분포의 여러 모드를 효과적으로 탐색할 수 있는 메타 샘플러도 도입합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 복잡한 타겟 분포에서 Euclidean 방법에 비해 개선된 샘플링 효율성을 보여주었습니다. 특히 높은 차원에서 여러 모드를 혼합하는 성능이 증진되어, parallel tempering 및 diffusive Gibbs sampler와 비교할 때 뛰어난 탐색 능력을 입증하였습니다. 우리의 알고리즘은 탐색과 혼합에서 좋은 성과를 내지만, 지오데식을 계산하는 과정에서 느린 반복 속도를 가지는 것이 특징입니다.



### SYN-LUNGS: Towards Simulating Lung Nodules with Anatomy-Informed Digital Twins for AI Training (https://arxiv.org/abs/2502.21187)
Comments:
          6 figures, 12 pages

- **What's New**: 이번 연구에서는 랜덤한 데이터를 사용하여 다양한 폐 결절을 생성하는 SYN-LUNGS라는 프레임워크를 새롭게 도입했습니다. 이 프레임워크는 고품질의 3D CT 이미지를 생성하여 AI 모델의 훈련을 개선합니다. 특히, XCAT3 팬텀을 활용해 디지털 인간 쌍둥이를 생성하고, X-Lesions 도구를 통해 다양한 크기의 결절을 시뮬레이션하여 고유한 데이터 세트를 구성하였습니다.

- **Technical Details**: SYN-LUNGS는 물리적 및 해부학적 정보를 통합하여 폐 CT 이미지를 생성하는 방법론입니다. XCAT3를 기반으로 한 디지털 인간 모델과 절차적인 병변 모델링을 통해 해부학적 적합성을 높이고, DukeSim을 적용하여 실제 CT 촬영의 물리적 변수를 시뮬레이션합니다. 결과적으로 1,044개의 CT 스캔에서 3,072개의 결절 이미지를 포함하는 포괄적인 데이터 세트를 만들어냈습니다.

- **Performance Highlights**: SYN-LUNGS로 훈련된 모델은 임상 데이터만으로 훈련된 모델보다 10% 향상된 검출 성능을 보였습니다. 세분화 및 분류에서도 2-9%의 향상이 이루어졌으며, 특히 드문 질병의 표현과 모델의 신뢰성을 확보하는 데 기여했습니다. 이러한 성과는 AI 모델의 생성 및 신뢰성 향상에 중요한 진전을 나타냅니다.



### Scalable Decision-Making in Stochastic Environments through Learned Temporal Abstraction (https://arxiv.org/abs/2502.21186)
Comments:
          Accepted by ICLR2025. Code would be available at \href{this https URL}{this https URL}

- **What's New**: 본 논문은 고차원 연속 행동 공간에서의 순차적인 의사결정 문제를 다룹니다. 특히, 전통적인 Offline RL 환경에서 에이전트가 확률적 행동 정책을 바탕으로 의사결정을 학습해야 하는 복잡한 도전 과제를 해결하는 방안을 제시합니다. 이를 위해 저자들은 Latent Macro Action Planner (L-MAP)라는 새로운 접근법을 도입하여, 상태 조건적 Vector Quantized Variational Autoencoder (VQ-VAE)를 활용해 행동 차원을 효과적으로 감소시킵니다.

- **Technical Details**: L-MAP은 잠재적 전이 모델로 작동하는 별도의 학습된 사전(prior) 모델을 사용하여 plausible한 행동을 효율적으로 샘플링합니다. 이 접근법은 Monte Carlo Tree Search (MCTS)를 통해 환경과 행동 정책의 확률성을 고려하여 계획을 수립하며, 고차원 행동 공간에서의 의사결정 절차에서 이전 모델들이 처리했던 비효율성을 크게 줄입니다. L-MAP은 디스크리트한 잠재 행동을 탐색하여 높은 기대 수익을 제공하는 방식으로 작동합니다.

- **Performance Highlights**: 실험 결과, L-MAP는 고차원 행동 공간과 본래 확률적 동역학을 포함한 다양한 과제에서 기존 모델 기반 방법을 초월하며, 강력한 모델 없는 Actor-Critic 기반 모델과도 동등한 성능을 보였습니다. 특히, L-MAP은 결정 지연을 줄이면서도 높은 성능을 유지하여, 스토캐스틱한 환경에서의 의사결정에 있어 탁월한 효과성을 입증합니다.



### Reducing Reward Dependence in RL Through Adaptive Confidence Discounting (https://arxiv.org/abs/2502.21181)
- **What's New**: 본 연구는 비싼 보상을 요구하는 인간-참여 강화학습 환경에서 학습 효율성을 극대화하기 위한 알고리즘을 제안합니다. 제안된 알고리즘은 환경 상태에서 행동 가치에 대한 지식이 낮을 때만 보상을 요청하며, 이는 학습 속도를 높이는 데 기여합니다. 이러한 접근 방식을 통해 우리는 보상의 의존도를 줄이면서도 정책의 품질을 유지할 수 있도록 하였습니다.

- **Technical Details**: 우리의 알고리즘은 학습자의 신뢰도를 측정하기 위해 모델의 출력 분포를 활용합니다. 초기에는 상대적으로 밀집된 보상 신호를 사용하고, 이후 학습 모델이 개선됨에 따라 보상 신호를 희소하게 조정합니다. 두 가지 신뢰도 측정 방법(행동 모델의 엔트로피와 보상 모델의 엔트로피)을 비교하고, 다양한 환경에서 실험을 통해 성능을 평가하였습니다.

- **Performance Highlights**: 우리의 접근법은 기본선 알고리즘과 비교할 때 누적 보상 면에서 동등한 성과를 달성하였으며, 학습에 필요한 보상을 20%로 줄일 수 있었습니다. 이는 보상을 줄이면서도 안정적인 학습을 보장할 수 있음을 시사합니다. 실험 결과는 제안된 방법이 실제 문제 해결에 효과적임을 보여줍니다.



### QFAL: Quantum Federated Adversarial Learning (https://arxiv.org/abs/2502.21171)
Comments:
          10 pages

- **What's New**: 이 논문은 양자 연합 학습(Quantum Federated Learning, QFL)에서 적대적 훈련(adversarial training)을 통합한 새로운 프레임워크인 양자 연합 적대적 학습(Quantum Federated Adversarial Learning, QFAL)을 제안합니다. QFAL은 클라이언트들이 로컬 적대적 예제 생성을 결합하여 협력적으로 방어하는 방식을 채택합니다. 실험을 통해 클라이언트 수, 적대적 훈련 커버리지, 공격 강도 간의 상호작용을 평가하였습니다.

- **Technical Details**: 연합 학습(Federated Learning, FL)의 기본 개념은 데이터 중앙화 없이 여러 클라이언트가 협력하여 머신 러닝 모델을 훈련하는 것입니다. 효율적인 양자 신경망(Quantum Neural Networks, QNNs)을 통해 이러한 FL의 적용 가능성이 확대됩니다. 본 연구에서는 클라이언트 수, 적대적 훈련 비율, 공격 강도를 체계적으로 vary하여 QFAL의 견고성과 확장성을 평가했습니다.

- **Performance Highlights**: 실험 결과에 따르면, 적은 수의 클라이언트를 사용할 때 청정 데이터 정확도가 높지만, 더 큰 연합은 정확성과 견고성을 효과적으로 균형을 이룹니다. 부분 적대적 훈련(예: 20%-50%)이 매우 유용하며, 이는 적당한 강도의 공격에 대해 내성을 높이는 데 기여할 수 있습니다. 모든 클라이언트에서 완전한 적대적 훈련(100%)을 따르면 청정 정확도를 회복할 수 있지만, 강한 공격에 취약해질 수 있음을 밝혔습니다.



### Autonomous Curriculum Design via Relative Entropy Based Task Modifications (https://arxiv.org/abs/2502.21166)
- **What's New**: 이 논문에서는 자율적 커리큘럼 설계(autonomous curriculum design)를 위한 새로운 방법인 READ-C를 제안합니다. 이 방법은 에이전트(Agent)의 불확실성을 이용하여 학습 효율을 높이고, 인간의 개입 없이도 커리큘럼을 생성하는데 중점을 두고 있습니다. READ-C는 상대 엔트로피(relative entropy)를 통해 에이전트의 정책 간의 차이를 측정하며, 불확실성이 높은 상태(high uncertainty states)에서 학습을 촉진합니다.

- **Technical Details**: READ-C는 에이전트의 정책 표현에서 파생된 확률 값과 참 정책(true policy)에서 파생된 확률 값을 비교하여 비율 엔트로피(relative entropy)를 측정합니다. 이를 통해 에이전트의 상태 공간(state space)에서 불확실성이 높은 영역을 선택하고, 각 커리큘럼 단계에서 시작 상태를 조정하여 마르코프 결정 과정(MDP)을 수정합니다. 이 알고리즘은 두 가지 버전인 READ-C-TD(교사 의존)와 READ-C-SA(자체 평가)를 제공하며, 후자는 기존 정책에 대한 의존성을 줄이는 방법입니다.

- **Performance Highlights**: READ-C는 무작위로 생성된 커리큘럼 및 기존 커리큘럼 학습 기준과 비교하여, 여러 분야에서 성능이 우수함을 입증하였습니다. 또한, 알고리즘의 수렴(convergence) 보증을 제공하며, 에이전트의 성능 향상을 위해 불확실성이 높은 영역에서의 학습을 장려하는 등, 학습 효율성을 높이는 데 기여합니다. 결과적으로 READ-C는 기존 알고리즘과 유사한 성과를 보이면서도 커리큘럼 생성 오버헤드를 줄입니다.



### Parallel-Learning of Invariant and Tempo-variant Attributes of Single-Lead Cardiac Signals: PLITA (https://arxiv.org/abs/2502.21162)
Comments:
          Published in The 39th Annual AAAI Conference on Artificial Intelligence. Main Track

- **What's New**: 본 논문에서는 단일 리드 심전도(ECG) 신호의 불변(invariant) 속성과 시간 변화(tempo-variant) 속성을 모두 포착하기 위해 설계된 새로운 자기 지도 학습(Self-Supervised Learning, SSL) 방법인 PLITA(Parallel-Learning of Invariant and Tempo-variant Attributes)를 소개합니다. 기존의 SSL 방법은 주로 불변 속성에만 집중함으로써 시간 변화한 정보는 간과하고 있습니다. PLITA는 직접적으로 시간 변화 속성을 모델링하여 비슷한 시간의 입력들이 가까운 표현을 갖도록 강제함으로써, 더욱 유의미한 ECG 분석을 가능하게 합니다. 이는 디지털 건강의 미래에서 웨어러블 센서가 갖는 중요성을 더욱 부각시킵니다.

- **Technical Details**: 제안된 PLITA 방법은 불변 속성과 tempo-variant 속성을 동시에 고려하여 학습할 수 있도록 설계되었습니다. 여기서는 Tempo-variant Loss Function(ℒtv)을 통해 서로 시간적으로 인접한 입력의 표현이 가까워지도록 하는 원칙을 적용합니다. 플리타는 악성 부정맥, 수면 단계 분류 및 성별 식별 작업에서 이러한 두 가지 속성을 모델이 효과적으로 캡처해야 한다고 주장합니다. 이는 심전도 데이터의 본질을 이해하는데 필수적이며, 분석 결과 신뢰할 수 있는 결과를 제공합니다.

- **Performance Highlights**: PLITA 방법은 전통적인 SSL 방법에 비해 심전도 분석에서 훨씬 더 우수한 성능을 보였습니다. tempo-variant 특징들이 중요한 역할을 하는 실험 세트 업에서 현저한 성능 향상이 관찰되었습니다. 또한, 성별 식별 작업에서도 강력한 결과를 보였으며, 이러한 결과는 불변 속성이 응용 프로그램에서 제대로 작용하고 있음을 확인시켜 줍니다. 결국, 본 연구는 ECG 데이터에 대한 새로운 접근 방식을 제시하고 있으며, 이는 향후 SSL 연구의 방향성을 제시할 것입니다.



### Same accuracy, twice as fast: continuous training surpasses retraining from scratch (https://arxiv.org/abs/2502.21147)
- **What's New**: 이 논문은 기존의 데이터로 훈련된 모델을 활용하여 새로운 데이터에 대한 훈련 비용을 줄이는 방법을 제시합니다. 기존 모델을 사용함으로써 새로운 데이터에서도 성능을 유지하거나 향상시킬 수 있으며, 이를 통해 리소스를 절약할 수 있습니다. 특히, 모델과 데이터 크기가 커질수록 'from scratch' 방식의 훈련이 비효율적이라는 점을 강조합니다.

- **Technical Details**: 논문은 SGD 업데이트 규칙을 기반으로 초기화, 정규화(regularization), 데이터 선택(data selection), 하이퍼 파라미터(hyper-parameters)와 같은 최적화 측면을 통해 계산 비용을 크게 줄일 수 있는 방안을 모색합니다. 이러한 각 측면에 대해 효율적인 첫 단계 방법(first-step methods)을 제안하고, 이를 통해 모델의 학습 속도를 향상시킬 수 있음을 보입니다. 더욱이, 이러한 방법들은 서로 보완적이어서 결합했을 때 더 큰 계산 절약 효과를 얻을 수 있습니다.

- **Performance Highlights**: 제안된 방법들은 여러 컴퓨터 비전 작업에서 최대 2.7배의 계산 시간 절약을 보여주며, 이는 재훈련이 반복적으로 이루어지는 경우 의미 있는 차이를 생성합니다. 또한 다양한 이미지 분류 데이터셋과 다중 작업(multi-task) 환경 및 도메인 점진적 시나리오에서 훈련 효율성 향상을 입증했습니다. 이러한 결과는 제안된 방법이 머신러닝 모델 재훈련의 계산 부담을 줄일 수 있는 잠재력을 가지고 있음을 강조합니다.



### Variational Bayesian Pseudo-Cores (https://arxiv.org/abs/2502.21143)
Comments:
          The Thirteenth International Conference on Learning Representations (ICLR2025)

- **What's New**: 딥러닝의 성공은 방대한 데이터셋과 철저한 훈련에 의존하지만, 이는 상당한 계산적 도전을 야기합니다. 이러한 문제를 해결하기 위해, 전체 데이터를 모방하는 작은 학습 가능한 데이터셋인 pseudo-coresets가 제안되었습니다. 본 논문에서는 Variational Bayesian Pseudo-Coreset (VBPC)라는 새로운 접근법을 소개합니다. 이는 변분 추론(Variational Inference)을 활용하여 메모리 사용을 줄이고 계산 비용을 개선하며, 성능을 향상시키는 방법입니다.

- **Technical Details**: Bayesian Neural Networks (BNNs)는 고차원 매개변수 공간으로 인해 대규모 데이터셋을 처리할 때 상당한 문제에 직면합니다. 기존의 Bayesian Pseudo-Coreset (BPC) 방법은 작은 합성 데이터셋을 사용하여 BNN의 가중치 후방 분포를 효율적으로 계산하려고 하지만, 메모리 비효율성과 하위 최적화 결과 같은 문제점이 존재합니다. 본 연구에서는 가중치 분포를 근사하기 위해 변분 추론을 사용하고, 마지막 층의 가중치에 대한 폐쇄 형태의 후방 분포를 제공하는 방법을 제안합니다.

- **Performance Highlights**: VBPC는 여러 벤치마크 데이터셋에서 기존 방법들보다 더 나은 성능을 달성하는 것으로 나타났습니다. 이 방법은 메모리 사용을 감소시키고 계산 효율성을 높이며, 단일 포워드 패스를 통해 예측 분포를 근사하는 메모리 효율적인 방법을 제공합니다. 실험적으로 VBPC는 기존의 BPC 방법보다 더 나은 결과를 보였으며, 이는 더욱 효과적인 연속 학습을 가능하게 합니다.



### Predicting clinical outcomes from patient care pathways represented with temporal knowledge graphs (https://arxiv.org/abs/2502.21138)
- **What's New**: 이 논문은 뇌동맥류( intracranial aneurysm )의 예후를 예측하기 위해 그래프 임베딩(Graph Embedding)과 그래프 신경망(Graph Convolutional Network, GCN)을 이용하는 방법을 탐구합니다. 특히, 환자의 개인적 특성과 병원에서 관찰된 치료 정보를 결합하여 adverse outcomes를 예측하는 방법론을 제시하고 있습니다. 데이터는 합성으로 생성된 것이지만 실제 환자의 경과를 반영하였으며, tabular 데이터와 그래프 기반 표현을 비교하여 최적의 예측 성능을 도출했습니다.

- **Technical Details**: 연구에서는 그래프의 구성 요소인 노드와 간선을 연속 벡터 공간으로 매핑하는 그래프 임베딩을 사용하여 데이터 분석을 수행합니다. Knowledge Graph(KG)에서 정의된 임베딩 기법을 적용하여 환자 데이터를 표현하고, 특히 GCN을 이용해 노드 간의 이웃 정보를 활용하여 예측 성능을 높이는 데 중점을 두었습니다. 또한 개인 데이터 표현과 시간 데이터 표현의 모델링 방식이 예측 성능에 미치는 영향을 실험하였습니다.

- **Performance Highlights**: 실험 결과, GCN 임베딩을 이용한 그래프 표현이 tabular 데이터 기반의 전통적 분류 기법보다 우수한 성능을 보임을 확인했습니다. 연구의 결과는 환자 특성의 컴팩트한 표현이 예측 성능에 유의미한 긍정적 영향을 미친다는 것을 보여주며, 시간 표현 방식이 예측 성능에는 큰 영향을 미치지 않는다는 점도 강조되었습니다. 이는 생물의학 예측 모델링에서 그래프 기반 접근 방식의 가능성을 제시합니다.



### CuPID: Leveraging Masked Single-Lead ECG Modelling for Enhancing the Representations (https://arxiv.org/abs/2502.21127)
Comments:
          Paper under review

- **What's New**: 이 논문에서는 단일 리드 ECG(심전도) 데이터에 최적화된 새로운 Masked Data Modelling (MDM) 방법인 Cueing the Predictor Increments the Detailing (CuPID)을 소개합니다. CuPID는 인코더가 더 세부적인 표현을 생성하도록 유도하기 위해 스펙트로그램에서 유도된 맥락을 디코더에 제공함으로써 기존 MDM 기법을 향상시킵니다. 이 접근 방식은 ECG 데이터에서 디코더의 성능을 크게 향상시킵니다.

- **Technical Details**: CuPID는 입력 신호의 스펙트로그램을 Key로 사용하는 attention 메커니즘을 디코더에 통합하여 단일 리드 ECG의 불규칙한 심장 박동 간격에 대한 문제를 해결합니다. 스펙트로그램은 12 리드 ECG 프레임워크에서 다른 리드에서 제공되는 맥락 정보를 반영하여 변화하는 시간을 기반으로 한 표현을 제공합니다. 이 과정은 손실 함수 값을 유의미하게 감소시키고, 원래 신호의 형태에 더 잘 맞도록 재구성됩니다.

- **Performance Highlights**: CuPID는 MIT-BIH Atrial Fibrillation, Physionet Challenge 2017, Long Term AF 데이터베이스에서의 평가에서 기존 최첨단 Self-Supervised Learning (SSL) 방법들과 비교하여 뛰어난 성능을 제공합니다. CuPID의 성능은 단일 리드 ECG 분석을 위해 고안된 방법들에 비해 상당히 우수하며, 스펙트로그램을 포함한 결과는 MTAE 기준과 비교했을 때도 장점을 보입니다.



### Causality Is Key to Understand and Balance Multiple Goals in Trustworthy ML and Foundation Models (https://arxiv.org/abs/2502.21123)
- **What's New**: 최근 기계 학습(ML) 시스템의 신뢰성을 보장하는 것이 중요해졌습니다. 이 논문은 형평성(fairness), 개인 정보 보호(privacy), 견고성(robustness), 정확성(accuracy), 설명 가능성(explainability) 등의 원칙 간의 균형을 위해 인과적(causal) 방법의 통합을 주장합니다. 이를 통해 서로 상충하는 목표를 해결하고, 새로운 해결책을 제시합니다.

- **Technical Details**: 이 논문에서는 인과성(causality) 모델을 통해 신뢰할 수 있는 ML 시스템의 다양한 차원을 조화롭게 통합할 수 있는 방법을 제안합니다. Pearl의 구조적 인과 모델(Structural Causal Models, SCMs)을 사용하여 인과성을 정의하고, 유향 비순환 그래프(directed acyclic graphs, DAGs)를 통해 인과 종속성을 표현합니다. 이러한 접근법은 데이터의 근본적인 메커니즘을 파악하는 데 도움을 줍니다.

- **Performance Highlights**: 인과적 방법론은 형평성과 정확성 또는 개인 정보 보호와 견고성과 같은 서로 대립하는 목표를 성공적으로 조화시켜 ML의 신뢰성을 향상시키는 데 기여할 수 있습니다. 논문은 미래 연구 방향과 제약에 대해서도 논의하며, 신뢰할 수 있는 AI 시스템을 구축하기 위한 더 윤리적이고 책임감 있는 접근 방식을 모색합니다.



### Rare event modeling with self-regularized normalizing flows: what can we learn from a single failure? (https://arxiv.org/abs/2502.21110)
Comments:
          Published at ICLR 2025

- **What's New**: 이 논문에서는 자율 시스템의 증가에 따른 안전 문제를 해결하기 위해 CalNF(calibrated normalizing flows)라는 새로운 방식의 프레임워크를 제안합니다. 이는 제한된 데이터에서 포스터리어( posterior) 학습을 위한 자기 조정된 방법으로, 데이터가 부족한 상황에서도 뛰어난 성능을 보입니다. CalNF는 2022년 Southwest Airlines 스케줄링 위기의 근본 원인을 분석하는 데에도 성공적으로 적용되었습니다.

- **Technical Details**: CalNF는 데이터 제약 환경에서 희귀 사건 모델링을 다양한 방식으로 해결합니다. 이 방법은 정규화 흐름(normalizing flows)과 변분 추론(variational inference) 기술을 결합하여 명확하지 않은 포스터리어 분포를 근사하는 데 사용됩니다. 또한, CalNF는 허용된 수의 관찰값에서 저차원 임베딩을 학습하고, 이를 통해 대상 포스터리어를 최적화하는 데 필요한 규제를 제공합니다.

- **Performance Highlights**: CalNF는 자율 시스템의 실제 데이터로 이루어진 여러 데이터 제약 추론 벤치마크에서 최첨단 성능을 달성했습니다. 특히, 2022년 Southwest Airlines 사건을 분석하는 데 있어 강화된 기능을 보여주어, 사건이 어떻게 시스템 전반에 걸쳐 확산되었는지를 이해하는 데 도움을 주었습니다. 이를 통해 연구의 실용성과 적용 가능성이 더욱 부각되었습니다.



### Efficient Transformer-based Decoder for Varshamov-Tenengolts Codes (https://arxiv.org/abs/2502.21060)
Comments:
          9 pages, 2 figures, 9 tables

- **What's New**: 최근 DNA 데이터 저장 기술의 발전으로 인한 IDS(삽입, 삭제, 대체) 오류 수정 문제에 대한 관심이 높아졌다. 본 논문에서는 주로 단일 오류 수정용으로 설계된 Varshamov-Tenengolts(VT) 코드의 다중 오류 수정 가능성을 탐구하고, transformer 기반의 VT decoder(TVTD)를 제안한다. TVTD는 symbol-과 statistic 기반 코드워드 임베딩을 사용하여 높은 오류 수정 성능을 보인다.

- **Technical Details**: VT 코드는 이론적으로 단일 IDS 오류를 수정하기 위해 적은 비트 수를 요구하는 고효율 오류 수정 스킴으로, 복잡한 관계를 캡처할 수 있는 transformer 아키텍처를 활용하여 다중 오류를 처리한다. 실험 결과에 따르면 TVTD는 기존의 하드 디시전(hard decision) 및 소프트인 소프트아웃(soft-in soft-out) 알고리즘과 비교하여 비트 오류율(BER)과 프레임 오류율(FER)을 크게 개선하였다. 또한, 모델 아키텍처 최적화를 통해 TVTD는 다른 소프트 디코더들에 비해 시간 소모를 10배 줄였다.

- **Performance Highlights**: TVTD는 비교적 긴 코드워드에서도 최고의 실험 결과를 달성했으며, 코드워드 길이가 증가함에 따라 BER과 FER이 각각 2%에서 20%, 20%에서 86%까지 감소하였다. 훈련 속도는 전체 transformer 모델과 비교해 40% 증가했으며, 실제 테스트에서 TVTD의 디코딩 속도는 SISO 디코딩에 비해 5배 이상 증가했다. 특히 긴 코드워드의 경우 디코딩 속도가 46배 이상 향상되어 대규모 데이터 저장 응용 프로그램에 큰 잠재력을 보여준다.



### Detection of anomalies in cow activity using wavelet transform based features (https://arxiv.org/abs/2502.21051)
Comments:
          17 pages, 8 figures, 4 tables, 1 algorithm

- **What's New**: 정밀 축산 농업(Precision Livestock Farming)에서 최적 값 또는 기준 값의 편차를 감지하는 것이 필수적입니다. 이 논문에서는 소의 활동 데이터를 24시간 동안 분석하여 질병이나 발정 상태를 조기에 감지하는 방법을 제안합니다. 특히, 생물학적 데이터의 경우 노이즈를 잘 구분하는 것이 중요하며, 파동 변환(Wavelet Transform)을 이용하여 데이터를 정제하는 접근 방식이 주목받고 있습니다.

- **Technical Details**: 본 연구에서는 소의 활동 데이터에 대한 Anomaly Detection 알고리즘의 성능을 평가하였으며, 이는 생리적 또는 병리적 상태에 따라 발생하는 편차를 식별하는 데 중점을 두었습니다. 우리는 파동 변환의 평균과 개별 시간 시리즈의 파동 변환을 비교하여 특징을 개발했습니다. Isolation Forest 알고리즘을 활용하여 이러한 특성과 통계적 특징을 활용한 결과, 파동 변환 기반의 특징이 Anomaly Detection에 중요한 기여를 한다고 판단하였습니다.

- **Performance Highlights**: 결과적으로, 파동 변환을 적용한 소의 행동 데이터 분석이 질병 및 발정 상태에 관련된 이상 징후를 조기에 발견하는 데 효과적임을 나타냈습니다. 알고리즘이 예측한 이상 상태가 수의사가 이상 징후를 눈으로 인지하기 전의 날에 발생하는 경우가 많아, 농부가 빠르게 대응할 수 있는 가능성을 제공합니다. 연구 결과, 파동 변환 기반의 특징들이 Anomaly Detection에서 가장 기여를 크게 하였습니다.



### Fast Adversarial Training against Sparse Attacks Requires Loss Smoothing (https://arxiv.org/abs/2502.21041)
- **What's New**: 이 논문은 $l_0$ 노름에 의해 제한된 희소(side-by-side sparse) 적대적 perturbations에 대한 빠른 적대적 훈련 방법을 연구합니다. 기존의 1단계 공격을 사용할 때 발생하는 성능 저하와 치명적인 과적합(cO) 문제를 강조합니다. 특히, $l_0$ 적대적 훈련에서 cO는 1단계 공격의 최적화되지 않은 perturbation 위치에 의해 초래됩니다.

- **Technical Details**: 이 연구는 $l_0$ 적대적 훈련의 손실 경관(loss landscape)이 다른 노름인 $l_	ext{infinity}$, $l_2$, $l_1$에 비해 더욱 억세고(craggy), 이는 치명적인 과적합(cO)을 악화시켜 성능 저하를 초래하는 것을 이론적 및 실증적으로 분석합니다. 논문에서 제안한 Fast-LS-$l_0$는 소프트 레이블(soft labels)과 trade-off 손실 함수를 포함하여 적대적 손실 경관을 매끄럽게 만들어 cO 문제를 해결합니다.

- **Performance Highlights**: 광범위한 실험 결과는 제안된 방법이 치명적인 과적합 문제를 극복하고 최첨단의(state-of-the-art) 성능을 달성할 수 있음을 보여줍니다. 1단계 적대적 훈련과 다단계(multi-step) 적대적 훈련 사이의 성능 격차를 효과적으로 축소함으로써, 이 연구는 희소 공격(sparse attacks)에 대한 효율적인 적대적 훈련의 새로운 최첨단 성능을 확립합니다.



### Reward Learning from Multiple Feedback Types (https://arxiv.org/abs/2502.21038)
Comments:
          Published as a conference paper at ICLR 2025

- **What's New**: 이 논문은 다양한 유형의 피드백을 사용하여 강화 학습(Reinforcement Learning, RL)에서의 보상을 학습하는 새로운 접근 방식을 소개합니다. 특히, 기존의 이진 비교(preference-based feedback) 방법 대신, 다양한 피드백 타입을 통해 더 효과적인 인간 피드백을 수집하는 방법을 제안합니다. 다양한 피드백 유형을 실험하고 평가하여 이러한 방식의 효과성을 입증한 첫 번째 연구로, 차별화된 피드백이 보상 모델링에 긍정적인 영향을 미칠 수 있다는 가능성을 보여줍니다.

- **Technical Details**: 본 연구에서는 여섯 가지 다른 유형의 피드백에 대한 고품질 시뮬레이션 피드백을 생성하는 프로세스를 개발하였습니다. 각 피드백 유형에 대해 보상 모델(reward models)과 후속 강화 학습 훈련을 실행하며, 이를 통해 10개의 다양한 RL 환경에서 피드백 유형을 비교하고 순수 이진 비교 기준선(preference-based baselines)과의 성능 차이를 분석합니다. 이를 통해 여러 피드백 타입을 동시에 활용할 수 있는 방법론을 제시합니다.

- **Performance Highlights**: 다양한 피드백 유형이 강화 학습에서 강력한 보상 모델링 성능을 이끌어낼 수 있음을 경험적으로 입증하였습니다. 이 연구는 다중 피드백 유형이 RLHF(Reinforcement Learning from Human Feedback)에서의 잠재력을 강하게 나타내는 것을 강조합니다. 결과적으로, 다양한 피드백의 동시 사용이 동일한 트레이닝 환경에서도 성능 개선을 가져올 수 있다는 점을 발견하였습니다.



### S4ConvD: Adaptive Scaling and Frequency Adjustment for Energy-Efficient Sensor Networks in Smart Buildings (https://arxiv.org/abs/2502.21035)
Comments:
          Submitted to TOSN Journal

- **What's New**: 이 연구에서는 S4ConvD라는 새로운 컨볼루션 변형을 도입하여 스마트 빌딩의 에너지 소비 예측을 개선합니다. S4ConvD는 기존의 데이터 전처리에 대한 의존도를 줄이고, 자원 제약이 있는 환경에서의 효율성을 최적화할 수 있도록 설계되었습니다. 이 모델은 적응형 스케일링과 주파수 조정을 통해 복잡한 시간적 패턴을 포착하며, ASHRAE Great Energy Predictor III 데이터셋 실험에서 경쟁 모델과 비교하여 우수한 성능을 입증했습니다.

- **Technical Details**: S4ConvD는 다이나믹하게 파라미터화된 상태 행렬과 적응형 입력 변환을 도입하여 S4D 프레임워크를 확장한 모델입니다. 주파수 성분의 최적화된 처리를 통해 관련 시간적 패턴을 강조하고, 정적 컨볼루션 방법보다 더 나은 성능을 발휘합니다. CUDA 메모리 최적화 기술인 Block Tiling을 활용하여 현대 GPU 아키텍처에서 36%의 런타임 개선을 보여줍니다.

- **Performance Highlights**: S4ConvD는 다양한 빌딩 유형에 걸쳐 뛰어난 일반화 성능을 보이며, 경쟁 모델을 상회하는 결과를 보여줍니다. 이는 특정한 빌딩 유형에 맞추어진 기존 접근 방식의 한계를 극복하는 데 기여할 것으로 예상됩니다. 또한, GitHub를 통해 코드베이스와 데이터셋을 공개하여 향후 연구와 오픈 소스 기여를 촉진하는 데 도움을 줍니다.



### Synthesizing Tabular Data Using Selectivity Enhanced Generative Adversarial Networks (https://arxiv.org/abs/2502.21034)
Comments:
          This thesis submitted to the University of Melbourne for partial fulfillment of the degree of Master of Data Science

- **What's New**: 이 논문에서는 주요 쇼핑 이벤트(예: Black Friday) 동안 증가하는 거래량에 대비하기 위해 E-commerce 플랫폼의 리소스 계획을 위한 새로운 방법을 제안합니다. 이전 연구는 Generative Adversarial Networks (GAN)를 이용하여 데이터 생성에 초점을 맞추었지만, 이러한 방법은 E-commerce 스트레스 테스트에 적합하지 않은 계산 요구 사항을 간과했습니다. 본 연구는 쿼리 선택성 제약(query selectivity constraints)을 포함한 새로운 GAN 기반 접근 방식을 소개합니다.

- **Technical Details**: 논문에서는 사전 훈련된 딥 신경망(pre-trained deep neural network)을 통합하여 실제 데이터와 합성 데이터 간의 선택성 일관성(selectivity consistency)을 유지하는 방법을 제시합니다. 데이터 처리 과정에서 쿼리 선택성 제약은 데이터베이스 트랜잭션 처리의 중요한 요소로, 본 연구의 핵심적인 기술적 요소입니다. 본 방법론은 5개의 실세계 데이터셋에서 테스트되었으며, 선택성 추정 정확도(selectivity estimation accuracy)와 머신러닝 유틸리티(machine learning utility)를 향상시켰습니다.

- **Performance Highlights**: 본 연구의 방법은 세 가지 최첨단 GAN 모델과 VAE 모델을 초월하여 성능을 입증합니다. 선택성 추정 정확도는 최대 20% 향상되었으며, 머신러닝 유틸리티는 최대 6% 향상되었습니다. 이 결과는 E-commerce 플랫폼에서 실질적인 데이터 처리에 있어 더 나은 성능을 제공할 수 있음을 보여줍니다.



### When Unsupervised Domain Adaptation meets One-class Anomaly Detection: Addressing the Two-fold Unsupervised Curse by Leveraging Anomaly Scarcity (https://arxiv.org/abs/2502.21022)
- **What's New**: 이 논문은 비지도적 이상 탐지 (Unsupervised Anomaly Detection, UAD) 를 위한 최초의 완전 비지도적 도메인 적응 (Unsupervised Domain Adaptation, UDA) 프레임워크를 소개합니다. 전통적인 이상 탐지 기법은 도메인 변경이 발생할 경우 성능 저하가 심각하게 발생하는데, 이는 실제 상황에서 피하기 어려운 문제입니다. 이 논문에서는 비지도적 문제의 두 가지 측면을 강조하며, 이상 데이터는 일반적으로 희귀하다는 아이디어를 활용하여 이 문제를 해결하는 혁신적인 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 우선 대상 특징 공간에서 주요 클러스터를 식별하고 이를 정상 원본 특징과 정렬합니다. 또한, ResNet 기반 특징 추출기를 사용하여 원본 및 대상 특징을 처리하며, CLIP 시각 인코더를 이용해 대응하는 대상 특징을 생성하고 K-평균 군집화를 통해 주요 클러스터의 샘플을 식별합니다. 이후 주요 클러스터의 샘플은 ResNet 기반 특징 공간으로 매핑되고 원본 특징과 정렬됩니다.

- **Performance Highlights**: 이 연구는 여러 표준 UDA 벤치마크에서 실험을 수행하여 제안된 방법의 효과성을 입증합니다. 또한, 제안된 접근법은 최신 수준의 (state-of-the-art, SoA) 성능을 달성했으며, 이는 기존 few-shot 방법들과 비교했을 때도 높은 성능을 제공합니다. 이 프레임워크는 모듈식으로 구성되어 있어 유연한 구성 요소 변경이 가능하며, 다양한 적응 전략을 지원합니다.



### Improving Open-world Continual Learning under the Constraints of Scarce Labeled Data (https://arxiv.org/abs/2502.20974)
- **What's New**: 이 논문은 오픈 월드 지속 학습(Open-world Continual Learning, OWCL)에서 희소 레이블 데이터(scarce labeled data)로 발생하는 문제를 탐구하고, 특히 새로운 범주가 제한된 주석과 소량으로 등장하는 현실적인 상황에서의 오픈 월드 소수 샷 지속 학습(Open-world Few-shot Continual Learning, OFCL)을 제안합니다. OFCL은 이전 지식 유지 및 과적합(overfitting) 방지, 레이블 데이터가 제한된 상황에서의 컴팩트한 결정 경계(compact decision boundaries) 구성, 그리고 레이블을 학습한 후에 미지의 샘플의 지식을 업데이트하는 것을 포함하는 도전 과제가 있습니다.

- **Technical Details**: 제안된 OFCL 프레임워크는 세 가지 핵심 구성 요소로 이루어져 있습니다: (1) 인스턴스 기반 토큰 증강(Instance-wise Token Augmentation, ITA)은 샘플 표현을 추가 지식으로 풍부하게 구성합니다. (2) 마진 기반 오픈 경계(Margin-based Open Boundary, MOB)는 새로운 작업이 시간에 따라 나타나는 오픈 감지를 지원합니다. (3) 적응 지식 공간(Adaptive Knowledge Space, AKS)은 미지의 샘플에 지식을 부여하여 미지의 것을 알려진 것으로 업데이트하도록 돕습니다.

- **Performance Highlights**: 광범위한 실험 결과는 OFCL 프레임워크가 모든 기준 방법과 비교해 현저한 성능 향상을 보이며, 실제 중요성과 재현성을 강조합니다. OWCL 접근법, 소수 샷 증분 학습(few-shot incremental learning) 방법 및 오픈 감지 기준과 비교했을 때, 제안된 OFCL 프레임워크가 그 효율성과 내구성을 입증하고 있습니다.



### Retrieval Augmented Generation for Topic Modeling in Organizational Research: An Introduction with Empirical Demonstration (https://arxiv.org/abs/2502.20963)
Comments:
          30 pages, 4 figures

- **What's New**: 이번 논문은 텍스트 데이터 분석의 새로운 접근법인 Agentic Retrieval-Augmented Generation (Agentic RAG)을 소개합니다. 기존의 LLM 기반 주제 모델링 방법이 가지고 있는 데이터 전처리의 어려움, 해석 가능성, 신뢰성 문제를 해결하기 위한 자동화된 방법으로 자리잡고 있습니다. 이 방법은 검색, 생성, 에이전트 기반 학습의 세 가지 주요 요소를 통합하여 효과적인 주제 모델링을 가능하게 합니다.

- **Technical Details**: Agentic RAG는 세 가지 구성 요소로 이루어져 있습니다: 첫 번째는 검색(retrieval)으로, LLM의 사전 학습 지식 외부에서 데이터에 접근할 수 있는 자동화된 방법을 제공합니다. 두 번째는 생성(generation)으로, LLM의 텍스트 합성 능력을 활용하여 데이터를 생성합니다. 세 번째는 에이전트 기반 학습(agent-driven learning)으로, 검색과 쿼리 구성 과정을 반복적으로 개선하여 더 나은 성과를 도출합니다.

- **Performance Highlights**: 논문은 Agentic RAG가 이전에 Mu et al. (2024a)가 분석한 Twitter/X 데이터셋에 대한 재분석을 통해 방법론이 효율적이고 해석 가능하며 높은 신뢰성과 타당성을 달성했음을 보여줍니다. 이는 전통적인 기계 학습 접근법에 비해나 LLM 프롬프트 기반 주제 모델링과 비교할 때도 우수한 성능을 입증합니다. 결과적으로 Agentic RAG는 리더십, 관리 및 조직 연구에서 AI 기반 질적 연구의 강력하고 확장 가능하며 투명한 대안으로 자리잡고 있습니다.



### Reward Dimension Reduction for Scalable Multi-Objective Reinforcement Learning (https://arxiv.org/abs/2502.20957)
Comments:
          Accepted to ICLR 2025

- **What's New**: 이 논문에서는 다목적 강화 학습(Multi-Objective Reinforcement Learning, MORL) 알고리즘의 확장성 문제를 해결하기 위해 간단하면서도 효과적인 보상 차원 축소 방법을 소개합니다. 기존 방법들이 두 개에서 네 개의 목표에 최적화되는 데 중점을 두었다면, 본 연구는 더 많은 목표를 가진 환경에 대한 효과적인 확장성에 중점을 둡니다. 새로운 훈련 및 평가 프레임워크를 제안하며, 원본 보상 공간에서 파레토 최적성을 유지하는 동시에 정책 성능을 향상시키는 방법을 보여줍니다.

- **Technical Details**: 다목적 마르코프 결정 과정(Multi-Objective Markov Decision Process, MOMDP)은 상태 집합, 행동 집합, 상태 전이 확률, 초기 상태 분포, 보상 함수와 할인 계수로 정의됩니다. MOMDP에서는 보상 함수가 벡터 값으로 주어져, 각 상태-행동 쌍에 대해 여러 개의 보상을 받을 수 있도록 설계됩니다. 우리의 방법은 온라인 학습 환경에서 작동하며, 학습 중에도 변환 후 파레토 최적성을 보장합니다.

- **Performance Highlights**: 본 연구는 16개의 목표를 포함한 환경에서도 기존 온라인 차원 축소 방법보다 우수한 성능을 발휘함을 입증하였습니다. 우리의 방법은 특히 목표의 수가 많을수록 중요한 장점을 제공하며, 실세계의 비즈니스 및 전략적 결정에서 즉각적인 적용 가능성을 지니고 있습니다. 이러한 뛰어난 성능은 복잡한 다차원 보상 공간을 효과적으로 축소하여 도출된 것임을 확인할 수 있습니다.



### Robust and Efficient Writer-Independent IMU-Based Handwriting Recognization (https://arxiv.org/abs/2502.20954)
- **What's New**: 이 논문은 IMU 데이터(비관성 측정 단위)를 활용한 온라인 필기 인식(HWR) 모델을 제시합니다. CNN(합성곱 신경망) 기반의 인코더와 BiLSTM(양방향 장기 단기 메모리) 디코더 구조를 사용하여 다양한 길이의 입력을 처리할 수 있습니다. 이 방법은 기존의 방법들보다 뛰어난 정확도와 데이터 효율성을 보여줍니다. 특히, 다양한 연령대와 필기 조건을 아우르며, 제한된 데이터로도 효과적으로 학습할 수 있는 시스템을 개발하는 데 기여합니다.

- **Technical Details**: 이 모델은 필기 스타일 변화 및 센서 노이즈 관련 문제들을 다루기 위해 시퀀스-투-시퀀스 모델을 사용합니다. CNN과 BiLSTM을 결합한 인코더-디코더 구조를 통해 입력 특성을 효과적으로 추출하고 문맥 정보를 양방향에서 캡처합니다. 데이터 수집은 STABILO의 IMU 기반 펜을 통해 이루어졌으며, 다양한 연령대 및 사용자 특성에 따라 54,666개의 샘플을 포함하는 데이터셋이 사용되었습니다.

- **Performance Highlights**: 모델 성능은 다양한 필기 스타일과 조건에서 높은 정확도를 유지하며, 작가 독립(WI) 데이터셋에서 기존 모델들보다 우수한 결과를 보였습니다. 또한, 아동과 성인 그룹에서 모델의 성능을 평가하였고, 이를 통해 성과의 안정성을 입증했습니다. 최종적으로 이 연구는 IMU 기반 온라인 HWR 분야에서의 새로운 가능성을 제시하며, 깊이 있는 설계 선택과 성능 간의 균형을 달성하였습니다.



### Concealed Adversarial attacks on neural networks for sequential data (https://arxiv.org/abs/2502.20948)
- **What's New**: 이번 연구에서는 다양한 시계열 모델에 대해 숨겨진 적대적 공격(concealed adversarial attack)을 제안합니다. 이 공격은 보다 사실적인 교란을 제공하여 인간이나 모델 탐지기(discriminator)에 의해 쉽게 감지되지 않도록 설계되었습니다. 공격의 강도를 높이기 위해 훈련된 탐지기를 통한 접근법도 도입되어 다양한 방어 기법에 대응할 수 있는 폭넓은 커버리지를 제공합니다.

- **Technical Details**: 제안된 공격 방법은 분류기(classifier)와 훈련된 탐지기 손실(discriminator loss)의 집합체를 최대화하는 방식으로 진행됩니다. 이번 연구는 4개의 다양한 아키텍처(모델)에 걸쳐 6개의 UCR 시계열 데이터셋을 활용하여 수행되며, 순환 신경망(RNN), 합성곱 신경망(CNN), 상태 공간(state-space) 및 변환기(transformer) 기반 모델을 포함합니다. 또한, 정확성과 탐지의 균형을 이룰 수 있도록 다양한 강도로 악성 샘플을 탐지하는 탐지기 훈련 절차도 제안합니다.

- **Performance Highlights**: 이 공격 기법은 기존의 방법들보다 훌륭한 성능을 발휘하며, 인간의 눈이나 도메인 특정 이상 탐지기(detection model)에서 잘 탐지되지 않는 것을 강조합니다. 다루어진 여러 데이터셋을 통해 이 공격 방식이 시계열 데이터의 적대적 공격에서 얼마나 중요한지를 보여줍니다. 마지막으로, 연구진은 코드와 데이터셋을 GitHub에 제공함으로써 재현 가능성과 후속 연구의 촉진을 도모하고 있습니다.



### Generative Uncertainty in Diffusion Models (https://arxiv.org/abs/2502.20946)
- **What's New**: 이 논문에서는 생성 모델의 생성적 불확실성(generative uncertainty)을 추정하기 위한 베이지안 프레임워크( Bayesian framework)를 제안합니다. 최근 확산 모델(diffusion models)은 생성 모델링에서 중요한 발전을 이끌어내었지만, 개별 샘플의 품질이 낮을 수 있는 문제를 해결하는 것이 어려웠습니다. 본 연구는 고차원의 샘플 공간에서 발생하는 문제를 해결하기 위해 새로운 의미론적 가능도(semantic likelihood)를 도입했습니다.

- **Technical Details**: 제안된 프레임워크는 대규모 확산 모델에 대해 실용적인 베이지안 추정을 가능하게 하며, 라플라스 근사(Laplace approximation)를 이용하여 모델의 예측을 평가합니다. 이 연구에서 소개된 의미론적 가능도는 이미지 인코더(pretrained image encoders)를 활용하여 잠재적( latent ) 의미 공간에서 변동성을 계산합니다. 이 접근법을 이용하여 생성적 불확실성을 효과적으로 검출할 수 있음을 보였습니다.

- **Performance Highlights**: 실험 결과, 제안된 생성적 불확실성이 기존의 불확실성 기반 방법들보다 우수한 성능을 보여주었으며, 낮은 품질의 샘플을 효과적으로 식별할 수 있음을 입증했습니다. 또한, 샘플링 중 베이지안 불확실성의 계산 비용을 최소화할 수 있는 간단하면서도 효과적인 기술을 제시하였습니다. 이 프레임워크는 확산 모델 외에도 다른 생성을 위한 관계모델(latent flow matching model)에 적용 가능하다는 점이 특히 주목할 만합니다.



### Everything, Everywhere, All at Once: Is Mechanistic Interpretability Identifiable? (https://arxiv.org/abs/2502.20914)
- **What's New**: 본 연구는 AI 시스템의 해석 가능성(interpretability)을 보장하는 기준으로서 기계적 해석 가능성(Mechanistic Interpretability, MI)의 역할을 조사합니다. 전통적인 통계에서의 식별성(identifiability) 문제를 AI 해석 가능성에 적용하여, 특정 행동에 대한 독특한 설명이 존재하는지를 탐색합니다. 이 연구는 두 가지 주요 MI 전략을 통해 다양한 설명 기법을 제시하고 검토합니다.

- **Technical Details**: MI는 복잡한 신경망의 행동을 단순한 알고리즘으로 역설명하고, 이를 통해 내부 계산을 추적하는 접근 방식을 취합니다. 본 논문에서는 '어디-그 다음 무엇(where-then-what)'과 '무엇-그 다음 어디(what-then-where)'의 두 가지 전략을 정의하며, 이를 통해 부울 함수와 다층 퍼셉트론(Multi-layer Perceptrons)에서 다양한 후보 설명을 평가합니다. 각각의 전략에서 최적의 서브셋과 causally aligned 알고리즘을 찾아내는 과정을 진행합니다.

- **Performance Highlights**: 실험 결과, 기계적 해석 가능성의 기준이 충분히 엄격하지 않음을 보여줍니다. 여러 회로가 동일한 모델 행동을 복제할 수 있으며, 하나의 회로에 대해 다수의 해석이 가능함을 발견했습니다. 최종적으로 이 연구는 AI의 설명 기준을 정의하는 데 기여하며, 개별 설명의 독창성이 이해에 필수적인지를 재조명합니다.



### A Fused Gromov-Wasserstein Approach to Subgraph Contrastive Learning (https://arxiv.org/abs/2502.20885)
- **What's New**: 본 논문에서는 FOSSIL(Fused Gromov Wasserstein Subgraph Contrastive Learning)이라는 새로운 방법을 제안합니다. 이 모델은 노드 수준(node-level)과 서브그래프 수준(subgraph-level)의 대조 학습(contrastive learning)을 통합하여, 표준 노드 수준 대조 손실과 Fused Gromov-Wasserstein 거리의 조합을 활용합니다. FOSSIL은 동질성(homophily) 및 이질성(heterophily) 그래프 모두에 잘 작동하며, 긍정적 및 부정적 쌍을 생성하기 위한 동적 뷰(view)를 만들 수 있는 장점을 가지고 있습니다.

- **Technical Details**: FOSSIL은 그래프 데이터셋의 동질성 수준에 대한 변동을 견딜 수 있는 강력한 아키텍처를 설계하였습니다. 이 모델은 노드의 특징과 그래프의 구조를 동시에 캡처하기 위해 Fused Gromov-Wasserstein 거리(FGWD)를 활용합니다. 또한, GNN 인코더는 이질성 데이터에 효과적으로 처리할 수 있도록 디커플링(decoupled) 방식으로 설계되었습니다. 이 방식을 통해 FOSSIL은 서브그래프의 구조적 및 특징적 특성을 동시에 인코딩하여 대조 손실을 계산할 수 있습니다.

- **Performance Highlights**: FOSSIL은 다수의 기준 그래프 데이터셋에서 광범위한 실험을 수행하였으며, 최신 방법들과 비교하여 우수한 성능을 보였습니다. 특히, 이 모델은 동질성 및 이질성 데이터 모두에서 일관된 성능을 발휘하여 실용성을 높입니다. 또한, 다양한 데이터셋에서 자가 감독 노드 분류(task)에서 FOSSIL의 효과를 입증하였습니다. 각 디자인 선택에 대한 철저한 분석을 통해 FOSSIL의 성능을 더욱 강화하였습니다.



### Oscillation-Reduced MXFP4 Training for Vision Transformers (https://arxiv.org/abs/2502.20853)
- **What's New**: 이 논문에서는 FP4 정밀도로 트랜스포머를 사전 훈련하는 새로운 방법인 TetraJet을 제안합니다. 저 precision 훈련은 큰 신경망의 학습을 가속화하는 유망한 기술로 부각되고 있으며, FP4 형식은 특히 속도 측면에서 강력한 잠재력을 가지고 있습니다. 그러나 FP4 훈련의 정확도 저하 문제를 해결하기 위해 MXFP4 데이터를 사용하면서 발생하는 문제에 대해 체계적으로 연구한 결과를 보여줍니다.

- **Technical Details**: TetraJet은 MXFP4 계산을 사용하여 트랜스포머의 앞으로 및 앞으로(Backward) 패스를 수행하는 새로운 훈련 방법론입니다. 이 방법은 모든 가중치/활성화/그래디언트 텐서를 MXFP4로 양자화하여 하드웨어의 가속화 잠재력을 최대한 활용합니다. 특히, 저자들은 weight oscillation 문제를 발견하고 이를 해결하기 위한 Q-EMA(EMA Quantizer) 및 Q-Ramping(Adaptive Ramping Optimizer) 방법을 제안하여 성능을 개선했습니다.

- **Performance Highlights**: TetraJet을 사용한 광범위한 실험 결과, 기존의 4비트 훈련 방법에 비해 항상 더 나은 성능을 나타내었습니다. 특히, Q-EMA 및 Q-Ramping은 진동을 효과적으로 줄여줌으로써 추가적인 향상을 가져왔고, 기반선(기존 방법) 대비 정확도 저하를 50% 이상 줄이는 데 성공했습니다. 결과적으로 TetraJet은 풀 정밀도 훈련(Full Precision Training)과 비교할 때 경쟁력 있는 성과를 달성했습니다.



### Gradient Imbalance in Direct Preference Optimization (https://arxiv.org/abs/2502.20847)
Comments:
          15 pages, 2 figures

- **What's New**: 이 논문은 Direct Preference Optimization (DPO)이 Proximal Policy Optimization (PPO) 기반 강화 학습 시스템에서 하위 최적 성능을 보이는 이유를 분석했습니다. 특히, DPO의 훈련 동역학을 체계적으로 분석하고 'gradient imbalance'를 주요 한계로 지적하였습니다. 이 문제를 해결하기 위해 'Balanced-DPO'라는 수정안을 제안하며, 이 방법은 더 효과적인 성능 향상을 목표로 하고 있습니다.

- **Technical Details**: DPO는 인간의 선호 피드백을 기반으로 모델을 직접 최적화하며, 별도의 보상 모델을 필요로 하지 않는 접근법입니다. 그러나 DPO는 훈련 중 승리와 패배 응답의 불균형한 그래디언트 업데이트로 인해 성능이 저하됩니다. Balanced-DPO는 이러한 문제점을 해결하기 위한 간단하면서도 효과적인 그래디언트 재조정 메커니즘을 도입하여, 인간의 선호와의 정렬을 강화하고 OOD 응답의 과대 추정을 줄입니다.

- **Performance Highlights**: Balanced-DPO의 실험 결과는 이론적 발견을 검증하며 DPO의 효과를 향상시키는 데 중요한 역할을 할 수 있음을 확인하였습니다. 본 연구는 복잡한 작업에서 미세한 인간의 선호를 올바르게 반영하여 DPO의 성능을 개선하는 효율적인 방향성을 제시합니다. 향후 연구 방향성에서 DPO와 관련된 다양한 알고리즘의 성능 향상을 위한 기초 데이터를 제공할 것으로 기대됩니다.



### Reinforcement Learning with Curriculum-inspired Adaptive Direct Policy Guidance for Truck Dispatching (https://arxiv.org/abs/2502.20845)
- **What's New**: 이 논문에서는 공개광산에서 트럭 배치 문제를 해결하기 위해 새로운 curriculum learning 전략인 Curriculum-inspired Adaptive Direct Policy Guidance를 소개하고 있습니다. 이 방법은 Proximal Policy Optimization (PPO)을 적응시켜 불균일한 결정 간격을 처리하고, Shortest Processing Time teacher policy를 사용해 정책 탐색을 안내합니다. 이를 통해 기존 PPO 대비 10%의 성능 향상과 더 빠른 수렴성을 보여줍니다.

- **Technical Details**: 트럭 배치 알고리즘 문제는 OpenMines를 gym API를 통해 정의하였고, 다양한 환경 정보를 포함하여 모델 성능을 향상시켰습니다. 관찰 공간은 Order State, Truck Self State, Road States 및 Target States로 나뉘며, M=5, N=5, K=71로 설정되었습니다. 보상 함수 설계는 중요한 과제로, 밀집 보상과 희소 보상의 두 가지 형태를 정의하여 직접 정책 안내의 효과를 검증합니다.

- **Performance Highlights**: 제안된 방법은 여러 트럭이 동일한 에이전트로부터 배차를 요청하는 복합적인 환경에서도 우수한 성능을 나타냅니다. 기존의 최적화 기반 방법들이 직면한 동적 불확실성 문제를 해결하며, 보상 설계에 대한 탄력성을 보여줍니다. 이러한 전략은 RL 기반 트럭 배치 분야에서 일반적이고 효과적인 커리큘럼 학습 기법으로 자리매김할 것으로 기대됩니다.



### Neuro-Symbolic Learning for Galois Groups: Unveiling Probabilistic Trends in Polynomials (https://arxiv.org/abs/2502.20844)
- **What's New**: 이 논문은 다항식의 Galois 그룹을 분류하기 위해 신경 기호적(neurosymbolic) 접근 방식을 제시합니다. 고전적인 Galois 이론(classical Galois theory)과 머신 러닝(machine learning)을 결합하여 대수적 계산(algebraic computation)의 도전 과제를 해결하고 있습니다. 이 연구는 신경망(neural networks)과 기호적 추론(symbolic reasoning)을 함께 사용하여 순수 수치적 방법보다 더 높은 정확성과 해석 가능성을 제공하는 모델을 개발했습니다.

- **Technical Details**: 주요 초점은 높이(height) $ 	$ 이하인 육차(sextic) 다항식에 있으며, 53,972개의 비가환(irreducible) 예제를 포함한 데이터베이스를 분석합니다. 연구자들은 Galois 그룹이 $C_6$인 20개의 육차 다항식이 단지 7개의 불변 정의 등가 클래스(invariant-defined equivalence classes)를 포함하고 있다는 새로운 분포적 경향을 발견했습니다. 이러한 결과는 높이 제약 하에서 Galois 그룹 확률(Galois group probabilities)에 대한 최초의 경험적 통찰(empirical insights)을 제공하며, 기하학적 해법을 탐구하기 위한 기초를 마련합니다.

- **Performance Highlights**: 이 연구는 AI가 전통적인 기호적 기술로는 드러나지 않는 패턴을 발견하는 데 기여할 수 있는 잠재력을 보여줍니다. 연구 결과는 확률적 추측(probabilistic conjectures) 및 고차원 분류(higher degree classifications)와 같은 대수적 문제에 대한 미래 연구의 방향성을 제시합니다. 이는 대수적 계산(computational algebra) 분야에서 상당한 영향을 미칠 것으로 기대됩니다.



### Tuning-Free Structured Sparse PCA via Deep Unfolding Networks (https://arxiv.org/abs/2502.20837)
- **What's New**: 이 논문에서는 구조화된 희소 주성분 분석(Structured Sparse PCA) 방법론을 제안하여 기존의 희소 주성분 분석(PCA)에서 발생하는 매개변수 조정의 어려움을 해결하고자 합니다. 새로운 접근법은 ℓ_1-norm과 ℓ_{2,1}-norm을 통합하여 지역적 및 전역적 구조를 동시에 포착하여 더 나은 성능을 목표로 합니다. 특히, 딥 언폴딩 네트워크(deep unfolding network)를 이용해 반복 최적화 단계를 신경망 아키텍처로 변환함으로써 자동으로 매개변수를 학습하는 혁신적인 방법론을 개발했습니다.

- **Technical Details**: 제안된 방법론은 교대로 방향 방법(ADMM)에 기반하여 각 최적화 단계를 신경망의 레이어로 변환합니다. 문제를 변형하기 위해 보조 변수 Y와 Z를 도입하고, 보조 변수를 고정한 상태에서 하나의 변수를 업데이트하는 일반적인 ADMM 방식을 활용하여 문제를 해결합니다. 이 과정에서 Lagrange 승수와 페널티 매개변수를 사용하여 문제를 보다 간단하게 업데이트할 수 있습니다.

- **Performance Highlights**: 성능 평가 결과, 제안된 SPCA-Net 방법이 기존의 최고 성능(State-of-the-art) 방법들과 비교하여 가시적인 이점을 보였음을 확인했습니다. 특히, 새로운 구조를 통해 기존의 방법들이 직면했던 매개변수 조정의 필요성을 효과적으로 회피하며, 여러 벤치마크 데이터셋에서 우수한 성능을 발휘했습니다.



### LADs: Leveraging LLMs for AI-Driven DevOps (https://arxiv.org/abs/2502.20825)
Comments:
          17 pages with Appendix, 8 figures, and 7 tables. This paper is currently Under Review

- **What's New**: 최근 발표된 연구에서는 자동화된 클라우드 구성 및 배포의 복잡성을 극복하기 위한 LADS 프레임워크를 소개합니다. 이 프레임워크는 Large Language Models (LLMs)를 사용하여 클라우드 관리의 견고함과 효율성을 보장합니다. LADs는 기존 기술의 한계를 넘어, 최적화 방법을 심층 분석하여 적응성과 효율성을 제공합니다.

- **Technical Details**: LADS는 Retrieval-Augmented Generation, Few-Shot Learning, Chain-of-Thought 및 Feedback-Based Prompt Chaining을 활용하여 새로운 구성 설정을 생성하고 배포 실패로부터 학습하여 시스템 설정을 반복적으로 개선합니다. 이를 통해 DevOps 팀의 작업 부하를 줄이고, 리소스 활용성을 최적화하며, 시스템의 신뢰성을 향상시킵니다.

- **Performance Highlights**: 광범위한 평가를 통해 LADS는 수동 노력을 크게 줄이고, 리소스 사용을 최적화하며, 시스템의 전반적인 신뢰성을 증가시켰습니다. 이 연구 결과는 성능, 비용 및 확장성 간의 트레이드 오프(key insights)와 함께 다양한 배포 시나리오에 적합한 전략을 수립하는 데 도움을 줍니다.



### Digital Player: Evaluating Large Language Models based Human-like Agent in Games (https://arxiv.org/abs/2502.20807)
Comments:
          neurips datasets and benchmarks 2024, not accepted

- **What's New**: 이 논문에서는 LLM(대형 언어 모델) 기반의 디지털 에이전트를 활용한 새로운 데이터 생성 및 학습 플랫폼인 CivSim을 소개합니다. CivSim은 인기 있는 전략 게임 'Unciv'를 기반으로 하여 수백만의 활성 사용자와 함께 진행되며, LLM을 이용한 인간 유사 에이전트 연구에 기여할 수 있는 기회를 제공합니다. 이 플랫폼은 게이머와의 상호작용을 통하여 데이터 플라이휠(data flywheel) 생성의 가능성을 제시하는 데 중점을 둡니다.

- **Technical Details**: CivSim은 'Unciv'라는 게임을 활용하여 생성된 LLM 기반 인간 같은 디지털 플레이어입니다. 이 에이전트는 경제 발전, 과학 연구, 외교, 전쟁 등 다양한 결정 공간을 탐색하며, 사회적 상호작용을 위한 언어 기반의 협상 및 기만을 포함하는 복합적인 의사결정 능력을 요구합니다. 또한, OpenAI의 ChatGPT를 사용하는 기본 LLM과 같은 최신 기술을 통합하여 에이전트의 학습 능력을 향상시키고자 합니다.

- **Performance Highlights**: CivSim은 기존의 LLM 기반 에이전트와 비교하여 보다 정교한 전략적 사고 능력과 장기 계획 능력을 요구하며, 다양한 인간 플레이어와의 상호작용을 통해 에이전트의 실질적인 성능을 검증할 수 있습니다. 특히, 이 프로젝트는 게임 내의 의사소통 메커니즘 부족을 해결하기 위해 Discord 기반의 챗봇 인터페이스를 개발하여 게이머가 CivAgent와 직접적인 상호작용을 통해 전략적 협상을 진행할 수 있도록 합니다.



### FlexPrefill: A Context-Aware Sparse Attention Mechanism for Efficient Long-Sequence Inferenc (https://arxiv.org/abs/2502.20766)
Comments:
          Accepted at ICLR 2025 (Oral)

- **What's New**: 본 논문에서는 FlexPrefill이라는 유연한 스프레드 프리필 메커니즘을 제안합니다. 이 기법은 스프레드 어텐션 패턴과 계산 예산을 실시간으로 조정하여 입력의 특정 요구사항 및 어텐션 헤드에 맞춰 조절할 수 있습니다. 기존의 고정된 스프레드 어텐션 방식의 한계를 극복하고, 다양한 입력 요구사항에 더욱 효율적으로 대응할 수 있는 방법을 제공합니다.

- **Technical Details**: FlexPrefill의 핵심 요소는 두 가지입니다: 첫째, Query-Aware Sparse Pattern Determination입니다. 이 방법은 쿼리-특정 다양한 어텐션 패턴과 사전 정의된 패턴 간의 전환을 가능하게 해주는 Jensen-Shannon divergence를 측정합니다. 둘째, Cumulative-Attention Based Index Selection으로, 입력에 따라 계산할 쿼리-키 인덱스를 동적으로 선택하여, 어텐션 점수의 합이 사전 정의된 임계값을 충족하도록 보장합니다.

- **Performance Highlights**: 다양한 장기 맥락 벤치마크에서 FlexPrefill을 통해 이전 방법들에 비해 속도와 정확도에서 유의미한 개선이 있음을 보여주었습니다. 실험 결과, FlexPrefill은 여러 맥락 길이와 작업에서 모델 성능을 유지하거나 심지어 향상시키는 데 성공했습니다. 이로 인해 LLM(대형 언어 모델) 추론에 더욱 유연하고 효율적인 솔루션을 제공할 수 있음을 입증했습니다.



### Visual Attention Exploration in Vision-Based Mamba Models (https://arxiv.org/abs/2502.20764)
Comments:
          6 pages, 8 figures

- **What's New**: 본 연구에서는 Mamba 모델의 시각 기반 주의 메커니즘을 해석하는 데 중점을 두고 있습니다. Mamba는 선택적 스캔 메커니즘을 도입하여 입력 토큰에 학습 가능한 가중치를 부여함으로써 주의 메커니즘을 모방합니다. 본 논문은 이미지 처리 분야로 Mamba를 확장하는 가운데, 이미지 패치를 1D 시퀀스로 배열할 때의 상호작용에 대한 이해를 돕기 위한 도구를 제시합니다.

- **Technical Details**: Mamba는 SSM(상태 공간 모델)로서, 각 패치를 토큰으로 간주하고 2D 이미지를 p×p 크기의 작은 패치들로 분해하여 Mamba 모델에 입력합니다. 특히, VMamba 블록은 네 가지 다른 스캔 순서를 사용하여 패치를 처리하며, 각 스캔 순서가 패치 간의 의존성을 다르게 만듭니다. 이 논문에서 사용하는 시각 분석 도구는 Mamba의 다양한 블록 전반에 걸쳐 주의가 어떻게 분포되는지를 살펴보는 데에 목적이 있습니다.

- **Performance Highlights**: Mamba는 다양한 어플리케이션에서 트랜스포머와 동등한 성능을 발휘하는 한편, 선형 복잡성을 통해 레이턴시가 중요한 어플리케이션에서 유리한 선택이 됩니다. 연구 결과를 통해 서로 다른 패치 순서 전략이 학습된 주의에 미치는 영향을 분석하고, Mamba 모델 내에서의 주의 패턴을 살펴봄으로써 Mamba의 동작에 대한 깊은 통찰을 제공합니다.



### Information-Theoretic Perspectives on Optimizers (https://arxiv.org/abs/2502.20763)
- **What's New**: 이 논문에서는 Neural Networks의 optimizer와 architecture 간의 복잡한 상호작용을 분석하며, 기존의 검정 척도 대신 entropy gap이라는 정보 이론적 척도를 도입하여 성능 분석을 개선하고 있습니다. 즉, sharpness와 entropy gap이 성능에 미치는 영향을 연구하였고, Lion optimizer의 업데이트 규칙을 정보 이론적 도구를 통해 분석하여 향상 방법을 모색하였습니다. 이 연구는 optimizer 평가 및 개선을 위한 새로운 프레임워크를 제안하고 있습니다.

- **Technical Details**: 논문에서는 ResNet과 ViT 아키텍처에서의 다양한 optimizer의 학습 동역학을 고려하고 있으며, SGD와 Adam을 비교합니다. ResNet은 ViT보다 더 나은 landscape를 제공하여 SGD의 성능이 더 뛰어나며, sharpness 메트릭만으로는 이러한 동작을 완벽하게 설명할 수 없는 것으로 나타났습니다. 정보 이론에 기반한 새로운 메트릭인 entropy gap을 통해 optimizer의 경로를 더 잘 이해할 수 있어, 훈련 중 convergence 행동과 generalization을 탐구할 수 있습니다.

- **Performance Highlights**: Sharpness와 entropy gap은 다양한 optimizer의 성능에 중요한 영향을 미친다는 것을 실험과 이론적 분석을 통해 입증했습니다. 실험 결과, ResNet 및 ViT와 같은 대표적인 아키텍처에서 다양한 optimizer의 훈련 동역학을 살펴봄으로써, optimizer의 효과성을 평가할 때 두 메트릭을 모두 고려하는 것이 중요하다는 점을 강조했습니다. 이 연구는 정보 이론적 도구를 활용하여 더 효율적이고 일반화 가능한 최적화 전략으로 이어질 수 있는 지식을 확장하고 있습니다.



### Generating Clinically Realistic EHR Data via a Hierarchy- and Semantics-Guided Transformer (https://arxiv.org/abs/2502.20719)
- **What's New**: 이 논문에서는 전통적인 전자 건강 기록(EHR) 데이터 생성 방식의 한계를 극복하기 위해, 계층적 및 의미 정보를 활용한 새로운 프레임워크인 Hierarchy- and Semantics-Guided Transformer (HiSGT)를 제안합니다. 기존의 방법들이 EHR을 평면적인 의학 코드 시퀀스로 처리함으로써 발생하는 문제점을 해결하고자 하며, 특히 임상 코드의 계층적 구조와 그 세부적인 의미를 반영합니다.

- **Technical Details**: HiSGT는 임상 코드 간의 부모-자식 및 형제 관계를 인코딩하기 위해 계층적 그래프를 구성하고, 이를 기반으로 그래프 신경망(Graph Neural Network, GNN)을 사용하여 계층 인식 임베딩을 도출합니다. 이와 함께, 사전 훈련된 임상 언어 모델(예: ClinicalBERT)로부터 추출된 의미 임베딩과 융합하여, 진짜 EHR의 세밀한 임상 패턴을 정확하게 모델링하는 Transformer 기반 생성기를 개발합니다.

- **Performance Highlights**: MIMIC-III 및 MIMIC-IV 데이터셋에서의 광범위한 실험을 통해 HiSGT가 생성한 데이터가 실제 환자 기록과 통계적으로 우수한 일치를 이룬다는 점을 입증했습니다. 또한 HiSGT는 만성 질환의 분류와 같은 하류 애플리케이션에서 매우 강력한 성능을 지원하여, 데이터 품질 향상 및 개인 정보 보호 기능을 향상하는 데 기여합니다.



### Unlearning through Knowledge Overwriting: Reversible Federated Unlearning via Selective Sparse Adapter (https://arxiv.org/abs/2502.20709)
Comments:
          Accepted by CVPR2025

- **What's New**: 제안된 방법 FUSED(Federated Unlearning via SElective sparse aDapter)는 연합 학습의 기존 문제를 해결하기 위해 독립적인 희소 어댑터를 사용하여 모델의 민감한 레이어를 분석합니다. 이 방법은 원래의 모델 파라미터를 변경하지 않고, 학습된 지식을 재작성하여 잊혀진 지식을 효과적으로 덮어씌웁니다. 이를 통해 무차별적인 학습 제거의 영향을 완화하고, 지식을 잊는 과정을 가역적으로 만들어 비용을 크게 줄일 수 있습니다.

- **Technical Details**: FUSED는 모델의 레이어를 민감도에 따라 분석하여 중요한 레이어를 식별한 후, 이러한 레이어를 위해 희소한 학습 제거 어댑터를 생성합니다. 이러한 어댑터는 클라이언트에게 분배되어 원본 모델이 동결된 상태에서 독립적으로 훈련됩니다. 캡슐화된 어댑터는 잊혀져야 할 지식을 빠른 시간 안에 복구할 수 있게 하며, 전체 연합 모델과 통합되어 잊혀져야 할 지식을 덮어씌우는 역할을 합니다.

- **Performance Highlights**: FUSED는 세 가지 데이터셋에서 다양한 학습 제거 시나리오를 통해 광범위한 실험을 진행했으며, 그 효과가 Retraining과 동등하지만 다른 모든 기준선보다 뛰어난 성능을 보였습니다. FUSED는 또한 연합 학습 시스템에서 클라이언트 잊기 및 샘플 잊기 시나리오를 효율적으로 처리하여 성능을 극대화하고, 학습 제거 비용을 현저히 줄이는 데 성공했습니다.



### FoCTTA: Low-Memory Continual Test-Time Adaptation with Focus (https://arxiv.org/abs/2502.20677)
- **What's New**: 본 논문에서는 IoT 애플리케이션을 위한 저메모리 지속적 테스트 시간 적응 전략인 FoCTTA를 제안합니다. 기존 TTA 방법이 모든 배치 정규화(BN) 레이어를 업데이트 하는 것과는 달리, FoCTTA는 드리프트에 민감한 몇몇 표현 레이어를 자동으로 식별하고 이를 적응시키는 접근법을 취합니다. 이로 인해 대량의 메모리 사용을 피할 수 있으며, 메모리 효율성과 효과적인 적응을 동시에 달성합니다.

- **Technical Details**: FoCTTA는 고차원 데이터를 다룰 때 요구되는 큰 배치 사이즈 없이도 작동합니다. 주요 기법은 TTA가 요구하는 고차원의 레이어 중에서 적응에 중요한 상위 K개 레이어만을 업데이트하여 활성화 저장 소요를 줄이는 것입니다. 이 기법을 통해 오프라인 워밍업 훈련 단계에서 사전 학습 후 아무도 보지 못한 분포 변화 감지를 위한 간단한 그래디언트 기준의 중요도 지표를 사용하여 적응에 필요한 레이어를 식별합니다.

- **Performance Highlights**: FoCTTA는 CIFAR10-C, CIFAR100-C, ImageNet-C 데이터 집합에서 기존 최첨단 방법들에 비해 각각 4.5%, 4.9%, 14.8%의 정확도 향상을 달성했습니다. 평균적으로 다양한 배치 사이즈에서 메모리 사용량이 3배 감소되었으며, 정확도는 각각 8.1%, 3.6%, 0.2% 향상되었습니다. 이러한 결과는 IoT 기기에서의 메모리 효율적이고 효과적인 적응의 필요성을 잘 보여 줍니다.



### Dimension Agnostic Neural Processes (https://arxiv.org/abs/2502.20661)
Comments:
          10 pages, 5 figures, Accepted to ICLR 2025 (International Conference on Learning Representations)

- **What's New**: 본 논문에서는 고차원 입력 데이터와 다양성 있는 특성에 효과적으로 대응할 수 있도록 Dimension Agnostic Neural Processes (DANP)를 제안합니다. 기존 Neural Process (NP) 모델들이 겪었던 한계를 극복하고 이를 일반 회귀 모델로서 활용할 수 있는 가능성을 확장합니다. 이 모델은 Dimension Aggregator Block (DAB)을 포함하여 입력 기능을 고정 차원으로 변환하는 혁신적인 접근 방식을 사용합니다.

- **Technical Details**: DANP는 Transformer 아키텍처를 활용하여 다양한 데이터셋에서 더 많은 특성을 학습합니다. 이를 통해 모델이 기능적 불확실성을 효과적으로 포착하고, 다양한 작업에서 예방적 확률 분포를 생성할 수 있도록 합니다. 이 시스템은 기존 NP 모델들이 겪었던 두 가지 주요 제약으로부터 자유로워, 입력 및 출력 차원의 변화를 포괄적으로 처리할 수 있습니다.

- **Performance Highlights**: 다양한 합성 및 실제 회귀 작업에 대한 포괄적인 실험을 통해 DANP가 이전의 NP 변형보다 뛰어난 예측 성능을 발휘하는 것을 입증했습니다. DANP는 전통적인 NP 모델의 한계를 극복하여 다양한 회귀 시나리오에서 폭넓은 적용 가능성을 보여주었습니다.



### FedConv: A Learning-on-Model Paradigm for Heterogeneous Federated Clients (https://arxiv.org/abs/2502.20639)
- **What's New**: 이 논문에서는 사용자가 자원 제약이 있는 클라이언트에 대해 맞춤형 서브 모델을 제공하는 클라이언트 친화적 연합 학습(FL) 프레임워크인 FedConv를 제안합니다. FedConv는 통상적인 FL이 동질적인 대규모 글로벌 모델을 모든 클라이언트에 전송하고 훈련시키는 문제를 해결하여, 자원 부족 클라이언트의 부담을 최소화합니다. 이전의 기술들과는 달리, FedConv는 합성곱 압축(convolutional compression)을 통해 이질적인 서브 모델의 파라미터를 학습합니다.

- **Technical Details**: FedConv는 클라이언트가 로컬 훈련 후 모델 파라미터만을 서버에 업로드하고, 서버는 모델 집합을 조정하여 글로벌 모델을 업데이트 하는 전통적인 FL 시스템에 기반합니다. 이 프레임워크는 클라이언트가 각자의 자원 예산에 맞게 다양한 서브 모델의 파라미터를 학습할 수 있도록 합성곱을 이용하여 글로벌 모델을 압축합니다. 서브 모델은 클라이언트에서 직접 훈련 가능하며, 이 과정에서 클라이언트에 대한 추가 부담 없이도 사용자 데이터의 개인화된 정보를 유지합니다.

- **Performance Highlights**: 실험 결과, FedConv는 여섯 개의 공용 데이터 세트에서 기존 최첨단 FL 시스템보다 평균 35% 이상의 모델 정확도를 기록하며, 계산 및 통신 오버헤드는 각각 33% 및 25% 감소하였습니다. 이러한 성과는 FedConv가 자원 제약이 있는 클라이언트를 위한 연합 학습의 새로운 가능성을 제시하며, 전체 훈련 시간을 절감하는 데 기여합니다. FedConv는 글로벌 모델의 중요 정보를 효과적으로 유지하면서도 압축하는 최초의 방법으로, FL 시스템에서 비대칭 자원 활용 문제를 심각하게 개선하였습니다.



### A Compact Model for Large-Scale Time Series Forecasting (https://arxiv.org/abs/2502.20634)
- **What's New**: UltraSTF 모델이 소개되었으며, 이 모델은 스페이셜-템포럴(spatio-temporal) 데이터에 최적화된 접근 방식을 제공합니다. 기존의 SparseTSF 모델의 한계를 극복하기 위해 초-컴팩트(shape bank) 구성 요소를 추가하였습니다. 이 모델은 주기성을 활용하고 intra-period(기간 내) 시간 의존성을 효과적으로 포착하여 예측 성능을 개선합니다.

- **Technical Details**: UltraSTF는 교차 주기 예측 모듈과 초-컴팩트 형태의 뱅크(shape bank) 요소를 통합하여 설계되었습니다. 이 모델은 형태 뱅크의 주의(attention) 메커니즘을 통해 시간 시계열의 반복 패턴을 탐지하고 학습할 수 있습니다. 이러한 특성으로 인해 UltraSTF는 고차원 데이터에서도 높은 성능을 발휘할 수 있습니다.

- **Performance Highlights**: UltraSTF는 LargeST 벤치마크에서 최첨단 성능을 달성했습니다. 이 모델은 두 번째로 우수한 방법들이 요구하는 것의 0.2%도 안 되는 파라미터(parameter)만으로 이러한 성능을 나타내어 기존 방법들의 Pareto 경계를 더욱 확장했습니다.



### Are LLMs Ready for Practical Adoption for Assertion Generation? (https://arxiv.org/abs/2502.20633)
Comments:
          7 Pages, 9 Figures, Accepted in DATE 2025. arXiv admin note: substantial text overlap with arXiv:2406.18627

- **What's New**: 이 논문은 하드웨어 설계의 검증을 위한 어설션(Assertions) 생성에 대한 새로운 접근법을 제시합니다. 기존의 상용 LLM 모델은 문법적(syntactically) 및 의미적(semantically)으로 올바른 어설션을 생성하는 데 충분하지 않은 것을 확인했습니다. 이를 해결하기 위해, 논문에서는 어설션 생성을 위해 특별히 조정된 새로운 LLM 모델인 AssertionLLM을 소개하며, 이 모델이 기존 LLM보다 생성된 어설션의 품질을 향상시킨다고 주장합니다.

- **Technical Details**: 상반된 디자인 이론이나 요구 사항에 대한 효과적인 어설션을 생성하기 위해 AssertionBench라는 벤치마크 세트를 구축했습니다. 이를 통해 다양한 복잡도를 가진 하드웨어 디자인을 평가하고, 상용 LLM의 성능을 비교 분석합니다. 특히, AssertionLLM은 정교히 조정된 모델로, 인간의 추가적인 입력없이도 더욱 정확한 어설션을 생성할 수 있어 자동화된 검증 프로세스의 발전을 이끌 수 있습니다.

- **Performance Highlights**: 초기 실험 결과에 따르면, AssertionLLM은 상용 LLM에 비해 25% 더 많은 문법적 및 의미적으로 올바른 어설션을 생성하는 것으로 나타났습니다. 이는 하드웨어 검증 작업에서의 효율성을 크게 향상시키는 결과로 평가됩니다. 이 논문은 실제 산업에서의 어설션 품질 요구에 보다 잘 부응할 수 있는 방법을 제안하면서 미래의 연구 방향을 제시합니다.



### Discovering Global False Negatives On the Fly for Self-supervised Contrastive Learning (https://arxiv.org/abs/2502.20612)
- **What's New**: GloFND는 자기 감독 기반의 대비 학습(self-supervised contrastive learning)에서 가짜 부정 쌍(false negatives)을 식별하기 위한 새로운 최적화 접근 방식을 제안합니다. 기존 방법들이 미니배치 내에서 로컬하게 작동하는 것과 달리, GloFND는 전체 데이터셋에서 가짜 부정을 글로벌하게 탐지하여 보다 정밀한 학습을 가능하게 합니다. 이 방법은 훈련 중 각 앵커 데이터의 가짜 부정에 대한 임계값(threshold)을 자동으로 학습하여 효과적인 대비 학습을 지원합니다.

- **Technical Details**: GloFND의 α(알파) 하이퍼파라미터는 가짜 부정의 다양한 정의에 적응할 수 있게 도와줍니다. 이 하이퍼파라미터의 값은 가짜 부정의 수나 가능성에 대한 사전 지식, 표현의 세밀함에 따라 설정할 수 있으며, 최소의 연산 비용으로 계산됩니다. GloFND를 SimCLR, SogCLR 및 CLIP과 같은 대비 기반 메서드와 결합할 경우, 성능 저하 없이 잘 작동하며, 실험에서는 2%의 훈련 시간 증가만 보여줍니다.

- **Performance Highlights**: GloFND는 이미지 및 이미지-텍스트 데이터 실험에서 유의미한 성능 향상을 나타냈습니다. 특히, ResNet-50을 기반으로 한 선행 학습된 모델에서 더욱 효과적인 특성 표현을 생성하는 능력을 보여주었습니다. 실험 결과는 GloFND가 다양한 레이블 비율에 대한 준지도 학습에서도 효과적인 성능을 발휘하며, 전체적인 준지도 성능을 평균화한 결과를 통해 기여도를 입증하였습니다.



### Exploring the Impact of Temperature Scaling in Softmax for Classification and Adversarial Robustness (https://arxiv.org/abs/2502.20604)
- **What's New**: 이 연구는 deep learning에서 softmax 함수의 '온도'(temperature) 매개변수의 중요성을 강조합니다. 저자들은 온도 스케일링이 이미지 분류에서 모델의 학습 및 최적화 성능에 미치는 영향을 실증적 및 이론적으로 탐구하였고, 중간 온도가 전반적인 성능을 개선할 수 있음을 발견했습니다. 특히, 높은 온도가 적대적 공격에 대한 모델의 견고성을 향상시킨다는 새로운 이점을 제시했습니다.

- **Technical Details**: 연구에서는 convolutional neural networks (CNNs)와 transformers를 사용하여 다양한 벤치마크 데이터셋에서temperature의 영향력을 분석했습니다. 높은 온도가 모델의 학습 방향을 조정하며, 더 균형 잡힌 학습을 촉진한다는 논의가 이루어졌습니다. 저자들은 또한, 낮은 온도가 오히려 오류가 잦은 클래스에 집중하게 하고, 높은 온도는 모든 클래스에 대한 균형 잡힌 학습을促進한다고 설명했습니다.

- **Performance Highlights**: 저자들은 온도를 조절하여 모델을 훈련했을 때 성능 향상과 함께, 높은 온도를 사용하는 경우에 적대적 훈련의 성능이 개선된다고 밝혔습니다. 이 연구는 모델의 견고성을 높이는 새로운 방법론을 제시하며, deep learning 응용에서의 성능과 보안 강화를 위한 새로운 방향을 제공합니다. 저자들은 이후 모델 성능과 적대적 훈련을 결합하여 온도 조절의 가능성을 보여주었습니다.



### Deep Learning of the Evolution Operator Enables Forecasting of Out-of-Training Dynamics in Chaotic Systems (https://arxiv.org/abs/2502.20603)
- **What's New**: 이번 연구에서는 chaos 시스템을 위한 딥러닝 에뮬레이터가 훈련 데이터에서 보이지 않는 현상을 예측할 수 있음을 입증하였습니다. Kuramoto-Sivashinsky 및 beta-plane turbulence 모델을 사용하여 자발적인 재점착(spontaneous relaminarisation) 예측 및 임의의 chaos 상태 초기화(initialisation)와 같은 여러 시나리오를 통한 성능을 평가했습니다. 주요 결과는 딥러닝 에뮬레이터가 복잡한 시스템의 긴급 행동(emergent behaviours)과 드문 사건(rare events)을 발견할 수 있다는 점입니다.

- **Technical Details**: 이 연구에서는 각 점을 중심으로 한 공간적으로 국소화된 창(window)에 집중하는 local attention 메커니즘을 적용하였습니다. 이는 CNN에서의 receptive field와 유사하며, 공간적 가중치(spatial weights)를 동적으로 계산하여 다양한 입력 스케일에 적응 가능합니다. 입력 데이터는 latent space로 인코딩되며, 이러한 변환은 선형 변환을 통해 이루어지고, transformer 블록을 통과한 후 최종 예측 결과가 도출됩니다.

- **Performance Highlights**: Kuramoto-Sivashinsky 방정식에 대한 에뮬레이터는 훈련 중에 접하지 않았던 상태 전환을 성공적으로 재현할 수 있었습니다. 주목할 점은 Fourier Neural Operator(FNO)의 한계로, 크기가 증가하는 도메인에서는 적절한 예측을 생성하는 데 어려움을 겪는 반면, local attention 메커니즘은 다양한 크기의 도메인에서 유동적으로 동작할 수 있도록 설계되었다는 것입니다.



### Cache-of-Thought: Master-Apprentice Framework for Cost-Effective Vision Language Model Inferenc (https://arxiv.org/abs/2502.20587)
Comments:
          Mingyuan, Jize, and Haozhen contributed equally, while Minjia, Chengxiang, and Klara advised equally

- **What's New**: 이번 논문에서는 Cache of Thought (CoT)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 대형 비전 언어 모델(large Vision Language Models, VLMs)과 소형 VLM 간의 협업 추론을 가능하게 합니다. CoT는 대형 VLM(mater)에서 획득한 고품질 쿼리 결과를 캐시에 저장하여, 소형 VLM(apprentice)의 성능을 향상시키는 데 도움을 줍니다.

- **Technical Details**: CoT는 멀티 모달 검색(multi modal retrieval)과 in-context 학습을 통해 대형 모델이 생성한 답변을 소형 모델의 학습 과정에 통합합니다. 이를 통해 보다 효율적으로 자원을 사용하면서도 응답 품질을 유지할 수 있습니다. 이 프레임워크는 다양한 일반적인 VQA(Visual Question Answering) 벤치마크에서 철저히 평가되었습니다.

- **Performance Highlights**: 연구 결과, CoT는 동일한 예산 하에서 VQA 성능을 최대 7.7% 향상시켰습니다. 또한, 소형 VLM의 성능을 최대 36.6%까지 증대시켰으며, 이는 소형 모델의 효율성을 극대화하는 데 중요한 의의를 갖습니다.



### Training LLMs with MXFP4 (https://arxiv.org/abs/2502.20586)
Comments:
          AISTATS 2025

- **What's New**: 이 연구에서는 MXFP4를 사용하는 거의 손실 없는 훈련 레시피를 처음으로 소개합니다. 전통적인 BF16 대신 MXFP4를 사용하면 GEMM(General Matrix Multiply)가 지원되는 하드웨어에서 2배 빠르게 작동하므로 훈련 비용을 줄일 수 있습니다. 이 방법을 통해 훈련 중 모델 품질 저하를 최소화할 수 있습니다.

- **Technical Details**: 그 핵심 아이디어는 stochastic rounding(SR)을 사용하여 편향이 없는 기울기 추정치를 계산하는 것입니다. 그러나 SR을 MXFP4에 직접 적용하면 block-level outliers로 인해 높은 분산이 발생할 수 있어 수렴에 문제를 일으킬 수 있습니다. 이를 해결하기 위해 무작위 Hadamard 변환을 사용하여 SR의 분산을 이론적으로 제한했습니다.

- **Performance Highlights**: 이 방법을 사용하여 최대 67억 개의 매개변수를 가진 GPT 모델을 훈련했으며, 이는 BF16 혼합 정밀 훈련에 비해 최소한의 품질 저하를 초래했습니다. 우리의 레시피는 MXFP4로 훈련 FLOPs의 절반 이상을 계산하며, 이는 FP8에 비해 1.3배, BF16에 비해 1.7배의 속도 향상을 가능하게 합니다.



### LiteASR: Efficient Automatic Speech Recognition with Low-Rank Approximation (https://arxiv.org/abs/2502.20583)
- **What's New**: LiteASR는 Automatic Speech Recognition (ASR) 인코더에 대한 저품질 압축 방법으로, 특히 Whisper와 같은 현대 모델에서 발견된 강력한 저랭크 특성을 활용합니다. 본 연구에서는 주어진 소량의 교정 데이터셋을 사용하여 주성분 분석(principal component analysis, PCA)을 수행하고, 이를 통해 인퍼런스 성능을 크게 향상시키면서 정확도를 유지합니다. LiteASR는 Whisper large-v3의 인코더 크기를 50% 이상 감소시킬 수 있음을 보여주며, 이는 효율성과 성능의 새로운 Pareto-optimal 경계를 설정합니다.

- **Technical Details**: LiteASR는 숨겨진 활성화에서 저랭크 구조를 활용하여 ASR 인코더의 연산 부하를 줄이는 압축 메커니즘입니다. 이 방법은 주성분 분석을 통해 기본 성분을 추출하고, 이를 통해 선형 변환을 저랭크 행렬 곱셈 체인으로 근사합니다. 또한, 자기 주의(self-attention)를 저차원에서 작동하도록 최적화하여 인퍼런스에 필요한 부동 소수점 연산(FLOPs)을 줄입니다. 이 프로세스는 적응형 메커니즘을 통해 각 레이어의 최적 저랭크 근사도를 결정합니다.

- **Performance Highlights**: LiteASR를 Whisper large-v3에 적용했을 때, 인코더 크기를 약 40% 줄이는 동시에 실행 속도는 약 1.4배 개선되었습니다. 이 방법은 다른 언어와 모델에 걸쳐도 효과적으로 적용 가능성을 보여줍니다. 또한, 모델 크기를 Whisper medium과 유사한 수준으로 줄이면서 향상된 정확도를 달성할 수 있음을 입증했습니다.



### Training Large Neural Networks With Low-Dimensional Error Feedback (https://arxiv.org/abs/2502.20580)
- **What's New**: 이 논문은 전통적인 신경망 훈련 방식인 역전파(backpropagation)에서 벗어나, 저차원 오차 신호(low-dimensional error signals)를 활용하는 새로운 학습 규칙을 제안합니다. 이 규칙은 피드백 정렬(Feedback Alignment)을 기반으로 하며, 대규모 네트워크의 훈련을 효율적으로 수행할 수 있도록 한다는 점이 혁신적입니다. 실험 결과, 저차원 오차 신호가 기존의 고차원 방법과 유사한 성능을 낼 수 있다는 점을 강조하며, 신경망 훈련 및 생물학적 학습 모델에 대한 새로운 관점을 제시합니다.

- **Technical Details**: 논문에서 제안하는 접근 방식은 신경망의 순방향 전파(forward pass)와 역방향 전파(backward pass)를 분리하는 것입니다. 이를 통해 오차 신호의 차원을 정확하게 제어하면서도 고차원의 표현을 유지할 수 있습니다. 이 방법은 선형 네트워크를 대상으로 이론적 분석을 통해 저차원 오차의 효용성을 보여주고, 비선형, 합성곱(convolutional), 트랜스포머(transformer) 아키텍처에까지 확장 가능합니다.

- **Performance Highlights**: 제안된 저차원 오차 신호를 이용한 학습 방법은 전통적인 역전파와 유사한 성능을 달성할 수 있음을 입증합니다. 특히, 합성곱 신경망(convolutional networks)에서는 피드백 정렬 방법으로는 훈련이 어렵던 기존의 문제를 해결하며, 훈련 효율성을 크게 향상시킵니다. 이는 신경망 훈련에 대한 기존의 고차원 기울기 신호 의존성을 재검토하도록 유도하며, 인공지능 및 생물학적 학습 메커니즘 이해에 기여합니다.



### PFformer: A Position-Free Transformer Variant for Extreme-Adaptive Multivariate Time Series Forecasting (https://arxiv.org/abs/2502.20571)
Comments:
          PAKDD 2025 special session on Data Science: Foundations and Applications (DSFA)

- **What's New**: 이 연구는 PFformer라는 새로운 위치 비기반( position-free) Transformer 모델을 소개합니다. 이 모델은 극단적인 변동성을 특징으로 하는 다변량 시계열(Multivariate time series, MTS) 예측을 위해 설계되었습니다. PFformer는 기존의 모델들이 데이터의 중요한 상관관계를 제대로 반영하지 못하는 문제를 해결하도록 만들어졌습니다.

- **Technical Details**: PFformer는 두 가지 새로운 임베딩 전략을 통합합니다: 향상된 특성 기반 임베딩(Enhanced Feature-based Embedding, EFE)과 자동 인코더 기반 임베딩(Auto-Encoder-based Embedding, AEE)입니다. EFE는 관련된 시퀀스 하위 집합을 위치 제약 없이 고차원 공간으로 매핑하여 상관관계를 효과적으로 인코딩합니다. 이를 통해 모델은 기존의 위치 인코딩 한계 없이 더 나은 성능을 발휘할 수 있습니다.

- **Performance Highlights**: PFformer는 3일 후 장기 예측과 수자원 관리의 실시간 의사결정을 반영한 4시간마다의 롤링 예측 시나리오에서 효과적으로 평가되었습니다. 이 모델은 최신 모델들과 비교하여 예측 정확도를 20%에서 60% 향상시켰습니다. 이러한 성과는 PFformer의 우수한 예측 능력을 보여줍니다.



### Stochastic Rounding for LLM Training: Theory and Practic (https://arxiv.org/abs/2502.20566)
Comments:
          AISTATS 2025

- **What's New**: 이 연구에서는 스토캐스틱 라운딩(Stochastic Rounding, SR)을 활용하여 낮은 정밀도 표현의 훈련에서 발생할 수 있는 수치 오류를 해결하려고 합니다. 기존의 Mixed Precision (MP) 방안에 비해 BF16 + SR 접근법이 대규모 훈련에서의 안정성과 성능을 향상시키는 것으로 나타났습니다. 이 연구는 BF16 + SR 전략이 기존의 MP 방식인 BF16 및 FP32 보다 우수한 성능을 보여주었다고 주장합니다.

- **Technical Details**: 연구에 따르면, 스토캐스틱 라운딩(SR)을 사용한 경우 Adam 옵티마이저에서의 암묵적 정규화 효과와 수렴 특성을 이론적으로 분석하였습니다. 또한, SR이 적용된 BF16-AdamW 옵티마이저는 Mixed Precision 훈련을 뛰어넘는 성능을 보였으며, 훈련 시 처리량을 향상시키고 메모리 사용량을 줄였습니다. 이러한 개선은 고속 학습률과 결합된 SR 훈련의 효과를 통해 이루어졌습니다.

- **Performance Highlights**: 실험 결과, 최대 6.7B 파라미터를 가진 모델을 사전 훈련한 결과, BF16 + SR 전략이 Mixed Precision 전략들보다 유의미하게 개선된 valdiation perplexity와 더불어 $1.54\times$ 높은 처리량 및 $30\%$ 적은 메모리 사용량을 기록했습니다. 이 연구는 훈련 효율성과 정확도에서 SR의 이점을 분명히 보여 주며, LLM 훈련의 적합성을 증명하고 있습니다.



### DPZV: Resource Efficient ZO Optimization For Differentially Private VFL (https://arxiv.org/abs/2502.20565)
- **What's New**: 본 논문은 Vertical Federated Learning (VFL)에서의 개인정보 보호 및 메모리 효율성을 향상시키기 위해 DPZV라는 새로운 메모리 효율적 Zeroth-Order (ZO) 최적화 프레임워크를 제안합니다. 이 프레임워크는 Differential Privacy (DP)의 개념을 포함하여 세 가지 주요 문제를 해결합니다: (1) gradient 누출로 인한 개인정보 취약점, (2) 첫 번째 순서 방법의 높은 계산 및 통신 비용, (3) 기존 ZO 방식의 지나치게 큰 메모리 사용량입니다.

- **Technical Details**: DPZV는 비동기 통신을 가능하게 하고, 두 점 gradient 추정을 통해 클라이언트의 메모리 사용량을 첫 번째 순서 방법에 비해 90%까지 줄입니다. 또한, 서버에서 가우시안 노이즈를 주입함으로써, 제3자의 신뢰 가정 없이 강력한 $(B5, B4)$-DP 보장을 달성합니다. 이론적으로는 비볼록 목표에서 중앙집중식 케이스와 동등한 수렴 속도를 보여줍니다.

- **Performance Highlights**: 이미지 및 NLP 벤치마크에서의 광범위한 실험 결과, DPZV가 모든 기준 모델에 비해 높은 정확도를 기록하며 강력한 개인정보 보호를 제공했습니다($B5  3C 10$). 자원 제한 환경에서도 계산 자원 요구량이 적어 새로운 최첨단 개인정보-유용성 트레이드오프를 확립하며, 실용적이고 안전하며 확장 가능한 federated learning 솔루션으로 발돋움합니다.



### Towards Statistical Factuality Guarantee for Large Vision-Language Models (https://arxiv.org/abs/2502.20560)
- **What's New**: 이 논문에서는 LVLM(대형 비전-언어 모델)의 신뢰성을 높이기 위해 ConfLVLM이라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 기존 LVLM이 생성한 텍스트가 시각적 맥락과 불일치하는 문제를 해결하기 위해 설계되었습니다. ConfLVLM은 통계적 가설 검정을 통해 생성된 텍스트의 신뢰성을 평가하고, 잘못된 주장을 사전에 필터링함으로써 사용자에게 정확한 정보를 제공합니다.

- **Technical Details**: ConfLVLM은 LVLM을 가설 생성기로 간주하고, 각 생성된 텍스트는 개별적인 주장으로 다룹니다. 이 프레임워크는 효율적인 불확실성 측정치를 사용하여 주장을 검증할 수 있도록 설계되었습니다. 통계적 가설 검정 절차를 활용하여, 잘못된 주장을 걸러내고 신뢰할 수 있는 답변만 사용자에게 제공합니다.

- **Performance Highlights**: ConfLVLM은 LLaVa-1.5에 의해 생성된 장면 설명의 오류율을 87.8%에서 10.0%로 대폭 줄였습니다. 이 과정에서 95.3%의 진정한 양성률(true positive rate)을 기록하여, 필터링의 효과성을 입증했습니다. 또한, ConfLVLM은 다양한 LVLM 및 불확실성 측정치와 조합되어 다양한 비전-언어 작업에 유연하게 적용될 수 있는 특징을 가지고 있습니다.



### $Q\sharp$: Provably Optimal Distributional RL for LLM Post-Training (https://arxiv.org/abs/2502.20548)
- **What's New**: 이 연구에서는 KL-정규화된 강화 학습(RL)을 위한 가치 기반 알고리즘인 $Q\sharp$을 소개합니다. 기존의 정책 기반 방법이 가진 한계를 개선하고자 하였으며, 최적의 정규화된 $Q$ 함수를 통해 레퍼런스 정책을 안내합니다. 특히 $Q\sharp$은 수학적 이론을 입증받은 방법으로, KL 정규화된 RL 문제에 대한 최적 정책을 학습하는 데 유리합니다.

- **Technical Details**: 이 알고리즘은 분포론적 강화 학습(distributional RL)을 활용하여 집계된 온라인 데이터셋에서 최적 $Q$ 함수를 학습합니다. $Q\sharp$는 정책의 가중치를 변경하지 않고도 레퍼런스 정책의 생성을 안내할 수 있으며, 이 과정에서 더 작은 모델을 사용해도 성능을 향상시킬 수 있습니다. 전통적인 RL 방법과 달리 $Q\sharp$은 복잡한 시계열 차이 학습(temporal difference learning) 없이 직접 감독 학습(supervised learning)으로 대체합니다.

- **Performance Highlights**: $Q\sharp$은 수학적 추론 벤치마크에서 이전의 기준 알고리즘들보다 뛰어난 성능을 보였습니다. 또한, 레퍼런스 정책과의 KL 발산을 줄임으로써 안정적인 학습을 도모하였습니다. 실험 결과는 $Q\sharp$이 LLM 후속 훈련에 효과적인 접근법임을 보여줍니다.



### SoS1: O1 and R1-Like Reasoning LLMs are Sum-of-Square Solvers (https://arxiv.org/abs/2502.20545)
- **What's New**: 본 연구에서는 대규모 언어 모델(LLMs)의 수학적 문제 해결 능력을 시험하기 위해, 주어진 다변수 다항식이 비음수인지 판단하는 문제를 조사합니다. 이 문제는 힐베르트의 17번째 문제와 밀접하게 관련되어 있으며, 전 세계적인 다항식 최적화에 핵심적인 역할을 합니다. 연구에 사용된 SoS-1K 데이터세트는 약 1,000개의 다항식으로 구성되어 있으며, 다섯 가지 점진적으로 어려워지는 기준에 기반한 전문가 설계의 추론 지침이 포함되어 있습니다.

- **Technical Details**: 다항식의 비음성 여부를 판단하는 문제는 NP-난해하다는 것이 알려져 있습니다. 연구에서는 기존의 SoS 조건을 활용하여 다항식을 제곱 다항식의 합으로 표현함으로써 이 문제를 해결하려고 합니다. SoS-1K 데이터세트를 바탕으로 여러 최신 LLM을 평가한 결과, 정형화된 지침 없이 모델들이 평균 50%의 정확도를 기록했으나, 고품질의 추론 지침을 제공했을 때 성능은 81%까지 증가했습니다.

- **Performance Highlights**: 전문가 설계의 추론을 포함한 SoS-1K 데이터세트로 fine-tuning한 SoS-7B 모델은 671B의 DeepSeek-V3 및 GPT-4o-mini 모델보다 더 높은 정확도를 기록했습니다. 특히, SoS-7B는 답변 속도 면에서도 우수한 성과를 보였으며, 더 낮은 연산 시간을 요구했습니다. 이러한 결과는 LLM이 NP-난해 문제를 해결하는 데 있어 중요한 가능성을 가지고 있음을 보여줍니다.



### Revisiting Kernel Attention with Correlated Gaussian Process Representation (https://arxiv.org/abs/2502.20525)
Comments:
          21 pages, 4 figures

- **What's New**: 이 논문에서는 Correlated Gaussian Process Transformer (CGPT)라는 새로운 변형의 트랜스포머를 제안합니다. 기존 트랜스포머의 한계인 대칭 주의 메커니즘을 탈피하여 비대칭성을 허용하여 모델의 표현 능력을 향상시킵니다. CGPT는 두 개의 상관된 Gaussian Process (GP) 간의 교차 공분산을 모델링하여 불확실성을 효과적으로 보정할 수 있습니다.

- **Technical Details**: CGPT는 다중 헤드 자기 주의 (multi-head self-attention) 메커니즘의 자기 주의 단위를 두 개의 상관된 GP 간의 공분산으로 표현합니다. 이를 통해 각각의 쿼리와 키 기능을 배정하는 서로 다른 아핀 변환을 사용할 수 있습니다. 또한 CGPT는 입출력 토큰 수의 세제곱 의존성을 제거하는 희소 근사법을 개발하여 효율성을 증가시킵니다.

- **Performance Highlights**: 실험 결과에 따르면 CGPT와 희소 CGPT 기반 트랜스포머는 여러 컴퓨터 비전 및 자연어 처리 벤치마크에서 기존의 GP 기반 트랜스포머보다 더 나은 예측 성능을 보여주었습니다. 이러한 성능 향상은 CGPT의 새로운 구조적 특징에 기인합니다. 논문에서 제안된 방법은 다양한 안전 및 신뢰성 중심 작업에 적용될 수 있습니다.



### Data Distributional Properties As Inductive Bias for Systematic Generalization (https://arxiv.org/abs/2502.20499)
- **What's New**: 이 연구에서는 딥 신경망(Deep Neural Networks, DNNs)의 체계적 일반화(systematic generalization, SG) 능력을 증진시키기 위한 훈련 데이터의 속성 역할을 탐구합니다. 특히, 데이터의 다양성(data diversity), 버스티니스(burstiness), 잠재적 개입(latent intervention)이라는 세 가지 속성이 SG에 미치는 영향을 분석했습니다. 이 연구는 DNN이 새로운 작업에 유연하게 적응할 수 있도록 돕는 데이터 속성을 이해하는 데 기여합니다.

- **Technical Details**: 연구에서는 다양한 데이터 분포 속성을 induce(유도)하는 방법을 통해 SG를 증진시키는 방법을 설명합니다. 데이터 다양성은 잠재적인 속성이 취할 수 있는 값의 범위를 확대하는 것을 포함하며, 버스티니스는 특정 입력에 대한 잠재적 요소의 값의 수를 확률적으로 제한합니다. 마지막으로, 잠재적 개입은 훈련 중 특정 잠재적 요소의 값을 무작위로 변경하여 SG를 높이는 데 기여합니다.

- **Performance Highlights**: 연구 결과, 이 세 가지 요소 모두 SG를 크게 향상시키며, 특히 데이터 다양성이 절대 정확도를 89%까지 증가시키는 데 기여했습니다. 버스티니스는 기준선에 비해 15%의 SG 증가를 가져오며, 잠재적 개입 또한 15% 증가를 보여주었습니다. 이 연구는 SG를 촉진하는 특정 데이터 속성의 기능적 설명을 제공하는 중요한 기초를 마련합니다.



### Unified Kernel-Segregated Transpose Convolution Operation (https://arxiv.org/abs/2502.20493)
- **What's New**: 본 논문에서는 딥러닝 애플리케이션을 위한 transpose convolution layer의 최적화를 위해 kernel segregation 메커니즘을 제안합니다. 기존의 kernel segregation이 가지는 단점, 즉 스레드를 실행할 때 홀수 차원을 가진 출력 특성 맵을 얻기 위해 추가 요소를 계산해야 하는 문제를 해결하고자 합니다. 이를 위해, 네 개의 서브 커널을 실행하기 위해 하나의 통합된 커널을 사용하는 통합 커널 세그리게이션 접근 방식을 도입하였습니다.

- **Technical Details**: 이 접근 방식은 메모리와 계산 자원의 사용을 제한하면서 더욱 효율적으로 transpose convolution layer를 최적화합니다. 연구 결과에 따르면, RTX 2070 GPU를 사용할 때 특정 데이터셋에서 평균 2.03배의 계산 속도 향상이 이루어졌고, Intel Xeon CPU에서는 3.89배의 향상이 있음을 보여줍니다. 또한, 널리 알려진 Generative Adversarial Networks(GANs)의 transpose convolution layers를 평가했을 때 평균 3.5배의 계산 속도 향상을 보였습니다.

- **Performance Highlights**: 제안된 방법을 EB-GAN 모델의 transpose convolution layers에 적용한 결과, 최대 35MB의 메모리 절약을 이루었습니다. 이는 메모리 효율성을 극대화하면서도 계산 성능을 향상하는 데 중요한 기여를 합니다. 이러한 개선은 앞으로의 딥러닝 모델 구축에 있어 매우 유용할 것으로 기대됩니다.



### Unifying Model Predictive Path Integral Control, Reinforcement Learning, and Diffusion Models for Optimal Control and Planning (https://arxiv.org/abs/2502.20476)
- **What's New**: 이번 논문에서는 Model Predictive Path Integral (MPPI) 제어, Reinforcement Learning (RL), 그리고 Diffusion Models를 하나의 통합된 관점에서 연결하는 방법을 제시하고 있습니다. 이 연구는 각 방법론이 독립적인 최적화 프레임워크로 취급되어 온 전통적인 관점을 탈피하여, Gibbs measure에 대한 gradient 기반 최적화 통해 이들을 연관짓습니다. MPPI가 부드러운 에너지 함수에서의 gradient 상승을 수행하는 것으로 해석될 수 있음을 보여줍니다.

- **Technical Details**: 사전 지식 섹션에서는 Zeroth-order (ZO) 최적화가 첫 번째 주문(First-order, FO) 최적화와 달리, 명시적인 gradient 정보 없이도 목적 함수를 최적화할 수 있는 방법으로 설명됩니다. ZO 최적화는 Gradient 기반 알고리즘의 수정으로 간단히 구현할 수 있으며, MPPI는 효율적인 온라인 motion planning을 위한 유연한 제어 방법으로 자리 잡고 있습니다. 중요한 수식을 통해 MPPI 업데이트 규칙을 고찰하고, Bayesian prior와 결합된 MPPI의 확장된 형태도 다루고 있습니다.

- **Performance Highlights**: 논문에서는 정책 매개변수를 제어 변수로 간주할 때 Policy Gradient 방법이 MPPI로 축소될 수 있음을 보여주며, 이는 Reinforcement Learning에서 중요한 역할을 하는 기법입니다. 또한, Denoising diffusion models를 통해 단순한 prior 분포에서 복잡한 데이터 분포로 매핑하는 과정을 설명하며, 개별 데이터 포인트 수준에서의 최적화 문제를 다룹니다. 이러한 새롭게 제시된 통합 접근법은 MPPI, RL, 그리고 Diffusion Models 간의 관계를 명확히 하고 성능을 향상시키는 데 기여할 수 있을 것으로 기대됩니다.



### MobiLLM: Enabling LLM Fine-Tuning on the Mobile Device via Server Assisted Side Tuning (https://arxiv.org/abs/2502.20421)
- **What's New**: MobiLLM(모바일 대형 언어 모델)는 서버 지원 사이드 튜닝을 통해 모바일 기기에서 메모리 효율적인 변환기 LLM 미세 조정을 가능하게 합니다. 기존의 미세 조정 방법과 달리, MobiLLM은 고정된 백본 모델을 유지하면서 메모리와 계산 집약적인 역전파를 고성능 서버로 오프로드합니다. 이를 통해 저사양 모바일 기기에서도 LLM 미세 조정이 가능해지며, 데이터의 프라이버시도 유지됩니다.

- **Technical Details**: MobiLLM은 고정된 사전 훈련된 모듈과 학습 가능한 사이드 네트워크로 LLM 미세 조정 작업을 분리하여 수행합니다. 이 네트워크는 모바일 기기와 서버 간의 인터레이어 활성화를 교환하도록 설계되어, 모바일 기기에서는 메모리와 계산 효율적인 포워드 패스를 유지하면서 서버에서 역전파를 처리합니다. 이를 통해 MobiLLM은 모바일 기기의 메모리 및 계산 한계를 극복하고, 데이터가 모바일 기기를 떠나지 않도록 보장합니다.

- **Performance Highlights**: MobiLLM은 NVIDIA Xavier와 CPU 전용 노트북 등을 포함한 여러 모바일 기기에 배포되어 평가되었습니다. 결과적으로, MobiLLM은 메모리 제한이 있는 모바일 기기에서도 기가바이트 규모의 LLM을 미세 조정할 수 있으며, 최신 기법(SOTA)보다 메모리 사용량을 4배 감소시키고, 수렴 시간을 2.3배 단축시키는 성과를 보여주었습니다.



### Unsupervised Parameter Efficient Source-free Post-pretraining (https://arxiv.org/abs/2502.21313)
- **What's New**: 이 논문은 UpStep이라는 새로운 접근 방식을 제안하여, 사전 학습된 모델을 소스 도메인과는 독립적으로 목표 도메인에 효율적으로 적응시키는 방법입니다. UpStep은 라벨이 없는 목표 도메인에서 자기 지도(self-supervised) 학습 방식으로 모델을 조정하는 데 집중하며, 재학습 없이도 파라미터의 낭비를 최소화합니다. 이 접근은 카타스트로픽 포겟팅(catatstrophic forgetting) 문제를 해결하기 위해 CVR(center vector regularization)이라는 보조 기능을 도입하였습니다.

- **Technical Details**: UpStep은 임계점 없이 소스 도메인 데이터에 접근하지 않고도 사전 학습된 모델을 조정할 수 있도록 설계되었습니다. 이 과정에서 고정 클러스터 센터를 사용한 두 가지 스트림의 자기 지도 클러스터링 방식을 채택하였으며, 각 스트림은 동일한 아키텍처를 공유하면서 모델의 효율성을 극대화합니다. 또한, 모델을 적응시키는 과정에서 저차원(low-rank) 적응 기법을 사용하여 필요한 파라미터 수를 줄입니다.

- **Performance Highlights**: 여러 종류의 일반화된 기반 모델을 사용하여 다양한 목표 도메인에서 UpStep의 적응성과 일반화 능력을 입증하였습니다. 이 접근법은 목표 도메인에 대해 사전 학습된 모델을 최적화하는 데 필요한 파라미터 수를 크게 줄이며, 비용 효율적인 방법으로 평가되었습니다. 실험 결과, 다양한 데이터셋에서 UpStep의 성능이 뛰어난 것으로 나타났습니다.



### FANformer: Improving Large Language Models Through Effective Periodicity Modeling (https://arxiv.org/abs/2502.21309)
- **What's New**: 이 연구는 최근의 대형 언어 모델(LLMs)들이 데이터와 계산 자원에 대한 높은 요구로 인해 학습 효율성이 낮다는 점을 지적합니다. 새로운 모델 FANformer를 제안하여 Fourier Analysis Network (FAN)를 Transformer의 attention 메커니즘에 통합하여 주기성(periodicity) 모델링을 보다 효율적으로 수행하고 있습니다. FANformer는 Transformer 아키텍처의 학습 효율성과 성능을 개선하기 위해 주기적 패턴을 포착하고 표현하는 Fourier 원리를 활용합니다. 이 연구 결과는 FANformer가 LLM을 발전시키기 위한 유망한 아키텍처가 될 수 있음을 보여줍니다.

- **Technical Details**: FANformer는 샘플 입력 𝐗𝐗oldsymbol{X}의 폴리폼 및 주기 기저 함수(frequency basis functions)를 사용하는 FAN 레이어를 기반으로 구성됩니다. FAN 레이어는 행동 기능을 유지하면서 주기적 패턴을 explicit하게 인코딩하여 MLP 레이어와 차별화됩니다. 또한 attention 메커니즘의 feature projection 과정을 수정하여 frequency domain representations을 통합하고 있습니다. 이러한 구조는 주기성을 효과적으로 캡처하고 모델링할 수 있도록 설계되었습니다.

- **Performance Highlights**: FANformer는 1조 개의 토큰을 활용하여 11억 파라미터의 FANformer-1B를 사전 훈련하였으며, 결과적으로 오픈 소스 LLM보다 뛰어난 성능을 보였습니다. 모델 파라미터와 훈련 토큰이 같은 조건에서 FANformer는 기존의 Transformers보다 적은 자원으로 유사한 성능을 발휘하면서 뛰어난 학습 효율성을 가지고 있음을 확인했습니다. FANformer의 학습 효율성이 더 크고 복잡한 모델에서 더욱 돋보이며, 이는 LLM의 발전 가능성을 시사합니다.



### Contextualizing biological perturbation experiments through languag (https://arxiv.org/abs/2502.21290)
Comments:
          The Thirteenth International Conference on Learning Representations (2025)

- **What's New**: 이 연구에서 제안된 PerturbQA는 생물학적 동원 실험의 결과를 정량적으로 쿼리하는 기준 벤치마크입니다. 기존에 사용되던 벤치마크와는 달리, PerturbQA는 차별적 표현 같은 기존 실험에서의 미해결 문제를 해결하는 데 중점을 두고 있습니다. 이 연구에서는 Summer라는 LLM 프레임워크도 소개되었으며, 이는 현재의 최첨단 기술을 능가하거나 맞먹는 성능을 보입니다.

- **Technical Details**: PerturbQA는 생물학적 동원 실험의 결과를 정량적으로 묻는 질문-응답 방식으로 구성됩니다. 이 시스템은 언어 모델을 이용하여 생물학적 지식 그래프 및 실험 데이터를 통합하여 생물학적 질문에 대한 응답을 도출합니다. Summer 프레임워크는 LLM을 사용하여 유전자에 대한 설명을 요약하고, 이 정보를 기반으로 실험 결과를 조회하며, 마지막으로 비슷한 결과를 보이는 유전자 클러스터를 특성화합니다.

- **Performance Highlights**: 연구 결과 현재의 최첨단 기계 학습 및 통계적 접근 방식이 PerturbQA에서 저조한 성능을 보이는 것으로 나타났습니다. 그러나 Summer는 이러한 기준에서 훌륭한 성능을 발휘하며, 모든 조정 없이 생물학자가 이해할 수 있는 언어로 작동하는 경량 모델입니다. 이 연구는 기계 학습 모델링의 접근성과 관심을 높일 수 있는 기반이 될 것입니다.



### Enabling AutoML for Zero-Touch Network Security: Use-Case Driven Analysis (https://arxiv.org/abs/2502.21286)
Comments:
          Published in IEEE Transactions on Network and Service Management (TNSM); Code is available at Github link: this https URL

- **What's New**: 제로 터치 네트워크(Zero-Touch Networks, ZTNs)는 차세대(6G) 네트워크를 위한 완전 자동화되고 지능적인 네트워크 관리의 패러다임으로, 인공지능(AI)과 머신 러닝(ML)을 활용해 운영 효율성을 높이고 자원 배분을 최적화하는 것을 목표로 합니다. 그러나 ZTN의 구현은 AI/ML 기반 보안 기법의 개발 시 필요한 전문 인력 부족과 같은 보안 도전과제에 직면해 있습니다. 이 논문에서는 이러한 보안 문제에 대한 이론적 검토 및 AutoML의 잠재력을 포함한 해결책을 제시합니다.

- **Technical Details**: ZTNs는 자가 구성(self-configuration), 자가 복구(self-healing), 자가 최적화(self-optimization) 등 중요한 자가 관리 기능을 포함합니다. 또한, 머신 러닝(ML) 알고리즘을 통한 네트워크 및 서비스 관리 개선이 이루어지고 있으나, 이러한 알고리즘 개발은 여전히 많은 전문가의 손길을 필요로 합니다. 자동화된 머신 러닝(AutoML) 기법은 데이터 분석과 ML 절차의 자동화를 통해 ZTN 구현의 가능성을 높이는데 도움을 줄 수 있습니다.

- **Performance Highlights**: ZTNs는 AI/ML과 같은 최신 기술을 활용하여 서비스 효율성, 성능 및 시간 관리에서 큰 개선을 가져오는 것을 목표로 합니다. 본 논문에서는 AutoML 기술을 활용한 두 가지 실제 사례를 통해 ZTN 보안 작업을 적용하고, AML(Adversarial Machine Learning) 공격에 대한 사이버 방어 훈련을 수행했습니다. 이를 통해 ZTNs 영역의 보안 강화를 위한 실질적인 접근법이 제시되었습니다.



### Adaptive Keyframe Sampling for Long Video Understanding (https://arxiv.org/abs/2502.21271)
Comments:
          CVPR2025

- **What's New**: 본 논문은 비디오 이해를 위한 Adaptive Keyframe Sampling (AKS)이라는 새로운 알고리즘을 제안합니다. 이는 기존의 비디오 기반 MLLM들이 비디오 토큰을 선택하는 방식에서 발생하는 정보 손실 문제를 해결하고자 합니다. AKS는 고정된 수의 비디오 토큰으로 유용한 정보를 극대화하는 데 중점을 두며, 키프레임 선택을 통해 비디오의 관련성과 커버리지를 동시에 고려합니다.

- **Technical Details**: 이 연구는 키프레임 선택을 최적화 문제로 모델링하여, 키프레임과 질문 간의 관련성을 측정하고, 키프레임 집합이 비디오에서 유용한 정보를 얼마나 잘 커버하는지를 평가합니다. 전체 알고리즘은 비디오와 텍스트 입력을 결합하여 최적의 키프레임을 선택하고, 고품질의 비주얼 데이터를 기반으로 MLLM의 성능을 향상시키는데 초점을 맞추고 있습니다. AKS는 LLaVA-Video와 같은 기존 MLLM에 통합되어 성능 개선을 확인합니다.

- **Performance Highlights**: AKS는 LongVideoBench와 VideoMME에서 비디오 질문 응답의 정확도를 개선하는 데 성공하였습니다. 본 연구에서 제안된 알고리즘은 고품질 키프레임의 발견 덕분에 MLLM의 전반적인 성능을 향상시켰음을 입증하였습니다. 실험 결과, AKS가 통합된 모델은 모든 테스트에서 일관된 정확도 향상을 보여주며, 이로 인해 새로운 벤치마크 기록을 세우게 되었습니다.



### Dynamical Decoupling of Generalization and Overfitting in Large Two-Layer Networks (https://arxiv.org/abs/2502.21269)
Comments:
          89 pages; 62 pdf figures

- **What's New**: 이 논문은 대규모 기계 학습 모델의 유도 편향(inductive bias)과 일반화(generalization) 특성이 훈련에 사용된 최적화 알고리즘의 부산물임을 강조합니다. 특히 랜덤 초기화(random initialization), 학습률(learning rate), 조기 중단(early stopping) 등이 모델 품질에 중요한 영향을 미친다는 것을 보여줍니다. 저자들은 두 층 신경망의 훈련 역학(training dynamics)을 연구하여 이러한 현상을 이해하고자 합니다.

- **Technical Details**: 논문에서는 비평형 통계 물리(non-equilibrium statistical physics)에서 기인한 동적 평균장 이론(dynamical mean field theory)을 활용하여 훈련 역학을 고차원적으로 설명합니다. 이 분석은 숨겨진 뉴런의 비선형성(non-linearity)에 대한 가우시안 근사(Gaussian approximation)를 기반으로 하며, 실제 신경망 모델의 행동을 잘 포착합니다. 저자들은 훈련 역학에서 여러 흥미로운 새로운 현상을 발견하였습니다.

- **Performance Highlights**: 첫째, 가우시안/Rademacher 복잡성의 성장에 연관된 느린 시간 척도가 나타납니다. 둘째, 초기화가 복잡성이 낮을 경우에만 작은 복잡성에 대한 알고리즘적 유도 편향이 발생합니다. 셋째, 특성 학습(feature learning)과 과적합(overfitting) 간의 시간 척도가 분리되며, 넷째, 테스트 오류가 비단조적(non-monotone)으로 변화하며, 큰 시간에서 '특성 비학습(feature unlearning)' 단계가 존재함을 보여줍니다.



### Modeling Human Beliefs about AI Behavior for Scalable Oversigh (https://arxiv.org/abs/2502.21262)
Comments:
          53 pages

- **What's New**: 이번 연구에서는 AI 시스템의 인간 피드백 신뢰성이 떨어지는 문제를 다룹니다. AI 시스템이 발전함에 따라 인간의 능력을 초과하게 되었고, 이에 따라 AI 시스템을 감독하는 문제에 대한 새로운 접근 방식을 제안합니다. 연구진은 인간 평가자의 믿음을 모델링하여 AI 시스템의 행동에 대한 피드백을 더 효과적으로 해석할 수 있는 방법을 모색합니다.

- **Technical Details**: 연구에서는 인간 믿음 모델(human belief models)을 공식화하고 이 모델들이 인간의 가치를 추론하는 데 어떤 역할을 하는지 이론적으로 분석합니다. 또한 이러한 추론에서 남아 있는 모호성을 특성화하고, 모호성이 사라지는 조건을 규명합니다. 정확한 믿음 모델에 대한 의존성을 줄이기 위해, 인간 믿음 모델의 커버링(relaxation of human belief model covering) 개념을 도입합니다.

- **Performance Highlights**: 기본 모델(foundational models)을 활용하여 커버링 신념 모델을 구성하는 방안을 제안하였습니다. 이는 AI 시스템의 확장 가능한 감독(sc可alability oversight) 문제에 대한 새로운 접근법을 제공할 것으로 기대됩니다. 이러한 연구 결과는 AI alignment 분야에 중요한 기여를 할 것으로 보입니다.



### The Structural Complexity of Matrix-Vector Multiplication (https://arxiv.org/abs/2502.21240)
Comments:
          36 pages

- **What's New**: 이 논문은 n×n 행렬 M의 전처리 문제를 다루며, 벡터 v에 대해 행렬-벡터 곱 Mv를 반환하는 쿼리 지원에 대한 연구를 심화했습니다. 기존의 이론과 실제 사이의 간극을 해소하기 위해, VC 차원 d를 가진 구조화된 행렬에 대해 O(n²) 전처리 및 O(n²-1/d) 쿼리 시간으로 문제를 해결할 수 있음을 보여주었습니다. 이 결과는 실제 데이터에서 낮은 VC 차원을 관측함으로써 문제를 더 빠르게 해결할 수 있는 이유를 설명합니다.

- **Technical Details**: 구조화된 행렬에 대한 행렬-벡터 곱 문제는 VC 차원 d가 주어질 때 O(n²) 전처리와 O(n²-1/d) 쿼리 시간을 통해 해결할 수 있습니다. 또한, 저자들은 저차원 VC 행렬의 항목을 서브쿼드라틱 수 만큼만 손상시킨 행렬에 대해서도 이러한 경계가 유지되며, 이는 많은 응용에 대한 첫 번째 비트리비얼 상한을 제공합니다. 이 논문은 온라인 행렬-벡터 가설을 수정하여 고정밀 문제를 효율적으로 해결할 수 있는 방법을 제시합니다.

- **Performance Highlights**: 저자들은 이전 연구들에서의 O(n²/log n) 시간 경계와 비교해, 입력이 구조화된 경우 효과적으로 처리할 수 있음을 보여주며, 높은 정확도의 경우 서브쿼드라틱 상한을 제공합니다. 이는 동적 알고리즘에 대한 기존의 시간 하한을 위반하지 않고도 문제를 해결할 수 있는 새로운 방법을 제시하며, 특히 그래프에서의 노드 삽입 및 삭제에 강력한 영향을 미칩니다. 이러한 연구 결과는 머신러닝과 최적화 분야에서 실질적인 성과를 낼 수 있는 기초를 제공합니다.



### ByteScale: Efficient Scaling of LLM Training with a 2048K Context Length on More Than 12,000 GPUs (https://arxiv.org/abs/2502.21231)
Comments:
          12 pages, 21 figures

- **What's New**: 최근 대규모 언어 모델(LLM)의 긴 맥락 처리가 필수적이게 되었습니다. 이에 따라 ByteScale이라는 새로운 훈련 프레임워크가 제안되었습니다. 이 프레임워크는 Hybrid Data Parallelism(HDP)이라 불리는 새로운 병렬 처리 전략을 통해 장단기 시퀀스를 효율적으로 혼합하여 훈련할 수 있도록 설계되었습니다.

- **Technical Details**: ByteScale의 핵심은 기존의 inter-data partitioning(데이터 병렬 처리)과 intra-data partitioning(맥락 병렬 처리)을 통합하는 HDP입니다. 이 방식은 동적 메쉬 디자인을 활용하여, 각 시퀀스가 각기 다른 길이를 가질 때에도 유연하게 처리가 가능합니다. 또한 통신 최적화 기법을 통해 짧은 시퀀스에서 불필요한 통신을 줄이고, 긴 시퀀스에서는 선택적 오프로드를 통해 통신 비용을 축소합니다.

- **Performance Highlights**: 12,000개 이상의 GPU가 운영되는 생산 클러스터에서 7B에서 141B까지 다양한 모델 크기와 256K에서 2048K까지의 맥락 길이로 ByteScale을 평가했습니다. 실험 결과, ByteScale은 기존 훈련 방법에 비해 최대 7.89배의 속도 향상을 달성하여 훈련 효율성을 크게 개선했습니다.



### ARIES: Autonomous Reasoning with LLMs on Interactive Thought Graph Environments (https://arxiv.org/abs/2502.21208)
- **What's New**: 본 논문에서는 큰 언어 모델(LLM)이 생각 그래프(thought graph) 변환을 마르코프 결정 과정(Markov Decision Process, MDP)의 행동으로 보고, 효율적인 액션 정책을 제안합니다. 특히, ARIES라는 다중 에이전트 아키텍처를 통해 LLM이 서브 문제를 해결하고 생각 그래프 상태를 모니터링하며, 문제 해결 전략을 동적으로 조정합니다. 이를 통해 LLM이 상황에 따라 자동으로 탐색 전략을 조정할 수 있는 가능성을 제시합니다.

- **Technical Details**: ARIES 프레임워크에서는 정책 LLM 에이전트가 상태를 관찰하고, 특정 액션을 선택하여 문제를 해결하는 과정이 포함됩니다. LLM은 생각 그래프 환경에서 피드백을 수집하여 탐색 전략을 동적으로 조정하며, 기존의 정적 transform schedule을 초월할 수 있는 가능성을 보여 줍니다. 이 연구는 생각 그래프를 독립적으로 해결할 수 있는 서브 문제로 나누어, 각 에이전트가 협력하여 문제를 해결하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 비지도 미세 조정(supervised fine-tuning, SFT)을 하지 않은 LLM을 정책 에이전트로 사용할 경우, 정적 변환 일정에 비해 HumanEval에서 최대 29% 더 높은 정확도를 얻었으며, 추론 비용을 35% 줄이고, 탐색 비용이 필요 없음을 확인했습니다. 기존 연구와의 비교를 통해, 우리의 접근 방식이 LLM의 계획 및 추론 능력을 극대화할 수 있음을 입증하고 있습니다.



### AMPLE: Event-Driven Accelerator for Mixed-Precision Inference of Graph Neural Networks (https://arxiv.org/abs/2502.21196)
- **What's New**: 이번 논문에서 제안하는 AMPLE(Accelerated Message Passing Logic Engine)는 이벤트 기반 프로그래밍 흐름을 활용하여 GNN(그래프 신경망) 추론을 최적화하는 FPGA(전계 프로그래머블 게이트 어레이) 가속기입니다. GNN은 비유클리드 데이터에서 성능을 발휘하는 강력한 방법으로 주목받고 있으며, AMPLE은 특히 불규칙한 노드 분포로 인한 비효율적인 메모리 접근 문제를 해결하고자 합니다. 이 아키텍처는 다양한 정밀도의 다중 집합 Aggregation Core를 통합하여 노드의 차수와 정밀도에 따라 동적으로 할당하는 특징이 있습니다.

- **Technical Details**: IMPLE의 아키텍처는 메모리 맵 된 레지스터를 통해 비동기적으로 노드를 프로그래밍할 수 있는 이벤트 기반 프로그래밍 모델을 제공하며, 그래프 데이터의 다양성을 효과적으로 관리합니다. GNN 추론을 노드 수준에서 양자화할 수 있도록 혼합 산술 아키텍처가 개발되었습니다. 또한 외부 메모리에 대한 데이터 및 명령 프리패쳐를 구현하여 메모리 접근 최적화 및 노드의 병렬성을 극대화합니다.

- **Performance Highlights**: 2K에서 700K 노드 범위의 데이터셋에 대한 성능 평가에서 AMPLE은 CPU 및 GPU에 비해 각각 평균 243배 및 7.2배의 속도를 기록했습니다. 이러한 성능 향상은 GNN의 집합(gathering) 단계의 비효율성을 극복하고 연산 패턴을 최적화한 결과로, 향후 GNN 처리 시스템의 발전에 기여할 것으로 기대됩니다.



### Class prior estimation for positive-unlabeled learning when label shift occurs (https://arxiv.org/abs/2502.21194)
- **What's New**: 이 연구는 새로운 클래스 우선도(class prior) 추정기를 제안합니다. 이 추정기는 포스터리어 확률(posterior probabilities)의 추정을 피하고 기하학적 해석을 쉽게 제공합니다. 특히, 긍정적-비표기 데이터(positive-unlabeled data)를 이용한 상황에서의 문제를 다루고 있으며, 기존 방법의 한계를 극복하는 방법을 모색합니다.

- **Technical Details**: 제안된 새로운 추정기는 분포 매칭(distribution matching) 기법과 커널 임베딩(kernel embedding)을 기반으로 합니다. 이를 통해 최적화 과제로서 명시적 해를 구함으로써, 높은 신뢰성을 확보할 수 있습니다. 또한, 고전적인 우선도 추정기인 KM 추정기를 클래스 우선도 클래스 시프트(class prior shift) 상황에 맞게 조정하는 방법도 설명합니다.

- **Performance Highlights**: 다양한 합성 및 실제 데이터에 대한 실험 결과, 제안된 방법은 기존 방법들과 비슷하거나 더 나은 성능을 보였습니다. 특히, 클래스 우선도 확률의 추정에서 새로운 접근법이 효과적임을 입증하였습니다. 이 연구의 결과는 다양한 분야에서 PU 데이터 문제를 해결하는 데 기여할 것으로 기대됩니다.



### HQColon: A Hybrid Interactive Machine Learning Pipeline for High Quality Colon Labeling and Segmentation (https://arxiv.org/abs/2502.21183)
- **What's New**: 이 논문에서는 고해상도 대장 세분화를 위한 최초의 완전 자동화된 방법을 제시하고 있습니다. 기존의 오픈 소스 도구인 TotalSegmentator가 대장의 복잡한 형태로 인해 정확성을 보장하지 못하는 문제를 해결하기 위해, 연구진은 새로운 세분화 모델을 개발하는데 필요한 데이터세트를 생성했습니다. 이 데이터셋은 CT 대장조영술 이미지의 435개 레이블이 있는 샘플로 구성되어 있으며, 이를 기반으로 nnU-Net 모델을 훈련하여 세분화의 정확성을 획기적으로 향상시켰습니다.

- **Technical Details**: 研究者들은 825명의 대장 내시경 검사가 필요한 환자들을 대상으로 한 공개된 CTC 데이터셋을 사용하여 연구를 진행했습니다. 초기 세분화를 위해 용이한 주석 생성을 위한 전통적인 방법을 적용하고, 대장 내용물의 세분화를 위한 RootPainter라는 인터랙티브 머신러닝 방법을 활용하였습니다. 모델을 학습하는 과정에서 3D nnU-Net v2를 사용하고, 입력 및 마스크 이미지 두 가지 데이터셋을 통해 대장 전체를 포함한 세분화 모델을 훈련시켰습니다.

- **Performance Highlights**: 최종적으로, 연구진의 모델은 평균 대칭 표면 거리(ASSD) 0.2 mm, 95번째 백분위수 하우스도르프 거리(HD 95%) 1.0 mm를 기록하며, 기존의 TotalSegmentator의 결과보다 상당히 개선된 성능을 보여줍니다. 이 모델은 고해상도 대장 세분화에서 훨씬 더 높은 정확성을 제공하며, 연구진은 해당 훈련된 모델과 코드도 민간에 공개하여 접근성을 높였습니다.



### Multimodal Dreaming: A Global Workspace Approach to World Model-Based Reinforcement Learning (https://arxiv.org/abs/2502.21142)
Comments:
          Under review in a conference

- **What's New**: 이번 연구에서는 세계 모델(World Models)과 글로벌 작업(Global Workspace) 이론을 결합한 새로운 강화 학습 시스템(GW-Dreamer)을 평가합니다. 기존의 세계 모델은 환경 변수를 직접 처리하여 학습 속도가 느린 문제를 가지고 있었습니다. 이번 연구는 고차원 잠재 공간(latent space)에서 정신 시뮬레이션(dreaming process)을 수행하면 환경 단계가 적어도 훈련 가능하다는 것을 보여줍니다. 이로 인해 다양한 상황에 적응하는데 효과적인 모델을 제안합니다.

- **Technical Details**: GW-Dreamer는 기존의 정책 최적화 알고리즘인 PPO와 원래의 Dreamer 알고리즘 여러 버전과 비교되었습니다. 연구 결과, GW 잠재 공간 내에서 진행되는 정신 시뮬레이션이 환경 단계 수를 줄이고, 훈련을 효과적으로 수행하는 데 도움을 줍니다. 이 시스템은 또한 이미지나 시뮬레이션 속성과 같은 관찰 모달리티 하나의 부재에도 강한 견고성을 보여줍니다.

- **Performance Highlights**: GW-Dreamer 모델은 기존의 비교 기준 모델들에 비해 우수한 성능을 보여줍니다. 특히, GW와 세계 모델의 결합이 강화 학습 에이전트의 의사결정 능력을 크게 향상시키는 잠재력을 보입니다. 이러한 결과는 강화 학습 분야에 새로운 접근 방식을 제시하며, 다양한 상황에 대한 적응력을 증가시킬 수 있습니다.



### Microscopic Propagator Imaging (MPI) with Diffusion MRI (https://arxiv.org/abs/2502.21129)
- **What's New**: 이번 연구에서는 Microscopic Propagator Imaging (MPI)라는 새로운 방법을 제안하여, 신경 조직의 미세 구조 내에서 물의 확산에 따른 이동 확률 밀도 함수(microscopic propagator)를 정확하게 밝혀낼 수 있는 지표를 제공합니다. 기존의 Ensemble Average Propagator (EAP)와는 달리, MPI 지표는 조직의 메조스코픽(mesoscopic) 조직 구성에서 독립적이기 때문에 더욱 구체적인 마이크로 구조에 대한 정보를 제공합니다.

- **Technical Details**: MPI 방법론은 구형 조화 함수(spherical harmonics)의 영역 모델링(zonal modeling), 신호 시뮬레이션(signal simulation), 그리고 머신러닝 회귀(machine learning regression)에 기반을 두고 있습니다. 이는 합성 데이터(synthetic data)와 실인간 확산 MRI 데이터(Human Diffusion MRI data)에서 적용되어 그 효과를 입증합니다. MPI는 여러 축 방향(bundle directions)과 방향 분산(orientation dispersion)의 영향을 받지 않는 마이크로 구조의 지표를 분리하여 제공합니다.

- **Performance Highlights**: MPI 지표는 조직 내의 미세 구조, 즉 축(axons)과 세포(cells)의 존재, 크기, 형태를 보다 직접적으로 반영할 수 있습니다. 이는 미세 구조의 변화와 관련된 MPI 지표의 변화를 더욱 명확하게 연결할 수 있게 만들어, 기존 기술의 한계를 극복하는 데 기여합니다. 이러한 특성으로 인해, MPI는 신경조직의 상태를 평가하는 데 더 나은 신뢰성을 제공할 것으로 기대됩니다.



### The two filter formula reconsidered: Smoothing in partially observed Gauss--Markov models without information parametrization (https://arxiv.org/abs/2502.21116)
Comments:
          14 pages, 2 figures

- **What's New**: 이 논문에서는 부분적으로 관찰된 Gauss-Markov 모델에 대한 두 가지 필터 공식을 재검토합니다. 전통적인 접근 방식에서는 이 필터가 시간 역으로 실행되며, 가우시안 밀도가 ``정보 형식''으로 매개변수화됩니다. 그러나 후방 재귀에서 발생하는 양은 분포가 아니라 우도(likelihood)임을 지적하며, 이를 바탕으로 로그-2차 우도를 기반으로 한 재귀를 제시하여 ``정보'' 매개변수화의 필요성을 없앱니다.

- **Technical Details**: 제안된 접근 방식은 알고리즘의 제곱근 공식(square-root formulation)을 크게 단순화합니다. 또한 예상되는 우도 표현에서 경로에 대한 사후 확률의 전방 마르코프 표현을 생성하는 공식도 제시됩니다. 이 구성 요소들은 전체 모델의 효율성과 유연성을 강화합니다.

- **Performance Highlights**: 이 새로운 접근 방식은 기존의 방법들보다 더 쉽게 적용할 수 있으며, smoothing distributions와 같은 유용한 양을 효율적으로 계산합니다. 논문에서는 이러한 개선 방법이 기존 forward-backward과 backward-forward 방법과 비교했을 때 어떻게 더 나은 성능을 발휘하는지를 시뮬레이션 결과와 함께 설명합니다.



### Are foundation models useful feature extractors for electroencephalography analysis? (https://arxiv.org/abs/2502.21086)
- **What's New**: 의료 데이터가 제한된 환경에서도 효과적인 성능을 보이는 foundation 모델을 EEG 분석에 적용하는 연구가 진행되었습니다. 이 연구는 OTiS 모델을 통해 여러 EEG 관련 작업의 진단 정확도를 측정하였으며, 그 결과 OTiS가 전문화된 EEG 모델보다 뛰어난 성능을 보였음을 밝혔습니다. 이러한 모델링 접근법은 대규모 도메인 특화 데이터셋의 필요성을 최소화하여, 임상 분야에서 중요한 도구로 자리잡을 수 있을 것으로 보입니다.

- **Technical Details**: OTiS 모델은 12개의 레이어, 3개의 헤드, 192의 폭, 그리고 8M의 파라미터를 가진 기본 변형에 대해 분석되었습니다. 이 모델은 다양한 도메인에서 수집된 640,187개의 타임 시리즈 샘플로 사전 훈련되었습니다. 본 연구는 OTiS가 EEG 분석을 위해 얼마나 효과적인지를 평가하기 위해 세 가지 공공 데이터셋에 대해 전문화된 EEG 모델과의 비김을 통해 검토하였습니다.

- **Performance Highlights**: 실험 결과 OTiS가 EEG 기준에서 최첨단 성능을 달성하며, 특정 작업에 맞춤화된 모델보다 더 높은 품질의 EEG 특징을 추출함을 입증했습니다. OTiS는 또한 주파수 대역 across frequency bands에 따른 EEG 특징을 포착할 수 있어, 중요 인구 통계학적 정보나 질병 관련 바이오마커를 지역화하는 데 도움이 됩니다. 아키텍쳐 설계의 주요 선택이 진단 정확도에 큰 영향을 미치는 것으로 나타났습니다.



### Spatial Reasoning with Denoising Models (https://arxiv.org/abs/2502.21075)
Comments:
          Project website: this https URL

- **What's New**: 본 연구에서는 Denoising Generative Models를 통해 연속 변수 집합에 대한 추론을 수행할 수 있는 Spatial Reasoning Models (SRMs) 프레임워크를 소개합니다. SRMs는 관측된 변수에 대한 관찰을 통해 관찰되지 않은 변수의 연속 표현을 추론합니다. 현재의 생성 모델들은 복잡한 분포에서 환각(hallucination) 문제로 인해 한계를 보이는데, 이를 측정하기 위한 벤치마크 작업을 제안합니다.

- **Technical Details**: 조건부 생성 모델은 복잡하고 다모드 분포를 모델링 할 수 있는 가능성을 제공하는데, 이는 고차원, 연속 데이터에서 학습된 의미적 구조를 활용하는 것을 포함합니다. 본 연구는 SRMs라는 새로운 프레임워크를 도입하여 상관관계가 있는 변수 간의 이유를 추론할 때 sequentialization의 중요성을 강조합니다. 또한, 이는 Denoising Network에 의해서 생성 순서가 예측 가능함을 보여주며, 특정 추론 작업의 정확도를 향상시킵니다.

- **Performance Highlights**: SRMs는 복잡한 추론을 위한 체계적 접근 방식을 제공하며, 생성 모델이 간단한 시각적 추론에는 능하지만, 복잡한 분포에 대해서는 환각 현상이 발생하는 것을 보여줍니다. 학습 샘플링 전략의 선택이 중요하며, 추론 작업에서 순차적 접근방식 사용이 환각을 줄이고 추론을 개선하는 데 기여합니다. 이 연구의 결과는 생성 모델들의 성능을 향상시키는 데 중요한 영향을 미칠 것으로 기대됩니다.



### FC-Attack: Jailbreaking Large Vision-Language Models via Auto-Generated Flowcharts (https://arxiv.org/abs/2502.21059)
Comments:
          13 pages, 6 figures

- **What's New**: 이번 연구에서는 뷰와 텍스트를 통합한 대형 비전-언어 모델(LVLMs)이 멀티모달 jailbreak 공격에 취약함을 발견했습니다. 이러한 공격은 모델이 해로운 콘텐츠를 생성하도록 유도하며, 이는 안전 위험을 초래할 수 있습니다. 우리는 자동 생성된 플로우차트를 기반으로 한 FC-Attack이라는 새롭고 효과적인 jailbreak 공격 방법을 제안하고, 이를 통해 LVLMs의 취약점을 더욱 조사합니다.

- **Technical Details**: FC-Attack은 사전 학습된 LLM을 미세 조정하여 단계 설명 생성기를 만들고, 이를 사용하여 해로운 쿼리에 대한 단계 설명을 생성합니다. 생산된 단계 설명은 세 가지 형태(수직, 수평, S자형)의 플로우차트로 변환되어 LVLMs에 주입됩니다. 이 공격의 실험은 Advbench 데이터셋을 사용하여 수행되었으며, 다양한 모델에서 90% 이상의 공격 성공률(ASR)을 달성했습니다.

- **Performance Highlights**: FC-Attack은 Gemini-1.5, Llava-Next, Qwen2-VL, InternVL-2.5 모델에서 기존 방법을 초월하는 수치의 ASR을 기록했습니다. 플로우차트 내 글꼴 스타일의 변화와 같은 다양한 요소가 공격 성과에 미치는 영향을 조사한 결과, Claude-3.5 모델에서 ASR이 4%에서 28%로 향상되는 성과를 거두었습니다. 방어 전략 또한 고려하여 AdaShield-A가 공격 성과를 크게 줄일 수 있었지만 유용성의 감소라는 대가를 치러야 했습니다.



### Quantum-aware Transformer model for state classification (https://arxiv.org/abs/2502.21055)
Comments:
          13 pages, 1 figure

- **What's New**: 이 논문은 양자 정보 처리에서 중요한 역할을 하는 얽힘(entanglement) 상태의 분류 문제에 데이터 기반 접근 방식을 제안합니다. 특히, bipartite(2부분 시스템) 양자 상태에 대한 transformer 기반 신경망을 활용하고 있습니다. 이 연구는 다양한 이론을 활용하여 얽힘 특성을 효과적으로 식별할 수 있는 방법을 보여줍니다.

- **Technical Details**:  연구진은 unsupervised 방식으로 양자 상태의 벡터화된 Hermitian 행렬 표현에서 요소를 마스킹하여 transformer를 사전 학습합니다. 이를 통해 모델은 기초적인 양자 밀도 행렬의 구조적 특성을 학습하고, 다양한 상태 클래스에서 얽힘 특성을 일반화할 수 있게 됩니다. 이 과정은 양자 상태의 특성을 신뢰성이 높은 방식으로 탐지하고 분류하는 방법론을 제공합니다.

- **Performance Highlights**: 본 논문에서 제안한 방법은 거의 완벽한 분류 정확도를 달성하며, 분리 가능한 상태와 얽힌 상태를 효과적으로 구분합니다. 기존 기계 학습 방법들과 비교했을 때, transformer가 양자 상태 분석에 성공적으로 적용되어 bipartite 시스템에서의 얽힘을 체계적으로 식별할 수 있음을 입증합니다. 이 결과는 최신 기계 학습 기술이 얽힘 탐지 및 분류 자동화에 기여할 수 있는 잠재력을 강조합니다.



### A data augmentation strategy for deep neural networks with application to epidemic modelling (https://arxiv.org/abs/2502.21033)
- **What's New**: 이 연구에서는 전통적인 전염병 모델과 머신러닝 기법을 결합하여 데이터-driven 예측 모델을 개발했습니다. 기존의 SIR 모델의 한계를 보완하기 위해 사회적 특성을 포함한 모델을 사용하고, 포화 감염율을 설정하여 전염병 예측 정확성을 향상시키고 있습니다. 이러한 접근법은 비선형 역학을 처리하는 데 강력한 접근법을 제공하며, 데이터 기반의 확장 가능하고 신뢰할 수 있는 외기 예측 솔루션을 제시합니다.

- **Technical Details**: 우리는 기존의 SIR 모델에 비해 포화 감염율을 적용한 사회-SIR 모델을 도입하여 전염병 예측 모델링의 유연성을 높였습니다. 데이터 보강을 통해 Feed-Forward Neural Networks (FNNs)와 Nonlinear Autoregressive Networks (NARs)를 병합하여, 이를 통해 실세계의 전염병 데이터와 합성 시뮬레이션 데이터를 효과적으로 통합합니다. 이 모델은 전염병 전파의 비선형성을 이해하는 데 기여하며, 공중 보건 개입의 영향을 반영할 수 있도록 설계되었습니다.

- **Performance Highlights**: COVID-19 팬데믹의 포스트 락다운 단계에 대해 이탈리아와 스페인 데이터를 활용하여 본 방법론의 신뢰성을 검증했습니다. 제안된 방법은 실제와 합성 데이터를 결합하여 네트워크가 전염병 역학을 효과적으로 학습하고 예측 정확도를 개선할 수 있도록 설정했습니다. 이를 통해 대체로 FNNs와 NARs를 활용하여 단기와 장기 전염병 경향을 포착하며, 데이터 기반 학습을 통해 모델의 적응성을 높였습니다.



### Sixth-Sense: Self-Supervised Learning of Spatial Awareness of Humans from a Planar Lidar (https://arxiv.org/abs/2502.21029)
- **What's New**: 이 논문은 서비스 로봇이 1D LiDAR 데이터를 이용해 사람을 감지하고 2D 자세(pose)를 추정하는 자가 감독(self-supervised) 접근 방식을 제안합니다. 기존의 RGB-D 카메라를 이용한 고가의 시스템 대신, 저비용의 센서를 통해도 환경에 사람을 인식할 수 있는 가능성을 보여줍니다. 로봇은 2개의 환경에서 자율적으로 수집한 70분의 데이터를 통해 훈련되었으며, 새로운 환경에서도 71%의 정밀도(precision)와 80%의 재현율(recall)을 달성했습니다.

- **Technical Details**: 로봇은 중심에 위치한 평면 LiDAR 센서를 이용해 2D 사람의 자세를 추정하기 위해 1D FCN 모델을 교육합니다. 모델은 LiDAR 스캔을 기반으로 사람의 존재, 거리, 방향을 예측합니다. 특히, 자가 감독 신호는 RGB-D 카메라에서 제공되는 3D 자세 정보를 사용하며, 이를 통해 LiDAR 스캔의 각 레이(ray)와 교차하는 사람의 2D 자세로 변환합니다.

- **Performance Highlights**: 최종 실험에서는 TIAGo 로봇이 Azure Kinect 카메라와 2개의 1D LiDAR 센서를 장착하고 수행되었습니다. 모델은 로봇 주변에서 인간의 존재를 효과적으로 감지하는 능력을 보여주었으며, 정량적 분석을 통해 기존 기술에 비해 개선된 성능을 입증했습니다. 이는 혼잡한 환경에서도 유용하며, 자율주행 및 다양한 서비스 로봇 응용 프로그램에 적용할 수 있는 잠재력을 지니고 있습니다.



### AutoQML: A Framework for Automated Quantum Machine Learning (https://arxiv.org/abs/2502.21025)
Comments:
          9 pages, 4 figures

- **What's New**: AutoQML은 자동화된 기계 학습(AutoML) 접근 방식을 양자 기계 학습(Quantum Machine Learning, QML)에 적응시킨 새로운 프레임워크입니다. 이 프레임워크는 모듈화된 프로그래밍 인터페이스를 제공하여 QML 파이프라인의 개발을 용이하게 합니다. AutoQML은 QML 라이브러리인 sQUlearn을 활용해 다양한 QML 알고리즘을 지원합니다.

- **Technical Details**: AutoQML은 Ray Tune 및 Optuna를 기반으로 하여 QML 파이프라인의 최적화, 병렬 처리 및 스케줄링을 다룹니다. 이 프레임워크는 데이터 청소, 전처리 및 예측 단계의 복잡성을 관리하며, 이를 통해 최적의 파이프라인 구조를 탐색할 수 있게 돕습니다. 각 단계는 다양한 데이터 클리닝 및 전처리 방법과 모델 테스트를 포함하여 유연성을 제공합니다.

- **Performance Highlights**: AutoQML은 4개의 산업 사례에 대해 평가되었으며, 기존의 고전적인 기계 학습 모델 및 수동으로 구성된 양자 솔루션과 경쟁할 수 있는 높은 성능의 QML 파이프라인을 생성할 수 있음을 입증했습니다. 이러한 결과는 AutoQML의 효율성과 접근성을 강조하며, 양자 기계 학습의 미래 가능성을 열어줍니다.



### Position: Solve Layerwise Linear Models First to Understand Neural Dynamical Phenomena (Neural Collapse, Emergence, Lazy/Rich Regime, and Grokking) (https://arxiv.org/abs/2502.21009)
- **What's New**: 이 논문은 복잡한 시스템의 핵심 원리를 남긴 채 최소한의 해결 가능한 모델로 단순화하는 과정을 탐구합니다. 특히, 층별 선형 모델(layerwise linear models)이 신경망의 역학을 단순화하여 설명할 수 있다는 새로운 관점을 제시합니다. 특히, 이 모델들은 'dynamical feedback principle'을 통해 심층 신경망의 다양한 동적 현상을 분석할 수 있는 기초를 제공합니다.

- **Technical Details**: 이 논문은 층별 선형 모델의 동역학을 이해하기 위한 'dynamical feedback principle'을 제안합니다. 이 원리는 각 층의 변화율이 다른 층에 어떻게 영향을 미치는지를 설명합니다. 이 모델들은 다차원 독립 제로 평균 가우시안 확률 변수를 입력으로 사용하며, 평균 제곱 오차(mean square error, MSE) 손실 함수를 고려합니다.

- **Performance Highlights**: 층별 선형 모델을 활용한 연구는 비선형 활성화 함수의 한계를 넘어 심층 신경망(dense neural networks)에서 관찰되는 다양한 동적 현상들을 이해하는 데 기여할 수 있습니다. 이러한 접근법은 데이터를 해석하고 비선형 활성화의 복잡한 역할을 간소화하는 데 도움을 줄 수 있습니다. 나아가, 이러한 모델들의 해결 가능성 덕분에 이론적 접근을 통해 심층 학습(deep learning) 과학의 발전을 가속화할 수 있습니다.



### TeleRAG: Efficient Retrieval-Augmented Generation Inference with Lookahead Retrieva (https://arxiv.org/abs/2502.20969)
- **What's New**: 이번 논문에서 제안하는 TeleRAG는 Retrieval-augmented generation (RAG) 시스템의 지연(latency)을 줄이고 GPU 메모리 요구 사항을 최소화하는 효율적인 추론(inference) 시스템입니다. TeleRAG의 핵심 혁신은 lookahead retrieval로, 이는 필요한 데이터를 예측하고 LLM 생성과 병행하여 CPU에서 GPU로 전송하는 사전 가져오기(pre-fetching) 메커니즘입니다. 이 시스템은 RAG 파이프라인의 모듈성을 활용하여 데이터 이동과 계산을 최적화합니다.

- **Technical Details**: TeleRAG는 inverted file index (IVF) 검색 알고리즘을 기반으로 하여 전반적인 시스템 효율을 높입니다. 이를 통해 각 쿼리에 대해 GPU 메모리를 보다 효율적으로 사용하고 CPU에서 GPU로 데이터 전송 오버헤드를 숨길 수 있습니다. TeleRAG의 lookahead retrieval은 초기 쿼리에 대한 IVF 클러스터를 예측하고 선제적으로 GPU에 로드하여 성능을 향상시킵니다.

- **Performance Highlights**: 실험 결과, TeleRAG는 최신 시스템에 비해 평균적으로 최대 1.72배 지연을 줄일 수 있음을 확인했습니다. 61GB의 Wikipedia 기반 데이터 스토어를 사용하여 Llama 모델과 함께 진행된 실험에서 RTX 4090 GPU에서 최대 2.68배의 속도를 나타냈습니다. 이러한 성과는 자원 제약이 있는 배포 환경에서도 고급 RAG 애플리케이션을 효율적으로 실행할 수 있는 능력을 보여줍니다.



### Post-Hoc Uncertainty Quantification in Pre-Trained Neural Networks via Activation-Level Gaussian Processes (https://arxiv.org/abs/2502.20966)
Comments:
          10 pages, 8 figures, 7th Symposium on Advances in Approximate Bayesian Inference

- **What's New**: 본 연구에서는 가중치 공간에서의 불확실성(uncertainty) 대신 활성화 수준에서의 불확실성에 초점을 맞추어 Gaussian Process Activation function (GAPA)을 제안합니다. 이 방법은 사전 훈련된 신경망의 평균 예측 결과를 보존하고, 기존 방법에서 자주 발생하는 underfitting 문제를 피할 수 있게 해줍니다. GAPA는 두 가지 방법론, 즉 GAPA-Free와 GAPA-Variational을 제공하여 다양한 하이퍼파라미터 학습 방식을 제안합니다. 이러한 접근법은 대규모 데이터셋에 대한 계산 부담을 줄이면서도 효과적인 결과를 제공합니다.

- **Technical Details**: GAPA 방법은 사전 훈련된 신경망에 Gaussian Process를 적용하여 각 뉴런의 불확실성을 정량화하는 데 중점을 둡니다. 특히, 첫 번째 은닉층의 각 뉴런에 대해 독립적인 일차원 Gaussian Process를 부착하여 귀납적으로 불확실성을 평가합니다. 이 방법은 네트워크의 모든 층에서 불확실성을 모델링 할 수 있는 가능성을 제공하며, 재훈련이나 미세조정 없이 적용될 수 있습니다. GAPA-Free는 경험적 커널 학습(empirical kernel learning)을 통한 하이퍼파라미터 계산을 사용하며, GAPA-Variational은 커널에 대해 경량화된 경량승산(variational inducing points)을 활용해 더 큰 유연성을 제공합니다.

- **Performance Highlights**: GAPA-Variational은 여러 데이터셋에서 Laplace 근사법을 초과하는 성능을 보였습니다. 본 연구는 활성화 수준에서 불확실성을 모델링하여 심층 신경망이 원래의 예측을 변경하지 않으면서도 불확실성을 표현할 수 있도록 합니다. 불확실성 전파와 출력을 연결하는 새로운 접근 방식은 특히 고도화된 예측 모델에서 높은 신뢰성을 제공하여 다양한 응용 프로그램에서의 활용 가능성을 열어줍니다.



### Efficient Jailbreaking of Large Models by Freeze Training: Lower Layers Exhibit Greater Sensitivity to Harmful Conten (https://arxiv.org/abs/2502.20952)
- **What's New**: 이 연구는 다양한 분야에서 대규모 언어 모델(LLM)의 보안 문제에 대한 관심이 증가하고 있음을 보여주고 있습니다. 연구진은 LLM의 파라미터 샘플링 및 정규화를 통해 시각적 표현과 히트맵을 생성하여 특정 레이어 간의 파라미터 분포의 불일치를 발견했습니다. 따라서 연구에서는 Freeze training 전략을 적용하여 하위 레이어에서만 Supervised Fine-Tuning(SFT)을 수행하는 방식으로 유해한 콘텐츠 생성에 대한 민감성을 분석했습니다.

- **Technical Details**: 연구는 Qwen2.5-7B-Instruct 모델을 대상으로 약 1000만 개의 파라미터를 샘플링하고 이들의 통계적 지표를 살펴보았습니다. 각 레이어에 대해 최대값, 최소값, 평균, 표준편차 및 분산을 계산하여 파라미터의 분포 변동성을 시각화했습니다. Comprehensive Sensitivity Score(S_score) 메커니즘을 도입하여 유해 콘텐츠 생성에 민감한 레이어를 정량적으로 평가하고, 특정 레이어의 유해 모델과 원본 모델 간의 차이를 수치적으로 분석합니다.

- **Performance Highlights**: 이 접근법은 훈련 시간과 GPU 메모리 소비를 크게 줄이면서도 높은 jailbreak 성공률과 유해 점수를 유지하는 성과를 얻었습니다. 또한, LoRA 방법을 적용한 것보다 개선된 결과를 보여주며, 이 방법은 다른 오픈소스 대규모 모델에도 성공적으로 확장되었습니다. 이 연구는 대규모 모델의 해석 가능성을 높이며, 지속적인 연구와 적응형 보안 조치의 필요성을 강조합니다.



### Large Language Models Are Innate Crystal Structure Generators (https://arxiv.org/abs/2502.20933)
Comments:
          Preprint, 18 pages

- **What's New**: 이 논문에서는 재료 발견(materials discovery)에서 중요한 결정 구조 생성(crystal structure generation)의 새로운 접근 방식을 제안합니다. 기존의 방식들이 광범위한 fine-tuning을 필요로 하는 반면, 저자들은 사전 훈련된 LLM(large language models)이 추가 학습 없이 안정적인 결정 구조를 생성할 수 있음을 보여줍니다. 이러한 새로운 프레임워크인 MatLLMSearch는 진화적 검색 알고리즘(evolutionary search algorithms)을 통합하여 높은 성능을 보입니다.

- **Technical Details**: 논문에서 제안하는 MatLLMSearch는 사전 훈련된 LLM과 진화적 검색 알고리즘을 결합하여 작동합니다. 이 프레임워크는 78.38%의 메타 안정성(metastable rate)과 31.7% DFT(밀도 범함수 이론) 검증 안정성을 입증하였으며, 이는 CrystalTextLLM과 같은 전문 모델보다 우수한 성능을 보여줍니다. 결정 구조 예측 및 다목적 최적화(multi-objective optimization)와 같은 다양한 재료 설계 작업에도 쉽게 응용될 수 있습니다.

- **Performance Highlights**: MatLLMSearch는 추가적인 fine-tuning 없이도 결정 구조 생성에서 뛰어난 결과를 보여줍니다. 이 결과는 사전 훈련된 LLM이 재료 발견에 있어 다재다능하고 효과적인 도구임을 입증하며, 결정 구조 생성을 위한 새로운 경로를 열었습니다. 이러한 접근 방식은 계산 부담을 줄이고 더 넓은 접근성을 제공함으로써 재료 과학 분야의 탐색을 촉진할 것으로 기대됩니다.



### Amortized Conditional Independence Testing (https://arxiv.org/abs/2502.20925)
Comments:
          Accepted at PAKDD 2025

- **What's New**: 이 연구에서는 새로운 접근법인 ACID(Amortized Conditional Independence Testing)를 제안합니다. 기존 방법들은 조건부 독립성을 검증하기 위한 명시적인 테스트 통계량을 설계하는 데 초점을 맞췄으나, 본 연구에서는 신경망 아키텍처를 통해 조건부 독립성 테스트를 학습합니다. 이 방법은 데이터 기반으로 Prior knowledge를 활용하지 못하는 기존의 한계를 극복합니다.

- **Technical Details**: ACID는 Transformer 아키텍처를 기반으로 한 새로운 신경망 구조로, 조건부 독립성 테스트를 학습하는 데 효과적입니다. 이 구조는 훈련 과정에서 시뮬레이션된 데이터를 사용하며, 다양한 변수 수와 데이터 오더에 대해서도 유연성을 가지고 반응할 수 있도록 설계되었습니다. 결과적으로, ACID는 정확한 테스트 결과를 제공하면서도 입력 데이터에 대한 일반화 능력을 발휘합니다.

- **Performance Highlights**: ACID는 기존의 여러 기준선과 비교했을 때, 우수한 성능을 자랑하는 것으로 나타났습니다. 실제 데이터 및 시뮬레이션된 데이터 모두에서 향상된 정확도를 기록하였고, 샘플 크기나 차원, 비선형성 등 다양한 데이터 특성에 대해 효율적인 일반화를 보여주었습니다. 특히, 낮은 추론 시간을 통해 비용을 효과적으로 절감하는 성능을 발휘했습니다.



### Hamiltonian Neural Networks approach to fuzzball geodesics (https://arxiv.org/abs/2502.20881)
Comments:
          25 pages + Appendices, 39 figures

- **What's New**: 이 논문은 Hamiltonian Neural Networks (HNNs)을 이용하여 무질량 입자가 D1-D5 circular fuzzball 기하학 안에서 운동하는 것을 설명하는 Hamilton 방정식을 해결하는 방법을 제시합니다. 이는 이론적 고에너지 물리학 분야에서는 처음 적용되는 기술로, 기존의 수치적 통합기와 비교할 때 더 높은 정확도를 제공합니다. HNNs는 특히 불안정한 경로를 다룰 때 효과적이며, 물리학적 법칙을 학습하면서 모델에 종속되지 않고 문제를 해결할 수 있는 특징을 가지고 있습니다.

- **Technical Details**: 물리정보 신경망(Physics-informed Neural Networks, PINNs)은 시스템의 물리 법칙을 지배하는 편미분 방정식을 푸는 과정에서 사용자가 사전 지식이나 제약 조건을 통합할 수 있는 장점이 있습니다. HNN은 주어진 Hamilton 방정식 집합을 해결하기 위해 설계된 신경망으로, 기존의 수치적 방법들보다 더 높은 안정성과 신뢰성을 보장합니다. 본 연구는 D1-D5 fuzzball 기하학 내에서 운동하는 무질량 입자의 비극성 및 불안정한 경로를 해결하는 HNN의 적용성을 중점적으로 다루고 있습니다.

- **Performance Highlights**: HNN의 성능은 여러 실험을 통해 검증되었으며, 비극성 경로에 대해서는 기존의 수치적 통합기와 비슷한 성능을 보였고, 불안정한 경로의 경우에는 여기에 비해 성능이 우수한 것을 확인했습니다. 이 결과는 HNN이 복잡한 동역학을 가진 시스템에서도 신뢰할 수 있는 해결책을 제공할 수 있음을 시사하며, 향후 문자열 이론의 기하학적 모델링에서도 중요하게 활용될 것입니다. 본 연구 결과는 HNN을 보다 복잡한 기하학 연구에 적용할 가능성을 보여줍니다.



### MAMUT: A Novel Framework for Modifying Mathematical Formulas for the Generation of Specialized Datasets for Language Model Training (https://arxiv.org/abs/2502.20855)
- **What's New**: 이 논문에서는 Math Mutator (MAMUT) 프레임워크를 소개하여, 수학 식을 LaTeX 표기법으로 변환한 후 동등한 표현과 허위 표현을 생성할 수 있도록 설계하였다. 머신러닝 모델의 수학 콘텐츠 인코딩 향상을 위한 특화된 데이터셋의 개발에 중점을 두고 있다. 이를 통해 다양한 수학 표기법을 포괄하는 4개의 대규모 데이터셋을 생성하였다.

- **Technical Details**: MAMUT는 동등한 수학적 식 (EquVG)과 비슷하지만 서로 다른 수학적 식 (FalseVG)을 생성하는 기능을 제공한다. 이 프레임워크는 변수 및 함수 식별자를 무작위로 변경하고 LaTeX 표기법의 변화를 포함하여 교환 가능성과 대칭성 같은 수학적 특성을 활용한다. 또한, MAMUT는 텍스트 내의 수학 LaTeX 표기에도 적용되어, 식별자와 표기 스타일의 일관된 변경을 보장한다.

- **Performance Highlights**: MAMUT를 통해 생성된 데이터셋은 계산식 완성 과제와 같은 수학 언어 모델의 추가 훈련을 위한 자료로 활용될 수 있다. 기존 데이터셋의 한계를 극복하고, 다양한 수학적 표기법을 통해 모델의 학습 능력을 향상시키는 데 기여할 것으로 기대된다. 이를 통해 수학 정보 검색과 같은 응용 프로그램에서의 성능 개선이 가능할 것으로 보인다.



### Hierarchical and Modular Network on Non-prehensile Manipulation in General Environments (https://arxiv.org/abs/2502.20843)
Comments:
this http URL

- **What's New**: 이 논문은 가정과 같이 일반적인 환경에서 로봇이 비파악형 조작(Non-prehensile manipulation) 작업을 수행할 수 있는 능력을 향상시키기 위한 새로운 아키텍처를 제안합니다. 특히, 객체의 기하학적 다양성을 고려해 환경 제약 조건에 적응할 수 있는 모듈형 리컨피겨러블 아키텍처(Modular and reconfigurable architecture)를 도입합니다. 또한, 다양한 환경을 생성하기 위한 프로시저 알고리즘을 제안하여, 실제 객체와 환경에 대해 제로샷 전이(Zero-shot transfer)가 가능하도록 합니다.

- **Technical Details**: 제안된 정책 아키텍처는 HAMnet(Hierarchical and Modular Network)으로 불리며, 로봇이 현재 환경에 따라 모듈을 조절하여 다양한 조작 전략을 학습할 수 있게 합니다. 이 아키텍처는 모듈형 네트워크(modular network)로 구성되어 있어, 로봇의 기능을 빠르게 변화하는 환경에 적응시킬 수 있습니다. 또한, 새로운 UniCORN이란 기하학적 표현을 통한 컨택트의 가능성을 모델링하여, 객체와 환경 간의 상호 작용을 고려합니다.

- **Performance Highlights**: 이 연구에서는 비파악형 조작 정책이 다양한 환경 및 객체에 대해 일반화될 수 있음을 보여줍니다. 훈련은 전적으로 시뮬레이션 내에서 이루어지며, 실제 환경과 객체에 대해 제로샷 전이를 성공적으로 수행합니다. 또한, 9개의 디지털 트윈 환경과 353개의 객체로 구성된 벤치마크를 제공하여 비파악형 조작 연구를 지원합니다.



### Weakly Supervised Multiple Instance Learning for Whale Call Detection and Localization in Long-Duration Passive Acoustic Monitoring (https://arxiv.org/abs/2502.20838)
- **What's New**: 이번 연구에서는 DSMIL-LocNet이라는 새로운 다중 인스턴스 학습 프레임워크를 소개합니다. 이 모델은 고래 소리의 탐지와 위치 추정을 위해 단지 가방 수준의 라벨만을 사용하며, 2-30분의 긴 오디오 구간을 처리할 수 있도록 설계되었습니다. 기존의 분석 방법들은 전통적인 방법으로는 처리하기 어려운 대량의 PAM 데이터를 효율적으로 분석할 필요성이 커지고 있습니다.

- **Technical Details**: DSMIL-LocNet은 dual-stream 아키텍처를 채택하여 CNN 기반의 스펙트로그램 인코딩과 MLP 기반의 시간 특징 추출을 결합합니다. 이 모델은 classification 정확도, 시간 일관성, attention sparsity, 인스턴스 일관성을 균형 있게 조절하는 4가지 손실 함수를 활용합니다. 이를 통해 모델은 긴 오디오 구간에서도 고래 소리의 시간적 위치를 정확히 추정할 수 있게 됩니다.

- **Performance Highlights**: 테스트 결과, 남극 고래 데이터에서 분류 성능이 F1 점수 0.8-0.9에 도달하며, 중간 수준의 인스턴스가 локализация 정밀도를 0.65-0.70으로 보장하는 것으로 나타났습니다. 이러한 결과는 MIL이 해양 모니터링의 확장 가능성을 향상시킬 수 있음을 시사합니다. 또한, 고래 소리를 탐지하고 위치를 추정할 수 있는 새로운 방법론이 제시되어, PAM 데이터 분석에 큰 기여를 할 것으로 기대됩니다.



### Enhanced Derivative-Free Optimization Using Adaptive Correlation-Induced Finite Difference Estimators (https://arxiv.org/abs/2502.20819)
- **What's New**: 본 논문에서는 gradient 기반의 비미분 최적화(DFO) 문제를 해결하기 위한 새로운 접근 방식을 제안합니다. 특히, 기존의 Kiefer-Wolfowitz(KW)와 simultaneous perturbation stochastic approximation(SPSA) 방법의 한계를 극복하기 위해 배치 기반의 correlation-induced FD estimate를 소개합니다. 이를 통해 각 반복마다 동적으로 배치 크기를 조정하는 적응 샘플링 전략을 결합함으로써 gradient 추정의 효율성 및 샘플 효율성을 향상시킬 수 있는 알고리즘을 개발했습니다.

- **Technical Details**: 저자들은 비미분 최적화 문제에서 gradient 추정과 샘플 효율성을 높이기 위해 새로운 알고리즘을 고안했습니다. 본 논문에서는 배치 기반의 FD 추정 기법을 사용하여 불확실한 성능 함수 F의 추정치를 얻고, 이를 통해 gradient-based stochastic search 알고리즘을 적용합니다. 이 알고리즘은 gradient 추정의 정확도를 높이기 위해 다수의 샘플을 활용하고, 이는 Kiefer-Wolfowitz와 SPSA 방법과 동일한 수렴 속도를 달성합니다.

- **Performance Highlights**: 제안된 알고리즘은 포괄적인 수치 실험을 통해 우수한 성능을 입증했습니다. 본 논문에서 개발된 stochastic line search 기법은 실용적으로 단계 크기를 조정하는 데 도움을 주며, 기존 방법들에 비해 성능 개선을 가져왔습니다. 이러한 혁신적인 접근은 특히 복잡한 시스템의 최적화 문제에서 더욱 중요한 역할을 하게 될 것입니다.



### Towards Ultimate NMR Resolution with Deep Learning (https://arxiv.org/abs/2502.20793)
- **What's New**: 이 논문에서는 다차원 NMR(Nuclear Magnetic Resonance) 분광학에서의 신호 해상도를 개선하기 위해 Peak Probability Presentations ($P^3$)라는 새로운 통계적 스펙트럼 표현 방식을 도입하였습니다. 이 방법은 각 스펙트럼 지점에 대해 피크 최대가 발생할 확률을 할당하여 신호 위치를 명확히 해결하는 데 도움을 줍니다. 또한, MR-Ai라는 인공지능 기반의 신경망 구조를 사용하여 다차원 NMR 스펙트럼의 처리를 개선하고, 여러 스펙트럼 간의 직접적인 정보 교환을 가능하게 합니다.

- **Technical Details**: 이 연구에서는 전통적인 스펙트럼 해상도 문제를 통계적 프레임워크로 재정의하고, 신호 중심을 찾을 확률을 결정하는 접근법을 취했습니다. 이를 통해 AI와 통계 분석 간의 간극을 줄이며, 피크 또는 비피크와 같은 분류 작업으로 재형성함으로써 계산 복잡성을 없애고 직접적인 확률 예측을 가능하게 했습니다. 또한, P3는 스펙트럼의 동적 범위를 줄이고 스펙트럼 아티팩트를 효과적으로 억제하는 장점을 제공하며, MR-Ai는 비정상 샘플링 데이터를 처리할 수 있는 능력을 보유하고 있습니다.

- **Performance Highlights**: MR-Ai는 MALT1 단백질의 2D 1H-15N 상관 스펙트럼을 생성하여 P3 표현의 뛰어난 성능을 입증하였습니다. P3를 통한 해상도 개선이 이루어졌으며, 이는 3D HNCA 및 HN(CO)CA 스펙트럼을 통한 백본 할당에서 효과적으로 나타났습니다. 혼잡한 스펙트럼에서도 중요한 피크만을 정확히 식별할 수 있어, 고해상도 분석이 가능한 새로운 기반을 마련하였습니다.



### Triple Phase Transitions: Understanding the Learning Dynamics of Large Language Models from a Neuroscience Perspectiv (https://arxiv.org/abs/2502.20779)
Comments:
          46 pages

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 학습 과정에서 관찰되는 '상태 전이(phase transition)' 현상을 분석했습니다. 기존 연구들은 이러한 전이를 개별적 측면에서만 분석했으나, 본 연구는 인간의 뇌와의 유사성, LLM의 내부 상태, 그리고 하위 과제 성능 간의 상관관계를 통합적으로 분석하였습니다. 이로써 LLM의 학습 역동성을 새로운 관점에서 이해하고자 합니다.

- **Technical Details**: 연구에서는 OLMo-2, OLMo-0724, LLM-jp, Amber의 네 가지 사전 훈련된 모델을 사용하여 학습 동역학을 분석하였습니다. 각 모델은 서로 다른 데이터셋과 토크나이저를 활용하며, 파라미터 수는 6.74~7.3B 범위로 설정되어 있습니다. 연구 팀은 세 가지 분석 접근법을 통해 모델의 학습 과정에서 뇌 활동과의 정렬도를 조사하고, 여러 단계의 상관관계를 밝혀내었습니다.

- **Performance Highlights**: 주목할 만한 결과로, LLMs는 세 가지 주요 단계의 상태 전이를 겪습니다: (1) 작업 지침을 따르기 시작할 때 전체 뇌와의 정렬이 급증합니다. (2) 하위 과제 정확도가 일시적으로 정체될 때 LLM들이 뇌와 멀어지는 현상이 관찰됩니다. (3) LLM들이 하위 과제를 해결하는 능력을 갖추게 되면 다시 뇌와의 정렬이 나타납니다. 이러한 연구 결과는 LLM의 emergent capabilities가 훈련 과정 동안 어떻게 형성되고 통합되는지를 보여줍니다.



### Damper-B-PINN: Damper Characteristics-Based Bayesian Physics-Informed Neural Network for Vehicle State Estimation (https://arxiv.org/abs/2502.20772)
- **What's New**: 본 연구에서는 차량 섀시 시스템과 같은 노이즈가 있는 다중 입력 다중 출력 (MIMO) 시스템을 위한 상태 추정 기술이 도입됩니다. 기존 방법의 한계를 극복하기 위해, 댐퍼 특성을 기반으로 한 베이지안 물리 정보 신경망(Damper-B-PINN) 설계를 제안합니다. 이를 통해 신경망의 수치적 안정성을 강화하고 입력-출력 관계의 복잡성을 해결합니다.

- **Technical Details**: 본 연구에서는 댐퍼의 기계적 성질을 영감으로 하는 뉴런 전진 과정을 도입하여, 에포크 사이에서 뉴런 값의 급격한 변화가 발생하지 않도록 합니다. 또한, MIMO 시스템에 최적화된 베이지안 드롭아웃 레이어를 적용하여 노이즈에 대한 강인성을 높이고 비수렴 문제를 방지합니다. 물리적 정보는 손실 함수에 포함되어 신경망의 물리적 사전(prior) 역할을 합니다.

- **Performance Highlights**: Damper-B-PINN 구조는 10개의 데이터 세트와 14종의 차량 유형에서 테스트되어 다른 최신 기술과 비교할 때 정확성, 계산 효율성 및 수렴성에서 우수한 성능을 입증했습니다. 특히, 차량 상태 추정(예: 동적 휠 하중)에서 그 효과가 두드러집니다.



### Learning to Steer Learners in Games (https://arxiv.org/abs/2502.20770)
- **What's New**: 본 논문에서는 학습 알고리즘이 게임에서 반복적인 상호작용을 통해 학습하도록 설계하는 문제를 다룹니다. 특히, 최적화자가 비확률적 학습자(no-regret learner)를 Stackelberg 균형(Stackelberg equilibrium)으로 유도하는 것을 목표로 하는 두 명의 플레이어가 참여하는 유한 행동 게임에 중점을 둡니다. 최적화자가 학습자의 보상 구조를 알지 못할 경우, 이를 달성하는 것은 불가능하다는 것을 보여줍니다.

- **Technical Details**: 연구에서는 반복적인 두 플레이어 비행렬 게임에서의 최적화자의 역할을 탐구합니다. 최적화자는 비확률적 학습자가 사용하는 알고리즘의 정보를 정확히 파악하지 못할 경우, Stackelberg 가치를 달성할 수 없음을 입증합니다. 그러나 학습자의 보상 구조를 완전하게 알 필요는 없으며, 그 보상 행렬을 근사적으로 추정함으로써 최적화자는 게임에서 성공적인 성과를 거둘 수 있도록 설계된 알고리즘을 사용할 수 있습니다.

- **Performance Highlights**: 최적화자는 학습자의 업데이트 규칙에 대한 일부 정보가 알고 있을 경우, 학습자의 보상 구조를 효과적으로 학습하고 Stackelberg 균형으로 유도할 수 있다는 두 가지 구체적인 예시를 제시합니다. 이러한 접근 방식은 반복적인 상호작용을 통해 최적화자가 학습자를 유도하기 위한 효율적인 전략을 개발할 수 있는 가능성을 제공합니다. 이와 같은 발견은 포괄적으로 비확률적 알고리즘의 다양한 동적 특성을 이해하는 데 기여합니다.



### Minimax Optimal Kernel Two-Sample Tests with Random Features (https://arxiv.org/abs/2502.20755)
Comments:
          82 pages, 10 figures, 5 tables

- **What's New**: 본 논문에서는 확률 분포의 RKHS (Reproducing Kernel Hilbert Space) 임베딩을 사용한 두 샘플 테스트에 대한 새로운 접근 방식인 스펙트럴 정규화(Spectral Regularized) 방법을 제안합니다. 이 테스트는 기존의 최대 평균 불일치(MMD) 접근법과는 달리 평균 요소와 정규화된 공분산 연산자를 모두 포함하여 보다 공간적인 분석을 가능하게 합니다.

- **Technical Details**: 제안된 방법은 랜덤 푸리에 특징(Random Fourier Feature, RFF) 근사를 기반으로 하며, 통계적 최적성과 계산 효율성 사이의 균형을 고려합니다. RFF의 근사 차수가 충분히 클 경우, 제안된 테스트가 minimax 최적성(minimax optimal)을 진정으로 보장할 수 있음을 보입니다. 또한, 데이터 적응적인 정규화 파라미터 및 커널 선택 전략을 가진 구현 가능한 순열 기반 버전(pemutation-based version)을 개발하였습니다.

- **Performance Highlights**: 수치 실험을 통해 제안된 RFF 기반 테스트는 계산 효율성이 뛰어나며, 기존의 정확한 테스트와 비슷한 성능(소폭의 전력 감소는 있지만)을 보여줍니다. 이는 실제 적용 가능성을 증대시키는 결과로, 비유클리드 공간에서의 비모수 가설 검정 문제에 대한 응용 가능성을 크게 확장합니다.



### Indoor Localization for Autonomous Robot Navigation (https://arxiv.org/abs/2502.20731)
Comments:
          10 pages, 6 figures

- **What's New**: 실내 위치 추적 시스템(IPS)에 대한 연구가 활발하게 진행되고 있으며, 최근에는 자율 로봇 내비게이션(autonomous navigation) 응용에도 IPS를 활용하려는 노력이 증가하고 있습니다. 이 논문에서는 RSSI(수신 신호 강도 표시)와 머신러닝(ML)을 사용하여 로봇의 실내 내비게이션 가능성을 탐구하였습니다. A* 경로 계획 알고리즘을 통해 로봇이 예측된 방향으로 자율적으로 내비게이션하도록 개발하였으며, 다양한 네트워크 구조를 테스트한 결과 로봇이 구석을 성공적으로 탐색할 수 있는 확률이 50%에 달했습니다.

- **Technical Details**: IPS를 활용하는 자율 로봇 내비게이션 가능성을 연구하기 위해, 먼저 특정 건물에서 데이터를 수집하고, 이를 기본 네트워크를 통해 처리하여 로봇에서 테스트할 예정입니다. 기존 연구를 기반으로 네트워크를 수정하여 문제 해결을 위한 최적의 솔루션을 찾아내고자 하며, 최종적으로 이 과정에서 수집된 데이터를 바탕으로 성능을 재현 가능하게 할 것입니다. 이 접근방식은 단순히 스마트폰 내비게이션을 위한 기존 프레임워크와는 차별화된 점이 있습니다.

- **Performance Highlights**: 이 연구에서 자율 로봇이 실내에서 성공적으로 내비게이션할 수 있는 가능성을 제시하였습니다. 특히, 현재 기술이 스마트폰 내비게이션에 초점을 두고 있는 것과 달리, 자율 로봇을 위한 내비게이션 기술에 대한 연구는 새로운 시장 기회를 창출할 것으로 기대됩니다. 이러한 접근법은 높은 정확도를 유지하면서도 임베디드 하드웨어에서의 실행 가능성을 감안하여 최적화된 솔루션을 제공하는 데 중점을 두고 있습니다.



### SPD: Sync-Point Drop for efficient tensor parallelism of Large Language Models (https://arxiv.org/abs/2502.20727)
Comments:
          Preprint

- **What's New**: 본 논문에서는 Sync-Point Drop (SPD)라는 혁신적인 최적화 기법을 소개하여, Tensor Parallelism (TP)에서 통신 오버헤드를 줄이면서 레이턴시를 최소화하는 방법을 제시합니다. 특히, 모델의 인퍼런스를 위한 분산 환경에서 효율적인 사용할 수 있는 블록 설계를 제안하였습니다. SPD는 모델 정확도에 미치는 영향을 최소화하면서도 통신 병목 현상을 효과적으로 완화합니다.

- **Technical Details**: 우리는 SPD를 통해 통신을 선택적으로 제거하여 다수의 컴퓨팅 장치에서의 분산 추론 성능을 개선합니다. 또한, 통신 감수성에 따라 다른 SPD 전략을 각각의 블록에 적용하여 최적화를 합니다. 이는 정보 손실을 최소화하고, 통신 간섭 없이도(decoded block의 경우) 실행할 수 있도록 설계되었습니다.

- **Performance Highlights**: LLaMA2-70B 모델의 분산 인퍼런스 실험에서 SPD는 약 20%의 전반적인 인퍼런스 레이턴시 감소를 기록했으며, 정확도는 1% 미만으로 저하되었습니다. 제안된 방법은 다양한 모델 크기 및 데이터셋에서 효과적인 가능성을 보여주었습니다.



### Retrieval Backward Attention without Additional Training: Enhance Embeddings of Large Language Models via Repetition (https://arxiv.org/abs/2502.20726)
- **What's New**: 이 논문은 미리 학습된(Pre-trained) 언어 모델을 제로샷(Zero-shot) 상황에서 성능을 향상시키기 위한 새로운 방법론을 제안합니다. 연구팀은 문맥 정보 인코딩을 개선하기 위해 새로운 역방향 주의 메커니즘(Backward Attention Mechanism)을 도입하였습니다. 제안된 방법은 C-MTEB(Chinese Massive Text Embedding Benchmark)에서 성능 개선을 입증하여 제로샷 학습 역량을 발전시키기 위한 귀중한 통찰을 제공하고 있습니다.

- **Technical Details**: 이 연구는 텍스트 임베딩 학습(Text Embedding Learning)에 초점을 맞추고 있으며, 자연어 처리(NLP)에서의 텍스트를 벡터 표현으로 변환하는 방법을 다룹니다. 특히, 자동 회귀 언어 모델(Autoregressive Language Model)이 어떻게 이전 토큰을 기반으로 다음 토큰을 예측하는지 설명하고, 주의 메커니즘(Attention Mechanism)의 수학적 원리를 다룹니다. 새롭게 도입된 역방향 주의 메커니즘은 기존의 하삼각(attention matrix) 구조와 대조적으로 후속 컨텍스트와의 관계를 고려하여 문맥 인코딩 품질을 향상시킵니다.

- **Performance Highlights**: 최신의 자동 회귀 언어 모델에서 텍스트 반복(Text Repetition)을 적용하여 성능을 대폭 향상시켰습니다. 이 방식은 모델이 정보를 효과적으로 캡처할 수 있도록 도와줍니다. 또, 제안된 방법은 추가적인 모델 학습 없이 기존 모델의 성능을 개선하여 연산 비용을 절약하는 동시에 구체적인 토큰의 임베딩을 선택적으로 개선합니다.



### JAM: Controllable and Responsible Text Generation via Causal Reasoning and Latent Vector Manipulation (https://arxiv.org/abs/2502.20684)
Comments:
          10 pages, 3 figures, and 6 tables

- **What's New**: 이 논문에서는 JAM (Just A Move)이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 LLM의 생성 프로세스를 제어하고 해석하기 위해 원인과 결과 분석을 통합하며, 이를 통해 보다 책임감 있게 텍스트를 생성할 수 있습니다. JAM은 기존의 CTG (Controllable Text Generation) 방법보다 최대 22% 높은 성능 향상을 보여주며, 모델의 원인과 결과성을 유지하는 동시에 출력의 질과 신뢰성을 보장합니다.

- **Technical Details**: JAM은 잠재 공간(latent space) 내에서의 원인과 결과 분석을 통해 효과적으로 LLM의 생성 과정을 제어하는 접근법입니다. 잠재 벡터(latent vector)는 LLM 아키텍처의 기본 요소로, 모델이 텍스트 생성을 수행하는 데 필요한 정보를 인코딩합니다. 이 연구에서는 LLM 생성 과정에서의 잠재 벡터의 행동과 내재된 원인 관계를 조사하고, 이러한 관계를 분석하여 더 정교한 텍스트 생성을 위한 방법론을 제안합니다.

- **Performance Highlights**: JAM은 HHH (Helpful, Honest, and Harmless) 기준 및 독성 감소 작업에서 다양한 인기 LLM 모델에서 실험을 수행하여 성능을 평가했습니다. JAM은 기존 방법에 비해 여러 지표에서 최대 10% 개선된 점수를 보여주었으며, GPT-4 평가에서도 대부분의 경우 우세한 결과를 보였습니다. 이는 JAM이 책임감 있고 현실적인 텍스트 생성을 위한 효과적이고 효율적인 방법임을 강조합니다.



### Disentangling Feature Structure: A Mathematically Provable Two-Stage Training Dynamics in Transformers (https://arxiv.org/abs/2502.20681)
- **What's New**: 이번 연구는 Transformers의 두 단계 학습 동역학(two-stage training dynamics)에 대한 이론적 분석을 제공합니다. 기존의 이론 분석은 이러한 현상을 거의 고려하지 않았으나, 이 논문에서는 GPT-2가 Counterfact 데이터셋에서 훈련되는 사례를 통해 시각적으로 이 과정을 보여줍니다. 이 논문의 주안점은 주어진 특성 구조를 분해(disentangled)하여 변별력을 분석하는 데 있습니다.

- **Technical Details**: 저자들은 컨텍스트 학습(in-context learning) 규칙 하에서 특성 학습(feature learning) 기법을 활용하여 Transformers의 동역학을 분석합니다. 이 과정에서 두 가지 유형의 특성 구조를 가지는 분리된 구조가 일반적이라는 점을 강조합니다. 예를 들어, 자연어에는 구문(syntax)과 의미(semantics)가 포함되며, 단백질(proteins)의 경우 1차 및 2차 구조가 존재합니다.

- **Performance Highlights**: 이 연구는 Transformers의 두 단계 최적화 과정에 대한 최초의 엄밀한 결과를 제시합니다. 또한, 이 과정이 attention weights의 스펙트럼 속성과 밀접하게 관련되어 있다는 보조 정리가 제시되어, 실험적으로 발견된 결과와 일치함을 보여줍니다.



### OpenEarthSensing: Large-Scale Fine-Grained Benchmark for Open-World Remote Sensing (https://arxiv.org/abs/2502.20668)
- **What's New**: 본 논문에서는 OpenEarthSensing라는 대규모의 세밀한 벤치마크를 제안합니다. 이 벤치마크는 5개의 도메인과 3개의 모달리티를 포함하여 189개의 카테고리와 총 157,674개의 이미지를 포함합니다. OpenEarthSensing은 실제 세상에서 발생할 수 있는 의미적 변화를 다룰 수 있도록 설계되었습니다.

- **Technical Details**: OpenEarthSensing는 RGB 위성 이미지, RGB 드론 공중 이미지, MSRGB, 적외선 등의 다양한 데이터 도메인을 보유하고 있어, 서로 다른 공변량 변화(covariate shifts) 조건을 갖춘 데이터 세트를 제공합니다. 이를 통해 모델의 일반화 성능을 평가할 수 있는 종합적인 테스트베드를 제공합니다.

- **Performance Highlights**: 기존의 열린 세계(open-world) 원거리 감지 기술과 방법을 기준으로 평가를 수행하여, OpenEarthSensing이 지속적으로 진화하는 환경에서의 모델 성능을 시험하는 데 있어 도전적인 벤치마크 역할을 한다는 것을 입증합니다. 이 논문에서 제안하는 벤치마크는 연구 및 개발을 촉진할 것으로 기대됩니다.



### Advancing AI-Powered Medical Image Synthesis: Insights from MedVQA-GI Challenge Using CLIP, Fine-Tuned Stable Diffusion, and Dream-Booth + LoRA (https://arxiv.org/abs/2502.20667)
- **What's New**: MEDVQA-GI 챌린지는 의료 진단에서 AI 기반 text-to-image 생성 모델의 통합을 다룹니다. 기존 방법은 정적인 이미지 분석에 주로 초점을 맞추고 있으며, 텍스트 설명에서 동적으로 의료 이미지를 생성하는 데는 부족하고, 이에 대한 새로운 접근 방식을 제안하고 있습니다. 본 연구는 Fine-tuned Generative Models을 사용하여 동적이고 확장 가능하며 정밀한 이미지를 생성하는 방법을 제시합니다.

- **Technical Details**: 이 시스템은 Fine-tuned Stable Diffusion, DreamBooth 모델 및 Low-Rank Adaptation (LORA)을 통합하여 고품질 의료 이미지를 생성합니다. 연구의 두 가지 주요 태스크는 Image Synthesis (IS)와 Optimal Prompt Generation (OPG)으로, 전자는 언어 프롬프트를 통해 이미지를 생성하고 후자는 주어진 카테고리 내에서 고품질 이미지를 생성하는 프롬프트를 제공합니다. 기존 의료 이미지 생성 방법의 한계에 대해 강조하며, Stable Diffusion이 이미지 품질 면에서 다른 모델들보다 우수함을 보여줍니다.

- **Performance Highlights**: Stable Diffusion은 Fréchet Inception Distance (FID) 스코어가 가장 낮았으며, 평균 Inception Score도 가장 높아(2.327) 뛰어난 다양성과 품질을 나타냅니다. 이는 AI 기반 의료 진단 분야의 발전을 의미하며, 향후 연구는 모델 개선, 데이터 세트 증대 및 윤리적 고려 사항에 중점을 두어야 할 것입니다.



### Dataset Distillation with Neural Characteristic Function: A Minmax Perspectiv (https://arxiv.org/abs/2502.20653)
Comments:
          Accepted by CVPR 2025, 11 pages, 7 figures

- **What's New**: 이 논문에서는 데이터셋 증류(dataset distillation)의 새로운 접근 방식을 제안하여, Neural Characteristic Function Discrepancy (NCFD)라는 새로운 지표를 도입합니다. 기존의 거리 측정 방법들이 분포의 차이를 정확하게 구별하지 못하는 문제를 해결하기 위해, NCFD는 특성 함수를 활용하여 전체 분포 정보를 포괄적으로 캡슐화하도록 설계되었습니다. 이러한 방법은 깊은 신경망의 학습 과정에서 생성 데이터와 실제 데이터 간의 차이를 최소화하는 동시에 정확도 향상을 도모합니다.

- **Technical Details**: NCFD는 특성 함수를 기반으로 하는 매개변수화된 지표로, 확률 밀도 함수의 푸리에 변환으로 정의됩니다. 이 접근 방식은 고차원 매니폴드에서의 시맨틱 구조를 포착하는데 효과적이며, 분포 매칭을 적대적인 미니맥스 최적화 문제로 재구성하여, 실시간으로 데이터 간의 불일치를 극대화합니다. 결과적으로, NCFD는 기존 최대 평균 불일치(maximum mean discrepancy, MMD) 방법보다 훨씬 더 높은 성능과 효율성을 보여줍니다.

- **Performance Highlights**: 실험 결과, NCFM 방법은 최신 기법들(SOTA)과 비교하여 ImageSquawk 데이터셋에서 20.5%의 정확도 향상을 달성했습니다. 또한 GPU 메모리 사용량을 300배 이상 줄이고, 처리 속도를 20배 향상시키는 성과를 올렸습니다. 특히, CIFAR-100 데이터셋에 대해 단일 NVIDIA 2080 Ti GPU에서 2.3GB 메모리만으로 무손실 압축을 달성하여 주목받고 있습니다.



### Consistency Evaluation of News Article Summaries Generated by Large (and Small) Language Models (https://arxiv.org/abs/2502.20647)
Comments:
          21 pages, 6 figures, 4 tables

- **What's New**: 이번 논문은 다양한 텍스트 요약 기법을 탐구하고 있습니다. 기존의 전통적인 평가 지표인 ROUGE 점수 및 BERT 점수를 사용하여 생성된 요약을 평가하며, LLM(Large Language Model) 기반의 평가 방법도 도입하고 있습니다. 특히, 생성된 요약의 일관성을 측정하기 위한 새로운 메타 평가 점수를 소개하고 있습니다. XL-Sum 데이터셋을 사용하여 여러 요약 모델의 성능을 비교한 결과, 모든 요약 모델은 참조 요약보다 일관성을 나타냈습니다.

- **Technical Details**: 논문은 요약 방법을 추출적(extractive) 및 추상적(abstractive) 접근 방식으로 나누어 다루고 있습니다. XL-Sum 데이터셋은 BBC 웹사이트에서 수집한 100만 쌍의 뉴스 기사와 요약을 포함하고 있으며, 본 작업에서는 영어 데이터만을 사용합니다. 해당 데이터셋에서 37.37%의 수동 검토된 영어 요약이 사실과 상충하는 정보가 포함되어 있음을 지적하면서, 이러한 문제의 최소화를 위해 기사 길이에 대한 필터링을 진행했습니다.

- **Performance Highlights**: 요약 방법으로는 TextRank, BART, Mistral-7B-Instruct, OpenAI GPT-3.5-Turbo 등의 모델이 포함되어 있습니다. 논문은 LLM의 발전에도 불구하고, 소규모 모델들이 여전히 비용 효율적인 솔루션으로 빛날 수 있는 여지를 강조하고 있습니다. 이에 따라 대규모 LLM들이 보유하지 못하는 다양한 성능을 가진 소형 모델들과의 비교가 이루어졌습니다.



### Can LLM Assist in the Evaluation of the Quality of Machine Learning Explanations? (https://arxiv.org/abs/2502.20635)
- **What's New**: 본 논문에서는 기계 학습(ML) 설명 방법을 평가하기 위해 LLM(대형 언어 모델)을 사용하여 인간 평가자와 LLM 기반 평가자의 성능을 비교하는 새로운 방법론을 제안합니다. LLM이 평가자인 'LLM-as-a-Judge' 접근 방식을 채택하여 ML 설명의 품질을 평가하도록 하고, 이 과정에서 다양한 설명 방법의 질을 측정합니다. 이를 통해 LLM을 통한 평가가 인간 평가자에 비해 어떤 장점과 한계를 가지는지 탐구합니다.

- **Technical Details**: 연구는 다양한 설명 방법(LIME, similarity-based explanations 등)의 품질을 평가하기 위해 인공지능 모델인 GPT-4o 및 Mistral-7.2B와 인간 평가자 간의 비교를 포함합니다. 주관적 및 객관적 메트릭을 사용하여 설명의 질을 평가하는 작업을 수행하며, 5점 리커트 척도로 주관적 척도를 개발하고 설명의 정확도를 객관적 메트릭으로 활용합니다. 이 실험은 층화된 시뮬레이션을 통해 이루어지고, 38명의 LLM 및 인간 참가자가 포함됩니다.

- **Performance Highlights**: 이 연구는 LLM이 주관적 메트릭에서 설명의 품질을 효과적으로 평가할 수 있지만, 아직 인간 평가자를 완전히 대체할 수는 없다는 결론에 도달합니다. LLM은 설명의 질을 평가하는 데 신뢰할 수 있는 대안으로 작용할 수 있는 가능성을 보여주며, 이는 ML 설명의 품질 평가에 대한 새로운 통찰을 제공합니다. 최종적으로, LLM과 인간 평가자 간의 정량적 및 정성적 차이를 분석하였고, 그 결과는 향후 설명 가능성 연구의 방향을 제시할 수 있습니다.



### Lattice Protein Folding with Variational Annealing (https://arxiv.org/abs/2502.20632)
Comments:
          Github respository will be provided soon

- **What's New**: 본 논문에서는 2차원 Hydrophobic-Polar(HP) 격자 단백질 접힘 문제에서 최저 에너지 폴드를 식별하기 위해 마스킹을 사용하는 새로운 상한(training scheme) 훈련 방식을 소개합니다. 이 접근법은 Dilated Recurrent Neural Networks(RNN)와 온도 변동에 의해 구동되는 소거 프로세스를 통합하여 최적의 폴드를 예측합니다. 60개의 비드(beads)까지의 벤치마크 시스템에 대해 정확한 예측이 가능하며, 이를 통해 기존 머신러닝 기법의 잠재력을 강조합니다.

- **Technical Details**: 단백질 접힘은 아미노산의 선형 서열이 3차원 구조를 갖는 생물학적 과정입니다. 본 연구에서는 2D 격자에서 접힘의 복잡성을 줄이기 위해 HP 모델을 사용하여 단백질 접힘을 연구합니다. Dilated RNN을 온도 소거과정과 결합함으로써 RNN의 훈련 안정성을 높이고, 샘플링 과정에서 유효한 폴드를 마스킹하여 잘못된 샘플링 부담을 없애는 방식입니다.

- **Performance Highlights**: 본 방법은 최적의 폴드를 찾아내는 데 있어 다른 기존의 머신러닝 방법보다 경쟁력 있는 결과를 보입니다. 60 비드까지의 낮은 에너지를 가진 폴드를 효과적으로 찾을 수 있으며, RNN의 자기회귀적 샘플링 속성을 유지하게 됩니다. 실제 실험을 통해 소거 및 상한 훈련의 효과를 입증하며, 3차원으로 일반화할 수 있는 가능성을 제시합니다.



### Subtask-Aware Visual Reward Learning from Segmented Demonstrations (https://arxiv.org/abs/2502.20630)
Comments:
          Project webpage: this https URL

- **What's New**: 이번 논문은 REDS(REward learning from Demonstration with Segmentations)라는 새로운 보상 학습 프레임워크를 소개합니다. 이는 최소한의 감독으로 행동 없는 비디오를 활용하여 로봇 조작 작업에서의 보상을 학습합니다. REDS는 비디오 시연을 세부 작업으로 나누어 각 세그먼트를 실제 보상으로 간주하여 훈련합니다.

- **Technical Details**: REDS는 비디오 세그먼트를 기반으로 한 밀집 보상 함수(dense reward function)를 훈련하며, 이러한 과정에서 Equivalent-Policy Invariant Comparison (EPIC) 거리를 최소화하는 방식으로 실제 보상 신호와의 정렬을 보장합니다. 또한, 대조 학습(constrastive learning) 목표를 사용하여 비디오 표현과 세부 작업을 정렬함으로써 실시간 상호작용 중에 세부 작업을 정확하게 추론합니다.

- **Performance Highlights**: REDS는 Meta-World에서 복잡한 로봇 조작 작업을 수행하는 데 있어 기본 방법들보다 현저하게 개선된 성능을 보여줍니다. 특히, 가구 조립과 같은 복잡한 장기 작업에서 최소한의 인간 개입으로 학습할 수 있는 가능성을 보여줍니다. 또한, REDS는 이전에 보지 못한 작업 및 로봇 구조에 대한 일반화 능력을 지니고 있습니다.



### Towards Zero Touch Networks: Cross-Layer Automated Security Solutions for 6G Wireless Networks (https://arxiv.org/abs/2502.20627)
Comments:
          Accepted and To Appear in IEEE Transactions on Communications (TCOM); Code is available at Github: this https URL

- **What's New**: 본 논문에서는 5G에서 6G로의 전환을 위한 자동화된 보안 프레임워크를 제안합니다. 이 프레임워크는 물리 계층 인증(Physical Layer Authentication, PLA) 및 크로스 레이어 침입 탐지 시스템(Cross-Layer Intrusion Detection Systems, CLIDS)에 초점을 맞추고 있습니다. 특히, 자동화된 알고리즘을 통해 ZTN(Zero-Touch Networks) 및 미래 네트워크를 보호하는 데 기여하고자 합니다.

- **Technical Details**: 제안된 프레임워크는 드리프트 적응 온라인 학습(Drift-Adaptive Online Learning) 기술과 새로운 Successive Halving(SH) 기반 자동 머신러닝(AutoML) 방법을 활용하여 동적 네트워킹 환경에 최적화된 기계 학습 모델을 자동 생성합니다. 이 모델들은 여러 인터넷 프로토콜 층의 보안 문제를 해결하기 위한 강력한 도구로 자리 잡고 있습니다. 실험 결과는 이 프레임워크가 공공 무선 주파수(RF) 지문 인식과 CICIDS2017 데이터 세트에서 높은 성과를 달성했음을 보여줍니다.

- **Performance Highlights**: 본 연구에서 제안된 자동화된 보안 프레임워크는 복잡한 5G/6G 환경에서 PLA 및 CLIDS 작업을 효과적으로 해결하는 데 도움을 주며, 완전 자동화되고 안전한 6G 네트워크로 나아가는 중요한 진전을 의미합니다. 논문은 또한 5G/6G 사이버 보안 영역에서의 열린 도전 과제와 연구 방향에 대해 논의하고 있어, 미래의 네트워크 자동화 및 사이버 보안 혁신을 위한 토대를 마련하고자 합니다.



### Map Space Belief Prediction for Manipulation-Enhanced Mapping (https://arxiv.org/abs/2502.20606)
Comments:
          14 pages, 10 figures, currently under review

- **What's New**: 본 연구에서는 복잡한 환경에서 물체의 가시성을 높이기 위한 조작 강화 시맨틱 맵핑(manoeuver-enhanced semantic mapping) 문제를 해결합니다. 시맨틱 맵을 기반으로 하는 Partially Observable Markov Decision Processes (POMDP)를 활용하여 신뢰성 높은 정보 획득 메커니즘을 제안합니다. 새로운 신경망 기반의 믿음 업데이트 기법인 Calibrated Neural-Accelerated Belief Updates (CNABUs)를 통해 불확실성을 보정하며 데이터 획득의 효과성을 극대화합니다.

- **Technical Details**: POMDP 모델은 시맨틱 맵의 믿음을 유지하여 다양한 객체가 있는 비구조적 환경에서 적용 가능합니다. 이론적으로, 불확실성을 고려하여 다음 최적의 관찰 지점(viewpoint)이나 조작 작업(manipulation action)을 선택하는 과정을 정의합니다. CNABU 기술은 관찰된 객체와 잠재적으로 관찰되지 않은 객체를 처리하는 통합된 믿음 전파 모델을 학습합니다.

- **Performance Highlights**: 실험 결과, 제안된 MEM 계획 기법은 기존 방법들에 비해 시뮬레이션 환경에서 더 높은 맵 컴플리트 및 정확도를 보여줍니다. 또한, UR5 로봇을 이용한 실험에서도 제안된 방법의 효능을 입증하며 실제 복잡한 선반 환경에서 제어를 성공적으로 수행했습니다. 연구 결과는 GitHub에 공개될 예정입니다.



### NutriGen: Personalized Meal Plan Generator Leveraging Large Language Models to Enhance Dietary and Nutritional Adherenc (https://arxiv.org/abs/2502.20601)
- **What's New**: 본 논문에서는 NutriGen이라 불리는 개인화된 음식 추천 시스템을 소개합니다. NutriGen은 사용자 정의 식습관과 제약 사항에 부합하는 맞춤형 식단을 생성하며, 큰 언어 모델(LLM)을 활용하여 더 나은 유연성과 사용 편의성을 제공합니다. 기존 시스템의 한계점을 해결하고 지속 가능한 식습관을 지원하기 위한 혁신적인 접근법을 제공합니다.

- **Technical Details**: NutriGen은 사용자의 음식 섭취 이력, 식습관, 선호 및 식이 제한을 기반으로 개인화된 식단을 생성하는 프레임워크를 제공합니다. 시스템은 사용자의 입력을 받아 구조화된 영양 데이터베이스를 통합하고, 이를 통해 칼로리 카운트, 대체 음식 옵션 및 제안된 레시피를 포함한 식단을 구성합니다. 식사 계획 생성 과정은 구조화된 프롬프트를 사용하여 사용자 요구 사항에 맞춘 맞춤형 출력이 이루어집니다.

- **Performance Highlights**: NutriGen의 평가 결과, Llama 3.1 8B 모델과 GPT-3.5 Turbo가 각각 1.55% 및 3.68%의 낮은 비율 오차를 기록하며, 사용자 정의 칼로리 목표에 밀접하게 일치하는 식단을 효과적으로 생성함을 보여주었습니다. 또한 DeepSeek V3와 여러 기존 모델들의 성능을 비교하여 개인화된 영양 계획에서의 잠재력을 평가하였습니다.



### LLMs Have Rhythm: Fingerprinting Large Language Models Using Inter-Token Times and Network Traffic Analysis (https://arxiv.org/abs/2502.20589)
- **What's New**: 이 논문에서는 에 대한 새로운 접근 방식을 제안합니다. 저자들은 대형 언어 모델(LLMs)의 비侵입적인 실시간 지문 기술을 개발했으며, 이는 네트워크 트래픽이 암호화된 경우에도 모델을 식별할 수 있도록 돕습니다. 이 기법은 토큰 간 시간 간격(Inter-Token Times, ITTs)을 측정하여 각 모델의 고유한 타이밍 패턴을 식별합니다.

- **Technical Details**: 제안된 방법은 딥 러닝(Deep Learning) 기반의 파이프라인을 사용하여 네트워크 트래픽 데이터를 처리하고, 36개의 특징을 추출합니다. 이 특징들은 양방향 장기 단기 기억(BiLSTM) 레이어와 다중 헤드 주의 메커니즘(multi-head attention mechanism)을 포함하는 하이브리드 DL 아키텍처로 전달되어 모델을 식별합니다. 저자들은 이 기술을 다양한 배포 시나리오에서 평가하여 효과적이고 강력함을 입증했습니다.

- **Performance Highlights**: 실험 결과는 16개의 소형 언어 모델(SLMs)과 10개의 독점 LLM에서 제안된 기술이 높은 정확도로 모델을 식별할 수 있음을 보여줍니다. 여러 네트워크 조건에서도 탁월한 성능을 유지하며, 이로 인해 모델 식별에 대한 새로운 관점을 제시하고 더 안전한 LLM 배포를 위해 기여할 수 있음을 확인할 수 있었습니다.



### Interpreting CLIP with Hierarchical Sparse Autoencoders (https://arxiv.org/abs/2502.20578)
- **What's New**: 본 논문에서는 Matryoshka SAE (MSAE)라는 새로운 아키텍처를 소개합니다. MSAE는 여러 세분화 수준에서 계층적 표현을 동시에 학습하여 재구성 품질과 희소성(sparsity)이라는 두 가지 지표를 효율적으로 최적화할 수 있도록 돕습니다. 이를 통해 CLIP 모델의 해석 가능성과 제어 가능성을 향상시킵니다.

- **Technical Details**: MSAE는 TopK 연산을 h번 적용하여 점진적으로 증가하는 활성 뉴런 수(k)를 학습합니다. 이러한 방식으로, MSAE는 coarse 개념에서 fine-grained 특징까지 다양한 granularities를 동시에 학습하며, 모든 수준에서 재구성 손실을 결합하여 보다 유연하고 적응적인 희소성 패턴을 달성합니다. 이로써 기존의 TopK 제약조건과 L1 정규화의 문제를 해결합니다.

- **Performance Highlights**: MSAE는 CLIP의 경우에서 0.99 코사인 유사도(cosine similarity)와 0.1 미만의 설명되지 않은 분산(fraction of variance unexplained)을 달성하면서 80%의 희소성을 유지합니다. MSAE의 실용성을 검증하기 위해 120개 이상의 해석 가능한 개념을 추출하고 이를 기반으로 유사도 검색 및 편향 분석을 수행했습니다.



### An Integrated Deep Learning Framework Leveraging NASNet and Vision Transformer with MixProcessing for Accurate and Precise Diagnosis of Lung Diseases (https://arxiv.org/abs/2502.20570)
- **What's New**: 이번 연구는 NASNet과 Vision Transformer(ViT)의 장점을 결합한 새로운 딥러닝 프레임워크인 NASNet-ViT를 제안합니다. 이 모델은 폐 질환을 다섯 가지 범주로 분류하는 데 초점을 맞추고 있으며, 조기 및 정확한 진단을 요구하는 여러 폐 질환에 대한 해결책을 제공합니다.

- **Technical Details**: NASNet-ViT는 MixProcessing이라는 다면적 전처리 전략을 사용하여 진단 정확도를 높입니다. MixProcessing은 웨이브렛 변환(wavelet transform), 적응형 히스토그램 평활화(adaptive histogram equalization), 형태학적 필터링(morphological filtering) 기술을 결합하여 특징을 효과적으로 추출합니다.

- **Performance Highlights**: NASNet-ViT 모델은 98.9%의 정확도와 0.99의 민감도(sensitivity), 0.989의 F1-score, 0.987의 특이도(specificity)를 달성하여 기존의 최신 아키텍처를 능가합니다. 이 모델은 25.6MB의 작은 크기와 12.4초의 짧은 계산 시간을 제공하여 실제 임상 환경에서도 효율적으로 사용할 수 있습니다.



### LISArD: Learning Image Similarity to Defend Against Gray-box Adversarial Attacks (https://arxiv.org/abs/2502.20562)
- **What's New**: 이 논문은 기존의 방어 메커니즘들이 주로 화이트 박스 공격을 기반으로 평가받고 있다는 점을 지적하며, 실제 공격자가 모델의 기울기를 접근할 수 없다는 현실을 반영한 그레이 박스(grayscale box) 공격 상황을 제안합니다. 새로운 방어 메커니즘인 Learning Image Similarity Adversarial Defense (LISArD)를 소개하며, 이는 템포럴과 컴퓨테이셔널 비용을 증가시키지 않으며 그레이 박스와 화이트 박스 공격 모두에 대한 강인함을 제공합니다.

- **Technical Details**: LISArD는 변형된 이미지와 깨끗한 이미지의 임베딩(embedding)을 기반으로 교차 상관 행렬(cross-correlation matrix)을 대각 행렬(diagonal matrix)로 근사화하면서 동시에 분류 학습(classification learning)을 실시하는 방법입니다. 이 접근 방식은 모형이 깨끗한 이미지와 변형된 이미지를 유사하게 인식하도록 유도하여 공격의 영향을 감소시키는 것을 목표로 합니다. 이 메커니즘은 추가적인 훈련 에포크(epoch)나 파라미터 없이 적용 가능합니다.

- **Performance Highlights**: 실험 결과 LISArD는 그레이 박스 공격에 대한 강인함을 입증하며, 다양한 아키텍처에서 사용될 수 있고 화이트 박스 환경에서도 탄력성을 유지한다고 보고되었습니다. 또한, 기존의 Adversarial Distillation(AD) 모델들이 AT(Adversarial Training) 제거 시에 성능이 급격히 하락하는 반면, LISArD는 훈련 비용을 증가시키지 않으면서 우수한 성과를 보였습니다. 이로 인해 LISArD는 기존 이론에서 나오는 약점을 해결하여 다양한 조건에서 효과적으로 방어할 수 있는 가능성을 제시합니다.



### NANOGPT: A Query-Driven Large Language Model Retrieval-Augmented Generation System for Nanotechnology Research (https://arxiv.org/abs/2502.20541)
Comments:
          61 pages, 3 figures

- **What's New**: 이 논문은 나노기술 연구를 위해 설계된 대형 언어 모델 기반의 정보 검색 증강 생성 시스템(LLM-RAG)의 개발 및 응용을 제시합니다. 이 시스템은 지능형 연구 보조 도구로서의 역할을 하여 나노기술 분야의 문헌 조사를 보다 효율적이고 포괄적으로 만드는 데 기여합니다. Google Scholar의 고급 검색과 Elsevier, Springer Nature 및 ACS Publications의 오픈 액세스 논문을 활용하여 신뢰할 수 있는 여러 출처에서 데이터를 통합합니다.

- **Technical Details**: LLM-RAG 시스템은 고급 쿼리 백엔드 검색 메커니즘을 중심으로 구성되어 있으며, 이는 다수의 신뢰할 수 있는 출처로부터 데이터를 결합합니다. LLM은 대규모 훈련 데이터를 통해 인간과 유사한 텍스트를 생성하고 이해할 수 있으며, 이러한 모델은 자연어 처리(NLP) 분야에서 혁신적인 기술로 부각되고 있습니다. 본 논문에서는 LLM을 기반으로 한 RAG 기법이 LLM의 성능을 개선하지고 구조적 오류를 줄이는 방법에 대해 논의합니다.

- **Performance Highlights**: LLM-RAG 시스템은 포괄적인 문헌 리뷰에 필요한 시간과 노력을 크게 줄이는 동시에 높은 정확성과 쿼리 관련성을 유지하는 데 효과적이라는 것이 엄격한 테스트를 통해 검증되었습니다. 이 시스템은 표준 공개 LLM보다 성능이 우수하며, 나노기술 분야에서 연구를 가속화하는 데 중요한 잠재력을 보여줍니다. 다양한 연구 분야에서 새로운 나노 재료의 발견과 실험 데이터의 해석을 원활하게 하는 데 기여할 것으로 기대됩니다.



### Learning Dynamics of Deep Linear Networks Beyond the Edge of Stability (https://arxiv.org/abs/2502.20531)
Comments:
          Published in ICLR 2025

- **What's New**: 이 연구는 고정된 학습율 $$를 사용하여 훈련된 깊은 신경망(deep neural networks)의 학습 역학을 정밀 분석합니다. 특히, '안정성의 경계'(edge of stability) 영역을 넘어서는 깊은 선형 네트워크(deep linear networks)의 손실(loss) 진동을 다룹니다. 본 결과는 깊은 네트워크에서 발생하는 두 가지 주요 현상을 설명하는 데 기여합니다: 얕은 모델(shallow models)과 단순한 작업(simple tasks)은 항상 EOS를 나타내지 않으며, 주요 특징(top features) 내에서 진동이 발생하는 점입니다.

- **Technical Details**: 연구에서는 손실 진동이 혼沌(chaos)으로 가는 주기 이중화(period-doubling) 경로를 따른다는 점을 이론적으로 분석했습니다. 2주기 궤도(regime of 2-period orbit) 내에서 손실 진동이 특정한 작은 부분공간(subspace) 내에서 발생한다는 사실을 밝히고, 이 부분공간의 차원이 학습율에 의해 정확하게 정의됨을 보여주었습니다. 경량의 그래디언트 흐름(gradient flow)에 대한 대칭 유도 보존법칙(symmetry-induced conservation law)이 EOS에서 어떻게 깨지고 점진적으로 0으로 감소하는지에 대한 분석이 연구의 핵심입니다.

- **Performance Highlights**: 본 연구는 실험을 통해 이론을 지지하며, 비선형 네트워크(non-linear networks)에서 이러한 현상이 어떻게 발생하는지에 대한 사례를 제시합니다. 또한, 깊은 선형 네트워크(DLNs)와 같은 benign landscape를 가진 네트워크와의 차이를 명확히 합니다. 이 연구 결과는 깊은 신경망의 동작을 이해하는 데 중요한 통찰을 제공합니다.



### Personas Evolved: Designing Ethical LLM-Based Conversational Agent Personalities (https://arxiv.org/abs/2502.20513)
- **What's New**: 이 논문은 Large Language Models(LLMs)가 Conversational User Interfaces(CUIs)을 혁신하고, 동적이고 맥락을 반영하는 인간 같은 상호작용을 가능케 한다고 강조합니다. 그러나 LLM 기반의 페르소나의 빠른 채택은 편향, 조작, 예상치 못한 사회적 결과 등의 윤리적 및 실용적 문제를 야기하고 있습니다. 이는 전통적인 CUI와는 달리, LLM 기반 페르소나는 광범위한 데이터셋에서 동적으로 반응을 생성하므로 그 행동을 예측하고 관리하기 더 어렵습니다.

- **Technical Details**: CUI에서 에이전트의 독특한 페르소나와 정체성을 설계하는 것은 오랫동안 중요한 역할을 해왔습니다. LLMs는 자연스럽고 일관된 대화를 가능하게 하며, 텍스트 기반의 상황에서 인간 행동을 시뮬레이션할 수 있는 가능성을 열어줍니다. 이러한 기술은 복잡한 사회적 행동을 시뮬레이션하고 인간 행동을 예측하기 위해 상세한 페르소나 입력을 통합하여 현실적이고 유연한 디지털 생태계를 창조하는 데 기여하고 있습니다.

- **Performance Highlights**: LLM 기반의 페르소나가 다양한 응용 프로그램과 이벤트에 통합된 실례는 많습니다. 예를 들어, LLM 기반의 가상 에이전트가 알버트 아인슈타인 페르소나를 구현하고 전문가들과 실시간으로 토론한 사례에서 볼 수 있듯, LLM은 새로운 상호작용 경험을 제공합니다. 그러나 이러한 발전과 함께, 사용자 조작, misinformation, 예상치 못한 정서적 애착과 같은 윤리적 문제도 제기되고 있습니다.



### Creator-Side Recommender System: Challenges, Designs, and Applications (https://arxiv.org/abs/2502.20497)
Comments:
          9 pages and 9 figures

- **What's New**: 이 논문에서는 콘텐츠 제작자의 경험을 향상시키기 위해 DualRec이라는 새로운 창작자 측 추천 시스템을 제안합니다. 기존의 추천 시스템이 사용자 측에 집중되어 있었던 반면, DualRec은 제작자가 적절한 사용자와 연결될 수 있도록 지원합니다. 이 시스템은 기존 사용자 측 알고리즘을 간단하게 변경하여 창작자 측 버전으로 전환할 수 있도록 설계되었습니다.

- **Technical Details**: DualRec 시스템은 사용자 요청에 따라 적합한 사용자를 찾아 콘텐츠를 추천하는 구조로, 이를 통해 제작자의 만족도를 높이는 것을 목표로 합니다. DualRec은 두 가지 주요 모듈인 검색 모듈과 순위 매기기 모듈을 포함하며, 사용자 가용성 계산(User Availability Calculation, UAC) 모듈을 도입하여 추천 성능을 개선합니다. 이 시스템은 변경이 간단해 기존 사용자 측 연구 결과를 창작자 측으로 직접 적용할 수 있습니다.

- **Performance Highlights**: DualRec은 이미 1억 명 이상의 사용자와 1천만 명 이상의 제작자가 있는 Kwai 플랫폼에 구현되어 있으며, 제작자의 경험을 크게 향상시켰습니다. 사용자 가용성 문제를 해결함으로써, 추천 시스템이 사용자를 효과적으로 목표로 설정할 수 있도록 지원합니다. 이는 제작자들이 더 많은 피드백을 받고, 콘텐츠 제작을 촉진하는 긍정적인 결과를 가져옵니다.



### Promote, Suppress, Iterate: How Language Models Answer One-to-Many Factual Queries (https://arxiv.org/abs/2502.20475)
- **What's New**: 이번 연구에서는 언어 모델(LM)이 다중 사실 질의(one-to-many factual queries)에 응답하는 기제를 분석합니다. 구체적으로, 모델은 모든 대답을 회상한 뒤, 이전에 생성된 대답을 억제하는 promote-then-suppress 메커니즘을 사용합니다. 이 연구는 모델이 어떻게 주제와 이전의 답변 토큰을 활용하여 지식을 회상하고, 반복을 피하는지를 탐구합니다.

- **Technical Details**: 모델은 attention과 다층 퍼셉트론(MLP)을 사용하여 주제 정보와 이전의 답변을 처리합니다. 중간 계층에서는 주제 정보를 복사하고, MLP는 가능한 모든 답변을 증진시키며, 후반 계층에서는 이전 답변을 억제합니다. Token Lens 기법과 knockout 방법을 통해 모델의 작동 기제를 분석하고, 특정 토큰들이 어텐션과 MLP 계층에서 어떻게 사용되는지를 연구합니다.

- **Performance Highlights**: 이 연구는 Llama-3-8B-Instruct 및 Mistral-7B-Instruct 모델이 다층 네트워크를 통해 어떻게 지식 회상 및 반복 방지를 수행하는지를 다양한 데이터 세트를 통해 증명합니다. 모델은 주제와 이전 답변 토큰들을 사용하여 명확한 답변을 생성하는 데 필요한 모든 정보를 집계합니다. 이 연구는 복잡한 사실 회상을 위한 LMs의 동적 정보 통합 방식을 밝혀내며, 향후 연구에 중요한 기초 자료를 제공합니다.



### Large Language Model Strategic Reasoning Evaluation through Behavioral Game Theory (https://arxiv.org/abs/2502.20432)
- **What's New**: 이번 논문에서는 큰 언어 모델(LLMs)의 전략적 의사결정 메커니즘을 분석하는 평가 프레임워크를 소개합니다. 기존 연구는 네시 균형(Nash Equilibrium, NE) 근사화에 초점을 맞추고 있었는데, 이는 모델의 전략적 선택을 이끌어내는 과정과는 무관한 접근이었습니다. 새로운 연구는 행동 게임 이론(behavioral game theory)을 기반으로 하여 추론 능력과 맥락적 효과를 분리하여 평가합니다.

- **Technical Details**: 22종의 최신 LLM들을 대상으로 한 실험 결과, GPT-o3-mini, GPT-o1 및 DeepSeek-R1 모델이 여러 게임에서 우수한 성능을 보였습니다. 그러나 모델의 규모가 성능을 결정짓는 유일한 요소는 아님을 알게 되었습니다. 체인 오브 생각(Chain-of-Thought, CoT) 프롬프팅의 효과는 모델마다 다르며, 특정 수준의 모델에게만 전략적 추론을 증가시키는 경향이 있음을 발견했습니다. 또한, 인코딩된 인구 통계적 특성의 영향이 모델의 의사결정 패턴에 미치는 영향을 조사했습니다.

- **Performance Highlights**: 대규모 모델들이 대부분의 작업에서 우위를 보였지만, 특정 작업에서는 소규모 모델이 이를 초과하거나 일치하는 성과를 냈습니다. 이러한 결과는 모델 크기와 구조보다는 특정 과제의 활용 가능성에 더 중점을 두어야 함을 시사합니다. 또한, 인구 통계적 특성이 포함된 결과들은 스트레티지적 추론에 편향이 존재함을 암시하는데, 이는 향후 LLM 개발에 균형 잡히고 윤리적인 접근이 필요하다는 점을 강조합니다.



### Efficient Risk-sensitive Planning via Entropic Risk Measures (https://arxiv.org/abs/2502.20423)
- **What's New**: 본 논문은 Markov Decision Processes (MDPs)에서 tail-focused metrics를 최대화하는 risk-sensitive planning의 새로운 접근법을 제시합니다. 기존 연구들은 Entropic Risk Measures (EntRM)가 동적 프로그래밍을 통해 효율적으로 최적화할 수 있음을 보여주었지만, 해석하기 어려운 매개변수 선택 문제를 남겼습니다. 저자는 EntRM의 전체 최적 정책 집합을 계산함으로써 관심 있는 메트릭에 대한 강력한 근사를 제공할 수 있음을 입증합니다.

- **Technical Details**: 제안된 방법은 새로운 구조 분석 및 entropic risks의 smoothness properties를 활용하여 최적성 경계를 효과적으로 계산할 수 있도록 합니다. 이러한 접근 방식에 따르면, 다양한 매개변수 값에 대해 EntRM을 통해 최적의 정책을 찾아낼 수 있습니다. 이러한 과정을 통해 얻어진 최적 정책은 여러 의사결정 시나리오에서의 성능을 높이는 데 기여합니다.

- **Performance Highlights**: 실험 결과는 제안된 방식이 다양한 의사결정 시나리오에서 우수한 성능을 달성함을 보여줍니다. 이는 현재 널리 사용되는 메트릭인 threshold probabilities나 (Conditional) Values at Risk와 비교했을 때, 훨씬 더 효율적이고 해석 가능한 방안을 제공한다는 것을 의미합니다. 결과적으로, 저자들은 이 접근법이 실제 세계의 복잡한 의사결정 문제에 적용될 수 있음을 강조합니다.



### Transfer Learning through Enhanced Sufficient Representation: Enriching Source Domain Knowledge with Target Data (https://arxiv.org/abs/2502.20414)
Comments:
          44 pages

- **What's New**: 이번 논문에서는 전이 학습(Transfer Learning)의 새로운 방법인 TESR(Transfer learning through Enhanced Sufficient Representation)을 소개합니다. TESR은 원본 도메인(source domains)에서 충분하고 불변하는 표현(representation)을 추정한 후, 타겟 데이터(target data)에서 도출된 독립 구성요소를 활용하여 표현을 강화하는 접근 방식을 제공합니다. 이 방법은 전통적인 전이 학습 접근법이 가진 엄격한 모델 가정과 유사성 요구사항의 제약을 극복할 수 있도록 설계되었습니다.

- **Technical Details**: TESR의 프레임워크는 원본 데이터로부터 충분하고 불변하는 표현(SIRep)을 학습하고, 이를 타겟 데이터로부터의 보강된 요소로 강화하는 방식입니다. 이 보강 요소는 SIRep에 포함되지 않은 타겟 데이터의 정보를 포착하도록 설계되어 있습니다. 이 모델은 서로 다른 도메인 간에 유사한 구조를 가정하지 않기 때문에, 다양한 감독 학습 문제에서 유연하게 적용될 수 있는 장점을 가지고 있습니다.

- **Performance Highlights**: TESR은 시뮬레이션 연구와 실제 데이터 응용을 통해 성능을 검증하였으며, 유한한 샘플 환경에서도 효과성을 입증하였습니다. 이 방법은 모델 파라미터를 전이하는 전통적인 접근법과 달리, 데이터 표현의 전이에 중점을 두어 높은 적응성과 유연성을 제공합니다. 논문에서는 다양한 테스트와 실험을 통해 TESR의 이론적 특성과 수치적 성능을 평가하고 있습니다.



### Backpropagation-free Spiking Neural Networks with the Forward-Forward Algorithm (https://arxiv.org/abs/2502.20411)
- **What's New**: 이번 연구에서는 Forward-Forward (FF) 알고리즘을 통해 SNN(Spiking Neural Networks) 훈련을 위한 새로운 접근 방식을 제안합니다. 기존의 역전파(Backpropagation)와 달리 FF 알고리즘은 두 번의 전진 전달만을 사용하여 지역적 학습(localized learning)을 촉진하며, 계산 효율성을 높이고 신경형 하드웨어(neuromorphic hardware)와의 호환성을 개선합니다. 새로운 FF 기반 SNN 훈련 프레임워크를 소개하고, 다양한 데이터셋에서의 성능을 평가하였습니다.

- **Technical Details**: SNN은 이산 스파이크 이벤트를 통해 정보를 처리하는 생물학적으로 영감을 받은 모델입니다. 하지만 SNN 훈련은 주로 신경망 출력 오류에 대한 개별 신경의 기여를 할당하는 데 어려움을 겪고 있습니다. FF 알고리즘은 두 개의 전진 전달만을 필요로 하며, 이를 통해 모든 중간 활동을 저장할 필요가 없고 대칭적인 가중치 연결이 필요하지 않아 생물학적 원리에 더 가깝습니다. 또한, FF는 데이터 파이프라인을 중단하지 않고도 즉각적인 학습이 가능 합니다.

- **Performance Highlights**: 실험 결과, FF 기반 모델은 MNIST와 Fashion-MNIST에서 기존의 FF 기반 SNN보다 5% 이상 높은 성능을 보였고, 최신 역전파 훈련 SNN과 유사한 정확성을 달성했습니다. 더욱 복잡한 작업인 CIFAR-10과 SHD에서도 최대 6%의 성능 향상을 보여주며, 역전파 기반 SNN과 경쟁력을 차지하고 있습니다. 이러한 결론은 FF 알고리즘이 SNN 훈련 방법론의 발전과 신경형 컴퓨팅의 확장을 가능하게 할 수 있음을 강조합니다.



### Brain-Inspired Exploration of Functional Networks and Key Neurons in Large Language Models (https://arxiv.org/abs/2502.20408)
Comments:
          13 pages, 5 figures

- **What's New**: 본 연구는 대규모 언어 모델(LLMs)에서의 기능적 네트워크(FBNs)를 탐구하기 위한 새로운 접근 방식을 소개합니다. 기존의 연구들은 개별 뉴런의 기여도에 중점을 두었으나, 본 연구는 뉴런 간의 상호작용을 통해 성능이 발휘되는 복잡한 네트워크를 분석합니다. 이러한 새로운 관점은 LLM의 메커니즘을 이해하는 데 기여하며, 이를 통해 모델의 투명성 및 신뢰성을 향상시킬 수 있습니다.

- **Technical Details**: 본 연구에서는 LLM의 다층 퍼셉트론(MLP) 층의 출력 데이터를 기능적 자기 공명 영상(fMRI) 신호와 유사하게 간주하여 독립 성분 분석(ICA) 기법을 적용합니다. 이러한 분석 방법을 통해 LLM 내에서 다수의 기능적 네트워크가 존재함을 확인하였으며, 이들 네트워크는 다양한 입력 자극 간의 공간적 일관성을 보여줍니다. 특히, 핵심 기능 네트워크를 차단할 경우 모델의 성능이 상당히 저하된다는 결과를 아울러 제시하였습니다.

- **Performance Highlights**: 연구 결과, LLM에서 2% 미만의 뉴런으로 구성된 핵심 기능 네트워크를 유지할 경우 모델 성능을 유지하거나 향상시킬 수 있음을 발견하였습니다. 불필요한 뉴런을 차단하면서 새로운 기능적 네트워크를 점진적으로 통합함으로써 모델의 성능이 낮은 단계에서 높은 단계로 개선될 수 있음을 보여주었습니다. 마지막으로, MLP 층의 10% 이하의 뉴런만으로도 원래 네트워크와 맞먹는 성능을 달성했습니다.



### Adversarial Robustness of Partitioned Quantum Classifiers (https://arxiv.org/abs/2502.20403)
- **What's New**: 이 논문은 양자 분류기(quantum classifiers)의 회로 분할(circuit cutting) 기법을 통해 적대적 공격(adversarial attack)에 대한 취약성을 어떻게 증가시키는지를 조사하는 최초 연구로, 기존의 문헌에서 다루지 않았던 중요한 주제를 다룹니다. 또한, 원래 회로의 출력을 재현하지 못하게 하는 적대적 게이트(adversarial gates)의 삽입이 양자 분류기의 설계에 미치는 영향을 확인합니다. 이로 인해 양자 기계 학습(quantum machine learning)의 장점과 고전 모델(classical models)과의 성능 비교에서 중요한 고찰을 제공합니다.

- **Technical Details**: NISQ(Noise Intermediate-Scale Quantum) 시대에서, 회로 분할 기법을 통해 여러 양자 처리 장치에서 양자 회로를 실행할 수 있도록 하고, 고전적 통신(classical communication)을 통해 결합할 수 있습니다. 이 연구는 회로 분할 과정을 통해 준비된 상태에 적대적 변화를 추가함으로써 양자 분류기의 설계를 더욱 취약하게 만드는 방법을 이론적 및 실험적 관점에서 조사합니다. 또한, 다양한 깊이에서 적대적 게이트를 도입함으로써 예측 신뢰도(predictive confidence)에 미치는 잠재적 변화를 이론적으로 구속하는 정리를 제시합니다.

- **Performance Highlights**: 이 연구의 실험 결과는 변형 양자 분류기(variational quantum classifiers)에서 적대적 게이트를 배치할 경우, 전통적인 입력 상태를 공격하는 것과는 다른 방식으로 예측 성능에 큰 영향을 미친다는 것을 보여줍니다. 실험적 조사를 통해서 적대적 게이트의 깊이에 따른 효과를 비교하며, 회로 조각의 결과를 결합할 때 발생하는 전반적인 신뢰도 저하를 시각적으로 확인합니다. 이 논문은 양자 기계 학습 분야에서 중요성이 증가하고 있는 적대적 견고성을 다루며, 양자 분류기의 미래 개선에 중요한 통찰을 제공합니다.



### Momentum Posterior Regularization for Multi-hop Dense Retrieva (https://arxiv.org/abs/2502.20399)
Comments:
          Accepted by COLING 2025

- **What's New**: 이 논문은 Multi-hop 질문 응답 시스템의 효과적인 지식 증류(knowledge distillation) 방법인 MoPo(Momentum Posterior Regularization)를 제시합니다. 기존의 지식 증류 방법들은 Multi-hop QA에 적합하지 않았으나, 본 연구에서는 새로운 접근 방식을 도입하여 이전 조건에서의 응답을 활용할 수 있는 모델을 개발했습니다. 특히, 이 연구는 새로운 데이터셋인 PostSumQA를 소개하여 Multi-hop 질문에 대한 포스터리어 요약(postorior summary) 정보를 제공하고 있어 향후 연구에 기여할 전망입니다.

- **Technical Details**: MoPo는 두 가지 핵심 혁신을 기반으로 합니다. 첫째, 포스터리어 정보는 이전 단계와 현재 단계의 골드 지식에서 쿼리 포커스 요약(query-focus summary)으로 정의됩니다. 둘째, MoPo는 모멘텀 이동 평균(momentum moving average) 방법을 사용하여 포스터리어 검색이 우선 검색과 함께 업데이트되도록 하여 smoother한 훈련을 가능하게 합니다. 이 두 가지 접근 방식은 지식 간 큰 갭을 줄이고 효과적인 지식 증류를 가능하게 합니다.

- **Performance Highlights**: HotpotQA와 StrategyQA에서의 실험 결과는 MoPo가 기존 방법들보다 우수한 성능을 내는 것을 보여줍니다. MoPo를 사용한 간단한 파이프라인은 기존의 Multi-hop reranking 및 LLM을 기반으로 한 방법보다 더 뛰어난 다운스트림 성능을 기록했습니다. 전체적으로 MoPo는 이전 Multi-hop 검색 방법들과 포스터리어 정규화 방법들보다 향상된 인출 성능을 보여줍니다.



### Regional climate projections using a deep-learning-based model-ranking and downscaling framework: Application to European climate zones (https://arxiv.org/abs/2502.20132)
Comments:
          This manuscript has been submitted to Environmental Science and Pollution Research (ESPR) for review

- **What's New**: 이 논문은 Global Climate Models (GCMs)의 고해상도 다운스케일링을 위해 딥러닝 기반의 다중 모델 평가 및 다운스케일링 프레임워크를 제안합니다. 32개의 Coupled Model Intercomparison Project Phase 6 (CMIP6) 모델을 Deep Learning-TOPSIS (DL-TOPSIS) 메커니즘을 통해 평가하고 순위 매기기를 수행하여, 기후 예측의 품질을 개선했습니다. 다섯 가지 타입의 Köppen-Geiger 기후 지역을 계절별로 분석하여 온도 편향을 검증했습니다.

- **Technical Details**: 논문에서는 0.1° 해상도로 GCM 출력을 다운스케일링하기 위해 Vision Transformer (ViT), Geospatial Spatiotemporal Transformer with Attention and Imbalance-Aware Network (GeoSTANet), CNN-LSTM 및 ConvLSTM의 네 가지 모델을 사용했습니다. GeoSTANet은 온도 극단값을 효과적으로 포착하여 가장 높은 정확도를 달성했습니다. DL-TOPSIS는 다수의 성능 기준을 학습하여 더 객관적인 순위를 생성하는 방식을 채택했습니다.

- **Performance Highlights**: GeoSTANet은 RMSE 1.57°C, KGE 0.89, NSE 0.85, 상관관계(r) 0.92로 다른 모델보다 낮은 RMSE를 기록하며 성능이 우수합니다. CNN-LSTM과 ConvLSTM은 대륙 및 온대 지역에서 좋은 성능을 보였습니다. 이 논문의 결과는 다기준 순위 결정 방식이 GCM 선택을 향상시키며, transformer 기반의 다운스케일링 방법이 기존의 딥러닝 방법을 초월한다는 것을 확인해줍니다.



### HVI: A New Color Space for Low-light Image Enhancemen (https://arxiv.org/abs/2502.20272)
Comments:
          Qingsen Yan, Yixu Feng, and Cheng Zhang contributed equally to this work

- **What's New**: 이 논문은 Low-Light Image Enhancement (LLIE) 분야의 새로운 접근 방식을 제안합니다. 기존의 방법들은 주로 RGB (sRGB) 색 공간에 기반을 두어 색 균형 문제와 밝기 왜곡을 발생시킵니다. 새로운 수평/수직-강도 색 공간인 Horizontal/Vertical-Intensity (HVI)를 도입하여 이러한 문제를 해결하고자 합니다.

- **Technical Details**: HVI 색 공간은 편광된 HS 맵과 학습 가능한 강도를 기반으로 정의됩니다. 이 시스템은 빨간색 좌표 간 작은 거리를 강제하여 빨간색 아티팩트를 제거하고, 저조도 영역을 압축하여 검은색 아티팩트를 줄입니다. 또한, 새로운 Color and Intensity Decoupling Network (CIDNet)를 통해 다양한 조명 조건에서 정확한 포토메트릭 맵핑 함수(photometric mapping function)를 학습합니다.

- **Performance Highlights**: 벤치마크 및 아블레이션 실험 결과, HVI 색 공간과 CIDNet을 활용한 접근법이 10개 데이터셋에서 최첨단 방법들보다 우수한 성능을 보였습니다. 이 연구는 저조도 이미지 복원 분야의 기법 개선에 크게 기여할 것으로 기대됩니다.




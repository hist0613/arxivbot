New uploads on arXiv(cs.CL)

### I Could've Asked That: Reformulating Unanswerable Questions (https://arxiv.org/abs/2407.17469)
- **What's New**: 새로운 연구에서는 사용자가 낯선 문서에서 정보를 검색할 때 발생하는 미답변 질문을 재구성(Reformulation)하는 문제에 대해 탐구하고 있습니다. 이를 위해 CouldAsk라는 평가 벤치마크(benchmark)를 새로 제작했습니다. 이 벤치마크는 기존 및 새로운 데이터셋으로 구성되었으며, 문서 기반 질문 답변(Document-grounded Question Answering)을 위한 것입니다.

- **Technical Details**: CouldAsk 벤치마크를 사용하여 최신의 오픈 소스 및 독점 대형 언어 모델(LLMs, Large Language Models)을 평가했습니다. 평가는 특히 질문 재구성 능력에 초점을 두었습니다. 실험에서 사용된 모델에는 GPT-4와 Llama2-7B가 포함되었습니다.

- **Performance Highlights**: 실험 결과, 최신 모델들도 질문을 재구성하는 데 많은 어려움을 겪는다는 것이 확인되었습니다. 구체적으로 GPT-4는 26%의 재구성 성공률을 보였으며, Llama2-7B는 12%에 그쳤습니다. 오류 분석 결과, 실패한 재구성 중 62%는 모델이 단순히 질문을 다시 표현하거나 동일한 질문을 생성하는 데서 기인한 것으로 나타났습니다. 연구진은 벤치마크와 실험 재현을 위한 코드를 공개했습니다.



### WildHallucinations: Evaluating Long-form Factuality in LLMs with Real-World Entity Queries (https://arxiv.org/abs/2407.17468)
- **What's New**: 새로운 벤치마크 'WildHallucinations'가 소개되었습니다. 이 벤치마크는 실제 사용자-챗봇 대화에서 얻은 엔터티를 통해 대형 언어 모델(LLM)의 팩츄얼리티(Factuality)를 평가합니다. 특히, 이 엔터티의 절반은 위키피디아 페이지가 없는 것이 특징입니다.

- **Technical Details**: WildHallucinations는 웹 검색에서 수집한 체계적으로 정리된 지식 소스를 바탕으로 자동으로 사실 확인을 수행합니다. 이번 평가에서는 15개의 대형 언어 모델이 생성한 118,785개의 데이터를 7,919개의 엔터티를 기준으로 평가했습니다. 이 벤치마크는 LLM들이 위키피디아 페이지가 없는 엔터티에 대해 더 자주 헛소리를(혹은 환각을) 생성한다는 것을 발견했습니다.

- **Performance Highlights**: LLM들은 위키피디아 페이지가 없는 엔터티에 대해 더 높은 환각률을 보였으며, 도메인마다 환각률이 다르게 나타났습니다. 기본 모델에 검색을 추가하는 경우 환각률이 약간 감소했지만, 완전히 없애지는 못했습니다.



### CMR Scaling Law: Predicting Critical Mixture Ratios for Continual Pre-training of Language Models (https://arxiv.org/abs/2407.17467)
- **What's New**: 이번 연구에서는 Large Language Models (LLMs)이 특정 도메인에 특화된 작업에서 성능이 저하되는 문제를 해결하기 위해 Continual Pre-training (CPT) 방식을 활용하여 도메인 특화 지식과 일반 지식을 동시에 습득하도록 개선했습니다. 그 과정에서 일반 데이터와 도메인 특화 데이터의 혼합 비율을 최적화하는 방법을 새롭게 제안합니다.

- **Technical Details**: 연구진은 LLMs의 손실(loss), 혼합 비율(mixtur ratio), 및 학습 토큰 규모(training tokens scale) 사이의 멱법칙(power-law) 관계를 발견했습니다. 이를 바탕으로 일반 능력과 도메인 특화 능력 간의 상충 관계를 공식화하여, Critical Mixture Ratio (CMR)라는 최적 비율을 제안했습니다. CMR은 모델의 전반적인 능력을 유지하면서 도메인 전이를 효과적으로 달성할 수 있도록 합니다.

- **Performance Highlights**: 광범위한 실험을 통해 CMR의 예측 가능성을 확인하였으며, CMR 스케일링 법칙(scaling law) 또한 제안하고 이를 입증했습니다. 이러한 연구 결과는 특화된 도메인의 LLM 훈련을 최적화하는 실용적인 지침을 제공하며, 일반 성능과 도메인 특화 성능을 모두 효율적으로 관리할 수 있게끔 합니다.



### Fluent Student-Teacher Redteaming (https://arxiv.org/abs/2407.17447)
- **What's New**: 최근 연구는 Llama-2 및 Phi-3와 같은 안전 조정된 모델을 대상으로 한 새로운 종류의 강력하고 유창한 공격 방법을 제안합니다. 이 연구에서는 기존 알고리즘(GCG 및 BEAST)을 개선하여, 안전 조정된 언어 모델을 대상으로 강력하고 자연스러운 공격을 수행하는 방법을 개발했습니다.

- **Technical Details**: 제안된 기술은 'distillation-based approach'(증류 기반 접근법)에 중심을 둡니다. 이 접근법을 통해 피해 모델이 감염된 상태(tuned to toxified state)를 모방하도록 유도합니다. 내부 활성화 또는 출력 확률에 기반하여 이러한 모방을 이끌어냅니다. 인간 수준의 유창함을 유지하기 위해 다중 모델 perplexity 패널티와 반복 패널티도 추가했습니다. 더 나은 최적화를 위해 토큰 삽입, 교체 및 삭제를 허용하고, 더 긴 공격 시퀀스를 사용합니다.

- **Performance Highlights**: Advbench 실험에서 Llama-2-7B, Llama-3-8B, Vicuna-7B 모델에 대해 공격 성공률 $>93$%에 도달했으며, 모델이 측정한 perplexity는 $<33$로 유지되었습니다. Phi-3 모델에 대해서는 공격 성공률 $95$%를 기록했으나 perplexity는 다소 높았습니다. 또한, Llama-2-7B, Phi-3-mini, Vicuna-7B 모델에서 이전에 보지 못한 작업에 대해 $>88$의 준수를 유도하는 단일 유창한 프롬프트를 발견했습니다.



### Dependency Transformer Grammars: Integrating Dependency Structures into Transformer Language Models (https://arxiv.org/abs/2407.17406)
- **What's New**: 새로운 Dependency Transformer Grammars (DTGs)은 Transformer 언어 모델에 명시적 의존성 기반 유도 편향을 도입합니다. 이는 기존의 구성 요소 기반 구조를 추가한 Transformer와 다르게, 의존성 트리(Dependency Trees)와 문장을 동시에 모델링하여 더 나은 일반화를 달성합니다.

- **Technical Details**: DTGs는 의존성 전이 시스템을 모방하기 위해, attention 패턴을 수정하고, attention 마스크를 변경합니다. 또한 stack 정보를 상대적 위치 인코딩을 통해 통합하고, 토큰 임베딩(embeddding)과 운영 임베딩(operation embedding)의 조합으로 의존성 아크(dependency arc) 표현을 강화합니다. 의존성 트리로 주석된 문장 데이터셋을 사용하여 훈련됩니다.

- **Performance Highlights**: DTGs는 Transformer 언어 모델 기준과 비교해 유사한 perplexity를 유지하면서도 더 나은 일반화를 달성했습니다. 또한 최근의 구성 요소 기반 모델을 능가, 의존성이 Transformer 언어 모델을 더 잘 이끌 수 있음을 보여줍니다.



### CovScore: Evaluation of Multi-Document Abstractive Title Set Generation (https://arxiv.org/abs/2407.17390)
- **What's New**: 이 논문은 문서 집합에서 추출된 주제 제목 세트의 자동 참조 없는 평가 방법론인 CovScore를 소개합니다. 기존 평가 방법이 느리고 힘든 인간 주석 절차에 크게 의존하는 것과 달리, CovScore는 최근에 도입된 LLM 기반 판사 방법에 영감을 받아 개발되었습니다.

- **Technical Details**: CovScore는 품질을 평가하는 주요 측정을 다섯 가지 주요 메트릭(metric)으로 분해하여 평가의 여러 측면을 다룹니다. 이는 수동 평가 과정을 단순화하고 신속하게 하며, 자동 및 독립적인 LLM 기반 평가를 가능하게 합니다. 이 방법론을 테스트하기 위해 홀로코스트 생존자 증언(corpus of Holocaust survivor testimonies)을 적용하여 관련성과 도덕적 중요성을 검증합니다.

- **Performance Highlights**: 이 방법론의 유효성을 확인하기 위해 자연 및 합성 제목 세트 생성 시스템을 실험하고 그 성능을 새로운 CovScore 방법론과 비교하였습니다.



### PERSONA: A Reproducible Testbed for Pluralistic Alignmen (https://arxiv.org/abs/2407.17387)
- **What's New**: PERSONA는 다수의 사용자 가치와 정렬되는 언어 모델(LMs)을 평가하고 개선하기 위해 설계된 재현 가능한 테스트 베드(test bed)를 소개합니다. 이는 다양한 사용자 의견을 포착하는 데 실패하는 현재의 선호 최적화 접근 방식을 보완합니다. PERSONA는 다양한 인구 통계 및 특이한 속성을 가진 1,586개의 합성 페르소나를 생성하여 이용자의 다원적 정렬을 평가합니다.

- **Technical Details**: PERSONA는 미국 인구조사 데이터에 기반하여 다양한 사용자 프로필을 절차적으로 생성합니다. 이를 통해 1,586개의 합성 페르소나와 3,868개의 프롬프트 및 317,200개의 피드백 쌍을 포함하는 대규모 평가 데이터를 생성합니다. 이 데이터셋을 활용하여 LM의 다양한 사용자 역할 수행 능력을 평가하며, 인간 심판을 통해 이를 검증합니다. 또한, 다원적 정렬 접근 방식을 위한 벤치마크 데이터인 'PERSONA Bench'를 설정하고 새로운 벤치마크를 만들기 위한 방대한 데이터셋을 제공합니다.

- **Performance Highlights**: PERSONA는 다양한 사용자 의견을 반영하는 능력을 갖춘 언어 모델을 평가하는 데 있어 중요한 역할을 합니다. 이를 통해 다수 의견 우선에서 벗어나 소수의 관점을 포착할 수 있는 효율적인 선호 최적화 접근 방식을 개발하고 평가할 수 있습니다. 전체 데이터셋과 벤치마크는 공개되어 있어, 연구자들이 이를 활용한 후속 연구를 진행할 수 있습니다.



### A Comprehensive Approach to Misspelling Correction with BERT and Levenshtein Distanc (https://arxiv.org/abs/2407.17383)
Comments:
          12 pages, 9 figures, 5 tables

- **What's New**: 이 연구는 신경망(neural networks) 기술을 사용하여 텍스트의 다양한 철자 오류를 식별하고 수정하는 것을 목표로 하고 있습니다. 특히, BERT(Bidirectional Encoder Representations from Transformers) 마스크드 언어 모델을 활용하여 철자 오류를 교정하는 접근법을 제안합니다.

- **Technical Details**: 연구진은 비실제 단어 오류와 실제 단어 오류를 포함하는 포괄적인 데이터셋을 수집하였습니다. 다양한 종류의 철자 오류를 분류한 후, 여러 사전 훈련된 BERT 모델을 적용하였습니다. 철자 오류 교정의 최적 성능을 보장하기 위해 BERT 마스크드 언어 모델과 Levenshtein 거리(Levenshtein distance)를 결합한 접근법을 제안합니다.

- **Performance Highlights**: 평가 데이터에서, 제안된 시스템은 페르시아어 언어에 맞춰진 기존 시스템을 능가하는 탁월한 철자 오류 식별 및 수정 능력을 보였습니다.



### Boosting Large Language Models with Socratic Method for Conversational Mathematics Teaching (https://arxiv.org/abs/2407.17349)
Comments:
          Accepted By CIKM 2024

- **What's New**: 이 논문에서는 Socratic 방식의 대화를 통해 학습자들이 스스로 깊이 생각하고 발견할 수 있도록 유도하는 수학 교육 LLM (	exttt{SocraticLLM})을 소개하고 있습니다. 이 모델은 주제에 대한 추가 지식을 제공하는 Socratic 스타일의 대화를 포함하고 있는 고품질 수학 교육 데이터셋인 	exttt{SocraticMATH}을 수집하고 공개했습니다.

- **Technical Details**: 	exttt{SocraticLLM}은 수학 문제 해결 정확도를 높이기 위해 Chain-of-Thought와 같은 기법을 사용하는 기존 방법과는 달리, 학습자에게 명확하고 자주적인 사고를 유도합니다. 이를 위해 신뢰할 수 있는 응답 생성을 위해 검토, 안내/휴리스틱, 수정, 요약을 포함하는 지식 강화 LLM을 제안합니다. 이 모델을 통해 Socratic 대화 기반의 문제 해결 과정을 강화할 수 있습니다.

- **Performance Highlights**: 실험 결과, 	exttt{SocraticLLM}은 여러 강력한 생성 모델과 비교하여 큰 장점을 보였습니다. 이 모델은 문제를 해결하는 능력뿐만 아니라 학습자에게 더 깊은 이해와 자주적인 문제 해결 능력을 부여하는 데 탁월함을 입증했습니다. 코드와 데이터셋은 공개되어 있어 추가 연구 및 적용에 사용될 수 있습니다.



### Label Alignment and Reassignment with Generalist Large Language Model for Enhanced Cross-Domain Named Entity Recognition (https://arxiv.org/abs/2407.17344)
Comments:
          9 pages, 4 figures

- **What's New**: 이 논문에서는 교차 도메인(named entity recognition) NER에서의 라벨 불일치 문제를 해결하기 위해 '라벨 정렬 및 재할당 접근법'이라는 LAR을 제안합니다. 이는 출발 도메인(source domain)과 목표 도메인(target domain) 간의 라벨 정렬과 타입 추론을 위한 라벨 재할당을 포함하는 두 가지 핵심 절차로 구성되어 있습니다.

- **Technical Details**: LAR 방법은 크게 두 가지 핵심 절차로 나눌 수 있습니다. 첫째, 라벨 정렬은 출발 도메인의 라벨 정보를 목표 도메인과 상호 연관시키는 작업을 의미합니다. 둘째, 라벨 재할당은 대형 언어 모델(large-scale language model) 예를 들어, ChatGPT와 통합하여 타입 추론 성능을 크게 향상시킬 수 있습니다. 이를 통해 교차 도메인 NER에서 라벨 충돌 문제를 최소화할 수 있습니다.

- **Performance Highlights**: LAR 방법은 여러 NER 데이터셋에서 광범위한 실험을 통해 검증되었으며, 특히 감시 환경(supervised setting) 및 영사투하 환경(zero-shot out-of-domain setting)에서 기존 최첨단(SOTA) 방법들에 비해 우수한 성능을 보여 주었습니다.



### Improving ICD coding using Chapter based Named Entities and Attentional Models (https://arxiv.org/abs/2407.17230)
Comments:
          10 Pages

- **What's New**: 최근 자연어 처리(NLP)의 발전은 여러 분야에서 자동화를 가능하게 했습니다. 그러나 임상 NLP는 종종 실제 시나리오를 정확하게 반영하지 않는 벤치마크 데이터 세트에 의존합니다. 저희 연구는 ICD 코딩을 향상시키기 위한 새 접근 방식을 도입하여, 챕터 기반 명명된 엔티티 및 어텐셔널 모델(attentional models)을 활용하여 F1 점수를 향상시킵니다.

- **Technical Details**: 이 방법은 퇴원 요약을 ICD-9 챕터로 분류하고, 챕터별 데이터와 함께 어텐셔널 모델을 개발하여 코드 식별을 위해 외부 데이터를 고려할 필요성을 없앱니다. 분류를 위해 Chapter-IV를 사용하여 중요한 엔티티와 가중치를 편향을 줄이고 신경망 없이 설정하여 정확한 임계값을 생성하며, 인간 검증을 위한 해석 가능성을 제공합니다. 검증 후에는 Attention과 Transformer with Multi-head Attention 아키텍처를 갖춘 Bidirectional-Gated Recurrent Units (GRUs)을 사용하여 Chapter-IV에서 빈번하고 비빈번한 코드에 대한 어텐셔널 모델을 개발합니다.

- **Performance Highlights**: 이 모델들의 평균 Micro-F1 점수는 0.79와 0.81로, ICD 코딩에서 상당한 성능 향상을 보여줍니다.



### NarrationDep: Narratives on Social Media For Automatic Depression Detection (https://arxiv.org/abs/2407.17174)
- **What's New**: 소셜 미디어 게시물에서 사용자의 내러티브와 의도를 자동으로 모델링하여 우울증 여부를 판단할 가능성을 탐구하는 새로운 모델 	exttt{NarrationDep}가 개발되었습니다. 이 모델은 사용자 트윗을 분석하여 중요한 내러티브를 정확하게 식별합니다.

- **Technical Details**: 	exttt{NarrationDep}는 개별 사용자 트윗 표현(representations)과 사용자의 트윗 클러스터(clusters)를 공동으로 모델링하는 심층 학습 프레임워크입니다. 두 계층(layer)으로 구성되어 있으며, 첫 번째 계층은 소셜 미디어 텍스트 게시물을 모델링하고, 두 번째 계층은 클러스터와 관련된 트윗의 의미론적 표현(submic representation)을 학습합니다. 두 번째 계층에는 사용자의 게시물로부터 계층적으로 학습하는 새로운 구성 요소가 통합되어 있습니다.

- **Performance Highlights**: 우리의 프레임워크가 다양한 데이터셋에서 최근 개발된 모델을 포함한 다른 비교 모델보다 앞서는 성능을 보였습니다.



### A Comparative Analysis of Bilingual and Trilingual Wav2Vec Models for Automatic Speech Recognition in Multilingual Oral History Archives (https://arxiv.org/abs/2407.17160)
Comments:
          Accepted to INTERSPEECH2024

- **What's New**: 이 논문에서는 단일 언어 Wav2Vec 2.0 모델과 다양한 다국어 모델을 비교하여, 혼합 언어 문장이 많이 포함된 독특한 구술 역사 아카이브에서 음성 인식 성능을 향상시킬 수 있는지 조사합니다. 본 연구의 주요 목표는 이 독특한 데이터셋에 대한 연구를 진전시키는 것으로, 이는 우리의 문화 유산의 중요한 부분입니다.

- **Technical Details**: 논문에서는 단일 언어 음성 인식 모델이 대부분의 경우 다국어 모델보다 우수함을 보여줍니다. 특히 비원어민 화자들의 혼합 언어 문장이 많은 구술 역사 아카이브를 처리할 때 이러한 경향이 두드러집니다. 더불어, 결과의 검증을 위해 CommonVoice 데이터셋에서도 동일한 실험을 수행했습니다.

- **Performance Highlights**: 본 연구는 단일 언어 모델이 다국어 모델보다 뛰어난 성능을 보임을 보여주었으며, 연구 커뮤니티에 기여하기 위해 사전 훈련된 모델을 공개합니다.



### SimCT: A Simple Consistency Test Protocol in LLMs Development Lifecyc (https://arxiv.org/abs/2407.17150)
- **What's New**: 이번 연구에서는 산업 분야에서의 대형 언어 모델(LLMs) 또는 LLMs 기반 시스템 및 서비스 개발의 표준 운영 절차를 발전시키려는 노력을 보고합니다. 우리는 대형 언어 모델 개발 생명 주기(LDLC)의 개념을 도입하고, 일관성 테스트의 중요성을 강조합니다. 현재의 실질적인 솔루션은 충분히 엄격하지 않고 노동 집약적임을 지적하고, 이를 해결하기 위해 간단하지만 효과적인 일관성 테스트 프로토콜인 SimCT를 제안합니다.

- **Technical Details**: SimCT는 'Bare Metal' LLMs 또는 관련 서비스의 다양한 개발 단계 간의 일관성을 적극적으로 점검하기 위한 프로토콜로, 모델 아티팩트에 접근하지 않고도 여러 개발 단계에 참여하는 팀 간의 반복적인 커뮤니케이션을 줄여서 배달 시간을 단축시키는 것을 목표로 합니다. 구체적으로, SimCT는 반응-wise 및 모델-wise 테스트를 포괄합니다. 우리는 LightGBM과 Student's t-test를 각각 두 가지 컴포넌트에 활용하여 프로토콜을 구현하였습니다.

- **Performance Highlights**: 광범위한 실험을 통해 SimCT와 그 관련 컴포넌트들의 효과를 입증하였습니다.



### SDoH-GPT: Using Large Language Models to Extract Social Determinants of Health (SDoH) (https://arxiv.org/abs/2407.17126)
- **What's New**: 새로운 연구에서는 SDoH-GPT라는 효율적인 대규모 언어 모델(few-shot Large Language Model, LLM)을 도입하여 의료 기록에서 사회적 건강 결정 요인(SDoH)을 추출하는 방법을 제시했습니다. 이는 기존의 노동 집약적이고 과제에 특화된 주석 작업에 대한 의존도를 줄입니다.

- **Technical Details**: SDoH-GPT는 대조적 예시(contrastive examples)와 간결한 지침(concise instructions)을 활용하여 광범위한 의료 주석이나 비용이 많이 드는 인간 개입 없이 SDoH를 추출합니다. 또한, SDoH-GPT와 XGBoost의 혁신적인 결합은 높은 정확도와 계산 효율성을 제공하며 일관되게 0.90 이상의 AUROC 점수를 유지합니다.

- **Performance Highlights**: SDoH-GPT는 시간과 비용 면에서 각각 10배 및 20배의 절감 효과를 거두었으며, 코헨의 카파(Cohen's kappa)로 측정된 인간 주석자와의 일관성에서 최대 0.92의 값을 기록하였습니다. 세 가지 다른 데이터 세트에서 테스트한 결과, 이 방법의 견고성과 정확성이 입증되었습니다.



### Behavioral Testing: Can Large Language Models Implicitly Resolve Ambiguous Entities? (https://arxiv.org/abs/2407.17125)
- **What's New**: 최근 대형 언어 모델(LLMs)의 두드러진 성능은 사전 학습 기간 동안 축적된 방대한 양의 사실적 지식 덕분입니다. 그러나 많은 LLM이 자기 모순(self-inconsistency)을 겪고 있으며, 이는 신뢰성과 신뢰성에 의문을 제기합니다. 이번 논문에서는 엔티티 타입 모호성(entity type ambiguity)에 집중하여, 모호한 엔티티에 대한 지식을 활용할 때 LLM이 얼마나 능숙하고 일관성 있게 적용하는지 분석합니다.

- **Technical Details**: 이 연구에서는 지식을 알고 있는 것과 적용하는 것을 분리하는 평가 프로토콜을 제안하며, 49개의 엔티티로 최신 LLM을 테스트합니다. 실험 결과, 모호한 프롬프트에서 LLM의 정확도는 80%에 불과하다는 것을 밝혔습니다. 추가로, 체계적인 불일치(discrepancy)가 LLM의 행동에서 발견되었으며, 정보 일관성 있게 적용하는 데 실패한다는 점을 보여줍니다.

- **Performance Highlights**: LLM는 지식을 나타내면서도 이를 활용하는 데는 능숙하지 않으며, 선호하는 해석에 의한 편향(bias)을 나타내고 자기 모순(self-inconsistency)을 드러냅니다. 이번 연구는 미래의 더욱 신뢰할 수 있는 LLM을 위해 엔티티 모호성 처리가 중요하다는 점을 강조합니다.



### A Survey Forest Diagram : Gain a Divergent Insight View on a Specific Research Topic (https://arxiv.org/abs/2407.17081)
Comments:
          This paper will submit to IEEE SMC 2024

- **What's New**: 이 연구는 초보 연구자들이 특정 분야에 익숙하지 않아서 Generative AI(생성형 인공지능)를 활용한 정보 검색 및 질문 응답에서 효율성을 크게 향상시키지 못하는 문제를 해결하려고 합니다. 이를 위해 이 연구는 인용 정보를 통해 여러 논문 간의 관계를 나타내어 연구 주제에 대한 확산적 사고를 도와주는 Survey Forest Diagram(서베이 포레스트 다이어그램)을 개발합니다.

- **Technical Details**: Survey Forest Diagram(서베이 포레스트 다이어그램)은 논문들 간의 인용 단서를 표시하여 초보 연구자들이 연구 주제에 대해 보다 폭넓은 관점을 가질 수 있도록 유도합니다. 이는 생성형 AI와의 상호작용을 통해 이루어지며, 초보 연구자들이 확산적 사고를 발전시킬 수 있도록 설계되었습니다.

- **Performance Highlights**: 이 다이어그램은 초보 연구자들이 보다 독자적이고 확산적인 방식으로 연구 자료를 탐색할 수 있도록 돕는 도구로 기능합니다. 이를 통해 초보 연구자들은 더 많은 자료에 대한 구체적이고 광범위한 조사 관점을 확보하게 됩니다.



### SAFETY-J: Evaluating Safety with Critiqu (https://arxiv.org/abs/2407.17075)
- **What's New**: 큰 언어 모델(Large Language Models, LLMs)의 콘텐츠 생성 과정에서 투명성과 해석 가능성에 대한 안전 문제를 해결하기 위해 SAFETY-J를 도입했습니다. SAFETY-J는 영어와 중국어를 지원하며, 비평 기반 평가 방식(critique-based judgment)을 적용한 안전 평가 모델입니다.

- **Technical Details**: SAFETY-J는 다양한 대화와 증강된 쿼리-응답 쌍을 포함하는 강력한 훈련 데이터 세트를 활용하여 여러 시나리오에서 안전성을 평가합니다. 자동화된 메타 평가 벤치마크(meta-evaluation benchmark)를 설정하여 최소한의 인간 개입으로 비평의 품질을 객관적으로 평가하고, 반복적 선호 학습 기법(iterative preference learning)을 사용하여 메타 평가와 비평에 따라 안전 평가를 동적으로 개선합니다.

- **Performance Highlights**: 평가 결과, SAFETY-J는 복잡한 콘텐츠 시나리오에서 더 정교하고 정확한 안전 평가를 제공하여 비평 품질과 예측 신뢰성을 향상시킵니다. 연구 및 응용을 촉진하기 위해 SAFETY-J의 훈련 프로토콜, 데이터 세트 및 코드를 오픈소스화할 예정입니다.



### From Internal Conflict to Contextual Adaptation of Language Models (https://arxiv.org/abs/2407.17023)
Comments:
          22 pages, 15 figures

- **What's New**: 이 논문에서는 지식 집약적 언어 이해 작업에서 언어 모델(LM)이 관련된 맥락을 통합하는 방법을 탐구합니다. 특히, 기존의 연구들이 분리하여 다루었던 지식 충돌의 두 가지 유형(맥락-기억 충돌과 내재적 기억 충돌)을 통합적으로 연구합니다. 이를 위해, 시간에 따라 변하고 관점에 따라 달라질 수 있는 동적 사실을 포함하는 DYNAMICQA 데이터셋을 소개합니다.

- **Technical Details**: DYNAMICQA 데이터셋은 사실의 시간적 동적 본질을 포함하며, 사실이 시간에 따라 다른 빈도로 변경될 수 있는 경우와 관점에 따라 변경될 수 있는 논란의 여지가 있는 동적 사실을 포함합니다. 이 데이터셋을 활용하여 내재적 기억 충돌을 측정하는 불확실성 측정 방법을 평가하고, 문맥(context)이 언어 모델의 의미적 출력을 설득하는 능력을 평가하기 위해 새로운 Coherent Persuasion (CP) 스코어를 도입합니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, 변경 가능성이 낮은 정적 사실들은 추가된 맥락과 함께 업데이트되는 것이 더 용이하다는 것이 밝혀졌습니다. 반면에 시간적 및 논란의 여지가 있는 사실들은 보다 어려운 업데이트를 요합니다.



### Can Language Models Evaluate Human Written Text? Case Study on Korean Student Writing for Education (https://arxiv.org/abs/2407.17022)
Comments:
          Work In Progress

- **What's New**: 이번 연구에서 대형 언어 모델(LLM: Large Language Model)에 기반한 평가 파이프라인이 인간이 작성한 텍스트를 교육 목적으로 평가할 수 있는 가능성을 조사하였습니다. 32명의 한국 학생들이 15가지 유형의 글에서 작성한 100개의 텍스트를 수집하여 GPT-4-Turbo를 통해 평가하였으며, 문법적 정확성, 유창성, 일관성, 일치성 및 관련성을 기준으로 평가하였습니다.

- **Technical Details**: GPT-4-Turbo가 문법적 정확성(grammaticality)과 유창성(fluency)은 신뢰성 있게 평가하는 것으로 나타났지만, 다른 기준이나 다양한 유형의 글에서는 어려움을 겪었습니다. 본 연구에서는 데이터를 공개하고, 각 텍스트에 대한 피드백도 함께 제공하고 있습니다.

- **Performance Highlights**: LLM 평가자는 문법적 정확성과 유창성 측면에서 높은 평가 신뢰성을 보였으며, 특히 더 객관적인 글 유형에서 이를 확인할 수 있었습니다. 하지만 일관성(coherence), 일치성(consistency), 그리고 관련성(relevance)의 평가에서는 한계를 보였습니다.



### Unveiling In-Context Learning: A Coordinate System to Understand Its Working Mechanism (https://arxiv.org/abs/2407.17011)
- **What's New**: 최근의 연구는 대형 언어 모델(LLMs)의 문맥 학습(In-Context Learning, ICL)의 작동 메커니즘을 보다 명확하게 이해하고자 두 가지 상반된 견해를 체계화하는 '2차원 좌표 시스템(Two-Dimensional Coordinate System)'을 제안했습니다. 이 시스템은 LLM이 작업(task)을 인식할 수 있는지 여부와 시연된 예시들 간의 유사성을 가지고 ICL의 작동 방식을 설명합니다.

- **Technical Details**: 이 연구는 두 가지 변수: 1) LLM이 작업을 인식할 수 있는지 여부, 2) 시연 예시의 유사성 여부를 바탕으로 ICL 행동을 체계적으로 설명합니다. '피크 역순위(metric, Peak inverse rank metric)'를 제안하여 LLM의 작업 인식 능력을 탐지하고, 유사성의 다양한 정의에 LLM이 어떻게 반응하는지를 연구합니다. 이를 통해 여러 분류 작업에서 ICL이 각 사분면에서 어떻게 작용하는지를 실험적으로 분석했습니다.

- **Performance Highlights**: 이 연구는 ICL이 다양한 대표적인 분류 작업에서 어떻게 작용하는지를 광범위한 실험을 통해 밝혔다. 또한, 생성 작업에서도 이 좌표 시스템이 ICL의 작동을 효과적으로 해석할 수 있음을 보여줍니다.



### Revisiting Who's Harry Potter: Towards Targeted Unlearning from a Causal Intervention Perspectiv (https://arxiv.org/abs/2407.16997)
- **What's New**: 이번 논문은 LLM(대형 언어 모델) 비학습 (unlearning) 방법인 WHP(Who's Harry Potter)를 조사합니다. 논문에서는 두 가지 새로운 연구 단계를 제시합니다. 첫째, 대상 비학습(targeted unlearning)이라는 새로운 작업을 도입하며, 이는 특정 대상(예: 사람)과 비학습 문서를 제공받아 그 대상에 관한 정보만을 비학습하는 것입니다. 일반적인 비학습 문서의 모든 정보를 제거하는 것이 아니라 특정 정보만을 선택적으로 제거합니다.

- **Technical Details**: 논문은 비학습 대상(targeted unlearning)인 목표의 정보를 인과적 요인(confounder)으로 모델링하는 인과 개입(causal intervention) 프레임워크를 구축합니다. 이를 통해 LLM의 입력과 출력 사이의 인과성을 제거하는 비학습 과정을 정의합니다. 이 프레임워크는 WHP를 확장하여 간단한 비학습 알고리즘을 도출하며, WHP를 특별한 경우로 포함합니다.

- **Performance Highlights**: 새로운 데이터셋과 기존 데이터셋에서 우리의 접근법은 명시적인 기준 최적화 없이도 경쟁력 있는 성능을 보였습니다. 성공적인 비학습의 기준으로 의미 없는 출력 방지, 대상에 대한 사실 왜곡 방지, 그리고 적대적 공격 (jailbreak attacks)에 대해 사실 정보 노출 금지를 충족시킵니다. 이러한 기준을 충족하면서도 해당 접근법이 유효함을 실험적으로 증명했습니다. 코드가 공개되어 있어 추가적인 재현 및 검증이 가능합니다.



### Towards Aligning Language Models with Textual Feedback (https://arxiv.org/abs/2407.16970)
- **What's New**: 이번 논문에서는 사용자 선호도를 텍스트로 표현해 언어 모델을 정렬하는 접근인 ALT(ALignment with Textual feedback)를 소개합니다. 텍스트는 더 큰 표현력을 제공하여 사용자가 단순한 비교보다 풍부한 피드백을 제공할 수 있으며, 이는 더 효율적이고 효과적인 정렬로 이어질 수 있습니다. ALT는 언어 모델링 기술을 활용하며, 최소한의 하이퍼파라미터 조정만으로도 텍스트 피드백의 주요 혜택을 얻을 수 있습니다.

- **Technical Details**: ALT는 모델의 생성을 텍스트 피드백에 조건화함으로써 모델을 정렬합니다. 우리의 방법은 전적으로 언어 모델링 기술에 의존하며 RL(강화학습) 기반의 정렬 알고리즘의 주요 이점을 제공하면서도 최소한의 하이퍼파라미터 튜닝만 필요로 합니다. 우리는 독성 감소, 요약 및 대화 응답 생성과 같은 다양한 작업에서 텍스트 피드백의 효율성과 유효성을 탐구합니다.

- **Performance Highlights**: 독성 감소 작업에서 ALT는 PPO를 능가하며, 요약 작업에서는 샘플의 20%만으로도 동일한 성능을 낼 수 있음을 발견했습니다. 또한, 기존의 대형 언어 모델(LLM)이 제공하는 제한된 및 제한 없는 텍스트 피드백과 함께 ALT를 사용하는 방법도 탐구합니다. 향후 자연어 피드백과 모델 정렬을 중심으로 한 연구 방향도 제시합니다.



### Towards Transfer Unlearning: Empirical Evidence of Cross-Domain Bias Mitigation (https://arxiv.org/abs/2407.16951)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)의 편향된 내용을 제거하기 위한 새로운 접근 방식을 탐구합니다. 전통적인 디바이어싱(debiasing) 방법은 그 한계가 있었지만, 이 연구에서는 주로 소수 민족에 대한 혐오 발언에 대한 그래디언트 상승(gradient ascent)을 통해 편향된 또는 독성 콘텐츠의 가능성을 최소화하는 기법을 제안합니다. 특히, 마스크 언어 모델링(Mask Language Modeling) 기술을 사용하여 유해한 텍스트 부분만 선택적으로 잊어버리고 분리하게 합니다.

- **Technical Details**: 이 연구에서는 마스크 언어 모델링 언러닝 기술을 사용하여 텍스트의 유해한 부분을 '언러닝(unlearning)'하게 한다는 점에서 차별화됩니다. 이 방법은 LLM이 편향되고 유해한 콘텐츠에서 선택적으로 분리될 수 있도록 합니다. 이를 통해 특정 편향의 비율을 줄이는 것이 가능합니다. 예를 들어, 젠더(gender) 편향을 줄이는 것이 다른 편향(인종이나 종교 등)을 줄이는 데에도 도움이 됨을 실험 결과로 보여줍니다.

- **Performance Highlights**: 실험 결과는 이 접근법이 언어 모델 능력을 유지하면서도 편향을 줄이는 데 효과적임을 입증합니다. 더욱이, 한 종류의 편향을 줄이는 것이 다른 종류의 편향도 완화시키는 '크로스 도메인 전이 언러닝(cross-domain transfer unlearning)'의 잠재력을 가지고 있음을 밝혀냈습니다. 이 예기치 않은 결과는 앞서 언급한 새로운 디바이어싱 접근법이 다양한 영역에서 효과적일 수 있음을 시사합니다.



### Early screening of potential breakthrough technologies with enhanced interpretability: A patent-specific hierarchical attention network mod (https://arxiv.org/abs/2407.16939)
- **What's New**: 신약 개발에서 잠재적 혁신 기술을 조기에 선별하기 위한 해석 가능한(machine learning) 접근법이 제안되었습니다. 이 접근법은 특허 텍스트를 기반으로 미래 인용 횟수를 예측하며, 특히 제약 특허에 대해 적용됩니다.

- **Technical Details**: 제안된 모델 PatentHAN(Patent-specific Hierarchical Attention Network)은 (1) 특허 청구항의 기술적 단어 의미를 포착하는 특허 전용 사전 훈련 언어 모델, (2) 청구항 수준에서 상세한 분석을 가능하게 하는 계층적 네트워크 구조, (3) 선별 과정에서 중요한 청구항을 밝혀주는 청구항 자체 주의 메커니즘을 중심으로 구성됩니다.

- **Performance Highlights**: 총 35,376개의 제약 특허를 대상으로 한 사례 연구에서 제안된 접근법이 잠재적 혁신 기술의 조기 선별에 있어서 효과적임이 입증되었습니다. 또한, 다양한 언어 모델과 청구항 타입을 사용한 추가 분석에서도 이 접근법의 견고함이 확인되었습니다.



### ScholarChemQA: Unveiling the Power of Language Models in Chemical Research Question Answering (https://arxiv.org/abs/2407.16931)
Comments:
          14 pages

- **What’s New**: 이번 연구에서는 화학 분야 논문에서 추출된 대규모 질문답변(QA) 데이터셋 ScholarChemQA를 소개합니다. 이 데이터셋은 불균형한 데이터 분포와 많은 양의 라벨이 없는 데이터 등 현실적인 어려움을 반영하고 있습니다. 이를 해결하기 위해, 특정 화학 질문에 효과적으로 답변할 수 있는 QAMatch 모델을 제안합니다.

- **Technical Details**: QAMatch 모델은 두 가지 주요 문제를 해결합니다. 첫째, 불균형한 라벨 분포 문제는 각 클래스의 역빈도를 기반으로 인스턴스별 손실을 재가중하여 소수 클래스가 다수 클래스에 의해 지배되지 않도록 합니다. 둘째, 라벨이 없는 데이터를 활용하여 SoftMix 연산을 통해 다양한 증강 데이터를 생성하고, 이에 대한 예측이 동일한 목표(즉, 의사 라벨)와 일치하도록 합니다. 의사 라벨의 품질을 보장하기 위해, 각 샘플의 의사 라벨 추정치를 원하는 실제 분포와 가깝게 맞추는 보정 절차를 제안합니다.

- **Performance Highlights**: 실험 결과, QAMatch는 최근 유사 규모의 베이스라인 모델과 대형 언어 모델(LLMs)보다 ScholarChemQA 데이터셋뿐만 아니라 네 가지 벤치마크 데이터셋에서도 뛰어난 성능을 보였습니다. 이를 통해 화학 QA 연구가 더 활성화되기를 희망합니다.



### Train-Attention: Meta-Learning Where to Focus in Continual Knowledge Learning (https://arxiv.org/abs/2407.16920)
- **What's New**: 대규모 언어 모델(Large Language Models, LLMs)의 지속적 지식 학습(Continual Knowledge Learning, CKL)에 대한 기존 연구는 주로 규제(regularization), 구조 수정(architectural modifications), 리허설 기법(rehearsal techniques)에 초점을 맞춰 망각을 방지하였습니다. 이런 방법들은 모든 토큰에 균일하게 가중치를 적용하면서 불필요한 매개변수 업데이트와 망각 증가를 초래할 수 있습니다. 이를 해결하기 위해, 새로운 CKL 접근 방식인 'Train-Attention-Augmented Language Model(TAALM)'을 제안합니다. 이 접근 방식은 토큰의 유용성에 기반해 동적으로 가중치를 예측하고 적용하여 학습 효율성을 높입니다.

- **Technical Details**: TAALM은 메타 학습 프레임워크(meta-learning framework)를 사용하여 토큰 중요도 예측을 최적화함으로써 목표된 지식 업데이트를 촉진하고 망각을 최소화합니다. 이것은 모든 토큰에 일률적인 가중치를 적용하는 전통적인 방법의 비효율성을 극복합니다. 또한 기존 벤치마크들이 학습과 유지 간의 트레이드 오프(trade-off)를 명확하게 보여주지 않기 때문에 새로운 벤치마크인 LAMA-ckl을 제안합니다.

- **Performance Highlights**: 새롭게 소개된 벤치마크와 기존의 CKL 벤치마크 실험을 통해, TAALM이 기존 방법과의 통합 시 상호 보완적인 성능을 보이며 최첨단 성능(state-of-the-art performance)을 입증했습니다.



### Generation Constraint Scaling Can Mitigate Hallucination (https://arxiv.org/abs/2407.16908)
Comments:
          7 pages; accepted at ICML 2024 Workshop on Large Language Models and Cognition

- **What's New**: 이 연구는 대규모 언어 모델(Large Language Models, LLMs)에서 발생하는 환각(hallucinations) 문제를 해결하기 위한 새로운 접근 방식을 제안합니다. 특별히 명시적 메모리 메커니즘을 갖춘 LLM에서 환각을 줄이는 방법을 탐구합니다.

- **Technical Details**: 이 연구는 메모리-증강(memory-augmented) LLM 디코더에서 발생하는 환각을 메모리 읽기 벡터(readout vector)를 스케일링함으로써 완화할 수 있음을 실험적으로 증명합니다. 이 방법은 지오메트리-영감(geometry-inspired) 방식이며, 별도의 학습 과정 없이 적용할 수 있는 특징이 있습니다.

- **Performance Highlights**: 제안된 방법은 Wikipedia와 유사한 전기 항목(biography entries)을 생성하는 과제에서 최신 LLM 편집 방법을 능가합니다. 이 방법은 생성 품질과 런타임 복잡성 측면에서 우수한 성능을 나타냅니다.



### $\textit{BenchIE}^{FL}$ : A Manually Re-Annotated Fact-Based Open Information Extraction Benchmark (https://arxiv.org/abs/2407.16860)
- **What's New**: 이번 논문에서는 Open Information Extraction (OIE) 분야에서 새로운 벤치마크인 BenchIE의 단점을 극복한 $	extit{BenchIE}^{FL}$을 소개합니다. 이 새로운 기준은 기존 BenchIE의 원칙을 완전히 따르면서도 오류, 누락 및 한계를 줄였습니다.

- **Technical Details**: $	extit{BenchIE}^{FL}$은 사실 후보(fact)가 참조 사실(reference ones)과 일치할 때 발생하는 오류와 누락을 줄이는데 초점을 맞추고 있습니다. 이를 통해 OIE 추출기의 실제 성능에 대해 더 깊이 있는 결론을 도출할 수 있게 합니다.

- **Performance Highlights**: $	extit{BenchIE}^{FL}$은 기존 BenchIE와 비교할 때 적은 오류와 누락을 통해 더 정확한 성능 평가를 제공합니다. 이를 통해 OIE 시스템의 진정한 성능을 보다 객관적으로 측정할 수 있습니다.



### Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach (https://arxiv.org/abs/2407.16833)
- **What's New**: LLM의 긴맥락(Long Context)을 이해하는 최신 기술과 RAG(Retrieval Augmented Generation)를 비교 분석한 연구입니다. 연구는 특히 Gemini-1.5와 GPT-4 같은 최신 LLM들이 긴맥락을 직접 처리하는 데서 탁월한 성능을 보임에 따라, RAG와 LC(긴맥락) 각각의 장점을 활용하려고 시도합니다.

- **Technical Details**: 연구팀은 최신 LLM 세 가지를 사용해 다양한 공공 데이터셋을 기준으로 RAG와 LC 기술을 벤치마크했습니다. 실험 결과, 리소스가 충분히 제공될 경우 LC 기술이 RAG에 비해 평균 성능에서 일관되게 우수하다는 것을 발견했습니다. 다만, RAG의 낮은 비용은 여전히 분명한 이점으로 작용하고 있습니다. 이러한 관찰에 기반하여 연구팀은 Self-Route라는 방법을 제안합니다. 이 방법은 모델의 자기 반성을 기반으로 쿼리를 RAG나 LC로 라우팅하여 계산 비용을 크게 줄이면서도 LC와 비슷한 성능을 유지합니다.

- **Performance Highlights**: Self-Route 방법은 LC의 높은 성능을 유지하면서도 RAG의 경제적인 이점을 활용해, 긴맥락을 사용하는 LLM 애플리케이션에 방향성을 제시합니다.



### A Survey of Text Style Transfer: Applications and Ethical Implications (https://arxiv.org/abs/2407.16737)
- **What's New**: 텍스트 스타일 전환 (TST) 기술이 최근 몇 년 간 많은 연구 관심을 받고 있으며, 이제는 실제 생산 및 배포 준비 단계에 접어들고 있음을 강조합니다. 이 논문은 전통적인 언어학적 접근 방식과 최근의 딥러닝 방법을 모두 포함하여, TST 애플리케이션에 대한 포괄적인 리뷰를 제공합니다. 이를 통해 TST 연구에서 애플리케이션 관점을 포함하는 것의 중요성을 논의하고 있습니다.

- **Technical Details**: TST는 언어 사용의 특정 속성, 예를 들어 예의, 형식성 또는 감정을 제어하면서 스타일에 독립적인 텍스트 내용을 변경하지 않는 것을 목표로 합니다. 본 논문에서는 감독 학습(supervised learning), 비감독 학습(unsupervised learning), 도메인 외 학습(out-of-domain learning) 등 다양한 데이터 유형과 알고리즘 개발에 중점을 두어 기존 연구를 리뷰했습니다. 또한, TST 기술의 배포 준비 상황을 고려하여 윤리적 측면도 면밀히 검토합니다.

- **Performance Highlights**: TST 관련 기술들이 점차 생산 및 배포 준비 수준에 도달하고 있으며, 이를 통해 TST 애플리케이션이 실제로 어떻게 적용될 수 있는지에 대한 다양한 예시와 활용 사례를 제공합니다. 윤리적 고려 사항 역시 중요한 논점으로 다루며, 향후 연구 방향에 대한 제언도 포함하고 있습니다.



### Educating LLMs like Human Students: Structure-aware Injection of Domain Knowledg (https://arxiv.org/abs/2407.16724)
Comments:
          N/A

- **What's New**: 새로운 방법론인 StructTuning을 소개합니다. 이는 대형 언어 모델(LLMs)을 효율적으로 특정 분야 전문가로 변형시키는 방법으로, 훈련 데이터 요구량을 단지 0.3%로 줄이면서도 전통적인 지식 삽입(performance) 성능의 50%를 달성합니다.

- **Technical Details**: StructTuning은 사람 학생들이 교과서에서 구조화된 지식을 흡수하고 이를 특정 연습 문제를 통해 실제 문제 해결에 적용하는 교육 과정에서 영감을 받았습니다. 이 방법은 두 단계로 이루어진 새로운 지식 삽입 전략을 제안합다: Structure-aware Continual Pre-Training(SCPT)과 Structure-aware Supervised Fine-Tuning(SSFT)입니다. SCPT 단계에서는 훈련 데이터를 도메인 지식의 자동 생성된 분류 체계에 따라 조직하여 LLM이 특정 전문 지식과 연결된 텍스트 세그먼트를 효과적으로 기억할 수 있게 합니다. 이후 SSFT 단계에서는 모델이 출력에서 기본 지식 구조를 드러내도록 명시적으로 프로밍(prompt)하여, 이러한 구조화된 도메인 통찰을 활용해 실질적인 문제를 능숙하게 해결할 수 있게 합니다.

- **Performance Highlights**: 우리의 방법은 LongBench와 MMedBench 데이터셋에서 폐쇄형 질문-응답(closed-book question-answering) 작업을 통해 광범위한 평가를 거쳤습니다. 놀랍게도 우리 방법은 MMedBench에서 최신 기술인 MMedLM2가 보여준 성능 향상의 50%를 추구하면서도 훈련 데이터 요구량은 단지 0.3%에 불과한 성과를 보여주었습니다. 이는 더욱 강력한 도메인 특정 LLM을 위한 StructTuning의 확장 가능성을 시사합니다. 코드도 곧 공개될 예정입니다.



### Media Manipulations in the Coverage of Events of the Ukrainian Revolution of Dignity: Historical, Linguistic, and Psychological Approaches (https://arxiv.org/abs/2407.17425)
Comments:
          14 pages

- **What's New**: 이 논문에서는 우크라이나 혁명의 존엄성(Ukrainian Revolution of Dignity) 사건의 보도에서 발생하는 조작(manipulation)의 사용을 분석합니다. 연구 대상은 온라인 신문인 우크라이나 프라우다 (Ukrainska pravda), 비소키이 자목 (Vysokyi Zamok), 그리고 ZIK이며, 이는 대중 시위 동안의 보도 내용을 다룹니다.

- **Technical Details**: 연구는 역사적(historical), 언어적(linguistic), 심리적(psychological) 접근법을 사용하여 온라인 신문의 콘텐츠를 분석합니다. 인터넷 리소스의 뉴스 보도를 평가하고 현재 가장 인기 있는 인터넷 리소스를 식별합니다. 온라인 신문의 콘텐츠를 분석 및 통계 처리하며, 데이터의 중요도 수준에 따라 (매우 중요한 데이터, 중요한 데이터, 중요하지 않은 데이터) 분류합니다. 우크라이나 혁명의 과정을 조명하는 미디어 조작을 감지하는 알고리즘을 설계하고, 정보 공격에 대처하는 방법을 개발합니다.

- **Performance Highlights**: 연구는 미디어 조작을 감지하기 위해 역사적, 언어적, 심리적 접근법을 기반으로 한 알고리즘을 개발하였으며, 이를 통해 다양한 중요도 수준의 데이터를 분석하고 분류하는데 성공했습니다. 또한, 정보 공격(counteracting information attacks)에 대처하는 방법을 개발하는 성과도 이뤘습니다.



### Explaining Spectrograms in Machine Learning: A Study on Neural Networks for Speech Classification (https://arxiv.org/abs/2407.17416)
Comments:
          5th International Conference on Artificial Intelligence and Speech Technology (AIST-2023), New Delhi, India

- **What's New**: 이 연구는 정확한 음성 분류를 위해 신경망(neural networks)이 학습한 차별적 패턴을 조사하며, 특히 모음 분류 작업에 중점을 둡니다. 이를 통해 모음 분류를 위한 신경망의 활성화와 특징을 분석하고 스펙트로그램(spectrogram)에서 신경망이 '보는 것'에 대한 통찰을 얻습니다.

- **Technical Details**: 클래스 활성화 매핑(class activation mapping)을 사용하여 모음 분류에 기여하는 주파수를 식별하고, 이러한 결과를 언어학적 지식과 비교합니다. 미국 영어 모음 데이터셋을 기반으로 한 실험을 통해 신경망의 설명 가능성을 입증하고, 무성 음성과 구별할 때의 오분류 원인과 특성에 대한 귀중한 통찰을 제공합니다.

- **Performance Highlights**: 이 연구는 모음 분류에서의 기본적인 음향 단서를 이해하는 것을 향상시키며, 신경망의 추상적 표현과 확립된 언어학적 지식을 연결함으로써 음성 인식을 개선할 수 있는 기회를 제공합니다.



### MMRA: A Benchmark for Multi-granularity Multi-image Relational Association (https://arxiv.org/abs/2407.17379)
Comments:
          VLMS, Multi-Image Association

- **What's New**: 새로운 연구는 이미지 인식 작업에서 큰 성공을 거둔 대형 시각 언어 모델(LVLMs)이 인간처럼 세상을 인식하도록 하기 위한 노력이 증가하고 있다는 것을 강조합니다. 현재의 다중 모달 벤치마크는 주로 이미지 내의 잠재 지식을 중점으로 하지만, 여러 이미지 간의 연관 관계를 간과하고 있습니다. 이에 따라 다중 이미지 연관 과제를 정의하고, 1026개의 샘플로 구성된 다중 세분화 다중 이미지 연관 벤치마크(MMRA)를 신중히 큐레이션했습니다.

- **Technical Details**: MMRA 벤치마크는 ConceptNet의 관계를 기반으로 이미지 간의 연관 관계 시스템을 설정하며, 두 가지 세분화 수준 ('이미지'와 '엔티티')에서 11가지 부과제(UsageSimilarity, SubEvent 등)를 포함합니다. 이를 통해 주요 LVLMs를 체계적이고 포괄적으로 평가합니다.

- **Performance Highlights**: MMRA 벤치마크 실험 결과, 현재 주요 LVLMs는 각기 다른 부과제에서 장단점을 보입니다. 엔티티 수준에서의 성능이 이미지 수준보다 떨어져서, 세분화된 다중 이미지 인식 작업이 여전히 LVLMs에게는 도전적이라는 것을 나타냅니다. 공간적 인식과 관련된 과제는 LVLMs가 처리하기 어려운 것으로 드러났습니다. 그러나 LVLMs는 이미지 세부 사항 인식에 뛰어난 능력을 보여주며, 다중 이미지 연관 능력을 강화하는 핵심은 언어 모델 구성 요소의 추론 능력을 강화하는 것입니다.



### How Good (Or Bad) Are LLMs at Detecting Misleading Visualizations? (https://arxiv.org/abs/2407.17291)
Comments:
          To be presented at IEEE VIS 2024

- **What's New**: 이 연구는 정보 전달의 신뢰성을 저해하는 오도된 차트(misleading charts) 문제를 해결하려 합니다. 최근의 멀티모달 대형 언어 모델(multimodal Large Language Models, LLMs) 발전이 이 문제를 해결하는 유망한 방향을 제시하고 있습니다.

- **Technical Details**: 우리는 인터넷에서 수집된 오도된 차트 데이터셋과 9개의 다양한 프롬프트(prompts)를 사용하여 4개의 서로 다른 멀티모달 LLMs가 21개 이상의 차트 문제를 감지하는 능력을 테스트했습니다. 세 가지 실험을 통해 LLMs가 오도된 차트를 식별하는 능력과 프롬프트에 대한 반응성을 분석하고, 탐색 문제에서 21개의 문제까지 확장하는 과정에서의 확장성 문제를 해결하는 전략을 개발했습니다.

- **Performance Highlights**: 이 연구의 결과, 멀티모달 LLMs는 차트 해석과 데이터 해석에서 뛰어난 비판적 사고 능력을 가지고 있음을 확인할 수 있었습니다. 멀티모달 LLMs를 사용하여 오도된 정보를 차단하고 시각화 능력을 향상시키는 데 큰 잠재력이 있음을 보여주었습니다.



### LEAN-GitHub: Compiling GitHub LEAN repositories for a versatile LEAN prover (https://arxiv.org/abs/2407.17227)
- **What's New**: 최근 대형 언어 모델(large language models)이 수학적 추론(formal mathematical reasoning)을 지원하는 데 있어 유망한 성과를 보이고 있습니다. 그러나 공식적인 정리 증명(formal theorem-proving) 데이터가 부족하여 성능이 제한되고 있습니다. 이를 해결하기 위해, 우리는 GitHub에 있는 거의 모든 Lean 4 저장소(공식 데이터베이스)에서 추출한 대규모 공식 데이터를 포함하는 LEAN-GitHub를 제안합니다.

- **Technical Details**: 이 데이터셋을 토대로 InternLM-math-plus 모델을 미세 조정(fine-tuning)한 결과, 단일 패스 시 48.8%의 정확도, 64회 패스 시 54.5%의 정확도를 달성했습니다. 이는 최첨단 기술(state-of-the-art)의 52%를 초과하는 성과입니다. 또한, 우리 모델은 다른 두 Lean 4 벤치마크(ProofNet과 Putnam)에서도 최첨단 성과를 기록하였습니다. 다양한 수학 주제에 대해 유용한 공식 추론 데이터를 제공함을 입증했습니다.

- **Performance Highlights**: 우리 모델은 Lean 4 miniF2F 테스트에서 단일 패스로 48.8%, 64회 패스로 54.5%의 정확도를 기록하였습니다. 또 다른 Lean 4 벤치마크인 ProofNet과 Putnam에서도 최첨단 성과를 달성했습니다. 이는 광범위한 수학 주제에 대한 공식 추론에서 우리의 데이터셋이 유익하다는 것을 보여줍니다.

- **Links**: {'Model': 'https://GitHub.com/InternLM/InternLM-Math', 'Data': 'https://datasets/InternLM/Lean-GitHub'}



### Speech Editing -- a Summary (https://arxiv.org/abs/2407.17172)
- **What's New**: 비디오 제작과 소셜 미디어의 성장으로 인해, 발음 오류, 누락된 단어, 혹은 말더듬 문제를 해결하는 음성 편집의 중요성이 대두되고 있습니다. 이 논문은 텍스트 기반 음성 편집 방법을 탐구하여 텍스트 원고를 통해 오디오를 수정하며, 수동적인 파형(Waveform) 편집 없이도 가능합니다. 특히 최근에는 맥락-인식 프로소디(Context-aware prosody) 수정과 고급 어텐션 메커니즘(Attention Mechanisms) 같은 최신 기술이 음성 편집 품질을 크게 향상시켰습니다.

- **Technical Details**: 이 논문은 멜-스펙트로그램(Mel-spectrogram)을 수정하여 편집된 오디오가 원본과 구분되지 않도록 합니다. 프러소디(Context-aware prosody) 수정과 어텐션 메커니즘(Attention mechanisms) 같은 최신 기술 도입으로 음성 편집의 품질이 크게 향상되었습니다. 다양한 최신 기법들을 검토하고 주요 지표들을 비교하였으며, 널리 사용되는 데이터셋도 분석하였습니다.

- **Performance Highlights**: 최신 기법들을 비교 분석한 결과, 맥락-인식 프로소디 수정과 고급 어텐션 메커니즘이 도입된 방법들이 특히 뛰어난 성능을 보였습니다. 이러한 기술들은 편집된 오디오가 원본과 거의 구분되지 않을 정도로 자연스러운 음성을 제공합니다. 이로 인해 음성 편집의 품질이 한층 더 향상되었으며, 음성 편집의 문제점들을 해결하는 데 큰 기여를 했습니다.



### Zero-Shot vs. Few-Shot Multi-Speaker TTS Using Pre-trained Czech SpeechT5 Mod (https://arxiv.org/abs/2407.17167)
Comments:
          Accepted to TSD2024

- **What's New**: 본 논문에서는 대규모 데이터셋에 사전 학습된 SpeechT5 모델에 대한 실험 결과를 다루고 있습니다. 우리는 기초 모델을 처음부터 사전 학습시키고, 대규모 다중 화자 텍스트-음성 변환(TTS) 작업에 대해 미세 조정(fine-tuning)을 수행했습니다.

- **Technical Details**: 이번 연구에서는 제로-샷(zero-shot) 및 퓨-샷(few-shot) 시나리오에서 모델의 성능을 테스트했습니다. 두 가지 청취 테스트를 기반으로 합성 오디오의 품질과 합성 음성이 실제 음성을 얼마나 유사하게 닮았는지 평가했습니다. 이러한 평가를 통해, SpeechT5 모델이 목표 화자의 데이터를 단 1분만 사용하여 해당 화자의 합성 음성을 생성할 수 있음을 확인했습니다.

- **Performance Highlights**: 우리의 결과는 유명 체코 정치인 및 셀러브리티의 합성 음성에서 높은 품질과 유사성을 성공적으로 입증했습니다.



### High Efficiency Image Compression for Large Visual-Language Models (https://arxiv.org/abs/2407.17060)
- **What's New**: 최근 몇 년 동안, 대형 시각 언어 모델(Large Visual Language Models, LVLMs)이 다중 모달 작업에서 인상적인 성능과 유망한 일반화 능력을 보여주었으며, 다양한 응용 시나리오에서 시각 정보의 수신자로서 인간을 대체하고 있습니다. 이 논문에서는 LVLMs에 적합한 유망한 비율-정확도 성능을 달성하기 위해 사전 편집 모듈과 종단 간 코덱으로 구성된 가변 비트레이트 이미지 압축 프레임워크를 최초로 제안합니다.

- **Technical Details**: 특정 작업 또는 여러 대표 작업에 대한 적응형 사전 편집 네트워크를 최적화하는 대신, 우리는 토큰 수준의 왜곡과 순위로 표현 및 판별 능력을 기반으로 한 새로운 LVLM 최적화 전략을 제안합니다. 사전 편집 모듈과 가변 비트레이트 종단 간 이미지 코덱은 대형 모델의 의미론적 토큰(Semantic Tokens)을 기반으로 한 손실에 의해 공동으로 훈련되어 다양한 데이터와 작업에 대한 일반화 능력을 향상시킵니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크가 최신 코딩 표준인 범용 비디오 코딩(Versatile Video Coding, VVC)과 비교하여 훨씬 나은 비율-정확도 성능을 효율적으로 달성할 수 있음을 보여줍니다. 또한, 다중 모달 작업에 대한 실험은 제안된 프레임워크의 견고성과 일반화 능력을 나타냅니다.



### A Voter-Based Stochastic Rejection-Method Framework for Asymptotically Safe Language Model Outputs (https://arxiv.org/abs/2407.16994)
Comments:
          7 pages, 2 figures

- **What's New**: 이 연구는 대형 언어 모델(LLM)에서 발생할 수 있는 안전하지 않거나 품질이 낮은 출력을 방지하기 위한 새로운 방법을 제안합니다. 연구진은 LLM 체커(LLM checkers)가 생성된 출력의 수용 가능성에 대해 투표하고, 거부 임계치에 도달하면 출력을 재생성하는 시스템을 제안합니다.

- **Technical Details**: 이 시스템은 LLM의 확률적 특성을 활용합니다(stochasticity of LLMs). 연구진은 비용과 실패율을 추정하는 추정기(estimators)를 제안하며, 이 추정기와 응용 프로그램에 맞춘 실험 데이터를 기반으로 최소한의 비용으로 원하는 실패율을 달성하는 알고리즘을 제안합니다.

- **Performance Highlights**: 이 모델 하에서, 투표자 수와 임계치를 알고리즘에 따라 선택할 경우 실패율이 비용의 함수로서 기하급수적으로 감소한다는 점을 실험을 통해 입증하였습니다. 또한, 제한된 데이터로도 이러한 시스템의 실제 성능을 합리적으로 평가할 수 있음을 보여줍니다.



### Free to play: UN Trade and Development's experience with developing its own open-source Retrieval Augmented Generation Large Language Model application (https://arxiv.org/abs/2407.16896)
- **What's New**: 최근 ChatGPT의 GPT-3.5 모델 출시 이후, 대규모 언어 모델(LLMs)을 포함한 생성 인공지능(AI)이 폭발적으로 인기와 관심을 끌고 있습니다. 이러한 강력한 모델들로 인해 다양한 도메인, 특히 공식 통계 및 국제 기구의 업무에서 매우 유용할 수 있습니다. UNCTAD에서는 자체 오픈 소스 Retrieval Augmented Generation (RAG) LLM 애플리케이션을 개발했습니다. 이는 조직의 특정 도메인과 업무에 맞추어 LLM을 더욱 유용하게 만듭니다.

- **Technical Details**: UNCTAD의 애플리케이션 개발에는 세 개의 주요 라이브러리(nlp_pipeline, local_rag_llm, streamlit_rag)가 사용되었습니다. 첫 번째, 'nlp_pipeline'은 문서 처리 및 통계 분석을 위해 사용됩니다. 두 번째, 'local_rag_llm'은 로컬 RAG LLM을 실행하는 라이브러리입니다. 세 번째, 'streamlit_rag'는 사용자 인터페이스를 위한 라이브러리입니다. 이 모든 라이브러리는 PyPI와 GitHub에서 Dockerfiles와 함께 공개적으로 제공됩니다. 추가적으로, 기존 LLM을 미세 조정(fine-tuning)할 수 있는 'local_llm_finetune' 라이브러리도 제공됩니다.

- **Performance Highlights**: 자체 솔루션을 개발하는 것에는 비용 및 유연성, 지식 습득 등의 장점이 있지만, 시간과 기술 투자, 애플리케이션의 완성도와 성능 측면에서는 단점이 존재합니다. 이번 UNCTAD의 접근 방식은 특히 예산이 제한된 국제 기구에 유용할 수 있으며, 공공자원으로 활용 가능한 오픈 소스를 통해 더욱 쉽게 도입할 수 있습니다.



### The Price of Prompting: Profiling Energy Use in Large Language Models Inferenc (https://arxiv.org/abs/2407.16893)
Comments:
          11 pages, 5 figures. Submitted to NeurIPS 2024. The released code and dataset are available at this https URL

- **What's New**: MELODI(모니터링 에너지 레벨과 데이터 주도 추론 최적화)라는 새로운 프레임워크를 도입했습니다. MELODI는 대형 언어 모델(LLM) 추론 과정에서 소비되는 에너지를 모니터링하고 분석할 수 있는 다면적 도구입니다. 이 프레임워크는 다양한 배포 시나리오에서 에너지 효율성을 반영하는 포괄적인 데이터셋을 생성할 수 있습니다.

- **Technical Details**: MELODI는 LLM 배포 프레임워크, 여러 언어 모델, 광범위한 프롬프트 데이터셋 등을 포함하는 광범위한 데이터셋을 생성합니다. 이 데이터셋을 바탕으로 프롬프트의 길이와 복잡성 같은 속성이 에너지 소비와 어떻게 상관있는지 조사합니다. 우리의 연구 결과는 에너지원 소비 효율성에 상당한 차이가 있음을 보여주며, 이는 LLM 배포 최적화와 지속 가능한 조치를 채택할 여지가 많음을 시사합니다.

- **Performance Highlights**: MELODI는 에너지 효율성에 대한 상세한 분석을 가능하게 하며, 다양한 배포 시나리오에서의 에너지 사용을 비교할 수 있는 데이터를 제공합니다. 이 데이터는 다른 연구자들이 확장할 수 있는 새로운 자원이기도 합니다. MELODI는 에너지 의식적인 LLM 배포 연구를 진전시키는 기초 도구이자 데이터셋으로서, 보다 지속 가능한 미래로 나아가는 데 기여할 것입니다.



### Exploring Fusion Techniques in Multimodal AI-Based Recruitment: Insights from FairCVdb (https://arxiv.org/abs/2407.16892)
- **What's New**: 이 연구에서는 다중모달리티(AI-based recruitment systems)를 사용하는 AI 기반 채용 시스템에서의 공정성과 편향의 영향을 조사했습니다. 기존의 단일 모달리티(표 형 데이터, 이미지, 텍스트 등)에 대한 공정성 인식 학습에 대한 연구는 많이 이루어졌지만, 다양한 모달리티를 융합하여 종합적으로 분석하는 다중모달 데이터에 대한 연구는 부족했습니다. 본 논문은 이런 다중모달 데이터의 융합 기법이 공정성에 미치는 영향을 분석합니다.

- **Technical Details**: FairCVdb라는 데이터셋을 사용하여 다중모달리티(AI-based recruitment systems) 채용 시스템의 공정성과 편향성을 실험했습니다. 연구에서는 early-fusion과 late-fusion 두 가지 융합 기법을 비교했습니다. early-fusion은 각 모달리티의 고유 특성을 통합하여 두 인구 집단 모두에서 실제 값과 가깝게 일치했으며, Mean Absolute Error(MAE)이 가장 낮았습니다. 반면, late-fusion은 일반화된 평균 점수를 생성하여 더 높은 MAE를 보였습니다.

- **Performance Highlights**: early-fusion 기법이 공정성과 정확성 면에서 late-fusion보다 더 우수하며, 인구 집단 간의 편향을 줄이는 데에도 효과적임을 발견했습니다. 특히, early-fusion은 demographic biases(인구 통계학적 편향)가 존재하더라도 더 정확하고 공정한 애플리케이션에 중요한 잠재력을 가지고 있다고 강조합니다. 향후 연구에서는 대안적인 융합 전략을 탐색하고, 모달리티 관련 공정성 제약을 포함하여 공정성을 개선할 수 있는 방법을 모색할 수 있습니다.



### Cultural Value Differences of LLMs: Prompt, Language, and Model Siz (https://arxiv.org/abs/2407.16891)
Comments:
          20 pages

- **What's New**: 이번 연구는 대형 언어 모델(LLM)들이 발현하는 문화적 가치의 행동 패턴을 식별하려는 것을 목적으로 합니다. 질문 순서, 프롬프트 언어 그리고 모델의 크기를 다루었습니다. 우리의 실험은 테스트된 각 LLM이 다양한 문화적 가치와 함께 효율적으로 행동할 수 있음을 보여줍니다.

- **Technical Details**: 연구는 세 가지 주요 발견사항을 보였습니다: (i) 단일 언어로 프롬프트를 제시하면 LLM은 비교적 일관된 문화적 가치를 나타냅니다. (ii) 프롬프트 언어 (예: 중국어 또는 영어)가 문화적 가치의 표현에 영향을 미칠 수 있습니다. 동일한 질문도 다른 언어로 LLM을 쿼리할 때 다양한 문화적 가치를 유발할 수 있습니다. (iii) 동일 모델 내에서의 크기 차이 (예: Llama2-7B vs 13B vs 70B)가 모델 간의 차이 (예: Llama2 vs Mixtral)보다 더 큰 문화적 가치 차이를 초래합니다.

- **Performance Highlights**: 쿼리 언어와 LLM의 모델 크기가 문화적 가치의 차이를 발생시키는 주요 요인임을 우리의 실험이 밝혀냈습니다.



### GPT-4's One-Dimensional Mapping of Morality: How the Accuracy of Country-Estimates Depends on Moral Domain (https://arxiv.org/abs/2407.16886)
- **What's New**: 본 연구는 이전 연구에서 사용된 Open AI의 GPT 모델이 국가 간 도덕적 의견 차이를 예측할 수 있지만, 저소득 국가에 비해 고소득 국가에서 정확도가 현저히 높다는 결과를 재현하고 더 나아가 다양한 유형의 도덕적 질문에 대한 정확도가 어떻게 달라지는지를 조사합니다. 본 연구에서는 63개국을 대상으로 18개의 도덕적 문제에 대한 응답을 세계 가치 조사(World Value Survey)와 유럽 가치 조사(European Value Study)를 통해 분석하였습니다.

- **Technical Details**: 본 연구에서는 국가별 평균 점수를 계산하여 각 도덕적 문제에 대한 GPT-4의 예측과 비교하였습니다. 요인 분석을 통해 GPT-4는 주로 보수/진보 정도를 반영하는 단일 차원에 기초하여 예측을 수행했다는 것을 밝혔습니다. 반면 실제 도덕적 풍경은 개인-성적 문제와 폭력-부정직 문제로 구분되는 이차원적 특성을 가지고 있었습니다.

- **Performance Highlights**: GPT-4는 개인-성적 도덕적 문제 도메인에서는 높은 예측 정확도를 보였으며, 고소득 국가(r = .77)와 저소득 국가(r = .58) 모두에서 높은 상관관계를 보여주었습니다. 그러나 폭력-부정직 도메인에서는 예측 정확도가 고소득 국가(r = .30)와 저소득 국가(r = -.16) 모두에서 크게 떨어졌습니다. 이는 GPT-4의 단일 차원적 세계관이 도덕적 풍경의 복잡성을 완전히 포착하지 못했음을 나타냅니다.



### CompBench: A Comparative Reasoning Benchmark for Multimodal LLMs (https://arxiv.org/abs/2407.16837)
- **What's New**: 일상 생활에서 물체, 장면 또는 상황을 비교하는 능력은 효과적인 의사결정과 문제 해결에 필수적입니다. 하지만 광범위한 인공지능(AGI)에서는 이 비교 능력이 크게 탐구되지 않았습니다. 이번 논문에서는 다양한 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 비교 추론 능력을 평가하기 위해 CompBench라는 벤치마크를 도입했습니다.

- **Technical Details**: CompBench는 시각적으로 지향된 질문들을 통해 8가지 상대적 비교 차원을 다룹니다: 시각적 속성(visual attribute), 존재(existence), 상태(state), 감정(emotion), 시간성(temporality), 공간성(spatiality), 양(quantity), 그리고 질(quality). 다양한 시각 데이터셋과 CLIP 유사도 점수를 사용해 40,000여 개의 이미지 쌍을 수집했습니다. 질문들은 두 이미지 간 상대적 특징을 구별하기 위해 신중하게 제작되었으며, 인간 주석자들에 의해 정확성과 관련성을 검사받았습니다.

- **Performance Highlights**: CompBench를 사용해 GPT-4V(ision), Gemini-Pro, 그리고 LLaVA-1.6과 같은 최신 MLLMs를 평가한 결과, 이들의 비교 능력에는 상당한 한계가 있음을 발견했습니다. CompBench는 이러한 한계를 조명함으로써 MLLMs의 비교 능력 향상을 위한 견고한 기초를 마련합니다.



### VisMin: Visual Minimal-Change Understanding (https://arxiv.org/abs/2407.16772)
Comments:
          Project URL at this https URL

- **What's New**: 새롭게 도입된 VisMin(Visual Minimal-Change Understanding) 벤치마크는 이미지와 캡션의 최소 변경을 기반으로 이미지-캡션 매칭을 예측해야 하는 모델의 능력을 평가합니다. 이는 기존 벤치마크와 달리 두 이미지와 두 캡션 사이에서 하나의 변화만 포함되도록 설계되었습니다.

- **Technical Details**: VisMin 벤치마크는 객체, 속성(예: 색상, 재질, 모양), 수량, 공간적 관계에서 하나의 변경 사항만 반영됩니다. 이 벤치마크의 구축에는 대형 언어 모델(large language models)과 디퓨젼 모델(diffusion models)을 사용했으며, 4단계의 엄격한 검증 절차를 거친 인간 주석자(human annotators)가 포함되었습니다.

- **Performance Highlights**: 실험 결과 현존하는 많은 Visual-Language Models(VLMs)이 공간적 관계와 수량 감지 능력에서 상당한 결함을 보였음을 확인했습니다. 새로운 대규모 데이터셋을 생성하고 CLIP과 Idefics2 모델을 미세 조정하여, 세분화된 이해 능력 및 이미지-텍스트 정렬에서 상당한 개선을 보여주었습니다. 모든 리소스, 벤치마크, 학습 데이터 및 미세 조정된 모델 체크포인트는 공개될 예정입니다.



### Benchmarks as Microscopes: A Call for Model Metrology (https://arxiv.org/abs/2407.16711)
Comments:
          Conference paper at COLM 2024

- **What's New**: 현대 언어 모델(LMs)의 능력 평가에 새로운 도전 과제가 생겼습니다. 기존의 정적 벤치마크는 언어 모델 기반 시스템의 배포 허용 범위에 대한 신뢰성을 제공하지 못한 채 포화 상태에 이릅니다. 개발자들은 이러한 결함이 있는 지표에 근거하여 모델이 추론 또는 열린 도메인 언어 이해와 같은 일반화된 특성을 가지고 있다고 주장합니다. LMs의 과학과 실습에는 동적 평가(dynamic assessments)를 통한 특정 능력을 측정하는 새로운 벤치마킹 접근 방식이 필요합니다.

- **Technical Details**: 메트릭스에 대한 신뢰를 가지려면 배포 중 성능을 예측할 수 있는 벤치마크를 생성하는 방법에 초점을 맞춘 모델 계측(model metrology)의 새로운 분야가 필요합니다. 평가 기준에 따라, 시스템 능력을 측정하는 도구를 구축하고 연구하는 모델 계측 실무자 커뮤니티를 구축하는 것이 필요합니다.

- **Performance Highlights**: LMs의 성능 평가에 있어 동적 평가를 사용하면 실제 배포 환경에서의 성능을 더 정확히 예측할 수 있습니다. 이는 AI 논의에 더 명확성을 더할 것입니다.



New uploads on arXiv(cs.IR)

### Intent-Guided Heterogeneous Graph Contrastive Learning for Recommendation (https://arxiv.org/abs/2407.17234)
Comments:
          14pages, 11figures

- **What's New**: 이번 연구에서는 '의도 안내 이종 그래프 대조 학습 (Intent-Guided Heterogeneous Graph Contrastive Learning, IHGCL)' 프레임워크를 제안합니다. 이 프레임워크는 메타-패스(meta-paths)에 포함된 잠재적 의도를 캡처하여 대조 학습(Contrastive Learning, CL) 기반 추천 시스템의 성능을 향상시키는 것을 목표로 합니다.

- **Technical Details**: IHGCL 프레임워크는 두 가지 주요 요소를 포함합니다: i) 잠재적 의도를 효과적으로 통합하기 위해 메타-패스 기반 이중 대조 학습(metapath-based dual contrastive learning) 접근 방식을 사용하며, 메타-패스 대조(meta-path contrast)와 뷰 대조(view contrast)를 구성합니다; ii) 메타-패스가 도입한 노이즈를 상당히 줄이기 위해 마스크 전파(mask propagation)와 정보 병목 원칙(information bottleneck principle)을 결합한 병목 오토인코더(bottlenecked autoencoder)를 사용합니다.

- **Performance Highlights**: 여섯 개의 독립된 데이터셋을 대상으로 한 경험적 평가에서 IHGCL 프레임워크는 기존의 기준 방법(baseline methods)들보다 우수한 성능을 보였습니다. 모델 구현은 온라인에서 사용할 수 있습니다.



### Reinforced Prompt Personalization for Recommendation with Large Language Models (https://arxiv.org/abs/2407.17115)
- **What's New**: 새로운 연구에서는 개별 사용자에게 맞춤형 프롬프트(instance-wise prompting)를 제공하는 Reinforced Prompt Personalization (RPP)을 제안합니다. 기존의 작업별 프롬프트(task-wise prompting)는 사용자 간의 차이를 무시했지만, 이번 연구는 이러한 문제를 해결하고자 합니다. 또한, RPP+라고 불리는 확장판은 LLMs를 사용하여 반복적인 과정에서 동적으로 행동공간(action space)을 정제하며, 그 효율성을 더욱 높입니다.

- **Technical Details**: RPP는 다중 에이전트 강화 학습(Multi-Agent Reinforcement Learning, MARL)을 사용하여 프롬프트의 네 가지 패턴(역할 놀이(role-playing), 히스토리 기록(history records), 추론 가이드(reasoning guidance), 출력 형식(output format))을 최적화합니다. 프롬프트 개인화는 단어별 최적화가 아닌 네 가지 패턴 전체에서 최적의 문장을 선택하는 방식으로 진행됩니다. 또한 RPP는 특정 추천 작업에 대한 여러 분석 관점을 고려하여 네 가지 패턴에 대해 다양한 표현을 신중하게 제작합니다.

- **Performance Highlights**: RPP/RPP+의 성능은 여러 데이터셋의 순위 매기기 작업에서 기존의 추천 모델, few-shot 방법, 다른 프롬프트 기반 방법들보다 우수한 성과를 보였습니다. 이는 추천 작업에서 LLM의 개별 프롬프트(instance-wise)가 중요하며, RPP/RPP+의 효과를 뒷받침합니다.



### A Standardized Machine-readable Dataset Documentation Format for Responsible AI (https://arxiv.org/abs/2407.16883)
Comments:
          10 pages, appendix

- **What's New**: 이번 논문에서는 AI 데이터셋의 발견 가능성(discoverability), 상호운용성(interoperability), 신뢰성(trustworthiness)을 향상시키기 위해 설계된 기계-판독 가능한 메타데이터 형식인 Croissant-RAI를 소개합니다. 이 형식은 기존의 Croissant 메타데이터 형식을 확장하고 responsible AI (RAI) 문서화 프레임워크를 기반으로 하고 있습니다.

- **Technical Details**: Croissant-RAI는 데이터셋 사용자가 RAI 메타데이터를 손쉽게 찾고 활용할 수 있도록 URL 기반의 웹 출판 관행을 활용하여 설계되었습니다. 주요 데이터 검색 엔진, 저장소 및 머신러닝 프레임워크에 원활히 통합되어 실무자들의 기존 워크플로우 내에서 책임 있는 AI 메타데이터의 읽기와 쓰기를 간소화합니다. 또한 Python 라이브러리와 시각적 편집기(visual editor)의 지원을 받으며, 커뮤니티 주도 하에 개발되었습니다.

- **Performance Highlights**: Croissant-RAI는 진화하는 문서화 요구사항에 적응 가능하도록 설계되었으며, 커뮤니티 전체의 채택을 촉진하기 위해 표준화된 속성과 관행을 제공합니다.



### Pareto Front Approximation for Multi-Objective Session-Based Recommender Systems (https://arxiv.org/abs/2407.16828)
- **What's New**: 이번 연구는 MultiTRON을 소개합니다. 이는 파레토 프론트(approximation techniques to pareto front)를 다중 목표 세션 기반 추천 시스템(multi-objective session-based recommender systems)을 위해 활용하는 접근 방식입니다. 이 접근법은 변환기 신경망(transformer neural network)을 사용하여 클릭률(click-through rate)과 전환율(conversion rate)과 같은 주요 지표 간의 트레이드오프(trade-offs)를 최적화합니다.

- **Technical Details**: MultiTRON은 샘플링된 선호 벡터(preference vectors)를 학습하는 방식으로 훈련되어, 단일 모델(single model)이 전체 파레토 프론트를 이용할 수 있게 합니다. 이는 추가 입력 벡터(additional input vector)를 조정하여 다른 이해 관계자들의 특정 요구 사항을 충족시킬 수 있음을 의미합니다. 모델의 성능은 광범위한 오프라인 및 온라인 평가(extensive offline and online evaluation)를 통해 검증되었습니다.

- **Performance Highlights**: 결과는 모델이 여러 추천 목표(recommendation objectives)를 효과적으로 관리할 수 있음을 확인시켜 주었으며, 다양한 비즈니스 요구에 유연하게 대응할 수 있는 도구임을 증명했습니다.



### BlueTempNet: A Temporal Multi-network Dataset of Social Interactions in Bluesky Socia (https://arxiv.org/abs/2407.17451)
Comments:
          to appear in IEEE Data Description

- **What's New**: 분산형 소셜 미디어 플랫폼인 Bluesky Social(이하 Bluesky)이 사용자의 행동을 밀리초 수준까지 공개할 수 있게 되었습니다. 이를 기반으로 사용자 주도의 소셜 상호작용의 시간적 역동성을 처음으로 수집한 BlueTempNet이 소개되었습니다.

- **Technical Details**: BlueTempNet은 사용자 간의 상호작용(user-to-user interactions)과 사용자와 커뮤니티 간의 상호작용(user-to-community interactions)을 단일 다중 네트워크(multi-network)로 통합합니다. 사용자 간의 팔로잉(following)과 차단(blocking)뿐 아니라 커뮤니티 생성 및 가입 등 다양한 상호작용을 포함합니다. 또한 Bluesky의 공개 데이터 정책에 따라 기존 Feeds와 해당 Feeds를 좋아하거나 생성한 사용자들의 데이터를 수집하고, 특정 날짜 범위 내에서 사용자 상호작용을 수집할 수 있는 도구를 제공합니다.

- **Performance Highlights**: 이 데이터 수집 전략은 과거 사용자 행동을 포착함과 동시에 향후 사용자 행동 데이터를 수집하는 데에도 도움을 줍니다.



### A Novel Two-Step Fine-Tuning Pipeline for Cold-Start Active Learning in Text Classification Tasks (https://arxiv.org/abs/2407.17284)
Comments:
          11 pages, 4 figures, 2 Tables, and 1 algorithm

- **What's New**: 이 논문은 콜드 스타트 시나리오에서 전통적인 파인 튜닝이 라벨이 된 데이터의 부재로 인해 불가능한 상황에서 BERT 기반의 문맥 임베딩의 효과를 조사한 첫 번째 연구입니다. 주요 기여는 라벨이 된 데이터의 의존성을 줄이는 보다 강력한 파인 튜닝 파이프라인인 DoTCAL을 제안하는 것에 있습니다. DoTCAL은 두 가지 단계로 구성됩니다: (1) 마스킹된 언어 모델링을 통해 임베딩의 도메인 적응을 통해 라벨이 되지 않은 데이터를 완전히 활용하고 (2) AL이 선택한 라벨이 된 데이터를 사용하여 모델 가중치를 추가로 조정합니다.

- **Technical Details**: DoTCAL은 두 가지 단계를 사용하여 라벨이 되지 않은 데이터를 최대한 활용하고 모델 가중치를 조정합니다. 첫 번째 단계는 마스킹된 언어 모델링(masked language modeling)을 사용하여 도메인 적응을 통해 라벨이 되지 않은 데이터를 활용하는 것이고, 두 번째 단계는 AL이 선택한 라벨이 된 데이터를 사용하여 모델 가중치를 추가로 조정하는 것입니다. 또한, BERT 기반의 임베딩을 BoW(Bag of Words), LSI(Latent Semantic Indexing), FastText 등 다른 텍스트 표현 패러다임과 비교하여 AL 프로세스의 중요한 두 단계인 인스턴스 선택(instance selection)과 분류(classification)에서 성능을 평가했습니다.

- **Performance Highlights**: 연구는 다양한 AL 예산(라벨이 된 인스턴스 수)과 인스턴스 수(약 5,000에서 300,000)로 구성된 8개의 ATC 벤치마크에서 DoTCAL의 탁월한 효과를 입증했습니다. 전통적인 한 단계 방법과 비교하여 라벨링 노력을 절반으로 줄이면서도 Macro-F1에서 최대 33%의 향상을 이루었습니다. 또한 여러 작업에서 BoW와 LSI가 특히 저예산 시나리오와 분류하기 어려운 작업에서 BERT보다 더 뛰어난 (최대 59%) 결과를 생성한다는 사실을 발견했습니다.



### scGHSOM: Hierarchical clustering and visualization of single-cell and CRISPR data using growing hierarchical SOM (https://arxiv.org/abs/2407.16984)
Comments:
          Abstract presentation at BIOKDD@ACM KDD 2024

- **What's New**: 새로운 제안은 고차원 단일 세포 데이터(high-dimensional single-cell data)의 생물학적 패턴 식별을 위해 GHSOM(Growing Hierarchical Self-Organizing Map)을 활용한 시각화를 제안합니다. 이는 단일 세포 시퀀싱(single-cell sequencing) 및 CRISPR 스크린과 같은 데이터 분석에 적합하도록 설계되었습니다. GHSOM은 샘플을 계층적 구조로 클러스터링(cluster)하여 군집 간 및 군집 내 변동을 만족하는 자가 성장 구조를 형성합니다. 또한, 군집을 구별짓는 특징을 식별하는 새로운 Significant Attributes Identification Algorithm을 제안합니다.

- **Technical Details**: GHSOM은 계층적 구조(hierarchical structure) 내에서 데이터 샘플을 군집화(clustering)하며, 자가 성장(self-growth) 구조를 통해 군집 간 및 군집 내 변동을 최적화합니다. Significant Attributes Identification Algorithm은 군집 내 변동이 최소이고, 군집 간 변동이 큰 특성을 식별합니다. 이들은 타겟 데이터 검색 및 추가 분석에 사용될 수 있습니다. 두 가지 혁신적인 시각화 도구도 소개되었습니다. Cluster Feature Map은 GHSOM 군집 구조 내에서 특정 특성의 분포를 강조하여 신속한 시각적 평가를 가능하게 합니다. Cluster Distribution Map은 GHSOM 그리드에서 리프 클러스터(leaf clusters)를 원으로 나타내어 군집 데이터 크기를 반영하고 세포 유형 또는 기타 속성을 시각화할 수 있습니다.

- **Performance Highlights**: 세 개의 단일 세포 데이터셋과 하나의 CRISPR 데이터셋(cell-gene database)에 분석을 적용하여 내부 및 외부 CH와 ARI 점수로 클러스터링 방법들을 평가했습니다. 내부 평가에서 GHSOM이 최고의 성능을 보였으며(CH=4.2), 외부 평가에서는 세 번째로 좋은 성능을 기록했습니다.



### Covering a Graph with Dense Subgraph Families, via Triangle-Rich Sets (https://arxiv.org/abs/2407.16850)
- **What's New**: 이번 논문에서는 새로운 정의인 RTR (Regularly Triangle-Rich) 패밀리를 사용하여 고밀도 부분 그래프를 효율적으로 발견하는 방법을 제안합니다. 이 방법은 특히 세 모서리 구조가 많고 부분 그래프의 크기에 비례하는 차수를 가지는 고밀도 그래프를 탐지하는데 적합합니다. 이를 통해 실제 응용에서는 최적의 밀도를 가진 단일 또는 소수의 부분 집합을 찾는 대신, 입력 그래프의 상당 부분을 포괄하는 많은 고밀도 부분 집합을 찾는 데 중점을 둡니다.

- **Technical Details**: 논문의 주요 기여는 RTRExtractor 알고리즘입니다. 이 알고리즘은 약간의 수학적 공식화와 함께 RTR 패밀리로 구성된 고밀도 부분 그래프를 효율적으로 발견합니다. 또한, 삼각형 개수를 활용한 커뮤니티 테스트와 군집화 (clustering)와 관련된 최근 결과를 참고하여 설계되었습니다. RTR 패밀리는 많은 삼각형을 포함하며 부분 그래프의 크기에 비례하는 차수를 가져야 합니다.

- **Performance Highlights**: RTRExtractor는 다양한 실제 데이터셋에서 우수한 성능을 보입니다. 수백만 개의 모서리를 가진 그래프도 수 분 내로 처리할 수 있으며, 높은 엣지 밀도를 가진 데이터셋에서 높은 커버리지를 달성합니다. 예를 들어, 1천만개 이상의 모서리를 가진 데이터셋에서도 전체 정점의 1/4 이상을 엣지 밀도 0.5 이상의 부분 그래프로 포괄하는 성과를 보였습니다. 인용 네트워크에서 RTRExtractor의 출력이 유사한 정점 집합과 의미 있게 상관됨을 보여주는 예시도 제시됩니다.




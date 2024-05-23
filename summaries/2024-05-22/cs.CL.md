### Aggregation of Reasoning: A Hierarchical Framework for Enhancing Answer Selection in Large Language Models (https://arxiv.org/abs/2405.12939)
Comments:
          17 pages, 14 figures, accepted by LREC-COLING 2024

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 복잡한 추론 작업을 향상시키기 위해 계층적 추론 집계 프레임워크인 AoR(Aggregation of Reasoning)을 도입했습니다. 기존의 샘플링 및 앙상블 기법은 정답이 소수인 경우 실패할 수 있는데, 이러한 제한을 극복하기 위해 각 추론 체인을 평가하여 답변을 선택하는 방식을 제안했습니다.

- **Technical Details**: AoR은 계층적 추론 집계(Hierarchical Reasoning Aggregation) 프레임워크로, 각 추론 체인의 평가를 바탕으로 답변을 선택합니다. 또한, 과제의 복잡성에 따라 필요한 추론 체인의 수를 동적으로 조정하는 동적 샘플링 기법(Dynamic Sampling)을 통합하여 추론 효율성을 높였습니다.

- **Performance Highlights**: 여러 복잡한 추론 작업에서 실시한 실험 결과, AoR은 기존의 주요 앙상블 방법들보다 우수한 성능을 보여주었습니다. 추가 분석을 통해 AoR이 다양한 언어 모델에 적응하며 현재 방법들보다 더 높은 성능 한계를 달성한다는 점도 확인할 수 있었습니다.



### Skin-in-the-Game: Decision Making via Multi-Stakeholder Alignment in LLMs (https://arxiv.org/abs/2405.12933)
Comments:
          ACL 2024, long paper

- **What's New**: 이번 연구는 도덕적 추론(moral reasoning)과 윤리적 의사 결정에서 어려움을 겪는 대형 언어 모델(LLMs)을 개선하기 위해 Skin-in-the-Game (SKIG) 프레임워크를 소개합니다. SKIG는 여러 이해관계자의 관점에서 의사 결정의 결과를 탐구하여 도덕적 추론을 향상시키는 것을 목표로 합니다.

- **Technical Details**: SKIG의 주요 메커니즘은 행동에 대한 책임(accountability)을 시뮬레이션하는 것입니다. 이와 함께 공감(empathy) 연습과 위험 평가(risk assessment)도 SKIG의 효과성에 중요한 역할을 합니다. SKIG는 독점 및 오픈 소스 LLM을 사용하여 다양한 도덕적 추론 벤치마크에서 그 성능을 검증했습니다. 또한, 광범위한 ablation 분석을 통해 그 핵심 구성 요소를 조사했습니다.

- **Performance Highlights**: SKIG 프레임워크는 다양한 도덕적 추론 벤치마크에서 기존 모델들에 비해 성능이 우수함을 확인했습니다. 특히, 여러 이해관계자의 관점을 통합하여 문제를 해결하는 데 있어 뛰어난 결과를 보였습니다.



### Code-mixed Sentiment and Hate-speech Prediction (https://arxiv.org/abs/2405.12929)
- **What's New**: 새로운 연구는 영어-힌디(English-Hindi)와 영어-슬로베니아어(English-Slovene) 쌍을 위한 새로운 이중언어(pre-trained) 마스크드 언어 모델을 개발하였으며, 이 모델들은 비공식적인 언어 사용을 지원하는 데 특히 적합합니다.

- **Technical Details**: 연구진은 초대형 언어 모델이 코드 혼합 상태(code-mixed settings)에서 어떻게 성능을 발휘하는지를 조사했습니다. 이를 위해 sentiment analysis(감정 분석)와 offensive language detection(공격성 언어 탐지)라는 두 가지 주요 과제를 선택했습니다. 이 연구는 단일 언어, 이중 언어, 몇 가지 언어, 그리고 거대 다중 언어(massively multilingual) 모델의 성능을 비교 평가했습니다.

- **Performance Highlights**: 최종 결과는 세부적으로 훈련된 이중 언어 모델과 소셜 미디어 텍스트에 특화된 다중 언어 모델이 가장 성공적인 분류기로 나타났으며, 비전문적으로 거대 다중 언어 및 단일 언어 모델이 그 뒤를 이었습니다. 흥미롭게도, 거대한 생성 모델은 경쟁력이 없었습니다. 이번 연구는 코드 혼합 데이터에 대해 모델들이 대부분 비코드 혼합 데이터보다 약간 더 나은 성능을 보였다는 것을 발견했습니다.



### G-DIG: Towards Gradient-based DIverse and hiGh-quality Instruction Data Selection for Machine Translation (https://arxiv.org/abs/2405.12915)
Comments:
          Accepted to ACL 2024 main conference

- **What's New**: 이번 연구에서는 머신 번역을 위해 자동으로 고품질 및 다양한 지도 데이터를 선택하는 새로운 기울기 기반 방법을 제안합니다. 이 방법은 모델 학습 중 각 훈련 예제가 모델에 미치는 영향을 분석하여 고품질 데이터를 선택합니다.

- **Technical Details**: 제안된 방법의 주요 혁신은 'Influence Function'(영향 함수)과 소규모 고품질 데이터셋을 사용해 모델에 유익한 영향을 미치는 훈련 예제를 고품질로 분류하는 것입니다. 또한, 데이터의 다양성을 높이기 위해 기울기에 기반한 클러스터링과 리샘플링을 통해 모델에 다양한 영향을 미치는 데이터를 선택합니다.

- **Performance Highlights**: WMT22와 FLORES 번역 작업에 대한 광범위한 실험에서 제안된 방법의 우수성이 입증되었으며, 심층 분석을 통해 이 방법의 효과성과 일반화 가능성도 검증되었습니다.



### Topic Modelling Case Law Using a Large Language Model and a New Taxonomy for UK Law: AI Insights into Summary Judgmen (https://arxiv.org/abs/2405.12910)
- **What's New**: 이 논문은 영국의 요약 판결 사례들을 주제로 새롭고 혁신적인 분류 체계를 개발 및 적용했습니다. 이 논문에서 Claude 3 Opus라는 대형 언어 모델(Large Language Model)을 사용하여 기능적 주제 및 트렌드를 탐구하였습니다. 중요한 점은 영국의 사례 법은 원래 키워드 또는 주제 필터링 옵션으로 라벨링 되어 있지 않아서, 이 연구는 요약 판결의 주제적 기반에 대한 이해를 정제하고, 전통적인 접근 방식과 AI 구동 접근 방식을 결합한 법률 분류의 잠재력을 보여주었다는 것입니다.

- **Technical Details**: 이 논문에서는 요약 판결 사례들의 큐레이션된 데이터셋을 사용했습니다. Claude 3 Opus 모델을 활용하여 다양한 법적 도메인에서 요약 판결의 적용에 관한 패턴을 분석했습니다. Claude 3 Opus 모델은 주제를 87.10%의 정확도로 올바르게 분류했습니다. 이로써 새로운 영국 법의 일반 분류 체계를 제공하며, 이는 사법 행정과 계산법 연구 방법론 분야에서 추가 연구 및 정책 논의를 위한 기초를 제공합니다.

- **Performance Highlights**: Claude 3 Opus 모델은 요약 판결 사례들의 주제를 87.10%의 정확도로 올바르게 분류하는 성능을 보였습니다. 이는 전통적인 접근 방식과 AI 기반 법률 분류 방법론을 결합하여 달성한 결과로, 법률 분석의 새로운 지평을 열 수 있는 잠재력을 보여줍니다.



### Adversarial DPO: Harnessing Harmful Data for Reducing Toxicity with Minimal Impact on Coherence and Evasiveness in Dialogue Agents (https://arxiv.org/abs/2405.12900)
Comments:
          15 pages, 7 figures, accepted to NAACL findings 2024

- **What's New**: 최근 연구에서는 개방영역(open-domain) 대화 시스템의 품질을 크게 향상시키는 고품질 대형 언어 모델(LLMs)과 다양한 효과적인 훈련 방법론이 소개되었습니다. 본 연구에서는 기존의 직접 선호 최적화(Direct Preference Optimization, DPO)를 개선한 '적대적 DPO(Adversarial DPO, ADPO)'라는 혁신적인 훈련 알고리즘을 소개합니다. ADPO 알고리즘은 유해한 대화를 최소화하면서 모델의 성능 저하를 줄이는 것을 목적으로 합니다.

- **Technical Details**: ADPO 알고리즘은 모델이 선호하는 응답에는 높은 확률 분포를, 유해한 응답에는 낮은 확률 분포를 부여하도록 훈련됩니다. 이 유해한 응답은 독성 통제 토큰(toxic control token)을 사용하여 모델 자체적으로 생성됩니다. ADPO는 전통적인 DPO에 비해 더 안정적인 훈련 절차를 제공하며, 인위적으로 안전한 대화 데이터를 생성할 필요성을 줄입니다.

- **Performance Highlights**: 연구 결과, ADPO는 모델이 유해한 대화에 대해 더욱 강력한 저항력을 보이는 동시에, 성능 저하를 최소화하는 것을 확인했습니다. 또한, ADPO는 전통적인 DPO보다 더 안정적인 훈련 과정(stable training procedure)을 제공합니다. 현재까지, 유해한 데이터를 생성 모델에 직접 통합한 최초의 DPO 알고리즘 적응 사례로 평가됩니다.



### Investigating Persuasion Techniques in Arabic: An Empirical Study Leveraging Large Language Models (https://arxiv.org/abs/2405.12884)
- **What's New**: 이 논문은 아랍어 소셜 미디어 콘텐츠에서 설득 기술을 식별하기 위한 포괄적 실증 연구를 제공합니다. 디지털 커뮤니케이션과 소셜 미디어의 확산으로 인해 올바른 정보를 구별하고 현명한 결정을 내리기 위해 이러한 기술에 대한 이해가 중요해졌습니다.

- **Technical Details**: 이 연구는 Pre-trained Language Models (PLMs)와 ArArEval 데이터셋을 활용하여 진행되었습니다. 연구는 두 가지 과제를 포함합니다. 첫째, 이분법적 분류를 통해 설득 기술의 존재 여부를 결정하는 과제와, 둘째, 다중 레이블 분류를 통해 텍스트에서 사용된 특정 설득 기술의 유형을 식별하는 과제입니다. 세 가지 학습 접근법을 탐구했으며, PLMs를 활용한 특징 추출, 미세 조정 (fine-tuning), 프롬프트 엔지니어링 (prompt engineering) 기술을 포함합니다.

- **Performance Highlights**: 미세 조정 (fine-tuning) 접근법이 가장 높은 성과를 보여, f1-micro 점수 0.865와 f1-weighted 점수 0.861을 달성했습니다. 또 다른 중요한 발견으로는 GPT 모델의 성능이 상대적으로 낮았지만, few-shot 학습 기술을 통해 성능을 최대 20%까지 개선할 수 있었습니다. 이는 향후 연구와 탐구에 유망한 방향을 제시합니다.



### Large Language Models Meet NLP: A Survey (https://arxiv.org/abs/2405.12819)
- **What's New**: 이번 연구는 LLMs (Large Language Models, 대형 언어 모델)이 NLP (Natural Language Processing, 자연어 처리) 작업에서 현재까지 얼마나 적용되었는지, 전통적인 NLP 작업이 LLMs으로 이미 해결되었는지, 그리고 LLMs의 미래 가능성에 대해 탐구하고 있습니다. 주로, 현재 LLMs가 NLP 작업에 어떻게 적용되고 있는지의 체계적인 조사를 제공하고 있습니다.

- **Technical Details**: 이 연구는 LLMs의 두 가지 주요 분류에 대해 논의합니다: (1) 파라미터를 동결시키는 응용 (parameter-frozen application)과 (2) 파라미터를 조정하는 응용 (parameter-tuning application). 이를 통해 LLMs가 NLP에 어떻게 적용되고 있으며, 그 과정에서의 진보를 이해할 수 있는 통합적인 관점을 제공합니다.

- **Performance Highlights**: 본 논문은 LLMs가 NLP에서 엄청난 가능성을 가지고 있지만 여전히 도전 과제들이 남아있다는 점을 강조합니다. 이 도전 과제들을 해결함으로써 앞으로의 발전을 도모하는 것을 목표로 하고 있습니다. 이를 통해 LLMs의 잠재력과 한계를 이해하고, 효과적인 LLMs를 구축하는 데 실용적인 가이드를 제시합니다.



### Comparing Neighbors Together Makes it Easy: Jointly Comparing Multiple Candidates for Efficient and Effective Retrieva (https://arxiv.org/abs/2405.12801)
- **What's New**: 새로운 연구는 CMC(Comparing Multiple Candidates) 프레임워크를 제안합니다. 이는 쿼리와 복수의 후보 임베딩(embeddings)을 얕은 자기 주의층(self-attention layers)을 통해 공동으로 비교하는 방법입니다. 비효율적인 교차 인코더(cross-encoder)에 의한 오류 전이를 방지하고 성능을 향상시키는 데 중점을 둡니다.

- **Technical Details**: CMC는 대규모 바이 인코더(bi-encoder)를 사용하여 다양한 후보를 검색한 후, 이러한 후보들의 임베딩을 얕은 자기 주의층으로 처리합니다. 이로 인해 컨텍스트화된 표현을 제공하면서도 동일 시간 내에 여러 후보를 비교할 수 있게 합니다. CMC는 2K 후보를 비교하는 데 단 2배의 시간만 필요로 하여 매우 확장성이 높습니다. 또한, CMC를 경량의 효과적인 리랭커(reranker)로 사용할 수 있으며, 기존 리트리버(retriever)와 결합하면 사실상 향상된 리트리버 역할을 할 수 있습니다.

- **Performance Highlights**: ZeSHEL 데이터셋에서 CMC는 리트리벌 단계에서 Recall@k를 크게 향상시켰습니다(R@16: +6.7, R@64: +3.5%). 또한 엔티티, 패시지, 대화 랭킹 등의 직접 리랭킹 실험에서 CMC는 크로스 인코더보다 11배 빠르고, 위키피디아 엔티티 링크(+0.7%-p)와 DSTC7 대화 랭킹(+3.3%-p)에서 더 나은 성능을 보였습니다.



### What Have We Achieved on Non-autoregressive Translation? (https://arxiv.org/abs/2405.12788)
Comments:
          ACL 2024 Findings

- **What's New**: 최근 비자동회귀(non-autoregressive, NAT) 번역 기술이 자율회귀(autoregressive, AT) 방법과 비교할만한 성능을 보이고 있습니다. 그러나, BLEU를 사용한 평가는 인간 주관 평가와의 상관성이 낮다는 문제가 있습니다. 이번 연구는 NAT와 AT의 성능을 여러 차원에서 체계적으로 비교하여 NAT가 AT에 얼마나 근접했는지에 대한 불확실성을 해소하고자 합니다.

- **Technical Details**: 네 가지 대표적인 NAT 방법을 다양한 차원에서 평가했으며, 여기에는 인간 평가도 포함되어 있습니다. 연구 결과, 최첨단 NAT가 성능 격차를 줄였음에도 불구하고, 보다 신뢰할 수 있는 평가 지표에서 여전히 AT보다 성능이 떨어짐이 확인되었습니다. 또한, 명시적으로 의존성을 모델링하는 것이 자연스러운 언어를 생성하고 분포 외 시퀀스를 일반화하는 데 중요하다는 것을 발견했습니다.

- **Performance Highlights**: NAT는 최근 큰 발전을 이루었지만, AT에 비해 여전히 성능 차이가 존재합니다. 특히, 인간 평가 및 기타 신뢰성 높은 평가 지표를 통해 확인된 결과에서는 NAT의 성능이 덜 자연스럽고 일반화 능력이 떨어짐을 확인할 수 있었습니다.



### The Echoes of Multilinguality: Tracing Cultural Value Shifts during LM Fine-tuning (https://arxiv.org/abs/2405.12744)
- **What's New**: 이 연구는 다국어 언어 모델(Multilingual Language Models, MLMs)이 다국어 학습 과정에서 문화적 가치를 어떻게 반영하고 변화시키는지 최초로 조사했습니다. 저자들은 fine-tuning 단계에서 데이터 소스와 언어에 따라 발생하는 가치 변화의 상호작용을 분석했습니다.

- **Technical Details**: 연구진은 fine-tuning 단계에 주목하여 새로운 언어적 경험을 통해 문화적 가치가 어떻게 답습되고 수정되는지 조사했습니다. 또한, 'training data attribution method'을 사용하여 fine-tuning 예제와 가치 변화를 유도하는 언어의 패턴을 발견했습니다.

- **Performance Highlights**: 이 연구는 다국어 언어 모델의 문화적 가치의 교차언어적 상호작용을 이해하는 데 중요한 기초 자료를 제공합니다. 이는 언어 모델이 커뮤니티의 편향에 더 민감하게 반응할 수 있도록 하는 데 도움이 되며, 다국어 콘텐츠 생성에서 효율성을 높일 수 있습니다.



### OLAPH: Improving Factuality in Biomedical Long-form Question Answering (https://arxiv.org/abs/2405.12701)
- **What's New**: 이번 논문에서는 의학 분야에서 대형 언어 모델(Large Language Models, LLMs)의 장문 생성 능력이 필요한 다양한 시나리오에 대해 다룹니다. 특히 환자의 질문에 대한 답변을 제공할 때 사실성을 유지하는 것이 중요합니다. 이를 위해 MedLFQA라는 벤치마크 데이터를 소개하며, 이는 생의학(biomedical) 관련 장문 질문-답변 데이터셋을 재구성하여 만들어졌습니다.

- **Technical Details**: MedLFQA를 통해 자동으로 사실성을 평가할 수 있는 방법을 제시했습니다. 또한 OLAPH라는 간단하면서도 새로운 프레임워크를 제안했는데, 이는 자동 평가를 통해 사실성을 개선할 수 있게 해줍니다. OLAPH 프레임워크는 샘플링 예측과 선호 최적화를 통해 반복적으로 학습하여 환각(hallucinations)을 줄이는 방식으로 LLMs를 훈련시킵니다. 구체적으로는, 최고의 점수를 받은 응답을 선호 응답(preferred response)으로 설정하고, 이를 기준으로 모델을 훈련시켜 사실성을 향상시킵니다.

- **Performance Highlights**: OLAPH 프레임워크로 훈련된 LLMs는 훈련에 사용되지 않은 평가 지표에서도 사실성이 크게 향상된 성능을 보였습니다. 특히, OLAPH 프레임워크로 훈련된 70억(7B) 파라미터 LLM은 사실성 측면에서 의학 전문가의 답변에 필적하는 장문 답변을 제공할 수 있음을 발견했습니다. 이는 의학 분야에서 LLMs의 장문 생성 능력을 평가하는 데 중요한 시사점을 제공합니다.



### Spotting AI's Touch: Identifying LLM-Paraphrased Spans in Tex (https://arxiv.org/abs/2405.12689)
Comments:
          ACL 2024 Findings

- **What's New**: AI-생성 텍스트 감지는 점점 더 주목받고 있는데, 이는 강력한 언어 모델들이 인간 수준의 텍스트 생성을 실현하고 있기 때문입니다. 이번 연구에서는 부분적으로 AI가 패러프레이즈(paraphrase)한 텍스트 감지에 중점을 둡니다. 흔히 텍스트 개선과 다양성을 위해 AI 패러프레이징이 사용됩니다. 이를 위해 새롭게 제안된 검출 프레임워크인 패러프레이즈 텍스트 스팬 감지(PTD)를 소개합니다. PTD는 텍스트 레벨이 아닌, 각 문장 별로 패러프레이징 정도를 점수로 매기는 방식입니다.

- **Technical Details**: 패러프레이즈 텍스트 스팬 감지(PTD)는 전체 텍스트를 입력으로 받아 개별 문장의 패러프레이징 정도를 점수로 매깁니다. 이를 위해 전용 데이터셋인 PASTED를 구성했습니다. 실험 결과 인-디스트리뷰션(in-distribution) 및 아웃-오브-디스트리뷰션(out-of-distribution) 모두에서 PTD 모델이 AI 패러프레이징 텍스트 스팬을 효과적으로 식별하는 것을 보여줍니다. 통계적 및 모델 분석을 통해 패러프레이즈된 텍스트 스팬 주변의 컨텍스트가 중요한 역할을 한다는 것을 설명합니다.

- **Performance Highlights**: 광범위한 실험을 통해 PTD 모델이 다양한 패러프레이징 프롬프트(prompt)와 여러 패러프레이징 텍스트 스팬에 대해 잘 일반화할 수 있음을 보여줍니다. 연구 자원은 공개되어 추가 연구에 활용될 수 있습니다.



### A Survey on Multi-modal Machine Translation: Tasks, Methods and Challenges (https://arxiv.org/abs/2405.12669)
- **What's New**: 최근 다중 모드 기계 번역(multi-modal machine translation)을 다룬 논문에서, 이전 연구들을 포괄적으로 정리하고 다양한 요인들이 모델 성능에 미치는 영향을 분석했습니다. 뿐만 아니라 미래 연구 방향에 대해서도 논의합니다. 이 논문은 다중 모드 기계 번역의 다양한 유형들을 다루어, 연구자들이 현재 상태를 더 잘 이해할 수 있도록 돕기 위한 포괄적인 개요를 제공합니다.

- **Technical Details**: 다중 모드 기계 번역은 텍스트 및 시각적 모달리티(visual modalities)를 입력으로 받아 소스 텍스트의 모호성을 해결하기 위해 시각적 맥락을 활용합니다. 본 논문은 99개의 이전 연구들을 바탕으로, 주요 모델들, 데이터셋, 평가 지표 관점에서 대표적인 연구들을 종합적으로 요약하였습니다.

- **Performance Highlights**: 다양한 요인들이 모델 성능에 미치는 영향을 분석함으로써 최적의 모델링 접근 방식에 대한 인사이트를 제공합니다. 논문은 초기 단계에 국한되지 않고 최근 등장한 다중 모드 기계 번역 유형들을 전반적으로 다루어, 연구자들에게 현재 상태에 대한 깊이 있는 이해를 돕고자 합니다.



### Retrieval-Augmented Language Model for Extreme Multi-Label Knowledge Graph Link Prediction (https://arxiv.org/abs/2405.12656)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)이 가진 두 가지 주요 문제인 환각(hallucination)과 높은 훈련 비용을 해결하고자 새로운 접근 방식을 소개합니다. 특히, 구조화된 실세계 지식을 사용하여 여러 응답을 예측할 수 있는 극한 멀티 라벨 지식 그래프 링크 예측 작업(extreme multi-label KG link prediction task)을 제안하였습니다.

- **Technical Details**: 제안된 프레임워크는 지식 그래프(KG)의 개체(entity), 관계(relation), 텍스트 데이터를 함께 고려하여 적절한 1-홉 이웃을 식별할 수 있는 검색기(retriever)를 포함하고 있습니다. 실험을 통해 지식 그래프의 상이한 특성에 따라 다른 보강 방식을 적용하는 것이 필요하다는 것을 확인하였으며, 텍스트 데이터로 언어 모델의 입력을 보강하는 것이 성능을 크게 향상시킴을 보였습니다.

- **Performance Highlights**: 소규모 파라미터 크기를 가진 본 프레임워크는 주어진 지식 그래프를 기반으로 한 외삽(extrapolation)을 효과적으로 수행할 수 있습니다. 코드 및 추가 자료는 GitHub에서 확인할 수 있습니다.



### Exploration of Masked and Causal Language Modelling for Text Generation (https://arxiv.org/abs/2405.12630)
Comments:
          working paper

- **What's New**: 대형 언어 모델(LLMs)은 자연어 처리(NLP) 분야에서 혁신을 일으켰으며, 거의 모든 작업에서 최첨단 성능을 달성했습니다. 하지만 기존의 텍스트 생성 방식인 순차 언어 모델링(CLM)은 왼쪽에서 오른쪽으로 순차적으로 텍스트를 생성하므로 모델의 자유도를 제한합니다. 반면에 주로 언어 이해 작업에 사용되는 마스크드 언어 모델링(MLM)은 텍스트의 어디에서나 어떤 순서로든 토큰을 생성할 수 있습니다. 이 논문은 텍스트 생성 작업에 대해 MLM과 CLM 접근 방식을 광범위하게 비교합니다.

- **Technical Details**: 세 가지 다른 데이터셋(의료 퇴원 요약, 영화 줄거리 요약, 저자 확인 데이터셋)에서 유사한 크기의 여러 언어 모델을 사전 훈련했습니다. 생성된 텍스트의 품질을 평가하기 위해 정량적 지표를 사용하고, 텍스트의 일관성과 문법적 정확성을 분석하기 위해 질적 인간 평가도 수행했습니다. 또한, 세 가지 다운스트림 작업(엔티티 인식, 텍스트 분류, 저자 확인)에서 생성된 텍스트의 유용성을 평가했습니다.

- **Performance Highlights**: 모든 데이터셋에서 MLM이 CLM보다 텍스트 생성에서 일관되게 우수했으며, 더 높은 정량적 점수와 텍스트 일관성을 보였습니다. 생성된 텍스트의 품질과 다운스트림 작업에서 모델의 성능 사이에 강한 상관관계는 발견되지 않았습니다. 이 연구는 MLM이 텍스트 생성에서 미래 연구에 큰 잠재력을 가지고 있음을 보여줍니다.



### MentalQA: An Annotated Arabic Corpus for Questions and Answers of Mental Healthcar (https://arxiv.org/abs/2405.12619)
Comments:
          Ongoing (under-review), 10 pages, 7 figures, 5 tables

- **What's New**: 이번 연구에서는 MentalQA라는 새로운 아랍어 데이터셋을 소개합니다. 이 데이터셋은 대화 형식의 질문-응답(QA) 상호작용으로 구성되며, 정신 건강 관련 텍스트 마이닝 도구 개발을 위한 자원 부족 문제를 해결하고자 합니다.

- **Technical Details**: 데이터 품질을 보장하기 위해 엄격한 주석(annotation) 과정을 거쳤습니다. 주석 스키마는 기존의 분류 체계를 기반으로 일부 수정되었습니다. 질문 유형은 진단, 치료, 해부학 및 생리학, 역학, 건강한 생활 습관, 제공자 선택의 6가지 범주로 나뉩니다. 답변 전략은 정보 제공, 직접적인 지침, 정서적 지원을 포함합니다. 세 명의 경험 많은 주석자가 데이터 일관성을 위해 협력하였습니다. 질문 유형에 대한 Fleiss' Kappa는 0.61, 답변 전략에 대한 Fleiss' Kappa는 0.98로 나타났습니다.

- **Performance Highlights**: 심층 분석 결과, 연령대별 질문 선호도에 차이가 있다는 흥미로운 패턴이 발견되었으며, 질문 유형과 답변 전략 간의 강한 상관관계가 드러났습니다. MentalQA는 정신 건강 전문가와 정보를 찾는 개인들을 지원할 수 있는 아랍어 텍스트 마이닝 도구 개발에 귀중한 기반을 제공합니다.



### Quantifying Emergence in Large Language Models (https://arxiv.org/abs/2405.12617)
- **What's New**: 최근 대형 언어 모델(LLMs)의 '지능적' 행동, 즉 발현(emergence)을 정량화하기 위한 새로운 방법이 제안되었습니다. 기존에는 넓은 데이터셋과 다양한 작업을 통해 통계적으로 추정되었으나, 자원 소모가 크고 해석이 어렵다는 문제가 있었습니다. 이번 연구에서는 동적 발현주의(emergentism in dynamics)에서 영감을 받아, entropy reduction을 사용하여 발현의 강도를 평가하는 새로운 접근법을 제안합니다.

- **Technical Details**: 연구진은 transformer block 내부 표현에서 파생된 거시적(semantic) 수준과 미시적(token) 수준의 엔트로피(entropy) 감소를 비교하여 발현의 강도를 측정했습니다. 저비용 추정기를 사용하여 이 접근법을 적용했으며, GPT-2와 같은 대형 언어 모델과 GEMMA 등의 다양한 모델에서 일관된 행동을 확인했습니다. 특히, 맥락 학습(in-context learning, ICL)과 자연 문장에서의 평가를 진행했습니다.

- **Performance Highlights**: (1) 제안된 방법은 기존 성능 지표에 기반한 관찰과 일치하는 일관된 측정을 제공하며, 발현 정량화의 유효성을 검증했습니다. (2) 제안된 지표는 ICL에서 '샷'(shots) 수와의 상관관계를 포함한 새롭고 독특한 발현 패턴을 발견했으며, 이는 LLMs의 환각 현상을 해석하는 새로운 방식을 제시합니다. (3) GPT-2와 같은 소형 모델을 통해 더 큰 폐쇄 자원 LLM의 발현 예측에 대한 잠재적 솔루션을 제공했습니다. 코드와 관련 자료는 공개된 URL을 통해 확인할 수 있습니다.



### Tagengo: A Multilingual Chat Datas (https://arxiv.org/abs/2405.12612)
- **What's New**: 최근 오픈소스 대형 언어 모델(LLMs) 영역에서 큰 발전이 있었습니다. 그러나 대부분의 모델은 인기가 많은 몇몇 언어에만 집중하고 있습니다. 이에 우리는 74개의 언어로 구성된 70,000개 이상의 프롬프트-응답 쌍을 포함하는 고품질 데이터셋을 소개합니다. 이 데이터셋은 인간이 생성한 프롬프트와 합성된 응답으로 구성되어 있습니다. 이 데이터를 이용해 최첨단 오픈소스 영어 LLM을 다국어 채팅 환경에서 학습시켰습니다.

- **Technical Details**: 이 연구에서는 인간이 생성한 프롬프트와 합성된 응답으로 구성된 74개의 언어로 된 70,000개 이상의 프롬프트-응답 쌍 데이터셋을 사용했습니다. 이러한 데이터를 통해 다국어 채팅이 가능한 최첨단 오픈소스 영어 LLM을 학습시켰습니다. 특히, MT-Bench 채팅 벤치마크에서 6개의 언어로 모델을 평가한 결과, 다국어 모델이 각 언어별로 기존의 오픈소스 LLM들을 능가하는 성과를 보였습니다.

- **Performance Highlights**: 우리의 다국어 모델은 MT-Bench 채팅 벤치마크에서 6개 언어에 대해 이전 최첨단 오픈소스 LLM들을 능가하는 성과를 거뒀습니다. 특히, 일본어를 목표 언어로 삼았을 때, 단순히 해당 언어로만 된 데이터로 학습하는 것보다 다국어 데이터를 더 많이 학습하는 것이 성능 향상에 유리하다는 것이 확인되었습니다.



### Tiny Refinements Elicit Resilience: Toward Efficient Prefix-Model Against LLM Red-Teaming (https://arxiv.org/abs/2405.12604)
Comments:
          Preprint, 10 pages main with 10 pages appendix

- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 안전성과 견고성을 개선하는 방어 전략이 부족하다는 문제를 해결하기 위해, 	extbf{Sentinel} 모델을 소개합니다. 이 모델은 플러그 앤 플레이(prefix module) 형식으로 설계되어, 추가적인 30개 미만의 토큰으로 입력 프롬프트를 재구성함으로써 목표 LLM의 응답에서 독성을 효과적으로 줄입니다.

- **Technical Details**: Sentinel 모델은 대형 목표 모델의 파라미터 비효율성(parameter inefficiency)과 제한된 모델 접근성(limited model accessibility)을 자연스럽게 극복합니다. 우리는 Proximal Policy Optimization (PPO)를 사용한 인터리브(interleaved) 훈련 체계를 채택하여, 레드 팀(red team)과 Sentinel 모델 모두를 동적으로 최적화합니다. 이 과정에서 다중 에이전트 중심 평론가(multi-agent centralized critic)에서 영감을 얻은 가치 헤드 공유(value head-sharing) 메커니즘을 도입하여 에이전트 간의 복잡한 상호작용을 관리합니다.

- **Performance Highlights**: 광범위한 실험을 통해, 텍스트-텍스트(text-to-text)와 텍스트-이미지(text-to-image)에서 우리 접근법의 효용성을 입증했습니다. 특히 	exttt{Llama-2}, 	exttt{GPT-3.5}, 	exttt{Stable-Diffusion}와 같은 대형 모델을 다룰 때에도 독성 출력 감소 효과를 확인했으며, 다양한 응용 분야에서 안전성과 견고성을 향상시키는 데 있어 우리의 프레임워크가 잠재력을 가지고 있음을 강조했습니다.



### Unlocking Data-free Low-bit Quantization with Matrix Decomposition for KV Cache Compression (https://arxiv.org/abs/2405.12591)
Comments:
          11 pages, 6 figures

- **What's New**: 이 논문에서는 	extbf{DecoQuant}이라는 새로운 데이터가 필요 없는 저비트해상도 양자화(quantization) 기술을 소개합니다. 이 기술은 텐서 분해 방식(tensor decomposition methods)을 사용하여 KV 캐시의 크기를 효과적으로 압축하는 데 초점을 맞추고 있습니다. 기존 방법론들과 달리, DecoQuant는 정밀도를 저하시키거나 추가적인 데이터 칼리브레이션(calibration)을 필요로 하지 않습니다.

- **Technical Details**: DecoQuant의 핵심 아이디어는 원래 행렬의 이상치(outlier) 분포를 조정하여, 양자화의 어려움을 분해된 지역 텐서(local tensors)로 이전하는 것에 있습니다. 구체적으로, 이상치들이 주로 작은 지역 텐서에 집중되어 있으며, 큰 텐서는 보다 좁은 값 범위를 가지는 것을 발견했습니다. 이러한 발견을 바탕으로, 큰 텐서에 대해서는 저비트 양자화를 적용하고 작은 텐서에 대해서는 고정밀 표현을 유지하도록 제안합니다. 또한, DecoQuant를 위한 효율적인 디양자화(dequantization) 커널을 개발하여 고속 추론을 지원합니다.

- **Performance Highlights**: 광범위한 실험 결과, DecoQuant는 최대 약 75%의 메모리 사용량 감소를 달성하면서 비슷한 수준의 생성 품질을 유지하는 것으로 나타났습니다. 이를 통해 LLMs의 추론 속도와 메모리 효율성을 크게 향상시킬 수 있음을 입증합니다.



### Mining the Explainability and Generalization: Fact Verification Based on Self-Instruction (https://arxiv.org/abs/2405.12579)
- **What's New**: 상업적 대규모 언어 모델(LLMs)을 기반으로 한 팩트체크(fact-checking)가 주류가 되었습니다. 본 논문에서는 정확도와 설명 가능성의 균형을 맞추기 위해 자기 지시 기반(fine-tuning) 접근 방식을 제안합니다. 이 방법은 데이터 증강(Data Augmentation)과 개선된 DPO 파인튜닝(Improved DPO fine-tuning)을 포함합니다.

- **Technical Details**: 먼저 데이터 증강 단계에서는 모델로 하여금 주장-증거 쌍(claim-evidence pairs)과 라벨을 기반으로 긍정적 및 부정적 설명을 생성하도록 한 후, 우리만의 난이도 기준에 따라 데이터셋을 샘플링(sample)합니다. 그 다음, 개선된 DPO를 사용하여 생성된 샘플로 모델을 파인튜닝(fine-tuning)합니다. 이 방식으로 LLaMA-7B 모델을 테스트하였으며, 기존 방식과 비교하여 높은 정확도와 설명 가능성을 유지할 수 있음을 보여주었습니다.

- **Performance Highlights**: 우리의 접근 방식은 기존의 파인튜닝 방법과 비교하여 정확도를 유지하거나 이를 능가할 뿐만 아니라 유창한 설명 텍스트도 생성할 수 있습니다. 뿐만 아니라 높은 일반화 성능까지 보유하고 있습니다. 이는 팩트체크를 위해 자기 지도 학습(self-supervised learning)을 활용한 최초의 방법이며, 대조 학습(contrastive learning)과 개선된 DPO를 결합한 혁신적인 방법입니다.



### PyramidInfer: Pyramid KV Cache Compression for High-throughput LLM Inferenc (https://arxiv.org/abs/2405.12532)
Comments:
          Accepted by ACL 2024

- **What's New**: 대형 언어 모델(LLMs)은 뛰어난 이해 능력을 보이지만, 추론 시 GPU 메모리 사용량이 높아 실시간 애플리케이션, 예를 들어 챗봇 등에서는 확장성이 떨어지는 문제가 있습니다. 이를 개선하기 위해 PyramidInfer라는 새로운 방법을 제안합니다. PyramidInfer는 층별로 중요한 컨텍스트를 유지하며 KV 캐시(KV cache)를 압축합니다. 이를 통해 적은 메모리로 더 높은 성능을 유지하는 것이 가능합니다.

- **Technical Details**: PyramidInfer는 층별로 중요한 키(key)와 값(value)을 유지하여 KV 캐시를 압축합니다. 기존 방법들은 사전 계산된 KV 캐시를 축소하여 메모리를 줄이는 데 집중했지만, 계층 간 종속성 및 사전 계산에 의한 메모리 대량 소비를 간과했습니다. 반면 PyramidInfer는 주의(attention) 가중치의 일관성을 바탕으로 중요한 키와 값을 추출하고, 이를 통해 중요한 컨텍스트만 유지하는 방식으로 메모리를 효율적으로 사용합니다.

- **Performance Highlights**: PyramidInfer는 기존 Accelerate 방법에 비해 2.2배의 처리량을 증가시키며, KV 캐시에 대한 GPU 메모리 사용량을 54% 이상 줄이는 데 성공했습니다.



### SirLLM: Streaming Infinite Retentive LLM (https://arxiv.org/abs/2405.12528)
- **What's New**: 대형 언어 모델(LLMs)이 다양한 분야에서 점점 더 많이 사용됨에 따라, 긴 텍스트 입력을 처리하고 기억을 유지하는 능력이 중요해지고 있습니다. SirLLM(Streaming Infinite Retentive LLM)은 이러한 과제를 해결하기 위해 도입되었습니다. SirLLM은 파인튜닝(fine-tuning) 없이도 무한 길이 대화 동안 더 긴 기억을 유지할 수 있게 해줍니다.

- **Technical Details**: SirLLM은 Token Entropy 메트릭과 메모리 감쇠 메커니즘(memory decay mechanism)을 사용하여 중요한 구문을 필터링하고, 이는 LLM에 오래 지속되고 유연한 기억을 부여합니다. 세 가지 작업(DailyDialog, Grocery Shopping, Rock-Paper-Scissors)을 설계하고 세 가지 데이터셋을 구성하여 SirLLM의 효율성을 다양한 각도에서 측정했습니다.

- **Performance Highlights**: 실험 결과, SirLLM은 다양한 LLM 및 작업에 걸쳐 안정적이고 큰 개선을 달성했습니다. 이는 SirLLM의 효율성을 강력하게 입증합니다. 이를 통해 '신사(사일)는 자신을 잊을 수 있지만, SirLLM은 절대 그렇지 않다'는 사실을 보여줍니다.



### Sparse Autoencoders Enable Scalable and Reliable Circuit Identification in Language Models (https://arxiv.org/abs/2405.12522)
- **What's New**: 이 논문은 대형 언어 모델 (large language models)에서 해석 가능한 회로를 발견하기 위한 효율적이고 견고한 방법을 소개합니다. 이 방법은 기존 기술의 계산 복잡도와 하이퍼파라미터 민감도와 같은 주요 한계를 해결합니다. 제안된 방법은 sparse autoencoders를 사용하여 고안된 positive 예제와 negative 예제를 학습하는 것입니다. 모델이 positive 예제에서만 다음 토큰을 정확하게 예측할 수 있도록 하는 방식입니다.

- **Technical Details**: 저자들은 attention head 출력의 학습된 표현이 특정 계산에 engage된 head를 signal 할 것이라고 가정합니다. 이러한 학습된 표현을 정수 코드로 이산화(discretise)하고, 각 head에 대한 positive 예제에 고유한 코드 간의 중첩을 측정함으로써, 비싼 절제(ablations)나 아키텍처 수정 없이 회로에 관여하는 attention heads를 직접 식별할 수 있습니다. 제안된 방법은 세 가지 잘 연구된 작업 - 간접 객체 식별, 비교(>), 단락 완성 -에서 ground-truth 회로를 복구하는 데 있어 최첨단 기준보다 높은 정밀도와 재현율을 달성했습니다. 또한, 실행 시간을 몇 시간에서 몇 초로 단축했습니다. 각 작업에 대해 단 5-10개의 텍스트 예제만으로도 견고한 표현을 학습할 수 있다는 점이 특징입니다.

- **Performance Highlights**: 제안된 메소드는 간접 객체 식별, 비교(>), 단락 완성과 같은 세 가지 작업에서 최첨단 baselines보다 높은 정밀도(precision)와 재현율(recall)을 달성했습니다. 더불어, 실행 시간을 몇 시간에서 몇 초로 크게 단축시켰으며, 각 작업에 대해 단 5-10개의 텍스트 예제만으로도 견고한 표현을 학습할 수 있음을 입증했습니다.



### Leveraging Diverse Data Generation for Adaptable Zero-Shot Dialogue State Tracking (https://arxiv.org/abs/2405.12468)
- **What's New**: 본 연구에서는 제로샷 다이얼로그 상태 추적(DST) 정확도를 크게 향상시킬 수 있는 새로운 방법을 제안하였습니다. 기존의 DST 데이터는 수집 비용이 높아 제한된 도메인 및 슬롯 타입만을 다루는 한계를 가졌습니다. 이 연구는 완전 자동화된 데이터 생성 기법을 통해 다양한 합성 데이터를 사용하여 이러한 한계를 극복합니다. 새로운 데이터 생성 접근 방식은 새로운 애플리케이션 도메인을 만들어 대화와 함께 실버 대화 상태 주석과 슬롯 설명을 포함시킵니다.

- **Technical Details**: 본 접근 방식은 합성 제로샷 DST 훈련 데이터를 만들기 위해 새로운 도메인을 완전히 생성합니다. 이는 기존의 DST 데이터 생성 방법과 달리 완전히 새로운 애플리케이션 도메인에서 대화를 생성해냅니다. 생성된 대화 데이터에는 실버 대화 상태 주석과 슬롯 설명이 포함되어 있어 다양성과 정확도를 높이는데 기여합니다. 이 방법으로 D0T 데이터셋이 만들어졌으며, 이는 1,000개 이상의 도메인을 포함하고 있습니다.

- **Performance Highlights**: 다양한 합성 데이터를 기반으로 모델을 훈련한 결과, MultiWOZ 벤치마크에서 +6.7%의 Joint Goal Accuracy 향상을 보였습니다. 이는 더 큰 모델과 경쟁할 수 있는 수준의 성과입니다.



### Resolving Word Vagueness with Scenario-guided Adapter for Natural Language Inferenc (https://arxiv.org/abs/2405.12434)
Comments:
          IJCAI24

- **What's New**: 이 논문에서는 기존의 자연어 추론(NLI) 모델들이 독립적인 문장의 의미 정보에만 의존하여 시각적 정보가 부족한 문제를 해결하기 위해 ScenaFuse 어댑터를 제안합니다. 이 어댑터는 대규모 사전 학습된 언어 지식과 관련된 시각적 정보를 통합하여 NLI 작업의 성능을 극대화합니다.

- **Technical Details**: ScenaFuse는 두 가지 핵심 모듈로 구성됩니다. 첫째, 이미지-문장 상호작용 모듈(image-sentence interaction module)을 설계하여 사전 학습된 모델의 어텐션 메커니즘에 시각 정보를 포함시킵니다. 둘째, 이미지-문장 융합 모듈(image-sentence fusion module)을 통해 이미지의 시각 정보와 문장의 의미 정보를 적응적으로 결합합니다. 이를 통해 언어와 시각 정보를 종합적으로 활용하여 NLI 작업에서의 이해와 추론 능력을 향상시킵니다.

- **Performance Highlights**: 광범위한 벤치마크 실험 결과, 제안한 ScenaFuse가 일관되게 NLI 성능을 향상시킴을 입증했습니다.



### Targeted Multilingual Adaptation for Low-resource Language Families (https://arxiv.org/abs/2405.12413)
- **What's New**: 새로운 연구에서는 다중언어 모델의 '거대 다언어(multilingual)' 학습이 특정 언어에 대한 유용성을 제한하고, 특히 저자원 언어에서 성능이 저조하다는 점을 지적합니다. 이 문제를 해결하기 위해 모델을 관련된 언어들로 훈련시키는 '목표 다언어(targeted multilinguality)' 접근 방식을 제안하고, 이를 철저히 테스트합니다. 연구는 우랄어(Uralic) 계열 언어들을 대상으로 XLM-R 모델을 다양한 설정으로 적응시켰습니다.

- **Technical Details**: 실험에서는 XLM-R 모델을 우랄어 계열의 15개 언어로 적응시키기 위해 다양한 설정을 사용했습니다. 두 가지 downstream 태스크와 11개의 평가 언어에서 성능을 평가하였습니다. 설정별로 하이퍼파라미터의 효과를 회귀 분석한 결과, 저자원 언어에서는 적응된 vocabulary 크기가 상대적으로 중요하지 않다는 것을 발견했습니다. 또한, 저자원 언어를 학습할 때 높은 자원 언어의 성능에 거의 영향을 주지 않으면서 저자원 언어의 데이터를 보다 많이 사용할 수 있다는 것을 발견했습니다.

- **Performance Highlights**: 적응된 모델들이 단일언어(monolingual)와 다언어(multilingual) 기반 모델들보다 훨씬 뛰어난 성능을 보였습니다. 특히, 저자원 언어들에서의 성능 향상이 두드러졌습니다. 이번 연구는 목표 다언어 설정에서 언어를 적응시키기 위한 새로운 모범 사례를 제시합니다.



### Question-Based Retrieval using Atomic Units for Enterprise RAG (https://arxiv.org/abs/2405.12363)
Comments:
          10 pages, 2 figures, 3 tables

- **What's New**: 새로운 연구에서는 엔터프라이즈 검색 보강 생성 (Enterprise retrieval augmented generation, RAG) 프레임워크에서 보다 정확한 청크(chunks) 검색을 위한 제로샷(Zero-shot) 적응 방법을 제안합니다. 이 접근법은 기존의 조밀 검색(dense retrieval) 단계를 개선하여, 사용자 쿼리에 대해 높은 리콜(recall)을 달성할 수 있도록 합니다.

- **Technical Details**: 제안된 방법은 청크를 원자적 진술(atoms)로 분해한 후, 이 원자들에 대해 합성된 질문(synthetic questions)을 생성합니다(청크를 문맥으로 사용). 조밀 검색 과정에서는 원자에 대한 합성 질문과 관련된 청크를 사용자 쿼리에 가장 가까운 셋으로 찾습니다. 원자 단위로 검색했을 때, 전체 청크를 이용한 검색보다 높은 리콜을 달성했으며, 특히 원자를 기반으로 생성된 합성 질문을 사용한 검색 시 성능이 더욱 향상되었습니다.

- **Performance Highlights**: 이 연구는 청크를 더 작은 원자로 분해하고 합성 질문을 사용하여 검색 단계에서 더 높은 리콜을 달성할 수 있음을 보여주었습니다. 이는 엔터프라이즈 LLM의 RAG 파이프라인 성능을 향상시키는 데 기여합니다.



### Reducing Transformer Key-Value Cache Size with Cross-Layer Attention (https://arxiv.org/abs/2405.12981)
- **What's New**: 이번 논문에서 소개된 새로운 기술은 Cross-Layer Attention (CLA)입니다. CLA는 다중-쿼리 어텐션 (Multi-Query Attention: MQA)의 개념을 확장하여, 인접 레이어 간에 키(key)와 값(value) 헤드를 공유함으로써 KV 캐시(KV Cache)의 크기를 추가로 2배 더 줄일 수 있는 방법을 제안합니다. 이를 통해 거의 동일한 정확도를 유지하면서도 메모리 사용량을 크게 줄일 수 있습니다.

- **Technical Details**: CLA는 Transformer 기반의 대규모 언어 모델(LLM)에서 디코딩 속도를 높이는 데 필수적인 KV 캐시를 최적화하는 기법입니다. 기존의 MQA와 GQA(Grouped-Query Attention) 방법은 다수의 쿼리 헤드가 단일 키/밸류 헤드를 공유하도록 설계하여, 키/밸류 헤드의 수를 크게 줄였습니다. CLA는 이 아이디어를 확장하여 인접한 레이어 간에서도 키/밸류 헤드를 공유할 수 있도록 했습니다.

- **Performance Highlights**: 1B 및 3B 파라미터 모델을 처음부터 학습시키는 실험에서 CLA는 기존의 MQA가 제공하는 메모리/정확도 트레이드오프보다 더 나은 Pareto 개선을 보여주었습니다. 이를 통해 더 긴 시퀀스 길이와 더 큰 배치 사이즈로 추론(inference)을 수행할 수 있게 되었습니다.



### Diffusion-RSCC: Diffusion Probabilistic Model for Change Captioning in Remote Sensing Images (https://arxiv.org/abs/2405.12875)
- **What's New**: 이 논문에서는 이시적(remote sensing) 이미지 변화 설명(RSICC)을 위한 새로운 접근 방식을 소개합니다. 이 기법은 바이타임(bi-temporal) 원격 감지 이미지 쌍의 의미적 변화를 설명하는 인간 같은 언어를 생성하는 것을 목표로 합니다. 이를 통해 환경 역학 및 토지 관리에 중요한 통찰력을 제공합니다.

- **Technical Details**: 기존의 변화 설명 과제와 달리, RSICC는 서로 다른 모달리티(modality)간의 관련 정보를 검색하고 유창한 설명을 생성하는 것뿐만 아니라, 장기간에 걸친 픽셀 수준의 차이를 완화하여 지형 변화 위치를 정확히 찾아내야 합니다. 이번 연구에서는 확산 모델(diffusion model)의 뛰어난 생성 능력에 영감을 받아 RSICC 문제를 해결하기 위해 확률적 확산 모델(probabilistic diffusion model)을 제안합니다. 학습 과정에서, 우리는 노이즈 예측기를 설계하여 교차 모달 특징(cross-modal features)으로 조건화하여 실제 설명 분포에서 표준 가우시안 분포로의 분포를 배우도록 했습니다. 또한, 노이즈 예측기를 역 과정에서 사용하기 위해 교차 모드 융합(cross-mode fusion)과 적층 자기 주의(stacking self-attention) 모듈을 설계하였습니다.

- **Performance Highlights**: LEVIR-CC 데이터셋에서의 광범위한 실험 결과는 Diffusion-RSCC와 그 구성 요소의 효과성을 입증하였습니다. 정량적 결과는 기존 방법들에 비해 새로 증강된 지표들에서도 우수한 성능을 보여주었습니다. 코드와 관련 자료는 온라인에서 제공될 예정입니다.



### LLM Processes: Numerical Predictive Distributions Conditioned on Natural Languag (https://arxiv.org/abs/2405.12856)
- **What's New**: 이번 연구는 사용자의 사전 지식을 자연어 텍스트를 통해 통합하고, 이를 통해 임의의 위치에서 수치 데이터를 처리하고 확률 예측을 수행할 수 있는 회귀 모델을 구축하고자 합니다. 이를 위해 대규모 언어 모델(Large Language Models, LLMs)을 활용하여 사용자가 자연어로 전문가의 통찰을 통합할 수 있는 인터페이스를 제공하며, 사용자가 직접 가지고 있지 않을 수도 있는 문제 관련 지식을 활용할 수 있는 기회를 제공합니다.

- **Technical Details**: 연구진은 LLMs로부터 명시적이고 일관된 수치 예측 분포를 도출하는 전략을 탐구했습니다. 이 공동 예측 분포를 'LLM Processes'라고 부르며, 예측, 다차원 회귀, 블랙박스 최적화, 이미지 모델링과 같은 설정에서 임의의 양에 대해 사용합니다. 논문의 주요 초점은 일관된 예측 분포를 유도하기 위한 프롬프트(prompting)의 실질적인 세부 사항을 조사하고, 회귀에서의 효과를 입증하는 것입니다.

- **Performance Highlights**: 연구는 텍스트를 수치 예측에 유용하게 통합하는 능력을 입증했으며, 이를 통해 예측 성능을 향상시키고 정성적 설명을 반영한 정량적 구조를 제공합니다. 이를 통해 LLMs가 암시적으로 인코딩한 풍부한 가설 공간을 탐구할 수 있는 첫 걸음을 내딛습니다.



### Unsupervised Multimodal Clustering for Semantics Discovery in Multimodal Utterances (https://arxiv.org/abs/2405.12775)
Comments:
          Accepted by ACL 2024, Main Conference, Long Paper

- **What's New**: 본 논문에서는 새로운 비지도 멀티모달 클러스터링 기법(UMC)을 소개합니다. 기존 방법이 비언어적 정보를 활용해 복잡한 의미를 파악하는 데 한계가 있는 비지도 시나리오에서, UMC는 멀티모달 데이터의 증강 뷰를 구성하는 독창적인 접근법을 도입합니다. 이를 통해 사전 훈련을 수행하고 이후 클러스터링을 위한 잘 초기화된 표현(representations)을 설정합니다.

- **Technical Details**: UMC는 각 샘플의 최근접 이웃의 밀도로 측정된 고품질 샘플을 동적으로 선택하여 표현 학습의 가이던스로 사용합니다. 또한, 각 클러스터에서 샘플 선택을 개선하기 위해 최적의 top-$K$ 매개변수 값을 자동으로 결정합니다. 마지막으로 고품질 및 저품질 샘플을 모두 사용해 효과적인 클러스터링을 가능하게 하는 표현을 학습합니다.

- **Performance Highlights**: UMC는 벤치마크 멀티모달 의도(intent) 및 대화 행위(dialogue act) 데이터셋에서 최신 방법들에 비해 클러스터링 측정 지표에서 2-6%의 주목할 만한 성능 향상을 보여줍니다. 이는 이 분야에서의 첫 번째 성공적인 시도입니다.



### RecGPT: Generative Pre-training for Text-based Recommendation (https://arxiv.org/abs/2405.12715)
Comments:
          Accepted to the ACL 2024 main conference

- **What's New**: 이번 연구에서는 텍스트 기반 추천 시스템을 위한 첫 도메인 적응 및 완전 훈련된 대형 언어 모델 RecGPT-7B와 그 명령-따르기 변형(variant)인 RecGPT-7B-Instruct를 소개합니다. 이는 추천 시스템 분야에서 중요한 발전을 나타냅니다.

- **Technical Details**: RecGPT-7B와 RecGPT-7B-Instruct 모델은 각각 도메인 적응을 통해 최적화되었습니다. 이 연구에서는 평가 예측(rating prediction)과 순차적 추천(sequential recommendation) 작업에서의 실험 결과를 제시하며, RecGPT-7B-Instruct 모델이 이전의 강력한 기준선(baseline)들을 능가함을 확인했습니다.

- **Performance Highlights**: RecGPT-7B-Instruct 모델은 평가 예측 및 순차적 추천 과제에서 뛰어난 성능을 보였습니다. 이 결과는 텍스트 기반 추천 시스템에서 중요한 발전을 의미하며, 다른 연구자들이 접근할 수 있도록 모델과 사전 훈련(pre-training) 및 세부 튜닝(fine-tuning) 데이터셋을 공개하여 연구 및 응용 가능성을 확대했습니다. 공개된 모델과 데이터셋은 'huggingface' 링크를 통해 접근할 수 있습니다.



### From Human-to-Human to Human-to-Bot Conversations in Software Engineering (https://arxiv.org/abs/2405.12712)
Comments:
          Accepted at the 1st ACM International Conference on AI-powered Software (AIware) 2024

- **What's New**: 이 논문은 소프트웨어 개발 과정에서의 대화 동태를 이해하기 위해 AI와 챗봇이 통합된 후의 대화 흐름을 분석합니다. 특히, 사람과 사람 사이의 대화와 사람과 챗봇 사이의 대화의 유사점과 차이점을 연구합니다.

- **Technical Details**: 기존의 인간과 NLU(Natural Language Understanding)-기반 챗봇 간 대화 속성을 소프트웨어 개발 문맥에 맞게 조정하고, 이를 LLM(Large Language Model)-기반 챗봇과의 대화 속성과 비교합니다. 연구는 관찰 연구를 통해 진행되었습니다.

- **Performance Highlights**: LLM 기반 챗봇의 대화 스타일을 통해 개발자들이 기대를 조정하고 소프트웨어 팀 내 소통을 지원하는 방법을 제시합니다. 하지만 사회적 측면에서 LLM 챗봇 대화가 사람 간의 대화를 완전히 대체할 수는 없음을 결론으로 내립니다.



### Multimodal Adaptive Inference for Document Image Classification with Anytime Early Exiting (https://arxiv.org/abs/2405.12705)
Comments:
          Accepted at ICDAR 2024

- **What's New**: 이번 연구는 성능과 효율성 사이의 균형을 필요로 하는 시각적으로 풍부한 문서 이해(VDU) 작업을 위한 새로운 접근 방식을 제안합니다. 기존의 대형 문서 기반 모델들이 우수한 기능을 제공하지만, 높은 연산 비용을 수반하기 때문에, 이 연구에서는 '멀티모달 조기 종료(Early Exit, EE)' 모델 설계를 제안합니다. 이 설계는 다양한 훈련 전략과 종료 레이어 타입 및 배치를 포함하고 있습니다. 본 연구는 VDU 커뮤니티 내에서 최초로 멀티모달 EE 설계를 탐색한 것입니다.

- **Technical Details**: 이번 논문에서는 VDU 작업을 위한 멀티모달 EE 모델을 설계하고 있습니다. 이 모델은 훈련 전략과 종료 레이어의 다양한 타입 및 배치를 포함하고 있어, 예측 성능과 효율성 사이의 '파레토 최적'(Pareto-optimal) 균형을 달성하는 것이 목표입니다. 또한, 전통적인 종료 정책과 비교해, 예측 성능과 효율성 간의 개선된 트레이드오프를 보여줍니다. 모델의 예측 기능을 유지하면서 속도와 지연 시간을 향상시키며, 다양한 레이어에서 종료할 때 자신감 점수를 향상시키기 위해 '캘리브레이션'(calibration)을 활용합니다.

- **Performance Highlights**: 연구 결과에 따르면, 이 멀티모달 EE 설계는 지연 시간을 20% 이상 줄이면서도 기본 정확도를 완전히 유지하는 성과를 보였습니다. 예측 성능을 유지하면서 VDU 애플리케이션의 실용성을 높이는 데 기여하게 됩니다.



### ProtT3: Protein-to-Text Generation for Text-based Protein Understanding (https://arxiv.org/abs/2405.12564)
Comments:
          ACL 2024, 9 pages

- **What's New**: 새로운 프레임워크인 ProtT3가 도입되었습니다. 이 프레임워크는 Protein Language Model(PLM)과 Language Model(LM)을 결합하여 아미노산 서열과 같은 원시 단백질 데이터를 텍스트로 변환하는 기능을 제공합니다. ProtT3 프레임워크는 특히 단백질-텍스트 생성(protein-to-text generation) 분야를 개척하였습니다.

- **Technical Details**: ProtT3는 PLM을 단백질 이해 모듈로 사용하여 아미노산 서열 데이터를 처리하도록 하고, 그 데이터를 LM의 입력 공간으로 전달할 수 있게 하는 cross-modal projector(Q-Former)를 이용하여 단백질과 텍스트 사이의 모달리티 격차를 해소합니다. 이를 통해 효과적인 단백질-텍스트 생성이 가능하게 합니다.

- **Performance Highlights**: 프로틴 캡셔닝(protein captioning), 단백질 질문-응답(protein question-answering), 그리고 단백질-텍스트 검색(protein-text retrieval) 작업에 대한 종합 벤치마크를 설정했습니다. 실험 결과, ProtT3는 현재의 기준 모델들을 크게 웃돌며 주요 구성 요소들의 효능을 입증한 ablation studies도 포함되었습니다.



### CoCo Matrix: Taxonomy of Cognitive Contributions in Co-writing with Intelligent Agents (https://arxiv.org/abs/2405.12438)
- **What's New**: 본 논문은 Flower와 Hayes의 인지적 과정 이론을 적용하여 새로운 인간-지능형 에이전트 공동 작문(Co-writing) 모델을 제안하며, CoCo Matrix라는 이차원적 분류 체계를 소개합니다. 이 모델은 엔트로피와 정보 이득을 기준으로 분류하여, 인간과 에이전트의 상호작용을 심층적으로 이해할 수 있도록 합니다.

- **Technical Details**: CoCo Matrix는 엔트로피(entropy)와 정보 이득(information gain)을 기준으로 네 가지 사분면으로 분류됩니다. 엔트로피가 낮고 정보 이득이 높은 시스템이 아직 충분히 탐구되지 않았지만, 이는 에이전트의 다각적 계획과 인간의 집중된 번역이 필요로 하는 작문 작업에서 유망한 가능성을 제시합니다. 이 체계는 34개의 출판된 시스템을 분류하여 각 시스템의 위치를 시각적으로 표현합니다.

- **Performance Highlights**: CoCo Matrix는 작문 과정에서 발생하는 최소한의 변화를 분석함으로써 작가의 정신 모델을 반영하는 척도로 기능합니다. 이는 작가가 자신의 기여도에 대해 성찰할 수 있도록 돕고, 정보 이득과 엔트로피 지표를 통해 작문 시스템에 관계없이 유익한 통찰을 제공합니다.



### Layout Agnostic Human Activity Recognition in Smart Homes through Textual Descriptions Of Sensor Triggers (TDOST) (https://arxiv.org/abs/2405.12368)
- **What's New**: 이 논문에서는 스마트 홈(smart home)에서의 인간 활동 인식(HAR)을 위한 일반적인 모델을 만들기 위해, 새로운 레이아웃 독립형(layout-agnostic) 접근 방식을 소개하고 있습니다. 본 접근 방식은 원시 센서 데이터의 자연 언어 설명을 활용해 센서 트리거의 텍스트 설명(TDOST: Textual Descriptions Of Sensor Triggers)을 생성하고, 이를 통해 스마트 홈 간 활동 인식 모델의 일반성을 향상시킵니다.

- **Technical Details**: TDOST는 센서 트리거의 주변 조건을 설명하고, 이러한 설명을 통해 숨겨진 활동에 대한 단서를 제공합니다. 기존 원시 센서 데이터 대신 텍스트 임베딩(textual embeddings)을 활용하여 스마트 홈 간 표준 활동을 예측합니다. 이로 인해, 새로운 스마트 홈에서 재훈련이나 적응 과정 없이도 정확한 활동 인식이 가능해집니다. 본 연구는 CASAS 데이터셋을 사용한 벤치마크 실험을 통해 TDOST 기반 모델의 성능을 검증합니다.

- **Performance Highlights**: 광범위한 평가 결과, TDOST 기반 모델이 이전에 보지 못한 스마트 홈에서도 효과적임을 입증하였습니다. 또한, 접근 방식의 개별 구성 요소가 활동 인식 성능에 미치는 영향을 상세히 분석하였습니다.



### Your Transformer is Secretly Linear (https://arxiv.org/abs/2405.12250)
Comments:
          9 pages, 9 figures

- **What's New**: 이 연구는 Transformer 디코더(Transformer decoders)의 새로운 선형적 특성을 밝혀냈습니다. 연구 대상에는 GPT, LLaMA, OPT, BLOOM 등의 모델이 포함되며, 순차 계층 간 임베딩 변환(embedding transformations)을 분석한 결과 거의 완벽한 선형 관계(Procrustes similarity score 0.99)를 발견했습니다.

- **Technical Details**: 연구에서는 Transformer 계층의 출력 노름(output norm)이 일관되게 낮기 때문에, 잔차(residual) 구성 요소를 제거하면 선형성이 감소하는 것을 발견했습니다. 또한, 가장 선형적인 블록(blocks)을 제거하거나 선형 근사(linearly approximating)를 적용해도 손실(loss)이나 모델 성능에 큰 영향을 미치지 않는다고 합니다. 추가로, 소규모 모델(pretraining experiments)에서 층의 선형성을 감소시키기 위해 코사인 유사성 기반 정규화(cosine-similarity-based regularization)를 도입했습니다.

- **Performance Highlights**: 이 정규화(regulation)는 Tiny Stories 및 SuperGLUE와 같은 벤치마크에서 성능 지표를 향상시키고 모델의 선형성을 성공적으로 감소시켰습니다. 이는 Transformer 아키텍처에 대한 기존의 이해에 도전하여, 그 작동 방식이 이전에 생각했던 것보다 더 선형적일 수 있음을 시사합니다.



### Leveraging Discourse Structure for Extractive Meeting Summarization (https://arxiv.org/abs/2405.11055)
- **What's New**: 우리 연구팀은 회의의 복잡한 다자 토론에서 중요한 정보를 더 잘 식별하기 위해 담화 구조(discourse structure)를 활용한 발췌 요약 시스템을 소개합니다. 이 시스템은 발화 내용 간의 의미적 관계를 나타내는 담화 그래프(discourse graphs)를 사용하여 가장 중요한 발화를 선택하고 이를 결합하여 발췌 요약을 생성합니다.

- **Technical Details**: 우리는 담화 그래프(discourse graphs)를 사용하여 회의 발화 간의 의미적 관계를 나타내고, GNN(Graph Neural Network) 기반 노드 분류 모델(node classification model)을 훈련시켜 가장 중요한 발화를 선택합니다. 그 후, 선택된 발화를 결합하여 발췌 요약을 생성합니다.

- **Performance Highlights**: 실험 결과, AMI와 ICSI 데이터를 사용한 우리 모델은 기존의 텍스트 기반 및 그래프 기반 발췌 요약 시스템을 능가했습니다. 이는 분류 및 요약 메트릭 모두에서 입증되었습니다. 추가적으로 담화 구조와 관계 유형에 대한 소거 연구(ablation studies)를 통해 담화 분석 이론을 활용한 향후 NLP 응용 프로그램에 대한 통찰력을 제공했습니다.




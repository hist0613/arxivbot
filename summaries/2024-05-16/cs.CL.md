### Modeling Bilingual Sentence Processing: Evaluating RNN and Transformer Architectures for Cross-Language Structural Priming (https://arxiv.org/abs/2405.09508)
Comments:
          9 pages, 6 figures

- **What's New**: 이 연구는 Recurrent Neural Network (RNN)와 Transformer 모델을 사용하여 언어 간 구조적 priming을 평가합니다. 특히, 이 연구는 중국어-영어 priming에 중점을 두고 있으며, 이는 문장 구조가 두드러지는 현상을 통해 노출된 구조를 반복 사용하게 되는 현상을 의미합니다. Transformer 모델이 RNN보다 이러한 구조적 priming을 더 잘 재현한다는 결과를 도출함으로써, 인간의 문장 처리 과정이 주로 순차적이고 즉각적인 처리에 기반한다고 여겨졌던 기존의 믿음을 도전합니다. 이 연구는 다국어 환경에서 컴퓨터 모델이 인간의 인지 과정을 어떻게 반영할 수 있는지에 대한 새로운 통찰을 제공합니다.

- **Technical Details**: RNN과 Transformer 모델을 사용하여 중국어와 영어 병렬 텍스트로 구성된 거대 코퍼스를 분석했습니다. 모델은 sequence-to-sequence (시퀀스-투-시퀀스) 작업을 처리하기 위해 encoder-decoder 아키텍처를 사용했습니다. Transformer 모델은 self-attention 기제를 사용하여 문장 구조를 학습하고, 다양한 위치 간의 의존성을 포착할 수 있습니다. 반면에 RNN 모델은 순차적 재현을 통해 정보를 처리합니다. 데이터 처리를 위해 Helsinki-NLP 토크나이저를 사용하여 중국어와 영어 텍스트를 토큰 ID로 변환시켰습니다.

- **Performance Highlights**: 실험 결과, Transformer 모델이 RNN보다 cross-lingual structural priming을 더 효과적으로 처리할 수 있음을 보여줍니다. Transformer는 회귀 및 즉각적 처리보다는 cue-based retrieval mechanism을 통한 구조적 priming을 더 잘 재현합니다. 이러한 결과는 인간의 문장 처리 과정이 단순 재현 처리보다는 보다 복잡한 메커니즘을 포함할 수 있음을 시사합니다.



### QueryNER: Segmentation of E-commerce Queries (https://arxiv.org/abs/2405.09507)
Comments:
          Accepted to LREC-COLING 2024

- **What's New**: QueryNER는 전자 상거래(e-commerce) 쿼리를 세분화하는 것을 목표로 하는 새로운 데이터셋과 관련 모델을 소개합니다. 기존의 연구는 주로 특정한 측면(aspect)와 값(value) 추출에 초점을 맞췄지만, QueryNER는 사용자의 쿼리를 의미 있는 청크(chunks)로 분할하는 데 집중합니다. 이 데이터셋은 공개적으로 사용할 수 있으며, 이는 연구자들과 개발자들이 더 나은 쿼리 세분화 모델을 개발하는 데 기여할 것입니다.

- **Technical Details**: QueryNER는 Amazon Shopping Queries Dataset을 기반으로 하여 17가지 유형의 엔터티(entity) 타입을 정의하고 있습니다. 주요 특징으로는 BIO(Beginning, Inside, Outside) 형식(BIO format)을 따르며, 엔터티 타입으로는 core_product_type, product_name, product_number, modifier 등이 포함됩니다. 또한, 실험을 통해 토큰(token)과 엔터티 드롭핑(dropping)을 비교하여 쿼리 복구율을 확인했습니다.

- **Performance Highlights**: 기본 태깅(tagging) 결과를 보고하고, 자동 변환을 사용한 테스트 세트를 통해 모델의 견고성을 확인했습니다. 데이터 증강(data augmentation) 기법이 모델의 노이즈(noise) 처리 능력을 개선하는 데 유용함을 실험적으로 증명했습니다.



### ParaNames 1.0: Creating an Entity Name Corpus for 400+ Languages using Wikidata (https://arxiv.org/abs/2405.09496)
Comments:
          Accepted to LREC-COLING 2024. arXiv admin note: text overlap with arXiv:2202.14035

- **What's New**: ParaNames는 400개 이상의 언어에 걸쳐 1억 4천만 개의 이름을 포함하는 대규모 다국어 병렬 이름 자원을 소개합니다. 1,680만 개의 엔티티에 대해 PER, LOC, ORG와 같은 표준 타입으로 매핑된 이름을 제공합니다. 이는 현재까지 가장 큰 규모의 자원으로, Wikidata를 소스로 사용하여 데이터의 필터링과 표준화를 통해 고품질을 제공합니다.

- **Technical Details**: ParaNames는 Wikidata를 소스로 사용하여 다국어 이름 자원을 자동화된 전처리 절차를 통해 생성합니다. 데이터 필터링 및 표준화 절차를 통해 각 언어에서 사용되는 스크립트와 불필요한 정보를 제거하여 데이터 품질을 보장합니다. 엔티티의 주요 라벨(main label)을 사용하며, 부가 라벨(alias)은 포함하지 않습니다. MongoDB에 엔티티 기록을 저장하고 신속한 처리를 위해 사용됩니다.

- **Performance Highlights**: ParaNames는 두 가지 주요 작업에서 유용성을 입증했습니다. 첫째, 영어와 17개의 다른 언어 사이의 정식 이름 번역을 수행했습니다. 둘째, 다국어 명명 엔티티 인식(named entity recognition)에서 성능 개선을 달성했습니다. ParaNames를 가제티어(gazetteer)로 사용하여 10개의 언어 모두에서 성능 향상을 이루었습니다.



### Beyond Flesch-Kincaid: Prompt-based Metrics Improve Difficulty Classification of Educational Texts (https://arxiv.org/abs/2405.09482)
- **What's New**: 최근의 연구에서는 교육에 활용할 수 있는 대형 언어 모델(LLMs)의 텍스트 난이도 적응능력을 개선하기 위해 새로운 프롬프트 기반 메트릭(Prompt-based metrics)을 도입하였습니다. 기존의 정적 메트릭(Static metrics)인 Flesch-Kincaid Reading Ease 점수는 한계가 명확합니다. 이 연구는 새로운 프롬프트 기반 메트릭이 다양한 교육 수준에 적합한 텍스트 적응성 평가에서 정적 메트릭보다 더 효과적이라고 주장합니다.

- **Technical Details**: LLMs의 일반적인 언어 이해 능력을 활용하여 텍스트의 더 추상적이고 복잡한 특징을 포착하려는 프롬프트 기반 메트릭이 개발되었습니다. 예를 들어, 특정 교육 수준에 맞게 텍스트 난이도를 조절하기 위한 기준으로 사용자의 설명과 반복적으로 언급된 속성을 반영한 메트릭을 포함합니다. 이 연구는 대학생 그룹을 대상으로 1일 간의 사용자 연구를 통해 프롬프트 기반 메트릭의 초기 데이터를 수집하고, 이를 바탕으로 메트릭을 설계 및 평가하였습니다.

- **Performance Highlights**: 회귀 실험 결과, 프롬프트 기반 메트릭은 정적 메트릭 단독 사용보다 텍스트 난이도 분류에서 성능을 크게 향상시켰음을 보여줍니다. 특히, 정적 메트릭과 프롬프트 기반 메트릭을 결합하면 개별 메트릭들이 포착하지 못한 중요한 신호를 캡처하여 텍스트의 난이도를 더 정확하게 평가할 수 있게 됩니다.



### Tell Me Why: Explainable Public Health Fact-Checking with Large Language Models (https://arxiv.org/abs/2405.09454)
- **What's New**: 이번 논문은 대형 언어 모델 (Large Language Models, 이하 LLM)이 공중보건 분야의 주장을 검증하고 그에 대한 설명을 제공하는 능력을 다양한 실험을 통해 분석한 연구를 제시합니다. 특히, zero-shot 및 few-shot 프롬팅과 파라미터 효율적 미세 조정을 통해 공개 및 비공개 모델들의 성능을 평가하며, 자동화된 메트릭과 인간 평가를 사용하여 평가했습니다. GPT-4는 zero-shot 시나리오에서 우수한 성능을 보였지만, few-shot 및 파라미터 효율적 미세 조정 상황에서는 오픈소스 모델들이 성능 격차를 좁히거나 경우에 따라 능가했습니다. 인간 평가에서는 더 섬세한 측면과 함께 gold explanations의 잠재적 문제를 지적했습니다.

- **Technical Details**: 이 연구는 COVID-19 팬데믹과 같은 상황에서 신속한 정보 확산의 문제점을 해결하기 위해 자동화된 팩트체킹 메커니즘의 필요성을 강조합니다. Neural Network 기반의 LLM은 높은 성능을 보여주지만, 해석 가능성 및 설명 가능성에 큰 문제가 존재합니다. 이 문제를 해결하기 위해 Attention Mechanism, Rule Discovery, Summarization Technique 등 다양한 방법이 제안되었습니다. 이번 연구는 Natural Language Explanation (NLE)을 사용하여 팩트체킹 문맥에서 LLM의 설명 생성 능력을 평가합니다. 두 가지 평가 방법(기존의 자동 메트릭과 인간 평가)을 통해 모델의 성능을 전체적으로 평가합니다.

- **Performance Highlights**: 자동화된 평가에서는, zero-shot 시나리오에서 GPT-4가 최고 성능을 보였지만, few-shot 및 파라미터 효율적 미세 조정 상황에서는 오픈소스 모델(예: Falcon-180B, Llama-70b, Vicuna-13, Mistral-7b)들이 경쟁력 있는 성능을 보였으며, 일부 경우에는 GPT-4를 능가했습니다. PEFT 사용 시 최고 성능을 기록했으며, veracity prediction과 explanation generation 작업 모두에서 일관된 성능을 보였습니다. 인간 평가에서는 joint task가 획기적인 향상을 보여줬으며, zero-shot 설정에서 GPT-4가 우수한 설명 생성을 보여줬습니다.



### Facilitating Opinion Diversity through Hybrid NLP Approaches (https://arxiv.org/abs/2405.09439)
Comments:
          Accepted at NAACL 2024, Student Research Workshop

- **What's New**: 현대 민주주의는 시민들의 참여 감소라는 중요한 문제에 직면해 있습니다. 이 논문 제안서는 대규모 온라인 토론을 위한 자연어 처리(NLP)의 도입과 관련된 도전 과제를 식별하고, 인간-인공지능 하이브리드 기술을 통합함으로써 이를 해결하는 방법을 제안합니다. 또한, 이러한 기술이 온라인 토론에서 개인적 관점을 어떻게 드러낼 수 있는지 조사하고자 합니다.

- **Technical Details**: 논문은 세 가지 주요 연구 질문을 다룹니다: 1) NLP를 사용하여 온라인 토론에서 관점을 분석할 때 발생하는 근본적인 문제는 무엇인가? 2) 인간 지능과 NLP를 결합하여 다양한 관점을 효과적으로 포착하는 방법은 무엇인가? 3) 온라인 토론에서 다양한 의견을 표현하기 위해 서로 다른 작업을 어떻게 결합할 것인가? 이 연구는 자연어 처리(NLP)와 대형 언어 모델(LLMs)의 조합을 사용하여 이러한 문제를 해결하고자 합니다.

- **Performance Highlights**: 논문은 대규모 온라인 토론에서 발생하는 엄청난 양의 데이터를 관리하고 분석할 수 있는 방법을 제시합니다. 이를 통해 다양한 관점을 포착하고, 사용자의 상호작용을 더욱 효과적으로 구조화할 수 있는 가능성을 보여줍니다. 또한, 기존의 NLP 방법론이 가지고 있는 한계와 편향성을 극복하기 위해 인간 지능을 통합한 접근 방식을 제시합니다.



### PolygloToxicityPrompts: Multilingual Evaluation of Neural Toxic Degeneration in Large Language Models (https://arxiv.org/abs/2405.09373)
- **What's New**: 최근 대형 언어 모델(LLMs)의 발전으로 인해 전 세계적으로 광범위하게 배포되고 있으며, 이들의 안전성을 보장하기 위해 포괄적이고 다국어적인 독성 평가가 필요합니다. 이를 위해 최초의 대규모 다국어 독성 평가 벤치마크인 'PolygloToxicityPrompts (PTP)'를 소개합니다. PTP는 17개 언어로 구성된 425K 자연 발생 프롬프트를 포함하며, 자동으로 1억 개 이상의 웹 텍스트 문서를 스크래핑하여 다양한 언어 자원을 확보했습니다.

- **Technical Details**: PTP는 Perspective API를 사용하여 자연 발생 독성 프롬프트를 수집하고 점수를 매기는 방식을 통해 만들어졌습니다. 100M 이상의 웹 텍스트 문서에서 425K개의 프롬프트를 수집했으며, 이 프롬프트는 비독성부터 고독성 프롬프트까지 다양합니다. 이 데이터셋은 Arabic(아랍어), Chinese(중국어), Korean(한국어)을 포함한 17개 언어를 지원합니다. PTP를 사용하여 60개 이상의 LLMs를 벤치마킹하고 모델 크기, 프롬프트 언어, 지시 및 선호도 조정 방법이 독성에 미치는 영향을 연구하였습니다.

- **Performance Highlights**: 주요 연구 결과로서, 언어 자원이 감소하거나 모델 크기가 증가할수록 독성이 증가한다는 것을 발견했습니다. 지시 및 선호도 조정은 모델의 독성을 감소시키는 데 효과적이었으나, 선호도 조정 방법에 따른 영향 차이는 거의 없었습니다. 이러한 결과는 LLM의 안전성에 있어 중요한 단점을 밝혀내고, 다국어 독성 완화를 위한 추가 연구의 필요성을 강조합니다.



### Large Language Model Bias Mitigation from the Perspective of Knowledge Editing (https://arxiv.org/abs/2405.09341)
- **What's New**: 기존의 편향 감소(debiasing) 방법은 다양한 사회 집단 간의 동등성을 목표로 하지만 개별 사실을 무시하여 기존 지식을 수정하는 문제가 있었습니다. 이러한 문제를 해결하기 위해 BiasKE라는 새로운 편향 감소 벤치마크를 제안하고, 새로운 편향 감소 방법인 Fairness Stamp(FAST)를 소개합니다. 이 방법은 개별 편향된 지식에 대해 세밀한 보정을 통해 편향을 수정할 수 있습니다.

- **Technical Details**: BiasKE 벤치마크는 공정성(fairness), 특이성(specifity), 일반화(generalization)를 평가하는 보완 지표를 통해 편향 감소 성능을 체계적으로 평가합니다. 또한, FAST 방법은 편향을 조정하는 과정에서 편향된 지식을 특정하는 인과 추적 방법(이하 'causal-tracing-based method')을 활용하여 결정적인 층(layer)을 찾아내고, 경량 모듈형 네트워크를 추가하여 세밀하고 효율적인 편향 감소를 수행합니다.

- **Performance Highlights**: 광범위한 실험을 통해 FAST가 StereoSet 및 Crows-Pairs와 같은 데이터셋에서 뛰어난 편향 감소 성능을 보임을 확인했습니다. 또한, 더 큰 모델(GPT-Neo, Llama)로 확장하여 실세계 응용에서의 확장성을 입증했습니다. 후속 실험에서는 다운스트림 작업 및 지속적인 편향 감소에서의 효과를 강조했습니다.



### Prompting-based Synthetic Data Generation for Few-Shot Question Answering (https://arxiv.org/abs/2405.09335)
Comments:
          LREC-COLING 2024

- **What's New**: 새로운 연구는 대형 언어 모델(Large Language Models, LMs)을 활용하여 소량의 레이블 데이터로도 높은 성능을 발휘하는 질문 응답(Question Answering, QA) 모델을 개발하는 방법을 제안합니다. 기존에는 많은 데이터 주석 작업이 필요했으나, 언어 모델에 내재된 지식을 활용함으로써 이 과정을 대폭 간소화할 수 있습니다.

- **Technical Details**: 연구에서는 'Prompting' 프레임워크를 활용하여 데이터 생성 작업을 수행합니다. 이는 사전 훈련된 언어 모델(Pre-trained Language Models, PLMs)이 가진 언어적 이해를 통해 고품질의 질문과 답변 쌍을 생성하는 방법을 말합니다. MRQA(Machine Reading Question Answering) 과제를 위주로 진행되었고, 우리의 접근 방식은 주어진 문서에서 답 후보를 추출한 후 사전 훈련된 LM을 이용해 질문을 생성하는 두 단계를 포함합니다.

- **Performance Highlights**: 몇 가지 주요 데이터셋에 대해 소수 샘플만을 사용한 질의응답에서 기존 최신 기법들을 능가하는 성능을 보였습니다. 특히, TextbookQA 데이터셋에서는 전체 데이터 설정에서도 뛰어난 성능을 발휘하며, 16개의 라벨된 샘플만으로도 기존 최고 성능인 49.9% F1을 능가하는 64% F1 점수를 기록했습니다.



### Comparing the Efficacy of GPT-4 and Chat-GPT in Mental Health Care: A Blind Assessment of Large Language Models for Psychological Suppor (https://arxiv.org/abs/2405.09300)
- **What's New**: 최신의 연구는 GPT-4와 Chat-GPT라는 두 개의 대형 언어 모델의 심리적 프롬프트(response)에 대한 성능을 비교하여 정신 건강 관리 환경에서의 적용 가능성을 평가했습니다.

- **Technical Details**: 이 연구는 시각 차단 방법을 사용하여 임상 심리학자가 프롬프트에 대한 모델의 응답을 출처를 알지 못한 상태에서 평가하였습니다. 평가에는 우울증, 불안 및 외상을 포함한 다양한 정신 건강 주제가 포함되었습니다.

- **Performance Highlights**: {'Overall Ratings': 'GPT-4는 평균 8.29/10점을 받았고, Chat-GPT는 평균 6.52/10점을 받았습니다.', 'Quality of Responses': 'GPT-4는 임상적으로 더욱 관련 있고 공감적인 응답을 생성하는 데 있어서 더 효율적임을 시사했습니다.'}

- **Conclusion**: 이 연구는 대형 언어 모델이 정신 건강 관리 환경에서 어떻게 채택될 수 있는지를 보여주며, 해당 모델들을 임상적으로 최적화하기 위한 지속적인 연구 개발의 중요성을 강조합니다. 또한, 두 모델 간의 성능 차이의 특정 요인을 이해하고 다양한 인구 및 정신 건강 상태에 걸쳐 일반화 가능성을 탐구하는 추가 조사가 필요합니다.



### Do language models capture implied discourse meanings? An investigation with exhaustivity implicatures of Korean morphology (https://arxiv.org/abs/2405.09293)
Comments:
          Proceedings of the Society for Computation in Linguistics (SCiL) 2024, Association for Computational Linguistics (ACL) Anthology

- **What's New**: 이 논문은 한국어의 차별적 목적격 표시(Differential Object Marking, DOM)가 대형 언어 모델(Large Language Models, LLMs)에 의해 어떻게 처리되는지 평가합니다. 논문에서는 LLMs가 단어의 분포적 의미 뿐만 아니라 문맥을 통해 주어지는 비문자적 의미를 캡처할 수 있는지 여부를 탐구합니다.

- **Technical Details**: 한국어 DOM은 명사구의 의미 특징뿐만 아니라 이에 무관한 담화 특징에 따라 후치사가 선택됩니다. 연구는 세 가지 후치사(를, 는, 생략)가 문법적 표지자 및 담화 표지자로서 갖는 의미를 평가합니다. 예를 들어, 는은 대조 의미를 포함하고, 를은 선택된 대상의 완전성을 나타내며, 생략은 담화 문맥에서 대상이 논의된 것임을 나타냅니다.

- **Performance Highlights**: 결과는 LLM이 문법적 표지자의 담화적 의미를 인코딩하는 것이 담화 표지자의 의미를 인코딩하는 것보다 더 어려울 수 있음을 시사합니다. 이를 통해 현재의 LLM이 비문자적 의미를 포착하는 데 있어 제한적인 능력을 보여줍니다.



### Sign of the Times: Evaluating the use of Large Language Models for Idiomaticity Detection (https://arxiv.org/abs/2405.09279)
Comments:
          Presented at the MWE-UD Workshop at LREC-COLING 2024

- **What's New**: 이 연구에서는 최신 대형 언어 모델(LLMs)이 관용 표현(idiomatic expressions)을 포함한 작업에서 얼마나 잘 수행하는지 조사합니다. SemEval 2022 Task 2a, FLUTE, MAGPIE 데이터를 통해 이러한 모델의 성능을 평가한 결과, 경쟁력 있는 성능을 보이지만 특수 작업에 맞게 미세 조정된 모델에 비해 여전히 뒤처진다는 점을 발견했습니다. 모델 크기가 커질수록 성능이 향상되는 경향이 있음을 확인했습니다.

- **Technical Details**: 대형 언어 모델(LLMs)의 성능은 세 가지 관용 표현 데이터셋인 SemEval 2022 Task 2a, FLUTE, MAGPIE에서 평가되었습니다. FLUTE는 비유적 언어를 자연어 추론(NLI) 작업으로 구성했으며, SemEval은 이진 분류 관용 표현 탐지 작업을 포함하고 있습니다. MAGPIE는 다중 의미를 가진 표현의 예시가 포함된 코퍼스입니다. 이 작업에서는 모델을 미세 조정하지 않고 사전 학습된 LLMs만을 사용했습니다.

- **Performance Highlights**: 대형 LLMs는 관용 표현 데이터셋에서 좋은 성능을 보였으나, 여전히 미세 조정된 encoder-only 모델보다는 성능이 떨어졌습니다. 예를 들어, FLUTE 데이터셋의 비유적 언어 부분에서 T5 모델을 사용한 경우, 79.2%의 정확도를 기록한 반면, 더 작은 encoder-only 모델을 미세 조정했을 때는 더 높은 성능을 보였습니다. SemEval 2022 Task 2a 데이터셋에서 최고 성능을 기록한 모델은 XLM-RoBERTa를 미세 조정한 모델이었습니다.



### New Textual Corpora for Serbian Language Modeling (https://arxiv.org/abs/2405.09250)
- **What's New**: 이 논문은 대형 언어 모델(large language models) 훈련을 위한 세르비아어 및 세르보-크로아티아어 텍스트 코퍼스(textual corpora)를 소개합니다. 특히, 새로운 세 가지 코퍼스를 도입하였습니다: 세르보-크로아티아어의 새로운 통합 웹 코퍼스(umbrella web corpus), 세르비아 모든 대학의 박사 논문을 기반으로 한 고품질 코퍼스, 그리고 같은 출처에서 번역된 초록의 병렬 코퍼스(parallel corpus)입니다. 이 논문은 또한 스타일로미트리(stylometric) 방법을 사용하여 기존과 새로운 코퍼스의 독특성을 평가하고 결과를 간략히 논의합니다.

- **Technical Details**: 코퍼스는 웹 코퍼스(web corpora), 교과서 코퍼스(textbook corpora), 문예 코퍼스(literary corpora), 합성 코퍼스(synthetic corpora), 혼합 코퍼스(mixed corpora)로 분류됩니다. 또한, 형태에 따라 단순 텍스트 코퍼스(plain-textual corpora), 주석이 있는 코퍼스(annotated corpora), 병렬 코퍼스(parallel corpora)로 나뉘어 집니다. 이 연구에서는 특정 온라인 소스(Hugging Face, CLARIN, European Language Grid)로부터 세르비아어 및 세르보-크로아티아어 코퍼스를 수집하여 분석했습니다.

- **Performance Highlights**: 논문에 소개된 대부분의 코퍼스는 웹 기반이고, 웹 스크래핑(web scraping)을 통해 수집되었습니다. 주요 코퍼스로는 srWac, meWac, hrWac, bsWac 등이 있으며, 이들은 대규모 단어 집합을 포함합니다. 또한, MaCoCu 프로젝트를 통해 얻어진 대규모 코퍼스도 포함되어 있습니다. 여러 코퍼스를 중복 제거하여 BERTić-data와 XLM-R-BERTić-data라는 통합 코퍼스가 만들어졌으며, 이는 각각 8.4억, 11.5억 단어를 포함하고 있습니다.



### Word Alignment as Preference for Machine Translation (https://arxiv.org/abs/2405.09223)
- **What's New**: 본 논문에서는 기계 번역(MT)에서 발생하는 환각(hallucination)과 생략(omission) 문제를 개선하기 위해 새로운 접근 방식을 제안하고 있습니다. 대형 언어 모델(LLM)을 사용할 때 이러한 문제는 더욱 두드러지며, 이를 해결하기 위해 단어 정렬(word alignment)을 활용하여 LLM 기반의 MT 모델을 최적화하고자 합니다.

- **Technical Details**: 단어 정렬과 번역 중 발생하는 환각 및 생략 현상 간의 상관관계를 연구한 후, 이를 선호 데이터(preference data)로 활용하여 LLM 기반 MT 모델을 최적화하는 Word Alignment Preference(WAP) 방법을 제안합니다. 다양한 번역 도구를 사용하여 선호되는 번역과 거부된 번역을 선택한 데이터를 구축한 후, 직접 선호 최적화(DPO)를 통해 모델을 최적화합니다.

- **Performance Highlights**: 환각과 생략을 평가할 수 있는 기존 평가자가 존재하지 않기 때문에, GPT-4를 평가자로 활용하여 문제를 효과적으로 평가하는 방안을 제시했습니다. 실험 결과, WAP 방식이 환각 및 생략 문제를 완화하는 데 효과적임을 확인했습니다.



### Bridging the gap in online hate speech detection: a comparative analysis of BERT and traditional models for homophobic content identification on X/Twitter (https://arxiv.org/abs/2405.09221)
Comments:
          6 pages, Homophobia detection model available at: this https URL. The dataset used for this study is available at: this https URL - This paper has been accepted by the 6th International Conference on Computing and Data Science (CONF-CDS 2024)

- **What's New**: 이 연구는 온라인 혐오 발언 탐지 분야에서 특히 간과되었던 동성애 혐오 (homophobia)에 초점을 맞추고 있습니다. 저자들은 최신 감정 분석 모델인 BERT와 전통적인 머신 러닝 방법을 활용하여 X/Twitter에서 동성애 혐오 내용을 식별하는 정교한 접근 방식을 개발했습니다. 이는 동성애 혐오가 탐지 모델에서 지속적으로 과소대표되는 문제를 해결하기 위한 중요한 연구입니다.

- **Technical Details**: 이 연구는 BERT와 같은 최신 감정 분석 모델 뿐만 아니라 전통적인 머신 러닝 방법을 이용하였으며, 그 결과 BERT가 전통적인 방법보다 뛰어난 성능을 보인다는 것을 발견했습니다. 그러나 검증 기술(validation technique)의 선택이 모델 성능에 영향을 미칠 수 있다는 점도 강조했습니다. 다양한 모델 성능 분석을 통해 가장 강력한 BERT 기반 모델을 공개하였습니다.

- **Performance Highlights**: BERT 모델이 동성애 혐오 탐지에 있어 전통적인 모델보다 월등히 뛰어난 성능을 보였습니다. 또한, 저자들은 동성애 혐오 탐지에 관한 가장 큰 오픈 소스 라벨링된 영어 데이터셋을 공개함으로써 온라인 안전과 포용성을 향상시키기 위해 노력하고 있습니다. 이 연구는 온라인 혐오 발언 탐지 분야에 큰 기여를 하고 있으며, 더 나아가 LGBTQIA+ 혐오 발언 탐지로 범위를 넓힐 계획입니다.



### HumanRankEval: Automatic Evaluation of LMs as Conversational Assistants (https://arxiv.org/abs/2405.09186)
Comments:
          Accepted to NACCL 2024 main conference

- **What's New**: 새로운 평가 과제인 HumanRankEval(HRE)을 제안합니다. HRE는 대규모, 다양한, 고품질의 질문과 답변 집합을 포함하여 LMs(Language Models)의 대화 능력을 자동으로 평가합니다.

- **Technical Details**: HRE는 StackOverflow와 StackExchange에서 수집된 다양한 주제의 질문 및 인간이 평가한 답변을 포함합니다. 각 질문에 대한 여러 답변을 LM의 분포 하에서 log-likelihood로 평가하고, 이를 인간의 평가와의 상관성을 계산합니다. 이 과정은 LM이 인간의 선호도와 얼마나 일치하는지를 측정합니다. 데이터셋과 코드는 GitHub에서 다운로드할 수 있습니다.

- **Performance Highlights**: HRE는 pretrained 및 instruction-tuned LMs의 성능을 효과적으로 구분하며, 다른 평가 프레임워크와 비교하였을 때 인간의 평가와 잘 일치합니다. OpenLLM 리더보드와 비교하였을 때, HRE는 pretrained와 instruction-tuned LMs를 보다 효과적으로 구분합니다. HRE는 빠른 개발 반복을 위해 인간의 판단을 대체할 수 있는 제안으로 사용될 수 있습니다.



### Adapting Abstract Meaning Representation Parsing to the Clinical Narrative -- the SPRING THYME parser (https://arxiv.org/abs/2405.09153)
Comments:
          Accepted to the 6th Clinical NLP Workshop at NAACL, 2024

- **What's New**: 이 논문은 임상 노트(clinical notes)에 특화된 AMR(추상 의미 표현, Abstract Meaning Representation) 파서를 설계하고 평가한 첫 연구입니다. 주된 목표는 임상 텍스트 데이터를 구조화된 AMR 표현으로 정확하게 변환하여 해석성과 사용성을 증대시키는 것입니다.

- **Technical Details**: 최신 AMR 파서를 지속적인 훈련(continuous training)기법을 활용하여 THYME(Your Medical Events)의 대장암 데이터셋에 맞게 적응시켰습니다. 데이터 증강(data augmentation) 기법을 사용하여 AMR 구조 예측의 정확성을 향상시켰습니다. 또한, 임상 노트의 도메인 적응(domain adaptation)을 위한 데이터 요구사항도 탐구했습니다.

- **Performance Highlights**: THYME 코퍼스의 대장암 데이터셋에서 88%의 뛰어난 F1 점수를 달성했습니다. 이는 our parser의 강력한 성능을 강조하며, 임상 내러티브를 구조화된 의미 표현으로 더 깊이 이해하는 데 기여할 수 있음을 보여줍니다.



### A safety realignment framework via subspace-oriented model fusion for large language models (https://arxiv.org/abs/2405.09055)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 안전성을 위한 새로운 재조정 방법을 소개합니다. 전통적인 안전 미세 조정 방식을 개선하여 '하위 공간 지향 모델 융합 스킴(Subspace-Oriented Model Fusion, SOMF)'을 통해 모델을 재조정하여 안전성을 유지하면서도 여러 다운스트림 작업의 성능을 저하시키지 않도록 설계되었습니다. 이 새로운 스킴은 Task-specific 모델들의 안전성을 재조정할 수 있을 뿐만 아니라, 여러 Task-specific 모델을 융합할 때도 효과적으로 적용할 수 있습니다.

- **Technical Details**: SOMF는 모델 융합을 통해 안전성과 성능 간의 균형을 유지하는 새로운 접근법입니다. 초기 재조정된 모델과 현재의 미세 조정된 모델의 안전 관련 영역을 하위 공간 마스킹 기법을 통해 식별한 후, 안전 하위 공간을 기반으로 이들을 융합합니다. 이를 위해 모든 Task 벡터를 모델의 가중치로부터 분리하고, 식별된 안전 하위 공간을 통해 초기 안전 재조정된 모델과 모든 Task 벡터를 융합하는 과정을 거칩니다.

- **Performance Highlights**: SOMF는 한 개의 Task-specific 모델 뿐만 아니라 여러 모델의 융합 시에도 안전하게 성능을 유지할 수 있음을 실험적으로 입증했습니다. 실험 결과, SOMF는 영어, 중국어, 힌디의 지침 준수 및 코드와 수학 문제 해결 능력을 포함한 여러 다운스트림 작업에서 탁월한 성능을 유지하면서도 안전성을 효과적으로 재조정했습니다. 즉, 성능 저하 없이도 모델의 안전성을 확립할 수 있음을 확인했습니다.



### A Japanese-Chinese Parallel Corpus Using Crowdsourcing for Web Mining (https://arxiv.org/abs/2405.09017)
Comments:
          Work in progress

- **What's New**: 이 연구에서는 크라우드소싱을 통해 10,000개의 URL 쌍(병렬 톱 페이지 쌍)을 수집하여 460만 쌍의 일본어-중국어 병렬 코퍼스를 생성했습니다. 이 코퍼스는 고품질의 일본어-중국어 병렬 문장 쌍 120만 쌍을 사용하여 통계적 언어 모델과 단어 번역 확률에 기반한 병렬 코퍼스 필터를 훈련시키는 데 사용되었습니다.

- **Technical Details**: 문서와 문장 정렬을 위해 16만 쌍의 일본어-중국어 이중언어 사전을 사용했습니다. 크라우드소싱 기반의 웹 마이닝을 통해 일본어-중국어 평행 데이터를 수집하는 것이 실제로 가능하다는 것을 확인하기 위해 CCMatrix(전 세계 웹 마이닝을 통해 생성된 1,240만 쌍의 병렬 코퍼스)와 비교하였습니다.

- **Performance Highlights**: 코퍼스의 크기가 CCMatrix의 1/3에 불과함에도 불구하고, 두 모델의 번역 정확도가 비슷하다는 것을 발견했습니다. 이는 크라우드소싱을 통해 병렬 데이터를 수집하는 것이 매우 효과적임을 시사합니다.



### LLM-Assisted Rule Based Machine Translation for Low/No-Resource Languages (https://arxiv.org/abs/2405.08997)
- **What's New**: 기존 자원 기반 언어(전혀 공개된 데이터가 없는 언어)를 위한 새로운 기계 번역 패러다임을 제안하며, 이를 통해 오웬스 밸리 파이우트 (OVP) 언어, 즉 심각하게 위기에 처한 토착 아메리카 언어를 위한 첫 번째 언어 교육/활성화 지향 번역기를 설계했습니다. 이 접근법은 LLM-Assisted Rule Based Machine Translation(LLM-RBMT)을 사용하여 간단한 문장 번역을 수행합니다.

- **Technical Details**: 제안된 OVP to English 번역기는 선택 기반 문장 빌더를 사용하여 사용자가 문장을 구성할 수 있도록 합니다. 문장 빌더는 주어, 주어 접미사, 동사, 동사 접미사, 목적어, 목적어 접미사 등을 선택할 수 있게 합니다. 또한, LLM을 사용하여 자연어 문장을 구조화된 단순 문장으로 분해하고 JSON 형식의 정보를 자연어 문장으로 변환합니다. 이 방법을 통해 LLM은 번역하고자 하는 목표 언어와 직접 상호작용하지 않습니다.

- **Performance Highlights**: 이 번역기는 자연어 처리 작업에서 인간 수준의 성능을 발휘하는 OpenAI의 GPT-3.5-turbo 및 GPT-4를 사용하여 문장을 처리합니다. 이는 초자연 언어 데이터가 전혀 없는 상황에서도 높은 정확도로 간단한 문장 번역을 수행할 수 있음을 보여줍니다. 예를 들어, OVP 문장을 빌드하여 영어로 번역하는 도구와 영어 문장을 OVP로 번역하는 도구를 구성했습니다.



### Large Language Models for Human-Machine Collaborative Particle Accelerator Tuning through Natural Languag (https://arxiv.org/abs/2405.08888)
Comments:
          22 pages, 5 figures

- **What's New**: 이 연구는 입자 가속기를 자동으로 조정(tuning)하는 데 있어서 대형 언어 모델(large language models, LLMs)을 적용하는 새로운 접근 방법을 제안합니다. 연구진은 LLM이 운영자의 자연어 지시(prompt) 만을 기반으로 입자 가속기 하위 시스템을 성공적으로 자동 조정하는 가능성을 증명하였습니다. 이를 통해 LLM의 활용 범위를 입자 가속기 자동화 분야까지 확장하는 데 기여할 수 있음을 보여줍니다.

- **Technical Details**: 이번 연구에서는 Bayesian 최적화(BO)와 강화학습 기반 최적화(RLO) 같은 최신 최적화 알고리즘과 비교하여 LLM 기반 솔루션의 성능을 평가하였습니다. 특히, LLM이 매우 비선형적인(real-world highly non-linear) 목적 함수를 수치적으로 최적화하는 방법을 시연하였습니다. 이 작업은 LLM의 또 다른 복잡한 문제 해결 능력을 의미하며, 입자 가속기의 일상 운영에서 자율 조정 알고리즘의 배포를 가속화할 것입니다.

- **Performance Highlights**: LLM 기반 솔루션은 기존의 BO 및 RLO와 유사한 성능을 보여주었으며, 특히 전문가의 개입 없이 자연어 지시만으로 입자 가속기를 조정할 수 있는 능력을 입증했습니다. 이로 인해 최적화 및 머신러닝 전문가의 도움 없이도 다양한 새로운 조정 작업을 수행할 수 있는 가능성이 열렸습니다.



### Matching domain experts by training from scratch on domain knowledg (https://arxiv.org/abs/2405.09395)
- **What's New**: 최근 연구에 따르면, 소형 LLM(Large Language Model)인 124M-parameter GPT-2 모델이 특화된 도메인 지식으로 훈련될 경우 신경과학 실험 결과를 예측하는 데 있어 인간 전문가들과 유사한 성과를 낼 수 있다고 밝혀졌습니다. 이는 광범위한 트레이닝 데이터 대신 도메인 특화 데이터만으로도 높은 성능을 달성할 수 있음을 시사합니다.

- **Technical Details**: 본 연구에서는 1.3억개 토큰(token)을 포함하는 신경과학 데이터셋으로 124M-parameter GPT-2 모델을 훈련했습니다. 두 가지 접근법을 사용했습니다: 사전 학습된 GPT-2 모델을 신경과학 데이터로 파인튜닝(finetune)하거나, 신경과학 텍스트에 특화된 토크나이저(tokenizer)로 처음부터 재훈련했습니다. 이후 BrainBench라는 벤치마크 테스트를 통해 모델의 성능을 측정했습니다.

- **Performance Highlights**: 파인튜닝된 GPT-2 모델은 BrainBench에서 인간 전문가들과 유사한 63.5%의 정확도를 기록했으며, 처음부터 신경과학 데이터를 사용해 훈련한 모델도 63%의 정확도를 달성했습니다. 이는 형성 토크나이저를 사용한 도메인 특화 훈련이 유효함을 보여줍니다. 또한, 신경과학 토크나이저는 사전 학습된 토크나이저에 비해 두 배 많은 신경과학 관련 토큰을 포함하고 있었습니다.



### ALPINE: Unveiling the Planning Capability of Autoregressive Learning in Language Models (https://arxiv.org/abs/2405.09220)
- **What's New**: 이번 논문에서는 Project ALPINE (Autoregressive Learning for Planning In Networks)에 대한 연구 결과를 다루고 있습니다. 프로젝트 ALPINE은 트랜스포머(Transformer) 기반 언어 모델의 자동회귀 학습 메커니즘을 통해 네트워크에서의 계획 능력을 개발하는 것을 목표로 합니다. 이 논문은 트랜스포머가 경로 찾기(path-finding) 작업을 수행하는 방법과 트랜스포머의 한계를 이론적으로 분석하고 실험적으로 검증합니다.

- **Technical Details**: 트랜스포머 모델을 통해 인접 행렬(adjacency matrix)과 도달 가능성 행렬(reachability matrix) 정보를 추출하는 방법을 분석했습니다. 경사하강법(gradient descent)에 기반한 학습 과정을 통해 트랜스포머가 인접 행렬과 제한적인 형태의 도달 가능성 행렬을 학습할 수 있음을 보였습니다. 실험 결과 인접 행렬은 피드포워드 층(feed-forward layer)에, 도달 가능성 행렬은 어텐션 층(attention layer)에 내재됨을 확인했습니다.

- **Performance Highlights**: 프로젝트 ALPINE의 실험 결과, 트랜스포머 모델은 새로운 소스-타겟 쌍에 대한 유효 경로를 생성하는 데 높은 정확성을 보였습니다. 그러나 경로 부분들을 연결하여 경로를 완성해야 하는 경우 트랜스포머의 성능은 현저히 떨어졌습니다. 이는 트랜스포머가 복잡한 도달 가능성 관계, 특히 트랜지티브(transitive) 관계를 학습하는 데 제한이 있음을 시사합니다. 또한 실제 계획 벤치마크인 Blocksworld 문제에서도 트랜스포머의 학습 능력을 입증했습니다.



### Spatial Semantic Recurrent Mining for Referring Image Segmentation (https://arxiv.org/abs/2405.09006)
- **What's New**: 이 논문에서는 고품질의 크로스 모달리티 퓨전(cross-modality fusion)을 달성하기 위한 공간적 의미 반복 채굴(Spatial Semantic Recurrent Mining, S²RM) 방식을 제안합니다. 이는 언어 기능의 분배, 공간적 의미 반복 공동 분석(spatial semantic recurrent coparsing), 그리고 해석된 의미 균형(parsed-semantic balancing)의 3단계를 따릅니다. 또한, 서로 다른 구획(full-resonance)을 저비용으로 통합하는 교차 스케일 추상 의미 중심 디코더(Cross-scale Abstract Semantic Guided Decoder, CASG)도 제안되었습니다.

- **Technical Details**: S²RM은 처음에 분배 기준이 약하지만 분포를 인식하는 언어 기능을 생성합니다. 이후 하나의 모달리티 컨텍스트(context)의 회전된 기능에서 각 행과 열의 기능을 번들로 묶고, 다른 모달리티 컨텍스트에서 관련 의미를 반복적으로 파싱합니다. 마지막으로 복자중석화된 중량을 통해 다른 파싱 의미의 기여도를 균형을 맞춥니다. CASG는 이전 단계의 교차 스케일 기능과 언어 기능을 활용하여 적응적인 시각적 기능 정제를 성취합니다.

- **Performance Highlights**: 제안된 방법은 RefCOCO, RefCOCO+, RefCOCOg, 그리고 ReferIt의 네 가지 최신 도전적인 데이터셋에서 기타 최첨단 알고리즘에 비해 유리한 성능을 보였습니다.



### What is it for a Machine Learning Model to Have a Capability? (https://arxiv.org/abs/2405.08989)
Comments:
          forthcoming in the British Journal for the Philosophy of Science (BJPS)

- **What's New**: 이 논문은 머신러닝(ML) 모델의 능력을 평가하는 새로운 접근법을 제안합니다. 특히, 대형 언어 모델(LLMs)의 능력을 사례로 사용하여, 머신러닝 모델이 특정 작업을 수행하는 능력을 명확히 정의하고 평가하기 위한 조건부 분석(method)을 개발합니다. 이를 통해 공평한 모델 간 비교를 위한 절차를 제안합니다.

- **Technical Details**: 제안된 접근법인 조건부 모델 능력 분석(CAMA: Conditional Analysis of Model Abilities)은 머신러닝 모델이 특정 작업을 '시도'할 때 성공할지를 기반으로 모델의 능력을 평가합니다. 예를 들어, LLM이 법 시험을 통과할 수 있는지, 인간을 기만할 수 있는지 등의 능력을 이 분석법을 이용해 평가할 수 있습니다.

- **Performance Highlights**: 이 논문은 LLM과 같은 일반 목적의 모델이 실제로 어떤 능력을 갖추고 있는지에 대한 더 명확한 이해를 제공합니다. 또한, 이러한 모델의 능력을 평가하는 방식에 공평성과 일관성을 더할 수 있는 평가 방법론을 제시합니다. 이는 정책 입안자, 규제 기관, 연구자 등 다양한 이해 관계자에게 중요한 통찰을 제공합니다.



### Challenges in Deploying Long-Context Transformers: A Theoretical Peak Performance Analysis (https://arxiv.org/abs/2405.08944)
- **What's New**: 새로운 연구로 긴 컨텍스트 Transformer 모델의 배포 비용을 줄이기 위한 동시 프로그래밍 프레임워크가 제안되었습니다. 특히, 1백만 토큰(100K~10M 토큰)의 긴 컨텍스트 모델을 4K 토큰 모델만큼 저렴하게 배포하는 방법을 탐구합니다. 이 연구는 긴 컨텍스트 Transformer의 주요 비용 원인이 KV 캐시의 크기라는 것을 강조합니다.

- **Technical Details**: 이 연구는 긴 컨텍스트 요청을 제한된 GPU 고대역 메모리(HBM) 환경에서 다중으로 처리하는 효율성 도전 과제를 분석하는 동시 프로그래밍 프레임워크를 제공합니다. GPT-3.5 수준의 모델을 A100 NVLink를 사용하여 50K 컨텍스트에서 실행하며, 대규모 KV 캐시로 인해 발생하는 4가지 배포 도전 과제(프리필링 시간 증가, 동시 사용자 수 제한, 디코딩 지연 시간, 컨텍스트 전환 지연)를 다룹니다.

- **Performance Highlights**: 제안된 프레임워크를 통해 30억 매개변수 모델을 예로 들며, 4K 컨텍스트에서는 약 100명의 사용자를 동시에 처리할 수 있지만, 100K 컨텍스트에서는 단 5명만 처리할 수 있음을 보여줍니다. 연구는 Session-Based Throughput을 주요 평가 지표로 사용하며, 메모리 제한 및 컨텍스트 전환이 성능에 미치는 영향을 분석합니다.



### Self-supervised vision-langage alignment of deep learning representations for bone X-rays analysis (https://arxiv.org/abs/2405.08932)
- **What's New**: 이 논문은 프랑스어 보고서와 쌍을 이루는 뼈 X-레이(bone X-rays)를 이용해, 다양한 뼈 방사선 촬영(tacks)에서 성능을 개선하기 위한 비전-언어 사전학습(vision-language pretraining) 방법을 제안합니다. 특히, 프랑스어 의료 보고서를 익명화하고 처리하는 실용적인 파이프라인을 도입한 것이 주목할 만합니다.

- **Technical Details**: 이 방법은 심층 모델 인코더(deep model encoders)에서 도출된 시각적 및 텍스트 임베딩 공간을 자가 지도 정렬(self-supervised alignment)하는 것으로 구성됩니다. 비전-언어 사전학습을 수행하며, 이를 통해 얻은 이미지 인코더(image encoder)를 사용하여 다양한 다운스트림 작업을 처리할 수 있습니다. 사용된 텍스트 인코더는 프랑스어 보고서에 적합하게 설계된 모델들을 탐구했으며, 이미지 인코더는 최신의 Vision Transformer(ViT)을 채택했습니다.

- **Performance Highlights**: 해당 접근법은 인간 전문가의 주석을 크게 요구하는 대안 모델들과 비교했을 때, 최소한의 주석으로도 경쟁력 있는 성능을 발휘했습니다. 주요 작업은 골관절염(osteoarthritis) 정량화, 소아 손목의 뼈 나이(bone age) 추정, 뼈 골절 및 이상 탐지 등이 포함됩니다.

- **Significance**: 이 연구는 프랑스어 보고서를 이용해 뼈 X-레이 표현을 형성하는 최초의 시도로서, 병원에 비축된 대량의 이미지-보고서 쌍 데이터에서 큰 이점을 얻었습니다. 이 방법은 비전 모델을 다양한 의료 응용 분야에 널리 배포할 수 있게 하는 중요한 기여를 합니다. 또한, 프랑스어 데이터를 활용한 비전-언어 사전학습이 의료 분야에서 효과적으로 사용될 수 있음을 보여줍니다.



### PromptMind Team at EHRSQL-2024: Improving Reliability of SQL Generation using Ensemble LLMs (https://arxiv.org/abs/2405.08839)
Comments:
          Accepted as a poster for Clinical NLP workshop at NAACL 2024

- **What's New**: 이 논문은 EHRSQL-2024 공동 과제에 대한 접근법을 소개합니다. 본 과제는 전자 건강 기록(EHR)의 자연어 질문을 SQL 쿼리로 변환하는 신뢰성 있는 Text-to-SQL 시스템을 개발하는 것을 목표로 합니다. 우리는 대형 언어 모델(LLMs)을 활용한 프롬프트 및 미세 조정을 통한 두 가지 접근법을 제안합니다. 이 논문의 결과는 개별 접근법이 높은 실행 정확도를 달성함을 보여주며, 앙상블 접근법이 에러를 감소시켜 신뢰성을 더욱 향상시킨다는 것을 나타냅니다. 우리의 접근법은 의료 도메인뿐 아니라 다른 도메인 별 Text-to-SQL 문제에도 적용할 수 있습니다.

- **Technical Details**: 우리의 접근법은 다음 두 단계로 나뉩니다: SQL 생성(SQL Generation)과 SQL 검증(SQL Validation). 첫 번째 단계에서는 프롬프트와 미세 조정을 포함한 다양한 기술을 활용합니다. LLM에 데이터베이스 정보와 질문 관련 컨텍스트를 제공하며, 임베딩 기반 유사성 기술을 사용해 훈련 데이터에서 유사한 질문을 식별합니다. 두 번째 단계에서는 생성된 SQL의 정확도를 평가합니다. 여러 강력한 LLM의 결과를 결합하여 잘못된 SQL 쿼리의 수를 최소화합니다.

- **Performance Highlights**: 우리는 대회에서 2위라는 성과를 거두었습니다. 우리의 두 가지 접근 방식 모두 높은 실행 정확도를 보였으며, 앙상블 접근 방식이 신뢰성을 더욱 향상시켰습니다.



### A Turkish Educational Crossword Puzzle Generator (https://arxiv.org/abs/2405.07035)
Comments:
          This paper has been accepted for presentation at AIED2024 LBR

- **What's New**: 이 논문은 대형 언어 모델(LLMs)을 활용한 최초의 터키어 십자말풀이 생성기를 소개합니다. 이를 위해 두 개의 특별히 만든 데이터셋을 도입하여, 하나는 180,000개 이상의 고유한 답-단서 쌍으로 구성되어 주어진 답에서 관련 단서를 생성하며, 다른 하나는 텍스트와 단어, 카테고리 및 단서 데이터를 포함한 35,000개 이상의 샘플로 특정 텍스트 및 키워드에 대한 단서를 생성하는 데 중점을 둡니다. 이 생성기는 인터랙티브한 교육 도구로서 기억력, 어휘력 및 문제 해결 능력을 향상시키는 데 도움이 됩니다. 터키어 학습 환경에서 십자말풀이의 발전을 촉진하는 새로운 표준을 세우는 중요한 단계입니다.

- **Technical Details**: 이 연구는 두 개의 주요 데이터셋을 만들었습니다. 첫 번째 데이터셋은 온라인 소스에서 수집된 180,000개 이상의 터키어 답-단서 쌍으로 구성되어 있으며, 두 번째 데이터셋은 텍스트, 답, 카테고리 및 단서를 통합한 35,000개의 샘플로 구성되어 있습니다. 이를 통해 LLMs을 활용한 터키어 십자말풀이 단서 생성 모델을 개발했습니다. 주요 사용 사례에는 주어진 답을 바탕으로 십자말풀이를 생성하는 것과 입력된 텍스트에서 키워드와 단서를 자동으로 추출해 십자말풀이를 생성하는 것이 포함됩니다. 다양한 모델(GPT-3.5 Turbo, Llama-2-7b-chat-hf, Llama-2-13b-chat-hf)을 사용하여 미세 조정(fine-tuning)을 통해 단서의 질을 평가하고 최적화했습니다.

- **Performance Highlights**: 두 가지 방법을 통해 터키어 십자말풀이 단서 생성의 실험을 수행했습니다. 전문가들이 제공한 TAC 데이터셋에서 60,000개의 쌍을 활용하여 GPT-3.5 Turbo를 미세 조정했습니다(배치 크기: 16, 학습률: 0.01, 3 에포크). 모델의 성능은 2,135개의 학술 키워드를 사용하여 평가되었으며, 터키어 천연 언어 전문가 두 명이 생성된 단서의 품질을 평가했습니다.




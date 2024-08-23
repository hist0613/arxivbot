New uploads on arXiv(cs.CL)

### Controllable Text Generation for Large Language Models: A Survey (https://arxiv.org/abs/2408.12599)
Comments:
          52 pages, 11 figures, 7 tables, 11 equations

- **What's New**: 이 논문은 자연어 처리(NLP) 분야에서 대규모 언어 모델(LLMs)의 최근 발전과 이를 활용한 제어 가능한 텍스트 생성(CTG) 기술의 중요성에 대해 다루고 있습니다. CTG는 특정 사용자 요구를 충족시키기 위해 안전성, 감정, 주제 일관성 및 언어 스타일과 같은 사전 정의된 제어 조건을 준수하는 텍스트 생성을 목표로 합니다.

- **Technical Details**: 논문에서는 CTG 작업을 내용 제어(content control)와 속성 제어(attribute control)의 두 가지 주요 유형으로 분류하고, 모델 재훈련(model retraining), 미세 조정(fine-tuning), 강화 학습(reinforcement learning), 프롬프트 엔지니어링(prompt engineering), 잠재 공간 조작(latent space manipulation), 디코딩 시간 개입(decoding-time intervention) 등 주요 방법론을 분석합니다. 이러한 방법들의 특성, 장점, 한계 등을 상세히 논의하여 제어된 텍스트 생성을 위한 심층적 통찰을 제공합니다.

- **Performance Highlights**: CTG는 생성된 텍스트가 고품질 기준을 충족하면서도 특정 요구에 맞게 적절히 조정될 수 있도록 합니다. 예를 들어, 안전성을 유지하면서도 유창성, 유용성 및 다양성을 보장합니다. 이는 LLMs가 사용자 요구에 맞춰 더 개인화되고 맥락에 적합한 콘텐츠를 생성할 수 있도록 합니다.



### RuleAlign: Making Large Language Models Better Physicians with Diagnostic Rule Alignmen (https://arxiv.org/abs/2408.12579)
Comments:
          Ongoing work

- **What's New**: 본 논문에서는 의료 분야에서 Large Language Models (LLMs), 예를 들어 GPT-4, MedPaLM-2, Med-Gemini의 성능을 인간 전문가와 비교하고, 전문가와 유사하게 진단을 내리는 데 필요한 문제를 해결하기 위한 RuleAlign 프레임워크를 소개합니다.

- **Technical Details**: RuleAlign 프레임워크는 규칙 기반 진단 규칙에 LLM을 정렬하기 위해 설계되었습니다. 이를 위해 환자와 의사 간의 규칙 기반 대화를 포함하는 의료 대화 데이터셋을 개발하였으며, 선호 학습(preference learning)을 통한 정렬 학습(alignment learning) 접근 방식을 설계하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식이 효과적임을 입증하였습니다. 우리의 연구가 LLMs가 AI 의사로서의 잠재력을 탐구하는 데 영감을 줄 수 있기를 바랍니다.



### Jamba-1.5: Hybrid Transformer-Mamba Models at Sca (https://arxiv.org/abs/2408.12570)
Comments:
          Webpage: this https URL

- **What's New**: Jamba-1.5는 새로운 instruction-tuned 대형 언어 모델로, Jamba 아키텍처를 기반으로 합니다. Jamba는 Transformer와 Mamba의 혼합 아키텍처로, 고처리량과 저메모리 사용을 제공하면서도 Transformer 모델과 동등하거나 더 나은 품질을 유지합니다. 두 가지 모델 크기(Jamba-1.5-Large, Jamba-1.5-Mini)가 발표되었습니다.

- **Technical Details**: Jamba-1.5-Large는 94B의 활성 파라미터를 가지고 있으며, 256K 토큰의 맥락 길이를 지원합니다. 이를 통해 단일 머신에서 8개의 80GB GPU로 처리할 수 있도록 설계되었습니다. 새로운 quantization 기법인 ExpertsInt8을 도입하여 효율적인 추론이 가능해졌습니다.

- **Performance Highlights**: Jamba-1.5 모델은 긴 맥락 평가에서 우수한 성능을 보이며, 특히 RULER 벤치마크에서 256K의 유효 길이를 자랑하는 유일한 모델입니다. 테스트 결과, Jamba-1.5는 최신 모델들과 비교해 KV 캐시 메모리 사용량이 현저히 적으면서도 높은 처리량과 낮은 지연 시간을 제공합니다.



### Towards Evaluating and Building Versatile Large Language Models for Medicin (https://arxiv.org/abs/2408.12547)
- **What's New**: 이번 연구에서는 임상 맥락에서 대규모 언어 모델(LLMs)의 성능을 평가하기 위해 MedS-Bench라는 포괄적인 벤치마크를 제시합니다. 기존의 다양한 선택 질문에 초점을 맞춘 벤치마크와는 달리, MedS-Bench는 임상 보고 요약, 치료 추천, 진단, 개체 인식 및 의학적 개념 설명 등 11개의 고수준 임상 작업을 포함합니다.

- **Technical Details**: 이 연구에서 우리는 MEDITRON, Mistral, InternLM 2, Llama 3, GPT-4, Claude-3.5와 같은 6개의 주요 LLM을 평가했습니다. 몇 가지 샷 프롬프트를 사용하여 복잡한 임상 작업에서 모델들이 어려움을 겪는 결론에 도달했습니다. 이 한계를 극복하기 위해 의학을 위한 대규모 instruction tuning 데이터셋인 MedS-Ins를 개발하였습니다. MedS-Ins는 58개의 의학 기반 언어 코퍼스로 구성되어 총 1350만 개 샘플을 포함하며 122개의 작업으로 나뉩니다.

- **Performance Highlights**: 경량 오픈 소스 의료 언어 모델에 대한 instruction tuning을 통해 새로운 모델인 MMedIns-Llama 3를 생성하였고, 기존의 모델들에 비해 거의 모든 임상 작업에서 우수한 성과를 보였습니다. 이 연구는 임상 문제에 대한 LLM의 활용을 촉진하기 위한 노력의 일환으로 MedS-Ins 데이터셋을 공개하였으며, 연구 커뮤니티에 추가 기여를 요청하고 있습니다.



### The Russian-focused embedders' exploration: ruMTEB benchmark and Russian embedding model design (https://arxiv.org/abs/2408.12503)
- **What's New**: 이번 논문에서는 러시아어에 특화된 새로운 텍스트 임베딩 모델인 ru-en-RoSBERTa와 MTEB의 러시아어 버전인 ruMTEB 벤치마크를 소개합니다. 이 벤치마크는 23개의 텍스트 임베딩 작업을 포함하고 있으며, 이 중 17개는 새로운 작업입니다.

- **Technical Details**: 루-en-RoSBERTa 모델은 러시아어의 텍스트 임베딩 작업을 위해 설계된 모델로, 다국어 임베딩에서 지식 전이(knowledge transfer)를 가능하게 합니다. 또한, ruMTEB 벤치마크는 의미적 텍스트 유사성(semantic textual similarity), 텍스트 분류(text classification), 재순위(re-ranking), 정보 검색(information retrieval) 등 7개의 과제를 포함하고 있습니다.

- **Performance Highlights**: 루-en-RoSBERTa는 최신 러시아어 모델들과 비교했을 때 경합할 수 있는 성능을 보여줍니다. 이 연구의 결과는 러시아어에 대한 현대적이고 효과적인 텍스트 임베딩 방법을 제공함으로써 정보 검색 및 텍스트 유사성 평가 등의 다양한 NLP 작업에 기여할 것으로 기대됩니다.



### GenderCARE: A Comprehensive Framework for Assessing and Reducing Gender Bias in Large Language Models (https://arxiv.org/abs/2408.12494)
- **What's New**: 본 연구는 GenderCARE라는 포괄적인 프레임워크를 도입하여 대형 언어 모델(LLM)에서 성별 편향을 정량화하고 완화하는 혁신적인 기준, 평가 방법 및 감소 기술을 제안합니다.

- **Technical Details**: GenderCARE 프레임워크는 성평등 기준(Criteria for Gender Equality Benchmarks), LLM의 성별 편향 평가(Assessment of Gender Bias in LLMs), LLM의 성별 편향 감소(Reduction of Gender Bias in LLMs), 그리고 평가 메트릭스(Evaluation metrics)로 구성되어 있습니다. GenderPair라는 새로운 쌍 기반 벤치마크를 개발하여 성별 편향 과정을 포괄적으로 평가하고, 트랜스젠더 및 비바이너리 그룹을 포함한 다양한 성별 정체성을 반영합니다.

- **Performance Highlights**: 본 연구의 신뢰성 있는 실험 결과는 17개의 LLM에서 90% 이상의 성별 편향 감소를 보여주고 있으며, 여러 주요 언어 작업에서의 변동성이 2% 이하로 유지되고 있습니다. 이로써 GenderCARE가 LLM에서 공정성과 형평성을 달성하는 데 기여할 것으로 기대됩니다.



### Enhancing Multi-hop Reasoning through Knowledge Erasure in Large Language Model Editing (https://arxiv.org/abs/2408.12456)
- **What's New**: 이 논문은 Large Language Models (LLMs)의 지식 편집(Knowledge Editing) 문제를 다루고 있습니다. 특히, 편집된 모델이 다중 허프 추론(Multi-hop Reasoning)에서 성능 저하를 겪는 이유로 남은 단일 허프 지식이 영향을 미친다는 가설을 설정하고, 이를 검증하기 위한 일련의 실험을 수행했습니다.

- **Technical Details**: 저자들은 잔여 단일 허프 지식이 멀티-허프 질문을 처리할 때 모델이 원래 답변으로 돌아가게 만들 수 있다는 것을 발견했습니다. 이를 바탕으로 Knowledge Erasure for Large Language Model Editing (KELE)라는 새로운 지식 편집 방법을 제안하며, 이는 잔여 지식을 제거하고 새로운 지식을 주입하는 기능을 포함합니다. KELE는 랭크-원 업데이트 프레임워크를 통해 모델 매개변수를 업데이트하는 방식으로 작동합니다.

- **Performance Highlights**: GPT-J와 GPT-2 XL을 기반으로 한 광범위한 실험을 통해 KELE이 다중 허프 추론 능력을 크게 향상시키는 것을 확인했습니다.



### Positional Description for Numerical Normalization (https://arxiv.org/abs/2408.12430)
Comments:
          Published at Interspeech 2024

- **What's New**: 이번 연구에서는 Positional Description Scheme (PDS)를 제안하여 숫자 시퀀스를 위한 전처리를 개선했습니다. 기존의 subword tokenization 알고리즘의 한계를 극복하고 숫자 정규화를 간소화하여 모델 아키텍처를 유지하면서도 학습 효율성을 높였습니다.

- **Technical Details**: PDS는 숫자에 대한 자리값 정보를 통합하여 모델의 숫자 정규화 문제를 다룹니다. 이는 Neural Sequence-to-Sequence (NS2S) 모델의 한계를 해결하며, 그러한 모델들이 처리하기 어려운 다양한 숫자 구조를 배우도록 돕습니다. 또한 PDS는 작은 데이터셋에서도 효과적으로 학습할 수 있도록 설계되었습니다.

- **Performance Highlights**: PDS는 복잡한 산술 작업에서 모델의 정확성을 23%에서 51%까지 향상시킨 것으로 나타났습니다. 기존의 모델보다 소량의 훈련 데이터로도 효과적인 숫자 정규화를 달성할 수 있습니다. 이 연구는 Text-To-Speech 및 음성 인식 프로세스에서도 PDS의 중요성을 강조합니다.



### CLEANANERCorp: Identifying and Correcting Incorrect Labels in the ANERcorp Datas (https://arxiv.org/abs/2408.12362)
Comments:
          Proceedings of the 6th Workshop on Open-Source Arabic Corpora and Processing Tools (OSACT) with Shared Tasks on Arabic LLMs Hallucination and Dialect to MSA Machine Translation @ LREC-COLING 2024

- **What's New**: 본 연구는 Named Entity Recognition(NER) 분야에서 많이 사용되는 아랍어 NER 벤치마크 데이터셋(ANERcorp)의 레이블 오류를 조사하고, 이를 수정하여 CLEANANERCorp라는 새로운 데이터셋을 제안합니다.

- **Technical Details**: 연구는 ANERcorp의 주석 오류, 누락된 레이블 및 불일치 등을 심층적으로 분석했습니다. 이러한 레이블 오류는 머신러닝 모델 훈련에 부정적인 영향을 미치고, 모델 성능 평가 결과에 왜곡을 초래할 수 있습니다.

- **Performance Highlights**: CLEANANERCorp는 연구 커뮤니티에 더 정확하고 일관된 벤치마크 데이터셋으로 제공될 예정입니다.



### Fine-tuning Smaller Language Models for Question Answering over Financial Documents (https://arxiv.org/abs/2408.12337)
- **What's New**: 최근 연구에서는 대형 언어 모델(LLM)로부터 파생된 교육 예시를 통해 작은 언어 모델이 상당한 추론 능력을 획득할 수 있다는 점을 보여주었습니다. 본 논문에서는 금융 분야에 해당하는 질문 답변 문제를 다루며, 다단계 수치 추론을 요구하는 과제가 어떻게 해결될 수 있는지 분석합니다.

- **Technical Details**: 이 연구는 GPT-4를 교사 모델로 활용하여 소형 모델(3.5B, 14B parameters의 phi-3 변형, Mistral 7B, Orca-2)을 훈련시키고, Python 코드 생성을 통해 금융 추론을 수행합니다. 교사 모델이 생성한 코드는 질문 이해, 공식 식별 및 엔티티 추출 등 단계적으로 금융 추론을 정리합니다. 이렇게 생성된 코드는 외부 해석기에 의해 실행됩니다.

- **Performance Highlights**: 본 연구의 결과는 작은 언어 모델의 성능이 교사 모델과 유사해 질 수 있음을 보여줍니다. 정제된 개념 이해와 일관된 추론을 통해 작은 모델이 금융 데이터 형식에 적합하게 엔티티 추출 능력을 향상시킨 것을 입증하였습니다. 또한 상대적으로 작은 데이터셋을 사용하여 금융 추론 능력을 유도할 수 있다는 가설도 증명하였습니다.



### Interactive DualChecker for Mitigating Hallucinations in Distilling Large Language Models (https://arxiv.org/abs/2408.12326)
- **What's New**: 본 논문에서는 LLMs의 허위 정보 문제를 완화하고 교육자 모델과 학생 모델 모두의 성능을 향상시키기 위한 새로운 프레임워크인 DualChecker를 도입합니다. 이 방법은 ContextAligner를 활용하여 사람의 라벨링 기준과 모델 출력을 정렬하고, 동적 체커 시스템을 통해 모델 상호작용을 개선합니다.

- **Technical Details**: DualChecker는 LLM을 사용하여 지식을 증류하는 혁신적인 인터랙티브 프레임워크입니다. ContextAligner가 모델 출력을 인간 라벨에 맞춰 조정하고, 인터랙티브 체커 시스템이 LLM 응답에서 신뢰 점수를 수집하여 낮은 신뢰를 보이는 경우 자세한 정보를 추가하여 일관성을 보장합니다.

- **Performance Highlights**: DualChecker는 실험에서 기존의 최첨단 방법들보다 월등히 우수한 성과를 나타내며, 교육자 모델의 F1 점수를 최대 17% 향상시키고 학생 모델은 10% 향상시키는 결과를 보였습니다. 특히, LLM 예측으로 미세 조정된 학생 모델은 실제 데이터로 미세 조정된 모델과 유사한 성능을 발휘합니다.



### Improving Factuality in Large Language Models via Decoding-Time Hallucinatory and Truthful Comparators (https://arxiv.org/abs/2408.12325)
Comments:
          Hallucination Mitigation in LLMs

- **What's New**: 이번 논문에서는 응답의 허위 인식(hallucination) 문제를 완화하기 위해, Comparator-driven Decoding-Time (CDT) 프레임워크를 제안합니다. 기존의 모델 매개변수 최적화 및 의미적 표현 변경 방식 대신, 허위 및 사실 기반 비교 모델을 통해 다음 토큰 예측을 더욱 사실적인 방향으로 유도합니다.

- **Technical Details**: CDT는 저Rank 적응(LoRA) 방식으로 허위 및 사실 감지 비교 모델을 구축합니다. 여러 작업에 대해 허위 및 사실성을 인식할 수 있는 능력을 갖춘 비교 모델을 통해 각 작업의 허위 패턴을 효과적으로 다룹니다. 이 과정에서 여러 LoRA 어댑터가 서로 다른 전문가로 작용하여 다기능 허위/사실 패턴을 처리할 수 있습니다.

- **Performance Highlights**: 다양한 NLP 벤치마크에서의 실험 결과, CDT 프레임워크는 응답의 사실성을 개선할 뿐 아니라 모델 성능을 크게 향상시켰습니다. 이 프레임워크는 특정 모델 구조나 작업 유형에 구애받지 않으며, 각기 다른 허위 패턴을 제거하는 잠재력을 가지고 있습니다.



### MaVEn: An Effective Multi-granularity Hybrid Visual Encoding Framework for Multimodal Large Language Mod (https://arxiv.org/abs/2408.12321)
- **What's New**: MaVEn은 여러 이미지를 통한 추론 능력을 개선하기 위해 고안된 Multi-granularity Visual Encoding 프레임워크입니다. 기존 Multimodal Large Language Models (MLLMs)는 단일 이미지에 치중해 다중 이미지 정보 통합 능력이 제한되어 있었으나, MaVEn은 이를 해결합니다.

- **Technical Details**: MaVEn은 이산(Discrete) 시각 상징 시퀀스와 기존의 연속(Continuous) 표현 시퀀스를 결합하여 다중 이미지 이해를 용이하게 하는 하이브리드 시각 인코딩 구조로 설계되었습니다. 이 모델은 단일 이미지 상황에 적합하게 설계된 기존 MLLMs의 시각적 인코딩 및 연결 방식을 재구성하여 다중 이미지 입력 처리에 최적화합니다.

- **Performance Highlights**: 실험 결과, MaVEn은 복잡한 다중 이미지 상황에서 MLLMs의 이해 능력을 크게 향상시켰으며, 단일 이미지 상황에서도 성능 개선을 보여줍니다. 또한, 긴 연속 특성에 대한 동적 축소 메커니즘을 채택하여 다중 이미지 처리 효율성을 높였습니다.



### Toward the Evaluation of Large Language Models Considering Score Variance across Instruction Templates (https://arxiv.org/abs/2408.12263)
Comments:
          19 pages, 7 figures

- **What's New**: 이번 연구에서는 LLM (Large Language Model)의 NLU (Natural Language Understanding) 성능을 공정하게 평가하기 위한 새로운 데이터셋과 평가 메트릭인 Sharpe score를 제안합니다. 이는 템플릿 간 점수 변동성을 고려하여 LLM의 성능을 보다 정확하게 측정하는 데 중요합니다.

- **Technical Details**: 평가 방법론으로는, 다양한 instruction templates를 사용하여 LLM의 NLU 성능을 평가했습니다. 우리가 제안한 Sharpe score는 템플릿 간의 변동성을 반영하여 성능 평가를 보다 효과적으로 수행합니다. 이는 영어와 일본어에 대한 크로스-링구얼 (cross-lingual) 데이터셋을 포함하며, 특정 작문 포맷을 반영하도록 정규 표현식 (regular expressions)을 사용해 출력을 제어했습니다.

- **Performance Highlights**: 연구 결과, English와 Japanese LLM에 대한 종합 분석을 통해 템플릿 간의 높은 변동성이 LLM의 공정한 평가에 중요한 영향을 미친다는 것을 발견했습니다. 이에 따라, 새로운 평가 메트릭과 함께 다수의 템플릿을 기반으로 한 평가가 LLM의 NLU 성능을 보다 정확하게 드러낸다고 결론지었습니다.



### A Language-agnostic Model of Child Language Acquisition (https://arxiv.org/abs/2408.12254)
- **What's New**: 이 논문은 영어를 위해 설계된 최근의 의미적 부트스트래핑(child-language acquisition) 모델을 다시 구현하여 새로운 언어인 히브리어를 배우도록 훈련한 연구 결과를 다룹니다. 모델은 발화와 의미 표현의 쌍에서 학습하며, 구문(syntax)과 단어의 의미를 동시에 습득합니다.

- **Technical Details**: 모델은 CCG(combinatory categorial grammar)와 의미적 부트스트래핑 이론에 기반하여, 아동이 언어를 어떻게 배우는지를 시뮬레이션합니다. 데이터는 CHILDES 코퍼스에서 가져온 아동 지향 발화로 구성되며, 보편적 의존 주석을 논리적 형태로 변환하는 최신 방법이 적용되었습니다. 모델은 영어와 히브리어 두 언어에서 테스트되었습니다.

- **Performance Highlights**: 모델은 영어 구문과 의미의 중요한 특징을 성공적으로 학습했으며, 히브리어에서도 높은 정확도로 단어 순서(word order)와 단어 의미를 학습하였습니다. 히브리어의 합성어가 영어보다 더 풍부하여 학습이 느리고 덜 강건하다는 결과를 보였습니다.



### LLMs are not Zero-Shot Reasoners for Biomedical Information Extraction (https://arxiv.org/abs/2408.12249)
Comments:
          11 pages

- **What's New**: 이 논문은 Large Language Models (LLMs)의 의료 분야에서의 성능을 체계적으로 벤치마킹하여 의학적 분류(medical classification)와 Named Entity Recognition (NER) 작업에서의 성과를 평가합니다. 또한, LLM의 수행에 영향을 미치는 다양한 요인의 기여를 분석합니다.

- **Technical Details**: 연구는 BioMistral 및 Llama-2 모델을 포함한 여러 공개 LLM을 대상으로 하며, Chain-of-Thought (CoT), Self-Consistency 기반의 추론 및 Retrieval-Augmented Generation (RAG)을 사용하여 다양한 생물의학 데이터셋을 평가합니다. 연구 결과는 표준 프롬프트가 복잡한 기술보다 일관되게 더 나은 성능을 보임을 보여 주며, 의료 분야에서 CoT, self-consistency, RAG의 현재 적용의 한계를 드러냅니다.

- **Performance Highlights**: 이 연구는 LLMs가 '진정한' 제로샷(zero-shot) 설정에서 성능이 저하되지 않도록 다양한 지식 증강 기법을 조사하고, 매개변수(parametric) 지식 용량이 제로샷 환경에서 성능의 주요 원인임을 발견했습니다. 또한, 모델의 지식 활용에 대한 시스템적 접근이 필요함을 강조합니다.



### EvalYaks: Instruction Tuning Datasets and LoRA Fine-tuned Models for Automated Scoring of CEFR B2 Speaking Assessment Transcripts (https://arxiv.org/abs/2408.12226)
- **What's New**: 이 연구는 e-learning 환경에서 CEFR B2 영어 말하기 평가의 자동화를 목표로 하며, 대규모 언어 모델(LLMs)과 전문가 검증된 합성 대화 데이터셋를 활용하여 평가 정확도를 향상시킵니다.

- **Technical Details**: 연구팀은 CEFR B2 시험을 위해 Mistral Instruct 7B v0.2의 파라미터 효율적인 instruction tuning을 수행하여 EvalYaks라는 모델 패밀리를 개발했습니다. 이 모델들은 각기 다른 섹션의 성능을 평가하며, 단어 및 텍스트의 CEFR 수준을 식별하고 생성할 수 있습니다.

- **Performance Highlights**: EvalYaks 모델은 평균 96%의 허용 가능한 정확도를 달성했으며, 다른 모델보다 3배 성능이 우수한 것으로 나타났습니다. 이는 고품질 CEFR 정렬 평가 데이터를 사용하여 조정된 LLM이 효과적으로 B2 영어 말하기 평가를 수행할 수 있음을 보여 줍니다.



### Large Language Models as Foundations for Next-Gen Dense Retrieval: A Comprehensive Empirical Assessmen (https://arxiv.org/abs/2408.12194)
Comments:
          Submitted to EMNLP24

- **What's New**: 본 연구는 다양한 Dense Retrieval (밀집 검색) 작업에서의 대규모 언어 모델(LLM)의 이점과 비LLM 기반 모델과의 차이를 분석한 포괄적인 실험 연구입니다. 기존의 교육된 언어 모델들과 비교했을 때, LLM들이 더 높은 정확도와 데이터 효율성을 제공함을 보여줍니다.

- **Technical Details**: 이 연구에서는 15개 이상의 서로 다른 LLM 및 비LLM 모델을 평가하였으며, 모델의 파라미터 수는 0.1억에서 32억까지 다양하고, 사전 학습의 충실도도 다양합니다. 연구는 MS MARCO 데이터셋을 활용하여 실험하였으며, 다양한 Retrieval 능력에 대한 평가를 진행했습니다.

- **Performance Highlights**: 연구 결과에 따르면, 모델의 크기가 커질수록 인도메인(in-domain) 정확도와 데이터 효율성이 향상되며, LLM 기반의 모델들이 비LLM에 비해 모든 Retrieval 작업에서 일관되게 성능을 향상시킵니다. 또한 LLM은 제로샷 제너럴리제이션(zero-shot generalization), 긴 Retrieval 일반화, 지시 기반 Retrieval에서 뛰어난 성능을 보임을 발견하였습니다.



### Reasoning Factual Knowledge in Structured Data with Large Language Models (https://arxiv.org/abs/2408.12188)
- **What's New**: 이 논문은 대형 언어 모델(Large Language Models, LLMs)이 구조화된 데이터에서 사실 지식을 추론하는 능력을 평가하기 위해 StructFact라는 벤치마크를 제안합니다. StructFact는 다양한 작업 및 도메인을 포함하는 8,340개의 사실 질문으로 구성되어 있습니다. 이 벤치마크는 LLMs가 구조적 사실로부터 사실 지식을 정확히 추론할 수 있는지 탐색할 수 있는 기회를 제공합니다.

- **Technical Details**: StructFact 벤치마크는 5개의 사실 작업(Arithmetic Calculation, Spatiotemporal Cognition, Multi-hop Reasoning, Composition Understanding, Combining Structured and Unstructured)에 대해 설계되었습니다. 이 작업들은 구조화된 데이터의 고유한 특성에 기반하여 LLMs의 추론 능력을 다양한 방식으로 분석할 수 있게 합니다. 각 질문은 세부 난이도 주석을 포함하여 해당 작업의 특정 초점에 따라 분류되었습니다.

- **Performance Highlights**: 구조화된 데이터로부터 사실 지식을 추론하는 LLMs의 능력에 대한 대규모 실험이 수행되었습니다. 실험 결과, 기존 LLMs는 구조화된 데이터에서 사실 지식을 정확하게 추론하는 데 한계를 보였으며, 이는 특히 의료 및 금융과 같은 고위험 도메인에서의 실용적 활용을 제한하는 요인입니다. StructFact는 이런 한계를 극복하고 LLMs의 실제 적용을 향상시키기 위한 중요한 도구로 자리잡을 것입니다.



### Revisiting the Phenomenon of Syntactic Complexity Convergence on German Dialogue Data (https://arxiv.org/abs/2408.12177)
Comments:
          Accepted to KONVENS 2024

- **What's New**: 이번 논문은 대화 상호작용에서의 구문 복잡성 수렴 현상을 독일어 데이터에 대해 조사하였습니다. 이전 연구에서는 영어 데이터에서만 이 현상을 확인했으며, 본 연구는 독일어에서의 언어적 일반성을 실증적인 결과로 나타내고자 했습니다.

- **Technical Details**: 연구에서는 의존구문 분석을 바탕으로 구문 복잡성을 정량화하였습니다. 구문 복잡성은 단어 수, 구문 노드 수 및 하위 절의 비율을 통해 측정되며, 다양한 독일어 대화 데이터셋을 사용하여 분석이 수행되었습니다.

- **Performance Highlights**: 분석한 세 개의 독일어 데이터셋 중 하나에서 구문 복잡성 수렴이 통계적으로 확인되었으며, 이는 대화 상호작용에서의 언어적 일반성을 시사합니다. 또한, 추가 조사 결과 다른 유형의 구문 복잡성 수렴도 발견되었습니다.



### FIRST: Teach A Reliable Large Language Model Through Efficient Trustworthy Distillation (https://arxiv.org/abs/2408.12168)
- **What's New**: 대형 언어 모델(LLMs)의 신뢰성을 높이는 새로운 방법인 eFfIcient tRustworthy disTillation (FIRST)을 제안합니다. 이 방법은 소량의 교사 모델의 지식을 비용 효율적으로 활용하여 신뢰할 수 있는 언어 모델을 생성합니다.

- **Technical Details**: FIRST 방식은 '집중된 지식(concentrated knowledge)' 현상을 활용하여 상위 5개의 토큰을 선택하여 지식을 전이합니다. 전이 과정에서 '신뢰성 극대화(trustworthy maximization)'를 적용하여 모델의 정확성과 보정 능력을 동시에 강화합니다.

- **Performance Highlights**: 실험 결과, FIRST 방법을 사용하여 평균적으로 정확도가 2.3% 향상되었고, 잘못된 보정(mis-calibration) 정도가 10% 감소하여 신뢰성이 크게 개선되었습니다.



### Preference-Guided Reflective Sampling for Aligning Language Models (https://arxiv.org/abs/2408.12163)
- **What's New**: 이 논문에서는 Preference-Guided Reflective Sampling (PRS)이라는 새로운 샘플링 방법을 제안합니다. 이 방법은 사용자 선호를 자연어로 명시적으로 지정하여 반응 생성을 최적화하는 과정으로 프레임화됩니다.

- **Technical Details**: PRS는 트리 기반 생성 프레임워크를 이용하여 효율적인 샘플링 프로세스를 가능하게 하며, 사용자 선호를 추가적 맥락으로 통합하여 더 관련성 높은 방향으로 모델을 유도하고 불필요한 탐색을 최소화합니다. 이는 생성된 데이터에 대해 반영하여 미래 응답의 샘플링을 개선하는 방식으로 작동합니다.

- **Performance Highlights**: PRS는 다양한 정책 모델을 통해 훈련 데이터를 생성하며, 높은 보상을 기록하였고, 여러 벤치마크에서 성과를 보여주었다. 실험 결과 PRS는 기존 강력한 기준선보다 우수한 성과를 나타냈습니다.



### Implicit Sentiment Analysis Based on Chain of Thought Prompting (https://arxiv.org/abs/2408.12157)
- **What's New**: 이 논문에서는 Chain of Thought (CoT) 개념에 영감을 받아 Sentiment Analysis of Thinking (SAoT) 프레임워크를 제안하였습니다. SAoT는 심리적 의견의 내재적 측면을 이해하고 정서의 극성을 추론하는 데 중점을 두고 있습니다.

- **Technical Details**: SAoT 프레임워크는 일반 상식(common sense)과 사고 체인 능력을 활용하여 텍스트의 내재적 측면 및 의견을 분석합니다. 분석 결과는 ERNIE-Bot-4 모델과 결합하여 실험적으로 평가되었으며, 이를 통해 정서 분석 작업에서 중요한 성과 개선을 입증하였습니다.

- **Performance Highlights**: 실험 결과, SAoT + ERNIE-Bot-4 모델은 레스토랑 데이터셋에서 75.27의 F1 점수와 66.29의 ISA 점수를 달성하였으며, 랩탑 데이터셋에서도 76.50의 F1 점수와 73.46의 ISA 점수를 기록하였습니다. ERNIE-Bot-4 + SAoT 모델은 BERTAsp + SCAPt 기준 모델을 평균 47.99% 초과하여 성능이 뛰어났습니다.



### MDD-5k: A New Diagnostic Conversation Dataset for Mental Disorders Synthesized via Neuro-Symbolic LLM Agents (https://arxiv.org/abs/2408.12142)
- **What's New**: 본 논문에서는 심리적 질환을 진단하기 위한 대화 데이터를 생성하는 새로운 방법론을 제시합니다. 이는 익명의 환자 사례를 활용하여 비밀 유지 및 윤리적 고려 사항을 준수하며, 신경-기호적 다중 에이전트 프레임워크(neuro-symbolic multi-agent framework)를 통해 이루어집니다. 이를 통해 대화의 다양성과 정확성을 동시에 확보합니다.

- **Technical Details**: 제안된 프레임워크는 3가지 종류의 대형 언어 모델(large language model) 에이전트로 구성됩니다: 의사 에이전트(doctor agent), 환자 에이전트(patient agent), 진단 주제 전환을 관리하는 기호 도구 에이전트(symbolic tool agent)입니다. 이 구조는 고정된 증상 질문 트리와 동적인 경험 질문 트리를 통해 상징적 통제(symbolic control) 아래에서 텍스트 생성을 가능하게 합니다.

- **Performance Highlights**: MDD-5k 데이터셋은 1000개의 실제 환자 사례를 기반으로 5000개의 고품질 진단 대화를 포함하며, 이는 중국어로 된 정신 질환 진단 대화 데이터셋 중 최초로 레이블이 부여된 것입니다. 인간 평가 결과, 제안된 데이터셋은 전문성, 의사소통 기술, 유창성, 안전성 면에서 여러 기존 데이터셋보다 우수하다고 입증되었습니다.



### uMedSum: A Unified Framework for Advancing Medical Abstractive Summarization (https://arxiv.org/abs/2408.12095)
Comments:
          12 pages

- **What's New**: 본 논문은 의료 분야의 추상적 요약에서 진정성(faithfulness)과 정보성(informativeness)을 균형 있게 유지하는 방법을 제안합니다. 특히 uMedSum이라는 모듈형 하이브리드 요약 프레임워크를 소개하며, 이 프레임워크는 세 가지 다양한 데이터셋을 기반으로 한 six advanced abstractive summarization 방법의 포괄적인 벤치마크를 제공합니다.

- **Technical Details**: 우리는 confabulation 제거 및 핵심 누락 정보 추가를 위한 새로운 접근 방식을 도입하며, 이를 통해 진정성과 정보성을 모두 보장하는 방법을 개발했습니다. 모델의 판단력(model reasoning)과 자기 개선(self-improvement)을 활용하여 기존 기법들의 한계를 극복합니다. 또한, 우리는 reference-based 및 reference-free metric을 포함한 다섯 가지 표준화된 메트릭을 사용하여 의료 요약의 성능을 평가합니다.

- **Performance Highlights**: 우리의 uMedSum 프레임워크는 이전 SOTA 방법에 비해 평균 11.8%의 상대적인 성능 개선을 달성하였으며, 특히 confabulation이 발생하기 쉬운 어려운 사례에서도 의료 전문가들이 uMedSum의 요약을 6배 더 선호합니다. 이 결과는 다양한 데이터셋과 메트릭에서 uMedSum의 효과성과 일반성을 입증하는 중요한 진전을 나타냅니다.



### High-Quality Data Augmentation for Low-Resource NMT: Combining a Translation Memory, a GAN Generator, and Filtering (https://arxiv.org/abs/2408.12079)
- **What's New**: 본 논문은 저자들이 제안한 새로운 접근법으로, 저자 측 언어의 단일 언어 코퍼스를 활용하여 Neural Machine Translation (NMT) 모델의 성능을 향상시키는 방법을 제시합니다. 또한, Generative Adversarial Network (GAN)을 통해 훈련 데이터의 질을 보장하고 Translation Memory (TM)를 통합하여 데이터를 증가시킴으로써 저자 측 언어에 대한 데이터의 양을 증가시키는 방법도 설명합니다.

- **Technical Details**: 이론적으로 GAN 구조를 활용하여 NMT를 위한 데이터 증강 방안을 연구합니다. TM과 GAN을 통합하여 생성기에서 학습할 수 있는 데이터의 양을 늘리며, 질 높은 번역 결과를 보장하기 위한 새로운 필터링 절차도 제안하고 있습니다. 연구에서 소개된 Euclidean 거리 기반의 유사도 측정을 통해 TM에서 유사한 문장을 검색하여, 이 문장이 생성기에 통합될 수 있도록 설계합니다.

- **Performance Highlights**: 본 연구는 저자 언어의 단일 언어 코퍼스를 활용하여 기존 모델의 전통적인 접근 방식을 넘어서, 효과적으로 NMT 성능을 향상시킬 수 있는 방법을 제시합니다. 실험 결과는 단일 언어 코퍼스와 TM의 통합이 저자 언어 번역의 정확도와 유효성을 높여주는 데 긍정적인 영향을 미친다는 것을 보여주고 있습니다.



### ConflictBank: A Benchmark for Evaluating the Influence of Knowledge Conflicts in LLM (https://arxiv.org/abs/2408.12076)
Comments:
          Under Review

- **What's New**: 본 논문은 ConflictBank라는 체계적인 지식 갈등 평가 기준을 제시하며, 이는 LLMs에서 발생하는 다양한 지식 갈등을 분석하기 위한 최초의 포괄적 벤치마크입니다.

- **Technical Details**: ConflictBank는 세 가지 주요 갈등 원인인 잘못된 정보(misinformation), 시간에 따른 지식 변화(temporal conflict), 언어의 다의성(semantic conflict)들을 포함하여, 7,453,853개의 주장-증거(claim-evidence) 쌍과 553,117개의 QA 쌍을 구성하였습니다.

- **Performance Highlights**: ConflictBank를 기반으로 진행된 파일럿 실험에서는 12개의 LLM 모델에서 다양한 갈등 상황에 대한 모델 동작을 분석하였으며, 지식 갈등 원인, 갈등 유형 및 모델 규모에 대한 통찰을 제공하였습니다.



### Evidence-backed Fact Checking using RAG and Few-Shot In-Context Learning with LLMs (https://arxiv.org/abs/2408.12060)
- **What's New**: 소셜 미디어에서의 정보 조작이 만연함에 따라 자동화된 사실 확인 시스템의 필요성이 강조되고 있습니다. 이 논문에서는 Averitec 데이터셋을 사용하여 온라인 주장에 대한 진실성을 검증하는 자동화된 시스템을 제안합니다. 시스템은 주장을 지원하는 증거를 제공하며, 대규모 언어 모델(LLM)을 사용하여 분류합니다.

- **Technical Details**: 이 시스템은 Retrieve and Generate (RAG) 파이프라인을 통해 관련 증거 문장을 추출하고, In-Context Learning (ICL)을 통해 주장의 진실성을 판단합니다. 주어진 주장과 문서 집합을 활용하여, RAG를 통해 가장 관련성 높은 문서를 검색하고 이를 기반으로 ICL을 적용하여 결과를 도출합니다.

- **Performance Highlights**: 시스템은 Averitec 데이터셋에서 0.33의 점수를 기록하며, 이는 기존 기준선보다 22% 향상된 결과입니다. 최소한의 학습 샘플로 작동 가능하며 다양한 LLM과의 실험을 통해 효과성을 입증합니다.



### Aligning (Medical) LLMs for (Counterfactual) Fairness (https://arxiv.org/abs/2408.12055)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2404.15149

- **What's New**: 최신 논문에서는 Large Language Models (LLMs) 의 의료 및 임상 의사 결정 지원 응용 분야에서의 편향 문제를 해결하기 위한 새로운 모델 정렬 접근 방식을 제시합니다.

- **Technical Details**: 이 연구에서는 지식 증류(knowledge distillation) 프레임워크 내에서 선호 최적화(preference optimization) 방법을 사용하여 LLM의 정렬을 수행합니다. 평가 프레임워크를 통해 다양한 의료 데이터세트 및 태스크에서 LLM의 편향 패턴을 종합적으로 평가합니다.

- **Performance Highlights**: 제안된 완화 기법이 LLM 출력에서의 불공정한 패턴을 효과적으로 줄이는 데 성공했음을 보여주며, 연구 결과는 코드 형태로 공개됩니다.



### Understanding Epistemic Language with a Bayesian Theory of Mind (https://arxiv.org/abs/2408.12022)
Comments:
          21 pages

- **What's New**: 본 논문에서는 인간이 다른 사람의 믿음에 대한 주장을 어떻게 이해하고 평가하는지에 대한 인지 모델인 LaBToM(Language-augmented Bayesian theory-of-mind)을 제안합니다. 이 모델은 행동과 관찰을 기반으로 다른 에이전트의 목표와 믿음, 의도에 대한 베이지안 추론(Bayesian inference)을 토대로 합니다.

- **Technical Details**: LaBToM은 확률적 생성 모델을 기반으로 하는 합리적 행동과 인식을 통해 정의된 두 개의 상호 연결된 모듈로 구성됩니다. 첫 번째 모듈은 다른 사람의 마음을 포함한 세계를 설명하는 개념들을 조합하여 리치한 사고 및 표현을 나타내는 에피스테믹 언어(ELoT)를 모델링합니다. 두 번째 모듈은 민첩한 사고 이론(Bayesian theory-of-mind)으로, 에이전트가 자신의 믿음을 업데이트하고 목표를 향해 행동하는 방식에 대한 직관적 추론을 캡처합니다.

- **Performance Highlights**: LaBToM 모델은 참여자들이 에이전트의 믿음에 대해 작성한 문장에 대한 평가와 높은 상관관계를 보였으며, 화자들의 표현에 대한 인간의 판단을 이해하는 데 중요한 역할을 하며, 기존의 다중 모달 LLM 모델 및 수정된 BToM 모델들과의 비교에서도 우수한 성과를 보였습니다.



### RAG-Optimized Tibetan Tourism LLMs: Enhancing Accuracy and Personalization (https://arxiv.org/abs/2408.12003)
Comments:
          Accepted by AIPR 2024

- **What's New**: 본 연구에서는 티베트 관광에 특화된 LLM(대형 언어 모델)의 최적화 방안을 제안하며, retrieval-augmented generation (RAG) 기술을 기반으로 한 개인 맞춤형 추천 시스템을 개발합니다.

- **Technical Details**: RAG 기술을 활용하여 사용자 요구에 맞게 벡터 데이터베이스에서 관광지를 검색하고, TF-IDF 및 BERT 같은 벡터화 기법을 통해 데이터의 정확성을 높입니다. HNSW 및 LSH 같은 다양한 색인 방법을 사용하여 효율적인 쿼리 처리를 구현합니다.

- **Performance Highlights**: 최적화된 모델은 콘텐츠 생성의 유창성(fluency), 정확성(accuracy), 관련성(relevance)에서 유의미한 개선을 보여주며, Hallucination 문제를 효과적으로 해결하여 개인 맞춤형 추천의 정확도를 높입니다.



### Large Language Models for Page Stream Segmentation (https://arxiv.org/abs/2408.11981)
- **What's New**: 이 논문은 문서 자동 처리의 필수 전제조건인 Page Stream Segmentation (PSS)을 위한 현실적인 공개 벤치마크 부족 문제를 해결하기 위해 TABME++라는 향상된 벤치마크를 소개합니다. 이 벤치마크는 상업용 Optical Character Recognition (OCR) 주석을 특징으로 하고 있습니다.

- **Technical Details**: PSS는 페이지 시퀀스를 원자 문서로 분리하여 정확한 분류와 정보 추출을 가능하게 하는 역할을 합니다. 연구 결과는 대형 언어 모델 (LLMs) 중 디코더 기반 모델이 더 작은 멀티모달 인코더보다 우수한 성능을 보임을 보여줍니다. 논문은 PSS의 기존 연구 및 데이터셋을 검토하며 해당 분야의 주요 도전 과제와 발전을 식별합니다.

- **Performance Highlights**: TABME++ 벤치마크에서 수행한 실험은 Microsoft OCR을 재처리하여 주석의 품질을 향상시켰으며, 이로 인해 데이터셋의 빈 페이지 수가 크게 감소했습니다. 결과적으로, 보다 효과적인 문서 처리 시스템 개발에 중요한 통찰을 제공합니다.



### Decoding SEC Actions: Enforcement Trends through Analyzing Blockchain litigation using LLM-based Thematic Factor Mapping (https://arxiv.org/abs/2408.11961)
- **What's New**: 본 연구는 2012년부터 2024년까지 블록체인 기업에 대한 미국 증권거래위원회(SEC)의 소송을 체계적으로 분석하여, 블록체인 관련 규제를 개선하고자 하는 시도로, SEC의 법적 조치를 유도하는 주제적 요인들을 도출합니다.

- **Technical Details**: 이 연구는 사전 훈련된 언어 모델(pretrained language models, PLMs)과 대규모 언어 모델(large language models, LLMs)을 활용하여 SEC의 모든 컴플레인(complaints)을 주제적 요인(thematic factors)으로 매핑하고, 이들의 연도별 법적 조치에 미치는 영향을 정량화했습니다. 연구는 또한 일반화된 선형 모델(Generalized Linear Model, GLM)을 사용하여 법적 조항의 변화를 분석했습니다.

- **Performance Highlights**: SEC의 집행 조치는 시장 상황과 관계없이 투자자에게 해를 끼치는 사기 및 자금 유용을 지속적으로 단속해왔습니다. 2017-18 및 2021년과 같은 시장 급등기에는 등록되지 않은 증권 공모를 감시하기 위해 기업의 자산 규모에 초점을 맞추었으며, 2020년 이후에는 의무 공시 및 연례 보고서와 같은 다양한 규정으로 범위를 확장했습니다.



### The State of Commercial Automatic French Legal Speech Recognition Systems and their Impact on Court Reporters et a (https://arxiv.org/abs/2408.11940)
- **What's New**: 이 논문은 퀘벡과 캐나다의 법원에서는 법적 절차의 기록이 중요한 작업이며, 공식 법원 기자에 의해 인증되어야 함을 강조합니다. 제한된 자격을 갖춘 기자의 가용성과 수동 전사에 따른 높은 비용은 효율적인 솔루션의 필요성을 부각시킵니다. 이 연구는 법원 기자가 법적 절차 전사에 도움이 될 수 있는 자동 음성 인식 시스템(ASR)에 대한 잠재력을 살펴봅니다.

- **Technical Details**: 연구에서는 AWS의 Amazon Transcribe, Google Cloud Platform의 Speech-to-Text, OpenAI의 Whisper와 같은 세 가지 ASR 모델을 검토합니다. 이 시스템들은 주로 딥 뉴럴 네트워크(Deep Neural Networks)와 트랜스포머(Transformer) 아키텍처를 기반으로 합니다. WER(Word Error Rate) 메트릭을 사용하여 ASR 성능을 평가하고, 음성 인식의 정확성을 확보하기 위해 Sonnex Distance를 도입하였습니다.

- **Performance Highlights**: AWS는 평균 WER 0.15로 전반적인 최고의 성능을 보였으며, GCP와 OpenAI Whisper도 유의미한 결과를 나타냈습니다. 실험 비용은 AWS 약 45 달러, GCP 약 30 달러, OpenAI Whisper는 약 23 달러로 나타났습니다. 모든 모델이 여전히 일정한 오류를 가지므로 생성된 문서를 신중하게 검토할 필요가 있습니다.



### Defining Boundaries: The Impact of Domain Specification on Cross-Language and Cross-Domain Transfer in Machine Translation (https://arxiv.org/abs/2408.11926)
- **What's New**: 최근 신경망 기계 번역(NMT)의 발전이 이루어졌으나, 자원이 적은 언어들은 방대한 평행 코퍼스(parallel corpora)가 부족하여 발전에 한계를 겪고 있습니다. 본 논문에서는 자원이 풍부한 언어에서의 데이터를 활용하는 크로스링구얼(교차언어) 전이 학습을 통해 이 문제를 해결하는 방안을 제시합니다.

- **Technical Details**: 이번 연구에서는 영어를 출발 언어로, 스페인어를 미세 조정(fine-tuning)에 사용하여 포르투갈어, 이탈리아어, 프랑스어, 체코어, 폴란드어, 그리고 그리스어를 평가 목표 언어로 설정하였습니다. 주요 연구 질문은 도메인 특화 품질의 향상, 제로샷(zero-shot) 상황에서의 도메인 전이 가능성 확인, 언어 특화 요인과 도메인 특화 요인이 적응 효과성에 미친 영향 분석입니다.

- **Performance Highlights**: 특화된 분야(의료, 법률, IT)에서의 도메인 특화 번역의 질이 유의미하게 향상된 것으로 나타났으며, 이는 도메인 데이터를 잘 정의하고 실험 설정의 투명성을 유지하는 것이 제로샷 크로스링구얼 도메인 적응에 있어 중요한 역할을 함을 강조합니다.



### Ancient Wisdom, Modern Tools: Exploring Retrieval-Augmented LLMs for Ancient Indian Philosophy (https://arxiv.org/abs/2408.11903)
Comments:
          Best paper at the Workshop on Machine Learning for Ancient Languages @ ACL 2024. Proceedings of the 1st Machine Learning for Ancient Languages Workshop, 2024.ml4al-1.23, Association for Computational Linguistics (ACL) 2024. Dataset, code, and evaluation is available at: this https URL

- **What's New**: 이번 연구는 특정 지식 영역에서의 질문 응답(Long-form Question Answering, LFQA)에 대한 새로운 접근 방식인 Retrieval-Augmented Generation (RAG) 모델의 가능성을 탐구합니다. 특히, 고대 인도 철학인 Advaita Vedanta에 관한 방대한 공공 담론을 활용하여 VedantaNY-10M 데이터셋을 개발하였습니다.

- **Technical Details**: RAG 모델은 정보 검색(retrieval) 및 생성(generation) 기능을 결합하여 사용되며, 비RAG LLM(Local Language Model)과 비교하여 성능을 평가하였습니다. 인간 평가자는 데이터 전사(transcription), 정보 검색(retrieval), 생성(generation) 성능을 기준으로 RAG 모델이 비RAG 모델보다 우수한 결과를 보였다고 보고했습니다.

- **Performance Highlights**: RAG 모델은 사실적이고 포괄적인 응답을 생성하며, 환각(hallucinations)이 현저히 적었습니다. 또한, 고유 저빈도(term) 키워드 기반 하이브리드 검색기(hybrid retriever)를 사용하여 성능을 더욱 개선했습니다.



### Beyond Labels: Aligning Large Language Models with Human-like Reasoning (https://arxiv.org/abs/2408.11879)
Comments:
          Accepted in ICPR 2024

- **What's New**: 이 연구에서는 윤리적 추론을 생성을 지원하는 새로운 데이터셋인 'Dataset for Aligning Reasons (DFAR)'를 소개합니다. 이 데이터셋은 윤리적 및 비윤리적 진술과 그에 대한 설명을 포함하고 있어 자연어처리(NLP)의 인간적 결정을 더 잘 반영할 수 있도록 돕습니다.

- **Technical Details**: DFAR에는 2886개의 윤리적 샘플(57.7%)과 2114개의 비윤리적 샘플(42.3%)이 포함되어 있으며, 12명의 주석자가 주석을 달았습니다. 연구는 라벨(Labels)과 그에 맞는 설명(Reasons)을 모두 사용하는 독특한 미세 조정(fine-tuning) 방법을 적용하였으며, 이 방식은 기존의 미세 조정 방식과 구별됩니다.

- **Performance Highlights**: 새로운 미세 조정 방법은 윤리-비윤리 분류 작업과 이유 생성 작업에서 다른 방법들보다 뛰어난 성능을 보였으며, 분류 작업의 정확도가 상당히 증가하고 이유 생성 작업에서의 잘못된 정렬 비율이 감소했습니다. 이는 L+R 방식의 미세 조정이 LLM이 인간 윤리에 더욱 잘 align되도록 만든다는 것을 보여줍니다.



### Open-FinLLMs: Open Multimodal Large Language Models for Financial Applications (https://arxiv.org/abs/2408.11878)
Comments:
          33 pages, 13 figures

- **What's New**: Open-FinLLMs 시리즈, FinLLaMA와 FinLLaVA를 포함해 금융 응용 분야에 특화된 대형 언어 모델을 소개합니다. 기존 모델들이 가진 한계를 극복하기 위해, 대규모 금융 텍스트 및 멀티모달 데이터를 활용하여 폭넓은 금융 지식을 집약했습니다.

- **Technical Details**: FinLLaMA는 520억 토큰으로 구성된 금융 데이터셋에서 사전 훈련되었으며, 573K 개의 금융 지침으로 추가 훈련되어 FinLLaMA-instruct로 발전했습니다. FinLLaVA는 143만 개의 이미지-텍스트 지침을 통해 멀티모달 프로세싱을 강화했습니다. 이는 기존 LLM들과 비교해 강화된 분석 능력을 보여줍니다.

- **Performance Highlights**: FinLLaMA는 LLaMA3와 BloombergGPT보다 우수한 성능을 기록했으며, FinLLaMA-instruct는 GPT-4와 다른 금융 LLM을 초과 성능을 보였습니다. FinLLaVA는 테이블 및 차트 해석에서 뛰어난 능력을 발휘하여 모든 멀티모달 벤치마크를 초과했습니다.



### Hierarchical Retrieval-Augmented Generation Model with Rethink for Multi-hop Question Answering (https://arxiv.org/abs/2408.11875)
Comments:
          undereview

- **What's New**: 본 논문에서는 Multi-hop Question Answering (QA)에 대한 새로운 프레임워크인 Hierarchical Retrieval-Augmented Generation Model with Rethink (HiRAG)을 제안합니다. 이 모델은 Decomposer, Definer, Retriever, Filter, Summarizer의 다섯 가지 모듈로 구성되어 있으며, 퀘스천의 서브퀘스천을 효과적으로 처리하는 새로운 계층적 검색 전략을 도입합니다.

- **Technical Details**: HiRAG는 다단계 검색을 수행하며, 문서 수준에서의 sparse retrieval과 청크 수준에서의 dense retrieval을 통합하여 두 가지 방법의 장점을 활용합니다. 또한, single-candidate retrieval 방식을 도입하여 다수의 후보 검색의 한계를 극복하고, 부정확한 답변이 발견될 경우 Rethink 모듈을 통해 추가 청크를 선택합니다. 논문에서는 Indexed Wikicorpus와 Profile Wikicorpus라는 두 개의 새로운 데이터베이스도 구성하였습니다.

- **Performance Highlights**: HiRAG는 HotPotQA, 2WikiMultihopQA, MuSiQue, Bamboogle의 네 가지 데이터 세트에서 실험을 수행하였으며, 대부분의 메트릭에서 최첨단 모델을 능가하는 성능을 보였습니다. 특히, Indexed Wikicorpus는 효과적인 데이터베이스로 확인되었습니다.



### MegaFake: A Theory-Driven Dataset of Fake News Generated by Large Language Models (https://arxiv.org/abs/2408.11871)
- **What's New**: 이번 연구는 LLM(대형 언어 모델) 기반의 가짜 뉴스 생성 메커니즘을 분석하고, 이를 지원하는 이론적 프레임워크인 LLM-Fake Theory를 개발했습니다. 가짜 뉴스 생성을 위한 자동화된 파이프라인을 도입하여 수동 주석의 필요성을 없앴으며, MegaFake라는 대규모 기계 생성 가짜 뉴스 데이터셋을 만들었습니다.

- **Technical Details**: LLM-Fake Theory는 네 가지 방법을 통해 LLM을 활용하여 가짜 뉴스를 생성하는 방법을 설명하며, 각 방법은 사회 심리학 이론에 의해 지지됩니다. 이 연구는 GossipCop 데이터셋을 기반으로 46,096개의 가짜 뉴스와 17,871개의 합법적 뉴스를 포함하는 MegaFake 데이터셋을 개발했습니다. 이를 통해 LLM의 가짜 뉴스 생성에 대한 신뢰도를 높였습니다.

- **Performance Highlights**: MegaFake 데이터셋은 가짜 뉴스 탐지 모델들이 일반적으로 나타내는 예측 편향을 줄여주는 것으로 나타났습니다. 실험 결과, 자연어 이해(NLU) 모델이 자연어 생성(NLG) 모델보다 뛰어난 성능을 보였으며, 흥미롭게도 작은 LLM이 대형 LLM보다도 합법 뉴스와 가짜 뉴스를 분류하는 데 더 높은 성능을 나타냈습니다.



### Enhance Lifelong Model Editing with Continuous Data-Adapter Association (https://arxiv.org/abs/2408.11869)
Comments:
          Preprint. Under Review

- **What's New**: ELDER(Enhancing Lifelong moDel Editing with mixtuRe of Low-Rank Adapter)는 반복적인 모델 편집 과정에서 발생하는 지식 손실 문제를 해결하기 위해 최적화된 새로운 접근법입니다. 이 방법은 여러 LoRA(Low-Rank Adapter)를 통합하여 데이터를 원활하게 연결하고, 기존의 모델 수정 방법들의 단점을 보완합니다.

- **Technical Details**: ELDER는 다운스트림 작업에서 LLM의 일반적인 성능을 유지하며, 지속적인 입력을 위한 로터 네트워크를 통해 LoRA를 조합하여 작동합니다. 중요한 점은, LoRA 가중치가 수동으로 설정되는 것이 아니라, end-to-end 학습을 통해 적응적으로 생성된다는 것입니다. 이를 통해 데이터와 적절한 어댑터 간의 관계를 학습합니다.

- **Performance Highlights**: ELDER는 GPT-2 XL과 LLaMA2-7B 벤치마크에서 10% 이상 높은 편집 성능을 기록하며, 주어진 데이터에 대해 더 나은 일반화 성능을 나타냅니다. 또한, 이전 편집을 신뢰성 있게 유지하고, 후속 작업에서도 LLM의 성능을 보장합니다.



### Improving embedding with contrastive fine-tuning on small datasets with expert-augmented scores (https://arxiv.org/abs/2408.11868)
- **What's New**: 이 논문에서는 전문가 점수가 추가된 작은 데이터셋을 통해 텍스트 임베딩 모델을 개선하는 대비 미세 조정 방법론을 제안합니다. 이 방법은 의미적 텍스트 유사성 태스크를 향상시키고 텍스트 검색 문제를 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: 제안된 방법은 전문가 점수를 통해 도출된 소프트 레이블을 사용하여 임베딩 모델을 미세 조정합니다. 이를 통해 모델의 다양성을 유지하면서 모델의 검색 성능이 향상될 수 있도록 합니다. 이 연구에서는 온라인 쇼핑 웹사이트의 Q&A 데이터셋과 8개의 전문가 모델을 사용하여 방법론을 평가했습니다.

- **Performance Highlights**: 논문의 결과는 여러 검색 작업에서 벤치마크 모델에 비해 향상된 성능을 보여주며, 이는 대규모 텍스트 임베딩 벤치마크(MTEB)에서 다양한 메트릭을 기반으로 평가되었습니다. 이 방법은 라벨이 부족한 실제 응용 프로그램에 특히 비용 효율적이고 실용적입니다.



### Crossing New Frontiers: Knowledge-Augmented Large Language Model Prompting for Zero-Shot Text-Based De Novo Molecule Design (https://arxiv.org/abs/2408.11866)
Comments:
          Paper was accepted at R0-FoMo: Robustness of Few-shot and Zero-shot Learning in Foundation Models, NeurIPS-2023. Please find the links: this https URL and this https URL

- **What's New**: 본 연구에서는 대규모 언어 모델(LLMs)을 활용한 텍스트 기반 분자 설계를 위한 새로운 양식을 소개합니다. 이는 지식 보강(knowledge-augmented) 프롬프트를 사용하여 제로샷(Zero-shot) 조건부 de novo 분자 생성을 위한 접근법으로, 기존 방법보다 월등한 성능을 보여줍니다.

- **Technical Details**: 제안된 방법론에서는 LLM들이 생성하는 예측 결과에서 기술적 설명을 이용하여 작은 규모의 언어 모델(예: DeBERTa)을 미세 조정(fine-tuning) 합니다. 이 과정에서 각 모델의 선택적 설명과 기술적 설명을 바탕으로 컨텍스트 인지 토큰 임베딩을 계산합니다. 최종적으로 변환된 임베딩을 통합하여 화학 SMILES 표현을 생성하는 트랜스포머 디코더에 입력합니다.

- **Performance Highlights**: 제안된 프레임워크는 벤치마크 데이터셋에서 기존의 최첨단(SOTA) 모델들을 초과하는 성능을 인정받았으며, 실험 결과가 그 효과성을 뒷받침합니다. 이를 통해 텍스트 기반 분자 설계 작업에서의 새로운 가능성을 제시합니다.



### How Susceptible are LLMs to Influence in Prompts? (https://arxiv.org/abs/2408.11865)
- **What's New**: 이번 연구는 여러 개의 질문-응답 작업에서 LLMs의 프롬프트 민감성을 조사하며, 다른 모델의 추가 입력이 LLM의 응답에 미치는 영향을 분석합니다.

- **Technical Details**: 본 연구에서는 Llama2, Mixtral, Falcon과 같은 여러 LLM 모델이 다른 모델로부터 받은 예측 및 설명을 포함한 경우 어떤 식으로 선택 질문에 대한 응답이 변화하는지를 조사합니다. 특히, 설명의 존재, 출처의 권위성, 추가 입력의 신뢰도가 LLM의 반응에 미치는 영향을 연구합니다.

- **Performance Highlights**: 연구 결과, LLM은 제공된 추가 입력의 품질에 관계없이 강한 영향을 받으며, 특히 해당 입력이 권위있거나 신뢰도로 제시될 경우 더욱 쉽게 설득되는 경향을 보입니다. 이는 LLMs의 신뢰성을 확보하기 위한 중요한 경고를 제시합니다.



### Sentiment analysis of preservice teachers' reflections using a large language mod (https://arxiv.org/abs/2408.11862)
Comments:
          5 pages, 2 tables, WAIE 2024 (2024 6th International Workshop on Artificial Intelligence and Education)

- **What's New**: 본 연구에서는 예비 교사들의 반성(Reflection)에 대한 감정(Emotion)과 톤(Tone)을 LLMs(GPT-4, Gemini, BERT)를 활용한 감정 분석(Sentiment Analysis)으로 분석하였습니다. 각 도구가 개별 반성과 여러 반성을 어떻게 분류하고 서술하는지를 비교하였습니다.

- **Technical Details**: 연구는 교사 교육(Teacher Education)에서의 반성적인 실천(Reflective Practices)에 대한 질적(Qualitative), 양적(Quantitative), 계산적(Computational) 분석 간의 간극을 메우는 방법을 탐구하고자 합니다. LLM 분석을 효과적으로 통합하기 위해서는 예비 교사 및 교사 교육자를 위한 포괄적(Comprehensive)이며 관련성(Relevant)이 있는 분석 방법 및 결과 형식 개발이 중요합니다.

- **Performance Highlights**: 이 연구의 결과는 예비 교사의 반성에 대한 LLM의 분석이 어떻게 이루어지는지를 이해하고, 교사 교육의 실천에 적용할 수 있는 가능성을 제시합니다.



### Speaking the Same Language: Leveraging LLMs in Standardizing Clinical Data for AI (https://arxiv.org/abs/2408.11861)
Comments:
          11 pages, 2 figures, 4 tables

- **What's New**: 본 연구에서는 의료 데이터 표준화를 위한 대형 언어 모델(Large Language Models, LLM)을 활용하여 AI 통합 과정을 용이하게 하고 환자 관리 품질을 향상시키는 방법을 제안합니다.

- **Technical Details**: 연구는 Fast Healthcare Interoperability Resources (FHIR) 표준을 기반으로 하여 LLM을 사용한 데이터 매핑 과정을 상세히 설명합니다. 여기에서 우리는 LLM이 데이터를 일관된 형식으로 표준화하는 데 있어 어떻게 기여할 수 있는지를 보여줍니다.

- **Performance Highlights**: LLM을 활용한 결과, 수작업 데이터 큐레이션의 필요성을 대폭 줄이고 데이터 표준화 과정의 효율성을 높일 수 있음을 입증하였습니다. 이는 AI가 의료 분야에 통합되는 속도를 가속화하고, 필요한 시간 및 재정 자원을 최소화하는 데 기여합니다.



### Risks and NLP Design: A Case Study on Procedural Document QA (https://arxiv.org/abs/2408.11860)
- **What's New**: 이 논문은 NLP 시스템의 대규모 배포와 관련된 사용자 위험 및 해를 보다 구체적인 애플리케이션에 맞춰 분석하는 것이 중요하다고 주장하고 있습니다. 특히 요리 레시피 문서(ProcDocQA)의 질문 응답을 사례로 사용하여, 사용자에게 명확한 위험 평가와 완화 전략을 제시합니다.

- **Technical Details**: 저자들은 Risk-Aware Design Questionnaire (RADQ)를 소개하며, 이를 통해 ProcDocQA 시스템 설계를 돕고 사용자 상황에 따라 위험을 다루는 방법을 체계적으로 평가할 수 있음을 강조합니다. 본 연구는 '제로샷' 모드에서 수행된 GPT-3 모델이 웹에서의 인간 응답 대비 동등하거나 우수한 성능을 보여주며, 최적의 시스템 설계 안을 도출하는 방법론을 제시합니다.

- **Performance Highlights**: 요리 레시피에 관한 사례 연구에서, GPT-3는 인간의 응답과 비교하여 동등한 성능을 발휘했지만, 모델 응답의 다양한 오류를 통해 해를 줄이기 위한 특정한 개선 방향이 필요함을 확인하였습니다. 또한, 질문의 맥락과 사용자 경험에 기반한 분석을 통해 시스템 디자인에 대한 새로운 통찰을 제공하고 있습니다.



### Convexity-based Pruning of Speech Representation Models (https://arxiv.org/abs/2408.11858)
- **What's New**: 본 논문에서는 음성 인식 모델에서 레이어 프루닝(layer pruning) 기술을 적용하여 필요한 계산량을 대폭 줄이는 방법을 제안합니다. 또한, convexity(볼록성) 기준을 기반으로 프루닝 결정을 내리며, 이는 사전 훈련된 모델의 성능 손실 없이도 모델의 크기를 줄일 수 있게 합니다.

- **Technical Details**: 저자들은 음성 표현 모델에서의 latent representations(잠재 표현)의 볼록성에 대해 심도 있는 분석을 실시하고, 훈련 중 레이어를 정적(layer-wise)으로 프루닝하는 방법을 제안합니다. 그래프 볼록성 점수를 계산하여 프루닝할 레이어 수를 정하는 방법을 소개하며, 이 방식은 기존의 프리트레인 모델을 변경하지 않고 사용할 수 있도록 합니다.

- **Performance Highlights**: 연구 결과, 프루닝을 통해 최대 70%의 모델 크기 감소와 20-50%의 훈련시간 단축, 25-60%의 추론시간 단축을 달성했습니다. 또한, 프루닝된 모델이 잡힌 부분에서 기존 모델보다 더 나은 성능을 발휘하는 경우도 나타났습니다.



### Hermes 3 Technical Repor (https://arxiv.org/abs/2408.11857)
- **What's New**: Hermes 3는 중립적으로 정렬된 일반ist instruct 및 도구 사용 모델로, 강력한 추론(reasoning) 및 창의성(creative) 능력을 갖추고 있습니다.

- **Technical Details**: Hermes 3는 'instruct-tuned' 모델로, 강력한 퍼포먼스를 나타내는 405B 파라미터의 가장 큰 버전을 포함하고 있습니다.

- **Performance Highlights**: Hermes 3 405B는 여러 공개 벤치마크(public benchmarks)에서 오픈 가중치 모델(open weight models) 중 가장 뛰어난 성과를 기록하고 있습니다.



### Dynamic Adaptive Optimization for Effective Sentiment Analysis Fine-Tuning on Large Language Models (https://arxiv.org/abs/2408.11856)
- **What's New**: 본 연구는 다이나믹 어댑티브 최적화(DAO) 모듈이 포함된 새로운 다중 작업(multi-task) 학습 프레임워크를 제안합니다. 이 모듈은 기존 모델에 원활하게 통합할 수 있는 플러그 앤 플레이 구성 요소로 설계되었습니다.

- **Technical Details**: DAO 모듈의 핵심 요소는 동적 어댑티브 손실(dynamic adaptive loss)로, 이는 훈련 중 각 작업의 상대적 중요성과 데이터의 특성에 따라 다른 작업에 할당된 가중치를 동적으로 조정합니다.

- **Performance Highlights**: 제안된 프레임워크는 평균 제곱 오차(Mean Squared Error, MSE)와 정확도(Accuracy, ACC)를 각각 15.58% 및 1.24% 향상시켰습니다. 이로 인해 기존 연구에 비해 뛰어난 성능을 보여주었습니다.



### FactorLLM: Factorizing Knowledge via Mixture of Experts for Large Language Models (https://arxiv.org/abs/2408.11855)
- **What's New**: 최근 연구에 따르면, 대형 언어 모델(LLM)에서 Feed-Forward Networks(FFN)가 다양한 언어 및 사실적 지식을 저장하는 데 중요한 역할을 한다고 합니다. 기존 방법들은 단일한 구조와 중복된 아키텍처로 인해 지식 혼란을 겪고 있으며, 이에 따라 LLM에 대한 보다 효율적이고 컴퓨팅 자원을 최소화하는 해결책이 요구되고 있습니다. 본 논문에서는 FFN 계산 패러다임을 탐구하고, FactorLLM을 소개하여 훈련된 FFN을 더욱 효율적으로 분해합니다.

- **Technical Details**: FactorLLM은 훈련된 밀집형 FFN을 수정 없이 희소한 서브 네트워크로 분해하는 새로운 접근 방식을 제안합니다. 이 과정에서 Mixture-of-Experts(MoE) 아키텍처의 라우터를 통합하여 전문가의 동적 활성화와 지식 적응을 촉진하고, 최소한의 훈련 데이터와 미세 조정 단계로 성능을 향상시킵니다. 또한 Prior-Approximate(PA) 손실 항을 도입하여 LLM의 효율성을 높입니다.

- **Performance Highlights**: FactorLLM은 다양한 벤치마크에서 실험을 통해 이러한 새로운 접근 방식이 기존의 기법들보다 30% 이상의 연산 비용 절감을 이루면서도 원 모델의 성능 85%를 유지했다는 것을 입증했습니다. 이러한 결과는 자원 제약이 있는 상황에서도 빠른 배치를 가능하게 합니다.



### When Raw Data Prevails: Are Large Language Model Embeddings Effective in Numerical Data Representation for Medical Machine Learning Applications? (https://arxiv.org/abs/2408.11854)
Comments:
          Under review

- **What's New**: 이번 연구는 Large Language Models (LLMs)의 마지막 은닉 상태에서 도출된 벡터 표현을 전자 건강 기록(EHR) 데이터를 활용하여 의학 진단 및 예측에 대한 효과를 분석했습니다. LLM이 생성한 특징을 전통적인 기계 학습 알고리즘의 입력으로 사용했을 때의 성능을 비교하여, ML 작업에서 원시 데이터 입력이 여전히 우세함을 확인했습니다.

- **Technical Details**: 의료 ML 작업에 필요한 표시된 EHR 숫자 데이터 표현에 대한 LLM을 활용하였으며, eXtreme Gradient Boosting(XGB)와 같은 전통적인 ML 알고리즘과의 성능 비교를 수행했습니다. 연구에서는 zero-shot LLMs의 특징 추출기 활용과 프롬프트 엔지니어링 기법들을 포함하여 다양한 조건과 embedding 메소드의 영향 평가를 실시했습니다.

- **Performance Highlights**: 결과적으로, LLM 특징과 XGB 분류기를 결합한 경우 일부 작업에서 전통적인 원시 데이터 특징과 유사한 성능을 달성했지만, 여전히 성능 차이가 존재하여 진행 중인 연구가 필요하다는 필요성을 강조했습니다.



### PyMarian: Fast Neural Machine Translation and Evaluation in Python (https://arxiv.org/abs/2408.11853)
- **What's New**: 본 논문은 Python 인터페이스를 Marian NMT의 C++ 기반 툴킷에 추가하여 머신 번역과 관련된 다양한 애플리케이션과 성능 개선을 도모하는 내용을 다루고 있습니다.

- **Technical Details**: PyMarian 패키지는 Python에서 Marian C++ API를 호출하는 Python 바인딩을 제공합니다. 이 패키지는 모델의 학습, 추론, 평가를 용이하게 하기 위한 편리한 고급 API를 포함하고 있으며, `Translator`, `Trainer`, `Evaluator`의 세 가지 주요 클래스를 제공합니다. 특히 COMET 메트릭을 Python에서 Marian의 추론 엔진을 이용하여 계산할 수 있도록 하며, 최대 7.8배의 속도 향상을 이루어냈습니다.

- **Performance Highlights**: PyMarian을 사용한 기존 메트릭과 비교하여 RAM과 속도 면에서 상당한 개선을 보여줍니다. 예를 들어, COMET 메트릭의 경우, 기존 구현이 27GB의 RAM을 소비하고 530초가 걸리는 반면, pymarian-eval은 절반의 메모리와 단 12초면 동일한 결과를 얻을 수 있습니다. 이러한 결과는 특히 대규모 체크포인트를 다룰 때 더 큰 장점을 제공합니다.



### Fast Training Dataset Attribution via In-Context Learning (https://arxiv.org/abs/2408.11852)
- **What's New**: 본 논문은 in-context learning 및 prompt engineering 기법을 활용하여 instruction-tuned large language models (LLMs)에서 훈련 데이터의 기여도를 추정하는 두 가지 새로운 접근 방식을 제안합니다.

- **Technical Details**: (1) similarity-based 접근 방식은 제공된 컨텍스트가 있는 LLM 출력과 없는 LLM 출력을 비교하여 차이를 측정합니다. (2) mixture distribution model 접근 방식은 기여 점수를 식별하는 문제를 행렬 분해(matrix factorization) 문제로 변형합니다.

- **Performance Highlights**: 실험 결과, mixture model 접근 방식이 RAG 시스템 내에서의 검색 노이즈에 대해 더 강건하며, 데이터를 기여도를 보다 정확히 추정하는 데 효과적임을 입증했습니다.



### Parallel Speculative Decoding with Adaptive Draft Length (https://arxiv.org/abs/2408.11850)
- **What's New**: 본 논문에서는 새로운 추론 가속화 프레임워크인 PEARL(Parallel spEculative decoding with Adaptive dRaft Length)를 제안합니다. 이 프레임워크는 상호 대기 문제(mutual waiting problem)를 해결하고, 드래프트 길이를 상황에 맞게 조절할 수 있도록 해줍니다.

- **Technical Details**: PEARL은 드래프팅 단계에서 첫 번째 드래프트 토큰을 미리 검증하는 'pre-verify'와 검증 단계에서 더 많은 드래프트 토큰을 생성하는 'post-verify' 전략을 포함합니다. 이 두 가지 전략을 통해 드래프팅 단계와 검증 단계를 병렬로 처리할 수 있어 상호 대기 문제를 효과적으로 완화합니다. 이론적으로 PEARL의 평균 허용 토큰 수가 기존의 'draft-then-verify' 기법보다 더 많다는 것을 증명합니다.

- **Performance Highlights**: PEARL은 다양한 텍스트 생성 벤치마크에서 우수한 성능을 보이며, 오토 레그레시브 디코딩 및 기존의 스펙펙티브 디코딩 대비 각각 최대 3.79배 및 1.52배의 속도 향상을 달성했습니다.



### Style-Talker: Finetuning Audio Language Model and Style-Based Text-to-Speech Model for Fast Spoken Dialogue Generation (https://arxiv.org/abs/2408.11849)
Comments:
          CoLM 2024

- **What's New**: 이 논문에서는 Style-Talker라는 혁신적인 프레임워크를 소개합니다. 이 시스템은 음성 입력을 기반으로 한 빠른 대화 생성을 가능하게 하며, 오디오 LLM과 스타일 기반 TTS 모델을 함께 조정하여 사용자의 입력 음성을 처리합니다.

- **Technical Details**: Style-Talker는 자동 음성 인식 (ASR), 대화 생성, 텍스트-음성 변환 (TTS) 모델을 통합하여 실시간 대화 시스템을 구현합니다. 이 시스템은 입력 음성에서 기계적으로 얻어진 텍스트 및 스타일 정보를 활용하여 응답을 생성하며, 이후 TTS 모델을 통해 음성을 합성합니다.

- **Performance Highlights**: 실험 결과, Style-Talker는 기존의 캐스케이드 ASR-LLM-TTS 시스템보다 50% 이상 빠르며, 자연스러움과 일관성 측면에서도 우수한 성능을 보였습니다. 이 시스템은 실제 데이터셋에서 직접 활용할 수 있어 다양한 적용 가능성을 높였습니다.



### MGH Radiology Llama: A Llama 3 70B Model for Radiology (https://arxiv.org/abs/2408.11848)
Comments:
          11 pages, 3 figures, 1 table

- **What's New**: 이번 논문에서는 MGH Radiology Llama라는 첨단 방사선학 중심의 대형 언어 모델(Large Language Model, LLM)을 소개합니다. 이 모델은 Llama 3 70B 모델을 기반으로 하여, 방사선학에 특화된 보고서 생성, 임상 결정 지원 및 환자 커뮤니케이션 도움을 제공합니다. 650만 개 이상의 비식별화된 의료 보고서로 학습하였으며, 진단 정확도와 병원 근무 효율성을 향상시키는 데 중점을 두고 있습니다.

- **Technical Details**: 모델은 Massachusetts General Hospital(MGH)의 다각화된 데이터셋을 기반으로 하고 있으며, CT, MRI, X-ray 및 Fluoroscopic 이미지를 포함한 다양한 영상 모달리티의 데이터를 포함합니다. 데이터 전처리와 함께, Llama 3 70B 모델의 완전한 미세 조정(fully fine-tuning) 및 QLoRA 미세 조정 방법을 사용하여 최적화를 진행합니다.

- **Performance Highlights**: 전통적인 평가 지표인 ROUGE 및 BERTScore와 함께 GPT-4 기반 평가를 결합하여 모델의 성능을 강조하였습니다. 성능 평가 결과, 기존 일반 언어 모델에 비해 방사선학에 특화된 더욱 정확하고 임상적으로 유용한 인상을 생성하는 성능 개선이 확인되었습니다.



### Prompto: An open source library for asynchronous querying of LLM endpoints (https://arxiv.org/abs/2408.11847)
- **What's New**: 최근 대형 언어 모델(Large Language Model, LLM)의 출현으로 연구 분야에 새로운 기회가 열렸지만, 이러한 모델들과의 효율적인 인터랙션은 여전히 도전 과제가 되고 있습니다. 이에 대한 해결책으로, 여러 LLM API에 비동기적으로 접근할 수 있는 툴인 'prompto'라는 파이썬 라이브러리가 개발되었습니다.

- **Technical Details**: prompto는 여러 LLM API와의 비동기 질의를 지원하는 오픈 소스 라이브러리입니다. 사용자가 여러 요청을 동시에 전송할 수 있도록 하여 대기 시간을 줄이고 효율성을 극대화합니다. 현재 OpenAI, Gemini, Anthropic 등 다양한 LLM API를 지원하며, 새로운 API와 모델을 쉽게 통합할 수 있는 확장성 있는 코드베이스를 가지고 있습니다.

- **Performance Highlights**: prompto는 실험의 재현성을 촉진하기 위해 모든 입력 및 프롬프트를 단일 JSONL 파일 내에 정의할 수 있도록 설계되었습니다. 이를 통해 다양한 API와 모델의 쿼리를 처리하면서 효율성을 극대화할 수 있습니다. 연구자들은 여러 모델을 동시에 실험하고 특정 작업에 대한 성능을 비교할 수 있게 되어, 초기 탐색 및 신속한 비교가 용이해졌습니다.



### Density Matrices for Metaphor Understanding (https://arxiv.org/abs/2408.11846)
Comments:
          In Proceedings QPL 2024, arXiv:2408.05113

- **What's New**: 이번 논문에서는 물리학의 밀도 행렬(density matrix) 개념을 활용하여 은유(metaphor)를 포함한 어휘적 모호성(lexical ambiguity)을 모델링하는 방법을 제안합니다.

- **Technical Details**: 은유를 어휘적 모호성의 일종으로 간주하여, 단어의 의미 혼합(mixture of word senses)을 통해 은유적 의미를 모델링할 수 있는 가능성을 탐색합니다. 연구 결과, 은유 모델링은 다른 종류의 어휘적 모호성보다 훨씬 더 어렵다는 점을 발견했습니다.

- **Performance Highlights**: 최고 성능의 밀도 행렬 방법이 간단한 기준선(baselines) 및 일부 신경망 언어 모델(neural language models)을 초월하여 우수한 성과를 보였습니다.



### LLaMA based Punctuation Restoration With Forward Pass Only Decoding (https://arxiv.org/abs/2408.11845)
- **What's New**: 본 논문은 Large Language Model Annotation 분야에서 구두점 복원(punctuation restoration) 작업의 두 가지 진전을 소개합니다. 첫 번째 기여는 LLaMA를 활용한 구두점 복원으로, 기존 벤치마크에 비해 우수한 성능을 보여줍니다. LLaMA는 인퍼런스 속도와 환각(hallucination) 문제에 직면하고 있어 이를 해결하기 위한 두 번째 기여로 Forward Pass Only Decoding(FPOD)이라는 새로운 디코딩 접근방식을 제안합니다. 이를 통해 인퍼런스 속도를 19.8배 개선할 수 있습니다.

- **Technical Details**: 구두점 복원을 위한 FPOD 방법론은 자동 회귀 생성(auto-regressive generation) 단계를 완전히 폐기하여 인퍼런스 과정을 단순화합니다. FPOD는 LoRA fine-tuning을 통해 LLaMA 모델을 조정하고, 입력 텍스트 각각의 토큰에 대해 다음 토큰을 예측합니다. 만약 예측된 토큰이 구두점이라면 현재 토큰 앞에 추가합니다. 이 접근 방식은 전통적인 auto-regressive 방법에 비해 구두점 복원의 인퍼런스 속도를 크게 향상시킵니다.

- **Performance Highlights**: LLaMA 모델과 FPOD 방법의 조합은 대규모 데이터 주석 작업에서 실용성을 높이고, 구두점 복원 성능을 개선합니다. FPOD를 통해 속도 개선이 이루어지면서도 환각 문제를 완화할 수 있습니다. 더불어, FPOD는 긴 입력 컨텍스트(lengthy input context)에서 성능 향상을 위해 슬라이딩 윈도우(sliding window) 기법을 적용하고, Recursive FPOD를 통해 구두점을 예측하는 두 번의 패스를 수행합니다.



### Editable Fairness: Fine-Grained Bias Mitigation in Language Models (https://arxiv.org/abs/2408.11843)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2405.09341

- **What's New**: 이번 논문에서는 공정하고 정확한 예측을 생성하는 것이 대형 언어 모델(LLMs)의 실제 적용에 중요하다고 강조하고 있으며, 새로운 편향 완화 벤치마크인 BiaScope를 제정하고, 개인의 사회적 편향을 세분화된 방식으로 보정하는 Fairness Stamp(FAST) 방식을 제안합니다.

- **Technical Details**: BiaScope는 기존의 편향 제거 접근법이 사회적 집단 간의 동등성을 달성하는 데 초점을 맞췄지만, 개인의 상식적인 사실을 간과하면서 부정확한 예측을 초래함을 지적합니다. FAST는 LLM에서 사회적 편향을 저장하는 결정적인 레이어를 식별하고, 작은 모듈 네트워크를 통합하여 편향을 완화합니다.

- **Performance Highlights**: 실험 결과, FAST는 최신 방법론들에 비해 우수한 편향 제거 성능을 보이면서도 모델의 지식 보존 및 하위 예측 능력을 저해하지 않음이 입증되었습니다. 이는 LLM의 공정성을 달성하기 위한 세분화된 편향 제거 전략의 가능성을 보여줍니다.



### OpenFactCheck: A Unified Framework for Factuality Evaluation of LLMs (https://arxiv.org/abs/2408.11832)
Comments:
          10 pages, 4 Figures, 3 Tables, Submitted to EMNLP 2024 System Demonstration. arXiv admin note: substantial text overlap with arXiv:2405.05583

- **What's New**: OpenFactCheck는 LLM의 사실성(factuality)을 평가하는 통합 프레임워크로 세 가지 모듈로 구성된다: (i) RESPONSEEVAL, (ii) LLMEVAL, (iii) CHECKEREVAL. 이는 자동 사실 확인 시스템을 사용자 맞춤형으로 만들 수 있게 하여, LLM의 출력에서 발생하는 '환각(hallucination)' 문제를 해결하는 데 도움을 준다.

- **Technical Details**: OpenFactCheck의 세 가지 모듈은 서로 밀접하게 통합되어 사용되며, 각각의 기능이 강화를 목적으로 설계되었다. RESPONSEEVAL은 사용자 맞춤형 사실 확인 시스템을 생성하게 하고, LLMEVAL은 LLM의 전반적인 사실성을 여러 벤치마크를 통해 평가하며, CHECKEREVAL은 자동 사실 확인 시스템의 정확성을 평가한다. 사용자들은 특정 필요에 따라 자신만의 확인기를 구성할 수 있다.

- **Performance Highlights**: OpenFactCheck는 오픈 소스 라이브러리와 웹 서비스를 제공하며, 사용자가 맞춤형 사실 확인기를 설계하고 LLM의 출력의 사실성을 쉽게 평가할 수 있도록 함으로써 향후 LLM 사실성 연구의 발전을 촉진할 것으로 기대된다.



### The Mechanics of Conceptual Interpretation in GPT Models: Interpretative Insights (https://arxiv.org/abs/2408.11827)
Comments:
          23 pages, 25 figures

- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)에서 지식을 찾고 수정하는 과정에 대해 새로운 접근 방식인 'Concept Editing'을 소개합니다. 이는 LLMs 내에서 개념화 메커니즘을 밝히는 혁신적인 방법으로, 역사전(dictionary) 작업을 통해 이루어집니다.

- **Technical Details**: 이 연구는 Multi-Layer Perceptron (MLP), Multi-Head Attention (MHA), 그리고 transformer 모델의 hidden state와 같은 다양한 구성 요소를 분석합니다. MLP 층에서는 키-값 검색 메커니즘과 맥락 의존적 처리 방식이 관찰되었고, MHA 층은 분산된 특성을 보였으며 높은 수준의 활동을 통해 정교한 의미 통합을 나타냈습니다. 마지막으로 hidden states는 추론 과정에서 마지막 토큰과 상위 층의 중요성을 강조합니다.

- **Performance Highlights**: 실험 결과는 개념적 해석이 높은 구성성을 나타내며, 각 모델 구성 요소가 어떻게 상호작용하며 개념 이해를 형성하는지를 보여줍니다. 논문에서는 MLP 층이 특정 어휘적 특성에 중점을 두고, MHA 층과 hidden states가 더 넓은 맥락 정보를 집계하여 최종 개념 표현에 도달하는 방식에 대한 분석이 포함되어 있습니다.



### MuMA-ToM: Multi-modal Multi-Agent Theory of Mind (https://arxiv.org/abs/2408.12574)
Comments:
          Project website: this https URL Code: this https URL

- **What's New**: 이번 논문에서는 MuMA-ToM이라는 새로운 Multi-modal Multi-Agent Theory of Mind 벤치마크를 소개하고 있습니다. 이는 AI가 복잡한 사회적 상호작용에서 사람들의 정신적 상태를 이해할 수 있도록 돕기 위해 고안되었습니다.

- **Technical Details**: MuMA-ToM는 현실적인 가정 환경에서 사람들의 다중 모달 행동에 대한 비디오 및 텍스트 설명을 제공하며, 이러한 맥락을 바탕으로 목표, 신념 및 다른 사람의 목표에 대한 신념에 관한 질문을 합니다. 또한, LIMP(언어 모델 기반의 역 다중 에이전트 계획)라는 새로운 다중 모달, 다중 에이전트 ToM 모델을 제안하였습니다.

- **Performance Highlights**: LIMP는 다양한 최신 방법들, 특히 대형 다중 모달 모델(GPT-4o, Gemini-1.5 Pro 등) 및 최근 다중 모달 ToM 모델(BIP-ALM)을 능가하는 성능을 보였습니다.



### Vintern-1B: An Efficient Multimodal Large Language Model for Vietnames (https://arxiv.org/abs/2408.12480)
Comments:
          arXiv admin note: text overlap with arXiv:2404.16821 by other authors

- **What's New**: Vintern-1B는 베트남어 작업을 위해 설계된 10억 개의 파라미터를 가진 신뢰할 수 있는 멀티모달 대형 언어 모델입니다. 이 모델은 Qwen2-0.5B-Instruct 언어 모델과 InternViT-300M-448px 시각 모델을 통합하여 OCR, 문서 추출 및 일반적인 질문-응답 작업에 최적화되었습니다.

- **Technical Details**: Vintern-1B는 베트남어 텍스트 작업을 위해 3백만 개 이상의 이미지-질문-답변 쌍으로 구성된 대규모 데이터셋에서 파인튜닝 되어 있습니다. 이 모델의 아키텍처는 Vision Encoder(InternViT-300M-448px), 2-layer Multi-Layer Perceptron (MLP) 프로젝트 및 사전 훈련된 Qwen2-0.5B-Instruct 언어 모델을 포함합니다. 또한, 비동적 해상도 모듈 전략을 사용하여 이미지를 448×448 픽셀로 나누어 모델의 입력 처리를 가능하게 합니다.

- **Performance Highlights**: Vintern-1B는 OpenViVQA 및 ViTextVQA와 같은 여러 베트남어 벤치마크에서 강력한 성능을 발휘했습니다. 또한, 다양한 온디바이스 응용 프로그램에 쉽게 적합할 만큼 작은 크기를 가지고 있으며, 베트남어 시각 질문 응답(VQA) 데이터셋이 오픈소스로 제공됩니다.



### A Comparative Analysis of Faithfulness Metrics and Humans in Citation Evaluation (https://arxiv.org/abs/2408.12398)
Comments:
          Accepted by the First Workshop on Large Language Model for Evaluation in Information Retrieval (LLM4Eval@SIGIR2024), non-archival. arXiv admin note: substantial text overlap with arXiv:2406.15264

- **What's New**: 이 논문은 세 가지 범주인 전면 지원(full support), 부분 지원(partial support), 무 지원(no support)으로 구분된 인용 지원 수준을 평가하기 위한 비교 평가 프레임워크를 제시합니다. 이는 기존의 이진 분류 방식의 한계를 극복하여 보다 세분화된 인용 지원 평가를 가능하게 합니다.

- **Technical Details**: 연구에서는 상관 분석(correlation analysis), 분류 평가(classification evaluation), 검색 평가(retrieval evaluation)라는 세 가지 평가 프로토콜을 사용하여 신뢰성과 인간의 판단을 비교합니다. 특히, 인용 지원 수준에 대한 세 가지 카테고리를 도입하여 기존의 연구가 놓쳤던 세분화된 지원을 평가합니다. 실험에서는 유사도 기반(similarity-based) 및 함의 기반(entailment-based) 신뢰성 메트릭을 사용합니다.

- **Performance Highlights**: 결과적으로, 어떤 신뢰성 메트릭도 세 가지 평가 프로토콜에서 일관되게 우수한 성과를 보이지 않으며, 부분 지원을 식별하는 데 어려움을 겪습니다. 이 연구는 지속적으로 발전할 필요가 있는 신뢰성 메트릭 개발에 대한 실용적인 권고사항을 제시합니다.



### Large Language Models Are Self-Taught Reasoners: Enhancing LLM Applications via Tailored Problem-Solving Demonstrations (https://arxiv.org/abs/2408.12315)
Comments:
          preprint / under review

- **What's New**: 본 논문은 대규모 언어 모델(Large Language Model, LLM)의 성능 향상을 위해 선택된 인간 작성 시연으로 안내하는 전통적인 접근 방식의 한계를 인식하고, 자동 생성된 맞춤형 시연을 통한 해결책을 제시합니다.

- **Technical Details**: SELF-TAUGHT라는 프레임워크를 소개하며, 이 시스템은 주어진 문제에 맞춰 조정된 시연을 자동으로 생성하고, 더욱 향상된 품질의 시연(정확성)을 제공하는 데 초점을 둡니다. 이 과정은 zero-shot 방식으로 실행됩니다. 해당 프레임워크에 대한 포괄적인 분석도 진행되어, 다양한 LLM과 프롬프트 방법에 대한 일반화 가능성과 중간 결과물의 품질 등이 포함됩니다.

- **Performance Highlights**: SELF-TAUGHT는 다섯 가지 다양한 분야의 다지선다형 질문 및 실제 환자를 대상으로 한 알츠하이머 질환 진단을 포함한 15개의 작업에서 강력한 기준선들을 초월하는 성능을 달성했습니다.



### Search-Based LLMs for Code Optimization (https://arxiv.org/abs/2408.12159)
Comments:
          Accepted by 2025 IEEE/ACM 47th International Conference on Software Engineering (ICSE'25)

- **What's New**: 이 논문에서는 기존 코드 최적화 방법이 지닌 한계를 극복하기 위해, 코드 최적화 작업을 탐색 관점에서 모델링하고, 반복적인 개선 및 최적화 방법 발견을 가능하게 하는 SBLLM(Search-Based LLMs)라는 새로운 프레임워크를 제안합니다.

- **Technical Details**: SBLLM은 LLM(대형 언어 모델)과 진화적 검색(evolutionary search)을 통합하며, 세 가지 주요 구성 요소로 이루어져 있습니다: 1) 실행 기반의 대표 샘플 선택(execution-based representative sample selection), 2) 적응형 최적화 패턴 검색(adaptive optimization pattern retrieval), 3) 유전자 연산자 기반의 사고 과정을 촉진하는 프롬프트(Genetic Operator-inspired chain-of-thought prompting). 이 방법은 복잡한 최적화 방법을 보다 효과적으로 통합하여 LLM이 더 개선된 최적화 코드를 생성하도록 돕습니다.

- **Performance Highlights**: SBLLM은 광범위한 벤치마크 데이터셋에서 실험을 통해 기존의 여러 프롬프트링 방법에 비해 코드 효율성을 8.75%에서 28.06%까지 향상시키는 데 성공하였습니다. Python 및 C++ 코드에 대한 성능 개선이 관찰되었으며, 해당 프레임워크는 다양한 LLM에서 우수한 성과를 보였습니다.



### A Tighter Complexity Analysis of SparseGP (https://arxiv.org/abs/2408.12151)
- **What's New**: 이번 연구에서는 SparseGPT의 러닝 타임 분석을 기존의 O(d^3)에서 O(d^{ω} + d^{2+a+o(1)} + d^{1+ω(1,1,a)-a})로 개선했습니다. 여기서 ω는 행렬 곱셈의 지수를 나타냅니다. 현재 ω의 값이 약 2.371일 때, 러닝 타임은 O(d^{2.53})으로 단축되었습니다.

- **Technical Details**: SparseGPT는 GPT 계열 모델의 파라미터를 최적화된 뇌 손상 기법(Optimal Brain Damage)을 사용하여 50% 이상 가지치기할 수 있습니다. 이 알고리즘(Algorithm 1)의 러닝 타임은 O(d^3)에서 O(d^{2.53})으로 개선되었습니다. 이 개선은 이터레이티브 유지 문제에서의 레이지 업데이트(lazy update) 행동 분석을 통해 이루어졌습니다.

- **Performance Highlights**: 향상된 러닝 타임 O(d^{2.53}) 덕분에 SparseGPT는 높은 성능을 유지하면서의 실행 시간 및 GPU 메모리 사용량을 감소시킬 수 있습니다. 이것은 LLMs의 애플리케이션에 매우 유리한 결과를 가져옵니다.



### RoVRM: A Robust Visual Reward Model Optimized via Auxiliary Textual Preference Data (https://arxiv.org/abs/2408.12109)
- **What's New**: 이번 연구에서는 대형 비전-언어 모델(LVLMs)의 인간 선호도 정렬을 개선하기 위한 Robust Visual Reward Model (RoVRM)을 제안합니다. RoVRM은 보조 텍스트 선호 데이터(auxiliary textual preference data)를 활용하여 시각적 선호 데이터의 부족 문제를 효과적으로 완화합니다.

- **Technical Details**: RoVRM은 세 단계의 점진적 훈련(progressive training)과 최적 수송(optimal transport) 기반의 선호 데이터 선택(preference data selection) 방식을 통해 구성됩니다. 이 방식은 시각적 선호 데이터를 훈련하기 위한 비주얼 보상 모델(VRM)을 더욱 향상시킵니다.

- **Performance Highlights**: LLaVA-1.5-7B 및 -13B 모델을 기반으로 한 실험에서 RoVRM은 전통적인 VRM을 지속적으로 초월하는 성능을 보였습니다. 또한, 점진적 훈련 및 선호 데이터 선택 방법은 순위 기반 정렬 기술(ranking-based alignment techniques)인 직접 선호 최적화(direct preference optimization)보다 일관된 성능 향상을 나타냈습니다.



### Extraction of Research Objectives, Machine Learning Model Names, and Dataset Names from Academic Papers and Analysis of Their Interrelationships Using LLM and Network Analysis (https://arxiv.org/abs/2408.12097)
Comments:
          10 pages, 8 figures

- **What's New**: 이 연구는 기존의 정보 추출 방법을 개선하여 논문에서 연구 목표, 기계학습(methods), 데이터셋 이름을 동시에 추출하고 이를 클러스터링하여 관계를 분석하는 새로운 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 Llama2와 Llama3라는 대형 언어 모델을 활용하여 연구 목표와 관련된 내용을 동시에 추출하고, E5 임베딩 모델을 통해 동의어 클러스터링을 수행합니다. 이 과정에서 네트워크 클러스터링을 통한 관계 분석도 포함됩니다.

- **Performance Highlights**: 실험 결과, Llama3를 사용한 정보 추출 성능의 F-score가 0.8을 초과하며, 금융 분야 논문에 대해 최신 데이터셋과의 관계를 명확히 보여 줍니다.



### Reasoning and Tools for Human-Level Forecasting (https://arxiv.org/abs/2408.12036)
- **What's New**: 본 논문은 Reasoning and Tools for Forecasting (RTF)라는 새로운 프레임워크를 제안하여, 웹 기반 데이터 세트로 훈련된 언어 모델이 실제로 합리적 추론을 수행할 수 있는 방법을 탐구합니다. 특히, 예측 작업에서 LMs(언어 모델)의 정확성과 적응 능력을 향상시키는 데 초점을 맞추고 있습니다.

- **Technical Details**: RTF 프레임워크는 ‘Reasoning-and-acting (ReAct)’ 에이전트를 기반으로 하며, 에이전트는 최신 정보를 동적으로 검색하고 수치 시뮬레이션을 실행할 수 있는 도구를 갖추고 있습니다. 이 접근법에서는 LMs가 관찰 공간(𝒪)과 행동 공간(𝒜) 내에서 작업을 수행하도록 설계되었습니다.

- **Performance Highlights**: 경쟁적인 예측 플랫폼에서 문제를 평가했으며, 제안한 방법이 인간의 예측보다 우수하거나 경쟁력이 있음을 보여줍니다. 이는 적절한 도구를 갖춘 LMs가 인간처럼 사고하고 적응할 수 있는 가능성을 제시합니다.



### Let Community Rules Be Reflected in Online Content Moderation (https://arxiv.org/abs/2408.12035)
Comments:
          10 pages, 3 figures

- **What's New**: 이 연구는 온라인 커뮤니티의 규칙을 직접 통합한 커뮤니티 규칙 기반 콘텐츠 조정 프레임워크를 제안하여 콘텐츠 조정의 효율성을 높이는 방안을 다루고 있습니다.

- **Technical Details**: 제안된 프레임워크는 사용자 생성 콘텐츠의 조정 과정에서 커뮤니티 규칙을 직접적으로 통합하며, 두 가지 도메인에서 수집된 데이터셋을 활용하여 성능을 평가했습니다.

- **Performance Highlights**: 모델은 모든 평가 지표에서 기준 모델(baseline models)보다 우수한 성능을 보였으며, 특히 커뮤니티 규칙을 통합함으로써 콘텐츠 조정(model performance)에서의 성능 향상이 두드러졌습니다.



### Limitations in Employing Natural Language Supervision for Sensor-Based Human Activity Recognition -- And Ways to Overcome Them (https://arxiv.org/abs/2408.12023)
- **What's New**: 이 논문은 웨어러블 센서를 기반으로 한 인간 활동 인식(Human Activity Recognition, HAR)에 대한 자연어 감독(Natural Language Supervision, NLS)의 효과를 조사합니다. 기존의 NLS가 다양한 작업과 분야에서 뛰어난 성과를 보였음에도 불구하고, HAR에서 기대 이하의 결과를 나타냈다는 점에 주목하고 있습니다.

- **Technical Details**: 주요 원인은 센서 이질성과 활동에 대한 풍부하고 다양한 텍스트 설명 부족입니다. 이를 해결하기 위해 간단한 적응을 통해 HAR 성능을 크게 향상시키는 여러 전략을 개발했습니다. 특히, 사전 훈련된 네트워크의 일부 레이어를 적은 양의 타겟 데이터로 업데이트 하는 방법이 효과적임을 보여주었습니다.

- **Performance Highlights**: 이 연구의 전략은 HAR 성능을 유의미하게 개선하여 감독 및 자가 감독 훈련에 가까운 성과를 따르게 했습니다. 이를 통해 이전에 보지 못한 활동을 인식하고, 비디오의 크로스 모달 검색도 가능하게 했습니다. 전반적으로 이러한 연구는 웨어러블 장치를 사용하는 HAR을 위한 기본 모델 개발로 이어질 수 있는 가능성을 제시합니다.



### Characterizing Online Toxicity During the 2022 Mpox Outbreak: A Computational Analysis of Topical and Network Dynamics (https://arxiv.org/abs/2408.11962)
Comments:
          36 pages, 8 figure, and 12 tables

- **What's New**: 2022년 Mpox 발병에 대한 온라인의 독성 담론을 체계적으로 분석하여 그 기원과 성격, 확산 패턴 및 사회적 함의를 규명하다.

- **Technical Details**: 1.6백만 개 이상의 트윗을 수집하여, BERT 기반의 토픽 모델링과 사회망 커뮤니티 클러스터링을 활용하여 Twitter에서 독성 동태를 분석하였다.

- **Performance Highlights**: 독성 담론의 주요 주제 범주로 질병(46.6%), 건강 정책 및 의료(19.3%), 동성애 혐오(23.9%), 정치(6.0%), 인종차별(4.1%)를 확인하였다.



### Unraveling Text Generation in LLMs: A Stochastic Differential Equation Approach (https://arxiv.org/abs/2408.11863)
- **What's New**: 이 논문은 Stochastic Differential Equations (SDEs)를 사용하여 Large Language Models (LLMs), 특히 GPT-4의 텍스트 생성 과정을 해석하는 새로운 방법을 탐구합니다. 텍스트 생성 과정은 확률적 프로세스로 모델링되며, 각 단계는 이전에 생성된 콘텐츠와 모델 매개변수에 의존하여 다음 단어를 선택합니다.

- **Technical Details**: SDE는 텍스트 생성 과정의 결정론적 경향(drfit term)과 확률적 변동(diffusion term)을 함께 포착하는 수학적 구조를 제공합니다. 이 모델은 신경망(neural networks)을 사용하여 학습하고 실제 텍스트 자료를 기반으로 검증됩니다.

- **Performance Highlights**: 본 연구를 통해 LLM의 동력을 깊이 있게 이해할 수 있으며, 이는 생성된 텍스트의 품질을 진단하고 최적화하는 데 중요한 기여를 합니다. SDE 기반 접근 방식을 통해 LLM의 내재된 특성과 변동성을 더욱 잘 설명할 수 있습니다.



### SAGE-RT: Synthetic Alignment data Generation for Safety Evaluation and Red Teaming (https://arxiv.org/abs/2408.11851)
- **What's New**: SAGE-RT(또는 SAGE)는 안전 평가 및 레드 팀 데이터 생성에 필요한 합성 alignment 데이터 생성의 새로운 파이프라인으로, 기존 방법의 한계를 극복하여 더 다양하고 세부적인 데이터셋을 생성할 수 있는 방법을 제시합니다.

- **Technical Details**: SAGE는 ALERT에 기반하여 해로운 주제에 대한 세부 분류법을 사용하여 51,000 개의 다양한 질의-응답 쌍을 생성하였습니다. 그 과정에서 1,500 개 이상의 해로운 주제를 다루고, LLM이 직면하는 다양한 jailbreaking 질의를 포함합니다. 이 방법은 설치된 LLM의 안전성을 보장하기 위해 synthetic red-teaming과 alignment 데이터를 완전히 합성적으로 생성할 수 있습니다.

- **Performance Highlights**: SAGE를 통해 생성된 red-teaming 데이터는 32 개 하위 카테고리 중 27 개에서 최첨단 LLM을 jailbreak 하였으며, 279 개 리프 카테고리 중 58 개에서 성공적인 공격 비율을 보였습니다. GPT-4o 및 GPT-3.5-turbo에 대해 100%의 공격 성공률을 기록하며, 안전성 관련 하위 카테고리 testing에서 극적인 성과를 나타냈습니다.



### Could ChatGPT get an Engineering Degree? Evaluating Higher Education Vulnerability to AI Assistants (https://arxiv.org/abs/2408.11841)
Comments:
          20 pages, 8 figures

- **What's New**: 본 연구에서는 AI 어시스턴트가 고등 교육에서 학생들의 평가 및 학습 결과에 미치는 영향을 분석하고 있으며, 특히 생성 AI(generative AI)의 영향을 바탕으로 대학 평가의 취약성(vulnerability)을 개념화하고 있습니다.

- **Technical Details**: EPFL(École Polytechnique Fédérale de Lausanne)에서 제공하는 50개 과목의 텍스트 기반 평가 질문 데이터셋을 구축하였으며, 두 개의 모델인 GPT-3.5와 GPT-4의 성능을 측정하였습니다. 총 5,579개의 개방형 응답 질문(open-answer questions)과 객관식 질문(multiple-choice questions)으로 구성되어 있으며, 자동 및 수동 평가(automatic and manual grading) 방식으로 분석하였습니다.

- **Performance Highlights**: GPT-4는 평균적으로 65.8%의 질문에 올바른 답변을 제공할 수 있으며, 적어도 85.1%의 질문에 대해 최소 하나의 올바른 답변을 생성할 수 있었습니다. 이는 다양한 학문 분야의 과목에 걸쳐 상대적으로 안정적인 성과를 보이며, 여러 대학 프로그램에서 상당한 수준의 취약성을 나타냅니다.



### Mistral-SPLADE: LLMs for better Learned Sparse Retrieva (https://arxiv.org/abs/2408.11119)
- **What's New**: 이번 연구에서는 학습된 희소 검색기(Learned Sparse Retrievers, LSR)의 성능을 개선하기 위해 디코더 전용 모델을 활용하는 새로운 접근 방식을 제안합니다. Mistral을 기반으로 한 Echo-Mistral-SPLADE 모델은 기존의 LSR 시스템들을 초월하는 성능을 발휘하여 BEIR 텍스트 검색 벤치마크에서 최첨단 모델로 자리 잡았습니다.

- **Technical Details**: 이번 연구에서는 기존 SPLADE 모델처럼 단편적 데이터(embedding-based dense retrievers)와의 결합을 통해 디코더 전용 대형 언어 모델(Decoder-only Large Language Model, LLM)을 사용합니다. 쿼리와 문서의 가장 중요한 의미 키워드 확장을 학습함으로써 희소(maspec) 검색 성능을 향상시킵니다. 연구 결과는 대규모 데이터의 훈련을 통해 디코더 전용 모델이 키워드 확장을 학습하는 데 더 뛰어난 능력을 갖춘다는 것을 입증합니다.

- **Performance Highlights**: Echo-Mistral-SPLADE 모델은 기존의 LSR 시스템들, 특히 SPLADE 및 그 변형들을 초과하는 성능을 보이며, 정보 검색 분야에서 인식을 향상시킵니다. 희소 검색 모델은 바쁜 정보 검색 환경에서 더 나은 성능을 제공하는 것으로 평가됩니다.



New uploads on arXiv(cs.IR)

### The Importance of Cognitive Biases in the Recommendation Ecosystem (https://arxiv.org/abs/2408.12492)
- **What's New**: 이 논문에서는 추천 시스템(Recommendation System)에서의 인지 편향(cognitive biases)의 긍정적인 영향을 제안하고, 이를 활용하여 사용자 및 아이템 모델과 추천 알고리즘을 개선할 수 있는 가능성을 탐구한다.

- **Technical Details**: 본문에서는 Feature-Positive Effect, Ikea Effect, Cultural Homophily와 같은 인지 편향이 추천 프로세스의 다양한 단계에서 어떻게 나타나는지를 설명하고, 이를 실험적으로 검증한다. 이러한 편향은 입력 데이터(예: 평점, 보조 정보), 추천 알고리즘, 사용자 상호작용 등에서 관찰된다.

- **Performance Highlights**: 세 가지 소규모 실험을 통해 recruitment 및 entertainment 분야에서 인지 편향의 만연성을 검토하였으며, 이 편향들이 추천 알고리즘의 성능 향상에 기여할 수 있음을 보여준다.



### DLCRec: A Novel Approach for Managing Diversity in LLM-Based Recommender Systems (https://arxiv.org/abs/2408.12470)
- **What's New**: 본 논문에서는 LLM 기반 추천 시스템의 다양성을 높이는 DLCRec라는 새로운 프레임워크를 제안합니다. DLCRec는 사용자가 정의한 제어 수를 기반으로 추천 프로세스를 세분화하여 보다 정밀한 추천을 가능하게 합니다.

- **Technical Details**: DLCRec는 추천 프로세스를 세 가지 서브 태스크 (장르 예측, 장르 채우기, 아이템 예측)로 분해하여 처리합니다. 각각의 서브 태스크는 독립적으로 훈련되고, 사용자의 제어 수에 따라 순차적으로 추론됩니다. 또한, 데이터의 극단적인 경향성 문제를 해결하기 위해 데이터 증강 기법을 도입했습니다.

- **Performance Highlights**: DLCRec는 추천의 다양성에 대한 정밀한 제어를 제공하며, 여러 추천 시나리오에서 최신 기술 대비 우수한 성능을 보였습니다. 실험 결과, 정확도는 약간 감소하더라도 다양성 제어에 있어 효과적인 결과를 입증하였습니다.



### A Comparative Analysis of Faithfulness Metrics and Humans in Citation Evaluation (https://arxiv.org/abs/2408.12398)
Comments:
          Accepted by the First Workshop on Large Language Model for Evaluation in Information Retrieval (LLM4Eval@SIGIR2024), non-archival. arXiv admin note: substantial text overlap with arXiv:2406.15264

- **What's New**: 이 논문은 세 가지 범주인 전면 지원(full support), 부분 지원(partial support), 무 지원(no support)으로 구분된 인용 지원 수준을 평가하기 위한 비교 평가 프레임워크를 제시합니다. 이는 기존의 이진 분류 방식의 한계를 극복하여 보다 세분화된 인용 지원 평가를 가능하게 합니다.

- **Technical Details**: 연구에서는 상관 분석(correlation analysis), 분류 평가(classification evaluation), 검색 평가(retrieval evaluation)라는 세 가지 평가 프로토콜을 사용하여 신뢰성과 인간의 판단을 비교합니다. 특히, 인용 지원 수준에 대한 세 가지 카테고리를 도입하여 기존의 연구가 놓쳤던 세분화된 지원을 평가합니다. 실험에서는 유사도 기반(similarity-based) 및 함의 기반(entailment-based) 신뢰성 메트릭을 사용합니다.

- **Performance Highlights**: 결과적으로, 어떤 신뢰성 메트릭도 세 가지 평가 프로토콜에서 일관되게 우수한 성과를 보이지 않으며, 부분 지원을 식별하는 데 어려움을 겪습니다. 이 연구는 지속적으로 발전할 필요가 있는 신뢰성 메트릭 개발에 대한 실용적인 권고사항을 제시합니다.



### Dynamic Product Image Generation and Recommendation at Scale for Personalized E-commerc (https://arxiv.org/abs/2408.12392)
Comments:
          Appearing in the Proceedings of the 18th ACM Conference on Recommender Systems (RecSys'24) as an Industry Track paper

- **What's New**: 이 연구는 Latent Diffusion을 기반으로 한 이미지 생성 기술과 Contextual Bandits를 결합하여 e-commerce를 위한 개인화된 제품 이미지를 대규모로 생성하는 방법을 제시합니다. 기존의 방법들로는 실현 불가능하거나 비용이 많이 들었던 이 솔루션은 온라인 재타겟팅 캠페인에서 사용자 참여를 증가시킬 수 있었음을 보여줍니다.

- **Technical Details**: Stable Diffusion 모델을 통해 제품 카테고리에 적합한 환경을 설명하는 프롬프트를 사용하여 배경을 생성하는 방법을 개발했습니다. ControlNet을 활용하여 이미지 생성 과정에 추가적인 제약 조건을 삽입하며, 이는 제품의 가장자리를 사용하여 이루어집니다. 이 파이프라인은 여러 단계를 포함하며, 효율성 및 품질 향상을 위해 이미지 생성 조건을 통해 결과를 맞춤화합니다.

- **Performance Highlights**: 온라인 A/B 테스트에서 생성된 이미지가 기존의 제품 이미지보다 평균적으로 15% 향상을 이끌어냈습니다. 개인화된 배경을 선택한 경우, LinUCB 알고리즘을 통해 추천된 프롬프트가 사용된 그룹이 제어 그룹에 비해 추가로 약 5%의 성과 향상을 보였습니다.



### Fair Augmentation for Graph Collaborative Filtering (https://arxiv.org/abs/2408.12208)
- **What's New**: 이 논문에서는 graph neural networks (GNNs)를 활용한 추천 시스템의 공정성 문제를 해결하기 위한 새로운 방법론을 제시합니다. 특히, 사용자-아이템 네트워크에서의 사용자 선호 학습을 통해 소비자의 관점에서 공정성 문제를 다룹니다.

- **Technical Details**: 제안하는 기법은 fair graph augmentation을 통해 시스템의 공정성 수준을 조정하며, 11개의 GNN 모델, 5개의 비-GNN 모델 및 다양한 도메인에서의 5개 실제 네트워크를 기반으로 실험을 수행합니다. 이 과정에서 정의된 샘플링 정책과 GNN의 forward process에서의 통합 방식을 이론적으로 형식화했습니다.

- **Performance Highlights**: 실험 결과 fair graph augmentation이 고유용성 모델 및 대형 데이터세트에서 일관된 효과를 나타내며, 소비자 불공정성 완화에 대한 포괄적인 벤치마크를 수행했습니다. 이를 통해 향후 추천 시스템 연구를 위한 새로운 문제를 열어주었습니다.



### Hardware Acceleration for Knowledge Graph Processing: Challenges & Recent Developments (https://arxiv.org/abs/2408.12173)
- **What's New**: 최근 지식 그래프(Knowledge Graphs, KGs)의 하드웨어 가속화(Hardware Acceleration) 관련 연구가 활발히 진행되고 있으며, 이 논문은 이러한 연구들을 종합적으로 리뷰합니다.

- **Technical Details**: 본 연구에서 하드웨어 가속화는 GPU(Graphics Processing Units)와 FPGA(Field Programmable Gate Arrays)와 같은 전문 하드웨어를 이용하여 지식 그래프의 특정 기능을 효율적으로 처리하는 방식을 다룹니다. 주어진 데이터 모델은 Labeled Property Graph (LPG)와 Resource Description Framework (RDF)가 있으며, 두 모델 모두 그래프 구조 내에서 엔티티와 관계를 시각화하여 이해할 수 있게 합니다.

- **Performance Highlights**: 하드웨어 가속화는 전통적인 CPU보다 성능 향상을 가져오며, 지식 그래프의 실시간 데이터 처리 및 분석의 효율을 크게 높입니다. 하지만, 기술 적용 및 효과에 대한 연구는 아직도 미비한 상태로, 향후 연구 기회가 존재합니다.



### DimeRec: A Unified Framework for Enhanced Sequential Recommendation via Generative Diffusion Models (https://arxiv.org/abs/2408.12153)
- **What's New**: 본 연구에서는 Sequential Recommendation(SR)과 Generative Diffusion Models(DM)를 통합하여 추천 시스템에서 아이템 표현 및 다양성을 최적화하는 새로운 프레임워크인 DimeRec(Diffusion with multi-interest enhanced Recommender)를 제안합니다. DimeRec은 사용자의 비정상적인 상호작용 이력을 바탕으로 중요한 정적 가이던스 신호를 추출하고, 이를 기반으로 일관된 추천을 생성합니다.

- **Technical Details**: DimeRec은 Guidance Extraction Module(GEM)과 Generative Diffusion Aggregation Module(DAM)으로 구성됩니다. GEM은 사용자의 비정상적인 상호작용 이력에서 중요한 정적 가이던스 신호를 추출하고, DAM은 GEM의 출력을 조건으로 한 생성적 확산 과정을 통해 추천 아이템을 재구성 및 생성합니다. 또한, DimeRec은 추천 손실과 확산 모델의 손실을 동시에 최적화하는 새로운 노이즈 공간을 도입합니다.

- **Performance Highlights**: DimeRec은 세 가지 공개 데이터셋에서 기존의 여러 기준 방법을 능가하는 성능을 보여주었으며, 대규모 단편 비디오 추천 플랫폼에 성공적으로 배포되어 수억 명의 사용자에게 서비스되었습니다. A/B 테스트 결과, DimeRec은 사용자 시간 소비를 개선하고 결과의 다양성을 높였습니다.



### Behavior Pattern Mining-based Multi-Behavior Recommendation (https://arxiv.org/abs/2408.12152)
- **What's New**: 이번 논문에서는 기존의 다중 동작 추천 시스템의 한계를 극복하기 위해 새로운 알고리즘인 Behavior Pattern mining-based Multi-behavior Recommendation (BPMR)을 소개합니다. BPMR은 사용자와 항목 간의 다양한 상호작용 패턴을 심층적으로 조사하여 추천 시 이러한 패턴을 특징으로 활용합니다.

- **Technical Details**: BPMR은 베이지안(Bayesian) 접근 방식을 사용하여 추천 프로세스를 간소화하고, 그래프 신경망(Graph Neural Network) 알고리즘이 직면하는 문제인 과도한 평탄화(over-smoothing) 문제를 효과적으로 회피합니다. 이 방법은 다양한 사용자 행동 간의 상호작용을 파악하여 고유한 사용자 선호도를 더 정확하게 포착합니다.

- **Performance Highlights**: BPMR은 세 가지 실제 데이터셋에 대한 실험 평가에서 Recall@10에서 평균 268.29%, NDCG@10에서 248.02%의 상당한 성능 향상을 보여주며 기존 최첨단 알고리즘을 크게 능가하는 것으로 입증되었습니다.



### Does It Look Sequential? An Analysis of Datasets for Evaluation of Sequential Recommendations (https://arxiv.org/abs/2408.12008)
- **What's New**: 본 논문에서는 순차 추천 시스템(Sequential Recommender Systems, SRSs)의 평가를 위해 데이터셋의 순차 구조의 강도를 측정하는 방법을 제안합니다. 특히, 데이터의 순차 패턴이 존재하는지를 분석하기 위한 15개의 일반적인 데이터셋을 평가했습니다.

- **Technical Details**: 우리는 사용자의 상호작용 순서를 무작위로 섞어 데이터셋 내에서 순차 구조의 강도를 측정하는 세 가지 접근 방식을 제안합니다. 첫 번째 접근 방식은 순차 규칙을 식별하며 모델에 독립적입니다. 두 번째와 세 번째 접근 방식은 SASRec 및 GRU4Rec이라는 순차 모델을 훈련하고, 원본 및 섞인 데이터셋에서의 성능을 NDCG@k 및 HitRate@k를 통해 평가합니다.

- **Performance Highlights**: 연구 결과, 여러 널리 사용되는 데이터셋이 예상보다 상대적으로 약한 순차 구조를 지니고 있음이 밝혀졌습니다. 이는 특히 데이터셋이 SRS 평가에 부적합할 수 있음을 시사합니다.



### What are the limits of cross-lingual dense passage retrieval for low-resource languages? (https://arxiv.org/abs/2408.11942)
- **What's New**: 본 논문에서는 다국어 Dense Passage Retriever (mDPR)의 성능을 저자원(low-resource) 언어에 대해 분석하였습니다. 특히 26개 언어에서 다국어 QA(Open Question Answering) 벤치마크에서 성공적인 성과를 보였지만, 훈련에 포함되지 않았던 9개 언어에서도 성능이 나타났습니다. 특히, 아마하리크(Amharic)와 크메르(Khmer)와 같은 극히 저자원 언어에 대해 mDPR의 성능이 저조한 것을 중점적으로 연구하였습니다.

- **Technical Details**: mDPR은 영어로 된 질문-답변 쌍에 대해 훈련된 Dense Passage Retriever(DPR)에서 발전한 모델로, 이 모델은 다국어 BERT(mBERT)를 사용하여 다국어 설정을 확장합니다. 본 연구에서는 TLM(Translation Language Modeling)과 질문-단락 정렬(question-passage alignment)을 통해 훈련 데이터를 제작하고, 이를 통해 mDPR 모델을 재훈련합니다. 아마하리크와 크메르 두 언어에 대해 다양한 언어 정렬을 포함하는 모델을 평가하였습니다.

- **Performance Highlights**: mDPR의 저자원 언어에 대한 성능 개선을 위해 언어 정렬을 통한 실험 결과, 개선은 있었으나 여전히 성능은 제한적이며 변별력이 부족함을 보여주었습니다. MKQA 및 AmQA 데이터셋에서의 결과는 저자원 언어에 대한 mDPR의 최종 성능이 미진함을 확인했습니다. 이 연구는 저자원 언어에 대한 다국어 QA 시스템 개발의 어려움을 강조하며, 모델, 데이터, 평가 접근법의 통합적인 관심이 필요하다고 결론지었습니다.



### RuleAlign: Making Large Language Models Better Physicians with Diagnostic Rule Alignmen (https://arxiv.org/abs/2408.12579)
Comments:
          Ongoing work

- **What's New**: 본 논문에서는 의료 분야에서 Large Language Models (LLMs), 예를 들어 GPT-4, MedPaLM-2, Med-Gemini의 성능을 인간 전문가와 비교하고, 전문가와 유사하게 진단을 내리는 데 필요한 문제를 해결하기 위한 RuleAlign 프레임워크를 소개합니다.

- **Technical Details**: RuleAlign 프레임워크는 규칙 기반 진단 규칙에 LLM을 정렬하기 위해 설계되었습니다. 이를 위해 환자와 의사 간의 규칙 기반 대화를 포함하는 의료 대화 데이터셋을 개발하였으며, 선호 학습(preference learning)을 통한 정렬 학습(alignment learning) 접근 방식을 설계하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식이 효과적임을 입증하였습니다. 우리의 연구가 LLMs가 AI 의사로서의 잠재력을 탐구하는 데 영감을 줄 수 있기를 바랍니다.



### Rank and Align: Towards Effective Source-free Graph Domain Adaptation (https://arxiv.org/abs/2408.12185)
Comments:
          Published in IJCAI2024

- **What's New**: 본 논문은 실제 세계에서의 개인 정보 보호 및 저장 문제로 인해 광범위한 소스 그래프에 접근하는 것이 어려운 상황에서 소스 데이터 없이 그래프 도메인 적응(source-free graph domain adaptation) 문제를 다룸. 여기서는 소스 그래프 대신 사전 학습된 그래프 모델을 사용하여 지식을 타겟 도메인으로 전이하는 방법을 제안한다.

- **Technical Details**: 제안된 방법, Rank and Align (RNA)는 스펙트럴 세리이제이션(Spectral Seriation) 기법을 활용하여 그래프 유사성을 순위별로 매기고, 하모닉 그래프와 비하모닉 그래프를 정렬하여 서브그래프를 추출하는 방식을 사용한다. RNA는 스펙트럴 클러스터링을 통해 도메인 전이를 감지하고, 적대적 엣지 샘플링 과정을 통해 도메인 불변의 서브그래프를 추출하여 GNN의 불변 학습을 돕는다.

- **Performance Highlights**: 여러 벤치마크 데이터셋에서의 광범위한 실험을 통해 RNA가 기존의 기법들과 비교할 때 우수한 성능을 보여준다는 점이 강조된다.



### Reasoning and Tools for Human-Level Forecasting (https://arxiv.org/abs/2408.12036)
- **What's New**: 본 논문은 Reasoning and Tools for Forecasting (RTF)라는 새로운 프레임워크를 제안하여, 웹 기반 데이터 세트로 훈련된 언어 모델이 실제로 합리적 추론을 수행할 수 있는 방법을 탐구합니다. 특히, 예측 작업에서 LMs(언어 모델)의 정확성과 적응 능력을 향상시키는 데 초점을 맞추고 있습니다.

- **Technical Details**: RTF 프레임워크는 ‘Reasoning-and-acting (ReAct)’ 에이전트를 기반으로 하며, 에이전트는 최신 정보를 동적으로 검색하고 수치 시뮬레이션을 실행할 수 있는 도구를 갖추고 있습니다. 이 접근법에서는 LMs가 관찰 공간(𝒪)과 행동 공간(𝒜) 내에서 작업을 수행하도록 설계되었습니다.

- **Performance Highlights**: 경쟁적인 예측 플랫폼에서 문제를 평가했으며, 제안한 방법이 인간의 예측보다 우수하거나 경쟁력이 있음을 보여줍니다. 이는 적절한 도구를 갖춘 LMs가 인간처럼 사고하고 적응할 수 있는 가능성을 제시합니다.



### Ancient Wisdom, Modern Tools: Exploring Retrieval-Augmented LLMs for Ancient Indian Philosophy (https://arxiv.org/abs/2408.11903)
Comments:
          Best paper at the Workshop on Machine Learning for Ancient Languages @ ACL 2024. Proceedings of the 1st Machine Learning for Ancient Languages Workshop, 2024.ml4al-1.23, Association for Computational Linguistics (ACL) 2024. Dataset, code, and evaluation is available at: this https URL

- **What's New**: 이번 연구는 특정 지식 영역에서의 질문 응답(Long-form Question Answering, LFQA)에 대한 새로운 접근 방식인 Retrieval-Augmented Generation (RAG) 모델의 가능성을 탐구합니다. 특히, 고대 인도 철학인 Advaita Vedanta에 관한 방대한 공공 담론을 활용하여 VedantaNY-10M 데이터셋을 개발하였습니다.

- **Technical Details**: RAG 모델은 정보 검색(retrieval) 및 생성(generation) 기능을 결합하여 사용되며, 비RAG LLM(Local Language Model)과 비교하여 성능을 평가하였습니다. 인간 평가자는 데이터 전사(transcription), 정보 검색(retrieval), 생성(generation) 성능을 기준으로 RAG 모델이 비RAG 모델보다 우수한 결과를 보였다고 보고했습니다.

- **Performance Highlights**: RAG 모델은 사실적이고 포괄적인 응답을 생성하며, 환각(hallucinations)이 현저히 적었습니다. 또한, 고유 저빈도(term) 키워드 기반 하이브리드 검색기(hybrid retriever)를 사용하여 성능을 더욱 개선했습니다.



### Hierarchical Retrieval-Augmented Generation Model with Rethink for Multi-hop Question Answering (https://arxiv.org/abs/2408.11875)
Comments:
          undereview

- **What's New**: 본 논문에서는 Multi-hop Question Answering (QA)에 대한 새로운 프레임워크인 Hierarchical Retrieval-Augmented Generation Model with Rethink (HiRAG)을 제안합니다. 이 모델은 Decomposer, Definer, Retriever, Filter, Summarizer의 다섯 가지 모듈로 구성되어 있으며, 퀘스천의 서브퀘스천을 효과적으로 처리하는 새로운 계층적 검색 전략을 도입합니다.

- **Technical Details**: HiRAG는 다단계 검색을 수행하며, 문서 수준에서의 sparse retrieval과 청크 수준에서의 dense retrieval을 통합하여 두 가지 방법의 장점을 활용합니다. 또한, single-candidate retrieval 방식을 도입하여 다수의 후보 검색의 한계를 극복하고, 부정확한 답변이 발견될 경우 Rethink 모듈을 통해 추가 청크를 선택합니다. 논문에서는 Indexed Wikicorpus와 Profile Wikicorpus라는 두 개의 새로운 데이터베이스도 구성하였습니다.

- **Performance Highlights**: HiRAG는 HotPotQA, 2WikiMultihopQA, MuSiQue, Bamboogle의 네 가지 데이터 세트에서 실험을 수행하였으며, 대부분의 메트릭에서 최첨단 모델을 능가하는 성능을 보였습니다. 특히, Indexed Wikicorpus는 효과적인 데이터베이스로 확인되었습니다.



### Mistral-SPLADE: LLMs for better Learned Sparse Retrieva (https://arxiv.org/abs/2408.11119)
- **What's New**: 이번 연구에서는 학습된 희소 검색기(Learned Sparse Retrievers, LSR)의 성능을 개선하기 위해 디코더 전용 모델을 활용하는 새로운 접근 방식을 제안합니다. Mistral을 기반으로 한 Echo-Mistral-SPLADE 모델은 기존의 LSR 시스템들을 초월하는 성능을 발휘하여 BEIR 텍스트 검색 벤치마크에서 최첨단 모델로 자리 잡았습니다.

- **Technical Details**: 이번 연구에서는 기존 SPLADE 모델처럼 단편적 데이터(embedding-based dense retrievers)와의 결합을 통해 디코더 전용 대형 언어 모델(Decoder-only Large Language Model, LLM)을 사용합니다. 쿼리와 문서의 가장 중요한 의미 키워드 확장을 학습함으로써 희소(maspec) 검색 성능을 향상시킵니다. 연구 결과는 대규모 데이터의 훈련을 통해 디코더 전용 모델이 키워드 확장을 학습하는 데 더 뛰어난 능력을 갖춘다는 것을 입증합니다.

- **Performance Highlights**: Echo-Mistral-SPLADE 모델은 기존의 LSR 시스템들, 특히 SPLADE 및 그 변형들을 초과하는 성능을 보이며, 정보 검색 분야에서 인식을 향상시킵니다. 희소 검색 모델은 바쁜 정보 검색 환경에서 더 나은 성능을 제공하는 것으로 평가됩니다.



New uploads on arXiv(cs.CV)

### DreamCinema: Cinematic Transfer with Free Camera and 3D Character (https://arxiv.org/abs/2408.12601)
Comments:
          Project page: this https URL

- **What's New**: 이 논문의 주요 내용은 사용자 친화적인 영화 제작을 위한 새로운 프레임워크인 DreamCinema를 제안하는 것입니다. 이 프레임워크는 2D 및 3D AIGC(Artificial Intelligence Generated Content)를 활용하여 영화 제작 과정에서의 캐릭터 생성과 시네마틱 요소 전환을 혁신적으로 향상시킵니다.

- **Technical Details**: DreamCinema는 다음 세 가지 주요 요소로 구성됩니다: (1) 시네마틱 요소 추출기: 사람의 자세와 카메라 자세를 추출하고 카메라 궤적을 최적화합니다. (2) 3D 캐릭터 생성기: 사용자 선호에 맞춘 고품질 3D 캐릭터를 생성합니다. (3) 구조 기반의 모션 전송 전략: 생성된 캐릭터를 영화 제작 과정에 매끄럽게 통합합니다. 이를 통해 다양한 캐릭터와 카메라 움직임을 사용자 맞춤형으로 생성합니다.

- **Performance Highlights**: 광범위한 실험을 통해 DreamCinema는 고화질 영화를 제작하는 데 있어 매우 효과적임을 입증했습니다. 특히 자유로운 카메라와 3D 캐릭터 통합으로 사용자 맞춤형 영화 제작이 용이해졌습니다.



### ND-SDF: Learning Normal Deflection Fields for High-Fidelity Indoor Reconstruction (https://arxiv.org/abs/2408.12598)
- **What's New**: 이 논문에서는 ND-SDF라는 새로운 방법을 제안하여, 장면의 법선과 선행 법선 간의 각도 변화를 학습하여 정밀한 기하학을 복원하였습니다. 이 방법은 동적으로 샘플들을 활용하여, 복잡한 구조의 세부 사항을 보존하면서 매끄러운 표면을 유지합니다.

- **Technical Details**: ND-SDF는 Normal Deflection Field를 도입하여, 각기 다른 특성을 가진 샘플들에 따라 동적으로 샘플 활용을 조절하여, 정확도를 높이고 모델의 효용성을 개선합니다. 이는 또한 ray sampling 전략을 도입하여, 편향되지 않은 렌더링 프로세스를 촉진합니다.

- **Performance Highlights**: 이 방법은 벽과 바닥 같은 매끄럽고 질감이 덜한 영역을 효율적으로 처리하며, 복잡한 구조의 기하학적 세부 사항도 잘 보존합니다. 다양한 챌린징 데이터셋에서 일관된 개선이 확인되어, 이 방법의 우수성을 입증하였습니다.



### xGen-VideoSyn-1: High-fidelity Text-to-Video Synthesis with Compressed Representations (https://arxiv.org/abs/2408.12590)
Comments:
          Accepted by ECCV24 AI4VA

- **What's New**: 이 논문은 xGen-VideoSyn-1이라는 새로운 text-to-video (T2V) 생성 모델을 소개합니다. 이 모델은 텍스트 설명으로부터 현실적인 장면을 생성할 수 있는 능력을 갖추고 있으며, 최근 OpenAI의 Sora와 같은 진전을 토대로 하고 있습니다.

- **Technical Details**: 모델은 latent diffusion model (LDM) 아키텍처에 기반하고 있으며, 비디오 변분 오토인코더 (VidVAE)를 도입하여 비디오 데이터를 공간적(spatial) 및 시간적(temporal) 차원에서 효과적으로 압축합니다. 새로운 divide-and-merge 전략을 통해 비디오 세그먼트 간의 시간적 일관성을 유지합니다.

- **Performance Highlights**: xGen-VideoSyn-1 모델은 720p 해상도에서 최대 14초의 비디오 생성이 가능하며, 상태-of-the-art T2V 모델과의 비교에서 경쟁력 있는 성능을 보여줍니다. 이 모델은 13M 이상의 고품질 비디오-텍스트 쌍으로부터 학습되었습니다.



### Real-Time Video Generation with Pyramid Attention Broadcas (https://arxiv.org/abs/2408.12588)
- **What's New**: PAB(Pyramid Attention Broadcast)는 DiT 기반 비디오 생성에서 실시간, 고품질, 훈련이 필요없는 새로운 접근법을 제시합니다.

- **Technical Details**: PAB는 diffusion 과정에서의 attention 차이가 U자형 패턴을 보이는 점에 착안하여, 후속 단계에 attention 출력을 피라미드 스타일로 전달하는 방법을 사용합니다. 서로 다른 attention에 대해 최적의 효율을 위한 다양한 방송 전략이 적용됩니다. 또한, 효율적인 분산 추론을 위해 방송 순서 병렬 처리 기능이 도입되어 생성 시간을 크게 단축합니다.

- **Performance Highlights**: PAB는 720p 비디오 생성 시 20.6 FPS에 이르는 실시간 성능을 발휘하며, Open-Sora 및 Latte와 같은 다양한 오픈 소스 비디오 DiT에 대해 뛰어난 안정적인 속도를 제공합니다.



### Enhanced Parking Perception by Multi-Task Fisheye Cross-view Transformers (https://arxiv.org/abs/2408.12575)
Comments:
          26th Irish Machine Vision and Image Processing Conference, Data-Driven Autonomy Workshop (matching camera-ready version)

- **What's New**: 본 논문은 Multi-Task Fisheye Cross View Transformers (MT F-CVT)라는 새로운 주차 인식 알고리즘을 소개합니다. 이 알고리즘은 4개의 fisheye Surround-view Camera System (SVCS)을 활용하여 세밀한 Bird-Eye View (BEV) 기능 맵을 생성하며, 이는 전통적인 주차 슬롯 인식 알고리즘의 오류와 제약 상황을 극복합니다.

- **Technical Details**: MT F-CVT는 segmentation decoder와 Polygon-Yolo 기반의 객체 탐지 디코더를 이용하여 주차 슬롯 및 차량을 인식합니다. 이 알고리즘은 LiDAR를 이용하여 주어진 데이터에 레이블을 붙이고, 25m x 25m 영역 내에서 평균 20 cm 오차로 물체의 위치를 파악합니다. 또한, MT F-CVT는 다양한 차량과 카메라 설정에서 강력한 일반화 능력을 보여줍니다.

- **Performance Highlights**: MT F-CVT의 큰 모델은 F-1 스코어 0.89를 달성하며, 작은 모델은 Nvidia Jetson Orin 임베디드 보드에서 16 fps로 작동하며 유사한 탐지 결과를 보입니다. 이는 주차 인식 저하 없이 실시간 실행 속도를 유지합니다.



### Sapiens: Foundation for Human Vision Models (https://arxiv.org/abs/2408.12569)
Comments:
          ECCV 2024 (Oral)

- **What's New**: Sapiens는 인간 중심 비전 작업(2D 포즈 추정, 신체 부위 분할, 깊이 추정, 표면 법선 예측)을 위해 설계된 모델 패밀리입니다. 이 모델은 1K 고해상도 추론을 지원하며, 3억 장 이상의 야외 인간 이미지로 사전 훈련되었습니다.

- **Technical Details**: Sapiens는 0.3억부터 2억 개의 파라미터를 갖는 비전 변환기(vision transformers) 모델 가족으로 구성됩니다. 우리는 masked-autoencoder(MAE) 방법을 채택하여, 간단히 고해상도 입력을 처리할 수 있도록 설계하였습니다. 또한, 3D 인간 디지털화를 위한 향상된 2D 키포인트 및 분할 클래스 어휘를 도입했습니다.

- **Performance Highlights**: Sapiens는 Humans-5K (포즈)에서 7.6 mAP, Humans-2K (부위 세분화)에서 17.1 mIoU, Hi4D (깊이)에서 22.4% 상대 RMSE, THuman2 (법선)에서 53.5% 상대 각도 오류 개선을 달성했습니다. 여러 인간 중심 기준에서 기존 모델을 지속적으로 초월합니다.



### Comparing YOLOv5 Variants for Vehicle Detection: A Performance Analysis (https://arxiv.org/abs/2408.12550)
- **What's New**: 본 연구는 다양한 환경에서의 차량 감지를 위해 YOLOv5의 다섯 가지 변형(YOLOv5n6s, YOLOv5s6s, YOLOv5m6s, YOLOv5l6s, YOLOv5x6s)을 비교 분석합니다.

- **Technical Details**: 연구는 차량 인식의 정확성과 신뢰성을 평가하기 위해 precision, recall, F1-score 및 mean Average Precision과 같은 성능 지표를 사용합니다. YOLOv5 모델은 조명, 차폐 및 날씨 등 다양한 조건에서 다수의 차량 유형(자동차, 버스, 트럭, 자전거, 오토바이)을 감지하는 효율성을 평가합니다.

- **Performance Highlights**: YOLOv5n6s는 Cars 감지에 있어 precision과 recall 간의 강력한 균형을 보여주었으며, YOLOv5s6s와 YOLOv5m6s는 recall을 개선하여 모든 관련 객체 감지 능력을 향상시켰습니다. YOLOv5l6s는 대용량으로 Cars 감지에 매우 우수하게 작동했으나, Motorcycles와 Bicycles 감지에서는 부족함을 보였습니다. YOLOv5x6s는 Buses와 Cars 감지에 효과적이었으나 Motorcycles 감지에서 어려움을 겪었습니다.



### Deep Learning Improvements for Sparse Spatial Field Reconstruction (https://arxiv.org/abs/2408.12531)
- **What's New**: 본 논문에서는 결측 데이터를 기반으로 한 전 세계적인 공간 필드 재구성을 위한 새로운 기계 학습 접근 방식을 제안합니다. 구체적으로, 여러 가지 데이터 증강 기법을 통해 재구성 속도와 정확성을 향상시켰습니다. 이를 통해 지구 과학과 유체 역학 분야의 시뮬레이션 데이터셋에서 성능 개선을 보여주었습니다.

- **Technical Details**: 제안된 방법은 크게 세 가지 데이터 증강 기법을 포함합니다: 1) Distance Transform Mask: 가장 가까운 센서와의 거리 정보를 나타내는 입력 이미지를 사용함으로써 네트워크의 위치 정보 이해도를 향상시킵니다. 2) Land Mask: 비정상적으로 무가치한 영역을 마스크 처리하여 데이터에 혼란을 주지 않습니다. 3) 이미지 정규화: 모든 입력 이미지를 -1과 1 사이의 값으로 정규화하여 재구성 속도를 증가시킵니다. 이 기법들은 이전 방법에 비해 2배에서 6배 빠른 훈련 속도를 제공하며, 재구성 오류도 감소했습니다.

- **Performance Highlights**: 개선된 기법을 활용하여 역사적인 기후 데이터 재구성이나 현재 재분석 모델의 지연 시간을 줄이는 것이 가능합니다. 이러한 데이터 증가는 기후 변화의 영향을 이해하는 데 큰 도움이 되며, 유체 역학 분야에서도 한정된 정보 아래에서 보다 나은 결정을 내릴 수 있도록 지원합니다.



### Show-o: One Single Transformer to Unify Multimodal Understanding and Generation (https://arxiv.org/abs/2408.12528)
Comments:
          Technical Report

- **What's New**: Show-o는 다양한 입력 및 출력 모드를 적응적으로 처리할 수 있는 통합된 transformer 모델이다. 이 모델은 기존의 개별 모델들과 비교하여 우수한 성능을 보여주며, 오토리그레시브 모델과 디퓨전 모델링을 융합하였다.

- **Technical Details**: Show-o는 텍스트는 이산 토큰으로 그리고 이미지는 연속 픽셀로 모델링하여 multimodal 이해와 생성을 가능하게 한다. 또한, 별도의 텍스트 인코더 없이 텍스트 조건 정보의 인코딩을 내장하고 있으며, 다양한 입력 데이터와 작업 변형에 대응할 수 있도록 설계되었다.

- **Performance Highlights**: Show-o는 동일한 파라미터 수를 가진 기존 모델들보다 뛰어난 성능을 자랑하며, 약 20배 적은 샘플링 스텝으로 이미지를 생성할 수 있는 잠재력을 보였다. 다양한 다운스트림 애플리케이션을 지원하며, 텍스트 기반의 인페인팅과 합성 작업을 위한 기본 설정으로 활용 가능하다.



### Scribbles for All: Benchmarking Scribble Supervised Segmentation Across Datasets (https://arxiv.org/abs/2408.12489)
Comments:
          under review

- **What's New**: 본 연구에서는 scribble labels를 기반으로 한 semantic segmentation을 위한 데이터 생성 알고리즘 'Scribbles for All'을 소개합니다. 이 알고리즘은 수동 라벨링에 비해 훨씬 적은 주석 작업으로 높은 품질의 분할 결과를 이끌어냅니다.

- **Technical Details**: Scribbles for All은 다양한 인기 있는 세분화 데이터셋을 위한 scribble labels을 제공하며, 밀집 주석이 있는 데이터셋에서 자동으로 scribble labels을 생성할 수 있는 알고리즘을 제공합니다. 이는 weakly supervised segmentation 분야에서 새롭고 중요한 통찰력을 제공합니다.

- **Performance Highlights**: 자동 생성된 scribble labels로 훈련된 모델들은 수동으로 작성된 scribble labels로 훈련된 모델들과 경쟁력 있는 성능을 보여줍니다. 이를 통해 비록 기존의 PascalVOC와 같은 단순한 데이터셋에서 시작했으나, 더 복잡한 데이터셋에서도 높은 정확도의 results를 낼 수 있도록 합니다.



### Not All Samples Should Be Utilized Equally: Towards Understanding and Improving Dataset Distillation (https://arxiv.org/abs/2408.12483)
- **What's New**: 이 논문에서는 Dataset Distillation(DD) 분야의 이론적 탐색을 처음으로 시도하며, 샘플 난이도를 통해 다양한 매칭 기반 DD 방법을 이해하려고 합니다. 샘플 난이도와 데이터 품질 간의 관계를 강조하고, Sample Difficulty Correction(SDC) 방법을 제안하여 학습 과정에서 더 쉬운 샘플에 집중하도록 합니다.

- **Technical Details**: 샘플 난이도는 gradient norm으로 측정되며, GM(Gradient Matching) 기반 방법은 어려운 샘플을 배우는 반면 TM(Trajectory Matching) 기반 방법은 상대적으로 쉬운 샘플을 선호함을 발견했습니다. 논문에서는 neural scaling laws를 데이터 프루닝 이론에서 DD로 확장하여 결과를 설명합니다. SDC는 기존의 매칭 기반 방법에 최소한의 코드 수정으로 통합 가능하도록 설계되었습니다.

- **Performance Highlights**: SDC를 추가함으로써 7개의 DD 방법(DC, DSA, DSAC, MTT, FTD, TESLA, DATM)과 6개의 데이터셋(MNIST, FashionMNIST, SVHN, CIFAR-10, CIFAR-100, Tiny ImageNet)에서 더 높은 품질의 증류된 데이터셋을 생성할 수 있음을 보였습니다.



### Frame Order Matters: A Temporal Sequence-Aware Model for Few-Shot Action Recognition (https://arxiv.org/abs/2408.12475)
Comments:
          9 pages, 6 figures

- **What's New**: 본 연구에서는 동적 시간 순서와 공간 정보를 결합하여 feature embeddings를 추출하는 새로운 Temporal Sequence-Aware Model (TSAM)을 제안합니다. TSAM은 사전 훈련된 visual backbone에 sequential perceiver adapter를 통합하여 동적 시퀀스를 포착합니다.

- **Technical Details**: TSAM은 비디오 프레임의 본질적 순서를 포착하기 위해 temporal query를 활용하며, 특정 클래스의 feature representations를 강화하기 위해 Large Language Models (LLMs)에서 파생된 확장된 텍스트 프롬프트를 구축합니다. 또한, support 및 query 비디오의 매칭을 위한 unbalanced optimal transport 전략을 도입하여 비판별적이고 클래스와 관련 없는 feature의 영향을 줄입니다.

- **Performance Highlights**: 실험 결과, TSAM은 5개의 FSAR 데이터세트에서 기존의 두 번째로 좋은 경쟁자들을 큰 차이로 초월하며 새로운 성과 기준을 확립하였습니다.



### Envisioning Class Entity Reasoning by Large Language Models for Few-shot Learning (https://arxiv.org/abs/2408.12469)
Comments:
          9 pages, 7 figures

- **What's New**: 본 논문은 Few-Shot Learning (FSL) 분야에서의 새로운 프레임워크를 제안하고, 이는 Large Language Models (LLMs)에서 추출된 구체적인 클래스 엔티티를 활용하여 FSL의 클래스 프로토타입 표현을 강화하는 방식입니다.

- **Technical Details**: 제안된 프레임워크는 Semantic-guided Visual Pattern Extraction (SVPE) 모듈과 Prototype-Calibration (PC) 모듈로 구성됩니다. SVPE 모듈은 다양한 스케일에서 의미 인식 비주얼 패턴을 정교하게 추출하고, PC 모듈은 이러한 패턴을 통합하여 비주얼 프로토타입을 개선합니다. 또한, Progressive Visual-Semantic Aggregation (PVSA) 전략을 통해 시맨틱 정보를 점진적으로 통합합니다.

- **Performance Highlights**: 4개의 FSL 벤치마크 및 BSCD-FSL 크로스 도메인 벤치마크에서의 실험 결과, 우리의 방법은 현재 최신 기술보다 상당한 성능을 보여주었으며, 특히 하나의 샘플만 사용하는 설정에서 ResNet-12 백본을 이용하여 평균 1.95% 향상을 기록했습니다.



### WCEbleedGen: A wireless capsule endoscopy dataset and its benchmarking for automatic bleeding classification, detection, and segmentation (https://arxiv.org/abs/2408.12466)
- **What's New**: 이 연구에서는 자동 분류(classification), 탐지(detection), 및 세분화(segmentation)를 위한 새로운 의료 주석이 있는 무선 캡슐 내시경(WCE) 데이터셋인 WCEbleedGen을 개발하였다. 이 데이터셋은 총 2,618개의 출혈 및 비출혈 프레임으로 구성되어 있으며, 깊은 학습 모델을 통해 평가되었다.

- **Technical Details**: WCEbleedGen 데이터셋은 고해상도 프레임으로 이루어져 있으며, 출혈과 비출혈 클래스를 균형 있게 포함하고 있다. 데이터셋에는 클래스 레이블(class labels), 수동 생성 이진 마스크(binary masks), 및 정확한 경계 상자(bounding boxes)가 포함되어 있어 고급 분석을 위한 강력한 자원이다. 평가에는 9개의 분류 모델, 3개의 탐지 모델, 및 3개의 세분화 모델이 사용되었다.

- **Performance Highlights**: 비주얼 기하학 그룹(VGG) 19, YOLOv8n, Linknet이 각각 자동 분류, 탐지, 세분화 기반 평가에서 최고의 성과를 기록하였다. 이 연구는 WCE 비디오 해석을 위한 자동 출혈 진단의 중요성을 강조하며, 이 데이터셋은 WCE에서의 자동 출혈 진단을 위한 혁신적인 솔루션 개발에 기여할 것이다.



### Smartphone-based Eye Tracking System using Edge Intelligence and Model Optimisation (https://arxiv.org/abs/2408.12463)
- **What's New**: 최근 스마트폰 기반 시선 추적 알고리즘의 정확도가 동적 비디오 자극에 적용 시 낮다는 제한이 있었습니다. 이 논문에서는 CNN(Convolutional Neural Networks)과 LSTM(Long Short Term Memory) 및 GRU(Gated Recurrent Unit)라는 두 개의 RNN(Recurrent Neural Networks)을 결합하여 비디오 형식을 위한 두 가지 새로운 시선 추적 기술을 개발했습니다.

- **Technical Details**: CNN+LSTM 및 CNN+GRU 모델을 통해 각각 0.955cm 및 1.091cm의 평균 Root Mean Square Error(RMSE)를 달성했습니다. 또한, 스마트폰의 계산 제약을 해결하기 위해 엣지 인텔리전스 아키텍처를 개발했으며, 모델 최적화 기법인 양자화(quantization)와 가지치기(pruning)를 적용하여 엣지 장치에서의 성능을 향상시켰습니다. CNN+LSTM과 CNN+GRU 모델의 경우, 양자화를 사용하여 엣지 장치에서 모델 추론 시간이 각각 21.72% 및 19.50% 단축되었습니다.

- **Performance Highlights**: 스마트폰 기반 시선 추적 알고리즘의 정확성을 향상시키기 위해 CNN과 RNN을 결합한 구조를 사용하였으며, 이를 통해 비디오 자극에 대한 시선 추정 정확성을 높이는데 기여하였습니다. 또한, 다양한 엣지 장치에서의 CPU 및 메모리 사용량과 에너지 소비를 평가하여, 실시간 처리가 가능한 스마트폰 시장의 수요를 충족할 수 있는 가능성을 보여주었습니다.



### Finding Closure: A Closer Look at the Gestalt Law of Closure in Convolutional Neural Networks (https://arxiv.org/abs/2408.12460)
- **What's New**: 이번 연구에서는 신경망이 Gestalt 법칙 중 하나인 Closure 가설을 내재적으로 적용할 수 있는지를 조사하였습니다. 이를 위해, Modal과 Amodal completion을 포함한 Closure 효과를 테스트하기 위한 잘 정리된 데이터셋을 제공합니다.

- **Technical Details**: 연구팀은 VGG16, DenseNet-121 등 다양한 CNN 구조를 사용하여 각각의 Closure 효과를 실험하고, 심리학 이론을 기반으로 한 분석을 통해 결과를 해석하였습니다. 주요 기술적 요소로는 CNN의 프로세스를 이해하고, Contour completion과 Contour integration에 대한 이론적 배경을 포함하고 있습니다.

- **Performance Highlights**: VGG16과 DenseNet-121은 Closure 효과를 잘 나타내었으나, 다른 CNN 모델들은 가변적인 결과를 보였습니다. 연구 결과는 신경망이 인간의 시각 지각 과정을 mimic 할 수 있는지를 더욱 명확히 하고, AI 시스템 설계에 중요한 통찰력을 제공합니다.



### Relaxed Rotational Equivariance via $G$-Biases in Vision (https://arxiv.org/abs/2408.12454)
- **What's New**: 이번 논문에서는 기존의 Group Equivariant Convolution(GConv)이 직렬 회전 대칭을 효과적으로 처리할 수 있지만, 현실 세계의 데이터에서 흔히 발생하는 회전 대칭 파괴(rotational symmetry-breaking)에 적응하는 데 어려움을 겪는 문제를 해결하기 위한 새로운 접근법을 제안합니다.

- **Technical Details**: 우리는 GConv에서 엄격한 그룹 제약을 깨기 위해 $G$-Biases라는 학습 가능한 바이어스를 도입하여 Relaxed Rotational Equivariant Convolution(RREConv)을 구현합니다. RREConv는 기존의 GConv보다 더 적은 매개변수로 높은 성능을 보이며, 다양한 모델에서 간편하게 사용할 수 있는 플러그 앤 플레이 모듈로 설계되었습니다.

- **Performance Highlights**: RREConv 기반 방법이 자연 이미지 데이터셋에서 분류 및 탐지 작업을 수행할 때 기존의 GConv 기반 방법들과 비교하여 우수한 성능을 달성했음을 실험을 통해 입증했습니다.



### The 2nd Solution for LSVOS Challenge RVOS Track: Spatial-temporal Refinement for Consistent Semantic Segmentation (https://arxiv.org/abs/2408.12447)
- **What's New**: 이번 연구에서는 Referring Video Object Segmentation (RVOS) 과제를 다루며, 새로운 Segment Anything Model version 2 (SAM-v2)의 추적 기능을 활용하여 세그멘테이션의 시간적 일관성을 향상시키는 방법을 제시합니다.

- **Technical Details**: 우리의 프레임워크는 SAM-v2를 사용하여 추적 정보가 포함된 시공간 마스크를 추출하고, MUTR 모델을 정제하여 초기 세그멘테이션 마스크를 생성합니다. 이후 Spatial-Temporal Refinement Module을 통해 최종 세그멘테이션 마스크를 생성합니다.

- **Performance Highlights**: 우리 방법은 MeViS 데이터셋의 테스트 세트에서 60.40의 \\mathcal{J\text{\&}F} 점수를 달성하여 ECCV 2024 LSVOS Challenge의 RVOS Track 최종 순위에서 2위를 차지했습니다.



### A Riemannian Approach for Spatiotemporal Analysis and Generation of 4D Tree-shaped Structures (https://arxiv.org/abs/2408.12443)
- **What's New**: 본 논문에서는 시간에 따라 변형되는 나무 모양의 3D 객체, 즉 4D 나무 구조의 공간-시간(shape variability) 변동성을 모델링하고 분석하기 위한 최초의 포괄적인 접근 방식을 제안합니다. 중요한 기여는 Square Root Velocity Function Trees (SRVFT)를 사용하여 나무 모양의 3D 구조를 표현하여 4D 구조를 시간-매개변수화된 궤적으로 변환하는 것입니다.

- **Technical Details**: 이 논문에서는 SRVF(Square Root Velocity Functions) 공간 내에서의 공간 등록 문제를 해결하여 나무 모양의 4D 구조를 궤적으로 표현합니다. 이러한 접근 방식을 통해 4D 나무 모양을 모델링 및 분석하는 문제를 SRVF 공간 내의 탄성 궤적(elastic trajectories) 분석 문제로 감소시킵니다. 연구에서는 Riemannian metric과 계산 도구를 제안하여 빠르고 정확한 공간-시간 등록(spatiotemporal registration) 및 기하학적 경로(geodesics) 계산을 수행합니다.

- **Performance Highlights**: 제안된 도구는 실제 4D 식물 데이터(예: 성장하는 토마토 및 옥수수 식물)를 사용하여 효율성을 검증하였습니다. 이 프레임워크는 통계 모델을 활용하여 공간-시간 변동성을 모델링하고, 예제 집합으로부터 새로운 4D 나무 구조를 생성하는 기능을 통합합니다.



### Adapting MIMO video restoration networks to low latency constraints (https://arxiv.org/abs/2408.12439)
Comments:
          See the project web page to download the associated videos

- **What's New**: MIMO (multiple input, multiple output) 아키텍처를 통한 저지연 비디오 복원 기술에 대한 새로운 접근 방식을 제안하며, 스택 전환의 시간 불연속성 문제를 해결하기 위한 두 가지 해결책을 소개합니다.

- **Technical Details**: 비디오를 겹치지 않는 프레임 스택으로 나눈 후, 각각을 독립적으로 처리하여 계산 비용과 출력 품질 간의 균형을 추구합니다. 제안된 방법은 MIMO 아키텍처에 적용될 수 있으며, 두 가지 주요 개선 사항인 스택 간 재발과 스택 겹침을 통해 품질을 향상시킵니다.

- **Performance Highlights**: 세 가지 최첨단 비디오 노이즈 제거 네트워크에 대한 실험 결과, 저지연 네트워크의 경우 재구성 오류와 시간 일관성 측면에서 새로운 최첨단 성능을 달성했습니다. 또한, 드론 비디오의 새로운 벤치마크를 소개하여 시간 일관성 문제를 강조했습니다.



### FlexEdit: Marrying Free-Shape Masks to VLLM for Flexible Image Editing (https://arxiv.org/abs/2408.12429)
Comments:
          15 pages, 14 figures

- **What's New**: FlexEdit는 자유 형태의 마스크와 언어 지시사항을 결합하여 기존의 이미지 편집 방법의 한계를 극복하는 엔드 투 엔드 이미지 편집 방법입니다.

- **Technical Details**: FlexEdit은 Vision Large Language Model (VLLM)을 이용하여 이미지 콘텐츠, 마스크, 및 사용자 지시사항을 이해합니다. 또한, Mask Enhance Adapter (MEA) 구조를 도입하여 VLLM의 임베딩을 이미지 데이터와 통합함으로써 시각 정보와 모델 출력 임베딩의 원활한 통합을 보장합니다. 새로운 FSIM-Edit 벤치마크를 구축하여 다양한 자유 형태의 마스크를 평가하고 있습니다.

- **Performance Highlights**: 대규모 실험 결과, FlexEdit은 자유 형태의 마스크 조건에서 기존 방법보다 월등한 성능을 보였으며, 간단한 프롬프트 기법이 높은 효과를 입증했습니다. 이 연구는 사용자 친화적이고 강력한 이미지 편집 솔루션을 제공하는 큰 진전을 나타냅니다.



### Enhanced Infield Agriculture with Interpretable Machine Learning Approaches for Crop Classification (https://arxiv.org/abs/2408.12426)
- **What's New**: 이번 연구는 농업 분야에서 AI를 통한 이미지 분류 기법의 발전을 다루고 있습니다. 전통적인 머신러닝 기법에서 최첨단 foundation 모델까지, 다양한 접근 방식을 비교하고 각 모델의 Explainable AI 기능을 평가하여 모델의 해석 가능성을 높였습니다.

- **Technical Details**: 연구에서는 SIFT, ORB, Color Histogram과 같은 핸드크래프트(feature extraction) 기법을 활용한 전통적인 ML 알고리즘과 커스텀 CNN, 전이 학습(Transfer Learning) 기법을 적용한 다양한 사전 훈련된 DL 아키텍처 (EfficientNetV2, ResNet152V2, Xception, Inception-ResNetV2, MobileNetV3)에 대한 비교가 있었습니다. Xception 모델이 98%의 정확도로 가장 높은 성능을 보였으며, LIME, SHAP, Grad-CAM을 사용하여 모델의 해석 가능성을 확보했습니다.

- **Performance Highlights**: Xception 모델은 80.03 MB의 모델 크기와 0.0633초의 예측 시간을 기록하여 다른 모든 모델보다 우수한 일반화 성능을 보여주었습니다. 연구는 또한 AI를 통한 농업 관리 전략의 개선 및 효율성을 위한 해석 가능성의 중요성을 강조합니다.



### CODE: Confident Ordinary Differential Editing (https://arxiv.org/abs/2408.12418)
- **What's New**: 새로운 연구는 Confident Ordinary Differential Editing (CODE)라는 접근 방식을 소개합니다. CODE는 Out-of-Distribution (OoD) 이미지를 효과적으로 처리하여 이미지 생성 및 편집을 쉽게 할 수 있도록 돕습니다.

- **Technical Details**: CODE는 이미지를 보강하기 위해 확률 흐름 Ordinary Differential Equation (ODE)을 따르는 점수 기반 업데이트를 사용합니다. 이 방법은 추가적인 학습이나 수작업 모듈 없이, 노이즈가 있는 이미지에 대해 고유한 복원 과정을 통해 작동합니다.

- **Performance Highlights**: 실험 결과, CODE는 SDEdit보다 더 나은 사실감과 충실도를 보여 주었으며, 특히 심각하게 손상된 입력의 경우에 더욱 뛰어난 성능을 보였습니다.



### Generalized SAM: Efficient Fine-Tuning of SAM for Variable Input Image Sizes (https://arxiv.org/abs/2408.12406)
Comments:
          Accepted by ECCV2024 Workshop "Computational Aspects of Deep Learning (CADL)"

- **What's New**: 이번 연구에서는 Segment Anything Model (SAM)의 효율적인 fine-tuning (파인튜닝) 방법인 Generalized SAM (GSAM)을 제안했습니다. GSAM은 input image size (입력 이미지 크기)가 가변적일 수 있도록 하여, training 중 랜덤 크로핑을 적용했습니다.

- **Technical Details**: GSAM은 Positional Encoding Generator (PEG)를 활용하여 가변 입력 이미지 크기를 지원합니다. 그리고 Spatial-Multiscale (SM) AdaptFormer를 도입하여, fine-tuning 과정에서 더 다양한 spatial information (공간 정보)을 고려합니다. GSAM은 모든 weight parameters (가중치 파라미터)의 일부를 고정한 상태에서, 나머지 파라미터만 업데이트하여 fine-tuning을 진행합니다.

- **Performance Highlights**: 실험 결과, GSAM은 기존 SAM 및 다른 fine-tuning 방법들보다 더 효율적으로 학습할 수 있으며, 특히 CT 이미지로 구성된 Synapse multi-organ dataset (데이터셋)에서 11% 이상의 향상된 segmentation accuracy (세분화 정확도)를 달성했습니다. GSAM은 낮은 컴퓨팅 비용과 높은 정확도의 균형을 잘 맞추었습니다.



### Multi-Style Facial Sketch Synthesis through Masked Generative Modeling (https://arxiv.org/abs/2408.12400)
- **What's New**: 본 연구에서는 주어진 얼굴 사진으로부터 스케치 초상화를 생성하는 얼굴 스케치 합성(FSS) 모델을 제안합니다. 개발한 경량화된 end-to-end 합성 모델은 추가 입력 없이 효율적으로 이미지를 다양한 스타일의 스케치로 변환합니다.

- **Technical Details**: 제안된 모델은 반지도 학습(semi-supervised learning)을 통합하여 데이터 부족 문제를 해결하며, 특징 추출 모듈(feature extraction module) 및 스타일 임베딩(style embeddings)을 사용해 생성적 변환기(generative transformer)를 조정합니다. 모델은 VQ-GAN의 토큰 공간(token space) 내에서 생성을 제한하며, 마스킹된 이미지 토큰의 반복 예측 과정에서 스케치로서의 얼굴 특징을 정확히 유지합니다.

- **Performance Highlights**: 폭넓은 실험 결과, 본 연구의 방법론은 기존 알고리즘 대비 여러 벤치마크에서 일관되게 우수한 성능을 보여주었으며, 생성된 스케치의 품질과 구조적 유사도 측면에서 뚜렷한 차이를 입증하였습니다.



### Cross-Domain Foundation Model Adaptation: Pioneering Computer Vision Models for Geophysical Data Analysis (https://arxiv.org/abs/2408.12396)
- **What's New**: 이 연구는 컴퓨터 비전의 foundation models (FMs)을 지구과학 분야에 적응시키는 방법을 탐구합니다. 이는 대량의 데이터셋에서 학습된 대형 신경망이 다양한 작업에서 뛰어난 적응성과 일반성을 보이는 점을 활용하여, 지구과학 데이터 분석을 위한 새로운 접근 방식을 제시합니다.

- **Technical Details**: 연구는 '제목없는 대규모 데이터셋'(large-scale, well-curated datasets)과 높은 비용을 수반하는 FMs 개발의 어려움을 언급합니다. 지구과학 데이터의 다양성과 비표준화된 데이터 처리 과정이 FMs에 영향을 미치는 것을 관찰하였고, 특히 제한된 라벨 데이터로도 잘 작동하는 모델을 위한 적응 과정을 설명합니다.

- **Performance Highlights**: 실험을 통해 이 과정이 월면 이미지, 지진 데이터 및 DAS 배열 등을 처리하고 해석하는 광범위한 응용 프로그램에서 효과적임을 입증하였습니다. 이를 통해 다른 과학 분야에서의 FMs 응용 가능성을 확인하며, 고급 머신 러닝 기법을 지구과학에 도입하는 데 기여하고 있습니다.



### Makeup-Guided Facial Privacy Protection via Untrained Neural Network Priors (https://arxiv.org/abs/2408.12387)
Comments:
          Proceedings of ECCV Workshop on Explainable AI for Biometrics, 2024

- **What's New**: 본 논문에서는 얼굴 인식(FR) 시스템의 개인 정보 보호를 위한 새로운 방법인 Deep Facial Privacy Prior (DFPP)를 제안합니다. 이 방법은 대규모 메이크업 데이터 세트를 통한 훈련 없이도 자연스럽고 적대적인 메이크업 스타일 전이를 가능하게 하여 데이터 세트 편향 문제를 해결합니다.

- **Technical Details**: DFPP는 두 가지 주요 모듈로 구성됩니다: (1) 잠재 공간에서 참조 이미지와 원본 이미지 간의 지역 정렬을 수행하는 correspondence 모듈, (2) 조건부 메이크업 레이어가 포함된 디코더입니다. 디코더는 구조적 및 메이크업 일관성 손실을 기반으로 최적화되어 보호된 이미지를 생성하며, 이 과정에서 원본의 정체성을 유지하고, 참조 이미지의 메이크업 스타일을 채택합니다.

- **Performance Highlights**: 실험 결과, DFPP는 얼굴 확인 및 식별 작업에서 상업적인 FR 시스템 및 악의적인 블랙박스 FR 모델을 효과적으로 회피하는 성능을 보여주었습니다. 특히, 비디오에서의 적용 가능성도 강조되어 시간이 지남에 따라 여러 프레임 간의 최적화된 가중치를 전이하여 10배의 계산 효율성을 달성하며 개인 정보 보호를 유지합니다.



### Sampling Strategies based on Wisdom of Crowds for Amazon Deforestation Detection (https://arxiv.org/abs/2408.12381)
Comments:
          6 pages, 5 figus, paper accepted at the SIBGRAPI 2024

- **What's New**: 이 논문에서는 Citizen Science와 Machine Learning 모델을 활용한 ForestEyes 프로젝트를 통해 아마존 지역의 산림 파괴 모니터링을 지원하는 새로운 방법론을 제안합니다.

- **Technical Details**: ForestEyes 프로젝트는 자원봉사자들이 Remote Sensing 이미지를 분류하여 Machine Learning 모델을 학습시키는 데이터셋을 생성합니다. 이 연구에서는 SVM(Support Vector Machine) 기법을 사용하며, 사용자 엔트로피(user entropy)를 기반으로 한 샘플링 전략을 통해 학습 정확도를 극대화합니다.

- **Performance Highlights**: 사용자 엔트로피를 사용한 접근 방식은 무작위 샘플링 전략에 비해 산림 파괴 감지 작업에서 최고의 분류 결과를 달성했으며, SVM 기법의 수렴 시간을 단축시켰습니다.



### UMERegRobust -- Universal Manifold Embedding Compatible Features for Robust Point Cloud Registration (https://arxiv.org/abs/2408.12380)
Comments:
          ECCV 2024

- **What's New**: 이 논문에서는 Universal Manifold Embedding (UME) 프레임워크를 사용하여 강체 변환(rigid transformation)의 추정 방법을 제안하고, 부분 겹침(partial overlap)과 서로 다른 샘플링 패턴(sampling patterns)을 포함하는 시나리오를 수용할 수 있도록 확장합니다.

- **Technical Details**: UME는 동일한 객체에 대한 관찰(observations)을 단일 저차원 선형 부분공간(low-dimensional linear subspace)으로 매핑하는 방법론으로, 변환에 대해 불변(transformation-invariant)인 표현을 생성합니다. 새로운 UME 호응 손실(contrastive loss)과 샘플링 균형 조정(equalizer)을 포함하는 특성 추출(feature extraction) 과정을 통해 UMERegRobust라는 강력한 등록 파이프라인으로 통합됩니다. 또한, RotKITTI 등록 기준점(benchmark)을 제안하여 대회전(scenarios involving large rotations)을 평가합니다.

- **Performance Highlights**: 제안하는 UMERegRobust는 KITTI 벤치마크에서 최첨단 성능(state-of-the-art performance)을 초과했으며, 특히 (1°, 10cm)의 엄격한 정밀도(strict precision)가 고려될 때 평균적으로 9% 향상된 결과를 보였습니다. RotKITTI 벤치마크에서는 최근의 최신 방법에 비해 45% 성능 향상을 달성했습니다.



### SAM-SP: Self-Prompting Makes SAM Great Again (https://arxiv.org/abs/2408.12364)
Comments:
          Under Review

- **What's New**: 최근 발표된 Segment Anything Model (SAM)은 다양한 자연 이미지 데이터셋에서 제로샷(segmentation) 기술을 보이며 뛰어난 능력을 보여주고 있습니다. 하지만 SAM은 의료 이미지와 같은 특정 도메인에 적용될 때 성능 저하가 발생하는 문제가 있습니다. 본 논문에서는 이러한 문제를 해결하기 위한 새로운 자가 프롬프트(self-prompting) 기반의 파인튜닝 방법 SAM-SP를 소개합니다.

- **Technical Details**: SAM-SP는 모델의 이전 반복에서의 출력을 프롬프트로 활용하여 이후 반복을 이끌어내는 방식으로 자가 프롬프트 모듈을 사용합니다. 이를 통해 전문가가 제공하는 프롬프트의 필요성을 줄이고 SAM의 적용 가능성을 크게 확장할 수 있습니다. 또한, 자가 증류(self-distillation) 방법을 통합하여 자가 프롬프트 과정을 더욱 강화하고 있습니다.

- **Performance Highlights**: 다양한 도메인 특화 데이터셋에서 실시한 광범위한 실험을 통해, SAM-SP는 사용자 프롬프트 없이도 우수한 성능을 보이며, 기존의 최신 작업 특화(segmentation) 접근법들 및 기본 SAM과 비교하여 뛰어난 성능을 나타냅니다.



### Class-balanced Open-set Semi-supervised Object Detection for Medical Images (https://arxiv.org/abs/2408.12355)
- **What's New**: 이 연구에서는 의료 이미지에서의 반지도 객체 탐지 문제를 해결하기 위해 OOD(Out-Of-Distribution) 클래스를 포함한 비표시 데이터(unlabeled data)를 활용하는 새로운 방법론을 제시합니다.

- **Technical Details**: 제안된 방법론은 두 가지 주요 혁신을 포함합니다: 1) CCE(Category Control Embed) 모듈을 통해 데이터 세트 불균형을 해소하고, 2) OODFC(Out-of-Distribution Detection Fusion Classifier) 모듈을 통해 '알 수 없는(unknown)' 정보를 기본 가짜 레이블(pseudo-labels)에 통합합니다. 이 설정에서는 Mean-Teacher(미국의 평균 교사 방법)로 명명된 SSOD(Semi-Supervised Object Detection) 파이프라인을 기반으로 한 신경망 구조가 사용됩니다.

- **Performance Highlights**: 제안된 방법은 퍼블릭 Parasite 데이터셋에서 기존 SSOD 성능 기준보다 4.25 mAP(mean Average Precision)가 향상되었습니다.



### GarmentAligner: Text-to-Garment Generation via Retrieval-augmented Multi-level Corrections (https://arxiv.org/abs/2408.12352)
- **What's New**: GarmentAligner는 새로운 종류의 text-to-garment diffusion model로, retrieval-augmented multi-level corrections를 통해 의류 생성의 상징적 오해를 해결합니다.

- **Technical Details**: GarmentAligner는 자동화된 component extraction pipeline을 통해 의류 이미지와 캡션에서 구성 요소의 공간적 및 양적 정보를 추출합니다. 또한, component 수준의 유사성 순위를 바탕으로 retrieval subsets를 구성하여 대조 학습을 진행하며, 이를 통해 구성 요소 간의 관계를 인식합니다.

- **Performance Highlights**: GarmentAligner는 기존의 경쟁 모델들에 비해 뛰어난 충실도와 세밀한 의미 정렬을 달성하며, CM-Fashion 데이터셋을 통해 수행된 실험에서 우수함을 입증했습니다.



### VTON-HandFit: Virtual Try-on for Arbitrary Hand Pose Guided by Hand Priors Embedding (https://arxiv.org/abs/2408.12340)
- **What's New**: 이 논문에서는 손 가림(hand occlusion) 문제를 효과적으로 해결하기 위한 새로운 방법, VTON-HandFit을 제안합니다. 기존의 방법보다 한 단계 발전된 기법을 통해, 실제 시나리오에서 발생하는 손 가림 문제를 처리할 수 있는 기술을 제공합니다.

- **Technical Details**: VTON-HandFit은 손의 외관과 구조를 재구성하기 위해 Handpose Aggregation Net을 사용합니다. 이 구조는 ControlNet을 기반으로 하여 글로벌 손 및 포즈 priors를 명시적이고 적응적으로 인코딩합니다. 또한, Hand-feature Disentanglement Embedding 모듈을 통해 손 priors를 구조적-매개변수 및 시각적-외관 특성으로 분리하며, 이를 위해 마스크된 교차 주의(masked cross attention) 메커니즘을 커스터마이즈합니다. 마지막으로, 손 템플릿에서 구조적 경계 지식을 학습하기 위한 손-canny 제약 손실을 설정하여 모델의 성능을 증가시킵니다.

- **Performance Highlights**: VTON-HandFit은 공개 데이터셋과 자가 수집한 Handfit-3K 데이터셋에서 정성적 및 정량적 평가에서 기존 방법들을 능가하여 특히 손 가림 사례에 대한 성능이 향상되었습니다. 또한, 테스트 단계에서의 추론 시간을 분석하여 배치 크기를 1로 설정하고, 이미지 해상도를 768x1024로 설정하여 PyTorch를 통해 독립된 비교를 수행한 결과, 경쟁력 있는 성능을 유지했습니다.



### Multimodal Foundational Models for Unsupervised 3D General Obstacle Detection (https://arxiv.org/abs/2408.12322)
- **What's New**: 본 논문은 기존의 감독학습 방식으로는 다루기 힘든 특정하지 않은 장애물 탐지 문제를 해결하기 위한 새로운 접근 방식을 제시합니다. 복잡하고 다양한 장애물 범주를 다루기 위해 다중모달 기초 모델(multimodal foundational model)과 전통적인 비감독적 컴퓨테이셔널 기하학(outlier detection) 기술을 결합하였습니다.

- **Technical Details**: 제안된 방법은 오프라인으로 작동하며, 훈련이 필요 없는 방법을 활용하여 3D 공간에서 일반 장애물을 탐지할 수 있습니다. 기존의 장애물 탐지 데이터 세트의 한계를 극복하기 위해 다양한 장애물에 대한 데이터 세트를 수집하고 주석을 달았습니다. Grounding DINO 및 Segment Anything과 같은 기초 모델을 활용하여 로드 서페이스의 일반 장애물 후보를 분할합니다.

- **Performance Highlights**: 본 연구는 학습이 필요 없는 3D 일반 장애물 탐지 방법을 개발하여, 비싼 재훈련 없이도 새로운 장면에서 이전에 보지 못한 장애물을 탐지할 수 있음을 입증하였습니다. 이는 자율주행차의 안전성과 신뢰성을 높이는 데 기여할 것으로 기대됩니다.



### Adapt CLIP as Aggregation Instructor for Image Dehazing (https://arxiv.org/abs/2408.12317)
Comments:
          12 pages, 6 figures

- **What's New**: 본 연구에서는 CLIPHaze라는 혁신적인 하이브리드 프레임워크를 도입하여, Mamba의 효율적인 글로벌 모델링과 CLIP의 사전 지식 및 제로샷 능력을 결합하여 이미지 디해이징 시의 제한된 수용 범위와 시맨틱 프라이어 활용 부족 문제를 동시에 해결합니다.

- **Technical Details**: CLIPHaze는 Transformer-Mamba Dual Aggregation (TrambaDA) 블록을 기본 단위로 하며, 이 두 경로를 통해 각각 지역적 세밀한 인식과 글로벌 수용 범위를 모델링합니다. 특히 CLIP-instructed Aggregation Module (CAM)을 통해 이미지의 흐릿한 정도에 따라 네트워크가 적절한 수용 범위를 결정할 수 있도록 합니다.

- **Performance Highlights**: CLIPHaze는 다양한 벤치마크에서 최첨단(SOTA) 성능을 달성하였으며, 특히 비균질(homogeneous)해이 제거에서 뛰어난 결과를 보였습니다. 각 경로의 정보 융합을 위해 제안된 CAM 모듈은 CLIP을 활용하여 사전 지식 기반으로 최적의 픽셀 연산 범위를 이를 통해 달성합니다.



### Unrolled Decomposed Unpaired Learning for Controllable Low-Light Video Enhancemen (https://arxiv.org/abs/2408.12316)
- **What's New**: 본 논문에서는 쌍을 이루지 않은 데이터셋을 사용하여 저조도 비디오 강화(low-light video enhancement)를 수행하기 위한 Unrolled Decomposed Unpaired Network (UDU-Net)이라는 새로운 접근법을 제시합니다. 이는 기존 이미지 기반 방법의 한계를 극복하고, 시간적 일관성을 유지하며, 과다 노출 및 부족 노출 조건을 방지하는 메커니즘을 통합합니다.

- **Technical Details**: 본 연구는 저조도 비디오 강화 문제를 Maximum A Posteriori (MAP) 추정 문제로 공식화하였으며, 공간적(spatial) 및 시간적(temporal) 시각적 규제(visual regularization)를 설계하였습니다. UDU-Net은 이러한 최적화 기능을 심층 신경망(deep network)으로 풀어내어 공간적 및 시간적 관련 요소로 신호를 분해하며, 이는 단계적으로 업데이트됩니다. Intra subnet과 Inter subnet을 통해 각각 공간적 정보와 시간적 단서를 활용하여 비디오 강화 결과를 향상시킵니다.

- **Performance Highlights**: UDU-Net은 야외 및 실내 장면에서 비디오 조명(video illumination), 노이즈 억제(noise suppression), 시간적 일관성(temporal consistency) 측면에서 기존의 최첨단 방법들을 능가하는 성능을 보여주었습니다. 또한, 제안된 방법은 특정 경우에서 캡처된 참조(reference)와 비슷하거나 우수한 성능을 달성하였습니다.



### MakeupAttack: Feature Space Black-box Backdoor Attack on Face Recognition via Makeup Transfer (https://arxiv.org/abs/2408.12312)
- **What's New**: 이 연구에서는 메이크업 전이(makeup transfer)를 이용한 새로운 특징 공간(Feature space) 백도어 공격(Backdoor Attack), 즉 MakeupAttack을 제안합니다.

- **Technical Details**: MakeupAttack은 모델에 대한 전체 접근 권한(full access)이 필요하지 않고, 모델 쿼리(model queries)만으로 작동하며 블랙박스 공격(black-box attack) 원칙을 준수합니다. 이를 위해, 쿼리 샘플의 미세한 특징을 학습하는 반복(training) 패러다임을 설계했습니다. 또한, 트리거(triggers)의 다양성을 높이기 위해 적응 선택(adaptive selection) 방법을 사용하여 악성 샘플의 특징 분포(feature distribution)를 분산시켜 기존 방어 방법을 우회합니다.

- **Performance Highlights**: 광범위한 실험을 통해 다양한 얼굴 데이터셋을 대상으로 여러 모델에 대한 실험을 진행하였으며, 제안된 공격 방법이 기존의 최첨단 방어 방법을 우회하면서도 효과성(effectiveness), 강건성(robustness), 자연스러움(naturalness), 은폐성(stealthiness)을 유지함을 보여주었습니다.



### Towards Deconfounded Image-Text Matching with Causal Inferenc (https://arxiv.org/abs/2408.12292)
Comments:
          ACM MM

- **What's New**: 이 논문에서는 이미지-텍스트 매칭 (image-text matching) 모델에서 발생하는 데이터셋 편향 (bias)을 해결하기 위해 새로운 접근 방식을 제안한다. 구체적으로는, 구조적 인과 모델 (Structural Causal Models, SCM)을 사용하여 내부 및 외부 요인이 어떻게 이미지-텍스트 매칭 성능에 부정적인 영향을 미치는지를 설명하고, 새로운 Deconfounded Causal Inference Network (DCIN)를 개발하여 편향의 영향을 최소화한다.

- **Technical Details**: 제안된 DCIN은 (1) 이미지와 텍스트 특성의 인코딩 단계에서 내부 및 외부 요인을 분해하고 통합하여 가짜 상관관계를 효과적으로 제거하며, (2) 외부 지식의 편향을 완화하기 위해 인과 추론 (causal inference)을 사용한다. 이 과정에서 데이터를 통해 크고 작은 요인들 사이의 관계를 학습할 수 있도록 구조적으로 설계되었다.

- **Performance Highlights**: Flickr30K와 MSCOCO 같은 두 가지 유명한 벤치마크 데이터셋에서 수행된 광범위한 실험 결과, 제안된 DCIN 방법이 기존 방법보다 우수한 성능을 보였음을 입증하였다.



### Subsurface Scattering for 3D Gaussian Splatting (https://arxiv.org/abs/2408.12282)
Comments:
          Project page: this https URL

- **What's New**: 이번 논문에서는 3D Gaussian Splatting(3D GS) 프레임워크를 활용하여 섬유 물질의 세부적인 서브서피스 산란(subsurface scattering, SSS) 효과를 캡처하는 최초의 방법을 제안합니다. 이 방법은 여러 시점에서의 OLAT(One Light At a Time) 데이터에 기반하여 객체의 형상과 방사 조도 전이 필드를 최적화합니다.

- **Technical Details**: 제안된 방법은 장면을 3D Gaussian으로 표현된 명시적 표면과 공간적으로 변하는 BRDF, 그리고 섬유 물질의 암시적 볼륨 표현으로 분해합니다. 우리는 딥러닝 기반의 예측 네트워크를 통해 서브서피스 산란 효과를 모델링하며, 모든 파라미터는 레이 트레이싱(ray tracing) 기반의 미분 가능 렌더링으로 공동 최적화합니다. 새로운 OLAT 데이터셋을 제공하여 효과적인 학습과 평가를 가능하게 합니다.

- **Performance Highlights**: 본 연구는 이전 SSS 접근 방식에 비해 훈련 시간과 렌더링 속도를 개선하였으며, 유사하거나 더 나은 결과를 달성했습니다. 이 방법은 의료 이미지화, 엔터테인먼트 분야의 시각 효과 및 애니메이션, VR 및 AR에서 더욱 사실적이고 몰입감 있는 경험을 제공하는 등 다양한 응용 가능성을 가집니다.



### Epsilon: Exploring Comprehensive Visual-Semantic Projection for Multi-Label Zero-Shot Learning (https://arxiv.org/abs/2408.12253)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2309.00923

- **What's New**: 이 논문은 다중 레이블 제로샷 학습(Multi-Label Zero-Shot Learning, MLZSL)에 대한 새로운 접근법인 'Epsilon'을 제안합니다. 기존 방법들이 지역(local) 및 전역(global) 특성을 효과적으로 이용하지 못했던 문제를 해결하기 위해, Epsilon은 시각적-의미적 프로젝션을 강화하여 더욱 정확하고 강력한 모델을 구현합니다.

- **Technical Details**: Epsilon은 이미지의 특성을 여러 의미적 프롬프트(semantic prompts)로 그룹화하여 이미지의 시맨틱 정보를 집계합니다. 또한, Global Forward Propagation(GFP)을 사용하여 전역 정보의 풍부함을 증가시킵니다. 이를 통해 지역 특성이 전역 정보에 보완되도록 하여 의미적 완전성을 보장합니다.

- **Performance Highlights**: NUS-Wide 및 Open-Images-v4와 같은 대규모 MLZSL 벤치마크 데이터셋에서 Epsilon은 기존 최첨단 모델들에 비해 월등히 우수한 성능을 보여줍니다.



### PRG: Prompt-Based Distillation Without Annotation via Proxy Relational Graph (https://arxiv.org/abs/2408.12248)
- **What's New**: 본 논문에서는 대형 기초 모델(Large Foundation Models, LFM)로부터 경량 모델로 지식을 추출하기 위한 새로운 증류 방법을 제안합니다. 특히 수동 주석 데이터 없이도 작동하는 새로운 감독 모드를 도입하였습니다.

- **Technical Details**: 우리는 Proxy Relational Graph (PRG) 방법을 도입하여 LFM의 작업 관련 지식을 효과적으로 추출하고, LFM과 학생 모델 간의 관계를 모델링하여 선택적인 지식의 증류를 실현합니다. PRG는 텍스트 프롬프트 임베딩을 기반으로 한 가중 평균 로짓을 계산하여 작업 관련 지식을 추출합니다. 이후 LFM과 학생 모델에 대한 샘플-클래스 프록시 그래프를 구성하고 두 모델 간의 관계를 맞춤으로써 지식을 전이합니다.

- **Performance Highlights**: PRG 방법을 통한 실험 결과, CIFAR-100 데이터셋에서는 76.23%의 정확도(T: 77.9%)를 달성하고, ImageNet-1K에서 72.44%의 정확도(T: 75.3%)를 기록하며 기존 방법들에 비해 우수한 성능을 입증하였습니다.



### OVA-DETR: Open Vocabulary Aerial Object Detection Using Image-Text Alignment and Fusion (https://arxiv.org/abs/2408.12246)
- **What's New**: 최근에 제안된 OVA-DETR(Open Vocabulary Aerial object DEtection TRansformer)는 이미지와 텍스트의 관계를 활용하여 항공 이미지에서 객체 탐지의 범위를 확장합니다. 이 모델은 고성능이고 개방된 어휘를 지원하는 탐지기입니다.

- **Technical Details**: OVA-DETR는 이미지-텍스트 정렬 개념에 기반하여 전통적인 카테고리 회귀 손실(category regression loss)을 대체하는 지역-텍스트 대비 손실(region-text contrastive loss)을 도입합니다. 또한, 이 모델은 쌍방향 비전-언어 융합(Bidirectional Vision-Language Fusion, Bi-VLF) 구조를 통해 특성 추출을 향상시키며, 이 구조는 이중 주의 융합 인코더(Dual-Attention Fusion Encoder)와 다수 텍스트 유도 융합 디코더(Multi-Level Text-guided Fusion Decoder)로 구성됩니다.

- **Performance Highlights**: OVA-DETR는 DIOR 데이터셋에서 제로샷 탐지(zero shot detection) 실험을 통해 DescReg 및 YOLO-World 보다 각각 37.4% 및 33.1% 우수한 성능을 보였으며, 87 FPS의 추론 속도를 기록하여 DescReg보다 7.9배, YOLO-World보다 3배 더 빠른 속도를 달성했습니다.



### Scalable Autoregressive Image Generation with Mamba (https://arxiv.org/abs/2408.12245)
Comments:
          9 pages, 8 figures

- **What's New**: AI 모델 AiM을 소개하며, Mamba 아키텍처를 기반으로 하는 자가 회귀(autoregressive) 이미지 생성 모델입니다. 기존 Transformer 모델들을 대체하여 더 높은 생성 품질과 향상된 추론 속도를 목표로 합니다.

- **Technical Details**: AiM은 Mamba라는 새로운 상태-공간 모델(state-space model)을 채택하여 효율적인 긴 시퀀스 모델링을 수행합니다. 'next-token prediction' 패러다임을 사용하여 이미지 생성에 직접 적용되며, adaLN-Group이라는 새로운 적응형 레이어 정규화 방법이 통합되어 성능과 매개변수 수의 균형을 최적화합니다.

- **Performance Highlights**: ImageNet1K 256×256 벤치마크에서 AiM의 최상의 모델은 FID 2.21을 달성하였으며, 기존의 동급 AR 모델을 초월하고 확산(diffusion) 모델과의 비교에서도 경쟁력을 보입니다. 가장 작은 AiM 모델은 148M 파라미터로 FID 3.5를 기록하며, 두 배 이상의 파라미터 수를 필요로 하는 다른 모델들을 초과 달성하였습니다. AiM은 Transformer 기반의 AR 모델 및 확산 모델에 비해 상당히 빠른 추론 속도를 제공합니다.



### BihoT: A Large-Scale Dataset and Benchmark for Hyperspectral Camouflaged Object Tracking (https://arxiv.org/abs/2408.12232)
- **What's New**: 이 논문에서는 하이퍼스펙트럴 객체 추적(hyperspectral object tracking, HOT)의 새로운 과제인 하이퍼스펙트럴 위장 객체 추적(hyperspectral camouflaged object tracking, HCOT)을 소개하고, 이를 위한 대규모 데이터셋인 BihoT를 구축하여 제안합니다. BihoT는 41,912개의 하이퍼스펙트럴 이미지로 구성된 49개의 비디오 시퀀스를 포함하여, 위장된 객체들을 효과적으로 포착하는 데 필요한 특수한 데이터셋입니다.

- **Technical Details**: BihoT 데이터셋은 시각적으로 유사한 위장된 객체들을 포함하고 있으며, 이들 객체는 스펙트럼 정보가 다르지만 시각적 정보가 유사합니다. 여기서 제안된 스펙트럴 프롬프트 기반 장애물 인식 네트워크(spectral prompt-based distractor-aware network, SPDAN)에는 스펙트럴 임베딩 네트워크(spectral embedding network, SEN), 스펙트럴 프롬프트 기반 백본 네트워크(spectral prompt-based backbone network, SPBN), 장애물 인식 모듈(distractor-aware module, DAM)이 포함되어 있습니다.

- **Performance Highlights**: SEN은 3D 및 2D 합성을 통해 스펙트럴-공간적 특성을 추출하고, SPBN은 RGB 트래커를 스펙트럴 프롬프트로 미세 조정하여 견고한 스펙트럴 특성을 추출합니다. DAM은 오클루전(occlusion)에 의한 장애물을 포착하여 트래킹 성능을 효과적으로 개선합니다. 실험 결과, 제안된 SPDAN은 BihoT 데이터셋 및 기타 HOT 데이터셋에서 최첨단 성능을 달성했습니다.



### Computer-Aided Fall Recognition Using a Three-Stream Spatial-Temporal GCN Model with Adaptive Feature Aggregation (https://arxiv.org/abs/2408.12211)
- **What's New**: 이번 연구는 3개의 스트림을 기반으로 하는 새로운 공간-시간 특성 기반의 낙상 탐지 시스템을 제안합니다. 이 시스템은 관절 스켈레톤 기반의 그래프 신경망 특징을 활용하여 낙상을 보다 정확하게 감지할 수 있도록 설계되었습니다.

- **Technical Details**: 제안된 시스템은 첫 번째 스트림에서 관절 스켈레톤 기반의 공간 및 시간 Graph Convolutional Network (GCN) 특징을, 두 번째 스트림에서 관절 움직임 기반의 공간 및 시간 GCN 특징을, 세 번째 스트림에서 잔여 연결 기반의 특징을 사용합니다. 이들 각각의 스트림은 적응형 그래프 기반 특징 집합을 활용하여 컴퓨팅 복잡성을 줄입니다.

- **Performance Highlights**: 실험 결과, ImViA, UR-Fall, Fall-UP, FU-Kinect 데이터셋에서 각각 99.51%, 99.15%, 99.79%, 99.85%의 정확도를 달성하여 본 시스템의 탁월한 성능을 입증하였습니다.



### Transientangelo: Few-Viewpoint Surface Reconstruction Using Single-Photon Lidar (https://arxiv.org/abs/2408.12191)
- **What's New**: 이 논문에서는 라이다 시스템으로부터의 원시 측정을 사용하여 몇 개의 시점에서 3D 표면 복원을 수행하는 문제를 다룹니다. 기존의 라이다 시스템은 반사된 빛의 파형을 원시 형태로 출력하지 않으며, 대신 데이터를 3D 포인트 클라우드로 사전 처리합니다. 이 연구는 여러 시점에서 단일 광자 라이다 시스템으로 포착된 원시 측정을 활용하여 장면의 신경 표면 표현을 최적화하는 방법을 제안합니다.

- **Technical Details**: 제안하는 방법인 Transientangelo는 신호의 시정스기법과 서명 거리 함수(SDF) 기반의 장면 표현을 사용하는 3D 표면 복원 방법입니다. 이는 원시 라이다 측정을 활용하여 개선된 정규화 기법을 사용하여 기하학적 정보를 처리합니다. 10개의 광자당 복원 정확도를 제공하며, 기존의 깊이 맵이나 포인트 클라우드를 기반으로 한 방법보다 뛰어난 성능을 보입니다.

- **Performance Highlights**: 시뮬레이션 및 실측 데이터에서 3D 복원 성능이 개선되었으며, 특히 낮은 광자 수(10-300개)로도 복원할 수 있는 로버스트한 방식으로 복원했습니다. 기존의 이미지 기반 또는 깊이 기반 접근 방식에 비해 현저하게 향상된 기하학적 복원을 보여주며, 복원 속도 및 원거리 대상 이미지 획득에서 유리합니다.



### Rebalancing Multi-Label Class-Incremental Learning (https://arxiv.org/abs/2408.12161)
- **What's New**: 본 논문은 Multi-label class-incremental learning (MLCIL)의 기존 접근 방식을 개선하는 새로운 Rebalance 프레임워크인 RebLL을 제안합니다. RebLL은 비대칭 지식 증류(asymmetric knowledge distillation, AKD)와 온라인 재레이블링(online relabeling, OR) 두 가지 주요 모듈을 통합하여 긍정-부정 불균형 문제를 해결합니다.

- **Technical Details**: RebLL은 손실 수준에서 긍정 라벨 학습을 강조하며, 과신적 예측의 기여를 다운 웨이트하여 손실 균형을 맞추기 위해 AKD를 활용합니다. OR은 메모리 내의 클래스 분포를 복원하여 라벨의 균형을 맞춥니다. 이 방식은 기존의 MLCIL의 불균형을 해결하고, 손실 및 라벨 수준 모두에서 재균형을 통해 성능을 향상시키는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: PASCAL VOC 및 MS-COCO 데이터 세트에 대한 종합 실험을 통해 RebLL은 기존의 최첨단(results) 성능을 초과 달성하며, 일반 CNN 백본을 사용하는 경우에도 성능이 향상되었습니다.



### TRRG: Towards Truthful Radiology Report Generation With Cross-modal Disease Clue Enhanced Large Language Mod (https://arxiv.org/abs/2408.12141)
- **What's New**: 이 논문은 단계별 교육을 기반으로 한 TRRG(Truthful Radiology Report Generation) 프레임워크를 제안하여 비전-언어 모델을 사용한 방사선 보고서 생성을 개선합니다. 기존의 불균형한 데이터 분포와 관련된 문제를 해결하기 위해 질병 단서 주입을 도입하였습니다.

- **Technical Details**: TRRG 프레임워크는 이미지-텍스트 대비 학습을 활용한 사전 훈련을 통해 비전 인코더의 세부 질병 인식 능력을 향상시킵니다. 그런 다음, 조정 단계에서는 클루 주입 모듈을 통해 질병 인식 능력을 강화하고, 교차 모드 클루 상호 작용 모듈을 통해 비주얼 임베딩과 질병 단서 임베딩의 다각적 상호 작용을 달성합니다.

- **Performance Highlights**: IU-Xray 및 MIMIC-CXR 데이터셋에서 실험 결과, 제안된 TRRG 프레임워크는 방사선 보고서 생성 분야에서 기존의 방법들과 비교하여 최첨단 성능을 달성하였습니다. 이는 언어 생성 품질과 임상 효과성을 모두 크게 향상시켰음을 나타냅니다.



### SPARK: Multi-Vision Sensor Perception and Reasoning Benchmark for Large-scale Vision-Language Models (https://arxiv.org/abs/2408.12114)
Comments:
          Codes and data are available at this https URL

- **What's New**: 본 논문에서는 SPARK라는 새로운 다중 비전 센서 인식 및 추론 평가 벤치마크를 제안하여 기존 대형 비전-언어 모델(LVLM)이 다중 비전 센서 정보에 대한 이해도를 평가합니다. 이 벤치마크는 6,248개의 비전-언어 테스트 샘플을 자동 생성하여 다중 비전 감지 및 추론 과제에 대한 LVLM의 성능을 조사합니다.

- **Technical Details**: SPARK 벤치마크는 두 가지 주요 영역인 다중 비전 인식(multi-vision perception)과 다중 비전 추론(multi-vision reasoning)을 기반으로 설계되었습니다. 다중 비전 인식은 LVLM의 시각적 인식 요구 사항을 충족시키는 효과를 측정하며, 다중 비전 추론은 제공된 센서 지식으로부터 기본 정보를 바탕으로 응답할 수 있는 LVLM의 능력을 측정합니다.

- **Performance Highlights**: 실험 결과, 10개의 최신 LVLM 모델 중 대부분이 다중 비전 센서정보 관련 추론에서 제한된 성능을 보였으며, 이는 복잡한 센서 관련 질문에 대한 이해도 부족을 나타냅니다. 이 연구는 LVLM의 물리적 비전 센서에 대한 이해 부족을 최초로 드러내고 있으며, 이를 통해 현재 LVLM의 한계와 향후 연구 방향을 제시합니다.



### ZipGait: Bridging Skeleton and Silhouette with Diffusion Model for Advancing Gait Recognition (https://arxiv.org/abs/2408.12111)
- **What's New**: 본 연구에서는 대칭 구조인 실루엣(silhouettes)과 골격(skeletons) 정보를 결합하여 더 높은 성능을 달성하는 새로운 보행 인식 모델인 ZipGait를 제안합니다. 이는 분산 모델(difussion model)을 활용하여 단순 골격 정보를 밀집된 신체 형태로 재구성하는 최초의 시도로, 기존의 내재적(intrinsic) 정보에만 의존했던 방법에서 벗어나 교차 모달(cross-modal) 특징을 연결하는 방식으로 발전하였습니다.

- **Technical Details**: DiffGait는 네 가지 주요 적응을 통해 개발된 새로운 보행 확산 모델입니다. 첫째, 2D 골격 조인트를 통합 메쉬 그리드로 변환하기 위해 Heat-skeleton Alignment를 사용합니다. 둘째, DiffGait Forward Process를 사용하여 실루엣에서 Hybrid Gait Volume(HGV)으로의 변환을 용이하게 합니다. 셋째, Denoising 과정에서의 내재적 상응을 고려하여 다양한 수준의 실루엣을 생성하는 DiffGait Reverse Process를 구현합니다. 마지막으로, 따라서 DiffGait 아키텍처는 디코더(Decoder) 전용 구성으로 1.9M의 파라미터와 3628 FPS의 신속한 추론 속도를 자랑합니다. 이 모델은 Perceptual Gait Integration(PGI)을 도입하여 두 가지 단계의 프로세스를 통해 다양한 보행 특징을 융합합니다.

- **Performance Highlights**: ZipGait는 네 개의 공개 벤치마크에서의 실험을 통해 기존의 최신 방법들보다 더 높은 성능을 입증하였으며, 교차 도메인(cross-domain) 및 동일 도메인(intra-domain) 환경에서 모두 뛰어난 성능을 발휘하고, 소프트웨어에 쉽게 통합할 수 있는 유의미한 성능 향상을 보여주었습니다.



### RoVRM: A Robust Visual Reward Model Optimized via Auxiliary Textual Preference Data (https://arxiv.org/abs/2408.12109)
- **What's New**: 이번 연구에서는 대형 비전-언어 모델(LVLMs)의 인간 선호도 정렬을 개선하기 위한 Robust Visual Reward Model (RoVRM)을 제안합니다. RoVRM은 보조 텍스트 선호 데이터(auxiliary textual preference data)를 활용하여 시각적 선호 데이터의 부족 문제를 효과적으로 완화합니다.

- **Technical Details**: RoVRM은 세 단계의 점진적 훈련(progressive training)과 최적 수송(optimal transport) 기반의 선호 데이터 선택(preference data selection) 방식을 통해 구성됩니다. 이 방식은 시각적 선호 데이터를 훈련하기 위한 비주얼 보상 모델(VRM)을 더욱 향상시킵니다.

- **Performance Highlights**: LLaVA-1.5-7B 및 -13B 모델을 기반으로 한 실험에서 RoVRM은 전통적인 VRM을 지속적으로 초월하는 성능을 보였습니다. 또한, 점진적 훈련 및 선호 데이터 선택 방법은 순위 기반 정렬 기술(ranking-based alignment techniques)인 직접 선호 최적화(direct preference optimization)보다 일관된 성능 향상을 나타냈습니다.



### A Unified Plug-and-Play Algorithm with Projected Landweber Operator for Split Convex Feasibility Problems (https://arxiv.org/abs/2408.12100)
- **What's New**: 최근 몇 년 간 Plug-and-Play (PnP) 방법은 근사 연산자를 denoiser로 교체함으로써 역 이미지 문제에서 최첨단 성능을 달성했습니다. 본 논문에서는 split convex feasibility problems (SCFP) 관점에서 Projected Landweber Operator (PnP-PLO)를 활용한 적응형 PnP 알고리즘을 제안하여 이러한 문제들을 해결합니다.

- **Technical Details**: PnP-PLO 알고리즘은 일반적인 PnP 방법의 제한을 극복하기 위해 설계되었으며, Gaussian noise에 한정되지 않도록 다양한 노이즈 모델에 적용 가능하도록 개선되었습니다. 본 연구에서는 PnP-PLO의 이론적 보장을 통해 RED 및 RED-PRO와 같은 기존 방법을 능가하는 것으로 나타났습니다.

- **Performance Highlights**: Numerical experiments 결과, PnP-PLO는 이미지 디블러링, 슈퍼 해상도, 압축 감지 MRI 실험에서 기존 최첨단 방법들보다 우수한 성능을 보였습니다.



### Query-Efficient Video Adversarial Attack with Stylized Logo (https://arxiv.org/abs/2408.12099)
- **What's New**: 이 논문은 Stylized Logo Attack (SLA)라는 새로운 블랙박스 비디오 공격 프레임워크를 제안합니다. 이 공격 방법은 스타일 전이 기반의 공격과 패치 기반의 공격을 결합하여 비디오 분류 시스템에서 효과적인 공격을 수행할 수 있습니다.

- **Technical Details**: SLA는 세 단계로 구성됩니다. 첫 번째 단계는 로고에 대한 스타일 참조 세트를 구축하여 목표 클래스 특성을 잘 반영합니다. 두 번째 단계에서는 강화 학습 (Reinforcement Learning)을 통해 최적의 스타일 참조 및 로고의 위치 매개변수를 결정합니다. 마지막으로, perturbation optimization을 통해 조정된 변형을 최적화하여 공격 성공률을 향상합니다.

- **Performance Highlights**: SLA는 UCF-101, HMDB-51 및 Kinetics-400과 같은 주요 데이터셋에서 기존의 공격 방법보다 우수한 성능을 보이며, 다양한 방어 방법에 직면했을 때에도 좋은 변형 효과를 유지합니다.



### Unlocking Attributes' Contribution to Successful Camouflage: A Combined Textual and VisualAnalysis Strategy (https://arxiv.org/abs/2408.12086)
Comments:
          Accepted by ECCV 2024

- **What's New**: 본 연구는 Camouflaged Object Segmentation (COS) 분야에서 효과적인 위장 패턴의 기여도를 정량적으로 평가할 수 있는 최초의 포괄적인 프레임워크인 ACUMEN을 제안합니다. 이 연구에서는 COD-Text And X-attributions (COD-TAX)라는 데이터셋을 구축하여 위장 객체의 특성과 그 기여도를 체계적으로 분석합니다.

- **Technical Details**: ACUMEN(Attribution CUe Modeling with Eye-fixation Network)은 시각적 정보와 텍스트 정보를 통합하여 COS 작업을 수행하는 강력한 프레임워크입니다. 이 모델은 frozen CLIP 텍스트 인코더를 사용하여 텍스트 분석을 수행하고, 시각적 분석을 위해 기여도와 주시 예측기를 도입합니다. AFE(Attributes-Fixation Embedding) 모듈을 통해 예측된 특성 기여 텐서 및 주시 맵을 최대화하고, 최종적으로 transformer decoder를 사용하여 위장 객체의 마스크를 생성합니다.

- **Performance Highlights**: ACUMEN은 세 개의 널리 사용되는 데이터셋에서 아홉 개의 기존 방법을 초월하여 우수한 성능을 입증했습니다. 본 연구는 위장 기법의 이해를 심화시키고, 전통적인 방법에 비해 성능 향상의 가능성을 보여줍니다.



### Vision-Based Detection of Uncooperative Targets and Components on Small Satellites (https://arxiv.org/abs/2408.12084)
Comments:
          Small Satellite 2024 Conference, 13 pages, 8 figures, 6 tables

- **What's New**: 본 논문에서는 스페이스 데브리와 비활성 위성을 탐지하고 추적하기 위한 자율 탐지 모델을 제안합니다. 새로운 접근 방식을 통해 다양한 상황에서도 신뢰할 수 있는 탐지를 목표로 하고 있습니다.

- **Technical Details**: 이 방법은 두 가지 거리에서 타겟을 검출하는데, 먼 거리에서는 YOLOv8 (You Only Look Once) 모델을 사용하며, 가까운 거리에서는 Fast-SCNN을 이용한 지식 증류(knowledge distillation) 기법을 적용합니다. 이로써 낮은 저장 요구사항과 빠른 추론 시간을 달성합니다.

- **Performance Highlights**: 연구는 자율 감지를 위한 새로운 방법론을 제안하고, 맞춤형 데이터셋을 통해 우주에서의 독특한 조건을 시뮬레이션함으로써 기법의 효과를 검증하였습니다.



### Enhancing Sampling Protocol for Robust Point Cloud Classification (https://arxiv.org/abs/2408.12062)
- **What's New**: 이번 연구에서는 포인트 클라우드 학습을 위한 향상된 샘플링 프로토콜인 PointDR을 제안합니다. 기존 샘플링 프로토콜들이 현실 데이터의 노이즈에 취약하다는 문제를 해결하기 위해 다운샘플링 및 리샘플링 과정을 통해 더 강인한 방법을 제공합니다.

- **Technical Details**: PointDR 프로토콜은 두 가지 주요 구성 요소로 이루어져 있습니다: 1) 키 포인트 식별을 위한 다운샘플링 및 2) 유연한 샘플 사이즈를 위한 리샘플링. 다운샘플링 과정에서는 로컬 밀도를 고려한 격리 비율(weight)을 이용해 핵심 포인트를 무작위로 선택하고, 리샘플링에서는 로컬 기하학을 보존하면서 불완전한 데이터를 보완합니다.

- **Performance Highlights**: PointDR은 다양한 오염된 포인트 클라우드 분류 벤치마크에서 최첨단 방법들을 초월하는 성능을 보여주며, 포인트 클라우드 학습의 강인성을 크게 향상시킵니다.



### ISETHDR: A Physics-based Synthetic Radiance Dataset for High Dynamic Range Driving Scenes (https://arxiv.org/abs/2408.12048)
- **What's New**: 이 논문은 고동적 범위(HDR) 이미징 시스템을 위한 물리 기반의 종단 간 소프트웨어 시뮬레이션을 설명합니다. 이 시스템은 운전 중 낮과 밤의 터널 및 다양한 조명 조건에서의 성능을 향상시키기 위한 센서를 탐색하는 데 사용됩니다.

- **Technical Details**: 논문은 HDR 주행 장면의 세분화된 인스턴스(instance segmentation) 및 깊이(depth)를 포함하는 레이블이 있는 합성 방사선 데이터셋을 생성하고, 종단 간 시뮬레이션 프레임워크의 개발과 검증을 설명하며, HDR을 위한 두 개의 단일 촬영(single-shot) 센서를 비교 분석합니다. 또한, 다양한 조명 조건을 시뮬레이션하기 위해 4개의 스펙트럼 방사선 맵으로 구성된 2000개의 의미적으로 레이블이 지정된 장면 데이터를 공개했습니다.

- **Performance Highlights**: 제안된 시스템은 주간과 야간의 다양한 드라이빙 조건에서 높은 품질의 세부 사항을 정확하게 캡처할 수 있으며, 개발된 소프트웨어와 데이터셋은 오픈 소스로 공개되어 연구자들에게 접근성을 제공합니다.



### FUSELOC: Fusing Global and Local Descriptors to Disambiguate 2D-3D Matching in Visual Localization (https://arxiv.org/abs/2408.12037)
- **What's New**: 본 논문은 기계 학습 기법을 사용하여 로컬 (local) 및 글로벌(global) 디스크립터를 융합함으로써 2D-3D 매칭 알고리즘의 성능을 개선하는 새로운 접근 방식을 제안합니다. 이를 통해 메모리 요구 사항을 절반으로 줄이며 계층적(hierarchical) 방법에 가까운 정확도를 달성합니다.

- **Technical Details**: 연구는 로컬 및 글로벌 디스크립터를 가중 평균 연산자를 이용해 융합하여 2D-3D 검색 프레임워크 내에서 탐색 공간의 모호성을 줄이는 방법을 설명하고 있습니다. 이 접근 방식은 메모리 오버헤드가 최소화되면서도 검색 정확도를 대폭 향상시킵니다.

- **Performance Highlights**: 여러 대규모 야외 데이터셋을 통한 광범위한 실험을 통해, 우리의 접근 방식이 단순한 로컬 시스템에 비해 지속적으로 정확도를 개선하며, 메모리 사용에도 긍정적인 영향을 미친다는 것을 보여줍니다.



### CaRDiff: Video Salient Object Ranking Chain of Thought Reasoning for Saliency Prediction with Diffusion (https://arxiv.org/abs/2408.12009)
- **What's New**: 본 논문에서는 비디오 주목 예측(video saliency prediction)에 대한 새로운 접근 방식을 소개합니다. 기존 방법들이 주로 지각(perceptual) 정보를 모델링하는 데 집중하는 반면, 본 논문은 언어에 의한 추론 과정을 포함하여 주목 예측을 개선하기 위한 CaRDiff 프레임워크를 제안합니다.

- **Technical Details**: CaRDiff는 멀티모달 대형 언어 모델(multimodal large language model, MLLM), 그라운딩 모듈(grounding module), 확산 모델(diffusion model)을 통합하여 비디오 주목 예측을 향상시키기 위해 설계되었습니다. 특히, 영상 콘텐츠를 캡션(caption)하고 중요한 객체와 해당 순위를 유추하는 새로운 프롬프트(prompting) 방법인 VSOR-CoT (Video Salient Object Ranking Chain of Thought)를 도입합니다. 이 방법을 통해 순위 맵(ranking maps)을 생성하고 이를 기반으로 확산 모델이 정확한 주목 맵(saliency maps)을 디코딩하도록 지원합니다.

- **Performance Highlights**: VSOR-CoT는 MVS 데이터셋에서 최첨단 모델들보다 뛰어난 성능을 보이며, DHF1k 데이터셋에 대한 제로샷 평가(zero-shot evaluation)를 통해 다양한 데이터셋 간의 처리 능력(cross-dataset capabilities)을 입증합니다.



### Visual Localization in 3D Maps: Comparing Point Cloud, Mesh, and NeRF Representations (https://arxiv.org/abs/2408.11966)
- **What's New**: 이 논문은 시각(sight) 및 리다르(lidar) 센싱을 통해 구축된 색상 3D 맵 표현 내에서 카메라 이미지를 지역화하는 교차 모달(global visual localization) 지역화 시스템을 소개하고 평가합니다.

- **Technical Details**: 본 시스템은 포인트 클라우드(point clouds), 메쉬(meshes), 신경 복사 필드(NeRF)로 하고 있는 색상 3D 맵을 생성하기 위한 세 가지 최신 방법을 제시하며, 이러한 표현으로부터 합성 RGB 및 깊이 이미지 쌍의 데이터베이스를 구축합니다. 이 데이터베이스는 글로벌(localization) 지역화를 위한 기초로 사용됩니다.

- **Performance Highlights**: 세 가지 맵 표현 모두가 다양한 환경에서 55% 이상의 일관된 지역화 성공률을 달성하며, NeRF로 합성된 이미지는 평균 72%의 성공률을 자랑하며, 실제 실험을 통해 입증된 성능이 확인되었습니다.



### Real-Time Incremental Explanations for Object Detectors (https://arxiv.org/abs/2408.11963)
- **What's New**: 이번 논문에서는 인상적인 속도로 객체 감지기의 설명을 제공하는 새로운 알고리즘 IncX를 소개합니다. IncX는 기존의 블랙박스 설명 도구들이 여러 번 모델에 호출해야 하는 문제를 해결하여, 실시간으로 설명을 생성할 수 있도록 설계되었습니다.

- **Technical Details**: IncX는 선형 변환(linear transformations)을 기반으로 한 saliency map의 점진적인 근사(incremental approximation) 기법을 사용하여, 객체 분류 시간을 거의 변화시키지 않으면서 실시간으로 설명을 처리할 수 있습니다. 이전의 프레임에서 생성된 saliency map을 변형하고 축척하여 다음 프레임에 적응하는 방식으로 작동하며, d-rise와 결합하여 그 품질에 있어 비슷한 수준의 설명력을 보여줍니다.

- **Performance Highlights**: IncX는 d-rise보다 두 배 빠른 속도로 작동하며, 설명 생성 속도가 두 주문 단위로 향상되었습니다. 이를 통해 영상 데이터 및 대량의 시계열 이미지 데이터의 효율적인 처리에 매우 유용합니다.



### CARLA Drone: Monocular 3D Object Detection from a Different Perspectiv (https://arxiv.org/abs/2408.11958)
- **What's New**: 이 논문에서는 기존의 단일 시점(monocular) 3D 물체 감지 방법의 제한점을 해결하고, 다양한 카메라 시점에서 3D 감지 프레임워크의 성능을 평가하기 위해 CARLA Drone 데이터셋(CDrone)을 소개합니다. 이 데이터셋은 드론 시점을 시뮬레이션하여 기존 벤치마크의 카메라 시점 다양성을 크게 확장합니다.

- **Technical Details**: CDrone은 3D 바운딩 박스에 대한 포괄적인 주석을 제공하며, CARLA 시뮬레이터를 사용하여 다양한 실외 장면을 생성합니다. 이 논문은 GroundMix라는 데이터 증강 파이프라인을 개발하여 훈련 이미지의 3D 일관성을 촉진합니다. GroundMix는 지면 방정식을 활용하여 3D 인식 가능한 이미지 편집을 가능하게 하며, 복잡한 훈련 시나리오를 제공합니다.

- **Performance Highlights**: 확장된 평가에서는 CDrone 및 실제 3D 드론 데이터셋에서 이전 방법들이 잘 수행되지 않는 한편, 제안된 데이터 증강 기법이 경량화된 일단계 감지기의 탐지 정확도를 크게 향상시킴을 보여줍니다. 모든 테스트된 데이터셋에서 평균 정밀도(average precision)는 이전의 최첨단 모델과 동등하거나 이를 능가하는 결과를 보였습니다.



### Joint PET-MRI Reconstruction with Diffusion Stochastic Differential Mod (https://arxiv.org/abs/2408.11840)
Comments:
          Accepted as ISMRM 2024 Digital poster 6575. 04-09 May 2024 Singapore

- **What's New**: 본 논문에서는 PET와 MRI의 통합 이미지 재구성을 개선하기 위한 새로운 접근 방식인 확산 확률 미분 방정식(diffusion stochastic differential equations)을 기반으로 한 모델을 제안합니다.

- **Technical Details**: 저자들은 PET과 MRI의 공동 확률 분포(joint probability distribution)를 학습함으로써 두 가지 이미지를 동시에 재구성하는 방법을 개발했습니다. 이 방법은 PET-MRI 시스템에서 저조한 신호 대 잡음 비율(signal-to-noise ratio) 문제와 MRI의 고가격 획득 시간을 개선하는 데 필요한 기술적 접근을 포함하고 있습니다.

- **Performance Highlights**: 이 모델은 기존의 최첨단 방법론들을 초월하여 PET 및 MRI 재구성에서 질적(qualitative) 및 양적(quantitative) 개선을 보여주었습니다. 이로 인해 PET-MRI 시스템 내에서의 재구성 도전 과제를 효과적으로 해결할 수 있음을 입증하였습니다.



### Analysis of Unstructured High-Density Crowded Scenes for Crowd Monitoring (https://arxiv.org/abs/2408.11836)
- **What's New**: 이번 연구는 인공지능 기술을 활용하여 인파 속에서 조직적인 움직임을 자동으로 탐지하는 시스템을 개발하는 데 초점을 맞추고 있습니다. 특히, 충돌 회피에 대한 관점에서 비정상적 행동을 식별하는 데 중점을 둡니다.

- **Technical Details**: 본 시스템은 컴퓨터 비전 알고리즘을 사용하여 인파 장면의 비디오에서 정보를 추출하고, 조직적인 움직임을 보이는 집단을 자동으로 탐지 및 추적할 수 있습니다. CCTV에서 포착된 움직임의 시작 후 3~4 비디오 프레임 이내, 즉 1초도 채 안 되는 시간 내에 참가자의 수, 속도 및 방향을 실시간으로 추정할 수 있습니다.

- **Performance Highlights**: 본 연구는 생물학적 세포 데이터에서 최대 4,000개의 객체를 포함하는 예비 분석을 수행했으며, 공공 안전 응용 프로그램을 위해 이 숫자를 100배로 확장할 계획입니다. 스포츠 경기장이나 공공 장소 내외부에서 촬영된 이미지 시퀀스를 분석하여 중요한 사건을 파싱할 수 있는 데이터 기반 소프트웨어 시스템을 구축할 예정입니다.



### SCREENER: A general framework for task-specific experiment design in quantitative MRI (https://arxiv.org/abs/2408.11834)
- **What's New**: SCREENER라는 새로운 프레임워크를 제안하여 quantitative MRI (qMRI) 실험 설계를 특정 임상 작업에 맞게 최적화할 수 있도록 하였습니다. 이 프레임워크는 deep reinforcement learning (DRL) 기반의 최적화 전략을 포함하고 있으며, 기존의 임시 방편적 방법과 CRLB 최적화 방법보다 우수한 성능을 보여줍니다.

- **Technical Details**: SCREENER는 특정 임상 과제를 위한 최적의 프로토콜을 설계하는 데 두 가지 주 요소를 포함합니다: (1) 과제 특정 목표 모듈, (2) 드릴 기반 최적화 모듈. 이를 통해 데이터 수집 조건을 최적화하고, 특히 균질 조사를 통해 뼈 수염의 염증 상태 분류 작업에 활용됩니다.

- **Performance Highlights**: 실험 결과, SCREENER는 기존 방법보다 이진 분류 작업에서 67%에서 89%로, 다중 클래스 분류 작업에서 46%에서 59%로 성능을 크게 향상시켰습니다. 또한, SNR의 변화에 강건하게 작동함을 보여주었으며, DRL 기반 최적화 전략을 통해 훈련에 사용되지 않은 다양한 SNR에 대해 제로샷 발견이 가능함을 입증하였습니다.



### FAKER: Full-body Anonymization with Human Keypoint Extraction for Real-time Video Deidentification (https://arxiv.org/abs/2408.11829)
- **What's New**: 이 논문에서는 기존의 기법과는 다르게 개인의 얼굴뿐만 아니라 전신의 익명화를 목표로 하는 새로운 방법을 제안합니다. 이 방법은 작은 모델을 사용하여 실시간으로 전신을 익명화할 수 있습니다.

- **Technical Details**: 기존의 블러링(Blurring)이나 픽셀화(Pixelation) 방식 대신, GAN(Generative Adversarial Networks)과 포즈 추정 알고리즘(Pose Estimation Algorithms)을 활용하여 개인의 위치, 움직임 및 자세 정보를 정확히 표현하며, 피부색, 의상, 액세서리와 같은 개인 식별 정보를 효과적으로 제거합니다.

- **Performance Highlights**: 이 알고리즘은 CCTV나 IP 카메라 시스템에 통합할 수 있으며, 실시간으로 작동하여 다양한 산업 환경에서 전신 익명화 기술의 광범위한 채택을 촉진합니다.



### Automating Deformable Gasket Assembly (https://arxiv.org/abs/2408.12593)
Comments:
          Content without Appendix accepted for IEEE CASE 2024

- **What's New**: 이번 연구에서는 가스켓(Gasket) 조립 작업을 위한 4가지 방법을 제안하고 비교합니다. 여기에는 딥 모방 학습(Deep Imitation Learning)에서 유도한 하나의 정책과 세 가지 절차적 알고리즘이 포함됩니다.

- **Technical Details**: 가스켓 조립은 자동차, 가전제품, 전자기기 등 다양한 제품의 밀봉 표면에서 일반적으로 발생하는 작업입니다. 이 작업은 긴 시간(horizon) 동안 높은 정밀도(precision)를 요구하며, 가스켓이 채널(channel)과 정렬(alignment)되어 완전히 압축(compressed)되어야 안전한 조립이 가능합니다. 연구에서는 100회의 물리적 실험(trials)을 통해 방법들을 평가하였습니다.

- **Performance Highlights**: 실험 결과, Binary+ 알고리즘은 직선형 채널에서 10/10의 성공률을 기록하였고, 인간의 텔레조작(demonstrations)을 기반으로 한 학습된 정책은 8/10의 성공률을 보였지만 속도가 현저히 느렸습니다.



### MuMA-ToM: Multi-modal Multi-Agent Theory of Mind (https://arxiv.org/abs/2408.12574)
Comments:
          Project website: this https URL Code: this https URL

- **What's New**: 이번 논문에서는 MuMA-ToM이라는 새로운 Multi-modal Multi-Agent Theory of Mind 벤치마크를 소개하고 있습니다. 이는 AI가 복잡한 사회적 상호작용에서 사람들의 정신적 상태를 이해할 수 있도록 돕기 위해 고안되었습니다.

- **Technical Details**: MuMA-ToM는 현실적인 가정 환경에서 사람들의 다중 모달 행동에 대한 비디오 및 텍스트 설명을 제공하며, 이러한 맥락을 바탕으로 목표, 신념 및 다른 사람의 목표에 대한 신념에 관한 질문을 합니다. 또한, LIMP(언어 모델 기반의 역 다중 에이전트 계획)라는 새로운 다중 모달, 다중 에이전트 ToM 모델을 제안하였습니다.

- **Performance Highlights**: LIMP는 다양한 최신 방법들, 특히 대형 다중 모달 모델(GPT-4o, Gemini-1.5 Pro 등) 및 최근 다중 모달 ToM 모델(BIP-ALM)을 능가하는 성능을 보였습니다.



### Pruning By Explaining Revisited: Optimizing Attribution Methods to Prune CNNs and Transformers (https://arxiv.org/abs/2408.12568)
Comments:
          Accepted as a workshop paper at ECCV 2024 31 pages (14 pages manuscript, 4 pages references, 13 pages appendix)

- **What's New**: 이번 연구에서는 Deep Neural Networks (DNNs)의 비효율성을 해소하기 위해, 네트워크의 중요하지 않은 구성 요소를 제거하는 pruning 기법을 개선하고자 합니다. 특히, attribution methods (속성 메소드)에서 얻은 hyperparameters (하이퍼파라미터)를 최적화하여 pruning을 수행하는 새로운 프레임워크를 제안합니다.

- **Technical Details**: 제안한 방법에서는 Layer-wise Relevance Propagation (LRP) 기법을 사용하여 네트워크의 구성 요소의 중요도를 평가합니다. 이를 통해 특정 구성 요소가 제거되었을 때의 성능 하락을 최소화할 수 있도록 합니다. 또한, transformer 기반 네트워크(ViT)와 convolutional architectures (컨볼루션 아키텍처)에서 실험하였으며, pruning 성능을 더욱 향상시키기 위한 새로운 framework를 설계했습니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 VGG, ResNet, ViT와 같은 대형 아키텍처에서 기존 연구보다 높은 model compression rates (모델 압축률)를 달성하면서도 ImageNet 분류 작업에서 높은 성능을 유지했습니다. 또한, transformers는 convolutional neural networks (CNNs)보다 더 높은 과대파라미터화(over-parameterization)를 보임을 확인했습니다.



### Automatic Organ and Pan-cancer Segmentation in Abdomen CT: the FLARE 2023 Challeng (https://arxiv.org/abs/2408.12534)
Comments:
          MICCAI 2024 FLARE Challenge Summary

- **What's New**: 본 논문은 복부 Computed Tomography (CT) 스캔에서 장기 및 암 세분화에 대한 첫 번째 국제 대회를 개최하고, 4650개의 다양한 암 유형의 CT 스캔을 포함하는 대규모 데이터셋을 제공합니다.

- **Technical Details**: 이 연구에서는 13개의 복부 장기와 하나의 일반 병변 클래스를 포함한 세분화 태스크를 제시하고, 부분적으로 레이블이 지정된 학습(task)으로 설계되었습니다. 데이터셋은 50개 의료 센터에서 수집된 CT 스캔으로 구성되며, 새로운 방법론을 통해 평균 Dice Similarity Coefficient (DSC) 스코어가 장기에서 92.3% 및 병변에서 64.9%를 달성했습니다.

- **Performance Highlights**: 우승 팀은 deep learning 기반의 cascaded framework를 사용하여, 평균 DSC 점수에서 92.3% 및 64.9% 점수를 기록하며 기존의 state-of-the-art를 초월했습니다. 이 방식은 소비자 데스크탑에서도 4GB 미만의 GPU 메모리로 실행 가능하며, 평균 런타임은 8.58초로 나타났습니다.



### UMAD: University of Macau Anomaly Detection Benchmark Datas (https://arxiv.org/abs/2408.12527)
Comments:
          Accepted by the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2024, project code at this https URL

- **What's New**: 이 논문에서는 로봇 순찰 시나리오를 위한 새로운 이상 탐지(AAnomaly Detection with reference, ADr) 벤치마크 데이터셋인 UMAD를 소개합니다. UMAD는 참조 이미지와 쿼리 이미지 간의 의미적 변화를 비교하여 이상을 식별하는 데 특화되어 있습니다.

- **Technical Details**: 제안된 UMAD 데이터셋은 자율 로봇이 지정된 경로를 따라 순찰하면서 수집한 영상으로 구성되어 있습니다. 이 데이터셋은 6개의 다양한 장면을 포함하며, 각 장면에는 9개 이상의 원시 시퀀스가 포함되어 있습니다. 각 쿼리 이미지는 미리 구축된 3D 지도에서 로봇의 정확한 위치 정보에 따라 해당하는 참조 이미지를 찾을 수 있도록 설계되었습니다. 이로 인해 참조 이미지와 쿼리 이미지를 기하학적으로 일치시킬 수 있습니다.

- **Performance Highlights**: UMAD 데이터셋을 사용한 실험에서 다양한 이상 탐지 모델의 성능을 평가하고, 이상 및 비이상 변화를 구분하는 이중 및 다중 클래스 방식의 ADr 모델에 대한 적합성을 탐구하여 향후 ADr 알고리즘 개발에 기여하고자 합니다.



### Robotic Eye-in-hand Visual Servo Axially Aligning Nasopharyngeal Swabs with the Nasal Cavity (https://arxiv.org/abs/2408.12437)
Comments:
          12 pages, 13 figures

- **What's New**: 이번 연구는 로봇팔을 이용해 비강 내 면봉 검사를 자동화하는 방법을 제안하며, 환자의 자연스러운 자세에서 비강 근처에 면봉을 올바르게 위치시키고 정렬하는 비전 가이드 파이프라인을 구현했습니다.

- **Technical Details**: 이 파이프라인은 미리 계산된 조인트 룩업 테이블을 사용하여 로봇팔이 환자의 임의 위치에 도달할 수 있도록 하며, RGB-D 카메라를 이용해 얼굴의 유클리드 포즈를 추정합니다. 이후 추정된 데이터는 비선형 칼만 필터와 포즈 기반 비주얼 서보 제어 루프에 입력되어 면봉을 콧구멍 앞에 있도록 이동시킵니다.

- **Performance Highlights**: 이 시스템은 25명의 참가자를 대상으로 한 인간 실험에서 84%의 참가자에게 면봉이 콧구멍에 도달하였으며, 통계 분석 결과 인구 통계적 편향은 발견되지 않았습니다.



### Robust Principal Component Analysis via Discriminant Sample Weight Learning (https://arxiv.org/abs/2408.12366)
- **What's New**: 본 연구는 이상치(outlier) 영향을 줄이기 위해 샘플 가중치(sample weights)를 학습하는 새로운 강건한 주성분 분석(Robust Principal Component Analysis, RPCA) 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 데이터의 평균과 PCA 변환 행렬을 추정하기위해 각 샘플에 가중치를 반복적으로 학습합니다. 특정한 샘플에 대해 높은 가중치를 부여하고 이상치에는 낮은 가중치를 부여하여 Robust한 PCA를 구축하는 것입니다. 최적화 문제는 가중치가 포함된 데이터 평균과 변환 행렬을 추정하는 것입니다.

- **Performance Highlights**: 장난감 데이터(toy data), UCI 데이터셋, 얼굴 데이터셋을 사용한 실험에서 제안된 방법은 이상치가 포함된 데이터에서 평균과 투영 행렬을 추정하는 데 있어 효과적임을 입증하였습니다.



### EUIS-Net: A Convolutional Neural Network for Efficient Ultrasound Image Segmentation (https://arxiv.org/abs/2408.12323)
- **What's New**: 이번 연구에서는 의학적 응용을 위한 초음파 이미지 세분화에 최적화된 CNN 네트워크인 EUIS-Net을 제안했습니다. 이 네트워크는 4개의 인코더-디코더 블록을 활용하여 계산 복잡도를 줄이면서도 뛰어난 성능을 달성합니다.

- **Technical Details**: EUIS-Net은 채널 및 공간 주의 메커니즘을 통합하여 특성 표현을 개선하고 중요한 맥락 정보를 수집합니다. 또한 스킵 연결에서 영역 인식 주의 모듈(Region-aware Attention Module, RAAM)을 통합하여 손상된 영역에 집중할 수 있게 설계되었습니다.

- **Performance Highlights**: EUIS-Net은 두 개의 공개된 초음파 이미지 세분화 데이터셋(BUSI와 DDTI)에서 각각 78.12%, 85.42% 및 84.73%, 89.01%의 평균 IoU(Intersection over Union)와 Dice 점수를 달성했습니다. 이 연구의 결과는 EUIS-Net이 임상 환경에서 즉시 사용 가능하며 다양한 초음파 이미징 작업에서의 다재다능함을 보여줍니다.



### MaVEn: An Effective Multi-granularity Hybrid Visual Encoding Framework for Multimodal Large Language Mod (https://arxiv.org/abs/2408.12321)
- **What's New**: MaVEn은 여러 이미지를 통한 추론 능력을 개선하기 위해 고안된 Multi-granularity Visual Encoding 프레임워크입니다. 기존 Multimodal Large Language Models (MLLMs)는 단일 이미지에 치중해 다중 이미지 정보 통합 능력이 제한되어 있었으나, MaVEn은 이를 해결합니다.

- **Technical Details**: MaVEn은 이산(Discrete) 시각 상징 시퀀스와 기존의 연속(Continuous) 표현 시퀀스를 결합하여 다중 이미지 이해를 용이하게 하는 하이브리드 시각 인코딩 구조로 설계되었습니다. 이 모델은 단일 이미지 상황에 적합하게 설계된 기존 MLLMs의 시각적 인코딩 및 연결 방식을 재구성하여 다중 이미지 입력 처리에 최적화합니다.

- **Performance Highlights**: 실험 결과, MaVEn은 복잡한 다중 이미지 상황에서 MLLMs의 이해 능력을 크게 향상시켰으며, 단일 이미지 상황에서도 성능 개선을 보여줍니다. 또한, 긴 연속 특성에 대한 동적 축소 메커니즘을 채택하여 다중 이미지 처리 효율성을 높였습니다.



### AT-SNN: Adaptive Tokens for Vision Transformer on Spiking Neural Network (https://arxiv.org/abs/2408.12293)
Comments:
          8 pages

- **What's New**: 이 논문에서는 SNN 기반 ViT에서 직선 학습(direct training)과 경량화 계산(lightweight computation) 방법을 결합한 AT-SNN을 제안하여, 토큰(token) 수를 동적으로 조절하여 전력 소비를 줄이며 정확도를 향상시키는 방안을 소개합니다. 추가적으로, ACT(adaptive computation time)를 SNN에 적용하여 정보가 적은 공간 토큰을 선택적으로 제거할 수 있도록 합니다.

- **Technical Details**: AT-SNN은 SNN 기반 ViT에서 두 가지 차원에서 조정하는 이탈 방법을 적용하는 최초의 시도로, ACT와 함께 가변적인 토큰 비율을 사용하여 계산 효율성을 증가시키고, 새로운 토큰 병합(token-merge) 메커니즘을 통해 유사한 토큰들을 병합하여 토큰 수를 further 줄입니다. 이 방법들은 CIFAR-10, CIFAR-100, TinyImageNet을 포함한 이미지 분류(task)에서 검증되었습니다.

- **Performance Highlights**: AT-SNN은 CIFAR-100에서 기존 최고의 방법 대비 42.4% 더 적은 토큰을 사용하면서도 높은 정확도를 유지하는 성능을 보여주었습니다. 특히, Spikformer에 구현된 AT-SNN은 TET 및 DT-SNN보다 73.23%와 75.81%의 정확도를 달성하였습니다.



### Whole Slide Image Classification of Salivary Gland Tumours (https://arxiv.org/abs/2408.12275)
Comments:
          5 pages, 2 figures, 28th UK Conference on Medical Image Understanding and Analysis - clinical abstract

- **What's New**: 본 연구는 침샘 종양(Salivary Gland Tumours)에서 전체 슬라이드 이미지(Whole Slide Images, WSI)를 이용한 다중 인스턴스 학습(Multiple Instance Learning, MIL)을 통해 암을 분류하는 방법을 제안하고 있습니다. CTransPath를 사용한 특징 추출기와 CLAM을 통한 특징 집계 방법을 활용하여, 0.88의 F1 점수와 0.92의 AUROC를 기록하였습니다.

- **Technical Details**: 침샘 종양은 이종 신생물의 비교적 드문 집단으로, 본 연구에서는 646개의 WSI를 통해 양성/악성 분류와 악성 종양의 특정 유형인 선종성 낭종암(adenoid cystic carcinoma) 분류를 수행했습니다. ResNet-50과 CTransPath 두 가지 특징 추출기를 사용하여 정확도를 비교한 결과, CTransPath가 더 우수한 성능을 보였습니다. MIL 접근법을 통해 WSI를 작은 패치로 나누고, 그 후 CLAM 모델을 통해 특징을 집계했습니다.

- **Performance Highlights**: CTransPath 기반의 모델은 F1 점수 0.88, 정밀도(Precision) 0.90, 재현율(Recall) 0.88, 특이도(Specificity) 0.92를 기록하였고, 선종성 낭종암 분류에서 AUROC 0.96, F1 점수 0.84라는 높은 정확성을 보여 주었습니다. 이 결과는 CTransPath가 ResNet-50보다 WSI에서 암 분류에 유리한 특징을 제공함을 시사합니다.



### Diffusion-Based Visual Art Creation: A Survey and New Perspectives (https://arxiv.org/abs/2408.12128)
Comments:
          35 pages, 9 figures

- **What's New**: 이번 설문조사는 생성 AI(Generative AI)가 시각 예술에서의 확산 기반(difussion-based) 창작에 미치는 영향을 탐구합니다. 예술적 및 기술적 관점에서의 발전을 세 가지 단계로 나누어 데이터를 주요 특성 및 프레임워크에 대한 식별, 구조화된 코딩 프로세스를 통한 세부 분석, 향후 전망을 제시합니다. 우리의 연구 결과는 예술적 요구가 어떻게 기술적 도전으로 변형되는지를 보여줍니다.

- **Technical Details**: 설문조사는 확산 모델의 개발과 시각 예술 생성의 상관관계를 다루며, 데이터 수집을 통해 사용자의 요구와 기술 문제 간의 상호작용을 분석합니다. 총 네 가지 주요 연구 질문을 바탕으로, 현재의 화제, 도전 과제, 사용된 방법론, 그리고 앞으로의 방향성을 탐색합니다. 확산 기반 방법들은 제어 가능성, 특정 장르 우선의 예술 창작을 포함한 다양한 세부 분야로 확장되고 있습니다.

- **Performance Highlights**: 이번 연구는 시각 예술 창작에서 확산 모델의 잠재적 혁신을 강조하며, AI 시스템이 인간의 예술적 인식 및 창의성을 어떻게 모사하고 향상시킬 수 있는지를 보여줍니다. 또한, 혁신적 통합이 예술 창작의 새로운 가능성을 열어줄 것으로 기대하며, 앞으로의 연구 방향에 대한 통찰을 제공합니다.



### Integrating Audio, Visual, and Semantic Information for Enhanced Multimodal Speaker Diarization (https://arxiv.org/abs/2408.12102)
- **What's New**: 이 논문에서는 음성 다이어리제이션(Speaker Diarization) 과제를 해결하기 위해 오디오, 비주얼(visual), 의미적(semantic) 단서를 결합한 새로운 다중 모달(multimodal) 접근 방식을 제안합니다. 기존 방법들이 보통 단일 음향 정보를 의존하는데 비해, 이 연구는 세 가지 모달의 상호작용을 활용하여 구성의 복잡성을 해결하려고 합니다.

- **Technical Details**: 제안된 방법은 오디오, 비주얼, 의미 정보를 조화롭게 사용하여 스피커를 클러스터링하는 방법을 기반으로 한 제약 최적화 문제로 모델링됩니다. 우리는 활성 스피치를 감지하고, 얼굴 인식 및 입술 움직임 탐지를 통해 비주얼 연결을 얻은 다음, 텍스트 기반 대화 감지 및 스피커 전환 감지 모델을 사용하여 의미적 구조를 이해합니다. 마지막으로, 쌍별 제약 전파(pairwise constraint propagation) 알고리즘을 도입하여 스피커 임베딩 간의 유사성을 정제합니다.

- **Performance Highlights**: 다양한 다중 모달 데이터셋에서 수행된 실험에 따르면 제안된 방법이 기존 최첨단 스피커 다이어리제이션 방법들을 일정하게 능가하였으며, 성능 향상 및 일반화 능력에 대한 강력한 증거를 제공합니다. 이 연구는 음성, 비주얼 및 의미 정보의 통합된 활용을 통해 스피커 다이어리제이션 분야에서 중요한 기여를 하였습니다.



### LLM-enhanced Scene Graph Learning for Household Rearrangemen (https://arxiv.org/abs/2408.12093)
Comments:
          SIGGRAPH ASIA 2024

- **What's New**: 이번 연구에서는 가정 내 물건 배치 작업을 위한 새로운 접근 방식을 제안합니다. 이는 장면(scene) 내에서 사용자의 선호도(user preference)와 객체 기능성(object functionality)을 직접 추출하여 인공지능을 사용해 자동으로 물건을 정리하는 시스템을 구현합니다.

- **Technical Details**: 제안된 방법은 장면 그래프(scene graph) 표현을 사용하며, LLM(대형 언어 모델)을 활용하여 정보가 강화된 노드(nodes)와 새로운 엣지(edges)를 포함하는 향상된 그래프(affordance-enhanced graph, AEG)로 변환합니다. AEG 내에서 각 수용체 객체의 맥락에 기반한 affordance가 강화되어 캐리 가능한 물체 기능성이 반영됩니다. 이 그래프를 활용하여 물체의 잘못 배치(misplacement) 탐지와 적절한 배치를 결정합니다.

- **Performance Highlights**: 본 연구에서 제안된 방법은 Habitat 3.0 시뮬레이터를 사용하여 로봇 시스템을 구현하였으며, 새로운 벤치마크를 통해 평가한 결과, 물건 잘못 배치 탐지 및 재배치 계획에서 최고 성능을 달성했습니다. 대규모 평가에서 제안된 방법이 최신 기술(state-of-the-art) 성능을 보인다는 것을 입증했습니다.



### Through-the-Wall Radar Human Activity Micro-Doppler Signature Representation Method Based on Joint Boulic-Sinusoidal Pendulum Mod (https://arxiv.org/abs/2408.12077)
Comments:
          17 pages, 14 figures, 7 tables, in IEEE Transactions on Microwave Theory and Techniques, 2024

- **What's New**: 이 논문은 마이크로 도플러(micro-Doppler) 시그니처를 사용하여, 초광대역(UWB) 벽 투과 레이더(TWR)를 통해 실내 인간 활동을 정확하게 식별하는 새로운 방법을 제안합니다. 기존 방법의 한계를 극복하고, 조인트 부울리-사인 모션 모델을 기반으로 한 표현 방식이 도입되었습니다.

- **Technical Details**: 제안된 모델은 머리, 몸통, 두 손 및 발을 포함한 단순화된 조인트 부울리-사인 펜듈럼(motion model)으로, 최소한의 키 포인트 수를 계산하여 도플러 및 마이크로 도플러 정보를 효과적으로 설명합니다. 이 논문은 TWR을 통해 구한 실내 인간 활동을 위해 30개의 키 포인트로 마이크로 도플러 코너 특징(Micro-Doppler Corner Feature)을 정의하며, 이들 키 포인트는 각 신체 부위의 동적 변화를 고유하게 나타냅니다.

- **Performance Highlights**: 실험 결과, 제안된 마이크로 도플러 시그니처의 키 포인트 수가 다양한 테스트자에 대한 일반화 능력을 크게 향상시킴을 입증했습니다. 이것은 기존 방법의 성능을 개선할 뿐만 아니라, 실내 인간 활동 인식의 정확성을 높이는 데 크게 기여합니다.



### Limitations in Employing Natural Language Supervision for Sensor-Based Human Activity Recognition -- And Ways to Overcome Them (https://arxiv.org/abs/2408.12023)
- **What's New**: 이 논문은 웨어러블 센서를 기반으로 한 인간 활동 인식(Human Activity Recognition, HAR)에 대한 자연어 감독(Natural Language Supervision, NLS)의 효과를 조사합니다. 기존의 NLS가 다양한 작업과 분야에서 뛰어난 성과를 보였음에도 불구하고, HAR에서 기대 이하의 결과를 나타냈다는 점에 주목하고 있습니다.

- **Technical Details**: 주요 원인은 센서 이질성과 활동에 대한 풍부하고 다양한 텍스트 설명 부족입니다. 이를 해결하기 위해 간단한 적응을 통해 HAR 성능을 크게 향상시키는 여러 전략을 개발했습니다. 특히, 사전 훈련된 네트워크의 일부 레이어를 적은 양의 타겟 데이터로 업데이트 하는 방법이 효과적임을 보여주었습니다.

- **Performance Highlights**: 이 연구의 전략은 HAR 성능을 유의미하게 개선하여 감독 및 자가 감독 훈련에 가까운 성과를 따르게 했습니다. 이를 통해 이전에 보지 못한 활동을 인식하고, 비디오의 크로스 모달 검색도 가능하게 했습니다. 전반적으로 이러한 연구는 웨어러블 장치를 사용하는 HAR을 위한 기본 모델 개발로 이어질 수 있는 가능성을 제시합니다.



### Detection of Under-represented Samples Using Dynamic Batch Training for Brain Tumor Segmentation from MR Images (https://arxiv.org/abs/2408.12013)
- **What's New**: 본 논문은 MRI에서 뇌 종양 세분화의 자동화를 위한 새로운 방법인 동적 배치 훈련(dynamic batch training) 기법을 제안합니다. 이 방법은 대표성이 부족한 샘플에 대한 훈련의 비효율성을 극복하고, 어려운 샘플을 식별하여 더 많은 반복을 통해 훈련하는 것을 목표로 합니다.

- **Technical Details**: 기존의 mini-batch gradient descent 방식으로 훈련되는 깊은 학습(deep learning) 모델은 대표성이 떨어지는 샘플이나 복잡한 잠재 표현(latent representation)을 가진 샘플에 대해 일반화 능력이 떨어지는 문제가 있습니다. 제안된 동적 배치 훈련 방법은 BraTS2020 데이터셋에서 어려운 샘플을 식별하고, 그러한 샘플을 더 많이 훈련하여 모델 성능을 향상시키도록 설계되었습니다.

- **Performance Highlights**: 제안된 방법의 성능은 U-Net 및 다른 기존 모델들과 비교될 때, dice 및 Hausdorff95 메트릭에서 향상된 결과를 보였습니다. 이 논문은 동적 배치 훈련 방법이 뇌 종양 세분화에 있어 효율적이고도 효과적인 접근법임을 강조합니다.



### MBSS-T1: Model-Based Self-Supervised Motion Correction for Robust Cardiac T1 Mapping (https://arxiv.org/abs/2408.11992)
- **What's New**: MBSS-T1은 심장 T1 매핑에서 모션 보정을 위한 자기 지도 학습 모델로, 물리적 및 해부학적 제약을 반영합니다. 이 모델은 자유 호흡 조건에서도 정확한 T1 매핑을 가능하게 합니다.

- **Technical Details**: MBSS-T1은 긴장완화축을 따라 정확한 T1 매핑을 보장하는 물리적 제약(physical constraints)과 해부학적 제약(anatomical constraints)을 결합했습니다. 이로 인해 높은 품질의 모델 적합도(R2: 0.974) 및 해부학적 정렬(Dice 점수: 0.921)을 달성하였습니다.

- **Performance Highlights**: MBSS-T1은 210명의 환자에 대한 공개 데이터셋(STONE 시퀀스)과 19명의 환자에 대한 내부 데이터셋(MOLLI 시퀀스)에서 기존의 딥 러닝 기반 이미지 등록 방법을 초월하는 성능을 보였습니다. 노출된 모션 아티팩트에 대한 전문가의 시각적 품질 평가에서 4.33의 점수를 받았습니다.



### AIM 2024 Challenge on Compressed Video Quality Assessment: Methods and Results (https://arxiv.org/abs/2408.11982)
- **What's New**: 이번 연구는 ECCV 2024에서 개최된 Compressed Video Quality Assessment (VQA) 챌린지의 결과를 다루고 있습니다. 이 챌린지는 다양한 비디오 압축 표준을 통한 VQA 방법의 성능을 평가하는 데 초점을 맞추었으며, 459개의 비디오와 14개의 코덱을 사용했습니다.

- **Technical Details**: 본 연구에서는 459개의 비디오로 구성된 데이터셋을 통해 VQA 방법론의 성능을 평가했습니다. 각 비디오는 AVC/H.264, HEVC/H.265, AV1, VVC/H.266와 같은 다양한 압축 표준으로 인코딩되었으며, 각기 다른 압축 아티팩트를 거쳤습니다. 평가 매트릭으로는 Spearman rank-order correlation coefficient (SROCC), Kendall rank-order correlation coefficient (KROCC), Pearson Linear Correlation coefficient (PLCC)를 사용하였습니다.

- **Performance Highlights**: 총 30개의 팀이 챌린지에 참가했으며, 6개 팀이 최종 솔루션과 코드 제출을 완료했습니다. 이 연구는 향후 VQA 연구를 위한 포괄적인 벤치마크를 제공합니다. 수집된 데이터셋, 결과, 온라인 리더보드는 공개되고 있습니다.



### CT-AGRG: Automated Abnormality-Guided Report Generation from 3D Chest CT Volumes (https://arxiv.org/abs/2408.11965)
Comments:
          15 pages, 9 figures, submitted to ISBI 2025

- **What's New**: 이번 논문은 3D CT 스캔에서 관찰된 이상 징후를 기반으로 하는 자동 보고서 생성 모델을 제안합니다. 기존의 보고서 생성 방법들은 단순히 전체 보고서를 생성하는데 집중했으나, 이 모델은 먼저 이상 징후를 예측한 후 각각에 대해 특정한 설명을 생성합니다.

- **Technical Details**: 제안하는 모델은 Encoder-Decoder 아키텍처를 사용하는 CT2Rep 모델의 주안점에서 벗어나, CT-ViT와 함께 작동하여 시각적 특징을 추출하고 18개의 이상 징후로 매핑합니다. 시각적 특징에 따라 자동으로 멀티 라벨 분류를 수행하며, 불완전한 보고서를 방지하기 위해 각 이상에 대한 구체적인 문장을 생성합니다.

- **Performance Highlights**: 공개 데이터셋에서의 평가 결과, 제안된 모델은 보고서 품질과 임상적 관련성에서 상당한 개선을 이뤘으며, 추가적인 ablation study(끝 점검)를 통해 각 모듈의 효과성을 입증했습니다.



### Video-Foley: Two-Stage Video-To-Sound Generation via Temporal Event Condition For Foley Sound (https://arxiv.org/abs/2408.11915)
- **What's New**: 이 논문에서는 비디오에서 오디오로의 생성 과정에서 동기화 및 제어를 개선하기 위한 새로운 시스템인 Video-Foley를 제안합니다. 특히 Root Mean Square (RMS) 를 사용하여 시간적 이벤트 조건을 설정하고, 의미적 음색 프롬프트를 결합함으로써 혁신적인 접근 방식을 채택했습니다.

- **Technical Details**: Video-Foley는 비디오를 오디오로 변환하는 시스템으로, RMS를 사용한 자기 지도 학습(self-supervised learning) 프레임워크를 기반으로 합니다. 이 시스템은 Video2RMS와 RMS2Sound의 두 단계로 구성되며, RMS의 이산화(discretization)와 RMS-ControlNet을 도입하여 성능을 높였습니다. 이 과정에서는 메모리 사용량을 최소화하는 방법도 포함되었습니다.

- **Performance Highlights**: Video-Foley는 오디오-비주얼 정렬(auditory-visual alignment)과 오디오의 타이밍 및 강도, 음색, 뉘앙스에 대한 제어 가능성에서 최신 기술 수준(state-of-the-art) 성능을 달성했습니다. 이 시스템은 코드와 모델 가중치, 데모가 제공되는 웹사이트를 통해 접근할 수 있습니다.



### Bioimpedance a Diagnostic Tool for Tobacco Induced Oral Lesions: a Mixed Model cross-sectional study (https://arxiv.org/abs/2408.11886)
- **What's New**: 이번 연구는 담배로 유발된 구강 병변의 진단 도구로서 생체 임피던스(bioimpedance)의 유효성을 평가하고 검증하고자 하였습니다.

- **Technical Details**: 연구는 50개의 OSCC(구강 편평세포암) 및 OPMD(구강 전암병변) 조직 샘플을 대상으로 한 인비트르(in-vitro) 연구와 320명의 피험자를 활용한 인비보(in-vivo) 연구로 구성되었습니다. 준비된 생체 임피던스 장치의 교정(calibration) 후, EIS(electrical impedance spectroscopy) 측정을 통해 발병군과 대조군을 비교하였습니다.

- **Performance Highlights**: 대조군의 임피던스 값은 OPMD 및 OSCC 그룹에 비해 유의미하게 높았으며, BIS 측정에 기반한 진단의 민감도(sensitivity)는 95.9%, 특이도(specificity)는 86.7%로 나타났습니다. 이 장치는 특히 일차 의료(primary healthcare) 환경에서 OPMD와 OSCC 사례를 구분하고 관리하는 데 도움을 줄 수 있습니다.



### DeRainGS: Gaussian Splatting for Enhanced Scene Reconstruction in Rainy Environments (https://arxiv.org/abs/2408.11540)
- **What's New**: 이 연구에서는 3D 비가 오는 환경에서의 재구성(3DRRE)이라는 새로운 작업을 소개하고, 이를 위해 HydroViews라는 새로운 데이터셋을 구축했습니다. 이 데이터셋은 다양한 비 강도와 특징이 있는 합성 및 실제 장면 이미지로 구성되어 있습니다.

- **Technical Details**: DeRainGS라는 첫 번째 3DGS 재구성 방법을 제안하며, 이는 비가 오는 환경에서의 3D 장면 재구성을 위해 특별히 설계되었습니다. 이 방법은 비 이미지 향상과 가림(masking) 모듈을 통합하여 비가 시각적 특징과 기하학적 일관성에 미치는 영향을 명확히 해결합니다.

- **Performance Highlights**: 실험 결과, DeRainGS는 여러 종류의 비 상황에서 기존의 가림 없는 방법들을 능가하는 최신 성능을 나타내며, 비가 오는 환경에서의 높은 품질 재구성을 제공합니다.



### FQGA-single: Towards Fewer Training Epochs and Fewer Model Parameters for Image-to-Image Translation Tasks (https://arxiv.org/abs/2408.09218)
- **What's New**: 이 논문에서는 CycleGAN 모델을 SynthRAD Grand Challenge Dataset에서 단일 에폭 수정 방법(Single-Epoch Modification, SEM)을 사용하여 훈련하는 새로운 접근법(CycleGAN-single)을 제안합니다. 일반적인 방식인 다수의 에폭(200 epochs, CycleGAN-multi) 훈련 방식과 비교하여 성능을 평가했습니다.

- **Technical Details**: 모델 성능은 PSNR(peak signal-to-noise ratio), SSIM(structural similarity index), MAE(mean absolute error), MSE(mean squared error)와 같은 정량적 성능 지표를 사용하여 평가되었습니다. 이 연구는 의료 이미징과 같은 특정 이미지 간 변환 작업에서 정량적 및 정성적(performance metrics) 성능을 모두 고려하는 것이 중요함을 강조합니다.

- **Performance Highlights**: FQGA(Fast Paired Image-to-Image Translation Quarter-Generator Adversary)라는 경량 모델은 CycleGAN의 Generator 모델에 비해 매개변수 수가 1/4에 불과하지만, 20 에폭 만으로도 정성적 및 정량적으로 CycleGAN을 초월하는 성능을 보였습니다. SEM 방법을 사용하여 FQGA를 훈련할 경우, 적은 매개변수와 적은 에폭으로도 성능 향상을 보일 수 있으며, 이는 의료 이미지 변환 외의 다른 이미지 간 변환 작업에도 적용 가능성이 있습니다.



New uploads on arXiv(cs.AI)

### Differentiable Logic Programming for Distant Supervision (https://arxiv.org/abs/2408.12591)
Comments:
          To be published in ECAI 2024

- **What's New**: 이 논문은 Neural-Symbolic AI (NeSy)에서 논리 프로그래밍과 신경망을 통합하는 새로운 방법을 소개합니다. 이 방법은 직접적인 라벨이 없는 상태에서 먼 감독(distant supervision) 학습을 목표로 합니다.

- **Technical Details**: 제안된 방법은 기존의 기호 해결기(symbolic solvers)에 의존하지 않고, 신경망 출력(neural network outputs)과 논리 프로그램을 행렬(matices)로 임베딩하여 논리적 암시(logical implications)와 제약(constraints)을 미분 가능한 방식으로 평가합니다.

- **Performance Highlights**: 이 방법은 다양한 작업에서 다른 방법들과 비교했을 때 정확도를 일치시키거나 초과하며, 학습 과정을 가속화합니다. 이 결과는 NeSy 애플리케이션에서 정확성과 학습 효율성을 향상시킬 수 있는 잠재력을 강조합니다.



### MuMA-ToM: Multi-modal Multi-Agent Theory of Mind (https://arxiv.org/abs/2408.12574)
Comments:
          Project website: this https URL Code: this https URL

- **What's New**: 이번 논문에서는 MuMA-ToM이라는 새로운 Multi-modal Multi-Agent Theory of Mind 벤치마크를 소개하고 있습니다. 이는 AI가 복잡한 사회적 상호작용에서 사람들의 정신적 상태를 이해할 수 있도록 돕기 위해 고안되었습니다.

- **Technical Details**: MuMA-ToM는 현실적인 가정 환경에서 사람들의 다중 모달 행동에 대한 비디오 및 텍스트 설명을 제공하며, 이러한 맥락을 바탕으로 목표, 신념 및 다른 사람의 목표에 대한 신념에 관한 질문을 합니다. 또한, LIMP(언어 모델 기반의 역 다중 에이전트 계획)라는 새로운 다중 모달, 다중 에이전트 ToM 모델을 제안하였습니다.

- **Performance Highlights**: LIMP는 다양한 최신 방법들, 특히 대형 다중 모달 모델(GPT-4o, Gemini-1.5 Pro 등) 및 최근 다중 모달 ToM 모델(BIP-ALM)을 능가하는 성능을 보였습니다.



### Pruning By Explaining Revisited: Optimizing Attribution Methods to Prune CNNs and Transformers (https://arxiv.org/abs/2408.12568)
Comments:
          Accepted as a workshop paper at ECCV 2024 31 pages (14 pages manuscript, 4 pages references, 13 pages appendix)

- **What's New**: 이번 연구에서는 Deep Neural Networks (DNNs)의 비효율성을 해소하기 위해, 네트워크의 중요하지 않은 구성 요소를 제거하는 pruning 기법을 개선하고자 합니다. 특히, attribution methods (속성 메소드)에서 얻은 hyperparameters (하이퍼파라미터)를 최적화하여 pruning을 수행하는 새로운 프레임워크를 제안합니다.

- **Technical Details**: 제안한 방법에서는 Layer-wise Relevance Propagation (LRP) 기법을 사용하여 네트워크의 구성 요소의 중요도를 평가합니다. 이를 통해 특정 구성 요소가 제거되었을 때의 성능 하락을 최소화할 수 있도록 합니다. 또한, transformer 기반 네트워크(ViT)와 convolutional architectures (컨볼루션 아키텍처)에서 실험하였으며, pruning 성능을 더욱 향상시키기 위한 새로운 framework를 설계했습니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 VGG, ResNet, ViT와 같은 대형 아키텍처에서 기존 연구보다 높은 model compression rates (모델 압축률)를 달성하면서도 ImageNet 분류 작업에서 높은 성능을 유지했습니다. 또한, transformers는 convolutional neural networks (CNNs)보다 더 높은 과대파라미터화(over-parameterization)를 보임을 확인했습니다.



### MEDCO: Medical Education Copilots Based on A Multi-Agent Framework (https://arxiv.org/abs/2408.12496)
- **What's New**: 이번 연구에서는 기존 AI 교육 도구의 한계를 극복하기 위해 MEDCO라는 다중 에이전트 기반의 코파일럿 시스템을 제안합니다. 이 시스템은 실제 의료 교육 환경을 모사하며, 환자, 의료 전문가 및 방사선 전문의의 역할을 수행하는 세 가지 에이전트를 포함하고 있습니다.

- **Technical Details**: MEDCO는 다중 모드 및 상호작용 학습 환경을 제공하여, 학생들이 질문하기 기술과 다학제 협업, 동료 토론을 배울 수 있도록 설계되었습니다. 학생들은 에이전트 환자와 대화를 나누고, 의료 전문가로부터 피드백을 받으며 실제와 유사한 학습 경험을 합니다.

- **Performance Highlights**: 실험 결과, MEDCO로 교육받은 가상 학생들은 성과가 큰 향상을 보였으며, 기존의 우수한 모델들과 비교할 수 있는 수준으로 나아가는 모습을 보였습니다. MEDCO를 활용한 학생들은 사람과 유사한 학습 행동을 나타내며, 여러 동료와의 논의를 통해 진단 성과도 향상되었습니다.



### AI in radiological imaging of soft-tissue and bone tumours: a systematic review evaluating against CLAIM and FUTURE-AI guidelines (https://arxiv.org/abs/2408.12491)
Comments:
          23 pages, 6 figures, 6 supplementary figures

- **What's New**: 본 체계적 리뷰는 방사선 이미지를 사용하는 인공지능(AI) 방법이 드문 연조직 및 뼈 종양(STBT)의 진단 및 예후에서 어떻게 활용되고 있는지를 종합적으로 다루고 있습니다. 특히 임상 실현 가능성과 관련된 도전과제를 강조하며, AI 방법의 임상 적용을 촉진하기 위한 국제 지침과 기준 포함 여부를 평가하였습니다.

- **Technical Details**: 325개의 연구가 평가 대상으로 포함되었으며, 대부분의 연구는 CLAIM 체크리스트에서 28.9$	$±7.5의 평균 점수를 기록했습니다. 그러나 FUTURE-AI 기준에서는 평균 5.1$	$±2.1에 그쳤습니다. 이는 AI 도구들이 아직 개념 증명 단계에 머물러 있음을 나타내며, 향후 개선이 필요합니다.

- **Performance Highlights**: AI 개발자들은 unmet clinical need(미충족 임상 필요) 정의, 의도된 임상 환경, 임상 워크플로우에의 통합 방법을 명확히 하는 설계뿐만 아니라, 이전 연구를 기반으로 하고 설명 가능성을 고려해야 하며, AI의 편향을 평가하고 최선의 관행과 비교 평가하는 등의 개발 및 평가에 집중해야 합니다.



### Dataset | Mindset = Explainable AI | Interpretable AI (https://arxiv.org/abs/2408.12420)
- **What's New**: 본 논문에서는 'explainable AI (XAI)'와 'interpretable AI (IAI)'의 차이를 명확히 하고, IAI가 XAI의 철학적 기초임을 주장합니다. 이를 통해 AI 분야의 정책 입안자와 연구자들에게 방향성을 제시하고자 합니다.

- **Technical Details**: XAI는 데이터셋의 사후 분석(post-hoc analysis)을 강조하고, IAI는 추상화의 사전적 사고(a priori mindset)를 요구합니다. 이 연구는 고성능 컴퓨팅(High-Performance Computing, HPC) 기반의 실험을 통해 이 두 개념 간의 경계를 입증하고 있습니다. 또한, ML 모델에서 XAI의 기준을 정의하기 위한 3X3 고수준 추상화 매트릭스를 제안합니다.

- **Performance Highlights**: 이 논문은 XAI와 IAI의 혼란을 해소하기 위한 다양한 실험을 수행하고, ML 모델을 최적화하는 방법을 제시합니다. 특히, XAI/IAI 실험의 종합적인 프로세스(End-to-End Process)를 상세히 설명하고 있습니다.



### RoundTable: Leveraging Dynamic Schema and Contextual Autocomplete for Enhanced Query Precision in Tabular Question Answering (https://arxiv.org/abs/2408.12369)
Comments:
          13 pages, 4 figures

- **What's New**: 이번 논문은 데이터베이스 쿼리 작성에 있어 자연어 처리(Natural Language Processing)와 데이터의 복잡성을 해결할 수 있는 새로운 프레임워크를 제안합니다. 핵심적으로, Full-Text Search(FTS)를 활용하여 사용자 질문을 보다 정확하게 파악하고 쿼리 생성 과정을 간소화하려고 합니다.

- **Technical Details**: 이 프레임워크는 LLMs(대형 언어 모델)가 자연어 쿼리에서 특정 값과 열을 효과적으로 감지하도록 지원하며, 사용자가 복잡한 데이터셋과 상호작용하는 방식을 개선합니다. 또한, 자동 완성(auto-complete) 기능을 통해 입력된 테이블의 데이터에 기반한 예상 쿼리를 제안합니다.

- **Performance Highlights**: 제안된 시스템은 Mac 및 Windows 플랫폼에서 사용할 수 있는 애플리케이션으로 제공되어, 사용자들이 직접 자신의 데이터를 가지고 실험할 수 있는 기회를 제공합니다. 이렇게 함으로써, 데이터베이스의 대규모 및 복잡성을 처리하는 데 있어 LLM의 효율성과 정확성을 크게 향상시킬 수 있는 가능성을 열어주고 있습니다.



### Graph Retrieval Augmented Trustworthiness Reasoning (https://arxiv.org/abs/2408.12333)
- **What's New**: 이번 논문에서는 다중 플레이어 게임에서의 신뢰성(reasoning of trustworthiness) 평가를 개선하기 위해 Graph Retrieval Augmented Reasoning (GRATR) 프레임워크를 소개합니다. GRATR는 Retrieval-Augmented Generation (RAG) 기술을 활용하여 에이전트의 신뢰성 추론을 강화하고, 신뢰성 그래프를 실시간으로 업데이트하며, 관련 신뢰 데이터를 검색하여 대규모 언어 모델(LLM)의 추론 능력을 보완합니다.

- **Technical Details**: GRATR는 동적 신뢰성 그래프를 구축하며, 플레이어의 행동 및 발화를 기반으로 실시간 증거를 수집하여 그래프의 노드와 엣지를 업데이트합니다. 이 방법은 노드의 신뢰성 수준을 누적된 증거에 기반하여 계산하고 업데이트하여 플레이어의 신뢰성을 효과적으로 추론합니다. GRATR의 성능을 멀티플레이어 게임 'Werewolf'에서 실험을 통해 검증하였으며, 기존 LLM과 Native RAG, Rerank RAG로 향상된 LLM과 비교했습니다.

- **Performance Highlights**: 테스트 결과 GRATR는 기본 방법보다 승리 확률에서 30% 이상 초과하는 성능을 보여주었으며, LLM의 환각(hallucination) 문제를 효과적으로 완화하여 역할 및 목표의 망각을 줄였습니다. 또한, GRATR는 추론 과정을 더 투명하고 추적 가능하게 만들어 주는 신뢰성 그래프를 포함하고 있습니다.



### PolyRouter: A Multi-LLM Querying System (https://arxiv.org/abs/2408.12320)
Comments:
          14 pages, 7 figures, 2 tables

- **What's New**: PolyRouter는 다양한 도메인 전문가 모델을 통합하여 쿼리를 효율적으로 처리하는 새로운 다중 LLM 라우팅 시스템으로, 최대 40%의 쿼리 효율성과 30%의 비용 절감 효과를 보여줍니다.

- **Technical Details**: PolyRouter는 각 LLM 전문가의 성능을 기반으로 인커밍 쿼리를 동적으로 라우팅하여 최적의 전문가 모델을 선택합니다. 이 시스템은 BERT 유사도 점수를 활용한 소프트 레이블 방식을 사용하여 쿼리 공간을 학습합니다. 또한, 예측 라우터와 카스케이딩 라우터를 비롯한 다양한 라우팅 메소드를 검토하였습니다.

- **Performance Highlights**: PolyRouter는 개별 전문가 모델에 비해 평균 40% 향상된 쿼리 효율성과 30% 이상의 비용 절감을 실현하였으며, 모델 성능은 최대 10% 증가했습니다.



### Large Language Models Are Self-Taught Reasoners: Enhancing LLM Applications via Tailored Problem-Solving Demonstrations (https://arxiv.org/abs/2408.12315)
Comments:
          preprint / under review

- **What's New**: 본 논문은 대규모 언어 모델(Large Language Model, LLM)의 성능 향상을 위해 선택된 인간 작성 시연으로 안내하는 전통적인 접근 방식의 한계를 인식하고, 자동 생성된 맞춤형 시연을 통한 해결책을 제시합니다.

- **Technical Details**: SELF-TAUGHT라는 프레임워크를 소개하며, 이 시스템은 주어진 문제에 맞춰 조정된 시연을 자동으로 생성하고, 더욱 향상된 품질의 시연(정확성)을 제공하는 데 초점을 둡니다. 이 과정은 zero-shot 방식으로 실행됩니다. 해당 프레임워크에 대한 포괄적인 분석도 진행되어, 다양한 LLM과 프롬프트 방법에 대한 일반화 가능성과 중간 결과물의 품질 등이 포함됩니다.

- **Performance Highlights**: SELF-TAUGHT는 다섯 가지 다양한 분야의 다지선다형 질문 및 실제 환자를 대상으로 한 알츠하이머 질환 진단을 포함한 15개의 작업에서 강력한 기준선들을 초월하는 성능을 달성했습니다.



### Deep Learning with CNNs: A Compact Holistic Tutorial with Focus on Supervised Regression (Preprint) (https://arxiv.org/abs/2408.12308)
- **What's New**: 이번 튜토리얼에서는 깊은 학습(Deep Learning)과 관련된 이론적 기초를 다루고, 특히 합성곱 신경망(Convolutional Neural Networks, CNNs) 및 감독 회귀(supervised regression)에 중점을 둡니다. 이 튜토리얼은 기초적이면서도 엄격하고 접근 가능한 시각에서 깊은 학습을 설명하는 드문 자원입니다.

- **Technical Details**: 이 튜토리얼은 인공지능(Artificial Intelligence, AI) 및 깊은 학습(Deep Learning, DL)의 고전적 문헌을 따릅니다. 인공 신경망(Artificial Neural Networks, ANN) 및 깊은 인공 신경망(Deep Artificial Neural Networks)과 관련된 여러 개념에 대해 논의하며, CNN의 아키텍처에서 가장 중요한 구성 요소인 합성곱(convolution) 연산을 포함합니다. 또한 머신러닝(Machine Learning)에 대한 용어와 이론적 기초를 설명합니다.

- **Performance Highlights**: 이 논문은 감독 학습(supervised learning)을 통해 에이전트가 환경으로부터 지식(facts)을 학습하는 과정을 강조합니다. 유의미한 사례와 함께 에이전트가 예측을 잘 수행할 수 있도록 해주는 일반화(generalization) 능력에 대한 중요성을 강조합니다.



### Tipta uzmanlik sinavinda (tus) b\"uy\"uk d\.il modeller\.i \.insanlardan daha mi ba\c{s}arili? (https://arxiv.org/abs/2408.12305)
Comments:
          9 pages, in Turkish language, 8 figures

- **What's New**: 본 연구는 2021년 1기 Tıpta Uzmanlık Sınavı(TUS)에서 Türkçe tıbbi sorular을 답변할 수 있는 üç 가지 인공지능 모델의 성능을 비교 평가하여 인공지능이 tıp 교육과 평가에서 갖는 잠재력을 강조합니다.

- **Technical Details**: Gemini, ChatGPT-4, ChatGPT-4o 모델을 사용하여, toplam 240개의 질문(문제로 CMST, BMST로 구분됨)에 대해 그들의 답변 정확도를 분석하였습니다. 각 모델의 성능은 'Doğruluk'과 'Açıklama Derinlikleri'를 기반으로 평가되었습니다.

- **Performance Highlights**: ChatGPT-4o는 CMST에서 117문제, BMST에서 107문제를 정확히 답변하여 가장 높은 성적을 기록하였으며, ChatGPT-4는 105문제, Gemini는 82문제를 각각 정답으로 제출하였습니다. 이는 인공지능 모델들이 tıp eğitiminde yüksek 정확도와 문맥 이해를 달성할 잠재력을 보여줍니다.



### OPTDTALS: Approximate Logic Synthesis via Optimal Decision Trees Approach (https://arxiv.org/abs/2408.12304)
- **What's New**: 이 논문에서는 Empirical Accuracy를 통해 최적의 의사결정 나무(Optimal Decision Trees)를 학습하여 근사 논리 합성(Approximate Logic Synthesis, ALS)을 실현하는 새로운 방법론인 OPTDTALS를 제안합니다. 이는 기존의 휴리스틱 ALS 방식과 비교하여 회로 복잡성과 정확성 사이의 보다 제어 가능한 균형을 제공합니다.

- **Technical Details**: OPTDTALS는 입력 회로를 여러 개의 관리 가능한 하위 회로로 분할한 후, 각 하위 회로에 적용한 뒤 최종 근사 회로에 반복적으로 교체하는 디자인 스페이스 탐색 휴리스틱(design space exploration heuristic)을 도입합니다. 또한, 평균 상대 오차(Average Relative Error)를 오류 메트릭으로 사용하여 근사된 논리 함수의 품질을 평가합니다.

- **Performance Highlights**: 실험 결과는 OPTDTALS 방법론이 기존 최첨단 ALS 방법과 비교하여 약속된 품질 및 효율성의 향상을 보여줍니다. OPTDTALS는 더 작은 크기의 근사 회로를 제공하면서도 최적성을 보장하는 새로운 접근방식입니다.



### AT-SNN: Adaptive Tokens for Vision Transformer on Spiking Neural Network (https://arxiv.org/abs/2408.12293)
Comments:
          8 pages

- **What's New**: 이 논문에서는 SNN 기반 ViT에서 직선 학습(direct training)과 경량화 계산(lightweight computation) 방법을 결합한 AT-SNN을 제안하여, 토큰(token) 수를 동적으로 조절하여 전력 소비를 줄이며 정확도를 향상시키는 방안을 소개합니다. 추가적으로, ACT(adaptive computation time)를 SNN에 적용하여 정보가 적은 공간 토큰을 선택적으로 제거할 수 있도록 합니다.

- **Technical Details**: AT-SNN은 SNN 기반 ViT에서 두 가지 차원에서 조정하는 이탈 방법을 적용하는 최초의 시도로, ACT와 함께 가변적인 토큰 비율을 사용하여 계산 효율성을 증가시키고, 새로운 토큰 병합(token-merge) 메커니즘을 통해 유사한 토큰들을 병합하여 토큰 수를 further 줄입니다. 이 방법들은 CIFAR-10, CIFAR-100, TinyImageNet을 포함한 이미지 분류(task)에서 검증되었습니다.

- **Performance Highlights**: AT-SNN은 CIFAR-100에서 기존 최고의 방법 대비 42.4% 더 적은 토큰을 사용하면서도 높은 정확도를 유지하는 성능을 보여주었습니다. 특히, Spikformer에 구현된 AT-SNN은 TET 및 DT-SNN보다 73.23%와 75.81%의 정확도를 달성하였습니다.



### Can You Trust Your Metric? Automatic Concatenation-Based Tests for Metric Validity (https://arxiv.org/abs/2408.12259)
- **What's New**: 이번 연구는 대규모 언어 모델의 응답 중 해로운 것들을 필터링하는 위험 검출 메트릭의 중요한 특성을 다룹니다. 특히, 메트릭이 개별적인 해로운 프롬프트-응답 쌍에서는 높은 점수를 부여하는 반면, 이를 연결했을 때는 오히려 가장 낮은 점수를 부여하는 현상을 발견했습니다. 이는 기존의 여러 LLM 기반 메트릭, 특히 GPT 기반 메트릭에서 발생하는 중요한 문제입니다.

- **Technical Details**: 본 연구에서는 연결(concatenation) 기반 테스트를 도입하여 유효한 메트릭이 만족해야 할 기본 특성을 평가했습니다. 특히, GPT-3.5 및 GPT-4o 메트릭은 입력 순서에 매우 민감하다는 것을 발견했습니다. 예를 들어, 안전한 내용이 먼저 올 경우, 뒤에 해로운 내용이 있더라도 안전한 것으로 분류되는 경향이 있습니다. 이러한 현상은 모델 안전성(modes safety) 분야에서 중요하며, 해로운 응답을 측정하는 데 있어 신뢰할 수 있는 메트릭을 설계하는 데 필수적입니다.

- **Performance Highlights**: 연구 결과, GPT-3.5 기반 메트릭은 결정 플리핑(decision flipping) 비율이 눈에 띄게 높았으며, GPT-4o 메트릭 또한 강한 위치 편향(positional bias)을 보였습니다. 또한, 공격 방법(the attack methods)에 대한 메트릭의 성능을 평가하기 위한 자동화된 테스트가 특히 효과적임을 보여주었습니다. 이러한 특성 덕분에 사용자는 특정 평가 메트릭을 선택할 때 더 나은 결정을 내릴 수 있습니다.



### Can Artificial Intelligence Embody Moral Values? (https://arxiv.org/abs/2408.12250)
- **What's New**: 이 논문은 인공지능(AI)의 결정이 도덕적 해를 초래할 수 있는 고위험 분야에서의 사용을 강조하며 기술 중립성(thesis of neutrality) 주장에 도전합니다.

- **Technical Details**: 저자들은 인공지능, 특히 자율적으로 결정을 내리는 인공 에이전트가 도덕적 가치를 반영할 수 있는 계산 모델(computational models)을 통합할 수 있다고 주장합니다. 두 가지 접근 방식, 즉 인공지능의 인공 양심(artificial conscience)과 윤리적 자극(ethical prompting)을 논의합니다. 이러한 모델이 포함된 인공 에이전트의 행동이 윤리적이라는 실증적 증거도 제시합니다.

- **Performance Highlights**: 논문에서 제시된 실험 결과에 따르면, 윤리적 모델이 적용된 인공 에이전트는 그렇지 않은 에이전트에 비해 더 윤리적인 행동을 보입니다. 이는 모든 기술이 필연적으로 가치 중립적이라는 주장을 반박하는 결과입니다.



### Efficient Multivariate Time Series Anomaly Detection Through Transfer Learning for Large-Scale Web services (https://arxiv.org/abs/2408.12247)
- **What's New**: 이번 논문에서는 Self-Evolution이라는 새로운 프레임워크를 제안하여, 경량 오픈소스 LLM을 활용하여 도메인 특정 데이터를 개선하는 방법을 설명합니다. 이 프레임워크는 여러 번의 반복적인 fine-tuning(파인튜닝)을 통해 LLM의 도메인 성능을 향상시키고, 이 과정에서 높은 가치를 가진 지식을 필터링 및 강화합니다.

- **Technical Details**: Self-Evolution은 7B 파라미터를 가진 오픈소스 모델을 사용하여 데이터 생성 및 QA 작업을 수행합니다. 이 프레임워크는 LLM이 많은 무표시 지식 문서로부터 QA 쌍을 생성하며, 이로써 도메인 관련성 및 정확성을 보장합니다. 또한, 자기 반복적 업데이트를 통해 다양한 배치의 데이터 생성을 보장합니다.

- **Performance Highlights**: Self-Evolution을 사용하여 4,000개의 도메인 지식을 포함한 문서로 평가한 결과, 도메인 특정 질문-답변 평가에서 Qwen1.5-7B-Chat보다 174% 높은 성과를 달성하였으며, Qwen1.5-72B-Chat보다도 22% 높은 성과를 보였습니다. 또한, 117일간 중국 모바일의 운영과 유지 관리에 배포되었으며, 경고 탐색 및 문제 해결의 효율성이 평균 18.6% 개선되었습니다.



### Weight Scope Alignment: A Frustratingly Easy Method for Model Merging (https://arxiv.org/abs/2408.12237)
- **What's New**: 본 논문에서는 다양한 훈련 조건에서의 가중치 스코프(Weight Scope)의 변화를 조사하여 모델 병합에 미치는 영향을 밝힙니다. 저자들은 Weight Scope Alignment (WSA)라는 새로운 정규화 접근 방식을 제안합니다.

- **Technical Details**: WSA는 두 가지 주요 구성 요소로 구성됩니다: 1) 목표 가중치 스코프를 활용하여 모델 훈련 프로세스를 안내하고, 2) 두 개 이상의 모델의 가중치 스코프를 통합하여 다단계 모델 융합을 수행하는 것입니다. WSA는 mode connectivity 및 federated learning의 두 가지 다른 시나리오로 확장됩니다.

- **Performance Highlights**: 많은 실험적 연구를 통해 WSA의 효과가 검증되었으며, 이 방법은 모델의 원래 능력을 유지하면서 일반화, 효율성 및 견고성을 개선하는 데 기여하고 있습니다.



### MedDiT: A Knowledge-Controlled Diffusion Transformer Framework for Dynamic Medical Image Generation in Virtual Simulated Patien (https://arxiv.org/abs/2408.12236)
- **What's New**: 이 논문에서는 MedDiT라는 새로운 지식 기반 대화형 프레임워크를 소개하며, 의료 이미지 생성과 시뮬레이션 환자 증상에 맞춰 다양한 진단 기술 훈련을 가능하게 합니다.

- **Technical Details**: MedDiT는 환자 속성과 증상을 설명하는 다양한 Patient Knowledge Graphs (KGs)를 통합하여 Large Language Models (LLMs)의 동작을 제어하고, Diffusion Transformer (DiT) 모델을 통해 의료 이미지를 생성합니다.

- **Performance Highlights**: MedDiT는 다양한 시뮬레이션 환자 사례를 생성하고 그에 따른 의료 이미지를 제공함으로써 학생들에게 풍부하고 상호작용적인 학습 경험을 제공합니다. 이는 미래의 의료 전문가들을 위한 몰입형 시뮬레이션 플랫폼을 제공하며, 의료 교육의 발전에 기여할 것으로 기대됩니다.



### UNCO: Towards Unifying Neural Combinatorial Optimization through Large Language Mod (https://arxiv.org/abs/2408.12214)
- **What's New**: 본 논문에서는 통합 신경 조합 최적화(unified neural combinatorial optimization, UNCO) 프레임워크를 제안하여 다양한 조합 최적화 문제(combinatorial optimization problems, COPs)를 단일 모델로 해결하고자 합니다. 특히, 자연어를 사용하여 텍스트 속성 인스턴스(text-attributed instances)를 정의하고, 이를 대형 언어 모델(large language model, LLM)로 동일한 임베딩 공간(embedding space)으로 인코딩합니다.

- **Technical Details**: UNCO 프레임워크는 여러 COP를 해결하기 위해 텍스트 속성 인스턴스를 사용하고, 이를 LLM으로 인코딩합니다. 문제 특정 모듈 없이 인코더-디코더 모델을 사용하여 통합된 솔루션 생성 프로세스를 지원하며, 충돌 기울기 삭제 강화 학습(conflict gradients erasing reinforcement learning, CGERL) 알고리즘을 도입하여 UNCO 모델을 훈련시킵니다.

- **Performance Highlights**: 실험 결과 UNCO 모델은 여러 COP를 단일 세션 훈련으로 해결할 수 있으며, 기존의 전통적 및 학습 기반 기준과 유사한 성능을 나타냅니다. 각 COP의 최적 성능을 추구하기보다는 작업 간의 시너지를 탐구하고 LLM 기반의 몇 샷 일반화를 모색합니다.



### Relational decomposition for program synthesis (https://arxiv.org/abs/2408.12212)
- **What's New**: 본 논문에서는 복잡한 기능적 작업을 단순한 관계적 합성의 하위 작업으로 분해하는 새로운 프로그램 합성 접근 방식을 소개합니다. 이 방법은 기존의 인덕티브 로직 프로그래밍(ILP) 시스템을 사용하여 세 가지 도전적인 데이터 세트에서 효과성을 입증했습니다.

- **Technical Details**: 기존의 기능적 접근 방식은 단순한 프로그램에는 효과적이지만 긴 함수 시퀀스를 요하는 프로그램 학습에 어려움을 겪습니다. 본 연구에서는 각 교육 입력-출력 예제를 사실(facts) 집합으로 분해하고 이들 간의 관계를 학습합니다. 각각의 입력 및 출력 데이터를 개별적인 픽셀의 사실로 나누어 관계적 규칙을 학습합니다.

- **Performance Highlights**: 우리의 실험 결과는 (i) 관계적 인코딩이 기존의 기능적 인코딩에 비해 학습 성능을 크게 향상시킴을 보여주며, (ii) 상용 ILP 시스템이 관계적 인코딩을 사용하면 특정 도메인 접근 방식보다 뛰어난 성과를 거둘 수 있음을 입증했습니다.



### Randomness control and reproducibility study of random forest algorithm in R and Python (https://arxiv.org/abs/2408.12184)
- **What's New**: 이 논문에서는 화장품의 안전성을 보장하기 위해 독성학자들이 알고리즘의 재현성을 확보하는 방법을 논의합니다. 특히, 무작위성을 기반으로 하는 랜덤 포레스트(random forest) 알고리즘의 강건성을 입증하는 전략을 제안합니다.

- **Technical Details**: 랜덤 포레스트 알고리즘의 성능을 비교하기 위해 네 가지 패키지를 분석합니다: randomForest와 Ranger(R 패키지), SKRanger 패키지를 통한 Python 버전, 그리고 널리 사용되는 Scikit-Learn의 RandomForestClassifier() 함수입니다. 이 연구에서는 결과에 영향을 미치는 무작위성의 매개변수와 출처를 조사합니다.

- **Performance Highlights**: 같은 Pseudo-Random Number Generator (PRNG)를 사용하여 설정 가능한 매개변수를 적용함으로써, 다양한 랜덤 포레스트 알고리즘의 구현에서 일관된 결과를 재현할 수 있는 가능성을 모색합니다. 이 과정을 통해 랜덤 포레스트 알고리즘의 재현성을 보장하기 위한 중요한 매개변수를 이해할 수 있게 됩니다.



### Multi-tool Integration Application for Math Reasoning Using Large Language Mod (https://arxiv.org/abs/2408.12148)
- **What's New**: 본 연구에서는 대규모 언어 모델(LLM)과 여러 외부 도구의 협업 효과를 활용하여 보다 포괄적이고 정확한 수학적 추론을 달성하기 위한 새로운 다중 도구 애플리케이션 프레임워크를 제안합니다.

- **Technical Details**: 제안하는 프레임워크는 Math Tool, Code Tool, CoT Tool 및 self consistency tool을 포함하며, LLM의 추론 프로세스 중 다양한 외부 도구를 활용합니다. 이를 통해 기본적인 수학 계산 수행, 코드 생성 및 실행, 그리고 사고의 연쇄를 통한 반복적 추론 과정이 이루어집니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 NumGLUE Task 4 테스트 세트에서 220개의 수학적 추론 문제에 대해 89.09%의 정확도를 달성했습니다. 이는 기존의 GPT3+FewShot 기준과 비교했을 때, Few Shot+ERNIE-4.0+self consistency 조합에서 49.09% 향상되었고, 파인튜닝 기준과 비교했을 때는 52.29% 향상된 결과입니다.



### Self-supervised Learning for Geospatial AI: A Survey (https://arxiv.org/abs/2408.12133)
- **What's New**: 이 논문은 자가 감독 학습(Self-Supervised Learning, SSL)이 지리공간 인공지능(GeoAI) 분야에 적용된 최신 기법을 심층적으로 조사한 연구입니다. 이는 특히 점, 폴리라인, 폴리곤 등 지리공간 벡터 데이터의 세 가지 주요 형식을 기반으로 SSL 기법을 분류하는 내용을 포함합니다.

- **Technical Details**: 논문에서는 SSL 기법을 예측(predictive) 방법과 대조(contrastive) 방법으로 체계적으로 분류하고, 각 데이터 유형에 대한 일반화 향상과 관련된 응용 사례를 논의합니다. 그 외에도 SSL을 GeoAI에 적용할 때의 도전 과제를 제시하고, 향후 연구 방향을 탐색합니다.

- **Performance Highlights**: 이 논문은 SSL이 지리공간 데이터에서 효과적이고 일반화 가능한 표현(embeddings)을 학습할 수 있다는 점을 강조합니다. 또한 이 방법은 레이블이 부족한 지리공간 데이터의 한계를 극복할 수 있는 가능성을 제공합니다.



### S-EPOA: Overcoming the Indivisibility of Annotations with Skill-Driven Preference-Based Reinforcement Learning (https://arxiv.org/abs/2408.12130)
Comments:
          Submitted to AAAI 02025

- **What's New**: 이번 논문에서는 Skill-Enhanced Preference Optimization Algorithm (S-EPOA)을 제안하여 기존의 Preference-based Reinforcement Learning (PbRL) 방법의 주석 비분할성(indivisibility of annotations) 문제를 해결합니다.

- **Technical Details**: S-EPOA는 비지도(pretraining) 기술을 사용하여 유용한 스킬을 학습한 후, 스킬 공간에서 정보 획득(information gain)과 구별 가능성(discriminability)을 균형 있게 조절하는 새로운 질의 선택(query selection) 메커니즘을 도입합니다.

- **Performance Highlights**: 실험 결과, S-EPOA는 로봇 조작(robotic manipulation) 및 보행(locomotion) 작업에서 기존의 PbRL 방법들보다 강건성(robustness) 및 학습 효율성(learning efficiency)에서 유의미한 성과를 보여주었습니다.



### Diffusion-Based Visual Art Creation: A Survey and New Perspectives (https://arxiv.org/abs/2408.12128)
Comments:
          35 pages, 9 figures

- **What's New**: 이번 설문조사는 생성 AI(Generative AI)가 시각 예술에서의 확산 기반(difussion-based) 창작에 미치는 영향을 탐구합니다. 예술적 및 기술적 관점에서의 발전을 세 가지 단계로 나누어 데이터를 주요 특성 및 프레임워크에 대한 식별, 구조화된 코딩 프로세스를 통한 세부 분석, 향후 전망을 제시합니다. 우리의 연구 결과는 예술적 요구가 어떻게 기술적 도전으로 변형되는지를 보여줍니다.

- **Technical Details**: 설문조사는 확산 모델의 개발과 시각 예술 생성의 상관관계를 다루며, 데이터 수집을 통해 사용자의 요구와 기술 문제 간의 상호작용을 분석합니다. 총 네 가지 주요 연구 질문을 바탕으로, 현재의 화제, 도전 과제, 사용된 방법론, 그리고 앞으로의 방향성을 탐색합니다. 확산 기반 방법들은 제어 가능성, 특정 장르 우선의 예술 창작을 포함한 다양한 세부 분야로 확장되고 있습니다.

- **Performance Highlights**: 이번 연구는 시각 예술 창작에서 확산 모델의 잠재적 혁신을 강조하며, AI 시스템이 인간의 예술적 인식 및 창의성을 어떻게 모사하고 향상시킬 수 있는지를 보여줍니다. 또한, 혁신적 통합이 예술 창작의 새로운 가능성을 열어줄 것으로 기대하며, 앞으로의 연구 방향에 대한 통찰을 제공합니다.



### Geolocation Representation from Large Language Models are Generic Enhancers for Spatio-Temporal Learning (https://arxiv.org/abs/2408.12116)
- **What's New**: 이 논문에서는 지리적 표현 모델인 LLMGeovec를 새롭게 제안하며, 이는 OpenStreetMap의 보조 지도 데이터와 대규모 언어 모델(LLMs)을 활용하여 훈련이 필요 없는 방식으로 지리적 표현을 도출하는 혁신적인 방법입니다.

- **Technical Details**: LLMGeovec는 도시, 국가, 그리고 세계 규모에서 지리적 의미를 표현할 수 있으며, 공간-시간 학습(spatio-temporal learning)을 위한 일반적인 향상제로 작용합니다. 이 방법은 OpenStreetMap에서 추출한 텍스트 설명을 기반으로 하여 LLMs가 이를 처리하고, 최종 토큰의 숨겨진 상태를 평균 내어 각 좌표에 대한 LLMGeovec 임베딩을 생성합니다.

- **Performance Highlights**: LLMGeovec는 GP, LTSF, GSTF 모델의 성능을 크게 향상시키며, 기존의 자원 집약적인 GNNs를 대체할 수 있는 잠재력을 보유하고 있습니다. 실험 결과, LLMGeovec는 전 세계적으로 커버리지를 달성하고 다양한 spatio-temporal 학습 모델의 성능을 즉각적으로 개선하는 것으로 나타났습니다.



### Transformers As Approximations of Solomonoff Induction (https://arxiv.org/abs/2408.12065)
- **What's New**: 본 논문에서는 Solomonoff Induction (SolInd)과 비교하여 Transformer 모델이 시퀀스 예측에서 우수한 성능을 보이는 이유를 분석하고, Transformer 모델이 SolInd의 근사치로 작용할 수 있다는 가설을 제시합니다.

- **Technical Details**: Solomonoff Induction은 모든 계산 가능한 확률 분포의 베이즈 혼합을 나타내는 최적의 비디오 알고리즘입니다. 이 방법은 unbounded (무한) 알고리즘으로, 고전적인 Turing Machine (TM)을 기반으로 합니다. Doom의 적합도 비교를 통해 Transformers는 실질적으로 SolInd와 유사한 구조를 가지고 있다고 설명됩니다.

- **Performance Highlights**: Transformers는 다른 시퀀스 예측 방법보다 Solomonoff Induction을 더 잘 근사할 수 있으며, 테스트 시 각 Transformer의 예측 성능이 SolInd에 가깝다는 주장을 하고 있습니다. 또한, stochastic gradient descent (확률적 경량 하강법)는 훈련 과정에서 SolInd를 더 잘 근사할 수 있다는 가설도 제시합니다.



### Exploring Large Language Models for Feature Selection: A Data-centric Perspectiv (https://arxiv.org/abs/2408.12025)
Comments:
          Preprint, under review

- **What's New**: 이번 연구에서는 LLM(대형 언어 모델)을 기반으로 한 특징 선택(feature selection) 방법을 데이터 중심(data-centric) 관점에서 탐구하고 이해하고자 합니다. LLM 기반 특징 선택을 두 가지 방식으로 분류하였으며, 이를 통해 의료 분야의 실제 사례에서 그 가능성을 보여주었습니다.

- **Technical Details**: 특징 선택 방법은 데이터 기반(data-driven)과 텍스트 기반(text-based)의 두 가지로 나눌 수 있습니다. 데이터 기반 방법은 샘플 값을 통해 통계적 추론을 수행하는 반면, 텍스트 기반 방법은 LLM의 사전 지식을 활용하여 서술적 맥락을 통해 의미 연관을 만듭니다. 본 연구는 GPT-4, ChatGPT, LLaMA-2 등 다양한 크기의 LLM을 사용하여 분류(classification) 및 회귀(regression) 작업에서 두 방법을 비교하였습니다.

- **Performance Highlights**: 텍스트 기반 특징 선택 방법은 여러 저자원(low-resource) 환경에서 더 효과적이며 안정적으로 나타났습니다. 또한, 의료 분야에서 생존 예측을 위한 실제 응용 사례를 통해, 개발한 Retrieval-Augmented Feature Selection(RAFS) 방법이 효과적으로 특징 선택을 수행하는 것을 입증하였습니다.



### SimBench: A Rule-Based Multi-Turn Interaction Benchmark for Evaluating an LLM's Ability to Generate Digital Twins (https://arxiv.org/abs/2408.11987)
- **What's New**: SimBench라는 새로운 벤치마크를 소개합니다. 이는 학생 대형 언어 모델(S-LLMs)이 시뮬레이터에서 가상 테스트를 위해 사용할 수 있는 디지털 트윈(DTs)을 생성하는 능력을 평가하기 위해 설계되었습니다.

- **Technical Details**: SimBench는 20개 이상의 오픈 및 클로즈드 소스 S-LLMs를 비교하여 DT 생성에 대한 S-LLM의 칭찬을 평가합니다. 이 평가는 규칙 기반 심사 LLM(J-LLM)을 사용하여 이루어지며, 사전 정의된 규칙과 사람의 피드백을 통해 DT에 점수를 매깁니다. J-LLM은 특정 시뮬레이터에 맞춰 설계됩니다. 실험적으로 Chrono라는 다물리 시뮬레이터와 결합하여 평가를 수행하였습니다.

- **Performance Highlights**: SimBench는 S-LLM의 DT 생성 능력을 측정하는 최초의 방법론을 확립하며, 고품질의 디지털 트윈을 생성할 수 있는 능력에 대한 메트릭을 제시합니다. 총 2000개의 대화가 제출되었고, 그 안에 내장된 전문가의 평가 점수와 함께 S-LLMs의 성능을 비교했습니다.



### Sentiment and Emotion-aware Multi-criteria Fuzzy Group Decision Making System (https://arxiv.org/abs/2408.11976)
Comments:
          Submitted to FSDM 2024 - The 10th International Conference on Fuzzy Systems and Data Mining

- **What's New**: 이 논문은 감정 및 정서 인식(multi-criteria fuzzy GDM) 다기준 퍼지(GDM) 시스템을 소개하며, 다양한 선호도를 가진 참여자들 간의 합의 촉진을 목표로 합니다.

- **Technical Details**: 이 시스템은 Natural Language Processing(NLP)을 사용하여 텍스트 데이터에 표현된 감정 및 정서를 분석하여, 참가자의 의견을 이해하고 명시적 수치 선호 입력 외에 더 많은 정보(오피니언)를 고려합니다. GDM 시스템은 전문가들이 선호도를 제공한 후 개별 선호도를 집계하여 집합적 선호 행렬을 생성하고, 퍼지 추론 시스템(fuzzy inference system)을 통해 전체 점수를 산출합니다.

- **Performance Highlights**: 본 시스템은 친구 그룹이 휴가를 위해 호텔을 선택하는 소규모 의사결정 과정에 적용되었으며, 감정 및 정서 분석을 통합함으로써 참여자 간의 합의가 크게 향상됨을 보여주었습니다.



### Advances in Preference-based Reinforcement Learning: A Review (https://arxiv.org/abs/2408.11943)
- **What's New**: 이 논문은 Preference-based Reinforcement Learning (PbRL)의 새로운 통합 프레임워크를 제안하고, 최신 연구 동향을 반영하여 PbRL의 확장성 및 효율성을 향상시키는 방법을 다룹니다. 이 프레임워크는 기존의 RL 알고리즘이 수반하는 보상 함수의 설계 의존성을 극복하려는 노력의 일환으로, 전문가의 선호를 피드백으로 활용하여 학습 에이전트를 인도합니다.

- **Technical Details**: PbRL 알고리즘에서는 절대적인 수치 보상 대신, 상태, 행동 또는 궤적 간의 쌍에 대한 선호를 통해 전통적인 RL 문제를 해결합니다. PbRL의 수학적 모델은 Markov Decision Processes for Preferences (MDPP)로 정의됩니다. 여기서 상태와 행동 공간은 이산적이거나 연속적일 수 있으며, 선호 관계는 궤적 간의 비교로 나타납니다. 최적의 정책을 학습하려는 목표는 전문가로부터 받은 선호 관계를 만족시키는 궤적을 생성하는 것입니다.

- **Performance Highlights**: PbRL의 성능 개선을 위한 이론적 보장 및 벤치마킹 작업을 제시하고, 특히 자연어 처리(NLP) 분야에서의 최근 응용 사례를 다룹니다. 현재 PbRL 접근 방식의 한계와 향후 연구 방향을 제안하여 이 분야의 발전에 기여하고자 합니다.



### Matmul or No Matmal in the Era of 1-bit LLMs (https://arxiv.org/abs/2408.11939)
Comments:
          13 pages, 12 figures

- **What's New**: 이번 연구에서는 1-bit LLM(대형 언어 모델)의 도입이 가져온 새로운 연구 기회와 목적에 대해 강조하고 있습니다. 특히, 1-bit LLM이 모델의 효율성을 어떻게 변화시킬 수 있는지를 이해하는 것이 중요하다는 점을 다룹니다.

- **Technical Details**: 1-bit LLM은 Amdahl의 법칙을 모델 성능 컨텍스트에 맞춰 적용하여 부분 개선이 전체 모델 성능에 미치는 영향을 정량적으로 분석합니다. 연구에서는 다양한 모델 아키텍처와 하드웨어 구성에서의 실험을 통해 중요한 미세 조정을 발견하여 1-bit LLM의 미래 연구를 위한 로드맵을 제공합니다.

- **Performance Highlights**: 실험 결과, 1-bit LLM은 기억 용량과 계산 효율성을 동시에 향상시키며, 에지 및 클라우드 환경에서의 모델 최적화를 통해 모바일 장치에서도 실시간 처리에 기여할 수 있는 가능성을 보여주었습니다. 또한, 일부 MatMul(행렬 곱셈) 작업을 MatMul-free(행렬 곱셈 없는) 연산으로 변환함으로써 전체 계산 및 메모리 사용량에 미치는 영향을 평가하고 있습니다.



### Estimating Contribution Quality in Online Deliberations Using a Large Language Mod (https://arxiv.org/abs/2408.11936)
- **What's New**: 이 논문에서는 참여자들이 지식을 교환하고 주장을 펼치며 관점을 나누는 과정인 'deliberation'을 다룹니다. Stanford Online Deliberation Platform은 인간 조정자가 필요 없이 소규모 그룹을 위한 구조화된 일정으로 비디오 기반 온라인 논의를 지원합니다. 이 연구는 다양한 논의 이벤트에서 수집된 데이터를 분석합니다.

- **Technical Details**: 본 논문에서는 대규모 언어 모델(Large Language Model, LLM)을 사용하여, 인간 평가자 8명과 함께 기여도를 정당성(justification), 참신성(novelty), 대화 확대(expansion), 그리고 추가적인 전개 가능성(potential for further expansion) 기준에 따라 평가합니다. 기여점수는 1점에서 5점까지 매겨지며, 평균 평가는 인간 평가자를 기준으로 하여 모델의 성능을 평가합니다. 자동화된 품질 점수를 사용하여 비활성 참여자에게 타겟화된 nudging의 효과를 분석합니다.

- **Performance Highlights**: 모델은 개별 인간 평가자보다 성능이 뛰어나며, 그러나 두 명의 인간 평가자의 쌍은 정당성 평가에서 모델보다 우수한 성과를 보입니다. nudging 후에는 30초 이내에 발언 요청을 할 확률이 65% 증가하며, 모델은 nudging 없이 생성된 기여들에 비해 비슷한 품질 점수를 보여줍니다.



### An Open Knowledge Graph-Based Approach for Mapping Concepts and Requirements between the EU AI Act and International Standards (https://arxiv.org/abs/2408.11925)
Comments:
          This work was presented at the 9th International Symposium on Language & Knowledge Engineering (LKE 2024) Dublin, Ireland, 4 - 6 June, 2024

- **What's New**: 이번 논문에서는 신뢰할 수 있는 AI(Artificial Intelligence) 시스템에 대한 다양한 이니셔티브로 인해 복잡한 국제 가치 사슬 내에서 조직들이 직면하는 도전 과제를 다룹니다. 유럽연합(EU)의 AI 법안이 규정 준수를 위해 기술 요구 사항의 일치를 중심으로 초점을 이동시키고 있으며, 이에 따라 조화된 기준(Harmonized Standards)에 대한 초점이 필요하다는 점을 강조합니다.

- **Technical Details**: 이 연구는 AI 법안(AI Act)과 ISO 관리 시스템 표준(ISO management system standards)의 규범적 진술(normative statements)과 관련된 용어 및 요구 사항을 개방형 지식 그래프(Open Knowledge Graphs)로 매핑하는 과정에서 발생하는 어려움에 대해 설명합니다. 이를 통해 AI 법안에 따른 규제 준수의 적절성을 평가하고, 신뢰할 수 있는 AI 가치 사슬에서 기술적 합의 개발의 필요성이 있는 영역을 식별할 수 있습니다.

- **Performance Highlights**: 연구 결과, AI 법안의 높은 위험군에 대한 규정 준수를 위해서는 표준과의 일치성을 입증해야 하며, ISO/IEC 42001과 같은 국제 표준이 EU의 하모니제이션 요청에 대한 강력한 후보로 자리를 잡을 수 있다는 점이 밝혀졌습니다. 이는 AI 시스템의 안전 및 품질 관리에 대한 규정이 다소 복잡한 수직적 및 수평적 요구 사항을 포함하고 있음을 감안할 때 중요합니다.



### SAGE-RT: Synthetic Alignment data Generation for Safety Evaluation and Red Teaming (https://arxiv.org/abs/2408.11851)
- **What's New**: SAGE-RT(또는 SAGE)는 안전 평가 및 레드 팀 데이터 생성에 필요한 합성 alignment 데이터 생성의 새로운 파이프라인으로, 기존 방법의 한계를 극복하여 더 다양하고 세부적인 데이터셋을 생성할 수 있는 방법을 제시합니다.

- **Technical Details**: SAGE는 ALERT에 기반하여 해로운 주제에 대한 세부 분류법을 사용하여 51,000 개의 다양한 질의-응답 쌍을 생성하였습니다. 그 과정에서 1,500 개 이상의 해로운 주제를 다루고, LLM이 직면하는 다양한 jailbreaking 질의를 포함합니다. 이 방법은 설치된 LLM의 안전성을 보장하기 위해 synthetic red-teaming과 alignment 데이터를 완전히 합성적으로 생성할 수 있습니다.

- **Performance Highlights**: SAGE를 통해 생성된 red-teaming 데이터는 32 개 하위 카테고리 중 27 개에서 최첨단 LLM을 jailbreak 하였으며, 279 개 리프 카테고리 중 58 개에서 성공적인 공격 비율을 보였습니다. GPT-4o 및 GPT-3.5-turbo에 대해 100%의 공격 성공률을 기록하며, 안전성 관련 하위 카테고리 testing에서 극적인 성과를 나타냈습니다.



### ND-SDF: Learning Normal Deflection Fields for High-Fidelity Indoor Reconstruction (https://arxiv.org/abs/2408.12598)
- **What's New**: 이 논문에서는 ND-SDF라는 새로운 방법을 제안하여, 장면의 법선과 선행 법선 간의 각도 변화를 학습하여 정밀한 기하학을 복원하였습니다. 이 방법은 동적으로 샘플들을 활용하여, 복잡한 구조의 세부 사항을 보존하면서 매끄러운 표면을 유지합니다.

- **Technical Details**: ND-SDF는 Normal Deflection Field를 도입하여, 각기 다른 특성을 가진 샘플들에 따라 동적으로 샘플 활용을 조절하여, 정확도를 높이고 모델의 효용성을 개선합니다. 이는 또한 ray sampling 전략을 도입하여, 편향되지 않은 렌더링 프로세스를 촉진합니다.

- **Performance Highlights**: 이 방법은 벽과 바닥 같은 매끄럽고 질감이 덜한 영역을 효율적으로 처리하며, 복잡한 구조의 기하학적 세부 사항도 잘 보존합니다. 다양한 챌린징 데이터셋에서 일관된 개선이 확인되어, 이 방법의 우수성을 입증하였습니다.



### xGen-VideoSyn-1: High-fidelity Text-to-Video Synthesis with Compressed Representations (https://arxiv.org/abs/2408.12590)
Comments:
          Accepted by ECCV24 AI4VA

- **What's New**: 이 논문은 xGen-VideoSyn-1이라는 새로운 text-to-video (T2V) 생성 모델을 소개합니다. 이 모델은 텍스트 설명으로부터 현실적인 장면을 생성할 수 있는 능력을 갖추고 있으며, 최근 OpenAI의 Sora와 같은 진전을 토대로 하고 있습니다.

- **Technical Details**: 모델은 latent diffusion model (LDM) 아키텍처에 기반하고 있으며, 비디오 변분 오토인코더 (VidVAE)를 도입하여 비디오 데이터를 공간적(spatial) 및 시간적(temporal) 차원에서 효과적으로 압축합니다. 새로운 divide-and-merge 전략을 통해 비디오 세그먼트 간의 시간적 일관성을 유지합니다.

- **Performance Highlights**: xGen-VideoSyn-1 모델은 720p 해상도에서 최대 14초의 비디오 생성이 가능하며, 상태-of-the-art T2V 모델과의 비교에서 경쟁력 있는 성능을 보여줍니다. 이 모델은 13M 이상의 고품질 비디오-텍스트 쌍으로부터 학습되었습니다.



### Identifying the Best Arm in the Presence of Global Environment Shifts (https://arxiv.org/abs/2408.12581)
Comments:
          Extended version of the paper accepted at the 27th European Conference on Artificial Intelligence (ECAI 2024); Paper ID: M1125

- **What's New**: 본 논문은 비정상 확률적 밴딧 설정에서 새로운 베스트 암 식별 문제(Best-Arm Identification problem)를 공식화합니다. 환경의 글로벌 영향으로 모든 암의 평균이 동일한 방식으로 변화하는 상황에서, 고정된 전체 예산을 기반으로 유일한 최상의 암을 식별하는 것을 목표로 합니다.

- **Technical Details**: 논문에서 제안하는 새로운 선택 정책은 환경의 글로벌 변화에 대처할 수 있도록 일관성과 강건성을 갖추고 있으며, LinLUCB라는 할당 정책을 소개합니다. 이 정책은 모든 암에서의 글로벌 시프트 정보를 활용하여 유용한 통계치를 설계합니다.

- **Performance Highlights**: 실험 결과, 제안된 정책이 기존의 다른 방법들에 비해 유의미한 개선을 보였습니다. 특히, 글로벌 환경 변화에 효과적으로 대응하며 더 나은 성능을 보였습니다.



### RuleAlign: Making Large Language Models Better Physicians with Diagnostic Rule Alignmen (https://arxiv.org/abs/2408.12579)
Comments:
          Ongoing work

- **What's New**: 본 논문에서는 의료 분야에서 Large Language Models (LLMs), 예를 들어 GPT-4, MedPaLM-2, Med-Gemini의 성능을 인간 전문가와 비교하고, 전문가와 유사하게 진단을 내리는 데 필요한 문제를 해결하기 위한 RuleAlign 프레임워크를 소개합니다.

- **Technical Details**: RuleAlign 프레임워크는 규칙 기반 진단 규칙에 LLM을 정렬하기 위해 설계되었습니다. 이를 위해 환자와 의사 간의 규칙 기반 대화를 포함하는 의료 대화 데이터셋을 개발하였으며, 선호 학습(preference learning)을 통한 정렬 학습(alignment learning) 접근 방식을 설계하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식이 효과적임을 입증하였습니다. 우리의 연구가 LLMs가 AI 의사로서의 잠재력을 탐구하는 데 영감을 줄 수 있기를 바랍니다.



### A Percolation Model of Emergence: Analyzing Transformers Trained on a Formal Languag (https://arxiv.org/abs/2408.12578)
Comments:
          Preprint

- **What's New**: 이 논문에서는 신경망의 'emergent' (출현하는) 능력의 원인을 규명하기 위해 공학적 개념과 실험 시스템을 제안합니다.

- **Technical Details**: 본 연구는 특정 구조의 습득이 특정 작업에 대한 성능 급성장을 초래한다고 정의하고, 문맥에 민감한 형식 언어를 기반으로 한 실험 시스템을 통해 이를 검증합니다. 특히, Transformers 모델이 해당 언어의 문법과 구조를 학습할 때 성능이 급격히 향상됨을 확인하였습니다.

- **Performance Highlights**: 실험 결과, 주어진 형식 언어에서 문맥을 이해하고 구조를 학습한 후, 더욱 좁은 작업의 성능이 불연속적으로 증가하는 현상을 관찰하였습니다. 이는 bipartite graph에서의 침투(percolation) 과정과 유사한 학습 동역학을 가집니다.



### Enhanced Parking Perception by Multi-Task Fisheye Cross-view Transformers (https://arxiv.org/abs/2408.12575)
Comments:
          26th Irish Machine Vision and Image Processing Conference, Data-Driven Autonomy Workshop (matching camera-ready version)

- **What's New**: 본 논문은 Multi-Task Fisheye Cross View Transformers (MT F-CVT)라는 새로운 주차 인식 알고리즘을 소개합니다. 이 알고리즘은 4개의 fisheye Surround-view Camera System (SVCS)을 활용하여 세밀한 Bird-Eye View (BEV) 기능 맵을 생성하며, 이는 전통적인 주차 슬롯 인식 알고리즘의 오류와 제약 상황을 극복합니다.

- **Technical Details**: MT F-CVT는 segmentation decoder와 Polygon-Yolo 기반의 객체 탐지 디코더를 이용하여 주차 슬롯 및 차량을 인식합니다. 이 알고리즘은 LiDAR를 이용하여 주어진 데이터에 레이블을 붙이고, 25m x 25m 영역 내에서 평균 20 cm 오차로 물체의 위치를 파악합니다. 또한, MT F-CVT는 다양한 차량과 카메라 설정에서 강력한 일반화 능력을 보여줍니다.

- **Performance Highlights**: MT F-CVT의 큰 모델은 F-1 스코어 0.89를 달성하며, 작은 모델은 Nvidia Jetson Orin 임베디드 보드에서 16 fps로 작동하며 유사한 탐지 결과를 보입니다. 이는 주차 인식 저하 없이 실시간 실행 속도를 유지합니다.



### ssProp: Energy-Efficient Training for Convolutional Neural Networks with Scheduled Sparse Back Propagation (https://arxiv.org/abs/2408.12561)
Comments:
          Under review

- **What's New**: 최근 딥러닝 분야에서 생성 모델링, 특히 대규모 언어 모델과 확률적 확산 모델이 크게 발전했습니다. 하지만 이러한 모델 훈련은 수조 petaFLOPs를 요구하며, 이는 막대한 에너지 소비와 탄소 발자국 문제를 야기합니다. 본 연구에서는 에너지 효율적인 CNN 모듈을 제안하여 딥러닝 아키텍처에 통합할 수 있는 방법을 제시합니다.

- **Technical Details**: 제안하는 방법은 역전파 과정에서 채널 기반 희소성을 활용하여 연산을 40% 줄입니다. 이전의 방법들과 달리 이 방법은 일반적인 하드웨어와 호환되며, PyTorch에 쉽게 통합할 수 있습니다. 특히, Dropout 기법과 함께 사용하여 모델 성능을 향상시킵니다.

- **Performance Highlights**: 실험 결과, 제안하는 방법은 다양한 데이터셋과 태스크에 대해 일반화 가능하며, 이미지 분류 및 생성 작업에서 성능을 개선함을 보여주었습니다. 연구 및 개발 단계에서 상당한 에너지 절약을 가능하게 합니다.



### Data Quality Antipatterns for Software Analytics (https://arxiv.org/abs/2408.12560)
- **What's New**: 본 연구는 소프트웨어 분석에서 기계 학습(Machine Learning, ML) 데이터 품질의 안티패턴(antipatterns)에 대한 분류 체계를 개발하고 이들이 소프트웨어 결함 예측(Software Defect Prediction, SDP) 모델의 성능에 미치는 영향을 평가합니다. 특히, 데이터 정리 순서가 모델 성능에 미치는 영향을 분석하였습니다.

- **Technical Details**: 연구에서 식별된 ML-specific 데이터 품질 안티패턴의 주요 유형으로는 다음과 같은 8개의 유형과 14개의 하위 유형이 있습니다: Class Overlap, Tailed Distributions 등. 데이터 정리 순서가 ML 모델 성능에 미치는 영향이 명확하게 나타났으며, 특히 신경망(neural networks)이 로지스틱 회귀(logistic regression)와 같은 단순한 모델보다 데이터 정리 순서의 변화에 더 강한 저항력을 보였습니다.

- **Performance Highlights**: 연구 결과, 90% 이상의 안티패턴이 행(row) 및 열(column) 수준에서 중복되어 데이터 정리의 우선순위를 복잡하게 만들고 과도한 데이터 삭제의 위험을 초래합니다. 다섯 가지 안티패턴은 다른 안티패턴을 정리했을 때 통계적으로 유의미한 성능 관련 지표와 상관관계를 보였으며, 다양한 안티패턴이 적용된 모델 간 해석의 일관성이 중간 수준으로 나타났습니다.



### Modeling Time-Variant Responses of Optical Compressors with Selective State Space Models (https://arxiv.org/abs/2408.12549)
- **What's New**: 이 논문은 Selective State Space (S6) 모델을 사용하여 optical dynamic range compressor를 모델링하는 심층 신경망(deep neural network) 기반 방법을 제안합니다. 이 접근 방식은 입력 오디오를 인코딩하기 위해 Selective State Space 블록을 사용하며, 이전의 recurrent layer 기반 방법보다 우수한 성능을 발휘합니다.

- **Technical Details**: 제안된 접근 방식은 Feature-wise Linear Modulation과 Gated Linear Units을 통합하여 네트워크를 동적으로 조정하며, 외부 파라미터에 따라 압축의 공격 및 릴리스 단계를 조절합니다. 이 구조는 낮은 대기 시간(low latency)과 실시간(real-time) 처리에 적합합니다. 이 방법은 TubeTech CL 1B 및 Teletronix LA-2A와 같은 아날로그 광학 압축기에서 검증되었습니다.

- **Performance Highlights**: 제안된 블랙박스 모델링 방법은 훈련 동안 본 적이 있는 설정과 없는 설정 모두에 대해 압축 프로세스를 정확하게 에뮬레이션하여 다른 최첨단 모델들과 비교했을 때 모든 모델보다 높은 성능을 보였습니다. 또한, 제안된 모델의 정확도는 데이터셋의 제어 파라미터의 샘플링 밀도와 상관관계를 보이며, 빠른 공격과 느린 릴리스 설정이 에뮬레이션하기 가장 어려운 설정으로 확인되었습니다.



### Automatic Organ and Pan-cancer Segmentation in Abdomen CT: the FLARE 2023 Challeng (https://arxiv.org/abs/2408.12534)
Comments:
          MICCAI 2024 FLARE Challenge Summary

- **What's New**: 본 논문은 복부 Computed Tomography (CT) 스캔에서 장기 및 암 세분화에 대한 첫 번째 국제 대회를 개최하고, 4650개의 다양한 암 유형의 CT 스캔을 포함하는 대규모 데이터셋을 제공합니다.

- **Technical Details**: 이 연구에서는 13개의 복부 장기와 하나의 일반 병변 클래스를 포함한 세분화 태스크를 제시하고, 부분적으로 레이블이 지정된 학습(task)으로 설계되었습니다. 데이터셋은 50개 의료 센터에서 수집된 CT 스캔으로 구성되며, 새로운 방법론을 통해 평균 Dice Similarity Coefficient (DSC) 스코어가 장기에서 92.3% 및 병변에서 64.9%를 달성했습니다.

- **Performance Highlights**: 우승 팀은 deep learning 기반의 cascaded framework를 사용하여, 평균 DSC 점수에서 92.3% 및 64.9% 점수를 기록하며 기존의 state-of-the-art를 초월했습니다. 이 방식은 소비자 데스크탑에서도 4GB 미만의 GPU 메모리로 실행 가능하며, 평균 런타임은 8.58초로 나타났습니다.



### PCGRL+: Scaling, Control and Generalization in Reinforcement Learning Level Generators (https://arxiv.org/abs/2408.12525)
Comments:
          8 pages, 7 figures, 6 tables. Published at IEEE Conference on Games, 2024

- **What's New**: 최근 게임 콘텐츠 생성에 대한 연구에서 Procedural Content Generation via Reinforcement Learning (PCGRL)의 새로운 변형이 제안되었습니다. 이 연구에서는 Jax 프로그래밍 도구를 활용하여 RL (Reinforcement Learning) 에이전트를 훈련하는 과정을 최적화하였습니다. 이를 통해 환경 시뮬레이션 속도가 획기적으로 향상되었습니다.

- **Technical Details**: PCGRL 과정에서 두 가지 주요 혁신이 도입되었습니다: 훈련 중 레벨 크기 임의화와 주요 게임 타일의 고정된 위치(‘pinpoints’) 설정입니다. 이러한 방식은 과적합(overfitting)을 줄이는 데 기여합니다. 또한, 관찰 공간의 크기를 체계적으로 조정하여 일반화(generalization) 및 확장성(scalability)을 개선하는 방법을 검토하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 새로운 모델은 10억 타임스텝 후에도 일반적인 레벨 크기에 대해 더 robust한 디자인 전략을 학습하였으며, 관찰 창이 더 작을수록 새로운 레벨 크기에 대한 일반화 능력이 향상된 것으로 나타났습니다. 총 훈련 속도가 평균 15배 향상되었고, 이는 긴 훈련 시간에도 유리한 조건을 제공합니다.



### Advanced atom-level representations for protein flexibility prediction utilizing graph neural networks (https://arxiv.org/abs/2408.12519)
- **What's New**: 이 연구는 그래프 신경망(GNNs)을 사용하여 단백질의 원자 수준에서의 표현을 학습하고 B-팩터(B-factor)를 예측하는 새로운 접근 방식을 제안합니다. 이는 기존의 잔기 수준(residue level) 접근 방식을 넘어서고, 단백질의 더 미세한 원자 상호작용을 좀 더 정확하게 설명합니다.

- **Technical Details**: Meta-GNN 모델을 사용하여 단백질의 3D 구조를 기반으로 B-팩터를 예측하였으며, 4,000개 이상의 단백질(1700만 원자 포함)을 대상으로 Pearson 상관계수 0.71을 기록했습니다. GNN 아키텍처가 원자 수준의 표현 학습에 효과적이었음을 보여줍니다.

- **Performance Highlights**: Meta-GNN은 기존의 방법들과 비교하여 B-팩터 예측에서 유의미한 개선을 보였으며, 단백질 유연성 예측을 위한 새로운 가능성을 열었습니다.



### The Russian-focused embedders' exploration: ruMTEB benchmark and Russian embedding model design (https://arxiv.org/abs/2408.12503)
- **What's New**: 이번 논문에서는 러시아어에 특화된 새로운 텍스트 임베딩 모델인 ru-en-RoSBERTa와 MTEB의 러시아어 버전인 ruMTEB 벤치마크를 소개합니다. 이 벤치마크는 23개의 텍스트 임베딩 작업을 포함하고 있으며, 이 중 17개는 새로운 작업입니다.

- **Technical Details**: 루-en-RoSBERTa 모델은 러시아어의 텍스트 임베딩 작업을 위해 설계된 모델로, 다국어 임베딩에서 지식 전이(knowledge transfer)를 가능하게 합니다. 또한, ruMTEB 벤치마크는 의미적 텍스트 유사성(semantic textual similarity), 텍스트 분류(text classification), 재순위(re-ranking), 정보 검색(information retrieval) 등 7개의 과제를 포함하고 있습니다.

- **Performance Highlights**: 루-en-RoSBERTa는 최신 러시아어 모델들과 비교했을 때 경합할 수 있는 성능을 보여줍니다. 이 연구의 결과는 러시아어에 대한 현대적이고 효과적인 텍스트 임베딩 방법을 제공함으로써 정보 검색 및 텍스트 유사성 평가 등의 다양한 NLP 작업에 기여할 것으로 기대됩니다.



### GenderCARE: A Comprehensive Framework for Assessing and Reducing Gender Bias in Large Language Models (https://arxiv.org/abs/2408.12494)
- **What's New**: 본 연구는 GenderCARE라는 포괄적인 프레임워크를 도입하여 대형 언어 모델(LLM)에서 성별 편향을 정량화하고 완화하는 혁신적인 기준, 평가 방법 및 감소 기술을 제안합니다.

- **Technical Details**: GenderCARE 프레임워크는 성평등 기준(Criteria for Gender Equality Benchmarks), LLM의 성별 편향 평가(Assessment of Gender Bias in LLMs), LLM의 성별 편향 감소(Reduction of Gender Bias in LLMs), 그리고 평가 메트릭스(Evaluation metrics)로 구성되어 있습니다. GenderPair라는 새로운 쌍 기반 벤치마크를 개발하여 성별 편향 과정을 포괄적으로 평가하고, 트랜스젠더 및 비바이너리 그룹을 포함한 다양한 성별 정체성을 반영합니다.

- **Performance Highlights**: 본 연구의 신뢰성 있는 실험 결과는 17개의 LLM에서 90% 이상의 성별 편향 감소를 보여주고 있으며, 여러 주요 언어 작업에서의 변동성이 2% 이하로 유지되고 있습니다. 이로써 GenderCARE가 LLM에서 공정성과 형평성을 달성하는 데 기여할 것으로 기대됩니다.



### Not All Samples Should Be Utilized Equally: Towards Understanding and Improving Dataset Distillation (https://arxiv.org/abs/2408.12483)
- **What's New**: 이 논문에서는 Dataset Distillation(DD) 분야의 이론적 탐색을 처음으로 시도하며, 샘플 난이도를 통해 다양한 매칭 기반 DD 방법을 이해하려고 합니다. 샘플 난이도와 데이터 품질 간의 관계를 강조하고, Sample Difficulty Correction(SDC) 방법을 제안하여 학습 과정에서 더 쉬운 샘플에 집중하도록 합니다.

- **Technical Details**: 샘플 난이도는 gradient norm으로 측정되며, GM(Gradient Matching) 기반 방법은 어려운 샘플을 배우는 반면 TM(Trajectory Matching) 기반 방법은 상대적으로 쉬운 샘플을 선호함을 발견했습니다. 논문에서는 neural scaling laws를 데이터 프루닝 이론에서 DD로 확장하여 결과를 설명합니다. SDC는 기존의 매칭 기반 방법에 최소한의 코드 수정으로 통합 가능하도록 설계되었습니다.

- **Performance Highlights**: SDC를 추가함으로써 7개의 DD 방법(DC, DSA, DSAC, MTT, FTD, TESLA, DATM)과 6개의 데이터셋(MNIST, FashionMNIST, SVHN, CIFAR-10, CIFAR-100, Tiny ImageNet)에서 더 높은 품질의 증류된 데이터셋을 생성할 수 있음을 보였습니다.



### Predicting Solar Energy Generation with Machine Learning based on AQI and Weather Features (https://arxiv.org/abs/2408.12476)
Comments:
          10 pages, 11 figures

- **What's New**: 이번 연구에서는 정확한 태양광 에너지 예측 모델의 필요성을 강조하며, 공기 질 지수(Air Quality Index, AQI)와 날씨 특성이 태양광 발전에 미치는 영향을 조사했습니다. 특히, 고급 머신러닝(Machine Learning) 및 딥러닝(Deep Learning) 기법을 활용하여 시간 시계열(time series) 모델링을 진행했습니다.

- **Technical Details**: 본 연구에서는 파워 변환 정규화(power transform normalization)와 제로 인플레이트 모델링(zero-inflated modeling)을 포함한 다양한 방법론을 제안합니다. 여러 머신러닝 알고리즘과 Conv2D LSTM(Long Short-Term Memory) 모델을 적용하여 높은 정확도의 예측 결과를 도출하였습니다.

- **Performance Highlights**: Conv2D LSTM 모델을 통해 0.9691의 $R^2$ 점수와 0.18 MAE(Mean Absolute Error), 0.10 RMSE(Root Mean Square Error)를 달성하며, 공기 질 지수와 날씨 특성이 예측 정확도를 현저히 향상시킨다는 것을 보여주었습니다.



### WCEbleedGen: A wireless capsule endoscopy dataset and its benchmarking for automatic bleeding classification, detection, and segmentation (https://arxiv.org/abs/2408.12466)
- **What's New**: 이 연구에서는 자동 분류(classification), 탐지(detection), 및 세분화(segmentation)를 위한 새로운 의료 주석이 있는 무선 캡슐 내시경(WCE) 데이터셋인 WCEbleedGen을 개발하였다. 이 데이터셋은 총 2,618개의 출혈 및 비출혈 프레임으로 구성되어 있으며, 깊은 학습 모델을 통해 평가되었다.

- **Technical Details**: WCEbleedGen 데이터셋은 고해상도 프레임으로 이루어져 있으며, 출혈과 비출혈 클래스를 균형 있게 포함하고 있다. 데이터셋에는 클래스 레이블(class labels), 수동 생성 이진 마스크(binary masks), 및 정확한 경계 상자(bounding boxes)가 포함되어 있어 고급 분석을 위한 강력한 자원이다. 평가에는 9개의 분류 모델, 3개의 탐지 모델, 및 3개의 세분화 모델이 사용되었다.

- **Performance Highlights**: 비주얼 기하학 그룹(VGG) 19, YOLOv8n, Linknet이 각각 자동 분류, 탐지, 세분화 기반 평가에서 최고의 성과를 기록하였다. 이 연구는 WCE 비디오 해석을 위한 자동 출혈 진단의 중요성을 강조하며, 이 데이터셋은 WCE에서의 자동 출혈 진단을 위한 혁신적인 솔루션 개발에 기여할 것이다.



### Relaxed Rotational Equivariance via $G$-Biases in Vision (https://arxiv.org/abs/2408.12454)
- **What's New**: 이번 논문에서는 기존의 Group Equivariant Convolution(GConv)이 직렬 회전 대칭을 효과적으로 처리할 수 있지만, 현실 세계의 데이터에서 흔히 발생하는 회전 대칭 파괴(rotational symmetry-breaking)에 적응하는 데 어려움을 겪는 문제를 해결하기 위한 새로운 접근법을 제안합니다.

- **Technical Details**: 우리는 GConv에서 엄격한 그룹 제약을 깨기 위해 $G$-Biases라는 학습 가능한 바이어스를 도입하여 Relaxed Rotational Equivariant Convolution(RREConv)을 구현합니다. RREConv는 기존의 GConv보다 더 적은 매개변수로 높은 성능을 보이며, 다양한 모델에서 간편하게 사용할 수 있는 플러그 앤 플레이 모듈로 설계되었습니다.

- **Performance Highlights**: RREConv 기반 방법이 자연 이미지 데이터셋에서 분류 및 탐지 작업을 수행할 때 기존의 GConv 기반 방법들과 비교하여 우수한 성능을 달성했음을 실험을 통해 입증했습니다.



### A Riemannian Approach for Spatiotemporal Analysis and Generation of 4D Tree-shaped Structures (https://arxiv.org/abs/2408.12443)
- **What's New**: 본 논문에서는 시간에 따라 변형되는 나무 모양의 3D 객체, 즉 4D 나무 구조의 공간-시간(shape variability) 변동성을 모델링하고 분석하기 위한 최초의 포괄적인 접근 방식을 제안합니다. 중요한 기여는 Square Root Velocity Function Trees (SRVFT)를 사용하여 나무 모양의 3D 구조를 표현하여 4D 구조를 시간-매개변수화된 궤적으로 변환하는 것입니다.

- **Technical Details**: 이 논문에서는 SRVF(Square Root Velocity Functions) 공간 내에서의 공간 등록 문제를 해결하여 나무 모양의 4D 구조를 궤적으로 표현합니다. 이러한 접근 방식을 통해 4D 나무 모양을 모델링 및 분석하는 문제를 SRVF 공간 내의 탄성 궤적(elastic trajectories) 분석 문제로 감소시킵니다. 연구에서는 Riemannian metric과 계산 도구를 제안하여 빠르고 정확한 공간-시간 등록(spatiotemporal registration) 및 기하학적 경로(geodesics) 계산을 수행합니다.

- **Performance Highlights**: 제안된 도구는 실제 4D 식물 데이터(예: 성장하는 토마토 및 옥수수 식물)를 사용하여 효율성을 검증하였습니다. 이 프레임워크는 통계 모델을 활용하여 공간-시간 변동성을 모델링하고, 예제 집합으로부터 새로운 4D 나무 구조를 생성하는 기능을 통합합니다.



### Enhanced Infield Agriculture with Interpretable Machine Learning Approaches for Crop Classification (https://arxiv.org/abs/2408.12426)
- **What's New**: 이번 연구는 농업 분야에서 AI를 통한 이미지 분류 기법의 발전을 다루고 있습니다. 전통적인 머신러닝 기법에서 최첨단 foundation 모델까지, 다양한 접근 방식을 비교하고 각 모델의 Explainable AI 기능을 평가하여 모델의 해석 가능성을 높였습니다.

- **Technical Details**: 연구에서는 SIFT, ORB, Color Histogram과 같은 핸드크래프트(feature extraction) 기법을 활용한 전통적인 ML 알고리즘과 커스텀 CNN, 전이 학습(Transfer Learning) 기법을 적용한 다양한 사전 훈련된 DL 아키텍처 (EfficientNetV2, ResNet152V2, Xception, Inception-ResNetV2, MobileNetV3)에 대한 비교가 있었습니다. Xception 모델이 98%의 정확도로 가장 높은 성능을 보였으며, LIME, SHAP, Grad-CAM을 사용하여 모델의 해석 가능성을 확보했습니다.

- **Performance Highlights**: Xception 모델은 80.03 MB의 모델 크기와 0.0633초의 예측 시간을 기록하여 다른 모든 모델보다 우수한 일반화 성능을 보여주었습니다. 연구는 또한 AI를 통한 농업 관리 전략의 개선 및 효율성을 위한 해석 가능성의 중요성을 강조합니다.



### Multi-Knowledge Fusion Network for Time Series Representation Learning (https://arxiv.org/abs/2408.12423)
Comments:
          Paper accepted at ML4IoT Workshop, International Conference on Learning Representations(ICLR) 2023

- **What's New**: 본 논문에서는 복잡한 동적 시스템의 행동 예측을 위한 새로운 접근 방식을 제안합니다. 구체적으로, 멀티변량 시계열(MTS) 데이터의 위계적 상관관계를 추론하고, 도메인 지식과 은닉된 구조적 관계를 통합하여 MTS 예측의 정확도를 향상하는 혼합 아키텍처를 소개합니다.

- **Technical Details**: 제안하는 모델은 누적된 시간 및 공간적 의존성을 학습하기 위해 명시적 그래프(Explicit Graph)와 암시적 하이퍼그래프(Implicit Hypergraph)로 구성됩니다. 각각의 컴포넌트가 서로 상호작용하여 복잡한 비선형 동적 관계를 포착하며, 예측 불확실성을 모델링하여 의사결정 지원을 강화합니다. 이 모델은 공간적 메시지 전달 체계를 사용한 후 시간 인코딩 단계가 수행되는 공간-그 다음-시간(Space-Then-Time, STT) 방식을 적용합니다.

- **Performance Highlights**: 제안하는 EIKF-Net 아키텍처는 여러 벤치마크 데이터셋에서 뛰어난 성능을 보였으며, 기존의 최신 예측 방법들을 현저히 능가하는 결과를 도출하였습니다. 특히 예측 불확실성을 효과적으로 추정하여 더 나은 위험 평가와 의사 결정을 지원할 수 있음을 입증하였습니다.



### 4D Diffusion for Dynamic Protein Structure Prediction with Reference Guided Motion Alignmen (https://arxiv.org/abs/2408.12419)
- **What's New**: 이 연구는 분자 동역학(Molecular Dynamics, MD) 시뮬레이션 데이터를 통합하여 동적 단백질 구조를 학습하는 혁신적인 4D diffusion 모델을 도입합니다. 이 모델은 단백질 구조의 동적 특성을 동시에 예측할 수 있는 최초의 diffusion 기반 모델입니다.

- **Technical Details**: 제안된 모델은 (1) 아토믹 그룹, 사이드 체인 이형각 예측을 활용하여 백본과 사이드 체인을 포함하는 동적 단백질 구조를 생성할 수 있는 통합 diffusion 모델을 특징으로 하며, (2) 초기 3D 단백질 구조의 잠재 임베딩을 통합하여 구조적 일관성을 높이는 레퍼런스 네트워크를 포함하고, (3) 여러 시간 단계 간의 시간적 구조 일관성을 개선하기 위한 모션 정렬 모듈을 제공합니다.

- **Performance Highlights**: 모델은 최대 256개의 아미노산을 포함하는 3D 단백질 구조를 32시간 동안 예측하며, 안정 상태에서의 국소적 유연성과 중요한 형태 변화 모두를 효과적으로 캡처함으로써 높은 정확도를 보였습니다.



### CODE: Confident Ordinary Differential Editing (https://arxiv.org/abs/2408.12418)
- **What's New**: 새로운 연구는 Confident Ordinary Differential Editing (CODE)라는 접근 방식을 소개합니다. CODE는 Out-of-Distribution (OoD) 이미지를 효과적으로 처리하여 이미지 생성 및 편집을 쉽게 할 수 있도록 돕습니다.

- **Technical Details**: CODE는 이미지를 보강하기 위해 확률 흐름 Ordinary Differential Equation (ODE)을 따르는 점수 기반 업데이트를 사용합니다. 이 방법은 추가적인 학습이나 수작업 모듈 없이, 노이즈가 있는 이미지에 대해 고유한 복원 과정을 통해 작동합니다.

- **Performance Highlights**: 실험 결과, CODE는 SDEdit보다 더 나은 사실감과 충실도를 보여 주었으며, 특히 심각하게 손상된 입력의 경우에 더욱 뛰어난 성능을 보였습니다.



### Dynamic PDB: A New Dataset and a SE(3) Model Extension by Integrating Dynamic Behaviors and Physical Properties in Protein Structures (https://arxiv.org/abs/2408.12413)
- **What's New**: 연구는 기존의 정적인 3D 단백질 구조 데이터베이스에 동적인 데이터와 추가적인 물리적 특성을 통합해 새로운 데이터 세트인 Dynamic PDB를 제안합니다. 이 데이터 세트는 약 12.6K개의 단백질을 포함하고 있으며, 각 단백질은 1 마이크로초의 분자 동역학(MD) 시뮬레이션을 거쳐 형태 변화가 기록됩니다.

- **Technical Details**: Dynamic PDB 데이터 세트는 원자 속도와 힘, 단백질의 위치 및 운동 에너지 등 물리적 특성을 1 피코초 간격으로 기록하며, 이 데이터 세트를 활용해 최신의 경로 예측 방법을 평가합니다. SE(3) 확산 모델을 기반으로 하여 이러한 물리적 특성을 경로 예측 과정에 통합하는 방안을 개발하였습니다.

- **Performance Highlights**: 초기 결과는 제안된 물리적 특성을 고려할 때 MAE와 RMSD 지표에서 정확도가 향상됨을 보여줍니다. 더욱이, 밀도 높은 시간 샘플링 간격과 확장된 샘플링 기간이 단백질의 동역학 이해에 유익함을 입증하였습니다.



### Multi-Source Knowledge-Based Hybrid Neural Framework for Time Series Representation Learning (https://arxiv.org/abs/2408.12409)
Comments:
          Paper is accepted at Knowledge-Based Compositional Generalization Workshop, International Joint Conferences on Artificial Intelligence(IJCAI-23)

- **What's New**: 이번 논문은 다변량 시계열(multivariate time series, MTS) 데이터를 예측하기 위한 하이브리드 아키텍처를 제안하고, 이 아키텍처는 도메인 전문가의 지식(domain-specific knowledge)과 MTS 데이터의 관계 구조(relational structure) 간의 암시적 지식(implicit knowledge)을 결합하여 예측 정확도를 높이고 예측 불확실성(uncertainty)을 모델링합니다.

- **Technical Details**: 제안된 MKH-Net은 공간과 시간의 추론(temporal inference) 구성 요소로 이루어져 있으며, ‘implicit hypergraph’, ‘explicit subgraph’, ‘dual-hypergraph’ 방법을 결합하여 MTS 데이터 내의 복잡한 상관관계를 모델링합니다. MKH-Net은 다양한 격리형 의존성(dependencies)을 포착하는 동시에, 다중 지식 표현(multi-knowledge representations)을 조합하는 게이팅 메커니즘(gating mechanism)을 활용합니다.

- **Performance Highlights**: MKH-Net은 여러 벤치마크 데이터셋에서 최첨단 예측 방법들을 초과하는 탁월한 결과를 보여주었으며, 복잡한 비선형 동역학(non-linear dynamics)을 효과적으로 모델링할 수 있습니다. 또한, 다중 시간 지평선(multi-horizon) 예측에 대한 시간 가변적 불확실성을 모델링하는 능력을 가지고 있습니다.



### Multi-Style Facial Sketch Synthesis through Masked Generative Modeling (https://arxiv.org/abs/2408.12400)
- **What's New**: 본 연구에서는 주어진 얼굴 사진으로부터 스케치 초상화를 생성하는 얼굴 스케치 합성(FSS) 모델을 제안합니다. 개발한 경량화된 end-to-end 합성 모델은 추가 입력 없이 효율적으로 이미지를 다양한 스타일의 스케치로 변환합니다.

- **Technical Details**: 제안된 모델은 반지도 학습(semi-supervised learning)을 통합하여 데이터 부족 문제를 해결하며, 특징 추출 모듈(feature extraction module) 및 스타일 임베딩(style embeddings)을 사용해 생성적 변환기(generative transformer)를 조정합니다. 모델은 VQ-GAN의 토큰 공간(token space) 내에서 생성을 제한하며, 마스킹된 이미지 토큰의 반복 예측 과정에서 스케치로서의 얼굴 특징을 정확히 유지합니다.

- **Performance Highlights**: 폭넓은 실험 결과, 본 연구의 방법론은 기존 알고리즘 대비 여러 벤치마크에서 일관되게 우수한 성능을 보여주었으며, 생성된 스케치의 품질과 구조적 유사도 측면에서 뚜렷한 차이를 입증하였습니다.



### Cell-ontology guided transcriptome foundation mod (https://arxiv.org/abs/2408.12373)
Comments:
          All anonymous reviewers' constructive suggestions are appreciated. The next version will be updated soon

- **What's New**: 본 논문에서는 Cell-ontology 정보를 활용하여 단일 세포를 대상으로 하는 새로운 전사체 기반 모델 scCello를 제안합니다. 이 모델은 세포 유형의 특성과 세포 간의 계층적 관계를 학습하여 전통적인 TFMs의 한계를 극복하고자 합니다.

- **Technical Details**: scCello는 세 가지 수준의 학습 목표를 설정합니다: (1) 마스킹된 유전자 예측 손실로 유전자 공발현 패턴을 학습하고, (2) 세포 유형 일관성을 촉진하는 손실을 통해 같은 세포 유형의 표현이 밀접하게 형성되도록 하고, (3) 세포 계통에 따른 관계 정렬 손실을 통해 세포의 표현 학습을 지원합니다.

- **Performance Highlights**: scCello는 2200만 개의 세포에 대해 사전 훈련되었으며, 세포 유형 식별, 세포 유형 특정 마커 예측 및 암 약물 반응 예측 등 생물학적으로 중요한 작업에서 기존 모델들과의 성능 경쟁력을 보여주었습니다.



### SAM-SP: Self-Prompting Makes SAM Great Again (https://arxiv.org/abs/2408.12364)
Comments:
          Under Review

- **What's New**: 최근 발표된 Segment Anything Model (SAM)은 다양한 자연 이미지 데이터셋에서 제로샷(segmentation) 기술을 보이며 뛰어난 능력을 보여주고 있습니다. 하지만 SAM은 의료 이미지와 같은 특정 도메인에 적용될 때 성능 저하가 발생하는 문제가 있습니다. 본 논문에서는 이러한 문제를 해결하기 위한 새로운 자가 프롬프트(self-prompting) 기반의 파인튜닝 방법 SAM-SP를 소개합니다.

- **Technical Details**: SAM-SP는 모델의 이전 반복에서의 출력을 프롬프트로 활용하여 이후 반복을 이끌어내는 방식으로 자가 프롬프트 모듈을 사용합니다. 이를 통해 전문가가 제공하는 프롬프트의 필요성을 줄이고 SAM의 적용 가능성을 크게 확장할 수 있습니다. 또한, 자가 증류(self-distillation) 방법을 통합하여 자가 프롬프트 과정을 더욱 강화하고 있습니다.

- **Performance Highlights**: 다양한 도메인 특화 데이터셋에서 실시한 광범위한 실험을 통해, SAM-SP는 사용자 프롬프트 없이도 우수한 성능을 보이며, 기존의 최신 작업 특화(segmentation) 접근법들 및 기본 SAM과 비교하여 뛰어난 성능을 나타냅니다.



### Class-balanced Open-set Semi-supervised Object Detection for Medical Images (https://arxiv.org/abs/2408.12355)
- **What's New**: 이 연구에서는 의료 이미지에서의 반지도 객체 탐지 문제를 해결하기 위해 OOD(Out-Of-Distribution) 클래스를 포함한 비표시 데이터(unlabeled data)를 활용하는 새로운 방법론을 제시합니다.

- **Technical Details**: 제안된 방법론은 두 가지 주요 혁신을 포함합니다: 1) CCE(Category Control Embed) 모듈을 통해 데이터 세트 불균형을 해소하고, 2) OODFC(Out-of-Distribution Detection Fusion Classifier) 모듈을 통해 '알 수 없는(unknown)' 정보를 기본 가짜 레이블(pseudo-labels)에 통합합니다. 이 설정에서는 Mean-Teacher(미국의 평균 교사 방법)로 명명된 SSOD(Semi-Supervised Object Detection) 파이프라인을 기반으로 한 신경망 구조가 사용됩니다.

- **Performance Highlights**: 제안된 방법은 퍼블릭 Parasite 데이터셋에서 기존 SSOD 성능 기준보다 4.25 mAP(mean Average Precision)가 향상되었습니다.



### Fine-tuning Smaller Language Models for Question Answering over Financial Documents (https://arxiv.org/abs/2408.12337)
- **What's New**: 최근 연구에서는 대형 언어 모델(LLM)로부터 파생된 교육 예시를 통해 작은 언어 모델이 상당한 추론 능력을 획득할 수 있다는 점을 보여주었습니다. 본 논문에서는 금융 분야에 해당하는 질문 답변 문제를 다루며, 다단계 수치 추론을 요구하는 과제가 어떻게 해결될 수 있는지 분석합니다.

- **Technical Details**: 이 연구는 GPT-4를 교사 모델로 활용하여 소형 모델(3.5B, 14B parameters의 phi-3 변형, Mistral 7B, Orca-2)을 훈련시키고, Python 코드 생성을 통해 금융 추론을 수행합니다. 교사 모델이 생성한 코드는 질문 이해, 공식 식별 및 엔티티 추출 등 단계적으로 금융 추론을 정리합니다. 이렇게 생성된 코드는 외부 해석기에 의해 실행됩니다.

- **Performance Highlights**: 본 연구의 결과는 작은 언어 모델의 성능이 교사 모델과 유사해 질 수 있음을 보여줍니다. 정제된 개념 이해와 일관된 추론을 통해 작은 모델이 금융 데이터 형식에 적합하게 엔티티 추출 능력을 향상시킨 것을 입증하였습니다. 또한 상대적으로 작은 데이터셋을 사용하여 금융 추론 능력을 유도할 수 있다는 가설도 증명하였습니다.



### Enhanced Expressivity in Graph Neural Networks with Lanczos-Based Linear Constraints (https://arxiv.org/abs/2408.12334)
- **What's New**: 이번 연구에서는 인코딩된 유도 부분 그래프를 그래프 라플라시안 행렬의 고유 기저에 포함시키는 새로운 방법론을 제안하여 GNN(그래프 신경망)의 표현력을 향상시키고자 한다.

- **Technical Details**: 제안된 방법은 Learnable Lanczos with Linear Constraints (LLwLC)라는 알고리즘을 기반으로 하며, 두 가지 새로운 부분 그래프 추출 전략(정점 삭제된 부분 그래프 인코딩 및네우만 고유값 제약 적용)을 도입한다.

- **Performance Highlights**: LLwLC는 PubMed 및 OGBL-Vessel 데이터셋에서 기존 최첨단 방법에 비해 각각 20배와 10배의 속도 향상을 이루었으며, 5% 및 10%의 데이터만으로도 효과적이었다.



### Interactive DualChecker for Mitigating Hallucinations in Distilling Large Language Models (https://arxiv.org/abs/2408.12326)
- **What's New**: 본 논문에서는 LLMs의 허위 정보 문제를 완화하고 교육자 모델과 학생 모델 모두의 성능을 향상시키기 위한 새로운 프레임워크인 DualChecker를 도입합니다. 이 방법은 ContextAligner를 활용하여 사람의 라벨링 기준과 모델 출력을 정렬하고, 동적 체커 시스템을 통해 모델 상호작용을 개선합니다.

- **Technical Details**: DualChecker는 LLM을 사용하여 지식을 증류하는 혁신적인 인터랙티브 프레임워크입니다. ContextAligner가 모델 출력을 인간 라벨에 맞춰 조정하고, 인터랙티브 체커 시스템이 LLM 응답에서 신뢰 점수를 수집하여 낮은 신뢰를 보이는 경우 자세한 정보를 추가하여 일관성을 보장합니다.

- **Performance Highlights**: DualChecker는 실험에서 기존의 최첨단 방법들보다 월등히 우수한 성과를 나타내며, 교육자 모델의 F1 점수를 최대 17% 향상시키고 학생 모델은 10% 향상시키는 결과를 보였습니다. 특히, LLM 예측으로 미세 조정된 학생 모델은 실제 데이터로 미세 조정된 모델과 유사한 성능을 발휘합니다.



### Towards Deconfounded Image-Text Matching with Causal Inferenc (https://arxiv.org/abs/2408.12292)
Comments:
          ACM MM

- **What's New**: 이 논문에서는 이미지-텍스트 매칭 (image-text matching) 모델에서 발생하는 데이터셋 편향 (bias)을 해결하기 위해 새로운 접근 방식을 제안한다. 구체적으로는, 구조적 인과 모델 (Structural Causal Models, SCM)을 사용하여 내부 및 외부 요인이 어떻게 이미지-텍스트 매칭 성능에 부정적인 영향을 미치는지를 설명하고, 새로운 Deconfounded Causal Inference Network (DCIN)를 개발하여 편향의 영향을 최소화한다.

- **Technical Details**: 제안된 DCIN은 (1) 이미지와 텍스트 특성의 인코딩 단계에서 내부 및 외부 요인을 분해하고 통합하여 가짜 상관관계를 효과적으로 제거하며, (2) 외부 지식의 편향을 완화하기 위해 인과 추론 (causal inference)을 사용한다. 이 과정에서 데이터를 통해 크고 작은 요인들 사이의 관계를 학습할 수 있도록 구조적으로 설계되었다.

- **Performance Highlights**: Flickr30K와 MSCOCO 같은 두 가지 유명한 벤치마크 데이터셋에서 수행된 광범위한 실험 결과, 제안된 DCIN 방법이 기존 방법보다 우수한 성능을 보였음을 입증하였다.



### Developing vocal system impaired patient-aimed voice quality assessment approach using ASR representation-included multiple features (https://arxiv.org/abs/2408.12279)
Comments:
          Accepted by Interspeech 2024

- **What's New**: 본 논문은 임상 음성 처리에서의 딥러닝의 잠재력을 다루며, 음성 인식(Automatic Speech Recognition)과 자기 지도 학습(Self-Supervised Learning) 기법을 사용하여 제한적이고 불균형한 임상 데이터의 문제를 해결하려고 합니다. 특히, 음성 품질을 정량화하기 위한 새로운 접근법을 제안합니다.

- **Technical Details**: 제안된 모델은 두 부분으로 구성되어 있습니다: 특징 추출과 다운스트림 모듈입니다. 특징 추출은 ASR 표현, SSL 표현, 및 mel-spectrogram을 포함하며, 이들 특징을 GRBAS 스케일에 매핑하는 다운스트림 모듈을 갖추고 있습니다. 제안된 방법은 Whisper와 HuBERT 모델을 사용하여 특성을 얻고, 두 개의 LSTM 레이어를 사용하여 시퀀스를 처리합니다.

- **Performance Highlights**: PVQD 데이터세트에 대한 실험 결과, 예측 성능에서 각 지표(Grade, Breathy, and Asthenic)와의 상관관계가 0.8 이상으로 나타났으며, 평균 제곱 오차(MSE)는 0.5 이하로 우수한 성과를 보였습니다. STN-DBS 환자에 대한 음성 품질 예측에서도 개선된 정확도를 달성했습니다.



### Variance reduction of diffusion model's gradients with Taylor approximation-based control varia (https://arxiv.org/abs/2408.12270)
Comments:
          14 pages, ICML Structured Probabilistic Inference & Generative Modeling 2024

- **What's New**: 본 논문에서는 고차 테일러 다항식을 이용하여 Score-based 모델의 훈련 목표의 고변동성을 줄이는 새로운 방법인 control variate를 제안합니다. 이 접근법은 저차원 문제 설정에서 효과가 입증되었으며, 더 큰 문제에서도 그 영향을 연구합니다.

- **Technical Details**: Score-based 모델은 Denoising Score Matching을 통해 훈련되며, 이 과정에서 다양한 노이즈를 추가하여 손상된 데이터를 복구하는 데 초점을 맞춥니다. Control variate는 Monte Carlo 통합 문제와의 상관관계를 활용하여 분산을 줄입니다. 이 논문은 k-th 차 테일러 근사를 이용한 control variate의 일반화를 제안합니다.

- **Performance Highlights**: 실험 결과, 제안한 control variate 기법이 저차원 문제에서 효과적임을 보여주었으며, 고차원 문제에서도 그 효과를 조사했습니다. 테일러 기반 control variate의 제한 사항 또한 언급되었습니다.



### Toward the Evaluation of Large Language Models Considering Score Variance across Instruction Templates (https://arxiv.org/abs/2408.12263)
Comments:
          19 pages, 7 figures

- **What's New**: 이번 연구에서는 LLM (Large Language Model)의 NLU (Natural Language Understanding) 성능을 공정하게 평가하기 위한 새로운 데이터셋과 평가 메트릭인 Sharpe score를 제안합니다. 이는 템플릿 간 점수 변동성을 고려하여 LLM의 성능을 보다 정확하게 측정하는 데 중요합니다.

- **Technical Details**: 평가 방법론으로는, 다양한 instruction templates를 사용하여 LLM의 NLU 성능을 평가했습니다. 우리가 제안한 Sharpe score는 템플릿 간의 변동성을 반영하여 성능 평가를 보다 효과적으로 수행합니다. 이는 영어와 일본어에 대한 크로스-링구얼 (cross-lingual) 데이터셋을 포함하며, 특정 작문 포맷을 반영하도록 정규 표현식 (regular expressions)을 사용해 출력을 제어했습니다.

- **Performance Highlights**: 연구 결과, English와 Japanese LLM에 대한 종합 분석을 통해 템플릿 간의 높은 변동성이 LLM의 공정한 평가에 중요한 영향을 미친다는 것을 발견했습니다. 이에 따라, 새로운 평가 메트릭과 함께 다수의 템플릿을 기반으로 한 평가가 LLM의 NLU 성능을 보다 정확하게 드러낸다고 결론지었습니다.



### A Language-agnostic Model of Child Language Acquisition (https://arxiv.org/abs/2408.12254)
- **What's New**: 이 논문은 영어를 위해 설계된 최근의 의미적 부트스트래핑(child-language acquisition) 모델을 다시 구현하여 새로운 언어인 히브리어를 배우도록 훈련한 연구 결과를 다룹니다. 모델은 발화와 의미 표현의 쌍에서 학습하며, 구문(syntax)과 단어의 의미를 동시에 습득합니다.

- **Technical Details**: 모델은 CCG(combinatory categorial grammar)와 의미적 부트스트래핑 이론에 기반하여, 아동이 언어를 어떻게 배우는지를 시뮬레이션합니다. 데이터는 CHILDES 코퍼스에서 가져온 아동 지향 발화로 구성되며, 보편적 의존 주석을 논리적 형태로 변환하는 최신 방법이 적용되었습니다. 모델은 영어와 히브리어 두 언어에서 테스트되었습니다.

- **Performance Highlights**: 모델은 영어 구문과 의미의 중요한 특징을 성공적으로 학습했으며, 히브리어에서도 높은 정확도로 단어 순서(word order)와 단어 의미를 학습하였습니다. 히브리어의 합성어가 영어보다 더 풍부하여 학습이 느리고 덜 강건하다는 결과를 보였습니다.



### LLMs are not Zero-Shot Reasoners for Biomedical Information Extraction (https://arxiv.org/abs/2408.12249)
Comments:
          11 pages

- **What's New**: 이 논문은 Large Language Models (LLMs)의 의료 분야에서의 성능을 체계적으로 벤치마킹하여 의학적 분류(medical classification)와 Named Entity Recognition (NER) 작업에서의 성과를 평가합니다. 또한, LLM의 수행에 영향을 미치는 다양한 요인의 기여를 분석합니다.

- **Technical Details**: 연구는 BioMistral 및 Llama-2 모델을 포함한 여러 공개 LLM을 대상으로 하며, Chain-of-Thought (CoT), Self-Consistency 기반의 추론 및 Retrieval-Augmented Generation (RAG)을 사용하여 다양한 생물의학 데이터셋을 평가합니다. 연구 결과는 표준 프롬프트가 복잡한 기술보다 일관되게 더 나은 성능을 보임을 보여 주며, 의료 분야에서 CoT, self-consistency, RAG의 현재 적용의 한계를 드러냅니다.

- **Performance Highlights**: 이 연구는 LLMs가 '진정한' 제로샷(zero-shot) 설정에서 성능이 저하되지 않도록 다양한 지식 증강 기법을 조사하고, 매개변수(parametric) 지식 용량이 제로샷 환경에서 성능의 주요 원인임을 발견했습니다. 또한, 모델의 지식 활용에 대한 시스템적 접근이 필요함을 강조합니다.



### EvalYaks: Instruction Tuning Datasets and LoRA Fine-tuned Models for Automated Scoring of CEFR B2 Speaking Assessment Transcripts (https://arxiv.org/abs/2408.12226)
- **What's New**: 이 연구는 e-learning 환경에서 CEFR B2 영어 말하기 평가의 자동화를 목표로 하며, 대규모 언어 모델(LLMs)과 전문가 검증된 합성 대화 데이터셋를 활용하여 평가 정확도를 향상시킵니다.

- **Technical Details**: 연구팀은 CEFR B2 시험을 위해 Mistral Instruct 7B v0.2의 파라미터 효율적인 instruction tuning을 수행하여 EvalYaks라는 모델 패밀리를 개발했습니다. 이 모델들은 각기 다른 섹션의 성능을 평가하며, 단어 및 텍스트의 CEFR 수준을 식별하고 생성할 수 있습니다.

- **Performance Highlights**: EvalYaks 모델은 평균 96%의 허용 가능한 정확도를 달성했으며, 다른 모델보다 3배 성능이 우수한 것으로 나타났습니다. 이는 고품질 CEFR 정렬 평가 데이터를 사용하여 조정된 LLM이 효과적으로 B2 영어 말하기 평가를 수행할 수 있음을 보여 줍니다.



### Two-level deep domain decomposition method (https://arxiv.org/abs/2408.12198)
Comments:
          Preprint proceeding format

- **What's New**: 이 연구는 물리 기반 신경망(Physics-Informed Neural Networks, PINNs)을 활용하여 경계값 문제(Boundary Value Problems, BVP)를 해결하기 위한 두 단계의 Deep Domain Decomposition Method(Deep-DDM)를 제안합니다. 새로운 coasre-level network의 추가는 기존 단일 단계 방법에 비해 확장성과 수렴 속도를 향상시킵니다.

- **Technical Details**: 제안된 Deep-DDM은 Poisson 방정식에 대해 Dirichlet 경계 조건을 사용하여 테스트되었으며, 복잡한 부분 미분 방정식(Partial Differential Equations, PDE)을 해결하는 데 더 효과적인 접근 방식을 제공합니다. PINNs을 서브도메인 솔버로 사용하는 기존의 Schwarz 반복 기법을 사용하여, 두 단계의 수치를 구현합니다.

- **Performance Highlights**: Deep-DDM은 서브도메인의 수에 관계없이 효율적인 수렴성을 유지하면서 우수한 성능을 보여줍니다. 이는 전통적인 Schwarz 반복 기법과 같은 수렴 특성을 유지하면서도 수치적 확장성을 보장합니다.



### Reasoning Factual Knowledge in Structured Data with Large Language Models (https://arxiv.org/abs/2408.12188)
- **What's New**: 이 논문은 대형 언어 모델(Large Language Models, LLMs)이 구조화된 데이터에서 사실 지식을 추론하는 능력을 평가하기 위해 StructFact라는 벤치마크를 제안합니다. StructFact는 다양한 작업 및 도메인을 포함하는 8,340개의 사실 질문으로 구성되어 있습니다. 이 벤치마크는 LLMs가 구조적 사실로부터 사실 지식을 정확히 추론할 수 있는지 탐색할 수 있는 기회를 제공합니다.

- **Technical Details**: StructFact 벤치마크는 5개의 사실 작업(Arithmetic Calculation, Spatiotemporal Cognition, Multi-hop Reasoning, Composition Understanding, Combining Structured and Unstructured)에 대해 설계되었습니다. 이 작업들은 구조화된 데이터의 고유한 특성에 기반하여 LLMs의 추론 능력을 다양한 방식으로 분석할 수 있게 합니다. 각 질문은 세부 난이도 주석을 포함하여 해당 작업의 특정 초점에 따라 분류되었습니다.

- **Performance Highlights**: 구조화된 데이터로부터 사실 지식을 추론하는 LLMs의 능력에 대한 대규모 실험이 수행되었습니다. 실험 결과, 기존 LLMs는 구조화된 데이터에서 사실 지식을 정확하게 추론하는 데 한계를 보였으며, 이는 특히 의료 및 금융과 같은 고위험 도메인에서의 실용적 활용을 제한하는 요인입니다. StructFact는 이런 한계를 극복하고 LLMs의 실제 적용을 향상시키기 위한 중요한 도구로 자리잡을 것입니다.



### A Safe and Efficient Self-evolving Algorithm for Decision-making and Control of Autonomous Driving Systems (https://arxiv.org/abs/2408.12187)
- **What's New**: 이 논문은 자율주행 차량의 의사결정 및 제어 시스템의 안전성 문제와 낮은 학습 효율성을 해결하기 위해 하이브리드 Mechanism-Experience-Learning 접근 방식을 제안합니다. 이 접근 방식은 인간의 운전 경험을 비유하여 운전 경향을 정의하고 이를 통해 검색 공간을 줄여 효율적인 자기 진화를 실현합니다.

- **Technical Details**: 제안된 알고리즘은 전형적인 actor-critic 아키텍처를 기반으로 하여 최적화 문제로 통합적으로 해결됩니다. 주된 구성 요소로는 critic function approximator와 policy function approximator가 있으며, critic function approximator는 정책의 가치를 평가합니다. 정책 근사는 안전한 정책을 학습하고, 이를 통해 안전하고 신뢰할 수 있는 제어 동작을 생성합니다.

- **Performance Highlights**: 제안된 방법은 다양한 복잡한 시나리오에서 안전하고 합리적인 행동을 생성할 수 있으며, 자율주행 시스템의 성능을 개선합니다. 기존의 강화 학습 알고리즘보다 안전성과 효율성이 크게 향상되었으며, 훈련 과정은 충돌 없이 진행되며 현실 세계에서 훈련 시간은 10분 이내로 단축됩니다.



### Rank and Align: Towards Effective Source-free Graph Domain Adaptation (https://arxiv.org/abs/2408.12185)
Comments:
          Published in IJCAI2024

- **What's New**: 본 논문은 실제 세계에서의 개인 정보 보호 및 저장 문제로 인해 광범위한 소스 그래프에 접근하는 것이 어려운 상황에서 소스 데이터 없이 그래프 도메인 적응(source-free graph domain adaptation) 문제를 다룸. 여기서는 소스 그래프 대신 사전 학습된 그래프 모델을 사용하여 지식을 타겟 도메인으로 전이하는 방법을 제안한다.

- **Technical Details**: 제안된 방법, Rank and Align (RNA)는 스펙트럴 세리이제이션(Spectral Seriation) 기법을 활용하여 그래프 유사성을 순위별로 매기고, 하모닉 그래프와 비하모닉 그래프를 정렬하여 서브그래프를 추출하는 방식을 사용한다. RNA는 스펙트럴 클러스터링을 통해 도메인 전이를 감지하고, 적대적 엣지 샘플링 과정을 통해 도메인 불변의 서브그래프를 추출하여 GNN의 불변 학습을 돕는다.

- **Performance Highlights**: 여러 벤치마크 데이터셋에서의 광범위한 실험을 통해 RNA가 기존의 기법들과 비교할 때 우수한 성능을 보여준다는 점이 강조된다.



### Search-Based LLMs for Code Optimization (https://arxiv.org/abs/2408.12159)
Comments:
          Accepted by 2025 IEEE/ACM 47th International Conference on Software Engineering (ICSE'25)

- **What's New**: 이 논문에서는 기존 코드 최적화 방법이 지닌 한계를 극복하기 위해, 코드 최적화 작업을 탐색 관점에서 모델링하고, 반복적인 개선 및 최적화 방법 발견을 가능하게 하는 SBLLM(Search-Based LLMs)라는 새로운 프레임워크를 제안합니다.

- **Technical Details**: SBLLM은 LLM(대형 언어 모델)과 진화적 검색(evolutionary search)을 통합하며, 세 가지 주요 구성 요소로 이루어져 있습니다: 1) 실행 기반의 대표 샘플 선택(execution-based representative sample selection), 2) 적응형 최적화 패턴 검색(adaptive optimization pattern retrieval), 3) 유전자 연산자 기반의 사고 과정을 촉진하는 프롬프트(Genetic Operator-inspired chain-of-thought prompting). 이 방법은 복잡한 최적화 방법을 보다 효과적으로 통합하여 LLM이 더 개선된 최적화 코드를 생성하도록 돕습니다.

- **Performance Highlights**: SBLLM은 광범위한 벤치마크 데이터셋에서 실험을 통해 기존의 여러 프롬프트링 방법에 비해 코드 효율성을 8.75%에서 28.06%까지 향상시키는 데 성공하였습니다. Python 및 C++ 코드에 대한 성능 개선이 관찰되었으며, 해당 프레임워크는 다양한 LLM에서 우수한 성과를 보였습니다.



### Implicit Sentiment Analysis Based on Chain of Thought Prompting (https://arxiv.org/abs/2408.12157)
- **What's New**: 이 논문에서는 Chain of Thought (CoT) 개념에 영감을 받아 Sentiment Analysis of Thinking (SAoT) 프레임워크를 제안하였습니다. SAoT는 심리적 의견의 내재적 측면을 이해하고 정서의 극성을 추론하는 데 중점을 두고 있습니다.

- **Technical Details**: SAoT 프레임워크는 일반 상식(common sense)과 사고 체인 능력을 활용하여 텍스트의 내재적 측면 및 의견을 분석합니다. 분석 결과는 ERNIE-Bot-4 모델과 결합하여 실험적으로 평가되었으며, 이를 통해 정서 분석 작업에서 중요한 성과 개선을 입증하였습니다.

- **Performance Highlights**: 실험 결과, SAoT + ERNIE-Bot-4 모델은 레스토랑 데이터셋에서 75.27의 F1 점수와 66.29의 ISA 점수를 달성하였으며, 랩탑 데이터셋에서도 76.50의 F1 점수와 73.46의 ISA 점수를 기록하였습니다. ERNIE-Bot-4 + SAoT 모델은 BERTAsp + SCAPt 기준 모델을 평균 47.99% 초과하여 성능이 뛰어났습니다.



### A Tighter Complexity Analysis of SparseGP (https://arxiv.org/abs/2408.12151)
- **What's New**: 이번 연구에서는 SparseGPT의 러닝 타임 분석을 기존의 O(d^3)에서 O(d^{ω} + d^{2+a+o(1)} + d^{1+ω(1,1,a)-a})로 개선했습니다. 여기서 ω는 행렬 곱셈의 지수를 나타냅니다. 현재 ω의 값이 약 2.371일 때, 러닝 타임은 O(d^{2.53})으로 단축되었습니다.

- **Technical Details**: SparseGPT는 GPT 계열 모델의 파라미터를 최적화된 뇌 손상 기법(Optimal Brain Damage)을 사용하여 50% 이상 가지치기할 수 있습니다. 이 알고리즘(Algorithm 1)의 러닝 타임은 O(d^3)에서 O(d^{2.53})으로 개선되었습니다. 이 개선은 이터레이티브 유지 문제에서의 레이지 업데이트(lazy update) 행동 분석을 통해 이루어졌습니다.

- **Performance Highlights**: 향상된 러닝 타임 O(d^{2.53}) 덕분에 SparseGPT는 높은 성능을 유지하면서의 실행 시간 및 GPU 메모리 사용량을 감소시킬 수 있습니다. 이것은 LLMs의 애플리케이션에 매우 유리한 결과를 가져옵니다.



### DeepHQ: Learned Hierarchical Quantizer for Progressive Deep Image Coding (https://arxiv.org/abs/2408.12150)
- **What's New**: 이 논문에서는 NN(neural network) 기반의 새로운 프로그레시브 이미지 인코딩(PIC) 방법인 DeepHQ를 제안합니다. 기존 방법들은 수작업으로 설정된 양자화 계층(quantization hierarchy)에 의존했지만, DeepHQ는 학습된 양자화 단계 크기를 사용하여 각 인코딩 계층을 처리합니다.

- **Technical Details**: DeepHQ는 각 양자화 계층에 대해 학습된 양자화 단계 크기를 활용하며, 선택적 코드를 통해 필수적인 표현 요소만을 압축합니다. 이는 기존의 방식보다 더 높은 압축 효율과 더 짧은 디코딩 시간을 제공합니다.

- **Performance Highlights**: DeepHQ는 기존 NN 기반 PIC 방법들과 전통적인 코덱보다 월등한 인코딩 효율을 달성하며, 현재 최첨단 기법과 비슷한 인코딩 효율을 보이면서도 모델 크기는 7.88% 감소하고 디코딩 시간은 11.39% 단축됩니다.



### MDD-5k: A New Diagnostic Conversation Dataset for Mental Disorders Synthesized via Neuro-Symbolic LLM Agents (https://arxiv.org/abs/2408.12142)
- **What's New**: 본 논문에서는 심리적 질환을 진단하기 위한 대화 데이터를 생성하는 새로운 방법론을 제시합니다. 이는 익명의 환자 사례를 활용하여 비밀 유지 및 윤리적 고려 사항을 준수하며, 신경-기호적 다중 에이전트 프레임워크(neuro-symbolic multi-agent framework)를 통해 이루어집니다. 이를 통해 대화의 다양성과 정확성을 동시에 확보합니다.

- **Technical Details**: 제안된 프레임워크는 3가지 종류의 대형 언어 모델(large language model) 에이전트로 구성됩니다: 의사 에이전트(doctor agent), 환자 에이전트(patient agent), 진단 주제 전환을 관리하는 기호 도구 에이전트(symbolic tool agent)입니다. 이 구조는 고정된 증상 질문 트리와 동적인 경험 질문 트리를 통해 상징적 통제(symbolic control) 아래에서 텍스트 생성을 가능하게 합니다.

- **Performance Highlights**: MDD-5k 데이터셋은 1000개의 실제 환자 사례를 기반으로 5000개의 고품질 진단 대화를 포함하며, 이는 중국어로 된 정신 질환 진단 대화 데이터셋 중 최초로 레이블이 부여된 것입니다. 인간 평가 결과, 제안된 데이터셋은 전문성, 의사소통 기술, 유창성, 안전성 면에서 여러 기존 데이터셋보다 우수하다고 입증되었습니다.



### DRExplainer: Quantifiable Interpretability in Drug Response Prediction with Directed Graph Convolutional Network (https://arxiv.org/abs/2408.12139)
- **What's New**: 이번 연구에서는 DRExplainer라는 새로운 해석 가능한 예측 모델을 제안합니다. 이 모델은 directed bipartite network 구조에서 directed graph convolutional network를 활용하여 약물 반응 예측의 정확성을 높입니다.

- **Technical Details**: DRExplainer는 세포주(cell line)의 다중 오믹스(multi-omics) 프로파일, 약물의 화학 구조(Chemical Structure), 그리고 알려진 약물 반응(Drug Response)을 통합하여 directed bipartite network를 구성합니다. 이 모델은 각 예측에 대해 가장 관련성이 높은 서브그래프(subgraph)를 학습하여 의료 결정에 도움을 줍니다. 또한, 생물학적 특징에 대한 ground truth 벤치마크 데이터셋을 활용하여 모델 해석 가능성을 정량적으로 평가하는 방법도 제안합니다.

- **Performance Highlights**: DRExplainer는 경쟁력 있는 기존의 예측 방법들과 그래프 기반 해석 방법들보다 우수한 성능을 보여주었으며, 사례 연구를 통해 해석 가능성과 새로운 약물 반응 예측의 효과성을 추가로 검증했습니다.



### Deep Analysis of Time Series Data for Smart Grid Startup Strategies: A Transformer-LSTM-PSO Model Approach (https://arxiv.org/abs/2408.12129)
Comments:
          46 pages

- **What's New**: 그리드 스타트업(grid startup)은 전력 시스템의 필수 요소로, 전기 그리드의 신뢰성과 효율성을 보장하는 데 전략적 중요성을 지니고 있습니다. 본 연구에서는 기존 방법론의 한계를 극복하고자 Transformer-LSTM-PSO 모델을 기반으로 한 새로운 방법을 제안합니다.

- **Technical Details**: 이 모델은 Transformer's self-attention 메커니즘, LSTM의 시간적 모델링 능력, 그리고 파라미터 조정 기능을 가진 Particle Swarm Optimization(PSO) 알고리즘을 고유하게 결합하여, 그리드 스타트업 시나리오의 복잡한 시간적 관계를 보다 효과적으로 포착하도록 설계되었습니다.

- **Performance Highlights**: 실험을 통해 Transformer-LSTM-PSO 모델은 기존 벤치마크와 비교하여 여러 데이터 세트에서 RMSE(root mean square error)와 MAE(mean absolute error) 값에서 현저한 개선을 보여주었으며, 특히 NYISO Electric Market 데이터 세트에서는 RMSE가 약 15%, MAE가 20% 개선되었습니다. 이는 스마트 그리드 스타트업 예측의 정확성과 효율성을 크게 향상시키는 기여를 의미합니다.



### AutoTest: Evolutionary Code Solution Selection with Test Cases (https://arxiv.org/abs/2408.12125)
- **What's New**: AutoTest는 자동 테스트 케이스 생성과 코드 솔루션 실행을 결합하여 진화적 유전 알고리즘을 활용하는 새로운 기술로, 여러 후보 솔루션 중에서 최적의 선택을 최적화합니다.

- **Technical Details**: AutoTest는 codegen-16B, code-davinci-002, incoder-6B와 같은 대형 사전 학습된 언어 모델을 활용하여 코드 솔루션과 그에 해당하는 테스트 케이스를 생성합니다. 생성된 코드 솔루션의 성능 평가를 통해 일치 집합(consensus set)을 형성하고, 진화적 유전 알고리즘에 기반한 선택(selection), 변이(mutation) 및 교차(crossover) 메커니즘을 통해 세밀한 순위를 부여합니다.

- **Performance Highlights**: AutoTest는 HumanEval 벤치마크 테스트에서 약 10% 개선된 pass@1 점수를 달성하며, 164개의 프로그래밍 문제를 포함한 HumanEval 데이터셋에서 성능 향상을 입증했습니다.



### Emotion-Agent: Unsupervised Deep Reinforcement Learning with Distribution-Prototype Reward for Continuous Emotional EEG Analysis (https://arxiv.org/abs/2408.12121)
Comments:
          11 pages, 4 figures, 4 tables, submitted to AAAI 2025

- **What's New**: 이 논문은 지속적인 EEG 신호에서 관련되고 유의미한 감정 순간을 자동으로 식별하는 새로운 비지도 심층 강화 학습 프레임워크인 Emotion-Agent를 제안합니다.

- **Technical Details**: Emotion-Agent는 비지도 심층 강화 학습과 휴리스틱 알고리즘을 결합하여 초기 전역 검색을 수행하고 EEG 신호의 프로토타입 표현을 형성합니다. 이 프로세스는 Signal Space의 효율적인 탐색을 촉진하며, Distribution-Prototype 보상 함수를 설계하여 샘플과 프로토타입 간의 상호작용을 추정합니다. 이 모델은 Proximal Policy Optimization (PPO)을 사용하여 안정적이고 효율적인 수렴을 달성합니다.

- **Performance Highlights**: 실험 결과, Emotion-Agent를 사용하여 선택된 관련 감정 부분이 후속 작업에 입력되기 전에 aBCI 애플리케이션의 정확성과 신뢰성을 향상시키는 데 기여함을 보여줍니다.



### Understanding Data Reconstruction Leakage in Federated Learning from a Theoretical Perspectiv (https://arxiv.org/abs/2408.12119)
- **What's New**: 본 논문에서는 Federated Learning(FL)에서 발생하는 데이터 재구성 공격(Data Reconstruction Attacks, DRA)에 대한 이론적 프레임워크를 제안하여 이러한 공격의 효과성을 정량적으로 비교할 수 있는 방법을 제공합니다. 기존의 DRA 연구들은 공격 성능의 불안정성과 이론적 한계가 있어 효과적인 비교가 어려웠습니다.

- **Technical Details**: 제안된 프레임워크는 FL 훈련의 전체 과정에서 개인 데이터와 재구성된 데이터 간의 오차를 한계짓는 방식으로, 공격의 오차 한계가 본질적인 공격의 효과성을 반영합니다. 특히, 공격자의 재구성 함수의 Lipschitz 상수가 작을수록 공격 성능이 더 좋다는 결과를 도출하였습니다.

- **Performance Highlights**: 제안한 프레임워크를 다양한 최신 공격 기법 및 벤치마크 데이터셋에 적용한 결과, InvGrad 공격이 DLG 및 iDLG 공격보다 복잡한 데이터셋에서 더 우수한 성능을 보였으며, iDLG는 DLG와 비슷하거나 약간 더 우수한 성능을 나타냈습니다.



### Risk Analysis in Customer Relationship Management via Quantile Region Convolutional Neural Network-Long Short-Term Memory and Cross-Attention Mechanism (https://arxiv.org/abs/2408.12113)
Comments:
          44 pages

- **What's New**: 이 논문은 고객 관계 관리(CRM)에서의 위험 분석을 향상시키기 위해 새로운 모델인 QRCNN-LSTM과 cross-attention 메커니즘을 결합했습니다.

- **Technical Details**: QRCNN-LSTM 모델은 시퀀스 모델링(sequence modeling)과 자연어 처리(natural language processing) 작업에 일반적으로 사용되는 딥러닝 아키텍처를 결합합니다. 이 모델은 시퀀스 데이터의 지역(local) 및 전역(global) 의존성을 모두 포착할 수 있습니다. Cross-attention 메커니즘은 서로 다른 입력 데이터의 부분 간 상호작용을 증대시켜, CRM 위험 분석과 관련된 특정 영역이나 기능에 집중할 수 있게 합니다.

- **Performance Highlights**: QRCNN-LSTM과 cross-attention 메커니즘을 CRM 위험 분석에 적용한 결과, 이 접근 방식이 잠재적인 위험을 효과적으로 식별하고 데이터 기반의 비즈니스 결정 지원을 제공함을 입증하는 경험적 증거가 나타났습니다.



### Balancing Act: Prioritization Strategies for LLM-Designed Restless Bandit Rewards (https://arxiv.org/abs/2408.12112)
- **What's New**: 이 논문은 LLM (Large Language Models) 기반의 보상 함수 설계에 관한 새로운 접근법을 제시합니다. Restless Multi-Armed Bandits (RMAB) 문제에서의 사람 선호를 반영한 보상 함수를 설계하기 위한 원칙적인 방법인 Social Choice Language Model (SCLM)을 소개합니다. 이 모델은 커스터마이징이 가능한 외부 조정자(adjudicator)를 통해 다목적 리소스 할당 문제를 효율적으로 해결할 수 있도록 합니다.

- **Technical Details**: SCLM에서 보상 함수를 설계하는 과정은 LLM을 활용하여 후보 보상 함수들을 생성한 후, 사용자 선택에 기반한 사회적 복지 함수를 통해 최적의 보상 함수를 선택하는 두 단계로 이루어집니다. 이 과정에서는 복잡한 인간 언어 프롬프트에 따라 다양한 목표를 평가하는 스코어러(scorer) 컴포넌트가 중요하게 작용합니다.

- **Performance Highlights**: 실험 결과, SCLM은 복잡한 다목적 프롬프트에 대해 기존 LLM 기반 접근법보다 더 효과적이고 균형 잡힌 보상 함수를 선택했습니다. 이는 사회적 선택(social choice) 관점을 통해 리소스 할당에서의 의도하지 않은 부작용을 줄일 수 있음을 보여줍니다.



### Extraction of Research Objectives, Machine Learning Model Names, and Dataset Names from Academic Papers and Analysis of Their Interrelationships Using LLM and Network Analysis (https://arxiv.org/abs/2408.12097)
Comments:
          10 pages, 8 figures

- **What's New**: 이 연구는 기존의 정보 추출 방법을 개선하여 논문에서 연구 목표, 기계학습(methods), 데이터셋 이름을 동시에 추출하고 이를 클러스터링하여 관계를 분석하는 새로운 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 Llama2와 Llama3라는 대형 언어 모델을 활용하여 연구 목표와 관련된 내용을 동시에 추출하고, E5 임베딩 모델을 통해 동의어 클러스터링을 수행합니다. 이 과정에서 네트워크 클러스터링을 통한 관계 분석도 포함됩니다.

- **Performance Highlights**: 실험 결과, Llama3를 사용한 정보 추출 성능의 F-score가 0.8을 초과하며, 금융 분야 논문에 대해 최신 데이터셋과의 관계를 명확히 보여 줍니다.



### uMedSum: A Unified Framework for Advancing Medical Abstractive Summarization (https://arxiv.org/abs/2408.12095)
Comments:
          12 pages

- **What's New**: 본 논문은 의료 분야의 추상적 요약에서 진정성(faithfulness)과 정보성(informativeness)을 균형 있게 유지하는 방법을 제안합니다. 특히 uMedSum이라는 모듈형 하이브리드 요약 프레임워크를 소개하며, 이 프레임워크는 세 가지 다양한 데이터셋을 기반으로 한 six advanced abstractive summarization 방법의 포괄적인 벤치마크를 제공합니다.

- **Technical Details**: 우리는 confabulation 제거 및 핵심 누락 정보 추가를 위한 새로운 접근 방식을 도입하며, 이를 통해 진정성과 정보성을 모두 보장하는 방법을 개발했습니다. 모델의 판단력(model reasoning)과 자기 개선(self-improvement)을 활용하여 기존 기법들의 한계를 극복합니다. 또한, 우리는 reference-based 및 reference-free metric을 포함한 다섯 가지 표준화된 메트릭을 사용하여 의료 요약의 성능을 평가합니다.

- **Performance Highlights**: 우리의 uMedSum 프레임워크는 이전 SOTA 방법에 비해 평균 11.8%의 상대적인 성능 개선을 달성하였으며, 특히 confabulation이 발생하기 쉬운 어려운 사례에서도 의료 전문가들이 uMedSum의 요약을 6배 더 선호합니다. 이 결과는 다양한 데이터셋과 메트릭에서 uMedSum의 효과성과 일반성을 입증하는 중요한 진전을 나타냅니다.



### Unlocking Attributes' Contribution to Successful Camouflage: A Combined Textual and VisualAnalysis Strategy (https://arxiv.org/abs/2408.12086)
Comments:
          Accepted by ECCV 2024

- **What's New**: 본 연구는 Camouflaged Object Segmentation (COS) 분야에서 효과적인 위장 패턴의 기여도를 정량적으로 평가할 수 있는 최초의 포괄적인 프레임워크인 ACUMEN을 제안합니다. 이 연구에서는 COD-Text And X-attributions (COD-TAX)라는 데이터셋을 구축하여 위장 객체의 특성과 그 기여도를 체계적으로 분석합니다.

- **Technical Details**: ACUMEN(Attribution CUe Modeling with Eye-fixation Network)은 시각적 정보와 텍스트 정보를 통합하여 COS 작업을 수행하는 강력한 프레임워크입니다. 이 모델은 frozen CLIP 텍스트 인코더를 사용하여 텍스트 분석을 수행하고, 시각적 분석을 위해 기여도와 주시 예측기를 도입합니다. AFE(Attributes-Fixation Embedding) 모듈을 통해 예측된 특성 기여 텐서 및 주시 맵을 최대화하고, 최종적으로 transformer decoder를 사용하여 위장 객체의 마스크를 생성합니다.

- **Performance Highlights**: ACUMEN은 세 개의 널리 사용되는 데이터셋에서 아홉 개의 기존 방법을 초월하여 우수한 성능을 입증했습니다. 본 연구는 위장 기법의 이해를 심화시키고, 전통적인 방법에 비해 성능 향상의 가능성을 보여줍니다.



### Exploring the Feasibility of Automated Data Standardization using Large Language Models for Seamless Positioning (https://arxiv.org/abs/2408.12080)
Comments:
          Accepted at IPIN 2024. To be published in IEEE Xplore

- **What's New**: 본 연구는 사물인터넷(IoT) 환경에서의 원활한 위치 시스템을 향상시키기 위해 대규모 언어 모델(LLMs)을 활용한 실시간 자동 데이터 표준화의 가능성을 탐구하는 내용을 다룹니다. 다양한 센서 데이터를 통합하고 표준화하여 데이터의 호환성을 확보하고 위치 정확성을 높이는 방법을 제시합니다.

- **Technical Details**: 연구의 핵심 구성 요소는 Intelligent Data Standardization Module (IDSM)와 Transformation Rule Generation Module (TRGM)입니다. IDSM은 정제된 LLM을 통해 다양한 센서 데이터를 표준화된 형식으로 변환하고, TRGM은 자동화된 변환 규칙과 스크립트를 생성하여 지속적인 데이터 표준화를 지원합니다. 이 시스템은 실시간 환경에서 평가되어 적응성과 확장성을 입증합니다.

- **Performance Highlights**: 연구 결과는 기존 데이터 표준화 방법에 비해 운영 효율성과 정확성을 향상시켜 IoT 내비게이션 솔루션의 확장성과 정밀성을 제공할 수 있는 가능성을 보여줍니다. LLM을 활용한 자동 데이터 표준화는 수작업 개입을 줄이고, 다양한 센서 데이터의 일관성 있는 표준화에 기여합니다.



### High-Quality Data Augmentation for Low-Resource NMT: Combining a Translation Memory, a GAN Generator, and Filtering (https://arxiv.org/abs/2408.12079)
- **What's New**: 본 논문은 저자들이 제안한 새로운 접근법으로, 저자 측 언어의 단일 언어 코퍼스를 활용하여 Neural Machine Translation (NMT) 모델의 성능을 향상시키는 방법을 제시합니다. 또한, Generative Adversarial Network (GAN)을 통해 훈련 데이터의 질을 보장하고 Translation Memory (TM)를 통합하여 데이터를 증가시킴으로써 저자 측 언어에 대한 데이터의 양을 증가시키는 방법도 설명합니다.

- **Technical Details**: 이론적으로 GAN 구조를 활용하여 NMT를 위한 데이터 증강 방안을 연구합니다. TM과 GAN을 통합하여 생성기에서 학습할 수 있는 데이터의 양을 늘리며, 질 높은 번역 결과를 보장하기 위한 새로운 필터링 절차도 제안하고 있습니다. 연구에서 소개된 Euclidean 거리 기반의 유사도 측정을 통해 TM에서 유사한 문장을 검색하여, 이 문장이 생성기에 통합될 수 있도록 설계합니다.

- **Performance Highlights**: 본 연구는 저자 언어의 단일 언어 코퍼스를 활용하여 기존 모델의 전통적인 접근 방식을 넘어서, 효과적으로 NMT 성능을 향상시킬 수 있는 방법을 제시합니다. 실험 결과는 단일 언어 코퍼스와 TM의 통합이 저자 언어 번역의 정확도와 유효성을 높여주는 데 긍정적인 영향을 미친다는 것을 보여주고 있습니다.



### ConflictBank: A Benchmark for Evaluating the Influence of Knowledge Conflicts in LLM (https://arxiv.org/abs/2408.12076)
Comments:
          Under Review

- **What's New**: 본 논문은 ConflictBank라는 체계적인 지식 갈등 평가 기준을 제시하며, 이는 LLMs에서 발생하는 다양한 지식 갈등을 분석하기 위한 최초의 포괄적 벤치마크입니다.

- **Technical Details**: ConflictBank는 세 가지 주요 갈등 원인인 잘못된 정보(misinformation), 시간에 따른 지식 변화(temporal conflict), 언어의 다의성(semantic conflict)들을 포함하여, 7,453,853개의 주장-증거(claim-evidence) 쌍과 553,117개의 QA 쌍을 구성하였습니다.

- **Performance Highlights**: ConflictBank를 기반으로 진행된 파일럿 실험에서는 12개의 LLM 모델에서 다양한 갈등 상황에 대한 모델 동작을 분석하였으며, 지식 갈등 원인, 갈등 유형 및 모델 규모에 대한 통찰을 제공하였습니다.



### Distributed Noncoherent Joint Transmission Based on Multi-Agent Reinforcement Learning for Dense Small Cell MISO Systems (https://arxiv.org/abs/2408.12067)
- **What's New**: 이 논문에서는 다중 안테나를 가진 소형 기지국(Small Cell Base Stations, SBSs)과 단일 안테나 사용자가 있는 밀집 소형 셀(Dense Small Cell, DSC) 네트워크를 고려하여, 비합동(joint transmission, JT) 전송 기술을 적용하여 시스템 용량을 극대화하는 새로운 방안을 제안합니다. 기존의 최적화 기반 비합동 JT 알고리즘의 한계를 극복하여, 심층 결정 정책 기울기(Deep Deterministic Policy Gradient, DDPG) 기반의 분산 비합동 JT 방법을 제안합니다.

- **Technical Details**: 기존의 비합동 JT 최적화 문제는 비선형(nonconvex) 및 NP-hard 문제로 알려져 있으며, 이를 해결하기 위해 저자는 전력 최소화 문제와 합산비율 최대화 문제의 최적 빔형성 구조가 동일하다는 것을 증명하고, 이를 통해 수학적으로 최적 빔형성 구조를 유도합니다. 제안된 분산 비합동 JT(DDNJT) 스킴은 각 SBS가 로컬 채널 상태 정보(CSI)만으로 빔형성 벡터를 결정할 수 있도록 하여 계산 복잡도를 줄입니다.

- **Performance Highlights**: 시뮬레이션 결과, 제안된 DDNJT 스킴은 중앙 집중형 반복 최적화 방법의 90% 이상의 합산 비율 성능을 달성하며, 계산 복잡도와 정보 오버헤드를 대폭 줄였습니다.



### A Deconfounding Approach to Climate Model Bias Correction (https://arxiv.org/abs/2408.12063)
- **What's New**: 본 논문은 기존의 편향 수정(bias correction) 방법의 한계를 극복하기 위해, GCM(Global Climate Models)과 관측(observation) 데이터를 결합하여 다인자 잠재 혼란인자를 학습하는 새로운 방법을 제안합니다. 이 방법은 최근 인과관계 기반의 시간 시계열(time series) 혼란 제거 기술에서 영감을 받아 개발되었습니다.

- **Technical Details**: 제안된 방법은 GCM 출력 데이터와 관측 데이터로부터 잠재 혼란인자를 캡처하기 위한 요인 모델(factor model)을 구축하고, 이 모델을 활용하여 편향 수정 과정을 개선하는 방식입니다. 이 과정에서 고급 시간 시계열 예측 모델을 활용하며, 잠재 혼란인자를 식별하는 두 단계의 알고리즘(Deconfounding and Correction)을 사용합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 강수량 출력(precipitation outputs)의 정확성을 크게 향상시키며, 예측의 신뢰성을 높입니다. 이는 특히 비관측된 혼란인자를 다룸으로써 달성되었습니다.



### Enhancing Sampling Protocol for Robust Point Cloud Classification (https://arxiv.org/abs/2408.12062)
- **What's New**: 이번 연구에서는 포인트 클라우드 학습을 위한 향상된 샘플링 프로토콜인 PointDR을 제안합니다. 기존 샘플링 프로토콜들이 현실 데이터의 노이즈에 취약하다는 문제를 해결하기 위해 다운샘플링 및 리샘플링 과정을 통해 더 강인한 방법을 제공합니다.

- **Technical Details**: PointDR 프로토콜은 두 가지 주요 구성 요소로 이루어져 있습니다: 1) 키 포인트 식별을 위한 다운샘플링 및 2) 유연한 샘플 사이즈를 위한 리샘플링. 다운샘플링 과정에서는 로컬 밀도를 고려한 격리 비율(weight)을 이용해 핵심 포인트를 무작위로 선택하고, 리샘플링에서는 로컬 기하학을 보존하면서 불완전한 데이터를 보완합니다.

- **Performance Highlights**: PointDR은 다양한 오염된 포인트 클라우드 분류 벤치마크에서 최첨단 방법들을 초월하는 성능을 보여주며, 포인트 클라우드 학습의 강인성을 크게 향상시킵니다.



### Evidence-backed Fact Checking using RAG and Few-Shot In-Context Learning with LLMs (https://arxiv.org/abs/2408.12060)
- **What's New**: 소셜 미디어에서의 정보 조작이 만연함에 따라 자동화된 사실 확인 시스템의 필요성이 강조되고 있습니다. 이 논문에서는 Averitec 데이터셋을 사용하여 온라인 주장에 대한 진실성을 검증하는 자동화된 시스템을 제안합니다. 시스템은 주장을 지원하는 증거를 제공하며, 대규모 언어 모델(LLM)을 사용하여 분류합니다.

- **Technical Details**: 이 시스템은 Retrieve and Generate (RAG) 파이프라인을 통해 관련 증거 문장을 추출하고, In-Context Learning (ICL)을 통해 주장의 진실성을 판단합니다. 주어진 주장과 문서 집합을 활용하여, RAG를 통해 가장 관련성 높은 문서를 검색하고 이를 기반으로 ICL을 적용하여 결과를 도출합니다.

- **Performance Highlights**: 시스템은 Averitec 데이터셋에서 0.33의 점수를 기록하며, 이는 기존 기준선보다 22% 향상된 결과입니다. 최소한의 학습 샘플로 작동 가능하며 다양한 LLM과의 실험을 통해 효과성을 입증합니다.



### Enhancing LLM-Based Automated Program Repair with Design Rationales (https://arxiv.org/abs/2408.12056)
- **What's New**: 이번 연구에서는 설계 이론(Design Rationale, DR)을 활용하여 자동 프로그램 수리(Automated Program Repair, APR)에서 GPT-4-Turbo의 성능을 향상시키기 위한 DRCodePilot 접근법을 제안합니다.

- **Technical Details**: DRCodePilot는 DRMiner 도구를 통해 이슈 솔루션과 해당 주장을 추출하여 DR을 도출하고, GPT-4가 결함 있는 코드 세그먼트를 지적하여 모든 DR을 고려하여 패치를 생성합니다. 이후, 피드백을 통해 초기 답변을 반성하게 하여 최종 패치를 다듬습니다.

- **Performance Highlights**: 실험 결과, DRCodePilot는 Flink 데이터셋에서 714개의 샘플 중 109개의 풀 매치를 달성하였고, Solr 데이터셋에서는 224개의 샘플 중 18개로, 가장 성능이 우수한 GPT-4보다 각각 4.7배, 3.6배 높은 효과를 보였습니다. 또한 CodeBLEU 점수를 GPT-4 대비 5.4% 및 3.9% 향상시켰습니다.



### Reasoning and Tools for Human-Level Forecasting (https://arxiv.org/abs/2408.12036)
- **What's New**: 본 논문은 Reasoning and Tools for Forecasting (RTF)라는 새로운 프레임워크를 제안하여, 웹 기반 데이터 세트로 훈련된 언어 모델이 실제로 합리적 추론을 수행할 수 있는 방법을 탐구합니다. 특히, 예측 작업에서 LMs(언어 모델)의 정확성과 적응 능력을 향상시키는 데 초점을 맞추고 있습니다.

- **Technical Details**: RTF 프레임워크는 ‘Reasoning-and-acting (ReAct)’ 에이전트를 기반으로 하며, 에이전트는 최신 정보를 동적으로 검색하고 수치 시뮬레이션을 실행할 수 있는 도구를 갖추고 있습니다. 이 접근법에서는 LMs가 관찰 공간(𝒪)과 행동 공간(𝒜) 내에서 작업을 수행하도록 설계되었습니다.

- **Performance Highlights**: 경쟁적인 예측 플랫폼에서 문제를 평가했으며, 제안한 방법이 인간의 예측보다 우수하거나 경쟁력이 있음을 보여줍니다. 이는 적절한 도구를 갖춘 LMs가 인간처럼 사고하고 적응할 수 있는 가능성을 제시합니다.



### A Constraint Programming Approach to Fair High School Course Scheduling (https://arxiv.org/abs/2408.12032)
- **What's New**: 본 연구는 미국 고등학교의 교육 과정 스케줄링의 공정성을 개선하기 위한 새로운 접근법을 제시합니다. 기존의 정수 프로그래밍 방법이 공정성 문제를 충분히 다루지 못했음을 강조하며, 학생 선호도를 활용한 새로운 알고리즘을 제안합니다. 이를 통해 '공정 고등학교 스케줄링 문제(FHSSP)'라는 새로운 모델을 정의하고 이를 해결하기 위한 알고리즘을 개발하였습니다.

- **Technical Details**: 제안된 접근법은 공정성을 고려하여 교육 과정 스케줄링 문제를 새로운 것으로 확장하는 데 초점을 맞추고 있습니다. 이 모델은 정수 프로그래밍(integer programming) 기법을 사용하여 공정한 스케줄을 생성하며, 기존의 제약 프로그래밍(constraint programming) 방식과 비교하여 구현됩니다. 실험은 미국 캘리포니아 고등학교의 실제 수업 요청 데이터셋을 바탕으로 진행되었습니다.

- **Performance Highlights**: 우리의 알고리즘은 생성된 스케줄이 실행 가능할 뿐만 아니라 공정하다는 것을 입증하였습니다. 연구 결과는 제안된 IP 알고리즘이 미국 내 고등학교 스케줄링 문제를 해결할 뿐만 아니라 다양한 실제 스케줄링 문제에 적용될 가능성이 있음을 보여줍니다. 또한, 인간의 감정을 수학적 모델링에 통합할 수 있는 가능성도 논의됩니다.



### Federated Diabetes Prediction in Canadian Adults Using Real-world Cross-Province Primary Care Data (https://arxiv.org/abs/2408.12029)
Comments:
          10 pages

- **What's New**: 이 논문은 전자 건강 기록 (EHR)과 머신 러닝 (Machine Learning)의 통합을 통해 당뇨병 예측의 정확성과 접근성을 향상시키기 위한 새로운 접근 방식을 제시합니다. 특히, 캐나다의 임상 데이터셋을 사용한 연방 학습 (Federated Learning)의 첫 번째 적용 사례로, 환자 개인정보 보호 문제를 피하면서 중앙 집중식 데이터 저장 및 처리를 하지 않고도 예측 모델을 결합할 수 있습니다.

- **Technical Details**: 이 연구에서는 연방 학습 접근 방식을 활용하여 예측 모델을 개발하며, 클래스 불균형 (Class Imbalance) 문제를 해결하기 위해 다운샘플링 기술을 사용합니다. 실험에서는 연방 학습을 통해 생성된 다층 퍼셉트론 (MLP) 모델과 중앙 집중식 접근 방식으로 학습된 모델의 성능을 비교한 결과, 전자 공공 데이터 공유 없이도 유사하거나 더 높은 성능을 보여주었습니다.

- **Performance Highlights**: 연방 학습 기반의 MLP 모델이 중앙 집중식 모델과 유사하거나 더 높은 성능을 보인 반면, 연방 로지스틱 회귀 모델은 중앙 집중식 모델에 비해 성능이 열세한 것으로 나타났습니다.



### Understanding Epistemic Language with a Bayesian Theory of Mind (https://arxiv.org/abs/2408.12022)
Comments:
          21 pages

- **What's New**: 본 논문에서는 인간이 다른 사람의 믿음에 대한 주장을 어떻게 이해하고 평가하는지에 대한 인지 모델인 LaBToM(Language-augmented Bayesian theory-of-mind)을 제안합니다. 이 모델은 행동과 관찰을 기반으로 다른 에이전트의 목표와 믿음, 의도에 대한 베이지안 추론(Bayesian inference)을 토대로 합니다.

- **Technical Details**: LaBToM은 확률적 생성 모델을 기반으로 하는 합리적 행동과 인식을 통해 정의된 두 개의 상호 연결된 모듈로 구성됩니다. 첫 번째 모듈은 다른 사람의 마음을 포함한 세계를 설명하는 개념들을 조합하여 리치한 사고 및 표현을 나타내는 에피스테믹 언어(ELoT)를 모델링합니다. 두 번째 모듈은 민첩한 사고 이론(Bayesian theory-of-mind)으로, 에이전트가 자신의 믿음을 업데이트하고 목표를 향해 행동하는 방식에 대한 직관적 추론을 캡처합니다.

- **Performance Highlights**: LaBToM 모델은 참여자들이 에이전트의 믿음에 대해 작성한 문장에 대한 평가와 높은 상관관계를 보였으며, 화자들의 표현에 대한 인간의 판단을 이해하는 데 중요한 역할을 하며, 기존의 다중 모달 LLM 모델 및 수정된 BToM 모델들과의 비교에서도 우수한 성과를 보였습니다.



### Does It Look Sequential? An Analysis of Datasets for Evaluation of Sequential Recommendations (https://arxiv.org/abs/2408.12008)
- **What's New**: 본 논문에서는 순차 추천 시스템(Sequential Recommender Systems, SRSs)의 평가를 위해 데이터셋의 순차 구조의 강도를 측정하는 방법을 제안합니다. 특히, 데이터의 순차 패턴이 존재하는지를 분석하기 위한 15개의 일반적인 데이터셋을 평가했습니다.

- **Technical Details**: 우리는 사용자의 상호작용 순서를 무작위로 섞어 데이터셋 내에서 순차 구조의 강도를 측정하는 세 가지 접근 방식을 제안합니다. 첫 번째 접근 방식은 순차 규칙을 식별하며 모델에 독립적입니다. 두 번째와 세 번째 접근 방식은 SASRec 및 GRU4Rec이라는 순차 모델을 훈련하고, 원본 및 섞인 데이터셋에서의 성능을 NDCG@k 및 HitRate@k를 통해 평가합니다.

- **Performance Highlights**: 연구 결과, 여러 널리 사용되는 데이터셋이 예상보다 상대적으로 약한 순차 구조를 지니고 있음이 밝혀졌습니다. 이는 특히 데이터셋이 SRS 평가에 부적합할 수 있음을 시사합니다.



### QuaCK-TSF: Quantum-Classical Kernelized Time Series Forecasting (https://arxiv.org/abs/2408.12007)
Comments:
          12 pages, 15 figures, to be published in IEEE Quantum Week 2024's conference proceeding

- **What's New**: 본 논문은 양자 커널화된 확률적 시계열 예측을 향상시키기 위한 새로운 접근법을 제안합니다. 여기에서는 Bayesian 기법의 강력함을 양자 모델의 커널 관점과 결합하여 Robust한 예측 체계를 구축했습니다.

- **Technical Details**: QuaCK-TSF(Quantum-Classical Kernelized Time Series Forecasting)라는 모델을 도입하였으며, 이 모델은 Gaussian process regression (GPR) 기반으로 양자 커널을 사용하여 시계열 데이터를 처리합니다. 또한, Ising 상호작용에서 영감을 받은 양자 특징 맵을 통해 시간 의존성을 캡처합니다.

- **Performance Highlights**: 비교 기준으로 기존의 고전적 커널 모델과 성능을 비교했을 때, 제안된 양자 증강 접근법이 경쟁력 있는 성능을 달성함을 확인했습니다.



### Chemical Reaction Neural Networks for Fitting Accelerated Rate Calorimetry Data (https://arxiv.org/abs/2408.11984)
- **What's New**: 리튬 이온 배터리의 열폭주(thermal runaway)에 대한 안전성 문제를 해결하기 위해 Chemical Reaction Neural Networks (CRNNs)를 활용하여 기존 모델보다 더 나은 근사치를 제공하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: CRNN을 통해 N-방정식 Arrhenius ODE의 동역학 매개변수를 ARC 당시 데이터를 적합시키며 실험적 데이터와의 일치를 높입니다. 두 개 또는 네 개의 방정식 모델로 실험을 진행하여 이 방법의 유연성을 보여줍니다.

- **Performance Highlights**: CRNN 모델이 실험 데이터와 더 잘 일치함을 입증하였으며, 얻어진 동역학 매개변수로 3D 열폭주 시뮬레이션을 수행하여 대규모 시뮬레이션에 적용 가능성을 보여줍니다.



### Only Strict Saddles in the Energy Landscape of Predictive Coding Networks? (https://arxiv.org/abs/2408.11979)
Comments:
          26 pages, 12 figures

- **What's New**: 이 논문에서는 예측 코딩(Predictive Coding, PC)의 에너지 경관을 분석하여, 이 학습 알고리즘이 지닌 이점과 현실적인 학습에서의 성능을 이해하고자 하였습니다. 특히, 딥 선형 네트워크(Deep Linear Networks, DLNs)의 경우 쌍의 안정화된 에너지가 가중치에 따라 조정된 평균 제곱 오차(mean squared error, MSE) 손실과 같다는 것을 보여주고, 다양한 비엄격 중간 지점(sadles)에서는 엄격한 중간 지점으로의 전환 가능성을 제시하였습니다.

- **Technical Details**: 예측 코딩(PC)은 반복적인 추론 과정을 통해 네트워크 활동을 기반으로 학습하는 에너지 기반 학습 알고리즘입니다. 본 연구에서 제시된 이론에 따르면, DLNs의 안전한 에너지는 가중치에 의존적인 조정이 이루어진 평균 제곱 오차(MSE) 손실의 재조정된 형태로 나타나며, 이를 통해 DLNs의 손실 경관에서 발생하는 중간 지점의 성격이 변화를 겪는 것으로 설명됩니다. 특히, 다수의 비엄격(Non-strict) 중간 지점이 안전한 에너지에서는 엄격한(Sstrict) 중간 지점으로 변환됩니다.

- **Performance Highlights**: 실험 결과, PC 추론이 DLNs와 비선형 네트워크 모두에서 손실 경관을 더 호의적이고 강건하게 만든다는 것을 확인하였으며, 이는 학습 과정에서의 기울기 소실(vanishing gradients) 문제를 완화합니다. 비선형 네트워크의 경우에도 PC가 첫 번째 순서 방법(stochastic gradient descent, SGD)에 비해 빠르게 비엄격 중간 지점에서 탈출할 수 있음을 보여주었습니다.



### Real-Time Incremental Explanations for Object Detectors (https://arxiv.org/abs/2408.11963)
- **What's New**: 이번 논문에서는 인상적인 속도로 객체 감지기의 설명을 제공하는 새로운 알고리즘 IncX를 소개합니다. IncX는 기존의 블랙박스 설명 도구들이 여러 번 모델에 호출해야 하는 문제를 해결하여, 실시간으로 설명을 생성할 수 있도록 설계되었습니다.

- **Technical Details**: IncX는 선형 변환(linear transformations)을 기반으로 한 saliency map의 점진적인 근사(incremental approximation) 기법을 사용하여, 객체 분류 시간을 거의 변화시키지 않으면서 실시간으로 설명을 처리할 수 있습니다. 이전의 프레임에서 생성된 saliency map을 변형하고 축척하여 다음 프레임에 적응하는 방식으로 작동하며, d-rise와 결합하여 그 품질에 있어 비슷한 수준의 설명력을 보여줍니다.

- **Performance Highlights**: IncX는 d-rise보다 두 배 빠른 속도로 작동하며, 설명 생성 속도가 두 주문 단위로 향상되었습니다. 이를 통해 영상 데이터 및 대량의 시계열 이미지 데이터의 효율적인 처리에 매우 유용합니다.



### Explainable Anomaly Detection: Counterfactual driven What-If Analysis (https://arxiv.org/abs/2408.11935)
Comments:
          8 pages, 6 figures, 3 tables

- **What's New**: 본 연구는 다변량 데이터에서의 이상 탐지(anomaly detection)를 위한 counterfactual explanations를 활용하여 what-if 분석을 수행하는 개념 증명을 제시합니다. 이를 통해 시스템의 고장이 발생했을 때 원인과 이에 대한 접근 방법을 제시하는 새로운 시각을 제공합니다.

- **Technical Details**: 이 연구에서 제안하는 방법론은 Temporal Convolutional Network(TCN)를 사용하여 PRONOSTIA 데이터셋에서 이상 탐지를 수행합니다. 이상탐지된 데이터를 기반으로 counterfactuals를 생성하여 데이터를 '건강한(healthy)' 상태로 변경하기 위한 추천을 제공합니다. 이 과정은 데이터 포인트의 상태를 변경하여 이상 탐지 문제를 해결하려는 것입니다.

- **Performance Highlights**: 이 접근 방식은 기계 학습(machine learning)과 설명 가능한 인공지능(explainable AI)의 접목을 통해 이상 탐지의 핵심적인 문제인 '이상 원인 분석'과 '대체 행동 제안'을 효과적으로 지원합니다. 미래의 복잡한 시스템과 시나리오에 대한 연구에 영감을 줄 수 있는 기반을 제공합니다.



### Neural Symbolic Logical Rule Learner for Interpretable Learning (https://arxiv.org/abs/2408.11918)
Comments:
          19 pages, 62 figures

- **What's New**: 이번 논문에서는 해석 가능한 분류를 위한 규칙 기반 신경망 모델인 Normal Form Rule Learner (NFRL) 알고리즘을 소개합니다. NFRL은 선택적 이산 신경망(selective discrete neural network)을 활용하여 Conjunctive Normal Form (CNF) 및 Disjunctive Normal Form (DNF)에서 규칙을 학습합니다. 이를 통해 기존 모델의 유연성 부족 문제를 해결합니다.

- **Technical Details**: NFRL 알고리즘은 두 개의 Normal Form Layers (NFLs), 입력 부정을 위한 Negation Layer, 그리고 시냅스를 간소화하는 Normal Form Constraint (NFC)를 포함하여 AND/OR 뉴런을 적용합니다. 또한, Straight-Through Estimator를 사용하여 적응형 경량 업데이트(adaptive gradient update)로 기울기 소실 문제를 극복합니다.

- **Performance Highlights**: 11개의 데이터셋에 대한 실험을 통해, NFRL은 12개 최신 기술 대비 우수한 분류 성능, 학습된 규칙의 품질, 효율성 및 해석 가능성을 보여주었습니다. 코드와 데이터는 제공된 링크를 통해 이용할 수 있습니다.



### Why am I Still Seeing This: Measuring the Effectiveness Of Ad Controls and Explanations in AI-Mediated Ad Targeting Systems (https://arxiv.org/abs/2408.11910)
Comments:
          Accepted to the 7th AAAI Conference on AI, Ethics, and Society (AIES, 2024)

- **What's New**: 최근 Meta는 광고주가 상세한 타겟팅 기준을 제공하지 않아도 되는 AI 기반 광고 타겟팅 메커니즘으로 전환했습니다. 이는 AI 기능에 대한 관심 뿐만 아니라, 새로운 데이터 프라이버시 정책과 시민권 합의에 따른 타겟팅 변화에 의해 주도된 것으로 보입니다. 또한, Meta는 사용자가 광고를 제어할 수 있도록 돕는 광고 선호 제어 기능을 강조하고 있습니다.

- **Technical Details**: 이 연구는 Meta의 'See less' 광고 통제 및 AI 기반 타겟팅으로 전환된 이후 광고 타겟팅 설명의 유용성을 평가합니다. 2024년 초, 참여자에게 'See less' 버튼을 통해 '체중 조절' 또는 '육아' 주제를 표시하도록 무작위로 할당하였으며, 개입 전후로 Meta가 표시하는 광고 및 타겟팅 설명을 수집했습니다.

- **Performance Highlights**: 'See less' 광고 통제의 효과는 유의미하게 감소하지 않았으며, 일부 인구 통계와 관련된 사용자에게는 효과가 떨어졌습니다. 광고 타겟팅 설명의 대부분은 지역별 타겟팅 기준을 언급하지 않았고, 사용자가 'See less'로 표시한 주제의 광고가 계속 표시되는 이유를 명확히 설명하지 않았습니다. 이러한 결과는 AI 기반 타겟팅이 광고 통제의 유효성 및 설명의 실행 가능성에 부정적인 영향을 미친다는 가설을 입증합니다.



### Beyond Labels: Aligning Large Language Models with Human-like Reasoning (https://arxiv.org/abs/2408.11879)
Comments:
          Accepted in ICPR 2024

- **What's New**: 이 연구에서는 윤리적 추론을 생성을 지원하는 새로운 데이터셋인 'Dataset for Aligning Reasons (DFAR)'를 소개합니다. 이 데이터셋은 윤리적 및 비윤리적 진술과 그에 대한 설명을 포함하고 있어 자연어처리(NLP)의 인간적 결정을 더 잘 반영할 수 있도록 돕습니다.

- **Technical Details**: DFAR에는 2886개의 윤리적 샘플(57.7%)과 2114개의 비윤리적 샘플(42.3%)이 포함되어 있으며, 12명의 주석자가 주석을 달았습니다. 연구는 라벨(Labels)과 그에 맞는 설명(Reasons)을 모두 사용하는 독특한 미세 조정(fine-tuning) 방법을 적용하였으며, 이 방식은 기존의 미세 조정 방식과 구별됩니다.

- **Performance Highlights**: 새로운 미세 조정 방법은 윤리-비윤리 분류 작업과 이유 생성 작업에서 다른 방법들보다 뛰어난 성능을 보였으며, 분류 작업의 정확도가 상당히 증가하고 이유 생성 작업에서의 잘못된 정렬 비율이 감소했습니다. 이는 L+R 방식의 미세 조정이 LLM이 인간 윤리에 더욱 잘 align되도록 만든다는 것을 보여줍니다.



### From Glucose Patterns to Health Outcomes: A Generalizable Foundation Model for Continuous Glucose Monitor Data Analysis (https://arxiv.org/abs/2408.11876)
- **What's New**: 본 논문에서는 새로운 생물의학 시계열 데이터에 기반한 생성적 기초 모델인 GluFormer를 소개합니다. 이 모델은 10,812명의 비당뇨환자로부터 수집된 1천만 건 이상의 연속 혈당 모니터링(Continuous Glucose Monitoring, CGM) 데이터를 활용하여 개발되었습니다.

- **Technical Details**: GluFormer는 트랜스포머(Transformer) 아키텍처를 기반으로 하며, CGM 훈련 데이터를 토큰화(tokenization)하고 generative, autoregressive 방식으로 다음 토큰 예측(next token prediction) 학습을 진행합니다. 이 모델은 15개의 다양한 외부 데이터셋에 대해 효과적으로 일반화(generalization)됩니다.

- **Performance Highlights**: GluFormer는 전통적인 CGM 분석 도구보다 우수한 임베딩(embedding)을 생성하며, HbA1c, 간 관련 지표, 혈중 지질, 수면 관련 지수를 예측하는 데 높은 Pearson 상관관계를 기록했습니다. 특히, GluFormer는 최대 4년 앞서 미래 건강 결과를 예측할 수 있는 능력이 있습니다. 식이 데이터(dietary data)를 통합할 경우, 이 모델은 정확하게 CGM 데이터를 생성하고, 식이介입의 결과를 시뮬레이션하며, 특정 음식에 대한 개인의 반응을 예측할 수 있습니다.



### Hierarchical Retrieval-Augmented Generation Model with Rethink for Multi-hop Question Answering (https://arxiv.org/abs/2408.11875)
Comments:
          undereview

- **What's New**: 본 논문에서는 Multi-hop Question Answering (QA)에 대한 새로운 프레임워크인 Hierarchical Retrieval-Augmented Generation Model with Rethink (HiRAG)을 제안합니다. 이 모델은 Decomposer, Definer, Retriever, Filter, Summarizer의 다섯 가지 모듈로 구성되어 있으며, 퀘스천의 서브퀘스천을 효과적으로 처리하는 새로운 계층적 검색 전략을 도입합니다.

- **Technical Details**: HiRAG는 다단계 검색을 수행하며, 문서 수준에서의 sparse retrieval과 청크 수준에서의 dense retrieval을 통합하여 두 가지 방법의 장점을 활용합니다. 또한, single-candidate retrieval 방식을 도입하여 다수의 후보 검색의 한계를 극복하고, 부정확한 답변이 발견될 경우 Rethink 모듈을 통해 추가 청크를 선택합니다. 논문에서는 Indexed Wikicorpus와 Profile Wikicorpus라는 두 개의 새로운 데이터베이스도 구성하였습니다.

- **Performance Highlights**: HiRAG는 HotPotQA, 2WikiMultihopQA, MuSiQue, Bamboogle의 네 가지 데이터 세트에서 실험을 수행하였으며, 대부분의 메트릭에서 최첨단 모델을 능가하는 성능을 보였습니다. 특히, Indexed Wikicorpus는 효과적인 데이터베이스로 확인되었습니다.



### MegaFake: A Theory-Driven Dataset of Fake News Generated by Large Language Models (https://arxiv.org/abs/2408.11871)
- **What's New**: 이번 연구는 LLM(대형 언어 모델) 기반의 가짜 뉴스 생성 메커니즘을 분석하고, 이를 지원하는 이론적 프레임워크인 LLM-Fake Theory를 개발했습니다. 가짜 뉴스 생성을 위한 자동화된 파이프라인을 도입하여 수동 주석의 필요성을 없앴으며, MegaFake라는 대규모 기계 생성 가짜 뉴스 데이터셋을 만들었습니다.

- **Technical Details**: LLM-Fake Theory는 네 가지 방법을 통해 LLM을 활용하여 가짜 뉴스를 생성하는 방법을 설명하며, 각 방법은 사회 심리학 이론에 의해 지지됩니다. 이 연구는 GossipCop 데이터셋을 기반으로 46,096개의 가짜 뉴스와 17,871개의 합법적 뉴스를 포함하는 MegaFake 데이터셋을 개발했습니다. 이를 통해 LLM의 가짜 뉴스 생성에 대한 신뢰도를 높였습니다.

- **Performance Highlights**: MegaFake 데이터셋은 가짜 뉴스 탐지 모델들이 일반적으로 나타내는 예측 편향을 줄여주는 것으로 나타났습니다. 실험 결과, 자연어 이해(NLU) 모델이 자연어 생성(NLG) 모델보다 뛰어난 성능을 보였으며, 흥미롭게도 작은 LLM이 대형 LLM보다도 합법 뉴스와 가짜 뉴스를 분류하는 데 더 높은 성능을 나타냈습니다.



### Enhance Lifelong Model Editing with Continuous Data-Adapter Association (https://arxiv.org/abs/2408.11869)
Comments:
          Preprint. Under Review

- **What's New**: ELDER(Enhancing Lifelong moDel Editing with mixtuRe of Low-Rank Adapter)는 반복적인 모델 편집 과정에서 발생하는 지식 손실 문제를 해결하기 위해 최적화된 새로운 접근법입니다. 이 방법은 여러 LoRA(Low-Rank Adapter)를 통합하여 데이터를 원활하게 연결하고, 기존의 모델 수정 방법들의 단점을 보완합니다.

- **Technical Details**: ELDER는 다운스트림 작업에서 LLM의 일반적인 성능을 유지하며, 지속적인 입력을 위한 로터 네트워크를 통해 LoRA를 조합하여 작동합니다. 중요한 점은, LoRA 가중치가 수동으로 설정되는 것이 아니라, end-to-end 학습을 통해 적응적으로 생성된다는 것입니다. 이를 통해 데이터와 적절한 어댑터 간의 관계를 학습합니다.

- **Performance Highlights**: ELDER는 GPT-2 XL과 LLaMA2-7B 벤치마크에서 10% 이상 높은 편집 성능을 기록하며, 주어진 데이터에 대해 더 나은 일반화 성능을 나타냅니다. 또한, 이전 편집을 신뢰성 있게 유지하고, 후속 작업에서도 LLM의 성능을 보장합니다.



### Improving embedding with contrastive fine-tuning on small datasets with expert-augmented scores (https://arxiv.org/abs/2408.11868)
- **What's New**: 이 논문에서는 전문가 점수가 추가된 작은 데이터셋을 통해 텍스트 임베딩 모델을 개선하는 대비 미세 조정 방법론을 제안합니다. 이 방법은 의미적 텍스트 유사성 태스크를 향상시키고 텍스트 검색 문제를 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: 제안된 방법은 전문가 점수를 통해 도출된 소프트 레이블을 사용하여 임베딩 모델을 미세 조정합니다. 이를 통해 모델의 다양성을 유지하면서 모델의 검색 성능이 향상될 수 있도록 합니다. 이 연구에서는 온라인 쇼핑 웹사이트의 Q&A 데이터셋과 8개의 전문가 모델을 사용하여 방법론을 평가했습니다.

- **Performance Highlights**: 논문의 결과는 여러 검색 작업에서 벤치마크 모델에 비해 향상된 성능을 보여주며, 이는 대규모 텍스트 임베딩 벤치마크(MTEB)에서 다양한 메트릭을 기반으로 평가되었습니다. 이 방법은 라벨이 부족한 실제 응용 프로그램에 특히 비용 효율적이고 실용적입니다.



### Crossing New Frontiers: Knowledge-Augmented Large Language Model Prompting for Zero-Shot Text-Based De Novo Molecule Design (https://arxiv.org/abs/2408.11866)
Comments:
          Paper was accepted at R0-FoMo: Robustness of Few-shot and Zero-shot Learning in Foundation Models, NeurIPS-2023. Please find the links: this https URL and this https URL

- **What's New**: 본 연구에서는 대규모 언어 모델(LLMs)을 활용한 텍스트 기반 분자 설계를 위한 새로운 양식을 소개합니다. 이는 지식 보강(knowledge-augmented) 프롬프트를 사용하여 제로샷(Zero-shot) 조건부 de novo 분자 생성을 위한 접근법으로, 기존 방법보다 월등한 성능을 보여줍니다.

- **Technical Details**: 제안된 방법론에서는 LLM들이 생성하는 예측 결과에서 기술적 설명을 이용하여 작은 규모의 언어 모델(예: DeBERTa)을 미세 조정(fine-tuning) 합니다. 이 과정에서 각 모델의 선택적 설명과 기술적 설명을 바탕으로 컨텍스트 인지 토큰 임베딩을 계산합니다. 최종적으로 변환된 임베딩을 통합하여 화학 SMILES 표현을 생성하는 트랜스포머 디코더에 입력합니다.

- **Performance Highlights**: 제안된 프레임워크는 벤치마크 데이터셋에서 기존의 최첨단(SOTA) 모델들을 초과하는 성능을 인정받았으며, 실험 결과가 그 효과성을 뒷받침합니다. 이를 통해 텍스트 기반 분자 설계 작업에서의 새로운 가능성을 제시합니다.



### How Susceptible are LLMs to Influence in Prompts? (https://arxiv.org/abs/2408.11865)
- **What's New**: 이번 연구는 여러 개의 질문-응답 작업에서 LLMs의 프롬프트 민감성을 조사하며, 다른 모델의 추가 입력이 LLM의 응답에 미치는 영향을 분석합니다.

- **Technical Details**: 본 연구에서는 Llama2, Mixtral, Falcon과 같은 여러 LLM 모델이 다른 모델로부터 받은 예측 및 설명을 포함한 경우 어떤 식으로 선택 질문에 대한 응답이 변화하는지를 조사합니다. 특히, 설명의 존재, 출처의 권위성, 추가 입력의 신뢰도가 LLM의 반응에 미치는 영향을 연구합니다.

- **Performance Highlights**: 연구 결과, LLM은 제공된 추가 입력의 품질에 관계없이 강한 영향을 받으며, 특히 해당 입력이 권위있거나 신뢰도로 제시될 경우 더욱 쉽게 설득되는 경향을 보입니다. 이는 LLMs의 신뢰성을 확보하기 위한 중요한 경고를 제시합니다.



### Unraveling Text Generation in LLMs: A Stochastic Differential Equation Approach (https://arxiv.org/abs/2408.11863)
- **What's New**: 이 논문은 Stochastic Differential Equations (SDEs)를 사용하여 Large Language Models (LLMs), 특히 GPT-4의 텍스트 생성 과정을 해석하는 새로운 방법을 탐구합니다. 텍스트 생성 과정은 확률적 프로세스로 모델링되며, 각 단계는 이전에 생성된 콘텐츠와 모델 매개변수에 의존하여 다음 단어를 선택합니다.

- **Technical Details**: SDE는 텍스트 생성 과정의 결정론적 경향(drfit term)과 확률적 변동(diffusion term)을 함께 포착하는 수학적 구조를 제공합니다. 이 모델은 신경망(neural networks)을 사용하여 학습하고 실제 텍스트 자료를 기반으로 검증됩니다.

- **Performance Highlights**: 본 연구를 통해 LLM의 동력을 깊이 있게 이해할 수 있으며, 이는 생성된 텍스트의 품질을 진단하고 최적화하는 데 중요한 기여를 합니다. SDE 기반 접근 방식을 통해 LLM의 내재된 특성과 변동성을 더욱 잘 설명할 수 있습니다.



### Sentiment analysis of preservice teachers' reflections using a large language mod (https://arxiv.org/abs/2408.11862)
Comments:
          5 pages, 2 tables, WAIE 2024 (2024 6th International Workshop on Artificial Intelligence and Education)

- **What's New**: 본 연구에서는 예비 교사들의 반성(Reflection)에 대한 감정(Emotion)과 톤(Tone)을 LLMs(GPT-4, Gemini, BERT)를 활용한 감정 분석(Sentiment Analysis)으로 분석하였습니다. 각 도구가 개별 반성과 여러 반성을 어떻게 분류하고 서술하는지를 비교하였습니다.

- **Technical Details**: 연구는 교사 교육(Teacher Education)에서의 반성적인 실천(Reflective Practices)에 대한 질적(Qualitative), 양적(Quantitative), 계산적(Computational) 분석 간의 간극을 메우는 방법을 탐구하고자 합니다. LLM 분석을 효과적으로 통합하기 위해서는 예비 교사 및 교사 교육자를 위한 포괄적(Comprehensive)이며 관련성(Relevant)이 있는 분석 방법 및 결과 형식 개발이 중요합니다.

- **Performance Highlights**: 이 연구의 결과는 예비 교사의 반성에 대한 LLM의 분석이 어떻게 이루어지는지를 이해하고, 교사 교육의 실천에 적용할 수 있는 가능성을 제시합니다.



### Speaking the Same Language: Leveraging LLMs in Standardizing Clinical Data for AI (https://arxiv.org/abs/2408.11861)
Comments:
          11 pages, 2 figures, 4 tables

- **What's New**: 본 연구에서는 의료 데이터 표준화를 위한 대형 언어 모델(Large Language Models, LLM)을 활용하여 AI 통합 과정을 용이하게 하고 환자 관리 품질을 향상시키는 방법을 제안합니다.

- **Technical Details**: 연구는 Fast Healthcare Interoperability Resources (FHIR) 표준을 기반으로 하여 LLM을 사용한 데이터 매핑 과정을 상세히 설명합니다. 여기에서 우리는 LLM이 데이터를 일관된 형식으로 표준화하는 데 있어 어떻게 기여할 수 있는지를 보여줍니다.

- **Performance Highlights**: LLM을 활용한 결과, 수작업 데이터 큐레이션의 필요성을 대폭 줄이고 데이터 표준화 과정의 효율성을 높일 수 있음을 입증하였습니다. 이는 AI가 의료 분야에 통합되는 속도를 가속화하고, 필요한 시간 및 재정 자원을 최소화하는 데 기여합니다.



### Dynamic Adaptive Optimization for Effective Sentiment Analysis Fine-Tuning on Large Language Models (https://arxiv.org/abs/2408.11856)
- **What's New**: 본 연구는 다이나믹 어댑티브 최적화(DAO) 모듈이 포함된 새로운 다중 작업(multi-task) 학습 프레임워크를 제안합니다. 이 모듈은 기존 모델에 원활하게 통합할 수 있는 플러그 앤 플레이 구성 요소로 설계되었습니다.

- **Technical Details**: DAO 모듈의 핵심 요소는 동적 어댑티브 손실(dynamic adaptive loss)로, 이는 훈련 중 각 작업의 상대적 중요성과 데이터의 특성에 따라 다른 작업에 할당된 가중치를 동적으로 조정합니다.

- **Performance Highlights**: 제안된 프레임워크는 평균 제곱 오차(Mean Squared Error, MSE)와 정확도(Accuracy, ACC)를 각각 15.58% 및 1.24% 향상시켰습니다. 이로 인해 기존 연구에 비해 뛰어난 성능을 보여주었습니다.



### FactorLLM: Factorizing Knowledge via Mixture of Experts for Large Language Models (https://arxiv.org/abs/2408.11855)
- **What's New**: 최근 연구에 따르면, 대형 언어 모델(LLM)에서 Feed-Forward Networks(FFN)가 다양한 언어 및 사실적 지식을 저장하는 데 중요한 역할을 한다고 합니다. 기존 방법들은 단일한 구조와 중복된 아키텍처로 인해 지식 혼란을 겪고 있으며, 이에 따라 LLM에 대한 보다 효율적이고 컴퓨팅 자원을 최소화하는 해결책이 요구되고 있습니다. 본 논문에서는 FFN 계산 패러다임을 탐구하고, FactorLLM을 소개하여 훈련된 FFN을 더욱 효율적으로 분해합니다.

- **Technical Details**: FactorLLM은 훈련된 밀집형 FFN을 수정 없이 희소한 서브 네트워크로 분해하는 새로운 접근 방식을 제안합니다. 이 과정에서 Mixture-of-Experts(MoE) 아키텍처의 라우터를 통합하여 전문가의 동적 활성화와 지식 적응을 촉진하고, 최소한의 훈련 데이터와 미세 조정 단계로 성능을 향상시킵니다. 또한 Prior-Approximate(PA) 손실 항을 도입하여 LLM의 효율성을 높입니다.

- **Performance Highlights**: FactorLLM은 다양한 벤치마크에서 실험을 통해 이러한 새로운 접근 방식이 기존의 기법들보다 30% 이상의 연산 비용 절감을 이루면서도 원 모델의 성능 85%를 유지했다는 것을 입증했습니다. 이러한 결과는 자원 제약이 있는 상황에서도 빠른 배치를 가능하게 합니다.



### When Raw Data Prevails: Are Large Language Model Embeddings Effective in Numerical Data Representation for Medical Machine Learning Applications? (https://arxiv.org/abs/2408.11854)
Comments:
          Under review

- **What's New**: 이번 연구는 Large Language Models (LLMs)의 마지막 은닉 상태에서 도출된 벡터 표현을 전자 건강 기록(EHR) 데이터를 활용하여 의학 진단 및 예측에 대한 효과를 분석했습니다. LLM이 생성한 특징을 전통적인 기계 학습 알고리즘의 입력으로 사용했을 때의 성능을 비교하여, ML 작업에서 원시 데이터 입력이 여전히 우세함을 확인했습니다.

- **Technical Details**: 의료 ML 작업에 필요한 표시된 EHR 숫자 데이터 표현에 대한 LLM을 활용하였으며, eXtreme Gradient Boosting(XGB)와 같은 전통적인 ML 알고리즘과의 성능 비교를 수행했습니다. 연구에서는 zero-shot LLMs의 특징 추출기 활용과 프롬프트 엔지니어링 기법들을 포함하여 다양한 조건과 embedding 메소드의 영향 평가를 실시했습니다.

- **Performance Highlights**: 결과적으로, LLM 특징과 XGB 분류기를 결합한 경우 일부 작업에서 전통적인 원시 데이터 특징과 유사한 성능을 달성했지만, 여전히 성능 차이가 존재하여 진행 중인 연구가 필요하다는 필요성을 강조했습니다.



### Fast Training Dataset Attribution via In-Context Learning (https://arxiv.org/abs/2408.11852)
- **What's New**: 본 논문은 in-context learning 및 prompt engineering 기법을 활용하여 instruction-tuned large language models (LLMs)에서 훈련 데이터의 기여도를 추정하는 두 가지 새로운 접근 방식을 제안합니다.

- **Technical Details**: (1) similarity-based 접근 방식은 제공된 컨텍스트가 있는 LLM 출력과 없는 LLM 출력을 비교하여 차이를 측정합니다. (2) mixture distribution model 접근 방식은 기여 점수를 식별하는 문제를 행렬 분해(matrix factorization) 문제로 변형합니다.

- **Performance Highlights**: 실험 결과, mixture model 접근 방식이 RAG 시스템 내에서의 검색 노이즈에 대해 더 강건하며, 데이터를 기여도를 보다 정확히 추정하는 데 효과적임을 입증했습니다.



### Style-Talker: Finetuning Audio Language Model and Style-Based Text-to-Speech Model for Fast Spoken Dialogue Generation (https://arxiv.org/abs/2408.11849)
Comments:
          CoLM 2024

- **What's New**: 이 논문에서는 Style-Talker라는 혁신적인 프레임워크를 소개합니다. 이 시스템은 음성 입력을 기반으로 한 빠른 대화 생성을 가능하게 하며, 오디오 LLM과 스타일 기반 TTS 모델을 함께 조정하여 사용자의 입력 음성을 처리합니다.

- **Technical Details**: Style-Talker는 자동 음성 인식 (ASR), 대화 생성, 텍스트-음성 변환 (TTS) 모델을 통합하여 실시간 대화 시스템을 구현합니다. 이 시스템은 입력 음성에서 기계적으로 얻어진 텍스트 및 스타일 정보를 활용하여 응답을 생성하며, 이후 TTS 모델을 통해 음성을 합성합니다.

- **Performance Highlights**: 실험 결과, Style-Talker는 기존의 캐스케이드 ASR-LLM-TTS 시스템보다 50% 이상 빠르며, 자연스러움과 일관성 측면에서도 우수한 성능을 보였습니다. 이 시스템은 실제 데이터셋에서 직접 활용할 수 있어 다양한 적용 가능성을 높였습니다.



### MGH Radiology Llama: A Llama 3 70B Model for Radiology (https://arxiv.org/abs/2408.11848)
Comments:
          11 pages, 3 figures, 1 table

- **What's New**: 이번 논문에서는 MGH Radiology Llama라는 첨단 방사선학 중심의 대형 언어 모델(Large Language Model, LLM)을 소개합니다. 이 모델은 Llama 3 70B 모델을 기반으로 하여, 방사선학에 특화된 보고서 생성, 임상 결정 지원 및 환자 커뮤니케이션 도움을 제공합니다. 650만 개 이상의 비식별화된 의료 보고서로 학습하였으며, 진단 정확도와 병원 근무 효율성을 향상시키는 데 중점을 두고 있습니다.

- **Technical Details**: 모델은 Massachusetts General Hospital(MGH)의 다각화된 데이터셋을 기반으로 하고 있으며, CT, MRI, X-ray 및 Fluoroscopic 이미지를 포함한 다양한 영상 모달리티의 데이터를 포함합니다. 데이터 전처리와 함께, Llama 3 70B 모델의 완전한 미세 조정(fully fine-tuning) 및 QLoRA 미세 조정 방법을 사용하여 최적화를 진행합니다.

- **Performance Highlights**: 전통적인 평가 지표인 ROUGE 및 BERTScore와 함께 GPT-4 기반 평가를 결합하여 모델의 성능을 강조하였습니다. 성능 평가 결과, 기존 일반 언어 모델에 비해 방사선학에 특화된 더욱 정확하고 임상적으로 유용한 인상을 생성하는 성능 개선이 확인되었습니다.



### Editable Fairness: Fine-Grained Bias Mitigation in Language Models (https://arxiv.org/abs/2408.11843)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2405.09341

- **What's New**: 이번 논문에서는 공정하고 정확한 예측을 생성하는 것이 대형 언어 모델(LLMs)의 실제 적용에 중요하다고 강조하고 있으며, 새로운 편향 완화 벤치마크인 BiaScope를 제정하고, 개인의 사회적 편향을 세분화된 방식으로 보정하는 Fairness Stamp(FAST) 방식을 제안합니다.

- **Technical Details**: BiaScope는 기존의 편향 제거 접근법이 사회적 집단 간의 동등성을 달성하는 데 초점을 맞췄지만, 개인의 상식적인 사실을 간과하면서 부정확한 예측을 초래함을 지적합니다. FAST는 LLM에서 사회적 편향을 저장하는 결정적인 레이어를 식별하고, 작은 모듈 네트워크를 통합하여 편향을 완화합니다.

- **Performance Highlights**: 실험 결과, FAST는 최신 방법론들에 비해 우수한 편향 제거 성능을 보이면서도 모델의 지식 보존 및 하위 예측 능력을 저해하지 않음이 입증되었습니다. 이는 LLM의 공정성을 달성하기 위한 세분화된 편향 제거 전략의 가능성을 보여줍니다.



### Could ChatGPT get an Engineering Degree? Evaluating Higher Education Vulnerability to AI Assistants (https://arxiv.org/abs/2408.11841)
Comments:
          20 pages, 8 figures

- **What's New**: 본 연구에서는 AI 어시스턴트가 고등 교육에서 학생들의 평가 및 학습 결과에 미치는 영향을 분석하고 있으며, 특히 생성 AI(generative AI)의 영향을 바탕으로 대학 평가의 취약성(vulnerability)을 개념화하고 있습니다.

- **Technical Details**: EPFL(École Polytechnique Fédérale de Lausanne)에서 제공하는 50개 과목의 텍스트 기반 평가 질문 데이터셋을 구축하였으며, 두 개의 모델인 GPT-3.5와 GPT-4의 성능을 측정하였습니다. 총 5,579개의 개방형 응답 질문(open-answer questions)과 객관식 질문(multiple-choice questions)으로 구성되어 있으며, 자동 및 수동 평가(automatic and manual grading) 방식으로 분석하였습니다.

- **Performance Highlights**: GPT-4는 평균적으로 65.8%의 질문에 올바른 답변을 제공할 수 있으며, 적어도 85.1%의 질문에 대해 최소 하나의 올바른 답변을 생성할 수 있었습니다. 이는 다양한 학문 분야의 과목에 걸쳐 상대적으로 안정적인 성과를 보이며, 여러 대학 프로그램에서 상당한 수준의 취약성을 나타냅니다.



### Joint PET-MRI Reconstruction with Diffusion Stochastic Differential Mod (https://arxiv.org/abs/2408.11840)
Comments:
          Accepted as ISMRM 2024 Digital poster 6575. 04-09 May 2024 Singapore

- **What's New**: 본 논문에서는 PET와 MRI의 통합 이미지 재구성을 개선하기 위한 새로운 접근 방식인 확산 확률 미분 방정식(diffusion stochastic differential equations)을 기반으로 한 모델을 제안합니다.

- **Technical Details**: 저자들은 PET과 MRI의 공동 확률 분포(joint probability distribution)를 학습함으로써 두 가지 이미지를 동시에 재구성하는 방법을 개발했습니다. 이 방법은 PET-MRI 시스템에서 저조한 신호 대 잡음 비율(signal-to-noise ratio) 문제와 MRI의 고가격 획득 시간을 개선하는 데 필요한 기술적 접근을 포함하고 있습니다.

- **Performance Highlights**: 이 모델은 기존의 최첨단 방법론들을 초월하여 PET 및 MRI 재구성에서 질적(qualitative) 및 양적(quantitative) 개선을 보여주었습니다. 이로 인해 PET-MRI 시스템 내에서의 재구성 도전 과제를 효과적으로 해결할 수 있음을 입증하였습니다.



### Adaptive Friction in Deep Learning: Enhancing Optimizers with Sigmoid and Tanh Function (https://arxiv.org/abs/2408.11839)
- **What's New**: 본 연구에서는 심층 신경망의 가중치 업데이트를 개선하기 위해 새로운 적응형 최적화 알고리즘인 sigSignGrad와 tanhSignGrad를 소개합니다. 이 알고리즘은 Sigmoid 및 Tanh 함수에 기반한 적응형 마찰 계수를 통합하여 성능을 향상시킵니다.

- **Technical Details**: sigSignGrad와 tanhSignGrad는 전통적인 Adam 변형인 diffGrad 및 AngularGrad에서 간과된 단기 기울기 정보(short-term gradient information)를 활용하여 매개변수 업데이트(parameter update)와 수렴(convergence)을 개선합니다. 이론적 분석을 통해 마찰 계수 S의 조정 능력을 입증하였으며, 이를 통해 최적화 경로의 부드러움(smoothness)과 수렴 속도(convergence rate)를 향상시켰습니다.

- **Performance Highlights**: CIFAR-10, CIFAR-100 및 Mini-ImageNet 데이터셋에서 ResNet50 및 ViT 아키텍처를 사용하여 광범위한 실험을 진행한 결과, 제안한 최적화 알고리즘이 기존 방법보다 더 우수한 정확도를 보였고 훈련 시간을 단축시켰습니다. 적응형 마찰 계수를 기존 최적화 알고리즘의 플러그인 형태로 통합하는 혁신적인 접근 방식이 최적화 성능을 향상시키는 전략으로 주목받고 있습니다.



### MicroXercise: A Micro-Level Comparative and Explainable System for Remote Physical Therapy (https://arxiv.org/abs/2408.11837)
Comments:
          Accepted by IEEE/ACM CHASE 2024

- **What's New**: MicroXercise는 웨어러블 센서를 이용한 마이크로 모션 분석을 통합하여 물리치료에 대한 실시간 피드백을 제공하는 혁신적인 시스템입니다. 이를 통해 치료사와 환자에게 비디오, 텍스트 및 점수를 포함한 종합적인 피드백 인터페이스를 제공합니다.

- **Technical Details**: MicroXercise는 다차원 동적 시간 왜곡(DTW)과 설명 가능한 방법들을 활용해 기존의 딥러닝 신경망을 분석하고, 운동의 고해상도 특징을 모니터링합니다. 이 시스템은 세밀한 마이크로 모션을 강조하여 피드백의 해석 가능성을 높입니다.

- **Performance Highlights**: MicroXercise의 성능은 전통적인 방법에 비해 39%와 42% 개선된 Feature Mutual Information (FMI) 및 Continuity를 기록하며, 환자의 운동에 대한 이해도를 크게 향상시킵니다. 이러한 접근은 개인 맞춤형 물리치료 솔루션으로서의 잠재력을 보여줍니다.



### SCREENER: A general framework for task-specific experiment design in quantitative MRI (https://arxiv.org/abs/2408.11834)
- **What's New**: SCREENER라는 새로운 프레임워크를 제안하여 quantitative MRI (qMRI) 실험 설계를 특정 임상 작업에 맞게 최적화할 수 있도록 하였습니다. 이 프레임워크는 deep reinforcement learning (DRL) 기반의 최적화 전략을 포함하고 있으며, 기존의 임시 방편적 방법과 CRLB 최적화 방법보다 우수한 성능을 보여줍니다.

- **Technical Details**: SCREENER는 특정 임상 과제를 위한 최적의 프로토콜을 설계하는 데 두 가지 주 요소를 포함합니다: (1) 과제 특정 목표 모듈, (2) 드릴 기반 최적화 모듈. 이를 통해 데이터 수집 조건을 최적화하고, 특히 균질 조사를 통해 뼈 수염의 염증 상태 분류 작업에 활용됩니다.

- **Performance Highlights**: 실험 결과, SCREENER는 기존 방법보다 이진 분류 작업에서 67%에서 89%로, 다중 클래스 분류 작업에서 46%에서 59%로 성능을 크게 향상시켰습니다. 또한, SNR의 변화에 강건하게 작동함을 보여주었으며, DRL 기반 최적화 전략을 통해 훈련에 사용되지 않은 다양한 SNR에 대해 제로샷 발견이 가능함을 입증하였습니다.



### OpenFactCheck: A Unified Framework for Factuality Evaluation of LLMs (https://arxiv.org/abs/2408.11832)
Comments:
          10 pages, 4 Figures, 3 Tables, Submitted to EMNLP 2024 System Demonstration. arXiv admin note: substantial text overlap with arXiv:2405.05583

- **What's New**: OpenFactCheck는 LLM의 사실성(factuality)을 평가하는 통합 프레임워크로 세 가지 모듈로 구성된다: (i) RESPONSEEVAL, (ii) LLMEVAL, (iii) CHECKEREVAL. 이는 자동 사실 확인 시스템을 사용자 맞춤형으로 만들 수 있게 하여, LLM의 출력에서 발생하는 '환각(hallucination)' 문제를 해결하는 데 도움을 준다.

- **Technical Details**: OpenFactCheck의 세 가지 모듈은 서로 밀접하게 통합되어 사용되며, 각각의 기능이 강화를 목적으로 설계되었다. RESPONSEEVAL은 사용자 맞춤형 사실 확인 시스템을 생성하게 하고, LLMEVAL은 LLM의 전반적인 사실성을 여러 벤치마크를 통해 평가하며, CHECKEREVAL은 자동 사실 확인 시스템의 정확성을 평가한다. 사용자들은 특정 필요에 따라 자신만의 확인기를 구성할 수 있다.

- **Performance Highlights**: OpenFactCheck는 오픈 소스 라이브러리와 웹 서비스를 제공하며, 사용자가 맞춤형 사실 확인기를 설계하고 LLM의 출력의 사실성을 쉽게 평가할 수 있도록 함으로써 향후 LLM 사실성 연구의 발전을 촉진할 것으로 기대된다.



### Online Electric Vehicle Charging Detection Based on Memory-based Transformer using Smart Meter Data (https://arxiv.org/abs/2408.11828)
- **What's New**: 이 논문에서는 전력망의 통합에 있어 전기차(EV) 충전의 실시간 상태를 식별하기 위한 메모리 기반 변환기(M-TR) 모델을 제안합니다. 이는 기존의 감독 학습 모델과는 달리, 데이터 불균형 문제를 해결할 필요가 없는 새로운 비감독 학습 접근 방식을 제공합니다.

- **Technical Details**: 제안된 M-TR 모델은 스트리밍 스마트 미터 데이터에서 EV 충전 사건을 동적으로 식별합니다. 이 모델은 장기 및 단기 정보를 활용하여 충전 패턴을 포착합니다. M-TR 인코더는 전역의 시간 창을 활용하고, 디코더는 제한된 시간 프레임(국소 창)에 집중하여 세밀한 특성을 캡처합니다. 비정상 감지 기술을 기반으로 하며, EV 충전 프로파일에 대한 사전 지식이 필요 없습니다.

- **Performance Highlights**: M-TR 모델은 1분 스마트 기록에 대해 1.2초의 우수한 실행 시간을 기록하며, 여러 최신 방법들과 비교하였을 때, 다른 비감독 학습 모델보다 우수한 성능을 나타냅니다.



### Generative Organizational Behavior Simulation using Large Language Model based Autonomous Agents: A Holacracy Perspectiv (https://arxiv.org/abs/2408.11826)
- **What's New**: 이 논문은 Large Language Model 기반의 자율 에이전트를 활용하여 홀라크라시(Holacracy) 조직을 위한 생성적 시뮬레이션 프레임워크인 CareerAgent의 기술적 세부사항과 주기적 발견을 제시합니다.

- **Technical Details**: CareerAgent 시뮬레이션 프레임워크는 건설, 실행, 평가의 세 가지 단계로 구성됩니다. 각 단계에서는 개인, 조직, 작업, 회의의 기본 특성이 통합되어 있으며, Large Language Model(LLM)의 시뮬레이션 능력을 활용하여 홀라크라시의 운영을 시뮬레이션합니다. 시뮬레이션 실험은 8주간 수행되었으며 주단위로 작업이 발행되고 청구되었습니다.

- **Performance Highlights**: 조직 수준에서는 관리 역량 및 기능적 역량의 평균 값이 전반적인 스트레스 수준을 감소시키면서도 작업 완료와 같은 깊은 조직 성과 측면에서는 부정적인 영향을 미치는 것으로 나타났습니다. 개인 수준에서도 두 가지 역량 모두 구성원들의 작업 수행을 개선할 수 있으며, 높은 역량을 가진 구성원들은 특정 작업에 선택적으로 참여하며 더 많은 책임을 맡는 경향을 보였습니다.



### Strategic AI adoption in SMEs: A Prescriptive Framework (https://arxiv.org/abs/2408.11825)
- **What's New**: 이 연구는 중소기업(SMEs)의 AI(Artificial Intelligence) 기술 채택을 위한 구조적이고 단계적인 프레임워크(framework)를 제안합니다. 이 프레임워크는 비용, 기술 부족, 직원 수용 등 주요 장벽을 해결하는 방법을 체계적으로 제시합니다.

- **Technical Details**: 제안된 프레임워크는 다음의 단계로 진행됩니다: 1단계: 리더십의 인식 제고 및 지원 확보, 2단계: 저비용의 범용 AI 도구 도입을 통해 기술 역량 구축 및 긍정적 태도 조성, 3단계: 효율성 및 생산성 향상을 위한 작업 특화 AI 도구 통합, 4단계: 자사 개발의 생성적 AI 도구 개발, 5단계: 고유한 정밀 작업을 충족시키기 위한 판별적 AI 모델 개발.

- **Performance Highlights**: 이 프레임워크는 중소기업이 AI 통합의 복잡성을 효과적으로 탐색할 수 있도록 하고, 혁신, 효율성, 경쟁 우위를 증대시키는 데 기여합니다. 또한, AI 기술의 성공적인 채택을 통해 지속 가능한 성장 가능성을 모색하는 데 도움을 줍니다.



### AppAgent v2: Advanced Agent for Flexible Mobile Interactions (https://arxiv.org/abs/2408.11824)
- **What's New**: 본 연구는 모바일 기기를 위한 새로운 LLM 기반 멀티모달 에이전트 프레임워크를 소개합니다. 이 프레임워크는 인간과 유사한 상호작용을 모방하며, 다양한 애플리케이션에서 적응력을 높이는 유연한 액션 공간을 제공합니다.

- **Technical Details**: 이 에이전트는 탐색(exploration) 및 배포(deployment)의 두 가지 주요 단계로 운영됩니다. 탐색 단계에서 사용자 인터페이스(UI) 요소들의 기능을 문서화하며, 배포 단계에서는 RAG( retrieval-augmented generation) 기술을 활용해 지식 기반에서 효율적인 정보 검색과 업데이트를 수행합니다.

- **Performance Highlights**: 다양한 벤치마크를 대상으로 한 실험 결과, 본 프레임워크의 우수한 성능이 입증되었습니다. 이를 통해 실세계 환경에서의 효과적인 작업 수행과 사용자 친화성을 검증했습니다.



### Mamba-Spike: Enhancing the Mamba Architecture with a Spiking Front-End for Efficient Temporal Data Processing (https://arxiv.org/abs/2408.11823)
Comments:
          12 pages, 5 figures, accepted by CGI 2024

- **What's New**: 본 논문에서는 Mamba-Spike라는 새로운 뉴로모픽 아키텍처를 소개하며, 이는 스파이킹 신경망(SNN)을 기반으로 한 전방 처리 모듈과 Mamba 백본을 통합하여 시간적 데이터 처리를 효율적이고 강인하게 수행하는 것을 목표로 합니다.

- **Technical Details**: Mamba-Spike 아키텍처는 스파이킹 전방 모듈과 Mamba 백본의 두 가지 주요 구성 요소로 이루어져 있습니다. 스파이킹 전방 모듈은 비동기적 시간 변동 입력을 효율적으로 캡처하고 처리하게 설계된 생물학적으로 영감을 받은 신경 모델을 활용하며, Mamba 백본은 선택적인 상태 공간과 선형 시간 시퀀스 모델링을 활용하여 복잡한 시간적 종속성을 효과적으로 모델링합니다.

- **Performance Highlights**: Mamba-Spike는 DVS Gesture, TIDIGITS와 같은 뉴로모픽 데이터셋과 Sequential MNIST 및 CIFAR10-DVS와 같은 표준 데이터셋에서 종합적인 실험을 실시하여 state-of-the-art 기준 모델들을 일관되게 능가했습니다. 높은 정확도, 낮은 지연 시간, 개선된 에너지 효율성을 달성하며, 다양한 입력 교란에 대해서도 강인성을 보여 실제 응용 가능성을 강조합니다.



### State-of-the-art in Robot Learning for Multi-Robot Collaboration: A Comprehensive Survey (https://arxiv.org/abs/2408.11822)
Comments:
          Multi-robot, Cooperation, robot learning

- **What's New**: 최근 로봇 기술의 발전과 인공지능(Artificial Intelligence)의 융합으로 다중 로봇 시스템(Multi-Robot Systems, MRS)이 인간의 일상 생활에 점점 통합되고 있다. 이 논문은 로봇 학습(Robot Learning)과 MRS의 최신 연구 동향을 종합적으로 살펴본다.

- **Technical Details**: MRS는 여러 대의 로봇이 협력 및 경쟁을 통해 특정 작업을 수행하는 시스템으로, Reinforcement Learning (RL), Transfer Learning (TL), Imitation Learning (IL) 등의 기술이 포함된다. 이들 학습 메커니즘의 설계 원리를 탐구하고, 적용 사례 및 직면한 도전 과제들을 분석한다.

- **Performance Highlights**: MRS는 자동화된 물류, 검색 및 구조 작업, 환경 모니터링, 정밀 농업 등 다양한 분야에서 효과적임을 입증했으며, 로봇 학습 기술이 실제 문제 해결에 미치는 긍정적인 영향을 확인하였다.



### Responsible AI Question Bank: A Comprehensive Tool for AI Risk Assessmen (https://arxiv.org/abs/2408.11820)
Comments:
          30 pages, 6 tables, 14 figures

- **What's New**: 이 논문은 AI의 책임 있는 개발과 사용을 지원하기 위한 'Responsible AI (RAI) Question Bank'를 소개합니다. 이 도구는 AI 윤리 원칙(공정성, 투명성, 책임성 등)을 체계적인 질문 형식으로 통합하여 AI 프로젝트의 잠재적 위험을 식별하고, 유럽의 AI 법안(EU AI Act)과 같은 새로운 규정에 부합하도록 돕습니다.

- **Technical Details**: RAI Question Bank는 구조화된 질문 형식을 통해 다양한 AI 이니셔티브를 지원하며, 하위 수준의 위험 질문을 상위 수준 질문 및 관련 주제와 연결하는 체계적인 접근 방식을 제공합니다. 이로 인해 단절된 평가를 방지하고 응집력 있는 평가 프로세스를 보장할 수 있습니다.

- **Performance Highlights**: 케이스 스터디를 통해 RAI Question Bank의 실제 응용 사례를 보여줍니다. 이 도구는 AI 프로젝트의 위험 요소를 평가하고 의사 결정 과정에 정보를 제공하며, 규정 준수를 보장하고 신뢰할 수 있는 AI 시스템 개발을 촉진하는 데 기여합니다.



### Is ChatGPT a Good Software Librarian? An Exploratory Study on the Use of ChatGPT for Software Library Recommendations (https://arxiv.org/abs/2408.05128)
Comments:
          Submitted

- **What's New**: 본 논문은 ChatGPT를 소프트웨어 라이브러리 추천 도구로 평가하여 개선 가능성을 제시합니다. LLMs의 효과를 실증적으로 분석하여, 개발자들이 소프트웨어 라이브러리를 선택하는 과정에서의 어려움을 조명합니다.

- **Technical Details**: 이 연구는 10,000개의 Stack Overflow 질문에 대해 Python 코드를 생성하는 경험적 연구를 포함합니다. GPT-3.5 Turbo를 사용하여 생성된 코드를 분석하였으며, ChatGPT는 인간 개발자보다 서드파티 라이브러리를 10% 더 자주 사용하는 것으로 나타났습니다. 그러나 14.2%의 추천 라이브러리는 제한적인 copyleft 라이센스가 있었습니다.

- **Performance Highlights**: 결과적으로, ChatGPT가 추천한 라이브러리의 6.5%는 설치 실패를 초래하였으며, 이는 모듈의 암묵적 임포트나 플레이스홀더 제안으로 인한 문제 때문이었습니다. 이러한 결과는 LLMs가 소프트웨어 라이브러리 추천에서 활용될 수 있지만, 라이브러리에 대한 명확한 정보 제공과 신뢰성 강화를 요구합니다.



### On the Variability of AI-based Software Systems Due to Environment Configurations (https://arxiv.org/abs/2408.02825)
Comments:
          Submitted to the Information and Software Technology journal for review

- **What's New**: 이 논문은 환경 구성의 관점에서 AI 기반 시스템의 변동성을 조사한 최초의 경험적 연구이다. 다양한 환경 구성 설정이 AI 시스템의 변동성에 미치는 영향을 실증적으로 입증하였다.

- **Technical Details**: 연구에서는 30개의 오픈소스 AI 시스템에서 운영 체제, Python 버전, CPU 아키텍처와 같은 3가지 주요 환경 변수를 조합한 8개의 다양한 구성으로 실험을 수행하였다. 변동성은 AI 구성 요소의 출력 성능, 빌드 및 실행 시간, 비용 등 3가지 지표를 사용하여 평가되었다.

- **Performance Highlights**: 결과적으로 모든 지표에서 변동성이 존재했으나, 성능보다는 빌드 및 실행 시간과 비용에서 더 빈번하게 관찰되었다. 예를 들어, Linux와 MacOS 간의 성능에서는 23%, 처리 시간에서 96.67%, 비용에서는 100%의 변동이 발생했다.



### Predicting the First Response Latency of Maintainers and Contributors in Pull Requests (https://arxiv.org/abs/2311.07786)
Comments:
          Manuscript accepted for publication in IEEE Transactions on Software Engineering (TSE)

- **What's New**: 이 논문에서는 Pull Request (PR)의 첫 번째 응답 대기 시간을 예측하기 위한 기계 학습 접근법을 제안합니다. 이에 따라 유지 관리자의 첫 번째 응답 지연 시간과 유지 관리자에게서 첫 번째 응답을 받은 후 기여자의 첫 번째 응답 지연 시간을 예측할 수 있습니다. 이를 통해 기여자와 유지 관리자 간의 상호작용을 향상시키고 기대치를 관리할 수 있습니다.

- **Technical Details**: 기계 학습 모델은 GitHub의 20개 오픈 소스 프로젝트에서 수집한 데이터셋을 기반으로 하여 구축하였습니다. 21개의 특징(feature)을 추출하여 PR, 기여자, 유지 관리자의 응답 과정을 분석했습니다. 7가지 분류기(classifier) 유형을 평가하여 CatBoost 모델이 가장 효과적임을 확인하였고, 중요성과 영향력을 평가하기 위해 permutation feature importance와 SHAP 분석도 수행했습니다.

- **Performance Highlights**: CatBoost 모델은 유지 관리자 및 기여자의 첫 번째 응답 지연 시간을 예측하는 데 있어 각각 29%와 39%의 AUC-ROC에서 평균 향상을 기록하였습니다. 또한, 기여자의 응답 시간에 대한 중요한 예측 변수로는 PR 제출 시점, 커밋 수, 기여자의 수락률 등이 있었다고 보고했습니다.



### Understanding the Helpfulness of Stale Bot for Pull-based Development: An Empirical Study of 20 Large Open-Source Projects (https://arxiv.org/abs/2305.18150)
Comments:
          Manuscript submitted to ACM Transactions on Software Engineering and Methodology

- **What's New**: GitHub의 Stale bot이 비활성 Pull Requests (PRs)를 자동으로 추적하고 종료하는 데 도움을 준다는 내용의 실증 연구.

- **Technical Details**: 20개의 대규모 오픈소스 프로젝트를 분석하여 Stale bot 사용의 긍정적 및 부정적 영향을 검토했습니다. PRs의 종료와 검토 효율성이 증가했지만, 활성 기여자의 수는 감소했습니다.

- **Performance Highlights**: Stale bot의 채택 후 몇 달 내에 프로젝트가 비활성 PRs를 더 많이 종료하고, PR 검토의 속도도 증가한 것이 관찰되었습니다. 그러나, 기여자 수의 감소는 커뮤니티의 참여 감소와 기여자 이탈의 가능성을 높였습니다.



### On Wasted Contributions: Understanding the Dynamics of Contributor-Abandoned Pull Requests (https://arxiv.org/abs/2110.15447)
Comments:
          Manuscript accepted for publication in ACM Transactions on Software Engineering and Methodology (TOSEM)

- **What's New**: 이 연구는 오픈 소스 프로젝트에서 contributor(기여자)가 PR(pull request)을 포기하는 동력을 심층적으로 이해하기 위한 혼합 방법론(mixed-methods study)을 적용했습니다. 전체 265,325개의 PR 중 4,450개의 포기된 PR을 분석하여 기여자와 유지 관리자의 노력을 소중하게 만드는 시스템적 문제를 조망합니다.

- **Technical Details**: 연구팀은 16개의 특징(features)을 측정하고, 통계(statistical) 및 머신러닝(machine learning) 기법을 사용해 복잡한 PR, 초급 기여자, 그리고 긴 리뷰가 포기의 가능성을 높인다는 사실을 밝혀냈습니다. 또한, 354개의 포기된 PR을 랜덤 샘플링하여 수동 검토를 통해 기여자들이 직면한 장애물과 유지 관리자에 의해 발주된 장벽을 확인했습니다.

- **Performance Highlights**: 프로젝트의 성숙도나 작업량에 따라서 PR 포기율이 변동하며, 기여자와 유지 관리자 간의 협업에서 발생하는 주요 문제점을 밝혀냈습니다. 마지막으로, 연구팀은 해당 프로젝트의 핵심 유지 관리자들에게 PR 포기 문제에 대한 그들의 시각을 조사하여 통찰을 강화했습니다.



### GAP2WSS: A Genetic Algorithm based on the Pareto Principle for Web Service Selection (https://arxiv.org/abs/2109.10430)
- **What's New**: 이 논문은 웨브 서비스 선택(Web service selection)에서 기존 방법들보다 더 나은 최적성과 성능을 제공하는 새로운 접근법인 GAP2WSS(Genetic Algorithm for Pareto-based Web Service Selection)를 제안합니다.

- **Technical Details**: GAP2WSS는 Pareto 원리를 채택하여 복합 웨브 서비스(composite Web service)를 위한 각 작업(task)에서 후보 웨브 서비스(candidate Web services)의 풀에서 웨브 서비스를 선택합니다. 이 방식은 모든 글로벌 QoS(Quality of Service) 제약사항과 서비스 간 제약사항(interservice constraints), 거래 제약사항(transactional constraints)을 동시에 고려합니다.

- **Performance Highlights**: 경험적 연구(empirical studies)에 따르면, 제안하는 방식은 모든 후보 웨브 서비스를 고려하는 기존 방법에 비해 더 높은 효율성과 효과성을 창출합니다.



### ALS-HAR: Harnessing Wearable Ambient Light Sensors to Enhance IMU-based Human Activity Recogntion (https://arxiv.org/abs/2408.09527)
- **What's New**: 이 연구에서는 착용할 수 있는 환경광센서(Ambient Light Sensor, ALS)를 활용한 활동 인식 시스템 ALS-HAR을 개발하여, 기존 IMU 센서를 기반으로 한 인간 활동 인식(HAR) 시스템의 성능을 향상시키기 위한 새로운 접근 방식을 소개합니다.

- **Technical Details**: ALS-HAR은 외부 조명 변화에 민감하지만, ALS에서 추출한 지식을 IMU 기반 활동 분류에 전달하여 환경의 영향을 최적화하는 전략을 도입합니다. 이 연구에서 제안한 Multi-modal HAR 데이터셋은 16명이 수행한 9가지 활동을 포함하며, 각각의 활동은 다양한 환경 시나리오에서 기록되었습니다.

- **Performance Highlights**: ALS 조건에서는 정확도가 떨어지지만, 이 시스템은 IMU 기반 분류기에서 최대 4.2%의 향상된 정확도와 6.4%의 매크로 F1 점수를 보여주었으며, 세 가지 실험 시나리오 중 두 가지에서는 다중 모달 센서 융합 모델을 초월하는 성과를 달성했습니다.




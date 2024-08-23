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




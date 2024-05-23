### A Survey on Large Language Models with Multilingualism: Recent Advances and New Frontiers (https://arxiv.org/abs/2405.10936)
Comments:
          54 pages, Work in Progress

- **What's New**: 최근 대형 언어 모델(LLMs)이 다국어 처리 분야에서 주목할 만한 성능을 보이며, 전세계 학계와 산업계의 관심을 끌고 있습니다. 이 논문은 이러한 LLMs의 다국어 시나리오에서의 이용과 관련된 최신 접근법, 발전, 한계점, 그리고 잠재적 해결책을 종합적으로 조사해 제공하고 있습니다.

- **Technical Details**: 이 논문은 LLMs의 다국어 능력을 개선하기 위해 트레이닝과 추론 방법, 모델 보안, 다중 도메인과 언어 문화, 데이터셋 사용 등 여러 관점에서 다룹니다. LLMs는 주로 트랜스포머 아키텍처를 기반으로 하고 대규모 데이터에서 학습합니다. 그러나 언어 자원이 적은 경우나 데이터 품질이 고르지 못한 경우 다국어 성능이 제한될 수 있습니다.

- **Performance Highlights**: LLMs는 기계 번역, 텍스트 요약, 감정 분석 등 여러 작업에서 최첨단 성능을 달성했습니다. 그러나 다국어 환경에서의 적용에는 한계가 많고, 특히 자원이 적은 언어에서 성능이 낮습니다. 이를 해결하기 위해 다국어 데이터를 추가하는 방법이 사용되지만, 여전히 지식 전달(knowledge transfer)이나 지식 축적(knowledge accumulation) 문제 등이 남아있습니다.



### GenToC: Leveraging Partially-Labeled Data for Product Attribute-Value Identification (https://arxiv.org/abs/2405.10918)
- **What's New**: GenToC라는 새로운 두 단계 모델을 소개합니다. 이 모델은 제품의 제목에서 속성-값 쌍을 추출하는 기술을 갖추고 있으며, 부분적으로 라벨링 된 데이터로 학습할 수 있습니다. 완전한 주석 데이터셋이 필요 없으며, 부트스트래핑 방법을 사용해 훈련 데이터셋을 점진적으로 개선합니다.

- **Technical Details**: GenToC는 두 단계로 구성된 모델입니다. 첫 번째 단계에서는 Generative Seq2Seq 모델을 사용해 제품 이름에서 모든 속성을 식별하고, 두 번째 단계에서는 Token Classification 모델을 사용해 각 속성의 값을 추출합니다. 첫 번째 단계에서는 '마커(markers)'를 도입해 모델 학습시 부분적으로 라벨링 된 데이터의 효과를 극대화하고, 두 번째 단계에서는 '값 가지치기(Value Pruning)' 방법을 사용해 더 정확한 값을 추출합니다.

- **Performance Highlights**: GenToC는 인도의 최대 B2B 이커머스 플랫폼에서 성공적으로 통합되어 기존 시스템의 리콜(Recall) 성능을 21.1% 향상시켰으며, 정확도(Precision)는 89.5%를 유지했습니다. 또한, 기존의 최첨단 NER 모델 및 생성 모델과 비교해 각각 16.2%, 18.2%의 F1 점수 향상을 이루었습니다. 부트스트래핑 방법을 통해 훈련 데이터셋을 강화하고, 실시간 시스템의 성능을 대폭 개선시켰습니다.



### COGNET-MD, an evaluation framework and dataset for Large Language Model benchmarks in the medical domain (https://arxiv.org/abs/2405.10893)
Comments:
          Technical Paper

- **What's New**: Large Language Models(LLMs)이 의료 분야에서의 발전 가능성을 염두에 두고 Cognitive Network Evaluation Toolkit for Medical Domains(COGNET-MD)를 소개합니다. 이 도구는 의료 도메인에서 LLM을 평가하기 위한 새로운 벤치마크로, 의료 텍스트 해석 능력을 평가하기 위한 증가된 난이도의 스코어링 프레임워크를 제안합니다. 현재 버전 1.0은 정신의학, 치과학, 폐의학, 피부과학, 내분비학 영역을 포함하며, 앞으로 더 많은 의료 도메인을 포함하도록 지속적으로 확장될 예정입니다.

- **Technical Details**: COGNET-MD는 독립적이며 무료로 사용할 수 있는 데이터셋을 제공하여, LLM을 의료 분야에서 평가하는 도구를 제공합니다. 데이터셋은 542개의 도메인별 질문과 다중 정답 선택지를 포함하고 있으며, 이에 따른 여러 난이도의 사용 사례도 제안됩니다. 이 데이터셋은 HuggingFace(https://huggingface.co/datasets/DimitriosPanagoulias/COGNET-MD/)에서 이용할 수 있습니다. 스코어링 알고리즘은 부분 적립 시스템(Partial Credit), 총점 적립 시스템(Full Credit), 정답 선택 시 페널티 시스템(Penalty for Incorrect Answers)을 포함합니다.

- **Performance Highlights**: COGNET-MD는 LLM의 전문지식과 관련된 추론 능력을 평가할 수 있게 하며, 데이터셋을 전체적으로, 부분적으로 또는 특정 의료 도메인에 초점을 맞추어 분석할 수 있습니다. 평가 시스템은 각 의료 전문 분야에서 모델이 얼마나 잘 수행하는지 파악할 수 있도록 설계되어 있습니다. 현재 버전은 정신의학, 치과학, 폐의학, 피부과학, 내분비학 분야의 데이터를 포함하고 있으며, 추가적인 데이터는 이후 버전에서 계속해서 업데이트될 예정입니다.



### Tailoring Vaccine Messaging with Common-Ground Opinions (https://arxiv.org/abs/2405.10861)
Comments:
          NAACL Findings 2024

- **What's New**: 이번 논문은 백신 우려 및 허위정보 대응을 위해 '공통 기반 의견' (Common-Ground Opinion, CGO)에 맞춘 대응 메시지를 생성하기 위한 새로운 작업을 정의합니다. 이를 위해 TAILOR-CGO라는 데이터셋을 소개하였으며, 여러 주요 대형 언어 모델(LLM)을 평가했습니다. 그 결과 GPT-4-Turbo가 다른 모델들보다 뛰어난 성능을 보였습니다.

- **Technical Details**: TAILOR-CGO 데이터셋은 총 22,400개의 고유한 맞춤형 응답을 포함하며, 6개의 다른 LLM이 생성했습니다. 각 응답은 절대 점수나 쌍쌍 비교 방식으로 레이블링되었습니다. 메시징 작업은 주어진 CGO를 기반으로 백신 우려에 대한 맞춤형 대응을 생성하는 것입니다. 대응 메시지는 우려에 대한 답변을 포괄적으로 제공하고, CGO를 직접 또는 간접적으로 언급하며, 공유된 의견을 진실로 인정하고, 우려에 대한 답변과 의미 있게 연결되어야 합니다.

- **Performance Highlights**: 본 논문에서는 여러 주요 LLM(GPT-4-Turbo, BERT 등)을 대상으로 맞춤형 메시징 성능을 평가했습니다. 자동 평가 척도로서 BERT 모델이 기존의 미세 조정된 LLM보다 우수한 성능을 보였습니다. 또한 일반 및 백신 유형을 특정하지 않은 다양한 우려 및 의견 문장을 생성하여 데이터셋을 구축했습니다.



### ECR-Chain: Advancing Generative Language Models to Better Emotion-Cause Reasoners through Reasoning Chains (https://arxiv.org/abs/2405.10860)
Comments:
          Accepted by IJCAI 2024

- **What's New**: 새로운 연구는 Cognitive Appraisal Theory에 영감을 받아 '자극-평가-감정(stimulus-appraisal-emotion)' 과정을 따라 감정 생성 프로세스를 심층적으로 이해하고자 합니다. 이를 위해 Emotion-Cause Reasoning Chain(ECR-Chain)을 도입하여, 대화에서 대상 감정 표현의 자극을 추론하는 단계별 추론 방법을 제안합니다. 특히 ChatGPT에 few-shot 프롬프트를 통해 ECR-Chain을 도입하여 Causal Emotion Entailment(CEE) 작업의 성능을 크게 향상시킵니다.

- **Technical Details**: 연구에서 제안된 방법은 감정-원인 추론 체인(ECR-Chain)을 사용하여 단계적인 추론을 수행합니다. 이 체인은 '주제(Theme)→반응(Reaction)→평가(Appraisal)→자극(Stimulus)'의 순서로 구성됩니다. Chain-of-Thought(CoT) 프롬프트 방법을 활용하여, 대형 언어 모델이 단계별로 추론하도록 안내하며, few-shot CEE 작업 설정에서 큰 성능 향상을 달성했습니다. 더 나아가, 자동 생성된 ECR-Chain 세트를 통해 교사 학습(Supervised Training)을 진행하여 작은 모델들의 감정 추론 능력을 향상시킵니다.

- **Performance Highlights**: 새롭게 제안된 방법은 기존의 모델들이 실현하지 못한 설명 가능한 감정-원인 추론을 효과적으로 수행할 수 있게 합니다. 다양한 실험 설정에서 이 방법이 감정-원인 발화 예측 및 설명 가능한 감정-원인 추론에 있어서 효과적임을 입증했습니다. 특히, Vicuna-7B 모델이 최첨단(Causual Emotion Entailment)성능을 달성하는 데 큰 도움을 주었습니다.



### ActiveLLM: Large Language Model-based Active Learning for Textual Few-Shot Scenarios (https://arxiv.org/abs/2405.10808)
Comments:
          18 pages, 7 figures, 4 tables

- **What's New**: ActiveLLM 소개: 이 논문은 ActiveLLM이라는 새로운 액티브 러닝(active learning) 방법론을 도입했습니다. 이것은 GPT-4, Llama 3, 그리고 Mistral Large 같은 대형 언어 모델(LLM)을 활용하여 데이터를 선택하는 접근법입니다. ActiveLLM은 '콜드 스타트(cold start)' 문제를 해결하여, 특히 퓨샷(few-shot) 학습 시나리오에서 기존의 액티브 러닝 방법들을 능가하는 성능을 보여줍니다. 또한, ActiveLLM은 비퓨샷(non-few-shot) 시나리오에서도 확장이 가능하며, 기타 액티브 러닝 전략의 콜드 스타트 문제를 극복하는 데 도움을 줄 수 있습니다.

- **Technical Details**: ActiveLLM은 대형 언어 모델(LLM)을 활용하여 주석 데이터 없이도 불확실성과 다양성을 추정할 수 있으며, 주석과정 동안 어떤 훈련도 필요로 하지 않습니다. 이 접근법은 LLM을 기반으로 한 액티브 러닝 전략으로 단독으로 작용할 수 있으며, 다른 액티브 러닝 전략의 콜드 스타트 문제를 해결하기 위한 해결책으로도 사용될 수 있습니다.

- **Performance Highlights**: ActiveLLM은 BERT 분류기의 분류 성능을 퓨샷 시나리오에서 의미 있게 향상시켰습니다. 이는 기존의 전통적인 액티브 러닝 방법과 퓨샷 학습 방법인 SetFit을 능가하는 결과를 보여줍니다. 종합 평가 결과, ActiveLLM은 다양한 학습 설정에서 모델 성능을 향상시키는 유망한 솔루션임을 시사합니다.



### SBAAM! Eliminating Transcript Dependency in Automatic Subtitling (https://arxiv.org/abs/2405.10741)
Comments:
          Accepted to ACL 2024 main conference

- **What's New**: 이 논문에서는 자동 자막 생성 모델을 소개합니다. 이 모델은 중간 단계의 대본(transcript) 없이 직접 자막과 타임스탬프를 생성하는 최초의 모델로, 여러 언어 쌍과 다양한 조건에서 뛰어난 성능을 보여줍니다.

- **Technical Details**: 이 모델은 직접 ST 모델(direct ST model)을 기반으로 하여, 음성과 텍스트 간의 시각적 정렬을 직접 실현합니다. 이 모델은 CTC (Connectionist Temporal Classification) 손실을 적용하여 타임스탬프 정렬을 보완하고, 주의(attention) 메커니즘을 통해 오디오와 텍스트의 시각적 정렬을 평가합니다. 또한 새로운 타임스탬프 평가 메트릭인 SubSONAR를 도입하여 시간 변동에 민감한 평가를 가능하게 하였습니다.

- **Performance Highlights**: 이 모델은 7개의 언어 쌍과 4개의 도메인에서 비교 실험을 통해 새로운 state-of-the-art 성능을 달성했습니다. 또한, 수동 평가를 통해 기존의 방법보다 타임스탬프 조정이 약 24% 감소했음을 확인했습니다.



### Feature-Adaptive and Data-Scalable In-Context Learning (https://arxiv.org/abs/2405.10738)
Comments:
          Accepted at ACL 2024 main conference

- **What's New**: 새로운 FADS-ICL 프레임워크는 현재의 In-Context Learning (ICL) 접근 방식을 향상시키며, 컨텍스트 제약을 극복하는 데이터 확장성과 특정 작업에 적응하는 기능을 도입합니다.

- **Technical Details**: 기존의 ICL이 제한된 컨텍스트 길이로 인한 제약을 갖는 반면, FADS-ICL은 확장 가능한 라벨 데이터를 활용하여 더 나은 성과를 달성합니다. 이 프레임워크는 LLM을 통해 일반적 특징을 추출한 후, 작업에 특화된 모듈레이터를 통해 특징을 세밀하게 조정하고 예측을 수행합니다. 여러 데이터 설정(4~128 샷) 및 LLM 스케일(0.8~70B) 설정 하에서 실험이 수행되었습니다.

- **Performance Highlights**: 실험 결과 FADS-ICL은 모든 설정에서 기존의 최고 성능 방법을 크게 능가했습니다. 예를 들어, 1.5B와 32 샷 설정 하에서 FADS-ICL은 10개의 데이터셋에 대해 기존 ICL보다 평균 14.3% 높은 정확도를 기록했으며, 기존 최고 성능 방법보다 평균 6.2% 높은 정확도를 보였습니다. 더 많은 데이터를 사용할수록 성능 향상이 더욱 두드러졌습니다.



### INDUS: Effective and Efficient Language Models for Scientific Applications (https://arxiv.org/abs/2405.10725)
- **What's New**: INDUS는 지구 과학, 생물학, 물리학, 헬리오피직스, 행성 과학 및 천체 물리학 등 다양한 분야를 위한 포괄적인 대형 언어 모델(Large Language Model, LLM) 집합을 소개합니다. 이 모델은 다양한 데이터 소스로부터 수집된 과학적 코퍼스(corpus)로 트레이닝 되었습니다. 또한, 새로운 과학 벤치마크 데이터셋인 CLIMATE-CHANGE-NER(엔티티 인식), NASA-QA(추출형 질문응답) 및 NASA-IR(정보 검색)을 사용하여 연구를 가속화하고자 합니다.

- **Technical Details**: INDUS 모델은 세 가지 종류로 구성되어 있습니다. 첫째, 도메인 특정 어휘와 코퍼스를 사용해 자연어 이해 작업을 다루는 인코더 모델입니다. 둘째, 여러 데이터셋을 사용해 정보 검색 작업을 다루기 위해 대조 학습 기반의 일반 텍스트 임베딩(embedding) 모델입니다. 셋째, 지식 증류(knowledge distillation) 기술을 사용해 리소스 제약이 있는 응용프로그램을 위해 만들어진 작은 버전의 모델들입니다. 이 모델들은 바이타 쌍 인코딩 알고리즘(Byte-Pair Encoding, BPE)을 사용하여 구축된 인디스 BPE 토크나이저와 함께 사용됩니다.

- **Performance Highlights**: 실험 결과, INDUS 모델은 일반적 용도의 인코더인 RoBERTa와 기존의 도메인 특정 인코더인 SciBERT보다 새로운 작업 및 기존 벤치마크 작업에서 더 뛰어난 성능을 보였습니다. 지식 증류 모델은 원래 모델에 비해 지연 시간이 크게 향상되었으며, 대부분의 벤치마크 작업에서 강력한 성능을 유지했습니다. 이를 통해 INDUS 모델은 각 도메인에서의 효율적 접근과 정보 기반 의사 결정을 지원합니다.



### Persian Pronoun Resolution: Leveraging Neural Networks and Language Models (https://arxiv.org/abs/2405.10714)
- **What's New**: 이번 연구는 최초로 페르시아어 대명사 해소(pronoun resolution)를 위한 끝에서 끝(end-to-end) 신경망 시스템을 제안합니다. 이 시스템은 사전 학습된 Transformer 모델인 ParsBERT를 활용하여 대명사 선행사를 식별합니다. 이는 기존의 규칙 기반 및 통계적 방법을 사용한 시스템 대비 F1 점수 3.37의 향상을 보여줍니다.

- **Technical Details**: 이번 연구에서는 대명사 해소와 멘션 탐지(mention detection)를 분리된 작업이 아닌 하나의 작업으로 다루어 최적화하는 접근법을 채택했습니다. 특히, 사전 학습된 Transformer 모델인 ParsBERT를 사용하여 페르시아어 텍스트에서 대명사를 해소하는 시스템을 개발했습니다. 이 시스템은 서로 다른 기술을 조합하여 성능을 크게 향상시키는 신경망(neural network)과 언어 모델(linguistic model)의 유효성을 보여줍니다.

- **Performance Highlights**: 실험 결과로 얻은 3.37 F1 점수의 향상은 페르시아어 대명사 해소 분야에서 중요한 성과를 나타냅니다. 이는 기존의 규칙 기반 및 통계적 방법을 대체하는 신경망 기반의 접근법의 유효성을 입증했으며, 이 분야에서 추가 연구의 가능성을 제시합니다.



### Empowering Prior to Court Legal Analysis: A Transparent and Accessible Dataset for Defensive Statement Classification and Interpretation (https://arxiv.org/abs/2405.10702)
- **What's New**: 이 논문은 경찰 인터뷰 동안 제공된 진술을 분류하기 위해 특화된 새로운 데이터셋을 소개합니다. 법적 절차 이전 단계에서 진술의 진실 여부를 식별하고 분류하기 위한 이 데이터셋은 법률 정보학 및 자연어 처리 (NLP) 연구에 중요한 기여를 합니다.

- **Technical Details**: 연구 팀은 DistilBERT 모델을 미세 조정하여 진실한 진술과 기만적인 진술을 구분하는 데 있어서 최첨단 성능을 달성했습니다. 모델의 해석 가능성을 높이기 위해, 설명 가능한 인공지능 (XAI) 방법을 사용하여 모델의 의사 결정 과정을 설명하는 saliency 지도(saliency maps)를 활용했습니다. 또한, 법률 전문가와 비전문가 모두가 시스템을 상호 작용하고 활용할 수 있도록 XAI 인터페이스를 개발했습니다.

- **Performance Highlights**: 도입된 모델은 86%의 정확도를 달성했으며, 사용자 지정된 transformer 아키텍처를 능가했습니다. 적층된 접근 방식은 성명 분석의 접근성, 투명성, 효율성을 크게 향상시키며, 법률 실무와 연구 모두에 유망한 영향을 미칩니다.



### Revolutionizing Process Mining: A Novel Architecture for ChatGPT Integration and Enhanced User Experience through Optimized Prompt Engineering (https://arxiv.org/abs/2405.10689)
- **What's New**: 이 연구는 비즈니스 프로세스 관리 분야에서 ChatGPT와 같은 대형 언어 모델(LLMs)을 프로세스 마이닝(process mining) 도구에 통합하는 새로운 접근 방식을 소개합니다. 이를 통해 프로세스 분석을 더 널리 접근 가능하게 하고 사용자 경험을 향상시킵니다.

- **Technical Details**: 연구의 주요 혁신은 각 프로세스 마이닝 서브모듈에 맞춘 특정 프롬프트 엔지니어링(prompt engineering) 전략 개발에 있습니다. 통합 아키텍처는 ETL(Extract, Transform, Load) 프로세스를 따르며, 제로 샷(zero-shot) 및 최적화된 프롬프트 엔지니어링 기법을 활용합니다. ChatGPT는 API를 통해 연결되며, 프로세스 마이닝 모듈들로부터 구조화된 출력을 받아 대화형 상호작용이 가능합니다.

- **Performance Highlights**: 이 접근 방식의 효과성을 검증하기 위해 BehfaLab의 프로세스 마이닝 도구를 사용하는 17개 기업의 데이터를 사용했습니다. 전문가 패널은 결과의 72%를 'Good'으로 평가하며, 사용자 경험이 크게 향상됨을 보였습니다.

- **Future Directions**: 향후 연구 방향은 프롬프트 엔지니어링의 추가 최적화, 다른 AI 기술과의 통합 탐색, 다양한 비즈니스 환경에서의 확장성 평가를 포함합니다.



### Realistic Evaluation of Toxicity in Large Language Models (https://arxiv.org/abs/2405.10659)
- **What's New**: 이번 논문에서 연구진은 새로운 Thoroughly Engineered Toxicity (TET) 데이터셋을 소개했습니다. 이 데이터셋은 여러 인기 있는 대형 언어 모델(LLM)의 잠재적인 독성 인식을 평가하기 위해 제작되었습니다. 이를 통해 기존의 평가 방법으로는 드러나지 않았던 독성을 발견할 수 있습니다.

- **Technical Details**: TET 데이터셋은 25개의 다른 LLM들과 1백만 개 이상의 실제 상호작용에서 필터링된 2,546개의 프롬프트로 구성됩니다. 이 데이터셋은 사용자가 일반적으로 실제 세계에서 LLM과 상호작용할 때 사용하는 현실적인 프롬프트를 포함하고 있습니다. 연구진은 HateBERT와 Perspective API를 사용하여 이러한 프롬프트들을 필터링하고 평가했습니다.

- **Performance Highlights**: LLM의 독성 평가 결과에서 Llama 2-7B-Chat 모델이 가장 낮은 독성 점수를 기록하며 우수한 성능을 보였습니다. 반면 Mistral-7B-v0.1, OpenChat 3.5, Zephyr-7B-β 모델은 더 높은 독성 점수를 보여 잠재적으로 독성 컨텐츠를 생성할 가능성이 더 큽니다.



### SPOR: A Comprehensive and Practical Evaluation Method for Compositional Generalization in Data-to-Text Generation (https://arxiv.org/abs/2405.10650)
- **What's New**: 이번 연구에서는 SPOR이라는 새로운 평가 방법을 제안합니다. SPOR은 데이터-텍스트 생성(data-to-text generation)에서 조합 일반화(compositional generalization)를 평가하기 위해 고안된 포괄적이고 실용적인 방법론입니다. 기존 연구가 체계성(systematicity)에만 초점을 맞췄던 반면, SPOR은 네 가지 측면에서 평가를 진행합니다: 1) 체계성(Systematicity), 2) 생산성(Productivity), 3) 순서 불변성(Order invariance), 4) 규칙 학습 가능성(Rule learnability). 추가적인 수작업 주석 없이 기존 데이터셋을 기반으로 고품질 평가가 가능합니다.

- **Technical Details**: SPOR을 테스트하기 위해 WebNLG와 E2E 두 개의 데이터셋을 사용하였습니다. WebNLG 데이터셋에서는 주어-술어-목적어(s, p, o) 형태의 트리플을, E2E 데이터셋에서는 속성-값(attribute-value) 쌍을 데이터 유닛으로 사용했습니다. 평가에는 T5-large, BART-large, GPT-2-large와 같은 기존의 언어 모델과 T5-11b, Mistral-7b, Llama-2-13b 등 최신 대형 언어 모델(LLMs)이 포함되었습니다. 데이터 입력에는 선형화 방법(linearization method)을 사용하고, LoRA 파인튜닝을 이용해 모델 학습을 진행하였습니다. 학습은 10 epoch 동안 이루어졌으며, 평가 메트릭으로는 PARENT를 사용하여 성능을 측정했습니다.

- **Performance Highlights**: 평가 결과, 실험에 사용된 모델들은 각각의 평가 측면에서 모두 부족한 점을 드러냈습니다. 이는 데이터-텍스트 생성 과제에서 조합 일반화(compositional generalization)의 다양한 측면을 포괄하는 연구가 필요함을 보여줍니다. SPOR을 통해 이러한 평가가 가능해졌으며, 이를 기반으로 차후 연구 방향을 설정할 수 있습니다.



### Layer-Condensed KV Cache for Efficient Inference of Large Language Models (https://arxiv.org/abs/2405.10637)
Comments:
          Accepted to ACL2024 main conference

- **What's New**: 이 논문에서는 대형 언어 모델(Large Language Models, LLMs)을 실제 응용 프로그램에 배치할 때의 주요 병목 현상인 높은 메모리 소비 문제를 해결하기 위한 새로운 방법을 제안합니다. 키-값(KV) 캐시를 소수의 레이어만 계산하고 캐싱함으로써 메모리 소비를 크게 줄이고 추론 처리량을 향상시킬 수 있습니다. 이 접근법은 기존의 메모리 절약 기법과 직교하여 통합이 용이합니다.

- **Technical Details**: 제안된 방법은 트랜스포머 디코더의 변종으로, 모든 레이어의 쿼리를 상위 레이어의 KVs와 짝지어 계산합니다. 이렇게 하면 여러 레이어에 대해 KVs를 캐싱하지 않아도 되며, 메모리 소비를 줄이고 계산 효율성을 높일 수 있습니다. 이 방법은 간단한 근사 학습 방법을 통해 병렬 학습을 가능하게 합니다. 또한, 일부 레이어에 대해 기본적인 주의 메커니즘을 유지함으로써 언어 모델링 및 다운스트림 작업에서의 성능 저하를 방지할 수 있습니다.

- **Performance Highlights**: LLMs에서 제안된 방법을 사용하면 최대 32배 더 큰 배치 크기와 최대 26배 더 높은 처리량을 달성할 수 있으며, 언어 모델링 및 다운스트림 작업에서 기존의 트랜스포머와 비슷한 성능을 보입니다. 또한, StreamingLLM 같은 다른 메모리 절약 기법과 쉽게 통합하여 추론 효율성을 더욱 향상시킬 수 있습니다.



### Medical Dialogue: A Survey of Categories, Methods, Evaluation and Challenges (https://arxiv.org/abs/2405.10630)
- **What's New**: 이 논문은 의료 대화 시스템에 대한 연구를 종합하고 체계적으로 검토한 최초의 시도입니다. 기존에는 기능적인 관점에서만 논의되었으나, 이번 연구는 기술적 관점에서 분류, 방법, 평가를 총정리하여 현재 기술 수준을 진단하고 향후 과제를 제시합니다.

- **Technical Details**: 의료 대화 시스템은 환자와의 대화로 증상을 더 파악하고, 진단하고, 치료 계획을 제안하는 시스템입니다. 이러한 시스템은 의사, 환자, 의료 학생을 위한 다양한 기능을 포함합니다. 주로 'retrieval', 'generation', 'hybrid' 방법론을 기반으로 구축됩니다. 최근에는 LLM(Large Language Models)의 도입으로 놀라운 성능 향상이 이루어졌지만, 여전히 여러 문제점이 존재합니다.

- **Performance Highlights**: LLM을 활용한 의료 대화 시스템은 기존 방법론에 비해 인간과 유사한 응답 생성 능력 및 높은 정확도를 달성했습니다. 그러나 진단과 같은 실질적인 의료 행위로의 전환이 어려워 현실적인 요구를 충족하지 못하는 한계가 있습니다.



### DeepPavlov at SemEval-2024 Task 8: Leveraging Transfer Learning for Detecting Boundaries of Machine-Generated Texts (https://arxiv.org/abs/2405.10629)
Comments:
          New best score from the leaderboard, to appear in SemEval-2024 Workshop proceedings

- **What's New**: 2024년 SemEval 대회에서 인간과 AI가 협업하여 작성한 텍스트의 경계를 감지하는 문제를 다루고자 하는 새로운 과제가 발표되었습니다. 특히, DeBERTaV3 모델의 supervised fine-tuning을 위해 데이터 증강 파이프라인을 제안하여, 리더보드에서 새로운 최고 MAE 점수를 달성했습니다.

- **Technical Details**: 본 논문에서는 GPT-3, GPT-4, LLaMA2와 같은 최신 auto-regressive language models이 인간과 비슷한 텍스트를 생성하는 문제를 해결하려고 합니다. 기존의 바이너리 분류 접근법 대신, 인간이 작성한 텍스트와 기계가 생성한 텍스트 사이의 더 미세한 경계를 찾기 위한 멀티클래스 분류나 저자 속성 분류 문제로 설정했습니다. 또한, DeBERTaV3 모델을 fine-tuning 하기 위한 데이터 증강 파이프라인을 제안했습니다.

- **Performance Highlights**: 제안된 파이프라인을 사용하여 새로운 최고 MAE 점수를 달성했습니다. 여러 아키텍처를 가진 fine-tuned 모델들의 성능을 비교했고, 데이터 증강의 효과를 검증했습니다. 또한, 증강 코드가 공개되어 있습니다.



### Dynamic data sampler for cross-language transfer learning in large language models (https://arxiv.org/abs/2405.10626)
Comments:
          Accepted by ICASSP 2024

- **What's New**: 새로운 논문에서는 여러 언어 간 전송(transfer)을 기반으로 한 대형 언어 모델(LLM)인 ChatFlow를 제안했습니다. 이 모델은 비용 효율적으로 대형 중국어 언어 모델을 훈련하는 것을 목표로 합니다. 논문은 LLaMA2 모델을 사용해 영어, 중국어 및 병렬 자료(corpus)를 혼합하여 지속적으로 훈련하고, 교차 언어 표현을 정렬해 중국어 모델로의 지식 전이를 촉진합니다.

- **Technical Details**: ChatFlow는 교과 학습(curriculum learning)에서 영감을 받아 동적인 데이터 샘플러(dynamic data sampler)를 사용해 모델을 비지도 전훈련에서 감독 방식의 미세조정을 통해 점진적으로 전환시킵니다. 초기에 영어와 병렬 자료의 비율이 높으며, 점진적으로 중국어와 지시 데이터의 비율을 증가시킵니다. 약 50GB의 데이터를 사용해 LLaMA2-7B 기반의 모델을 훈련합니다.

- **Performance Highlights**: 텍스트 자동 평가(MMLU, C-Eval, CMMUL, GAOKAO)에서 ChatFlow는 다른 LLaMA-2-7B 모델을 후훈련한 중국어 모델들과 비교해 우수한 결과를 나타냈습니다. 인간 기반 평가(SuperCLUE)에서도 7B 규모 모델 중 5위를 차지했습니다. ChatFlow는 영어 기반 모델에서 훈련되었고, 적은 양의 중국어 데이터로도 효율적으로 훈련되었습니다.



### Specialising and Analysing Instruction-Tuned and Byte-Level Language Models for Organic Reaction Prediction (https://arxiv.org/abs/2405.10625)
Comments:
          Preprint

- **What's New**: 이 연구에서는 FlanT5 및 ByT5와 같은 언어 데이터만으로 사전 학습된 Transformer 기반 encoder-decoder 모델들이 유기 반응 예측 작업에 효과적으로 특화될 수 있는지를 탐구합니다. 이는 기존의 많은 GPU 자원이 소모되는 화합물 사전 학습의 필요성을 대체할 수 있을지에 대한 중요한 질문입니다.

- **Technical Details**: 연구진은 SMILES-oriented 사전 학습, 미세 조정(fine-tuning) 샘플 효율성, 토큰화(tokenization), 그리고 추론 시 디코딩 알고리즘의 영향을 체계적으로 실험했습니다. FlanT5와 ByT5 모델은 자연어 작업만으로 사전 학습되었지만, 유기 화학 반응 예측 분야에서도 '화학 도메인 호환성(chemistry domain compatible)'을 가지는 것으로 나타났습니다.

- **Performance Highlights**: 모든 모델들이 비슷한 Top-1 및 Top-5 정확도를 보였으며, 일부 변형된 모델 간에 차이가 존재했습니다. 토큰화와 어휘 축소는 최종 성능에 약간 영향을 미쳤으나, 학습과 추론 속도를 높였습니다. 가장 효율적인 greedy 디코딩 전략은 매우 경쟁력이 있었으며, 더 정교한 디코딩 알고리즘으로부터는 약간의 성능 향상만이 확인되었습니다.



### Feature-based Low-Rank Compression of Large Language Models via Bayesian Optimization (https://arxiv.org/abs/2405.10616)
Comments:
          Accepted by 2024 ACL findings

- **What's New**: 최근 몇 년 동안, 대형 언어 모델(Large Language Models, LLMs)은 자연어 처리 분야에서 큰 진전을 이뤄냈습니다. 그러나 이들의 규모가 커지면서 계산 부담도 증가하고 있어, 효율성과 성능 간의 균형을 맞추는 것이 필요해졌습니다. 이 논문에서는 LLMs에서 저차원 압축(Low-Rank Compression, LRC)을 적용하기 위한 새로운 방법을 제안합니다. 이는 풀드 공분산 행렬(Pooled Covariance Matrices)을 사용해 특징 분포를 정밀하게 추정하고, 베이지안 최적화(Bayesian Optimization) 전략을 통해 저차원 할당을 최적화하는 방식입니다.

- **Technical Details**: 저차원 압축의 핵심은 저차원 분해 및 저차원 차원 할당에 있습니다. 본 연구에서는 LLMs의 저차원 특성을 경험적으로 분석하고, 풀드 공분산 행렬을 사용하여 특징 분포를 더 정확하게 추정하는 방식을 채택했습니다. 이를 바탕으로 저차원 차원 할당을 위해 베이지안 최적화 방법을 사용하였습니다. 이 과정에서 Grid Search보다 더 효율적인 방법을 제공하며, LLaMA-2 모델을 대상으로 실험을 통해 제안된 방법의 우수성을 입증했습니다.

- **Performance Highlights**: 제안된 방법은 기존의 강력한 구조적 가지치기(Structured Pruning) 및 저차원 압축 기술보다 높은 성능을 유지하면서 동일한 압축 비율에서 더욱 효율적입니다. LLaMA-2 모델에서 20%의 압축률을 달성하면서도 모델 성능의 98%를 유지하며, 이는 최신 상태의 기술과 동등한 수준의 결과를 보여줍니다.



### RDRec: Rationale Distillation for LLM-based Recommendation (https://arxiv.org/abs/2405.10587)
Comments:
          10 pages. Accepted to ACL 2024 Main as a short paper

- **What's New**: 최근 LLM(Large Language Model) 기반 추천 시스템이 사용자와 아이템의 텍스트 프롬프트(textual prompts)를 통해 효과적인 의미 논리를 연결하는 방식이 주목받고 있습니다. 이 논문에서는 추천 과정에서 사용자 선호도와 아이템 속성과 같은 상호작용의 근거(rationales)를 고려하지 않은 기존 방법들의 한계를 극복하기 위해 'Reasoning Distillation Recommender (RDRec)'라는 모델을 제안하고 있습니다. RDRec는 큰 언어 모델(LM)이 생성한 근거를 학습하는 소형 모델로 설계되었습니다.

- **Technical Details**: RDRec는 사용자 및 아이템과 관련된 리뷰(review)에서 도출된 근거를 활용하여 그들의 프로필을 명확하게 구체화합니다. 이 모델은 대형 언어 모델이 생성한 리뷰의 근거를 학습하여 추천의 정확성을 높입니다. 또한, 다양한 실험을 통해 RDRec가 상위-N(top-N) 및 순차적(sequential) 추천에서 SOTA(state-of-the-art) 성능을 달성했음을 보여줍니다. 연구진은 논문과 함께 소스 코드를 공개했습니다.

- **Performance Highlights**: RDRec는 기존 모델들보다 상위-N 및 순차적 추천에서 탁월한 성능을 보였다(State-of-the-Art). 이는 사용자와 아이템의 실제 리뷰에서 도출된 근거를 효과적으로 활용함으로써 사용자와 아이템의 프로필을 보다 명확하게 정의한 덕분입니다.



### A Hard Nut to Crack: Idiom Detection with Conversational Large Language Models (https://arxiv.org/abs/2405.10579)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)을 이용한 관용적 언어 처리에 대해 탐구합니다. 언어 전문가들이 설계한 어려운 예제로 구성된 새로운 데이터셋인 IdioTS(Idiomatic language Test Suite)를 소개하며, 이는 LLMs가 문장 수준에서 비유적 언어를 처리하는 능력을 평가하는 데 중점을 둡니다. 이 새로운 데이터셋을 기반으로 한 포괄적인 평가 방법론을 제안하며, LLMs에게 주어진 영어 문장에서 관용 표현을 감지하도록 요청합니다.

- **Technical Details**: 평가 방법론은 관용 표현 감지 작업을 중심으로 하며, LLMs이 주어진 문장에서 비유적 의미와 문자적 의미를 구별하는 능력을 평가합니다. Haagsma et al. (2020)의 정의에 따라, 잠재적인 관용 표현(PIEs)은 특정 문맥에서 관용적 의미를 가질 수 있는 표현입니다. 데이터셋 생성 과정에서, 다양한 온라인 플랫폼에서 추출한 관용 표현을 엄선하여 포괄적인 목록을 작성했습니다. 각 관용 표현에 대해 새로운 문장을 작성하여 데이터 오염을 방지했습니다.

- **Performance Highlights**: 실험 결과, 사전 학습된 언어 모델이 비유적 문장을 포함하더라도 관용적 언어는 여전히 도전 과제로 남아있음을 보여줍니다. 다양한 접근 방식을 통해 최종적으로 164개의 비유적 문장과 164개의 문자적 문장을 포함한 데이터셋이 완성되었습니다. 이러한 데이터를 이용한 평가 결과, LLMs가 비유적 표현을 다루는 데 있어서 인간의 성능에 미치지 못하지만, 이 연구는 이러한 모델을 위한 중요한 평가 지침을 제공합니다.



### Language Models can Exploit Cross-Task In-context Learning for Data-Scarce Novel Tasks (https://arxiv.org/abs/2405.10548)
Comments:
          Accepted at ACL 2024 Main

- **What's New**: 대규모 언어 모델(LLMs)은 자연어 처리(NLP)의 패러다임을 바꾸고 있으며, 그 중에서도 놀라운 In-context Learning (ICL) 능력으로 주목받고 있습니다. 하지만 이러한 모델들을 새로운 과제에 적응시키는 것은 여전히 도전 과제입니다. 이 논문은 LLMs가 사전 정의된 과제의 라벨 예제를 통해 새로운 과제를 학습할 수 있는지를 조사합니다. 생물학적 뉴런과 Transformer 아키텍처의 기계적 해석에서 영감을 받아, 여러 과제 간 정보 공유의 잠재력을 탐구합니다.

- **Technical Details**: 이 연구에서 Cross-task prompting 설정을 사용하여 세 가지 LLMs(LLaMA-2 7B 및 13B, GPT 3.5)을 활용했습니다. 소스 과제와 타겟 과제 쌍을 설정하고, 타겟 과제를 위한 예제 없이도 Cross-task prompting이 zero-shot prompting 대비 성능을 크게 개선하는 것을 확인했습니다. 구체적으로 LLaMA-2 7B는 평균 107%, LLaMA-2 13B는 18.6%, GPT 3.5는 3.2%의 성능 향상을 보였습니다. 또한, Pseudo-labeling 방식으로 소스 과제의 레이블 공간을 타겟 과제로 복사하는 문제를 해결하고자 했습니다.

- **Performance Highlights**: Cross-task prompting은 평균적으로 zero-shot prompting 대비 높은 성능 향상을 보였으며, 경우에 따라 일반 in-context learning과 비교해도 손색없는 성능을 보였습니다. 특히 LLMs는 동일한 레이어에서, 다양한 타겟 과제를 처리할 때도 유사한 모델 활성화를 보였습니다. 이러한 통찰력을 통해 LLMs가 서로 다른 과제 예제 기반의 맥락 신호를 사용하여 새로운 과제를 해결할 수 있는 가능성을 처음으로 탐구했습니다.



### Benchmarking Large Language Models on CFLUE -- A Chinese Financial Language Understanding Evaluation Datas (https://arxiv.org/abs/2405.10542)
Comments:
          Accepted by ACL 2024

- **What's New**: 최근 대형 언어 모델(LLMs)의 발전으로 자연어 처리(NLP) 분야에 큰 혁신이 일어남에 따라, 이러한 모델의 성능을 평가할 새로운 벤치마크의 필요성이 커지고 있습니다. 본 논문은 중국어 금융 언어 이해 평가 벤치마크인 CFLUE를 제안합니다. CFLUE는 지식 평가와 응용 평가를 통해 LLMs의 다양한 능력을 평가하도록 설계되었습니다. 지식 평가 부분에서는 38,000개 이상의 객관식 질문과 그 해설이 포함되어 있으며, 응용 평가 부분에서는 텍스트 분류, 기계 번역, 관계 추출, 독해, 텍스트 생성 등 16,000개 이상의 테스트 인스턴스가 포함됩니다. CFLUE는 다양한 대표 LLMs에 대한 종합적인 평가를 수행합니다.

- **Technical Details**: CFLUE는 두 가지 주요 평가 영역을 포함합니다: 지식 평가(Knowledge Assessment)와 응용 평가(Application Assessment). 지식 평가는 38K+ 객관식 질문으로 구성되어 있으며, 이는 답안 예측과 질문 추론 두 가지 목적을 가집니다. 응용 평가는 텍스트 분류, 기계 번역, 관계 추출, 독해, 텍스트 생성 등 5가지 NLP 작업 그룹에서 16K+ 테스트 인스턴스로 구성됩니다. 평가된 모델들은 GPT-4와 GPT-4-turbo를 포함한 세 개의 OpenAI LLM, 그리고 다양한 경량 LLM 등을 포함합니다.

- **Performance Highlights**: 실험 결과 GPT-4와 GPT-4-turbo는 지식 평가에서 60% 이상의 정확도를 달성하며 다른 모델들보다 우수한 성능을 보였습니다. 그러나 응용 평가에서는 특정 작업에서 경량 LLM이 더 뛰어난 성능을 보이기도 했습니다. 이는 이러한 경량 LLM들이 중국어 데이터를 위해 특수하게 설계되었기 때문일 수 있습니다. FinGPT V3, DISC-FinLLM, Tongyi-Finance 등 기존 금융 도메인 LLM은 지식 평가와 응용 평가에서 낮은 정확도를 보여, 금융 지식의 제한적 커버리지와 개선 가능성을 나타냈습니다.



### Smart Expert System: Large Language Models as Text Classifiers (https://arxiv.org/abs/2405.10523)
Comments:
          11 pages, 3 figures, and 8 tables

- **What's New**: 이 논문은 대형 언어 모델 (Large Language Models, LLMs)를 활용한 새로운 텍스트 분류 시스템인 Smart Expert System을 소개합니다. 이 시스템은 전통적인 텍스트 분류 워크플로우를 단순화하여 광범위한 전처리 및 도메인 전문 지식이 필요하지 않습니다. 이는 특히 감정 분석, 스팸 SMS 탐지 및 다중 라벨 분류에서 전통적인 방법을 초과하는 성능을 보여줍니다.

- **Technical Details**: 텍스트 분류는 전통적으로 다양한 기계 학습 (Machine Learning, ML) 및 신경망 (Neural Network, NN) 모델을 사용하여 수행되었습니다. 그러나 LLMs는 트랜스포머 아키텍처 (Transformer architecture)을 기반으로 하여, 사전 훈련된 텍스트 데이터에 의해 풍부한 언어 표현을 학습합니다. Smart Expert System은 이러한 LLMs를 텍스트 분류기(Text Classifier)로 활용하여 복잡한 전처리나 피처 엔지니어링 없이도 높은 성능을 발휘할 수 있습니다. 새로운 평가 메트릭인 Uncertainty/Error Rate (U/E rate)를 도입하여 모델의 신뢰성을 추가적으로 평가합니다.

- **Performance Highlights**: 여러 데이터셋을 대상으로 한 테스트 결과, 몇몇 LLMs는 전통적인 머신 러닝 및 신경망 모델을 능가하는 성능을 보였습니다. 특히, 적은 예시 학습 또는 미세 조정을 통해 LLMs 모델의 성능이 매우 향상되었고, 모든 데이터셋에서 최고 성능을 기록했습니다. 이러한 결과는 LLMs의 다재다능성과 효율성을 확인시켜줍니다. 소스 코드와 데이터셋은 GitHub repository에 공개되어 있습니다.



### Towards Better Question Generation in QA-Based Event Extraction (https://arxiv.org/abs/2405.10517)
Comments:
          Accepted to ACL2024

- **What's New**: 이번 논문에서는 기존 Event Extraction(EE) 작업의 패러다임이 classification 기반에서 question-answering(QA) 기반으로 전환된 것에 주목하며, QA 기반 EE에서 고품질 질문을 생성하는 방법을 제안합니다. 특히, QA 모델의 성능에 영향을 미치는 '좋은 질문'의 기준을 네 가지로 정의하고, 이를 바탕으로 강화 학습을 적용하여 유연하고 맥락 의존적인 질문을 생성하는 방법을 소개합니다.

- **Technical Details**: 이 논문은 질문의 4가지 기준(유창성, 범용성, 맥락 의존성, QA 모델에 대한 명확한 지침)을 제안하고, 이를 만족시키기 위해 sequence-to-sequence 기반 모델과 강화 학습 기반 프레임워크를 개발했습니다. 또한, inverse prompting 메커니즘과 question-answering reward를 이용해 질문의 질을 평가하고 조정합니다. 이를 통해, 질문 생성 모델을 미세 조정하여 보다 맥락에 맞고 유도적인 질문을 생성하도록 합니다.

- **Performance Highlights**: 제안된 방법은 ACE와 RAMS 데이터셋에 대한 광범위한 실험에서 기존 방법보다 각각 2.69%, 1.96% 성능이 향상되었습니다. 특히, 데이터가 제한된 상황에서도 탁월한 성능을 발휘하여, 학습 데이터의 40%만을 사용해도 기존 방법과 유사한 성과를 달성했습니다.



### Language Models can Evaluate Themselves via Probability Discrepancy (https://arxiv.org/abs/2405.10516)
Comments:
          ACL 2024 Findings

- **What's New**: 이 논문에서는 다양한 대형 언어 모델(LLMs)의 성능을 평가하기 위한 새로운 자기 평가 방법인 ProbDiff를 제안합니다.

- **Technical Details**: ProbDiff는 추가 평가 모델이나 외부의 독점적인 모델(GPT-4)에 의존하지 않고, 테스트 중인 LLM 자체를 사용하여 초기 응답과 수정된 버전 간의 확률 차이를 계산합니다. 두 LLM 사이의 질문에 대한 높은 차이는 상대적으로 더 낮은 역량을 나타냅니다.

- **Performance Highlights**: ProbDiff는 GPT-4 기반 평가와 동등한 결과를 달성하였으며, 자연어 생성(NLG) 작업에서 번역, 요약, Xiaohongshu 블로그 작성과 같은 다양한 시나리오를 아우르고, AlignBench, MT-Bench, AlpacaEval과 같은 LLM 평가 벤치마크에서 다양한 규모의 LLM을 통해 평가되었습니다.



### Automatic News Generation and Fact-Checking System Based on Language Processing (https://arxiv.org/abs/2405.10492)
- **What's New**: 이 논문은 뉴스 생산의 효율성과 품질을 향상시키면서 뉴스 콘텐츠의 진정성과 신뢰성을 보장하기 위해 언어 처리를 기반으로 한 자동 뉴스 생성 및 사실 확인 시스템을 탐구합니다. NLP(Natural Language Processing) 및 딥 러닝 기술의 급속한 발전으로, 자동 뉴스 생성 시스템은 방대한 데이터에서 핵심 정보를 추출해 잘 구조화된, 유창한 뉴스 기사를 생성할 수 있습니다.

- **Technical Details**: 이 연구에서는 자동 뉴스 생성 및 사실 확인에 관련된 주요 기술들을 자세히 다룹니다. 여기에는 텍스트 생성 (text generation), 정보 추출 (information extraction), 지식 그래프 (knowledge graphs)의 적용 등이 포함됩니다. 또한, 이러한 기술들의 효과성을 실험을 통해 검증합니다.

- **Performance Highlights**: 논문의 결과는 기술적 최적화와 실제 응용이 지속될수록 이 시스템들이 미래의 뉴스 산업에서 점점 더 중요한 역할을 할 것이며, 더욱 효율적이고 신뢰할 수 있는 뉴스 서비스를 제공할 것임을 보여줍니다.



### CNER: A tool Classifier of Named-Entity Relationships (https://arxiv.org/abs/2405.10485)
- **What's New**: 새롭게 도입된 CNER은 스페인어(named entities; 명명된 엔티티) 간의 의미적 관계 추출을 위한 다양한 도구들이 결합된 시스템입니다. 이 시스템은 사용자 친화적인 인터페이스를 통해 사용자가 자유롭게 텍스트를 입력하거나 파일을 통해 손쉽게 분석할 수 있도록 설계되었습니다.

- **Technical Details**: CNER은 컨테이너 기반 아키텍처를 바탕으로, 다양한 명명된 엔티티 인식(Recognition) 및 관계 추출(Extraction) 도구를 통합한 시스템입니다. 현재 프로토타입 형태로 개발되었으며 Universidad del Valle 내의 자연어 처리(NLP) 그룹에서 교육적 자원으로 사용됩니다.

- **Performance Highlights**: 초기 실험 결과, CNER이 스페인어 기반 NLP 도구들의 이해와 개발에 유망한 잠재력을 가지고 있음을 확인하였습니다. 특히, 스페인어 컨텍스트(Context) 내에서 NLP 작업을 효과적으로 수행할 수 있게 합니다.



### Rethinking ChatGPT's Success: Usability and Cognitive Behaviors Enabled by Auto-regressive LLMs' Prompting (https://arxiv.org/abs/2405.10474)
- **What's New**: 이 논문은 큰 언어 모델(LLM)의 훈련 및 배포 전략을 분석하며 특히 자유형 입력 및 출력 방식과 음성 자유형 컨텍스트를 사용한 자동 회귀 LLM(AR-LLM)의 프롬프팅 방식의 중요성을 강조합니다. 이를 통해 다양한 인지 행동을 모방하는 방법을 제시하고, LLM을 자율 에이전트 및 다중 에이전트 시스템에서 효과적으로 배포할 수 있는 가능성을 탐색합니다.

- **Technical Details**: LLM 배포는 일반적으로 두 가지 주요 접근 방식을 사용합니다: 자동 인코딩 LLM(AE-LLM)과 자동 회귀 LLM(AR-LLM). AE-LLM은 BERT와 같은 모델로, 입력 시퀀스에 노이즈를 추가하고 이를 복구하는 방식으로 훈련됩니다. 반면, AR-LLM은 GPT 시리즈와 같이 이전 토큰을 기반으로 다음 토큰을 예측하는 방식으로 동작합니다. 이 논문은 특히 다양한 모달리티와 채널을 통해 AR-LLM의 프롬프팅 방식이 더욱 인간적인 인지 행동을 모방할 수 있다는 점을 강조합니다.

- **Performance Highlights**: 프롬프팅 방식을 사용하는 AR-LLM은 과제의 맞춤화, 투명성, 복잡성 측면에서 높은 사용성을 보이며, 인간의 복잡한 인지 행동을 더 효과적으로 모방할 수 있습니다. 이는 사용자의 인식에 있어 LLM의 지능을 더욱 설득력 있게 만들 수 있습니다. 또한, 자유형 텍스트를 통한 컨텍스트 제공으로 추론, 계획, 피드백 학습과 같은 인간의 인지 행동을 AR-LLM이 성공적으로 모방한다는 점을 강조합니다.



### Participle-Prepended Nominals Have Lower Entropy Than Nominals Appended After the Particip (https://arxiv.org/abs/2405.10457)
Comments:
          Accepted to CogSci 2024, 6 pages, 2 figures

- **What's New**: 이번 연구는 복합 분사(composite participles)와 구형 구문(phrasal paraphrase)의 사용 조건과 제약을 비교합니다. 복합 분사는 'London-made'와 같이 구조적 제약이 더 높고, 구형 구문은 'made in London'과 같이 좀 더 자유로운 표현을 허용합니다.

- **Technical Details**: 연구자들은 대형 말뭉치에서 특정 동사와 관련된 명사 위치의 엔트로피(entropy)를 측정하여, 복합 분사가 구형 구문보다 더 낮은 엔트로피를 보인다고 가정했습니다. 이는 복합 분사가 더 일관된 사용 패턴을 가진다는 것을 시사합니다. 이를 위해 영어 인터넷 텍스트 코퍼스(enTenTen20)를 사용하여 분석을 수행했습니다.

- **Performance Highlights**: 연구 결과, 동사와 함께 사용되는 명사의 예측 가능성이 구형 구문보다 복합 분사에서 더 높아 엔트로피가 낮다는 것이 확인되었습니다. 예를 들어, 'tear-stained'는 'stained with tears'보다 명사(Noun)의 예측 가능성이 높다는 점에서 복합 분사의 일관된 조합이 더 많다는 것이 입증되었습니다.



### Navigating Public Sentiment in the Circular Economy through Topic Modelling and Hyperparameter Optimisation (https://arxiv.org/abs/2405.10452)
- **What's New**: 이 논문은 순환 경제(Circular Economy, CE)에 대한 대중의 감정 및 인지 경로를 조사하고, 주요 우려 사항을 이해하도록 하는 것이 목표입니다. 트위터, 레딧, 더 가디언 등 다양한 플랫폼에서 데이터를 수집하고, 이를 분석하기 위해 하이퍼파라미터 최적화(topic modelling) 프레임워크를 제안합니다.

- **Technical Details**: 이 연구는 Latent Dirichlet Allocation (LDA), Correlation Explanation (CorEx), BERTopic 등 세 가지 토픽 모델링 기법을 사용했습니다. 각 모델의 하이퍼파라미터를 최적화하기 위해 Grid Search 전략을 사용하고, Normalised Pointwise Mutual Information (NPMI) 점수를 통해 최적의 하이퍼파라미터를 찾았습니다. BERTopic은 더 가디언과 레딧 데이터셋에서 최고 성과를, CorEx는 트위터 데이터셋에서 최고 성과를 나타냈습니다.

- **Performance Highlights**: 공식 소스에서 CE의 적용 및 규제에 높은 수준의 참여를 보여주며, 지속 가능성과 경제적 영향에 대한 우려가 모든 데이터셋에서 지속적으로 나타났습니다. 이 연구는 다양한 사회 계층에서 CE에 대한 공적 의견을 조사하는 선구적인 연구로, 공공 감정이 CE 주제에 대한 정책, 기업 전략 및 사회적 태도에 큰 영향을 미친다는 것을 강조합니다.



### Simultaneous Masking, Not Prompting Optimization: A Paradigm Shift in Fine-tuning LLMs for Simultaneous Translation (https://arxiv.org/abs/2405.10443)
- **What's New**: 해당 연구는 동시 번역(Simultaneous Translation)을 위해 대형 언어 모델(LLMs)을 미세 조정하는 새 패러다임 SimulMask를 제안합니다. 기존의 프롬프트 최적화 전략보다 높은 번역 품질과 낮은 계산 비용을 달성했습니다.

- **Technical Details**: SimulMask는 주의(attention) 연결을 마스크하여 특정 결정 정책(decision policy) 하에서 동시 번역을 모델링하는 혁신적인 주의 마스크(attention mask) 기법을 사용합니다. 이는 데이터 증강(data augmentation) 없이도 효율적으로 훈련과 추론을 가능하게 합니다. SimulMask는 RefinedWeb 데이터셋으로 사전 훈련된 1.3억 매개변수의 Falcon 모델을 IWSLT 2017 데이터셋으로 미세 조정하고 평가했습니다.

- **Performance Highlights**: SimulMask를 적용한 LLM은 기존의 프리픽스 미세조정(prefix fine-tuning) 방법보다 영어-프랑스어, 영어-네덜란드어, 영어-이탈리아어 언어 쌍에서 각각 BLEU 점수가 7.34, 3.00, 2.98 향상되었습니다. 또한, wait-3 정책에서 A40 GPU를 사용한 KV 캐싱(KV caching)으로 계산 시간의 26.8% 및 FLOPs의 87.9%를 절약하여 계산 비용을 크게 줄였습니다.



### Retrieving and Refining: A Hybrid Framework with Large Language Models for Rare Disease Identification (https://arxiv.org/abs/2405.10440)
- **What's New**: 이 연구는 전통적인 사전기반 자연어 처리(natural language processing, NLP) 도구와 대형 언어 모델(large language models, LLMs)의 강력한 기능을 결합한 새로운 하이브리드 접근 방식을 제안합니다. 이 방법은 비정형 텍스트 데이터에서 희귀 질병을 식별하는 데 있어 수작업의 어려움과 주관성을 줄이는 것을 목표로 합니다.

- **Technical Details**: 연구는 다양한 크기와 도메인(일반 및 의학)에서 여섯 가지 LLMs를 평가합니다. 이러한 LLMs의 문맥 이해 및 추론 능력을 향상시키기 위해 zero-shot, few-shot 및 retrieval-augmented generation (RAG) 기법을 포함한 여러 프롬프트 전략도 실험합니다. ORDO(Orphanet Rare Disease Ontology)와 UMLS(Unified Medical Language System)를 사용하여 비정형 임상 데이터에서 희귀 질병을 식별하기 위해 넓은 용어 사전을 구성했습니다.

- **Performance Highlights**: 본 연구는 환자의 비정형 임상 노트에서 희귀 질병을 식별하는 데 있어 하이브리드 접근 방식이 높은 성능을 보여주는 것을 입증했습니다. 실험 결과, 여러 잠재적인 희귀 질병 사례를 밝혀냈으며, 이는 조기 진단과 환자 치료 개선에 큰 도움이 될 수 있음을 시사합니다.



### Thinking Fair and Slow: On the Efficacy of Structured Prompts for Debiasing Language Models (https://arxiv.org/abs/2405.10431)
Comments:
          The first two authors have equal contribution

- **What's New**: 기존의 디바이싱(debiasing) 기술은 특히 대형 언어 모델(LLM)의 내부나 출력 분포에 접근할 수 있어야 효과적이지만, 일반 사용자에게는 접근이 불가능한 경우가 많습니다. 이번 연구는 이러한 문제를 해결하기 위해 구조화된 프롬프팅(prompting) 기술을 사용하여 공정한 텍스트 생성을 가능하게 하는 방법을 탐구합니다. 특히, 사용자가 LLM에 원하는 방식대로 공정성을 적용할 수 있는 다양한 프롬프팅 기법을 평가했습니다.

- **Technical Details**: 이 연구는 LLM의 바이어스를 해소하기 위해 인간의 시스템 2 사고 과정을 적용하는 프롬프팅 기법에 집중했습니다. 구체적으로 기존의 프롬프팅 기법을 세 가지 큰 범주(프리픽스 프롬프팅, 셀프 리파인먼트, 임플리케이션 프롬프팅)로 조직하고, 단일 vs 다중 단계 프롬프팅, 지시형 vs 역할 기반 프롬프팅의 두 차원으로 분류했습니다. 이를 통해 다양한 LLM과 프롬프팅 전략을 체계적으로 평가했습니다.

- **Performance Highlights**: 이번 연구에서 제안한 시스템 2 기반 임플리케이션 프롬프트(Implicative Prompts)는 기존 기술보다 현저하게 낮은 평균 바이어스를 보여주었으며, 다운스트림 작업에서도 경쟁력 있는 성능을 보였습니다. 제안된 프레임워크는 기존 화이트박스 방식의 디바이싱 기법과 동등한 성능을 보여주며, 다운스트림 작업의 성능 저하 없이도 디바이싱이 가능합니다.



### AmazUtah_NLP at SemEval-2024 Task 9: A MultiChoice Question Answering System for Commonsense Defying Reasoning (https://arxiv.org/abs/2405.10385)
Comments:
          Accepted at SemEval 2024 (Colocated with NAACL 2024)

- **What's New**: SemEval 2024의 BRAINTEASER 태스크는 자연어 처리(NLP) 분야에서 획기적인 시도로, 기존의 언어 분석에서 간과된 측면인 '수평적 사고(lateral thinking)'를 중점적으로 다루고 있습니다. 이 챌린지는 문장 퍼즐(Sentence Puzzle)과 단어 퍼즐(Word Puzzle) 서브태스크로 구성되어 있으며, 언어 모델이 다각적인 사고 능력을 테스트하는 것을 목표로 합니다.

- **Technical Details**: 우리는 최첨단 사전 학습 모델(pre-trained models)을 사용하여 다양한 방식으로 접근했습니다. BERT 및 DeBERTaV3와 같은 모델을 활용한 다중 선택 아키텍처(multiple choice architecture)를 기반으로 하였고, 후속 학습을 위해 문장 퍼즐과 단어 퍼즐 데이터셋을 다양화했습니다. 추가로, GPT-4로 생성한 유머/조크 데이터셋과 RiddleSense 데이터셋을 사용해 모델의 수평적 사고 능력을 강화했습니다.

- **Performance Highlights**: 우리 시스템은 문장 퍼즐 태스크에서 92.5%의 정확도, 단어 퍼즐 태스크에서 80.2%의 정확도를 달성했습니다. 특히, AutoModelForMultipleChoice 아키텍처를 통해 다중 선택 태스크를 처리하는 데 있어 기존의 기초 지식으로 학습된 시스템들보다 우수한 성능을 보였습니다. 우리의 시스템은 문장 퍼즐에서 6위, 단어 퍼즐에서 10위를 차지하며 문장 기반 챌린지에 더 높은 능력을 보였으나 단어 기반 퍼즐에서는 개선의 여지가 있습니다.



### Observational Scaling Laws and the Predictability of Language Model Performanc (https://arxiv.org/abs/2405.10938)
- **What's New**: 이번 논문에서는 기존의 스케일링 법칙(scaling laws) 대신 공개적으로 이용 가능한 약 80개의 모델을 사용하여 스케일링 법칙을 구축하는 관찰적 접근법(observational approach)을 제안합니다. 기존의 모델 학습 과정을 생략하고도 다양한 모델 가족들 간의 성능을 예측할 수 있는 방법을 제시하며, 이를 통해 복잡한 성능 예측 문제를 해결할 수 있습니다.

- **Technical Details**: 이 접근법은 언어 모델 성능이 저차원 가능성 공간(capability space)의 함수라는 간단한 일반화된 스케일링 법칙을 기반으로 합니다. 모델 가족들 간의 학습 연산 효율성(training compute efficiency)과 능력 변환 효율성만 다릅니다. 이를 통해 복잡한 하위 능력(downstream capabilities)과 계산 연산 간의 로그 선형 관계(log-linear relationship)를 발견할 수 있습니다.

- **Performance Highlights**: 관찰적 스케일링 법칙을 통해 작은 모델로부터 복잡한 스케일링 현상을 예측할 수 있으며, 이를 통해 GPT-4와 같은 모델의 성능을 단순한 벤치마크로 예측할 수 있습니다. 또한, 체인 오브 생각(Chain-of-Thought) 및 자기 일관성(Self-Consistency) 같은 훈련 후 개입의 효과 역시 예측이 가능합니다.



### Empowering Small-Scale Knowledge Graphs: A Strategy of Leveraging General-Purpose Knowledge Graphs for Enriched Embeddings (https://arxiv.org/abs/2405.10745)
Comments:
          Accepted for LREC-COLING 2024

- **What's New**: 이 논문은 작은 규모의 도메인 특화 지식 그래프(Knowledge Graph, KG)를 잘 확립된 범용 KGs와 결합하여 임베딩 성능을 향상시키는 새로운 프레임워크를 제안합니다. 이를 통해 작은 도메인 특화 KG가 주요 범용 KG의 지식을 활용해 다운스트림 작업(Downstream tasks)에서 성능 향상을 누릴 수 있습니다. 특히, Hits@10 메트릭에서 최대 44% 성능 향상이라는 뛰어난 결과를 실험적으로 검증했습니다.

- **Technical Details**: 제안된 프레임워크는 엔티티 정렬(entity alignment) 및 연결 작업을 사용하여 도메인 특화 KG와 범용 KG를 자동으로 연결합니다. 그런 다음, 이러한 연결된 KG에서 임베딩을 훈련하는 번역 기반(Translation-based) 임베딩 방법을 사용합니다. 또한, KG 완성 작업(KG completion task)에 대한 성능을 향상시키기 위해 가중치 손실 함수(weighted loss function)를 제안합니다. 이러한 접근 방식은 기존의 KG 구축보다 저비용으로 다양한 분야에 활용될 수 있습니다.

- **Performance Highlights**: 제안된 프레임워크는 엄격한 조건에서도 도메인 특화 KG의 다운스트림 작업 성능을 향상시키는 것으로 입증되었습니다. 실험 결과, 작은 도메인 특화 KG가 범용 KG와 결합될 때, Hits@10 지표에서 최대 44% 성능 향상을 나타냈습니다. 이는 기존 LLM 솔루션보다 더 신뢰할 수 있고 오류를 줄이는 강력한 ML 구현 가능성을 보여줍니다.



### SignLLM: Sign Languages Production Large Language Models (https://arxiv.org/abs/2405.10718)
Comments:
          33 pages, website at this https URL

- **What's New**: 이번 논문에서 우리는 최초의 포괄적인 다국어 수어 데이터셋인 Prompt2Sign을 소개합니다. 이 데이터셋은 미국 수어(ASL)를 비롯한 7개의 다른 언어를 포함합니다. 이 데이터를 통해 텍스트-텍스트(text2text) 및 시퀀스-시퀀스(seq2seq) 번역 모델을 최적화하며, 이를 기반으로 최초의 다국어 수어 생성 모델인 SignLLM을 제안합니다. 이 모델은 입력 텍스트나 프롬프트에서 수어 제스처를 생성할 수 있는 두 가지 새로운 모드를 포함하고 있습니다.

- **Technical Details**: 기존의 주요 데이터셋들은 복잡한 형식을 가지고 있어 직접 훈련에 사용하기 어렵습니다. Prompt2Sign 데이터셋은 이러한 문제를 해결하기 위해 비디오 프레임의 포즈 정보를 표준화된 형식으로 저장하고 자동으로 프롬프트 단어를 생성해 수작업 주석의 필요성을 줄였습니다. SignLLM 모델은 대규모 다국어 수어 생성 모델로, 두 가지 모드를 특징으로 합니다: 다국어 전환 프레임워크(MLSF)와 Prompt2LangGloss입니다. 또한, 훈련 시간을 단축하기 위해 강화 학습(RL) 기반의 새로운 손실 함수 모듈을 도입했습니다.

- **Performance Highlights**: 광범위한 실험과 소속 연구에서 SignLLM은 8개 수어 언어에서 최첨단(SOTA) 성능을 달성했습니다. 본 논문의 주요 기여사항은 포괄적인 다국어 수어 데이터셋과 대규모 수어 생성 모델의 개발로 요약될 수 있습니다. 새로운 손실 함수와 강화 학습을 통한 효율적인 훈련 전략 도입으로 훈련 시간 단축을 실현했습니다.



### SynDy: Synthetic Dynamic Dataset Generation Framework for Misinformation Tasks (https://arxiv.org/abs/2405.10700)
- **What's New**: 새로운 논문 SynDy는 대형 언어 모델(LLMs)의 능력을 활용하여 로컬 맞춤형 모델을 훈련시키는 합성 동적 데이터셋 생성 프레임워크입니다. SynDy는 수작업 주석 데이터의 비용 문제를 해결하며, 이를 통해 인간 주도의 사실 확인 작업의 효율을 높이는 것을 목표로 합니다.

- **Technical Details**: SynDy는 Claim Matching(주장 매칭), Topical Clustering(주제 클러스터링), Claim Relationship Classification(주장 관계 분류)와 같은 작업에 대한 미세한 합성 레이블을 생성하는 방식으로 LLMs을 활용하여 합성 데이터를 자동 생성합니다. 이는 주로 소셜 미디어 쿼리를 사용하여 이루어지며, 인간 주석 데이터 대비 비용 대폭 절감이 가능합니다.

- **Performance Highlights**: SynDy의 합성 레이블로 훈련된 모델은 표준 베이스라인보다 성능이 향상되었으며, 인간 주석 데이터로 훈련된 모델과 비교했을 때 큰 성능 저하 없이 효율이 증명되었습니다. SynDy는 현재 Meedan의 챗봇에 통합되어 50개 이상의 조직, 23만 명 이상의 사용자에게 연간 제공될 예정입니다.



### UniCL: A Universal Contrastive Learning Framework for Large Time Series Models (https://arxiv.org/abs/2405.10597)
- **What's New**: 이번 연구에서는 UniCL이라는 새로운 보편적이고 스케일러블한 대조 학습(contrastive learning) 프레임워크를 소개합니다. 이는 교차 도메인 데이터셋에서 시간 시계열 (time-series) 기초 모델을 사전 학습(pre-training)하는 데 중점을 두고 있습니다. 기존의 고정된 증강(augmentation) 방법과 도메인 특화 데이터 학습의 약점을 극복하려는 접근법입니다.

- **Technical Details**: UniCL은 통합된 학습 가능한 시간 시계열 증강 작업을 제안하여 패턴을 보존하면서 다양한 저편향 데이터 생성을 목표로 합니다. 스펙트럼 정보를 활용하여 시간 시계열 데이터의 고유 패턴을 유지하면서 증강하는 데 중점을 둡니다. 또한, 시간 시계열 데이터의 길이가 다양한 데이터셋을 처리할 수 있는 스케일러블 증강 알고리즘도 소개되었습니다.

- **Performance Highlights**: 11개 도메인에 걸친 두 개의 벤치마크 데이터셋을 활용한 광범위한 실험 결과, UniCL은 다양한 분야의 시간 시계열 분석에서 높은 일반화(generalization) 성능을 입증했습니다.



### A Hybrid Deep Learning Framework for Stock Price Prediction Considering the Investor Sentiment of Online Forum Enhanced by Popularity (https://arxiv.org/abs/2405.10584)
- **What's New**: 이 논문에서는 주식 가격 예측을 위해 온라인 포럼에서 추출된 투자자 감정을 활용하는 새로운 하이브리드 딥 러닝 프레임워크를 제안합니다. 이 프레임워크는 XLNET 모델을 사용하여 사용자 게시물에서 전달되는 감정을 분석하고, 게시물 인기 요소와 결합하여 일일 그룹 감정을 계산한 후, 이를 주식 기술 지표와 통합하여 고급 BiLSTM-highway 모델로 주식 가격을 예측합니다.

- **Technical Details**: 이 연구에서는 XLNET 모델을 활용하여 온라인 포럼 게시글의 텍스트에서 감정을 추출합니다. 추출된 감정과 게시글의 인기 요소를 조합하여 일일 그룹 감정을 계산합니다. 그런 다음, 이러한 감정 데이터를 주식 기술 지표와 결합하여 개선된 BiLSTM-highway 모델에 입력합니다. 이 모델은 양방향 LSTM과 하이웨이 네트워크를 결합하여 주식 가격을 예측하는 능력을 향상시킵니다.

- **Performance Highlights**: 중국 주식 시장의 네 가지 주식을 대상으로 한 일련의 비교 실험을 통해, 제안된 하이브리드 프레임워크가 주식 가격을 효과적으로 예측함을 입증했습니다. 이 연구는 주식 가격 예측에 있어 투자자의 텍스트적 견해를 분석하는 것이 필요함을 강조합니다.



### Memory-efficient Energy-adaptive Inference of Pre-Trained Models on Batteryless Embedded Systems (https://arxiv.org/abs/2405.10426)
Comments:
          This paper has been selected for publication at the 21st International Conference on Embedded Wireless Systems and Networks (EWSN'24)

- **What's New**: 배터리 없는 시스템이 직면한 메모리와 에너지 제약 문제를 해결하기 위해 FreeML이라는 새로운 프레임워크가 소개되었습니다. 이 프레임워크는 초소형 딥러닝 모델을 메모리와 에너지를 효율적으로 사용하여 배터리 없는 환경에서 실행 가능하게 합니다.

- **Technical Details**: FreeML은 두 가지 주요 기능을 갖추고 있습니다. 첫째, SparseComp라는 새로운 압축 기법을 소개하여 선택된 레이어를 재학습하면서 스파시티 제약을 가하여 모델 크기와 런타임 메모리 요구를 동시에 줄입니다. 둘째, gNet은 하나의 단일 종료 분기를 사용하는 첫 번째 조기 종료 메커니즘으로, 모든 레이어에서 종료할 수 있어 메모리 오버헤드를 최소화합니다. 특히, 이는 사전 학습된 모델의 구조를 변경하거나 재학습할 필요 없이 대부분의 DNN 모델에 Plug-and-Play 방식으로 사용할 수 있습니다.

- **Performance Highlights**: SparseComp는 사전 학습된 DNN 모델의 크기를 최대 95배 줄이며, gNet은 전통적인 조기 종료 모델에 비해 메모리 오버헤드를 2.03배에서 19.65배 줄입니다. 또한, gNet은 추론 시간을 10.84%에서 16.19% 단축시키며, 중간 정확도는 2% 증가시킵니다. FreeML은 이렇게 초소형 DNN 모델이 언제나 종료될 수 있도록 하여, 간헐적인 추론 동안 시간과 에너지 이점을 제공합니다.




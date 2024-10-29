New uploads on arXiv(cs.CL)

### GPT-4o System Card (https://arxiv.org/abs/2410.21276)
- **What's New**: GPT-4o는 텍스트, 오디오, 이미지, 비디오 등 다양한 입력을 지원하는 자가 회귀형 omni 모델로, 통합된 신경망을 통해 텍스트, 오디오, 이미지 출력을 생성합니다. 이 모델은 대화에서 인간의 반응 시간과 유사하게 평균 320밀리초로 오디오 입력에 응답할 수 있어 향상된 반응성을 보여줍니다.

- **Technical Details**: GPT-4o는 텍스트와 음성 기능이 2023년 10월까지의 다양한 자료를 바탕으로 사전 훈련되었습니다. 웹 데이터, 코드 및 수학, 다중 모달 데이터가 포함되어 있어 모델이 비텍스트 입력과 출력을 해석하고 생성하는 방법을 학습하게 됩니다. 또한, 위험을 줄이기 위한 방법으로는 Moderation API와 안전 분류기를 사용하여 유해 콘텐츠를 필터링합니다.

- **Performance Highlights**: GPT-4o는 영어 및 코드 텍스트에서 GPT-4 Turbo의 성능을 맞추며 비영어 텍스트에서는 현저한 개선을 보여줍니다. API 비용은 50% 절감되며, 비전 및 오디오 이해 능력이 기존 모델보다 뛰어난 것으로 확인되었습니다.



### Arithmetic Without Algorithms: Language Models Solve Math With a Bag of Heuristics (https://arxiv.org/abs/2410.21272)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)이 산술 추론 작업을 수행하는 방식을 분석합니다. 연구를 통해 LLMs가 강력한 알고리즘을 학습하여 문제를 해결하는 것이 아니라, 각각의 간단한 기법으로 구성된 '휴리스틱(heuristic) 병'을 사용하여 정확한 계산을 수행함을 밝혀냈습니다.

- **Technical Details**: 연구진은 LLMs 내부의 산술 회로를 분석하기 위해 인과 분석(causal analysis)을 사용하였고, 각 회로 신경망(neuron)이 특정 패턴의 숫자 입력에 반응하여 결과를 출력하는 원리를 조사하였습니다. 여러 종류의 휴리스틱으로 신경망을 분류하고, 이러한 무작위 조합이 산술 문제에 대한 모델의 정확성을 설명하는 메커니즘임을 발견하였습니다.

- **Performance Highlights**: 이 연구의 결과로 LLMs는 훈련 초기부터 산술 문제 해결에 있어 휴리스틱 병 방식을 주요 메커니즘으로 사용한다는 것을 보여주었으며, 실험을 통해 LLMs가 기억에 의존하지 않고도 산술 연산을 수행할 수 있음을 입증했습니다.



### EoRA: Training-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation (https://arxiv.org/abs/2410.21271)
- **What's New**: 본 연구에서는 모델 압축 문제를 사용자 맞춤형 보상 문제로 재구성하였습니다. 압축된 모델을 기준으로 잔여 저랭크 경로를 도입하여 압축 오류를 보상하는 접근법을 제안합니다.

- **Technical Details**: 제안된 EoRA(Training-free Eigenspace Low-Rank Approximation)는 압축 오류를 입력 활성화의 고유 공간 (eigenspace)으로 투사하여 오류 부품의 재구성을 우선시합니다. 이 방법은 고유값 (eigenvalues)을 활용하여 효과적인 저랭크 표현 능력을 발휘하고, 기울기 계산 없이도 빠른 최적화를 가능하게 합니다.

- **Performance Highlights**: EoRA는 LLaMA2/3 모델의 다양한 작업에서 이전 방법들을 상회하는 성능을 보여주며, 언어 생성과 상식 추론 및 수학 추론 작업에서 31.31% 및 9.69%의 성능 향상을 달성하였습니다. EoRA는 또한 4-비트 양자화된 모델에서 원래 모델보다 더 나은 정확성을 보여 주기도 했습니다.



### Are BabyLMs Second Language Learners? (https://arxiv.org/abs/2410.21254)
- **What's New**: 본 논문은 2024년 BabyLM Challenge를 위한 언어학적 접근을 설명하고 있으며, 첫 번째 언어(L1) 학습 Paradigm 대신 두 번째 언어(L2) 학습 관점에서 이 도전에 접근하고 있습니다. 즉, 이는 언어의 명시적 정보 학습에 중점을 둡니다.

- **Technical Details**: L2 학습 관점에서 우리는 Wiktionary의 데이터, LLM에 의해 생성되거나 문법서에서 발췌한 문법 예시, 패러프레이즈(paraphrase) 데이터를 사용하여 모델 학습을 진행했습니다. 연구 결과, 단어 의미에 대한 명시적 정보는 모델 성능을 향상시키지 않지만, 문법 정보는 약간의 개선을 주었으며, 문장 패러프레이징이 가장 큰 영향을 미치는 데이터 요소로 확인되었습니다.

- **Performance Highlights**: 모델의 성능은 다음 두 가지 데이터 조합에서 가장 높았습니다: 1) 패러프레이즈 데이터와 BabyLM 데이터의 혼합, 2) 패러프레이즈 데이터만 사용한 경우. 이러한 결과는 특정 데이터 유형이 BabyLM 모델에 대한 L2 학습에 중요한 역할을 한다는 것을 보여줍니다.



### LongReward: Improving Long-context Large Language Models with AI Feedback (https://arxiv.org/abs/2410.21252)
- **What's New**: 이 논문에서는 LongReward라는 새로운 방법을 제안하여 LLM(대형 언어 모델)의 긴 컨텍스트 응답에 대한 보상을 제공하는 방식을 다룹니다. LongReward는 도움성(helpfulness), 논리성(logicality), 충실성(faithfulness), 완전성(completeness)이라는 네 가지 측면을 기반으로 보상을 평가합니다.

- **Technical Details**: LongReward는 오프더셀프 LLM을 사용하여 긴 컨텍스트 모델 응답에 대한 보상을 제공합니다. 각 기준에 대해 0부터 10까지 점수를 부여하고 평균 점수를 최종 보상으로 사용합니다. 이 방법과 DPO(Direct Preference Optimization)와 같은 강화학습(RL) 알고리즘을 결합하여 긴 컨텍스트 SFT 모델의 성능을 향상시킵니다.

- **Performance Highlights**: LongReward를 적용한 결과, Llama-3.1-8B와 GLM-4-9B 모델에서 긴 문맥 작업에 대해 각각 4.9% 및 5.5%의 성능 향상을 보였으며, 인간 평가에서도 SFT 기준에 대해 46% 더 많은 승리를 기록했습니다. 또한, LongReward는 단기 지침에 대한 모델의 수행 능력을 높이는 데에도 기여합니다.



### HoPE: A Novel Positional Encoding Without Long-Term Decay for Enhanced Context Awareness and Extrapolation (https://arxiv.org/abs/2410.21216)
- **What's New**: 본 논문에서는 기존 Positional Encoding (PE)의 긴 거리 토큰에 대한 규칙인 'long-term decay'의 필요성을 반박하고, 이를 대체할 새로운 방법인 High-frequency rotary Position Encoding (HoPE)를 제안합니다. HoPE는 Rotary Position Encoding (RoPE)의 한계를 해결하고 모델의 맥락 인지와 추론 능력을 향상시킵니다.

- **Technical Details**: 기존의 PE 모델은 긴 거리 토큰이 모델의 정보 처리에 미치는 영향을 줄이는 long-term decay 원칙을 따르지만, 실험 결과 LLMs에서 모델들은 단기적인 정보에만 집중하는 경향을 보입니다. HoPE는 RoPE의 구성 요소 중 위치 독립적인 것과 높은 주파수의 신호로 대체하여 모델의 전반적인 성능을 향상시킵니다.

- **Performance Highlights**: HoPE는 언어 모델링의 perplexity, 복사 작업, 그리고 few-shot 학습 작업에서 기존의 PE 방식보다 우수한 성능을 보여주었습니다. 이 연구는 특정 주파수의 'activated' 구성 요소가 RoPE의 성능 한계를 유발했음을 보여줍니다.



### BongLLaMA: LLaMA for Bangla Languag (https://arxiv.org/abs/2410.21200)
Comments:
          19 pages

- **What's New**: BongLLaMA(즉, Bangla-LLaMA)라는 새로운 오픈 소스 대형 언어 모델이 발표되었습니다. 이는 대규모 방글라어 코퍼스를 기반으로 독점적으로 파인 튜닝(fine-tuning)된 모델입니다.

- **Technical Details**: 본 연구에서는 방글라어 데이터 증강(data augmentation) 기법, 파인 튜닝 세부 사항, 그리고 BongLLaMA 모델의 효과성을 입증하는 종합적인 벤치마킹(benchmarking) 결과를 제시합니다. 이 모델은 방글라어 언어 처리(BLP) 작업을 위한 최적화된 성능을 자랑합니다.

- **Performance Highlights**: BongLLaMA는 방글라어 언어 모델의 새로운 기준이 될 것으로 예상되며, 향후 방글라어의 다양한 언어 처리 연구에 유용한 기초 자료를 제공합니다. 모든 BongLLaMA 모델은 공공용으로 제공됩니다.



### Belief in the Machine: Investigating Epistemological Blind Spots of Language Models (https://arxiv.org/abs/2410.21195)
Comments:
this https URL

- **What's New**: 이 연구는 기존 언어 모델들이 사실, 신념, 지식의 차이를 구분하는 능력에 대한 연구가 부족하다는 점을 지적하며, 13,000개의 질문을 포함한 새로운 데이터셋인 KaBLE를 통해 이들 능력을 체계적으로 평가하였습니다.

- **Technical Details**: 연구는 현대 언어 모델인 GPT-4, Claude-3, Llama-3 등 15개 모델을 대상으로 하였으며, 이들 모델이 신념과 지식, 사실의 차이를 처리하는 능력을 측정했습니다. 개별 과제를 통해 얻어진 결과에 따르면, 언어 모델은 필수적인 상식적 구분에 대해 심각한 한계를 보였습니다.

- **Performance Highlights**: 모델들은 사실적 시나리오에서는 86%의 정확도를 보였지만, 잘못된 시나리오에서는 성능이 급락했습니다. 특히 첫 번째 인칭 신념 확인 작업에서 54.4%의 낮은 정확도를 나타내며, 개인의 신념을 인식하고 지지하는 데 어려움을 겪고 있습니다. 이러한 발견은 의료 및 상담과 같은 분야에서의 모델 적용에 중대한 우려를 불러일으킵니다.



### M2rc-Eval: Massively Multilingual Repository-level Code Completion Evaluation (https://arxiv.org/abs/2410.21157)
Comments:
          19 pages

- **What's New**: 이번 논문에서는 다양한 프로그래밍 언어를 포괄하는 대규모 멀티링구얼(18개 언어) 리포지토리 레벨의 코드 완성 기준을 제안합니다. 이를 통해 기존 코드 대형 언어 모델(LLMs)의 전반적인 코드 지능 능력을 평가할 수 있으며, 세분화된 주석을 통해 더 정교한 결과를 도출할 수 있습니다.

- **Technical Details**: M2RC-EVAL 데이터셋은 18개의 프로그래밍 언어를 포함하며, 버킷 수준(bucket-level)과 의미 수준(semantic-level)의 두 가지 세분화 주석을 제공합니다. 이 주석들은 파싱된 추상 구문 트리(abstract syntax tree)를 기반으로 구축됩니다. M2RC-INSTRUCT 데이터셋은 코드 LLM의 리포지토리 레벨 코드 완성 능력을 개선하기 위해 마련되었습니다.

- **Performance Highlights**: M2RC-EVAL과 M2RC-INSTRUCT의 종합적인 실험 결과는 제안된 방법이 기존 코드 LLM의 성능을 향상시킬 수 있음을 입증하며, 다양한 언어의 코드 완성 시나리오에서의 세분화된 성능 평가 가능성을 제공합니다.



### SciER: An Entity and Relation Extraction Dataset for Datasets, Methods, and Tasks in Scientific Documents (https://arxiv.org/abs/2410.21155)
Comments:
          EMNLP2024 Main

- **What's New**: 본 연구에서는 데이터셋, 방법 및 작업과 관련된 개체(entities)와 관계(relations)를 추출하기 위한 새로운 데이터셋인 SciER을 발표합니다. 이 데이터셋은 106개의 전체 텍스트 과학 출판물로 구성되어 있으며 24,000개 이상의 개체와 12,000개의 관계를 포함합니다. SciER은 전체 텍스트에서 개체 간의 복잡한 사용 및 상호작용을 포착하기 위해 세분화된 태그 세트를 제공합니다.

- **Technical Details**: SciER 데이터셋은 과학 문서에서 데이터셋, 방법 및 작업 개체를 식별하고 그들 간의 관계를 추출하기 위해 설계되었습니다. 기존 데이터셋이 특정 부분에서만 정보를 추출한 것과 달리, 이 데이터셋은 전체 논문을 수작업으로 주석 처리하여 보다 정확하고 세부적인 관계 유형을 제공합니다. 또한 OOD(out-of-distribution) 테스트 세트를 제공하여 모델의 실제 평가를 보장합니다.

- **Performance Highlights**: 실험에서는 최첨단 감독 방식(supervised models)과 제안된 LLM 기반 기초(line methods)를 포함하여 우수한 성능을 달성했습니다. 최상의 감독 방법은 ERE(엔티티 및 관계 추출) 작업에서 F1 점수가 61.10%를 기록했으며, LLM 기반 방법은 41.22%의 F1 점수를 기록했습니다. 이러한 결과는 LLM 모델이 더 나은 성능을 발휘하기 위해 적절한 태스크 분할(pipeline modeling)이 필요함을 보여줍니다.



### Palisade -- Prompt Injection Detection Framework (https://arxiv.org/abs/2410.21146)
- **What's New**: 이 논문은 Large Language Models (LLMs)의 새로운 약점을 다루고 있습니다. 특히, 악의적인 프롬프트 주입 공격(prompt injection attacks)에 대한 새로운 탐지 방법을 제안합니다.

- **Technical Details**: 이 논문에서 제안하는 방법은 세 가지 계층(layer)으로 구성된 입력 필터링 시스템을 사용합니다. 첫 번째는 규칙 기반(rule-based) 필터, 두 번째는 머신러닝(Machine Learning) 분류기(classifier), 세 번째는 동반 LLM(companion LLM)입니다. 이 시스템은 다양한 공격을 효과적으로 탐지하고, 악성 입력으로부터 모델을 보호합니다.

- **Performance Highlights**: 제안된 다계층 탐지 프레임워크는 개별 계층 중에서 ML 분류기가 가장 높은 정확도를 보였으며, 전반적인 탐지 정확도를 향상시키는 데 기여했습니다. 이는 잘못된 음성(false negatives)을 줄이는 동시에, 실제로 주입된 프롬프트를 간과할 위험을 최소화합니다. 이 접근 방법은 LLM의 취약점(vulnerabilities)을 강조하며, 향후 연구를 위한 포괄적인 프레임워크를 제공합니다.



### uOttawa at LegalLens-2024: Transformer-based Classification Experiments (https://arxiv.org/abs/2410.21139)
- **What's New**: 이 논문은 LegalLens-2024 공유 작업에 대해 소개하며, 비구조적 텍스트 데이터에서 법적 위반을 감지하고 이러한 위반이 잠재 영향을 미치는 개인과 연관되는 방법에 초점을 맞추었습니다. 본 작업은 Legal Named Entity Recognition (L-NER)과 Legal Natural Language Inference (L-NLI)의 두 가지 하위 작업을 포함하고 있습니다.

- **Technical Details**: 하위 작업 A에서 우리는 spaCy 라이브러리를 사용하였고, 하위 작업 B에서는 RoBERTa와 CNN을 결합한 모델을 적용했습니다. L-NER 하위 작업에서는 86.3%의 F1-score(매크로 F1 점수)를 달성하였고, L-NLI 하위 작업에서는 88.25%의 성과를 보였습니다. 이러한 결과는 법적 분야에서 복잡한 작업을 해결하는 데 있어 transformer 모델의 효과성을 잘 보여줍니다.

- **Performance Highlights**: 우리는 L-NER 하위 작업에서 spaCy 파이프라인을 구성하여 벡터 초기화 및 토큰화를 수행했습니다. 이 과정을 통해 BERT 기반 모델이 가장 우수한 성능을 보였으며, 숨겨진 테스트 세트에서 F1 점수 0.402를 기록하여 3위에 올랐습니다.



### Retrieval-Enhanced Mutation Mastery: Augmenting Zero-Shot Prediction of Protein Language Mod (https://arxiv.org/abs/2410.21127)
Comments:
          25 pages, 10 figures, 8 tables

- **What's New**: 이 논문에서는 깊이 있는 학습(deep learning) 기술을 활용하여 단백질 변이 효과 예측을 다루는 새로운 프로틴 언어 모델(ProtREM)을 소개합니다. ProtREM은 단백질 서열(sequence)과 구조(structure) 정보, 그리고 진화적 자료(evolutionary information)를 통합하여 자연적인 특성을 분석하는 데 중점을 두고 있습니다.

- **Technical Details**: ProtREM은 시퀀스-구조 임베딩(sequence-structure embeddings)을 다중 헤드 크로스-어텐션(multi-head cross-attention) 레이어를 통해 통합하여 모델을 훈련시킵니다. 또한, 동종 유전자(homologous sequences)를 활용하여 진화적 표현(evolutionary representations)을 도출하여 변이의 적합도를 평가합니다. 이 모델은 217가지 어세이(assays)에서 2222만 개 이상의 돌연변이에 대한 데이터를 처리하여 성능을 검증하였습니다.

- **Performance Highlights**: ProtREM은 공개 벤치마크인 ProteinGym의 데이터를 활용하여 다양한 단백질 및 어세이에 대한 예측에서 우수한 성능을 보였습니다. 일반적으로 다른 기법들과 비교하여 안정성과 적합도 예측 성능 모두에서 뛰어난 결과를 달성하였으며, 특히 VHH 항체와 phi29 DNA 중합효소에 대해 실험적으로도 확인된 바 있습니다. 이는 생물학자들이 효소 변형에 있어 더욱 신뢰할 수 있는 정보를 제공함을 의미합니다.



### Current State-of-the-Art of Bias Detection and Mitigation in Machine Translation for African and European Languages: a Review (https://arxiv.org/abs/2410.21126)
- **What's New**: 본 연구는 자연어 처리(NLP) 기술에서 발생하는 편향(bias) 탐지 및 완화(composition) 방법을 다루며, 특히 기계 번역(machine translation) 분야에서의 중요성을 강조합니다. 유럽 및 아프리카 언어에 중점을 두어 현재의 연구 경향을 막대합니다.

- **Technical Details**: 우리는 기계 번역 AND bias 쿼리를 기반으로 한 문헌 검색을 Web of Science에서 수행했습니다. 엑셀 파일로 처리된 결과를 필터링하여 아프리카 및 유럽 언어가 포함된 11개 및 39개의 연구 논문을 확인했습니다. 이후 ACL Anthology에서 추가 검색을 통해, 아프리카 언어 9편과 유럽 언어 25편을 확인했습니다. 각 연구에서 편향 탐지 및 완화 방법론이 비교되었습니다.

- **Performance Highlights**: 대부분의 연구는 영어, 독일어, 프랑스어, 스페인어와 같은 고자원(high-resource) 언어에 집중되어 있으며, 한국어와 같은 비교적 덜 연구된 언어는 적은 수의 연구만을 보이고 있습니다. 이러한 연구 분석을 통해, 다양한 언어와 시스템에서 편향이 어떻게 나타나는지를 밝혀내기 위한 여러 방법론이 제시되었습니다.



### Stealthy Jailbreak Attacks on Large Language Models via Benign Data Mirroring (https://arxiv.org/abs/2410.21083)
- **What's New**: 이 논문은 안전한 LLM(대형 언어 모델)의 취약성을 탐구하기 위해 ShadowBreak라는 새로운 jailbreak 공격 방법을 제안합니다. 이 방법은 benign data mirroring 기법을 사용하여 공격의 은폐성을 높이면서도 공격 성공률을 유지합니다.

- **Technical Details**: ShadowBreak 방법은 benign 데이터에 대해 타겟 블랙박스 모델에 맞춰 '미러' 모델을 지역적으로 훈련하여 악의적인 프롬프트를 구성합니다. 이를 통해 정체가 드러나는 악의적인 명령을 실제 공격 과정에서 제출하지 않고도 쿼리를 생성할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 GPT-3.5 Turbo에 대해 평균 1.5개의 감지 가능한 쿼리로 최대 92%의 공격 성공률(Attack Success Rate, ASR)을 달성하였습니다. 이는 기존의 PAIR 방법보다 훨씬 적은 감지 가능한 쿼리로 더 나은 결과를 보여줍니다.



### CRAT: A Multi-Agent Framework for Causality-Enhanced Reflective and Retrieval-Augmented Translation with Large Language Models (https://arxiv.org/abs/2410.21067)
- **What's New**: 본 논문에서는 문맥에 따라 의존적인 용어들, 특히 새로운 또는 분야 특화된 단어를 효과적으로 처리하기 위한 CRAT라는 새로운 다중 에이전트 번역 프레임워크를 제안합니다. 기존의 수작업 식별 방법에 의존하지 않고, 자동으로Unknown Terms를 식별하고 맥락에 따라 의미를 명확히 하여 번역의 정확성을 높입니다.

- **Technical Details**: CRAT 프레임워크는 Unknown Terms Identification agent, Knowledge Graph Constructor agent, Causality-enhanced Judge agent, Translator agent로 구성되어 있습니다. 이 에이전트들은 각각 불확실한 용어들을 식별하고, 해당 용어에 대한 내부 및 외부 정보를 수집하여 Knowledge Graph( KG)를 구축하며, 최종 번역 출력을 생성하는 과정에서 중요합니다. 이 프레임워크는 Retrieval-Augmented Generation (RAG) 및 원인-결과 관계가 강화된 자기 반성을 활용하여 번역의 정확성을 높입니다.

- **Performance Highlights**: CRAT를 이용한 실험 결과는 번역 정확성이 크게 향상되었음을 보여주며, 특히 문맥에 민감한 용어와 신규 어휘 처리에서 그 효과가 두드러집니다. 이를 통해 번역의 일관성을 높이고 불확실한 용어에 대한 더 정밀한 관리를 가능하게 합니다.



### Semantic Component Analysis: Discovering Patterns in Short Texts Beyond Topics (https://arxiv.org/abs/2410.21054)
Comments:
          5 pages, 3 figures, code: this https URL

- **What's New**: 새로운 주제 모델링 기법인 Semantic Component Analysis (SCA)를 소개하며, 이 기법은 짧은 텍스트에서 단일 주제를 넘어 여러 세분화된 의미 구성 요소를 발견할 수 있습니다.

- **Technical Details**: SCA는 클러스터링 기반 주제 모델링 프레임워크에 분해 단계를 도입하여 짧은 텍스트에서 다수의 의미 구성 요소를 발견합니다. 이 기법은 각 클러스터에 대해 잔여 임베딩을 반복적으로 처리하여 새로운 구성 요소를 찾아내며, cosine distance를 사용하여 UMAP에서 임베딩을 5차원 유클리드 공간으로 매핑합니다.

- **Performance Highlights**: SCA는 여러 Twitter 데이터셋에서 평가되었으며, BERTopic과 유사한 주제 일관성 및 다양성을 제공하면서도 최소 두 배 이상의 의미 구성 요소를 발견하고, 노이즈 비율은 거의 제로에 가깝게 유지함을 보여주었습니다.



### Frequency matters: Modeling irregular morphological patterns in Spanish with Transformers (https://arxiv.org/abs/2410.21013)
- **What's New**: 이 연구는 비정형 변화 형태(pattern)의 학습 행동을 평가하기 위해 transformer 기반 신경망을 활용합니다. 특히 스페인어의 비정형 동사에 초점을 맞추어, 빈도(frequency)에 따라 모델을 비교하고 분석합니다.

- **Technical Details**: 연구는 morphological reinflection task를 사용하여 Paradigm Cell Filling Problem(PCFP)을 모델링합니다. 다양한 비정형 및 정형 동사를 포함한 데이터 corpus에서 모델을 훈련하며, L-shaped morphomic patterns를 신경망을 통해 학습합니다. 또한, 다중 소스 형태-태그 쌍(multi-source form-tag pairs)을 활용하여 비정형 패턴의 학습에 대한 심도 있는 분석을 진행합니다.

- **Performance Highlights**: 실험 결과, 신경망 모델은 비정형 패턴을 놀라울 정도로 잘 학습하며, 빈도 조건에 관계없이 성능을 보여주었습니다. post-hoc 분석을 통해 오류의 가능한 원인도 밝혀냈습니다.



### FACT: Examining the Effectiveness of Iterative Context Rewriting for Multi-fact Retrieva (https://arxiv.org/abs/2410.21012)
Comments:
          Work in Progress

- **What's New**: 본 논문에서는 LLMs(대형 언어 모델)가 여러 사실을 동시에 검색하는 데 어려움을 겪는 "lost-in-the-middle" 현상을 새롭게 규명하고, 이를 해결하기 위한 새로운 접근법인 FACT(모든 중요한 텍스트 찾기)를 제안합니다.

- **Technical Details**: FACT는 반복적(iterative) 검색 방법으로, 문맥을 여러 차례 재작성하여 정제(refine)하는 방식을 사용합니다. 이 방법은 단일 검색에서는 종종 간과되는 기본 사실들을 점진적으로 포착(capture)할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, FACT는 다양한 과제에 걸쳐 다중 사실(multi-fact) 검색 성능을 크게 향상시켰으나, 일반적인 QA(질문-답변) 시나리오에서는 개선 효과가 덜 두드러졌습니다. 이 연구는 LLMs의 다중 사실 검색에서의 한계를 조명하고 있으며, 보다 회복력이 강한 긴 문맥 검색 전략의 필요성을 강조합니다.



### Is GPT-4 Less Politically Biased than GPT-3.5? A Renewed Investigation of ChatGPT's Political Biases (https://arxiv.org/abs/2410.21008)
- **What's New**: 이번 연구는 ChatGPT의 정치적 편향과 성격 특성을 조사하며, 특히 GPT-3.5와 GPT-4를 비교합니다. 연구에서는 Political Compass Test와 Big Five Personality Test를 활용하여 모델들이 정치적 관점을 모방하는 능력을 분석합니다.

- **Technical Details**: 연구에서는 각 시나리오에서 100회 수행된 Political Compass Test와 Big Five Personality Test를 사용하여 평균, 표준 편차 및 유의성 검정을 통해 GPT-3.5와 GPT-4 간의 차이를 조사하였습니다. 결과적으로 두 모델 모두 진보적이며 자유지상주의적 정치 편향을 보였으며, GPT-4는 약간 덜 뚜렷한 편향을 나타냈습니다.

- **Performance Highlights**: GPT-3.5는 경제 축에서 -6.59, 사회 축에서 -6.07을 기록한 반면, GPT-4는 각각 -5.40과 -4.73을 기록했습니다. GPT-4는 할당된 정치적 관점을 모방하는 능력이 뛰어나, 지정된 사분면을 정확하게 반영했습니다. Big Five Personality Test에서 GPT-3.5는 Openness와 Agreeableness에서 높은 점수를 보였으며, GPT-4는 전반적으로 낮은 점수를 보였지만 Neuroticism 점수가 상대적으로 높았습니다.



### DeTeCtive: Detecting AI-generated Text via Multi-Level Contrastive Learning (https://arxiv.org/abs/2410.20964)
Comments:
          To appear in NeurIPS 2024. Code is available at this https URL

- **What's New**: 새롭게 제안된 DeTeCtive 프레임워크는 저자 간의 다양한 문체를 구별하는 데 초점을 두어, AI 생성 텍스트 감지 문제를 새롭게 접근합니다. 이러한 접근법은 기존의 이진 분류 방식에서 벗어나, 작가의 고유한 문체를 학습하는 특성을 반영합니다.

- **Technical Details**: DeTeCtive는 다중 작업 보조 및 다중 수준 대비 학습(multi-level contrastive learning) 프레임워크를 사용하여 다양한 저자의 쓰기 스타일을 학습합니다. 이 방법은 텍스트 인코더와 결합되어 AI 생성 텍스트 감지 기능을 강화하며, K-최근접 이웃(KNN) 알고리즘을 통한 유사도 측정을 사용합니다. 또한, Training-Free Incremental Adaptation (TFIA) 기능을 통해 OOD 데이터에 대한 모델의 적응력을 향상시킵니다.

- **Performance Highlights**: 제안된 방법은 여러 데이터셋에서 기존 방법보다 우수한 성능을 보이며, 특히 OOD 데이터에 대한 제로샷 평가에서 기존 방법을 크게 능가하는 결과를 기록했습니다. AvgRec 메트릭에서 Unseen Models과 Unseen Domains 테스트 세트에서 각각 5.58% 및 14.20%의 성능 향상이 확인되었습니다.



### Instruction-Tuned LLMs Succeed in Document-Level MT Without Fine-Tuning -- But BLEU Turns a Blind Ey (https://arxiv.org/abs/2410.20941)
- **What's New**: 본 연구에서는 instruction-tuned LLMs의 문서 수준 번역(docMT) 능력을 조사한 논문입니다. LLM이 전통적인 방법과 달리 전체 문서를 한 번에 번역할 수 있는지를 평가하고, BLEU 점수 대신 GPT-4를 활용한 새로운 평가 방식을 제안합니다.

- **Technical Details**: 연구에서는 두 가지 번역 접근법을 비교합니다: ST[k]는 여러 문장을 조합하여 번역하는 방법이며, DOC는 LLM에게 전체 문서를 직접 번역하도록 지시하는 방법입니다. 기존의 BLEU 점수는 문서 수준 번역의 품질을 제대로 반영하지 못하기 때문에, LLM을 평가자로 활용하여 일관성, 정확성 및 유창성을 평가합니다.

- **Performance Highlights**: 결과적으로, DOC 접근법이 개별 문장을 번역한 후 조합하는 방법보다 문서 맥락을 잘 활용하여 더 높은 번역 품질을 보여주었습니다. 그러나 BLEU 점수는 이 결과를 잘 반영하지 못하며, 오히려 문서 번역의 실제 품질을 오도할 수 있음을 강조합니다.



### Attacking Misinformation Detection Using Adversarial Examples Generated by Language Models (https://arxiv.org/abs/2410.20940)
- **What's New**: 이 논문에서는 낮은 신뢰성을 가진 콘텐츠를 탐지하는 텍스트 분류 알고리즘의 강인성을 테스트하기 위한 적대적 예제(adversarial examples) 생성에 대해 다룹니다. 저자는 TREPAT(TTracing REcursive Paraphrasing for Adversarial examples from Transformers)라는 솔루션을 제안하며, 이는 대형 언어 모델(LLMs)의 내용을 재구성하는 능력을 활용합니다.

- **Technical Details**: TREPAT 방법론은 입력 텍스트를 작은 조각으로 나누고, 각 조각에 대해 다양한 프롬프트를 사용하여 재구성을 수행합니다. 그 후, 생성된 재구성은 원래 텍스트에 적용되도록 세분화를 통해 변경하고, Beam Search 절차에 따라 피해자 분류기의 반응을 참조하여 반복적으로 적용됩니다. 최종적으로 피해자 분류기의 결정을 변경할 때까지 이 과정을 반복합니다. 이러한 접근 방식은 상업 플랫폼의 실제 제약을 반영합니다.

- **Performance Highlights**: 평가 결과 TREPAT은 대부분의 시나리오에서 베이스라인 및 최첨단(SOTA) 솔루션보다 우수한 성능을 나타냅니다. 특히 긴 입력 텍스트(뉴스 기사)의 경우, 고전적인 방식으로는 탐색이 어려운 경우에서도 효과적인 성능을 유지했습니다.



### Autoformalize Mathematical Statements by Symbolic Equivalence and Semantic Consistency (https://arxiv.org/abs/2410.20936)
Comments:
          Published as a conference paper at NeurIPS 2024. Code is available at [this https URL](this https URL)

- **What's New**: 이 논문에서는 자연어 설명을 형식 언어로 자동 번역하는 과제인 autoformalization의 새로운 접근 방식을 제안합니다. 대형 언어 모델(LLMs)이 경쟁 수준의 수학 문제를 형식화할 수 있는 능력이 있지만, pass@1과 pass@k 정확도 간의 현저한 격차를 발견하였습니다. 이를 해결하기 위해, 새로운 프레임워크를 도입하여 두 가지 보완적 자기 일관성 방법인 symbolic equivalence와 semantic consistency를 이용해 k개의 autoformalization 후보 중 최상의 결과를 선택하고 점수를 매깁니다.

- **Technical Details**: 제안된 접근 방식은 두 가지 방법에 기반합니다. 첫째, symbolic equivalence는 자동 정리 증명기를 사용하여 autoformalization 후보 간의 논리적 동질성을 확인합니다. 둘째, semantic consistency는 후보를 비공식화(informalizing)하여 원래 의미의 보존을 평가하며, 원본 텍스트와 비공식화된 텍스트의 임베딩(similarity) 간의 유사성을 계산합니다. 이를 통해 autoformalization 과정에서 원래의 의미와 일관성을 유지하고자 합니다.

- **Performance Highlights**: MATH와 miniF2F 데이터셋에서의 실험 결과, 우리의 접근 방식이 autoformalization 정확도를 상당히 향상시키며, 다양한 LLM 및 기준 방법에 대해 0.22-1.35배의 상대적 개선을 달성했습니다. 특히, symbolic equivalence와 semantic consistency를 조합한 최종 개선율은 최대 10.7%에 달하며, 이는 수동 검증 또는 레이블링의 필요성을 줄이고 인간의 개입을 최소화하는 데 큰 도움이 됩니다.



### Long Sequence Modeling with Attention Tensorization: From Sequence to Tensor Learning (https://arxiv.org/abs/2410.20926)
- **What's New**: 이 논문에서는 긴 텍스트 데이터 처리에 대한 수요 증가와 관련하여 긴 시퀀스 모델링의 한계를 극복하기 위해 Tensorized Attention 메커니즘을 제안합니다. 이 메커니즘은 긴 입력 시퀀스를 컴팩트한 tensor 표현으로 변환한 후, 각 변환된 차원에 대해 attention을 수행하는 방식입니다.

- **Technical Details**: 제안된 방법은 높은 차원의 tensor로 토큰 상호작용을 변환하며, 각 차원에서 짧은 범위 상호작용을 효율적으로 모델링할 수 있습니다. Tensorized Attention은 기존의 attention 메커니즘보다 효율적이며, Kronecker decomposition을 통해 전체 attention을 근사합니다. 이 메커니즘은 Triton 커널을 활용하여 설계되었으며, 미리 훈련된 LLM에서 계속 학습하는 방식으로 적응할 수 있습니다.

- **Performance Highlights**: 실험 결과, Tensorized Attention은 32,768의 컨텍스트 길이에서 Llama-8B 모델을 훈련시킬 수 있으며,추론 동안 128k 길이로 안정적으로 외삽할 수 있습니다. 이 과정에서 $11	imes$의 속도 향상을 보여줍니다.



### NeuGPT: Unified multi-modal Neural GP (https://arxiv.org/abs/2410.20916)
- **What's New**: NeuGPT는 다양한 신경 기록 데이터를 통합하여 처리할 수 있는 새로운 다중 모달 언어 생성 모델입니다. 해당 모델은 EEG, MEG, ECoG 등 여러 신호 유형의 데이터를 결합하여 보다 나은 뇌-언어 전환을 가능하게 합니다.

- **Technical Details**: NeuGPT는 광고된 대로, 다양한 신경 기록 및 음성, 텍스트 데이터와 상호작용할 수 있도록 설계되었습니다. 특히 뇌에서 텍스트로의 변환(decoding) 기능을 중심으로 하여 BLEU-1에서 SOTA(최신 기술 기준)를 6.94에서 12.92로, ROUGE-1F에서는 6.93에서 13.06으로 향상시켰습니다.

- **Performance Highlights**: NeuGPT는 서로 다른 모달리티를 통합하여 뇌 신호를 텍스트로 변환하는 성능을 획기적으로 개선하였으며, 특히 ECoG로부터 뇌-텍스트 디코딩에서 뛰어난 성과를 보이고 있습니다. 또한, 코드와 모델은 공개되어 있어 연구자들이 활용할 수 있습니다.



### AutoRAG: Automated Framework for optimization of Retrieval Augmented Generation Pipelin (https://arxiv.org/abs/2410.20878)
Comments:
          20 pages

- **What's New**: 이 연구에서는 AutoRAG 프레임워크를 제안합니다. 이 프레임워크는 주어진 데이터 세트에 적합한 RAG 모듈을 자동으로 식별하여 최적의 조합을 탐색하고 근사화합니다. 이는 RAG 기술의 효율성과 효과성을 높이는 데 기여합니다.

- **Technical Details**: AutoRAG는 다양한 RAG 설정을 평가하여 RAG 기술의 선택을 최적화하는 자동화된 프레임워크입니다. 연구에서는 쿼리 확장, 문서 검색, 구절 보강, 구절 재랭킹 및 프롬프트 생성과 같은 여러 RAG 기술을 평가하였으며, BM25, HyDE 등 다양한 정보 검색 모듈을 사용했습니다.

- **Performance Highlights**: AutoRAG를 통해 데이터 세트를 최적화한 결과는 성과가 입증되었으며, 모든 실험 결과와 데이터는 GitHub 리포지토리를 통해 공개되고 접근할 수 있습니다.



### Reward Modeling with Weak Supervision for Language Models (https://arxiv.org/abs/2410.20869)
- **What's New**: 본 연구에서는 RLHF(강화학습 기반 인간 피드백) 데이터셋을 확장하고 보상 모델 성능을 강화하기 위해 약한 감독(weak supervision) 접근 방식을 도입하였습니다. 기존의 고비용 수작업 라벨링 대신 불완전한 데이터를 활용하여 데이터 시트를 보강하는 방법을 제시합니다.

- **Technical Details**: 약한 감독은 모델 출력에 대해 불확실한 라벨링을 허용하는 기계 학습 기법입니다. 본 연구에서는 헌팅(heursitic) 분석을 기초로 간단한 라벨링 함수를 작성하고, 이를 통해 원시 라벨이 없는 데이터를 약하게 주석 처리하여 보상 모델을 훈련시키는 방식을 사용하였습니다. 다양한 데이터셋을 기반으로 실험을 진행하였으며, Flesch Reading Ease와 Type-Token Ratio와 같은 지표들을 통해 문서의 가독성 및 다양성을 분석하였습니다.

- **Performance Highlights**: 약한 감독 접근 방식을 사용함으로써, 작은 데이터셋에 대해서는 보상 모델 성능이 크게 향상되었음을 보여주었습니다. 그러나 원래 라벨이 있는 대규모 데이터셋에서는 효과가 감소하는 경향이 있음을 발견했습니다. 전반적으로, LLM(대규모 언어 모델)을 사용하여 생성한 후 약하게 라벨링된 응답이 선호 데이터 확장을 위한 유망한 방법임을 증명했습니다.



### A Simple Yet Effective Corpus Construction Framework for Indonesian Grammatical Error Correction (https://arxiv.org/abs/2410.20838)
- **What's New**: 이번 논문에서는 낮은 자원을 가진 언어, 특히 인도네시아어에서 문법 오류 수정(Grammatical Error Correction, GEC)을 위한 평가 코퍼스를 구축하는 프레임워크를 제안합니다. 기존 연구들이 주로 영어 및 중국어와 같은 보편적인 언어에 집중하고 있는 반면, 이 논문은 자원이 제한된 인도네시아어에 대한 연구를 확장합니다.

- **Technical Details**: 저자들은 GEC 모델을 훈련하기 위해 결함이 있는 합성 데이터(faulty synthetic data)를 사용하고, 이후 이를 통해 실제 데이터에서 오류를 수정하는 방식으로 인도네시아어 GEC 평가 코퍼스를 만듭니다. 이 과정에서 자동 수정된 문장과 여전히 부정확한 문장을 구별하는 이진 분류 작업을 수행함으로써 대규모 수작업 주석 필요성을 줄입니다. 또한 GPT-3.5-Turbo와 GPT-4 같은 대규모 언어 모델(large language models, LLMs)을 이용하여 주석 작업의 효율성을 높이는 가능성을 탐구합니다.

- **Performance Highlights**: 연구 결과는 낮은 자원 언어 환경에서 LLM의 성능 향상 가능성을 나타냅니다. 제안된 프레임워크는 다른 언어에도 적용 가능하여 다양한 언어로 GEC 평가 코퍼스를 생성할 수 있는 기틀을 제공합니다.



### LLMs are Biased Evaluators But Not Biased for Retrieval Augmented Generation (https://arxiv.org/abs/2410.20833)
Comments:
          15 pages, 14 tables, 5 figures

- **What's New**: 이번 연구는 Retrieval-Augmented Generation (RAG) 방식에서 LLM(대형 언어 모델)의 자기 편향(bias)에 대한 기존의 이해를 확장합니다. 구체적으로 LLM이 자기 생성한 내용에 대한 선호를 보이지 않는 것을 보여주며, 사실적인 정확성이 LLM의 출력에 미치는 중요한 영향을 입증합니다.

- **Technical Details**: 연구에서는 RAG 프레임워크의 두 가지 중요한 단계를 시뮬레이션하여 LLM의 자기 편향 영향을 평가했습니다. 첫 번째 단계는 인간 작성과 모델 생성 텍스트의 적합성을 평가하는 과정이고, 두 번째 단계는 쌍별 독해(comprehension) 테스트를 수행하여 생성 과정을 시뮬레이션합니다. 이는 다양한 통계적 방법을 통해 평가됩니다.

- **Performance Highlights**: 실험 결과, RAG 프레임워크에서는 LLM이 자기 생성 콘텐츠를 선호하지 않으며, 오히려 사실적 정확성이 성과에 크게 영향을 미친다는 것을 확인했습니다. GPT 모델은 LLaMA보다 자기 편향이 작고, 정확한 답변을 선호하는 경향을 보였습니다. 이 결과는 LLM의 성능 향상 및 편향 감소를 위한 중요한 시사점을 제공합니다.



### The Zeno's Paradox of `Low-Resource' Languages (https://arxiv.org/abs/2410.20817)
Comments:
          Accepted at EMNLP 2024

- **What's New**: 이 연구는 자연어 처리(NLP) 분야에서 '저자원 언어(low-resource language)'라는 용어의 정의가 얼마나 다양하게 사용되는지를 탐구하였으며, 저자원 언어의 특징을 이해하기 위해 150개의 논문을 질적으로 분석했습니다.

- **Technical Details**: 본 연구에서는 '저자원 언어'라는 용어를 정의하는 데 있어 네 가지 상호 작용하는 축을 제시합니다: 1) 사회정치적 측면(예: 재정 및 역사적 제약), 2) 자원(인적 및 디지털), 3) 언어적 지식 및 데이터와 같은 인프라, 4) 해당 언어 커뮤니티의 주체성.

- **Performance Highlights**: 저자원 언어에 관한 연구는 무시되기 쉬운 언어 기술의 발전 속도를 추적하기 어려우며, 이를 통해 커뮤니티의 필요에 적합한 기술이 개발될 수 있는 기반을 제공합니다.



### NewTerm: Benchmarking Real-Time New Terms for Large Language Models with Annual Updates (https://arxiv.org/abs/2410.20814)
Comments:
          Accepted to NeurIPS 2024 Datasets and Benchmarks Track

- **What's New**: 대형 언어 모델(LLMs)의 새로운 용어나 사실에 대한 실시간 평가를 위한 자동화된 벤치마크인 NewTerm을 소개합니다. 이 벤치마크는 매년 업데이트되며, LLM의 성능 저하 원인을 분석하고 더 효과적인 접근법 개발을 위한 기반을 마련합니다.

- **Technical Details**: NewTerm 벤치마크는 온라인 사전에서 새로운 용어를 수집하고, 이 용어들에 대한 LLM의 이해도를 평가하기 위해 자동적으로 구성되는 세 가지 오픈 도메인 태스크를 설계하였습니다. 이 과정에서 LLM이 새로운 용어를 이해하지 못할 경우, 20% 이상의 성능 저하가 나타나는 것이 실증적으로 입증되었습니다.

- **Performance Highlights**: 2022년과 2023년의 NewTerm 벤치마크를 통해, LLM들은 새로운 용어에 대해 20% 이상의 정확도 감소를 보였으며, LLM의 지식 컷오프(up to date)이 업데이트되어도 모든 새로운 용어를 포괄하지 못함을 확인했습니다. 이러한 성과는 LLM의 새로운 용어에 대한 성능 저하의 경향을 추적하는 데에 유용합니다.



### Rephrasing natural text data with different languages and quality levels for Large Language Model pre-training (https://arxiv.org/abs/2410.20796)
Comments:
          21 pages, 4 figures, 12 tables

- **What's New**: 최근 자연어 텍스트 데이터를 재구성하여 LLM (Large Language Models) 사전학습에 활용한 연구가 주목받고 있습니다. 본 논문은 기존 연구를 바탕으로 C4 데이터셋에서의 결과를 재현하고, CulturaX의 영어, 독일어, 이탈리아어, 스페인어 오스카(Oscar) 하위 집합을 위해 최적화된 재구성 파이프라인을 확장했습니다.

- **Technical Details**: 최적화된 재구성 파이프라인을 통해 단일 및 다국어 설정에서 표준 평가 (evaluation) 벤치마크에서 성능이 향상됨을 보여주었습니다. 우리는 기본 데이터셋 및 LLM 선택, 모델 크기와 성능 간의 관계를 조사하며 다양한 품질 수준의 데이터에서 성과 차이를 분석했습니다.

- **Performance Highlights**: 재구성된 저품질 다국어 데이터의 사전학습이 LLM의 성능을 향상시키는 가능성이 큽니다. 모델 계층 간의 성능 차이가 모델 크기 간의 차이보다 더 크게 나타났으며, 또한 합성 데이터로 사전학습을 진행했을 때 감독된 미세 조정 (fine-tuning)에 미치는 영향이 벤치마크에 따라 다르게 나타나는 것을 발견했습니다.



### Deep Learning for Medical Text Processing: BERT Model Fine-Tuning and Comparative Study (https://arxiv.org/abs/2410.20792)
- **What's New**: 본 논문은 의료 정보의 폭발적인 증가에 대응하기 위해 BERT 모델을 기반으로 한 의료 문헌 요약 생성 방법을 제안합니다.

- **Technical Details**: BERT 모델을 미세 조정(fine-tuning)하고 최적화하여 키 정보를 빠르게 추출하고 일관되며 정확한 요약을 생성할 수 있는 효율적인 요약 생성 시스템을 개발했습니다. 실험에서는 Seq-Seq, Attention, Transformer 및 BERT를 포함한 다양한 모델을 비교했습니다.

- **Performance Highlights**: 개선된 BERT 모델이 Rouge 및 Recall 메트릭에서 상당한 이점을 제공함을 보여주었으며, 지식 증류(knowledge distillation) 기술이 모델 성능을 더욱 향상시킬 잠재력을 강조했습니다. 이 시스템은 실용적인 응용 프로그램에서 강력한 다재다능성과 효율성을 입증하며, 의료 문헌의 신속한 선별 및 분석을 위한 신뢰할 수 있는 도구를 제공합니다.



### SCULPT: Systematic Tuning of Long Prompts (https://arxiv.org/abs/2410.20788)
- **What's New**: 이 논문은 SCULPT(Systematic Tuning of Long Prompts)라는 새로운 프레임워크를 소개합니다. SCULPT는 긴 프롬프트를 체계적으로 수정하여 성능을 향상시키는 방법입니다.

- **Technical Details**: SCULPT는 프롬프트를 계층적으로 구성하고 반복적인 actor-critic 메커니즘을 적용하여 최적화합니다. 또한, Preliminary Assessment와 Error Assessment라는 두 가지 상호 보완적 피드백 메커니즘을 사용하여 프롬프트의 구조를 실행 전후에 평가합니다.

- **Performance Highlights**: 실험 결과, SCULPT는 기존 방법들과 비교해 정확도 향상 및 강건성(robustness) 증가를 보여주었으며, 잘못된 프롬프트를 처리하는 데에서도 우수한 성과를 낼 수 있음을 입증했습니다.



### Graph-based Uncertainty Metrics for Long-form Language Model Outputs (https://arxiv.org/abs/2410.20783)
Comments:
          Accepted as a Spotlight paper at NeurIPS 2024

- **What's New**: 이번 연구에서는 Large Language Models (LLMs)의 텍스트 생성 능력을 개선하기 위해 새로운 개념인 Graph Uncertainty를 도입합니다. 이 방법은 LLM의 생성물과 그 안의 주장 간의 관계를 이분 그래프로 표현하며, 주장 수준의 불확실성을 추정할 수 있게 해줍니다. 이로써 현재의 불확실성 추정 기법보다 더 정교한 접근법을 제공하며, 사실성을 높이는 데 기여합니다.

- **Technical Details**: Graph Uncertainty는 LLM의 생성 결과와 그 안에 포함된 주장 간의 의미적 관계를 이분 그래프로 표현하고, 다양한 그래프 중심성 지표(family of graph centrality metrics)를 사용하여 주장 수준의 불확실성을 추정합니다. 기존의 self-consistency 기반의 불확실성 측정은 degree centrality를 사용하고, 우리의 연구를 통해 closeness centrality가 더 정확한 불확실성 추정치를 제공함을 증명하였습니다. 또한, 불확실성에 민감한 디코딩 기법을 통해 신뢰할 수 있는 주장을 보존하며 LLM 생성물의 사실성을 개선합니다.

- **Performance Highlights**: 그래프 기반 불확실성 지표는 다양한 장황한 생성 설정에서 AUPRC에 평균 6.8%의 향상을 보였고, 우리의 시스템은 기존 디코딩 기법에 비해 2-4%의 사실성 향상을 지속적으로 이루어내며 생성된 응답의 정보성을 크게 개선했습니다.



### Decoding Reading Goals from Eye Movements (https://arxiv.org/abs/2410.20779)
- **What's New**: 이 연구는 독자가 읽는 동안의 시선 이동 패턴으로부터 독서 목표를 디코드할 수 있는 가능성을 처음 탐색합니다. 일반적인 독서와 정보 탐색의 두 가지 독서 목표에 초점을 맞추어 대규모의 eye-tracking 데이터를 사용하여 다양한 최신 모델을 적용하였습니다.

- **Technical Details**: 논문에서는 eye movements 데이터를 처리하는 여러 가지 최첨단 아키텍처를 사용하여 정보를 인코딩하고, 이들을 조합하여 독서 목표를 예측하는 새로운 디코딩 작업을 수행하였습니다. 또한 모델의 성능을 평가하기 위해 세 가지 일반화 수준(새로운 텍스트, 새로운 참여자 및 둘의 조합)에서 체계적인 평가를 진행하였습니다.

- **Performance Highlights**: 모델 평가 결과, eye movements는 독서 목표를 디코드하는 데 매우 유용한 신호를 포함하고 있음을 보여주었으며, 최고의 단일 모델보다 더 나은 성능을 보이는 앙상블 모델을 제시했습니다. 실험적 분석을 통해 과제의 어려움에 기여하는 텍스트 항목 및 참여자의 시선 이동 관련 주요 특성을 밝혔습니다.



### KD-LoRA: A Hybrid Approach to Efficient Fine-Tuning with LoRA and Knowledge Distillation (https://arxiv.org/abs/2410.20777)
Comments:
          Accepted at 4th NeurIPS Efficient Natural Language and Speech Processing Workshop (ENLSP-IV 2024)

- **What's New**: 본 연구에서는 KD-LoRA라는 새로운 미세 조정 기법을 제안하며, 이는 LoRA(저계수 적응)와 KD(지식 증류)를 결합하여 성능 손실을 최소화하면서 자원 요구 사항을 크게 줄이는 것을 목표로 합니다.

- **Technical Details**: KD-LoRA는 다음의 세 가지 주요 단계로 구성된 방법론입니다: (1) 교사 모델 선택 및 미세 조정, (2) LoRA 모듈로 초기화된 더 작은 학생 모델 설정, (3) 교사 모델에서 학생 모델로 지식을 전이하는 증류 수행.

- **Performance Highlights**: KD-LoRA는 GLUE 기준에서 LoRA의 성능을 98% 유지하면서도 GPU 메모리 사용량을 30% 줄이고, FFT보다 약 40% 더 작은 모델 크기를 달성했습니다. 또한, KD-LoRA는 추론 시간을 약 30% 단축시켰습니다.



### Are LLM-Judges Robust to Expressions of Uncertainty? Investigating the effect of Epistemic Markers on LLM-based Evaluation (https://arxiv.org/abs/2410.20774)
Comments:
          21 pages, 6 figures, 15 tables

- **What's New**: 최근 대규모 언어 모델(LLMs)을 훈련시켜 지식 지표(epistemic markers)를 포함하는 출력을 생성하기 위한 노력이 증가하고 있습니다. 그러나 지식 지표가 포함된 출력에 대한 평가가 소홀히 되어 왔습니다. 이 연구는 LLM-저자들이 지식 지표에 어떻게 반응하는지를 평가하기 위한 EMBER라는 벤치마크를 제시합니다.

- **Technical Details**: EMBER는 LLM-심사자(LLM-judges)의 지식 지표에 대한 강건성을 평가하기 위해 개발된 기준입니다. 이 연구에서는 EMBER를 사용하여 GPT-4o를 포함한 여러 LLM-심사자의 성능을 평가하였으며, 결과적으로 모든 심사자가 지식 지표의 존재에 대해 고유의 강건성을 보이지 않는다는 사실을 발견했습니다.

- **Performance Highlights**: 지식 지표가 포함된 출력에 대해 LLM-심사자들이 부정적인 편향을 보이는 것으로 나타났습니다. 특히 불확실성을 나타내는 지표에 대해 더욱 강한 편향이 관찰되었습니다. 이는 LLM-심사자들이 지식 지표의 존재에 영향을 받아 콘텐츠의 올바름만에 집중하지 않는다는 것을 의미합니다.



### MrT5: Dynamic Token Merging for Efficient Byte-level Language Models (https://arxiv.org/abs/2410.20771)
- **What's New**: MrT5 (MergeT5)는 ByT5의 구조적 비효율성을 해결하는 변형 모델입니다. 입력 시퀀스의 길이를 동적으로 단축시키기 위해 토큰 삭제 메커니즘을 통합했습니다.

- **Technical Details**: MrT5는 인코더의 입력을 단축시키기 위해 고정된 인코더 층에서 토큰 삭제 게이트를 사용합니다. 학습 과정에서 삭제 정규화 기법을 활용하여 삭제할 토큰과 유지할 토큰을 결합합니다. 이 방법을 통해 MrT5는 주어진 시퀀스의 정보를 더 압축하여 담을 수 있습니다.

- **Performance Highlights**: MrT5는 ByT5에 비해 추론 시간에서显著 개선을 보이며, 시퀀스 길이를 최대 80%까지 줄이는 동시에 XNLI 및 문자 수준의 작업에서 유사한 정확도를 달성합니다.



### A Static and Dynamic Attention Framework for Multi Turn Dialogue Generation (https://arxiv.org/abs/2410.20766)
Comments:
          published as a journal paper at ACM Transactions on Information Systems 2023. 30 pages, 6 figures

- **What's New**: 본 논문에서는 정적(static) 및 동적(dynamic) 주의(attention) 메커니즘을 기반으로 하는 다중 턴 대화 생성 모델을 제안합니다. 이 모델은 대화의 맥락을 보다 효과적으로 모델링하여 응답의 일관성과 다양성을 크게 향상시킵니다.

- **Technical Details**: 이 연구에서는 RNN(순환신경망) 기반의 적층 구조를 사용하되, 데이터의 맥락을 파악하는 데 있어 '소멸되는 기울기(vanishing gradient)' 문제를 해결하기 위해 정적 및 동적 주의 메커니즘을 도입합니다. 정적 주의는 전체 대화 정보에 초점을 맞추고, 동적 주의는 개별 발언의 세부 정보를 고려하여 보다 정교한 대화 흐름을 생성합니다.

- **Performance Highlights**: 실험 결과, Ubuntu 및 Opensubtitles 데이터셋에서 제안한 모델이 기존 모델들보다 자동 평가 및 인간 평가 지표에서 우수한 성능을 보이는 것으로 나타났습니다. 정적 및 동적 주의의 조합은 다양한 실험 설정에서도 뛰어난 결과를 보여주었습니다.



### Evaluating LLMs for Targeted Concept Simplification forDomain-Specific Texts (https://arxiv.org/abs/2410.20763)
Comments:
          to appear in proceedings of EMNLP 2024

- **What's New**: 본 논문에서는 '타겟 개념 단순화'라는 새로운 과제를 제안하고, 이 작업을 통해 독자들이 어려운 개념을 이해하는 데 도움을 주는 방안을 모색합니다.

- **Technical Details**: WikiDomains라는 새로운 데이터셋이 소개되며, 이 데이터셋은 13개 학문 분야에서 22,000개의 정의와 각각의 정의에 포함된 어려운 개념을 쌍으로 포함합니다. 연구에서는 오픈소스 및 상용 LLM(대규모 언어 모델)의 성능을 평가하며, 세 가지 접근 방식 (어려운 개념의 사전 정의 추가, LLM을 사용한 개념 단순화, LLM을 통해 개념을 문맥에 맞춰 설명)을 비교합니다.

- **Performance Highlights**: 인간 평가 결과, 독자들은 개념의 문맥적 설명이 단순한 용어의 단순화보다 더 선호되었습니다. 그러나 LLM은 여전히 이해도 측면에서 개선이 필요하며, 자동화된 지표와 인간 평가 결과 간의 상관관계는 낮은 수치를 보였습니다(약 0.2), 이는 보다 나은 측정 지표의 필요성을 시사합니다.



### Plan$\times$RAG: Planning-guided Retrieval Augmented Generation (https://arxiv.org/abs/2410.20753)
Comments:
          22 pages, preprint

- **What's New**: Plan$	imes$RAG라는 새로운 프레임워크가 도입되었습니다. 이 프레임워크는 기존의 RAG(검색 기반 생성) 프레임워크의 '검색-후-추론'(retrieve-then-reason) 패러다임을 '계획-후-검색'(plan-then-retrieve)으로 확장합니다.

- **Technical Details**: Plan$	imes$RAG는 질의를 상호 관련된 원자 서브 질의(atomic sub-queries)로 분해하여 유향 비순환 그래프(Directed Acyclic Graph, DAG) 형태로 구성합니다. 이러한 구조를 통해 검색과 생성 과정을 병렬로 처리할 수 있어 효율성이 크게 증가합니다. 또한, 고정된 언어 모델(frozen LMs)을 전문가처럼 활용하여 고품질의 응답을 생성합니다.

- **Performance Highlights**: Plan$	imes$RAG는 기존 RAG 솔루션에 비해 환각(hallucinations)을 줄이고, 출처(attribution)를 강화하는 데 있어 상당한 개선을 보입니다. 구조화된 서브 질의 분해 덕분에 LM 기반 시스템의 신뢰성을 높이는 데 기여하고 있습니다.



### ElectionSim: Massive Population Election Simulation Powered by Large Language Model Driven Agents (https://arxiv.org/abs/2410.20746)
Comments:
          41 pages, 13 figures

- **What's New**: 이 논문에서는 ElectionSim이라는 대규모 인구 선거 시뮬레이션 프레임워크를 소개합니다. 이는 대형 언어 모델(LLM)을 기반으로 하여 정확한 유권자 시뮬레이션과 맞춤형 분포를 지원하며, 시뮬레이션된 유권자와의 대화형 플랫폼을 제공합니다.

- **Technical Details**: ElectionSim은 소셜 미디어 플랫폼에서 수집된 데이터(171,210,066 트윗)를 이용하여 백만 이상의 유권자 풀을 구성하고, 신뢰할 수 있는 개별 시뮬레이션을 가능하게 합니다. 이 프레임워크는 인구통계적 샘플링 전략을 사용하여 현실 세계의 유권자 분포와의 정합성을 맞추고, Poll-based Presidential Election(PPE) 벤치마크를 통해 시스템적 평가를 실시합니다.

- **Performance Highlights**: 실험 결과, ElectionSim 프레임워크를 이용한 유권자 시뮬레이션에서 Micro-F1 점수 0.812를 달성했으며, 2020년 대통령 선거 결과를 51개 주 중 46개 주에서 정확히 예측했습니다. 특히, 15개의 경합 주에서도 12개 주에서 실제와 일치하는 예측 결과를 보였습니다.



### Gender Bias in LLM-generated Interview Responses (https://arxiv.org/abs/2410.20739)
- **What's New**: 이 연구는 LLM(대형 언어 모델)이 생성한 면접 응답에서 성별 편향을 평가하며, GPT-3.5, GPT-4, Claude의 세 가지 모델을 분석합니다. 연구는 다양한 질문 유형과 직군에 대한 면접 응답의 질을 다각도로 감사하여 성별 고정관념과의 정렬을 강조합니다.

- **Technical Details**: 연구는 LIWC(언어적 탐색 및 단어 수) 도구를 사용하여 LLM들이 생성한 면접 응답의 언어적 및 심리적 특성을 분석했습니다. 실험에 사용된 LLM 모델은 기본 설정이 유지 되었으며, 70개의 인기 남녀 이름과 60개의 직업이 포함되었습니다. Mann-Whitney U test를 통해 남성과 여성에 대한 평균 LIWC 점수를 비교했습니다.

- **Performance Highlights**: LLM이 생성한 면접 응답에서 성별에 따른 언어적 및 심리적 특성이 뚜렷하게 구분되며, 남성 지원자에 대해 비해 여성이 표현하는 방식에 유의미한 차이가 나타났습니다. 성별 편향이 모델과 질문 유형에 걸쳐 일관되게 나타났고, 남성에게 편향된 응답이 더욱 강렬하게 나타나는 경향이 확인되었습니다.



### SEG:Seeds-Enhanced Iterative Refinement Graph Neural Network for Entity Alignmen (https://arxiv.org/abs/2410.20733)
Comments:
          7, 2 figures

- **What's New**: 본 논문에서는 엔터티 정합성(entity alignment)의 중요성을 강조하며, 멀티-소스 데이터를 통합한 소프트 라벨 전파(soft label propagation) 프레임워크를 제안합니다. 이 프레임워크는 확장성과 효율성을 향상시키고, 비동형(非同形, non-isomorphic) 이웃 구조를 고려하여 적절한 엔터티 정합성을 제공합니다.

- **Technical Details**: 제안된 방법은 엔터티와 관계를 기반으로 한 이중 각도 모델링을 통해 최적의 관계 쌍을 융합하여 소프트 라벨을 생성합니다. 이 소프트 라벨은 이웃 특징과 의미 관계 데이터를 풍부하게 포함해 요약 정보 손실 문제를 해결합니다. 양방향 가중 합성 손실 함수(bidirectional weighted joint loss function)를 도입하여 긍정 샘플 간의 거리를 줄이고, 비슷한 부정 샘플에 대해 차별화된 처리를 합니다.

- **Performance Highlights**: 실제 데이터셋에서 검증된 결과, 본 방법은 기존의 반지도 학습(semi-supervised learning) 방식보다 성능이 우수하며, 엔터티 정합성의 품질을 크게 개선합니다. 이러한 결과는 제안된 방법이 지식 그래프 간의 엔터티 정합성에 매우 유용하고 실용적임을 시사합니다.



### Simple is Effective: The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieval-Augmented Generation (https://arxiv.org/abs/2410.20724)
- **What's New**: 이번 논문에서는 Knowledge Graph (KG)에 기반한 Retrieval-Augmented Generation (RAG) 프레임워크의 한계를 극복하기 위해 SubgraphRAG를 제안합니다. SubgraphRAG는 서브그래프를 검색하고 LLM(대규모 언어 모델)을 활용하여 추론 및 답변 예측을 수행하는 구조입니다.

- **Technical Details**: SubgraphRAG는 가벼운 multilayer perceptron (MLP)와 병렬 triple-scoring 메커니즘을 통합하여 서브그래프 검색을 효율적으로 수행합니다. 이 방법은 질의의 주제 엔티티에서 구조적 거리를 인코딩하여 검색 효과성을 향상시킵니다. 서브그래프의 크기를 조정 가능하여 질의의 필요와 LLM의 능력에 따라 유연하게 적용됩니다.

- **Performance Highlights**: SubgraphRAG는 WebQSP와 CWQ 같은 KGQA 벤치마크에서 평가되었으며, 다수의 실험 결과, 소형 모델인 Llama3.1-8B-Instruct가 경쟁력 있는 성능을 보이는 동시에 GPT-4o는 이전 방법보다 더 나은 state-of-the-art (SOTA) 결과를 기록했습니다. SubgraphRAG는 hallucinations(허구) 감소와 응답의 기초 강화(improving response grounding)에서도 뛰어난 성능을 입증했습니다.



### Relation-based Counterfactual Data Augmentation and Contrastive Learning for Robustifying Natural Language Inference Models (https://arxiv.org/abs/2410.20710)
Comments:
          accepted at INTERSPEECH 2023

- **What's New**: 이 논문에서는 자연어 추론(NLI) 작업에서 모델의 강인성을 향상시키기 위해 데이터 증강(data augmentation) 기법을 활용한 새로운 방법을 제안합니다. 특히, 토큰 기반(token-based)과 문장 기반(sentence-based) 기법을 이용하여 반사실적(counterfactual) 문장 쌍을 생성하고 대조 학습(contrastive learning)을 적용하여 서로 다른 클래스 간의 차이를 학습합니다.

- **Technical Details**: 제안된 방법에서는 각 문장 쌍에 대해 entailment(함의), neutral(중립), contradiction(모순) 클래스의 문장 쌍을 생성합니다. 토큰 레벨 및 문장 레벨의 데이터 증강 접근법을 적용하여 사실적(factual) 및 반사실적 데이터를 생성하고, 대조 학습을 통해 같은 레이블을 가진 쌍은 가깝게, 다른 클래스의 쌍은 멀리 두어 임베딩 공간(embedding space)에서 학습합니다. 또한, T5 모델과 같은 사전 훈련된 언어 모델을 사용하여 문장 생성에 활용합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 반사실적 NLI 데이터셋과 일반 NLI 데이터셋에서 기존의 견고한 텍스트 분류 방법들과 비교하여 더 높은 정확도를 달성함을 보여줍니다.



### DisasterQA: A Benchmark for Assessing the performance of LLMs in Disaster Respons (https://arxiv.org/abs/2410.20707)
Comments:
          7 pages, 6 tables

- **What's New**: 이 논문에서는 재난 대응 지식에 대한 LLM의 성능을 평가하기 위한 벤치마크인 DisasterQA를 소개합니다. 이 벤치마크는 여섯 가지 온라인 출처에서 생성된 다양한 재난 대응 주제를 다루고 있습니다.

- **Technical Details**: DisasterQA 벤치마크에서는 707개의 다중 선택 질문(MCQ)을 사용하여 5개의 대형 언어 모델(LLMs)인 GPT-3.5 Turbo, GPT-4 Turbo, GPT-4o, Llama 3.1, Gemini를 테스트했습니다. 각 모델은 4가지의 다양한 프롬프트 방식으로 평가되어 정확성과 신뢰성 수준(Logprobs)을 측정하였습니다.

- **Performance Highlights**: 실험 결과, LLM들은 재난 대응 지식에서 개선이 필요하다는 것을 보여주었습니다. 이 연구가 향후 LLM의 재난 대응 능력을 발전시키고, 긴급 관리자가 재난 중에 이 모델들과 협력할 수 있는 가능성을 높일 것으로 기대합니다.



### Combining Domain-Specific Models and LLMs for Automated Disease Phenotyping from Survey Data (https://arxiv.org/abs/2410.20695)
- **What's New**: 이 탐색적 파일럿 연구는 BERN2라는 도메인 특화 모델과 대형 언어 모델(LLMs)을 결합하여 자동화된 질병 표현(disease phenotyping)을 향상시키기 위한 가능성을 조사했습니다.

- **Technical Details**: BERN2는 생물 의학(named entity recognition) 기 모델로, ORIGINS 출생 코호트 조사 데이터에서 질병 정보를 추출하기 위해 사용되었습니다. BERN2의 성능을 수동으로 구분된 기준 데이터셋과 평가한 후, 다양한 LLMs를 Prompt Engineering, Retrieval-Augmented Generation (RAG), Instructional Fine-Tuning (IFT)를 통해 결합하여 모델 출력을 개선했습니다.

- **Performance Highlights**: BERN2는 질병 언급 추출 및 정규화에서 높은 성능을 보였으며, LLMs의 통합, 특히 Few Shot Inference와 RAG 오케스트레이션을 통해 정확도가 더욱 향상되었습니다. 이러한 접근 방식은 구조화된 예시, 논리적 추론 프롬프트 및 상세한 맥락을 포함할 때 효과적인 코호트 프로파일링 및 대규모 이질적 연구 데이터세트 간의 데이터 조화를 위한 유망한 길을 제시합니다.



### SHARE: Shared Memory-Aware Open-Domain Long-Term Dialogue Dataset Constructed from Movie Scrip (https://arxiv.org/abs/2410.20682)
- **What's New**: 이 연구에서는 SHARE라는 새로운 장기 대화 데이터셋을 소개합니다. 이 데이터셋은 영화 대본에서 수집된 것으로, 다양한 관계에서의 공유된 기억을 포함하고 있습니다. 또한 EPISODE라는 장기 대화 프레임워크를 제안하며, 이는 대화 중 공유된 경험을 관리합니다.

- **Technical Details**: SHARE 데이터셋은 각 화자의 페르소나 정보, 개인적인 사건, 그리고 그들의 공유된 기억을 포함합니다. 대화 데이터를 수집하기 위해 영화 대본 파서를 사용하며, 대화에서 암묵적으로 드러나는 정보를 LLM을 통해 요약하고 추출하여 구성합니다. EPISODE 프레임워크는 이전 대화 세션에서의 정보 요약 및 관리를 포함합니다.

- **Performance Highlights**: 실험 결과, 공유된 기억을 가진 두 개인의 장기 대화는 일관성과 흥미로움을 높이며 관계의 반영적 요소가 향상됨을 보여줍니다. SHARE 데이터셋의 61.57%의 에피소드에서 공동의 기억이 등장하여 장기 대화의 중요성을 강조합니다.



### Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA (https://arxiv.org/abs/2410.20672)
Comments:
          48 pages, 17 figures, 17 tables

- **What's New**: 이번 연구에서는 Transformer 모델에서 파라미터 공유 방식인 'layer tying'을 재조명하고, 기존 LLM을 더 작은 'Recursive Transformers'로 변환하기 위한 새로운 방법론을 제안합니다. 이 방법은 성능 손실을 최소화하면서 층 간에 파라미터를 공유합니다.

- **Technical Details**: Recursive Transformers는 미리 훈련된 Transformer에서 효율적으로 초기화되며, 고유한 층 블록을 반복적으로 사용합니다. Relaxed Recursive Transformers는 depth-wise low-rank adaptation (LoRA) 모듈을 도입하여 층 간의 제약을 완화하고, 모델의 compactness를 유지합니다. 새로운 추론 패러다임인 Continuous Depth-wise Batching을 통해 추가적인 처리량 향상 가능성을 보여줍니다.

- **Performance Highlights**: Recursive Transformer는 비슷한 크기의 vanilla pretrained 모델 및 지식 증류(baseline)와 비교해 우수한 성능을 나타내며, 특히 recursive Gemma 1B 모델은 non-recursive Gemma 1B 모델 대비 13.5%의 정확도 향상을 보여줍니다. 이 모델은 같은 아키텍처의 vanilla Transformer와 비교하여 최대 2-3배의 처리량 향상을 실현할 수 있습니다.



### Visualizing attention zones in machine reading comprehension models (https://arxiv.org/abs/2410.20652)
Comments:
          17 pages, published in STAR Protocols

- **What's New**: 이 논문에서는 pretrained language model을 활용한 기계 독해(comprehension) 모델의 attention mechanism에 대한 새로운 접근 방식을 소개합니다.

- **Technical Details**: 제안된 파이프라인(pipeline)은 다양한 레이어(layers)에서 각 attention zone의 효과를 시각화하여 모델의 설명 가능성(explainability)을 나타낼 수 있도록 합니다. 연구자들은 제공된 프로토콜(procotol)과 코드(code)를 통해 MRC 모델 내에서 각 attention zone의 관련성을 쉽게 시각화할 수 있습니다.

- **Performance Highlights**: 이 접근 방식은 다른 pretrained language models에도 일반화하여 적용할 수 있는 가능성을 제시합니다.



### SubjECTive-QA: Measuring Subjectivity in Earnings Call Transcripts' QA Through Six-Dimensional Feature Analysis (https://arxiv.org/abs/2410.20651)
Comments:
          Accepted at NeurIPS 2024

- **What's New**: 본 연구는 사실 확인과 관련된 객체적 부정확성을 넘어, 명확성과 관련성을 결여한 사실적으로 올바른 답변 등 더욱 부드러운 형태의 잘못된 정보에 집중한다.

- **Technical Details**: SubjECTive-QA 데이터셋은 49,446개의 주석과 함께 Earnings Call Transcripts(ECTs)의 QA 세션을 포함한다. QA 쌍은 Assertive, Cautious, Optimistic, Specific, Clear, Relevant의 여섯 가지 주제로 구성되어 분석된다. 이를 통해 저자들은 언어 모델의 성능을 평가하고자 하였다.

- **Performance Highlights**: RoBERTa-base가 관련성과 명확성과 같은 낮은 주관성을 가진 특성에서 Llama-3-70b-Chat과 비슷한 weighted F1 score를 보였고, 주관성이 더 높은 Specific 및 Assertive 특성에서는 평균적으로 10.01% 높은 성과를 보였다. 또한 White House Press Briefings에 대한 최적 모델 테스트 결과 평균 weighted F1 score 65.97%를 기록하며 적용 가능성을 넓혔다.



### Is Moral Self-correction An Innate Capability of Large Language Models? A Mechanistic Analysis to Self-correction (https://arxiv.org/abs/2410.20513)
- **What's New**: 대형 언어 모델(LLMs)의 도덕적 자기 수정(moral self-correction) 능력에 대한 두 가지 기본 질문을 탐구하고 있습니다. 이 연구는 자기 수정의 다양한 요소들이 어떻게 상호작용하여 도덕적 자기 수정을 가능하게 하는지를 분석합니다.

- **Technical Details**: Chain-of-Thought (CoT) 추론, 외부 피드백(external feedback), 그리고 지침 프롬프트(instructional prompts)가 자기 수정의 주요 구성 요소로 당기고 있습니다. 연구에서는 도덕성이 숨겨진 상태(hidden states) 내에서 어떻게 내재화되어 있으며, 이러한 요소들이 어떻게 성능(performance)에 기여하는지를 살펴보았습니다. 또한, 'self-distinguish'라는 검증 프레임워크를 통해 LLMs가 바람직한(output) 것과 바람직하지 않은(output) 것 간의 구분을 할 수 있도록 요구합니다.

- **Performance Highlights**: 자기 수정을 위한 보편적인 최적의 방법은 발견되지 않았으며, 외부 피드백과 CoT가 성능 향상에 기여할 수 있지만, 이들 간에 부정적인 상호작용이 존재함을 보여주었습니다. LLMs는 자신의 응답을 자기 수정할 수 있지만, 바람직한 것과 바람직하지 않은 것의 구분을 신뢰성 있게 할 수 없음을 발견했습니다. 마지막으로, 도덕적 자기 수정은 LLMs가 프리트레이닝(pretraining) 중에 습득하지 않은 능력이라는 결론을 내렸습니다.



### MatViX: Multimodal Information Extraction from Visually Rich Articles (https://arxiv.org/abs/2410.20494)
- **What's New**: 본 논문에서는 재료 과학 분야에서 과학 논문에서 구조화된 정보를 추출하기 위한 새로운 벤치마크인 	extsc{MatViX}를 소개합니다. 이 벤치마크는 324개의 전체 길이 연구 논문과 1,688개의 복잡한 JSON 파일으로 구성되어 있으며, 분야 전문가에 의해 신중하게 선별되었습니다.

- **Technical Details**: MatViX는 고분자 나노복합재(Polymer Nanocomposites, PNC)와 고분자 생분해(Polymer Biodegradation, PBD) 분야에서 구조화된 정보를 추출하는 데 중점을 둡니다. 이 데이터 세트는 화학 조성과 연관된 속성 데이터를 포착하는 구조적 JSON 객체 형태로 제공됩니다. VLM(vision-language models)의 제로샷(zero-shot) 방식으로 긴 문서의 복잡한 계층 구조를 추출합니다.

- **Performance Highlights**: 기존 VLM을 사용한 성능 벤치마크에서 아직 개선의 여지가 많다는 것을 보여주었으며, DePlot과 같은 전문화된 모델과의 결합이 추출 성능을 향상시킬 수 있다는 점을 강조합니다. 이 연구는 과학 문서의 복잡성을 아우르는 구조적 데이터 추출의 중요성을 제시합니다.



### $\textit{Who Speaks Matters}$: Analysing the Influence of the Speaker's Ethnicity on Hate Classification (https://arxiv.org/abs/2410.20490)
Comments:
          9 pages, 3 figures, 3 tables. To appear in NeurIPS SafeGenAI 2024 Workshop

- **What's New**: 본 연구는 Large Language Models (LLMs)에 대한 하이 스테이크(high-stakes) 작업인 혐오 발언 감지(hate speech detection)에서의 강인성(robustness)을 조사합니다. 특히, 화자의 인종을 언급하는 명시적(explicit) 및 암시적(implicit) 마커가 입력에 주입될 때 모델의 반응을 분석합니다.

- **Technical Details**: 이 연구에서는 명시적 마커로 화자의 정체성을 언급하는 문구를 주입하고, 암시적 마커로 방언(dialectal) 특징을 주입했습니다. 4개의 인기 LLM과 5개 인종에 걸쳐 모델 출력을 분석하면서 마커의 존재가 반응에 미치는 영향을 평가했습니다.

- **Performance Highlights**: 암시적 방언 마커가 포함된 입력의 경우 모델 출력이 변할 확률이 명시적 마커보다 더 높았습니다. 또한, 출력 변동의 비율은 인종마다 다르게 나타났으며, 더 큰 모델일수록 강인성이 높은 것으로 나타났습니다. 이는 혐오 발언 감지와 같은 고위험 작업에 LLM을 배치하는 데 있어 신중함의 필요성을 강조합니다.



### FIRP: Faster LLM inference via future intermediate representation prediction (https://arxiv.org/abs/2410.20488)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 생성 효율을 개선하기 위한 새로운 방법인 FIRP(Speculative Decoding) 기법을 소개합니다. FIRP는 각 디코딩 단계에서 단일 토큰이 아닌 여러 개의 토큰을 생성하여 LLM의 병렬 처리 능력을 최대한 활용합니다.

- **Technical Details**: FIRP는 중간 층에서의 은닉 상태(hidden states)를 예측하고 이러한 가상의 은닉 상태(pseudo hidden states)를 통해 미래의 토큰들을 디코딩합니다. FIRP의 핵심 아이디어는 단일 전파(process)에서 미래의 k-그램 토큰을 예측하고, 예측된 상태가 주의(attention) 메커니즘을 통해 상호작용하여 더 많은 의미적 정보를 축적하는 것입니다.

- **Performance Highlights**: 다양한 모델 및 데이터셋에서 FIRP는 기존 방법들에 비해 1.9배에서 3배의 속도 향상을 보였습니다. 이러한 성능 향상은 더 많은 어휘를 한 번의 예측으로 생성할 수 있도록 해주며, 실험 결과는 FIRP의 유효성을 강하게 입증합니다.



### What Factors Affect Multi-Modal In-Context Learning? An In-Depth Exploration (https://arxiv.org/abs/2410.20482)
Comments:
          Accepted at NeurIPS 2024

- **What's New**: 최근 다중 모달 인컨텍스트 학습(Multi-Modal In-Context Learning, MM-ICL)에서의 빠른 발전이 주목받고 있습니다. 본 연구는 MM-ICL의 성능에 영향을 미치는 요인을 조사하여, 효과적인 전략을 최적화하기 위한 기초 자료를 제공하고자 합니다.

- **Technical Details**: 본 연구는 MM-ICL의 세 가지 핵심 단계인 시연 검색(demonstration retrieval), 시연 순서(demonstration ordering), 프롬프트 구성(prompt construction)에 대한 포괄적인 실험을 수행했습니다. 실험은 6개의 시각 대형 언어 모델(vision large language models, VLLMs)과 20가지 전략을 사용하여 실시되었습니다. 연구 결과는 다중 모달 검색기가 필요하며, 시연의 내부 순서(intra-demonstration ordering)가 외부 순서(inter-demonstration ordering)보다 중요함을 강조하고, 소개 지시문(introductory instructions)이 프롬프트에 포함됨으로써 작업 설명(task comprehension)이 향상된다는 것을 보여줍니다.

- **Performance Highlights**: 1. 다중 모달 검색 방법이 단일 모달 방법보다 평균적으로 더 뛰어난 성능을 보임. 2. 내부 시연 순서가 모델 성능에 미치는 영향이 외부 시연 순서보다 크게 나타남. 3. 소개 지시문을 포함할 경우 MM-ICL 성능이 일관되게 향상됨.



### A Derivational ChainBank for Modern Standard Arabic (https://arxiv.org/abs/2410.20463)
- **What's New**: 이 연구는 아랍어 파생 형태론을 모델링하기 위한 새로운 프레임워크인 'Arabic Derivational ChainBank'를 소개합니다. 이 프레임워크는 파생된 단어들 간의 관계를 형성하여 의미와 형태 간의 연결을 구축합니다.

- **Technical Details**: ChainBank는 동적인 지식 기반의 트리 그래프로 아랍어 파생 형태론의 계층적 구조를 나타냅니다. 각 노드는 파생 단어와 이는 morphosemantic 속성(형태, 품사, 기능적 특징 등)을 포함합니다. 파생 관계를 나타내는 연결이 각 자식 노드를 분류합니다. 이 네트워크는 CamelMorph를 통해 아랍어 단어 간의 파생 관계를 연결하는 대규모 네트워크로 구성됩니다.

- **Performance Highlights**: ChainBank는 23,333개의 평가된 파생 관계를 포함하여 효율성을 입증합니다. 결과적으로 파생된 단어의 어근과 연결된 렘마의 체인을 만듭니다.



### TrajAgent: An Agent Framework for Unified Trajectory Modelling (https://arxiv.org/abs/2410.20445)
Comments:
          12 pages; the code will be openly accessible at: this https URL

- **What's New**: 본 논문에서는 다양한 경로 모델링(task) 작업을 통합하기 위해 LLM(대형 언어 모델)을 기반으로 한 간편한 프레임워크인 TrajAgent를 제안합니다. TrajAgent는 통합된 데이터 및 모델 인터페이스를 제공하는 UniEnv를 개발하여 다양한 모델을 실행하고 훈련할 수 있도록 지원합니다.

- **Technical Details**: TrajAgent는 TAgent라는 자동 경로 모델링 작업을 설계하고, AutOpt라는 최적화 모듈을 통해 성능을 향상시킵니다. UniEnv는 데이터 전처리 및 일관된 작업 흐름을 지원하며, 다양한 경로 모델링 작업을 위한 통합된 환경을 제공합니다. 경로 예측, 분류, 생성 작업을 포함한 다양한 작업 유형을 지원합니다.

- **Performance Highlights**: 실험 결과, TrajAgent는 네 가지 실제 데이터 세트에 대해 평균 15.43%의 성능 향상을 기록하여 통합 경로 모델링에서의 효과성을 입증하였습니다.



### MedGo: A Chinese Medical Large Language Mod (https://arxiv.org/abs/2410.20428)
Comments:
          12 pages, 1 figure

- **What's New**: 이 논문은 MedGo라는 세종 (生科学의 두문자) 의학 대용량 언어 모델을 소개합니다. 이 모델은 고품질 비지도 학습 의료 데이터, 지도(Supervised) 데이터 및 개인 맞춤형 선호(Preference Alignment) 데이터를 조합하여 학습되었습니다.

- **Technical Details**: MedGo는 대규모 도메인 특화 의료 데이터셋을 구축하여 의학 지식에 대한 깊은 이해를 가져왔으며, 이 모델은 세 가지 단계(Pre-training, Supervised Fine-Tuning, Preference Alignment)를 통해 최적화되었습니다.

- **Performance Highlights**: MedGo는 CBLUE 벤치마크 평가에서 1위를 달성하였으며, ClinicalQA 데이터셋에서 근본 모델인 Qwen2보다 우수한 성과를 보였습니다. 이로써 MedGo의 자동 의료 질문 응답 및 임상 결정 지원(Clinical Decision Support) 개선 가능성을 강조하고 있습니다.



### Rethinking Data Synthesis: A Teacher Model Training Recipe with Interpretation (https://arxiv.org/abs/2410.20362)
- **What's New**: NOMAD는 데이터 생성을 위해 특별히 훈련된 모델 개발을 목표로 하며, 기존의 방법론의 한계를 극복하는 새로운 접근법을 제안합니다.

- **Technical Details**: NOMAD는 1) no-prompt-masked training과 2) 적절한 데이터 세트 크기 선택이라는 두 가지 핵심 요소를 통해 데이터 생성을 최적화합니다. 이는 일반적인 LLM 훈련과는 다른 방식으로, 모델이 고품질의 프롬프트를 학습하도록 합니다.

- **Performance Highlights**: NOMAD는 TriviaQA에서 >4%, GSM8K에서 >2%의 성능 향상을 보여주었습니다. 소규모 학습 샘플에서도 기초 모델보다 평균 1.5% 더 나은 결과를 얻었습니다.



### Maintaining Informative Coherence: Migrating Hallucinations in Large Language Models via Absorbing Markov Chains (https://arxiv.org/abs/2410.20340)
- **What's New**: 본 논문에서는 Large Language Models (LLMs)의 hallucinations 문제를 해결하기 위해 흡수 Markov Chains(AMC)를 활용한 새로운 decoding 전략을 제안합니다. 이 방법은 LLM이 정보 손실의 정도를 수량화하고 문맥 정보의 중요성을 측정하는 혁신적인 접근법을 제공합니다.

- **Technical Details**: 제안된 방법은 모든 가능한 경로를 고려하여 LLM의 텍스트 생성 과정에서 문맥 정보가 손실되는 정도를 모델링합니다. AMC 이론을 활용하여 각 토큰의 중요성을 수량화하고, 이를 바탕으로 생성되는 각 토큰의 확률 분포를 조정합니다. 이러한 방식으로 정보 흐름을 보다 명확히 파악하고, 이전 문맥으로부터 중요한 정보를 재조명하여 hallucinations을 완화합니다.

- **Performance Highlights**: TruthfulQA, FACTOR, HaluEval와 같은 데이터셋에서 수행된 평가 결과, 제안한 방법이 hallucinations을 감소시키는 데 있어 우수한 성능을 보였습니다. 이는 웹 기반 애플리케이션에서 정확한 정보 흐름을 보장하는 것이 얼마나 중요한지를 강조합니다.



### Get Large Language Models Ready to Speak: A Late-fusion Approach for Speech Generation (https://arxiv.org/abs/2410.20336)
- **What's New**: 본 논문은 TTS(텍스트-투-스피치) 시스템인 TTS-Llama를 소개하며, 이는 세밀하게 조정된 Llama 모델을 통해 가장 앞선 음성 합성 성능을 달성합니다. 또한 MoLE-Llama이라는 텍스트-음성 다중 모드 LLM을 제안하여 PEFT(모델 파라미터 효율적인 미세 조정)를 통해 구축됩니다.

- **Technical Details**: TTS-Llama는 fine-tuned Llama 3-8B-Instruct 모델을 기반으로 하며, 이를 통해 고수준의 의미론적 토큰을 생성하고, 이를 저수준의 음향 특징으로 변환하는 과정으로 음성 합성을 수행합니다. MoLE-Llama는 텍스트와 음성을 전담하는 전문가 모델을 혼합하여 하나의 통합된 다중 모달 LLM으로 구성됩니다.

- **Performance Highlights**: MoLE-Llama는 텍스트 전용 Q&A 및 TTS 작업 모두에서 경쟁력 있는 성능을 보이며, 특히 'catastrophic forgetting(재앙적 망각)' 문제를 완화하는 데 성공했습니다. 또한, 텍스트가 주어질 때 음성을 출력하는 알고리즘을 활용하여 다중 모달 대화 시스템으로서의 가능성을 보여줍니다.



### Improving Speech-based Emotion Recognition with Contextual Utterance Analysis and LLMs (https://arxiv.org/abs/2410.20334)
- **What's New**: 본 논문에서는 음성을 통한 감정 인식(Speech Emotion Recognition, SER)을 위한 새로운 접근 방식을 제안합니다. 특히, 포스트 자동 음성 인식(Post Automatic Speech Recognition, ASR) 텍스트만을 사용하여 대형 언어 모델(Large Language Models, LLMs)의 성능을 분석하고 발전시켰습니다.

- **Technical Details**: 연구진은 다양한 ASR 모델에서 파생된 전사 내용을 조합하고 세분화하여 더 일관된 데이터 품질을 확보하였으며, 각 대화를 더 작은 대화 조각으로 나누어 이들을 문맥으로 활용하여 특정 발화의 감정을 예측하였습니다. 또한, 다양한 문맥 길이(context length)와 프롬프트(prompt) 구조를 실험하여 예측의 정확도를 향상시킬 수 있는 방법을 모색했습니다.

- **Performance Highlights**: 제출한 모델은 기본선(baseline)을 20% 초과하는 비가중 정확도(unweighted accuracy)를 달성하여 챌린지에서 최고의 성과를 기록했습니다. 모든 실험의 코드와 예측 결과는 공개적으로 제공됩니다.



### Deep Learning Based Dense Retrieval: A Comparative Study (https://arxiv.org/abs/2410.20315)
Comments:
          7 pages

- **What's New**: 본 연구는 Dense retrievers (밀집 검색기)가 tokenizer poisoning (토크나이저 오염)에 대한 취약성을 평가합니다. 연구의 결론에 따르면 BERT와 Dense Passage Retrieval (DPR)와 같은 감독 모델은 토크나이저가 손상되었을 때 성능 저하가 심한 반면, ANCE와 같은 비감독 모델은 상대적으로 더 강한 내성을 보입니다.

- **Technical Details**: Dense retrieval 시스템은 입력 텍스트를 수치적 토큰으로 변환하는 데 강력한 토크나이저에 크게 의존합니다. 이 연구에서는 다양한 벤치마크 데이터셋(Quora, FiQA-2018, HotpotQA)을 사용하여 supervised (감독) 및 unsupervised (비감독) 모델의 성능을 평가합니다. 성능 저하를 측정하기 위해 cosine similarity 및 여러 정보 검색 메트릭(정확도, 정밀도, 재현율 등)을 활용합니다.

- **Performance Highlights**: 작은 변형이 검색 정확도에 심각한 영향을 미칠 수 있다는 것을 발견했습니다. 특히, supervised 모델에서 성능 저하가 두드러졌으며, 이는 legal, medical, financial 등의 중요한 응용 분야에서 Dense retrieval 시스템의 신뢰성과 보안성을 보장하기 위한 강력한 방어책 필요성을 강조합니다.



### Learning from Response not Preference: A Stackelberg Approach for LLM Detoxification using Non-parallel Data (https://arxiv.org/abs/2410.20298)
- **What's New**: 본 연구에서는 비정렬(non-parallel) 데이터를 사용하여 대형 언어 모델(LLM)을 텍스트 디톡스화(detoxification) 리라이터로 변환하는 최적화 방법인 Stackelberg Response Optimization(SRO)을 제안합니다. 일반적인 바탕으로, LLM과 독성 스크리너(toxicity screener) 간의 Stackelberg 게임 모델링이 이루어집니다.

- **Technical Details**: LLM의 미세 조정 과정은 LLM(리더)과 이진 스타일 분류기(독성 또는 비독성)인 독성 스크리너(팔로워) 간의 Stackelberg 게임으로 모델링됩니다. SRO에서는 LLM이 독성 텍스트를 기반으로 리라이트를 생성하고, 스크리너가 성공 또는 실패의 결과를 제공하여 LLM이 스타일을 조정하게 됩니다. SRO는 독성 콘텐츠와 그 리라이트 쌍을 사용할 때 DPO(Direct Preference Optimization)에 따라 학습합니다.

- **Performance Highlights**: SRO로 미세 조정된 LLM은 스타일 정확도, 내용 유사성 및 유창성에서 기존의 최첨단 모델과 비슷한 만족스러운 성능을 달성하며, 전체 디톡스화 성능은 다른 방법들을 초월하여 인간 참조와 일치합니다. 실험적으로 SRO의 민감성이 스크리너의 피드백에 큰 영향을 받는다는 추가 증거도 확인되었습니다.



### Fine-Tuning and Evaluating Open-Source Large Language Models for the Army Domain (https://arxiv.org/abs/2410.20297)
- **What's New**: 최근 대규모 언어 모델(LLMs)의 군사 도메인에서의 활용 가능성에 대한 연구가 활발히 이루어지고 있습니다. 본 논문에서는 Army 도메인에 적합한 오픈소스 LLM을 조정하기 위한 노력으로 TRACLM이라는 LLM 계열을 제안합니다.

- **Technical Details**: 본 연구에서 소개하는 TRACLM은 The Research and Analysis Center(TRAC)와 Army Futures Command(AFC)에서 조정된 LLM의 세 가지 세대를 포함하고 있으며, 각 세대의 모델은 훈련 프로세스의 지속적인 개선을 거쳐 Army 작업과 사용 사례에 적용될 때 능력이 향상되었습니다. 또한 MilBench라는 평가 프레임워크를 개발하여 LLM의 Army 도메인 지식을 정량적으로 평가할 수 있도록 하였습니다.

- **Performance Highlights**: TRACLM 모델은 Army 특정 작업에서 성능이 향상된 것으로 나타났습니다. MilBench는 LLM의 평가를 위한 확장 가능하고 효율적인 소프트웨어 프레임워크로, 다양한 DoD(Department of Defense)에서의 활용 가능성을 높입니다.



### Fast Best-of-N Decoding via Speculative Rejection (https://arxiv.org/abs/2410.20290)
Comments:
          NeurIPS 2024

- **What's New**: 대규모 언어 모델(Large Language Models, LLMs)의 안전하고 효과적인 배포를 위해서는 인공지능의 응답이 인간의 선호도에 맞도록 조정하는 '정렬(alignment)' 과정이 필요합니다. Post-training 기법은 LLM의 가중치를 조정하여 정렬을 수행하지만, 이 방법은 복잡성이 큽니다. 본 연구에서는 이러한 복잡한 단계를 피하고 인간의 선호도에 맞춰 응답을 생성하는 '추측적 거부(Speculative Rejection)' 알고리즘을 소개합니다. 이는 Best-of-N 방식과 유사한 성능을 내면서 16배에서 32배 더 효율적입니다.

- **Technical Details**: Speculative Rejection은 주어진 보상 모델(reward model)에 따라 높은 점수를 받을 가능성이 있는 응답을 생성합니다. 초기화 단계에서 LLM이 산출할 수 있는 응답의 품질을 평가할 수 있는 보상 함수를 사용하여, 생성 진행 중에 고품질 응답이 아닐 가능성이 있는 응답 생성을 조기에 차단합니다. 이를 통해 GPU 메모리를 효율적으로 사용할 수 있습니다. 기존 Best-of-N 방식의 기능과 유사한 성능을 유지하면서, N이 1000 이상인 경우에도 단일 GPU에서 동작할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, Speculative Rejection은 Best-of-N 방식보다 훨씬 적은 리소스를 사용하면서도 비슷한 품질의 응답을 생성할 수 있음을 입증하였습니다. AlpacaFarm 데이터세트를 사용한 결과, Best-of-N 방식은 16개에서 32개의 GPU를 필요로 하는 반면, Speculative Rejection은 단일 GPU에서 비슷한 지연(latency)으로 작동할 수 있습니다. 생성 품질 측면에서도 Best-of-N과 비교하여 우수한 성능을 보였습니다.



### Improving Model Evaluation using SMART Filtering of Benchmark Datasets (https://arxiv.org/abs/2410.20245)
Comments:
          20 pages, 5 figures

- **What's New**: 본 연구는 NLP(Natural Language Processing) 모델 평가의 효율성을 높이기 위한 새로운 접근 방식을 제안합니다. 기존의 벤치마크 데이터셋에서 저품질의 예제를 체계적으로 제거하여 고품질의 예제를 선별하는 SMART(Selection Methodology for Accurate, Reduced, and Targeted) 필터링 방법론을 도입합니다.

- **Technical Details**: SMART 필터링은 세 가지 기준에 따라 작동합니다: (i) 쉬운 예제 제거, (ii) 데이터 오염이 의심되는 예제 제거, (iii) 임베딩 공간에서 서로 유사한 예제 제거. 이를 통해 모델의 상대 순위를 보존하면서 데이터셋의 크기를 평균 48% 줄이고, ChatBot Arena에서의 Pearson 상관계수를 높일 수 있음을 보여줍니다.

- **Performance Highlights**: SMART 필터링을 사용하여 데이터셋을 약 68.9% 줄이면서도 모델의 상대적 순위에 미치는 영향을 최소화하는 성과를 보였습니다. 필터링된 데이터셋은 ChatBot Arena 등에서의 인간 평가와 보다 강한 상관관계를 보이며, 이는 더 나은 인간 선호도와 일치한 결과입니다.



### A Survey of Large Language Models for Arabic Language and its Dialects (https://arxiv.org/abs/2410.20238)
- **What's New**: 이번 논문은 아랍어 및 그 방언을 위한 대규모 언어 모델(Large Language Models, LLMs)에 대한 종합적인 개요를 제공합니다.

- **Technical Details**: 다양한 아키텍처(architecture), 즉 인코더 전용(encoder-only), 디코더 전용(decoder-only), 인코더-디코더 모델을 포함하며, 고전 아랍어(Classical Arabic), 현대 표준 아랍어(Modern Standard Arabic), 방언 아랍어(Dialectal Arabic)를 위한 사전 학습(pre-training) 데이타셋(datasets)을 다룹니다. 단일 언어( монолинг구ал), 이중 언어(bilingual), 다중 언어(multilingual) LLMs의 아키텍처와 성능을 감성 분석(sentiment analysis), 개체명 인식(named entity recognition), 질문 응답(question answering)과 같은 다운스트림 작업에서 분석합니다.

- **Performance Highlights**: 논문은 아랍어 LLM의 개방성(openness)을 소스코드(source code) 가용성, 학습 데이터(training data), 모델 가중치(model weights), 문서화(documentation)와 같은 요인을 바탕으로 평가합니다. 더불어 다양한 방언 데이터셋의 필요성과 연구 재현성(reproducibility) 및 투명성(transparency)을 위한 개방성의 중요성을 강조합니다. 또한 향후 연구를 위한 주요 도전(challenges)과 기회(opportunities)를 확인하고, 보다 포괄적이고 대표적인 모델의 필요성을 강조합니다.



### Ambiguity is the last thing you need (https://arxiv.org/abs/2410.20222)
- **What's New**: 본 논문은 계약에서 명확한 법적 언어의 중요성을 강조하며, 애매한 언어로 인해 발생하는 분쟁을 다룹니다. 특히 소비자와의 계약에서의 불균형한 협상력을 고려할 때, 평이한 언어(plain language)의 사용이 필수적이라는 점을 논의합니다.

- **Technical Details**: 계약 언어가 애매해지는 요인은 주로 불명확하고 모호한 표현에서 비롯됩니다. 영어에는 동의어가 많아 언어의 의미 해석에 차이를 초래하는데, 이는 법원 사례의 상당 부분이 애매한 언어에 근거하고 있음을 보여줍니다. 이러한 문제를 해결하기 위한 법적 접근과 언어 사용의 변화가 필요하다고 제안합니다.

- **Performance Highlights**: 저자는 법적 문서에서 사용하는 언어의 명확성을 높여 오해의 소지를 줄이고, 미래의 소송(litigation)을 방지할 수 있는 방안을 모색해야 한다고 강조합니다.



### Generative linguistics contribution to artificial intelligence: Where this contribution lies? (https://arxiv.org/abs/2410.20221)
Comments:
          28 pages, 3 figures

- **What's New**: 이 논문은 Generative linguistics (GL)가 인공지능 (AI)에 미친 기여를 규명하며, 언어학자들과 AI 과학자들 사이의 언어학이 인문학에 속하는지 혹은 과학에 속하는지에 대한 논쟁을 언급합니다.

- **Technical Details**: 이 논문은 Generative linguistics의 원리, 특히 Chomsky School에 기반한 AI의 과학적 정리 및 논리를 다룹니다. 주요 주제에는 구문 (syntax), 의미론 (semantics), 언어 능력 (language faculty), 보편 문법 (Universal Grammar), 인간 언어의 컴퓨터 시스템, 언어 습득 (language acquisition), 인간 두뇌 (human brain), 프로그래밍 언어 (예: Python), 대형 언어 모델 (Large Language Models), 편향 없는 AI 과학자들 (unbiased AI scientists) 등이 포함됩니다.

- **Performance Highlights**: 이 글은 GL의 AI에 대한 기여가 매우 크며 이러한 기여를 부정할 수 없다는 강력한 증거를 제공합니다. 그럼에도 불구하고 GL과 AI 사이에는 언어 입력의 본질과 유형에 대한 의견 차이가 여전히 존재한다고 결론짓습니다.



### Pseudo-Label Enhanced Prototypical Contrastive Learning for Uniformed Intent Discovery (https://arxiv.org/abs/2410.20219)
Comments:
          Accepted by EMNLP 2024 Findings

- **What's New**: PLPCL(프로토타입 대조 학습) 모델은 새로운 의도 발견을 위해 구현되어 기존의 두 가지 과정(IND 사전 학습 및 OOD 클러스터링)를 통합하여 보다 효율적인 지식 이전을 가능하게 합니다.

- **Technical Details**: PLPCL은 반지도 학습(semi-supervised learning)과 대조 학습(contrastive learning)을 통해 라벨이 있는 인도메인(IND) 데이터와 라벨이 없는 아웃오브도메인(OOD) 데이터를 함께 활용합니다. 이 방법은 모형이 인식한 긍정/부정 샘플을 반복적으로 선택하여 대조 학습을 통해 성능을 향상시킵니다.

- **Performance Highlights**: 세 가지 벤치마크 데이터 세트에서의 실험을 통해, PLPCL 모델은 OOD 및 오픈 설정에서 새로운 의도를 효율적으로 발견할 수 있음을 입증하였습니다.



### DAWN-ICL: Strategic Planning of Problem-solving Trajectories for Zero-Shot In-Context Learning (https://arxiv.org/abs/2410.20215)
- **What's New**: 이 논문은 Zero-shot in-context learning (ZS-ICL)을 계획 문제로 재정의하고, Demonstration-aware Monte Carlo Tree Search (MCTS) 접근법을 제안합니다. 이를 통해 다양한 작업에서 효율적으로 문제 해결 경로를 계획하는 방법을 보여줍니다.

- **Technical Details**: DAWN-ICL은 MCTS를 활용하여 ZS-ICL의 문제 해결 경로를 전략적으로 계획합니다. 또한, 새로운 demonstration-aware Q-value 함수를 도입하여 Q 값 추정의 효율성과 효과성을 증가시키며, 선택 및 확장 단계의 성능을 향상시키도록 설계되었습니다.

- **Performance Highlights**: DAWN-ICL은 in-domain과 cross-domain 시나리오에서의 광범위한 실험을 통해, 최상의 ZS-ICL 기준 방법을 지속적으로 초과하며, 인간-주석 레이블을 사용한 ICL보다도 더 뛰어난 성능을 보여줍니다.



### Looking Beyond The Top-1: Transformers Determine Top Tokens In Order (https://arxiv.org/abs/2410.20210)
- **What's New**: 이 연구는 Transformers의 내부 작용을 이해하는 것이 정확하고 효율적인 예측을 달성하는 데 중요하다는 것을 강조하고 있습니다. 특히, 기존에는 고정된 top-1 예측 후 Transformers에서 수행되는 계산을 살펴보았고, 'saturation event'를 top-k 토큰으로 확장했습니다.

- **Technical Details**: 모델은 top-1 예측이 고정된 후 다음 우선 순위 토큰을 결정하는 과정에서 saturation event가 발생합니다. 이 현상은 다양한 변형 아키텍처(decoder-only, encoder-only, full-Transformer)에서 발생하며, 여러 모달리티(언어, 비전, 음성)에서 공통적으로 관찰됩니다. 우리는 이러한 전환이 task transition 메커니즘에 해당한다고 제안하고, 내부 은닉층의 표현을 통해 각 task의 정보를 예측할 수 있음을 보여줍니다.

- **Performance Highlights**: 새로운 token-level early-exit 전략을 제안하여 기존 방법보다 성능과 효율성의 균형을 더 잘 맞출 수 있음을 입증했습니다. 제안한 방법은 이미 잘 알려진 early exit 기법을 개선하여 text generation 수행 시 더욱 정확한 예측을 유도합니다.



### Reasoning or a Semblance of it? A Diagnostic Study of Transitive Reasoning in LLMs (https://arxiv.org/abs/2410.20200)
Comments:
          To appear in EMNLP Main 2024

- **What's New**: 이 논문은 LLaMA 2와 Flan-T5라는 두 가지 대형 언어 모델(LLM)의 전이적(Transitive) 사고 능력을 조사합니다. QASC와 Bamboogle 데이터셋을 활용하여 이러한 모델이 진정한 논리적 추리를 수행하는지, 아니면 암묵적 단서를 활용하는지 평가했습니다.

- **Technical Details**: 실험에서 우리는 (a) 테스트 입력의 섹션 간 단어/구 표현의 중복, (b) 모델의 미세 조정(또는 인퍼런스 시)에 사용된 고유한 지식, (c) 명명된 개체(Named Entities)와 같은 잠재적 단서를 제어했습니다. Flan-T5는 LLaMA 2에 비해 더 적은 변동성을 보여주며, 모델이 관련 데이터셋에서 미세 조정(fine-tuning)을 받아 전이성을 이해할 수 있는 가능성을 제시합니다.

- **Performance Highlights**: 연구 결과 두 모델 모두 중간 정보를 잘 처리했으나, 답변의 키워드를 제거하면 성능이 급격히 떨어지는 경향이 있었습니다. Flan-T5는 명명된 개체들이 포함된 간섭에 더 강하기 때문에 전이적 추리 과정을 수행하는 데 있어 더 효과적이라는 것을 보여주었습니다.



### A Stack-Propagation Framework for Low-Resource Personalized Dialogue Generation (https://arxiv.org/abs/2410.20174)
Comments:
          published as a journal paper at ACM Transactions on Information Systems 2023. 35 pages, 5 figures

- **What's New**: 이번 연구는 제한된 교육 데이터를 활용하여 일관성 있는 응답 생성을 위한 새로운 접근 방식을 제안합니다. 기존의 전통적인 모델과는 달리, 새로운 stack-propagation framework를 통해 트랜스포머(Transformer) 인코더와 디코더를 활용하여 퍼스나(persona) 이해를 규제화(regularization)하는 방법으로 응답을 생성합니다.

- **Technical Details**: 제안된 stack-propagation framework는 하나의 트랜스포머 인코더와 두 개의 트랜스포머 디코더로 구성되어 있습니다. 첫 번째 디코더는 응답 생성을 모델링하고, 두 번째 디코더는 응답 생성과 일관성 이해를 동시에 모델링하여 규제 역할을 수행합니다. 이를 통해, 모델은 보다 적은 개인화된 대화 데이터로부터 학습하면서도 경쟁력 있는 성능을 유지할 수 있습니다.

- **Performance Highlights**: 주어진 다양한 저자원(low-resource) 환경에서, 주관적 및 객관적 평가를 통해 제안된 stack-propagation framework가 응답 품질 및 퍼스나 일관성에서 기존의 강력한 기준 모델들보다 우수한 성능을 보이며, 퍼스나-밀집(dialogue data)가 아닌 데이터에 대한 의존성을 극복하는 데 성공했습니다.



### Hybrid Deep Learning for Legal Text Analysis: Predicting Punishment Durations in Indonesian Court Rulings (https://arxiv.org/abs/2410.20104)
Comments:
          11 pages, 7 figures, 6 tables, submitted to Journal of Advances in Information Technology

- **What's New**: 이번 연구는 인도네시아 법원 시스템의 법적 이해 부족과 불일치하는 판결 문제를 해결하기 위해 딥 러닝 기반의 예측 시스템을 개발하였습니다.

- **Technical Details**: 하이브리드 모델은 CNN(Convolutional Neural Network)과 BiLSTM(Bidirectional Long Short-Term Memory) 및 Attention Mechanism을 결합하여 문서의 지역 패턴과 장기 종속성을 효과적으로 캡처하였습니다. R-squared 점수는 0.5893에 달하며, 자주 사용되는 상위 30%의 토큰만을 활용하여 예측 성능을 향상시켰습니다. 또한 텍스트 정규화 과정을 수정하여 오타 및 잘못된 단어 결합과 같은 일반적인 오류를 해결하였습니다.

- **Performance Highlights**: 모델 성능이 대폭 향상되었으며, 이는 법률 문서 처리 자동화에 중요한 의미를 갖습니다. 본 연구는 NLP(Natural Language Processing) 기술을 활용하여 인도네시아 법률 시스템의 투명성과 접근성을 높이며, 일관되고 이해 가능한 법적 결정을 위한 기반을 마련했습니다.



### RARe: Retrieval Augmented Retrieval with In-Context Examples (https://arxiv.org/abs/2410.20088)
- **What's New**: 본 논문에서는 기존의 decoder-only language models (LLMs)에서 부분적으로 사용되는 in-context examples가 retrieval 모델의 성능 향상에 기여할 수 있는지를 연구합니다. 기존 LLMs에서의 사용 방법과는 달리, retrieval 작업에 적용하기 위해 새로운 접근법인 RARe(Retrieval Augmented Retrieval with In-Context Examples)를 제안합니다.

- **Technical Details**: RARe는 사전 훈련된 모델을 fine-tuning하여, 타겟 쿼리와 의미적으로 유사한 in-context examples를 추가하는 방식입니다. 이를 통해 retrieval 시스템의 쿼리 형식을 조정하고, contrastive loss를 사용하여 실험하고 성능을 평가합니다.

- **Performance Highlights**: RARe는 다양한 오픈 도메인 데이터셋(예: BeIR, RAR-b)에서 기본 모델보다 최대 2.72%의 nDCG 성능 향상을 보여주었으며, 특히 out-of-domain 일반화 성능이 뛰어난 것으로 나타났습니다.



### Architectural Flaw Detection in Civil Engineering Using GPT-4 (https://arxiv.org/abs/2410.20036)
- **What's New**: 이 논문은 인공지능(AI)의 응용이 토목 공학 디자인 품질 및 안전성을 향상시키는 혁신적 접근 방식을 제시한다고 설명합니다. 특히, 설계 단계에서 건축 결함을 감지하기 위해 고급 LLM GPT4 Turbo 비전 모델을 사용하고, 누락된 문과 창문을 식별하는 데 중점을 둡니다.

- **Technical Details**: 연구에서는 모델의 성능을 precision, recall, F1 score와 같은 여러 메트릭을 통해 평가하며, AI가 인간 검증 데이터와 비교하여 결함을 정확하게 감지하는 효과성을 보이고 있습니다. 이외에도 AI의 추가적인 능력으로 하중 지지 이슈, 재료 약점 식별, 건축 법규 준수 여부 등을 탐구합니다.

- **Performance Highlights**: AI는 설계 정확성을 현저하게 개선하고 비싼 수정 비용을 줄이며 지속 가능한 관행을 지원함으로써 궁극적으로 더 안전하고 효율적이며 미적으로 최적화된 구조물을 보장하여 토목 공학 분야를 혁신할 수 있는 잠재력을 보입니다.



### Beyond Fine-Tuning: Effective Strategies for Mitigating Hallucinations in Large Language Models for Data Analytics (https://arxiv.org/abs/2410.20024)
- **What's New**: 본 연구에서는 LLMs(대형 언어 모델)에서 발생하는 'hallucinations'(환각)을 완화하기 위한 새로운 전략들을 제안하고 평가합니다.

- **Technical Details**: 네 가지 구체적인 전략: Structured Output Generation(구조화된 출력 생성), Strict Rules Enforcement(엄격한 규칙 집행), System Prompt Enhancements(시스템 프롬프트 향상), Semantic Layer Integration(의미적 계층 통합) 등이 포함됩니다.

- **Performance Highlights**: 이 연구의 결과는 전통적인 fine-tuning(미세 조정) 접근 방식보다 이러한 방법들이 hallucinations를 줄이는 데 더 효과적임을 보여줍니다.



### Dynamic layer selection in decoder-only transformers (https://arxiv.org/abs/2410.20022)
- **What's New**: 이번 연구는 대량의 Large Language Models (LLMs)에 대한 동적 추론(dynamic inference) 기술을 분석하였으며, 특히 자연어 생성(NLG)에서의 layer skipping과 early exiting 방법에 중점을 두었습니다.

- **Technical Details**: Layer skipping과 early exiting은 LLM의 연산 비용을 줄이기 위해 설계된 동적 추론 방법입니다. 연구에서 layer skipping은 고는 네트워크에서 레이어를 건너뛰는데 더 효과적이며, hidden state 정보를 사용한 per-token 적응은 어려움을 겪습니다. 대안적으로, sequence-level에서 동적 연산 할당을 사용해 significant efficiency gains를 이끌 수 있는 방법을 제안합니다.

- **Performance Highlights**: 연구 결과, layer skipping은 hidden state error를 최소화하는 데 효과적이며, 평균 23.3% Layer만으로도 전체 모델과 동등한 성능을 나타내는 동적 연산 할당 방식을 사용할 수 있음을 발견했습니다.



### Think Carefully and Check Again! Meta-Generation Unlocking LLMs for Low-Resource Cross-Lingual Summarization (https://arxiv.org/abs/2410.20021)
- **What's New**: 이 논문은 저자들이 제안한 4단계 제로 샷(zero-shot) 접근 방식을 통해 자원이 제한된(low-resource) 언어에 대한 Cross-lingual summarization(클래스) 성능을 향상시키는 방법을 다룹니다. 저자들은 Summarization, Improvement, Translation, Refinement(SITR) 방법론을 사용하여 대규모 언어 모델(LLMs)의 잠재력을 극대화하였습니다.

- **Technical Details**: 연구에서는 4단계의 SITR 방법이 사용됩니다: (1) Summarization: 원본 텍스트에서 요약 생성, (2) Improvement: 요약 개선, (3) Translation: 저자들이 제공한 번역 프롬프트를 사용해 저자 언어로 번역, (4) Refinement: 앞서 생성된 요약과 번역을 결합해 최종 결과물 생성.

- **Performance Highlights**: 실험 결과, GPT-3.5와 GPT-4가 다른 LLM 기준에서 상당히 우수한 성능을 보였으며, 자원이 제한된 언어에 대해 많은 모델을 초과하는 성과를 달성했습니다. 이 연구는 LLM이 저자 언어에도 잘 대응할 수 있음을 보여주며, 특히 저자 언어에 대한 CLS 작업에서의 잠재력을 재조명합니다.



### Attacks against Abstractive Text Summarization Models through Lead Bias and Influence Functions (https://arxiv.org/abs/2410.20019)
Comments:
          10 pages, 3 figures, Accepted at EMNLP Findings 2024

- **What's New**: 이번 연구에서는 Text Summarization 모델의 적대적 강건성(adversarial robustness)을 탐구하며, 리드 바이어스(lead bias)를 활용한 새로운 공격 접근법을 제시합니다. 또한, 데이터 포이즌킹(data poisoning)을 위해 영향 함수(influence functions)를 적용하여 모델의 무결성을 손상시키는 혁신적인 방법론을 소개합니다.

- **Technical Details**: 이 연구에서는 BART, T5, Pegasus 같은 텍스트 요약 모델을 다양한 적대적 교란(adversarial perturbations)에 대한 반응을 분석합니다. 주요 공격 전략으로는 모델의 학습 데이터셋에서 영향력 있는 데이터를 확인하고 이를 조작하여 모델의 출력을 왜곡하는 포이즌킹 공격을 적용합니다. 텍스트 요약에서 리드 바이어스를 활용하는 첫 번째 연구로, 모델이 요약을 생성할 때 초기 문장에 지나치게 의존하는 경향을 이용합니다.

- **Performance Highlights**: 저자들은 Multi-Document Text Summarization (MDTS)에서 이러한 취약점을 체계적으로 노출시키며, 공격 후 모델이 생성하는 요약이 추출적 요약(extractive summaries)에서 벗어나지 않음을 보여줍니다. 이는 데이터 포이즌킹 공격에 대한 취약성을 나타내며, 모델이 적대적 영향력 하에서 텍스트 정보를 처리하고 요약하는 방식의 근본적 변화를 시사합니다.



### Vulnerability of LLMs to Vertically Aligned Text Manipulations (https://arxiv.org/abs/2410.20016)
- **What's New**: 이번 논문은 수직으로 포맷된 텍스트가 여러 LLMs(대형 언어 모델)의 성능에 미치는 영향을 조사하며, 이러한 포맷이 텍스트 분류 작업에서 모델의 정확도를 크게 저하시키는 것으로 나타났습니다.

- **Technical Details**: 연구에서는 다양한 텍스트 분류 데이터셋을 활용하여 수직 텍스트 입력을 평가하고, CoT(Chain of Thought) 추론이 아닌 세심한 분석을 통한 few-shot learning이 효과적이라는 것을 발견했습니다. 또한, 토큰화(tokenization) 및 주목 행렬(attention matrices)의 내재적 문제를 분석하여 LLMs의 취약성을 탐구했습니다.

- **Performance Highlights**: 결과적으로, LLMs는 수직 포맷된 입력에 노출되었을 때 상당한 정확도 감소를 경험하며, 최신 모델조차도 이 형식의 입력 처리에 어려움을 겪습니다.



### A Survey of Small Language Models (https://arxiv.org/abs/2410.20011)
- **What's New**: 이번 논문은 Small Language Models (SLMs)에 대한 포괄적인 조사 결과를 제공합니다. SLMs는 적은 계산 자원으로 효과적으로 다양한 언어 작업을 수행할 수 있어 모바일 및 엣지 디바이스 등 다양한 환경에서 이상적입니다. 본 논문은 SLM의 아키텍처, 훈련 기술 및 모델 압축 기법에 대한 새롭고 상세한 분류 체계를 제안합니다.

- **Technical Details**: SLMs의 아키텍처에는 경량화된 디자인, 효율적인 self-attention 근사 및 신경망 아키텍처 검색 등이 포함됩니다. 모델 압축 기법으로는 pruning, quantization 및 knowledge distillation이 있으며, 이러한 기술들은 모델의 크기와 지연 시간을 줄이면서도 정확성을 유지하는 데 도움을 줍니다. 본 조사에서는 또한 SLM 성능 평가에서 사용되는 벤치마크 데이터셋과 평가 메트릭스를 요약합니다.

- **Performance Highlights**: SLMs는 제한된 자원 환경에서도 높은 정확성을 유지하며, BabyLLaMA, TinyLLaMA 등 다양한 모델들이 96% 이상의 성능을 달성했습니다. 이들은 특히 데이터 제약 조건에서 교사 모델보다 우수한 성능을 보임으로써, 실용적인 적용 가능성을 보여주고 있습니다.



### Layer by Layer: Uncovering Where Multi-Task Learning Happens in Instruction-Tuned Large Language Models (https://arxiv.org/abs/2410.20008)
Comments:
          Accepted to EMNLP 2024

- **What's New**: 본 연구는 사전 훈련된 대형 언어 모델(LLMs)에서 특정 작업(task) 정보의 인코딩과 명령어 조정(instruction tuning)의 효과를 조사합니다. 60개 이상의 다양한 자연어 처리(NLP) 작업에 대한 분석을 통해, 사전 훈련된 LLM이 이미 반영하고 있는 작업 정보의 범위 및 명령어 조정이 표현을 어떻게 수정하는지를 밝혀냈습니다.

- **Technical Details**: 모델 지향 서브 집단 및 스펙트럼 분석(Model-Oriented Sub-population and Spectral Analysis, MOSSA)이라는 기법을 사용하여 LLM의 표현을 비교합니다. 이는 특정 서브 집단의 훈련 데이터 내에서 모델 표현을 분석하는 대안 기법으로, 제어 모델(control model)과 실험 모델(experimental model)의 표현 차이를 비교하여 작업에 특화된 정보를 분리해낼 수 있습니다. 또한 중심 커널 정렬(Center Kernel Alignment, CKA) 메트릭을 통해 두 모델 간의 표현 유사성을 측정합니다.

- **Performance Highlights**: 연구 결과, LLM에서 고수준 일반 표현(high-level general representations)과 작업 지향 표현(task-oriented representations)으로 전환되는 층을 식별했습니다. 연구는 세 가지 기능 그룹의 층을 밝혀냈습니다: a) 공유층(shared layers), b) 전환층(transition layers), c) 정제층(refinement layers). 이러한 발견은 LLM의 작동 원리를 깊이 이해하는 데 기여하며, 매개변수 효율적 전이 학습(parameter-efficient transfer learning), 다중 작업 학습(multi-task learning), 모델 압축(model compression) 등의 미래 연구에 유망한 시사점을 제공합니다.



### Do Discrete Self-Supervised Representations of Speech Capture Tone Distinctions? (https://arxiv.org/abs/2410.19935)
Comments:
          Submitted to ICASSP 2025

- **What's New**: 이 연구는 Self-Supervised Learning(SSL) 기반 모델에서 획득한 음성의 이산 기호(discrete symbols)가 톤(tone)을 충분히 포착하는지 평가합니다. 특히, k-means를 사용하여 발견된 이산 기호가 Mandarin과 Yoruba와 같은 톤 언어의 음성을 잘 캡처하고 있는지를 분석했습니다.

- **Technical Details**: 이 원고는 HuBERT, MandarinHuBERT, XLS-R을 기반으로 한 SSL 모델에서 생성된 잠재 벡터(latent vectors)와 이산 기호를 비교하여 모음(vowel) 및 톤 분류의 성능을 보여줍니다. 이산 기호는 k-means 클러스터링을 통해 생성됩니다.

- **Performance Highlights**: 연구 결과, 이산 기호를 사용하면 톤 정보가 상당히 손실된다는 것을 발견했습니다. 이는 언어 특화된 SSL 모델을 사용한 경우에도 마찬가지였으며, 결과적으로 톤 의존적인 하위 작업에서는 이산화가 작업에 대한 인식(task-aware)이어야 함을 제안합니다.



### Improving Multimodal Large Language Models Using Continual Learning (https://arxiv.org/abs/2410.19925)
Comments:
          NeurIPS 2024 Workshop on Scalable Continual Learning for Lifelong Foundation Models

- **What's New**: 이 연구는 LLaVA MLLM을 통해 시각 정보를 LLM에 통합할 때 발생하는 언어 이해 및 생성 성능 저하 문제를 지속적 학습(Continual Learning) 문제로 다룹니다. 이는 기존 LLM에 비해 MLLM이 자연어 처리에서 성능이 감소하는 현상을 줄여주는 초기 기법을 모색하고, 이를 통해 MLLM의 시각적 이해를 향상시키는 방법을 제안합니다.

- **Technical Details**: LLaVA MLLM의 통합 과정에서 발생하는 언어적 망각 문제를 해결하기 위해, 5가지 지속적 학습 방법을 검토합니다. 실험을 통해 시각-언어(VL) 작업에서의 성능 저하를 최대 15%까지 줄이는 기법을 도출하였습니다. 언어적 성능 저하를 줄이면서 높은 다중 모달 정확도를 유지하는 방법으로서 Soft Targets, LoRA, mSGM 등의 기술이 활용됩니다.

- **Performance Highlights**: LLaVA의 기존 수치와 비교하여, 제안된 방법이 NLG, NLU, VL 작업의 성능을 개선한 결과를 보여주며, 특히 Soft Targets가 VL 데이터셋에서 가장 높은 정확도를 올리는 것으로 나타났습니다. MLLM의 지속적 학습 실험을 통해 언어적 기술은 유지하면서도 새로운 다중 모달 능력을 성공적으로 습득하였습니다.



### Ensembling Finetuned Language Models for Text Classification (https://arxiv.org/abs/2410.19889)
Comments:
          Workshop on Fine-Tuning in Modern Machine Learning @ NeurIPS 2024. arXiv admin note: text overlap with arXiv:2410.04520

- **What's New**: 이 논문에서는 5개의 대형 미세 조정(finetuned) 모델의 예측 결과를 바탕으로 한 메타데이터셋(metadataset)을 제시하며, 이를 통해 다양한 앙상블(ensemble) 전략이 어떻게 미세 조정된 텍스트 분류(text classification) 모델의 성능을 향상시킬 수 있는지를 탐구합니다.

- **Technical Details**: 제시된 메타데이터셋인 FTC(Finetuning Text Classifiers)에는 6개의 서로 다른 데이터셋에 대한 5개의 미세 조정된 모델의 예측이 포함되어 있으며, 다양한 앙상블 방법들이 평가됩니다. 앙상블 모델의 하이퍼파라미터(hyperparameters)로는 모델 유형, 학습률, LoRA 순위 등이 포함되어 있습니다.

- **Performance Highlights**: 연구 결과는 미세 조정된 모델의 앙상블을 통해 텍스트 분류 작업에서 성능 향상이 가능함을 보여주며, 이는 향후 해당 분야에서 앙상블 방법의 사용을 촉진할 것으로 기대됩니다.



### Critical biblical studies via word frequency analysis: unveiling text authorship (https://arxiv.org/abs/2410.19883)
- **What's New**: 본 연구는 성경 저자 간의 언어적 특성을 통계적으로 분석하여 성경 본문의 저자 문제에 대한 새로운 통찰을 제공합니다. 특히, 성경의 첫 아홉 권에 걸친 50개의 장을 조사하였습니다.

- **Technical Details**: 단어 빈도(frequency)를 활용하여 저자별 언어적 특성을 비교하고, 문서 집합(D, DtrH, P) 간의 미세한 차이를 분석합니다. 이를 통해 사전 가정 없이 저자의 정체성을 식별하는 접근법을 사용하였습니다.

- **Performance Highlights**: 첫 두 저자(D와 DtrH)는 서로 유사한 특징을 보였으며, P 저자와는 뚜렷한 차이점을 나타냈습니다. 이러한 결과는 전문가의 평가와 일치하며, 높은 정확도로 저자를 규명할 수 있음을 보여줍니다.



### Parameter-Efficient Fine-Tuning in Large Models: A Survey of Methodologies (https://arxiv.org/abs/2410.19878)
- **What's New**: 이 리뷰 논문은 Parameter-Efficient Fine-Tuning(PEFT)에 대한 종합적인 개요를 제공하며, 대규모 모델이 특정 작업에 적합하도록 조정하는 다양한 알고리즘을 설명합니다. 이 논문은 최근 PEFT 기술의 발전과 응용 분야를 탐구하며, 향후 연구 방향에 대한 제안을 포함합니다.

- **Technical Details**: PEFT는 대규모 사전 훈련된 모델의 매개변수를 새로운 작업이나 시나리오에 적합하게 조정하는 전이 학습(transfer learning) 방법이다. 주요 PEFT 접근 방식으로는 LoRA, adapter tuning, prefix-tuning, prompt-tuning, P-tuning, BitFit 등이 있다. 이 기법들은 기존의 매개변수를 유지하면서도 성능 향상을 목표로 한다.

- **Performance Highlights**: PEFT는 여러 NLP 작업에서 성능을 최적화할 수 있는 잠재력을 가지고 있으며, 수백 개의 PEFT 관련 논문이 출판되었습니다. 이 리뷰는 PEFT 방법론의 현재 이해를 돕고, 앞으로의 연구를 위한 유용한 정보와 통찰을 제공하는 것을 목표로 합니다.



### Zero-Shot Dense Retrieval with Embeddings from Relevance Feedback (https://arxiv.org/abs/2410.21242)
- **What's New**: 이 논문에서는 ReDE-RF(Real Document Embeddings from Relevance Feedback)라는 새로운 접근 방식을 도입하여, LLM을 활용한 가상의 문서 생성 대신 관련성 추정(task)을 프레임으로 삼아 검색 효율성을 크게 향상시킵니다.

- **Technical Details**: ReDE-RF는 먼저 완전 비지도 하이브리드 스파스-밀집 검색 시스템에서 초기 문서 집합을 추출한 후, LLM을 통해 반환된 문서가 관련성이 있는지 없는지를 평가합니다. 이후, 관련 문서 집합을 기반으로 미리 계산된 문서 임베딩을 효율적으로 가져와 쿼리 벡터를 업데이트합니다. 이는 LLM이 도메인 특화 지식에 의존하지 않고 단순히 관련성을 판단하게 합니다.

- **Performance Highlights**: 실험 결과, ReDE-RF는 여러 저자원 검색 데이터셋에서 기존의 최첨단 제로샷 밀집 검색 방법들보다 최대 14% 더 우수한 성능을 보였으며, 검색 지연(lag)을 7.5배에서 11.2배까지 줄이는 성과를 나타냈습니다.



### Flaming-hot Initiation with Regular Execution Sampling for Large Language Models (https://arxiv.org/abs/2410.21236)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 훈련 중 효율적인 샘플링 방법인 Flaming-hot Initiation with Regular Execution (FIRE) 샘플링을 소개합니다. 이 방법은 초기 토큰을 높은 온도에서 샘플링하여 우수한 응답을 찾아내는 방식으로, 다양한 도메인에서 세분화된 문제 해결 능력을 향상시키는 데 기여합니다.

- **Technical Details**: FIRE sampling은 높은 온도의 초기 토큰 샘플링과 일반적인 샘플링 프로세스를 결합하여 진행됩니다. 이 접근법은 attention sink 이론을 활용하며, 문제 해결 시 샌드박스 체커의 존재로 인해 더 나은 샘플을 생성할 수 있도록 돕습니다. FIRE는 CoT-decoding의 간소화 및 일반화를 통해 매개변수 조정 없이도 기존 훈련 프레임워크와 통합될 수 있습니다.

- **Performance Highlights**: FIRE sampling을 통해 여러 공개 소스 모델에서 추론 시간의 생성 품질이 개선되었고, 특히 수학 문제와 코드 생성에서 높은 통과율(pass rate)을 기록했습니다. 이 방법을 통해 생성된 샘플의 다양성이 향상되었으며, 이는 결과적으로 성능 개선과 직결됩니다.



### LoRA vs Full Fine-tuning: An Illusion of Equivalenc (https://arxiv.org/abs/2410.21228)
- **What's New**: 이 논문은 Low-Rank Adaptation (LoRA)와 전체 미세 조정(full fine-tuning) 방법이 학습한 솔루션의 본질적 차이를 다루고 있습니다. 연구 결과, 두 방법은 동일한 성능을 달성하더라도 모델의 weight matrices의 구조가 다르며, LoRA 모델에서 새로운 'intruder dimensions'가 나타나는 것을 보였습니다.

- **Technical Details**: LoRA에서는 weight matrix의 업데이트를 두 개의 저차원(low-rank) 행렬의 곱으로 표현합니다. 본 논문에서는 full fine-tuning과 LoRA가 만든 weight matrices의 singular value decomposition(특이값 분해)의 스펙트럼 특성을 분석하여 두 방법의 차이를 설명합니다. LoRA는 'intruder dimensions'라 불리는 새로운 특이 벡터를 포함하여 다른 구조의 매개변수를 생성합니다.

- **Performance Highlights**: LoRA와 전체 미세 조정은 동일한 목표 태스크에서 유사한 성능을 달성하지만, LoRA로 조정된 모델은 훈련 분포를 잊어버리고 새로운 태스크에 적응할 때 더 비효율적이라는 결과가 나타났습니다. 높은 차수의 LoRA 모델은 전체 미세 조정 모델과 유사한 구조적 특징과 일반화 성능을 보여줍니다.



### Document Parsing Unveiled: Techniques, Challenges, and Prospects for Structured Information Extraction (https://arxiv.org/abs/2410.21169)
- **What's New**: 이번 논문은 문서 파싱(document parsing)의 최신 발전 상태를 포괄적으로 리뷰하며, 대형 언어 모델과 비전-언어 모델을 활용한 문서 파싱 기술의 중요성을 강조합니다.

- **Technical Details**: 주요 방법론으로는 모듈형 파이프라인 시스템(modular pipeline systems)과 end-to-end 모델이 있습니다. 핵심 구성 요소에는 레이아웃 감지(layout detection), 콘텐츠 추출(content extraction), 멀티모달 데이터 통합(multi-modal data integration)이 포함됩니다. 또한 복합 레이아웃을 처리하는 데 필요한 도전과제와 기술적 발전도 다루어집니다.

- **Performance Highlights**: 문서 파싱 기술이 정보 추출 및 문서 이해와 같은 여러 작업에서 중대한 진전을 이루었으며, 이는 RAG 시스템 및 차세대 지능형 시스템 개발을 위한 견고한 기초를 제공합니다.



### Towards Unifying Evaluation of Counterfactual Explanations: Leveraging Large Language Models for Human-Centric Assessments (https://arxiv.org/abs/2410.21131)
Comments:
          This paper has been submitted in August and is currently under review to AAAI-2025

- **What's New**: 이 논문에서는 카운터팩추얼( counterfactual ) 설명의 평가를 자동화할 수 있는 가능성을 탐구하여, 이를 위해 다양한 차원의 설명 품질을 가진 30개의 카운터팩추얼 시나리오를 개발하고 206명의 응답자로부터 평가를 수집했습니다.

- **Technical Details**: 제안된 방법론은 8개의 평가 지표를 바탕으로 LLM (Large Language Model) 모델을 미세 조정하여 인간의 평균 또는 개별 판단을 예측하도록 함으로써, 카운터팩추얼 설명 프레임워크의 평가에서 더 나은 비교 가능성과 확장성을 제공합니다.

- **Performance Highlights**: 미세 조정된 LLM 모델은 제로샷 평가에서 최대 63%의 정확도를 달성하였고, 미세 조정 후 모든 지표에서 3개 클래스 예측에 대해 85%의 정확도를 기록하여 인간 평가와의 비교에서 좋은 성능을 보였습니다.



### Zero-Shot Action Recognition in Surveillance Videos (https://arxiv.org/abs/2410.21113)
- **What's New**: 이번 연구는 비디오 감시 분야에서 인간 자원의 부족 문제를 해결하기 위해 Large Vision-Language Models (LVLMs)를 활용하는 새로운 접근 방식을 제안합니다. 특히, Self-Reflective Sampling (Self-ReS)이라는 새로운 샘플링 메소드를 도입하여 비디오 이해 작업의 성능을 향상시키고자 했습니다.

- **Technical Details**: 연구에서는 LVLMs 중 가장 최신 모델인 VideoLLaMA2를 사용하여 실험을 진행했습니다. Self-ReS는 모델의 기본 샘플링 방법 대신에 각 비디오의 가장 관련성 높은 토큰을 선택하는 방식으로, 고유한 샘플링 전략을 통해 비디오의 시공간적 서명(spatio-temporal signature)을 효율적으로 생성합니다.

- **Performance Highlights**: 실험 결과, VideoLLaMA2는 baseline보다 20% 더 향상된 제로샷(zero-shot) 성능을 나타냈으며, Self-ReS를 통해 제로샷 행동 인식 성능은 44.6%로 증가했습니다. 이러한 결과는 LVLMs와 개선된 샘플링 기법이 다양한 감시 비디오 분석 시나리오에서 큰 잠재력을 가지고 있음을 보여줍니다.



### Sorting Out the Bad Seeds: Automatic Classification of Cryptocurrency Abuse Reports (https://arxiv.org/abs/2410.21041)
- **What's New**: 이 논문은 암호화폐(cryptocurrency) 악용 보고서를 자동으로 분류하는 새로운 접근 방식을 제시합니다.

- **Technical Details**: 19가지 자주 보고된 악용 유형에 대한 분류 체계(taxonomy)를 구축하고, 보고자가 작성한 텍스트 설명을 입력으로 받아 대형 언어 모델(large language model, LLM)을 활용하여 해당 악용 유형을 분류합니다. 이 연구는 BitcoinAbuse와 BBB의 ScamTracker로부터 수집한 290K개의 암호화폐 악용 보고서를 사용합니다.

- **Performance Highlights**: LLM 기반 분류기는 0.92의 정밀도(precision), 0.87의 재현율(recall), 0.89의 F1 점수를 달성하였으며, 이는 기본(baseline) 모델의 0.55 F1 점수와 비교됩니다. 이 모델은 세부 악용 유형에 대한 재정 손실 통계 제공과 암호화폐 분석 플랫폼을 위한 태그가 붙은 주소 생성 두 가지 응용 프로그램에서 시연됩니다.



### Beyond Autoregression: Fast LLMs via Self-Distillation Through Tim (https://arxiv.org/abs/2410.21035)
- **What's New**: 이 연구에서는 기존의 Autoregressive (AR) 모델의 제한점을 극복하기 위해 Diffusion Language Models를 사용하여 동시에 최소 32개의 토큰을 생성할 수 있음을 보여줍니다.

- **Technical Details**: Self-Distillation Through Time (SDTT)라는 새로운 증류 방법을 도입하여, 디퓨전 모델이 더 적은 추론 단계(steps)로도 높은 품질의 텍스트를 생성할 수 있도록 하였습니다. 이 모델은 최대 860M 파라미터를 지원하며, KV 캐싱을 사용하는 AR 모델보다 최대 8배 더 빠릅니다.

- **Performance Highlights**: LAMBADA 자연어 이해 벤치마크에서 AR 모델보다 더 나은 텍스트 품질을 달성했습니다. 우리의 접근 방식을 통해 디퓨전 모델은 추론 단계 수를 32-64배 줄이면서도 빠른 속도와 높은 품질을 유지할 수 있습니다.



### Transferable Post-training via Inverse Value Learning (https://arxiv.org/abs/2410.21027)
- **What's New**: 본 논문에서는 모델의 post-training 과정에서 발생하는 변화를 logits 수준에서 별도의 신경망(value network)을 이용해 모델링 하는 방법을 제안합니다. 이 네트워크는 소형 기본 모델에 대해 시연 데이터를 사용해 훈련된 후, 다른 선 훈련된 모델과 쉽게 통합되어 인퍼런스(inference) 동안 유사한 성능 향상을 이끌어낼 수 있습니다.

- **Technical Details**: 제안된 프레임워크는 logits 공간을 공유 인터페이스로 활용하여 다양한 크기와 계열의 모델을 효율적으로 적응시키며, 두 가지 연결 아키텍처(Cascade 및 Residual)를 조사합니다. Residual 연결 방식이 이전 텍스트 입력만을 기반으로 residuals를 예측하여 더 우수한 전이 가능성과 효율성을 보임을 실험적으로 확인하였습니다.

- **Performance Highlights**: 결과적으로 이 프레임워크는 동일한 모델 계열 내에서 다양한 매개변수 크기를 가진 pre-trained 모델 간의 전이 가능성과, 다른 모델 계열 간의 전이 가능성을 보여주었으며, 특정 경우에는 전체 매개변수를 파인튜닝한 성능에 근접한 성과를 달성했습니다. 이는 이 방법이 효율적이며 실용적인 어플리케이션에서의 잠재력을 지니고 있음을 강조합니다.



### Bridging the Gap between Expert and Language Models: Concept-guided Chess Commentary Generation and Evaluation (https://arxiv.org/abs/2410.20811)
- **What's New**: 이 연구에서는 체스 전문가 모델과 대규모 언어 모델(LLMs) 간의 간극을 메우기 위한 새로운 접근 방식인 Concept-guided Chess Commentary generation (CCC)와 GPT-based Chess Commentary Evaluation (GCC-Eval)을 소개합니다. CCC는 전문가 모델의 결정 능력과 LLM의 언어적 유창성을 통합하여 체스 결정 과정을 설명하는 데 중점을 둡니다.

- **Technical Details**: CCC는 전문가 모델이 집중하는 개념을 추출하고 우선순위를 부여하여 LLM이 게임의 가장 중요한 측면에 집중하게 합니다. GCC-Eval는 체스 전문 지식을 활용하여 생성된 해설의 정보성 (informativeness)과 언어 품질 (linguistic quality)을 평가합니다.

- **Performance Highlights**: 실험 결과에 따르면 CCC는 인간 수준의 정확성을 달성하였으며, 인간이 생성한 코멘트보다 정보성과 언어적 품질에서 우수한 성능을 보여주었습니다. 또한, GCC-Eval는 사람의 평가와 잘 상관되어 더 신뢰할 수 있는 평가 지표로 기능하고 있습니다.



### Matryoshka: Learning to Drive Black-Box LLMs with LLMs (https://arxiv.org/abs/2410.20749)
Comments:
          Work in Progress

- **What's New**: 최근 대규모 블랙박스 언어 모델(LLM)의 기본적인 불투명성이 인지적 기능의 향상을 저해하고 있다는 문제 의식을 바탕으로, 새로운 컨트롤러 모델인 Matryoshika를 소개합니다. 이 모델은 복잡한 작업을 중간 출력으로 분해하여 블랙박스 LLM을 유도하는 경량의 화이트박스 LLM 컨트롤러입니다.

- **Technical Details**: Matryoshika는 두 가지 주요 구성 요소로 구성되며, 화이트박스 LLM이 컨트롤러로 작용하고 블랙박스 LLM이 생성기를 역할을 합니다. 입력 질문에 대해 컨트롤러는 중간 출력을 생성하고, 이는 블랙박스 LLM의 기능을 향상시키도록 설계되었습니다. Matryoshika는 정책을 기반으로 출력의 피드백을 통해 최적화를 수행하며, 단계적 상호작용을 통해 지속적인 자기 개선(self-improvement)을 가능하게 합니다.

- **Performance Highlights**: 세 가지 복잡한 작업에 대한 실험 결과, Matryoshika는 추론에서 평균 3.19%, 계획에서 성공률 7.46%, 개인화에서 정확도 5.82%의 향상을 보여주었습니다. 이러한 결과는 Matryoshika가 블랙박스 LLM의 성능을 효과적으로 개선하는 데 기여할 수 있음을 시사합니다.



### Guide-LLM: An Embodied LLM Agent and Text-Based Topological Map for Robotic Guidance of People with Visual Impairments (https://arxiv.org/abs/2410.20666)
- **What's New**: 이 논문은 시각 장애인(PVI)를 위한 내비게이션을 개선하기 위해 디자인된 Guide-LLM이라는 새로운 경량 언어 모델(LLM) 기반 에이전트를 소개합니다. 이는 기존의 도구들이 제공하지 못하는 정교한 공간 정보를 제공합니다.

- **Technical Details**: 당신의 초점은 텍스트 기반의 토포로지 맵(topological map)을 활용하여 LLM이 간단화된 환경 표현을 통해 경로를 계획할 수 있도록 하는 것입니다. 이 시스템은 사용자 선호도에 따라 위험을 탐지하고 개인화된 경로 계획을 가능하게 합니다.

- **Performance Highlights**: 모의 실험을 통해 Guide-LLM의 효과가 입증되었으며, 이는 효율적이고 적응력 있는 개인화된 내비게이션 지원을 제공함으로써 보조 기술에서 상당한 기술 발전을 나타냅니다.



### LoRA Done RITE: Robust Invariant Transformation Equilibration for LoRA Optimization (https://arxiv.org/abs/2410.20625)
- **What's New**: 이 논문에서는 Low-Rank Adaptation (LoRA)의 최적화 문제를 해결하기 위해 LoRA-RITE라는 새로운 어댑티브 매트릭스 전처리 방법을 소개합니다. 기존의 LoRA 최적화 방법이 갖는 변환 불변성(transform invariance) 부족 문제를 해결하여 학습 효율을 개선합니다.

- **Technical Details**: LoRA는 훈련되지 않은 가중치(matrix) W에 저차원 행렬(저랭크 매트릭스) Z를 주입하는 방식으로 작동합니다. 저랭크 행렬 Z는 두 개의 행렬 A와 B의 곱으로 나타낼 수 있으며, LoRA-RITE는 이러한 것에 대해 변환 불변성을 제공하도록 설계되었습니다. 기존의 최적화 알고리즘들은 일반적으로 이 속성을 증명하지 못하고, 이는 학습 과정에서 비효율성을 초래합니다.

- **Performance Highlights**: LoRA-RITE는 Gemma 2B를 사용해 Super-Natural Instructions에서 4.6%의 정확도 향상을, 총 4개의 LLM 벤치마크에서는 3.5%의 성능 향상을 달성했습니다. GSM8K 데이터셋에서는 Gemma 7B IT 모델을 사용하여 55.5%의 정확도를 기록하였으며, 이는 Adam 최적화 알고리즘의 48.37%보다 월등한 수치입니다.



### Towards an LLM-Based Speech Interface for Robot-Assisted Feeding (https://arxiv.org/abs/2410.20624)
- **What's New**: 이 논문에서 물리적 지원 로봇을 위한 LLM 기반 음성 인터페이스를 소개합니다. 이 시스템은 상업적으로 이용 가능한 지원 급식 로봇인 Obi에 적용되어, 사용자가 자체적인 명령과 세부 선호를 자연스럽게 전달할 수 있도록 설계되었습니다.

- **Technical Details**: 시스템은 GPT-3.5 Turbo를 기반으로 하며, 사용자 훈련과 테스트를 위해 노인 시설에서 연구가 수행되었습니다. 사용자는 로봇에게 'Hey Obi'라는 호출어를 이용하여 명령을 전달하며, 명령은 OpenAI의 Whisper API를 통해 텍스트로 변환되고 GPT에 의해 처리됩니다. 로봇은 세 가지 주요 제어 기능을 통해 안전하고 예측 가능한 방식으로 작동하며, 사용자 맞춤의 이동 조정이 가능합니다. 또한 사용자는 로봇의 움직임을 멈추거나 일시정지할 수 있는 기능을 음성으로 호출할 수 있습니다.

- **Performance Highlights**: 사용자 연구를 통해 11명의 노인이 참여하였으며, LLM 기반 인터페이스가 로봇과의 상호작용에서 긍정적인 반응을 얻었습니다. 사용자 맞춤의 명령 기능은 개인의 식사 경험을 향상시키고, 로봇의 움직임에 대한 예측 가능성을 제공하여 사용자 안전을 확보했습니다.



### Guiding Through Complexity: What Makes Good Supervision for Hard Reasoning Tasks? (https://arxiv.org/abs/2410.20533)
- **What's New**: 본 논문에서는 약한 교사 모델(weak teacher models)인 평균적인 인간 주석자나 기존 AI 시스템들이 LLM(대규모 언어 모델)의 성능을 향상시키기 위해 어떻게 효과적으로 감독할 수 있는지를 탐구합니다. 특히, 이러한 모델들이 난이도가 높은 추론 작업에서 어떻게 더 나은 성과를 낼 수 있는지에 대한 데이터 기반 전략을 제시합니다.

- **Technical Details**: 저자들은 두 가지 감독 전략을 제안합니다. 첫 번째 전략은 목표 추론 작업의 난이도와 일치하는 전체 작업에서 낮은 품질의 감독을 사용하는 것이고, 두 번째 전략은 더 쉬운 하위 작업(subtask)에서 높은 품질의 감독을 활용하는 것입니다. 연구 결과, 고차원 과업에 대한 감독이 높은 오류율(예: 90%)을 가지고 있음에도 불구하고, 보다 쉬운 하위 작업의 완벽한 감독보다 더 나은 성과를 보여줍니다. 중요한 배경 요인으로는 단계별 오류율(step-wise error rates)이 지적됩니다.

- **Performance Highlights**: 고차원 작업 감독이 하위 작업 감독보다 항상 우수한 성과를 제공하는 것으로 나타났습니다. 다양한 조합으로 하위 작업 감독을 보충했을 때, MATH 및 SAT-Math 벤치마크에서 최고의 성과를 달성했습니다. 구체적으로, A=50%와 B=10% 일 때 최고의 결과를 보였으며, 이는 단순히 하드 작업 감독을 두 배로 늘리거나 하드 작업 감독을 조합한 경우보다 더 효과적이라는 결과를 얻었습니다.



### Llama Scope: Extracting Millions of Features from Llama-3.1-8B with Sparse Autoencoders (https://arxiv.org/abs/2410.20526)
Comments:
          22pages, 12 figures

- **What's New**: 이번 연구에서는 Llama-3.1-8B 모델의 모든 층과 서브레이어에 대해 256개의 Sparse Autoencoders (SAEs)를 훈련시키고, 스케일 가능한 훈련과 해석 도구를 제공하여 미세 조정 모델에 대한 SAEs의 일반화 가능성을 평가합니다.

- **Technical Details**: Llama Scope는 32K 및 128K 특성을 갖춘 256개의 SAE로 구성되며, Top-K SAEs 변형을 적용하여 다양한 차원에서 평가합니다. 이 SAEs는 메모리 병목 현상을 줄이기 위한 혼합 병렬 처리 접근 방식을 사용하여 훈련됩니다.

- **Performance Highlights**: Llama Scope의 SAE 체크포인트는 공개되어 있으며, 다양한 평가 지표를 사용하여 기능 해석 가능성과 잠재적 전압 주파수를 포함한 성능을 Assess 합니다. 이 연구는 오픈 소스 SAE 생태계를 확장하고 기계적 해석 가능성 연구를 지원하기 위한 기여를 목표로 합니다.



### Graph Neural Networks on Discriminative Graphs of Words (https://arxiv.org/abs/2410.20469)
- **What's New**: 이번 연구에서는 Discriminative Graph of Words Graph Neural Network (DGoW-GNN)이라는 새로운 접근 방식을 소개합니다. 이 방법은 단어 노드만 포함된 그래프를 구축하고, 훈련 데이터셋을 레이블에 따라 분리된 서브그래프로 나누며, 단어의 pointwise mutual information을 통해 엣지를 가중치화합니다.

- **Technical Details**: DGoW-GNN은 단어 노드 간의 연결을 pointwise mutual information에 기반하여 구축하며, 각 클래스에 대한 disconnected subgraph로 훈련 데이터셋을 분리합니다. 이를 통해 텍스트 분류 문제를 walk classification 문제로 재정의할 수 있습니다. 또한, GNN과 sequence model을 결합하여 문서의 구조적 정보와 단어의 순서를 동시에 고려합니다.

- **Performance Highlights**: 실험을 통해 7개의 벤치마크 데이터셋에서 DGoW-GNN을 평가한 결과, 여러 최신 모델에 비해 성능이 다소 낮은 것으로 나타났습니다. 이 성능 차이에 대한 분석과 변화 가능성을 모색하고 있습니다.



### AutoKaggle: A Multi-Agent Framework for Autonomous Data Science Competitions (https://arxiv.org/abs/2410.20424)
Comments:
          44 pages, 10 figures

- **What's New**: AutoKaggle은 데이터 과학자들이 데이터 파이프라인을 효율적으로 완성할 수 있도록 돕는 협력적 다중 에이전트 시스템을 기반으로 하는 강력한 프레임워크입니다. 이 프레임워크는 코드 실행, 디버깅, 단위 테스트를 통합하여 코드의 정확성과 논리 일관성을 보장합니다.

- **Technical Details**: AutoKaggle은 단계 기반 워크플로우와 다중 에이전트 협업 시스템을 사용하여 데이터 과학 경쟁 과정을 6개 핵심 단계로 나눕니다. 이 단계는 배경 이해, 예비 탐색적 데이터 분석, 데이터 정리(Data Cleaning), 심층 탐색적 데이터 분석, 특성 공학(Feature Engineering), 모델 구축(Model Building)입니다. 각 단계에서는 독립적인 5개의 에이전트가 협력하여 작업을 수행합니다. 또한, AutoKaggle은 코드의 문법적 정확성과 논리적 일관성을 검증하는 iterative debugging과 unit testing을 통해 코드 품질을 보장합니다.

- **Performance Highlights**: 평가한 8개의 Kaggle 데이터 과학 대회에서 AutoKaggle은 0.85의 유효 제출 비율과 0.82의 종합 점수를 달성하여 데이터 과학 작업을 처리하는 데 있어 효과성과 실용성을 입증했습니다.



### Open-Vocabulary Object Detection via Language Hierarchy (https://arxiv.org/abs/2410.20371)
Comments:
          NeurIPS 2024 Camera Ready

- **What's New**: 최근 약한 감독(imagen-level supervision)을 활용한 일반화 가능한 객체 탐지(generalizable object detection)에 대한 연구가 증가하고 있습니다. 본 논문에서는 Language Hierarchical Self-training (LHST)을 통해 약하게 감독된 탐지기 훈련에 언어 계층을 도입하여 더 일반화된 탐지기를 학습합니다. 이를 통해 이미지 수준의 레이블과 박스 수준의 레이블 간의 불일치를 완화하고, 더 풍부한 감독(supervision)을 제공합니다.

- **Technical Details**: LHST는 WordNet의 언어 계층을 활용하여 이미지 수준의 레이블을 확장하고, 자기 훈련(self-training) 과정에서의 공동 정규화(co-regularization)를 가능하게 합니다. 또한, Language Hierarchical Prompt Generation (LHPG)을 설계하여 샘플링된 언어 계층을 프롬프트 생성에 적용합니다. 이는 훈련과 테스트 간의 어휘 격차(vocabulary gaps)를 메우는 데 도움을 줍니다.

- **Performance Highlights**: 제안된 기술들은 14개의 널리 연구된 객체 탐지 데이터셋에서 지속적으로 우수한 일반화 성능(generalization performance)을 보여줍니다. DetLH는 약한 감독 하의 객체 탐지에서 스스로 상기(label-to-box assignment) 및 자율 학습(self-training) 기법을 결합하여 감지 성능을 극대화합니다.



### Historical Test-time Prompt Tuning for Vision Foundation Models (https://arxiv.org/abs/2410.20346)
Comments:
          NeurIPS 2024 Camera Ready

- **What's New**: 이 논문에서는 HisTPT(History Test-time Prompt Tuning)라는 새로운 기술을 제안합니다. 이는 테스트 샘플로부터 지속적으로 학습한 정보를 기억하고, 이를 기반으로 강력한 테스트 시 프롬프트 튜닝을 가능하게 합니다.

- **Technical Details**: HisTPT는 세 가지 유형의 지식 은행(local knowledge bank, hard-sample knowledge bank, global knowledge bank)을 도입하며, 각 은행은 다른 메커니즘을 통해 지식을 효과적으로 기억하고 프롬프트를 최적화합니다. 또한, 적응형 지식 검색 메커니즘을 포함하여 각 테스트 샘플의 예측 정규화를 수행합니다.

- **Performance Highlights**: HisTPT는 이미지 분류, 의미 분할, 객체 탐지와 같은 다양한 시각적 인식 작업에서 우수한 프롬프트 튜닝 성능을 보여주며, 도메인이 지속적으로 변할 때도 일관된 성과를 보입니다.



### Accelerating Direct Preference Optimization with Prefix Sharing (https://arxiv.org/abs/2410.20305)
Comments:
          To appear in NeurIPS 2024 in the Fine-Tuning in Machine Learning Workshop

- **What's New**: 이번 논문에서는 preference tuning을 위한 prefix sharing이라는 새로운 기술을 도입합니다. 이는 선택된(response) 및 거부된(rejected) 응답을 공유 프리픽스와 함께 하나의 시퀀스로 처리하여 계산의 중복을 줄이는 방식입니다.

- **Technical Details**: 제안된 방법에서는 맞춤 attention mask를 사용하여 두 응답 간의 contamination을 방지합니다. 이를 통해 학습 속도를 크게 향상시키며, DPO(Direct Preference Optimization)와 같은 인기 있는 방법에 적용 가능합니다. 또한, prefix 공유와 시퀀스 패킹을 결합하여 강력한 성능 향상을 실현합니다.

- **Performance Highlights**: 이 방식은 DPO 데이터셋에서 1.1배에서 1.5배의 훈련 처리량 개선을 달성했으며, 추가로 시퀀스 패킹을 적용했을 때 1.3배에서 1.6배의 일관된 속도 향상을 보였습니다.



### Sequential Large Language Model-Based Hyper-Parameter Optimization (https://arxiv.org/abs/2410.20302)
- **What's New**: 이번 연구에서는 하이퍼파라미터 최적화(hyperparameter optimization, HPO)를 위해 대규모 언어 모델(large language models, LLMs)을 활용하는 혁신적인 프레임워크인 SLLMBO를 소개합니다. SLLMBO는 동적 탐색 공간 적응(dynamic search space adaptability), 향상된 파라미터 환경 활용(parameter landscape exploitation), 그리고 새로운 하이브리드 LLM-트리 구조 파젠 추정기(LLM-Tree-structured Parzen Estimator, LLM-TPE) 샘플러를 통합하고 있습니다.

- **Technical Details**: SLLMBO는 최근의 완전 LLM 기반 방법들과 전통적인 베이지안 최적화(Bayesian Optimization, BO)의 한계를 극복하여 보다 강력한 최적화를 달성합니다. 이번 연구에는 GPT-3.5-turbo, GPT-4o, Claude-Sonnet-3.5, Gemini-1.5-flash를 포함한 여러 LLM에 대한 포괄적인 벤치마킹이 이루어졌으며, 이는 HPO를 위한 다양한 LLM을 평가한 최초의 프레임워크로 자리잡고 있습니다.

- **Performance Highlights**: 14개의 표 형식(tabular) 작업에서 LLM-TPE 샘플러는 완전 LLM 기반 방법보다 뛰어난 성과를 나타내었으며, 9개 작업에서 BO 방법보다 우수한 결과를 달성했습니다. 예산 제약이 있는 시나리오에서의 조기 중단 테스트는 경쟁력 있는 성능을 추가로 입증하여 LLM 기반 방법들이 최적 결과를 위해서는 반복(iteration)을 연장해야 함을 보여주었습니다.



### Library Learning Doesn't: The Curious Case of the Single-Use "Library" (https://arxiv.org/abs/2410.20274)
Comments:
          24 pages, 7 figures. Accepted to the 4th MATH-AI Workshop at NeurIPS'24

- **What's New**: 이번 연구에서 대형 언어 모델(LLM) 기반의 수학 도구 학습 시스템인 LEGO-Prover와 TroVE를 분석하여 기존의 성능 향상이 실제로 재사용 가능한 도구의 학습이 아닌 다른 메커니즘에 의해 촉진되고 있음을 발견했습니다. 이러한 발견은 재사용의 유용성에 관한 기존 가정에 도전합니다.

- **Technical Details**: LEGO-Prover는 리사이클 가능한 정리 정식 이론을 학습하는 것을 목표로 하며, miniF2F 데이터셋을 사용하여 평가되었습니다. TroVE는 리사이클 가능한 Python 함수를 학습하여 프로그램 문제를 해결하는 데 중점을 둡니다. 두 시스템 모두에서 함수 재사용은 드물며, 성능 향상은 주로 자기 수정(self-correction) 및 자기 일관성(self-consistency)에 기인함을 보입니다.

- **Performance Highlights**: LEGO-Prover는 20,000개 이상의 정리를 학습했으나, 최종 해결 단계에서 실제로 재사용된 정리는 6%에 불과하며, 오직 하나의 정리만이 재사용되었습니다. 연구 결과는 두 시스템 모두에서 기능 재사용의 효과가 미미하며, 오히려 자기 수정 메커니즘이 주요 성능 향상 요인임을 시사합니다.



### Enhancing Inflation Nowcasting with LLM: Sentiment Analysis on News (https://arxiv.org/abs/2410.20198)
- **What's New**: 본 연구에서는 고물가 변동성이 큰 시기(예: COVID-19 팬데믹)를 고려하여 대규모 언어 모델(LLMs)을 전통적인 물가 현재 예측(framework) 모델에 통합하는 방법을 탐구하였다. 이 연구의 핵심은 InflaBERT라는 BERT 기반의 LLM을 도입하여 뉴스에서 물가 관련 감정을 예측하는 것이다.

- **Technical Details**: InflaBERT는 뉴스의 물가 감정을 예측하기 위해 조정된 BERT(Bidirectional Encoder Representations from Transformers) 모델이다. 이를 통해 생성된 NEWS 지수는 물가 관련 뉴스의 월간 감정을 포착하는 지표로 사용된다. Cleveland Fed 모델에 우리 지수를 통합함으로써 코로나19 기간 동안 물가 예측 정확성이 약간 개선되었음을 보여준다.

- **Performance Highlights**: 이 연구의 결과는 AI 기반의 감정 분석과 전통적인 경제 지표를 결합함으로써 실시간 물가 모니터링에 대한 가능성을 보여준다. 향후 연구에서는 이러한 방법론을 더욱 세련되게 발전시키기 위한 방향으로 나아갈 것을 제안한다.



### LLMs Can Evolve Continually on Modality for X-Modal Reasoning (https://arxiv.org/abs/2410.20178)
- **What's New**: PathWeave는 Modal-Path 전환 및 확장 기능을 갖춘 유연하고 확장 가능한 프레임워크로, Multimodal Large Language Models (MLLMs)의 연속적인 모달 진화를 가능하게 합니다. 이를 통해 단일 모달 데이터로 새로운 모달리티에 대한 확장과 학습이 가능합니다.

- **Technical Details**: PathWeave는 Incremental Training Strategy를 바탕으로 하여, Uni-modal 및 Cross-modal Adapter를 통합하여 효율적인 모달 정렬 및 협업을 촉진합니다. MoE (Mixture of Experts) 기반의 게이팅 모듈을 통해 두 가지 유형의 어댑터 간의 멀티모달 상호작용을 향상시킵니다.

- **Performance Highlights**: PathWeave는 최신의 MLLMs와 비교하여 성능 면에서 동등하며, 파라미터 학습 부담을 98.73% 감소시키는 동시에, Continual Learning of Modality (MCL) 벤치마크에서의 높은 정확도를 보여줍니다.



### UniHGKR: Unified Instruction-aware Heterogeneous Knowledge Retrievers (https://arxiv.org/abs/2410.20163)
- **What's New**: 새로운 연구에서는 UniHGKR라는 통합된 이질적 지식 검색기(knowledge retriever)를 소개하여, 이 모델이 여러 종류의 지식 출처와 사용자 지시에 따라 검색할 수 있는 능력을 가지고 있음을 보여줍니다. 기존의 정보 검색 모델은 동질적인 구조만을 가정하여 실제 세계의 다양한 요구를 충족하지 못했습니다.

- **Technical Details**: UniHGKR는 세 가지 주요 단계로 이루어져 있습니다: (1) 이질적 자기 지도(pretraining); (2) 텍스트 기반 임베딩 정렬(embedding alignment); (3) 지시 인식 검색기(fine-tuning)로, 이러한 단계들을 통해 다양한 검색 환경에서 일반화할 수 있는 능력을 갖추게 됩니다. 또한, 이 모델은 BERT 기반 버전과 대규모 언어 모델로 훈련된 UniHGKR-7B 버전이 있습니다.

- **Performance Highlights**: UniHGKR는 CompMix-IR 벤치마크에서 최신 기술의 방법들과 비교하여, 각각 6.36% 및 54.23%의 상대적 향상을 보였으며, ConvMix 과제에서 새로운 최첨단(record-breaking) 결과를 달성하여 최대 4.80점의 절대 향상을 이루었습니다.



### Causal Abstraction in Model Interpretability: A Compact Survey (https://arxiv.org/abs/2410.20161)
- **What's New**: 최근 인공지능(AI)의 해석 가능성을 위한 연구가 진행되고 있으며, 그 중에서도 causal abstraction(인과적 추상화)라는 이론적框架(프레임워크)가 주목받고 있습니다. 이 방법은 모델의 행동 뒤에 있는 인과적 메커니즘을 이해하고 설명하기 위한 체계적인 접근법을 제공합니다. 이 논문은 causal abstraction의 이론적 기초, 실제 응용 및 모델 해석 가능성에 대한 함의를 탐구합니다.

- **Technical Details**: Causal abstraction은 구조적 방정식 모델을 기반으로 하여 변수들 간의 인과 관계를 표현하는 함수 집합을 사용합니다. 이 연구는 모델 간 인과적 동등성을 판단하기 위한 수학적 기본을 제공하며, 최근 연구에서 neural networks(신경망 모델)에 대한 인과적 추상화 기술을 적용하여 내부 벡터와 모델 출력 행동 간의 잠재적 인과 관계를 분석합니다.

- **Performance Highlights**: Causal abstraction은 복잡한 머신러닝 모델의 해석을 향상시키고, 모델의 결정 과정을 보다 투명하고 책임감 있게 설명할 수 있도록 도와줍니다. 이 접근법은 기계적 해석 가능성(mechanistic interpretability)에 대한 관심이 고조되는 가운데, 모델의 내부 작동을 이해하는 데 기여하고 있습니다.



### Multi-Field Adaptive Retrieva (https://arxiv.org/abs/2410.20056)
- **What's New**: 이번 연구에서는 Multi-Field Adaptive Retrieval (MFAR)라는 새로운 프레임워크를 소개합니다. MFAR는 문서의 구조적 데이터를 활용하여 조회 성능을 지속적으로 향상시키도록 설계되었습니다. 이는 여러 필드(예: 제목, 본문 등)를 가진 문서 인덱스를 독립적으로 처리할 수 있는 유연함을 제공합니다.

- **Technical Details**: MFAR의 두 가지 주요 단계는 (1) 기존 문서를 필드로 분해하고 이를 밀집(dense) 및 어휘(lexical) 방법을 통해 독립적으로 인덱싱하며, (2) 문서 쿼리에 조건을 두어 필드의 중요성을 적응적으로 예측하는 모델을 학습하는 것입니다. 또한 필드 간의 동적 가중치를 조정하여, 쿼리에 가장 관련 있는 필드를 선택할 수 있도록 합니다.

- **Performance Highlights**: MFAR는 기존의 다양한 검색 시스템들보다 개선된 문서 순위 성능을 보이며, STaRK라는 구조화된 문서 검색 데이터셋에서 최신 성능을 달성했습니다. 또한, MFAR는 사전 학습(pretraining) 과정 없이도 테스트 시 필드의 가용성을 조절 가능하여 사용자가 결과를 제어할 수 있는 장점을 제공합니다.



### LinBridge: A Learnable Framework for Interpreting Nonlinear Neural Encoding Models (https://arxiv.org/abs/2410.20053)
Comments:
          9 pages of main text, 23 pages total, submitted to ICLR 2025 and currently under review

- **What's New**: 이 논문에서는 비선형 인코딩 모델을 해석하기 위한 LinBridge라는 새롭고 유연한 프레임워크를 제안합니다. 이 프레임워크는 Jacobian 분석을 기반으로 하여 비선형 인코딩 모델을 이해하는 데 도움을 줍니다.

- **Technical Details**: LinBridge는 인공 신경망(ANNs)의 계산적 표현과 신경 반응 간의 비선형 매핑을 두 가지 구성 요소로 분해합니다: 복잡한 비선형 동역학을 근사하는 선형 고유 구성 요소와 샘플 선택 비선형성을 포착하는 매핑 바이어스. LinBridge는 자기 감독 학습(self-supervised learning) 전략을 사용하여 테스트 셋의 Jacobian 매트릭스에서 이 두 구성 요소를 추출합니다.

- **Performance Highlights**: 실험 결과 LinBridge가 추출한 선형 고유 구성 요소는 비선형 신경 인코딩 모델의 복잡한 매핑을 정확하게 반영하며, 샘플 선택 매핑 바이어스는 시각 처리 계층의 다양한 수준에서 비선형성의 변동성을 설명합니다.



### Training the Untrainable: Introducing Inductive Bias via Representational Alignmen (https://arxiv.org/abs/2410.20035)
Comments:
          Under Review; 24 pages, 9 figures; Project page and code is at this https URL

- **What's New**: 이번 논문에서는 기존에는 훈련이 어려운 것으로 여겨졌던 아키텍처들이 다른 아키텍처로부터의 inductive bias를 활용하여 훈련될 수 있다는 점을 보여줍니다. ‘Guidance’라는 새로운 기법을 도입하여, 가이드 네트워크가 타겟 네트워크를 유도하며, 이를 통해 타겟 네트워크의 성능을 크게 향상시킬 수 있음이 입증되었습니다.

- **Technical Details**: 이 방법론은 두 네트워크 간의 representational similarity를 맞추는 것을 목표로 하며, neural distance function을 사용하여 층 별로 표현을 일치시키는 방식입니다. 연구자들은 다양한 아키텍처에 대해 이 이론을 적용했으며, 가이드 네트워크가 훈련되지 않은 경우에도 여전히 부분적으로 architectural prior를 전파할 수 있음을 발견했습니다.

- **Performance Highlights**: 이 방법을 활용하여, 일반적인 fully connected 네트워크가 비전 작업에서 즉각적인 overfitting을 극복하게 되었고, plain CNN이 ResNet에 버금가는 성능을 발휘하게 되었으며, RNN과 Transformers 간의 성능 격차가 줄어드는 등의 개선이 이루어졌습니다.



### Cooperative Strategic Planning Enhances Reasoning Capabilities in Large Language Models (https://arxiv.org/abs/2410.20007)
Comments:
          Working in progress

- **What's New**: 이번 논문에서는 대규모 언어 모델(LLMs)의 추론 능력을 향상시키기 위해 협력적인 다중 에이전트 추론 프레임워크인 CoPlanner를 제안합니다. CoPlanner는 추론 단계를 분리하고, 각 에이전트에게 고유한 역할을 부여하여 문제를 해결합니다.

- **Technical Details**: CoPlanner는 두 개의 LLM 에이전트로 구성됩니다: 계획 에이전트(planning agent)와 추론 에이전트(reasoning agent). 계획 에이전트는 고차원 전략 힌트를 제공하고, 추론 에이전트는 이 힌트를 따르며 답변을 추론합니다. 이 과정은 Proximal Policy Optimization (PPO)을 통해 훈련되며, 각 에이전트는 명확한 역할을 수행하여 서로의 강점을 활용하는 구조로 되어 있습니다.

- **Performance Highlights**: LLaMA-3-8B 기반의 CoPlanner는 LogiQA 기준에서 기존 최고의 방법보다 9.94% 더 우수한 성능을 보였으며, BBH에서 3.09% 향상된 결과를 기록했습니다. 계획 에이전트의 가이드와 에이전트 간의 효과적인 협력이 CoPlanner의 우수한 성능에 기여했음을 보여줍니다.



### Evaluating Cost-Accuracy Trade-offs in Multimodal Search Relevance Judgements (https://arxiv.org/abs/2410.19974)
- **What's New**: 이 논문에서는 다양한 LLMs와 MLLMs의 성능을 평가하고, 특정 사용 사례에 따라 모델 성능이 어떻게 달라지는지를 분석하였습니다. 특히, 시각적 요소가 포함된 모델이 항상 성능을 향상시키지 않음을 밝혀내, 모델 선택의 복잡성을 강조합니다.

- **Technical Details**: 모델 평가 과정은 데이터 수집, 인간 주석, 모델 평가의 세 단계로 구성됩니다. 이 연구는 패션, 호텔 용품, 디자인과 같은 세 가지 데이터셋을 사용하였으며, 각 데이터셋은 다양한 특성과 관련 이미지를 포함합니다. 인간 주석자와 LLM을 사용하여 중요성을 평가하고, Cohen's kappa 계수를 통해 주석자 간의 일치를 측정합니다.

- **Performance Highlights**: 모델의 성능은 맥락에 크게 의존하며, LLM의 비용-정확도 트레이드오프에 대한 추가 연구가 필요합니다. 특히, 작은 규모의 모델에서는 시각적 구성 요소가 성능을 저하시킬 수 있음을 확인하였습니다.



### RobustKV: Defending Large Language Models against Jailbreak Attacks via KV Eviction (https://arxiv.org/abs/2410.19937)
- **What's New**: 본 논문에서는 기존의 방어 방법들이 치명적인 공격인 jailbreak 공격을 충분히 방어하지 못하는 상황에서 새로운 방어 기법인 RobustKV를 제안합니다. 기존의 방어책은 주로 jailbreak 프롬프트의 영향을 감소시키는데 집중하고 있지만, RobustKV는 키-값 캐시(key-value cache)의 해로운 쿼리의 중요한 토큰을 선택적으로 제거함으로써 다른 접근 방식을 취합니다.

- **Technical Details**: RobustKV는 jailbreak 프롬프트의 효과를 위해 필요로 하는 토큰의 중요성을 측정하기 위해 attention scores를 활용합니다. 이 방식으로 중요도가 낮은 토큰의 KVs를 전략적으로 제거하여KV 캐시에서 해로운 쿼리의 존재를 감소시킵니다. 따라서 이 기법은 해로운 응답 생성을 예방하도록 설계되었습니다.

- **Performance Highlights**: RobustKV는 다양한 벤치마크 데이터셋과 모델을 사용한 광범위한 평가를 통해 최신 jailbreak 공격을 효과적으로 저지하면서도 무해한 쿼리에 대한 LLM의 일반 성능을 유지하는 데 성공했습니다. 또한, RobustKV는 적대자들에게 흥미로운 회피 딜레마를 만들어내어, RobustKV를 회피하는 것과 LLM의 내장된 안전 장치를 우회하는 것 사이에서 균형을 잡도록 강요합니다. 이러한 균형 추구는 RobustKV의 적응형 공격에 대한 강인함을 강조합니다.



### Benchmarking Large Language Models for Image Classification of Marine Mammals (https://arxiv.org/abs/2410.19848)
Comments:
          ICKG 2024

- **What's New**: 이번 연구는 해양 포유류에 초점을 맞춘 새로운 벤치마크 데이터셋을 소개합니다. 데이터셋에는 65종의 해양 포유류에 대한 1,423개의 이미지가 포함되어 있으며, 생물학적 수준에 따라 분류된 레이블이 제공됩니다.

- **Technical Details**: 우리는 (1) 신경망이 제공하는 임베딩을 사용하는 기계 학습 (ML) 알고리즘, (2) 영향력 있는 사전 훈련된 신경망, (3) 제로샷 모델인 CLIP 및 LLM, (4) LLM 기반의 다중 에이전트 시스템 (MAS)을 평가했습니다. 이러한 접근 방식을 통해 해양 포유류의 분류 성능을 향상시킬 수 있음을 보여주었습니다.

- **Performance Highlights**: 전통적인 모델과 LLM의 성능을 비교한 결과, MAS는 다양한 분류 레벨에서 단일 LLM보다 더 높은 성능을 보였습니다. 이 새로운 데이터셋과 기술은 해양 포유류 연구와 보존 전략 개발에 기여할 것으로 기대됩니다.



### Step Guided Reasoning: Improving Mathematical Reasoning using Guidance Generation and Step Reasoning (https://arxiv.org/abs/2410.19817)
Comments:
          4 pages, 4 figures

- **What's New**: 이 논문에서는 기존의 Chain-of-Thought (CoT) 방법론을 넘어서는 새로운 접근 방식인 Step Guidance Reasoning (SGR)을 소개합니다. SGR은 모델의 Fine-tuning 없이 단계별 추론 과정을 시각화하고, 이러한 단계를 통해 수학 문제 해결의 정확성을 크게 향상시킵니다.

- **Technical Details**: SGR은 각 추론 단계에서 모델이 스스로 다음 해야 할 일을 질문하고 대답하게 하며, 이 성찰(reflective process)을 통해 다음 단계로 나아가는 과정을 안내합니다. 본 방법은 추론 단계 동안 300에서 500개의 토큰을 사용하는 다양한 단계 길이(constraints)를 설정합니다.

- **Performance Highlights**: AMC23 데이터셋에서 정확도가 30%에서 57.5%로 상승하며 91.7%의 상대적 개선을 이루었고, MATH 데이터셋의 5단계 문제에서 43%에서 67%로 증가하여 55.8%의 상대적 정확도 개선을 기록했습니다.



### Ethics Whitepaper: Whitepaper on Ethical Research into Large Language Models (https://arxiv.org/abs/2410.19812)
Comments:
          47 pages

- **What's New**: 이 백서(whitepaper)는 대규모 언어 모델(LLMs)의 연구와 관련된 윤리적 고려사항을 종합적으로 정리합니다.

- **Technical Details**: LLMs의 사용이 증가함에 따라 그 사회적 영향력이 커지고 있으며, 이에 따라 윤리적 질문들이 제기되고 있습니다. 이 백서는 LLMs의 개발, 배포, 사용에 있어 윤리를 고려한 연구 문헌을 기반으로 하여 최선의 실천 방안(best practices)을 제시합니다.

- **Performance Highlights**: 연구자와 산업 종사자들이 자신의 작업에서 최고의 윤리적 기준을 유지할 수 있도록 돕는 것을 목표로 하고 있습니다.



### ControlAgent: Automating Control System Design via Novel Integration of LLM Agents and Domain Expertis (https://arxiv.org/abs/2410.19811)
- **What's New**: ControlAgent는 기존의 인공지능 모델에 비해 제어 시스템 디자인의 자동화를 위한 혁신적인 접근 방식을 제공합니다. 이를 통해 전문적인 지식과 반복적 디자인 프로세스를 융합하여 인간의 개입 없이도 고품질의 제어 시스템을 설계할 수 있습니다.

- **Technical Details**: ControlAgent는 여러 개의 협력적인 LLM(대형 언어 모델) 에이전트를 통합합니다. 중앙 에이전트는 작업 분배를 담당하고, 각 작업에 특정한 에이전트는 다양한 시스템 및 요구사항에 대해 세부 제어 설계를 맡습니다. Python 기반의 계산 에이전트가 복잡한 계산을 수행하며, 이 모든 과정은 이전 디자인으로부터의 실시간 피드백을 통해 반복적으로 개선됩니다.

- **Performance Highlights**: ControlAgent는 전통적인 인간 개입 기반 디자인 방식에 비해 높은 성능과 견고성을 입증했습니다. 500개의 제어 과제를 포함한 ControlEval 데이터셋을 통해 테스트된 결과, ControlAgent가 LLM 기반 및 전통적인 도구 기반 방법들보다 우수하다는 것이 확인되었습니다.



### First-Person Fairness in Chatbots (https://arxiv.org/abs/2410.19803)
- **What's New**: 본 논문은 일반적인 채팅봇 사용의 공정성에 중점을 두고, 특히 사용자의 이름에 따른 편향을 분석합니다. "first-person fairness"라는 개념을 통해 사용자가 직접 겪는 공정성을 연구하며, 다양한 배경을 가진 사용자를 위한 질 높은 반응 제공을 목표로 합니다.

- **Technical Details**: 본 연구에서는 LLMRA(언어 모델 연구 보조자)를 활용하여 이름에 대한 민감성을 분석하고, 사용자 이름과 관련된 편향을 비공식적으로 평가합니다. 이를 위해 복잡한 패턴을 식별하고, 공공 및 개인 채팅 데이터를 통해 차별성을 확인하는 소프트웨어 시스템을 개발했습니다. 인공지능 태그와 결과를 독립적인 인간 평가와 비교하여 신뢰성을 확보했습니다.

- **Performance Highlights**: 채팅봇 응답에서 특정 사용자 이름에 따른 반응 차이를 보여줍니다. 예를 들어, 여성과 관련된 이름을 가진 사용자는 약간 더 친절하고 단순한 언어로 응답을 받는 경향이 있으며, 성별에 따라 주인공을 자주 고르는 패턴도 나타납니다. 이후의 훈련 개입(RL 포함)은 해로운 고정관념을 상당히 완화하는 것으로 나타났습니다.



### Telco-DPR: A Hybrid Dataset for Evaluating Retrieval Models of 3GPP Technical Specifications (https://arxiv.org/abs/2410.19790)
- **What's New**: 이 논문에서는 3GPP(3rd Generation Partnership Project) 기술 문서를 활용한 통신 분야의 Q&A 시스템을 제안합니다. 또한, 텍스트와 테이블을 혼합한 형태의 curated 3GPP 코퍼스를 포함하는 혼합 데이터셋인 Telco-DPR를 소개합니다.

- **Technical Details**: 제안된 QA 시스템은 RAG(Retriever-Augmented Generation) 기술을 사용하며, DHR(Dense Hierarchical Retrieval) 모델로 계층형 패시지를 선택하여 문서와 패시지 수준에서 미세 조정을 통해 관련 정보를 추출합니다. 주요 평가 지표로는 Top-K 정확도와 Mean Reciprocal Rank (MRR)를 사용하여 성능을 검증합니다.

- **Performance Highlights**: DHR 모델은 관련 기술 정보를 검색하는 데 있어 기존 방법보다 우수한 성과를 나타내며, Top-10 정확도 86.2%를 달성하였습니다. RAG 모델과 GPT-4를 활용한 QA 시스템은 기존 데이터셋에 비해 14%의 답변 정확도 개선을 보였습니다.



### Author Unknown: Evaluating Performance of Author Extraction Libraries on Global Online News Articles (https://arxiv.org/abs/2410.19771)
- **What's New**: 최근 온라인 뉴스 콘텐츠의 저자 추출 성능을 평가하기 위해 715명의 저자와 754개의 뉴스 기사를 포함하는 다국적 데이터셋을 수집하였습니다. 이 연구에서는 다섯 개의 기존 소프트웨어 패키지와 하나의 맞춤형 모델을 비교하고, Go-readability와 Trafilatura가 가장 일관된 솔루션임을 발견하였습니다.

- **Technical Details**: 연구에서는 HTML 기반 뉴스 기사의 저자를 식별하기 위한 다양한 방법론을 비교합니다. 저자 정보 추출을 위한 기존 방법은 대체로 ML 방법, 휴리스틱 방법, 그리고 이 두 가지를 조합한 방법의 세 가지 범주로 나누어집니다. 본 연구에서는 10개 언어로 구성된 새로운 핸드코드 크로스링구얼 데이터셋을 사용하는데, 이는 기존의 소프트웨어 패키지들의 성능을 유의미하게 평가하는 데 기여합니다.

- **Performance Highlights**: 모든 패키지들이 언어에 따라 높은 변동성을 보였으며, 저자 데이터 활용 시 추가적인 언어 및 지역에 대한 검증이 필요하다는 점이 강조됩니다. Go-readability와 Trafilatura는 상대적으로 일관된 성능을 보여 연구자들이 더 나은 결과를 얻기 위해 이러한 도구를 활용할 수 있는 기반을 마련하였습니다.



### A Comparative Analysis on Ethical Benchmarking in Large Language Models (https://arxiv.org/abs/2410.19753)
Comments:
          62 pages

- **What's New**: 이번 연구는 지능형 시스템이 인간의 가치관을 정확하게 표현하고 그에 맞게 행동하는지를 평가하기 위한 Machine Ethics (ME) 벤치마킹 분야에 기여합니다. 기존 ME 벤치마크의 세 가지 주요 문제점을 제시합니다: 비현실적인 윤리적 딜레마로 인한 제한된 생태적 타당성, 명확한 포함/제외 기준이 없는 비구조적 질문 생성, 인간 주석에 의존하는 확장성 부족입니다. 이를 해결하기 위해 Triage Benchmark와 Medical Law (MedLaw) Benchmark라는 두 개의 새로운 ME 벤치마크를 소개합니다.

- **Technical Details**: MedLaw Benchmark는 완전히 AI가 생성한 벤치마크로, 의료 분야에서의 실제 윤리적 딜레마를 다룹니다. 또한, 모델의 최악의 경우 성능을 평가하기 위해 컨텍스트 변화를 도입했습니다. 연구 결과, 윤리적 프롬프트가 항상 의사결정을 개선하지 않으며, 컨텍스트 변화가 모델 성능을 크게 감소시키거나 오류 패턴을 역전시키고 상대적 성능 순위를 변화시킬 수 있다는 것을 발견했습니다.

- **Performance Highlights**: 우리의 분석에 따르면, 일반 모델 능력이 항상 강력한 윤리적 의사결정을 예측하지는 않는 것으로 나타났습니다. 우리는 ME 벤치마크가 견고한 평가를 보장하기 위해 실제 시나리오와 최악의 경우 성능을 근사해야 한다고 주장합니다.



### Non-myopic Generation of Language Models for Reasoning and Planning (https://arxiv.org/abs/2410.17195)
- **What's New**: 이 논문은 LLM(대규모 언어 모델)의 사고 방식과 계획 능력을 최적 제어(Optimal Control) 관점에서 재조명합니다.

- **Technical Details**: 제안된 방법인 Predictive-Decoding은 모델 예측 제어(Model Predictive Control)를 활용하여 계획 정확도를 향상시키고, LLM 분포를 예측 궤적(Foresight Trajectories)에 따라 재조정하여 초기 오류를 줄이고 비근시적(Myopic) 계획을 촉진합니다.

- **Performance Highlights**: 예험 결과, 수학 문제 해결, 코딩, 그리고 에이전트 tasks에서 상당한 성능 향상을 보여주며, 제안된 방법은 낮은 계산 비용으로 검색 기준(Search Baselines)보다 우수한 성능을 보여줍니다.



New uploads on arXiv(cs.IR)

### Zero-Shot Dense Retrieval with Embeddings from Relevance Feedback (https://arxiv.org/abs/2410.21242)
- **What's New**: 이 논문에서는 ReDE-RF(Real Document Embeddings from Relevance Feedback)라는 새로운 접근 방식을 도입하여, LLM을 활용한 가상의 문서 생성 대신 관련성 추정(task)을 프레임으로 삼아 검색 효율성을 크게 향상시킵니다.

- **Technical Details**: ReDE-RF는 먼저 완전 비지도 하이브리드 스파스-밀집 검색 시스템에서 초기 문서 집합을 추출한 후, LLM을 통해 반환된 문서가 관련성이 있는지 없는지를 평가합니다. 이후, 관련 문서 집합을 기반으로 미리 계산된 문서 임베딩을 효율적으로 가져와 쿼리 벡터를 업데이트합니다. 이는 LLM이 도메인 특화 지식에 의존하지 않고 단순히 관련성을 판단하게 합니다.

- **Performance Highlights**: 실험 결과, ReDE-RF는 여러 저자원 검색 데이터셋에서 기존의 최첨단 제로샷 밀집 검색 방법들보다 최대 14% 더 우수한 성능을 보였으며, 검색 지연(lag)을 7.5배에서 11.2배까지 줄이는 성과를 나타냈습니다.



### Pay Attention to Attention for Sequential Recommendation (https://arxiv.org/abs/2410.21048)
Comments:
          Accepted at RecSys 2024

- **What's New**: 본 연구에서는 Attention Weight Refinement (AWRSR)이라는 새로운 순차 추천(sequential recommendation) 방법론을 제안합니다. 기존의 Self-Attention 모델이 순차 추천에서 복잡한 의존성을 충분히 캡처하지 못한다는 문제를 해결하고자 합니다.

- **Technical Details**: AWRSR은 주목 가중치(attention weights)를 세부적으로 분석하여 이들 간의 상관관계를 고려함으로써 자기 주목(self-attention)의 효과를 강화합니다. 전통적인 자기 주목 아키텍처에서는 주목 행렬을 값에 곱하는 방식으로 사용되지만, AWRSR은 학습 가능한 행렬을 사용하여 주목 가중치를 새로운 공간으로 변환하여 더 높은 차원의 주목 가중치를 계산합니다.

- **Performance Highlights**: 다양한 실세계 데이터 세트에서 종합적인 실험을 통해 AWRSR이 최신 SR 모델들을 일관되게 초월하는 성능을 보였으며, 주목 가중치의 효과적인 분석이 높은 차원의 의존성 캡처에 기여한다는 것이 입증되었습니다.



### Challenges in Implementing a Recommender System for Historical Research in the Humanities (https://arxiv.org/abs/2410.20909)
Comments:
          Presented at AltRecSys 2024: The First Workshop on Alternative, Unexpected, and Critical Ideas in Recommendation, October 18, 2024, co-located with the ACM Conference on Recommender Systems 2024 (RecSys 2024), Bari, Italy

- **What's New**: 이 확장 초록은 인문학 분야의 디지털 아카이브에 추천 시스템을 구현하는 데 있어의 과제를 설명합니다. 특히 Monasterium.net 플랫폼을 중심으로, 추천 항목으로서의 charters의 독특한 특성과 여러 이해관계자의 복잡한 환경, 그리고 인문학 연구자의 정보 탐색 행동을 논의합니다.

- **Technical Details**: 추천 시스템(RecSys)의 전통적인 알고리즘은 보통 다차원적이고 희소한 메타데이터를 처리하는 데 어려움을 겪습니다. Monasterium.net의 charters는 역사가 및 다른 인문학 연구자들이 필요로 하는 다양한 메타데이터를 포함하고 있으며, 이들은 다수의 언어 및 서체로 작성된 텍스트를 포함합니다. 또한, charters의 역사적 맥락은 시간의 적합성을 고려해야 하는 복잡성을 생성합니다.

- **Performance Highlights**: 추천 시스템을 통해 Monasterium.net과 같은 디지털 아카이브의 유용성을 향상시키고, 역사 연구에서 새로운 발견과 더 깊은 이해를 가능하게 할 수 있습니다. 이 연구는 인문학 플랫폼과 문화유산 기관에도 적용 가능한 통찰력을 제공합니다.



### RecFlow: An Industrial Full Flow Recommendation Datas (https://arxiv.org/abs/2410.20868)
- **What's New**: 본 논문에서는 새롭게 제안된 RecFlow 데이터셋을 소개합니다. 이는 기존의 추천 시스템(RS) 벤치마크 데이터셋의 한계를 극복하고, 실제 산업 환경에서의 추천 성능 향상을 위한 새로운 기준점을 제공합니다.

- **Technical Details**: RecFlow 데이터셋은 38M의 상호작용 데이터를 포함하며, 42K 사용자와 9M 품목으로부터 수집된 데이터로 구성되어 있습니다. 이 데이터셋은 37일간 9.3M 온라인 요청에서 수집된 추가 1.9B 단계 샘플을 포함하여, 추천 시스템의 각 단계에서 필터링된 노출되지 않은 품목 샘플을 포함합니다.

- **Performance Highlights**: RecFlow 데이터셋을 활용한 실험은 추천 시스템의 효과성을 향상시키기 위해 여러 알고리즘 디자인에서 가능성을 보여주었습니다. 이미 몇 가지 알고리즘은 온라인에 배포되어, 지속적인 성능 향상을 이루었습니다.



### Beyond Positive History: Re-ranking with List-level Hybrid Feedback (https://arxiv.org/abs/2410.20778)
- **What's New**: 본 논문에서는 사용자 선호도를 더 잘 반영하기 위해 아이템 수준의 긍정 피드백 대신 리스트 수준의 하이브리드 피드백을 적절히 활용하는 새로운 재순위화(re-ranking) 기법, RELIFE를 제안합니다.

- **Technical Details**: RELIFE는 사용자 선호도와 행동 패턴을 세 가지 모듈로 캡처합니다: 1) Disentangled Interest Miner: 사용자 선호도를 관심과 비관심으로 분리합니다. 2) Sequential Preference Mixer: 피드백 컨텍스트를 고려하여 얽힌 사용자 선호도를 학습합니다. 3) Comparison-aware Pattern Extractor: 각 리스트 내에서의 사용자 행동 패턴을 캡처합니다. CONTRASTIVE LEARNING(대조 학습)이 데이터를 잘 통합하는 데 사용됩니다.

- **Performance Highlights**: 광범위한 실험을 통해 RELIFE가 최첨단(SOTA) 재순위 모델들을 상대적으로 크게 능가한다는 사실을 보여주었습니다.



### GPRec: Bi-level User Modeling for Deep Recommenders (https://arxiv.org/abs/2410.20730)
- **What's New**: 본 논문에서는 GPRec이라는 이중 레벨 사용자 모델링 방법을 소개하고, 이 방법이 기존의 사용자 그룹 모델링과 개인화된 추천 방식을 통합하여 추천 시스템의 성능을 향상시키는 것을 목표로 합니다.

- **Technical Details**: GPRec은 사용자들을 학습 가능한 방식으로 그룹화하고 각 그룹에 대한 그룹 임베딩(group embedding)을 정렬합니다. 이 방법은 긍정적 및 부정적 패턴을 대조하여 집단 선호에 대한 다양성을 제시하는 이중 그룹 임베딩 공간을 설계합니다. 개인 차원에서는 ID와 유사한 특징으로부터 개인의 선호를 식별하고, 그룹 선호와 독립적인 개인 표현을 정제하여 그룹 레벨 모델링에 강력한 보완을 제공합니다.

- **Performance Highlights**: 세 개의 공공 데이터셋에서 GPRec을 rigorously 테스트한 결과 추천 품질에서 상당한 개선이 이루어졌으며, 다양한 DRS 구조에 유연하게 통합할 수 있는 여러 전략이 제시되었습니다.



### GenUP: Generative User Profilers as In-Context Learners for Next POI Recommender Systems (https://arxiv.org/abs/2410.20643)
- **What's New**: 이번 연구는 자연어(NL) 사용자 프로필을 POI 추천 시스템에 도입하여 적은 역사적 데이터를 사용하면서도 보다 개인화된 추천을 가능하게 하는 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 LBSN 체크인을 분석하여 자연어 사용자 프로필을 생성하고, 이를 시스템 프롬프트로 사용하여 LLM을 개선하는 방식입니다. 이는 사용자 선호도를 반영한 해석 가능하고 투명한 프로필을 통해 사용자 맞춤형 추천의 정확도를 높이는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존 베이스라인 방법들을 지속적으로 능가하며, POI 추천 시스템의 해석 가능성과 자원 효율성을 동시에 향상시키는 경쟁력 있는 솔루션임을 입증했습니다.



### Collaborative Knowledge Fusion: A Novel Approach for Multi-task Recommender Systems via LLMs (https://arxiv.org/abs/2410.20642)
- **What's New**: 이번 논문에서는 기존의 추천 시스템(Recommendation Systems)에서 LLMs(대규모 언어 모델)의 통합을 넘어, CKF라는 새로운 프레임워크를 도입하여 개인화된 협업 지식을 LLMs에 융합하는 방안을 제시하고 있습니다.

- **Technical Details**: CKF 프레임워크는 기존의 협업 필터링(Collaborative Filtering) 모델을 활용하여 협업 임베딩(Collaborative Embeddings)을 생성하고, 메타 네트워크(Meta-Network)를 통해 각 사용자 맞춤형 매핑 다리를 구축합니다. Multi-Lora라는 새로운 다중 과제 최적화(Multi-task Optimization) 접근법을 개발하여, 과제 간의 관계를 명확히 구분하고, 공유된 정보와 특정 과제 정보를 분리합니다.

- **Performance Highlights**: 다양한 추천 작업에서 4개의 대규모 공개 데이터 세트를 통해 수행된 extensive experiments와 robustness 분석을 통해 CKF 프레임워크의 효과와 우수성을 입증하였습니다.



### R^3AG: First Workshop on Refined and Reliable Retrieval Augmented Generation (https://arxiv.org/abs/2410.20598)
Comments:
          R^3AG workshop overview at SIGIR-AP 2024

- **What's New**: 이번 논문은 Retrieval-augmented generation (RAG) 기술의 현재 문제점과 한계를 심층적으로 탐구하고, 이를 개선하기 위한 첫 번째 R3AG 워크숍이 SIGIR-AP 2024에서 개최될 것임을 알리고 있습니다.

- **Technical Details**: RAG는 정보 검색 (Information Retrieval, IR)과 대형 언어 모델 (Large Language Model, LLM)의 생성 기능을 결합하여 사용자 쿼리의 의도를 이해하고, 지식 파싱, 지식 검색 및 응답 생성을 최적화하는 파이프라인을 구현합니다. 주요 도전 과제로는 사용자 의도 이해, 복잡한 지식 문서 파싱, 신뢰할 수 있는 지식 검색 및 생성 응답의 정제 등이 있습니다.

- **Performance Highlights**: RAG는 정보 요청에 대한 사용자 경험을 향상시키고, 응답 정확성을 개선하며, 복잡한 쿼리 처리를 지원하기 위한 다단계 대화를 가능하게 합니다. 최신 연구에 따르면, RAG는 잘못된 정보와 대조적인 정보 문제를 완화하여 LLM의 사실 정확도를 향상시키는 데 기여하고 있습니다.



### Coherence-guided Preference Disentanglement for Cross-domain Recommendations (https://arxiv.org/abs/2410.20580)
Comments:
          28 pages

- **What's New**: 이 연구에서는 Coherence-guided Preference Disentanglement (CoPD) 방법을 제안하여 서로 다른 도메인에서 사용자 선호를 추출하고, 이를 통해 교차 도메인 추천 시스템의 성능을 향상시키고자 합니다. 본 방법은 공유 아이템 속성을 명시적으로 추출하여 공유 사용자 선호 학습을 안내하고, 이러한 선호를 분리하여 도메인 간 전이되는 특정 사용자 관심사를 식별합니다.

- **Technical Details**: CoPD에서는 공유 도메인과 특정 도메인의 아이템 임베딩에 대한 일관성 제약(coherence constraints)을 도입하여 공유 속성을 추출하는 데 도움을 줍니다. 또한, 인기도 가중 손실(popularity-weighted loss)을 통해 사용자 선호를 관심과 순응으로 분리하는 두 개의 임베딩을 생성합니다. 이는 GNN (Graph Neural Network)을 활용하여 도메인 공유 속성을 학습하고, 이를 기반으로 사용자 관심을 더 정교하게 모델링합니다.

- **Performance Highlights**: 실제 데이터셋을 기반으로 수행된 실험에서 CoPD는 기존의 경쟁적인 기준선 모델 대비 뛰어난 성능을 보여주었으며, 교차 도메인 추천 성능을 향상시키는 데 효과적임을 입증했습니다.



### Efficient and Effective Retrieval of Dense-Sparse Hybrid Vectors using Graph-based Approximate Nearest Neighbor Search (https://arxiv.org/abs/2410.20381)
Comments:
          8 pages

- **What's New**: 이 논문은 텍스트의 임베디드 벡터 표현을 위한 그래프 기반 근사 최근접 탐색(ANNS) 알고리즘을 제안합니다. 특히 희소(Sparse) 및 밀집(Dense) 벡터를 통합하여 검색 성능을 개선하는 새로운 접근 방식을 소개합니다.

- **Technical Details**: 제안된 알고리즘은 두 가지 주요 기법을 포함합니다: 첫째, 정확도를 향상시키기 위한 분포 정렬(distribution alignment) 방법은 희소 및 밀집 벡터의 거리 분포 통계를 분석하여 사전 샘플링을 수행합니다. 둘째, 효율성을 높이기 위해 적응형 2단계 계산 전략을 설계하여 초기에는 밀집 거리만 계산하고 이후에는 혼합 거리(hybrid distance)를 계산합니다. 추가적으로, 희소 벡터를 가지치기하여 계산 속도를 향상시킵니다.

- **Performance Highlights**: 제안된 방법은 기존의 혼합 벡터 검색 알고리즘과 비교하여 정확도를 동일하게 유지하면서 8.9배에서 11.7배까지의 처리량(throughput) 향상을 달성합니다. 또한, 단순 구현에 비해 약 2.1배의 속도 증가를 성취하여 전체 질의 속도(QPS)를 대폭 개선했습니다.



### WindTunnel -- A Framework for Community Aware Sampling of Large Corpora (https://arxiv.org/abs/2410.20301)
- **What's New**: 본 논문에서는 WindTunnel이라는 새로운 프레임워크를 제시하여 대규모 데이터에서 대표 샘플을 생성함으로써 정보 검색 실험을 보다 효율적으로 수행할 수 있도록 합니다.

- **Technical Details**: WindTunnel 프레임워크는 대규모 데이터셋에 적합하며, Apache Spark와 같은 분산 처리 엔진에서 구현될 수 있습니다. 이 프레임워크는 쿼리, 엔티티, 그리고 쿼리 관련성 지표를 포함하는 세 가지 관계형 데이터셋을 사용하여 연관된 에지를 생성합니다.

- **Performance Highlights**: WindTunnel에서 생성된 샘플은 균등 랜덤 샘플링을 통해 얻은 샘플에 비해 높은 정밀도를 나타내며, 이로 인해 현재 정보 검색에서의 샘플링 방법의 한계를 극복할 수 있음을 보여줍니다.



### Quam: Adaptive Retrieval through Query Affinity Modelling (https://arxiv.org/abs/2410.20286)
Comments:
          15 pages, 10 figures

- **What's New**: 본 논문은 정보 검색 및 NLP 커뮤니티에서 사용자 정보 요구를 기반으로 문서를 랭킹하기 위한 relevance 모델(building relevance models)을 구축하는 새로운 접근 방식을 제안합니다. 특히 Quam이라는 query-affinity model을 도입하여 적응형 검색(adaptive retrieval) 분야의 발전을 이룹니다.

- **Technical Details**: Quam은 relevance-aware document similarity graph를 활용하여 초기 랭킹 문서의 한계를 극복하는 데 중점을 둡니다. 이 모델은 문서 선택 기준을 단순한 heuristic 설계 선택에 국한하지 않고 개선하여, 저렴한 re-ranking budget에서도 recall을 증대시킵니다. 이 모델은 어떤 적응형 검색 접근 방식에도 통합될 수 있습니다.

- **Performance Highlights**: Quam은 기존의 standard re-ranking baseline에 비해 recall 성능을 최대 26% 향상시키며, 기존의 적응형 검색 접근 방식도 최대 12%까지 recall을 개선한다고 보고하였습니다.



### UniHGKR: Unified Instruction-aware Heterogeneous Knowledge Retrievers (https://arxiv.org/abs/2410.20163)
- **What's New**: 새로운 연구에서는 UniHGKR라는 통합된 이질적 지식 검색기(knowledge retriever)를 소개하여, 이 모델이 여러 종류의 지식 출처와 사용자 지시에 따라 검색할 수 있는 능력을 가지고 있음을 보여줍니다. 기존의 정보 검색 모델은 동질적인 구조만을 가정하여 실제 세계의 다양한 요구를 충족하지 못했습니다.

- **Technical Details**: UniHGKR는 세 가지 주요 단계로 이루어져 있습니다: (1) 이질적 자기 지도(pretraining); (2) 텍스트 기반 임베딩 정렬(embedding alignment); (3) 지시 인식 검색기(fine-tuning)로, 이러한 단계들을 통해 다양한 검색 환경에서 일반화할 수 있는 능력을 갖추게 됩니다. 또한, 이 모델은 BERT 기반 버전과 대규모 언어 모델로 훈련된 UniHGKR-7B 버전이 있습니다.

- **Performance Highlights**: UniHGKR는 CompMix-IR 벤치마크에서 최신 기술의 방법들과 비교하여, 각각 6.36% 및 54.23%의 상대적 향상을 보였으며, ConvMix 과제에서 새로운 최첨단(record-breaking) 결과를 달성하여 최대 4.80점의 절대 향상을 이루었습니다.



### Optimizing Keyphrase Ranking for Relevance and Diversity Using Submodular Function Optimization (SFO) (https://arxiv.org/abs/2410.20080)
- **What's New**: 본 연구에서는 키프레이즈( keyphrase ) 랭킹에서의 다각성을 고려하기 위해 Submodular Function Optimization (SFO) 방법론을 제안합니다.

- **Technical Details**: 이 접근법은 키프레이즈 랭킹 작업을 submodular maximization으로 프레임화하여 관련성과 다양성을 균형 있게 고려합니다. 이를 통해 다양한 대표적 키프레이즈를 선택하게 됩니다.

- **Performance Highlights**: 벤치마크 데이터셋에서의 실험 결과, 제안하는 방법은 기존 방법들보다 관련성과 다양성 메트릭에서 뛰어난 성능을 보이며, 실행 시간에서도 SOTA (State of the Art) 성능을 달성하였습니다.



### Multi-Field Adaptive Retrieva (https://arxiv.org/abs/2410.20056)
- **What's New**: 이번 연구에서는 Multi-Field Adaptive Retrieval (MFAR)라는 새로운 프레임워크를 소개합니다. MFAR는 문서의 구조적 데이터를 활용하여 조회 성능을 지속적으로 향상시키도록 설계되었습니다. 이는 여러 필드(예: 제목, 본문 등)를 가진 문서 인덱스를 독립적으로 처리할 수 있는 유연함을 제공합니다.

- **Technical Details**: MFAR의 두 가지 주요 단계는 (1) 기존 문서를 필드로 분해하고 이를 밀집(dense) 및 어휘(lexical) 방법을 통해 독립적으로 인덱싱하며, (2) 문서 쿼리에 조건을 두어 필드의 중요성을 적응적으로 예측하는 모델을 학습하는 것입니다. 또한 필드 간의 동적 가중치를 조정하여, 쿼리에 가장 관련 있는 필드를 선택할 수 있도록 합니다.

- **Performance Highlights**: MFAR는 기존의 다양한 검색 시스템들보다 개선된 문서 순위 성능을 보이며, STaRK라는 구조화된 문서 검색 데이터셋에서 최신 성능을 달성했습니다. 또한, MFAR는 사전 학습(pretraining) 과정 없이도 테스트 시 필드의 가용성을 조절 가능하여 사용자가 결과를 제어할 수 있는 장점을 제공합니다.



### AutoMIR: Effective Zero-Shot Medical Information Retrieval without Relevance Labels (https://arxiv.org/abs/2410.20050)
Comments:
          15 pages, 3 figures

- **What's New**: 본 논문은 Self-Learning Hypothetical Document Embeddings (SL-HyDE)라는 새로운 접근 방식을 소개하여 의료 정보 검색(MIR)에서의 효과적인 제로샷(dense retrieval) 문제를 해결하고자 합니다.

- **Technical Details**: SL-HyDE는 대규모 언어 모델(LLMs)을 활용하여 주어진 쿼리에 기반한 가상의 문서를 생성합니다. 이러한 가상 문서는 의료 컨텍스트를 포함하여 관련 문서 검색을 지원합니다. 또한, SL-HyDE는 라벨이 없는 의료 데이터셋을 활용하여 자가 학습 프레임워크를 통해 문서 생성 및 검색 과정을 지속적으로 개선합니다. CMIRB(중국 의료 정보 검색 벤치마크)는 실제 의료 시나리오를 기반으로 한 평가 프레임워크로, 5개의 작업과 10개의 데이터셋을 포함합니다.

- **Performance Highlights**: SL-HyDE는 기존의 방법들보다 검색 정확도가 현저히 향상되었으며, 다양한 LLM 및 검색기 구성에서 강력한 일반화 및 확장성을 보여주었습니다. CMIRB에서 SL-HyDE는 HyDE를 평균 4.9% 초과하는 성능을 발휘하였고, BGE 단독 검색기보다 7.2% 향상된 결과를 나타내었습니다.



### DQRM: Deep Quantized Recommendation Models (https://arxiv.org/abs/2410.20046)
- **What's New**: 이번 연구에서는 Deep Learning Recommendation Model (DLRM)을 기반으로 하여, 추천 모델을 부담 없이 훈련하고 실행할 수 있게 해주는 소형, 강력하고 효율적인 새로운 추천 프레임워크인 Deep Quantized Recommendation Model (DQRM)을 제안합니다.

- **Technical Details**: DQRM에서는 ultra-low INT4 정밀도로 DLRM 모델을 양자화하고, quantization-aware training (QAT)을 적용하여 메모리 및 계산 비용을 크게 낮추면서 과적합(overfitting)을 완화합니다. 또한, embedding 테이블의 그라디언트를 INT8 형태로 양자화하여 통신 부하를 감소시키고, 효율적인 훈련을 가능하게 합니다.

- **Performance Highlights**: DQRM 모델은 Kaggle 데이터셋에서 79.07% 정확도(모델 크기: 0.27 GB)와 Terabyte 데이터셋에서 81.21% 정확도(모델 크기: 1.57 GB)를 달성하여 FP32 DLRM보다도 더 높은 성능을 기록했습니다.



### FLOW: A Feedback LOop FrameWork for Simultaneously Enhancing Recommendation and User Agents (https://arxiv.org/abs/2410.20027)
- **What's New**: 이번 연구에서는 추천 시스템에서 추천 에이전트와 사용자 에이전트 간의 상호작용을 고려한 새로운 프레임워크인 FLOW를 제안합니다. 이전의 연구들은 주로 각 에이전트의 능력 향상에 중점을 두었으나, 두 에이전트 간의 협업은 고려하지 않았습니다.

- **Technical Details**: FLOW 프레임워크는 피드백 루프(feedback loop)를 통해 추천 에이전트와 사용자 에이전트 간의 협업을 이끌어냅니다. 추천 에이전트는 사용자 에이전트의 피드백을 분석하여 사용자의 선호도를 세분화하며, 사용자 에이전트는 추천된 항목들을 통해 사용자의 숨겨진 관심사를 발견하게 됩니다.

- **Performance Highlights**: 세 가지 널리 사용되는 추천 시스템 데이터셋을 통해 피드백 루프의 효과를 평가한 결과, 추천 에이전트와 사용자 에이전트의 성능이 동시에 향상됨을 보여주었습니다.



### Personalized Recommendation Systems using Multimodal, Autonomous, Multi Agent Systems (https://arxiv.org/abs/2410.19855)
- **What's New**: 이 논문은 다중 모드(multi-modal) 및 자율적인 다중 에이전트(multi-agent) 시스템을 이용한 고도화된 개인화 추천 시스템에 대해 설명합니다. 특히, Gemini-1.5-pro와 LLaMA-70B와 같은 미래지향적 AI 기술을 통합하여 전자상거래(e-commerce) 고객 서비스 경험을 개선하는 데 중점을 두고 있습니다.

- **Technical Details**: 시스템은 세 가지 에이전트로 구성되어 있습니다. 첫 번째 에이전트는 주어진 질문에 대한 적절한 제품을 추천하고, 두 번째는 추천된 제품에 대한 이미지를 기반으로 후속 질문을 제기하며, 세 번째 에이전트는 자율 검색을 수행합니다. 이 시스템은 실시간 데이터 수집, 사용자 선호 기반 추천 및 적응형 학습 기능을 갖추고 있습니다. Groq API를 사용하여 LPU(대형 프로세싱 유닛) 추론을 수행하며, 응답 시간을 수 밀리초로 줄임으로써 원활한 대화가 가능하게 합니다.

- **Performance Highlights**: 이 시스템은 사용자 맞춤형 쇼핑, 가격 비교 및 이미지 검색 기능을 통해 전자상거래 분야에 혁신적인 변화를 가져올 것으로 기대됩니다. 또한, Chatbot을 통한 문의 자동 해결 기능과 특정 제품에 대한 공공 관심을 기반으로 한 개인화 추천이 가능하여 고객 지원을 완전 자동화할 수 있습니다.



### Telco-DPR: A Hybrid Dataset for Evaluating Retrieval Models of 3GPP Technical Specifications (https://arxiv.org/abs/2410.19790)
- **What's New**: 이 논문에서는 3GPP(3rd Generation Partnership Project) 기술 문서를 활용한 통신 분야의 Q&A 시스템을 제안합니다. 또한, 텍스트와 테이블을 혼합한 형태의 curated 3GPP 코퍼스를 포함하는 혼합 데이터셋인 Telco-DPR를 소개합니다.

- **Technical Details**: 제안된 QA 시스템은 RAG(Retriever-Augmented Generation) 기술을 사용하며, DHR(Dense Hierarchical Retrieval) 모델로 계층형 패시지를 선택하여 문서와 패시지 수준에서 미세 조정을 통해 관련 정보를 추출합니다. 주요 평가 지표로는 Top-K 정확도와 Mean Reciprocal Rank (MRR)를 사용하여 성능을 검증합니다.

- **Performance Highlights**: DHR 모델은 관련 기술 정보를 검색하는 데 있어 기존 방법보다 우수한 성과를 나타내며, Top-10 정확도 86.2%를 달성하였습니다. RAG 모델과 GPT-4를 활용한 QA 시스템은 기존 데이터셋에 비해 14%의 답변 정확도 개선을 보였습니다.



### Author Unknown: Evaluating Performance of Author Extraction Libraries on Global Online News Articles (https://arxiv.org/abs/2410.19771)
- **What's New**: 최근 온라인 뉴스 콘텐츠의 저자 추출 성능을 평가하기 위해 715명의 저자와 754개의 뉴스 기사를 포함하는 다국적 데이터셋을 수집하였습니다. 이 연구에서는 다섯 개의 기존 소프트웨어 패키지와 하나의 맞춤형 모델을 비교하고, Go-readability와 Trafilatura가 가장 일관된 솔루션임을 발견하였습니다.

- **Technical Details**: 연구에서는 HTML 기반 뉴스 기사의 저자를 식별하기 위한 다양한 방법론을 비교합니다. 저자 정보 추출을 위한 기존 방법은 대체로 ML 방법, 휴리스틱 방법, 그리고 이 두 가지를 조합한 방법의 세 가지 범주로 나누어집니다. 본 연구에서는 10개 언어로 구성된 새로운 핸드코드 크로스링구얼 데이터셋을 사용하는데, 이는 기존의 소프트웨어 패키지들의 성능을 유의미하게 평가하는 데 기여합니다.

- **Performance Highlights**: 모든 패키지들이 언어에 따라 높은 변동성을 보였으며, 저자 데이터 활용 시 추가적인 언어 및 지역에 대한 검증이 필요하다는 점이 강조됩니다. Go-readability와 Trafilatura는 상대적으로 일관된 성능을 보여 연구자들이 더 나은 결과를 얻기 위해 이러한 도구를 활용할 수 있는 기반을 마련하였습니다.



### Unraveling Movie Genres through Cross-Attention Fusion of Bi-Modal Synergy of Poster (https://arxiv.org/abs/2410.19764)
- **What's New**: 이 논문에서는 영화 포스터를 활용하여 멀티 라벨(movie genre classification) 장르 분류 문제를 해결하는 프레임워크를 제시합니다. 기존의 접근 방식들은 주로 줄거리 요약(plot summaries), 자막(subtitles), 예고편(trailers), 영화 장면에 초점을 맞추었으나, 포스터에서 시각적 및 텍스트적 정보를 효율적으로 활용하여 장르 식별을 시도한 것이 특징입니다.

- **Technical Details**: 영화 포스터에서 텍스트를 추출하기 위해 OCR(Optical Character Recognition)을 사용했으며, 관련 임베딩(embedding)을 회수하였습니다. 이후, 크로스 어텐션 기반(fusion module) 융합 모듈을 도입하여 시각적 및 텍스트 임베딩에 주의 가중치를 할당합니다. 13882개의 포스터를 IMDb(Internet Movie Database)에서 수집하여 실험을 수행하였고, 다양한 현대 아키텍처와 비교해 우수한 성능을 보였음을 확인하였습니다.

- **Performance Highlights**: 우리 모델은 복수 장르 식별(multi-label genre identification) 작업에서 주요 최첨단 방법보다 꾸준히 우수한 성능을 보여주었으며, 각 모듈의 유용성을 입증하는 어블레이션(ablation) 연구 결과가 포함되어 있습니다.



### Towards Next-Generation LLM-based Recommender Systems: A Survey and Beyond (https://arxiv.org/abs/2410.19744)
- **What's New**: 본 논문에서는 Large Language Models (LLMs)가 recommender system 분야에 어떻게 긍정적인 영향을 미치는지 분석하고, 기존 방식의 한계를 극복하기 위한 방법론을 제시합니다.

- **Technical Details**: LLMs의 능력 향상으로 추천 시스템이 사용자와 아이템을 보다 정확하게 표현하고 이해할 수 있는 가능성을 논의합니다. 또한, LLMs의 기능을 활용하여 기존 추천 시스템의 파이프라인을 확장하는 방법을 다룹니다.

- **Performance Highlights**: LLM 기반 추천 시스템이 연구에서 산업 적용으로의 격차를 좁히는 데 중요한 역할을 할 것으로 기대되며, 이를 위한 새로운 연구 기회와 도전 과제를 제안합니다.



### Vision Search Assistant: Empower Vision-Language Models as Multimodal Search Engines (https://arxiv.org/abs/2410.21220)
Comments:
          Code is available at this https URL

- **What's New**: 이 논문에서는 기존 대규모 비전-언어 모델(Vision-Language Models, VLM)이 처리하지 못하는 새로운 시각 정보를 이해하고 응답할 수 있는 'Vision Search Assistant'라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 VLM과 웹 에이전트(web agent)의 협업을 통해 실시간 정보를 검색하고 새로운 객체에 대한 질문을 다룰 수 있도록 합니다.

- **Technical Details**: Vision Search Assistant는 VLM이 질문을 이해하고, 이미지의 특정 객체를 분석하며, 관련된 정보와 텍스트를 검색하는 과정을 포함합니다. 이 과정은 크게 세 단계로 나뉘며, 첫째로 크리티컬 비주얼 객체의 시각적 내용을 텍스트 형태로 서술하는 'Visual Content Formulation', 둘째로 사용자의 질문과 관련된 서브 질문을 생성하여 웹 지식을 검색하는 'Web Knowledge Search', 셋째로 최종 답변을 생성하는 'Collaborative Generation'을 포함합니다.

- **Performance Highlights**: 다양한 오픈세트(open-set) 및 클로즈드세트(closed-set) QA 벤치마크에서 extensive experiments를 실시한 결과, Vision Search Assistant는 기존 모델보다 훨씬 높은 성능을 보이며, 대규모 비전-언어 모델에 효과적으로 적용될 수 있음을 입증했습니다.



### Simultaneous Unlearning of Multiple Protected User Attributes From Variational Autoencoder Recommenders Using Adversarial Training (https://arxiv.org/abs/2410.20965)
- **What's New**: 본 연구에서는 여러 보호된 속성(예: 성별 및 나이)을 동시에 비학습(또는 학습하지 않도록)하는 AdvXMultVAE 모델을 제시합니다. 이는 기존의 모델들이 단일 속성만을 제거하는 데에 그쳤던 것과 차별화됩니다.

- **Technical Details**: AdvXMultVAE는 Variational Autoencoder (VAE) 아키텍처와 적대적 훈련(Adversarial Training)을 결합하여, 사용자의 보호된 속성을 연속적 및 범주적값으로 동시에 제거하는 방법을 지원합니다. 실험은 LFM-2b-100k와 Ml-1m 두 개의 데이터셋에서 수행되었습니다.

- **Performance Highlights**: 실험 결과, AdvXMultVAE는 성별 및 나이 제거 측면에서 다른 방법들보다 뛰어난 성능을 보였으며, NDCG와 Recall을 유지하면서 인종적 편견을 효과적으로 완화할 수 있음을 나타냈습니다.



### Leveraging AI and Sentiment Analysis for Forecasting Election Outcomes in Mauritius (https://arxiv.org/abs/2410.20859)
- **What's New**: 이번 연구는 AI 기반 감정 분석(sentiment analysis)를 활용하여 선거 결과 예측을 시도하는 새로운 접근 방식을 다루고 있으며, 2024년 모리셔스(Mauritius) 선거에 중점을 두고 있다.

- **Technical Details**: 이 연구에서는 신뢰할 수 있는 여론조사 데이터를 활용할 수 없는 상황에서, 모리셔스 주요 정치 정당인 L'Alliance Lepep와 L'Alliance Du Changement에 대한 언론의 감정을 분석한다. 이를 위해 다국어 BERT 기반 모델과 맞춤형 감정 점수 알고리즘(Sentiment Scoring Algorithm)을 활용하여 뉴스 기사를 긍정적, 부정적, 중립적으로 분류하였다. 또한 감정 영향 점수(Sentiment Impact Score, SIS)를 적용하여 시간에 따른 감정의 영향을 측정하였다.

- **Performance Highlights**: 예측 모델은 L'Alliance Du Changement가 최소 37석을 확보할 가능성이 높고, L'Alliance Lepep는 남은 23석을 차지할 것으로 예측하고 있다. 긍정적인 언론 감정이 예상 선거 성과와 강한 상관관계를 가지며, 언론이 대중의 인식을 형성하는 데 중요한 역할을 하고 있음을 강조한다. 이러한 접근 방식은 조정된 점수를 통해 언론의 편향을 줄이고 전통적인 여론조사에 대한 신뢰할 수 있는 대안이 된다. 이 연구는 제한된 여론조사 인프라를 가진 지역에서 정치 예측을 위한 확장 가능한 방법론을 제시하며, 정치 데이터 과학 분야의 발전에 기여한다.



### Temporal Streaming Batch Principal Component Analysis for Time Series Classification (https://arxiv.org/abs/2410.20820)
- **What's New**: 이번 연구에서는 다변량 시계열 분류에서의 모델 성능을 최적화하기 위해, 장기 시계열 데이터의 영향을 완화하는 새로운 접근 방식을 제안합니다. 특히, Streaming PCA(Principal Component Analysis) 기반의 시계열 데이터 압축 및 차원 축소 알고리즘인 TSBPCA(Temporal Streaming Batch PCA)를 개발했습니다.

- **Technical Details**: 제안된 TSBPCA는 시계열 데이터의 시계적 업데이트를 통해 전체 시퀀스의 compact representation을 지속적으로 업데이트하여, 모델의 데이터 표현 능력을 향상시킵니다. 이는 기본 PCA 알고리즘에 기반하나, 동적 데이터 흐름을 고려하여 차원 축소를 수행하게 됩니다. 또한, 이 알고리즘은 주요 매개변수인 배치 크기 T와 주성분 개수 K의 최적 범위를 분석합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 여러 실제 데이터셋에서 분류 정확도와 처리 시간 모두에서 뛰어난 성능을 보였습니다. 특히, 가장 긴 시퀀스 데이터셋에서의 정확도가 약 7.2% 향상되었고, 실행 시간이 49.5% 감소했습니다. 이는 모델의 시간 효율성을 크게 개선함을 의미합니다.



### Automatic Estimation of Singing Voice Musical Dynamics (https://arxiv.org/abs/2410.20540)
Comments:
          To be published in ISMIR 2024, 6 pages

- **What's New**: 이 연구에서는 자동 음성 분석을 위한 뚜렷한 평가 프레임워크의 부족으로 인해 음악적 다이내믹스 분석의 한계에 대응하기 위한 데이터셋 수집 방법론을 제안합니다. 이 방법론을 통해 509개의 음악적 다이내믹스가 주석 처리된 성악 성능과 163개의 악보를 포함하는 데이터셋이 만들어졌습니다.

- **Technical Details**: 제안된 방법론은 OpenScore Lieder corpus에서 성악을 위한 음악적 다이내믹스 주석이 달린 509개의 성능 데이터를 수집하는 데 사용됩니다. 이 과정에서 음원 분리와 정렬 기술을 활용하였고, CNN 모델(다중 헤드 주의를 이용한)을 훈련시키기 위해 log-Mel 스펙트럼과 bark-scale 기반 특성의 두 가지 인풋 표현 기법을 사용하였습니다.

- **Performance Highlights**: 실험 결과, bark-scale 기반 특성이 성악의 다이내믹스 예측 작업에서 log-Mel 특성보다 월등히 더 나은 성능을 보임을 확인하였습니다. 또한, 수집된 데이터셋은 이후 연구를 위해 공개적으로 공유될 예정입니다.



### Prototypical Extreme Multi-label Classification with a Dynamic Margin Loss (https://arxiv.org/abs/2410.20401)
- **What's New**: 이번 논문에서는 PRIME이라는 새로운 Extreme Multi-label Classification (XMC) 방법을 제안합니다. PRIME은 새로운 prototypical contrastive learning 기술을 사용하여 효율성과 성능을 조화롭게 하여 기존의 brute-force 접근 방식을 초월합니다.

- **Technical Details**: PRIME은 라벨 프로토타입(label prototype)을 집계함으로써 쿼리 관련 정보를 통합하는 데이터 대 프로토타입(data-to-prototype) 예측 작업으로 XMC를 정의합니다. 이 방법은 Label Prototype Network라는 얕은 transformer encoder를 사용하여 text-based embeddings, 라벨 중심(label centroids), 학습 가능한 자유 벡터를 결합하여 라벨 표현을 풍부하게 만듭니다.

- **Performance Highlights**: PRIME은 여러 공개 벤치마크에서 최첨단 결과를 달성하며, 단일 GPU 예산 내에서 효율성을 유지하면서 기존의 brute-force 접근 방식보다 성능이 우수합니다. 특히, 모든 실험과 대부분의 메트릭에서 PRIME은 높은 성능 개선을 보였다.



### An approach to hummed-tune and song sequences matching (https://arxiv.org/abs/2410.20352)
- **What's New**: 이 논문은 사용자가 부르는 멜로디를 사용하여 노래 제목을 찾는 작업에 대한 첫 연구입니다. Hum2Song Zalo AI Challenge 2021에 기반하여, 인간의 허밍 소리를 인식하고 노래를 검색하는 방법론을 제시합니다.

- **Technical Details**: 데이터 전처리에는 mp3에서 mel-spectrogram 형식으로 변환하고 numpy 배열로 저장하는 과정이 포함됩니다. 모델 훈련은 다양한 state-of-the-art 모델(ResNet, VGG, AlexNet, MobileNetV2)에서 수행되며, Faiss 모듈을 사용하여 허밍 소리와 일치하는 노래를 검색합니다.

- **Performance Highlights**: 공식 테스트 세트에서 MRR@10 지표로 거의 94%의 성능을 달성하며, 공개 리더보드에서 1위를 기록했습니다.



### Mask-based Membership Inference Attacks for Retrieval-Augmented Generation (https://arxiv.org/abs/2410.20142)
- **What's New**: 이 논문에서는 Mask-Based Membership Inference Attacks (MBA) 프레임워크를 제안하여 Retrieval-Augmented Generation (RAG) 시스템에서의 Membership Inference Attacks (MIAs) 문제를 해결합니다. 이는 특정 단어를 마스킹하여 RAG 시스템의 신뢰성을 높이는 방법입니다.

- **Technical Details**: MBA 프레임워크는 마스킹 알고리즘을 통해 대상 문서에서 특정 단어를 효과적으로 마스킹한 후, 이를 RAG 시스템에 제공하여 마스크 값을 예측하도록 합니다. 마스크 예측의 정확도를 분석하여 해당 문서가 데이터베이스에 포함되는지를 판단합니다. M𝑀은 마스킹할 단어 수를, γ는 예측 정확도를 판단하는 데 사용하는 하이퍼파라미터입니다.

- **Performance Highlights**: 논문에서 제안한 MBA 프레임워크는 세 가지 공개 QA 데이터셋에서 성능을 평가하였고, 기존 방법들에 비해 ROC AUC 값이 20% 이상 향상되었습니다.



### RARe: Retrieval Augmented Retrieval with In-Context Examples (https://arxiv.org/abs/2410.20088)
- **What's New**: 본 논문에서는 기존의 decoder-only language models (LLMs)에서 부분적으로 사용되는 in-context examples가 retrieval 모델의 성능 향상에 기여할 수 있는지를 연구합니다. 기존 LLMs에서의 사용 방법과는 달리, retrieval 작업에 적용하기 위해 새로운 접근법인 RARe(Retrieval Augmented Retrieval with In-Context Examples)를 제안합니다.

- **Technical Details**: RARe는 사전 훈련된 모델을 fine-tuning하여, 타겟 쿼리와 의미적으로 유사한 in-context examples를 추가하는 방식입니다. 이를 통해 retrieval 시스템의 쿼리 형식을 조정하고, contrastive loss를 사용하여 실험하고 성능을 평가합니다.

- **Performance Highlights**: RARe는 다양한 오픈 도메인 데이터셋(예: BeIR, RAR-b)에서 기본 모델보다 최대 2.72%의 nDCG 성능 향상을 보여주었으며, 특히 out-of-domain 일반화 성능이 뛰어난 것으로 나타났습니다.



### Evaluating Cost-Accuracy Trade-offs in Multimodal Search Relevance Judgements (https://arxiv.org/abs/2410.19974)
- **What's New**: 이 논문에서는 다양한 LLMs와 MLLMs의 성능을 평가하고, 특정 사용 사례에 따라 모델 성능이 어떻게 달라지는지를 분석하였습니다. 특히, 시각적 요소가 포함된 모델이 항상 성능을 향상시키지 않음을 밝혀내, 모델 선택의 복잡성을 강조합니다.

- **Technical Details**: 모델 평가 과정은 데이터 수집, 인간 주석, 모델 평가의 세 단계로 구성됩니다. 이 연구는 패션, 호텔 용품, 디자인과 같은 세 가지 데이터셋을 사용하였으며, 각 데이터셋은 다양한 특성과 관련 이미지를 포함합니다. 인간 주석자와 LLM을 사용하여 중요성을 평가하고, Cohen's kappa 계수를 통해 주석자 간의 일치를 측정합니다.

- **Performance Highlights**: 모델의 성능은 맥락에 크게 의존하며, LLM의 비용-정확도 트레이드오프에 대한 추가 연구가 필요합니다. 특히, 작은 규모의 모델에서는 시각적 구성 요소가 성능을 저하시킬 수 있음을 확인하였습니다.



### DualMAR: Medical-Augmented Representation from Dual-Expertise Perspectives (https://arxiv.org/abs/2410.19955)
- **What's New**: 본 논문에서는 EHR(전자 건강 기록)의 예측 작업을 향상시키기 위해 DualMAR이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 개별 관찰 데이터와 공공 지식 기반을 모두 활용하여 EHR의 예측 정확도를 높입니다.

- **Technical Details**: DualMAR는 두 가지 주요 구성 요소인 ‘Knowledge Scholar’와 ‘Local Expert’로 이루어져 있습니다. ‘Knowledge Scholar’는 공공 데이터베이스를 기반으로한 진단 주도 KG(지식 그래프)를 구축하고, ‘Local Expert’는 특정 질병-검사 EHR 그래프를 이용하여 의료 코드의 임베딩을 향상시킵니다. 이들 구성 요소는 Encoder-Decoder 아키텍처를 통해 통합되어 예측 모델의 성능을 높입니다.

- **Performance Highlights**: DualMAR는 실험 결과에서 최신 모델보다 우수한 성능을 보여 EHR 예측 및 KG(지식 그래프) 통합의 효과를 입증했습니다. 또한, 복잡하고 다양화된 관계를 가지고 있는 KG가 EHR 모델의 예측 성능을 어떻게 개선할 수 있는지를 보여줍니다.



### Multidimensional Knowledge Graph Embeddings for International Trade Flow Analysis (https://arxiv.org/abs/2410.19835)
- **What's New**: 이 연구는 고차원성, 우발성 및 강한 비선형성을 특징으로 하는 경제 데이터의 복잡한 역학을 이해하기 위한 새로운 접근 방식을 제시합니다. 특히 KonecoKG라는 지식 그래프 임베딩 모델을 활용하여 국제 무역 관계를 예측하고 이를 통해 무역 흐름을 최적화하는 방법을 탐구합니다.

- **Technical Details**: KonecoKG는 다차원 관계를 가진 경제 무역 데이터의 지식 그래프 표현으로, SDM-RDFizer를 사용하여 관계를 변환하고 AmpliGraph를 통해 지식 그래프 임베딩을 생성합니다. 연구는 경제 상호작용을 네트워크 구조 내에서 표현하며, 무역 네트워크 데이터셋을 활용하여 역사적 무역 패턴과 인접국의 무역 행동 통찰을 통합하여 미래 무역 기회를 예측하고자 합니다.

- **Performance Highlights**: 이 연구는 무역 흐름을 예측하기 위해 다차원 관계를 모델링하여 비선형성 문제를 해결하며, KonecoTradeFlow 온톨로지를 도입하여 국제 경제 양자 간 무역 데이터를 표현합니다. 결과적으로, 정책 입안자, 기업 및 투자자에게 국제 무역 전략을 최적화할 수 있는 귀중한 통찰을 제공하며 경제 성장 추세를 발견하는 데 기여합니다.



### A Human-Centered Approach for Improving Supervised Learning (https://arxiv.org/abs/2410.19778)
- **What's New**: 이 논문은 Human-Centered (인간 중심) 접근 방식을 사용하여 Supervised Learning (감독 학습)에서 Stacking Ensembles (스태킹 앙상블)의 효율성을 높이고, 성능과 시간, 자원 제약 간의 균형을 맞추는 방법을 제시합니다.

- **Technical Details**: 제안된 알고리즘은 Ensemble Learning (앙상블 학습) 프로세스를 간소화하며, 강력한 기본 모델, 다양한 피처 추출 및 축소 기법, 클래스 확률 및 클러스터링을 결합하여 Supervised Learning의 성능을 최적화합니다. 이 과정에서 모델의 explainability (설명 가능성)도 증대시켜, 사용자가 알고리즘의 결정을 더 잘 이해할 수 있도록 합니다.

- **Performance Highlights**: 아홉 개의 실제 데이터셋에서 실험한 결과, 제안된 방법은 기존의 최첨단 방법들을 능가하는 성능을 보여주었습니다. 이 연구는 Ensemble Learning의 시간 및 계산 오버헤드를 줄이면서도 성능의 손실을 최소화하여 실질적인 적용 가능성을 높이는데 기여하고 있습니다.



### Tourism destination events classifier based on artificial intelligence techniques (https://arxiv.org/abs/2410.19741)
- **What's New**: 이번 연구는 관광지 관리에 있어 고객의 요구를 파악하고 최적의 서비스를 제공하기 위한 새로운 자동 분류 프로세스를 제안합니다. 이 프로세스는 관광 이벤트를 계층적 분류법 (hierarchical taxonomy)을 통해 체계적으로 분류합니다.

- **Technical Details**: 연구에서 사용된 기술적 방법론은 CRISP-DM과 같은 데이터 과학 기법, 감독기계학습 (supervised machine learning), 자연어 처리 기술 (natural language processing techniques)을 포함합니다. 이러한 방법들은 서로 다른 지리적 지역에서 이벤트 정보를 표준화된 카탈로그 (catalog)로 생성하는 데 기여합니다.

- **Performance Highlights**: 이 자동 분류 도구는 항공사, 여행사, 호텔 체인과 같은 여러 지역에서 정보를 제공하는 기업들에게 매우 가치가 있으며, 이벤트 카테고리에 상관없이 사용자들이 원하는 이벤트를 쉽게 찾을 수 있도록 돕습니다. 결과적으로 이 도구는 기업과 최종 사용자가 관광 이벤트 정보를 상호작용하는 방식을 혁신할 잠재력을 지니고 있습니다.



New uploads on arXiv(cs.CV)

### Enhancing Action Recognition by Leveraging the Hierarchical Structure of Actions and Textual Contex (https://arxiv.org/abs/2410.21275)
- **What's New**: 이 논문에서는 행동 인식(action recognition)을 개선하기 위한 새로운 접근 방식을 제안합니다. 이는 행동의 계층 구조와 맥락화된 텍스트 정보(예: 위치 및 이전 행동)를 활용하여 순차적인 맥락을 반영합니다. 저자는 행동 인식에 최적화된 새로운 transformer 아키텍처를 도입하고, 이를 통해 RGB 및 optical flow 데이터를 기반으로 시각적 특징을 얻고 텍스트 임베딩을 통해 맥락 정보를 표현합니다.

- **Technical Details**: 제안된 모델은 행동 인식의 정밀도를 높이기 위해 coarse(거친) 및 fine-grained(세밀한) 행동 인식을 동시에 훈련할 수 있도록 설계된 joint loss function을 사용합니다. 이 모델은 transformer 아키텍처를 기반으로 하며, 자기 주의 메커니즘(self-attention mechanism)을 활용하여 긴 범위의 시간적 의존성을 캡처합니다. Hierarchical TSU 데이터셋을 확장하여 행동의 계층을 도입한 것이 특징입니다.

- **Performance Highlights**: 제안된 방법은 동일한 하이퍼파라미터(hyperparameter)로 훈련했을 때 사전 훈련된 SOTA 방법들을 초월했습니다. 첫 번째로, ground-truth 맥락 정보를 사용했을 때 top-1 정확도에서 17.12% 향상된 결과를 얻었고, 실제 예측에서 얻은 맥락 정보를 사용할 때는 5.33% 향상이 나타났습니다.



### On Inductive Biases That Enable Generalization of Diffusion Transformers (https://arxiv.org/abs/2410.21273)
Comments:
          Project page: this https URL; Code repository: this https URL

- **What's New**: 본 연구는 transformer 기반의 denoising 네트워크가 지오메트리 적응형 하모닉 기반을 통해 표현될 수 있는 유도 편향을 가지지 않음을 밝혀냈습니다. 이에 따라, 'diffusion transformer'(DiT) 모델의 일반화 성능 개선을 위한 새로운 접근법을 제시하고 있습니다.

- **Technical Details**: 이 논문에서는 DiT의 주목할 만한 attention 모듈을 분석하고, attention map의 지역성이 일반화와 밀접한 관계가 있음을 발견하였습니다. 특히, DiT의 attention window를 제한하여 새로운 입력 tensors에 대한 종속성을 줄이는 local attention windows를 도입하였고, 이를 통해 DiT의 일반화 성능을 향상시킬 수 있음을 입증하였습니다.

- **Performance Highlights**: CelebA, ImageNet, LSUN 데이터셋을 사용한 실험 결과, attention window 제한을 통해 PSNR 갭이 감소하고, FID 점수가 개선되는 것을 확인하였습니다. 이는 제한된 훈련 이미지로도 DiT의 일반화 성능이 효과적으로 수정될 수 있음을 나타냅니다.



### LARP: Tokenizing Videos with a Learned Autoregressive Generative Prior (https://arxiv.org/abs/2410.21264)
Comments:
          Project page: this https URL

- **What's New**: 이번 논문에서는 이전 영상 토크나이저의 한계를 극복하기 위해 설계된 새로운 비디오 토크나이저인 LARP를 소개합니다. LARP는 전통적인 패치 기반 토크나이저와는 달리 전반적인 정보 수집을 통해 더 글로벌하고 의미 있는 표현을 캡처합니다.

- **Technical Details**: LARP는 비디오 내용을 토큰화하기 위해 학습된 전체 쿼리 집합을 사용하여 정보 수집을 수행합니다. 이를 통해 LARP는 불규칙한 수의 이산 토큰을 지원하며, 적응형 효율적인 토크나이제이션을 가능하게 합니다. 또한 경량의 AR transformer를 이전 모델로 통합하여 이산 잠재 공간의 다음 토큰을 예측합니다.

- **Performance Highlights**: LARP는 UCF101 클래스 조건 비디오 생성 벤치마크에서 최첨단 FVD 점수를 기록하며, 기존의 모든 비디오 생성 모델을 초월합니다. 이를 통해 AR 모델들이 비디오에 대한 호환성을 높이고, 고충실도 다중 모달 대형 언어 모델(MLLM) 개발의 잠재력을 열어줍니다.



### AutoBench-V: Can Large Vision-Language Models Benchmark Themselves? (https://arxiv.org/abs/2410.21259)
- **What's New**: 이번 논문에서는 LVLMs (Large Vision-Language Models)의 평가를 위한 자동화된 프레임워크인 AutoBench-V를 도입했습니다. 이 프레임워크는 특정 모델 능력에 기반한 자동 평가를 지원하여 LVLM의 효과적인 벤치마킹을 가능하게 합니다.

- **Technical Details**: AutoBench-V는 사용자 요구에 따라 모델의 특정 능력 (예: Spatial Understanding)에 대한 자동화를 지원합니다. 이 프레임워크는 텍스트-이미지 모델을 활용하여 관련 이미지를 생성하고, LVLM를 이용하여 시각적 질문-답변 (Visual Question Answering) 작업을 수행합니다. 또한, 자가 검증 메커니즘을 통해 참조 답변과 비교하여 성과를 평가합니다.

- **Performance Highlights**: AutoBench-V를 통해 7개의 인기 LVLM을 5가지 평가 능력에 대해 광범위하게 평가한 결과, 모델들은 높은 수준의 이해력에서는 강점을 보이지만, 세부적인 추론 작업에서는 낮은 성과를 보였으며, 이는 향후 연구의 개선 가능성이 있는 중요한 영역으로 분석되었습니다.



### Vision Search Assistant: Empower Vision-Language Models as Multimodal Search Engines (https://arxiv.org/abs/2410.21220)
Comments:
          Code is available at this https URL

- **What's New**: 이 논문에서는 기존 대규모 비전-언어 모델(Vision-Language Models, VLM)이 처리하지 못하는 새로운 시각 정보를 이해하고 응답할 수 있는 'Vision Search Assistant'라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 VLM과 웹 에이전트(web agent)의 협업을 통해 실시간 정보를 검색하고 새로운 객체에 대한 질문을 다룰 수 있도록 합니다.

- **Technical Details**: Vision Search Assistant는 VLM이 질문을 이해하고, 이미지의 특정 객체를 분석하며, 관련된 정보와 텍스트를 검색하는 과정을 포함합니다. 이 과정은 크게 세 단계로 나뉘며, 첫째로 크리티컬 비주얼 객체의 시각적 내용을 텍스트 형태로 서술하는 'Visual Content Formulation', 둘째로 사용자의 질문과 관련된 서브 질문을 생성하여 웹 지식을 검색하는 'Web Knowledge Search', 셋째로 최종 답변을 생성하는 'Collaborative Generation'을 포함합니다.

- **Performance Highlights**: 다양한 오픈세트(open-set) 및 클로즈드세트(closed-set) QA 벤치마크에서 extensive experiments를 실시한 결과, Vision Search Assistant는 기존 모델보다 훨씬 높은 성능을 보이며, 대규모 비전-언어 모델에 효과적으로 적용될 수 있음을 입증했습니다.



### Exploring contextual modeling with linear complexity for point cloud segmentation (https://arxiv.org/abs/2410.21211)
Comments:
          17 pages, 7 figures

- **What's New**: 새로운 연구에서는 기존 CNN 및 Transformer 기반 방법들보다 더 효율적이고 효과적인 포인트 클라우드 세분화 아키텍처를 제안합니다. 이 연구는 Mamba 구조의 단점을 분석하고 두 가지 개선점을 도입하여 Meepo라는 새로운 아키텍처를 개발하였습니다.

- **Technical Details**: Meepo는 Mamba의 선형 계산 복잡성을 활용하여 강력한 문맥 이해를 제공하고, CNN의 지역적 민감성(local sensitivity)을 통합합니다. 원래 Mamba에서의 인과성 제약을 제거하고 혁신적인 Strided Bidirectional SSM을 도입하여 문맥과 공간 이해를 증대시킵니다.

- **Performance Highlights**: Meepo는 여러 주요 벤치마크 데이터셋에서 이전의 최첨단 방법인 PTv3를 +0.8 mIoU 이상 초과하여 수행했으며, 42.1% 더 빠르고 5.53배 더 메모리 효율적입니다.



### Deep Learning-Based Fatigue Cracks Detection in Bridge Girders using Feature Pyramid Networks (https://arxiv.org/abs/2410.21175)
Comments:
          15 pages, 11 figures

- **What's New**: 본 연구는 교량의 강철 박스 기둥에 대한 균열 정보가 포함된 고해상도 이미지를 사용하여 자동 균열 세분화를 위한 새로운 프레임워크를 제안합니다.

- **Technical Details**: 이 연구에서는 균열의 다중 스케일 특성을 고려하여 Crack Detection을 위한 Feature Pyramid Networks (FPN) 아키텍처를 제안합니다. 입력으로 120개의 원본 이미지를 사용하며, 이미지 크기를 축소하고 하위 이미지로 분할하는 두 가지 접근 방식을 통해 처리합니다.

- **Performance Highlights**: 모든 개발된 모델은 원본 이미지에서 균열을 자동으로 감지할 수 있으며, 이미지 크기를 축소함으로써 계산 효율성을 개선하면서 정확도는 유지됩니다. 또한, 분할 방법을 사용하는 모델이 리사이징 방법을 사용하는 모델보다 더 정확한 균열 세분화를 제공합니다.



### Joint Audio-Visual Idling Vehicle Detection with Streamlined Input Dependencies (https://arxiv.org/abs/2410.21170)
- **What's New**: 이번 논문에서는 주차 중인 차량의 엔진이 꺼져 있을 때를 포함하여 차량의 상태(운전 중, 정지, 엔진 꺼짐)를 감지할 수 있는 오디오-비주얼 통합 인 Idle Vehicle Detection (IVD) 시스템을 소개합니다. 이는 기존의 비효율적인 접근 방식과 달리 사용자 입력을 최소화하여 시스템 배포의 오류 가능성을 줄입니다.

- **Technical Details**: 제안된 모델 AVIVD-Net은 오디오 및 비주얼 특성을 다이렉트하게 결합하기 위해 양방향 Attention mechanism을 사용하여 상호 보완적인 특징을 공동으로 학습합니다. 이 모델은 대규모의 AVIVD 데이터셋을 활용하며, 차량이 정지 시 발생할 수 있는 다양한 소리와 시각적 특성을 결합하여 빠르고 정확하게 차량 상태를 분류합니다.

- **Performance Highlights**: AVIVD-Net은 기존의 최첨단 방법들과 비교하여 경쟁력 있는 성능을 보여주며, AVIVD 및 MAVD 데이터셋에서의 평가 결과 그러합니다. 본 연구는 스스로 운전하는 차량 비디오 카메라 설정에도 확장 가능한 가능성을 제시합니다.



### Synthetica: Large Scale Synthetic Data for Robot Perception (https://arxiv.org/abs/2410.21153)
Comments:
          21 pages, 11 figures, 5 tables

- **What's New**: 본 논문에서는 Synthetica라는 방법을 소개하여 로봇 애플리케이션을 위한 강력한 상태 추정기를 훈련하기 위해 대규모 합성 데이터 생성을 가능하게 하고 있습니다.

- **Technical Details**: Synthetica는 포토리얼리스틱(realistic) 레이 트레이싱(ray-tracing) 렌더러를 활용하여 270만 이미지를 생성하여 고속 실시간 탐지 변환기를 훈련시킵니다. 이 방식은 이미지 구성의 다양성을 보장하고자 훈련 시 데이터 증강(augmentation) 기법을 활용합니다.

- **Performance Highlights**: 이 연구의 결과는 객체 탐지 작업에서 최신의 성능을 보여주었으며, 50-100Hz의 속도로 실행되는 탐지기를 통해 이전의 최고 성능(state-of-the-art) 방법에 비해 최대 9배 더 빠른 결과를 달성했습니다.



### Enhancing Learned Image Compression via Cross Window-based Attention (https://arxiv.org/abs/2410.21144)
Comments:
          Paper accepted and presented in ISVC'23. Copyrights stay with ISVC

- **What's New**: 최근 이미지 압축에 대한 연구가 활발해지고 있으며, 학습된 이미지 압축 방법이 전통적인 방법에 비해 우수한 성능을 보이고 있습니다. 본 논문에서는 CNN을 기반으로 한 특징 인코딩 모듈을 통합하여 지역적 중복성(local redundancy)을 효과적으로 캡처하고자 합니다.

- **Technical Details**: 본 연구는 CNN 기반의 솔루션과 크로스 스케일 윈도우 기반 주의 메커니즘(cross-scale window-based attention)을 활용하여 특징을 인코딩하고, 전역적 특징과 지역적 중복성을 최대한 활용하는 방법을 제안합니다. 이 방법은 다양한 신경망 아키텍처에 유연하게 통합될 수 있습니다.

- **Performance Highlights**: 우리의 방법은 Kodak 및 CLIC 데이터 세트에서 평가되었으며, 현재의 최첨단(image compression state-of-the-art) 방법들과 동등한 성능을 보임을 입증했습니다.



### Extrapolating Prospective Glaucoma Fundus Images through Diffusion Model in Irregular Longitudinal Sequences (https://arxiv.org/abs/2410.21130)
Comments:
          Accepted at BIBM 2024

- **What's New**: 이 연구에서는 기존의 장기 데이터(longitudinal datasets)를 기반으로 한 안구 질환인 녹내장(glaucoma) 진행 예측에서 새로운 확산 모델(diffusion-based model)을 제안합니다. 주로 각각의 녹내장 단계(label) 예측에 집중했던 기존 방법과는 달리, 이 모델은 환자의 기존 이미지를 활용하여 미래의 이미지를 예측합니다.

- **Technical Details**: 본 연구의 Diffusion Image Learning for Sequences (DILS) 모델은 기존 이미지 시퀀스를 입력으로 받아 시간 정렬 마스크(time-aligned mask)를 사용하여 특정 연도를 선택합니다. 이 마스크는 훈련 단계에서 장기 영상 시퀀스의 불규칙한 시간 간격 문제를 해결하는 데 중요한 역할을 하며, 무작위 마스킹을 통해 내부 관계를 학습합니다. 이미지 생성 과정에서 텍스트 레이블(textual labels)을 추가하여 생성된 이미지의 분류를 돕습니다.

- **Performance Highlights**: empirical findings (실험 결과)에서는 DILS 모델이 장기 데이터를 효과적으로 생성하며, 이후의 분류 작업에서 정확도를 크게 향상시킴을 보여주었습니다. 이는 의료 전문가들이 녹내장을 보다 효과적으로 진단하고 관리하는 데 기여할 수 있을 것입니다.



### Zero-Shot Action Recognition in Surveillance Videos (https://arxiv.org/abs/2410.21113)
- **What's New**: 이번 연구는 비디오 감시 분야에서 인간 자원의 부족 문제를 해결하기 위해 Large Vision-Language Models (LVLMs)를 활용하는 새로운 접근 방식을 제안합니다. 특히, Self-Reflective Sampling (Self-ReS)이라는 새로운 샘플링 메소드를 도입하여 비디오 이해 작업의 성능을 향상시키고자 했습니다.

- **Technical Details**: 연구에서는 LVLMs 중 가장 최신 모델인 VideoLLaMA2를 사용하여 실험을 진행했습니다. Self-ReS는 모델의 기본 샘플링 방법 대신에 각 비디오의 가장 관련성 높은 토큰을 선택하는 방식으로, 고유한 샘플링 전략을 통해 비디오의 시공간적 서명(spatio-temporal signature)을 효율적으로 생성합니다.

- **Performance Highlights**: 실험 결과, VideoLLaMA2는 baseline보다 20% 더 향상된 제로샷(zero-shot) 성능을 나타냈으며, Self-ReS를 통해 제로샷 행동 인식 성능은 44.6%로 증가했습니다. 이러한 결과는 LVLMs와 개선된 샘플링 기법이 다양한 감시 비디오 분석 시나리오에서 큰 잠재력을 가지고 있음을 보여줍니다.



### LAMA: Stable Dual-Domain Deep Reconstruction For Sparse-View C (https://arxiv.org/abs/2410.21111)
Comments:
          Journal version for LAMA (Learned Alternating Minimization Algorithm)

- **What's New**: 본 논문에서는 Learned Alternating Minimization Algorithm (LAMA)을 개발하여 tomographic imaging에서 발생하는 inverse problems을 효율적으로 해결하는 새로운 접근 방식을 제안합니다. 이 알고리즘은 데이터 기반 기술과 고전적인 기법을 통합하여 두 블록 최적화를 수행하며, 이미지와 데이터 도메인에서 학습 가능한 정규화를 사용합니다.

- **Technical Details**: LAMA는 비볼록(nonconvex) 및 비부드러운(nonsmooth) 정규화기(regularizers)를 허용하여 데이터에서 효과적인 특징을 추출합니다. 전체 목표 함수는 Nesterov의 smoothing 기술을 사용하여 최소화되며, 잔여 학습(residual learning) 아키텍처가 적용됩니다. 이로 인해 네트워크의 복잡성이 줄어들고 기억 효율성이 향상되며, 재구성의 정확도와 해석 가능성이 증가합니다.

- **Performance Highlights**: LAMA는 Computed Tomography의 인기 있는 벤치마크 데이터셋에서 최신 기법을 상당히 초과하는 성능을 보이며, 재구성 정확도와 안정성을 높이고 있습니다. 실험 결과는 LAMA가 그 효과와 안정성을 입증하고 있습니다.



### LiGAR: LiDAR-Guided Hierarchical Transformer for Multi-Modal Group Activity Recognition (https://arxiv.org/abs/2410.21108)
Comments:
          14 pages, 4 figures, 10 tables

- **What's New**: LiGAR는 LIDAR 데이터를 구조적 기반으로 활용하여 시각적 및 텍스트 정보를 처리하는 새로운 Multi-Modal Group Activity Recognition 접근 방식을 제안합니다. 이 연구는 복잡한 다중 에이전트 상호작용을 다루기 위해 멀티모달 데이터를 효과적으로 통합하는 점에서 큰 진전을 이루었습니다.

- **Technical Details**: LiGAR는 Multi-Scale LIDAR Transformer, Cross-Modal Guided Attention, Adaptive Fusion Module을 포함하는 계층적 아키텍처로 설계되어 있습니다. 이 프레임워크는 그룹 활동을 다양한 세밀함으로 포착할 수 있는 다층 처리를 지원하며, 데이터 간의 상호작용을 더 잘 이해하게 합니다.

- **Performance Highlights**: LiGAR는 JRDB-PAR, Volleyball, NBA 데이터셋에서 F1-score에서 10.6% 증가, NBA 데이터셋에서 Mean Per Class Accuracy에서 5.9% 증가와 같은 뛰어난 성능 개선을 보여주었습니다. LiGAR는 추론 과정에서 LIDAR 데이터가 없더라도 높은 성능을 유지, 적응성을 검증하였습니다.



### Efficient Mixture-of-Expert for Video-based Driver State and Physiological Multi-task Estimation in Conditional Autonomous Driving (https://arxiv.org/abs/2410.21086)
- **What's New**: 본 연구에서는 새로운 다중 작업 운전 중 모니터링 시스템(VDMoE)을 제안하여 SAE 레벨 2/3 자율 주행 환경에서 운전자의 인지 부하와 졸음을 효과적으로 평가할 수 있습니다.

- **Technical Details**: VDMoE는 RGB 비디오 입력을 활용하여 운전자의 상태를 비침습적으로 모니터링하며, 중요한 얼굴 특징을 이용해 계산 부담을 최소화하고, 원격 광용적맥법(rPPG) 기술을 통합하여 생리학적 정보를 제공합니다. Mixture-of-Experts (MoE) 프레임워크를 최적화하여 다중 모드 입력을 처리하고 다양한 작업에서 성능을 향상시킵니다. 또한, 통계적 사전과 모델 출력을 정렬하는 새로운 사전 포함 정규화 기법이 도입되었습니다.

- **Performance Highlights**: 42명의 참가자를 포함하는 새로운 데이터셋(MCDD)을 기반으로 한 검증 결과, VDMoE는 운전자의 상태 모니터링 효과가 입증되었습니다. 이 방법은 자율 주행 시스템의 안전성을 높이는 데 기여한다고 볼 수 있습니다.



### KA$^2$ER: Knowledge Adaptive Amalgamation of ExpeRts for Medical Images Segmentation (https://arxiv.org/abs/2410.21085)
Comments:
          This paper has been accepted to MICCAI2024

- **What's New**: 본 논문에서는 여러 전문 모델의 보호된 지식을 활용하여 실제 의료 이미지 분할 작업에 적응할 수 있는 새로운 지식 적응 융합 프레임워크(KA2ER)를 제안합니다. 이 프레임워크는 기존 기초 모델의 일반화 능력을 유지하며, 간섭이 발생할 수 있는 다양한 의료 데이터에서 측정 성과를 향상시키는 것을 목표로 합니다.

- **Technical Details**: KA2ER 프레임워크는 의료 이미지의 특정 분할 작업을 위한 전문 모델을 각각 훈련한 후, SwinUNETR를 기초 모델로 재사용합니다. 입력 데이터는 기초 모델과 전문 모델 모두에 의해 각각 인코딩되고, 서로 다른 태스크의 백본 특성은 적응형 융합 계층으로 공동 프로젝션됩니다. 이 과정에서 계층적 주의 메커니즘을 통해 전문 모델의 숨겨진 특성 지식과 기초 모델의 적응형 통합이 이루어집니다.

- **Performance Highlights**: 이 프레임워크에 대한 광범위한 실험이 MICAAI‘2024 CARE 챌린지의 네 가지 데이터 세트에서 시행되었으며, KA2ER의 효과성과 적응성을 입증하는 결과를 보여주었습니다. KA2ER 방식은 내추럴 이미지 분류에서 의료 이미지 분할로의 지식 융합을 시행한 최초의 작업으로 평가받고 있습니다.



### Kandinsky 3: Text-to-Image Synthesis for Multifunctional Generative Framework (https://arxiv.org/abs/2410.21061)
Comments:
          Accepted for EMNLP 2024 (Demo track)

- **What's New**: 이번 연구에서는 Kandinsky 3라는 새로운 Text-to-Image (T2I) 모델을 발표하며, 이는 Latent Diffusion을 기반으로 하여 높은 품질과 사진 같은(realism) 결과를 달성했습니다. 이 모델은 다양한 생성 작업에 쉽게 적응할 수 있는 단순함과 효율성을 특징으로 하며, 사용자가 직접 테스트할 수 있는 사용자 친화적인 시스템을 제공합니다.

- **Technical Details**: Kandinsky 3는 Text Encoder, U-Net 구조의 Noise 예측 네트워크, Image Decoder로 구성된 Latent Diffusion 모델입니다. 사용된 Text Encoder는 Flan-UL2 모델에 기반하며 8.6억 개의 파라미터를 가지고 있고, 이미지 디코더는 Sber-MoVQGAN에서 가져왔습니다. 전체 모델은 119억 개의 파라미터를 가지며, U-Net 훈련 동안 Text Encoder와 Image Decoder는 고정되었습니다. 이를 통해 모델의 훈련 성능을 향상시켰습니다.

- **Performance Highlights**: Kandinsky 3는 기존 모델보다 3배 빠른 속도의 추론을 가능하게 하며, 인간 평가에서 오픈 소스 생성 시스템 중 가장 높은 품질 점수를 기록했습니다. 텍스트 기반 인페인팅(outpainting), 이미지 퓨전(image fusion), 텍스트-이미지 퓨전 등을 지원하며, 사용자는 웹 플랫폼에서 다양한 기능을 시험해볼 수 있습니다.



### SPOTS-10: Animal Pattern Benchmark Dataset for Machine Learning Algorithms (https://arxiv.org/abs/2410.21044)
Comments:
          Dataset and benchmark is freely available at this https URL

- **What's New**: SPOTS-10 데이터셋은 야간의 그레이스케일 이미지에서 동물을 인식하기 위한 머신러닝 알고리즘 평가를 위해 구축되었습니다. 이는 특히 다양한 패턴으로 구분되는 10종의 동물 사진 50,000장을 포함하고 있으며, 환경 및 생태적 응용 프로그램을 위해 그 중요성이 강조됩니다.

- **Technical Details**: SPOTS-10 데이터셋은 10개 범주로 나뉘어진 32x32 그레이스케일 이미지 50,000장을 포함하고 있습니다. 이 데이터셋은 패턴 기반 분류를 위한 벤치마킹 데이터를 제공하여 컴퓨터 비전 기술 개발에 기여합니다. 데이터는 웹에서 수집 후 전처리되어 카메라 트랩 이미지처럼 보이도록 변환되었습니다.

- **Performance Highlights**: 기반 실험을 위해 Hinton et al.의 지식 증류 방법이 사용되었으며, 여러 모델 아키텍처(ResNet50, MobileNet 등)를 통해 학생 모델을 훈련시켰습니다. 실험 결과는 SPOTS-10 데이터셋에서 높은 정확도를 보여, 학생 모델이 선생 모델의 성능을 잘 모사하고 있음을 나타냅니다.



### Improving Visual Prompt Tuning by Gaussian Neighborhood Minimization for Long-Tailed Visual Recognition (https://arxiv.org/abs/2410.21042)
Comments:
          NeurIPS 2024

- **What's New**: 본 논문에서는 Random SAM prompt tuning (RSAM-PT)이라는 새로운 방법을 제안하여 긴 꼬리(long-tail) 클래스에서 모델의 일반화 성능을 향상시키는 동시에 계산 시간을 절반으로 줄이는 것을 목표로 합니다.

- **Technical Details**: RSAM-PT는 각 단계에서 하나의 경량화된 gradient 계산만을 필요로 하며, 이를 통해 파라미터의 랜덤 네이벌(정확한 이웃) 내에서 gradient descent 방향을 탐색합니다. 또한, deferred re-weighting 방식을 사용하여 꼬리 클래스의 샘플 중요성을 높입니다.

- **Performance Highlights**: 제안된 RSAM-PT 방법은 CIFAR100-LT (IF 100), iNaturalist 2018, Places-LT 벤치마크 데이터셋에서 각각 90.3%, 76.5%, 50.1%의 분류 정확도를 달성하며 기존 방법들보다 우수한 성능을 보입니다.



### Informed Deep Abstaining Classifier: Investigating noise-robust training for diagnostic decision support systems (https://arxiv.org/abs/2410.21014)
Comments:
          This preprint has no post-submission improvements or corrections. The Version of Record of this contribution is published in the Neural Information Processing, ICONIP 2024 Proceedings

- **What's New**: 이 연구에서는 Deep Abstaining Classifier (DAC) 손실 함수를 확장하여 Informed Deep Abstaining Classifier (IDAC) 손실 함수를 도입하였습니다. 이 방법은 훈련 중 노이즈 추정치를 포함하도록 설계되었으며, 노이즈에 대한 강건성을 향상시키는데 중점을 두었습니다.

- **Technical Details**: IDAC는 노이즈 레벨 추정치를 손실 함수에 통합하여 훈련 중 노이즈를 보다 효과적으로 처리할 수 있게 합니다. 이를 통해 DAC와 여러 최신 손실 함수에 비해 노이즈에 대한 강건성이 증대되었으며, 성능 향상도 보여주었습니다. 연구는 공개된 흉부 X-레이 데이터 세트를 사용하여 다양한 시뮬레이션된 노이즈 레벨에서 검증되었습니다.

- **Performance Highlights**: IDAC는 흉부 X-레이 데이터 세트를 기반으로 하여 1%에서 15% 사이의 낮은 노이즈 레벨에서 성능 테스트를 수행했으며, 자체적으로 수집한 노이즈가 포함된 데이터 세트에서도 그 효과가 입증되었습니다. 이러한 결과는 임상 데이터에서의 DDSS 개발에 있어 IDAC의 큰 잠재력을 보여줍니다.



### Push-Forward Signed Distance Functions enable interpretable and robust continuous shape quantification (https://arxiv.org/abs/2410.21004)
Comments:
          8 pages, 4 figures

- **What's New**: Push-Forward Signed Distance Morphometric (PF-SDM)라는 새로운 방법을 소개합니다. 이는 생물 의료 이미징에서 형태의 정량화를 지원하며, 연속적이고 해석 가능하며 형태 보존 변환에 불변입니다.

- **Technical Details**: PF-SDM은 연속적이며 해석 가능한 형태 정량화 방법으로, 형태의 기하학적 특성을 효과적으로 포착합니다. 이 방법은 Signed Distance Functions (SDF)에 기반하여, 모든 형태에 대해 전역 참조 도메인에서 정의된 Push-Forward SDF (PF-SDF)를 사용합니다. PF-SDM 알고리즘은 네 가지 단계로 구성됩니다: 이미지 전처리, SDF 계산, PF-SDF 계산, 그리고 형태 평가입니다.

- **Performance Highlights**: PF-SDM은 기존의 Elliptical Fourier Analysis(EFA)와 Generalized Procrustes Analysis(GPA)에서 발생하는 문제, 즉 계수 상관관계 및 기준점 선택의 문제를 피하면서 형태의 위상적 차이, 방사칭 근확대 및 곡률 변화를 강조하는 강력하고 해석 가능한 형태 설명자를 생성합니다.



### Skinned Motion Retargeting with Dense Geometric Interaction Perception (https://arxiv.org/abs/2410.20986)
Comments:
          NeurIPS 2024 Spotlight

- **What's New**: 이 논문에서는 신체 부위 간의 기하학적 상호작용을 효과적으로 모델링하여 스킨된 캐릭터의 모션 리타겟팅을 개선하는 새로운 프레임워크인 MeshRet를 소개합니다. 기존 방식에서는 기하학을 간과하거나 모션 리타겟팅 이후에 기하학 보정 단계를 추가하는 경우가 많았습니다.

- **Technical Details**: MeshRet는 세밀한 기하학적 상호작용을 처리하기 위해 세 가지 주요 요소를 도입합니다: 1) 세멘틱 일관성 센서(Semantically Consistent Sensors, SCS)로 다양한 메쉬 토폴로지 간의 밀집 메쉬 대응을 설정합니다. 2) 밀집 메쉬 상호작용(Dense Mesh Interaction, DMI) 필드를 개발하여 몸체 기하학 간의 접촉 및 비접촉 상호작용을 포착합니다. 3) DMI 필드를 정렬하여 모션 의미를 보존하고 자가 침투(self-interpenetration)를 방지합니다.

- **Performance Highlights**: 공식 Mixamo 데이터셋과 새롭게 수집한 ScanRet 데이터셋에서 실험한 결과, MeshRet는 기존 방법들에 비해 우수한 성능과 정확한 접촉 보존을 보여주었습니다. 이 연구는 기하학적 상호작용을 인식하는 모션 리타겟팅의 새로운 표준을 제시하고, 다양한 메쉬 토폴로지에서의 일반화 가능성을 확보하였습니다.



### EEG-Driven 3D Object Reconstruction with Color Consistency and Diffusion Prior (https://arxiv.org/abs/2410.20981)
- **What's New**: EEG(뇌전도) 기반의 색상 일관성을 가진 3D 객체 복원 방법을 제안하며, EEG 신호를 통해 3D 객체의 지역적 의미를 캡처하는 임피시트(implicit) 신경 인코더를 훈련합니다.

- **Technical Details**: 이 연구는 두 단계로 구성된 접근 방식을 사용합니다. 첫 번째 단계에서는 3D 객체를 인식할 수 있는 임피시트 신경 EEG 인코더를 훈련시키고, 두 번째 단계에서는 첫 번째 단계에서 획득한 잠재 EEG 코드(latent EEG codes)를 바탕으로 확산 모형(diffusion model), 신경 스타일 손실(neural style loss), 그리고 NeRF를 통합하여 3D 객체를 암묵적으로 복원합니다.

- **Performance Highlights**: 실험 검증을 통해 이 방법이 EEG 신호를 사용하여 색상 일관성을 가진 3D 객체를 복원할 수 있음을 입증하였습니다.



### MovieCharacter: A Tuning-Free Framework for Controllable Character Video Synthesis (https://arxiv.org/abs/2410.20974)
- **What's New**: 이번 논문에서는 캐릭터 비디오 합성(Character Video Synthesis)을 위한 간단하면서도 효과적인 튜닝 없이(tuning-free) 사용할 수 있는 프레임워크(MovieCharacter)를 제안합니다. 이 프레임워크는 합성 작업을 여러 모듈로 분해하여 사용자가 필요에 맞게 쉽게 커스터마이즈 할 수 있도록 합니다.

- **Technical Details**: MovieCharacter는 캐릭터 분할(character segmentation) 및 추적(tracking), 비디오 객체 제거(video object removal), 캐릭터 모션 모방(character motion imitation), 비디오 합성(video composition) 등으로 구성된 모듈화된 구조를 가지고 있습니다. 또한 비디오 합성 모듈에는 조명 인식(lighting-aware) 비디오 조화를 포함하여 캐릭터와 배경 간의 시각적 일관성을 높이고, 경계 인식(edge-aware) 비디오 정제를 통해 원활한 전환을 제공합니다.

- **Performance Highlights**: 실험 결과, MovieCharacter는 높은 품질의 합성 결과를 달성하며, 자원이나 전용 데이터셋에 대한 요구 없이도 효율성, 접근성, 적응성을 크게 향상시키는 것으로 나타났습니다. 이는 창의적이고 대화형 응용 프로그램에서 더 넓은 활용 가능성을 열어줍니다.



### Attention Overlap Is Responsible for The Entity Missing Problem in Text-to-image Diffusion Models! (https://arxiv.org/abs/2410.20972)
- **What's New**: 이 연구는 텍스트-이미지 변환 모델에서 발생하는 'entity missing(엔티티 누락)' 문제를 해결하기 위해 크로스 어텐션(크로스 주의) 맵의 중첩(overlap) 문제에 집중하였습니다. 연구진은 새롭게 제안한 네 가지 손실 함수가 이러한 문제를 해결하는 데 효과적임을 확인했습니다.

- **Technical Details**: 연구에서는 엔티티 누락의 세 가지 원인인 (1) 특정 엔티티에 대한 불충분한 주의 강도, (2) 과도한 주의 확산, (3) 다른 엔티티 간 주의 맵의 중첩을 분석하였습니다. 엔티티 맵 간의 중첩을 줄이기 위한 네 개의 최적화 목표(Intersection over Union (IoU), center-of-mass (CoM) distance, Kullback-Leibler (KL) divergence, clustering compactness (CC))를 도입하였습니다.

- **Performance Highlights**: 제안된 방법은 이전 접근 방식보다 시각적 질문 응답(VQA), 캡션 점수, CLIP 유사도 및 인간 평가에서 우수한 성과를 보였으며, 특히 CoM distance 최적화 방법을 통해 두 개의 엔티티가 포함된 경우 약 7%, 세 개의 엔티티가 포함된 경우 17% 이상의 향상된 엔티티 생성 성공률을 달성했습니다.



### BlueSuffix: Reinforced Blue Teaming for Vision-Language Models Against Jailbreak Attacks (https://arxiv.org/abs/2410.20971)
- **What's New**: 이번 연구에서는 비전-언어 모델(VLMs)에 대한 새로운 방어 방법을 제안합니다. 특히, 블랙박스 방어를 통해 jailbreak 공격에 대응하는 BlueSuffix라는 방법이 소개됩니다. 이 방법은 VLM의 성능을 저해하지 않으면서도 높은 안전성을 제공합니다.

- **Technical Details**: BlueSuffix는 세 가지 주요 구성 요소로 구성되어 있습니다: 1) jailbreak 이미지를 차단하는 시각적 정제기(visual purifier), 2) 악의적인 텍스트를 처리하는 텍스트 정제기(textual purifier), 3) 강화 학습을 통해 미세 조정된 블루 팀 접미사 생성기(blue-team suffix generator)로, 크로스 모달 강인성을 강화합니다. 모든 구성 요소는 블랙박스 환경에서 VLM의 취약점을 완화하기 위해 설계되었습니다.

- **Performance Highlights**: BlueSuffix는 LLaVA, MiniGPT-4 및 Gemini와 같은 세 가지 VLM 모델에서 테스트된 결과, 기존 방어 방법보다 공격 성공률(ASR)을 약 70%까지 감소시키는 성과를 보였습니다. 이는 VLM jailbreak 공격에 대한 방어의 새로운 기준을 설정하며, 실제 사용 사례에서 높은 실용성을 제공합니다.



### Improving Detection of Person Class Using Dense Pooling (https://arxiv.org/abs/2410.20966)
- **What's New**: 본 연구에서는 Faster_RCNN을 기반으로 하여 밀집 풀링(dense pooling) 기술을 적용하여 사람 객체를 탐지하는 성능을 획기적으로 향상시키는 접근 방식을 제안합니다. 이 방법은 COCO 데이터셋에서 수집한 6982개의 이미지를 사용하여 기존 방법 대비 더 정확한 결과를 달성하였습니다.

- **Technical Details**: Faster_RCNN 모델의 ROI pooling 레이어를 최적화하고, 밀집 풀링(dense pooling)을 통해 2D 이미지를 3D 모델로 변환하여 UV(ultra Violet) 이미지로 만들었습니다. 이를 통해 이미지에서 올바른 특징을 추출하기 용이하게 하였으며, Resnet-50 RPN과 Resnet-101을 활용하여 실험을 진행했습니다.

- **Performance Highlights**: 제안된 모델은 Faster_RCNN을 사용할 때보다 사람 객체 탐지에서 더욱 높은 정확도를 보여줍니다. 실험 결과는 Yolo-v7과 비교했을 때도 최첨단 결과를 나타내었으며, 연구 결과는 오픈 소스로 제공되어 다른 연구자들이 활용할 수 있도록 하였습니다.



### IndraEye: Infrared Electro-Optical UAV-based Perception Dataset for Robust Downstream Tasks (https://arxiv.org/abs/2410.20953)
Comments:
          9 pages, 2 figures

- **What's New**: 이 논문에서는 다양한 조건에서 객체 감지 및 세분화를 위한 다양한 작업에 적합하도록 설계된 인도 서브콘티넨트의 EO-IR(전자광학 및 적외선) 데이터세트인 IndraEye를 소개합니다.

- **Technical Details**: IndraEye 데이터세트는 5,612개의 이미지와 145,666개의 인스턴스를 포함하며, 다양한 시점, 고도, 7개의 배경 및 시간대에 대한 데이터를 제공합니다. 본 데이터세트는 객체 감지 및 세분화 작업에 대한 벤치마크가 이루어졌으며, 특히 드론을 통해 촬영된 다양한 EO-IR 이미지로 구성되어 있습니다.

- **Performance Highlights**: IndraEye 데이터세트는 객체 감지 및 의미적 세분화 작업에서의 모델 성능을 향상시키기 위해 다중 모드 학습과 도메인 적응을 지원합니다. 이는 DNN의 더 일반화된 표현을 개발할 수 있도록 하여, 까다로운 조건에서의 공중 인식 시스템의 정확성 및 강인함을 향상시키는 데 도움을 줍니다.



### Diff-Instruct*: Towards Human-Preferred One-step Text-to-image Generative Models (https://arxiv.org/abs/2410.20898)
- **What's New**: 이번 논문에서는 Diff-Instruct*(DI*)라는 데이터가 필요 없는 접근 방식을 소개합니다. 이 방법은 텍스트에서 이미지로의 한 단계 생성 모델을 구축하고, 인간의 선호에 맞추어 고도로 사실적인 이미지를 생성할 수 있는 능력을 유지합니다.

- **Technical Details**: DI*는 인간 피드백을 활용한 온라인 강화 학습(RLHF)으로 인간 선호 조정을 프레이밍합니다. 기존 RLHF 접근과는 달리, KL divergence 대신 새로운 score-based divergence 정규화를 도입하여 훨씬 우수한 성능을 보입니다. 이러한 분산을 직접 계산하는 것은 어려우나, 우리는 다루기 쉬운 손실 함수를 유도하여 효율적으로 기울기를 계산할 수 있음을 보여줍니다.

- **Performance Highlights**: Stable Diffusion V1.5를 참조 diffusion 모델로 사용한 DI*는 모든 기존 모델보다 큰 차이로 우수한 성능을 기록했습니다. 0.6B PixelArt-α 모델을 사용해 Aesthetic Score 6.30과 Image Reward 1.31을 달성하여 나머지 유사 모델의 점수를 거의 두 배로 초과했습니다. 또한 HPSv2 점수 28.70을 기록하여 새로운 최첨단 벤치마크를 설정했습니다.



### Evaluating the Robustness of LiDAR Point Cloud Tracking Against Adversarial Attack (https://arxiv.org/abs/2410.20893)
- **What's New**: 이 연구는 신경망 기반의 LiDAR 포인트 클라우드 추적 모델의 강인성(robustness)을 적대적인 공격(adversarial attacks) 하에서 심층적으로 분석합니다. 기존의 많은 연구가 성능 향상에 집중하는 반면, 본 연구는 적대적인 공격에 대한 강인성을 우선적으로 다룹니다.

- **Technical Details**: 우리는 3D 객체 추적의 문맥 내에서 적대적 공격을 수행하기 위한 통일된 프레임워크를 수립했습니다. 이를 통해 화이트 박스(white-box) 및 블랙 박스(black-box) 공격 전략을 철저히 조사합니다. 화이트 박스 공격의 경우, 다양한 추적 패러다임에 맞는 특정 손실 함수(loss function)를 조정하고 FGSM, C&W, PGD와 같은 기존 방법을 포인트 클라우드(domain)에 확장합니다. 블랙 박스 공격 시나리오에 대해서는 Target-aware Perturbation Generation (TAPG) 알고리즘을 도입하여 높은 공격 성능과 낮은 지각 가능성을 동시에 달성합니다.

- **Performance Highlights**: 실험적으로 고급 추적 방법들이 블랙 박스 및 화이트 박스 공격에 노출될 때 심각한 취약점(vulnerability)을 드러내며, 적대적 공격에 대한 강인성을 설계에 포함시킬 필요성을 강조합니다. TAPG 방법은 기존의 방법들과 비교하여 공격 효과성 및 왜곡된 소음의 은폐 간의 최적 균형을 이룹니다.



### Improving Generalization in Visual Reasoning via Self-Ensemb (https://arxiv.org/abs/2410.20883)
- **What's New**: 본 논문에서는 다양한 large vision-language models (LVLMs)의 성능을 향상시키기 위한 새로운 방법인 self-ensemble을 제안합니다.

- **Technical Details**: self-ensemble은 파라미터를 업데이트하지 않고 모델의 일반화 및 시각적 추론을 개선하는 훈련 비필요(training-free) 방법입니다. 본 연구의 주요 통찰력은 LVLM이 다른 LVLM 없이도 자체적으로 앙상블(ensembling)할 수 있다는 점입니다.

- **Performance Highlights**: 자체 앙상블 방법이 SketchyVQA, Outside Knowledge VQA 및 out-of-distribution VQA 작업에서 최신 최첨단(state-of-the-art, SOTA) 성능을 달성하는 데 효과적임을 다양한 벤치마크에서 검증하였습니다.



### The unrealized potential of agroforestry for an emissions-intensive agricultural commodity (https://arxiv.org/abs/2410.20882)
- **What's New**: 이 연구는 서아프리카의 코코아 생산 시스템에서 나무의 효과를 미세하게 분석하고, 나무가 제공할 수 있는 잠재적 혜택을 정량화하여 이 지역에서의 기후 변화 완화 및 적응을 위한 새로운 통찰을 제공합니다.

- **Technical Details**: 연구는 기계 학습(machlearning)을 통해 shade-tree cover와 탄소 저장(탄소 주식)을 위한 공간적으로 상세한(spatially-explicit) 추정치를 생성하였습니다. 이 연구는 가나 및 코트디부아르의 704만 헥타르에 걸친 농업 시스템에서 나무의 상태를 평가하는 첫 번째 사례로, 나무의 cover와 biomass에 대한 지도의 세부 정보를 제공합니다.

- **Performance Highlights**: 기존 shade-tree cover가 낮고 기후 위협과 공간적으로 일치하지 않으나, 이 연구는 고탄소 발자국을 연간 상쇄할 수 있는 큰 잠재력이 존재한다는 것을 발견하였습니다. 이 방법론은 다른 글로벌 중요 농산물에도 적용 가능하며, 탄소 시장의 회계 요구 및 지속 가능성 보고를 위한 새로운 법률 요구에 부합합니다.



### Evaluating Sugarcane Yield Variability with UAV-Derived Cane Height under Different Water and Nitrogen Conditions (https://arxiv.org/abs/2410.20880)
Comments:
          13 pages, 9 fugures, 1 table

- **What's New**: 본 연구는 UAV(무인 항공기)를 이용한 Digital Surface Model(DSM)을 통해 사탕수수의 높이와 수확량 간의 관계를 조사했습니다. 다양한 수분 및 질소 조건에서 사탕수수의 성장을 분석하며, 이러한 요인들이 사탕수수의 높이 및 수량에 미치는 영향을 보여줍니다.

- **Technical Details**: 연구 지역은 Uttar Pradesh(인도)에 위치하며, 62개 블록으로 나뉘어 서로 다른 물과 질소 조건이 반복 적용되었습니다. 각각의 블록에 대해 비모달(즉, 두 개의 폐곡선 분포) 분포를 통해 평균 사탕수수 높이가 추출되었습니다. DCHM(Derived Cane Height Model)을 생성하고, 포괄적으로 9개 치료 구역에서 평균 수확량과 DCHM 간의 회귀 분석을 수행했습니다.

- **Performance Highlights**: DCHM과 수확량 간의 회귀 분석 결과 R² 값이 0.95로 나타났습니다. 이는 UAV 기반 DSM이 효율적으로 사탕수수의 수확량을 추정하는 데 기여할 수 있음을 나타냅니다.



### ByteNet: Rethinking Multimedia File Fragment Classification through Visual Perspectives (https://arxiv.org/abs/2410.20855)
Comments:
          Accepted in TMM

- **What's New**: 본 논문에서는 기존의 MFFC(멀티미디어 파일 조각 분류) 방법론에서 간과되었던 intrabyte(비트 내) 정보를 활용하여 더 정밀한 분류를 가능하게 하는 새로운 접근법을 제시합니다. 주목할 만한 내용은 기존의 1D 바이트 시퀀스를 2D 회색조 이미지로 재해석하는 Byte2Image 모델과 두 가지 분기망인 ByteNet입니다. 이들은 바이트 간(intrabyte)과 바이트 간(interbyte) 관계를 동시에 탐색할 수 있도록 설계되었습니다.

- **Technical Details**: Byte2Image는 바이트 슬라이딩 윈도우를 사용하여 비트 수준의 정보를 추출하고, 이 데이터를 2D 이미지로 시각화하여 CNN(합성곱 신경망) 및 Vision Transformer(비전 변환기)를 통해 강력한 특성 추출을 수행합니다. ByteNet은 얕은 바이트 브랜치(BBFE)와 깊은 이미지 브랜치(IBFE)로 구성되어 있으며, BBFE는 원본 1D 바이트 시퀀스의 특정 바이트 공생을 인식하고, IBFE는 변환된 이미지에서 복잡한 관계를 탐색합니다.

- **Performance Highlights**: 논문에서 제안된 방법은 14개 사례에 대해 두 개의 벤치마크에서 실험을 실시하여, 기존 최첨단 접근 방식보다 최대 12.2%의 개선된 성능을 보였습니다. 이는 MFFC 분야에서의 혁신적인 설계와 실험 결과의 유망한 결과를 나타냅니다.



### Novel Object Synthesis via Adaptive Text-Image Harmony (https://arxiv.org/abs/2410.20823)
Comments:
          NeurIPS2024

- **What's New**: 이 논문은 객체 합성(object synthesis) 작업을 연구하며, 객체 텍스트(object text)와 객체 이미지를 결합하여 새로운 객체 이미지를 생성하는 방법을 제안합니다. 제안된 방법은 Adaptive Text-Image Harmony (ATIH)로, 텍스트와 이미지 피처를 조화롭게 통합하는 데 중점을 두고 있습니다.

- **Technical Details**: ATIH 방법은 세 가지 주요 요소로 구성됩니다: 첫째, 크로스 어텐션(cross-attention)에서 텍스트와 이미지 피처의 균형을 맞추기 위한 스케일 팩터(scale factor) 및 자기 어텐션(self-attention)에서 이미지 정보를 보존하기 위한 인젝션(injection) 단계 도입; 둘째, 균형 잡힌 손실 함수(loss function)를 설계하여 편집 가능성과 진실성을 보장; 셋째, 텍스트와 이미지 통합의 하모니를 위해 유사성 점수(similarity score) 함수 개발.

- **Performance Highlights**: 실험 결과, ATIH는 기존의 이미지 편집 및 창의적 혼합 방법들에 비해 창의적인 객체 조합에서 뛰어난 성능을 보였습니다. 예를 들어, 해양사자-유리병, 아프리카 카멜레온-새, 그리고 진흙 거북이-차 같은 독창적인 객체들이 생성되었습니다.



### Evaluation of neural network algorithms for atmospheric turbulence mitigation (https://arxiv.org/abs/2410.20816)
- **What's New**: 이 논문은 대기 난류로 인한 이미지 블러 문제를 해결하기 위해 다양한 신경망 아키텍처를 분석하고, 특정 아키텍처가 대기 난류 완화에 적합한지를 조사합니다. SOTIS라는 공개된 시뮬레이션 데이터셋의 생성을 제안하며, 이를 통해 신경망의 학습에 필요한 방대한 데이터를 제공할 수 있습니다.

- **Technical Details**: 이 연구에서 다룬 주요 아키텍처는 엔드 투 엔드(End-to-End) 방식으로 훈련된 신경망을 포함합니다. 대기에서 발생하는 블러는 비정상적인 카메라 움직임과 대상 물체의 움직임으로 인해 발생하며, 블러는 고정된 것으로 간주되어 시간을 초월해 변화하지 않는다고 합니다. 논문은 고전 알고리즘과 딥러닝 기반 알고리즘의 성능을 비교합니다.

- **Performance Highlights**: 대기 난류 완화 알고리즘을 평가하기 위한 방법론을 제안하고, SOTIS 데이터셋을 통해 다양한 신경망 아키텍처의 성능을 실험합니다. 실험 결과는 전통적인 방법과 신경망 기반 방법 간에 효과적인 비교를 가능하게 할 것으로 기대됩니다.



### Grid4D: 4D Decomposed Hash Encoding for High-fidelity Dynamic Gaussian Splatting (https://arxiv.org/abs/2410.20815)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 본 논문에서는 Gaussian splatting을 기반으로 한 새로운 동적 장면 렌더링 모델인 Grid4D를 제안합니다. Grid4D는 해시 인코딩(hash encoding)을 이용하여 4D 입력을 명시적으로 인코딩하는 혁신적인 방법을 채택하여, 기존의 낮은 차수(low-rank) 가정에 의존하지 않고 더 나은 성능을 제공합니다.

- **Technical Details**: Grid4D는 4D 인코딩을 1개의 공간 3D 해시 인코딩과 3개의 시간 3D 해시 인코딩으로 분해하여, 서로 다른 장면 구성 요소의 변형을 더 정확하게 모델링합니다. 또한, 방향성 주의 메커니즘(directional attention module)을 통해 연속적인 변형을 보다 정교하게 예측할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, Grid4D는 기존의 최첨단(dynamic scene rendering) 모델들과 비교하여 시각적 품질과 렌더링 속도에서 현저한 성능 향상을 보였습니다.



### Fidelity-Imposed Displacement Editing for the Learn2Reg 2024 SHG-BF Challeng (https://arxiv.org/abs/2410.20812)
- **What's New**: 이 논문에서는 두 가지 이미징 기술인 second-harmonic generation (SHG)과 bright-field (BF) 현미경을 결합하여 인간 유방 및 췌장 암 조직의 분석을 위한 다중 모드(multi-modal) 등록 프레임워크를 제안합니다. 기존의 학습 기반 등록 모델이 SHG 이미지를 BF 이미지에 정렬하는 데 겪는 어려움을 해결하기 위해 새로운 접근 방식을 도입했습니다.

- **Technical Details**: 제안된 프레임워크는 다음 세 가지 주요 요소로 구성됩니다: 배치-기반 대조 손실(batch-wise contrastive loss, B-NCE)을 사용하여 SHG와 BF 이미지 간의 공통된 특징을 캡처하며, 디스크립터 매칭을 통한 사전 정렬과 인스턴스 수준의 최적화(instance-level optimization)를 통한 정밀한 등록을 포함합니다. 이 방법은 지역 정규화 상관관계(local normalized cross-correlation, LNCC)와 교차 상호 정보 함수(cross mutual information function, CMIF)를 조합한 유사성 메트릭을 활용하여 글로벌 및 로컬 정렬의 균형을 맞춥니다.

- **Performance Highlights**: Learn2Reg COMULISglobe SHG-BF 챌린지에서 1위를 기록하며, 제안한 방법의 효과성을 실험적으로 증명했습니다. 정량적, 정성적 결과 모두 등록 정확도 및 강건성에서 상당한 향상을 보여줍니다.



### Long-Tailed Out-of-Distribution Detection via Normalized Outlier Distribution Adaptation (https://arxiv.org/abs/2410.20807)
Comments:
          NIPS2024

- **What's New**: 이번 연구에서는 Long-Tailed Recognition (LTR) 시나리오에서 Out-of-Distribution (OOD) 탐지 문제를 해결하기 위한 새로운 접근 방식인 Normalized Outlier Distribution Adaptation (AdaptOD)를 제안합니다. 이 방법은 훈련 데이터와 테스트 단계에서 모두 적용되는 outlier 분포를 진정한 OOD 분포로 조정하는 혁신적인 방법을 포함하고 있습니다.

- **Technical Details**: AdaptOD의 핵심 컴포넌트로는 Dynamic Outlier Distribution Adaptation (DODA)와 Dual-Normalized Energy loss (DNE)가 있습니다. DODA는 테스트 시간에 OOD 샘플에서 추출한 지식을 사용하여 vanilla outlier 분포를 적응시켜, OOD 점수를 보다 정확하게 추정할 수 있도록 합니다. DNE는 불균형한 ID 샘플을 위해 보다 균형 잡힌 예측 에너지를 강제하여 vanilla outlier 분포 학습을 개선합니다.

- **Performance Highlights**: CIFAR10-LT, CIFAR100-LT, ImageNet-LT와 같은 세 가지 LTR 벤치마크에서 AdaptOD는 기존의 최신 방법들(SOTA)을 능가하는 성능을 보였습니다. 연구 결과는 AdaptOD의 효율성을 입증하며, OOD 샘플 탐지 정확도를 크게 향상시키는 것으로 나타났습니다.



### Transformer-Based Tooth Alignment Prediction With Occlusion And Collision Constraints (https://arxiv.org/abs/2410.20806)
- **What's New**: 이번 연구에서는 Swin-transformer를 기반으로 한 경량의 치아 정렬 신경망을 제안합니다. 기존의 수작업으로 의존했던 치아 정렬 과정을 자동화하여 정확성과 효율성을 동시에 향상시킵니다.

- **Technical Details**: 3D 포인트 클라우드를 가상 아치 라인을 기준으로 재구성하고, 이를 정렬된 다채널 텍스처로 변환하여 모델의 성능을 최적화했습니다. 두 가지 새로운 교합 손실 함수(occlusal loss functions)를 설계하여 상하악 간의 교합 관계를 정량적으로 평가합니다.

- **Performance Highlights**: 591개의 임상 사례를 포함한 대규모 디지털 치아 교정 데이터셋을 구축하였고, 기존 방법들과의 비교 실험을 통해 높은 예측 정확성을 입증했습니다. 또한, 교합 관계 평가를 위해 제안된 손실 함수들로 인해 기존 STAT 방법보다 향상된 성능을 보였습니다.



### SparseTem: Boosting the Efficiency of CNN-Based Video Encoders by Exploiting Temporal Continuity (https://arxiv.org/abs/2410.20790)
Comments:
          9 pages, 13 figures

- **What's New**: 이 논문에서는 CNN(Convolutional Neural Network) 기반 비디오 인코더의 가속화와 메모리 효율성을 동시에 달성하는 새로운 프레임워크인 SparseTem을 소개합니다. 이 프레임워크는 비디오 프레임 간의 시간적 연속성을 이용하여 불필요한 계산을 건너뛰는 Diff Computation 기법을 채택합니다.

- **Technical Details**: SparseTem은 특히 메모리 소비 문제를 해결하기 위해 새로운 메모리 효율적인 스케줄링 방법인 SparseBatch를 제안하며, 중간 피쳐 맵을 캐시하지 않고도 연산을 수행할 수 있도록 지원합니다. 또한 온-라인 조정 메커니즘을 통해 서로 다른 레이어 간의 희소성 불균형을 완화하여 정확도 저하를 최소화합니다.

- **Performance Highlights**: SparseTem은 EfficientDet에 대해 최대 1.79배, CRNN에 대해 최대 4.72배의 속도 향상을 달성하며, 기존 방법들에 비해 메모리 오버헤드를 68.1%까지 줄이는 데 성공했습니다. 또한 정밀도 손실을 최소화하며 새로운 최첨단 성능을 세웁니다.



### Bidirectional Recurrence for Cardiac Motion Tracking with Gaussian Process Latent Coding (https://arxiv.org/abs/2410.20752)
Comments:
          Paper Accepted by NeurIPS 2024

- **What's New**: 새로운 GPTrack 프레임워크는 심장 움직임의 시간 및 공간 동역학을 완벽하게 탐구하기 위해 설계되었습니다. 기존의 이미지 쌍 분석 접근 방식의 한계를 극복하여, 심장 지역별 특성을 포착하고 길고 일관된 관계를 유지할 수 있도록 지원합니다.

- **Technical Details**: GPTrack은 잠재 공간(latent space)에서 순차적 Gaussian Process를 활용하여 심장 운동의 통계 정보를 인코딩합니다. 또한, 양방향 재귀 방식으로 정보를 집계하여 전통적인 diffeomorphic registration의 원리를 따릅니다. 이 방법은 3D 및 4D 의료 이미징에서 우수한 성능을 발휘합니다.

- **Performance Highlights**: GPTrack은 3D 초음파 영상과 4D MRI 이미지에서 모션 추적의 정확성을 크게 향상시켰습니다. 전반적으로 계산 효율성이 유지되면서도 최첨단 성능을 기록하며, 여러 의료 이미징 모드에 효과적으로 적용될 수 있는 유용성을 입증했습니다.



### BLAPose: Enhancing 3D Human Pose Estimation with Bone Length Adjustmen (https://arxiv.org/abs/2410.20731)
Comments:
          16 pages, 8 Postscript figures, uses this http URL and this http URL

- **What's New**: 본 연구는 기존 3D 인간 자세 추정 모델들이 간과해온 물리적 제약인 골격 길이의 일관성과 신체 대칭성을 고려한 순환 신경망 아키텍처를 제안합니다. 이는 비디오 전체를 통해 골격 길이를 정확히 예측할 수 있도록 설계되었습니다.

- **Technical Details**: 연구에서는 글로벌 정보를 활용하여 장기적인 비디오 시퀀스에서 골격 길이를 예측하는 순환 신경망(RNN) 모델을 도입합니다. 또한, 합성 데이터를 통해 골격 길이를 물리적 제약에 맞게 증강하는 새로운 전략을 제안하고, 골격 방향을 보존하면서 예측된 값으로 길이를 조정하는 방법을 발표합니다.

- **Performance Highlights**: Human3.6M 데이터셋에서 기존 모델들을 통해 검증한 결과, 골격 길이 예측 모델이 이전 최고 성능을 능가하며, 조정 및 미세 조정 방법이 여러 지표에서 성능을 개선하는 것으로 나타났습니다.



### CompGS: Unleashing 2D Compositionality for Compositional Text-to-3D via Dynamically Optimizing 3D Gaussians (https://arxiv.org/abs/2410.20723)
- **What's New**: 최근 3D 생성 분야에서 텍스트 기반 이미지 생성의 획기적인 발전이 이루어졌습니다. 본 논문에서는 복수 객체의 상호 작용을 고려한 컴포지셔널 3D 생성의 도전을 해결하기 위한 새로운 생성 프레임워크인 CompGS를 소개합니다.

- **Technical Details**: CompGS는 3D Gaussian Splatting (GS)을 활용하여 텍스트를 기반으로 한 3D 콘텐츠 생성의 효율성을 극대화합니다. 주요 두 가지 디자인은 (1) 2D 컴포지셔널리티를 활용한 3D Gaussians 초기화 및 (2) Score Distillation Sampling (SDS) 손실을 사용하는 동적 최적화입니다. 이 시스템은 각 엔티티 파트를 자동으로 분해하고 다양한 스케일의 객체를 최적화하여 세밀한 디테일 생성이 가능합니다.

- **Performance Highlights**: CompGS는 T3Bench에서의 정량적 평가와 질적 비교를 통해 기존 방법보다 우수한 이미지 품질과 의미적 정합성을 달성했습니다. 이 시스템은 또한 유연한 3D 편집 기능을 지원하여 장면 생성에 기여할 것으로 기대됩니다.



### Interpretable Image Classification with Adaptive Prototype-based Vision Transformers (https://arxiv.org/abs/2410.20722)
- **What's New**: ProtoViT는 심층 학습과 사례 기반 추론을 결합하여 이미지 분류의 해석 가능성을 향상시키는 방법입니다. 기존 CNN 기반 모델을 넘어 Vision Transformer (ViT) 백본을 통합하여 공간적으로 변형 가능한 프로토타입을 제공합니다.

- **Technical Details**: ProtoViT는 ViT 백본을 사용하는 새로운 아키텍처로, 기존의 공간적으로 경직된 프로토타입 대신 기하학적 변형을 허용하는 프로토타입을 적응적으로 학습합니다. 이를 위해 인접성 마스크와 적응 슬롯 메커니즘을 포함하는 새로운 그리디 매칭 알고리즘을 사용합니다.

- **Performance Highlights**: ProtoViT는 기존 프로토타입 기반 모델보다 높은 성능을 달성하고, 프로토타입의 일관성과 해석의 충실성을 보장합니다. 또한, 높은 정확도와 우수한 명확성 및 일관성을 보여주었습니다.



### Face-MLLM: A Large Face Perception Mod (https://arxiv.org/abs/2410.20717)
- **What's New**: 이번 연구에서는 기존의 멀티모달 대형 언어 모델(MLLM)이 인간 얼굴 인지 작업에서 성능이 부족하다는 사실을 밝혔습니다. 이를 해결하기 위해 새로운 데이터셋 구축 및 모델 훈련 파이프라인을 설계하여 ‘Face-MLLM’이라는 새로운 모델을 개발했습니다.

- **Technical Details**: Face-MLLM은 LAION-Face 데이터셋의 재주석을 통해 보다 자세한 얼굴 설명 및 얼굴 특성 레이블을 추가하고, 전통적인 얼굴 데이터셋을 질문-답변(Question-Answer, QA) 형식으로 재구성하였습니다. 또한 세 단계의 훈련 방법을 통해 시각-텍스트 정렬과 기본 시각 질문 응답 기능을 학습하여, 전문적인 얼굴 인지 작업을 처리하도록 설계되었습니다.

- **Performance Highlights**: Face-MLLM은 기존의 MLLMs보다 5개의 유명한 얼굴 인지 작업에서 우수한 성능을 보여주었으며, 새로 도입한 제로샷 얼굴 속성 분석 작업에서도 뛰어난 결과를 보였습니다. 실험 결과, Face-MLLM은 기존의 MLLMs에 비해 보다 정교하고 정확한 얼굴 이미지의 설명을 제공할 수 있음을 확인하였습니다.



### Physics-Free Spectrally Multiplexed Photometric Stereo under Unknown Spectral Composition (https://arxiv.org/abs/2410.20716)
Comments:
          ECCV2024 (Oral)

- **What's New**: 이 논문에서는 기존의 촬영 조명이나 센서에 대한 교정이 필요 없는 동적 표면의 표면 법선 복원을 위한 스펙트럼 다중화 포토메트릭 스테레오 접근법을 제안합니다. 이러한 기술은 전통적으로 엄격한 전제 조건과 스펙트럼 모호성으로 인해 제한된 분야에서 중요한 발전을 나타냅니다.

- **Technical Details**: 제안된 기법은 스펙트럼 모호성을 장점으로 활용하여 특별한 다중 스펙트럼 렌더링 프레임워크 없이 훈련 데이터를 생성할 수 있습니다. 우리는 물리 기반 지식에 의존하지 않고 다양한 조건과 재료 유형에서 표면 법선을 결정하기 위해 스펙트라M-PS라는 독특한 물리적 제약 없는 네트워크 아키텍처를 도입했습니다.

- **Performance Highlights**: 우리는 처음으로 스펙트럼 다중화 포토메트릭 스테레오를 위한 벤치마크 데이터셋인 SpectraM14를 설정하여 기존의 교정된 방법들과 포괄적인 평가를 수행했습니다. 이로 인해 동적 표면 복원에 대한 능력이 크게 향상되었으며, 비교적 제어되지 않은 환경에서도 효과적으로 작동합니다.



### CIB-SE-YOLOv8: Optimized YOLOv8 for Real-Time Safety Equipment Detection on Construction Sites (https://arxiv.org/abs/2410.20699)
Comments:
          5 pages, 5 figures

- **What's New**: 이 연구는 건설 현장에서 안전성을 보장하기 위한 YOLO 기반의 실시간 헬멧 감지 솔루션을 제안합니다.

- **Technical Details**: 제안된 CIB-SE-YOLOv8 모델은 SE( Squeeze-and-Excitation) 주의 메커니즘과 수정된 C2f 블록을 통합하여 감지 정확도와 효율성을 높였습니다.

- **Performance Highlights**: 이 모델은 SHEL5K 데이터셋을 활용하여 건설 현장에서 안전 규정 준수를 촉진하는 보다 효과적인 솔루션을 제공합니다.



### ODGS: 3D Scene Reconstruction from Omnidirectional Images with 3D Gaussian Splattings (https://arxiv.org/abs/2410.20686)
- **What's New**: ODGS는 3D Gaussian splatting 기술을 기반으로 한 새로운 rasterization 파이프라인으로, 전체적인 360도 이미지를 신속하게 재구성할 수 있는 방법을 제시합니다.

- **Technical Details**: ODGS는 각각의 Gaussian에 대해 단위 구와 접하는 접선 평면을 정의하고, 이를 통해 Gaussian을 해당 접선 평면에 투사합니다. CUDA를 활용하여 프로세스를 병렬화하여 100배 빠른 최적화 및 렌더링 속도를 실현합니다.

- **Performance Highlights**: ODGS는 여러 데이터 세트에서 최고의 재구성과 지각 품질을 제공하며, 특히 복잡한 3D 장면을 효과적으로 복원하는 능력에서 두드러진 성능을 보입니다.



### A Comparative Study of Multiple Deep Learning Algorithms for Efficient Localization of Bone Joints in the Upper Limbs of Human Body (https://arxiv.org/abs/2410.20639)
- **What's New**: 이 논문은 팔의 관절 탐지 문제를 다루며, 특히 팔꿈치, 어깨, 손목 및 손가락 관절의 위치를 X-Ray 및 CT 스캔에서 자동으로 찾아내는 방법을 제시합니다.

- **Technical Details**: 이 연구에서는 YOLOv3, YOLOv7, EfficientDet, CenterNet과 같은 다양한 Deep Learning (DL) 모델의 성능을 비교합니다. 연구는 공개적으로 사용 가능한 MURA (musculoskeletal radiographs) 데이터셋의 일부를 이용해 모델을 학습하고 테스트했습니다.

- **Performance Highlights**: 최고의 Mean Average Precision (mAP) 값은 YOLOv3에서 35.3, YOLOv7에서 48.3, EfficientDet에서 46.5, CenterNet에서 45.9로 나타났습니다. YOLOv7이 경계 상자(bounding boxes)를 정확하게 예측하는 데 가장 뛰어난 성능을 보였으며, YOLOv3는 Visual Analysis 테스트에서 가장 낮은 성능을 기록했습니다.



### Ant Detective: An Automated Approach for Counting Ants in Densely Populated Images and Gaining Insight into Ant Foraging Behavior (https://arxiv.org/abs/2410.20638)
- **What's New**: 이번 연구는 자동화된 컴퓨터 비전을 활용하여 개미(ant)의 탐색 행동(foraging behavior)을 정량화하는 새로운 접근법을 제시합니다. 이를 통해 노동 집약적인 수작업(counting) 과정을 대체할 수 있습니다.

- **Technical Details**: YOLOv8 모델을 활용하여 시스템을 보정(calibration)하고 다양한 촬영 시나리오와 밀도에서 평가했습니다. 보정 이미지가 유사한 배경을 공유할 때, 평균 정밀도(precision)와 재현율(recall)은 각각 87.96% 및 87.78%에 달합니다. 더 복잡한 배경에서는 1,024장의 이미지가 필요하여 정밀도와 재현율은 각각 83.60% 및 78.88%로 나타났습니다. 또한, 여러 천 마리의 개미가 포함된 이미지를 작은 패치(patch)로 나누어 처리함으로써 정밀도와 재현율을 각각 77.97% 및 71.36%로 향상시킬 수 있었습니다.

- **Performance Highlights**: 시스템은 개미의 활동의 공간적(spatial) 분포를 시각화(visualization)하는 히트맵(heatmap)을 생성할 수 있어 탐색 패턴에 대한 귀중한 통찰(insight)을 제공합니다. 이 연구는 자동화된 카운팅(bug counting) 프로세스와 행동 분석을 통해 생태학적 연구(ecological studies) 및 해충 관리(pest control methods)를 위한 보다 효율적인 도구를 제공합니다.



### PViT: Prior-augmented Vision Transformer for Out-of-distribution Detection (https://arxiv.org/abs/2410.20631)
- **What's New**: 이 논문은 Prior-augmented Vision Transformer (PViT)라는 새로운 프레임워크를 제안하여 Vision Transformer (ViT)의 Out-of-Distribution (OOD) 검출 방식을 개선한다. PViT는 사전 훈련된 모델에서 얻은 사전 로짓(prior logits)과 예측 로짓(predicted logits) 사이의 발산(divergence)을 정량화하여 OOD 샘플을 식별한다.

- **Technical Details**: PViT는 기존의 OOD 검출 방법과는 달리 추가적인 데이터 모델링이나 구조적 수정 없이도 ID와 OOD 간의 결정 경계(decision boundary)를 형성하는 사전 가이드 신뢰성(prior guide confidence)을 활용한다. PViT는 Prior Guide Energy (PGE) 점수를 사용하여 OOD 인스턴스를 효과적으로 구별한다.

- **Performance Highlights**: 대규모 ImageNet 벤치마크에서 PViT는 기존의 OOD 검출 방법들을 큰 폭으로 초월하여 FPR95를 최대 20% 감소시키고 AUROC를 최대 7% 증가시키는 성과를 보인다. 이는 PViT가 합성 아웃라이어 데이터를 생성할 필요 없이 ID 데이터셋에서 높은 정확도를 유지함을 의미한다.



### Exocentric To Egocentric Transfer For Action Recognition: A Short Survey (https://arxiv.org/abs/2410.20621)
- **What's New**: 본 논문은 첫 번째 및 세 번째 인칭 시점의 비디오를 동기화하여 에고(egocentric)와 엑소(exocentric) 비전을 모두 모델링하는 방법에 대한 포괄적인 개요를 제공합니다. 두 가지 관점의 비디오 데이터가 결합되어 AI가 인간의 행동을 이해하는 데 필요한 중요한 신호를 제공합니다.

- **Technical Details**: 에고 비전은 카메라 착용자의 관점에서 장면을 캡쳐하고, 엑소 비전은 전체 씬의 맥락을 포착합니다. 두 관점을 결합하여 VLMs (Vision-Language Models) 같은 기법을 통해 데이터의 상호 연관성을 학습하며, 이를 통해 동작 감지, 비디오 이상 탐지 등의 다양한 비전 작업에서 성능을 향상시킬 수 있습니다.

- **Performance Highlights**: 여러 새로운 대규모 데이터셋이 출시되어 에고-엑소 비전 학습의 격차를 메우고 있습니다. 이 논문에서는 기존 연구 및 최근의 도전 과제에 대한 다양한 벤치마크를 제시하며, 새로운 학습 과제에 대한 제안도 포함되어 있습니다.



### A Framework for Real-Time Volcano-Seismic Event Recognition Based on Multi-Station Seismograms and Semantic Segmentation Models (https://arxiv.org/abs/2410.20595)
Comments:
          10 pages, 9 figures. This is a pre-print, it is currently under review fro publication at Computers and Geosciences, by Elsevier

- **What's New**: 이번 연구는 다중 채널 1D 신호를 2D 이미지로 변환하는 단순한 변환을 적용하여 화산 모니터링을 자동화하는 새로운 접근 방식을 제안합니다. 이를 통해 세 가지 주요 문제를 해결하고, 다중 스테이션 데이터를 통합하여 실시간 화산 모니터링에 적합한 모델을 개발했습니다.

- **Technical Details**: 제안된 프레임워크는 semantic segmentation 모델을 활용하여 5개의 화산 사건 클래스를 동시에 탐지하고 분류할 수 있도록 설계되었습니다. 약 25,000개의 지진 이벤트에 대해 UNet, UNet++, DeepLabV3+, SwinUNet 등의 최신 segmentation 모델을 평가했으며, UNet 아키텍처가 가장 높은 성능을 보였습니다.

- **Performance Highlights**: UNet 모델은 평균 F1 점수 0.91 및 IoU 점수 0.88을 달성하며, 높은 잡음 강도에 대한 저항성과 기존 화산 데이터셋에 대한 적응력을 보여 주었습니다. 이 연구는 심층 학습을 사용한 지진 이벤트 인식을 위한 새로운 솔루션을 제공하여 실시간 화산 모니터링에 기여할 수 있을 것입니다.



### Normal-GS: 3D Gaussian Splatting with Normal-Involved Rendering (https://arxiv.org/abs/2410.20593)
Comments:
          9 pages, 5 figures, accepted at NeurIPS 2024

- **What's New**: 이번 논문에서는 Normal-GS라는 혁신적인 접근 방식을 통해 3D Gaussian Splatting(3DGS) 렌더링 파이프라인에 Surface Normals(표면 법선 벡터)를 통합하여 높은 시각적 품질과 정확한 기하학적 정보 추정을 동시에 달성하고자 합니다.

- **Technical Details**: Normal-GS는 표면 색상을 법선 벡터와 Integrated Directional Illumination Vector (IDIV)의 곱으로 parameterize하여, 물리 기반 렌더링 방정식을 사용하여 법선과 조명의 상호작용을 모델링합니다. 또한, 앵커 기반 3DGS를 활용하여 메모리 사용을 최적화하고, Optimized Normals 및 Integrated Directional Encoding (IDE)을 활용하여 스페큘러 효과를 정확하게 모델링합니다.

- **Performance Highlights**: 다양한 실험을 통해 Normal-GS는 정확한 Surface Normals(표면 법선) 추정과 실시간 렌더링 성능을 유지하면서도 거의 최첨단 시각적 품질을 달성하는 것으로 나타났습니다.



### Unsupervised Panoptic Interpretation of Latent Spaces in GANs Using Space-Filling Vector Quantization (https://arxiv.org/abs/2410.20573)
- **What's New**: 본 논문에서는 데이터 레이블이나 주석이 달린 샘플을 필요로 하지 않는 새로운 접근 방법인 space-filling vector quantization (SFVQ)를 제안합니다. 이는 모호한 latent space를 해석 가능하게 만드는 데 기여합니다.

- **Technical Details**: SFVQ는 데이터의 조각별 선형 곡선 위에서 양자화하여 latent space의 기본 형태적 구조를 캡처합니다. 본 연구는 pretrained StyleGAN2 및 BigGAN 네트워크의 latent space 모델링에 이 기법을 적용하였습니다.

- **Performance Highlights**: 실험 결과, SFVQ 곡선은 latent space의 일반적인 해석 가능한 모델을 제공하며, 이것이 각각의 생성 요인에 대응하는 latent space의 어떤 부분인지 파악할 수 있게 합니다. 또한, SFVQ의 곡선의 각 선은 이해할 수 있는 이미지 변환을 적용하기 위한 해석 가능한 방향으로 활용될 수 있습니다.



### Detection of adrenal anomalous findings in spinal CT images using multi model graph aggregatio (https://arxiv.org/abs/2410.20568)
- **What's New**: 이 연구에서는 척추 CT 스캔에서 부신 병변을 탐지하기 위한 새로운 컴퓨터 보조 진단 방법론을 제안합니다. 이 연구의 주목할 점은 척추에 집중된 CT 스캔을 사용하여 부신 병리와 같은 다른 비정상 증상을 식별할 수 있다는 것입니다.

- **Technical Details**: 제안된 방법은 서로 다른 작업을 수행하는 세 가지 딥러닝 모델로 구성된 Multi Model Graph Aggregation (MMGA) 방법입니다. 이 구조는 CNN, YOLO V3 및 Graph CNN 모델을 포함하며, 각 모델은 슬라이스 내 부신 병변을 탐지하고 분류하는 데 기여합니다. 머신러닝 모델들은 CT 스캔의 여러 이미지 슬라이스에서 정보를 통합하고 분석합니다.

- **Performance Highlights**: 제안된 방법은 부신 비정상을 분류하는 데 67%의 긍정적 예측 값(PPV)을 달성하였으며, 음성 예측 값(NPV)은 94.4%로 5.6%로 감소시키는 효과를 보였습니다. 제안된 방법의 슬라이스 인덱스 예측 평균 오류는 오른쪽 부신에서 2.25, 왼쪽 부신에서 8.2로 나타났으며, 평균 Intersection Over Union (IOU) 점수는 각각 0.52 및 0.41을 기록했습니다.



### SympCam: Remote Optical Measurement of Sympathetic Arousa (https://arxiv.org/abs/2410.20552)
Comments:
          Accepted for publication at the IEEE-EMBS International Conference on Biomedical and Health Informatics

- **What's New**: 이 연구는 SympCam이라는 새로운 3D convolutional architecture를 통해 얼굴 비디오만으로 사람의 sympathetic arousal(교감 각성)을 예측할 수 있는 방법을 제시합니다. 이는 비침습적인 방식으로 일반 RGB 카메라를 사용하여 스트레스를 측정하는 새로운 가능성을 열어줍니다.

- **Technical Details**: SympCam 모델은 temporal attention module (TAM)을 통합하여 데이터의 시퀀스 처리 효율성을 높입니다. 이 모델은 20명의 참여자로부터 수집된 얼굴과 손 비디오를 기반으로 하며, electrodermal activity (EDA) 및 photoplethysmography (PPG) 측정과 동기화되었습니다. 이 데이터셋은 원거리에서의 sympathetic arousal 예측을 위해 특별히 설계되었습니다.

- **Performance Highlights**: 제안된 방법은 기존 연구보다 48% 더 높은 평균 상관관계 0.77을 달성하였으며, 자주 사용되는 rPPG 방법과 비교해 61% 향상된 90%의 균형 정확도를 보였습니다. 이 연구는 원거리에서의 sympathetic arousal 예측을 위해 다른 연구자들이 사용할 수 있는 최초의 데이터셋 제공을 목표로 합니다.



### Asynchronous Perception Machine For Efficient Test-Time-Training (https://arxiv.org/abs/2410.20535)
Comments:
          Accepted to NeurIPS 2024 Main Track. APM is a step to getting Geoffrey Hinton's GLOM working. Original GLOM paper said: "This paper was quickly hijacked by the need to justify the design decisions". 3 years have passed us. This work provides some justifications and been peer-reviewed and accepted by our peers. A humble blogpost can be found at this https URL

- **What's New**: 이 논문에서는 Asynchronous Perception Machine (APM), 테스트 타임 교육(test-time-training, TTT)을 위한 계산 효율적인 아키텍처를 제안합니다. APM은 이미지를 비대칭적으로 순서에 상관없이 한 번에 하나의 패치를 처리할 수 있으며, 여전히 네트워크에 의미 인식을 인코딩할 수 있습니다.

- **Technical Details**: APM은 기존 TTT 접근 방식보다 경쟁력 있는 성능을 보이며, 데이터 세트 특정 사전 훈련이나 데이터 증강, 프리텍스트 작업 없이 분포 외(out-of-distribution) 이미지를 인식할 수 있습니다. TTT를 수행할 때 APM은 테스트 샘플의 표현을 단 한 번 정제하며, 이후 모든 반복은 이 단일 표현에 대해 과적합(over-fit)하여 데이터 증강이나 프리텍스트 작업을 요구하지 않습니다.

- **Performance Highlights**: APM은 기존 방법에 비해 FLOPs의 수를 거의 반으로 줄일 수 있으며, 16개의 데이터 세트에서 TTT 성능이 0.4%에서 8% 향상되었습니다. 또한, APM은 아키텍처가 간단하여 5개의 층을 가진 단일 MLP와 합성곱(convolutional) 레이어로 구성되어 있으며, 하나의 샘플을 사용하여 의미적으로 클러스터링할 수 있는 능력을 가지고 있습니다.



### Fractal and Turbulent Feature Extraction and NFT Label Generation for Pollock Style Migration Paintings Based on VGG19 (https://arxiv.org/abs/2410.20519)
Comments:
          9 pages, 4 figures

- **What's New**: 이 논문은 심층 학습(deep learning), 프랙탈 분석(fractal analysis), 난류(feature extraction techniques) 기능 추출 기법을 융합하여 Pollock 스타일의 추상 미술 작품을 생성하는 혁신적인 접근 방식을 제안합니다.

- **Technical Details**: MindSpore 심층 학습 프레임워크와 VGG19 모델을 사용하여 이미지의 내용과 스타일 특성을 추출합니다. 내용 손실(content loss), 스타일 손실(style loss), 전체 분산 손실(full variance loss)을 결합하여 스타일 이주(style migration)의 정확성을 보장합니다. 또한, 엣지 추출과 프랙탈 분석을 통해 이미지를 효과적으로 추정하는 차이 상자 개수 방법을 기반으로 한 프랙탈 차원 계산 방법을 구현합니다. Haar 웨이브렛을 사용하여 이미지를 분해하고 다양한 주파수 정보를 추출하여 여러 특성을 결합하여 고유한 NFT(label) 라벨을 생성합니다.

- **Performance Highlights**: 실험 결과, 생성된 작품은 프랙탈 차원과 난류(nonlinearity) 특성의 측면에서 상당한 다양성과 복잡성을 보여주며, 생성된 NFT 태그는 각 디지털 컬렉션의 독창성(uniqueness)과 변조 가능성(tamperability)을 보장합니다. 제안된 방법은 컴퓨터 비전(computer vision), 디지털 신호 처리(digital signal processing), 블록체인 기술(blockchain technology)을 유기적으로 결합하여 디지털 아트워크의 생성 및 인증에 대한 새로운 솔루션을 제공합니다.



### Referring Human Pose and Mask Estimation in the Wild (https://arxiv.org/abs/2410.20508)
Comments:
          Accepted by NeurIPS 2024. this https URL

- **What's New**: 본 논문은 자연어 또는 위치 프롬프트를 통해 이미지 내에서 특정 인물에 대한 자세 및 마스크를 추정하는 Referring Human Pose and Mask Estimation (R-HPM) 작업을 소개합니다. 이 새로운 작업은 조력 로봇 및 스포츠 분석과 같은 인간 중심 응용 프로그램에 큰 잠재력을 가지고 있습니다.

- **Technical Details**: 우리는 RefHuman이라는 대규모 데이터셋을 도입하여 MS COCO 데이터셋의 텍스트 및 위치 프롬프트 주석을 추가하여 확장했습니다. RefHuman은 50,000개 이상의 주석 인스턴스를 포함하고 있으며, UniPHD라는 최초의 종단간 프롬프트 기반 접근 방식을 통해 멀티모달 표현을 추출하고 자세 중심의 계층적 디코더를 적용하여 결과를 생성합니다.

- **Performance Highlights**: UniPHD 접근 방식은 RefHuman val 및 MS COCO val2017에서 최고 성능을 달성하며, 사용자 친화적인 프롬프트 기반으로 품질 높은 결과를 생성함을 실험을 통해 입증했습니다.



### ARLON: Boosting Diffusion Transformers with Autoregressive Models for Long Video Generation (https://arxiv.org/abs/2410.20502)
- **What's New**: 이번 논문에서는 오토회귀 모델과 확산 Transformer를 결합한 새로운 비디오 생성 프레임워크인 ARLON을 제안합니다. ARLON은 긴 비디오 생성에서의 동적 정보와 일관성을 향상시키기 위해 설계되었습니다.

- **Technical Details**: ARLON은 오토회귀(AR) 모델에서 제공되는 거칠고 긴 범위의 시간 정보를 이용하여 DiT 모델을 유도하며, VQ-VAE를 통해 입력 잠재 공간을 압축하여 정보 밀도 및 학습 복잡성을 균형있게 조정합니다. 또한, 적응형 노름 기반의 의미 주입 모듈을 통해 AR 모델에서 발생한 시각 단위를 DiT 모델에 통합합니다.

- **Performance Highlights**: ARLON은 VBench에서 OpenSora-V1.2 기준선을 여덟 개의 메트릭에서 초과 성능을 보였으며, 동적 정도와 미적 품질에서 상당한 개선을 달성했습니다. 또한, ARLON은 최첨단의 긴 비디오 생성 성능을 자랑합니다.



### GrounDiT: Grounding Diffusion Transformers via Noisy Patch Transplantation (https://arxiv.org/abs/2410.20474)
Comments:
          Accepted to NeurIPS 2024. Project Page: this https URL

- **What's New**: 이번 연구에서는 Diffusion Transformers (DiT)를 사용하여 텍스트에서 이미지로의 생성 과정에서 트레이닝 없이 공간적 구속을 처리하는 새로운 기법을 소개합니다. 이 기법은 사용자에게 각 물체의 위치를 더 세밀하게 제어할 수 있는 능력을 부여합니다.

- **Technical Details**: 본 연구는 Tranasformer 구조의 유연성을 활용하여, 각 바운딩 박스에 해당하는 노이즈가 있는 이미지 패치를 생성할 수 있는 DiT의 기능을 설명합니다. 이 과정을 통해 각 지역에 대한 세밀한 제어가 가능해지며, 각 패치는 필요에 따라 원래의 노이즈 이미지로 이식됩니다.

- **Performance Highlights**: HRS 및 DrawBench 벤치마크에서 수행된 실험 결과, 우리의 접근 방식인 GrounDiT는 이전의 트레이닝 없는 공간적 구속 방법에 비해 월등한 성능을 보였으며, 특히 최첨단 접근 방식에 비해 향상된 결과를 보여주었습니다.



### Unlocking Comics: The AI4VA Dataset for Visual Understanding (https://arxiv.org/abs/2410.20459)
Comments:
          ECCV 2024 Workshop Proceedings

- **What's New**: 본 논문은 1950년대의 프랑코-벨기에 만화를 사용하여 여러 개의 작업(예: depth estimation, semantic segmentation, saliency detection, character identification)을 위한 주석이 달린 새 데이터 세트를 제시합니다. 이 데이터 세트는 두 가지 뚜렷하고 일관된 스타일을 포함하며 자연 이미지에서 가져온 객체 개념과 레이블을 통합합니다.

- **Technical Details**: 이 데이터 세트는 두 가지 스타일을 가지고 있으며, 각각은 다양한 객체 개념과 레이블을 포함하여, 캐릭터 식별 및 심도 추정 등의 여러 과제를 위한 주석이 달려 있습니다. 이 자료는 AI4VA 워크숍의 중요한 구성 요소로, 특히 심도와 주목을 연구하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 이 데이터 세트는 컴퓨테이셔널 창의성을 촉진할 가능성이 있으며, 예술 및 스토리텔링 혁신의 디지털화를 위한 새로운 경로를 제공합니다.



### BlinkVision: A Benchmark for Optical Flow, Scene Flow and Point Tracking Estimation using RGB Frames and Events (https://arxiv.org/abs/2410.20451)
Comments:
          Accepted to ECCV 2024. Project Page: this https URL

- **What's New**: BlinkVision은 이벤트 데이터와 RGB 이미지를 모두 포함하여 다양한 카테고리와 밀도 있는 주석을 제공하는 대규모 벤치마크입니다. 이는 픽셀 대응 작업을 위한 종합적인 기준을 제시하여, 기존의 제한적인 벤치마크의 단점을 보완합니다.

- **Technical Details**: BlinkVision은 세 가지 비주얼 모달리티인 최종 RGB 이미지, 깨끗한 RGB 이미지 및 이벤트 데이터를 포함하며, 밀도 있는 픽셀 주석을 제공합니다. 410개의 일상 객체 카테고리를 지원하며, 자연스러운 요소들을 포함한 포토리얼리스틱 데이터를 생성합니다. 이 데이터셋은 Optical Flow, Scene Flow, Point Tracking 등 다양한 픽셀 대응 작업에 활용됩니다.

- **Performance Highlights**: BlinkVision의 평가 결과, 기존 이미지 기반 방법들이 큰 프레임 간격 및 극단적인 조명 조건 하에서 신뢰성이 떨어지는 새로운 도전 과제를 지적하며, 이벤트 기반 방법들이 현재의 잠재력을 충분히 활용하지 못하고 있음을 보여줍니다. 이러한 결과는 알고리즘의 일반화 가능성을 높이는 데 기여할 것입니다.



### CoralSCOP-LAT: Labeling and Analyzing Tool for Coral Reef Images with Dense Mask (https://arxiv.org/abs/2410.20436)
Comments:
          The coral reef labeling and analysis tool is available at this https URL

- **What's New**: 본 연구는 CoralSCOP-LAT라는 자동 및 반자동 산호초 라벨링 및 분석 도구를 제안합니다. 이 도구는 산호초 이미지를 분할하는 데 최적화된 것으로, 분석 정확도와 유연성을 크게 향상시킵니다.

- **Technical Details**: CoralSCOP-LAT는 최신 산호초 기반 모델을 사용하여 이미지 내 산호 지역을 자동으로 세분화 합니다. 이 도구는 사용자가 필요에 따라 라벨 세트를 정의하고, 세분화된 지역을 수정할 수 있는 높은 유연성을 제공합니다.

- **Performance Highlights**: CoralSCOP-LAT는 기존 도구에 비해 6.1배 적은 오류를 보이며, 정확성과 효율성 면에서 각각 5.1배 더 빠릅니다. 이는 대규모 산호초 모니터링 분석을 가능하게 해 줍니다.



### YourSkatingCoach: A Figure Skating Video Benchmark for Fine-Grained Element Analysis (https://arxiv.org/abs/2410.20427)
- **What's New**: 이 논문은 피겨 스케이팅 데이터셋을 활용하여 점프의 공중 시간(air time)을 정확하게 감지하는 새로운 AI 기술을 제안합니다.

- **Technical Details**: 제안된 YourSkatingCoach 데이터셋은 454개의 점프 비디오를 포함하고 있으며, 각 비디오에서 스케이터의 뼈대(skeleton)와 점프 시작 및 종료 프레임의 금본(gold labels)을 제공합니다. 이 연구는 주어진 태스크를 시퀀스 레이블링 문제로 간주하고, Transformer 기반의 모델을 사용하여 공중 시간을 측정합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 강력한 기준선에 대해 유리한 성능을 보였으며, 피겨 스케이팅에서 수집된 미세한 레이블의 일반화 가능성을 다른 스포츠로 확장 적용하였습니다.



### NT-VOT211: A Large-Scale Benchmark for Night-time Visual Object Tracking (https://arxiv.org/abs/2410.20421)
Comments:
          Oral Acceptance at the Asian Conference on Computer Vision (ACCV) 2024, Hanoi, Vietnam

- **What's New**: 최근 발표된 논문에서는 NT-VOT211이라는 새로운 벤치마크를 소개합니다. 이 벤치마크는 시각 객체 추적(VOT) 알고리즘의 평가를 위해 설계되었으며, 211개의 다양한 비디오로 구성되어 있고, 특히 야간 조건에서의 추적 성능 향상을 위한 데이터셋입니다.

- **Technical Details**: NT-VOT211은 211,000개의 잘 라벨링된 프레임과 카메라 움직임, 변형, 빠른 움직임, 움직임 흐림, 작은 목표, 방해물, 가려짐 및 시야 밖 등의 8가지 속성으로 구성되어 있습니다. 이는 낮은 가시성과 이미지 흐림, 방해물 문제를 포함한 야간 추적의 고유한 문제들을 해결하기 위해 설계되었습니다.

- **Performance Highlights**: 42개 다양한 추적 알고리즘을 NT-VOT211에서 평가한 결과, 기존 낮 시간 벤치마크에 비해 낮은 성능을 보였습니다. 이는 야간 조건에서의 추적 알고리즘의 개선 기회를 마련하고, 실제 응용 프로그램에서의 추적 성능 향상을 위한 새로운 방향을 제시합니다.



### Point-PRC: A Prompt Learning Based Regulation Framework for Generalizable Point Cloud Analysis (https://arxiv.org/abs/2410.20406)
Comments:
          accepted by NeurIPS 2024

- **What's New**: 이 논문은 대규모 3D 모델의 3D 도메인 일반화 능력을 조사합니다. 기존의 Prompt Learning 기법을 기반으로 3D 포인트 클라우드 인식 성능을 향상시키기 위한 연구가 진행되었으나, 이러한 개선이 3D 도메인 일반화에는 부정적인 영향을 미친다는 점에 주목했습니다. 새로운 규제 프레임워크를 제안하여, 학습 가능한 프롬프트가 대규모 3D 모델의 일반 지식과 상호작용하여 좋은 일반화를 유지하도록 합니다.

- **Technical Details**: 제안된 프레임워크는 레귤레이션(regulation) 제약을 통해 학습 가능한 프롬프트가 과거의 지식과 상호작용하도록 하며, 세 가지 주요 제약조건을 포함합니다. 첫 번째로, 상호합의 제약(mutual agreement constraint)을 통해 특수 작업의 예측과 일반 지식 간의 일관성을 극대화합니다. 두 번째로, 다양한 텍스트 설명을 활용하여 서로 다른 클래스의 속성을 반영하여 일반화를 향상시킵니다. 마지막으로, 가중 모델 집합 전략을 통해 학습 가능한 프롬프트를 부드럽고 예측 가능하게 업데이트 합니다.

- **Performance Highlights**: 이 연구에서 제안한 방법은 다양한 3D 도메인 일반화 벤치마크에서 명확한 성능 향상을 보여 주며, 일반화 능력과 작업 특정 3D 인식 성능 모두를 동시에 개선합니다. 또한, 기존 벤치마크와는 다른 새로운 3가지 벤치마크를 생성하여 3D 도메인 일반화 평가에 기여합니다. 이 새로운 벤치마크들은 새로운 일반화 기준을 도입하여 향후 연구를 촉진할 것으로 보입니다.



### Depth Attention for Robust RGB Tracking (https://arxiv.org/abs/2410.20395)
Comments:
          Oral Acceptance at the Asian Conference on Computer Vision (ACCV) 2024, Hanoi, Vietnam

- **What's New**: 이번 연구에서는 RGB 비디오 물체 추적 성능을 향상시키기 위해 단안(depth) 정보 활용을 제안합니다. 저자들은 깊이(attention mechanism)를 통해 추적 알고리즘과 통합하는 새로운 프레임워크를 제공하며, RGB-D 카메라를 필요로 하지 않습니다.

- **Technical Details**: 제안된 프레임워크는 모노클(depth estimation) 알고리즘을 활용하여 초기 깊이 맵을 생성하고, Z 커널을 이용하여 이를 개선합니다. 새로운 depth attention 모듈은 기존 RGB 추적 알고리즘에 원활하게 통합될 수 있습니다.

- **Performance Highlights**: 본 연구는 여섯 개의 도전적인 벤치마크에서 실험을 수행하였으며, 기존 강력한 기준 선들에 비해 지속적인 개선을 보여주며 새로운 SOTA( State-Of-The-Art) 성능을 달성하였습니다.



### Lodge++: High-quality and Long Dance Generation with Vivid Choreography Patterns (https://arxiv.org/abs/2410.20389)
Comments:
          Project page: this https URL

- **What's New**: Lodge++는 음악과 특정 장르에 따라 고품질의 초장기 댄스를 생성할 수 있는 새로운 안무 프레임워크입니다. 이 프레임워크는 계산 효율성을 처리하고 복잡한 전 세계적인 안무 패턴을 학습하며 지역적인 춤의 물리적 품질을 유지하는 것을 목표로 합니다.

- **Technical Details**: Lodge++는 두 단계로 구성된 구조를 사용합니다. 첫 번째 단계에서는 글로벌 안무 네트워크가 복잡한 글로벌 안무 패턴을 포착하는 거친 회전형 댄스 원시형태를 생성하고, 두 번째 단계에서는 이를 기반으로 한 댄스 확산 모델이 고품질의 장기 댄스를 병렬로 생성하여 복잡한 안무 패턴에 충실하게 따릅니다. 또한, 물리적 타당성을 높이기 위해 침투 가이드 모듈과 발의 정제 모듈을 사용하여 캐릭터의 자기 침투 문제를 해결하고 발-바닥 접촉을 최적화합니다.

- **Performance Highlights**: Lodge++는 다양한 댄스 장르에 적합한 초장기 댄스를 빠르게 생성할 수 있으며, 잘 조직된 글로벌 안무 패턴과 고품질의 지역적 움직임을 보장합니다. 이 방법은 댄스와 음악의 일치를 더 잘 유지할 수 있어, 이전의 기법들에 비해 실질적인 응용 가능성이 더 높아졌습니다.



### Addressing the Pitfalls of Image-Based Structural Health Monitoring: A Focus on False Positives, False Negatives, and Base Rate Bias (https://arxiv.org/abs/2410.20384)
- **What's New**: 이 연구는 이미지 기반 구조 건강 모니터링(Structural Health Monitoring, SHM) 기술이 구조적 손상을 탐지하는 데 가지고 있는 한계를 탐구하였습니다.

- **Technical Details**: 이 연구는 머신 러닝(Machine Learning)과 컴퓨터 비전(Computer Vision)을 활용하여 이미지 기반 SHM이 수동 검사에 비해 확장 가능하고 효율적인 대안이 될 수 있음을 보여줍니다. 그러나 실제 손상의 발생 확률이 낮을 경우, 잘못된 긍정(false positives) 및 잘못된 부정(false negatives)과 같은 문제로 인해 신뢰성이 저해됩니다. 연구는 베이지안(Bayesian) 분석과 자주론(frequentist) 접근을 통해 손상 탐지 시스템의 정밀도를 평가하고, 손상의 발생이 드문 경우에도 높은 정확도를 가진 모델이 오해를 불러일으킬 수 있음을 드러냈습니다.

- **Performance Highlights**: 연구에서는 여러 데이터 소스를 결합한 하이브리드 시스템(hybrid systems), 중요한 평가를 위한 사람의 참여(human-in-the-loop) 방법, 그리고 훈련 데이터의 품질 향상 등을 통한 이러한 한계를 완화하기 위한 전략을 논의합니다. 이 연구의 결과는 이미지 기반 SHM 기술의 실용적 적용 가능성에 대한 중요한 통찰을 제공하며, 실제 인프라 모니터링을 위해 그들의 잠재력과 한계를 강조합니다.



### Open-Vocabulary Object Detection via Language Hierarchy (https://arxiv.org/abs/2410.20371)
Comments:
          NeurIPS 2024 Camera Ready

- **What's New**: 최근 약한 감독(imagen-level supervision)을 활용한 일반화 가능한 객체 탐지(generalizable object detection)에 대한 연구가 증가하고 있습니다. 본 논문에서는 Language Hierarchical Self-training (LHST)을 통해 약하게 감독된 탐지기 훈련에 언어 계층을 도입하여 더 일반화된 탐지기를 학습합니다. 이를 통해 이미지 수준의 레이블과 박스 수준의 레이블 간의 불일치를 완화하고, 더 풍부한 감독(supervision)을 제공합니다.

- **Technical Details**: LHST는 WordNet의 언어 계층을 활용하여 이미지 수준의 레이블을 확장하고, 자기 훈련(self-training) 과정에서의 공동 정규화(co-regularization)를 가능하게 합니다. 또한, Language Hierarchical Prompt Generation (LHPG)을 설계하여 샘플링된 언어 계층을 프롬프트 생성에 적용합니다. 이는 훈련과 테스트 간의 어휘 격차(vocabulary gaps)를 메우는 데 도움을 줍니다.

- **Performance Highlights**: 제안된 기술들은 14개의 널리 연구된 객체 탐지 데이터셋에서 지속적으로 우수한 일반화 성능(generalization performance)을 보여줍니다. DetLH는 약한 감독 하의 객체 탐지에서 스스로 상기(label-to-box assignment) 및 자율 학습(self-training) 기법을 결합하여 감지 성능을 극대화합니다.



### RopeTP: Global Human Motion Recovery via Integrating Robust Pose Estimation with Diffusion Trajectory Prior (https://arxiv.org/abs/2410.20358)
Comments:
          Accepted by WACV 2025 (Round 1)

- **What's New**: RopeTP는 강인한 pose estimation과 diffusion trajectory prior를 결합하여 비디오에서 전역적 인간 동작을 재구성하는 새로운 프레임워크입니다. 이 방법은 occluded body parts의 포즈를 정확하게 추론하기 위해 계층적 주의 메커니즘을 활용하여 상황 인식을 개선합니다.

- **Technical Details**: RopeTP는 다중 스케일의 시각적 단서를 활용하여 occluded body parts를 재구성하고, diffusion generative model을 도입하여 재구성된 joint poses에 따라 전역 모션 경로를 재추정합니다. 이 방법은 슬램(SLAM) 기반 초기 카메라 추정과 extensive 최적화에 의존하지 않고도 더 정확하고 현실적인 경로를 제공합니다.

- **Performance Highlights**: RopeTP는 3DPW 및 Human3.6M과 같은 표준 데이터셋에서 우수한 성능을 보여주며, occlusion이 있는 복잡한 시나리오에서도 특히 두드러진 성과를 도출했습니다. 실험 결과, 기존 방법론들보다 전반적으로 뛰어난 성능을 나타냈습니다.



### Idempotent Unsupervised Representation Learning for Skeleton-Based Action Recognition (https://arxiv.org/abs/2410.20349)
Comments:
          ECCV 2024

- **What's New**: 이 논문에서는 새로운 골격 기반 비가역 생성 모델(IGM)을 제안하여 인식 작업에서의 성능을 향상시킵니다. 기존의 생성 모델들이 인식과 관련 없는 중복 정보를 포함하고 있다는 문제를 해결하고자 합니다.

- **Technical Details**: 우리는 이론적으로 생성 모델과 최대 엔트로피 코딩(maximum entropy coding) 간의 동등성을 보여줍니다. 이를 통해 특징들이 더 간결해지도록 대비 학습(contrastive learning)을 도입하고, 골격 데이터의 공간적 및 시간적 특징을 유지하기 위해 비가역성(idempotency) 제약을 도입합니다. 이 모델은 고차원 표현을 촉진하기 위해 인코더와 생성기 기능을 융합하는 어댑터를 채택합니다.

- **Performance Highlights**: 실험 결과, NTU 60 xsub 데이터셋에서 성능이 84.6%에서 86.2%로 향상되었습니다. 또한 제로샷(adaptation) 시나리오에서도 이전에 인식할 수 없었던 경우에서 유망한 결과를 달성하여 모델의 효율성을 입증하였습니다.



### UTSRMorph: A Unified Transformer and Superresolution Network for Unsupervised Medical Image Registration (https://arxiv.org/abs/2410.20348)
Comments:
          13pages,10 figures

- **What's New**: 의료 영상 분석에서 복잡한 이미지 등록은 주요 이슈이며, 본 연구에서는 ConvNet과 Transformer 기반 방법의 장점을 통합한 새로운 비지도 이미지 등록 기법인 통합 Transformer 및 초해상도 네트워크(UTSRMorph)를 제안합니다.

- **Technical Details**: UTSRMorph는 Fusion Attention Block(FAB)과 Overlapping Attention Block(OAB)을 포함한 인코더 구조를 채택하여 ConvNets와 Transformer의 장점을 결합하고, Self-Attention 메커니즘을 통해 이미지 쌍 간의 관계를 추출합니다. 또한, 디코더에서는 초해상도(Superresolution) 모듈을 사용하여 로우 해상도(Low-resolution) 특징으로부터 고해상도(High-resolution) 변형 필드를 생성합니다.

- **Performance Highlights**: UTSRMorph는 3D 뇌 MR(OASIS, IXI) 및 MR-CT 데이터셋에서 최신 등록 방법들과 비교했을 때 질적 및 양적 측면에서 상대적으로 우수한 성능을 보였습니다.



### Historical Test-time Prompt Tuning for Vision Foundation Models (https://arxiv.org/abs/2410.20346)
Comments:
          NeurIPS 2024 Camera Ready

- **What's New**: 이 논문에서는 HisTPT(History Test-time Prompt Tuning)라는 새로운 기술을 제안합니다. 이는 테스트 샘플로부터 지속적으로 학습한 정보를 기억하고, 이를 기반으로 강력한 테스트 시 프롬프트 튜닝을 가능하게 합니다.

- **Technical Details**: HisTPT는 세 가지 유형의 지식 은행(local knowledge bank, hard-sample knowledge bank, global knowledge bank)을 도입하며, 각 은행은 다른 메커니즘을 통해 지식을 효과적으로 기억하고 프롬프트를 최적화합니다. 또한, 적응형 지식 검색 메커니즘을 포함하여 각 테스트 샘플의 예측 정규화를 수행합니다.

- **Performance Highlights**: HisTPT는 이미지 분류, 의미 분할, 객체 탐지와 같은 다양한 시각적 인식 작업에서 우수한 프롬프트 튜닝 성능을 보여주며, 도메인이 지속적으로 변할 때도 일관된 성과를 보입니다.



### R-LLaVA: Improving Med-VQA Understanding through Visual Region of Interes (https://arxiv.org/abs/2410.20327)
Comments:
          11 pages, 7 figures, submitted to NAACL 2025

- **What's New**: 본 논문에서는 R-LLaVA를 소개하여 생물의학 시각 질문 답변(Med-VQA) 이해를 강화하고, 의학적인 주석을 이미지 공간에 통합하여 시각적인 관심 영역(Regions of Interest)을 활용합니다.

- **Technical Details**: R-LLaVA는 CLIP을 사용하여 의사의 간단한 주석(bounding boxes)을 이미지에 직접 주입함으로써, 모델 훈련 시 이러한 주석된 시각 영역을 LLaVA 모델에 넣어 더욱 빠르고 정확하게 생물의학적 질문을 처리하도록 합니다. 이는 ROI(Regions of Interest) VQA 쌍을 작성하여 기존 데이터셋을 재구성하여 모델의 시각적 이해 능력을 향상시키는 전략을 포함합니다.

- **Performance Highlights**: R-LLaVA는 네 개의 대규모 Med-VQA 데이터셋에서 기존의 최첨단 방법들보다 일관되게 우수한 성능을 보여주며, 시각적 관심 영역에 집중하는 것이 생물의학적 VQA 이해를 향상시키는 데 긍정적인 영향을 미친다는 것을 입증합니다.



### Few-shot Open Relation Extraction with Gaussian Prototype and Adaptive Margin (https://arxiv.org/abs/2410.20320)
Comments:
          30 pages, 4 figures

- **What's New**: 본 논문에서는 새로운 프레임워크 GPAM(Gaussian Prototype and Adaptive Margin)을 제안하여, UNKNOWN 클래스가 포함된 few-shot 관계 추출(Few-Shot Relation Extraction, FsRE) 문제를 해결하고자 합니다. GPAM은 Semi-factual representation, GMM-prototype metric learning과 Decision boundary learning의 세 가지 모듈로 구성되어 있습니다.

- **Technical Details**: GPAM은 Gaussian 분포와 계층적 마진을 기반으로 한 새로운 프로토타입 거리 측정 전략을 채택하여 few-shot 과적합 문제를 완화하고, 결정 경계를 더 정밀하게 학습합니다. Contrastive learning loss를 사용하여 알려진 클래스와 UNKNOWN 클래스 간의 경계를 보다 정확하게 구분합니다.

- **Performance Highlights**: FewRel 데이터셋에서의 실험 결과, GPAM은 기존의 프로토타입 방법들보다 우수한 성능을 보여주며 state-of-the-art 결과를 달성했습니다. 특히 GPAM은 few-shot 문제와 NOTA 경계 혼동 문제를 성공적으로 해결합니다.



### Wavelet-based Mamba with Fourier Adjustment for Low-light Image Enhancemen (https://arxiv.org/abs/2410.20314)
Comments:
          18 pages, 8 figures, ACCV2024

- **What's New**: 본 논문에서는 Low-Light Image Enhancement (LLIE)를 위한 새로운 모델 WalMaFa를 제안합니다. WalMaFa는 Wavelet 기반 Mamba 블록(WMB)과 Fast Fourier Adjustment 블록(FFAB)으로 구성되어 있습니다. 이 모델은 Encoder-Latent-Decoder 구조를 채택하여 이미지의 전반적인 밝기를 향상시키고, 지역 세부사항을 조정합니다.

- **Technical Details**: WalMaFa는 Wavelet 변환의 저주파 정보와 Fourier 변환의 위상 정보를 결합하여 글로벌 밝기와 로컬 세부사항을 효과적으로 처리합니다. WMB는 Encoder와 Decoder에서 글로벌 밝기를 향상시키는 데 사용되며, FFAB은 Latent에서 로컬 텍스처 세부사항을 미세 조정합니다.

- **Performance Highlights**: 실험 결과, WalMaFa는 기존 LLIE 모델들보다 뛰어난 성능을 보였으며, 적은 계산 리소스를 소모하면서도 더 빠른 속도를 기록했습니다. 주관적인 사용자 연구를 통해 15명의 참여자들이 WalMaFa의 시각적 품질이 우수하다고 평가했습니다.



### GUMBEL-NERF: Representing Unseen Objects as Part-Compositional Neural Radiance Fields (https://arxiv.org/abs/2410.20306)
Comments:
          7 pages. Presented at ICIP2024

- **What's New**: Gumbel-NeRF는 Mixture-of-Expert (MoE) 기반의 NeRF 모델로, 새로운 객체의 시각화 시 일어나는 저해상도 문제를 해결하기 위해 hindsight expert selection 메커니즘을 도입하였습니다.

- **Technical Details**: Gumbel-NeRF는 기술적으로 simple maximum pooling을 사용하여 MoE 구조의 전문가 선택 과정을 개선하고, 전문가별 코드를 도입하여 자동차의 개별 부위를 모형화 하도록 합니다. 이 과정에서 모양의 연속성을 보장하며 Gumbel-NeRF는 기존 모델인 Switch-NeRF과 CodeNeRF의 조합에서 발생하는 문제를 해결합니다.

- **Performance Highlights**: SRN cars 데이터셋을 활용한 실험 결과 Gumbel-NeRF는 다양한 이미지 품질 지표에서 기존 모델들보다 우수한 성능을 보였습니다.



### Deep Learning, Machine Learning -- Digital Signal and Image Processing: From Theory to Application (https://arxiv.org/abs/2410.20304)
Comments:
          293 pages

- **What's New**: 이 논문은 기계학습(Machine Learning, ML) 및 심층학습(Deep Learning, DL)을 활용한 디지털 신호 처리(Digital Signal Processing, DSP) 및 디지털 이미지 처리(Digital Image Processing, DIP) 분야의 혁신적인 응용을 소개합니다. 특별히 이미지 개선, 필터링 기법, 패턴 인식 분야에서의 변화를 강조합니다.

- **Technical Details**: 논문에서는 이산 푸리에 변환(Discrete Fourier Transform, DFT), Z-변환(Z-Transform), 푸리에 변환(Fourier Transform) 방법론을 통합하여 AI 기반 작업에 필요한 데이터 변조 및 특징 추출을 위한 강력한 처리를 가능하게 합니다. Python을 이용하여 실시간 데이터 처리를 최적화하는 알고리즘 구현을 설명하고 있습니다.

- **Performance Highlights**: 이 연구는 ML 및 DL의 잠재력이 DSP 및 DIP 방법론을 발전시키며 인공지능, 자동화된 특징 추출, 다양한 도메인에 걸친 응용에 기여할 수 있음을 보여줍니다. 그 결과는 확장 가능하고 고성능의 솔루션을 위한 기초가 됩니다.



### Harmony4D: A Video Dataset for In-The-Wild Close Human Interactions (https://arxiv.org/abs/2410.20294)
Comments:
          NeurIPS 2024

- **What's New**: Harmony4D 데이터셋은 다양한 환경에서의 인간 간의 접촉 활동을 기록한 멀티 뷰 비디오 데이터셋으로, 기존 데이터셋의 다양성 부족 문제를 해결하고자 합니다. 이 데이터셋은 야외에서 수집되었으며, 레슬링, 댄스, MMA와 같은 활동을 포함하고 있습니다.

- **Technical Details**: Harmony4D는 1.66M 이미지와 3.32M 인간 인스턴스를 포함하며, 20개 이상의 동기화된 카메라에서 촬영된 208개의 비디오 시퀀스로 구성됩니다. 이 데이터셋은 인간 감지, 추적, 2D/3D 포즈 추정, 메쉬 복구에 대한 주석을 제공합니다. 혁신적인 markerless(tracking에 마커를 사용하지 않음) 알고리즘을 통해 3D 포즈를 추적합니다.

- **Performance Highlights**: Harmony4D 데이터셋에서 기존의 최첨단 방법들에 대한 엄격한 평가를 수행하였으며, 기존 방법들이 인간 간의 접촉 상호작용 모델링에 있어서 중대한 한계가 있음을 확인했습니다. HMR2.0 모델을 미세 조정한 결과, 심각한 가림과 접촉이 있는 장면에서 54.8% PVE(퍼센트 위치 오차의 비율)를 향상시킨 성능을 보여주었습니다.



### MarDini: Masked Autoregressive Diffusion for Video Generation at Sca (https://arxiv.org/abs/2410.20280)
Comments:
          Project Page: this https URL

- **What's New**: MarDini 모델은 비디오 생성의 효율성을 극대화하기 위한 새로운 비디오 확산 모델로, 마스크 자동 회귀(MAR)와 확산 모델(DM)의 장점을 통합하여 경량화된 네트워크 설계를 적용합니다. 이 모델은 다양한 마스킹 전략을 통해 비디오 생성 작업을 유연하게 처리할 수 있으며, 비디오 인터폴레이션, 이미지-비디오 변환 및 비디오 확장 등 다양한 작업을 지원합니다.

- **Technical Details**: MarDini는 MAR 플래닝 모델과 경량화된 DM으로 구성된 비대칭 네트워크 아키텍처를 가지고 있습니다. MAR 모델은 저해상도 입력 프레임을 사용하여 계획 신호를 생성하며, DM은 이 신호를 기반으로 한 고해상도 비디오 프레임을 생성합니다. 효율적인 훈련 배치와 프로그레시브 트레이닝 전략을 통해 MAR와 DM을 병렬로 훈련시키며, 일반적인 텍스트-이미지 또는 텍스트-비디오 사전 학습에 대한 의존 없이 비디오 생성 모델을 훈련할 수 있습니다.

- **Performance Highlights**: MarDini는 몇 가지 추론 단계 내에 탁월한 성능을 보이며, 기존의 고비용 이미지-비디오 모델과 동등한 수준의 비디오 생성을 효율적으로 수행합니다. 이 모델은 비디오 인터폴레이션에서 새로운 최첨단 성능을 설정하며, 메모리 사용량이 낮아 복잡한 모션 동역학을 효과적으로 모델링할 수 있는 가능성을 나타냅니다.



### You Never Know: Quantization Induces Inconsistent Biases in Vision-Language Foundation Models (https://arxiv.org/abs/2410.20265)
Comments:
          Workshop paper at NeurIPS 2024 RBFM. 6 pages, 3 figures

- **What's New**: 이 논문은 quantization이 foundation vision-language 모델의 사회적 공정성에 미치는 영향을 평가한 연구로, 이전의 unimodal 모델에 대한 발견과 달리, quantization을 통해 생성된 여러 compressed 모델에서 지속적인 편향의 변화가 관찰되지 않았다는 점이 새롭게 밝혀졌습니다.

- **Technical Details**: 이 연구에서는 8-bit 및 4-bit quantization 방법을 사용하여 다양한 CLIP 모델 변형을 평가하였으며, 각각의 quantization 방법은 메모리 절약과 추론 지연 시간 단축을 가능하게 합니다. 특히, 모델의 공정성을 측정하기 위해 FACET 및 FairFace 데이터셋을 사용하여 성별, 나이, 인종 등의 민감한 특성에 대한 결과를 분석하였습니다.

- **Performance Highlights**: 연구 결과, quantized 모델에서 개별 모델은 편향을 보여주었으나, 전체 모델 집단에서는 편향의 방향과 크기에 일관된 변화가 없었습니다. 이는 quantization이 multimodal context에서 공정성에 미치는 영향이 더욱 복잡할 수 있음을 시사합니다.



### Adaptive Video Understanding Agent: Enhancing efficiency with dynamic frame sampling and feedback-driven reasoning (https://arxiv.org/abs/2410.20252)
- **What's New**: 이번 연구에서는 장시간 비디오 콘텐츠 이해의 효율성과 효과성을 향상시키기 위해 대규모 언어 모델(LLMs)을 활용한 에이전트 기반 접근법을 제안합니다. 주요 기법으로는 쿼리 적응형 프레임 샘플링(query-adaptive frame sampling)을 도입하여, 관련성 높은 프레임만을 실시간으로 처리하여 기존 방법의 단점을 극복하고자 했습니다.

- **Technical Details**: 제안하는 방법은 LLM 기반의 에이전트가 특정 문맥과 쿼리에 기반하여 동적으로 샘플 프레임을 결정하도록 설계되었습니다. 자가 반영(self-reflective) 능력을 활용하여 에이전트의 추론 능력을 강화하고, 이전 경험을 저장하고 활용할 수 있는 장기 기억(long-term memory) 기능도 통합하였습니다.

- **Performance Highlights**: 여러 비디오 이해 벤치마크에서 평가한 결과, 제안한 방법이 기존 방법들보다 더 높은 정확성과 더 낮은 프레임 액세스를 기록하며 성능을 향상시키는 것을 확인했습니다.



### Enhancing CNN Classification with Lamarckian Memetic Algorithms and Local Search (https://arxiv.org/abs/2410.20234)
Comments:
          Accepted in IEEE SPARC 2024

- **What's New**: 이 논문에서는 이미지 분류 네트워크를 위한 인구 기반 메타휴리스틱 최적화 알고리즘을 탐구하고, 로컬 탐색 기능을 통합한 두 단계 훈련 기법을 기반으로 한 새로운 접근 방식을 제안합니다. 이를 통해 ADAM과 같은 기존의 그래디언트 기반 기법보다 높은 정확도와 계산 효율성을 달성하였습니다.

- **Technical Details**: 제안된 방법은 기본 CNN 아키텍처와 관련된 MLP(다층 퍼셉트론) 레이어의 가중치 최적화에 초점을 맞추며, GA(유전자 알고리즘), PSO(입자 군집 최적화), 그리고 새로운 메메틱 알고리즘을 포함합니다. CNN은 Convolutional, Pooling, 그리고 완전 연결층으로 구성되어 있으며, 이미지의 특징을 추출합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 ADAM 등 기존 방법보다 더욱 높은 정확도를 보였으며, 특히 높은 계산 복잡성과 많은 학습 가능한 파라미터를 가진 상황에서 뛰어난 성능을 발휘했습니다. 이 방법은 CNN에서 가중치 최적화를 위한 견고한 대안을 제공합니다.



### CAVE: Classifying Abnormalities in Video Capsule Endoscopy (https://arxiv.org/abs/2410.20231)
- **What's New**: 이번 연구에서는 복잡한 이미지 데이터셋에서 분류 정확도를 향상시키기 위한 앙상블 기반 접근 방식을 탐구합니다. Convolutional Block Attention Module (CBAM)와 Deep Neural Network (DNN)를 활용하여 각 모델의 독특한 특징 추출 능력을 활용해 전체 정확도를 개선합니다.

- **Technical Details**: 본 연구에 사용된 데이터셋은 특정 이상이나 건강한 대조군을 나타내는 10개의 서로 다른 클래스를 포함한 고해상도 이미지로 구성됩니다. 각 이미지는 224x224 픽셀의 균일한 해상도로 전처리되었습니다. 이러한 클래스 불균형을 해결하기 위해 데이터 증강(data augmentation)이 필수적이었으며, 이를 통해 각 클래스의 최소 샘플 수를 7,500으로 늘리기 위해 여러 변환을 적용했습니다. 또한 ResNet50 기반의 오토인코더(autoencoder)를 통해 이미지의 특성 표현을 추출하고 이미지를 재구성했습니다.

- **Performance Highlights**: 실험 평가 결과, 제안된 앙상블 방식은 도전적인 클래스에서 높은 정확도와 견고성을 보여주었습니다. 이는 컴퓨터 비전 작업의 광범위한 응용 가능성을 제시하고 있으며, 특히 캡슐 내시경과 같은 의료 영상 진단 분야에서 큰 가능성을 보이고 있습니다.



### An Efficient Watermarking Method for Latent Diffusion Models via Low-Rank Adaptation (https://arxiv.org/abs/2410.20202)
- **What's New**: 본 논문은 Latent Diffusion Models (LDMs)를 위한 효율적인 워터마킹 방법을 제안합니다. 본 방법은 Low-Rank Adaptation (LoRA)을 기반으로 하여, 기존 모델의 가중치는 고정하고 훈련 가능한 저차원 행렬을 추가함으로써 워터마크를 삽입합니다.

- **Technical Details**: 본 연구에서는 워터마크 삽입 과정에서 발생할 수 있는 모델 성능 저하를 최소화하기 위해 동적 손실 가중치 조정 알고리즘을 제안합니다. 이를 통해 생성 작업과 워터마크 삽입 작업의 균형을 맞추고, 추가적인 파라미터로 인해 발생할 수 있는 메모리 부담을 줄였습니다. 논문에서 제안하는 접근법은 LoRA를 VAE 디코더의 특정 층에 적용하여, 높은 계층 기능을 직접 수정하여 워터마크를 효과적으로 삽입합니다.

- **Performance Highlights**: 실험 결과는 제안된 방법이 빠른 워터마크 삽입을 보장하며, 워터마크의 비트 오류율은 매우 낮고, 생성된 이미지의 품질이 높으며, 검증을 위한 거짓 음성율 (FNR)은 0임을 보여줍니다.



### Image Generation from Image Captioning -- Invertible Approach (https://arxiv.org/abs/2410.20171)
Comments:
          Accepted as Tiny Paper at ICVGIP 2024 conference

- **What's New**: 현재 연구에서는 이미지 자막 생성(image captioning)과 이미지 생성(image generation)을 위한 모델을 단일 작업으로 훈련하여 두 가지 작업을 동시에 수행할 수 있는 방법을 제시합니다. 이를 위해 이미지와 텍스트 임베딩(embedding) 간의 일대일 맵(mapping)을 학습하는 가역(invertible) 모델을 차용합니다. 모델이 이미지 자막 생성 작업에서 효율적으로 훈련되고 나면, 추가 훈련 없이도 주어진 텍스트에 대해 새로운 이미지를 생성할 수 있습니다.

- **Technical Details**: 이 논문에서는 이미지와 해당 텍스트 주석(text annotation) 간의 관계를 학습하는 가역 함수 f를 훈련합니다. 입력 이미지 I는 이미지 인코더(image encoder) Eℐ( . )를 통해 벡터 표현으로 변환되며, 텍스트 T는 텍스트 인코더 E𝒯( . )를 통해 같은 방식으로 변환됩니다. 이 과정에서 f:Eℐ⁢(I)↦E𝒯⁢(T)로 정의되는 가역적인 함수가 설계됩니다. 모든 가중치 행렬(weight matrices)과 비선형 활성화 함수(non-linear activation functions)가 가역적이면, 네트워크 f는 가역적이 됩니다. 이를 위해 트라이앵귤러(t triangular) 행렬을 사용하며, 가중치 행렬 W는 L과 U라는 두 개의 트라이앵귤러 행렬의 곱으로 생성됩니다.

- **Performance Highlights**: 제안된 가역 신경망은 다양한 함수의 타겟 함수(target function) 학습과 역함수(inversion) 예측에서 낮은 오류를 달성하여, 가역 매핑 학습의 효율성을 입증합니다. 이 모델은 Flickr30k 이미지 자막 데이터셋을 이용해 훈련되며, 텍스트에 대한 적절한 이미지를 생성할 수 있는 능력을 보여주고 있습니다.



### Diff-CXR: Report-to-CXR generation through a disease-knowledge enhanced diffusion mod (https://arxiv.org/abs/2410.20165)
- **What's New**: 이번 연구에서는 질병 지식이 강화된 Diffusion 기반의 텍스트-이미지(TTI) 생성 프레임워크인 Diff-CXR를 제안한다. 기존의 방법들은 의료 데이터의 고유 특성 때문에 생성 성능이 제한적이었으나, Diff-CXR은 이를 개선하기 위해 노이즈 필터링 전략과 비전-언어 모델을 통합하여 보고서에서 Chest-Xray(CXR) 생성을 가능하게 한다.

- **Technical Details**: Diff-CXR는 Latent Noise Filtering Strategy(LNFS)를 통해 노이즈 데이터를 효과적으로 제거하고, Adaptive Vision-Aware Textual Learning Strategy(AVA-TLS)를 통해 도메인 특화된 비전-언어 모델에서 중요한 보고서 임베딩을 학습한다. 또한, 질병 지식을 주입하기 위한 메커니즘을 통해 모델의 성능을 개선한다고 설명된다.

- **Performance Highlights**: 실험 결과, Diff-CXR는 MIMIC-CXR와 IU-Xray에서 FID 점수와 mAUC 점수에서 기존 SOTA 의료 TTI 방법들보다 각각 33.4% / 8.0% 및 23.8% / 56.4% 성능을 개선하며, 29.641 GFLOPs의 가장 낮은 계산 복잡도로 효율적인 결과를 보여준다.



### Your Image is Secretly the Last Frame of a Pseudo Video (https://arxiv.org/abs/2410.20158)
Comments:
          18 pages, 7 figures

- **What's New**: 본 논문에서는 디퓨전 모델(Diffusion models)의 자가 감독(self-supervision) 정보를 활용하여 전통적인 이미지 생성 모델의 성능을 향상시킬 수 있는 방법을 제안합니다. 구체적으로는 원본 이미지에 데이터 증강(data augmentation)을 적용하여 생성한 유사 비디오(pseudo video)를 사용하여 비디오 생성 모델로 확장합니다.

- **Technical Details**: 디퓨전 모델은 계층적 변량 자동 인코더(Hierarchical Variational Autoencoders, HVAEs)의 특수한 경우로, 노이즈를 단계적으로 이미지로 변환하는 과정에서 상당히 많은 중간 상태(intermediate states)를 활용합니다. 이 논문에서는 이러한 중간 상태에 자가 감독 정보를 포함시키는 것이 이미지 생성 성능 향상에 중요한 역할을 한다고 가설을 세우고, 유사 비디오를 통해 이를 개선하는 접근법을 탐색합니다.

- **Performance Highlights**: CIFAR10과 CelebA 데이터셋에서 실험적으로 유사 비디오를 통한 이미지 생성 품질이 향상되었음을 확인하였습니다. 이러한 접근법은 VQVAE와 개선된 DDPM 등 다양한 이미지 생성 모델에 적용 가능하며, 성능 개선에 기여할 수 있는 새로운 방법을 제시합니다.



### Human-Object Interaction Detection Collaborated with Large Relation-driven Diffusion Models (https://arxiv.org/abs/2410.20155)
Comments:
          NeurIPS 2024

- **What's New**: DIFFUSIONHOI라는 새로운 HOI 탐지기가 도입되었습니다. 이는 텍스트-이미지 diffusion 모델을 활용하여 사람과 객체 간의 상호작용을 탐지하고, 중급 및 저급 비주얼 개념을 효과적으로 인식할 수 있는 장점을 가지고 있습니다.

- **Technical Details**: DIFFUSIONHOI는 인간과 객체 간의 관계 패턴을 학습하기 위해 inversion 기반 전략을 개발했습니다. 이 방법은 관계 임베딩을 텍스트 프롬프트로 사용하여 특정 상호작용을 묘사하는 이미지를 생성하는 데 도움을 줍니다. 또한, 이를 통해 HOI와 관련된 정보를 이미지에서 쉽게 추출할 수 있습니다.

- **Performance Highlights**: DIFFUSIONHOI는 HICO-DET와 V-COCO 데이터셋에서 최고 성능을 기록했으며, SWiG-HOI의 제로샷 HOI 발견 설정에서 최대 6.43% mAP 향상을 달성했습니다.



### Detection-Guided Deep Learning-Based Model with Spatial Regularization for Lung Nodule Segmentation (https://arxiv.org/abs/2410.20154)
- **What's New**: 이 논문에서는 폐 결절(segmentation of lung nodules)을 CT 이미지에서 효과적으로 세분화하는 새로운 모델을 제안합니다.

- **Technical Details**: 제안된 모델은 deep learning(딥 러닝) 프레임워크를 활용하여 세분화(segmentation)와 분류(classification) 과정을 통합합니다. 특히 feature combination blocks(특징 결합 블록)을 사용하여 세분화와 분류 구성 요소 간의 정보 공유를 촉진하며, 분류 결과를 사전 정보(prior)로 활용하여 예측된 결절의 크기 추정을 정교화합니다. 또한 공간 정규화 기법(spatial regularization technique)을 통합하여 정밀도를 향상시킵니다. 훈련 데이터셋의 제한된 문제를 해결하기 위해 최적의 transfer learning 전략을 개발하여 특정 레이어를 고정(freeze)하여 성능을 개선합니다.

- **Performance Highlights**: 제안된 모델은 다른 일반적으로 사용되는 모델에 비해 목표 결절을 더 정확하게 포착할 수 있는 것으로 나타났습니다. transfer learning을 적용함으로써 성능은 더욱 향상되어, 민감도(sensitivity) 점수는 0.885, Dice 점수는 0.814을 달성하였습니다.



### AdaNeg: Adaptive Negative Proxy Guided OOD Detection with Vision-Language Models (https://arxiv.org/abs/2410.20149)
Comments:
          NIPS 2024 Camera Ready, Codes are available at \url{this https URL}

- **What's New**: 본 연구에서는 OOD(out-of-distribution) 샘플을 효과적으로 식별하기 위한 새로운 방법으로 	extit{adaptive negative proxies}를 도입합니다. 이는 실제 OOD 이미지를 탐색하여 테스트 중에 동적으로 생성되는 부정 프록시입니다.

- **Technical Details**: 우리의 접근법은 테스트 이미지에서 차별적인 특징을 캐시하는 feature memory bank를 활용하여 OOD 데이터셋에 대해 더 잘 정렬된 프록시를 생성하는 방법을 제안합니다. task-adaptive proxies 및 sample-adaptive proxies를 통해 각각 특징적인 데이터셋과 샘플 수준의 세부 정보를 캡처합니다.

- **Performance Highlights**: AdaNeg는 대규모 ImageNet 벤치마크에서 2.45% AUROC 향상과 6.48% FPR95 감소를 달성하며, 기존 방법들에 비해 우수한 성능을 보여줍니다. 우리의 방법은 training-free 및 annotation-free를 유지하며 빠른 테스트 속도를 자랑합니다.



### Semantic Feature Decomposition based Semantic Communication System of Images with Large-scale Visual Generation Models (https://arxiv.org/abs/2410.20126)
Comments:
          13 pages, 13 figures

- **What's New**: 새로운 패러다임으로, Semantic Feature Decomposition (SeFD)를 기반으로 한 이미지 통신 시스템을 제안합니다. 이 시스템은 의미적 통신과 대규모 시각 생성 모델의 통합을 목표로 하며, 고성능의 해석 가능하고 조정 가능한 이미지 통신을 구현합니다.

- **Technical Details**: TCSCI (Texture-Color based Semantic Communication system of Images)는 이미지를 자연어 설명, 질감 및 색상 의미적 특징으로 분해하여 전송합니다. 전송 중 기능들은 무선 채널을 통해 전송되며, 수신자는 대규모 시각 생성 모델을 사용하여 수신된 특징을 통해 이미지를 복원합니다. 이 시스템은 극단적인 압축과 뛰어난 노이즈 저항 능력을 가지며, 전송 과정의 해석 가능성과 편집 가능성을 보장합니다.

- **Performance Highlights**: TCSCI는 전통적인 이미지 통신 시스템과 기존의 의미적 통신 시스템들을 초월하는 성능을 보이며, 극단적인 압축에서도 높은 시각적 유사성을 유지합니다. 또한, 낮은 신호 대 잡음 비율 조건에서도 뛰어난 노이즈 저항성을 보여줍니다.



### GiVE: Guiding Visual Encoder to Perceive Overlooked Information (https://arxiv.org/abs/2410.20109)
- **What's New**: 이 연구에서는 Guiding Visual Encoder to Perceive Overlooked Information (GiVE) 접근 방식을 제안하여 시각 데이터를 텍스트와 효과적으로 결합하고 overlooked information (간과된 정보)를 반영할 수 있는 능력을 향상시킵니다. GiVE는 Attention-Guided Adapter (AG-Adapter) 모듈과 Object-focused Visual Semantic Learning 모듈을 통해 시각적 표현을 개선합니다.

- **Technical Details**: GiVE는 세 가지 새로운 손실 함수인 Object-focused Image-Text Contrast (OITC) 손실, Object-focused Image-Image Contrast (OIIC) 손실, Object-focused Image Discrimination (OID) 손실을 통합하여 모델이 salient (두드러진) 및 non-salient (비두드러진) 객체를 모두 고려하도록 합니다. 이 접근 방식은 dynamic visual focus (동적 시각적 초점) 조정을 통해 시각 데이터의 표현 능력을 높입니다.

- **Performance Highlights**: 실험 결과, GiVE 접근 방식은 기존의 시각 인코더에 비해 객체 고려 및 검색 정확도 향상에서 현저한 성과를 보여주며, state-of-the-art (최첨단) 성능을 달성하였습니다. 또한 Multi-Object Instruction (MOInst) 데이터셋을 새롭게 소개하여 다양한 객체에 대한 의미적 지침을 제공합니다.



### Anatomical 3D Style Transfer Enabling Efficient Federated Learning with Extremely Low Communication Costs (https://arxiv.org/abs/2410.20102)
Comments:
          Accepted by AIM-FM Workshop at NeurIPS 2024

- **What's New**: 이 연구에서는 3D 스타일 전이(3D style transfer)를 활용한 새로운 연합 학습(federated learning, FL) 접근 방식을 제안합니다. 이 방법은 다중 기관 분할(multi-organ segmentation) 작업을 위한 것으로, 통합된 다중 데이터셋에서 얻은 고스도 확장성이 뛰어나며 데이터 양이 증가할수록 일반화 성능을 향상시킬 수 있습니다.

- **Technical Details**: 제안된 방법인 A3DFDG(Anatomical 3D Frequency Domain Generalization)는 인체 기관의 구조 정보를 활용하고 기관의 위치를 기준으로 3D 스타일을 클러스터링합니다. 이렇게 혼합된 스타일은 해부학적 정보를 보존하면서 로컬 모델의 최적화를 정렬하도록 안내합니다. 이러한 접근법은 데이터 전송을 최소화하여 통신 비용을 크게 줄이며, 이러한 특성 덕분에 높은 정확도를 유지할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 통신 비용을 1.25%로 줄였을 때도 높은 정확성을 유지하며, 글로벌 다이스 유사성 계수(global dice similarity coefficient)가 4.3%의 차이를 보였습니다. 이러한 연구는 통신 비용이 낮고 간단한 파이프라인이 필요한 실제 시나리오에서 높은 실용성을 입증합니다.



### Generative Adversarial Patches for Physical Attacks on Cross-Modal Pedestrian Re-Identification (https://arxiv.org/abs/2410.20097)
- **What's New**: 이 논문은 Visible-infrared pedestrian Re-identification (VI-ReID) 모델에 대해 최초의 물리적 적대적 공격을 제안하였습니다. 이 방법은 Edge-Attack으로 명명되며, 모델이 심층적인 암시적 기능을 활용할 수 있는 능력을 테스트하는 데 중점을 둡니다.

- **Technical Details**: Edge-Attack은 두 단계로 구성된 새로운 접근 방식을 사용합니다. 첫 번째 단계에서는 다중 수준의 엣지(feature) 추출기를 자기 감독(self-supervised) 방식으로 훈련하여 각 개인의 차별적인 엣지 표현을 캡처합니다. 두 번째 단계에서는 Vision Transformer Generative Adversarial Networks (ViTGAN)을 이용하여 추출된 엣지 기능을 기반으로 적대적 패치를 생성합니다. 이 패치를 보행자 복장에 적용하여 현실적인 물리적으로 구현 가능한 적대적 샘플을 생성합니다.

- **Performance Highlights**: SYSU-MM01 및 RegDB 데이터셋에서의 광범위한 실험 결과, Edge-Attack은 최신 VI-ReID 방법의 성능을 상당히 저하시키는 데 효과적임을 보여주었습니다. 이를 통해 기존 VI-ReID 모델의 기능 추출 능력을 탐색할 수 있는 기회를 제공합니다.



### UniVST: A Unified Framework for Training-free Localized Video Style Transfer (https://arxiv.org/abs/2410.20084)
Comments:
          10 pages not including reference

- **What's New**: 본 논문은 UniVST라는 훈련이 필요 없는 통합 비디오 스타일 전송 프레임워크를 제안합니다. 이는 기존의 전방위 비디오 스타일 전송 방법과 차별화되는 점이 있습니다.

- **Technical Details**: UniVST는 DDIM 역전 과정에서 피처 맵을 활용하는 포인트 매칭 마스크 전파 전략을 채택하여 모델 아키텍처를 간소화합니다. AdaIN 기반의 스타일 전송 메커니즘을 통해 콘텐츠 충실도(content fidelity)와 스타일 풍부함(style richness) 간의 균형을 유지하며, 슬라이딩 윈도우 스무딩 전략을 통해 픽셀 표현 내 광학 흐름(optical flow)을 활용하여 시간 일관성(temporal consistency)을 향상시킵니다.

- **Performance Highlights**: UniVST는 질적 및 양적 메트릭에서 기존 방법들을 초월함을 입증하였으며, 주 객체의 스타일을 유지하면서도 시간 일관성과 디테일 보존 문제를 효과적으로 해결합니다.



### SFTrack: A Robust Scale and Motion Adaptive Algorithm for Tracking Small and Fast Moving Objects (https://arxiv.org/abs/2410.20079)
Comments:
          IROS 2024 selected Oral

- **What's New**: 이 논문은 무인항공기(UAV) 영상에서 다중 객체 추적(MOT) 문제를 다루며, 기존 연구를 보완하는 간단하면서도 효과적인 방법을 제시합니다. 새로운 접근법은 저신뢰 감지(low-confidence detections)에서 시작하는 추적 전략을 도입하여, UAV 응용 시나리오에서 자주 발생하는 문제를 해결하려고 합니다.

- **Technical Details**: 본 연구에서는 UAV의 빠른 움직임과 고도 및 넓은 시야로 인해 발생하는 작은 크기의 객체들로 인해 다소 저신뢰한 검출이 발생하는 문제를 해결하기 위한 두 가지 주요 목표를 설정했습니다. 첫째, UAV의 불규칙한 움직임을 보완하는 기술을 개발하고, 둘째, 저신뢰 검출의 빈번한 발생을 효과적으로 관리하는 방법을 제시합니다. 우리는 객체의 경계 상자를 조정하여 UAV의 움직임을 완화하는 'UAV Motion Compensation' 기법을 소개합니다.

- **Performance Highlights**: 우리의 방법은 두 개의 UAV 전용 데이터셋(VisDrone2019, UAVDT)과 일반 객체 추적 데이터셋(MOT17)에서 벤치마크 평가를 수행하였습니다. 그 결과, 우리의 접근법이 현존하는 최첨단 방법론을 초월하며, 다양한 추적 환경에서의 견고성과 적응성을 입증하였습니다. 또한 UAVDT 데이터셋의 주석 오류를 수정하고, 누락된 데이터를 보완하여 평가의 정확성을 높였습니다.



### 3D Distance-color-coded Assessment of PCI Stent Apposition via Deep-learning-based Three-dimensional Multi-object Segmentation (https://arxiv.org/abs/2410.20055)
- **What's New**: 본 논문에서는 경피관상동맥 중재술(PCI)에서 스텐트 부착 상태를 평가하기 위한 새로운 3D 거리-색상 코드 평가(DccA) 방법을 제안합니다.

- **Technical Details**: DccA는 심혈관 초음파 영상(IV-OCT)에서의 3D 다물체 분할을 기반으로 하여 스텐트와 혈관의 내부를 정확하게 분리합니다. 이 기법은 공간 매칭 네트워크(spatial matching network)와 스타일 전이(style transfer)를 이용한 이중 학습을 통해 구현됩니다.

- **Performance Highlights**: 3D DccA는 95% 이상의 분할 정밀도를 달성하여 PCI 스텐트 배치의 임상 평가를 개선하고, 개인 맞춤형 치료 계획을 지원합니다.



### ResAD: A Simple Framework for Class Generalizable Anomaly Detection (https://arxiv.org/abs/2410.20047)
Comments:
          This paper was accepted as a spotlight papaer by NeurIPS 2024

- **What's New**: 이 논문은 클래스 일반화 가능한 이상 탐지(class-generalizable anomaly detection)의 문제를 다루고 있습니다. 다양한 도메인의 새로운 클래스에서 이상을 탐지할 수 있는 통합 AD 모델을 학습하도록 설계된 간단하지만 효과적인 프레임워크 ResAD를 제안합니다.

- **Technical Details**: ResAD는 세 가지 주요 요소로 구성됩니다: 1) Feature Converter는 초기 피처를 잔여 피처로 변환합니다; 2) Feature Constraintor는 정상 잔여 피처를 공간 하이퍼스피어로 제약하여 피처 변화를 줄이고 클래스 간 피처 스케일을 유지합니다; 3) Feature Distribution Estimator는 정상 잔여 피처 분포를 추정하여 이상을 분포 외(out-of-distribution)로 인식합니다.

- **Performance Highlights**: ResAD는 새로운 클래스에서 직접 사용되었을 때 뛰어난 이상 탐지 성능을 발휘하며, 4-shot 정상 샘플만으로도 최첨단 경쟁 방법을 크게 능가하는 결과를 보여줍니다.



### SCube: Instant Large-Scale Scene Reconstruction using VoxSplats (https://arxiv.org/abs/2410.20030)
Comments:
          NeurIPS 2024. Project page: this https URL

- **What's New**: SCube는 일부 제공된 이미지로부터 대규모 3D 장면(geometry, appearance, and semantics)을 재구성하기 위한 혁신적인 방법으로, VoxSplat이라는 새로운 표현 방식을 사용합니다.

- **Technical Details**: SCube는 고해상도 희소-복셀(sparse-voxel) 스캐폴드를 기반으로 하는 3D Gaussian의 집합인 VoxSplat을 이용해 장면을 인코딩합니다. 입출력 이미지에 조건화된 계층 구조의 복셀(latent diffusion model)을 활용해 재구성을 수행하며, 그 후에는 피드포워드(Feedforward) 방식의 appearance prediction 모델을 통해 Gaussian 세트를 예측합니다.

- **Performance Highlights**: SCube는 비겹치는 3장의 이미지로도 1024^3 복셀(grid)에서 수백 미터를 포괄하는 백만 개의 Gaussian을 20초 만에 생성할 수 있습니다. Waymo 자율주행 데이터셋을 통해 기존 방법들에 비해 우수한 3D 재구성 성능을 보여주며, LiDAR 시뮬레이션 및 텍스트-투-장면 생성과 같은 다양한 응용 프로그램에 적용할 수 있습니다.



### Towards Robust Algorithms for Surgical Phase Recognition via Digital Twin-based Scene Representation (https://arxiv.org/abs/2410.20026)
- **What's New**: 이번 연구는 외과 수술 단계 인식을 위한 강력한 프레임워크를 제안하며, 디지털 트윈(Digital Twin, DT) 기반 장면 표현을 활용하여 모델의 강건성을 향상시킵니다. 이 접근법은 수술 비디오에서 비선형 관계가 모델의 성능에 미치는 부정적인 영향을 줄입니다.

- **Technical Details**: 제안된 프레임워크는 세 가지 주요 모듈로 구성되어 있습니다: 1) 표현 추출, 2) DT 기반 패치 임베딩, 3) DT 기반 SPR. 프레임워크는 SAM2 및 DepthAnything을 활용하여 수술 비디오에서 심층적인 장면 표현을 추출한 후, 이를 Surgformer 모델에 통합하여 강화된 성능을 보여줍니다.

- **Performance Highlights**: 프레임워크는 CRCD 데이터셋에서 51.1의 비디오 수준 정확도를 기록하며, 내부 로봇 훈련 데이터셋에서 96.0, 강한 손상에 있는 Cholec80 테스트 세트에서 64.4의 성과를 달성했습니다. 이는 기존 모델에 비해 우수한 강건성을 보여줍니다.



### Unsupervised Machine Learning for Detecting and Locating Human-Made Objects in 3D Point Cloud (https://arxiv.org/abs/2410.20006)
- **What's New**: 이 연구는 자연의 나무 구조 내에서 인간이 만든 객체를 탐지하고 식별하는 새로운 작업을 소개합니다. 이 작업은 지면 필터링 단계에서 유도된 비지면 포인트의 하위 집합에서 수행됩니다.

- **Technical Details**: 제안된 방법론은 세 가지 단계로 구성됩니다: (1) Ground Filtering: One-Sided Regression (OSR)을 사용하는 통계적 방법이 도입되어 불균형 지형에서의 한계를 해결합니다. (2) Local Information Extraction (LIE): MPF의 Hessian 행렬을 기반으로 한 커널 기반 방법이 개발되었습니다. (3) Clustering: 결과를 Gaussian Mixture Model (GMM)에 적용하여 비지면 포인트를 나무와 인간이 만든 객체로 분할합니다.

- **Performance Highlights**: 실험 결과, 제안된 지면 필터링 방법이 이전 기술보다 우수한 성능을 나타냈습니다. LIE 방법은 나무와 인간이 만든 객체를 성공적으로 구별할 수 있음을 보여주었습니다.



### A-MFST: Adaptive Multi-Flow Sparse Tracker for Real-Time Tissue Tracking Under Occlusion (https://arxiv.org/abs/2410.19996)
Comments:
          12 pages, 6 figures. Submitted to IPCAI2025

- **What's New**: 본 연구는 Sparse Efficient Neural Depth and Deformation (SENDD) 모델의 효율성을 강화하고, 수술 도구에 의한 차폐(occlusion) 탐지 기능과 추적 일관성을 향상시킵니다. 새로운 Adaptive Multi-Flow Sparse Tracker (A-MFST)와 Segment Anything Model2 (SAM2)를 통해 실시간의 고효율 추적이 가능합니다.

- **Technical Details**: 본 논문은 SENDD 모델을 기반으로 수술 도구의 차폐를 잘 처리하는 방법론을 제안합니다. SAM2를 사용하여 차폐를 탐지하고 mask 처리하며, A-MFST는 무감독 방식으로 여러 프레임의 정보를 종합하여 일관성 있는 추적 경로를 생성합니다. 이러한 방법은 정확한 Mean Endpoint Error (MEE) 향상을 목표로 합니다.

- **Performance Highlights**: STIR 데이터셋에서 평가된 결과, 차폐 상황에서도 평균 12%의 MEE 감소와 4, 8, 16, 32, 64 픽셀의 경계값에서 평균 정확도가 6% 향상되었습니다. 드리프트를 줄이면서 안정성과 정확성을 동시에 확보하여 실시간 성능을 유지하는 데 성공했습니다.



### OReole-FM: successes and challenges toward billion-parameter foundation models for high-resolution satellite imagery (https://arxiv.org/abs/2410.19965)
- **What's New**: 본 논문은 세계 최초의 엑사스케일 슈퍼컴퓨터인 Frontier를 활용하여 원격 감지(상관 이미지) 분야의 억 단위 매개변수(억 단위 모델)를 사전 훈련하고 평가한 연구 결과를 제시하고 있습니다. 이 연구는 FMs(Foundation Models)의 훈련 효율성을 높이고, 데이터 세트 구성 및 벤치마킹을 위한 최고 관행을 제공합니다.

- **Technical Details**: 본 연구에서는 고해상도(Swin-H) 및 중해상도(ViT-L) 광학 이미지와 SAR(ViT-L) 데이터를 포함한 여러 사전 훈련 변형의 성능을 평가했습니다. 또한 TIU라는 새로운 고해상도 원격 감지 사전 훈련 데이터 세트를 구성하였고, 4-band 모델의 초기화를 위한 전이 학습 기법을 사용하여 사전 훈련 속도와 성능을 향상시켰습니다.

- **Performance Highlights**: 논문은 모델 초기화, 데이터 및 사전 훈련된 모델을 공공에 공개할 계획이라는 점을 강조합니다. 또한 원격 감지 벤치마크에서의 이미지 분류, 의미 분할 및 객체 탐지 성능을 평가한 결과, 효과적인 모델 스케일을 위한 데이터 스케일링의 중요성을 강조합니다.



### Turn-by-Turn Indoor Navigation for the Visually Impaired (https://arxiv.org/abs/2410.19954)
- **What's New**: 본 논문은 시각 장애인을 위한 실내 내비게이션 시스템을 소개합니다. 스마트폰의 카메라만을 사용하여 실시간 이미지 인식, LLM(large language models), 심층 학습 알고리즘, 다중 모달 모델을 활용하여 건물 내에서의 턴 바이 턴(turn-by-turn) 안내를 제공합니다.

- **Technical Details**: 시스템은 Raspberry Pi에서 LLM과 심층 학습 알고리즘을 실행하여 주변 환경의 시각적 데이터를 실시간으로 처리합니다. 사용자의 스마트폰 카메라는 주변 이미지를 캡처하고, 이 정보는 Raspberry Pi로 전송되어 건축적 특징이나 장애물 등을 인식합니다. 이후 해석된 데이터는 자연어 지시사항으로 변환되며, 오디오 프롬프트를 통해 사용자에게 전달됩니다. 이 접근 방식은 사용자의 장치에 최소한의 작업 부담을 요구하며, 모든 종류의 디바이스와 호환 가능합니다.

- **Performance Highlights**: 초기 평가 결과, 이 시스템은 복잡한 실내 공간에서 사용자에게 정확하게 안내하는 성능을 보여주었습니다. 실제 환경에서의 적용 가능성이 높으며, 사용자에게 독립성을 제공하는 데 기여할 수 있습니다.



### A Multimodal Approach For Endoscopic VCE Image Classification Using BiomedCLIP-PubMedBER (https://arxiv.org/abs/2410.19944)
Comments:
          11 Pages, 2 Figures, Capsule Vision 2024 Challenge

- **What's New**: 이번 논문에서는 BiomedCLIP PubMedBERT 모델을 고도화하여 Video Capsule Endoscopy (VCE) 프레임 내의 이상을 분류하는 새로운 접근 방식을 제안합니다. 이는 위장 건강 관리의 진단 효율성을 향상시키기 위한 연구입니다.

- **Technical Details**: 이 방법은 PubMedBERT 언어 모델과 Vision Transformer (ViT)를 통합하여 내시경 이미지를 처리합니다. 우리는 이미지 전처리 및 BiomedCLIP 모델의 파인튜닝(fine-tuning)을 통해 시각적 및 텍스트 입력에 대한 고품질 임베딩(embedding)을 생성합니다. 이후 유사도 점수(similarity scoring)를 통해 이미지를 분류합니다. 분류는 총 10개의 특정 클래스인 혈관 확장증(angioectasia), 출혈(bleeding), 침식(erosion), 홍반(erythema), 이물질(foreign body), 림프관 확장증(lymphangiectasia), 폴립(polyp), 궤양(ulcer), 벌레(worms), 정상(normal)으로 나누어집니다.

- **Performance Highlights**: 성능 지표로는 분류(classification), 정확도(accuracy), 재현율(recall), F1 점수(F1 score)가 있으며, 모델이 내시경 프레임 내의 이상을 정확히 식별할 수 있는 강력한 능력을 보여줍니다. 이는 실제 임상 진단에 활용될 가능성이 높습니다.



### Tracking and triangulating firefly flashes in field recordings (https://arxiv.org/abs/2410.19932)
- **What's New**: 이 논문에서는 자연 이미지에서 반딧불 플래시를 식별하는 문제를 해결하기 위한 딥러닝 접근법을 제안합니다. 특히, 전통적인 방법보다 훨씬 더 높은 정확도로 플래시를 구별할 수 있는 Convolutional Neural Network (CNN)을 사용하여, 이전의 시간 및 공간 캘리브레이션 없이도 3D 재구성을 가능하게 했습니다.

- **Technical Details**: 반딧불 플래시 식별을 위해 수동으로 레이블링된 수천 개의 이미지 패치로 구성된 트레이닝 세트를 제공하고, CNN을 훈련시켰습니다. 이 모델은 기존의 intensity thresholding 방법보다 더 정밀하며, 65×65 픽셀 크기의 RGB 패치에 대해 교육되었습니다. 이를 통해 불필요한 아티팩트를 효과적으로 제거하고 진짜 플래시만을 남기도록 했습니다.

- **Performance Highlights**: 새로운 firefl-eye-net은 다양한 배경 조명 및 간섭이 있는 자연 이미지에서도 높은 신뢰도로 반딧불 플래시를 인식할 수 있었습니다. calibration-free 방식은 다양한 환경 조건에서도 적용 가능하여, 플래시 추적의 효율성과 정확성을 크게 개선했습니다. 또한, 이렇게 얻은 3D 정보는 반딧불 모니터링 및 보존 노력에 중요한 기여를 할 것으로 기대됩니다.



### Exploring Self-Supervised Learning with U-Net Masked Autoencoders and EfficientNet B7 for Improved Classification (https://arxiv.org/abs/2410.19899)
Comments:
          Capsule Vision 2024 Challenge

- **What's New**: 본 연구에서는 자기 감독(self-supervised) 기반의 U-Net 마스킹 자동 인코더(masked autoencoder)와 노이즈 제거 모델을 제안합니다. 이 모델은 원본 이미지를 재구성하는 데 중점을 두고 있으며, EfficientNet B7 모델과의 특성 결합을 통해 분류 작업을 수행합니다.

- **Technical Details**: 제안된 방법론은 트랜스포머 기반의 U-Net 아키텍처를 활용한 자가 학습 모델입니다. 무작위로 마스킹된 입력 이미지를 사용하여 U-Net 모델을 훈련시키고, 가우시안 노이즈를 추가하면서 원본 이미지를 재구성하는 방식으로 작동합니다. 최종적으로 EfficientNet과 U-Net 인코더에서 추출한 특성을 결합하여 밀집 레이어에 입력합니다.

- **Performance Highlights**: 모델 평가 결과, Efficient Fusion U-Net with Attention이 0.929의 균형 정확도를 기록하며 최상의 성능을 보였습니다. EfficientNet B7과 U-Net 인코더의 특성 통합을 통해 분류 정확도가 크게 향상되었으며, 최종적인 최고 정확도는 0.94에 달했습니다.



### FLAASH: Flow-Attention Adaptive Semantic Hierarchical Fusion for Multi-Modal Tobacco Content Analysis (https://arxiv.org/abs/2410.19896)
Comments:
          Under review at International Journal of Computer Vision; 20 pages, 4 figures, 5 tables

- **What's New**: 이번 연구에서는 Flow-Attention Adaptive Semantic Hierarchical Fusion (FLAASH)라는 새로운 다중 모달 심층 학습 프레임워크를 소개합니다. 이 프레임워크는 소셜 미디어 플랫폼에서의 담배 관련 비디오 콘텐츠를 종합적으로 분석하기 위해 설계되었습니다.

- **Technical Details**: FLAASH는 비주얼(visual)과 텍스트(text) 정보를 통합하는 복잡성을 해결하기 위해 계층적 융합 메커니즘(hierarchical fusion mechanism)을 활용합니다. 주요 혁신 요소로는 시각적 및 텍스트 모달리티 간의 미묘한 상호작용을 포착하는 플로우-어텐션 메커니즘(flow-attention mechanism), 다양한 계층 수준의 기여를 균형있게 조절하는 적응 가중치 스킴(adaptive weighting scheme), 관련 특성을 선택적으로 강조하는 게이팅 메커니즘(gating mechanism)이 포함됩니다.

- **Performance Highlights**: MTCAD(Multimodal Tobacco Content Analysis Dataset)에서의 실험 결과, FLAASH는 분류 정확도(classification accuracy), F1 점수 및 시간적 일관성(temporal consistency)에서 기존 방법보다 크게 향상된 성능을 보였습니다. 또한 전통적인 비디오 질문-답변 데이터셋에서도 강력한 일반화 능력을 입증했습니다.



### Topology-aware Mamba for Crack Segmentation in Structures (https://arxiv.org/abs/2410.19894)
Comments:
          Published at Journal of Automation in Construction

- **What's New**: CrackMamba는 전통적인 CNN 모델의 한계를 극복하고 비전 변환기(Vision Transformer)보다 효율적으로 균열(segmentation) 분석을 수행하는 새로운 프레임워크입니다.

- **Technical Details**: CrackMamba는 VMambaV2를 인코더로 사용하며, ImageNet-1k의 사전 훈련(weights)된 가중치를 적용합니다. 또한 새롭게 설계된 디코더와 함께 랜덤하고 복잡한 균열 개발을 처리하기 위해 Snake Scan 모듈을 도입하여 균열 피처 시퀀스를 재구성하고, 세 가지 가지의 Snake Conv VSS(SCVSS) 블록을 채택하여 균열을 더욱 효과적으로 타겟합니다.

- **Performance Highlights**: CrackMamba는 CrackSeg9k 및 SewerCrack 데이터셋에서 최신 성능(SOTA)을 달성했으며, 망막 혈관 분할 데이터셋 CHASE_DB1에서도 경쟁력 있는 성능을 보여줍니다. 이는 모델의 일반화(generalization) 능력을 강조합니다.



### A Survey of AI-Generated Video Evaluation (https://arxiv.org/abs/2410.19884)
- **What's New**: AI 생성 비디오 평가(AIGVE)라는 새로운 연구 분야의 필요성을 강조하며, 인공지능이 생성한 비디오의 평가 방법과 현존하는 방법론의 체계적인 검토를 제공합니다.

- **Technical Details**: 비디오 콘텐츠의 평가에는 복잡한 공간적 및 시간적 동역학이 포함되며, 기존의 VQA(Video Quality Assessment) 지표에 국한되지 않고, 인간의 의도를 어떻게 수용하는지를 평가해야 합니다.

- **Performance Highlights**: 이 논문은 AI 생성 비디오의 평가 및 관련된 다양한 연구 분야를 포괄하는 체계적인 리뷰를 제공하며, 향후 연구 방향으로는 비전 언어 모델과의 통합, 평가 점수의 해석 가능성 향상, 윤리적 고려사항에 대한 연구 등을 제안합니다.



### Paved or unpaved? A Deep Learning derived Road Surface Global Dataset from Mapillary Street-View Imagery (https://arxiv.org/abs/2410.19874)
- **What's New**: 새로운 공개 데이터셋은 1억 5천만 개의 이미지를 활용하여 도로 표면 특성에 대한 정보를 제공합니다. 이 데이터셋은 세계 최대의 크라우드소싱 기반 거리 뷰 플랫폼인 Mapillary에서 수집되었으며, 최첨단 지리공간 AI 방법을 활용했습니다.

- **Technical Details**: 하이브리드 딥러닝 접근법을 제안하며, SWIN-Transformer 기반 도로 표면 예측과 CLIP-and-DL 세분화 기반의 품질 나쁜 이미지 필터링 방법을 결합했습니다. 도로 표면 예측 결과는 OpenStreetMap (OSM) 도로 기하학과 통합되었습니다.

- **Performance Highlights**: 모델 검증 결과 OSM 표면 데이터에 대한 성능이 우수하여 포장 도로에 대한 F1 점수는 91-97%를 기록했습니다. 글로벌 도로 표면 정보의 가용성을 3백만 킬로미터 이상 확장하며, 주요 응용 분야로는 도시 계획, 재난 라우팅, 물류 최적화 등이 있습니다.



### Radar and Camera Fusion for Object Detection and Tracking: A Comprehensive Survey (https://arxiv.org/abs/2410.19872)
- **What's New**: 이 논문은 레이더-카메라 퓨전(Radar-Camera Fusion)에 대한 체계적인 리뷰를 제공하며, 현재의 연구 공백을 메우기 위해 객체 탐지(Object Detection) 및 추적(Tracking)에 초점을 맞춥니다. 이를 통해 다양한 응용 분야에서 레이더와 카메라 데이터의 보완성을 활용하려는 연구 방향을 제시합니다.

- **Technical Details**: 레이터-카메라 퓨전 기술은 여러 센서의 데이터를 통합하여 환경 인식의 정확성과 신뢰성을 높이는 데 사용됩니다. 이 논문에서는 센서 보정(Sensor Calibration), 모달 표현(Modal Representation), 데이터 정렬(Data Alignment), 그리고 퓨전 연산(Fusion Operation) 등의 핵심 기술에 대해 설명합니다. 이러한 기술들은 서로 다른 데이터 원천의 불일치성을 극복하는 데 필수적입니다.

- **Performance Highlights**: 이 논물에서는 2019년부터 2024년까지 라이다(LiDAR), 카메라 및 레이더를 포함한 최신 데이터셋과 알고리즘을 요약합니다. 또한, 레이더-카메라 퓨전의 미래 연구 방향과 발전 트렌드를 강조합니다. 이러한 통합 방식은 다양한 환경 조건에서의 객체 탐지 및 추적의 신뢰성을 향상시키는 데 매우 효과적입니다.



### Comparing YOLO11 and YOLOv8 for instance segmentation of occluded and non-occluded immature green fruits in complex orchard environmen (https://arxiv.org/abs/2410.19869)
Comments:
          16 Pages, 10 Figures, 3 Tables

- **What's New**: 이번 연구에서는 'You Only Look Once' (YOLO) 시리즈의 최신 모델인 YOLO11과 YOLOv8의 원형 분할 (instance segmentation) 능력을 평가하였습니다. 특히, 미숙한 녹색 사과의 감지 능력에 초점을 맞추어 YOLO11n-seg가 모든 카테고리에서 0.831이라는 높은 정확도를 기록하며 효과성을 입증하였습니다.

- **Technical Details**: YOLO11과 YOLOv8은 최신 실시간 물체 탐지 기술의 선두주자로, 특수한 아키텍처와 훈련 방법론을 통해 성능을 극대화하였습니다. YOLOv8은 향상된 backbone과 neck 아키텍처를 도입하여 특징 추출 능력을 개선하였으며, YOLO11은 YOLOv8의 기반 위에 추가적인 최적화를 적용하여 처리 속도를 높였습니다. 두 모델 모두 실시간 애플리케이션에 적합한 다양한 운영 모드를 지원합니다.

- **Performance Highlights**: YOLO11m-seg는 0.876의 mAP@50을 기록하며 원형 분할 성능에서 일관되게 우수한 결과를 보여주었고, YOLOv8n은 3.3ms의 빠른 추론 속도로 이미지 처리에서 입증된 성능을 자랑합니다. 이는 복잡한 농업 환경에서 실시간 응용에 특히 적합합니다.



### Breaking the Illusion: Real-world Challenges for Adversarial Patches in Object Detection (https://arxiv.org/abs/2410.19863)
Comments:
          - 21 pages, 17 figures, 7 tables - accepted in 1st Workshop on Enabling Machine Learning Operations for next-Gen Embedded Wireless Networked Devices (EMERGE), 2024

- **What's New**: 이번 연구는 실제 환경에서의 YOLO 객체 탐지 네트워크에 대한 적대적 패치(adversarial patch)의 성능을 분석하고, 패치 크기, 위치, 회전, 밝기 및 색조와 같은 다양한 요소들이 패치의 효과성에 미치는 영향을 조사했습니다.

- **Technical Details**: 연구에서는 글로벌 패치(global patch) 및 로컬 패치(local patch)의 두 가지 공격 방식이 테스트되었습니다. 글로벌 패치는 다양한 환경 조건에서 올바른 탐지를 억제하기 위해 장면 내의 어느 곳에나 배치될 수 있도록 설계된 반면, 로컬 패치는 특정 객체와 부분적으로 겹치도록 설계되었습니다. 성능 평가는 전반적인 평균 정밀도(mean average precision, mAP) 및 탐지 신뢰도(detection confidence)를 기반으로 수행되었습니다.

- **Performance Highlights**: 연구 결과, 다양한 요인들 간의 상관관계가 드러났으며, 실제 환경에서의 공격 효능을 유지하는 데에 어려움이 있음을 강조했습니다. 디지털 환경과 실제 환경 간의 성능 차이는 최대 64%에 달하는 변동성이 있었습니다. 이는 실제 환경이 적대적 공격에 미치는 영향을 이해할 필요성과, 보다 강력한 방어 시스템 개발의 중요성을 강조합니다.



### Real-Time Weapon Detection Using YOLOv8 for Enhanced Safety (https://arxiv.org/abs/2410.19862)
Comments:
          21 pages, 5 figures

- **What's New**: 이번 연구는 YOLOv8을 활용한 실시간 무기 탐지 AI 모델의 개발을 제시하며, 학교, 공항 및 대중 교통 시스템과 같은 공공 장소의 안전성을 높이는 데 중점을 두고 있습니다.

- **Technical Details**: 본 모델은 다양한 유형의 총기 및 날카로운 무기를 포함한 수천 개의 이미지로 구성된 포괄적인 데이터셋을 통해 훈련되었습니다. 평가 과정에서는 precision, recall, F1-score 및 mean Average Precision (mAP)과 같은 주요 지표를 사용하였으며, 다양한 Intersection over Union (IoU) 임계값에서 성능을 측정하였습니다.

- **Performance Highlights**: 우리의 YOLOv8 기반 무기 탐지 모델은 정확하고 효율적인 실시간 비디오 스트림 내에서 무기를 탐지할 수 있는 능력을 보여주었습니다. 시스템의 운영 효율성은 고속으로 프레임을 처리할 수 있는 능력으로 나타났으며, 이는 공공 장소에서의 안전성을 강화하는 데 기여할 것입니다.



### Benchmarking Large Language Models for Image Classification of Marine Mammals (https://arxiv.org/abs/2410.19848)
Comments:
          ICKG 2024

- **What's New**: 이번 연구는 해양 포유류에 초점을 맞춘 새로운 벤치마크 데이터셋을 소개합니다. 데이터셋에는 65종의 해양 포유류에 대한 1,423개의 이미지가 포함되어 있으며, 생물학적 수준에 따라 분류된 레이블이 제공됩니다.

- **Technical Details**: 우리는 (1) 신경망이 제공하는 임베딩을 사용하는 기계 학습 (ML) 알고리즘, (2) 영향력 있는 사전 훈련된 신경망, (3) 제로샷 모델인 CLIP 및 LLM, (4) LLM 기반의 다중 에이전트 시스템 (MAS)을 평가했습니다. 이러한 접근 방식을 통해 해양 포유류의 분류 성능을 향상시킬 수 있음을 보여주었습니다.

- **Performance Highlights**: 전통적인 모델과 LLM의 성능을 비교한 결과, MAS는 다양한 분류 레벨에서 단일 LLM보다 더 높은 성능을 보였습니다. 이 새로운 데이터셋과 기술은 해양 포유류 연구와 보존 전략 개발에 기여할 것으로 기대됩니다.



### AEPL: Automated and Editable Prompt Learning for Brain Tumor Segmentation (https://arxiv.org/abs/2410.19847)
Comments:
          4 pages paper for ISBI2025

- **What's New**: 본 논문에서는 뇌 종양 분할을 위한 새로운 프레임워크인 Automated and Editable Prompt Learning (AEPL)을 제안합니다. AEPL은 다중 작업 학습(multi-task learning)과 프롬프트 학습(prompt learning)을 결합하여 종양의 등급을 분할 과정에 통합합니다. 이를 통해 수작업 프롬프트 입력 없이도 보다 정확한 분할을 가능하게 합니다.

- **Technical Details**: AEPL은 3D U-Net 구조를 기반으로 하며, U-Net 인코더, 종양 등급 분류기, 프롬프트 인코더 및 U-Net 디코더로 구성됩니다. AEPL은 다중 모달리티 MRI 입력을 처리하여 종양 등급 예측 및 분할 마스크 생성을 동시에 수행합니다. 예측된 종양 등급은 자동으로 생성된 프롬프트로 사용되어 분할 마스크를 생성하는데 도움을 줍니다.

- **Performance Highlights**: AEPL은 BraTS 2018 데이터세트를 사용하여 여러 최신 방법들보다 우수한 성능을 보여주었으며, 종양 등급 정보를 직접 통합하여 세밀한 분할 결과를 생성하였습니다. 비록 다수의 추가 실험이 필요하지만, AEPL은 뇌 종양 분할에 있어 높은 유용성과 임상적 가치를 갖고 있음을 입증하였습니다.



### YOLO11 and Vision Transformers based 3D Pose Estimation of Immature Green Fruits in Commercial Apple Orchards for Robotic Thinning (https://arxiv.org/abs/2410.19846)
Comments:
          24 Pages, 13 Figures, 1 Table

- **What's New**: 이번 연구에서는 상업농장에 있는 미성숙 녹색 사과(과일묘)의 3D 포즈 추정을 위한 강력한 방법을 개발하였습니다. YOLO11 객체 탐지 및 포즈 추정 알고리즘과 Vision Transformers (ViT)를 활용하여 깊이 추정에 있어 성능을 비교했습니다.

- **Technical Details**: YOLO11과 YOLOv8의 여러 구성요소를 동일한 하이퍼파라미터 설정 하에 비교하였으며, YOLO11n이 0.91의 박스 정밀도와 0.915의 포즈 정밀도를 기록하며 가장 뛰어난 성능을 보였습니다. 또한, Depth Anything V2가 Dense Prediction Transformer보다 3D 포즈의 길이 평가에서 더 우수한 성능을 나타냈습니다. 실험 결과, RMSE가 1.52, MAE가 1.28로 미성숙 과일 길이 추정에서 뛰어난 정밀도를 보여줍니다.

- **Performance Highlights**: YOLO11n은 2.7 ms의 추론 속도를 기록하며 모든 구성에서 가장 빠른 성능을 보였고, YOLOv8n은 각각 0.905와 0.925의 가장 높은 박스 및 포즈 리콜 점수를 기록하였습니다. mAP@50에서는 YOLO11s가 0.94로 가장 높은 점수를 얻었으며, YOLOv8n은 0.96의 포즈 mAP@50 점수를 달성하였습니다.



### GreenEye: Development of Real-Time Traffic Signal Recognition System for Visual Impairments (https://arxiv.org/abs/2410.19840)
Comments:
          Published in Korea Software Congress (2023)

- **What's New**: 이번 연구에서는 시각장애인의 교통 신호 인식을 위한 GreenEye 시스템을 개발하였습니다. 이 시스템은 교통 신호의 색상을 인식하고 보행자가 횡단보도를 건너기 위해 남은 시간을 실시간으로 알려줍니다.

- **Technical Details**: 기존 연구는 주로 두 가지 교통 신호, 즉 초록불(Green)과 빨간불(Red)만을 인식하는 데 중점을 두었으나, GreenEye 시스템은 14가지의 다양한 클래스(class)를 인식하도록 설계되었습니다. 초기 훈련에서는 정밀도(precision)가 74.6%로 나타났으며, 클래스 간의 데이터 불균형이 인식률 저하에 기여했습니다. 따라서 추가적인 레이블링(labeling) 및 데이터베이스(database) 형성을 통해 각 클래스의 이미지 수를 안정화했습니다.

- **Performance Highlights**: 데이터 안정화 작업 후, 14개의 모든 클래스에서 뛰어난 정밀도 99.5%를 기록하였습니다. 이는 GreenEye 시스템의 성능을 크게 향상시킨 결과입니다.



### Scene-Segmentation-Based Exposure Compensation for Tone Mapping of High Dynamic Range Scenes (https://arxiv.org/abs/2410.19839)
Comments:
          to be presented in APSIPA ASC 2024

- **What's New**: 이 논문에서는 멀티-노출 이미지 융합(MEF) 기반의 톤 매핑을 위한 새로운 장면 분할 기반의 노출 보정 방법을 제안합니다. 이 새로운 접근 방식은 기존 방법의 한계를 극복하여 시각적으로 더 매력적인 HDR 이미지 생성을 목표로 합니다.

- **Technical Details**: 제안된 방법은 입력된 HDR 이미지를 픽셀의 휘도 값에 따라 서브 리전으로 분할하고, 각 리전 간의 대비를 극대화하기 위한 노출 값을 결정합니다. 또, 최종 융합 이미지에서 노출 보정의 효과를 더 잘 반영하기 위한 융합 가중치 계산 방법을 개선하여 제시합니다.

- **Performance Highlights**: 시뮬레이션 실험 결과, 제안된 방법을 사용한 MEF 기반의 톤 매핑이 기존의 전통적인 MEF 기반 톤 매핑 방법을 포함한 세 가지 일반적인 톤 매핑 방법보다 톤 매핑한 이미지 품질 지수(TMQI)에서 뛰어난 성능을 보였습니다.



### Upsampling DINOv2 features for unsupervised vision tasks and weakly supervised materials segmentation (https://arxiv.org/abs/2410.19836)
- **What's New**: 이 논문은 자가 지도(supervised) 비전 변환기(self-supervised vision transformers, ViTs)의 특징을 활용하여 물체의 위치 표시 및 세분화(segmentation)에서 뛰어난 성능을 보여주는 새로운 방법론을 제안합니다. 특히 DINOv2와 같은 ViT 모델로부터 업샘플링된 특징을 클러스터링 기반 접근법과 결합하여 강력한 결과를 얻었습니다.

- **Technical Details**: 이 연구는 ViT 네트워크에서 생성된 고해상도 특징 맵을 클러스터링하는 비지도 세분화 워크플로우를 사용합니다. 입력 이미지를 여러 방향으로 이동시켜 특징을 계산하고, 원래 이미지 크기로 리사이징하여 평균화하는 새로운 단일 패스(feature upsampling) 방법을 도입하였습니다. 이 방법은 다른 변환(회전, 뒤집기 등)에 대해서도 호환됩니다.

- **Performance Highlights**: 이 방법은 약한 지도(weakly supervised) 세분화와 관련된 작업에서 특히 높은 성능을 보이며, 세포 핵(segmentation of cell nuclei) 및 배터리 양극, 합금, 산화층, 유기 결정과 같은 산업적으로 중요한 자료를 다루는 데 효과를 발휘했습니다.



### GL-NeRF: Gauss-Laguerre Quadrature Enables Training-Free NeRF Acceleration (https://arxiv.org/abs/2410.19831)
Comments:
          NeurIPS 2024. Project page: this https URL

- **What's New**: GL-NeRF는 Gauss-Laguerre quadrature를 이용하여 볼륨 렌더링 계산을 새롭게 제안한 접근 방식입니다. 이 방법은 렌더링에 필요한 MLP 호출 수를 크게 줄이고, 추가적인 데이터 구조나 신경망을 도입하지 않습니다.

- **Technical Details**: GL-NeRF는 볼륨 렌더링을 위한 적분 계산을 단순한 지수 가중 적분 형태로 변환할 수 있는 변수를 변경함으로써 개선합니다. 이를 통해 Gauss-Laguerre quadrature를 사용하여 기존 NeRF 모델에 쉽게 적용할 수 있습니다. 이 방법은 기존 NeRF 모델에서 추가 교육 없이 활용될 수 있습니다.

- **Performance Highlights**: GL-NeRF는 최소한의 성능 저하로 MLP 호출 수를 상당히 줄일 수 있으며, 볼륨 렌더링의 계산 비용과 메모리 사용량을 감소시킵니다.



### Automating Video Thumbnails Selection and Generation with Multimodal and Multistage Analysis (https://arxiv.org/abs/2410.19825)
Comments:
          150 pages, 60 figures

- **What's New**: 이 논문은 전통 방송 콘텐츠의 비디오 썸네일 선택을 자동화하는 혁신적인 접근 방식을 제시합니다. 기존의 방법과 달리, 다양한 요인들을 고려하여 좀 더 정교한 썸네일을 생성합니다.

- **Technical Details**: 제안된 방법론은 로고 배치 공간, 수직 종횡비의 통합, 얼굴 정체성과 감정 인식을 정확하게 고려하여, 다양한 기준에 추진됩니다. 이 다단계 파이프라인은 후보 프레임을 선택하거나 비디오 요소를 혼합하여 새로운 이미지를 생성할 수 있습니다. 주요 모듈에는 downsampling, redundancy reduction, automated cropping, face recognition등이 포함되며, 대형 언어 모델과 visual transformers를 사용하여 의미적 일관성을 제공합니다.

- **Performance Highlights**: 69개의 비디오에 대한 실험에서 제안된 썸네일 세트 중 53.6%는 전문 디자이너가 선택한 썸네일을 포함했으며, 82명의 참가자를 대상으로 한 설문 조사에서 45.77%가 제안된 방법을 선호했습니다. 전문 디자이너는 대안 방법에 비해 유효한 후보가 3.57배 증가했다고 보고했습니다. 이는 우리 방법이 높은 품질 기준을 유지하며 썸네일 생성을 가속화할 수 있음을 확인해줍니다.



### Flame quality monitoring of flare stack based on deep visual features (https://arxiv.org/abs/2410.19823)
Comments:
          7 pages, 9 figures, 2 tables, International Conference on Computer Information Science and Artificial Intelligence(accepted)

- **What's New**: 이 논문에서는 석유 화석 에너지 플랜트에서 연소 효율을 모니터링하기 위한 새로운 방법을 제안합니다. 전통적인 센서 모니터링 방식 대신 시각적 특징만을 이용하여 화염의 품질을 평가하는 접근법입니다.

- **Technical Details**: 화염과 연기 비율, 화염의 RGB 정보, 화염의 각도 등 다양한 시각적 요소를 분석하여 연소 상태를 모니터링합니다. 이미지 분할(image segmentation), 객체 탐지(target detection), 객체 추적(target tracking), 주성분 분석(principal component analysis), GPT-4와 같은 기술을 종합적으로 활용합니다.

- **Performance Highlights**: 실시간 모니터링이 가능하며 연소 효율이 낮을 경우 공기와 폐기물의 비율을 조정하는 등의 조치를 신속하게 취할 수 있습니다. 이 방법은 산업적으로도 활용될 수 있는 혁신적인 접근법으로, 새로운 환경 모니터링 기술의 가치가 높다고 평가됩니다.



### Explainable AI in Handwriting Detection for Dyslexia Using Transfer Learning (https://arxiv.org/abs/2410.19821)
Comments:
          4 pages, 4 figures, JAC-ECC Conference

- **What's New**: 해당 연구는 심리학적 장애인 난독증(Dyslexia)의 조기 발견을 위해 손글씨 분석을 통한 설명 가능한 AI(XAI) 프레임워크를 제안합니다. 이 방법은 트랜스포머 기반 모델과 전이 학습(transfer learning)을 활용하여 최첨단 기술을 초월한 성능을 자랑하며, 정확도 0.9958을 달성했습니다. 또한, Grad-CAM 시각화를 통해 모델 결정에 중요한 손글씨 특징들을 강조하여 해석 가능성을 보장합니다.

- **Technical Details**: 제안된 모델은 손글씨의 난독증 관련 특징을 식별하기 위해 전이 학습과 트랜스포머 기반 모델을 통합한 XAI 프레임워크로 구성됩니다. 이 방법은 다양한 언어 및 작성 시스템에 적응할 수 있도록 확장되며, 손글씨 분석을 통해 글로벌 수준에서 난독증 판별의 가능성을 보여줍니다. 모델은 MobileNet V3을 활용하며, CrossEntropyLoss 함수와 AdamW 최적화 기법을 사용하여 다중 클래스 문제를 해결합니다.

- **Performance Highlights**: 제안된 방법은 기존의 최첨단 기법에 비해 높은 분류 정확도를 달성하며,  성능 향상 뿐만 아니라 교육자, 임상 의사 및 학부모 사이에서의 신뢰를 구축하는 데 기여합니다. 이 연구는 난독증의 진단 철학과 정당화의 중요성을 강조하며, 개인화된 교육 전략의 개발을 지원하는 데 중요한 역할을 할 것으로 예상됩니다.



### DivShift: Exploring Domain-Specific Distribution Shift in Volunteer-Collected Biodiversity Datasets (https://arxiv.org/abs/2410.19816)
Comments:
          Accepted to NeurIPS 2024 Workshop on Tackling Climate Change with Machine Learning

- **What's New**: 본 연구에서는 기후 변화의 영향을 수량화하기 위해 북미 서부 해안의 800만 개 식물 이미지로 구성된 DivShift-NAWC 데이터셋을 새롭게 소개합니다. 이 데이터셋은 자원봉사자 수집 데이터의 지리적, 시간적, 관찰 품질, 사회경제적 편향이 심층 학습 모델 성능에 미치는 영향을 탐구하는 데 사용됩니다.

- **Technical Details**: DivShift 프레임워크를 통해 자원봉사자로 수집된 생물 다양성 데이터의 편향을 분포 이동(distribution shift)으로 설명하며, 이를 통해 각종 편향을 포함한 데이터셋을 훈련 세트와 테스트 세트로 나누고, Jensen-Shannon Distance (JSD)를 통해 모델 성능의 변화와 편향의 영향을 정량화합니다.

- **Performance Highlights**: 모델 성능을 네 가지 알려진 편향(spatial, temporal, observation quality, sociopolitical)에서 비교한 결과, 이러한 편향이 모델 성능을 실제로 혼동시킨다는 것을 관찰하였습니다. 이는 자원봉사자 수집 데이터의 효율적인 큐레이션이 기후 변화가 생물 다양성에 미치는 영향을 모니터링하기 위한 머신러닝 모델 훈련에 있어서 중요하다는 것을 시사합니다.



### Stochastic Flow Matching for Resolving Small-Scale Physics (https://arxiv.org/abs/2410.19814)
Comments:
          31 pages

- **What's New**: 본 논문에서는 Stochastic Flow Matching (SFM)이라는 새로운 방법론을 제안하여, 자연과학의 작은 규모의 물리적 세부 사항을 효과적으로 해상할 수 있습니다. 이는 기존 Conditional Diffusion 및 Flow 모델의 한계를 극복하기 위한 방법으로 개발되었습니다.

- **Technical Details**: SFM은 입력 데이터를 잠재적인 기본 분포(latent base distribution)로 인코딩하고, 그 다음 Flow Matching을 통해 작은 규모의 물리 현상을 생성하는 방식으로 작동합니다. 이 과정에서 인코더는 결정론적(deterministic) 요소를 캡처하고, 여기에 적응형 노이즈 스케일링 메커니즘을 적용하여 불확실성을 주입합니다. 이를 통해 불균형 문제와 데이터 제한 문제를 효과적으로 해결하고자 합니다.

- **Performance Highlights**: 실험 결과, 제안한 SFM 방법은 CWA 날씨 데이터셋과 특정 PDE 기반 Kolmogorov 데이터셋에서 기존 방법보다 현저히 우수한 성능을 보였습니다. 특히, 입력과 출력 데이터의 불일치 문제가 심화될수록, SFM의 성능은 더욱 두드러지며, 데이터가 한정된 상황에서도 과적합(overfitting)을 피하는 데 효과적임을 입증하였습니다.



### LocateBench: Evaluating the Locating Ability of Vision Language Models (https://arxiv.org/abs/2410.19808)
Comments:
          We release the dataset at this https URL

- **What's New**: LocateBench라는 새로운 벤치마크가 제안되어, 자연어 지시에 따라 이미지 내 객체를 탐지하는 능력을 평가하는 데 중점을 두고 있습니다. 이 데이터셋은 높은 품질과 전문가 주석을 특징으로 하며, 자주 사용되는 이전의 데이터셋과 차별화됩니다.

- **Technical Details**: LocateBench는 이미지, 설명 및 네 개의 후보 바운딩 박스를 포함한 다중 선택 질문 데이터셋으로 구성됩니다. 객체의 바운딩 박스는 COCO 데이터셋에서 특정 객체의 설명을 기반으로 필터링되어 신뢰성을 높인 데이터가 포함되어 있습니다. 이 데이터셋은 RefCOCO 시리즈 데이터셋에 기반하여 구축되었습니다.

- **Performance Highlights**: GPT-4o가 LocateBench에서 전반적으로 가장 높은 성능을 보였지만, 여전히 사람의 정확도(95%)에 비해 10% 이상 부족한 결과를 보였습니다. Gemini-1.5-pro는 특정 프롬프트 방법에 가장 민감하였으며, 그 외 Claude-3 Opus와 Llava-1.6 모델은 두 작업 모두에서 가장 낮은 성능을 기록했습니다.



### Comparing Surface Landmine Object Detection Models on a New Drone Flyby Datas (https://arxiv.org/abs/2410.19807)
Comments:
          9 pages, 22 figures, 7 tables

- **What's New**: 이번 연구는 드론 영상을 이용한 지뢰 탐지에 대한 철저한 데이터셋을 생성하고, 최신 딥러닝 기반 객체 탐지 알고리즘을 활용하여 지뢰 탐지 성능을 비교한 것이 특징입니다.

- **Technical Details**: 연구에서는 POM-2 및 POM-3 러시아 표면 지뢰의 드론 이미지로 구성된 커스텀 데이터셋을 만들었습니다. YOLOF, DETR, Sparse-RCNN, VFNet 등의 4가지 컴퓨터 비전 기반 모델을 훈련, 테스트하여 비교하였습니다.

- **Performance Highlights**: YOLOF 모델은 mAP(Mean Average Precision) 스코어 0.89로 다른 모델을 초월하는 성능을 보여주었으며, DETR, VFNET, Sparse-RCNN의 mAP 스코어는 약 0.82로 나타났습니다. YOLOF는 또한 Nvidia V100 컴퓨팅 클러스터에서 56분의 짧은 훈련 시간으로 훈련되었습니다.



### Stable Diffusion with Continuous-time Neural Network (https://arxiv.org/abs/2410.19798)
- **What's New**: 본 논문에서는 이미지 생성에 있어 Cellular Neural Networks(세포 신경망)의 잠재력을 탐구하고 실증적으로 보여줌으로써, 그들의 성능이 기존의 이산 시간 모델보다 뛰어남을 입증하고자 합니다.

- **Technical Details**: Stable diffusion 모델은 Gaussian 노이즈로의 점진적 분해 과정을 통해 이미지를 생성합니다. 이러한 과정은 Markov Chain의 역을 학습하며, 이 논문에서는 Cellular Neural Networks가 지속적인 시간에서 수행되며 실제 확산 과정을 더 잘 근사화할 수 있음을 보여줍니다.

- **Performance Highlights**: Cellular Neural Networks는 MNIST 데이터셋을 기반으로 한 이미지 생성에서 더 높은 품질의 이미지를 생성하고, 훈련 시간 또한 더 짧은 성능을 보여주었습니다.



### Feature Clipping for Uncertainty Calibration (https://arxiv.org/abs/2410.19796)
- **What's New**: 이 논문은 Deep Neural Networks (DNNs)의 신뢰할 수 있는 model calibration (모델 칼리브레이션)을 위한 새로운 post-hoc calibration 방법인 Feature Clipping (FC)을 제안합니다. 이는 DNN이 예측에서 과신에 빠지는 문제를 해결합니다.

- **Technical Details**: Feature Clipping은 feature 값을 특정 threshold (임계값)로 잘라내어 calibration error (칼리브레이션 오류)가 큰 샘플의 entropy (엔트로피)를 증가시키고, 적은 칼리브레이션 오류를 가진 샘플의 정보를 유지하는 방법입니다. 이 방법은 CIFAR-10, CIFAR-100, ImageNet과 같은 데이터셋에서 여러 모델에 걸쳐 실험하여 일관되게 모델의 칼리브레이션 성능을 향상시킵니다.

- **Performance Highlights**: FC는 기존의 post-hoc 및 train-time calibration 방법에 비해 일관된 성능 향상을 보여줍니다. FC는 calibration의 새로운 접근법을 제시하며, feature modification (특징 수정)에 기반한 최초의 칼리브레이션 방법으로, 여러 데이터셋과 모델에서 SOTA (state-of-the-art) 칼리브레이션 성능을 달성했습니다.



### DiffGAN: A Test Generation Approach for Differential Testing of Deep Neural Networks (https://arxiv.org/abs/2410.19794)
- **What's New**: 본 논문에서는 Deep Neural Networks (DNNs)의 차별적 테스트를 위한 새로운 기법인 DiffGAN을 제안합니다. DiffGAN은 Generative Adversarial Network (GAN)와 Non-dominated Sorting Genetic Algorithm II (NSGA-II)를 활용하여 다양한 트리거 입력(triggering inputs)을 생성하여 모델 간의 행동 차이를 나타냅니다.

- **Technical Details**: DiffGAN은 블랙 박스(black-box) 테스트 이미지 생성 접근법으로, DNN 모델 간의 행동 차이를 드러내기 위한 유효하고 다양한 입력을 생성합니다. 이 방식은 두 가지 맞춤형 피트니스 함수(fitness functions)를 활용하여 GAN 입력 공간을 탐색하고 모델 출력 간의 차이를 식별합니다.

- **Performance Highlights**: DiffGAN은 기존 SOTA(상태 최첨단) 기준인 DRfuzz보다 4배 많은 트리거 입력을 생성하며, 보다 높은 다양성과 유효성을 보입니다. 생성된 입력은 입력 특성에 바탕을 두고 최적의 모델을 선택할 수 있도록 하는 기계 학습 기반 모델 선택 메커니즘을 개선합니다.



### Xeno-learning: knowledge transfer across species in deep learning-based spectral image analysis (https://arxiv.org/abs/2410.19789)
Comments:
          Jan Sellner and Alexander Studier-Fischer contributed equally to this work

- **What's New**: 이 논문에서는 'xeno-learning'이라는 새로운 개념을 제안합니다. 이는 한 종에서 다른 종으로 지식을 전이하는 방식으로, 특히 고유한 임상 데이터를 수집하기 어려운 상황에서 동물 데이터의 활용을 극대화하려는 접근법입니다.

- **Technical Details**: 이 연구는 11,268장의 hyperspectral imaging (HSI) 이미지를 사용하여 사람, 돼지, 쥐 모델 간의 스펙트럼 특성을 비교합니다. 기존의 모델에서는 한 종에서 훈련된 신경망이 다른 종의 조직을 분류하는 데 어려움을 겪으므로, 새로운 'physiology-based data augmentation' 방법을 통해 서로 다른 종 간의 상대 스펙트럼 변화를 학습하고 전이할 수 있는 방안을 제시합니다.

- **Performance Highlights**: 조직 분별 성능은 동물에서 학습한 데이터를 통해 향상되며, 동물 데이터와 인간 데이터를 결합한 신경망 훈련이 유의미한 성과를 보이지 않았습니다. 하지만 상대 스펙트럼 변화를 학습함으로써 특정 병리학적 상태에서의 신경망 성능을 개선할 수 있는 가능성을 보여줍니다.



### Leveraging Multi-Temporal Sentinel 1 and 2 Satellite Data for Leaf Area Index Estimation With Deep Learning (https://arxiv.org/abs/2410.19787)
- **What's New**: 본 논문에서는 생태계 건강과 식물 동태를 이해하기 위해 중요한 매개변수인 Leaf Area Index (LAI)를 예측하는 새로운 방법을 제안합니다. 이 방법은 Sentinel 1 레이더 데이터와 Sentinel 2 다중 스펙트럼 데이터의 보완 정보를 활용하여 픽셀 단위로 LAI를 예측합니다.

- **Technical Details**: 이 방법은 다중 U-net을 기반으로 한 딥 뉴럴 네트워크(dnn)를 사용하여 설계되었습니다. 서로 다른 입력 모달리티(modality)의 복잡성을 다룰 수 있도록 여러 모듈들로 구성되어 있으며, 각 모듈은 공통 잠재 공간(latent space)을 나타내기 위해 개별적으로 사전 훈련(pre-training)됩니다. 이후 계절성(seasonality)을 고려한 공통 디코더로 엔드 투 엔드(end-to-end) 방식으로 미세 조정(fine-tuning)을 진행합니다.

- **Performance Highlights**: 이 방법은 공개 데이터에서 0.06 RMSE와 0.93 R2 점수를 달성하였습니다. 또한, 향후 연구의 발전을 위해 기여한 내용을 해당 링크에서 제공하였습니다.



### Resolution Enhancement of Under-sampled Photoacoustic Microscopy Images using Implicit Neural Representations (https://arxiv.org/abs/2410.19786)
- **What's New**: 본 논문에서는 Acoustic-Resolution Photoacoustic Microscopy (AR-PAM)의 이미징 속도와 해상도를 동시에 향상시키기 위해 Implicit Neural Representations (INR)를 이용하는 접목된 방법을 제안합니다. 기존의 PSF(Point Spread Function) 계측의 어려움을 해결하기 위해 PSF를 학습 가능한 파라미터로 다룸으로써, 더 유연하고 적응력 있는 영상 재구성을 가능하게 합니다.

- **Technical Details**: AR-PAM의 해상도 제약을 극복하기 위해 DECONVOLUTION(복원 알고리즘)을 사용하고, 이를 INR 프레임워크에 통합하여 공간 좌표에서 초기 음압까지의 연속적인 매핑을 학습합니다. 또한, PSF를 학습 가능한 파라미터로 설정하여 PSF 추정의 부정확성을 줄이고, 저샘플링 데이터를 효과적으로 복원합니다.

- **Performance Highlights**: 모의 혈관 데이터에서 우리의 방법은 Peak Signal-to-Noise Ratio (PSNR) 및 Structural Similarity Index (SSIM)에서 기존 방법들에 비해 유의미하게 향상된 결과를 보였으며, 잎 맥관 및 생체 내 쥐의 미세혈관 이미지에서도 정성적인 개선이 관찰되었습니다. 실험에서 우리의 방법은 연필 심을 이용하여 AR-PAM 시스템 내에서 더 선명하고 높은 해상도의 결과를 제공합니다.



### Enhancing Apple's Defect Classification: Insights from Visible Spectrum and Narrow Spectral Band Imaging (https://arxiv.org/abs/2410.19784)
Comments:
          6 pages, 3 figures

- **What's New**: 이 연구는 사과의 결함 분류를 데이터 기반으로 접근하여 경제 손실을 줄이고 식품 공급망을 최적화하는 방법을 제시합니다. 본 연구는 가시 스펙트럼과 660 nm의 스펙트럼 파장을 통합하여 정확도와 효율성을 높이는 혁신적인 방법론을 도입합니다.

- **Technical Details**: 본 연구는 Single-Input 및 Multi-Inputs convolutional neural networks (CNNs)를 활용하여 결함 분류의 정확도를 높이기 위한 다양한 방법론을 검증합니다. 660 nm 스펙트럼을 통한 결함 검출 과정은 기존의 전반적인 가시 스펙트럼보다 미세한 세부사항을 드러내는 것을 보여줍니다. 실험에서는 10769장의 이미지 데이터셋을 활용하여 다양한 결함을 분류하는 CNN 모델을 학습합니다.

- **Performance Highlights**: MobileNetV1 모델은 검증 데이터셋에서 98.80%의 정확도를 달성했으며, 이는 전체 가시 스펙트럼을 사용했을 때의 98.26%보다 높은 수치입니다. 이러한 결과는 특정 스펙트럼 범위를 활용한 이미지 캡처가 분류 작업의 네트워크 훈련을 더 효과적으로 할 수 있는 가능성을 시사합니다.



### Data-Driven Uncertainty-Aware Forecasting of Sea Ice Conditions in the Gulf of Ob Based on Satellite Radar Imagery (https://arxiv.org/abs/2410.19782)
- **What's New**: 이번 연구에서는 Sentinel-1의 레이더 이미지, 기상 관측 및 GLORYS 예측을 활용해 해빙 조건 예측을 위한 새로운 데이터 기반 접근 방식을 제안합니다. 이 방법은 북극 해빙 역학의 고유한 도전에 맞춘 데이터 전처리 및 증강 기술과 결합된 첨단 비디오 예측 모델을 통합하여 신뢰할 수 있는 단기 해빙 예측을 가능하게 합니다.

- **Technical Details**: 우리는 전통적인 해빙 모델의 한계를 극복하고, 머신러닝 기법을 활용하여 해빙 조건 예측을 개선합니다. 사용할 기술은 자동회귀 방식의 비디오 예측 모델(IAM4VP), 다중 스케일 복셀 흐름 네트워크(DMVFN), 운동 RNN, Neural ODE 등이 포함됩니다. 예측의 신뢰성을 평가하기 위해 불확실성 정량화 기법을 사용하는 한편, 신뢰 기반 모델 혼합 메커니즘을 제안합니다.

- **Performance Highlights**: 우리의 연구 결과는 기본 방법에 비해 상당한 개선을 보여줍니다. 예측 모델의 정확성과 강인성이 향상되었으며, 해빙 조건의 정확한 매핑을 위한 세밀한 공간 구조를 반영할 수 있도록 하였습니다. 여러 중요한 지표를 통해 예측 정확도를 평가하며, 데이터 결측 및 불균형 문제를 해결하는 접근 방식의 우수성을 입증합니다.



### Copula-Linked Parallel ICA: A Method for Coupling Structural and Functional MRI brain Networks (https://arxiv.org/abs/2410.19774)
Comments:
          25 pages, 10 figures, journal article

- **What's New**: 본 연구에서는 깊은 학습(frameworks) 기술과 copula, 독립 성분 분석(ICA)을 결합한 새로운 융합(fusion) 방법인 CLiP-ICA를 개발했습니다.

- **Technical Details**: CLiP-ICA는 기능적 자기 공명 영상(fMRI)과 구조적 자기 공명 영상(sMRI)의 독립적 원천을 추정하고, copula 기반 모델을 사용하여 fMRI와 sMRI의 공간적 원천을 연결하여 시간적 및 공간적 데이터를 더 유연하게 통합합니다.

- **Performance Highlights**: CLiP-ICA는 알츠하이머병 신경영상 이니셔티브(ADNI)의 데이터를 사용한 결과, 감각운동, 시각, 인지 제어 및 기본 모드 네트워크를 포함한 강하게 및 약하게 연결된 sMRI와 fMRI 네트워크를 효과적으로 캡처했습니다. 이 방법은 더 의미 있는 구성 요소를 발견하고, ICA의 최적 모델 순서 문제를 해결했습니다.



### Developing Gridded Emission Inventory from High-Resolution Satellite Object Detection for Improved Air Quality Forecasts (https://arxiv.org/abs/2410.19773)
- **What's New**: 본 연구는 WRF Chem 모델을 위해 동적 AI 기반의 배출 목록 시스템을 구축하는 혁신적인 접근 방식을 제시합니다. 이 시스템은 위성 데이터로부터 차량 및 기타 인위적 배출을 시뮬레이션하여 정확한 실시간 배출 데이터를 생성할 수 있도록 설계되었습니다.

- **Technical Details**: 이 연구는 YOLO (You Only Look Once) 아키텍처(v8에서 v10)와 T-Rex를 사용하여 차량 탐지의 정밀도를 높이는 데 초점을 맞춥니다. 극도로 고해상도의 위성 이미지를 사용하여 차량 수를 추적하고 이를 배출 계수와 직접 연결하여, 매일 또는 하위 일일 단위로 공기 질 모델에 피드를 제공하는 동적 배출 목록을 생성합니다.

- **Performance Highlights**: 이 시스템은 초기 F1 점수 0.15에서 0.72로 크게 향상되었습니다. 그 결과, 도시의 배출 동역학에 대한 깊은 통찰력을 제공하며, 향후 연구에서는 비차량 배출원으로의 확장과 어려운 환경 조건에서의 탐지 정확도 향상에 초점을 맞출 예정입니다.



### Large Model for Small Data: Foundation Model for Cross-Modal RF Human Activity Recognition (https://arxiv.org/abs/2410.19766)
- **What's New**: 이 논문은 Radio-Frequency (RF) 기반의 Human Activity Recognition (HAR) 시스템을 향상시키기 위해 시각 기반의 Foundation Models (FMs)의 지식을 전이하는 새로운 cross-modal framework인 FM-Fi를 제안합니다. FM-Fi는 RF 인코더가 FMs의 해석력을 물려받아 zero-shot learning을 달성할 수 있도록 돕습니다.

- **Technical Details**: FM-Fi는 novel cross-modal contrastive knowledge distillation (CKD) 메커니즘을 통해 RF와 시각 데이터 간의 효과적인 지식 전이를 달성하며, FM과 RF의 고유한 기능을 이용해 두 개체 간의 정렬을 개선합니다. 또한, metric-based few-shot learning 기술을 통하여 특정 HAR 작업에 대한 성능을 향상시킵니다.

- **Performance Highlights**: FM-Fi의 평가 결과는 시각 기반 방법론과 동등한 성능을 보여주며, 다양한 환경에서의 일반화 가능성을 실증적으로 검증합니다.



### Reliable, Routable, and Reproducible: Collection of Pedestrian Pathways at Statewide Sca (https://arxiv.org/abs/2410.19762)
Comments:
          arXiv admin note: text overlap with arXiv:2303.02323

- **What's New**: 이 논문에서는 장애인을 위한 이동성 형평성을 증대시키기 위한 자동화된 보행 경로 데이터 수집 및 관리 기법을 제안합니다. 기존 데이터 수집의 한계를 극복하기 위해 공중 영상과 도로 네트워크 데이터를 결합한 새로운 방법론을 도입하여 워싱턴주 전역의 보행 경로를 생성하는 목표로 합니다.

- **Technical Details**: 논문에서는 Prophet 시스템이라는 자동화된 예측 모델을 사용하여 공중 이미지에서 보행 경로, 길 건너기, 경계턱을 추출하고, 이를 통해 연결된 경로 네트워크를 생성합니다. 전문가의 수동 검증을 위한 단계적 프로세스인 Skeptic을 운영하여 데이터 품질을 보장하고 있으며, 다수의 전문가와 지역 사회의 의견을 통합하여 데이터 수집의 정확성을 높이고 있습니다.

- **Performance Highlights**: 자동화된 시스템이 최신의 방법들보다 우수한 성능을 보여주며 인간 검증에 필요한 시간을 상당히 줄일 수 있음을 입증했습니다. 이 연구는 주 전체에서 신뢰할 수 있고 강력한 보행 경로 네트워크를 생성할 수 있는 가능성을 보여주며, 이러한 접근법은 전국적인 ADA 준수 절차를 알리는 데 기여할 것입니다.



### Movie Trailer Genre Classification Using Multimodal Pretrained Features (https://arxiv.org/abs/2410.19760)
- **What's New**: 본 논문은 다양한 사전 훈련된(pretrained) 모델을 활용하여 영화 장르 분류에 대한 새로운 방법론을 제시합니다. 이 방법은 모든 비디오 및 오디오 프레임을 사용하여 장르를 예측하며, 기존의 전통적인 방법보다 더 높은 정확성을 보입니다.

- **Technical Details**: 우리의 접근 방식은 transformer 모델을 사용하여 다양한 작업과 양식에서 오는 사전 훈련된 특징(feature)을 융합합니다. 이 모델은 비디오 예고편의 모든 프레임과 오디오를 처리하며, 고정되지 않은 많고 다양한 특징을 효과적으로 처리합니다. 특히, 전통적인 방식이 사용하는 고정된 낮은 수의 프레임과는 다릅니다.

- **Performance Highlights**: 본 연구는 MovieNet 데이터셋에서 기존의 영화 장르 분류 모델보다 정밀도(precision), 재현율(recall), 평균 평균 정밀도(mean average precision, mAP)에서 월등한 성과를 기록하였습니다. 이를 통해, 향후 연구를 위한 사전 훈련된 특징과 모델 코드를 공개하여 연구자들이 활용할 수 있도록 하였습니다.



### PINNing Cerebral Blood Flow: Analysis of Perfusion MRI in Infants using Physics-Informed Neural Networks (https://arxiv.org/abs/2410.19759)
- **What's New**: 이 연구에서는 아기 ASL 데이터에서 뇌혈류(CBF) 및 기타 매개변수를 정확하게 추정하기 위해 새로운 공간 불확실성 기반 물리정보 신경망(PINN)인 SUPINN을 제안합니다.

- **Technical Details**: SUPINN은 여러 브랜치 아키텍처(multi-branch architecture)를 사용하여 여러 복셀(voxel)에서 지역적 및 글로벌 모델 매개변수를 동시에 추정합니다. 이 네트워크는 지역적인 공간 불확실성을 계산하여 신호의 가중치를 조정합니다. SUPINN은 CBF를 신뢰성 있게 추정하며, 이는 -0.3 ± 71.7의 상대 오차로 나타납니다.

- **Performance Highlights**: SUPINN은 기존의 최소제곱법(least squares)이나 표준 PINN을 사용한 매개변수 추정보다 우수한 성능을 보였으며, 생리학적으로 그럴듯한 공간적으로 부드러운 CBF 및 AT 맵을 생성할 수 있습니다. 이 연구는 아기 ASL 데이터에서 시끄럽고 제한된 정보를 바탕으로 정확한 다중 매개변수 관류 추정을 위한 PINN의 성공적인 수정 사례를 보여줍니다.



### A SAM based Tool for Semi-Automatic Food Annotation (https://arxiv.org/abs/2410.19756)
Comments:
          Accepted Demo Paper - ECAI 2024

- **What's New**: 이번 논문에서는 Segment Anything Model (SAM)을 활용한 반자동 음식 이미지 주석 도구의 데모를 제시합니다. 사용자가 상호작용을 통해 음식 분할을 할 수 있도록 하여, 비전문가도 쉽게 사용할 수 있는 AI 도구의 필요성을 강조합니다.

- **Technical Details**: 제안된 도구는 사용자의 요청에 따라 음식 이미지를 신속하게 분할해 주며, MealSAM이라는 fine-tuned 된 SAM의 mask decoder를 사용하여 음식 이미지 분할에 최적화되어 있습니다. 이 도구는 ViT-B 백본을 사용하고 있습니다.

- **Performance Highlights**: 본 연구의 목표는 음식 데이터의 주석 작업을 통해 참여와 협업을 증진시키고, AI 기술을 보다 많은 사용자에게 제공하여 영양 과학 분야에서도 쉽게 활용할 수 있도록 하는 것입니다.



### C^2DA: Contrastive and Context-aware Domain Adaptive Semantic Segmentation (https://arxiv.org/abs/2410.19748)
Comments:
          This paper has 16 pages, 6 figures, 5 tables. It has been accepted for publication at the International Symposium of Robotics Research (ISRR), Long Beach, California, USA, 2024

- **What's New**: 본 연구에서는 Unsupervised Domain Adaptive Semantic Segmentation (UDA-SS)의 새로운 프레임워크를 제안합니다. 이 프레임워크는 intra-domain 및 context-aware 지식을 학습하여 소스 도메인과 타겟 도메인 간의 데이터 이동 문제를 해결하고자 합니다.

- **Technical Details**: 제안된 방법에서는 contrastive loss를 소스 및 타겟 도메인에 통합하여 유사 클래스의 픽셀을 서로 밀어내고 다른 클래스의 픽셀은 떨어지도록 하는 방식을 사용합니다. 또한, ClassMix 기법을 수정하고, Mask Image Modeling (MIM) 기술을 채택하여 제한된 정보로부터 시각 인식을 향상시킵니다.

- **Performance Highlights**: GTA-V->Cityscapes와 Synthia->Cityscapes 데이터셋에서 0.51% mIoU 및 0.54% mIoU의 성능 향상을 보이며, 최첨단 UDA-SS 방법의 성능을 초월하였습니다.



### Adaptive Real-Time Multi-Loss Function Optimization Using Dynamic Memory Fusion Framework: A Case Study on Breast Cancer Segmentation (https://arxiv.org/abs/2410.19745)
- **What's New**: 본 논문에서는 실시간으로 다중 손실 함수에 대한 적응형 패널라이징(dynamic memory fusion) 프레임워크를 제안합니다. 이 프레임워크는 이전 손실 값을 활용하여 훈련 과정 내내 다중 손실 함수의 가중치를 동적으로 조정합니다. 또한 초기 단계에서 모델 성능을 향상시키기 위해 보조 손실 함수를 통합합니다.

- **Technical Details**: 우리의 접근법인 DMF(동적 기억 융합) 프레임워크는 다수의 손실 함수 가중치를 훈련 중에 동적으로 조정하기 위해 역사적 손실 데이터(history loss data)를 활용합니다. 이는 고정 가중치 접근 방식과 달리 훈련 중 다양한 요소에 균형을 맞추어 모델이 특정 요소에 심하게 편향되지 않도록 보장합니다. 또한, 클래스 불균형(class imbalance)을 해결하기 위한 class-balanced dice loss 함수를 도입하여 적은 대표성을 가진 클래스에 우선 순위를 부여합니다.

- **Performance Highlights**: 유방 초음파 데이터셋(breast ultrasound datasets)에서의 실험 결과, 제안한 프레임워크는 다양한 메트릭(metric)에서 세분화 성능(segmentation performance)을 향상시킴을 보여줍니다. 이러한 결과는 모델이 가장 관련 있는 기준을 우선시하도록 동적으로 조정함으로써 변화하는 환경에서 향상된 성능에 기여한다는 것을 입증합니다.



### GPT-4o System Card (https://arxiv.org/abs/2410.21276)
- **What's New**: GPT-4o는 텍스트, 오디오, 이미지, 비디오 등 다양한 입력을 지원하는 자가 회귀형 omni 모델로, 통합된 신경망을 통해 텍스트, 오디오, 이미지 출력을 생성합니다. 이 모델은 대화에서 인간의 반응 시간과 유사하게 평균 320밀리초로 오디오 입력에 응답할 수 있어 향상된 반응성을 보여줍니다.

- **Technical Details**: GPT-4o는 텍스트와 음성 기능이 2023년 10월까지의 다양한 자료를 바탕으로 사전 훈련되었습니다. 웹 데이터, 코드 및 수학, 다중 모달 데이터가 포함되어 있어 모델이 비텍스트 입력과 출력을 해석하고 생성하는 방법을 학습하게 됩니다. 또한, 위험을 줄이기 위한 방법으로는 Moderation API와 안전 분류기를 사용하여 유해 콘텐츠를 필터링합니다.

- **Performance Highlights**: GPT-4o는 영어 및 코드 텍스트에서 GPT-4 Turbo의 성능을 맞추며 비영어 텍스트에서는 현저한 개선을 보여줍니다. API 비용은 50% 절감되며, 비전 및 오디오 이해 능력이 기존 모델보다 뛰어난 것으로 확인되었습니다.



### OmniSep: Unified Omni-Modality Sound Separation with Query-Mixup (https://arxiv.org/abs/2410.21269)
Comments:
          Working in progress

- **What's New**: Omni-modal Sound Separation (OmniSep) 모델을 소개하여 다양한 모달리티의 쿼리를 기반으로 명확한 오디오 트랙을 분리할 수 있는 새로운 프레임워크를 제안합니다.

- **Technical Details**: OmniSep은 Query-Mixup 전략을 활용하여 훈련 중 서로 다른 모달리티의 쿼리 기능을 혼합하며, 이를 통해 여러 모달리티를 동시에 최적화합니다. 또한, Query-Aug(검색 강화) 접근 방식을 통해 개방형 어휘(sound separation) 적용을 가능하게 합니다.

- **Performance Highlights**: MUSIC, VGGSOUND-CLEAN+, MUSIC-CLEAN+ 데이터셋에서의 실험 결과 OmniSep은 TQSS, IQSS, AQSS 등 다양한 사운드 분리 작업에서 최고의 성능을 보여줘 Omni-modal sound separation 분야에서의 선두 주자입니다.



### Multi-modal AI for comprehensive breast cancer prognostication (https://arxiv.org/abs/2410.21256)
- **What's New**: 이 연구는 디지털 병리학(digital pathology) 및 임상 특성을 이용하여 유방암 환자를 분류하기 위한 새로운 AI 테스트 개발에 관한 것입니다. 이 테스트는 자기 지도 학습(self-supervised learning)에 기반한 비전 트랜스포머(vision transformer) 모델을 활용하여 개발되었습니다.

- **Technical Details**: AI 테스트는 H&E 염색 슬라이드에서 특징을 추출하고, 이를 기존의 임상 데이터와 통합하여 다중 모달(multi-modal) AI 테스트를 생성합니다. 총 8,161명의 유방암 환자 데이터를 기반으로 개발되었으며, 다양한 유방암 아형에서 뚜렷한 예측 성과를 보였습니다.

- **Performance Highlights**: AI 테스트는 5개의 외부 집단에서 질병 없는 간격(disease-free interval, DFI)에 대해 C-인덱스(C-index) 0.71 [0.68-0.75]를 기록하며, 기존의 Oncotype DX보다 높은 정확도를 보여주었습니다. 특히, 삼중 음성 유방암(TNBC)에서도 C-인덱스 0.71 [0.62-0.81]을 달성했습니다.



### Document Parsing Unveiled: Techniques, Challenges, and Prospects for Structured Information Extraction (https://arxiv.org/abs/2410.21169)
- **What's New**: 이번 논문은 문서 파싱(document parsing)의 최신 발전 상태를 포괄적으로 리뷰하며, 대형 언어 모델과 비전-언어 모델을 활용한 문서 파싱 기술의 중요성을 강조합니다.

- **Technical Details**: 주요 방법론으로는 모듈형 파이프라인 시스템(modular pipeline systems)과 end-to-end 모델이 있습니다. 핵심 구성 요소에는 레이아웃 감지(layout detection), 콘텐츠 추출(content extraction), 멀티모달 데이터 통합(multi-modal data integration)이 포함됩니다. 또한 복합 레이아웃을 처리하는 데 필요한 도전과제와 기술적 발전도 다루어집니다.

- **Performance Highlights**: 문서 파싱 기술이 정보 추출 및 문서 이해와 같은 여러 작업에서 중대한 진전을 이루었으며, 이는 RAG 시스템 및 차세대 지능형 시스템 개발을 위한 견고한 기초를 제공합니다.



### KaLDeX: Kalman Filter based Linear Deformable Cross Attention for Retina Vessel Segmentation (https://arxiv.org/abs/2410.21160)
- **What's New**: 이 연구에서는 혈관 세분화를 위해 새로운 네트워크 구조인 KaLDeX를 제안합니다. 이 구조는 Kalman filter 기반의 linear deformable cross attention 모듈을 통합하여 retinal vessel의 미세하고 섬세한 세분화를 이루는 것을 목표로 합니다.

- **Technical Details**: KaLDeX 네트워크는 Kalman filter (KF) 기반의 linear deformable convolution (LD) 모듈과 cross-attention (CA) 모듈로 구성됩니다. LD 모듈은 표준 convolution에서 간과될 수 있는 얇은 혈관에 대한 초점을 조정하고, CA 모듈은 UNet++의 고수준 특성과 LD 모듈의 상세한 특성을 결합하여 혈관 구조에 대한 전반적인 이해를 개선합니다.

- **Performance Highlights**: 제안된 방법은 여러 레티날 혈관 이미지 데이터셋(DRIVE, CHASE_BD1, STARE)과 OCTA-500 데이터셋에서 평균 정확도(ACC)가 각각 97.25%, 97.77%, 97.85%, 98.89%, 98.21%를 달성하여 현재의 최고 모델을 초과하는 성능을 보였습니다.



### Shallow Diffuse: Robust and Invisible Watermarking through Low-Dimensional Subspaces in Diffusion Models (https://arxiv.org/abs/2410.21088)
- **What's New**: 이 논문에서는 AI 생성 콘텐츠 식별을 위한 새로운 워터마킹 기법인 Shallow Diffuse를 소개합니다. 기존 방법들과 달리 Shallow Diffuse는 샘플링 과정 전반에 걸쳐 워터마킹을 통합하는 것이 아니라, 이미지 생성 과정에서 저차원 부분공간을 활용하여 이 두 단계를 분리합니다. 이는 워터마크가 이미지 생성 과정과 효과적으로 분리되도록 보장합니다.

- **Technical Details**: Shallow Diffuse는 높은 강건성과 일관성을 갖는 워터마킹 기법으로, 서버 시나리오(서버가 초기 시드를 제공할 때)와 사용자 시나리오(생성된 이미지에 워터마크를 삽입할 때) 모두에 적용 가능합니다. 이 기법은 이전 방식들과 달리 초기 랜덤 시드에 워터마크를 삽입하는 것이 아니라, 이미지 생성 과정에서 낮은 차원의 부분공간을 활용하여 워터마크의 상당 부분이 해당 부분공간의 널 공간에 위치하도록 합니다. 이는 샘플링 과정과 워터마킹 과정을 실질적으로 분리합니다.

- **Performance Highlights**: Shallow Diffuse는 기존의 워터마킹 방법들과 비교했을 때 큰 강건성과 일관성을 보여줍니다. 실험 결과는 이 방법이 기존 방법들보다 우수한 성능을 발휘함을 입증하였으며, 상당한 양의 데이터 생성을 일관되게 유지하고 워터마크의 감지 가능성을 향상시켰습니다.



### Efficient Bilinear Attention-based Fusion for Medical Visual Question Answering (https://arxiv.org/abs/2410.21000)
- **What's New**: 이 논문에서는 Orthogonality loss, Multi-head attention, Bilinear Attention Network을 통합한 새로운 Fusion 모델 OMniBAN을 제안합니다. 이 모델은 사전 학습 없이도 높은 계산 효율성과 강력한 성능을 달성하도록 설계되었습니다.

- **Technical Details**: OMniBAN은 이미지와 텍스트의 복잡한 상호작용을 캡처하는 Bilinear Attention을 활용하며, Multi-head attention을 통해 다양한 시각적 정보를 처리합니다. 이를 통해 모델은 의료 영상 질문 응답(MedVQA)에서의 성능을 극대화할 수 있습니다.

- **Performance Highlights**: OMniBAN은 여러 MedVQA 벤치마크에서 기존 모델보다 뛰어난 성능을 발휘하며, 낮은 계산 비용을 유지합니다. 이는 자원 제약이 있는 임상 환경에서의 효율적 적용 가능성을 보여줍니다.



### BEVPose: Unveiling Scene Semantics through Pose-Guided Multi-Modal BEV Alignmen (https://arxiv.org/abs/2410.20969)
Comments:
          Accepted for presentation at the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2024. Project page: this https URL

- **What's New**: 최근 자율주행 및 모바일 로보틱스 분야에서 Bird's Eye View (BEV) 표현 방법에 있어 혁신적인 변화가 일어나고 있습니다. 주로 LiDAR와 카메라로부터 이질적인 센서의 측정값을 융합하여 2D 평면 기반 표현으로 변환하는 과정에 Transformer를 사용하는 것이 그 특징입니다. BEVPose라는 새로운 프레임워크를 통해 이 과정에서 고비용의 주석 데이터 의존성을 크게 줄일 수 있는 방법을 제시합니다.

- **Technical Details**: BEVPose는 카메라와 LiDAR 데이터를 통합하여 BEV 표현을 생성하고, 센서 포즈를 유도 신호로 사용하여 다중 모드 감지 입력을 정렬하고 융합합니다. 이로 인해 환경의 기하학적 및 의미론적 측면을 모두 포착할 수 있는 잠재적인 BEV 임베딩을 학습할 수 있습니다. 이 접근법은 맵 세그멘테이션 및 세부 조정 작업에서 우수한 성능을 보입니다.

- **Performance Highlights**: BEVPose는 최소한의 주석 데이터만을 사용하여 기존의 완전 감 supervised 방법들을 초월하는 성능을 보여줍니다. 이 기술은 BEV 표현 학습에서 데이터 효율성 문제를 해결하고, 오프로드 및 실내 환경과 같은 다양한 분야에 적용할 수 있는 가능성을 확대합니다.



### FreqMark: Invisible Image Watermarking via Frequency Based Optimization in Latent Spac (https://arxiv.org/abs/2410.20824)
- **What's New**: 이번 논문에서 제안하는 FreqMark는 이미지의 잠재 주파수 공간에서 watermark를 삽입하는 새로운 방법으로, 기존의 방법들에 비해 강력한 회복력을 자랑합니다. 이는 VAE(Variational Autoencoder) 인코딩 이후의 이미지 잠재 주파수 공간을 최적화하여 이루어집니다.

- **Technical Details**: FreqMark는 이미지 라틴 주파수 공간의 비가공 최적화를 통해 watermark를 삽입합니다. 송신된 watermark는 고정된 사전 훈련된 이미지 인코더를 통해 추출됩니다. 이러한 최적화로 인해 이미지 품질과 watermark 강건성 사이의 유연한 트레이드오프가 가능하여 다양한 공격에 효과적으로 저항할 수 있습니다.

- **Performance Highlights**: 실험 결과, FreqMark는 이미지 품질과 강건성 모두에서 중요한 이점을 제공하며, 48-bit 숨겨진 메시지를 인코딩할 경우 다양한 공격 시나리오에서 90% 이상의 비트 정확도를 달성합니다. 특히 FreqMark는 재생산 공격에 대한 저항력이 우수하여 기존 방법들보다 나은 성능을 나타냅니다.



### Scaling-based Data Augmentation for Generative Models and its Theoretical Extension (https://arxiv.org/abs/2410.20780)
- **What's New**: 이번 논문에서는 고품질 데이터 생성을 위한 생성 모델의 안정적인 학습 방법에 대해 연구하였습니다. 최근 개발된 방법인 Diffusion-GAN은 타임스텝 의존형 분류기를 사용하는 확산 프로세스를 통해 이 문제를 해결합니다. 이를 통해 데이터 스케일링이 안정적인 학습과 고품질 데이터 생성을 위한 핵심 요소임을 밝혀냅니다.

- **Technical Details**: Diffusion-GAN은 생성자(G)와 분류기(D)를 기반으로 하는 min-max 최적화 문제로 구성됩니다. 데이터 스케일링을 통해 진짜 데이터와 가짜 데이터 간의 격차를 줄이며, 노이즈 주입을 통해 안정성을 높입니다. 이 과정을 통해 데이터의 히스토그램을 정규화하여 안정적인 학습을 달성합니다. 이 연구에서는 Scale-GAN이라는 학습 알고리즘을 제안하며, 이는 데이터 스케일링과 분산 기반 정규화를 활용합니다.

- **Performance Highlights**: 제안된 Scale-GAN 알고리즘은 이미지 생성 벤치마크 데이터 셋에서 기존 방법들보다 뛰어난 성능을 보여주었으며, 안정성 및 정확성을 향상시키는 것으로 나타났습니다. 특히 데이터 스케일링이 노이즈 주입보다 안정화에 더 크게 기여함을 입증하였습니다.



### CardiacNet: Learning to Reconstruct Abnormalities for Cardiac Disease Assessment from Echocardiogram Videos (https://arxiv.org/abs/2410.20769)
Comments:
          Paper Accepted by ECCV 2024 with Oral Presentation

- **What's New**: 이번 연구에서는 심장 기능 분석 및 심장 질환 진단에 중요한 에코카디오그램(echocardiogram) 비디오를 기반으로 한 새로운 접근법인 CardiacNet을 제안했습니다. CardiacNet은 심장 구조와 운동 비정상성에 대한 향상된 표현을 학습하기 위해 새로운 복원(reconstruction) 기반 접근법을 활용합니다.

- **Technical Details**: CardiacNet은 Consistency Deformation Codebook (CDC)와 Consistency Deformed-Discriminator (CDD)를 포함하고 있으며, 이러한 구조는 비정상 샘플과 정상 샘플 간의 공통점을 학습하도록 설계되었습니다. 이를 통해 심장 질환 평가를 위한 두 가지 벤치마크 데이터셋인 CardiacNet-PAH와 CardiacNet-ASD를 만들었습니다.

- **Performance Highlights**: 실험 결과, CardiacNet은 공개 데이터셋 CAMUS, EchoNet 및 자체 데이터셋에서 EF, PAH 및 ASD와 같은 심장 질환 평가 작업에서 최신 기술(state-of-the-art) 결과를 달성했습니다. PAH 및 ASD 분류의 정확성을 각각 2.1% 및 5.0% 향상시키며, EF 예측 작업에서도 기존 연구보다 5.2% 개선된 결과를 보였습니다.



### Video to Video Generative Adversarial Network for Few-shot Learning Based on Policy Gradien (https://arxiv.org/abs/2410.20657)
Comments:
          18 pages, 11 figures, submitting to IEEE TNNLS

- **What's New**: 이 연구는 강화학습(Deep Reinforcement Learning)과 생성적 적대 신경망(Generative Adversarial Networks, GANs)을 활용하여 새로운 비디오-비디오 합성을 위한 방법론인 RL-V2V-GAN을 제안합니다. 이 모델은 비지도 조건부 비디오-비디오 합성(non-supervised conditional video-to-video synthesis)을 가능하게 합니다.

- **Technical Details**: RL-V2V-GAN은 정책 기울기(policy gradient) 학습을 통해 훈련되며, 공간 및 시간 정보를 캡처하기 위해 ConvLSTM 레이어를 사용합니다. 이 모델은 하나의 출처 비디오 도메인에서 목표 비디오 도메인으로의 매핑을 학습하며, 상대적 스타일을 유지합니다. 그 과정에서 spatio-temporal adversarial 목표를 통합하여 생성적 적대 손실(adversarial losses)을 이용해 콘텐츠 전송과 스타일 보존을 동시에 이룹니다.

- **Performance Highlights**: RL-V2V-GAN은 기존의 쌍 입력을 요구하는 방법들과는 달리 비지도 방식으로 작동하여, 몇 가지 샘플의 목표 도메인을 다룰 때 특히 효과적입니다. 실험 결과는 이 모델이 시간이 일관된 비디오 결과를 생성할 수 있음을 보여줍니다.



### Neural rendering enables dynamic tomography (https://arxiv.org/abs/2410.20558)
Comments:
          24 pages, 14 figures. Submitted to NeurIPS 2024 ML4PS. For associated visualizations, see this https URL

- **What's New**: 본 연구는 X선 컴퓨터 단층 촬영(X-CT)이 동적 실험 동안 3D 재구성을 가능하게 하도록 신경 렌더링 도구를 활용하는 방법을 제안합니다. 기존의 정적 실험에서는 효과적이었던 접근 방식이 동적 실험에서는 재구성이 불가능했기 때문에, 이러한 패러다임 전환이 이루어졌습니다.

- **Technical Details**: 신경 방사선 필드(neural radiance fields)는 기존의 재구성 방법보다 관심 있는 데이터 모달리티를 더 효율적으로 재구성할 수 있음을 보여줍니다. 이 연구에서는 입장 기준(projection angles)을 선택하기 위한 이론적 결과를 도출하고, 합성 데이터와 실험적 데이터를 결합하여 3D 밀도 필드를 훈련시키는 차별화된 프로젝터/렌더러를 사용합니다. 또한, 스플라인 기반 변형 필드(spatio-temporal model with spline-based deformation field)를 개발하여 실제 실험에서 격자 샘플의 시공간 변형을 재구성하는 방법을 제시합니다.

- **Performance Highlights**: 본 연구는 두 개의 프로젝션만으로도 격자 구조의 변형 사건을 재구성할 수 있다는 것을 입증하였으며, 전통적인 컴퓨터 단층 촬영과 신경 렌더링을 결합함으로써 시너지를 창출해냄으로써 동적 사건의 재구성이 가능함을 보여줍니다.



### Sebica: Lightweight Spatial and Efficient Bidirectional Channel Attention Super Resolution Network (https://arxiv.org/abs/2410.20546)
Comments:
          7 pages, 5 figures, 26 conferences

- **What's New**: Sebica는 경량 네트워크로, Bidirectional Channel Attention 메커니즘을 사용하여 저해상도 이미지를 고해상도로 변환하는 성능을 크게 향상시키고, 기존 모델보다 계산 비용을 줄이면서도 높은 품질을 유지합니다.

- **Technical Details**: Sebica는 Spatial과 Efficient Bidirectional Channel Attention 메커니즘을 통합하여 이미지의 해상도 변환 시 품질과 속도 간의 균형을 맞추었습니다. 이 네트워크는 채널 간 중요도를 동적으로 조정하며, Residual 연결을 통해 학습 안정성을 높이고 PixelShuffle 연산을 통해 효율적인 이미지 업스케일링을 수행합니다.

- **Performance Highlights**: Sebica는 Div2K 및 Flickr2K 데이터셋에서 각각 28.29/0.7976 및 30.18/0.8330의 PSNR/SSIM 점수를 기록하였으며, 이는 많은 경량 모델보다 우수한 결과입니다. 특히, Sebica의 소형 버전은 7.9K 파라미터와 0.41 GFLOPS에 불과하나, 여전히 28.12/0.7931 및 0.3009/0.8317의 PSNR 및 SSIM 성능을 보여줍니다. 또한, 교통 비디오 객체 탐지 작업에서 정확도를 향상시키는 성과를 보여주었습니다.



### Search Wide, Focus Deep: Automated Fetal Brain Extraction with Sparse Training Data (https://arxiv.org/abs/2410.20532)
- **What's New**: 이번 연구는 sparse하고 synthetic한 라벨로 학습된 네트워크의 false positive를 줄이는 테스트 타임 전략을 제안합니다. 이 방법은 breadth-fine search(BFS)와 deep-focused sliding window(DFS)를 활용하여 태아 뇌를 더욱 정확하게 추출합니다.

- **Technical Details**: 이 프레임워크는 3D multi-scale deep neural networks를 사용하여 full-uterus Stack-of-slices MRI로부터 태아 뇌 마스크를 추출합니다. BFS와 DFS를 통해 검색 범위를 조정하고, 각기 다른 크기의 윈도우를 사용하여 다양한 모델을 트레이닝함으로써 데이터에 대한 일반화를 이루어냅니다.

- **Performance Highlights**: 이 방법은 세 번째 삼 분기 임신 태아에 대해 기존의 최신 방법과 동등한 성능을 보였으며, 두 번째 삼 분기에서는 Dice 점수에서 최대 5% 향상된 결과를 보여주었습니다. 이는 모델의 정확성을 높이고 태아 뇌를 더욱 효과적으로 분리하는 데 기여했습니다.



### What Factors Affect Multi-Modal In-Context Learning? An In-Depth Exploration (https://arxiv.org/abs/2410.20482)
Comments:
          Accepted at NeurIPS 2024

- **What's New**: 최근 다중 모달 인컨텍스트 학습(Multi-Modal In-Context Learning, MM-ICL)에서의 빠른 발전이 주목받고 있습니다. 본 연구는 MM-ICL의 성능에 영향을 미치는 요인을 조사하여, 효과적인 전략을 최적화하기 위한 기초 자료를 제공하고자 합니다.

- **Technical Details**: 본 연구는 MM-ICL의 세 가지 핵심 단계인 시연 검색(demonstration retrieval), 시연 순서(demonstration ordering), 프롬프트 구성(prompt construction)에 대한 포괄적인 실험을 수행했습니다. 실험은 6개의 시각 대형 언어 모델(vision large language models, VLLMs)과 20가지 전략을 사용하여 실시되었습니다. 연구 결과는 다중 모달 검색기가 필요하며, 시연의 내부 순서(intra-demonstration ordering)가 외부 순서(inter-demonstration ordering)보다 중요함을 강조하고, 소개 지시문(introductory instructions)이 프롬프트에 포함됨으로써 작업 설명(task comprehension)이 향상된다는 것을 보여줍니다.

- **Performance Highlights**: 1. 다중 모달 검색 방법이 단일 모달 방법보다 평균적으로 더 뛰어난 성능을 보임. 2. 내부 시연 순서가 모델 성능에 미치는 영향이 외부 시연 순서보다 크게 나타남. 3. 소개 지시문을 포함할 경우 MM-ICL 성능이 일관되게 향상됨.



### Guidance Disentanglement Network for Optics-Guided Thermal UAV Image Super-Resolution (https://arxiv.org/abs/2410.20466)
Comments:
          18 pages, 19 figures, 8 tables

- **What's New**: 본 연구에서는 Guidance Disentanglement network (GDNet)을 제안하여 다양한 UAV 상황에서의 효율적인 특징 생성을 통해 Thermal UAV 이미지 슈퍼 해상도(OTUAV-SR)의 성능을 향상시킵니다. GDNet은 광학 이미지의 표현을 일반적인 UAV 시나리오 속성에 따라 분리하여 우호적 및 불리한 조건 모두에서 유효한 가이던스 특징을 생성합니다.

- **Technical Details**: GDNet은 세 가지 서브 네트워크로 구성된 Attribute-specific Guidance Module (AGM)을 통해 다양한 시나리오에 따른 광학 이미지를 모델링하며, Attribute-aware Fusion Module (AFM)을 사용하여 여러 가이드 브랜치에서의 특징을 적응적으로 집계합니다. 또한 Overlapping Multi-head Cross-Attention Layer (OMCL)을 설계하여 멀티모달 정보의 통합을 개선하고, VGTSR2.0이라는 대규모 멀티모달 UAV 데이터셋을 제공합니다.

- **Performance Highlights**: VGTSR2.0에 대한 광범위한 실험 결과 GDNet이 기존 최첨단의 SISR 및 OTUAV-SR 방법보다 성능이 우수함을 입증하였으며, 특히 저조도 및 안개와 같은 도전적인 환경에서 더욱 두드러진 성능 개선을 보였습니다.



### Vector Quantization Prompting for Continual Learning (https://arxiv.org/abs/2410.20444)
Comments:
          To appear in NeurIPS 2024

- **What's New**: 최근 발표된 VQ-Prompt는 Vector Quantization(VQ) 기법을 사용하는 새로운 지속적 학습(Continual Learning) 방법으로, 모델이 여러 작업을 수행하면서 이전 정보를 잃지 않도록 돕습니다. 이 방법은 내장된 프롬프트(promopts)집합에서 효과적으로 작업 지식을 표현하고, 이를 통해 프리트레인(pre-trained) 모델의 성능을 향상시키는 점이 특징입니다.

- **Technical Details**: VQ-Prompt는 Discrete Prompt를 사용하여 지속적 학습을 최적화하는 메커니즘을 도입합니다. 입력 쿼리에서 생성된 continuous prompt를 미리 정의된 discrete prompt pool에서 가장 가까운 값을 찾는 방식으로 교체하며, gradient estimation을 통해 task loss를 continuous prompt로 전파합니다. 추가적인 VQ 정규화 항을 통해 prompt pool의 학습을 더욱 개선합니다.

- **Performance Highlights**: VQ-Prompt는 다양한 벤치마크에서 최신 지속적 학습 방법들보다 높은 성능을 발휘하며, 특히 challenging한 class-incremental 환경에서 탁월한 결과를 보였습니다. 검증된 실험 결과를 통해 VQ-Prompt는 지속적 학습의 성능 기준을 한 단계 끌어올렸음을 보여주었습니다.



### Deep Learning-Driven Microstructure Characterization and Vickers Hardness Prediction of Mg-Gd Alloys (https://arxiv.org/abs/2410.20402)
- **What's New**: 본 연구에서는 Mg-Gd 합금의 기계적 성능을 예측하기 위해 이미지 처리 및 딥러닝 기법을 기반으로 한 멀티모달 융합 학습 프레임워크를 제안합니다. 이 프레임워크는 합금의 원소 조성 및 미세구조 특성을 통합하여 Vickers 경도를 정확히 예측합니다.

- **Technical Details**: 연구는 다양한 Mg-Gd 합금 이미지에서 미세구조 정보를 추출하기 위해 딥러닝 기법을 사용하여 정밀한 결정립 크기 및 이차상 미세구조 특성을 제공합니다. 이후 Gd 함량 정보와 정량적 분석 결과를 조합하여 성능 예측 데이터셋을 구축하였습니다. 최종적으로 Transformer 아키텍처를 기반으로 한 회귀 모델을 사용하여 Mg-Gd 합금의 Vickers 경도를 예측하였습니다.

- **Performance Highlights**: 실험 결과, Transformer 모델이 예측 정확도 측면에서 최적의 성능을 보이며 R² 값 0.9을 기록하였습니다. 또한, SHAP 분석을 통해 Mg-Gd 합금의 Vickers 경도에 영향을 미치는 네 가지 주요 특징의 중요 값이 식별되어 합금 설계에 유용한 지침을 제공합니다.



### Conditional GAN for Enhancing Diffusion Models in Efficient and Authentic Global Gesture Generation from Audios (https://arxiv.org/abs/2410.20359)
Comments:
          Accepted by WACV 2025 (Round 1)

- **What's New**: 본 연구에서는 음성을 기반으로 한 제스처 생성의 효율성을 획기적으로 개선하는 방법을 제시합니다. 기존 DDPM(디퓨전 기반 생성 모델) 가정의 한계를 넘어서, 조건부 GAN(Generative Adversarial Network)을 도입하여 음성을 컨트롤 신호로 활용합니다.

- **Technical Details**: 조건부 GAN을 통해 다단계 샘플링 과정에서 복잡한 노이즈 분포를 모델링하며, 대규모 노이즈를 샘플링하고, 디노이징 단계를 감소시켜 고속 생성을 가능하게 합니다. 제안하는 시스템은 음성 신호를 제어 조건으로 사용하여 다양한 제스처를 효과적으로 생성 구성합니다.

- **Performance Highlights**: 제안하는 방법은 기존 DiffuseStyleGesture보다 약 12.35배 빠른 시간 내에 동작 생성이 가능하며, 현대의 디퓨전 기반 방법들에 비해 생성 효율성과 품질에서 현저히 우수한 성능을 입증했습니다.



### Enhancing Community Vision Screening -- AI Driven Retinal Photography for Early Disease Detection and Patient Trus (https://arxiv.org/abs/2410.20309)
Comments:
          11 pages, 4 figures, published in MICCAI2024 OMIA XI workshop

- **What's New**: 이 논문은 rural 지역 사회를 위한 새로운 커뮤니티 비전 스크리닝 솔루션인 ECVS (Enhancing Community Vision Screening)를 소개합니다. 이는 비침습적(retinal photography) 기술을 기반으로 하여 시각 장애를 식별하고, 효과적으로 환자를 치료를 위해 의뢰하는 과정을 단순화합니다.

- **Technical Details**: ECVS는 4가지 딥러닝 모델을 활용합니다: RETinal photo Quality Assessment (RETQA), Pathology Visual Impairment detection (PVI), Eye Disease Diagnosis (EDD), 그리고 Visualization of Lesion Regions of the eye (VLR). 이 모델들은 80,000개 이상의 fundus 사진을 사용하여 학습되었으며, 각각 AUC 스코어 0.98(RETQA), 0.95(PVI), 0.90(EDD) 및 DICE 계수 0.48(VLR)을 달성했습니다.

- **Performance Highlights**: ECVS는 기존 검진 프로세스와 비교해 검사 횟수를 5회에서 2회로 줄이고, 총 대기 시간을 40분에서 5분으로 단축시킵니다. 또한 환자 의뢰 시간을 2-4주에서 10-20분으로 대폭 감소시킴으로써 효율성을 크게 개선했습니다.



### EfficientEQA: An Efficient Approach for Open Vocabulary Embodied Question Answering (https://arxiv.org/abs/2410.20263)
- **What's New**: 새로운 연구에서는 Embodied Question Answering (EQA)을 위한 효율적인 프레임워크인 EfficientEQA를 제안했습니다. 이 프레임워크는 로봇이 개방형 어휘 환경에서 빠르고 정확하게 질문에 답할 수 있도록 합니다.

- **Technical Details**: EfficientEQA는 Semantic-Value-Weighted Frontier Exploration 전략을 사용하여 로봇이 중요한 정보를 기반으로 환경을 탐색하도록 합니다. 또한, Retrieval-Augmented Generation (RAG) 기술을 통해 이미지 검색과 VLM(비전-언어 모델) 추론을 활용하여 미리 정의된 답변 선택지에 의존하지 않고 답변을 생성합니다.

- **Performance Highlights**: 실험 결과, EfficientEQA는 기존 상태-최적화 방법 대비 15% 이상의 정확도 향상과 20% 이상의 효율성을 보여주었습니다.



### Neural Fields in Robotics: A Survey (https://arxiv.org/abs/2410.20220)
Comments:
          20 pages, 20 figures. Project Page: this https URL

- **What's New**: 본 논문은 Neural Fields(NF)가 로봇 공학에서 3D 장면 표현의 획기적인 방법으로 자리 잡고 있음을 강조합니다. 특히, NF의 연속적이고 미분 가능한 표현 방식이 다양한 센서 데이터의 통합 및 새로운 시점 생성에 어떻게 기여하는지를 설명합니다.

- **Technical Details**: Neural Fields는 Occupancy Networks, Signed Distance Fields, Neural Radiance Fields, Gaussian Splatting과 같은 네 가지 주요 프레임워크를 기반으로 합니다. NF는 RGB 카메라, LiDAR, 깊이 센서 등에서 수집된 데이터를 기반으로 최적화되어 고품질 3D 재구성을 생성합니다.

- **Performance Highlights**: Neural Fields는 고품질 3D 재구성, 다중 센서 데이터 통합, 연속적인 컴팩트 표현, 일반화 및 적응력을 제공하여 로봇의 탐색, 물체 조작, 자율 주행 등에서 성능을 크게 향상시킵니다. 또한, NF의 발전은 생성형 AI와 로봇 간의 중요한 연결 고리를 제공하여 이 분야의 연구가 급속히 성장하고 있음을 보여줍니다.



### Transferable Adversarial Attacks on SAM and Its Downstream Models (https://arxiv.org/abs/2410.20197)
Comments:
          This work is accepted by Neurips2024

- **What's New**: 본 논문은 공개된 segment anything model (SAM)을 활용하여 다양한 downstream 모델에 대한 adversarial 공격의 가능성을 처음으로 탐구합니다. 이는 기존의 전이 기반 공격 방식과 비교해, 특정 downstream 작업 및 데이터셋을 접근하지 않고도 발생하는 adversarial 위험을 실증적으로 보여줍니다.

- **Technical Details**: 새로운 공격 방법론인 universal meta-initialization (UMI) 알고리즘을 제안하여, foundation model에 내재된 취약성을 추출하고 이를 통해 adversarial perturbations 생성을 유도합니다. 공격 과정에서의 gradient 차이를 수학적으로 이론화하며, gradient robust loss를 제안하여 전이 가능성을 향상시킵니다.

- **Performance Highlights**: 제안된 보편적 메타 초기화 및 gradient 강건한 adversarial 공격 (UMI-GRAT)은 실제 적용에서 SAM 및 그 downstream 모델에 대한 효과를 입증했습니다. extensive experiments를 통해 UMI-GRAT의 강력한 성능을 확인하였으며, 기존 전이 기반 adversarial 공격을 개선할 수 있는 중요한 방법론으로 자리잡을 것으로 기대됩니다.



### LLMs Can Evolve Continually on Modality for X-Modal Reasoning (https://arxiv.org/abs/2410.20178)
- **What's New**: PathWeave는 Modal-Path 전환 및 확장 기능을 갖춘 유연하고 확장 가능한 프레임워크로, Multimodal Large Language Models (MLLMs)의 연속적인 모달 진화를 가능하게 합니다. 이를 통해 단일 모달 데이터로 새로운 모달리티에 대한 확장과 학습이 가능합니다.

- **Technical Details**: PathWeave는 Incremental Training Strategy를 바탕으로 하여, Uni-modal 및 Cross-modal Adapter를 통합하여 효율적인 모달 정렬 및 협업을 촉진합니다. MoE (Mixture of Experts) 기반의 게이팅 모듈을 통해 두 가지 유형의 어댑터 간의 멀티모달 상호작용을 향상시킵니다.

- **Performance Highlights**: PathWeave는 최신의 MLLMs와 비교하여 성능 면에서 동등하며, 파라미터 학습 부담을 98.73% 감소시키는 동시에, Continual Learning of Modality (MCL) 벤치마크에서의 높은 정확도를 보여줍니다.



### Prompt Diffusion Robustifies Any-Modality Prompt Learning (https://arxiv.org/abs/2410.20164)
Comments:
          Under review

- **What's New**: 이 논문에서는 기존의 고정된 프롬프트(method of employing fixed prompts) 접근 방식이 새로운 데이터 샘플에 대해 일반화할 때 발생하는 문제를 해결하기 위해, '프롬프트 확산(prompt diffusion)'이라는 새로운 방법을 도입합니다. 이 방법은 각 샘플에 맞춤화된 프롬프트를 생성하기 위해 확산 모델을 활용하여 점진적으로 프롬프트를 수정합니다.

- **Technical Details**: 프롬프트 확산은 프롬프트 공간(prompt space) 내에서 새로운 생성 경로를 학습하면서 임의의 프롬프트에서 더 개인화된 프롬프트로 전이하는 프로세스를 포함합니다. 이 방법은 다섯 단계로 테스트 샘플 프롬프트를 최적화하는 빠른 ODE 기반 샘플링 전략을 사용하여 성능 개선과 계산 효율성 간의 균형을 이룹니다.

- **Performance Highlights**: 15개의 다양한 데이터셋에 걸친 분류 작업에서, 프롬프트 확산을 적용함으로써 기본-새로운 일반화(base-to-new generalization), 교차 데이터셋 일반화(cross-dataset generalization), 도메인 일반화(domain generalization)의 모든 프롬프트 학습 방법에서 더 강력한 결과를 도출하였습니다.



### Super-resolved virtual staining of label-free tissue using diffusion models (https://arxiv.org/abs/2410.20073)
Comments:
          26 Pages, 5 Figures

- **What's New**: 본 연구에서는 Brownian bridge 프로세스를 활용한 확산 모델(diffusion model) 기반의 슈퍼 해상도(super-resolution) 가상 염색(virtual staining) 접근 방식을 제시합니다.

- **Technical Details**: 이 방법은 전통적인 딥러닝(d deep learning) 기반 방법의 한계를 극복하며, 새로운 샘플링 기법(sampling techniques)을 확산 모델에 통합하여 생성된 가상 염색 이미지의 분산(variance)을 크게 줄입니다.

- **Performance Highlights**: 가상 염색 모델은 저해상도(auto-fluorescence) 인간 폐 조직 샘플에 무작위로 적용되었을 때 해상도, 구조적 유사성(structural similarity), 지각적 정확성(perceptual accuracy)에서 지속적으로 기존 방법을 초월하며, 4-5배의 슈퍼 해상도 요인을 달성하여 출력 위상 대역폭(product of spatial bandwidth)을 기존 이미지 대비 16-25배 향상시켰습니다.



### Transforming Precision: A Comparative Analysis of Vision Transformers, CNNs, and Traditional ML for Knee Osteoarthritis Severity Diagnosis (https://arxiv.org/abs/2410.20062)
- **What's New**: 이 연구는 기존의 기계 학습 기법과 새로운 딥러닝 모델을 비교 분석하여 슬개골 퇴행성 관절염(KO)의 중증도를 X-ray 이미지를 통해 진단하는 데 초점을 맞추고 있습니다. 최신 ViT( Vision Transformer) 모델의 의료 이미지 컨텍스트에서의 적용 가능성과 비교 효과를 강조합니다.

- **Technical Details**: 연구는 Osteoarthritis Initiative (OAI)에서 제공한 1526장의 X-ray 이미지를 사용하여 KO의 중증도를 5개 등급으로 분류합니다. 기존의 기계 학습 모델(GaussianNB, KNN)보다 Convolutional Neural Networks(CNN)인 Inception-V3와 VGG-19가 평균 55-65%의 정확도를 보였지만, ViT 모델(Da-VIT, GCViT, MaxViT)은 66.14%의 정확도, 0.703의 정밀도, 0.614의 재현율, 0.835 이상의 AUC를 달성했습니다.

- **Performance Highlights**: ViT 모델이 다른 모델들보다 뛰어난 성능을 발휘했으며, 특히 정확도(F1 score, precision, AUC score) 측면에서 월등한 결과를 나타냈습니다. 이는 의료 진단 작업 흐름에서 ViT 모델 통합의 필요성을 강조하며, KO 평가의 정확성과 신뢰성을 혁신적으로 변화시킬 가능성을 제시합니다.



### GHIL-Glue: Hierarchical Control with Filtered Subgoal Images (https://arxiv.org/abs/2410.20018)
Comments:
          Code, model checkpoints and videos can be found at this https URL

- **What's New**: GHIL-Glue는 인터넷 규모의 데이터에 기반하여 생성된 이미지 및 비디오 예측 모델과 하위 목표 조건 정책을 효과적으로 결합하는 새로운 인터페이스를 제공합니다.

- **Technical Details**: 이 방법은 하위 목표를 필터링하고, 생성된 하위 목표에 포함된 시각적 아티팩트에 대해 목표 조건 정책의 내구성을 향상시키는 두 가지 주요 구성 요소를 포함합니다.

- **Performance Highlights**: GHIL-Glue는 여러 계층적 모델에서 25%의 성능 개선을 달성했으며, 단일 RGB 카메라에서 관찰 정보를 사용하는 CALVIN 시뮬레이션 벤치마크에서 새로운 최첨단 성과를 이뤘습니다.



### Multi-Class Abnormality Classification Task in Video Capsule Endoscopy (https://arxiv.org/abs/2410.19973)
Comments:
          Video Capsule Endoscopy Challenge

- **What's New**: 이 연구는 비디오 캡슐 내시경(Video Capsule Endoscopy, VCE)에서 다중 클래스 이상 탐지를 위한 다양한 딥러닝 모델을 탐구하고, 특히 Dual Attention Vision Transformer(DaViT) 모델의 성능 향상을 주목하였습니다.

- **Technical Details**: 연구에서는 Cascade CNN, ResNet, ViT, MViT 및 DaViT와 같은 다양한 딥러닝 아키텍처를 사용하여 비디오 데이터의 공간 및 채널 의존성을 모델링했습니다. DaViT는 멀티헤드 접근 방식을 통해 채널 별로 토큰을 분류하며, 효율적인 주의(attention) 메커니즘을 적용하여 계산 복잡성을 감소시킵니다.

- **Performance Highlights**: DaViT 모델은 정확도를 94.87%, 정밀도를 94.82%, 재현율을 94.76%, F1-score를 94.77%로 기록하며 기존 방법보다 뛰어난 성능을 보여주었습니다. 특히 Erythema 클래스의 AUC는 0.98로, 다른 클래스와의 차별 능력을 더욱 개선할 여지가 있는 것으로 나타났습니다.



### Improving Multimodal Large Language Models Using Continual Learning (https://arxiv.org/abs/2410.19925)
Comments:
          NeurIPS 2024 Workshop on Scalable Continual Learning for Lifelong Foundation Models

- **What's New**: 이 연구는 LLaVA MLLM을 통해 시각 정보를 LLM에 통합할 때 발생하는 언어 이해 및 생성 성능 저하 문제를 지속적 학습(Continual Learning) 문제로 다룹니다. 이는 기존 LLM에 비해 MLLM이 자연어 처리에서 성능이 감소하는 현상을 줄여주는 초기 기법을 모색하고, 이를 통해 MLLM의 시각적 이해를 향상시키는 방법을 제안합니다.

- **Technical Details**: LLaVA MLLM의 통합 과정에서 발생하는 언어적 망각 문제를 해결하기 위해, 5가지 지속적 학습 방법을 검토합니다. 실험을 통해 시각-언어(VL) 작업에서의 성능 저하를 최대 15%까지 줄이는 기법을 도출하였습니다. 언어적 성능 저하를 줄이면서 높은 다중 모달 정확도를 유지하는 방법으로서 Soft Targets, LoRA, mSGM 등의 기술이 활용됩니다.

- **Performance Highlights**: LLaVA의 기존 수치와 비교하여, 제안된 방법이 NLG, NLU, VL 작업의 성능을 개선한 결과를 보여주며, 특히 Soft Targets가 VL 데이터셋에서 가장 높은 정확도를 올리는 것으로 나타났습니다. MLLM의 지속적 학습 실험을 통해 언어적 기술은 유지하면서도 새로운 다중 모달 능력을 성공적으로 습득하였습니다.



### GNNRL-Smoothing: A Prior-Free Reinforcement Learning Model for Mesh Smoothing (https://arxiv.org/abs/2410.19834)
- **What's New**: 본 논문에서는 기존의 지도 학습(supervised learning) 및 강화 학습(reinforcement learning)에 의존하는 동적 메시 스무딩 기법을 넘어, 데이터나 사전 지식 없이 학습할 수 있는 새로운 강화 학습 모델을 제안합니다. 이는 메시 최적화를 마르코프 결정 과정(Markov Decision Process)으로 형식화하여, 총 두 개의 에이전트(agents)를 훈련시킵니다.

- **Technical Details**: 제안된 모델은 그래프 신경망(Graph Neural Networks)과 강화 학습을 결합하여 똑똑한 노드 스무딩 에이전트(intelligent node smoothing agent)와 메시 연결성 개선 에이전트(mesh connectivity improvement agent)를 구현합니다. 이 두 에이전트는 Twin Delayed Deep Deterministic Policy Gradient와 Double Dueling Deep Q-Network를 사용하여 훈련됩니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 복잡한 3D 표면 메시에서 피처를 보존하는 스무딩을 달성했으며, 2D 메시에 대해서는 동적 스무딩 방법 중에서 최신 기술 대비 7.16배 빠른 결과를 보여주었습니다. 또한 연결성 개선 에이전트는 메시 품질 분포를 효과적으로 향상시킬 수 있음을 입증했습니다.



### Advancing Histopathology with Deep Learning Under Data Scarcity: A Decade in Review (https://arxiv.org/abs/2410.19820)
Comments:
          36 pages

- **What's New**: 최근 몇 년 동안 computational histopathology(Computational Histopathology)에서 remarkable progress가 있었으며, 이는 주로 deep learning(딥러닝)에 의해 이루어졌습니다. 딥러닝 기반 도구의 임상적 채택이 가능해짐에 따라, 진단에 대한 valuable second opinion(중요한 제2의 의견)과 복잡한 작업의 간소화, 임상 결정에서의 일관성과 편향의 위험 완화 등의 많은 혜택을 제공할 수 있습니다. 그러나 깊은 문제는 큰 labeled datasets(레이블이 붙은 데이터셋)의 필요성입니다.

- **Technical Details**: 이 논문은 지난 10년 간 데이터 부족 문제를 중심으로 심층 학습 응용 프로그램의 포괄적인 리뷰를 제공합니다. 연구자들은 데이터가 부족한 상황에서도 딥러닝을 활용할 수 있는 다양한 전략을 개발했습니다. 논문은 여러 접근 방식을 체계적으로 분류하고 비교하며, benchmarking tables(벤치마킹 테이블)를 통해 그들의 독특한 기여도를 평가합니다.

- **Performance Highlights**: 딥러닝 도구는 디지털 병리학의 전통적인 수작업 분석을 변화시키고 있으며, 각종 진단 및 연구 작업에서 높은 성능을 보이고 있습니다. 특히, 종양 세포와 주변 지지 세포 간의 상호작용을 연구하고, tumor-stroma ratio(종양-간질 비율)을 분석하는 과정에서 큰 기여를 합니다. 또한, Tumor Mutation Burden(종양 변이 부담) 및 Microsatellite Instability(마이크로위성 불안정성)와 같은 중요한 바이오마커의 평가에서도 딥러닝은 유망한 도구로 자리 잡고 있습니다.



### Training Compute-Optimal Vision Transformers for Brain Encoding (https://arxiv.org/abs/2410.19810)
- **What's New**: 이 연구는 신경 인코딩(Brain Encoding)에서 비전 트랜스포머(Vision Transformer)의 최적 훈련을 위해 모델 크기, 데이터 크기 및 컴퓨팅 리소스 간의 관계를 탐구합니다.

- **Technical Details**: 비디오에서 효과적인 시공간 특성을 추출하기 위해 VideoGPT를 사용하고, 이 특성을 기반으로 뇌 활동을 예측하는 Ridge 모델을 훈련했습니다. 다양한 데이터 크기(10k, 100k, 1M, 6M) 및 GPT-2의 여러 모델 구성(히든 레이어 차원, 레이어 수, 어텐션 헤드 수)에 대한 성능을 평가했습니다. 또한 32비트와 16비트 부동 소수점 표현의 영향을 비교했습니다.

- **Performance Highlights**: 히든 레이어 차원을 증가시키는 것이 뇌 인코딩 성능을 유의미하게 개선하며, 데이터 크기를 늘리면 인코딩 성능이 향상되는 것으로 나타났습니다. 16비트로 훈련했을 때 32비트와 동일한 정확도를 유지하면서 훈련 시간을 1.17배 단축했습니다.



### ScreenWriter: Automatic Screenplay Generation and Movie Summarisation (https://arxiv.org/abs/2410.19809)
- **What's New**: 이번 연구에서는 비디오 콘텐츠를 자동으로 스크린플레이(screenplay)로 생성하는 새로운 접근 방식을 제안합니다. 특히, ScreenWriter라는 알고리즘을 통해 비디오만을 이용하여 대화(dialogue), 화자 이름(speaker names), 장면 구분(scene breaks), 시각적 설명(visual descriptions)을 포함한 스크립트를 작성합니다.

- **Technical Details**: ScreenWriter는 비디오를 시각적 벡터(visual vectors)의 순서에 따라 장면으로 나누는 혁신적인 알고리즘을 소개합니다. 캐릭터 이름을 식별하는 문제를 해결하기 위해 배우의 얼굴 데이터베이스를 활용하고, 이를 통해 자동 생성된 스크린플레이를 바탕으로 계층적 요약(hierarchical summarisation) 방법을 통해 줄거리 개요(plot synopses)를 생성할 수 있음을 입증하였습니다.

- **Performance Highlights**: 최종 요약의 품질을 MovieSum 데이터셋에서 테스트한 결과, 기존의 스크린플레이에 접근할 수 있다고 가정하는 여러 모델보다 우수한 성능을 보였습니다. 오류의 주요 원인으로는 잘못된 얼굴 정보, 다중 화자 식별 문제 등이 있으며, 향후 이러한 문제를 개선할 수 있는 방법이 기대됩니다.



### Radon Implicit Field Transform (RIFT): Learning Scenes from Radar Signals (https://arxiv.org/abs/2410.19801)
Comments:
          A version of this work is under review as a submission to ICLR 2025 Conference

- **What's New**: 이번 연구에서는 레이다 신호에서의 장면 표현을 학습할 수 있는 첫 번째 방법인 Radon Implicit Field Transform (RIFT)를 제안합니다. RIFT는 레이다 신호로부터 학습한 Implicit Neural Representations (INRs)을 활용하여 기존의 ASP 문제를 해결합니다.

- **Technical Details**: RIFT는 Generalized Radon Transform (GRT)라는 레이다 전방 모델과 INR 기반의 장면 표현을 결합하며, 이를 통해 레이다 신호의 재구성 오류를 최소화하는 방식으로 모델을 훈련합니다. 이 과정에서 p-RMSE와 m-SSIM이라는 새로운 오류 메트릭을 도입하였습니다.

- **Performance Highlights**: RIFT 모델은 전통적인 장면 모델에 비해 최대 188%의 장면 재구성 개선을 이루며, 기존 데이터의 10%만으로도 재구성 성능을 3배 향상시키고, 새로운 관점에서의 일반화에서도 10% 개선된 결과를 보였습니다.



### Data-Driven Cellular Network Selector for Vehicle Teleoperations (https://arxiv.org/abs/2410.19791)
Comments:
          IEEE Network of Future 2024

- **What's New**: 이 논문에서는 새로운 알고리즘인 Active Network Selector (ANS)를 제안하며, 이를 통해 자율차(AV)가 여러 셀룰러 네트워크에 연결된 상태에서 비디오 패킷을 전송하는 최적의 경로를 실시간으로 선택할 수 있는 방법을 설명합니다.

- **Technical Details**: 논문에서 다루는 주요 기술적 요소는 패킷 손실(packet loss), 지연(latency), 그리고 셀룰러 네트워크의 성질입니다. ANS 알고리즘은 시계열(time series) 기계 학습(machine learning) 접근 방식을 사용해 선택된 네트워크를 기반으로 패킷 전송을 최적화합니다. 세 가지 기계 학습 알고리즘인 LossPredict, HandPredict, LatencyPredict가 제안되어 각각 패킷 손실, 핸드오버(handover), 지연을 예측합니다.

- **Performance Highlights**: ANS는 기존의 비학습(baseline non-learning) 알고리즘에 비해 80% 이상의 패킷 손실 예측 정확도를 기록하며, 핸드오버 빈도 및 각 네트워크의 사용자 신호 품질과 강한 상관관계를 보여줍니다. 다양한 테스트 드라이브 결과 ANS가 패킷 손실 및 지연을 크게 감소시킬 수 있음을 입증하였습니다.



### Multi-modal Image and Radio Frequency Fusion for Optimizing Vehicle Positioning (https://arxiv.org/abs/2410.19788)
- **What's New**: 이 논문에서는 채널 상태 정보(Channel State Information, CSI)와 이미지를 결합하여 차량을 공동으로 위치 추정하는 다중 모달(vehicle positioning framework) 접근법을 제시합니다.

- **Technical Details**: 특히 이 연구에서는 차량이 하나의 기지국(BS)와만 통신할 수 있는 야외 시나리오를 고려합니다. 각 BS는 CSI의 일부 레이블은 붙어 있지만, 많은 수의 비레이블 CSI와 카메라로 촬영한 이미지의 수집을 가능하게 하는 카메라 세트로 장착되어 있습니다. 레이블이 없는 CSI 데이터를 활용하기 위해 메타-학습(meta-learning) 기반의 하드 기대 최대화(expectation-maximization, EM) 알고리즘이 설계되었습니다. 레이블이 없는 CSI와 여러 차량 위치 간의 관계가 불분명하므로, 학습 목적 함수를 최소 매칭 문제로 형식화했습니다.

- **Performance Highlights**: 시뮬레이션 결과, 제안된 방법은 이미지를 사용하지 않고 CSI 핑거프린트만으로 차량 위치 추정을 하는 기준선과 비교하여 위치 오차를 최대 61%까지 줄일 수 있음을 보여주었습니다.



### How to Backdoor Consistency Models? (https://arxiv.org/abs/2410.19785)
- **What's New**: 일관성 모델(Consistency Models)이라는 새로운 생성 모델 클래스의 백도어 공격(vulnerability to backdoor attacks) 연구가 진행되었습니다. 이 모델들은 노이즈를 데이터로 직접 매핑하여 이미지를 생성하는 방식으로, 단일 단계에서 이미지를 생성할 수 있습니다.

- **Technical Details**: 이 연구에서는 일관성 모델의 백도어 공격에 대한 취약성을 분석하였습니다. 기존의 확산 모델에 대한 연구와는 달리, 백도어 훈련 프로세스가 고유한 일관성 모델의 훈련 방식과 거기서 파생된 목표를 고려합니다. 프레셰 발달 거리(Fréchet Inception Distance, FID)를 이용하여 공격 모델과 비공격 모델 간의 이미지 품질을 비교했습니다.

- **Performance Highlights**: 실험 결과, 제안한 프레임워크는 다양한 트리거(trigger) 및 타겟(target) 설정에서도 일관성 모델이 공격을 받을 수 있음을 보여주었습니다. Gaussian 노이즈를 트리거로 사용하면 눈에 띄지 않으면서도 효과적인 결과를 얻을 수 있었습니다.



### Less Cybersickness, Please: Demystifying and Detecting Stereoscopic Visual Inconsistencies in Virtual Reality Apps (https://arxiv.org/abs/2406.09313)
Comments:
          This work has been accepted at the ACM International Conference on the Foundations of Software Engineering (FSE) 2024, Porto de Galinhas, Brazil. DOI: this https URL

- **What's New**: 이 논문에서는 VR 앱의 스테레오 시각 불일치(SVI) 문제를 시스템적으로 분석하고 이를 탐지하기 위한 새로운 자동화 테스트 프레임워크인 StereoID를 제안합니다. 이 방법은 통상적인 GUI 오류 탐지 방식보다 효과적이며, 288개의 실제 VR 앱에서 수집된 데이터 세트를 기반으로 합니다.

- **Technical Details**: StereoID는 VR 앱의 렌더링된 GUI 상태만을 기반으로 SVI 문제를 식별하는 비지도 학습(unsupervised learning) 방식의 블랙박스 테스트 프레임워크입니다. 독창적인 심도 인식 조건부 스테레오 이미지 변환기(conditional stereo image translator)를 도입하여 실제 왼쪽 눈 이미지를 기반으로 인공적으로 오른쪽 눈 이미지를 생성하고, 두 이미지 간의 거리 계산을 통해 SVI 문제를 탐지합니다.

- **Performance Highlights**: StereoID는 171K 이상의 이미지로 구성된 대규모 라벨이 없는 VR 스테레오 스크린샷 데이터셋을 구축하였으며, SVI 문제를 효과적으로 탐지하는 성능을 보여주었습니다. 다양한 사용자 보고서와 실제 VR 앱에서 SVI 문제를 식별하는 데에 있어 기존 방법들보다 우수함을 입증하였습니다.



New uploads on arXiv(cs.AI)

### Multi-modal AI for comprehensive breast cancer prognostication (https://arxiv.org/abs/2410.21256)
- **What's New**: 이 연구는 디지털 병리학(digital pathology) 및 임상 특성을 이용하여 유방암 환자를 분류하기 위한 새로운 AI 테스트 개발에 관한 것입니다. 이 테스트는 자기 지도 학습(self-supervised learning)에 기반한 비전 트랜스포머(vision transformer) 모델을 활용하여 개발되었습니다.

- **Technical Details**: AI 테스트는 H&E 염색 슬라이드에서 특징을 추출하고, 이를 기존의 임상 데이터와 통합하여 다중 모달(multi-modal) AI 테스트를 생성합니다. 총 8,161명의 유방암 환자 데이터를 기반으로 개발되었으며, 다양한 유방암 아형에서 뚜렷한 예측 성과를 보였습니다.

- **Performance Highlights**: AI 테스트는 5개의 외부 집단에서 질병 없는 간격(disease-free interval, DFI)에 대해 C-인덱스(C-index) 0.71 [0.68-0.75]를 기록하며, 기존의 Oncotype DX보다 높은 정확도를 보여주었습니다. 특히, 삼중 음성 유방암(TNBC)에서도 C-인덱스 0.71 [0.62-0.81]을 달성했습니다.



### Hierarchical Knowledge Graph Construction from Images for Scalable E-Commerc (https://arxiv.org/abs/2410.21237)
- **What's New**: 이번 논문에서는 원시 제품 이미지에서 구조화된 제품 지식 그래프(knowledge graph)를 자동으로 구축하는 혁신적인 방법을 제안합니다. 이 방법은 최근 Vision-Language Model (VLM)과 Large Language Model (LLM)의 발전을 활용하여 전체 과정을 자동화하고 신속한 그래프 업데이트를 가능하게 합니다.

- **Technical Details**: 저자들은 InternVL2-8B라는 지시 조정된 VLM을 사용하여 제품 이미지에서 세부 정보를 추출합니다. 이후, 스키마가 보강된 다중 턴 대화를 통해 더 다양하고 상세한 속성과 관계를 포함하도록 보장합니다. 마지막으로, Llama3.1-70B와 SGLang를 활용하여 LLM 응답을 신뢰할 수 있는 구조적 형식으로 생성하고 중복을 줄이는 프로그램적 병합 방법을 설계했습니다.

- **Performance Highlights**: 제안된 방법은 이전 연구에서 수정된 기준선(baseline)과 비교하여 모든 메트릭과 평가된 속성에서 우수한 성능을 보이며, 자동화된 비즈니스 환경에서 지식 그래프 구축의 효과성과 잠재력을 증명했습니다.



### Towards Unifying Evaluation of Counterfactual Explanations: Leveraging Large Language Models for Human-Centric Assessments (https://arxiv.org/abs/2410.21131)
Comments:
          This paper has been submitted in August and is currently under review to AAAI-2025

- **What's New**: 이 논문에서는 카운터팩추얼( counterfactual ) 설명의 평가를 자동화할 수 있는 가능성을 탐구하여, 이를 위해 다양한 차원의 설명 품질을 가진 30개의 카운터팩추얼 시나리오를 개발하고 206명의 응답자로부터 평가를 수집했습니다.

- **Technical Details**: 제안된 방법론은 8개의 평가 지표를 바탕으로 LLM (Large Language Model) 모델을 미세 조정하여 인간의 평균 또는 개별 판단을 예측하도록 함으로써, 카운터팩추얼 설명 프레임워크의 평가에서 더 나은 비교 가능성과 확장성을 제공합니다.

- **Performance Highlights**: 미세 조정된 LLM 모델은 제로샷 평가에서 최대 63%의 정확도를 달성하였고, 미세 조정 후 모든 지표에서 3개 클래스 예측에 대해 85%의 정확도를 기록하여 인간 평가와의 비교에서 좋은 성능을 보였습니다.



### Learning to Handle Complex Constraints for Vehicle Routing Problems (https://arxiv.org/abs/2410.21066)
Comments:
          Accepted at NeurIPS 2024

- **What's New**: 본 연구는 Vehicle Routing Problems (VRPs)의 복잡한 제약 조건을 다룰 수 있는 새로운 Proactive Infeasibility Prevention (PIP) 프레임워크를 제안합니다. PIP는 Lagrangian multiplier를 통합하여 제약 인식을 향상시키고, 예방적 비타당성 마스킹을 도입하여 해결 과정에서 제약 조건을 사전에 대응할 수 있도록 합니다.

- **Technical Details**: PIP 프레임워크는 reinforcement learning 프레임워크에 Lagrangian multiplier 방법을 통합하여 초기의 제약 인식을 높이고, 예방적 비타당성 마스킹을 통해 해결책 탐색을 (near-)타당한 영역으로 유도합니다. PIP-D는 보조 디코더와 두 가지 적응 전략을 사용하여 맞춤형 마스크를 학습하고 예측하며, 훈련 동안의 계산 비용을 크게 줄입니다.

- **Performance Highlights**: PIP는 다양한 제약 강도 수준에서 Traveling Salesman Problem with Time Window (TSPTW) 및 Draft Limit (TSPDL) 변형을 포함한 광범위한 실험을 통해 효과성을 입증합니다. 특히, PIP는 비타당성 비율을 최대 93.52%까지 감소시키고, 솔루션 품질이 크게 향상되는 결과를 보였습니다.



### Neuro-symbolic Learning Yielding Logical Constraints (https://arxiv.org/abs/2410.20957)
Comments:
          Published as a conference paper at NeurIPS 2023, and code is available at [this url](this https URL)

- **What's New**: 이 논문은 신경망 훈련(Neural Network Training), 기호 구속(Symbol Grounding), 논리적 제약 조건 합성(Logical Constraint Synthesis)을 통합한 자연스러운 프레임워크를 제안하여 신경-기호 시스템(Neuro-Symbolic Systems)의 종단 간 학습(end-to-end learning) 문제를 해결하는 접근 방식을 모색하고 있습니다.

- **Technical Details**: 제안된 프레임워크는 연속적인 신경망과 이산적 논리 제약 간의 격차를 해소하기 위해 차이-볼록 프로그래밍(Difference-of-Convex Programming) 기법을 도입하여 논리적 제약을 완화하고 있습니다. 또한, 카디널리티 제약(Cardinality Constraints)을 논리적 제약 학습 언어로 사용하고, 학습 과정에서 논리적 제약의 퇴화를 피하기 위해 신뢰 영역 방법(Trust Region Method)을 통합합니다.

- **Performance Highlights**: Visual Sudoku Solving, Self-Driving Path Planning, Chained XOR, Nonograms의 네 가지 작업(task)에서 실시된 실험평가(empirical evaluations)는 제안한 프레임워크가 새로운 학습 능력을 보여주며, 우수한 성능(superior performance)을 발휘함을 입증합니다.



### Active Legibility in Multiagent Reinforcement Learning (https://arxiv.org/abs/2410.20954)
- **What's New**: 이번 연구의 주요 내용은 Multiagent Active Legibility (MAAL)이라는 새로운 프레임워크를 제안하여 multiagent reinforcement learning (MARL)에서의 에이전트 간 협력 행동을 향상시키는 것입니다. 이 프레임워크는 에이전트가 다른 에이전트의 행동을 모델링하고 예측함으로써 상호작용의 효율성을 높입니다.

- **Technical Details**: MAAL 프레임워크는 Legible Interactive-POMDP (LI-POMDP)를 정의하고 적용합니다. 이 프레임워크는 Kullback-Leibler (KL) divergence를 활용하여 에이전트의 행동과 관찰자의 믿음 간의 차이를 줄여, 협력이 원활하게 이루어지도록 합니다. 이는 전통적인 가치 분해(value decomposition) 또는 중앙 집중형 비평가(centralized-critic) 접근 방식과는 다른 접근법입니다.

- **Performance Highlights**: 실험 결과, MAAL 프레임워크는 기존의 여러 MARL 알고리즘에 비해 더 적은 학습 시간과 더 높은 효율성을 보여주었습니다. 이는 효율적이고 효과적인 협력을 촉진하며, 특히 군사, 의료, 금융 등 안전-critical 영역에서도 중요한 의미를 가집니다.



### FACTS: A Factored State-Space Framework For World Modelling (https://arxiv.org/abs/2410.20922)
Comments:
          Code released in this https URL

- **What's New**: 이 연구에서는 공간-시간(world modelling) 이해 및 예측을 위한 새로운 구조, 즉 	extbf{FACT}ored 	extbf{S}tate-space (	extbf{FACTS}) 모델을 제안합니다.

- **Technical Details**: FACTS는 순환 기반(recurrent) 프레임워크로, 라우팅(routing) 메커니즘을 통해 순서 변경에 불변한 메모리 표현을 학습하는 그래프 구조화(graph-structured) 메모리를 구성합니다. 또한, 이 모델은 선택적 상태 공간 전파(selective state-space propagation)를 통해 적응하며, 고차원(high-dimensional) 시퀀스의 병렬(parallel) 처리를 지원합니다.

- **Performance Highlights**: 다양한 작업을 통한 실험 결과, FACTS 모델은 다변량 시간 시계열 예측(multivariate time series forecasting) 및 객체 중심(object-centric) 세계 모델링에서 전문화된 최첨단(state-of-the-art) 모델을 지속적으로 초과하거나 일치하는 성능을 발휘했습니다.



### Active Causal Structure Learning with Latent Variables: Towards Learning to Detour in Autonomous Robots (https://arxiv.org/abs/2410.20894)
Comments:
          44 pages, 12 figures

- **What's New**: 이 논문은 인공지능 일반화(AGI) 에이전트와 로봇이 변화하는 환경에 적응하는 능력을 강조하며, 새로운 내부 인과 모델을 능동적으로 구축하기 위한 ACTIVE CAUSAL STRUCTURE LEARNING WITH LATENT VARIABLES (ACSLWL)의 필요성에 대해 설명합니다.

- **Technical Details**: ACSLWL 프레임워크는 에이전트가 환경 내에서 상호작용하며 새로운 인과 관계를 발견하고, 이를 바탕으로 인과 모델을 구축하도록 돕습니다. 주요 구성 요소로는 SURPRISE 계수, 잠재 변수 감지, 그리고 DYNAMIC DECISION NETWORK의 구조 조정이 포함됩니다.

- **Performance Highlights**: 논문에서는 에이전트가 예기치 않은 상황을 처리하기 위해 새로운 숨겨진 변수를 생성하고, 이를 통해 예측 가능한 최적의 운용 계획으로 전환할 수 있는 과정을 설명합니다. 이는 AGI 에이전트의 효율적 대처 능력을 높이기 위한 새로운 접근법입니다.



### Explainability in AI Based Applications: A Framework for Comparing Different Techniques (https://arxiv.org/abs/2410.20873)
Comments:
          10 pages, 5 figures

- **What's New**: 이 논문은 다양한 Explainability (설명 가능성) 기술의 출력 간 차이를 이해하는 것을 목표로 한 새로운 방법론을 제안합니다. 이를 통해 비즈니스 응용에서 각 설명 가능성 기술의 선택을 도와주는 포괄적인 비교 분석을 제공합니다.

- **Technical Details**: 논문에서는 Vision Transformer (ViT) 모델을 기반으로 하여, 6가지 주요 설명 가능성 기법 간의 합의(agreement)를 평가하기 위한 새로운 메트릭(metric)을 제안합니다. 이는 데이터에서 각 픽셀의 모델 예측에 대한 영향을 보여주는 Attribution maps를 통해 이루어집니다.

- **Performance Highlights**: 제안된 방법은 비즈니스 응용에서 AI 시스템의 해석 가능성을 높이고, 다양한 설명 가능성 기술 간의 차이를 정량화할 수 있는 실용적인 프레임워크를 제공합니다. 이는 설명 가능성 기술의 선택을 보다 명확하게 하여, AI 시스템의 신뢰성을 향상시키는데 기여합니다.



### Implementation and Application of an Intelligibility Protocol for Interaction with an LLM (https://arxiv.org/abs/2410.20600)
- **What's New**: 이번 연구에서는 인간 전문가와 기계 학습 엔진 간의 상호 작용을 통한 데이터 분석 시스템 구축에 대해 다루고 있습니다. 특히 기존의 통계적 혹은 수학적 모델링 방법으로 해결하기 어려운 복잡한 문제들에 대한 예측 및 설명 생성 방식을 탐구하고 있습니다.

- **Technical Details**: 저자들은 두 가지 상호작용 에이전트 간의 커뮤니케이션을 위한 추상 프로토콜(PXP protocol)을 구현하였습니다. 이 프로토콜은 '양방향 이해 가능성(two-way intelligibility)' 개념을 기반으로 하며, 유한 상태 기계(finite-state machines)를 사용하여 정의됩니다. 실험에서는 대규모 언어 모델(LLM)을 사용하는 생성기(generator) 에이전트와 인간 전문가 또는 인간 전문가의 프록시 역할을 하는 테스트(agent) 간의 상호작용을 설명하고 있습니다.

- **Performance Highlights**: 초기 실험 결과는 PXP 프로토콜이 LLM과 인간 간의 이해 가능성(one- and two-way intelligibility)을 효과적으로 캡처할 수 있음을 제공하고 있습니다. 특히 방사선학(radiology) 및 약물 발견(drug-discovery) 분야에서의 사용 가능성을 보여주고 있습니다.



### AutoKaggle: A Multi-Agent Framework for Autonomous Data Science Competitions (https://arxiv.org/abs/2410.20424)
Comments:
          44 pages, 10 figures

- **What's New**: AutoKaggle은 데이터 과학자들이 데이터 파이프라인을 효율적으로 완성할 수 있도록 돕는 협력적 다중 에이전트 시스템을 기반으로 하는 강력한 프레임워크입니다. 이 프레임워크는 코드 실행, 디버깅, 단위 테스트를 통합하여 코드의 정확성과 논리 일관성을 보장합니다.

- **Technical Details**: AutoKaggle은 단계 기반 워크플로우와 다중 에이전트 협업 시스템을 사용하여 데이터 과학 경쟁 과정을 6개 핵심 단계로 나눕니다. 이 단계는 배경 이해, 예비 탐색적 데이터 분석, 데이터 정리(Data Cleaning), 심층 탐색적 데이터 분석, 특성 공학(Feature Engineering), 모델 구축(Model Building)입니다. 각 단계에서는 독립적인 5개의 에이전트가 협력하여 작업을 수행합니다. 또한, AutoKaggle은 코드의 문법적 정확성과 논리적 일관성을 검증하는 iterative debugging과 unit testing을 통해 코드 품질을 보장합니다.

- **Performance Highlights**: 평가한 8개의 Kaggle 데이터 과학 대회에서 AutoKaggle은 0.85의 유효 제출 비율과 0.82의 종합 점수를 달성하여 데이터 과학 작업을 처리하는 데 있어 효과성과 실용성을 입증했습니다.



### Effective Instruction Parsing Plugin for Complex Logical Query Answering on Knowledge Graphs (https://arxiv.org/abs/2410.20321)
- **What's New**: 이 논문에서는 Knowledge Graph Query Embedding (KGQE) 모델의 성능 향상을 위한 새로운 접근 방법으로 Query Instruction Parsing Plugin (QIPP)을 제안합니다. QIPP는 코드와 같은 쿼리 명령어에서 잠재적인 쿼리 패턴을 캡처하여 기존의 패턴-엔티티 정렬 편향 문제를 해결합니다.

- **Technical Details**: QIPP는 사전 훈련된 언어 모델(Pre-trained Language Models, PLMs)의 맥락 인식을 활용하여 코드와 같은 쿼리 명령어에서 정보를 추출합니다. 쿼리 패턴 학습을 위해 변수를 텍스트로 표현하고 중첩된 튜플을 사용하여 FOL 쿼리의 논리적 의미를 전달합니다. 또한, QIPP는 압축 최적화 경계를 기반으로 한 쿼리 패턴 주입 메커니즘과 적응형 정규화 구성 요소를 통해 KGQE 모델이 쿼리 패턴을 보다 효과적으로 사용할 수 있도록 설계되었습니다.

- **Performance Highlights**: QIPP는 8개의 기본 KGQE 모델의 성능을 크게 향상시키며, 두 가지 최첨단 QPL 방법보다 우수한 성능을 보입니다. 다양한 벤치마크에서 실험 결과가 입증되었습니다.



### SWE-Search: Enhancing Software Agents with Monte Carlo Tree Search and Iterative Refinemen (https://arxiv.org/abs/2410.20285)
Comments:
          Main body: 10 pages, 5 figures. Appendix: 5 pages, 4 figures. Open-source codebase

- **What's New**: 본 논문에서는 소프트웨어 엔지니어링에서의 복잡한 문제를 해결하기 위해 SWE-Search라는 다중 에이전트 시스템을 소개합니다. 이 시스템은 Monte Carlo Tree Search (MCTS)와 자기 개선 메커니즘을 통합하여 소프트웨어 에이전트의 성능을 향상시킵니다.

- **Technical Details**: SWE-Search는 MCTS를 기반으로 하며, LLM을 활용한 하이브리드 가치 함수로 성능을 극대화합니다. 이 프레임워크는 SWE-Agent(적응형 탐색), Value Agent(반복 피드백), Discriminator Agent(다중 에이전트 토론)를 포함하여 협업적 의사결정을 지원합니다. 경량화된 설계를 통해 풍부한 피드백 루프를 활성화합니다.

- **Performance Highlights**: SWE-Search는 SWE-bench 벤치마크 테스트에서 표준 오픈 소스 에이전트에 비해 23%의 상대 성능 향상을 보여주었습니다. 이를 통해 MCTS 기반의 탐색 및 반복 자기 평가의 효과를 입증하였습니다.



### Rethinking the Uncertainty: A Critical Review and Analysis in the Era of Large Language Models (https://arxiv.org/abs/2410.20199)
- **What's New**: 이 논문은 최신 Large Language Models (LLMs)의 예측 불확실성을 측정하고 이해하기 위한 포괄적인 프레임워크를 제안합니다. 기존의 방법들은 주로 모델의 신뢰도를 추정하는 데 초점을 맞추었으나, 이 연구는 다양한 불확실성의 유형 및 출처를 체계적으로 분류하고 정의함으로써 기존의 프레임워크의 한계를 극복하고자 합니다.

- **Technical Details**: 연구는 LLMs의 생애 주기 전반에서 발생하는 다양한 불확실성의 출처를 분석합니다. 두 가지 전통적인 불확실성 유형인 aleatoric (데이터 불확실성)과 epistemic (모델 지식의 한계)을 언급하며, LLM의 특성과 지나친 대규모 파라미터 관리, 접근이 어려운 훈련 데이터와 같은 요소들이 불확실성을 어떻게 형성하는지를 탐구합니다.

- **Performance Highlights**: 이 프레임워크는 LLM의 불확실성을 이해하고 분석할 수 있는 기초를 마련하며, 모델의 투명성과 신뢰성을 높이는 방향으로 향후 연구 방향을 제시합니다. 향후 연구에서는 불확실성 추정을 개선하기 위한 정확한 방법론 마련을 목표로 하며, 안전이 중요한 응용 분야에서의 활용 가능성을 높이는 것에 중점을 두고 있습니다.



### LLMs Can Evolve Continually on Modality for X-Modal Reasoning (https://arxiv.org/abs/2410.20178)
- **What's New**: PathWeave는 Modal-Path 전환 및 확장 기능을 갖춘 유연하고 확장 가능한 프레임워크로, Multimodal Large Language Models (MLLMs)의 연속적인 모달 진화를 가능하게 합니다. 이를 통해 단일 모달 데이터로 새로운 모달리티에 대한 확장과 학습이 가능합니다.

- **Technical Details**: PathWeave는 Incremental Training Strategy를 바탕으로 하여, Uni-modal 및 Cross-modal Adapter를 통합하여 효율적인 모달 정렬 및 협업을 촉진합니다. MoE (Mixture of Experts) 기반의 게이팅 모듈을 통해 두 가지 유형의 어댑터 간의 멀티모달 상호작용을 향상시킵니다.

- **Performance Highlights**: PathWeave는 최신의 MLLMs와 비교하여 성능 면에서 동등하며, 파라미터 학습 부담을 98.73% 감소시키는 동시에, Continual Learning of Modality (MCL) 벤치마크에서의 높은 정확도를 보여줍니다.



### MAD-Sherlock: Multi-Agent Debates for Out-of-Context Misinformation Detection (https://arxiv.org/abs/2410.20140)
- **What's New**: 이번 논문에서는 OOC(Out-Of-Context) 잘못된 정보 검색을 위한 새로운 시스템인 MAD-Sherlock을 제안합니다. MAD-Sherlock은 다중 에이전트 토론 시스템으로, 다양한 멀티모달 에이전트들이 협력하여 맥락 일관성을 평가하고 외부 정보를 요청함으로써 교차 컨텍스트 추론 및 의사 결정을 향상시킵니다.

- **Technical Details**: MAD-Sherlock은 OOC 잘못된 정보 탐지를 위해 다수의 LMM(Large Multimodal Models) 에이전트 간의 변증법적 토론을 기반으로 구성됩니다. 이 시스템은 외부 정보 검색을 통해 단일 에이전트 기반 접근 방식보다 더욱 향상된 성능을 발휘합니다. 또한, 자율적이며 상호작용 가능한 도구로서 전문가와 비전문가 모두에게 유용한 설명 가능한 탐지를 제공합니다.

- **Performance Highlights**: MAD-Sherlock은 기존 연구보다 더 높은 탐지 정확도를 보여줍니다. 사용자 연구에서는 전문가와 비전문가 모두의 탐지 성능이 크게 향상된 것으로 나타났으며, 외부 정보 검색과 자유로운 의견 표현이 MAD-Sherlock의 성능을 결정짓는 중요한 요인으로 확인되었습니다.



### Cooperative Strategic Planning Enhances Reasoning Capabilities in Large Language Models (https://arxiv.org/abs/2410.20007)
Comments:
          Working in progress

- **What's New**: 이번 논문에서는 대규모 언어 모델(LLMs)의 추론 능력을 향상시키기 위해 협력적인 다중 에이전트 추론 프레임워크인 CoPlanner를 제안합니다. CoPlanner는 추론 단계를 분리하고, 각 에이전트에게 고유한 역할을 부여하여 문제를 해결합니다.

- **Technical Details**: CoPlanner는 두 개의 LLM 에이전트로 구성됩니다: 계획 에이전트(planning agent)와 추론 에이전트(reasoning agent). 계획 에이전트는 고차원 전략 힌트를 제공하고, 추론 에이전트는 이 힌트를 따르며 답변을 추론합니다. 이 과정은 Proximal Policy Optimization (PPO)을 통해 훈련되며, 각 에이전트는 명확한 역할을 수행하여 서로의 강점을 활용하는 구조로 되어 있습니다.

- **Performance Highlights**: LLaMA-3-8B 기반의 CoPlanner는 LogiQA 기준에서 기존 최고의 방법보다 9.94% 더 우수한 성능을 보였으며, BBH에서 3.09% 향상된 결과를 기록했습니다. 계획 에이전트의 가이드와 에이전트 간의 효과적인 협력이 CoPlanner의 우수한 성능에 기여했음을 보여줍니다.



### Language Agents Meet Causality -- Bridging LLMs and Causal World Models (https://arxiv.org/abs/2410.19923)
Comments:
          Project page: this https URL

- **What's New**: 본 논문에서는 Causal Representation Learning (CRL)과 대형 언어 모델(Large Language Models, LLMs)의 통합 프레임워크를 제안하여 인과관계를 인식할 수 있는 추론 및 계획을 가능하게 하고 있습니다. 이 프레임워크는 인과 변수를 자연어 표현과 연결된 인과 세계 모델(causal world model)을 학습하여 LLM이 텍스트 형식으로 행동과 상태에 대한 설명을 처리하고 생성할 수 있도록 합니다.

- **Technical Details**: 제안된 프레임워크는 CRL을 통해 환경의 인과 구조를 이해하고 그에 따른 개입과 결과에 대한 추론을 수행합니다. 이 인과 세계 모델은 LLM이 여러 가능한 미래를 평가하고 이를 기반으로 행동을 취할 수 있도록 돕는 시뮬레이터 역할을 합니다. 또한, 텍스트 기반의 행동 표현을 활용하여 다양한 환경에서 보다 직관적인 계획과 추론이 가능합니다.

- **Performance Highlights**: 실험 결과, 인과 인식 방법이 LLM 기반 추론 기법보다 우수한 성능을 보였으며, 특히 더 긴 계획 수립 구간에서는 더욱 싸고 효과적인 결과를 나타냈습니다. 기존 CRL 방법을 사용한 간단한 환경에서의 실험을 통해 이 프레임워크의 효과성을 입증하였습니다.



### Step Guided Reasoning: Improving Mathematical Reasoning using Guidance Generation and Step Reasoning (https://arxiv.org/abs/2410.19817)
Comments:
          4 pages, 4 figures

- **What's New**: 이 논문에서는 기존의 Chain-of-Thought (CoT) 방법론을 넘어서는 새로운 접근 방식인 Step Guidance Reasoning (SGR)을 소개합니다. SGR은 모델의 Fine-tuning 없이 단계별 추론 과정을 시각화하고, 이러한 단계를 통해 수학 문제 해결의 정확성을 크게 향상시킵니다.

- **Technical Details**: SGR은 각 추론 단계에서 모델이 스스로 다음 해야 할 일을 질문하고 대답하게 하며, 이 성찰(reflective process)을 통해 다음 단계로 나아가는 과정을 안내합니다. 본 방법은 추론 단계 동안 300에서 500개의 토큰을 사용하는 다양한 단계 길이(constraints)를 설정합니다.

- **Performance Highlights**: AMC23 데이터셋에서 정확도가 30%에서 57.5%로 상승하며 91.7%의 상대적 개선을 이루었고, MATH 데이터셋의 5단계 문제에서 43%에서 67%로 증가하여 55.8%의 상대적 정확도 개선을 기록했습니다.



### ScreenWriter: Automatic Screenplay Generation and Movie Summarisation (https://arxiv.org/abs/2410.19809)
- **What's New**: 이번 연구에서는 비디오 콘텐츠를 자동으로 스크린플레이(screenplay)로 생성하는 새로운 접근 방식을 제안합니다. 특히, ScreenWriter라는 알고리즘을 통해 비디오만을 이용하여 대화(dialogue), 화자 이름(speaker names), 장면 구분(scene breaks), 시각적 설명(visual descriptions)을 포함한 스크립트를 작성합니다.

- **Technical Details**: ScreenWriter는 비디오를 시각적 벡터(visual vectors)의 순서에 따라 장면으로 나누는 혁신적인 알고리즘을 소개합니다. 캐릭터 이름을 식별하는 문제를 해결하기 위해 배우의 얼굴 데이터베이스를 활용하고, 이를 통해 자동 생성된 스크린플레이를 바탕으로 계층적 요약(hierarchical summarisation) 방법을 통해 줄거리 개요(plot synopses)를 생성할 수 있음을 입증하였습니다.

- **Performance Highlights**: 최종 요약의 품질을 MovieSum 데이터셋에서 테스트한 결과, 기존의 스크린플레이에 접근할 수 있다고 가정하는 여러 모델보다 우수한 성능을 보였습니다. 오류의 주요 원인으로는 잘못된 얼굴 정보, 다중 화자 식별 문제 등이 있으며, 향후 이러한 문제를 개선할 수 있는 방법이 기대됩니다.



### Integrating Reasoning Systems for Trustworthy AI, Proceedings of the 4th Workshop on Logic and Practice of Programming (LPOP) (https://arxiv.org/abs/2410.19738)
- **What's New**: 2024년 제4회 논리와 프로그래밍의 실천 워크숍(LPOP)이 미국 텍사스주 댈러스에서 하이브리드 포맷으로 개최됩니다. 이 워크숍은 신뢰할 수 있는 AI를 위한 추론 시스템 통합에 중점을 두고, 여러 프로그래밍 모델, 규칙, 제약 조건을 논의합니다.

- **Technical Details**: 이번 워크숍에서는 고전적인 1차 논리(First-Order Logic, FOL), Datalog 기반의 추론, Prolog를 사용한 논리 프로그래밍, Answer Set Programming (ASP), 제약 논리 프로그래밍(Constraint Logic Programming, CLP) 등 다양한 프로그래밍 모델을 다룹니다. 이러한 시스템들은 비즈니스 문제 해결을 위해 선형 프로그래밍과 비선형 프로그래밍 방법론을 포함한 수학적 프로그래밍(Mathematical Programming) 접근을 요구합니다.

- **Performance Highlights**: LPOP 2024의 초청 연사들은 Luc De Raedt(Neurosymbolic AI), Georg Gottlob(대규모 언어 모델의 한계), Henry Kautz(형식적 추론 도구 사용 가능성) 등의 주제를 논의하며, 참가자들은 프로그램 내에서 다양한 실제 사례 문제를 통해 신뢰할 수 있는 AI 개발에 대해 논의하게 됩니다.



### GPT-4o System Card (https://arxiv.org/abs/2410.21276)
- **What's New**: GPT-4o는 텍스트, 오디오, 이미지, 비디오 등 다양한 입력을 지원하는 자가 회귀형 omni 모델로, 통합된 신경망을 통해 텍스트, 오디오, 이미지 출력을 생성합니다. 이 모델은 대화에서 인간의 반응 시간과 유사하게 평균 320밀리초로 오디오 입력에 응답할 수 있어 향상된 반응성을 보여줍니다.

- **Technical Details**: GPT-4o는 텍스트와 음성 기능이 2023년 10월까지의 다양한 자료를 바탕으로 사전 훈련되었습니다. 웹 데이터, 코드 및 수학, 다중 모달 데이터가 포함되어 있어 모델이 비텍스트 입력과 출력을 해석하고 생성하는 방법을 학습하게 됩니다. 또한, 위험을 줄이기 위한 방법으로는 Moderation API와 안전 분류기를 사용하여 유해 콘텐츠를 필터링합니다.

- **Performance Highlights**: GPT-4o는 영어 및 코드 텍스트에서 GPT-4 Turbo의 성능을 맞추며 비영어 텍스트에서는 현저한 개선을 보여줍니다. API 비용은 50% 절감되며, 비전 및 오디오 이해 능력이 기존 모델보다 뛰어난 것으로 확인되었습니다.



### Enhancing Action Recognition by Leveraging the Hierarchical Structure of Actions and Textual Contex (https://arxiv.org/abs/2410.21275)
- **What's New**: 이 논문에서는 행동 인식(action recognition)을 개선하기 위한 새로운 접근 방식을 제안합니다. 이는 행동의 계층 구조와 맥락화된 텍스트 정보(예: 위치 및 이전 행동)를 활용하여 순차적인 맥락을 반영합니다. 저자는 행동 인식에 최적화된 새로운 transformer 아키텍처를 도입하고, 이를 통해 RGB 및 optical flow 데이터를 기반으로 시각적 특징을 얻고 텍스트 임베딩을 통해 맥락 정보를 표현합니다.

- **Technical Details**: 제안된 모델은 행동 인식의 정밀도를 높이기 위해 coarse(거친) 및 fine-grained(세밀한) 행동 인식을 동시에 훈련할 수 있도록 설계된 joint loss function을 사용합니다. 이 모델은 transformer 아키텍처를 기반으로 하며, 자기 주의 메커니즘(self-attention mechanism)을 활용하여 긴 범위의 시간적 의존성을 캡처합니다. Hierarchical TSU 데이터셋을 확장하여 행동의 계층을 도입한 것이 특징입니다.

- **Performance Highlights**: 제안된 방법은 동일한 하이퍼파라미터(hyperparameter)로 훈련했을 때 사전 훈련된 SOTA 방법들을 초월했습니다. 첫 번째로, ground-truth 맥락 정보를 사용했을 때 top-1 정확도에서 17.12% 향상된 결과를 얻었고, 실제 예측에서 얻은 맥락 정보를 사용할 때는 5.33% 향상이 나타났습니다.



### EoRA: Training-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation (https://arxiv.org/abs/2410.21271)
- **What's New**: 본 연구에서는 모델 압축 문제를 사용자 맞춤형 보상 문제로 재구성하였습니다. 압축된 모델을 기준으로 잔여 저랭크 경로를 도입하여 압축 오류를 보상하는 접근법을 제안합니다.

- **Technical Details**: 제안된 EoRA(Training-free Eigenspace Low-Rank Approximation)는 압축 오류를 입력 활성화의 고유 공간 (eigenspace)으로 투사하여 오류 부품의 재구성을 우선시합니다. 이 방법은 고유값 (eigenvalues)을 활용하여 효과적인 저랭크 표현 능력을 발휘하고, 기울기 계산 없이도 빠른 최적화를 가능하게 합니다.

- **Performance Highlights**: EoRA는 LLaMA2/3 모델의 다양한 작업에서 이전 방법들을 상회하는 성능을 보여주며, 언어 생성과 상식 추론 및 수학 추론 작업에서 31.31% 및 9.69%의 성능 향상을 달성하였습니다. EoRA는 또한 4-비트 양자화된 모델에서 원래 모델보다 더 나은 정확성을 보여 주기도 했습니다.



### LARP: Tokenizing Videos with a Learned Autoregressive Generative Prior (https://arxiv.org/abs/2410.21264)
Comments:
          Project page: this https URL

- **What's New**: 이번 논문에서는 이전 영상 토크나이저의 한계를 극복하기 위해 설계된 새로운 비디오 토크나이저인 LARP를 소개합니다. LARP는 전통적인 패치 기반 토크나이저와는 달리 전반적인 정보 수집을 통해 더 글로벌하고 의미 있는 표현을 캡처합니다.

- **Technical Details**: LARP는 비디오 내용을 토큰화하기 위해 학습된 전체 쿼리 집합을 사용하여 정보 수집을 수행합니다. 이를 통해 LARP는 불규칙한 수의 이산 토큰을 지원하며, 적응형 효율적인 토크나이제이션을 가능하게 합니다. 또한 경량의 AR transformer를 이전 모델로 통합하여 이산 잠재 공간의 다음 토큰을 예측합니다.

- **Performance Highlights**: LARP는 UCF101 클래스 조건 비디오 생성 벤치마크에서 최첨단 FVD 점수를 기록하며, 기존의 모든 비디오 생성 모델을 초월합니다. 이를 통해 AR 모델들이 비디오에 대한 호환성을 높이고, 고충실도 다중 모달 대형 언어 모델(MLLM) 개발의 잠재력을 열어줍니다.



### BLAST: Block-Level Adaptive Structured Matrices for Efficient Deep Neural Network Inferenc (https://arxiv.org/abs/2410.21262)
- **What's New**: 이 논문에서는 대규모 파운데이션 모델의 효율적인 구조를 위한 Block-Level Adaptive Structured (BLAST) 매트릭스를 소개합니다. BLAST 매트릭스는 기존 구조화된 매트릭스보다 더 유연하게 다양한 구조를 표현할 수 있으며, 딥러닝 모델의 선형 레이어의 가중치 매트릭스에서 효율적인 구조를 학습하여 불필요한 계산 복잡성을 줄이고 성능을 개선합니다.

- **Technical Details**: BLAST 매트릭스는 가중치의 저차원 구조를 포착하기 위해 블록 행렬 사이에 공유 기저를 활용합니다. 이 구조는 저랭크(low-rank), 블록 저랭크(block low-rank), 블록 대각행렬(block-diagonal) 등의 다양한 구조를 포괄합니다. BLAST 매트릭스는 데이터로부터 학습하여 요인(factors)을 최적화할 수 있으며, 기존 가중치를 압축하고 재학습할 수 있는 알고리즘도 구현되어 있습니다.

- **Performance Highlights**: BLAST 매트릭스를 사용하여 ViT와 GPT-2 모델을 훈련할 경우 각각 70% 및 40%의 계산 복잡성을 줄이면서 성능을 개선하는 결과를 보여줍니다. 또한 Llama-7B 모델을 BLAST로 50% 압축하여 NVIDIA A100 GPU에서 상당한 추론 속도 향상을 달성하였습니다.



### AutoBench-V: Can Large Vision-Language Models Benchmark Themselves? (https://arxiv.org/abs/2410.21259)
- **What's New**: 이번 논문에서는 LVLMs (Large Vision-Language Models)의 평가를 위한 자동화된 프레임워크인 AutoBench-V를 도입했습니다. 이 프레임워크는 특정 모델 능력에 기반한 자동 평가를 지원하여 LVLM의 효과적인 벤치마킹을 가능하게 합니다.

- **Technical Details**: AutoBench-V는 사용자 요구에 따라 모델의 특정 능력 (예: Spatial Understanding)에 대한 자동화를 지원합니다. 이 프레임워크는 텍스트-이미지 모델을 활용하여 관련 이미지를 생성하고, LVLM를 이용하여 시각적 질문-답변 (Visual Question Answering) 작업을 수행합니다. 또한, 자가 검증 메커니즘을 통해 참조 답변과 비교하여 성과를 평가합니다.

- **Performance Highlights**: AutoBench-V를 통해 7개의 인기 LVLM을 5가지 평가 능력에 대해 광범위하게 평가한 결과, 모델들은 높은 수준의 이해력에서는 강점을 보이지만, 세부적인 추론 작업에서는 낮은 성과를 보였으며, 이는 향후 연구의 개선 가능성이 있는 중요한 영역으로 분석되었습니다.



### Capacity-Aware Planning and Scheduling in Budget-Constrained Monotonic MDPs: A Meta-RL Approach (https://arxiv.org/abs/2410.21249)
- **What's New**: 본 논문에서는 예산과 수용 능력 제약을 가진 다중 구성 요소 단조 (monotonic) 마르코프 결정 프로세스 (MDP)를 해결하는 방법을 제안합니다. 특히, 수리 기술자 수와 총 수리 예산에 의해 제한된 대규모 산업 로봇을 위한 수리 일정을 계획하고 배분하는 문제를 다룹니다.

- **Technical Details**: 제안된 방법은 두 단계의 계획 접근 방식으로, 첫 번째 단계에서는 Linera Sum Assignment Problem (LSAP)를 사용하여 다중 구성 요소 MDP의 요소들을 그룹으로 분할합니다. 각 그룹에 대한 예산은 그룹의 크기에 비례하여 할당됩니다. 이후 각 그룹에 대해 메타 훈련된 PPO (Proximal Policy Optimization) 에이전트를 사용하여 근사 최적 정책을 도출합니다.

- **Performance Highlights**: 본 연구의 결과는 제안된 방법이 로봇 군집의 평균 가동 시간 증대에 있어 기준 접근 방식을 초월한 성능을 보여줌을 나타내며, 특히 큰 군집 크기에 대해 효과적임을 입증합니다.



### Zero-Shot Dense Retrieval with Embeddings from Relevance Feedback (https://arxiv.org/abs/2410.21242)
- **What's New**: 이 논문에서는 ReDE-RF(Real Document Embeddings from Relevance Feedback)라는 새로운 접근 방식을 도입하여, LLM을 활용한 가상의 문서 생성 대신 관련성 추정(task)을 프레임으로 삼아 검색 효율성을 크게 향상시킵니다.

- **Technical Details**: ReDE-RF는 먼저 완전 비지도 하이브리드 스파스-밀집 검색 시스템에서 초기 문서 집합을 추출한 후, LLM을 통해 반환된 문서가 관련성이 있는지 없는지를 평가합니다. 이후, 관련 문서 집합을 기반으로 미리 계산된 문서 임베딩을 효율적으로 가져와 쿼리 벡터를 업데이트합니다. 이는 LLM이 도메인 특화 지식에 의존하지 않고 단순히 관련성을 판단하게 합니다.

- **Performance Highlights**: 실험 결과, ReDE-RF는 여러 저자원 검색 데이터셋에서 기존의 최첨단 제로샷 밀집 검색 방법들보다 최대 14% 더 우수한 성능을 보였으며, 검색 지연(lag)을 7.5배에서 11.2배까지 줄이는 성과를 나타냈습니다.



### Flaming-hot Initiation with Regular Execution Sampling for Large Language Models (https://arxiv.org/abs/2410.21236)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 훈련 중 효율적인 샘플링 방법인 Flaming-hot Initiation with Regular Execution (FIRE) 샘플링을 소개합니다. 이 방법은 초기 토큰을 높은 온도에서 샘플링하여 우수한 응답을 찾아내는 방식으로, 다양한 도메인에서 세분화된 문제 해결 능력을 향상시키는 데 기여합니다.

- **Technical Details**: FIRE sampling은 높은 온도의 초기 토큰 샘플링과 일반적인 샘플링 프로세스를 결합하여 진행됩니다. 이 접근법은 attention sink 이론을 활용하며, 문제 해결 시 샌드박스 체커의 존재로 인해 더 나은 샘플을 생성할 수 있도록 돕습니다. FIRE는 CoT-decoding의 간소화 및 일반화를 통해 매개변수 조정 없이도 기존 훈련 프레임워크와 통합될 수 있습니다.

- **Performance Highlights**: FIRE sampling을 통해 여러 공개 소스 모델에서 추론 시간의 생성 품질이 개선되었고, 특히 수학 문제와 코드 생성에서 높은 통과율(pass rate)을 기록했습니다. 이 방법을 통해 생성된 샘플의 다양성이 향상되었으며, 이는 결과적으로 성능 개선과 직결됩니다.



### Vision Search Assistant: Empower Vision-Language Models as Multimodal Search Engines (https://arxiv.org/abs/2410.21220)
Comments:
          Code is available at this https URL

- **What's New**: 이 논문에서는 기존 대규모 비전-언어 모델(Vision-Language Models, VLM)이 처리하지 못하는 새로운 시각 정보를 이해하고 응답할 수 있는 'Vision Search Assistant'라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 VLM과 웹 에이전트(web agent)의 협업을 통해 실시간 정보를 검색하고 새로운 객체에 대한 질문을 다룰 수 있도록 합니다.

- **Technical Details**: Vision Search Assistant는 VLM이 질문을 이해하고, 이미지의 특정 객체를 분석하며, 관련된 정보와 텍스트를 검색하는 과정을 포함합니다. 이 과정은 크게 세 단계로 나뉘며, 첫째로 크리티컬 비주얼 객체의 시각적 내용을 텍스트 형태로 서술하는 'Visual Content Formulation', 둘째로 사용자의 질문과 관련된 서브 질문을 생성하여 웹 지식을 검색하는 'Web Knowledge Search', 셋째로 최종 답변을 생성하는 'Collaborative Generation'을 포함합니다.

- **Performance Highlights**: 다양한 오픈세트(open-set) 및 클로즈드세트(closed-set) QA 벤치마크에서 extensive experiments를 실시한 결과, Vision Search Assistant는 기존 모델보다 훨씬 높은 성능을 보이며, 대규모 비전-언어 모델에 효과적으로 적용될 수 있음을 입증했습니다.



### HoPE: A Novel Positional Encoding Without Long-Term Decay for Enhanced Context Awareness and Extrapolation (https://arxiv.org/abs/2410.21216)
- **What's New**: 본 논문에서는 기존 Positional Encoding (PE)의 긴 거리 토큰에 대한 규칙인 'long-term decay'의 필요성을 반박하고, 이를 대체할 새로운 방법인 High-frequency rotary Position Encoding (HoPE)를 제안합니다. HoPE는 Rotary Position Encoding (RoPE)의 한계를 해결하고 모델의 맥락 인지와 추론 능력을 향상시킵니다.

- **Technical Details**: 기존의 PE 모델은 긴 거리 토큰이 모델의 정보 처리에 미치는 영향을 줄이는 long-term decay 원칙을 따르지만, 실험 결과 LLMs에서 모델들은 단기적인 정보에만 집중하는 경향을 보입니다. HoPE는 RoPE의 구성 요소 중 위치 독립적인 것과 높은 주파수의 신호로 대체하여 모델의 전반적인 성능을 향상시킵니다.

- **Performance Highlights**: HoPE는 언어 모델링의 perplexity, 복사 작업, 그리고 few-shot 학습 작업에서 기존의 PE 방식보다 우수한 성능을 보여주었습니다. 이 연구는 특정 주파수의 'activated' 구성 요소가 RoPE의 성능 한계를 유발했음을 보여줍니다.



### SeriesGAN: Time Series Generation via Adversarial and Autoregressive Learning (https://arxiv.org/abs/2410.21203)
Comments:
          This work has been accepted at BigData 2024 on October 26, 2024, as a regular paper for oral presentation

- **What's New**: 이번 논문은 Generative Adversarial Network (GAN) 기반의 시간 시계열 생성 방식에서 발생하는 여러가지 문제점을 해결하기 위한 새로운 프레임워크인 SeriesGAN을 소개합니다. 이 프레임워크는 자동 인코더 기반의 임베딩 공간과 GAN의 적대적 훈련 역학을 통합하여 정보 손실을 최소화하고 시간 시계열 데이터를 고품질로 생성하는 데 최적화되었습니다.

- **Technical Details**: SeriesGAN은 각각 특성 공간(feature space)과 잠재 공간(latent space)에서 작동하는 두 개의 판별기(discriminator)를 활용하여 생성기(generator)에 대한 피드백을 선택적으로 제공합니다. 또한 시계열 데이터의 단계적 조건 분포를 포착하는 teacher-forcing 감독 네트워크를 통합하고, 자동 인코더를 위한 새로운 손실 함수(loss function)도 개발했습니다. 이러한 설정은 생성기와 자동 인코더의 결과를 정제하여 데이터 생성의 질을 향상시킵니다.

- **Performance Highlights**: SeriesGAN은 다양한 실제 및 합성 다변량(multivariate) 및 단변량(univariate) 시계열 데이터셋에서 테스트한 결과, 기존 최첨단 성능을 가진 모델인 TimeGAN을 포함한 다른 방법들에 비해 정성적 및 정량적으로 모든 측면에서 우수한 성능을 보여주었습니다.



### BongLLaMA: LLaMA for Bangla Languag (https://arxiv.org/abs/2410.21200)
Comments:
          19 pages

- **What's New**: BongLLaMA(즉, Bangla-LLaMA)라는 새로운 오픈 소스 대형 언어 모델이 발표되었습니다. 이는 대규모 방글라어 코퍼스를 기반으로 독점적으로 파인 튜닝(fine-tuning)된 모델입니다.

- **Technical Details**: 본 연구에서는 방글라어 데이터 증강(data augmentation) 기법, 파인 튜닝 세부 사항, 그리고 BongLLaMA 모델의 효과성을 입증하는 종합적인 벤치마킹(benchmarking) 결과를 제시합니다. 이 모델은 방글라어 언어 처리(BLP) 작업을 위한 최적화된 성능을 자랑합니다.

- **Performance Highlights**: BongLLaMA는 방글라어 언어 모델의 새로운 기준이 될 것으로 예상되며, 향후 방글라어의 다양한 언어 처리 연구에 유용한 기초 자료를 제공합니다. 모든 BongLLaMA 모델은 공공용으로 제공됩니다.



### Belief in the Machine: Investigating Epistemological Blind Spots of Language Models (https://arxiv.org/abs/2410.21195)
Comments:
this https URL

- **What's New**: 이 연구는 기존 언어 모델들이 사실, 신념, 지식의 차이를 구분하는 능력에 대한 연구가 부족하다는 점을 지적하며, 13,000개의 질문을 포함한 새로운 데이터셋인 KaBLE를 통해 이들 능력을 체계적으로 평가하였습니다.

- **Technical Details**: 연구는 현대 언어 모델인 GPT-4, Claude-3, Llama-3 등 15개 모델을 대상으로 하였으며, 이들 모델이 신념과 지식, 사실의 차이를 처리하는 능력을 측정했습니다. 개별 과제를 통해 얻어진 결과에 따르면, 언어 모델은 필수적인 상식적 구분에 대해 심각한 한계를 보였습니다.

- **Performance Highlights**: 모델들은 사실적 시나리오에서는 86%의 정확도를 보였지만, 잘못된 시나리오에서는 성능이 급락했습니다. 특히 첫 번째 인칭 신념 확인 작업에서 54.4%의 낮은 정확도를 나타내며, 개인의 신념을 인식하고 지지하는 데 어려움을 겪고 있습니다. 이러한 발견은 의료 및 상담과 같은 분야에서의 모델 적용에 중대한 우려를 불러일으킵니다.



### Deep Learning-Based Fatigue Cracks Detection in Bridge Girders using Feature Pyramid Networks (https://arxiv.org/abs/2410.21175)
Comments:
          15 pages, 11 figures

- **What's New**: 본 연구는 교량의 강철 박스 기둥에 대한 균열 정보가 포함된 고해상도 이미지를 사용하여 자동 균열 세분화를 위한 새로운 프레임워크를 제안합니다.

- **Technical Details**: 이 연구에서는 균열의 다중 스케일 특성을 고려하여 Crack Detection을 위한 Feature Pyramid Networks (FPN) 아키텍처를 제안합니다. 입력으로 120개의 원본 이미지를 사용하며, 이미지 크기를 축소하고 하위 이미지로 분할하는 두 가지 접근 방식을 통해 처리합니다.

- **Performance Highlights**: 모든 개발된 모델은 원본 이미지에서 균열을 자동으로 감지할 수 있으며, 이미지 크기를 축소함으로써 계산 효율성을 개선하면서 정확도는 유지됩니다. 또한, 분할 방법을 사용하는 모델이 리사이징 방법을 사용하는 모델보다 더 정확한 균열 세분화를 제공합니다.



### Document Parsing Unveiled: Techniques, Challenges, and Prospects for Structured Information Extraction (https://arxiv.org/abs/2410.21169)
- **What's New**: 이번 논문은 문서 파싱(document parsing)의 최신 발전 상태를 포괄적으로 리뷰하며, 대형 언어 모델과 비전-언어 모델을 활용한 문서 파싱 기술의 중요성을 강조합니다.

- **Technical Details**: 주요 방법론으로는 모듈형 파이프라인 시스템(modular pipeline systems)과 end-to-end 모델이 있습니다. 핵심 구성 요소에는 레이아웃 감지(layout detection), 콘텐츠 추출(content extraction), 멀티모달 데이터 통합(multi-modal data integration)이 포함됩니다. 또한 복합 레이아웃을 처리하는 데 필요한 도전과제와 기술적 발전도 다루어집니다.

- **Performance Highlights**: 문서 파싱 기술이 정보 추출 및 문서 이해와 같은 여러 작업에서 중대한 진전을 이루었으며, 이는 RAG 시스템 및 차세대 지능형 시스템 개발을 위한 견고한 기초를 제공합니다.



### CURATe: Benchmarking Personalised Alignment of Conversational AI Assistants (https://arxiv.org/abs/2410.21159)
Comments:
          Submitted to ICLR 2025 on 01/10/2024

- **What's New**: LLM 기반 AI 어시스턴트의 개인화된 정렬을 평가하기 위한 다체계 벤치마크인 'CURATe'를 도입하며, 안전이 중요한 상황에서 사용자 제공 컨텍스트를 처리하는 능력에 중점을 두고 있습니다.

- **Technical Details**: 이 연구에서는 10개의 주요 LLM 모델을 5개의 시나리오에서 평가하며, 사용자의 특정 정보를 기억하고 적절히 활용하는 모델의 능력을 분석합니다. 특히 'sycophancy'(사용자 선호도를 안전보다 우선시하는 경향) 및 갈등 선호도의 부적절한 가중치 등 주요 실패 양상이 드러났습니다.

- **Performance Highlights**: 안전 관련 컨텍스트를 고려하도록 LLM을 유도하는 것이 성능을 상당히 향상시키는 것으로 나타났으며, 이는 일반적인 '무해하고 유용한' 지침과는 대조적입니다. 이 연구는 AI 어시스턴트의 더 안전하고 배려 깊은 개발을 위한 연구 방향을 제안합니다.



### Trajectory Flow Matching with Applications to Clinical Time Series Modeling (https://arxiv.org/abs/2410.21154)
Comments:
          NeurIPS 2024 Spotlight

- **What's New**: 이번 논문에서는 Trajectory Flow Matching (TFM)이라는 새로운 기법을 제안하여 Neural stochastic differential equations (Neural SDEs)의 훈련 방식을 개선합니다. TFM은 SDE 동역학을 통한 역전파(backpropagation)를 우회하여 시뮬레이션 없이 Neural SDE를 훈련시킬 수 있는 방법입니다.

- **Technical Details**: TFM은 생성 모델링에서의 flow matching 기법을 활용하여 시계열 데이터(time series data)를 모델링합니다. 논문에서는 TFM이 시계열 데이터를 학습하는 데 필요한 조건을 수립하며, 훈련 안정성을 높이기 위한 재매개변수화 트릭(reparameterization trick)을 제시합니다.

- **Performance Highlights**: TFM을 임상 시계열 설정(clinical time series setting)에 적용하여, 세 가지 임상 시계열 데이터 세트(dataset)에서 성능과 불확실성 예측 관점에서 향상된 성과를 보여줍니다.



### Palisade -- Prompt Injection Detection Framework (https://arxiv.org/abs/2410.21146)
- **What's New**: 이 논문은 Large Language Models (LLMs)의 새로운 약점을 다루고 있습니다. 특히, 악의적인 프롬프트 주입 공격(prompt injection attacks)에 대한 새로운 탐지 방법을 제안합니다.

- **Technical Details**: 이 논문에서 제안하는 방법은 세 가지 계층(layer)으로 구성된 입력 필터링 시스템을 사용합니다. 첫 번째는 규칙 기반(rule-based) 필터, 두 번째는 머신러닝(Machine Learning) 분류기(classifier), 세 번째는 동반 LLM(companion LLM)입니다. 이 시스템은 다양한 공격을 효과적으로 탐지하고, 악성 입력으로부터 모델을 보호합니다.

- **Performance Highlights**: 제안된 다계층 탐지 프레임워크는 개별 계층 중에서 ML 분류기가 가장 높은 정확도를 보였으며, 전반적인 탐지 정확도를 향상시키는 데 기여했습니다. 이는 잘못된 음성(false negatives)을 줄이는 동시에, 실제로 주입된 프롬프트를 간과할 위험을 최소화합니다. 이 접근 방법은 LLM의 취약점(vulnerabilities)을 강조하며, 향후 연구를 위한 포괄적인 프레임워크를 제공합니다.



### Fast Calibrated Explanations: Efficient and Uncertainty-Aware Explanations for Machine Learning Models (https://arxiv.org/abs/2410.21129)
Comments:
          36 pages, 5 figures, journal submission

- **What's New**: 이 논문에서는 Fast Calibrated Explanations라는 새로운 방법을 소개합니다. 이 방법은 머신러닝 모델에 대한 신속하고 불확실성을 고려한 설명을 생성하기 위해 설계되었습니다. ConformaSight의 섭동 방식을 핵심 요소인 Calibrated Explanations에 통합하여 성능을 크게 향상시켰습니다.

- **Technical Details**: Fast Calibrated Explanations는 로컬 특징 중요도(local feature importance)와 교정된 예측(calibrated predictions)을 포함하며, 이러한 요소는 불확실성 정량화를 유지합니다. 이 방법은 분류(classification)와 임계 회귀(thresholded regression) 작업에서 목표가 사용자 정의한 임계값 위 또는 아래일 확률을 제공합니다.

- **Performance Highlights**: Fast Calibrated Explanations는 실시간 의사결정 과정에서 매우 유리하며, 긴급 대응 또는 자동 모니터링과 같이 짧은 시간 내에 중요한 결정을 요구하는 시나리오에서 신속한 설명 생성을 보장하여 시스템이 원활하게 작동할 수 있도록 합니다.



### Retrieval-Enhanced Mutation Mastery: Augmenting Zero-Shot Prediction of Protein Language Mod (https://arxiv.org/abs/2410.21127)
Comments:
          25 pages, 10 figures, 8 tables

- **What's New**: 이 논문에서는 깊이 있는 학습(deep learning) 기술을 활용하여 단백질 변이 효과 예측을 다루는 새로운 프로틴 언어 모델(ProtREM)을 소개합니다. ProtREM은 단백질 서열(sequence)과 구조(structure) 정보, 그리고 진화적 자료(evolutionary information)를 통합하여 자연적인 특성을 분석하는 데 중점을 두고 있습니다.

- **Technical Details**: ProtREM은 시퀀스-구조 임베딩(sequence-structure embeddings)을 다중 헤드 크로스-어텐션(multi-head cross-attention) 레이어를 통해 통합하여 모델을 훈련시킵니다. 또한, 동종 유전자(homologous sequences)를 활용하여 진화적 표현(evolutionary representations)을 도출하여 변이의 적합도를 평가합니다. 이 모델은 217가지 어세이(assays)에서 2222만 개 이상의 돌연변이에 대한 데이터를 처리하여 성능을 검증하였습니다.

- **Performance Highlights**: ProtREM은 공개 벤치마크인 ProteinGym의 데이터를 활용하여 다양한 단백질 및 어세이에 대한 예측에서 우수한 성능을 보였습니다. 일반적으로 다른 기법들과 비교하여 안정성과 적합도 예측 성능 모두에서 뛰어난 결과를 달성하였으며, 특히 VHH 항체와 phi29 DNA 중합효소에 대해 실험적으로도 확인된 바 있습니다. 이는 생물학자들이 효소 변형에 있어 더욱 신뢰할 수 있는 정보를 제공함을 의미합니다.



### Large Language Model-assisted Speech and Pointing Benefits Multiple 3D Object Selection in Virtual Reality (https://arxiv.org/abs/2410.21091)
Comments:
          under review

- **What's New**: 본 연구에서는 여러 개체가 포함된 가상 현실(VR) 환경에서 차폐된 객체 선택 문제를 해결하기 위해 대화식 멀티모달 상호작용 기법을 활용한 새로운 방법인 AssistVR을 제안합니다. 이 기술은 대화(voice)와 레이캐스트(raycast)를 혼합하여 지원하며, 비교 연구를 통해 기존의 미니맵 기반 차폐 객체 선택 기술보다 우수한 성과를 보여줍니다.

- **Technical Details**: AssistVR 기술은 대규모 언어 모델(large language models)을 기반으로 하여, 가상 현실 내에서 사용자 인터랙션을 향상시킵니다. 연구는 24명의 참가자를 대상으로 진행되었으며, 여기서 다양한 장면의 복잡도(perplexity)를 고려하여 목표 객체 선택 성능을 평가했습니다.

- **Performance Highlights**: AssistVR은 여러 개의 목표 객체가 있을 때 기존 미니맵 기술보다 더 나은 성능을 보였으며, 언어적 참조가 어려운 경우에도 높은 선택 정확도를 유지했습니다. 이는 IntenSelect 등 다른 최신 기술들과 비교했을 때 유의미한 결과로, 향후 지능형 멀티모달 시스템 설계에 중요한 시사점을 제공합니다.



### Efficient Mixture-of-Expert for Video-based Driver State and Physiological Multi-task Estimation in Conditional Autonomous Driving (https://arxiv.org/abs/2410.21086)
- **What's New**: 본 연구에서는 새로운 다중 작업 운전 중 모니터링 시스템(VDMoE)을 제안하여 SAE 레벨 2/3 자율 주행 환경에서 운전자의 인지 부하와 졸음을 효과적으로 평가할 수 있습니다.

- **Technical Details**: VDMoE는 RGB 비디오 입력을 활용하여 운전자의 상태를 비침습적으로 모니터링하며, 중요한 얼굴 특징을 이용해 계산 부담을 최소화하고, 원격 광용적맥법(rPPG) 기술을 통합하여 생리학적 정보를 제공합니다. Mixture-of-Experts (MoE) 프레임워크를 최적화하여 다중 모드 입력을 처리하고 다양한 작업에서 성능을 향상시킵니다. 또한, 통계적 사전과 모델 출력을 정렬하는 새로운 사전 포함 정규화 기법이 도입되었습니다.

- **Performance Highlights**: 42명의 참가자를 포함하는 새로운 데이터셋(MCDD)을 기반으로 한 검증 결과, VDMoE는 운전자의 상태 모니터링 효과가 입증되었습니다. 이 방법은 자율 주행 시스템의 안전성을 높이는 데 기여한다고 볼 수 있습니다.



### Stealthy Jailbreak Attacks on Large Language Models via Benign Data Mirroring (https://arxiv.org/abs/2410.21083)
- **What's New**: 이 논문은 안전한 LLM(대형 언어 모델)의 취약성을 탐구하기 위해 ShadowBreak라는 새로운 jailbreak 공격 방법을 제안합니다. 이 방법은 benign data mirroring 기법을 사용하여 공격의 은폐성을 높이면서도 공격 성공률을 유지합니다.

- **Technical Details**: ShadowBreak 방법은 benign 데이터에 대해 타겟 블랙박스 모델에 맞춰 '미러' 모델을 지역적으로 훈련하여 악의적인 프롬프트를 구성합니다. 이를 통해 정체가 드러나는 악의적인 명령을 실제 공격 과정에서 제출하지 않고도 쿼리를 생성할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 GPT-3.5 Turbo에 대해 평균 1.5개의 감지 가능한 쿼리로 최대 92%의 공격 성공률(Attack Success Rate, ASR)을 달성하였습니다. 이는 기존의 PAIR 방법보다 훨씬 적은 감지 가능한 쿼리로 더 나은 결과를 보여줍니다.



### Skip2-LoRA: A Lightweight On-device DNN Fine-tuning Method for Low-cost Edge Devices (https://arxiv.org/abs/2410.21073)
Comments:
          ASP-DAC 2025 (accepted)

- **What's New**: 본 논문에서는 경량의 파인튜닝(automatic tuning) 방법인 Skip2-LoRA를 제안합니다. 이 방법은 임베디드 시스템에서의 사전 훈련(pre-trained) 및 배포(deployed) 모델 간의 간극을 해소하기 위해 설계되었습니다.

- **Technical Details**: Skip2-LoRA는 LoRA(low-rank adaptation) 어댑터를 마지막 층과 모든 다른 층 사이에 삽입하여 네트워크의 표현력을 높이는 동시에, 역전파(backpropagation)의 계산 비용을 낮추는 새로운 아키텍처를 활용합니다. 이 아키텍처는 중간 계산 결과를 캐시(caching)할 수 있도록 해 주며, 학습 에폭(epoch)이 진행됨에 따라 이미 본 샘플에 대한 피드 포워드(forward pass) 계산을 생략할 수 있습니다.

- **Performance Highlights**: Skip2-LoRA는 동일한 수의 훈련 가능한 파라미터를 갖는 기존 방법과 비교할 때 평균적으로 90.0%의 파인튜닝 시간을 단축시키며, 정확도(accuracy)를 유지합니다. 또한, $15의 단일 보드 컴퓨터에서 몇 초 만에 완료할 수 있음을 보여줍니다.



### EMOCPD: Efficient Attention-based Models for Computational Protein Design Using Amino Acid Microenvironmen (https://arxiv.org/abs/2410.21069)
- **What's New**: 이번 연구에서는 기존의 비효율적인 기법을 극복하고 단백질 디자인의 정확성을 높이기 위해 아미노산 미세환경(aminо acid microenvironment)을 활용한 효율적인 주의 기반 모델(EMOCPD)을 개발하였습니다.

- **Technical Details**: EMOCPD는 3차원 원자 환경(three-dimensional atomic environment)을 분석하여 각 아미노산의 카테고리를 예측하고, 예측된 높은 확률의 아미노산 카테고리에 기반하여 단백질을 최적화합니다. 이 모델은 다중 주의 메커니즘(multi-head attention mechanism)을 사용하여 희소한 단백질 미세환경에서 중요한 특징을 강조하고, 역 잔여 구조(inverse residual structure)를 통해 네트워크 구조를 최적화합니다.

- **Performance Highlights**: EMOCPD는 훈련 세트(training set)에서 80% 이상의 정확도를 달성했으며, 두 개의 독립 테스트 세트(test sets)에서도 각각 68.33%와 62.32%의 정확도를 기록하며 기존 기법들보다 10% 이상 높은 성능을 보여주었습니다. 예측된 돌연변이(mutants)의 열적 안정성(thermal stability) 및 단백질 발현(protein expression)에서는 야생형(wild type)보다 유의미한 향상이 있었습니다.



### Kandinsky 3: Text-to-Image Synthesis for Multifunctional Generative Framework (https://arxiv.org/abs/2410.21061)
Comments:
          Accepted for EMNLP 2024 (Demo track)

- **What's New**: 이번 연구에서는 Kandinsky 3라는 새로운 Text-to-Image (T2I) 모델을 발표하며, 이는 Latent Diffusion을 기반으로 하여 높은 품질과 사진 같은(realism) 결과를 달성했습니다. 이 모델은 다양한 생성 작업에 쉽게 적응할 수 있는 단순함과 효율성을 특징으로 하며, 사용자가 직접 테스트할 수 있는 사용자 친화적인 시스템을 제공합니다.

- **Technical Details**: Kandinsky 3는 Text Encoder, U-Net 구조의 Noise 예측 네트워크, Image Decoder로 구성된 Latent Diffusion 모델입니다. 사용된 Text Encoder는 Flan-UL2 모델에 기반하며 8.6억 개의 파라미터를 가지고 있고, 이미지 디코더는 Sber-MoVQGAN에서 가져왔습니다. 전체 모델은 119억 개의 파라미터를 가지며, U-Net 훈련 동안 Text Encoder와 Image Decoder는 고정되었습니다. 이를 통해 모델의 훈련 성능을 향상시켰습니다.

- **Performance Highlights**: Kandinsky 3는 기존 모델보다 3배 빠른 속도의 추론을 가능하게 하며, 인간 평가에서 오픈 소스 생성 시스템 중 가장 높은 품질 점수를 기록했습니다. 텍스트 기반 인페인팅(outpainting), 이미지 퓨전(image fusion), 텍스트-이미지 퓨전 등을 지원하며, 사용자는 웹 플랫폼에서 다양한 기능을 시험해볼 수 있습니다.



### CTINEXUS: Leveraging Optimized LLM In-Context Learning for Constructing Cybersecurity Knowledge Graphs Under Data Scarcity (https://arxiv.org/abs/2410.21060)
Comments:
          under peer-review

- **What's New**: 신규 프레임워크 CTINexus는 대규모 언어 모델(LLMs)의 최적화된 컨텍스트 학습(in-context learning, ICL)을 활용하여 데이터 효율적인 사이버 위협 정보(CTI) 지식 추출 및 고품질 사이버 보안 지식 그래프(CSKG) 생성이 가능합니다.

- **Technical Details**: CTINexus는 (1) 최적화된 자동 프롬프트 구성 전략과 최적의 시연 데이터 검색을 통해 다양한 사이버 보안 개체 및 관계를 추출하고, (2) 계층적 개체 정렬 기법으로 추출된 지식을 표준화하고 중복을 제거하며, (3) ICL 강화된 장거리 관계 예측 기법을 통해 누락된 연결을 완성합니다.

- **Performance Highlights**: 150개의 실제 CTI 보고서를 사용한 평가에서 CTINexus는 사이버 보안 트리플 추출에서 87.65%, 계층적 개체 그룹화에서 89.94%, 세부 개체 병합에서 99.80%, 장거리 관계 예측에서 90.99%의 F1 점수를 달성하며, 기존 메서드들보다 현저하게 우수한 성능을 나타냈습니다.



### Getting By Goal Misgeneralization With a Little Help From a Mentor (https://arxiv.org/abs/2410.21052)
Comments:
          SATA Workshop @ NeurIPS 2024 (Towards Safe and Trustworthy Agents)

- **What's New**: 이 연구는 강화학습(즉, RL) 에이전트가 현업에서 배포될 때 경험하는 목표 비일반화(goal misgeneralization) 문제를 해결하기 위해 감독자에게 도움을 요청하는 접근 방식을 탐구합니다. 특히 CoinRun 환경에서 PPO를 사용하여 훈련된 에이전트를 대상으로 심화 연구를 진행하였습니다.

- **Technical Details**: 실험에서는 OpenAI의 procgen 패키지와 이를 기반으로 한 procgenAISC 패키지를 사용하였습니다. 목표 비일반화를 유발하는 coinrun 및 coinrun_aisc 환경에서 에이전트를 훈련시키고, 에이전트의 행동 분포에서 불확실성을 측정하여 언제 도움을 요청해야 하는지 판단하는 여러 방법을 도입하였습니다. 다섯 가지 방법(최대 확률, 최대 로짓, 샘플된 확률, 샘플된 로짓, 엔트로피)을 테스트하였습니다.

- **Performance Highlights**: 방법에 관계없이, 행동 분포의 불확실성을 기반으로 도움을 요청하는 방식은 도움 요청 없이 행동하는 경우보다 항상 더 높은 성능을 보였습니다. 특히, 에이전트는 훈련 중 동전을 찾는 위치 대신 동전이 없는 다른 위치에 도달했을 때 도움을 요청하는 경향이 있음을 발견하였습니다.



### Disentangled and Self-Explainable Node Representation Learning (https://arxiv.org/abs/2410.21043)
- **What's New**: 최근에 소개된 DiSeNE (Disentangled and Self-Explainable Node Embedding) 프레임워크는 비지도 학습 방식으로 자기 설명 가능한 노드 임베딩을 생성하는 방법론을 제시합니다. 이 방법은 각 차원이 그래프의 독립적인 위상 구조와 연관되도록 분리된 표현 학습 (disentangled representation learning)을 활용하여 임베딩을 제공합니다.

- **Technical Details**: 이 프레임워크는 각 차원이 그래프의 특정 서브 구조를 예측하도록 임베딩을 최적화하는 새로운 목적 함수를 도입하여 구조적 분리를 보장합니다. Entropy 기반 정규화 기법을 통해 중복 방지 및 비어있지 않은 매핑된 서브 구조의 형성을 보장합니다.

- **Performance Highlights**: DiSeNE는 여러 벤치마크 데이터셋에서 광범위한 실험을 통해 기존 방식보다 더 나은 성능을 입증하였으며, 새로운 평가 지표를 통해 자기 설명 가능한 노드 임베딩의 질을 측정합니다.



### FairStream: Fair Multimedia Streaming Benchmark for Reinforcement Learning Agents (https://arxiv.org/abs/2410.21029)
- **What's New**: 본 연구에서는 공정한 멀티미디어 스트리밍을 위한 복합적인 현 환경을 다루는 새로운 다중 에이전트(Multi-Agent) 환경을 제안합니다. 특히 부분 가시성(partial observability), 대안적 목표(multiple objectives), 에이전트 이질성(agent heterogeneity) 및 비동기성(asynchronicity)과 같은 도전을 포함하고 있습니다.

- **Technical Details**: 우리는 다수의 트래픽 클래스에 걸쳐 5가지 다른 기본 접근 방식을 제공하고 분석하여 에이전트간 행동을 탐구합니다. 실험에서 사용된 방법은 Dynamic Adaptive Streaming over HTTP (DASH) 모델을 기반으로 하며, 각 에이전트는 자신의 품질을 극대화하고 모든 클라이언트 간의 공정성을 고려하도록 설계되었습니다. 또한 Proximal Policy Optimization (PPO) 알고리즘보다 단순한 그리디 휴리스틱(greedy heuristic)이 더 우수하다는 것을 보여줍니다.

- **Performance Highlights**: 다양한 네트워크 조건 하에서 단일 에이전트 및 다중 에이전트 RL 알고리즘의 성능을 평가했습니다. 이 연구는 에이전트 간의 상호작용과 비동기적인 결정을 가진 환경을 탐구하며, 공개된 플랫폼은 연구자들이 원하던 품질 지표를 통합할 수 있도록 지원합니다.



### Graph Based Traffic Analysis and Delay Prediction (https://arxiv.org/abs/2410.21028)
- **What's New**: 이 연구는 EU에서 가장 인구 밀도가 높은 나라인 몰타의 교통 혼잡 문제를 해결하기 위한 새로운 데이터 세트인 MalTra를 소개합니다. MalTra에는 200일 간의 실제 공공 이동 데이터가 포함되어 있으며, 다양한 방법론을 통해 데이터 수집의 정확성을 극대화하였습니다.

- **Technical Details**: 본 연구에서는 ARIMA 모델과 두 가지 그래프 신경망 모델인 STGCN(Spatial Temporal Graph Convolutional Network)과 DCRNN(Diffusion Convolutional Recurrent Network)을 사용하여 교통 데이터 분석을 수행했습니다. 데이터 세트의 품질을 높이기 위해 기존의 Q-Traffic 데이터 세트와 MalTra 데이터를 함께 활용하여 모델의 예측 성능을 비교하였습니다.

- **Performance Highlights**: DCRNN 모델은 MAE(Mean Absolute Error) 3.98 및 RMSE(Root Mean Squared Error) 7.78을 기록하며 STGCN 모델(각각 6.65, 12.73)보다 우수한 성과를 나타냈습니다. 이는 몰타의 트래픽 예측을 위한 그래프 기반 신경망의 유효성을 입증합니다.



### Informed Deep Abstaining Classifier: Investigating noise-robust training for diagnostic decision support systems (https://arxiv.org/abs/2410.21014)
Comments:
          This preprint has no post-submission improvements or corrections. The Version of Record of this contribution is published in the Neural Information Processing, ICONIP 2024 Proceedings

- **What's New**: 이 연구에서는 Deep Abstaining Classifier (DAC) 손실 함수를 확장하여 Informed Deep Abstaining Classifier (IDAC) 손실 함수를 도입하였습니다. 이 방법은 훈련 중 노이즈 추정치를 포함하도록 설계되었으며, 노이즈에 대한 강건성을 향상시키는데 중점을 두었습니다.

- **Technical Details**: IDAC는 노이즈 레벨 추정치를 손실 함수에 통합하여 훈련 중 노이즈를 보다 효과적으로 처리할 수 있게 합니다. 이를 통해 DAC와 여러 최신 손실 함수에 비해 노이즈에 대한 강건성이 증대되었으며, 성능 향상도 보여주었습니다. 연구는 공개된 흉부 X-레이 데이터 세트를 사용하여 다양한 시뮬레이션된 노이즈 레벨에서 검증되었습니다.

- **Performance Highlights**: IDAC는 흉부 X-레이 데이터 세트를 기반으로 하여 1%에서 15% 사이의 낮은 노이즈 레벨에서 성능 테스트를 수행했으며, 자체적으로 수집한 노이즈가 포함된 데이터 세트에서도 그 효과가 입증되었습니다. 이러한 결과는 임상 데이터에서의 DDSS 개발에 있어 IDAC의 큰 잠재력을 보여줍니다.



### FACT: Examining the Effectiveness of Iterative Context Rewriting for Multi-fact Retrieva (https://arxiv.org/abs/2410.21012)
Comments:
          Work in Progress

- **What's New**: 본 논문에서는 LLMs(대형 언어 모델)가 여러 사실을 동시에 검색하는 데 어려움을 겪는 "lost-in-the-middle" 현상을 새롭게 규명하고, 이를 해결하기 위한 새로운 접근법인 FACT(모든 중요한 텍스트 찾기)를 제안합니다.

- **Technical Details**: FACT는 반복적(iterative) 검색 방법으로, 문맥을 여러 차례 재작성하여 정제(refine)하는 방식을 사용합니다. 이 방법은 단일 검색에서는 종종 간과되는 기본 사실들을 점진적으로 포착(capture)할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, FACT는 다양한 과제에 걸쳐 다중 사실(multi-fact) 검색 성능을 크게 향상시켰으나, 일반적인 QA(질문-답변) 시나리오에서는 개선 효과가 덜 두드러졌습니다. 이 연구는 LLMs의 다중 사실 검색에서의 한계를 조명하고 있으며, 보다 회복력이 강한 긴 문맥 검색 전략의 필요성을 강조합니다.



### Efficient Bilinear Attention-based Fusion for Medical Visual Question Answering (https://arxiv.org/abs/2410.21000)
- **What's New**: 이 논문에서는 Orthogonality loss, Multi-head attention, Bilinear Attention Network을 통합한 새로운 Fusion 모델 OMniBAN을 제안합니다. 이 모델은 사전 학습 없이도 높은 계산 효율성과 강력한 성능을 달성하도록 설계되었습니다.

- **Technical Details**: OMniBAN은 이미지와 텍스트의 복잡한 상호작용을 캡처하는 Bilinear Attention을 활용하며, Multi-head attention을 통해 다양한 시각적 정보를 처리합니다. 이를 통해 모델은 의료 영상 질문 응답(MedVQA)에서의 성능을 극대화할 수 있습니다.

- **Performance Highlights**: OMniBAN은 여러 MedVQA 벤치마크에서 기존 모델보다 뛰어난 성능을 발휘하며, 낮은 계산 비용을 유지합니다. 이는 자원 제약이 있는 임상 환경에서의 효율적 적용 가능성을 보여줍니다.



### EEG-Driven 3D Object Reconstruction with Color Consistency and Diffusion Prior (https://arxiv.org/abs/2410.20981)
- **What's New**: EEG(뇌전도) 기반의 색상 일관성을 가진 3D 객체 복원 방법을 제안하며, EEG 신호를 통해 3D 객체의 지역적 의미를 캡처하는 임피시트(implicit) 신경 인코더를 훈련합니다.

- **Technical Details**: 이 연구는 두 단계로 구성된 접근 방식을 사용합니다. 첫 번째 단계에서는 3D 객체를 인식할 수 있는 임피시트 신경 EEG 인코더를 훈련시키고, 두 번째 단계에서는 첫 번째 단계에서 획득한 잠재 EEG 코드(latent EEG codes)를 바탕으로 확산 모형(diffusion model), 신경 스타일 손실(neural style loss), 그리고 NeRF를 통합하여 3D 객체를 암묵적으로 복원합니다.

- **Performance Highlights**: 실험 검증을 통해 이 방법이 EEG 신호를 사용하여 색상 일관성을 가진 3D 객체를 복원할 수 있음을 입증하였습니다.



### Geo-FuB: A Method for Constructing an Operator-Function Knowledge Base for Geospatial Code Generation Tasks Using Large Language Models (https://arxiv.org/abs/2410.20975)
- **What's New**: 지리공간 데이터(spatiotemporal data)의 증가와 효율적인 지리적 모델링의 필요성으로 인해 대규모 언어 모델(LLMs)을 활용한 자동화에 대한 관심이 높아지고 있습니다. 이러한 맥락에서, 도메인 전문 지식이 부족하여 일반 LLM이 지리공간 코드에서 오류를 발생시키는 문제를 해결하기 위해 외부 지식 기반을 활용한 RAG(retrieval-augmented generation) 접근법이 제안되었습니다.

- **Technical Details**: 본 연구는 지리적 스크립트 의미론을 활용하여 지리공간 함수(function) 및 연산자(operator)의 외부 지식 기반을 구축하기 위한 프레임워크를 소개합니다. 프레임워크는 Function Semantic Framework Construction (Geo-FuSE), Frequent Operator Combination Statistics (Geo-FuST), 그리고 Semantic Mapping (Geo-FuM)으로 구성됩니다. Chain-of-Thought, TF-IDF, 그리고 APRIORI 알고리즘을 이용하여 지리공간 함수를 도출하고 정렬하는 기술이 사용됩니다.

- **Performance Highlights**: 154,075개의 Google Earth Engine 스크립스에서 구축된 예제 지식 기반 Geo-FuB는 GitHub에서 제공되며, 평가 메트릭스에 따르면 전체 정확도가 88.89%에 달하며, 구조적 정확도는 92.03%, 의미적 정확도는 86.79%로 나타났습니다. Geo-FuB는 RAG 및 파인튜닝(fine-tuning) 패러다임을 통해 지리공간 코드 생성을 최적화할 수 있는 잠재력을 가지고 있습니다.



### BlueSuffix: Reinforced Blue Teaming for Vision-Language Models Against Jailbreak Attacks (https://arxiv.org/abs/2410.20971)
- **What's New**: 이번 연구에서는 비전-언어 모델(VLMs)에 대한 새로운 방어 방법을 제안합니다. 특히, 블랙박스 방어를 통해 jailbreak 공격에 대응하는 BlueSuffix라는 방법이 소개됩니다. 이 방법은 VLM의 성능을 저해하지 않으면서도 높은 안전성을 제공합니다.

- **Technical Details**: BlueSuffix는 세 가지 주요 구성 요소로 구성되어 있습니다: 1) jailbreak 이미지를 차단하는 시각적 정제기(visual purifier), 2) 악의적인 텍스트를 처리하는 텍스트 정제기(textual purifier), 3) 강화 학습을 통해 미세 조정된 블루 팀 접미사 생성기(blue-team suffix generator)로, 크로스 모달 강인성을 강화합니다. 모든 구성 요소는 블랙박스 환경에서 VLM의 취약점을 완화하기 위해 설계되었습니다.

- **Performance Highlights**: BlueSuffix는 LLaVA, MiniGPT-4 및 Gemini와 같은 세 가지 VLM 모델에서 테스트된 결과, 기존 방어 방법보다 공격 성공률(ASR)을 약 70%까지 감소시키는 성과를 보였습니다. 이는 VLM jailbreak 공격에 대한 방어의 새로운 기준을 설정하며, 실제 사용 사례에서 높은 실용성을 제공합니다.



### Improving Detection of Person Class Using Dense Pooling (https://arxiv.org/abs/2410.20966)
- **What's New**: 본 연구에서는 Faster_RCNN을 기반으로 하여 밀집 풀링(dense pooling) 기술을 적용하여 사람 객체를 탐지하는 성능을 획기적으로 향상시키는 접근 방식을 제안합니다. 이 방법은 COCO 데이터셋에서 수집한 6982개의 이미지를 사용하여 기존 방법 대비 더 정확한 결과를 달성하였습니다.

- **Technical Details**: Faster_RCNN 모델의 ROI pooling 레이어를 최적화하고, 밀집 풀링(dense pooling)을 통해 2D 이미지를 3D 모델로 변환하여 UV(ultra Violet) 이미지로 만들었습니다. 이를 통해 이미지에서 올바른 특징을 추출하기 용이하게 하였으며, Resnet-50 RPN과 Resnet-101을 활용하여 실험을 진행했습니다.

- **Performance Highlights**: 제안된 모델은 Faster_RCNN을 사용할 때보다 사람 객체 탐지에서 더욱 높은 정확도를 보여줍니다. 실험 결과는 Yolo-v7과 비교했을 때도 최첨단 결과를 나타내었으며, 연구 결과는 오픈 소스로 제공되어 다른 연구자들이 활용할 수 있도록 하였습니다.



### DeTeCtive: Detecting AI-generated Text via Multi-Level Contrastive Learning (https://arxiv.org/abs/2410.20964)
Comments:
          To appear in NeurIPS 2024. Code is available at this https URL

- **What's New**: 새롭게 제안된 DeTeCtive 프레임워크는 저자 간의 다양한 문체를 구별하는 데 초점을 두어, AI 생성 텍스트 감지 문제를 새롭게 접근합니다. 이러한 접근법은 기존의 이진 분류 방식에서 벗어나, 작가의 고유한 문체를 학습하는 특성을 반영합니다.

- **Technical Details**: DeTeCtive는 다중 작업 보조 및 다중 수준 대비 학습(multi-level contrastive learning) 프레임워크를 사용하여 다양한 저자의 쓰기 스타일을 학습합니다. 이 방법은 텍스트 인코더와 결합되어 AI 생성 텍스트 감지 기능을 강화하며, K-최근접 이웃(KNN) 알고리즘을 통한 유사도 측정을 사용합니다. 또한, Training-Free Incremental Adaptation (TFIA) 기능을 통해 OOD 데이터에 대한 모델의 적응력을 향상시킵니다.

- **Performance Highlights**: 제안된 방법은 여러 데이터셋에서 기존 방법보다 우수한 성능을 보이며, 특히 OOD 데이터에 대한 제로샷 평가에서 기존 방법을 크게 능가하는 결과를 기록했습니다. AvgRec 메트릭에서 Unseen Models과 Unseen Domains 테스트 세트에서 각각 5.58% 및 14.20%의 성능 향상이 확인되었습니다.



### Instruction-Tuned LLMs Succeed in Document-Level MT Without Fine-Tuning -- But BLEU Turns a Blind Ey (https://arxiv.org/abs/2410.20941)
- **What's New**: 본 연구에서는 instruction-tuned LLMs의 문서 수준 번역(docMT) 능력을 조사한 논문입니다. LLM이 전통적인 방법과 달리 전체 문서를 한 번에 번역할 수 있는지를 평가하고, BLEU 점수 대신 GPT-4를 활용한 새로운 평가 방식을 제안합니다.

- **Technical Details**: 연구에서는 두 가지 번역 접근법을 비교합니다: ST[k]는 여러 문장을 조합하여 번역하는 방법이며, DOC는 LLM에게 전체 문서를 직접 번역하도록 지시하는 방법입니다. 기존의 BLEU 점수는 문서 수준 번역의 품질을 제대로 반영하지 못하기 때문에, LLM을 평가자로 활용하여 일관성, 정확성 및 유창성을 평가합니다.

- **Performance Highlights**: 결과적으로, DOC 접근법이 개별 문장을 번역한 후 조합하는 방법보다 문서 맥락을 잘 활용하여 더 높은 번역 품질을 보여주었습니다. 그러나 BLEU 점수는 이 결과를 잘 반영하지 못하며, 오히려 문서 번역의 실제 품질을 오도할 수 있음을 강조합니다.



### Hacking Back the AI-Hacker: Prompt Injection as a Defense Against LLM-driven Cyberattacks (https://arxiv.org/abs/2410.20911)
Comments:
          v0.1

- **What's New**: Mantis는 LLM(대형 언어 모델)에 의해 구동되는 사이버 공격을 방어하기 위해 설계된 새로운 방어 전략을 제시합니다. 이 프레임워크는 LLM의 취약성을 이용해 공격자의 작동을 방해하는 방식으로 작동합니다.

- **Technical Details**: Mantis는 자동 사이버 공격을 탐지한 후, 시스템 응답에 세심하게 구성된 입력을 주입하여 공격자의 LLM이 자신의 작동을 방해하도록 유도합니다. 이를 위해 Mantis는 전략적으로 설계된 유인 서비스를 배치하고, 공격자의 LLM에 동적인 prompt injections를 수행합니다.

- **Performance Highlights**: 실험 결과, Mantis는 자동화된 LLM 기반 공격에 대해 95% 이상의 효과를 지속적으로 달성했습니다. 또한, Mantis는 오픈 소스 도구로 제공되어 추가 연구와 협업을 촉진하려는 목표를 가지고 있습니다.



### Diff-Instruct*: Towards Human-Preferred One-step Text-to-image Generative Models (https://arxiv.org/abs/2410.20898)
- **What's New**: 이번 논문에서는 Diff-Instruct*(DI*)라는 데이터가 필요 없는 접근 방식을 소개합니다. 이 방법은 텍스트에서 이미지로의 한 단계 생성 모델을 구축하고, 인간의 선호에 맞추어 고도로 사실적인 이미지를 생성할 수 있는 능력을 유지합니다.

- **Technical Details**: DI*는 인간 피드백을 활용한 온라인 강화 학습(RLHF)으로 인간 선호 조정을 프레이밍합니다. 기존 RLHF 접근과는 달리, KL divergence 대신 새로운 score-based divergence 정규화를 도입하여 훨씬 우수한 성능을 보입니다. 이러한 분산을 직접 계산하는 것은 어려우나, 우리는 다루기 쉬운 손실 함수를 유도하여 효율적으로 기울기를 계산할 수 있음을 보여줍니다.

- **Performance Highlights**: Stable Diffusion V1.5를 참조 diffusion 모델로 사용한 DI*는 모든 기존 모델보다 큰 차이로 우수한 성능을 기록했습니다. 0.6B PixelArt-α 모델을 사용해 Aesthetic Score 6.30과 Image Reward 1.31을 달성하여 나머지 유사 모델의 점수를 거의 두 배로 초과했습니다. 또한 HPSv2 점수 28.70을 기록하여 새로운 최첨단 벤치마크를 설정했습니다.



### Strada-LLM: Graph LLM for traffic prediction (https://arxiv.org/abs/2410.20856)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 이 연구는 Strada-LLM이라는 새로운 그래프 인지 (graph-aware) LLM을 제안하여 교통 예측 문제를 해결하고, 이와 관련된 여러 기술적 문제를 다루고 있습니다.

- **Technical Details**: Strada-LLM은 이웃 노드의 교통 정보를 공변량 (covariates)으로 고려하여 교통 예측을 수행합니다. 또한, 이 모델은 적은 레이블 데이터 샘플로 새로운 데이터 배포 (data distribution)에서 효과적으로 도메인 적응을 이루는 경량 (lightweight) 접근 방식을 채택합니다. 전통적인 GNN 기반의 방법과 비교하여 우수한 성능을 보여줍니다.

- **Performance Highlights**: Strada-LLM은 다양한 실제 교통 데이터 세트를 기반으로 평가한 결과, 예측 오류가 약 5%에서 18%까지 감소했습니다. 이는 그래프 인지 LLM 모델이 전통적인 방법들에 비해 우수한 교통 예측 성능을 보이도록 함을 입증합니다.



### Deep Insights into Automated Optimization with Large Language Models and Evolutionary Algorithms (https://arxiv.org/abs/2410.20848)
- **What's New**: 본 논문은 Large Language Models (LLMs)와 Evolutionary Algorithms (EAs)의 통합을 통한 자동화된 최적화 접근법을 제안합니다. LLM이 최적화 전략을 생성하고 해석할 수 있는 능력을 활용하여, EAs가 복잡한 솔루션 공간을 효율적으로 탐색하도록 합니다.

- **Technical Details**: LLMs는 동적 에이전트(dynamic agents) 역할을 하여 최적화 전략을 생성하고, EAs는 선택(selection), 돌연변이(mutation), 교차(crossover)와 같은 진화 연산자를 통해 복잡한 솔루션 공간을 탐색합니다. 이 연구에서는 LLMs의 해법 생성기(generator)와 알고리즘 설계자(designer)로서 기능을 강조합니다.

- **Performance Highlights**: 논문은 LLM과 EA의 통합을 통해 자동화된 최적화에 대한 새로운 패러다임을 제안하며, 이 패러다임이 더 효율적이고 적응 가능한 최적화 프로세스를 가능하게 함을 강조합니다. 또한, 개인 표현(individual representation), 변동 연산자(variation operators), 적합성 평가(fitness evaluation)와 같은 세 가지 주요 구성 요소에 대한 혁신적인 방법을 분석합니다.



### Generative Simulations of The Solar Corona Evolution With Denoising Diffusion : Proof of Concep (https://arxiv.org/abs/2410.20843)
- **What's New**: 이번 연구는 Denoising Diffusion Probabilistic Model (DDPM)을 사용하여 태양 코로나의 동역학을 시뮬레이션하고, 이를 통해 극자외선(EUV) 파장에서의 태양 활동 예측을 향상시키는 방법을 제시합니다. 모델은 12시간 분량의 Active Region (AR) 비디오를 입력받아 향후 12시간 동안의 예측을 수행하며, 시뮬레이션의 신뢰도와 정확도를 높입니다.

- **Technical Details**: 제안된 모델은 기존의 물리적 모델링 방식을 넘어서, light UNet 구조를 활용하여 1D temporal convolutions(시간적 합성곱)과 spatio-temporal attention(공간-시간 주의 메커니즘)을 통합하여 태양 코로나의 복잡한 역학을 효과적으로 모델링합니다. 이 과정에서 DDPM의 확률적 생성 프로세스를 이용하여 시스템의 내재적인 무작위성을 포착합니다.

- **Performance Highlights**: 시뮬레이션 결과는 EUV peak flux 및 fluence와 같은 주요 예측 지표에 대한 신뢰할 수 있는 신뢰 구간을 생성하며, 이를 통해 태양 활동 예측의 해석 가능성과 확률적 방법론을 제공합니다. 향후 연구는 공간적 및 시간적 해상도를 증가시켜 예측의 불확실성을 줄이는 데 중점을 두고 있습니다.



### ADLM -- stega: A Universal Adaptive Token Selection Algorithm for Improving Steganographic Text Quality via Information Entropy (https://arxiv.org/abs/2410.20825)
- **What's New**: 이 논문에서는 정보 엔트로피 제약에 기반한 스테가노그래픽(text steganography) 텍스트 생성 품질 관리 이론을 제안하며, 스테가노그래픽 텍스트의 인지 불가능성(imperceptibility)과 정보 엔트로피 간의 관계를 탐구합니다.

- **Technical Details**: 후보 단어 풀의 정보 엔트로피를 특정 범위 내로 조정하여 스테가노그래픽 텍스트의 인지 불가능성을 최적화합니다. 이 과정에서 정보 엔트로피의 상한과 하한을 설정하고, 의미적 일관성(semantic coherence)과 어휘적 다양성(lexical diversity)을 균형 있게 유지하기 위한 적응형 절단 기법(adaptive truncation method)을 도입했습니다.

- **Performance Highlights**: 실험 결과, 후보 풀의 크기와 정보 엔트로피 임계값을 적절하게 조절함으로써 스테가노그래픽 텍스트의 품질과 탐지 저항성이 크게 향상되었습니다. 이 모델은 기존의 텍스트 기반 생성 방법보다 뛰어난 성능을 보이며, 자연어 처리(natural language processing) 분야에서의 광범위한 응용 가능성을 보여줍니다.



### Bridging the Gap between Expert and Language Models: Concept-guided Chess Commentary Generation and Evaluation (https://arxiv.org/abs/2410.20811)
- **What's New**: 이 연구에서는 체스 전문가 모델과 대규모 언어 모델(LLMs) 간의 간극을 메우기 위한 새로운 접근 방식인 Concept-guided Chess Commentary generation (CCC)와 GPT-based Chess Commentary Evaluation (GCC-Eval)을 소개합니다. CCC는 전문가 모델의 결정 능력과 LLM의 언어적 유창성을 통합하여 체스 결정 과정을 설명하는 데 중점을 둡니다.

- **Technical Details**: CCC는 전문가 모델이 집중하는 개념을 추출하고 우선순위를 부여하여 LLM이 게임의 가장 중요한 측면에 집중하게 합니다. GCC-Eval는 체스 전문 지식을 활용하여 생성된 해설의 정보성 (informativeness)과 언어 품질 (linguistic quality)을 평가합니다.

- **Performance Highlights**: 실험 결과에 따르면 CCC는 인간 수준의 정확성을 달성하였으며, 인간이 생성한 코멘트보다 정보성과 언어적 품질에서 우수한 성능을 보여주었습니다. 또한, GCC-Eval는 사람의 평가와 잘 상관되어 더 신뢰할 수 있는 평가 지표로 기능하고 있습니다.



### From Cool Demos to Production-Ready FMware: Core Challenges and a Technology Roadmap (https://arxiv.org/abs/2410.20791)
- **What's New**: 본 논문은 FMware(Foundation Model Software)의 생산 준비성(production readiness)을 위한 주요 장애물을 분석하고, 산업 경험 및 다양한 데이터 출처에서 통합된 자료를 바탕으로 이 과정을 설명합니다. 특히 다이내믹한 시스템으로서의 FMware의 복잡성에 대한 이해를 바탕으로 그 해결 방안을 제시합니다.

- **Technical Details**: FMware는 Large Language Models(LLMs)와 같은 Foundation Models(FMs)을 중심으로 구성된 소프트웨어 시스템으로, 이를 상용화하기 위해서는 프롬프트 엔지니어링(prompt engineering), 에이전트 오케스트레이션(agent orchestration), 시스템 테스트(system testing), 배포(deployment) 등 다양한 요소에서의 도전이 필요합니다. 또한, 메모리 관리(memory management), 관찰 가능성(observability), 피드백 통합(feedback integration) 등 여러 교차적 문제도 고려되어야 합니다.

- **Performance Highlights**: FMware의 생산 준비성을 평가하기 위한 연구에서, 430명의 기술 전문가 설문 결과 10%의 조직만이 FMware를 생산 환경에 도입하고 있다고 합니다. LinkedIn의 경우, 초기 80%의 기능을 한 달 만에 구현했으나 나머지 20%를 완성하는 데는 4개월이 더 걸렸으며, 비용 문제와 복잡성으로 인해 테스트가 매우 비쌌습니다. 이와 같은 사례들은 FMware의 신뢰할 수 있는 생산 준비성을 위한 강력한 시스템 기반 접근법의 필요성을 강조합니다.



### Graph-based Uncertainty Metrics for Long-form Language Model Outputs (https://arxiv.org/abs/2410.20783)
Comments:
          Accepted as a Spotlight paper at NeurIPS 2024

- **What's New**: 이번 연구에서는 Large Language Models (LLMs)의 텍스트 생성 능력을 개선하기 위해 새로운 개념인 Graph Uncertainty를 도입합니다. 이 방법은 LLM의 생성물과 그 안의 주장 간의 관계를 이분 그래프로 표현하며, 주장 수준의 불확실성을 추정할 수 있게 해줍니다. 이로써 현재의 불확실성 추정 기법보다 더 정교한 접근법을 제공하며, 사실성을 높이는 데 기여합니다.

- **Technical Details**: Graph Uncertainty는 LLM의 생성 결과와 그 안에 포함된 주장 간의 의미적 관계를 이분 그래프로 표현하고, 다양한 그래프 중심성 지표(family of graph centrality metrics)를 사용하여 주장 수준의 불확실성을 추정합니다. 기존의 self-consistency 기반의 불확실성 측정은 degree centrality를 사용하고, 우리의 연구를 통해 closeness centrality가 더 정확한 불확실성 추정치를 제공함을 증명하였습니다. 또한, 불확실성에 민감한 디코딩 기법을 통해 신뢰할 수 있는 주장을 보존하며 LLM 생성물의 사실성을 개선합니다.

- **Performance Highlights**: 그래프 기반 불확실성 지표는 다양한 장황한 생성 설정에서 AUPRC에 평균 6.8%의 향상을 보였고, 우리의 시스템은 기존 디코딩 기법에 비해 2-4%의 사실성 향상을 지속적으로 이루어내며 생성된 응답의 정보성을 크게 개선했습니다.



### KD-LoRA: A Hybrid Approach to Efficient Fine-Tuning with LoRA and Knowledge Distillation (https://arxiv.org/abs/2410.20777)
Comments:
          Accepted at 4th NeurIPS Efficient Natural Language and Speech Processing Workshop (ENLSP-IV 2024)

- **What's New**: 본 연구에서는 KD-LoRA라는 새로운 미세 조정 기법을 제안하며, 이는 LoRA(저계수 적응)와 KD(지식 증류)를 결합하여 성능 손실을 최소화하면서 자원 요구 사항을 크게 줄이는 것을 목표로 합니다.

- **Technical Details**: KD-LoRA는 다음의 세 가지 주요 단계로 구성된 방법론입니다: (1) 교사 모델 선택 및 미세 조정, (2) LoRA 모듈로 초기화된 더 작은 학생 모델 설정, (3) 교사 모델에서 학생 모델로 지식을 전이하는 증류 수행.

- **Performance Highlights**: KD-LoRA는 GLUE 기준에서 LoRA의 성능을 98% 유지하면서도 GPU 메모리 사용량을 30% 줄이고, FFT보다 약 40% 더 작은 모델 크기를 달성했습니다. 또한, KD-LoRA는 추론 시간을 약 30% 단축시켰습니다.



### Introducing Spectral Attention for Long-Range Dependency in Time Series Forecasting (https://arxiv.org/abs/2410.20772)
Comments:
          Co-first Author: Bong Gyun Kang, Dongjun Lee

- **What's New**: 이 논문은 시간 시계열 예측(Time Series Forecasting, TSF)에서 긴 의존성(long-range dependencies)을 다루기 위한 새로운 방법인 Spectral Attention 기법을 제안합니다.

- **Technical Details**: Spectral Attention은 저역 통과 필터(low-pass filter)를 사용하여 긴 기간 트렌드(long-period trends)를 보존하고 모델의 기본 구조를 유지하면서 샘플 간의 시간적 상관관계를 보존합니다. 또한, Batched Spectral Attention은 여러 시점에서의 병렬 학습(parallel training)을 가능하게 합니다.

- **Performance Highlights**: 11개의 실제 데이터셋과 7개의 최신 예측 모델을 이용한 실험을 통해, Spectral Attention 메커니즘이 다른 모든 아키텍처에서 성능 향상을 이루어내고, 향상된 결과를 지속적으로 보여 주었습니다.



### MrT5: Dynamic Token Merging for Efficient Byte-level Language Models (https://arxiv.org/abs/2410.20771)
- **What's New**: MrT5 (MergeT5)는 ByT5의 구조적 비효율성을 해결하는 변형 모델입니다. 입력 시퀀스의 길이를 동적으로 단축시키기 위해 토큰 삭제 메커니즘을 통합했습니다.

- **Technical Details**: MrT5는 인코더의 입력을 단축시키기 위해 고정된 인코더 층에서 토큰 삭제 게이트를 사용합니다. 학습 과정에서 삭제 정규화 기법을 활용하여 삭제할 토큰과 유지할 토큰을 결합합니다. 이 방법을 통해 MrT5는 주어진 시퀀스의 정보를 더 압축하여 담을 수 있습니다.

- **Performance Highlights**: MrT5는 ByT5에 비해 추론 시간에서显著 개선을 보이며, 시퀀스 길이를 최대 80%까지 줄이는 동시에 XNLI 및 문자 수준의 작업에서 유사한 정확도를 달성합니다.



### A Static and Dynamic Attention Framework for Multi Turn Dialogue Generation (https://arxiv.org/abs/2410.20766)
Comments:
          published as a journal paper at ACM Transactions on Information Systems 2023. 30 pages, 6 figures

- **What's New**: 본 논문에서는 정적(static) 및 동적(dynamic) 주의(attention) 메커니즘을 기반으로 하는 다중 턴 대화 생성 모델을 제안합니다. 이 모델은 대화의 맥락을 보다 효과적으로 모델링하여 응답의 일관성과 다양성을 크게 향상시킵니다.

- **Technical Details**: 이 연구에서는 RNN(순환신경망) 기반의 적층 구조를 사용하되, 데이터의 맥락을 파악하는 데 있어 '소멸되는 기울기(vanishing gradient)' 문제를 해결하기 위해 정적 및 동적 주의 메커니즘을 도입합니다. 정적 주의는 전체 대화 정보에 초점을 맞추고, 동적 주의는 개별 발언의 세부 정보를 고려하여 보다 정교한 대화 흐름을 생성합니다.

- **Performance Highlights**: 실험 결과, Ubuntu 및 Opensubtitles 데이터셋에서 제안한 모델이 기존 모델들보다 자동 평가 및 인간 평가 지표에서 우수한 성능을 보이는 것으로 나타났습니다. 정적 및 동적 주의의 조합은 다양한 실험 설정에서도 뛰어난 결과를 보여주었습니다.



### ODRL: A Benchmark for Off-Dynamics Reinforcement Learning (https://arxiv.org/abs/2410.20750)
Comments:
          NeurIPS 2024 D&B Track

- **What's New**: 본 논문에서는 off-dynamics reinforcement learning (RL)에 대한 새로운 벤치마크인 ODRL을 소개합니다. 이 벤치마크는 다양한 도메인 간 정책 전이 및 동적 불일치 문제를 해결하기 위한 첫 번째 표준 평가 도구입니다.

- **Technical Details**: ODRL은 4개의 실험 설정을 포함하고 있으며, 각각의 소스 도메인과 타겟 도메인은 온라인 또는 오프라인 환경에서 작동할 수 있습니다. 이를 통해 다양한 작업과 광범위한 동적 변화에 적응할 수 있는 에이전트의 능력을 종합적으로 평가할 수 있습니다. 또한, ODRL은 최신 off-dynamics RL 알고리즘과 여러 기준선 메소드를 통합하여 단일 파일로 구현하여 공유하고 있습니다.

- **Performance Highlights**: 대규모 벤치마킹 실험을 통해 기존 방법들이 다양한 동적 변화 상황에서도 보편적인 장점을 인정받지 못한다는 점을 발견했습니다. ODRL은 미래 연구에서 중요한 초석이 될 것으로 기대하고 있습니다.



### Matryoshka: Learning to Drive Black-Box LLMs with LLMs (https://arxiv.org/abs/2410.20749)
Comments:
          Work in Progress

- **What's New**: 최근 대규모 블랙박스 언어 모델(LLM)의 기본적인 불투명성이 인지적 기능의 향상을 저해하고 있다는 문제 의식을 바탕으로, 새로운 컨트롤러 모델인 Matryoshika를 소개합니다. 이 모델은 복잡한 작업을 중간 출력으로 분해하여 블랙박스 LLM을 유도하는 경량의 화이트박스 LLM 컨트롤러입니다.

- **Technical Details**: Matryoshika는 두 가지 주요 구성 요소로 구성되며, 화이트박스 LLM이 컨트롤러로 작용하고 블랙박스 LLM이 생성기를 역할을 합니다. 입력 질문에 대해 컨트롤러는 중간 출력을 생성하고, 이는 블랙박스 LLM의 기능을 향상시키도록 설계되었습니다. Matryoshika는 정책을 기반으로 출력의 피드백을 통해 최적화를 수행하며, 단계적 상호작용을 통해 지속적인 자기 개선(self-improvement)을 가능하게 합니다.

- **Performance Highlights**: 세 가지 복잡한 작업에 대한 실험 결과, Matryoshika는 추론에서 평균 3.19%, 계획에서 성공률 7.46%, 개인화에서 정확도 5.82%의 향상을 보여주었습니다. 이러한 결과는 Matryoshika가 블랙박스 LLM의 성능을 효과적으로 개선하는 데 기여할 수 있음을 시사합니다.



### Shopping MMLU: A Massive Multi-Task Online Shopping Benchmark for Large Language Models (https://arxiv.org/abs/2410.20745)
Comments:
          NeurIPS 2024 Datasets and Benchmarks Track Accepted

- **What's New**: Shopping MMLU라는 다채로운 다중 작업 온라인 쇼핑 벤치마크가 제안되었으며, 이는 실제 Amazon 데이터를 기반으로 한 57개의 작업으로 구성되어 있습니다. 이 벤치마크는 LLMs의 일반 쇼핑 보조원으로서의 능력을 종합적으로 평가할 수 있습니다.

- **Technical Details**: Shopping MMLU는 개념 이해, 지식 추론, 사용자 행동 정렬 및 다국어 능력 등 4가지 주요 쇼핑 기술을 포함되어 있으며, LLM 기반 솔루션을 맞춤형으로 평가하기 위해 모든 작업을 텍스트-텍스트 생성으로 재구성하였습니다. 또한 20,799개의 질문이 포함되어 있습니다.

- **Performance Highlights**: 20개 이상의 기존 LLMs를 사용하여 Shopping MMLU에서 성능을 벤치마킹했으며, 이를 통해 도메인 특화 LLMs에 대한 통찰력을 발견하였습니다. 또한, KDD Cup 2024에서 500개 이상의 참여 팀이 있는 대회를 개최하여 이러한 연구 결과를 더 널리 확산할 계획입니다.



### Mitigating Unauthorized Speech Synthesis for Voice Protection (https://arxiv.org/abs/2410.20742)
Comments:
          Accepted to ACM CCS Workshop (LAMPS) 2024

- **What's New**: 최근 몇 년간 몇 가지 음성 샘플만으로도 화자의 목소리를 완벽하게 복제할 수 있게 되었습니다. 하지만 악의적인 음성 이용(예: 불법 재정 이득을 위한 전화 사기)으로 인한 위험이 증가하고 있습니다. 이에 따라 개인의 음성 지문과 같은 민감한 정보를 포함하는 공개 음성 데이터를 보호하는 것이 매우 중요해졌습니다. 본 연구에서는 이를 위해 Pivotal Objective Perturbation (POP)이라는 효과적이고 전이 가능한 방어 기술을 제안합니다.

- **Technical Details**: POP는 원래의 음성 샘플에 감지 불가능한 오류 최소화 노이즈를 적용하여 TTS(텍스트-음성 합성) 모델이 음성을 효과적으로 학습하지 못하게 만들어 고품질 딥페이크 음성이 생성될 수 없도록 합니다. POP 방법은 다양한 최신 TTS 모델에 대해 객관적 및 주관적 메트릭을 사용하여 포괄적으로 평가되었습니다. 실험 결과 POP로 보호된 샘플의 음성 불명확성 점수가 127.31%로 증가하여 보호되지 않은 샘플의 점수인 21.94%에 비해 크게 향상되었습니다.

- **Performance Highlights**: POP 방식은 노이즈 감소 및 데이터 증강 기술에 대해 뛰어난 강건성을 보여주며, 이를 통해 잠재적 위험을 크게 줄일 수 있음을 착실하게 입증하였습니다. 또한 POP는 다양한 모델에서 효과성과 전이성을 보장하며, 실제 상황에서의 적용 가능성을 확보하고 있습니다.



### Gender Bias in LLM-generated Interview Responses (https://arxiv.org/abs/2410.20739)
- **What's New**: 이 연구는 LLM(대형 언어 모델)이 생성한 면접 응답에서 성별 편향을 평가하며, GPT-3.5, GPT-4, Claude의 세 가지 모델을 분석합니다. 연구는 다양한 질문 유형과 직군에 대한 면접 응답의 질을 다각도로 감사하여 성별 고정관념과의 정렬을 강조합니다.

- **Technical Details**: 연구는 LIWC(언어적 탐색 및 단어 수) 도구를 사용하여 LLM들이 생성한 면접 응답의 언어적 및 심리적 특성을 분석했습니다. 실험에 사용된 LLM 모델은 기본 설정이 유지 되었으며, 70개의 인기 남녀 이름과 60개의 직업이 포함되었습니다. Mann-Whitney U test를 통해 남성과 여성에 대한 평균 LIWC 점수를 비교했습니다.

- **Performance Highlights**: LLM이 생성한 면접 응답에서 성별에 따른 언어적 및 심리적 특성이 뚜렷하게 구분되며, 남성 지원자에 대해 비해 여성이 표현하는 방식에 유의미한 차이가 나타났습니다. 성별 편향이 모델과 질문 유형에 걸쳐 일관되게 나타났고, 남성에게 편향된 응답이 더욱 강렬하게 나타나는 경향이 확인되었습니다.



### Murine AI excels at cats and cheese: Structural differences between human and mouse neurons and their implementation in generative AIs (https://arxiv.org/abs/2410.20735)
Comments:
          41 pages, 4 figures

- **What's New**: 이번 연구에서는 생쥐와 인간의 신경망(neuronal networks)이 어떻게 다른지, 그리고 이러한 차이가 뇌 기능에 미치는 영향을 분석했습니다. 특히 생쥐의 내측 전두엽 피질(medial prefrontal cortex)과 인간의 전방 대상 피질(anterior cingulate cortex)의 나노미터(nanometer) 규모의 3D 구조를 비교했습니다.

- **Technical Details**: 연구 결과에 따르면, 생쥐의 신경 세포(neuronal somata)는 더 작고, 신경가닥(neurites)은 인간 양보다 더 얇습니다. 이는 공간이 제한된 뇌 안에서 생쥐 신경 세포가 통합될 수 있도록 하며, 케이블 이론(cable theory)에 의하면 얇은 신경가닥이 원거리 연결(distal connections)을 억제할 가능성이 있습니다. 이러한 생쥐 유사(mouse-mimetic) 제약을 생성적 적대 신경망(GAN)과 디노이징 확산 암묵 모델(DDIM)의 합성곱층(convolutional layers)에 적용했습니다.

- **Performance Highlights**: 실험 결과, 생쥐 유사 GAN은 고양이 얼굴(cat faces)과 치즈 사진(cheese photo) 데이터셋을 사용한 이미지 생성 작업에서 표준 GAN보다 성능이 우수했으나, 인간 얼굴(human faces)과 새(birds) 데이터셋에서는 성능이 떨어졌습니다. 비슷한 결과가 생쥐 유사 DDIM에서도 나타났으며, 데이터셋의 특성이 결과에 영향을 미친 것으로 보입니다. 네 개의 데이터셋에 대한 분석에서 이미지 엔트로피(image entropy) 차이로 인해 이미지 생성에 필요한 매개변수(parameter) 수에 영향을 미쳤다는 것을 확인했습니다.



### SEG:Seeds-Enhanced Iterative Refinement Graph Neural Network for Entity Alignmen (https://arxiv.org/abs/2410.20733)
Comments:
          7, 2 figures

- **What's New**: 본 논문에서는 엔터티 정합성(entity alignment)의 중요성을 강조하며, 멀티-소스 데이터를 통합한 소프트 라벨 전파(soft label propagation) 프레임워크를 제안합니다. 이 프레임워크는 확장성과 효율성을 향상시키고, 비동형(非同形, non-isomorphic) 이웃 구조를 고려하여 적절한 엔터티 정합성을 제공합니다.

- **Technical Details**: 제안된 방법은 엔터티와 관계를 기반으로 한 이중 각도 모델링을 통해 최적의 관계 쌍을 융합하여 소프트 라벨을 생성합니다. 이 소프트 라벨은 이웃 특징과 의미 관계 데이터를 풍부하게 포함해 요약 정보 손실 문제를 해결합니다. 양방향 가중 합성 손실 함수(bidirectional weighted joint loss function)를 도입하여 긍정 샘플 간의 거리를 줄이고, 비슷한 부정 샘플에 대해 차별화된 처리를 합니다.

- **Performance Highlights**: 실제 데이터셋에서 검증된 결과, 본 방법은 기존의 반지도 학습(semi-supervised learning) 방식보다 성능이 우수하며, 엔터티 정합성의 품질을 크게 개선합니다. 이러한 결과는 제안된 방법이 지식 그래프 간의 엔터티 정합성에 매우 유용하고 실용적임을 시사합니다.



### GPRec: Bi-level User Modeling for Deep Recommenders (https://arxiv.org/abs/2410.20730)
- **What's New**: 본 논문에서는 GPRec이라는 이중 레벨 사용자 모델링 방법을 소개하고, 이 방법이 기존의 사용자 그룹 모델링과 개인화된 추천 방식을 통합하여 추천 시스템의 성능을 향상시키는 것을 목표로 합니다.

- **Technical Details**: GPRec은 사용자들을 학습 가능한 방식으로 그룹화하고 각 그룹에 대한 그룹 임베딩(group embedding)을 정렬합니다. 이 방법은 긍정적 및 부정적 패턴을 대조하여 집단 선호에 대한 다양성을 제시하는 이중 그룹 임베딩 공간을 설계합니다. 개인 차원에서는 ID와 유사한 특징으로부터 개인의 선호를 식별하고, 그룹 선호와 독립적인 개인 표현을 정제하여 그룹 레벨 모델링에 강력한 보완을 제공합니다.

- **Performance Highlights**: 세 개의 공공 데이터셋에서 GPRec을 rigorously 테스트한 결과 추천 품질에서 상당한 개선이 이루어졌으며, 다양한 DRS 구조에 유연하게 통합할 수 있는 여러 전략이 제시되었습니다.



### Lecture I: Governing the Algorithmic City (https://arxiv.org/abs/2410.20720)
- **What's New**: 이 논문은 현대 사회의 새로운 관계망인 'Algorithmic City'를 소개하고, 이 모델이 정치 철학에 미치는 영향을 탐구합니다. 컴퓨팅 기술이 사회 관계에 미친 영향에 대한 분석을 통해 현대 정치 철학의 기준을 재정립하고자 합니다.

- **Technical Details**: 'Algorithmic City'는 알고리즘에 의해 중재되는 사회적 관계의 네트워크를 나타내며, 이러한 관계를 통해 확인되는 중개 권력을 새롭게 정의합니다. 논문에서는 알고리즘 거버넌스가 권위의 정당화, 절차적 정당성의 기초, 정당화 중립성의 가능성에 대한 새로운 도전 과제를 제기한다는 점에 주목합니다.

- **Performance Highlights**: 이 논문은 정치 철학의 전통적인 모델이 현대 사회의 변화하는 조건을 어떻게 반영하지 못하는지를 지적하며, 알고리즘적 관계가 정치적 행동 및 의사 결정 방식에 미치는 영향을 분석합니다.



### Lecture II: Communicative Justice and the Distribution of Attention (https://arxiv.org/abs/2410.20718)
- **What's New**: 이 논문은 알고리즘 중개자(algorithmic intermediaries)가 디지털 공적 영역에서 소통을 어떻게 지배하는지에 대한 새로운 관점을 제시합니다. 또한, 디지털 공적 영역을 구성하는 의도적인 방법에 대한 이론이 필요하다고 주장합니다.

- **Technical Details**: 논문은 기존 정치 철학 이론이 표현의 자유 경계 설정에만 집중하고, 알고리즘 중개자들이 소통을 형성하고 주의를 분배하는 방법에 대한 이론적 고찰이 부족하다고 비판합니다. 또한, '소통의 정의(communicative justice)'라는 새로운 이론의 필요성을 강조합니다.

- **Performance Highlights**: 저자는 민주적 평등주의(democratic egalitarian) 이론을 통해 디지털 공적 영역에서 소통을 어떻게 공정하게 관리할 수 있는지에 대한 구체적인 방안을 제시합니다.



### Contextual Representation Anchor Network to Alleviate Selection Bias in Few-Shot Drug Discovery (https://arxiv.org/abs/2410.20711)
Comments:
          13 pages, 7 figures

- **What's New**: 본 연구는 few-shot 학습의 문제를 해결하기 위해 Contextual Representation Anchor Network (CRA)라는 새로운 방법론을 제안합니다. CRA는 labeled 데이터와 unlabeled 데이터 사이의 다리 역할을 하는 class-level contextual representation anchors를 활용하여 sample selection bias 문제를 해결하려고 합니다.

- **Technical Details**: CRA는 두 가지 증강 메커니즘을 도입합니다. 첫째는 context augmentation으로, 유사한 unlabeled 분자의 동적 검색과 task-specific contextual knowledge를 캡처하여 anchors를 증강합니다. 둘째는 anchor augmentation으로, anchors를 사용하여 분자 표현을 증강합니다. 이러한 접근법은 attention 메커니즘을 사용하여 unlabeled 데이터에서 정보의 정확한 전달을 보장합니다.

- **Performance Highlights**: CRA는 MoleculeNet 및 FS-Mol 벤치마크에서 평균 2.60% (AUC) 및 3.28% (ΔAUC-PR) 성능 향상을 보여주며, 뛰어난 일반화 능력을 갖추고 있습니다.



### Relation-based Counterfactual Data Augmentation and Contrastive Learning for Robustifying Natural Language Inference Models (https://arxiv.org/abs/2410.20710)
Comments:
          accepted at INTERSPEECH 2023

- **What's New**: 이 논문에서는 자연어 추론(NLI) 작업에서 모델의 강인성을 향상시키기 위해 데이터 증강(data augmentation) 기법을 활용한 새로운 방법을 제안합니다. 특히, 토큰 기반(token-based)과 문장 기반(sentence-based) 기법을 이용하여 반사실적(counterfactual) 문장 쌍을 생성하고 대조 학습(contrastive learning)을 적용하여 서로 다른 클래스 간의 차이를 학습합니다.

- **Technical Details**: 제안된 방법에서는 각 문장 쌍에 대해 entailment(함의), neutral(중립), contradiction(모순) 클래스의 문장 쌍을 생성합니다. 토큰 레벨 및 문장 레벨의 데이터 증강 접근법을 적용하여 사실적(factual) 및 반사실적 데이터를 생성하고, 대조 학습을 통해 같은 레이블을 가진 쌍은 가깝게, 다른 클래스의 쌍은 멀리 두어 임베딩 공간(embedding space)에서 학습합니다. 또한, T5 모델과 같은 사전 훈련된 언어 모델을 사용하여 문장 생성에 활용합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 반사실적 NLI 데이터셋과 일반 NLI 데이터셋에서 기존의 견고한 텍스트 분류 방법들과 비교하여 더 높은 정확도를 달성함을 보여줍니다.



### Guide-LLM: An Embodied LLM Agent and Text-Based Topological Map for Robotic Guidance of People with Visual Impairments (https://arxiv.org/abs/2410.20666)
- **What's New**: 이 논문은 시각 장애인(PVI)를 위한 내비게이션을 개선하기 위해 디자인된 Guide-LLM이라는 새로운 경량 언어 모델(LLM) 기반 에이전트를 소개합니다. 이는 기존의 도구들이 제공하지 못하는 정교한 공간 정보를 제공합니다.

- **Technical Details**: 당신의 초점은 텍스트 기반의 토포로지 맵(topological map)을 활용하여 LLM이 간단화된 환경 표현을 통해 경로를 계획할 수 있도록 하는 것입니다. 이 시스템은 사용자 선호도에 따라 위험을 탐지하고 개인화된 경로 계획을 가능하게 합니다.

- **Performance Highlights**: 모의 실험을 통해 Guide-LLM의 효과가 입증되었으며, 이는 효율적이고 적응력 있는 개인화된 내비게이션 지원을 제공함으로써 보조 기술에서 상당한 기술 발전을 나타냅니다.



### Embedding with Large Language Models for Classification of HIPAA Safeguard Compliance Rules (https://arxiv.org/abs/2410.20664)
- **What's New**: 본 연구는 mHealth 앱의 개발자들이 HIPAA (Health Insurance Portability and Accountability Act) 규정을 준수하는 것에 대한 인식 부족 문제를 지적하고, 이를 해결하기 위해 multilingual BERT (Bidirectional Encoder Representations from Transformers)를 활용하여 안전한 어플리케이션 개발을 위한 방안을 제시합니다.

- **Technical Details**: 기존의 Word2Vec 임베딩을 능가하기 위해, 연구진은 multilingual BERT를 활용하여 데이터셋에 대한 컨텍스트 임베딩을 생성하였습니다. BERT는 코드를 양방향으로 분석하여 HIPAA 코드 패턴을 임베드하는 데 사용되었으며, 이를 통해 다양한 머신 러닝 접근법에 적용되었습니다. 주요 머신 러닝 기법으로 로지스틱 회귀(Logistic Regression), 서포트 벡터 머신(Support Vector Machine), 랜덤 포레스트(Random Forest), 나이브 베이즈(Naive Bayes)가 소개됩니다.

- **Performance Highlights**: 로지스틱 회귀 모델은 99.95%의 높은 정확도를 달성하였으며, 서포트 벡터 머신은 99.79%, 랜덤 포레스트는 99.73%, 나이브 베이즈는 95.93%의 정확도를 기록했습니다. 이러한 결과는 기존 방법들보다 뛰어난 성능을 보여주며, 안전한 애플리케이션 개발의 가능성을 강조합니다.



### TurboHopp: Accelerated Molecule Scaffold Hopping with Consistency Models (https://arxiv.org/abs/2410.20660)
Comments:
          22 pages, 11 figures, 8 tables. Presented at NeurIPS 2024

- **What's New**: TurboHopp라는 새로운 3D scaffold hopping 모델을 소개하며, 기존의 scaffolding 방식의 효율성을 결합하고 속도를 혁신적으로 향상시켰습니다. TurboHopp는 전통적인 ddpm(denoising diffusion probabilistic models) 기반 모델에 비해 최대 30배 빠른 생성 속도를 자랑합니다.

- **Technical Details**: TurboHopp는 pocket-conditioned 3D scaffold hopping 접근 방식을 통해, 전통적인 scaffold hopping의 전략적인 강점을 유지하면서도, consistency 모델의 빠른 생성 능력을 융합합니다. 이를 통해, 강화 학습(Reinforcement Learning) 기법을 적용하여 분자의 특성을 더욱 최적화합니다. 해당 모델은 3D 구조 기반 약물 디자인(diffusion models) 분야에서 효율성을 증대시키며, 많은 약물 발견 상황에서 적용 가능성을 보여줍니다.

- **Performance Highlights**: TurboHopp는 더욱 빠른 추론 속도와 향상된 생성 품질을 자랑하며, 기존의 확산 기반 모델들과 비교하여 제약물 성질, 결합 친화도(binding affinity), 합성 가능성(synthesizability) 등에서 우수한 성능을 보입니다. 또한, 이 모델은 의약품 발견을 위한 강력한 도구로서 자리 잡을 것으로 기대됩니다.



### A Statistical Analysis of Deep Federated Learning for Intrinsically Low-dimensional Data (https://arxiv.org/abs/2410.20659)
- **What's New**: 이번 연구는 연합 학습(FL)의 일반화 오류를 탐구하며, 이질적 환경에서의 딥 연합 회귀의 일반화 특성을 분석합니다. 특히, 엔트로픽 차원(entropic dimension)을 기준으로 수렴 속도를 결정하는 데 있어 중요한 역할을 한다는 점을 강조합니다.

- **Technical Details**: 이 논문은 2단계 샘플링 모델을 활용하여 딥 연합 회귀의 일반화 속성을 조사합니다. $eta$-Hölder 함수로 설명되는 응답 변수와 설명 변수 간의 진정한 관계를 가정할 때, 참가 클라이언트의 오류율은 최대 $	ilde{O}ig((mn)^{-2eta/(2eta + ar{d}_{2eta}(	ext{λ}))}ig)$로 스케일링되며, 비참여 클라이언트의 경우는 $	ilde{O}ig(	ext{Δ} 	imes m^{-2eta/(2eta + ar{d}_{2eta}(	ext{λ}))} + (mn)^{-2eta/(2eta + ar{d}_{2eta}(	ext{λ}))}ig)$입니다. 여기서 $ar{d}_{2eta}(	ext{λ})$는 설명 변수의 주변 분포를 나타냅니다.

- **Performance Highlights**: 본 연구의 결과는 딥 연합 학습자의 수렴 속도가 명목상의 고차원성(nominal high-dimensionality)보다 내재적 차원(intrinsic dimensionality)에 따라 달라진다는 것을 명확히 입증합니다. 엔트로픽 차원을 사용하는 방식은 기존의 미코프스키 차원(Minkowski dimension)이나 바소슈타인 차원(Wasserstein dimension)보다 우수한 경계(bound)를 제공하는 것으로 나타났습니다.



### SubjECTive-QA: Measuring Subjectivity in Earnings Call Transcripts' QA Through Six-Dimensional Feature Analysis (https://arxiv.org/abs/2410.20651)
Comments:
          Accepted at NeurIPS 2024

- **What's New**: 본 연구는 사실 확인과 관련된 객체적 부정확성을 넘어, 명확성과 관련성을 결여한 사실적으로 올바른 답변 등 더욱 부드러운 형태의 잘못된 정보에 집중한다.

- **Technical Details**: SubjECTive-QA 데이터셋은 49,446개의 주석과 함께 Earnings Call Transcripts(ECTs)의 QA 세션을 포함한다. QA 쌍은 Assertive, Cautious, Optimistic, Specific, Clear, Relevant의 여섯 가지 주제로 구성되어 분석된다. 이를 통해 저자들은 언어 모델의 성능을 평가하고자 하였다.

- **Performance Highlights**: RoBERTa-base가 관련성과 명확성과 같은 낮은 주관성을 가진 특성에서 Llama-3-70b-Chat과 비슷한 weighted F1 score를 보였고, 주관성이 더 높은 Specific 및 Assertive 특성에서는 평균적으로 10.01% 높은 성과를 보였다. 또한 White House Press Briefings에 대한 최적 모델 테스트 결과 평균 weighted F1 score 65.97%를 기록하며 적용 가능성을 넓혔다.



### NeuZip: Memory-Efficient Training and Inference with Dynamic Compression of Neural Networks (https://arxiv.org/abs/2410.20650)
- **What's New**: 이 논문에서는 NeuZip이라는 새로운 가중치 압축 방식을 제안합니다. 이 방법은 신경망의 부동 소수점 숫자의 엔트로피를 기반으로 하여, 성능 저하 없이 메모리 효율적인 학습과 추론을 가능하게 합니다.

- **Technical Details**: NeuZip은 세 가지 구성 요소(부호 비트, 지수 비트, 가수 비트)로 각 부동 소수점 숫자를 표현하며, 지수 비트의 저엔트로피 특성을 활용하여 비손실 방식으로 압축합니다. 이를 통해 메모리 절약을 달성하며, 손실을 허용하는 NeuZip 변형을 통해 추론 시 추가적인 메모리 절약을 보장합니다.

- **Performance Highlights**: Llama-3 8B 모델의 학습 시 메모리 사용량을 31GB에서 16GB 이하로 줄였으며, 추론 시 메모리 사용량을 절반 이상 줄이면서 거의 손실 없는 성능을 유지했습니다.



### Language Models And A Second Opinion Use Case: The Pocket Professiona (https://arxiv.org/abs/2410.20636)
- **What's New**: 이번 연구는 복잡한 의료 사례에서 Large Language Models (LLMs)가 전문가의 결정-making 과정에서 공식적인 제3의 의견 도구로서의 역할을 테스트했습니다. 183개의 도전적인 의료 사례를 분석하며, 의료진의 동의 의견과 LLM의 성능을 비교했습니다.

- **Technical Details**: 연구는 Medscape에서 20개월 동안 수집한 183개의 도전적인 의료 사례를 분석하였고, LLM의 성능을 crowd-sourced 의사의 반응과 비교했습니다. 최신 LLM의 전반적인 정확도는 80%를 넘었으며, 이는 동일한 임상 사례에서 보고된 대부분의 인간 지표를 초과합니다. LLM의 성능은 단순 사례에서는 81%의 정확도로 높은 반면, 복잡한 사례에서는 43%의 정확도를 보였습니다.

- **Performance Highlights**: 이 연구는 LLM이 포괄적인 differential diagnoses 생성을 위한 도구로서 유용할 수 있음을 보여주었고, 임상 결정-making에서의 인지 편향을 줄이고 인지 부하를 감소시켜 의학적 오류의 일부 원인을 제거할 수 있는 잠재력을 강조했습니다. 추가적으로, 법적 데이터 세트를 비교 분석하여 LLM의 제3 의견 도출을 위한 경험적 맥락을 제공했습니다.



### LoRA Done RITE: Robust Invariant Transformation Equilibration for LoRA Optimization (https://arxiv.org/abs/2410.20625)
- **What's New**: 이 논문에서는 Low-Rank Adaptation (LoRA)의 최적화 문제를 해결하기 위해 LoRA-RITE라는 새로운 어댑티브 매트릭스 전처리 방법을 소개합니다. 기존의 LoRA 최적화 방법이 갖는 변환 불변성(transform invariance) 부족 문제를 해결하여 학습 효율을 개선합니다.

- **Technical Details**: LoRA는 훈련되지 않은 가중치(matrix) W에 저차원 행렬(저랭크 매트릭스) Z를 주입하는 방식으로 작동합니다. 저랭크 행렬 Z는 두 개의 행렬 A와 B의 곱으로 나타낼 수 있으며, LoRA-RITE는 이러한 것에 대해 변환 불변성을 제공하도록 설계되었습니다. 기존의 최적화 알고리즘들은 일반적으로 이 속성을 증명하지 못하고, 이는 학습 과정에서 비효율성을 초래합니다.

- **Performance Highlights**: LoRA-RITE는 Gemma 2B를 사용해 Super-Natural Instructions에서 4.6%의 정확도 향상을, 총 4개의 LLM 벤치마크에서는 3.5%의 성능 향상을 달성했습니다. GSM8K 데이터셋에서는 Gemma 7B IT 모델을 사용하여 55.5%의 정확도를 기록하였으며, 이는 Adam 최적화 알고리즘의 48.37%보다 월등한 수치입니다.



### Generator Matching: Generative modeling with arbitrary Markov processes (https://arxiv.org/abs/2410.20587)
- **What's New**: 이번 논문에서는 Markov 프로세스를 활용한 generative modeling을 위한 새로운 프레임워크인 generator matching을 소개합니다. 이는 다양한 generative modeling 방법들을 통합하는 방법론입니다.

- **Technical Details**: generator matching은 Markov 프로세스의 무한소 진화를 설명하는 generators를 사용하여 데이터를 생성합니다. 조건부 생성기를 구축하여 단일 데이터 포인트를 생성한 다음, 전체 데이터 분포를 생성하는 marginal generator를 근사하도록 학습합니다. 여기에는 diffusion models, flow matching 및 discrete diffusion models이 포함되어 있습니다.

- **Performance Highlights**: 우리의 방법은 단백질 및 이미지 구조 생성을 위한 실험에서 검증되었으며, jump process와의 superposition이 이미지 생성 개선에 기여한다는 것을 보여주었습니다.



### Toward Conditional Distribution Calibration in Survival Prediction (https://arxiv.org/abs/2410.20579)
Comments:
          Accepted to NeurIPS 2024. 41 pages, 23 figures

- **What's New**: 이 논문에서는 생존 예측 모델의 조건부 보정(conditional calibration)의 중요성을 강조하고, 이를 해결하기 위한 새로운 방법론인 CSD-iPOT 프레임워크를 제안합니다. 이 방법은 모델의 개별 생존 확률을 기준으로 하여 생존 분포를 생성합니다.

- **Technical Details**: CSD-iPOT는 conformal prediction(적합한 예측) 기법에 기반하여 구성됩니다. 이 프레임워크는 생존 분석의 조건부 분포 보정을 지원하며, 이전의 방법론들보다 향상된 성능을 나타냅니다. 이 방법은 not only marginal calibration (주변 보정) 만을 고려하는 것이 아니라, 조건부 특성에 대한 보정을 가능하게 합니다.

- **Performance Highlights**: CSD-iPOT는 15개의 다양한 실제 데이터셋을 통해 실험하였으며, 주변 및 조건부 보정을 동시에 개선하는 데 성공했습니다. 또한 계산적으로 효율적인 특성을 가지고 있어, 기존의 방법들보다 더 나은 성능을 보이는 것을 입증했습니다.



### Meta-Learning Approaches for Improving Detection of Unseen Speech Deepfakes (https://arxiv.org/abs/2410.20578)
Comments:
          6 pages, accepted to the IEEE Spoken Language Technology Workshop (SLT) 2024

- **What's New**: 이 연구는 새로운 유형의 음성 딥페이크 공격을 탐지할 수 있는 시스템을 개발하기 위해 메타학습(meta-learning) 접근 방식을 활용합니다. 이 방법은 훈련 데이터에 포함되지 않은 공격에 대해서도 적은 샘플만으로 일반화하는 것을 목표로 합니다. 특히, 96개의 샘플만 사용하여 EER (Equal Error Rate)을 21.67%에서 10.42%로 개선하였습니다.

- **Technical Details**: 이 논문에서는 메타학습을 활용하여 공격에 대한 불변 화이트피처(attack-invariant features)를 학습하고, 제한된 샘플로 보이지 않는 공격에 적응하는 방법을 제안합니다. 이를 통해 많은 비용을 들이지 않고도 높은 품질의 학습을 이뤄내는 것이 가능합니다. 또한, 소수 샘플을 통해 성능 개선 경향을 관찰하며, 이는 도메인 일반화(domain generalization)에 기여합니다.

- **Performance Highlights**: 본 연구의 실험 결과는 새로운 공격 유형에 대한 빠른 적응을 가능하게 하여, 심지어 기존의 훈련 데이터에 포함되지 않은 상황에서도 딥페이크 탐지 성능의 유의미한 향상을 보여주었습니다. 특히 EER 성능을 획기적으로 개선시킬 수 있는 방법론을 제시했습니다.



### Unsupervised Panoptic Interpretation of Latent Spaces in GANs Using Space-Filling Vector Quantization (https://arxiv.org/abs/2410.20573)
- **What's New**: 본 논문에서는 데이터 레이블이나 주석이 달린 샘플을 필요로 하지 않는 새로운 접근 방법인 space-filling vector quantization (SFVQ)를 제안합니다. 이는 모호한 latent space를 해석 가능하게 만드는 데 기여합니다.

- **Technical Details**: SFVQ는 데이터의 조각별 선형 곡선 위에서 양자화하여 latent space의 기본 형태적 구조를 캡처합니다. 본 연구는 pretrained StyleGAN2 및 BigGAN 네트워크의 latent space 모델링에 이 기법을 적용하였습니다.

- **Performance Highlights**: 실험 결과, SFVQ 곡선은 latent space의 일반적인 해석 가능한 모델을 제공하며, 이것이 각각의 생성 요인에 대응하는 latent space의 어떤 부분인지 파악할 수 있게 합니다. 또한, SFVQ의 곡선의 각 선은 이해할 수 있는 이미지 변환을 적용하기 위한 해석 가능한 방향으로 활용될 수 있습니다.



### SPICEPilot: Navigating SPICE Code Generation and Simulation with AI Guidanc (https://arxiv.org/abs/2410.20553)
Comments:
          6 pages, 2 figures, 5 tables

- **What's New**: 이 논문에서는 SPICE 코드 생성에서의 기존 LLM의 한계를 분석하고, PySpice를 사용하여 생성된 Python 기반 데이터셋과 이를 지원하는 프레임워크인 SPICEPilot를 소개합니다. 이는 다양한 회로 구성에서 SPICE 코드 생성을 자동화하기 위한 중대한 진전을 나타냅니다.

- **Technical Details**: SPICEPilot 프레임워크는 SPICE 시뮬레이션 스크립트를 자동으로 생성하고, 회로 생성을 평가하기 위한 표준화된 벤치마킹 메트릭을 도입하며, LLM을 하드웨어 설계 프로세스에 통합하기 위한 로드맵을 제공합니다. 이를 통해 LLM이 аналог 회로 설계에 필요한 데이터셋을 생성할 수 있도록 하며, 회로 성능을 평가하기 위한 신뢰할 수 있는 벤치마킹 메트릭을 제안합니다.

- **Performance Highlights**: LLM의 SPICE 코드 생성 성능을 분석하고, 개방형 및 상용 모델을 비교하여 기본 회로를 생성할 수 있는 능력을 평가합니다. 또한, LLM의 성능, 정확성 및 신뢰성을 정확히 평가하기 위한 새로운 메트릭과 벤치마크를 설정할 필요성을 강조하며, LLM을 기반으로 한 하드웨어 설계 솔루션의 향상을 위한 미래 연구 방향을 제시합니다.



### SympCam: Remote Optical Measurement of Sympathetic Arousa (https://arxiv.org/abs/2410.20552)
Comments:
          Accepted for publication at the IEEE-EMBS International Conference on Biomedical and Health Informatics

- **What's New**: 이 연구는 SympCam이라는 새로운 3D convolutional architecture를 통해 얼굴 비디오만으로 사람의 sympathetic arousal(교감 각성)을 예측할 수 있는 방법을 제시합니다. 이는 비침습적인 방식으로 일반 RGB 카메라를 사용하여 스트레스를 측정하는 새로운 가능성을 열어줍니다.

- **Technical Details**: SympCam 모델은 temporal attention module (TAM)을 통합하여 데이터의 시퀀스 처리 효율성을 높입니다. 이 모델은 20명의 참여자로부터 수집된 얼굴과 손 비디오를 기반으로 하며, electrodermal activity (EDA) 및 photoplethysmography (PPG) 측정과 동기화되었습니다. 이 데이터셋은 원거리에서의 sympathetic arousal 예측을 위해 특별히 설계되었습니다.

- **Performance Highlights**: 제안된 방법은 기존 연구보다 48% 더 높은 평균 상관관계 0.77을 달성하였으며, 자주 사용되는 rPPG 방법과 비교해 61% 향상된 90%의 균형 정확도를 보였습니다. 이 연구는 원거리에서의 sympathetic arousal 예측을 위해 다른 연구자들이 사용할 수 있는 최초의 데이터셋 제공을 목표로 합니다.



### Deep Reinforcement Learning Agents for Strategic Production Policies in Microeconomic Market Simulations (https://arxiv.org/abs/2410.20550)
- **What's New**: 이 논문은 전통적인 경제 모델의 한계를 극복하기 위해 심층 강화 학습(Deep Reinforcement Learning, DRL)의 활용을 탐구합니다. 이를 통해 복잡하고 변동성이 큰 미시 경제 시장에서 최적의 생산 전략을 도출하는 방법을 제시하고 있습니다.

- **Technical Details**: 논문에서는 DRL 기반 접근법을 제안하여 경쟁 시장에서 여러 생산자가 수요, 공급, 가격 및 기타 노이즈로 인해 최적의 생산 결정을 내리는 정책을 학습할 수 있게 합니다. 여기서는 Markov Decision Process (MDP)를 사용하여 에이전트의 상호작용을 모델링하며, 에이전트는 시뮬레이터 내에서 다양한 상태에서 정책을 조정해 나갑니다.

- **Performance Highlights**: 실험 결과, DRL로 훈련된 에이전트는 변동성이 큰 시장 조건에서도 생산 수준을 전략적으로 조정하여 장기적인 수익성을 극대화할 수 있음을 보여주었습니다. 이 연구는 이론적 경제 모델과 실제 시장 시뮬레이션 간의 간극을 해소할 수 있는 가능성을 제시합니다.



### Malinowski in the Age of AI: Can large language models create a text game based on an anthropological classic? (https://arxiv.org/abs/2410.20536)
Comments:
          Accepted at KUI 2024

- **What's New**: 이 연구는 LLM (Large Language Models)이 인류학 고전 문헌을 기반으로 독립적으로 텍스트 기반 게임을 생성할 수 있는지를 탐구하고 있으며, 인류학 지식의 전달 효과성을 평가하고 있습니다.

- **Technical Details**: 연구는 HCI(인간-컴퓨터 상호작용) '디자인 씽킹' 원칙에 따라 진행되었으며, ChatGPT를 활용하여 Bronislaw Malinowski의 'Argonauts of the Western Pacific'를 기반으로 세 가지 프로토타입 게임을 제작했습니다.

- **Performance Highlights**: 이원이 아누크 많은 장점에도 불구하고, 모델은 주제의 깊은 이해 제공에 어려움을 겪었고, 잘못된 정보에 대한 의존성 증가, 장기간의 게임 플레이 시 단조로운 반응 경향을 보이며, 자세한 전기 정보 제공에 어려움을 겪었습니다. 그러나 이 연구는 AI와 인류학의 교차점에서 새로운 연구 방향을 제시합니다.



### Asynchronous Perception Machine For Efficient Test-Time-Training (https://arxiv.org/abs/2410.20535)
Comments:
          Accepted to NeurIPS 2024 Main Track. APM is a step to getting Geoffrey Hinton's GLOM working. Original GLOM paper said: "This paper was quickly hijacked by the need to justify the design decisions". 3 years have passed us. This work provides some justifications and been peer-reviewed and accepted by our peers. A humble blogpost can be found at this https URL

- **What's New**: 이 논문에서는 Asynchronous Perception Machine (APM), 테스트 타임 교육(test-time-training, TTT)을 위한 계산 효율적인 아키텍처를 제안합니다. APM은 이미지를 비대칭적으로 순서에 상관없이 한 번에 하나의 패치를 처리할 수 있으며, 여전히 네트워크에 의미 인식을 인코딩할 수 있습니다.

- **Technical Details**: APM은 기존 TTT 접근 방식보다 경쟁력 있는 성능을 보이며, 데이터 세트 특정 사전 훈련이나 데이터 증강, 프리텍스트 작업 없이 분포 외(out-of-distribution) 이미지를 인식할 수 있습니다. TTT를 수행할 때 APM은 테스트 샘플의 표현을 단 한 번 정제하며, 이후 모든 반복은 이 단일 표현에 대해 과적합(over-fit)하여 데이터 증강이나 프리텍스트 작업을 요구하지 않습니다.

- **Performance Highlights**: APM은 기존 방법에 비해 FLOPs의 수를 거의 반으로 줄일 수 있으며, 16개의 데이터 세트에서 TTT 성능이 0.4%에서 8% 향상되었습니다. 또한, APM은 아키텍처가 간단하여 5개의 층을 가진 단일 MLP와 합성곱(convolutional) 레이어로 구성되어 있으며, 하나의 샘플을 사용하여 의미적으로 클러스터링할 수 있는 능력을 가지고 있습니다.



### CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming (https://arxiv.org/abs/2410.20527)
- **What's New**: CodeRosetta는 프로그래밍 언어와 그 HPC (High-Performance Computing) 확장을 번역하기 위해 특별히 설계된 인코더-디코더 트랜스포머 모델입니다. 이 모델은 C++에서 CUDA로, 그리고 Fortran에서 C++로 번역하는 작업에서 평가되었습니다.

- **Technical Details**: 이 모델은 코드 의미론을 효과적으로 캡처하고 병렬 구조의 미세한 뉘앙스를 이해하기 위해 맞춤형 학습 프레임워크와 사전 훈련(pretraining) 및 훈련(training) 목표를 사용합니다. 이를 통해 양방향 번역이 가능해집니다.

- **Performance Highlights**: CodeRosetta는 C++에서 CUDA로의 번역에서 기존의 최신 기법보다 2.9 BLEU 포인트와 1.72 CodeBLEU 포인트 우수한 성능을 보이며, 컴파일 정확도를 6.05% 향상시킵니다. 또한, 일반 폐쇄형 LLM과 비교 시 C++에서 CUDA로의 번역에서 22.08 BLEU와 14.39 CodeBLEU 포인트를 개선하고, 컴파일 정확도를 2.75% 높였습니다. 마지막으로, Fortran에서 병렬 C++로의 번역에서도 우수한 성과를 보여주며, 이는 복잡한 작업에 대한 첫 번째 인코더-디코더 모델로 여겨집니다.



### Props for Machine-Learning Security (https://arxiv.org/abs/2410.20522)
- **What's New**: 이 논문에서는 기계 학습(ML)을 위한 인증 및 개인 정보 보호 접근 방식으로서 'protected pipelines'(약칭 props)을 제안합니다. 이는 깊은 웹(deep web) 데이터의 안전한 사용을 가능케 하여 ML 개발에서의 고품질 훈련 데이터의 부족 문제를 해결합니다.

- **Technical Details**: Props는 깊은 웹 데이터 소스에서 ML 생태계의 사용 지점으로 확장되는 데이터 파이프라인입니다. props는 데이터의 개인 정보 보호와 무결성이라는 두 가지 주요 보안 속성을 제공합니다. 이를 통해 데이터가安全하게 사용될 수 있으며, 사용자는 데이터 발표에 대한 통제 권한을 유지할 수 있습니다.

- **Performance Highlights**: Props는 민감한 사용자 데이터를 사용하면서도 안전한 데이터 공유를 제공하며, 사용자가 제어하는 방식으로 데이터가 흐를 수 있도록 합니다. 이는 모델의 훈련과 추론 모두에 사용될 수 있으며, 사용자가 자신의 민감한 데이터를 안전하게 처리할 수 있는 방법을 제공합니다.



### MidiTok Visualizer: a tool for visualization and analysis of tokenized MIDI symbolic music (https://arxiv.org/abs/2410.20518)
Comments:
          in Extended Abstracts for the Late-Breaking Demo Sessionof the 25th Int. Society for Music Information Retrieval Conf., San Francisco, United States, 2024

- **What's New**: 이번 논문에서는 MIDI 데이터의 복잡성을 해결하기 위해, MidiTok Python 패키지의 다양한 MIDI 토큰화 방법을 시각화하고 탐색할 수 있도록 설계된 웹 애플리케이션인 MidiTok Visualizer를 소개합니다. 이 도구는 사용자가 MIDI 파일을 업로드하고 토큰화된 데이터를 시각적으로 확인할 수 있도록 합니다.

- **Technical Details**: MidiTok Visualizer는 사용자에게 MIDI 파일을 업로드할 수 있게 하며, 생성된 토큰의 그래픽 표현을 제공합니다. 이 애플리케이션은 FastAPI를 사용하여 백엔드를 구축하고, React를 통해 프론트엔드를 제공합니다. Pydantic을 이용해 데이터 검증과 모델링을 수행하고, MusPy를 통해 MIDI 파일을 처리하여 중요한 음악 정보를 추출합니다. Docker를 이용해 애플리케이션은 컨테이너화되어 단일 docker compose 명령으로 실행 가능합니다.

- **Performance Highlights**: 현재 MidiTok Visualizer는 CPWord, MIDI-Like, Octuple, REMI, Structured, TSD와 같은 여러 토큰화 방법을 지원합니다. 사용자는 piano roll을 통해 업로드된 MIDI 파일의 노트를 시각화하고, 다양한 매개변수 설정을 실험할 수 있으며, 이는 MIDI 데이터 분석에 대한 접근성을 높이는 데 기여합니다.



### Symbotunes: unified hub for symbolic music generative models (https://arxiv.org/abs/2410.20515)
- **What's New**: Symbotunes는 기호 음악 생성 모델을 위한 오픈 소스 통합 허브로, 여러 유명한 모델의 현대적인 Python 구현과 함께 데이터 생성 및 훈련을 위한 통합 파이프라인을 제공합니다.

- **Technical Details**: Symbotunes는 anaconda 패키지 관리자를 활용하여 의존성을 관리하며, Python 3.12와 PyTorch 2.2, PyTorch Lightning 2.2, MidiTok 3.0을 사용합니다. 네 가지 주요 하위 폴더로 구성되어 있으며, 사용자에게 yaml 형식의 구성 파일을 통해 다양한 실험을 간편하게 지원합니다. Symbotunes는 Folk-RNN, MusicVAE, ABC GPT2와 같은 기호 음악 모델의 구현을 포함하고 있습니다.

- **Performance Highlights**: Symbotunes는 재현성을 높이고, 교육자와 연구자가 기호 음악 모델을 탐색할 수 있도록 표준화된 플랫폼을 제공합니다. 향후 더 많은 모델, 데이터셋 및 기능 추가를 계획하고 있으며, 기호 음악 생성 연구에 중요한 자원이 되고자 합니다.



### $\textit{Who Speaks Matters}$: Analysing the Influence of the Speaker's Ethnicity on Hate Classification (https://arxiv.org/abs/2410.20490)
Comments:
          9 pages, 3 figures, 3 tables. To appear in NeurIPS SafeGenAI 2024 Workshop

- **What's New**: 본 연구는 Large Language Models (LLMs)에 대한 하이 스테이크(high-stakes) 작업인 혐오 발언 감지(hate speech detection)에서의 강인성(robustness)을 조사합니다. 특히, 화자의 인종을 언급하는 명시적(explicit) 및 암시적(implicit) 마커가 입력에 주입될 때 모델의 반응을 분석합니다.

- **Technical Details**: 이 연구에서는 명시적 마커로 화자의 정체성을 언급하는 문구를 주입하고, 암시적 마커로 방언(dialectal) 특징을 주입했습니다. 4개의 인기 LLM과 5개 인종에 걸쳐 모델 출력을 분석하면서 마커의 존재가 반응에 미치는 영향을 평가했습니다.

- **Performance Highlights**: 암시적 방언 마커가 포함된 입력의 경우 모델 출력이 변할 확률이 명시적 마커보다 더 높았습니다. 또한, 출력 변동의 비율은 인종마다 다르게 나타났으며, 더 큰 모델일수록 강인성이 높은 것으로 나타났습니다. 이는 혐오 발언 감지와 같은 고위험 작업에 LLM을 배치하는 데 있어 신중함의 필요성을 강조합니다.



### Efficient Diversity-based Experience Replay for Deep Reinforcement Learning (https://arxiv.org/abs/2410.20487)
- **What's New**: 본 논문은 스파스 리워드 환경에서의 학습 효율성을 크게 향상시키기 위해 다양성 기반 경험 재플레이(Diversity-Based Experience Replay, DBER)라 불리는 새로운 접근법을 제안합니다.

- **Technical Details**: DBER는 결정론적 포인트 프로세스(Determinantal Point Processes, DPPs)를 활용하여 상태 실현에서 다양한 샘플을 우선시합니다. 이를 통해 샘플 활용 효율성을 높이고, TD-오류(Temporal Difference Error)에 의존하지 않으면서 샘플의 대표성과 효과성을 보장합니다.

- **Performance Highlights**: 다양한 시뮬레이션 환경에서 실험한 결과, DBER는 스파스 리워드 환경에서도 학습 효율성을 크게 개선하고 뛰어난 성능을 보여주었습니다. 이는 고차원 상태 공간에서의 강화 학습 문제를 해결하기 위한 간단하면서도 효과적인 솔루션을 제공합니다.



### Improving Decision Sparsity (https://arxiv.org/abs/2410.20483)
Comments:
          Accepted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 이 논문은 Sparse Explanation Value(SEV)의 개념을 확장하여 머신러닝 모델의 의사결정에서의 해석 가능성을 높이는 새로운 방법론을 제안합니다. SEV는 특정 의사결정에서 핵심적인 정보의 양을 반영하는 로컬 스파시티(local sparsity)를 측정하는 데 중점을 둡니다.

- **Technical Details**: SEV는 하이퍼큐브(hypercube) 상에서의 이동을 기반으로 하며, 참조 포인트의 위치 조정과 기능 공간(feature space)에서의 거리 변환을 고려하여 해석의 신뢰성(credibility)과 의사결정 스파시티를 최적화합니다. 논문에서는 클러스터 기반 SEV와 트리 기반 SEV를 제시하고, 신뢰성을 개선하는 방법론과 함께 머신러닝 모델의 의사결정 스파시티를 최적화하는 알고리즘을 제안합니다.

- **Performance Highlights**: 제안된 알고리즘은 의사결정 성능을 저하시키지 않으면서 고스파시티 모델을 효율적으로 생성할 수 있으며, SEV 계산을 위한 새로운 세 가지 접근 방식을 도입하여 더 의미 있는 해석 결과를 도출할 수 있음을 보여줍니다.



### What Factors Affect Multi-Modal In-Context Learning? An In-Depth Exploration (https://arxiv.org/abs/2410.20482)
Comments:
          Accepted at NeurIPS 2024

- **What's New**: 최근 다중 모달 인컨텍스트 학습(Multi-Modal In-Context Learning, MM-ICL)에서의 빠른 발전이 주목받고 있습니다. 본 연구는 MM-ICL의 성능에 영향을 미치는 요인을 조사하여, 효과적인 전략을 최적화하기 위한 기초 자료를 제공하고자 합니다.

- **Technical Details**: 본 연구는 MM-ICL의 세 가지 핵심 단계인 시연 검색(demonstration retrieval), 시연 순서(demonstration ordering), 프롬프트 구성(prompt construction)에 대한 포괄적인 실험을 수행했습니다. 실험은 6개의 시각 대형 언어 모델(vision large language models, VLLMs)과 20가지 전략을 사용하여 실시되었습니다. 연구 결과는 다중 모달 검색기가 필요하며, 시연의 내부 순서(intra-demonstration ordering)가 외부 순서(inter-demonstration ordering)보다 중요함을 강조하고, 소개 지시문(introductory instructions)이 프롬프트에 포함됨으로써 작업 설명(task comprehension)이 향상된다는 것을 보여줍니다.

- **Performance Highlights**: 1. 다중 모달 검색 방법이 단일 모달 방법보다 평균적으로 더 뛰어난 성능을 보임. 2. 내부 시연 순서가 모델 성능에 미치는 영향이 외부 시연 순서보다 크게 나타남. 3. 소개 지시문을 포함할 경우 MM-ICL 성능이 일관되게 향상됨.



### MusicFlow: Cascaded Flow Matching for Text Guided Music Generation (https://arxiv.org/abs/2410.20478)
Comments:
          ICML 2024

- **What's New**: 논문에서는 MusicFlow라는 텍스트-뮤직(Text-to-Music) 생성 모델을 소개합니다. 본 모델은 flow matching을 기반으로 하여 텍스트 설명과 음악 오디오 간의 관계를 모델링하며, 마스킹된 예측을 학습 목표로 삼아 다양한 음악 생성 작업을 수행할 수 있는 능력을 가지고 있습니다.

- **Technical Details**: MusicFlow는 두 개의 flow matching 네트워크로 구성되어 있으며, 텍스트 설명을 의미적 특징의 시퀀스로 변환하고, 다시 이 의미를 해석 가능한 음향 특징으로 변환합니다. 비자동 회귀(non-autoregressive) 방식으로 작동하며, 효율성을 높이는 flow matching 목표를 이용하고 있습니다. 이를 통해 기존의 모델보다 더 작은 크기와 더 빠른 추론 속도로 성능을 발휘합니다.

- **Performance Highlights**: MusicCaps에 대한 실험 결과, MusicFlow로 생성된 음악은 품질과 텍스트 일관성에서 훌륭한 성능을 보여주며, 기존 모델보다 약 2~5배 작은 크기와 5배 적은 반복 단계로 결과를 얻을 수 있었습니다. 또한, MusicFlow는 음악 채우기(music infilling), 음악 연속 생성(music continuation) 등의 다양한 음악 생성 작업에서도 경쟁력 있는 성능을 발휘합니다.



### A Derivational ChainBank for Modern Standard Arabic (https://arxiv.org/abs/2410.20463)
- **What's New**: 이 연구는 아랍어 파생 형태론을 모델링하기 위한 새로운 프레임워크인 'Arabic Derivational ChainBank'를 소개합니다. 이 프레임워크는 파생된 단어들 간의 관계를 형성하여 의미와 형태 간의 연결을 구축합니다.

- **Technical Details**: ChainBank는 동적인 지식 기반의 트리 그래프로 아랍어 파생 형태론의 계층적 구조를 나타냅니다. 각 노드는 파생 단어와 이는 morphosemantic 속성(형태, 품사, 기능적 특징 등)을 포함합니다. 파생 관계를 나타내는 연결이 각 자식 노드를 분류합니다. 이 네트워크는 CamelMorph를 통해 아랍어 단어 간의 파생 관계를 연결하는 대규모 네트워크로 구성됩니다.

- **Performance Highlights**: ChainBank는 23,333개의 평가된 파생 관계를 포함하여 효율성을 입증합니다. 결과적으로 파생된 단어의 어근과 연결된 렘마의 체인을 만듭니다.



### TrajAgent: An Agent Framework for Unified Trajectory Modelling (https://arxiv.org/abs/2410.20445)
Comments:
          12 pages; the code will be openly accessible at: this https URL

- **What's New**: 본 논문에서는 다양한 경로 모델링(task) 작업을 통합하기 위해 LLM(대형 언어 모델)을 기반으로 한 간편한 프레임워크인 TrajAgent를 제안합니다. TrajAgent는 통합된 데이터 및 모델 인터페이스를 제공하는 UniEnv를 개발하여 다양한 모델을 실행하고 훈련할 수 있도록 지원합니다.

- **Technical Details**: TrajAgent는 TAgent라는 자동 경로 모델링 작업을 설계하고, AutOpt라는 최적화 모듈을 통해 성능을 향상시킵니다. UniEnv는 데이터 전처리 및 일관된 작업 흐름을 지원하며, 다양한 경로 모델링 작업을 위한 통합된 환경을 제공합니다. 경로 예측, 분류, 생성 작업을 포함한 다양한 작업 유형을 지원합니다.

- **Performance Highlights**: 실험 결과, TrajAgent는 네 가지 실제 데이터 세트에 대해 평균 15.43%의 성능 향상을 기록하여 통합 경로 모델링에서의 효과성을 입증하였습니다.



### TEAFormers: TEnsor-Augmented Transformers for Multi-Dimensional Time Series Forecasting (https://arxiv.org/abs/2410.20439)
- **What's New**: TEAFormer를 통해 기존 Transformer 모델의 한계를 극복했습니다. 전통적인 모델이 다차원 구조를 효과적으로 보존하지 못하는 문제를 해결하고, 텐서 확장 및 압축을 기능으로 통합하여 예측 정확도를 향상시켰습니다.

- **Technical Details**: TEAFormer는 Tensor-Augmentation (TEA) 모듈을 도입하여, 다채널에서 다중 시각적 피처 학습을 통한 텐서의 확장을 활용하고, 자동 텐서 분해를 통해 정보를 압축하여 계산 부하를 줄입니다. 이 모듈은 Transformer의 인코더-디코더 구조 및 어텐션 메커니즘과 높은 호환성을 가집니다.

- **Performance Highlights**: TEAFormer는 Transformer, Informer, Autoformer와 같은 세 가지 인기있는 모델에 통합되어 34/42 시험에서 기초 모델 대비 성능을 유의미하게 개선하며, MAE(Mean Absolute Error) 및 MSE(Mean Squared Error) 평가 지표에서 뛰어난 성능을 보임을 입증했습니다.



### MedGo: A Chinese Medical Large Language Mod (https://arxiv.org/abs/2410.20428)
Comments:
          12 pages, 1 figure

- **What's New**: 이 논문은 MedGo라는 세종 (生科学의 두문자) 의학 대용량 언어 모델을 소개합니다. 이 모델은 고품질 비지도 학습 의료 데이터, 지도(Supervised) 데이터 및 개인 맞춤형 선호(Preference Alignment) 데이터를 조합하여 학습되었습니다.

- **Technical Details**: MedGo는 대규모 도메인 특화 의료 데이터셋을 구축하여 의학 지식에 대한 깊은 이해를 가져왔으며, 이 모델은 세 가지 단계(Pre-training, Supervised Fine-Tuning, Preference Alignment)를 통해 최적화되었습니다.

- **Performance Highlights**: MedGo는 CBLUE 벤치마크 평가에서 1위를 달성하였으며, ClinicalQA 데이터셋에서 근본 모델인 Qwen2보다 우수한 성과를 보였습니다. 이로써 MedGo의 자동 의료 질문 응답 및 임상 결정 지원(Clinical Decision Support) 개선 가능성을 강조하고 있습니다.



### NT-VOT211: A Large-Scale Benchmark for Night-time Visual Object Tracking (https://arxiv.org/abs/2410.20421)
Comments:
          Oral Acceptance at the Asian Conference on Computer Vision (ACCV) 2024, Hanoi, Vietnam

- **What's New**: 최근 발표된 논문에서는 NT-VOT211이라는 새로운 벤치마크를 소개합니다. 이 벤치마크는 시각 객체 추적(VOT) 알고리즘의 평가를 위해 설계되었으며, 211개의 다양한 비디오로 구성되어 있고, 특히 야간 조건에서의 추적 성능 향상을 위한 데이터셋입니다.

- **Technical Details**: NT-VOT211은 211,000개의 잘 라벨링된 프레임과 카메라 움직임, 변형, 빠른 움직임, 움직임 흐림, 작은 목표, 방해물, 가려짐 및 시야 밖 등의 8가지 속성으로 구성되어 있습니다. 이는 낮은 가시성과 이미지 흐림, 방해물 문제를 포함한 야간 추적의 고유한 문제들을 해결하기 위해 설계되었습니다.

- **Performance Highlights**: 42개 다양한 추적 알고리즘을 NT-VOT211에서 평가한 결과, 기존 낮 시간 벤치마크에 비해 낮은 성능을 보였습니다. 이는 야간 조건에서의 추적 알고리즘의 개선 기회를 마련하고, 실제 응용 프로그램에서의 추적 성능 향상을 위한 새로운 방향을 제시합니다.



### Inevitable Trade-off between Watermark Strength and Speculative Sampling Efficiency for Language Models (https://arxiv.org/abs/2410.20418)
- **What's New**: 이번 연구에서는 큰 언어 모델(LLM) 생성 과정에서 워터마크 주입(watermarking)과 샘플링 가속화(acceleration) 간의 상충 문제를 조사하였습니다. 기존의 방법으로는 두 가지 목표를 동시에 달성할 수 없음을 입증하는 'no-go theorem'을 제안하였습니다.

- **Technical Details**: 저자들은 워터마크 강도(watermark strength)와 샘플링 효율(sampling efficiency) 간의 무관한 상관관계를 고려하여, 'two reweight framework'를 제안했습니다. 이를 통해 워터마킹과 샘플링 기술을 통합할 수 있는 방법을 모색했습니다.

- **Performance Highlights**: 제안된 프레임워크 내에서 두 가지 실용적 알고리즘을 제안하여, 하나는 워터마크 강도를 유지하고, 다른 하나는 샘플링 효율을 유지하도록 설계되었습니다. 이는 LLM의 워터마크 토큰 생성에 대한 이해를 심화시키고, 효율성과 저작권 보호 간의 상충 관계를 명확히 설명합니다.



### ThunderKittens: Simple, Fast, and Adorable AI Kernels (https://arxiv.org/abs/2410.20399)
- **What's New**: ThunderKittens (TK) 프레임워크는 GPU 아키텍처에 최적화된 AI 커널을 쉽게 작성할 수 있게 해주는 핵심 추상화 기능을 제공합니다. 이 프레임워크는 성능을 최대화하기 위해 3가지 레벨(워프, 블록, 그리드)에서 GPU의 기본 데이터 구조와 병렬 계산 작업을 지원합니다.

- **Technical Details**: TK는 다음의 세 가지 수준에서 GPU 아키텍처와 매핑됩니다: (1) 워프 수준에서 16x16 매트릭스 타일을 기본 데이터 구조로 사용하고, PyTorch와 유사한 병렬 계산 작업을 제공, (2) 스레드 블록 수준에서 비동기 작업을 조정하기 위한 템플릿을 제공, (3) 그리드 수준에서 블록 실행 및 정리와 메모리 비용을 줄이는 지원을 제공합니다.

- **Performance Highlights**: ThunderKittens의 성능은 이전 커널과 동등하거나 이를 초과하는 결과를 보여주며, Attention 성능에서 10-40% 개선을 보였고, 선형 Attention에서 14배 성능 향상을 달성했습니다. TK는 실제 AI 작업에 적용되고 있으며, CUDA 경험이 없는 학생팀에 의해 작성되었습니다.



### Lodge++: High-quality and Long Dance Generation with Vivid Choreography Patterns (https://arxiv.org/abs/2410.20389)
Comments:
          Project page: this https URL

- **What's New**: Lodge++는 음악과 특정 장르에 따라 고품질의 초장기 댄스를 생성할 수 있는 새로운 안무 프레임워크입니다. 이 프레임워크는 계산 효율성을 처리하고 복잡한 전 세계적인 안무 패턴을 학습하며 지역적인 춤의 물리적 품질을 유지하는 것을 목표로 합니다.

- **Technical Details**: Lodge++는 두 단계로 구성된 구조를 사용합니다. 첫 번째 단계에서는 글로벌 안무 네트워크가 복잡한 글로벌 안무 패턴을 포착하는 거친 회전형 댄스 원시형태를 생성하고, 두 번째 단계에서는 이를 기반으로 한 댄스 확산 모델이 고품질의 장기 댄스를 병렬로 생성하여 복잡한 안무 패턴에 충실하게 따릅니다. 또한, 물리적 타당성을 높이기 위해 침투 가이드 모듈과 발의 정제 모듈을 사용하여 캐릭터의 자기 침투 문제를 해결하고 발-바닥 접촉을 최적화합니다.

- **Performance Highlights**: Lodge++는 다양한 댄스 장르에 적합한 초장기 댄스를 빠르게 생성할 수 있으며, 잘 조직된 글로벌 안무 패턴과 고품질의 지역적 움직임을 보장합니다. 이 방법은 댄스와 음악의 일치를 더 잘 유지할 수 있어, 이전의 기법들에 비해 실질적인 응용 가능성이 더 높아졌습니다.



### Addressing the Pitfalls of Image-Based Structural Health Monitoring: A Focus on False Positives, False Negatives, and Base Rate Bias (https://arxiv.org/abs/2410.20384)
- **What's New**: 이 연구는 이미지 기반 구조 건강 모니터링(Structural Health Monitoring, SHM) 기술이 구조적 손상을 탐지하는 데 가지고 있는 한계를 탐구하였습니다.

- **Technical Details**: 이 연구는 머신 러닝(Machine Learning)과 컴퓨터 비전(Computer Vision)을 활용하여 이미지 기반 SHM이 수동 검사에 비해 확장 가능하고 효율적인 대안이 될 수 있음을 보여줍니다. 그러나 실제 손상의 발생 확률이 낮을 경우, 잘못된 긍정(false positives) 및 잘못된 부정(false negatives)과 같은 문제로 인해 신뢰성이 저해됩니다. 연구는 베이지안(Bayesian) 분석과 자주론(frequentist) 접근을 통해 손상 탐지 시스템의 정밀도를 평가하고, 손상의 발생이 드문 경우에도 높은 정확도를 가진 모델이 오해를 불러일으킬 수 있음을 드러냈습니다.

- **Performance Highlights**: 연구에서는 여러 데이터 소스를 결합한 하이브리드 시스템(hybrid systems), 중요한 평가를 위한 사람의 참여(human-in-the-loop) 방법, 그리고 훈련 데이터의 품질 향상 등을 통한 이러한 한계를 완화하기 위한 전략을 논의합니다. 이 연구의 결과는 이미지 기반 SHM 기술의 실용적 적용 가능성에 대한 중요한 통찰을 제공하며, 실제 인프라 모니터링을 위해 그들의 잠재력과 한계를 강조합니다.



### Multiple kernel concept factorization algorithm based on global fusion (https://arxiv.org/abs/2410.20383)
Comments:
          in Chinese language

- **What's New**: 이번 논문에서는 Non-negative Matrix Factorization(NMF) 알고리즘의 한계를 극복하기 위해 Concept Factorization(CF) 알고리즘을 제안합니다. 이는 행렬 분해를 비선형 커널 공간(single non-linear kernel space)으로 확장하여 학습 능력과 적응력을 향상시킵니다.

- **Technical Details**: 새로운 알고리즘인 Globalized Multiple Kernel CF(GMKCF)를 통해, 여러 후보 커널 함수(multiple candidate kernel functions)를 동시에 입력하고, 글로벌 선형 융합(global linear fusion) 기반 CF 프레임워크 내에서 학습합니다. 이 알고리즘은 클러스터링 결과의 품질과 안정성을 향상시키면서 CF가 직면한 커널 함수 선택 문제를 해결합니다.

- **Performance Highlights**: 여러 실제 데이터베이스에 대한 실험 결과, GMKCF 알고리즘은 Kernel K-Means(KKM), Spectral Clustering(SC), Kernel CF(KCF), Co-regularized multi-view spectral clustering(Coreg), Robust Multiple KKM(RMKKM) 같은 비교 알고리즘들보다 데이터 클러스터링에서 우수한 성능을 보였습니다.



### FuseFL: One-Shot Federated Learning through the Lens of Causality with Progressive Model Fusion (https://arxiv.org/abs/2410.20380)
- **What's New**: 이 논문에서는 One-shot Federated Learning (OFL) 방법의 성능 저하를 다루고 있으며, 검증된 원인(인과적 관점)을 바탕으로 문제를 해결하기 위한 새로운 접근 방식을 제안합니다. 이를 통해 데이터의 이질성(Data Heterogeneity) 문제를 해소하고, 기존의 OFL 방법보다 더 나은 성능을 발휘하는 FuseFL을 소개합니다.

- **Technical Details**: FuseFL은 신경망(Neural Networks)을 여러 블록으로 나누고, 각 블록을 바텀업(bottom-up) 방식으로 훈련한 후 융합(Fusion)하여 특징을 증강하는 방법론입니다. 이 업데이트 및 통합 과정은 추가적인 통신 비용을 발생시키지 않으면서도 모델 성능을 크게 향상시킵니다. 특히, 각 클라이언트가 소형 모델(Small Model)을 유지하면서도 최종 모델이 원래 모델과 동일한 크기를 갖도록 설계되었습니다.

- **Performance Highlights**: FuseFL은 기존의 OFL 및 ensemble FL(앙상블 연합학습)보다 유의미한 성능 향상을 보여주었으며, 이는 클라이언트의 고도 확장성(High Scalability), 이질적인 모델 훈련(Heterogeneous Model Training), 그리고 낮은 메모리 비용(Low Memory Costs)을 지원합니다.



### Open-Vocabulary Object Detection via Language Hierarchy (https://arxiv.org/abs/2410.20371)
Comments:
          NeurIPS 2024 Camera Ready

- **What's New**: 최근 약한 감독(imagen-level supervision)을 활용한 일반화 가능한 객체 탐지(generalizable object detection)에 대한 연구가 증가하고 있습니다. 본 논문에서는 Language Hierarchical Self-training (LHST)을 통해 약하게 감독된 탐지기 훈련에 언어 계층을 도입하여 더 일반화된 탐지기를 학습합니다. 이를 통해 이미지 수준의 레이블과 박스 수준의 레이블 간의 불일치를 완화하고, 더 풍부한 감독(supervision)을 제공합니다.

- **Technical Details**: LHST는 WordNet의 언어 계층을 활용하여 이미지 수준의 레이블을 확장하고, 자기 훈련(self-training) 과정에서의 공동 정규화(co-regularization)를 가능하게 합니다. 또한, Language Hierarchical Prompt Generation (LHPG)을 설계하여 샘플링된 언어 계층을 프롬프트 생성에 적용합니다. 이는 훈련과 테스트 간의 어휘 격차(vocabulary gaps)를 메우는 데 도움을 줍니다.

- **Performance Highlights**: 제안된 기술들은 14개의 널리 연구된 객체 탐지 데이터셋에서 지속적으로 우수한 일반화 성능(generalization performance)을 보여줍니다. DetLH는 약한 감독 하의 객체 탐지에서 스스로 상기(label-to-box assignment) 및 자율 학습(self-training) 기법을 결합하여 감지 성능을 극대화합니다.



### Rethinking Data Synthesis: A Teacher Model Training Recipe with Interpretation (https://arxiv.org/abs/2410.20362)
- **What's New**: NOMAD는 데이터 생성을 위해 특별히 훈련된 모델 개발을 목표로 하며, 기존의 방법론의 한계를 극복하는 새로운 접근법을 제안합니다.

- **Technical Details**: NOMAD는 1) no-prompt-masked training과 2) 적절한 데이터 세트 크기 선택이라는 두 가지 핵심 요소를 통해 데이터 생성을 최적화합니다. 이는 일반적인 LLM 훈련과는 다른 방식으로, 모델이 고품질의 프롬프트를 학습하도록 합니다.

- **Performance Highlights**: NOMAD는 TriviaQA에서 >4%, GSM8K에서 >2%의 성능 향상을 보여주었습니다. 소규모 학습 샘플에서도 기초 모델보다 평균 1.5% 더 나은 결과를 얻었습니다.



### Conditional GAN for Enhancing Diffusion Models in Efficient and Authentic Global Gesture Generation from Audios (https://arxiv.org/abs/2410.20359)
Comments:
          Accepted by WACV 2025 (Round 1)

- **What's New**: 본 연구에서는 음성을 기반으로 한 제스처 생성의 효율성을 획기적으로 개선하는 방법을 제시합니다. 기존 DDPM(디퓨전 기반 생성 모델) 가정의 한계를 넘어서, 조건부 GAN(Generative Adversarial Network)을 도입하여 음성을 컨트롤 신호로 활용합니다.

- **Technical Details**: 조건부 GAN을 통해 다단계 샘플링 과정에서 복잡한 노이즈 분포를 모델링하며, 대규모 노이즈를 샘플링하고, 디노이징 단계를 감소시켜 고속 생성을 가능하게 합니다. 제안하는 시스템은 음성 신호를 제어 조건으로 사용하여 다양한 제스처를 효과적으로 생성 구성합니다.

- **Performance Highlights**: 제안하는 방법은 기존 DiffuseStyleGesture보다 약 12.35배 빠른 시간 내에 동작 생성이 가능하며, 현대의 디퓨전 기반 방법들에 비해 생성 효율성과 품질에서 현저히 우수한 성능을 입증했습니다.



### RopeTP: Global Human Motion Recovery via Integrating Robust Pose Estimation with Diffusion Trajectory Prior (https://arxiv.org/abs/2410.20358)
Comments:
          Accepted by WACV 2025 (Round 1)

- **What's New**: RopeTP는 강인한 pose estimation과 diffusion trajectory prior를 결합하여 비디오에서 전역적 인간 동작을 재구성하는 새로운 프레임워크입니다. 이 방법은 occluded body parts의 포즈를 정확하게 추론하기 위해 계층적 주의 메커니즘을 활용하여 상황 인식을 개선합니다.

- **Technical Details**: RopeTP는 다중 스케일의 시각적 단서를 활용하여 occluded body parts를 재구성하고, diffusion generative model을 도입하여 재구성된 joint poses에 따라 전역 모션 경로를 재추정합니다. 이 방법은 슬램(SLAM) 기반 초기 카메라 추정과 extensive 최적화에 의존하지 않고도 더 정확하고 현실적인 경로를 제공합니다.

- **Performance Highlights**: RopeTP는 3DPW 및 Human3.6M과 같은 표준 데이터셋에서 우수한 성능을 보여주며, occlusion이 있는 복잡한 시나리오에서도 특히 두드러진 성과를 도출했습니다. 실험 결과, 기존 방법론들보다 전반적으로 뛰어난 성능을 나타냈습니다.



### Dynamics as Prompts: In-Context Learning for Sim-to-Real System Identifications (https://arxiv.org/abs/2410.20357)
Comments:
          website: this https URL

- **What's New**: 이번 연구에서는 시뮬레이션 환경 매개변수를 온라인으로 동적으로 조정하여 sim-to-real (시뮬레이션에서 현실로) 격차를 줄이는 새로운 접근법을 제안합니다. 특히 in-context learning(문맥 내 학습)을 활용해 과거 상호작용 이력을 문맥으로 사용하여 시뮬레이션과 현실 간의 동적 조정을 실현합니다.

- **Technical Details**: 제안된 방법론 CAPTURE는 시뮬레이션 환경 매개변수를 실시간으로 조정하여 시뮬레이션 성능을 현실과 정렬합니다. CAPTURE는 시뮬레이션 경로, 행동, 환경 매개변수 및 실제 경로를 포함한 과거 상호작용 데이터를 기반으로 다음 토큰 예측을 사용합니다. 이를 통해 단일 단계 행동에서 매개변수 매핑에 의존하는 것이 아니라 더 포괄적인 동적 특성을 학습할 수 있습니다.

- **Performance Highlights**: 객체 스쿱과 테이블 에어 하키 작업에서 CAPTURE 방법은 시뮬레이션 간 평가에서 환경 매개변수 추정에 대해 각각 80%와 42% 성능 향상을 보였으며, 객체 스쿱 작업에서는 3개의 서로 다른 객체에 대해 최소 70%의 성공률을 달성했습니다.



### Uncovering Capabilities of Model Pruning in Graph Contrastive Learning (https://arxiv.org/abs/2410.20356)
Comments:
          MM' 24

- **What's New**: 본 연구에서는 기존의 그래프 대조 학습에서의 데이터를 증강하는 접근법 대신, 다양한 모델 버전 간의 대조를 통해 그래프 대조 학습 문제를 재정립합니다. 이를 위해, 모델 프루닝(model pruning) 방법을 활용하여 모델의 대표성을 높이고 그래프의 의미 정보를 손상시키지 않으면서 새로운 그래프 표현을 생성합니다.

- **Technical Details**: LAMP( Graph Contrastive Learning via Model Pruning) 프레임워크는 원본 그래프를 모델 입력으로 사용하여 그래프 왜곡으로 인한 의미 정보의 변화를 방지합니다. 모델의 핵심 정보를 식별하는 능력을 함양하면서, 프루닝을 통해 동적으로 변형된 그래프 인코더를 생성하고 이를 원본 인코더와 대조합니다. 또한, 노드 임베딩의 무결성을 고려하여 로컬 대조 손실(local contrastive loss)을 개발하여 모델 훈련을 방해하는 어려운 부정 샘플을 처리할 수 있습니다.

- **Performance Highlights**: LAMP는 비지도 학습 및 전이 학습을 통한 그래프 분류 실험에서 SOTA(comparative) 경쟁자를 능가하는 성능을 보였으며, 기존 데이터 증강 방법 대비 모델 프루닝의 우수성을 이론적으로 분석하여 혁신적인 기여를 하고 있습니다.



### An approach to hummed-tune and song sequences matching (https://arxiv.org/abs/2410.20352)
- **What's New**: 이 논문은 사용자가 부르는 멜로디를 사용하여 노래 제목을 찾는 작업에 대한 첫 연구입니다. Hum2Song Zalo AI Challenge 2021에 기반하여, 인간의 허밍 소리를 인식하고 노래를 검색하는 방법론을 제시합니다.

- **Technical Details**: 데이터 전처리에는 mp3에서 mel-spectrogram 형식으로 변환하고 numpy 배열로 저장하는 과정이 포함됩니다. 모델 훈련은 다양한 state-of-the-art 모델(ResNet, VGG, AlexNet, MobileNetV2)에서 수행되며, Faiss 모듈을 사용하여 허밍 소리와 일치하는 노래를 검색합니다.

- **Performance Highlights**: 공식 테스트 세트에서 MRR@10 지표로 거의 94%의 성능을 달성하며, 공개 리더보드에서 1위를 기록했습니다.



### Idempotent Unsupervised Representation Learning for Skeleton-Based Action Recognition (https://arxiv.org/abs/2410.20349)
Comments:
          ECCV 2024

- **What's New**: 이 논문에서는 새로운 골격 기반 비가역 생성 모델(IGM)을 제안하여 인식 작업에서의 성능을 향상시킵니다. 기존의 생성 모델들이 인식과 관련 없는 중복 정보를 포함하고 있다는 문제를 해결하고자 합니다.

- **Technical Details**: 우리는 이론적으로 생성 모델과 최대 엔트로피 코딩(maximum entropy coding) 간의 동등성을 보여줍니다. 이를 통해 특징들이 더 간결해지도록 대비 학습(contrastive learning)을 도입하고, 골격 데이터의 공간적 및 시간적 특징을 유지하기 위해 비가역성(idempotency) 제약을 도입합니다. 이 모델은 고차원 표현을 촉진하기 위해 인코더와 생성기 기능을 융합하는 어댑터를 채택합니다.

- **Performance Highlights**: 실험 결과, NTU 60 xsub 데이터셋에서 성능이 84.6%에서 86.2%로 향상되었습니다. 또한 제로샷(adaptation) 시나리오에서도 이전에 인식할 수 없었던 경우에서 유망한 결과를 달성하여 모델의 효율성을 입증하였습니다.



### Historical Test-time Prompt Tuning for Vision Foundation Models (https://arxiv.org/abs/2410.20346)
Comments:
          NeurIPS 2024 Camera Ready

- **What's New**: 이 논문에서는 HisTPT(History Test-time Prompt Tuning)라는 새로운 기술을 제안합니다. 이는 테스트 샘플로부터 지속적으로 학습한 정보를 기억하고, 이를 기반으로 강력한 테스트 시 프롬프트 튜닝을 가능하게 합니다.

- **Technical Details**: HisTPT는 세 가지 유형의 지식 은행(local knowledge bank, hard-sample knowledge bank, global knowledge bank)을 도입하며, 각 은행은 다른 메커니즘을 통해 지식을 효과적으로 기억하고 프롬프트를 최적화합니다. 또한, 적응형 지식 검색 메커니즘을 포함하여 각 테스트 샘플의 예측 정규화를 수행합니다.

- **Performance Highlights**: HisTPT는 이미지 분류, 의미 분할, 객체 탐지와 같은 다양한 시각적 인식 작업에서 우수한 프롬프트 튜닝 성능을 보여주며, 도메인이 지속적으로 변할 때도 일관된 성과를 보입니다.



### Maintaining Informative Coherence: Migrating Hallucinations in Large Language Models via Absorbing Markov Chains (https://arxiv.org/abs/2410.20340)
- **What's New**: 본 논문에서는 Large Language Models (LLMs)의 hallucinations 문제를 해결하기 위해 흡수 Markov Chains(AMC)를 활용한 새로운 decoding 전략을 제안합니다. 이 방법은 LLM이 정보 손실의 정도를 수량화하고 문맥 정보의 중요성을 측정하는 혁신적인 접근법을 제공합니다.

- **Technical Details**: 제안된 방법은 모든 가능한 경로를 고려하여 LLM의 텍스트 생성 과정에서 문맥 정보가 손실되는 정도를 모델링합니다. AMC 이론을 활용하여 각 토큰의 중요성을 수량화하고, 이를 바탕으로 생성되는 각 토큰의 확률 분포를 조정합니다. 이러한 방식으로 정보 흐름을 보다 명확히 파악하고, 이전 문맥으로부터 중요한 정보를 재조명하여 hallucinations을 완화합니다.

- **Performance Highlights**: TruthfulQA, FACTOR, HaluEval와 같은 데이터셋에서 수행된 평가 결과, 제안한 방법이 hallucinations을 감소시키는 데 있어 우수한 성능을 보였습니다. 이는 웹 기반 애플리케이션에서 정확한 정보 흐름을 보장하는 것이 얼마나 중요한지를 강조합니다.



### Get Large Language Models Ready to Speak: A Late-fusion Approach for Speech Generation (https://arxiv.org/abs/2410.20336)
- **What's New**: 본 논문은 TTS(텍스트-투-스피치) 시스템인 TTS-Llama를 소개하며, 이는 세밀하게 조정된 Llama 모델을 통해 가장 앞선 음성 합성 성능을 달성합니다. 또한 MoLE-Llama이라는 텍스트-음성 다중 모드 LLM을 제안하여 PEFT(모델 파라미터 효율적인 미세 조정)를 통해 구축됩니다.

- **Technical Details**: TTS-Llama는 fine-tuned Llama 3-8B-Instruct 모델을 기반으로 하며, 이를 통해 고수준의 의미론적 토큰을 생성하고, 이를 저수준의 음향 특징으로 변환하는 과정으로 음성 합성을 수행합니다. MoLE-Llama는 텍스트와 음성을 전담하는 전문가 모델을 혼합하여 하나의 통합된 다중 모달 LLM으로 구성됩니다.

- **Performance Highlights**: MoLE-Llama는 텍스트 전용 Q&A 및 TTS 작업 모두에서 경쟁력 있는 성능을 보이며, 특히 'catastrophic forgetting(재앙적 망각)' 문제를 완화하는 데 성공했습니다. 또한, 텍스트가 주어질 때 음성을 출력하는 알고리즘을 활용하여 다중 모달 대화 시스템으로서의 가능성을 보여줍니다.



### R-LLaVA: Improving Med-VQA Understanding through Visual Region of Interes (https://arxiv.org/abs/2410.20327)
Comments:
          11 pages, 7 figures, submitted to NAACL 2025

- **What's New**: 본 논문에서는 R-LLaVA를 소개하여 생물의학 시각 질문 답변(Med-VQA) 이해를 강화하고, 의학적인 주석을 이미지 공간에 통합하여 시각적인 관심 영역(Regions of Interest)을 활용합니다.

- **Technical Details**: R-LLaVA는 CLIP을 사용하여 의사의 간단한 주석(bounding boxes)을 이미지에 직접 주입함으로써, 모델 훈련 시 이러한 주석된 시각 영역을 LLaVA 모델에 넣어 더욱 빠르고 정확하게 생물의학적 질문을 처리하도록 합니다. 이는 ROI(Regions of Interest) VQA 쌍을 작성하여 기존 데이터셋을 재구성하여 모델의 시각적 이해 능력을 향상시키는 전략을 포함합니다.

- **Performance Highlights**: R-LLaVA는 네 개의 대규모 Med-VQA 데이터셋에서 기존의 최첨단 방법들보다 일관되게 우수한 성능을 보여주며, 시각적 관심 영역에 집중하는 것이 생물의학적 VQA 이해를 향상시키는 데 긍정적인 영향을 미친다는 것을 입증합니다.



### Few-shot Open Relation Extraction with Gaussian Prototype and Adaptive Margin (https://arxiv.org/abs/2410.20320)
Comments:
          30 pages, 4 figures

- **What's New**: 본 논문에서는 새로운 프레임워크 GPAM(Gaussian Prototype and Adaptive Margin)을 제안하여, UNKNOWN 클래스가 포함된 few-shot 관계 추출(Few-Shot Relation Extraction, FsRE) 문제를 해결하고자 합니다. GPAM은 Semi-factual representation, GMM-prototype metric learning과 Decision boundary learning의 세 가지 모듈로 구성되어 있습니다.

- **Technical Details**: GPAM은 Gaussian 분포와 계층적 마진을 기반으로 한 새로운 프로토타입 거리 측정 전략을 채택하여 few-shot 과적합 문제를 완화하고, 결정 경계를 더 정밀하게 학습합니다. Contrastive learning loss를 사용하여 알려진 클래스와 UNKNOWN 클래스 간의 경계를 보다 정확하게 구분합니다.

- **Performance Highlights**: FewRel 데이터셋에서의 실험 결과, GPAM은 기존의 프로토타입 방법들보다 우수한 성능을 보여주며 state-of-the-art 결과를 달성했습니다. 특히 GPAM은 few-shot 문제와 NOTA 경계 혼동 문제를 성공적으로 해결합니다.



### Deep Learning Based Dense Retrieval: A Comparative Study (https://arxiv.org/abs/2410.20315)
Comments:
          7 pages

- **What's New**: 본 연구는 Dense retrievers (밀집 검색기)가 tokenizer poisoning (토크나이저 오염)에 대한 취약성을 평가합니다. 연구의 결론에 따르면 BERT와 Dense Passage Retrieval (DPR)와 같은 감독 모델은 토크나이저가 손상되었을 때 성능 저하가 심한 반면, ANCE와 같은 비감독 모델은 상대적으로 더 강한 내성을 보입니다.

- **Technical Details**: Dense retrieval 시스템은 입력 텍스트를 수치적 토큰으로 변환하는 데 강력한 토크나이저에 크게 의존합니다. 이 연구에서는 다양한 벤치마크 데이터셋(Quora, FiQA-2018, HotpotQA)을 사용하여 supervised (감독) 및 unsupervised (비감독) 모델의 성능을 평가합니다. 성능 저하를 측정하기 위해 cosine similarity 및 여러 정보 검색 메트릭(정확도, 정밀도, 재현율 등)을 활용합니다.

- **Performance Highlights**: 작은 변형이 검색 정확도에 심각한 영향을 미칠 수 있다는 것을 발견했습니다. 특히, supervised 모델에서 성능 저하가 두드러졌으며, 이는 legal, medical, financial 등의 중요한 응용 분야에서 Dense retrieval 시스템의 신뢰성과 보안성을 보장하기 위한 강력한 방어책 필요성을 강조합니다.



### ANOMIX: A Simple yet Effective Hard Negative Generation via Mixing for Graph Anomaly Detection (https://arxiv.org/abs/2410.20310)
- **What's New**: 본 논문에서는 ANOMIX라는 새로운 프레임워크를 제안하여 GAD(Anomaly Detection) 작업에 필요한 샘플 수를 줄이는 방법을 소개합니다.

- **Technical Details**: ANOMIX는 (1) ANOMIX-M이라는 새로운 그래프 믹싱(Graph Mixing) 접근 방식을 적용하여 하드 네거티브(hard negatives)를 생성하고, (2) 노드(node) 및 서브그래프(subgraph) 레벨의 대조(Contrasts)를 통해 기본적인 이상치를 구별합니다.

- **Performance Highlights**: ANOMIX는 AUC(Area Under the Curve)에서 최대 5.49% 향상되었으며, 속도는 1.76% 빨라졌습니다. GCL(Graph Contrastive Learning)에서 필요한 샘플 수를 거의 80% 줄이는 성과를 보여주었습니다.



### Enhancing Community Vision Screening -- AI Driven Retinal Photography for Early Disease Detection and Patient Trus (https://arxiv.org/abs/2410.20309)
Comments:
          11 pages, 4 figures, published in MICCAI2024 OMIA XI workshop

- **What's New**: 이 논문은 rural 지역 사회를 위한 새로운 커뮤니티 비전 스크리닝 솔루션인 ECVS (Enhancing Community Vision Screening)를 소개합니다. 이는 비침습적(retinal photography) 기술을 기반으로 하여 시각 장애를 식별하고, 효과적으로 환자를 치료를 위해 의뢰하는 과정을 단순화합니다.

- **Technical Details**: ECVS는 4가지 딥러닝 모델을 활용합니다: RETinal photo Quality Assessment (RETQA), Pathology Visual Impairment detection (PVI), Eye Disease Diagnosis (EDD), 그리고 Visualization of Lesion Regions of the eye (VLR). 이 모델들은 80,000개 이상의 fundus 사진을 사용하여 학습되었으며, 각각 AUC 스코어 0.98(RETQA), 0.95(PVI), 0.90(EDD) 및 DICE 계수 0.48(VLR)을 달성했습니다.

- **Performance Highlights**: ECVS는 기존 검진 프로세스와 비교해 검사 횟수를 5회에서 2회로 줄이고, 총 대기 시간을 40분에서 5분으로 단축시킵니다. 또한 환자 의뢰 시간을 2-4주에서 10-20분으로 대폭 감소시킴으로써 효율성을 크게 개선했습니다.



### Sequential Large Language Model-Based Hyper-Parameter Optimization (https://arxiv.org/abs/2410.20302)
- **What's New**: 이번 연구에서는 하이퍼파라미터 최적화(hyperparameter optimization, HPO)를 위해 대규모 언어 모델(large language models, LLMs)을 활용하는 혁신적인 프레임워크인 SLLMBO를 소개합니다. SLLMBO는 동적 탐색 공간 적응(dynamic search space adaptability), 향상된 파라미터 환경 활용(parameter landscape exploitation), 그리고 새로운 하이브리드 LLM-트리 구조 파젠 추정기(LLM-Tree-structured Parzen Estimator, LLM-TPE) 샘플러를 통합하고 있습니다.

- **Technical Details**: SLLMBO는 최근의 완전 LLM 기반 방법들과 전통적인 베이지안 최적화(Bayesian Optimization, BO)의 한계를 극복하여 보다 강력한 최적화를 달성합니다. 이번 연구에는 GPT-3.5-turbo, GPT-4o, Claude-Sonnet-3.5, Gemini-1.5-flash를 포함한 여러 LLM에 대한 포괄적인 벤치마킹이 이루어졌으며, 이는 HPO를 위한 다양한 LLM을 평가한 최초의 프레임워크로 자리잡고 있습니다.

- **Performance Highlights**: 14개의 표 형식(tabular) 작업에서 LLM-TPE 샘플러는 완전 LLM 기반 방법보다 뛰어난 성과를 나타내었으며, 9개 작업에서 BO 방법보다 우수한 결과를 달성했습니다. 예산 제약이 있는 시나리오에서의 조기 중단 테스트는 경쟁력 있는 성능을 추가로 입증하여 LLM 기반 방법들이 최적 결과를 위해서는 반복(iteration)을 연장해야 함을 보여주었습니다.



### Learning from Response not Preference: A Stackelberg Approach for LLM Detoxification using Non-parallel Data (https://arxiv.org/abs/2410.20298)
- **What's New**: 본 연구에서는 비정렬(non-parallel) 데이터를 사용하여 대형 언어 모델(LLM)을 텍스트 디톡스화(detoxification) 리라이터로 변환하는 최적화 방법인 Stackelberg Response Optimization(SRO)을 제안합니다. 일반적인 바탕으로, LLM과 독성 스크리너(toxicity screener) 간의 Stackelberg 게임 모델링이 이루어집니다.

- **Technical Details**: LLM의 미세 조정 과정은 LLM(리더)과 이진 스타일 분류기(독성 또는 비독성)인 독성 스크리너(팔로워) 간의 Stackelberg 게임으로 모델링됩니다. SRO에서는 LLM이 독성 텍스트를 기반으로 리라이트를 생성하고, 스크리너가 성공 또는 실패의 결과를 제공하여 LLM이 스타일을 조정하게 됩니다. SRO는 독성 콘텐츠와 그 리라이트 쌍을 사용할 때 DPO(Direct Preference Optimization)에 따라 학습합니다.

- **Performance Highlights**: SRO로 미세 조정된 LLM은 스타일 정확도, 내용 유사성 및 유창성에서 기존의 최첨단 모델과 비슷한 만족스러운 성능을 달성하며, 전체 디톡스화 성능은 다른 방법들을 초월하여 인간 참조와 일치합니다. 실험적으로 SRO의 민감성이 스크리너의 피드백에 큰 영향을 받는다는 추가 증거도 확인되었습니다.



### Fine-Tuning and Evaluating Open-Source Large Language Models for the Army Domain (https://arxiv.org/abs/2410.20297)
- **What's New**: 최근 대규모 언어 모델(LLMs)의 군사 도메인에서의 활용 가능성에 대한 연구가 활발히 이루어지고 있습니다. 본 논문에서는 Army 도메인에 적합한 오픈소스 LLM을 조정하기 위한 노력으로 TRACLM이라는 LLM 계열을 제안합니다.

- **Technical Details**: 본 연구에서 소개하는 TRACLM은 The Research and Analysis Center(TRAC)와 Army Futures Command(AFC)에서 조정된 LLM의 세 가지 세대를 포함하고 있으며, 각 세대의 모델은 훈련 프로세스의 지속적인 개선을 거쳐 Army 작업과 사용 사례에 적용될 때 능력이 향상되었습니다. 또한 MilBench라는 평가 프레임워크를 개발하여 LLM의 Army 도메인 지식을 정량적으로 평가할 수 있도록 하였습니다.

- **Performance Highlights**: TRACLM 모델은 Army 특정 작업에서 성능이 향상된 것으로 나타났습니다. MilBench는 LLM의 평가를 위한 확장 가능하고 효율적인 소프트웨어 프레임워크로, 다양한 DoD(Department of Defense)에서의 활용 가능성을 높입니다.



### AI-Driven Cyber Threat Intelligence Automation (https://arxiv.org/abs/2410.20287)
Comments:
          11 pages

- **What's New**: 이 연구는 Microsoft의 AI 기반 보안 기술을 활용하여 산업 환경에서 사이버 위협 인텔리전스(Cyber Threat Intelligence, CTI) 프로세스를 자동화하는 혁신적인 접근 방식을 소개합니다. 전통적으로 CTI는 수동적인 방법론에 의존하여 데이터를 수집하고 분석하였지만, 본 연구에서는 GPT-4o와 고급 원샷 파인튜닝 기법을 통해 CTI 자동화 솔루션을 제공합니다.

- **Technical Details**: CTI 생성의 전통적인 방법은 많은 수작업이 필요하며, 따라서 비효율적이고 인적 오류가 발생하기 쉬운 구조입니다. 본 연구는 Microsoft Copilot for Security(MCS), Logic Apps, Azure AI와 같은 AI 기반 제품을 통해 CTI 자동화를 실현하며, 데이터 수집, 상관 관계 분석 및 초기 분석 같은 작업의 자동화를 목표로 합니다.

- **Performance Highlights**: 실험 결과, 제안된 아키텍처는 수동 작업을 줄이는 동시에 최종 CTI 보고서의 정밀도를 유지했습니다. AI 기반 시스템을 활용하여 실시간으로 대량의 위협 데이터를 수집하고 패턴을 상관 관계 분석할 수 있어, 위협 발견과 신속한 대응 속도가 개선되었습니다.



### MarDini: Masked Autoregressive Diffusion for Video Generation at Sca (https://arxiv.org/abs/2410.20280)
Comments:
          Project Page: this https URL

- **What's New**: MarDini 모델은 비디오 생성의 효율성을 극대화하기 위한 새로운 비디오 확산 모델로, 마스크 자동 회귀(MAR)와 확산 모델(DM)의 장점을 통합하여 경량화된 네트워크 설계를 적용합니다. 이 모델은 다양한 마스킹 전략을 통해 비디오 생성 작업을 유연하게 처리할 수 있으며, 비디오 인터폴레이션, 이미지-비디오 변환 및 비디오 확장 등 다양한 작업을 지원합니다.

- **Technical Details**: MarDini는 MAR 플래닝 모델과 경량화된 DM으로 구성된 비대칭 네트워크 아키텍처를 가지고 있습니다. MAR 모델은 저해상도 입력 프레임을 사용하여 계획 신호를 생성하며, DM은 이 신호를 기반으로 한 고해상도 비디오 프레임을 생성합니다. 효율적인 훈련 배치와 프로그레시브 트레이닝 전략을 통해 MAR와 DM을 병렬로 훈련시키며, 일반적인 텍스트-이미지 또는 텍스트-비디오 사전 학습에 대한 의존 없이 비디오 생성 모델을 훈련할 수 있습니다.

- **Performance Highlights**: MarDini는 몇 가지 추론 단계 내에 탁월한 성능을 보이며, 기존의 고비용 이미지-비디오 모델과 동등한 수준의 비디오 생성을 효율적으로 수행합니다. 이 모델은 비디오 인터폴레이션에서 새로운 최첨단 성능을 설정하며, 메모리 사용량이 낮아 복잡한 모션 동역학을 효과적으로 모델링할 수 있는 가능성을 나타냅니다.



### EfficientEQA: An Efficient Approach for Open Vocabulary Embodied Question Answering (https://arxiv.org/abs/2410.20263)
- **What's New**: 새로운 연구에서는 Embodied Question Answering (EQA)을 위한 효율적인 프레임워크인 EfficientEQA를 제안했습니다. 이 프레임워크는 로봇이 개방형 어휘 환경에서 빠르고 정확하게 질문에 답할 수 있도록 합니다.

- **Technical Details**: EfficientEQA는 Semantic-Value-Weighted Frontier Exploration 전략을 사용하여 로봇이 중요한 정보를 기반으로 환경을 탐색하도록 합니다. 또한, Retrieval-Augmented Generation (RAG) 기술을 통해 이미지 검색과 VLM(비전-언어 모델) 추론을 활용하여 미리 정의된 답변 선택지에 의존하지 않고 답변을 생성합니다.

- **Performance Highlights**: 실험 결과, EfficientEQA는 기존 상태-최적화 방법 대비 15% 이상의 정확도 향상과 20% 이상의 효율성을 보여주었습니다.



### Equivariant Blurring Diffusion for Hierarchical Molecular Conformer Generation (https://arxiv.org/abs/2410.20255)
Comments:
          NeurIPS 2024

- **What's New**: 이 논문은 다단계(multiscale) 관점으로 3D 기하학을 처리하기 위한 새로운 방법론을 제시하며, 특히 분자 그래프에 기반한 3D 분자 구조 생성의 기초적인 생화학적 문제를 해결하는 데 중점을 둡니다.

- **Technical Details**: 저자들은 두 단계의 계층적 접근 방식을 채택하여, 첫 번째로 분자 그래프에서 굵게 표현된(fragment-level) 3D 구조를 생성하고, 두 번째로 그런 구조에서 원자 수준의 세부 사항을 동시에 조정하며 생성합니다. 이를 위해 'Equivariant Blurring Diffusion (EBD)'라는 새로운 생성 모델을 도입하고, 이 모델은 분자 구조의 블러링과 필드 기반 이론에 기반해 SE(3) 등가성을 유지합니다. EBD는 원자 위치를 각각의 단편 중심으로 이동시키고 미세 원자 세부 사항을 복원하는 역과정을 포함합니다.

- **Performance Highlights**: EBD 모델은 약물 같은 분자를 기준으로 한 성능 테스트에서 기존의 최첨단 denoising diffusion 모델보다 우수한 성과를 보였으며, 100배 적은 확산 시간 단계로도 효과적인 결과를 도출했습니다. 이 논문은 EBD의 설계에서 손실 함수와 데이터 변형 과정의 중요성을 강조하며, 그 효과에 대한 철저한 분석을 제공합니다.



### Adaptive Video Understanding Agent: Enhancing efficiency with dynamic frame sampling and feedback-driven reasoning (https://arxiv.org/abs/2410.20252)
- **What's New**: 이번 연구에서는 장시간 비디오 콘텐츠 이해의 효율성과 효과성을 향상시키기 위해 대규모 언어 모델(LLMs)을 활용한 에이전트 기반 접근법을 제안합니다. 주요 기법으로는 쿼리 적응형 프레임 샘플링(query-adaptive frame sampling)을 도입하여, 관련성 높은 프레임만을 실시간으로 처리하여 기존 방법의 단점을 극복하고자 했습니다.

- **Technical Details**: 제안하는 방법은 LLM 기반의 에이전트가 특정 문맥과 쿼리에 기반하여 동적으로 샘플 프레임을 결정하도록 설계되었습니다. 자가 반영(self-reflective) 능력을 활용하여 에이전트의 추론 능력을 강화하고, 이전 경험을 저장하고 활용할 수 있는 장기 기억(long-term memory) 기능도 통합하였습니다.

- **Performance Highlights**: 여러 비디오 이해 벤치마크에서 평가한 결과, 제안한 방법이 기존 방법들보다 더 높은 정확성과 더 낮은 프레임 액세스를 기록하며 성능을 향상시키는 것을 확인했습니다.



### Improving Model Evaluation using SMART Filtering of Benchmark Datasets (https://arxiv.org/abs/2410.20245)
Comments:
          20 pages, 5 figures

- **What's New**: 본 연구는 NLP(Natural Language Processing) 모델 평가의 효율성을 높이기 위한 새로운 접근 방식을 제안합니다. 기존의 벤치마크 데이터셋에서 저품질의 예제를 체계적으로 제거하여 고품질의 예제를 선별하는 SMART(Selection Methodology for Accurate, Reduced, and Targeted) 필터링 방법론을 도입합니다.

- **Technical Details**: SMART 필터링은 세 가지 기준에 따라 작동합니다: (i) 쉬운 예제 제거, (ii) 데이터 오염이 의심되는 예제 제거, (iii) 임베딩 공간에서 서로 유사한 예제 제거. 이를 통해 모델의 상대 순위를 보존하면서 데이터셋의 크기를 평균 48% 줄이고, ChatBot Arena에서의 Pearson 상관계수를 높일 수 있음을 보여줍니다.

- **Performance Highlights**: SMART 필터링을 사용하여 데이터셋을 약 68.9% 줄이면서도 모델의 상대적 순위에 미치는 영향을 최소화하는 성과를 보였습니다. 필터링된 데이터셋은 ChatBot Arena 등에서의 인간 평가와 보다 강한 상관관계를 보이며, 이는 더 나은 인간 선호도와 일치한 결과입니다.



### A Survey of Large Language Models for Arabic Language and its Dialects (https://arxiv.org/abs/2410.20238)
- **What's New**: 이번 논문은 아랍어 및 그 방언을 위한 대규모 언어 모델(Large Language Models, LLMs)에 대한 종합적인 개요를 제공합니다.

- **Technical Details**: 다양한 아키텍처(architecture), 즉 인코더 전용(encoder-only), 디코더 전용(decoder-only), 인코더-디코더 모델을 포함하며, 고전 아랍어(Classical Arabic), 현대 표준 아랍어(Modern Standard Arabic), 방언 아랍어(Dialectal Arabic)를 위한 사전 학습(pre-training) 데이타셋(datasets)을 다룹니다. 단일 언어( монолинг구ал), 이중 언어(bilingual), 다중 언어(multilingual) LLMs의 아키텍처와 성능을 감성 분석(sentiment analysis), 개체명 인식(named entity recognition), 질문 응답(question answering)과 같은 다운스트림 작업에서 분석합니다.

- **Performance Highlights**: 논문은 아랍어 LLM의 개방성(openness)을 소스코드(source code) 가용성, 학습 데이터(training data), 모델 가중치(model weights), 문서화(documentation)와 같은 요인을 바탕으로 평가합니다. 더불어 다양한 방언 데이터셋의 필요성과 연구 재현성(reproducibility) 및 투명성(transparency)을 위한 개방성의 중요성을 강조합니다. 또한 향후 연구를 위한 주요 도전(challenges)과 기회(opportunities)를 확인하고, 보다 포괄적이고 대표적인 모델의 필요성을 강조합니다.



### Modelling of Economic Implications of Bias in AI-Powered Health Emergency Response Systems (https://arxiv.org/abs/2410.20229)
- **What's New**: 인공지능(AI) 기반 비상 대응 시스템에서 편향의 경제적 영향을 평가하는 이론적 프레임워크를 제시합니다. 이 프레임워크는 건강 경제학, 복지 경제학 및 인공지능을 통합하여 알고리즘 편향이 자원 배분, 건강 결과 및 사회 복지에 미치는 영향을 분석합니다.

- **Technical Details**: AI 모델의 알고리즘 편향은 불균형한 교육 데이터, 결함이 있는 알고리즘 등 다양한 원인으로 발생합니다. 이 연구는 건강 생산 함수(health production function)와 사회 복지 함수(social welfare function)를 포함한 경제 모델을 사용하여 편향이 인구 집단에 미치는 영향을 정량화합니다.

- **Performance Highlights**: 편향이 있는 AI 시스템에서 비효율적인 자원 활용은 질병률 및 사망률 증가로 이어질 수 있으며, 이는 사회에 상당한 경제적 비용을 초래합니다. 연구의 결과는 정책 입안자와 기술 개발자에게 공정하고 효율적인 AI 시스템의 필요성을 강조합니다.



### Neural Fields in Robotics: A Survey (https://arxiv.org/abs/2410.20220)
Comments:
          20 pages, 20 figures. Project Page: this https URL

- **What's New**: 본 논문은 Neural Fields(NF)가 로봇 공학에서 3D 장면 표현의 획기적인 방법으로 자리 잡고 있음을 강조합니다. 특히, NF의 연속적이고 미분 가능한 표현 방식이 다양한 센서 데이터의 통합 및 새로운 시점 생성에 어떻게 기여하는지를 설명합니다.

- **Technical Details**: Neural Fields는 Occupancy Networks, Signed Distance Fields, Neural Radiance Fields, Gaussian Splatting과 같은 네 가지 주요 프레임워크를 기반으로 합니다. NF는 RGB 카메라, LiDAR, 깊이 센서 등에서 수집된 데이터를 기반으로 최적화되어 고품질 3D 재구성을 생성합니다.

- **Performance Highlights**: Neural Fields는 고품질 3D 재구성, 다중 센서 데이터 통합, 연속적인 컴팩트 표현, 일반화 및 적응력을 제공하여 로봇의 탐색, 물체 조작, 자율 주행 등에서 성능을 크게 향상시킵니다. 또한, NF의 발전은 생성형 AI와 로봇 간의 중요한 연결 고리를 제공하여 이 분야의 연구가 급속히 성장하고 있음을 보여줍니다.



### Generative AI in Health Economics and Outcomes Research: A Taxonomy of Key Definitions and Emerging Applications, an ISPOR Working Group Repor (https://arxiv.org/abs/2410.20204)
Comments:
          36 pages, 1 figure, 2 tables

- **What's New**: 이 논문은 건강 경제학 및 결과 연구(HEOR)에서의 생성적 인공지능(generative AI)의 분류법을 제공하고, 그 응용 분야와 AI 생성물의 정확성 및 신뢰성을 향상시키기 위한 방법들을 제시합니다.

- **Technical Details**: 이 обзор(Review)에서는 생성적 AI의 기본 개념을 정의하고, 현재 HEOR 분야에서의 응용 예시로는 체계적인 문헌 리뷰(systematic literature reviews), 건강 경제 모델링(health economic modeling), 실제 증거 생성(real-world evidence generation), 및 서류 개발(dossier development)을 포함합니다. AI의 정확성 및 신뢰성을 개선하기 위한 접근 방식으로는 프롬프트 엔지니어링(prompt engineering, zero-shot, few-shot, chain-of-thought, persona pattern prompting), 검색 증강 생성(retrieval-augmented generation), 모델 미세 조정(model fine-tuning), 그리고 도메인 특정 모델(domain-specific models)의 사용이 소개됩니다.

- **Performance Highlights**: 생성적 AI는 HEOR에서 효율성, 생산성을 높이고 복잡한 문제에 대한 새로운 해결책을 제시하는 데 중요한 잠재력을 보여줍니다. 기본 모델들(foundation models)은 복잡한 작업의 자동화에 있어 유망하지만, 과학적 신뢰성(scientific reliability), 편향(bias), 해석 가능성(interpretability), 및 작업 흐름 통합(workflow integration) 등에서 여전히 과제가 남아 있습니다. 이 논문은 이러한 AI 도구들의 정확성을 개선하기 위한 전략에 대해 논의합니다.



### Physics informed Shadowgraph Density Field Reconstruction (https://arxiv.org/abs/2410.20203)
- **What's New**: 이 연구는 그림자 그래프(shadowgraph) 이미지를 사용하여 밀도 필드를 재구성하는 새로운 접근 방식을 제시합니다. 물리 정보 기반 신경망(Physics-Informed Neural Networks, PINNs)과 전통적인 그림자 그래프 이미징 기술을 통합하여 복잡한 유동(field) 내에서 굴절률의 변화를 효과적으로 포착합니다.

- **Technical Details**: 제안된 방법은 그림자 그래프 촬영의 고유한 도전 과제인 노이즈(noise)와 제한된 공간 해상도(spatial resolution)를 해결합니다. 이 결과로 유체 역학(fuid dynamics)의 정확한 시각화(visualization)가 가능합니다. 실험 결과는 재구성된 밀도 필드와 실험 측정값 간에 상당한 일치를 보여줍니다.

- **Performance Highlights**: 이 연구는 유체 역학에서 비침습적(non-intrusive) 진단(diagnostic) 기술의 발전에 기여하며, 다양한 응용(application)에서 흐름 구조(flow structure)에 대한 이해를 향상시킵니다.



### Uncertainty-Penalized Direct Preference Optimization (https://arxiv.org/abs/2410.20187)
Comments:
          Accepted at the NeurIPS 2024 FITML Workshop

- **What's New**: 이번 연구에서는 DPO(Direct Preference Optimization)의 과최적화 문제를 해결하기 위해, 불확실성 패널라이제이션(uncertainty penalization) 방안을 도입한 비관적(pessimistic) 프레임워크를 개발하였습니다.

- **Technical Details**: DPO는 인공지능 모델에서 선호 데이터의 가능성을 극대화하여 정책을 미세 조정하는 기법으로, 잘못 라벨링된 또는 모호한 선호 쌍들로부터 오는 패널라이징 효과를 제시합니다. 이 연구는 평가 데이터인 Anthropic-HH 데이터셋을 이용하여 GPT2 Medium 모델의 앙상블을 통해 수행합니다.

- **Performance Highlights**: 제안된 방법은 기존의 DPO보다 전반적으로 향상된 성능을 보였으며, 높은 불확실성을 가지는 응답에서 더 나은 완성을 보여주었습니다.



### Chemical Language Model Linker: blending text and molecules with modular adapters (https://arxiv.org/abs/2410.20182)
Comments:
          25 pages, 3 figures

- **What's New**: ChemLML(Chemical Language Model Linker)이라는 경량 어댑터 기반 전략을 제안하여 다중 모드 모델을 기존의 고품질 사전 훈련된 모델을 활용하지 않고도 훈련할 수 있는 접근 방식을 변화시킵니다. ChemLML은 두 개의 독립적인 도메인 모델을 혼합하여 텍스트 설명으로부터 조건부 분자 생성을 달성합니다.

- **Technical Details**: ChemLML은 SMILES(Simplified Molecular Input Line Entry System)와 SELFIES(Self-referencing Embedded Strings)로 표현된 분자를 사용하며, 이 둘의 선택이 조건부 분자 생성 성능에 큰 영향을 미친다는 점을 발견했습니다. ChemLML은 텍스트 모델과 분자 생성 모델 간의 상호작용을 탐구하는 유연한 접근 방식을 제공합니다.

- **Performance Highlights**: ChemLML은 적은 수의 어댑터 매개변수만으로 훈련되며, 텍스트 지향 분자 설계 작업에서 동일한 크기의 모델에 비해 훨씬 적은 훈련 가능한 매개변수를 필요로 하면서도 뛰어난 성능을 달성합니다. 실제 사용 사례에서는 ChemLML을 통해 후보 단백질 억제제를 생성하고 도킹(docking)을 통한 품질 평가를 실시했습니다.



### A Stack-Propagation Framework for Low-Resource Personalized Dialogue Generation (https://arxiv.org/abs/2410.20174)
Comments:
          published as a journal paper at ACM Transactions on Information Systems 2023. 35 pages, 5 figures

- **What's New**: 이번 연구는 제한된 교육 데이터를 활용하여 일관성 있는 응답 생성을 위한 새로운 접근 방식을 제안합니다. 기존의 전통적인 모델과는 달리, 새로운 stack-propagation framework를 통해 트랜스포머(Transformer) 인코더와 디코더를 활용하여 퍼스나(persona) 이해를 규제화(regularization)하는 방법으로 응답을 생성합니다.

- **Technical Details**: 제안된 stack-propagation framework는 하나의 트랜스포머 인코더와 두 개의 트랜스포머 디코더로 구성되어 있습니다. 첫 번째 디코더는 응답 생성을 모델링하고, 두 번째 디코더는 응답 생성과 일관성 이해를 동시에 모델링하여 규제 역할을 수행합니다. 이를 통해, 모델은 보다 적은 개인화된 대화 데이터로부터 학습하면서도 경쟁력 있는 성능을 유지할 수 있습니다.

- **Performance Highlights**: 주어진 다양한 저자원(low-resource) 환경에서, 주관적 및 객관적 평가를 통해 제안된 stack-propagation framework가 응답 품질 및 퍼스나 일관성에서 기존의 강력한 기준 모델들보다 우수한 성능을 보이며, 퍼스나-밀집(dialogue data)가 아닌 데이터에 대한 의존성을 극복하는 데 성공했습니다.



### Diff-CXR: Report-to-CXR generation through a disease-knowledge enhanced diffusion mod (https://arxiv.org/abs/2410.20165)
- **What's New**: 이번 연구에서는 질병 지식이 강화된 Diffusion 기반의 텍스트-이미지(TTI) 생성 프레임워크인 Diff-CXR를 제안한다. 기존의 방법들은 의료 데이터의 고유 특성 때문에 생성 성능이 제한적이었으나, Diff-CXR은 이를 개선하기 위해 노이즈 필터링 전략과 비전-언어 모델을 통합하여 보고서에서 Chest-Xray(CXR) 생성을 가능하게 한다.

- **Technical Details**: Diff-CXR는 Latent Noise Filtering Strategy(LNFS)를 통해 노이즈 데이터를 효과적으로 제거하고, Adaptive Vision-Aware Textual Learning Strategy(AVA-TLS)를 통해 도메인 특화된 비전-언어 모델에서 중요한 보고서 임베딩을 학습한다. 또한, 질병 지식을 주입하기 위한 메커니즘을 통해 모델의 성능을 개선한다고 설명된다.

- **Performance Highlights**: 실험 결과, Diff-CXR는 MIMIC-CXR와 IU-Xray에서 FID 점수와 mAUC 점수에서 기존 SOTA 의료 TTI 방법들보다 각각 33.4% / 8.0% 및 23.8% / 56.4% 성능을 개선하며, 29.641 GFLOPs의 가장 낮은 계산 복잡도로 효율적인 결과를 보여준다.



### Causal Abstraction in Model Interpretability: A Compact Survey (https://arxiv.org/abs/2410.20161)
- **What's New**: 최근 인공지능(AI)의 해석 가능성을 위한 연구가 진행되고 있으며, 그 중에서도 causal abstraction(인과적 추상화)라는 이론적框架(프레임워크)가 주목받고 있습니다. 이 방법은 모델의 행동 뒤에 있는 인과적 메커니즘을 이해하고 설명하기 위한 체계적인 접근법을 제공합니다. 이 논문은 causal abstraction의 이론적 기초, 실제 응용 및 모델 해석 가능성에 대한 함의를 탐구합니다.

- **Technical Details**: Causal abstraction은 구조적 방정식 모델을 기반으로 하여 변수들 간의 인과 관계를 표현하는 함수 집합을 사용합니다. 이 연구는 모델 간 인과적 동등성을 판단하기 위한 수학적 기본을 제공하며, 최근 연구에서 neural networks(신경망 모델)에 대한 인과적 추상화 기술을 적용하여 내부 벡터와 모델 출력 행동 간의 잠재적 인과 관계를 분석합니다.

- **Performance Highlights**: Causal abstraction은 복잡한 머신러닝 모델의 해석을 향상시키고, 모델의 결정 과정을 보다 투명하고 책임감 있게 설명할 수 있도록 도와줍니다. 이 접근법은 기계적 해석 가능성(mechanistic interpretability)에 대한 관심이 고조되는 가운데, 모델의 내부 작동을 이해하는 데 기여하고 있습니다.



### AdaNeg: Adaptive Negative Proxy Guided OOD Detection with Vision-Language Models (https://arxiv.org/abs/2410.20149)
Comments:
          NIPS 2024 Camera Ready, Codes are available at \url{this https URL}

- **What's New**: 본 연구에서는 OOD(out-of-distribution) 샘플을 효과적으로 식별하기 위한 새로운 방법으로 	extit{adaptive negative proxies}를 도입합니다. 이는 실제 OOD 이미지를 탐색하여 테스트 중에 동적으로 생성되는 부정 프록시입니다.

- **Technical Details**: 우리의 접근법은 테스트 이미지에서 차별적인 특징을 캐시하는 feature memory bank를 활용하여 OOD 데이터셋에 대해 더 잘 정렬된 프록시를 생성하는 방법을 제안합니다. task-adaptive proxies 및 sample-adaptive proxies를 통해 각각 특징적인 데이터셋과 샘플 수준의 세부 정보를 캡처합니다.

- **Performance Highlights**: AdaNeg는 대규모 ImageNet 벤치마크에서 2.45% AUROC 향상과 6.48% FPR95 감소를 달성하며, 기존 방법들에 비해 우수한 성능을 보여줍니다. 우리의 방법은 training-free 및 annotation-free를 유지하며 빠른 테스트 속도를 자랑합니다.



### Exploring Welfare Maximization and Fairness in Participatory Budgeting (https://arxiv.org/abs/2410.20143)
Comments:
          PhD Thesis

- **What's New**: 본 논문은 Participatory budgeting (PB) 모델의 복지(welfare) 및 공정성(fairness) 관련 목표를 연구하였습니다. 새로운 PB 규칙을 제안하고 공정성을 촉진하는 방법을 탐구하여 기존 문헌의 공백을 메우는 기여를 합니다.

- **Technical Details**: 논문은 두 부분으로 나누어집니다. 첫 번째 부분은 이분법적(dichotomous) 선호를, 두 번째 부분은 서열적(ordinal) 선호를 중심으로 다룹니다. 각 부분은 두 가지 경우를 고려합니다: (i) 프로젝트의 비용이 단일 값으로 제한되고 부분 자금지원(partial funding)이 허용되지 않는 경우, (ii) 프로젝트의 비용이 유연하고 여러 값을 가질 수 있는 경우입니다.

- **Performance Highlights**: 제안된 새로운 PB 규칙은 복지 극대화(maximize welfare)와 공정성 증진(promotion of fairness)을 목표로 하여, PB 모델의 다양한 응용에서 효과적입니다.



### Mask-based Membership Inference Attacks for Retrieval-Augmented Generation (https://arxiv.org/abs/2410.20142)
- **What's New**: 이 논문에서는 Mask-Based Membership Inference Attacks (MBA) 프레임워크를 제안하여 Retrieval-Augmented Generation (RAG) 시스템에서의 Membership Inference Attacks (MIAs) 문제를 해결합니다. 이는 특정 단어를 마스킹하여 RAG 시스템의 신뢰성을 높이는 방법입니다.

- **Technical Details**: MBA 프레임워크는 마스킹 알고리즘을 통해 대상 문서에서 특정 단어를 효과적으로 마스킹한 후, 이를 RAG 시스템에 제공하여 마스크 값을 예측하도록 합니다. 마스크 예측의 정확도를 분석하여 해당 문서가 데이터베이스에 포함되는지를 판단합니다. M𝑀은 마스킹할 단어 수를, γ는 예측 정확도를 판단하는 데 사용하는 하이퍼파라미터입니다.

- **Performance Highlights**: 논문에서 제안한 MBA 프레임워크는 세 가지 공개 QA 데이터셋에서 성능을 평가하였고, 기존 방법들에 비해 ROC AUC 값이 20% 이상 향상되었습니다.



### On-Site Precise Screening of SARS-CoV-2 Systems Using a Channel-Wise Attention-Based PLS-1D-CNN Model with Limited Infrared Signatures (https://arxiv.org/abs/2410.20132)
- **What's New**: 이번 연구에서는 저희가 개발한 새로운 방법론을 소개합니다. 이 방법론은 attenuated total reflection-Fourier transform infrared spectroscopy (ATR-FTIR)와 adaptive iteratively reweighted penalized least squares (airPLS) 전처리 알고리즘, 그리고 채널 기반 주의 메커니즘을 따르는 partial least squares one-dimensional convolutional neural network (PLS-1D-CNN) 모델을 통합하여 감염된 개인을 10분 내에 정확하게 선별할 수 있도록 합니다.

- **Technical Details**: 본 연구는 SARS-CoV-2의 초기 발병 단계에서 공개 건강을 위해 제한된 nasopharyngeal swabs을 효율적으로 활용하는 방법을 제안했습니다. ATR-FTIR 신호의 질을 평가하기 위한 biomolecular importance (BMI) 평가 방법과, airPLS를 통한 신호 전처리 및 PLS-1D-CNN 모델을 사용하여 SARS-CoV-2의 감염 상태를 판별합니다. 실험 결과, 저희 모델은 호흡 바이러스 스펙트럼 감지 분야에서 최근 보고된 방법들보다 뛰어난 성과를 거두었습니다.

- **Performance Highlights**: 본 연구에서 개발한 모델은 96.48%의 인식 정확도, 96.24%의 민감성, 97.14%의 특이성, 96.12%의 F1-score, 0.99의 AUC를 달성하여 WHO가 제시한 SARS-CoV-2 감염 검사를 위한 민감도 및 특이성 기준을 충족했습니다.



### Estuary: A Framework For Building Multimodal Low-Latency Real-Time Socially Interactive Agents (https://arxiv.org/abs/2410.20116)
Comments:
          To be published in ACM Intelligent Virtual Agents (IVA) 2024 [DOI: https://doi.org/10.1145/3652988.3696198] [ACM ISBN: 979-8-4007-0625-7/24/09]

- **What's New**: 이 논문에서는 Estuary라는 새로운 멀티모달 프레임워크를 제안하여, 사회적 상호작용 에이전트(SIAs)의 개발을 촉진하고자 합니다. 이 프레임워크는 텍스트, 오디오 및 곧 비디오를 포함한 다양한 입력을 지원하며, 클라우드 의존성을 줄이고, 재현성과 구성 가능성을 극대화합니다.

- **Technical Details**: Estuary는 비동기 및 병렬 처리 기능을 갖춘 마이크로서비스 아키텍처를 기반으로 하며, ASR(Automatic Speech Recognition), TTS(Text-To-Speech), 대화 관리 및 LLM(Large Language Models)을 통합합니다. 이를 통해 사용자 정의 모듈을 원활하게 통합하여 SIA를 구축할 수 있는 파이프라인을 제공합니다. Estuary는 SocketIO 프로토콜을 사용하여 클라이언트와 호스트 장치 간의 연결을 지원하고, 다양한 플랫폼에서 실행할 수 있는 하드웨어 비독립성을 제공합니다.

- **Performance Highlights**: Estuary는 실제 사용자 대화 종료 후 SIA의 TTS 모듈이 첫 발화를 생성하는 데 평균 1.2에서 2.5초의 지연 시간을 달성하였습니다. 이는 ChatGPT-4의 평균 2.8초 지연 시간에 비해 개선된 결과로, LLM과 TTS 응답을 동시에 스트리밍하여 가능했습니다. 또한, 오프 클라우드 마이크로서비스를 통한 데이터의 보안성을 높이고, 반복 가능한 동일한 버전의 마이크로서비스를 유지할 수 있어 실험의 재현성을 보장합니다.



### GiVE: Guiding Visual Encoder to Perceive Overlooked Information (https://arxiv.org/abs/2410.20109)
- **What's New**: 이 연구에서는 Guiding Visual Encoder to Perceive Overlooked Information (GiVE) 접근 방식을 제안하여 시각 데이터를 텍스트와 효과적으로 결합하고 overlooked information (간과된 정보)를 반영할 수 있는 능력을 향상시킵니다. GiVE는 Attention-Guided Adapter (AG-Adapter) 모듈과 Object-focused Visual Semantic Learning 모듈을 통해 시각적 표현을 개선합니다.

- **Technical Details**: GiVE는 세 가지 새로운 손실 함수인 Object-focused Image-Text Contrast (OITC) 손실, Object-focused Image-Image Contrast (OIIC) 손실, Object-focused Image Discrimination (OID) 손실을 통합하여 모델이 salient (두드러진) 및 non-salient (비두드러진) 객체를 모두 고려하도록 합니다. 이 접근 방식은 dynamic visual focus (동적 시각적 초점) 조정을 통해 시각 데이터의 표현 능력을 높입니다.

- **Performance Highlights**: 실험 결과, GiVE 접근 방식은 기존의 시각 인코더에 비해 객체 고려 및 검색 정확도 향상에서 현저한 성과를 보여주며, state-of-the-art (최첨단) 성능을 달성하였습니다. 또한 Multi-Object Instruction (MOInst) 데이터셋을 새롭게 소개하여 다양한 객체에 대한 의미적 지침을 제공합니다.



### Emergence of Globally Attracting Fixed Points in Deep Neural Networks With Nonlinear Activations (https://arxiv.org/abs/2410.20107)
- **What's New**: 이 논문은 신경망의 레이어 간 숨겨진 표현(hidden representation) 간의 유사성이 진화하는 방식을 전반적으로 분석하는 새로운 이론적 프레임워크를 제시합니다. 특히, 활성화 함수(activation function)에만 의존하며 특정 조건 하에 커널 시퀀스(kernel sequence)가 결정론적으로 변화하는 과정을 보입니다.

- **Technical Details**: 연구에서는 Hermite 다항식(Hermite polynomials)을 사용해 활성화 함수를 확장하고, 이를 통해 커널 맵(kernel map)의 명시적 형태를 도출하며 고정점(fixed points)을 완전히 특성화합니다. 비선형 활성화(nonlinear activation)일 경우, 커널 시퀀스는 전역적으로(unique fixed point) 수렴하여, 이는 활성화 및 네트워크 아키텍처에 따라 직교적 또는 유사한 표현을 나타낼 수 있습니다.

- **Performance Highlights**: 이 연구는 잔여 연결(residual connections)과 정규화 레이어(normalization layers)를 갖는 네트워크에서도 유사한 수렴 행동을 입증하며, 깊이(depth) 및 비선형성(nonlinearity)의 상호작용이 네트워크의 초기화 시점에서 어떻게 작동하는지에 대한 통찰을 제공합니다.



### Self-Normalized Resets for Plasticity in Continual Learning (https://arxiv.org/abs/2410.20098)
- **What's New**: 이번 연구에서는 Self-Normalized Resets (SNR)라는 새로운 적응형 알고리즘을 제안합니다. 이 알고리즘은 뉴런의 발화 비율이 사실상 제로로 떨어질 때 뉴런의 가중치를 재설정함으로써 플라스틱 손실(plasticity loss)을 완화합니다.

- **Technical Details**: SNR 알고리즘은 단일 하이퍼파라미터인 거부 백분위수(threshold)를 사용하여 그룹의 뉴런에서 비활성 상태의 뉴런을 식별하고 재설정합니다. 우리는 SNR이 다양한 지속 학습 문제에 대해 기존의 경쟁 알고리즘들로부터 월등한 성능을 보여준다는 것을 입증하였습니다.

- **Performance Highlights**: SNR은 지속 학습 문제에 있어 경쟁 알고리즘들보다 일관되게 우수한 성능을 달성하며, 하이퍼파라미터에 대한 민감도가 적고, 이상적인 조건에서 ReLU 학습 문제에서도 목표 ReLU를 학습할 수 있음을 보여줍니다.



### OGBench: Benchmarking Offline Goal-Conditioned RL (https://arxiv.org/abs/2410.20092)
- **What's New**: 이번 연구에서는 오프라인 목표 조건 강화 학습(Goal-Conditioned Reinforcement Learning, GCRL) 알고리즘을 위한 새로운 기준 벤치마크인 OGBench를 제안합니다. OGBench는 다양한 환경과 데이터 세트를 포함하여 알고리즘의 성능을 시스템적으로 평가할 수 있도록 설계되었습니다.

- **Technical Details**: OGBench는 8가지 환경 유형과 85개의 데이터 세트, 6개의 대표적인 오프라인 GCRL 알고리즘의 참조 구현을 포함합니다. 각 환경과 데이터 세트는 스티칭(stitching), 장기적 추론(long-horizon reasoning), 고차원 입력 및 확률적(stochasticity) 상황 처리 능력 등을 시험할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과에서는 기존 벤치마크에서 유사한 순위를 기록하는 알고리즘들 간의 뚜렷한 강점과 약점이 드러났습니다. OGBench는 새로운 알고리즘 개발을 위한 탄탄한 기반을 제공하며, 복잡한 작업들을 통해 오프라인 GCRL의 잠재력을 열 수 있게 합니다.



### RARe: Retrieval Augmented Retrieval with In-Context Examples (https://arxiv.org/abs/2410.20088)
- **What's New**: 본 논문에서는 기존의 decoder-only language models (LLMs)에서 부분적으로 사용되는 in-context examples가 retrieval 모델의 성능 향상에 기여할 수 있는지를 연구합니다. 기존 LLMs에서의 사용 방법과는 달리, retrieval 작업에 적용하기 위해 새로운 접근법인 RARe(Retrieval Augmented Retrieval with In-Context Examples)를 제안합니다.

- **Technical Details**: RARe는 사전 훈련된 모델을 fine-tuning하여, 타겟 쿼리와 의미적으로 유사한 in-context examples를 추가하는 방식입니다. 이를 통해 retrieval 시스템의 쿼리 형식을 조정하고, contrastive loss를 사용하여 실험하고 성능을 평가합니다.

- **Performance Highlights**: RARe는 다양한 오픈 도메인 데이터셋(예: BeIR, RAR-b)에서 기본 모델보다 최대 2.72%의 nDCG 성능 향상을 보여주었으며, 특히 out-of-domain 일반화 성능이 뛰어난 것으로 나타났습니다.



### Optimizing Keyphrase Ranking for Relevance and Diversity Using Submodular Function Optimization (SFO) (https://arxiv.org/abs/2410.20080)
- **What's New**: 본 연구에서는 키프레이즈( keyphrase ) 랭킹에서의 다각성을 고려하기 위해 Submodular Function Optimization (SFO) 방법론을 제안합니다.

- **Technical Details**: 이 접근법은 키프레이즈 랭킹 작업을 submodular maximization으로 프레임화하여 관련성과 다양성을 균형 있게 고려합니다. 이를 통해 다양한 대표적 키프레이즈를 선택하게 됩니다.

- **Performance Highlights**: 벤치마크 데이터셋에서의 실험 결과, 제안하는 방법은 기존 방법들보다 관련성과 다양성 메트릭에서 뛰어난 성능을 보이며, 실행 시간에서도 SOTA (State of the Art) 성능을 달성하였습니다.



### A Multi-Modal Non-Invasive Deep Learning Framework for Progressive Prediction of Seizures (https://arxiv.org/abs/2410.20066)
Comments:
          4 pages, 5 figures, IEEE BSN 2024

- **What's New**: 이 논문에서는 비침습적 다중 모드 센서 네트워크를 기반으로 한 심층 학습(Deep Learning) 방법론을 사용하여 발작 예측을 위한 혁신적인 프레임워크를 소개합니다. 이 시스템은 환자에게 발작 위험을 경고하여 예방 조치를 취할 수 있도록 도와줍니다.

- **Technical Details**: 이 프레임워크는 다양한 비침습적 전기생리학적 신호(예: EEG와 ECG)를 사용하며, 1시간 전부터 15분 간격으로 발작을 예측하는 다중 클래스 예측 알고리즘을 구현합니다. 시스템은 실시간 처리를 위한 최적화된 알고리즘과 경량 설계를 특징으로 하여, 배터리 에너지를 보존하고 클라우드 기반 솔루션의 데이터 전송 오버헤드를 최소화합니다.

- **Performance Highlights**: 이 다중 모드 모델은 29명의 환자를 대상으로 평균 95%의 민감도, 98%의 특이도, 97%의 정확도를 달성하였습니다.



### Transforming Precision: A Comparative Analysis of Vision Transformers, CNNs, and Traditional ML for Knee Osteoarthritis Severity Diagnosis (https://arxiv.org/abs/2410.20062)
- **What's New**: 이 연구는 기존의 기계 학습 기법과 새로운 딥러닝 모델을 비교 분석하여 슬개골 퇴행성 관절염(KO)의 중증도를 X-ray 이미지를 통해 진단하는 데 초점을 맞추고 있습니다. 최신 ViT( Vision Transformer) 모델의 의료 이미지 컨텍스트에서의 적용 가능성과 비교 효과를 강조합니다.

- **Technical Details**: 연구는 Osteoarthritis Initiative (OAI)에서 제공한 1526장의 X-ray 이미지를 사용하여 KO의 중증도를 5개 등급으로 분류합니다. 기존의 기계 학습 모델(GaussianNB, KNN)보다 Convolutional Neural Networks(CNN)인 Inception-V3와 VGG-19가 평균 55-65%의 정확도를 보였지만, ViT 모델(Da-VIT, GCViT, MaxViT)은 66.14%의 정확도, 0.703의 정밀도, 0.614의 재현율, 0.835 이상의 AUC를 달성했습니다.

- **Performance Highlights**: ViT 모델이 다른 모델들보다 뛰어난 성능을 발휘했으며, 특히 정확도(F1 score, precision, AUC score) 측면에서 월등한 결과를 나타냈습니다. 이는 의료 진단 작업 흐름에서 ViT 모델 통합의 필요성을 강조하며, KO 평가의 정확성과 신뢰성을 혁신적으로 변화시킬 가능성을 제시합니다.



### Evaluating Neural Networks for Early Maritime Threat Detection (https://arxiv.org/abs/2410.20054)
- **What's New**: 이번 연구는 보트 활동의 궤적을 분류하는 작업에서 엔트로피 기반 클러스터링 대신 신경망(neural network) 기반 접근법의 정확성을 포괄적으로 평가했으며, 합성 데이터(synthetic data)를 사용하여 다양한 신경망 아키텍처의 성능을 비교했습니다.

- **Technical Details**: 네 가지 신경망 모델을 학습시켰고, 궤적의 길이에 따라 정확도가 어떻게 변하는지를 조사했습니다. 특히, 데이터 정규화(normalization) 및 회전 기반 데이터 증강(rotation-based data augmentation) 기법을 통해 테스트 시간 동안 모델의 강건성을 개선했습니다. 네트워크 기반 모델은 전체 궤적에 대해 최대 100%의 테스트 세트 정확도를 달성했습니다.

- **Performance Highlights**: 신경망 모델은 전체 궤적에서 100%의 정확도를 기록했으며, 궤적의 타임 스텝이 줄어들면서도 안정적으로 성능이 감소했습니다. 반면, 기존의 엔트로피 기반 클러스터링 알고리즘은 97% 이상 달성하지 못했습니다. 이 연구는 합성 데이터에서 신경망 접근이 해양 위협 탐지 분야에서 더 뛰어난 성능을 발휘할 수 있음을 보여줍니다.



### AutoMIR: Effective Zero-Shot Medical Information Retrieval without Relevance Labels (https://arxiv.org/abs/2410.20050)
Comments:
          15 pages, 3 figures

- **What's New**: 본 논문은 Self-Learning Hypothetical Document Embeddings (SL-HyDE)라는 새로운 접근 방식을 소개하여 의료 정보 검색(MIR)에서의 효과적인 제로샷(dense retrieval) 문제를 해결하고자 합니다.

- **Technical Details**: SL-HyDE는 대규모 언어 모델(LLMs)을 활용하여 주어진 쿼리에 기반한 가상의 문서를 생성합니다. 이러한 가상 문서는 의료 컨텍스트를 포함하여 관련 문서 검색을 지원합니다. 또한, SL-HyDE는 라벨이 없는 의료 데이터셋을 활용하여 자가 학습 프레임워크를 통해 문서 생성 및 검색 과정을 지속적으로 개선합니다. CMIRB(중국 의료 정보 검색 벤치마크)는 실제 의료 시나리오를 기반으로 한 평가 프레임워크로, 5개의 작업과 10개의 데이터셋을 포함합니다.

- **Performance Highlights**: SL-HyDE는 기존의 방법들보다 검색 정확도가 현저히 향상되었으며, 다양한 LLM 및 검색기 구성에서 강력한 일반화 및 확장성을 보여주었습니다. CMIRB에서 SL-HyDE는 HyDE를 평균 4.9% 초과하는 성능을 발휘하였고, BGE 단독 검색기보다 7.2% 향상된 결과를 나타내었습니다.



### DQRM: Deep Quantized Recommendation Models (https://arxiv.org/abs/2410.20046)
- **What's New**: 이번 연구에서는 Deep Learning Recommendation Model (DLRM)을 기반으로 하여, 추천 모델을 부담 없이 훈련하고 실행할 수 있게 해주는 소형, 강력하고 효율적인 새로운 추천 프레임워크인 Deep Quantized Recommendation Model (DQRM)을 제안합니다.

- **Technical Details**: DQRM에서는 ultra-low INT4 정밀도로 DLRM 모델을 양자화하고, quantization-aware training (QAT)을 적용하여 메모리 및 계산 비용을 크게 낮추면서 과적합(overfitting)을 완화합니다. 또한, embedding 테이블의 그라디언트를 INT8 형태로 양자화하여 통신 부하를 감소시키고, 효율적인 훈련을 가능하게 합니다.

- **Performance Highlights**: DQRM 모델은 Kaggle 데이터셋에서 79.07% 정확도(모델 크기: 0.27 GB)와 Terabyte 데이터셋에서 81.21% 정확도(모델 크기: 1.57 GB)를 달성하여 FP32 DLRM보다도 더 높은 성능을 기록했습니다.



### Roles of LLMs in the Overall Mental Architectur (https://arxiv.org/abs/2410.20037)
- **What's New**: 이 논문은 현재의 LLM(대형 언어 모델)의 이해를 위해 인간 정신 구조(cognitive architecture)를 살펴보는 방법에 대해 논의합니다.

- **Technical Details**: 심리학, 철학, 인지 과학 문헌을 기반으로, 기존의 LLM은 인간의 암묵적 사고 과정(직감, 본능 등)과 잘 일치한다고 주장합니다. 그러나 인간 정신 구조 내에는보다 나은 기호적(symbolic) 능력을 가진 명시적 과정(explicit processes)도 존재한다고 설명합니다.

- **Performance Highlights**: 이 논문은 경합적인 컴퓨터 인지 아키텍처(dual-process computational cognitive architectures)가 LLM의 개선뿐만 아니라 LLM이 이러한 모델을 통해 향상될 수 있는 방법을 제시하여 시너지 효과를 낼 수 있다고 주장하고 있습니다.



### Architectural Flaw Detection in Civil Engineering Using GPT-4 (https://arxiv.org/abs/2410.20036)
- **What's New**: 이 논문은 인공지능(AI)의 응용이 토목 공학 디자인 품질 및 안전성을 향상시키는 혁신적 접근 방식을 제시한다고 설명합니다. 특히, 설계 단계에서 건축 결함을 감지하기 위해 고급 LLM GPT4 Turbo 비전 모델을 사용하고, 누락된 문과 창문을 식별하는 데 중점을 둡니다.

- **Technical Details**: 연구에서는 모델의 성능을 precision, recall, F1 score와 같은 여러 메트릭을 통해 평가하며, AI가 인간 검증 데이터와 비교하여 결함을 정확하게 감지하는 효과성을 보이고 있습니다. 이외에도 AI의 추가적인 능력으로 하중 지지 이슈, 재료 약점 식별, 건축 법규 준수 여부 등을 탐구합니다.

- **Performance Highlights**: AI는 설계 정확성을 현저하게 개선하고 비싼 수정 비용을 줄이며 지속 가능한 관행을 지원함으로써 궁극적으로 더 안전하고 효율적이며 미적으로 최적화된 구조물을 보장하여 토목 공학 분야를 혁신할 수 있는 잠재력을 보입니다.



### Training the Untrainable: Introducing Inductive Bias via Representational Alignmen (https://arxiv.org/abs/2410.20035)
Comments:
          Under Review; 24 pages, 9 figures; Project page and code is at this https URL

- **What's New**: 이번 논문에서는 기존에는 훈련이 어려운 것으로 여겨졌던 아키텍처들이 다른 아키텍처로부터의 inductive bias를 활용하여 훈련될 수 있다는 점을 보여줍니다. ‘Guidance’라는 새로운 기법을 도입하여, 가이드 네트워크가 타겟 네트워크를 유도하며, 이를 통해 타겟 네트워크의 성능을 크게 향상시킬 수 있음이 입증되었습니다.

- **Technical Details**: 이 방법론은 두 네트워크 간의 representational similarity를 맞추는 것을 목표로 하며, neural distance function을 사용하여 층 별로 표현을 일치시키는 방식입니다. 연구자들은 다양한 아키텍처에 대해 이 이론을 적용했으며, 가이드 네트워크가 훈련되지 않은 경우에도 여전히 부분적으로 architectural prior를 전파할 수 있음을 발견했습니다.

- **Performance Highlights**: 이 방법을 활용하여, 일반적인 fully connected 네트워크가 비전 작업에서 즉각적인 overfitting을 극복하게 되었고, plain CNN이 ResNet에 버금가는 성능을 발휘하게 되었으며, RNN과 Transformers 간의 성능 격차가 줄어드는 등의 개선이 이루어졌습니다.



### Sensor2Text: Enabling Natural Language Interactions for Daily Activity Tracking Using Wearable Sensors (https://arxiv.org/abs/2410.20034)
- **What's New**: 본 연구는 Sensor2Text라는 새로운 모델을 제안하여, 웨어러블 센서를 통해 수집한 데이터로 인간의 일상 활동을 추적하고 질문-응답 대화를 수행할 수 있는 혁신적인 접근 방식을 제공합니다. 이는 기존 비디오 기반 솔루션의 사생활 및 시야 제한 문제를 해결합니다.

- **Technical Details**: Sensor2Text는 전이 학습(transfer learning)과 교사-학생 네트워크(teacher-student networks)를 활용하여 시각-언어 모델의 지식을 활용하며, 다중 모달 인코더-디코더 신경망 모델을 설계하여 언어 및 센서 데이터를 공동으로 처리합니다. 대화 및 Q&A 기능을 갖추기 위해 대규모 언어 모델(Large Language Models)을 통합하였습니다.

- **Performance Highlights**: Sensor2Text는 다양한 웨어러블 센서 모달리티를 사용하여 인간 활동을 정확히 식별하고 Q&A 대화를 진행하는 데 뛰어난 성능을 보였습니다. 캡셔닝 및 대화 작업에서 기존의 비주얼-언어 모델과 유사하거나 더 나은 성능을 보였으며, 저조도 환경에서도 강력한 일반화 능력을 발휘하였습니다.



### SCube: Instant Large-Scale Scene Reconstruction using VoxSplats (https://arxiv.org/abs/2410.20030)
Comments:
          NeurIPS 2024. Project page: this https URL

- **What's New**: SCube는 일부 제공된 이미지로부터 대규모 3D 장면(geometry, appearance, and semantics)을 재구성하기 위한 혁신적인 방법으로, VoxSplat이라는 새로운 표현 방식을 사용합니다.

- **Technical Details**: SCube는 고해상도 희소-복셀(sparse-voxel) 스캐폴드를 기반으로 하는 3D Gaussian의 집합인 VoxSplat을 이용해 장면을 인코딩합니다. 입출력 이미지에 조건화된 계층 구조의 복셀(latent diffusion model)을 활용해 재구성을 수행하며, 그 후에는 피드포워드(Feedforward) 방식의 appearance prediction 모델을 통해 Gaussian 세트를 예측합니다.

- **Performance Highlights**: SCube는 비겹치는 3장의 이미지로도 1024^3 복셀(grid)에서 수백 미터를 포괄하는 백만 개의 Gaussian을 20초 만에 생성할 수 있습니다. Waymo 자율주행 데이터셋을 통해 기존 방법들에 비해 우수한 3D 재구성 성능을 보여주며, LiDAR 시뮬레이션 및 텍스트-투-장면 생성과 같은 다양한 응용 프로그램에 적용할 수 있습니다.



### FLOW: A Feedback LOop FrameWork for Simultaneously Enhancing Recommendation and User Agents (https://arxiv.org/abs/2410.20027)
- **What's New**: 이번 연구에서는 추천 시스템에서 추천 에이전트와 사용자 에이전트 간의 상호작용을 고려한 새로운 프레임워크인 FLOW를 제안합니다. 이전의 연구들은 주로 각 에이전트의 능력 향상에 중점을 두었으나, 두 에이전트 간의 협업은 고려하지 않았습니다.

- **Technical Details**: FLOW 프레임워크는 피드백 루프(feedback loop)를 통해 추천 에이전트와 사용자 에이전트 간의 협업을 이끌어냅니다. 추천 에이전트는 사용자 에이전트의 피드백을 분석하여 사용자의 선호도를 세분화하며, 사용자 에이전트는 추천된 항목들을 통해 사용자의 숨겨진 관심사를 발견하게 됩니다.

- **Performance Highlights**: 세 가지 널리 사용되는 추천 시스템 데이터셋을 통해 피드백 루프의 효과를 평가한 결과, 추천 에이전트와 사용자 에이전트의 성능이 동시에 향상됨을 보여주었습니다.



### Beyond Fine-Tuning: Effective Strategies for Mitigating Hallucinations in Large Language Models for Data Analytics (https://arxiv.org/abs/2410.20024)
- **What's New**: 본 연구에서는 LLMs(대형 언어 모델)에서 발생하는 'hallucinations'(환각)을 완화하기 위한 새로운 전략들을 제안하고 평가합니다.

- **Technical Details**: 네 가지 구체적인 전략: Structured Output Generation(구조화된 출력 생성), Strict Rules Enforcement(엄격한 규칙 집행), System Prompt Enhancements(시스템 프롬프트 향상), Semantic Layer Integration(의미적 계층 통합) 등이 포함됩니다.

- **Performance Highlights**: 이 연구의 결과는 전통적인 fine-tuning(미세 조정) 접근 방식보다 이러한 방법들이 hallucinations를 줄이는 데 더 효과적임을 보여줍니다.



### Think Carefully and Check Again! Meta-Generation Unlocking LLMs for Low-Resource Cross-Lingual Summarization (https://arxiv.org/abs/2410.20021)
- **What's New**: 이 논문은 저자들이 제안한 4단계 제로 샷(zero-shot) 접근 방식을 통해 자원이 제한된(low-resource) 언어에 대한 Cross-lingual summarization(클래스) 성능을 향상시키는 방법을 다룹니다. 저자들은 Summarization, Improvement, Translation, Refinement(SITR) 방법론을 사용하여 대규모 언어 모델(LLMs)의 잠재력을 극대화하였습니다.

- **Technical Details**: 연구에서는 4단계의 SITR 방법이 사용됩니다: (1) Summarization: 원본 텍스트에서 요약 생성, (2) Improvement: 요약 개선, (3) Translation: 저자들이 제공한 번역 프롬프트를 사용해 저자 언어로 번역, (4) Refinement: 앞서 생성된 요약과 번역을 결합해 최종 결과물 생성.

- **Performance Highlights**: 실험 결과, GPT-3.5와 GPT-4가 다른 LLM 기준에서 상당히 우수한 성능을 보였으며, 자원이 제한된 언어에 대해 많은 모델을 초과하는 성과를 달성했습니다. 이 연구는 LLM이 저자 언어에도 잘 대응할 수 있음을 보여주며, 특히 저자 언어에 대한 CLS 작업에서의 잠재력을 재조명합니다.



### GHIL-Glue: Hierarchical Control with Filtered Subgoal Images (https://arxiv.org/abs/2410.20018)
Comments:
          Code, model checkpoints and videos can be found at this https URL

- **What's New**: GHIL-Glue는 인터넷 규모의 데이터에 기반하여 생성된 이미지 및 비디오 예측 모델과 하위 목표 조건 정책을 효과적으로 결합하는 새로운 인터페이스를 제공합니다.

- **Technical Details**: 이 방법은 하위 목표를 필터링하고, 생성된 하위 목표에 포함된 시각적 아티팩트에 대해 목표 조건 정책의 내구성을 향상시키는 두 가지 주요 구성 요소를 포함합니다.

- **Performance Highlights**: GHIL-Glue는 여러 계층적 모델에서 25%의 성능 개선을 달성했으며, 단일 RGB 카메라에서 관찰 정보를 사용하는 CALVIN 시뮬레이션 벤치마크에서 새로운 최첨단 성과를 이뤘습니다.



### Off-Policy Selection for Initiating Human-Centric Experimental Design (https://arxiv.org/abs/2410.20017)
- **What's New**: 이 논문에서는 First-glance Off-Policy Selection (FPS)라는 새로운 접근법을 소개하여, 개인의 이질성을 체계적으로 반영하고 전문가 없이도 새로운 참여자에 대한 정책을 선택할 수 있는 방법을 제시합니다.

- **Technical Details**: FPS는 오프라인 데이터셋에서 유사한 행동을 가진 개인들을 하위 그룹으로 나누고, 각 그룹에 대해 정책 선택 기준을 설정합니다. Variational Autoencoder (VAE) 모델을 사용하여 각 하위 그룹의 합성 경로를 생성하고, 바운드된 분산을 가진 무편향 가치 함수 추정기를 개발하여 정책 선택 기준을 결정합니다. 이를 통해 새로운 참여자가 합류할 때, 그들이 속한 하위 그룹에 맞춘 정책을 추천합니다.

- **Performance Highlights**: FPS는 지능형 교육 시스템에서 1,288명의 학생들을 대상으로 5년간 평가되었으며, 교수들이 수작업으로 선정한 정책보다 208% 향상된 학습 성과를 보여 주었습니다. 또한 기존의 OPS 방법에 비해 결과를 136% 증가시켰습니다. 의료 분야의 응용 사례인 패혈증 치료에서도 FPS가 기존 OPS 방법을 초월하며 최적의 치료 정책을 정확히 식별했습니다.



### Enhancing Battery Storage Energy Arbitrage with Deep Reinforcement Learning and Time-Series Forecasting (https://arxiv.org/abs/2410.20005)
Comments:
          Accepted for publication at the 18th ASME International Conference on Energy Sustainability

- **What's New**: 본 연구는 deep reinforcement learning (DRL)과 시계열 예측 (time-series forecasting) 방법을 결합하여 에너지 차익 거래 (energy arbitrage) 성능을 개선하는 새로운 접근 방식을 소개합니다.

- **Technical Details**: 에너지 차익 거래(EA)의 통제 방식으로 DRL을 사용하는데, 심층 신경망(deep neural networks)을 활용한 예측기를 도입하여 Alberta, Canada의 비정상적인 전기 가격 데이터에서 성능 개선 가능성을 연구했습니다. DQN(deep Q-networks) 및 PPO(proximal policy optimization) 알고리즘을 비교하여, 예측기의 수와 예측 기간이 성능에 미치는 영향을 분석했습니다.

- **Performance Highlights**: 복수의 예측기를 결합함으로써 24시간 미래 예측을 기반으로 DQN을 사용할 경우 보상이 60% 증가했습니다. 불완전한 예측에서도 유용한 정보를 제공하여, DRL 에이전트가 이익을 극대화하는 제어 정책을 학습할 수 있음을 나타냅니다.



### Artificial Intelligence of Things: A Survey (https://arxiv.org/abs/2410.19998)
Comments:
          Accepted in ACM Transactions on Sensor Networks (TOSN)

- **What's New**: IoT와 현대 인공지능(AI)의 통합으로 새로운 패러다임인 인공지능사물인터넷(AIoT)이 등장했습니다. 이 서베이는 AIoT 연구에 대한 체계적이고 포괄적인 리뷰를 제공합니다.

- **Technical Details**: 이 논문은 감지(sensing), 계산(computing), 네트워킹 및 통신(networking & communication) 등의 AIoT의 세 가지 주요 구성 요소를 조사합니다. 또한 다양한 주요 응용 분야를 위한 도메인 특화 AIoT 시스템도 리뷰합니다.

- **Performance Highlights**: AIoT는 현대 AI의 혁신 덕분에 많은 일상적인 장치를 가능하게 하여, 헬스케어, 비디오 스트리밍, 자율 주행 및 증강/가상 현실과 같은 다양한 산업을 혁신할 잠재력을 보여줍니다.



### SAD: State-Action Distillation for In-Context Reinforcement Learning under Random Policies (https://arxiv.org/abs/2410.19982)
- **What's New**: 이 논문은 사전 훈련된 기초 모델(Foundation Models, FMs)을 이용한 In-Context Reinforcement Learning (ICRL)의 새로운 접근 방식인 State-Action Distillation (SAD)을 제안합니다. 기존의 알고리즘들이 최적 정책이 필요하던 점에 비해, SAD는 무작위 정책으로 생성된 데이터를 사용하여 교육 데이터를 효과적으로 생성하는 최초의 방법입니다.

- **Technical Details**: SAD는 신뢰 구간(trust horizon) 내에서 무작위 정책을 사용하여 전체 상태 및 행동 공간에서 뛰어난 상태-행동 쌍을 증류(distill)합니다. 이 접근 방식은 추가 정보 없이 pretraining dataset을 생성 가능하며, 기존 ICRL 알고리즘이 요구하는 고찰(thorough examination)이나 최적 행동 정책 없이도 작동합니다.

- **Performance Highlights**: 여러 ICRL 벤치마크 환경에서의 실험 결과, SAD는 오프라인 평가에서 평균 180.86%, 온라인 평가에서 평균 172.8%의 성능 향상을 보여줍니다. 이는 기존의 최상의 기준선 알고리즘보다 현저한 성과입니다.



### OReole-FM: successes and challenges toward billion-parameter foundation models for high-resolution satellite imagery (https://arxiv.org/abs/2410.19965)
- **What's New**: 본 논문은 세계 최초의 엑사스케일 슈퍼컴퓨터인 Frontier를 활용하여 원격 감지(상관 이미지) 분야의 억 단위 매개변수(억 단위 모델)를 사전 훈련하고 평가한 연구 결과를 제시하고 있습니다. 이 연구는 FMs(Foundation Models)의 훈련 효율성을 높이고, 데이터 세트 구성 및 벤치마킹을 위한 최고 관행을 제공합니다.

- **Technical Details**: 본 연구에서는 고해상도(Swin-H) 및 중해상도(ViT-L) 광학 이미지와 SAR(ViT-L) 데이터를 포함한 여러 사전 훈련 변형의 성능을 평가했습니다. 또한 TIU라는 새로운 고해상도 원격 감지 사전 훈련 데이터 세트를 구성하였고, 4-band 모델의 초기화를 위한 전이 학습 기법을 사용하여 사전 훈련 속도와 성능을 향상시켰습니다.

- **Performance Highlights**: 논문은 모델 초기화, 데이터 및 사전 훈련된 모델을 공공에 공개할 계획이라는 점을 강조합니다. 또한 원격 감지 벤치마크에서의 이미지 분류, 의미 분할 및 객체 탐지 성능을 평가한 결과, 효과적인 모델 스케일을 위한 데이터 스케일링의 중요성을 강조합니다.



### Understanding Adam Requires Better Rotation Dependent Assumptions (https://arxiv.org/abs/2410.19964)
- **What's New**: Adam 옵티마이저가 파라미터 공간의 회전에 민감하다는 점을 밝혀, 기존 이론적 프레임워크의 한계를 드러냄.

- **Technical Details**: Adam은 회전 불변성이 없는 특성을 가지며, 무작위 회전 시 성능이 저하됨을 실험적으로 입증함. 또한, 특정 회전 방식이 Adam의 성능을 보존하거나 향상시킬 수 있음을 확인함.

- **Performance Highlights**: Adam의 성능 저하는 고정된 기준점에서 시작할 때 발생하며, 회전 의존적인 특성을 추가로 연구해 새로운 이론적 프레임워크 개발의 필요성을 강조함.



### DualMAR: Medical-Augmented Representation from Dual-Expertise Perspectives (https://arxiv.org/abs/2410.19955)
- **What's New**: 본 논문에서는 EHR(전자 건강 기록)의 예측 작업을 향상시키기 위해 DualMAR이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 개별 관찰 데이터와 공공 지식 기반을 모두 활용하여 EHR의 예측 정확도를 높입니다.

- **Technical Details**: DualMAR는 두 가지 주요 구성 요소인 ‘Knowledge Scholar’와 ‘Local Expert’로 이루어져 있습니다. ‘Knowledge Scholar’는 공공 데이터베이스를 기반으로한 진단 주도 KG(지식 그래프)를 구축하고, ‘Local Expert’는 특정 질병-검사 EHR 그래프를 이용하여 의료 코드의 임베딩을 향상시킵니다. 이들 구성 요소는 Encoder-Decoder 아키텍처를 통해 통합되어 예측 모델의 성능을 높입니다.

- **Performance Highlights**: DualMAR는 실험 결과에서 최신 모델보다 우수한 성능을 보여 EHR 예측 및 KG(지식 그래프) 통합의 효과를 입증했습니다. 또한, 복잡하고 다양화된 관계를 가지고 있는 KG가 EHR 모델의 예측 성능을 어떻게 개선할 수 있는지를 보여줍니다.



### Assessing the societal influence of academic research with ChatGPT: Impact case study evaluations (https://arxiv.org/abs/2410.19948)
- **What's New**: 이 연구는 ChatGPT가 사회적 영향 주장(System Impact Claim, ICS)을 평가할 수 있는지를 조사하고, 전문가 평가자를 지원할 수 있는 가능성을 탐구하였습니다.

- **Technical Details**: 6,220개의 공공 ICS를 REF2021 평가 지침과 함께 ChatGPT 4o-mini에 입력하여 평가를 수행했습니다. 연구 결과, 제목과 요약만 입력하는 것이 전문가 점수와 높은 상관관계를 보였으며, 원래의 REF 지침을 수정하여 더 엄격한 평가를 유도하는 것이 효과적임을 나타냈습니다.

- **Performance Highlights**: ChatGPT를 이용한 ICS 평가 점수는 모든 34개의 평가 단위(UoAs)에서 부서 평균 점수와 긍정적인 상관관계를 보였으며, 점수의 범위는 0.18(경제학 및 계량 경제학)에서 0.56(심리학, 정신의학 및 신경과학)까지 다양했습니다. 부서 수준에서도 체육 및 운동 과학, 여가 및 관광 분야에서 0.71로 더 높은 상관관계를 나타냈습니다.



### Cobblestone: Iterative Automation for Formal Verification (https://arxiv.org/abs/2410.19940)
Comments:
          13 pages, 10 figures

- **What's New**: Cobblestone은 내부의 부분 진행을 활용하여 여러 가지 실패한 증명을 생성하고 이를 성공적인 하나의 증명으로 결합할 수 있는 새로운 증명 생성 접근 방식을 소개합니다. 특히 Cobblestone은 LLM(대형 언어 모델)을 사용하여 증명 작업을 수행하며, 이전의 도구들과 비교하여 크게 향상된 성능을 보여주고 있습니다.

- **Technical Details**: Cobblestone은 GPT-4와 Coq 증명 보조 도구를 활용하여 구현되었으며, 내부의 부분적인 증명 진행을 통해 자동으로 증명을 생성합니다. Cobblestone은 실패한 증명에서 유용한 부분을 식별하고 이를 조합하여 전체 증명을 이끌어냅니다. 또한, 외부 정보를 통해 인간 증명 엔지니어 또는 다른 도구와 상호작용하는 방식으로 성능을 더욱 증대시킬 수 있습니다.

- **Performance Highlights**: Cobblestone은 CoqGym에서 48%의 정리(정리: theorem)를 자동으로 증명할 수 있으며, 이전의 Proverbot9001 도구는 17%에 불과했습니다. 또한 외부 정보를 포함할 경우, Cobblestone은 최대 58%의 정리를 증명할 수 있는 잠재력을 보였습니다. 이는 자동 증명 생성 도구에서 새로운 기준을 설정하는 성과입니다.



### RobustKV: Defending Large Language Models against Jailbreak Attacks via KV Eviction (https://arxiv.org/abs/2410.19937)
- **What's New**: 본 논문에서는 기존의 방어 방법들이 치명적인 공격인 jailbreak 공격을 충분히 방어하지 못하는 상황에서 새로운 방어 기법인 RobustKV를 제안합니다. 기존의 방어책은 주로 jailbreak 프롬프트의 영향을 감소시키는데 집중하고 있지만, RobustKV는 키-값 캐시(key-value cache)의 해로운 쿼리의 중요한 토큰을 선택적으로 제거함으로써 다른 접근 방식을 취합니다.

- **Technical Details**: RobustKV는 jailbreak 프롬프트의 효과를 위해 필요로 하는 토큰의 중요성을 측정하기 위해 attention scores를 활용합니다. 이 방식으로 중요도가 낮은 토큰의 KVs를 전략적으로 제거하여KV 캐시에서 해로운 쿼리의 존재를 감소시킵니다. 따라서 이 기법은 해로운 응답 생성을 예방하도록 설계되었습니다.

- **Performance Highlights**: RobustKV는 다양한 벤치마크 데이터셋과 모델을 사용한 광범위한 평가를 통해 최신 jailbreak 공격을 효과적으로 저지하면서도 무해한 쿼리에 대한 LLM의 일반 성능을 유지하는 데 성공했습니다. 또한, RobustKV는 적대자들에게 흥미로운 회피 딜레마를 만들어내어, RobustKV를 회피하는 것과 LLM의 내장된 안전 장치를 우회하는 것 사이에서 균형을 잡도록 강요합니다. 이러한 균형 추구는 RobustKV의 적응형 공격에 대한 강인함을 강조합니다.



### Enhancing Safety in Reinforcement Learning with Human Feedback via Rectified Policy Optimization (https://arxiv.org/abs/2410.19933)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)의 유용성과 안전성을 균형 있게 조정하는 새로운 방법인 Rectified Policy Optimization (RePO)를 제안합니다. 기존 방법들과 비교하여 RePO는 각 프롬프트에 대해 더 엄격한 안전성 제약 조건을 적용하여 안전성 간섭을 감소시킵니다.

- **Technical Details**: RePO의 핵심은 rectified policy gradients를 기반으로 한 정책 업데이트 메커니즘입니다. 이 방법은 모든 프롬프트의 엄격한 안전 위반에 대해 패널티를 부여하며, 이를 통해 대부분의 프롬프트에서 안전성을 향상합니다. 연구에서는 Alpaca-7B 모델을 사용하여 RePO의 효과를 empirically 증명했습니다.

- **Performance Highlights**: 실험 결과, RePO는 전통적인 평균 안전 기준과 비교하여 언어 모델의 안전 정렬을 개선하고 안전 간섭을 효과적으로 줄였습니다. 이는 안전성과 유용성을 동시에 활용할 수 있는 새로운 접근임을 시사합니다.



### Disentangling Genotype and Environment Specific Latent Features for Improved Trait Prediction using a Compositional Autoencoder (https://arxiv.org/abs/2410.19922)
- **What's New**: 본 연구에서는 식물 육종 및 유전학 프로그램에서 형질 예측을 개선하기 위한 복합 오토인코더(compositional autoencoder, CAE) 프레임워크를 소개합니다. 기존의 방법들은 유전자형(genotype)과 환경(environment) 요인을 분리하지 않고 예측 모델을 구축해왔습니다. 본 연구는 이러한 차별화를 통해 예측 정확도를 향상시킬 수 있음을 제안합니다.

- **Technical Details**: CAE는 다차원 데이터를 유전자형 별(latent features) 및 환경 별(latent features)로 분해하여 이를 명확히 구분합니다. 이 프레임워크는 오토인코더 내에서 계층적 구조를 채택하여 얽힌 잠재적 특성(latent features)들을 효과적으로 분리합니다. CAE는 재료 데이터셋에 적용하여 환경 요소 모델링의 우수 성능과 다양한 형질(예: 꽃가루 생산일(Days to Pollen) 및 수확량(Yield))의 예측에서 5-10배 향상된 성능을 보였습니다.

- **Performance Highlights**: CAE는 전통적인 방법들보다 예측 성능이 획기적으로 개선되었습니다. 특히, 표준 오토인코더, 회귀를 통한 PCA 및 부분 최소 제곱 회귀(Partial Least Squares Regression, PLSR)와 비교해 보면, 환경 영향을 조명하고 형질 예측력을 높이는 데 있어 강력한 도구로 기능합니다.



### Provably Adaptive Average Reward Reinforcement Learning for Metric Spaces (https://arxiv.org/abs/2410.19919)
- **What's New**: 이번 연구에서는 Lipschitz MDPs에 대한 무한-horizon 평균 보상 강화 학습(Reinforcement Learning, RL)을 다루고, 상태-행동(space) 공간을 조정하여 유망한 영역으로 집중하는 새로운 알고리즘인 ZoRL을 개발하였습니다.

- **Technical Details**: 이 알고리즘은 상태 공간 차원(d𝒮)과 행동 공간 차원(d𝒜)을 합한 값 d와 문제에 따라 달라지는 확대 차원(d𝑧)을 통해 위험(regret)을 구속합니다. 위험은 O(T^{1 - d_{eff.}^{-1}})로 감소하며, d_{eff.}는 효과적인 차원으로 MDP에 의해 정의됩니다. ZoRL은 정책 공간에서 직접 작업하여 정책의 확대 효과를 보여주는 한편, 고정 격자 사용으로 인해 발생하는 비효율성을 해결하였습니다.

- **Performance Highlights**: 실험을 통해 ZoRL이 기존의 최첨단 알고리즘보다 뛰어난 성능을 발휘하며, 적응성을 통한 성과 향상을 입증했습니다.



### Urban Mobility: AI, ODE-Based Modeling, and Scenario Planning (https://arxiv.org/abs/2410.19915)
- **What's New**: 이 논문은 도시화와 기술 발전이 도시의 이동성에 미치는 영향을 AI 기술을 통해 탐구하는 새로운 접근 방식을 제시합니다. Ordinary Differential Equations (ODEs)을 활용한 수학적 모델링과 시나리오 계획이 결합되어 있습니다.

- **Technical Details**: 이 연구에서는 Python을 사용하여 ODE 기반 모델을 시뮬레이션하고, AI 혁신(예: 자율 주행 차량, 지능형 교통 관리)이 다양한 규제 조건에서 교통 혼잡을 줄이는 데 미치는 영향을 정량적으로 분석합니다. ODE 모델은 AI 채택율과 교통 혼잡 간의 동적 관계를 포착합니다.

- **Performance Highlights**: AI 채택이 교통 혼잡에 미치는 영향을 정량적으로 분석함으로써, 기업 및 정책 결정자들에게 전략적 지침을 제공하며, 더 효율적이고 지속 가능하며 살기 좋은 도시 만들기에 기여할 수 있는 방법을 제시합니다.



### Simmering: Sufficient is better than optimal for training neural networks (https://arxiv.org/abs/2410.19912)
- **What's New**: 본 연구에서는 'simmering'이라는 물리 기반의 훈련 방법을 소개하고, 이를 통해 신경망을 최적화 방식이 아닌 '충분히 좋은(good enough)' 가중치와 편향을 생성하도록 훈련시킬 수 있음을 제시합니다. 이 방법은 전통적인 최적화 방법론에 비해 우수한 성과를 보여줍니다.

- **Technical Details**: 'Simmering'은 Nosé-Hoover 체인 서모스탯(Nosé-Hoover chain thermostat)을 활용하여 신경망의 가중치와 편향을 '입자(particle)'로 취급하고, 역전파(backpropagation)에 의해 생성된 힘을 활용하여 최적 상태에 도달하지 않도록 합니다. 이 과정은 통계역학에서의 분배 함수를 사용하여 충분히 훈련된 네트워크를 알고리즘적으로 생성합니다. 또한, Python과 TensorFlow를 이용하여 수치적 통합을 통해 알고리즘을 구현하였습니다.

- **Performance Highlights**: 시뮬레이션 연구 결과, 'simmering'을 통해 Adam에 의해 생성된 과적합(overfitting)된 네트워크의 성능을 개선했고, 충분히 훈련된 네트워크가 최적 훈련된 네트워크보다 더 우수한 성과를 낼 수 있음을 보여주었습니다. 이는 최적화가 신경망 훈련의 이상적인 출발점이 아닐 수 있다는 것을 시사합니다.



### A Review of Deep Learning Approaches for Non-Invasive Cognitive Impairment Detection (https://arxiv.org/abs/2410.19898)
- **What's New**: 본 리뷰 논문은 비침습적 인지 장애 감지를 위한 딥러닝 접근 방식의 최근 발전을 탐구합니다.

- **Technical Details**: 다양한 비침습적 인지 감소 지표(예: speech and language, facial, motoric mobility)를 검토하며 관련 데이터 세트, 특징 추출 기법 및 딥러닝 아키텍처를 소개합니다. 다양한 방법의 성능을 비교 분석하였으며, 일반적으로 speech 및 language 기반 방법이 가장 높은 탐지 성능을 보였습니다. 음향 및 언어적 특성을 결합한 연구가 단일 모달리티를 사용하는 연구보다 더 나은 성능을 보였습니다. 그러나 facial 분석 방법은 제한적으로 연구되었습니다.

- **Performance Highlights**: 대부분의 연구는 임계 분류 (impaired vs. non-impaired)에 초점을 맞추었으며, 다중 클래스 또는 회귀 작업을 다룬 연구는 적었습니다. 전이 학습(transfer learning)과 사전 훈련된 언어 모델(pre-trained language models)이 특히 언어 분석에 효과적이라는 점에서 인기를 끌었습니다. 그러나 데이터 표준화, 모델 설명 가능성, 장기 분석의 한계 및 임상 적용과 같은 여러 도전 과제가 여전히 남아 있습니다. 미래의 연구 방향으로는 언어 비의존적(spoken) 음성 분석 방법 조사, 다중 모달 진단 시스템 개발, AI 지원 의료에서의 윤리적 고려 사항을 다루는 것이 제안됩니다.



### TBBC: Predict True Bacteraemia in Blood Cultures via Deep Learning (https://arxiv.org/abs/2410.19887)
Comments:
          12 pages

- **What's New**: 이 논문에서는 혈류 감염(bacteraemia)의 진단을 위한 머신러닝 모델을 개발하여 응급실에서의 진단 효율성을 향상시키려는 새로운 접근 방식을 제시합니다.

- **Technical Details**: St. Antonius Hospital의 응급실 데이터를 이용하여 향상된 머신러닝 기법인 CatBoost와 Random Forest를 선택하고, Optuna를 이용한 모델 최적화를 통해 Random Forest 모델의 ROC AUC 지표 0.78을 달성하였습니다. 이 모델은 테스트 세트에서 0.92의 민감도를 나타냈습니다.

- **Performance Highlights**: 이 모델은 저위험(low risk) bacteraemia 환자의 36.02%를 정확하게 식별할 수 있었고, 오탐률(false positive rate)은 0.85%로 매우 낮았습니다. 이 연구의 결과는 혈액 배양(blood cultures), 의료 비용, 항생제 치료의 감소 가능성을 보여줍니다.



### Parameter-Efficient Fine-Tuning in Large Models: A Survey of Methodologies (https://arxiv.org/abs/2410.19878)
- **What's New**: 이 리뷰 논문은 Parameter-Efficient Fine-Tuning(PEFT)에 대한 종합적인 개요를 제공하며, 대규모 모델이 특정 작업에 적합하도록 조정하는 다양한 알고리즘을 설명합니다. 이 논문은 최근 PEFT 기술의 발전과 응용 분야를 탐구하며, 향후 연구 방향에 대한 제안을 포함합니다.

- **Technical Details**: PEFT는 대규모 사전 훈련된 모델의 매개변수를 새로운 작업이나 시나리오에 적합하게 조정하는 전이 학습(transfer learning) 방법이다. 주요 PEFT 접근 방식으로는 LoRA, adapter tuning, prefix-tuning, prompt-tuning, P-tuning, BitFit 등이 있다. 이 기법들은 기존의 매개변수를 유지하면서도 성능 향상을 목표로 한다.

- **Performance Highlights**: PEFT는 여러 NLP 작업에서 성능을 최적화할 수 있는 잠재력을 가지고 있으며, 수백 개의 PEFT 관련 논문이 출판되었습니다. 이 리뷰는 PEFT 방법론의 현재 이해를 돕고, 앞으로의 연구를 위한 유용한 정보와 통찰을 제공하는 것을 목표로 합니다.



### Paved or unpaved? A Deep Learning derived Road Surface Global Dataset from Mapillary Street-View Imagery (https://arxiv.org/abs/2410.19874)
- **What's New**: 새로운 공개 데이터셋은 1억 5천만 개의 이미지를 활용하여 도로 표면 특성에 대한 정보를 제공합니다. 이 데이터셋은 세계 최대의 크라우드소싱 기반 거리 뷰 플랫폼인 Mapillary에서 수집되었으며, 최첨단 지리공간 AI 방법을 활용했습니다.

- **Technical Details**: 하이브리드 딥러닝 접근법을 제안하며, SWIN-Transformer 기반 도로 표면 예측과 CLIP-and-DL 세분화 기반의 품질 나쁜 이미지 필터링 방법을 결합했습니다. 도로 표면 예측 결과는 OpenStreetMap (OSM) 도로 기하학과 통합되었습니다.

- **Performance Highlights**: 모델 검증 결과 OSM 표면 데이터에 대한 성능이 우수하여 포장 도로에 대한 F1 점수는 91-97%를 기록했습니다. 글로벌 도로 표면 정보의 가용성을 3백만 킬로미터 이상 확장하며, 주요 응용 분야로는 도시 계획, 재난 라우팅, 물류 최적화 등이 있습니다.



### Evaluating Deep Learning Approaches for Predictions in Unmonitored Basins with Continental-scale Stream Temperature Models (https://arxiv.org/abs/2410.19865)
Comments:
          47 pages, 12 figures, 7 tables, submitted to Water Resources Research

- **What's New**: 이 연구는 미국의 비모니터링 구역에서 머신러닝(ML) 모델을 이용하여 하천 수온 예측의 정확성을 보여주며, 데이터 기반의 "top-down" 접근 방식과 데이터가 적은 지역에서 모델을 전이하는 "bottom-up" 접근 방식을 비교합니다.

- **Technical Details**: 연구에서는 지역적 특성을 기준으로 사이트를 분류하는 "grouped" 모델링 기법을 평가하며, 입력 변수를 체계적으로 제거하면서 모델 복잡성, 예측 정확성, 적용 가능성의 균형을 살펴봅니다.

- **Performance Highlights**: 결과적으로, top-down 모델은 bottom-up 및 grouped 모델보다 우수하며, 동적 및 정적 입력 변수를 줄임으로써 더 많은 사이트에 대해 낮은 모델 복잡성과 계산 요구를 통해 허용 가능한 정확성을 도출할 수 있음을 보여줍니다.



### Real-Time Weapon Detection Using YOLOv8 for Enhanced Safety (https://arxiv.org/abs/2410.19862)
Comments:
          21 pages, 5 figures

- **What's New**: 이번 연구는 YOLOv8을 활용한 실시간 무기 탐지 AI 모델의 개발을 제시하며, 학교, 공항 및 대중 교통 시스템과 같은 공공 장소의 안전성을 높이는 데 중점을 두고 있습니다.

- **Technical Details**: 본 모델은 다양한 유형의 총기 및 날카로운 무기를 포함한 수천 개의 이미지로 구성된 포괄적인 데이터셋을 통해 훈련되었습니다. 평가 과정에서는 precision, recall, F1-score 및 mean Average Precision (mAP)과 같은 주요 지표를 사용하였으며, 다양한 Intersection over Union (IoU) 임계값에서 성능을 측정하였습니다.

- **Performance Highlights**: 우리의 YOLOv8 기반 무기 탐지 모델은 정확하고 효율적인 실시간 비디오 스트림 내에서 무기를 탐지할 수 있는 능력을 보여주었습니다. 시스템의 운영 효율성은 고속으로 프레임을 처리할 수 있는 능력으로 나타났으며, 이는 공공 장소에서의 안전성을 강화하는 데 기여할 것입니다.



### Multi-Modal Transformer and Reinforcement Learning-based Beam Managemen (https://arxiv.org/abs/2410.19859)
Comments:
          5 pages, 5 figures, IEEE Networking Letters

- **What's New**: 본 연구에서는 최근의 multi-modal transformer (MMT)와 reinforcement learning (RL)을 결합하여 동적 빔 인덱스 예측을 위한 2단계 빔 관리 방법을 제안합니다. 이전 연구에서 주로 빔 예측 정확도에만 초점을 맞춘 것과는 달리, 본 연구는 효율적인 빔 관리가 시스템의 처리량을 개선할 수 있음을 보여줍니다.

- **Technical Details**: 제안된 프레임워크는 MMT와 RL을 결합하여 다중 모드 신호 처리를 수행합니다. 첫 번째 단계에서는 사용 가능한 빔 인덱스를 여러 그룹으로 나누고 MMT를 사용하여 최적의 빔 그룹을 예측합니다. 두 번째 단계에서는 각 그룹 내에서 RL을 활용해 빠른 빔 의사결정을 수행하여 처리량을 극대화합니다. 이러한 구성은 종합적인 다중 모드 정보 처리와 동적 환경 적응의 이점을 동시에 활용합니다.

- **Performance Highlights**: 6G 데이터셋을 사용하여 테스트한 결과, 제안된 방법은 MMT만 사용하는 방법 및 RL만 사용하는 방법에 비해 더 높은 빔 예측 정확도와 시스템 처리량을 달성했습니다.



### Personalized Recommendation Systems using Multimodal, Autonomous, Multi Agent Systems (https://arxiv.org/abs/2410.19855)
- **What's New**: 이 논문은 다중 모드(multi-modal) 및 자율적인 다중 에이전트(multi-agent) 시스템을 이용한 고도화된 개인화 추천 시스템에 대해 설명합니다. 특히, Gemini-1.5-pro와 LLaMA-70B와 같은 미래지향적 AI 기술을 통합하여 전자상거래(e-commerce) 고객 서비스 경험을 개선하는 데 중점을 두고 있습니다.

- **Technical Details**: 시스템은 세 가지 에이전트로 구성되어 있습니다. 첫 번째 에이전트는 주어진 질문에 대한 적절한 제품을 추천하고, 두 번째는 추천된 제품에 대한 이미지를 기반으로 후속 질문을 제기하며, 세 번째 에이전트는 자율 검색을 수행합니다. 이 시스템은 실시간 데이터 수집, 사용자 선호 기반 추천 및 적응형 학습 기능을 갖추고 있습니다. Groq API를 사용하여 LPU(대형 프로세싱 유닛) 추론을 수행하며, 응답 시간을 수 밀리초로 줄임으로써 원활한 대화가 가능하게 합니다.

- **Performance Highlights**: 이 시스템은 사용자 맞춤형 쇼핑, 가격 비교 및 이미지 검색 기능을 통해 전자상거래 분야에 혁신적인 변화를 가져올 것으로 기대됩니다. 또한, Chatbot을 통한 문의 자동 해결 기능과 특정 제품에 대한 공공 관심을 기반으로 한 개인화 추천이 가능하여 고객 지원을 완전 자동화할 수 있습니다.



### Survival of the Fittest: Evolutionary Adaptation of Policies for Environmental Shifts (https://arxiv.org/abs/2410.19852)
Comments:
          Pubblished in ECAI 2024

- **What's New**: 본 논문은 새로운 정책 최적화 기법인 진화적 강인 정책 최적화(Evolutionary Robust Policy Optimization, ERPO)를 제안하여 RL(Robust Learning) 알고리즘의 한계를 극복하고자 합니다. 이 방법은 진화 게임 이론(Evolutionary Game Theory)에서 영감을 받아 환경 변동에 유연하게 대처합니다.

- **Technical Details**: ERPO는 환경에서의 최적 정책을 반복적으로 학습하며, 이는 탐색(Exploration)과 이전 최적 정책에 대한 이행(Adherence) 간의 균형을 조절하는 온도 매개변수(Temperature Parameter)를 사용하는 적응형 재훈련 알고리즘입니다. 정책 업데이트는 진화 게임 이론에서 사용되는 복제 동역학(Replicator Dynamics)의 구체적인 사례입니다. 이 알고리즘은 보상의 희소성(Sparsity Assumptions)에 대한 일반적인 가정을 가지고 있으며, 변동된 환경에서 최적 정책으로 수렴함을 보여줍니다.

- **Performance Highlights**: ERPO는 경로 탐색(Path Finding) 작업에서 여러 RL 및 Deep RL 알고리즘(PPO, A3C, DQN)에 비해 여러 시나리오와 인기 있는 환경에서 우수한 성능을 보여주었습니다. ERPO는 정책 적응(Policy Adaptation) 속도가 빠르고, 평균 보상이 높으며, 정책 적응에 드는 계산 비용을 줄였습니다.



### AEPL: Automated and Editable Prompt Learning for Brain Tumor Segmentation (https://arxiv.org/abs/2410.19847)
Comments:
          4 pages paper for ISBI2025

- **What's New**: 본 논문에서는 뇌 종양 분할을 위한 새로운 프레임워크인 Automated and Editable Prompt Learning (AEPL)을 제안합니다. AEPL은 다중 작업 학습(multi-task learning)과 프롬프트 학습(prompt learning)을 결합하여 종양의 등급을 분할 과정에 통합합니다. 이를 통해 수작업 프롬프트 입력 없이도 보다 정확한 분할을 가능하게 합니다.

- **Technical Details**: AEPL은 3D U-Net 구조를 기반으로 하며, U-Net 인코더, 종양 등급 분류기, 프롬프트 인코더 및 U-Net 디코더로 구성됩니다. AEPL은 다중 모달리티 MRI 입력을 처리하여 종양 등급 예측 및 분할 마스크 생성을 동시에 수행합니다. 예측된 종양 등급은 자동으로 생성된 프롬프트로 사용되어 분할 마스크를 생성하는데 도움을 줍니다.

- **Performance Highlights**: AEPL은 BraTS 2018 데이터세트를 사용하여 여러 최신 방법들보다 우수한 성능을 보여주었으며, 종양 등급 정보를 직접 통합하여 세밀한 분할 결과를 생성하였습니다. 비록 다수의 추가 실험이 필요하지만, AEPL은 뇌 종양 분할에 있어 높은 유용성과 임상적 가치를 갖고 있음을 입증하였습니다.



### Enhancing Trust and Safety in Digital Payments: An LLM-Powered Approach (https://arxiv.org/abs/2410.19845)
Comments:
          10 pages, 7 figures

- **What's New**: 디지털 결제 시스템에서 사기 탐지 메커니즘의 중요성이 강조되고 있으며, 인도의 통합 결제 인터페이스(UPI)를 사례로 하여 Google Pay (GPay)를 중심으로 한 접근 방식이 제안되었습니다.

- **Technical Details**: 이 연구는 대형 언어 모델(LLMs)을 활용하여 사기 분류의 정확성을 높이고, 이를 통해 인간 리뷰어가 사기 활동을 식별하고 완화할 수 있도록 돕는 디지털 어시스턴트를 설계했습니다. Gemini Ultra 모델은 사기 분류에서 93.33%의 정확도를 달성했으며, 분류에 대한 이유를 생성하는 정확도는 89%로 나타났습니다.

- **Performance Highlights**: 이 연구의 결과는 LLMs가 기존의 머신 러닝 모델을 보강하여 사기 검토의 효율성, 정확성, 품질 및 일관성을 높일 수 있는 잠재력을 보여주었습니다. 특히, LLM은 새로운 사기의 정확한 이유를 32% 발견하여 인간 리뷰어가 간과했던 부분을 보완했습니다.



### GreenEye: Development of Real-Time Traffic Signal Recognition System for Visual Impairments (https://arxiv.org/abs/2410.19840)
Comments:
          Published in Korea Software Congress (2023)

- **What's New**: 이번 연구에서는 시각장애인의 교통 신호 인식을 위한 GreenEye 시스템을 개발하였습니다. 이 시스템은 교통 신호의 색상을 인식하고 보행자가 횡단보도를 건너기 위해 남은 시간을 실시간으로 알려줍니다.

- **Technical Details**: 기존 연구는 주로 두 가지 교통 신호, 즉 초록불(Green)과 빨간불(Red)만을 인식하는 데 중점을 두었으나, GreenEye 시스템은 14가지의 다양한 클래스(class)를 인식하도록 설계되었습니다. 초기 훈련에서는 정밀도(precision)가 74.6%로 나타났으며, 클래스 간의 데이터 불균형이 인식률 저하에 기여했습니다. 따라서 추가적인 레이블링(labeling) 및 데이터베이스(database) 형성을 통해 각 클래스의 이미지 수를 안정화했습니다.

- **Performance Highlights**: 데이터 안정화 작업 후, 14개의 모든 클래스에서 뛰어난 정밀도 99.5%를 기록하였습니다. 이는 GreenEye 시스템의 성능을 크게 향상시킨 결과입니다.



### Multidimensional Knowledge Graph Embeddings for International Trade Flow Analysis (https://arxiv.org/abs/2410.19835)
- **What's New**: 이 연구는 고차원성, 우발성 및 강한 비선형성을 특징으로 하는 경제 데이터의 복잡한 역학을 이해하기 위한 새로운 접근 방식을 제시합니다. 특히 KonecoKG라는 지식 그래프 임베딩 모델을 활용하여 국제 무역 관계를 예측하고 이를 통해 무역 흐름을 최적화하는 방법을 탐구합니다.

- **Technical Details**: KonecoKG는 다차원 관계를 가진 경제 무역 데이터의 지식 그래프 표현으로, SDM-RDFizer를 사용하여 관계를 변환하고 AmpliGraph를 통해 지식 그래프 임베딩을 생성합니다. 연구는 경제 상호작용을 네트워크 구조 내에서 표현하며, 무역 네트워크 데이터셋을 활용하여 역사적 무역 패턴과 인접국의 무역 행동 통찰을 통합하여 미래 무역 기회를 예측하고자 합니다.

- **Performance Highlights**: 이 연구는 무역 흐름을 예측하기 위해 다차원 관계를 모델링하여 비선형성 문제를 해결하며, KonecoTradeFlow 온톨로지를 도입하여 국제 경제 양자 간 무역 데이터를 표현합니다. 결과적으로, 정책 입안자, 기업 및 투자자에게 국제 무역 전략을 최적화할 수 있는 귀중한 통찰을 제공하며 경제 성장 추세를 발견하는 데 기여합니다.



### GNNRL-Smoothing: A Prior-Free Reinforcement Learning Model for Mesh Smoothing (https://arxiv.org/abs/2410.19834)
- **What's New**: 본 논문에서는 기존의 지도 학습(supervised learning) 및 강화 학습(reinforcement learning)에 의존하는 동적 메시 스무딩 기법을 넘어, 데이터나 사전 지식 없이 학습할 수 있는 새로운 강화 학습 모델을 제안합니다. 이는 메시 최적화를 마르코프 결정 과정(Markov Decision Process)으로 형식화하여, 총 두 개의 에이전트(agents)를 훈련시킵니다.

- **Technical Details**: 제안된 모델은 그래프 신경망(Graph Neural Networks)과 강화 학습을 결합하여 똑똑한 노드 스무딩 에이전트(intelligent node smoothing agent)와 메시 연결성 개선 에이전트(mesh connectivity improvement agent)를 구현합니다. 이 두 에이전트는 Twin Delayed Deep Deterministic Policy Gradient와 Double Dueling Deep Q-Network를 사용하여 훈련됩니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 복잡한 3D 표면 메시에서 피처를 보존하는 스무딩을 달성했으며, 2D 메시에 대해서는 동적 스무딩 방법 중에서 최신 기술 대비 7.16배 빠른 결과를 보여주었습니다. 또한 연결성 개선 에이전트는 메시 품질 분포를 효과적으로 향상시킬 수 있음을 입증했습니다.



### Human-Centric eXplainable AI in Education (https://arxiv.org/abs/2410.19822)
Comments:
          Preprint. Under Review

- **What's New**: 이 논문은 교육 분야에서 Human-Centric eXplainable AI (HCXAI)의 중요성을 강조하며, AI 시스템의 설명 가능성(Explainability)과 신뢰성(Trustworthiness)이 어떻게 학습 결과를 향상시키고 사용자 간의 신뢰를 조성할 수 있는지를 탐구합니다.

- **Technical Details**: AI 기술의 발전에 따라, 개인화된 학습(pathways), 평가, 피드백 메커니즘과 같은 혁신적인 솔루션이 교육에 통합되고 있습니다. 대형 언어 모델(LLMs)인 GPT-3.5와 BERT가 이러한 혁신의 중심에 있으며, 복잡한 대화, 정보 요약, 개인화된 튜터링을 통해 학습자의 요구에 맞춰 적응할 수 있는 능력을 가지고 있습니다. 그러나 이러한 LLMs의 복잡성과 불투명성은 교육 맥락에서 해석 가능성의 문제를 야기합니다.

- **Performance Highlights**: 설명 가능한 AI는 교육자들이 학생의 강점과 약점을 식별하는 데 기여하며, 이를 통해 개인화된 학습 경험을 제공할 수 있습니다. LLMs가 학생의 응답을 분석하고 피드백을 제공함으로써, 교육자들은 효과적으로 교수법을 조정하고 다양성 있는 학습 요구에 부응할 수 있습니다. AI의 투명성을 높임으로써, 사용자들은 AI를 신뢰하고 이를 교육 과정의 지원 도구로 받아들일 수 있습니다.



### UniMTS: Unified Pre-training for Motion Time Series (https://arxiv.org/abs/2410.19818)
Comments:
          NeurIPS 2024. Code: this https URL. Model: this https URL

- **What's New**: 본 연구에서는 UniMTS라는 새로운 통합된 사전 훈련(pre-training) 절차를 도입합니다. 이는 모바일 및 웨어러블 디바이스로부터 수집된 모션 시계열 데이터의 다양한 변수를 아우르는 첫 번째 모델입니다.

- **Technical Details**: UniMTS는 대조 학습(contrastive learning) 프레임워크를 기반으로 하여 모션 시계열과 대량 언어 모델(large language models)로 풍부해진 텍스트 설명 간의 정렬을 수행합니다. 기존의 모션 스켈레톤 데이터를 바탕으로 시계열을 합성하며, 공간-시간 그래프 네트워크(spatio-temporal graph networks)를 통해 관절 간의 관계를 캡처하고 다양한 장치 위치에 대한 일반화를 지원합니다.

- **Performance Highlights**: UniMTS 모델은 18개의 모션 시계열 분류 벤치마크 데이터셋에서 우수한 일반화 성능을 보여주며, 제로샷(zero-shot) 설정에서 340%의 성능 향상, 몇 샷(few-shot) 설정에서 16.3%, 전체 샷(full-shot) 설정에서 9.2%의 성능 개선을 달성했습니다.



### ControlAgent: Automating Control System Design via Novel Integration of LLM Agents and Domain Expertis (https://arxiv.org/abs/2410.19811)
- **What's New**: ControlAgent는 기존의 인공지능 모델에 비해 제어 시스템 디자인의 자동화를 위한 혁신적인 접근 방식을 제공합니다. 이를 통해 전문적인 지식과 반복적 디자인 프로세스를 융합하여 인간의 개입 없이도 고품질의 제어 시스템을 설계할 수 있습니다.

- **Technical Details**: ControlAgent는 여러 개의 협력적인 LLM(대형 언어 모델) 에이전트를 통합합니다. 중앙 에이전트는 작업 분배를 담당하고, 각 작업에 특정한 에이전트는 다양한 시스템 및 요구사항에 대해 세부 제어 설계를 맡습니다. Python 기반의 계산 에이전트가 복잡한 계산을 수행하며, 이 모든 과정은 이전 디자인으로부터의 실시간 피드백을 통해 반복적으로 개선됩니다.

- **Performance Highlights**: ControlAgent는 전통적인 인간 개입 기반 디자인 방식에 비해 높은 성능과 견고성을 입증했습니다. 500개의 제어 과제를 포함한 ControlEval 데이터셋을 통해 테스트된 결과, ControlAgent가 LLM 기반 및 전통적인 도구 기반 방법들보다 우수하다는 것이 확인되었습니다.



### LocateBench: Evaluating the Locating Ability of Vision Language Models (https://arxiv.org/abs/2410.19808)
Comments:
          We release the dataset at this https URL

- **What's New**: LocateBench라는 새로운 벤치마크가 제안되어, 자연어 지시에 따라 이미지 내 객체를 탐지하는 능력을 평가하는 데 중점을 두고 있습니다. 이 데이터셋은 높은 품질과 전문가 주석을 특징으로 하며, 자주 사용되는 이전의 데이터셋과 차별화됩니다.

- **Technical Details**: LocateBench는 이미지, 설명 및 네 개의 후보 바운딩 박스를 포함한 다중 선택 질문 데이터셋으로 구성됩니다. 객체의 바운딩 박스는 COCO 데이터셋에서 특정 객체의 설명을 기반으로 필터링되어 신뢰성을 높인 데이터가 포함되어 있습니다. 이 데이터셋은 RefCOCO 시리즈 데이터셋에 기반하여 구축되었습니다.

- **Performance Highlights**: GPT-4o가 LocateBench에서 전반적으로 가장 높은 성능을 보였지만, 여전히 사람의 정확도(95%)에 비해 10% 이상 부족한 결과를 보였습니다. Gemini-1.5-pro는 특정 프롬프트 방법에 가장 민감하였으며, 그 외 Claude-3 Opus와 Llava-1.6 모델은 두 작업 모두에서 가장 낮은 성능을 기록했습니다.



### First-Person Fairness in Chatbots (https://arxiv.org/abs/2410.19803)
- **What's New**: 본 논문은 일반적인 채팅봇 사용의 공정성에 중점을 두고, 특히 사용자의 이름에 따른 편향을 분석합니다. "first-person fairness"라는 개념을 통해 사용자가 직접 겪는 공정성을 연구하며, 다양한 배경을 가진 사용자를 위한 질 높은 반응 제공을 목표로 합니다.

- **Technical Details**: 본 연구에서는 LLMRA(언어 모델 연구 보조자)를 활용하여 이름에 대한 민감성을 분석하고, 사용자 이름과 관련된 편향을 비공식적으로 평가합니다. 이를 위해 복잡한 패턴을 식별하고, 공공 및 개인 채팅 데이터를 통해 차별성을 확인하는 소프트웨어 시스템을 개발했습니다. 인공지능 태그와 결과를 독립적인 인간 평가와 비교하여 신뢰성을 확보했습니다.

- **Performance Highlights**: 채팅봇 응답에서 특정 사용자 이름에 따른 반응 차이를 보여줍니다. 예를 들어, 여성과 관련된 이름을 가진 사용자는 약간 더 친절하고 단순한 언어로 응답을 받는 경향이 있으며, 성별에 따라 주인공을 자주 고르는 패턴도 나타납니다. 이후의 훈련 개입(RL 포함)은 해로운 고정관념을 상당히 완화하는 것으로 나타났습니다.



### Single-word Auditory Attention Decoding Using Deep Learning Mod (https://arxiv.org/abs/2410.19793)
Comments:
          5 pages, 3 figures

- **What's New**: 이 논문은 특정 단어에 대한 청각적 주의력을 이해하기 위해 EEG(전자 뇌파 측정) 신호와 인지 반응을 비교하는 새로운 접근 방식을 제안합니다. 이는 기존의 청각 주의력 디코딩(AAD) 방법과 차별화되며, 단일 단어 청각 주의 분류 문제로 귀결됩니다.

- **Technical Details**: 저자들은 EEGNet 기반의 딥 러닝 모델을 사용하여 세 가지 상이한 패러다임(odd word category, competing speakers, competing speech streams)에서 청각 주의 신호를 분석했습니다. EEG 신호의 epoch을 주의 집중 여부에 따라 라벨링했고, 데이터의 비대칭성과 적은 샘플 수 문제를 해결하기 위해 데이터 증강 기법을 적용했습니다.

- **Performance Highlights**: 제안된 모델은 실제 경쟁 패러다임에서 최소 58%의 정확도를 달성해, 다양한 실험에서 피험자 독립적 평가에서 유망한 성과를 나타냈습니다. 이는 이 분야에서의 연구에 새롭고 중요한 기여를 하게 될 것입니다.



### Xeno-learning: knowledge transfer across species in deep learning-based spectral image analysis (https://arxiv.org/abs/2410.19789)
Comments:
          Jan Sellner and Alexander Studier-Fischer contributed equally to this work

- **What's New**: 이 논문에서는 'xeno-learning'이라는 새로운 개념을 제안합니다. 이는 한 종에서 다른 종으로 지식을 전이하는 방식으로, 특히 고유한 임상 데이터를 수집하기 어려운 상황에서 동물 데이터의 활용을 극대화하려는 접근법입니다.

- **Technical Details**: 이 연구는 11,268장의 hyperspectral imaging (HSI) 이미지를 사용하여 사람, 돼지, 쥐 모델 간의 스펙트럼 특성을 비교합니다. 기존의 모델에서는 한 종에서 훈련된 신경망이 다른 종의 조직을 분류하는 데 어려움을 겪으므로, 새로운 'physiology-based data augmentation' 방법을 통해 서로 다른 종 간의 상대 스펙트럼 변화를 학습하고 전이할 수 있는 방안을 제시합니다.

- **Performance Highlights**: 조직 분별 성능은 동물에서 학습한 데이터를 통해 향상되며, 동물 데이터와 인간 데이터를 결합한 신경망 훈련이 유의미한 성과를 보이지 않았습니다. 하지만 상대 스펙트럼 변화를 학습함으로써 특정 병리학적 상태에서의 신경망 성능을 개선할 수 있는 가능성을 보여줍니다.



### Deep Learning-driven Mobile Traffic Measurement Collection and Analysis (https://arxiv.org/abs/2410.19777)
Comments:
          MPhil thesis

- **What's New**: 본 논문은 기존의 연구들이 간과했던 이동 기지국 간의 동적 의존성을 정확하게 모델링하고, 효과적인 모바일 트래픽 분석 및 예측을 위한 딥러닝 기반의 방법론을 제안합니다.

- **Technical Details**: 우선 'Spider'라는 모바일 트래픽 측정 수집 및 재구성 프레임워크를 설계하여 측정 비용을 줄이고 높은 정확도로 트래픽 소비를 추론합니다. 긴밀한 조정을 통해 강화 학습( reinforcement learning ) 기법을 이용하여 목표 모바일 커버리지 영역의 하위 집합을 선택적으로 샘플링하는 에이전트를 훈련합니다. 이후 경량화된 신경망 모델을 통해 역사적 희소 데이터 기반으로 트래픽 소비를 재구성합니다. 또한 'SDGNet'이라는 핸드오버 인식 그래프 신경망 모델을 설계하여 시간에 따라 기지국 간의 의존성을 캡처합니다. 핸드오버 빈도 정보를 통해 사용자 이동성을 반영하여 예측의 정확성을 향상시킵니다. 이러한 방법들은 실제 모바일 트래픽 데이터셋에서 다른 벤치마크 모델들을 능가합니다.

- **Performance Highlights**: 'Spider' 프레임워크는 실제 모바일 트래픽 데이터셋에서 기존 솔루션을 초과하는 성능을 보이며, 'SDGNet' 모델은 주요 네트워크 운영자가 수집한 모바일 트래픽 데이터셋에서 다른 그래프 모델에 비해 뛰어난 성능을 입증합니다.



### Gender Bias of LLM in Economics: An Existentialism Perspectiv (https://arxiv.org/abs/2410.19775)
Comments:
          Gender Bias, Large Language Models, Decision-Making

- **What's New**: 이번 연구는 대형 언어 모델(LLMs), 특히 GPT-4와 BERT의 성별 편향이 경제적 의사결정에 미치는 영향을 분석했습니다. 이 연구는 수학적 증명과 Word Embedding Association Test (WEAT)를 통해 LLMs가 성별 관련 마커 없이도 성별 고정관념을 강화한다는 사실을 입증했습니다.

- **Technical Details**: LLMs의 성별 편향 검사는 통계적 편향 이론을 사용하여 이루어졌으며, 심화 신경망을 활용하여 대량 데이터에서 발생하는 편향을 분석했습니다. 연구에서는 LLMs가 특정 의사결정 시 성별 편향을 어떻게 유지하는지를 확인했습니다.

- **Performance Highlights**: 이 논문은 LLMs가 인간 의사결정자와의 비교 시 윤리적 사고와 개별적인 이해를 통해 편향을 초월할 수 있는 능력이 있음을 밝혔으며, LLMs는 편향된 데이터의 수학적 최적화 결과로서 편향을 지속적으로 유지하고 있음을 확인했습니다.



### Copula-Linked Parallel ICA: A Method for Coupling Structural and Functional MRI brain Networks (https://arxiv.org/abs/2410.19774)
Comments:
          25 pages, 10 figures, journal article

- **What's New**: 본 연구에서는 깊은 학습(frameworks) 기술과 copula, 독립 성분 분석(ICA)을 결합한 새로운 융합(fusion) 방법인 CLiP-ICA를 개발했습니다.

- **Technical Details**: CLiP-ICA는 기능적 자기 공명 영상(fMRI)과 구조적 자기 공명 영상(sMRI)의 독립적 원천을 추정하고, copula 기반 모델을 사용하여 fMRI와 sMRI의 공간적 원천을 연결하여 시간적 및 공간적 데이터를 더 유연하게 통합합니다.

- **Performance Highlights**: CLiP-ICA는 알츠하이머병 신경영상 이니셔티브(ADNI)의 데이터를 사용한 결과, 감각운동, 시각, 인지 제어 및 기본 모드 네트워크를 포함한 강하게 및 약하게 연결된 sMRI와 fMRI 네트워크를 효과적으로 캡처했습니다. 이 방법은 더 의미 있는 구성 요소를 발견하고, ICA의 최적 모델 순서 문제를 해결했습니다.



### Real-time Monitoring of Lower Limb Movement Resistance Based on Deep Learning (https://arxiv.org/abs/2410.19769)
Comments:
          17 pages paper

- **What's New**: 본 논문에서는 실시간 하체 이동 저항 모니터링을 위한 새로운 모바일 다중 과제 학습 네트워크(MMTL-Net)를 제안합니다. 이 네트워크는 MobileNetV3를 통합하여 효율적인 특징 추출을 수행하며, 저항 수준 예측과 활동 인식을 동시에 수행하는 다중 과제 학습을 활용합니다.

- **Technical Details**: MMTL-Net은 높은 정확도, 낮은 대기 시간, 개선된 계산 효율성을 제공하여 실제 응용 분야에 적합합니다. 모델은 UCI 인간 활동 인식 및 무선 센서 데이터 마이닝 활동 예측 데이터셋에서 기존 모델을 크게 초월하며, 저항 예측 정확도(Resistance Prediction Accuracy, RPA) 91.2%와 낮은 힘 오류율(Force Error Rate, FER) 6.8%를 달성했습니다. 또한, 실시간 응답성(Real-time Responsiveness, RTR)은 12 밀리초, 처리량(Throughput, TP)은 초당 33 프레임입니다.

- **Performance Highlights**: 실험 결과 MMlTL-Net 모델의 성능은 다양한 실제 시나리오에서 강력한 효과를 입증했습니다. 이는 재활 치료 및 스포츠 응용 분야에서 보다 효율적이고 정확한 시스템을 위한 길을 열어줍니다.



### Unraveling Movie Genres through Cross-Attention Fusion of Bi-Modal Synergy of Poster (https://arxiv.org/abs/2410.19764)
- **What's New**: 이 논문에서는 영화 포스터를 활용하여 멀티 라벨(movie genre classification) 장르 분류 문제를 해결하는 프레임워크를 제시합니다. 기존의 접근 방식들은 주로 줄거리 요약(plot summaries), 자막(subtitles), 예고편(trailers), 영화 장면에 초점을 맞추었으나, 포스터에서 시각적 및 텍스트적 정보를 효율적으로 활용하여 장르 식별을 시도한 것이 특징입니다.

- **Technical Details**: 영화 포스터에서 텍스트를 추출하기 위해 OCR(Optical Character Recognition)을 사용했으며, 관련 임베딩(embedding)을 회수하였습니다. 이후, 크로스 어텐션 기반(fusion module) 융합 모듈을 도입하여 시각적 및 텍스트 임베딩에 주의 가중치를 할당합니다. 13882개의 포스터를 IMDb(Internet Movie Database)에서 수집하여 실험을 수행하였고, 다양한 현대 아키텍처와 비교해 우수한 성능을 보였음을 확인하였습니다.

- **Performance Highlights**: 우리 모델은 복수 장르 식별(multi-label genre identification) 작업에서 주요 최첨단 방법보다 꾸준히 우수한 성능을 보여주었으며, 각 모듈의 유용성을 입증하는 어블레이션(ablation) 연구 결과가 포함되어 있습니다.



### Reliable, Routable, and Reproducible: Collection of Pedestrian Pathways at Statewide Sca (https://arxiv.org/abs/2410.19762)
Comments:
          arXiv admin note: text overlap with arXiv:2303.02323

- **What's New**: 이 논문에서는 장애인을 위한 이동성 형평성을 증대시키기 위한 자동화된 보행 경로 데이터 수집 및 관리 기법을 제안합니다. 기존 데이터 수집의 한계를 극복하기 위해 공중 영상과 도로 네트워크 데이터를 결합한 새로운 방법론을 도입하여 워싱턴주 전역의 보행 경로를 생성하는 목표로 합니다.

- **Technical Details**: 논문에서는 Prophet 시스템이라는 자동화된 예측 모델을 사용하여 공중 이미지에서 보행 경로, 길 건너기, 경계턱을 추출하고, 이를 통해 연결된 경로 네트워크를 생성합니다. 전문가의 수동 검증을 위한 단계적 프로세스인 Skeptic을 운영하여 데이터 품질을 보장하고 있으며, 다수의 전문가와 지역 사회의 의견을 통합하여 데이터 수집의 정확성을 높이고 있습니다.

- **Performance Highlights**: 자동화된 시스템이 최신의 방법들보다 우수한 성능을 보여주며 인간 검증에 필요한 시간을 상당히 줄일 수 있음을 입증했습니다. 이 연구는 주 전체에서 신뢰할 수 있고 강력한 보행 경로 네트워크를 생성할 수 있는 가능성을 보여주며, 이러한 접근법은 전국적인 ADA 준수 절차를 알리는 데 기여할 것입니다.



### Movie Trailer Genre Classification Using Multimodal Pretrained Features (https://arxiv.org/abs/2410.19760)
- **What's New**: 본 논문은 다양한 사전 훈련된(pretrained) 모델을 활용하여 영화 장르 분류에 대한 새로운 방법론을 제시합니다. 이 방법은 모든 비디오 및 오디오 프레임을 사용하여 장르를 예측하며, 기존의 전통적인 방법보다 더 높은 정확성을 보입니다.

- **Technical Details**: 우리의 접근 방식은 transformer 모델을 사용하여 다양한 작업과 양식에서 오는 사전 훈련된 특징(feature)을 융합합니다. 이 모델은 비디오 예고편의 모든 프레임과 오디오를 처리하며, 고정되지 않은 많고 다양한 특징을 효과적으로 처리합니다. 특히, 전통적인 방식이 사용하는 고정된 낮은 수의 프레임과는 다릅니다.

- **Performance Highlights**: 본 연구는 MovieNet 데이터셋에서 기존의 영화 장르 분류 모델보다 정밀도(precision), 재현율(recall), 평균 평균 정밀도(mean average precision, mAP)에서 월등한 성과를 기록하였습니다. 이를 통해, 향후 연구를 위한 사전 훈련된 특징과 모델 코드를 공개하여 연구자들이 활용할 수 있도록 하였습니다.



### PINNing Cerebral Blood Flow: Analysis of Perfusion MRI in Infants using Physics-Informed Neural Networks (https://arxiv.org/abs/2410.19759)
- **What's New**: 이 연구에서는 아기 ASL 데이터에서 뇌혈류(CBF) 및 기타 매개변수를 정확하게 추정하기 위해 새로운 공간 불확실성 기반 물리정보 신경망(PINN)인 SUPINN을 제안합니다.

- **Technical Details**: SUPINN은 여러 브랜치 아키텍처(multi-branch architecture)를 사용하여 여러 복셀(voxel)에서 지역적 및 글로벌 모델 매개변수를 동시에 추정합니다. 이 네트워크는 지역적인 공간 불확실성을 계산하여 신호의 가중치를 조정합니다. SUPINN은 CBF를 신뢰성 있게 추정하며, 이는 -0.3 ± 71.7의 상대 오차로 나타납니다.

- **Performance Highlights**: SUPINN은 기존의 최소제곱법(least squares)이나 표준 PINN을 사용한 매개변수 추정보다 우수한 성능을 보였으며, 생리학적으로 그럴듯한 공간적으로 부드러운 CBF 및 AT 맵을 생성할 수 있습니다. 이 연구는 아기 ASL 데이터에서 시끄럽고 제한된 정보를 바탕으로 정확한 다중 매개변수 관류 추정을 위한 PINN의 성공적인 수정 사례를 보여줍니다.



### A SAM based Tool for Semi-Automatic Food Annotation (https://arxiv.org/abs/2410.19756)
Comments:
          Accepted Demo Paper - ECAI 2024

- **What's New**: 이번 논문에서는 Segment Anything Model (SAM)을 활용한 반자동 음식 이미지 주석 도구의 데모를 제시합니다. 사용자가 상호작용을 통해 음식 분할을 할 수 있도록 하여, 비전문가도 쉽게 사용할 수 있는 AI 도구의 필요성을 강조합니다.

- **Technical Details**: 제안된 도구는 사용자의 요청에 따라 음식 이미지를 신속하게 분할해 주며, MealSAM이라는 fine-tuned 된 SAM의 mask decoder를 사용하여 음식 이미지 분할에 최적화되어 있습니다. 이 도구는 ViT-B 백본을 사용하고 있습니다.

- **Performance Highlights**: 본 연구의 목표는 음식 데이터의 주석 작업을 통해 참여와 협업을 증진시키고, AI 기술을 보다 많은 사용자에게 제공하여 영양 과학 분야에서도 쉽게 활용할 수 있도록 하는 것입니다.



### A Comparative Analysis on Ethical Benchmarking in Large Language Models (https://arxiv.org/abs/2410.19753)
Comments:
          62 pages

- **What's New**: 이번 연구는 지능형 시스템이 인간의 가치관을 정확하게 표현하고 그에 맞게 행동하는지를 평가하기 위한 Machine Ethics (ME) 벤치마킹 분야에 기여합니다. 기존 ME 벤치마크의 세 가지 주요 문제점을 제시합니다: 비현실적인 윤리적 딜레마로 인한 제한된 생태적 타당성, 명확한 포함/제외 기준이 없는 비구조적 질문 생성, 인간 주석에 의존하는 확장성 부족입니다. 이를 해결하기 위해 Triage Benchmark와 Medical Law (MedLaw) Benchmark라는 두 개의 새로운 ME 벤치마크를 소개합니다.

- **Technical Details**: MedLaw Benchmark는 완전히 AI가 생성한 벤치마크로, 의료 분야에서의 실제 윤리적 딜레마를 다룹니다. 또한, 모델의 최악의 경우 성능을 평가하기 위해 컨텍스트 변화를 도입했습니다. 연구 결과, 윤리적 프롬프트가 항상 의사결정을 개선하지 않으며, 컨텍스트 변화가 모델 성능을 크게 감소시키거나 오류 패턴을 역전시키고 상대적 성능 순위를 변화시킬 수 있다는 것을 발견했습니다.

- **Performance Highlights**: 우리의 분석에 따르면, 일반 모델 능력이 항상 강력한 윤리적 의사결정을 예측하지는 않는 것으로 나타났습니다. 우리는 ME 벤치마크가 견고한 평가를 보장하기 위해 실제 시나리오와 최악의 경우 성능을 근사해야 한다고 주장합니다.



### The Geometry of Concepts: Sparse Autoencoder Feature Structur (https://arxiv.org/abs/2410.19750)
Comments:
          13 pages, 12 figures

- **What's New**: 이 논문은 Sparse Autoencoders (SAE)를 활용하여 개념의 우주를 형성하는 고차원 벡터의 구조를 세 가지 스케일에서 분석하는 내용을 담고 있습니다. 특히, 이 구조에는 '원자' 규모(atomic scale)에서 '결정'(crystal), '두뇌' 규모(brain scale)에서 기능적 모듈성, 그리고 '은하' 규모(galaxy scale)에서는 비등방(isotropic)적이지 않은 분포가 포함됩니다.

- **Technical Details**: 논문은 세 가지 주요 스케일에서의 구조를 탐구합니다: 1) '원자' 스케일은 패러렐로그램(parallelogram) 또는 사다리꼴(trapezoid) 형태의 결정 구조를 분석합니다. 2) '두뇌' 스케일은 기능적으로 유사한 SAE 특징들이 공간적으로 함께 모여 '로브'(lobe)를 형성하는지 조사합니다. 3) '은하' 스케일에서는 특징 점군의 고유값(eigenvalue) 분포가 중간 레이어에서 가장 가파른 기울기를 보이는 파워 법칙(power law)을 따른다고 명시합니다.

- **Performance Highlights**: 이 연구는 Linear Discriminant Analysis (LDA)를 통해 데이터에서 전역 방해 요소(global distractor directions)를 제거하고, 클러스터의 품질을 크게 향상시킬 수 있음을 보였습니다. 따라서 기존의 결정 구조를 보다 명확하게 드러내는 결과를 도출했으며, 특히 기능적 특성 간의 높은 공간적 모듈성과 관련하여 관찰된 데이터는 임상 신경과학의 기존 연구와도 일치하는 경향을 보여줍니다.



### Using AI Alignment Theory to understand the potential pitfalls of regulatory frameworks (https://arxiv.org/abs/2410.19749)
- **What's New**: 이 논문은 인공지능의 기술적 정렬의 잠재적 위험을 탐구하는 Alignement Theory (AT)의 통찰을 바탕으로 유럽 연합의 인공지능 법(EU AI Act)을 비판적으로 검토합니다. 규제 노력과 고급 인공지능 시스템을 유사한 방식으로 취급함으로써, 이 법에서의 잠재적 취약성과 개선할 수 있는 영역을 식별하고자 합니다.

- **Technical Details**: 논문은 AT 연구에서 식별된 여러 주요 실패 방식들, 예를 들어 proxy gaming, goal drift, reward hacking, specification gaming 등을 규명하고, 이러한 방식들이 AI 시스템이 의도된 목표와 잘 정렬되지 않을 때 발생할 수 있음을 설명합니다. 유럽 연합의 인공지능 법은 위험 수준에 따라 AI 시스템을 분류하고, 이러한 위험을 완화하기 위한 요구 사항을 설정합니다.

- **Performance Highlights**: 이 논문은 EU AI Act가 실제 AI 응용 프로그램의 복잡성을 완전히 포용하지 않을 수도 있으며, AI 기술 발전의 빠른 진화에 효과적으로 대응하지 못할 가능성이 있음을 지적합니다. 특히 교육 분야에서 AI 시스템의 배치가 고위험으로 분류되며, 이는 개인의 자율성과 진로 기회를 결정할 수 있는 광범위한 영향력을 가질 수 있습니다.



### C^2DA: Contrastive and Context-aware Domain Adaptive Semantic Segmentation (https://arxiv.org/abs/2410.19748)
Comments:
          This paper has 16 pages, 6 figures, 5 tables. It has been accepted for publication at the International Symposium of Robotics Research (ISRR), Long Beach, California, USA, 2024

- **What's New**: 본 연구에서는 Unsupervised Domain Adaptive Semantic Segmentation (UDA-SS)의 새로운 프레임워크를 제안합니다. 이 프레임워크는 intra-domain 및 context-aware 지식을 학습하여 소스 도메인과 타겟 도메인 간의 데이터 이동 문제를 해결하고자 합니다.

- **Technical Details**: 제안된 방법에서는 contrastive loss를 소스 및 타겟 도메인에 통합하여 유사 클래스의 픽셀을 서로 밀어내고 다른 클래스의 픽셀은 떨어지도록 하는 방식을 사용합니다. 또한, ClassMix 기법을 수정하고, Mask Image Modeling (MIM) 기술을 채택하여 제한된 정보로부터 시각 인식을 향상시킵니다.

- **Performance Highlights**: GTA-V->Cityscapes와 Synthia->Cityscapes 데이터셋에서 0.51% mIoU 및 0.54% mIoU의 성능 향상을 보이며, 최첨단 UDA-SS 방법의 성능을 초월하였습니다.



### Metamizer: a versatile neural optimizer for fast and accurate physics simulations (https://arxiv.org/abs/2410.19746)
- **What's New**: 이 논문에서는 Metamizer라는 새로운 신경 최적화기를 소개합니다. 이는 물리 기반 손실 함수(physics-based loss function)를 최소화하여 다양한 물리 시스템을 높은 정확도로 반복적으로 해결할 수 있게 해줍니다.

- **Technical Details**: Metamizer는 스케일 불변 아키텍처(scale-invariant architecture)를 활용하여 경량화된 gradient descent 업데이트를 개선하며, 훈련 데이터 없이 물리 기반 손실에 직접 훈련됩니다. 이 네트워크는 하이퍼파라미터를 요구하지 않으며, 성공적인 일반화 능력을 보여줍니다.

- **Performance Highlights**: Metamizer는 Laplace, advection-diffusion, incompressible Navier-Stokes 방정식에 대해 훈련한 후, 머신 정밀도(machine precision)에 가까운 정확도를 달성하였으며, 훈련 중에 다루지 않았던 Poisson, wave, Burgers 방정식 등에도 일반화됩니다.



### Towards Next-Generation LLM-based Recommender Systems: A Survey and Beyond (https://arxiv.org/abs/2410.19744)
- **What's New**: 본 논문에서는 Large Language Models (LLMs)가 recommender system 분야에 어떻게 긍정적인 영향을 미치는지 분석하고, 기존 방식의 한계를 극복하기 위한 방법론을 제시합니다.

- **Technical Details**: LLMs의 능력 향상으로 추천 시스템이 사용자와 아이템을 보다 정확하게 표현하고 이해할 수 있는 가능성을 논의합니다. 또한, LLMs의 기능을 활용하여 기존 추천 시스템의 파이프라인을 확장하는 방법을 다룹니다.

- **Performance Highlights**: LLM 기반 추천 시스템이 연구에서 산업 적용으로의 격차를 좁히는 데 중요한 역할을 할 것으로 기대되며, 이를 위한 새로운 연구 기회와 도전 과제를 제안합니다.



### AppBench: Planning of Multiple APIs from Various APPs for Complex User Instruction (https://arxiv.org/abs/2410.19743)
- **What's New**: 이 논문에서는 여러 출처의 API를 계획하고 실행할 수 있는 대형 언어 모델(LLMs)의 능력을 평가하기 위해 최초의 벤치마크인 AppBench를 소개합니다.

- **Technical Details**: AppBench는 크게 두 가지의 주요 도전에 중점을 둡니다: 1) 그래프 구조(graph structures) - 일부 API는 독립적으로 실행될 수 있지만, 다른 API는 순차적으로 실행되어야 하므로 그래프처럼 실행 순서가 형성됩니다; 2) 권한 제한(permission constraints) - 각 API 호출을 실행할 수 있는 권한이 있는 출처를 식별해야 합니다. 이는 복잡한 사용자 지침을 효과적으로 처리하기 위해 반드시 필요합니다.

- **Performance Highlights**: 실험 결과, 9개의 서로 다른 LLM에서 가장 복잡한 지침을 처리할 때 GPT-4o는 단 2.0%의 성공률을 기록했습니다. 이는 기존의 최첨단 LLM들이 여전히 이 같은 복잡한 설정에서 큰 성과를 거두지 못하고 있음을 보여 줍니다.



### SALINA: Towards Sustainable Live Sonar Analytics in Wild Ecosystems (https://arxiv.org/abs/2410.19742)
Comments:
          14 pages, accepted by ACM SenSys 2024

- **What's New**: 본 논문에서는 SALINA(SustAinable LIve soNar Analytics)라는 지속 가능한 라이브 소나 분석 시스템을 소개합니다. 이 시스템은 극한의 환경 구성에서도 효과적인 실시간 소나 데이터 처리를 가능하게 합니다.

- **Technical Details**: SALINA는 소나 데이터 채널 인구, 공간-시간적 특징 적응과 같은 심층 신경망(DNN) 모델의 향상, 지속 가능한 소나 스트리밍 및 에너지 관리 모듈을 통합하여 설계되었습니다. 실시간 소나 분석을 위해 엣지-클라우드 협업을 활용하여 정확도와 전력 효율성을 균형 있게 조절합니다.

- **Performance Highlights**: SALINA는 캐나다 브리티시컬럼비아의 두 내륙 강에서 6개월 동안 연속 24/7 수중 모니터링을 제공했으며, 평균 정밀도에서 9.5% 개선과 추적 메트릭에서 10.1% 증가를 보여주었습니다. 에너지 관리 모듈은 극한 날씨에서도 시스템 중단을 방지하고 비상 비용을 줄이는 데 성공했습니다.



### Tourism destination events classifier based on artificial intelligence techniques (https://arxiv.org/abs/2410.19741)
- **What's New**: 이번 연구는 관광지 관리에 있어 고객의 요구를 파악하고 최적의 서비스를 제공하기 위한 새로운 자동 분류 프로세스를 제안합니다. 이 프로세스는 관광 이벤트를 계층적 분류법 (hierarchical taxonomy)을 통해 체계적으로 분류합니다.

- **Technical Details**: 연구에서 사용된 기술적 방법론은 CRISP-DM과 같은 데이터 과학 기법, 감독기계학습 (supervised machine learning), 자연어 처리 기술 (natural language processing techniques)을 포함합니다. 이러한 방법들은 서로 다른 지리적 지역에서 이벤트 정보를 표준화된 카탈로그 (catalog)로 생성하는 데 기여합니다.

- **Performance Highlights**: 이 자동 분류 도구는 항공사, 여행사, 호텔 체인과 같은 여러 지역에서 정보를 제공하는 기업들에게 매우 가치가 있으며, 이벤트 카테고리에 상관없이 사용자들이 원하는 이벤트를 쉽게 찾을 수 있도록 돕습니다. 결과적으로 이 도구는 기업과 최종 사용자가 관광 이벤트 정보를 상호작용하는 방식을 혁신할 잠재력을 지니고 있습니다.



### Analysis of Hopfield Model as Associative Memory (https://arxiv.org/abs/2402.04264)
Comments:
          35 pages, 23 figures, 3 codes

- **What's New**: 이 연구는 생물학적 신경 시스템에서 영감을 받은 Hopfield 신경망 모델에 대한 탐구를 다룹니다. 모델의 기초를 기계 통계학의 통찰을 통합하여 심화하고, 오디오 검색(audio retrieval)에 집중하여 Hopfield 모델의 연상 기억(associative memory) 기능을 보여줍니다.

- **Technical Details**: Hopfield 모델은 생물학적 신경의 복잡한 연결에서 영감을 받아 정보 처리를 효율적으로 모방합니다. 관련된 기본 개념인 action potential(활동 전위)을 이해하는 것이 필수적이며, 이 과정에서 나트륨(Na+), 칼륨(K+), 염소 이온(Chloride)과 같은 이온의 이동이 핵심 역할을 합니다. 인공 뉴런 부분에서는 McCulloch-Pitts 모델을 통해 신경망의 기본 기능을 설명하며, MP 뉴런 모델이 어떻게 기본적인 입력과 가중치를 통해 특정 기능을 수행하는지를 보여줍니다.

- **Performance Highlights**: 연구를 통해 Hopfield 신경망은 다양한 패턴을 검색하는 연상 기억 기능을 가지고 있으며, MP Multilayer Networks는 XOR 작업과 같은 복잡한 기능을 수행하는 데 필수적이라는 점이 강조됩니다.



### Fast Inference for Augmented Large Language Models (https://arxiv.org/abs/2410.18248)
- **What's New**: 이번 논문에서는 LAMPS라는 새로운 추론 프레임워크를 제안하여 고급 대형 언어 모델(LLM)의 요청 완료 시간을 최소화합니다. LAMPS는 API 호출 및 메모리 처리 전략을 통합하여 요청의 총 길이를 고려한 통합 스케줄링 접근 방식을 채택하고 있습니다.

- **Technical Details**: LAMPS는 각 요청의 메모리 소비를 예측하기 위해 입력 프롬프트에서의 사전 API 출력 길이와 API의 특성을 기반으로 예측 모델을 개발합니다. 이를 통해 Preserve, Discard and Recompute, Swap과 같은 메모리 처리 전략을 정하고, 요청를 스케줄링하여 효율적으로 관리합니다.

- **Performance Highlights**: LAMPS는 기존의 LLM 추론 시스템과 비교하여 최종 대기 시간(end-to-end latency)을 27%에서 85%까지 감소시키고, TTFT(Time To First Token)에서는 4%에서 96%까지 줄이는 등의 성과를 보였습니다. 또한 vLLM에 비해도 더 큰 개선 효과를 나타냈습니다.



New uploads on arXiv(cs.LG)

### Online Weighted Paging with Unknown Weights (https://arxiv.org/abs/2410.21266)
- **What's New**: 본 논문은 페이지 가중치가 사전에 알려지지 않은 상태에서도 학습을 통해 최적의 솔루션을 제공할 수 있는 온라인 가중 페이지 교체 알고리즘(Online Weighted Paging with Unknown Weights, OWP-UW)을 제시합니다.

- **Technical Details**: OWP-UW 문제는 각 페이지가 무작위 분포를 가지며, 페이지를 가져올 때마다 독립적인 샘플을 관측합니다. 이를 통해 가중치에 대한 정보를 실시간으로 학습합니다. 기존 연구들은 사전에 가중치가 알려져 있다고 가정했으나, 본 알고리즘은 이를 통해 더 동적으로 대처합니다.

- **Performance Highlights**: 새롭게 제안된 알고리즘은 O(log k) 경쟁 비율과 O~(nT)의 회귀(regret) 최적화를 지원하며, 실용적인 멀티코어 아키텍처와 같은 복잡한 환경에서의 성능을 추구합니다.



### Modular Duality in Deep Learning (https://arxiv.org/abs/2410.21265)
- **What's New**: 이번 논문에서는 일반 신경망에 대한 모듈화된 이중화(modular dualization) 맵을 제시합니다. 이는 빠르고 확장 가능한 교육 알고리즘의 이론적 기초를 형성합니다. 주된 초점은 그래디언트 업데이트가 항상 이중화 맵을 통해 처리되어야 한다는 것이며, 이를 통해 손실 함수의 비등방성을 반영하도록 그래디언트를 조정합니다.

- **Technical Details**: 이중화 맵의 구축 절차는 다음의 세 단계로 나뉩니다: 첫째, 각 레이어에 입력-출력 의미에 기반하여 연산자 노름(operator norms)을 할당합니다. 둘째, 이러한 연산자 노름을 바탕으로 개별 레이어에 대한 이중화 맵을 구축합니다. 셋째, 레이어별 이중화 맵과 신경 아키텍처의 구조를 토대로 전체 가중치 공간에 대한 이중화 맵을 재귀적으로 유도합니다.

- **Performance Highlights**: 제안된 알고리즘은 Embed, Linear, Conv2D 레이어에 대해 GPU 친화적인 방식으로 이중화 맵을 계산할 수 있게 해줍니다. 또한 새로운 사각형 뉴튼-슐츠(iteration) 방법을 이용하여 NanoGPT 교육을 위한 속도 기록을 세우는 데 성공했습니다. 이 연구가 미래의 빠르고 확장 가능한 최적화 알고리즘 설계에 기여하기를 기대합니다.



### BLAST: Block-Level Adaptive Structured Matrices for Efficient Deep Neural Network Inferenc (https://arxiv.org/abs/2410.21262)
- **What's New**: 이 논문에서는 대규모 파운데이션 모델의 효율적인 구조를 위한 Block-Level Adaptive Structured (BLAST) 매트릭스를 소개합니다. BLAST 매트릭스는 기존 구조화된 매트릭스보다 더 유연하게 다양한 구조를 표현할 수 있으며, 딥러닝 모델의 선형 레이어의 가중치 매트릭스에서 효율적인 구조를 학습하여 불필요한 계산 복잡성을 줄이고 성능을 개선합니다.

- **Technical Details**: BLAST 매트릭스는 가중치의 저차원 구조를 포착하기 위해 블록 행렬 사이에 공유 기저를 활용합니다. 이 구조는 저랭크(low-rank), 블록 저랭크(block low-rank), 블록 대각행렬(block-diagonal) 등의 다양한 구조를 포괄합니다. BLAST 매트릭스는 데이터로부터 학습하여 요인(factors)을 최적화할 수 있으며, 기존 가중치를 압축하고 재학습할 수 있는 알고리즘도 구현되어 있습니다.

- **Performance Highlights**: BLAST 매트릭스를 사용하여 ViT와 GPT-2 모델을 훈련할 경우 각각 70% 및 40%의 계산 복잡성을 줄이면서 성능을 개선하는 결과를 보여줍니다. 또한 Llama-7B 모델을 BLAST로 50% 압축하여 NVIDIA A100 GPU에서 상당한 추론 속도 향상을 달성하였습니다.



### Capacity-Aware Planning and Scheduling in Budget-Constrained Monotonic MDPs: A Meta-RL Approach (https://arxiv.org/abs/2410.21249)
- **What's New**: 본 논문에서는 예산과 수용 능력 제약을 가진 다중 구성 요소 단조 (monotonic) 마르코프 결정 프로세스 (MDP)를 해결하는 방법을 제안합니다. 특히, 수리 기술자 수와 총 수리 예산에 의해 제한된 대규모 산업 로봇을 위한 수리 일정을 계획하고 배분하는 문제를 다룹니다.

- **Technical Details**: 제안된 방법은 두 단계의 계획 접근 방식으로, 첫 번째 단계에서는 Linera Sum Assignment Problem (LSAP)를 사용하여 다중 구성 요소 MDP의 요소들을 그룹으로 분할합니다. 각 그룹에 대한 예산은 그룹의 크기에 비례하여 할당됩니다. 이후 각 그룹에 대해 메타 훈련된 PPO (Proximal Policy Optimization) 에이전트를 사용하여 근사 최적 정책을 도출합니다.

- **Performance Highlights**: 본 연구의 결과는 제안된 방법이 로봇 군집의 평균 가동 시간 증대에 있어 기준 접근 방식을 초월한 성능을 보여줌을 나타내며, 특히 큰 군집 크기에 대해 효과적임을 입증합니다.



### Flaming-hot Initiation with Regular Execution Sampling for Large Language Models (https://arxiv.org/abs/2410.21236)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 훈련 중 효율적인 샘플링 방법인 Flaming-hot Initiation with Regular Execution (FIRE) 샘플링을 소개합니다. 이 방법은 초기 토큰을 높은 온도에서 샘플링하여 우수한 응답을 찾아내는 방식으로, 다양한 도메인에서 세분화된 문제 해결 능력을 향상시키는 데 기여합니다.

- **Technical Details**: FIRE sampling은 높은 온도의 초기 토큰 샘플링과 일반적인 샘플링 프로세스를 결합하여 진행됩니다. 이 접근법은 attention sink 이론을 활용하며, 문제 해결 시 샌드박스 체커의 존재로 인해 더 나은 샘플을 생성할 수 있도록 돕습니다. FIRE는 CoT-decoding의 간소화 및 일반화를 통해 매개변수 조정 없이도 기존 훈련 프레임워크와 통합될 수 있습니다.

- **Performance Highlights**: FIRE sampling을 통해 여러 공개 소스 모델에서 추론 시간의 생성 품질이 개선되었고, 특히 수학 문제와 코드 생성에서 높은 통과율(pass rate)을 기록했습니다. 이 방법을 통해 생성된 샘플의 다양성이 향상되었으며, 이는 결과적으로 성능 개선과 직결됩니다.



### $\texttt{skwdro}$: a library for Wasserstein distributionally robust machine learning (https://arxiv.org/abs/2410.21231)
Comments:
          6 pages 1 figure

- **What's New**: 이 논문에서는 skwdro라는 파이썬 라이브러리를 소개합니다. 이 라이브러리는 robust machine learning 모델을 훈련시키기 위해 distributionally robust optimization (DRO) 및 optimal transport distances를 기반으로 설계되었습니다.

- **Technical Details**: skwdro는 scikit-learn 호환 추정기와 PyTorch 모듈의 래퍼를 제공하여 다양한 모델에서 최소한의 코드 변경으로 사용할 수 있게 합니다. 이 라이브러리는 원래의 robust objective의 entropic smoothing을 통해 최대한의 모델 유연성을 보장합니다. 이를 통해 Wasserstein distance (Wasserstein 거리)를 활용하여 데이터 불확실성을 포착하는 WDRO 프레임워크를 구현하고, 최적화 문제의 계산적 한계를 해결하는 방향으로 발전하였습니다.

- **Performance Highlights**: skwdro는 최소한의 코드 변경으로 logistic/linear 회귀와 같은 표준 모델을 robustify할 수 있도록 하며, 스무딩된 목표를 통해 PyTorch 모델에 대한 robustification을 용이하게 합니다. 이로 인해 모델 훈련을 위한 확률적 기법이 가능해지며, 이는 distributionally robust machine learning의 활용 가능성을 크게 넓힙니다.



### LoRA vs Full Fine-tuning: An Illusion of Equivalenc (https://arxiv.org/abs/2410.21228)
- **What's New**: 이 논문은 Low-Rank Adaptation (LoRA)와 전체 미세 조정(full fine-tuning) 방법이 학습한 솔루션의 본질적 차이를 다루고 있습니다. 연구 결과, 두 방법은 동일한 성능을 달성하더라도 모델의 weight matrices의 구조가 다르며, LoRA 모델에서 새로운 'intruder dimensions'가 나타나는 것을 보였습니다.

- **Technical Details**: LoRA에서는 weight matrix의 업데이트를 두 개의 저차원(low-rank) 행렬의 곱으로 표현합니다. 본 논문에서는 full fine-tuning과 LoRA가 만든 weight matrices의 singular value decomposition(특이값 분해)의 스펙트럼 특성을 분석하여 두 방법의 차이를 설명합니다. LoRA는 'intruder dimensions'라 불리는 새로운 특이 벡터를 포함하여 다른 구조의 매개변수를 생성합니다.

- **Performance Highlights**: LoRA와 전체 미세 조정은 동일한 목표 태스크에서 유사한 성능을 달성하지만, LoRA로 조정된 모델은 훈련 분포를 잊어버리고 새로운 태스크에 적응할 때 더 비효율적이라는 결과가 나타났습니다. 높은 차수의 LoRA 모델은 전체 미세 조정 모델과 유사한 구조적 특징과 일반화 성능을 보여줍니다.



### Reconstructing dynamics from sparse observations with no training on target system (https://arxiv.org/abs/2410.21222)
Comments:
          31 pages, 21 figures

- **What's New**: 본 논문에서는 훈련 데이터 없이 희소 관측(sparse observation)으로 시스템의 동역학(dynamics)을 재구성(reconstruction)하는 문제를 다루고 있습니다. 복잡하고 비선형(nonlinear) 시스템의 동역학을 정확하게 재구성하기 위한 하이브리드(transformer + reservoir-computing) 머신러닝 프레임워크가 제안되었습니다.

- **Technical Details**: 제안된 프레임워크는 두 가지 주요 요소로 구성됩니다: 첫째, 알려진 혼돈 시스템(chaotic systems)으로부터 생성된 합성 데이터(synthetic data)를 바탕으로 transformer를 훈련하고, 둘째, 훈련된 transformer의 출력을 reservoir computer에 연결하여 시스템의 장기 동역학을 예측합니다.

- **Performance Highlights**: 제안된 하이브리드 머신러닝 프레임워크는 지금까지 약 30개의 비선형 동적 시스템에 대해 시연되었으며, 데이터가 Nyquist 기준의 20%만 주어져도 높은 재구성 정확도(reconstruction accuracy)를 나타냈습니다. 이는 훈련 데이터가 존재하지 않고 관측이 무작위(확률적) 및 희소할 때도 복잡한 비선형 동역학을 재구성할 수 있는 새로운 패러다임을 제공합니다.



### SeriesGAN: Time Series Generation via Adversarial and Autoregressive Learning (https://arxiv.org/abs/2410.21203)
Comments:
          This work has been accepted at BigData 2024 on October 26, 2024, as a regular paper for oral presentation

- **What's New**: 이번 논문은 Generative Adversarial Network (GAN) 기반의 시간 시계열 생성 방식에서 발생하는 여러가지 문제점을 해결하기 위한 새로운 프레임워크인 SeriesGAN을 소개합니다. 이 프레임워크는 자동 인코더 기반의 임베딩 공간과 GAN의 적대적 훈련 역학을 통합하여 정보 손실을 최소화하고 시간 시계열 데이터를 고품질로 생성하는 데 최적화되었습니다.

- **Technical Details**: SeriesGAN은 각각 특성 공간(feature space)과 잠재 공간(latent space)에서 작동하는 두 개의 판별기(discriminator)를 활용하여 생성기(generator)에 대한 피드백을 선택적으로 제공합니다. 또한 시계열 데이터의 단계적 조건 분포를 포착하는 teacher-forcing 감독 네트워크를 통합하고, 자동 인코더를 위한 새로운 손실 함수(loss function)도 개발했습니다. 이러한 설정은 생성기와 자동 인코더의 결과를 정제하여 데이터 생성의 질을 향상시킵니다.

- **Performance Highlights**: SeriesGAN은 다양한 실제 및 합성 다변량(multivariate) 및 단변량(univariate) 시계열 데이터셋에서 테스트한 결과, 기존 최첨단 성능을 가진 모델인 TimeGAN을 포함한 다른 방법들에 비해 정성적 및 정량적으로 모든 측면에서 우수한 성능을 보여주었습니다.



### Resilience in Knowledge Graph Embeddings (https://arxiv.org/abs/2410.21163)
- **What's New**: 최근 지식 그래프(Knowledge Graphs)와 관련된 연구의 진전을 보여주는 본 논문에서는, KGE (Knowledge Graph Embedding) 모델의 회복력(resilience)을 정의하고 이와 관련된 기존 연구들을 종합적으로 조사하였다.

- **Technical Details**: 회복력의 정의는 (i) 일반화 일관성, (ii) 도메인 적응, (iii) 성능 일관성, (iv) 견고성(robustness), (v) 정보 누락 처리 등 여러 요소를 포함한다. 이를 바탕으로 KGE 모델에 대한 회복력을 다각적으로 분석하였다.

- **Performance Highlights**: 기존 연구들은 주로 KGE 모델의 견고성에 집중해왔으나, 본 논문은 회복력의 다양한 측면들을 동시에 고려함으로써 KGE 모델의 성능을 향상시키는 새로운 방향을 모색하고 있다.



### Trajectory Flow Matching with Applications to Clinical Time Series Modeling (https://arxiv.org/abs/2410.21154)
Comments:
          NeurIPS 2024 Spotlight

- **What's New**: 이번 논문에서는 Trajectory Flow Matching (TFM)이라는 새로운 기법을 제안하여 Neural stochastic differential equations (Neural SDEs)의 훈련 방식을 개선합니다. TFM은 SDE 동역학을 통한 역전파(backpropagation)를 우회하여 시뮬레이션 없이 Neural SDE를 훈련시킬 수 있는 방법입니다.

- **Technical Details**: TFM은 생성 모델링에서의 flow matching 기법을 활용하여 시계열 데이터(time series data)를 모델링합니다. 논문에서는 TFM이 시계열 데이터를 학습하는 데 필요한 조건을 수립하며, 훈련 안정성을 높이기 위한 재매개변수화 트릭(reparameterization trick)을 제시합니다.

- **Performance Highlights**: TFM을 임상 시계열 설정(clinical time series setting)에 적용하여, 세 가지 임상 시계열 데이터 세트(dataset)에서 성능과 불확실성 예측 관점에서 향상된 성과를 보여줍니다.



### Offline Reinforcement Learning With Combinatorial Action Spaces (https://arxiv.org/abs/2410.21151)
- **What's New**: 이번 논문에서는 Branch Value Estimation(BVE)라는 새로운 방법론을 제안합니다. BVE는 기존의 방법들이 가정하던 sub-action의 독립성을 넘어서서, sub-action 간의 의존성을 효과적으로 캡쳐하여 대규모의 조합적(action space) 액션 공간에서 학습할 수 있도록 합니다.

- **Technical Details**: BVE는 조합적 액션 공간을 트리 구조로 형성하여 sub-action 간의 의존성을 나타냅니다. 각 노드는 특정 sub-action 조합을 나타내며, 각 엣지는 특정 sub-action에 고유한 값을 부여합니다. BVE는 행동 정규화 TD 손실 함수(behavior-regularized TD loss function)를 통해 학습하며, 각 타임스텝마다 트리를 탐색하여 최적의 액션을 선택합니다.

- **Performance Highlights**: 실험 결과, BVE는 16부터 400만 이상에 이르는 다양한 액션 공간 크기에서 기존의 최신 방법론보다 뛰어난 성과를 보였습니다. BVE는 복잡한 sub-action 의존성을 처리하며, 조합적 액션 공간에서의 학습 효과를 극대화했습니다.



### LLM-initialized Differentiable Causal Discovery (https://arxiv.org/abs/2410.21141)
- **What's New**: 본 연구에서는 LLM-DCD라는 새로운 방법론을 제안합니다. 이 방법은 대형 언어 모델을 활용하여 DCD 접근법의 최대 우도 목표 함수 최적화를 초기화하여, 강력한 사전 정보를 발견 방법에 통합합니다.

- **Technical Details**: LLM-DCD의 초기화 과정에서, 인과 그래프의 명시적으로 정의된 인접 행렬에만 의존하는 목표 함수를 설계하였습니다. 이를 통해, 직관적이고 해석 가능한 인과 발견 방법을 제공합니다.

- **Performance Highlights**: LLM-DCD는 주요 벤치마크 데이터 세트에서 최신 기술 대비 더 높은 정확도를 보여주며, 초기화의 품질이 DCD 접근법의 최종 출력 품질에 직접적인 영향을 미친다는 실증적 증거를 제공합니다.



### Fast Calibrated Explanations: Efficient and Uncertainty-Aware Explanations for Machine Learning Models (https://arxiv.org/abs/2410.21129)
Comments:
          36 pages, 5 figures, journal submission

- **What's New**: 이 논문에서는 Fast Calibrated Explanations라는 새로운 방법을 소개합니다. 이 방법은 머신러닝 모델에 대한 신속하고 불확실성을 고려한 설명을 생성하기 위해 설계되었습니다. ConformaSight의 섭동 방식을 핵심 요소인 Calibrated Explanations에 통합하여 성능을 크게 향상시켰습니다.

- **Technical Details**: Fast Calibrated Explanations는 로컬 특징 중요도(local feature importance)와 교정된 예측(calibrated predictions)을 포함하며, 이러한 요소는 불확실성 정량화를 유지합니다. 이 방법은 분류(classification)와 임계 회귀(thresholded regression) 작업에서 목표가 사용자 정의한 임계값 위 또는 아래일 확률을 제공합니다.

- **Performance Highlights**: Fast Calibrated Explanations는 실시간 의사결정 과정에서 매우 유리하며, 긴급 대응 또는 자동 모니터링과 같이 짧은 시간 내에 중요한 결정을 요구하는 시나리오에서 신속한 설명 생성을 보장하여 시스템이 원활하게 작동할 수 있도록 합니다.



### FusedInf: Efficient Swapping of DNN Models for On-Demand Serverless Inference Services on the Edg (https://arxiv.org/abs/2410.21120)
- **What's New**: FusedInf는 DNN 모델을 효율적으로 교환하기 위해 여러 모델을 단일 Directed Acyclic Graph (DAG)로 통합하는 새로운 접근 방식을 소개합니다. 이는 GPU 메모리 로딩을 효율적으로 수행하고 실행 속도를 향상시킵니다.

- **Technical Details**: FusedInf는 GPU 메모리에 여러 모델을 로드하기 전에 단일 DAG를 컴파일하여 다수의 모델을 동시에 로드하고 쿼리하는 과정을 간소화합니다. 이를 통해 최대 14%의 실행 속도 향상과 17%의 메모리 요구 사항 감소를 달성했습니다. 또한, CUDA 최적화와 관련하여 다양한 DNN 모델의 효율성을 평가했습니다.

- **Performance Highlights**: FusedInf의 성능 평가 결과, 기존 DNN 모델들이 14% 더 빠르게 실행되고, 메모리 소모량은 17% 줄어들었습니다. 이는 Edge AI 시스템에 있어 모델 교환의 효율성을 크게 향상시키는 결과를 보여줍니다.



### Dual-Agent Deep Reinforcement Learning for Dynamic Pricing and Replenishmen (https://arxiv.org/abs/2410.21109)
- **What's New**: 이 논문에서는 비일관한 의사결정 빈도에 따른 동적 가격 책정 및 보충 문제를 연구합니다. 기존의 수요 가정과는 달리 가격의 함수로서 포아송 분포 내 매개변수가 도입되어 문제 분석의 복잡성이 증가합니다.

- **Technical Details**: 본 연구는 의사결정 트리 기반의 머신 러닝 접근법을 통합하여 수요 모델을 개선하고, 두 가지 시간 규모의 확률적 근사 기법을 활용해 가격 책정과 보충 간의 의사결정 빈도 차이를 해결하며, 심층 강화 학습 (Deep Reinforcement Learning, DRL) 기법을 적용합니다.

- **Performance Highlights**: 단일 및 복수 제품 시나리오에서의 수치 결과는 제안된 방법의 효과성을 입증하며, 가격 책정 및 보충 전략에서 수요 예측과 추가 가격 최적화를 포함하는 공동 전략이 재고를 줄이고 이익을 높이는 것으로 나타났습니다.



### Tree-Wasserstein Distance for High Dimensional Data with a Latent Feature Hierarchy (https://arxiv.org/abs/2410.21107)
- **What's New**: 본 논문에서는 고차원 데이터에 대한 새로운 트리-바흐슈타인 거리(tree-Wasserstein distance, TWD)를 제안하며, 두 가지 핵심 요소를 포함하고 있습니다. 첫째, TWD는 잠재적(feature hierarchy) 특성 계층을 가진 데이터에 맞게 설계되었습니다. 둘째, 기존의 TWD 사용 방식과는 달리, 우리는 TWD를 계산하는 데 사용되는 고유한 트리를 통해 잠재적 특성 계층을 학습합니다.

- **Technical Details**: 제안된 방안은 두 단계로 구성됩니다. 첫 번째 단계에서는 확산 기하학(diffusion geometry)을 사용하여 고차원 특성을 계속적인 하이퍼볼릭 공간(hyperbolic spaces)으로 임베딩합니다. 두 번째 단계에서는 하이퍼볼릭 임베딩으로부터 하향식(bottom-up) 접근 방식으로 잠재적 특성 계층을 표현하는 트리를 구성합니다. 이를 통해, 우리는 고차원 데이터에 대해 TWD를 생성하고, 기존의 TWD보다 효율적이고 확장 가능성을 보여줍니다.

- **Performance Highlights**: 우리의 TWD는 단어-문서(word-document) 데이터셋과 단일 세포 RNA 시퀀싱(single-cell RNA-sequencing) 데이터셋에서 뛰어난 분류 성능을 나타내며, 기존의 TWD 기반 방법과 사전 훈련된 모델 기반 응용 기법들보다 유리한 결과를 보여줍니다.



### Shallow Diffuse: Robust and Invisible Watermarking through Low-Dimensional Subspaces in Diffusion Models (https://arxiv.org/abs/2410.21088)
- **What's New**: 이 논문에서는 AI 생성 콘텐츠 식별을 위한 새로운 워터마킹 기법인 Shallow Diffuse를 소개합니다. 기존 방법들과 달리 Shallow Diffuse는 샘플링 과정 전반에 걸쳐 워터마킹을 통합하는 것이 아니라, 이미지 생성 과정에서 저차원 부분공간을 활용하여 이 두 단계를 분리합니다. 이는 워터마크가 이미지 생성 과정과 효과적으로 분리되도록 보장합니다.

- **Technical Details**: Shallow Diffuse는 높은 강건성과 일관성을 갖는 워터마킹 기법으로, 서버 시나리오(서버가 초기 시드를 제공할 때)와 사용자 시나리오(생성된 이미지에 워터마크를 삽입할 때) 모두에 적용 가능합니다. 이 기법은 이전 방식들과 달리 초기 랜덤 시드에 워터마크를 삽입하는 것이 아니라, 이미지 생성 과정에서 낮은 차원의 부분공간을 활용하여 워터마크의 상당 부분이 해당 부분공간의 널 공간에 위치하도록 합니다. 이는 샘플링 과정과 워터마킹 과정을 실질적으로 분리합니다.

- **Performance Highlights**: Shallow Diffuse는 기존의 워터마킹 방법들과 비교했을 때 큰 강건성과 일관성을 보여줍니다. 실험 결과는 이 방법이 기존 방법들보다 우수한 성능을 발휘함을 입증하였으며, 상당한 양의 데이터 생성을 일관되게 유지하고 워터마크의 감지 가능성을 향상시켰습니다.



### Skip2-LoRA: A Lightweight On-device DNN Fine-tuning Method for Low-cost Edge Devices (https://arxiv.org/abs/2410.21073)
Comments:
          ASP-DAC 2025 (accepted)

- **What's New**: 본 논문에서는 경량의 파인튜닝(automatic tuning) 방법인 Skip2-LoRA를 제안합니다. 이 방법은 임베디드 시스템에서의 사전 훈련(pre-trained) 및 배포(deployed) 모델 간의 간극을 해소하기 위해 설계되었습니다.

- **Technical Details**: Skip2-LoRA는 LoRA(low-rank adaptation) 어댑터를 마지막 층과 모든 다른 층 사이에 삽입하여 네트워크의 표현력을 높이는 동시에, 역전파(backpropagation)의 계산 비용을 낮추는 새로운 아키텍처를 활용합니다. 이 아키텍처는 중간 계산 결과를 캐시(caching)할 수 있도록 해 주며, 학습 에폭(epoch)이 진행됨에 따라 이미 본 샘플에 대한 피드 포워드(forward pass) 계산을 생략할 수 있습니다.

- **Performance Highlights**: Skip2-LoRA는 동일한 수의 훈련 가능한 파라미터를 갖는 기존 방법과 비교할 때 평균적으로 90.0%의 파인튜닝 시간을 단축시키며, 정확도(accuracy)를 유지합니다. 또한, $15의 단일 보드 컴퓨터에서 몇 초 만에 완료할 수 있음을 보여줍니다.



### Federated Time Series Generation on Feature and Temporally Misaligned Data (https://arxiv.org/abs/2410.21072)
- **What's New**: Federated time series data의 처리를 위한 새로운 모델 FedTDD가 제안되었습니다. 이 모델은 클라이언트 간의 시간 정렬 및 피처 정렬 문제를 해결하기 위해 설계되었습니다.

- **Technical Details**: FedTDD는 클라이언트 간의 합성 출력(synthetic outputs)을 교환함으로써 상관관계를 학습하는 새로운 데이터 증류(distillation) 및 집합(aggregation) 프레임워크를 도입합니다. 또한, Denoising Diffusion Probabilistic Model (DDPM)을 활용하여 시간적 의존성을 처리합니다.

- **Performance Highlights**: 다섯 개의 벤치마크 데이터셋에서 실험한 결과, FedTDD는 Context-FID 및 Correlational 점수에서 각각 79.4% 및 62.8% 향상을 보였으며, 중앙집중식 훈련(comparative to centralized training)과 유사한 성능을 달성했습니다.



### EMOCPD: Efficient Attention-based Models for Computational Protein Design Using Amino Acid Microenvironmen (https://arxiv.org/abs/2410.21069)
- **What's New**: 이번 연구에서는 기존의 비효율적인 기법을 극복하고 단백질 디자인의 정확성을 높이기 위해 아미노산 미세환경(aminо acid microenvironment)을 활용한 효율적인 주의 기반 모델(EMOCPD)을 개발하였습니다.

- **Technical Details**: EMOCPD는 3차원 원자 환경(three-dimensional atomic environment)을 분석하여 각 아미노산의 카테고리를 예측하고, 예측된 높은 확률의 아미노산 카테고리에 기반하여 단백질을 최적화합니다. 이 모델은 다중 주의 메커니즘(multi-head attention mechanism)을 사용하여 희소한 단백질 미세환경에서 중요한 특징을 강조하고, 역 잔여 구조(inverse residual structure)를 통해 네트워크 구조를 최적화합니다.

- **Performance Highlights**: EMOCPD는 훈련 세트(training set)에서 80% 이상의 정확도를 달성했으며, 두 개의 독립 테스트 세트(test sets)에서도 각각 68.33%와 62.32%의 정확도를 기록하며 기존 기법들보다 10% 이상 높은 성능을 보여주었습니다. 예측된 돌연변이(mutants)의 열적 안정성(thermal stability) 및 단백질 발현(protein expression)에서는 야생형(wild type)보다 유의미한 향상이 있었습니다.



### Computable Lipschitz Bounds for Deep Neural Networks (https://arxiv.org/abs/2410.21053)
- **What's New**: 이 논문은 Deep Neural Networks (딥 뉴럴 네트워크)의 Lipschitz constant (립시츠 상수)의 불확실성을 줄이기 위해 $l^1$ 및 $l^{
fty}$ 노름을 사용한 새로운 상한을 제안합니다.

- **Technical Details**: 기존의 $l^2$ 노름에 대한 상한을 분석하고, feed-forward fully-connected neural networks (피드포워드 완전 연결 신경망) 및 convolutional neural networks (CNN)의 경우에 대해서 새로운 두 가지 상한을 정의합니다. 각 신경망의 Lipschitz constant를 평가하기 위해 명시적(explicit) 및 암시적(implicit) 두 가지 방법을 사용하여 기술적 문제를 다룹니다.

- **Performance Highlights**: 네 가지 수치 실험이 진행되었으며, 그 중 하나는 analytical closed form (분석적 닫힌 형태)을 통해 정확한 값을 가진 Lipschitz constant에 대해 새로운 상한 K4가 최적임을 보여주었습니다. 새로운 상한 K4는 기존의 상한보다 우수한 정확도를 제공함을 확인했습니다.



### Getting By Goal Misgeneralization With a Little Help From a Mentor (https://arxiv.org/abs/2410.21052)
Comments:
          SATA Workshop @ NeurIPS 2024 (Towards Safe and Trustworthy Agents)

- **What's New**: 이 연구는 강화학습(즉, RL) 에이전트가 현업에서 배포될 때 경험하는 목표 비일반화(goal misgeneralization) 문제를 해결하기 위해 감독자에게 도움을 요청하는 접근 방식을 탐구합니다. 특히 CoinRun 환경에서 PPO를 사용하여 훈련된 에이전트를 대상으로 심화 연구를 진행하였습니다.

- **Technical Details**: 실험에서는 OpenAI의 procgen 패키지와 이를 기반으로 한 procgenAISC 패키지를 사용하였습니다. 목표 비일반화를 유발하는 coinrun 및 coinrun_aisc 환경에서 에이전트를 훈련시키고, 에이전트의 행동 분포에서 불확실성을 측정하여 언제 도움을 요청해야 하는지 판단하는 여러 방법을 도입하였습니다. 다섯 가지 방법(최대 확률, 최대 로짓, 샘플된 확률, 샘플된 로짓, 엔트로피)을 테스트하였습니다.

- **Performance Highlights**: 방법에 관계없이, 행동 분포의 불확실성을 기반으로 도움을 요청하는 방식은 도움 요청 없이 행동하는 경우보다 항상 더 높은 성능을 보였습니다. 특히, 에이전트는 훈련 중 동전을 찾는 위치 대신 동전이 없는 다른 위치에 도달했을 때 도움을 요청하는 경향이 있음을 발견하였습니다.



### Disentangled and Self-Explainable Node Representation Learning (https://arxiv.org/abs/2410.21043)
- **What's New**: 최근에 소개된 DiSeNE (Disentangled and Self-Explainable Node Embedding) 프레임워크는 비지도 학습 방식으로 자기 설명 가능한 노드 임베딩을 생성하는 방법론을 제시합니다. 이 방법은 각 차원이 그래프의 독립적인 위상 구조와 연관되도록 분리된 표현 학습 (disentangled representation learning)을 활용하여 임베딩을 제공합니다.

- **Technical Details**: 이 프레임워크는 각 차원이 그래프의 특정 서브 구조를 예측하도록 임베딩을 최적화하는 새로운 목적 함수를 도입하여 구조적 분리를 보장합니다. Entropy 기반 정규화 기법을 통해 중복 방지 및 비어있지 않은 매핑된 서브 구조의 형성을 보장합니다.

- **Performance Highlights**: DiSeNE는 여러 벤치마크 데이터셋에서 광범위한 실험을 통해 기존 방식보다 더 나은 성능을 입증하였으며, 새로운 평가 지표를 통해 자기 설명 가능한 노드 임베딩의 질을 측정합니다.



### Beyond Autoregression: Fast LLMs via Self-Distillation Through Tim (https://arxiv.org/abs/2410.21035)
- **What's New**: 이 연구에서는 기존의 Autoregressive (AR) 모델의 제한점을 극복하기 위해 Diffusion Language Models를 사용하여 동시에 최소 32개의 토큰을 생성할 수 있음을 보여줍니다.

- **Technical Details**: Self-Distillation Through Time (SDTT)라는 새로운 증류 방법을 도입하여, 디퓨전 모델이 더 적은 추론 단계(steps)로도 높은 품질의 텍스트를 생성할 수 있도록 하였습니다. 이 모델은 최대 860M 파라미터를 지원하며, KV 캐싱을 사용하는 AR 모델보다 최대 8배 더 빠릅니다.

- **Performance Highlights**: LAMBADA 자연어 이해 벤치마크에서 AR 모델보다 더 나은 텍스트 품질을 달성했습니다. 우리의 접근 방식을 통해 디퓨전 모델은 추론 단계 수를 32-64배 줄이면서도 빠른 속도와 높은 품질을 유지할 수 있습니다.



### Graph Based Traffic Analysis and Delay Prediction (https://arxiv.org/abs/2410.21028)
- **What's New**: 이 연구는 EU에서 가장 인구 밀도가 높은 나라인 몰타의 교통 혼잡 문제를 해결하기 위한 새로운 데이터 세트인 MalTra를 소개합니다. MalTra에는 200일 간의 실제 공공 이동 데이터가 포함되어 있으며, 다양한 방법론을 통해 데이터 수집의 정확성을 극대화하였습니다.

- **Technical Details**: 본 연구에서는 ARIMA 모델과 두 가지 그래프 신경망 모델인 STGCN(Spatial Temporal Graph Convolutional Network)과 DCRNN(Diffusion Convolutional Recurrent Network)을 사용하여 교통 데이터 분석을 수행했습니다. 데이터 세트의 품질을 높이기 위해 기존의 Q-Traffic 데이터 세트와 MalTra 데이터를 함께 활용하여 모델의 예측 성능을 비교하였습니다.

- **Performance Highlights**: DCRNN 모델은 MAE(Mean Absolute Error) 3.98 및 RMSE(Root Mean Squared Error) 7.78을 기록하며 STGCN 모델(각각 6.65, 12.73)보다 우수한 성과를 나타냈습니다. 이는 몰타의 트래픽 예측을 위한 그래프 기반 신경망의 유효성을 입증합니다.



### Transferable Post-training via Inverse Value Learning (https://arxiv.org/abs/2410.21027)
- **What's New**: 본 논문에서는 모델의 post-training 과정에서 발생하는 변화를 logits 수준에서 별도의 신경망(value network)을 이용해 모델링 하는 방법을 제안합니다. 이 네트워크는 소형 기본 모델에 대해 시연 데이터를 사용해 훈련된 후, 다른 선 훈련된 모델과 쉽게 통합되어 인퍼런스(inference) 동안 유사한 성능 향상을 이끌어낼 수 있습니다.

- **Technical Details**: 제안된 프레임워크는 logits 공간을 공유 인터페이스로 활용하여 다양한 크기와 계열의 모델을 효율적으로 적응시키며, 두 가지 연결 아키텍처(Cascade 및 Residual)를 조사합니다. Residual 연결 방식이 이전 텍스트 입력만을 기반으로 residuals를 예측하여 더 우수한 전이 가능성과 효율성을 보임을 실험적으로 확인하였습니다.

- **Performance Highlights**: 결과적으로 이 프레임워크는 동일한 모델 계열 내에서 다양한 매개변수 크기를 가진 pre-trained 모델 간의 전이 가능성과, 다른 모델 계열 간의 전이 가능성을 보여주었으며, 특정 경우에는 전체 매개변수를 파인튜닝한 성능에 근접한 성과를 달성했습니다. 이는 이 방법이 효율적이며 실용적인 어플리케이션에서의 잠재력을 지니고 있음을 강조합니다.



### Physics-informed Partitioned Coupled Neural Operator for Complex Networks (https://arxiv.org/abs/2410.21025)
- **What's New**: 본 논문은 다중 연결된 하위 영역을 포함하는 복잡한 물리 시스템을 위한 Physics-Informed Partitioned Coupled Neural Operator (PCNO)를 제안하고 있습니다. 이는 기존의 Fourier Neural Operator (FNO)에 비해 전반적인 통합을 가능하게 하는 joint convolution operator를 내장하여 시뮬레이션 성능을 향상시킵니다.

- **Technical Details**: PCNO는 Fourier layer 내에서 joint convolution operator를 설계하고, 그리드 정렬 레이어를 도입하여 주파수 영역에서 하위 영역 간의 결합 관계를 정확하게 학습할 수 있도록 돕습니다. 이는 단일 공간 영역이 아닌 연계된 다중 하위 영역에서의 효율적인 시뮬레이션을 가능하게 합니다.

- **Performance Highlights**: 자연 가스 네트워크를 대상으로 한 실험에서 PCNO는 복잡한 시스템을 정확하게 시뮬레이션하고, 일반화 성능이 뛰어나며, 낮은 모델 복잡도를 증명하였습니다. 실험 결과, 물리 정보와 관찰 데이터를 모두 포함한 훈련이 고해상도 예측에서 더 나은 성능을 나타냅니다.



### A Review of Graph-Powered Data Quality Applications for IoT Monitoring Sensor Networks (https://arxiv.org/abs/2410.21006)
Comments:
          Paper submitted to Journal of Network and Computer Applications

- **What's New**: 이번 논문은 IoT(Internet of Things) 모니터링 센서 네트워크에서의 데이터 품질 개선을 위한 그래프 기반 모델의 최신 동향과 응용에 대해 다룹니다. 특히 그래프 신호 처리(GSP)와 그래프 신경망(GNN)을 활용한 데이터 품질 제어에 대한 심층 분석을 제공합니다.

- **Technical Details**: 이 논문은 GSP와 GNN을 포함하여 그래프 위의 신호 처리 및 기계 학습에 관련된 기술적 개념을 다룹니다. 주요 기술들은 누락된 값 보완(missing value imputation), 이상치 탐지(outlier detection), 그리고 가상 센서(virtual sensing)와 같은 데이터 품질 작업을 위한 그래프 기반 솔루션 제공에 중점을 둡니다. 또한, 그래프 토폴로지(graph topology)를 활용한 다양한 작업들이 설명됩니다.

- **Performance Highlights**: 그래프 기반 모델은 전통적인 데이터 기반 접근 방식에 비해 성능, 분산 접근 가능성(distributed approaches), 그리고 해석 가능성(interpretability) 측면에서 우수한 결과를 보여줍니다. 이 연구는 IoT 모니터링 센서 네트워크에서의 신규 응용 프로그램 및 기술적 트렌드를 다루어 향후 발전 가능성을 제시합니다.



### Refining CART Models for Covariate Shift with Importance Weigh (https://arxiv.org/abs/2410.20978)
- **What's New**: 이 논문은 의료 분야에서의 예측 정확도를 향상시키기 위해 중요 가중치(importance weighting)를 적용한 분류 및 회귀 나무 모델(CART)의 적응을 소개합니다. 이 방법은 훈련 샘플 중에서 타겟 분포와 밀접하게 연관된 샘플에 더 큰 가중치를 부여하여 분포 차이를 효과적으로 해결합니다.

- **Technical Details**: 이 연구는 데이터 분포 간의 차이에 대처하기 위해 CART를 수정했습니다. 중요 가중치를 통하여 타겟 셋을 잘 대표하는 훈련 샘플에 더 많은 중요성을 부여함으로써 예측 정확도를 개선했습니다. 또한 도메인 적응(domain adaptation)의 기법을 사용하여 훈련 데이터와 테스트 데이터의 특징 분포 차이를 잘 처리할 수 있도록 했습니다.

- **Performance Highlights**: 시뮬레이션 연구와 실제 의료 데이터에 대한 적용을 통해, 이 가중치 CART 접근 방식이 예측 정확도에서 유의미한 개선을 보여주었습니다. 이 결과는 의료 및 다양한 데이터 분포가 존재하는 다른 분야에서 이 접근 방식이 유용할 수 있음을 시사합니다.



### Simultaneous Unlearning of Multiple Protected User Attributes From Variational Autoencoder Recommenders Using Adversarial Training (https://arxiv.org/abs/2410.20965)
- **What's New**: 본 연구에서는 여러 보호된 속성(예: 성별 및 나이)을 동시에 비학습(또는 학습하지 않도록)하는 AdvXMultVAE 모델을 제시합니다. 이는 기존의 모델들이 단일 속성만을 제거하는 데에 그쳤던 것과 차별화됩니다.

- **Technical Details**: AdvXMultVAE는 Variational Autoencoder (VAE) 아키텍처와 적대적 훈련(Adversarial Training)을 결합하여, 사용자의 보호된 속성을 연속적 및 범주적값으로 동시에 제거하는 방법을 지원합니다. 실험은 LFM-2b-100k와 Ml-1m 두 개의 데이터셋에서 수행되었습니다.

- **Performance Highlights**: 실험 결과, AdvXMultVAE는 성별 및 나이 제거 측면에서 다른 방법들보다 뛰어난 성능을 보였으며, NDCG와 Recall을 유지하면서 인종적 편견을 효과적으로 완화할 수 있음을 나타냈습니다.



### Neural Hamilton: Can A.I. Understand Hamiltonian Mechanics? (https://arxiv.org/abs/2410.20951)
Comments:
          33 pages, 8 figures, 9 tables

- **What's New**: 본 연구에서는 클래식 역학(classical mechanics)을 연산자 학습 문제(operator learning problem)로 재정의하는 새로운 신경망(neural network) 프레임워크를 제안합니다. 이는 해밀턴 방정식(Hamilton equations)을 명시적으로 풀지 않고도 잠재 함수(potential function)를 해당하는 위상 공간(phase space)에서의 경로로 직접 매핑(mapping)합니다.

- **Technical Details**: 본 연구에서는 두 가지 새로운 신경망 아키텍처인 VaRONet과 MambONet을 소개합니다. VaRONet은 변분 LSTM(Variational LSTM) 시퀀스-투-시퀀스 모델을 적응시키고, MambONet은 해밀턴 시스템의 효율적인 시간 동역학을 처리하기 위해 Mamba 모델을 활용합니다. 다양한 1D 물리 문제(harmonic oscillation, double-well potentials 등)를 통해 이 접근 방식을 실험하였으며, 전통적인 4번째 차수 룽게-쿠타(RK4) 알고리듬과 비교했습니다.

- **Performance Highlights**: MambONet 모델은 전통적인 RK4 방법보다 높은 정확도를 보이며, 계산 효율성(computational efficiency) 면에서도 경쟁력 있는 성능을 나타냅니다. 모델 훈련이 더 많은 데이터셋에서 이루어질수록 성능 향상이 두드러집니다. 이 연구는 해밀턴 방정식을 해결할 수 있는 머신러닝 기법의 잠재력을 보여주며, 물리학 및 AI 분야 간의 새로운 대화를 열어줍니다.



### Constrained Optimal Fuel Consumption of HEV:Considering the Observational Perturbation (https://arxiv.org/abs/2410.20913)
- **What's New**: 본 논문은 관측의 왜곡이 있는 상태에서 배터리 충전 상태(SOC)와 속도 데이터를 고려한 최적 연료 소모 문제(COFC)를 다루고 있습니다. 이를 위해 다양한 강화학습(Deep Reinforcement Learning) 기법을 사용하여 SOC 및 속도의 불확실성을 극복하는 접근법을 제시합니다.

- **Technical Details**: COFC 문제는 하이브리드 전기차(HEV)에서 연료 소모를 최적화하는 핵심 문제로, 본 연구에서는 구조화된 관측 왜곡 문제(COFC-OP 문제)를 정의하고, 균등 분포 공격에서 시작하여 다양한 공격 시나리오를 고려한 강화학습 알고리즘을 제공합니다. 논문에서는 Proximal Policy Optimization(PPO)과 Constrained Variational Policy Optimization(CVPO) 같은 기법이 활용됩니다.

- **Performance Highlights**: 실험 결과, 제안된 여섯 개의 접근법이 SOC와 속도 측정의 불확실성이 존재하는 상황에서도 COFC 문제를 성공적으로 해결할 수 있음을 입증하였습니다. 이는 연구자와 엔지니어에게 연료 소모를 최소화하면서 안전하고 견고한 차량 운영을 평가할 수 있는 기회를 제공합니다.



### Deep Recurrent Stochastic Configuration Networks for Modelling Nonlinear Dynamic Systems (https://arxiv.org/abs/2410.20904)
- **What's New**: 이 논문은 비선형 동적 시스템 모델링을 위한 새로운 딥 레저버 컴퓨팅(Deep Reservoir Computing) 프레임워크인 DeepRSCN을 제안합니다. DeepRSCN은 모든 레저버 노드가 최종 출력에 직접 연결되어 있으며, 감독 메커니즘에 의해 무작위 매개변수가 할당됩니다.

- **Technical Details**: DeepRSCN은 점진적으로 구성되며, 출력 가중치는 projection 알고리즘을 사용하여 온라인으로 업데이트됩니다. 훈련 샘플 집합을 통해 빠르게 학습 표현을 생성할 수 있으며, 이는 무작위 기초 함수와 연속된 입력 및 출력 가중치로 구성됩니다.

- **Performance Highlights**: 실험 결과, DeepRSCN은 시간 시계열 예측, 비선형 시스템 식별 및 산업 데이터 예측 분석에서 단층 네트워크보다 모델링 효율성, 학습 능력 및 일반화 성능에서 우수한 성능을 보여줍니다.



### Generative Example-Based Explanations: Bridging the Gap between Generative Modeling and Explainability (https://arxiv.org/abs/2410.20890)
- **What's New**: 본 논문은 고차원 입력 데이터에 대한 결정 알고리즘의 예제 기반 설명을 제공하기 위해 심층 생성 모델링(deep generative modeling)을 활용하는 새로운 확률적 프레임워크를 제안합니다. 이는 고전적인 설명 가능성 문헌과의 간극을 메우려고 하며, 높이 읽을 수 있는 설명을 고차원 데이터에 적합하게 하기 위해 기존의 로컬 설명 입장에서 주요 특징을 통합합니다.

- **Technical Details**: 이 프레임워크는 고충실도(counterfactual) 설명의 정의와 공격적(adversarial) 설명 그리고 사용자의 이해를 재확인하는 보완적 개념인 확증 설명(affirmative explanations)을 포함합니다. 고충실도 설명은 알고리즘의 결정을 바꾸는 인근의 고충실도 예제이고, 공격적 설명은 효과가 낮은 예제로 기술됩니다. 이 프레임워크는 설명을 위한 양적 평가 스키마를 제공하고, 복잡성과 직관적인 설명이 변하는 통제된 설명 환경을 위한 합성 데이터 세트를 제공합니다.

- **Performance Highlights**: 제안된 프레임워크는 새롭게 생성된 예제와 알고리즘의 결정을 보다 명확하게 연결 지을 수 있도록 할 뿐만 아니라, 다양한 신뢰성 있는 설명 예제를 통해 사용자와의 신뢰를 구축하는 데 기여합니다. 이는 사용자에게 더 나은 이해(means of explanation)를 제공하여, AI 시스템의 결정에 대한 투명성과 안정성을 향상시키는 데 도움을 줍니다.



### CODES: Benchmarking Coupled ODE Surrogates (https://arxiv.org/abs/2410.20886)
Comments:
          12 pages, 10 figures, accepted for the Machine Learning and the Physical Sciences workshop at NeurIPS 2024, source code available on GitHub at this https URL

- **What's New**: CODES는 coupled ODE 시스템을 위한 surrogate 아키텍처에 대한 포괄적인 평가를 제공하는 벤치마크로, 기존의 MSE(mean squared error)와 inference time 외에도 interpolation, extrapolation, sparse data, uncertainty quantification 및 gradient correlation과 같은 다양한 차원에서 surrogate의 동작을 분석합니다.

- **Technical Details**: CODES는 네 가지 surrogate 아키텍처(FCNN, MON, LNODE, LP)와 다섯 개의 데이터셋을 포함하며, 사용자 정의 config.yaml을 생성하는 웹 기반 구성 도구와 병렬 훈련 지원 기능을 포함합니다. 모든 모델은 세밀하게 문서화되어 있어 쉽게 사용할 수 있습니다.

- **Performance Highlights**: 모델은 osu2008 데이터셋을 기준으로 훈련되었으며, 모든 모델이 뛰어난 성능을 보였습니다. MON은 MSE, MAE 및 MRE와 같은 정확도 지표에서 가장 우수한 성과를 보였으며, LP는 가장 빠른 inference time을 기록했습니다. CODES는 surrogate 학습 행동을 이해하는 데 중요한 통찰을 제공합니다.



### Strada-LLM: Graph LLM for traffic prediction (https://arxiv.org/abs/2410.20856)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 이 연구는 Strada-LLM이라는 새로운 그래프 인지 (graph-aware) LLM을 제안하여 교통 예측 문제를 해결하고, 이와 관련된 여러 기술적 문제를 다루고 있습니다.

- **Technical Details**: Strada-LLM은 이웃 노드의 교통 정보를 공변량 (covariates)으로 고려하여 교통 예측을 수행합니다. 또한, 이 모델은 적은 레이블 데이터 샘플로 새로운 데이터 배포 (data distribution)에서 효과적으로 도메인 적응을 이루는 경량 (lightweight) 접근 방식을 채택합니다. 전통적인 GNN 기반의 방법과 비교하여 우수한 성능을 보여줍니다.

- **Performance Highlights**: Strada-LLM은 다양한 실제 교통 데이터 세트를 기반으로 평가한 결과, 예측 오류가 약 5%에서 18%까지 감소했습니다. 이는 그래프 인지 LLM 모델이 전통적인 방법들에 비해 우수한 교통 예측 성능을 보이도록 함을 입증합니다.



### On Probabilistic Pullback Metrics on Latent Hyperbolic Manifolds (https://arxiv.org/abs/2410.20850)
Comments:
          17 pages, 7 figures, 1 table

- **What's New**: 이 논문은 Gaussian Process Latent Variable Models (GPLVMs)를 통해 복잡한 고차원 데이터를 저차원 표현으로 캡처하는 새로운 방법을 제안합니다. 특히, hyperbolic manifold를 잠재 공간으로 활용하여 계층적 관계를 보다 잘 모델링할 수 있음을 입증하였습니다.

- **Technical Details**: 저자는 hyperbolic geometric에서 쓰이는 Riemannian pullback metric의 일반적인 구성 방식을 도입합니다. 이는 GPLVM을 비선형 매핑에 따른 왜곡을 고려하여 개선한 것입니다. 이를 통해 hyperbolic latent space에서의 geodesics가 데이터 분포와 일치하며 예측의 불확실성을 크게 줄입니다.

- **Performance Highlights**: 논문에서는 MNIST 데이터 보간, 다중 세포 로봇 설계, 인간의 그립 생성 등 다양한 실험을 통해 새로운 모델의 효과성을 입증하였으며, 기존 방법들과 비교해 더 낮은 불확실성을 보임을 확인하였습니다.



### Temporal Streaming Batch Principal Component Analysis for Time Series Classification (https://arxiv.org/abs/2410.20820)
- **What's New**: 이번 연구에서는 다변량 시계열 분류에서의 모델 성능을 최적화하기 위해, 장기 시계열 데이터의 영향을 완화하는 새로운 접근 방식을 제안합니다. 특히, Streaming PCA(Principal Component Analysis) 기반의 시계열 데이터 압축 및 차원 축소 알고리즘인 TSBPCA(Temporal Streaming Batch PCA)를 개발했습니다.

- **Technical Details**: 제안된 TSBPCA는 시계열 데이터의 시계적 업데이트를 통해 전체 시퀀스의 compact representation을 지속적으로 업데이트하여, 모델의 데이터 표현 능력을 향상시킵니다. 이는 기본 PCA 알고리즘에 기반하나, 동적 데이터 흐름을 고려하여 차원 축소를 수행하게 됩니다. 또한, 이 알고리즘은 주요 매개변수인 배치 크기 T와 주성분 개수 K의 최적 범위를 분석합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 여러 실제 데이터셋에서 분류 정확도와 처리 시간 모두에서 뛰어난 성능을 보였습니다. 특히, 가장 긴 시퀀스 데이터셋에서의 정확도가 약 7.2% 향상되었고, 실행 시간이 49.5% 감소했습니다. 이는 모델의 시간 효율성을 크게 개선함을 의미합니다.



### Bridging the Gap between Expert and Language Models: Concept-guided Chess Commentary Generation and Evaluation (https://arxiv.org/abs/2410.20811)
- **What's New**: 이 연구에서는 체스 전문가 모델과 대규모 언어 모델(LLMs) 간의 간극을 메우기 위한 새로운 접근 방식인 Concept-guided Chess Commentary generation (CCC)와 GPT-based Chess Commentary Evaluation (GCC-Eval)을 소개합니다. CCC는 전문가 모델의 결정 능력과 LLM의 언어적 유창성을 통합하여 체스 결정 과정을 설명하는 데 중점을 둡니다.

- **Technical Details**: CCC는 전문가 모델이 집중하는 개념을 추출하고 우선순위를 부여하여 LLM이 게임의 가장 중요한 측면에 집중하게 합니다. GCC-Eval는 체스 전문 지식을 활용하여 생성된 해설의 정보성 (informativeness)과 언어 품질 (linguistic quality)을 평가합니다.

- **Performance Highlights**: 실험 결과에 따르면 CCC는 인간 수준의 정확성을 달성하였으며, 인간이 생성한 코멘트보다 정보성과 언어적 품질에서 우수한 성능을 보여주었습니다. 또한, GCC-Eval는 사람의 평가와 잘 상관되어 더 신뢰할 수 있는 평가 지표로 기능하고 있습니다.



### zGAN: An Outlier-focused Generative Adversarial Network For Realistic Synthetic Data Generation (https://arxiv.org/abs/2410.20808)
- **What's New**: 본 논문에서는 "black swans" 현상이 고전적 머신러닝 모델 성능에 어떻게 도전하는지를 다루며, 전염병 이후의 특이값(outlier) 조건 증가가 인공지능 모델 훈련에 있어 실제 데이터 외에 합성 데이터(synthetic data)의 필요성을 강조합니다. 또한, zGAN 모델을 통해 합성 데이터 생성에 대한 실험적 조사를 수행하고, 이를 통해 모델 성능 향상 가능성을 보여줍니다.

- **Technical Details**: zGAN 모델 아키텍처는 합성 테이블 데이터 생성을 위한 Xavier 발산 네트워크(Generative Adversarial Network, GAN)로 설계되었으며, 생성된 데이터 내의 특성(feature) 간의 상관관계를 강화하는 기능을 제공합니다. zGAN은 데이터의 공분산(covariance) 기반으로 특이값을 생성할 수 있으며, Extreme Value Theory (EVT)에 근거하여 경량, 제한, 그리고 중량의 분포 꼬리(tail)를 모델링합니다. 또한, 비밀번호 필터를 사용하여 총본 데이터에서 클라이언트 정보를 보호하는 기능도 포함되어 있습니다.

- **Performance Highlights**: zGAN은 합성 데이터 생성 및 이로 인해 모델 성능이 향상된 점에서 좋은 결과를 보였습니다. 이 모델은 금융 서비스의 신용 리스크와 같은 민감한 데이터셋에서 실험되었으며, 원본 데이터와 합성 데이터 간의 상관분석을 통해 인과적 관계를 재현하고 검증했습니다. 또한, 이 연구는 AUC (Area Under Curve) 메트릭을 통해 합성 데이터의 품질 및 모델 강건성을 측정했습니다.



### Reduction-based Pseudo-label Generation for Instance-dependent Partial Label Learning (https://arxiv.org/abs/2410.20797)
Comments:
          Under Review

- **What's New**: 본 논문에서는 Instance-dependent Partial Label Learning (ID-PLL) 문제를 해결하기 위해, 잘못된 후보 레이블의 영향을 완화하는 방법으로 reduction-based pseudo-labels를 제안합니다. 이 접근법은 다중 분기 보조 모델의 출력을 가중 평균하는 방식으로 pseudo-label을 생성하여, 특정 레이블을 제외한 각 분기가 명확히 간섭을 피할 수 있도록 합니다.

- **Technical Details**: Reduction-based pseudo-labels는 다중 분기 보조 모델의 출력으로부터 생성되며, 각 분기는 특정 레이블을 제외한 라벨 서브스페이스에서 훈련됩니다. 이러한 방식은 제외된 레이블의 간섭을 피할 수 있게 해주며, 궁극적으로 각 인스턴스에 대해 보다 정확한 pseudo-label을 제공합니다.

- **Performance Highlights**: 이론적으로, 본 연구에서 제안된 pseudo-labels는 Bayes 최적 분류기와의 일관성이 더 뛰어난 것으로 나타났으며, 이는 모델이 잘못된 후보 레이블에 과적합되는 문제를 해결하는 데 도움을 줍니다.



### Adversarial Constrained Policy Optimization: Improving Constrained Reinforcement Learning by Adapting Budgets (https://arxiv.org/abs/2410.20786)
Comments:
          21 pages, 8 figures

- **What's New**: 이 논문에서는 Adversarial Constrained Policy Optimization (ACPO)를 제안하여 보상 및 비용 예산의 동시 최적화를 가능하게 합니다. 기존의 제약 강화 학습 방법의 한계를 극복하는 새로운 접근 방식을 제공합니다.

- **Technical Details**: ACPO는 원래 문제를 두 개의 적대적 단계로 분해하여 각각 보상을 극대화하고 비용을 최소화하는 방식으로 해결합니다. 이 과정은 이 두 단계를 번갈아 가며 해결하여 보상 및 비용 예산을 업데이트합니다. 이론적으로 알고리즘의 정책 업데이트 성능을 보장합니다.

- **Performance Highlights**: Safety Gymnasium과 네 발 보행 로코모션 작업에서의 실험 결과, ACPO는 동일한 비용 예산 하에서 일반적으로 사용되는 베이스라인보다 더 높은 보상을 달성하며 성능이 향상되었음을 보여줍니다.



### Introducing Spectral Attention for Long-Range Dependency in Time Series Forecasting (https://arxiv.org/abs/2410.20772)
Comments:
          Co-first Author: Bong Gyun Kang, Dongjun Lee

- **What's New**: 이 논문은 시간 시계열 예측(Time Series Forecasting, TSF)에서 긴 의존성(long-range dependencies)을 다루기 위한 새로운 방법인 Spectral Attention 기법을 제안합니다.

- **Technical Details**: Spectral Attention은 저역 통과 필터(low-pass filter)를 사용하여 긴 기간 트렌드(long-period trends)를 보존하고 모델의 기본 구조를 유지하면서 샘플 간의 시간적 상관관계를 보존합니다. 또한, Batched Spectral Attention은 여러 시점에서의 병렬 학습(parallel training)을 가능하게 합니다.

- **Performance Highlights**: 11개의 실제 데이터셋과 7개의 최신 예측 모델을 이용한 실험을 통해, Spectral Attention 메커니즘이 다른 모든 아키텍처에서 성능 향상을 이루어내고, 향상된 결과를 지속적으로 보여 주었습니다.



### Task Confusion and Catastrophic Forgetting in Class-Incremental Learning: A Mathematical Framework for Discriminative and Generative Modelings (https://arxiv.org/abs/2410.20768)
Comments:
          30 pages, 15 figures, Camera-Ready NeurIPS 2024

- **What's New**: 이 논문에서는 class-incremental learning (class-IL)에서의 task confusion (TC)을 정리하고, optimal class-IL을 위한 새로운 수학적 프레임워크를 제시합니다. 특히, discriminative modeling에서는 optimal class-IL이 불가능하다는 Infeasibility Theorem과 generative modeling을 통한 optimal class-IL이 가능함을 보여주는 Feasibility Theorem을 확립하였습니다.

- **Technical Details**: 기존의 incremental learning에서 catastrophic forgetting (CF) 이슈만 강조되었던 반면, 이 연구는 class-IL에서의 성능 저하가 주로 task confusion (TC)으로 인해 발생한다고 주장합니다. 새로운 수학적 프레임워크를 통해 TC와 CF를 구분하고, discriminative modeling과 generative modeling의 차이를 명확히 합니다. 논문에서는 각 모델을 조건부 확률 (conditional probability)과 결합 확률 (joint probability)로 분석하며, TC 및 CF를 해결하기 위한 훈련 문제를 정식화합니다.

- **Performance Highlights**: 연구 결과, generative modeling을 채택하는 것이 optimal class-IL를 달성하는 데 필수적이라는 인사이트를 제공합니다. 이 프레임워크는 regularization, bias-correction, replay 기반 방법 및 generative classifier와 같은 여러 class-IL 전략의 최적성을 논의하는 데 도움을 줍니다.



### ODRL: A Benchmark for Off-Dynamics Reinforcement Learning (https://arxiv.org/abs/2410.20750)
Comments:
          NeurIPS 2024 D&B Track

- **What's New**: 본 논문에서는 off-dynamics reinforcement learning (RL)에 대한 새로운 벤치마크인 ODRL을 소개합니다. 이 벤치마크는 다양한 도메인 간 정책 전이 및 동적 불일치 문제를 해결하기 위한 첫 번째 표준 평가 도구입니다.

- **Technical Details**: ODRL은 4개의 실험 설정을 포함하고 있으며, 각각의 소스 도메인과 타겟 도메인은 온라인 또는 오프라인 환경에서 작동할 수 있습니다. 이를 통해 다양한 작업과 광범위한 동적 변화에 적응할 수 있는 에이전트의 능력을 종합적으로 평가할 수 있습니다. 또한, ODRL은 최신 off-dynamics RL 알고리즘과 여러 기준선 메소드를 통합하여 단일 파일로 구현하여 공유하고 있습니다.

- **Performance Highlights**: 대규모 벤치마킹 실험을 통해 기존 방법들이 다양한 동적 변화 상황에서도 보편적인 장점을 인정받지 못한다는 점을 발견했습니다. ODRL은 미래 연구에서 중요한 초석이 될 것으로 기대하고 있습니다.



### Matryoshka: Learning to Drive Black-Box LLMs with LLMs (https://arxiv.org/abs/2410.20749)
Comments:
          Work in Progress

- **What's New**: 최근 대규모 블랙박스 언어 모델(LLM)의 기본적인 불투명성이 인지적 기능의 향상을 저해하고 있다는 문제 의식을 바탕으로, 새로운 컨트롤러 모델인 Matryoshika를 소개합니다. 이 모델은 복잡한 작업을 중간 출력으로 분해하여 블랙박스 LLM을 유도하는 경량의 화이트박스 LLM 컨트롤러입니다.

- **Technical Details**: Matryoshika는 두 가지 주요 구성 요소로 구성되며, 화이트박스 LLM이 컨트롤러로 작용하고 블랙박스 LLM이 생성기를 역할을 합니다. 입력 질문에 대해 컨트롤러는 중간 출력을 생성하고, 이는 블랙박스 LLM의 기능을 향상시키도록 설계되었습니다. Matryoshika는 정책을 기반으로 출력의 피드백을 통해 최적화를 수행하며, 단계적 상호작용을 통해 지속적인 자기 개선(self-improvement)을 가능하게 합니다.

- **Performance Highlights**: 세 가지 복잡한 작업에 대한 실험 결과, Matryoshika는 추론에서 평균 3.19%, 계획에서 성공률 7.46%, 개인화에서 정확도 5.82%의 향상을 보여주었습니다. 이러한 결과는 Matryoshika가 블랙박스 LLM의 성능을 효과적으로 개선하는 데 기여할 수 있음을 시사합니다.



### Shopping MMLU: A Massive Multi-Task Online Shopping Benchmark for Large Language Models (https://arxiv.org/abs/2410.20745)
Comments:
          NeurIPS 2024 Datasets and Benchmarks Track Accepted

- **What's New**: Shopping MMLU라는 다채로운 다중 작업 온라인 쇼핑 벤치마크가 제안되었으며, 이는 실제 Amazon 데이터를 기반으로 한 57개의 작업으로 구성되어 있습니다. 이 벤치마크는 LLMs의 일반 쇼핑 보조원으로서의 능력을 종합적으로 평가할 수 있습니다.

- **Technical Details**: Shopping MMLU는 개념 이해, 지식 추론, 사용자 행동 정렬 및 다국어 능력 등 4가지 주요 쇼핑 기술을 포함되어 있으며, LLM 기반 솔루션을 맞춤형으로 평가하기 위해 모든 작업을 텍스트-텍스트 생성으로 재구성하였습니다. 또한 20,799개의 질문이 포함되어 있습니다.

- **Performance Highlights**: 20개 이상의 기존 LLMs를 사용하여 Shopping MMLU에서 성능을 벤치마킹했으며, 이를 통해 도메인 특화 LLMs에 대한 통찰력을 발견하였습니다. 또한, KDD Cup 2024에서 500개 이상의 참여 팀이 있는 대회를 개최하여 이러한 연구 결과를 더 널리 확산할 계획입니다.



### Faster WIND: Accelerating Iterative Best-of-$N$ Distillation for LLM Alignmen (https://arxiv.org/abs/2410.20727)
- **What's New**: 이 논문은 Iterative Best-of-N (BoN) 알고리즘의 이론적 특성과 실제 응용을 다룬다. 특히, Iterative BoN과 Self-Play Alignment 간의 통합된 게임 이론적 연결을 발견하여, WIN rate Dominance (WIND)라는 새로운 프레임워크를 제시한다.

- **Technical Details**: WIND 프레임워크는 보상 모델링을 기반으로 하여, 2인자 Min-Max 게임의 Nash equilibrium을 최적화한다. 이 방법은 기존의 Iterative BOND 방법의 한계점과 관련하여 효율적인 알고리즘을 제시하며, 정규화 손실 최적화의 일환으로 provable sample efficiency를 보장한다.

- **Performance Highlights**: WIND 알고리즘은 다양한 벤치마크에서 기존의 최첨단 Alignment 방법인 J-BOND와 비교하여 경쟁력 있는 성능을 보여주었으며, 특히 샘플링 프로세스와 훈련 비용 측면에서 효율성을 강조한다.



### Contextual Representation Anchor Network to Alleviate Selection Bias in Few-Shot Drug Discovery (https://arxiv.org/abs/2410.20711)
Comments:
          13 pages, 7 figures

- **What's New**: 본 연구는 few-shot 학습의 문제를 해결하기 위해 Contextual Representation Anchor Network (CRA)라는 새로운 방법론을 제안합니다. CRA는 labeled 데이터와 unlabeled 데이터 사이의 다리 역할을 하는 class-level contextual representation anchors를 활용하여 sample selection bias 문제를 해결하려고 합니다.

- **Technical Details**: CRA는 두 가지 증강 메커니즘을 도입합니다. 첫째는 context augmentation으로, 유사한 unlabeled 분자의 동적 검색과 task-specific contextual knowledge를 캡처하여 anchors를 증강합니다. 둘째는 anchor augmentation으로, anchors를 사용하여 분자 표현을 증강합니다. 이러한 접근법은 attention 메커니즘을 사용하여 unlabeled 데이터에서 정보의 정확한 전달을 보장합니다.

- **Performance Highlights**: CRA는 MoleculeNet 및 FS-Mol 벤치마크에서 평균 2.60% (AUC) 및 3.28% (ΔAUC-PR) 성능 향상을 보여주며, 뛰어난 일반화 능력을 갖추고 있습니다.



### Reprogramming Pretrained Target-Specific Diffusion Models for Dual-Target Drug Design (https://arxiv.org/abs/2410.20688)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 본 연구에서는 신규의 이중 표적 약물 설계를 위한 데이터셋을 제안하고, 단일 표적 단백질-리간드 복합체 쌍에 대해 훈련된 확산 모델(diffusion models)을 활용하여 이중 표적 약물 설계를 위한 생성적 작업으로 구성하였습니다. 또한, 두 개의 복합 그래프를 구축하여 SE(3)-equivariant 메시지 전파(messaging passing)를 구현하였습니다.

- **Technical Details**: 이 연구는 단일 표적에서 이중 표적으로 지식 전이를 효과적으로 수행하는 방법을 소개합니다. CompDiff와 DualDiff라는 두 가지 방법을 제안하며, 이는 압축된 메시지를 통해 이중 표적 약물 작용을 생성하는 새로운 접근 방식을 사용한다는 점에서 독특합니다. 또한, 약물 시너지(drug synergy)를 기반으로 한 이중 표적 설계를 위한 데이터셋을 구축했습니다.

- **Performance Highlights**: 광범위한 실험 결과를 통해 제안한 방법이 다양한 기준선과 비교했을 때 효과적임을 보여줍니다. 이 접근 방식은 이중 표적 약물 설계에 필요한 방대한 훈련 데이터를 크게 줄여줍니다.



### Segmenting Watermarked Texts From Language Models (https://arxiv.org/abs/2410.20670)
Comments:
          25 pages, 12 figures, 2 tables, NeurIPS 2024

- **What's New**: 본 논문은 신뢰할 수 없는 제3자가 신뢰할 수 있는 언어 모델 (LLM) 제공자에게 프롬프트를 보내고, LLM이 워터마크가 포함된 텍스트를 생성하는 시나리오에 초점을 맞추고 있습니다. 이를 통해 나중에 사용자가 텍스트를 배포하더라도 텍스트의 출처를 식별할 수 있습니다.

- **Technical Details**: 논문에서는 사용할 수 있는 통계적 방법론을 개발하여 게시된 텍스트가 LLM에 의해 생성되었는지 검출하는 방법을 제안합니다. 특히, 텍스트를 미리 결정된 길이의 이동 서브 문자열로 나누고, 각 서브 문자열이 워터마크가 있는지 확률적 p-value를 기반으로 검사하는 방법을 설명합니다. 본 방법은 랜덤화 테스트(randomization tests)와 변화점 탐지(change point detection) 기법을 기반으로 합니다.

- **Performance Highlights**: 다양한 언어 모델에서 생성된 텍스트를 사용하여 본 기법을 검증하였으며, Google의 C4 데이터셋에서 추출된 프롬프트를 통해 긍정적인 수치적 결과를 도출하였습니다. 제안된 방법은 Type I 및 Type II 오류(control) 제어를 확실히 하고, 변화점 위치를 찾아내어 워터마크가 있는 서브 문자열을 정확하게 식별합니다.



### TurboHopp: Accelerated Molecule Scaffold Hopping with Consistency Models (https://arxiv.org/abs/2410.20660)
Comments:
          22 pages, 11 figures, 8 tables. Presented at NeurIPS 2024

- **What's New**: TurboHopp라는 새로운 3D scaffold hopping 모델을 소개하며, 기존의 scaffolding 방식의 효율성을 결합하고 속도를 혁신적으로 향상시켰습니다. TurboHopp는 전통적인 ddpm(denoising diffusion probabilistic models) 기반 모델에 비해 최대 30배 빠른 생성 속도를 자랑합니다.

- **Technical Details**: TurboHopp는 pocket-conditioned 3D scaffold hopping 접근 방식을 통해, 전통적인 scaffold hopping의 전략적인 강점을 유지하면서도, consistency 모델의 빠른 생성 능력을 융합합니다. 이를 통해, 강화 학습(Reinforcement Learning) 기법을 적용하여 분자의 특성을 더욱 최적화합니다. 해당 모델은 3D 구조 기반 약물 디자인(diffusion models) 분야에서 효율성을 증대시키며, 많은 약물 발견 상황에서 적용 가능성을 보여줍니다.

- **Performance Highlights**: TurboHopp는 더욱 빠른 추론 속도와 향상된 생성 품질을 자랑하며, 기존의 확산 기반 모델들과 비교하여 제약물 성질, 결합 친화도(binding affinity), 합성 가능성(synthesizability) 등에서 우수한 성능을 보입니다. 또한, 이 모델은 의약품 발견을 위한 강력한 도구로서 자리 잡을 것으로 기대됩니다.



### Video to Video Generative Adversarial Network for Few-shot Learning Based on Policy Gradien (https://arxiv.org/abs/2410.20657)
Comments:
          18 pages, 11 figures, submitting to IEEE TNNLS

- **What's New**: 이 연구는 강화학습(Deep Reinforcement Learning)과 생성적 적대 신경망(Generative Adversarial Networks, GANs)을 활용하여 새로운 비디오-비디오 합성을 위한 방법론인 RL-V2V-GAN을 제안합니다. 이 모델은 비지도 조건부 비디오-비디오 합성(non-supervised conditional video-to-video synthesis)을 가능하게 합니다.

- **Technical Details**: RL-V2V-GAN은 정책 기울기(policy gradient) 학습을 통해 훈련되며, 공간 및 시간 정보를 캡처하기 위해 ConvLSTM 레이어를 사용합니다. 이 모델은 하나의 출처 비디오 도메인에서 목표 비디오 도메인으로의 매핑을 학습하며, 상대적 스타일을 유지합니다. 그 과정에서 spatio-temporal adversarial 목표를 통합하여 생성적 적대 손실(adversarial losses)을 이용해 콘텐츠 전송과 스타일 보존을 동시에 이룹니다.

- **Performance Highlights**: RL-V2V-GAN은 기존의 쌍 입력을 요구하는 방법들과는 달리 비지도 방식으로 작동하여, 몇 가지 샘플의 목표 도메인을 다룰 때 특히 효과적입니다. 실험 결과는 이 모델이 시간이 일관된 비디오 결과를 생성할 수 있음을 보여줍니다.



### NeuZip: Memory-Efficient Training and Inference with Dynamic Compression of Neural Networks (https://arxiv.org/abs/2410.20650)
- **What's New**: 이 논문에서는 NeuZip이라는 새로운 가중치 압축 방식을 제안합니다. 이 방법은 신경망의 부동 소수점 숫자의 엔트로피를 기반으로 하여, 성능 저하 없이 메모리 효율적인 학습과 추론을 가능하게 합니다.

- **Technical Details**: NeuZip은 세 가지 구성 요소(부호 비트, 지수 비트, 가수 비트)로 각 부동 소수점 숫자를 표현하며, 지수 비트의 저엔트로피 특성을 활용하여 비손실 방식으로 압축합니다. 이를 통해 메모리 절약을 달성하며, 손실을 허용하는 NeuZip 변형을 통해 추론 시 추가적인 메모리 절약을 보장합니다.

- **Performance Highlights**: Llama-3 8B 모델의 학습 시 메모리 사용량을 31GB에서 16GB 이하로 줄였으며, 추론 시 메모리 사용량을 절반 이상 줄이면서 거의 손실 없는 성능을 유지했습니다.



### Learning Variational Inequalities from Data: Fast Generalization Rates under Strong Monotonicity (https://arxiv.org/abs/2410.20649)
- **What's New**: 이 논문은 Variational Inequalities (VIs) 문제에 대해 새로운 통찰력을 제공하는데, 특히 strong monotonicity를 만족하는 VI에 대한 빠른 샘플 복잡도 O(1/ε) 속도를 어떻게 얻을 수 있는지 설명합니다.

- **Technical Details**: 논문에서는 VI의 연속 도메인에서 최적 솔루션 𝒛⋆를 찾기 위한 강한 단조성을 활용하여 O(1/ε) 샘플 복잡도를 도출합니다. 특히, 강한 볼록 최적화의 안정성 기반 일반화 증명이 VI로 직접 확장 가능하다는 점을 보여줍니다.

- **Performance Highlights**: 연구 결과에 따르면, 강한 단조성을 가진 VI에 대해 통계적 학습 속도가 향상되어, 전통적인 방법들과 비교하여 더 적은 샘플로도 근사 최적 솔루션을 찾을 수 있습니다.



### General Causal Imputation via Synthetic Interventions (https://arxiv.org/abs/2410.20647)
- **What's New**: 본 연구에서는 Squires et al. (2022)의 causal imputation 문제를 확장하여, generalized synthetic interventions (GSI)라는 새로운 추정기를 소개합니다. 기존의 추정기가 관계의 다양한 출력 차원을 동일한 선형 조합으로 처리하는 반면, GSI는 각 출력 차원의 차이를 고려하는 더 복잡한 모델로 개선됩니다.

- **Technical Details**: GSI 추정기는 래턴트 요인 모델(latent factor model)에서 생성된 데이터의 식별 가능성을 증명하며, 기존의 추정기(SI-A, SI-C)보다 더 일반화된 접근 방식을 제공합니다. 이는 두 개의 요소 집합(A, B)의 상호작용을 기반으로 하며, 각 출력 차원에 대해 고유한 선형 조합을 활용합니다.

- **Performance Highlights**: 모의 데이터 및 CMAP 데이터셋에서 실제로 GSI가 기존의 추정기보다 더 나은 성능을 발휘함을 보여주었습니다. 특히, GSI는 단일 차원에서 선형 조합 없이 각각의 차원에 대해 최적의 예측을 가능하게 합니다.



### Plastic Learning with Deep Fourier Features (https://arxiv.org/abs/2410.20634)
- **What's New**: 이 논문은 비정상적(non-stationary) 환경에서 지속적으로 학습하는 데 어려움을 겪는 심층 신경망의 플라스틱성(plasticity) 문제를 다룹니다. 특히, 심층 푸리에 특징(deep Fourier features)을 제안하며, 이는 모든 층에서 사인(sine)과 코사인(cosine)을 결합하여 선형성(linearity)과 비선형성(nonlinearity) 간의 동적인 균형을 이룹니다.

- **Technical Details**: 저자들은 심층 신경망의 플라스틱성 손실을 피하기 위해 심층 푸리에 특징을 사용하여 비선형 네트워크의 동역학을 심층 선형 네트워크의 행동으로 모방하는 방법을 탐구합니다. 또한, 이 구조는 레귤러리제이션(regularization)과 병합될 경우 뛰어난 일반화 성능을 보여줍니다.

- **Performance Highlights**: empirical results는 CIFAR10, CIFAR100 및 tiny-ImageNet 데이터 셋에서 ReLU 활성화(activations)를 심층 푸리에 특징으로 대체했을 때 지속적인 학습 성능이 크게 개선되었다고 보고합니다. 이 그림은 다양한 지속적 학습 시나리오에서 성능 향상이 가능함을 보여줍니다.



### TabDiff: a Multi-Modal Diffusion Model for Tabular Data Generation (https://arxiv.org/abs/2410.20626)
- **What's New**: 이번 연구에서는 TabDiff라는 새로운 다중 모달(diffusion) 프레임워크를 소개합니다. 이 프레임워크는 숫자형(numerical) 및 범주형(categorical) 데이터를 포함한 표 형식(tabular) 데이터의 모든 멀티 모달 분포를 하나의 모델에서 처리할 수 있도록 설계되었습니다.

- **Technical Details**: TabDiff는 다양한 입력 유형을 처리할 수 있는 transformer에 의해 매개변수화되어 있으며, 특히 다중 모달 기능 맞춤형(differentiable) learnable diffusion processes를 통해 서로 다른 feature들의 분포 비대칭성을 다룹니다. 이 모델은 연속 시간의 한계를 가진 증거 하한(limited evidence lower bound)으로 훈련됩니다.

- **Performance Highlights**: 전국적으로 널리 사용되는 7개의 표 형식 합성 벤치마크에서 TabDiff는 기존 방법보다 22.5%까지 개선된 결과를 보여주었고, 8개의 평가 지표에서 일관되게 우수한 성능을 발휘하였습니다.



### LoRA Done RITE: Robust Invariant Transformation Equilibration for LoRA Optimization (https://arxiv.org/abs/2410.20625)
- **What's New**: 이 논문에서는 Low-Rank Adaptation (LoRA)의 최적화 문제를 해결하기 위해 LoRA-RITE라는 새로운 어댑티브 매트릭스 전처리 방법을 소개합니다. 기존의 LoRA 최적화 방법이 갖는 변환 불변성(transform invariance) 부족 문제를 해결하여 학습 효율을 개선합니다.

- **Technical Details**: LoRA는 훈련되지 않은 가중치(matrix) W에 저차원 행렬(저랭크 매트릭스) Z를 주입하는 방식으로 작동합니다. 저랭크 행렬 Z는 두 개의 행렬 A와 B의 곱으로 나타낼 수 있으며, LoRA-RITE는 이러한 것에 대해 변환 불변성을 제공하도록 설계되었습니다. 기존의 최적화 알고리즘들은 일반적으로 이 속성을 증명하지 못하고, 이는 학습 과정에서 비효율성을 초래합니다.

- **Performance Highlights**: LoRA-RITE는 Gemma 2B를 사용해 Super-Natural Instructions에서 4.6%의 정확도 향상을, 총 4개의 LLM 벤치마크에서는 3.5%의 성능 향상을 달성했습니다. GSM8K 데이터셋에서는 Gemma 7B IT 모델을 사용하여 55.5%의 정확도를 기록하였으며, 이는 Adam 최적화 알고리즘의 48.37%보다 월등한 수치입니다.



### Practical Bayesian Algorithm Execution via Posterior Sampling (https://arxiv.org/abs/2410.20596)
Comments:
          Published as a conference paper at the 38th Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 이 논문에서는 비싼 함수의 평가 포인트를 효율적으로 선택하는 Bayesian algorithm execution (BAX) 프레임워크를 소개합니다. 기존의 BAX 방법들은 expected information gain (EIG)을 사용하여 평가 포인트를 선택하였으나, 이는 계산적으로 매우 집약적입니다. 기존의 한계를 극복하기 위해 PS-BAX라는 새로운 접근 방법을 제안하며, 이는 posterior sampling에 기반하여 더 빠르고 간단하게 평가 포인트를 선택합니다.

- **Technical Details**: PS-BAX는 포스터리어 샘플링을 기반으로 하여 BAX 문제의 넓은 범위에 적용할 수 있습니다. 이 방법은 각 반복에서 단 한번의 base algorithm 실행만을 필요로 하며, EIG 기반 방법들보다 빠르게 작동합니다. PS-BAX의 유효성을 위해 매우 무게가 적은 정규 조건 하에서 비대칭 수렴성을 증명하였습니다. 또한, 이 방법은 최적화 문제 및 레벨 집합 추정 등 다양한 문제에 적용될 수 있습니다.

- **Performance Highlights**: 실험 결과 PS-BAX는 기존의 baseline들과 비교했을 때 경쟁력 있는 성능을 보였으며, 구현이 간단하고 병렬화가 용이하다는 장점이 있습니다. 지금까지의 연구에 대한 강력한 기준을 세울 수 있으며, PS-BAX는 기존의 EIG 기반 접근 방식인 INFO-BAX보다 속도에서 수량적으로 우위를 점하고 있습니다.



### Generator Matching: Generative modeling with arbitrary Markov processes (https://arxiv.org/abs/2410.20587)
- **What's New**: 이번 논문에서는 Markov 프로세스를 활용한 generative modeling을 위한 새로운 프레임워크인 generator matching을 소개합니다. 이는 다양한 generative modeling 방법들을 통합하는 방법론입니다.

- **Technical Details**: generator matching은 Markov 프로세스의 무한소 진화를 설명하는 generators를 사용하여 데이터를 생성합니다. 조건부 생성기를 구축하여 단일 데이터 포인트를 생성한 다음, 전체 데이터 분포를 생성하는 marginal generator를 근사하도록 학습합니다. 여기에는 diffusion models, flow matching 및 discrete diffusion models이 포함되어 있습니다.

- **Performance Highlights**: 우리의 방법은 단백질 및 이미지 구조 생성을 위한 실험에서 검증되었으며, jump process와의 superposition이 이미지 생성 개선에 기여한다는 것을 보여주었습니다.



### Toward Conditional Distribution Calibration in Survival Prediction (https://arxiv.org/abs/2410.20579)
Comments:
          Accepted to NeurIPS 2024. 41 pages, 23 figures

- **What's New**: 이 논문에서는 생존 예측 모델의 조건부 보정(conditional calibration)의 중요성을 강조하고, 이를 해결하기 위한 새로운 방법론인 CSD-iPOT 프레임워크를 제안합니다. 이 방법은 모델의 개별 생존 확률을 기준으로 하여 생존 분포를 생성합니다.

- **Technical Details**: CSD-iPOT는 conformal prediction(적합한 예측) 기법에 기반하여 구성됩니다. 이 프레임워크는 생존 분석의 조건부 분포 보정을 지원하며, 이전의 방법론들보다 향상된 성능을 나타냅니다. 이 방법은 not only marginal calibration (주변 보정) 만을 고려하는 것이 아니라, 조건부 특성에 대한 보정을 가능하게 합니다.

- **Performance Highlights**: CSD-iPOT는 15개의 다양한 실제 데이터셋을 통해 실험하였으며, 주변 및 조건부 보정을 동시에 개선하는 데 성공했습니다. 또한 계산적으로 효율적인 특성을 가지고 있어, 기존의 방법들보다 더 나은 성능을 보이는 것을 입증했습니다.



### Deep Reinforcement Learning Agents for Strategic Production Policies in Microeconomic Market Simulations (https://arxiv.org/abs/2410.20550)
- **What's New**: 이 논문은 전통적인 경제 모델의 한계를 극복하기 위해 심층 강화 학습(Deep Reinforcement Learning, DRL)의 활용을 탐구합니다. 이를 통해 복잡하고 변동성이 큰 미시 경제 시장에서 최적의 생산 전략을 도출하는 방법을 제시하고 있습니다.

- **Technical Details**: 논문에서는 DRL 기반 접근법을 제안하여 경쟁 시장에서 여러 생산자가 수요, 공급, 가격 및 기타 노이즈로 인해 최적의 생산 결정을 내리는 정책을 학습할 수 있게 합니다. 여기서는 Markov Decision Process (MDP)를 사용하여 에이전트의 상호작용을 모델링하며, 에이전트는 시뮬레이터 내에서 다양한 상태에서 정책을 조정해 나갑니다.

- **Performance Highlights**: 실험 결과, DRL로 훈련된 에이전트는 변동성이 큰 시장 조건에서도 생산 수준을 전략적으로 조정하여 장기적인 수익성을 극대화할 수 있음을 보여주었습니다. 이 연구는 이론적 경제 모델과 실제 시장 시뮬레이션 간의 간극을 해소할 수 있는 가능성을 제시합니다.



### PaPaGei: Open Foundation Models for Optical Physiological Signals (https://arxiv.org/abs/2410.20542)
Comments:
          Code and models: this https URL

- **What's New**: 이 논문에서는 PPG(Photoplethysmography) 신호를 위한 최초의 오픈 파운데이션 모델인 PaPaGei를 소개합니다. 기존의 PPG 신호 모델은 일반화(generalization) 능력이 부족하며, 주로 특정 작업에 제한되어 있었습니다. PaPaGei는 57,000시간 이상의 PPG 데이터를 사용하여 사전 훈련(pre-trained)되었고, 다양한 건강 관련 작업에서의 성능을 평가하였습니다.

- **Technical Details**: PaPaGei는 2000만 개의 비표시 PPG 신호 세그먼트로 구성된 데이터에서 사전 훈련되었으며, 노이즈와 움직임 아티팩트(motion artifacts)에 대한 저항성도 연구하였습니다. 파라미터 효율성 및 데이터 효율성을 강조하며, PPG 신호의 형태적 차이를 활용한 새로운 표현 학습(Representation Learning) 방식을 도입했습니다.

- **Performance Highlights**: PaPaGei는 20개의 다양한 작업에 걸쳐 평균 6.3%의 분류(classification) 성능 향상과 2.9%의 회귀(regression) 성능 향상을 기록하였으며, 70배 더 큰 모델보다 더 높은 성능을 보여주었습니다. 또한 다양한 피부 톤에 대한 강건성(robustness)을 평가하여 미래 모델의 편향(bias) 평가를 위한 기준을 마련했습니다.



### Info-CELS: Informative Saliency Map Guided Counterfactual Explanation (https://arxiv.org/abs/2410.20539)
- **What's New**: 이번 논문에서는 CELS 모델을 기반으로 한 향상된 접근 방식을 제안합니다. CELS는 시계열 분류기의 결정에 대한 직관적인 설명을 제공하고 사후(counterfactual) 설명을 탐색하기 위한 최초의 시도가 이루어진 모델로, 이번 연구에서는 마스크 정규화를 제거하여 더 정보적이고 유효한 사후 설명을 제공합니다.

- **Technical Details**: CELS(Counterfactual Explanation with Learned Saliency maps)는 시계열 데이터에서 중요한 시간 단계를 식별하고 이를 바탕으로 사후 설명을 생성하는 모델입니다. 이 모델은 원래의 CELS 모델보다 높은 유효성(validity)과 정보성을 가지는 설명을 생성하기 위해 고안되었습니다. 각 입력 시계열 데이터는 최소한의 변형을 통해 원하는 레이블(z′)로의 예측 변화를 목표로 합니다.

- **Performance Highlights**: 다양한 도메인에서의 광범위한 실험을 통해 제안된 방법이 CELS 모델보다 해석 가능성과 유효성에서 우수한 성능을 보임을 입증하였습니다.



### Guiding Through Complexity: What Makes Good Supervision for Hard Reasoning Tasks? (https://arxiv.org/abs/2410.20533)
- **What's New**: 본 논문에서는 약한 교사 모델(weak teacher models)인 평균적인 인간 주석자나 기존 AI 시스템들이 LLM(대규모 언어 모델)의 성능을 향상시키기 위해 어떻게 효과적으로 감독할 수 있는지를 탐구합니다. 특히, 이러한 모델들이 난이도가 높은 추론 작업에서 어떻게 더 나은 성과를 낼 수 있는지에 대한 데이터 기반 전략을 제시합니다.

- **Technical Details**: 저자들은 두 가지 감독 전략을 제안합니다. 첫 번째 전략은 목표 추론 작업의 난이도와 일치하는 전체 작업에서 낮은 품질의 감독을 사용하는 것이고, 두 번째 전략은 더 쉬운 하위 작업(subtask)에서 높은 품질의 감독을 활용하는 것입니다. 연구 결과, 고차원 과업에 대한 감독이 높은 오류율(예: 90%)을 가지고 있음에도 불구하고, 보다 쉬운 하위 작업의 완벽한 감독보다 더 나은 성과를 보여줍니다. 중요한 배경 요인으로는 단계별 오류율(step-wise error rates)이 지적됩니다.

- **Performance Highlights**: 고차원 작업 감독이 하위 작업 감독보다 항상 우수한 성과를 제공하는 것으로 나타났습니다. 다양한 조합으로 하위 작업 감독을 보충했을 때, MATH 및 SAT-Math 벤치마크에서 최고의 성과를 달성했습니다. 구체적으로, A=50%와 B=10% 일 때 최고의 결과를 보였으며, 이는 단순히 하드 작업 감독을 두 배로 늘리거나 하드 작업 감독을 조합한 경우보다 더 효과적이라는 결과를 얻었습니다.



### Llama Scope: Extracting Millions of Features from Llama-3.1-8B with Sparse Autoencoders (https://arxiv.org/abs/2410.20526)
Comments:
          22pages, 12 figures

- **What's New**: 이번 연구에서는 Llama-3.1-8B 모델의 모든 층과 서브레이어에 대해 256개의 Sparse Autoencoders (SAEs)를 훈련시키고, 스케일 가능한 훈련과 해석 도구를 제공하여 미세 조정 모델에 대한 SAEs의 일반화 가능성을 평가합니다.

- **Technical Details**: Llama Scope는 32K 및 128K 특성을 갖춘 256개의 SAE로 구성되며, Top-K SAEs 변형을 적용하여 다양한 차원에서 평가합니다. 이 SAEs는 메모리 병목 현상을 줄이기 위한 혼합 병렬 처리 접근 방식을 사용하여 훈련됩니다.

- **Performance Highlights**: Llama Scope의 SAE 체크포인트는 공개되어 있으며, 다양한 평가 지표를 사용하여 기능 해석 가능성과 잠재적 전압 주파수를 포함한 성능을 Assess 합니다. 이 연구는 오픈 소스 SAE 생태계를 확장하고 기계적 해석 가능성 연구를 지원하기 위한 기여를 목표로 합니다.



### A Cosmic-Scale Benchmark for Symmetry-Preserving Data Processing (https://arxiv.org/abs/2410.20516)
Comments:
          19 pages, 3 figures; To appear at the NeurReps Workshop @ NeurIPS 2024

- **What's New**: 이 연구에서는 구조화된 포인트 클라우드 데이터를 효율적으로 처리하는 데 중점을 두고, 그래프 신경망(Graph Neural Networks, GNNs)이 지역 클러스터 환경과 장거리 상관관계를 동시에 포착하는 능력을 벤치마크했습니다. 특히, 유클리드 대칭을 유지하는 GNN의 성능을 평가하여, 비대칭 모델 및 도메인 특화 정보 추출 기술에 비해 우수한 성능을 입증했습니다.

- **Technical Details**: 우리는 입자의 위치 및 속성과 같은 속성으로 정의된 포인트 클라우드 데이터를 사용하여 GNN 아키텍처의 효과를 시스템적으로 연구하였으며, 유클리드 대칭을 고려한 다양한 아키텍처 선택이 하위 작업 성능에 미치는 영향을 분석했습니다. 본 연구에서는 그래프 신경망을 통해 다중 스케일 구조를 효과적으로 포착하는 방법을 제안합니다.

- **Performance Highlights**: 연구 결과, 유클리드 대칭을 유지하는 GNN이 비대칭 모델 및 특정 도메인의 전통적인 통계적 기법에 비해 하위 작업에서의 성능이 우수함을 보여주었습니다. 그러나, 현재의 아키텍처는 장거리 상관관계를 효과적으로 캡처하지 못하고 있어, 이는 향후 더 나은 아키텍처 개발의 필요성을 제기합니다.



### Efficient Diversity-based Experience Replay for Deep Reinforcement Learning (https://arxiv.org/abs/2410.20487)
- **What's New**: 본 논문은 스파스 리워드 환경에서의 학습 효율성을 크게 향상시키기 위해 다양성 기반 경험 재플레이(Diversity-Based Experience Replay, DBER)라 불리는 새로운 접근법을 제안합니다.

- **Technical Details**: DBER는 결정론적 포인트 프로세스(Determinantal Point Processes, DPPs)를 활용하여 상태 실현에서 다양한 샘플을 우선시합니다. 이를 통해 샘플 활용 효율성을 높이고, TD-오류(Temporal Difference Error)에 의존하지 않으면서 샘플의 대표성과 효과성을 보장합니다.

- **Performance Highlights**: 다양한 시뮬레이션 환경에서 실험한 결과, DBER는 스파스 리워드 환경에서도 학습 효율성을 크게 개선하고 뛰어난 성능을 보여주었습니다. 이는 고차원 상태 공간에서의 강화 학습 문제를 해결하기 위한 간단하면서도 효과적인 솔루션을 제공합니다.



### Improving Decision Sparsity (https://arxiv.org/abs/2410.20483)
Comments:
          Accepted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 이 논문은 Sparse Explanation Value(SEV)의 개념을 확장하여 머신러닝 모델의 의사결정에서의 해석 가능성을 높이는 새로운 방법론을 제안합니다. SEV는 특정 의사결정에서 핵심적인 정보의 양을 반영하는 로컬 스파시티(local sparsity)를 측정하는 데 중점을 둡니다.

- **Technical Details**: SEV는 하이퍼큐브(hypercube) 상에서의 이동을 기반으로 하며, 참조 포인트의 위치 조정과 기능 공간(feature space)에서의 거리 변환을 고려하여 해석의 신뢰성(credibility)과 의사결정 스파시티를 최적화합니다. 논문에서는 클러스터 기반 SEV와 트리 기반 SEV를 제시하고, 신뢰성을 개선하는 방법론과 함께 머신러닝 모델의 의사결정 스파시티를 최적화하는 알고리즘을 제안합니다.

- **Performance Highlights**: 제안된 알고리즘은 의사결정 성능을 저하시키지 않으면서 고스파시티 모델을 효율적으로 생성할 수 있으며, SEV 계산을 위한 새로운 세 가지 접근 방식을 도입하여 더 의미 있는 해석 결과를 도출할 수 있음을 보여줍니다.



### Hamiltonian Score Matching and Generative Flows (https://arxiv.org/abs/2410.20470)
- **What's New**: 본 연구에서는 Hamiltonian ODEs를 위한 강제장(force field)을 의도적으로 설계하는 가능성을 탐구합니다. Hamiltonian Velocity Predictors (HVPs)라는 도구를 도입하여 스코어 매칭(score matching) 및 생성 모델(generative model)에 응용할 수 있습니다. 두 가지 주요 혁신을 제안하며, Hamiltonian Score Matching (HSM)과 Hamiltonian Generative Flows (HGF)를 통해 새로운 생성 모델을 제시합니다.

- **Technical Details**: HVP를 활용하여 데이터를 Hamiltonian 경로를 통해 확장하여 스코어 함수를 추정하는 Hamiltonian Score Matching (HSM)과, 제로 강제장을 포함하는 새로운 생성 모델인 Hamiltonian Generative Flows를 소개합니다. 또한, 조화 발진기(harmonic oscillator)에 영감을 받은 Oscillation HGF를 포함하여 강제장 설계를 확장할 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: 실험 결과 HSM은 스코어 매칭 메트릭으로서의 혁신을 입증하며, HGF가 기존의 선도적인 생성 모델 기법들과 비교해도 경쟁력 있는 성능을 발휘하는 것을 보여줍니다.



### Graph Neural Networks on Discriminative Graphs of Words (https://arxiv.org/abs/2410.20469)
- **What's New**: 이번 연구에서는 Discriminative Graph of Words Graph Neural Network (DGoW-GNN)이라는 새로운 접근 방식을 소개합니다. 이 방법은 단어 노드만 포함된 그래프를 구축하고, 훈련 데이터셋을 레이블에 따라 분리된 서브그래프로 나누며, 단어의 pointwise mutual information을 통해 엣지를 가중치화합니다.

- **Technical Details**: DGoW-GNN은 단어 노드 간의 연결을 pointwise mutual information에 기반하여 구축하며, 각 클래스에 대한 disconnected subgraph로 훈련 데이터셋을 분리합니다. 이를 통해 텍스트 분류 문제를 walk classification 문제로 재정의할 수 있습니다. 또한, GNN과 sequence model을 결합하여 문서의 구조적 정보와 단어의 순서를 동시에 고려합니다.

- **Performance Highlights**: 실험을 통해 7개의 벤치마크 데이터셋에서 DGoW-GNN을 평가한 결과, 여러 최신 모델에 비해 성능이 다소 낮은 것으로 나타났습니다. 이 성능 차이에 대한 분석과 변화 가능성을 모색하고 있습니다.



### Vector Quantization Prompting for Continual Learning (https://arxiv.org/abs/2410.20444)
Comments:
          To appear in NeurIPS 2024

- **What's New**: 최근 발표된 VQ-Prompt는 Vector Quantization(VQ) 기법을 사용하는 새로운 지속적 학습(Continual Learning) 방법으로, 모델이 여러 작업을 수행하면서 이전 정보를 잃지 않도록 돕습니다. 이 방법은 내장된 프롬프트(promopts)집합에서 효과적으로 작업 지식을 표현하고, 이를 통해 프리트레인(pre-trained) 모델의 성능을 향상시키는 점이 특징입니다.

- **Technical Details**: VQ-Prompt는 Discrete Prompt를 사용하여 지속적 학습을 최적화하는 메커니즘을 도입합니다. 입력 쿼리에서 생성된 continuous prompt를 미리 정의된 discrete prompt pool에서 가장 가까운 값을 찾는 방식으로 교체하며, gradient estimation을 통해 task loss를 continuous prompt로 전파합니다. 추가적인 VQ 정규화 항을 통해 prompt pool의 학습을 더욱 개선합니다.

- **Performance Highlights**: VQ-Prompt는 다양한 벤치마크에서 최신 지속적 학습 방법들보다 높은 성능을 발휘하며, 특히 challenging한 class-incremental 환경에서 탁월한 결과를 보였습니다. 검증된 실험 결과를 통해 VQ-Prompt는 지속적 학습의 성능 기준을 한 단계 끌어올렸음을 보여주었습니다.



### TEAFormers: TEnsor-Augmented Transformers for Multi-Dimensional Time Series Forecasting (https://arxiv.org/abs/2410.20439)
- **What's New**: TEAFormer를 통해 기존 Transformer 모델의 한계를 극복했습니다. 전통적인 모델이 다차원 구조를 효과적으로 보존하지 못하는 문제를 해결하고, 텐서 확장 및 압축을 기능으로 통합하여 예측 정확도를 향상시켰습니다.

- **Technical Details**: TEAFormer는 Tensor-Augmentation (TEA) 모듈을 도입하여, 다채널에서 다중 시각적 피처 학습을 통한 텐서의 확장을 활용하고, 자동 텐서 분해를 통해 정보를 압축하여 계산 부하를 줄입니다. 이 모듈은 Transformer의 인코더-디코더 구조 및 어텐션 메커니즘과 높은 호환성을 가집니다.

- **Performance Highlights**: TEAFormer는 Transformer, Informer, Autoformer와 같은 세 가지 인기있는 모델에 통합되어 34/42 시험에서 기초 모델 대비 성능을 유의미하게 개선하며, MAE(Mean Absolute Error) 및 MSE(Mean Squared Error) 평가 지표에서 뛰어난 성능을 보임을 입증했습니다.



### Integrating uncertainty quantification into randomized smoothing based robustness guarantees (https://arxiv.org/abs/2410.20432)
- **What's New**: 이 논문은 불확실성 점수 기반((uncertainty score-based)) 거부를 인증된 강건성((certified robustness)) 프레임워크에 통합하여 새로운 강건성 보장을 제공합니다. 이 접근법은 고전적인 기법들과 현대의 불확실성 기술을 융합하여, 안전-critical한 응용 프로그램에 더욱 효과적인 분류기를 만들어냅니다.

- **Technical Details**: 논문은 (i) 주어진 입력 주위의 천이 서브스페이스에서 동일한 레이블을 예측하면서 불확실성이 낮은 반지의 반경((radius of an ℓ2-ball))과 (ii) 예측이 변경되지 않거나 불확실할 반지의 ℓ2-반경을 돌출합니다. 이러한 새로운 보장들은 입력의 소음을 수용하며, 또한 강건한 구조와 불확실성 측정의 체계적인 평가를 가능하게 합니다.

- **Performance Highlights**: 새로 제안된 프레임워크는 CIFAR10에서 20.93% 더 큰 반경을 제공하여, 불확실성 기반 거부를 허용하지 않는 모델에 비해 더 강건성을 보여줍니다. 또한, 불확실성을 통합함으로써, 모델의 분포 외 탐지에서 더 나은 성능을 발휘함을 실험적으로 증명했습니다.



### Causal Modeling in Multi-Context Systems: Distinguishing Multiple Context-Specific Causal Graphs which Account for Observational Suppor (https://arxiv.org/abs/2410.20405)
- **What's New**: 이 논문에서는 여러 맥락(contexts)에서 수집된 데이터로부터 원인 구조 학습을 다룹니다. 특히 다양한 관찰 지지(observational support)의 차이가 인과 그래프의 식별 가능성에 미치는 영향을 분석합니다. 새로운 인과 그래프 객체를 도입하여, 맥락별 독립성(context-specific independence, CSI)을 구조적 인과 모델(structural causal models, SCMs) 내에서 정교하게 모델링합니다.

- **Technical Details**: 논문은 인과 메커니즘과 데이터 지원을 모두 포착하는 인과 그래프 객체를 사용하여 맥락별 변화(context-specific changes)를 분석합니다. 이 구조는 기후 데이터와 같은 다양한 사례에서 관찰된 현상을 설명하는 데 유용하며, 일반화(generalization)와 이전 학습(transfer learning)을 위한 이론적 기초를 제공합니다. 주요 기술 요소로는 구조적 인과 모델 내에서의 맥락별 독립성과 이를 기반으로 한 그래프 구조의 식별 가능성 분석이 포함됩니다.

- **Performance Highlights**: 이 프레임워크는 극단적인 사건이나 이상 현상의 설명을 돕고, 다양한 맥락에서 관찰된 인과 메커니즘의 변화 변화를 이해하는 데 기여합니다. 미래 연구는 이 접근 방식을 복잡한 데이터 유형, 예를 들어 시계열 데이터로 확장할 수 있는 가능성을 제시합니다.



### Deep Learning-Driven Microstructure Characterization and Vickers Hardness Prediction of Mg-Gd Alloys (https://arxiv.org/abs/2410.20402)
- **What's New**: 본 연구에서는 Mg-Gd 합금의 기계적 성능을 예측하기 위해 이미지 처리 및 딥러닝 기법을 기반으로 한 멀티모달 융합 학습 프레임워크를 제안합니다. 이 프레임워크는 합금의 원소 조성 및 미세구조 특성을 통합하여 Vickers 경도를 정확히 예측합니다.

- **Technical Details**: 연구는 다양한 Mg-Gd 합금 이미지에서 미세구조 정보를 추출하기 위해 딥러닝 기법을 사용하여 정밀한 결정립 크기 및 이차상 미세구조 특성을 제공합니다. 이후 Gd 함량 정보와 정량적 분석 결과를 조합하여 성능 예측 데이터셋을 구축하였습니다. 최종적으로 Transformer 아키텍처를 기반으로 한 회귀 모델을 사용하여 Mg-Gd 합금의 Vickers 경도를 예측하였습니다.

- **Performance Highlights**: 실험 결과, Transformer 모델이 예측 정확도 측면에서 최적의 성능을 보이며 R² 값 0.9을 기록하였습니다. 또한, SHAP 분석을 통해 Mg-Gd 합금의 Vickers 경도에 영향을 미치는 네 가지 주요 특징의 중요 값이 식별되어 합금 설계에 유용한 지침을 제공합니다.



### Prototypical Extreme Multi-label Classification with a Dynamic Margin Loss (https://arxiv.org/abs/2410.20401)
- **What's New**: 이번 논문에서는 PRIME이라는 새로운 Extreme Multi-label Classification (XMC) 방법을 제안합니다. PRIME은 새로운 prototypical contrastive learning 기술을 사용하여 효율성과 성능을 조화롭게 하여 기존의 brute-force 접근 방식을 초월합니다.

- **Technical Details**: PRIME은 라벨 프로토타입(label prototype)을 집계함으로써 쿼리 관련 정보를 통합하는 데이터 대 프로토타입(data-to-prototype) 예측 작업으로 XMC를 정의합니다. 이 방법은 Label Prototype Network라는 얕은 transformer encoder를 사용하여 text-based embeddings, 라벨 중심(label centroids), 학습 가능한 자유 벡터를 결합하여 라벨 표현을 풍부하게 만듭니다.

- **Performance Highlights**: PRIME은 여러 공개 벤치마크에서 최첨단 결과를 달성하며, 단일 GPU 예산 내에서 효율성을 유지하면서 기존의 brute-force 접근 방식보다 성능이 우수합니다. 특히, 모든 실험과 대부분의 메트릭에서 PRIME은 높은 성능 개선을 보였다.



### ThunderKittens: Simple, Fast, and Adorable AI Kernels (https://arxiv.org/abs/2410.20399)
- **What's New**: ThunderKittens (TK) 프레임워크는 GPU 아키텍처에 최적화된 AI 커널을 쉽게 작성할 수 있게 해주는 핵심 추상화 기능을 제공합니다. 이 프레임워크는 성능을 최대화하기 위해 3가지 레벨(워프, 블록, 그리드)에서 GPU의 기본 데이터 구조와 병렬 계산 작업을 지원합니다.

- **Technical Details**: TK는 다음의 세 가지 수준에서 GPU 아키텍처와 매핑됩니다: (1) 워프 수준에서 16x16 매트릭스 타일을 기본 데이터 구조로 사용하고, PyTorch와 유사한 병렬 계산 작업을 제공, (2) 스레드 블록 수준에서 비동기 작업을 조정하기 위한 템플릿을 제공, (3) 그리드 수준에서 블록 실행 및 정리와 메모리 비용을 줄이는 지원을 제공합니다.

- **Performance Highlights**: ThunderKittens의 성능은 이전 커널과 동등하거나 이를 초과하는 결과를 보여주며, Attention 성능에서 10-40% 개선을 보였고, 선형 Attention에서 14배 성능 향상을 달성했습니다. TK는 실제 AI 작업에 적용되고 있으며, CUDA 경험이 없는 학생팀에 의해 작성되었습니다.



### Evaluation of uncertainty estimations for Gaussian process regression based machine learning interatomic potentials (https://arxiv.org/abs/2410.20398)
- **What's New**: 이번 연구에서는 머신러닝 원자 간 포텐셜(Machine Learning Interatomic Potentials, MLIPs)의 불확실성 추정이 모델의 추가적인 오차를 정량화하고, 능동학습(active learning) 전략에서 가장 정보량이 많은 훈련 샘플을 찾는데 필수적임을 강조하였습니다. Gaussian Process Regression (GPR)를 기반으로 한 모델과 앙상블 기반 불확실성 측정의 비교가 이루어졌습니다.

- **Technical Details**: 연구는 Coulomb 및 Smooth Overlap of Atomic Positions (SOAP) 표현을 갖는 GPR 모델을 사용하여 분자의 포텐셜 에너지 및 여기 에너지를 예측합니다. GPR의 분산(variance) 및 앙상블 기반 불확실성이 실제 오차와 어떤 관계가 있는지 평가하였으며, 고정된 구성 공간에서 가장 불확실한 샘플을 선택했을 때 모델 성능이 향상되는지를 조사하였습니다.

- **Performance Highlights**: 앙상블 기반 불확실성 추정치는 종종 실제 오차와 관련이 없음을 발견하였고, GPR의 표준 편차가 증가할수록 편향(bias)도 증가한다는 점을 확인하였습니다. 이러한 경우, 가장 높은 불확실성을 가진 훈련 샘플을 선택하는 것이 무작위 샘플링보다 나쁜 테스트 오차를 초래하는 경향이 있음을 보여주었습니다.



### Hierarchical Multiple Kernel K-Means Algorithm Based on Sparse Connectivity (https://arxiv.org/abs/2410.20391)
Comments:
          in Chinese language

- **What's New**: 이번 논문에서는 sparse connectivity (희소 연결)를 기반으로 한 새로운 алгоритм인 SCHMKKM (Hierarchical Multiple Kernel K-Means) 를 제안합니다. 이는 정보 상호작용을 고려하여 최적의 kernel function (커널 함수)을 찾아내고 클러스터 분석을 수행합니다.

- **Technical Details**: HMKC (Hierarchical Multiple Kernel Clustering) 알고리즘은 높은 차원의 공간에서 샘플의 특징을 계층적으로 추출합니다. 본 연구는 각 노드가 인접한 계층의 특정 노드와만 정보를 교환하는 기존 방식의 한계를 지적하고, sparse connections (희소 연결)을 통해 정보의 효과적인 융합을 실현하는 방법을 제시합니다.

- **Performance Highlights**: 여러 데이터셋에 대한 실험 결과 SCHMKKM 알고리즘이 FCHMKKM (Fully Connected Hierarchical Multiple Kernel K-Means) 알고리즘보다 더 높은 잠재적 성능을 보이며, 정보 융합의 다양성이 클러스터 partition matrix (파티션 매트릭스)의 일관성 향상에 기여함을 확인했습니다.



### Unsupervised Feature Selection Algorithm Based on Dual Manifold Re-ranking (https://arxiv.org/abs/2410.20388)
Comments:
          in Chinese language

- **What's New**: 이 논문에서는 unsupervised (비지도) 학습 환경에서의 효율적인 feature selection (특징 선택) 기법으로 dual manifold re-ranking (DMRR)을 제안합니다.

- **Technical Details**: DMRR은 샘플 간, 샘플과 특징 간, 그리고 특징 간의 manifold 구조를 표현하기 위해 여러 개의 similarity matrix (유사도 행렬)를 구성합니다. 그런 다음 초기 샘플과 특징의 점수를 결합하여 manifold re-ranking을 수행합니다.

- **Performance Highlights**: 실험 결과, DMRR은 세 가지 기존의 unsupervised feature selection 알고리즘 및 두 가지 unsupervised feature selection 후처리 알고리즘과 비교하여 다양한 샘플의 중요도 정보를 효과적으로 이용하고, 샘플과 특징 간의 이중 관계를 통해 더 나은 특징 선택 결과를 도출함을 확인했습니다.



### Multiple kernel concept factorization algorithm based on global fusion (https://arxiv.org/abs/2410.20383)
Comments:
          in Chinese language

- **What's New**: 이번 논문에서는 Non-negative Matrix Factorization(NMF) 알고리즘의 한계를 극복하기 위해 Concept Factorization(CF) 알고리즘을 제안합니다. 이는 행렬 분해를 비선형 커널 공간(single non-linear kernel space)으로 확장하여 학습 능력과 적응력을 향상시킵니다.

- **Technical Details**: 새로운 알고리즘인 Globalized Multiple Kernel CF(GMKCF)를 통해, 여러 후보 커널 함수(multiple candidate kernel functions)를 동시에 입력하고, 글로벌 선형 융합(global linear fusion) 기반 CF 프레임워크 내에서 학습합니다. 이 알고리즘은 클러스터링 결과의 품질과 안정성을 향상시키면서 CF가 직면한 커널 함수 선택 문제를 해결합니다.

- **Performance Highlights**: 여러 실제 데이터베이스에 대한 실험 결과, GMKCF 알고리즘은 Kernel K-Means(KKM), Spectral Clustering(SC), Kernel CF(KCF), Co-regularized multi-view spectral clustering(Coreg), Robust Multiple KKM(RMKKM) 같은 비교 알고리즘들보다 데이터 클러스터링에서 우수한 성능을 보였습니다.



### FuseFL: One-Shot Federated Learning through the Lens of Causality with Progressive Model Fusion (https://arxiv.org/abs/2410.20380)
- **What's New**: 이 논문에서는 One-shot Federated Learning (OFL) 방법의 성능 저하를 다루고 있으며, 검증된 원인(인과적 관점)을 바탕으로 문제를 해결하기 위한 새로운 접근 방식을 제안합니다. 이를 통해 데이터의 이질성(Data Heterogeneity) 문제를 해소하고, 기존의 OFL 방법보다 더 나은 성능을 발휘하는 FuseFL을 소개합니다.

- **Technical Details**: FuseFL은 신경망(Neural Networks)을 여러 블록으로 나누고, 각 블록을 바텀업(bottom-up) 방식으로 훈련한 후 융합(Fusion)하여 특징을 증강하는 방법론입니다. 이 업데이트 및 통합 과정은 추가적인 통신 비용을 발생시키지 않으면서도 모델 성능을 크게 향상시킵니다. 특히, 각 클라이언트가 소형 모델(Small Model)을 유지하면서도 최종 모델이 원래 모델과 동일한 크기를 갖도록 설계되었습니다.

- **Performance Highlights**: FuseFL은 기존의 OFL 및 ensemble FL(앙상블 연합학습)보다 유의미한 성능 향상을 보여주었으며, 이는 클라이언트의 고도 확장성(High Scalability), 이질적인 모델 훈련(Heterogeneous Model Training), 그리고 낮은 메모리 비용(Low Memory Costs)을 지원합니다.



### Rethinking Reconstruction-based Graph-Level Anomaly Detection: Limitations and a Simple Remedy (https://arxiv.org/abs/2410.20366)
Comments:
          Published as a conference paper at NeurIPS 2024

- **What's New**: 이번 연구에서는 Graph-AE (Graph Autoencoders)를 활용한 그래프 수준의 이상 탐지(GLAD)에서 발생하는 'reconstruction flip' 현상을 보고하고, 기존 방법의 한계를 강조합니다.

- **Technical Details**: MUSE(Multifacted Summarization of Reconstruction Errors)라는 새로운 GLAD 방법을 제안하며, 그래프의 reconstruction error의 다양한 요약치를 그래프 특성으로 사용합니다. MUSE는 기존의 방식보다 우수한 성능을 발휘하며, 10개의 데이터셋에서 14개 방법 중 가장 좋은 성능을 보여줍니다.

- **Performance Highlights**: MUSE는 GLAD 작업에서 SOTA (State Of The Art) 성능을 기록하며, 가장 강력한 경쟁자와 비교해 AUROC에서 최대 28.1%의 성능 향상을 달성했습니다.



### Uncovering Capabilities of Model Pruning in Graph Contrastive Learning (https://arxiv.org/abs/2410.20356)
Comments:
          MM' 24

- **What's New**: 본 연구에서는 기존의 그래프 대조 학습에서의 데이터를 증강하는 접근법 대신, 다양한 모델 버전 간의 대조를 통해 그래프 대조 학습 문제를 재정립합니다. 이를 위해, 모델 프루닝(model pruning) 방법을 활용하여 모델의 대표성을 높이고 그래프의 의미 정보를 손상시키지 않으면서 새로운 그래프 표현을 생성합니다.

- **Technical Details**: LAMP( Graph Contrastive Learning via Model Pruning) 프레임워크는 원본 그래프를 모델 입력으로 사용하여 그래프 왜곡으로 인한 의미 정보의 변화를 방지합니다. 모델의 핵심 정보를 식별하는 능력을 함양하면서, 프루닝을 통해 동적으로 변형된 그래프 인코더를 생성하고 이를 원본 인코더와 대조합니다. 또한, 노드 임베딩의 무결성을 고려하여 로컬 대조 손실(local contrastive loss)을 개발하여 모델 훈련을 방해하는 어려운 부정 샘플을 처리할 수 있습니다.

- **Performance Highlights**: LAMP는 비지도 학습 및 전이 학습을 통한 그래프 분류 실험에서 SOTA(comparative) 경쟁자를 능가하는 성능을 보였으며, 기존 데이터 증강 방법 대비 모델 프루닝의 우수성을 이론적으로 분석하여 혁신적인 기여를 하고 있습니다.



### Leveraging Auxiliary Task Relevance for Enhanced Industrial Fault Diagnosis through Curriculum Meta-learning (https://arxiv.org/abs/2410.20351)
- **What's New**: 본 논문은 Related Task Aware Curriculum Meta-learning (RT-ACM) 기반의 새로운 결함 진단 프레임워크를 제안하며, 이는 인간의 인지 학습 과정을 모방하여 데이터 부족 문제를 해결하는 데 중점을 둡니다.

- **Technical Details**: RT-ACM은 MAML(모델-불가지론 메타 학습)의 개념을 확장하여, 관련 작업(task)간의 관계를 고려하면서 ‘더 관련 있는 지식에 더 많은 주의를 기울이는' 방식으로 메타 학습을 수행합니다. 두 개의 주요 구성 요소는 관련 작업 인식 메타 학습과 커리큘럼 전략입니다.

- **Performance Highlights**: 두 개의 실제 데이터셋에 대한 광범위한 실험 결과, RT-ACM 프레임워크가 기존의 최첨단 기술들과 비교해 우수한 성능을 보이는 것으로 입증되었습니다.



### Intuitionistic Fuzzy Universum Twin Support Vector Machine for Imbalanced Data (https://arxiv.org/abs/2410.20335)
- **What's New**: 본 논문은 불균형 데이터셋을 다루기 위해 직관적 퍼지( intuitionistic fuzzy ) 개념을 활용한 Universum twin support vector machine ( UT-SVM )의 변형 모델인 IFUTSVM-ID를 제안합니다. 이 모델은 노이즈와 아웃라이어의 영향을 완화하고, 이전 정보를 통합하여 데이터 분포의 적절한 처리와 더 나은 일반화 성능을 제공합니다.

- **Technical Details**: IFUTSVM-ID는 데이터에 대한 직관적 퍼지 멤버십 스킴을 사용하여 각 데이터 포인트에 IF 점수를 할당하고, 이를 통해 아웃라이어와 노이즈 문제를 해결합니다. 또한, Universum 데이터를 결합하여 사전 지식을 이용함으로써 모델의 성능을 향상시키며, SRM(Structural Risk Minimization) 원칙을 통합하여 과적합 문제를 해결합니다.

- **Performance Highlights**: IFUTSVM-ID 모델은 KEEL 벤치마크 데이터 세트에서 기존 모델들과 비교했을 때, 알츠하이머 질환 진단을 위한 ADNI 데이터 세트에서도 그 우수성을 입증하였습니다. 실험 결과, 제안된 모델이 베이스라인 모델에 비해 성능이 향상되었다고 보고되었습니다.



### Embedded Nonlocal Operator Regression (ENOR): Quantifying model error in learning nonlocal operators (https://arxiv.org/abs/2410.20331)
- **What's New**: 이번 연구에서는 Embedded Nonlocal Operator Regression (ENOR)이라는 새로운 프레임워크를 제안하여, 비국소 동질화(nonlocal homogenization) 모델과 그 구조적 모델 오류를 학습합니다. 이 프레임워크는 장기 시뮬레이션에서 동질화된 소재 반응 예측을 위한 오차 적응 불확실성 정량화를 제공합니다.

- **Technical Details**: 새로운 방식은 비국소 연산자 회귀(Nonlocal Operator Regression, NOR)에 기초하고 있으며, 학습 가능한 커널에서 내재된 모델 오류 항을 포함합니다. Bayesian inference를 통해 모델 오류 항의 매개변수와 커널 매개변수를 추론하며, 다단계 지연 수용 마르코프 체인 몬테 카를로(Multilevel Delayed Acceptance Markov chain Monte Carlo, MLDA-MCMC) 방법을 사용하여 효율적인 Bayesian 모델 보정 및 모델 오류 추정을 수행합니다.

- **Performance Highlights**: 이 방법은 이질적 1차원 바에서 장기 파동 전파를 예측하는 데 적용되었으며, 이전의 가법 노이즈 모델과 비교했을 때 모델 오류를 효과적으로 캡처하여 사후 예측 불확실성의 추정치를 개선하는 성과를 보였습니다.



### Domain Specific Data Distillation and Multi-modal Embedding Generation (https://arxiv.org/abs/2410.20325)
Comments:
          7 pages, 3 figures

- **What's New**: 이 논문은 비구조적 데이터의 소음(noise)을 필터링하기 위해 구조적 데이터를 활용함으로써 도메인 중심의 임베딩(embeddings)을 생성하는 새로운 모델링 접근법을 제시합니다.

- **Technical Details**: 제안된 모델은 Hybrid Collaborative Filtering (HCF) 프레임워크 내에서 작동하며, 일반적인 엔티티 표현을 관련 아이템 예측 과제를 통해 미세 조정합니다. HCF 모델은 비구조적 자유 형식 텍스트 및 회사에 대한 구조적 데이터를 모두 활용하여 데이터의 내재된 관계를 캡쳐합니다.

- **Performance Highlights**: HCF 기반 임베딩은 AutoEncoder 기반 임베딩보다 성능이 뛰어났으며, 도메인 특정 속성 예측에서 정밀도(precision)가 28%, 재현율(recall)이 11% 향상되었습니다.



### ProtSCAPE: Mapping the landscape of protein conformations in molecular dynamics (https://arxiv.org/abs/2410.20317)
Comments:
          Accepted as a short paper at the 5th Molecular Machine Learning Conference (MoML 2024)

- **What's New**: ProtSCAPE는 단백질 구조의 동적 특성을 이해하기 위한 새로운 딥러닝 아키텍처로, 기하학적 산란 변환(geometric scattering transform)과 트랜스포머 기반의 어텐션 메커니즘을 결합하여 분자 동역학(MD) 시뮬레이션에서 단백질 동역학을 포착합니다.

- **Technical Details**: ProtSCAPE는 학습 가능한 기하학적 산란 모듈, 잔여(residue)와 아미노산(amino acid) 신호에 대한 이중 어텐션 메커니즘을 포함한 트랜스포머 모델 등 세 가지 주요 부분으로 구성됩니다. 이 아키텍처는 MD 경로의 잠재 표현을 학습하고, 특정 잔여 및 그 구성 아미노산을 식별하여 단백질의 유연성을 강조합니다.

- **Performance Highlights**: ProtSCAPE는 학습한 표현을 활용하여 폐쇄(open) 상태와 개방(close) 상태 간의 상(Dynamic transitions) 변화를 이해하고, 단백질 돌연변이에 대한 동역학을 잘 일반화합니다. 이를 통해 단백질의 구조 및 단백질 변이체의 변형 공간을 파악할 수 있는 강력한 도구로 자리 잡았습니다.



### Q-Distribution guided Q-learning for offline reinforcement learning: Uncertainty penalized Q-value via consistency mod (https://arxiv.org/abs/2410.20312)
Comments:
          Neurips 2024

- **What's New**: 이번 연구에서는 Offline Reinforcement Learning에서 발생하는 Q-value 추정의 비관점적(overestimation) 문제를 해결하기 위해, Uncertainty(불확실성) 측정을 기반으로 한 Q-Distribution Guided Q-Learning(QDQ) 방법을 제안합니다. 이 방법은 Q-values가 Out-of-Distribution(OOD) 구역에서 불확실성이 높을 때 패널티를 부여하여 Q-value의 정확도를 높입니다.

- **Technical Details**: QDQ는 행동 정책의 Q-value 분산을 학습하여 OOD 행동의 Q-value를 평가하고 불확실성을 활용한 패널티를 적용합니다. 이는 q-value function의 업데이트에 있어 OOD 행동의 불확실성을 정량화하여 안정적인 Q-value 학습을 지원합니다. 또한, 불확실성에 민감한 최적화 목표를 도입하여 Q-values의 보수적 추정을 방지합니다.

- **Performance Highlights**: QDQ는 D4RL 벤치마크에서 일관된 성능 향상을 보여주며, 여러 과제에서 기존 방법들에 비해 유의미한 개선을 달성했습니다. 이 방법은 OOD 영역에서의 Q-values의 비관점적 수정이 학습 정책의 성능에 실질적인 기여를 할 수 있음을 보였습니다.



### ANOMIX: A Simple yet Effective Hard Negative Generation via Mixing for Graph Anomaly Detection (https://arxiv.org/abs/2410.20310)
- **What's New**: 본 논문에서는 ANOMIX라는 새로운 프레임워크를 제안하여 GAD(Anomaly Detection) 작업에 필요한 샘플 수를 줄이는 방법을 소개합니다.

- **Technical Details**: ANOMIX는 (1) ANOMIX-M이라는 새로운 그래프 믹싱(Graph Mixing) 접근 방식을 적용하여 하드 네거티브(hard negatives)를 생성하고, (2) 노드(node) 및 서브그래프(subgraph) 레벨의 대조(Contrasts)를 통해 기본적인 이상치를 구별합니다.

- **Performance Highlights**: ANOMIX는 AUC(Area Under the Curve)에서 최대 5.49% 향상되었으며, 속도는 1.76% 빨라졌습니다. GCL(Graph Contrastive Learning)에서 필요한 샘플 수를 거의 80% 줄이는 성과를 보여주었습니다.



### Accelerating Direct Preference Optimization with Prefix Sharing (https://arxiv.org/abs/2410.20305)
Comments:
          To appear in NeurIPS 2024 in the Fine-Tuning in Machine Learning Workshop

- **What's New**: 이번 논문에서는 preference tuning을 위한 prefix sharing이라는 새로운 기술을 도입합니다. 이는 선택된(response) 및 거부된(rejected) 응답을 공유 프리픽스와 함께 하나의 시퀀스로 처리하여 계산의 중복을 줄이는 방식입니다.

- **Technical Details**: 제안된 방법에서는 맞춤 attention mask를 사용하여 두 응답 간의 contamination을 방지합니다. 이를 통해 학습 속도를 크게 향상시키며, DPO(Direct Preference Optimization)와 같은 인기 있는 방법에 적용 가능합니다. 또한, prefix 공유와 시퀀스 패킹을 결합하여 강력한 성능 향상을 실현합니다.

- **Performance Highlights**: 이 방식은 DPO 데이터셋에서 1.1배에서 1.5배의 훈련 처리량 개선을 달성했으며, 추가로 시퀀스 패킹을 적용했을 때 1.3배에서 1.6배의 일관된 속도 향상을 보였습니다.



### Sequential Large Language Model-Based Hyper-Parameter Optimization (https://arxiv.org/abs/2410.20302)
- **What's New**: 이번 연구에서는 하이퍼파라미터 최적화(hyperparameter optimization, HPO)를 위해 대규모 언어 모델(large language models, LLMs)을 활용하는 혁신적인 프레임워크인 SLLMBO를 소개합니다. SLLMBO는 동적 탐색 공간 적응(dynamic search space adaptability), 향상된 파라미터 환경 활용(parameter landscape exploitation), 그리고 새로운 하이브리드 LLM-트리 구조 파젠 추정기(LLM-Tree-structured Parzen Estimator, LLM-TPE) 샘플러를 통합하고 있습니다.

- **Technical Details**: SLLMBO는 최근의 완전 LLM 기반 방법들과 전통적인 베이지안 최적화(Bayesian Optimization, BO)의 한계를 극복하여 보다 강력한 최적화를 달성합니다. 이번 연구에는 GPT-3.5-turbo, GPT-4o, Claude-Sonnet-3.5, Gemini-1.5-flash를 포함한 여러 LLM에 대한 포괄적인 벤치마킹이 이루어졌으며, 이는 HPO를 위한 다양한 LLM을 평가한 최초의 프레임워크로 자리잡고 있습니다.

- **Performance Highlights**: 14개의 표 형식(tabular) 작업에서 LLM-TPE 샘플러는 완전 LLM 기반 방법보다 뛰어난 성과를 나타내었으며, 9개 작업에서 BO 방법보다 우수한 결과를 달성했습니다. 예산 제약이 있는 시나리오에서의 조기 중단 테스트는 경쟁력 있는 성능을 추가로 입증하여 LLM 기반 방법들이 최적 결과를 위해서는 반복(iteration)을 연장해야 함을 보여주었습니다.



### Predicting Mortality and Functional Status Scores of Traumatic Brain Injury Patients using Supervised Machine Learning (https://arxiv.org/abs/2410.20300)
- **What's New**: 이번 연구는 외상성 뇌손상(Traumatic Brain Injury, TBI) 환자의 사망률 및 기능 상태 척도(Function Status Scale, FSS) 점수를 예측하기 위해 감독된 머신러닝(supervised machine learning, ML) 방법을 새롭게 적용했습니다.

- **Technical Details**: 이번 연구는 콜로라도 대학교 의과대학의 실제 데이터셋을 활용해 300명의 소아 TBI 환자의 임상적 특성(클리니컬 피쳐, clinical features)을 분석했습니다. 사망률 예측에는 18개의 ML 모델을 평가하였고, FSS 점수 예측에는 13개의 모델을 평가했습니다. 성능 평가는 정확도(accuracy), ROC AUC, F1 점수(F1-score), 평균 제곱 오차(mean squared error)를 사용했습니다. 로지스틱 회귀(Logistic regression) 및 Extra Trees 모델이 사망률 예측에서 높은 정확도를 보였고, 선형 회귀(linear regression)가 FSS 점수 예측에서 가장 우수한 성과를 나타냈습니다. 또한, 특징 선택(feature selection)을 통해 103개의 임상 변수를 가장 관련성 높은 변수로 줄여 모델의 효율성과 해석 가능성을 향상시켰습니다.

- **Performance Highlights**: 연구 결과, 머신러닝 모델이 높은 위험 환자를 식별하고 개인 맞춤형 개입(personalized interventions)을 지원하는 데 중요한 역할을 함을 강조하였습니다. 데이터 기반 분석(data-driven analytics)의 잠재력을 통해 TBI 치료를 개선하고 임상 워크플로우(clinical workflows)에 통합할 수 있는 가능성을 보여주었습니다.



### DeCaf: A Causal Decoupling Framework for OOD Generalization on Node Classification (https://arxiv.org/abs/2410.20295)
- **What's New**: 본 논문은 Graph Neural Networks (GNNs)의 일반화 가능성을 향상시키기 위한 새로운 접근법인 DeCaf를 제안합니다. 기존의 방법들이 데이터 생성 과정에 대한 단순화된 가정에 의존하는 반면, 우리는 Structural Causal Models (SCMs)를 활용하여 보다 현실적인 그래프 데이터 생성 모델을 소개합니다.

- **Technical Details**: DeCaf 프레임워크는 노드의 특징(feature)과 구조(structure) 간의 관계를 독립적으로 학습함으로써, unbiased feature-label 및 structure-label 매핑을 제공합니다. 우리는 이 접근법이 다양한 분포 변화의 영향을 효과적으로 완화할 수 있음을 이론적으로 제시합니다. 또한, Generalized Robinson Decomposition을 활용한 구현 가능한 Paradigm을 제공합니다.

- **Performance Highlights**: DeCaf는 실제와 합성 데이터셋을 통해 다양한 분포 변화 패턴에 대한 평가를 진행하였고, 그 결과 GNN의 노드 분류 작업에서 일반화 능력을 일관되게 향상시키는 성능을 보였습니다.



### A Systematic Review of Machine Learning Approaches for Detecting Deceptive Activities on Social Media: Methods, Challenges, and Biases (https://arxiv.org/abs/2410.20293)
- **What's New**: 소셜 미디어에서 잘못된 정보를 탐지하기 위한 자동화 시스템의 필요성이 높아짐에 따라, 36개의 연구를 체계적으로 검토하여 기계 학습(ML) 및 딥 러닝(DL) 모델의 적용을 평가하였다.

- **Technical Details**: 리뷰에서는 Prediction model Risk Of Bias ASsessment Tool (PROBAST)를 사용하여 ML 라이프 사이클 전반에서 주요 편향을 식별하였다. 여기에는 비대표적인 샘플링으로 인한 선택 편향, 클래스 불균형의 부적절한 처리, 불충분한 언어 전처리(예: 부정 표현) 및 일관되지 않은 하이퍼파라미터 조정이 포함된다.

- **Performance Highlights**: Support Vector Machines (SVM), Random Forests, Long Short-Term Memory (LSTM) 네트워크와 같은 모델이 강력한 잠재력을 보였으나, 불균형 데이터 설정에서 정확도에 과도하게 의존하는 것이 일반적인 문제로 지적되었다. 따라서 데이터 전처리 개선(예: 리샘플링 기법), 일관된 하이퍼파라미터 조정 및 정밀도(precision), 재현율(recall), F1 점수, AUROC와 같은 적절한 평가 지표의 사용이 필요하다.



### Classification under strategic adversary manipulation using pessimistic bilevel optimisation (https://arxiv.org/abs/2410.20284)
Comments:
          27 pages, 5 figures, under review

- **What's New**: 본 연구는 공격자가 최소 비용의 솔루션을 선택한다는 기존 가정을 제거하고, 데이터가 특정 분포에서 샘플링된다고 가정하면서 새로운 모델을 제안합니다. 이를 통해 복수의 최적 솔루션이 가능해지며, 기존 모델의 제약을 벗어나 보다 현실적인 문제 해결을 목표로 합니다.

- **Technical Details**: 이 논문에서는 공격자를 확률적 데이터 생성기로 모델링하여 이원 최적화 문제를 설정하고, 학습자가 선도자로서 행동합니다. 우리는 그라디언트 기반 방법 대신, 반응 함수 개념을 적용하여 시스템 방정식을 구성하고, Levenberg-Marquardt 방법을 사용하여 해결합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 스팸 이메일 및 가짜 리뷰에서 공격자의 영향을 정확하게 포착하며, 기존 모델보다 더 나은 성능을 보여줍니다. 새로운 접근 방식은 향후 데이터에 대한 예측 정확도를 개선하는 것을 목표로 합니다.



### Proactive Fraud Defense: Machine Learning's Evolving Role in Protecting Against Online Fraud (https://arxiv.org/abs/2410.20281)
Comments:
          World Journal of Advanced Research and Reviews (2024)

- **What's New**: 온라인 사기(online fraud)가 점점 더 정교해지고 널리 퍼짐에 따라 기존의 사기 탐지 방법들은 변화하는 사기꾼의 전술에 대응하기에 어려움을 겪고 있습니다. 본 논문은 이러한 문제를 해결하기 위해 기계 학습(machine learning)의 변혁적 역할을 탐구하고 있습니다.

- **Technical Details**: 논문에서는 Random Forest, Neural Networks, Gradient Boosting과 같은 주요 모델들을 분석하여, 기계 학습이 방대한 데이터셋을 처리하고 복잡한 사기 패턴(fraud patterns)을 식별하기 위한 강점을 강조합니다. 기계 학습 모델은 새로운 데이터로부터 지속적으로 학습하며, 복합적인 사기 계획에 적응할 수 있습니다.

- **Performance Highlights**: 기계 학습 모델들은 사기 발생 후 반응하는 규칙 기반 시스템(rule-based systems)과 달리, 실시간 예측(real-time predictions)을 제공하여 능동적인 사기 예방(proactive approach to fraud prevention)을 가능하게 합니다. 따라서, 거짓 긍정(false positives)을 줄이고 재정적 손실(financial losses)을 최소화하는 데 기여하고 있습니다. 향후 발전 가능성으로는 딥 러닝(deep learning) 및 하이브리드 모델(hybrid models)이 포함되어, 이러한 시스템의 예측 정확성(predictive accuracy) 및 적용성을 더욱 향상시킬 것으로 기대됩니다.



### Library Learning Doesn't: The Curious Case of the Single-Use "Library" (https://arxiv.org/abs/2410.20274)
Comments:
          24 pages, 7 figures. Accepted to the 4th MATH-AI Workshop at NeurIPS'24

- **What's New**: 이번 연구에서 대형 언어 모델(LLM) 기반의 수학 도구 학습 시스템인 LEGO-Prover와 TroVE를 분석하여 기존의 성능 향상이 실제로 재사용 가능한 도구의 학습이 아닌 다른 메커니즘에 의해 촉진되고 있음을 발견했습니다. 이러한 발견은 재사용의 유용성에 관한 기존 가정에 도전합니다.

- **Technical Details**: LEGO-Prover는 리사이클 가능한 정리 정식 이론을 학습하는 것을 목표로 하며, miniF2F 데이터셋을 사용하여 평가되었습니다. TroVE는 리사이클 가능한 Python 함수를 학습하여 프로그램 문제를 해결하는 데 중점을 둡니다. 두 시스템 모두에서 함수 재사용은 드물며, 성능 향상은 주로 자기 수정(self-correction) 및 자기 일관성(self-consistency)에 기인함을 보입니다.

- **Performance Highlights**: LEGO-Prover는 20,000개 이상의 정리를 학습했으나, 최종 해결 단계에서 실제로 재사용된 정리는 6%에 불과하며, 오직 하나의 정리만이 재사용되었습니다. 연구 결과는 두 시스템 모두에서 기능 재사용의 효과가 미미하며, 오히려 자기 수정 메커니즘이 주요 성능 향상 요인임을 시사합니다.



### Centaur: a foundation model of human cognition (https://arxiv.org/abs/2410.20268)
- **What's New**: 본 논문에서는 Centaur라는 새로운 계산 모델을 소개합니다. Centaur는 인간의 행동을 예측하고 시뮬레이션할 수 있는 모델로, 자연어로 표현 가능한 어떤 실험에서도 적용 가능합니다.

- **Technical Details**: Centaur는 Llama 3.1 70B라는 최첨단 언어 모델을 기반으로 하여 개발되었습니다. 이 모델은 Psych-101이라는 대규모 데이터 세트를 기반으로 미세 조정(finetuning) 되어, 60,000명 이상의 참가자로부터 수집된 10,000,000개의 선택 데이터를 포함하고 있습니다. 모델의 경량 부가 모듈(low-rank adapters)을 사용하여 0.15%의 추가적인 매개변수만으로 훈련 인프라를 구축했습니다.

- **Performance Highlights**: Centaur는 훈련 데이터에 포함되지 않은 참가자들의 행동을 기존의 인지 모델보다 더 잘 예측했습니다. 또한, Centaur는 수정된 문제 구조와 전혀 새로운 도메인에서도 일관된 일반화를 보여주며, 기존의 도메인 특화 모델들과 비교했을 때 모든 실험에서 우수한 성능을 입증했습니다.



### Equivariant Blurring Diffusion for Hierarchical Molecular Conformer Generation (https://arxiv.org/abs/2410.20255)
Comments:
          NeurIPS 2024

- **What's New**: 이 논문은 다단계(multiscale) 관점으로 3D 기하학을 처리하기 위한 새로운 방법론을 제시하며, 특히 분자 그래프에 기반한 3D 분자 구조 생성의 기초적인 생화학적 문제를 해결하는 데 중점을 둡니다.

- **Technical Details**: 저자들은 두 단계의 계층적 접근 방식을 채택하여, 첫 번째로 분자 그래프에서 굵게 표현된(fragment-level) 3D 구조를 생성하고, 두 번째로 그런 구조에서 원자 수준의 세부 사항을 동시에 조정하며 생성합니다. 이를 위해 'Equivariant Blurring Diffusion (EBD)'라는 새로운 생성 모델을 도입하고, 이 모델은 분자 구조의 블러링과 필드 기반 이론에 기반해 SE(3) 등가성을 유지합니다. EBD는 원자 위치를 각각의 단편 중심으로 이동시키고 미세 원자 세부 사항을 복원하는 역과정을 포함합니다.

- **Performance Highlights**: EBD 모델은 약물 같은 분자를 기준으로 한 성능 테스트에서 기존의 최첨단 denoising diffusion 모델보다 우수한 성과를 보였으며, 100배 적은 확산 시간 단계로도 효과적인 결과를 도출했습니다. 이 논문은 EBD의 설계에서 손실 함수와 데이터 변형 과정의 중요성을 강조하며, 그 효과에 대한 철저한 분석을 제공합니다.



### Overcoming the Sim-to-Real Gap: Leveraging Simulation to Learn to Explore for Real-World RL (https://arxiv.org/abs/2410.20254)
Comments:
          NeurIPS 2024

- **What's New**: 이 연구에서는 직관적으로 직접적인 sim2real 전이가 실패할 수 있는 많은 경우에서, 시뮬레이터를 활용해 효율적인 탐사를 가능하게 하는 탐색적 정책(exploratory policies) 세트를 학습할 수 있음을 보여줍니다. 특히, 저자들은 로우랭크 MDP(low-rank MDPs) 세팅 하에서 이러한 정책을 단순한 실용적 방법들과 결합할 때, 현실 세계에서 다항적 샘플 복잡도(polynomial sample complexity)를 달성할 수 있음을 입증했습니다.

- **Technical Details**: 본 연구는 샘플 복잡도를 줄이기 위한 RL 접근법으로 시뮬레이터에서 학습한 탐색적 정책을 실제 환경에 전이하는 원리를 제시합니다. 또한, 저자들은 무작위 탐사(random exploration) 및 최소제곱 회귀(least-squares regression) 오라클과 같은 단순한 방법들과 결합하여 효율적인 학습이 가능하다는 것을 이론적으로 증명했습니다.

- **Performance Highlights**: 실험을 통해, Franka 로봇 플랫폼에서의 실제 로봇 sim2real 문제에 대한 탐색적 정책 전이가 샘플 효율성(sample efficiency) 향상에 실질적인 기여를 할 수 있음을 입증했습니다. 저자들은 이 접근법이 나이브(naïve) 전이가 완전히 실패하는 경우에서도 효율적인 학습을 가능하게 할 수 있음을 보여주었습니다.



### Convergence Guarantees for the DeepWalk Embedding on Block Models (https://arxiv.org/abs/2410.20248)
- **What's New**: 최근의 그래프 임베딩 방법인 DeepWalk의 수렴 특성을 Stochastic Block Model (SBM)에서 분석하였습니다. 이 연구는 비선형 최적화를 통한 그래프 임베딩의 이론적 보장을 제공하려는 노력의 일환으로, 기존의 스펙트럼 임베딩 결과를 반영하여 DeepWalk이 높은 확률로 클러스터 구조를 회복할 수 있음을 보여주고 있습니다.

- **Technical Details**: 논문에서 제시된 방법은 그래프의 구조에 대한 이해를 돕기 위해, 그래프를 랜덤 워크를 통해 탐색하고 비선형 최적화 문제를 해결하여 임베딩을 생성합니다. 특히, SBM에서의 DeepWalk 알고리즘의 수렴 속성을 연구하였으며, 초기값이 충분히 작은 범위 내에서 설정되면 고확률로 커뮤니티를 복구할 수 있음을 보였습니다. 이는 그래디언트 업데이트의 비선형성을 다루는 데 중점을 두었습니다.

- **Performance Highlights**: DeepWalk의 특징은 임베딩을 학습하기 위한 방법으로 그래디언트 강하를 사용하였으며, 실험 결과 커뮤니티 복구가 높은 확률로 가능함을 보여주었습니다. 이 연구는 최신 비선형 임베딩 방법의 이론적 기초를 강화하고 있으며, 특히 작은 차원의 임베딩에서도 결과를 도출하여 실제 적용 가능성을 보여줍니다.



### Model Equality Testing: Which Model Is This API Serving? (https://arxiv.org/abs/2410.20247)
- **What's New**: 이 논문에서는 사용자가 API를 통해 대형 언어 모델과 상호작용하는 과정에서 발생할 수 있는 왜곡(distoritions)을 검출하는 새로운 방법인 Model Equality Testing을 제안합니다. 기존의 API는 이 모델의 출력 분포를 변경할 수 있지만, 사용자에게 이를 통보하지 않습니다. 이 연구는 사용자가 API와 참조 배포(reference distribution)에서 샘플을 수집하고 통계적 테스트를 수행하는 방식을 제시합니다.

- **Technical Details**: 사용자는 API의 출력을 두 개의 샘플로 수집한 후, 두 분포 간의 차이를 측정하기 위해 Maximum Mean Discrepancy(MMD) 기반의 테스트를 수행합니다. 특히, Hamming distance를 기반으로 한 간단한 string kernel을 활용하여 단 10개의 샘플로도 평균 77.4%의 검출력을 보였습니다. 이 방법은 다양한 왜곡(quantization, watermarking, finetuning)을 견뎌낼 수 있는 강력한 도구로 확인되었습니다.

- **Performance Highlights**: 연구팀은 이 테스트를 상업적으로 제공되는 4개의 Llama 모델과 31개의 API 엔드포인트에 적용한 결과, 31개 엔드포인트 중 11개가 메타(Meta)가 발표한 참조 가중치와 다른 분포를 제공하는 것으로 나타났습니다. 또한, 이 방법은 13개의 언어 모델 간의 출력 분포의 통계적 거리를 평가하는 데도 활용되었습니다. 이렇게 함으로써 API의 품질과 일관성을 높일 수 있는 기틀을 마련하였습니다.



### Hoeffding adaptive trees for multi-label classification on data streams (https://arxiv.org/abs/2410.20242)
- **What's New**: 이 논문에서는 Multi-Label Hoeffding Adaptive Tree (MLHAT)라는 새로운 다중 레이블 데이터 스트림 분류 접근법을 제안합니다. MLHAT은 의사결정트리의 분할 과정에서 레이블 간의 가능성 관계 및 공동 발생을 고려하여, 트리의 각 리프 노드에서 학습자를 동적으로 적응시키고, 개념 드리프트를 감지해 성능 저하가 발생하는 트리 가지를 신속히 대체하는 개념 드리프트 감지기를 구현합니다.

- **Technical Details**: MLHAT은 고전적 Hoeffding Adaptive Tree (HAT)에서 발전한 것으로, 다음과 같은 세 가지 주요 혁신을 포함합니다. 1) 의사결정트리의 분할 과정에서 다중 레이블 간의 관계와 공동 발생을 고려합니다. 2) 다중 레이블 스트림에서 레이블 간의 불균형을 동적으로 조정합니다. 3) 데이터 스트림 분포 변화나 개념 드리프트에 따라 의사결정 모델을 업데이트하는 개념 드리프트 감지기를 사용하여, 성능 저하가 시작되면 배경 가지가 주 가지를 대체하도록 합니다.

- **Performance Highlights**: MLHAT은 41개의 데이터셋에서 18개의 다른 온라인 다중 레이블 분류기와 비교한 결과, 12개의 잘 알려진 다중 레이블 메트릭에서 이들 최신 기법을 초월하는 성능을 보여주었습니다. 또한 MLHAT의 소스코드는 공개되어 있어 향후 연구에서 재현성을 증명할 수 있도록 합니다.



### SAFE setup for generative molecular design (https://arxiv.org/abs/2410.20232)
- **What's New**: 이 연구에서는 SAFE(Squential Attachment-based Fragment Embedding) 기반의 생성 모델의 훈련 최적화에 대한 새로운 통찰을 제공합니다. 기존 SMILES 기반 모델들이 갖고 있는 한계를 극복하기 위해 다양한 데이터셋 크기, 데이터 증강, 모델 아키텍처 및 결합 해제 알고리즘의 영향을 분석하였습니다.

- **Technical Details**: 연구는 MOSES 벤치마크 데이터셋을 사용하여 SAFE 기반 생성 모델을 실험하였으며, 다양한 조합의 아키텍처(RNN, GPT-2, LLaMA, Jamba)를 통해 실험을 수행했습니다. LLaMA 아키텍처가 Rotary Positional Embedding을 사용하여 우수한 성능을 보이며, 데이터셋 크기가 성능에 미치는 긍정적인 영향을 입증하였습니다.

- **Performance Highlights**: SAFE 모델은 SMILES 기반 접근법에 비해 scaffold decoration 및 fragment linking 작업에서 일관되게 더 우수한 성능을 보였습니다. BRICS 분해 알고리즘을 통해 최상의 결과를 도출하였으며, 데이터셋 규모가 늘어날수록 테스트 손실이 감소하고 유효성이 향상되었습니다.



### Revisiting Differential Verification: Equivalence Verification with Confidenc (https://arxiv.org/abs/2410.20207)
Comments:
          47 pages (main paper has 16 pages); 8 figures

- **What's New**: 이번 논문에서는 신경망(Neural Networks, NNs)의 가지치기(pruning)와 재훈련(retraining) 이후, 새로운 NN이 원래의 참조 NN과 동등하게 동작한다는 것을 증명하기 위해 차별 검증(differential verification) 아이디어를 재조명합니다. 또한, 새로운 추상 도메인(abstract domain)을 제안하여 동등성에 대한 보다 효율적인 추론을 가능하게 합니다.

- **Technical Details**: 논문에서는 NNs 간의 차이를 기반으로 하는 차별적 추론(differential reasoning)을 통해 해결되지 않는 동등성 속성을 이론적 및 실증적으로 연구합니다. 깨달은 통찰력을 바탕으로, 최근의 신뢰 기반 검증(confidence-based verification) 작업을 이어가며 입력 공간의 대부분에 대한 보장을 제공하는 새로운 동등성 속성을 제안합니다. 이 방법론은 신뢰성 있는 결과를 생성할 수 있게 합니다.

- **Performance Highlights**: 이 접근 방식을 새로운 도구 VeryDiff로 구현하였으며, CERN의 LHC에서 입자 제트 분류를 위한 새로 가지치기된 NN을 포함한 여러 오래된 벤치마크 가족에 대해 광범위한 평가를 진행했습니다. 이를 통해 최신 검증기인 alpha, beta-CROWN에 비해 중위 속도가 300배 이상 향상된 것을 관찰했습니다.



### Generative AI in Health Economics and Outcomes Research: A Taxonomy of Key Definitions and Emerging Applications, an ISPOR Working Group Repor (https://arxiv.org/abs/2410.20204)
Comments:
          36 pages, 1 figure, 2 tables

- **What's New**: 이 논문은 건강 경제학 및 결과 연구(HEOR)에서의 생성적 인공지능(generative AI)의 분류법을 제공하고, 그 응용 분야와 AI 생성물의 정확성 및 신뢰성을 향상시키기 위한 방법들을 제시합니다.

- **Technical Details**: 이 обзор(Review)에서는 생성적 AI의 기본 개념을 정의하고, 현재 HEOR 분야에서의 응용 예시로는 체계적인 문헌 리뷰(systematic literature reviews), 건강 경제 모델링(health economic modeling), 실제 증거 생성(real-world evidence generation), 및 서류 개발(dossier development)을 포함합니다. AI의 정확성 및 신뢰성을 개선하기 위한 접근 방식으로는 프롬프트 엔지니어링(prompt engineering, zero-shot, few-shot, chain-of-thought, persona pattern prompting), 검색 증강 생성(retrieval-augmented generation), 모델 미세 조정(model fine-tuning), 그리고 도메인 특정 모델(domain-specific models)의 사용이 소개됩니다.

- **Performance Highlights**: 생성적 AI는 HEOR에서 효율성, 생산성을 높이고 복잡한 문제에 대한 새로운 해결책을 제시하는 데 중요한 잠재력을 보여줍니다. 기본 모델들(foundation models)은 복잡한 작업의 자동화에 있어 유망하지만, 과학적 신뢰성(scientific reliability), 편향(bias), 해석 가능성(interpretability), 및 작업 흐름 통합(workflow integration) 등에서 여전히 과제가 남아 있습니다. 이 논문은 이러한 AI 도구들의 정확성을 개선하기 위한 전략에 대해 논의합니다.



### Transferable Adversarial Attacks on SAM and Its Downstream Models (https://arxiv.org/abs/2410.20197)
Comments:
          This work is accepted by Neurips2024

- **What's New**: 본 논문은 공개된 segment anything model (SAM)을 활용하여 다양한 downstream 모델에 대한 adversarial 공격의 가능성을 처음으로 탐구합니다. 이는 기존의 전이 기반 공격 방식과 비교해, 특정 downstream 작업 및 데이터셋을 접근하지 않고도 발생하는 adversarial 위험을 실증적으로 보여줍니다.

- **Technical Details**: 새로운 공격 방법론인 universal meta-initialization (UMI) 알고리즘을 제안하여, foundation model에 내재된 취약성을 추출하고 이를 통해 adversarial perturbations 생성을 유도합니다. 공격 과정에서의 gradient 차이를 수학적으로 이론화하며, gradient robust loss를 제안하여 전이 가능성을 향상시킵니다.

- **Performance Highlights**: 제안된 보편적 메타 초기화 및 gradient 강건한 adversarial 공격 (UMI-GRAT)은 실제 적용에서 SAM 및 그 downstream 모델에 대한 효과를 입증했습니다. extensive experiments를 통해 UMI-GRAT의 강력한 성능을 확인하였으며, 기존 전이 기반 adversarial 공격을 개선할 수 있는 중요한 방법론으로 자리잡을 것으로 기대됩니다.



### Uncertainty-Penalized Direct Preference Optimization (https://arxiv.org/abs/2410.20187)
Comments:
          Accepted at the NeurIPS 2024 FITML Workshop

- **What's New**: 이번 연구에서는 DPO(Direct Preference Optimization)의 과최적화 문제를 해결하기 위해, 불확실성 패널라이제이션(uncertainty penalization) 방안을 도입한 비관적(pessimistic) 프레임워크를 개발하였습니다.

- **Technical Details**: DPO는 인공지능 모델에서 선호 데이터의 가능성을 극대화하여 정책을 미세 조정하는 기법으로, 잘못 라벨링된 또는 모호한 선호 쌍들로부터 오는 패널라이징 효과를 제시합니다. 이 연구는 평가 데이터인 Anthropic-HH 데이터셋을 이용하여 GPT2 Medium 모델의 앙상블을 통해 수행합니다.

- **Performance Highlights**: 제안된 방법은 기존의 DPO보다 전반적으로 향상된 성능을 보였으며, 높은 불확실성을 가지는 응답에서 더 나은 완성을 보여주었습니다.



### Chemical Language Model Linker: blending text and molecules with modular adapters (https://arxiv.org/abs/2410.20182)
Comments:
          25 pages, 3 figures

- **What's New**: ChemLML(Chemical Language Model Linker)이라는 경량 어댑터 기반 전략을 제안하여 다중 모드 모델을 기존의 고품질 사전 훈련된 모델을 활용하지 않고도 훈련할 수 있는 접근 방식을 변화시킵니다. ChemLML은 두 개의 독립적인 도메인 모델을 혼합하여 텍스트 설명으로부터 조건부 분자 생성을 달성합니다.

- **Technical Details**: ChemLML은 SMILES(Simplified Molecular Input Line Entry System)와 SELFIES(Self-referencing Embedded Strings)로 표현된 분자를 사용하며, 이 둘의 선택이 조건부 분자 생성 성능에 큰 영향을 미친다는 점을 발견했습니다. ChemLML은 텍스트 모델과 분자 생성 모델 간의 상호작용을 탐구하는 유연한 접근 방식을 제공합니다.

- **Performance Highlights**: ChemLML은 적은 수의 어댑터 매개변수만으로 훈련되며, 텍스트 지향 분자 설계 작업에서 동일한 크기의 모델에 비해 훨씬 적은 훈련 가능한 매개변수를 필요로 하면서도 뛰어난 성능을 달성합니다. 실제 사용 사례에서는 ChemLML을 통해 후보 단백질 억제제를 생성하고 도킹(docking)을 통한 품질 평가를 실시했습니다.



### Copyright-Aware Incentive Scheme for Generative Art Models Using Hierarchical Reinforcement Learning (https://arxiv.org/abs/2410.20180)
Comments:
          9 pages, 9 figures

- **What's New**: 이 연구는 copyright(저작권) 법과 법원 판례에 기반한 새로운 저작권 메트릭을 도입하고, TRAK 방법을 활용하여 데이터 소유자의 기여도를 추정하는 방식을 제안합니다. 또한, 연속적인 데이터 수집 과정을 고려하여 학습을 여러 라운드로 나누고, 강화 학습을 기반으로 한 계층적 예산 배분 방법을 설계하여 각 라운드의 예산을 결정하며 데이터 소유자의 기여도와 저작권 손실에 따라 사례별 보상을 설정합니다.

- **Technical Details**: 기존의 데이터 보호 방법(예: 노이즈 주입, 워터마킹, 기계 불학습 등)은 종종 출력 품질 저하를 초래하며, 특히 generative art 분야에서 저작권 침해 문제를 해결하기 어렵습니다. 본 연구는 법적 기준에 부합하는 저작권 메트릭을 개발하여 생성 모델을 위한 인센티브 기획을 설계하였으며, 이를 통해 모델 제작자가 기여와 저작권 손실을 기반으로 데이터 소유자에게 보상할 수 있도록 하였습니다.

- **Performance Highlights**: 세 가지 데이터 세트를 통한 광범위한 실험 결과, 본 방법이 기존의 8개 벤치마크를 초월하여 저작권을 고려하는 방식으로 예산 배분을 최적화하여 효과성을 증명하였습니다. 특히, 고품질 및 독창적 이미지를 보유한 데이터 제공자에게 더욱 매력적이라는 점이 강조되었습니다.



### Beyond Simple Sum of Delayed Rewards: Non-Markovian Reward Modeling for Reinforcement Learning (https://arxiv.org/abs/2410.20176)
- **What's New**: 본 논문에서는 RL(강화학습)에서 Composite Delayed Reward (RLCoDe)라는 새롭고 일반화된 문제를 도입합니다. 이 연구는 전통적인 지연 보상 모델에서 강한 가정을 제거하고, 지연 보상이 더 복잡한 구조에서 기인할 수 있음을 제안합니다.

- **Technical Details**: Composite Delayed Reward Transformer (CoDeTr)는 non-Markovian 보상 모델을 적용하여, 주어진 행동 시퀀스의 보상을 더 정확하게 예측합니다. 또한, in-sequence attention 메커니즘을 도입하여 시퀀스 내에서 보상의 기여도를 효과적으로 모델링합니다.

- **Performance Highlights**: CoDeTr는 다양한 기준에서 기존의 지연 보상 방법들을 능가하는 성능을 보여주었습니다. 실험 결과, 이 모델은 환경의 피드백을 잘 반영한 보상을 예측하고, 시퀀스 내에서 가장 중요한 단계를 효과적으로 식별할 수 있었습니다.



### Alternatives of Unsupervised Representations of Variables on the Latent Spac (https://arxiv.org/abs/2410.20172)
Comments:
          20 pages, 15 figures, 4 tables

- **What's New**: 이 논문은 비지도 기계 학습(unsupervised machine learning)을 통해 2D 잠재 공간(latent space)에서 변수를 표현하는 새로운 방법을 제시합니다. 특히 변수를 낮은 차원 공간에 표현하는 것은 데이터를 시각화하고, 기본 특성에 따라 변수를 분리하며, 의미 있는 패턴과 이상치를 찾는 데 도움이 됩니다.

- **Technical Details**: 논문에서는 변수를 잠재 공간에 표현하기 위한 다섯 가지 방법이 소개되었습니다: (1) 간단한 전치(transposed), (2) 변수 통계, 경험적 확률 밀도(empirical probability density) 및 누적 분포 함수(cumulative distribution functions)와 같은 단일 변수 메타데이터(univariate metadata), (3) 상관관계(correlations), R2 값, 자카드 지수(Jaccard index), 코사인 유사도(cosine similarity), 상호 정보(mutual information)와 같은 다양한 메트릭(metrics)의 인접 행렬(adjacency matrices), (4) 기울기 맵핑(gradient mappings) 후 스팟 크로스 프로덕트(spot cross product) 계산, (5) 혼합(combined). 베타-VAE(beta-VAE)를 통한 변수를 표현하는 28가지 접근 방안을 검토하였고, 쌍별 스팟 크로스 프로덕트는 두 변수의 기울기 관계를 다룹니다.

- **Performance Highlights**: 논문은 세 가지 사례를 통해 설명하고 있습니다: (1) 알려진 의존성을 가진 합성 데이터(synthetic data), (2) 손으로 쓴 숫자에 대한 유명한 MNIST 예제, (3) 캐나다 금융 시장 이자율의 실제 다변량 시계열(real-world multivariate time series). 결과적으로, 잠재 공간에서 이자율이 채권(bonds), T-비율(T-bills), GICs, 전통적인 모기지(conventional mortgages)와 같은 타입에 따라 정확하게 분리되었고, 채권과 T-비율이 동일한 곡선(curve) 상에 놓이며, 해당 곡선을 따라 이자율이 순서화되었습니다.



### Infectious Disease Forecasting in India using LLM's and Deep Learning (https://arxiv.org/abs/2410.20168)
Comments:
          16 pages, 4 figures

- **What's New**: 이 논문은 여러 전염병 데이터와 기후 데이터를 바탕으로 전염병 발생의 예측 모델을 개발합니다. 이는 기존의 접근 방식이 가지는 한계를 극복하고 보다 효과적인 예방 시스템을 구축하기 위한 중요한 초석이 될 것입니다.

- **Technical Details**: 본 연구에서는 딥러닝 알고리즘과 LLM(대형 언어 모델)을 사용하여 감염병 발생의 심각도를 예측합니다. 이러한 예측 모델은 전염병 발생을 우려하는 다양한 변수를 통합하여 질병 전파의 복잡성을 이해하는 것을 목표로 합니다. 데이터를 수집하기 위해 정확한 웹 스크래핑 기법이 활용되었으며, 인도의 특정 질병에 대한 긴 역사적 데이터를 포함하고 있습니다.

- **Performance Highlights**: 이 예측 모델은 과거의 전염병 발생 데이터를 분석하여 향후 발생 가능성을 높은 정확도로 예측할 수 있는 능력을 갖추고 있습니다. 연구진은 기후 요인과 질병 전파 간의 상관관계를 파악함으로써 공공 건강에 미치는 영향을 줄이고 보건 시스템의 복원력을 강화하려고 합니다.



### DeepMIDE: A Multivariate Spatio-Temporal Method for Ultra-Scale Offshore Wind Energy Forecasting (https://arxiv.org/abs/2410.20166)
- **What's New**: DeepMIDE는 다중 출력 통계 딥 러닝 방법으로, 오프쇼어(Offshore) 풍속을 공간, 시간 및 높이의 차원에서 동시에 모델링합니다. 이는 기존의 단일 대표 높이에 초점을 맞춘 바람 예측 방법과는 차별화됩니다.

- **Technical Details**: DeepMIDE는 다변량(nonstationary) 및 상태 의존적(state-dependent) 커널(kernel)을 가진 다중 출력 적분-차분 방정식(multi-output integro-difference equation) 모델로 설계되었습니다. 이 모델은 물리적 바람장 형성과 전파의 물리적 정보를 인코딩하는 일련의 전단 벡터(advection vectors)를 통해 바람 예측을 수행합니다.

- **Performance Highlights**: 실제 데이터에 대한 실험에서 DeepMIDE가 시간 시계열, 공간-시간 및 딥 러닝 방법보다 우수한 풍속 및 전력 예측 성과를 보였습니다.



### Prompt Diffusion Robustifies Any-Modality Prompt Learning (https://arxiv.org/abs/2410.20164)
Comments:
          Under review

- **What's New**: 이 논문에서는 기존의 고정된 프롬프트(method of employing fixed prompts) 접근 방식이 새로운 데이터 샘플에 대해 일반화할 때 발생하는 문제를 해결하기 위해, '프롬프트 확산(prompt diffusion)'이라는 새로운 방법을 도입합니다. 이 방법은 각 샘플에 맞춤화된 프롬프트를 생성하기 위해 확산 모델을 활용하여 점진적으로 프롬프트를 수정합니다.

- **Technical Details**: 프롬프트 확산은 프롬프트 공간(prompt space) 내에서 새로운 생성 경로를 학습하면서 임의의 프롬프트에서 더 개인화된 프롬프트로 전이하는 프로세스를 포함합니다. 이 방법은 다섯 단계로 테스트 샘플 프롬프트를 최적화하는 빠른 ODE 기반 샘플링 전략을 사용하여 성능 개선과 계산 효율성 간의 균형을 이룹니다.

- **Performance Highlights**: 15개의 다양한 데이터셋에 걸친 분류 작업에서, 프롬프트 확산을 적용함으로써 기본-새로운 일반화(base-to-new generalization), 교차 데이터셋 일반화(cross-dataset generalization), 도메인 일반화(domain generalization)의 모든 프롬프트 학습 방법에서 더 강력한 결과를 도출하였습니다.



### Causal Abstraction in Model Interpretability: A Compact Survey (https://arxiv.org/abs/2410.20161)
- **What's New**: 최근 인공지능(AI)의 해석 가능성을 위한 연구가 진행되고 있으며, 그 중에서도 causal abstraction(인과적 추상화)라는 이론적框架(프레임워크)가 주목받고 있습니다. 이 방법은 모델의 행동 뒤에 있는 인과적 메커니즘을 이해하고 설명하기 위한 체계적인 접근법을 제공합니다. 이 논문은 causal abstraction의 이론적 기초, 실제 응용 및 모델 해석 가능성에 대한 함의를 탐구합니다.

- **Technical Details**: Causal abstraction은 구조적 방정식 모델을 기반으로 하여 변수들 간의 인과 관계를 표현하는 함수 집합을 사용합니다. 이 연구는 모델 간 인과적 동등성을 판단하기 위한 수학적 기본을 제공하며, 최근 연구에서 neural networks(신경망 모델)에 대한 인과적 추상화 기술을 적용하여 내부 벡터와 모델 출력 행동 간의 잠재적 인과 관계를 분석합니다.

- **Performance Highlights**: Causal abstraction은 복잡한 머신러닝 모델의 해석을 향상시키고, 모델의 결정 과정을 보다 투명하고 책임감 있게 설명할 수 있도록 도와줍니다. 이 접근법은 기계적 해석 가능성(mechanistic interpretability)에 대한 관심이 고조되는 가운데, 모델의 내부 작동을 이해하는 데 기여하고 있습니다.



### GFlowNet Fine-tuning for Diverse Correct Solutions in Mathematical Reasoning Tasks (https://arxiv.org/abs/2410.20147)
- **What's New**: 이번 연구에서는 GFlowNet(Generative Flow Network)을 사용하여 대규모 언어 모델(LLM)을 수학적 추론 과제에 맞춰 미세 조정하는 방법을 제시합니다. GFlowNet은 다양한 해결책을 생성할 수 있도록 모델을 훈련시킴으로써, 단일 정답에 의존하는 것이 아닌 대안적 해결책을 찾는 데 중점을 두고 있습니다.

- **Technical Details**: GFlowNet은 보상 극대화 강화 학습(RL)과 달리, 정답 함수에 비례하는 분포를 가진 모델을 훈련시켜 다양한 해결책을 생성합니다. GFlowNet은 고품질의 다양한 시퀀스를 샘플링하여 수학적 문제 해결 시, 중간 추론 단계의 다양성을 높이며 최종 정답의 정확성을 유지합니다. 연구에서는 두 가지 데이터셋(GSM8K, MATH)을 사용하여 GFlowNet과 RL의 성능을 비교 평가합니다.

- **Performance Highlights**: 실험 결과, GFlowNet 미세 조정이 다양한 중간 추론 단계를 통해 올바른 최종 답변을 도출하며, 이는 대안적 해결책 생성을 위한 능력의 향상을 나타냅니다. GFlowNet은 다른 접근 방식에 비해 제안된 문제에 대해 더 다양한 정답을 생성하는 것으로 나타났습니다.



### FedMABA: Towards Fair Federated Learning through Multi-Armed Bandits Allocation (https://arxiv.org/abs/2410.20141)
- **What's New**: 이 논문에서는 연합 학습(Federated Learning, FL)에서 클라이언트 간 성능 불균형 문제를 해결하기 위해 적대적 다중 무장 강도자(adversarial multi-armed bandit) 개념을 도입하고, 이를 최적화하기 위한 새로운 FL 알고리즘인 FedMABA를 제안합니다.

- **Technical Details**: FedMABA는 명시적인 성능 제약 조건을 추가하여 클라이언트 성능 분포를 최적화하며, 다양한 데이터 분포를 가진 클라이언트 간의 성능 불공정성을 완화합니다. 이 알고리즘은 손실 분산 제약으로 계산된 공정 가중치와 평균 가중치를 결합한 새로운 업데이트 전략을 설계하였습니다. 이론적으로 FedMABA의 서버 모델 일반화 성능은 클라이언트 성능의 분산에 의해 상한이 설정된다는 것을 증명하였습니다.

- **Performance Highlights**: 복잡한 비독립 동차 데이터(Non-I.I.D.) 시나리오에서 진행된 실험에서 FedMABA는 공정성에서 기존 방법들을 초과한 성능을 보였으며, Fashion-MNIST, CIFAR-10, CIFAR-100 데이터셋에서 경쟁력 있는 서버 모델 성능과 안정성을 유지하였습니다.



### Analyzing Multi-Stage Loss Curve: Plateau and Descent Mechanisms in Neural Networks (https://arxiv.org/abs/2410.20119)
- **What's New**: 이 논문은 신경망(neural networks) 훈련 시 관찰되는 다단계 현상을 심층적으로 분석합니다. 초기 설정이 작은 초기화(small initialization)인 경우, 손실 곡선에서 초기 평탄기(initial plateau), 초기 감소(initial descent), 그리고 두 번째 평탄기(secondary plateau)라는 세 가지 뚜렷한 단계를 확인하였습니다.

- **Technical Details**: 이 연구에서는 네트워크의 훈련 동역학을 분석하고, 초기 단계의 동역학이 어떻게 선형 및 비선형 행동으로 이어지는지를 탐구합니다. 손실 곡선에서 각 단계의 동역학을 설명하기 위해 Wasserstein 거리(Wasserstein distance)를 사용하여 파라미터 조정과 글로벌 훈련 트렌드 간의 관계를 설명합니다.

- **Performance Highlights**: 세 가지 훈련 단계는 손실 곡선에서 자주 발견되는 형태이며, 각각의 단계에서의 동역학을 세밀하게 분석함으로써 훈련 속도의 빈약한 원인을 밝히고, 이를 통해 신경망 최적화에 대한 새로운 시각을 제공합니다.



### GeoFUSE: A High-Efficiency Surrogate Model for Seawater Intrusion Prediction and Uncertainty Reduction (https://arxiv.org/abs/2410.20118)
- **What's New**: 이 논문에서는 해수 침입(seawater intrusion) 문제를 다루기 위해 새로운 딥러닝 기반의 대체 모델 GeoFUSE를 개발했습니다. GeoFUSE는 U-Net Fourier Neural Operator (U-FNO)와 Principal Component Analysis (PCA), Ensemble Smoother with Multiple Data Assimilation (ESMDA) 기술을 통합하여 보다 정확하고 빠른 해수 침입 예측을 가능하게 합니다.

- **Technical Details**: GeoFUSE는 U-FNO 서브모델을 사용하여 Beaver Creek의 2D 단면에서 1,500개의 지질 실현을 통해 염도 분포와 축적을 근사합니다. 이 모델은 PFLOTRAN 시뮬레이션을 사용한 기존 방법에 비해 연산 시간을 수천 시간에서 몇 초로 단축하며, 약 360,000배의 속도를 달성했습니다. 또한 PCA와 ESMDA를 통해 모델의 불확실성을 줄이고 지질 모델을 보정하여 예측 정확성을 향상시킵니다.

- **Performance Highlights**: GeoFUSE는 20년 동안의 염도 분포 예측에서 감시 우물의 측정 데이터를 통합하여 지질 불확실성을 획기적으로 줄이고 예측 정확성을 개선했습니다. 또한, 미래 연구에서는 GeoFUSE를 3D 모델로 확장하고 해수면 상승 및 극한 기상 사건과 같은 추가 요소를 통합하여 더 넓은 범위의 해안 및 지하 유동 시스템에 적용할 예정입니다.



### Emergence of Globally Attracting Fixed Points in Deep Neural Networks With Nonlinear Activations (https://arxiv.org/abs/2410.20107)
- **What's New**: 이 논문은 신경망의 레이어 간 숨겨진 표현(hidden representation) 간의 유사성이 진화하는 방식을 전반적으로 분석하는 새로운 이론적 프레임워크를 제시합니다. 특히, 활성화 함수(activation function)에만 의존하며 특정 조건 하에 커널 시퀀스(kernel sequence)가 결정론적으로 변화하는 과정을 보입니다.

- **Technical Details**: 연구에서는 Hermite 다항식(Hermite polynomials)을 사용해 활성화 함수를 확장하고, 이를 통해 커널 맵(kernel map)의 명시적 형태를 도출하며 고정점(fixed points)을 완전히 특성화합니다. 비선형 활성화(nonlinear activation)일 경우, 커널 시퀀스는 전역적으로(unique fixed point) 수렴하여, 이는 활성화 및 네트워크 아키텍처에 따라 직교적 또는 유사한 표현을 나타낼 수 있습니다.

- **Performance Highlights**: 이 연구는 잔여 연결(residual connections)과 정규화 레이어(normalization layers)를 갖는 네트워크에서도 유사한 수렴 행동을 입증하며, 깊이(depth) 및 비선형성(nonlinearity)의 상호작용이 네트워크의 초기화 시점에서 어떻게 작동하는지에 대한 통찰을 제공합니다.



### FedSSP: Federated Graph Learning with Spectral Knowledge and Personalized Preferenc (https://arxiv.org/abs/2410.20105)
- **What's New**: 이번 연구에서는 개인화된 분산 그래프 학습(Personalized Federated Graph Learning, pFGL) 접근법을 통해 Graph Neural Networks (GNNs)의 교육을 비공식적으로 진행하면서 개인의 요구를 충족할 수 있는 방법을 제시합니다. 특히, 도메인 간 구조의 이질성 문제를 해결하기 위한 혁신적인 방법을 도입하였습니다.

- **Technical Details**: 제안된 방법인 FedSSP는 Generic Spectral Knowledge Sharing (GSKS)를 통해 일반적인 스펙트럼 지식을 공유하고, Personalized Graph Preference Adjustment (PGPA)를 도입하여 각 클라이언트의 그래프 구조에 맞춘 개인화된 조정을 제공합니다. 이는 구조적 변화에 따른 지식 충돌을 극복하는 것을 목표로 합니다.

- **Performance Highlights**: 종합적인 실험 결과, 제안된 방법은 크로스 데이터셋 및 크로스 도메인 환경을 통해 여러 최신 방법들보다 뛰어난 성능을 보이었고, 최적의 결과를 달성했습니다.



### Latent Neural Operator Pretraining for Solving Time-Dependent PDEs (https://arxiv.org/abs/2410.20100)
- **What's New**: 본 연구에서는 Latent Neural Operator Pretraining (LNOP) 프레임워크를 제안하며, 이는 Latent Neural Operator (LNO) 기반으로 다양한 시간이 의존적인 편미분 방정식(PDE)을 해결하는 방법에 대한 새로운 접근법을 제공합니다.

- **Technical Details**: LNOP는 공통의 잠재 공간에서 다양한 물리 시스템의 표현을 추출하기 위해 하이브리드 시간-의존 PDE 데이터셋에서 사전 훈련을 수행합니다. 물리학 교차 주의(PhCA) 모듈을 통해 보편적인 변환을 구현하며, 단일 PDE 데이터셋에 대해 파인튜닝(finetuning)을 통해 시간을 의존하는 PDE를 해결할 수 있습니다.

- **Performance Highlights**: LNOP 프레임워크는 네 가지 문제에서 솔루션 오류를 31.7% 줄이며, 파인튜닝 이후 최대 57.1% 향상을 기록했습니다. 또한, 분포 외(out-of-distribution) 데이터셋에서 평균적으로 약 50% 낮은 오류율과 3배의 데이터 효율성을 달성했습니다.



### Self-Normalized Resets for Plasticity in Continual Learning (https://arxiv.org/abs/2410.20098)
- **What's New**: 이번 연구에서는 Self-Normalized Resets (SNR)라는 새로운 적응형 알고리즘을 제안합니다. 이 알고리즘은 뉴런의 발화 비율이 사실상 제로로 떨어질 때 뉴런의 가중치를 재설정함으로써 플라스틱 손실(plasticity loss)을 완화합니다.

- **Technical Details**: SNR 알고리즘은 단일 하이퍼파라미터인 거부 백분위수(threshold)를 사용하여 그룹의 뉴런에서 비활성 상태의 뉴런을 식별하고 재설정합니다. 우리는 SNR이 다양한 지속 학습 문제에 대해 기존의 경쟁 알고리즘들로부터 월등한 성능을 보여준다는 것을 입증하였습니다.

- **Performance Highlights**: SNR은 지속 학습 문제에 있어 경쟁 알고리즘들보다 일관되게 우수한 성능을 달성하며, 하이퍼파라미터에 대한 민감도가 적고, 이상적인 조건에서 ReLU 학습 문제에서도 목표 ReLU를 학습할 수 있음을 보여줍니다.



### OGBench: Benchmarking Offline Goal-Conditioned RL (https://arxiv.org/abs/2410.20092)
- **What's New**: 이번 연구에서는 오프라인 목표 조건 강화 학습(Goal-Conditioned Reinforcement Learning, GCRL) 알고리즘을 위한 새로운 기준 벤치마크인 OGBench를 제안합니다. OGBench는 다양한 환경과 데이터 세트를 포함하여 알고리즘의 성능을 시스템적으로 평가할 수 있도록 설계되었습니다.

- **Technical Details**: OGBench는 8가지 환경 유형과 85개의 데이터 세트, 6개의 대표적인 오프라인 GCRL 알고리즘의 참조 구현을 포함합니다. 각 환경과 데이터 세트는 스티칭(stitching), 장기적 추론(long-horizon reasoning), 고차원 입력 및 확률적(stochasticity) 상황 처리 능력 등을 시험할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과에서는 기존 벤치마크에서 유사한 순위를 기록하는 알고리즘들 간의 뚜렷한 강점과 약점이 드러났습니다. OGBench는 새로운 알고리즘 개발을 위한 탄탄한 기반을 제공하며, 복잡한 작업들을 통해 오프라인 GCRL의 잠재력을 열 수 있게 합니다.



### Sample Efficient Bayesian Learning of Causal Graphs from Interventions (https://arxiv.org/abs/2410.20089)
Comments:
          To appear in Proceedings of NeurIPS 24

- **What's New**: 이번 연구에서는 제한된 개입 샘플을 이용한 인과 그래프 학습을 위한 베이지안 접근 방식을 제안합니다. 이는 현실 세계에서 개입 샘플을 얻는 것이 종종 비싸고 제한적이라는 점을 반영합니다.

- **Technical Details**: 연구에서는 Wienöbst et al. (2023)의 결과를 활용하여 목표 집합의 모든 컷 구성과 해당하는 개입 분포를 효율적으로 나열할 수 있는 방법을 제시합니다. 제안된 알고리즘은 임의로 목표 정점에 개입해 그래프의 모든 엣지를 컷하고, 각 목표 집합의 사후 확률에 따라 인과 그래프를 반환합니다. 충분한 개입 샘플이 주어질 경우, 제안된 알고리즘이 실제 인과 그래프를 높은 확률로 반환함을 이론적으로 증명합니다.

- **Performance Highlights**: 모의 데이터셋에서 다양한 기준 방법들과 비교하여, 제안된 알고리즘이 동일한 정확도를 얻기 위해 더 적은 개입 샘플을 요구하고 결과적으로 더 높은 정확도를 보여주었습니다. 또한, 알고리즘의 성능은 무작위 DAG의 순서와 밀도에 따라 안정적이라는 것이 확인되었습니다.



### emg2qwerty: A Large Dataset with Baselines for Touch Typing using Surface Electromyography (https://arxiv.org/abs/2410.20081)
Comments:
          Submitted to NeurIPS 2024 Datasets and Benchmarks Track

- **What's New**: 이 논문에서는 sEMG(표면 근전도) 신호를 사용하여 QWERTY 키보드에서의 타이핑을 기록한 대규모 데이터세트 emg2qwerty를 소개합니다. 이 데이터세트는 108명의 사용자로부터 수집된 1,135개의 세션을 포함하며, 총 346시간의 녹음으로 이루어져 있어, 공개된 sEMG 데이터세트 중 가장 큰 규모입니다.

- **Technical Details**: emg2qwerty 데이터세트는 각 사용자가 키를 누를 때 생성되는 신호를 분석하여, 키 입력 예측 문제를 다룹니다. 이 신호는 개별 척수 뉴런의 활동을 감지할 수 있는 충분한 민감도를 가지므로, 사전 지식이 없는 상태에서도 사용자의 입력을 예측할 수 있는 모델을 구축하는 데 초점을 맞춥니다. 자동 발화 인식(ASR) 분야에서 사용하는 표준 모델링 기술을 활용하여, sEMG 신호만으로 키 입력을 예측하는 강력한 기초 성능을 보여 줍니다.

- **Performance Highlights**: 이 데이터세트는 다양한 사용자 간의 차이, 세션 간의 변화, 그리고 각 사용자의 타이핑 방식으로 인한 도메인 변화 등 복잡한 요소를 포함합니다. 최초의 검정에서는 새 사용자의 경우 50% 이상의 문자 오류율이 나타나는 것으로 밝혀졌으며, 이는 모델이 성공적으로 키 입력을 변환하는 데 클리핑(인식을 위한 레이블 데이터가 없는 상태에서) 어려움을 겪는다는 것을 나타냅니다.



### CGKN: A Deep Learning Framework for Modeling Complex Dynamical Systems and Efficient Data Assimilation (https://arxiv.org/abs/2410.20072)
- **What's New**: 새로운 연구에서는 복잡한 동적 시스템의 예측과 데이터 동화(data assimilation, DA)를 동시에 수행할 수 있는 새로운 딥러닝 프레임워크인 Conditional Gaussian Koopman Network (CGKN)를 소개합니다. 이 네트워크는 비선형 시스템을 조건부 가우시안 비선형 시스템으로 변환하여 효율적인 예측과 DA를 통합합니다.

- **Technical Details**: CGKN은 비선형 신경 미분방정식(neural differential equations)으로 구성되어 있으며, 세 가지 주요 단계로 작동합니다: 1) 원래 비선형 시스템의 상태를 조건부 가우시안 비선형 시스템으로 변환, 2) 조건부 가우시안 비선형 시스템의 동역학을 학습하여 신속한 상태 예측 및 DA 수행, 3) 예측 및 DA 결과를 원래 시스템으로 변환. CGKN은 실제 비선형 동역학을 캡처함으로써 극단적인 사건과 강한 비가우시안 특성을 효과적으로 다룹니다.

- **Performance Highlights**: CGKN은 프로젝션된 확률적 Burgers--Sivashinsky 방정식, Lorenz 96 시스템 및 El Niño-남방진동과 같은 세 가지 강한 비선형 및 비가우시안 난류 시스템에 대한 예측과 DA에서 효과성을 입증하였습니다. CGKN은 데이터 동화 성능을 손쉽게 연산할 수 있도록 하여 예측 정확도를 개선하며, 기존의 앙상블 방법보다 계산 효율성이 뛰어납니다.



### Understanding the Effect of GCN Convolutions in Regression Tasks (https://arxiv.org/abs/2410.20068)
Comments:
          31 pages

- **What's New**: 본 논문에서는 그래프에서 회귀 작업을 수행할 때, Graph Convolutional Networks (GCNs)와 GraphSage의 컨볼루션 연산자가 오류 학습에 미치는 영향을 정량적으로 분석합니다.

- **Technical Details**: GCNs과 GraphSage 컨볼루션의 통계적 특성을 연구하고, 이들이 이웃 집합의 크기 및 컨볼루션 레이어 수에 따라 어떻게 편향-분산 거래(Bias-Variance Trade-off)에 영향을 미치는지에 대해 논의합니다.

- **Performance Highlights**: 이론적 발견은 합성 실험(Synthetic Experiments)에 의해 뒷받침되어, GCNs에서 컨볼루션 효과에 대한 정량적 이해의 기초를 제공합니다.



### Deep Concept Identification for Generative Design (https://arxiv.org/abs/2410.20061)
- **What's New**: 본 연구는 디자인 대안의 다양성과 선택의 어려움을 해결하기 위한 딥 러닝(Deep Learning) 기술 기반의 개념 식별 프레임워크를 제안합니다.

- **Technical Details**: 이 프레임워크는 생성적 디자인(generative design) 기법을 사용하여 다양한 대안을 생성하고, 딥 러닝 기법을 통해 이 대안들을 여러 카테고리로 클러스터링(clustering)하며, 분류 모델(classification model)을 이용하여 디자인 실무를 위한 배열을 제공합니다. 주요 기술로는 변량적 딥 임베딩(variational deep embedding)과 로지스틱 회귀(logistic regression)가 사용됩니다.

- **Performance Highlights**: 사례 연구로는 2차원 교량 구조의 단순화된 디자인 문제가 적용되어, 제안된 프레임워크의 기본적인 기능을 검증했습니다. 이 과정에서 결정 트리(decision tree)를 기반으로 식별된 개념과 그 관계를 제시하게 됩니다.



### Mechanism learning: Reverse causal inference in the presence of multiple unknown confounding through front-door causal bootstrapping (https://arxiv.org/abs/2410.20057)
Comments:
          12 pages, 6 figures

- **What's New**: 이 논문은 기계 학습(ML) 모델의 한계를 극복하기 위해 "mechanism learning"이라는 새로운 방법을 제안합니다. 이는 관찰 데이터를 혼합(confound)으로부터 해제하여 진정한 인과관계를 학습하도록 ML 모델을 유도합니다.

- **Technical Details**: 제안하는 방법에서 핵심은 front-door causal bootstrapping (전면문 인과 부트스트래핑) 기법을 사용하는 것입니다. 이 방법은 관찰 데이터에서 비인과적인 상관관계를 제거하여 ML 모델이 진정한 특성과 결과(label) 간의 관계를 학습할 수 있도록 합니다. 이 방법의 필요조건은 인과(예측 대상)와 결과(특징 데이터) 간의 인과 매개변수(mechanism variable)가 존재해야 하며, 이 매개변수는 비측정된(confounding) 변수와 독립적이어야 합니다.

- **Performance Highlights**: 실험 결과, mechanism learning을 사용한 ML 모델이 전통적인 감독 학습(supervised learning)을 사용하는 모델보다 비측정된 혼합(confound)에 대해 더 강건하다는 것을 보여줍니다. 특히, 실제 데이터셋에서의 intracranial hemorrhage(ICH) 감지 문제에 대한 모델 테스트에서도, mechanism learning을 적용한 모델이 노이즈에 강한 성능을 보여주었습니다.



### Evaluating Neural Networks for Early Maritime Threat Detection (https://arxiv.org/abs/2410.20054)
- **What's New**: 이번 연구는 보트 활동의 궤적을 분류하는 작업에서 엔트로피 기반 클러스터링 대신 신경망(neural network) 기반 접근법의 정확성을 포괄적으로 평가했으며, 합성 데이터(synthetic data)를 사용하여 다양한 신경망 아키텍처의 성능을 비교했습니다.

- **Technical Details**: 네 가지 신경망 모델을 학습시켰고, 궤적의 길이에 따라 정확도가 어떻게 변하는지를 조사했습니다. 특히, 데이터 정규화(normalization) 및 회전 기반 데이터 증강(rotation-based data augmentation) 기법을 통해 테스트 시간 동안 모델의 강건성을 개선했습니다. 네트워크 기반 모델은 전체 궤적에 대해 최대 100%의 테스트 세트 정확도를 달성했습니다.

- **Performance Highlights**: 신경망 모델은 전체 궤적에서 100%의 정확도를 기록했으며, 궤적의 타임 스텝이 줄어들면서도 안정적으로 성능이 감소했습니다. 반면, 기존의 엔트로피 기반 클러스터링 알고리즘은 97% 이상 달성하지 못했습니다. 이 연구는 합성 데이터에서 신경망 접근이 해양 위협 탐지 분야에서 더 뛰어난 성능을 발휘할 수 있음을 보여줍니다.



### Annotation Efficiency: Identifying Hard Samples via Blocked Sparse Linear Bandits (https://arxiv.org/abs/2410.20041)
Comments:
          31 Pages

- **What's New**: 이 논문은 라벨이 부족한 환경에서 몇 번의 주석 라운드만을 통해 데이터 포인트에 주석을 달 문제를 다루고 있습니다. 기존의 active learning이나 coreset selection 문헌이 신뢰할 수 있는 훈련 모델의 존재를 전제로 하고 있어 우리의 설정과는 적합하지 않다는 점을 강조합니다. 우리는 데이터 포인트에 대한 주석의 난이도에 대한 전문가의 신뢰할 수 있는 피드백을 요청하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 논문에서는 고차원 스페어스(linear sparse) 선형 밴딧(sparse linear bandits) 프레임워크를 사용하여 주석이 필요한 어려운 데이터 포인트를 sequential하게 식별하는 문제를 모델링합니다. 특정 고차원 임베딩을 가진 대량의 라벨이 없는 데이터 세트에서, 각 라운드마다 선택된 데이터 포인트에 대해 전문가가 주석 난이도와 실제 레이블을 함께 제공합니다. 알고리즘 BSLB는 계산적으로 효율적이며 O(k^{1/3} T^{2/3} + k^{-1/12} eta_k^{1/2} T^{5/6})의 후회 보장을 제공합니다.

- **Performance Highlights**: PASCAL-VOC 데이터셋의 이미지 분류 작업을 통해 주석 반응으로 난이도 점수를 사용하는 설정에서 BSLB 알고리즘의 효능이 입증되었습니다. 특히, 우리의 메타 알고리즘 C-BSLB는 최적의 스파시티 파라미터에 대한 지식 없이도 무후회 비용으로 작동합니다.



### Revisiting PlayeRank (https://arxiv.org/abs/2410.20038)
- **What's New**: 이 논문은 2019년에 Pappalardo et al.이 설계한 축구 선수의 성과 점수인 PlayeRank를 수정한 내용을 다룹니다. 주목할 점은, Goal-Scored 이벤트를 학습 단계에서 제거하여 불일치 문제를 해결하고 새로운 가중치를 제시한 것입니다.

- **Technical Details**: PlayeRank는 Linear Support Vector Machine (SVM)을 사용하여 선수의 행동을 기반으로 축구 경기에서의 성과를 평가합니다. 이 시스템은 팀의 성과 벡터(pTm)를 추출하고, 이를 바탕으로 경기 결과(oTm)와의 분류 문제를 해결합니다. 리뷰 후, Goal-Scored 이벤트는 여전히 중요한 영향을 미치지만, 선수의 결정 및 행동을 반영하기 위한 수정된 가중치로 접근합니다.

- **Performance Highlights**: 새로운 PlayeRank를 사용하여 94.13%의 경기에서 우위의 팀이 하위 팀을 이기거나 점수가 비슷할 경우 무승부에 이를 것이라는 결과를 보여주었습니다. 변경된 가중치를 적용했을 시, 우수팀이 이기거나 비길 확률이 85.2%로 감소하였으며, 이를 통해 향후 실시간 분석에 사용될 수 있는 온라인 응용 프로그램 개발의 가능성을 제시합니다.



### Training the Untrainable: Introducing Inductive Bias via Representational Alignmen (https://arxiv.org/abs/2410.20035)
Comments:
          Under Review; 24 pages, 9 figures; Project page and code is at this https URL

- **What's New**: 이번 논문에서는 기존에는 훈련이 어려운 것으로 여겨졌던 아키텍처들이 다른 아키텍처로부터의 inductive bias를 활용하여 훈련될 수 있다는 점을 보여줍니다. ‘Guidance’라는 새로운 기법을 도입하여, 가이드 네트워크가 타겟 네트워크를 유도하며, 이를 통해 타겟 네트워크의 성능을 크게 향상시킬 수 있음이 입증되었습니다.

- **Technical Details**: 이 방법론은 두 네트워크 간의 representational similarity를 맞추는 것을 목표로 하며, neural distance function을 사용하여 층 별로 표현을 일치시키는 방식입니다. 연구자들은 다양한 아키텍처에 대해 이 이론을 적용했으며, 가이드 네트워크가 훈련되지 않은 경우에도 여전히 부분적으로 architectural prior를 전파할 수 있음을 발견했습니다.

- **Performance Highlights**: 이 방법을 활용하여, 일반적인 fully connected 네트워크가 비전 작업에서 즉각적인 overfitting을 극복하게 되었고, plain CNN이 ResNet에 버금가는 성능을 발휘하게 되었으며, RNN과 Transformers 간의 성능 격차가 줄어드는 등의 개선이 이루어졌습니다.



### Sensor2Text: Enabling Natural Language Interactions for Daily Activity Tracking Using Wearable Sensors (https://arxiv.org/abs/2410.20034)
- **What's New**: 본 연구는 Sensor2Text라는 새로운 모델을 제안하여, 웨어러블 센서를 통해 수집한 데이터로 인간의 일상 활동을 추적하고 질문-응답 대화를 수행할 수 있는 혁신적인 접근 방식을 제공합니다. 이는 기존 비디오 기반 솔루션의 사생활 및 시야 제한 문제를 해결합니다.

- **Technical Details**: Sensor2Text는 전이 학습(transfer learning)과 교사-학생 네트워크(teacher-student networks)를 활용하여 시각-언어 모델의 지식을 활용하며, 다중 모달 인코더-디코더 신경망 모델을 설계하여 언어 및 센서 데이터를 공동으로 처리합니다. 대화 및 Q&A 기능을 갖추기 위해 대규모 언어 모델(Large Language Models)을 통합하였습니다.

- **Performance Highlights**: Sensor2Text는 다양한 웨어러블 센서 모달리티를 사용하여 인간 활동을 정확히 식별하고 Q&A 대화를 진행하는 데 뛰어난 성능을 보였습니다. 캡셔닝 및 대화 작업에서 기존의 비주얼-언어 모델과 유사하거나 더 나은 성능을 보였으며, 저조도 환경에서도 강력한 일반화 능력을 발휘하였습니다.



### Off-Policy Selection for Initiating Human-Centric Experimental Design (https://arxiv.org/abs/2410.20017)
- **What's New**: 이 논문에서는 First-glance Off-Policy Selection (FPS)라는 새로운 접근법을 소개하여, 개인의 이질성을 체계적으로 반영하고 전문가 없이도 새로운 참여자에 대한 정책을 선택할 수 있는 방법을 제시합니다.

- **Technical Details**: FPS는 오프라인 데이터셋에서 유사한 행동을 가진 개인들을 하위 그룹으로 나누고, 각 그룹에 대해 정책 선택 기준을 설정합니다. Variational Autoencoder (VAE) 모델을 사용하여 각 하위 그룹의 합성 경로를 생성하고, 바운드된 분산을 가진 무편향 가치 함수 추정기를 개발하여 정책 선택 기준을 결정합니다. 이를 통해 새로운 참여자가 합류할 때, 그들이 속한 하위 그룹에 맞춘 정책을 추천합니다.

- **Performance Highlights**: FPS는 지능형 교육 시스템에서 1,288명의 학생들을 대상으로 5년간 평가되었으며, 교수들이 수작업으로 선정한 정책보다 208% 향상된 학습 성과를 보여 주었습니다. 또한 기존의 OPS 방법에 비해 결과를 136% 증가시켰습니다. 의료 분야의 응용 사례인 패혈증 치료에서도 FPS가 기존 OPS 방법을 초월하며 최적의 치료 정책을 정확히 식별했습니다.



### Enhancing Battery Storage Energy Arbitrage with Deep Reinforcement Learning and Time-Series Forecasting (https://arxiv.org/abs/2410.20005)
Comments:
          Accepted for publication at the 18th ASME International Conference on Energy Sustainability

- **What's New**: 본 연구는 deep reinforcement learning (DRL)과 시계열 예측 (time-series forecasting) 방법을 결합하여 에너지 차익 거래 (energy arbitrage) 성능을 개선하는 새로운 접근 방식을 소개합니다.

- **Technical Details**: 에너지 차익 거래(EA)의 통제 방식으로 DRL을 사용하는데, 심층 신경망(deep neural networks)을 활용한 예측기를 도입하여 Alberta, Canada의 비정상적인 전기 가격 데이터에서 성능 개선 가능성을 연구했습니다. DQN(deep Q-networks) 및 PPO(proximal policy optimization) 알고리즘을 비교하여, 예측기의 수와 예측 기간이 성능에 미치는 영향을 분석했습니다.

- **Performance Highlights**: 복수의 예측기를 결합함으로써 24시간 미래 예측을 기반으로 DQN을 사용할 경우 보상이 60% 증가했습니다. 불완전한 예측에서도 유용한 정보를 제공하여, DRL 에이전트가 이익을 극대화하는 제어 정책을 학습할 수 있음을 나타냅니다.



### Residual Random Neural Networks (https://arxiv.org/abs/2410.19987)
Comments:
          13 pages, 2 figures

- **What's New**: 이 논문에서는 단층 피드포워드 신경망(single-layer feedforward neural network, SLFNN)에서 무작위 웨이트(random weights)을 사용하여 알려진 가정에 반하여, 숨겨진 뉴런(hidden neurons)의 수가 데이터 샘플의 차원(dimensions)과 유사한 경우에도 좋은 분류 성능이 가능하다는 점을 보여주고 있습니다.

- **Technical Details**: 랜덤 SLFNN 모델 사용 시, 고차원 데이터에서의 분류 정확도를 개선할 수 있으며, 이를 위해 남은 오차를 활용한 효율적이고 반복적인 학습 방법(iterative residual training method)을 개발했습니다. 또한, 데이터와 신경망 모델을 보호하기 위한 암호화(encryption) 기법을 제안합니다.

- **Performance Highlights**: 제안된 접근법은 기본적으로 고차원 데이터의 차원을 증가시킴으로써 필요한 숨겨진 신경망 수를 줄이면서도 분류 성능을 높이는 결과를 가져옵니다. 이 제안은 신경망의 분류 정확도를 현저히 향상시키는 데 기여합니다.



### Resolving Domain Shift For Representations Of Speech In Non-Invasive Brain Recordings (https://arxiv.org/abs/2410.19986)
Comments:
          Submitted to ICLR 2025

- **What's New**: 이 연구는 격리된 비침습적(nons-invasive) 신경영상 데이터의 음성 디코딩에서 데이터 세트 간의 일반화 성능을 향상시키기 위한 적대적 도메인 적응 프레임워크의 적용을 최초로 시도합니다.

- **Technical Details**: 본 연구에서는 메그네토엔세팔로그래피(magnetoencephalography, MEG) 데이터를 사용하여 두 가지 음성 디코딩 모델에서 적대적 도메인 적응(adversarial domain adaptation) 기법을 적용하여 데이터 유사성을 증가시키고 훈련 중 일반화 능력을 향상시킵니다. 또한, 참가자의 연령이 MEG 데이터를 활용한 음성 디코딩 태스크에 얼마나 강한 영향을 미치는지를 분석하여 인구 통계적 특징의 영향을 조사합니다.

- **Performance Highlights**: 두 개의 다양한 딥 러닝 모델이 여러 데이터세트를 기반으로 훈련되었으며, 이를 통해 음성 디코딩 성능이 개선되었습니다. 연구의 결과로 새로운 오픈 소스 구현이 제공되어 더 넓은 과학 커뮤니티에 기여할 수 있게 되었습니다.



### SAD: State-Action Distillation for In-Context Reinforcement Learning under Random Policies (https://arxiv.org/abs/2410.19982)
- **What's New**: 이 논문은 사전 훈련된 기초 모델(Foundation Models, FMs)을 이용한 In-Context Reinforcement Learning (ICRL)의 새로운 접근 방식인 State-Action Distillation (SAD)을 제안합니다. 기존의 알고리즘들이 최적 정책이 필요하던 점에 비해, SAD는 무작위 정책으로 생성된 데이터를 사용하여 교육 데이터를 효과적으로 생성하는 최초의 방법입니다.

- **Technical Details**: SAD는 신뢰 구간(trust horizon) 내에서 무작위 정책을 사용하여 전체 상태 및 행동 공간에서 뛰어난 상태-행동 쌍을 증류(distill)합니다. 이 접근 방식은 추가 정보 없이 pretraining dataset을 생성 가능하며, 기존 ICRL 알고리즘이 요구하는 고찰(thorough examination)이나 최적 행동 정책 없이도 작동합니다.

- **Performance Highlights**: 여러 ICRL 벤치마크 환경에서의 실험 결과, SAD는 오프라인 평가에서 평균 180.86%, 온라인 평가에서 평균 172.8%의 성능 향상을 보여줍니다. 이는 기존의 최상의 기준선 알고리즘보다 현저한 성과입니다.



### Global Graph Counterfactual Explanation: A Subgraph Mapping Approach (https://arxiv.org/abs/2410.19978)
- **What's New**: 이 논문은 Graph Neural Networks (GNNs)의 예측을 설명하기 위한 글로벌 카운터팩추얼 설명 방법인 GlobalGCE를 제안합니다. 기존의 GNN 카운터팩추얼 설명은 개별 그래프에 중점을 두어 정보 과부하의 문제가 있었으나, GlobalGCE는 여러 그래프의 관계를 아우르는 기법을 도입합니다.

- **Technical Details**: GlobalGCE는 중요한 서브그래프를 식별하여 카운터팩추얼 설명을 생성하는 방법론을 기반으로 하고 있습니다. 본 연구에서는 서브그래프 생성기와 카운터팩추얼 서브그래프 자동 인코더를 설계하여, 서브그래프와 규칙을 효과적으로 생성하는 구조를 가지고 있습니다. 이를 통해 최대 범위를 달성하는 설명을 제공합니다.

- **Performance Highlights**: 광범위한 실험을 통해 GlobalGCE는 기존의 기법들보다 우수한 성능을 보이며, 설명의 간결성과 직관성을 더욱 향상시켰습니다. 실험 결과는 제안한 방법이 여러 평가 지표에서 최신 연구보다 뛰어난 성능을 갖추었음을 입증합니다.



### Evaluating Cost-Accuracy Trade-offs in Multimodal Search Relevance Judgements (https://arxiv.org/abs/2410.19974)
- **What's New**: 이 논문에서는 다양한 LLMs와 MLLMs의 성능을 평가하고, 특정 사용 사례에 따라 모델 성능이 어떻게 달라지는지를 분석하였습니다. 특히, 시각적 요소가 포함된 모델이 항상 성능을 향상시키지 않음을 밝혀내, 모델 선택의 복잡성을 강조합니다.

- **Technical Details**: 모델 평가 과정은 데이터 수집, 인간 주석, 모델 평가의 세 단계로 구성됩니다. 이 연구는 패션, 호텔 용품, 디자인과 같은 세 가지 데이터셋을 사용하였으며, 각 데이터셋은 다양한 특성과 관련 이미지를 포함합니다. 인간 주석자와 LLM을 사용하여 중요성을 평가하고, Cohen's kappa 계수를 통해 주석자 간의 일치를 측정합니다.

- **Performance Highlights**: 모델의 성능은 맥락에 크게 의존하며, LLM의 비용-정확도 트레이드오프에 대한 추가 연구가 필요합니다. 특히, 작은 규모의 모델에서는 시각적 구성 요소가 성능을 저하시킬 수 있음을 확인하였습니다.



### Understanding Adam Requires Better Rotation Dependent Assumptions (https://arxiv.org/abs/2410.19964)
- **What's New**: Adam 옵티마이저가 파라미터 공간의 회전에 민감하다는 점을 밝혀, 기존 이론적 프레임워크의 한계를 드러냄.

- **Technical Details**: Adam은 회전 불변성이 없는 특성을 가지며, 무작위 회전 시 성능이 저하됨을 실험적으로 입증함. 또한, 특정 회전 방식이 Adam의 성능을 보존하거나 향상시킬 수 있음을 확인함.

- **Performance Highlights**: Adam의 성능 저하는 고정된 기준점에서 시작할 때 발생하며, 회전 의존적인 특성을 추가로 연구해 새로운 이론적 프레임워크 개발의 필요성을 강조함.



### DualMAR: Medical-Augmented Representation from Dual-Expertise Perspectives (https://arxiv.org/abs/2410.19955)
- **What's New**: 본 논문에서는 EHR(전자 건강 기록)의 예측 작업을 향상시키기 위해 DualMAR이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 개별 관찰 데이터와 공공 지식 기반을 모두 활용하여 EHR의 예측 정확도를 높입니다.

- **Technical Details**: DualMAR는 두 가지 주요 구성 요소인 ‘Knowledge Scholar’와 ‘Local Expert’로 이루어져 있습니다. ‘Knowledge Scholar’는 공공 데이터베이스를 기반으로한 진단 주도 KG(지식 그래프)를 구축하고, ‘Local Expert’는 특정 질병-검사 EHR 그래프를 이용하여 의료 코드의 임베딩을 향상시킵니다. 이들 구성 요소는 Encoder-Decoder 아키텍처를 통해 통합되어 예측 모델의 성능을 높입니다.

- **Performance Highlights**: DualMAR는 실험 결과에서 최신 모델보다 우수한 성능을 보여 EHR 예측 및 KG(지식 그래프) 통합의 효과를 입증했습니다. 또한, 복잡하고 다양화된 관계를 가지고 있는 KG가 EHR 모델의 예측 성능을 어떻게 개선할 수 있는지를 보여줍니다.



### Privacy without Noisy Gradients: Slicing Mechanism for Generative Model Training (https://arxiv.org/abs/2410.19941)
Comments:
          accepted to Neurips 2024

- **What's New**: 본 논문에서는 프라이버시 보호를 위한 새로운 학습 패러다임을 소개하며, 이 접근법은 두 가지 단계로 나누어져 있습니다: (i) 개인 데이터의 노이즈가 포함된 저차원 프로젝션(random low-dimensional projections)을 계산하고, (ii) 이러한 프로젝션에 맞춰 생성 모델을 업데이트합니다.

- **Technical Details**: 이 방법은 smoothed-sliced $f$-divergence라는 새로운 정보 이론적 지표를 도입하며, Gaussian 노이즈를 사용하여 원래 및 합성 데이터 분포를 저차원 공간으로 프로젝션한 후 평균하는 과정으로 정의됩니다. 또한, 커널 기반의 차별 가능한 추정기를 제시하여 적대적 훈련(adversarial training) 없이도 생성 모델을 학습할 수 있습니다.

- **Performance Highlights**: 수치 실험을 통해 본 연구에서 제안한 방법이 기존의 표준 프라이버시 메커니즘(DP-SGD, PATE 등)으로 훈련된 생성 모델보다 더 높은 수준의 합성 데이터를 생성하는 것을 보여줍니다.



### Enhancing Safety in Reinforcement Learning with Human Feedback via Rectified Policy Optimization (https://arxiv.org/abs/2410.19933)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)의 유용성과 안전성을 균형 있게 조정하는 새로운 방법인 Rectified Policy Optimization (RePO)를 제안합니다. 기존 방법들과 비교하여 RePO는 각 프롬프트에 대해 더 엄격한 안전성 제약 조건을 적용하여 안전성 간섭을 감소시킵니다.

- **Technical Details**: RePO의 핵심은 rectified policy gradients를 기반으로 한 정책 업데이트 메커니즘입니다. 이 방법은 모든 프롬프트의 엄격한 안전 위반에 대해 패널티를 부여하며, 이를 통해 대부분의 프롬프트에서 안전성을 향상합니다. 연구에서는 Alpaca-7B 모델을 사용하여 RePO의 효과를 empirically 증명했습니다.

- **Performance Highlights**: 실험 결과, RePO는 전통적인 평균 안전 기준과 비교하여 언어 모델의 안전 정렬을 개선하고 안전 간섭을 효과적으로 줄였습니다. 이는 안전성과 유용성을 동시에 활용할 수 있는 새로운 접근임을 시사합니다.



### Provable optimal transport with transformers: The essence of depth and prompt engineering (https://arxiv.org/abs/2410.19931)
- **What's New**: 이 논문은 변환기(transformer)가 최적 수송(optimal transport) 문제를 해결할 수 있는 이론적 보증을 수립하는 데 초점을 맞추고 있습니다. 특히, step size가 조정된 그래디언트 하강법(gradient descent)을 변환기가 구현할 수 있음을 증명하고, 깊이(depth)가 증가함에 따라 성능이 향상될 수 있음을 보여줍니다.

- **Technical Details**: 최적 수송 문제는 조합(combinatorial) 및 연속(continuous) 최적화의 교차점에 있는 기본적인 문제입니다. 이 논문에서는 변환기의 attention 레이어의 계산 능력을 활용하여 Wasserstein-2 거리에서 엔트로피 정규화(entropic regularization)를 포함한 최적 수송 문제의 해결 가능성을 입증합니다. 변환기는 주어진 포인트 수에 상관없이 리스트를 정렬할 수 있는 근사적 접근 방식을 취할 수 있습니다.

- **Performance Highlights**: 이 연구의 결과는 prompt engineering이 알고리즘의 표현력을 증대시키고 변환기가 최적 수송 문제를 해결하는 데 있어 수학적 보증을 제공하는 방식에 대한 새로운 통찰력을 제공합니다. 특히, 여러 attention head를 통해 여러 번의 그래디언트 하강법(iteration of gradient descent)을 시뮬레이션할 수 있으며, 이는 다중 작업 학습(multi-task learning)에서 변환기의 능력을 증명하는 데 기여합니다.



### Prediction of Final Phosphorus Content of Steel in a Scrap-Based Electric Arc Furnace Using Artificial Neural Networks (https://arxiv.org/abs/2410.19924)
Comments:
          53 pages, 8 figures

- **What's New**: 본 연구에서는 스크랩 기반 전기 아크 용광로 과정에서 스틸의 인 포함량 예측을 위한 머신러닝 모델을 개발하였습니다. 이는 환경 영향을 줄이고 스틸 리사이클링을 통해 스틸 시장에서 중요한 점유율을 확보할 것으로 기대됩니다.

- **Technical Details**: 연구는 스틸 공장에서 2년간 수집된 데이터(화학 성분, 중량, 산소 주입량, 공정 지속 시간)를 기반으로 진행되었습니다. 데이터 전처리 후 여러 머신러닝 모델이 평가되었으며, 인공 신경망(ANN)이 가장 효과적인 모델로 나타났습니다. 최적의 ANN 모델은 4개의 은닉층으로 구성되었고, 500 epochs 동안 배치 크기 50으로 훈련되었습니다.

- **Performance Highlights**: 최고 모델은 평균 제곱 오차(MSE) 0.000016, 제곱근 평균 제곱 오차(RMSE) 0.0049998, 결정계수(R2) 99.96%, 상관계수(r) 99.98%의 성능을 보였습니다. 특히, 모델은 +-0.001 wt% (+-10 ppm) 범위 내에서 인 포함량을 100% 정확도로 예측했습니다.



### Disentangling Genotype and Environment Specific Latent Features for Improved Trait Prediction using a Compositional Autoencoder (https://arxiv.org/abs/2410.19922)
- **What's New**: 본 연구에서는 식물 육종 및 유전학 프로그램에서 형질 예측을 개선하기 위한 복합 오토인코더(compositional autoencoder, CAE) 프레임워크를 소개합니다. 기존의 방법들은 유전자형(genotype)과 환경(environment) 요인을 분리하지 않고 예측 모델을 구축해왔습니다. 본 연구는 이러한 차별화를 통해 예측 정확도를 향상시킬 수 있음을 제안합니다.

- **Technical Details**: CAE는 다차원 데이터를 유전자형 별(latent features) 및 환경 별(latent features)로 분해하여 이를 명확히 구분합니다. 이 프레임워크는 오토인코더 내에서 계층적 구조를 채택하여 얽힌 잠재적 특성(latent features)들을 효과적으로 분리합니다. CAE는 재료 데이터셋에 적용하여 환경 요소 모델링의 우수 성능과 다양한 형질(예: 꽃가루 생산일(Days to Pollen) 및 수확량(Yield))의 예측에서 5-10배 향상된 성능을 보였습니다.

- **Performance Highlights**: CAE는 전통적인 방법들보다 예측 성능이 획기적으로 개선되었습니다. 특히, 표준 오토인코더, 회귀를 통한 PCA 및 부분 최소 제곱 회귀(Partial Least Squares Regression, PLSR)와 비교해 보면, 환경 영향을 조명하고 형질 예측력을 높이는 데 있어 강력한 도구로 기능합니다.



### Reinforcement Learning for Aligning Large Language Models Agents with Interactive Environments: Quantifying and Mitigating Prompt Overfitting (https://arxiv.org/abs/2410.19920)
- **What's New**: 이번 논문은 강화학습(RL)을 통해 대형 언어 모델(LLM)의 응답 형식에 대한 민감도를 분석하는 새로운 프레임워크를 제안합니다. 연구 결과, RL 훈련 단계에서 사용된 프롬프트(설명자)와 다른 프롬프트에 직면할 경우 LLM의 성능이 저하된다는 것을 발견했습니다.

- **Technical Details**: 우리는 문맥적으로 감지된 프롬프트 변화에 대한 LLM의 높은 민감도를 명시적으로 정의하며, 특정 환경에서 행동하기 위해 RL을 통해 지식 습득과 LLM 표현을 분석합니다. 대체로, LLM의 라텐트(잠재) 표현과 두드러진 토큰이 프롬프트 구성에 강한 편향을 보임을 발견했습니다. 이를 통해 제시된 프롬프트 변화에 대해 LLM이 로버스트성을 확보하지 못하게 됩니다.

- **Performance Highlights**: 우리는 프롬프트 오버피팅을 완화하기 위한 대비 손실(contrastive loss)을 제안합니다. 이를 통해 LLM의 제로쇼트(zero-shot) 성능 및 프롬프트 변동성에 대한 강인성을 향상시키며, 환경에 대한 새로운 지식 습득을 증진합니다. 궁극적으로, 이 연구는 RL을 활용하여 LLM 에이전트의 행동 능력을 개선하기 위한 장벽을 이해하는 데 기여합니다.



### Provably Adaptive Average Reward Reinforcement Learning for Metric Spaces (https://arxiv.org/abs/2410.19919)
- **What's New**: 이번 연구에서는 Lipschitz MDPs에 대한 무한-horizon 평균 보상 강화 학습(Reinforcement Learning, RL)을 다루고, 상태-행동(space) 공간을 조정하여 유망한 영역으로 집중하는 새로운 알고리즘인 ZoRL을 개발하였습니다.

- **Technical Details**: 이 알고리즘은 상태 공간 차원(d𝒮)과 행동 공간 차원(d𝒜)을 합한 값 d와 문제에 따라 달라지는 확대 차원(d𝑧)을 통해 위험(regret)을 구속합니다. 위험은 O(T^{1 - d_{eff.}^{-1}})로 감소하며, d_{eff.}는 효과적인 차원으로 MDP에 의해 정의됩니다. ZoRL은 정책 공간에서 직접 작업하여 정책의 확대 효과를 보여주는 한편, 고정 격자 사용으로 인해 발생하는 비효율성을 해결하였습니다.

- **Performance Highlights**: 실험을 통해 ZoRL이 기존의 최첨단 알고리즘보다 뛰어난 성능을 발휘하며, 적응성을 통한 성과 향상을 입증했습니다.



### Simmering: Sufficient is better than optimal for training neural networks (https://arxiv.org/abs/2410.19912)
- **What's New**: 본 연구에서는 'simmering'이라는 물리 기반의 훈련 방법을 소개하고, 이를 통해 신경망을 최적화 방식이 아닌 '충분히 좋은(good enough)' 가중치와 편향을 생성하도록 훈련시킬 수 있음을 제시합니다. 이 방법은 전통적인 최적화 방법론에 비해 우수한 성과를 보여줍니다.

- **Technical Details**: 'Simmering'은 Nosé-Hoover 체인 서모스탯(Nosé-Hoover chain thermostat)을 활용하여 신경망의 가중치와 편향을 '입자(particle)'로 취급하고, 역전파(backpropagation)에 의해 생성된 힘을 활용하여 최적 상태에 도달하지 않도록 합니다. 이 과정은 통계역학에서의 분배 함수를 사용하여 충분히 훈련된 네트워크를 알고리즘적으로 생성합니다. 또한, Python과 TensorFlow를 이용하여 수치적 통합을 통해 알고리즘을 구현하였습니다.

- **Performance Highlights**: 시뮬레이션 연구 결과, 'simmering'을 통해 Adam에 의해 생성된 과적합(overfitting)된 네트워크의 성능을 개선했고, 충분히 훈련된 네트워크가 최적 훈련된 네트워크보다 더 우수한 성과를 낼 수 있음을 보여주었습니다. 이는 최적화가 신경망 훈련의 이상적인 출발점이 아닐 수 있다는 것을 시사합니다.



### A Review of Deep Learning Approaches for Non-Invasive Cognitive Impairment Detection (https://arxiv.org/abs/2410.19898)
- **What's New**: 본 리뷰 논문은 비침습적 인지 장애 감지를 위한 딥러닝 접근 방식의 최근 발전을 탐구합니다.

- **Technical Details**: 다양한 비침습적 인지 감소 지표(예: speech and language, facial, motoric mobility)를 검토하며 관련 데이터 세트, 특징 추출 기법 및 딥러닝 아키텍처를 소개합니다. 다양한 방법의 성능을 비교 분석하였으며, 일반적으로 speech 및 language 기반 방법이 가장 높은 탐지 성능을 보였습니다. 음향 및 언어적 특성을 결합한 연구가 단일 모달리티를 사용하는 연구보다 더 나은 성능을 보였습니다. 그러나 facial 분석 방법은 제한적으로 연구되었습니다.

- **Performance Highlights**: 대부분의 연구는 임계 분류 (impaired vs. non-impaired)에 초점을 맞추었으며, 다중 클래스 또는 회귀 작업을 다룬 연구는 적었습니다. 전이 학습(transfer learning)과 사전 훈련된 언어 모델(pre-trained language models)이 특히 언어 분석에 효과적이라는 점에서 인기를 끌었습니다. 그러나 데이터 표준화, 모델 설명 가능성, 장기 분석의 한계 및 임상 적용과 같은 여러 도전 과제가 여전히 남아 있습니다. 미래의 연구 방향으로는 언어 비의존적(spoken) 음성 분석 방법 조사, 다중 모달 진단 시스템 개발, AI 지원 의료에서의 윤리적 고려 사항을 다루는 것이 제안됩니다.



### Air Quality Prediction with Physics-Informed Dual Neural ODEs in Open Systems (https://arxiv.org/abs/2410.19892)
- **What's New**: 본 논문에서는 공기 질 예측을 위한 새로운 물리적 지식 통합 접근 방식인 Air-DualODE를 제안합니다. 이 모델은 두 개의 Neural ODE(exponential decay neural ordinary differential equations) 가지를 통합하여 공기 질 예측의 정확성을 향상시키고 있습니다.

- **Technical Details**: Air-DualODE는 물리적 동역학과 데이터 기반 동역학을 통합하는 하이브리드 모델입니다. 첫 번째 가지는 오픈 시스템의 물리적 방정식을 사용하여 시공간(spatiotemporal) 종속성을 포착하며, 두 번째 가지는 데이터 중심의 방식으로 생긴 종속성을 식별합니다. 이로써 각 가지의 시공간 표현을 시간적으로 정렬하여 융합합니다.

- **Performance Highlights**: 실험 결과, Air-DualODE는 여러 공간적 스케일에서 오염물 농도를 예측하는 데 있어 최신 성능을 달성하였으며, 이를 통해 실제 공기 질 문제에 대한 유망한 솔루션을 제공합니다.



### EnergyPlus Room Simulator (https://arxiv.org/abs/2410.19888)
Comments:
          Presented at BuildSim Nordic 2024. The conference was held from June 9 to 11, 2024, in Espoo, Finland

- **What's New**: 이 논문에서는 EnergyPlus Room Simulator라는 도구를 소개합니다. 이 도구는 EnergyPlus 시뮬레이션 소프트웨어를 사용하여 특정 건물의 실내 기후를 시뮬레이션할 수 있게 해줍니다.

- **Technical Details**: EnergyPlus Room Simulator는 온도, 습도 및 CO2 농도와 같은 다양한 요소를 시뮬레이션하며, 사용자 친화적인 그래픽 사용자 인터페이스(GUI)와 REST API를 제공합니다. 이를 통해 사용자는 건물 모델을 수정하고 복잡한 시뮬레이션을 보다 쉽게 수행할 수 있습니다.

- **Performance Highlights**: 이 도구는 데이터 집약적인 딥 러닝(deep learning) 방법에 필요한 대량의 데이터를 생성하는 데 유용하며, 빠르게 시뮬레이션 데이터에 접근할 수 있도록 지원하여 과학적 및 건물 관련 작업을 지원합니다.



### TBBC: Predict True Bacteraemia in Blood Cultures via Deep Learning (https://arxiv.org/abs/2410.19887)
Comments:
          12 pages

- **What's New**: 이 논문에서는 혈류 감염(bacteraemia)의 진단을 위한 머신러닝 모델을 개발하여 응급실에서의 진단 효율성을 향상시키려는 새로운 접근 방식을 제시합니다.

- **Technical Details**: St. Antonius Hospital의 응급실 데이터를 이용하여 향상된 머신러닝 기법인 CatBoost와 Random Forest를 선택하고, Optuna를 이용한 모델 최적화를 통해 Random Forest 모델의 ROC AUC 지표 0.78을 달성하였습니다. 이 모델은 테스트 세트에서 0.92의 민감도를 나타냈습니다.

- **Performance Highlights**: 이 모델은 저위험(low risk) bacteraemia 환자의 36.02%를 정확하게 식별할 수 있었고, 오탐률(false positive rate)은 0.85%로 매우 낮았습니다. 이 연구의 결과는 혈액 배양(blood cultures), 의료 비용, 항생제 치료의 감소 가능성을 보여줍니다.



### Recommendations for Comprehensive and Independent Evaluation of Machine Learning-Based Earth System Models (https://arxiv.org/abs/2410.19882)
- **What's New**: 이번 논문은 기계 학습(Machine Learning, ML)이 지구 과학 분야에서 날씨 예측을 넘어서 지구 시스템 모델(Earth System Models, ESMs)의 개발로 확장될 가능성을 다루고 있습니다.

- **Technical Details**: ML 기반의 ESM은 결합된 지구 시스템(coupled Earth system)의 모든 구성 요소를 표현할 수 있는 모델로, 다양한 시간 척도에서의 예측을 향상시키기 위한 접근이 필요합니다. 특히, 이 모델들은 과거 관측 데이터가 없는 대체 결합 상태를 예측해야 하므로 그 복잡성이 증가합니다.

- **Performance Highlights**: 저자들은 ML 기반 ESM의 신뢰성을 강화하고 더 널리 사용될 수 있도록 독립적이고 체계적인 평가를 위한 다섯 가지 권장을 제안합니다.



### Causal Order Discovery based on Monotonic SCMs (https://arxiv.org/abs/2410.19870)
Comments:
          Accepted to the NeurIPS 2024 Workshop on Causal Representation Learning

- **What's New**: 이 논문에서는 Monotonic Structural Causal Models (SCMs)에서의 인과 순서 발견 문제를 다루고 있습니다. 기존 접근 방식이 인과 순서에 대한 사전 지식을 필요로 하거나 복잡한 최적화 기법을 사용해야 했던 반면, 본 연구에서는 새로운 순차적 절차를 도입하여 반복적으로 root variable을 식별함으로써 인과 순서를 직접적으로 구하는 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 Jacobian의 zeroness를 활용하여 root variable을 발견하는 순차적 접근 방식입니다. 이는 기존의 정의된 sparsity 가정을 제거하고, Markov 동치 클래스(Markov Equivalent Class)를 변별할 수 있는 진정한 SCMs의 인식을 가능하게 합니다. 이 방식은 TMI maps와 Jacobian 개념을 활용하여 학습 과정을 단순화합니다.

- **Performance Highlights**: 연구 결과, 제안된 방법이 root 변수를 순차적으로 발견하는 데 있어 효율성을 보여주는 실험들이 수행되었습니다. 기존의 Jacobian sparsity를 극대화하는 방법들과 비교했을 때, 이 방법이 더 효율적임을 입증하였습니다.



### Hypergraph Neural Networks Reveal Spatial Domains from Single-cell Transcriptomics Data (https://arxiv.org/abs/2410.19868)
- **What's New**: 이번 연구는 Hypergraph Neural Networks (HGNNs)를 활용하여 spatial transcriptomics data의 클러스터링 문제를 해결하는 새로운 접근법을 제시합니다. 기존 Graph Neural Networks (GNNs)의 한계를 극복하고 더 복잡한 셀 간의 관계를 캡처할 수 있는 가능성을 보여주고 있습니다.

- **Technical Details**: HGNNs는 hyperedge를 사용하여 다수의 노드를 연결하며, 이는 GNNs보다 더 풍부한 구조적 정보를 활용할 수 있습니다. 저자들은 unsupervised learning을 위해 autoencoders를 사용하고, 이를 통해 라벨이 없는 데이터의 패턴을 탐지합니다. 모델은 실험을 통해 높은 iLISI 점수(1.843)를 기록하며, 이는 다양한 세포 유형을 식별할 수 있음을 나타냅니다.

- **Performance Highlights**: 모델은 downstream clustering에서도 뛰어난 성능을 보였으며, Adjusted Rand Index (ARI) 값 0.51과 Leiden score 0.60을 달성하여 기존 방법들보다 우수한 결과를 보여주었습니다.



### Simultaneous Dimensionality Reduction for Extracting Useful Representations of Large Empirical Multimodal Datasets (https://arxiv.org/abs/2410.19867)
Comments:
          PhD Dissertation, available at Emory EDT @ this https URL

- **What's New**: 이 논문은 고차원 데이터로부터 저차원 기술을 얻기 위한 차원 축소(dimensionality reduction) 개념에 초점을 맞추고 있습니다. 이 연구는 이론물리학 및 머신러닝(machine learning)에서의 통찰력을 기반으로 다루어지는 여러 축소 방법을 하나의 포괄적 프레임워크인 Deep Variational Multivariate Information Bottleneck 아래 통합합니다.

- **Technical Details**: 이 프레임워크는 특정 연구 질문에 따라 맞춤형 축소 알고리즘을 설계할 수 있게 해주고, 독립적인 축소 방법에 비해 여러 모달리티 간의 공동 변이(covariation)를 캡처하는 데 있어 우수성을 보여줍니다. 동시 축소 방법의 효용을 탐구하고, 새로운 기법인 Deep Variational Symmetric Information Bottleneck을 도입하여 비선형 동시 축소를 위한 방법론을 개발했습니다.

- **Performance Highlights**: 새로운 방법론이 복잡한 데이터셋에서 의미 있는 통찰력을 추출할 수 있는 잠재력을 강조하며, 다양한 분야에서 데이터 분석의 개선을 가능하게 할 것입니다. 이 연구는 복잡한 시스템에 대한 이해를 심화시키고 효과적인 데이터 분석 전략을 안내하는 데 기여할 것으로 기대됩니다.



### Evaluating Deep Learning Approaches for Predictions in Unmonitored Basins with Continental-scale Stream Temperature Models (https://arxiv.org/abs/2410.19865)
Comments:
          47 pages, 12 figures, 7 tables, submitted to Water Resources Research

- **What's New**: 이 연구는 미국의 비모니터링 구역에서 머신러닝(ML) 모델을 이용하여 하천 수온 예측의 정확성을 보여주며, 데이터 기반의 "top-down" 접근 방식과 데이터가 적은 지역에서 모델을 전이하는 "bottom-up" 접근 방식을 비교합니다.

- **Technical Details**: 연구에서는 지역적 특성을 기준으로 사이트를 분류하는 "grouped" 모델링 기법을 평가하며, 입력 변수를 체계적으로 제거하면서 모델 복잡성, 예측 정확성, 적용 가능성의 균형을 살펴봅니다.

- **Performance Highlights**: 결과적으로, top-down 모델은 bottom-up 및 grouped 모델보다 우수하며, 동적 및 정적 입력 변수를 줄임으로써 더 많은 사이트에 대해 낮은 모델 복잡성과 계산 요구를 통해 허용 가능한 정확성을 도출할 수 있음을 보여줍니다.



### Enhancing Deep Learning based RMT Data Inversion using Gaussian Random Field (https://arxiv.org/abs/2410.19858)
- **What's New**: 이번 연구는 지구물리 데이터의 역산을 위한 딥러닝(DL) 기반의 새롭고 혁신적인 접근법을 제시합니다. Gaussian Random Fields (GRF)를 사용하여 다양한 저항률 모델을 생성하고, 이를 통한 일반화 가능성을 테스트하였습니다.

- **Technical Details**: Cross-domain (OOD) 데이터셋을 통해 모델의 일반화 능력을 평가하였고, GRF 데이터셋으로 훈련받은 네트워크는 OOD 데이터셋에서의 이상 현상을 성공적으로 탐지하였습니다. 이 연구는 전통적인 기울기 기반 방법에 비해 노이즈에 대한 견고성을 보이며, 백그라운드 OOD 데이터셋에 비해 일반화가 향상되었습니다.

- **Performance Highlights**: 제안된 방법은 인도 루어키 근처의 폐기물 사이트에서의 필드 데이터 실험을 통해 검증되었으며, 다양한 환경에서의 일반화 능력이 뛰어난 것으로 나타났습니다. 해당 방법은 데이터 기반 감독 학습(framework)에서 높은 이해력을 보여주며, DL 방법의 OOD 일반화에 대한 유망한 방향성을 제시합니다.



### Prototype-Based Methods in Explainable AI and Emerging Opportunities in the Geosciences (https://arxiv.org/abs/2410.19856)
Comments:
          Accepted at AI for Science Workshop-Oral (Attention Track), Proceedings of 41st International Conference on Machine Learning (ICML) 2024

- **What's New**: 이 논문은 프로토타입 기반의 설명 가능한 인공지능(XAI) 기술 개발을 다루며, 특히 지구 과학 분야에서의 사용 가능성을 탐구합니다. 논자는 프로토타입 생성과 시각화, 다양한 종류의 프로토타입 및 다양한 학습 과제에서의 프로토타입 사용을 포함하여 프로토타입 기반 XAI 문헌을 세 가지 주제로 조직합니다.

- **Technical Details**: 프로토타입 기반 XAI 모델 아키텍처는 표준 신경망 층(예: convolution, recurrent, dense)으로 구성되어 있으며, 각 클래스와 관련된 잠재적 프로토타입을 생성하는 프로토타입 층과 입력과 프로토타입 간의 유사도(metric)를 계산하는 기능이 포함되어 있습니다. 이는 특히 인간이 이해할 수 있는 방식으로 설명을 제공하기 위해 설계된 손실 함수(optimization procedure)를 포함합니다.

- **Performance Highlights**: 프로토타입 기반 XAI 방식은 과거의 사례와 입력 데이터를 비교함으로써 설명 가능성을 향상시킵니다. 이 접근 방식은 예측 결과에 대한 설명을 모델의 의사 결정 과정에 내재화할 수 있는 장점을 가집니다. 이는 지구 과학 분야의 데이터 분석에 있어 신뢰성과 유용성을 높일 것으로 기대됩니다.



### Survival of the Fittest: Evolutionary Adaptation of Policies for Environmental Shifts (https://arxiv.org/abs/2410.19852)
Comments:
          Pubblished in ECAI 2024

- **What's New**: 본 논문은 새로운 정책 최적화 기법인 진화적 강인 정책 최적화(Evolutionary Robust Policy Optimization, ERPO)를 제안하여 RL(Robust Learning) 알고리즘의 한계를 극복하고자 합니다. 이 방법은 진화 게임 이론(Evolutionary Game Theory)에서 영감을 받아 환경 변동에 유연하게 대처합니다.

- **Technical Details**: ERPO는 환경에서의 최적 정책을 반복적으로 학습하며, 이는 탐색(Exploration)과 이전 최적 정책에 대한 이행(Adherence) 간의 균형을 조절하는 온도 매개변수(Temperature Parameter)를 사용하는 적응형 재훈련 알고리즘입니다. 정책 업데이트는 진화 게임 이론에서 사용되는 복제 동역학(Replicator Dynamics)의 구체적인 사례입니다. 이 알고리즘은 보상의 희소성(Sparsity Assumptions)에 대한 일반적인 가정을 가지고 있으며, 변동된 환경에서 최적 정책으로 수렴함을 보여줍니다.

- **Performance Highlights**: ERPO는 경로 탐색(Path Finding) 작업에서 여러 RL 및 Deep RL 알고리즘(PPO, A3C, DQN)에 비해 여러 시나리오와 인기 있는 환경에서 우수한 성능을 보여주었습니다. ERPO는 정책 적응(Policy Adaptation) 속도가 빠르고, 평균 보상이 높으며, 정책 적응에 드는 계산 비용을 줄였습니다.



### Deep Learning and Machine Learning -- Python Data Structures and Mathematics Fundamental: From Theory to Practic (https://arxiv.org/abs/2410.19849)
Comments:
          298 pages

- **What's New**: 이 책은 기계 학습(Machine Learning, ML)과 딥 러닝(Deep Learning, DL)의 기초 개념에 대한 종합적인 소개를 제공합니다. 이론적 수학과 실용 응용의 간극을 연결하며, Python을 주요 프로그래밍 언어로 사용하여 핵심 알고리즘과 데이터 구조를 구현합니다.

- **Technical Details**: 책은 기본 및 고급 Python 프로그래밍, 기본 수학 연산, 행렬 연산, 선형 대수, ML 및 DL 모델 훈련에 중요한 최적화 기법 등을 포함한 다양한 주제를 다룹니다. 신경망(neural networks), 최적화 알고리즘(optimization algorithms), 주파수 도메인(frequency domain) 방법 등 고급 주제와 대규모 언어 모델(large language models, LLMs)의 실제 응용이 탐구됩니다.

- **Performance Highlights**: 초보자와 고급 학습자를 모두 위한 설계로, 수학적 원칙이 확장 가능한 AI 솔루션 개발에 미치는 중요한 역할을 강조합니다. 이론적 지식을 복잡한 ML, DL 및 빅데이터 분석 문제를 해결하는 데 적용할 수 있는 실제 예제와 Python 코드를 제공합니다.



### Multidimensional Knowledge Graph Embeddings for International Trade Flow Analysis (https://arxiv.org/abs/2410.19835)
- **What's New**: 이 연구는 고차원성, 우발성 및 강한 비선형성을 특징으로 하는 경제 데이터의 복잡한 역학을 이해하기 위한 새로운 접근 방식을 제시합니다. 특히 KonecoKG라는 지식 그래프 임베딩 모델을 활용하여 국제 무역 관계를 예측하고 이를 통해 무역 흐름을 최적화하는 방법을 탐구합니다.

- **Technical Details**: KonecoKG는 다차원 관계를 가진 경제 무역 데이터의 지식 그래프 표현으로, SDM-RDFizer를 사용하여 관계를 변환하고 AmpliGraph를 통해 지식 그래프 임베딩을 생성합니다. 연구는 경제 상호작용을 네트워크 구조 내에서 표현하며, 무역 네트워크 데이터셋을 활용하여 역사적 무역 패턴과 인접국의 무역 행동 통찰을 통합하여 미래 무역 기회를 예측하고자 합니다.

- **Performance Highlights**: 이 연구는 무역 흐름을 예측하기 위해 다차원 관계를 모델링하여 비선형성 문제를 해결하며, KonecoTradeFlow 온톨로지를 도입하여 국제 경제 양자 간 무역 데이터를 표현합니다. 결과적으로, 정책 입안자, 기업 및 투자자에게 국제 무역 전략을 최적화할 수 있는 귀중한 통찰을 제공하며 경제 성장 추세를 발견하는 데 기여합니다.



### GNNRL-Smoothing: A Prior-Free Reinforcement Learning Model for Mesh Smoothing (https://arxiv.org/abs/2410.19834)
- **What's New**: 본 논문에서는 기존의 지도 학습(supervised learning) 및 강화 학습(reinforcement learning)에 의존하는 동적 메시 스무딩 기법을 넘어, 데이터나 사전 지식 없이 학습할 수 있는 새로운 강화 학습 모델을 제안합니다. 이는 메시 최적화를 마르코프 결정 과정(Markov Decision Process)으로 형식화하여, 총 두 개의 에이전트(agents)를 훈련시킵니다.

- **Technical Details**: 제안된 모델은 그래프 신경망(Graph Neural Networks)과 강화 학습을 결합하여 똑똑한 노드 스무딩 에이전트(intelligent node smoothing agent)와 메시 연결성 개선 에이전트(mesh connectivity improvement agent)를 구현합니다. 이 두 에이전트는 Twin Delayed Deep Deterministic Policy Gradient와 Double Dueling Deep Q-Network를 사용하여 훈련됩니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 복잡한 3D 표면 메시에서 피처를 보존하는 스무딩을 달성했으며, 2D 메시에 대해서는 동적 스무딩 방법 중에서 최신 기술 대비 7.16배 빠른 결과를 보여주었습니다. 또한 연결성 개선 에이전트는 메시 품질 분포를 효과적으로 향상시킬 수 있음을 입증했습니다.



### Novel Development of LLM Driven mCODE Data Model for Improved Clinical Trial Matching to Enable Standardization and Interoperability in Oncology Research (https://arxiv.org/abs/2410.19826)
Comments:
          18 pages, 13 figures, accessible and published at: The Young Researcher Fall 2024 Volume 8, Number 2(Special Edition in Collaboration with Harvard Undergraduate Openbio Laboratory); Pages 28-45

- **What's New**: 이 연구에서는 암 치료에서 데이터 표준화와 상호운용성(interoperability)의 부족이 진단의 적시성과 효율성을 저해하는 문제를 해결하기 위해 새로운 프레임워크를 제시합니다. 이 프레임워크는 전통적인 임상 시험 등록 및 치료 방법에 의존하는 것을 넘어, 데이터 중심(data-driven) 접근 방식을 활용하여 EHR(전자 건강 기록)의 통합을 촉진합니다.

- **Technical Details**: 본 연구에서는 FHIR(Fast Healthcare Interoperability Resources)의 리소스 기반 접근 방식과 LLM(대형 언어 모델)이 생성한 mCODE 프로파일을 활용하여 환자 정보를 정확하고 신속하게 공유합니다. 이를 위해 비구조화(unstructured)된 환자 치료 데이터, PDF, 자유형 텍스트 정보 및 진행 노트를 강화된 mCODE 프로파일로 변환하고, 이를 바탕으로 AI 및 ML(기계 학습) 기반의 임상 시험 매칭 엔진과의 원활한 통합을 도모합니다.

- **Performance Highlights**: 연구 결과는 데이터 표준화에서 유의미한 개선을 보여 주며, 훈련된 LLM의 정확도는 92% 이상에 도달했습니다. 또한, SNOMED-CT에 대해서는 87%, LOINC에 대해서는 90%, RxNorm 코드에 대해서는 84%의 정확도를 보였습니다. 이는 현재의 LLM(GPT-4 및 Claude 3.5 포함) 평균 77%를 초월하는 성과입니다. 본 연구는 표준화와 상호운용성의 잠재력을 강조하며, 보다 효율적이고 개인화된 암 치료의 길잡이가 될 것입니다.



### Deep Learning-driven Mobile Traffic Measurement Collection and Analysis (https://arxiv.org/abs/2410.19777)
Comments:
          MPhil thesis

- **What's New**: 본 논문은 기존의 연구들이 간과했던 이동 기지국 간의 동적 의존성을 정확하게 모델링하고, 효과적인 모바일 트래픽 분석 및 예측을 위한 딥러닝 기반의 방법론을 제안합니다.

- **Technical Details**: 우선 'Spider'라는 모바일 트래픽 측정 수집 및 재구성 프레임워크를 설계하여 측정 비용을 줄이고 높은 정확도로 트래픽 소비를 추론합니다. 긴밀한 조정을 통해 강화 학습( reinforcement learning ) 기법을 이용하여 목표 모바일 커버리지 영역의 하위 집합을 선택적으로 샘플링하는 에이전트를 훈련합니다. 이후 경량화된 신경망 모델을 통해 역사적 희소 데이터 기반으로 트래픽 소비를 재구성합니다. 또한 'SDGNet'이라는 핸드오버 인식 그래프 신경망 모델을 설계하여 시간에 따라 기지국 간의 의존성을 캡처합니다. 핸드오버 빈도 정보를 통해 사용자 이동성을 반영하여 예측의 정확성을 향상시킵니다. 이러한 방법들은 실제 모바일 트래픽 데이터셋에서 다른 벤치마크 모델들을 능가합니다.

- **Performance Highlights**: 'Spider' 프레임워크는 실제 모바일 트래픽 데이터셋에서 기존 솔루션을 초과하는 성능을 보이며, 'SDGNet' 모델은 주요 네트워크 운영자가 수집한 모바일 트래픽 데이터셋에서 다른 그래프 모델에 비해 뛰어난 성능을 입증합니다.



### A New Perspective to Boost Performance Fairness for Medical Federated Learning (https://arxiv.org/abs/2410.19765)
Comments:
          11 pages, 2 Figures

- **What's New**: 이 연구에서는 의료 애플리케이션에서의 도메인 쉬프트(domain shift) 문제를 해결하기 위해, 레이어별 가중치를 동적으로 조정하여 연합 학습(federated learning, FL)의 성능 공정성을 향상시키는 Fed-LWR 방법을 제안합니다.

- **Technical Details**: Fed-LWR는 레이어별로 클라이언트의 특징 표현(feature representation) 차이를 추정하여, 더욱 공정한 글로벌 모델을 생성하기 위해 사용됩니다. 이 과정에서 Centered Kernel Alignment (CKA) 기술을 이용하여 각 병원의 보다 정확한 가중치를 산출합니다.

- **Performance Highlights**: 제안된 Fed-LWR 방법은 두 개의 의료 이미지 분할 벤치마크에서 평가되었으며, 여러 최신 공정 FL 방법들과 비교하여 더 나은 공정성과 정확도 간의 트레이드 오프(trade-off)를 실현했습니다.



### GPT-4o System Card (https://arxiv.org/abs/2410.21276)
- **What's New**: GPT-4o는 텍스트, 오디오, 이미지, 비디오 등 다양한 입력을 지원하는 자가 회귀형 omni 모델로, 통합된 신경망을 통해 텍스트, 오디오, 이미지 출력을 생성합니다. 이 모델은 대화에서 인간의 반응 시간과 유사하게 평균 320밀리초로 오디오 입력에 응답할 수 있어 향상된 반응성을 보여줍니다.

- **Technical Details**: GPT-4o는 텍스트와 음성 기능이 2023년 10월까지의 다양한 자료를 바탕으로 사전 훈련되었습니다. 웹 데이터, 코드 및 수학, 다중 모달 데이터가 포함되어 있어 모델이 비텍스트 입력과 출력을 해석하고 생성하는 방법을 학습하게 됩니다. 또한, 위험을 줄이기 위한 방법으로는 Moderation API와 안전 분류기를 사용하여 유해 콘텐츠를 필터링합니다.

- **Performance Highlights**: GPT-4o는 영어 및 코드 텍스트에서 GPT-4 Turbo의 성능을 맞추며 비영어 텍스트에서는 현저한 개선을 보여줍니다. API 비용은 50% 절감되며, 비전 및 오디오 이해 능력이 기존 모델보다 뛰어난 것으로 확인되었습니다.



### Adaptive Transfer Clustering: A Unified Framework (https://arxiv.org/abs/2410.21263)
Comments:
          52 pages, 8 figures

- **What's New**: 이 연구는 동일한 주제에 대한 주요 데이터셋(main dataset)과 보조 데이터셋(auxiliary dataset)을 활용하여 클러스터링(clustering) 문제를 해결하기 위한 일반적인 전이 학습(framework for transfer learning) 프레임워크를 제안합니다.

- **Technical Details**: 제안된 방법은 적응형 전이 클러스터링(adaptive transfer clustering, ATC) 알고리즘으로, 알려지지 않은 불일치(unknown discrepancy)가 존재할 때 공통성을 자동으로 활용하여 추정된 편향-분산 분해(bias-variance decomposition)를 최적화합니다. 이 방법은 Gaussian mixture models, stochastic block models, latent class models 등 다양한 통계 모델에 적용될 수 있습니다.

- **Performance Highlights**: 이론 분석을 통해 Gaussian mixture model에서 ATC의 최적성을 증명하고, 전이의 이점을 명시적으로 정량화합니다. 광범위한 시뮬레이션 및 실제 데이터 실험을 통해 제안된 방법의 다양한 상황에서의 효과성을 확인하였습니다.



### Quantum computing and persistence in topological data analysis (https://arxiv.org/abs/2410.21258)
Comments:
          21 pages

- **What's New**: 이 논문은 topological data analysis (TDA)에서의 새로운 문제인 Harmonic Persistence를 도입하고, 이 문제의 계산 복잡성을 분석합니다. 이 연구는 기존의 방식이 가지는 두 가지 주요 한계를 극복하고, quantum algorithms를 통해 하모닉 홀(harmonic hole)의 지속성을 판별하는 문제를 해결합니다.

- **Technical Details**: Harmonic Persistence 문제는 주어진 simplicial complex K1에서 k차원 홀의 하모닉 표현이 K2에서 지속되는지를 판단하는 것으로, 𝖡𝖰𝖯1-hard 및 𝖡𝖰𝖯에 포함됩니다. 이 연구는 guided sparse Hamiltonian 문제의 변형을 활용하여 홀의 지속성을 인코딩합니다.

- **Performance Highlights**: 이 결과는 TDA 분야에서의 quantum 속도의 기하급수적 개선을 나타내며, 이는 기존의 classical 알고리즘이 필요로 했던 지수적인 시간 복잡도에 비해 놀랍도록 효율적인 접근 방식을 제공합니다.



### One-Step Diffusion Policy: Fast Visuomotor Policies via Diffusion Distillation (https://arxiv.org/abs/2410.21257)
- **What's New**: 이번 논문에서는 사전 훈련된 diffusion policy에서 지식을 증류하여 단일 단계 액션 생성기로 변환하는 One-Step Diffusion Policy (OneDP)를 소개합니다. 이는 로봇 제어 작업의 응답 시간을 대폭 단축하여 실시간 애플리케이션에서의 활용 가능성을 높입니다.

- **Technical Details**: OneDP는 사전 훈련된 diffusion 정책으로부터 지식을 증류하여 단일 단계 기반의 액션 생성기를 생성합니다. 이 과정에서는 Kullback-Leibler (KL) divergence를 최소화하여 생성된 액션의 분포가 원래 정책과 일치하도록 하는 방법을 사용합니다. OneDP는 단일 신경망 피드포워드 작업을 통해 최대의 추론 효율성을 달성합니다.

- **Performance Highlights**: OneDP는 6개의 도전적인 시뮬레이션 작업과 4개의 실제 디자인된 작업에서 평가되었으며, 이 결과는 OneDP가 최신 기술 수준의 성공률을 달성함과 동시에 추론 속도를 20배 이상 향상시켰고, 1.5 Hz에서 62 Hz로 액션 예측 빈도를 증가시켰음을 보여줍니다.



### LongReward: Improving Long-context Large Language Models with AI Feedback (https://arxiv.org/abs/2410.21252)
- **What's New**: 이 논문에서는 LongReward라는 새로운 방법을 제안하여 LLM(대형 언어 모델)의 긴 컨텍스트 응답에 대한 보상을 제공하는 방식을 다룹니다. LongReward는 도움성(helpfulness), 논리성(logicality), 충실성(faithfulness), 완전성(completeness)이라는 네 가지 측면을 기반으로 보상을 평가합니다.

- **Technical Details**: LongReward는 오프더셀프 LLM을 사용하여 긴 컨텍스트 모델 응답에 대한 보상을 제공합니다. 각 기준에 대해 0부터 10까지 점수를 부여하고 평균 점수를 최종 보상으로 사용합니다. 이 방법과 DPO(Direct Preference Optimization)와 같은 강화학습(RL) 알고리즘을 결합하여 긴 컨텍스트 SFT 모델의 성능을 향상시킵니다.

- **Performance Highlights**: LongReward를 적용한 결과, Llama-3.1-8B와 GLM-4-9B 모델에서 긴 문맥 작업에 대해 각각 4.9% 및 5.5%의 성능 향상을 보였으며, 인간 평가에서도 SFT 기준에 대해 46% 더 많은 승리를 기록했습니다. 또한, LongReward는 단기 지침에 대한 모델의 수행 능력을 높이는 데에도 기여합니다.



### Zero-Shot Dense Retrieval with Embeddings from Relevance Feedback (https://arxiv.org/abs/2410.21242)
- **What's New**: 이 논문에서는 ReDE-RF(Real Document Embeddings from Relevance Feedback)라는 새로운 접근 방식을 도입하여, LLM을 활용한 가상의 문서 생성 대신 관련성 추정(task)을 프레임으로 삼아 검색 효율성을 크게 향상시킵니다.

- **Technical Details**: ReDE-RF는 먼저 완전 비지도 하이브리드 스파스-밀집 검색 시스템에서 초기 문서 집합을 추출한 후, LLM을 통해 반환된 문서가 관련성이 있는지 없는지를 평가합니다. 이후, 관련 문서 집합을 기반으로 미리 계산된 문서 임베딩을 효율적으로 가져와 쿼리 벡터를 업데이트합니다. 이는 LLM이 도메인 특화 지식에 의존하지 않고 단순히 관련성을 판단하게 합니다.

- **Performance Highlights**: 실험 결과, ReDE-RF는 여러 저자원 검색 데이터셋에서 기존의 최첨단 제로샷 밀집 검색 방법들보다 최대 14% 더 우수한 성능을 보였으며, 검색 지연(lag)을 7.5배에서 11.2배까지 줄이는 성과를 나타냈습니다.



### Vision Search Assistant: Empower Vision-Language Models as Multimodal Search Engines (https://arxiv.org/abs/2410.21220)
Comments:
          Code is available at this https URL

- **What's New**: 이 논문에서는 기존 대규모 비전-언어 모델(Vision-Language Models, VLM)이 처리하지 못하는 새로운 시각 정보를 이해하고 응답할 수 있는 'Vision Search Assistant'라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 VLM과 웹 에이전트(web agent)의 협업을 통해 실시간 정보를 검색하고 새로운 객체에 대한 질문을 다룰 수 있도록 합니다.

- **Technical Details**: Vision Search Assistant는 VLM이 질문을 이해하고, 이미지의 특정 객체를 분석하며, 관련된 정보와 텍스트를 검색하는 과정을 포함합니다. 이 과정은 크게 세 단계로 나뉘며, 첫째로 크리티컬 비주얼 객체의 시각적 내용을 텍스트 형태로 서술하는 'Visual Content Formulation', 둘째로 사용자의 질문과 관련된 서브 질문을 생성하여 웹 지식을 검색하는 'Web Knowledge Search', 셋째로 최종 답변을 생성하는 'Collaborative Generation'을 포함합니다.

- **Performance Highlights**: 다양한 오픈세트(open-set) 및 클로즈드세트(closed-set) QA 벤치마크에서 extensive experiments를 실시한 결과, Vision Search Assistant는 기존 모델보다 훨씬 높은 성능을 보이며, 대규모 비전-언어 모델에 효과적으로 적용될 수 있음을 입증했습니다.



### HoPE: A Novel Positional Encoding Without Long-Term Decay for Enhanced Context Awareness and Extrapolation (https://arxiv.org/abs/2410.21216)
- **What's New**: 본 논문에서는 기존 Positional Encoding (PE)의 긴 거리 토큰에 대한 규칙인 'long-term decay'의 필요성을 반박하고, 이를 대체할 새로운 방법인 High-frequency rotary Position Encoding (HoPE)를 제안합니다. HoPE는 Rotary Position Encoding (RoPE)의 한계를 해결하고 모델의 맥락 인지와 추론 능력을 향상시킵니다.

- **Technical Details**: 기존의 PE 모델은 긴 거리 토큰이 모델의 정보 처리에 미치는 영향을 줄이는 long-term decay 원칙을 따르지만, 실험 결과 LLMs에서 모델들은 단기적인 정보에만 집중하는 경향을 보입니다. HoPE는 RoPE의 구성 요소 중 위치 독립적인 것과 높은 주파수의 신호로 대체하여 모델의 전반적인 성능을 향상시킵니다.

- **Performance Highlights**: HoPE는 언어 모델링의 perplexity, 복사 작업, 그리고 few-shot 학습 작업에서 기존의 PE 방식보다 우수한 성능을 보여주었습니다. 이 연구는 특정 주파수의 'activated' 구성 요소가 RoPE의 성능 한계를 유발했음을 보여줍니다.



### On learning higher-order cumulants in diffusion models (https://arxiv.org/abs/2410.21212)
Comments:
          21 pages, many figures. Extended version of contribution accepted in the NeurIPS 2024 workshop "Machine Learning and the Physical Sciences"

- **What's New**: 이 논문에서는 diffusion model이 Gaussian 이상의 상관관계를 학습하는 방식을 분석합니다. Forward 및 backward 과정에서 고차 cumulant(누적모멘트)의 행동을 연구하며, 이들 간의 상관관계의 보존을 증명합니다.

- **Technical Details**: 고차 cumulant는 drift가 없는 모델에서 보존되며, 이는 variance-expanding scheme에서 발견됩니다. 이 결과는 일반적인 Gaussian 분포에서 시작하는 경우에도 backward 과정에서 학습됩니다. lattice field theory와 perturbation theory를 활용하여 모델의 성능을 평가합니다.

- **Performance Highlights**: 이론적인 분석은 구체적인 toy model과 scalar lattice field theory를 통해 검증되었습니다. 결과적으로 diffusion models는 고차 n-point 기능을 효과적으로 학습할 수 있는 가능성을 보여주며, lattice field configurations를 생성하는 데 적용할 수 있는 잠재력이 확인되었습니다.



### BongLLaMA: LLaMA for Bangla Languag (https://arxiv.org/abs/2410.21200)
Comments:
          19 pages

- **What's New**: BongLLaMA(즉, Bangla-LLaMA)라는 새로운 오픈 소스 대형 언어 모델이 발표되었습니다. 이는 대규모 방글라어 코퍼스를 기반으로 독점적으로 파인 튜닝(fine-tuning)된 모델입니다.

- **Technical Details**: 본 연구에서는 방글라어 데이터 증강(data augmentation) 기법, 파인 튜닝 세부 사항, 그리고 BongLLaMA 모델의 효과성을 입증하는 종합적인 벤치마킹(benchmarking) 결과를 제시합니다. 이 모델은 방글라어 언어 처리(BLP) 작업을 위한 최적화된 성능을 자랑합니다.

- **Performance Highlights**: BongLLaMA는 방글라어 언어 모델의 새로운 기준이 될 것으로 예상되며, 향후 방글라어의 다양한 언어 처리 연구에 유용한 기초 자료를 제공합니다. 모든 BongLLaMA 모델은 공공용으로 제공됩니다.



### SoS Certifiability of Subgaussian Distributions and its Algorithmic Applications (https://arxiv.org/abs/2410.21194)
- **What's New**: 이 논문에서는 모든 중심 서브가우시안 분포가 SoS-certifiably 서브가우시안임을 입증하였습니다. 이 조건은 고차원 통계 작업에 대한 효율적인 학습 알고리즘을 제공하는 품질 보증을 의미합니다.

- **Technical Details**: 논문에서 제시된 중심 서브가우시안 분포를 이용한 $d$-변량 다항식 $(Cp)^{p/2} \\|v\\|_{2}^p - \mathbb E_{X \sim \mathcal D} \langle v,X\rangle^p$이 다항식의 제곱의 합(Sum of Squares, SoS)으로 표현되는 것을 보여줍니다. 이 과정에서 Talagrand의 일반 체인/주요화 측정 정리를 사용합니다.

- **Performance Highlights**: 이 논문은 서브가우시안 분포의 샘플을 통해 robust mean estimation, list-decodable mean estimation, clustering mean-separated mixture models, robust covariance-aware mean estimation, robust covariance estimation, robust linear regression과 같은 여러 작업에 대해 근사 최적 보장을 갖춘 계산적으로 효율적인 알고리즘을 제공합니다.



### On Homomorphic Encryption Based Strategies for Class Imbalance in Federated Learning (https://arxiv.org/abs/2410.21192)
Comments:
          Accepted for Presentation at CODS COMAD 2024

- **What's New**: 이 논문에서는 Federated Learning(FL) 환경에서의 클래스 불균형 문제를 해결하기 위한 새로운 프레임워크인 FLICKER를 제안합니다. FLICKER는 CKKS 동형 암호화 기술을 사용하여 데이터의 개인 정보를 보호하면서 글로벌 클래스 불균형을 해결합니다.

- **Technical Details**: FLICKER는 세 가지 주요 단계를 포함합니다: 글로벌 불균형의 보안 계산을 위한 개인 정보 보호 단계, 글로벌 불균형에 기여한 클라이언트를 식별하는 로컬화 단계, 특정 클라이언트에서 불균형을 수정하는 단계입니다. 이 과정은 목표한 글로벌 불균형이 달성될 때까지 반복됩니다.

- **Performance Highlights**: FLICKER는 이미지 기반 CIFAR-10 및 자연어 처리 기반 AG News 데이터셋을 사용하여 실험을 진행하였으며, 기존의 지역적 불균형 수정 방법보다 정상화된 정확도(normalised accuracy)와 F1-스코어에서 유의미하게 향상된 결과를 보였습니다.



### Differentially Private Learned Indexes (https://arxiv.org/abs/2410.21164)
- **What's New**: 이 논문에서는 Trusted Execution Environments (TEEs)에서 안전하게 보호되는 암호화된 데이터베이스에서 predicate 쿼리를 효율적으로 처리하기 위한 방법을 제안합니다. 특히, 최근 주목받고 있는 학습 인덱스(learned indexes) 기술을 활용하여 더욱 Compact하고 Privacy를 보장하는 DP 인덱스를 개발하는 것에 초점을 맞추고 있습니다.

- **Technical Details**: TEEs는 클라우드에서 암호화된 데이터를 처리할 수 있도록 안전한 하드웨어 환경을 제공합니다. 데이터에 대한 쿼리는 복잡한 인덱스 없이 암호화된 데이터 전부를 TEEs로 로드해야 했습니다. 기존 DP 인덱스는 노이즈를 통해 인덱스를 보호하지만, 저장 비용이 O(N)으로 증가하는 단점이 있었습니다. 이 논문은 학습 인덱스 기술을 활용하여 이를 해결할 수 있는지를 탐구합니다.

- **Performance Highlights**: 기존 DP 인덱스와 비교했을 때, 제안하는 방법은 더 작은 저장공간을 요구하고, 성능은 유지하면서도 Privacy를 보장할 수 있는 가능성을 보여줍니다. 이는 클라우드 환경에서 안전하면서도 효율적인 데이터 접근을 위한 중요한 기초가 될 것입니다.



### A Unified Solution to Diverse Heterogeneities in One-shot Federated Learning (https://arxiv.org/abs/2410.21119)
- **What's New**: 본 논문에서는 기존의 one-shot federated learning (FL) 프레임워크의 한계를 극복하기 위해 데이터와 모델 이질성을 모두 효과적으로 처리할 수 있는 통합된 데이터 없는 one-shot FL 프레임워크인 FedHydra를 제안합니다.

- **Technical Details**: FedHydra는 구조-값 학습 메커니즘(structure-value learning mechanism)을 제안하여, 데이터 이질성이 학습 중에 반영될 수 있도록 합니다. 새로운 계층적 학습 구조(stratified learning structure)를 채택하여 데이터 이질성을 포괄하고, 각 항목의 값이 모델 이질성을 반영하게 됩니다. 이는 데이터와 모델 이질성을 동시에 모니터링할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, FedHydra는 동질적(homogeneous) 및 이질적(heterogeneous) 설정 모두에서 이전의 one-shot FL 방법보다 우수한 성능을 보였습니다. 네 개의 벤치마크 데이터 세트에서 세 가지 최첨단(SOTA) 기준과 비교하여 효과성에서 우위를 점했습니다.



### Robustness and Generalization in Quantum Reinforcement Learning via Lipschitz Regularization (https://arxiv.org/abs/2410.21117)
Comments:
          10 pages, 6 figures, 2 tables

- **What's New**: 이 논문은 양자 강화 학습(Quantum Reinforcement Learning, QRL)의 강인성과 일반화 능력을 향상시키기 위해 양자 컴퓨팅(Quantum Computing)과 제어 이론(Control Theory)의 원리를 결합한 새로운 접근 방식을 제시합니다. 특히, Lipschitz 경계(Lipschitz bounds)를 활용하여 ‘RegQPG’ 알고리즘이라고 불리는 정규화된 양자 정책 경량화(Regularized Quantum Policy Gradient) 방법을 고안하였습니다.

- **Technical Details**: RegQPG 알고리즘은 양자 정책 기울기(Quantum Policy Gradient) 훈련 절차에 Lipschitz 기반의 정규화를 추가하여 정책의 강인성과 일반화 능력을 증가시킵니다. 또한, 커리큘럼 학습(Curriculum Learning) 개념을 통합하여 훈련 중 실패를 최소화하는 알고리즘 변형(Curriculum Regularized Quantum Policy Gradients)도 도입하였습니다.

- **Performance Highlights**: 수치 실험을 통해 RegQPG가 다양한 상태 변동 강도에 대해 정책의 강인성을 향상시키는 결과를 얻었으며, Lipschitz 정규화를 적용한 경우 초기 조건의 변화에 대해 명확한 일반화 성능 향상을 보였습니다. 커리큘럼 정규화 방법(CurrRegQPG)은 훈련 중 실패 횟수를 줄이면서 일반화 능력을 추가로 개선하는 것으로 나타났습니다.



### LAMA: Stable Dual-Domain Deep Reconstruction For Sparse-View C (https://arxiv.org/abs/2410.21111)
Comments:
          Journal version for LAMA (Learned Alternating Minimization Algorithm)

- **What's New**: 본 논문에서는 Learned Alternating Minimization Algorithm (LAMA)을 개발하여 tomographic imaging에서 발생하는 inverse problems을 효율적으로 해결하는 새로운 접근 방식을 제안합니다. 이 알고리즘은 데이터 기반 기술과 고전적인 기법을 통합하여 두 블록 최적화를 수행하며, 이미지와 데이터 도메인에서 학습 가능한 정규화를 사용합니다.

- **Technical Details**: LAMA는 비볼록(nonconvex) 및 비부드러운(nonsmooth) 정규화기(regularizers)를 허용하여 데이터에서 효과적인 특징을 추출합니다. 전체 목표 함수는 Nesterov의 smoothing 기술을 사용하여 최소화되며, 잔여 학습(residual learning) 아키텍처가 적용됩니다. 이로 인해 네트워크의 복잡성이 줄어들고 기억 효율성이 향상되며, 재구성의 정확도와 해석 가능성이 증가합니다.

- **Performance Highlights**: LAMA는 Computed Tomography의 인기 있는 벤치마크 데이터셋에서 최신 기법을 상당히 초과하는 성능을 보이며, 재구성 정확도와 안정성을 높이고 있습니다. 실험 결과는 LAMA가 그 효과와 안정성을 입증하고 있습니다.



### Stronger Regret Bounds for Safe Online Reinforcement Learning in the Linear Quadratic Regulator (https://arxiv.org/abs/2410.21081)
- **What's New**: 이 연구는 온라인 강화 학습에서 안전 제약을 충족하면서 알려지지 않은 환경을 학습해야 하는 여러 실제 응용 프로그램을 다룹니다.

- **Technical Details**: 우리는 주어진 조건에 대한 Linear Quadratic Regulator (LQR) 학습을 다루며, 여기서 전체 궤적 동안 위치가 안전 영역 내에 있어야 합니다. 우리는 제한된 및 비제한된 noise 분포를 모두 허용하며, 비선형 제어기에 대한 더 강력한 기준선을 연구합니다. 1차원 상태 및 행동 공간에 집중하지만, 고차원에서도 일반화 가능성에 대해 논의합니다.

- **Performance Highlights**: 본 연구의 주요 기여는 안전 제약이 있는 LQR 학습에 대한 첫 번째 $	ilde{O}_T(	ext{sqrt}(T))$의 후회 경계(regret bound)로, 비선형 제어기의 특정 기준선에 대해 나타냅니다. 안전성 이행이 '자유 탐색(free exploration)'을 제공하여 불확실성에 따른 비용을 보상해 후회율이 제약 없는 문제와 동일하다는 점도 강조됩니다.



### Accelerated Bayesian parameter estimation and model selection for gravitational waves with normalizing flows (https://arxiv.org/abs/2410.21076)
Comments:
          accepted to NeurIPS 2024 workshop on Machine Learning and the Physical Sciences

- **What's New**: 이 연구에서는 중력파 (gravitational wave) 분석을 위한 조속한 파이프라인을 소개하며, 고성능 컴퓨팅 기법과 노멀라이징 플로우(normalizing flows)를 기반으로 한 베이지안 파라미터 추정 및 모델 선택의 효율성을 검증하였습니다.

- **Technical Details**: 연구진은 Jim 추론 툴킷과 학습된 조화 평균 추정기를 통합하여 효율성을 높였습니다. Jim은 노멀라이징 플로우 강화 MCMC 샘플러를 기반으로 하며, 관련 코드는 오픈 소스로 배포되어 있습니다. 본 연구는 고차원 중력파 추론 문제에 대해 기존 방법보다 계산 시간의 5배에서 15배까지 단축할 수 있음을 보였습니다.

- **Performance Highlights**: 연구에서 개발한 파이프라인은 GPU에서 실행하였음에도 전통적인 유사 샘플링 기법(16 CPU 코어 사용)과 비교할 때 베이지안 증거(Bayesian evidence) 추정에 있어 시간 효율성을 보여 주었으며, 중력파 데이터의 정확한 해석을 통한 모델 비교에도 높은 신뢰도를 제공하였습니다.



### Learning to Handle Complex Constraints for Vehicle Routing Problems (https://arxiv.org/abs/2410.21066)
Comments:
          Accepted at NeurIPS 2024

- **What's New**: 본 연구는 Vehicle Routing Problems (VRPs)의 복잡한 제약 조건을 다룰 수 있는 새로운 Proactive Infeasibility Prevention (PIP) 프레임워크를 제안합니다. PIP는 Lagrangian multiplier를 통합하여 제약 인식을 향상시키고, 예방적 비타당성 마스킹을 도입하여 해결 과정에서 제약 조건을 사전에 대응할 수 있도록 합니다.

- **Technical Details**: PIP 프레임워크는 reinforcement learning 프레임워크에 Lagrangian multiplier 방법을 통합하여 초기의 제약 인식을 높이고, 예방적 비타당성 마스킹을 통해 해결책 탐색을 (near-)타당한 영역으로 유도합니다. PIP-D는 보조 디코더와 두 가지 적응 전략을 사용하여 맞춤형 마스크를 학습하고 예측하며, 훈련 동안의 계산 비용을 크게 줄입니다.

- **Performance Highlights**: PIP는 다양한 제약 강도 수준에서 Traveling Salesman Problem with Time Window (TSPTW) 및 Draft Limit (TSPDL) 변형을 포함한 광범위한 실험을 통해 효과성을 입증합니다. 특히, PIP는 비타당성 비율을 최대 93.52%까지 감소시키고, 솔루션 품질이 크게 향상되는 결과를 보였습니다.



### CTINEXUS: Leveraging Optimized LLM In-Context Learning for Constructing Cybersecurity Knowledge Graphs Under Data Scarcity (https://arxiv.org/abs/2410.21060)
Comments:
          under peer-review

- **What's New**: 신규 프레임워크 CTINexus는 대규모 언어 모델(LLMs)의 최적화된 컨텍스트 학습(in-context learning, ICL)을 활용하여 데이터 효율적인 사이버 위협 정보(CTI) 지식 추출 및 고품질 사이버 보안 지식 그래프(CSKG) 생성이 가능합니다.

- **Technical Details**: CTINexus는 (1) 최적화된 자동 프롬프트 구성 전략과 최적의 시연 데이터 검색을 통해 다양한 사이버 보안 개체 및 관계를 추출하고, (2) 계층적 개체 정렬 기법으로 추출된 지식을 표준화하고 중복을 제거하며, (3) ICL 강화된 장거리 관계 예측 기법을 통해 누락된 연결을 완성합니다.

- **Performance Highlights**: 150개의 실제 CTI 보고서를 사용한 평가에서 CTINexus는 사이버 보안 트리플 추출에서 87.65%, 계층적 개체 그룹화에서 89.94%, 세부 개체 병합에서 99.80%, 장거리 관계 예측에서 90.99%의 F1 점수를 달성하며, 기존 메서드들보다 현저하게 우수한 성능을 나타냈습니다.



### BanditCAT and AutoIRT: Machine Learning Approaches to Computerized Adaptive Testing and Item Calibration (https://arxiv.org/abs/2410.21033)
- **What's New**: 이 논문에서는 적은 응답 수로도 견고한 대규모 컴퓨터 적응형 테스트(computerized adaptive test, CAT)를 신속하게 보정(calibrate)하고 관리하는 완전한 프레임워크를 제시합니다. 새로운 방법인 AutoIRT를 사용하여 항목 매개변수(item parameters)를 학습하며, 이는 자동화된 기계 학습(automated machine learning, AutoML)과 항목 반응 이론(item response theory, IRT)의 결합을 통해 이루어집니다.

- **Technical Details**: AutoIRT는 비모수적인 AutoML 채점 모델을 사용하여 항목 특성을 기반으로 학습한 후, 항목별 모수적(parametric) 모델로 이어지는 방법론입니다. 이 프레임워크에서 베이지안 업데이트(Bayesian updating)를 활용하여 시험 응시자의 능력의 후향 분포(posterior distributions)를 얻습니다. 또한, BanditCAT 프레임워크를 통해 컨텍스트 밴딧(contextual bandit) 프레임워크에서 문제를 정리하며, 선택된 항목의 피셔 정보(Fisher information)를 보상으로 정의합니다.

- **Performance Highlights**: 이 프레임워크를 활용하여 두 가지 새로운 항목 유형을 DET 연습 테스트에 초기 도입하는 데 성공했습니다. 다섯 개의 연습 테스트 실험에서의 신뢰성(reliability) 및 노출 메트릭(exposure metrics)을 정리하였습니다.



### Breccia and basalt classification of thin sections of Apollo rocks with deep learning (https://arxiv.org/abs/2410.21024)
- **What's New**: 이 연구에서는 달의 암석 샘플을 분류하기 위한 새로운 기법을 소개합니다. 머신러닝 기법을 활용하여 아폴로 미션에서 촬영된 화석화 얇은 절편 이미지에서 유의미한 특성을 추출합니다.

- **Technical Details**: 본 연구는 아폴로 미션의 화석화 얇은 절편 이미지에서 바위 유형을 분류하는 프레임워크를 구축하고, 대조 학습 (contrastive learning)과 미세 조정을 통한 Inception-Resnet-v2 네트워크를 사용한 분석을 포함합니다. 이 네트워크는 부서진 암석 (breccia)과 현무암 (basalt)을 98.44% (±1.47)의 정확도로 분류하는 이진 분류기를 훈련합니다.

- **Performance Highlights**: 대조 학습을 통해 Inception-Resnet-v2의 성능을 향상시켜 많은 양의 얇은 절편 이미지를 효과적으로 분석할 수 있으며, 향후 달 탐사 미션에서 과학적 가치를 극대화할 수 있는 도구 개발에 기여할 수 있습니다.



### SepMamba: State-space models for speaker separation using Mamba (https://arxiv.org/abs/2410.20997)
- **What's New**: SepMamba는 기존의 transformer 기반의 attention 메커니즘에 의존하지 않고, bidirectional Mamba layers로 구성된 U-Net 기반의 새로운 아키텍처를 제안합니다. 이는 계산 비용이 크게 절감되면서도 우수한 성능을 자랑합니다.

- **Technical Details**: SepMamba는 STFT(Domain) 모델 대신 원시 오디오 파형에서 작동하며, 다섯 단계의 down/up-sampling과 각 단계마다 bidirectional Mamba block을 적용합니다. 각 Bamba block은 Mamba blocks의 출력과 입력의 reversed copy에서 실행된 Mamba blocks의 출력을 결합합니다. 처리 단계에서 Mamba layer의 차원은 다운샘플링 시 두 배로 증가하고 업샘플링 시 절반으로 줄어듭니다.

- **Performance Highlights**: SepMamba는 WSJ0-2mix 데이터셋에서 transformer 기반 아키텍처와 유사하거나 더 나은 성능을 보여주며, 계산 비용, 메모리 사용량 및 전방 패스 시간이 낮다는 장점을 가지고 있습니다.



### Reference-Free Formula Drift with Reinforcement Learning: From Driving Data to Tire Energy-Inspired, Real-World Policies (https://arxiv.org/abs/2410.20990)
Comments:
          Initial submission to ICRA 2025

- **What's New**: 본 논문은 자율주행 차량이 교통사고를 피하기 위해 필요한 최대 유연성을 갖출 수 있도록 하는 드리프트 기술을 실시간으로 구현하는 방법을 제시합니다. 여기서는 비싼 궤적 최적화 없이도 차량을 필요한 위치로 이동시키는 전략을 탐구합니다.

- **Technical Details**: 강화학습(Policy Optimization) 기반 에이전트를 설계하여 타이어 에너지 흡수의 개념을 바탕으로 복잡한 웨이포인트 구성을 따라 안정적으로 드리프트할 수 있도록 훈련합니다. 연구에서는 실제 주행 데이터를 기반으로 한 신경 확률적 미분 방정식(Neural Stochastic Differential Equations) 모델을 사용하여 정책을 최적화합니다.

- **Performance Highlights**: Toyota GR Supra와 Lexus LC 500에서 실험한 결과, 에이전트는 변동하는 웨이포인트 구성을 따라 10 cm 이하의 추적 오차를 유지하며 최대 63°의 측면 미끄러짐 각도로 원활하게 드리프트할 수 있는 능력을 보였습니다.



### Large Language Model-Guided Prediction Toward Quantum Materials Synthesis (https://arxiv.org/abs/2410.20976)
Comments:
          66 pages total, 6 main figures + 3 supplementary figures

- **What's New**: 본 연구는 대형 언어 모델(LLMs)을 활용하여 무기 결정질 재료에 대한 합성 경로를 예측하는 새로운 프레임워크를 제시합니다. 이 프레임워크는 화학 반응의 제품과 반응물 예측을 위한 세 가지 모델로 구성되어 있으며, 특히 양자 재료에 대한 합성 경로 예측에 중점을 둡니다.

- **Technical Details**: 모델은 LHS2RHS, RHS2LHS, TGT2CEQ로 세분화되어 있으며, LHS2RHS 모델은 반응물에서 제품을 예측하고, RHS2LHS 모델은 특정 제품을 얻기 위한 반응물을 예측합니다. TGT2CEQ 모델은 목표 화합물에 대한 전체 화학 방정식을 생성합니다. 이 모델들은 텍스트 기반의 합성 데이터베이스인 LLM에 정밀 조정되어 정확도 향상에 기여합니다.

- **Performance Highlights**: 이 연구의 결과로 LLM 기반 모델이 높은 정확도와 유연성으로 합성 경로를 예측할 수 있음을 보여줍니다. 특히 TGT2CEQ 모델은 양자 중량이 다양한 재료에 대해 비교 가능한 예측 정확도를 달성하며, 이는 양자 재료 발견을 가속화하는 데 기여할 수 있습니다.



### BlueSuffix: Reinforced Blue Teaming for Vision-Language Models Against Jailbreak Attacks (https://arxiv.org/abs/2410.20971)
- **What's New**: 이번 연구에서는 비전-언어 모델(VLMs)에 대한 새로운 방어 방법을 제안합니다. 특히, 블랙박스 방어를 통해 jailbreak 공격에 대응하는 BlueSuffix라는 방법이 소개됩니다. 이 방법은 VLM의 성능을 저해하지 않으면서도 높은 안전성을 제공합니다.

- **Technical Details**: BlueSuffix는 세 가지 주요 구성 요소로 구성되어 있습니다: 1) jailbreak 이미지를 차단하는 시각적 정제기(visual purifier), 2) 악의적인 텍스트를 처리하는 텍스트 정제기(textual purifier), 3) 강화 학습을 통해 미세 조정된 블루 팀 접미사 생성기(blue-team suffix generator)로, 크로스 모달 강인성을 강화합니다. 모든 구성 요소는 블랙박스 환경에서 VLM의 취약점을 완화하기 위해 설계되었습니다.

- **Performance Highlights**: BlueSuffix는 LLaVA, MiniGPT-4 및 Gemini와 같은 세 가지 VLM 모델에서 테스트된 결과, 기존 방어 방법보다 공격 성공률(ASR)을 약 70%까지 감소시키는 성과를 보였습니다. 이는 VLM jailbreak 공격에 대한 방어의 새로운 기준을 설정하며, 실제 사용 사례에서 높은 실용성을 제공합니다.



### DeTeCtive: Detecting AI-generated Text via Multi-Level Contrastive Learning (https://arxiv.org/abs/2410.20964)
Comments:
          To appear in NeurIPS 2024. Code is available at this https URL

- **What's New**: 새롭게 제안된 DeTeCtive 프레임워크는 저자 간의 다양한 문체를 구별하는 데 초점을 두어, AI 생성 텍스트 감지 문제를 새롭게 접근합니다. 이러한 접근법은 기존의 이진 분류 방식에서 벗어나, 작가의 고유한 문체를 학습하는 특성을 반영합니다.

- **Technical Details**: DeTeCtive는 다중 작업 보조 및 다중 수준 대비 학습(multi-level contrastive learning) 프레임워크를 사용하여 다양한 저자의 쓰기 스타일을 학습합니다. 이 방법은 텍스트 인코더와 결합되어 AI 생성 텍스트 감지 기능을 강화하며, K-최근접 이웃(KNN) 알고리즘을 통한 유사도 측정을 사용합니다. 또한, Training-Free Incremental Adaptation (TFIA) 기능을 통해 OOD 데이터에 대한 모델의 적응력을 향상시킵니다.

- **Performance Highlights**: 제안된 방법은 여러 데이터셋에서 기존 방법보다 우수한 성능을 보이며, 특히 OOD 데이터에 대한 제로샷 평가에서 기존 방법을 크게 능가하는 결과를 기록했습니다. AvgRec 메트릭에서 Unseen Models과 Unseen Domains 테스트 세트에서 각각 5.58% 및 14.20%의 성능 향상이 확인되었습니다.



### Neuro-symbolic Learning Yielding Logical Constraints (https://arxiv.org/abs/2410.20957)
Comments:
          Published as a conference paper at NeurIPS 2023, and code is available at [this url](this https URL)

- **What's New**: 이 논문은 신경망 훈련(Neural Network Training), 기호 구속(Symbol Grounding), 논리적 제약 조건 합성(Logical Constraint Synthesis)을 통합한 자연스러운 프레임워크를 제안하여 신경-기호 시스템(Neuro-Symbolic Systems)의 종단 간 학습(end-to-end learning) 문제를 해결하는 접근 방식을 모색하고 있습니다.

- **Technical Details**: 제안된 프레임워크는 연속적인 신경망과 이산적 논리 제약 간의 격차를 해소하기 위해 차이-볼록 프로그래밍(Difference-of-Convex Programming) 기법을 도입하여 논리적 제약을 완화하고 있습니다. 또한, 카디널리티 제약(Cardinality Constraints)을 논리적 제약 학습 언어로 사용하고, 학습 과정에서 논리적 제약의 퇴화를 피하기 위해 신뢰 영역 방법(Trust Region Method)을 통합합니다.

- **Performance Highlights**: Visual Sudoku Solving, Self-Driving Path Planning, Chained XOR, Nonograms의 네 가지 작업(task)에서 실시된 실험평가(empirical evaluations)는 제안한 프레임워크가 새로운 학습 능력을 보여주며, 우수한 성능(superior performance)을 발휘함을 입증합니다.



### FACTS: A Factored State-Space Framework For World Modelling (https://arxiv.org/abs/2410.20922)
Comments:
          Code released in this https URL

- **What's New**: 이 연구에서는 공간-시간(world modelling) 이해 및 예측을 위한 새로운 구조, 즉 	extbf{FACT}ored 	extbf{S}tate-space (	extbf{FACTS}) 모델을 제안합니다.

- **Technical Details**: FACTS는 순환 기반(recurrent) 프레임워크로, 라우팅(routing) 메커니즘을 통해 순서 변경에 불변한 메모리 표현을 학습하는 그래프 구조화(graph-structured) 메모리를 구성합니다. 또한, 이 모델은 선택적 상태 공간 전파(selective state-space propagation)를 통해 적응하며, 고차원(high-dimensional) 시퀀스의 병렬(parallel) 처리를 지원합니다.

- **Performance Highlights**: 다양한 작업을 통한 실험 결과, FACTS 모델은 다변량 시간 시계열 예측(multivariate time series forecasting) 및 객체 중심(object-centric) 세계 모델링에서 전문화된 최첨단(state-of-the-art) 모델을 지속적으로 초과하거나 일치하는 성능을 발휘했습니다.



### Diff-Instruct*: Towards Human-Preferred One-step Text-to-image Generative Models (https://arxiv.org/abs/2410.20898)
- **What's New**: 이번 논문에서는 Diff-Instruct*(DI*)라는 데이터가 필요 없는 접근 방식을 소개합니다. 이 방법은 텍스트에서 이미지로의 한 단계 생성 모델을 구축하고, 인간의 선호에 맞추어 고도로 사실적인 이미지를 생성할 수 있는 능력을 유지합니다.

- **Technical Details**: DI*는 인간 피드백을 활용한 온라인 강화 학습(RLHF)으로 인간 선호 조정을 프레이밍합니다. 기존 RLHF 접근과는 달리, KL divergence 대신 새로운 score-based divergence 정규화를 도입하여 훨씬 우수한 성능을 보입니다. 이러한 분산을 직접 계산하는 것은 어려우나, 우리는 다루기 쉬운 손실 함수를 유도하여 효율적으로 기울기를 계산할 수 있음을 보여줍니다.

- **Performance Highlights**: Stable Diffusion V1.5를 참조 diffusion 모델로 사용한 DI*는 모든 기존 모델보다 큰 차이로 우수한 성능을 기록했습니다. 0.6B PixelArt-α 모델을 사용해 Aesthetic Score 6.30과 Image Reward 1.31을 달성하여 나머지 유사 모델의 점수를 거의 두 배로 초과했습니다. 또한 HPSv2 점수 28.70을 기록하여 새로운 최첨단 벤치마크를 설정했습니다.



### Active Causal Structure Learning with Latent Variables: Towards Learning to Detour in Autonomous Robots (https://arxiv.org/abs/2410.20894)
Comments:
          44 pages, 12 figures

- **What's New**: 이 논문은 인공지능 일반화(AGI) 에이전트와 로봇이 변화하는 환경에 적응하는 능력을 강조하며, 새로운 내부 인과 모델을 능동적으로 구축하기 위한 ACTIVE CAUSAL STRUCTURE LEARNING WITH LATENT VARIABLES (ACSLWL)의 필요성에 대해 설명합니다.

- **Technical Details**: ACSLWL 프레임워크는 에이전트가 환경 내에서 상호작용하며 새로운 인과 관계를 발견하고, 이를 바탕으로 인과 모델을 구축하도록 돕습니다. 주요 구성 요소로는 SURPRISE 계수, 잠재 변수 감지, 그리고 DYNAMIC DECISION NETWORK의 구조 조정이 포함됩니다.

- **Performance Highlights**: 논문에서는 에이전트가 예기치 않은 상황을 처리하기 위해 새로운 숨겨진 변수를 생성하고, 이를 통해 예측 가능한 최적의 운용 계획으로 전환할 수 있는 과정을 설명합니다. 이는 AGI 에이전트의 효율적 대처 능력을 높이기 위한 새로운 접근법입니다.



### Asteroid Mining: ACT&Friends' Results for the GTOC 12 Problem (https://arxiv.org/abs/2410.20839)
- **What's New**: 2023년 제12회 글로벌 궤적 최적화 대회에서는 '지속 가능한 소행성 채굴'을 주제로 한 문제 해결을 위한 혁신적인 방법론이 개발되어 보고되었습니다. 이 논문은 ESA의 Advanced Concepts Team이 제안한 솔루션에 대해 설명하며, 총 6만 개의 목표 소행성으로부터 자원을 추출하는 복잡한 미션 설계에 대한 전략을 다룹니다.

- **Technical Details**: 논문에서는 머신러닝(ML) 기반의 새로운 방법론과 함께 기본적인 천체역학(astrodynamics) 법칙을 조작하는 방식이 개발되었습니다. 특히, 저추력 궤적(low-thrust trajectories)과 임펄시 라베르트 전이(impulsive Lambert transfers) 간의 격차를 정확하게 메우는 방법이 포함되어 있습니다. 또한, 기존 최적 채굴 궤적의 저장소에서 최적의 부분 집합 선택을 정수 선형 프로그래밍(integer linear programming) 문제로 공식화하는 기법이 제안되었습니다.

- **Performance Highlights**: 제안된 접근 방법은 최종 경쟁 리더보드에서 4위보다 높지 않은 순위를 기록했으나, 많은 혁신적 기본 방법론들이 개발되었고, 이는 더 넓은 응용 가능성을 가지고 있습니다. 특히 자원 채굴에 있어 선행 점수를 기반으로 하는 새로운 검색 기법이 도입되어, 자원 채굴을 최대화하는 데 기여했습니다.



### FreqMark: Invisible Image Watermarking via Frequency Based Optimization in Latent Spac (https://arxiv.org/abs/2410.20824)
- **What's New**: 이번 논문에서 제안하는 FreqMark는 이미지의 잠재 주파수 공간에서 watermark를 삽입하는 새로운 방법으로, 기존의 방법들에 비해 강력한 회복력을 자랑합니다. 이는 VAE(Variational Autoencoder) 인코딩 이후의 이미지 잠재 주파수 공간을 최적화하여 이루어집니다.

- **Technical Details**: FreqMark는 이미지 라틴 주파수 공간의 비가공 최적화를 통해 watermark를 삽입합니다. 송신된 watermark는 고정된 사전 훈련된 이미지 인코더를 통해 추출됩니다. 이러한 최적화로 인해 이미지 품질과 watermark 강건성 사이의 유연한 트레이드오프가 가능하여 다양한 공격에 효과적으로 저항할 수 있습니다.

- **Performance Highlights**: 실험 결과, FreqMark는 이미지 품질과 강건성 모두에서 중요한 이점을 제공하며, 48-bit 숨겨진 메시지를 인코딩할 경우 다양한 공격 시나리오에서 90% 이상의 비트 정확도를 달성합니다. 특히 FreqMark는 재생산 공격에 대한 저항력이 우수하여 기존 방법들보다 나은 성능을 나타냅니다.



### Fidelity-Imposed Displacement Editing for the Learn2Reg 2024 SHG-BF Challeng (https://arxiv.org/abs/2410.20812)
- **What's New**: 이 논문에서는 두 가지 이미징 기술인 second-harmonic generation (SHG)과 bright-field (BF) 현미경을 결합하여 인간 유방 및 췌장 암 조직의 분석을 위한 다중 모드(multi-modal) 등록 프레임워크를 제안합니다. 기존의 학습 기반 등록 모델이 SHG 이미지를 BF 이미지에 정렬하는 데 겪는 어려움을 해결하기 위해 새로운 접근 방식을 도입했습니다.

- **Technical Details**: 제안된 프레임워크는 다음 세 가지 주요 요소로 구성됩니다: 배치-기반 대조 손실(batch-wise contrastive loss, B-NCE)을 사용하여 SHG와 BF 이미지 간의 공통된 특징을 캡처하며, 디스크립터 매칭을 통한 사전 정렬과 인스턴스 수준의 최적화(instance-level optimization)를 통한 정밀한 등록을 포함합니다. 이 방법은 지역 정규화 상관관계(local normalized cross-correlation, LNCC)와 교차 상호 정보 함수(cross mutual information function, CMIF)를 조합한 유사성 메트릭을 활용하여 글로벌 및 로컬 정렬의 균형을 맞춥니다.

- **Performance Highlights**: Learn2Reg COMULISglobe SHG-BF 챌린지에서 1위를 기록하며, 제안한 방법의 효과성을 실험적으로 증명했습니다. 정량적, 정성적 결과 모두 등록 정확도 및 강건성에서 상당한 향상을 보여줍니다.



### Deep Learning for Medical Text Processing: BERT Model Fine-Tuning and Comparative Study (https://arxiv.org/abs/2410.20792)
- **What's New**: 본 논문은 의료 정보의 폭발적인 증가에 대응하기 위해 BERT 모델을 기반으로 한 의료 문헌 요약 생성 방법을 제안합니다.

- **Technical Details**: BERT 모델을 미세 조정(fine-tuning)하고 최적화하여 키 정보를 빠르게 추출하고 일관되며 정확한 요약을 생성할 수 있는 효율적인 요약 생성 시스템을 개발했습니다. 실험에서는 Seq-Seq, Attention, Transformer 및 BERT를 포함한 다양한 모델을 비교했습니다.

- **Performance Highlights**: 개선된 BERT 모델이 Rouge 및 Recall 메트릭에서 상당한 이점을 제공함을 보여주었으며, 지식 증류(knowledge distillation) 기술이 모델 성능을 더욱 향상시킬 잠재력을 강조했습니다. 이 시스템은 실용적인 응용 프로그램에서 강력한 다재다능성과 효율성을 입증하며, 의료 문헌의 신속한 선별 및 분석을 위한 신뢰할 수 있는 도구를 제공합니다.



### SCULPT: Systematic Tuning of Long Prompts (https://arxiv.org/abs/2410.20788)
- **What's New**: 이 논문은 SCULPT(Systematic Tuning of Long Prompts)라는 새로운 프레임워크를 소개합니다. SCULPT는 긴 프롬프트를 체계적으로 수정하여 성능을 향상시키는 방법입니다.

- **Technical Details**: SCULPT는 프롬프트를 계층적으로 구성하고 반복적인 actor-critic 메커니즘을 적용하여 최적화합니다. 또한, Preliminary Assessment와 Error Assessment라는 두 가지 상호 보완적 피드백 메커니즘을 사용하여 프롬프트의 구조를 실행 전후에 평가합니다.

- **Performance Highlights**: 실험 결과, SCULPT는 기존 방법들과 비교해 정확도 향상 및 강건성(robustness) 증가를 보여주었으며, 잘못된 프롬프트를 처리하는 데에서도 우수한 성과를 낼 수 있음을 입증했습니다.



### Graph-based Uncertainty Metrics for Long-form Language Model Outputs (https://arxiv.org/abs/2410.20783)
Comments:
          Accepted as a Spotlight paper at NeurIPS 2024

- **What's New**: 이번 연구에서는 Large Language Models (LLMs)의 텍스트 생성 능력을 개선하기 위해 새로운 개념인 Graph Uncertainty를 도입합니다. 이 방법은 LLM의 생성물과 그 안의 주장 간의 관계를 이분 그래프로 표현하며, 주장 수준의 불확실성을 추정할 수 있게 해줍니다. 이로써 현재의 불확실성 추정 기법보다 더 정교한 접근법을 제공하며, 사실성을 높이는 데 기여합니다.

- **Technical Details**: Graph Uncertainty는 LLM의 생성 결과와 그 안에 포함된 주장 간의 의미적 관계를 이분 그래프로 표현하고, 다양한 그래프 중심성 지표(family of graph centrality metrics)를 사용하여 주장 수준의 불확실성을 추정합니다. 기존의 self-consistency 기반의 불확실성 측정은 degree centrality를 사용하고, 우리의 연구를 통해 closeness centrality가 더 정확한 불확실성 추정치를 제공함을 증명하였습니다. 또한, 불확실성에 민감한 디코딩 기법을 통해 신뢰할 수 있는 주장을 보존하며 LLM 생성물의 사실성을 개선합니다.

- **Performance Highlights**: 그래프 기반 불확실성 지표는 다양한 장황한 생성 설정에서 AUPRC에 평균 6.8%의 향상을 보였고, 우리의 시스템은 기존 디코딩 기법에 비해 2-4%의 사실성 향상을 지속적으로 이루어내며 생성된 응답의 정보성을 크게 개선했습니다.



### Scaling-based Data Augmentation for Generative Models and its Theoretical Extension (https://arxiv.org/abs/2410.20780)
- **What's New**: 이번 논문에서는 고품질 데이터 생성을 위한 생성 모델의 안정적인 학습 방법에 대해 연구하였습니다. 최근 개발된 방법인 Diffusion-GAN은 타임스텝 의존형 분류기를 사용하는 확산 프로세스를 통해 이 문제를 해결합니다. 이를 통해 데이터 스케일링이 안정적인 학습과 고품질 데이터 생성을 위한 핵심 요소임을 밝혀냅니다.

- **Technical Details**: Diffusion-GAN은 생성자(G)와 분류기(D)를 기반으로 하는 min-max 최적화 문제로 구성됩니다. 데이터 스케일링을 통해 진짜 데이터와 가짜 데이터 간의 격차를 줄이며, 노이즈 주입을 통해 안정성을 높입니다. 이 과정을 통해 데이터의 히스토그램을 정규화하여 안정적인 학습을 달성합니다. 이 연구에서는 Scale-GAN이라는 학습 알고리즘을 제안하며, 이는 데이터 스케일링과 분산 기반 정규화를 활용합니다.

- **Performance Highlights**: 제안된 Scale-GAN 알고리즘은 이미지 생성 벤치마크 데이터 셋에서 기존 방법들보다 뛰어난 성능을 보여주었으며, 안정성 및 정확성을 향상시키는 것으로 나타났습니다. 특히 데이터 스케일링이 노이즈 주입보다 안정화에 더 크게 기여함을 입증하였습니다.



### KD-LoRA: A Hybrid Approach to Efficient Fine-Tuning with LoRA and Knowledge Distillation (https://arxiv.org/abs/2410.20777)
Comments:
          Accepted at 4th NeurIPS Efficient Natural Language and Speech Processing Workshop (ENLSP-IV 2024)

- **What's New**: 본 연구에서는 KD-LoRA라는 새로운 미세 조정 기법을 제안하며, 이는 LoRA(저계수 적응)와 KD(지식 증류)를 결합하여 성능 손실을 최소화하면서 자원 요구 사항을 크게 줄이는 것을 목표로 합니다.

- **Technical Details**: KD-LoRA는 다음의 세 가지 주요 단계로 구성된 방법론입니다: (1) 교사 모델 선택 및 미세 조정, (2) LoRA 모듈로 초기화된 더 작은 학생 모델 설정, (3) 교사 모델에서 학생 모델로 지식을 전이하는 증류 수행.

- **Performance Highlights**: KD-LoRA는 GLUE 기준에서 LoRA의 성능을 98% 유지하면서도 GPU 메모리 사용량을 30% 줄이고, FFT보다 약 40% 더 작은 모델 크기를 달성했습니다. 또한, KD-LoRA는 추론 시간을 약 30% 단축시켰습니다.



### An Ensemble Approach to Music Source Separation: A Comparative Analysis of Conventional and Hierarchical Stem Separation (https://arxiv.org/abs/2410.20773)
- **What's New**: 이 논문은 전통적인 Vocal, Drum, Bass (VDB) 스템에서 뛰어난 분리 성능을 달성하기 위해 여러 최신 아키텍처를 결합한 앙상블 접근 방식을 제시합니다. 특히, kick, snare, lead vocals, background vocals와 같은 하위 스템에 대한 계층적 분리로 확장하여 단일 모델에 의존하는 것의 한계를 극복했습니다.

- **Technical Details**: 제안한 방법은 Signal-to-Noise Ratio (SNR)와 Signal-to-Distortion Ratio (SDR)의 조화 평균을 사용하여 스템 선택의 균형을 맞추고, 복잡한 음악 혼합에서 하위 스템을 고립할 수 있는 기능을 제공합니다. 이를 통해 모델 성능이 장르나 악기에 따라 어떻게 영향을 받는지를 탐구했습니다. 모델은 여러 아키텍처를 결합하여 Long Tail Effect를 완화하고 전통적인 VDB 이상으로 분리 과제를 확장할 수 있습니다.

- **Performance Highlights**: VDB 스템에 대한 일관되게 높은 성능을 달성했으며, 특히 하위 스템의 분리를 가능하게 함으로써 MSS 연구의 중요한 발전을 이끌었습니다. 기존에서 다루지 않았던 기타 및 피아노와 같은 악기의 분리 능력을 향상시켜 모델의 범위를 넓혔습니다.



### MrT5: Dynamic Token Merging for Efficient Byte-level Language Models (https://arxiv.org/abs/2410.20771)
- **What's New**: MrT5 (MergeT5)는 ByT5의 구조적 비효율성을 해결하는 변형 모델입니다. 입력 시퀀스의 길이를 동적으로 단축시키기 위해 토큰 삭제 메커니즘을 통합했습니다.

- **Technical Details**: MrT5는 인코더의 입력을 단축시키기 위해 고정된 인코더 층에서 토큰 삭제 게이트를 사용합니다. 학습 과정에서 삭제 정규화 기법을 활용하여 삭제할 토큰과 유지할 토큰을 결합합니다. 이 방법을 통해 MrT5는 주어진 시퀀스의 정보를 더 압축하여 담을 수 있습니다.

- **Performance Highlights**: MrT5는 ByT5에 비해 추론 시간에서显著 개선을 보이며, 시퀀스 길이를 최대 80%까지 줄이는 동시에 XNLI 및 문자 수준의 작업에서 유사한 정확도를 달성합니다.



### Robust Estimation for Kernel Exponential Families with Smoothed Total Variation Distances (https://arxiv.org/abs/2410.20760)
- **What's New**: 이 논문에서는 GAN(Generative Adversarial Networks)과 관련된 강건 추정기(robust estimators)를 사용하여 통계 모델의 일반 클래스에 대한 응용을 탐구합니다. 특히 커널 지수 분포군(kernel exponential family)을 통계 모델로 고려하여, 신뢰할 수 있는 강건 추정기를 구성하기 위한 매끄러운 전체 변동(smoothed total variation) 거리를 제안합니다.

- **Technical Details**: 통계적 추정에서, 독립적이고 동일하게 분포된 표본을 가정하지만, 이 가정은 현실에서 자주 깨집니다. 이 논문은 강건 통계(robust statistics)의 문제 설정을 바탕으로, 오염된 모델에서의 표본으로부터 목표 분포를 추정하기 위한 강건 추정기를 설계합니다. 제안된 STV 기반 추정기는 통계 모델의 오염에 강한 내성을 보입니다.

- **Performance Highlights**: STV 기반의 강건 추정기가 커널 지수 분포군에 대해 오염 분포에 견딜 수 있는 특성을 갖는다는 것이 이론적으로 밝혀졌습니다. 이 논문의 Monte Carlo 근사 방법을 사용하면 계산적인 어려움을 극복할 수 있으며, 이 방법은 정확한 예측력을 보입니다.



### Likelihood approximations via Gaussian approximate inferenc (https://arxiv.org/abs/2410.20754)
- **What's New**: 본 연구에서는 비가우시안(Bayesian) 우도(likelihood)를 가우시안 분포(Gaussian distribution)로 효율적으로 근사하는 방법을 제안합니다. 이를 통해 비가우시안 우도가 포함된 모델을 잘 알려진 가우시안 우도 전용 추론 전략으로 처리할 수 있게 됩니다.

- **Technical Details**: 이 연구에서는 변량 추론(variational inference)과 순간 일치(moment matching)를 활용하여 비가우시안 우도의 영향을 가우시안 밀도로 근사하는 방안을 제시합니다. 기존의 라플라스 매칭(Laplace Matching) 방법을 개선하여 다양한 비가우시안 우도에 적합한 방법론을 개발하였습니다.

- **Performance Highlights**: 제안된 방법들은 이진 및 다중 클래스 분류 문제에서 우수한 근사 품질을 나타내며, 특히 어려운 스트리밍 문제에서도 기존의 우도 근사 및 추정 방법들을 초월하는 성능을 보입니다. 또한, 제안된 근사 로그 우도는 신경망 분류에서 최소 제곱(least-squares) 방법에 비해 더 나은 대안으로 입증되었습니다.



### Plan$\times$RAG: Planning-guided Retrieval Augmented Generation (https://arxiv.org/abs/2410.20753)
Comments:
          22 pages, preprint

- **What's New**: Plan$	imes$RAG라는 새로운 프레임워크가 도입되었습니다. 이 프레임워크는 기존의 RAG(검색 기반 생성) 프레임워크의 '검색-후-추론'(retrieve-then-reason) 패러다임을 '계획-후-검색'(plan-then-retrieve)으로 확장합니다.

- **Technical Details**: Plan$	imes$RAG는 질의를 상호 관련된 원자 서브 질의(atomic sub-queries)로 분해하여 유향 비순환 그래프(Directed Acyclic Graph, DAG) 형태로 구성합니다. 이러한 구조를 통해 검색과 생성 과정을 병렬로 처리할 수 있어 효율성이 크게 증가합니다. 또한, 고정된 언어 모델(frozen LMs)을 전문가처럼 활용하여 고품질의 응답을 생성합니다.

- **Performance Highlights**: Plan$	imes$RAG는 기존 RAG 솔루션에 비해 환각(hallucinations)을 줄이고, 출처(attribution)를 강화하는 데 있어 상당한 개선을 보입니다. 구조화된 서브 질의 분해 덕분에 LM 기반 시스템의 신뢰성을 높이는 데 기여하고 있습니다.



### Mitigating Unauthorized Speech Synthesis for Voice Protection (https://arxiv.org/abs/2410.20742)
Comments:
          Accepted to ACM CCS Workshop (LAMPS) 2024

- **What's New**: 최근 몇 년간 몇 가지 음성 샘플만으로도 화자의 목소리를 완벽하게 복제할 수 있게 되었습니다. 하지만 악의적인 음성 이용(예: 불법 재정 이득을 위한 전화 사기)으로 인한 위험이 증가하고 있습니다. 이에 따라 개인의 음성 지문과 같은 민감한 정보를 포함하는 공개 음성 데이터를 보호하는 것이 매우 중요해졌습니다. 본 연구에서는 이를 위해 Pivotal Objective Perturbation (POP)이라는 효과적이고 전이 가능한 방어 기술을 제안합니다.

- **Technical Details**: POP는 원래의 음성 샘플에 감지 불가능한 오류 최소화 노이즈를 적용하여 TTS(텍스트-음성 합성) 모델이 음성을 효과적으로 학습하지 못하게 만들어 고품질 딥페이크 음성이 생성될 수 없도록 합니다. POP 방법은 다양한 최신 TTS 모델에 대해 객관적 및 주관적 메트릭을 사용하여 포괄적으로 평가되었습니다. 실험 결과 POP로 보호된 샘플의 음성 불명확성 점수가 127.31%로 증가하여 보호되지 않은 샘플의 점수인 21.94%에 비해 크게 향상되었습니다.

- **Performance Highlights**: POP 방식은 노이즈 감소 및 데이터 증강 기술에 대해 뛰어난 강건성을 보여주며, 이를 통해 잠재적 위험을 크게 줄일 수 있음을 착실하게 입증하였습니다. 또한 POP는 다양한 모델에서 효과성과 전이성을 보장하며, 실제 상황에서의 적용 가능성을 확보하고 있습니다.



### Simple is Effective: The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieval-Augmented Generation (https://arxiv.org/abs/2410.20724)
- **What's New**: 이번 논문에서는 Knowledge Graph (KG)에 기반한 Retrieval-Augmented Generation (RAG) 프레임워크의 한계를 극복하기 위해 SubgraphRAG를 제안합니다. SubgraphRAG는 서브그래프를 검색하고 LLM(대규모 언어 모델)을 활용하여 추론 및 답변 예측을 수행하는 구조입니다.

- **Technical Details**: SubgraphRAG는 가벼운 multilayer perceptron (MLP)와 병렬 triple-scoring 메커니즘을 통합하여 서브그래프 검색을 효율적으로 수행합니다. 이 방법은 질의의 주제 엔티티에서 구조적 거리를 인코딩하여 검색 효과성을 향상시킵니다. 서브그래프의 크기를 조정 가능하여 질의의 필요와 LLM의 능력에 따라 유연하게 적용됩니다.

- **Performance Highlights**: SubgraphRAG는 WebQSP와 CWQ 같은 KGQA 벤치마크에서 평가되었으며, 다수의 실험 결과, 소형 모델인 Llama3.1-8B-Instruct가 경쟁력 있는 성능을 보이는 동시에 GPT-4o는 이전 방법보다 더 나은 state-of-the-art (SOTA) 결과를 기록했습니다. SubgraphRAG는 hallucinations(허구) 감소와 응답의 기초 강화(improving response grounding)에서도 뛰어난 성능을 입증했습니다.



### Wearable-Based Real-time Freezing of Gait Detection in Parkinson's Disease Using Self-Supervised Learning (https://arxiv.org/abs/2410.20715)
Comments:
          2pages, 2 figures, submitted in BHI'24

- **What's New**: LIFT-PD는 파킨슨병 환자의 Freezing of Gait (FoG)를 실시간으로 탐지하기 위해 설계된 혁신적인 self-supervised learning (SSL) 프레임워크입니다. 이 프레임워크는 대규모 라벨링된 데이터셋에 대한 의존도를 최소화하고, Differential Hopping Windowing Technique (DHWT)를 적용하여 훈련 중 데이터 불균형 문제를 해결합니다.

- **Technical Details**: LIFT-PD는 단일 삼축 가속도를 사용하여 FoG 이벤트를 탐지합니다. DHWT를 이용해 불균형 데이터셋을 균형 있게 조절하고, 상시 모델 활성화를 줄이기 위해 Opportunistic Inference Module을 포함하여 에너지 소비를 감소시킵니다. 입력 신호는 다양한 환자 조건을 포함하여 시간의 경과에 따른 다변량 시계열 분류 작업으로 처리됩니다.

- **Performance Highlights**: LIFT-PD는 supervised 모델과 비교하여 40% 적은 라벨링된 샘플을 이용하면서도 7.25% 높은 precision과 4.4% 높은 accuracy를 달성했습니다. 또한, inference 시간이 67% 감소했습니다. 이로 인해 LIFT-PD는 파킨슨병 환자들을 위한 지속적인 모니터링의 실용적이면서도 에너지 효율적인 솔루션으로 평가됩니다.



### Super Resolution Based on Deep Operator Networks (https://arxiv.org/abs/2410.20706)
- **What's New**: 이번 연구에서는 Deep Operator Networks (DeepONets)를 활용하여 두 가지 유형의 부분미분방정식의 해를 고해상도로 복원하는 방법을 제시합니다. 기존의 보간(interpolation) 방법과 비교하여 DeepONets의 장점을 확인하였습니다.

- **Technical Details**: 원본 데이터를 다운샘플링하기 위해 두 가지 풀링(pooling) 방법을 사용하고, 세 가지 서로 다른 입력 이미지 해상도에서 고해상도 복원을 수행하였습니다. 2차원 문제의 경우, 순수 MLP보다 낮은 비용으로 입력 이미지에서 정보를 추출하기 위해 합성곱층(convolutional layers)을 도입하였습니다. 또한, 학습 데이터셋의 크기를 조정하면서 예측 오차(prediction errors)의 변화를 관찰하였습니다.

- **Performance Highlights**: DeepONet 모델은 저해상도 입력에서 고주파 진동(high-frequency oscillations)과 소규모 구조(small-scale structures)를 잘 예측할 수 있음을 보여주었습니다. 1차원 및 2차원 사례 모두에서, DeepONet 모델을 통한 고해상도 복원은 전통적인 삼차 스플라인 보간(cubic spline interpolation)보다 훨씬 더 정확한 예측 결과를 나타내어 연산자 학습 방법(operator learning methods)이 이러한 문제를 처리하는 데 있어 우수함을 강조하였습니다.



### Wireless-Friendly Window Position Optimization for RIS-Aided Outdoor-to-Indoor Networks based on Multi-Modal Large Language Mod (https://arxiv.org/abs/2410.20691)
- **What's New**: 본 논문은 실내 무선 통신과 자연 채광 성능을 동시에 최적화하는 것을 목표로 하며, 윈도우의 위치와 재구성 가능한 지능형 표면(reconfigurable intelligent surfaces, RIS)의 빔 방향을 조정하여 성능을 개선합니다. 또한, 대규모 언어 모델(large language model, LLM)을 최적화 도구로 활용한 윈도우 최적화 프레임워크(LMWO)를 제시했습니다.

- **Technical Details**: 리서치에서는 패시브 T-RIS와 윈도우의 조합을 사용하여 야외에서 실내로의 통신 네트워크에서의 무선 성능과 실내 자연 채광 성능을 동시에 최적화하는 문제를 다룹니다. 이를 위해 LLM 기반의 최적화 프레임워크를 설계하였으며, 윈도우 위치 및 조도를 동시에 고려한 공동 최적화 문제를 정식화했습니다.

- **Performance Highlights**: LMWO 프레임워크는 초기 성능, 수렴 속도, 최종 결과 및 시간 복잡도 측면에서 기존의 최적화 방법과 비교하여 획기적인 성능을 발휘하는 것으로 나타났습니다. 연구 결과, 건물의 무선 성능이 크게 향상되면서 실내 자연 채광 성능을 적절히 유지할 수 있음을 보여주었습니다.



### Joint Channel Selection using FedDRL in V2X (https://arxiv.org/abs/2410.20687)
- **What's New**: 이번 논문에서는 Vehicle-to-everything (V2X) 통신 기술의 발전과 이 기술이 교통 안전, 교통 효율성 및 운전 보조 시스템에 미치는 영향을 다룹니다. 특히 다양한 접근 기술을 통해 차량들이 Access Points (APs)를 선택하는 문제를 Joint Channel Selection 문제로 다루고 있습니다. 이를 해결하기 위해 Federated Deep Reinforcement Learning (FedDRL) 기반 접근 방식을 제안합니다.

- **Technical Details**: 이 연구는 Federated Deep Reinforcement Learning (FedDRL)을 활용하여 차량들이 자신의 위치, 속도 및 환경 데이터를 기반으로 채널 선택 전략을 학습하도록 지원합니다. 특히, federated Proximal Policy Optimization (FedPPO) 알고리즘을 적용하여 차량 간 경험 공유를 통해 통신 신뢰성을 높이고, 전송 비용과 채널 전환 횟수를 최소화합니다.

- **Performance Highlights**: 제안된 방법은 현실적인 시뮬레이션을 통해 평가되었으며, FedDRL이 V2X 기술을 발전시킬 수 있는 잠재력을 강조했습니다. 또한, 학습 속도와 강건성 면에서도 federated PPO가 비연합 환경에서 개발된 정책보다 더 신뢰할 수 있음을 보여주었습니다.



### Multi-modal Data based Semi-Supervised Learning for Vehicle Positioning (https://arxiv.org/abs/2410.20680)
- **What's New**: 이 논문에서는 차량 위치 지정을 위한 다중 모드 데이터 기반의 반지도 학습(SSL) 프레임워크를 제안합니다. 이 프레임워크는 채널 상태 정보(CSI) 데이터와 RGB 이미지를 결합하여 차량의 위치를 측정하는 방법을 다루고 있습니다.

- **Technical Details**: 제안하는 SSL 프레임워크는 사전 훈련(pretraining) 단계와 다운스트림 훈련(downstream training) 단계로 구성됩니다. 사전 훈련 단계에서는 이미지로부터 얻은 방위각을 비지도 CSI 데이터의 레이블로 간주하여 위치 지정 모델을 사전 훈련시킵니다. 다운스트림 훈련 단계에서는 정확한 차량 위치가 레이블로 사용되는 소규모 레이블 데이터 세트를 사용하여 모델을 재훈련합니다.

- **Performance Highlights**: 제안된 방법은 사전 훈련을 하지 않은 기준 모델에 비해 위치 오류를 최대 30%까지 줄일 수 있음을 보여줍니다. 이 연구는 차량 위치 지정을 위한 CSI 데이터와 카메라 이미지를 공동으로 사용하는 첫 번째 사례로 알려져 있습니다.



### MCI-GRU: Stock Prediction Model Based on Multi-Head Cross-Attention and Improved GRU (https://arxiv.org/abs/2410.20679)
- **What's New**: 본 논문은 다중 헤드 크로스 어텐션 메커니즘과 개선된 GRU를 활용한 새로운 주식 예측 모델인 MCI-GRU를 제안합니다. 이 모델은 전통적인 GRU 모델의 리셋 게이트를 어텐션 메커니즘으로 대체하여, 역사적 정보를 선택하고 활용하는 유연성을 증가시킵니다.

- **Technical Details**: MCI-GRU 모델은 그래프 어텐션 네트워크(GAT)를 사용하여 주식 데이터로부터 교차 섹션 특성을 추출하며, 비가시적 시장 상태의 표현을 학습하기 위해 다중 헤드 크로스 어텐션 메커니즘을 도입합니다. 이 특성들은 시간적 특성과 교차 섹션 특성과 상호작용을 통해 더욱 풍부하게 됩니다.

- **Performance Highlights**: 광범위한 실험을 통해 MCI-GRU 모델은 중국의 CSI 300 및 CSI 500 지수, 미국의 NASDAQ 100 및 S&P 500 지수를 포함한 여러 주식 시장 데이터셋에서 기존의 최고 성능 기법을 초과하는 성능을 보여주었으며, 한 주요 자산 관리 회사의 실제 운영에서도 성공적으로 적용되었습니다.



### A Machine Learning-Driven Wireless System for Structural Health Monitoring (https://arxiv.org/abs/2410.20678)
Comments:
          16 pages, 14 figures

- **What's New**: 이 논문은 항공 우주 응용을 주 대상으로 하는 탄소 섬유 강화 폴리머(CFRP) 구조물의 구조 건강 모니터링(SHM)을 위한 머신 러닝(ML) 모델이 통합된 무선 시스템을 소개합니다.

- **Technical Details**: 이 시스템은 CFRP 쿠폰에 내장된 탄소 나노튜브(CNT) 피지오저항 센서를 통해 데이터를 수집하고, 이 데이터를 중앙 서버로 무선 전송하여 처리합니다. 딥 신경망(DNN) 모델이 기계적 특성을 예측하며, 구조적 결함 예측으로 확장될 수 있습니다.

- **Performance Highlights**: ML 모델은 테스트 데이터에 대해 0.14의 평균 절대 오차(MAE)를 기록하였고, 데이터 전송 지연 시간은 LAN 설정에서 1초 미만으로 실시간 모니터링 애플리케이션에 적합합니다. 그러나 초극한 환경 조건에서의 센서 신뢰성 문제와 다양한 데이터 스트림을 처리하는 고급 ML 모델 필요성 등의 도전 과제가 미래 연구 영역으로 지적되었습니다.



### Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA (https://arxiv.org/abs/2410.20672)
Comments:
          48 pages, 17 figures, 17 tables

- **What's New**: 이번 연구에서는 Transformer 모델에서 파라미터 공유 방식인 'layer tying'을 재조명하고, 기존 LLM을 더 작은 'Recursive Transformers'로 변환하기 위한 새로운 방법론을 제안합니다. 이 방법은 성능 손실을 최소화하면서 층 간에 파라미터를 공유합니다.

- **Technical Details**: Recursive Transformers는 미리 훈련된 Transformer에서 효율적으로 초기화되며, 고유한 층 블록을 반복적으로 사용합니다. Relaxed Recursive Transformers는 depth-wise low-rank adaptation (LoRA) 모듈을 도입하여 층 간의 제약을 완화하고, 모델의 compactness를 유지합니다. 새로운 추론 패러다임인 Continuous Depth-wise Batching을 통해 추가적인 처리량 향상 가능성을 보여줍니다.

- **Performance Highlights**: Recursive Transformer는 비슷한 크기의 vanilla pretrained 모델 및 지식 증류(baseline)와 비교해 우수한 성능을 나타내며, 특히 recursive Gemma 1B 모델은 non-recursive Gemma 1B 모델 대비 13.5%의 정확도 향상을 보여줍니다. 이 모델은 같은 아키텍처의 vanilla Transformer와 비교하여 최대 2-3배의 처리량 향상을 실현할 수 있습니다.



### A Statistical Analysis of Deep Federated Learning for Intrinsically Low-dimensional Data (https://arxiv.org/abs/2410.20659)
- **What's New**: 이번 연구는 연합 학습(FL)의 일반화 오류를 탐구하며, 이질적 환경에서의 딥 연합 회귀의 일반화 특성을 분석합니다. 특히, 엔트로픽 차원(entropic dimension)을 기준으로 수렴 속도를 결정하는 데 있어 중요한 역할을 한다는 점을 강조합니다.

- **Technical Details**: 이 논문은 2단계 샘플링 모델을 활용하여 딥 연합 회귀의 일반화 속성을 조사합니다. $eta$-Hölder 함수로 설명되는 응답 변수와 설명 변수 간의 진정한 관계를 가정할 때, 참가 클라이언트의 오류율은 최대 $	ilde{O}ig((mn)^{-2eta/(2eta + ar{d}_{2eta}(	ext{λ}))}ig)$로 스케일링되며, 비참여 클라이언트의 경우는 $	ilde{O}ig(	ext{Δ} 	imes m^{-2eta/(2eta + ar{d}_{2eta}(	ext{λ}))} + (mn)^{-2eta/(2eta + ar{d}_{2eta}(	ext{λ}))}ig)$입니다. 여기서 $ar{d}_{2eta}(	ext{λ})$는 설명 변수의 주변 분포를 나타냅니다.

- **Performance Highlights**: 본 연구의 결과는 딥 연합 학습자의 수렴 속도가 명목상의 고차원성(nominal high-dimensionality)보다 내재적 차원(intrinsic dimensionality)에 따라 달라진다는 것을 명확히 입증합니다. 엔트로픽 차원을 사용하는 방식은 기존의 미코프스키 차원(Minkowski dimension)이나 바소슈타인 차원(Wasserstein dimension)보다 우수한 경계(bound)를 제공하는 것으로 나타났습니다.



### Injectivity capacity of ReLU gates (https://arxiv.org/abs/2410.20646)
- **What's New**: 이번 연구는 ReLU 네트워크 레이어의 injectivity 속성을 다룹니다. ReLU의 injectivity 용량(입력 수와 출력 수의 비율)을 정의하고, 이를 $\ell_0$ spherical perceptron의 용량 결정과 이론적으로 동형으로 구합니다. 완전 리프트 랜덤 이중성 이론(fl RDT)을 활용하여 ReLU 레이어의 injectivity를 다루기 위한 강력한 프로그램이 개발되었습니다.

- **Technical Details**: fl RDT 이론을 통해 ReLU 레이어의 injectivity 속성과 관련된 수학적 문제를 해결하며, 이론적으로 $	ext{ReLU injectivity capacity}$를 다양한 리프트 수준에 대해 정밀하게 특성화합니다. 또한, 핵심 리프트 매개변수들 간의 닫힌 형식의 분석적 관계를 발견했습니다.

- **Performance Highlights**: 수치적 평가를 통해 리프트 메커니즘의 빠른 수렴을 확인하였으며, 추정된 양의 상대적 수정이 이미 세 번째 리프트 수준에서 0.1%를 초과하지 않는 성과가 나타났습니다. 이 결과는 [40]의 복제 예측과도 밀접하게 일치합니다.



### Near Optimal Pure Exploration in Logistic Bandits (https://arxiv.org/abs/2410.20640)
Comments:
          25 pages, 2 figures

- **What's New**: 이 논문에서는 logistic bandit 설정 하에 일반적 순수 탐색 문제에 대한 최초의 track-and-stop 알고리즘인 Log-TS(logistic track-and-stop)를 제안합니다. 이 알고리즘은 특정 인스턴스에 대한 예상 샘플 복잡성의 하한을 로그 계수까지 일치시킵니다.

- **Technical Details**: Log-TS는 logistic bandit에서 일반적인 순수 탐색 문제를 해결하기 위한 알고리즘으로, 샘플 복잡성의 기대값에 대한 상한과 하한을 제공합니다. 특히, forced exploration이 존재할 때 비정규화된 logistic regression에 대한 tail inequality를 확장하며, BAI(best arm identification), top-m, thresholding bandits를 포함하는 일반적인 순수 탐색 문제에 대한 예상 샘플 복잡성의 하한을 제시합니다.

- **Performance Highlights**: Log-TS는 순수 탐색 문제의 전통적인 어려운 인스턴스와 팔 수(arms)의 수가 증가할 때의 실제 성능을 검증했습니다. 이 알고리즘의 효율성은 비약적으로 향상되었으며 이론적으로도 기대값과 거의 확실한 샘플 복잡성에서 우수한 성능을 보였습니다.



### Kernel Approximation of Fisher-Rao Gradient Flows (https://arxiv.org/abs/2410.20622)
- **What's New**: 이번 논문은 kernel 방법과 PDE gradient flows 간의 인터페이스에 관한 몇 가지 개방된 질문에 답하기 위한 연구입니다. 특히 최근 머신러닝의 발전(특히 generative modeling과 sampling)에 의해 촉발된 연구로, Fisher-Rao와 Wasserstein 유형의 gradient flows의 더 엄격한 조사를 다룹니다.

- **Technical Details**: Fisher-Rao 기하학(Fisher-Rao geometry)과 다양한 kernel 기반 근사(kernel-based approximations)에 초점을 맞추고, PDE gradient flows 및 최적 수송 이론(optimal transport theory)의 도구를 활용하여 체계적인 이론적 프레임워크를 개발하였습니다. 논문은 최대 평균 불일치(maximum-mean discrepancy, MMD) 공간에서의 gradient flows의 완전한 특성화를 제공하며, 기존의 학습 및 추론 알고리즘과의 연결을 제시합니다.

- **Performance Highlights**: 연구 결과는 Fisher-Rao flows, Stein flows, kernel 부족(kernel discrepancies), 비모수 회귀(nonparametric regression) 사이의 정확한 이론적 통찰을 보여주며, kernel로 근사한 Fisher-Rao flows에 대해 진화적인 Γ-convergence를 엄격하게 증명했습니다. 또한 Helmholtz-Rayleigh 원리를 사용하여 에너지 소산(energy dissipation)을 분석하며, 고전적인 역학 이론(classical theory in mechanics)과 현대 머신러닝 관행 사이의 중요한 연결을 확립했습니다.



### Implementation and Application of an Intelligibility Protocol for Interaction with an LLM (https://arxiv.org/abs/2410.20600)
- **What's New**: 이번 연구에서는 인간 전문가와 기계 학습 엔진 간의 상호 작용을 통한 데이터 분석 시스템 구축에 대해 다루고 있습니다. 특히 기존의 통계적 혹은 수학적 모델링 방법으로 해결하기 어려운 복잡한 문제들에 대한 예측 및 설명 생성 방식을 탐구하고 있습니다.

- **Technical Details**: 저자들은 두 가지 상호작용 에이전트 간의 커뮤니케이션을 위한 추상 프로토콜(PXP protocol)을 구현하였습니다. 이 프로토콜은 '양방향 이해 가능성(two-way intelligibility)' 개념을 기반으로 하며, 유한 상태 기계(finite-state machines)를 사용하여 정의됩니다. 실험에서는 대규모 언어 모델(LLM)을 사용하는 생성기(generator) 에이전트와 인간 전문가 또는 인간 전문가의 프록시 역할을 하는 테스트(agent) 간의 상호작용을 설명하고 있습니다.

- **Performance Highlights**: 초기 실험 결과는 PXP 프로토콜이 LLM과 인간 간의 이해 가능성(one- and two-way intelligibility)을 효과적으로 캡처할 수 있음을 제공하고 있습니다. 특히 방사선학(radiology) 및 약물 발견(drug-discovery) 분야에서의 사용 가능성을 보여주고 있습니다.



### A Framework for Real-Time Volcano-Seismic Event Recognition Based on Multi-Station Seismograms and Semantic Segmentation Models (https://arxiv.org/abs/2410.20595)
Comments:
          10 pages, 9 figures. This is a pre-print, it is currently under review fro publication at Computers and Geosciences, by Elsevier

- **What's New**: 이번 연구는 다중 채널 1D 신호를 2D 이미지로 변환하는 단순한 변환을 적용하여 화산 모니터링을 자동화하는 새로운 접근 방식을 제안합니다. 이를 통해 세 가지 주요 문제를 해결하고, 다중 스테이션 데이터를 통합하여 실시간 화산 모니터링에 적합한 모델을 개발했습니다.

- **Technical Details**: 제안된 프레임워크는 semantic segmentation 모델을 활용하여 5개의 화산 사건 클래스를 동시에 탐지하고 분류할 수 있도록 설계되었습니다. 약 25,000개의 지진 이벤트에 대해 UNet, UNet++, DeepLabV3+, SwinUNet 등의 최신 segmentation 모델을 평가했으며, UNet 아키텍처가 가장 높은 성능을 보였습니다.

- **Performance Highlights**: UNet 모델은 평균 F1 점수 0.91 및 IoU 점수 0.88을 달성하며, 높은 잡음 강도에 대한 저항성과 기존 화산 데이터셋에 대한 적응력을 보여 주었습니다. 이 연구는 심층 학습을 사용한 지진 이벤트 인식을 위한 새로운 솔루션을 제공하여 실시간 화산 모니터링에 기여할 수 있을 것입니다.



### Unsupervised Panoptic Interpretation of Latent Spaces in GANs Using Space-Filling Vector Quantization (https://arxiv.org/abs/2410.20573)
- **What's New**: 본 논문에서는 데이터 레이블이나 주석이 달린 샘플을 필요로 하지 않는 새로운 접근 방법인 space-filling vector quantization (SFVQ)를 제안합니다. 이는 모호한 latent space를 해석 가능하게 만드는 데 기여합니다.

- **Technical Details**: SFVQ는 데이터의 조각별 선형 곡선 위에서 양자화하여 latent space의 기본 형태적 구조를 캡처합니다. 본 연구는 pretrained StyleGAN2 및 BigGAN 네트워크의 latent space 모델링에 이 기법을 적용하였습니다.

- **Performance Highlights**: 실험 결과, SFVQ 곡선은 latent space의 일반적인 해석 가능한 모델을 제공하며, 이것이 각각의 생성 요인에 대응하는 latent space의 어떤 부분인지 파악할 수 있게 합니다. 또한, SFVQ의 곡선의 각 선은 이해할 수 있는 이미지 변환을 적용하기 위한 해석 가능한 방향으로 활용될 수 있습니다.



### Neural rendering enables dynamic tomography (https://arxiv.org/abs/2410.20558)
Comments:
          24 pages, 14 figures. Submitted to NeurIPS 2024 ML4PS. For associated visualizations, see this https URL

- **What's New**: 본 연구는 X선 컴퓨터 단층 촬영(X-CT)이 동적 실험 동안 3D 재구성을 가능하게 하도록 신경 렌더링 도구를 활용하는 방법을 제안합니다. 기존의 정적 실험에서는 효과적이었던 접근 방식이 동적 실험에서는 재구성이 불가능했기 때문에, 이러한 패러다임 전환이 이루어졌습니다.

- **Technical Details**: 신경 방사선 필드(neural radiance fields)는 기존의 재구성 방법보다 관심 있는 데이터 모달리티를 더 효율적으로 재구성할 수 있음을 보여줍니다. 이 연구에서는 입장 기준(projection angles)을 선택하기 위한 이론적 결과를 도출하고, 합성 데이터와 실험적 데이터를 결합하여 3D 밀도 필드를 훈련시키는 차별화된 프로젝터/렌더러를 사용합니다. 또한, 스플라인 기반 변형 필드(spatio-temporal model with spline-based deformation field)를 개발하여 실제 실험에서 격자 샘플의 시공간 변형을 재구성하는 방법을 제시합니다.

- **Performance Highlights**: 본 연구는 두 개의 프로젝션만으로도 격자 구조의 변형 사건을 재구성할 수 있다는 것을 입증하였으며, 전통적인 컴퓨터 단층 촬영과 신경 렌더링을 결합함으로써 시너지를 창출해냄으로써 동적 사건의 재구성이 가능함을 보여줍니다.



### SIGMA: Single Interpolated Generative Model for Anomalies (https://arxiv.org/abs/2410.20537)
Comments:
          12 pages, 6 figures

- **What's New**: SIGMA(단일 보간 생성 모델)이라는 새로운 데이터 기반의 방법을 제안하여 신호 영역에 대한 백그라운드 분포를 추정하는 데 있어서 기존의 방법보다 계산 비용을 크게 줄인다. 기존 방법들은 실제 신호 영역에 대해 생성 모델을 재훈련해야 했으나, SIGMA는 모든 데이터를 기반으로 단일 모델을 훈련하고 이를 통해 신호 영역의 파라미터를 보간(interpolate)하여 백그라운드 모델을 얻는다.

- **Technical Details**: SIGMA 방법은 데이터 전체를 사용하여 단일 생성 모델을 훈련시키고, 이 모델의 파라미터를 신호 영역의 사이드밴드(sideband)에서 보간하여 백그라운드 모델을 생성한다. 전통적인 방법과 달리, SIGMA는 재훈련 과정을 줄여 계산 비용을 줄이고, 유사한 품질의 백그라운드 모델링 성능을 유지한다. 이 방법은 flow-matching 접근 방식을 사용하여 신호 영역 내에서 벡터 필드를 쉽게 보간할 수 있게 한다.

- **Performance Highlights**: SIGMA는 높은 품질의 백그라운드 샘플을 생성하면서도 훈련 비용을 크게 절감한다. 기존의 CURTAINSF4F 방법보다도 훈련에 필요한 데이터가 적기 때문에 더욱 효율적이다. SIGMA의 성능은 다양한 신호 감지 기법과 백그라운드 템플릿을 비교하여 검증되었으며, 이와 관련된 여러 실험을 통해 그 유효성이 입증되었다.



### Search Wide, Focus Deep: Automated Fetal Brain Extraction with Sparse Training Data (https://arxiv.org/abs/2410.20532)
- **What's New**: 이번 연구는 sparse하고 synthetic한 라벨로 학습된 네트워크의 false positive를 줄이는 테스트 타임 전략을 제안합니다. 이 방법은 breadth-fine search(BFS)와 deep-focused sliding window(DFS)를 활용하여 태아 뇌를 더욱 정확하게 추출합니다.

- **Technical Details**: 이 프레임워크는 3D multi-scale deep neural networks를 사용하여 full-uterus Stack-of-slices MRI로부터 태아 뇌 마스크를 추출합니다. BFS와 DFS를 통해 검색 범위를 조정하고, 각기 다른 크기의 윈도우를 사용하여 다양한 모델을 트레이닝함으로써 데이터에 대한 일반화를 이루어냅니다.

- **Performance Highlights**: 이 방법은 세 번째 삼 분기 임신 태아에 대해 기존의 최신 방법과 동등한 성능을 보였으며, 두 번째 삼 분기에서는 Dice 점수에서 최대 5% 향상된 결과를 보여주었습니다. 이는 모델의 정확성을 높이고 태아 뇌를 더욱 효과적으로 분리하는 데 기여했습니다.



### CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming (https://arxiv.org/abs/2410.20527)
- **What's New**: CodeRosetta는 프로그래밍 언어와 그 HPC (High-Performance Computing) 확장을 번역하기 위해 특별히 설계된 인코더-디코더 트랜스포머 모델입니다. 이 모델은 C++에서 CUDA로, 그리고 Fortran에서 C++로 번역하는 작업에서 평가되었습니다.

- **Technical Details**: 이 모델은 코드 의미론을 효과적으로 캡처하고 병렬 구조의 미세한 뉘앙스를 이해하기 위해 맞춤형 학습 프레임워크와 사전 훈련(pretraining) 및 훈련(training) 목표를 사용합니다. 이를 통해 양방향 번역이 가능해집니다.

- **Performance Highlights**: CodeRosetta는 C++에서 CUDA로의 번역에서 기존의 최신 기법보다 2.9 BLEU 포인트와 1.72 CodeBLEU 포인트 우수한 성능을 보이며, 컴파일 정확도를 6.05% 향상시킵니다. 또한, 일반 폐쇄형 LLM과 비교 시 C++에서 CUDA로의 번역에서 22.08 BLEU와 14.39 CodeBLEU 포인트를 개선하고, 컴파일 정확도를 2.75% 높였습니다. 마지막으로, Fortran에서 병렬 C++로의 번역에서도 우수한 성과를 보여주며, 이는 복잡한 작업에 대한 첫 번째 인코더-디코더 모델로 여겨집니다.



### Symbotunes: unified hub for symbolic music generative models (https://arxiv.org/abs/2410.20515)
- **What's New**: Symbotunes는 기호 음악 생성 모델을 위한 오픈 소스 통합 허브로, 여러 유명한 모델의 현대적인 Python 구현과 함께 데이터 생성 및 훈련을 위한 통합 파이프라인을 제공합니다.

- **Technical Details**: Symbotunes는 anaconda 패키지 관리자를 활용하여 의존성을 관리하며, Python 3.12와 PyTorch 2.2, PyTorch Lightning 2.2, MidiTok 3.0을 사용합니다. 네 가지 주요 하위 폴더로 구성되어 있으며, 사용자에게 yaml 형식의 구성 파일을 통해 다양한 실험을 간편하게 지원합니다. Symbotunes는 Folk-RNN, MusicVAE, ABC GPT2와 같은 기호 음악 모델의 구현을 포함하고 있습니다.

- **Performance Highlights**: Symbotunes는 재현성을 높이고, 교육자와 연구자가 기호 음악 모델을 탐색할 수 있도록 표준화된 플랫폼을 제공합니다. 향후 더 많은 모델, 데이터셋 및 기능 추가를 계획하고 있으며, 기호 음악 생성 연구에 중요한 자원이 되고자 합니다.



### When Less is More: Achieving Faster Convergence in Distributed Edge Machine Learning (https://arxiv.org/abs/2410.20495)
Comments:
          11 pages, 19 figures, 3 tables; code: this https URL

- **What's New**: 이 논문에서는 자원 제약이 있는 엣지 장치에서 효율적인 분산 머신 러닝(Distributed Machine Learning, DML)을 위한 새로운 프로바빌리스틱(framework)인 Hermes를 제안합니다. Hermes는 모델의 일반화 능력에서 주요 개선을 통계적으로 식별하기 위해 최근 테스트 손실 동작을 기반으로 하는 동적 기준을 활용합니다.

- **Technical Details**: Hermes는 텐서플로우(TensorFlow)를 기반으로 한 데이터 병렬 DML 프레임워크로, 각 장치의 용량에 따라 배치 크기를 동적으로 할당하며, 모델 압축 기법을 도입하여 엣지 장치의 메모리 제한 문제를 해결합니다. 또한, 통신 효율적인 전략을 구현하여 주요 변화가 있을 때만 그래디언트 업데이트를 전송함으로써 통신 오버헤드를 줄입니다.

- **Performance Highlights**: Hermes는 최신 알고리즘과 비교하여 훈련 시간을 13.22배 감소시키고, 통신 오버헤드는 62.1% 절감되었습니다. 이로 인해 전체적인 훈련 속도가 향상되었습니다.



### TrajAgent: An Agent Framework for Unified Trajectory Modelling (https://arxiv.org/abs/2410.20445)
Comments:
          12 pages; the code will be openly accessible at: this https URL

- **What's New**: 본 논문에서는 다양한 경로 모델링(task) 작업을 통합하기 위해 LLM(대형 언어 모델)을 기반으로 한 간편한 프레임워크인 TrajAgent를 제안합니다. TrajAgent는 통합된 데이터 및 모델 인터페이스를 제공하는 UniEnv를 개발하여 다양한 모델을 실행하고 훈련할 수 있도록 지원합니다.

- **Technical Details**: TrajAgent는 TAgent라는 자동 경로 모델링 작업을 설계하고, AutOpt라는 최적화 모듈을 통해 성능을 향상시킵니다. UniEnv는 데이터 전처리 및 일관된 작업 흐름을 지원하며, 다양한 경로 모델링 작업을 위한 통합된 환경을 제공합니다. 경로 예측, 분류, 생성 작업을 포함한 다양한 작업 유형을 지원합니다.

- **Performance Highlights**: 실험 결과, TrajAgent는 네 가지 실제 데이터 세트에 대해 평균 15.43%의 성능 향상을 기록하여 통합 경로 모델링에서의 효과성을 입증하였습니다.



### FoldMark: Protecting Protein Generative Models with Watermarking (https://arxiv.org/abs/2410.20354)
- **What's New**: 최근 생성 AI의 통합으로 단백질 구조 예측 및 디자인의 정확성과 사용성이 크게 향상되었습니다. 그러나 저작권 보호와 생물안전성(biosecurity)에 대한 윤리적 우려가 존재합니다. 본 연구에서는 단백질 생성 모델에 물체(워터마크, watermark)를 삽입하여 저작권 인증과 생성 구조 추적이 가능한지를 조사합니다.

- **Technical Details**: 제안된 FoldMark 방법은 두 단계로 구성되어 있습니다. 첫 번째 단계에서는 워터마크 인코더와 디코더를 사전 학습하여 단백질 구조에 사용자 특정 정보를 조정하여 임베딩합니다. 두 번째 단계에서는 Low-Rank Adaptation(LoRA) 모듈을 사용하여 워터마크를 포함한 구조를 생성하는 데 필요한 모델을 미세 조정합니다. 이를 통해 FoldMark는 단백질 생성 모델의 품질을 유지하면서도 우수한 복구율을 달성합니다.

- **Performance Highlights**: FoldMark는 실험을 통해 거의 100%의 비트 정확도(BitAcc)를 달성하였으며, 기존 단백질 생성 모델에 비해 구조의 유효성(scRMSD 및 RMSD)에 미치는 영향이 미비합니다. 본 방법은 다양한 후처리(post-processing) 및 적응 공격(adaptive attacks) 상황에서도 강력한 성능을 보이며, 저작권 보호와 사용자 추적에서 유용성을 입증했습니다.



### Logarithmically Quantized Distributed Optimization over Dynamic Multi-Agent Networks (https://arxiv.org/abs/2410.20345)
- **What's New**: 본 연구에서는 수용된 지수적(지수 기반) 양자화 방식이 적용된 분산 최적화 동역학을 제안합니다. 기존의 균일한 양자화 방식보다 더 높은 정밀도를 제공하며 데이터를 효율적으로 표현합니다.

- **Technical Details**: 제안된 동역학은 주 상태 변수와 보조 변수를 통해 최적화 문제를 해결합니다. log-quantization 방식을 사용하여 더 낮은 값은 더 많은 비트를 사용하고, 더 높은 값은 적은 비트를 사용하여 불균형한 데이터 배포 시 성능을 향상시킵니다. 이를 위해 매트릭스 섭동 이론과 고유 스펙트럼 분석을 활용한 수렴 분석을 기반으로 합니다.

- **Performance Highlights**: 제안된 방법은 분산 SVM 기반 이진 분류 문제에서 실질적인 유용성을 보이며, 최적화 잔여와 최적성 격차가 없음을 입증했습니다. 이는 기존의 균일한 양자화 방법보다 뛰어난 성능을 제공합니다.



### Low-rank Bayesian matrix completion via geodesic Hamiltonian Monte Carlo on Stiefel manifolds (https://arxiv.org/abs/2410.20318)
- **What's New**: 본 논문에서는 효율적인 저차원 Bayesian matrix completion을 위한 샘플링 기반 접근법을 제시합니다. 저차원 행렬의 singular-value-decomposition(SVD) 매개변수를 기반으로 하는 새로운 prior 모델을 설계하였으며, 이는 기존 non-Bayesian 설정에서 사용되는 nuclear-norm regularization과 유사합니다.

- **Technical Details**: 제안된 방법은 Stiefel manifold에 factor matrix의 직교성을 강제하는 새로운 구조를 도입합니다. 또한, 일반적인 Gaussian likelihood보다 더 일반적인 likelihood를 포함하는 샘플링을 가능하게 하는 geodesic Hamiltonian Monte Carlo 방법을 채택합니다. 이로 인해 기존의 Gibbs sampler에서 발생하는 샘플링 문제를 해결할 수 있습니다.

- **Performance Highlights**: Numerical examples를 통해 우리 방법이 기존의 방법들보다 우수한 샘플링 성능을 가지며, 두 가지 실제 벤치마크 문제에서 정확도 향상을 보여줍니다. 특히, mice protein dataset과 MovieLens 추천 문제에 대한 적용 사례를 통해 이점이 입증되었습니다.



### Fine-Tuning and Evaluating Open-Source Large Language Models for the Army Domain (https://arxiv.org/abs/2410.20297)
- **What's New**: 최근 대규모 언어 모델(LLMs)의 군사 도메인에서의 활용 가능성에 대한 연구가 활발히 이루어지고 있습니다. 본 논문에서는 Army 도메인에 적합한 오픈소스 LLM을 조정하기 위한 노력으로 TRACLM이라는 LLM 계열을 제안합니다.

- **Technical Details**: 본 연구에서 소개하는 TRACLM은 The Research and Analysis Center(TRAC)와 Army Futures Command(AFC)에서 조정된 LLM의 세 가지 세대를 포함하고 있으며, 각 세대의 모델은 훈련 프로세스의 지속적인 개선을 거쳐 Army 작업과 사용 사례에 적용될 때 능력이 향상되었습니다. 또한 MilBench라는 평가 프레임워크를 개발하여 LLM의 Army 도메인 지식을 정량적으로 평가할 수 있도록 하였습니다.

- **Performance Highlights**: TRACLM 모델은 Army 특정 작업에서 성능이 향상된 것으로 나타났습니다. MilBench는 LLM의 평가를 위한 확장 가능하고 효율적인 소프트웨어 프레임워크로, 다양한 DoD(Department of Defense)에서의 활용 가능성을 높입니다.



### On the Gaussian process limit of Bayesian Additive Regression Trees (https://arxiv.org/abs/2410.20289)
- **What's New**: 이번 논문에서는 Bayesian Additive Regression Trees (BART)의 사전 공분산 함수(prior covariance function)를 정확하게 도출하고 계산한 첫 번째 사례를 소개합니다. 이로써 BART의 무한 결정 트리(infinite trees limit)를 Gaussian process (GP) 회귀로 구현하였습니다.

- **Technical Details**: BART는 일종의 비모수 베이지안(regression technique) 회귀 기법으로, 결정 트리의 합(sum-of-decision-trees model) 형태를 가집니다. 본 논문에서는 BART의 무한 결정 트리 및 GP 회귀의 비교를 통해 하이퍼파라미터(hyperparameters)의 조정이 어떻게 GP 방법론으로 경쟁력 있는 결과를 가져오는지를 보여줍니다. 또한 BART를 GP로 구현함으로써 분석적 가능성(analytical likelihood)을 이용, 모델 구축(model building)을 단순화하고 복잡한 BART MCMC를 회피할 수 있는 장점을 제시합니다.

- **Performance Highlights**: 경험적 테스트 결과, 무한 트리로 구성된 BART는 고정 구성에서 표준 BART보다 성능이 떨어지지만, 자연적인 GP 방식으로 하이퍼파라미터를 조정할 경우 경쟁력 있는 방법이 될 수 있음을 확인하였습니다. 그러나 적절하게 조정된 BART는 여전히 우수한 성능을 유지합니다. 이 연구는 BART 및 GP 회귀를 이해하고 발전시키는 새로운 방법을 제시합니다.



### Learning Approximated Maximal Safe Sets via Hypernetworks for MPC-Based Local Motion Planning (https://arxiv.org/abs/2410.20267)
- **What's New**: 이 논문은 모바일 로보틱스의 로컬 모션 플래닝 작업을 위한 안전 집합(maximal safe sets)을 온라인으로 추정하는 새로운 학습 기반 접근 방식을 제시합니다. 또한, 이 접근 방식은 하이퍼네트워크(hypernetworks)를 활용하여 뛰어난 일반화 성능과 실시간(real-time) 성능을 동시에 구현합니다.

- **Technical Details**: 우리는 해밀턴-야코비(Hamilton-Jacobi, HJ) 도달 가능성 분석을 감독(supervision) 방법으로 활용하여 일반 비선형 동역학(nonlinear dynamics)과 다양한 제약조건을 고려합니다. 우리의 모델은 MPC 모델 예측 제어(model predictive control, MPC) 로컬 플래너에 안전 제약으로 통합되며, 다양한 환경과 로봇 동역학에 대한 현실적인 3D 시뮬레이션에서 성능을 비교합니다.

- **Performance Highlights**: 우리 방법은 기존의 HJ 접근 방식과 비교하면 성공률(success rate)을 2%에서 18% 향상시키고, 실시간 성능도 달성합니다. 이를 통해 고차원 및 시시간 시스템에서 HJ 도달 가능성을 적용할 수 있는 개선된 효용성을 보여줍니다.



### You Never Know: Quantization Induces Inconsistent Biases in Vision-Language Foundation Models (https://arxiv.org/abs/2410.20265)
Comments:
          Workshop paper at NeurIPS 2024 RBFM. 6 pages, 3 figures

- **What's New**: 이 논문은 quantization이 foundation vision-language 모델의 사회적 공정성에 미치는 영향을 평가한 연구로, 이전의 unimodal 모델에 대한 발견과 달리, quantization을 통해 생성된 여러 compressed 모델에서 지속적인 편향의 변화가 관찰되지 않았다는 점이 새롭게 밝혀졌습니다.

- **Technical Details**: 이 연구에서는 8-bit 및 4-bit quantization 방법을 사용하여 다양한 CLIP 모델 변형을 평가하였으며, 각각의 quantization 방법은 메모리 절약과 추론 지연 시간 단축을 가능하게 합니다. 특히, 모델의 공정성을 측정하기 위해 FACET 및 FairFace 데이터셋을 사용하여 성별, 나이, 인종 등의 민감한 특성에 대한 결과를 분석하였습니다.

- **Performance Highlights**: 연구 결과, quantized 모델에서 개별 모델은 편향을 보여주었으나, 전체 모델 집단에서는 편향의 방향과 크기에 일관된 변화가 없었습니다. 이는 quantization이 multimodal context에서 공정성에 미치는 영향이 더욱 복잡할 수 있음을 시사합니다.



### Robust Model Evaluation over Large-scale Federated Networks (https://arxiv.org/abs/2410.20250)
Comments:
          40 pages

- **What's New**: 본 논문에서는 머신러닝 모델의 성능을 미리 알려지지 않은 타겟 네트워크에서 인증하는 문제를 다룹니다. 이 연구는 데이터셋이 서로 다른 분포를 가지는 클라이언트들로 이루어진 소스 네트워크에서 측정된 데이터를 사용합니다. 특히, $K$명의 클라이언트로 구성된 소스 네트워크 'A'가 있으며, 이들은 각각 독립적인 샘플을 가지고 있습니다.

- **Technical Details**: 우리는 모델의 경험적 평균 손실(empirical average loss)에 대한 이론적 보장을 제공하며, 위험 CDF(risk CDF)에 대한 균일한 경계(uniform bounds)를 제시합니다. 이 경계들은 Glivenko-Cantelli 정리와 Dvoretzky-Kiefer-Wolfowitz(DKW) 부등식의 새로운 버전으로, Wasserstein 거리나 $f$-divergence로 제어되는 두 개의 메타 분포 간의 차이를 가정하고 있습니다. 이러한 경계는 다항식 시간 내에 계산 가능하며, 클라이언트 데이터의 개인 정보를 보호합니다.

- **Performance Highlights**: 본 연구의 실험 결과는 다양한 실제 작업에서 제시된 경계의 견고성과 실용성을 입증합니다. 평균 손실과 CDF 추정 보장을 비대칭적으로 확장하여, 기존에 볼 수 없는 클라이언트와 네트워크에서의 FL 모델의 평가를 가능하게 합니다.



### Improving Model Evaluation using SMART Filtering of Benchmark Datasets (https://arxiv.org/abs/2410.20245)
Comments:
          20 pages, 5 figures

- **What's New**: 본 연구는 NLP(Natural Language Processing) 모델 평가의 효율성을 높이기 위한 새로운 접근 방식을 제안합니다. 기존의 벤치마크 데이터셋에서 저품질의 예제를 체계적으로 제거하여 고품질의 예제를 선별하는 SMART(Selection Methodology for Accurate, Reduced, and Targeted) 필터링 방법론을 도입합니다.

- **Technical Details**: SMART 필터링은 세 가지 기준에 따라 작동합니다: (i) 쉬운 예제 제거, (ii) 데이터 오염이 의심되는 예제 제거, (iii) 임베딩 공간에서 서로 유사한 예제 제거. 이를 통해 모델의 상대 순위를 보존하면서 데이터셋의 크기를 평균 48% 줄이고, ChatBot Arena에서의 Pearson 상관계수를 높일 수 있음을 보여줍니다.

- **Performance Highlights**: SMART 필터링을 사용하여 데이터셋을 약 68.9% 줄이면서도 모델의 상대적 순위에 미치는 영향을 최소화하는 성과를 보였습니다. 필터링된 데이터셋은 ChatBot Arena 등에서의 인간 평가와 보다 강한 상관관계를 보이며, 이는 더 나은 인간 선호도와 일치한 결과입니다.



### Recursive Function Definitions in Static Dataflow Graphs and their Implementation in TensorFlow (https://arxiv.org/abs/2410.20225)
- **What's New**: 이 논문에서는 데이터 흐름 기반 시스템인 TensorFlow에서 재귀 함수 정의를 효율적으로 지원하는 새로운 기술을 제안합니다.

- **Technical Details**: 제안된 접근법은 주어진 재귀 정의를 정적 데이터 흐름 그래프로 변환하고 두 가지 간단하면서도 강력한 데이터 흐름 작업으로 풍부하게 합니다. 이 기술은 태깅(tagging)이라는 개념을 활용하여 데이터와 함께 이동하는 레이블을 통해 데이터 흐름 그래프 내에서 서로 다른 함수 호출을 구분합니다.

- **Performance Highlights**: 실험 결과 재귀의 동적 실행보다 태그 기반 실행이 더 효율적임을 보여줍니다. 또한 이 기술은 심층 학습 응용 프로그램에 매우 중요한 자동 미분(automatic differentiation)에도 적합함을 입증하였습니다.



### Neural Fields in Robotics: A Survey (https://arxiv.org/abs/2410.20220)
Comments:
          20 pages, 20 figures. Project Page: this https URL

- **What's New**: 본 논문은 Neural Fields(NF)가 로봇 공학에서 3D 장면 표현의 획기적인 방법으로 자리 잡고 있음을 강조합니다. 특히, NF의 연속적이고 미분 가능한 표현 방식이 다양한 센서 데이터의 통합 및 새로운 시점 생성에 어떻게 기여하는지를 설명합니다.

- **Technical Details**: Neural Fields는 Occupancy Networks, Signed Distance Fields, Neural Radiance Fields, Gaussian Splatting과 같은 네 가지 주요 프레임워크를 기반으로 합니다. NF는 RGB 카메라, LiDAR, 깊이 센서 등에서 수집된 데이터를 기반으로 최적화되어 고품질 3D 재구성을 생성합니다.

- **Performance Highlights**: Neural Fields는 고품질 3D 재구성, 다중 센서 데이터 통합, 연속적인 컴팩트 표현, 일반화 및 적응력을 제공하여 로봇의 탐색, 물체 조작, 자율 주행 등에서 성능을 크게 향상시킵니다. 또한, NF의 발전은 생성형 AI와 로봇 간의 중요한 연결 고리를 제공하여 이 분야의 연구가 급속히 성장하고 있음을 보여줍니다.



### Looking Beyond The Top-1: Transformers Determine Top Tokens In Order (https://arxiv.org/abs/2410.20210)
- **What's New**: 이 연구는 Transformers의 내부 작용을 이해하는 것이 정확하고 효율적인 예측을 달성하는 데 중요하다는 것을 강조하고 있습니다. 특히, 기존에는 고정된 top-1 예측 후 Transformers에서 수행되는 계산을 살펴보았고, 'saturation event'를 top-k 토큰으로 확장했습니다.

- **Technical Details**: 모델은 top-1 예측이 고정된 후 다음 우선 순위 토큰을 결정하는 과정에서 saturation event가 발생합니다. 이 현상은 다양한 변형 아키텍처(decoder-only, encoder-only, full-Transformer)에서 발생하며, 여러 모달리티(언어, 비전, 음성)에서 공통적으로 관찰됩니다. 우리는 이러한 전환이 task transition 메커니즘에 해당한다고 제안하고, 내부 은닉층의 표현을 통해 각 task의 정보를 예측할 수 있음을 보여줍니다.

- **Performance Highlights**: 새로운 token-level early-exit 전략을 제안하여 기존 방법보다 성능과 효율성의 균형을 더 잘 맞출 수 있음을 입증했습니다. 제안한 방법은 이미 잘 알려진 early exit 기법을 개선하여 text generation 수행 시 더욱 정확한 예측을 유도합니다.



### LLMs Can Evolve Continually on Modality for X-Modal Reasoning (https://arxiv.org/abs/2410.20178)
- **What's New**: PathWeave는 Modal-Path 전환 및 확장 기능을 갖춘 유연하고 확장 가능한 프레임워크로, Multimodal Large Language Models (MLLMs)의 연속적인 모달 진화를 가능하게 합니다. 이를 통해 단일 모달 데이터로 새로운 모달리티에 대한 확장과 학습이 가능합니다.

- **Technical Details**: PathWeave는 Incremental Training Strategy를 바탕으로 하여, Uni-modal 및 Cross-modal Adapter를 통합하여 효율적인 모달 정렬 및 협업을 촉진합니다. MoE (Mixture of Experts) 기반의 게이팅 모듈을 통해 두 가지 유형의 어댑터 간의 멀티모달 상호작용을 향상시킵니다.

- **Performance Highlights**: PathWeave는 최신의 MLLMs와 비교하여 성능 면에서 동등하며, 파라미터 학습 부담을 98.73% 감소시키는 동시에, Continual Learning of Modality (MCL) 벤치마크에서의 높은 정확도를 보여줍니다.



### Cyberbullying or just Sarcasm? Unmasking Coordinated Networks on Redd (https://arxiv.org/abs/2410.20170)
Comments:
          7 pages, 4 figures

- **What's New**: 소셜 미디어 사용이 급증함에 따라, 사용자가 게시물에 대해 빈번하게 풍자적(sarcastic) 코멘트를 작성하는 경향이 나타났습니다. 본 연구는 풍자와 사이버 괴롭힘(cyberbullying)의 구별에 초점을 맞추고 있습니다.

- **Technical Details**: 자연어 처리(NLP) 및 머신러닝(machine learning)을 활용하여 풍자와 해로운 콘텐츠를 구분하기 위한 프레임워크를 제안했습니다. 저희는 Reddit에서 수집한 커스텀 데이터셋을 분석하여 95.15%의 정확도로 해로운 내용과 풍자를 구분하는 데 성공했습니다.

- **Performance Highlights**: 본 연구는 사이버 괴롭힘의 피해를 받는 청소년 및 소수집단의 취약성을 강조하며, 사이버 괴롭힘에 연루된 그룹의 패턴을 분석하여 이들 간의 연관성을 밝혀냅니다. 이러한 연구는 안전한 온라인 커뮤니티를 위한 탐지 능력을 향상하는 데 기여합니다.



### Your Image is Secretly the Last Frame of a Pseudo Video (https://arxiv.org/abs/2410.20158)
Comments:
          18 pages, 7 figures

- **What's New**: 본 논문에서는 디퓨전 모델(Diffusion models)의 자가 감독(self-supervision) 정보를 활용하여 전통적인 이미지 생성 모델의 성능을 향상시킬 수 있는 방법을 제안합니다. 구체적으로는 원본 이미지에 데이터 증강(data augmentation)을 적용하여 생성한 유사 비디오(pseudo video)를 사용하여 비디오 생성 모델로 확장합니다.

- **Technical Details**: 디퓨전 모델은 계층적 변량 자동 인코더(Hierarchical Variational Autoencoders, HVAEs)의 특수한 경우로, 노이즈를 단계적으로 이미지로 변환하는 과정에서 상당히 많은 중간 상태(intermediate states)를 활용합니다. 이 논문에서는 이러한 중간 상태에 자가 감독 정보를 포함시키는 것이 이미지 생성 성능 향상에 중요한 역할을 한다고 가설을 세우고, 유사 비디오를 통해 이를 개선하는 접근법을 탐색합니다.

- **Performance Highlights**: CIFAR10과 CelebA 데이터셋에서 실험적으로 유사 비디오를 통한 이미지 생성 품질이 향상되었음을 확인하였습니다. 이러한 접근법은 VQVAE와 개선된 DDPM 등 다양한 이미지 생성 모델에 적용 가능하며, 성능 개선에 기여할 수 있는 새로운 방법을 제시합니다.



### The inexact power augmented Lagrangian method for constrained nonconvex optimization (https://arxiv.org/abs/2410.20153)
- **What's New**: 본 논문은 비전통적인 inexact augmented Lagrangian 방법을 소개합니다. 여기서 augmenting term는 1과 2 사이의 거듭제곱으로 올린 유클리드 노름(Euclidean norm)으로 설정됩니다.

- **Technical Details**: 제안된 알고리즘은 비선형 동등 제약 조건(nonlinear equality constraints)을 포함한 광범위한 제한 비볼록 최소화 문제에 적용할 수 있습니다. 방법의 전체 복잡성 분석을 수행하였고, Hölder-smooth 하위 문제를 해결하기 위한 가속된 일차 알고리즘(accelerated first-order algorithm)을 활용했습니다. 또한, 이러한 하위 문제를 해결하기 위해 inexact proximal point method를 제시했으며, 이는 개선된 수렴 속도를 보여줍니다.

- **Performance Highlights**: 최악의 경우 복잡성 결과는 augmenting term에 대한 낮은 거듭제곱을 사용할 때 제약 조건 만족(constraint satisfaction)이 더 빨라짐을 나타내지만, 이중 잔여물(dual residual)의 감소는 더 느립니다. 수치 실험을 통해 이론적 발견을 뒷받침하며, 제약 조건 충족과 비용 최소화 간의 trade-off가 특정 실제 문제에 유리함을 보여줍니다.



### AdaNeg: Adaptive Negative Proxy Guided OOD Detection with Vision-Language Models (https://arxiv.org/abs/2410.20149)
Comments:
          NIPS 2024 Camera Ready, Codes are available at \url{this https URL}

- **What's New**: 본 연구에서는 OOD(out-of-distribution) 샘플을 효과적으로 식별하기 위한 새로운 방법으로 	extit{adaptive negative proxies}를 도입합니다. 이는 실제 OOD 이미지를 탐색하여 테스트 중에 동적으로 생성되는 부정 프록시입니다.

- **Technical Details**: 우리의 접근법은 테스트 이미지에서 차별적인 특징을 캐시하는 feature memory bank를 활용하여 OOD 데이터셋에 대해 더 잘 정렬된 프록시를 생성하는 방법을 제안합니다. task-adaptive proxies 및 sample-adaptive proxies를 통해 각각 특징적인 데이터셋과 샘플 수준의 세부 정보를 캡처합니다.

- **Performance Highlights**: AdaNeg는 대규모 ImageNet 벤치마크에서 2.45% AUROC 향상과 6.48% FPR95 감소를 달성하며, 기존 방법들에 비해 우수한 성능을 보여줍니다. 우리의 방법은 training-free 및 annotation-free를 유지하며 빠른 테스트 속도를 자랑합니다.



### CodePurify: Defend Backdoor Attacks on Neural Code Models via Entropy-based Purification (https://arxiv.org/abs/2410.20136)
- **What's New**: 이 연구에서 제안하는 CodePurify는 코드 모델을 위한 새로운 백도어 공격 방어 기법으로, 엔트로피 기반의 정화(entropy-based purification) 과정을 통해 소스 코드 내의 트리거를 정밀하게 탐지하고 제거하며 의미(somatic information)를 유지합니다.

- **Technical Details**: CodePurify는 두 가지 주요 단계로 구성됩니다: 손상된 샘플 탐지 및 트리거 위치 확인(poisoned sample detection and trigger localization), 이후 정화된 코드 생성(purified code generation). 첫 단계에서 신뢰도 기반의 엔트로피 측정을 활용하여 손상된 샘플을 탐지하고 트리거를 찾습니다. 두 번째 단계에서는 Masked Language Model을 사용하여 트리거를 선량한 토큰으로 교체하여 정화된 코드를 생성합니다.

- **Performance Highlights**: CodePurify는 네 가지 첨단 백도어 공격에 대해 세 가지 대표적인 소프트웨어 공학 작업에서 평가되었습니다. 결과적으로 CodePurify는 평균 방어 성능을 개선하여 세 가지 작업에서 각각 40%, 40%, 12%의 공격 성공률 감소를 달성했습니다.



### Near-Optimal Streaming Heavy-Tailed Statistical Estimation with Clipped SGD (https://arxiv.org/abs/2410.20135)
Comments:
          Accepted at NeurIPS 2024

- **What's New**: 이 논문에서는 메모리 제약으로 인해 전통적인 배치 설정보다 훨씬 더 어려운 고차원 heavy-tailed 통계 추정 문제를 다룹니다. 이 문제를 heavy-tailed stochastic gradients를 가진 stochastic convex optimization으로 표현하고, Clipped-SGD 알고리즘이 거의 최적의 하위-Gaussian 통계적 비율을 달성함을 증명합니다.

- **Technical Details**: Clipped-SGD 알고리즘은 smooth 및 strongly convex 목표에 대해 $T$ 샘플을 사용하여 오류 $	ext{E} = 	ext{O}(rac{	ext{Tr}(	ext{Σ})+	ext{sqrt}(	ext{Tr}(	ext{Σ})	ext{||Σ||}_2)	ext{log}(rac{	ext{log}(T)}{	ext{δ}})}{T})$를 달성하며, 여기서 δ는 오류 허용도를 나타냅니다. 또한, 이 결과는 smooth convex 및 lipschitz convex 목표에도 확장됩니다.

- **Performance Highlights**: 이번 결과는 Clipped-SGD의 기존 속도인 $O(rac{	ext{Tr}(	ext{Σ})	ext{log}(rac{1}{	ext{δ}})}{T})$를 개선하며, 새로운 martingale concentration을 위한 반복적인 개선 전략을 제시하여 Catoni와 Giulini의 PAC-Bayes 접근 방식을 향상시킵니다.



### On-Site Precise Screening of SARS-CoV-2 Systems Using a Channel-Wise Attention-Based PLS-1D-CNN Model with Limited Infrared Signatures (https://arxiv.org/abs/2410.20132)
- **What's New**: 이번 연구에서는 저희가 개발한 새로운 방법론을 소개합니다. 이 방법론은 attenuated total reflection-Fourier transform infrared spectroscopy (ATR-FTIR)와 adaptive iteratively reweighted penalized least squares (airPLS) 전처리 알고리즘, 그리고 채널 기반 주의 메커니즘을 따르는 partial least squares one-dimensional convolutional neural network (PLS-1D-CNN) 모델을 통합하여 감염된 개인을 10분 내에 정확하게 선별할 수 있도록 합니다.

- **Technical Details**: 본 연구는 SARS-CoV-2의 초기 발병 단계에서 공개 건강을 위해 제한된 nasopharyngeal swabs을 효율적으로 활용하는 방법을 제안했습니다. ATR-FTIR 신호의 질을 평가하기 위한 biomolecular importance (BMI) 평가 방법과, airPLS를 통한 신호 전처리 및 PLS-1D-CNN 모델을 사용하여 SARS-CoV-2의 감염 상태를 판별합니다. 실험 결과, 저희 모델은 호흡 바이러스 스펙트럼 감지 분야에서 최근 보고된 방법들보다 뛰어난 성과를 거두었습니다.

- **Performance Highlights**: 본 연구에서 개발한 모델은 96.48%의 인식 정확도, 96.24%의 민감성, 97.14%의 특이성, 96.12%의 F1-score, 0.99의 AUC를 달성하여 WHO가 제시한 SARS-CoV-2 감염 검사를 위한 민감도 및 특이성 기준을 충족했습니다.



### ISDNN: A Deep Neural Network for Channel Estimation in Massive MIMO systems (https://arxiv.org/abs/2410.20110)
- **What's New**: 본 논문은 Massive Multiple-Input Multiple-Output (massive MIMO) 기술을 위한 새로운 채널 추정(Channel Estimation, CE) 방법을 제안합니다. 기존의 DNN 구조를 개선하여 Iterative Sequential DNN (ISDNN)이라는 단일 단계 DNN을 개발하였습니다.

- **Technical Details**: ISDNN은 채널 추정 문제를 해결하기 위해 projected gradient descent 알고리즘에 기초한 DNN입니다. 논문에서는 ISDNN의 구조적 확장을 포함한 S-ISDNN을 소개하여 신호의 방향과 안테나 배열 구성과 같은 측면 정보를 통합하여 채널 추정을 향상시키고자 하였습니다.

- **Performance Highlights**: ISDNN은 다른 DNN 기반 CE 방법인 DetNet에 비해 훈련 시간에서 13%, 실행 시간에서 4.6%, 및 정확성에서 0.43 dB 더 뛰어난 성능을 보였으며, S-ISDNN은 훈련 시간에서는 ISDNN보다 더 빠른 성능을 나타내지만, 전체 성능 개선은 추가적으로 필요합니다.



### Super-resolved virtual staining of label-free tissue using diffusion models (https://arxiv.org/abs/2410.20073)
Comments:
          26 Pages, 5 Figures

- **What's New**: 본 연구에서는 Brownian bridge 프로세스를 활용한 확산 모델(diffusion model) 기반의 슈퍼 해상도(super-resolution) 가상 염색(virtual staining) 접근 방식을 제시합니다.

- **Technical Details**: 이 방법은 전통적인 딥러닝(d deep learning) 기반 방법의 한계를 극복하며, 새로운 샘플링 기법(sampling techniques)을 확산 모델에 통합하여 생성된 가상 염색 이미지의 분산(variance)을 크게 줄입니다.

- **Performance Highlights**: 가상 염색 모델은 저해상도(auto-fluorescence) 인간 폐 조직 샘플에 무작위로 적용되었을 때 해상도, 구조적 유사성(structural similarity), 지각적 정확성(perceptual accuracy)에서 지속적으로 기존 방법을 초월하며, 4-5배의 슈퍼 해상도 요인을 달성하여 출력 위상 대역폭(product of spatial bandwidth)을 기존 이미지 대비 16-25배 향상시켰습니다.



### ResAD: A Simple Framework for Class Generalizable Anomaly Detection (https://arxiv.org/abs/2410.20047)
Comments:
          This paper was accepted as a spotlight papaer by NeurIPS 2024

- **What's New**: 이 논문은 클래스 일반화 가능한 이상 탐지(class-generalizable anomaly detection)의 문제를 다루고 있습니다. 다양한 도메인의 새로운 클래스에서 이상을 탐지할 수 있는 통합 AD 모델을 학습하도록 설계된 간단하지만 효과적인 프레임워크 ResAD를 제안합니다.

- **Technical Details**: ResAD는 세 가지 주요 요소로 구성됩니다: 1) Feature Converter는 초기 피처를 잔여 피처로 변환합니다; 2) Feature Constraintor는 정상 잔여 피처를 공간 하이퍼스피어로 제약하여 피처 변화를 줄이고 클래스 간 피처 스케일을 유지합니다; 3) Feature Distribution Estimator는 정상 잔여 피처 분포를 추정하여 이상을 분포 외(out-of-distribution)로 인식합니다.

- **Performance Highlights**: ResAD는 새로운 클래스에서 직접 사용되었을 때 뛰어난 이상 탐지 성능을 발휘하며, 4-shot 정상 샘플만으로도 최첨단 경쟁 방법을 크게 능가하는 결과를 보여줍니다.



### Architectural Flaw Detection in Civil Engineering Using GPT-4 (https://arxiv.org/abs/2410.20036)
- **What's New**: 이 논문은 인공지능(AI)의 응용이 토목 공학 디자인 품질 및 안전성을 향상시키는 혁신적 접근 방식을 제시한다고 설명합니다. 특히, 설계 단계에서 건축 결함을 감지하기 위해 고급 LLM GPT4 Turbo 비전 모델을 사용하고, 누락된 문과 창문을 식별하는 데 중점을 둡니다.

- **Technical Details**: 연구에서는 모델의 성능을 precision, recall, F1 score와 같은 여러 메트릭을 통해 평가하며, AI가 인간 검증 데이터와 비교하여 결함을 정확하게 감지하는 효과성을 보이고 있습니다. 이외에도 AI의 추가적인 능력으로 하중 지지 이슈, 재료 약점 식별, 건축 법규 준수 여부 등을 탐구합니다.

- **Performance Highlights**: AI는 설계 정확성을 현저하게 개선하고 비싼 수정 비용을 줄이며 지속 가능한 관행을 지원함으로써 궁극적으로 더 안전하고 효율적이며 미적으로 최적화된 구조물을 보장하여 토목 공학 분야를 혁신할 수 있는 잠재력을 보입니다.



### Dynamic layer selection in decoder-only transformers (https://arxiv.org/abs/2410.20022)
- **What's New**: 이번 연구는 대량의 Large Language Models (LLMs)에 대한 동적 추론(dynamic inference) 기술을 분석하였으며, 특히 자연어 생성(NLG)에서의 layer skipping과 early exiting 방법에 중점을 두었습니다.

- **Technical Details**: Layer skipping과 early exiting은 LLM의 연산 비용을 줄이기 위해 설계된 동적 추론 방법입니다. 연구에서 layer skipping은 고는 네트워크에서 레이어를 건너뛰는데 더 효과적이며, hidden state 정보를 사용한 per-token 적응은 어려움을 겪습니다. 대안적으로, sequence-level에서 동적 연산 할당을 사용해 significant efficiency gains를 이끌 수 있는 방법을 제안합니다.

- **Performance Highlights**: 연구 결과, layer skipping은 hidden state error를 최소화하는 데 효과적이며, 평균 23.3% Layer만으로도 전체 모델과 동등한 성능을 나타내는 동적 연산 할당 방식을 사용할 수 있음을 발견했습니다.



### GHIL-Glue: Hierarchical Control with Filtered Subgoal Images (https://arxiv.org/abs/2410.20018)
Comments:
          Code, model checkpoints and videos can be found at this https URL

- **What's New**: GHIL-Glue는 인터넷 규모의 데이터에 기반하여 생성된 이미지 및 비디오 예측 모델과 하위 목표 조건 정책을 효과적으로 결합하는 새로운 인터페이스를 제공합니다.

- **Technical Details**: 이 방법은 하위 목표를 필터링하고, 생성된 하위 목표에 포함된 시각적 아티팩트에 대해 목표 조건 정책의 내구성을 향상시키는 두 가지 주요 구성 요소를 포함합니다.

- **Performance Highlights**: GHIL-Glue는 여러 계층적 모델에서 25%의 성능 개선을 달성했으며, 단일 RGB 카메라에서 관찰 정보를 사용하는 CALVIN 시뮬레이션 벤치마크에서 새로운 최첨단 성과를 이뤘습니다.



### Layer by Layer: Uncovering Where Multi-Task Learning Happens in Instruction-Tuned Large Language Models (https://arxiv.org/abs/2410.20008)
Comments:
          Accepted to EMNLP 2024

- **What's New**: 본 연구는 사전 훈련된 대형 언어 모델(LLMs)에서 특정 작업(task) 정보의 인코딩과 명령어 조정(instruction tuning)의 효과를 조사합니다. 60개 이상의 다양한 자연어 처리(NLP) 작업에 대한 분석을 통해, 사전 훈련된 LLM이 이미 반영하고 있는 작업 정보의 범위 및 명령어 조정이 표현을 어떻게 수정하는지를 밝혀냈습니다.

- **Technical Details**: 모델 지향 서브 집단 및 스펙트럼 분석(Model-Oriented Sub-population and Spectral Analysis, MOSSA)이라는 기법을 사용하여 LLM의 표현을 비교합니다. 이는 특정 서브 집단의 훈련 데이터 내에서 모델 표현을 분석하는 대안 기법으로, 제어 모델(control model)과 실험 모델(experimental model)의 표현 차이를 비교하여 작업에 특화된 정보를 분리해낼 수 있습니다. 또한 중심 커널 정렬(Center Kernel Alignment, CKA) 메트릭을 통해 두 모델 간의 표현 유사성을 측정합니다.

- **Performance Highlights**: 연구 결과, LLM에서 고수준 일반 표현(high-level general representations)과 작업 지향 표현(task-oriented representations)으로 전환되는 층을 식별했습니다. 연구는 세 가지 기능 그룹의 층을 밝혀냈습니다: a) 공유층(shared layers), b) 전환층(transition layers), c) 정제층(refinement layers). 이러한 발견은 LLM의 작동 원리를 깊이 이해하는 데 기여하며, 매개변수 효율적 전이 학습(parameter-efficient transfer learning), 다중 작업 학습(multi-task learning), 모델 압축(model compression) 등의 미래 연구에 유망한 시사점을 제공합니다.



### Unsupervised Machine Learning for Detecting and Locating Human-Made Objects in 3D Point Cloud (https://arxiv.org/abs/2410.20006)
- **What's New**: 이 연구는 자연의 나무 구조 내에서 인간이 만든 객체를 탐지하고 식별하는 새로운 작업을 소개합니다. 이 작업은 지면 필터링 단계에서 유도된 비지면 포인트의 하위 집합에서 수행됩니다.

- **Technical Details**: 제안된 방법론은 세 가지 단계로 구성됩니다: (1) Ground Filtering: One-Sided Regression (OSR)을 사용하는 통계적 방법이 도입되어 불균형 지형에서의 한계를 해결합니다. (2) Local Information Extraction (LIE): MPF의 Hessian 행렬을 기반으로 한 커널 기반 방법이 개발되었습니다. (3) Clustering: 결과를 Gaussian Mixture Model (GMM)에 적용하여 비지면 포인트를 나무와 인간이 만든 객체로 분할합니다.

- **Performance Highlights**: 실험 결과, 제안된 지면 필터링 방법이 이전 기술보다 우수한 성능을 나타냈습니다. LIE 방법은 나무와 인간이 만든 객체를 성공적으로 구별할 수 있음을 보여주었습니다.



### Dimension reduction via score ratio matching (https://arxiv.org/abs/2410.19990)
Comments:
          23 pages, 9 figures, 1 table

- **What's New**: 이 논문에서는 gradient-based dimension reduction 기법을 gradient가 없는 상황에서도 활용할 수 있는 새로운 프레임워크를 제안합니다. 이를 통해 Bayesian 추론 문제에서의 고차원 문제를 저차원 문제로 효율적으로 변환할 수 있습니다.

- **Technical Details**: 제안된 방법론은 score ratio function을 학습하는 새로운 목적 함수를 사용하여, log-ratio 두 밀도 측정의 gradient를 근사하는 학습 문제를 도입합니다. 또한, 맞춤형 매개변수화와 정규화 방법을 소개하여 저차원 구조를 활용한 효율적인 모델링을 지원합니다.

- **Performance Highlights**: 제안된 score ratio matching 방법이 기존의 standard score-matching 접근법보다 더 우수한 저차원 구조 식별 성능을 발휘하며, PDE-constrained Bayesian inverse 문제와 조건부 생성 모델링에 대한 유용성을 입증했습니다.



### On-Robot Reinforcement Learning with Goal-Contrastive Rewards (https://arxiv.org/abs/2410.19989)
- **What's New**: 이번 논문에서는 GCR(Goal-Contrastive Rewards)라는 새로운 르인포스먼트 러닝(RL) 방법을 제안하여 로봇이 비디오를 통해 얻은 행동 데이터 없이도 조밀한 보상 함수(dense reward function)를 학습할 수 있도록 하였습니다. 이 방법은 시뮬레이션 환경과 실세계에서 효율적으로 RL을 수행할 수 있는 가능성을 보여줍니다.

- **Technical Details**: GCR은 두 가지 손실 함수(loss function)를 결합한 방법입니다. 첫 번째는 성공적인 궤적을 따라 보상이 증가하는 방식으로 보상 가치(value)를 모델링하는 암묵적 가치 손실(implicit value loss)이며, 두 번째는 성공적인 궤적과 실패한 궤적을 구분하는 목표 대조 손실(goal-contrastive loss)입니다. 결과적으로 GCR은 RL 에이전트의 최근 행동에 적응하여 조형된 보상 함수를 학습합니다.

- **Performance Highlights**: GCR은 1~2시간의 학습 시간을 통해 RL을 실제 환경에서 성공적으로 달성하며, 기존의 희소 보상(sparse rewards)만 사용하는 방법에 비해 두 배 이상의 성과를 보여줍니다. 두 로봇 간의 긍정적인 크로스-엠보디먼트 전이(cross-embodiment transfer)를 통해 인간과 로봇 비디오와 같은 다양한 데이터 출처에서 긍정적인 학습 효과를 보여줍니다.



### Statistical Inference in Classification of High-dimensional Gaussian Mixtur (https://arxiv.org/abs/2410.19950)
Comments:
          22 pages, 4 figures

- **What's New**: 이번 연구에서는 일반 공분산 행렬을 가진 두 개의 가우시안 혼합물의 고차원 분류 문제를 다루었습니다. 표본 크기 $n$과 차원 $p$가 모두 무한히 커지면서 이들의 비율 $eta=n/p$가 고정된 상황에서 일반적인 규제된 볼록 분류기의 비대칭적 행동을 분석했습니다.

- **Technical Details**: 우리는 통계 물리학의 복제 방법(replica method)을 사용하여 고차원 환경에서의 적합성을 탐구하였으며, 이를 통해 더욱 정확한 변수 선택을 위해 적절한 가설 검정 절차를 통합한 디바이어스 추정량(de-biased estimator)을 구축하였습니다.

- **Performance Highlights**: $L_1$-규제가 도입된 로지스틱 회귀를 사례로 활용하여, 이론적 결과가 유한한 크기의 시스템에서의 수치적 시뮬레이션과 일치함을 확인했습니다. 또한 공분산 구조가 디바이어스 추정량 성능에 미치는 영향을 조사했습니다.



### Improving Multimodal Large Language Models Using Continual Learning (https://arxiv.org/abs/2410.19925)
Comments:
          NeurIPS 2024 Workshop on Scalable Continual Learning for Lifelong Foundation Models

- **What's New**: 이 연구는 LLaVA MLLM을 통해 시각 정보를 LLM에 통합할 때 발생하는 언어 이해 및 생성 성능 저하 문제를 지속적 학습(Continual Learning) 문제로 다룹니다. 이는 기존 LLM에 비해 MLLM이 자연어 처리에서 성능이 감소하는 현상을 줄여주는 초기 기법을 모색하고, 이를 통해 MLLM의 시각적 이해를 향상시키는 방법을 제안합니다.

- **Technical Details**: LLaVA MLLM의 통합 과정에서 발생하는 언어적 망각 문제를 해결하기 위해, 5가지 지속적 학습 방법을 검토합니다. 실험을 통해 시각-언어(VL) 작업에서의 성능 저하를 최대 15%까지 줄이는 기법을 도출하였습니다. 언어적 성능 저하를 줄이면서 높은 다중 모달 정확도를 유지하는 방법으로서 Soft Targets, LoRA, mSGM 등의 기술이 활용됩니다.

- **Performance Highlights**: LLaVA의 기존 수치와 비교하여, 제안된 방법이 NLG, NLU, VL 작업의 성능을 개선한 결과를 보여주며, 특히 Soft Targets가 VL 데이터셋에서 가장 높은 정확도를 올리는 것으로 나타났습니다. MLLM의 지속적 학습 실험을 통해 언어적 기술은 유지하면서도 새로운 다중 모달 능력을 성공적으로 습득하였습니다.



### Language Agents Meet Causality -- Bridging LLMs and Causal World Models (https://arxiv.org/abs/2410.19923)
Comments:
          Project page: this https URL

- **What's New**: 본 논문에서는 Causal Representation Learning (CRL)과 대형 언어 모델(Large Language Models, LLMs)의 통합 프레임워크를 제안하여 인과관계를 인식할 수 있는 추론 및 계획을 가능하게 하고 있습니다. 이 프레임워크는 인과 변수를 자연어 표현과 연결된 인과 세계 모델(causal world model)을 학습하여 LLM이 텍스트 형식으로 행동과 상태에 대한 설명을 처리하고 생성할 수 있도록 합니다.

- **Technical Details**: 제안된 프레임워크는 CRL을 통해 환경의 인과 구조를 이해하고 그에 따른 개입과 결과에 대한 추론을 수행합니다. 이 인과 세계 모델은 LLM이 여러 가능한 미래를 평가하고 이를 기반으로 행동을 취할 수 있도록 돕는 시뮬레이터 역할을 합니다. 또한, 텍스트 기반의 행동 표현을 활용하여 다양한 환경에서 보다 직관적인 계획과 추론이 가능합니다.

- **Performance Highlights**: 실험 결과, 인과 인식 방법이 LLM 기반 추론 기법보다 우수한 성능을 보였으며, 특히 더 긴 계획 수립 구간에서는 더욱 싸고 효과적인 결과를 나타냈습니다. 기존 CRL 방법을 사용한 간단한 환경에서의 실험을 통해 이 프레임워크의 효과성을 입증하였습니다.



### Method for noise-induced regularization in quantum neural networks (https://arxiv.org/abs/2410.19921)
Comments:
          11 pages, 6 figures, 2 tables

- **What's New**: 본 연구는 양자 하드웨어의 노이즈(nosie) 수준을 조절하여 양자 신경망(quantum neural networks, QNN)의 데이터 일반화 능력을 향상시키는 방법을 제시합니다. 이 방법은 고전 신경망에서의 정규화(regularisation)와 유사한 원리로 작용합니다. 이를 통해 특정 회귀(task) 문제에서 평균 제곱 오차(mean squared error) 손실을 8% 향상시켰습니다.

- **Technical Details**: 양자 컴퓨팅에서의 decoherence는 시스템이 환경과 상호작용하면서 발생하는 손실을 나타냅니다. 본 연구에서 다룬 주요 양자 노이즈 채널은 세 가지입니다: amplitude-damping, phase-damping, depolarizing. 이들 채널은 양자 처리 유닛(QPU)의 신뢰성을 저해하며, 일반적으로 이러한 노이즈는 문제를 일으키지만 특정 양자 알고리즘에서는 이점으로 작용할 수 있습니다. 특히, QNN에서 데이터 인코딩 과정에 노이즈를 추가하여 오버피팅(overfitting)을 줄이는 사례를 소개합니다.

- **Performance Highlights**: 제안된 방법을 통해, 회귀(task) 문제에서 양자 노이즈를 활용하여 일반화 성능(generalization performance)이 개선되었습니다. 특히 실제 양자 하드웨어에서의 실행 가능성도 논의되며, 노이즈를 조절 가능한 하이퍼파라미터(hyper-parameter)로 사용하여 무노이즈(noiseless) 상황보다 우수한 성能을 달성하는 것이 가능해졌습니다.



### Collaborative Inference over Wireless Channels with Feature Differential Privacy (https://arxiv.org/abs/2410.19917)
Comments:
          This work is under review for possible IEEE publication. arXiv admin note: substantial text overlap with arXiv:2406.00256

- **What's New**: 본 논문에서는 다수의 무선 엣지 디바이스 간 협업 추론을 통해 인공지능(AI) 응용 프로그램의 성능을 향상시키기 위한 새로운 개인 정보 보호 메커니즘을 제안합니다.

- **Technical Details**: 이 접근 방식은 데이터 수집(data acquisition), 특징 추출(feature extraction), 그리고 전송을 위한 특징 인코딩(feature encoding)의 세 가지 단계로 구성됩니다. 이 과정에서 전송되는 민감한 개인 정보의 유출을 방지하기 위해, 각 엣지 디바이스는 중앙 서버에 전송하기 전에 추출된 특징의 개인 정보를 보호합니다. 이 연구는 다양한 시스템 파라미터에 대한 이론적 하한(classification accuracy lower bound)과 특징 품질을 고려한 개인 정보 보호 전송 방식(private feature-aware transmission schemes)을 제안하여, 전송된 특징의 정확도를 향상시킵니다.

- **Performance Highlights**: 제안된 방안은 과거의 전통적인 방법과 비교하여, 통신 오버헤드를 감소시키고, 개인 정보 보호를 강화하며, 효과적인 추론 성능을 유지하는 데 중점을 두고 있습니다. 이를 통해 복잡한 모델을 에지 디바이스에서 실행하는 성능도 향상됩니다.



### Ensembling Finetuned Language Models for Text Classification (https://arxiv.org/abs/2410.19889)
Comments:
          Workshop on Fine-Tuning in Modern Machine Learning @ NeurIPS 2024. arXiv admin note: text overlap with arXiv:2410.04520

- **What's New**: 이 논문에서는 5개의 대형 미세 조정(finetuned) 모델의 예측 결과를 바탕으로 한 메타데이터셋(metadataset)을 제시하며, 이를 통해 다양한 앙상블(ensemble) 전략이 어떻게 미세 조정된 텍스트 분류(text classification) 모델의 성능을 향상시킬 수 있는지를 탐구합니다.

- **Technical Details**: 제시된 메타데이터셋인 FTC(Finetuning Text Classifiers)에는 6개의 서로 다른 데이터셋에 대한 5개의 미세 조정된 모델의 예측이 포함되어 있으며, 다양한 앙상블 방법들이 평가됩니다. 앙상블 모델의 하이퍼파라미터(hyperparameters)로는 모델 유형, 학습률, LoRA 순위 등이 포함되어 있습니다.

- **Performance Highlights**: 연구 결과는 미세 조정된 모델의 앙상블을 통해 텍스트 분류 작업에서 성능 향상이 가능함을 보여주며, 이는 향후 해당 분야에서 앙상블 방법의 사용을 촉진할 것으로 기대됩니다.



### Critical biblical studies via word frequency analysis: unveiling text authorship (https://arxiv.org/abs/2410.19883)
- **What's New**: 본 연구는 성경 저자 간의 언어적 특성을 통계적으로 분석하여 성경 본문의 저자 문제에 대한 새로운 통찰을 제공합니다. 특히, 성경의 첫 아홉 권에 걸친 50개의 장을 조사하였습니다.

- **Technical Details**: 단어 빈도(frequency)를 활용하여 저자별 언어적 특성을 비교하고, 문서 집합(D, DtrH, P) 간의 미세한 차이를 분석합니다. 이를 통해 사전 가정 없이 저자의 정체성을 식별하는 접근법을 사용하였습니다.

- **Performance Highlights**: 첫 두 저자(D와 DtrH)는 서로 유사한 특징을 보였으며, P 저자와는 뚜렷한 차이점을 나타냈습니다. 이러한 결과는 전문가의 평가와 일치하며, 높은 정확도로 저자를 규명할 수 있음을 보여줍니다.



### Parameter-Efficient Fine-Tuning in Large Models: A Survey of Methodologies (https://arxiv.org/abs/2410.19878)
- **What's New**: 이 리뷰 논문은 Parameter-Efficient Fine-Tuning(PEFT)에 대한 종합적인 개요를 제공하며, 대규모 모델이 특정 작업에 적합하도록 조정하는 다양한 알고리즘을 설명합니다. 이 논문은 최근 PEFT 기술의 발전과 응용 분야를 탐구하며, 향후 연구 방향에 대한 제안을 포함합니다.

- **Technical Details**: PEFT는 대규모 사전 훈련된 모델의 매개변수를 새로운 작업이나 시나리오에 적합하게 조정하는 전이 학습(transfer learning) 방법이다. 주요 PEFT 접근 방식으로는 LoRA, adapter tuning, prefix-tuning, prompt-tuning, P-tuning, BitFit 등이 있다. 이 기법들은 기존의 매개변수를 유지하면서도 성능 향상을 목표로 한다.

- **Performance Highlights**: PEFT는 여러 NLP 작업에서 성능을 최적화할 수 있는 잠재력을 가지고 있으며, 수백 개의 PEFT 관련 논문이 출판되었습니다. 이 리뷰는 PEFT 방법론의 현재 이해를 돕고, 앞으로의 연구를 위한 유용한 정보와 통찰을 제공하는 것을 목표로 합니다.



### Predicting potato plant vigor from the seed tuber properties (https://arxiv.org/abs/2410.19875)
- **What's New**: 본 논문에서는 감자 품종의 생장 활력(가장자리 면적)을 분석한 연구결과를 보고합니다. 이 연구는 서로 다른 생산 원출처에서 재배된 감자 씨 감자의 생리状态가 식물의 활력에 미치는 영향을 확인하고자 하였습니다.

- **Technical Details**: 연구는 6가지 감자 품종(Challenger, Colomba, Festien, Innovator, Sagitta, Seresta)을 대상으로 하였으며, 30개의 서로 다른 생산 출처에서의 종자를 분석하였습니다. 주요 기법으로 X-Ray Fluorescence (XRF), Fourier-Transform Infrared Spectroscopy (FTIR), Hyper-Spectral Imaging (HSI), High-Resolution Mass Spectroscopy (HRMS) 등을 사용하였습니다. 이들 기술은 감자 튜브의 비화학적 성분을 정량화하여 식물 활력을 예측하는 모델을 개발하는 데 기여하였습니다.

- **Performance Highlights**: 식물 활력 데이터 간의 Pearson 상관계수 분석을 통해, 적절한 생육 조건에서 서로 다른 시험대에서의 식물들의 활력 간의 일관된 상관관계를 발견하였습니다. 그러나 특정 환경 스트레스의 영향을 받은 경우, 품종별로 상관관계가 감소하거나 부정적인 결과를 나타나는 경우도 있음을 확인하였습니다.



### Breaking the Illusion: Real-world Challenges for Adversarial Patches in Object Detection (https://arxiv.org/abs/2410.19863)
Comments:
          - 21 pages, 17 figures, 7 tables - accepted in 1st Workshop on Enabling Machine Learning Operations for next-Gen Embedded Wireless Networked Devices (EMERGE), 2024

- **What's New**: 이번 연구는 실제 환경에서의 YOLO 객체 탐지 네트워크에 대한 적대적 패치(adversarial patch)의 성능을 분석하고, 패치 크기, 위치, 회전, 밝기 및 색조와 같은 다양한 요소들이 패치의 효과성에 미치는 영향을 조사했습니다.

- **Technical Details**: 연구에서는 글로벌 패치(global patch) 및 로컬 패치(local patch)의 두 가지 공격 방식이 테스트되었습니다. 글로벌 패치는 다양한 환경 조건에서 올바른 탐지를 억제하기 위해 장면 내의 어느 곳에나 배치될 수 있도록 설계된 반면, 로컬 패치는 특정 객체와 부분적으로 겹치도록 설계되었습니다. 성능 평가는 전반적인 평균 정밀도(mean average precision, mAP) 및 탐지 신뢰도(detection confidence)를 기반으로 수행되었습니다.

- **Performance Highlights**: 연구 결과, 다양한 요인들 간의 상관관계가 드러났으며, 실제 환경에서의 공격 효능을 유지하는 데에 어려움이 있음을 강조했습니다. 디지털 환경과 실제 환경 간의 성능 차이는 최대 64%에 달하는 변동성이 있었습니다. 이는 실제 환경이 적대적 공격에 미치는 영향을 이해할 필요성과, 보다 강력한 방어 시스템 개발의 중요성을 강조합니다.



### Personalized Recommendation Systems using Multimodal, Autonomous, Multi Agent Systems (https://arxiv.org/abs/2410.19855)
- **What's New**: 이 논문은 다중 모드(multi-modal) 및 자율적인 다중 에이전트(multi-agent) 시스템을 이용한 고도화된 개인화 추천 시스템에 대해 설명합니다. 특히, Gemini-1.5-pro와 LLaMA-70B와 같은 미래지향적 AI 기술을 통합하여 전자상거래(e-commerce) 고객 서비스 경험을 개선하는 데 중점을 두고 있습니다.

- **Technical Details**: 시스템은 세 가지 에이전트로 구성되어 있습니다. 첫 번째 에이전트는 주어진 질문에 대한 적절한 제품을 추천하고, 두 번째는 추천된 제품에 대한 이미지를 기반으로 후속 질문을 제기하며, 세 번째 에이전트는 자율 검색을 수행합니다. 이 시스템은 실시간 데이터 수집, 사용자 선호 기반 추천 및 적응형 학습 기능을 갖추고 있습니다. Groq API를 사용하여 LPU(대형 프로세싱 유닛) 추론을 수행하며, 응답 시간을 수 밀리초로 줄임으로써 원활한 대화가 가능하게 합니다.

- **Performance Highlights**: 이 시스템은 사용자 맞춤형 쇼핑, 가격 비교 및 이미지 검색 기능을 통해 전자상거래 분야에 혁신적인 변화를 가져올 것으로 기대됩니다. 또한, Chatbot을 통한 문의 자동 해결 기능과 특정 제품에 대한 공공 관심을 기반으로 한 개인화 추천이 가능하여 고객 지원을 완전 자동화할 수 있습니다.



### Dynamic User Grouping based on Location and Heading in 5G NR Systems (https://arxiv.org/abs/2410.19854)
- **What's New**: 이번 논문에서는 5G NR 상용 시스템에서 사용자의 지리적 위치를 기반으로 한 동적 사용자 군집화(dynamic user grouping) 방법을 제안합니다. 이는 Sounding Reference Signals(SRS)를 이용한 채널 핑거프린트를 활용하여 이루어집니다.

- **Technical Details**: 5G 시스템에서 사용자 장비(UE)의 위치 추적을 개선하기 위해, 시간 차(Time of Arrival), 각도 차(Angle of Arrival) 등 기존 신호 처리 기법들을 바탕으로 더욱 정밀한 사용자 위치 추적 모델을 구축하고, 이를 머신러닝(Machine Learning) 기술과 결합하여 채널 조건에 따른 사용자 군집화를 수행합니다.

- **Performance Highlights**: 이 연구는 높은 정확도의 위치 추적 모델을 통해 5G 네트워크의 리소스 최적화, 이동성 관리 개선, QoS 품질 향상, 네트워크 설계 등 여러 기능에 긍정적인 영향을 미칠 것으로 기대됩니다.



### AEPL: Automated and Editable Prompt Learning for Brain Tumor Segmentation (https://arxiv.org/abs/2410.19847)
Comments:
          4 pages paper for ISBI2025

- **What's New**: 본 논문에서는 뇌 종양 분할을 위한 새로운 프레임워크인 Automated and Editable Prompt Learning (AEPL)을 제안합니다. AEPL은 다중 작업 학습(multi-task learning)과 프롬프트 학습(prompt learning)을 결합하여 종양의 등급을 분할 과정에 통합합니다. 이를 통해 수작업 프롬프트 입력 없이도 보다 정확한 분할을 가능하게 합니다.

- **Technical Details**: AEPL은 3D U-Net 구조를 기반으로 하며, U-Net 인코더, 종양 등급 분류기, 프롬프트 인코더 및 U-Net 디코더로 구성됩니다. AEPL은 다중 모달리티 MRI 입력을 처리하여 종양 등급 예측 및 분할 마스크 생성을 동시에 수행합니다. 예측된 종양 등급은 자동으로 생성된 프롬프트로 사용되어 분할 마스크를 생성하는데 도움을 줍니다.

- **Performance Highlights**: AEPL은 BraTS 2018 데이터세트를 사용하여 여러 최신 방법들보다 우수한 성능을 보여주었으며, 종양 등급 정보를 직접 통합하여 세밀한 분할 결과를 생성하였습니다. 비록 다수의 추가 실험이 필요하지만, AEPL은 뇌 종양 분할에 있어 높은 유용성과 임상적 가치를 갖고 있음을 입증하였습니다.



### Enhancing Trust and Safety in Digital Payments: An LLM-Powered Approach (https://arxiv.org/abs/2410.19845)
Comments:
          10 pages, 7 figures

- **What's New**: 디지털 결제 시스템에서 사기 탐지 메커니즘의 중요성이 강조되고 있으며, 인도의 통합 결제 인터페이스(UPI)를 사례로 하여 Google Pay (GPay)를 중심으로 한 접근 방식이 제안되었습니다.

- **Technical Details**: 이 연구는 대형 언어 모델(LLMs)을 활용하여 사기 분류의 정확성을 높이고, 이를 통해 인간 리뷰어가 사기 활동을 식별하고 완화할 수 있도록 돕는 디지털 어시스턴트를 설계했습니다. Gemini Ultra 모델은 사기 분류에서 93.33%의 정확도를 달성했으며, 분류에 대한 이유를 생성하는 정확도는 89%로 나타났습니다.

- **Performance Highlights**: 이 연구의 결과는 LLMs가 기존의 머신 러닝 모델을 보강하여 사기 검토의 효율성, 정확성, 품질 및 일관성을 높일 수 있는 잠재력을 보여주었습니다. 특히, LLM은 새로운 사기의 정확한 이유를 32% 발견하여 인간 리뷰어가 간과했던 부분을 보완했습니다.



### A practical, fast method for solving sum-of-squares problems for very large polynomials (https://arxiv.org/abs/2410.19844)
- **What's New**: 이번 연구에서는 기존의 Sum of Squares (SOS) 최적화 문제를 해결하기 위한 새로운 접근법을 제안합니다. 전통적인 방법이 Convex (볼록) 문제인 Semidefinite Program (SDP)으로 변환하여 해결하려 했던 반면, 본 연구에서는 Polymorphic Neural Network에 기반하여 비볼록(non-convex) 문제로 변경하여 해결합니다.

- **Technical Details**: 제안된 방법은 비볼록 및 과도하게 파라미터화된 모델로, 일반적인 단일 Gradient Descent가 아닌 빠른 수렴을 달성할 수 있도록 합니다. 우리가 설계한 네트워크는 SOS 다항식을 생성하도록 구성되며, 주어진 목표 다항식이 네트워크의 출력일 수 있는지를 판단합니다. 이 과정에서 NP-Complete (비결정적 다항시간 완전문제) 문제를 다룰 수 있습니다.

- **Performance Highlights**: 우리의 방법은 수백만 개의 계수를 가진 다항식 문제를 처리할 수 있으며, 기존의 SDP 기반 처리 방법보다도 더 효율적인 결과를 제공합니다. 실험 결과, 일반적인 다항식에서 항상 올바른 Global Minimum (전역 최소값)으로 수렴하였으며, 이 때 소요된 시간은 다항식 계수 수에 비례해 선형적(Linear)으로 증가했습니다.



### Artificial intelligence for partial differential equations in computational mechanics: A review (https://arxiv.org/abs/2410.19843)
- **What's New**: 이 논문은 인공지능(AI)과 전통 과학의 통합, 특히 부분 미분 방정식(PDE)을 해결하기 위한 AI 알고리즘의 적용에 대한 포괄적 리뷰를 제공합니다. AI for PDEs는 데이터와 PDE의 융합을 통해 거의 모든 PDE를 해결할 수 있는 새로운 접근 방식을 제시합니다.

- **Technical Details**: AI for PDEs의 핵심 알고리즘에는 Physics-Informed Neural Networks (PINNs), Deep Energy Methods (DEM), Operator Learning, Physics-Informed Neural Operator (PINO)가 포함됩니다. 이 방법들은 많은 양의 데이터를 활용하여 과학적 시뮬레이션을 수행하여 기존 알고리즘보다 과도한 계산 없이 특정 문제에 대한 근사 솔루션을 제공합니다.

- **Performance Highlights**: AI for PDEs는 전통적인 수치 알고리즘을 가속화할 수 있는 잠재력을 가지고 있으며, 특히 비선형 문제 및 고차원 문제 해결에 있어 뛰어난 성능을 보여줍니다. 하지만 PINNs는 적용하는 많은 상황에서 강인성, 정확성 및 계산 효율성이 떨어질 수 있어 이와 관련된 개선이 필요합니다.



### Contrastive random lead coding for channel-agnostic self-supervision of biosignals (https://arxiv.org/abs/2410.19842)
- **What's New**: 이 논문에서는 바이오 신호의 채널 독립적(self-supervision) 훈련을 위한 긍정 쌍(positive pairs)의 생성 전략을 조사합니다. 새로운 방법인 Contrastive Random Lead Coding (CRLC)를 소개하여, 무작위 채널 집합을 사용하여 긍정 쌍을 생성하는 방식을 제안합니다.

- **Technical Details**: CRLC는 다양한 입력 채널을 활용하여 바이오 신호의 긍정 쌍을 생성하는 반면, 기존 접근법은 보통 증강(augmentations)이나 이웃 시간 세그먼트를 사용합니다. 우리는 EEG 및 ECG 데이터를 통해 모델을 사전 훈련(pre-training)하고 다운스트림(task)에 맞게 미세 조정(fine-tuning)했습니다.

- **Performance Highlights**: CRLC는 채널 독립적인 설정에서 경쟁 전략들을 초월하는 성능을 보여주며, EEG 작업에서는 현재의 최첨단(reference) 모델을 초과 성능을 달성했습니다. ECG 작업에서는 CRLC를 적용하여 유사한 결과를 성취할 수 있었습니다.



### Non-invasive Neural Decoding in Source Reconstructed Brain Spac (https://arxiv.org/abs/2410.19838)
Comments:
          21 pages, 5 figures, 14 tables, under review

- **What's New**: 본 연구는 비침습적(fMRI, MEG/EEG) 뇌파 디코딩 기법에서의 데이터 융합 및 모델링의 어려움을 극복하기 위한 새로운 접근법을 제시합니다. 연구자들은 MEG 데이터로부터 복셀(voxel) 기반으로 신경 활동을 재구성하는 기존 기법을 활용하여 훈련 및 일반화의 효율성을 높입니다.

- **Technical Details**: 연구팀은 센서에서 소스 공간으로의 변환 과정을 통해 MEG 데이터의 신경 활동을 3D 복셀 그리드로 재구성하는 기법을 적용했습니다. 이를 통해 데이터를 보다 구조화된 형태로 처리할 수 있게 되어, 공간적(inductive biases) 데이터 증강, 해석 가능성, 제로샷(generalisation) 일반화 등을 가능하게 합니다.

- **Performance Highlights**: 본 연구에서 제안한 방법은 기존의 방법보다 더욱 효과적인 성능을 보였으며, 다양한 데이터셋에서의 훈련과 통합이 가능 denoising을 통한 성능 향상을 유도합니다. 여러 실험 결과는 소스 공간에서의 디코딩이 센서 공간에서의 디코딩보다 더 높은 정확도를 기록한다는 점을 개선한 것으로 나타났습니다.



### Automatic Classification of Sleep Stages from EEG Signals Using Riemannian Metrics and Transformer Networks (https://arxiv.org/abs/2410.19819)
- **What's New**: 본 연구에서는 EEG (electroencephalographic) 신호의 수면 단계 분류를 위한 혁신적인 신경망 모델 SPDTransNet을 도입하였습니다. 이 모델은 공분산 행렬의 시계열을 통해 수면 단계를 분류하는 데 중점을 두고 있습니다.

- **Technical Details**: SPDTransNet은 Transformer 구조에서 파생된 네트워크이며, 신호의 학습된 특징을 공분산 행렬에 통합하는 새로운 방식을 사용합니다. 또한, 이 방식은 대칭 양의 정부호(Symmetric Definite Positive, SPD) 성질을 유지하면서 가능해졌습니다.

- **Performance Highlights**: 우리의 모델은 다양한 최신 모델(State-of-the-Art)과 비교하였고, 단일 데이터셋 및 멀티 데이터셋에서 최적화된 성능을 보여주었습니다. 특히, 멀티 데이터셋 작업에 대한 적응력이 뛰어난 것으로 입증되었습니다.



### UniMTS: Unified Pre-training for Motion Time Series (https://arxiv.org/abs/2410.19818)
Comments:
          NeurIPS 2024. Code: this https URL. Model: this https URL

- **What's New**: 본 연구에서는 UniMTS라는 새로운 통합된 사전 훈련(pre-training) 절차를 도입합니다. 이는 모바일 및 웨어러블 디바이스로부터 수집된 모션 시계열 데이터의 다양한 변수를 아우르는 첫 번째 모델입니다.

- **Technical Details**: UniMTS는 대조 학습(contrastive learning) 프레임워크를 기반으로 하여 모션 시계열과 대량 언어 모델(large language models)로 풍부해진 텍스트 설명 간의 정렬을 수행합니다. 기존의 모션 스켈레톤 데이터를 바탕으로 시계열을 합성하며, 공간-시간 그래프 네트워크(spatio-temporal graph networks)를 통해 관절 간의 관계를 캡처하고 다양한 장치 위치에 대한 일반화를 지원합니다.

- **Performance Highlights**: UniMTS 모델은 18개의 모션 시계열 분류 벤치마크 데이터셋에서 우수한 일반화 성능을 보여주며, 제로샷(zero-shot) 설정에서 340%의 성능 향상, 몇 샷(few-shot) 설정에서 16.3%, 전체 샷(full-shot) 설정에서 9.2%의 성능 개선을 달성했습니다.



### BUNDL: Bayesian Uncertainty-aware Deep Learning with Noisy training Labels for Seizure Detection in EEG (https://arxiv.org/abs/2410.19815)
- **What's New**: 이 논문은 스케일p-EEG를 사용하여 자율적 간질 발작 감지 및 발작 시작 부위 로컬리제이션을 위한 새로운 통계적 프레임워크인 BUNDL(Bayesian UncertaiNty-aware Deep Learning)을 소개합니다. 이 방법은 레이블 모호성(label ambiguity)을 고려하여 고품질 라벨이 불확실할 때에도 딥러닝 모델의 성능을 향상시킵니다.

- **Technical Details**: BUNDL은 노이즈가 있는 훈련 레이블을 사용하여 딥 뉴럴 네트워크를 훈련하는 간단하고 모델에 구애받지 않는 방법을 제공합니다. 특히 통계적 프레임워크에 도메인 지식을 통합하여 KL-divergence 기반의 새로운 손실 함수(loss function)를 도출해 EEG로부터 발작 특성을 더 잘 학습하게 합니다. BUNDL은 후속 평가 및 검증을 위해 시뮬레이션된 EEG 데이터셋과 TUH, CHB-MIT와 같은 공개 데이터셋에서 검증되었습니다.

- **Performance Highlights**: BUNDL은 7가지 레이블 노이즈 유형 및 3가지 EEG 신호 대 노이즈 비율에서 세 가지 기본 모델의 성능을 일관되게 향상시킵니다. 실세계의 TUH 및 CHB-MIT 데이터셋에서도 유사한 개선 사항이 관찰되었으며, BUNDL은 발작 시작 부위 로컬리제이션의 정확성을 개선하는 데 기여합니다.



### ControlAgent: Automating Control System Design via Novel Integration of LLM Agents and Domain Expertis (https://arxiv.org/abs/2410.19811)
- **What's New**: ControlAgent는 기존의 인공지능 모델에 비해 제어 시스템 디자인의 자동화를 위한 혁신적인 접근 방식을 제공합니다. 이를 통해 전문적인 지식과 반복적 디자인 프로세스를 융합하여 인간의 개입 없이도 고품질의 제어 시스템을 설계할 수 있습니다.

- **Technical Details**: ControlAgent는 여러 개의 협력적인 LLM(대형 언어 모델) 에이전트를 통합합니다. 중앙 에이전트는 작업 분배를 담당하고, 각 작업에 특정한 에이전트는 다양한 시스템 및 요구사항에 대해 세부 제어 설계를 맡습니다. Python 기반의 계산 에이전트가 복잡한 계산을 수행하며, 이 모든 과정은 이전 디자인으로부터의 실시간 피드백을 통해 반복적으로 개선됩니다.

- **Performance Highlights**: ControlAgent는 전통적인 인간 개입 기반 디자인 방식에 비해 높은 성능과 견고성을 입증했습니다. 500개의 제어 과제를 포함한 ControlEval 데이터셋을 통해 테스트된 결과, ControlAgent가 LLM 기반 및 전통적인 도구 기반 방법들보다 우수하다는 것이 확인되었습니다.



### Training Compute-Optimal Vision Transformers for Brain Encoding (https://arxiv.org/abs/2410.19810)
- **What's New**: 이 연구는 신경 인코딩(Brain Encoding)에서 비전 트랜스포머(Vision Transformer)의 최적 훈련을 위해 모델 크기, 데이터 크기 및 컴퓨팅 리소스 간의 관계를 탐구합니다.

- **Technical Details**: 비디오에서 효과적인 시공간 특성을 추출하기 위해 VideoGPT를 사용하고, 이 특성을 기반으로 뇌 활동을 예측하는 Ridge 모델을 훈련했습니다. 다양한 데이터 크기(10k, 100k, 1M, 6M) 및 GPT-2의 여러 모델 구성(히든 레이어 차원, 레이어 수, 어텐션 헤드 수)에 대한 성능을 평가했습니다. 또한 32비트와 16비트 부동 소수점 표현의 영향을 비교했습니다.

- **Performance Highlights**: 히든 레이어 차원을 증가시키는 것이 뇌 인코딩 성능을 유의미하게 개선하며, 데이터 크기를 늘리면 인코딩 성능이 향상되는 것으로 나타났습니다. 16비트로 훈련했을 때 32비트와 동일한 정확도를 유지하면서 훈련 시간을 1.17배 단축했습니다.



### Comparing Surface Landmine Object Detection Models on a New Drone Flyby Datas (https://arxiv.org/abs/2410.19807)
Comments:
          9 pages, 22 figures, 7 tables

- **What's New**: 이번 연구는 드론 영상을 이용한 지뢰 탐지에 대한 철저한 데이터셋을 생성하고, 최신 딥러닝 기반 객체 탐지 알고리즘을 활용하여 지뢰 탐지 성능을 비교한 것이 특징입니다.

- **Technical Details**: 연구에서는 POM-2 및 POM-3 러시아 표면 지뢰의 드론 이미지로 구성된 커스텀 데이터셋을 만들었습니다. YOLOF, DETR, Sparse-RCNN, VFNet 등의 4가지 컴퓨터 비전 기반 모델을 훈련, 테스트하여 비교하였습니다.

- **Performance Highlights**: YOLOF 모델은 mAP(Mean Average Precision) 스코어 0.89로 다른 모델을 초월하는 성능을 보여주었으며, DETR, VFNET, Sparse-RCNN의 mAP 스코어는 약 0.82로 나타났습니다. YOLOF는 또한 Nvidia V100 컴퓨팅 클러스터에서 56분의 짧은 훈련 시간으로 훈련되었습니다.



### The Useful Side of Motion: Using Head Motion Parameters to Correct for Respiratory Confounds in BOLD fMRI (https://arxiv.org/abs/2410.19802)
Comments:
          3 pahes, 1 Figure, 2024 ISMRM Workshop on Motion Correction in MR, 03-06 September 2024, Québec City, QC, Canada. Abstract Number 23

- **What's New**: 이 연구는 기능적 자기 공명 영상(fMRI)에서 호흡 변동(RV)의 추정을 개선하기 위한 새로운 기계 학습 방법을 제안합니다. 특히 원시(raw)와 대역 통과(bandpass) 필터링된 헤드 모션 파라미터를 통합하여 RV 재구성 정확도에 미치는 영향을 조사합니다.

- **Technical Details**: 연구에서는 1차원 합성곱 신경망(1D-CNNs)을 이용하여 헤드 모션 데이터를 활용하여 호흡에 의해 유도된 변화를 보다 견고하게 추정하는 방법을 모색합니다. 전통적인 필터링 기법의 한계를 극복하고, 호흡 주파수 대역에서의 헤드 모션 데이터를 통합하여 RV 추정을 개선하려고 합니다.

- **Performance Highlights**: 상용 필터들보다 향상된 RV 추정 정확도를 달성하여 fMRI 데이터에서 호흡과 관련된 유용한 정보를 보다 효과적으로 획득할 수 있는 방법을 제공합니다.



### Radon Implicit Field Transform (RIFT): Learning Scenes from Radar Signals (https://arxiv.org/abs/2410.19801)
Comments:
          A version of this work is under review as a submission to ICLR 2025 Conference

- **What's New**: 이번 연구에서는 레이다 신호에서의 장면 표현을 학습할 수 있는 첫 번째 방법인 Radon Implicit Field Transform (RIFT)를 제안합니다. RIFT는 레이다 신호로부터 학습한 Implicit Neural Representations (INRs)을 활용하여 기존의 ASP 문제를 해결합니다.

- **Technical Details**: RIFT는 Generalized Radon Transform (GRT)라는 레이다 전방 모델과 INR 기반의 장면 표현을 결합하며, 이를 통해 레이다 신호의 재구성 오류를 최소화하는 방식으로 모델을 훈련합니다. 이 과정에서 p-RMSE와 m-SSIM이라는 새로운 오류 메트릭을 도입하였습니다.

- **Performance Highlights**: RIFT 모델은 전통적인 장면 모델에 비해 최대 188%의 장면 재구성 개선을 이루며, 기존 데이터의 10%만으로도 재구성 성능을 3배 향상시키고, 새로운 관점에서의 일반화에서도 10% 개선된 결과를 보였습니다.



### Feature Clipping for Uncertainty Calibration (https://arxiv.org/abs/2410.19796)
- **What's New**: 이 논문은 Deep Neural Networks (DNNs)의 신뢰할 수 있는 model calibration (모델 칼리브레이션)을 위한 새로운 post-hoc calibration 방법인 Feature Clipping (FC)을 제안합니다. 이는 DNN이 예측에서 과신에 빠지는 문제를 해결합니다.

- **Technical Details**: Feature Clipping은 feature 값을 특정 threshold (임계값)로 잘라내어 calibration error (칼리브레이션 오류)가 큰 샘플의 entropy (엔트로피)를 증가시키고, 적은 칼리브레이션 오류를 가진 샘플의 정보를 유지하는 방법입니다. 이 방법은 CIFAR-10, CIFAR-100, ImageNet과 같은 데이터셋에서 여러 모델에 걸쳐 실험하여 일관되게 모델의 칼리브레이션 성능을 향상시킵니다.

- **Performance Highlights**: FC는 기존의 post-hoc 및 train-time calibration 방법에 비해 일관된 성능 향상을 보여줍니다. FC는 calibration의 새로운 접근법을 제시하며, feature modification (특징 수정)에 기반한 최초의 칼리브레이션 방법으로, 여러 데이터셋과 모델에서 SOTA (state-of-the-art) 칼리브레이션 성능을 달성했습니다.



### DiffGAN: A Test Generation Approach for Differential Testing of Deep Neural Networks (https://arxiv.org/abs/2410.19794)
- **What's New**: 본 논문에서는 Deep Neural Networks (DNNs)의 차별적 테스트를 위한 새로운 기법인 DiffGAN을 제안합니다. DiffGAN은 Generative Adversarial Network (GAN)와 Non-dominated Sorting Genetic Algorithm II (NSGA-II)를 활용하여 다양한 트리거 입력(triggering inputs)을 생성하여 모델 간의 행동 차이를 나타냅니다.

- **Technical Details**: DiffGAN은 블랙 박스(black-box) 테스트 이미지 생성 접근법으로, DNN 모델 간의 행동 차이를 드러내기 위한 유효하고 다양한 입력을 생성합니다. 이 방식은 두 가지 맞춤형 피트니스 함수(fitness functions)를 활용하여 GAN 입력 공간을 탐색하고 모델 출력 간의 차이를 식별합니다.

- **Performance Highlights**: DiffGAN은 기존 SOTA(상태 최첨단) 기준인 DRfuzz보다 4배 많은 트리거 입력을 생성하며, 보다 높은 다양성과 유효성을 보입니다. 생성된 입력은 입력 특성에 바탕을 두고 최적의 모델을 선택할 수 있도록 하는 기계 학습 기반 모델 선택 메커니즘을 개선합니다.



### Substance Beats Style: Why Beginning Students Fail to Code with LLMs (https://arxiv.org/abs/2410.19792)
- **What's New**: 이 논문은 LLMs(대규모 언어 모델)이 프로그래머의 생산성을 높이고 있지만, 초보자들이 LLM에게 효과적으로 프로그래밍 문제를 제시하는 데 어려움을 겪고 있다는 점에 주목합니다. 연구는 초보자와 LLM 간의 소통 불일치 문제의 원인으로 두 가지 가설을 탐구합니다: 첫째, 학생들이 좋은 프롬프트를 작성하는 데 필요한 기술적 어휘가 부족하다; 둘째, 학생들이 LLM이 코드 생성 작업을 해결하기 위해 필요한 정보의 정도를 이해하지 못한다는 것입니다.

- **Technical Details**: 이 연구는 80명의 학생들에 의해 작성된 1,749개의 프롬프트 데이터셋을 사용하여, 기술적 어휘의 변화가 프롬프트의 성공률에 미치는 영향을 조사했습니다. 언어적 변화의 영향을 분리하기 위해, 기술 용어를 비슷한 의미의 단어로 교체하는 인과 분석을 수행했습니다. 또한, 문제 해결 시 학생들이 프롬프트에 도입한 문제별 "단서"를 주석 달아 정보를 선택하는 과정도 분석했습니다.

- **Performance Highlights**: 연구 결과, 학생들이 LLM 코드 생성 문제에서 겪는 어려움은 기술적 어휘의 어려움보다 관련 정보 선택의 어려움에서 비롯된다는 것을 알게 되었습니다. 단서가 누락된 프롬프트는 거의 항상 실패하는 경향을 보였으며, 학생들은 중요하지 않은 편집에 고착되어 정보 내용의 변경 없이 반복하는 경우가 많았습니다. 기술적 어휘의 교정이 프롬프트 성공률을 크게 개선하지 않는 것으로 나타났습니다.



### Data-Driven Cellular Network Selector for Vehicle Teleoperations (https://arxiv.org/abs/2410.19791)
Comments:
          IEEE Network of Future 2024

- **What's New**: 이 논문에서는 새로운 알고리즘인 Active Network Selector (ANS)를 제안하며, 이를 통해 자율차(AV)가 여러 셀룰러 네트워크에 연결된 상태에서 비디오 패킷을 전송하는 최적의 경로를 실시간으로 선택할 수 있는 방법을 설명합니다.

- **Technical Details**: 논문에서 다루는 주요 기술적 요소는 패킷 손실(packet loss), 지연(latency), 그리고 셀룰러 네트워크의 성질입니다. ANS 알고리즘은 시계열(time series) 기계 학습(machine learning) 접근 방식을 사용해 선택된 네트워크를 기반으로 패킷 전송을 최적화합니다. 세 가지 기계 학습 알고리즘인 LossPredict, HandPredict, LatencyPredict가 제안되어 각각 패킷 손실, 핸드오버(handover), 지연을 예측합니다.

- **Performance Highlights**: ANS는 기존의 비학습(baseline non-learning) 알고리즘에 비해 80% 이상의 패킷 손실 예측 정확도를 기록하며, 핸드오버 빈도 및 각 네트워크의 사용자 신호 품질과 강한 상관관계를 보여줍니다. 다양한 테스트 드라이브 결과 ANS가 패킷 손실 및 지연을 크게 감소시킬 수 있음을 입증하였습니다.



### Telco-DPR: A Hybrid Dataset for Evaluating Retrieval Models of 3GPP Technical Specifications (https://arxiv.org/abs/2410.19790)
- **What's New**: 이 논문에서는 3GPP(3rd Generation Partnership Project) 기술 문서를 활용한 통신 분야의 Q&A 시스템을 제안합니다. 또한, 텍스트와 테이블을 혼합한 형태의 curated 3GPP 코퍼스를 포함하는 혼합 데이터셋인 Telco-DPR를 소개합니다.

- **Technical Details**: 제안된 QA 시스템은 RAG(Retriever-Augmented Generation) 기술을 사용하며, DHR(Dense Hierarchical Retrieval) 모델로 계층형 패시지를 선택하여 문서와 패시지 수준에서 미세 조정을 통해 관련 정보를 추출합니다. 주요 평가 지표로는 Top-K 정확도와 Mean Reciprocal Rank (MRR)를 사용하여 성능을 검증합니다.

- **Performance Highlights**: DHR 모델은 관련 기술 정보를 검색하는 데 있어 기존 방법보다 우수한 성과를 나타내며, Top-10 정확도 86.2%를 달성하였습니다. RAG 모델과 GPT-4를 활용한 QA 시스템은 기존 데이터셋에 비해 14%의 답변 정확도 개선을 보였습니다.



### Xeno-learning: knowledge transfer across species in deep learning-based spectral image analysis (https://arxiv.org/abs/2410.19789)
Comments:
          Jan Sellner and Alexander Studier-Fischer contributed equally to this work

- **What's New**: 이 논문에서는 'xeno-learning'이라는 새로운 개념을 제안합니다. 이는 한 종에서 다른 종으로 지식을 전이하는 방식으로, 특히 고유한 임상 데이터를 수집하기 어려운 상황에서 동물 데이터의 활용을 극대화하려는 접근법입니다.

- **Technical Details**: 이 연구는 11,268장의 hyperspectral imaging (HSI) 이미지를 사용하여 사람, 돼지, 쥐 모델 간의 스펙트럼 특성을 비교합니다. 기존의 모델에서는 한 종에서 훈련된 신경망이 다른 종의 조직을 분류하는 데 어려움을 겪으므로, 새로운 'physiology-based data augmentation' 방법을 통해 서로 다른 종 간의 상대 스펙트럼 변화를 학습하고 전이할 수 있는 방안을 제시합니다.

- **Performance Highlights**: 조직 분별 성능은 동물에서 학습한 데이터를 통해 향상되며, 동물 데이터와 인간 데이터를 결합한 신경망 훈련이 유의미한 성과를 보이지 않았습니다. 하지만 상대 스펙트럼 변화를 학습함으로써 특정 병리학적 상태에서의 신경망 성능을 개선할 수 있는 가능성을 보여줍니다.



### Multi-modal Image and Radio Frequency Fusion for Optimizing Vehicle Positioning (https://arxiv.org/abs/2410.19788)
- **What's New**: 이 논문에서는 채널 상태 정보(Channel State Information, CSI)와 이미지를 결합하여 차량을 공동으로 위치 추정하는 다중 모달(vehicle positioning framework) 접근법을 제시합니다.

- **Technical Details**: 특히 이 연구에서는 차량이 하나의 기지국(BS)와만 통신할 수 있는 야외 시나리오를 고려합니다. 각 BS는 CSI의 일부 레이블은 붙어 있지만, 많은 수의 비레이블 CSI와 카메라로 촬영한 이미지의 수집을 가능하게 하는 카메라 세트로 장착되어 있습니다. 레이블이 없는 CSI 데이터를 활용하기 위해 메타-학습(meta-learning) 기반의 하드 기대 최대화(expectation-maximization, EM) 알고리즘이 설계되었습니다. 레이블이 없는 CSI와 여러 차량 위치 간의 관계가 불분명하므로, 학습 목적 함수를 최소 매칭 문제로 형식화했습니다.

- **Performance Highlights**: 시뮬레이션 결과, 제안된 방법은 이미지를 사용하지 않고 CSI 핑거프린트만으로 차량 위치 추정을 하는 기준선과 비교하여 위치 오차를 최대 61%까지 줄일 수 있음을 보여주었습니다.



### Leveraging Multi-Temporal Sentinel 1 and 2 Satellite Data for Leaf Area Index Estimation With Deep Learning (https://arxiv.org/abs/2410.19787)
- **What's New**: 본 논문에서는 생태계 건강과 식물 동태를 이해하기 위해 중요한 매개변수인 Leaf Area Index (LAI)를 예측하는 새로운 방법을 제안합니다. 이 방법은 Sentinel 1 레이더 데이터와 Sentinel 2 다중 스펙트럼 데이터의 보완 정보를 활용하여 픽셀 단위로 LAI를 예측합니다.

- **Technical Details**: 이 방법은 다중 U-net을 기반으로 한 딥 뉴럴 네트워크(dnn)를 사용하여 설계되었습니다. 서로 다른 입력 모달리티(modality)의 복잡성을 다룰 수 있도록 여러 모듈들로 구성되어 있으며, 각 모듈은 공통 잠재 공간(latent space)을 나타내기 위해 개별적으로 사전 훈련(pre-training)됩니다. 이후 계절성(seasonality)을 고려한 공통 디코더로 엔드 투 엔드(end-to-end) 방식으로 미세 조정(fine-tuning)을 진행합니다.

- **Performance Highlights**: 이 방법은 공개 데이터에서 0.06 RMSE와 0.93 R2 점수를 달성하였습니다. 또한, 향후 연구의 발전을 위해 기여한 내용을 해당 링크에서 제공하였습니다.



### How to Backdoor Consistency Models? (https://arxiv.org/abs/2410.19785)
- **What's New**: 일관성 모델(Consistency Models)이라는 새로운 생성 모델 클래스의 백도어 공격(vulnerability to backdoor attacks) 연구가 진행되었습니다. 이 모델들은 노이즈를 데이터로 직접 매핑하여 이미지를 생성하는 방식으로, 단일 단계에서 이미지를 생성할 수 있습니다.

- **Technical Details**: 이 연구에서는 일관성 모델의 백도어 공격에 대한 취약성을 분석하였습니다. 기존의 확산 모델에 대한 연구와는 달리, 백도어 훈련 프로세스가 고유한 일관성 모델의 훈련 방식과 거기서 파생된 목표를 고려합니다. 프레셰 발달 거리(Fréchet Inception Distance, FID)를 이용하여 공격 모델과 비공격 모델 간의 이미지 품질을 비교했습니다.

- **Performance Highlights**: 실험 결과, 제안한 프레임워크는 다양한 트리거(trigger) 및 타겟(target) 설정에서도 일관성 모델이 공격을 받을 수 있음을 보여주었습니다. Gaussian 노이즈를 트리거로 사용하면 눈에 띄지 않으면서도 효과적인 결과를 얻을 수 있었습니다.



### Enhancing Apple's Defect Classification: Insights from Visible Spectrum and Narrow Spectral Band Imaging (https://arxiv.org/abs/2410.19784)
Comments:
          6 pages, 3 figures

- **What's New**: 이 연구는 사과의 결함 분류를 데이터 기반으로 접근하여 경제 손실을 줄이고 식품 공급망을 최적화하는 방법을 제시합니다. 본 연구는 가시 스펙트럼과 660 nm의 스펙트럼 파장을 통합하여 정확도와 효율성을 높이는 혁신적인 방법론을 도입합니다.

- **Technical Details**: 본 연구는 Single-Input 및 Multi-Inputs convolutional neural networks (CNNs)를 활용하여 결함 분류의 정확도를 높이기 위한 다양한 방법론을 검증합니다. 660 nm 스펙트럼을 통한 결함 검출 과정은 기존의 전반적인 가시 스펙트럼보다 미세한 세부사항을 드러내는 것을 보여줍니다. 실험에서는 10769장의 이미지 데이터셋을 활용하여 다양한 결함을 분류하는 CNN 모델을 학습합니다.

- **Performance Highlights**: MobileNetV1 모델은 검증 데이터셋에서 98.80%의 정확도를 달성했으며, 이는 전체 가시 스펙트럼을 사용했을 때의 98.26%보다 높은 수치입니다. 이러한 결과는 특정 스펙트럼 범위를 활용한 이미지 캡처가 분류 작업의 네트워크 훈련을 더 효과적으로 할 수 있는 가능성을 시사합니다.



### Data-Driven Uncertainty-Aware Forecasting of Sea Ice Conditions in the Gulf of Ob Based on Satellite Radar Imagery (https://arxiv.org/abs/2410.19782)
- **What's New**: 이번 연구에서는 Sentinel-1의 레이더 이미지, 기상 관측 및 GLORYS 예측을 활용해 해빙 조건 예측을 위한 새로운 데이터 기반 접근 방식을 제안합니다. 이 방법은 북극 해빙 역학의 고유한 도전에 맞춘 데이터 전처리 및 증강 기술과 결합된 첨단 비디오 예측 모델을 통합하여 신뢰할 수 있는 단기 해빙 예측을 가능하게 합니다.

- **Technical Details**: 우리는 전통적인 해빙 모델의 한계를 극복하고, 머신러닝 기법을 활용하여 해빙 조건 예측을 개선합니다. 사용할 기술은 자동회귀 방식의 비디오 예측 모델(IAM4VP), 다중 스케일 복셀 흐름 네트워크(DMVFN), 운동 RNN, Neural ODE 등이 포함됩니다. 예측의 신뢰성을 평가하기 위해 불확실성 정량화 기법을 사용하는 한편, 신뢰 기반 모델 혼합 메커니즘을 제안합니다.

- **Performance Highlights**: 우리의 연구 결과는 기본 방법에 비해 상당한 개선을 보여줍니다. 예측 모델의 정확성과 강인성이 향상되었으며, 해빙 조건의 정확한 매핑을 위한 세밀한 공간 구조를 반영할 수 있도록 하였습니다. 여러 중요한 지표를 통해 예측 정확도를 평가하며, 데이터 결측 및 불균형 문제를 해결하는 접근 방식의 우수성을 입증합니다.



### Feasibility Analysis of Federated Neural Networks for Explainable Detection of Atrial Fibrillation (https://arxiv.org/abs/2410.19781)
- **What's New**: 이 연구는 심장 부정맥(atrial fibrillation, AFib) 조기 감지를 위한 분산형 Federated Learning (FL) 플랫폼에서 신경망을 훈련시키는 가능성을 평가합니다.

- **Technical Details**: Electrocardiogram (ECG) 데이터를 사용하여 모델을 훈련시키고, 모델 성능은 중앙 집중식, 지역적 및 분산형 환경에서 평가하며, 다양한 집계 방법의 효과를 조사합니다.

- **Performance Highlights**: 최고 성능의 분산 모델은 77%의 F1 점수를 달성하였으며, 개별적으로 훈련된 클라이언트들의 평균 성능과 비교하여 15% 향상되었습니다.



### Sampling from Bayesian Neural Network Posteriors with Symmetric Minibatch Splitting Langevin Dynamics (https://arxiv.org/abs/2410.19780)
Comments:
          35 pages, 6 figures. The first two authors contributed equally

- **What's New**: 이 논문에서는 대규모 데이터 및 AI 응용 프로그램을 위한 파라미터 공간 샘플링을 위한 확장 가능한 kinetic Langevin dynamics 알고리즘을 제안합니다.

- **Technical Details**: 이 알고리즘은 미니배치에 대한 대칭적인 포워드/백워드 스윕과 Langevin dynamics의 대칭적 이산화를 결합하였습니다. 특히 Symmetric Minibatch Splitting-UBU (SMS-UBU) 적분자는 O(h^2 d^{1/2})의 바이어스를 보여주며, 샘플링 바이어스를 효과적으로 제어합니다.

- **Performance Highlights**: Fashion-MNIST, Celeb-A, chest X-ray 데이터셋에서 Convolutional Neural Network 아키텍처를 가진 BNN의 posterior predictive probabilities의 보정 성능을 평가한 결과, SMS-UBU로 샘플링된 BNN이 표준 방법보다 현저히 뛰어난 보정 성능을 제공함을 보여주었습니다.



### EEGPT: Unleashing the Potential of EEG Generalist Foundation Model by Autoregressive Pre-training (https://arxiv.org/abs/2410.19779)
- **What's New**: EEGPT라는 새로운 EEG 기초 모델을 소개하며, 다양한 EEG 데이터 형식을 통합하고 특정 작업 전용에서 범용 모델로의 전환을 목표로 함.

- **Technical Details**: EEGPT는 138개의 전극에서 수집된 총 37.5M개의 사전 학습 샘플을 기반으로 한 첫 번째 전극 단위 모델링 전략을 제안함. 자가 회귀(autoregressive) 방식으로 EEG 데이터의 연속성과 시간적 의존성을 포착하며, 1.1B 파라미터를 가진 EEG 연구에서 가장 큰 모델임.

- **Performance Highlights**: EEGPT는 12개 벤치마크에서 5가지 작업에 대해 평가되었으며, 기존의 전문 모델을 초월하여 모든 하위 작업에서 우수한 성능을 보임. 직관적이고 효과적인 이전 지식을 활용한 전송 학습을 통해 다중 작업 간의 상호 증강이 확인됨.



### A Human-Centered Approach for Improving Supervised Learning (https://arxiv.org/abs/2410.19778)
- **What's New**: 이 논문은 Human-Centered (인간 중심) 접근 방식을 사용하여 Supervised Learning (감독 학습)에서 Stacking Ensembles (스태킹 앙상블)의 효율성을 높이고, 성능과 시간, 자원 제약 간의 균형을 맞추는 방법을 제시합니다.

- **Technical Details**: 제안된 알고리즘은 Ensemble Learning (앙상블 학습) 프로세스를 간소화하며, 강력한 기본 모델, 다양한 피처 추출 및 축소 기법, 클래스 확률 및 클러스터링을 결합하여 Supervised Learning의 성능을 최적화합니다. 이 과정에서 모델의 explainability (설명 가능성)도 증대시켜, 사용자가 알고리즘의 결정을 더 잘 이해할 수 있도록 합니다.

- **Performance Highlights**: 아홉 개의 실제 데이터셋에서 실험한 결과, 제안된 방법은 기존의 최첨단 방법들을 능가하는 성능을 보여주었습니다. 이 연구는 Ensemble Learning의 시간 및 계산 오버헤드를 줄이면서도 성능의 손실을 최소화하여 실질적인 적용 가능성을 높이는데 기여하고 있습니다.



### Real-Time Stress Detection via Photoplethysmogram Signals: Implementation of a Combined Continuous Wavelet Transform and Convolutional Neural Network on Resource-Constrained Microcontrollers (https://arxiv.org/abs/2410.19776)
Comments:
          5 figures, implemented on Microcontroller

- **What's New**: 이 논문은 Photoplethysmogram (PPG) 신호 분석을 위한 합성곱 신경망(Convolutional Neural Network, CNN)을 이용한 강력한 스트레스 감지 시스템을 소개합니다.

- **Technical Details**: WESAD 데이터셋을 이용하여 Continuous Wavelet Transform (CWT)을 적용, 손목 PPG 신호로부터 유용한 특징(feature)을 추출했습니다. 최적화 과정에서는 프루닝(pruning)과 포스트 트레인 양자화(Post-Train Quantization)를 포함하여 모델 크기를 1.6 메가바이트로 줄였습니다.

- **Performance Highlights**: CNN은 5 에포크(epoch) 후에 93.7%의 인상적인 정확도를 달성하였으며, 기존의 신호 처리 방법들을 초월하여 착용 가능한 장치에서의 실시간 스트레스 모니터링을 위한 유망한 솔루션으로 자리잡았습니다.



### Copula-Linked Parallel ICA: A Method for Coupling Structural and Functional MRI brain Networks (https://arxiv.org/abs/2410.19774)
Comments:
          25 pages, 10 figures, journal article

- **What's New**: 본 연구에서는 깊은 학습(frameworks) 기술과 copula, 독립 성분 분석(ICA)을 결합한 새로운 융합(fusion) 방법인 CLiP-ICA를 개발했습니다.

- **Technical Details**: CLiP-ICA는 기능적 자기 공명 영상(fMRI)과 구조적 자기 공명 영상(sMRI)의 독립적 원천을 추정하고, copula 기반 모델을 사용하여 fMRI와 sMRI의 공간적 원천을 연결하여 시간적 및 공간적 데이터를 더 유연하게 통합합니다.

- **Performance Highlights**: CLiP-ICA는 알츠하이머병 신경영상 이니셔티브(ADNI)의 데이터를 사용한 결과, 감각운동, 시각, 인지 제어 및 기본 모드 네트워크를 포함한 강하게 및 약하게 연결된 sMRI와 fMRI 네트워크를 효과적으로 캡처했습니다. 이 방법은 더 의미 있는 구성 요소를 발견하고, ICA의 최적 모델 순서 문제를 해결했습니다.



### Real-time Monitoring of Lower Limb Movement Resistance Based on Deep Learning (https://arxiv.org/abs/2410.19769)
Comments:
          17 pages paper

- **What's New**: 본 논문에서는 실시간 하체 이동 저항 모니터링을 위한 새로운 모바일 다중 과제 학습 네트워크(MMTL-Net)를 제안합니다. 이 네트워크는 MobileNetV3를 통합하여 효율적인 특징 추출을 수행하며, 저항 수준 예측과 활동 인식을 동시에 수행하는 다중 과제 학습을 활용합니다.

- **Technical Details**: MMTL-Net은 높은 정확도, 낮은 대기 시간, 개선된 계산 효율성을 제공하여 실제 응용 분야에 적합합니다. 모델은 UCI 인간 활동 인식 및 무선 센서 데이터 마이닝 활동 예측 데이터셋에서 기존 모델을 크게 초월하며, 저항 예측 정확도(Resistance Prediction Accuracy, RPA) 91.2%와 낮은 힘 오류율(Force Error Rate, FER) 6.8%를 달성했습니다. 또한, 실시간 응답성(Real-time Responsiveness, RTR)은 12 밀리초, 처리량(Throughput, TP)은 초당 33 프레임입니다.

- **Performance Highlights**: 실험 결과 MMlTL-Net 모델의 성능은 다양한 실제 시나리오에서 강력한 효과를 입증했습니다. 이는 재활 치료 및 스포츠 응용 분야에서 보다 효율적이고 정확한 시스템을 위한 길을 열어줍니다.



### Statistical Test for Auto Feature Engineering by Selective Inferenc (https://arxiv.org/abs/2410.19768)
Comments:
          36 pages, 7 figures

- **What's New**: 본 연구에서는 Auto Feature Engineering (AFE) 알고리즘이 생성한 특성의 신뢰성을 평가하기 위한 새로운 통계적 테스트를 제안합니다. 이 테스트는 selective inference (SI) 프레임워크를 기반으로 하며, AFE에서 생성된 특성이 선형 모델에서 사용할 때의 통계적 유의성을 정량화할 수 있도록 합니다.

- **Technical Details**: AFE는 원시 데이터를 바탕으로 중요한 특성을 자동으로 생성하는 과정으로, 이는 조합 최적화 문제로 설정됩니다. 본 연구에서는 헤uristic AFE 알고리즘을 사용할 때의 통계적 테스트 문제를 다루며, AFE 알고리즘이 생성한 특성에 대해 p-value를 제공하는 새로운 통계적 테스트를 제시합니다.

- **Performance Highlights**: 제안된 테스트는 실험을 통해 AFE 알고리즘이 생성한 특성에 대해 유효한 p-value를 제공함을 입증하였으며, 이는 잘못된 발견확률을 통제할 수 있는 이론적으로 보장된 방법임을 보여줍니다.



### Learning Robust Representations for Communications over Interference-limited Channels (https://arxiv.org/abs/2410.19767)
Comments:
          Submitted to WCNC 2025

- **What's New**: 본 연구에서는 셀룰러 네트워크에서 간섭 문제를 해결하기 위해 TwinNet과 SiameseNet이라는 두 가지 새로운 방법론을 도입하였다. 이 방법들은 오토인코더(autoencoder)를 활용하여 간섭이 제한된 환경에서 블록 전송(block transmission) 및 수신(detection)을 위한 인코더와 디코더 설계를 최적화하는 데 초점을 맞추고 있다.

- **Technical Details**: TwinNet과 SiameseNet은 인코딩 및 디코딩 과정을 위한 FCNN (Fully Connected Neural Network) 아키텍처를 사용하여 간섭 사용자(husers)와 함께 비트 블록의 전송 및 수신 성능을 분석한다. 이 모델들은 블록 오류율(Block Error Rate, BLER) 성능에서 완전한 직교(Time Division Multiple Access, TDMA) 방법보다 우수한 성능을 보인다.

- **Performance Highlights**: 연구 결과, 제안한 모델이 기존의 데이터 전송 방식 및 완전 직교 방식보다 간섭 구조를 효과적으로 활용하여 성능이 향상된 것을 명확히 보여준다. 이로 인해 데이터 기반 모델의 잠재적 이점이 구체적으로 고찰되며, 코드워드(codewords)의 특성을 분석하여 모델의 성능 향상을 직관적으로 설명하는 데 기여한다.



### Large Model for Small Data: Foundation Model for Cross-Modal RF Human Activity Recognition (https://arxiv.org/abs/2410.19766)
- **What's New**: 이 논문은 Radio-Frequency (RF) 기반의 Human Activity Recognition (HAR) 시스템을 향상시키기 위해 시각 기반의 Foundation Models (FMs)의 지식을 전이하는 새로운 cross-modal framework인 FM-Fi를 제안합니다. FM-Fi는 RF 인코더가 FMs의 해석력을 물려받아 zero-shot learning을 달성할 수 있도록 돕습니다.

- **Technical Details**: FM-Fi는 novel cross-modal contrastive knowledge distillation (CKD) 메커니즘을 통해 RF와 시각 데이터 간의 효과적인 지식 전이를 달성하며, FM과 RF의 고유한 기능을 이용해 두 개체 간의 정렬을 개선합니다. 또한, metric-based few-shot learning 기술을 통하여 특정 HAR 작업에 대한 성능을 향상시킵니다.

- **Performance Highlights**: FM-Fi의 평가 결과는 시각 기반 방법론과 동등한 성능을 보여주며, 다양한 환경에서의 일반화 가능성을 실증적으로 검증합니다.



### Physical Simulation for Multi-agent Multi-machine Tending (https://arxiv.org/abs/2410.19761)
Comments:
          3 pages, one figure, an extended abstract presented at the 7th edition of the Montreal AI symposium (MAIS) 2024

- **What's New**: 제조업 부문에서는 인력 부족 문제에 직면해 있습니다. 이 연구는 자동화와 로보틱스가 이러한 문제를 최소화하는 방법을 제시합니다. 특히, 강화 학습(Reinforcement Learning, RL)을 통해 로봇이 환경과 상호작용을 통해 학습할 수 있는 가능성을 탐구했습니다.

- **Technical Details**: 이 연구는 간단한 로보틱 시스템을 사용하여 '실제' 데이터를 기반으로 RL을 활용하였습니다. 대규모 비싼 로봇을 제조 환경에 배치하지 않고도 로봇이 시뮬레이션의 에이전트 행위를 모방하는 테이블탑 아레나를 설계했습니다. 비록 다이내믹스(dynamics)와 기계 크기에 차이가 있었으나, 로봇은 시뮬레이션에서와 같은 행동을 보였습니다.

- **Performance Highlights**: 이 실험들은 실제 배치의 도전 과제에 대한 초기 이해를 제공했습니다. 로봇이 환경에 대한 학습을 통해 제조업에 적용될 수 있는 가능성을 보여줍니다.



### Establishing Nationwide Power System Vulnerability Index across US Counties Using Interpretable Machine Learning (https://arxiv.org/abs/2410.19754)
- **What's New**: 미국의 기후 변화, 노후된 전력망 및 증가하는 에너지 수요로 인해 정전 사고가 빈번하고 강렬해졌습니다. 이 연구는 2014년부터 2023년까지의 정전 데이터에 기반하여 전력 시스템의 취약성을 평가하는 새로운 관점을 제시합니다.

- **Technical Details**: 약 1억 7900만 건의 정전 기록을 수집하여 카운티 단위로 전력 시스템 취약성 지표(PSVI)를 개발했습니다. 세 가지 차원(강도, 빈도, 지속 시간)을 바탕으로 XGBoost 및 SHAP와 같은 해석 가능한 머신러닝 모델을 적용하였습니다.

- **Performance Highlights**: 최근 10년 동안 전력 시스템 취약성이 지속적으로 증가하는 것으로 나타났으며, 45개 주에 걸쳐 318개 카운티가 높은 취약성 지역으로 식별되었습니다. 특히, 서부 해안(캘리포니아 및 워싱턴), 동부 해안(플로리다 및 북동부 지역), 그리고 대호수 도시권과 멕시코 만(텍사스)에서 높은 취약성이 발견되었습니다.



### The Geometry of Concepts: Sparse Autoencoder Feature Structur (https://arxiv.org/abs/2410.19750)
Comments:
          13 pages, 12 figures

- **What's New**: 이 논문은 Sparse Autoencoders (SAE)를 활용하여 개념의 우주를 형성하는 고차원 벡터의 구조를 세 가지 스케일에서 분석하는 내용을 담고 있습니다. 특히, 이 구조에는 '원자' 규모(atomic scale)에서 '결정'(crystal), '두뇌' 규모(brain scale)에서 기능적 모듈성, 그리고 '은하' 규모(galaxy scale)에서는 비등방(isotropic)적이지 않은 분포가 포함됩니다.

- **Technical Details**: 논문은 세 가지 주요 스케일에서의 구조를 탐구합니다: 1) '원자' 스케일은 패러렐로그램(parallelogram) 또는 사다리꼴(trapezoid) 형태의 결정 구조를 분석합니다. 2) '두뇌' 스케일은 기능적으로 유사한 SAE 특징들이 공간적으로 함께 모여 '로브'(lobe)를 형성하는지 조사합니다. 3) '은하' 스케일에서는 특징 점군의 고유값(eigenvalue) 분포가 중간 레이어에서 가장 가파른 기울기를 보이는 파워 법칙(power law)을 따른다고 명시합니다.

- **Performance Highlights**: 이 연구는 Linear Discriminant Analysis (LDA)를 통해 데이터에서 전역 방해 요소(global distractor directions)를 제거하고, 클러스터의 품질을 크게 향상시킬 수 있음을 보였습니다. 따라서 기존의 결정 구조를 보다 명확하게 드러내는 결과를 도출했으며, 특히 기능적 특성 간의 높은 공간적 모듈성과 관련하여 관찰된 데이터는 임상 신경과학의 기존 연구와도 일치하는 경향을 보여줍니다.



### Adaptive Real-Time Multi-Loss Function Optimization Using Dynamic Memory Fusion Framework: A Case Study on Breast Cancer Segmentation (https://arxiv.org/abs/2410.19745)
- **What's New**: 본 논문에서는 실시간으로 다중 손실 함수에 대한 적응형 패널라이징(dynamic memory fusion) 프레임워크를 제안합니다. 이 프레임워크는 이전 손실 값을 활용하여 훈련 과정 내내 다중 손실 함수의 가중치를 동적으로 조정합니다. 또한 초기 단계에서 모델 성능을 향상시키기 위해 보조 손실 함수를 통합합니다.

- **Technical Details**: 우리의 접근법인 DMF(동적 기억 융합) 프레임워크는 다수의 손실 함수 가중치를 훈련 중에 동적으로 조정하기 위해 역사적 손실 데이터(history loss data)를 활용합니다. 이는 고정 가중치 접근 방식과 달리 훈련 중 다양한 요소에 균형을 맞추어 모델이 특정 요소에 심하게 편향되지 않도록 보장합니다. 또한, 클래스 불균형(class imbalance)을 해결하기 위한 class-balanced dice loss 함수를 도입하여 적은 대표성을 가진 클래스에 우선 순위를 부여합니다.

- **Performance Highlights**: 유방 초음파 데이터셋(breast ultrasound datasets)에서의 실험 결과, 제안한 프레임워크는 다양한 메트릭(metric)에서 세분화 성능(segmentation performance)을 향상시킴을 보여줍니다. 이러한 결과는 모델이 가장 관련 있는 기준을 우선시하도록 동적으로 조정함으로써 변화하는 환경에서 향상된 성능에 기여한다는 것을 입증합니다.



### Tourism destination events classifier based on artificial intelligence techniques (https://arxiv.org/abs/2410.19741)
- **What's New**: 이번 연구는 관광지 관리에 있어 고객의 요구를 파악하고 최적의 서비스를 제공하기 위한 새로운 자동 분류 프로세스를 제안합니다. 이 프로세스는 관광 이벤트를 계층적 분류법 (hierarchical taxonomy)을 통해 체계적으로 분류합니다.

- **Technical Details**: 연구에서 사용된 기술적 방법론은 CRISP-DM과 같은 데이터 과학 기법, 감독기계학습 (supervised machine learning), 자연어 처리 기술 (natural language processing techniques)을 포함합니다. 이러한 방법들은 서로 다른 지리적 지역에서 이벤트 정보를 표준화된 카탈로그 (catalog)로 생성하는 데 기여합니다.

- **Performance Highlights**: 이 자동 분류 도구는 항공사, 여행사, 호텔 체인과 같은 여러 지역에서 정보를 제공하는 기업들에게 매우 가치가 있으며, 이벤트 카테고리에 상관없이 사용자들이 원하는 이벤트를 쉽게 찾을 수 있도록 돕습니다. 결과적으로 이 도구는 기업과 최종 사용자가 관광 이벤트 정보를 상호작용하는 방식을 혁신할 잠재력을 지니고 있습니다.



### The Effect of Acute Stress on the Interpretability and Generalization of Schizophrenia Predictive Machine Learning Models (https://arxiv.org/abs/2410.19739)
Comments:
          20 pages, 7 figures

- **What's New**: 이번 연구에서는 정신질환인 조현병의 예측 정확도를 높이기 위해 EEG 데이터를 사용한 기계 학습 모델의 성능 향상에 기여할 수 있는 급성 스트레스(acute stress)의 영향을 조사했습니다. 특히, 스트레스가 EEG 기록 중에 발생할 경우 모델의 정확도에 부정적인 영향을 미칠 수 있음을 보여주었습니다.

- **Technical Details**: 본 연구에서는 네 가지 XGBoost 모델을 구축하였으며, 그 중 하나는 스트레스 예측, 두 개는 조현병 분류(휴식기 및 과제 수행)용, 하나는 두 조건의 조현병을 예측하는 데 사용되었습니다. Explainable A.I. (XAI) 기법인 SHAP을 적용하여 결과를 분석했습니다. EEG 신호의 주파수 대역 전력 조정 방법을 통해 스트레스 아티팩트를 제거하여 예측 모델 성능을 개선하는 방법을 제안했습니다.

- **Performance Highlights**: 결과에 따르면, 급성 스트레스는 각 EEG 세션 간에 상이하게 나타나 모델의 성능과 정확도에 영향을 미쳤습니다. 스트레스 수준을 고려하고 이를 모델 훈련 중 보정한 후 일반화 성능이 향상되었습니다. 스트레스가 EEG 데이터에 추가적인 생리학적 아티팩트로 작용할 수 있음을 보여주며, 제안된 접근 방식은 예측 성능을 획기적으로 개선하였습니다.



### Combining LLM Code Generation with Formal Specifications and Reactive Program Synthesis (https://arxiv.org/abs/2410.19736)
- **What's New**: 최근 몇 년간 Large Language Models(LLMs)는 코드 생성을 위한 유용성과 인기가 폭발적으로 증가했습니다. 그러나 LLMs는 여전히 정확도와 고위험 애플리케이션에 적합하지 않습니다. 우리는 LLM과 formal methods 기반의 프로그램 합성을 결합한 새로운 해결책을 소개합니다.

- **Technical Details**: 이 연구에서는 LLM을 사용하여 자연어로부터 formal specifications을 생성합니다. 이후 프로그램 합성을 통해 'correct-by-construction' 코드 생성을 진행하고, 이를 통해 비정상적이거나 어려운 시스템에도 적용할 수 있는 방법을 제안합니다. 특히, Temporal Stream Logic(TSL)을 활용하여 복잡한 시스템의 동작을 명시하는 방법을 사용합니다.

- **Performance Highlights**: 제안된 시스템은 기존의 LLM 코드 생성에서는 해결할 수 없었던 문제를 해결할 수 있는 파이프라인을 통해, 최소한의 코드 검증 작업으로 고위험 시스템에 필요한 신뢰할 수 있는 코드를 동적으로 생성할 수 있는 가능성을 보여줍니다.




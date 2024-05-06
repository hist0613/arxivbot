### Better & Faster Large Language Models via Multi-token Prediction (https://arxiv.org/abs/2404.19737)
- **What's New**: 이 연구에서는 언어 모델을 훈련할 때 다음 토큰(n-token)을 한 번에 여러 개 예측하도록 훈련하는 것이 더 높은 샘플 효율성을 가져온다는 새로운 접근 방식을 제안합니다. 기존의 GPT 및 Llama와 같은 대규모 언어 모델은 다음 토큰 예측 손실(next-token prediction loss)로 훈련되었습니다.

- **Technical Details**: 연구팀은 훈련 코퍼스(corpus)의 각 위치에서 모델이 공유 모델 트렁크(shared model trunk)를 활용하여 독립적인 출력 헤드(independent output heads)를 사용하여 앞서의 n 개의 토큰을 예측하도록 요청했습니다. 이 다중 토큰 예측(multi-token prediction) 방식은 부가적인 훈련 과제로 간주되며, 코드 및 자연어 모델에 대한 훈련 시간의 추가 비용 없이 하류 능력(downstream capabilities)의 향상을 측정했습니다.

- **Performance Highlights**: 13B 매개변수(parameter)를 가진 모델은 기존의 다음 토큰 모델들에 비해 HumanEval에서 12% 더 많은 문제를 해결했으며, MBPP에서는 17% 더 많은 문제를 해결하였습니다. 작은 알고리즘 과업(algorithmic tasks)에 대한 실험은 다중 토큰 예측이 유도 헤드(induction heads) 및 알고리즘 추론 능력(algorithmic reasoning capabilities) 개발에 유리하다는 것을 보여줍니다. 추가적인 혜택으로, 4 토큰 예측으로 훈련된 모델은 대형 배치 크기에서도 추론 시 최대 3배까지 빠르게 작동합니다.



### Iterative Reasoning Preference Optimization (https://arxiv.org/abs/2404.19733)
- **What's New**: 이 연구에서는 반복적 선호 최적화 기법 (Iterative preference optimization)을 사용하여 생각의 연쇄 (Chain-of-Thought; CoT) 향상에 중점을 둡니다. 연구팀은 승리 단계와 패배 단계를 최적화하여 정답으로 이끄는 과정을 찾고, 반복적으로 순환하며 훈련 및 최적화를 진행합니다. 또한 수정된 DPO 손실 함수와 추가 음의 로그 가능도 (Negative Log-Likelihood; NLL) 요소를 사용하여 모델의 성능을 향상시키는 데 중요한 역할을 했습니다.

- **Technical Details**: 이 연구의 기술적 세부사항으로는 Chain-of-Thought (CoT) 생성 후, 정확한 응답과 잘못된 응답을 기준으로 선호 쌍을 구성하고, DPO 프레임워크 내에서 반복적으로 모델을 업데이트 하는 과정이 포함됩니다. 각 반복에서 NLL 손실 조건을 추가하여 강화된 딥러닝 모델 훈련 방식을 도입하였습니다.

- **Performance Highlights**: 이 방법론으로 Llama-2-70B-Chat 모델의 GSM8K 데이터셋에 대한 정확도가 55.6%에서 81.6%로 향상되었으며, 32개 샘플의 다수결로는 88.7%까지 향상됨. 또한 ARC-Challenge에서는 77.8%에서 86.7%로, MATH에서는 12.5%에서 20.8%로 성능이 개선되었습니다. 이러한 결과는 추가 데이터셋 없이도 다른 Llama-2 기반 모델들보다 우수한 성능을 보여줍니다.



### ThangDLU at #SMM4H 2024: Encoder-decoder models for classifying text  data on social disorders in children and adolescents (https://arxiv.org/abs/2404.19714)
Comments: 4 pages

- **What's New**: 이 논문은 2024년 #SMM4H (소셜 미디어를 통한 건강 상태 데이터 마이닝) 워크샵의 Task 3 및 Task 5에 참여하여, 트윗 데이터 내의 분류 문제를 해결합니다. Task 3는 사회 불안 증상에 대한 외부 환경의 영향을 논하는 트윗들을 대상으로 하는 다중 클래스 (multi-class) 분류 과제이며, Task 5는 어린이의 의료 장애를 보고하는 트윗에 초점을 맞춘 이진 분류 (binary classification) 과제입니다.

- **Technical Details**: 저자들은 BART-base와 T5-small과 같은 사전 훈련된 인코더-디코더 (pre-trained encoder-decoder) 모델들에서 전이 학습 (transfer learning)을 적용해 트윗 데이터의 레이블을 식별했습니다. 또한 데이터 증강 (data augmentation) 방법을 도입하여 모델 성능에 미치는 영향을 평가했습니다.

- **Performance Highlights**: 시스템은 Task 3에서 최고 F1 스코어 0.627을, Task 5에서는 최고 F1 스코어 0.841을 달성했습니다. 이러한 결과는 특히 트윗 데이터를 활용한 건강 관련 이슈의 자동 분류에 있어 인코더-디코더 모델의 효과를 입증합니다.



### Automated Generation of High-Quality Medical Simulation Scenarios  Through Integration of Semi-Structured Data and Large Language Models (https://arxiv.org/abs/2404.19713)
Comments: 22 pages but 12 are appendices which are examples of the main text. 3 figures, 4 tables

- **What's New**: 이 연구에서는 의료 교육에 혁신적인 변화를 도입하고자 대형 언어 모델(Large Language Models, LLMs), 특히 OpenAI의 ChatGPT3.5를 결합하여 반정형 데이터(semi-structured data)를 사용하는 새로운 프레임워크를 제안합니다. 이는 의료 시뮬레이션 시나리오를 자동으로 생성하여 기존에 시간이 많이 소요되던 과정을 혁신적으로 개선함으로써 의료 교육의 다양성과 유연성을 향상시키는 데 크게 기여합니다.

- **Technical Details**: 제안된 접근 방식은 AI를 활용하여 교육 목표에 맞춤화된 상세하고 임상적으로 관련성 높은 시나리오를 효율적으로 생성합니다. 이는 OpenAI의 ChatGPT3.5와 같은 LLM을 사용하여 반정형 데이터를 구조화된 데이터와 통합하는 과정을 포함합니다.

- **Performance Highlights**: 이 기술의 도입으로 시나리오 개발에 필요한 시간과 자원이 크게 감소하였으며, 이로 인해 더욱 다양한 시뮬레이션을 제공할 수 있게 되었습니다. 초기 피드백은 교육자와 학습자 모두에서 높은 참여와 지식 습득의 개선을 보여주어, 시뮬레이션 기반 학습에서 AI 증강 방법론의 효과를 확인시켜 주었습니다.



### When to Retrieve: Teaching LLMs to Utilize Information Retrieval  Effectively (https://arxiv.org/abs/2404.19705)
- **What's New**: 이 논문에서는 대규모 언어 모델(Large Language Models, LLMs)이 외부 정보 검색 시스템(information retrieval, IR)을 효과적으로 사용하여 주어진 질문에 필요한 추가 컨텍스트를 언제 인식하고 사용해야 하는지 학습할 수 있음을 보여줍니다. 이는 특히 PopQA 데이터셋에서 발견된 바와 같이, 자주 묻는 질문은 LLM의 파라메트릭 메모리(parametric memory)를 사용하여 효과적으로 해결할 수 있지만, 인기가 덜한 질문은 IR 시스템의 사용을 필요로 합니다.

- **Technical Details**: 연구자들은 LLM이 질문에 대한 답을 모를 때 특별한 토큰(<RET>)을 생성하도록 훈련하는 새로운 훈련 방법을 제안하였습니다. 이를 위해 기존의 오픈 도메인 질문 응답 데이터셋(open-domain question answering datasets)을 이용했습니다. 적응형 검색 LLM(Adaptive Retrieval LLM, Adapt-LLM)은 PopQA 데이터셋에서 세 가지 설정을 비교 분석하여 평가되었으며, (i) 모든 질문에 대한 정보 검색, (ii) 항상 LLM의 파라메트릭 메모리 사용, (iii) 인기도 임계값을 사용하여 검색기(retriever)의 사용 여부를 결정하는 방법과 비교하였습니다.

- **Performance Highlights**: 분석 결과, Adapt-LLM은 필요할 때 <RET> 토큰을 생성하여 IR이 필요함을 표시하면서도, 자체 파라메트릭 메모리에만 의존하기로 결정했을 때 높은 정확도를 달성하는 것으로 나타났습니다. 이는 Adapt-LLM이 질문에 대해 어떻게 대응할지를 스스로 판단할 수 있는 능력을 개선한다는 것을 의미합니다.



### Transferring Troubles: Cross-Lingual Transferability of Backdoor Attacks  in LLMs with Instruction Tuning (https://arxiv.org/abs/2404.19597)
Comments: work in progress

- **What's New**: 이 연구에서는 다국어 대형 언어 모델(multilingual large language models, LLMs)이 크로스-링거얼(cross-lingual) 백도어 공격에 얼마나 취약한지에 중점을 두고 있습니다. 특히 하나 또는 두 개의 언어로 지시 튜닝 데이터(instruction-tuning data)를 오염시켰을 때, 그로 인해 튜닝되지 않은 다른 언어의 출력에 미치는 영향에 대하여 조사했습니다.

- **Technical Details**: 연구팀은 mT5, BLOOM, GPT-3.5-turbo 같은 모델을 사용하여 실험을 진행했으며, 트리거(trigger)가 문장을 완전히 다르게 바꾸어도 여전히 작동할 수 있는 것을 확인하였습니다. 이 연구는 크로스-링게얼 반응 설정(cross-lingual response settings)에서의 백도어 메커니즘(backdoor mechanism)의 효율성을 평가하였으며, 25개 언어에서 평균 공격 성공률(attack success rate) 50%를 달성했습니다.

- **Performance Highlights**: 다양한 시나리오에서의 공격 성공률은 95%을 웃돌았습니다. 특히 Llama2, Llama3, Gemma와 같이 주로 영어 데이터로 사전 학습된(pre-trained) 모델들이 더 큰 취약성을 보여주었습니다. 이러한 결과는 다국어 LLMs에서의 보안 위험이 높으며, 타겟팅된 보안 조치가 시급히 필요함을 시사합니다.



### RepEval: Effective Text Evaluation with LLM Representation (https://arxiv.org/abs/2404.19563)
- **What's New**: 이 연구에서는 평가를 위해 LLM(대량 언어 모델) 표현의 projection을 활용하는 첫 번째 메트릭인 RepEval을 소개합니다. RepEval은 트레이닝을 위한 최소한의 샘플 쌍만 요구하며, 간단한 프롬프트 수정을 통해 다양한 작업으로 쉽게 전환할 수 있습니다.

- **Technical Details**: RepEval을 도입하여 LLM의 표현을 활용함으로써, 기존 메트릭들이 특정 시나리오에 국한되는 문제를 해결하였습니다. 이 메트릭은 적은 양의 훈련 데이터와 간단한 프롬프트 변경만으로 다양한 자연어 생성(NLG) 작업에 적용 가능합니다.

- **Performance Highlights**: RepEval이 진행된 10개 데이터셋과 3가지 작업에서의 결과는 우리 메서드의 높은 효과성을 입증하였습니다. 특히, 인간 판단과의 상관관계에서 이전 메트릭들을 초과하며, 심지어 GPT-4를 능가하는 성능을 보여주었습니다.



### Extending Llama-3's Context Ten-Fold Overnigh (https://arxiv.org/abs/2404.19553)
- **What's New**: LLama-3-8B-Instruct 모델의 컨텍스트 길이(context length)를 8K에서 80K로 확장하는 데 성공했습니다. 이 과정에는 QLoRA fine-tuning을 사용하였으며, 모델은 단일 8xA800 (80G) GPU 시스템에서 8시간만에 훈련을 완료했습니다.

- **Technical Details**: 이 새로운 모델은 NIHS, 토픽 검색(topic retrieval), 그리고 긴 컨텍스트 언어 이해(long-context language understanding) 등 다양한 평가 작업에서 뛰어난 성능을 보여줍니다. 또한, 짧은 컨텍스트에서도 원래의 기능을 잘 유지합니다. 이러한 컨텍스트 길이의 대폭적인 확장은 GPT-4에 의해 생성된 3.5K의 합성 훈련 샘플(synthetic training samples)로 가능했습니다. 이는 LLMs(Large Language Models)가 원래의 컨텍스트 길이를 확장시킬 수 있는 상당히 과소평가된 잠재력을 가지고 있음을 시사합니다.

- **Performance Highlights**: 추가적인 계산 자원을 통해 컨텍스트 길이는 80K를 훨씬 넘어서 확장될 수 있습니다. 연구팀은 데이터(data), 모델(model), 데이터 생성 파이프라인(data generation pipeline), 훈련 코드(training code)를 포함한 모든 자원을 공개할 예정이어서, 커뮤니티의 미래 연구를 촉진할 수 있습니다.



### RAG and RAU: A Survey on Retrieval-Augmented Language Model in Natural  Language Processing (https://arxiv.org/abs/2404.19543)
Comments: 30 pages, 7 figures. Draft version 1

- **What's New**: 이 연구에서는 자연어 처리(Natural Language Processing, NLP) 분야에서 큰 발전을 이루어낸 대규모 언어 모델 (Large Language Models, LLMs)이 직면한 도전 과제들, 예를 들어 환각(hallucination) 현상과 도메인 특정 지식(domain-specific knowledge)의 필요성을 어떻게 완화할 수 있는지를 탐구하고 있습니다. 특히, 외부 자원에서 정보를 검색하여 LLM에 통합하는 최신 방법론들이 어떻게 NLP 작업에서 모델의 성능을 크게 향상시키는지를 설명합니다.

- **Technical Details**: 이 서베이 논문은 정보 검색 기반 언어 모델 (Retrieval-Augmented Language Models, RALMs), 구체적으로 검색-증강 생성 (Retrieval-Augmented Generation, RAG)과 검색-증강 이해 (Retrieval-Augmented Understanding, RAU)에 대한 포괄적인 검토를 제공합니다. RALMs의 필수 구성요소인 검색기 (Retrievers), 언어 모델, 그리고 증강(Augmentations)과 그들 간의 상호 작용이 어떻게 다양한 모델 구조와 응용 프로그램을 이끌어내는지에 대해 논의합니다.

- **Performance Highlights**: RALMs는 번역, 대화 시스템에서부터 지식 집약적인 애플리케이션에 이르기까지 다양한 과제에서 유용함을 보여줍니다. 또한, RALMs의 평가 방법에 대해 다루며, 로버스트성(robustness), 정확도(accuracy), 관련성(relevance)을 평가의 중요한 요소로 강조합니다. 이 논문은 또한 RALMs의 한계, 특히 검색 품질과 계산 효율성에 대해 지적하며, 이에 대한 향후 연구 방향을 제시합니다.



### Do Large Language Models Understand Conversational Implicature -- A case  study with a chinese sitcom (https://arxiv.org/abs/2404.19509)
Comments: 14 pages, 8 tables and 5 figures

- **What's New**: 이 연구에서는 대화 함축(conversational implicature)을 위한 첫 번째 중국어 멀티 턴 다이얼로그 기반 데이터셋인 SwordsmanImp를 소개합니다. 이 데이터셋은 중국 시트콤 $	extit{My Own Swordsman}$의 대화에서 추출되었으며, 그라이스의 준칙(Gricean maxims) 위반 여부에 대해 주석이 달린 200개의 질문을 포함하고 있습니다.

- **Technical Details**: 이 연구에서는 여러 종류의 닫힌 소스(close-source) 및 오픈 소스(open-source) 대규모 언어 모델(LLMs)에 대해 두 가지 태스크를 테스트했습니다: 다지선다형 질문(multiple-choice question task)과 함축 설명(implicature explanation task)입니다. GPT-4는 다지선다형 질문에서 인간 수준의 정확도(94%)를 달성했으며, CausalLM은 78.5%의 정확도를 보였습니다. 다른 모델들은 20%에서 60%의 정확도 범위를 보였습니다.

- **Performance Highlights**: GPT-4는 함축의 설명에 있어서도 높은 점수를 받았으며, 이는 사리성(reasonability), 논리성(logic), 유창성(fluency)을 기준으로 사람 평가자들에 의해 평가되었습니다. 다른 모델들은 텍스트의 유창성과 일관성은 높지만, 사리성에서는 낮은 점수를 받아 만족스러운 설명을 생성하지 못하는 것으로 나타났습니다. 또한, LLM들의 성능은 그라이스의 준칙에 따라 크게 달라지지 않아, 다양한 준칙에서 파생된 함축을 처리하는 방식에 차이가 없는 것으로 보입니다.



### Context-Aware Machine Translation with Source Coreference Explanation (https://arxiv.org/abs/2404.19505)
Comments: Accepted to TACL. This is a pre-MIT Press publication version

- **What's New**: 이 연구에서는 기계 번역(Machine Translation, MT) 모델의 성능을 향상시키기 위해 주요 입력 특성을 예측하는 새로운 방법을 제안합니다. 이 모델은 문맥 정보와 번역 출력 표현을 활용하여 입력에서의 대용어 정보(coreference features)를 예측합니다. 특히 이 연구는 문맥을 고려한 기계 번역에 있어 문맥 정보를 보다 효과적으로 활용하는 것을 목표로 하며, 더 긴 문맥 정보를 처리할 때 번역 품질을 높이는 데 중점을 둡니다.

- **Technical Details**: 제안된 모델은 기존의 변형기(Transformer) 기반 NMT 모델 위에 구축되며, 번역결과의 표현을 추가적인 특성으로 사용하여 입력의 문맥적 특성을 예측합니다. 다목적 학습(multi-task learning) 방식을 사용하지만, 번역 결정에 사용된 표현물에서 정보를 융합합니다. 이를 통해 문맥중심 모델(context-aware model)이 긴 문맥 정보를 활용할 수 있도록 돕는 대용어 해석(sub-model)을 통합했습니다.

- **Performance Highlights**: 이 모델은 영어-독일어(English-German), 영어-러시아어(English-Russian) 데이터셋 및 다양한 언어를 포함한 TED talk 데이터셋에서 1.0 BLEU 점수 이상의 개선을 보였습니다. 실제 테스트 결과에 따르면, 기존의 문맥을 고려하지 않은 변형기 모델 및 최신 문맥중심 모델보다 높은 성능을 보입니다. 이 모델은 자체 주의 집중도 맵(self-attention heat map)과 대용어 클러스터(coreference clusters) 간에 강한 상관관계를 보여주며, 대용어 예측 작업에서의 효과적인 훈련 방법을 입증하였습니다.



### Safe Training with Sensitive In-domain Data: Leveraging Data  Fragmentation To Mitigate Linkage Attacks (https://arxiv.org/abs/2404.19486)
- **What's New**: 이 연구에서는 링키지 공격(Linkage attacks)에 대한 보호를 강화하고자 식별자를 제거한 조각화된 데이터(Fragmented data)를 사용하여 LLMs(Language Learning Models)를 미세 조정하는 새로운 방법을 제안합니다. 특히, 심혈관 질환(Carviovascular diagnoses) 판단을 위해 BERT 기반 모델을 미세 조정하였습니다.

- **Technical Details**: 연구팀은 스탠퍼드 대학(Stanford University)의 Stanza 도구를 사용하여 명사구(NP)와 동사구(VP)를 추출하고, 의미 있는 구문 단위(Syntactic units)를 선택하여 데이터 공유 시 개인 정보를 보호하는 데 도움이 되도록 했습니다. 이렇게 조합된 데이터 조각은 원본과 연결될 확률을 크게 줄여줍니다.

- **Performance Highlights**: 실험 결과 BERT 모델은 조각화된 데이터(Fragmented data)와 전체 텍스트 데이터(Full training data)를 사용했을 때 유사한 분류 결과를 보여주었습니다. 이는 조각화된 데이터가 개인 식별 정보의 노출 위험을 줄이면서도 효과적인 모델 학습이 가능함을 시사합니다.



### FactCheck Editor: Multilingual Text Editor with End-to-End fact-checking (https://arxiv.org/abs/2404.19482)
Comments: Accepted in SIGIR 2024 (demo track)

- **What's New**: 새로운 'FactCheck Editor'는 사실 검증을 자동화하고 사실적 부정확성을 수정하는 데 도움을 주는 고급 텍스트 편집기를 소개합니다. 미디어에서의 잘못된 정보의 확산이 큰 문제가 되고 있는 가운데, 이 도구는 주로 콘텐츠 제작자들의 의도치 않은 오류를 방지하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 이 텍스트 에디터는 90개 이상의 언어를 지원하며, transformer models(트랜스포머 모델)을 활용하여 사람들이 검증 작업을 수행하는 데 필요한 노력을 줄여줍니다. 'FactCheck Editor'는 검증이 필요한 텍스트 주장을 탐지하는 전체 워크플로우를 시연하고, 관련 검색 엔진 쿼리를 생성하며, 웹에서 적절한 문서를 검색합니다. 또한, Natural Language Inference(NLI)를 이용하여 주장의 진위를 예측하고, Large Language Models(LLMs)을 사용하여 증거를 요약하고 텍스트의 오류를 수정할 수 있는 텍스트 개정을 제안합니다.

- **Performance Highlights**: 모델은 다양한 언어에서 주장 검출 및 진실성 평가의 효과성을 평가하며, 이를 통해 글로벌 사용자를 대상으로 한 실질적인 응용 가능성을 보여줍니다. 이 에디터는 정보의 정확성을 향상시키는 데 중요한 역할을 할 수 있습니다.



### Which Nigerian-Pidgin does Generative AI speak?: Issues about  Representativeness and Bias for Multilingual and Low Resource Languages (https://arxiv.org/abs/2404.19442)
Comments: Working paper

- **What's New**: 새로운 연구에서는 약 1억 2천만 명이 사용하는 나이지리아 피진 언어인 Naija가 어떻게 기존의 스포큰(spoken) 언어에서 두 개의 서면(genre)으로 다양화됐는지를 다루고 있습니다. 특히, BBC와 위키백과(Wikipedia)에서 사용되는 Naija 언어가 서로 다른 언어적 특징을 보이고 있으며, 이는 기계 번역(Machine Translation) 실험과 통계 분석을 통해 입증되었습니다.

- **Technical Details**: 연구팀은 Naija 언어의 두 서면 간의 어휘 및 문법 구조에서 차이가 있는지 조사했습니다. 결과적으로, BBC에서 사용되는 Naija와 위키백과에서 사용되는 Naija 간에는 단어 순서와 어휘 사용에서 명백한 차이가 발견되었습니다. 또한, 이 연구는 현재의 생성적 AI(Generative AI)가 BBC에서 쓰인 Naija에만 기반하여 작동하고 있음을 밝혔습니다. 이는 Wikipedia 장르에서 작성된 Naija가 생성적 AI에서 제대로 표현되지 않고 있음을 시사합니다.

- **Performance Highlights**: 이 분석은 Naija 언어의 처리에 있어서 특정 문서 장르에 편향된 AI 기술의 한계를 지적하며, 생성적 AI가 다양한 언어 장르를 포괄적으로 인식하고 처리할 필요성을 강조합니다. 더 나아가 이 연구는 언어적 다양성을 기계 학습 모델에 효과적으로 통합하는 방법에 대한 논의를 촉진할 것입니다.



### Can Large Language Models put 2 and 2 together? Probing for Entailed  Arithmetical Relationships (https://arxiv.org/abs/2404.19432)
- **What's New**: 본 연구는 대규모 언어 모델(Large Language Models, LLMs)의 지식과 추론 능력에 대해 탐구합니다. LLMs가 가지고 있는 지식에 대해 '추론'하는 부분을 새롭게 조사하며, '카디널리티 비교(cardinality comparisons)' 방법을 사용하여 실험을 수행하였습니다. 이는 새가 가진 다리의 갯수와 삼륜차가 가진 바퀴의 수를 비교하는 것과 같이 간단한 설정을 통해 이루어졌습니다.



### Sõnajaht: Definition Embeddings and Semantic Search for Reverse  Dictionary Creation (https://arxiv.org/abs/2404.19430)
Comments: Accepted to *SEM 2024

- **What's New**: 이 연구에서는 현대의 사전 훈련된 언어 모델과 근사 최근접 이웃 검색 알고리즘을 사용하여 정보 검색 기반 역사전(reverse dictionary) 시스템을 제시합니다. 이 시스템은 에스토니아어 어휘 자원인 'Sõnaveeb'에 적용되어 교차 언어 역사전 기능을 통해 의미 검색(semantic search)을 구현함으로써 향상되고 풍부해집니다.

- **Technical Details**: 사전 훈련된 변환자(transformer) 기반 언어 모델을 사용하여 단어 정의를 인코딩하고, 사용자 입력(개념의 설명 또는 정의)을 동일 모델로 인코딩하여 벡터 데이터베이스에 질의합니다. 이 때, 근사 최근접 이웃 검색(approximate nearest neighbors search) 기술을 통해 관련 단어를 반환합니다. 초기 IR 기반 접근법에서 벗어나, 모던 언어 모델을 활용하여 의미적 검색 기능을 구현합니다.

- **Performance Highlights**: 이 시스템은 문단어(monolingual) 설정에서는 중앙 순위(median rank) 1을, 교차언어(cross-lingual) 설정에서는 중앙 순위 2를 달성했습니다. 특히 에스토니아어를 포함한 교차언어 검색에서 훈련된 모델이 뛰어난 성능을 보였습니다. 또한, 레이블이 없는 평가 방법을 통한 검증도 수행되었습니다.



### Countering Reward Over-optimization in LLM with Demonstration-Guided  Reinforcement Learning (https://arxiv.org/abs/2404.19409)
- **What's New**: 이 연구에서 새롭게 제안된 기법 'Reward Calibration from Demonstration (RCfD)'은 보상 기능 자체를 재보정하는 것에 초점을 맞춰 Reward Over-Optimization (ROO) 문제를 해결합니다. RCfD는 인간의 시연(demonstration)과 보상 모델을 활용하여 언어 모델이 얻는 보상이 시연의 보상과 유사하게 조정되도록 합니다. 이는 보상 모델을 악용하는 것을 방지하고 더 자연스럽고 다양한 언어 생성을 촉진합니다.

- **Technical Details**: RCfD는 기존의 직접적인 보상 최대화에서 벗어나 시연에 근거한 보상 조정으로 전환하여 언어 모델의 보상을 최적화합니다. 이 접근 방식은 보상 모델을 이용하여 과도하게 최적화하는 것을 방지하고, 시퀀스 레벨에서 운영함으로써 노출 편향(exposure bias)을 완화하고 생성된 텍스트의 다양성을 증진시킵니다.

- **Performance Highlights**: RCfD는 언어 모델의 sequence log-likelihood를 최대화하고, 두 가지 RL 언어 태스크에서 기존에 조정된 기준(baseline)과 비교할 만한 성능을 달성하는 것을 실험을 통해 확인했습니다. 또한, RCfD는 다중 보상 설정에서도 효과적으로 작동하여, 서로 충돌할 수 있는 여러 보상들을 최적화하는 과정을 제어합니다. 이는 시연을 통해 Pareto frontier를 목표로 삼는 것을 포함합니다.



### Evaluating Telugu Proficiency in Large Language Models_ A Comparative  Analysis of ChatGPT and Gemin (https://arxiv.org/abs/2404.19369)
- **What's New**: 이 연구는 대규모 언어 모델(LLM, Large Language Model)의 다양한 언어능력을 검토하는 것을 목적으로 하며, 특히 텔루구어 사용 능력에 초점을 맞추었습니다. ChatGPT와 Gemini, 두 개의 주요 LLM을 사용하여 텔루구어 인터랙션의 'NLU(Natural Language Understanding)' 및 'NLG(Natural Language Generation)' 성능을 비교 분석했습니다. 연구는 총 20개의 질문을 설계하여 언어의 이해도와 작업 수행 능력을 평가하였습니다.

- **Technical Details**: 연구 방법론은 텔루구어의 기본 인사, 문법, 어휘 사용 등을 평가하기 위한 질문을 포함하였고, 이는 LLM이 실제 생활 상황에서 얼마나 효율적으로 텔루구어를 사용할 수 있는지를 평가하는 데 도움이 됩니다. 각 모델은 동일한 질문에 대하여 평가받았으며, 'Entity recognition', 'Sentiment analysis', 'Query understanding' 등 다양한 NLU 및 NLG 관련 요소들을 중점적으로 살펴보았습니다. 또한 상황적 이해와 추론 능력 이면의 성능을 분석하였습니다.

- **Performance Highlights**: 결과적으로 두 모델 모두 기본적인 인사와 자기 소개에서는 적절한 성능을 보였지만, 상황에 맞는 농담이나 은유적 표현 사용에서는 한계가 드러났습니다. 특히 Gemini는 보다 정교한 문법 구조와 풍부한 어휘력을 보여주며, 창의적 작업에 필요한 문장을 구성하는 능력이 뛰어났습니다. 반면, ChatGPT는 사실 기반의 정보 검색에서 우세를 보였습니다. 이러한 결과는 텔루구어 능력에 있어서 각 모델의 강점과 약점을 명확하게 보여주고 있으며, 다양한 상황에서의 적응력과 추론 능력 측면에서도 개선이 필요함을 시사합니다.



### Navigating Brain Language Representations: A Comparative Analysis of  Neural Language Models and Psychologically Plausible Models (https://arxiv.org/abs/2404.19364)
- **What's New**: 심리적으로 타당한 모델들이 높은 수준의 신경 언어 모델을 능가한다는 새로운 발견이 이 연구를 통해 밝혀졌습니다. 그 중에서도 신체화된 정보를 포함하는 모델은 단어 및 담화 수준 모두에서 우수한 성능을 보여 주며, 다양한 언어와 모달리티(예: fMRI, 안구 추적)에서의 뇌 활성 예측에서 두드러졌습니다.

- **Technical Details**: 연구팀은 다양한 신경 언어 모델(Neural Language Models, NLMs)과 심리적으로 타당한 모델(Psychologically Plausible Models, PPMs)을 비교 분석하여, 다중 모달 인지 데이터셋을 사용했습니다. 특히, 중국어와 영어 데이터셋을 사용하여 단어 및 담화 수준에서 인코딩 성능을 조사했습니다. fMRI와 안구 추적 데이터를 모두 포함하는 이 분석은, 모델이 뇌의 다양한 영역에서 어떻게 활성화를 예측하는지를 구체적으로 드러냈습니다.

- **Performance Highlights**: 심리적으로 타당한 모델들이, 특히 신체 정보를 통합한 모델이 NLMs보다 전반적으로 우수한 성능을 보였습니다. 이들 모델은 단어 수준에서는 물론 문단 수준의 인코딩에서도 뛰어난 예측력을 보여 주어, 언어 처리 및 인지적 데이터셋에 대한 더 깊은 이해를 가능하게 했습니다. 또한, 다양한 뇌 영역과의 독특한 상관관계를 밝혀내며, 모델 간의 정보 인코딩에 있어서의 차이점을 제시했습니다.



### Expressivity and Speech Synthesis (https://arxiv.org/abs/2404.19363)
Comments: Invited contribution. Under review

- **What's New**: 연구자들은 오랫동안 기계에 말할 수 있는 능력을 부여하는 것을 목표로 해왔습니다. 신속하게 발전하는 표현적 음성 합성(Expressive Speech Synthesis, ESS) 기술이 이제 막 단일 발화에서 아주 훌륭한 수준에 도달한 것으로 보입니다. 이는 복잡하고 장기적인 행동을 합성하는 새로운 가능성을 열어줍니다. 이 장에서는 이러한 성과에 이르게 된 방법론적 진전과 표현력 있는 음성 합성의 다음 단계를 향한 진행중인 노력을 개괄합니다.

- **Technical Details**: 텍스트-투-스피치(Text-to-Speech, TTS)는 초기의 모델 기반 및 규칙 기반 접근 방식에서 데이터 기반의 연접 합성(data-driven concatenative synthesis), 통계 모델(statistical models), 그리고 나중에 딥러닝(Deep Learning, DL)의 출현으로 큰 발전을 이루었습니다. 현재 ESS는 규칙 기반의 초기 방법에서 parametric, 그리고 최근에는 DL 기반의 방법으로 전환되었습니다. 최신 기술은 '확산 모델(diffusion models)'과 같은 확률적 생성 방법론을 활용하여 고도의 자연성과 신뢰성을 제공합니다.

- **Performance Highlights**: 이 최신 연구는 기계가 인간과 구별할 수 없는 점점 더 고급스럽고 표현력 있는 음성을 생성할 수 있게 함으로써, 사용자의 감정 상태를 모방하고 적절한 상황에서 적절한 발화를 생성하는 능력을 크게 향상시킵니다. 또한, 이는 대화형 인터페이스에서 표현상태를 더 정교하고 오래 지속되게 합성하는 연구로 발전하고 있습니다. 이러한 발전은 즉각적으로 맞춤형 대화 생성에 응용될 가능성을 보여줍니다.



### Evaluating Lexicon Incorporation for Depression Symptom Estimation (https://arxiv.org/abs/2404.19359)
Comments: Accepted to Clinical NLP workshop at NAACL 2024

- **What's New**: 이 논문은 우울증 증상 추정을 위해 감정, 정서 및 도메인 특화 사전(lexicons)을 통합한 트랜스포머 기반 모델의 영향을 조사합니다. 환자-치료사 대화의 입력 텍스트와 소셜 미디어 게시물에 단어를 표시하여 사전 정보를 추가했습니다. 특히, 우울증 수준의 추정에서 새로운 최고 기록(state-of-the-art)을 달성하였습니다.



### StablePT: Towards Stable Prompting for Few-shot Learning via Input  Separation (https://arxiv.org/abs/2404.19335)
Comments: Submitted to ACL 2024

- **What's New**: "sysname" 모델의 도입으로 큰 언어 모델들이 데이터 부족의 상황에서도 효율적인 'few-shot learning'을 구현할 수 있게 되었습니다. 이 모델은 prompt 초기화 때 발생하는 노이즈를 줄이기 위해 하드 프롬프트(hard prompt)와 소프트 프롬프트(soft prompt)를 별개의 입력으로 처리하는 새로운 접근 방법을 사용합니다.

- **Technical Details**: "sysname"은 클라스 정보(class-aware information)를 활용하고 모델의 성능을 유지하기 위해 컨트라스티브 러닝(contrastive learning)을 통해 소프트 프롬프트를 최적화합니다. 이를 통해 프롬프트 초기화로 인한 문제를 해결하고, 훈련 과정에서 더욱 효과적으로 정보를 활용할 수 있습니다.

- **Performance Highlights**: 실험 결과에 따르면, "sysname"은 기존 최고 기술 대비 정확도에서 평균 7.20% 높은 성능을 보이고, 표준 편차는 평균 2.02 감소시켰습니다. 아울러, 다양한 작업을 포함하는 7개의 데이터셋에 대한 광범위한 실험을 통해 이 모델의 안정성과 강건성을 확인할 수 있었습니다.



### Computational Approaches for Integrating out Subjectivity in Cognate  Synonym Selection (https://arxiv.org/abs/2404.19328)
Comments: Experiments available on GitHub (this https URL, this https URL)

- **What's New**: 이 논문에서는 언어 계통수(language phylogenetics)를 연구할 때, 동의어(synonyms)를 포함한 새로운 종류의 문자 행렬(character matrices)을 제안합니다. 이러한 문제에 대해 RAxML-NG 도구를 사용하여 최대 가능성 추론(maximum likelihood inference)을 수행함으로써, 어떤 데이터 집합을 대상으로 어떤 문자 행렬 유형이 '골드 스탠더드'(gold standard)와 가장 토폴로지적으로 가까운지를 밝히고 있습니다.

- **Technical Details**: 이 연구는 표준 이진 문자 행렬(standard binary character matrices) 외에 두 가지 새로운 문자 행렬, 즉 확률적 이진 문자 행렬(probabilistic binary character matrices)과 확률적 다값 문자 행렬(probabilistic multi-valued character matrices)을 도입합니다. RAxML-NG 도구는 더 복잡한 데이터 표현을 가능하게 하는 이 문자 행렬들을 입력으로 받아들일 수 있으며, 동의어를 모두 포함하여 데이터셋을 표현할 수 있습니다.

- **Performance Highlights**: RAxML-NG를 사용한 최대 가능성 추론 결과, 동의어를 전부 포함하는 경우와 미리 선택하는 경우의 트리 토폴로지(tree topology)가 크게 다를 수 있음을 보여줍니다. 동의어를 포함한 완성된 데이터셋을 사용했을 때 얻은 트리가 골드 스탠더드에 가장 근접한 경우가 많았습니다. 따라서, 동의어 선택을 수동으로 하기보다는 새로 제안된 문자 행렬을 사용하는 것이 더 바람직할 수 있습니다.



### Knowledge Distillation vs. Pretraining from Scratch under a Fixed  (Computation) Budg (https://arxiv.org/abs/2404.19319)
Comments: Accepted to the 5th Workshop on Insights from Negative Results in NLP at NAACL 2024

- **What's New**: 이 논문은 지식 증류(Knowledge Distillation, KD)가 언어 모형(Language Model, LM) 작은 모델 사전학습(pretraining)에 있어 기존의 기법과 비교할 때 얼마나 효과적인지에 대해 탐구합니다. 특히, 작은 모델을 더 많은 데이터로 학습시켜 큰 모델과의 성능 차이를 줄일 수 있는지에 대한 가능성을 평가하고, 다양한 지식 증류 전략과의 성능을 직접 비교분석하였습니다.

- **Technical Details**: 연구팀은 마스크된 언어 모델링(Masked Language Modeling, MLM)을 위해 기존의 사전학습 방법과 여러 지식 증류 전략을 비교했습니다. 실험은 고정된 계산 예산 하에서 동일한 양의 계산 및 사전학습 데이터를 사용하는 공정한 실험 환경에서 이루어졌습니다. 분석에는 TinyBERT와 MiniLM과 같은 고급 지식 증류 전략이 포함되었습니다.

- **Performance Highlights**: 실험 결과, 표준 사전학습 방법과 일반적인 지식 증류는 비슷한 성능을 보였지만, TinyBERT와 MiniLM과 같은 더 발전된 지식 증류 전략은 뛰어난 성능을 보였습니다. 특히, 고정된 계산 예산 하에서 데이터를 반복해 사용해야 할 경우 지식 증류가 사전학습 대비 훨씬 큰 이점을 제공하는 것으로 나타났습니다. 다운스트림 작업에서의 성능은 GLUE 벤치마크를 통해 평가되었습니다.



### QLSC: A Query Latent Semantic Calibrator for Robust Extractive Question  Answering (https://arxiv.org/abs/2404.19316)
Comments: Accepted by the 2024 International Joint Conference on Neural Networks (IJCNN 2024)

- **What's New**: 이 연구에서는 '질의 잠재 의미 교정기(Query Latent Semantic Calibrator, QLSC)'라는 새로운 모듈을 소개하여 추출형 질의응답(Extractive Question Answering, EQA)에서 형식이 다른 입력을 처리하는 문제를 해결합니다. 이 모듈은 전통적인 질의와 지문 임베딩에 잠재 의미 중심 기능을 통합하여 모델의 정확도를 향상시킵니다.

- **Technical Details**: QLSC는 잠재 의미 중심 특성을 포착하는 스케일링 전략(scaling strategy)과 '부드러운 의미 특성 선택(soft semantic feature selection)'을 사용하여 질의의 의미를 보다 잘 이해할 수 있게 돕습니다. 이 기법은 주목 메커니즘(attention mechanism)을 활용하여 기존의 질의 및 지문 임베딩과 결합되며, 이는 텍스트 형식의 변화에 대한 모델의 민감성을 줄이고 정확한 답변을 추출하는 능력을 강화합니다.

- **Performance Highlights**: 다양한 강인한 질의응답 데이터셋에서의 실험 결과, QLSC를 통합한 EQA 모델은 형식이 변형된 질문들을 효과적으로 처리할 수 있음을 확인했습니다. 이러한 결과는 모델이 원래 질문과 그 변형된 형태 모두에 대해 정확한 답변을 제공할 수 있음을 시연하며, 기존 모델 대비 우수한 강인성을 보였습니다.



### Modeling Orthographic Variation in Occitan's Dialects (https://arxiv.org/abs/2404.19315)
Comments: Accepted at VarDial 2024: The Eleventh Workshop on NLP for Similar Languages, Varieties and Dialects

- **What's New**: 텍스트 데이터의 효과적인 정규화는 표준화된 작문 체제가 결여된 저자원 언어에 대해 상당한 도전을 제기합니다. 이 연구에서는 여러 오크시탄 방언 데이터를 사용하여 다국어 모델을 미세 조정하고, 이 방언들의 표현을 평가하기 위한 일련의 실험을 수행하였습니다.

- **Technical Details**: 연구팀은 네 가지 오크시탄 방언을 포함하는 병렬 어휘집 (parallel lexicon)을 구축하여 평가를 진행했습니다. 모델의 임베딩 (embeddings)에 대한 내재적 평가는 방언 간의 표면적 유사성이 표현을 강화한다는 것을 밝혔습니다. 또한, 품사 태깅 (part-of-speech tagging) 및 범용 의존성 구문 분석 (Universal Dependency parsing)을 위해 모델을 추가로 미세 조정했을 때, 단일 방언의 품사 데이터만으로 훈련되었음에도 불구하고 모델의 성능은 방언 변화에 강한 것으로 나타났습니다.

- **Performance Highlights**: 대형 다국어 모델은 전처리 과정 중 맞춤법 정규화의 필요성을 최소화할 수 있다는 점을 시사하며, 그 성능은 방언적 변이에 강력하게 대응할 수 있는 것으로 평가되었습니다.



### Does Whisper understand Swiss German? An automatic, qualitative, and  human evaluation (https://arxiv.org/abs/2404.19310)
Comments: Accepted to VarDial 2024 (the eleventh Workshop on NLP for Similar Languages, Varieties and Dialects 2024), Mexico City

- **What's New**: Whisper는 Radford와 그의 동료들에 의해 개발된 최신 자동 음성 인식(ASR) 모델입니다. 이 연구에서는 Whisper가 스위스 독일어 방언을 인식하는 능력을 평가합니다. 특히, Whisper가 훈련 데이터에 스위스 독일어가 포함되어 있지 않음에도 불구하고 이 언어를 잘 인식하고 표준 독일어로 번역하는 것으로 나타났습니다.

- **Technical Details**: 이 연구는 세 개의 기존 테스트 세트(SwissDial, STT4SG-350, Swiss Parliaments Corpus)와 새로운 모의 임상 인터뷰 기반 테스트 세트를 사용하여 Whisper의 성능을 체계적으로 평가합니다. 자동 평가는 단어 오류율(WER)과 BLEU 점수를 사용하고, 질적 분석에서는 Whisper의 강점과 약점을 논의하며, 출력 예시를 분석합니다. 인간 평가는 28명의 참가자가 Whisper의 성능을 평가하는 설문조사를 실시하여 이루어졌습니다.

- **Performance Highlights**: 모든 평가 결과에 따르면, Whisper는 표준 독일어 출력이 필요한 경우 스위스 독일어를 처리하기에 적합한 ASR 시스템으로 나타났습니다. Whisper는 특히 스위스 독일어를 인식한 후 표준 독일어로의 번역에서 높은 성능을 보였습니다. 이는 자동, 질적, 인간 평가 모두에서 확인되었습니다.



### Octopus v4: Graph of language models (https://arxiv.org/abs/2404.19296)
- **What's New**: 새로운 Octopus v4 모델은 여러 오픈소스 모델을 통합하여 최적화된 기능 토큰(functional tokens)을 사용합니다. 이 모델은 사용자 질의를 가장 적절한 특정 모델로 지능적으로 연결하고, 질의를 재구성하여 최상의 성능을 달성합니다.

- **Technical Details**: Octopus v4는 함수 토큰을 사용하여 사용자 질의를 적절한 모델로 인도하고 질의를 재구성하는 기능을 갖추고 있습니다. 또한, 그래프(graph) 데이터 구조를 사용하여 여러 오픈소스 모델의 조정과 통합을 효과적으로 관리합니다.

- **Performance Highlights**: 10B 매개변수 미만의 모델을 활성화하여 동급 모델 중 SOTA MMLU 점수 74.8을 달성했습니다. 이것은 비슷한 수준의 모델들 사이에서 상당히 높은 점수입니다.



### Aspect and Opinion Term Extraction Using Graph Attention Network (https://arxiv.org/abs/2404.19260)
- **What's New**: 이 연구에서는 그래프 주목 네트워크(Graph Attention Network, GAN)가 어떻게 관점(aspect)과 의견(opinion) 용어를 추출하는지 조사합니다. 관점과 의견 용어 추출은 명명된 개체 인식(named entity recognition)과 유사한 토큰 수준 분류 작업으로 제안됩니다. 입력 쿼리의 의존성 트리(dependency tree)를 토큰 및 품사(part-of-speech) 기능과 함께 GAN에 추가 기능으로 사용합니다.

- **Technical Details**: 이 연구는 GAN에 CRF(Conditional Random Field) 계층과 함께 의존성 구조가 성능을 크게 향상시키고 SemEval 2014, 2015, 2016에서 사용된 데이터셋에 대해 최고의 결과를 생성한다는 것을 보여줍니다. 추가적으로 BiLSTM과 Transformer 계층도 CRF 계층과 함께 실험했습니다.

- **Performance Highlights**: 이 방법은 단일 관점에 기반한 의존성 트리를 수정할 필요 없이 같은 쿼리에서 다수의 관점이나 감정(sentiments)이 존재할 때도 잘 작동한다는 것을 입증합니다. 예시를 들어, SemEval 데이터셋에서 베스트 결과를 도출하기 위해 의존성 구조를 활용했습니다.



### Suvach -- Generated Hindi QA benchmark (https://arxiv.org/abs/2404.19254)
- **What's New**: 이 논문은 힌디어 추출형 질문 응답(EQA: Extractive Question Answering) 모델을 평가할 수 있는 새로운 벤치마크를 제안합니다. 기존에는 영어 데이터셋을 기계 번역하는 방식으로 인도어(Indic languages) EQA 모델을 평가했지만, 이는 번역의 편향과 부정확성 때문에 진정한 모델 성능을 반영하지 못하는 문제가 있었습니다. 새로운 벤치마크는 이러한 문제를 해결하고자 힌디어에 특화된 질문과 답변 데이터셋을 구축하여 보다 정확하고 신뢰성 있는 평가 도구를 제공하고자 합니다.

- **Technical Details**: 연구진은 대규모 언어 모델(LLMs: Large Language Models)을 활용하여 고품질의 힌디어 데이터셋을 생성합니다. 이 데이터셋은 추출형 설정(extractive setting)에서 수집되며, 특정 기술을 사용해 언어의 특성과 문맥을 깊이 있게 반영합니다. 이러한 접근 방식은 기계 번역에 의존하지 않으므로 원본 언어의 뉘앙스를 더 잘 보존하고, 질문 응답 모델 평가가 더 정밀해질 것으로 기대됩니다.

- **Performance Highlights**: 새로운 벤치마크는 힌디어 NLP(Natural Language Processing) 분야의 연구에 기여할 것으로 예상됩니다. 이는 힌디어 EQA 모델들이 실제 언어 사용 환경과 더 일치하는 평가를 가능하게 함으로써, 모델의 성능 개선에 중요한 통찰을 제공할 수 있기 때문입니다. 추가적으로, 이 벤치마크는 다른 인도어에 대해서도 확장 가능한 방법론을 제안함으로써, 광범위한 언어에 대한 정확한 평가 기준을 마련할 가능성을 제시합니다.



### Exploiting Hatred by Targets for Hate Speech Detection on Vietnamese  Social Media Texts (https://arxiv.org/abs/2404.19252)
- **What's New**: 이 논문은 ViTHSD (Vietnamese Targeted Hate Speech Detection) 데이터셋을 소개하여 베트남어 소셜 미디어 텍스트를 위한 '타겟된 혐오 발언 감지'에 대한 새로운 접근법을 제시합니다. 여기서 개인, 집단, 종교, 정치 등을 대상으로 한 다양한 '타겟'과 이를 향한 '혐오 수준'을 평가합니다.

- **Technical Details**: 이 데이터셋은 10K의 댓글을 포함하며, 각 댓글은 사람에 의해 엄격한 어노테이션 가이드라인(annotation guidelines)에 따라 클린(clean), 공격적인(offensive), 혐오적인(hate)의 세 가지 레벨로 수동적으로 레이블이 지정됩니다. 데이터셋의 타겟은 Bi-GRU-LSTM-CNN과 사전 훈련된 언어 모델이 결합된 베이스라인 모델을 사용하여 식별됩니다. 이 모델은 뛰어난 텍스트 표현 능력을 갖는 BERTology를 활용합니다.

- **Performance Highlights**: 베이스라인은 타겟 검출(target detection)과 타겟 레벨 감지(targe level detection) 두 가지 작업으로 세분화하여 평가되며, 정밀도(Precision), 재현율(Recall), F1-점수(F1-score)를 사용하여 성능을 측정합니다. 또한, 이 시스템은 실시간 데이터 처리 도구인 Spark Streaming과 Apache Kafka를 사용하여 YouTube 라이브 스트림에서 오는 댓글을 실시간으로 처리할 수 있는 방법론을 제안합니다.



### HydraLoRA: An Asymmetric LoRA Architecture for Efficient Fine-Tuning (https://arxiv.org/abs/2404.19245)
Comments: 19 pages, 7 figures

- **What's New**: 새로운 HydraLoRA 기법은 기존의 LoRA 프레임워크를 개선하여 대규모 언어 모델(Large Language Models, LLMs)을 새로운 과제에 효과적으로 적용할 수 있도록 합니다. 이는 비대칭 구조(asymmetric structure)를 도입하여 도메인 전문지식 없이도 훈련과 추론 단계에서 탁월한 성능을 발휘합니다.

- **Technical Details**: HydraLoRA는 공유 A 행렬(shared A matrix)과 여러 개의 B 행렬(multiple B matrices)을 사용하는 구조를 가지고 있습니다. 이는 모델이 'intrinsic components'를 자동으로 식별하고, 훈련 샘플을 구별된 B 행렬로 분류할 수 있게 하여, 추론 단계에서 여러 B 행렬을 expert 혼합(mixture-of-experts, MoE) 방식으로 활용합니다.

- **Performance Highlights**: HydraLoRA는 다른 Parameter-Efficient Fine-Tuning (PEFT) 접근법들을 능가하는 성능을 보여주며, 특히 도메인 지식을 활용하는 훈련 과정에서도 더 우수한 결과를 나타냈습니다. 이는 복잡한 데이터 집합에서도 강력한 일반화 성능을 제공합니다.



### GRAMMAR: Grounded and Modular Evaluation of Domain-Specific  Retrieval-Augmented Language Models (https://arxiv.org/abs/2404.19232)
- **What's New**: 이 연구는 도메인별 지식 기반을 질의하는 검색 강화 생성(Retrieval-augmented Generation, RAG) 시스템의 평가에 대한 새로운 접근 방식인 GRAMMAR (GRounded And Modular Methodology for Assessment of RAG) 평가 프레임워크를 소개합니다. 이 평가 방법은 관계형 데이터베이스와 LLMs를 활용한 데이터 생성 과정과 모듈 결함을 식별하는 평가 프레임워크를 포함합니다.

- **Technical Details**: GRAMMAR 평가 프레임워크는 1) SQL 쿼리를 통해 진실한 답변을 추출할 수 있도록 관계형 데이터베이스와 대규모 언어 모델(LLMs)을 활용하여 쿼리-답변 쌍을 효율적으로 생성하는 데이터 생성 프로세스; 2) 지식 격차와 강인성을 구분하고 결함이 있는 모듈을 식별할 수 있는 평가 프레임워크를 포함합니다. 특히, 동일한 의미를 가진 쿼리에 대해 모델의 반응이 언어적 표현의 차이에 따라 일관성이 없는 점을 다룹니다.

- **Performance Highlights**: GRAMMAR 이용의 효능은 현재의 참조 없는 평가 방식의 한계를 강조하며, 정확하게 모델 취약성을 식별할 수 있다는 점에서 신뢰성을 보여줍니다. 이 평가는 도메인 특화 RAG 시스템에 필수적인 정확한 지식 소스와 강인성을 평가하는 능력을 향상시킵니다.



### Mix of Experts Language Model for Named Entity Recognition (https://arxiv.org/abs/2404.19192)
- **What's New**: 새로운 자연어 처리(Natural Language Processing, NLP) 기술인 BOND-MoE는 멀리서 관리되는 데이터의 문제를 극복하기 위해 제안됐습니다. BOND-MoE는 Mixture of Experts (MoE) 기반의 강인한 Named Entity Recognition (NER) 모델로, 완벽하지 않고 잡음이 많은 주석이 모델 훈련을 오도할 수 있는 기존의 문제점을 해결합니다.

- **Technical Details**: BOND-MoE는 여러 모델을 훈련시키고 Expectation-Maximization (EM) 프레임워크 아래에서 앙상블하여 잡음이 많은 감독을 크게 줄입니다. 이 방법은 단일 모델에 의존하는 대신에 여러 전문가의 의견을 결합합니다. 또한 공정한 할당 모듈을 도입하여 문서-모델 할당 과정을 균형있게 조정합니다.

- **Performance Highlights**: 실제 데이터셋에 대한 광범위한 실험을 통해, BOND-MoE는 멀리서 감독받는 다른 NER 모델들에 비해 최고의 성능을 달성했습니다. 이 결과는 BOND-MoE가 잡음이 많은 주석 문제를 효과적으로 극복할 수 있음을 시사합니다.



### Revenge of the Fallen? Recurrent Models Match Transformers at Predicting  Human Language Comprehension Metrics (https://arxiv.org/abs/2404.19178)
- **What's New**: 최근 개발된 순환 신경망 아키텍처인 RWKV와 Mamba는 자연어 처리 과제에서 동급의 트랜스포머(transformer)와 비교하여 유사하거나 더 나은 성능을 보여줍니다. 이 논문은 이 두 가지 순환 모델(recurrent models)이 온라인 인간 언어 이해를 모델링하는데 있어 비슷한 크기의 트랜스포머의 성능을 맞추거나 넘어설 수 있음을 보여줍니다.

- **Technical Details**: RWKV와 Mamba는 순환 신경망 아키텍처로서, 자연어 처리 작업을 위해 설계되었습니다. 이들 아키텍처는 예측 가능성이 온라인 인간 언어 이해에 미치는 영향을 모델링하는 임무에서 트랜스포머와 비교됩니다. 본 연구에서는 이러한 순환 모델들이 트랜스포머와 비슷한 규모에서 인간 언어 이해를 모델링하는 데 있어 비슷하거나 더 높은 성능을 달성할 수 있음을 실험적으로 입증하였습니다.

- **Performance Highlights**: RWKV와 Mamba 모델은 온라인 언어 이해 영역에서 트랜스포머와 비교하여 탁월한 성능을 제공합니다. 이 결과는 트랜스포머 언어 모델이 이 작업에 유일하게 적합한 것은 아니라는 점을 시사하며, 언어 모델의 아키텍처적 특성이 인간 언어 이해 모델링에 더 나은 것인지에 대한 논의에 새로운 방향을 제시합니다.



### Game-MUG: Multimodal Oriented Game Situation Understanding and  Commentary Generation Datas (https://arxiv.org/abs/2404.19175)
- **What's New**: 이 논문은 새로운 다중 모드 게임 상황 이해 및 관객 참여형 코멘터리 생성 데이터셋인 GAME-MUG를 소개하고, 강력한 베이스라인을 제공합니다. 특히 이 데이터셋은 2020년부터 2022년까지 YouTube와 Twitch에서 방송된 LOL(Legend of Legends) 게임 라이브 스트림에서 수집된 정보를 기반으로 합니다.

- **Technical Details**: GAME-MUG 데이터셋은 텍스트, 오디오, 시간-시리즈 이벤트 로그를 포함한 다중 모드 이스포츠 게임 정보를 포함하고 있으며, 게임 상황을 감지하는 데 사용됩니다. 또한, 게임 상황과 관객 대화 이해를 다루고 강력한 조인트 이중 모드 학습 모델(Joint Multimodal Dual Learning Model)을 소개하여 관객 대화 증강 코멘터리 데이터셋도 제안합니다.

- **Performance Highlights**: 이 모델은 게임 상황 및 이벤트 이해 능력과 코멘터리 생성 기능을 통해 다중 모드 적용의 효과성과 조인트 통합 학습 접근법의 장점을 검증합니다.



### What Drives Performance in Multilingual Language Models? (https://arxiv.org/abs/2404.19159)
Comments: Accepted at VarDial @ NAACL 2024

- **What's New**: 이 연구에서는 다양한 언어에서 다국어 대규모 언어 모델(MLLMs)의 성능에 영향을 미치는 요인을 탐구합니다. 우리는 마스크 언어 모델(Masked Language Models), 자동 회귀 모델(autoregressive models), 지시적 튜닝된 LLMs(instruction-tuned LLMs)을 포함한 6개의 MLLMs를 204개 언어가 포함된 주제 분류 데이터셋인 SIB-200에서 평가했습니다. 연구는 모든 언어(ALL), 모델의 사전 훈련 데이터에 포함된 언어(SEEN), 그리고 사전 훈련 데이터에서 의미 있게 문서화되지 않은 새로운 언어(UNSEEN)의 세 가지 시나리오를 고려하여 이루어졌습니다. 사전 훈련 데이터 크기, 자원의 가용성, 언어 가족, 그리고 문자 유형과 같은 다양한 요인들이 모델 성능에 미치는 영향을 분석했습니다.

- **Technical Details**: 선택된 MLLMs는 다양한 크기의 mBERT, XLM-R, GPT-3.5, Bloom, Bloomz 및 XGLM 모델을 포함합니다. 이 연구에서는 제로-샷(zero-shot), 2-샷(2-shot), 완전 감독(fully supervised) 패러다임하에 평가되었습니다. 가장 주요한 요인으로는 SEEN 언어의 경우 사전 훈련 데이터의 크기가, UNSEEN 언어의 경우 문자 유형과 언어 가족이 각각 중요함을 밝혀내었습니다. 또한, 진행된 연구는 SIB-200 데이터셋을 사용하여 평가하였으며, 이는 204개의 다양한 언어를 포함하고 있습니다.

- **Performance Highlights**: 분석 결과, 사전 훈련 데이터 크기가 SEEN 언어에서 가장 큰 영향을 미치는 반면, UNSEEN 언어의 경우에는 문자 유형과 언어 가족이 중요한 요인으로 나타났습니다. 언어 모델의 크기나 아키텍처가 중요한 특징에 크게 영향을 미치지 않는 것으로 조사되었습니다. 이러한 발견은 다국어 NLP 시스템의 개발을 안내하는데 유용한 통찰력을 제공합니다.



### RTF: Region-based Table Filling Method for Relational Triple Extraction (https://arxiv.org/abs/2404.19154)
Comments: Rejected by EMNLP 2023

- **What's New**: 이 연구는 지역 기반 표 채우기 방법(Region-based Table Filling, RTF)을 제안하여, 관계 삼중항(Relational Triple) 추출에 있어서 고전적인 문제점인 공간적 지역 의존성을 무시하는 문제를 해결합니다. RTF 방법은 새로운 지역 기반 태깅 체계(Entity Pair as Region, EPR tagging scheme)와 양방향 디코딩 전략(bi-directional decoding strategy)을 사용하여 각 관계의 삼중항을 하나의 지역으로 간주하고, 이 지역의 두 끝점을 정해 삼중항을 식별합니다.

- **Technical Details**: RTF는 컨볼루션(Convolution)을 이용하여 표의 지역 수준 표현을 구성하고, 다른 관계들 간의 부분 태깅 점수를 공유하여 관계 분류기의 학습 효율을 개선합니다. 이 방법은 각 토큰 쌍이 자체적으로만 의존하는 것이 아니라 주변 토큰 쌍에도 의존하도록 하여, 컨볼루션 커널이 표를 슬라이딩할 때 전체 삼중항 또는 삼중항 간 상호작용을 동시에 처리할 수 있습니다.

- **Performance Highlights**: 이 방법은 NYT와 WebNLG 데이터셋의 세 가지 변형에서 최신 최고 성능(state-of-the-art)을 달성하였으며, 우수한 일반화 능력을 보여줍니다. 기존 연구들과 비교하여 공간적 지역 의존성을 활용하는 점에서 차별화되며, 관계 특정 표 구조에서 지역 수준 표현을 활용하여 인식 성능을 개선합니다.



### Accelerating Production LLMs with Combined Token/Embedding Speculators (https://arxiv.org/abs/2404.19124)
- **What's New**: 이 기술 보고서는 대형 언어 모델(Large Language Models, LLMs)의 추론 속도를 가속화하기 위해 새로운 추측 디코딩(speculative decoding) 초안 모델을 설계하고 훈련하는 방법에 관한 것입니다. 샘플링된 토큰과 함께 컨텍스트 벡터(context vectors)에 기초하여 추측을 조건부로 진행함으로써, 우리는 기존 모델이 수용하거나 거부할 수 있는 고품질의 n-gram을 효율적으로 예측할 수 있습니다. 이를 통해 추론 단계에서 여러 토큰을 예측할 수 있으며, 이는 기존 최적화된 베이스 모델의 실제 시계 기준 추론 속도를 2-3배 가속화합니다.

- **Technical Details**: 본 논문에서는 IBM Research의 최근 노력을 설명하며, Medusa라고 하는 새로운 아키텍처를 기반으로 추측 디코딩을 사용하여 생산 LLM 추론을 가속화합니다. 기존에는 독립적인 작은 LLM이 베이스 모델을 위한 초안을 제공했으나, Medusa 모델은 베이스 모델의 최신 토큰 예측에서 얻은 임베딩 벡터만을 입력으로 사용합니다. 이 임베딩 벡터는 그 자체로 이전 입력으로부터 파생된 맥락 정보를 암시적으로 포함하고 있습니다. 추가적으로, 우리는 샘플링된 토큰에 기초한 조건을 추가함으로써 추측기의 출력 품질을 크게 향상시킬 수 있었습니다.

- **Performance Highlights**: 이 연구에서 개발된 추측 디코딩 방식은 베이스 모델의 기존 추론 속도를 최대 3배까지 향상시킬 수 있었습니다. 특히, 우리가 사용한 고도로 최적화된 생산 LLM을 이용할 때, 추론 속도의 향상이 두드러졌습니다. 연구 코드는 오픈 소스로 제공되며 https://github.com/foundation-model-stack/fms-fsdp 에서 확인할 수 있으며, 13B파라미터 베이스 모델을 위한 추측기는 HuggingFace에서도 확인할 수 있습니다: https://huggingface.co/ibm-fms.



### Effects of Added Emphasis and Pause in Audio Delivery of Health  Information (https://arxiv.org/abs/2404.19119)
Comments: This manuscript is accepted to American Medical Informatics Association summit, 2024

- **What's New**: 새로운 연구에서는 건강 정보의 이해와 유지에 대한 오디오 강조 방식의 효과를 평가합니다. 다양한 난이도의 건강 텍스트를 사용하여 정보 강조와 휴식(Pause)의 추가가 어떤 영향을 미치는지 조사하였습니다.

- **Technical Details**: 이 연구는 아마존 메카니컬 터크(Amazon Mechanical Turk, AMT)에서 수행되었고, 어려운 텍스트와 쉬운 텍스트로부터 오디오 스니펫(Audio snippets)을 생성하여 실험하였습니다. 정보 강조(Emphasis)와 휴식의 추가가 건강 정보의 이해(Comprehension) 및 유지(Retention)에 미치는 영향을 분석했습니다.

- **Performance Highlights**: 강조가 추가되지 않았을 때보다 올바르게 강조된 어려운 텍스트에서 정보 이해도는 54%로 증가했으며, 휴식의 추가는 인지된 난이도를 낮추고 유지력을 향상시켰지만, 정보의 이해에는 부정적인 영향을 미쳤습니다.



### In-Context Symbolic Regression: Leveraging Language Models for Function  Discovery (https://arxiv.org/abs/2404.19094)
- **What's New**: 이 연구는 기호 회귀(Symbolic Regression, SR)에 대한 대규모 언어 모델(Large Language Models, LLMs)의 적용을 처음으로 탐구하고, 관찰 세트에서 예측 오류를 기반으로 기능 형태를 반복적으로 미세 조정하는 접근 방식을 사용합니다. 특히 비전-언어 모델(Vision-Language Models, VLMs)의 도입을 통해 최적화 과정에서 플롯(plots)을 시각적 입력으로 활용하는 새로운 방법을 제안하여 복잡한 벤치마크에 대한 흥미로운 결과를 보여줍니다.

- **Technical Details**: 이 연구에서는 사전 훈련된 LLM을 사용하여 초기 함수 세트를 제안하고, 이 함수들을 모델 자체와 외부 최적화자를 통해 계수와 함께 반복적으로 미세 조정합니다. 이 과정은 만족스러운 결과가 나올 때까지 반복됩니다. 또한, VLM을 사용하여 최적화 과정에 플롯을 도입하고, 이것이 수학적 개념을 전달하는데 어떻게 도움이 되는지 탐구합니다.

- **Performance Highlights**: LLMs를 활용한 이 접근법은 유전 프로그래밍(Genetic Programming, GP)에 기반한 기호 회귀 베이스라인을 능가하며, 입력으로 이미지를 추가하는 것이 가장 복잡한 벤치마크에서 유망한 결과를 보여줍니다. 이는 LLMs의 추론 및 일반화 능력을 활용하여 데이터에 적합한 기호 방정식을 성공적으로 복구할 수 있음을 보여줍니다.



### SuperCLUE-Fin: Graded Fine-Grained Analysis of Chinese LLMs on Diverse  Financial Tasks and Applications (https://arxiv.org/abs/2404.19063)
Comments: 11 pages, 19 figures, and tables

- **What's New**: SuperCLUE-Fin (SC-Fin) 벤치마크는 중국 금융 대규모 언어 모델(LLM)의 평가를 위한 최초의 표준입니다. 이는 금융 이론에서 실제 적용에 이르기까지 범위를 다루면서 6가지 금융 응용 영역과 25가지 과제를 포함하여 모델의 금융 이해력과 논리적 추론, 명료성, 계산 효율성을 평가합니다. 중국 규제에 대한 준수도 중요한 평가 기준입니다.

- **Technical Details**: SC-Fin은 멀티턴, 개방형 대화를 사용하여 실제 상황을 모방합니다. 평가는 기본 기능(Basic Capabilities)과 응용 능력(Applied Capabilities)의 두 가지 주요 차원으로 구분됩니다. 여기에는 판단 정확성(Information Correctness, IC), 금융 논리(Logic of Finance, LF), 언어 이해력(Language Intelligibility, LI) 등 여러 기준에 따라 평가됩니다. GLM-4와 MoonShot-v1-128k 같은 국내 모델들은 타사 모델들을 능가하는 A등급의 성능을 보여주었습니다.

- **Performance Highlights**: 천여 개의 질문을 포함한 광범위한 데이터세트를 통해 SC-Fin은 금융 관리(Financial Management), 증권(Securities), 투자(Investment) 등 다양한 재무 부문에서 모델의 성능을 종합적으로 평가하였습니다. 평가 결과, 특히 주요 금융 지식 데이터베이스를 개선하고, 금융 해석을 표준화하며, 준수 및 위험 관리를 우선시하는 모델을 촉진하는 데 중요한 도구로 자리 매김하였습니다.



### Plan of Thoughts: Heuristic-Guided Problem Solving with Large Language  Models (https://arxiv.org/abs/2404.19055)
Comments: 7 pages, 2 figures

- **What's New**: 새로운 연구에서는 언어 모델(Language Models, LMs)이 단순히 제로샷(Zero-Shot) 추론 작업에서 유용하지만, 다단계 추론을 필요로 하는 문제에서는 만족스럽게 수행되지 않는다는 점을 개선하기 위해 노력하고 있습니다. 이를 위해, 복잡한 다단계 작업을 여러 하위 작업으로 분할하고 언어 모델이 각 하위 작업마다 제안('생각')을 생성하도록 하며, 그러한 제안들을 통합하는 새로운 계획 기반 접근 방식을 공식화했습니다.

- **Technical Details**: 연구에서는 부분 관찰 가능 마르코프 결정 과정(Partially Observable Markov Decision Processes, POMDPs)을 사용하여 다단계 문제 해결을 위한 계획 기반 접근법을 소개합니다. POMDP 내에서 주어진 상태의 가치에 대한 언어 모델의 자체 반영을 탐색 휴리스틱으로 사용하면서, 온라인 POMDP 솔버인 POMCP(Online Partially Observable Monte Carlo Planning)를 활용하여 기존 접근법보다 높은 성공률을 달성하였습니다.

- **Performance Highlights**: 게임 오브 24(Game of 24) 작업에서 89.4%의 높은 성공률을 보여주었으며, 이는 고정된 트리 검색(Fixed Tree-Search)을 사용하는 기존 방법보다 더 나은 '언제든지 성능'(anytime performance) 특성을 제공합니다.



### A Framework for Real-time Safeguarding the Text Generation of Large  Languag (https://arxiv.org/abs/2404.19048)
- **What's New**: LLMSafeGuard는 실시간으로 대규모 언어 모델(Large Language Models, LLMs)의 텍스트 생성을 안전하게 관리하는 새로운 프레임워크를 제안합니다. 이는 유사성 기반의 검증 접근 방식(similarity-based validation approach)를 통해 안전 제약 조건을 위반하는 후보를 거부하고 유효한 후보를 허용함으로써, 특정 제어 모델을 훈련할 필요 없이 새로운 안전 제약 조건을 손쉽게 도입할 수 있게 합니다.

- **Technical Details**: LLMSafeGuard는 Beam Search 알고리즘과 외부 검증기(external validator)를 통합하여 실시간으로 최상위 후보를 검증합니다. 이 프레임워크는 데모 예제(demonstration examples)를 이용해 후보와의 유사성을 평가하고, 높은 유사성을 보이는 후보는 즉시 거부하며, 그렇지 않은 후보는 Beam Search를 통해 진행됩니다. 또한, 필요할 때만 LLM에 개입하는 컨텍스트별 타이밍 선택 전략(context-wise timing selection strategy)을 사용하여 불필요한 개입을 최소화하고 계산 오버헤드를 줄입니다.

- **Performance Highlights**: LLMSafeGuard는 탈독성(detoxification) 및 저작권 보호(copyright safeguarding) 작업에서 기존 최상의 기준보다 뛰어난 성능을 보였습니다. 특히, 탈독성 작업에서 LLM의 평균 독성 점수를 29.7% 감소시켰으며, 문맥 타이밍 선택 전략은 추론 시간을 최소 24% 줄이면서도 비슷한 효과를 유지합니다. 또한, LLMSafeGuard는 효과성과 효율성을 조절할 수 있는 조정 가능한(tunable) 매개변수를 제공합니다.



### How Did We Get Here? Summarizing Conversation Dynamics (https://arxiv.org/abs/2404.19007)
Comments: To appear in the Proceedings of NAACL 2024. Data available in ConvoKit this https URL

- **What's New**: 이 연구는 대화 동안 발생하는 상호 작용의 동역학을 요약하는 새로운 작업을 제안합니다. 대화의 주제적 내용(Context)뿐만 아니라 참가자들 사이의 상호 작용을 분석하여 대화의 전개 경로를 이해하는 데 도움을 주는 방법을 모색합니다. 이를 위해 인간이 작성한 요약 데이터셋을 구축하고 여러 자동화된 기준 모델들을 탐색하였습니다.

- **Technical Details**: 개발된 시스템은 대화의 주제뿐만 아니라 톤 변경(Tone), 전략 사용(Strategy), 상호작용 패턴(Interaction Patterns)등 참여자간의 상호작용 동역학을 파악하고 요약합니다. 연구팀은 인간 주석자들이 협력하여 대화 동역학 요약(Conversation Dynamics Summaries, SCDs)을 작성하는 다단계 절차를 개발하였으며, 큰 언어 모델을 사용하여 SCD를 생성하는 프롬프트를 개발하여 전통적인 요약과 비교하였습니다.

- **Performance Highlights**: SCD는 대화의 경로를 이해하고 톡식 행동(Toxic Behavior)으로 치우칠 가능성을 예측하는 데 있어 유용하다는 것을 보여줍니다. 인간은 SCD를 읽을 때 대화의 트랜스크립트를 직접 읽는 것보다 세 배 빠른 속도로, 더 큰 확신을 가지고 예측할 수 있었습니다. 또한, 자동화된 시스템은 대화의 전문을 직접 사용하는 것보다 SCD를 바탕으로 예측할 때 더 정확한 결과를 보였습니다.



### Markovian Agents for Truthful Language Modeling (https://arxiv.org/abs/2404.18988)
Comments: 21 pages, 6 figures

- **What's New**: 이 연구는 언어 모델(LM)의 '사고의 사슬' (Chain-of-Thought, CoT) 추론 방식을 향상시키기 위해 '마르코프 훈련' (Markovian Training) 절차를 도입합니다. 이 훈련 방법은 CoT가 독립적으로 텍스트의 미래를 예측할 수 있도록 함으로써, 언어 모델이 실제로 CoT를 이해하고 사용한다는 것을 보장합니다.

- **Technical Details**: 이 연구는 CoT의 유효성을 평가하고, CoT만을 사용하여 미래의 텍스트를 예측하는 마르코프 언어 모델을 정의합니다. 연구팀은 Proximal Policy Optimization (PPO)과 정책 그라디언트를 사용하여 비차별화된 CoT 토큰 생성 문제를 강화 학습 작업으로 형식화했습니다. 이를 통해 모델이 CoT를 생성하고 활용하는 과정을 최적화합니다.

- **Performance Highlights**: 마르코프 훈련 기법은 긴 컨텍스트 산술 문제에서 효과적임을 입증하였습니다. 또한, 생성된 CoT가 모델 추론 과정의 중요 부분으로 활용됨을 확인하였고, 다른 모델에서도 의미 있고 사용 가능하다는 것을 검증하였습니다. 이는 CoT의 해석 가능성과 전이 가능성을 크게 향상시키는 결과를 가져왔습니다.



### Computational Job Market Analysis with Natural Language Processing (https://arxiv.org/abs/2404.18977)
Comments: Ph.D. Thesis (315 total pages, 52 figures). The thesis slightly modified with this https URL ISBN (electronic): 978-87-7949-414-5

- **What's New**: 이 논문에서는 일자리 데이터에서 유의미한 정보를 추출하기 위한 천연어 처리(Natural Language Processing, NLP) 기술이 조사되었습니다. 특히, 공개된 언어 기술 시스템과 데이터 부족에 초점을 맞추면서, 작업 설명에서 중요 정보를 추출하기 위한 새로운 기법과 도전 과제들을 식별하고 논의합니다. 연구는 직업 설명 데이터셋(job description datasets), 비식별화 데이터셋(de-identification dataset), 그리고 효율적인 모델 훈련을 위한 새로운 활성 학습 알고리즘(active learning algorithm)을 포함한 여러 기여를 제공합니다.

- **Technical Details**: 이 논문에서 제안하는 기술적 접근 방식은 다국어 언어 모델(multilingual language models)을 직업 시장 도메인에 맞게 세밀하게 조정하는 분류 의식적 사전 훈련(taxonomy-aware pre-training) 방법론을 포함합니다. 또한, 다중 스킬 추출 데이터셋(multiple skill extraction datasets)을 활용하는 검색 증강 모델(retrieval-augmented model)을 통해 전반적인 성능을 향상시키는 방법을 제안합니다. 스킬 추출(skill extraction)은 약한 감독(weak supervision)을 사용하여 이루어집니다. 여러 데이터셋을 활용하여 스킬의 빈도에 따른 긴 꼬리 패턴(long-tail frequency pattern)을 처리하는 전략도 모색하였습니다.

- **Performance Highlights**: 제안된 시스템은 다양한 언어와 플랫폼에 걸친 고용 전망과 일자리 공석 데이터를 통합하여 노동 시장 수요에 대한 깊은 통찰력을 제공하며, 새로운 기술의 출현을 감지하고 다양한 이해관계자들을 위한 일자리 매칭을 용이하게 할 수 있다는 잠재력을 드러냅니다. 이는 특히 표준화된 주석 가이드라인이 부족하고 효과적인 추출 방법의 부재가 해결되면 더욱 효과적일 수 있습니다. 또한, 스킬 추출 모델의 예측을 평가하고 분석하는 비교 가능한 프레임워크를 제공함으로써, ESCO와 같은 세분화된 분류 체계를 활용한 라벨 공간의 넓은 범위를 다룰 수 있습니다.



### Credible, Unreliable or Leaked?: Evidence Verification for Enhanced  Automated Fact-checking (https://arxiv.org/abs/2404.18971)
- **What's New**: 본 논문은 자동 팩트체킹(Automated Fact-Checking, AFC) 시스템에서 '유출된 증거(leaked evidence)'와 '신뢰할 수 없는 증거(unreliable evidence)' 문제에 집중합니다. 이를 해결하기 위해 'CREDible, Unreliable or LEaked (CREDULE)' 데이터셋을 구축하고, EVidence VERification Network (EVVER-Net)을 개발하여 유출 및 신뢰할 수 없는 증거를 필터링하는 능력을 향상시켰습니다.

- **Technical Details**: CREDULE 데이터셋은 총 91,632개의 기사를 '신뢰할 수 있는(Credible)', '신뢰할 수 없는(Unreliable)', '팩트 체크됨(Fact-checked or Leaked)'의 세 가지 범주로 분류합니다. EVVER-Net은 이 데이터셋을 통해 훈련되었으며, 짧은 텍스트와 긴 텍스트 모두에서 뛰어난 식별 능력을 보여주었습니다. 또한, 도메인 신뢰도 점수(domain credibility scores)를 적용하여 분류 정확도를 높였습니다.

- **Performance Highlights**: EVVER-Net은 CREDULE 데이터셋에서 최대 91.5%와 94.4%의 인상적인 정확도를 달성했습니다. 이는 각각 짧은 텍스트와 긴 텍스트에 대한 결과입니다. 추가로, 널리 사용되는 팩트체킹 데이터셋들에서 유출된 증거와 신뢰할 수 없는 증거의 비율을 분석하여 EVVER-Net의 유용성을 입증했습니다.



### GuideWalk -- Heterogeneous Data Fusion for Enhanced Learning -- A  Multiclass Document Classification Cas (https://arxiv.org/abs/2404.18942)
- **What's New**: 이 연구에서는 의미 있는 문장의 그래프 구조에 기반한 새로운 텍스트 임베딩(Text Embedding) 방법을 제안합니다. 이 방법은 자연어 처리(Natural Language Processing, NLP)의 텍스트 분류(Text Classification) 문제에서 기존 알고리즘보다 우수한 성능을 보이는 것으로 나타났습니다.

- **Technical Details**: 제안된 임베딩 방법은 유도 전이 확률 행렬(Guided Transition Probability Matrix, GTPM) 모델을 사용합니다. 이 모델은 텍스트의 의미론적(Semantic) 특성을 추출하기 위해 단어 그래프(Word Graph)를 구축하고, 무작위 걷기(Random Walk)를 실행하여 전이 확률을 계산합니다. 그래프는 단어(Node)를 노드로, 단어 간 의미론적 관계(Link)를 가중치 있는 연결로 표현합니다.

- **Performance Highlights**: 실제 데이터 세트와 8개의 잘 알려진 임베딩 알고리즘을 사용한 테스트에서, 제안된 GTPM 방법은 이진 및 다중 클래스 데이터셋에서 기존 알고리즘보다 뛰어난 분류 성능을 나타냈습니다. 이 결과는 GTPM 알고리즘의 의미론적 의미 파악 능력이 텍스트 분류 결과에 긍정적인 영향을 미친다는 것을 입증합니다.



### DOCCI: Descriptions of Connected and Contrasting Images (https://arxiv.org/abs/2404.19753)
- **What's New**: DOCCI (Descriptions of Connected and Contrasting Images) 데이터셋 소개로, 이미지에 대한 긴, 인간이 주석을 단 영어 설명이 포함된 15,000개의 이미지를 제공합니다. 이 설명들은 공간 관계, 계수, 텍스트 렌더링 및 세계 지식과 같은 주요 도전 과제를 포착하려고 노력한 한 연구자가 촬영, 큐레이션 및 기증한 이미지들로 구성되어 있습니다.

- **Technical Details**: 이 설명들은 평균 136 단어 길이로, 연관된 또는 비슷한 이미지들과 확실히 구분될 수 있도록 세심하게 작성되었습니다. 각 설명은 고도로 구성적이며 일반적으로 여러 도전 과제를 포함합니다. DOCCI는 이미지-투-텍스트 (I2T) 및 텍스트-투-이미지 (T2I) 연구에 매우 유용한 데이터셋으로 검증되었습니다.

- **Performance Highlights**: DOCCI를 활용하여 훈련된 PaLI 5B 모델은 LLaVA-1.5 7B와 InstructBLIP 7B와 같은 규모가 큰 성능이 뛰어난 모델들과 비교했을 때 동등하거나 더 우수한 결과를 보였습니다. 또한, 현재의 텍스트-투-이미지 모델들이 긴 설명과 미세한 세부 사항을 포착하는 데 어려움이 있다는 점을 강조하면서 텍스트-투-이미지 생성을 위한 유용한 테스트베드로서 DOCCI의 역할을 입증했습니다.



### PANGeA: Procedural Artificial Narrative using Generative AI for  Turn-Based Video Games (https://arxiv.org/abs/2404.19721)
- **What's New**: 이번 연구에서는 'Procedural Artificial Narrative using Generative AI (PANGeA)'이라는 새로운 방식을 소개합니다. 이 접근법은 대규모 언어 모델(LLM)을 사용하여 게임 디자이너의 고급 기준에 따라 턴 기반 롤 플레잉 비디오 게임(RPG)의 내러티브 콘텐츠를 생성합니다. PANGeA는 기존 LLM의 비디오 게임 디자인 용도와 차별화되어, 게임 레벨 데이터뿐만 아니라, 플레이어와 환경 간의 동적이고 자유형식의 상호작용을 촉진하며 절차적 게임 내러티브와 연계됩니다.

- **Technical Details**: PANGeA는 Big 5 Personality Model을 반영하는 성격이 편향된 비플레이어 캐릭터(NPC)를 생성합니다. 이 시스템은 자유 형식 텍스트 입력을 처리하고, 게임 내러티브 범위를 벗어난 LLM 반응을 초래할 수 있는 문제에 대응합니다. 신규 검증 시스템은 LLM의 지능을 사용하여 텍스트 입력을 평가하고 생성된 반응을 절차적 내러티브와 일치시킵니다. PANGeA는 커스텀 메모리 시스템을 지원하는 서버를 통해, 생성된 반응을 증강하고 절차적 내러티브와 일치시키는 데 필요한 컨텍스트를 제공합니다. 또한, 모든 게임 엔진이 PANGeA와 직접 통합할 수 있도록 REST 인터페이스를 제공하며, 로컬 또는 사설 LLM과 호환 가능한 LLM 인터페이스를 갖추고 있습니다.

- **Performance Highlights**: PANGeA의 동적 내러티브 생성 능력은 두 버전의 데모 게임, 즉 맞춤형 브라우저 기반 GPT와 Unity 데모를 통한 경험적 연구와 절개 테스트를 통해 입증되었습니다. 결과는 PANGeA가 다양하고 예측할 수 없는 자유 형식 텍스트 입력을 제공받을 때에도 게임 디자이너가 내러티브 일관성 있는 콘텐츠를 생성하는 데 도움을 줄 수 있는 잠재력을 보여줍니다.



### Harmonic LLMs are Trustworthy (https://arxiv.org/abs/2404.19708)
Comments: 15 pages, 4 figures, 14 tables

- **What's New**: 새롭게 도입된 방법은 LLM(Large Language Model)의 견고성(안정성과 설명 가능성)을 실시간으로 검증할 수 있습니다. 이 방법은 로컬 편차로 표현되는 $	ext{harmoniticity}$로부터의 $	ext{gamma}(	ext{γ})$를 기반으로 하며, 이는 완전히 모델-불특정(model-agnostic)이며 비감독(unsupervised) 방식입니다. 이는 LLM이 제공하는 각 응답의 견고성을 순수 수학적 기준에 따라 측정하는 최초의 방법입니다.

- **Technical Details**: $	ext{gamma}(	ext{γ})$ 값의 측정은 잘못되거나 오해의 소지가 있는 응답과 긍정적 상관관계를 가진다는 것을 인간 주석 실험을 통해 보여주었습니다. 이러한 실험은 $	ext{gamma}(	ext{gamma})$의 그라디언트를 따르는 확률적 경사 상승법(stochastic gradient ascent)이 적대적 프롬프트(adversarial prompts)를 효율적으로 드러내는 데 사용될 수 있음을 시연하였습니다.

- **Performance Highlights**: 팝율러한 LLM 모델들(GPT-4, ChatGPT, Claude-2.1, Mixtral-8x7B, Smaug-72B, Llama2-7B, MPT-7B)을 대상으로 $	ext{gamma}(	ext{gamma})$를 수천 번의 쿼리(queries)에 적용해보았습니다. 이를 통해 잘못되거나 환각적인(hallucinatory) 대답의 가능성을 자동으로, 그리고 양적(quantitatively)으로 평가할 수 있었습니다. 또한, 다양한 목적의 도메인(Web QA, TruthfulQA, Programming QA)에서 이 모델들의 신뢰도를 순위화할 수 있었습니다. 모든 모델과 도메인에서 인간의 평가는 $	ext{gamma} 	o 0$이 신뢰성을 지시한다고 확인해주었고, 저-$	ext{gamma}$ 모델 중 선두는 GPT-4, ChatGPT, 그리고 Smaug-72B였습니다.



### Naturally Supervised 3D Visual Grounding with Language-Regularized  Concept Learners (https://arxiv.org/abs/2404.19696)
Comments: CVPR 2024. The first two authors contributed equally

- **What's New**: 이 논문에서는 3D 시각 grounding을 자연 감독(naturally supervised) 설정에서 연구합니다. 이는 Language-Regularized Concept Learner (LARC)라는 새로운 방법을 도입하는 것을 포함하여, 성능 향상과 데이터 효율성 증대에 중점을 둡니다.

- **Technical Details**: LARC는 대규모 언어 모델(LLMs)로부터 추출한 언어 제약을 사용하여 neuro-symbolic 컨셉 학습자의 중간 표현에 규제를 적용합니다. 이러한 언어 제약은 단어 간의 의미 관계(예: 대칭성, 배타성, 동의어성)에서 파생됩니다. 또한, 이 방법은 학습된 컨셉으로부터 새로운 컨셉의 실행을 구성할 수 있습니다.

- **Performance Highlights**: LARC는 3D referring expression comprehension을 위한 두 데이터셋에서 기존 작업들의 성능을 크게 향상시켰으며, zero-shot 시나리오에서 실행 가능한 결과를 보여주고, 사전 작업에 비해 데이터 효율성 및 데이터셋 간 전이성을 높였습니다.



### More Compute Is What You Need (https://arxiv.org/abs/2404.19484)
- **What's New**: 이 논문에서는 트랜스포머(Transformer) 기반 모델의 성능이 모델 크기나 데이터셋 크기의 특정 할당보다는 사용된 컴퓨트(compute)의 총량에 주로 의존한다는 새로운 스케일링 법칙(Scaling Law)을 제안합니다. 이는 기존의 Compute-Optimal 또는 Chinchilla Optimal 이라고 불리는 계산 예산 할당 법칙에 새로운 시각을 제공합니다.

- **Technical Details**: 이 연구는 트랜스포머 기반 모델들의 성능이 컴퓨트 자원의 총 사용량에 의존한다고 가정하고, 이를 통해 새로운 통합 스케일링 법칙을 제시합니다. 크기와 훈련 데이터셋의 균형에 관계없이 전체적인 컴퓨트 자원의 최적화가 중요하다고 분석합니다.

- **Performance Highlights**: 새로운 스케일링 법칙을 바탕으로, 추론 효율성(inference efficiency)을 위해서는 훈련이 작은 모델 크기와 큰 훈련 데이터셋을 우선시해야 함을 예측합니다. 또한, 사용 가능한 웹(Web) 데이터셋이 고갈된다고 가정할 때, 모델 크기를 증가시키는 것이 모델 성능을 향상시키는 유일한 방법이 될 수 있을 것입니다.



### Large Language Model Informed Patent Image Retrieva (https://arxiv.org/abs/2404.19360)
Comments: 8 pages. Under review

- **What's New**: 새로운 특허 이미지 검색 시스템이 제안되었습니다. 이 시스템은 Large Language Models (LLM)을 통해 특허 이미지의 의미론적 이해를 향상시키고, 분포 인식형 (distribution-aware) 대조적 손실을 통해 소수 클래스의 성능을 향상시킵니다. 이 방법은 특허 전문가들이 실제로 이미지 검색 작업을 수행하는 데 도움이 될 것임을 강조합니다.

- **Technical Details**: 이 연구는 언어 정보, 분포 인식 다중모드 접근 방식을 도입하여 특허 이미지의 특징 학습을 강화합니다. LLM을 사용하여 이미지 설명을 생성하고, 비대칭 데이터에 대응하기 위해 여러 코스-그레인 손실과 불확실성 요소를 적용합니다. 또한, DeepPatent2 데이터셋에서 실시한 광범위한 실험을 통해 이 방법이 상태 기술 (state-of-the-art) 성능을 달성하거나 비슷한 성능을 보여줍니다.

- **Performance Highlights**: 제안된 시스템은 특허 이미지 기반 검색에서 mAP +53.3%, Recall@10 +41.8%, MRR@10 +51.9%의 우수한 성능을 보여주었습니다. 이미지 검색의 실용성과 효과를 강조하기 위해 실제 사용자 분석을 통해 모델을 평가했으며, 이는 기존 방법들에 비해 사용자 만족도와 사용 가능성이 높음을 보여줍니다.



### Enhancing Trust in LLM-Generated Code Summaries with Calibrated  Confidence Scores (https://arxiv.org/abs/2404.19318)
- **What's New**: 이 연구는 코드 요약을 자동 생성하는 대형 언어 모델(Large Language Models, LLMs)의 성과를 평가하고, 생성된 요약이 인간이 만든 요약과 얼마나 유사한지를 판단할 수 있는 새로운 방법을 제시합니다. 특히, LLM이 생성한 코드 요약의 인간 유사성을 더 정확히 측정할 수 있는 새로운 신뢰도 척도를 도입하고 평가하는 것이 핵심입니다.

- **Technical Details**: 연구팀은 여러 언어에 대해 여러 설정에서 LLM을 사용하여 코드 요약을 생성하고, 이 요약들이 인간이 생산할 요약과 얼마나 유사한지를 평가합니다. 이를 위해 BERTScore와 BLEU 등의 성능 측정 지표를 사용하며, LLM으로부터 나온 요약의 신뢰도를 측정하는 새로운 방식을 개발하였습니다. 이 방법은 LLM의 출력이 인간이 생성한 요약과 얼마나 '유사할 가능성'이 있는지에 대한 예측값을 제안함으로써 요약의 품질을 보다 정확히 평가할 수 있도록 합니다.

- **Performance Highlights**: 이 연구는 기존의 LLM 코드 요약 평가보다 더 정제된 접근 방식을 제안하여, LLM이 생성한 요약이 인간 요약과 유사할 가능성에 대한 더 정확한 예측을 제공합니다. 실험 결과, 이 새로운 신뢰도 척도 방법은 여러 언어와 설정에 걸쳐 일관된 결과를 보여주며, 코드 유지보수 및 개발 과정에서의 LLM 응용의 신뢰성을 높일 수 있는 가능성을 시사합니다.



### Revisiting N-Gram Models: Their Impact in Modern Neural Networks for  Handwritten Text Recognition (https://arxiv.org/abs/2404.19317)
- **What's New**: 자동 텍스트 인식(ATR) 분야에서 딥 뉴럴 네트워크(deep neural networks)의 최근 발전을 고려할 때, 전통적인 언어 모델의 필요성이 감소하고 있는지 여부에 대한 연구입니다. 이번 연구에서는 특히 n-gram 모델과 같은 명시적인 언어 모델이 최신 딥러닝(deep learning) 아키텍처의 성능에 어떻게 기여하는지 직접 다루고 있습니다.

- **Technical Details**: PyLaia와 DAN 같은 두 가지 주요 신경망 아키텍처를 IAM, RIMES, NorHand v2와 같은 세 가지 데이터셋에서 테스트했습니다. 연구는 선(line) 및 페이지(page) 수준에서 n-gram 모델의 최적 파라미터를 평가했으며, 이에는 n-gram의 차수(order), 가중치(weight), 평활화 방법(smoothing methods), 및 토큰화 수준(tokenization level)이 포함됩니다.

- **Performance Highlights**: 문자(character) 또는 부분단어(subword) n-gram 모델을 통합하면 모든 데이터셋에서 ATR 모델의 성능이 크게 향상됩니다. 특히, 문자 언어 모델과 결합된 DAN은 현재 벤치마크를 능가하여, 현대 문서 분석 시스템에서 하이브리드 접근 방식의 가치를 확인시켜 주었습니다.



### Multi-hop Question Answering over Knowledge Graphs using Large Language  Models (https://arxiv.org/abs/2404.19234)
- **What's New**: 이 연구는 지식 그래프(KGs)를 사용하여 복잡한 자연어 질문에 대답하는 대형 언어 모델(LLMs)의 능력을 평가합니다. 구체적으로, 다양한 크기와 성격의 지식 그래프에서 질문을 해결하기 위해 필요한 정보를 효과적으로 추출하고 LLM에 공급하는 방법을 탐구하였습니다.

- **Technical Details**: 이 연구는 (a) 의미 분석(semantic parsing, SP)을 이용하거나, (b) 정보 검색(information retrieval, IR) 기반 방법을 사용하여 질문 응답 시스템을 구현합니다. 여기서 두 방법이 모두 대규모 언어 모델(LLMs)에 적용될 수 있음을 보여주며, 이는 지식 그래프에서 여러 에지를 거치는 복수의 점프(multi-hop) 질문에 효과적으로 대응할 수 있습니다.

- **Performance Highlights**: 실험은 여섯 개의 지식 그래프를 대상으로 수행되었으며, 예제 특화 서브-그래프의 유무에 따라 평가되었습니다. 결과적으로, 정보 검색 및 의미 분석 기반 방법이 LLM에서 구현되었을 때 매우 경쟁력 있는 성능을 제공함을 확인하였습니다.



### Transcrib3D: 3D Referring Expression Resolution through Large Language  Models (https://arxiv.org/abs/2404.19221)
Comments: CORLW 2023

- **What's New**: Transcrib3D는 로봇이 자연 언어를 사용하여 3D 환경의 객체를 해석하도록 하는 새로운 접근 방식을 소개합니다. 이 방법은 3D 탐지 (3D detection) 방법과 대규모 언어 모델 (LLMs)의 추론 능력을 결합하여, 복잡한 3D 참조 표현을 이해할 수 있습니다.

- **Technical Details**: Transcrib3D는 먼저 3D 객체 탐지기를 사용하여 각 객체의 범주, 위치, 공간 범위, 색상 등 공간적 및 의미적 정보를 텍스트로 변환합니다. 이 텍스트 기반의 3D 장면 트랜스크립트는 LLM에 의해 처리되며, 필터링을 통해 관련 없는 객체 정보는 제외되고, 주어진 참조 표현에 대한 프롬프트와 함께 LLM 기반 추론 메커니즘으로 입력됩니다. 이러한 추론 과정은 반복적인 코드 생성 및 추론, 원칙-지도된 제로-샷 프롬프팅 (zero-shot prompting), 자가 수정을 통한 미세 조정 등을 포함합니다.

- **Performance Highlights**: Transcrib3D는 3D 참조 해상도 벤치마크인 ReferIt3D 및 ScanRefer에서 최고의 성능을 달성하며, 이전 다중 모달성 기준보다 크게 성능이 향상되었습니다. 또한, 실제 로봇 실험을 통해 로봇 조작기가 복잡한 3D 공간-의미 추론을 필요로 하는 자연 언어 명령을 수행할 수 있음을 보여줍니다.



### Q-GroundCAM: Quantifying Grounding in Vision Language Models via GradCAM (https://arxiv.org/abs/2404.19128)
Comments: Accepted to CVPR 2024, Second Workshop on Foundation Models (WFM)

- **What's New**: 이 논문에서는 Vision and Language Models (VLM, 비전 및 언어 모델)의 구문 분석 능력을 평가하기 위한 새로운 정량적 메트릭스를 제안합니다. 특히, CLIP, BLIP, ALBEF 같은 사전 훈련된 VLMs의 grounding (구문) 능력을 평가하기 위해 GradCAM 활성화를 활용합니다. 이 메트릭스들은 zero-shot (제로샷) 능력의 상세 비교와 모델들의 grounding uncertainty (불확실성) 측정을 가능하게 합니다.

- **Technical Details**: 제안된 메트릭스는 GradCAM 활성화 지도와 ground-truth 이진 마스크 간의 유사도를 계산하고, bounding box 내부의 활성화는 보상하는 반면, 외부의 잘못된 활성화는 penalize(처벌) 합니다. 이 연구는 또한 Pointing Game (PG)과의 비교를 통해 개선된 성능 측정 방법을 제시하며, VLM의 더 정밀한 구문 능력 평가를 제공합니다. 다양한 VLMs, 특히 patch-based ViT 및 CNN-based vision transformer encoders와의 적용 가능성을 탐구합니다.

- **Performance Highlights**: 이 논문은 BLIP, CLIP 및 ALBEF 모델을 포함한 여러 VLMs의 grounding 능력을 종합적으로 평가하여, 다양한 grounding 작업에서의 성능을 비교 분석합니다. 이를 통해, 기존의 Pointing Game 메트릭스가 포착하지 못하는 grounding 성능의 미묘한 차이를 구별할 수 있습니다. 또한, 새로운 메트릭스는 spurious activations (부정확한 활성화)와 같은 문제들을 보다 효과적으로 다룰 수 있는 가능성을 보여 줍니다.



### Blind Spots and Biases: Exploring the Role of Annotator Cognitive Biases  in NLP (https://arxiv.org/abs/2404.19071)
- **What's New**: 이 연구는 인공 지능(AI) 시스템에서 발생할 수 있는 편견과 이를 완화하기 위해 인간 판단을 통합하는 방법의 잠재력과 함정을 탐구합니다. 최근에는 휴머노이드 상호 작용(Human-Computer Interaction, HCI) 관점과 연계하여 crowdwork 연구 설계의 표준화된 원칙 필요성에 초점을 맞추고 있습니다.

- **Technical Details**: 연구자들은 훈련 데이터 세트 라벨링에 인간 주석자(human annotators)의 사용이 기존 편견을 악화시키거나 새로운 편견을 만들 수 있다는 점을 지적합니다. 이는 인간의 인지 편향(cognitive biases) 때문으로, crowd-sourced 데이터의 문제점을 해결하기 위한 다양한 방법(예: 민감한 속성 제거, 데이터 리샘플링, 반복적인 훈련 가중치 조정)이 소개되었습니다.

- **Performance Highlights**: 발견된 바에 따르면, 적절한 데이터 주석(annotation)과 이를 검증하는 과정은 편견을 상당히 감소시킬 수 있으나, 인간의 개입이 반드시 완벽한 해결책은 아니라는 점이 강조됩니다. 인간 주석자들 사이에서의 서로 다른 해석 가능성이 존재하며, 이는 오류나 비합리적 판단으로 이어질 수 있습니다.



### HELPER-X: A Unified Instructable Embodied Agent to Tackle Four  Interactive Vision-Language Domains with Memory-Augmented Language Models (https://arxiv.org/abs/2404.19065)
Comments: Videos and code this https URL

- **What's New**: 최근 연구에서 지시 가능한(agent) 에이전트는 과제 계획(task planners)으로 메모리를 강화한 대규모 언어 모델(Large Language Models, LLMs)을 활용하고 있습니다. 본 기술 보고서에서는 HELPER의 기능을 확장하여 예제와 프롬프트의 범위를 넓히고, 질문을 요청할 수 있는 추가 API를 통합함으로써 HELPER를 공유 메모리(shared memory)로 확장합니다. 이러한 확장은 에이전트가 대화에서 계획을 실행하고, 자연어 지시(natural language instruction)를 따르며, 적극적인 질문(asking question)을 하고, 상식적인 방 재조정(commonsense room reorganization) 분야에서 작업할 수 있게 합니다.

- **Technical Details**: HELPER-X는 다양한 대화형 시각적 언어 에이전트 벤치마크(four diverse interactive visual-language embodied agent benchmarks)인 ALFRED, TEACh, DialFRED, 및 Tidy Task에서 평가되었습니다. 이 확장된 HELPER는 초기 예제(few-shot)를 사용하며 단일 에이전트를 사용하여 이 벤치마크들에서 최고 기록(state-of-the-art) 성능을 달성하였으며, 특정 도메인 내(in-domain) 훈련을 받지 않고도 훈련받은 에이전트들과 경쟁력을 유지합니다.

- **Performance Highlights**: HELPER-X는 훈련된 에이전트들과 비교하여 경쟁력 있는 성능을 보여주었으며, 특히 도메인 특화 훈련 없이도 상태를 예측하는 능력에서 주목할 만한 결과를 보여줍니다. 이는 향상된 메모리 시스템과 통합된 API가 효과적으로 작동함을 시사합니다.



### Foundations of Multisensory Artificial Intelligenc (https://arxiv.org/abs/2404.18976)
Comments: CMU Machine Learning Department PhD Thesis

- **What's New**: 이 연구는 다양한 센서 입력을 학습하는 멀티센서 AI 시스템의 기계 학습(machine learning) 기반을 발전시키는 것을 목표로 합니다. 건강과 웰빙(human health and well-being)을 지원하고, 멀티미디어 콘텐츠 처리(multimedia content processing)를 가능하게 하며, 자율 에이전트(autonomous agents)의 실제 세계 적용을 향상시키는 것에 실질적인 이점이 있습니다.

- **Technical Details**: 이 논문은 먼저 다양한 모달리티(modalities)가 상호 작용하여 작업에 새로운 정보를 제공하는 방식을 정식화하는 이론적 프레임워크(theoretical framework)를 제시합니다. 다양한 모달리티와 작업에 걸쳐 일반화하는 실용적인 멀티모달 기초 모델(multimodal foundation models)의 설계를 연구합니다. 크로스 모달 어텐션(cross-modal attention)과 멀티모달 트랜스포머(multimodal transformer) 아키텍처는 많은 오늘날의 멀티모달 기초 모델을 뒷받침합니다.

- **Performance Highlights**: MultiBench를 통해 스케일링된 이 아키텍처들은 일반 목적의 멀티센서 AI 시스템을 생성하는 데 사용되며, 이는 감정 컴퓨팅(affective computing), 정신 건강(mental health), 암 예후(cancer prognosis), 로봇공학(robotics) 등의 실제 세계 문제에 적용될 수 있습니다. 이러한 모델은 더 일반적이고, 상호작용적이며, 안전한 멀티센서 AI로의 발전을 위한 기초를 마련합니다.



### RE-GrievanceAssist: Enhancing Customer Experience through ML-Powered  Complaint Managemen (https://arxiv.org/abs/2404.18963)
- **What's New**: 최근 수년간 디지털 플랫폼 (digital platform) 회사들은 소비자 사용 증가에 따라 고객 불만 관리에 점점 더 큰 도전을 맞고 있다. 이 연구 논문에서는 부동산 고객 불만 관리를 위해 특별히 설계된 엔드-투-엔드 파이프라인인 'RE-GrievanceAssist'를 소개한다.

- **Technical Details**: RE-GrievanceAssist 파이프라인은 세 가지 주요 구성 요소로 구성된다. 첫 번째로는 TF-IDF 벡터화 및 XGBoost 분류기를 사용하는 응답/미응답 ML 모델이 있다. 두 번째로는 fasttext 분류기를 사용한 사용자 유형 분류기가 있으며, 세 번째로는 TF-IDF 벡터화와 XGBoost 분류기를 사용한 문제/부문제 분류기가 있다. 이 파이프라인은 Databricks에서 배치 작업으로 배포되었다.

- **Performance Highlights**: 이 파이프라인의 도입으로 인해 전반적인 수동 작업이 40% 감소하였으며, 2023년 8월 이후 매월 Rs 1,50,000의 비용 절감이 이루어졌다.




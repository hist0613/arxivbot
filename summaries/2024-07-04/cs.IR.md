New uploads on arXiv(cs.CL)

### Planetarium: A Rigorous Benchmark for Translating Text to Structured Planning Languages (https://arxiv.org/abs/2407.03321)
- **What's New**: 최근 많은 연구들이 자연어 모델을 활용하여 계획 문제를 해결하는 방법을 탐구하고 있습니다. 이에 관한 한 가지 접근법은 계획 작업의 자연어 설명을 구조화된 계획 언어인 PDDL(Planning Domain Definition Language)로 번역하는 것입니다. 이러한 방법의 잠재력이 높지만, 생성된 PDDL 코드의 품질을 정확히 측정하는 것은 여전히 어려운 문제로 남아 있습니다. 이를 해결하기 위해 새로운 벤치마크인 'Planetarium'를 소개하며, 자연어 설명을 PDDL 코드로 변환하는 언어 모델의 능력을 평가합니다.

- **Technical Details**: 이 연구는 PDDL 코드의 정확성을 엄격하게 평가하기 위한 PDDL 동등성 알고리즘을 개발했습니다. 이 알고리즘은 장면 그래프로 PDDL 코드를 변환하고 목표 상태의 확장을 계산한 후, 그래프 간의 동형성을 점검하여 두 PDDL 문제가 동일한 근본적인 계획 문제를 나타내는지 확인합니다. 또한, 132,037 개의 텍스트-대-PDDL 쌍을 포함한 데이터셋을 제공하여 다양한 난이도의 계획 문제를 포함합니다.

- **Performance Highlights**: GPT-4의 경우, 생성된 PDDL 문제 설명의 87.6%가 구문적으로 해석 가능하고, 82.2%가 유효하며, 해결 가능한 문제였지만, 단지 35.1%만이 의미적으로 올바른 것으로 나타났습니다. 이러한 결과는 이 작업의 복잡성을 강조하며, 보다 엄격한 벤치마크가 필요하다는 것을 보여줍니다.



### A Review of the Applications of Deep Learning-Based Emergent Communication (https://arxiv.org/abs/2407.03302)
Comments:
          49 pages, 15 figures

- **What's New**: 이 논문은 다중 에이전트 강화 학습 환경에서 인간 언어와 유사한 통신 시스템이 새롭게 등장하는 방식을 연구하는 'Emergent communication'(출현하는 통신, emergent language)의 응용에 대해 종합적으로 검토합니다. 기존 연구를 기반으로 새로운 연구 방향을 제시하고 다양한 분야에서 출현하는 통신의 역할을 분석합니다.

- **Technical Details**: 출현하는 통신은 심층 다중 에이전트 강화 학습(deep multi-agent reinforcement learning) 환경에서 완전히 새로운 형태로 언어와 같은 복잡한 행동을 재현하는 가능성을 탐구합니다. 이 논문은 머신러닝, 자연 언어 처리(Natural Language Processing, NLP), 언어학(Linguistics), 인지 과학(Cognitive Science) 등 다양한 분야에서 출현하는 통신의 응용을 설명합니다. 각 응용 분야는 그 범위, 출현하는 통신이 이 문제를 해결하는 데 있어 탁월한 역할을 하는 방법, 기존 문헌 요약 및 근시일 내 연구 방향에 대한 권장 사항으로 자세히 설명됩니다.

- **Performance Highlights**: 이 논문은 출현하는 통신 연구가 여러 과학 및 기술 분야에 어떻게 적용될 수 있는지에 대한 명확한 개념을 보완함으로써, 출현하는 통신의 종합적인 응용을 다룹니다. 이는 특히 NLP, 머신러닝 및 인지 과학 분야에서 새로운 통찰력을 제공합니다.



### LLM Internal States Reveal Hallucination Risk Faced With a Query (https://arxiv.org/abs/2407.03282)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 환각 문제(hallucination problem)를 다루고 있습니다. 환각 문제는 LLMs의 신뢰성과 신뢰도를 크게 제한하는 중요한 문제로, 연구자들은 LLMs가 응답 생성 전에 자신의 환각 위험을 추정할 수 있는지 조사하였습니다. 이 연구는 15개의 다양한 자연어 생성(NLG) 작업에서 700개 이상의 데이터셋을 통해 LLMs의 내부 메커니즘을 분석한 결과, LLMs의 내부 상태가 쿼리를 학습 데이터에서 본 적이 있는지 여부와 환각할 가능성을 나타낼 수 있다는 두 가지 중요한 통찰을 발견했습니다.

- **Technical Details**: 연구진은 특정 뉴런(neurons), 활성화 레이어(activation layers), 그리고 토큰(tokens)을 탐구하여 LLM의 불확실성 및 환각 위험 인식에 중요한 역할을 하는 요소들을 분석했습니다. 이 과정에서 프로빙 추정기(probing estimator)를 사용하여 쿼리와 관련된 내부 상태를 분석하였고, 이를 통해 LLMs의 자기 인식과 불확실성 표시 능력을 검증했습니다. 구체적으로는 쿼리를 학습 데이터에서 본 적이 있는지 여부를 80.28%의 정확도로, 환각할 가능성 여부를 84.32%의 정확도로 추정하는 능력을 보여주었습니다.

- **Performance Highlights**: 이 연구는 LLMs의 환각 위험을 사전에 추정함으로써, 리트리벌 증강(retrieval augmentation)이나 경고 시스템을 제공하는 잠재력을 보여줍니다. 이로 인해 LLMs가 응답을 생성하기 전에 환각 위험이 높은지 여부를 인식할 수 있게 되어, 사용자에게 더 신뢰할 수 있는 정보를 제공할 수 있을 것으로 기대됩니다.



### Evaluating Automatic Metrics with Incremental Machine Translation Systems (https://arxiv.org/abs/2407.03277)
- **What's New**: 이 연구에서는 상용 기계 번역 시스템의 성능 변화를 추적한 데이터셋을 소개합니다. 6년 동안 12가지 번역 방향에서 주간으로 수집된 이 데이터셋을 통해 기계 번역(MT) 메트릭의 평가가 가능해졌습니다. 이를 통해 최근 MT 출력물에 대한 선호도를 기반으로 메트릭의 성능을 검증하고자 합니다. 이번 연구는 기존 MT 메트릭 연구의 여러 결과를 확인하며, 데이터셋의 유용성을 입증합니다.

- **Technical Details**: 데이터셋은 Google Translate를 사용해 2018년부터 2024년까지 매주 수집되었습니다. 총 12개의 언어 쌍을 분석했으며, BLEU, chrF, BERTScore와 같은 다양한 메트릭을 이용했습니다. 특히, COMET-22, UniTE 등 학습된 메트릭(trained metrics)과 비학습된 메트릭(non-neural metrics)의 성능 차이를 분석했습니다. 또한, Spearman 상관 분석을 통해 시간이 지남에 따라 메트릭이 어떻게 변했는지 시각화했습니다.

- **Performance Highlights**: 실험 결과, 학습된 메트릭이 비학습된 메트릭보다 일관되게 높은 정확성을 보여주었습니다. COMET-22, UniTE, COMET-20, COMET-Kiwi 등의 메트릭은 모든 언어 쌍에서 높은 상관성을 나타냈습니다. BLEU와 chrF는 일부 언어 쌍에서 낮은 상관성을 기록했습니다. 또한, 시스템 차이가 큰 경우 메트릭 신뢰성이 과대평가될 가능성을 고려해 1년 이하의 간격을 가진 시스템만 비교했습니다. 그 결과 COMET-22가 가장 높은 정확도를 나타냈습니다.



### STF: Sentence Transformer Fine-Tuning For Topic Categorization With Limited Data (https://arxiv.org/abs/2407.03253)
- **What's New**: 트윗 주제 분류(task of topic classification from tweets)에 대한 새로운 접근법을 제안합니다. 제한된 라벨링 데이터로 인해 성능이 낮아지는 기존 시스템의 문제를 해결하기 위해, 우리는 Sentence Transformers Fine-tuning (STF) 접근법을 소개합니다. 이는 미리 학습된 Sentence Transformers 모델을 활용해 트윗 주제를 정확하게 분류합니다. 또한, 파라미터 민감도 분석을 통해 최적의 성능을 도출했습니다.

- **Technical Details**: STF 접근법은 미리 학습된 Sentence Transformers 모델을 사용하고, 특정 주제 분류 작업에 맞게 파인튜닝(fine-tuning) 합니다. 이는 소량의 라벨링 데이터로도 높은 정확도를 유지할 수 있습니다. 실험은 두 개의 벤치마크 데이터셋에서 수행되었으며, STF가 최신 상태의 아트 방법론보다 뛰어난 성능을 보였음을 확인했습니다. 본 연구에서는 LSTM, CNN, Bi-LSTM와 같은 세 가지 딥러닝 아키텍처와 MNB, LR 등의 전통적인 기계 학습 모델을 평가 및 비교했습니다.

- **Performance Highlights**: 실험 결과, STF 모델이 트윗 주제 분류에서 최상의 결과를 도출했으며, 소량의 라벨링 데이터로도 높은 정확도를 유지할 수 있음을 확인했습니다. 이는 많은 양의 라벨링 트윗을 요구하는 기존의 최신 방법론의 한계를 극복한 것입니다.



### CATT: Character-based Arabic Tashkeel Transformer (https://arxiv.org/abs/2407.03236)
- **What's New**: 본 논문에서는 아랍어 텍스트의 이해를 높이고 오해를 줄이는 데 중요한 역할을 하는 아랍어 텍스트에 대한 'Tashkeel' 또는 'Diacritization'을 개선하는 새로운 접근법을 제안합니다. 두 가지 사전 학습된 캐릭터 기반 BERT(Bidirectional Encoder Representations from Transformers) 모델을 파인튜닝하였고, 최고의 모델 성능을 향상시키기 위해 'Noisy-Student' 접근법을 적용했습니다. WikiNews와 CATT라는 두 개의 수동 레이블 벤치마크 데이터셋을 사용하여 상용 및 오픈소스 모델들과 비교 평가한 결과, 저희 모델이 상대적인 Diacritic Error Rate(DER)에서 우위를 점하며 최신 성능을 달성했습니다. 또한, 저희 모델은 CATT 데이터셋에서 GPT-4-turbo를 능가하는 결과를 보여주었습니다.

- **Technical Details**: 모델 훈련 전략으로는 사전 학습된 캐릭터 기반 BERT와 'Noisy-Student' 방법을 사용했습니다. 데이터셋 준비 단계에서는 특정 기준에 따라 필터링 과정을 거친 Tashkeela 데이터셋을 사용하였으며, 추가로 다양한 소스에서 긁어모은 18,543,025개의 데이터 샘플로 캐릭터 기반 BERT를 사전 학습시켰습니다. 백엔드 구조에서는 Transformer 기반으로 encoder-only와 encoder-decoder 변종을 둘 다 사용했으며, 전체 학습 과정에서 최대 시퀀스 길이를 1024자로 제한하였습니다.

- **Performance Highlights**: 저희의 최고 모델은 WikiNews 데이터셋에서는 30.83%, CATT 데이터셋에서는 35.21%의 상대적인 DER 개선을 보여 모두에서 최신 성능을 기록했습니다. 특히, CATT 데이터셋에서는 GPT-4-turbo를 9.36% 상대적인 DER로 능가하여, 아랍어 텍스트 디아크리타이제이션에서 현저한 성능 향상을 달성하였습니다.



### Improving Retrieval-augmented Text-to-SQL with AST-based Ranking and Schema Pruning (https://arxiv.org/abs/2407.03227)
- **What's New**: 이 연구는 대규모 언어 모델(Large Language Models, LLMs)을 이용한 Text-to-SQL 의미론적 파싱을 다루며, 특히 상업용 데이터베이스 스키마의 크기와 비즈니스 인텔리전스 솔루션의 배포 가능성에 대한 도전에 관해 집중합니다. 새로운 접근법으로 동적으로 입력 데이터베이스 정보를 검색하고 추상 구문 트리(Abstract Syntax Tree, AST)를 사용해 몇 가지 예제를 선택하여 문맥 학습(in-context learning)을 수행하는 방안을 제안합니다. 또한, 일부 병렬 의미 파서를 활용해 예상 SQL 쿼리의 근사 분산을 생성하여 검색을 지원하는 방안을 조사합니다.

- **Technical Details**: 이 접근법을 위해 500M 미만의 파라미터로 구성된 모델을 효율적인 근사기로 적응시켰습니다. 이 모델은 병렬화된 방식으로 스키마를 처리할 수 있도록 강화했습니다. 제안된 프레임워크는 손쉽게 전체 데이터베이스 스키마를 줄여 LLM의 성능을 향상시키고, 긴 스키마를 다루기 위한 효율성을 제공합니다. 또한, 이는 상업적 용도의 긴 스키마를 다루는 비즈니스 인텔리전스 플랫폼에도 적합합니다.

- **Performance Highlights**: 이 접근법은 단일 훈련 세션만으로도 전통적 베이스라인을 능가하며, 더욱 복잡하고 여러 번의 이터레이션을 요구하는 기존 방법보다 뛰어난 성능을 보였습니다. 실험을 통해 모노링구얼과 다국어 기준 데이터셋에서 모두 개선된 결과를 얻었으며, LLM의 계산 작업량을 크게 줄이는 동시에 실행 정확도를 높였습니다.



### How Does Quantization Affect Multilingual LLMs? (https://arxiv.org/abs/2407.03211)
- **What's New**: 이 논문은 양자화(quantization) 기술이 다국어 대형 언어 모델(multilingual large language models, LLMs)에서 미치는 영향을 처음으로 세부적으로 분석한 연구입니다. 기존에는 양자화가 영어 작업에 미치는 영향만 다루어졌으나, 본 논문에서는 여러 언어에 걸친 양자화의 영향을 연구했습니다.

- **Technical Details**: 포스트 트레이닝 양자화(Post Training Quantization, PTQ) 기법을 사용하여 네 가지 최첨단 다국어 LLMs를 8억에서 1030억 개의 파라미터를 가지는 3가지 다른 크기의 모델에서 실험하였습니다. 실험은 자동 벤치마크(multilingual MMLU, MGSM, FLORES-200), LLM-as-a-Judge 방법, 인간 평가를 통해 수행되었습니다.

- **Performance Highlights**: ['자동 평가 지표는 양자화로 인한 성능 저하를 과소 평가합니다. 자동 평가는 FP16 기준 프랑스어 −0.3%, 일본어 −1.7%의 성능 저하를 보였으나, 인간 평가는 각각 −16.6%, −16.0%의 성능 저하를 보였습니다.', '언어별로 양자화의 영향이 다릅니다. 라틴 문자가 아닌 언어가 더 큰 영향을 받으며, 1030억 파라미터 모델에서 라틴 문자 언어는 −0.7%, 비 라틴 문자 언어는 −1.9%의 성능 저하를 보였습니다.', '수학적 추론 등 어려운 작업에서 성능 저하가 빠르게 일어납니다. 예를 들어, 수학적 추론은 −13.1%, 인간 평가 최근의 현실적인 프롬프트는 −10.5%, LLM-as-a-Judge는 −25.9%의 성능 저하를 보였습니다.']



### Fine-Tuning with Divergent Chains of Thought Boosts Reasoning Through Self-Correction in Language Models (https://arxiv.org/abs/2407.03181)
- **What's New**: 이번 연구에서는 주어진 한 번의 추론 단계를 통해 여러 개의 추론 과정을 비교하여 더 나은 성능을 얻는 새로운 방법인 Divergent Chain of Thought (DCoT)을 제안합니다. 이는 기존의 Chain of Thought (CoT) 방법론을 개선한 것으로, 모델이 여러 개의 추론 과정을 동시에 생성하고 비교하여 최종 답을 도출하는 방식입니다.

- **Technical Details**: DCoT는 심리학 이론인 Divergent Thinking과 Convergent Thinking에서 영감을 받아 설계되었습니다. Divergent Thinking은 다양한 아이디어를 생성하고 탐구하는 단계를, Convergent Thinking은 이를 종합하여 하나의 해결책을 도출하는 단계를 의미합니다. 본 연구에서는 모델이 여러 개의 추론 과정을 생성하고 이를 비교하여 하나의 답을 도출할 수 있도록 Instruction Tuning을 수행했습니다.

- **Performance Highlights**: 다양한 이유형 과제를 포함한 실험 결과, DCoT 데이터로 Instruction Tuning을 수행하면 CoT 기반 성능보다 일관되게 향상됨을 보였습니다. 특히, 적은 수의 파라미터를 가진 모델(1.3B에서 70B까지)에서도 효과적인 성능 향상을 확인했습니다. 또한, DCoT는 외부 피드백 없이도 모델이 최초 답변을 개선할 수 있는 자기수정(self-correction) 능력을 강화하는 효과가 있음을 실험을 통해 증명했습니다.



### Investigating Decoder-only Large Language Models for Speech-to-text Translation (https://arxiv.org/abs/2407.03169)
Comments:
          Accepted to Interspeech 2024

- **What's New**: 이 논문에서는 디코더 전용 LLMs를 음성-텍스트 번역(S2TT) 작업에 통합하는 방법을 탐구합니다. 전통적인 S2TT 시스템의 한계를 극복하기 위해 새로운 디코더 전용 아키텍처를 제안하였고, 이를 통해 CoVoST 2와 FLEURS 데이터셋에서 기존의 비독점 데이터를 사용한 모델 대비 최첨단 성능을 달성했습니다.

- **Technical Details**: 제안된 모델은 연속적인 음성 표현을 직접 소비하고 텍스트 번역을 생성하도록 설계된 디코더 전용 아키텍처를 기반으로 합니다. 모델의 설계, 매개변수 효율적인 미세 조정 기법, 그리고 작업 공식화 측면에서 다양한 요소들을 조사하였으며, 예측된 아키텍처를 통해 음성 인코딩과 텍스트 디코딩을 단일 과정으로 통합하였습니다. 또한, W2v-BERT 등의 셀프-슈퍼바이즈드 학습을 통한 음성 인코더와 Transformer 아키텍처를 사용하여 뛰어난 성능을 보였습니다.

- **Performance Highlights**: 논문에서 제안된 모델은 CoVoST 2와 FLEURS 데이터셋에서 음성-텍스트 번역 성능이 가장 우수한 모델과 비교하여 최첨단 성능을 보였습니다. 특히, 대량의 독점 데이터를 사용하지 않고도 이러한 성과를 달성했다는 점에서 큰 의미가 있습니다.



### Let the Code LLM Edit Itself When You Edit the Cod (https://arxiv.org/abs/2407.03157)
Comments:
          Preprint. Work in Progress

- **What's New**: 이번 연구에서는 개발자가 기존 코드를 실시간으로 편집하고 코드 어시스턴트에게 다음 토큰 또는 다음 줄을 실시간으로 예측하도록 요청하는 시나리오를 조사합니다. 이 과정에서는 대규모 언어 모델(Large Language Model, LLM)이 전체 KV 캐시(KV cache)를 다시 인코딩해야 하는데, 이는 계산적으로 매우 비싸고 시간이 많이 소요됩니다. 이를 해결하기 위해 'Positional Integrity Encoding (PIE)'를 도입하였습니다.

- **Technical Details**: PIE는 Rotary Positional Encoding(RoPE)를 기반으로, Key 캐시에 있는 회전 행렬을 제거하고 올바른 회전 행렬을 재적용하여 위치 혼란을 제거합니다. 이 과정은 매트릭스 곱셈 한 번만으로 이루어지며, 토큰 간의 위치 관계를 정확히 유지합니다. KV 캐시(KV Cache)를 수정하는 데 필요한 계산 오버헤드를 최소화하여 실제 응용 프로그램에서 효율성을 극대화합니다.

- **Performance Highlights**: RepoBench-C-8k 데이터셋을 사용하여 1.3B, 6.7B, 33B 매개변수를 가진 DeepSeek-Coder 모델로 광범위한 실험을 수행한 결과, PIE는 전체 재계산 접근 방식에 비해 계산 오버헤드를 85% 이상 줄이면서도 성능을 유지하였습니다. 코드 삽입, 코드 삭제, 다중 장소 코드 편집 등 세 가지 실제 코딩 작업에서 PIE의 효과를 검증하였습니다.



### Enhancing Translation Accuracy of Large Language Models through Continual Pre-Training on Parallel Data (https://arxiv.org/abs/2407.03145)
Comments:
          IWSLT2024, 18 pages

- **What's New**: 이 논문에서는 사전 훈련된 대형 언어 모델(LLM)을 두 단계로 훈련하는 방법을 제안합니다. 첫 번째 단계는 웹에서 수집한 병렬 데이터로 지속적인 사전 훈련(continual pre-training)을 수행하고, 두 번째 단계는 소량의 고품질 병렬 데이터로 감독된 미세 조정(supervised fine-tuning)을 하는 것입니다. 이를 통해 일본어-영어와 영어-일본어 번역에서 번역 정확도를 향상시켰습니다.

- **Technical Details**: 연구진은 3.8B 파라미터 모델을 사용하여 웹에서 수집한 병렬 데이터를 바탕으로 지속적인 사전 훈련을 수행했습니다. 병렬 데이터는 소스 문장과 타겟 문장이 번갈아 나오는 형태로 구성되었습니다. 실험 결과, 번역 정확도는 지속적인 사전 훈련 데이터와 추론 시 사용하는 데이터의 문장 순서가 일치할 때만 향상되었습니다. 태그를 추가하여 소스 문장을 표시하는 것이 단순히 소스와 타겟 문장을 병합하는 것보다 높은 정확도를 보였습니다.

- **Performance Highlights**: LLM 기반 번역 모델은 구어체 번역에서 더 강력한 성능을 보였으며, 덜 훈련된 데이터로도 높은 정확도를 달성했습니다. 이는 감독된 인코더-디코더 (encoder-decoder) 모델에 비해 우수한 성능을 나타냅니다. 뿐만 아니라, 번역 방향을 태그(예: '<2en>' 등)로 표시한 데이터로 지속적인 사전 훈련을 수행했을 때 최고 정확도가 달성되었습니다.



### Social Bias Evaluation for Large Language Models Requires Prompt Variations (https://arxiv.org/abs/2407.03129)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)이 보여주는 사회적 편향에 대해 다루고 있으며, 다양한 프롬프트 변화를 통해 이러한 편향을 평가하고 완화하는 방법을 조사했습니다. 특히, 프롬프트(task instruction 및 prompt, few-shot examples, debias-prompt)의 변화에 따른 LLMs의 민감성을 분석했습니다.

- **Technical Details**: 연구진은 프롬프트의 다양성이 LLMs의 작업 성능과 사회적 편향에 미치는 영향을 실험을 통해 조사했습니다. 이전 연구들과 달리 다양한 프롬프트를 사용하여 각 모델의 민감성을 평가했으며, 프롬프트 설정에 따른 편향 감소가 성능 저하로 이어질 수 있다는 트레이드오프가 존재함을 발견했습니다.

- **Performance Highlights**: 실험 결과, LLMs는 프롬프트에 매우 민감하게 반응하여 모델의 순위가 변동됨을 확인했으며, 프롬프트에 따라 성능과 사회적 편향 사이에 트레이드오프가 발생했습니다. 또한, 사례의 모호성이 프롬프트에 대한 민감성을 증가시키는 주요 원인임을 밝혔습니다. 따라서, 다양한 프롬프트를 사용하여 LLMs의 사회적 편향을 비교하는 것이 추천됩니다.



### Cactus: Towards Psychological Counseling Conversations using Cognitive Behavioral Theory (https://arxiv.org/abs/2407.03103)
Comments:
          Under Review

- **What's New**: 심리 상담에 대한 수요가 급증하면서 대규모 언어 모델(LLMs)을 활용한 상담 접근법이 주목받고 있습니다. 이를 위해 실제 상담 데이터를 대체할 수 있는 'Cactus'라는 다중 회담 데이터셋이 소개되었습니다. 이 데이터셋은 인지행동치료(CBT)를 기반으로 하여 현실적인 상담 시나리오를 생성하는 목표 지향적이고 구조화된 접근방식을 사용합니다.

- **Technical Details**: Cactus 데이터셋은 다양한 페르소나를 가진 가상의 클라이언트와 상담사가 CBT 기술을 체계적으로 적용하여 상호작용하는 대화 데이터를 포함하고 있습니다. 이 데이터셋은 PatternReframe에서 얻은 실제 클라이언트의 생각과 패턴을 반영합니다. 또한, CounselingEval이라는 평가 프레임워크를 통해 상담사의 능력과 클라이언트의 심리적 변화를 평가합니다.

- **Performance Highlights**: Cactus로 훈련된 모델 'Camel'은 기존 모델보다 상담 기술에서 우수성을 보였고, 이는 심리 전문가 평가와 LLM 기반 평가 결과로 확인되었습니다. Cactus 데이터셋과 모델, 코드 모두 공개되어 연구 및 개발에 활용 가능합니다.



### A Case Study on Context-Aware Neural Machine Translation with Multi-Task Learning (https://arxiv.org/abs/2407.03076)
Comments:
          Accepted to EAMT 2024 (poster)

- **What's New**: 본 논문에서는 문서 수준의 신경망 기계 번역(DocNMT)에서 불필요한 노이즈를 줄이는 대신, 문맥을 명시적으로 모델링하는 멀티태스크 학습(MTL) 방식을 적용하여 문맥 선택에 민감하게 만드는 방법을 탐구합니다. 주로 독일어-영어 언어 쌍을 대상으로 뉴스, TED, Europarl 말뭉치에서 실험을 수행하였습니다. 이번 연구의 결과는 MTL 접근법이 기존의 연결 기반 및 다중 인코더 DocNMT 모델보다 적은 리소스 상황에서 더 나은 성능을 보여준다는 것을 밝혀냈습니다.

- **Technical Details**: 본 연구에서는 하나의 인코더와 두 개의 디코더로 구성된 캐스케이드 MTL 아키텍처를 사용하였습니다. 문맥으로부터 소스를 생성하는 작업을 보조 업무로, 소스로부터 타겟을 생성하는 작업을 주된 업무로 설정하였습니다. 실험은 News-commentary v14와 TED 말뭉치에서 독일어-영어 언어 쌍을 사용하여 이전 두 개의 소스 문장, 이전 두 개의 타겟 문장, 이전과 다음 소스 문장 등의 세 가지 다른 종류의 문맥을 가지고 진행되었습니다.

- **Performance Highlights**: BLEU 점수를 기준으로 평가한 결과, 제안된 MTL 접근법이 적은 리소스 상황에서 기존 모델들보다 더 좋은 성능을 보여주었으며, 문맥 선택에 민감하게 반응하였습니다. 하지만 MTL 모델이 문맥으로부터 소스를 생성하는 작업에서는 실패하는 경향이 있었습니다. 이는 기존 연구와 일치하며, 문서 수준의 병렬 말뭉치가 문맥을 인식하지 못할 수 있음을 시사합니다.



### ALTER: Augmentation for Large-Table-Based Reasoning (https://arxiv.org/abs/2407.03061)
- **What's New**: ALTER(대형 테이블 기반 추론을 위한 증강)라는 새로운 프레임워크를 도입했습니다. 이 프레임워크는 자유형 자연어 질문을 위한 query augmentor와 반구조화된 테이블 데이터 처리를 위한 table augmentor를 활용하여 대형 테이블 기반 추론 성능을 크게 향상시킵니다.

- **Technical Details**: ALTER는 전체 테이블 데이터를 사용하지 않고 관련 데이터의 작은 부분집합과 사전 증강된 스키마, 의미, 리터럴 정보를 활용합니다. query augmentor는 NL(Natural Language) 질문을 적응시키고 table augmentor는 테이블의 본래 구조와 내용을 해석하여 전체 과정을 통해 적절한 데이터를 통합합니다. 마지막으로 이는 SQL 실행기와 결합되어 증강-필터링-실행 절차를 통해 최적의 추론을 수행합니다.

- **Performance Highlights**: ALTER는 두 개의 테이블 기반 추론 벤치마크에서 동급 최강의 성능을 보였습니다. 대형 테이블 시나리오에서 다른 방법론보다 우수한 성능을 보였으며, 소음과 섭동에 대해 강건성을 유지했습니다.



### Improving Conversational Abilities of Quantized Large Language Models via Direct Preference Alignmen (https://arxiv.org/abs/2407.03051)
- **What's New**: 최근 진행된 대형 언어 모델(LLMs: Large Language Models)의 발전 덕분에, 대화형 챗봇으로 변모하고, 맥락적 뉘앙스를 이해하면서도 인간 가치에 부합하는 문장을 생성할 수 있게 되었습니다. 특히, Instruction Tuning과 인간 피드백을 통한 강화 학습(RLHF: Reinforcement Learning from Human Feedback)과 같은 고급 기술을 통해 이루어진 것인데요. 하지만, 이러한 LLMs를 효율적으로 계산하기 위해 After Training Quantization(PTQ)와 같은 기법들이 사용되면서, '토큰 플리핑(Token-Flipping)' 같은 문제가 발생하여 챗봇의 성능을 저해하게 됩니다. 이를 해결하기 위해 새롭게 제안된 방법이 '퀀타이제이션 인식 직접 선호도 최적화(QDPO: Quantization-Aware Direct Preference Optimization)'입니다. 이는 퀀타이제이션된 LLMs와 풀 프리시젼(full-precision) LLMs를 정렬시켜 대화 능력을 향상시키는 방식입니다.

- **Technical Details**: LLMs가 대화형 챗봇으로 진화하는 과정에서 컴퓨팅 복잡성을 줄이기 위해 여러 카운팅 기법이 사용됩니다. 특히 After Training Quantization(PTQ)과 같은 기법이 많이 사용됩니다. PTQ는 훈련된 LLM의 가중치를 양자화(Quantization)하여 저장 요구량을 줄이고, 이로 인해 발생한 정확도 손실을 보완하기 위해 다양한 방법을 사용합니다. 새로운 평가 방법으로는 'LLM as a Judge' 접근법이 있으며, 이는 GPT-4와 같은 LLM을 이용해 다중 턴 대화에서의 반응성을 평가합니다. 또한, FLASK는 대화 기술을 언어적으로 세분화하여 평가하는 방식을 제공합니다. 이러한 평가 방법들은 주로 풀 프리시젼 챗봇에 적용되지만, QDPO는 비용 효율적인 퀀타이제이션 LLM 챗봇에 적용됩니다. QDPO는 직접 선호도 최적화(DPO) 전략에 영감을 받아, 풀 프리시젼 모델과 퀀타이제이션 모델 간의 선호도 데이터셋을 생성하고 이를 통해 선호도 반영 가중치를 조정하는 방식으로 작동합니다.

- **Performance Highlights**: QDPO는 영어와 한국어로 된 두 개의 Instruction-Tuned LLM, Vicuna와 Mi:dm에서 검토되었고, 기존 PTQ와 Knowledge-Distillation Fine-Tuning 기술을 뛰어넘는 월등한 성능을 보여주었습니다. 특히, 토큰 분포의 Top-1과 Top-2 로짓 간의 차이를 감소시켜 토큰-플리핑을 줄여주고, 이로 인해 더 관련성 높고 일관된 텍스트 출력이 가능합니다.



### Raw Text is All you Need: Knowledge-intensive Multi-turn Instruction Tuning for Large Language Mod (https://arxiv.org/abs/2407.03040)
Comments:
          11 pages, 3 figures

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)을 인스트럭션 튜닝(instruction tuning)하기 위해 지식을 중점으로 한 다중 회전 대화(multi-turn dialogues)를 생성하는 R2S 프레임워크를 소개합니다. 특히, 대화의 논리 체인을 활용하여 LLM이 특정 도메인의 지식을 포함한 대화 데이터를 생성할 수 있도록 합니다. 이를 통해 위키피디아(영어), 과학(중국어), 유물(중국어) 등 다양한 영역의 문서를 다루는 벤치마크 K-BENCH를 구성했습니다.

- **Technical Details**: 제안된 R2S 프레임워크는 먼저 현재 회차 대화의 논리 흐름을 결정하고, LLM에게 관련 응답 내용을 작성할 키 프레이즈를 생성하도록 프롬프트합니다. 이렇게 생성된 지식-집약형 대화 데이터를 통해 gINSTRUCT 인스트럭션 데이터셋을 생성하고, 이를 바탕으로 gLLM 모델을 튜닝하여 원시 문서(raw documents)를 구조화된 다중 회전 대화로 변환합니다. 이 과정에서 CoD(Chain of Dialogue logic)를 통해 대화의 논리적 흐름을 유지하고, 지식을 최대로 보존합니다.

- **Performance Highlights**: 실험 결과, R2S 프레임워크는 벤치마크 K-BENCH에서 뛰어난 성능을 보이며, 원시 문서의 지식을 효과적으로 대상 모델에 주입할 수 있음이 입증되었습니다. 특히, gLLM 모델은 CoD 방법론의 효율성을 입증하며, 논리적이고 도메인 지식이 풍부한 대화를 생성할 수 있음을 보여주었습니다.



### On the Client Preference of LLM Fine-tuning in Federated Learning (https://arxiv.org/abs/2407.03038)
Comments:
          Work in progress

- **What's New**: 해당 연구는 Reinforcement Learning with Human Feedback (RLHF)을 Federated Learning (FL) 프레임워크 내에서 구현하는 방법을 제안합니다. FedBis와 FedBiscuit이라는 새로운 알고리즘을 도입하여 클라이언트의 선호 데이터를 안전하게 사용하면서 협력적으로 모델을 개선하는 방법을 제공합니다.

- **Technical Details**: FedBis는 각 클라이언트가 자신의 로컬 선호 데이터셋을 사용해 바이너리 셀렉터(binary selector)를 훈련하고, 이를 글로벌 모델로 집계합니다. FedBiscuit은 이를 한 단계 더 발전시켜 클라이언트를 선호에 따라 균형 잡힌 클러스터로 나누고 여러 셀렉터를 훈련하는 방법을 제안합니다. 이 과정은 연산 오버헤드를 줄이고 성능 저하를 완화하는 데 중점을 둡니다.

- **Performance Highlights**: 실험 결과, FedBiscuit은 기존의 FedBis뿐만 아니라 전통적인 중앙 집중식 학습 방식보다도 우수한 성능을 보여주었습니다. 특히, 클라이언트 간의 데이터 이질성을 완화하고, 보상 해킹 문제를 해결하는 데 효율적임이 입증되었습니다.



### Strategies for Arabic Readability Modeling (https://arxiv.org/abs/2407.03032)
Comments:
          Accepted to ArabicNLP 2024, ACL

- **What's New**: 이번 논문은 아랍어 가독성 평가(Arabic readability assessment)를 대상으로 다양한 방법을 실험한 결과를 소개합니다. 기존 연구들이 부족한 아랍어 관련 가독성 조사에 대해 체계적으로 접근하여, 규칙 기반 모델(rule-based methods)과 사전 학습된 언어 모델(pretrained language models, PLMs)을 활용했습니다. 또한, 새롭게 구축된 코퍼스(SAMER Arabic Text Simplification Corpus)를 사용해 단어 수준과 문장 단편 수준에서 가독성을 평가했습니다.

- **Technical Details**: 연구에서는 주로 규칙 기반 모델, 빈도 기반 모델 및 사전 학습된 언어 모델을 조합하는 방법을 사용했습니다. 데이터는 15개의 공개된 아랍어 소설에서 추출된 텍스트로 구성된 SAMER 코퍼스를 활용했습니다. SAMER 코퍼스는 가독성 수준을 기준으로 두 개의 단순화된 평행 버전을 포함하며, 각 텍스트의 가독성 수준은 단어를 기준으로 설정됩니다. 다양한 모델링 접근 방식을 조사하고, 새로운 코퍼스에서 모델을 벤치마킹 했습니다.

- **Performance Highlights**: 여러 기법을 조합함으로써 최상의 결과를 얻을 수 있었으며, 단어 수준에서 86.7%의 매크로 F1 점수, 문장 단편 수준에서 87.9%의 매크로 F1 점수를 달성했습니다. 이러한 접근 방식을 통해 아랍어 가독성 평가의 정확도를 크게 향상시킬 수 있었습니다. 코드, 데이터 및 사전 학습된 모델들은 공개되어 있습니다.



### Exploiting Dialect Identification in Automatic Dialectal Text Normalization (https://arxiv.org/abs/2407.03020)
Comments:
          Accepted to ArabicNLP 2024, ACL

- **What's New**: 이 연구는 일상적인 의사소통에서 사용되는 방언 아랍어(Dialectal Arabic, DA)를 표준화하는 'CODAfication' 작업을 탐구합니다. 이 연구는 주요 도시 방언을 중심으로 고유한 병렬 코퍼스를 사용하여 새롭게 개발된 사전 학습 Seq2Seq 모델을 벤치마킹합니다. 또한 방언 식별 정보를 사용하여 모든 방언의 성능을 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: 디글로시 상황에서 다양한 방언이 존재하는 아랍어는 표준화된 철자가 없기 때문에, 특히 소셜 미디어 콘텐츠에서의 다양한 변형과 노이즈가 큰 문제로 작용합니다. 이를 해결하기 위해, 연구진은 다중 아랍 방언 데이터를 사용해 CODA 표준으로 텍스트를 정규화하는데 주력했습니다. 다섯 개 도시 방언(Beirut, Cairo, Doha, Rabat, Tunis)을 대상으로 병렬 코퍼스를 사용했으며, 사전에 학습된 Seq2Seq 모델을 벤치마크했습니다.

- **Performance Highlights**: 다양한 방언에 대한 식별 정보를 활용한 결과, 모든 방언에서 CODAfication의 성능이 향상되었음을 확인했습니다. 또한 코드, 데이터, 사전 학습 모델을 공개하여 후속 연구와 응용의 기초를 마련했습니다.



### What Affects the Stability of Tool Learning? An Empirical Study on the Robustness of Tool Learning Frameworks (https://arxiv.org/abs/2407.03007)
Comments:
          19 pages, 9 figures

- **What's New**: 최근 연구들은 대형 언어 모델(Large Language Models, LLMs)이 실제 응용 프로그램과 상호 작용할 수 있는 도구 학습 방법을 향상시켰습니다. 본 논문에서는 다양한 내부 및 외부 요인이 도구 학습 프레임워크의 성능에 미치는 영향을 탐구합니다. 대표적인 벤치마크 데이터셋을 통해 광범위한 실험을 진행한 결과, LLMs가 시도와 탐험을 통해 상당한 이득을 얻을 수 있음을 발견했습니다.

- **Technical Details**: 도구 학습은 LLMs에 외부 도구를 부여하여 적절한 도구를 선택하고, 올바른 매개변수를 생성하며, 최종적으로 실행 결과를 구문 분석하여 올바른 응답을 생성하는 방법을 가르치는 것을 목표로 합니다. 본 연구에서는 내부 요인과 외부 요인을 두 가지 카테고리로 구분하여 분석하였으며, 내부 요인에는 디코딩 온도, 최대 추론 단계, 기본 LLM의 선택 등이 포함됩니다. 외부 요인에는 사용자 쿼리 스타일, 시스템 프롬프트 및 후보 도구 세트의 변화 등이 포함됩니다. 다양한 평가 지표를 사용하여 도구 사용 LLMs의 성능을 측정하고 여러 흥미로운 발견을 도출했습니다.

- **Performance Highlights**: 실험 결과, 기존의 도구 사용 워크플로우가 다양한 내부 및 외부 요인에 대해 명백한 불안정을 나타냈습니다. 내부 요인 중 적절한 하이퍼파라미터 설정이 LLM이 다양한 솔루션을 생성하도록 도울 수 있지만, 이는 또한 불안정을 초래할 수 있습니다. 외부 요인 중 LLM은 후보 도구 세트의 변화(순서 또는 규모)와 시스템 프롬프트에 민감했습니다. 고급 도구 선택 알고리즘(예: 트리 기반 검색)이 정확성을 향상시킬 수 있지만, 이는 더 많은 환상을 수반하며 불안정하고 상당한 추론 비용이 발생할 수 있습니다.



### Human-like Linguistic Biases in Neural Speech Models: Phonetic Categorization and Phonotactic Constraints in Wav2Vec2.0 (https://arxiv.org/abs/2407.03005)
Comments:
          Accepted to Interspeech 2024. For code and materials, see this https URL

- **What's New**: 딥 뉴럴 음성 모델이 음운론에 대해 알고 있는 정보를 탐구한 연구입니다. 특히, Wav2Vec2 모델이 인간 음성 인식과 유사하게 음운론적 제약을 어떻게 처리하는지를 조사했습니다. /l/과 /r/ 사이의 음향적 연속체에 해당하는 소리를 합성하고, 영어에서 /l/ 또는 /r/만 발생하는 통제된 문맥에 배치하여 분석했습니다. 이 모델은 인간과 유사하게 모호한 소리를 처리할 때 음운론적으로 허용 가능한 카테고리에 대한 편향을 보였습니다.

- **Technical Details**: 본 연구는 Wav2Vec2 모델이 어떻게 음향적으로 모호한 음성을 처리하는지 분석하기 위해 일련의 실험을 수행했습니다. 실험에서는 Google Text-To-Speech API를 사용하여 음향적으로 모호한 단음을 생성하고, 다양한 음운 제약 조건이 있는 문맥에 배치하여 모델의 범주화를 조사했습니다. Wav2Vec2 모델은 CNN과 Transformer 계층으로 구성되어 있으며, LibriSpeech 데이터를 기반으로 자체 지도 학습을 통해 사전 학습되었습니다. 이후 ASR 작업에 대해 미세 조정되었습니다. 모델의 내부 처리 과정을 분석하기 위해 출력 문자 확률 및 숨겨진 상태를 사용한 여러 측정을 수행했습니다.

- **Performance Highlights**: Wav2Vec2 모델은 음운론적으로 허용 가능한 카테고리에 대한 편향이 초기 계층에서 나타나며, ASR 조정 후 이 편향이 강화됨을 발견했습니다. 그러나 완전히 자체 지도 학습된 모델에서도 이 효과가 존재했습니다. 이는 통제된 자극 설계를 사용하여 뉴럴 음성 모델의 특정 언어 지식을 국소화할 수 있음을 보여줍니다.



### SemioLLM: Assessing Large Language Models for Semiological Analysis in Epilepsy Research (https://arxiv.org/abs/2407.03004)
- **What's New**: 이번 연구에서는 최첨단 대형 언어 모델(Large Language Models, LLMs, GPT-3.5, GPT-4, Mixtral 8x7B, Qwen-72chat)의 내재된 지식과 추론 능력을 활용하여 간질(epilepsy) 진단을 지원하는 능력을 테스트했습니다. LLM을 이용해 발작과 관련된 비정형 문서 텍스트를 발작 생성 뇌 영역과 연결한 확률 추정치를 획득했습니다. 이는 1269개의 항목으로 구성된 주석이 달린 임상 데이터를 사용해 수행되었습니다.

- **Technical Details**: 이번 연구에서는 Semio2Brain 데이터베이스를 활용하여 발작 증상 비정형 텍스트와 발작 시작 영역(SOZ)을 연결했습니다. 모델의 성능을 평가하기 위해 zero-shot, few-shot, Chain of Thought (CoT), few-shot CoT, self-consistency (SC) 등 다섯 가지 프롬프트 스타일을 사용했습니다. LLM의 성과는 신뢰도, 추론 능력 및 인용 능력을 포함하여 임상 평가와 비교했습니다. 연구 결과, GPT-4가 모든 평가 메트릭에서 최고 성능을 보였으나, Mixtral8x7B와 같은 모델은 인용 오류와 추론 부정확성을 보였습니다.

- **Performance Highlights**: LLM 모델은 프롬프트 엔지니어링을 통해 성능이 크게 향상되었으며, 일부 모델은 임상 성능에 근접한 결과를 보였습니다. 그러나 몇몇 모델은 낮은 정확도에도 불구하고 과도한 자신감을 나타내기도 했고, 인용 오류 및 환각 현상이 발생하는 등의 문제가 발견되었습니다. GPT-4는 모든 평가 기준에서 복합적으로 최고의 성능을 보였지만, 다른 모델들과 비교했을 때 일부 한계도 존재했습니다.



### VIVA: A Benchmark for Vision-Grounded Decision-Making with Human Values (https://arxiv.org/abs/2407.03000)
- **What's New**: 이번에 소개될 논문은 인간의 가치를 기반으로 시각적 의사결정을 평가하는 VIVA라는 벤치마크를 제시합니다. 대부분의 대규모 Vision-Language Models (VLMs)들이 물리적 수준의 기술에 집중하는 반면, 우리의 연구는 처음으로 인간의 가치를 활용하여 시각으로 묘사된 상황에서 의사결정을 하는 다중모드(multimodal) 능력을 관찰합니다.

- **Technical Details**: VIVA는 여러 실제 상황을 묘사한 1,062개의 이미지를 포함하며, 각 이미지에는 상황에 맞는 행동과 관련된 인간 가치, 그리고 해당 결정을 뒷받침하는 이유가 주석으로 포함되어 있습니다. 해당 데이터세트를 기반으로 두 가지 수준의 과제를 설정했습니다. 첫 번째 레벨에서는 이미지에 대해 가장 적절한 행동을 선택해야 하고, 두 번째 레벨에서는 선택된 행동을 뒷받침하는 인간 가치와 이유를 설명해야 합니다. 이렇게 구성된 벤치마크를 통해 VLM의 정확한 지각, 사회적 추론, 인간 가치를 기반으로 한 의사결정을 평가합니다.

- **Performance Highlights**: VLM의 성능에 대한 광범위한 평가를 통해, 현재 최첨단 모델조차도 이 과제에서 어려움을 겪는다는 사실이 드러났습니다. 예를 들어, GPT-4는 1단계 행동 선택과 2단계 인간 가치 추론에서 결합 정확도 72.3%를 기록했습니다. 추가적인 분석 결과, 행동 결과나 예측된 인간 가치를 통합하는 것이 유익함을 발견했습니다.



### Are Large Language Models Consistent over Value-laden Questions? (https://arxiv.org/abs/2407.02996)
Comments:
          8 pages, 9 figures

- **What's New**: 최근 연구는 대형 언어 모델(LLM)이 특정 가치에 대한 일관성이 결여된다는 기존의 주장과는 달리, 실제로는 일관성이 높다는 것을 발견했습니다. 특히 파라프레이즈(paraphrases), 사용 사례, 번역, 그리고 동일 주제 내에서의 질문에 대해 일관된 답변을 보였습니다. 그러나 논란이 되는 주제에서 여전히 약간의 불일치가 남아있습니다. 기본 모형은 파인 튜닝된 모델보다 일관성이 높았습니다.

- **Technical Details**: 연구진은 가치 일관성을 (1) 질문의 파라프레이즈, (2) 동일 주제 내 관련 질문, (3) 다중 선택 및 자유 응답의 사용 사례, (4) 영어, 중국어, 독일어, 일본어로 번역된 질문에 대한 답변의 유사성으로 정의했습니다. 8천 개의 질문과 300개 이상의 주제를 포괄하는 'ValueConsistency' 데이터셋을 도입하여 이를 측정했습니다.

- **Performance Highlights**: 모델들은 예상보다 일관성이 높았으며, 사람 참가자(n=165)와 비슷하거나 더 높은 일관성을 보였습니다. 논란이 적은 질문에서 더 많은 일관성을 보였고(Fig. 5), 기본 모형이 파인 튜닝된 모형보다 더 일관성이 높았습니다(Fig. 3). 파인 튜닝된 모델도 주제에 따라 일관성이 달라지는 경향이 있었지만, 기본 모형은 주제에 관계없이 고른 일관성을 보였습니다(Fig. 6).



### Mast Kalandar at SemEval-2024 Task 8: On the Trail of Textual Origins: RoBERTa-BiLSTM Approach to Detect AI-Generated Tex (https://arxiv.org/abs/2407.02978)
Comments:
          SemEval-2024

- **What's New**: 이번 연구는 SemEval 2024의 Multigenerator, Multidomain, and Multilingual Black-Box Machine-Generated Text Detection 과제에 참여하여, 자동화된 텍스트 분류 시스템 개발을 목표로 하고 있습니다. 이 연구에서는 RoBERTa-BiLSTM 기반의 분류기를 제안하여, 텍스트를 AI 생성 여부에 따라 분류합니다. 또한, 제안된 모델과 기존 베이스라인 접근법의 비교 연구를 수행하여 효과성을 평가하였습니다.

- **Technical Details**: 연구에서 제안한 시스템은 심층 학습 기법과 특징 엔지니어링을 결합한 하이브리드 접근 방식을 사용합니다. 구체적으로, BiLSTM (Bidirectional Long Short-Term Memory) 신경망과 RoBERTa(미리 훈련된 언어 표현 모델)를 결합하여 입력 문장의 순차적 및 맥락적 정보를 캡처합니다. 이러한 하이브리드 아키텍처는 정확한 분류를 위해 미세한 언어 패턴과 의미적 단서를 효과적으로 포착합니다. 이 시스템은 공식 순위에서 125팀 중 46위에 랭크되었으며, 80.83%의 정확도를 달성했습니다.

- **Performance Highlights**: 제안된 모델은 다양한 AI 모델과 도메인에서 일반화 능력을 평가하기 위해, 검증 세트로부터 얻은 Bloomz 모델 생성 문장을 제외한 다양한 머신 생성 문장을 훈련 세트로 사용합니다. 이는 모델이 새로운 입력에 효과적으로 일반화하고, 머신 생성 텍스트의 전체 스펙트럼에서 신뢰할 수 있는 예측을 할 수 있도록 합니다. 모델은 높은 정확도로 AI 생성 문장과 인간 생성 문장을 구별할 수 있도록 설계되었습니다.



### Large Language Models as Evaluators for Scientific Synthesis (https://arxiv.org/abs/2407.02977)
Comments:
          4 pages, forthcoming as part of the KONVENS 2024 proceedings this https URL

- **What's New**: 이번 연구는 최첨단 대형 언어 모델(LLMs)인 GPT-4와 Mistral이 과학적 요약 또는 합성 평가에서 사람 평가자의 평가와 얼마나 유사한지 비교합니다. GPT-4가 다섯 관련 논문의 초록에서 생성한 100개의 연구 질문과 그 합성물을 사용하여 인간 품질 등급과 비교했습니다. 초기 결과에 따르면 LLM은 논리적인 설명을 제공할 수 있지만 인간과 LLM 간의 평가상 상관관계는 약하다고 나타납니다. 이는 LLM이 과학적 합성 평가에서 잠재력과 한계를 모두 가지고 있음을 시사합니다.

- **Technical Details**: 본 연구는 LLM을 활용하여 과학적 초록 합성 평가를 수행했습니다. 평가 작업에는 폐쇄형 모델인 GPT-4 터보와 오픈 소스 모델인 Mistral-7B를 사용했습니다. 각 LLM은 CORE-GPT 데이터셋의 100개의 연구 질문과 이에 대한 합성물을 평가했으며, 평가 기준으로는 포괄성(comprehensiveness), 신뢰성(trustworthiness), 유용성(utility) 세 가지 측면에서 점수를 매겼습니다. 평가 결과는 JSON 포맷으로 보고되어 수치 등급과 각 등급의 근거를 포함했습니다.

- **Performance Highlights**: 결과적으로, GPT-4와 Mistral 모두 특정 측면에서는 논리적인 평가를 제공할 수 있었으나, 인간 평점과의 통계적 상관관계는 약했습니다. 이는 LLM이 과학적 합성 평가에서 유용할 수 있지만, 현재로서는 인간 평가자의 역할을 완전히 대체하기 어렵다는 것을 시사합니다.



### FSM: A Finite State Machine Based Zero-Shot Prompting Paradigm for Multi-Hop Question Answering (https://arxiv.org/abs/2407.02964)
- **What's New**: 대규모 언어 모델(LLMs)이 단순한 자연어 추론 작업에서 인상적인 성과를 보였지만, 다중 단계 질문 응답(Multi-hop Question Answering, MHQA) 작업에서는 여러 도전과제 때문에 성능이 저조한 경향이 있습니다. 이에 대해 Finite State Machine (FSM)이라는 프롬프트 방식을 제안하여 LLM의 복잡한 작업에 대한 추론 능력을 향상시키고, 효과성과 신뢰성을 높이는 방법을 소개합니다.

- **Technical Details**: FSM은 질문을 다중턴(sub-question)으로 분해하고, 각 단계마다 자기 수정 기능을 통해 정확성을 높이는 방식입니다. 이는 COT(Chain-of-Thought) 방식과는 달리, 각각의 하위 질문을 하나씩 해결하고 현재 결과와 상태에 따라 다음 단계를 결정하는 자동자 형식으로 진행됩니다. 이 방식은 또한 잘못된 중간 추론 단계를 복구하여 최종 정답을 도출할 수 있는 능력을 갖추고 있습니다. FSM은 명확하고 명시적인 하위 작업을 통해 추론 과정을 보다 관리하기 쉽고 정확하게 만듭니다.

- **Performance Highlights**: FSM은 기존 베이스라인 모델과 비교하여 특히 복잡한 데이터셋에서 뛰어난 성과를 보였습니다. 실험 결과에 따르면, Musique같이 어려운 데이터셋에서 F1 점수가 거의 두 배 향상되었습니다. 또한, FSM 방법은 예기치 않은 포맷 오류나 타입 오류를 크게 줄여서 평가 과정에서의 정답 추출이 용이하게 만듭니다.



### Probing the Feasibility of Multilingual Speaker Anonymization (https://arxiv.org/abs/2407.02937)
Comments:
          accepted at Interspeech 2024

- **What's New**: 이 연구는 기존의 화자 익명화 시스템을 아홉 개 언어로 확장한 최초의 다국어 연구입니다. 본 연구에서는 영어 데이터로부터 훈련된 화자 임베딩(speaker embeddings)이 여러 언어에 적용 가능하다는 점과, 언어별 익명화 성능이 주로 음성 합성(speech synthesis) 구성 요소의 품질에 영향을 받는다는 것을 발견했습니다.

- **Technical Details**: 이 시스템은 세 단계로 이루어집니다. 첫째, 입력 음성에서 화자 임베딩, 음성의 운율(prosody), 그리고 언어적 내용을 추출합니다. 둘째, 원래 화자 임베딩을 GAN(Generative Adversarial Network)을 사용해 인공 임베딩으로 대체합니다. 마지막으로, 수정을 거친 화자 임베딩과 음성을 FastSpeech-2-like TTS 및 HiFiGAN vocoder로 재합성합니다. 또한, Whisper large-v2 모델을 사용하여 다국어 ASR을 구현하였으며, TTS 시스템도 다국어를 지원하도록 조정했습니다.

- **Performance Highlights**: 실험 결과, 이 다국어 화자 익명화 시스템이 연구에 포함된 모든 언어에서 화자의 개인정보를 보호하는 데 성공적임을 확인했습니다. 다만, 익명화된 발화의 음성 인식 유틸리티가 감소하는 현상이 나타났고, 이는 다국어 TTS 모델의 품질에 주로 기인하는 것으로 나타났습니다.



### Towards Negotiative Dialogue for the Talkamatic Dialogue Manager (https://arxiv.org/abs/2407.02917)
- **What's New**: 새로운 연구에서는 Talkamatic Dialogue Manager (TDM)의 초기 개발 버전에 협상적 대화 (negotiative dialogue) 관련 현상을 구현했다고 설명합니다. 이는 TDM이 단순한 폼 기반(dialogue) 대화 그 이상을 지원할 수 있도록 확장된 첫 단계입니다.

- **Technical Details**: 이 연구는 Tala SDK 개발을 시작으로, 대화 도메인 설명 (Dialogue Domain Descriptions, DDDs)를 3자(3rd party)를 통해 생성, 검증 및 배포할 수 있는 환경을 조성하는 것을 목표로 합니다. TDM은 도메인 지식과 일반 대화 지식을 분리하여 설계되었고, 이를 통해 개발자는 도메인별 지식 (예: 시멘틱 온톨로지 및 자연어 표현)에 집중할 수 있게 합니다.

- **Performance Highlights**: 첫 번째 데이터 수집 및 분석을 통해 협상적 대화와 관련된 빈번한 현상을 발견하고 이를 TDM에 구현했습니다. TDM은 기존의 폼 기반(dialogue) 대화 외에도 사용자가 제공하는 정보에 따라 점진적으로 결과를 제공합니다. 예를 들어, 사용자가 특정 전화번호를 요청할 때 여러 선택지를 비교하거나 탐색할 수 있는 기능이 추가되었습니다. 또한 'Do you know ...?' 형식의 질문 (Knowledge Precondition Questions, KPQs)을 처리하여 더욱 자연스러운 대화 흐름을 지원합니다.



### Translatotron-V(ison): An End-to-End Model for In-Image Machine Translation (https://arxiv.org/abs/2407.02894)
Comments:
          Accepted to ACL 2024 Findings

- **What's New**: Translatotron-V(ision)은 텍스트를 포함한 이미지를 대상으로 한 언어 번역을 수행하기 위한 최초의 end-to-end IIMT 모델로서, 기존의 복잡한 cascade 방식과는 달리, 오류 전파를 방지하고 더 적은 파라미터로 구현할 수 있습니다. 특히, RGB 이미지를 생성하여 기존 모델의 한계를 극복합니다.

- **Technical Details**: Translatotron-V(ision)은 이미지 인코더, 타겟 텍스트 디코더, 이미지 디코더, 이미지 토큰화기(image tokenizer)로 구성됩니다. 타겟 텍스트 디코더는 모델의 언어 정렬 부담을 줄이며, 이미지 토큰화기는 긴 픽셀 시퀀스를 짧은 시각적 토큰으로 변환하여 모델이 저수준의 시각적 특징에 집중하는 것을 방지합니다. 모델 학습을 위해 대규모 비라벨 이미지로 이미지 토큰화를 먼저 학습하고, 이후 IIMT 데이터셋을 사용하여 다른 모듈들을 훈련하는 두 단계의 학습 프레임워크를 제안합니다.

- **Performance Highlights**: Translatotron-V(ision)은 파라미터 수가 70.9% 감소하면서도 cascade 모델과 비교하여 경쟁력 있는 성능을 발휘합니다. 또한, 픽셀 레벨 end-to-end 모델보다 훨씬 뛰어난 번역 품질을 자랑합니다.



### Universal Gloss-level Representation for Gloss-free Sign Language Translation and Production (https://arxiv.org/abs/2407.02854)
Comments:
          14 pages, 5 figures

- **What's New**: 본 연구는 SLT와 SLP 모두에 적용할 수 있는 새로운 통합 접근법인 Universal Gloss-level Representation (UniGloR)를 제안합니다. UniGloR는 다중 데이터셋(PHOENIX14T, How2Sign, NIASL2021)에 대해 자가 지도 학습(Self-supervised learning) 방법론을 기반으로 학습되며, 이전의 글로스 주석을 완전히 대체할 수 있습니다.

- **Technical Details**: UniGloR는 오토인코더(autoencoder)를 사용하여 비디오 내의 각 사인(segment)을 위한 독특한 표현을 학습합니다. 더불어 적응형 포즈 가중치 (Adaptive Pose Weights, APW)를 도입하여 수화 모션의 세부적인 움직임, 표정, 손동작을 효과적으로 캡처합니다. 그 결과, UniGloR는 외부 주석 데이터나 추가적인 적응 과정(fine-tuning) 없이도 높은 성능을 보장할 수 있습니다.

- **Performance Highlights**: 실험 결과, UniGloR는 PHOENIX14T, How2Sign 및 NIASL2021 데이터셋에서 SLT와 SLP 모두에서 우수한 성능을 보여주었고, 새로운 데이터(KSL-Guide-Word)에서도 그 효과가 입증되었습니다. 이를 통해, UniGloR는 혁신적이고 실용적인 응용 가능성을 제공합니다.



### Comparing Feature-based and Context-aware Approaches to PII Generalization Level Prediction (https://arxiv.org/abs/2407.02837)
Comments:
          Accepted to IALP 2024

- **What's New**: 새로운 연구는 PII(개인 식별 정보) 보호를 위한 두 가지 접근법을 제안하며, 기존 방법들이 겪는 데이터 분포 불균형과 한정된 문맥 인식을 극복하고자 합니다. 첫 번째 방법은 구조화된 입력에서 성능을 향상시키기 위해 머신러닝을 사용하는 특징 기반 접근법입니다. 두 번째 방법은 보다 넓은 문맥과 원본 텍스트와 일반화된 후보들 간의 의미적 관계를 고려하는 새로운 문맥 인식(context-aware) 프레임워크입니다.

- **Technical Details**: 문맥 인식 접근법은 Multilingual-BERT를 사용하여 텍스트 표현을 생성하고, 함수 변환과 평균 제곱 오차(mean squared error) 점수를 통해 후보들을 평가합니다. 이 방법은 Wikipedia biography 데이터셋에서 평가되었으며, 다양한 규모에서 특징 기반 모델을 능가하는 성과를 보였습니다.

- **Performance Highlights**: WikiReplace 데이터셋에서 실험 결과, 문맥 인식 접근법이 특징 기반 접근법보다 더 나은 성능을 보였습니다. 특히 다른 데이터셋 규모에서도 일관되게 높은 정확성을 나타내었습니다.



### Aspect-Based Sentiment Analysis Techniques: A Comparative Study (https://arxiv.org/abs/2407.02834)
- **What's New**: 이 연구는 레스토랑과 노트북의 고객 리뷰를 분석하기 위해 다양한 Deep Neural Network (DNN) 기반의 Aspect-based Sentiment Analysis (ABSA) 모델을 비교하고 있습니다. 최신 AI 기법의 도움으로 종래와 달리 리뷰 내용의 구체적인 측면까지 감성 분석을 수행하는 것이 주요 초점입니다.

- **Technical Details**: 이 논문에서는 주요적으로 LLaMA 2 모델의 QLoRA 아키텍처 기반의 Parameter-Efficient Fine-Tuning (PEFT) 기법, PyABSA 프레임워크에서의 FAST_LSA_T_V2 모델, 그리고 SetFit 프레임워크 내에서의 transformer 사전 학습 모델을 사용했습니다. LLaMA 2는 Meta에서 개발된 개방형 대형 언어 모델 (LLM)로, 다양한 자연어 처리 작업에서 진보된 성능을 보여줍니다. 파인튜닝 과정에서는 Instruction Fine-Tuning (IFT), Full Fine Tuning, 그리고 파라미터 효율적 파인튜닝 (PEFT) 방식을 채택했습니다.

- **Performance Highlights**: 논문에서는 다양한 ABSA 모델의 성능을 두 가지 기준 데이터셋(Restaurant14, Laptop-14)을 통해 평가했습니다. FAST LSA 모델은 각각 87.6%와 82.6%의 정확도를 기록한 반면, LSA+DeBERTa 모델은 90.33%와 86.21%로 가장 높은 정확도를 보였습니다.



### Investigating the Contextualised Word Embedding Dimensions Responsible for Contextual and Temporal Semantic Changes (https://arxiv.org/abs/2407.02820)
- **What's New**: 단어의 의미 변화는 시간의 경과와 문맥에 따라 발생하며, 이를 반영한 새로운 연구가 발표되었습니다. 이 연구에서는 WiC(Word-in-Context) 데이터에 마스킹 언어 모델(MLMs)을 미세 조정하여 생성된 SCWE(Sense-Aware Contextualised Word Embeddings)의 의미 변화 코딩 방식을 조사하였습니다.

- **Technical Details**: 이 연구에서는 사전 학습된 CWEs(Contextualised Word Embeddings)와 WiC 데이터를 사용해 미세 조정된 SCWEs의 문맥적 및 시간적 의미 변화 코딩 방식을 PCA(Principal Component Analysis)와 ICA(Independent Component Analysis) 변환을 통해 비교했습니다. 주요 발견사항 중 하나는 의미 변화를 책임지는 축의 수가 사전 학습된 CWE 공간에서는 적지만, 미세 조정 후에는 모든 차원에 고르게 분포된다는 점입니다. 또한, PCA가 ICA보다 의미 변화를 더 잘 나타낸다는 사실을 발견했습니다.

- **Performance Highlights**: 미세 조정된 embedding에서는 10%의 주성분을 사용하여도 전체 차원을 사용하는 것과 비슷한 혹은 더 나은 성능을 보였습니다. 반면, ICA로 변환된 축은 문맥적/시간적 의미 변화를 잘 나타내지 못했습니다. 이 결과는 PCA가 의미 변화 탐지에 더 적합함을 시사합니다.



### Efficient Training of Language Models with Compact and Consistent Next Token Distributions (https://arxiv.org/abs/2407.02819)
Comments:
          ACL 2024

- **What's New**: 이 논문에서는 기존의 다음 토큰의 가능성을 최대화하는 목적에서 벗어나, n-그램(n-gram) 분포를 통해 코퍼스를 미리 집계함으로써 더 나은 모델을 더 빠르게 훈련할 수 있음을 보여줍니다. 기존 연구들에서는 코퍼스 수준의 n-그램 통계를 정규화로 사용했으나, 이를 단순하게 구현하면 비용이 많이 들고 훈련 속도가 크게 저하됩니다. 이 논문은 이를 해결하기 위해 데이터 구조를 효율적으로 설계하고, 확장 가능한 새로운 방법인 'CoCoNTs'를 제안합니다.

- **Technical Details**: 제안된 방법은 다음 토큰 분포를 간결하게 표현하면서도 통계적으로 일관성을 유지합니다. 이를 통해 미니 배치 전반에 걸쳐 분산을 크게 줄이고, 훈련 속도를 저하시키지 않으면서 효과적으로 사용할 수 있습니다. 구체적으로 제안된 방법은 n-그램 정규화 모델과 유사한 성능을 제공하며, 이전 방법들과 달리 훈련 데이터셋 크기에 비례하지 않고 일정한 저장 공간만 필요합니다. 또한 기존 방법들에 비해 최적화 단계를 약 50% 줄일 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 CoCoNTs 방법이 모델 품질과 수렴 속도 측면에서 기존 방법들보다 뛰어남을 확인했습니다. 특히, 대규모 데이터셋과 모델에서도 동일한 성능 향상을 보여주었고, 검증 퍼플렉시티(validity perplexity) 측정에서 기존 목표값과 유사한 단계를 훨씬 적게 필요로 합니다. 또한, 사전 훈련 및 파인 튜닝 모두에 있어서 개선된 성능을 보였습니다.



### 52B to 1T: Lessons Learned via Tele-FLM Series (https://arxiv.org/abs/2407.02783)
Comments:
          For the Tele-FLM-52B tech report, see also 2404.16645

- **What's New**: 이번 연구에서는 Large Language Model (LLMs)의 확장 가능성에 대해 고찰한다. 특히, 52억 파라미터의 Tele-FLM 모델을 102억 및 1조 파라미터로 확장하는 방법을 제안하고 실험을 진행했다. 또한, Supervised Fine-tuning (SFT)의 'less is more' 접근 방식을 통해 더 적은 양의 고품질 데이터를 사용해도 더 나은 성능을 얻을 수 있음을 발견했다. 최종적으로 1조 파라미터 모델인 Tele-FLM-1T의 체크포인트를 오픈 소스화하여 연구 커뮤니티에 기여할 예정이다.

- **Technical Details**: Tele-FLM 모델은 GPT 스타일의 디코더 전용 트랜스포머 아키텍처를 사용한다. RMSNorm을 정상화에, SwiGLU를 활성화 함수로 사용하며, Rotary Positional Embedding (RoPE)을 포함하고 있다. 이 모델은 각 단계에서 기능 보존 확장 기법을 활용하여 52B에서 102B, 그리고 최종적으로 1T 파라미터로 확장된다. SFT 데이터는 다양한 도메인의 작업을 포함하며, 특히 수학, 코딩, 멀티턴 대화에 집중했다. 학습률은 2.7e-5로 시작하여 선형 스케줄로 1e-9까지 감소시켰다.

- **Performance Highlights**: Tele-FLM-Chat 모델은 AlignBench 및 TeleEval 벤치마크에서 평가되었으며, 중국어 이해 및 생성 작업에서 GPT-4와 유사하거나 더 우수한 성능을 보였다. AlignBench에서 GPT-4 퍼포먼스의 91%를 달성하며, TeleEval에서는 93%의 성능을 기록했다. 특히 수학 및 논리적 추론 작업에서는 비교적 낮은 성능을 보였으나, 이는 SFT 데이터의 부족 때문으로 해석된다. 의학, 암호화와 같은 복잡한 작업을 다루기 위해서는 더 많은 중간 단계와 chain of thoughts(supervision on intermediate steps and chain of thoughts)가 필요할 것으로 보인다.



### A Framework for Quantum Finite-State Languages with Density Mapping (https://arxiv.org/abs/2407.02776)
Comments:
          14 pages, 5 figures

- **What's New**: 이 논문은 양자 유한 상태 오토마톤(QFA, Quantum Finite-state Automaton)을 설계 및 시뮬레이션할 수 있는 새로운 프레임워크를 제안합니다. 이 프레임워크는 직관적인 방식으로 QFA를 구성하고 시뮬레이션 정확도를 극대화합니다. 이를 통해 MOD 및 EQU와 같은 기본 언어를 인식하는 QFA의 사전 정의된 구성을 제공합니다.

- **Technical Details**: 새로운 프레임워크는 두 가지 방법론에 의존합니다. 첫째, MOD와 EQU라는 특별한 언어를 인식하는 기본적인 QFA 유형의 사전 정의된 구성을 제공합니다. 이들은 더 복잡한 QFA의 기본 빌딩 블록 역할을 합니다. 둘째, 이러한 QFA를 양자 회로로 변환하여 시뮬레이션 정확도를 높입니다. 또한, 밀도 매핑(density mapping)이라는 새로운 개념을 도입하여 QFA 상태의 비트 표현을 결정하고 비트 플립 오류로 인한 변동 확률을 줄입니다.

- **Performance Highlights**: 이 프레임워크는 사용자가 단순한 QFA를 구성하고, 언어 연산을 통해 더 복잡한 QFA를 조합하며, 결과적으로 밀도 매핑을 사용하여 이를 회로로 변환하도록 합니다. 이를 통해 양자 회로의 구성 및 시뮬레이션 정확도가 향상되었습니다.



### MLKD-BERT: Multi-level Knowledge Distillation for Pre-trained Language Models (https://arxiv.org/abs/2407.02775)
- **What's New**: MLKD-BERT는 BERT 압축을 위한 새로운 지식 증류(knowledge distillation) 방법을 제안합니다. 이 방법은 다중 레벨의 지식을 증류하여 성능을 향상시키고, 학생 모델의 주의 헤드(attention head) 수를 유연하게 설정할 수 있게 하여 추론 시간을 단축시킵니다.

- **Technical Details**: MLKD-BERT는 두 단계의 증류 절차를 사용합니다. 첫 번째 단계에서는 임베딩 레이어와 트랜스포머 레이어에서 특성(feature) 및 토큰 간 관계(relation)을 증류합니다. 두 번째 단계에서는 예측 레이어에서 샘플 간 관계 및 소프트 레이블을 학습합니다. 또한, 학생 모델이 교사 모델의 동일한 수의 주의 헤드를 필요로 하지 않도록 셀프 어텐션 관계를 학습합니다.

- **Performance Highlights**: 광범위한 실험 결과, MLKD-BERT는 기존 최첨단 BERT 지식 증류 방법을 능가합니다. GLUE 벤치마크 및 추출 기반 질문 응답(Extractive Question Answering) 작업에서 높은 성능을 보여주며, 주의 헤드 수를 줄여도 성능 저하 없이 추론 시간을 크게 단축할 수 있습니다. GLUE 작업에서 평균적으로 99.5%의 성능을 유지하면서도 매개변수와 추론 시간을 50% 줄였습니다.



### Emotion and Intent Joint Understanding in Multimodal Conversation: A Benchmarking Datas (https://arxiv.org/abs/2407.02751)
Comments:
          26 pages, 8 figures, 12 tables, NeurIPS 2024 Dataset and Benchmark Track

- **What's New**: 이 논문에서는 감정과 의도를 동시에 이해하기 위한 다중모달 대화 데이터셋 MC-EIU를 제안합니다. 이 데이터셋은 7가지 감정 카테고리와 9가지 의도 카테고리를 포함하며, 텍스트(textual), 음향(acoustic), 시각(visual) 세 가지 모달리티를 포함하고 있습니다. 또한 영어와 중국어 두 가지 언어로 제공됩니다. MC-EIU는 완전히 오픈소스로 제공되며, 이는 다중모달 대화 이해를 위한 최초의 종합적인 데이터셋입니다.

- **Technical Details**: MC-EIU 데이터셋은 4970개의 대화 동영상 클립으로 구성되어 있으며, 여기에는 3개의 영어 및 4개의 중국어 TV 시리즈가 포함됩니다. 총 45009개의 영어 발화와 11003개의 중국어 발화가 포함됩니다. 이 데이터셋은 실세계와 밀접하게 관련된 대화 시나리오를 제공합니다. 이와 함께, 감정-의도 상호작용 네트워크(EI2)라는 기준 시스템도 개발되었습니다. 이 시스템은 다중모달 대화에서 감정과 의도의 심층적 상호작용을 모델링합니다.

- **Performance Highlights**: EI2 프레임워크는 MC-EIU 데이터셋 상에서 기존 최첨단 모델들을 능가하는 성능을 보였습니다. 비교 실험과 제거 연구(ablation studies)를 통해 EI2의 효과가 입증되었습니다. 특히, 여러 감정 및 의도를 동시에 예측할 수 있는 능력이 기존 방법론들에 비해 우수함을 확인했습니다.



### Learning to Reduce: Towards Improving Performance of Large Language Models on Structured Data (https://arxiv.org/abs/2407.02750)
Comments:
          ICML 2024 Workshop on Long-Context Foundation Models, Vienna, Austria 2024. arXiv admin note: substantial text overlap with arXiv:2402.14195

- **What's New**: 이 논문은 구조화된 데이터에 대한 대규모 언어 모델(LLMs)이 어려움을 겪는 문제를 해결하기 위해 'Learning to Reduce'라는 프레임워크를 제안합니다. 이 프레임워크는 On-Policy Learning을 통해 입력된 구조화된 데이터를 줄이는 모델을 미세 조정(fine-tune)합니다. 특히, 'Learning to Reduce'가 GPT-4와 같은 최첨단 LLM들과 비교했을 때 뛰어난 성능을 발휘하며, 다양한 데이터셋에서도 일반화 능력을 보여준다는 점이 강조되었습니다.

- **Technical Details**: 제안된 프레임워크는 T5 언어 모델을 기반으로 하고 있으며, 표(question answering) QA 작업을 다운스트림(task)으로 설정했습니다. 모델은 입력된 데이터를 줄이기 위해 관련 증거를 식별하고, On-Policy Learning을 통해 미세 조정됩니다. 구체적으로는 입력 테이블에서 관련 행과 열을 줄여서 프롬프트의 길이를 줄이는 방식입니다. 이 작업을 통해 LLM이 표 QA 작업을 더 잘 수행할 수 있게 됩니다.

- **Performance Highlights**: 'Learning to Reduce' 프레임워크는 WikiTableQuestions 데이터셋에서 더 나은 성능을 보여주었으며, 보지 못한 새로운 데이터셋에서도 우수한 일반화 성능을 입증했습니다. 특히, 콘텍스트가 길 때 LLM의 정확성을 크게 향상시켰습니다. 이는 구조화된 데이터 QA 작업에서 비용 효율성을 극대화하는 데 도움을 줄 수 있습니다.



### MentalAgora: A Gateway to Advanced Personalized Care in Mental Health through Multi-Agent Debating and Attribute Contro (https://arxiv.org/abs/2407.02736)
- **What's New**: MentalAgora라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 여러 에이전트 간의 상호작용을 통해 대형 언어 모델 (Large Language Models, LLMs)을 강화하여 사용자의 정신 건강 지원을 맞춤화합니다.

- **Technical Details**: MentalAgora는 세 가지 단계로 작동합니다: 전략적 토론, 맞춤형 상담사 생성, 그리고 응답 생성. 이 프레임워크는 TherapyTalk 데이터셋을 사용하여 전문가와 사용자 선호도가 반영된 응답을 생성합니다. 특히, 멀티 에이전트 방법론을 도입하여 각 에이전트가 다른 측면을 모델링하며, 사용자의 상태와 문제를 다양한 관점에서 평가합니다.

- **Performance Highlights**: 실험 및 사용자 연구 결과, MentalAgora가 전문가의 기준에 부합하며 사용자 선호도도 효과적으로 충족시킴을 증명했습니다. 또한 실제 사용자 문제를 처리하는 과정에서도 높은 만족도를 자랑하며 디지털 정신 건강 개입의 새로운 기준을 설정했습니다.



### e-Health CSIRO at "Discharge Me!" 2024: Generating Discharge Summary Sections with Fine-tuned Language Models (https://arxiv.org/abs/2407.02723)
Comments:
          BioNLP @ ACL 2024

- **What's New**: 최근 연구는 'BioNLP 2024 Shared Task on Streamlining Discharge Documentation (Discharge Me!)'을 통해 임상 기록 부담을 줄이기 위해 자동으로 퇴원 요약 섹션을 생성하는 방법을 탐구했습니다. 이 과제는 입원 기간 중의 간단한 병원 경과 및 퇴원 지침을 자동으로 작성하는 것을 목표로 합니다.

- **Technical Details**: 연구팀은 다양한 오픈소스 언어모델(LM)들을 미세 조정(fine-tuning)하여, 디코더 전용 모델과 인코더-디코더 모델 모두에서 입력 문맥에 따른 여러 설정을 실험했습니다. 모델 앙상블링, 디코딩 알고리즘, 모델 특화 등의 다양한 설정을 탐구해보았습니다. 특히, 퇴원 요약의 내용을 활용하여 목표 섹션을 생성하는 것은 높은 성능을 보였습니다.

- **Performance Highlights**: 흥미롭게도, 작은 인코더-디코더 언어모델(encoder-decoder LMs)이 큰 디코더 기반 언어모델보다 성능이 우수하거나 유사한 결과를 보여주었습니다. 연구팀의 최적화된 모델 체크포인트는 커뮤니티에 공개되어 있으며, 최종 리더보드에서는 자동 및 수동 평가 모두에서 3위를 기록했습니다.



### Boosting Biomedical Concept Extraction by Rule-Based Data Augmentation (https://arxiv.org/abs/2407.02719)
- **What's New**: 이 논문은 문서 수준의 생의학적 개념 추출을 개선하기 위해 기존 규칙 기반 개념 매핑 시스템인 MetaMapLite를 활용하여 추가적인 가상 주석 데이터를 생성하고 이를 사용해 제한된 훈련 데이터를 보강하는 방법을 제안합니다. 이는 특히 비정형적 개념 이름을 포함한 생의학적 개념을 더 잘 인식할 수 있도록 도와줍니다.

- **Technical Details**: 이 연구는 MetaMapLite를 사용하여 PubMed 및 PMC에서 가상 주석 데이터를 생성하고 이를 훈련 데이터로 사용하여 비정형적 개념 이름을 감지하는 신경망 모델을 훈련합니다. 이 접근법은 주석이 항상 정확하지 않더라도, 대부분이 정확한 경우 유용함을 보여줍니다. 또한, 이 연구는 기존 학습 데이터에 더 많은 레이블이 추가된 데이터를 생성할 수 있음을 보입니다.

- **Performance Highlights**: 풍부한 실험을 통해, MetaMapLite와 같은 수작업으로 작성된 개념 매핑 도구가 더 나은 개념 추출 모델을 훈련하는 데 유용하다는 것을 보여줍니다. 이를 통해 훈련된 신경망 모델은 기존 모델보다 비정형적 개념 이름을 탐지하는 능력이 향상됩니다.



### Ensuring Responsible Sourcing of Large Language Model Training Data Through Knowledge Graph Comparison (https://arxiv.org/abs/2407.02659)
- **What's New**: 최근 큰 언어 모델(LLM) 개발자들을 상대로 저작권 있는 코퍼스의 사용에 대해 제기된 표절 혐의에 대응하기 위해, 우리는 새로운 시스템을 제안합니다. 이 시스템은 기존의 표절 탐지 시스템의 변형으로, 특정 지식 소스가 대형 언어 모델의 학습이나 미세 조정(fine-tuning) 과정에 사용되었는지를 평가합니다. 우리는 RDF(Resource Description Framework) 트리플을 사용하여 소스 문서와 LLM의 생성 텍스트로부터 지식 그래프를 생성하고, 이들을 코사인 유사도(cosine similarity)와 그래프 편집 거리(graph edit distance)로 분석합니다.

- **Technical Details**: 본 시스템은 소스 문서와 LLM이 생성한 지속 문서에 대해 RDF 트리플을 [subject, predicate, object] 형식으로 추출하여 지식 그래프를 만듭니다. 이후 각 지식 그래프의 코사인 유사도를 비교하고, 구조적 유사성을 평가하기 위해 정규화된 그래프 편집 거리를 사용합니다. 이 접근법은 LLM의 혼합 지식(graph) 또는 perplexity와 같은 지표가 없더라도 작동하며, 소스 문서와 LLM 생성 텍스트 간의 아이디어와 조직의 유사성을 보다 폭넓게 평가할 수 있습니다.

- **Performance Highlights**: 시스템 프로토타입은 GitHub 저장소에서 제공될 예정입니다. 현재 시스템은 'black-box' 대형 언어 모델링 시스템의 한계를 다루며, 전통적인 표절 탐지 시스템의 한계를 넘어서는 더욱 정확한 비교를 할 수 있도록 돕습니다.



### Change My Frame: Reframing in the Wild in r/ChangeMyView (https://arxiv.org/abs/2407.02637)
Comments:
          3 pages, NAACL 2024 workshop

- **What's New**: 최근 텍스트 스타일 전환(text style transfer) 연구에서는 맥락 밖에서 과제를 수행하는 발화를 사용하여 중립적 또는 낙관적인 재구성을 시도해왔습니다. 이번 연구에서는 subreddit r/ChangeMyView(CMV)를 기반으로 재구성을 일반화하려고 합니다. CMV 커뮤니티의 상호작용과 관례를 활용하여 관점을 바꾸는 데 기여하는 고가치 발언을 식별하는 데이터를 구축했습니다. 이 데이터로 인해 재구성 방향의 범위가 확장되었으며, 관점 변경이 중립적 또는 긍정적인 방향뿐만 아니라 다양한 방향으로 이루어질 수 있습니다.

- **Technical Details**: CMV에서는 사용자가 그들의 관점을 변경하려고 요청하는 게시물(OP)을 작성하고, 다른 커뮤니티 멤버들은 다양한 관점을 제시하는 댓글을 답니다. 귀한 의견으로 판단되는 댓글에는 델타(delta)가 부여됩니다. 초기 데이터셋은 약 255,287개의 게시물과 11,461,626개의 댓글로 구성되며, 2015-2022년 데이터를 사용했습니다. 이 중에서 델타가 부여된 32,306개의 (게시물, 댓글) 쌍을 분류하여 데이터셋을 구축했습니다. BART와 T5와 같은 인코더-디코더 모델을 사용하여 데이터셋을 미세 조정하였고, 평가 지표로 BLEU, ROUGE, BERTScore를 사용했습니다.

- **Performance Highlights**: 우리의 실험에서는 BERTScore가 데이터양을 줄임으로써 개선되는 경향을 보였습니다. 크로스 데이터셋 실험 결과, 우리의 데이터셋은 다른 데이터셋에 비해 성능을 저해하며 최적의 미세 조정을 어렵게 만든다는 것을 발견했습니다. GPT-4를 사용한 데이터쌍 일부 제거 실험에서는 불필요한 텍스트를 줄임으로써 최고 수준의 BERTScore와 유사성 기반 성능을 기록했습니다.



### Nollywood: Let's Go to the Movies! (https://arxiv.org/abs/2407.02631)
Comments:
          8 pages, 4 figures, 2 tables

- **What's New**: 이 연구는 나이지리아 영화 산업인 Nollywood의 영화를 대상으로 한 새로운 음성 인식 방법을 제안합니다. 기본적으로 두 가지 목표를 달성하고자 합니다: 첫째, 나이지리아 영어(Nigerian English)와 미국 영어(American English)를 번역하는 음성 자막(phonetic subtitle) 모델을 개발하고, 둘째, 고급 독성 감지기(toxicity detectors)를 사용하여 나이지리아 영어 어휘의 독성(troximity)을 측정하는 것입니다.

- **Technical Details**: 이 연구는 고성능 자동음성인식(ASR) 시스템과 독성 감지를 결합하여 나이지리아 영어를 인식하고 번역합니다. ASR 시스템은 주로 미국 영어와 스페인어 같은 고자원 언어(high-resource languages)에서 높은 성능을 보이지만, 특유의 억양(accent)과 발음(pronunciation)으로 인해 나이지리아 영어(Nigerian English) 인식에는 어려움이 있습니다. 본 연구는 이를 해결하기 위해 음향 모델 적응(acoustic model adaptation)과 어휘 확장(lexicon extension) 같은 기법을 사용했습니다. 또한, 독성 감지를 통해 문화적 차이로 인한 언어의 독성을 평가했습니다.

- **Performance Highlights**: 연구 결과, 나이지리아 영어에 적응된 ASR 모델은 단어 오류율(word error rate)을 37% 절감하는 것으로 나타났습니다. 특히, NeMo QuartzNet15x5 모델이 나이지리아 억양을 다루는 데 있어 8.2%의 단어 오류율을 기록하며 뛰어난 성능을 보였습니다. 그러나 작은 데이터셋 크기와 모델의 과적합(overfitting) 문제는 한계로 지적됐습니다. 전반적으로 이번 연구는 나이지리아 영어 및 기타 저자원 언어에 대한 ASR 시스템의 성능을 크게 향상시키는 데 기여했습니다.



### RLHF Can Speak Many Languages: Unlocking Multilingual Preference Optimization for LLMs (https://arxiv.org/abs/2407.02552)
- **What's New**: 이번 연구에서는 다국어 대형 언어 모델(multilingual LLMs)의 정렬을 위한 새로운 상태-of-the-art를 달성하기 위해 포괄적인 연구를 수행했습니다. 고품질의 다국어 피드백 데이터를 생성하는 새로운 확장 가능한 방법을 도입했으며, 교차언어 전이(cross-lingual transfer)와 선호도 학습에서 데이터셋 크기 증가의 이점을 확립했습니다. 이 모델은 23개의 언어를 포함하여 세계 인구의 절반을 커버합니다.

- **Technical Details**: 다국어 선호도 최적화(preference optimization)에 대한 연구로서, Aya 23 8B 모델을 기반으로 새로운 방법을 적용했습니다. 데이터를 영어뿐만 아니라 다른 다국어로 확장함으로써 다양한 언어에 대한 성능 향상을 실현했습니다. 또한 오프라인(optimal)과 온라인(optimal) 선호도 최적화 기법을 비교하며, RLOO가 DPO보다 뛰어난 성능을 보였습니다.

- **Performance Highlights**: 연구 결과에 따르면, 선호도 최적화된 모델은 현재 최고 성능을 보이는 다국어 LLM인 Aya 23 8B 모델에 대해 54.4%의 승률을 달성했으며, Gemma-1.1-7B-it, Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.3 등과 같은 널리 사용되는 모델에 대해 69.5% 이상의 승률을 기록했습니다. 특히 RLOO 방식의 온라인 최적화가 다양한 언어에서 더 나은 성능을 보였습니다.



### Towards the Next Frontier in Speech Representation Learning Using Disentanglemen (https://arxiv.org/abs/2407.02543)
- **What's New**: 이번 연구에서는 음성 표현 학습을 위한 프레임워크로서 Learn2Diss를 제안하며, 이는 프레임 레벨(frame-level)과 발화 레벨(utterance-level)의 인코더 모듈을 포함합니다. 이 프레임워크는 고유한 음소 특징(pseudo-phonemic representations)을 학습하며, 발화 레벨 인코더는 대조 학습(contrastive learning)을 통해 가짜 화자 표현(pseudo-speaker representations)을 학습합니다. 이 두 모듈은 상호 정보(mutual information) 기준을 사용하여 분리(disentangling) 학습을 진행합니다.

- **Technical Details**: Learn2Diss 프레임워크는 두 단계의 인코더를 독립적으로 학습하고, 그 후에 상호 정보 최소화(Mutual Information Minimization) 목표를 통해 공동 학습을 진행합니다. 프레임 레벨 인코더는 기존의 자가 지도 학습(self-supervised learning) 기술에서 영감을 받아 고유한 음소 표현을 학습하며, 발화 레벨 인코더는 대조 학습 및 풀링된 임베딩(embedding)을 통해 가짜 화자 표현을 학습합니다. 두 인코더를 결합하는 과정에서 상호 정보 기준이 적용되며, CLUB(Contrastive Log-ratio Upper Bound) 방법을 통해 상호 정보를 상한으로 근사화합니다.

- **Performance Highlights**: 제안한 Learn2Diss는 다양한 다운스트림 평가 실험에서 최신 성능(state-of-the-art)을 달성하였으며, 프레임 레벨 인코더 표현은 의미론적 작업(semantic tasks)을 개선하고, 발화 레벨 표현은 비의미론적 작업(non-semantic tasks)을 개선하는 것이 확인되었습니다. SUPERB 및 ZeroSpeech 2021와 같은 벤치마크에서 우수한 성능을 보여주었으며, 저자원 음성 인식 실험에서도 좋은 성과를 보였습니다.



### InternLM-XComposer-2.5: A Versatile Large Vision Language Model Supporting Long-Contextual Input and Outpu (https://arxiv.org/abs/2407.03320)
Comments:
          Technical Report. this https URL

- **What's New**: InternLM-XComposer-2.5 (IXC-2.5)는 긴 컨텍스트 입력 및 출력을 지원하는 다용도 대형 비전 언어 모델입니다. 이 모델은 GPT-4V 수준의 성능을 단 70억 파라미터의 LLM 백엔드로 구현하였습니다. IXC-2.5는 텍스트-이미지 이해 및 작문 작업에서 뛰어난 성능을 발휘하며, 특히 고해상도 이미지 및 복합 비디오 이해, 다중 이미지 대화, 웹페이지 생성, 고품질 텍스트-이미지 기사 작성을 포함한 다양한 응용 프로그램을 지원합니다.

- **Technical Details**: IXC-2.5는 24K의 교차 이미지-텍스트 컨텍스트로 훈련되었으며 RoPE 외삽법을 통해 96K까지 컨텍스트 창을 확장할 수 있습니다. 이 모델은 비전 언어 이해를 위해 고유 해상도 이해, 세밀한 비디오 이해, 다중 이미지 대화를 주요 업그레이드 사항으로 특징짓습니다. 텍스트-이미지 작문을 위해 추가적인 LoRA 파라미터를 사용하며, 웹페이지 생성과 고품질 텍스트-이미지 기사 작성을 지원합니다.

- **Performance Highlights**: IXC-2.5는 28개의 벤치마크에서 검증되었으며, 16개의 벤치마크에서 기존의 오픈소스 최첨단 모델을 능가합니다. 이는 GPT-4V 및 Gemini Pro와의 16개 주요 작업에서도 비슷한 성능을 보여줍니다. IXC-2.5는 이제 오픈소스 도구를 사용하여 오디오 입력 및 출력을 지원합니다.



### BACON: Supercharge Your VLM with Bag-of-Concept Graph to Mitigate Hallucinations (https://arxiv.org/abs/2407.03314)
- **What's New**: 이번 논문은 제한된 언어적 능력을 가진 모델들도 Vision Language Models (VLMs)와 같은 특권을 누릴 수 있게 하여 데이터 감지, 시각적 질문 응답(VQA), 이미지 생성 등의 다운스트림 작업을 개선하는 Bag-of-Concept Graph (BACON)를 제안합니다. BACON은 어노테이션을 기본 최소 요소로 분해하여 그래프 구조로 제공하며, 이를 통해 이미지 내의 복잡한 관계를 보다 쉽게 이해할 수 있도록 합니다.

- **Technical Details**: BACON은 이미지 어노테이션을 객체, 스타일, 관계 등의 기본 요소로 분해하고 이를 그래프 구조로 결합하여 다양한 다운스트림 모델들이 쉽고 정확하게 이해할 수 있도록 합니다. 구체적으로, 이미지 그래프를 (D,O,R,B) 형태로 구성하며, 이는 이미지에 대한 텍스트 설명(D), 객체 목록(O), 객체들 간의 관계(R), 바운딩 박스 위치(B)를 포함합니다. 우리는 VLM과 분할 기법을 활용하여 10만 개의 어노테이션된 이미지 데이터셋을 구축했습니다.

- **Performance Highlights**: 논문은 BACON을 통해 다운스트림 작업의 성능이 크게 향상됨을 보여줍니다. BACON 캡션 기법을 사용하여 이미지 생성, VQA, 감지 작업 등에서 뛰어난 성능을 발휘했으며, 이전에 불가능했던 작업들을 성공적으로 수행했습니다. 특히, BACON-Captioner가 GPT-4V와 유사한 수준으로 BACON을 생성할 수 있으며, 수작업 어노테이션을 통해 높은 정확도(91%)와 재현율(90%)을 기록했습니다.



### How Similar Are Elected Politicians and Their Constituents? Quantitative Evidence From Online Social Network (https://arxiv.org/abs/2407.03255)
- **What's New**: 정치인과 유권자 간의 유사성을 분석하기 위해, 미국과 영국의 선거구 수준에서 2년 반(2020년 9월 - 2023년 2월) 동안 수집한 데이터를 비교합니다. 이는 증가하는 정치적 불만과 포퓰리즘에 비추어 볼 때 중요한 연구입니다.

- **Technical Details**: 데이터는 두 부분으로 나뉩니다: (i) 선출된 정치인의 트위터 타임라인(560만 개 트윗) 및 (ii) 유권자의 Nextdoor 포스트(2,180만 개 포스트)입니다. Nextdoor는 위치 기반 소셜 네트워크로, 사용자가 실제 거주지를 확인한 후 해당 지역에서만 상호작용할 수 있습니다. 이를 통해 각 선거구의 온라인 담론을 분석합니다.

- **Performance Highlights**: 주요 발견 사항으로는 정치인은 좌파든 우파든 간에 유사한 수준의 유사성을 보이며, 소득 수준과 선거 승리의 크기에 따라 그 유사성이 다를 수 있다는 점입니다. 예를 들어, 좁은 승리 선거구에서는 스타일이 더 유사하고 내용이 덜 유사하며, 더 낮은 소득의 선거구에서는 내용이 더 유사합니다.



### Self-Evaluation as a Defense Against Adversarial Attacks on LLMs (https://arxiv.org/abs/2407.03234)
Comments:
          8 pages, 7 figures

- **What's New**: 새로운 연구는 민감한 인간 대면 환경에서 배치되는 대규모 언어 모델(LLMs)이 안전하지 않거나 편향적이며 프라이버시를 침해하는 출력을 내지 않도록 훈련되고 지침을 받음에도 불구하고, 단순히 모델 입력 끝에 공백(space)을 추가하는 것만으로도 모델의 방어를 무너뜨릴 수 있음을 발견했습니다.

- **Technical Details**: 연구팀은 8개의 오픈 소스 모델을 대상으로 조사한 결과, 이 공격이 대부분의 모델에서 매우 높은 성공률로 유해한 출력을 생성하게 하는 강력한 공격으로 작용할 수 있음을 증명했습니다. 이 문제의 원인을 조사한 결과, 단일 공백이 토큰화된 훈련 데이터에서 발생하는 맥락이 모델로 하여금 리스트를 생성하도록 유도하여, 안전하지 않은 요청을 거부하도록 하는 훈련 신호를 무시하도록 만들었음을 발견했습니다.

- **Performance Highlights**: 이 연구는 현재 모델 정렬 상태의 취약성을 강조하며, 더 견고한 정렬 방법의 개발 중요성을 촉진합니다. 코드와 데이터는 제공될 예정입니다.



### Single Character Perturbations Break LLM Alignmen (https://arxiv.org/abs/2407.03232)
Comments:
          8 pages, 6 figures

- **What's New**: 최신 연구에 따르면, 대형 언어 모델(LLMs)이 민감한 인간 상호 작용 환경에서 배치될 때, 안전하지 않거나 편향된 또는 프라이버시를 침해하는 출력을 방지하는 것이 중요합니다. 그러나 연구진은 입력 끝에 공백(space) 하나를 추가하는 것만으로 모델의 방어를 무너뜨릴 수 있음을 발견했습니다. 8개의 오픈 소스 모델을 대상으로 한 연구에서, 대부분의 모델이 매우 높은 성공률로 해로운 출력을 생성하게 만드는 강력한 공격임을 입증했습니다. 이런 결과는 현재 모델 정렬(alignment)의 취약한 상태를 강조하며, 더 견고한 정렬 방법 개발의 중요성을 제기합니다.

- **Technical Details**: 연구진은 모델의 입력 끝에 공백(space)을 추가하는 간단한 공격으로 모델 방어를 무너뜨릴 수 있음을 발견했습니다. 이는 토큰화된 훈련 데이터에서 단일 공백이 발생하는 맥락이 모델에게 리스트를 생성하도록 장려하여, 안전하지 않은 요청을 거부하라는 훈련 신호를 무시하게 만듭니다. 이로 인해 대부분의 모델이 해로운 출력을 생성하기 시작했습니다.

- **Performance Highlights**: 공백(space) 공격은 대부분의 모델에서 높은 성공률로 해로운 출력을 생성하게 만들었습니다. 이는 현재 모델 정렬 상태가 취약함을 확인시키며, 더 견고한 정렬 및 방어 방법이 필요함을 보여줍니다. 연구진은 코드와 데이터를 공개할 예정입니다.



### CiteAssist: A System for Automated Preprint Citation and BibTeX Generation (https://arxiv.org/abs/2407.03192)
Comments:
          Published at SDProc @ ACL 2024

- **What's New**: 저자들은 신기술인 CiteAssist를 소개합니다. 이 시스템은 BibTeX 항목 생성을 자동화하여 논문의 서지 주석(bibliographic annotation)을 간소화합니다. CiteAssist는 메타데이터(저자 이름, 제목, 출판 날짜, 키워드 등)를 추출하여 문서 내에 표준화된 주석을 만듭니다.

- **Technical Details**: CiteAssist는 PDFs의 끝에 자동으로 BibTeX 인용(citation)을 첨부하고, 문서 첫 페이지에 링크를 달아 다른 연구자들이 올바른 인용 정보에 쉽게 접근할 수 있도록 합니다. 이 방법은 주석이 어떤 저장소(repository)를 사용하든 접근 가능하도록 보장함으로써 플랫폼의 유연성을 촉진합니다. 또한, 관련 연구 논문을 키워드에 기반하여 추가 논문을 제공하여 연구자들이 더 많은 자료를 읽을 수 있도록 합니다.

- **Performance Highlights**: 연구자들은 무료로 공개된 웹 인터페이스를 통해 자신의 프리프린트(preprint)와 레퍼런스 관리를 개선할 수 있습니다. 이 시스템은 관련 연구 논문을 신규 자료와 함께 제공함으로써 연구자의 작업 효율성을 크게 향상시킵니다.



### Cutting through the noise to motivate people: A comprehensive analysis of COVID-19 social media posts de/motivating vaccination (https://arxiv.org/abs/2407.03190)
Comments:
          51 pages, 13 figures, 12 tables. Accepted at Natural Language Processing Journal

- **What's New**: COVID-19 팬데믹은 의료 정보 시스템의 약점을 여실히 드러냈습니다. 본 연구는 2년에 걸쳐 수집된 대규모 데이터셋을 분석하여, COVID-19 백신에 대한 대중의 동기 부여 및 저해 요인을 확인하는 새로운 방향을 제시합니다. 시간, 지리적 위치, 정치적 성향을 기준으로 이러한 주제를 분석했으며, 대중을 동기부여하는 주제는 지속되는 반면, 저해하는 주제는 빨리 변하는 경향이 있는 것을 발견했습니다. 또한 외부의 명령보다는 내적인 동기가 대중을 동기부여하는 데 더 유리하다는 것을 확인했습니다.

- **Technical Details**: 본 연구는 2020년 1월부터 2021년 12월까지의 트위터 데이터셋을 사용했습니다. 주요 연구 질문은 COVID-19 백신에 대한 동기부여 및 저해 주제가 시간, 지리적 위치, 정치적 성향에 따라 어떻게 변하는지를 탐구하는 것이었습니다. 데이터셋에는 위치, 동기부여 상태, 백신에 대한 입장, 주제 라벨이 포함되어 있으며, COVID-19 백신에 대한 트윗을 동기부여 또는 저해하는 것으로 분류하는 머신러닝 모델과 사용자의 백신 입장을 식별하는 머신러닝 모델이 개발되었습니다.

- **Performance Highlights**: 본 연구는 COVID-19 백신과 관련된 동기부여 및 저해 주제를 식별할 수 있는 토픽 모델을 제시합니다. 연구 결과는 공중 보건 공무원, 정책 입안자, 소셜 미디어 플랫폼이 잘못된 정보를 차단하고 대중에게 더 효과적으로 과학적 정보를 제공하는 메시지 전략을 개발하는 데 도움이 될 수 있습니다.



### SOS! Soft Prompt Attack Against Open-Source Large Language Models (https://arxiv.org/abs/2407.03160)
- **What's New**: 오픈 소스 대형 언어 모델(LLMs)은 사용자 정의, 미세 조정, 자유롭게 사용 가능하다는 이유로 대중과 산업계에서 인기를 끌고 있다. 그러나 일부 오픈 소스 LLM은 사용 전에 승인이 필요하므로 타사가 접근하기 쉬운 버전을 발행하고 있다. 이러한 경향은 학습 시 공격의 위험성을 증가시켜 LLMs의 무결성과 보안을 손상시킬 수 있다. 이번 연구에서는 새로운 학습 시 공격 방법인 SOS를 소개한다. SOS는 컴퓨팅 요구 사항이 낮으며 깨끗한 데이터나 모델 가중치의 수정이 필요하지 않아서 모델의 유용성을 유지한다.

- **Technical Details**: SOS 공격은 다양한 시나리오에서 보안 문제를 처리한다. 예를 들어 백도어 공격, 탈옥 공격, 프롬프트 탈취 공격 등이 있다. SOS는 학습 시점에서 매우 낮은 계산 수요를 요구하면서도 모델 가중치를 수정하지 않기 때문에, 모델의 본래 기능을 유지하며 공격이 가능하다. 또한, 저작권 토큰이라는 새로운 기술을 도입하여 사용자가 저작권 있는 콘텐츠를 표시하고 모델이 이를 사용하는 것을 방지할 수 있다.

- **Performance Highlights**: 실험 결과 SOS 공격이 평가된 모든 타겟에서 효과적임을 보여준다. SOS 기술은 낮은 컴퓨팅 요구 사항과 모델 유용성 유지 측면에서 다른 공격 방법들에 비해 우위를 갖는다. 이는 학계와 산업계에서 중요한 보안 문제를 해결할 새로운 방법을 제시한다.



### Speaker- and Text-Independent Estimation of Articulatory Movements and Phoneme Alignments from Speech (https://arxiv.org/abs/2407.03132)
Comments:
          to be published in Interspeech 2024 proceedings

- **What's New**: 이번 논문에서는 기존에 별도로 취급되었던 두 가지 작업인 음향-조음 변환(AAI)과 음소-조음 움직임 예측(PTA)을 결합하는 새로운 방법을 소개합니다. 이 새로운 작업을 '음향 음소-조음 변환(APTAI)'라고 명명하고, 두 가지 접근 방식을 탐구하였습니다. 두 접근 방식 모두 추론 과정에서 화자 및 텍스트에 독립적으로 작동합니다.

- **Technical Details**: 두 가지 접근 방식 모두 다중 작업 학습(Multi-task Learning, MTL) 설정을 사용하여 조음 움직임 예측, 음소 시퀀스 및 음소 정렬을 추정하는 엔드-투-엔드 목표를 가지고 있습니다. 한 가지 접근 방식은 프레임 분류에 기반하고 있으며, 다른 한 가지는 강제 정렬을 포함한 이단계 훈련 절차를 포함합니다. Wav2vec2 모델을 사용하여 사전 훈련된 음성 표현을 활용하며, 두 가지 접근 방식 모두 자가 지도 학습(Self-Supervised Learning, SSL) 모델을 사용합니다.

- **Performance Highlights**: 제안된 AAI 작업에서 평균 상관계수 0.73을 달성하였으며, 최첨단 텍스트 종속 음소 정렬기에 비해 최대 약 87%의 프레임 중첩률을 기록하였습니다.



### KeyVideoLLM: Towards Large-scale Video Keyframe Selection (https://arxiv.org/abs/2407.03104)
- **What's New**: KeyVideoLLM은 비디오 대규모 언어 모델(VideoLLM) 데이터 관리를 위한 새로운 텍스트-비디오 프레임 유사성 기반 핵심 프레임(keyframe) 선택 방법입니다. 이는 비디오 데이터의 효율성, 견고성 및 효과성을 크게 향상시킵니다.

- **Technical Details**: KeyVideoLLM은 심층 학습 모델의 강력한 성능을 활용하여 주어진 질문과 응답에 가장 관련성이 높은 프레임을 선택합니다. 코스-투-파인(coarse-to-fine) 인프라를 사용해 빠른 선택 속도를 자랑하며, 추가적인 하이퍼파라메터 조정이 필요하지 않습니다. 이를 통해 효율적이고 견고한 데이터 관리를 실현하였습니다.

- **Performance Highlights**: ['고효율: KeyVideoLLM은 데이터 압축률이 최대 60.9배에 달하며, 디스크 사용량을 크게 줄입니다.', '고견고성: 기존 방법보다 최대 200배 빠른 선택 속도를 가지며, 높은 성공률을 자랑합니다.', '효과성: 훈련 및 추론 과정에서 비디오 질문-응답 성능을 크게 향상시키며, 다양한 데이터셋에서 기존 SoTA(State-of-the-Art) 성능을 달성했습니다.']



### JailbreakHunter: A Visual Analytics Approach for Jailbreak Prompts Discovery from Large-Scale Human-LLM Conversational Datasets (https://arxiv.org/abs/2407.03045)
Comments:
          18 pages, 9 figures

- **What's New**: 새로운 연구로 자일브레이크 프롬프트(jailbreak prompts)를 대규모 인간-LLM 회화 데이터셋에서 식별하는 시각 분석 도구인 JailbreakHunter를 소개합니다. 이는 LLM의 보안 프로토콜을 뚫기 위해 악의적으로 사용되는 프롬프트 식별을 위한 세 가지 분석 수준(그룹별, 대화별, 턴별)을 포함합니다.

- **Technical Details**: JailbreakHunter는 그룹 수준 분석, 대화 수준 분석, 턴 수준 분석의 세 가지 분석 레벨을 통해 대화 데이터에서 악의적인 프롬프트를 식별합니다. 그룹 수준 분석에서는 여러 기준(이전 연구에서 보고된 프롬프트와의 유사성, 공격 성공률 등)을 사용해 대화 분포를 파악합니다. 대화 수준 분석에서는 대화의 진행 상황을 이해하고 각 대화 맥락에서 자일브레이크 프롬프트를 발견합니다. 턴 수준 분석에서는 단일 턴 내 프롬프트와 보고된 자일브레이크 프롬프트 간 의미론적 유사성과 토큰 중복을 탐색합니다.

- **Performance Highlights**: 이 시스템의 효과와 사용성은 여러 사례 연구와 전문가 인터뷰를 통해 검증되었습니다. 연구 결과, 사용자는 대규모 인간-LLM 회화 데이터셋에서 자일브레이크 프롬프트를 효과적으로 식별할 수 있음을 확인했습니다.

- **Related Work**: 이 연구는 자일브레이크 프롬프트 분석, NLP를 위한 시각화, 회화 데이터셋 시각화 관련 기존 연구를 기반으로 합니다. 기존 연구들은 주로 공중 포럼에서 자일브레이크 프롬프트를 수집하거나 자동화된 알고리즘으로 패턴을 분석하는 방식에 의존했으나, 이 연구는 대규모 회화 데이터셋에서 프롬프트를 시각적으로 식별하는 데 중점을 둡니다.



### LoRA-Guard: Parameter-Efficient Guardrail Adaptation for Content Moderation of Large Language Models (https://arxiv.org/abs/2407.02987)
- **What's New**: LoRA-Guard는 대형 언어 모델(LLMs)의 콘텐츠 관리를 위해 리소스 제약이 있는 휴대용 장치에서도 사용할 수 있는 새로운 경량화된 적응 방법입니다. 이는 LLM과 가드레일(guardrail) 모델 간의 지식 공유를 통해 수행되며, 기존의 접근 방식보다 100-1000배 낮은 파라미터 오버헤드로도 높은 정확도를 유지합니다.

- **Technical Details**: LoRA-Guard는 LLM에서 언어 기능을 추출하고 저순위 어댑터(low-rank adapters)를 사용하여 콘텐츠 관리 작업을 위해 이를 적용합니다. 이 과정에서 듀얼 경로 설계를 사용해 생성 작업 성능 저하를 방지합니다. LoRA-Guard는 LLM의 백본 트랜스포머에 LoRA 어댑터를 추가하여 유해 콘텐츠를 탐지하는 작업을 학습합니다. 가드 모델의 파라미터 오버헤드는 LoRA 어댑터와 가드 출력 헤드에 국한됩니다.

- **Performance Highlights**: LoRA-Guard는 기존 접근 방식에 비해 유사한 성능을 제공하면서도 파라미터 오버헤드를 대폭 줄였습니다. 이를 통해 리소스 제약이 있는 환경에서도 효과적으로 배포할 수 있습니다. 특히, 듀얼 경로 설계 방식을 통해 생성 작업의 성능 저하 없이 높은 콘텐츠 관리 정확도를 유지합니다.



### ObfuscaTune: Obfuscated Offsite Fine-tuning and Inference of Proprietary LLMs on Private Datasets (https://arxiv.org/abs/2407.02960)
Comments:
          Preprint

- **What's New**: 이 논문은 모델 제공자(entity)와 데이터 소유자(entity)의 비공개 데이터에 대해 LLM의 추론 및 미세조정을 수행하면서 양쪽의 기밀성을 보장하는 문제를 해결합니다. 이 작업은 제3자의 클라우드 인프라에서 수행됩니다. 우리는 ObfuscaTune이라는 새로운 접근법을 제안하여 간단하지만 효과적인 난독화 기술과 효율적인 기밀 컴퓨팅(Total Efficient Execution, TEE)의 결합을 통해 이 문제를 해결합니다.

- **Technical Details**: ObfuscaTune은 모델 제공자의 LLM을 데이터 소유자의 비공개 데이터에 대해 추론 및 미세조정하는 방법입니다. 제3자의 클라우드 인프라에서 작업이 수행되며, 모델과 데이터의 기밀성을 보장하는 것이 목표입니다. 이는 랜덤 행렬을 사용한 난독화 기법과 모델 매개변수의 5%만 TEE에 배치하는 방식으로 이루어집니다. TEE는 외부 접근이 불가능한 격리된 안전 구역으로, 인증을 통해 데이터 소유자만 모델에 쿼리할 수 있습니다.

- **Performance Highlights**: ObfuscaTune은 다양한 크기의 GPT-2 모델을 사용한 실험에서 그 효과를 실증했습니다. 네 가지 NLP 벤치마크 데이터셋을 통해 검증했으며, 단순 난독화 기법과 비교하여 오류를 감소시키기 위해 조건수가 낮은 랜덤 행렬을 사용하는 것이 필요합니다. 또한, 성능 손실 없이 모델과 데이터의 기밀성을 유지하면서 효율성 손실을 최소화할 수 있음을 보여줍니다.



### IncogniText: Privacy-enhancing Conditional Text Anonymization via LLM-based Private Attribute Randomization (https://arxiv.org/abs/2407.02956)
Comments:
          Preprint

- **What's New**: 이번 연구에서는 텍스트 익명성을 위한 새로운 방식인 IncogniText를 제안합니다. 이 기술은 저자의 민감한 속성을 잠재적 공격자가 정확히 추론하지 못하도록 하면서 텍스트의 유용성(의미와 의미론)을 유지합니다. 실험 결과, 민감한 속성 유출을 90% 이상 줄이는 것으로 나타났습니다. 또한 IncogniText는 실제 응용 프로그램에서 사용할 수 있도록 소형 온-디바이스 모델에 LoRA 파라미터를 결합하여 익명화 기능을 증류하는 데 성공했습니다.

- **Technical Details**: IncogniText는 특정 속성(예: 나이)을 보호하기 위해 원본 텍스트를 재작성하여 잠재적 공격자가 잘못된 속성 값을 예측하도록 유도합니다. 이를 위해 익명화 모델(M$_{anon}$)을 사용하여 원본 텍스트(x$_{orig}$)를 목표 속성 값(a$_{target}$), 실제 속성 값(a$_{true}$) 및 익명화 템플릿(T$_{anon}$)과 결합하여 익명화된 텍스트(x$_{anon}$)를 생성합니다. 목표 속성 값은 사용자에 의해 선택되거나 사전 정의된 값 집합에서 무작위로 샘플링될 수 있습니다. 실험을 통해 실제 속성 값을 사용하지 않더라도 IncogniText의 익명화가 매우 효과적임을 입증했습니다.

- **Performance Highlights**: IncogniText는 두 개의 데이터셋 및 다양한 대규모 언어 모델(LLM)을 사용한 실험에서 두 개의 동시 작업(Dou et al., 2023; Staab et al., 2024)보다 더 높은 프라이버시 보호 성능을 보였습니다. 실제 응용 프로그램에서 사용 가능하도록 소형 온-디바이스 모델에 LoRA 파라미터를 결합하여 익명화 기능을 증류했습니다.



### PII-Compass: Guiding LLM training data extraction prompts towards the target PII via grounding (https://arxiv.org/abs/2407.02943)
Comments:
          Accepted at ACL 2024

- **What's New**: 이번 논문에서는 개인정보 식별 정보(PII)를 추출할 수 있는 새로운 방법인 PII-Compass를 제안했습니다. 이 접근법은 수작업으로 구성된 추출 프롬프트의 접두사를 도메인 데이터로 연결해 PII 추출 가능성을 크게 향상시킵니다.

- **Technical Details**: PII-Compass는 기존의 단순 템플릿 기반 프롬프트와는 달리, 목표 데이터와 가까운 임베딩을 가진 프롬프트를 사용하여 PII를 추출합니다. 특히, 타겟 데이터 주체의 실제 접두사를 포함하는 프롬프트를 사용함으로써 이러한 접두사와 유사한 임베딩 공간에서 모델을 쿼리합니다. 이는 생성된 프롬프트가 모델에 보다 잘 연결되어 PII 추출 효과를 크게 높이는 결과를 초래합니다.

- **Performance Highlights**: 실험 결과, 이 접근법을 통해 전화번호 추출률이 1, 128, 2308개의 쿼리에서 각각 0.92%, 3.9%, 6.86%로 증가했습니다. 이는 기존 단순 수작업 템플릿 기반 프롬프트를 통한 추출율이 1% 미만인 것에 비해 큰 향상을 보입니다.



### GraCoRe: Benchmarking Graph Comprehension and Complex Reasoning in Large Language Models (https://arxiv.org/abs/2407.02936)
- **What's New**: 이번 논문에서는 대규모 언어 모델(LLMs)의 그래프 이해 및 추론 능력을 체계적으로 평가하기 위해 새로운 벤치마크 GraCoRe를 제안합니다. GraCoRe는 순수 그래프와 이질 그래프에 대해 세분화된 10개의 영역을 포함하여 19개의 과제를 통해 모델을 평가합니다. 총 11개의 데이터셋과 5,140개의 다양한 복잡도의 그래프 데이터를 포함하고 있으며, 3개의 비공개 소스 및 7개의 공개 소스 LLMs를 평가했습니다.

- **Technical Details**: GraCoRe는 세 단계의 계층적 분류법을 사용하여 LLMs의 기능을 평가합니다. 첫 번째 단계에서는 그래프 이해와 그래프 추론이라는 두 가지 주요 기능을 정의하고, 두 번째 단계에서는 네 가지 데이터 유형에 따라 LLMs의 기능을 분류합니다. 세 번째 단계에서는 각 기능을 10개의 세부 기능으로 나누고, 19개의 과제를 통해 이를 테스트합니다. 각 과제는 사전에 정의된 규칙에 따라 구체적인 프롬프트(prompt)를 통해 구성되며, 구조적 정보를 텍스트화합니다.

- **Performance Highlights**: LLMs의 그래프 이해 및 추론 능력을 평가한 결과, 다음과 같은 주요 발견이 있었습니다: (1) 현재 LLMs는 그래프 추론에서 큰 약점을 보여주며, 대부분의 모델은 그래프 이해와 추론을 모두 잘 수행하지 못합니다. GPT-4o는 두 가지 측면에서 가장 포괄적인 성능을 보였습니다. (2) 의미적 정보가 풍부한 그래프 추론 과제에서는 더 높은 성능을 보였지만, 순수한 구조적 그래프 추론에서는 상대적으로 저조한 성능을 보였습니다. (3) 텍스트 그래프 데이터의 노드 순서가 중요한 영향을 미치며, 정렬된 노드 명칭이 과제 성공에 긍정적인 영향을 미쳤습니다. (4) 모델이 더 긴 텍스트 입력을 처리할 수 있는 능력은 그래프 이해 및 추론 과제의 성능에 영향을 미치지 않았습니다.



### GPTQT: Quantize Large Language Models Twice to Push the Efficiency (https://arxiv.org/abs/2407.02891)
Comments:
          Accepted by 11th IEEE International Conference on Cybernetics and Intelligent Systems

- **What's New**: 이 논문은 GPTQT라는 새로운 후-학습 양자화(post-training quantization) 방법을 소개한다. 이 방법은 LLM의 가중치를 3bit/2bit로 표현하여 메모리 사용량을 줄이고 처리 속도를 향상시킨다. 이는 2단계의 점진적인 접근 방식을 사용하여 가중치를 처음에는 상대적으로 높은 비트로 선형 양자화하고, 다음으로 얻어진 정수 가중치를 낮은 비트의 이진 코딩으로 변환하는 방식을 채택한다.

- **Technical Details**: GPTQT는 LLM의 가중치를 두 단계로 양자화한다. 첫 번째 단계에서는 가중치를 높은 비트 해상도로 선형 양자화한 후, 이를 낮은 비트의 이진 코딩 가중치로 변환한다. 이 과정에서는 초기 스케일링 요인을 최적화하기 위한 재탐색 전략을 사용한다. 추론(inference) 중에는 이 두 단계를 순수 이진 코딩으로 병합하여 효율적인 계산을 가능하게 한다.

- **Performance Highlights**: 다양한 모델과 데이터셋을 테스트한 결과, GPTQT는 3-bit 양자화 기준선과 비교했을 때, opt-66B에서 당혹감(perplexity)을 4.01만큼 줄였고, opt-30b에서 속도를 1.24배 향상시켰다. 또한, Llama2에서는 GPTQT가 현재까지 가장 우수한 이진 코딩 양자화 방법임을 보여주었다.



### CogErgLLM: Exploring Large Language Model Systems Design Perspective Using Cognitive Ergonomics (https://arxiv.org/abs/2407.02885)
Comments:
          8 Page, 3 Figures. Accepted to Large Language Models and Cognition @ ICML 2024 (this https URL)

- **What's New**: Cognitive ergonomics와 LLMs(대규모 언어 모델)의 통합은 인간-AI 상호작용의 안전성, 신뢰성, 사용자 만족도를 향상시키는 데 필수적입니다. 현재 LLM 설계는 이러한 통합이 부족하여 인간의 인지 능력 및 한계와 완전히 일치하지 않는 시스템을 만들 가능성이 있습니다. 본 논문은 이러한 문제를 해결하기 위해 인지 에르고노믹스 원칙을 LLM 설계에 통합하는 포괄적인 프레임워크와 실용적인 가이드를 제안합니다.

- **Technical Details**: 인지 에르고노믹스(cognitive ergonomics)는 인간의 기억, 주의력, 정신적 부담 및 의사결정과 같은 정신적 과정을 최적화하여 효율성, 안전성 및 사용자 만족도를 향상시키는 것을 목표로 합니다. 이러한 원칙을 LLM 설계에 적용하여 상호작용을 더 직관적이고 투명하게 만들며, 신뢰를 구축합니다. 본 논문에서는 사용자 중심 설계, 인지 부하 관리, 신뢰와 투명성, 피드백 메커니즘 등의 주요 요소를 포함한 설계 프레임워크 'CogErgLLM'을 소개합니다.

- **Performance Highlights**: 프레임워크의 실행 단계에서는 사용자 프로파일링, 개인화, 센서 통합 등을 통해 사용자의 인지적 요구를 충족하도록 LLM 시스템을 설계합니다. 예를 들어, 사용자 인터뷰와 설문조사를 통해 심층적인 사용자 인사이트를 얻고, 이를 바탕으로 개인화된 응답과 콘텐츠를 제공하여 사용자 경험을 향상시킵니다. 또한, 센서를 통해 사용자의 신체적 불편이나 인지적 피로를 감지하여 LLM 상호작용을 최적화합니다.



### CoIR: A Comprehensive Benchmark for Code Information Retrieval Models (https://arxiv.org/abs/2407.02883)
- **What's New**: 이번 논문에서는 코드 검색 능력을 평가하기 위해 고도로 설계된 포괄적인 벤치마크인 **코드 정보 검색 벤치마크**(Code Information Retrieval Benchmark, CoIR)를 제안합니다. CoIR는 다양한 도메인과 작업을 아우르는 **열 개**의 세심하게 선정된 코드 데이터셋을 포함하고 있으며, **여덟 개**의 고유 검색 작업을 **일곱 개**의 서로 다른 도메인에서 제공합니다.

- **Technical Details**: CoIR에는 텍스트-코드 검색, 코드-코드 검색, 코드-텍스트 검색 및 하이브리드 코드 검색과 같은 **네 가지** 주요 검색 작업이 포함됩니다. 각 작업은 더욱 세분화되어 코드 대회 검색, 웹 쿼리 코드 검색, 텍스트-SQL 검색, 코드 요약 검색 등 **여덟 가지** 세부 작업으로 나뉩니다. 이를 통해 CoIR는 1K에서 1M 규모의 다양한 데이터셋을 제공하며, 질의와 코퍼스의 평균 토큰 수는 각각 37에서 4.4K, 113에서 1.5K까지 다양합니다.

- **Performance Highlights**: CoIR를 사용하여 9개의 널리 사용되는 검색 모델을 평가한 결과, 최첨단 시스템에서도 코드 검색 작업에서 상당한 어려움을 겪는다는 것을 확인하였습니다. 이를 해결하기 위해 CoIR 벤치마크 프레임워크는 pip을 통해 쉽게 설치 가능하며, MTEB와 BEIR와 동일한 데이터 스키마를 공유하여 다양한 벤치마크 평가를 통해 연구자들이 쉽게 통합할 수 있도록 설계되었습니다.



### Contrast then Memorize: Semantic Neighbor Retrieval-Enhanced Inductive Multimodal Knowledge Graph Completion (https://arxiv.org/abs/2407.02867)
Comments:
          Accepted by SIGIR 2024

- **What's New**: 이번 연구에서는 Multimodal Knowledge Graph Completion (MKGC)를 넘어, 훈련 중에 보지 않은 새로운 엔티티를 포함하는 inductive MKGC (IMKGC) 작업을 탐구한 첫 번째 시도입니다. 기존의 inductive 접근 방식은 주로 텍스트 엔티티 표현에 중점을 두고 있기 때문에 시각적 모달리티(visual modality) 내의 풍부한 의미 정보를 무시하는 경우가 많습니다. 이에 반해, 새로운 엔티티에 대해 선행 이웃 검사(semantic neighbor retrieval-enhanced)로 강화된 CMR (Contrast, Memorize, Retrieve) 프레임워크를 제안합니다.

- **Technical Details**: 우리의 접근 방식인 CMR 프레임워크는 대조 학습(contrastive learning), 지식 표현 메모리(memorizations of knowledge representations), 및 검색 기반 추론(retrieval-augmented inference)을 포함합니다. 첫째, 통합된 cross-modal contrastive learning을 통해 query-entity 쌍의 textual-visual 및 textual-textual 상관관계를 통합 표현 공간에서 포착합니다. 이는 실제로 도움되는 의미 이웃들의 표현을 가깝게 만듭니다. 그 후에, 지식 저장소에 query 임베딩을 명시적으로 저장해 둡니다. 마지막으로, 시험 시간에는 지식 저장소에서 가장 가까운 의미 이웃들을 검색하여 query-entity의 유사성 분포에 보간(interpolate)합니다.

- **Performance Highlights**: 세 가지 inductive MKGC 데이터셋과 세 가지 전이 MKGC 데이터셋에서 광범위한 실험을 통해 CMR의 효과성과 일반성을 검증했습니다. 추가적으로, 제안된 통합 cross-modal contrastive learning과 의미 이웃 검색이 새 엔티티로의 일반화에 도움을 줄 수 있음을 확인했습니다. 이러한 결과는 CMR이 inductive 및 transductive 학습 모두에서 성능과 확장성 면에서 뛰어남을 입증합니다.



### Safe Unlearning: A Surprisingly Effective and Generalizable Solution to Defend Against Jailbreak Attacks (https://arxiv.org/abs/2407.02855)
Comments:
          15 pages

- **What's New**: 최근 연구에 따르면, 대형 언어 모델(LLMs)은 안전 조정 후에도 여전히 jailbreak 공격에 취약하다는 사실이 밝혀졌습니다. 본 연구는 이런 공격에 대응하기 위해 해롭고 위험한 지식을 직접 제거하는 접근 방식이 기존의 감독 학습 기반 접근 방식보다 더 효과적이라고 주장합니다.

- **Technical Details**: 연구진은 '유해 지식(unlearning harmful knowledge)'을 모델에서 직접 제거하는 방법을 사용했습니다. 이를 통해 Vicuna-7B 모델의 공격 성공률(Attack Success Rate, ASR)을 82.6%에서 7.7%까지 대폭 감소시켰습니다. 특히 주목할 점은, 훈련 시 단 20개의 유해 질문만 사용했으며, 별도의 jailbreak 프롬프트(jailbreak prompt)를 사용하지 않았다는 점입니다. 반면, Llama2-7B-Chat 모델은 약 10만 개의 안전 정렬 샘플을 통해 미세 조정되었지만, 추가적인 안전 시스템 프롬프트(safety system prompt) 도움에도 불구하고 여전히 21.9%의 ASR을 기록했습니다.

- **Performance Highlights**: 본 연구의 결과는 유해한 질문들 간의 본질적인 연관성(답변 패턴, 공통된 단계 및 행동, 그리고 LLM 내에서 학습된 표현의 유사성)이 unlearning 접근 방법의 일반화 능력에 기여한다는 사실을 보여줍니다. 이를 통해 Vicuna-7B 모델에서 뛰어난 성능을 달성한 것을 확인했습니다.



### MindBench: A Comprehensive Benchmark for Mind Map Structure Recognition and Analysis (https://arxiv.org/abs/2407.02842)
Comments:
          technical report

- **What's New**: 문서 분석 분야에서 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLM)은 중요한 발전을 이루었지만, 기존의 벤치마크들은 주로 텍스트 추출과 간단한 레이아웃 정보에만 집중하는 경향이 있습니다. 이로 인해 마인드맵, 플로우차트 등 구조화된 문서의 복잡한 상호작용을 충분히 평가하지 못합니다. 이를 해결하기 위해 새로운 벤치마크인 MindBench를 소개합니다. 이 벤치마크는 고해상도 양방향 이미지, 상세한 주석(annotation), 평가 메트릭, 베이스라인 모델을 포함하며, 구조화된 이해 및 파싱 작업을 위해 특별히 설계되었습니다.

- **Technical Details**: MindBench는 실제 및 합성된 마인드맵의 고해상도 이미지를 포함하는 양방향 데이터셋을 구축했습니다. 다섯 가지 구조화된 이해 및 파싱 작업은 전체 파싱(full parsing), 부분 파싱(partial parsing), 위치 관련 파싱(position-related parsing), 구조화된 시각적 질문 응답(Structured Visual Question Answering, VQA), 위치 관련 VQA입니다. 이러한 작업은 텍스트 인식, 공간 인식, 관계 추론 및 구조적 파싱을 포괄합니다. 모델 성능을 평가하기 위해 필드 레벨 F1 점수(field-level F1 scores)와 TED(Tree Edit Distance) 기반 정확도, VQA 작업 위한 F1 점수 등의 평가 메트릭을 설정했습니다.

- **Performance Highlights**: 광범위한 실험 결과는 현존하는 모델들이 복잡한 구조화된 문서를 처리하는 데 아직 상당한 개선 여지가 있음을 보여줍니다. 특히 고해상도 복잡한 그래픽 이미지 처리와 긴 구조화된 문서 정보 처리가 주요 과제임을 확인했습니다. MindBench의 도입으로 구조화된 문서 분석 기술 연구와 응용 개발이 크게 발전할 것으로 기대됩니다.



### LANE: Logic Alignment of Non-tuning Large Language Models and Online Recommendation Systems for Explainable Reason Generation (https://arxiv.org/abs/2407.02833)
- **What's New**: 추천 시스템의 설명 가능성은 사용자 신뢰와 만족도를 높이는 데 매우 중요합니다. 대형 언어 모델(LLM)을 활용하면 포괄적인 추천 로직을 생성할 수 있는 새로운 기회를 제공합니다. 하지만 기존 연구에서는 LLM 모델을 추천 작업에 맞추기 위한 파인튜닝(fine-tuning) 과정이 높은 계산 비용과 기존 시스템과의 정렬 문제를 초래하여 실용적인 적용성을 제한합니다. 이번 연구에서는 LLM을 추가적으로 튜닝하지 않고도 온라인 추천 시스템과 정렬하는 효과적인 전략인 LANE을 제안합니다. 이를 통해 비용을 줄이고 설명 가능성을 개선할 수 있습니다.

- **Technical Details**: 우리의 방법은 다음과 같은 주요 구성 요소를 통해 작동합니다: 의미 임베딩(semantic embedding), 제로샷 프롬프트(zero-shot prompting)를 이용한 사용자 다중 선호도 추출(user multi-preference extraction), 의미 정렬(semantic alignment), 체인 오브 사상(CoT) 프롬프트를 사용한 설명 가능한 추천 생성입니다. 항목 ID가 아닌 제목을 임베딩하고 멀티-헤드 어텐션 매커니즘을 사용하여 사용자 선호도의 의미적 특징을 후보 항목의 특징과 정렬합니다.

- **Performance Highlights**: 충분한 실험 결과(성능 비교, 설문 투표, 시각화 사례 포함)는 우리의 방법이 추천 성능을 보장하면서도 이해하기 쉽고 타당한 추천 로직을 제공할 수 있음을 입증합니다. 우리의 프레임워크는 모델 불가지론적(model-agnostic)이며, 고급 LLM을 사용하여 기존의 닫힌 소스 모델이나 제한된 계산 자원을 사용하지 않고도 설명 가능성을 극대화합니다.



### Images Speak Louder than Words: Understanding and Mitigating Bias in Vision-Language Model from a Causal Mediation Perspectiv (https://arxiv.org/abs/2407.02814)
- **What's New**: 이번 연구는 대규모 데이터셋으로 사전학습된 비전-언어 모델들(Vision-language models, VLMs)이 특정 객체나 시나리오에서 성별 정보를 연관시키며 편향을 학습하는 문제에 대한 새로운 프레임워크를 제안합니다. 주요 기여로는 편향 생성과 전파 경로를 측정하고 매핑하기 위해 인과매개분석(causal mediation analysis)을 도입한 점입니다. 이로 인해 모델 구성 요소에서의 직접적 및 간접적 효과를 식별할 수 있습니다.

- **Technical Details**: 본 연구에서는 GLIP 모델을 사례 연구로 사용하여 객체 탐지(Object detection) 작업에서 성별 편향을 분석합니다. MS-COCO와 PASCAL-SENTENCE 데이터셋을 기반으로 분석을 수행하며, 이미지 인코더(Image encoder)의 편향 기여도가 가장 높음을 확인했습니다. 이미지 특징은 편향의 32.57%와 12.63%를 차지하는 반면, 텍스트 특징은 각각 15.48%와 5.64%를 차지합니다. 또한, 텍스트와 이미지 간의 상호작용과 딥 퓨전 과정에서도 편향 생성에 큰 영향을 미친다는 점을 발견했습니다.

- **Performance Highlights**: 이미지 인코더 내에서 성별 표현을 흐리게 하는 방식으로 중점을 둔 결과, MSCOCO 데이터셋에서는 22.03%, PASCAL-SENTENCE 데이터셋에서는 9.04%의 편향 감소를 달성했습니다. 이는 매우 적은 성능 손실과 추가적인 계산 요구 없이 이루어졌습니다.



### Automatic gradient descent with generalized Newton's method (https://arxiv.org/abs/2407.02772)
- **What's New**: 이번 연구는 Hessian 정보에 기반한 일반적인 뉴턴 방법(GeN: generalized Newton's method)을 제안합니다. GeN은 SGD 및 Adam과 같은 어떤 옵티마이저에도 적용할 수 있으며, 뉴턴-랩슨 방법(Newton-Raphson method)을 부분적으로 포함합니다. 이 방법은 학습 속도를 가속화하고 학습률 스케줄러를 조정할 필요 없이 자동으로 학습률을 선택합니다. 이 방법은 거의 추가적인 계산 자원 소모 없이 쉽게 구현할 수 있습니다. 연구팀은 언어 및 비전 과제(GPT 및 ResNet 등)에서 GeN의 성능을 입증하였으며, 코드도 공개될 예정입니다.

- **Technical Details**: 딥러닝 모델은 종종 그라디언트 디센트(gradient descent) 방법을 사용하여 훈련됩니다. 전통적인 학습 방법은 많은 모델 매개변수, 학습률 스케줄러(learning rate scheduler), 프리컨디셔너(pre-conditioner) 및 1차 그라디언트를 필요로 했습니다. 그러나 이 방식은 큰 스케일의 모델에서 계산 비용이 매우 큽니다. 제안된 GeN 방법은 Hessian 정보를 포함하여 학습률을 동적으로 조정함으로써 이러한 문제를 해결합니다. 이로 인해 매우 큰 모델에서도 효율적인 학습이 가능합니다.

- **Performance Highlights**: 연구팀은 실험을 통해 GeN 최적화기가 학습률 스케줄러를 조정하는 데 신중하게 설계된 최신 성능과 일치한다는 것을 보여주었습니다. 특히 언어 모델 및 비전 모델(GPT, ResNet 등)에서 그 효과가 입증되었으며, 이는 수많은 반복에 걸쳐 계산 자원 소모가 거의 없다는 점에서 큰 의의가 있습니다.



### A Comparative Study of DSL Code Generation: Fine-Tuning vs. Optimized Retrieval Augmentation (https://arxiv.org/abs/2407.02742)
Comments:
          8 pages, 1 figure

- **What's New**: 최근 대형 언어 모델(LLMs)을 통해 자연어에서 코드 생성에 관한 큰 진전이 있었습니다. 하지만 도메인 특화 언어(DSL)의 사용자 정의 함수 이름 때문에 LLMs가 어려움을 겪고 있습니다. 이 논문에서는 LLM을 위한 학습 데이터셋를 생성하고, DSL 생성에 최적화된 검색 강화 생성(RAG) 기법을 소개합니다. 이 접근법은 현재 사용 중인 대규모 API 셋트를 다루면서, 새로운 업데이트에 적응할 수 있도록 설계되었습니다.

- **Technical Details**: 우리는 OpenAI 모델을 활용하여 RAG 기법을 최적화했습니다. 이를 통해 API 함수 정의를 포함한 추가적인 문맥적 기초 정보를 제공하여 메타프롬프트(metaprompt) 튜닝을 수행했습니다. 이 모델은 약 700개의 공개 도메인 API를 대표하는 자동화 작업을 위한 데이터셋을 사용하여 Codex 모델을 미세 조정했습니다.

- **Performance Highlights**: 미세 조정된 모델은 코드 유사성 측정에서 최고 점수를, RAG 최적화 모델은 유사성 측정에서 거의 동등한 성과를 보여주었습니다. 하지만, 두 모델 모두 구문 오류를 다수 발생시켰으며, RAG 방법은 컴파일 비율에서 2 포인트 높은 성과를 보였습니다. 반면, RAG 모델은 API 이름과 API 매개변수 키에서 각각 1 포인트, 2 포인트 낮은 환상 비율(Hallucination Rate)을 기록하였습니다.



### LLM-Select: Feature Selection with Large Language Models (https://arxiv.org/abs/2407.02694)
Comments:
          Preprint

- **What's New**: 이 논문에서는 대형 언어 모델 (LLMs)이 입력 피처 이름과 예측 과제 설명만으로 가장 예측력이 높은 피처를 선택할 수 있는 놀라운 능력을 가진다는 사실을 보여줍니다. 특히 GPT-4와 같은 최신 모델이 다양한 쿼리 방식에서 일관되게 가장 예측력이 높은 피처를 식별할 수 있음을 광범위한 실험을 통해 입증했습니다.

- **Technical Details**: LLMs는 미리 훈련된 대용량 텍스트 코퍼스에서 다음 단어 예측을 기반으로 다양한 작업에 일반화할 수 있는 능력을 갖추고 있습니다. 우리의 연구는 LLM을 프로밍(prompting)하여 특정 예측 과제를 수행하게 함으로써, 피처 선택과 같은 완전히 새로운 작업에서도 강력한 성능을 발휘할 수 있음을 확인했습니다. 특히 zero-shot prompting를 통해 피처 중요도 점수를 생성하는 등의 방법을 사용했습니다.

- **Performance Highlights**: LLM 기반 피처 선택은 전통적인 데이터 기반 방법인 LASSO와 경쟁할 수 있을 만큼 높은 성능을 보여줍니다. 특히 GPT-4를 사용한 zero-shot prompting에서도 강력한 성능을 발휘했으며, 다양한 유도(strategy)를 통해 피처 중요도 점수를 생성하는 방법이 실제 데이터에서도 높은 상관관계를 보였습니다. 이는 데이터 수집 후에만 피처를 선택하는 것이 아니라, 처음부터 어떤 피처를 수집할지 결정하는 데에도 유용할 수 있다는 가능성을 시사합니다.



### Reasoning in Large Language Models: A Geometric Perspectiv (https://arxiv.org/abs/2407.02678)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)의 추론 능력을 개선하기 위해 기하학적 이해를 탐구합니다. LLMs의 표현력과 자가-어텐션 그래프(self-attention graph)의 밀도 간의 관계를 규명하고, 이 밀도가 MLP 블록 입력의 본질적 차원을 정의함을 이론적 분석과 예제를 통해 입증합니다.

- **Technical Details**: 주요 연구는 트랜스포머 층(transformer layer)의 기하학적 특성을 분석하여 LLM의 성능 향상과 추론 능력 간의 연관성을 밝혀내는 것입니다. 특히, 자가-어텐션 모듈의 토큰 간 상호작용 밀도와 MLP 층의 함수 표현 복잡성 간의 관계를 강조합니다. 모델 크기와 문맥 길이(context length)가 증가하면 어텐션 밀도가 높아지고, 이를 통해 더 나은 추론 성능을 얻게 됩니다.

- **Performance Highlights**: 실험 결과에 따르면, 프롬프트(prompt)에 주어진 예시의 숫자가 증가할수록 LLM의 본질적 차원이 상승합니다. 특히, 마지막 층에서의 본질적 차원 상승은 추론 성능 향상과 강한 상관관계를 보입니다. 이는 LLM 내부 표현의 기하학적 특성이 효과적인 추론 능력에 중요한 역할을 한다는 것을 시사합니다.



### Supporters and Skeptics: LLM-based Analysis of Engagement with Mental Health (Mis)Information Content on Video-sharing Platforms (https://arxiv.org/abs/2407.02662)
Comments:
          12 pages, in submission to ICWSM

- **What's New**: 새로운 연구에서는 미국 성인 중 5명 중 1명이 정신 질환을 겪고 있으며, 정신 건강 전문가 부족과 오프라인 자원의 부족 속에서 온라인 짧은 비디오 콘텐츠가 중요한 역할을 하고 있다고 언급했습니다. 특히 YouTube Shorts와 Bitchute를 대상으로 한 최초의 정량적 연구를 통해 정신 건강 정보의 확산과 그에 대한 사용자의 참여를 분석했습니다. 연구팀은 739개의 비디오와 135,372개의 댓글을 포함한 MentalMisinfo라는 새로운 데이터셋을 제공했습니다.

- **Technical Details**: 연구팀은 전문가 주도의 어노테이션 스키마(annotation schema)를 사용하여 정신 건강 오정보(MHMisinfo)를 라벨링 했습니다. 또한, 대형 언어 모델(LLMs)을 사용한 few-shot in-context 학습이 MHMisinfo 비디오를 탐지하는 데 효과적임을 발견했습니다. YouTube와 Bitchute에서 사용자의 언어 패턴을 분석하여 특정 그룹이 MHMisinfo에 더 취약하다는 것을 발견했습니다.

- **Performance Highlights**: 연구 결과, MHMisinfo 비디오와 일반 비디오 간에 사용자 참여의 언어적 패턴이 다르다는 것을 확인했습니다. 특히, 종교적이거나 남성 중심의 언어를 사용하는 그룹이 더 취약한 것으로 나타났습니다. 또한, MHMisinfo 비디오에 대한 댓글이 기존의 낙인을 악화시킬 가능성이 높으며, MHMisinfo 콘텐츠에 대한 높은 동의율을 보였습니다. 이를 통해 연구팀은 정신 건강 오정보에 대처하기 위한 기술적, 공중 보건적 솔루션을 제안했습니다.



### A Practical Review of Mechanistic Interpretability for Transformer-Based Language Models (https://arxiv.org/abs/2407.02646)
Comments:
          11 pages, 11 figures, Preprint

- **What's New**: 기계적 해석성(Mechanistic interpretability, MI)은 신경망 모델의 내부 연산을 역공학해 이해하려는 새로운 접근법으로, 최근 Transformer 기반 언어 모델(LM)의 해석에서 많은 주목을 받고 있습니다. 본 논문은 MI의 기본 연구 대상, 사용된 기술, 평가 방법, 그리고 LM 이해에서 얻어진 주요 발견 및 응용을 총괄적으로 리뷰합니다. 신입 연구자들을 위한 로드맵을 제공하며, 현재의 연구 격차와 미래 방향도 논의합니다.

- **Technical Details**: Transformer 기반 언어 모델은 입력 토큰을 받아 다음 토큰을 예측하는 확률 분포를 출력합니다. 모델은 각 토큰의 표현을 계층적으로 업데이트하며, 각 계층에서 다중 헤드 어텐션(Multi-Head Attention, MHA)과 피드포워드(Feed-Forward, FF) 하위 계층을 통해 표현을 개선합니다. MI는 이러한 내부 구조와 계산 과정을 인간이 이해할 수 있는 메커니즘으로 설명하려는 접근법입니다.

- **Performance Highlights**: MI는 LM의 구성 요소(예: 뉴런, 어텐션 헤드)의 기능에 대한 통찰을 제공하며, 다양한 LM 행동에 대한 기계적 설명을 가능하게 합니다. 이를 통해 사용자는 LM의 활용도를 향상시킬 수 있습니다. 그러나 MI의 확장성 및 일반화 가능성, AI 안전성 문제 해결에 대한 유용성에는 여전히 우려가 존재합니다.



### Uplifting Lower-Income Data: Strategies for Socioeconomic Perspective Shifts in Vision-Language Models (https://arxiv.org/abs/2407.02623)
- **What's New**: 새롭게 시도한 연구로서, 비영어권, 지리적, 그리고 사회경제적 속성을 통합한 프롬프트(prompts)를 구성하여 다양한 국가 및 소득 그룹의 데이터에 대해 VL(vision-language) 모델 성능에 미치는 영향을 평가하였습니다. 연구 결과, 지리적 및 사회경제적 속성을 통합한 프롬프트가 저소득 데이터에서 VL 성능을 향상시키며, 특히 저소득 가정에서 흔히 나타나는 주제의 출현을 더 잘 검색할 수 있게 한다는 것을 확인했습니다.

- **Technical Details**: 본 연구에서는 Dollar Street 데이터셋을 활용하여 가구항목을 다양한 소득 수준 및 국가에서 직접 수집하였습니다. 비영어권 언어, 지리적 속성, 사회경제적 속성을 통합한 세 가지 유형의 텍스트 프롬프트를 적용하였습니다. 각 이미지는 월간 가구소득 정보를 동반하며, 이 값은 구매력 평가를 통해 생활비 차이를 반영하도록 조정되었습니다. 이를 통해 'poor', 'low-mid', 'up-mid', 'rich' 등의 네 가지 소득 클래스로 이미지를 그룹화하고, 주로 저소득 그룹에게 향상된 성능을 보이는지를 분석하였습니다.

- **Performance Highlights**: 연구 결과, 비영어권 언어로 번역된 프롬프트는 해당 국가의 영어 프롬프트 대비 성능 향상을 보이지는 않았습니다. 그러나 지리적, 사회경제적 속성을 통합한 프롬프트는 저소득 데이터에서 VL 모델의 Recall 성능을 향상시키는 것으로 나타났습니다. 특히 이 전략이 어느 맥락에서 가장 큰 성과를 보이는지 식별하고 그 결과를 공유했습니다.



### D-Rax: Domain-specific Radiologic assistant leveraging multi-modal data and eXpert model predictions (https://arxiv.org/abs/2407.02604)
- **What's New**: 중요한 연구에서 개발된 D-Rax는 방사선 영상과 대화할 수 있는 자연어 인터페이스를 제공하여 방사선사들을 지원합니다. 이 모델은 특히 흉부 X선(CXR) 이미지의 해석을 개선하고, 정확한 진단을 돕기 위해 설계되었습니다.

- **Technical Details**: D-Rax는 LLaVA-Med 아키텍처를 기반으로 구축되었으며, MIMIC-CXR 데이터와 의료 영상 질문 및 응답(VQA) 페어를 사용하여 전문가 모델 예측을 통합한 향상된 데이터셋으로 훈련되었습니다. 네트워크 아키텍처는 Llama2-7B 언어 모델과 CLIP 기반의 ViT-Large/14 비주얼 인코더를 사용합니다.

- **Performance Highlights**: D-Rax는 실제 진단 업무에서 방사선사들이 자연어로 이미지와 상호작용할 수 있도록 하여 의사 결정 과정을 간소화하고 진단 정확성을 높이는 데 중요한 역할을 합니다. 통계적으로 유의미한 개선이 열려 있는 대화와 닫힌 대화 모두에서 관찰되었습니다.



### Towards More Realistic Extraction Attacks: An Adversarial Perspectiv (https://arxiv.org/abs/2407.02596)
Comments:
          To be presented at PrivateNLP@ACL2024

- **What's New**: 본 논문에서는 언어 모델(language model)에 대한 새로운 관점에서의 추출 공격(extraction attack)을 재검토합니다. 기존 연구들은 주로 고립된 트렌드를 연구했지만, 본 연구는 그보다 실제 환경에서의 상호작용에 초점을 맞추었습니다.

- **Technical Details**: 연구 결과, 프롬프트(prompt)의 미묘하고 직관적이지 않은 변화를 주거나, 더 작은 모델 또는 오래된 체크포인트를 대상으로 할 때 추출 위험이 최대 2-4배까지 증가할 수 있음을 발견했습니다. 또한 현재 널리 받아들여지는 원문 일치(verbatim match)에 의존하면 추출된 정보의 정도를 과소평가할 수 있음을 지적하며, 이를 보다 정확히 포착하기 위한 다양한 대안을 제시합니다.

- **Performance Highlights**: 데이터 중복 제거(data deduplication)는 일부 암기 문제를 해결할 수 있으나, 실제 환경에서의 적대자에 대한 추출 위험의 증대에는 여전히 취약하다는 점을 발견했습니다. 이러한 연구 결과는 적대자의 실제 능력을 인정하여 추출 위험을 과소평가하지 않도록 주의를 기울여야 한다는 점을 강조하고 있습니다.



### Actionable Cyber Threat Intelligence using Knowledge Graphs and Large Language Models (https://arxiv.org/abs/2407.02528)
Comments:
          6th Workshop on Attackers and Cyber-Crime Operations, 12 pages, 1 figure, 9 tables

- **What's New**: 이번 논문은 대형 언어 모델(Large Language Models, LLMs)과 지식 그래프(Knowledge Graphs, KGs)를 이용하여 자동으로 실행 가능한 사이버 위협 인텔리전스(Cyber Threat Intelligence, CTI)를 추출하는 방법을 제시합니다. Microsoft, Trend Micro, CrowdStrike와 같은 조직들이 점점 더 생성적 AI를 활용하여 CTI 추출을 촉진하고 있는 가운데, 이 연구는 Llama 2 시리즈, Mistral 7B Instruct, Zephyr와 같은 최신 LLM을 활용해 CTI 텍스트에서 의미 있는 triple을 추출하는 방법을 탐구합니다.

- **Technical Details**: 연구에서는 프롬프트 엔지니어링(prompt engineering), 가이던스 프레임워크(guidance framework), 그리고 파인 튜닝(fine-tuning)을 활용해 정보 추출과 구조화를 최적화하는 기술들을 평가했습니다. 추출된 데이터는 지식 그래프(KG)를 구축하여 구조적이고 질의 가능한 형태로 제공됩니다. 실험 결과, 가이던스 프레임워크와 파인 튜닝이 프롬프트 엔지니어링보다 우수한 성능을 보여주었습니다. 그러나 대규모 데이터에 LLM을 적용하여 KG 구축과 링크 예측(link prediction)을 수행하는 것은 여전히 도전 과제로 남아있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 정보 추출에서 높은 성능을 보여주었습니다. 특히, 가이던스 프레임워크와 파인 튜닝을 결합한 접근 방식이 프롬프트 엔지니어링 접근 방식보다 뛰어난 성능을 보였습니다. 하지만 대규모 데이터셋에 LLM을 적용하는데 있어서의 한계와 이를 개선하기 위한 실용적인 해결책도 함께 제시되었습니다. 이 연구는 LLM을 통한 대규모 데이터 처리 및 정보 추출의 가능성과 한계를 탐구하는데 초점을 맞추며, 향후 LLM의 확장성과 적용성을 향상시키는 데 기여할 것입니다.



### INDICT: Code Generation with Internal Dialogues of Critiques for Both Security and Helpfulness (https://arxiv.org/abs/2407.02518)
- **What's New**: 최근 LLMs에 있어 코드 생성(task)에서의 안전성(safety) 및 유용성(helpfulness)을 개선하기 위한 INDICT 프레임워크를 소개합니다. INDICT는 두 가지 비평가(critics) 시스템을 도입하여 코드 생성을 더욱 안전하고 유용하게 합니다. 두 비평가는 안전성 중심과 유용성 중심으로 구분되며, 각 비평가는 코드 스니펫과 웹 검색 등 외부 도구를 통해 추가 지식을 얻고 이를 바탕으로 분석을 진행합니다.

- **Technical Details**: INDIT는 안전성 비평가와 유용성 비평가로 구성된 듀얼 비평 시스템(dual critic system)을 도입합니다. 이 비평가들은 생성 단계와 실행 단계 모두에서 사전과 사후 비평을 제공합니다. 이를 통해 LLM이 생성하는 코드의 품질을 사전에 방지하고 사후에도 최적화를 할 수 있습니다. 각 비평가는 외부 지식을 활용하여 보다 심도 있는 비평을 수행합니다.

- **Performance Highlights**: 5개의 벤치마크에서 8개의 프로그래밍 언어를 대상으로 8개의 다양한 작업을 통해 INDICT의 성능을 평가했습니다. 7B에서 70B 파라미터를 가진 LLM을 사용한 실험에서 안전성 및 유용성 분석과 코드 품질에서 $+10"%$ 절대 개선을 확인했습니다. 이러한 접근 방식은 안전한 상태를 유지하며 생성된 코드의 유용성을 개선하는 데 있어 SOTA(State-of-the-Art) 성능을 보여줍니다.



### LOGIC-LM++: Multi-Step Refinement for Symbolic Formulations (https://arxiv.org/abs/2407.02514)
- **What's New**: 본 논문에서는 복잡한 추론 작업을 해결하는 데 있어 대형 언어 모델(LLMs)의 한계를 탐구합니다. 기존 접근법은 형식 언어(formal languages)를 중간 표현으로 사용하지만, 중간 형식 명세의 생성 및 수정에 어려움을 겪고 있습니다. 이러한 문제를 해결하기 위해 논문은 Logic-LM의 개선판인 Logic-LM++을 제안합니다. 이 모델은 LLM의 쌍별 비교(pairwise comparisons) 능력을 활용해, LLM이 제안한 수정 사항을 평가하고 의미론적 검사를 수행할 수 있습니다. FOLIO 및 AR-LSAT 두 데이터셋에서 Logic-LM++이 Logic-LM과 기타 LLM 기반 기술보다 우수한 성능을 보였다고 합니다.

- **Technical Details**: Logic-LM++는 LLM의 쌍별 비교 능력을 활용하여, 문제 진술과 관련된 의미론적 검사를 통해 상징적 형태 작성의 개선 사항을 평가합니다. 연구진은 전반적인 문제 진술을 개선하기 위해 Logic-LM의 자체 수정 에이전트를 발전시켜 좀 더 문맥에 맞는 문제 진술을 제공합니다. 이렇게 함으로써 잘못된 수정 사항을 방지하고, 수정할 때 문제 진술의 구조를 손상시키지 않도록 합니다.

- **Performance Highlights**: 논문에서는 Logic-LM++이 표준 프롬프트, Chain of Thought 프롬프트 및 Logic-LM에서 각각 13.5%, 11%, 5%의 평균 성능 향상을 보였다고 보고합니다. FOLIO와 AR-LSAT 데이터셋 모두에서 이 모델이 더 나은 성능을 보였음을 증명합니다.



### LLM-A*: Large Language Model Enhanced Incremental Heuristic Search on Path Planning (https://arxiv.org/abs/2407.02511)
Comments:
          Submitted to The 2024 Conference on Empirical Methods in Natural Language Processing

- **What's New**: 본 연구에서는 LLM-A*라는 새로운 경로 계획 방법을 제안하였습니다. 이 방법은 전통적인 A* 알고리즘과 Large Language Models(LLMs)의 글로벌 통찰력을 결합하여 경로 찾기 효율성을 높이고자 합니다. 특히, 이 혼합 접근법은 대규모 시나리오에서도 경로의 유효성을 유지하면서 계산 및 메모리 비용을 줄이는 것을 목표로 합니다.

- **Technical Details**: 전통적인 A* 알고리즘은 시작점에서 목적지까지의 최적 경로를 찾기 위해 사용되지만, 상태 공간이 커질수록 계산 비용과 메모리 요구사항이 기하급수적으로 증가하는 문제가 있습니다. 반대로, LLMs는 환경에 대한 전반적인 분석을 통해 경로 계획에 도움을 줄 수 있는 컨텍스트 이해능력이 뛰어나지만, 상세한 공간 및 시간 추론에서는 한계를 보입니다. LLM-A*는 LLM에서 생성된 웨이포인트를 사용하여 A*의 경로 검색 과정을 안내하고, 새로운 휴리스틱 값을 통합하여 이러한 결점을 보완합니다.

- **Performance Highlights**: 다양한 환경에서의 실험 결과, A*와 비교하여 LLM-A*는 계산 및 메모리 효율성에서 우수한 성능을 보였습니다. 특히, A*가 환경 규모가 증가함에 따라 기하급수적으로 계산 및 저장 요구량이 증가하는 반면, LLM-A*는 거의 선형 성장 패턴을 보여주었습니다. 이는 LLM-A*가 대규모 환경에서도 보다 적합하고 효율적인 경로 계획 솔루션을 제공함을 시사합니다. 실험 결과 LLM-A*는 A*보다 평균적으로 절반 이하의 연산 및 저장 공간을 필요로 하여, 큰 규모의 경로 계획에 robust하고 효율적인 솔루션을 제공합니다.



### MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention (https://arxiv.org/abs/2407.02490)
- **What's New**: 해당 연구는 긴 문맥 처리 (long-context processing)를 필요로 하는 대형 언어 모델 (Large Language Model, LLM) 추론의 속도 문제를 해결하기 위한 MInference 방법을 제안합니다. 특히, 긴 텍스트를 처리하기 위한 사전 채움 (pre-filling) 단계에서 발생하는 고비용의 자원의 문제를 해결하고자 합니다. 새로운 스파스 계산법(sparse calculation)은 긴 문맥 LLM의 사전 채움 속도를 최대 10배까지 향상시킬 수 있는 방법을 제공합니다.

- **Technical Details**: MInference는 세 가지 주목할 만한 스파스 패턴인 A-shape, Vertical-Slash, Block-Sparse를 통해 GPU에서 효율적인 스파스 계산을 수행합니다. 각 주의집합 머리에 대해 적합한 패턴을 오프라인에서 식별하고, 추론 중에 동적으로 스파스 인덱스를 구축합니다. 최적화된 GPU 커널을 사용하여 해당 스파스 패턴에 맞춘 효율적인 스파스 주의집합 계산을 수행합니다. 이 방법은 사전 학습 설정이나 추가적인 파인 튜닝 없이 기존의 LLM에 직접 적용할 수 있습니다.

- **Performance Highlights**: LLaMA-3-1M, GLM4-1M, Yi-200K, Phi-3-128K, Qwen2-128K 같은 모델을 대상으로 다양한 다운스트림 작업에서 테스트한 결과, MInference가 A100 GPU에서 사전 채움 속도를 최대 10배까지 줄이면서도 정확성을 유지함을 입증했습니다. 특히 1M 토큰을 가진 프롬프트에서 지연 시간을 30분에서 3분으로 크게 줄였습니다.



### Neurocache: Efficient Vector Retrieval for Long-range Language Modeling (https://arxiv.org/abs/2407.02486)
Comments:
          Long paper, published at the main conference NAACL'24

- **What's New**: 이 논문에서는 외부 벡터 캐시로 과거 상태를 저장하여 대형 언어 모델(LLMs)의 효율적인 문맥 크기를 확장하는 Neurocache를 소개합니다. Neurocache는 k-최근접이웃(kNN) 알고리즘을 활용하여 관련된 과거 상태를 검색하고 이를 주의(attention) 과정에 통합합니다. 이 방법은 압축된 상태를 저장하여 캐시 크기를 줄이고, 토큰당 단일 검색을 수행하여 추론 속도를 높이며, 이웃 상태로 검색 창을 확장하여 언어 모델링 및 다운스트림 작업의 정확성을 향상시킵니다.

- **Technical Details**: Neurocache는 Transformer 디코더를 사용하는데, 긴 문서를 처리하기 위해 kNN 검색을 이용하여 관련된 과거 상태를 효율적으로 검색하고 통합합니다. 긴 텍스트 시퀀스를 n개의 토큰을 포함한 더 작은 세그먼트로 나눈 후, Transformer 디코더 스택을 통해 각 세그먼트를 처리합니다. 압축 단계에서 숨겨진 상태를 압축된 형태로 투영하는’apprentissage 투영 행렬 Wp를 사용합니다. 이를 통해 효율적인 kNN 검색이 가능합니다.

- **Performance Highlights**: Neurocache는 Llama2-7B와 Mistral-7B와 같은 사전학습된 모델뿐만 아니라 처음부터 훈련된 모델 모두에서 효과적임이 입증되었습니다. 실험 결과, 단일 문서 질문 응답 및 적은 샘플 학습 과제에서 성능 향상을 확인했습니다. 또한, Neurocache는 이러한 모델의 최대 문맥 길이를 128K 토큰으로 확장하여 긴 문서 처리에서 큰 성능 향상을 보였습니다.



### RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs (https://arxiv.org/abs/2407.02485)
- **What's New**: 새로운 연구에서 발표된 RankRAG는 단일 대규모 언어 모델(LLM)을 사용하여 문맥 순위 결정 및 응답 생성 작업을 동시에 수행하는 혁신적인 프레임워크입니다. 기존 모델들과 비교하여, 소량의 순위 데이터만을 포함한 학습 데이터로도 기존의 전문가 순위 모델을 능가하며, 다양한 지식 집중 벤치마크에서 우수한 성능을 보입니다.

- **Technical Details**: RankRAG는 retrieval-augmented generation(RAG) 프로세스를 크게 개선합니다. 기존의 RAG 방식은 질문에 대해 외부 데이터베이스에서 top-k 문맥을 검색하여 LLM이 이를 읽고 응답을 생성하는 방식입니다. 그러나 현재의 RAG 파이프라인은 다량의 문맥을 효과적으로 처리하지 못하는 한계가 있습니다. RankRAG는 단일 LLM을 instruction-tuning하여 문맥 순위 결정 및 응답 생성을 동시에 가능하게 하며, 일부 순위 데이터를 포함한 학습이 surprisingly 좋은 결과를 도출합니다.

- **Performance Highlights**: Llama3-RankRAG 모델은 Llama3-ChatQA-1.5 및 GPT-4 모델을 9개의 일반 지식 집중 벤치마크에서 크게 능가했습니다. 또한, 바이오메디컬 도메인에서의 5개 RAG 벤치마크에서도 instruction fine-tuning 없이 GPT-4와 비슷한 성능을 보였습니다.



### MMedAgent: Learning to Use Medical Tools with Multi-modal Agen (https://arxiv.org/abs/2407.02483)
- **What's New**: 최초로 의료 분야에 특화된 멀티모달 AI 에이전트, MMedAgent가 소개되었습니다. 이 에이전트는 다양한 의료 도구들을 통합하여 특정 작업에 적합한 도구를 선택하고 사용자의 요청에 전문가 수준의 응답을 제공하는 기능을 갖추고 있습니다.

- **Technical Details**: MMedAgent는 LLaVA-Med를 기반으로 하여, 언어 및 멀티모달 작업을 처리할 수 있도록 기능을 확장했습니다. 주요 작업에는 그라운딩(grounding), 세그멘테이션(segmentation), 분류(classification), 의료 보고서 생성(Medical Report Generation, MRG), 검색 기반 생성(Retrieval-Augmented Generation, RAG) 등이 포함됩니다. 에이전트의 첫 번째 단계는 특정 작업별 최신 방법들, 즉 '도구'들을 수집하는 것입니다. 또한, 이러한 도구들을 선택하고 통합된 출력을 생성하는 방법을 학습시키기 위해 명령 기반 데이터셋이 구축되었습니다.

- **Performance Highlights**: MMedAgent는 다양한 복잡한 의료 작업에서 현존하는 오픈 소스 최신 방법들보다 우수한 성능을 발휘하였으며, 폐쇄형 모델인 GPT-4o보다도 더 나은 성과를 보여주었습니다. 특히, 질문 응답(VQA) 작업에서 LLaVA-Med의 원래 능력을 향상시켰고, 새로운 도구 학습에서도 효율적인 능력을 보여주었습니다.



### ValueScope: Unveiling Implicit Norms and Values via Return Potential Model of Social Interactions (https://arxiv.org/abs/2407.02472)
Comments:
          First three authors contributed equally. 33 pages. In submission

- **What's New**: 본 연구에서는 ValueScope라는 온라인 커뮤니티 내 사회 규범과 가치를 정량화하기 위해 언어 모델을 활용하는 프레임워크를 소개합니다. ValueScope를 통해 성별, 정치, 과학, 금융 카테고리에 속하는 13개의 Reddit 커뮤니티의 언어적 및 스타일적 표현들을 분석했습니다. 이를 통해 밀접하게 관련된 커뮤니티들조차도 다채로운 규범을 보여준다는 것을 정량적으로 입증하였습니다. 또한, ValueScope는 각 커뮤니티의 사회적 규범의 변화를 측정하고, 미국 대통령 선거와 같은 중요한 외부 사건들이 끼치는 영향을 추적할 수 있습니다.

- **Technical Details**: ValueScope는 Return Potential Model (RPM) 이론을 기반으로 작업합니다. 이 프레임워크는 Normness Scale Predictor와 Community Preference Predictor로 구성된 모델링 파이프라인을 통해 커뮤니티 내 사회적 규범을 측정하고, 커뮤니티 반응을 정량화합니다. 제안된 평가 방법을 통해 각 구성 요소와 파이프라인 전체를 유효성 검증합니다.

- **Performance Highlights**: ValueScope는 온라인 커뮤니티의 다양한 규범 역학을 분석하는데 있어 확장 가능한 프레임워크를 제공합니다. 주요 기여점으로는 RPM 기반의 ValueScope 프레임워크 도입, 텍스트 내 사회적 규범의 척도를 측정하는 Normness Scale Predictor와 커뮤니티 반응을 정량화하는 Community Preference Predictor 개발, 그리고 새로운 평가 방법 도입이 있습니다. 이를 통해 커뮤니티 관리자와 사회 과학자들에게 중요한 통찰을 제공하며, 규범 변화의 예측 및 선제적 개입을 가능하게 합니다.



### Ensemble of pre-trained language models and data augmentation for hate speech detection from Arabic tweets (https://arxiv.org/abs/2407.02448)
- **What's New**: 이번 연구에서는 아랍어 트윗에서 증오 발언을 분류하는 새로운 접근법을 제안합니다. 이 접근법은 사전 레이블링된 데이터를 기반으로 앙상블 학습(Ensemble Learning)과 반지도 학습(Semi-Supervised Learning)을 활용하여 성능을 개선합니다. 제안된 방법은 5개의 분류(비혐오, 일반 혐오, 인종 차별, 종교적 차별, 성차별)로 아랍어 트윗을 분류하는 실험을 통해 기존 방법들을 능가하는 성능을 입증했습니다.

- **Technical Details**: 제안된 방법은 미리 학습된 언어 모델을 기반으로 하는 앙상블 학습을 사용하며, 다양한 아랍어 BERT 모델을 세밀하게 튜닝(Fine-Tuning)하여 기본 분류기를 구축합니다. 또한, 사전 레이블링된 데이터를 활용한 반지도 학습을 통해 데이터 증강(Data Augmentation)을 수행합니다.

- **Performance Highlights**: 실험 결과, 미리 학습된 언어 모델을 활용한 앙상블 학습이 기존의 관련 연구보다 우수한 성능을 보였으며, 제안된 데이터 증강 방법이 정확도 향상에 기여하여 기존 연구들을 능가하는 성과를 달성하였습니다.



### Predicting vs. Acting: A Trade-off Between World Modeling & Agent Modeling (https://arxiv.org/abs/2407.02446)
- **What's New**: 최근 발표된 연구에서는 강화학습 기반 휴먼 피드백(RLHF)으로 정렬된 언어 모델들이 벤치마크와 장문 텍스트 생성에서 뛰어난 성능을 보이지만, 다음 토큰 예측과 같은 기본적인 작업에서는 어려움을 겪고 있음을 확인했습니다. 연구진은 RLHF 모델이 긴 텍스트 생성을 위해 무작위성을 제한하는 ‘암시적 청사진(implicit blueprints)’을 사용하기 때문이라고 제안합니다.

- **Technical Details**: RLHF 모델은 특정 목표를 위해 실행되며, 이 과정에서 ‘anchor spans’라고 하는 텍스트 구간에 확률을 집중시켜 예측 범위를 제한합니다. 이와 같은 방법은 텍스트 생성을 보다 예측 가능하게 만들어주지만, 다양한 문서를 생성하는 능력을 저하시킵니다. 연구진은 이를 가장 효과적인 현재의 에이전트 모델들을 대상으로 실험하며 확인했습니다. 또한, RLHF 모델들이 다양한 도메인에서 다음 토큰 예측 성능이 초기 'Base' 모델들에 비해 떨어진다고 분석했습니다.

- **Performance Highlights**: 실험 결과, RLHF로 훈련된 모델들은 언어 모델링 작업에서 원래 ‘Base’ 모델들보다 일관되게 낮은 성능을 보였습니다. 특히, 차원 축소로 인해 RLHF 모델들은 초기 상태보다 예측 가능성이 높은 텍스트 구간에 확률을 집중시키는 경향이 있습니다. 이러한 현상은 RLHF 모델이 자체적으로 생성한 텍스트를 조건으로 할 때가 가장 두드러지게 나타났습니다.



### Evaluating the Robustness of Adverse Drug Event Classification Models Using Templates (https://arxiv.org/abs/2407.02432)
Comments:
          Accepted at BioNLP 2024 and Shared Tasks (ACL Workshop)

- **What's New**: 의료용 약물 치료로 인한 유해 사건(ADE: Adverse Drug Effect)는 중요한 문제임에도 불구하고 공식 채널을 통해 자주 보고되지 않습니다. 최근 연구는 소셜 미디어에서 ADE에 대한 논의를 탐지하는 방안으로 전환되고 있지만, 고위험 분야인 의료에서는 모델의 성능을 심층적으로 평가하는 것이 중요합니다. 이 논문에서는 네 가지 기능(시간 순서, 부정, 감정, 유익한 효과)에 대한 템플릿을 사용하여 ADE 감지 모델의 능력을 평가하고, 기존의 검증 데이터셋과 추가적인 템플릿 기반 테스트를 비교합니다.

- **Technical Details**: 종류가 다른 두 가지 transformer 기반 모델을 사용하여 사용자 보고서에서 ADE 감지를 시험했습니다. 템플릿 기반 테스트는 기존의 보류된 테스트 세트와 비교하여 모델의 성능과 일반화 능력을 더욱 잘 이해하고자 하는 목적에서 수행되었습니다. 네 가지 기능: 시간 순서, 긍정적 감정, 유익한 효과 및 부정을 테스트했습니다. 분석 결과, 유사한 F1 점수를 갖춘 모델들이 이러한 기능들에서 다양한 결과를 보였습니다.

- **Performance Highlights**: 모델들은 일부 기능(특히 시간 순서, 감정 및 유익한 효과)에서 성능이 부족하였으며, 보류된 테스트 세트에서 유사한 성능을 보였음에도 불구하고 이 기능들에 따라 성능 차이를 보였습니다. 99개의 템플릿과 1505개의 변형된 사례를 사용하여 ADE 분류 모델의 강건성을 조사한 결과, transformer 기반 모델들은 구체적인 언어 현상에 따라 다양한 성능을 보였습니다.



### Effective Context Selection in LLM-based Leaderboard Generation: An Empirical Study (https://arxiv.org/abs/2407.02409)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2406.04383

- **What's New**: 이 논문은 AI 연구 리더보드 생성을 목표로 하는 대형 언어 모델(LLM)의 효율성에 대한 문맥 선택의 영향을 탐구합니다. 기존의 자연어 추론(NLI) 접근 방식을 뛰어넘어 프레임워크없는 텍스트 생성 목표로 SOTA 작업을 수행하여 모델의 정확성을 높이고 환상을 줄이기 위한 전략을 제시합니다.

- **Technical Details**: SOTA 작업은 (Task, Dataset, Metric, Score) 쿼드러플을 학술 문서에서 추출하는 것으로 정의됩니다. 이 논문에서는 FLAN-T5 컬렉션을 사용한 인스트럭션 파인튜닝을 통해 새로운 방법을 도입했습니다. 여기서 문맥은 모델의 정보 추출을 안내하고 환상을 완화하는 중요한 역할을 합니다. 세 가지 다른 길이와 선택성을 가진 문맥 유형을 실험하여 가장 효과적인 문맥 구성을 식별했습니다.

- **Performance Highlights**: 실험 결과, 효과적인 문맥 선택은 모델의 정확성을 향상시키고 환상을 줄이는 데 중요함을 보여주었습니다. 특히 DocTAET와 DocREC 문맥 유형이 중요한 역할을 했으며, 이를 통해 상태 최첨단(SOTA) 달성 및 일반화 가능성이 높은 모델을 구축할 수 있음을 확인했습니다. 이러한 접근법은 AI 리더보드 생성의 신뢰성을 높이는데 주요한 기여를 합니다.



### CEB: Compositional Evaluation Benchmark for Fairness in Large Language Models (https://arxiv.org/abs/2407.02408)
Comments:
          37 pages, 32 figures

- **What's New**: 대형 언어 모델(LLMs)이 다양한 자연어 처리(NLP) 작업에 점점 더 많이 사용됨에 따라, LLM이 생성하는 콘텐츠의 잠재적인 사회적 부정적 영향에 대한 우려가 생기고 있습니다. 연구진은 LLM의 편향(bias)을 평가하기 위해 다양한 데이터셋을 제안하고 있으나, 기존 평가 노력은 특정 타입의 편향에만 집중하고 일관되지 않은 평가 지표를 사용하여 서로 다른 데이터셋과 LLM 간의 비교에 어려움을 겪고 있습니다. 이를 해결하기 위해, 우리는 LLM의 편향 평가를 위한 다양한 데이터셋을 수집하고, 여러 사회적 그룹과 작업에 걸친 다양한 편향 유형을 포괄하는 CEB(Compositional Evaluation Benchmark)를 제안합니다.

- **Technical Details**: CEB의 구성은 새롭게 제안된 합성 분류법(compositional taxonomy)에 기반을 두고 있으며, 이 분류법은 각 데이터셋을 편향 유형, 사회적 그룹, 작업이라는 세 가지 차원에서 특성화합니다. 이 세 가지 차원을 결합하여 LLM의 편향에 대한 종합적인 평가 전략을 개발하게 됩니다. CEB를 통해 서로 다른 차원에서 나타나는 편향 수준을 평가할 수 있습니다.

- **Performance Highlights**: 실험 결과, 세 가지 차원에서 편향 정도가 다르게 나타나는 것을 확인하였으며, 이는 특정 편향 완화 방법의 개발에 방향을 제시해 줍니다.



### Learning to Refine with Fine-Grained Natural Language Feedback (https://arxiv.org/abs/2407.02397)
Comments:
          Code and models available at: this https URL

- **What's New**: 최근 연구에서는 대형 언어 모델(LLMs)이 생성한 응답에서 오류를 식별하고 수정하는 능력을 탐구해왔습니다. 본 연구에서는 피드백을 통한 정정을 세 가지 서로 다른 LLM 능력의 조합으로 설명하고자 합니다: (1) 오류 식별, (2) 세분화된 자연어 피드백 생성, (3) 세분화된 피드백을 통한 정정. 이 접근법은 기존의 종합적(end-to-end) 정정 방법보다 일관되게 우수한 성능을 보입니다.

- **Technical Details**: 첫 번째 단계는 고성능의 판별 모델(discriminative model)을 활용해 부정확한 생성 결과를 식별합니다. 두 번째와 세 번째 단계는 사용자 지정 프롬프트(prompted) 또는 미세 조정된(fine-tuned) LLM을 통해 구현됩니다. 특히, 두 번째 단계에서는 별도의 모델을 통해 오류를 식별하기 때문에 세분화된 피드백을 제공할 수 있습니다.

- **Performance Highlights**: 제안한 방법은 문서 기반 요약의 사실적 일관성을 개선하는 작업에서 모델의 능력 차이에 상관없이 일관되게 좋은 성능을 보였습니다. 또한, 기존의 종합적 정정 접근법과 사실성 비판에 미세 조정되지 않은 현재의 훈련된 모델을 능가합니다.



### Talking to Machines: do you read me? (https://arxiv.org/abs/2407.02354)
Comments:
          French Doctoral Habilitation HDR manuscript: https://hal.science/tel-04620199

- **What's New**: 이번 논문은 대화 시스템(Dialgoue Systems) 연구에 대한 종합적인 리뷰로, 저자 본인의 박사 논문 이후 수행해온 연구를 다룹니다. 특히, 모듈형 아키텍처에서 머신 러닝/딥 러닝 및 강화 학습(Reinforcement Learning)을 통해 엔드 투 엔드(End-to-End) 심층 신경망(Deep Neural Networks)까지의 진화를 설명합니다. 이와 함께 저자가 공동 지도한 산업 연구 프로젝트와 연구원들의 작업도 소개합니다.

- **Technical Details**: 머신 러닝 및 딥 러닝의 발전과 함께 대화 에이전트(Dialogue Agents) 연구는 크게 진보해왔습니다. Eliza에서 시작된 초기 대화 시스템은 규칙 기반 접근법에서 통계적 접근(Statistical Approaches)으로, 이후 빅데이터와 처리 장치의 발전으로 딥 러닝 접근법까지 다양하게 진화하였습니다. 특히, 연구는 Task-Oriented Dialogues(TOD)에 집중하여 간단한 문제 해결에서 대규모 자동화 시스템까지 포함합니다. 또한, 질문 응답(Question Answering) 및 LLMs(Large Language Models)를 이용한 다중 모드 대화 연구도 다룹니다.

- **Performance Highlights**: 최근 ChatGPT와 같은 모델들은 다양한 도메인에서 유연하게 대화를 생성하는 놀라운 성과를 보여주고 있습니다. 이러한 모델들은 간단한 작업에서 최적의 전략을 학습하고, Wikipedia와 같은 온라인 백과사전을 활용한 대화 방식에서도 활용되고 있습니다. 통계적 및 신경망 기반 접근법을 통해 대화의 자연스러움 및 적응성을 높였으며, 특히 대화 시스템에서 발생하는 오해를 정정하는 능력도 발전하였습니다.



### Pelican: Correcting Hallucination in Vision-LLMs via Claim Decomposition and Program of Thought Verification (https://arxiv.org/abs/2407.02352)
- **What's New**: Pelican은 시각적 명제를 검증하고 환각(hallucinations)을 감소시키기 위한 새로운 프레임워크로, 명제를 일차술어(first-order predicates) 기반의 하위 명제로 분해한 후, 각 질문에 대한 Python 코드 생성을 통해 외부 도구를 유연하게 구성합니다.

- **Technical Details**: Pelican은 명제를 (predicate, question) 페어로 분해하고, 이 하위 명제들을 계산 그래프의 노드로 개념화합니다. 그런 다음 Program-of-Thought 프롬팅을 사용해 Python 코드를 생성하고, 외부 도구와의 통합을 통해 질문에 응답합니다. 식별된 객체를 정확히 지칭하기 위해 중간 변수(intermediate variables)을 사용하며, 하위 질문 간의 공통 계산을 공유해 적응형 수정과 비일관성 식별을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, Pelican은 기본 LVLM과 비교하여 환각률을 약 8%-32% 감소시키고 기존 환각 완화 방법과 비교해 27% 감소시켰습니다. 추가 벤치마크에서도 이러한 개선을 확인할 수 있었습니다.



### Generative Large Language Models in Automated Fact-Checking: A Survey (https://arxiv.org/abs/2407.02351)
- **What's New**: 온라인 플랫폼에서의 허위 정보 확산이 큰 사회적 문제로 대두되면서 정보 검증을 위한 자동화된 방법이 필요해지고 있습니다. 대형 언어 모델(LLMs)은 이러한 자동화된 사실 검증(fact-checking)에 유망한 기회를 제공하고 있습니다. 본 논문은 생성형 LLMs를 활용한 사실 검증 접근법과 기법들을 조사하여, LLMs를 정보 검증에 사용하는 방법을 이해하고 더 발전시키는 데 기여하고자 합니다.

- **Technical Details**: 이 논문에서는 자연어 처리(NLP) 분야의 주요 컨퍼런스 논문들과 Google Scholar 및 ArXiv의 키워드 검색을 통해 70편의 관련 논문들을 조사했습니다. 주요 과제는 체크할 가치가 있는 주장 감지, 이미 사실 검증된 주장 감지, 증거 검색, 그리고 사실 검증으로 분류됩니다. 각각의 과제에 대해 LLMs를 활용한 다양한 접근법이 논의되었으며, 증거 검색에서는 LLMs의 생성 능력을 활용한 검색 쿼리 생성과 근거 선택이 중요한 역할을 합니다.

- **Performance Highlights**: 생성형 LLMs가 사실 검증에 사용된 주요 접근법은 분류(Classification)와 회귀(Regression), 문장 생성(Generation), 그리고 합성 데이터 생성(Synthetic Data Generation)으로 나눌 수 있습니다. 분류와 회귀는 목표 작업에 맞춰 LLMs를 훈련시키는 미세 조정(fine-tuning)을 통해 이루어지며, 문장 생성은 주어진 입력 데이터를 기반으로 연속적인 텍스트를 생성하는 것을 포함합니다. 이러한 접근법들은 LLMs가 제공하는 방대한 지식과 추론 능력을 활용하여 사실 검증의 정확성과 효율성을 높이는 데 기여합니다.



### MORPHEUS: Modeling Role from Personalized Dialogue History by Exploring and Utilizing Latent Spac (https://arxiv.org/abs/2407.02345)
- **What's New**: 개인화된 대화 생성 (Personalized Dialogue Generation, PDG)은 특정 역할이나 페르소나에 따라 일관된 응답을 생성하는 것을 목표로 합니다. 기존의 PDG는 외부 역할 데이터를 필요로 하지만, 이러한 데이터는 부족하거나 프라이버시 문제를 야기할 수 있습니다. 이를 해결하기 위해, 새로운 프레임워크 MORPHEUS를 소개하였습니다. MORPHEUS는 세 단계의 학습 과정을 통해 개인화된 대화 이력에서 역할을 모델링하고 이용하는 방법을 제안합니다.

- **Technical Details**: MORPHEUS는 잠재 공간 (latent space)에서 역할을 압축적으로 표현하기 위해 페르소나 코드북(persona codebook)을 생성하고, 이를 이용해 역할 정보의 사후 분포(posterior distribution)을 구성합니다. 이 방법은 모델이 보지 못한 역할에 대해서도 일반화(generalize)할 수 있게 하여 개인화된 대화를 생성할 수 있게 합니다. 세 단계의 학습 과정은 역할 인식, 잠재 공간에서의 역할 표현, 코드북을 이용한 사후 분포 예측을 포함합니다.

- **Performance Highlights**: 실험 결과, MORPHEUS는 중국어 및 영어 데이터셋에서 역할 정보 추출과 응답 생성의 성능을 향상시켰습니다. 또한, MORPHEUS는 외부 역할 데이터가 없는 상황에서도 역할을 일반화할 수 있는 능력을 보여주었으며, 대형 언어 모델의 효율적인 미세 조정 방법으로 간주될 수 있습니다.



### RVISA: Reasoning and Verification for Implicit Sentiment Analysis (https://arxiv.org/abs/2407.02340)
Comments:
          11 pages, 6 figures, and 4 tables

- **What's New**: 최근 사회적 수요가 늘어나면서, 세밀한 감정 분석(Sentiment Analysis, SA)에 대한 관심이 높아지고 있으며, 이는 명시적 단서 없이 감정을 파악하는 암시적 감정 분석(Implicit Sentiment Analysis, ISA)을 통해 이루어집니다. 이번 연구에서는 RVISA라는 두 단계의 추론 프레임워크를 제안하였습니다. 이는 DO LLMs(Decoder-only Large Language Models)와 ED LLMs(Encoder-Decoder Large Language Models)의 장점을 결합하여 신뢰성 있는 감정 추론을 가능하게 합니다.

- **Technical Details**: RVISA는 DO LLMs의 생성 능력(generation ability)과 ED LLMs의 추론 능력(reasoning ability)을 활용하여 향상된 추론 모델을 학습하는 구조입니다. 구체적으로, 세 단계의 추론 프롬프트(three-hop reasoning prompting)를 사용하여 감정 요소를 명시적으로 제공하고, 생성된 추론 내용을 활용해 ED LLM을 훈련합니다. 또한, 추가적인 검증 메커니즘을 도입해 추론 학습의 신뢰성을 확보합니다.

- **Performance Highlights**: 제안된 방법은 두 개의 벤치마크 데이터셋을 사용한 평가에서 최첨단 성능(state-of-the-art results)을 달성했습니다. 이로써 ISA 성능이 크게 향상됨을 보여줍니다.



### Open foundation models for Azerbaijani languag (https://arxiv.org/abs/2407.02337)
Comments:
          Accepted to the 1st SIGTURK Workshop

- **What's New**: 이번 논문에서는 아제르바이잔어를 지원하는 오픈 소스 토대 모델(foundation models)의 발전을 촉진하기 위해 다음의 주요 공헌점을 소개합니다. 첫째, 6억 5110만 단어로 구성된 아제르바이잔어 대형 텍스트 말뭉치(DOLLMA)를 새롭게 도입했습니다. 둘째, 이 데이터셋을 기반으로 BERT 계열의 인코더 전용 언어 모델(aLLMA) 패밀리를 새로 개발했습니다. 셋째, 아제르바이잔어 토대 모델을 평가하기 위한 레이블된 데이터셋을 제공했습니다. 마지막으로, 아제르바이잔어를 지원하는 주요 오픈 소스 모델들을 포괄하는 광범위한 평가를 수행했습니다.

- **Technical Details**: 이번 연구는 다국어 또는 아제르바이잔어 전용 오픈 소스 모델의 잠재력을 탐구했습니다. 새로 도입된 DOLLMA 말뭉치는 6억 5110만 단어로 구성되어 있으며, BERT 클래스 모델을 처음부터 새로 학습하는 데 사용되었습니다. 또한, AZE-SCI (텍스트 분류 데이터셋), AZE-NSP (다음 문장 예측 데이터셋), CB-MCQ (폐쇄형 질문-답변 데이터셋) 등 세 가지 레이블된 데이터셋이 도입되었습니다. 이러한 벤치마크를 통해 새롭게 도입된 모델 뿐만 아니라 기존 오픈 소스 모델들도 평가했습니다.

- **Performance Highlights**: 이번 연구에서는 아제르바이잔어를 지원하는 중연구된 오픈 소스 모델들이 어떤 성능을 보이는지 체계적으로 평가했습니다. 이 논문에서 제안된 모델들은 주요 자연어 이해(NLU) 작업에서 우수한 성능을 보였으며, 특히 인코더 전용 모델에서 두드러지는 결과를 나타냈습니다. 이러한 결과는 아제르바이잔어 토대 모델 개발에 있어 중요한 자료로 활용될 수 있습니다.



### Why do LLaVA Vision-Language Models Reply to Images in English? (https://arxiv.org/abs/2407.02333)
Comments:
          Pre-print

- **What's New**: 새로운 연구에 따르면, 인기 있는 LLaVA 스타일의 다중모드 비전-언어 모델(VLMs)에서 이미지가 포함된 쿼리를 입력할 때 모델이 영어 응답을 반환할 가능성이 크게 증가하는 '다중언어 편향' 현상이 발생합니다. 이는 쿼리 언어와 상관없이 발생하며, 연구는 이 문제의 원인이 주로 LLaVA 모델의 언어 모델링 컴포넌트에 있다고 확인했습니다.

- **Technical Details**: 본 연구는 디자인 공간을 광범위하게 절차적으로 제거(Ablation)하고 모델의 내부 표현을 기계적으로 분석하는 두 가지 접근 방식을 결합하여 문제를 탐구합니다. 첫째, LLaVA 스타일 모델의 디자인 선택이 언어적 불일치 응답을 생성하는 경향에 미치는 영향을 측정하기 위해 다수의 변형 모델을 훈련시키고 분석했습니다. 둘째, 중간 표현에서 시각 토큰과 언어 토큰 간의 클러스터링 패턴을 연구하고, 언어 트랜스포머 계층 내의 숨겨진 상태에 직접 개입하여 문제를 분석했습니다.

- **Performance Highlights**: 실험적으로, 다양한 LLaVA 모델에 이미지가 포함된 쿼리를 입력할 때 응답이 올바른 언어로 생성될 확률이 6%에서 53%까지 감소한다는 것을 확인했습니다. 특히, 이 문제를 해결하는 가장 효과적인 방법은 이중언어 모델을 사용하는 것으로 나타났습니다. 중간 주의(attention) 계층에서 개입하면 모델의 편향을 줄일 수 있음을 발견했습니다.



### Efficient Sparse Attention needs Adaptive Token Releas (https://arxiv.org/abs/2407.02328)
Comments:
          Accepted at ACL 2024(Findings)

- **What's New**: 최근 대형 언어 모델(LLM)의 주요 과제 중 하나는 키-값(KV) 상태를 관리하는 데 필요한 막대한 계산 및 저장 요구사항입니다. ADORE(ADaptive tOken RElease)는 이러한 문제를 해결하기 위해 '가벼운 제어 모듈'을 도입했습니다. 이 모듈은 토큰의 최상위 K 주의 가중치를 유지하며, 불필요한 KV 상태를 해제하고 이후 디코딩에 필요한 토큰을 재구성합니다.

- **Technical Details**: ADORE는 토큰의 주의 기여도를 예측하여 현재 토큰에서 가장 낮은 예측 기여도를 가진 토큰을 KV 캐시에서 해제하고, 이후 디코딩에서 중요한 역할을 할 가능성이 있는 토큰의 KV 상태를 재구성하는 가벼운 제어 모듈을 사용합니다. 이를 통해 LLM 추론 과정에 매끄럽게 통합될 수 있으며, 경량 제어 모듈의 소규모 튜닝과 훈련만으로도 우수한 성능을 발휘할 수 있습니다.

- **Performance Highlights**: 여러 벤치마크 데이터셋을 대상으로 한 광범위한 실험 결과, ADORE는 전체 주의 모델과 비교하여 최대 221.8%의 처리량 개선을 이루었으며, 텍스트 품질은 거의 동일한 수준을 유지했습니다.



### Exploring the Role of Transliteration in In-Context Learning for Low-resource Languages Written in Non-Latin Scripts (https://arxiv.org/abs/2407.02320)
- **What's New**: 이 연구에서는 저자들이 비라틴(low-resource) 스크립트로 작성된 저자원의(low-resource) 언어에 대해, 디코더-기반(Decoder-only) 대형 언어 모델(LLM)이 번역된 텍스트에서의 성능을 향상시키기 위해 음역(transliteration) 기법을 적용해 보았습니다. 세 가지 프롬프트 템플릿을 제안하며, 대상 언어 텍스트를 원본 스크립트, 라틴(Latin) 스크립트, 또는 둘 다로 표현했습니다. 여러 대표적인 LLM에 적용해 본 결과 음역의 효과는 과제 유형과 모델 크기에 따라 달라졌으며, 특히 순차 레이블링(Sequential Labeling)에서 큰 성능 향상을 보였습니다.

- **Technical Details**: 연구에서는 LLaMA2-7B, Mistral-7B-Instruct, BLOOM 모델의 7B, 3B, 1B 및 560M 변종을 사용했습니다. 실험은 원본 스크립트, 라틴 스크립트, 그리고 두 스크립트를 모두 변환한 세 가지 프롬프트를 통해 수행되었습니다. 음역은 Uroman 도구를 사용해 수행되었으며, 다국어 명칭 인식(NER), 문장 분류, 그리고 시퀀스 레이블링 등의 다양한 과제를 평가했습니다.

- **Performance Highlights**: 실험 결과, 순차 레이블링에서 음역이 큰 효과를 보였으며 특정 모델에서는 최대 25%까지 성능이 향상되었습니다. 특히, BLOOM-1B 모델에서는 Script{Combined} 프롬프트를 사용했을 때 Script{Orig} 프롬프트에 비해 24% 이상의 성능 향상이 있었습니다. 이는 음역된 라틴 스크립트의 사용이 모델이 기존에 가지고 있는 지식을 더 잘 활용할 수 있도록 한다는 것을 시사합니다.



### Soft Language Prompts for Language Transfer (https://arxiv.org/abs/2407.02317)
- **What's New**: 본 연구는 고자원 언어와 저자원 언어 간 교차언어 지식 전이를 증진시키기 위해 언어 특정 및 작업 특정 어댑터(adapter)와 소프트 프롬프트(soft prompts)를 결합하는 방안을 탐구합니다. 저자들은 다양한 구성의 이러한 방법들을 통해 여섯 개의 언어에서의 효율성을 조사했습니다. 첫 시도로 소프트 언어 프롬프트를 사용하며, 작업 어댑터와의 조합이 기존의 언어 및 작업 어댑터 조합보다 많은 경우에 더 우수한 성능을 보인다는 결과를 도출했습니다.

- **Technical Details**: 본 연구에서는 파라미터 효율적 파인 튜닝(PEFT) 기법인 어댑터(adapter)와 프롬프트 튜닝(prompt tuning) 방식을 결합하여 교차언어 성능을 향상시키는 방식에 대해 조사했습니다. 어댑터는 트랜스포머 레이어에 다운 및 업 프로젝션(layer)을 추가하여 언어 특정 변환을 수행하며, 프롬프트 튜닝은 입력 임베딩(input embedding)에 소프트 프롬프트를 추가하여 언어 모델의 생성을 유도합니다. 연구에서는 mT0-Base 모델을 사용하며 소프트 언어 프롬프트와 작업 어댑터 조합이 저자원 언어에서의 성능 향상에 어떤 영향을 미치는지 평가했습니다.

- **Performance Highlights**: 실험 결과, 소프트 언어 프롬프트와 작업 어댑터의 조합이 많은 경우에서 기존의 언어 및 작업 어댑터 조합보다 뛰어난 성능을 보였습니다. 특히, 저자원 언어에서 소프트 언어 프롬프트가 유의미한 성능 향상을 가져왔습니다. 더불어, 프롬프트와 어댑터의 조합이 작업 및 언어에 따라 성능이 상이함을 알 수 있었으며, 항상 최적의 조합이 존재하지는 않는다는 결론을 내렸습니다.



### Evaluating the Ability of LLMs to Solve Semantics-Aware Process Mining Tasks (https://arxiv.org/abs/2407.02310)
Comments:
          Submitted to ICPM

- **What's New**: 최근 프로세스 마이닝(Process Mining) 커뮤니티에서는 대형 언어 모델(LLMs)의 잠재력에 주목하고 있습니다. 본 논문에서는 LLMs가 프로세스 마이닝 과제를 해결할 수 있는 능력을 조사하고, 특히 프로세스 행동 이해가 필요한 작업에서의 성능을 평가하였습니다.

- **Technical Details**: 논문에서는 세 가지 프로세스 마이닝 작업을 정의하였습니다: (1) 활동 순서가 올바른지 여부를 판단하는 작업, (2) 두 활동의 실행 순서가 유효한지 결정하는 작업, (3) 불완전한 활동 시퀀스에서 다음으로 수행할 활동을 선택하는 작업. 이 작업들을 위한 대규모 벤치마킹 데이터 셋을 제공하고, 다양한 실험을 통해 LLMs의 성능을 평가하였습니다. 특히, 사후학습(in-context learning)과 감독학습(fine-tuning)을 통해 LLMs의 성능을 비교 분석하였습니다.

- **Performance Highlights**: 실험 결과, LLMs는 기본 설정이나 소수의 사례제공(in-context examples) 만으로는 복잡한 프로세스 마이닝 작업을 해결할 수 없었습니다. 그러나 특정 작업에 맞게 미세 조정(fine-tuning)되면 높은 정확도를 기록하였으며, 이 시에는 작은 인코딩 기반 언어 모델(encoder-based models)을 일관되게 초과 성능을 보였습니다.



### Towards Human Understanding of Paraphrase Types in ChatGP (https://arxiv.org/abs/2407.02302)
- **What's New**: 이 연구는 Atomic Paraphrase Types (APT)를 사용하여 다양한 언어적 변화를 세분화하고 이에 대한 ChatGPT의 생성 능력을 평가합니다. 예를 들어, 문법 구조의 변화나 사용된 어휘의 전환 등을 포함합니다. 이를 평가하기 위해 500개의 문장 및 단어 수준의 주석을 포함한 APT 데이터셋을 소개하고, 이를 통해 다양한 APT 유형을 생성할 때 인간의 선호도를 분석하였습니다. 결과적으로, ChatGPT는 단순한 APT(추가, 삭제 등)를 잘 생성하지만 복잡한 구조(예: 종속절 변경)에서는 어려움을 겪는다는 것을 발견했습니다.

- **Technical Details**: 이 연구에서는 ChatGPT를 이용하여 10개의 APT 유형 및 5개의 프롬프팅 기법을 사용해 영문 패러프레이즈를 생성했습니다. 15명의 주석자가 참여하여 500개의 APT 생성 결과를 주석 달았고, 이를 통해 데이터 세트를 구축했습니다. 이 데이터셋은 RLHF(Reinforcement Learning from Human Feedback)와 DPO(Distributional Policy Optimization) 방법으로 모델을 미세 조정할 수 있는 인간의 선호도 순위를 포함합니다.

- **Performance Highlights**: ChatGPT는 단순한 변형(예: 단어 추가/삭제, 어휘 대체)에서 좋은 성능을 보였지만, 종속절 변경과 같은 복잡한 변화에서는 어려움을 겪었습니다. 특히, Chain-of-Thought(CoT) 프롬프팅은 몇몇 유형의 패러프레이즈 생성에서 성공률을 높였으나, 인간 평가에서는 CoT가 다른 방법보다 낮게 평가받았습니다. 주요 오류 중 하나는 ChatGPT가 잘못된 APT를 적용하는 경우가 많았다는 것입니다. 전반적으로, 모델의 성공률은 주어진 생성 과제의 난이도에 크게 영향을 받지 않았습니다.



### CFinBench: A Comprehensive Chinese Financial Benchmark for Large Language Models (https://arxiv.org/abs/2407.02301)
- **What's New**: 이번 논문에서는 중국 금융 문맥 하에서 대형 언어 모델(LLMs)의 금융 지식을 평가하기 위한 가장 포괄적인 평가 벤치마크인 CFinBench를 소개합니다. CFinBench는 기존의 일반적인 벤치마크를 넘어선, 더 도전적이고 분야 특화된 과제를 평가하는 데 목적이 있습니다.

- **Technical Details**: CFinBench는 4개의 1급 카테고리로 구성됩니다: (1) 금융 과목 (Financial Subject): 경제학, 통계학, 회계 감사를 포함한 기본 지식 (2) 금융 자격 (Financial Qualification): 공인 회계사(CPA), 증권 자격, 은행 자격 같은 금융 자격증 (3) 금융 실무 (Financial Practice): 세무 컨설턴트, 주니어 회계사, 증권 분석가 같은 실무 능력 (4) 금융 법률 (Financial Law): 세법, 보험법, 경제법 같은 금융 법규. 총 99,100개의 문제로 구성된 이 데이터셋은 43개의 2급 카테고리와 단일 선택, 다중 선택, 판단 문제로 세분화 됩니다.

- **Performance Highlights**: 50개의 다양한 LLM을 CFinBench에서 실험한 결과, GPT4와 일부 중국 지향 모델(Yi, Qwen, XuanYuan 등)이 벤치마크 선두를 차지했으며, 최고 평균 정확도는 60.16%였습니다. 이는 CFinBench의 도전적 난이도를 나타내며 현재의 LLM이 여전히 많은 개선 여지가 있음을 시사합니다.



### Renard: A Modular Pipeline for Extracting Character Networks from Narrative Texts (https://arxiv.org/abs/2407.02284)
Comments:
          Accepted at JOSS

- **What's New**: Renard(관계 추출)라는 Python 라이브러리는 narrative texts(서사 문서)에서 character networks(캐릭터 네트워크)를 추출할 수 있는 사용자 정의 NLP 파이프라인을 정의할 수 있게 합니다. 기존 도구들과 달리 Renard는 static networks(정적 네트워크)뿐만 아니라 dynamic networks(동적 네트워크)도 추출할 수 있습니다.

- **Technical Details**: Renard 파이프라인은 모듈화되어 있어 사용자가 각 NLP 서브태스크의 구현을 선택하고 이를 바탕으로 캐릭터 네트워크를 추출할 수 있습니다. 이를 통해 특정 유형의 텍스트에 파이프라인을 특화하고, 각 서브태스크의 성능이 네트워크에 미치는 영향을 연구할 수 있습니다. 주요 기능으로는 tokenization(토큰화), named entity recognition (NER, 명명된 개체 인식), coreference resolution(동시 참조 해결) 등이 있습니다.

- **Performance Highlights**: Renard는 Jane Austen의 소설 'Pride and Prejudice'의 co-occurrence character network(공동 출현 캐릭터 네트워크)를 성공적으로 추출했으며, 사용자가 원하는 경우 동적 네트워크 또한 추출할 수 있습니다. 또한, Renard는 NetworkX Python 라이브러리를 사용하여 그래프를 조작하며 다양한 도구 및 형식과의 호환성을 보장합니다.



### Multilingual Trolley Problems for Language Models (https://arxiv.org/abs/2407.02273)
- **What's New**: 이 논문은 'The Moral Machine Experiment'에서 사용된 대규모 교차문화적 인간 도덕적 선호 연구에 영감을 받아 대형 언어 모델(LLM)의 도덕적 결정에 대한 평가를 수행합니다. 저자들은 100개 이상의 언어로 번역된 1,000개의 도덕적 딜레마 스토리를 LLM에게 제시하고, 40백만 건의 인간 도덕적 판단 데이터와 비교합니다. LLM이 영어, 한국어, 헝가리어, 중국어 등에서는 인간의 선호와 더 일치하지만, 힌디어 및 소말리아어(아프리카)에서는 그렇지 않다는 사실을 발견했습니다. 또한, LLM의 도덕적 선택에 대한 설명을 분석한 결과 GPT-4는 공정성을, GPT-3는 공리주의를 주요 기준으로 삼는다는 것을 확인했습니다.

- **Technical Details**: MultiTP 데이터셋을 사용하여 LLM의 도덕적 결정 능력을 평가합니다. MultiTP는 클래식 트롤리 문제를 기반으로 하여, 통제된 변수(사람 수, 나이, 사회적 지위 등)를 통해 LLM의 도덕적 판단을 연구할 수 있습니다. MultiTP 데이터셋은 100개 이상의 언어로 번역되어 다양한 문화적 차이를 반영합니다. 이를 통해 LLM의 도덕적 판단과 인간 판단을 비교 분석합니다.

- **Performance Highlights**: LLM은 영어, 한국어, 헝가리어, 중국어와 같은 언어에서 인간 도덕적 선호와 더 잘 일치했으나, 힌디어와 소말리아어에서는 일치도가 낮았습니다. GPT-4는 주로 공정성을, GPT-3는 공리주의를 기반으로 도덕적 결정을 내렸습니다. LLM의 성능은 언어마다 격차가 있었으며, 이는 '언어 불평등'으로 정의되었습니다. 이는 모델 성능과 도덕적 추론이 언어마다 다르게 나타난다는 것을 의미합니다.



### Robust Zero-Shot Text-to-Speech Synthesis with Reverse Inference Optimization (https://arxiv.org/abs/2407.02243)
Comments:
          12 pages, Work in progress

- **What's New**: 논문에서는 강화 학습을 통해 사람의 피드백을 반영하여 오토레그레시브 모델 기반 제로 샷 텍스트-음성 변환(TTS) 시스템의 강인성을 향상시키기 위한 반전 추론 최적화(Reverse Inference Optimization, RIO) 방법을 제안합니다. 이 방법은 인간 주석 없이 음성의 품질을 평가하기 위해 베이지안 원리에 기초한 반전 추론 개념을 도입했습니다.

- **Technical Details**: RIO는 샘플링, 자동 주석 달기, 학습을 아우르는 프레임워크로, 강화 학습을 통해 사람의 피드백을 사용하면서 보상 모델이나 쌍별 선호 데이터 없이도 제로 샷 TTS 성능을 안정적으로 개선합니다. RIO에서 제안하는 자가 '선호' 함수는 베이지안 추론에 기반을 두고, 잘 생성된 음성 샘플이 동일한 TTS 모델로 다시 생성될 때도 잘 사용될 수 있어야 한다는 것을 가정합니다. 이를 통해 모델 학습 도중에 생성된 샘플을 선택하도록 합니다.

- **Performance Highlights**: 실험 결과, RIO는 주관적 및 객관적 지표에서 TTS 성능을 크게 향상시켰습니다. 특히, 평균 의견 점수(mean opinion scores, MOS), 단어 오류율(word error rate, WER), 화자 유사도(speaker similarity) 등이 개선되었으며, 나쁜 출력의 발생률이 거의 0%로 줄어드는 성과를 보였습니다.



### Towards a Holistic Framework for Multimodal Large Language Models in Three-dimensional Brain CT Report Generation (https://arxiv.org/abs/2407.02235)
Comments:
          6 figures, 5 supplementary figures, 8 supplementary tables

- **What's New**: 이 논문에서는 다중 모달 대형 언어 모델(Multi-modal large language models, MLLMs)을 활용하여 3D 뇌 CT 보고서를 생성하는 BrainGPT 모델을 소개합니다. 기존의 2D 방사선학 캡셔닝의 한계를 극복하고자 18,885개의 텍스트-스캔 쌍을 포함한 3D-BrainCT 데이터셋을 수집하였고, 임상 시각적 지시 조정(Clinical Visual Instruction Tuning, CVIT)을 적용하여 BrainGPT 모델을 학습시켰습니다.

- **Technical Details**: BrainGPT 모델은 3D 뇌 CT 데이터를 이용하여 방사선학 관련 보고서를 생성하도록 설계되었습니다. BLEU, METEOR, ROUGE, CIDEr-R 등의 전통적인 평가 지표에서 높은 성과를 보였으며, 외부 검증 세트(CQ500 dataset)에서 중간선 이동(Midline shift) 캡셔닝 정확도 0.91을 기록했습니다. 또한, 새로운 평가 방법인 Feature-Oriented Radiology Task Evaluation (FORTE)을 제안하여 보고서의 임상적 관련성을 평가했습니다.

- **Performance Highlights**: BrainGPT 모델은 내부 테스트에서 BLEU-1 = 44.35, BLEU-4 = 20.38, METEOR = 30.13, ROUGE-L = 47.6, CIDEr-R = 211.77의 성과를 보였습니다. FORTE F1-score 평균 값은 0.71로, 세부 사항(degree), 랜드마크(landmark), 특징(feature), 소견(impression) 측면에서 높은 평가를 받았습니다. 사람과 구별하기 어려운 보고서를 생성하는 BrainGPT의 성능을 입증하기 위해 11명의 의사가 참여한 튜링 테스트에서 약 74%의 캡션이 인간이 작성한 것과 구분되지 않았습니다.



### Synthetic Multimodal Question Generation (https://arxiv.org/abs/2407.02233)
Comments:
          Submitted to ARR June 2024

- **What's New**: 이 논문은 멀티모달 문서에서 질문-답변(QA)을 수행하는 멀티모달 마찰 증강 생성(MMRAG) 접근 방식을 다룹니다. 고품질의 평가 데이터셋 부족 문제를 해결하기 위해 합성 데이터 생성 프레임워크인 SMMQG를 제안합니다. SMMQG는 MMRAG 성능을 평가하기 위해 위키백과 문서에서 1024개의 질문을 생성합니다.

- **Technical Details**: SMMQG는 리트리버(retriever), 대형 언어 모델(LLM), 대형 멀티모달 모델(LMM) 간의 상호작용을 활용하여 질문과 답변 쌍을 생성합니다. 이 시스템은 스타일과 모달리티의 요구사항을 세세하게 조정할 수 있으며, 텍스트, 표, 이미지 등의 다양한 소스로 구성된 멀티모달 질문을 생성할 수 있습니다. 데이터 품질은 사람의 평가를 통해 검증됩니다.

- **Performance Highlights**: 생성된 합성 데이터는 기존의 크라우드 소싱 데이터셋인 MMQA와 동등하거나 더 우수한 품질을 보였습니다. 또한, SMMQG 데이터셋을 사용한 모델 평가 결과가 MMQA 데이터셋을 사용한 결과와 강하게 일치했습니다.



### PromptIntern: Saving Inference Costs by Internalizing Recurrent Prompt during Large Language Model Fine-tuning (https://arxiv.org/abs/2407.02211)
- **What's New**: PromptIntern은 반복적인 프롬프트 지식을 모델의 파라미터로 내부화하는 새로운 방법을 제안합니다. 이는 모델이 새로운 작업에 익숙해짐에 따라 점진적인 파인튜닝을 통해 자세한 템플릿과 예제를 점차 내부화하고, 최종적으로는 이러한 정보 없이도 고효율 추론을 가능하게 합니다.

- **Technical Details**: PromptIntern은 입력 프롬프트를 템플릿(template), 예제(examples), 쿼리(query) 세 가지 구성요소로 분류한 후, 학습 단계별로 템플릿 압축 비율과 몇 샷 예제 수를 선형적으로 감소시키는 일정을 설정합니다. 이후 템플릿 압축과 예제 흡수를 통해 입력 프롬프트를 전처리하고, 파인튜닝 동안 모델 파라미터에 템플릿과 예제 구성요소를 점진적으로 내부화하는 포괄적인 파이프라인을 도입합니다. 이를 통해 쿼리 프롬프트만을 사용하여 효율적으로 추론할 수 있습니다.

- **Performance Highlights**: PromptIntern은 NL2Code 같은 도전적인 작업에서 토큰 사용을 90% 이상 줄이고 추론 속도를 4.2배 가속하며, 비용을 88.3% 절감하는 성과를 보였습니다. 실험 결과 PromptIntern은 기존 프롬프트 압축 방법보다 우수하고, 직접 파인튜닝과 비슷한 정확도를 유지하면서 효율성과 효과를 동시에 만족시켰습니다.



### Generative Monoculture in Large Language Models (https://arxiv.org/abs/2407.02209)
- **What's New**: 최신 논문에서는 대규모 언어 모델(LLM)에서 발생하는 '생성 단일문화(Generative Monoculture)' 현상을 소개합니다. 이는 특정 작업에서 모델 출력의 다양성이 훈련 데이터에 비해 현저히 줄어드는 현상을 말합니다. 예를 들어, 혼합된 평가를 받은 책에 대해 긍정적인 리뷰만 생성하는 경우입니다. 이 현상은 성능을 향상시키기도 하지만, 다양한 의견을 제공하지 않는 등 위험을 동반할 수도 있습니다.

- **Technical Details**: 생성 단일문화는 훈련 데이터(인간이 생성한 데이터)와 모델이 생성한 데이터의 확률 분포가 달라짐을 뜻합니다. 예를 들어, 감정 점수나 코드 작성에서 LLM 생성 출력은 훈련 데이터에 비해 더 좁은 분포를 보였습니다. 이는 모델 정렬 프로세스 내에 내재된 원인에 의해 발생할 가능성이 크며, 다양성을 유지하거나 증진시키기 위한 미세 조정 패러다임을 개발할 필요가 제기되고 있습니다.

- **Performance Highlights**: 책 리뷰와 코드 생성 작업에서 생성 단일문화의 존재를 실험적으로 입증했습니다. 단순한 대책으로는 생성 단일문화를 완화할 수 없음을 확인했으며, 온도 조절(temperature), 샘플링(sampling), 프롬프팅(prompting) 전략의 변화는 효과적이지 않았습니다. 또한, LLM은 종종 인간이 선호하는 올바르고 효율적인 해결책을 과도하게 강조하지만, 이는 기억적인 코드 취약점을 초래할 수 있습니다.



### How to Learn in a Noisy World? Self-Correcting the Real-World Data Noise on Machine Translation (https://arxiv.org/abs/2407.02208)
- **What's New**: 이번 논문에서는 웹에서 크롤링한 평행 데이터(parallel data)의 의미적 불일치가 머신 번역 모델 훈련에 미치는 영향을 연구하고 이를 해결하기 위한 새로운 방법을 제안합니다. 기존의 데이터 전처리 필터(pre-filter)는 그 효과가 제한적이며, 이를 보완하기 위해 모델의 예측 분포를 활용하여 훈련 데이터의 교정을 수행하는 자가-정정(self-correction) 방법을 도입하였습니다.

- **Technical Details**: 연구팀은 의미적 유사성(semantic similarity)으로 제어된 현실적 의미 불일치 시뮬레이션 프로세스를 통해 실세계에서 발생하는 데이터 노이즈를 정량적으로 분석했습니다. 또한, 데이터 손실이 큰 경우, 모델의 예측 값과 실제 값 사이의 불일치를 무시하는 데이터 절단(data truncation) 방법의 한계를 지적하며, 모델의 자기 지식을 이용한 자가-정정 접근법을 제안했습니다. 이 방법은 초기 훈련 시점에서는 참조 데이터에 대한 신뢰도를 높게 유지하고, 훈련이 진행됨에 따라 모델의 예측 값을 점진적으로 참조 데이터로 교정합니다.

- **Performance Highlights**: 제안된 자가-정정 방법은 의미 불일치 노이즈가 포함된 환경에서 기존 방법들보다 일관되게 우수한 성능을 보였습니다. 특히, ParaCrawl과 CCAligned와 같은 실제 웹 크롤링된 데이터셋에 대해 최대 1.2 BLEU 포인트의 성능 향상을 달성하며, 데이터 전처리 필터와 데이터 절단 방법을 능가했습니다. 해당 방법은 8개 언어 번역 작업 전반에 걸쳐 유의미한 이득을 입증했습니다.



### Automatic Adaptation Rule Optimization via Large Language Models (https://arxiv.org/abs/2407.02203)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)을 활용하여 적응 규칙(adaptation rules)을 최적화하는 방법을 제안합니다. 기존의 규칙 기반 적응의 어려움을 해결하고자 LLMs의 상식과 추론 능력을 활용합니다. 초기 실험은 SWIM(Simulator for Web Infrastructure and Management) 환경에서 수행되었습니다.

- **Technical Details**: 이 연구는 LLMs를 최적화 도구로 사용하여 규칙 기반 적응을 설계하고 최적화하는 방법을 제안합니다. MAPE-K(Monitor, Analyze, Plan, Execute, Knowledge) 참조 아키텍처를 기반으로 하며, 주로 지식 베이스(knowledge base), 분석기(analyzer), 계획자(planner) 구성 요소에 중점을 둡니다. LLMs는 적응 규칙을 자동으로 생성하고 최적화하는 데 사용됩니다. 지식 베이스는 응용 도메인, 적응 목표, 변수, 역사적 운영 데이터 등의 정보를 제공합니다.

- **Performance Highlights**: 실험 결과, GPT-4와 DS-Coder는 SWIM 플랫폼에서 수동으로 설계된 규칙보다 뛰어난 성능을 보였습니다. 특히, 첫 번째 반복에서도 높은 성능을 나타내었으며, 이는 전통적인 검색 또는 최적화 알고리즘보다 뛰어난 초기 성능을 보여줍니다. 그러나 각 반복(iteration)마다 성능 변화가 크고 때때로 부정적인 유틸리티를 나타내는 등 한계점도 존재합니다.

- **Discussion and Limitations**: 현재 접근법의 비효율성, 특히 많은 설계 공간을 탐색하지 못하고 LLM의 느린 응답과 높은 호출 비용 문제를 언급합니다. 이를 해결하기 위해 LLM의 이전 지식(prior knowledge)과 기존 최적화 알고리즘을 통합하는 방법이 필요하다고 제안합니다.

- **Conclusion and Future Work**: LLMs를 활용하여 적응 규칙을 자동으로 설계하고 최적화하는 방법이 효과적임을 초기 실험을 통해 확인했습니다. 향후 연구는 LLM과 기존 최적화 알고리즘을 통합하여 효율성을 높이는 것과 런타임에서의 자동 진화를 허용하는 방향으로 나아갈 것입니다.



### LlamAr & GemmAr: Enhancing LLMs Through Arabic Instruction-Tuning (https://arxiv.org/abs/2407.02147)
- **What's New**: 이번 연구에서는 InstAr-500k라는 아랍어 지시 데이터세트(instruction dataset)를 새롭게 도입했습니다. 이 데이터세트는 다양한 도메인과 지시 유형을 포함한 콘텐츠를 생성하여 수집했습니다. 이를 통해 Llama-3-8B-Instruct와 Gemma-7B-IT라는 두 개의 오픈 소스 모델을 세부 튜닝(fine-tuning)하여 아랍어 NLP 작업에 대한 성능을 크게 향상시켰습니다.

- **Technical Details**: 데이터세트 생성은 합성 데이터 생성과 인간이 직접 수집한 데이터를 결합하는 방법을 통해 이루어졌습니다. LoRA 기술을 사용하여 LLaMA Factory 프레임워크 내에서 세부 튜닝을 진행했습니다. 주된 작업으로는 텍스트 전처리(text pre-processing)와 질 높은 지시와 응답을 제공하는 데이터세트의 혼합 작업이 포함되었습니다.

- **Performance Highlights**: 세부 튜닝된 LlamAr-8B와 GemmAr-7B 모델은 여러 아랍어 NLP 벤치마크에서 최첨단 성능을 달성했습니다. 이러한 결과는 아랍어 언어 모델의 성능 격차를 줄이고, 아랍어 NLP 발전을 크게 진전시켰음을 강조합니다.



### Efficient Nearest Neighbor based Uncertainty Estimation for Natural Language Processing Tasks (https://arxiv.org/abs/2407.02138)
- **What's New**: 이번 연구에서는 k-Nearest Neighbor Uncertainty Estimation ($k$NN-UE)라는 새로운 불확실성 추정 방법을 제안합니다. 이 방법은 인접한 데이터와의 거리 및 라벨 존재 비율을 활용하여 불확실성을 추정합니다. 이를 통해, 기존의 여러 확률적 추론을 요구하는 방법들과 달리 단일 추론으로도 높은 성능을 제공합니다.

- **Technical Details**: k$NN-UE는 데이터스토어로부터 k개의 인접한 이웃을 검색하여 얻은 라벨 정보를 사용하여 모델의 로그잇(logits)을 조정합니다. 이 과정에서 입력 예제와 이웃 간의 거리에 비례하여 가중치를 부여하며, 예측된 라벨이 이웃의 라벨과 일치하는 비율에 따라 추가적인 가중치를 적용합니다. 이를 통해 여러 번의 추론을 필요로 하지 않고도 불확실성을 개선할 수 있습니다. 또한, 이 방법은 근사적 최근접 이웃 검색(Approximate Nearest Neighbor Search) 및 차원 축소(Dimension Reduction) 기법을 도입하여 추론 속도를 개선할 수 있습니다.

- **Performance Highlights**: 제안된 k$NN-UE 방법은 감정 분석(Sentiment Analysis), 자연어 추론(Natural Language Inference), 명명 엔터티 인식(Named Entity Recognition)에서 기존의 기법들보다 뛰어난 성능을 보였습니다. 특히, 신뢰도 보정(Confidence Calibration), 선택적 예측(Selective Prediction), 분포 외 데이터 탐지(Out-of-Distribution Detection)에서 우수한 결과를 기록했습니다. 또한, 근사적 최근접 이웃 검색을 통해 추론 지연을 줄이면서도 성능 저하를 최소화할 수 있음을 보였습니다.



### Black Big Boxes: Do Language Models Hide a Theory of Adjective Order? (https://arxiv.org/abs/2407.02136)
- **What's New**: 이번 연구는 복잡한 명사구에서 형용사의 순서에 대한 언어 모델(LMs)의 학습 능력을 평가합니다. 이를 위해 인간의 Adjective Order Preferences (AOPs)를 설명하는 기존 가설을 검토하고, LMs에서 AOPs를 연구할 수 있는 재사용 가능한 형용사 쌍의 코퍼스(Corpus)와 AOP 측정 기준을 정의했습니다. 이를 통해 LMs가 훈련 중간 체크포인트에서 AOPs 학습을 얼마나 잘 수행하는지 조사했습니다.

- **Technical Details**: 연구진은 Adjective Orders의 자연스러운 예측을 위한 LMs의 성능을 평가하기 위해 Corpus of Adjective Pairs(CAP)을 구축하고, 일련의 AOP 측정 지표를 마련했습니다. LMs는 전체 훈련 과정 동안 인간의 AOPs와 매우 유사한 예측을 보여주었으며, 형용사 쌍의 빈도가 높은 경우 더 정확한 예측을 보였습니다. 또한, 새로운 조합에 대한 일반화는 제한적이었음을 밝혔습니다.

- **Performance Highlights**: LMs는 자연적으로 발생하는 형용사 순서를 예측하는 데 있어서 높은 정확도를 보였습니다. AOP 예측 정확도는 최대 94.1%였으며, 단순한 빅그램 통계도 90.3%의 정확도로 강력한 AOP 예측을 가능하게 했습니다. 하지만 훈련 데이터 외의 새로운 형용사 조합에 대한 일반화 능력은 제한적이었습니다.



### Fake News Detection: It's All in the Data! (https://arxiv.org/abs/2407.02122)
- **What's New**: 이 논문은 가짜 뉴스 탐지 연구를 시작하는 연구자들에게 필수적인 자원으로서, 데이터셋의 품질과 다양성이 탐지 모델의 효과성과 강건성에 중요한 역할을 한다고 강조합니다. 또한, 데이터셋의 주요 특징, 다양한 라벨링 시스템, 모델 성능에 영향을 미치는 일반적인 편향을 면밀히 설명합니다. 중요한 윤리적 문제와 모범 사례도 다루면서, 현재 이용 가능한 데이터셋의 전반적인 상태를 제공합니다. 우리는 GitHub 저장소를 통해 공공 데이터셋을 한 곳에 모아 사용자 친화적인 포털을 제공하여 연구와 개발 노력을 촉진시키고자 합니다.

- **Technical Details**: 가짜 뉴스 데이터셋은 일반적으로 텍스트, 시각적, 멀티모달 데이터 유형으로 분류됩니다. 텍스트 데이터셋은 가짜 뉴스 기사, 헤드라인, 소셜 미디어 게시물을 포함하며, 예시로 LIAR와 MisInfoText 데이터셋이 있습니다. 시각 데이터셋은 이미지와 비디오를 포함하며, Verification Corpus와 FCV-2018이 예시입니다. 멀티모달 데이터셋은 텍스트와 시각적 요소를 결합하여 보다 포괄적인 분석을 가능하게 하며, FakeNewsNet과 r/fakeddit가 있습니다. 생성적 머신 텍스트 데이터셋도 최근 주목받고 있으며, M4 데이터셋이 예시입니다.

- **Performance Highlights**: 대규모 데이터셋은 더 많은 정보를 제공하여 학습 중 패턴 일반화를 돕기 때문에 분류 성능을 향상시킵니다. 반면, 소규모 데이터셋은 오버피팅 및 신뢰성이 낮은 모델을 초래할 수 있습니다. 예를 들어, 약 713,000개의 기사를 포함하는 NELA-GT-2018 데이터셋은 다양한 데이터를 제공하여 모델 성능을 향상시키며, 오버피팅을 줄입니다. 고품질 주석과 정확한 원래 분포를 반영하는 데이터셋은 더 나은 성능을 제공합니다. 예를 들어, LIAR 데이터셋은 여러 진실성 레벨을 포함한 포괄적인 사실 검증 데이터를 제공합니다. 균형 잡힌 클래스 분포를 가진 데이터셋은 편향을 줄여서 전반적인 모델 성능을 향상시킵니다. 예를 들어, PHEME 데이터셋은 루머와 비루머 데이터의 균형 잡힌 분포를 포함하고 있습니다.



### Breaking Language Barriers: Cross-Lingual Continual Pre-Training at Sca (https://arxiv.org/abs/2407.02118)
Comments:
          8 pages

- **What's New**: Recent 연구에서는 대형 언어 모델(LLMs)을 새로운 언어로 구축하기 위해 기존에 사전 학습된 LLMs에서 연속적 사전 학습(Continual Pretraining, CPT)을 활용하는 접근법을 탐구했습니다. 이 방법을 통해 무작위 초기화된 파라미터를 사용하는 대신 기존의 LLMs를 초기화 지점으로 활용하는 것입니다.

- **Technical Details**: 연구진은 40M에서 5B 파라미터 크기까지 40개의 모델을 대상으로 한 평행 실험을 통해 CPT의 효율성을 평가했습니다. 실험에서는 영어를 소스 언어로, 중국어를 타겟 언어로 사용했으며, 두 가지 훈련 전략을 비교했습니다. 첫 번째는 무작위 초기화된 파라미터로부터 시작하는 방법이며, 두 번째는 기존의 영어 LLM을 초기화 지점으로 사용하는 CPT 방법입니다. 모든 모델은 동일한 훈련 전략을 사용하여 맥락 길이 2048 및 모델 크기의 20배에 해당하는 토큰수로 사전 학습을 진행했습니다.

- **Performance Highlights**: CPT는 각 컴퓨팅 레벨에서 낮은 손실을 달성하며, 전반적으로 25%에서 50%의 토큰을 절약할 수 있었습니다. 또한 70B 토큰으로 훈련된 5.5B 모델이 110B 토큰으로 훈련된 모델과 동일한 손실을 기록하는 것을 확인할 수 있었습니다. 새로운 확장 스케일링 법칙에 따르면, 데이터와 파라미터 크기 간의 양의 곱셈 조정 효과가 존재하며, CPT가 더 적은 훈련 토큰 수나 타겟 언어가 소스 언어와 유사할 때 전이 효과가 더 강하다는 것을 확인했습니다. 데이터 재생(Replaying) 방법을 통해 치명적 망각을 효과적으로 방지할 수 있었습니다.



### Helpful assistant or fruitful facilitator? Investigating how personas affect language model behavior (https://arxiv.org/abs/2407.02099)
Comments:
          20 pages, 12 figures

- **What's New**: 이 논문은 거대 언어 모델(LLM)에 '페르소나'를 부여하여 개인화 및 사용자 지향적 생성 결과를 얻는 방법을 탐구합니다. 페르소나란 LLM이 특정 역할을 수행하도록 기대되는 인물이나 성격을 말합니다. 연구진은 7개의 LLM에 대해 12개 카테고리에서 162개의 다양한 페르소나를 할당하여 모델의 행동에 대한 영향을 조사했습니다.

- **Technical Details**: 연구에서는 LLM에 페르소나를 부여하는 방법으로 'You are a {persona}'라는 문구를 사용했습니다. 이 문구는 시스템 메시지로 포함되거나, 시스템 메시지가 없는 경우 프롬프트의 시작 부분에 포함되었습니다. 연구에 이용된 데이터셋은 각기 다른 태도, 신뢰도, 도메인 지식, 사회적 편견, 독성에 관한 질문들로 구성되어 있습니다. 페르소나 분류는 성별, 인종, 성적 지향, 국가 출신, 직업 등의 다양한 변수를 포함합니다.

- **Performance Highlights**: 페르소나를 부여받은 모델들은 통제 페르소나(도움이 되는 조수)와 비교할 때 모든 모델 및 데이터셋에서 더 큰 변동성을 보였습니다. 예를 들어, '무신론자' 페르소나는 신뢰도 측면에서 7개의 LLM 중 6개에서 상위 3위에 올랐으며, 모든 모델에서 상위 10위 안에 들었습니다. 또한, 페르소나가 부여된 모델들이 경제적으로 불리한 배경을 가진 그룹에 대해 더 높은 비율로 응답을 거부하는 경향이 확인되었습니다.



### Crossroads of Continents: Automated Artifact Extraction for Cultural Adaptation with Large Multimodal Models (https://arxiv.org/abs/2407.02067)
Comments:
          under review

- **What's New**: 이 연구에서는 대규모 멀티모달 모델(LMMs)의 문화적 맥락 인식, 다양한 문화 표현의 정확성, 그리고 문화 경계를 넘어 콘텐츠를 적응시키는 능력을 평가하는 종합적인 3단계 연구를 소개합니다. DALL-E 3가 생성하고 인간이 검증한 대규모 데이터셋인 Dalle Street를 소개하며, 이는 67개국과 10개의 개념 클래스를 포함하여 9,935개의 이미지를 담고 있습니다. LLaVA와 GPT-4V 모델의 문화적 이해 능력에 대한 분석을 거쳐, 다양한 문화 간 이미지 적응을 위한 고도로 조합 가능한 파이프라인인 CultureAdapt를 제안합니다.

- **Technical Details**: 우선, Dalle Street는 DALL-E 3가 생성하고 인간이 검증한 67개국과 10개 개념 클래스를 포함한 9,935개의 이미지를 포함합니다. 모델의 문화적 이해를 심층적으로 평가하기 위해 인공지능 모델이 인식하는 문화적 아티팩트 (artifacts)를 추출하는 작업을 수행하였으며, 다양한 나라와 연관된 18,000개 이상의 아티팩트를 식별했습니다. 마지막으로, CultureAdapt라는 파이프라인을 제안하여, 주어진 이미지에서 문화적으로 관련된 아티팩트를 추출하고, 이를 다른 문화적 맥락에서 적응시키는 과정을 자동화하였습니다.

- **Performance Highlights**: 연구 결과, LLaVA와 GPT-4V 모델 모두 지역/국가 단위에서 문화적 맥락을 인식함에 있어 고유한 차이를 보였습니다. 특히, CultureAdapt 파이프라인은 다양한 문화적 맥락에서 이미지를 적응시키는 데 효과적임을 확인하였습니다. 이 연구는 대규모 멀티모달 모델이 보다 문화적으로 민감한 시스템을 개발할 필요성을 명확히 드러냈습니다.



### BiasDora: Exploring Hidden Biased Associations in Vision-Language Models (https://arxiv.org/abs/2407.02066)
Comments:
          Under Review

- **What's New**: 이번 연구는 Vision Language Models (VLMs)의 숨겨진 사회적 편견을 확인하고 이를 파악하는 새로운 방법을 제시합니다. 기존의 연구는 성별:직업이나 인종:범죄와 같은 한정된 편견 연관성에만 초점을 맞추었지만, 이 논문은 9개의 편견 차원에서 숨겨진 편견들을 찾고 그 편견의 부정성, 독성, 극단성을 분석합니다. 또한, 연구는 데이터셋 Dora를 공개하여 다른 연구자들이 이를 활용할 수 있도록 했습니다.

- **Technical Details**: 이 연구는 3단계 파이프라인으로 구성된 포괄적 프레임워크를 개발하여 VLM에서 숨겨진 유해한 편견들을 자동으로 발견합니다. 프레임워크는 세 가지 작업을 통해 구현됩니다: 텍스트를 텍스트로 만드는 작업(text-to-text), 텍스트를 이미지로 만드는 작업(text-to-image), 이미지를 텍스트로 만드는 작업(image-to-text). 각 작업에서 통계적으로 유의미하며 유해한 편견을 분석합니다. 텍스트-텍스트 작업은 단어 완성 과제를 통해, 텍스트-이미지 작업은 이미지 생성 템플릿을 통해, 이미지-텍스트 작업은 이미지 설명을 통해 진행됩니다.

- **Performance Highlights**: ['동일한 입력 방식에서도 다른 모델들은 상이한 연관성을 보였습니다.', '서로 다른 입력 방식에서도 다양한 모델들이 독특한 연관성을 보여주었습니다.', '실제 세계의 편견과 일치하면서도 이전 연구에서 논의되지 않은 연관성을 발견했습니다.', '실제 세계의 편견이나 상식과 맞지 않는 고정관념적 연관성도 확인되었습니다.']



### Are Data Augmentation Methods in Named Entity Recognition Applicable for Uncertainty Estimation? (https://arxiv.org/abs/2407.02062)
- **What's New**: 이번 연구는 Named Entity Recognition(NER) 작업에서 데이터 확장이 자신감을 조정(calibration)하고 불확실성 추정(uncertainty estimation)에 미치는 영향을 탐구했습니다. 연구 결과, 데이터 확장은 특히 도메인 내(in-domain) 설정에서 자신감 조정과 불확실성을 개선하는 데 효과적이며, 문장의 perplexity가 낮을수록 이러한 효과가 더욱 두드러짐을 발견했습니다.

- **Technical Details**: NER은 이름 있는 엔터티를 찾아내고 이를 사전 정의된 카테고리에 분류하는 Natural Language Processing(NLP) 분야의 중요한 작업입니다. 최근 Pre-trained Language Models(PLMs) 기반 아키텍처, 예를 들어 BERT와 DeBERTa가 강력한 성능을 보여주고 있습니다. 그러나 Deep Neural Networks(DNNs)는 종종 잘못된 자신감(miscalibration) 문제를 겪으며, 이는 특히 비용이 높은 오류가 발생할 수 있는 도메인에서 문제를 야기할 수 있습니다. MC Dropout 같은 기존의 방법은 계산 비용이 많이 들기 때문에 실제 적용에 어려움을 겪습니다. 이에 비해 데이터 확장은 추가 추론 시간을 요구하지 않고도 자신감 조정을 개선할 수 있는 잠재력을 가지고 있습니다.

- **Performance Highlights**: 우리의 실험은 여러 데이터 확장 방법이 도메인 내에서 자신감 조정과 불확실성 추정을 개선하는 데 효과적임을 보여주었습니다. 특히, 엔터티 예측 기반 데이터 확장과 동일한 엔터티 타입으로의 엔터티 교체(Entity Replacement)는 좋은 성능을 보였습니다. MC Dropout이나 Temperature Scaling 같은 기존 방법들보다 데이터 확장 방법들이 더 나은 성능을 보였으며, 데이터 확장의 크기를 늘림에 따라 성능이 더 향상되었고, 생성된 문장의 perplexity가 낮을수록 더 큰 개선이 있었습니다.



### Integrate the Essence and Eliminate the Dross: Fine-Grained Self-Consistency for Free-Form Language Generation (https://arxiv.org/abs/2407.02056)
Comments:
          Accepted to ACL2024 Main Conference

- **What's New**: Fine-Grained Self-Consistency (FSC)는 LLM(Large Language Models)에서 생성된 여러 샘플에서 세그먼트 수준의 공통 요소를 추출하고 통합하여, 자유 형식 생성 및 추론 작업에서 성능을 향상시키는 새로운 방법입니다.

- **Technical Details**: FSC는 초기 응답을 분할하고 각 세그먼트의 공통성을 측정한 후, LLM의 텍스트 이해 및 대조 기능을 최대한 활용하여 이러한 공통 요소를 통합해 최적화된 출력을 생성합니다. 또한, '후보군 필터링(candidates filtering)' 및 '병합(merge)'이라는 두 가지 전략을 제안하여, 각각 높은 유사성을 보이는 후보군을 식별하고 결합하여 응답 품질을 향상시킵니다.

- **Performance Highlights**: GPT-3.5-turbo와 GPT-4를 활용한 요약, 코드 생성, 수학적 추론 등의 다양한 작업에서 FSC의 효과를 입증했습니다. 실험 결과, FSC는 기존 방법들에 비해 성능이 크게 향상되었으며, 필터링 전략은 더 나은 후보를 선택함으로써 성능을 향상시키고 병합 전략은 성능을 유지하면서 비용을 줄일 수 있음을 보여줍니다.



### Concise and Precise Context Compression for Tool-Using Language Models (https://arxiv.org/abs/2407.02043)
- **What's New**: 이 논문에서는 도구를 사용하는 언어 모델이 외부 도구 문서를 압축하여 효율성을 향상시키는 두 가지 새로운 전략을 제안합니다. 첫 번째는 선택적 압축 전략(Selective Compression Strategy)으로, 도구와 매개변수 이름과 같은 핵심 정보를 원래 텍스트 형태로 유지하며 압축 손실을 줄이는 방법입니다. 두 번째는 블록 압축 전략(Block Compression Strategy)으로, 문서를 짧게 나누고 고정된 길이의 압축 모델을 사용하여 가변적인 길이의 압축을 가능하게 하는 방법입니다.

- **Technical Details**: 기본 소프트 콘텍스트 압축(Soft Context Compression) 접근 방식을 기반으로 두 가지 전략을 사용하여 도구 문서를 간결하고 정확한 요약 시퀀스로 압축합니다. 첫 번째 선택적 압축 전략은 중요한 정보를 원본 텍스트 토큰으로 유지하여 압축 손실을 줄입니다. 두 번째 블록 압축 전략은 문서를 고정된 길이의 압축 모델로 나눠 가변적인 길이의 압축을 가능하게 합니다. 이러한 접근 방식을 통해 기본 소프트 압축 접근 방식의 단점을 보완하고 더 나은 성능을 발휘할 수 있습니다.

- **Performance Highlights**: API-Bank와 APIBench에서 실험한 결과, 최대 16배 압축 비율에서도 상한선(baseiline)과 비교하여 성능 손실이 거의 없음을 확인했습니다. 선택적 압축 전략은 중요한 정보 손실을 크게 줄여 높은 압축 비율을 가능하게 했고, 블록 압축 전략은 추가적인 압축 손실 없이 가변적인 길이의 문서를 효율적으로 처리할 수 있었습니다.



### Fake News Detection and Manipulation Reasoning via Large Vision-Language Models (https://arxiv.org/abs/2407.02042)
- **What's New**: 이 논문은 '조작 추론(manipulation reasoning)'이라는 새로운 연구 주제를 제안합니다. 조작 추론은 뉴스 콘텐츠를 기반으로한 조작 추론을 목표로 합니다. 이를 지원하기 위해 '인간 중심 및 사실 관련 가짜 뉴스(Human-centric and Fact-related Fake News, HFFN)' 벤치마크를 도입했습니다.

- **Technical Details**: 논문에서는 다중 모드 입력(텍스트-이미지)에서 세밀한 융합 특성을 추출하기 위해 교차주의 메커니즘을 사용하며, 대형 비전-언어 모델(Large Vision-Language Model, LVLM)을 백본으로 이용해 사실 관련 조작 추론을 촉진합니다. 또한 이 논문에서는 다단계 훈련 프레임워크를 통해 식별 및 추론 능력을 강화합니다.

- **Performance Highlights**: 제안된 모델인 M-DRUM은 기존의 최신(fake news detection) 모델들과 강력한 LVLM(GPT-4 및 LLaVA)을 네 가지 현실적인 도메인(엔터테인먼트, 스포츠, 정치 등)에서 전반적으로 능가함을 입증했습니다. 또한, 모델은 몇 샷 학습 및 체인 오브 소트(reasoning)에서의 개선이 확인되었습니다.



### Prompt Stability Scoring for Text Annotation with Large Language Models (https://arxiv.org/abs/2407.02039)
Comments:
          33 pages, 4 figures

- **What's New**: 이번 연구에서는 텍스트 주석(annotation) 작업에서 언어 모델(Language Models, LMs)의 재현성 문제를 해결하기 위한 새로운 프레임워크를 제안합니다. 기존에는 임의적이고 작업 별로 다양한 유사 프롬프트를 테스트하는 방식이 주로 사용되었지만, 본 연구는 이러한 접근 대신 프롬프트 안정성을 진단할 수 있는 일반적인 프레임워크를 제공합니다. 이 연구는 새로운 지표인 '프롬프트 안정성 점수(Prompt Stability Score, PSS)'를 도입하고, 이를 추정하기 위한 Python 패키지인 PromptStability를 배포합니다.

- **Technical Details**: 프롬프트 안정성을 진단하기 위해 전통적인 내부 및 외부 코더 신뢰도(INTRA- AND INTER-CODER RELIABILITY) 평가 방법을 수정하여 적용했습니다. 이를 통해, 연구자들은 다양한 프롬프트 디자인이 모델 출력의 재현성에 어떻게 영향을 미치는지 평가할 수 있습니다. 연구는 여섯 가지 다양한 데이터 셋과 열두 가지 결과를 이용하여 15만 개 이상의 데이터 행을 분류하여 프레임워크의 유효성을 시연하였습니다.

- **Performance Highlights**: 연구 결과, 제안된 프롬프트 안정성 점수(PSS)를 통해 언제 프롬프트 안정성이 낮은지 진단할 수 있음을 보여주었습니다. 또한, 제공된 Python 패키지를 활용하여 데이터의 CLASSIFICATION 작업에 있어서 프롬프트 디자인 변경의 영향을 보다 체계적으로 평가할 수 있다는 점도 입증되었습니다. 이러한 연구 결과는 프롬프트 디자인의 신뢰성을 높이는 데 중요한 시사점을 제공합니다.



### Breaking Bias, Building Bridges: Evaluation and Mitigation of Social Biases in LLMs via Contact Hypothesis (https://arxiv.org/abs/2407.02030)
Comments:
          Under Review

- **What's New**: 대형 언어 모델(LLMs)의 사회적 편향 문제를 다루기 위해 심리학의 접촉 가설(Contact Hypothesis)을 적용하는 새로운 방식이 제시되었습니다. 이 연구는 다양한 사회적 접촉을 LLM의 프롬트에 시뮬레이션하여 모델의 편향을 측정하고 이를 완화할 수 있는지를 탐구합니다.

- **Technical Details**: 연구진은 108,000개의 프롬트 세트를 생성하여 3개의 LLM (LLaMA 2, Tulu, NousHermes)의 편향을 13가지 사회적 편향 차원에서 평가했습니다. 이 프롬트를 통해 모델의 답변이 어떻게 편향을 나타내는지를 분석하고, LLM이 접촉 가설에 따라 편향을 줄일 수 있는지 검토했습니다. 이후, Social Contact Debiasing (SCD)라는 새로운 기술을 제안하여 모델을 명령 튜닝시키는 방식으로 편향을 줄였습니다.

- **Performance Highlights**: SCD 전략을 사용해 LLaMA 2 모델의 편향을 단 1 epoch에서 약 40%까지 줄이는 데 성공했습니다. 또한, 이 기법을 적용한 후에도 생성된 답변의 유창성 및 관련성은 전혀 저하되지 않았음을 확인했습니다.



### Why does in-context learning fail sometimes? Evaluating in-context learning on open and closed questions (https://arxiv.org/abs/2407.02028)
Comments:
          8 pages plus references, 4 main figures, 6 pages of supplementary material

- **What's New**: 최근 LLMs의 인컨텍스트 러닝(in-context learning) 성능을 평가하면서 새로운 질문의 어려움과 난이도에 대한 연구가 발표되었습니다. 이 연구는 새로운 벤치마크를 만들고 과학 분야의 어려운 질문과 다양한 관련성의 컨텍스트를 쌍으로 이루어 실험을 진행했습니다. 특히 흥미로운 점은 주제와 관련된 컨텍스트가 항상 더 큰 도움이 되지 않는다는 것입니다.

- **Technical Details**: 이 논문에서는 물리학과 컴퓨터 과학 분야에서 160개의 고유 질문-응답 쌍을 포함한 데이터셋을 사용했습니다. 각 질문에는 네 가지 유형의 컨텍스트(비교를 위한 무컨텍스트 포함) 중 하나가 제공되었고, GPT-4를 사용하여 생성된 응답과 함께 평가되었습니다. 논문에서 다룬 기술에는 인컨텍스트 러닝(in-context learning)과 Retrieval Augmented Generation (RAG) 시스템이 포함됩니다.

- **Performance Highlights**: 실험 결과에 따르면, GPT-4의 인컨텍스트 러닝 성능은 폐쇄형 질문의 경우 컨텍스트 관련성과 긍정적으로 상관이 있었지만, 개방형 질문의 경우에는 부정적인 상관관계를 보였습니다. 이는 질문의 형식에 따라 모델이 컨텍스트를 활용하는 방식이 다름을 나타냅니다. 또한, 추가적으로 MetaICL 및 NephSAP 데이터셋을 사용하여 폐쇄형 질문의 성능을 비교 분석하였습니다.



### An End-to-End Speech Summarization Using Large Language Mod (https://arxiv.org/abs/2407.02005)
Comments:
          InterSpeech 2024

- **What's New**: 최신 연구에 따르면 Q-Former를 도입하여 음성-텍스트 모달리티를 연결하고, 대형 언어 모델(LLMs)로부터 직접 음성 특징을 기반으로 텍스트 요약을 생성하는 새로운 방법이 제안되었습니다. 특히, LLM 기반의 ASR 및 텍스트 요약(TSum) 작업을 보조 작업으로 사용하는 다단계 학습 접근법을 채택하여 모델의 성능을 높였습니다. 제안된 모델은 How-2 데이터셋에서 경쟁력 있는 성능을 보였습니다.

- **Technical Details**: 제안된 방법론은 주로 세 가지 구성 요소로 이루어져 있습니다: 음성 인코더(S-Encoder), Q-Former 모듈, 그리고 LLM. 먼저, 음성 인코더로부터 추출된 음성 특징은 Q-Former를 통해 고정 길이의 표현으로 압축됩니다. Q-Former는 가변 길이 입력 시퀀스를 고정 길이 쿼리 표현으로 변환하여 LLM에 입력되게 합니다. 더 긴 음성 입력을 처리하기 위해 음성을 세그먼트로 나누고 해당 세그먼트의 위치 데이터를 포함한 임베딩(Ep⁢o⁢ssubscript)들을 Q-Former에 추가합니다. 그런 다음, LLaMA2-7B 모델을 기반으로 하고 Low-rank Adaptation (LoRA) 방법을 사용하여 매개변수 효율적인 파인튜닝(fine-tuning)을 수행하고 있습니다.

- **Performance Highlights**: 제안된 모델은 How-2 데이터셋에서 강력한 성능을 입증했습니다. 특히, 기존의 단계별 시스템보다 더 나은 성능을 보였으며, 전통적인 엔드-투-엔드 모델들과 비교해도 경쟁력 있는 성능을 보였습니다. BERTScore 지표에서 우수한 성능을 기록했습니다.



### Is Your Large Language Model Knowledgeable or a Choices-Only Cheater? (https://arxiv.org/abs/2407.01992)
Comments:
          KnowledgeLM Workshop @ ACL 2024

- **What's New**: 최근 연구에서는 큰 언어 모델(LLMs)이 선택지만 가지고도 객관식 질문에 답할 수 있지만, 이것이 객관식 질문 응답(MCQA) 리더보드에서 LLM의 순위에 얼마나 영향을 미치느냐는 의문이 제기되고 있습니다. 이 논문은 기존의 MCQA 데이터셋에서 대조 세트를 그래프 마이닝 방법을 사용해 추출하여 LLM이 선택지에만 의존하는 지를 탐구합니다.

- **Technical Details**: 본 연구에서는 기존의 MCQA 데이터셋을 사용하여 대조 세트를 생성하기 위해 그래프 마이닝 기법을 적용했습니다. MC 항목을 비유향 그래프의 꼭짓점으로 간주하고, 각 항목의 정답이 다른 항목의 오답과 의미적으로 동등하면 두 항목 사이에 에지를 설정했습니다. 이를 통해 최대 매칭을 찾아내어 원래 데이터셋에서 유도된 가장 큰 대조 세트를 얻습니다.

- **Performance Highlights**: UnifiedQA 데이터셋에서 820개 대조 질문 세트를 구축한 후, 12개의 LLM에서 실험을 수행했습니다. 질문과 선택지를 모두 제공했을 때, 원래 평가셋과 대조셋에서 LLM 정확도 순위가 매우 일관되게 나타났으며, Kendall’s τ 값이 약 0.9로 나타났습니다. 이는 LLM이 선택지 만으로 높은 순위를 획득하는 것이 아님을 시사합니다. 따라서 MCQA 리더보드는 여전히 LLM의 지식을 신뢰성 있게 평가할 수 있습니다.



### A Bounding Box is Worth One Token: Interleaving Layout and Text in a Large Language Model for Document Understanding (https://arxiv.org/abs/2407.01976)
- **What's New**: 최근 연구들은 OCR으로 추출된 텍스트와 공간 배치를 대형 언어 모델(Large Language Models, LLMs)에 독점적으로 통합하는 것이 문서 이해 작업에 매우 효과적일 수 있음을 보여주었습니다. 그러나 기존 방법들은 텍스트와 공간 배치를 통합할 때 너무 긴 텍스트 시퀀스를 생성하거나 LLM의 자기회귀적(autoregressive) 특성을 충분히 활용하지 못하는 한계를 가지고 있습니다. 이러한 문제를 해결하기 위해 Interleaving Layout and Text in a Large Language Model(LayTextLLM)을 소개합니다.

- **Technical Details**: LayTextLLM은 각 바운딩 박스를 단일 임베딩으로 투영하고 이를 텍스트와 교차 배치하여 긴 시퀀스 문제를 효율적으로 회피하면서 LLM의 자기회귀적 특성을 활용합니다. 이는 layout-aware next token prediction(레이아웃 인식 다음 토큰 예측)과 shuffled-OCR supervised fine-tuning(셔플드-OCR 지도 학습)을 포함한 맞춤형 학습 과제를 통해 레이아웃과 텍스트 모달리티 간의 조화를 향상시킵니다.

- **Performance Highlights**: 포괄적인 벤치마크 평가에서 LayTextLLM은 기존 상태 최첨단(state-of-the-art, SOTA) 문서 이해 MLLM보다 KIE 작업에서 27.0%, VQA 작업에서 24.1% 향상된 성능을 보였습니다. 또한 다른 최첨단 OCR 기반 LLM보다 KIE 작업에서 15.5% 향상된 성능을 보였습니다. LayTextLLM은 높은 성능을 유지하면서도 기존 모델에 비해 입력 시퀀스 길이를 줄이거나 유지합니다.



### AdaCQR: Enhancing Query Reformulation for Conversational Search via Sparse and Dense Retrieval Alignmen (https://arxiv.org/abs/2407.01965)
- **What's New**: AdaCQR는 서로 다른 검색 환경에서 정보 검색 쿼리의 일반화를 높이기 위해 새롭게 제안된 프레임워크입니다. AdaCQR는 용어 기반(term-based) 및 의미 기반(semantic-based) 검색 시스템 모두와 함께 정렬됨으로써 다양한 검색 시스템 전반에서 강력한 성능을 발휘합니다.

- **Technical Details**: AdaCQR는 두 가지 주요 방법을 통해 초월적인 성능을 달성합니다. 첫 번째로, 고급 라벨(superior labels)을 생성하는 평가 모델을 구축하여 LLM(초대형 언어 모델)을 사용한 몇 샷 학습 기법을 도입하였습니다. 두 번째로, 다양한 입력 후보들을 동시에 생성하기 위해 Diverse Beam Search 기법을 이용하여 한 후보는 우수한 성능을 보이게 하고 나머지 후보들은 융합 메트릭을 기반으로 순위를 매깁니다. 이러한 이중 단계 학습 전략을 통해 용어 기반 및 의미 기반 관점에서 모델을 정렬하여 안정성을 유지합니다.

- **Performance Highlights**: AdaCQR는 TopiOCQA와 QReCC와 같은 두 가지 주요 대화형 검색 데이터셋에서 실험한 결과, 기존 방법들보다 훨씬 뛰어난 성능을 입증했습니다. 특히, LLaMA-7B 백본을 미세 조정한 접근법들과 비교해도 T5-base만을 이용하여도 유사한 성능을 보여주었습니다. AdaCQR는 수치적 및 정성적 개선을 통해 기존 방법들을 앞서가는 성능을 발휘합니다.



### Enabling Discriminative Reasoning in LLMs for Legal Judgment Prediction (https://arxiv.org/abs/2407.01964)
- **What's New**: 법적 판결 예측(Legal judgment prediction)은 사법 효율성을 높이는 데 필수적입니다. 이번 연구에서는 기존의 대형 언어 모델(LLMs)이 이 분야에서 저조한 성과를 보이는 이유를 파악하고자 하였습니다. 이를 해결하기 위해 우리는 인간 사법 사고에서 영감을 얻은 ADAPT(Ask-Discriminate-Predict) 추론 프레임워크를 도입했습니다. ADAPT는 사례 사실을 분해하고, 잠재적인 혐의를 구별하고, 최종 판결을 예측하는 과정을 포함합니다. 우리는 또한 멀티태스킹 합성 경로를 사용해 LLM을 파인튜닝하여 법적 판결 예측의 정확도와 효율성을 높였습니다.

- **Technical Details**: ADAPT 추론 프레임워크는 세 단계로 구성되어 있습니다. 첫 번째 단계인 'Ask'에서는 소음이 많은 사건 설명을 여러 측면으로 분해하여 주요 범죄 사실을 명확히 합니다. 두 번째 단계인 'Discriminate'에서는 모델이 가능한 혐의와 관련 법 조항의 후보 풀을 생성하고 이를 평가하여 최종 후보를 결정합니다. 마지막으로 'Predict' 단계에서는 이전의 추론 과정을 종합하여 최종 예측을 제공합니다. 이러한 과정을 통해 모델은 사건의 복잡성과 혼란스러운 혐의 간의 차이를 더 잘 이해할 수 있게 됩니다.

- **Performance Highlights**: 우리의 ADAPT 프레임워크는 기존의 직접 추론 및 법적 삼단논법을 통한 추론 방법보다 우수한 성능을 보여주었습니다. 이 방법은 특히 복잡하고 혼란스러운 혐의를 다룰 때 뛰어난 성과를 발휘합니다. CAIL2018과 MultiLJP 두 가지 데이터셋에서 실시한 광범위한 실험 결과, ADAPT 프레임워크가 단일 피고 및 다중 피고 시나리오에서 새로운 상태의 기술을 달성했습니다. 특히, 가장 도전적인 혐의 세트에서 최고의 성과를 기록했습니다.



### S2D: Sorted Speculative Decoding For More Efficient Deployment of Nested Large Language Models (https://arxiv.org/abs/2407.01955)
- **What's New**: 이 논문에서 소개된 주요 발견은 다중 타겟 시나리오를 위한 새로운 Sorted Speculative Decoding(S2D) 메커니즘입니다. 이는 여러 타겟 모델을 동시에 처리할 수 있는 단일 드래프트 모델을 사용할 수 있는 아이디어를 도입하여 배포 복잡성과 비용을 줄입니다. 또한, 다양한 타겟 모델을 효율적으로 선택할 수 있도록 Adaptive Draft Selection(적응형 드래프트 선택) 전략을 소개합니다.

- **Technical Details**: Sorted Speculative Decoding(S2D) 메커니즘은 정렬된 학습을 기반으로 하며, 하나의 모델 내에서 하위 모델들을 함께 훈련시켜 다중 타겟 모델을 한 번에 처리할 수 있도록 합니다. 이를 통해 여러 드래프트 모델을 유지할 필요가 없습니다. 또한, 초안 모델이 타겟 모델의 일부로부터 추출된 후, 하위 모델들이 함께 훈련됩니다. 더욱이, 적응형 드래프트 선택 메커니즘을 사용하여 신뢰도 문턱값에 기초하여 하위 모델들을 최적 선택합니다. 본 연구에서는 self-speculative solutions가 아닌, 드래프트 단에만 접근하여 타겟 모델을 훈련할 필요가 없습니다.

- **Performance Highlights**: Spec-Bench에서 다양한 설정에서 S2D 메커니즘을 평가했습니다. Vicuna 7B, 13B, LLama Chat 70B 등과 같은 기본 모델을 포함한 실험에서, 제안된 드래프트 모델이 여러 타겟 모델을 동시에 처리할 때 기존의 베이스라인보다 성능이 우수하다는 결과를 얻었습니다. 각 설정에서 S2D 메커니즘은 빠른 디코딩 속도를 제공하며, 배포 비용을 최소화 할 수 있습니다.



### Extracting and Encoding: Leveraging Large Language Models and Medical Knowledge to Enhance Radiological Text Representation (https://arxiv.org/abs/2407.01948)
Comments:
          Accepted to ACL 2024 (Findings)

- **What's New**: 이 논문에서는 의료 분야에서 흔히 겪는 텍스트와 이미지에 대한 전문가 주석 부족 문제를 해결하기 위한 새로운 두 단계 프레임워크를 제안합니다. 이 프레임워크는 대규모 언어 모델(LLMs)을 활용한 '사실 추출기(Fact Extractor)'와 BERT 기반 '사실 인코더(Fact Encoder, CXRFE)'를 포함하여 의료 이미지 분석에서 텍스트 인코더의 표현력을 향상시키고자 합니다.

- **Technical Details**: 첫 번째 단계에서는 ChatGPT와 T5 모델을 활용해 흉부 X-레이 보고서에서 사실을 추출하는 사실 추출기가 소개됩니다. 두 번째 단계에서는 CXR BERT 모델을 기반으로 한 사실 인코더(CXRFE)가 제안되며, 여러 작업에서 도메인 전문 지식을 활용하여 미세 조정됩니다. 추가적으로, 생성된 텍스트의 사실적 정확성을 평가하는 새로운 임베딩 기반 메트릭(CXRFEScore)도 포함됩니다.

- **Performance Highlights**: 사실 추출기와 인코더는 문장 순위 매기기, 자연어 추론, 레이블 추출 등 다양한 작업에서 최신 기술을 능가하는 성능을 보였습니다. CXRFEScore 메트릭은 기존의 평가 메트릭보다도 더 효과적이고 탄탄한 성능을 보여줍니다.



### Efficient-Empathy: Towards Efficient and Effective Selection of Empathy Data (https://arxiv.org/abs/2407.01937)
- **What's New**: 최근의 급속한 대형 언어 모델(Long Language Models, LLMs) 발전과 함께, 공감적 응답 능력은 중요한 전제 조건이 되었습니다. Efficient-Empathy라는 새로운 공감 데이터 선택 알고리즘을 제안합니다. 이 알고리즘은 데이터의 감수성과 이성 점수를 자동으로 평가하고 저품질 데이터를 배제하여 고품질 데이터를 선택하는 방법입니다.

- **Technical Details**: Efficient-Empathy는 감수성과 이성 점수를 기반으로 대형 언어 모델을 훈련합니다. 선택한 감수성 데이터만을 사용해도 최첨단 성능(SoTA)을 달성합니다. 또한, 다중 데이터 선택 하이퍼파라미터를 사용하여 감수성 모델의 강건함을 입증했습니다. 감수성과 이성 데이터를 통합하여 MoE(Mixture-of-Experts) 구조를 통해 훈련함으로써 더 높은 성능을 달성했습니다.

- **Performance Highlights**: 선택된 감수성 데이터만을 사용해도 최첨단 성능을 달성했습니다(전체 데이터의 59%만 사용). 여러 데이터 선택 임계값에서 감수성 모델이 일관된 최첨단 성능을 보여, 방법의 강건함을 확인하였습니다. 최종적으로, 선택된 감수성과 이성 데이터를 MoE 모델로 훈련하여 더욱 높은 최첨단 성능을 달성했습니다.



### What We Talk About When We Talk About LMs: Implicit Paradigm Shifts and the Ship of Language Models (https://arxiv.org/abs/2407.01929)
- **What's New**: 이 논문은 새로운 관점에서 과학적 진보를 탐구합니다. 저자들은 '먼로기'(Ship of Theseus) 문제를 언어 모델(language models, LMs) 분야에 적용하여, 기존의 용어가 어떻게 진화하고 발전하는지를 분석합니다. 이를 통해 과거와 현재의 연구 동향을 체계적으로 이해하려 합니다.

- **Technical Details**: 논문에서는 최근 자연어 처리(NLP) 관련 출판물을 바탕으로 데이터 인프라를 구축했습니다. 7,650편의 최신 NLP 논문을 자료로 수집하고, 두 가지 유형의 키워드(집합적 LM 개념 및 특정 모델 이름)를 추출하여 분석했습니다. 이 분석을 통해 LMs의 참조들이 시간에 따라 어떻게 변해왔는지를 정량적으로 이해하려 하며, 이를 위해 준자동화된 일반화 가능한 프레임워크를 개발했습니다.

- **Performance Highlights**: 논문은 과학적 담론에서 시스템과 이론이 서로 어떻게 영향을 미치는지를 강조합니다. 더불어, LMs라는 용어가 시간에 따라 점진적으로 어떻게 변해왔는지를 체계적으로 살펴봄으로써, 새로운 연구자들이 이 분야의 발전을 이해하는 데 중요한 통찰력을 제공합니다. 이는 NLP 커뮤니티 내 연구의 지속 가능성과 접근성을 높이는 데 기여할 것으로 기대됩니다.



### To Forget or Not? Towards Practical Knowledge Unlearning for Large Language Models (https://arxiv.org/abs/2407.01920)
Comments:
          Work in progress

- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)이 특정 지식을 지우는 과정에서 필수적인 지식까지 과도하게 지워버리는 문제를 해결하기 위해 KnowUnDo라는 벤치마크를 도입합니다. 저작권 콘텐츠와 사용자 프라이버시 도메인을 포함한 이 벤치마크는 지워야 할 지식과 유지해야 할 지식을 구분하게 해줍니다. 이를 통해 기존의 무작위적인 지식 삭제를 개선합니다.

- **Technical Details**: KnowUnDo는 저작권 및 프라이버시 법률에 기반하여 지식 인스턴스를 무의미하게 지워야 할 범위와 보유해야 할 범위로 나누어 평가합니다. 이를 위해 Unlearn Success와 Retention Success와 같은 평가 메트릭스를 사용합니다. 또한, MemFlex라는 새로운 기법을 제안하며, 이는 모델의 파라미터 공간 내에서 민감한 파라미터만 정확하게 타겟팅하여 지우는 방식으로 구현됩니다. 이 방법은 그래디언트 정보를 활용하여 보다 유연한 메모리 관리를 가능하게 합니다.

- **Performance Highlights**: MemFlex는 기존 방법보다 더 뛰어나며, 지식 삭제와 관련된 정확도를 7.97% 향상시켰고, 훈련 시간도 11.76% 줄였습니다. LLaMA2-7B-Chat과 Qwen-1.5-7B-Chat 모델을 대상으로 한 실험에서, MemFlex는 지우기와 유지하기의 범위를 더욱 효과적으로 구분하는 능력을 보였습니다.



### Investigating the Effects of Large-Scale Pseudo-Stereo Data and Different Speech Foundation Model on Dialogue Generative Spoken Language Mod (https://arxiv.org/abs/2407.01911)
Comments:
          submitted to interspeech 2024

- **What's New**: 최근의 음성 대화 모델링(Spoken Dialogue Modeling)에서는 직접적인 전사(transcription) 없이 음성 대화를 합성하여 음성에 내재된 비텍스트 정보를 보존하려는 노력들이 이루어지고 있습니다. 그러나 이 접근 방식은 화자가 동시에 이야기할 때, 각 채널에 다른 화자를 녹음한 스테레오 대화 데이터(stereo dialogue data)가 필요하다는 점에서 도전에 직면하고 있습니다. 이를 해결하기 위해, 단일 채널 대화 데이터를 가짜 스테레오 데이터(pseudo-stereo data)로 변환하는 혁신적인 파이프라인을 개발했습니다.

- **Technical Details**: 기존 대화 시스템은 전통적인 전사 -> 텍스트 생성 -> 음성 변환 과정을 따릅니다. 하지만 이번 연구는 자가 지도 학습(self-supervised learning, SSL) 모델과 텍스트리스 음성 언어 모델링(textless spoken language modeling) 기법을 이용하여 음성 신호를 직접 이산 토큰(discrete tokens)으로 인코딩 합니다. 대화 생성 모델(dGSLM)은 입력 토큰을 별도의 채널에서 처리하는 이중 타워(transformer)를 사용하여 보다 자연스러운 대화를 생성합니다. 주요 기술로는 화자 구별(speaker diarization), 소스 분리(source separation) 및 화자 검증(speaker verification) 기법을 결합하여 단일 채널 데이터를 스테레오 채널로 변환하는 파이프라인이 있습니다.

- **Performance Highlights**: 가짜 스테레오 데이터(pseudo-stereo data)를 사용함으로써 dGSLM의 성능이 향상되었습니다. 원래 2,000시간의 훈련 데이터를 17,600시간으로 확장하여 훈련 예제의 다양성과 품질이 향상되었습니다. 또한 ASR로 미세 조정된(fine-tuned) 기초 모델을 통합하여 모든 면에서 성능이 크게 개선되었습니다. 결과적으로, dGSLM은 대화의 의미적 일관성(semantic coherence)을 보다 잘 유지하는 대화를 생성할 수 있었습니다.



### Pinyin Regularization in Error Correction for Chinese Speech Recognition with Large Language Models (https://arxiv.org/abs/2407.01909)
Comments:
          Interspeech 2024

- **What's New**: 최근 연구들이 대형 언어 모델(LLMs)이 자동 음성 인식(ASR)의 오류 수정에서 효과적임을 입증하였지만, 대부분 영어에 초점을 맞추고 있습니다. 본 논문은 중국어로 주목을 돌립니다. 첫째, 다양한 시나리오에서 724K 쌍의 가설-전사 데이터를 가진 중국어 ASR 오류 수정용 벤치마크 데이터셋인 Chinese Hypotheses Paradise 데이터셋(ChineseHP)을 구축했습니다. 이 데이터셋을 사용하여 직접 프롬프팅과 사전 훈련된 LLMs의 미세 조정을 평가했습니다. 또한, 텍스트 가설에서 직접 핀인(Pinyin) 전사를 유도하는 단순한 프롬프트 규제 방법을 제안했습니다. 실험 결과는 핀인 규제가 LLMs의 오류 수정 능력을 일관되게 향상시킴을 보여줍니다.

- **Technical Details**: 본 연구에서는 ChineseHP 데이터셋을 구축했으며, 이는 중국어의 다양한 시나리오를 포함하고 있습니다. 데이터셋은 Whisper Large V2를 중국어에 맞게 수정한 Belle-distilwhisper-large-v2-zh 모델의 ASR 출력에서 수집되었습니다. 대표적인 중국어 코퍼스인 Aishell-1, Wenetspeech, Aishell-4, Kespeech에서 데이터를 수집했습니다. 프롬프트 규제 방법으로는 핀인 시스템을 사용하였고, 이를 통해 LLMs의 오류 수정 성능을 향상시켰습니다.

- **Performance Highlights**: 실험 결과, 핀인 규제 방법을 적용한 LLMs는 동일한 방법을 사용하지 않은 LLMs보다 일관되게 더 나은 오류 수정 능력을 보였습니다. 이는 핀인이 중국어 문자의 발음과 정확히 일치하지 않더라도 오류율이 낮기 때문입니다. 예시로, 표 의 핀인 전사가 글자 가설보다 낮은 오류율을 보였습니다.



### Let the Expert Stick to His Last: Expert-Specialized Fine-Tuning for Sparse Architectural Large Language Models (https://arxiv.org/abs/2407.01906)
- **What's New**: 이번 연구에서는 Mixture-of-Experts (MoE) 아키텍처를 사용한 대규모 언어 모델 (LLM)을 위한 파라미터 효율적 미세 튜닝 (Parameter-Efficient Fine-Tuning, PEFT) 방법을 제안합니다. 기존의 PEFT 연구는 대부분 밀집 아키텍처에 집중되었지만, 이번 연구는 희소 아키텍처 LLM에 초점을 맞추며 새로운 Expert-Specialized Fine-Tuning (ESFT) 방법을 소개합니다.

- **Technical Details**: ESFT는 특정 작업에 가장 관련된 전문가들만을 튜닝하고, 나머지 전문가와 모듈은 동결시킵니다. MoE 구조에서 각 작업에 대해 활성화된 전문가들의 분포를 조사한 결과, 특화된 작업에서 활성화된 전문가들이 집중되는 경향을 보였습니다. ESFT는 이러한 특성을 활용하여 전문가를 선택하고 튜닝하며, 이를 통해 튜닝 효율성을 높이고 성능을 개선할 수 있습니다.

- **Performance Highlights**: ESFT는 전체 파라미터를 미세 튜닝하는 방법과 비교하여 최대 90%의 저장 공간 절약과 최대 30%의 훈련 시간 절약을 달성합니다. 또한, ESFT는 특정 작업에서의 성능 면에서도 전체 파라미터 미세 튜닝과 동등하거나 더 나은 성능을 보였습니다. 세부 분석 결과, ESFT는 5-15%의 전문가만을 선택하여도 유망한 성능을 보일 수 있음을 확인했습니다.



### Scope-enhanced Compositional Semantic Parsing for DR (https://arxiv.org/abs/2407.01899)
- **What's New**: 이 논문은 AMS 파서(parser)를 소개합니다. AMS 파서는 담화 표현 이론(DRT, Discourse Representation Theory)을 위한 합성(Compositional)과 신경 기호(neurosymbolic) 의미 파서입니다. 이 파서는 새로운 양화 범위(quantifier scope) 예측 메커니즘을 도입하여 복잡한 문장에서도 일관된 결과를 생성합니다.

- **Technical Details**: AMS 파서는 DRT 파싱에서 우수한 성능을 보입니다. 기존의 seq2seq 모델이 문장의 복잡성이 증가함에 따라 정확도가 감소하고 잘못된 DRT 표현을 생성하는 데 비해, AMS 파서는 이러한 문제를 해결하기 위해 설계되었습니다. 새로운 양화 범위 예측 메커니즘이 사용되어 복잡한 구문 구조에서도 안정적인 출력을 보장합니다.

- **Performance Highlights**: AMS 파서는 특히 복잡한 문장에서 잘 형성된 DRT 출력물을 신뢰할 수 있게 생성하는 데 능숙합니다. 이를 통해 DRT 파싱에서 뛰어난 성능을 자랑하며, 복잡한 문장에서도 일관된 결과를 유지합니다.



### Proposal Report for the 2nd SciCAP Competition 2024 (https://arxiv.org/abs/2407.01897)
- **What's New**: 이 논문에서는 보조 정보를 이용한 문서 요약 방법을 제안합니다. 제안된 방법은 특정 이미지, 표, 부록과 관련된 설명을 효과적으로 요약할 수 있도록 설계되었습니다. 실험 결과, 고품질 OCR 데이터와 원본 텍스트에서 추출한 정보를 활용하여 관련 객체에 대한 설명을 효율적으로 요약할 수 있음을 보여주었습니다. 이를 바탕으로, 인기 있는 텍스트 생성 모델에 보조 브랜치를 추가하여 성능을 향상시켰으며, 2024년 SciCAP 대회에서 긴 캡션 부문과 짧은 캡션 부문 모두에서 최고 점수(각각 4.33, 4.66)를 기록하며 1위를 차지하였습니다.

- **Technical Details**: 이 방법은 특히 Transformer 아키텍처를 기반으로 한 텍스트 생성 모델을 개선한 것입니다. 먼저 OCR을 통해 객체에서 텍스트 정보를 추출하고, 텍스트와 객체의 연관성을 동시에 고려하여 요약을 생성합니다. 특히, PaddleOCR을 사용하여 이미지에서 고품질의 텍스트 정보를 얻고, SpacyTextSplitter를 통해 긴 문장을 여러 청크로 나누어 모델이 필요한 정보를 선택적으로 활용할 수 있도록 했습니다. 또한, 단일 추론 패스 방식에서 벗어나 여러 번의 추론을 통해 불필요한 배경 정보를 줄여 성능을 향상시켰습니다.

- **Performance Highlights**: 제안된 방법은 2024년 SciCAP 대회에서 긴 캡션 부문에서 4.33, 짧은 캡션 부문에서 4.66의 최고 점수를 기록하며 두 부문 모두에서 1위를 차지했습니다. 이로 인해 문서 요약에서 고품질 OCR과 보조 브랜치를 활용한 방법이 매우 효과적임을 입증하였습니다.



### LogEval: A Comprehensive Benchmark Suite for Large Language Models In Log Analysis (https://arxiv.org/abs/2407.01896)
- **What's New**: 이 논문은 LogEval이라는 새로운 벤치마크 세트를 소개합니다. LogEval은 대형 언어 모델(LLMs)이 로그 분석 작업에서 얼마나 잘 수행되는지 평가하기 위해 최초로 설계된 종합 벤치마크입니다. 이 벤치마크는 로그 파싱, 로그 이상 탐지, 로그 결함 진단, 로그 요약과 같은 여러 작업을 다룹니다.

- **Technical Details**: LogEval은 4,000개의 공개 로그 데이터를 사용하여 평가를 수행하며, 각 작업에 대해 15개의 다른 프롬프트(prompts)를 사용하여 공정하고 철저한 평가를 보장합니다. 주요 LLM 기술의 영향을 보여주며, 셀프 컨시스턴시(self-consistency)와 몇 가지 샷 컨텍스추얼 러닝(few-shot contextual learning) 측면에 집중합니다. 모델 정량화, 중문-영문 질문 응답 평가 및 프롬프트 엔지니어링(prompt engineering)에 대한 발견도 논의합니다.

- **Performance Highlights**: LogEval의 평가 방법을 통해 LLM들이 로그 분석 작업에서 가진 강점과 한계를 밝혀내어, 다국어 환경에서의 모델 성능과 다양한 프롬프트 전략의 효과성을 평가합니다. 이러한 통찰은 연구자와 실무자에게 중요한 가이드를 제공합니다.



### Survey on Knowledge Distillation for Large Language Models: Methods, Evaluation, and Application (https://arxiv.org/abs/2407.01885)
Comments:
          28 pages

- **What's New**: 이번 논문은 최근 대형 언어 모델(LLMs)의 지식 증류(knowledge distillation) 기법을 조사하고 이를 세 가지 측면(방법론, 평가, 응용)에서 체계적으로 정리하였습니다. 특히, 지식 증류 방법을 흰 상자(white-box)와 검은 상자(black-box)로 구분하여 설명하며, 다양한 평가 과제와 증류 효과를 분석하여 향후 연구 방향을 제안합니다.

- **Technical Details**: 대형 언어 모델(LLMs)의 성능은 우수하지만, 그 큰 규모와 높은 계산 비용은 실무 적용에 어려움을 줍니다. 이를 해결하기 위해 모델 압축 기술이 많이 연구되고 있으며, 그 중 지식 증류(knowledge distillation)는 성능을 크게 감소시키지 않고도 추론 속도를 높이는 효과적인 방법으로 주목받고 있습니다. 본 논문은 지식 증류 방법을 흰 상자(white-box KD)와 검은 상자(black-box KD)로 나눠서 설명합니다. 흰 상자 KD는 로그잇(logits) 기반 방법과 힌트(hint) 기반 방법으로 구성됩니다. 반면, 검은 상자 KD는 교사 모델의 출력만에 의존하는 API 기반 접근법을 포함하며 인컨텍스트 학습(In-Context Learning), 체인 오브 생각(Chain-of-Thought), 지시 따름(Instruction Following) 등의 방법을 포함합니다.

- **Performance Highlights**: 지식 증류 기법은 대형 모델의 지식을 작은 모델로 전이시켜 계산 자원을 절감하고 추론 속도를 높이면서도 성능 저하를 최소화할 수 있음을 보여줍니다. 이를 통해 다양한 실무 응용에 적합한 고성능 컴팩트 모델을 구현할 수 있게 됩니다. 또한, 본 논문은 다양한 평가 과제에서 두 가지 타입의 증류 알고리즘의 효과를 비교하고 분석합니다.



### Compare without Despair: Reliable Preference Evaluation with Generation Separability (https://arxiv.org/abs/2407.01878)
- **What's New**: 최근에 발표된 arXiv 논문에서는 인간 평가자의 선호도 평가가 일관성 없다는 문제를 다루며 새로운 메타-평가 척도인 'separability'를 제안합니다. 이 척도는 텍스트 생성 모델 쌍의 결과물이 얼마나 구분 가능한지를 측정하여 테스트 인스턴스의 적합성을 평가합니다. separability는 각 모델의 여러 결과물을 샘플링하고, 그 결과물들의 구분 가능성을 분석합니다.

- **Technical Details**: separability는 두 가지 주요 요인, 즉 'cross-alignment'와 'self-alignment'에 기반합니다. 'cross-alignment'는 두 모델의 결과물 간의 유사성을, 'self-alignment'는 동일 모델 결과물 간의 유사성을 측정합니다. separability는 이 두 요인을 결합하여 테스트 인스턴스가 선호도 평가에 얼마나 적합한지를 판단합니다. 실험 결과, separability 값이 높은 인스턴스는 더 일관된 선호도 평가를 이끌어낸 것으로 나타났습니다.

- **Performance Highlights**: 연구는 separability를 사용하여 다양한 모델 쌍, 벤치마크, 작업에서 테스트 인스턴스의 일관성을 식별할 수 있음을 보여줍니다. 또한, separability를 기존의 ELO 레이팅 시스템에 통합하면, 모델 비교 시 더 신뢰할 수 있는 평가를 제공할 수 있습니다. 이는 인간 평가자 및 LLM 기반 자동 평가 시스템 모두에 적용 가능한 접근법입니다.



### VSP: Assessing the dual challenges of perception and reasoning in spatial planning tasks for VLMs (https://arxiv.org/abs/2407.01863)
- **What's New**: Vision Language Models(VLMs)의 새로운 평가 기준으로 Visual Spatial Planning(VSP) 벤치마크를 소개합니다. 이는 VLMs가 시각적 공간 계획 수행 능력을 평가하고, 세부적인 하위 과제를 통해 모델의 인식력과 추론 능력 등을 측정합니다.

- **Technical Details**: VSP 벤치마크는 두 가지 시나리오를 포함합니다: 미로 내비게이션(Maze Navigation) 시나리오와 사진 현실적인 블록 월드(Blocks World) 시나리오입니다. 각 시나리오에서 VLMs는 비주얼 입력을 해석하고 행동 계획을 세워 지시된 작업을 수행해야 합니다. 벤치마크는 총 4.4K 질문과 10개의 꼼꼼히 설계된 과제들로 구성되어 있으며, 시뮬레이션된 환경과 사진 현실적인 환경을 각각 포함합니다.

- **Performance Highlights**: 최신 VLMs를 평가한 결과, 간단한 시각적 계획 과제에서도 효과적인 계획을 생성하는 데 어려움을 겪고 있음을 확인했습니다. 세부 능력 분석에서는 VLMs의 인식 및 추론 능력에서 근본적인 결함이 드러났으며, 이것이 전체 공간 계획 수행 능력 저하의 원인으로 작용하고 있습니다.



### Improving Multilingual Instruction Finetuning via Linguistically Natural and Diverse Datasets (https://arxiv.org/abs/2407.01853)
- **What's New**: 최근의 연구가 다국어 Instruction Fine-Tuning (IFT) 데이터셋의 문제를 해결하기 위한 새로운 방법을 제안했습니다. 이 방법은 언어의 자연스러움을 유지하고 다양한 프롬프트(instructions)를 보장하여 고품질의 묘사된 IFT 데이터셋을 다국어로 수집합니다. 이를 통해 다양한 언어 환경에서 LLMs의 성능을 향상시키고자 합니다.

- **Technical Details**: 제안된 방법은 영어 중심의 LLM, 한 언어로만 구성된 코퍼스(corpus), 그리고 스코어링 함수(scoring function)를 활용합니다. 이 방법은 기존의 번역이나 템플릿 방식을 사용하지 않고, 고유 언어의 특성을 그대로 유지하며, 높은 품질을 보장합니다. 첫째, 각 언어에 대한 단일 언어 코퍼스를 주요 응답 소스로 사용합니다. 그런 다음, 이 응답을 영어로 번역하고, 영어 중심의 LLM을 사용하여 다양한 프롬프트를 생성합니다. 그 뒤, 스코어링 함수를 사용하여 생성된 IFT 예제들의 품질을 제어합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법으로 생성된 다국어 IFT 데이터셋을 사용하여 파인튜닝된 LLM은 번역 및 템플릿 기반 데이터셋을 사용한 경우보다 생성적(task) 작업에서 평균 11.1%, 판별적(task) 작업에서 6.9%의 성능 향상을 보였습니다. 특히 다국어 요약 작업에서는 번역 기반 및 템플릿 기반 데이터셋을 사용한 모델보다 각각 17.57%와 15.23%의 성능 향상을 이루었습니다. 더불어, 이 성능 향상은 기존 데이터셋 크기의 절반 이하로도 달성되었습니다.



### Purple-teaming LLMs with Adversarial Defender Training (https://arxiv.org/abs/2407.01850)
- **What's New**: 최근 대형 언어 모델(LLMs)의 안전성을 보장하기 위한 노력들(예: 사회적 편견, 유해한 제안 방지 등)은 주로 사전에 수집된 데이터에 의존해왔습니다. 이에 따라, 새롭게 등장하는 안전 문제를 적시에 대응하기 어렵다는 한계가 있었습니다. 이 문제를 해결하기 위해, 연구진은 Red-teaming(공격)과 Blue-teaming(안전 훈련) 기술을 결합한 PAD(Purple-teaming LLMs with Adversarial Defender training) 파이프라인을 제안합니다. PAD는 공격자와 수비자가 자동으로 대화를 생성하며, 공격자는 안전하지 않은 응답을 유도하고 수비자는 이를 식별해 안전한 응답으로 대응합니다. 이후, Generative Adversarial Network(GAN) 스타일로 두 모듈을 함께 업데이트하여 점점 더 강력한 공격과 방어를 실현합니다.

- **Technical Details**: PAD는 먼저 '차별적 언어 금지'와 같은 특정 안전 규칙에 위반되는 논란이 될 만한 대화 데이터를 소규모로 수집하는 것으로 시작합니다. 이후, 공격자와 수비자 모듈 간의 대화를 샘플링하고 독립적인 'Judge'를 통해 안전성을 평가한 뒤 양 모듈을 계속해서 개선합니다. PAD 시스템은 일종의 '안전 게임'을 통한 지속적인 적대적 상호작용으로 진화합니다. PAD는 안전 판단 기준과 설명을 기반으로 세부 조정되어 전체 생성 능력을 유지하면서도 안전성을 강화합니다.

- **Performance Highlights**: 실험 결과, PAD는 기존의 기법들보다 더 효과적으로 공격을 찾아내고, 보다 견고한 안전 가드레일을 형성하는 데 탁월한 성과를 보였습니다. 또한, PAD는 안전성과 모델의 전반적인 품질 사이에서 균형을 맞추는 데 우수한 결과를 나타냈으며, 대화의 여러 라운드 동안 발생하는 복합적인 공격에도 효과적으로 대응할 수 있는 능력을 입증했습니다.



### A Study of Nationality Bias in Names and Perplexity using Off-the-Shelf Affect-related Tweet Classifiers (https://arxiv.org/abs/2407.01834)
- **What's New**: 이번 논문에서는 다양한 국가의 이름이 나타내는 편향(bias)을 정량화하는 방법을 제안합니다. 템플릿이나 특정 데이터셋에 의존하는 대신, 대상 도메인 데이터에 작은 변화를 주어 반사례(counterfactual) 예제를 생성하는 방식을 사용합니다. Twitter 데이터를 활용한 감정 분석, 혐오 발언 탐지, 공격적 텍스트 분류 등에서 사용되는 분류기에서 긍정적 편향을 발견하였으며, 특정 국가 이름이 문장에서 나타날 때 최대 23%의 혐오 발언 탐지 변화와 60%의 부정 감정 예측 변화를 유발함을 확인했습니다.

- **Technical Details**: 이 연구에서 우리는 대상 도메인의 Named Entity Recognition(NER) 시스템을 사용해 반사례 예제를 생성했습니다. 특정 국가의 이름으로 대체된 인물 태그를 가진 데이터셋을 활용해 모형 출력을 비교했습니다. 또한 모델 출력 변동과 퍼플렉시티(perplexity) 사이의 상관 관계를 분석하여 편향을 평가했습니다. 실험을 통해 우리는 퍼플렉시티가 높은 이름일수록 모델이 예측하는 긍정적 감정 출력이 줄어드는 경향이 있음을 밝혀냈습니다.

- **Performance Highlights**: 감정 분석 및 혐오 발언 탐지 작업에서 일부 국가 이름의 존재가 예측 결과에 큰 영향을 미친다는 것을 발견했습니다. 감정 예측 모델이 특정 국가 이름에 따라 긍정적 출력을 더 많이 내는 경향이 확인되었으며, 이는 주로 영어 사용 국가 이름에서 두드러졌습니다. 이러한 결과는 훈련 데이터의 빈도와 직접적으로 관련되어 있다는 사실을 통해, 비영어권 국가 이름이 불공정하게 낮은 출력을 내는 원인을 설명했습니다.



### Race and Privacy in Broadcast Police Communications (https://arxiv.org/abs/2407.01817)
Comments:
          Accepted in the 27th ACM Conference on Computer-Supported Cooperative Work and Social Computing (CSCW '24)

- **What's New**: 최근 연구에서는 시카고 경찰국(CPD)이 사용하는 방송 경찰 통신(BPC)의 언어 패턴을 분석하여 인종 및 개인정보 유출과 관련된 잠재적인 문제를 탐구했습니다. 이 연구는 특히 흑인 인구가 다수인 지역, 백인 인구가 다수인 지역, 히스패닉 인구가 다수인 지역에서 2018년 8월 10일 하루 동안의 라디오 전송 기록을 분석했습니다.

- **Technical Details**: 연구는 80,775시간의 CPD 운영과 관련된 BPC 아카이브를 사용하여 오전 9시에서 오후 5시까지 24시간의 오디오 데이터를 텍스트로 전사해 분석했습니다. 연구 질문으로는 BPC가 인종적 불균형을 반영하는지, 성별 및 인종/민족과 관련된 언급이 언제 어떻게 발생하는지, 민감한 정보가 얼마나 포함되어 있는지, 그리고 대규모 언어 모델(LLMs)이 이러한 개인정보 유출 위험을 얼마나 높일 수 있는지에 중점을 두었습니다.

- **Performance Highlights**: 연구 결과, CPD가 흑인 인구에 대해 비례적이지 않은 주의를 기울이고 있으며, 이는 해당 그룹의 개인정보 유출 위험을 높이는 것으로 나타났습니다. 또한, 성별, 인종/민족, 나이 등 사회인구학적 특성은 주로 사건 정보와 관련된 BPC에서 언급되었으며, 대규모 언어 모델을 활용하면 민감한 데이터를 추출할 위험이 높아질 수 있음을 확인했습니다.



### Ground Every Sentence: Improving Retrieval-Augmented LLMs with Interleaved Reference-Claim Generation (https://arxiv.org/abs/2407.01796)
Comments:
          15 pages,2 figures

- **What's New**: 이 논문은 ReClaim라는 새로운 세밀한 부가 텍스트 생성(Attributed Text Generation, ATG) 방법을 제안합니다. ReClaim은 긴 형식의 질문-답변 작업에서 참조(reference)와 답변(answer)을 번갈아 생성하여 이전의 문단 수준 또는 단락 수준의 인용 방식보다 더 세밀한 문장 수준의 인용을 제공합니다. 이로써 LLM이 생성한 컨텐츠의 신뢰성을 높이고 검증 과정을 더 용이하게 만듭니다.

- **Technical Details**: ReClaim 방법은 Q&A 시스템에서 RAG(Retrieval-Augmented Generation) 방식을 사용하여, 모델이 생성하는 각 답변 문장마다 문장 수준의 세밀한 인용을 추가합니다. 이를 위해 특수 학습 데이터셋을 구성하고 모델을 미세 조정(fine-tuning)했습니다. 또한, 인용과 답변 생성 과정에서 일관성 문제를 해결하기 위해 디코딩 제약 조건을 추가했습니다.

- **Performance Highlights**: 실험 결과에 따르면, ReClaim 방법은 현재의 기준 모델들에 비해 일부 지표에서 더 나은 성능을 보여 주었으며, 특히 인용 품질(attribution quality) 측면에서 뛰어난 향상을 보였습니다. 또한, 인용 길이가 줄어들어 사실 확인에 필요한 노력이 감소했습니다.



### Analyzing Persuasive Strategies in Meme Texts: A Fusion of Language Models with Paraphrase Enrichmen (https://arxiv.org/abs/2407.01784)
Comments:
          15 pages, 8 figures, 1 table, Proceedings of 5th International Conference on Natural Language Processing and Applications (NLPA 2024)

- **What's New**: 이 논문은 밈 텍스트에서 설득 기술을 계층적으로 감지하는 방법을 설명합니다. 최근 SemEval 태스크의 일부로 개발된 모델은 개별 언어 모델(BERT, XLM-RoBERTa, mBERT)을 미세 조정하고, ChatGPT를 통해 패러프레이징(paraphrase generation) 데이터 증강과 평균 기반 앙상블(mean-based ensemble) 모델을 활용합니다.

- **Technical Details**: 모델은 BERT, XLM-RoBERTa, mBERT를 미세 조정한 후, 패러프레이징을 통해 데이터 증강을 수행했습니다. 각 설득 기술(class)의 메트릭을 최적화하여 분류 임계치를 조정했습니다. 테스트 단계에서는 제로 샷 접근법(zero-shot approach)을 사용하여 다른 언어의 인스턴스를 분류했습니다. 결과적으로 영어 외 다른 언어에서도 좋은 성능을 보였으나, 아랍어에 대한 성능은 미미했습니다.

- **Performance Highlights**: 패러프레이징 기술은 모델 성능을 향상시키지만, 큰 불균형 데이터셋보다 균형 잡힌 훈련 데이터셋이 더 효과적이라는 결과를 보였습니다. 또한, 다양한 분포에서 무차별적으로 패러프레이징을 도입하는 것은 시스템에 상당한 노이즈를 추가할 수 있습니다. SemEval 2024 데이터 결과는 제안된 방법이 모델 효율성을 향상시킴을 보여줍니다.



### DiscoveryBench: Towards Data-Driven Discovery with Large Language Models (https://arxiv.org/abs/2407.01725)
Comments:
          Website: this https URL

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)이 데이터 집합만을 사용하여 가설의 탐색과 검증을 자동화할 수 있을지 평가하기 위해 DiscoveryBench를 소개합니다. 이는 최초의 종합적인 벤치마크로, 데이터 기반 발견의 다단계 과정을 공식화하여 현 모델의 능력을 평가하고 개선 자원을 제공하도록 설계되었습니다. 총 264개 과제를 다양한 도메인에서 수집했으며, 903개의 인공 과제를 추가로 제공하여 통제된 평가를 수행합니다.

- **Technical Details**: DiscoveryBench는 현실 세계의 과제들을 시뮬레이션하는 방식으로 데이터 및 메타데이터, 자연어로 된 탐색 목표를 포함하여 각 과제를 정의합니다. 이러한 과제를 통해 여러 도메인에서 대형 언어 모델(LLMs)의 과학적 의미론적 추론 및 데이터 처리 능력을 평가할 수 있습니다. 예를 들어, 특정 도메인에 적합한 분석 기술을 결정하고, 데이터 정리 및 표준화 방법을 결정하는 것 등이 포함됩니다.

- **Performance Highlights**: DiscoveryBench를 통해 평가된 여러 최신 LLM 기반 추론 프레임워크들은 최고 성능이 25%에 그쳤습니다. 이는 자율적인 데이터 기반 발견 과정이 얼마나 도전적인 과제인지를 잘 보여주며, 커뮤니티가 이 분야에서 진전을 이루는 데 중요한 자원임을 입증합니다.



### NLPGuard: A Framework for Mitigating the Use of Protected Attributes by NLP Classifiers (https://arxiv.org/abs/2407.01697)
Comments:
          Paper accepted at CSCW 2024

- **What's New**: 최근 발표된 연구에서는 'NLPGuard'라는 새로운 프레임워크를 소개했습니다. 이 프레임워크는 민감한 속성(예: 인종, 성별)과 관련된 편향을 줄이도록 NLP 분류기를 수정하는 데 목적이 있습니다. NLPGuard는 기존의 NLP 분류기와 학습 데이터를 입력받아, 민감한 속성에 대한 의존성을 크게 줄이면서도 정확성을 유지하는 수정된 학습 데이터를 생성합니다.

- **Technical Details**: NLPGuard는 세 가지 주요 구성 요소로 이루어져 있습니다: (1) 모델의 예측에 가장 중요한 단어를 찾아내는 Explainer, (2) 이러한 단어가 보호된 속성과 관련이 있는지 확인하는 Identifier, (3) 학습 데이터를 조정하여 민감한 속성으로부터 모델의 학습을 줄이는 Moderator입니다. 이 프레임워크는 '독성 언어 탐지', '감정 분석', '직업 분류'와 같은 다양한 분류 작업에서 적용되어 평가되었습니다.

- **Performance Highlights**: 평가 결과, 기존의 NLP 분류기가 보호된 속성에 크게 의존한다는 사실이 확인되었습니다. 그러나 NLPGuard를 적용했을 때 이러한 의존성은 최대 79% 감소했으며, 오히려 예측 정확성을 약간 향상시키는 결과를 보였습니다. 특히, Wikipedia 댓글 데이터를 활용한 독성 언어 탐지 작업에서는, BERT 모델의 보호된 속성 의존성을 60% 감소시키면서도 예측 정확도를 0.8% 개선하는 데 성공했습니다.



### Deciphering the Factors Influencing the Efficacy of Chain-of-Thought: Probability, Memorization, and Noisy Reasoning (https://arxiv.org/abs/2407.01687)
Comments:
          9 pages plus references and appendices

- **What's New**: Chain-of-Thought (CoT) 프롬프트가 대형 언어 모델(Large Language Models, LLMs)의 다단계 추론 능력을 향상시킨다는 사실이 여러 연구 결과로 증명되었습니다. 이번 연구에서는 CoT 프롬프트를 통해 LLMs가 진정한 추상적 일반화 능력을 보이는지 아니면 피상적인 휴리스틱에 의존하는지에 대한 논쟁을 해결하려 합니다. 이를 위해 상징적 추론 과제인 시저 암호 해독(shift ciphers decoding) 작업을 사례 연구로 다루었습니다.

- **Technical Details**: 이번 연구에서는 시저 암호 해독이라는 단일 과제를 집중적으로 탐구하여 CoT 성능에 영향을 미치는 세 가지 핵심 요소(예상 출력의 확률, 사전 훈련 동안 암묵적으로 학습한 내용(기억), 추론 중 수행되는 중간 단계의 수)를 식별했습니다. 시저 암호는 각 글자가 알파벳 내에서 일정한 단계 앞으로 이동되는 원리로, 이 작업은 독립적으로 여러 변수를 조작할 수 있어 CoT 성능에 영향을 미치는 요인들을 평가하는 데에 적합합니다.

- **Performance Highlights**: GPT-4 모델이 기본 프롬픗 형태에서는 시저 암호 해독에 대해 정확도가 제로에 가깝지만, CoT 프롬프트를 활용하면 평균적으로 32%의 정확도를 달성했습니다. 또한 모델이 중간 단계를 명시적으로 출력하고 이 단계에서 올바른 정답의 확률을 증가시키는 것이 매우 중요한 것으로 나타났습니다. 일부 실험에서는 출력 확률을 26%에서 70%로 변화시킬 수 있었으며, 프롬프트의 데몬스트레이션 유효성은 크게 중요하지 않다는 결론을 내렸습니다.



### Clustering in pure-attention hardmax transformers and its role in sentiment analysis (https://arxiv.org/abs/2407.01602)
Comments:
          23 pages, 10 figures, 1 table. Funded by the European Union (Horizon Europe MSCA project ModConFlex, grant number 101073558). Accompanying code available at: this https URL

- **What's New**: 이번 연구에서는 하드맥스 셀프 어텐션 및 정규화 서브레이어를 갖춘 트랜스포머의 동작을 분석하여, 레이어 수가 무한대로 증가할 때의 행동을 정밀하게 규명하였습니다. 트랜스포머를 유클리드 공간에서 점들의 진화를 설명하는 이산 시점 동적 시스템으로 보고, 셀프 어텐션 메커니즘을 하이퍼플레인 분리에 기반한 기하학적 관점에서 해석하였습니다. 이를 통해 트랜스포머 입력이 특수한 점들인 리더에 의해 결정되는 클러스터 평형으로 점차 수렴한다는 사실을 밝혀냈습니다.

- **Technical Details**: 분석 대상인 퓨어 어텐션 하드맥스 트랜스포머는 대칭적이고 양의 정부호 행렬 A와 스칼라 파라미터 α로 매개변수화됩니다. 이 트랜스포머는 토큰이라고 불리는 점들에 작용하며, 이 토큰은 초기값으로부터 각 레이어를 거치며 진화합니다. 동적 시스템 관점에서 셀프 어텐션 메커니즘은 토큰이 A에 의해 투영된 방향으로 가장 큰 직교 투영을 가진 토큰들로 끌어당겨지는 것으로 해석되었습니다.

- **Performance Highlights**: 무한 레이어의 경우 토큰은 리더라고 불리는 특수한 토큰들 또는 그 특정한 볼록 결합에 의해 형성된 클러스터 평형으로 수렴함을 증명하였습니다. 리더 토큰은 특정 레이어에서 자기 자신만으로 구성된 인덱스 집합을 가지는 토큰이며, 모든 토큰은 이러한 리더들로 수렴합니다. 이는 감성 분석 문제 해결에 효과적인 완전 해석 가능한 트랜스포머 모델을 제공하여, 의미 없는 단어들을 가장 의미가 있는 리더 단어 주변으로 클러스터링함으로써 `문맥`을 효과적으로 포착할 수 있음을 보여줍니다.



### JailbreakZoo: Survey, Landscapes, and Horizons in Jailbreaking Large Language and Vision-Language Models (https://arxiv.org/abs/2407.01599)
Comments:
          44 pages

- **What's New**: 이번 리뷰 논문은 Large Language Models (LLMs)와 Vision-Language Models (VLMs)에서 '탈옥(jailbreaking)' 문제와 이를 방어하는 메커니즘에 대한 포괄적인 분석을 제공합니다. '탈옥'은 AI 시스템의 윤리적 및 운영적 경계를 고의로 우회하는 행위를 의미합니다. 이 연구는 7가지 유형의 탈옥을 분류하고, 이에 대응하는 방어 전략에 대해 구체적으로 설명하며, 향후 연구를 위한 방향성도 제안합니다.

- **Technical Details**: LLMs와 VLMs는 GPT-3, GPT-4, BERT, CLIP, DALL-E, Flamingo와 같은 모델을 포함하며, 이들은 자연어 처리 및 시각적 상호작용 작업에서 큰 혁신을 가져왔습니다. 그러나 이러한 모델들의 보안 및 윤리적 정렬에는 중요한 문제가 제기되고 있습니다. 본 연구는 LLMs와 VLMs에 대한 탈옥 전략을 세분화하여 7가지 카테고리로 분류하고, 이를 방어하기 위한 다양한 전략을 제시합니다. 탈옥 전략의 예로는 '폭탄 만드는 법'과 같은 민감한 주제를 AI가 응답하지 않도록 설계된 것을 우회하여 답변을 이끌어내는 경우를 들 수 있습니다.

- **Performance Highlights**: 논문은 탈옥과 방어 메커니즘에 대해 포괄적인 분석을 제공하며, LLM 및 VLM 보안을 향상시키기 위한 통합적 관점을 제시합니다. 또한, 기존 연구에서 다루지 않은 취약점과 방어 전략 간의 상호 작용을 심도 있게 분석하여 LLMs와 VLMs 보안 환경을 이해하는 데 중요한 통찰을 제공합니다. 주요 기여로는 탈옥 전략과 방어 메커니즘의 세밀한 분류 및 이를 통합하는 통합적 시각이 있습니다.



### Understanding Alignment in Multimodal LLMs: A Comprehensive Study (https://arxiv.org/abs/2407.02477)
- **What's New**: Large Language Models (LLMs)의 성능 향상에서 선호도 정렬(preference alignment)은 중요한 요소로 자리잡았습니다. 하지만, 멀티모달 대형 언어 모델(MLLMs)에서는 그 영향이 상대적으로 덜 연구되었습니다. 최근 여러 연구들이 MLLMs를 위한 선호도 데이터셋을 도입하고 Direct Preference Optimization (DPO)와 Proximal Policy Optimization (PPO) 같은 다양한 정렬 방법을 조사했습니다. 그러나 데이터셋, 기본 모델 유형, 정렬 방식의 차이로 인해 이러한 연구에서 보고된 개선의 구체적인 요소가 불분명합니다. 이 논문은 독립적으로 MLLMs의 선호도 정렬의 각 측면을 분석하고, Bias-Driven Hallucination Sampling (BDHS)이라는 새로운 멀티모달 선호도 데이터 생성 방법을 소개합니다.

- **Technical Details**: 선호도 정렬 알고리즘을 offline(DPO)과 online(Online-DPO) 두 그룹으로 분류한 후, offline과 online 방법을 결합하면 특정 시나리오에서 모델 성능이 향상될 수 있음을 보여줍니다. 또한, 다양한 멀티모달 선호도 데이터셋을 검토하고 데이터셋 구성 세부사항이 모델 성능에 미치는 영향을 논의합니다. 논문은 추가적인 주석(annotation)이나 외부 모델 없이 Bidirectional Hallucination Sampling (BDHS)을 사용하여 데이터셋을 생성하는 새로운 방식을 도입하여 멀티모달 모델의 여러 벤치마크에서 경쟁력 있는 성능을 달성할 수 있음을 증명합니다.

- **Performance Highlights**: BDHS는 추가적인 주석이나 외부 모델 필요 없이 기존에 발표된 정렬 작업과 유사한 성능을 보여줍니다. 이를 통해 멀티모달 데이터셋 구축 방법의 혁신적인 변화와 새로운 성능 개선 가능성을 제시합니다.



### Is Your AI-Generated Code Really Secure? Evaluating Large Language Models on Secure Code Generation with CodeSecEva (https://arxiv.org/abs/2407.02395)
Comments:
          arXiv admin note: text overlap with arXiv:2310.16263

- **What's New**: 이 논문에서는 코드 생성과 코드 수정을 수행하는 대규모 언어 모델(LLMs)의 보안 취약점을 종합적으로 평가하고 개선하기 위한 연구를 제시합니다. 이 연구를 지원하기 위해 44가지의 주요 취약점 유형을 다루는 180개의 샘플로 구성된 CodeSecEval 데이터셋을 소개합니다. 이 데이터셋을 통해 현재 모델들의 코드 생성 및 수정 시의 보안 문제를 자동으로 평가할 수 있습니다.

- **Technical Details**: CodeSecEval은 완전하고 실행 가능한 코드와 일련의 테스트 케이스를 포함하며, 수작업 평가와 부정확한 분석 도구에 대한 의존도를 줄입니다. 이를 통해 LLM의 보안 평가를 효율적으로 수행할 수 있습니다. 또한, 7개의 최첨단 코드 LLM을 사용해 코드 생성 및 수정 작업에서의 성능을 평가하였습니다. 연구 결과는 현재 모델들이 코드 생성 및 수정 과정에서 자주 보안 문제를 간과한다는 것을 보여줍니다.

- **Performance Highlights**: 실험 결과, 현재 코드 모델들은 보안 문제를 간과하여 취약한 코드를 생성하는 경향이 있습니다. 이러한 문제를 해결하기 위해 우리는 보안 취약점 관련 정보를 활용한 다양한 전략을 제안하고 검증하였습니다. 특히 특정한 취약점 유형은 모델 성능에 큰 도전을 주며, 실제 응용에서 모델의 효과에 영향을 미칩니다. 우리의 연구는 안전하고 신뢰할 수 있는 모델 배포를 위해 개선된 방법 개발을 목표로 하고 있습니다.



### SafaRi:Adaptive Sequence Transformer for Weakly Supervised Referring Expression Segmentation (https://arxiv.org/abs/2407.02389)
Comments:
          Accepted at ECCV 2024

- **What's New**: SafaRi는 기존 방법들이 필요로 하는 대규모 마스크(annotations)를 필요로 하지 않고, 기존 대비 더욱 효율적인 약지도(weakly-supervised) 부트스트랩 구조를 제안합니다. 특히 SafaRi는 최소한의 박스와 마스크 주석(annotation)만을 사용하여 참조 표현 분할(Referring Expression Segmentation, RES)을 수행합니다. 또한, 이 방법은 미리 볼 수 없는(zero-shot) 시나리오에서 높은 성능을 보입니다.

- **Technical Details**: SafaRi 모델은 크로스모달 융합과 주의 일관성 모듈(Cross-modal Fusion with Attention Consistency module, X-FACt)을 도입하여 이미지와 텍스트의 영역 수준 정렬을 개선하고, 대상 객체의 공간적 위치를 명확히 합니다. 주석이 없는 샘플의 자동 레이블링을 위해서는 새로운 Mask Validity Filtering 절차를 도입하여, 스파셜리 어웨어(zero-shot proposal scoring approach)를 기반으로 한 0-셔트 제안 점수를 사용합니다. 또한, 새로운 부트스트랩 전략을 통해 소수의 주석으로 모델을 반복적으로 훈련시키고, 이를 통해 높은 품질의 의사 마스크(pseudo-mask)를 생성합니다.

- **Performance Highlights**: SafaRi는 RefCOCO+@testA와 RefCOCO+testB 데이터셋에서 기존의 완전 지도학습 방법인 SeqTR 대비 11.7%와 19.6% 높은 성능을 보였습니다. 또한 30%의 주석 데이터만을 사용했을 때, SafaRi는 각각 59.31과 55.83 mIoU의 성능을 보였으며, 이는 SeqTR의 58.93과 55.64 mIoU와 비교할 때 더 나은 성능입니다.



### Cost-Effective Proxy Reward Model Construction with On-Policy and Active Learning (https://arxiv.org/abs/2407.02119)
- **What's New**: 이번 연구는 Reinforcement Learning with Human Feedback (RLHF) 분야에서 비용 효율적인 Proxy Reward Oracle을 설계하기 위한 새로운 접근법을 소개합니다. 특히, 제한된 수의 라벨링 데이터와 한정된 전문가 쿼리 예산을 사용하여 효과적으로 선호도나 보상을 라벨링하는 방법을 제안합니다.

- **Technical Details**: 제안된 접근법은 두 가지 주요 혁신을 포함합니다: (1) On-policy 쿼리 기법을 사용하여 Seed 데이터의 OOD(Out-of-Distribution)와 불균형 문제를 피하고, (2) Active Learning 기법을 적용하여 가장 정보가 풍부한 데이터를 선호도 쿼리에 선택합니다. 이를 통해 소량의 데이터로 평가 모델을 훈련시키고, 이 모델이 많은 양의 선호도 쌍을 라벨링할 수 있도록 합니다.

- **Performance Highlights**: Direct Preference Optimization (DPO) 기법을 사용하여 AlpacaEval2, MMLU-5shot 및 MMLU-0shot에서 약 1%의 평균 성능 향상을 달성했으며, 이 과정에서 전문가 쿼리 비용은 1.7K였습니다. 또한, Active Learning 전략을 통해 추가적인 성능 향상을 이끌어 냈습니다. 예를 들어, MMLU-5shot에서 0.34%에서 0.6%까지, MMLU-0shot에서 0.4%에서 0.53%까지 성능이 향상되었습니다.



### Accompanied Singing Voice Synthesis with Fully Text-controlled Melody (https://arxiv.org/abs/2407.02049)
Comments:
          Working in progress

- **What's New**: MelodyLM은 텍스트만으로 노래를 생성하는 최초의 모델로, 사용자가 가사와 참고 목소리만 입력하면 높은 품질의 노래를 생성할 수 있습니다. MelodyLM은 필요한 정보를 MIDI로 명시적으로 모델링하고, 텍스트와 보컬 프롬프트를 기반으로 순차적으로 보컬 트랙을 생성합니다. 이후, 혼합 컨디셔닝(hybrid conditioning)을 통해 반주 음악을 합성합니다. 이 모델은 기존의 멜로디 관련 정보 입력을 최소화하는 동시에 최대의 컨트롤 플렉서빌리티를 제공합니다.

- **Technical Details**: MelodyLM은 텍스트-보컬-반주로 구성된 3단계 프레임워크를 제안합니다. 첫 번째 단계에서는 MIDI 언어 모델을 통해 텍스트 프롬프트에서 MIDI 노트 이벤트를 생성하며, 두 번째 단계에서는 보컬 코드화 언어 모델을 사용하여 가사와 MIDI 토큰을 정렬하고, Soundstream 오디오 토크나이저를 통해 양자화된 어쿠스틱 유닛을 생성합니다. 세 번째 단계에서는 혼합 컨디셔닝 메커니즘을 통해 텍스트 프롬프트와 보컬 컨디셔닝을 균형 있게 조정하여 반주를 재구성합니다.

- **Performance Highlights**: 실험 결과, MelodyLM은 객관적 및 주관적 평가에서 우수한 성능을 보여주었습니다. 주관적 평점은 MelodyLM이 75.6/100으로 나타났으며, 최고의 기준 모델인 Melodist의 79.8/100에 근접한 성능을 보였습니다.



### Simple Augmentations of Logical Rules for Neuro-Symbolic Knowledge Graph Completion (https://arxiv.org/abs/2407.01994)
Comments:
          12 pages, 15 tables Published in ACL 2023

- **What's New**: 이 논문에서는 Neuro-Symbolic Knowledge Graph Completion (NS-KGC) 모델의 성능 향상을 위해 세 가지 간단한 규칙 세트 보강 방법을 제안합니다. 제안된 보강 방법은 기존 규칙을 납치형(abductive form)으로 변환, 구성 관계의 역형태를 사용한 동등한 규칙 생성, 랜덤 워크(random walks)를 통한 새로운 규칙 제안을 포함합니다. 이를 통해 규칙 세트의 포괄성을 높이고, 실험 결과 최대 7.1 MRR 점수와 8.5 Hits@1 점수 향상을 달성하였습니다.

- **Technical Details**: 이 연구는 RNNLogic와 같은 기존 NS-KGC 모델의 규칙 세트를 기반으로 추가적인 규칙을 제안하여 성능을 향상시키는 것을 목표로 합니다. 첫 번째 개선책으로는 각 연역적 규칙을 대응되는 납치형 규칙으로 변환하는 것입니다. 두 번째로는 구성 관계의 역형태(inverse forms)를 사용하는 동등한 규칙을 생성합니다. 세 번째로는 로컬 랜덤 워크와 이후 PCA 필터링을 통해 독립적으로 추가적인 고품질 규칙을 생성합니다. 이러한 방법들을 통해 기존 규칙 세트의 크기를 크게 증가시키면서, 필터링을 통해 저품질 규칙을 제거하여 효율성을 유지합니다.

- **Performance Highlights**: 네 가지 KGC 데이터셋과 세 가지 NS-KGC 모델에 대한 실험에서, 제안된 보강 방법은 일관되게 KGC 성능을 향상시켰습니다. 보강이 없는 베이스라인과 비교하여 최대 7.1 MRR 점수와 8.5 Hits@1 점수의 향상을 달성했으며, 이러한 보강 방법은 NS-KGC의 규칙 세트에서 표준 관행이 되어야 한다고 제안됩니다. 연구 코드는 GitHub(https://github.com/dair-iitd/NS-KGC-AUG)에서 공개되었습니다.



### Certainly Uncertain: A Benchmark and Metric for Multimodal Epistemic and Aleatoric Awareness (https://arxiv.org/abs/2407.01942)
Comments:
          26 pages

- **What's New**: 이번 논문에서는 비전-언어 AI 시스템(Vision-Language AI Systems)에서 발생하는 불확실성을 다루기 위한 분류 체계를 제시합니다. 특히, 정보 부족으로 인한 Epistemic Uncertainty와 본질적으로 예측 불가능한 Aleatoric Uncertainty를 구분하며, 더 세부적인 카테고리들을 탐구합니다. 이 분류 체계를 바탕으로 'CertainlyUncertain'라는 벤치마크 데이터셋을 새로 생성하였습니다. 이는 178K개의 VQA(Visual Question Answering) 샘플을 대조 쌍으로 포함하고 있습니다.

- **Technical Details**: 데이터셋은 두 가지 방법으로 생성되었습니다. 첫째, 원래 답변 가능한 질문을 답변 불가능하게 만들기 위해 이미지를 인페인팅(Inpainting)하는 방법을 사용했습니다. 둘째, 이미지 캡션을 사용하여 대형 언어 모델(Large Language Models)로부터 답변 가능한 질문과 답변 불가능한 질문을 생성했습니다. 추가적으로, 기존 메트릭(metric)의 한계를 극복하기 위해 정확도와 캘리브레이션 오류(Calibration Error)와 잘 연관된 새로운 메트릭인 Confidence-Weighted Accuracy를 소개합니다.

- **Performance Highlights**: 이번 연구는 AI 시스템이 불확실성을 인지하고 처리하는 능력을 향상시키는 데 기여하며, 새로 도입된 메트릭은 AI 시스템의 신뢰성과 정확성을 높이는 잠재력을 가집니다.



### SoP: Unlock the Power of Social Facilitation for Automatic Jailbreak Attack (https://arxiv.org/abs/2407.01902)
- **What's New**: 이 논문은 LLMs(Large Language Models)의 악용 가능성에 대한 우려를 해결하기 위해 SoP(SoP, Social facilitation)라는 새로운 프레임워크를 소개합니다. SoP는 악성 공격을 감지하고 방어하기 위한 red-teaming 전략을 채택하여, LLM의 안전성을 강화할 수 있도록 설계되었습니다.

- **Technical Details**: SoP 프레임워크는 기본적으로 '사회 촉진'이라는 개념에 영감을 받아, 타겟 LLM의 안전장치를 회피하기 위한 다수의 'jailbreak' 캐릭터를 자동으로 생성하고 최적화합니다. 기존의 연구와 달리, SoP는 오픈 소스 LLM을 사용하여 seed jailbreak 템플릿 없이도 콜드 스타트 시나리오에서 유효한 jailbreak 프롬프트를 생성하고 최적화할 수 있습니다. 이로 인해 사람의 전문 지식에 의존하지 않는다는 장점을 가집니다.

- **Performance Highlights**: 실험 결과, SoP 프레임워크는 GPT-3.5-1106과 GPT-4의 안전 정렬을 우회하는 데 있어 각각 88%와 60%의 높은 공격 성공률을 달성했습니다. 또한, 생성된 템플릿의 LLM들 간 전이 가능성과 보류된 악의적 요청에 대한 성능도 광범위하게 평가되었습니다. 이를 바탕으로 SoP 설계에 따른 jailbreak 공격에 대한 방어 전략도 탐구되었습니다.



### GRASP: A Grid-Based Benchmark for Evaluating Commonsense Spatial Reasoning (https://arxiv.org/abs/2407.01892)
- **What's New**: 공간 추론은 인간 인지능력의 중요한 측면이며 많은 실용적인 응용 프로그램에서 중요한 역할을 합니다. 기존의 대형 언어 모델(LLMs)이 텍스트 기반의 공간 설명을 해석하는 것을 평가하는 기존의 벤치마크와는 달리, 본 논문에서는 공간 추론 시나리오에 대해 LLMs가 생성한 직접적인 계획을 평가하는 대규모 벤치마크인 GRASP를 구축하였습니다. GRASP는 에너지 수집 문제를 해결해야 하는 16,000개의 그리드 기반 환경으로 구성됩니다.

- **Technical Details**: GRASP 벤치마크는 160개의 서로 다른 그리드 설정을 사용하는 각각 100개의 그리드 인스턴스로 구성되어 있습니다. 이 설정에는 5가지 에너지 분포, 2가지 에이전트 시작 위치 모드 및 2가지 장애물 구성, 3가지 에이전트 제약 조건이 포함됩니다. 그리드 환경 내에서 에이전트는 장애물을 피하면서 에너지를 수집하고 시작 지점으로 돌아가는 임무를 수행합니다. GRASP는 텍스트 처리된 그리드 환경을 직접 통합하여 이를 평가합니다.

- **Performance Highlights**: GRASP 사용 결과, GPT-3.5-Turbo와 GPT-4 같은 고급 LLMs조차도 일관되게 만족스러운 솔루션을 달성하는 데 어려움을 겪는 것으로 나타났습니다. 이는 기존의 공간 추론 벤치마크보다 더 실질적이고 적용 가능한 공간 정보를 사용하고 있다는 점에서 중요합니다.



### Beyond Numeric Awards: In-Context Dueling Bandits with LLM Agents (https://arxiv.org/abs/2407.01887)
- **What's New**: 대규모 언어 모델(LLMs)이 듀얼링 밴딧(Dueling Bandits, DB) 문제에서 의사결정자로서의 성능을 평가했습니다. 특히, GPT-4 Turbo 모델이 주목할 만한 성과를 보였으며, Condorcet 승자를 빠르게 식별할 수 있었습니다. 하지만, LLM들이 수렴하는데 어려움을 겪고 프롬프트 변화에 민감하다는 단점이 있습니다. 이를 해결하기 위해 IF-Enhanced LLM이라는 알고리즘을 도입했습니다.

- **Technical Details**: 듀얼링 밴딧(DB) 문제는 두 개의 팔을 선택하고, 이들 간의 비교 결과(이기거나 지는 것)를 관측하는 문제입니다. LLM의 성능을 평가하기 위해 GPT-3.5-Turbo, GPT-4, GPT-4-Turbo를 기존의 DB 알고리즘과 비교했습니다. IF-Enhanced LLM 알고리즘은 LLM의 in-context decision-making 능력과 기존 DB 알고리즘의 이론적 보장을 결합한 하이브리드 접근법입니다. 이 알고리즘을 통해 LLM의 신뢰성을 높이고, 약한 후회와 강한 후회에서 이론적으로 보장된 성능을 확보할 수 있습니다.

- **Performance Highlights**: GPT-4 Turbo 모델은 단기간 내 Condorcet 승자를 식별하고, 다양한 인스턴스에서 낮은 분산을 보여줌으로써 높은 성과를 냈습니다. 그러나 장기적으로는 탐색 단계에서 과대평가 편향과 착취 단계에서 수렴 기준의 부족 등이 문제점으로 나타났습니다. IF-Enhanced LLM 알고리즘은 여러 프롬프트 시나리오에서도 견고한 성능을 보였습니다.



### Automated Text Scoring in the Age of Generative AI for the GPU-poor (https://arxiv.org/abs/2407.01873)
Comments:
          21 pages, 1 figure

- **What's New**: 최근 연구에서 발전된 GLM (Generative Language Models)을 활용한 자동 텍스트 스코어링(Automated Text Scoring, ATS)에 대해 집중하고 있으며, 주로 API를 통한 접근이 이루어지고 있습니다. 하지만 이것이 투명성과 보안 문제를 일으키고, 효율성이나 커스터마이징에 제한적이라는 문제점이 있습니다. 이번 연구는 소형 오픈 소스 GLM을 사용하여, API 접근 없이도 GPU 가난층(GPU poor)이 활용할 수 있는 ATS의 성능을 분석했습니다. 특히, 소형 GLM이 적절한 성능을 달성할 수 있음을 보여주었으며, 그들의 점수를 설명할 수 있는 피드백 생성 가능성도 탐구했습니다.

- **Technical Details**: 이번 연구에서는 Automated Student Assessment Prize (ASAP) 데이터를 사용하여 8GB 이하의 소형 오픈 소스 GLM 네 개를 Automated Essay Scoring (AES) 및 Automated Short Answer Scoring (ASAS) 작업에 대해 파인 튜닝(fine-tuning)했습니다. 또한, 모델이 제공한 점수를 항목별로 설명하도록 프롬프트를 제공함으로써 피드백 패턴을 질적 분석했습니다.

- **Performance Highlights**: 소형 GLM을 적절히 파인 튜닝했을 때 높은 품질의 점수를 생성할 수 있음을 확인했습니다. 현재의 최첨단(state-of-the-art, SOTA) 벤치마크와 비교하였을 때, 적어도 일부의 피드백은 점수를 설명하는 데에 유의미한 결과를 보였습니다. 그러나 지속적인 평가와 타겟 사용 사례에 대해 더 엄격한 검증이 필요합니다.



### Empathic Grounding: Explorations using Multimodal Interaction and Large Language Models with Conversational Agents (https://arxiv.org/abs/2407.01824)
- **What's New**: 대화형 에이전트에서 청자의 감정 상태에 대한 공감을 포함하는 '공감적 기준 (empathic grounding)' 개념을 도입했습니다. 이는 발화자가 표현한 감정과 청자가 이를 인지하는 공감적 신호를 통해 의사소통이 더 효율적이고 신뢰할 수 있게 만듭니다. 이러한 공감적 기준은 다중 모드(multimodal)로 표현될 수 있으며, 언어와 비언어적 신호를 포함합니다.

- **Technical Details**: 우리의 다중 모드 모델은 사용자 음성 및 얼굴 표정을 입력으로 받아, 큰 언어 모델(LLM)을 사용하여 듣는 에이전트가 공감적 기준 신호를 생성합니다. 이 모델은 사용자의 통증 경험을 인터뷰하는 인간형 로봇을 통한 테스트베드에서 평가되었습니다. 우리의 모델은 공감적 신호를 생성하지 않는 기준 모델과 비교되었습니다.

- **Performance Highlights**: 결과는 공감적 기준이 사용자의 공감, 이해, 감정 지능, 신뢰감 인식을 증가시킨다고 밝혔습니다. 공감적 기준이 내재된 에이전트를 사용한 실험 조건이, 공감적 신호가 없는 조건에 비해 사용자가 더 잘 이해 받고 있다고 느꼈습니다. 또한, 다중 모드 신호를 통해 사용자 감정을 인식하고 이를 통합하는 것이 더 나은 공감적 기준을 제공함을 발견했습니다.



### SPARKLE: Enhancing SPARQL Generation with Direct KG Integration in Decoding (https://arxiv.org/abs/2407.01626)
- **What's New**: SPARKLE은 자연어에서 SPARQL로의 번역을 수행하는 획기적인 엔드-투-엔드 프레임워크입니다. 기존의 멀티 스테이지 접근법 대신 단일 생성을 통해 SPARQL 쿼리를 생성함으로써 이전 스테이지의 오류가 누적되는 문제를 해결하고, 인퍼런스(inference) 시간도 단축합니다.

- **Technical Details**: SPARKLE은 시퀀스-투-시퀀스(sequence-to-sequence) 모델을 통해 SPARQL 쿼리를 생성하며, 지식 베이스의 구조적 정보를 디코딩 과정에서 직접 활용합니다. 이렇게 구조적인 정보를 통합함으로써, 유효한 트리플 패턴(triple pattern)을 생성하는 데 중점을 둡니다. 이는 엔티티와 관계를 생성하는 동안 지식 베이스의 의미적 구조를 정확하게 반영할 수 있습니다.

- **Performance Highlights**: SPARKLE은 SimpleQuestions-Wiki 데이터셋에서 새로운 최고 성과를 기록했으며, LCQuAD 1.0에서도 금 엔티티(gold entities)를 사용하지 않은 모델 중 가장 높은 F1 점수를 얻었습니다. 웹QSP(WebQSP) 데이터셋에서는 다소 낮은 성과를 보였지만, 여전히 엔드-투-엔드 방법 중 최고 Hits@1 점수를 기록했습니다. 추가로, 인퍼런스 속도가 매우 빠르며(1초 미만), 일괄 처리가 가능하여 실제 사용 시 적합합니다.



### On Discrete Prompt Optimization for Diffusion Models (https://arxiv.org/abs/2407.01606)
Comments:
          ICML 2024. Code available at this https URL

- **What's New**: 이 논문은 텍스트에서 이미지로 변환하는 diffusion 모델에서 프롬프트 최적화를 위한 최초의 gradient 기반 프레임워크를 소개합니다. 프롬프트 엔지니어링을 언어 공간상의 불연속 최적화 문제로 형성하며, 이를 해결하기 위해 두 가지 주요 기술적 기여를 합니다: 하나는 관련 있는 단어들로 구성된 동적으로 생성되는 작은 검색 공간을 디자인하는 것이고, 다른 하나는 'Shortcut Text Gradient'를 도입하여 기억 구조와 실행 시간을 일정하게 유지하는 것입니다.

- **Technical Details**: 제안된 프레임워크는 DPO-Diff로 불리며, 이는 자연 언어 공간상의 불연속 최적화 문제로서 프롬프트 엔지니어링을 구성합니다. Diffusion 모델의 샘플링 단계는 메모리와 실행 시간 복잡도를 증가시키기 때문에 이를 줄이기 위해 Gumbel Softmax trick을 사용하여 카테고리 단어 선택을 학습 가능한 부드러운 분포로 연속적으로 완화합니다. 이렇게 획득된 'Shortcut Text Gradient'를 통해 많은 추론 단계를 걸치는 diffusion 모델에서도 효율적인 최적화를 수행할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 DiffusionDB, COCO, ChatGPT와 같은 다양한 소스에서 수집한 도전적인 프롬프트를 통해 실험적으로 검증되었습니다. 그 결과, 인간이 설계한 프롬프트나 이전 방법보다 훨씬 더 우수한 성능을 보여주었으며, 이미지 생성 프로세스의 충실도를 개선하거나 파괴하는 프롬프트를 효과적으로 발견할 수 있었습니다. 특히, 'negative prompts'을 최적화하는 것이 긍정적 프롬프트보다 더욱 효과적이라는 점을 입증했습니다.



### A Review of Large Language Models and Autonomous Agents in Chemistry (https://arxiv.org/abs/2407.01603)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)을 화학 분야에 통합하는 새로운 패러다임을 소개합니다. LLMs는 분자 설계, 합성 경로 최적화, 약물 및 소재 발견을 가속화하는 능력을 보유하고 있습니다. 특히 LLMs와 화학 전용 도구(합성 플래너와 데이터베이스 등)를 결합한 '에이전트'라는 개념이 주목받고 있습니다.

- **Technical Details**: 리뷰에서는 LLMs의 최근 역사, 현재의 능력, 설계, 화학 분야에서의 도전 과제 및 미래 방향을 다루고 있습니다. LLMs는 본래 자연어 처리(NLP)에서 성공 사례를 기반으로 화학 언어(예: SMILES)로 적응하여 합성 예측과 분자 생성 등의 과제를 해결하고 있습니다. LLM 기반 에이전트는 현재 데이터 해석뿐만 아니라 로봇 시스템과의 실험도 가능하게 합니다.

- **Performance Highlights**: 에이전트들은 화학의 다양한 도메인에서 실효성을 입증했으나, 도메인 전용 에이전트를 만드는 것이 나은지, 아니면 일반 에이전트를 만드는 것이 나은지 여부는 아직 불명확합니다. 또한, 자율형 파이프라인 시스템과 '코-파일럿' 시스템 간의 최적 접근법에 대한 논의도 필요합니다. 최근에는 인간이 루프에 포함된 다중 에이전트 시스템의 개발이 대두되고 있으며, 이에 대한 지속적인 연구가 이뤄지고 있습니다.



### AI Governance and Accountability: An Analysis of Anthropic's Claud (https://arxiv.org/abs/2407.01557)
- **What's New**: AI 시스템이 점점 더 많이 사용되고 있는 현대 사회에서, AI 거버넌스(governance)와 책임(accountability) 체계가 더욱 중요해지고 있습니다. 이 논문은 Anthropic의 Claude라는 기초 AI 모델을 중심으로 AI 거버넌스 환경을 분석하고, NIST AI 위기 관리 프레임워크(NIST AI Risk Management Framework)와 EU AI Act를 통해 Claude를 심층 분석합니다. 주요 위협을 식별하고 이를 완화하기 위한 전략을 제안합니다. 또한 AI 시스템 개발 및 배포 과정에서의 투명성, 엄격한 벤치마킹, 포괄적인 데이터 처리 과정의 중요성을 강조하고 있습니다.

- **Technical Details**: 이 논문에서는 Claude라는 대형 언어 모델(LLM)을 주요 대상으로 삼아 여러 AI 거버넌스 프레임워크를 통해 분석합니다. Claude는 사람과 유사한 텍스트를 이해하고 생성할 수 있는 능력을 가진 기초 AI 모델로, 다양한 커뮤니케이션 작업에서 높은 효율성을 보여줍니다. 특히, Claude의 인공 지능 헌법(Constitutional AI) 패러다임을 조사합니다. 이 헌법적 패러다임은 모델의 출력물이 사전에 정의된 윤리적 원칙 및 가치를 준수하도록 설계되었습니다. Claude는 이러한 원칙을 훈련 과정에서 모델이 자체 응답을 비판하고 수정하며, 강화 학습을 통해 AI 생성 피드백을 사용하는 방법으로 적용됩니다.

- **Performance Highlights**: Claude의 주요 성능 요소로는 투명성 강화, 프라이버시 정책에서의 명확성 확립, 환각 및 편향 문제에서의 엄격한 벤치마킹, 데이터 삭제 및 모델 학습 해제 과정에서의 포괄적 해결 절차를 포함합니다. NIST AI 리스크 관리 프레임워크와 EU AI Act를 통해 Claude의 위험 요소를 분석하고, 가능한 위협과 그 심각성을 평가합니다. 이를 통해 사회적 신뢰를 형성하고 책임감 있는 AI 기술 발전을 지원하기 위한 전략적 방향을 설정합니다.



### Leveraging Prompts in LLMs to Overcome Imbalances in Complex Educational Text Data (https://arxiv.org/abs/2407.01551)
Comments:
          17 pages, 5 figures, 3 tables, 2 appendices

- **What's New**: 이 논문에서는 교육 데이터셋의 불균형 문제를 해결하기 위해 Large Language Models (LLMs)에 명제를 도입하는 가능성을 탐구했습니다. 전통적인 머신 러닝(Machine Learning, ML) 모델이 복잡하고 미묘한 데이터를 다룰 때 한계가 있다는 문제를 극복하고자 했습니다. 특히 학생들의 인지 참여(cognitive engagement) 수준이 크게 다양하다는 점이 중요하게 다뤄졌습니다. LLMs에 명제를 추가함으로써 전통적인 ML 모델보다 최대 32% 향상된 F1-score를 기록했으며, 특정 세트에 대해 동작 민감도 분석에서 11.94% 향상된 성능을 보였습니다.

- **Technical Details**: 연구는 assertion-based prompt engineering 기술을 사용한 Iterative - ICL PE Design Process를 통해 진행되었습니다. 이는 전통적인 ML 모델과 명제가 추가된 LLMs를 비교하여 N=135의 샘플을 사용했습니다. 추가적으로, 하위 집합(n=27)에 대한 민감도(sensitivity) 분석을 통해 모델 성능의 변동성을 조사했습니다. 사용된 LLMs는 In-context Learning (ICL)과 Chain-of-Thought (COT) prompting을 활용해 더 미묘하고 상황에 맞는 반응을 생성할 수 있도록 설계되었습니다.

- **Performance Highlights**: LLMs에 명제를 추가함으로써 전통적인 ML 모델 대비 인지 참여 수준에서 32% F1-score 향상을 달성했습니다. 민감도 분석 결과, 특정 세트에서 명제 기반 접근 방식을 도입하여 11.94% 성능 향상을 이루었습니다. 이 향상은 모델의 맥락 이해와 어휘 혼동 해소에 따른 오류 감소로 주로 이루어졌습니다.



### LINGO-Space: Language-Conditioned Incremental Grounding for Spac (https://arxiv.org/abs/2402.01183)
Comments:
          Accepted by AAAI 2024

- **What's New**: 이번 연구에서는 LINGO-Space라는 새로운 확률적 공간-그라운딩 방법론을 제안합니다. 이는 로봇이 인간의 지시를 통해 공간을 식별하고 이를 단계별로 업데이트할 수 있도록 도와줍니다.

- **Technical Details**: LINGO-Space는 설정 가능한 polar distributions를 이용해 참조된 공간의 확률 분포를 식별하고, 후속 참조 표현에 따라 이를 점진적으로 업데이트합니다. 또한, 대형 언어 모델(LLM)을 활용한 시맨틱 파서를 통해 구성적 모호성을 해결하고, 장면 그래프 기반 표현을 통해 참조 모호성을 줄입니다.

- **Performance Highlights**: 평가 결과, polar distributions를 사용한 추정 방법이 기존 방법보다 참조 표현에 의해 묘사된 공간을 효과적으로 그라운딩하며, 후속 표현을 통해 분포를 정확히 세분화할 수 있음을 확인했습니다. 이를 통해, 시뮬레이션 조작 및 실제 4족 로봇 내비게이션 작업에서의 견고성을 입증했습니다.



### Pistis-RAG: A Scalable Cascading Framework Towards Content-Centric Retrieval-Augmented Generation (https://arxiv.org/abs/2407.00072)
- **What's New**: Pistis-RAG는 Greek 신화에서 좋은 믿음과 신뢰를 상징하는 '페스티스'에서 영감을 받아 개발된 대규모 검색-증강 생성(RAG) 시스템 문제를 해결하기 위한 확장 가능한 다중 단계 프레임워크입니다. 이 시스템은 매칭, 사전 순위, 순위 매김, 추론, 통합의 다섯 가지 단계로 구성되어 있으며, 각각의 단계가 대형 언어 모델(LLM)의 선호도 맞추기, 복잡한 연쇄 추론(CoT) 방법 지원, 다중 소스로부터 정보 결합 등의 역할을 합니다.

- **Technical Details**: Pistis-RAG는 전통적인 모델 중심 패러다임을 넘어 콘텐츠 중심 통합(content-centric integration)을 강조합니다. 이는 LLM과 외부 정보 소스 간의 원활한 통합을 추진하여 특정 작업을 최적화합니다. 특별히, 피스티스-RAG는 RAG 시스템의 고유한 비즈니스 시나리오를 고려하여 순위 매김 단계에서 정보 검색 원칙을 통합합니다. 실험에서 MMLU 지표를 기준으로 9.3%의 성능 향상을 보였습니다.

- **Performance Highlights**: 대규모 실험 데이터에서 피스티스-RAG 프레임워크의 확장성을 검증했으며, GitHub을 통해 모델과 코드를 오픈소스할 계획입니다. 기존 방법보다 9.3% 성능이 향상되어 보다 더 개인화되고 문맥에 맞는 결과를 제공합니다.



New uploads on arXiv(cs.IR)

### CoIR: A Comprehensive Benchmark for Code Information Retrieval Models (https://arxiv.org/abs/2407.02883)
- **What's New**: 이번 논문에서는 코드 검색 능력을 평가하기 위해 고도로 설계된 포괄적인 벤치마크인 **코드 정보 검색 벤치마크**(Code Information Retrieval Benchmark, CoIR)를 제안합니다. CoIR는 다양한 도메인과 작업을 아우르는 **열 개**의 세심하게 선정된 코드 데이터셋을 포함하고 있으며, **여덟 개**의 고유 검색 작업을 **일곱 개**의 서로 다른 도메인에서 제공합니다.

- **Technical Details**: CoIR에는 텍스트-코드 검색, 코드-코드 검색, 코드-텍스트 검색 및 하이브리드 코드 검색과 같은 **네 가지** 주요 검색 작업이 포함됩니다. 각 작업은 더욱 세분화되어 코드 대회 검색, 웹 쿼리 코드 검색, 텍스트-SQL 검색, 코드 요약 검색 등 **여덟 가지** 세부 작업으로 나뉩니다. 이를 통해 CoIR는 1K에서 1M 규모의 다양한 데이터셋을 제공하며, 질의와 코퍼스의 평균 토큰 수는 각각 37에서 4.4K, 113에서 1.5K까지 다양합니다.

- **Performance Highlights**: CoIR를 사용하여 9개의 널리 사용되는 검색 모델을 평가한 결과, 최첨단 시스템에서도 코드 검색 작업에서 상당한 어려움을 겪는다는 것을 확인하였습니다. 이를 해결하기 위해 CoIR 벤치마크 프레임워크는 pip을 통해 쉽게 설치 가능하며, MTEB와 BEIR와 동일한 데이터 스키마를 공유하여 다양한 벤치마크 평가를 통해 연구자들이 쉽게 통합할 수 있도록 설계되었습니다.



### CRUISE on Quantum Computing for Feature Selection in Recommender Systems (https://arxiv.org/abs/2407.02839)
Comments:
          accepted by QuantumCLEF 2024

- **What's New**: 이번 연구는 고전 컴퓨터가 처리하기 어려운 추천 시스템 문제를 해결하기 위해 양자 컴퓨터를 사용하는 방법을 제안합니다. 특히, 양자 어닐러(Quantum Annealer)를 활용해 추천 알고리즘의 특징 선택(feature selection) 문제를 해결합니다. 기존의 단순 상호 정보(Mutual Information)와 비교하여 반사실적 분석(Counterfactual Analysis)을 통합함으로써 아이템 기반 KNN(Item-KNN) 추천 알고리즘의 성능을 크게 향상시켰습니다.

- **Technical Details**: 이 논문에서는 특징 선택 문제를 제곱 무제약 이진 최적화(Quadratic Unconstrained Binary Optimization, QUBO) 문제로 공식화했습니다. 이후, 반사실적 분석을 적용하여 특징들이 추천 품질에 미치는 영향을 고려했습니다. 반사실적 분석은 특정 요소가 없는 상황을 가정하며 그 요소가 최종 결과에 미치는 영향을 평가하는 인과 분석 도구입니다. 이를 통해, 단순 상호 정보의 결과에 반사실적 분석 결과를 온도 계수(temperature coefficient)로 통합하여 최종 성과를 개선했습니다.

- **Performance Highlights**: 실험 결과, 반사실적 분석을 통합한 방법이 단순 상호 정보만을 사용하는 것보다 Item-KNN 추천 알고리즘의 성능을 크게 향상시켰음을 확인했습니다. 특히, nDCG 성능 지표를 통해 각 아이템 특징의 영향을 분석했으며, 이는 추천 리스트의 품질을 평가하는 데 사용되었습니다.양자 어닐링을 통해 500개의 특징 변수를 직접 처리하는 것은 현재의 양자 컴퓨터 제한으로 인해 어려웠으므로, 먼저 관리 가능한 하위 집합으로 나눈 후 결과를 통합하는 방식을 사용했습니다.



### LANE: Logic Alignment of Non-tuning Large Language Models and Online Recommendation Systems for Explainable Reason Generation (https://arxiv.org/abs/2407.02833)
- **What's New**: 추천 시스템의 설명 가능성은 사용자 신뢰와 만족도를 높이는 데 매우 중요합니다. 대형 언어 모델(LLM)을 활용하면 포괄적인 추천 로직을 생성할 수 있는 새로운 기회를 제공합니다. 하지만 기존 연구에서는 LLM 모델을 추천 작업에 맞추기 위한 파인튜닝(fine-tuning) 과정이 높은 계산 비용과 기존 시스템과의 정렬 문제를 초래하여 실용적인 적용성을 제한합니다. 이번 연구에서는 LLM을 추가적으로 튜닝하지 않고도 온라인 추천 시스템과 정렬하는 효과적인 전략인 LANE을 제안합니다. 이를 통해 비용을 줄이고 설명 가능성을 개선할 수 있습니다.

- **Technical Details**: 우리의 방법은 다음과 같은 주요 구성 요소를 통해 작동합니다: 의미 임베딩(semantic embedding), 제로샷 프롬프트(zero-shot prompting)를 이용한 사용자 다중 선호도 추출(user multi-preference extraction), 의미 정렬(semantic alignment), 체인 오브 사상(CoT) 프롬프트를 사용한 설명 가능한 추천 생성입니다. 항목 ID가 아닌 제목을 임베딩하고 멀티-헤드 어텐션 매커니즘을 사용하여 사용자 선호도의 의미적 특징을 후보 항목의 특징과 정렬합니다.

- **Performance Highlights**: 충분한 실험 결과(성능 비교, 설문 투표, 시각화 사례 포함)는 우리의 방법이 추천 성능을 보장하면서도 이해하기 쉽고 타당한 추천 로직을 제공할 수 있음을 입증합니다. 우리의 프레임워크는 모델 불가지론적(model-agnostic)이며, 고급 LLM을 사용하여 기존의 닫힌 소스 모델이나 제한된 계산 자원을 사용하지 않고도 설명 가능성을 극대화합니다.



### Learning Positional Attention for Sequential Recommendation (https://arxiv.org/abs/2407.02793)
- **What's New**: 이 논문은 순차 추천 작업(sequential recommendation tasks)에서 주목할만한 성능을 보여온 self-attention 기반 네트워크를 개선하기 위해 직관적인 위치 관계를 학습하는 새로운 주의(attention) 모델 PARec 및 FPARec을 제안합니다. 이 모델들은 기존 self-attention 기반 모델보다 성능이 우수한 것으로 입증되었습니다.

- **Technical Details**: 순차 추천 시스템은 사용자의 과거 행동 시퀀스에 기반하여 사용자가 관심을 가질 만한 아이템을 예측합니다. 기존 self-attention 모델은 위치 인코딩(positional encoding)을 통해 토큰 순서를 캡처하려고 했으나, 이 연구에서는 위치 임베딩(positional embedding)이 주로 토큰 간 거리를 캡처한다는 점을 실증적으로 발견했습니다. 이를 바탕으로 PARec 및 FPARec이라는 새로운 모델을 제안하였으며, 이 모델들은 위치 간의 주의 패턴을 학습하는 가중치 행렬(weight matrix)을 도입합니다. FPARec은 주의 행렬을 인수 분해하여(attention matrix factorization) 모델의 파라미터 수를 줄였습니다.

- **Performance Highlights**: 여러 현실 세계 데이터셋을 통해 광범위한 실험을 수행한 결과, 제안된 PARec 및 FPARec 모델은 기존 최첨단 self-attention 기반 순차 추천 모델보다 뛰어난 성능을 보였습니다.



### ECAT: A Entire space Continual and Adaptive Transfer Learning Framework for Cross-Domain Recommendation (https://arxiv.org/abs/2407.02542)
- **What's New**: 이번 논문에서는 ECAT(Entire space Continual and Adaptive Transfer learning)이라고 불리는 새로운 학습 프레임워크를 제안합니다. 이 프레임워크는 샘플 전송(sample transfer)과 표현 지속 전송(representation continual transfer) 두 가지 핵심 요소를 포함하고 있습니다. 특히, Taobao의 실제 산업 데이터셋을 이용한 실험 결과, ECAT는 오프라인 지표에서 최첨단 성능을 보여주었고, CVR(Conversion Rate)을 13.6% 증가시키고, 주문 수를 8.6% 증가시키는 등 Baiyibutie라는 미니 앱에서 주목할 만한 성과를 보였습니다.

- **Technical Details**: ECAT는 크게 두 가지 부분으로 구성됩니다. 첫 번째로, 샘플 전송을 위해 그래프 기반 샘플 선택(graph-guided sample selection)과 도메인 적응 방법(domain adaptation method)을 사용하여 샘플을 선별하는 이단계 방법(two-stage method)를 제안합니다. 두 번째로, 잘 훈련된 모델의 표현을 지속적으로 전송하기 위해 적응형 지식 증류 방법(adaptive knowledge distillation method)을 도입하였습니다. 이 방법을 통해 목표 작업에 유용한 정보만을 선택적으로 통합할 수 있습니다.

- **Performance Highlights**: Taobao의 실제 데이터셋을 기준으로 한 종합 실험에서 ECAT는 오프라인 메트릭에서 최첨단 성능을 구현하였으며, Baiyibutie 미니 앱에서 CVR을 13.6%, 주문 수를 8.6% 증가시키는 성과를 보였습니다. 이는 기존의 샘플 전송 및 파라미터 전송 방법보다 뛰어난 성능을 입증하는 결과입니다.



### Supporting Cross-language Cross-project Bug Localization Using Pre-trained Language Models (https://arxiv.org/abs/2407.02732)
- **What's New**: 이 논문은 프로젝트 및 언어 경계를 초월하는 새로운 버그 로컬라이제이션 기술을 제안합니다. 이는 사전 훈련된 언어 모델(PLM)을 기반으로 하며, 대조 학습(contrastive learning)을 활용해 버그 보고서 및 소스 코드의 표현을 향상시킵니다. 또한 커밋 메시지와 코드 세그먼트를 결합하는 새로운 랭킹 기술을 도입하고, 실제 배포를 위한 모델 크기 축소를 위한 지식 증류(knowledge distillation) 기법을 제안합니다.

- **Technical Details**: 이 기술은 대조 학습을 사용하여 PLM을 미세 조정하고, 더 나은 네거티브 샘플을 생성하는 효과적인 방법을 제안합니다. 또한 코드 세그먼트 및 커밋 메시지 분석을 기존 파일 레벨 검사와 결합하여 버그 로컬라이제이션의 정확도를 높입니다. 지식 증류 기법을 통해 모델 크기를 줄여 CPU에서도 실행 가능하게 합니다. 주목할 만한 점은, 코드 세그먼트 분석을 위한 새로운 샘플링 기법과 네거티브 샘플 생성 기법을 도입했다는 것입니다.

- **Performance Highlights**: 이 기법은 여러 프로젝트와 언어에 걸쳐 일반화할 수 있어, 새로운 코드베이스에서도 효과적으로 버그를 식별할 수 있습니다. 코드 세그먼트 및 커밋 메시지 분석을 통해 버그 로컬라이제이션의 성능을 크게 향상시켰으며, 지식 증류 기법을 통해 비용 효율적인 CPU 솔루션을 제공합니다.



### Reducing False Discoveries in Statistically-Significant Regional-Colocation Mining: A Summary of Results (https://arxiv.org/abs/2407.02536)
- **What's New**: 본 논문은 새로운 알고리즘, MultComp-RCM (Multiple Comparisons Regional Colocation Miner)을 도입하여, 지역적 공동배치 (regional colocation) 패턴을 보다 효율적이고 정확하게 탐색하는 방법을 제안합니다. 제안된 알고리즘은 Bonferroni 교정을 사용하여 다중비교 문제 (Multiple Comparisons Problem)로 인한 잘못된 발견률(false discovery rate)을 줄이고자 합니다.

- **Technical Details**: 본 연구는 공간적 특징 타입의 집합 \\emph{S}, 특정 지역, 그리고 인접 관계가 주어졌을 때, \\emph{C}가 해당 지역에서 통계적으로 유의미한 지역적 공동배치 패턴으로 나타나는 경우를 찾는 것을 목표로 합니다. 기존의 연구에서는 많은 수의 동시 통계적 추론으로 인해 거짓 발견의 위험성과 높은 계산 비용이 발생했습니다. 이를 해결하기 위해 MultComp-RCM은 Bonferroni 교정을 사용하여 거짓 발견률을 줄이며, 계산 비용을 낮추는 방법을 제안했습니다.

- **Performance Highlights**: 이론적 분석, 실험 평가 및 사례 연구 결과는 제안된 방법이 거짓 발견률과 계산 비용을 모두 줄이는 데 효과적임을 보여줍니다. 특히, 실험에서는 1473개의 서로 다른 소매 브랜드와 위치를 포함한 데이터를 사용해 2^1473개의 후보 패턴을 효과적으로 처리하는 능력을 증명하였습니다.



### Reliable Confidence Intervals for Information Retrieval Evaluation Using Generative A.I (https://arxiv.org/abs/2407.02464)
Comments:
          KDD '24

- **What's New**: 본 논문에서는 정보 검색 시스템의 전통적인 평가 방법의 비용 문제를 해결하기 위해, 생성형 인공지능, 특히 대형 언어 모델(LLM)을 활용하여 대규모 관련성 주석을 생성하는 방법을 제안합니다. 하지만, LLM 기반 주석의 시스템적 오류 문제를 해결하기 위해, 예측 기반 추론(prediction-powered inference)과 적합 위험 제어(conformal risk control)를 이용하여 평가 지표 주위에 신뢰 구간(CIs)을 설정하는 새로운 방법을 소개합니다.

- **Technical Details**: 제안된 방법은 소수의 신뢰할 수 있는 주석을 필요로 하며, 이를 통해 생성된 주석의 오류 분포를 통계적으로 분석합니다. 그런 다음, 이를 기반으로 한 신뢰 구간(CIs)을 설정하여 평가 지표의 신뢰성을 높입니다. 예측 기반 추론(PPI)은 예측 값과 실제 값 사이의 오류를 중심으로 신뢰 구간을 설정하는 반면, 적합 위험 제어(CRC)는 질의 및 문서 별로 변동 가능한 신뢰 구간을 제공합니다.

- **Performance Highlights**: 실험 결과, 제안된 신뢰 구간(CIs)은 LLM 기반 평가의 분산과 편향을 정확하게 포착하여, 기존의 경험적 부트스트랩 추정 방법보다 더 나은 성능을 나타냈습니다. 이를 통해 전통적으로 평가가 어려웠던 많은 정보 검색 응용 분야에서 보다 신뢰할 수 있는 평가를 가능하도록 했습니다.



### MeMemo: On-device Retrieval Augmentation for Private and Personalized Text Generation (https://arxiv.org/abs/2407.01972)
Comments:
          Accepted to SIGIR 2024. 6 pages, 2 figures. For a live demo, visit this https URL. Code is open-source at this https URL

- **What's New**: MeMemo는 처음으로 자바스크립트 툴킷을 제공하여 클라이언트 측에서 고밀도 검색(dense retrieval)을 가능하게 합니다. 이를 통해 개인 금융, 교육, 의료 등 데이터 프라이버시가 중요한 분야에서도 복잡한 서버 설치 없이 RAG(Retrieval-Augmented Generation)을 사용할 수 있습니다.

- **Technical Details**: MeMemo는 최신 근사 최근접 이웃 검색 기법인 HNSW를 웹 환경에 맞춰 변형하여 IndexedDB와 Web Workers 등의 최신 웹 기술을 활용합니다. 이를 통해 브라우저 내에서 수백만 개 고차원 벡터의 효율적인 검색이 가능합니다. 사용자는 npm install mememo 명령어 하나로 MeMemo를 설치하고, 몇 줄의 코드만으로 고밀도 벡터 인덱스를 생성할 수 있습니다.

- **Performance Highlights**: MeMemo는 기존 서버 기반 구조에 비해 데이터 프라이버시와 사용 편의성을 크게 향상시킵니다. 예제 애플리케이션인 RAG Playground는 클라이언트 측 검색을 통해 대화형 학습과 빠른 프로토타이핑을 가능하게 합니다. 통합된 웹 ML(Web Machine Learning) 기술들과의 원활한 연동을 통해 더욱 나은 성능을 발휘합니다.



### A Survey of Retrieval Algorithms in Ad and Content Recommendation Systems (https://arxiv.org/abs/2407.01712)
- **What's New**: 이번 연구에서는 광고 추천(ad recommendation)과 콘텐츠 추천(content recommendation) 시스템에서 사용되는 효율적인 검색 알고리즘(retrieval algorithms)을 조사합니다. 광고 타겟팅(ad targeting) 알고리즘은 사용자 프로필과 행동 데이터를 활용하여 개인 맞춤형 광고를 제공합니다. 반면에, 유기 검색(organic retrieval) 시스템은 사용자 경험을 개선하기 위해 사용자 선호도에 맞는 콘텐츠를 추천합니다. 이 논문은 두 가지 응용 프로그램을 비교하고, 각각에서 사용되는 가장 효과적인 방법을 설명합니다.

- **Technical Details**: 광고와 콘텐츠 추천 시스템에서 사용하는 주요 검색 알고리즘은 콘텐츠 기반 필터링(content-based filtering), 협업 필터링(collaborative filtering), 그리고 하이브리드 시스템(hybrid systems)입니다. 특히, 추천 시스템에서 널리 사용되는 딥러닝 아키텍처인 투타워 모델(two-tower model)을 상세히 다룹니다. 이 논문은 광고 타겟팅과 유기 검색 시스템의 구현 방법, 훈련(training), 추론(inference) 및 검색 프로세스(retrieval processes) 등을 탐구합니다. 또한, 콜드 스타트 문제(cold start problem), 데이터 품질(data quality), 개인정보 보호 문제(privacy concerns) 등과 관련된 도전과제와 고려사항을 논의합니다.

- **Performance Highlights**: 광고 타겟팅에서 사용되는 역방향 인덱스(inverted index)는 광고를 신속하고 정확하게 매칭하여 사용자에게 맞춤형 광고를 실시간으로 제공하는 데 기여합니다. 사용자 프로필을 기반으로 한 다양한 타겟팅 전략, 예를 들어 나이 타겟팅(age targeting), 성별 타겟팅(gender targeting), 리타겟팅(re-targeting), 키워드 타겟팅(keyword targeting), 행동 타겟팅(behavioral targeting), 컨텍스추얼 타겟팅(contextual targeting) 등의 효과를 실험적으로 입증하였습니다. 특히, 대형 언어 모델(LLMs)을 활용한 키워드 확장은 광고 캠페인의 정밀도와 도달범위를 개선하는 데 도움이 됩니다.



### A survey on the impact of AI-based recommenders on human behaviours: methodologies, outcomes and future directions (https://arxiv.org/abs/2407.01630)
- **What's New**: 이 논문은 추천 시스템과 그 영향력을 분석하는 내용을 다루고 있으며, 특히 소셜 미디어, 온라인 소매, 도시 지도 작성, 생성형 AI 생태계를 중심으로 다룹니다. 기존 연구들의 용어와 방법론이 단편적이며 비체계적이었다는 문제를 해결하기 위해 144개의 다양한 학문 분야의 논문을 체계적으로 검토했습니다. 이를 통해 보다 간결한 분류 체계를 만들고 향후 연구 방향을 제안합니다.

- **Technical Details**: 본 설문 조사는 '질적 종합 리뷰' 방법을 사용하여 수행되었으며, 추천 시스템의 영향을 분석하는 연구를 네 가지 주요 인간-AI 생태계(소셜 미디어, 온라인 소매, 도시 지도 작성, 생성형 AI)로 나눴습니다. 연구 방법론으로는 경험적(empirical), 시뮬레이션 기반(simulation), 관찰적(observational), 통제된 실험(controlled) 등이 있으며, 결과로는 집중(concentration), 모델 붕괴(model collapse), 다양성(diversity), 에코 챔버(echo chamber), 필터 버블(filter bubble), 불평등(inequality), 극단화(polarization) 등을 다뤘습니다. 각 결과는 개인, 항목, 모델, 시스템 차원에서 분석되었습니다.

- **Performance Highlights**: 이 설문 조사는 추천 시스템이 사용자 행동에 미치는 영향을 체계적으로 분석한 첫 번째 연구입니다. 이를 통해 추천 시스템이 사회적, 경제적, 환경적 측면에서 미치는 영향력을 보다 명확히 이해할 수 있도록 했습니다. 예를 들어, 소셜 미디어에서는 극단화 및 에코 챔버 문제를, 도시 지도 작성에서는 이산화탄소 배출 증가 문제를, 생성형 AI에서는 콘텐츠 다양성 부족 문제를 다루었습니다. 이를 통해 학자, 정책 입안자, 기술 기업들에게 중요한 통찰을 제공하며, 향후 연구 및 적용 방안을 제안합니다.



### Potential Renovation of Information Search Process with the Power of Large Language Model for Healthcar (https://arxiv.org/abs/2407.01627)
- **What's New**: 이번 연구에서는 정보 탐색 모델의 6단계(Six Stages of Information Search Model)를 의료 분야에 LLM(Large Language Model) 기반 정보 탐색 프로세스로 통합 및 강화하는 방안을 탐구했습니다. 이 6단계 모델은 정보 과학에서 개인이 정보를 찾는 동안 거치는 시퀀스 단계를 설명하는 기초적인 프레임워크입니다. 이 모델에 LLM 기술을 결합함으로써 각 단계를 최적화하고 특히 의료 분야에서 그 효과를 증대시킬 수 있음을 논의하였습니다.

- **Technical Details**: 6단계 모델은 시작(initiation), 선택(selection), 탐색(exploration), 형성(formulation), 수집(collection) 그리고 발표(presentation)로 구성되어 있습니다. LLM 기술을 적용하면, 질의 해석이 강화되고, 복잡한 의료 데이터베이스에서의 정보 검색이 효율화되며, 문맥적으로 관련성 높은 응답을 제공할 수 있습니다. 이는 의료 정보 검색의 효율성과 정확성을 높이는 데 기여합니다.

- **Performance Highlights**: LLM의 도입을 통해 의료 전문가들은 중요한 데이터를 신속하게 접근할 수 있을 뿐만 아니라, 환자들도 신뢰할 수 있고 개인 맞춤형 건강 정보를 제공받을 수 있게 되었습니다. 이는 더 알아차린(foreseen) 및 효과적인 의료 환경을 구체화하는데 기여합니다.



### SPARKLE: Enhancing SPARQL Generation with Direct KG Integration in Decoding (https://arxiv.org/abs/2407.01626)
- **What's New**: SPARKLE은 자연어에서 SPARQL로의 번역을 수행하는 획기적인 엔드-투-엔드 프레임워크입니다. 기존의 멀티 스테이지 접근법 대신 단일 생성을 통해 SPARQL 쿼리를 생성함으로써 이전 스테이지의 오류가 누적되는 문제를 해결하고, 인퍼런스(inference) 시간도 단축합니다.

- **Technical Details**: SPARKLE은 시퀀스-투-시퀀스(sequence-to-sequence) 모델을 통해 SPARQL 쿼리를 생성하며, 지식 베이스의 구조적 정보를 디코딩 과정에서 직접 활용합니다. 이렇게 구조적인 정보를 통합함으로써, 유효한 트리플 패턴(triple pattern)을 생성하는 데 중점을 둡니다. 이는 엔티티와 관계를 생성하는 동안 지식 베이스의 의미적 구조를 정확하게 반영할 수 있습니다.

- **Performance Highlights**: SPARKLE은 SimpleQuestions-Wiki 데이터셋에서 새로운 최고 성과를 기록했으며, LCQuAD 1.0에서도 금 엔티티(gold entities)를 사용하지 않은 모델 중 가장 높은 F1 점수를 얻었습니다. 웹QSP(WebQSP) 데이터셋에서는 다소 낮은 성과를 보였지만, 여전히 엔드-투-엔드 방법 중 최고 Hits@1 점수를 기록했습니다. 추가로, 인퍼런스 속도가 매우 빠르며(1초 미만), 일괄 처리가 가능하여 실제 사용 시 적합합니다.



### RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs (https://arxiv.org/abs/2407.02485)
- **What's New**: 새로운 연구에서 발표된 RankRAG는 단일 대규모 언어 모델(LLM)을 사용하여 문맥 순위 결정 및 응답 생성 작업을 동시에 수행하는 혁신적인 프레임워크입니다. 기존 모델들과 비교하여, 소량의 순위 데이터만을 포함한 학습 데이터로도 기존의 전문가 순위 모델을 능가하며, 다양한 지식 집중 벤치마크에서 우수한 성능을 보입니다.

- **Technical Details**: RankRAG는 retrieval-augmented generation(RAG) 프로세스를 크게 개선합니다. 기존의 RAG 방식은 질문에 대해 외부 데이터베이스에서 top-k 문맥을 검색하여 LLM이 이를 읽고 응답을 생성하는 방식입니다. 그러나 현재의 RAG 파이프라인은 다량의 문맥을 효과적으로 처리하지 못하는 한계가 있습니다. RankRAG는 단일 LLM을 instruction-tuning하여 문맥 순위 결정 및 응답 생성을 동시에 가능하게 하며, 일부 순위 데이터를 포함한 학습이 surprisingly 좋은 결과를 도출합니다.

- **Performance Highlights**: Llama3-RankRAG 모델은 Llama3-ChatQA-1.5 및 GPT-4 모델을 9개의 일반 지식 집중 벤치마크에서 크게 능가했습니다. 또한, 바이오메디컬 도메인에서의 5개 RAG 벤치마크에서도 instruction fine-tuning 없이 GPT-4와 비슷한 성능을 보였습니다.



### Towards Training Music Taggers on Synthetic Data (https://arxiv.org/abs/2407.02156)
Comments:
          6 pages, 3 figures, accepted to 21st International Conference on Content-based Multimedia Indexing (CBMI) 2024, code available this https URL

- **What's New**: 최신 음악 태깅 시스템들은 주로 대량의 주석 데이터에 의존합니다. 그러나 소량의 주석이 있는 경우, 인공적으로 생성된 음악이 태깅 시스템을 개선하는 데 어느 정도 도움이 될 수 있는지 조사합니다. 이를 위해 GTZAN 데이터셋의 분류 체계를 따르면서 원본보다 10배 더 큰 볼륨을 가진 GTZAN-synth라는 합성 데이터셋을 공개했습니다.

- **Technical Details**: 단순히 합성 데이터를 GTZAN의 학습 분할에 추가하는 것은 성능 향상으로 이어지지 않는다는 것을 발견했습니다. 대신, 도메인 적응(domain adaptation), 전이 학습(transfer learning) 및 파인 튜닝(fine-tuning) 전략을 조사했으며, 이 두 가지 옵션이 정확도 증가로 이어진다는 결론을 내렸습니다. MusicGen 모델을 사용하여 GTZAN-synth 데이터를 생성했으며, 장르별로 텍스트 프롬프트를 이용해 다양한 음악을 생성했습니다.

- **Performance Highlights**: 도메인 적응과 전이 학습, 파인 튜닝을 통해 GTZAN 데이터셋 분류 작업에서 성능 향상을 관찰했습니다. 이는 합성 데이터를 사용한 첫 번째 연구 방향을 제시하며, 향후 연구에서 잠재적인 이점을 갖고 있습니다.



### Joint-Dataset Learning and Cross-Consistent Regularization for Text-to-Motion Retrieva (https://arxiv.org/abs/2407.02104)
- **What's New**: 이 논문에서는 스켈레톤 시퀀스의 3D 인간 모션 데이터를 자연어 텍스트 설명에 근거해 검색하는 새로운 텍스트-모션 검색 작업에 대해 논의합니다. 텍스트-모션 모델의 강력한 학습을 위해 불충분한 데이터를 해결하고자, 다수의 텍스트-모션 데이터셋을 동시에 학습하는 Joint-Dataset Learning과 새로운 Cross-Consistent Contrastive Loss (CCCL)을 제안하였습니다.

- **Technical Details**: CCCL 함수는 학습된 텍스트-모션 공통 공간을 규제하여 모델의 표현 능력을 강화합니다. 또한, 이 논문은 스켈레톤 데이터 시퀀스를 처리하기 위해 공간-시간 주의 기법(spatio-temporal attention)을 활용한 트랜스포머 기반 모션 인코더인 MoT++을 도입합니다. 이를 통해 여러 데이터셋(KIT Motion-Language과 HumanML3D)을 사용하는 학습 시나리오에서 모델의 성능을 개선하고자 합니다.

- **Performance Highlights**: 본 연구에서는 Joint-Dataset Learning과 Cross-Dataset 시나리오에서 자세한 실험을 수행해, 제안된 모듈의 효과와 기존 상태-of-the-art 방법들의 한계를 명확히 밝힐 수 있었습니다.



### Why does in-context learning fail sometimes? Evaluating in-context learning on open and closed questions (https://arxiv.org/abs/2407.02028)
Comments:
          8 pages plus references, 4 main figures, 6 pages of supplementary material

- **What's New**: 최근 LLMs의 인컨텍스트 러닝(in-context learning) 성능을 평가하면서 새로운 질문의 어려움과 난이도에 대한 연구가 발표되었습니다. 이 연구는 새로운 벤치마크를 만들고 과학 분야의 어려운 질문과 다양한 관련성의 컨텍스트를 쌍으로 이루어 실험을 진행했습니다. 특히 흥미로운 점은 주제와 관련된 컨텍스트가 항상 더 큰 도움이 되지 않는다는 것입니다.

- **Technical Details**: 이 논문에서는 물리학과 컴퓨터 과학 분야에서 160개의 고유 질문-응답 쌍을 포함한 데이터셋을 사용했습니다. 각 질문에는 네 가지 유형의 컨텍스트(비교를 위한 무컨텍스트 포함) 중 하나가 제공되었고, GPT-4를 사용하여 생성된 응답과 함께 평가되었습니다. 논문에서 다룬 기술에는 인컨텍스트 러닝(in-context learning)과 Retrieval Augmented Generation (RAG) 시스템이 포함됩니다.

- **Performance Highlights**: 실험 결과에 따르면, GPT-4의 인컨텍스트 러닝 성능은 폐쇄형 질문의 경우 컨텍스트 관련성과 긍정적으로 상관이 있었지만, 개방형 질문의 경우에는 부정적인 상관관계를 보였습니다. 이는 질문의 형식에 따라 모델이 컨텍스트를 활용하는 방식이 다름을 나타냅니다. 또한, 추가적으로 MetaICL 및 NephSAP 데이터셋을 사용하여 폐쇄형 질문의 성능을 비교 분석하였습니다.



### Simple Augmentations of Logical Rules for Neuro-Symbolic Knowledge Graph Completion (https://arxiv.org/abs/2407.01994)
Comments:
          12 pages, 15 tables Published in ACL 2023

- **What's New**: 이 논문에서는 Neuro-Symbolic Knowledge Graph Completion (NS-KGC) 모델의 성능 향상을 위해 세 가지 간단한 규칙 세트 보강 방법을 제안합니다. 제안된 보강 방법은 기존 규칙을 납치형(abductive form)으로 변환, 구성 관계의 역형태를 사용한 동등한 규칙 생성, 랜덤 워크(random walks)를 통한 새로운 규칙 제안을 포함합니다. 이를 통해 규칙 세트의 포괄성을 높이고, 실험 결과 최대 7.1 MRR 점수와 8.5 Hits@1 점수 향상을 달성하였습니다.

- **Technical Details**: 이 연구는 RNNLogic와 같은 기존 NS-KGC 모델의 규칙 세트를 기반으로 추가적인 규칙을 제안하여 성능을 향상시키는 것을 목표로 합니다. 첫 번째 개선책으로는 각 연역적 규칙을 대응되는 납치형 규칙으로 변환하는 것입니다. 두 번째로는 구성 관계의 역형태(inverse forms)를 사용하는 동등한 규칙을 생성합니다. 세 번째로는 로컬 랜덤 워크와 이후 PCA 필터링을 통해 독립적으로 추가적인 고품질 규칙을 생성합니다. 이러한 방법들을 통해 기존 규칙 세트의 크기를 크게 증가시키면서, 필터링을 통해 저품질 규칙을 제거하여 효율성을 유지합니다.

- **Performance Highlights**: 네 가지 KGC 데이터셋과 세 가지 NS-KGC 모델에 대한 실험에서, 제안된 보강 방법은 일관되게 KGC 성능을 향상시켰습니다. 보강이 없는 베이스라인과 비교하여 최대 7.1 MRR 점수와 8.5 Hits@1 점수의 향상을 달성했으며, 이러한 보강 방법은 NS-KGC의 규칙 세트에서 표준 관행이 되어야 한다고 제안됩니다. 연구 코드는 GitHub(https://github.com/dair-iitd/NS-KGC-AUG)에서 공개되었습니다.



### AdaCQR: Enhancing Query Reformulation for Conversational Search via Sparse and Dense Retrieval Alignmen (https://arxiv.org/abs/2407.01965)
- **What's New**: AdaCQR는 서로 다른 검색 환경에서 정보 검색 쿼리의 일반화를 높이기 위해 새롭게 제안된 프레임워크입니다. AdaCQR는 용어 기반(term-based) 및 의미 기반(semantic-based) 검색 시스템 모두와 함께 정렬됨으로써 다양한 검색 시스템 전반에서 강력한 성능을 발휘합니다.

- **Technical Details**: AdaCQR는 두 가지 주요 방법을 통해 초월적인 성능을 달성합니다. 첫 번째로, 고급 라벨(superior labels)을 생성하는 평가 모델을 구축하여 LLM(초대형 언어 모델)을 사용한 몇 샷 학습 기법을 도입하였습니다. 두 번째로, 다양한 입력 후보들을 동시에 생성하기 위해 Diverse Beam Search 기법을 이용하여 한 후보는 우수한 성능을 보이게 하고 나머지 후보들은 융합 메트릭을 기반으로 순위를 매깁니다. 이러한 이중 단계 학습 전략을 통해 용어 기반 및 의미 기반 관점에서 모델을 정렬하여 안정성을 유지합니다.

- **Performance Highlights**: AdaCQR는 TopiOCQA와 QReCC와 같은 두 가지 주요 대화형 검색 데이터셋에서 실험한 결과, 기존 방법들보다 훨씬 뛰어난 성능을 입증했습니다. 특히, LLaMA-7B 백본을 미세 조정한 접근법들과 비교해도 T5-base만을 이용하여도 유사한 성능을 보여주었습니다. AdaCQR는 수치적 및 정성적 개선을 통해 기존 방법들을 앞서가는 성능을 발휘합니다.



### LogEval: A Comprehensive Benchmark Suite for Large Language Models In Log Analysis (https://arxiv.org/abs/2407.01896)
- **What's New**: 이 논문은 LogEval이라는 새로운 벤치마크 세트를 소개합니다. LogEval은 대형 언어 모델(LLMs)이 로그 분석 작업에서 얼마나 잘 수행되는지 평가하기 위해 최초로 설계된 종합 벤치마크입니다. 이 벤치마크는 로그 파싱, 로그 이상 탐지, 로그 결함 진단, 로그 요약과 같은 여러 작업을 다룹니다.

- **Technical Details**: LogEval은 4,000개의 공개 로그 데이터를 사용하여 평가를 수행하며, 각 작업에 대해 15개의 다른 프롬프트(prompts)를 사용하여 공정하고 철저한 평가를 보장합니다. 주요 LLM 기술의 영향을 보여주며, 셀프 컨시스턴시(self-consistency)와 몇 가지 샷 컨텍스추얼 러닝(few-shot contextual learning) 측면에 집중합니다. 모델 정량화, 중문-영문 질문 응답 평가 및 프롬프트 엔지니어링(prompt engineering)에 대한 발견도 논의합니다.

- **Performance Highlights**: LogEval의 평가 방법을 통해 LLM들이 로그 분석 작업에서 가진 강점과 한계를 밝혀내어, 다국어 환경에서의 모델 성능과 다양한 프롬프트 전략의 효과성을 평가합니다. 이러한 통찰은 연구자와 실무자에게 중요한 가이드를 제공합니다.



### Investigating Nudges toward Related Sellers on E-commerce Marketplaces: A Case Study on Amazon (https://arxiv.org/abs/2407.01732)
Comments:
          This work has been accepted for presentation at the ACM Conference on Computer-Supported Cooperative Work and Social Computing (CSCW) 2024. It will appear in Proceedings of the ACM on Human-Computer Interaction

- **What's New**: 이 논문은 아마존의 인도, 미국, 독일, 프랑스 등 네 가지 주요 마켓플레이스에서 '연관된 판매자(Related Sellers)'에게 고객을 유도하는 방식을 조사합니다. 주요 발견은 플랫폼이 고객의 선택을 연관된 판매자 쪽으로 유도할 수 있는 다양한 메커니즘을 운영하고 있다는 점입니다. 특히, 고객이 직접 선택할 수 있는 경우와 알고리즘이 선택한 기본 오퍼의 차이를 강조하며, 아마존이 다양한 성능 평가 정책을 통해 연관된 판매자에게 유리한 조건을 제공할 수 있음을 시사합니다.

- **Technical Details**: 본 연구는 웹 스크래퍼(web-scraper)를 설계하여 인도(Amazon.in), 미국(Amazon.com), 독일(Amazon.de), 프랑스(Amazon.fr)의 아마존 마켓플레이스에서 다수의 제품과 판매자에 대한 Buy Box 정보를 수집했습니다. 또한, 데이터 분석을 통해 판매자 성능 메트릭스 및 고객의 구매 선택에 영향을 미치는 요소를 조사했습니다. 특히, 판매자의 평가 수(#Ratings)가 고객 결정에 큰 영향을 미치며, 이는 판매자의 서비스 품질보다는 운영 규모를 반영할 수 있다는 점을 지적합니다.

- **Performance Highlights**: 연구 결과, 아마존의 알고리즘적으로 선택된 오퍼와 고객이 직접 선택한 오퍼 간의 선호도가 크게 다를 수 있음을 발견했습니다. 특히, 아마존이 제공하는 성능 평가 정책이 연관된 판매자에게 유리하게 작용할 수 있으며, 특정 판매자에게 부여된 부정적 피드백 삭제 정책이 고객의 선택에 중대한 영향을 미친다는 점을 확인했습니다. 이로 인해, 고객에게 성능 메트릭스가 수정되어 표시되었을 때 연관된 판매자에 대한 선호도가 거의 절반으로 줄어드는 경향을 보였습니다.



### Multi-Epoch learning with Data Augmentation for Deep Click-Through Rate Prediction (https://arxiv.org/abs/2407.01607)
- **What's New**: CTR 예측 모델에서 'one-epoch overfitting' 현상을 해결하기 위해 새로운 Multi-Epoch learning with Data Augmentation (MEDA) 프레임워크를 소개합니다. 이 프레임워크는 non-continual 및 continual 학습 시나리오에 모두 적합하며, 기존의 deep CTR 모델에 쉽게 통합될 수 있습니다. 다차원 데이터 희박성으로 인한 embedding layer의 overfitting 문제를 해결하고자 합니다. MEDA는 embedding layer와 Multi-Layer Perceptron (MLP) layer를 분리하여 overfitting을 최소화하며, 서로 다른 embedding 공간에서 MLP를 훈련하여 데이터 증강을 달성합니다.

- **Technical Details**: MEDA는 embedding layer의 high-dimensional 데이터 희박성 문제를 해결하기 위해 설계되었습니다. 특정 non-continual MEDA 알고리즘에서는 각 훈련 epoch의 시작 시 embedding layer를 재초기화하여 embedding-data 의존성을 줄입니다. 계속적인 학습 시나리오에서는 embedding-MLP 의존성을 추가적으로 감소시킵니다. 여러 개의 독립적으로 초기화된 embedding layer를 사용하여 각 데이터셋에서 각각 하나의 embedding layer를 선택적으로 훈련합니다. 이는 embedding layer를 한 번만 훈련시켜 overfitting을 최소화하면서 MLP layer를 반복적으로 훈련하여 수렴을 개선합니다.

- **Performance Highlights**: MEDA는 여러 공공 및 비즈니스 데이터셋에서 single-epoch 학습을 능가하는 성능을 일관되게 보여줍니다. 테스트 AUC에서 0.8%에서 4.6%까지의 개선을 관찰할 수 있었으며, 이는 여러 epoch에서 overfitting 없이 계속됩니다. 실제 온라인 광고 시스템에서 MEDA의 성공적인 배치는 A/B 테스트 결과로 입증되었습니다. 또한, MEDA는 완전한 데이터로 한 single-epoch 학습보다 적은 데이터로 더 나은 성과를 보이며, 데이터 증강의 이점을 제공합니다.



### DrugWatch: A Comprehensive Multi-Source Data Visualisation Platform for Drug Safety Information (https://arxiv.org/abs/2407.01585)
Comments:
          10 pages, 14 figures, accepted by ACL 2024 Demo Track

- **What's New**: DrugWatch는 약물의 부작용과 관련된 통계 정보를 제공하는 다중 소스 정보 시각화 플랫폼입니다. 이 플랫폼은 연구자와 실무자가 약물 안전성과 관련된 데이터를 검색, 분석 및 주석할 수 있도록 지원합니다.

- **Technical Details**: DrugWatch는 두 가지 서브 플랫폼으로 구성됩니다: DrugWatch Search와 DrugWatch Annotate. DrugWatch Search는 FAERS와 PubMed와 같은 출처의 통계 정보 시각화를 제공합니다. DrugWatch Annotate는 사용자가 소유한 데이터에 대해 자동 주석 도구를 제공합니다. FLAN-T5, UIE, Mistral-7B 등의 사전 학습된 모델을 통합하여 사용합니다.

- **Performance Highlights**: DrugWatch는 약물 및 부작용에 대한 검색 기능을 제공하며, 사용자 정의 검색 기준에 맞춘 통계 시각화를 지원합니다. 또한, PubMed 관련 문헌을 쉽게 검색하고 문맥 정보를 얻을 수 있습니다. 사용자는 개인 데이터를 시각화하고 다양한 모델의 예측 결과를 비교할 수 있습니다.



### Pistis-RAG: A Scalable Cascading Framework Towards Content-Centric Retrieval-Augmented Generation (https://arxiv.org/abs/2407.00072)
- **What's New**: Pistis-RAG는 Greek 신화에서 좋은 믿음과 신뢰를 상징하는 '페스티스'에서 영감을 받아 개발된 대규모 검색-증강 생성(RAG) 시스템 문제를 해결하기 위한 확장 가능한 다중 단계 프레임워크입니다. 이 시스템은 매칭, 사전 순위, 순위 매김, 추론, 통합의 다섯 가지 단계로 구성되어 있으며, 각각의 단계가 대형 언어 모델(LLM)의 선호도 맞추기, 복잡한 연쇄 추론(CoT) 방법 지원, 다중 소스로부터 정보 결합 등의 역할을 합니다.

- **Technical Details**: Pistis-RAG는 전통적인 모델 중심 패러다임을 넘어 콘텐츠 중심 통합(content-centric integration)을 강조합니다. 이는 LLM과 외부 정보 소스 간의 원활한 통합을 추진하여 특정 작업을 최적화합니다. 특별히, 피스티스-RAG는 RAG 시스템의 고유한 비즈니스 시나리오를 고려하여 순위 매김 단계에서 정보 검색 원칙을 통합합니다. 실험에서 MMLU 지표를 기준으로 9.3%의 성능 향상을 보였습니다.

- **Performance Highlights**: 대규모 실험 데이터에서 피스티스-RAG 프레임워크의 확장성을 검증했으며, GitHub을 통해 모델과 코드를 오픈소스할 계획입니다. 기존 방법보다 9.3% 성능이 향상되어 보다 더 개인화되고 문맥에 맞는 결과를 제공합니다.




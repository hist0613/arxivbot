### MAP-Neo: Highly Capable and Transparent Bilingual Large Language Model Series (https://arxiv.org/abs/2405.19327)
Comments:
this https URL

- **What's New**: 최근에 많은 기관들이 강력한 LLM들을 오픈소스로 공개했지만 주요 세부 사항들이 부족했습니다. 이를 해결하기 위해, 7B 파라미터로 구성된 MAP-Neo를 처음으로 완전한 오픈소스 이중언어 LLM으로 발표했습니다. MAP-Neo는 기존 최첨단 모델들에 필적하는 성능을 자랑합니다.

- **Technical Details**: MAP-Neo는 4.5조의 고품질 토큰으로 처음부터 훈련되었으며, 모든 세부 사항을 공개합니다. 훈련 데이터의 관리 및 청소 파이프라인, 분산 데이터 처리 시스템, 모델 아키텍처, 훈련 코드, 중간 체크포인트 및 평가 프레임워크가 포함됩니다. 특히 NEO Scaling Law를 도입하여 다양한 코퍼스로부터 소싱된 사전 훈련 데이터셋을 최적화했습니다.

- **Performance Highlights**: MAP-Neo는 중국어와 영어 이해력(C-EVAL, MMLU), 수학 능력(GSM8K, MATH), 코딩 능력(HumanEval) 등 다양한 벤치마크에서 우수한 성능을 발휘합니다. 특히 투명성 측면에서 모든 체크포인트를 제공하며, 이는 기존의 다른 투명한 LLM들 대비 최고 점수를 기록했습니다.



### Nearest Neighbor Speculative Decoding for LLM Generation and Attribution (https://arxiv.org/abs/2405.19325)
- **What's New**: 이번 연구에서는 최신 언어 모델을 위한 새로운 준-파라메트릭(semiparametric) 접근 방식인 Nearest Neighbor Speculative Decoding (NEST)를 소개합니다. NEST는 실제 텍스트 조각을 언어 모델의 생성에 포함시키고 그 출처에 대한 기여를 제공할 수 있습니다. 이를 통해 기존 kNN-LM 모델의 한계를 극복하면서도 더 빠른 추론 속도와 자연스러운 텍스트 생성을 달성합니다.

- **Technical Details**: NEST는 각 추론 단계마다 토큰 수준의 검색을 수행하여 준-파라메트릭 혼합 분포를 계산하고, 코퍼스 내에서 유망한 스팬(span) 연속성을 식별합니다. 그런 다음, 검색된 스팬의 접두사를 수락하거나 새 토큰을 생성하는 근사적 추측 디코딩 절차를 사용합니다. 새로운 접근 방식의 세부 사항은 다음과 같습니다: 1) 새로운 'Relative Retrieval Confidence (RRC)' 점수를 사용하여 토큰 추출기의 불확실성을 측정하고, 이를 출력 확률 혼합의 보간 계수로 활용합니다. 2) 'Copy Generator (CoG)'에서 영감을 받아, 스팬의 선택과 함께 해당 스팬을 확률 혼합에 따라 선택합니다. 3) 스팬이 선택되면 혼합 확률에 따라 평가하고, 고확률로 간주되는 접두사만 수락하는 'Relaxed Speculative Decoding'을 수행합니다.

- **Performance Highlights**: NEST는 다양한 지식 집중형 작업에서 베이스 LM과 기존 kNN-LM을 능가하는 뛰어난 성능을 보였습니다. 예를 들어, NEST와 결합한 Llama-2-Chat 70B 모델은 WikiText-103에서 ROUGE-1 점수가 42.3% 향상되고, Biography에서 FActScore가 21.6% 향상되었습니다. 또한, MMLU, Pile-of-Law, TruthfulQA에서 인컨텍스트(in-context) 검색-증강 접근 방식과 경쟁력 있는 성능을 보여줍니다. 중요한 점은 NEST가 생성 속도를 크게 향상시켜 Llama-2-Chat 70B에서는 추론 시간을 1.8배 단축합니다.



### Are Large Language Models Chameleons? (https://arxiv.org/abs/2405.19323)
Comments:
          16 pages,8 figures

- **What's New**: 대형 언어 모델(LLMs)이 자신의 세계관과 성격 경향을 가지고 있는지를 탐구하는 연구입니다. 100만 번 이상의 주관적 질문에 대한 응답 시뮬레이션을 통해, 유럽 사회 조사(ESS)의 실제 데이터와 비교하여 문화적, 연령적, 성별 편향을 강조하고 있습니다.

- **Technical Details**: 연구에서는 LLMs가 주관적 질문에 어떻게 반응하는지, 그리고 실제 설문 조사 데이터와의 차이를 측정하는 방법을 다룹니다. 이에는 가중 평균 계산 및 Jaccard 유사성에서 영감을 받은 새로운 측정 방법도 포함됩니다. LLMs의 편향과 가변성을 분석하기 위해 다양한 프롬프트와 매개변수를 테스트했습니다. 또한 여러 개의 LLMs를 비교하고 실제 설문 조사 데이터와의 유사성 및 차이점을 분석했습니다.

- **Performance Highlights**: 이 연구는 LLMs가 주관적 질문에 대해 실제 사람들과 얼마나 일치하는지, 그리고 그 차이가 다양한 프롬프트와 매개변수에 따라 어떻게 변화하는지를 강조합니다. 예를 들어, 미국 외의 다른 국가에서도 LLMs의 정치적 편향이 양적으로 평가되었습니다. 또한, 다양한 LLMs와의 비교를 통해 모델별로 얼마나 편향이 존재하는지 평가했습니다.



### Expert-Guided Extinction of Toxic Tokens for Debiased Generation (https://arxiv.org/abs/2405.19299)
- **What's New**: 최근 연구는 LLM (Large Language Models)이 텍스트 생성 중 사회적 편견을 나타낼 수 있다는 문제를 지적합니다. 특히, 유해한 프롬프트와 함께 사용될 때 문제가 되는데, 이를 방지하기 위한 새로운 방법론을 소개합니다. EXPOSED (Expert-Guided Extinction of Toxic Tokens for Debiased Generation)은 기존의 데이터 수집, 미세 조정, 그리고 직접 프롬프트 기법의 한계를 극복하고, 유해한 출력을 제거하는 혁신적 접근법을 제안합니다.

- **Technical Details**: EXPOSED는 두 가지 주요 단계로 구성됩니다: 1) 디바이싱 전문가(debiasing expert) 구성 및 2) 분포 재구축(distribution reconstruction). 이 시스템은 우선 풍부한 유해 코퍼스를 사용하여 잠재적인 유해 토큰을 노출시키고, 이를 억제하여 공정한 분포를 생성합니다. 이를 위해 비지도 학습(pre-training)과 음의 로그 가능도(NLL) 손실 함수 등을 활용합니다.

- **Performance Highlights**: EXPOSED의 성능은 다양한 관점에서 평가되었습니다. 독성 프롬프트(open-ended text generation)를 사용한 텍스트 생성에서 독성 수준을 낮추고, 독해(reading comprehension) 테스트에서 고정관념 편견을 억제했으며, 채워넣기 시험(cloze test)에서는 직업에 대한 성별 편향을 검사했습니다. 결과적으로, EXPOSED는 모든 과제에서 독성 및 편견을 줄이면서도 매끄러운 생성과 경쟁력 있는 추론 지연 시간을 유지함을 보여줍니다.



### Integrating Multi-scale Contextualized Information for Byte-based Neural Machine Translation (https://arxiv.org/abs/2405.19290)
Comments:
          Accepted by ACL2024 Findings

- **What's New**: 본 논문은 Byte 기반 토큰화를 활용한 신경 기계 번역(Neural Machine Translation, NMT)에서 멀티 스케일 컨텍스트얼라이제이션(Multi-Scale Contextualization, MSC) 방법을 제안합니다. 이 방법은 기존의 Subword 토큰화 방식의 단점을 극복하고, 특히 다국어 및 도메인 외 번역 시 우수한 성능을 보입니다.

- **Technical Details**: MSC 방법은 입력을 여러 히든 스테이트(hidden state) 차원으로 나눈 후, 각 차원에 대해 서로 다른 크기의 컨텍스트를 학습합니다. 이후 어텐션 모듈(attention module)을 통해 다중 스케일의 컨텍스트 정보를 동적으로 통합합니다. 이를 위해 1-D 합성곱 신경망(CNN)을 활용해 국소적인 문맥 정보를 통합하며, 각 부분은 CNN의 커널 크기에 따라 다른 범위의 컨텍스트 정보를 학습하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, MSC 방법은 기존의 Subword 기반 및 다른 Byte 기반 방법보다 다국어 및 도메인 외 번역 시 우수한 성능을 보였습니다. 특히, 언어 및 데이터셋에 걸쳐 탁월한 적응성을 보여줍니다.



### MASSIVE Multilingual Abstract Meaning Representation: A Dataset and Baselines for Hallucination Detection (https://arxiv.org/abs/2405.19285)
- **What's New**: 새로운 MASSIVE-AMR dataset를 소개합니다. 이 데이터셋은 84,000개 이상의 text-to-graph annotations를 포함하고 있으며, 50여 개의 다양한 언어로 번역된 1,685개의 정보 탐색 질문들의 AMR 그래프를 제공합니다. 이 데이터셋은 현재까지 가장 크고 다양한 AMR 자료집입니다.

- **Technical Details**: MASSIVE-AMR은 MASSIVE 데이터셋에서 1685개의 QA 발언을 선택하고 이를 다양한 언어로 번역했습니다. 그런 다음 이 발언들에 대해 수작업으로 AMR 그래프를 작성했습니다. 이 데이터셋은 대부분의 기존 자원보다 20배 크고 5-6배 많은 언어를 다룹니다. 이를 통해 SPARQL parsing 및 다국어 구조적 parsing에서 모델의 신뢰도를 평가하고, LLMs (Large Language Models)을 사용하여 지식베이스 질문 응답에서 관계 환각(hallucination) 탐지를 수행했습니다.

- **Performance Highlights**: MASSIVE-AMR을 사용한 실험 결과, SPARQL query 생성에서 모델 신뢰도를 높이는 데 AMR이 유용하다는 점이 밝혀졌습니다. 실험 결과는 LLM을 사용한 다국어 구조적 parsing과 SPARQL 관계 환각(hallucination) 탐지에서 지속적인 문제를 해결하는 데 도움을 주었습니다.



### PediatricsGPT: Large Language Models as Chinese Medical Assistants for Pediatric Applications (https://arxiv.org/abs/2405.19266)
Comments:
          A Technical Report on a Powerful Chinese Medical Large Language Model

- **What's New**: 최초의 중국 소아과 LLM(대형 언어 모델) 보조 시스템 PediatricsGPT가 개발되었습니다. 이 시스템은 소아과 진료의 다양한 요구를 충족하기 위해 30만 개 이상의 다중 작업 지침 데이터셋인 PedCorpus를 기반으로 구축되었습니다. PedCorpus는 소아과 교과서, 가이드라인 및 지식 그래프 리소스를 포함하며, PediatricsGPT는 체계적이고 견고한 학습 파이프라인을 통해 훈련되었습니다.

- **Technical Details**: 학습 과정은 연속 사전 학습, 전체 매개변수 감독 미세조정(Supervised Fine-Tuning, SFT), 인간 선호도 정렬 및 매개변수 효율적인 2차 SFT로 구성됩니다. 특히, 내부와 외부에서 주입된 지식 간의 불일치를 완화하기 위해 하이브리드 지침 사전 학습 메커니즘 및 저순위 적응(LoRA)을 활용해 의료 일반 지식 스키마와 소아과 전문 지식을 통합합니다. 또한, 인간적인 반응을 향상시키기 위한 Direct Following Preference Optimization(DFPO)도 도입되었습니다.

- **Performance Highlights**: 소아과 및 공공 벤치마크에서 PediatricsGPT는 기존의 오픈 소스 중국 의료 LLM 및 기준 모델보다 우수한 성능을 보여주었습니다. 세 가지 실제 소아과 작업에 대해 실시한 광범위한 실험에서는 PediatricsGPT가 GPT-3.5-turbo와 비교하여 경쟁력 있는 성능을 나타냈습니다.



### AlchemistCoder: Harmonizing and Eliciting Code Capability by Hindsight Tuning on Multi-source Data (https://arxiv.org/abs/2405.19265)
Comments:
          Preprint with 20 pages and 20 figures. Source code and models at this https URL

- **What's New**: AlchemistCoder가 소개되었습니다. 이는 멀티소스 데이터를 활용하여 코드 생성 및 일반화 능력이 향상된 새로운 Code LLM(대규모 언어 모델)입니다. 특화된 데이터 프롬프트인 'AlchemistPrompts'를 통해 다양한 데이터 소스의 불일치를 해소하고 코드 이해 능력을 개선합니다.

- **Technical Details**: 기존의 단일소스 데이터의 한계를 극복하기 위해 멀티소스 데이터를 통합하여 파인튜닝을 수행하였습니다. 'AlchemistPrompts'는 데이터 소스별 프롬프트를 전달하여 다중 소스 코드 코퍼스의 불일치를 조화롭게 하며, 데이터 구성 프로세스를 코드 이해 작업에 포함시켜 모델의 성능을 향상합니다.

- **Performance Highlights**: AlchemistCoder는 HumanEval과 MBPP와 같은 주요 코드 벤치마크에서 동일 크기(6.7B/7B) 모델 중 가장 우수한 성과를 보여줍니다. 또한, MMLU, BBH, GSM8K 등의 다양한 테스트에서도 일반 코드 LLM의 능력 향상을 입증합니다.



### Weak-to-Strong Search: Align Large Language Models via Searching over Small Language Models (https://arxiv.org/abs/2405.19262)
- **What's New**: 이번 연구에서는 대형 언어 모델의 정렬 문제를 해결하기 위해 $	extit{weak-to-strong search}$를 소개합니다. 이 접근법은 조정되지 않은 대형 모델에서 샘플링을 하면서 작은 튜닝 모델과 기본 모델 간의 로그 가능성 차이를 최대화하는 테스트 시간 탐색으로 대형 모델의 정렬을 정의합니다.

- **Technical Details**: 구체적으로, 이 방법은 대형 모델을 직접 튜닝하지 않고 작은 튜닝 모델과 비튜닝 모델 사이의 로그 가능성 차이를 보상과 비평으로 사용합니다. Chunk-level Beam Search (CBS)라는 빔 탐색 변형을 소개하여 제시된 탐색 목표를 최적화합니다. 이 탐색 목표를 최적화하기 위해 CBS는 대형 모델에서 샘플링하고 작은 모델로 평가한 상태를 확장하는 절차를 교대로 수행합니다.

- **Performance Highlights**: 실험 결과, 다양한 작업에서 weak-to-strong search의 유연성을 입증했습니다. 특히, 제어된 감정 생성과 요약 작업에서 작은 언어 모델(gpt2)을 사용하여 훨씬 큰 언어 모델(gpt2-xl, Llama-2, Llama-3)의 정렬을 효과적으로 향상시켰습니다. 또한, 지침 따르기 벤치마크인 AlpacaEval 2.0에서, off-the-shelf 작은 모델(zephyr-7b-beta 및 그 비튜닝 버전)을 재사용하여 큰 모델의 성능을 크게 향상시켰습니다.



### Faster Cascades via Speculative Decoding (https://arxiv.org/abs/2405.19261)
- **What's New**: 새로운 연구는 대형 언어 모델(LMs)의 추론 효율을 개선하기 위한 새로운 'speculative cascading' 기술을 제안합니다. 이 기술은 'cascades'와 'speculative decoding'의 장점을 결합하여, 전자가 '어려운' 입력에 대해서만 큰 모델을 호출하는 반면, 후자는 추론 속도를 높이는 방식으로 동작합니다.

- **Technical Details**: 제안된 접근법은 작은 모델을 사용하여 자동 회귀(decoding)로 토큰 초안을 생성하고, 대형 모델이 병렬로 이를 검증하여 초안의 유효성을 판별하는 방식을 사용합니다. 이를 통해 'deferal rule'을 최적화된 방식으로 실행하며, 손실이 있더라도 예측 분포를 모방하는 일반적인 'speculative execution' 방법을 따릅니다.

- **Performance Highlights**: 이 기술을 T5 모델을 사용하여 벤치마크 언어 작업에 적용한 결과, 제안된 speculative cascading 방식이 기존의 cascading과 speculative decoding 방식에 비해 더 우수한 비용-품질 균형을 제공합니다.



### Lower Bounds on the Expressivity of Recurrent Neural Language Models (https://arxiv.org/abs/2405.19222)
- **What's New**: 이 논문은 최근에 높은 성과를 보이고 있는 대형 신경망 언어 모델(neural language models, LMs)의 계산 능력을 더 깊이 이해하기 위해 작성되었습니다. 특히, 이 연구는 RNN 기반 언어 모델들이 주어진 정규 언어 모델(regular language models, LMs)을 정확하게 표현할 수 있는지를 조사합니다. 이를 통해 비결정적 확률 유한 오토마타(non-deterministic probabilistic finite-state automata, PFSAs)와 약하게 동등한 RNN LMs가 존재함을 증명합니다.

- **Technical Details**: 기존 연구들은 주로 신경망 LMs의 표현 능력(representational capacity)을 공정 언어(formal languages)를 인식할 수 있는 능력으로 평가했습니다. 그러나 이 논문은 RNN 기반 언어 모델들이 확률적 FSA(probabilistic FSAs)와의 연결을 통해 정규 언어 LMs를 표현할 수 있는지를 새롭게 탐구합니다. 이를 위해 Elman RNNs 엘만 RNN(LMs)과 sparsemax 및 softmax 프로젝션 함수를 사용한 신경망 모델을 이용해 정확한 표현 또는 임의의 근사를 할 수 있음을 보입니다.

- **Performance Highlights**: 실험 결과, sparsemax 함수를 사용하는 Elman LMs는 어느 정규 언어 모델이든 정확하게 표현할 수 있으며, softmax 프로젝션 함수를 사용하는 Elman LMs는 임의의 정규 언어 모델을 임의의 수준으로 근사할 수 있음을 보였습니다. 이는 기존의 결정적 PFSA와 RNN LMs의 관계를 확장해 비결정적 PFSAs로까지 넓힌 것입니다.



### WRDScore: New Metric for Evaluation of Natural Language Generation Models (https://arxiv.org/abs/2405.19220)
- **What's New**: 자연어 생성 (Natural Language Generation, NLG)의 새로운 평가 방법으로 'Word Rotator’s Distance Score (WRDScore)'를 제안합니다. 이 메트릭은 Java 메소드 명 예측 문제에 특히 적합하며, 기존의 ROUGE와 같은 단순한 메트릭들이 잡아내지 못하는 문맥과 의미의 미묘한 차이를 포착할 수 있습니다.

- **Technical Details**: WRDScore는 출력값이 0과 1 사이로 정규화되며, 정확도 (precision)와 재현율 (recall)을 계산할 수 있습니다. 기존의 BERTScore와 달리, WRDScore는 의존하지 않고도 더 나은 성능을 보여줍니다. 또한, WRDScore는 다양한 단어 임베딩 (word embeddings) 기법들을 활용할 수 있습니다. 여기에는 ELMo, BERT와 같은 문맥적 임베딩과 Word2Vec, GloVe와 같은 비문맥적 임베딩이 포함됩니다.

- **Performance Highlights**: WRDScore는 토큰과 토큰 간의 유연한 비교를 위해 '최적 수송 행렬 (Optimal Transport Matrix)'을 활용합니다. 이는 문맥을 측정에 도입하면서도 출력 점수의 문맥 의존성을 완화시킵니다. 이러한 특성 덕분에 BERTScore와 같은 고급 NLG 메트릭보다도 더 우수한 성능을 발휘합니다. WRDScore는 BERTScore보다 일반화되고, BERTScore의 엄격한 가정을 완화시킨 버전으로 볼 수 있습니다.



### DGRC: An Effective Fine-tuning Framework for Distractor Generation in Chinese Multi-choice Reading Comprehension (https://arxiv.org/abs/2405.19139)
- **What's New**: 이번 연구는 중국어 독해에 대한 다지선다형 문제에서의 자연 질문 오답지 생성(NQDG)을 위한 새로운 정밀 조정 프레임워크인 DGRC를 소개합니다. 이 프레임워크는 하드 체인-오브-생각(hard chain-of-thought), 다중 작업 학습(multi-task learning), 생성 마스크 패턴(generation mask patterns) 등 세 가지 주요 구성 요소를 포함합니다. 실험 결과, DGRC는 BLEU 점수가 2.5배 이상 향상되는 등 성능이 크게 향상됨을 보여줍니다.

- **Technical Details**: DGRC는 세 가지 주요 구성 요소로 나뉘어집니다: 하드 체인-오브-생각(hard CoT), 다중 작업 학습, 생성 마스크 패턴입니다. 하드 CoT 메커니즘은 모델이 오답지를 생성하기 전에 정답을 추론하게 하여 성능을 극대화합니다. 다중 작업 학습에서는 질문 응답과 오답지 생성을 모두 포함하며, 모델이 구체적 지식을 반영하도록 합니다. 생성 마스크 패턴은 모델이 긴 텍스트와 문맥을 고려하여 적절한 오답지를 생성하는 데 도움을 줍니다.

- **Performance Highlights**: DGRC는 BLEU 점수에서 2.5배 이상의 향상을 기록하며, 이는 오답지 생성 성능이 매우 높음을 나타냅니다. 하드 체인-오브-생각 메커니즘은 모델의 입력 길이를 적절히 유지하면서 명시적으로 추론을 안내하여 성능을 크게 향상시킵니다. 이는 특히 중국어 다지선다형 독해 문제에서 고품질의 오답지를 생성하는 데 효과적입니다.



### PathReasoner: Modeling Reasoning Path with Equivalent Extension for Logical Question Answering (https://arxiv.org/abs/2405.19109)
Comments:
          Accepted by ACL 2024

- **What's New**: 논리적 추론(Logical Reasoning) 작업에서 성능을 강화하는 PathReasoner 아키텍처를 제안합니다. 기존 대형 언어 모델들(예: ChatGPT, PaLM 2)이 이 작업에서 낮은 성능을 보이는 것을 해결하기 위해, 논리적 샘플을 추론 경로(reasoning paths)로 변환하여 모델링하는 방식을 도입했습니다.

- **Technical Details**: PathReasoner는 논리적 샘플의 다양성을 확장하기 위해 등가적 논리 공식을 활용한 원자 확장(atom extension) 전략을 제안합니다. 모델 측면에서는, Transformer 스타일 블록을 쌓고, 경로-주의(path-attention) 모듈을 도입하여 원자 내부 및 원자 간의 고차 확산 전략(high-order diffusion strategy)을 모델링합니다. 논리 규칙 폼(predefined logical rule forms)에 기반하여 각 문장을 원자로 표현하고, 이를 입력으로 하여 추론 경로와 신뢰 점수를 형성합니다.

- **Performance Highlights**: PathReasoner는 두 가지 논리적 추론 벤치마크에서 경쟁력 있는 성능을 발휘하고 있으며, 일반화 능력(generalization abilities)에서도 뛰어난 성과를 보여줍니다.



### Faithful Chart Summarization with ChaTS-P (https://arxiv.org/abs/2405.19094)
Comments:
          To be published in the proceedings of the 2024 Annual Meeting of the Association for Computational Linguistics

- **What's New**: 이번 연구에서는 새로운 차트 요약 평가 지표인 CHATS-CRITIC을 소개합니다. 이 지표는 참조 없이 차트 요약의 충실성을 평가하며, 차트 이미지로부터 테이블 데이터를 추출하고, 이를 기반으로 요약 문장별로 평가합니다. 또한, CHATS-PI라는 파이프라인을 도입하여 CHATS-CRITIC을 활용해 요약 문장의 충실도를 점검하고 순위를 매깁니다.

- **Technical Details**: CHATS-CRITIC은 이미지에서 텍스트로 전환하는 모델을 사용하여 차트로부터 테이블 데이터를 복구하고, 이를 기반으로 '표 진술 모델(table entailment model)'이 요약 문장별로 점수를 매기는 방식입니다. CHATS-PI는 여러 후보 요약을 생성하고, CHATS-CRITIC을 통해 요약 문장을 다듬으며, 충실도 점수를 계산하여 최종 요약을 선택합니다. 이 과정에서 TAPEX, TAPAS-CS 등의 테이블 전문 모델이나 PALM-2 같은 대규모 언어 모델을 활용합니다.

- **Performance Highlights**: CHATS-CRITIC은 인간 평가와의 일관성이 더욱 높게 나타났으며, CHATS-PI는 두 가지 유명한 차트 요약 데이터셋에서 최신 성능을 기록하였습니다. CHATS-PI 파이프라인을 사용하여 기계 학습 모델의 후보 요약을 수리하고 순위를 매겨 최고의 요약을 출력할 수 있습니다.



### Multi-stage Retrieve and Re-rank Model for Automatic Medical Coding Recommendation (https://arxiv.org/abs/2405.19093)
Comments:
          Accepted to NAACL 2024 -- camera-ready version

- **What's New**: 국제질병분류(ICD)는 다양한 질병 및 상태를 포괄하는 결정적인 의료 분류 시스템으로, 의료 기록에 ICD 코드를 할당하여 표준화된 문서화 및 건강 상태 관리를 용이하게 합니다. 이 논문에서는 새로운 ICD 인덱싱 해결책으로 'retrieve and re-rank' 프레임워크를 제안합니다. 이 프레임워크는 하이브리드 이산 검색 방법(BM25)을 통해 후보 ICD 코드를 검색하고, 대비 학습을 사용하여 정확한 예측을 수행합니다.

- **Technical Details**: 이 프레임워크는 다단계 검색 및 재정렬 과정으로 구성됩니다. 우선 전자 건강 기록(EHR) 보조 지식과 BM25를 활용해 고품질 후보를 검색합니다. 최종 단계에서는 코드 공출현을 지침으로 하는 대비 학습 모델을 사용해 후보 코드를 재정렬합니다. 이를 통해 임상 노트와 긍정적 ICD 코드를 함께 묶어 더 정확한 예측을 할 수 있도록 합니다.

- **Performance Highlights**: 제안된 방법은 MIMIC-III 벤치마크에서 여러 측정 기준에 대해 최신의 성능(State-of-the-Art)을 달성했습니다. 이 프레임워크는 일반적인 다중 라벨 텍스트 분류 문제로 접근하는 기존의 방법들과 달리, 추천 문제로 개념화하여 다중 단계를 거쳐 ICD 코드를 추출하고 재정렬하여 더 정확한 결과를 도출합니다.



### Cracking the Code of Juxtaposition: Can AI Models Understand the Humorous Contradictions (https://arxiv.org/abs/2405.19088)
- **What's New**: 최신 대형 멀티모달 언어 모델(Multimodal Language Models)은 다양한 작업에서 뛰어난 성능을 보이고 있지만, 인간의 신체를 활용한 농담이나 유머의 미묘한 차이를 이해하는 데 어려움을 겪고 있습니다. 특히 비선형 서사를 포함한 유머 식별은 더욱 도전적인 과제입니다. 본 논문에서는 모순된 서사를 가진 만화(comics)를 중심으로 AI가 이러한 유머를 인식하고 해석하는 능력을 평가할 수 있는 YesBut 벤치마크를 소개합니다.

- **Technical Details**: YesBut 벤치마크는 모델의 만화 이해 능력을 다방면으로 평가하기 위해 네 가지 과제를 제안합니다: 1) 문자 그대로의 설명 작성(literal description writing), 2) 모순 생성(contradiction generation), 3) 철학 선택(underlying philosophy selection), 4) 제목 매칭(title matching). 이러한 과제를 통해 만화의 표면적인 이해에서부터 더 깊은 서사적 추론까지 평가할 수 있습니다. 각각의 YesBut 만화는 두 개의 패널로 구성되어 있으며, 모순되는 서사를 통해 유머를 제공합니다.

- **Performance Highlights**: 상업용 VLM(vision language models)은 공개된 모델들보다 대부분의 과제에서 우수한 성능을 보였으나, 여전히 인간 수준의 성능에는 미치지 못했습니다. 예를 들어, 철학 선택 과제에서 정확도 84.1%, 제목 매칭에서 63.3%의 정확도를 기록했으나, 이는 완벽과는 거리가 있습니다. 추가적으로, 모형에 만화의 설명을 보강했을 때 성능이 크게 향상되었으나, 이는 현재 모델들의 이야기 이해 능력에 여전히 큰 격차가 있음을 시사합니다.



### MEMoE: Enhancing Model Editing with Mixture of Experts Adaptors (https://arxiv.org/abs/2405.19086)
- **What's New**: 최신 arXiv 논문에서는 Large Language Models (LLMs)의 일부 동작을 효율적으로 수정하는 데 중점을 둔 새로운 모델 수정 방법인 MOMoE를 제안했습니다. MOMoE는 Mixture of Experts (MoE) 아키텍처와 지식 앵커 라우팅 전략을 활용하여 기존 파라미터를 변경하지 않고 새로운 지식을 업데이트합니다. 이를 통해 모델의 일반적 능력을 유지하며 특정 입력에 대해 더욱 일반화된 지식을 제공합니다. 또한, 논문은 해당 접근법이 배치 수정 및 순차 배치 수정 작업에서 뛰어난 성과를 보인다고 주장합니다.

- **Technical Details**: MOMoE는 MoE 구조와 지식 앵커 라우팅 전략을 사용하여 모델을 수정합니다. MoE 구조는 병렬 형태로 구현돼 있으며, 원래 모델의 파라미터는 그대로 유지됩니다. 이는 모델이 전반적인 성능을 유지하면서 특정 지식 업데이트에만 집중할 수 있도록 합니다. 지식 앵커 라우팅 전략은 유사한 지식을 필요로 하는 입력을 동일한 전문가에게 라우팅하여, 모델의 일반화 성능을 향상시킵니다. 본 연구는 MOMoE가 편집 신뢰성, 일반성, 지역성 세 가지 특성을 모두 수용할 수 있는 방안을 제시하고 있습니다.

- **Performance Highlights**: 실험 결과에 따르면, MOMoE는 기존의 배치 수정 및 순차 배치 수정 작업에서 우수한 성능을 나타냈습니다. 특히, MOMoE는 일반성과 지역성 사이의 균형을 뛰어나게 유지하며, 전반적인 성능에서도 높은 정확성을 기록했습니다. 또한, 모델의 일반적인 능력에 미치는 영향이 최소화되었음을 확인하였습니다.



### Auxiliary Knowledge-Induced Learning for Automatic Multi-Label Medical Document Classification (https://arxiv.org/abs/2405.19084)
Comments:
          Accepted to LREC-COLING 2024 -- camera-ready version

- **What's New**: 최근 연구에서 제안된 새로운 ICD 인덱싱 접근법은 임상 노트(clinical notes)와 보조 지식을 결합하여 ICD 코드 예측의 정확성을 향상시키는 것을 목표로 하고 있습니다. 주요 혁신점으로는 다중 수준 심층 팽창 잔차 합성곱 인코더(multi-level deep dilated residual convolution encoder)를 사용하여 임상 텍스트의 다양한 길이를 학습하고, 임상 기록에 대한 보조 지식을 활용하여 ICD 분류 작업을 공식화하며, 그래프 합성곱 네트워크(graph convolutional network)를 도입하여 ICD 코드 간의 공발 패턴을 분석하는 점이 포함됩니다.

- **Technical Details**: 제안된 방법은 크게 세 가지 주요 기술적 요소를 포함합니다. 첫째, 임상 노트의 정보를 집약하기 위해 다중 수준 심층 팽창 잔차 합성곱 인코더를 사용하여 긴 텍스트의 문서 표현 학습을 지원합니다. 둘째, 임상 텍스트뿐만 아니라 다양한 임상의 코드 용어 및 처방된 약물 정보를 통합하여 보조 지식을 활용한 ICD 분류 작업을 진행합니다. 셋째, ICD 코드 간의 공발 패턴을 활용하여 라벨 표현의 품질을 향상시키기 위해 그래프 합성곱 네트워크를 도입합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 다양한 측정치에서 최첨단 성능을 달성하는 것으로 나타났습니다. 특히 MIMIC-III 데이터셋을 사용한 자동 ICD 코딩 평가에서 이전 방법들보다 뛰어난 성능을 보였습니다.



### BLSP-KD: Bootstrapping Language-Speech Pre-training via Knowledge Distillation (https://arxiv.org/abs/2405.19041)
- **What's New**: BLSP-KD라는 새로운 접근법이 소개되었습니다. 이 방식은 Knowledge Distillation을 통해 LLM과 스피치 입력 간의 정렬 품질을 최적화하고 Speech-Text 길이 불일치를 해결합니다. 또한, Partial LoRA(PLoRA)라는 새로운 적응 기법을 도입하여 지식 증류(knowledge distillation) 하에서의 LLM 미세 조정을 지원합니다.

- **Technical Details**: BLSP-KD는 다음 두 가지 주요 기술을 사용합니다. 첫째, 스피치-텍스트 정렬을 지식 증류(knowledge distillation) 문제로 간주하여 다음 토큰 예측 분포의 KL-divergence를 줄임으로써 정렬 품질을 직접적으로 측정하고 최적화합니다. 둘째, Continuous-Integrate-and-Fire 전략을 사용하여 변형기 블록과 통합된 모듈을 통해 스피치를 토큰으로 분할하여 텍스트 토큰과 일대일 대응이 가능하게 합니다. 이를 통해 스피치와 텍스트 간의 미세한 정렬이 가능해지며 추가적인 스피치 지시 데이터 없이도 정렬을 최적화할 수 있습니다.

- **Performance Highlights**: BLSP-KD는 양적 평가에서 이전의 end-to-end 접근법과 유사한 규모의 파라미터를 가진 캐스케이드 시스템을 능가하는 성능을 보였습니다. 특히, LLM의 스피치 입력에 대한 지시 따르기(instruction-following) 능력을 크게 향상시켰습니다.



### Evaluating the External and Parametric Knowledge Fusion of Large Language Models (https://arxiv.org/abs/2405.19010)
Comments:
          15 pages, 3 figures, 3 tables

- **What's New**: 본 논문은 LLMs(대형 언어 모델)의 외부 지식(External Knowledge)과 기계 자체의 매개 변수 지식(Parametric Knowledge)을 융합하는 새로운 방법을 제안합니다. 특히, 외부 지식이 불완전할 때 LLMs가 어떻게 내부 지식을 사용하여 보완하는지를 탐구하였습니다.

- **Technical Details**: LLMs의 지식 융합을 4가지 시나리오로 나누어 연구를 진행하였습니다. 시나리오는 다음과 같습니다: S1 - 외부 지식만으로 답변이 충분한 경우, S2 - 외부 지식이 부분적인 정보를 제공하고 내부 지식이 보완해야 하는 경우, S3 - 외부 지식이 유용하지 않고 내부 지식만으로 답변이 가능한 경우, S4 - 어떤 지식으로도 답변이 불가능한 경우. 연구를 위해 최신 전자 제품 도메인 데이터를 수집하고 이를 외부 지식과 매개 변수 지식으로 나누어 실험을 설계하였습니다.

- **Performance Highlights**: 연구 결과, LLMs의 매개 변수 지식을 강화하면 지식 융합 능력이 크게 향상될 수 있음을 발견하였습니다. 그러나 여전히 매개 변수 지식의 기억과 추출, 그리고 지식 경계 설정에서의 도전 과제가 남아 있습니다.



### Encoding Hierarchical Schema via Concept Flow for Multifaceted Ideology Detection (https://arxiv.org/abs/2405.18974)
Comments:
          13pages, 4 figures (Accepted to Findings of ACL 2024)

- **What's New**: 이번 연구에서는 다면적 이념 탐지(Multifaceted Ideology Detection, MID) 작업을 위해 새로운 개념 의미 강화 프레임워크를 개발했습니다. 기존 연구가 일반적인 단일 측면에만 초점을 맞췄던 것과 달리, 이번 연구는 이념의 레이블 의미와 설명을 고려했습니다. 이를 통해 구체적인 이념 개념을 전달하고, 다면적 이념을 효과적으로 탐지할 수 있는 방법을 제공합니다.

- **Technical Details**: 이 연구에서는 이념 탐지를 위한 Bidirectional Iterative Concept Flow (BICo) 방식을 제안했습니다. BICo는 개념들이 스키마 트리의 계층 수준을 가로질러 흐르도록 하여 다중 세밀도의 의미를 풍부하게 합니다. 또한, 개념 주의 맞춤 및 개념 안내 대조 학습(Concept-Guided Contrastive Learning) 전략을 사용해 모델이 학습된 개념 의미를 바탕으로 이념 특징을 포착하도록 합니다.

- **Performance Highlights**: 이 접근법은 벤치마크 데이터셋인 MITweet에서 최첨단 성능을 달성했으며, 특히 교차 주제 시나리오(cross-topic scenario)에서도 훌륭한 성과를 보였습니다. 이는 레이블 의미와 설명을 포함한 최초의 MID 작업이며, 효율적인 다면적 이념 탐지 모델을 설계하는 데 중요한 기여를 했습니다.



### Are You Sure? Rank Them Again: Repeated Ranking For Better Preference Datasets (https://arxiv.org/abs/2405.18952)
- **What's New**: 이번 연구는 다국어로 된 사용자 프롬프트(prompt)에 대한 응답을 GPT-4로 여러 번 평가하는 'Repeat Ranking' 방법을 제안합니다. 이 방법을 통해 일관된 순위를 유지하는 응답만을 학습에 사용하여, 모델 성능을 향상시킵니다. 기존의 모든 프롬프트 데이터를 사용하는 방법보다 뛰어난 성능을 보여줍니다.

- **Technical Details**: 연구 팀은 62개 언어로 된 2,714개의 프롬프트를 선택하여 7개의 최첨단 다국어 LLM (Large Language Model)에서 응답을 생성했습니다. 그런 다음, 각 응답 세트를 GPT-4로 5번씩 평가하여 순위의 일관성을 필터링 도구로 사용했습니다. 일관성이 높은 순위를 갖춘 데이터를 사용하여 모델을 학습했습니다. 실험에서는 모든 순위를 사용한 모델과, 일관성이 가장 높은 75%, 50%, 25%의 순위 데이터로 학습한 모델을 비교했습니다.

- **Performance Highlights**: 제안된 Repeat Ranking 방법은 다국어 MT-Bench 채팅 벤치마크에서 더 높은 성능을 보여주었습니다. 특히, 일관된 순위를 갖춘 데이터를 사용한 모델이 대부분의 테스트된 언어에서 더 나은 다운스트림 평가 성능을 나타냈습니다. 이는 향후 데이터셋 생성 시의 품질과 일관성의 중요성을 강조합니다.



### Understanding and Addressing the Under-Translation Problem from the Perspective of Decoding Objectiv (https://arxiv.org/abs/2405.18922)
Comments:
          ACL 2024 main conference

- **What's New**: 이 연구에서는 NMT(Neural Machine Translation) 시스템에서 발생하는 under-translation 문제의 기저 원인을 분석하고, 이를 해결하기 위한 새로운 접근법을 제안합니다. 주요 아이디어는 End Of Sentence(EOS) 예측 확률을 활용하여 under-translation을 감지하고, 이를 기반으로 벌점을 강화해 under-translation 후보를 페널티를 가중시키는 것입니다.

- **Technical Details**: 연구팀은 문장 레벨 및 문서 레벨의 번역 시나리오를 모방한 합성 데이터를 활용해 실험을 수행했습니다. 특히, EOS 예측 확률이 낮을 때 모델이 번역을 완료하지 않았음을 감지할 수 있음을 밝혔습니다. 이를 바탕으로, EOS 예측 확률을 페널티로 사용하여 under-translation을 감지하고 벌점을 부여하는 방법을 제안했습니다. 이 방법은 Beam Search 디코딩에서 후보 선택 시 EOS 확률을 페널티로 적용하여 번역 길이에 비례하게 페널티를 가중하는 방식입니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 under-translation을 정확하게 감지하고 수정할 수 있었으며, 다른 올바른 번역에는 거의 영향을 미치지 않았습니다. 합성 및 실제 데이터 실험에서 키워드의 번역 확률을 관찰하여 EOS 확률을 페널티로 활용해 under-translation을 줄이는 데 성공했습니다.



### Towards Faithful Chain-of-Thought: Large Language Models are Bridging Reasoners (https://arxiv.org/abs/2405.18915)
Comments:
          25 pages, under review

- **What's New**: 이번 논문에서는 대형 언어 모델(LLM)이 체인오브생각(CoT) 생성 과정에서 신뢰성 문제를 보이는 이유와 이를 해결할 방법을 제시하고 있습니다. CoT의 각 단계를 세밀하게 분석해 두 가지 추론 패러다임(중앙집중형 추론과 분산형 추론)을 식별하였으며, 이와 신뢰성의 관계를 탐구하였습니다. 이 결론에 기반하여 정보누락 문제를 완화하기 위한 추론 브리징 방법을 제안하였습니다.

- **Technical Details**:  이 논문에서는 의존성 이론을 사용하여 CoT의 다양한 단계와 답변 간의 인과 관계를 측정했습니다. 특히, 통합 그래디언트(IG) 기반의 속성 방법을 사용하여 중앙집중형 추론과 분산형 추론이라는 두 가지 CoT 추론 패러다임을 식별하였습니다. 중앙집중형 추론은 CoT의 마지막 단계에 주로 의존하는 반면, 분산형 추론은 여러 단계에서 정보를 사용합니다. CoT의 신뢰성 문제는 주로 분산형 추론에서 발생하는 것으로 나타났습니다.

- **Performance Highlights**: 제안된 추론 브리징 방법은 CoT 생성 시 누락된 정보를 하인트로 제공하고 의미상 일관성이 낮은 CoT를 필터링하여 신뢰성 문제를 완화합니다. 다양한 추론 벤치마크에서 광범위한 실험을 통해 제안된 방법의 효과를 입증하였으며, 최대 8.8%의 성능 향상을 보였습니다.



### Language Generation with Strictly Proper Scoring Rules (https://arxiv.org/abs/2405.18906)
Comments:
          ICML 2024

- **What's New**: 언어 생성 모델의 학습 방법론에 새로운 접근 방식을 제안합니다. 기존의 최대 우도 추정(Maximum Likelihood Estimation, MLE)에 기반한 로그 확률(loss function) 대신, Brier score와 Spherical score 같은 클래식한 엄격한 적절 점수 규칙(strictly proper scoring rules)을 사용하여 모델의 언어 생성 능력을 향상시켰습니다.

- **Technical Details**: 언어 생성은 대개 MLE를 통해 로그 확률을 최소화하는 방식으로 수행됩니다. 이는 모델이 관측된 샘플의 확률만을 다루어 자연어의 큰 샘플 공간을 처리할 수 있기 때문입니다. 하지만, 로그 확률은 무한대로 갈 수 있어 작은 예측 오류에 민감합니다. 이를 극복하기 위해, 이번 연구에서는 토큰 레벨에서 적절 점수를 분배하는 전략을 통해 비지역적 점수(non-local scoring rules)를 언어 생성에 적용할 수 있는 방안을 제안하였습니다. 이를 통해 모델이 바람직한 확률을 생성할 때만 기대 손실이 최소화되도록 합니다.

- **Performance Highlights**: 실험 결과, 하이퍼파라미터를 조정하지 않고 단순히 손실 함수만을 대체한 경우에도 모델의 생성 능력이 상당히 향상됨을 확인했습니다. 이러한 향상은 LLaMA-7B와 LLaMA-13B 같은 대형 언어 모델에서도 확장 가능합니다.



### Simulation, Modelling and Classification of Wiki Contributors: Spotting The Good, The Bad, and The Ugly (https://arxiv.org/abs/2405.18845)
- **What's New**: 이 논문에서는 데이터 크라우드소싱(crowdsourcing) 플랫폼에서 인간과 비인간(봇) 기여자들을 자동으로 식별하고, 선의와 악의를 구분하는 접근 방식을 제안합니다. 이 연구는 WikiVoyage를 테스트베드로 사용해 클래스 균형 데이터 스트림을 통해 분류기의 정확성과 품질을 크게 향상시켰습니다. 실험 결과, 제안된 방법이 92%의 분류 정확도로 인간과 봇, 그리고 악성 및 선의 기여자를 구분하는 데 성공했습니다.

- **Technical Details**: 이 접근 방식은 데이터 시뮬레이션과 데이터 스트림 모델링을 사용하여 기여자 프로필을 실시간으로 구축하고, 자동적으로 분류를 진행합니다. 기여자들을 모델링하기 위해 리뷰 기반 정보(예: 리뷰 수, 되돌리기 빈도, 삽입 및 삭제된 문자 수)와 Wiki Platform의 ORES(Edit Quality API)를 이용합니다. 또한 이 논문은 클래스 균형을 맞추기 위해 합성 데이터를 생성하고, 이를 실제 데이터와 결합하여 성능을 향상시켰습니다.

- **Performance Highlights**: 실험은 실제 WikiVoyage 데이터 셋을 사용하여 진행되었으며, 417,417개의 봇과 46,952명의 인간 기여자를 포함합니다. 클래스 균형이 맞지 않는 상황에서도, 최종 두 단계 스태킹(stacking) 분류기는 80%에서 95% 사이의 정확도와 F-measure를 달성했습니다. 이 접근 방식은 기여자들의 악의를 사전에 식별하고 방지할 수 있는 혁신적인 솔루션을 제공합니다.



### Toxicity Detection for Fr (https://arxiv.org/abs/2405.18822)
- **What's New**: 이 논문은 'Moderation Using LLM Introspection (MULI)'라는 새로운 독성 탐지 방법을 소개합니다. MULI는 LLM(Introspection)을 활용해 독성 프롬프트를 탐지하는 접근법으로, 별도의 모델을 추가로 학습할 필요 없이 뛰어난 성능을 제공합니다.

- **Technical Details**: MULI는 모델 자체의 출력을 검사하여 독성 프롬프트를 탐지합니다. 특히 첫 번째 응답 토큰의 logit 분포를 분석하여 독성을 판별하는데, 'Sorry', 'Cannot'와 같은 특정 토큰이 독성 프롬프트에서 높은 logit을 보인다는 점을 이용합니다. 이에 기반하여 Sparse Logistic Regression (SLR)과 Lasso Regularization을 사용해 더욱 견고한 탐지기를 구축하였습니다.

- **Performance Highlights**: MULI는 최신 독성 탐지기 대비 여러 메트릭에서 훨씬 뛰어난 성능을 보였습니다. 특히, ToxicChat 데이터셋에서 0.1%의 FPR(위양성율)에서 42.54%의 TPR(진양성율)을 기록하여, 동일한 FPR에서 5.25%의 TPR을 보인 LlamaGuard를 크게 앞지릅니다.



### Genshin: General Shield for Natural Language Processing with Large Language Models (https://arxiv.org/abs/2405.18741)
- **What's New**: 대형 언어 모델(LLMs)인 ChatGPT, Gemini, LLaMA 등이 최근 많은 영역에서 큰 발전과 일반화 능력을 보여주고 있습니다. 그러나 이러한 모델들은 여전히 해석 가능성이 제한되어 있으며, 특히 금융 사기나 피싱과 같은 중요한 영역에서는 적용에 제약이 있습니다. 이를 해결하기 위해, LLMs를 방어적 플러그인으로 사용하는 새로운 계층적 프레임워크 Genshin이 제안되었습니다. Genshin은 텍스트를 원래 상태로 회복시키는 데 LLMs를 활용합니다.

- **Technical Details**: Genshin 프레임워크는 세 가지 주요 단계를 포함합니다. 첫 번째 단계는 LLM을 사용하여 텍스트를 복원하는 'denoising stage'입니다. 두 번째 단계는 복원된 텍스트를 분석하는 'analyzing stage'로, 주로 ML 모델을 통해 정보 분석을 수행합니다. 마지막 단계는 LLM의 출력을 정확하게 설명하는 'interpreting stage'입니다. 이 프레임워크는 문자 수준의 혼란(char-level disturbance), 단어 수준의 혼란(word-level disturbance), 유사성 기반 혼란(similarity-based disturbance)과 같은 다양한 공격을 극복할 수 있도록 설계되었습니다.

- **Performance Highlights**: 감정 분석과 스팸 탐지 작업에서 Genshin의 성능은 현존하는 중간 모델들의 치명적인 결함을 극복하고, LLM의 복구 능력에서 탁월한 결과를 보여주었습니다. 추가적으로, 4세대 패러다임에서 파생된 'LLM defender'를 이용하여, NLP의 3세대 패러다임에서 BERT의 15% 최적 마스크 비율 결과를 재현할 수 있었습니다. LLM을 잠재적 공격 도구로 사용하는 경우, 공격자는 거의 의미 손실 없이 효과적인 공격을 수행할 수 있는 능력을 가지고 있습니다.



### Reverse Image Retrieval Cues Parametric Memory in Multimodal LLMs (https://arxiv.org/abs/2405.18740)
- **What’s New**: 새로운 연구는 멀티모달 대형 언어 모델(MultiModal Large Language Models, MLLMs)에 대한 리버스 이미지 검색(Reverse Image Retrieval, RIR) 알고리즘의 효과를 탐구합니다. 이 새로운 접근법은 GPT-4 시리즈 모델들의 지식 집약적 시각 질문 응답(Visual Question Answering, VQA) 성능을 향상시킵니다.

- **Technical Details**: 연구는 브라우저 기반 API를 구축하여 웹 스케일의 리버스 이미지 검색을 수행합니다. 검색 결과로 얻어진 여러 이미지와 캡션을 스크린샷으로 캡처해 MLLM에게 제공하여 문맥을 보강합니다. 결과 이미지 요약을 컨텍스트로 제공하여 모델의 성능을 향상시킵니다.

- **Performance Highlights**: RIR는 GPT-4V 모델에서 37-43%, GPT-4 Turbo 모델에서 25-27%, GPT-4o 모델에서 18-20%의 성능 향상을 보였습니다. 실험 결과, RIR는 모델이 자신의 세계 지식을 더 잘 접근하도록 돕는 것으로 나타났습니다. 인류평가를 통해 사례별로 RIR이 성능에 어떤 영향을 미치는지도 분석했습니다.

- **Additional Insights**: RIR의 가장 큰 장점은 지식 집약적 시각 질문을 모델의 세계 지식과 더 잘 정렬할 수 있도록 돕는 것입니다. 또한, RIR은 웹에서 존재감이 적은 객체나 개념에 대해 더 많은 도움을 줍니다. 모든 코드는 GitHub에서 확인할 수 있습니다: https://github.com/mi92/reverse-image-rag.



### CtrlA: Adaptive Retrieval-Augmented Generation via Probe-Guided Contro (https://arxiv.org/abs/2405.18727)
Comments:
          28 pages, 7 figures, 9 tables

- **What's New**: 이번에 소개하는 연구는 'CtrlA'라는 새로운 probe-guided adaptive RAG (Retrieval-augmented generation) 프레임워크입니다. 이 프레임워크는 LLMs (Large Language Models)의 내부 상태를 탐구하여, 기존의 RAG 방식에서 발생하던 문제들을 극복하려는 첫 시도라고 할 수 있습니다. CtrlA는 내부 상태를 모니터링하는 honesty probe와 confidence probe를 사용하여 모델의 정직성을 높이고 필요한 경우에만 외부 지식을 가져오도록 합니다.

- **Technical Details**: CtrlA 프레임워크는 두 가지 주요 요소인 honesty probe와 confidence probe를 활용합니다. Honesty probe는 LLM의 행동을 조절하여 정직성을 증가시키고, confidence probe는 LLM의 내부 상태를 모니터링하여 신뢰도를 평가합니다. 이를 통해, LLM이 언제 외부 지식(retrieved external knowledge)을 필요로 하는지 동적으로 판단합니다. 기존의 방법들이 언어적인 피드백이나 확률 기반의 피드백에 의존하거나, 특별히 제작된 데이터셋으로 LLM을 미세 조정하는 방식과는 다르게, CtrlA는 LLM의 내부 표현을 이용하는 접근 방식을 채택합니다.

- **Performance Highlights**: 실험 결과, CtrlA는 다양한 작업에서 기존의 adaptive RAG 방법들보다 탁월한 성능을 입증했습니다. Honesty control을 통해 LLM을 더 정직하게 만들 수 있었고, confidence monitoring이 retrieval의 필요성을 판단하는 데 유망한 지표임이 드러났습니다.



### Contextual Position Encoding: Learning to Count What's Importan (https://arxiv.org/abs/2405.18719)
- **What's New**: 새로운 위치 인코딩 방식인 Contextual Position Encoding(CoPE)을 제안하였습니다. 이 방식은 토큰의 위치를 문맥(context)에 따라 조정함으로써 특정 단어, 명사 또는 문장에 대해 위치를 지정할 수 있습니다.

- **Technical Details**: CoPE는 현재 토큰을 쿼리(query)로 사용하여 이전의 각 토큰에 대해 게이트 값을 계산하고, 이를 통해 각 토큰의 상대 위치를 결정합니다. 이 상대적 위치는 분수 값을 가질 수 있어, 정수 값에 할당된 임베딩을 보간하여 위치 임베딩을 계산합니다. 이를 통해 여러 레이어(layer)와 쿼리(query)에 대해 동시다발적으로 위치를 측정할 수 있게 됩니다.

- **Performance Highlights**: CoPE는 selective copy, counting 및 Flip-Flop 작업에서 기존의 토큰 기반 위치 인코딩 방법보다 뛰어난 성능을 보였습니다. 또한 위키피디아 텍스트에 대한 언어 모델링 작업과 코드 작업에서도 더 나은 perplexity 성능을 보였습니다.



### Efficient Model-agnostic Alignment via Bayesian Persuasion (https://arxiv.org/abs/2405.18718)
- **What's New**: 이 논문은 Bayesian Persuasion Alignment 프레임워크를 소개하며, 작은 모델을 사용해 블랙박스 대형 모델의 정렬을 효율적으로 할 수 있는 방법을 탐구합니다. Supervised Fine-tuning(SFT)이나 Reinforcement Learning from Human Feedback(RLHF) 같은 기존 방법들은 많은 계산 자원과 방대한 데이터가 필요하지만, 새로운 프레임워크는 이 문제를 해결하려고 합니다.

- **Technical Details**: 제안된 프레임워크는 정보를 전달하는 작은 모델(Advisor)이 큰 모델(Receiver)을 설득하여 더 나은 응답을 유도하는 최적의 신호 전략을 학습하는 프로토콜을 수립합니다. 이 프레임워크는 Bayesian Persuasion 이론에 기반하여, 작은 모델이 큰 모델의 행위를 조정하며 정보 디자인 관점에서 작동합니다.

- **Performance Highlights**: Empirical 결과로는, GPT-2가 다양한 모델의 성능을 수학적 문제 해결 능력에서 평균 16.1% 향상, 코드 생성 작업에서 평균 13.7% 향상시키는 것을 보여주었습니다. 또한 Advisor로 사용된 모델 Phi-2는 GSM8K 데이터셋에서 22.5%, MATH 데이터셋에서 39.0%, HumanEval 데이터셋에서 24.7% 향상을 기록했습니다.



### Can GPT Redefine Medical Understanding? Evaluating GPT on Biomedical Machine Reading Comprehension (https://arxiv.org/abs/2405.18682)
- **What's New**: 본 논문에서는 GPT를 사용하여 네 가지 폐쇄형 바이오메디컬 MRC(바이오메디컬 머신 리딩 컴프리헨션) 벤치마크에 대해 평가하였습니다. 다양한 기존의 프롬프팅(프로빙) 기법을 실험할 뿐만 아니라 새로운 프롬프팅 방법인 Implicit Retrieval Augmented Generation (RAG)을 제안했습니다. 이는 전통적인 RAG 설정에서 중요한 텍스트 추출을 위해 벡터 데이터베이스를 사용해야 하는 문제를 해결합니다. 결과적으로, 제안된 새로운 프롬프팅 기법은 네 가지 데이터셋 중 두 개에서 최고 성능을 달성하고, 나머지 두 개에서는 두 번째로 높은 성능을 보였습니다.

- **Technical Details**: 폐쇄형 설정에서는 LLM(대형 언어 모델)이 주어진 컨텍스트만을 사용해 질문에 답해야 합니다. Implicit RAG 방법은 LLM이 쿼리에 관련된 텍스트 추출 섹션을 먼저 검색한 후 질문에 답하도록 합니다. 이를 통해 벡터 데이터베이스의 사용 없이 한 번의 시도로 검색을 수행할 수 있습니다. 실험 결과, 이 기법은 네 가지 데이터셋 중 두 개에서 최고 성능을 기록했으며 다른 두 개에서는 두 번째로 높은 성능을 보였습니다. 또한, 인간 전문가의 질적 평가에서도 제안된 접근 방식의 출력이 대부분 경우 수긍되었음을 확인했습니다.

- **Performance Highlights**: 폐쇄형 바이오메디컬 MRC 벤치마크에서 GPT가 두 가지 벤치마크에서 새로운 최첨단(SoTA) 결과를 달성했습니다. Implicit RAG 프롬프팅 기법을 사용한 결과, 네 가지 데이터셋 중 두 개에서 최고 성능을 기록하고 나머지 두 개에서는 두 번째로 높은 성능을 기록했음을 보여주었습니다. 이는 현대 LLM이 제로샷 설정에서도 감독 학습 모델을 능가할 수 있음을 시사합니다.



### Understanding Intrinsic Socioeconomic Biases in Large Language Models (https://arxiv.org/abs/2405.18662)
- **What's New**: 이 연구는 현재까지 잘 다뤄지지 않은 대규모 언어 모델(Large Language Models, LLMs)에서의 성향과 경제적 편향의 관계를 탐구합니다. 이를 위해 백만 개의 영어 문장으로 구성된 새로운 데이터를 소개하고, 설립된 모델(GPT-2)과 최첨단 모델(Llama 2, Falcon)에서 포괄적인 경제적 편향을 체계적으로 정량화합니다. 이 연구는 특히 교차성(intersectionality)이 편향을 어떻게 증폭시키는지를 강조하며, LLM이 이름에서 다수의 인구통계적 속성을 추출하고 특정 경제적 편향과 연관짓는 능력을 보여줍니다.

- **Technical Details**: 연구는 이름, 출생 시 할당된 성별, 혼인 상태, 인종, 종교 등의 인구통계적 속성에 기초하여 LLM의 내재적 경제적 편향을 평가합니다. 이를 위해 100만 개의 영어 문장으로 구성된 새로운 평가 데이터셋을 생성하고, LLM이 이름에서 인종 및 성별 정보를 추출할 수 있는지, 그리고 이러한 이름과 연관된 경제적 편향을 평가합니다. 또한, 성별, 인종, 혼인 상태 등의 교차성에 따른 편향 동태도 평가합니다.

- **Performance Highlights**: 연구 결과, Llama 2와 Falcon과 같은 최첨단 모델뿐만 아니라 GPT-2와 같은 기존 모델에서도 광범위한 경제적 편향이 발견되었습니다. 특히, 교차성을 고려할 때 이러한 편향이 크게 증폭되는 것으로 나타났습니다. 예를 들어, 특정 이름이 주어졌을 때 모델은 그 이름과 관련된 여러 인구통계적 속성을 추출하고, 이를 특정 경제적 편향과 연관짓는 능력을 보여주었습니다.



### Recent Advances of Foundation Language Models-based Continual Learning: A Survey (https://arxiv.org/abs/2405.18653)
- **What's New**: 최신 연구에서는 기반 언어 모델 (Foundation Language Models, LMs)의 지속 학습 (Continual Learning, CL) 메커니즘을 조사하고, 기존 문헌을 체계적으로 분류 및 비교하고자 하였습니다. 기반 LMs는 NLP와 CV 분야에서 중요한 성과를 이뤘지만, 여전히 인간과 같은 지속적인 학습에는 한계가 있으며, 이를 해결하고자 CL 기반 방법론들이 개발되었습니다.

- **Technical Details**: 기반 LMs는 크게 사전 학습 언어 모델 (Pre-trained Language Models, PLMs), 대형 언어 모델 (Large Language Models, LLMs), 그리고 시각-언어 모델 (Vision-Language Models, VLMs)로 나눌 수 있습니다. 연구는 오프라인 지속 학습 (Offline CL)과 온라인 지속 학습 (Online CL)으로 구분되며, 각 방법론들을 전통적 방법, 파라미터 효율적 방법, 지침 조정 기반 방법, 지속 사전 학습 방법 등으로 세분화했습니다. 오프라인 CL은 도메인-증가 학습 (Domain-Incremental Learning), 작업-증가 학습 (Task-Incremental Learning), 클래스-증가 학습 (Class-Incremental Learning)으로 분류되며, 온라인 CL은 하드 작업 경계 (Hard Task Boundary)와 흐릿한 작업 경계 (Blurry Task Boundary)로 구분됩니다.

- **Performance Highlights**: 연구에서는 CL 기반 연구에 사용되는 대표적인 데이터셋과 지표들을 소개하고, 기반 LMs의 지속 학습에서 발생하는 주요 도전 과제와 향후 연구 방향에 대한 상세 분석을 제공했습니다. 각 방법론의 구체적인 성과는 체계적으로 평가되어, 새로운 작업을 학습하면서도 이전의 지식을 잃지 않도록 유지하는 능력에서의 성과를 강조했습니다.



### Training LLMs to Better Self-Debug and Explain Cod (https://arxiv.org/abs/2405.18649)
- **What's New**: 이번 연구에서는 코드 생성 분야에서 LLMs의 자체 디버깅 능력을 크게 향상시키기 위한 새로운 훈련 프레임워크를 제안합니다. LLMs가 잘못된 코드를 설명하고 수정하는 자동화된 파이프라인을 통해 고품질의 데이터셋을 수집하고, 이를 통해 감독 강화 학습 및 보상 설계를 적용함으로써 기존 방법 대비 높은 성능 향상을 달성하였습니다.

- **Technical Details**: 본 연구에서는 잘못된 코드에 대한 체계적인 설명과 수정 과정을 통해 LLMs의 자체 디버깅 능력을 향상시키는 자동화된 파이프라인을 제안합니다. 이 파이프라인은 설명 및 수정 경로를 생성하고 실행 검증을 통해 높은 품질의 데이터셋을 수집합니다. 감시된 미세 조정(Supervised fine-tuning, SFT)과 강화 학습(Reinforcement Learning, RL)을 통해 코드 설명 및 수정 능력을 방대하게 향상시킵니다.

- **Performance Highlights**: SFT는 네 가지 벤치마크에서 최대 15.92%의 pass@1과 9.30%의 pass@10 성능 향상을 보여줍니다. RL 훈련은 추가적으로 pass@1에서 최대 3.54%, pass@10에서 최대 2.55%의 성능 향상을 달성합니다. 또한, 인간 평가에서도 이 프레임워크로 훈련된 LLMs가 더 유용한 코드 설명을 생성하고, 개발자가 소스 코드 내 버그를 더 잘 이해할 수 있도록 도와주는 것으로 나타났습니다.



### ConSiDERS-The-Human Evaluation Framework: Rethinking Human Evaluation for Generative Large Language Models (https://arxiv.org/abs/2405.18638)
Comments:
          Accepted in ACL 2024

- **What's New**: 본 논문에서는 생성형 대형 언어 모델(LLMs)의 인간 평가(human evaluation)를 다학문적으로 접근해야 한다고 주장합니다. 사용자 경험 연구(user experience research)와 인간 행동 심리학(human behavioral psychology) 등의 분야에서 얻은 인사이트를 바탕으로, 실험 설계와 결과가 신뢰할 만한 설명력을 가질 수 있도록 해야 한다고 강조합니다. 또한 평가에서 유용성, 미학적 측면, 인지 편향(cognitive biases) 등을 고려해야 한다고 지적합니다.

- **Technical Details**: 논문에서는 사용자 경험 및 인지 편향의 중요성을 강조하며, ConSiDERS-The-Human 평가 프레임워크를 제안합니다. 이 프레임워크는 6개의 주요 요소로 구성되어 있습니다: 일관성(Consistency), 채점 기준(Scoring Criteria), 구별(Differentiating), 사용자 경험(User Experience), 책임성(Responsible), 확장성(Scalability)입니다. 또한, 테스트 세트의 효과성이 모델의 역량을 측정하는 데 중요하다고 설명하며, 비효과적인 테스트 세트는 모델을 충분히 평가하지 못한다고 주장합니다.

- **Performance Highlights**: 지금까지 20년간 'human'과 'eval'이 포함된 논문 중 7% 미만만이 사용자 경험 관련 키워드를 언급했다고 지적하며, UX와 인지 편향이 인간 평가의 최전선이 되어야 할 필요성을 강조합니다. 인지 편향을 무시하면 잘못된 결론에 이를 수 있으며, 공통적 인지 편향의 효과를 줄이기 위한 구체적인 권장 사항을 제공합니다.



### GLOCON Database: Design Decisions and User Manual (v1.0) (https://arxiv.org/abs/2405.18613)
- **What's New**: GLOCON은 여러 나라의 다양한 언어로 작성된 국내 뉴스 소스로부터 자동으로 추출된 논쟁적 사건들의 데이터베이스입니다. 국내 뉴스 소스를 활용하며, 각 소스에 대한 사건 목록을 작성하기 위해 전체 뉴스 아카이브를 처리합니다.

- **Technical Details**: 자동화는 무작위로 샘플링된 표준 코퍼스(gold standard corpus)와 두 명 이상의 도메인 전문가의 주석을 기반으로 달성됩니다. 이 절차는 Yörük et al. (2022)와 Duruşan et al. (2022)의 사건 정의를 따릅니다.

- **Performance Highlights**: GLOCON 데이터베이스는 다양한 국가와 언어의 뉴스 소스를 통해 발생하는 논쟁적 사건들을 포착하여 풍부한 데이터 자산을 제공합니다.



### BioBERT-based Deep Learning and Merged ChemProt-DrugProt for Enhanced Biomedical Relation Extraction (https://arxiv.org/abs/2405.18605)
- **What's New**: 이번 논문에서는 생의학 텍스트에서 화합물-유전자 상호작용을 추출하는 방법론을 제시합니다. BioBERT 모델과 다층 완전 연결 네트워크 아키텍처를 활용하여 ChemProt와 DrugProt 데이터셋을 새로운 합병 전략으로 통합했습니다. 광범위한 실험을 통해 데이터셋 간 공유된 CPR 그룹에서 성능이 크게 향상됨을 입증했습니다. 데이터셋 병합의 중요성을 강조하며, 자동화된 정보 추출의 가능성을 생의학 연구 및 임상 실무에서 확인했습니다.

- **Technical Details**: 이 연구에서는 BioBERT 모델과 완전 연결 네트워크(fully connected network) 아키텍처를 활용하여 화합물-유전자 상호작용을 추출했습니다. ChemProt와 DrugProt 데이터셋을 통합하기 위해 새로운 병합 전략을 도입했으며, 두 데이터셋의 PubMed 초록에서 수집된 화합물 및 유전자 엔티티와 해당 관계를 수동으로 주석 처리한 데이터셋을 사용했습니다.

- **Performance Highlights**: 실험 결과, 데이터셋 간 공유된 CPR 그룹에서 상당한 성능 향상을 확인했습니다. 이는 샘플 수 증가와 모델 정확도 개선에 크게 기여했습니다. 이러한 접근 방법은 생의학 문헌 분석에서 자동 정보 추출의 가능성을 보여주며, 향후 자연어 처리(NLP) 연구의 잠재력을 강조했습니다.



### Learning diverse attacks on large language models for robust red-teaming and safety tuning (https://arxiv.org/abs/2405.18540)
- **What's New**: 이번 연구에서는 GFlowNet 기반의 미세조정을 통해 다양한 공격 유도 프롬프트를 생성하는 새로운 방법을 제안합니다. 제안된 방법은 기존 강화 학습 기반 접근법의 단점인 모드 붕괴(mode collapse)와 균형 잡히지 않은 공격 유도 프롬프트 생성 문제를 해결합니다.

- **Technical Details**: 기존 자동화된 레드팀 방식을 대체하기 위해, 우리는 GFlowNet 미세조정을 사용합니다. 이 방법은 초기 탐색 오프-폴리시(off-policy) 훈련을 통해 다양하고 효과적인 프롬프트를 수집한 후, 수집된 데이터를 최대 우도 추정(Maximum Likelihood Estimation, MLE)을 통해 부드러운 분포로 변환합니다. 이 프로세스를 통해 수집된 공격 유도 프롬프트의 전체 사후 분포를 샘플링할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 GPT-2, dolly-v2-7b, Gemma-2b-it, Llama-2-7b-chat 등 다양한 타겟 LLM에 대해 보다 다양한 공격 유도 프롬프트를 효과적으로 생성합니다. 더욱이, 다른 RL 기반 레드팀 기법으로 학습된 모델들보다 우리 방식으로 안전 튜닝된 모델이 더 견고한 것으로 나타났습니다. 제안된 방법은 추가적인 타겟 LLM으로도 적응이 가능하며 높은 전이성(transferability)을 보입니다.



### LLMs and Memorization: On Quality and Specificity of Copyright Complianc (https://arxiv.org/abs/2405.18492)
Comments:
          10 pages, 3 figures

- **What's New**: 대형 언어 모델(LLMs)의 특정 데이터를 복제하는 문제와 저작권 침해 가능성을 분석합니다. 이 연구에서는 유럽 법률을 예로 들어 저작권 침해 정도를 체계적으로 분석하며, 실제 사용 시나리오에서 instruction-finetuned 모델을 평가합니다.

- **Technical Details**: 분석은 독일 저작권 서비스 제공자법의 기준으로 160자를 임계값으로 설정하고 퍼지 텍스트 매칭 알고리즘(fuzzy text matching algorithm)을 사용해 저작권이 침해될 가능성이 있는 텍스트 복제를 식별합니다. 또한, 저작권이 보호된 텍스트 대신 모델이 취할 행동(거부나 허구 등)을 조사하고 그 행동에 대한 법적 평가를 제공합니다.

- **Performance Highlights**: 각 LLM들 간 저작권 준수와 관련한 큰 차이가 발견되었습니다. Alpaca, GPT 4, GPT 3.5, Luminous 모델이 비교에서 가장 잘 수행되었으며, OpenGPT-X, Alpaca, Luminous 모델은 특히 저작권 침해 가능성이 낮은 것으로 나타났습니다.



### Multi-objective Representation for Numbers in Clinical Narratives Using CamemBERT-bio (https://arxiv.org/abs/2405.18448)
Comments:
          Under the revision. arXiv admin note: substantial text overlap with arXiv:2404.10171

- **What's New**: 이번 연구에서는 의료 문서에서 추출한 수치를 7개의 독립적인 생리적 범주로 분류하는 방법을 탐구했습니다. 이를 위해 CamemBERT-bio 모델을 사용했으며, 두 가지 주요 혁신을 도입했습니다: 키워드 임베딩(keyword embeddings) 통합과 숫자 비중립 전략(number-agnostic strategy)을 채택한 것입니다. 특히, 텍스트에서 모든 숫자 데이터를 제외하는 방식을 사용하여 문맥 중심 학습을 강화했습니다.

- **Technical Details**: 모델 성능 향상을 위해 레이블 임베딩(label embeddings) 기법을 적용하여 주의 메커니즘(attention mechanism)을 개선했으며, '숫자 블라인드(numerical-blind)' 데이터셋을 사용하여 문맥 중심의 학습을 강화했습니다. 추출된 수치 데이터의 중요성을 판단하기 위해 표준 범위 내에 값이 포함되는지를 확인하는 단순한 접근 방식을 사용했습니다. 이 과정에서 CamemBERT-bio를 소규모 및 불균형한 훈련 데이터셋으로 학습했습니다.

- **Performance Highlights**: 실험 결과, CamemBERT-bio는 F1 스코어 0.89를 기록하며 기존의 전통적인 방법(0.73)과 최신 방법(0.82)을 크게 상회했습니다. 이 성과는 소규모 및 불균형 훈련 데이터셋을 사용했음에도 불구하고 달성된 것입니다.



### X-VILA: Cross-Modality Alignment for Large Language Mod (https://arxiv.org/abs/2405.19335)
Comments:
          Technical Report

- **What's New**: X-VILA는 이미지, 비디오, 오디오와 같은 다양한 모달리티를 통합하여 대형 언어 모델(LLMs)의 기능을 확장하는 새로운 모델입니다. 이를 통해 모달리티 간 이해, 추론, 생성을 가능하게 합니다. 특히, 모달리티 맞춤형 인코더와 확산 디코더를 사용하여 각 모달리티와 LLM 간의 정렬을 달성하며, 시각 정보 손실 문제를 해결하기 위해 시각 임베딩 하이웨이 모듈을 제안합니다. 이 프로젝트는 오픈 소스로 공개될 예정입니다.

- **Technical Details**: X-VILA는 텍스트, 이미지, 비디오, 오디오 모달리티를 통합하여 입력과 출력을 처리합니다. 먼저 LLM 입력을 위해 여러 모달리티에서 추출된 특징을 공유할 수 있는 통합 임베딩 공간을 사용하고, LLM 출력을 위해 모달리티별 확산 모델을 사용하여 생성된 출력을 해당 모달리티에 맞게 변환합니다. '텍스트 정렬'과 '시각 정렬'이라는 두 가지 주요 정렬 메커니즘을 도입합니다. 특히 시각 임베딩 하이웨이 모듈을 통해 LLM을 우회하여 시각 특징을 직접 시각 디코더로 전달함으로써 시각 정보의 손실을 방지합니다.

- **Performance Highlights**: X-VILA는 전통적인 방법보다 더욱 뛰어난 성능을 발휘하며, 학습 데이터가 부족한 상황에서도 다양한 모달리티 간의 이해와 생성을 가능하게 합니다. 특히, 1.5백만 개 이상의 멀티 모달리티 대화를 포함하는 새로운 X-to-X 데이터셋을 통해 모달리티 간의 상호작용을 효과적으로 훈련합니다. 이를 통해 텍스트, 이미지, 비디오, 오디오 모달리티 간의 심화된 대응 처리가 가능합니다.



### LLMs Meet Multimodal Generation and Editing: A Survey (https://arxiv.org/abs/2405.19334)
Comments:
          51 Pages with 16 Figures, 12 Tables, and 534 References. GitHub Repository at: this https URL

- **What's New**: 최신 대형 언어 모델(LLM)의 발전에 힘입어 여러 가지 모달리티(multimodality)를 결합한 학습이 주목받고 있습니다. 본 연구는 LLM과 멀티모달 생성(multimodal generation)의 융합을 탐구하며, 이미지, 비디오, 3D, 오디오 등 다양한 도메인에서의 발전을 다룹니다. 특히, 방법론 및 멀티모달 데이터셋에 대한 기술적 구성 요소를 자세히 조사하고, 인간-컴퓨터 상호 작용을 위한 도구로 증강된 멀티모달 에이전트에 대해 논의합니다. 또한, AI 안전성의 진전과 새로운 응용분야 및 향후 전망을 종합적으로 검토합니다.

- **Technical Details**: 멀티모달 생성의 기술적 탁월성에 대해 다각도로 분석합니다. 텍스트 생성 분야에서는 BERT, GPT-1/2/3/4 및 ChatGPT와 같은 모델들이 급격한 성장을 이뤘으며, 이미지 생성에서는 확산 모델(diffusion models)과 대규모 이미지-텍스트 데이터셋이 큰 성과를 거두었습니다. 비디오 생성에서는 비디오 확산 모델을 활용한 혁신적인 연구가 다수 발표되었으며, 3D 생성에서는 CLIP 모델을 이용한 텍스트-3D 변환이 주목받고 있습니다. 오디오 생성 분야에서도 텍스트-오디오, 텍스트-음악, 텍스트-음성 변환에서 우수한 결과를 보이고 있습니다. LLM을 활용한 비주얼, 비디오, 3D, 오디오 생성에 대한 세부 내용을 다룹니다.

- **Performance Highlights**: LLM 기반 멀티모달 생성 기술은 대체로 향상된 생성 품질, 사용자 프롬프트(prompt) 추종 능력 향상, 대화형 기능 및 인체 공학적 인터페이스를 특징으로 합니다. 이미지 생성에서는 시각적 정보가 토큰 ID로 인코딩되어 LLM이 이를 이해하고 생성할 수 있으며, 비디오 생성에서는 LLM이 백본(backbone) 역할을 수행하며 비디오 레이아웃 계획 및 시간적 프롬프트 생성에서도 중요한 역할을 합니다. 3D와 오디오 생성에서도 사용자의 이해를 돕고 상호작용 효율성을 높이는 다양한 방식으로 적용되고 있습니다.



### Robust Preference Optimization through Reward Model Distillation (https://arxiv.org/abs/2405.19316)
- **What's New**: 최근 연구에서는 Direct Preference Optimization (DPO) 방법의 한계를 분석하고, 이를 개선하기 위한 보정 방안을 제안했습니다. DPO는 보상 모델이나 강화 학습(리인포스먼트 러닝; RL)을 따로 훈련하지 않고, 선호 데이터만으로 정책을 훈련하는 오프라인 정렬 방법입니다. 그러나, 기존의 선호 데이터셋에는 각각의 선호 쌍에 대해 하나 또는 두 개의 주석만 들어 있어, DPO가 지나치게 자신감있는 보상을 부여하며, 결과적으로 퇴행적인 정책을 생성할 수 있는 문제가 있습니다.

- **Technical Details**: 연구진은 이러한 문제를 해결하기 위해 보상 모델을 사용해 선호 분포를 더 정확하게 예측하는 '증류' 방법을 제안하였습니다. 이 방법은 선호 데이터를 이용해 훈련된 보상 모델로부터 분포를 배울 수 있도록 언어 모델(LM)을 훈련시킵니다. 또한, 보상 모델의 불확실성을 고려해, 분석된 보상 모델을 사용하여 정책을 훈련합니다. 이렇게 함으로써, 선호 주석의 분포 이동에 대한 강인성을 향상시켜 줍니다.

- **Performance Highlights**: 실험 결과, 증류 방법을 사용한 경우 DPO에 비해 선호 데이터셋이 편향된 상황에서 더욱 향상된 성능을 보여주었습니다. 특히, 원래 DPO 방식은 편향된 데이터셋에서 문제를 겪지만, 보정된 방식은 보다 일관된 정책을 유지했습니다. 이론적 분석을 통해 DPO 방식의 퇴행 성향과 이에 내재된 문제점들에 대한 설명이 제공되며, 이에 대비해 명시적으로 정규화된 접근의 상대적 장점이 강조되었습니다.



### Matryoshka Query Transformer for Large Vision-Language Models (https://arxiv.org/abs/2405.19315)
Comments:
          Preprint. Our code and model are publicly available at this https URL

- **What's New**: 새로운 연구에서 대형 비전-언어 모델(LVLMs)이 다양한 계산 제약을 극복할 수 있는 유연성을 제공하는 Matryoshka Query Transformer(MQT)를 소개했습니다. 이 모델은 최대 m개의 시각적 토큰으로 이미지를 인코딩할 수 있으며, 이를 통해 다양한 작업과 계산 자원에 맞게 토큰 수를 조절할 수 있습니다.

- **Technical Details**: MQT는 M개의 잠재 쿼리 토큰을 사용하여 그리드 특징을 시각적 토큰으로 변환하는 쿼리 변환기를 사용합니다. 훈련 시각별로 무작위로 선택된 m개의 잠재 쿼리 토큰만 사용하고 나머지는 버리는 '테일-토큰 드로핑 전략'을 사용하여 유연성을 제공합니다.

- **Performance Highlights**: MQT-LLaVA 모델은 LLaVA-1.5의 성능을 256개의 시각적 토큰만 사용하여 일치시키며, 16개의 토큰으로 줄일 경우 계산 비용이 8배 감소하고 성능은 2.4포인트만 떨어집니다. ScienceQA와 MMMU 작업에서는 각각 성능이 3%와 6%만 감소하면서도 2개의 토큰으로도 좋은 성능을 유지합니다.



### Language Models Trained to do Arithmetic Predict Human Risky and Intertemporal Choic (https://arxiv.org/abs/2405.19313)
- **What's New**: 대규모 언어 모델(LLMs)과 인간의 행동에서 유사점을 활용하여 LLMs를 인간 인지 모델로 사용하는 방법이 제안되었습니다. 이 연구는 특히 위험 선택(risky choice)과 시간 간 선택(intertemporal choice)에서 기대값(arithmetic of expected value) 계산을 중점적으로 다룹니다. 연구에서는 생태학적으로 타당한(arithmetic dataset) 데이터셋을 이용해 사전학습된 Arithmetic-GPT가 기존의 전통적인 인지 모델보다 인간 행동을 더 잘 예측함을 보여줍니다.

- **Technical Details**: 이 접근법은 (i) LLM과 합리적 에이전트가 인지 문제를 해결하기 위해 마스터해야 할 계산적으로 동등한 작업을 활용하고, (ii) LLM이 인간과 유사한 행동을 보이기 위한 특정 작업 분포를 조사합니다. 실험에서 적절하게 생성된 예상값 계산 데이터셋을 통해 소형 언어 모델(약 10M 파라미터)을 훈련하여, 이 모델이 인간의 선택 패턴을 얼마나 잘 예측하는지 분석하였습니다.

- **Performance Highlights**: 연구 결과, 실험적으로 생성된 데이터셋이 실제 세계의 확률과 가치를 반영할 때, 훈련된 임베딩이 인간의 선택을 가장 잘 예측했습니다. 이러한 사전학습을 통해 도출된 모델은 많은 기존의 행동 모델들보다 위험 및 시간 간 선택에서 더 우수한 성능을 보였습니다.



### VideoTree: Adaptive Tree-based Video Representation for LLM Reasoning on Long Videos (https://arxiv.org/abs/2405.19209)
Comments:
          20 pages, first three authors contributed equally; Project page: this https URL

- **What's New**: VideoTree는 롱폼(long-form) 비디오 이해를 위한 새로운 프레임워크입니다. 이는 질의(query)에 적응적으로 반응하며, 계층적(hierarchical)으로 비디오를 해석할 수 있는 방법을 제공해줍니다. 이는 특히 긴 비디오의 QA(Question Answering) 작업에서 성능과 효율성을 크게 향상시킵니다.

- **Technical Details**: VideoTree는 처음에 비디오의 시각적 특징을 기반으로 프레임을 클러스터링하고, 질의의 관련성을 평가하여 캡션할 프레임을 선택합니다. 그런 다음 비디오의 시각적 클러스터를 질의에 적응적인 트리 구조로 조직합니다. 이 트리는 중요한 세그먼트에서는 더 상세한 정보를 제공하며, 덜 중요한 세그먼트에서는 간략한 요약을 제공합니다. 마지막으로, 트리의 키프레임(keyframes)을 순회하며 이들의 캡션을 LLM에게 전달하여 질문에 대한 답을 생성합니다.

- **Performance Highlights**: VideoTree는 EgoSchema, NExT-QA, IntentQA 벤치마크에서 각각 7.0%, 2.2%, 2.7%의 정확도 향상을 보여주며, 추론 시간을 40% 감축합니다. 이는 기존의 균일하게 샘플링된 프레임 기반 접근법들에 비해 성능과 효율성을 모두 향상시킵니다.



### MetaToken: Detecting Hallucination in Image Descriptions by Meta Classification (https://arxiv.org/abs/2405.19186)
Comments:
          18 pages, 8 figures

- **What's New**: 최근의 엔드투엔드 비전 언어 모델(LVLM)들이 뛰어난 성능을 보이지만, 여전히 시각 정보와 생성된 텍스트 간의 불일치(일종의 환각) 문제가 있습니다. 이를 해결하기 위해 'MetaToken'이라는 경량화된 바이너리 분류기를 제안했습니다. 이 모델은 토큰 수준에서 환각을 감지하며, 초경량 구조로 어떠한 오픈소스 LVLM에도 적용할 수 있습니다.

- **Technical Details**: MetaToken은 통계 분석을 통해 LVLM에서의 환각의 주요 요인을 밝혀냅니다. 이 분류기는 메타 분류(meta classification) 개념을 토대로 하여, 기존 대규모 언어 모델을 재학습하거나 데이터셋을 추가로 필요로 하지 않고도 높은 신뢰도로 환각 감지가 가능합니다. 통계 분석을 통해 밝혀진 주요 지표들을 사용하여, 우리의 제안은 이전 연구에서 제시된 것보다 더 깊은 인사이트를 제공합니다.

- **Performance Highlights**: MetaToken은 네 가지 최신 LVLM에서 테스트되었으며, ROC 곡선 면적(Area Under Receiver Operator Characteristic Curve)에서 최대 92.19%와 정밀도-재현율 곡선(Area Under Precision Recall Curve)에서 최대 82.69%를 기록했습니다. 또한, MetaToken을 LURE 환각 완화 방법에 통합할 경우, 최대 56.62%의 환각 감소 효과를 확인할 수 있었습니다.



### Cephalo: Multi-Modal Vision-Language Models for Bio-Inspired Materials Analysis and Design (https://arxiv.org/abs/2405.19076)
- **What's New**: 새로운 혁신으로, Cephalo라는 멀티모달 비전 대형 언어 모델(V-LLMs)이 재료 과학 애플리케이션을 위해 개발되었습니다. Cephalo는 시각적 데이터와 언어 데이터를 통합하여 인간-AI와 다중 에이전트 AI 프레임워크 내에서 향상된 이해와 상호작용을 가능하게 합니다. 특히 PDF 문서에서 이미지와 해당 설명 텍스트를 정교하게 추출하는 알고리즘을 도입한 데이터셋 생성 방법이 주요 혁신입니다.

- **Technical Details**: Cephalo의 핵심 기술 중 하나는 과학 논문에서 이미지와 텍스트 데이터를 정밀하게 추출하고 이미지-텍스트 쌍을 정련하는 통합 비전 및 언어 처리 방식을 포함하는 것입니다. 모델은 수천 개의 과학 논문과 과학 중심의 Wikipedia 페이지에서 추출한 데이터로 훈련되었으며, 복잡한 시각적 장면을 해석하고 정확한 언어 설명을 생성하며 이미지에 대한 질문에 효과적으로 답변할 수 있습니다. 비전 인코더와 오토회귀 변환기(autoregressive transformer)를 결합하여 복잡한 자연어 이해를 지원하는 통합 모델을 구성하였습니다.

- **Performance Highlights**: Cephalo 모델은 생물학적 재료, 파괴 및 공학 분석, 단백질 생물물리학, 곤충 행동을 기반으로 한 바이오 영감 설계 등 다양한 사용 사례에서 정확성과 효과를 입증했습니다. 특히, 태양의 일식 사진을 기반으로 한 바이오 영감 재료 미세구조 생성을 포함한 생성적 애플리케이션에서 뛰어난 성능을 보였습니다.



### DiveR-CT: Diversity-enhanced Red Teaming with Relaxing Constraints (https://arxiv.org/abs/2405.19026)
- **What's New**: DiveR-CT는 자동 적대적 팀(targeting)이 기존 접근 방식에서의 다양성 손실 문제를 해결하고자 새로운 제약 완화 최적화 프레임워크를 도입했습니다. 이 방법은 안전성 평가 과정에서 더 다양한 데이터를 생성하고, 블루 팀(blue team) 모델의 회복력을 향상시키며, 목표 가중치를 동적으로 조절하여 일관된 성공률을 제공합니다.

- **Technical Details**: {'Algorithm Framework': 'DiveR-CT는 기존의 최적화 편향 용어를 제약 최적화 프레임워크로 재구성합니다. 비안전한 보상을 최대화하는 대신 이 보상을 임계 제약으로 취급하여 정책이 다양성 지표를 최적화하는 자유도를 확장시킵니다.', 'Semantic Diversity Rewards': '생성된 데이터 히스토리의 임베딩(nearest neighbors)에서 동적으로 목표를 설정하여 정책이 의미론적 공간을 고르게 커버할 수 있도록 합니다. 이는 이전의 접근 방식이 시간이 지남에 따라 효과가 감소하는 문제를 해결합니다.'}

- **Performance Highlights**: {'Metrics': 'DiveR-CT는 기존 알고리즘보다 다양한 설정에서 더 높은 다양성 지표를 달성하며 공격 성공률을 제어할 수 있습니다.', 'Blue Team Resilience': '수집된 데이터를 기반으로 블루 팀의 모델을 더욱 안전하게 만듭니다. 더욱 회복력 있는 라마(Llama) 안전 모델에서도 강력한 성능을 보여줍니다.', 'Reward Overoptimization': '보상 최적화 문제를 완화하여 더 신뢰할 수 있는 결과를 도출합니다.'}



### EasyAnimate: A High-Performance Long Video Generation Method based on Transformer Architectur (https://arxiv.org/abs/2405.18991)
Comments:
          6 pages, 5 figures

- **What's New**: EasyAnimate는 고성능 비디오 생성을 위한 새로운 방법으로, Transformer 아키텍처를 활용합니다. DiT 프레임워크를 확장하여 3D 비디오 생성을 지원하며, 모션 모듈 블록을 추가해 시간적 역학(temporal dynamics)을 포착합니다.

- **Technical Details**: EasyAnimate는 PixArt-α 기반으로 구축되었으며, 텍스트 인코더, 비디오 VAE(비디오 인코더와 비디오 디코더), 그리고 Diffusion Transformer(DiT)로 구성돼 있습니다. T5 인코더(Raffel et al., 2020)를 텍스트 인코더로 사용하며, Slice VAE와 MagViT를 도입하여 시간 축의 압축 효율을 개선했습니다. 모션 모듈은 시간적 데이터를 활용하여 프레임 간의 자연스러운 움직임을 가능하게 합니다.

- **Performance Highlights**: EasyAnimate는 현재 144 프레임의 비디오를 생성할 수 있는 성능을 보유하고 있으며, 클라우드 GPU에서 긴 비디오 시퀀스를 효과적으로 처리할 수 있도록 Slice 메커니즘을 활용합니다. 코드와 전체적인 비디오 제작 생태계를 아우르는 훈련 파이프라인을 제공합니다.



### Kestrel: Point Grounding Multimodal LLM for Part-Aware 3D Vision-Language Understanding (https://arxiv.org/abs/2405.18937)
- **What's New**: Kestrel이라는 새로운 접근법을 소개합니다. 이는 3D MLLMs(3D 다중 모달 대형 언어 모델)에게 부분 인지 능력을 부여하여, 3D 객체의 부분 수준에서 해석하고 세분화할 수 있게 합니다. 이를 위해 두 가지 새로운 과제를 제안합니다: 1) Part-Aware Point Grounding, 사용자 지침에 따라 부분 수준의 세그먼트 마스크를 예측하는 과제와 2) Part-Aware Point Grounded Captioning, 부분 수준의 설명과 해당 마스크를 포함한 자세한 캡션을 제공하는 과제입니다. 또한, 이러한 과제들을 지원하고 평가하기 위해 3DCoMPaT Grounded Instructions Dataset (3DCoMPaT-GRIN)을 소개합니다.

- **Technical Details**: Kestrel 모델은 3D 세분화 그라운딩 모듈을 통합하여 부분별로 이해하도록 설계되었습니다. 3DCoMPaT-GRIN Vanilla는 789k 점 클라우드-지시-세그먼트 마스크 트리플렛으로 구성되어 있으며, 이는 부분 인식 세분화 그라운딩 평가에 사용됩니다. 3DCoMPaT-GRIN Grounded Caption은 107k 점 클라우드-지시-그라운드된 캡션 트리플렛으로 구성되어, MLLMs의 부분 인식 언어 이해력과 세분화 그라운딩 능력을 평가합니다. 또한, 새로운 데이터 주석 파이프라인을 통해 시각적 특징을 포함한 포괄적인 설명과 세분화 마스크를 생성하는 데 InstructBLIP 및 ChatGPT/GPT-4를 사용하였습니다.

- **Performance Highlights**: 3DCoMPaT-GRIN에서의 광범위한 실험 결과, Kestrel은 사용자 지정 세분화 마스크를 생성할 수 있으며, 이는 현재 존재하는 모든 3D MLLM에서는 볼 수 없는 능력입니다. Kestrel은 새로운 과제에서 뛰어난 성능을 발휘하며, 부분 인식 언어 이해와 세분화 그라운딩 측면에서 기준점을 세웠습니다.



### Are queries and keys always relevant? A case study on Transformer wave functions (https://arxiv.org/abs/2405.18874)
Comments:
          9 pages, 4 figures

- **What's New**: 이 논문에서는 자연어 처리(NLP) 작업에서 주로 사용되는 dot product attention 메커니즘이 양자 다체 스핀 해밀토니언의 기저 상태를 근사하는 경우에도 적합한지 조사하고 있습니다. 특히, 2차원 $J_1$-$J_2$ Heisenberg 모델에 대한 수치 시뮬레이션을 수행하여, 쿼리(queries)와 키(keys)를 제외하고 위치만을 기반으로 한 단순화된 주의 메커니즘을 사용해도 경쟁력 있는 결과를 얻을 수 있음을 입증하였습니다.

- **Technical Details**: Transformer 구조의 주의 메커니즘이 양자 다체 스핀 해밀토니언의 기저 상태를 근사하는 데 어떻게 사용될 수 있는지 조사했습니다. 표준 attention 메커니즘을 쿼리와 키를 제외한 단순화된 메커니즘과 비교하여 성능을 평가했습니다. 또한, 주의 맵(attention maps)을 분석하여, 쿼리와 키를 포함한 아키텍처의 최적화 과정이 입력 독립적인 주의 가중치를 효과적으로 생성함을 나타냈습니다.

- **Performance Highlights**: 쿼리와 키를 포함하지 않은 단순화된 attention 메커니즘이 표준 메커니즘에 비해 성능 저하 없이 계산 비용과 파라미터 사용을 줄일 수 있음을 확인했습니다. 이는 대규모 시스템에서 더 효과적으로 작동하며, NLP와 같은 다른 도메인에서도 동일한 원칙을 적용할 수 있음을 발견했습니다.



### LLMs achieve adult human performance on higher-order theory of mind tasks (https://arxiv.org/abs/2405.18870)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)이 고차원 마음 이론(ToM, Theory of Mind)을 얼마나 잘 이해할 수 있는지를 조사합니다. ToM은 인간이 다수의 정신적 및 감정적 상태를 순환적으로 추론하는 능력을 말합니다(예: 내가 너가 그녀가 알고 있다는 것을 믿는다고 생각한다). 이 논문에서는 새로운 손글씨 테스트인 Multi-Order Theory of Mind Q&A(MoToMQA)를 도입하여 다섯 가지 LLM의 성능을 새롭게 수집된 성인 인간 기준과 비교합니다.

- **Technical Details**: MoToMQA는 인간 성인을 위한 ToM 테스트를 기반으로 하며, 짧은 형식의 이야기에서 등장인물에 대한 참/거짓 질문을 포함합니다. GPT-4와 Flan-PaLM 모델이 전체적으로 성인 수준 또는 그 가까운 ToM 과제 성능을 보였으며, 특히 GPT-4는 6차 추론에서 성인의 성능을 초과했습니다. LLM 크기와 미세 조정이 ToM 능력 실현에 중요한 역할을 한다는 것을 발견했습니다. 또한, MoToMQA 벤치마크는 각 이야기마다 다수의 순수 사실과 ToM 주문에 대한 질문들로 구성되어 있습니다.

- **Performance Highlights**: GPT-4와 Flan-PaLM은 전반적으로 성인 수준의 ToM 과제 성능을 달성했습니다. 특히 GPT-4는 6차 추론에서 성인을 능가하는 성과를 보였습니다. 이는 사용자 대상 LLM 애플리케이션에서의 협력적 및 경쟁적 행동 수행 가능성을 시사합니다.



### LMO-DP: Optimizing the Randomization Mechanism for Differentially Private Fine-Tuning (Large) Language Models (https://arxiv.org/abs/2405.18776)
Comments:
          18 pages, 15 figures

- **What's New**: 최신 논문에서, 대규모 사전 학습된 언어 모델(Large-scale Pre-trained Language Models)을 강화된 프라이버시 보호를 통해 정밀하게 튜닝하기 위한 새로운 메커니즘, LMO-DP(Language Model-based Optimal Differential Privacy)을 제안했습니다. 기존 밀집화된 Gaussian 메커니즘을 대신하여, 더욱 정확한 정밀 튜닝을 가능하게 하는 최적/서브-최적 Differential Private 메커니즘을 소개합니다. 특히, $	ext{privacy budget} ε < 3$와 같은 강력한 프라이버시 레짐에서도 효율성을 극대화할 수 있습니다.

- **Technical Details**: LMO-DP 메커니즘은 두 단계의 혼합 분포(two-fold mixture distribution)에서 생성된 새로운 LMO 노이즈를 사용합니다. 첫 번째 단계는 Laplace 분포이고, 두 번째 단계는 가능한 선형 가중 확률 밀도 함수(pdf) 조합으로 구성됩니다. 이를 통해 Gaussian 메커니즘의 한계를 극복하고, DP-SGD와 같은 SOTA 메서드의 프라이버시 보장을 유지합니다. 또한, 새로운 오프라인 최적 노이즈 검색 방법을 통해 노이즈의 크기를 현저히 줄이는 것을 목표로 합니다.

- **Performance Highlights**: LMO-DP를 활용하여 RoBERTa-large, Llama-2와 같은 대규모 언어 모델을 프라이버시 보장을 유지하며 정밀 튜닝할 수 있습니다. 예를 들어, $	ext{ε=0.3, δ=10^{-10}}$일 때, SST-2 데이터셋에서 RoBERTa-large 모델의 정확도를 기존의 50%에서 92.20%로 극적으로 향상시켰습니다. 텍스트 생성 작업에서도 GPT-2 모델을 통해 유사한 성과를 달성했습니다. LMO-DP는 다양한 언어 모델 태스크에서 뛰어난 성능과 수렴 속도를 보여줍니다.



### Musical Phrase Segmentation via Grammatical Induction (https://arxiv.org/abs/2405.18742)
Comments:
          Extended version of a paper appearing in the proceedings of IJCAI 2024 that includes additional material in an appendix. Please cite the IJCAI version

- **What's New**: 이 연구는 문법 유도 알고리즘(grammatical induction algorithms)을 사용하여 음악적인 구문 분할 문제를 해결하는 방법을 제시합니다. 이 논문의 목표는 기존의 문법 유도 알고리즘을 평가하고 그 성능을 비교하는 것입니다.

- **Technical Details**: 연구진은 다섯 가지 문법 유도 알고리즘, 즉 LZ78, Repair, LongestFirst, MostCompressive, Sequitur를 세 가지 데이터셋에 적용하여 다양한 음악적 관점을 조합해 분석했습니다. 입력 인코딩에는 길이(duration) 관점을 포함시키는 것이 최상의 성능을 이끌어냈습니다. 학습 데이터셋은 Johannes Kepler University Patterns Development Dataset, Essen Folksong Collection, Latter-day Saint Hymns Annotation Collection으로 구성되었고, 각 데이터셋에서 수집된 음악 시퀀스와 주석이 사용되었습니다.

- **Performance Highlights**: 가장 눈에 띄는 성과는 LONGESTFIRST 알고리즘이 세 가지 데이터셋 모두에서 최고의 F1 점수를 기록했다는 것입니다. 길이 관점을 포함한 입력 인코딩이 가장 높은 성능을 나타냈습니다. 최종적으로 모든 코드와 데이터는 GitHub 저장소(https://github.com/reedperkins/grammatical-induction-phrase-segmentation)를 통해 공개되었습니다.



### Correctable Landmark Discovery via Large Models for Vision-Language Navigation (https://arxiv.org/abs/2405.18721)
Comments:
          Accepted by TPAMI 2024

- **What's New**: 새로운 VLN(Vision-Language Navigation) 패러다임인 CONSOLE(COrrectable LaNdmark DiScOvery via Large ModEls)을 도입하였습니다. 이는 ChatGPT와 CLIP 두 가지 대형 모델을 사용하여 개방형 세계(sequential landmark discovery)에서 규정 가능한 표지물(discoverable landmarks)을 찾아내는 방법을 제안합니다.

- **Technical Details**: CONSOLE에서는 ChatGPT로부터 풍부한 개방형 세계의 표지물 공동 발생 상식을 추출하고 CLIP를 이용하여 이 상식에 기반한 표지물 발견을 수행합니다. priors의 시각적 제약 부족으로 인한 노이즈를 줄이기 위해, 학습 가능한 공동 발생 점수 모듈(cooccurrence scoring module)을 도입하여 실제 관찰에 따라 각 공동 발생의 중요성을 수정합니다. 이 연구는 R2R, REVERIE, R4R 및 RxR 등의 VLN 벤치마크에서 평가되었습니다.

- **Performance Highlights**: CONSOLE은 강력한 기존 모델들을 크게 능가하며, 특히 R2R와 R4R의 새로운 최신 상태(state-of-the-art)을 달성하였습니다. 이는 노이즈를 억제하면서 조정 가능한 방식으로 대형 모델을 통해 유용한 탐색 지식을 활성화하고 활용하는 것을 보여줍니다.



### Calibrating Reasoning in Language Models with Internal Consistency (https://arxiv.org/abs/2405.18711)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)의 연쇄적 사고(chain-of-thought, CoT) 유도 기법을 통한 추론 과정에 대한 심층 분석을 제시합니다. 주요한 발견은 내부 표현(internal representations)에서 중간 레이어와 마지막 레이어 간의 불일치가 나타나면서, 이로 인해 모델의 추론 신뢰성에 대한 의문이 제기된다는 점입니다. 이를 해결하기 위해 내부 일관성(internal consistency)이라는 새로운 측정 방법을 제안하여 모델의 추론 과정을 교정(calibrate)합니다.

- **Technical Details**: 주로 디코더 전용 트랜스포머 아키텍처(decoder-only Transformer architecture)를 사용하여 분석을 진행했습니다. 각 토큰의 중간 활성화(intermediate activations)을 확인하고, 중간 레이어와 최종 레이어의 표현이 얼마나 일치하는지를 측정합니다. 이를 통해 추론 경로의 신뢰성을 평가하고, 내부 일관성이 높은 경로에 가중치를 두는 방법을 통해 성능을 향상시켰습니다. 또한, 내부 불일치를 야기하는 요인으로 주목받는 트랜스포머의 어텐션(attention) 및 피드-포워드 네트워크(feed-forward networks) 모듈을 분석했습니다.

- **Performance Highlights**: 내부 일관성을 사용하여 LLM의 추론 경로를 평가한 결과, 정확한 추론 경로와 잘못된 추론 경로를 효과적으로 구분해냈습니다. 내부 일관성이 높은 경로에 가중치를 둔 결과, 모델의 추론 성능이 크게 향상되었습니다. 특히, Llama-2-7B 모델과 PrOntoQA 데이터셋을 사용한 실험에서 이 점이 명확히 드러났습니다.



### Efficient Preference-based Reinforcement Learning via Aligned Experience Estimation (https://arxiv.org/abs/2405.18688)
- **What's New**: 이 연구에서는 SEER라는 새로운 Preference-based Reinforcement Learning (PbRL) 방법을 제안합니다. 기존 PbRL은 인간의 선호 피드백(human feedback)에 크게 의존하지만, SEER는 label smoothing과 policy regularization 기술을 통합하여 이러한 의존도를 줄이고 학습 효율성을 크게 향상시킵니다.

- **Technical Details**: SEER는 두 가지 주요 기술을 사용합니다. 첫째, label smoothing은 인간의 선호 레이블을 부드럽게 하여 reward 모델의 overfitting 문제를 완화합니다. 둘째, policy regularization은 현재의 replay memory에서 잘 지원되는 상태-액션 쌍을 사용하여 보수적인 Q 추정치를 만들고, 이를 통해 정책 학습을 정규화합니다. 보수적인 Q 추정은 overestimation bias를 줄이는 데 효과적입니다.

- **Performance Highlights**: SEER는 다양한 복잡한 작업(online 및 offline 환경)에서 최첨단 방법들을 능가하는 성능을 보여주었습니다. 특히 제한된 인간 피드백 상황에서도 높은 효율성을 발휘하며, Q 함수와 정책을 더욱 정확하게 훈련한다는 점에서 기존 방법들보다 우수합니다.



### LLM-based Hierarchical Concept Decomposition for Interpretable Fine-Grained Image Classification (https://arxiv.org/abs/2405.18672)
- **What's New**: 최근 비전-언어 작업(Vision-Language tasks)을 위한 해석 가능한 모델에서 성능이 경쟁력 있지만, 종종 대형 언어 모델(LLMs)의 비구조화 텍스트 출력에 의존하여 해석 가능성이 저하됩니다. 이에 새로운 프레임워크인 	exttt{Hi-CoDe} (Hierarchical Concept Decomposition)를 도입했습니다. 이 프레임워크는 구조화된 개념 분석을 통해 모델의 해석 가능성을 향상시키기 위해 설계되었습니다.

- **Technical Details**: 우리의 접근법은 두 가지 주요 구성 요소로 구성됩니다. 첫째, GPT-4를 사용하여 입력 이미지를 시각적 개념의 구조화된 계층으로 분해하여 시각적 개념 트리를 형성합니다. 둘째, CLIP에서 파생된 개념별 특징을 기반으로 간단한 선형 분류기의 앙상블을 사용하여 분류를 수행합니다. 이 방식은 투명성을 제공하며 모델의 의사 결정 과정을 명확히 이해할 수 있게 해 줍니다. 또한, 다양한 개념의 중요성을 강조하여 잠재적인 실패 모드를 세부적으로 분석할 수 있습니다.

- **Performance Highlights**: 우리의 접근법은 SOTA 모델의 성능을 따라가면서도, 해석 가능성 측면에서 새로운 벤치마크를 설정합니다. 이는 투명성을 제공하고, 오류 분석 및 디버깅을 용이하게 하며, 특징 업데이트에 대한 유연성을 제공합니다. 우리 모델은 주목할만한 데이터셋에서 실험을 통해 전통적인 분류 모델과 유사한 성능을 보여주었습니다.



### Zipper: A Multi-Tower Decoder Architecture for Fusing Modalities (https://arxiv.org/abs/2405.18669)
Comments:
          Under review at NeurIPS

- **What's New**: 새로운 연구인 Zipper는 다른 모달리티(modality)로 독립적으로 학습된 생성 모델들을 통합하여 멀티모달(multimodal) 생성 모델을 효과적으로 구축하는 혁신적 아키텍처를 제안합니다. 특징적인 점은 크로스 어텐션(cross-attention)을 이용해 독립적으로 사전 학습된 언리모달(unimodal) 디코더들을 유연하게 조합하는 것입니다. 이 구성으로 인해 각 모달리티별 독립적인 생성 성능을 최대한 유지하면서도 멀티모달 생성 능력을 확보할 수 있습니다.

- **Technical Details**: Zipper는 멀티 타워 디코더 아키텍처(multi-tower decoder architecture)로 구성되어 있습니다. 크로스 어텐션을 활용하여 독립적으로 사전 학습된 유니모달 디코더를 '지퍼(zip)'처럼 결합합니다. 이 접근 방식을 통해 많은 양의 정렬된 데이터 없이도 제한된 크로스 모달 데이터로 다중 모달 생성 능력을 확보할 수 있습니다. 주요 실험에서는 음성-텍스트 데이터에 대해 Zipper의 성능을 평가했으며, 텍스트 생성 등에서 각 디코더 타워를 동결(freezing)시킴으로써 특정 모달리티 성능을 유지할 수 있음을 확인했습니다.

- **Performance Highlights**: Zipper 아키텍처는 제한된 양의 정렬된 텍스트-음성 데이터에서도 경쟁력 있는 성능을 보였습니다. 예를 들어, 텍스트 백본(text backbone)을 동결할 경우 자동 음성 인식(ASR) 작업에서 성능 저하가 거의 없으며, 텍스트-음성 생성(TTS) 작업에서는 사전 학습된 음성 백본을 사용할 때 베이스라인과 비교해 더 나은 성능을 보였습니다. 특히 워드 에러 레이트(WER) 측면에서 40%의 상대적 오차 감소를 달성했습니다.



### JADS: A Framework for Self-supervised Joint Aspect Discovery and Summarization (https://arxiv.org/abs/2405.18642)
Comments:
          preprint

- **What's New**: 본 논문에서는 텍스트 문서의 다중 측면을 포함하는 요약을 생성하기 위해 'Joint Aspect Discovery and Summarization (JADS)' 알고리즘을 소개합니다. 기존 방식들은 요약과 클러스터링 알고리즘을 별도로 최적화하려 했으나, JADS는 이 두 가지 작업을 하나의 단계로 통합하여 보다 효율적인 요약을 가능하게 합니다. 자가 지도 학습(self-supervised learning)을 활용하여 대규모 라벨링된 데이터셋을 생성하고, 이를 통해 보다 고성능의 요약을 달성합니다.

- **Technical Details**: JADS는 Longformer(Beltagy et al., 2020)를 기반으로 하는 트랜스포머 기반 인코더-디코더 모델을 사용합니다. 긴 문서를 GPU 메모리에 적합하게 처리하면서 시간과 메모리 복잡도에서 균형을 맞출 수 있습니다. 모델은 주어진 텍스트 내 주제를 탐지하고 각각에 대해 요약을 생성합니다. 학습을 위해 Wikipedia와 CNN/Daily Mail 데이터를 활용하여 대규모 ADS (Aspect Discovery and Summarization) 데이터셋을 생성합니다. 다양한 서브 서머리가 [SEP] 토큰으로 연결된 형태로 요약물을 생성합니다.

- **Performance Highlights**: JADS 모델은 기존의 두 단계 방식의 베이스라인보다 우수한 성능을 보였습니다. 특히 Wikipedia 데이터셋으로 사전 학습을 실시한 후, CNN/Daily Mail 데이터셋으로 미세 조정을 통해 더 나은 성능과 안정성을 확보했습니다. JADS로부터 도출된 임베딩은 클러스터링 능력이 뛰어났으며, 사람 평가에서도 JADS의 요약이 실제 자료와 의미적으로 높은 정합성을 보였습니다.



### Improving Speech Decoding from ECoG with Self-Supervised Pretraining (https://arxiv.org/abs/2405.18639)
- **What's New**: 최신 연구에서는 두개 내 뇌-기계 인터페이스(Intracranial brain-machine interfaces)가 높은 정확도로 음성 언어를 디코딩(Decoding)할 수 있음을 보여주고 있습니다. 이 시스템은 신경 활동(Neural activity)에서 텍스트로 매핑하는 딥 뉴럴 네트워크(Deep neural networks)를 훈련함으로써 이를 수행합니다. 그러나 이러한 네트워크는 많은 양의 라벨링된 데이터가 필요하며, 이는 인체 침습성(Invasive) 신경 기록을 위해서는 특히 부담스럽습니다. 새 연구에서는 이러한 문제를 해결하기 위해 self-supervised 학습 방식을 사용하여 wav2vec 모델을 재구성해 electrocorticographic (ECoG) 데이터를 처리하는 방법을 제안합니다.

- **Technical Details**: wave2vec는 self-supervised, 완전히 convolutional 모델로, 오디오의 잠재 표현(Latent representations)을 학습하기 위해 noise-contrastive 손실을 사용합니다. 연구팀은 이 모델을 라벨링되지 않은 ECoG 기록으로 훈련시키고, 이후 라벨링된 음성 세션의 ECoG 데이터를 wave2vec의 표현 공간으로 변환한 다음, 이를 텍스트로 매핑하는 supervised 인코더-디코더(Encoder-Decoder)를 훈련시킵니다. 이 과정에서 다양한 수의 라벨링된 블록으로 실험을 진행했습니다.

- **Performance Highlights**: 새로운 표현 공간에서 디코딩 성능이 원래의 ECoG 데이터보다 우수함을 확인했으며, 아무 경우에도 성능이 더 나빠지지 않았습니다. 일부 경우에서는 다른 환자의 데이터로 wave2vec를 사전 훈련(Pretraining)하여 성능을 개선할 수 있었습니다. 최상의 경우, wave2vec의 표현은 원래 데이터 대비 단어 오류율(Word error rates)을 50% 이상 감소시켰습니다.



### A Theoretical Understanding of Self-Correction through In-context Alignmen (https://arxiv.org/abs/2405.18634)
- **What's New**: 인공지능 연구 논문은 최근 대형 언어 모델(LLMs)이 인간처럼 자가 수정(self-correction)을 통해 성능을 향상시킬 수 있다는 증거를 제시합니다. 이 연구는 이러한 능력이 어떻게 나타나는지 이론적으로 분석하여, LLM이 스스로 평가한 보상을 통해 응답을 개선할 수 있는 능력을 입증합니다. 특히, 소프트맥스 주의(attention), 멀티-헤드 주의(multi-head attention), MLP 블록 등 현실적인 트랜스포머(transformer) 설계가 자가 수정에 중요한 역할을 한다는 점을 강조합니다.

- **Technical Details**: 이 연구는 컨텍스트 내 정렬(in-context alignment) 작업을 통해 자가 수정을 이론적으로 분석합니다. 모델이 자체 수정 단계를 컨텍스트로 제공받아 보상이 높은 응답을 생성하도록 목표를 설정한 인과적 정렬(ICA)로 설명합니다. 이를 통해, 다층 트랜스포머가 일반적인 정렬 목표를 최적화할 수 있는 모델 가중치의 존재를 이론적으로 입증하였습니다. 또한, 소프트맥스 주의, 멀티-헤드 주의, 그리고 MLP 블록과 같은 트랜스포머 설계 요소가 자가 수정 성능에 미치는 영향을 분석했습니다.

- **Performance Highlights**: 합성 데이터셋과 실제 실험을 통해 이론적 설명을 검증했으며, 자가 수정 과정에서 사회적 편향 완화와 'jailbreak' 공격 방어 등의 응용도 제시되었습니다. Vicuna-7b와 Llama2-7b-chat 모델에서 자가 수정 절차를 통해 사회적 편향을 감소시키고, jailbreak 공격의 성공률을 크게 낮출 수 있음을 확인했습니다. 예를 들어, 공격 성공률이 95%에서 2%로 감소했습니다.



### Hardware-Aware Parallel Prompt Decoding for Memory-Efficient Acceleration of LLM Inferenc (https://arxiv.org/abs/2405.18628)
Comments:
          The code for this implementation is available at this https URL

- **What's New**: 새로운 블록에서 우리는 대규모 언어 모델(LLMs)에서 발생하는 하드웨어 성능 저하를 해결하기 위해 새로운 병렬 프롬프트 디코딩(parallel prompt decoding, PPD)을 제안합니다. 이 접근법은 단 0.0002%의 학습 가능한 매개변수만을 필요로 하며, A100-40GB GPU에서 16시간 만에 효율적으로 학습이 가능합니다. 이는 기존의 추론 기법들을 대체할 수 있는 잠재력을 가지고 있으며, 코드가 공개되어 있습니다.

- **Technical Details**: PPD는 다수의 프롬프트 토큰을 사용하여 앞으로의 시간 단계에서 생성될 출력을 병렬적으로 예측하는 방식으로, 다중 토큰 생성을 위한 일부 조건 종속성 정보를 복원합니다. 이를 통해 장기 예측에 대해 최대 28% 더 높은 승인율을 보입니다. 또한, 하드웨어 인식적 동적 희소 트리(hardware-aware dynamic sparse tree) 기법을 통해 다양한 GPU에서의 계산 용량을 최대로 활용하도록 디코딩 스키마를 최적화합니다. KD(Knowledge Distillation)는 예측 정확도 향상에 큰 기여를 합니다.

- **Performance Highlights**: MobileLlama부터 Vicuna-13B까지의 다양한 벤치마크에서 최대 2.49배의 속도 향상을 보였으며, 런타임 메모리 오버헤드는 최소 0.0004%에 불과합니다. 병렬 프롬프트 디코딩은 기존의 추론적 디코딩과 합쳐져 최대 1.22배의 추가 속도 향상을 함께 보여줍니다. 다양한 실험에서, EPT(External Prompt Tokens)의 수를 늘릴수록 학습 손실이 더 효과적으로 감소하며, 예측 정확도가 향상되는 경향을 보였습니다.



### RealitySummary: On-Demand Mixed Reality Document Enhancement using Large Language Models (https://arxiv.org/abs/2405.18620)
- **What's New**: RealitySummary는 실시간 혼합 현실(MR) 문서 개선 도구를 도입하여, 사전 처리가 필요없는 다양한 문서에 대해 즉석 텍스트 추출, 요약 및 증강을 제공합니다. Microsoft Hololens 2와 Apple Vision Pro를 사용하여 사용자가 인쇄물 및 디지털 문서에 대해 다양한 증강 기능을 실시간으로 경험할 수 있게 합니다.

- **Technical Details**: 이 시스템은 Google Cloud OCR과 GPT-4를 활용하여 텍스트를 자동으로 추출하고 요약합니다. 혼합 현실 환경에서 문서 주변에 정보가 임베딩되도록 Microsoft Hololens 2와 Apple Vision Pro를 사용합니다. 사용자는 음성 명령을 통해 구글의 음성 API를 사용하여 요약 기능을 활성화하거나 현재 컨텐츠와 관련된 질문을 할 수 있습니다. 증강 기능으로는 요약(summaries), 비교 테이블(comparison tables), 타임라인(timelines), 키워드 목록(keyword lists), 요약 강조(summary highlighting), 정보 카드(information cards)가 포함됩니다.

- **Performance Highlights**: 두 개의 사용자 연구(유용성 연구, 실생활 연구)를 통해, 12명의 사용자가 참여한 유용성 연구와 11명의 사용자가 참여한 실생활 연구는 사전 준비 없이 다양한 읽기 자료에 대해 서 사용자가 즉석 MR 문서 개선의 고유한 이점을 강조했습니다. 사용자는 항상 작동하는 기능을 통해 암시적 상호 작용을 가능하게 하는 혼합 현실 인터페이스를 높이 평가하였으며, 공간적이고 물리적인 탐색을 촉진하여 AI가 생성한 콘텐츠의 적용성을 다양한 시나리오에서 크게 향상시켰습니다.



### Low-rank finetuning for LLMs: A fairness perspectiv (https://arxiv.org/abs/2405.18572)
- **What's New**: Low-rank fine-tuning (로라, LoRA) 기술이 대형 언어 모델(LLMs)의 세부 조정에서 사용자 데이터 분포의 변화를 효과적으로 학습하지 못할 수 있다는 결과를 발표합니다. 특히, 기존 모델의 유해한 행동이나 편향을 줄이기 위한 세부 조정에서 이러한 단점이 두드러지게 나타났습니다.

- **Technical Details**: LoRA 방법은 이전에 학습된 모델의 매개변수를 고정하고, 적응층(adaptor layer)이 있는 낮은 차원의 투영 행렬(projection matrix)을 학습하여 이를 미세 조정된 매개변수로 매핑합니다. 이는 전통적인 전체 미세 조정 방식에 비해 계산 및 메모리 효율성을 크게 개선합니다. 그러나 LoRA가 데이터 분포의 변화를 충분히 포착하지 못하면, 세부 조정 데이터세트의 중요한 정보를 손실할 수 있습니다.

- **Performance Highlights**: 본 연구는 여러 모델, 데이터셋, 태스크에 대한 실험적 증거를 통해 LoRA 방식이 원래 모델의 편향적이고 유해한 행동을 보존하는 경향이 있음을 보여줍니다. 특히 낮은 차원의 랭크를 사용할 때 이러한 문제가 더 두드러져, 각 그룹 간의 성능 불균형이 발생할 수 있습니다. LoRA 방식의 미세 조정된 모델들은 원래 모델의 특성을 여전히 많이 보존하는 것으로 나타났습니다.



### Its Not a Modality Gap: Characterizing and Addressing the Contrastive Gap (https://arxiv.org/abs/2405.18570)
- **What's New**: 이 논문은 CLIP과 같은 멀티 모달 대조 모델(multi-modal contrastive model)에서 발견된 모달리티 갭(modality gap)이 실제로는 대조 손실에 의해 생성된다는 새로운 사실을 제시합니다. 이 갭을 '대조 갭(contrastive gap)'으로 재명명하고, 이를 수정하기 위해 단일 모달 대조 손실의 균일성(uniformity)과 정렬(alignment) 특성을 멀티 모달 설정으로 적용하는 방법을 제안합니다.

- **Technical Details**: 멀티 모달 대조 모델은 다양한 모달에서 입력을 통합된 표현 공간(representational space)으로 매핑합니다. CLIP 모델은 이미지와 텍스트를 이 공간에 매핑해 연관된 텍스트를 예측할 수 있게 합니다. 그러나 기존 연구에서는 모달리티 갭이 이미지와 텍스트 임베딩(embedding)이 서로 다른 영역에 존재하는 현상으로 보고되었으며, 이는 높은 차원의 CLIP 공간에서 대조 손실(contrastive loss)에 의해 저차원 매니폴드(manifold)를 차지하게 되는 방식에 의해 발생한다고 설명합니다.

- **Performance Highlights**: 제안된 수정된 CLIP 손실을 사용하여 대조 갭을 줄임으로써, 임베딩을 더 균일하게 분포시킬 수 있음을 실험으로 증명하였습니다. 이는 더 나아가 다운스트림 작업에서 기본 CLIP 손실보다 더 나은 성능을 달성하도록 만들어주며, 제로샷 이미지 분류(zero-shot image classification)와 멀티 모달 연구에서 특히 유리합니다.



### Automatic detection of cognitive impairment in elderly people using an entertainment chatbot with Natural Language Processing capabilities (https://arxiv.org/abs/2405.18542)
- **What's New**: 이 논문에서는 노인의 인지 장애를 모니터링하기 위한 새로운 지능형 대화 시스템(intelligent conversational system)을 소개합니다. 기존의 수동 테스트 방식의 문제점을 극복하고, 사용자가 관심 있는 뉴스를 통해 엔터테인먼트를 제공하면서 인지 장애를 투명하게 모니터링합니다. 이 시스템은 자동 대화 단계를 통해 인지 장애를 평가하며, 이를 위해 자연어 생성(Natural Language Generation) 기술을 활용합니다.

- **Technical Details**: 이 시스템은 최신 뉴스 항목에서 자동으로 대화 플로우를 생성하고, 질문에 대한 정답(gold standard)을 추론하여 사용자 응답과의 유사성을 비교합니다. 시스템은 0에서 1 사이의 유사성 메트릭(similarity metric)을 사용하여 평가하며, 텍스트 분석 특징을 기반으로 한 머신 러닝 알고리즘(Machine Learning algorithm)을 통해 자동으로 인지 장애를 검출합니다.

- **Performance Highlights**: 논문에서 제안된 시스템은 초기 치매 단계의 30명의 노인 테스트 그룹을 대상으로 현장 테스트를 수행했습니다. 실험 결과, 인지 장애가 없는 사용자는 최대 다섯 배 더 좋은 성과를 보였으며, 유사성 메트릭 값은 집중도와 스트레스 수준에 따라 0.03에서 0.36까지 변동했습니다. 제안된 알고리즘은 정확도, F-측정점, 재현율이 80%를 넘는 성능을 보였습니다.



### InversionView: A General-Purpose Method for Reading Information from Neural Activations (https://arxiv.org/abs/2405.17653)
- **What's New**: Neural 네트워크의 활성화 벡터(activation vectors)에 암호화된 정보를 해독하기 위한 새로운 방법인 InversionView를 소개합니다. InversionView는 활성화된 벡터를 조건으로 학습된 디코더 모델을 샘플링하여 이러한 입력의 하위 집합을 실질적으로 검사할 수 있게 합니다. 이를 통해 트랜스포머 모델(transformer models) 알고리즘을 이해하고 활성화 벡터의 정보 콘텐츠를 밝혀낼 수 있습니다.

- **Technical Details**: InversionView는 활성화 벡터의 정보 내용을 활성화를 발생시키는 입력 집합인 전상(preimage)으로 형식화할 수 있다는 직관에서 출발합니다. 활성화 벡터를 주어진 모델 하에서 특정 활성화를 발생시키는 입력 집합과 함께 탐색하여 정보를 해독합니다. 이를 위해 조건부 디코더 모델을 학습하고, 활성화 벡터를 조건으로 샘플링하여 유사한 활성화를 발생시키는 입력을 생성합니다. 이 사전 이미지는 활성화에 상당한 영향을 미치는 정보 조각만을 고려하여 해석합니다.

- **Performance Highlights**: 세 가지 사례 연구를 통해 InversionView의 유용성을 입증했습니다. 문자 카운팅(charater counting) 작업에서는 소형 트랜스포머가 정보를 처리하고 잊어버리는 방식을 밝혀냈습니다. Indirect Object Identification에서는 GPT-2 Small에서 활성화 벡터에 암호화된 정보를 쉽게 해석했습니다. 3자리 덧셈에서는 작업 수행을 위한 완전히 검증된 회로를 제공합니다. 모든 사례 연구에서 InversionView는 활성화 사이트마다 암호화된 정보에 대한 신속한 가설 생성을 가능하게 했습니다.



### SpinQuant: LLM quantization with learned rotations (https://arxiv.org/abs/2405.16406)
- **What's New**: 본 논문은 SpinQuant라는 새로운 기법을 제안합니다. SpinQuant는 Cayley 최적화(Cayley optimization)를 사용하여 회전 행렬(rotation matrices)을 최적화(또는 학습)합니다. 이 기법은 Large Language Models(대규모 언어 모델, LLMs)의 4-bit 양자화(quantization) 시, 메모리 사용량, 지연 시간(latency), 전력 소비를 크게 줄이는 역할을 합니다. 특히, LLaMA-2 7B 모델에서의 zero-shot(reasoning) 작업에서 풀 정밀도(full precision)와의 정확도 차이를 단 2.9 포인트로 줄여줍니다.

- **Technical Details**: SpinQuant는 회전 행렬(rotation matrices)을 최적화하기 위해 Cayley 최적화(Cayley optimization)를 소량의 검증 세트(validation set)를 사용하여 수행합니다. 회전 매개변수화(rotation parameterizations)를 활용하여, 특정 랜덤 회전이 양자화에 큰 차이를 만든다는 사실을 발견했습니다. 이 기법을 통해 일부 랜덤 회전이 다른 것에 비해 훨씬 더 나은 양자화를 이룰 수 있습니다.

- **Performance Highlights**: SpinQuant는 LLaMA-2 7B 모델에서 zero-shot reasoning 작업의 정확도 차이를 단 2.9 포인트로 줄였으며, 이는 LLM-QAT 대비 19.1 포인트, SmoothQuant 대비 25.0 포인트 더 우수합니다. 또한, 회전을 통해 outliers를 제거하는 동시 연구 QuaRot보다도 뛰어난 성능을 보입니다. 특히, LLaMA-2 7B모델과 LLaMA-3 8B 모델에서, SpinQuant는 QuaRot 대비 풀 정밀도와의 갭을 각각 30.2% 및 34.1% 줄였습니다.




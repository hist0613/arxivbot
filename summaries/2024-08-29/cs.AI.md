New uploads on arXiv(cs.CL)

### CoGen: Learning from Feedback with Coupled Comprehension and Generation (https://arxiv.org/abs/2408.15992)
Comments:
          17 pages, 9 figures

- **What's New**: 이번 연구는 언어 이해(comprehension)와 생성(generation) 능력을 결합하여 사용자와의 상호작용에서 지속적으로 학습하는 방법을 제안합니다. 두 능력을 통합하기 위한 새로운 기술이 도입되었습니다.

- **Technical Details**: 본 연구는 두 플레이어 참조 게임(two-player reference games)에서 진행되며, 수천 번의 사용자 상호작용을 통해 다양한 모델을 배포합니다. 상호작용 피드백 신호(interaction feedback signals)를 이용하여 학습하는 동시에, 이해와 생성 능력을 결합하여 성능을 개선합니다.

- **Performance Highlights**: 이 연구의 결과, 이해-생성 결합이 절대적으로 최대 26% 향상된 성능을 보여주고, 비결합 시스템에 비해 최대 17% 높은 정확도를 기록했습니다. 이러한 결합은 시스템의 언어에도 상당한 질적 영향을 미쳐, 더 인간적인 언어 사용을 가능하게 했습니다.



### BattleAgentBench: A Benchmark for Evaluating Cooperation and Competition Capabilities of Language Models in Multi-Agent Systems (https://arxiv.org/abs/2408.15971)
- **What's New**: 이번 논문에서는 언어 모델의 협력 능력을 측정하기 위해 새로운 베enchmark(BattleAgentBench)를 제안합니다. 이 벤치마크는 다양한 난이도의 세 가지 단계와 일곱 개의 하위 단계를 정의하고 있습니다.

- **Technical Details**: BattleAgentBench는 단일 에이전트의 내비게이션 능력, 쌍 에이전트의 작업 실행 능력, 다중 에이전트의 협력 및 경쟁 능력을 세분화하여 평가합니다. 기존 연구에서 다룬 적 없는 다중 에이전트 협업 및 경쟁 시나리오를 포함하였습니다.

- **Performance Highlights**: 평가 결과, API 기반 모델은 간단한 작업에서 탁월한 성능을 보였지만, 오픈 소스 소형 모델은 간단한 작업에서도 어려움을 겪었습니다. 협력 및 경쟁 능력이 필요한 어려운 작업에서는 API 기반 모델이 어느 정도 협력 능력을 보여주었으나, 개선 여지가 여전히 많습니다.



### LLM-Based Multi-Hop Question Answering with Knowledge Graph Integration in Evolving Environments (https://arxiv.org/abs/2408.15903)
- **What's New**: 이번 연구에서는 Large Language Models (LLMs)에서의 지식 편집 문제를 해결하기 위해 'Graph Memory-based Editing for Large Language Models (GMeLLo)'라는 새로운 접근법을 제안합니다. 기존 방법들의 한계를 극복하기 위해 LLM의 언어적 유연성과 Knowledge Graphs (KGs)의 명시적 지식 표현을 통합했습니다.

- **Technical Details**: GMeLLo는 LLM을 사용하여 문장에서 편집된 사실을 triple로 변환하고, 이를 통해 KG를 업데이트하여 정보를 적시에 반영합니다. 질문을 받을 경우 LLM을 활용하여 관계 체인을 추출한 후, 이를 정형 쿼리로 변환하여 업데이트된 KG를 검색합니다. 또한, KG에서의 정보를 기반으로 LLM에게 답변 생성을 요청하며, KG와 LLM의 답변이 상충할 경우 KG의 답변을 우선시합니다.

- **Performance Highlights**: GMeLLo는 MQuAKE 벤치마크에서 현재의 최첨단 지식 편집 방법들을 초월하는 성능을 입증하였으며, 특히 지식 수정이 광범위한 경우에도 우수한 다중 홉 질문 응답 능력을 보여주었습니다.



### Nexus: Specialization meets Adaptability for Efficiently Training Mixture of Experts (https://arxiv.org/abs/2408.15901)
- **What's New**: 이 논문에서는 Mixture of Experts (MoE) 아키텍처의 새로운 변형인 Nexus를 소개합니다. Nexus는 전문가 임베딩을 도메인 표현에서 학습하여 적응형 라우팅을 제공하여 초기 모델의 전문성을 유지하면서 새로운 작업에 쉽게 적응할 수 있도록 합니다.

- **Technical Details**: Nexus는 기존의 밀집 전문가 모델을 MoE 모델로 업사이클링(upcycling)하여 전문성을 향상시키면서 새로운 도메인을 수용할 수 있는 유연한 특징을 보여줍니다. Nexus의 라우터는 도메인 특정 데이터의 임베딩을 통해 각 도메인에 맞는 전문가 임베딩으로 프로젝션하도록 학습됩니다. 이는 새로운 데이터를 위한 전문가를 독립적으로 훈련하여 MoE 구조에 쉽게 추가할 수 있게 해줍니다.

- **Performance Highlights**: Nexus는 초기 업사이클링에서 기준선에 비해 최대 2.1%의 상대적 향상을 기록했으며, 새로운 전문가를 추가하여 제한된 파인튜닝(finetuning) 데이터로도 18.8%의 향상을 달성하였습니다. 이를 통해 각 사용자가 자신의 요구에 맞춘 MoE 믹스를 지속적으로 구성할 수 있는 오픈 소스 생태계를 가능하게 합니다.



### A New Method for Cross-Lingual-based Semantic Role Labeling (https://arxiv.org/abs/2408.15896)
- **What's New**: 이번 논문에서는 자연어 처리(Natural Language Processing)에서 중요한 작업인 의미역 레이블링(Semantic Role Labeling)을 위한 새로운 딥 러닝 알고리즘을 제안합니다. 특히, 다국어에서 주석 데이터의 부족 문제를 해결하기 위해 모델 전이(Model Transfer) 방법을 활용하였습니다.

- **Technical Details**: 제안된 알고리즘은 CoNLL2009의 영어 부분과 페르시아어의 의미 역할 코퍼스를 포함하는 데이터셋을 사용하였습니다. 훈련의 효율성을 최적화하기 위해 각 언어에서 교육 데이터의 10%만을 사용하였습니다.

- **Performance Highlights**: 제안된 모델은 기존 Niksirt et al. 모델에 비해 상당한 개선을 보였으며, 단일 언어 모드에서 F1-score에서 2.05% 향상을, 교차 언어 모드에서는 6.23% 향상을 기록했습니다. 이는 제안된 모델이 실제로 더욱 우수함을 나타내며, 특히 여러 언어에 대한 주석 데이터 부족 문제를 해결하는 데 기여할 것으로 기대됩니다.



### Bias in LLMs as Annotators: The Effect of Party Cues on Labelling Decision by Large Language Models (https://arxiv.org/abs/2408.15895)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)이 정치적 정보, 특히 정당 신호를 활용하여 정치적 진술을 판단하는 방법을 조사하였습니다. 기존의 연구를 복제하여 LLMs가 인간의 편향을 반영하며, 특히 극단적인 정당의 진술에만 편향되는 인간과 달리 LLMs는 중도 좌파 및 중도 우파 정당의 진술에 대해서도 상당한 편향을 보임을 확인했습니다.

- **Technical Details**: 이 연구는 Ennser-Jedenastik과 Meyer(2018)의 실험을 복제하여, OpenAI의 ChatGPT 및 Meta의 LLaMa라는 두 LLM 가족을 사용하여 정치적 진술의 감정(positive, negative, neutral)을 평가합니다. LLM은 각 진술에 대해 5개의 정당 신호를 부여받아 그 결과를 분석하였고, 내부 일관성, LLM와 인간 코더 간의 일치율, 처리 조건에 따른 차이를 살펴보았습니다.

- **Performance Highlights**: LLMs는 인간 코더들보다 낮은 일치율을 보였으며, 처리 조건에 따라 상당한 편향을 드러냈습니다. 이를 통해 LLMs를 정치적 텍스트 주석자로 사용할 때의 한계와 그 결과에 대한 논의가 필요함을 강조하고 있습니다.



### Scaling Up Summarization: Leveraging Large Language Models for Long Text Extractive Summarization (https://arxiv.org/abs/2408.15801)
- **What's New**: EYEGLAXS는 LLM(큰 언어 모델)을 활용한 기계 자동 요약을 위한 새로운 시스템입니다. 특히 LLAMA2-7B와 ChatGLM2-6B를 사용하여 긴 텍스트 문서에서 추출 요약을 수행하는 데 중점을 두고 있습니다.

- **Technical Details**: EYEGLAXS는 추출 요약을 위해 Flash Attention 및 PEFT(파라미터 효율적 미세 조정) 기술을 사용하여 LLM의 계산 및 자원 문제를 해결합니다. 이 프레임워크는 일반적으로 어려운 연속적 문서의 추출 요약 문제를 해결하기 위해 데이터 집합에서 새로운 성능 기준을 설정했습니다.

- **Performance Highlights**: EYEGLAXS는 PubMed와 ArXiv와 같은 널리 알려진 데이터셋에서 새로운 성능 기준을 제시하며, 다양한 시퀀스 길이를 처리하는 LLM의 적응성과 작은 데이터셋에서의 훈련 효율성을 탐구했습니다. 이 연구는 추출 텍스트 요약 분야에서 새로운 기준을 제시할 뿐만 아니라, 향후 연구에 대한 유망한 경로를 열었습니다.



### Language Adaptation on a Tight Academic Compute Budget: Tokenizer Swapping Works and Pure bfloat16 Is Enough (https://arxiv.org/abs/2408.15793)
Comments:
          WANT@ICML 2024

- **What's New**: 이 논문은 제한된 вычислительная 예산(tight academic budget)에서 LLM(대형 언어 모델)의 계속적인 사전 학습을 수행하여 특정 언어로 적응하는 방법을 조사합니다. 특히 독일어와 아랍어로 Mistral-7B 모델을 적응시키는 효율적인 기법들을 평가하였습니다.

- **Technical Details**: 이 연구에서는 훈련 정밀도(training precision)와 토크나이저 교체(tokenizer swapping)에 중점을 두었습니다. 순수 bfloat16 훈련이 혼합 정밀도 훈련(mixed-precision training)보다 메모리를 덜 사용하는 반면 훨씬 더 빠르다는 것을 보여주었습니다. 또한, 전문화된 독일어 토크나이저로의 교체가 효율적인 토크나이제이션을 제공하는 것으로 나타났습니다.

- **Performance Highlights**: 실험 결과에 따르면 독일어에 대한 Mistral-7B의 계속적인 사전 학습은 성능 저하를 초래하였으나, 아랍어로의 적응에서는 성능 향상이 있었음을 보여주었습니다. 이는 이미 잘 표현된 언어에 대한 적응이 항상 유익하지 않음을 암시합니다.



### Interactive Agents: Simulating Counselor-Client Psychological Counseling via Role-Playing LLM-to-LLM Interactions (https://arxiv.org/abs/2408.15787)
- **What's New**: 본 논문에서는 대형 언어 모델(LLM)을 활용하여 상담자-클라이언트 상호작용을 시뮬레이션하는 프레임워크를 제안합니다. LLM을 통한 가상 상담 시스템이 실제 심리 상담을 대체하는 가능성을 모색한 것입니다.

- **Technical Details**: 제안된 프레임워크는 Clien와 Counselor 역할을 각각 LLM으로 시뮬레이션하며, 복합적 치료 기법을 사용하여 전문가 수준의 반응을 생성합니다. 이 시스템은 GPT-4 모델을 제로샷 프롬프트(Zero-shot prompting) 방식으로 구현하였습니다.

- **Performance Highlights**: 실험 결과, LLM이 생성한 대화는 기존의 최첨단 모델들보다 뛰어난 성능을 보였으며, 심리 상담 대화의 질이 매우 높음을 확인했습니다. 연구진은 SimPsyDial이라고 하는 LLM 생성 데이터셋을 발표하며, 코드 및 모델을 GitHub를 통해 공개하였습니다.



### Form and meaning co-determine the realization of tone in Taiwan Mandarin spontaneous speech: the case of Tone 3 sandh (https://arxiv.org/abs/2408.15747)
- **What's New**: 본 연구는 대만 만다린에서 T2-T3 및 T3-T3 음조 패턴을 가진 두 글자 단어의 음높이 곡선을 조사하여 Tone 3 (T3) 음조의 한국어 중화(chenyang)와 완전한 중화 여부를 확인하고자 하였다.

- **Technical Details**: 이 연구에서는 Generative Additive Mixed Model (GAMM) 을 활용하여 기본 주파수(f0) 곡선을 분석하고, 성별, 발화 속도, 발화자, 이웃 음조, 단어 위치, 빅그램 확률, 그리고 새로운 예측 변수인 단어 및 단어 의미와 같은 다양한 영향을 고려하였다.

- **Performance Highlights**: T3-T3 단어가 T2-T3 단어와 구별이 불가능해지며, 이는 단어 의미의 영향을 고려했을 때 전체적인 중화가 이루어졌음을 나타낸다. 단어 빈도수의 영향은 확인되지 않았고, 반대로 단어 의미의 효과는 이웃 음조의 효과와 유사하게 강력하다.



### LM-PUB-QUIZ: A Comprehensive Framework for Zero-Shot Evaluation of Relational Knowledge in Language Models (https://arxiv.org/abs/2408.15729)
- **What's New**: 본 논문은 BEAR(이전 작업) 기반의 새로운 프레임워크인 LM-PUB-QUIZ를 소개합니다. 이 프레임워크는 다양한 사전 학습 목표를 가진 언어 모델(LM)을 비교하고, 기계학습의 지식 획득과 손실을 모니터링하며, Hugging Face TRANSFORMERS 라이브러리와 통합할 수 있는 기능을 제공합니다.

- **Technical Details**: LM-PUB-QUIZ는 Python으로 작성된 오픈 소스 프레임워크로, BEAR 프로빙 메커니즘을 중심으로 구성되어 있습니다. 사용자는 데이터셋의 관계를 기반으로 LM 지식을 평가할 수 있으며, 여기에는 각 관계의 특정 인스턴스 예측이 포함됩니다. 각 관계는 Knowledge Base(KB)와 연결된 주제와 답변 공간을 가지고 있습니다.

- **Performance Highlights**: 프레임워크는 다양한 평가 옵션을 제공하며, 특히 각 도메인 별, 카르다날리티(cardinality) 별로 BEAR 점수를 계산할 수 있습니다. 프레임워크는 사용자가 직접 LM의 지식을 검사하고 분석할 수 있는 강력한 도구가 됩니다.



### An Evaluation of Sindhi Word Embedding in Semantic Analogies and Downstream Tasks (https://arxiv.org/abs/2408.15720)
Comments:
          arXiv admin note: substantial text overlap with arXiv:1911.12579

- **What's New**: 본 논문에서는 6,100만 개 이상의 단어로 구성된 새로운 단어 임베딩(corpus)을 제안합니다. 이 데이터는 여러 웹 리소스에서 크롤링하여 수집하였습니다. 크롤링한 데이터에서 원치 않는 텍스트를 걸러내기 위한 전처리 파이프라인을 설계하였습니다.

- **Technical Details**: 제안한 방법에서 정리된 어휘는 최신의 continuous-bag-of-words(CBoW), skip-gram, GloVe 알고리즘을 통해 단어 임베딩을 훈련하는 데 사용됩니다. pretrained embeddings 평가를 위해서는 인기 있는 intrinsic 및 extrinsic 평가 접근 방식을 사용하였습니다.

- **Performance Highlights**: 평가 결과, continuous-bag-of-words 및 skip-gram이 GloVe와 기존 Sindhi fastText 단어 임베딩보다 intrinsic 및 extrinsic 평가 접근 방식 모두에서 더 우수한 성능을 보였습니다.



### Conan-embedding: General Text Embedding with More and Better Negative Samples (https://arxiv.org/abs/2408.15710)
- **What's New**: 이 논문에서는 embedding 모델의 성능을 극대화하기 위한 conan-embedding 모델을 제안합니다. 이 모델은 dynamic hard negative mining 방식을 도입하여 훈련 과정 동안 더 높은 품질의 negative examples를 지속적으로 활용할 수 있게 합니다.

- **Technical Details**: conan-embedding 모델은 contrastive learning에 기반하여, negative 예제의 질이 모델 성능에 미치는 영향을 극대화합니다. 우리는 Cross-GPU balancing Loss를 채택하여 배치 크기를 여러 작업 간에 조정하여 훈련 효율성을 높이고, LLM의 prompt-response 쌍을 훈련 데이터로 활용하여 embedding 모델의 성능을 더욱 향상시킵니다.

- **Performance Highlights**: 이 접근법은 중국 Massive text embedding benchmark (CMTEB) 리더보드에서 1위를 차지하여, embedding 모델의 성능과 넓은 응용 가능성을 입증하였습니다.



### TempoFormer: A Transformer for Temporally-aware Representations in Change Detection (https://arxiv.org/abs/2408.15689)
- **What's New**: 이 논문은 TempoFormer라는 새로운 동적 표현 학습 모델을 소개합니다. 이 모델은 transformer 기반이며, 태스크에 구애받지 않고 시간적 요소를 인식하여 문맥의 동적 변화를 효과적으로 학습합니다.

- **Technical Details**: TempoFormer는 내부 및 외부 문맥 동역학을 함께 훈련하며, 회전형 위치 임베딩을 시간 정보를 반영하도록 변형합니다. 이러한 아키텍처는 다양한 transformer 기반 모델에서 시간적 표현의 기초로 사용할 수 있습니다.

- **Performance Highlights**: TempoFormer는 3개의 실시간 변화 감지 태스크에서 최첨단(State of the Art, SOTA) 성능을 기록하였습니다. 이러한 태스크는 사용자 기분 변화 감지, 대화 흐름 중단 감지, 그리고 다양한 정보 변화 감지입니다.



### StyleRemix: Interpretable Authorship Obfuscation via Distillation and Perturbation of Style Elements (https://arxiv.org/abs/2408.15666)
- **What's New**: 본 논문은 저자 신원을 감추는 authorship obfuscation을 위한 새로운 방법론인 StyleRemix를 제안합니다. 이 방법은 기존의 대형 언어 모델(LLMs)을 활용하면서 저자 고유의 스타일 요소를 해석 가능하고 적응적으로 변형합니다.

- **Technical Details**: StyleRemix는 Low Rank Adaptation (LoRA) 모듈을 이용해 입력 텍스트의 특정 스타일 요소를 조정합니다. 스타일 축(axes)으로는 길이(length), 격식(formality) 등이 있으며, 이는 텍스트의 원본 내용과 유창성을 유지하는 동시에 저자의 특성화된 스타일을 반영할 수 있도록 설계되었습니다. 두 가지 단계로 진행되며, 첫 번째 단계는 스타일 축에 따라 다양한 데이터 세트를 생성하고, 두 번째 단계에서는 이 데이터 세트를 사용하여 텍스트 생성을 조정합니다.

- **Performance Highlights**: StyleRemix는 여러 도메인에서 기존의 최첨단 authorship obfuscation 방법과 비슷하거나 더 큰 LLM을 능가하는 성능을 보였습니다. 또한, 두 가지 데이터셋인 AuthorMix(30K 이상의 고품질 텍스트)와 DiSC(1,500개 텍스트의 병렬 코퍼스)를 공개하여 연구에 기여할 예정입니다.



### Harnessing the Intrinsic Knowledge of Pretrained Language Models for Challenging Text Classification Settings (https://arxiv.org/abs/2408.15650)
Comments:
          PhD thesis

- **What's New**: 이번 논문에서는 사전 학습된 언어 모델(PLMs)의 본질적인 지식을 활용하여 텍스트 분류의 세 가지 도전적 설정을 탐구합니다. 이 접근법은 오도된 그러나 부정확한 방해물을 선택하고, 보이지 않는 레이블에 대한 모델 일반화를 향상시키며, 대규모 언어 모델의 컨텍스트 학습 프롬프트에 대한 민감성을 해결하는 데 중점을 둡니다.

- **Technical Details**: 연구자들은 PLMs에서 파생된 맥락화된 단어 표현을 기반으로 방해물 생성을 위한 모델을 개발하였고, 도메인에 독립적인 작업 레이블 설명으로 작은 미세 조정 데이터 세트를 생성하였습니다. 이를 통해 모델의 강인성과 성능을 크게 향상시킬 수 있었습니다.

- **Performance Highlights**: 제안된 모델은 인간의 성능에 접근하거나 이를 초과하는 결과를 도출하였으며, 각 모델 구성요소의 중요성에 대한 정량적 분석을 통해 다양한 특징들이 퍼포먼스에 미치는 영향을 평가하였습니다.



### Beyond Levenshtein: Leveraging Multiple Algorithms for Robust Word Error Rate Computations And Granular Error Classifications (https://arxiv.org/abs/2408.15616)
Comments:
          Accepted in INTERSPEECH 2024

- **What's New**: 본 논문에서는 정밀한 표기법 오류 및 대문자 사용 오류를 평가하기 위한 비파괴적 토큰 기반 접근 방식을 제안합니다. 기존의 Word Error Rate (WER) 컴퓨테이션의 한계를 극복하며, 확장된 Levenshtein distance 알고리즘을 사용하여 강건한 WER을 계산합니다.

- **Technical Details**: 문서에서 설명된 방법론은 별도의 토큰 생성 및 다양한 정규화를 통해 트랜스미션 오류를 세분화하여 분석합니다. 특히, 대문자 및 구두점 오류를 효과적으로 처리하기 위해 존재하는 유사도 알고리즘 및 음성학 알고리즘을 활용합니다.

- **Performance Highlights**: 여러 데이터셋에서 우리가 제안한 접근법이 기존의 WER보다 실질적으로 동등한 성능을 보이는 것을 보여주었습니다. 웹 애플리케이션을 통해 인터랙티브한 사용 및 시각화가 가능합니다.



### SIaM: Self-Improving Code-Assisted Mathematical Reasoning of Large Language Models (https://arxiv.org/abs/2408.15565)
- **What's New**: 새로운 패러다임을 제안하여 대규모 웹에서 작성된 다양한 수학 질문-답변 쌍을 활용하여 코드 기반 비평 모델을 통해 코드 응답 평가와 모델 개선을 수행합니다.

- **Technical Details**: 제안된 방법은 초기 모델을 세밀하게 조정하고, 코드 샘플을 생성하여 비평 모델이 평가한 최고의 유효 코드 응답을 따르면 지속적인 개선을 촉진하는 반복적 자기 향상 구조를 채택합니다. 여러 가지 모델 패밀리를 사용하여 실험을 진행하였으며, DPO(Direct Preference Optimization)와 ORPO(Ordinal Response Pairwise Optimization)와 같은 다양한 정렬 알고리즘을 탐구합니다.

- **Performance Highlights**: 실험 결과, 제안된 패러다임을 사용한 LLM 모델이 영어 및 중국어의 도메인 내(+5.7%) 및 도메인 외(+4.4%) 벤치마크에서 큰 성과를 보여주며, 최첨단 70B 코드 기반 수학 LLM보다 11.9% 더 좋은 성능을 발휘했습니다.



### Boosting Lossless Speculative Decoding via Feature Sampling and Partial Alignment Distillation (https://arxiv.org/abs/2408.15562)
Comments:
          The work was not submitted to AAAI 2025

- **What's New**: FSPAD(Future Sampling and Partial Alignment Distillation for Lossless Speculative Decoding)는 기존의 접근 방식을 개선하여 손실 없는 예측을 위한 두 가지 간단하면서도 효과적인 요소를 도입합니다. 이 모델은 고차원 공간에서 LLM의 특징을 샘플링하고 부분 정렬 증류를 통해 훈련 단계에서의 특징과 로짓 간의 갈등을 줄임으로써 성능을 향상시킵니다.

- **Technical Details**: FSPAD는 목표 LLM의 토큰 임베딩을 활용하여 특징을 샘플링한 후, 이를 드래프트 모델에 입력합니다. 또한, 부분 정렬 증류를 도입하여 드래프트 모델의 특징과 로짓 간의 연결을 약화시킵니다. 이 방법은 EAGLE-2의 프레임워크 내에서 기존 LLM 모델과의 지식 증류를 최적화합니다.

- **Performance Highlights**: FSPAD는 여러 작업에서 기존 최첨단 방법을 초월하며, 대표적으로 다중 대화, 번역, 요약, 질문 응답, 수학적 추론, 검색 보강 생성 같은 분야에서 성능을 입증했습니다. FSPAD는 EAGLE-2와 비교하여 추가적인 0.28-0.48 토큰을 생성할 수 있는 능력을 보였습니다.



### WildFeedback: Aligning LLMs With In-situ User Interactions And Feedback (https://arxiv.org/abs/2408.15549)
Comments:
          24 pages

- **What's New**: WildFeedback는 사용자와 LLM 간의 상호작용을 통해 더 나은 선호 데이터 세트를 구축하는 혁신적인 프레임워크를 제안합니다. 전통적인 인간 주석 기반 방법들의 한계를 극복하기 위해 실제 사용자 피드백을 실시간으로 반영합니다.

- **Technical Details**: WildFeedback는 세 가지 주요 단계로 운영됩니다: feedback signal identification, preference data construction, 그리고 user-guided evaluation. 이 프레임워크는 실제 대화에서 수집된 피드백 신호를 기반으로 사용자의 진정한 선호를 반영하는 데이터 세트를 생성합니다.

- **Performance Highlights**: WildFeedback를 통해 세밀하게 조정된 LLM은 사용자 선호와의 정렬이 크게 향상되었습니다. 전통적인 벤치마크 및 새로운 사용자 주도 평가 방법 모두에서 효과가 입증되었습니다.



### An Investigation of Warning Erroneous Chat Translations in Cross-lingual Communication (https://arxiv.org/abs/2408.15543)
- **What's New**: 이 연구에서는 채팅 번역의 복잡성 문제를 해결하기 위해 Multidimensional Quality Metrics for Chat Translation (MQM-Chat)라는 새로운 평가 지표를 제시합니다. 다양한 모델의 실험을 통해 각 모델이 공통적인 오류를 발생시키고, 각 모델의 단점이 다르다는 점을 확인하였습니다.

- **Technical Details**: 연구는 MQM-Chat을 통해 다섯 개의 번역 모델 실험을 수행하였으며, 번역 모델의 불완전함을 인식하고 사용자에게 오류 경고 메시지를 제공함으로써 사용자 경험 개선을 목표로 합니다. 참여자들에게는 가상의 다국어 채팅 시나리오가 제공되며, 번역 오류 시 경고 메시지가 표시됩니다.

- **Performance Highlights**: 설문조사 결과, 경고 메시지는 (1) 다국어 채팅에서 유용하며, (2) 사용자가 채팅 행동을 변경하도록 유도할 수 있는 가능성을 보여줍니다. 이는 차후 채팅 번역에서의 사용자 지원 기능 개발에 중요한 기초 자료가 될 것입니다.



### LRP4RAG: Detecting Hallucinations in Retrieval-Augmented Generation via Layer-wise Relevance Propagation (https://arxiv.org/abs/2408.15533)
- **What's New**: 이번 논문에서는 RAG (Retrieval-Augmented Generation) 기술을 기반으로 한 hallucination(환각) 탐지 방법인 LRP4RAG를 제안합니다. LRP (Layer-wise Relevance Propagation) 알고리즘을 활용하여 RAG의 출력에 대한 입력의 관련성을 계산하는 새로운 접근 방식을 도입했습니다.

- **Technical Details**: LRP4RAG는 RAG 생성기의 입력과 출력 간의 관련성을 계산하기 위해 LRP를 사용하고, 이를 통해 관련성 행렬을 추가적으로 추출(resampling) 및 처리하는 방법을 포함합니다. 처리된 관련성 데이터는 여러 분류(classifier)를 통해 출력에 환각이 포함되어 있는지를 판단하는 데 사용됩니다.

- **Performance Highlights**: 광범위한 실험 결과, LRP4RAG가 기존의 기준 모델들보다 우수한 성능을 보임을 입증했습니다. 이는 RAG 관련 환각 탐지에 있어 LRP의 사용이 효과적임을 나타냅니다.



### Dolphin: Long Context as a New Modality for Energy-Efficient On-Device Language Models (https://arxiv.org/abs/2408.15518)
- **What's New**: 본 논문은 언어 모델에서 긴 맥락을 에너지 효율적으로 처리하기 위한 새로운 디코더-디코더 아키텍처인 Dolphin을 소개합니다. 이는 기존 모델들의 높은 에너지 소모와 지연 문제를 해결하기 위해 설계되었습니다.

- **Technical Details**: Dolphin은 0.5B 파라미터의 컴팩트한 디코더를 활용하여 긴 맥락 정보를 메모리 임베딩으로 별매하여 기본 7B 파라미터 디코더 모델의 입력 길이를 대폭 줄입니다. 비전-언어 모델에서 영감을 받아 이미지 임베딩 프로젝터를 재구성하여 긴 텍스트 맥락을 인코딩합니다.

- **Performance Highlights**: Dolphin 모델은 전통적인 전체 길이 맥락 처리 방법에 비해 에너지 효율성을 10배 향상시키고, 지연 시간을 5배 줄이며, 품질 손실 없이 반응 속도를 개선합니다.



### ReMamba: Equip Mamba with Effective Long-Sequence Modeling (https://arxiv.org/abs/2408.15496)
- **What's New**: 이 논문에서는 Mamba 아키텍처의 긴 컨텍스트 처리 능력을 개선하기 위해 새로운 모델인 ReMamba를 제안합니다. 기존 Mamba 모델은 짧은 컨텍스트 작업에서는 우수한 성능을 보이지만 긴 컨텍스트 작업에서는 Transformer 기반 모델에 비해 성능이 제한적이라는 점에 착안했습니다.

- **Technical Details**: ReMamba는 두 단계의 재전달 과정 내에서 선택적 압축 및 적응 기법을 통합하여 긴 컨텍스트 이해 능력을 향상시킵니다. 첫 번째 전방 패스에서 상위 k개의 숨겨진 상태를 선택하고, 두 번째 전방 패스에서 이를 상태 공간으로 통합하는 방식을 사용하여 정보의 간결화를 이루어냅니다.

- **Performance Highlights**: 실험 결과, ReMamba는 LongBench와 L-Eval 벤치마크에서 각각 3.2 및 1.6 포인트 향상을 달성하며, 같은 크기의 Transformer 모델과 거의 동등한 성능을 나타냈습니다. 또한, ReMamba는 Mamba2에 대한 전이 가능성도 보여줍니다.



### Enhancing and Accelerating Large Language Models via Instruction-Aware Contextual Compression (https://arxiv.org/abs/2408.15491)
Comments:
          20 pages

- **What's New**: 이 논문은 Instruction-Aware Contextual Compression(IACC)라는 새로운 방법을 소개하여, 대형 언어 모델(LLM)의 메모리 소모 및 추론 지연을 줄이면서도 성능을 유지하는 효율적인 방법을 제공합니다. 


- **Technical Details**: IACC는 정보를 삭제하는 이중 학습(pre-training) 단계를 통해 명령어에 따라 문맥을 압축하는 방법론입니다. 이를 통해 불필요한 정보는 제거되고, 더 중요한 정보는 효과적으로 유지됨으로써, LLM의 성능을 향상시킵니다.

- **Performance Highlights**: 실험 결과, context 관련 비용을 50% 줄였으며, 추론 메모리 사용량이 5% 감소하고 추론 속도는 2.2배 증가했습니다. Rouge-1 점수는 0.047 만큼 소폭 하락했지만 전반적인 성능은 크게 개선되었습니다.



### Legilimens: Practical and Unified Content Moderation for Large Language Model Services (https://arxiv.org/abs/2408.15488)
Comments:
          Accepted by ACM Conference on Computer and Communications Security (CCS) 2024

- **What's New**: 이 논문에서는 대형 언어 모델(LLM)을 위한 콘텐츠 모더레이션(Moderation) 프레임워크인 Legilimens를 제안합니다. 이는 기존의 라이벌 모델에 비해 효과성과 효율성을 모두 충족할 수 있는 최초의 시도입니다.

- **Technical Details**: Legilimens는 LLM의 디코딩 프로세스를 활용하여 효과적인 개념(feature)을 추출하는 방법론을 사용합니다. 이 프레임워크는 입력 텍스트 길이에 독립적이며, 다양한 유형의 모더레이션 작업에 적용할 수 있도록 설계되었습니다. 또한, 레드팀(Red-team) 모델 기반의 데이터 증강(data augmentation)을 통해 스테이트 오브 아트의 탈옥(jailbreaking) 공격에 강력하게 대응합니다.

- **Performance Highlights**: Legilimens는 5개의 호스트 LLM, 17개의 데이터셋 및 9개의 탈옥 방법을 통해 광범위한 실험을 진행하여 효과성과 효율성을 입증했습니다. 상업적 및 학술적 기준과 비교할 때, Legilimens는 뛰어난 성능을 보였으며, 극소수 샷(few-shot) 시나리오 및 다중 레이블 분류(multi-label classification) 작업에도 적용 가능하다는 것을 확인했습니다.



### Implicit Geometry of Next-token Prediction: From Language Sparsity Patterns to Model Representations (https://arxiv.org/abs/2408.15417)
Comments:
          Accepted at COLM 2024

- **What's New**: 이 논문에서는 대형 언어 모델의 훈련을 위해 널리 사용되는 다음 토큰 예측(Next-token prediction, NTP) 방식의 지리적 패턴과 기하학적 속성 간의 관계를 탐구합니다. NTP 훈련이 언어 통계와 모델 표현의 기하학적 관계를 형성하는 방식을 분석할 수 있는 새로운 프레임워크를 제안하며, 이로 인해 발생하는 '서브스페이스 붕괴(subspace-collapse)' 현상을 설명합니다.

- **Technical Details**: NTP는 희소 확률 라벨 벡터에 대한 소프트 라벨 분류로 프레임을 잡아, 로짓(logit) 영역에서의 랭크 제약(ranked-constrained), 핵 노름(nuclear-norm) 정규화를 연결합니다. NTP 과정에서 학습 로짓은 희소하고 낮은 랭크(low-rank) 구조를 선호하며, 이는 특정 문맥과 그에 뒤따르는 단어 쌍의 동시 발생 빈도를 포착합니다.

- **Performance Highlights**: 이 연구는 합성 데이터와 소규모 실제 언어 데이터셋을 통해 발견한 내용을 검증했으며, NTP의 언어 패턴 학습에 미치는 영향을 더욱 깊이 이해할 수 있는 연구 방향도 제시합니다.



### Awes, Laws, and Flaws From Today's LLM Research (https://arxiv.org/abs/2408.15409)
Comments:
          Under review

- **What's New**: 이 연구는 현대의 대형 언어 모델(LLM) 연구의 과학적 방법론에 대한 비판적 검토를 수행하였습니다. 2,000개 이상의 연구를 평가하고 나서, 종료 방침과 같은 여러 경향을 발견하였으며, 이는 LLM 연구 분야에서 더 많은 검토와 엄격함이 필요함을 강조합니다.

- **Technical Details**: 연구는 2,054개의 문헌을 평가하여, 결과와 관련된 실험 프로토콜의 중요성을 강조했습니다. 이러한 연구에서 데이터의 재현 가능성(reproducibility) 그리고 윤리적 고려사항의 필요성이 충분히 반영되지 않고 있음을 보여주었습니다.

- **Performance Highlights**: 관련 문헌 중 57%가 SOTA(State of the Art) 결과를 주장했으나, 25%만이 주장을 뒷받침하는 통계적 검증을 포함했습니다. LLM을 평가자로 사용하는 경향은 증가하고 있지만, 그 신뢰성에 대한 합의는 부족합니다.



### DualKanbaFormer: Kolmogorov-Arnold Networks and State Space Model DualKanbaFormer: Kolmogorov-Arnold Networks and State Space Model Transformer for Multimodal Aspect-based Sentiment Analysis (https://arxiv.org/abs/2408.15379)
Comments:
          10 pages, 2 figures, and 3 tables

- **What's New**: 이번 논문에서는 다중 모달 기반 감정 분석(Multimodal aspect-based sentiment analysis, MABSA)을 위한 새로운 아키텍처인 Kolmogorov-Arnold Networks (KANs)와 Selective State Space 모델(Mamba) 변환기(DualKanbaFormer)를 소개합니다. 이는 텍스트와 이미지 등 다양한 데이터 형식을 결합하여 감정 탐지를 향상시키고자 합니다.

- **Technical Details**: DualKanbaFormer는 Mamba를 활용하여 전역(context) 의존성을 포착하고, Multi-head Attention (MHA)을 통해 지역(context) 의존성을 포착하며, KANs를 사용하여 텍스트 표현과 시각적 표현 모두에 대한 비선형 모델링 패턴을 캡처합니다. 또한, 텍스트 KanbaFormer와 시각적 KanbaFormer를 게이티드 융합 레이어(gated fusion layer)를 활용하여 상호 모달 동역학(inter-modality dynamics)을 캡처합니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, 제안된 모델은 두 개의 공공 데이터셋에서 일부 최신 연구(state-of-the-art, SOTA) 보다 성능이 우수한 것으로 나타났습니다.



### Pitfalls and Outlooks in Using COME (https://arxiv.org/abs/2408.15366)
- **What's New**: 이 논문에서는 COMET 메트릭의 의도치 않은 문제들을 세 가지 측면에서 심층적으로 분석합니다: 기술적 문제, 데이터 문제, 사용 및 보고 문제. 또한, SacreCOMET 패키지를 출시하여 소프트웨어 및 모델 구성에 대한 서명을 생성하였습니다.

- **Technical Details**: COMET 메트릭은 사전 학습된 다국어 모델에서 미세 조정(fine-tuning)된 기계 학습 모델로, 번역 품질 평가를 위해 사용됩니다. 문제는 구식 소프트웨어 버전과 계산 정밀도, 비어 있는 콘텐츠 및 언어 불일치로 인해 발생합니다. COMET 메트릭은 빈 가설이나 언어 불일치에 취약하며, 이는 훈련 데이터의 편향에 따라 달라질 수 있습니다.

- **Performance Highlights**: 우리는 COMET 메트릭의 사용이 일관되지 않아 다른 논문이나 설정 간에 점수가 비교되지 않음을 보여주었습니다. 성능 실험에서는 다양한 소프트웨어 버전과 계산 정밀도를 테스트하며, 특히 모델의 낮은 정밀도 데이터 형식이 메트릭의 성능에 미치는 영향을 확인했습니다. 우리는 인퍼런스 처리 시간과 함께 메트릭의 효율성을 보고합니다.



### Multitask Fine-Tuning and Generative Adversarial Learning for Improved Auxiliary Classification (https://arxiv.org/abs/2408.15265)
- **What's New**: 이번 연구에서는 Multitask BERT라는 새로운 BERT 아키텍처를 구현하여 감성 분류(sentiment classification), 패러프레이즈 검출(paraphrase detection), 의미적 텍스트 유사성(prediction of semantic textual similarity) 예측의 세 가지 다운스트림 태스크에서 멀티태스크 파인튜닝(multitask fine-tuning)을 수행합니다.

- **Technical Details**: 이 모델은 레이어 공유(layer sharing) 및 트리플렛 아키텍처(triplet architecture), 맞춤형 문장 쌍 토크나이제이션(custom sentence pair tokenization), 손실 쌍(pairing) 및 기울기 수술(gradient surgery) 등의 최적화를 포함합니다. 이러한 최적화를 통해 테스트 데이터에서 감성 분류 정확도는 0.516, 패러프레이즈 검출 정확도는 0.886, 의미적 텍스트 유사성은 0.864에 도달하였습니다. 또한 생성적 적대 학습(generative adversarial learning)을 BERT에 적용하여 잠재 공간(latent space)에서 가짜 임베딩(fake embeddings)을 생성하는 조건부 생성기 모델을 구성하였습니다.

- **Performance Highlights**: AC-GAN-BERT라는 이 프레임워크를 사용하여 레이블이 없는 학습 데이터 양 증가가 AC-GAN-BERT의 테스트 정확도에 미치는 영향을 조사하기 위한 반감독 민감도 분석(semi-supervised sensitivity analyses)을 수행하였습니다. 연구 결과, 조건부 생성기는 클래스 레이블(class labels)과의 명확한 공간적 상관관계를 갖는 풍부한 임베딩을 성공적으로 생성하였으며, 모드 붕괴(mode collapse)를 회피함을 입증하였습니다.



### Text classification optimization algorithm based on graph neural network (https://arxiv.org/abs/2408.15257)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2405.17460 by other authors

- **What's New**: 본 논문에서는 그래프 신경망(Graph Neural Networks, GNN)을 활용한 텍스트 분류 최적화 알고리즘을 소개합니다. 이 알고리즘은 적응형 그래프 구성 전략과 효율적인 그래프 컨볼루션(Convolution) 연산을 도입하여 텍스트 분류의 정확도와 효율성을 크게 향상시킵니다.

- **Technical Details**: 전통적인 텍스트 분류 방법은 bag of words 모델이나 TF-IDF 같은 특성 표현에 의존하며, 단어 간의 의미적 연결을 간과합니다. 본 연구에서는 GNN을 통해 비 유클리드 데이터(non-Euclidean data)를 효율적으로 처리할 수 있는 가능성을 탐구하며, 그래프 구조 생성의 복잡성과 모델 학습의 높은 비용 문제를 해결합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 전통적인 접근법 및 기존 GNN 모델보다 여러 공공 데이터셋에서 뛰어난 성능을 보여주며, 텍스트 분류 작업에 대한 우수성과 실현 가능성을 강조합니다.



### More Text, Less Point: Towards 3D Data-Efficient Point-Language Understanding (https://arxiv.org/abs/2408.15966)
- **What's New**: 이번 논문에서는 3D 데이터 부족 문제를 해결하기 위해 3D Data-Efficient Point-Language Understanding(3DEPL)이라는 새로운 과제를 제안합니다. GreenPLM 모델을 통해 최소한의 3D 포인트 클라우드 데이터와 텍스트 데이터 쌍으로 강력한 3D 객체 이해를 가능하게 하고자 합니다.

- **Technical Details**: GreenPLM은 CLIP에서 영감을 받아 사전 학습된 포인트 클라우드-텍스트 인코더를 사용하여 3D 포인트 클라우드 공간을 텍스트 공간에 매핑합니다. 6M 개의 자유 텍스트 설명을 생성하고 세 단계의 훈련 전략을 설계하여 LLM이 다양한 모달리티 간의 본질적인 연결을 탐색하도록 돕습니다.

- **Performance Highlights**: GreenPLM은 기존의 최첨단 모델이 사용하는 3D 훈련 데이터의 12%만으로도 우수한 3D 이해력을 달성합니다. 또한, GreenPLM은 텍스트 전용 데이터만으로 경쟁력 있는 성능을 발휘할 수 있습니다.



### Leveraging Open Knowledge for Advancing Task Expertise in Large Language Models (https://arxiv.org/abs/2408.15915)
Comments:
          28 pages, 12 tables, 10 figures

- **What's New**: 이 논문은 특정 도메인 작업에서 대규모 언어 모델(LLMs)의 전문성을 개선하기 위해 몇 가지 인적 주석이 있는 샘플(K-shot)을 사용하는 새로운 접근 방식을 제안합니다. Open knowledge(공개 지식)를 활용하여 모델과 데이터 선택에서의 기존 방법들이 일반적인 능력 향상에 집중한 것에 비해, K-shot 데이터를 사용하여 전문성을 높이고자 합니다.

- **Technical Details**: 우리는 K-shot 학습 설정을 통해 LLM의 작업 전문 지식을 발전시키는 효율적이고 확장 가능한 파이프라인을 개발하였습니다. Mixture-of-Experts (MoE) 시스템을 통해 여러 전문가 간의 상호 보완적인 지식을 최대한 활용하며, K-shot 데이터를 통해 문제 해결 능력을 가진 모델을 선택합니다. 우리가 제안한 방법은 reasoning perplexity(추론 혼란도), exact match accuracy(정확한 매치 정확도), group diversity(그룹 다양성)를 바탕으로 합니다.

- **Performance Highlights**: 광범위한 실험 결과, 우리의 방법이 다양한 작업에서 공개 지식을 활용하는 기존 방법들보다 우수함을 입증하였습니다. K-shot 데이터를 통해 적합한 전문가 모델과 유용한 지시 사항을 선별하는 데 성공했으며, 저비용으로도 실세계에 적용 가능한 작업 전문성을 확보할 수 있었습니다.



### Persuasion Games using Large Language Models (https://arxiv.org/abs/2408.15879)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)이 인간과 유사한 텍스트를 이해하고 생성하는 데 탁월한 도구가 되고 있으며, 이러한 도구가 사용자에게 적합한 투자 계획, 신용 카드 및 보험 정책 선택에 도움을 준다는 점을 제시하고 있다.

- **Technical Details**: 주요 요점은 여러 에이전트가 협력하는 복잡한 다중 에이전트 프레임워크를 통해 사용자와의 설득적인 대화를 수행하며, 이 과정에서 정보 검색, 응답 분석, 설득 전략 개발 및 사실 확인을 수행하는 보조 에이전트들이 존재한다는 것이다.

- **Performance Highlights**: 실험을 통해 이 협력 방법론이 LLM의 설득력을 크게 향상시키는 것을 보여주었고, 사용자 저항을 지속적으로 분석해 감정 변화에 대한 판단 및 적절한 사실을 조합해 반박할 수 있는 전략을 개발하였다.



### Knowledge Navigator: LLM-guided Browsing Framework for Exploratory Search in Scientific Literatur (https://arxiv.org/abs/2408.15836)
- **What's New**: 본 연구에서는 과학 문헌의 기하급수적 증가에 대한 대응으로, Knowledge Navigator라는 새로운 시스템을 발표합니다. 이 시스템은 사용자가 넓은 주제(query)에서 검색한 문서를 두 단계 계층으로 조직하고 구조화하여 탐색적인 검색 능력을 향상시킵니다.

- **Technical Details**: Knowledge Navigator는 사용자가 특정 하위 주제(subtopic)에 집중하고 관련 문서를 추가로 검색할 수 있도록 하여 반복적인 검색(iterative search) 및 깊이 있는 지식 발견(deep knowledge discovery)을 가능하게 합니다. 또한, LLM(대형 언어 모델) 기능과 클러스터 기반(cluster-based) 방법을 결합하여 효율적인 탐색(browsing) 방법을 제공합니다.

- **Performance Highlights**: 우리의 접근 방식은 자동 및 수동 평가를 통해 CLUSTREC-COVID 및 SCITOC의 두 가지 새로운 벤치마크(benchmarks)에서 효과성을 입증했습니다. 또한, 우리의 코드, 프롬프트(prompts), 벤치마크는 공개적으로 제공됩니다.



### Automatic Differential Diagnosis using Transformer-Based Multi-Label Sequence Classification (https://arxiv.org/abs/2408.15827)
Comments:
          25 pages, 7 figures

- **What's New**: 본 연구에서는 환자의 나이, 성별, 병력 및 증상을 기반으로 차별 진단(differential diagnosis)을 제공하기 위해 transformer 기반의 접근 방식을 제안합니다. 기존 연구에서는 차별 진단을 제공하는 시스템 개발이 충분히 이루어지지 않았습니다.

- **Technical Details**: DDXPlus 데이터셋을 활용하여 환자 정보를 포함한 데이터 처리 기법을 적용하고, 훈련 데이터를 다양하게 만들기 위해 두 개의 데이터 수정 모듈을 도입했습니다. 이 작업은 다중 레이블 분류(multi-label classification) 문제로 접근하며, 네 가지 transformer 모델을 사용하여 실험을 진행했습니다.

- **Performance Highlights**: 모든 모델이 held-out 테스트 세트에서 97% 이상의 F1 점수를 달성하여 유망한 결과를 나타냈습니다. 추가적으로 설계한 행동 테스트를 통해 모델의 일반화 능력을 평가했으며, 의사와의 협력을 통해 준비한 맞춤형 테스트 세트에서도 모델의 성능 향상을 확인했습니다.



### LogicGame: Benchmarking Rule-Based Reasoning Abilities of Large Language Models (https://arxiv.org/abs/2408.15778)
- **What's New**: 이 논문에서는 LogicGame이라는 새로운 벤치마크를 소개하여 대형 언어 모델(LLMs)의 규칙 이해, 실행, 계획 능력을 종합적으로 평가하고자 합니다.

- **Technical Details**: LogicGame은 다양한 규칙 기반 게임으로 구성되어 있으며, 모델이 유지해야 할 규칙의 시리즈를 통해 문제를 해결해야 합니다. 각 게임 시나리오는 초기 상태와 함께 주어진 규칙을 이해하고 적용하여 최종 목표를 달성하는 것을 목표로 합니다. 평가 과정은 최종 결과뿐 아니라 모델이 거친 중간 단계도 고려하여 전체적인 성능을 평가합니다.

- **Performance Highlights**: LogicGame을 통해 다양한 LLM을 테스트한 결과, 모델들이 규칙 기반 논리 추론 능력에서 주목할 만한 한계를 보여주었습니다. 최고 성능의 LLM조차도 복잡한 추론 과제에서 약 20%의 전체 정확도 및 3단계 과제에서 10% 미만의 성과를 기록했습니다.



### A Survey on Evaluation of Multimodal Large Language Models (https://arxiv.org/abs/2408.15769)
- **What's New**: 이 논문은 Multimodal Large Language Models (MLLMs)의 평가 방법을 체계적이고 포괄적으로 검토하고 있습니다. MLLMs는 인간의 지각 및 추론 시스템을 모방하여 다양한 모달리티 인코더와 강력한 Large Language Models (LLMs)을 통합합니다. 이 논문은 기존 MLLM 평가 작업을 능력에 따라 검토하고 분류하여 인식, 인지, 추론 및 신뢰성 등 다양한 분야의 요구를 충족하는 방법을 제안합니다.

- **Technical Details**: MLLM은 LLMs를 '뇌'로 보고 다양한 모달리티 인코더를 감각 기관으로 배치하여 여러 모달리티 정보에 대한 이해와 추론 능력을 제공합니다. MLLM의 평가 단계와 메트릭스를 검토하고 평가 방법론을 다양한 시각에서 제시합니다.

- **Performance Highlights**: MLLMs의 평가 방법에 대한 체계적인 리뷰를 통해 MLLM의 강점과 한계를 정량화할 수 있는 기초를 제공합니다. 또한, MLLMs를 신뢰성 관점에서 탑재할 필요성을 강조하며, 다양한 하위 작업에서의 MLLMs 성능을 평가하는 방법을 제안합니다.



### Harmonized Speculative Sampling (https://arxiv.org/abs/2408.15766)
- **What's New**: 이번 연구에서는 대형 언어 모델의 디코딩 성능을 가속화하기 위한 새로운 접근법인 HArmonized Speculative Sampling (HASS)을 제안합니다. HASS는 교육(training)과 디코딩(decoding)의 목표와 컨텍스트를 조화롭게 하여 수용률(acceptance rate)을 개선합니다.

- **Technical Details**: HASS는 이전 모델들에서는 독립적으로 고려되었던 훈련과 디코딩 간의 연결(linkage)을 조사하였고, 두 가지 전략인 ranking distillation 및 context-aligned training을 통해 훈련과 디코딩을 조화롭게 맞추는 방법을 사용합니다. 이를 통해 HASS는 추가적인 추론(inference) 오버헤드 없이 수용률을 개선합니다.

- **Performance Highlights**: HASS는 EAGLE-2 모델을 기반으로 하여 세 가지 데이터셋에서 각각 2.81배에서 3.65배의 시간 가속화 성능을 달성했으며, 이는 EAGLE-2보다 8%에서 12% 향상된 수용 길이를 기록하였습니다.



### Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts (https://arxiv.org/abs/2408.15664)
- **What's New**: 이번 논문에서는 Mixture-of-Experts (MoE) 모델의 로드 밸런스를 관리하는 새로운 접근법인 Loss-Free Balancing을 제안합니다. 기존의 보조 손실(auxiliary loss) 접근 방식을 대체하여 훈련 중간에 간섭 그래디언트(interference gradients)를 발생시키지 않도록 설계되었습니다.

- **Technical Details**: Loss-Free Balancing은 각 전문가(expert)의 최근 로드(load)에 따라 라우팅 점수(routing scores)에 전문가별 편향(expert-wise bias)을 적용합니다. 이는 전문가의 로드를 동적으로 업데이트하여 균형 잡힌 로드 분포를 유지합니다. 기존 접근 방식의 보조 손실 대신에, 각 전문가의 편향을 조정하여 상위 K개의 루팅 결정을 만듭니다.

- **Performance Highlights**: Loss-Free Balancing은 MoE 모델을 훈련시킬 때, 전통적인 보조 손실 사용 모델들보다 더 나은 성능을 보이며 로드 밸런스 또한 크게 개선됩니다. 최종 실험 결과에서는 최대 3B의 파라미터를 가진 MoE 모델이 200B의 토큰으로 훈련되었고, 성능이 더 좋고 로드 분산 또한 더 효율적으로 이루어졌습니다.



### CBF-LLM: Safe Control for LLM Alignmen (https://arxiv.org/abs/2408.15625)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 정렬을 보장하기 위한 제어 기반의 프레임워크를 제안하며, 사용자에게 바람직한 텍스트 생성을 보장하기 위해 제어 장벽 함수(Control Barrier Function, CBF)를 활용합니다. CBF에 기반한 안전 필터를 적용하여 모델의 출력 생성을 조정하는 방법을 보여줍니다.

- **Technical Details**: 이 프레임워크는 LLM의 출력 형성 과정에서 CBF를 적용하여 사용자가 원하는 바람직한 내용으로 조정하며, Llama 3와 RoBERTa 모델을 활용하여 구현되었습니다. 이 방식은 모델 파라미터를 수정하지 않고 외부 필터를 추가하는 형태로, '학습 없는(learning-free)' 정렬 프레임워크로 작동합니다.

- **Performance Highlights**: 실험 결과, CBF 제어를 통해 사용자가 지정한 정렬 작업에서 필요한 개입의 수가 줄어들어 시스템의 제어 능력과 효율성이 높아졌음을 나타냅니다.



### SciLitLLM: How to Adapt LLMs for Scientific Literature Understanding (https://arxiv.org/abs/2408.15545)
- **What's New**: 이 논문에서는 과학 문헌 이해를 위해 특별히 설계된 Large Language Models (LLMs)을 제안하며, 지속적 사전 훈련(Continual Pre-training, CPT)과 감독된 세부 조정(Supervised Fine-tuning, SFT)을 통합한 하이브리드 전략을 적용합니다.

- **Technical Details**: 이 연구 전략은 과학 도메인 지식을 강화하고 특별한 작업에 대한 지시 준수 능력을 높이기 위해 두 가지 주된 도전 과제를 해결합니다. 첫째, 고품질 CPT 말뭉치 구축과 둘째, 다양한 SFT 지시 사항 생성이 포함됩니다. PDF에서 텍스트를 추출하고 내용 오류를 수정하는 파이프라인을 통해 이 과정을 수행합니다.

- **Performance Highlights**: SciLitLLM 모델은 과학 문헌 이해 벤치마크에서 뛰어난 성능을 보이며, 평균 3.6%의 개선과 10.1%의 향상이 나타났습니다. SciLitLLM-7B는 70B 파라미터의 Llama3 및 Qwen2 모델보다도 뛰어난 성능을 발휘했습니다.



### Towards Fully Autonomous Research Powered by LLMs: Case Study on Simulations (https://arxiv.org/abs/2408.15512)
Comments:
          For additional code and data, please visit our GitHub repository: this https URL

- **What's New**: 대형 언어 모델(LLM)의 발전으로 자동화된 과학 연구의 새로운 기회를 제시하고, 실험 설계에서 데이터 분석, 보고서 작성까지의 전 과정을 자동화할 수 있는 자율 시뮬레이션 에이전트(ASA)를 개발하였습니다.

- **Technical Details**: ASA는 LLM을 통해 API 통합을 실행하며, 연구 계획(RP)을 입력하면 AI가 전체 연구 과정을 수행합니다. 이러한 방식으로 AI가 완전 자율적으로 시뮬레이션 코드를 작성하고, 데이터를 처리하며, 과학 보고서를 작성합니다. 이 연구에서는 폴리머 체인 구성 시뮬레이션 문제를 사례로 사용하여 LLM의 성능을 평가했습니다.

- **Performance Highlights**: ASA-GPT-4o는 연구 임무를 거의 완벽하게 수행하였으며, 인간의 개입 없이 최대 20 사이클까지 반복 가능한 자동화 가능성을 보여주었습니다. 이 결과는 LLM이 복잡한 과학 연구를 자율적으로 관리할 수 있는 잠재력을 강조합니다.



### Measuring the Reliability of Causal Probing Methods: Tradeoffs, Limitations, and the Plight of Nullifying Interventions (https://arxiv.org/abs/2408.15510)
- **What's New**: 이 연구에서는 기존의 인과적 탐지 기법의 신뢰성을 평가하기 위한 일반적인 경험적 분석 프레임워크를 제안합니다. 특히, 두 가지 주요 속성을 정의하고 정량화하여 인과적 탐지의 효과성을 평가합니다: 완전성(Completeness)과 선택성(Selectivity).

- **Technical Details**: 논문에서는 여러 주요 인과적 탐지 방법론을 대상으로 실험을 수행하고, 이 방법들이 상호 간의 트레이드오프(tradeoff)를 보이며, 한 가지 방법이 두 가지 기준을 동시에 만족할 수 없음을 발견했습니다. 연구 결과는 또한 반증 방법(nullifying interventions)이 대안적 방법(counterfactual interventions)보다 항상 덜 완전하다는 것을 보여주었습니다.

- **Performance Highlights**: 연구에 따르면, 인과적 탐지 방법 간에는 명확한 트레이드오프가 존재하며, 일반적으로 반증 방법은 안정성이 떨어지고 완전성이 낮아 인과적 탐지에서 효과적인 접근이 아닐 수 있음을示했습니다.



### Intertwined Biases Across Social Media Spheres: Unpacking Correlations in Media Bias Dimensions (https://arxiv.org/abs/2408.15406)
Comments:
          Accepted to ASONAM 2024

- **What's New**: 본 연구는 YouTube와 Reddit에서 수집된 새로운 데이터셋을 소개하며, 다양한 미디어 편향 차원에 대한 자동 주석을 제공합니다. 이 데이터셋은 지난 5년간의 콘텐츠를 포함하며, 정치, 스포츠, 건강, 교육, 오락 등 다섯 개의 주제 도메인에 걸쳐 있습니다.

- **Technical Details**: 본 연구에서는 기계 학습 (machine learning)과 자연어 처리 (natural language processing) 기술을 활용하여 미디어 편향을 식별하고, 여러 편향 차원 간 상관관계를 분석합니다. 데이터셋은 YouTube 댓글을 포함하여 다차원 미디어 편향 인식 저널 데이터세트를 만들기 위해 주석이 달렸습니다.

- **Performance Highlights**: 정치 도메인에서는 편향된 내용의 비율이 다른 도메인에 비해 유의미하게 높은 것으로 나타났습니다. 이 분석을 통해 데이터의 시간적 변화에 따른 편향 차원 간의 상관관계가 동적으로 변동함을 발견했습니다. 이러한 통찰력은 미래의 미디어 편향 탐지 시스템의 주춧돌을 마련합니다.



### A Statistical Framework for Data-dependent Retrieval-Augmented Models (https://arxiv.org/abs/2408.15399)
- **What's New**: 이 논문에서는 retrieval-augmented models (RAMs)의 훈련 및 기본 속성과 성능에 대한 이해를 제공하기 위한 통계적 프레임워크를 제안합니다. RAM은 {	extit{retriever}}와 {	extit{predictor}}라는 두 가지 구성 요소로 이루어져 있어 훈련과 예측 과정에서 중요한 역할을 합니다.

- **Technical Details**: 저자들은 RAM의 end-to-end 훈련을 위한 공식적인 목표 함수를 제시하며, 이는 일반화 및 표현력을 분석하는 데 중요한 역할을 합니다. retriever와 predictor의 상호작용을 포착하고, 데이터 저장소(data-store)의 크기와 RAM의 최종 성능 간의 관계를 명확히 합니다. 또한, excess risk bound를 도출하여 RAM의 성능을 향상시키는 역할을 설명합니다.

- **Performance Highlights**: 제안한 방법은 NaturalQuestions(NQ) 및 TriviaQA와 같은 표준 QA 벤치마크에서 효과적임을 입증하였으며, RAM이 효율적인 성능과 계산 요구 사이의 균형을 잘 맞춘다는 것을 보여줍니다.



### UNA: Unifying Alignments of RLHF/PPO, DPO and KTO by a Generalized Implicit Reward Function (https://arxiv.org/abs/2408.15339)
- **What's New**: 본 논문에서는 RLHF, DPO 및 KTO 기술을 통합하는 새로운 정렬 방법론인 UNified Alignment (UNA)를 제안합니다.

- **Technical Details**: UNA는 RLHF의 고전적 목표에 대한 수학적 증명을 통하여 일반화된 암묵적 보상 함수(implicit reward function)를 통해 최적의 정책이 유도됨을 보여줍니다. UNA는 암묵적 보상과 명시적 보상의 차이를 최소화하는 감독 학습(supervised learning)으로 3가지 방법을 통합하여, 다양한 피드백 유형(예: pairwise, binary, scalar feedback)에 대응할 수 있습니다.

- **Performance Highlights**: 하류 작업에서 UNA는 DPO, KTO 및 RLHF를 초월하는 성과를 보였으며, RL fine-tuning 과정의 안정성, 속도 및 메모리 부담을 줄이면서 RLHF/PPO를 능가합니다.



### Bi-Factorial Preference Optimization: Balancing Safety-Helpfulness in Language Models (https://arxiv.org/abs/2408.15313)
- **What's New**: 이번 연구에서는 Bi-Factorial Preference Optimization (BFPO)이라는 새로운 supervised learning 프레임워크를 제안합니다. 이는 안전성과 유용성을 동시에 고려한 LLM의 fine-tuning을 위한 단일 학습 목표로 변환하여, RLHF에서 발생할 수 있는 안전성과 유용성의 갈등 문제를 해결합니다.

- **Technical Details**: BFPO에서는 labeling function을 통해 안전성과 유용성의 전반적인 선호 순위를 포착하여, 두 가지 목표를 균형 있게 조절합니다. 이 방법은 기존의 RLHF기반 접근법에 비해 인간의 프롬프트나 주석이 필요 없고, 10% 미만의 계산 자원으로도 동일한 안전 수준을 달성할 수 있습니다.

- **Performance Highlights**: BFPO는 안전성과 유용성 모두에서 기존 방법보다 유의미하게 뛰어난 결과를 보였습니다. 특히, 공개 데이터셋을 사용하여 유용성을 유지하면서도 15%의 무해성을 향상시키고, 1.5K의 red teaming 데이터로 13% 향상시키는 성과를 달성했습니다.



### YOLO-Stutter: End-to-end Region-Wise Speech Dysfluency Detection (https://arxiv.org/abs/2408.15297)
Comments:
          Interspeech 2024

- **What's New**: 본 논문에서는 비유창한 언어( dysfluent speech ) 검출을 위한 최초의 엔드 투 엔드 방법인 YOLO-Stutter를 제안합니다. 이 방법은 시간을 정확하게 조정하여 비유창한 언어를 탐지합니다.

- **Technical Details**: YOLO-Stutter는 비정상적인 발화-텍스트 정렬을 입력으로 받아, 공간적 특징 집합체(spatial feature aggregator) 및 시간적 의존성 추출기(temporal dependency extractor)를 통해 지역별 경계(boundary)와 클래스(class) 예측을 수행합니다. 또한, VCTK-Stutter 및 VCTK-TTS라는 두 개의 비유창성 데이터셋을 도입하여, 자연스러운 비유창성을 시뮬레이션합니다.

- **Performance Highlights**: YOLO-Stutter는 최소한의 학습 가능 매개변수로도 시뮬레이션된 데이터와 실제 실어증(aphasia) 언어에서 최첨단 성능을 달성했습니다. 이 성과는 38명의 실어증 환자와의 협업을 통해 이루어진 데이터 분석 결과를 통해 입증되었습니다.



### Learning Granularity Representation for Temporal Knowledge Graph Completion (https://arxiv.org/abs/2408.15293)
Comments:
          15 pages. Accepted at ICONIP 2024

- **What's New**: 이번 연구에서는 Temporal Knowledge Graphs (TKGs)의 완전성을 향상시키기 위한 새로운 방법인 Learning Granularity Representation (LGRe)를 제안합니다. LGRe는 시간의 세부 granularity(세분성)를 고려하여 TKG의 결측 정보를 예측합니다.

- **Technical Details**: LGRe는 Granularity Representation Learning (GRL)과 Adaptive Granularity Balancing (AGB)라는 두 가지 주요 구성 요소로 이루어져 있습니다. GRL은 시간 특화된 다층 convolutional neural networks (CNNs)를 사용하여 다양한 granularity 간의 상호작용을 캡처하고, AGB는 시간 의미에 따라 이러한 표현의 가중치를 조정하여 예측의 표현력을 높입니다.

- **Performance Highlights**: 4개의 이벤트 벤치마크에서의 실험 결과, LGRe는 최신 TKG 완전성 방법론에 비해 우수한 성능을 보였으며, 시간 관련 표현 학습에 있어서 그 효과를 입증했습니다.



### AutoGen Studio: A No-Code Developer Tool for Building and Debugging Multi-Agent Systems (https://arxiv.org/abs/2408.15247)
Comments:
          8 pages

- **What's New**: 다중 에이전트 시스템(Multi-agent systems)이 여러 도메인에서 복잡한 문제를 해결하기 위해 협력하는 효과적인 패턴으로 부상하고 있습니다. 이와 같은 시스템을 간편하게 프로토타입, 디버그 및 평가할 수 있도록 해주는 코드 없는 개발 도구인 AUTOGEN STUDIO가 소개되었습니다.

- **Technical Details**: AUTOGEN STUDIO는 LLM 활성화 에이전트를 선언적(JSON 기반) 사양으로 표현할 수 있는 웹 인터페이스와 Python API를 제공합니다. 이 도구는 에이전트 워크플로우 사양을 위한 직관적 드래그 앤 드롭 UI, 워크플로우의 대화형 평가 및 디버깅 도구, 재사용 가능한 에이전트 구성 요소 갤러리를 제공합니다.

- **Performance Highlights**: AUTOGEN STUDIO는 5개월 동안 20만 다운로드를 기록하며 오픈 소스 도구로서의 경험을 바탕으로 다중 에이전트 개발 도구의 새로운 디자인 패턴과 미래 연구 방향을 제시합니다.



### SimpleSpeech 2: Towards Simple and Efficient Text-to-Speech with Flow-based Scalar Latent Transformer Diffusion Models (https://arxiv.org/abs/2408.13893)
Comments:
          Submit to TASLP

- **What's New**: 이번 연구에서는 SimpleSpeech 2라는 새로운 비자기회귀(Non-Auto-Regressive, NAR) 기반 텍스트-음성합성(Text-to-Speech, TTS) 시스템을 제안합니다. 이 시스템은 기존의 AR 및 NAR 모델의 장점을 결합하여 단순화된 데이터 준비, 직관적인 모델 및 손실 설계, 높은 품질의 안정적인 생성 성능 및 빠른 추론 속도를 제공합니다.

- **Technical Details**: SimpleSpeech 2는 음성을 분해하고 의미 있는 토큰으로 변환하기 위한 음성 토크나이저(speech tokenizer)에 대한 폭넓은 분석과 사전 훈련된 음성 인식 모델(Automatic Speech Recognition, ASR)로부터 생성된 노이즈가 있는 라벨(label)의 효과를 연구합니다. 또한, 문장 길이를 예측하는 네 가지 다른 유형의 예측기를 사용하여 이를 통해 생기는 성능 향상도 논의합니다.

- **Performance Highlights**: 테스트 결과, SimpleSpeech 2는 다른 최첨단 모델들에 비해 생성 성능과 생성 속도가 현저히 향상된 것으로 나타 나며, 다국어 TTS로 확장이 가능하다는 점도 주목할 만합니다. 실제로, 다국어 음성 데이터셋으로 훈련하여 다양한 언어의 음성을 자연스럽게 합성할 수 있음을 보여주었습니다.



New uploads on arXiv(cs.IR)

### Modeling and Analyzing the Influence of Non-Item Pages on Sequential Next-Item Prediction (https://arxiv.org/abs/2408.15953)
Comments:
          36 pages, 19 figures; Work in Progress

- **What's New**: 이번 연구는 비아이템 페이지(non-item pages)를 활용하여 사용자 추천 시스템의 성능을 개선하는 접근법을 제안합니다. 기존의 방법들은 아이템 간의 상호작용에 집중했지만, 비아이템 페이지도 사용자의 관심사에 중요한 정보를 제공할 수 있음을 보여줍니다.

- **Technical Details**: HypTrails라는 가설 검정 프레임워크를 사용하여 비아이템 페이지와 사용자의 관심 아이템 간의 관계를 분석했습니다. 연구에서는 비아이템 페이지를 내용 기반으로 표현하여 다음 아이템 예측(sequential next-item prediction) 작업에 활용하는 여러 방법을 제안합니다. 또한, CNN, RNN, 트랜스포머(transformer) 기반의 8가지 인기 시퀀스 추천 모델을 조정하여 비아이템 페이지를 통합하고, 이러한 정보로 다음 아이템 예측 능력을 조사했습니다.

- **Performance Highlights**: 모델은 비아이템 페이지와의 상호작용에서 학습할 수 있음을 보여주었으며, 두 개의 실제 데이터 셋에서 비아이템 페이지를 포함했을 때 성능이 향상되었습니다. 비아이템 페이지는 중요한 정보 공급원으로 작용하지만, 이를 효과적으로 표현하는 것이 핵심입니다. 모든 모델 아키텍처에서 비아이템 페이지를 포함하면 다음 아이템 예측 성능이 향상되는 경향을 보입니다.



### Knowledge Navigator: LLM-guided Browsing Framework for Exploratory Search in Scientific Literatur (https://arxiv.org/abs/2408.15836)
- **What's New**: 본 연구에서는 과학 문헌의 기하급수적 증가에 대한 대응으로, Knowledge Navigator라는 새로운 시스템을 발표합니다. 이 시스템은 사용자가 넓은 주제(query)에서 검색한 문서를 두 단계 계층으로 조직하고 구조화하여 탐색적인 검색 능력을 향상시킵니다.

- **Technical Details**: Knowledge Navigator는 사용자가 특정 하위 주제(subtopic)에 집중하고 관련 문서를 추가로 검색할 수 있도록 하여 반복적인 검색(iterative search) 및 깊이 있는 지식 발견(deep knowledge discovery)을 가능하게 합니다. 또한, LLM(대형 언어 모델) 기능과 클러스터 기반(cluster-based) 방법을 결합하여 효율적인 탐색(browsing) 방법을 제공합니다.

- **Performance Highlights**: 우리의 접근 방식은 자동 및 수동 평가를 통해 CLUSTREC-COVID 및 SCITOC의 두 가지 새로운 벤치마크(benchmarks)에서 효과성을 입증했습니다. 또한, 우리의 코드, 프롬프트(prompts), 벤치마크는 공개적으로 제공됩니다.



### Evaluating Named Entity Recognition Using Few-Shot Prompting with Large Language Models (https://arxiv.org/abs/2408.15796)
Comments:
          Github repo: this https URL

- **What's New**: 이 논문은 Named Entity Recognition (NER) 작업에 대한 Few-Shot Prompting의 효과 및 최신 대형 언어 모델(Large Language Models, LLMs)의 성능을 평가합니다. 전통적인 NER 시스템은 방대한 수의 레이블이 붙은 데이터셋이 필요하지만, Few-Shot Prompting은 최소한의 예제로 모델이 엔티티를 인식할 수 있도록 도와줍니다.

- **Technical Details**: 본 연구는 NER에서 Few-Shot Learning (FSL) 기법의 실성을 조사합니다. 실험은 GeoEDdA 데이터셋을 사용하여 수행되었으며, LLMs인 GPT-4와 BERT처럼 잘 알려진 모델들과 비교합니다. 실험 결과, LLM은 적은 데이터로 새로운 엔티티 유형 및 도메인에 적응하는 데 강점을 보이지만, 전통적인 완전 감독 방식과 비교할 때 성능 차이가 존재합니다.

- **Performance Highlights**: GPT-3.5보다 개선된 성능을 보인 GPT-4의 span 수준에서는 49%의 정확도를 보여주었고, token 수준에서 미세 평균 precision, recall 및 F1-score를 평가했습니다. 또한, 각 모델의 결과는 매우 다양한 변동성을 보였고, 일부 모델은 필요한 레이블 세트를 이해하지 못하거나 입력 문장을 반복하기만 하는 경우도 있었습니다.



### PDSR: A Privacy-Preserving Diversified Service Recommendation Method on Distributed Data (https://arxiv.org/abs/2408.15688)
- **What's New**: 이 논문에서는 Privacy-preserving Diversified Service Recommendation (PDSR) 방법을 제안하여 서비스 추천의 정확성과 다양성 간의 균형을 유지하면서 데이터 공유 시 개인 정보 보호를 보장합니다. 특히, LSH (Locality-Sensitive Hashing) 메커니즘을 활용하여 서비스 유사성 그래프를 구성합니다.

- **Technical Details**: PDSR 방법은 다양한 플랫폼 간 개인 정보 보호를 유지하는 데이터 공유를 촉진하고, K 서비스를 선택하여 정확성과 다양성 평가를 극대화하는 새로운 정확성-다양성 메트릭을 설계합니다. 이 시스템은 NP-hard 문제로 증명되며, 2-근사 알고리즘을 통해 해결됩니다. 실 데이터 세트를 기반으로 한 광범위한 실험을 통해 효능을 입증합니다.

- **Performance Highlights**: PDSR 방법은 다양한 서비스 추천의 성과를 높이기 위해 기존의 연구들과 비교하여 데이터 공유의 위험을 최소화하며, 실험 결과에서 높은 정확성과 다양성을 보여줍니다.



### Lyrically Speaking: Exploring the Link Between Lyrical Emotions, Themes and Depression Risk (https://arxiv.org/abs/2408.15575)
Comments:
          Accepted at the 25th International Society for Music Information Retrieval Conference (ISMIR) 2024, San Francisco, United States

- **What's New**: 이 연구에서는 우울증 위험이 있는 개인의 온라인 음악 소비를 분석하여 가사 주제와 감정이 어떤 영향을 미치는지를 살펴보았습니다. 지난 음악 청취 기록에서 추출한 가사를 자연어 처리(Natural Language Processing) 기법으로 분석하였고, 우울증 위험이 있는 개인이 낮은 평온과 낮은 각성을 지닌 가사와 사랑, 자아참조 및 양가감정을 표현하는 가사를 선호한다는 결과를 발견했습니다.

- **Technical Details**: 우리는 541명의 Last.fm 사용자로부터 수집된 데이터를 사용하여 연구를 진행했습니다. Kessler의 심리적 고통 척도(K10)를 사용하여 정신적 웰빙을 평가하였고, 가사는 Genius.com 및 MetroLyrics.com에서 추출되었습니다. 음악 가사를 Russell의 복합 감정 모델을 바탕으로 감정의 평온(Valence)과 각성(Arousal)으로 분류하여 분석하였습니다. 이를 위해 XLNet 기반의 딥 뉴럴 네트워크 모델을 사용했습니다.

- **Performance Highlights**: 연구 결과, 우울증 위험이 있는 개인은 낮은 감정의 가사로 분류되는 노래를 더 선호하는 경향이 있으며, 이는 기존의 우울 증상을 악화시킬 수 있는 위험 요소로 작용할 수 있습니다. 이 연구는 개인의 디지털 발자국을 통해 우울증 위험을 평가하고, 개인화된 추천 시스템 개발 가능성을 제시합니다.



### Temporal Graph Neural Network-Powered Paper Recommendation on Dynamic Citation Networks (https://arxiv.org/abs/2408.15371)
Comments:
          10 pages, 4 figures, accepted by SDU@AAAI-2024. The AAAI Workshop on Scientific Document Understanding (2024)

- **What's New**: 본 논문은 과학 논문의 기하급수적인 증가로 인해 관련 참고 문헌을 식별하는 것이 점점 더 어려워지고 있다는 문제를 다룬다. 기존 방법들은 주로 정적인 관점에서 후보 논문을 평가했으나, 이 연구는 인용 관계가 발생할 때마다 논문의 embedding을 지속적으로 업데이트하여 더욱 정확한 추천을 목표로 한다.

- **Technical Details**: Temporal Graph Neural Network(TGN)를 사용하여 새로운 인용 관계가 추가될 때 논문의 embedding을 업데이트하며, RNN(Recurrent Neural Network)을 기반으로 한 학습 가능한 메모리 업데이트 모듈을 통해 논문의 embedding의 변화를 학습한다. 이를 통해 시간이 지남에 따라 논문의 인용 가능성을 예측하고, Graph Transformer convolutional(TransConv) 레이어와 멀티 헤드 어텐션 기법을 함께 사용하여 네트워크 내의 상호작용을 효과적으로 집계한다.

- **Performance Highlights**: 313,278개의 기계 학습 관련 논문을 포함한 오픈 데이터셋을 기반으로 한 실험 결과, 제안된 방법이 최신 기술들에 비해 논문 추천 정확성이 뛰어남을 입증했다.



### Interactive Agents: Simulating Counselor-Client Psychological Counseling via Role-Playing LLM-to-LLM Interactions (https://arxiv.org/abs/2408.15787)
- **What's New**: 본 논문에서는 대형 언어 모델(LLM)을 활용하여 상담자-클라이언트 상호작용을 시뮬레이션하는 프레임워크를 제안합니다. LLM을 통한 가상 상담 시스템이 실제 심리 상담을 대체하는 가능성을 모색한 것입니다.

- **Technical Details**: 제안된 프레임워크는 Clien와 Counselor 역할을 각각 LLM으로 시뮬레이션하며, 복합적 치료 기법을 사용하여 전문가 수준의 반응을 생성합니다. 이 시스템은 GPT-4 모델을 제로샷 프롬프트(Zero-shot prompting) 방식으로 구현하였습니다.

- **Performance Highlights**: 실험 결과, LLM이 생성한 대화는 기존의 최첨단 모델들보다 뛰어난 성능을 보였으며, 심리 상담 대화의 질이 매우 높음을 확인했습니다. 연구진은 SimPsyDial이라고 하는 LLM 생성 데이터셋을 발표하며, 코드 및 모델을 GitHub를 통해 공개하였습니다.



### CAPER: Enhancing Career Trajectory Prediction using Temporal Knowledge Graph and Ternary Relationship (https://arxiv.org/abs/2408.15620)
- **What's New**: 본 논문에서는 CAPER라는 새로운 Career Trajectory Prediction (CTP) 방법을 제안하며, 사용자, 직위, 회사 간의 상호 의존성을 고려하고 시간에 따른 특성 변화를 포착하는 복합적인 모델링 방식을 소개합니다.

- **Technical Details**: CAPER는 Temporal Knowledge Graph (TKG) 모델링을 활용하여 세 가지 주요 단위(사용자, 직위, 회사) 간의 상호 의존성을 모델링합니다. 이 모델은 경력 정보를 포함하는 KG(Knowledge Graph) 스냅샷의 시퀀스로 구성되어, 각 스냅샷은 특정 시점에서 공존하는 경력을 포함합니다. 이를 통해 시간에 따른 사용자와 회사 특성의 변화를 포착합니다.

- **Performance Highlights**: CAPER는 실제 경력 데이터셋을 기반으로 한 실험에서 네 가지 기초 모델 및 최근의 TKG 추론 방법, 그리고 다섯 가지 최첨단 CTP 방법을 능가하며, 회사와 직위 예측의 경우 각각 6.80% 및 34.58% 더 높은 정확도를 기록했습니다.



### Civiverse: A Dataset for Analyzing User Engagement with Open-Source Text-to-Image Models (https://arxiv.org/abs/2408.15261)
- **What's New**: 본 연구는 TTI (Text-to-Image) AI 플랫폼 CivitAI를 분석하여 문화적 관점에서 오픈소스 TTI 프레임워크를 체계적으로 조사합니다. 특히, 사용자 의도와 행동을 파악하기 위해 Civiverse 프롬프트 데이터셋을 소개하며, 이는 수백만 개의 이미지와 관련 메타데이터를 포함합니다.

- **Technical Details**: 대규모 데이터셋 Civiverse 6M을 사용하여 텍스트 프롬프트의 의미적 특성을 분석하였습니다. 이 데이터셋은 사용자가 제공한 프롬프트를 기반으로 생성된 6,546,165개의 이미지 URL과 메타데이터로 구성되어 있습니다. 분석은 사용자와 TTI 모델 간의 상호작용 패턴에 초점을 맞추어 진행되었습니다.

- **Performance Highlights**: 연구 결과, 성적 콘텐츠 생성에 대한 주된 선호가 나타났으며, 의미적 내용의 동질화 경향이 확인되었습니다. 이러한 통찰은 그림을 생성하는 모델 내에서 여성 혐오와 해로운 고정관념이 지속될 가능성을 강조하며, 문화적 다양성이 감소하고 있음을 나타냅니다.



New uploads on arXiv(cs.CV)

### Eagle: Exploring The Design Space for Multimodal LLMs with Mixture of Encoders (https://arxiv.org/abs/2408.15998)
Comments:
          Github: this https URL, HuggingFace: this https URL

- **What's New**: 이번 연구는 여러 비전 인코더를 혼합하여 MLLMs의 성능을 향상시키기 위한 체계적인 탐색을 진행했습니다. 특히, 시각 정보 인코딩의 효과를 비교하고, 비전 전문가의 선택 및 통합 방식에 대한 심층 분석이 포함되어 있습니다.

- **Technical Details**: 비전 전문가의 혼합 설계를 통해 MLLMs의 성능을 높일 수 있는 원리를 발견했습니다. 주요 요소로는 Pre-Alignment를 통해 비전 인코더와 언어 토큰 간의 간극을 줄이는 방식을 도입했습니다. 이 연구는 다양한 비전 인코더의 성능을 벤치마킹하고, 여러 인코더의 효과적인 결합 방식을 규명했습니다.

- **Performance Highlights**: Eagle 모델은 주요 MLLM 벤치마크에서 다른 오픈 소스 모델들을 초월하는 성과를 달성했으며, OCR 및 문서 이해 작업에서 뚜렷한 이점이 있었습니다. 이 연구는 비전 중심의 MLLMs 설계에서 상당한 성과를 거두었음을 보였습니다.



### Spatio-Temporal Context Prompting for Zero-Shot Action Detection (https://arxiv.org/abs/2408.15996)
- **What's New**: 이번 논문에서는 클립 모델(CLIP)을 활용하여 보지 못한 행동을 감지하는 새로운 방법인 ST-CLIP을 제안합니다. 이는 인물-맥락 상호작용(Person-Context Interaction)을 모델링하여, 기존의 행동 감지 방식을 개선하는 데 초점을 맞추고 있습니다. 특히, 다중 행동 비디오에서 개인의 행동을 인식할 수 있는 방법을 제공합니다.

- **Technical Details**: ST-CLIP은 사전 학습된 시각-언어 모델을 이용하여, 시각 및 텍스트 영역 모두에서 제로샷 스페이쇼-템포럴 액션 디텍션(Zero-Shot Spatio-Temporal Action Detection)을 수행합니다. 이 방법은 파라미터 및 자원을 추가로 필요로 하지 않으며, 멀티 레이어 Context Prompting 모듈과 Interest Token Spotting 메커니즘을 통해 각 개인의 행동과 관련된 맥락 토큰을 식별하고, 이를 활용하여 행동 분류를 정확하게 수행합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법(ST-CLIP)은 J-HMDB와 UCF101-24 데이터셋에서 기존 방법보다 우수한 성과를 거두었으며, AVA 데이터셋에서는 동일한 비디오 내에서 다양한 보지 못한 행동을 개별적으로 감지하는 능력을 입증했습니다. 이러한 결과는 실제 응용에 근접할 뿐만 아니라, 제로샷 학습의 강력한 일반화 능력을 강조합니다.



### TEDRA: Text-based Editing of Dynamic and Photoreal Actors (https://arxiv.org/abs/2408.15995)
Comments:
          For project page, see this this https URL

- **What's New**: 본 연구는 첫 번째로 텍스트 기반으로 동적 전신 아바타의 외관을 수정할 수 있는 TEDRA라는 새로운 방법을 제안합니다. TEDRA는 아바타의 높은 충실도와 공간-시간적 일관성을 유지하며 사용자 요청에 따른 패션 스타일 편집을 지원합니다.

- **Technical Details**: TEDRA는 두 단계 프로세스를 통해 제어 가능한 고충실도의 디지털 아바타를 생성합니다. 첫 번째 단계에서, 실제 배우의 다중 뷰 비디오로부터 아바타를 훈련한 후, 사전 훈련된 생성적 확산 모델을 다양한 프레임으로 개인화합니다. 결과적으로, Personalized Normal Aligned Score Distillation Sampling (PNA-SDS)을 활용하여 입력된 텍스트 프롬프트에 따라 동적 아바타를 수정합니다. 또한, 고품질 수정을 보장하기 위해 타임스텝 어닐링 전략을 도입하였습니다.

- **Performance Highlights**: TEDRA는 기존의 방법들과 비교하여 기능성 및 시각적 품질에서 뚜렷한 향상을 보여줍니다. 주관적 및 수치적 평가를 통해 다채로운 텍스트 기반 수정이 가능함을 입증하며, 초기 아이덴티티의 무결성을 유지합니다. 편집된 아바타의 애니메이션을 통해 시간적 일관성과 뛰어난 성능을 강조합니다.



### Perceive-IR: Learning to Perceive Degradation Better for All-in-One Image Restoration (https://arxiv.org/abs/2408.15994)
Comments:
          13 pages, 8 figures

- **What's New**: 본 논문에서는 All-in-One 이미지 복원 기술 중 Perceive-IR을 제안합니다. 이 기술은 다양한 왜곡 유형 및 수준에 관계없이 복원된 이미지가 원본 이미지와 더 유사하도록 하는 정밀 품질 제어를 가능하게 합니다.

- **Technical Details**: Perceive-IR은 두 단계로 구성됩니다: (1) prompt learning 단계에서 CLIP(Contrastive Language–Image Pretraining) 인식 공간에서 prompt-image 유사성을 제한하여 3단계 품질 수준을 구별하는 품질 인지자를 학습합니다. (2) 복원 단계에서 semantic guidance module(SGM)과 compact feature extraction(CFE)를 사용하여 복원 과정을 촉진합니다.

- **Performance Highlights**: Perceive-IR은 다양한 복원 작업에서 최신 방법들보다 더 뛰어난 성능을 보여주며, 이미지 노이즈 제거, 안개 제거, 비 내리기, 흐림 제거, 저조도 향상 작업 등에서 우수한 일반화 능력을 입증하였습니다.



### ClimDetect: A Benchmark Dataset for Climate Change Detection and Attribution (https://arxiv.org/abs/2408.15993)
- **What's New**: ClimDetect는 기후 변화 신호를 확인하는 데 사용될 수 있는 816,000개 이상의 일일 기후 스냅샷으로 구성된 표준화된 데이터 세트를 제공합니다. 이는 전통적인 기후 신호 감지 및 귀속(D&A) 방법론에 대안을 제시합니다.

- **Technical Details**: ClimDetect는 Coupled Model Intercomparison Project Phase 6 (CMIP6) 모델 집합에서 수집된 데이터를 통합하며, 과거 및 미래 기후 시나리오의 다양한 입력 변수와 목표 변수를 포함합니다. 본 연구는 Vision Transformers (ViT) 아키텍처를 도입하여 공간적 기후 데이터를 분석합니다.

- **Performance Highlights**: ClimDetect 데이터 세트와 관련된 분석 코드는 과학 연구의 기준을 설정하며, 기후과학 커뮤니티에서 다양한 모델링 기술 탐색을 촉진합니다. 이를 통해 연구자들은 기후 역학에 대한 명확한 통찰을 얻어 기후 변화 문제를 해소하는 데 기여할 수 있습니다.



### Distribution Backtracking Builds A Faster Convergence Trajectory for One-step Diffusion Distillation (https://arxiv.org/abs/2408.15991)
- **What's New**: 본 논문에서는 기존의 score distillation 기법의 문제를 해결하기 위해 새로운 기법인 Distribution Backtracking Distillation (DisBack)을 제안합니다. DisBack은 teacher 모델의 전체 수렴 경로를 활용하여 학생 생성기(student generator)를 훈련시키는 방식입니다.

- **Technical Details**: DisBack은 두 가지 단계로 구성됩니다: Degradation Recording과 Distribution Backtracking입니다. Degradation Recording 단계에서는 teacher 모델이 초기 학생 생성기의 분포에 맞춰 조정되어 degradation path를 기록합니다. 이후 Distribution Backtracking 단계에서 이 경로를 뒤로 추적하며 학생 모델을 훈련시킵니다.

- **Performance Highlights**: DisBack은 기존 distillation 방법보다 더 빠르고 효율적인 수렴 속도를 보여주며, 높은 생성 품질을 유지합니다. 실험 결과에 따르면, DisBack은 빠른 수렴 속도와 함께 기존 방법들과 유사한 생성 성능을 달성하였음을 나타냅니다.



### More Text, Less Point: Towards 3D Data-Efficient Point-Language Understanding (https://arxiv.org/abs/2408.15966)
- **What's New**: 이번 논문에서는 3D 데이터 부족 문제를 해결하기 위해 3D Data-Efficient Point-Language Understanding(3DEPL)이라는 새로운 과제를 제안합니다. GreenPLM 모델을 통해 최소한의 3D 포인트 클라우드 데이터와 텍스트 데이터 쌍으로 강력한 3D 객체 이해를 가능하게 하고자 합니다.

- **Technical Details**: GreenPLM은 CLIP에서 영감을 받아 사전 학습된 포인트 클라우드-텍스트 인코더를 사용하여 3D 포인트 클라우드 공간을 텍스트 공간에 매핑합니다. 6M 개의 자유 텍스트 설명을 생성하고 세 단계의 훈련 전략을 설계하여 LLM이 다양한 모달리티 간의 본질적인 연결을 탐색하도록 돕습니다.

- **Performance Highlights**: GreenPLM은 기존의 최첨단 모델이 사용하는 3D 훈련 데이터의 12%만으로도 우수한 3D 이해력을 달성합니다. 또한, GreenPLM은 텍스트 전용 데이터만으로 경쟁력 있는 성능을 발휘할 수 있습니다.



### Efficient Slice Anomaly Detection Network for 3D Brain MRI Volum (https://arxiv.org/abs/2408.15958)
Comments:
          15 pages, 5 figures

- **What's New**: 본 연구에서는 뇌 MRI 스캔에서 비정상 패턴을 식별할 수 있는 효과적인 방법이 필요하다는 점을 강조합니다. 기존 기술들은 정상 뇌 이미지의 다양성 때문에 종종 어려움을 겪으며, 처리 속도가 느린 경향이 있습니다. 이를 해결하기 위해 SimpleSliceNet이라는 간소화된 접근 방식을 개발하였습니다.

- **Technical Details**: SimpleSliceNet은 ImageNet에서 사전 훈련된 모델을 활용하여 2D 슬라이스(feature extractor) 특성 추출기로 사용하고, 이를 통해 3D 뇌 MRI 볼륨에서 이상 감지 작업을 수행합니다. 조건부 정규화 흐름(Conditional Normalizing Flow)을 통합하고, 반푸시( Semi-Push-Pull) 메커니즘을 적용합니다.

- **Performance Highlights**: 실험 결과, SimpleSliceNet은 최신 2D 및 3D 모형에 비해 정확도, 메모리 사용량, 처리 시간에서 모두 우수한 성능을 보였습니다. 이 모델은 임상 환경에서의 활용 가능성을 더욱 높이며, 의료 영상 접근성을 넓히는 데 기여할 수 있습니다.



### Fall Detection for Smart Living using YOLOv5 (https://arxiv.org/abs/2408.15955)
- **What's New**: YOLOv5mu 모델을 활용하여 스마트 홈 환경에서의 떨어짐 감지 시스템을 소개하며, 평균 정밀도(mAP) 0.995의 높은 정확도를 보임으로써 효과적인 저조도 조건에서도 유용성을 입증함.

- **Technical Details**: 본 연구는 YOLOv5mu를 사용하여 실시간으로 떨어짐 이벤트를 감지하는 시스템을 개발하였으며, 데이터 증강(data augmentation) 기술을 통해 다양한 환경에서도 높은 강건성과 적응성을 보여줌. 데이터셋은 고해상도 비디오에서 추출한 611개의 이미지를 기반으로 수집되었고, 4개의 범주로 수동으로 주석을 달아 분류함.

- **Performance Highlights**: 본 시스템은 뛰어난 실시간 처리 성능과 정확성을 바탕으로 안전을 개선하고 긴급 대응을 강화하여 거주자의 삶의 질을 향상시키는 데 기여할 것으로 기대됨.



### InstanSeg: an embedding-based instance segmentation algorithm optimized for accurate, efficient and portable cell segmentation (https://arxiv.org/abs/2408.15954)
Comments:
          12 pages,6 figures

- **What's New**: 새로운 알고리즘 InstanSeg는 세포와 핵을 현미경 이미지에서 보다 정확하게 식별하며, 최소 60%의 처리 시간을 단축합니다.

- **Technical Details**: InstanSeg는 수정된 U-Net 백본을 기반으로 하는 새로운 임베딩 기반 인스턴스 세분화 메서드입니다. 이 방법은 최적 선택된 시드 주위에서 픽셀 임베딩을 클러스터링하여 정확성과 효율성 및 이동성을 최적화합니다. 알고리즘은 TorchScript로 완전히 직렬화 가능하며, 다양한 하드웨어에서 GPU 가속을 지원합니다.

- **Performance Highlights**: InstanSeg는 가장 많이 사용되는 방법과 비교하여 정확성을 크게 향상시키며, 공공 세포 세분화 데이터셋 6개에서 새로운 성능 기준을 세우고, 처리 시간을 약 2.5배에서 45배까지 줄입니다.



### Local Descriptors Weighted Adaptive Threshold Filtering For Few-Shot Learning (https://arxiv.org/abs/2408.15924)
- **What's New**: 본 논문에서는 Few-shot learning(소수 샘플 학습)에서의 이미지 분류 성능을 향상시키기 위해 새로운 가중 가변 임계값 필터링(weighted adaptive threshold filtering, WATF) 전략을 제안합니다. 이 방법은 이미지 카테고리에 가장 적합한 지역적 특성을 선택하여 배경 잡음을 효과적으로 걸러낼 수 있도록 설계되었습니다.

- **Technical Details**: WATF는 현재의 작업과 이미지 맥락에 따라 동적으로 조정되어, 이미지 카테고리에 가장 관련성이 높은 지역적 descriptor를 선택합니다. 이를 통해 모델은 카테고리 관련 정보에 더욱 집중하게 되며, 불필요한 배경 지역으로부터 방해를 최소화합니다. 실험은 N-way K-shot 프레임워크에서 진행되었으며, 시각화 실험을 통해 지역적 descriptor의 가중치가 정규 분포를 따른다는 것을 입증하였습니다.

- **Performance Highlights**: 본 방법은 세 가지 widely-used few-shot learning(소수 샘플 학습) 분류 데이터셋에서 최신 기술을 뛰어넘는 성능을 보여주었으며, 특히 CUB-200 데이터셋에서는 몇 가지 최신 transfer learning(전이 학습) 기반의 방법보다도 뛰어난 효과를 보였습니다. 이로 인해 본 방법은 이후 few-shot learning 연구에 상당한 참고 가치가 있을 것으로 기대됩니다.



### DiffAge3D: Diffusion-based 3D-aware Face Aging (https://arxiv.org/abs/2408.15922)
- **What's New**: DiffAge3D는 첫 번째 3D 인식 노화 프레임워크로, 2D에서 3D로 확장되어 신뢰할 수 있는 노화 및 동일성 유지 기능을 제공합니다.

- **Technical Details**: DiffAge3D는 단일 이미지를 기반으로 3D 노화 모델을 생성하며, 3D GAN과 CLIP 모델을 활용하여 노화 데이터를 생성합니다. 이 프레임워크는 시점 인식 확산 모델을 이용하여 카메라 포즈와 얼굴 나이를 조절합니다.

- **Performance Highlights**: DiffAge3D는 기존 방법에 비해 다각도 일관성 있는 노화 생성 및 세부 사항 보존에서 우수한 성능을 보입니다.



### Leveraging Open Knowledge for Advancing Task Expertise in Large Language Models (https://arxiv.org/abs/2408.15915)
Comments:
          28 pages, 12 tables, 10 figures

- **What's New**: 이 논문은 특정 도메인 작업에서 대규모 언어 모델(LLMs)의 전문성을 개선하기 위해 몇 가지 인적 주석이 있는 샘플(K-shot)을 사용하는 새로운 접근 방식을 제안합니다. Open knowledge(공개 지식)를 활용하여 모델과 데이터 선택에서의 기존 방법들이 일반적인 능력 향상에 집중한 것에 비해, K-shot 데이터를 사용하여 전문성을 높이고자 합니다.

- **Technical Details**: 우리는 K-shot 학습 설정을 통해 LLM의 작업 전문 지식을 발전시키는 효율적이고 확장 가능한 파이프라인을 개발하였습니다. Mixture-of-Experts (MoE) 시스템을 통해 여러 전문가 간의 상호 보완적인 지식을 최대한 활용하며, K-shot 데이터를 통해 문제 해결 능력을 가진 모델을 선택합니다. 우리가 제안한 방법은 reasoning perplexity(추론 혼란도), exact match accuracy(정확한 매치 정확도), group diversity(그룹 다양성)를 바탕으로 합니다.

- **Performance Highlights**: 광범위한 실험 결과, 우리의 방법이 다양한 작업에서 공개 지식을 활용하는 기존 방법들보다 우수함을 입증하였습니다. K-shot 데이터를 통해 적합한 전문가 모델과 유용한 지시 사항을 선별하는 데 성공했으며, 저비용으로도 실세계에 적용 가능한 작업 전문성을 확보할 수 있었습니다.



### CoRe: Context-Regularized Text Embedding Learning for Text-to-Image Personalization (https://arxiv.org/abs/2408.15914)
- **What's New**: 본 연구는 텍스트-이미지 개인화의 새로운 접근 방식인 Context Regularization (CoRe)을 소개합니다. CoRe는 새로운 개념의 텍스트 임베딩을 학습하는 데 있어 문맥 토큰을 정규화함으로써 텍스트 정렬(text alignment)과 아이덴티티 보존(identity preservation) 간의 균형을 개선합니다.

- **Technical Details**: CoRe 방법은 CLIP 텍스트 인코더를 활용하여 문맥 토큰과 새로운 개념의 상호작용을 정확하게 처리합니다. 새로운 개념의 텍스트 임베딩을 기존 토큰과 원활하게 통합할 수 있도록 입력 임베딩 공간에 적절히 내장합니다. CoRe는 생성된 이미지를 요구하지 않고도 임의의 프롬프트에 적용 가능하며, 훈련 중에 임의의 정규화 프롬프트 세트를 통해 범위 있는 정규화를 진행합니다.

- **Performance Highlights**: 우리의 방법은 아이덴티티 보존과 텍스트 정렬 모두에서 여러 기준선 방법들을 초과하는 성능을 보였습니다. 특히 고도의 시각적 변동성을 요구하는 프롬프트에 대해 우수한 성능을 나타내며, 얼굴 개인화에서도 최근의 세 가지 얼굴 개인화 방법에 비해 더 나은 아이덴티티 보존 얼굴 이미지를 생성합니다.



### Disentangled Diffusion Autoencoder for Harmonization of Multi-site Neuroimaging Data (https://arxiv.org/abs/2408.15890)
- **What's New**: 이번 연구에서는 새로운 모델인 Disentangled Diffusion Autoencoder(DDAE)를 도입하여, 다양한 사이트에서 얻은 MR 이미지를 조화롭게 만드는 데에 초점을 맞추었습니다.

- **Technical Details**: DDAE는 특정 이미지의 생물학적 변동성을 보존하면서 사이트 효과를 제거할 수 있도록 설계된 확산 모델입니다. 이 모델은 7개 다른 사이트에서 수집된 데이터를 활용하여 고해상도의 조화된 2D MR 이미지를 생성합니다.

- **Performance Highlights**: DDAE는 이전의 ComBat, cVAE, 스타일-인코딩 GAN 모델들에 비해 더욱 우수한 성능을 보이며, 이미지의 질을 높이고 사이트 효과를 제거한 동시에 알려진 및 알려지지 않은 원천의 생물학적 변동성을 효과적으로 보존했습니다.



### LLaVA-MoD: Making LLaVA Tiny via MoE Knowledge Distillation (https://arxiv.org/abs/2408.15881)
- **What's New**: LLaVA-MoD는 대규모 다중 모달 언어 모델(l-MLLM)로부터 지식을 증류하여 소규모 다중 모달 언어 모델(s-MLLM)의 효율적인 훈련을 지원하는 새로운 프레임워크이다. 이 연구는 모델 구조 최적화와 지식 전이 전략 개선을 통해 s-MLLM의 성능을 높인다.

- **Technical Details**: LLaVA-MoD는 Sparse Mixture of Experts (MoE) 구조를 통합하여 s-MLLM의 네트워크 구조를 최적화하고, Kullback-Leibler (KL) 발산 최소화를 통한 mimic distillation 프로세스를 기반으로 진행된다. 또한, Direct Preference Optimization (DPO)를 통한 preference distillation 단계에서 l-MLLM을 기준 모델로 활용하여, exemplary와 inferior 사례를 구분하는 능력을 강화한다.

- **Performance Highlights**: LLaVA-MoD는 2B 활성 매개변수만으로 Qwen-VL-Chat-7B 모델을 평균 8.8% 초과 달성했으며, 훈련 데이터의 단 0.3%와 23%의 훈련 가능한 매개변수를 사용하여 이러한 결과를 도출하였다. 이 결과는 LLaVA-MoD의 효과적인 지식 증류 능력을 강조하며, 더 효율적인 다중 모달 언어 모델의 개발을 담보한다.



### Unleashing the Temporal-Spatial Reasoning Capacity of GPT for Training-Free Audio and Language Referenced Video Object Segmentation (https://arxiv.org/abs/2408.15876)
- **What's New**: 이번 논문에서는 오디오 및 언어 참조 비디오 객체 분할을 위한 훈련이 필요 없는 패러다임을 탐구하는 AL-Ref-SAM 2(오디오-언어-참조 SAM 2) 파이프라인을 제안합니다. 이는 GroundingDINO를 활용하여 단일 프레임에서 객체를 식별하고, SAM 2를 사용하여 비디오 전반에 걸쳐 이를 분할합니다.

- **Technical Details**: AL-Ref-SAM 2 파이프라인은 GPT-4를 통한 두 단계의 시공간 추론을 수행하는 GPT-보조 피벗 선택(GPT-PS) 모듈을 포함하고 있으며, 이로 인해 SAM 2에 고품질 초기 객체 프롬프트를 제공합니다. 또한, LBRU(Language-Binded Reference Unification) 모듈을 통해 오디오 신호를 언어 형식의 참조로 변환하여 AVS와 RVOS 작업을 통합적으로 처리할 수 있습니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, 우리의 훈련이 필요 없는 AL-Ref-SAM 2 파이프라인은 완전 감독학습 기반의 미세 조정 방법들과 유사한 또는 더 나은 성능을 보여줍니다.



### GenDDS: Generating Diverse Driving Video Scenarios with Prompt-to-Video Generative Mod (https://arxiv.org/abs/2408.15868)
- **What's New**: 이 연구는 자율주행 훈련을 위한 새로운 데이터 생성 방법인 GenDDS를 제안합니다. GenDDS는 다양한 교통 시나리오를 생성하기 위해 Stable Diffusion XL (SDXL) 모델을 활용하며, 고유한 환경 조건과 희귀한 발생 이벤트를 효과적으로 모델링하는데 중점을 두었습니다.

- **Technical Details**: GenDDS는 텍스트 설명을 기반으로 동영상 시나리오를 생성하는 방법으로, KITTI 데이터셋을 활용하여 LoRA(Low-Rank Adaptation) 모델을 훈련하고, ControlNet을 통해 다양한 주행 시나리오에 대한 제어 신호를 제공합니다. 이 접근 방식은 고해상도의 복잡한 비디오 시퀀스를 생성하는 데 효과적입니다.

- **Performance Highlights**: 실험 결과, GenDDS는 실제 주행 시나리오의 복잡성과 변동성을 잘 포착한 고품질 영상 생성을 보여주었습니다. 이는 자율주행 시스템 훈련을 위한 정교한 데이터 생성에 기여하며, 시뮬레이션 및 검증 목적의 가상 환경 구축에 새로운 가능성을 열어줍니다.



### microYOLO: Towards Single-Shot Object Detection on Microcontrollers (https://arxiv.org/abs/2408.15865)
Comments:
          Published at the ECML PKDD Conference 2023, at the 4th Workshop on IoT, Edge, and Mobile for Embedded Machine Learning

- **What's New**: 이 논문에서는 Cortex-M 기반 마이크로컨트롤러에서 YOLO를 사용한 단일 샷 객체 탐지의 가능성에 대한 결과를 제시합니다. 특히, microYOLO이라는 새로운 아키텍처를 통해 128x128 RGB 이미지를 분류할 때 3.5 FPS의 프레임 속도를 달성했습니다.

- **Technical Details**: microYOLO는 YOLO 아키텍처를 최적화하여 Flash 메모리가 800 KB 미만, RAM이 350 KB 미만인 자원 제한이 있는 플랫폼에서 작동하도록 설계되었습니다. 입력 이미지 해상도를 128x128으로 줄이고, 백본 네트워크의 학습 가능한 파라미터 수를 감소시켰습니다.

- **Performance Highlights**: microYOLO는 인체 탐지에서 평균 27.7%, 차량 탐지에서 평균 12.3%, 냉장고 내 장난감 식료품 탐지에서 평균 56.4%의 mAP0.5 값을 달성했습니다. 이는 데이터셋의 깊이와 복잡성의 차이에 따른 성능 차이를 보여줍니다.



### What is YOLOv8: An In-Depth Exploration of the Internal Features of the Next-Generation Object Detector (https://arxiv.org/abs/2408.15857)
- **What's New**: YOLOv8는 최신 YOLO 시리즈 모델로, 객체 탐지에서의 성능을 현저히 개선하기 위해 CSPNet 백본, FPN+PAN 넥, 앵커-없는 (anchor-free) 접근 방식을 도입하였습니다. 또한 Python 패키지와 CLI를 통해 개발자 친화적인 개선 사항을 추가하여 모델 훈련과 배포를 간소화했습니다.

- **Technical Details**: YOLOv8는 세 가지 주요 구성 요소로 이루어져 있습니다: 1) 백본 (Backbone) - 고급 CNN을 사용하여 멀티스케일 기능을 추출합니다. 2) 넥 (Neck) - Path Aggregation Network (PANet)에서 차별화된 멀티스케일 정보 흐름을 최적화합니다. 3) 헤드 (Head) - 앵커-없는 방법을 통해 경계 상자를 예측하고, 클래스 확률을 제공합니다. 데이터 증강 기법으로는 모자이크 및 믹스업을 사용하며, 분류 작업에 포컬 손실 (focal loss) 함수를 적용하여 성능을 향상시킵니다.

- **Performance Highlights**: YOLOv8는 Microsoft COCO 및 Roboflow 100 벤치마크에서 우수한 성능을 보이며, 다양한 하드웨어 플랫폼에서 실시간 처리와 높은 정확성을 유지합니다. 새로운 학습 알고리즘을 통해 기존 YOLO 버전들보다 더 빠른 훈련과 추론 시간을 달성하고, 작은 객체와 불완전한 객체의 탐지 능력이 개선되었습니다.



### Shot Segmentation Based on Von Neumann Entropy for Key Frame Extraction (https://arxiv.org/abs/2408.15844)
Comments:
          14 pages, 5 figures

- **What's New**: 이 논문에서는 Von Neumann 엔트로피를 기반으로 한 비디오 키 프레임 추출 알고리즘을 제안하며, 샷(segmentation) 분할을 통해 주요 프레임을 효과적으로 추출하는 방법을 소개합니다.

- **Technical Details**: 이 알고리즘은 비디오 시퀀스 내 프레임 간 유사성 행렬의 Von Neumann 엔트로피를 계산하여 각 샷의 초기 프레임을 키 프레임으로 선택합니다. 이 접근법은 프레임 간의 시간적(sequence) 정보도 통합하여 키 프레임을 효율적으로 추출할 수 있게 합니다.

- **Performance Highlights**: 실험 결과, 제안된 알고리즘은 중복 프레임을 최소화하면서 원본 비디오 내용을 완전하고 정확하게 표현할 수 있는 키 프레임을 추출하는 데 성공했습니다. 또한, 이 알고리즘의 계산 복잡도는 O(N²)로, 긴 입력 비디오 처리 시에도 짧은 처리 시간을 보장합니다.



### Network transferability of adversarial patches in real-time object detection (https://arxiv.org/abs/2408.15833)
Comments:
          7 pages, 6 figures, 1 table

- **What's New**: 이번 논문에서는 적대적 패치(adversarial patches)의 전이 가능성(transferability)을 다양한 객체 탐지기(object detectors) 아키텍처에서 조사합니다. 이를 통해 대형 모델로 최적화된 패치가 작은 모델로 최적화된 패치보다 더 나은 전이 가능성을 제공한다는 사실을 발견했습니다.

- **Technical Details**: 패치 최적화는 AdamW 옵티마이저를 사용하여 100 에포크(epoch) 동안 랜덤 노이즈로 초기화된 256x256 픽셀 크기의 패치를 최적화합니다. 패치는 COCO 데이터셋에서 사전 학습된 가중치를 사용하는 28개의 실시간 객체 탐지기에서 테스트됩니다. 이 논문은 YOLO(You Only Look Once) 아키텍처를 포함한 최신 객체 탐지기의 큰 그림을 제공합니다.

- **Performance Highlights**: 패치가 각기 다른 네트워크 아키텍처에서 훈련되었음에도 불구하고, 적대적 패치가 평균 평균 정밀도(mean average precision)에 미치는 영향이 상당히 높게 나타났습니다. 대규모 모델에서 최적화된 패치가 더 잘 전이되는 경향이 있다는 것을 입증했습니다.



### SITransformer: Shared Information-Guided Transformer for Extreme Multimodal Summarization (https://arxiv.org/abs/2408.15829)
Comments:
          8 pages, 5 figures, submitted to ACM Multimedia Asia 2024

- **What's New**: 이 논문에서는 극한의 멀티모달 요약(XMSMO)을 위한 SITransformer라는 새로운 모델을 제안합니다. 이 모델은 서로 다른 모달리티에서 공통의 중요한 정보를 추출하여 요약 품질을 향상시키는 크로스모달(shared information) 상호작용 모듈을 포함합니다.

- **Technical Details**: SITransformer는 크로스모달(shared information) 공유 정보 추출기 및 크로스모달 상호작용 모듈로 구성됩니다. 공유 정보 추출기는 차별화 가능한 top-k 선택 과정을 통해 서로 다른 모달리티에서 중요한 정보를 필터링해내며, 이는 요약에 관련된 내용을 더 효과적으로 선택하는 데 도움을 줍니다. 또한, transformer 구조를 사용하여 모달리티 간 학습을 수행하고, 최종적인 요약을 생성합니다.

- **Performance Highlights**: 광범위한 실험을 통해 SITransformer는 비디오 및 텍스트 요약 모두에서 요약 품질을 크게 향상시켰으며, 기존의 방법들과 비교했을 때 뛰어난 성능을 보였습니다.



### Mining Field Data for Tree Species Recognition at Sca (https://arxiv.org/abs/2408.15816)
- **What's New**: 새로운 연구에서는 공공 산림 재고 데이터에서 개별 나무 종 레이블을 자동으로 채굴하는 방법론을 제시하고 있습니다. 사전 훈련된 트리 감지 모델을 활용하여 공중 이미지에서 나무 인스턴스를 식별하고 현장 데이터와 일치시키는 과정이 거의 사람의 개입 없이 이루어집니다.

- **Technical Details**: 이 연구는 세 가지 주요 단계를 포함합니다: 1. 사전 훈련된 딥 모델을 사용하여 공중 이미지에서 개별 트리 탐지; 2. 감지된 나무와 NFI 데이터를 일치시켜 개별 나무의 데이터셋 구성; 3. 숲과 비숲 생태계를 아우르는 종 인식을 위한 딥 러닝 모델 훈련 및 검증.

- **Performance Highlights**: 실험 결과, 노이즈가 포함되거나 레이블이 없는 데이터 포인트를 추가했을 때 긍정적인 효과가 있음을 보여주어 대규모 개별 종 매핑의 강력한 잠재력을 강조합니다.



### DQFormer: Towards Unified LiDAR Panoptic Segmentation with Decoupled Queries (https://arxiv.org/abs/2408.15813)
Comments:
          13 pages, 10 figures

- **What's New**: 이번 논문에서는 LiDAR panoptic segmentation을 위해 새로운 프레임워크인 DQFormer을 제안합니다. DQFormer은 things와 stuff 쿼리를 본래적 특성에 따라 분리하여 개별적으로 디코딩하고, 분류와 세분화를 해체함으로써 모호성을 완화합니다.

- **Technical Details**: DQFormer은 multi-scale feature encoder, decoupled query generator 및 query-oriented mask decoder로 구성됩니다. 이는 voxel-level 특성과 multi-resolution 포인트 임베딩을 추출하여 객체 크기 및 작은 인스턴스를 정확히 찾기 위해 BEV(Top Down View) 임베딩에서 쿼리 제안을 생성합니다. 마스크 디코더는 masked cross-attention 메커니즘을 사용하여 쿼리와 mask 임베딩으로부터 대응하는 세분화 마스크를 디코딩합니다.

- **Performance Highlights**: nuScenes와 SemanticKITTI 데이터셋에서 DQFormer의 성능을 평가한 결과, 기존 방법들과 비교하여 탁월한 성능을 보였습니다. DQFormer은 LiDAR panoptic segmentation을 위한 통합된 워크플로우를 제공합니다.



### Multi-view Pose Fusion for Occlusion-Aware 3D Human Pose Estimation (https://arxiv.org/abs/2408.15810)
Comments:
          ECCV workshops 2024

- **What's New**: 이번 연구는 인간-로봇 협업 환경에서의 강력한 3D 인간 자세 추정을 위한 새로운 접근 방식을 제안합니다. 기존의 2D 특징 삼각측량에 의존하지 않고, 절대 단안 방법에서 제공하는 3D 골격을 기반으로 다중 뷰 융합을 수행합니다.

- **Technical Details**: 인간 자세 추정은 다중 카메라 시점을 통해 이루어지며, 각 카메라의 뷰에서 비롯된 3D 키포인트를 융합하여 보다 견고하고 신뢰할 수 있는 인간 자세 예측을 달성합니다. 이 과정에서 리프젝션 오차 최적화를 통해 정확한 3D 자세 추정이 이루어집니다. 특히, 지느러미 길이 대칭 제약을 도입하여 뼈의 길이에 대한 사전 정보 없이도 효과적인 결과를 제공합니다.

- **Performance Highlights**: 제안된 방법은 공공 데이터셋인 Human3.6M 및 인공적으로 차폐된 Human3.6M-Occluded 데이터셋에서 부각된 성능을 보이며, 실제 인간-로봇 협업 작업 세트에서도 기존 3D 인간 자세 추정 기법들을 초월하는 결과를 얻었습니다.



### Object Detection for Vehicle Dashcams using Transformers (https://arxiv.org/abs/2408.15809)
Comments:
          7 Pages, and 6 Figures

- **What's New**: 이 논문에서는 트랜스포머(transformers)를 사용한 새로운 객체 탐지 방법을 제안합니다. 이 방법은 첨단 DEtection TRansformer (DETR)를 기반으로 하여 다양한 주변 조건에서 강력한 성능을 보여줍니다.

- **Technical Details**: 제안된 시스템은 딥 러닝(Deep Learning) 기술을 활용하여 트랜스포머 기반의 객체 탐지 방식을 사용합니다. DETR 모델은 실세계 조건에서 수집된 데이터셋으로 훈련되었으며, mAP(Mean Average Precision) 값 0.95를 달성했습니다. 이는 인간 운전자가 어려움을 겪는 상황에서도 우수한 성능을 발휘합니다.

- **Performance Highlights**: 이 연구의 결과, 제안된 방법을 통해 스마트 대시캠 시스템의 객체 탐지 기능이 현저히 향상되었습니다. 특히, DETR 모델은 다양한 날씨와 조명 조건에서도 효과적으로 작동하며, 차량과 교통 표지판을 정확히 탐지할 수 있음을 입증했습니다.



### Visual Prompt Engineering for Medical Vision Language Models in Radiology (https://arxiv.org/abs/2408.15802)
Comments:
          Accepted at ECCV 2024 Workshop on Emergent Visual Abilities and Limits of Foundation Models

- **What's New**: 이번 연구에서는 Vision Language Models (VLM)에서 시각적 프롬프트(visual prompt) 엔지니어링을 통해 방사선 이미지 분류 성능 향상을 시도합니다. 주목할 점은, BiomedCLIP을 활용하여 방사선 이미지를 직접 수정하여 모델의 주의를 중요한 영역으로 유도하는 방법론을 개발했다는 것입니다.

- **Technical Details**: BioMedical 이미지를 기반으로 한 CLIP 모델에서 시각적 표시기(visual markers)를 사용하여 이미지 내의 특정 영역에 집중하도록 유도합니다. 이 연구는 JSRT 데이터셋에서 폐 결절 악성 분류를 중심으로 하며, 화살표, 원, 윤곽선 등의 시각적 프롬프트를 사용하여 AUROC, AUPRC, F1 점수, 정확도 등의 성능 지표를 개선하는 것으로 나타났습니다.

- **Performance Highlights**: 연구 결과, 시각적 프롬프트를 사용할 경우 AUROC, AUPRC, F1 점수, 그리고 일반 정확도 등이 유의미하게 향상되었습니다. 또한, 주의 맵(attention maps)을 제공하여 모델의 해석 가능성을 증대시키고, 임상적 관련성이 높은 영역에 초점을 맞추게 하였습니다. 이는 시각적 프롬프트가 방사선 이미지 분석에서 성능을 향상시킬 수 있는 간단하면서도 강력한 접근 방식임을 증명합니다.



### A Survey on Facial Expression Recognition of Static and Dynamic Emotions (https://arxiv.org/abs/2408.15777)
- **What's New**: 이번 논문은 정적 얼굴 표정 인식(Static Facial Expression Recognition, SFER) 및 동적 얼굴 표정 인식(Dynamic Facial Expression Recognition, DFER)을 아우르는 종합적인 서베이를 제공하며, 특히 현재 FER 분야의 도전 과제 및 최신 기술 동향을 심층적으로 분석합니다.

- **Technical Details**: 이 연구는 최근의 리뷰들을 비판적으로 비교하고, 일반적인 데이터셋과 평가 기준을 소개하며, SFER와 DFER의 주요 도전 과제 및 해결 방법을 체계적으로 정리합니다. SFER은 정적 이미지에서의 감정 분석을 다루며, DFER은 비디오 시퀀스에서의 감정 변화를 분석합니다.

- **Performance Highlights**: 연구는 SFER과 DFER의 주요 성능 지표를 논의하며, 다양한 벤치마크 데이터셋에서의 최신 발전 상황과 기술적 도전 과제를 분석하고, 향후 연구 방향을 제시합니다.



### A Survey on Evaluation of Multimodal Large Language Models (https://arxiv.org/abs/2408.15769)
- **What's New**: 이 논문은 Multimodal Large Language Models (MLLMs)의 평가 방법을 체계적이고 포괄적으로 검토하고 있습니다. MLLMs는 인간의 지각 및 추론 시스템을 모방하여 다양한 모달리티 인코더와 강력한 Large Language Models (LLMs)을 통합합니다. 이 논문은 기존 MLLM 평가 작업을 능력에 따라 검토하고 분류하여 인식, 인지, 추론 및 신뢰성 등 다양한 분야의 요구를 충족하는 방법을 제안합니다.

- **Technical Details**: MLLM은 LLMs를 '뇌'로 보고 다양한 모달리티 인코더를 감각 기관으로 배치하여 여러 모달리티 정보에 대한 이해와 추론 능력을 제공합니다. MLLM의 평가 단계와 메트릭스를 검토하고 평가 방법론을 다양한 시각에서 제시합니다.

- **Performance Highlights**: MLLMs의 평가 방법에 대한 체계적인 리뷰를 통해 MLLM의 강점과 한계를 정량화할 수 있는 기초를 제공합니다. 또한, MLLMs를 신뢰성 관점에서 탑재할 필요성을 강조하며, 다양한 하위 작업에서의 MLLMs 성능을 평가하는 방법을 제안합니다.



### Str-L Pose: Integrating Point and Structured Line for Relative Pose Estimation in Dual-Graph (https://arxiv.org/abs/2408.15750)
- **What's New**: 본 논문에서는 점(feature point)과 선(line segment) 특성을 통합한 기하학적 대응 그래프 신경망(Geometric Correspondence Graph neural network)을 제안합니다. 이는 포즈 추정의 정확성을 향상시키기 위해 기하학적 제약을 활용하는 새로운 접근 방식입니다.

- **Technical Details**: 제안된 Str-L Pose 모델은 Dual-Graph 모듈과 Feature Weighted Fusion Module을 활용하여 기하학적 및 시각적 특성을 효과적으로 집계합니다. Geometric Correspondence Graph는 특징 점과 선 세그먼트를 결합하여 효율적인 추정 프로세스를 가능하게 합니다.

- **Performance Highlights**: DeMoN 및 KITTI Odometry 데이터셋을 통한 실험에서, 제안된 방법은 최신 기술과 견줄 만한 성능을 보여주었습니다. 특히 복잡한 환경에서도 정확한 포즈 추정이 가능하다는 점에서 유망한 결과를 나타냈습니다.



### Segmentation-guided Layer-wise Image Vectorization with Gradient Fills (https://arxiv.org/abs/2408.15741)
- **What's New**: 이 논문에서는 레스터 이미지를 간결한 벡터 그래픽으로 변환하는 분할 중심(vectorization framework) 접근 방식을 제안합니다. 특히, 이 방법은 레이디얼 그라디언트 필을 지원하여 벡터화의 가능성을 확장합니다.

- **Technical Details**: 제안된 방법은 분할 중심(segmentation-guided) 초기화 절차를 통해 점진적으로 새 형태를 벡터 출력에 추가합니다. 추가적인 매개변수는 새로운 손실 함수로 최적화되어 기하학적(parameter) 및 그라디언트 매개변수를 최소화합니다.

- **Performance Highlights**: 여러 데이터셋에서의 수치적 메트릭 및 사용자 연구를 통해, 제안한 방법은 이전 작업 대비 개선된 시각 품질과 레이어별(topic) 토폴로지를 갖춘 벡터 그래픽을 자동 생성하는 데 효과적임을 보여줍니다.



### MambaPlace:Text-to-Point-Cloud Cross-Modal Place Recognition with Attention Mamba Mechanisms (https://arxiv.org/abs/2408.15740)
Comments:
          8 pages

- **What's New**: 이번 논문에서는 자연어 설명을 이미지에 통합하여 로봇의 위치 인식 성능을 향상시키는 Vision Language Place Recognition (VLVPR) 시스템을 제안합니다.

- **Technical Details**: 새로 제안된 MambaPlace 프레임워크는 거친(coarse) 단계에서 텍스트 설명과 3D 포인트 클라우드를 인코딩하며, 각기 다른 모달리티의 데이터를 향상시키고 정렬하기 위해 Text Attention Mamba (TAM) 및 Point Clouds Mamba (PCM)를 사용합니다. 세밀한(fine) 단계에서는 서로 모달리티의 정보를 융합하고 Cross Attention Mamba (CCAM)를 활용하여 특징을 강화합니다.

- **Performance Highlights**: MambaPlace는 KITTI360Pose 데이터세트에서 기존 최첨단 방법들에 비해 향상된 위치 정확도를 달성했습니다.



### Defending Text-to-image Diffusion Models: Surprising Efficacy of Textual Perturbations Against Backdoor Attacks (https://arxiv.org/abs/2408.15721)
Comments:
          ECCV 2024 Workshop The Dark Side of Generative AIs and Beyond

- **What's New**: 이 논문은 text-to-image diffusion models(텍스트-이미지 확산 모델)에 대한 backdoor attacks(백도어 공격)의 방어 전략으로 텍스트 변형(textual perturbation)을 제안합니다. 기존의 방어 방법이 충분히 연구되지 않은 상황에서, 간단한 변형을 통해 백도어 공격을 효과적으로 완화할 수 있음을 보여줍니다.

- **Technical Details**: 연구팀은 Textual Perturbation(텍스트 변형)을 사용하여 공격 트리거를 무효화하는 접근 방식을 고안했습니다. 이 방법은 text embedding space(텍스트 임베딩 스페이스)와 cross-attention maps(교차 주의 맵)에서 변화를 분석하였습니다. 텍스트를 변형하여 의미를 유지하면서 백도어 공격을 저지하는 방법으로, 동의어 변경 및 번역 등의 변형 전략을 사용합니다.

- **Performance Highlights**: 실험 결과, 제안된 텍스트 변형 방법은 Rickrolling, VillanDiffusion, Textual Inversion 등의 다양한 백도어 공격에 대해 효과적이며, 생성 품질에 미치는 영향을 최소화하면서도 공격 성공률(ASR)을 큰 폭으로 감소시킵니다. FID는 생성된 이미지와 실 이미지 간의 유사성을 평가하며, 원래 텍스트의 의미를 보존하는 범위 내에서 작은 감소를 보였습니다.



### Pixels to Prose: Understanding the art of Image Captioning (https://arxiv.org/abs/2408.15714)
- **What's New**: 이 논문은 이미지 캡셔닝(image captioning) 기술에 대한 포괄적인 리뷰를 제공하며, 기계 학습(machine learning) 분야에 들어서는 사람들을 위해 다양한 방법론을 포괄적으로 제공하고 있습니다.

- **Technical Details**: 이미지 캡셔닝 모델은 주로 인코더-디코더(encoder-decoder) 아키텍처를 사용하며, 인코더는 입력 이미지에서 여러 기능 벡터를 추출합니다. 본 논문에서는 CNN(Convolutional Neural Network)과 RNN(Recurrent Neural Network)의 진화, LSTM(Long Short-Term Memory) 및 GRU(Gated Recurrent Units)의 사용, 그리고 다양한 attention 메커니즘 등을 다루고 있습니다.

- **Performance Highlights**: 이미지 캡셔닝은 헬스케어(healthcare), 자율주행차(autonomous vehicles), 원격 sensing 등 다양한 분야에서 활용될 수 있으며, 높은 정확도와 다양한 결과를 제공하는 모델이 필요합니다.



### Towards Realistic Example-based Modeling via 3D Gaussian Stitching (https://arxiv.org/abs/2408.15708)
- **What's New**: 본 논문에서는 기존의 모델을 사용하여 새로운 모델을 재구축하는 방법, 즉 예시 기반 모델링을 활용하여 복잡한 3D 객체의 자연스러운 조합을 수행하는 새로운 접근 방식을 제안합니다. 이를 위해 3D Gaussian Splatting(3DGS) 기반의 포인트 표현 방식을 채택하여 실시간으로 여러 개의 Gaussian 필드를 결합하고 효과적으로 편집할 수 있는 사용자 친화적인 GUI를 개발했습니다.

- **Technical Details**: 제안된 방법은 세 가지 주요 단계로 구성됩니다. 첫째, 다수의 Gaussian 모델을 실시간으로 분할 및 변환하는 GUI의 사용, 둘째, KNN 분석을 통해 소스 모델과 타겟 모델 간의 경계 포인트를 식별, 셋째, 샘플링 기반 복제(Cloning) 및 그래디언트 제한을 사용하는 두 단계 최적화 방법을 적용하여 타겟 모델의 외관을 최적화합니다. 이러한 접근 방식은 3DGS의 불규칙적인 성격을 고려하여 기존의 gradient propagation 방식 대신 sampling 기반 최적화 전략을 사용합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 SeamlessNeRF와 같은 기존의 방법보다 실질적 합성을 수행하는 데 있어 뛰어난 성능을 발휘함을 보여줍니다. 특히, 두 단계 최적화 방식을 통해 외관의 질감과 색상을 원본 객체와 일치시킬 수 있으며, 실험에서까지 복잡한 현실 세계의 씬을 효과적으로 처리할 수 있음을 입증하였습니다. 더 많은 데모는 제공된 링크에서 확인할 수 있습니다.



### Synthetic Forehead-creases Biometric Generation for Reliable User Verification (https://arxiv.org/abs/2408.15693)
Comments:
          Accepted at Generative AI for Futuristic Biometrics - IJCB'24 Special Session

- **What's New**: 최근 연구는 수술 마스크로 가려진 얼굴을 인식할 수 있는 새로운 생체 인식 패턴으로 이마 주름 패턴을 제시하고, 고품질 이미지 수집에 대한 수요를 해결하기 위해 합성 생체 데이터 생성의 가능성을 강조합니다.

- **Technical Details**: 제안하는 프레임워크는 두 가지 주요 모듈로 구성됩니다: 1) Subject-Specific Generation Module (SSGM)은 이미지 쌍 간의 일대 다수 매핑을 학습하여 실제 인물에 대한 아이디어 인식 합성 이마 주름을 생성합니다. 2) Subject-Agnostic Generation Module (SAGM)은 새로운 합성 아이덴티티를 샘플링합니다. 생성된 이마 주름 이미지는 Fréchet Inception Distance (FID) 및 Structural Similarity Index Measure (SSIM)으로 평가되었습니다.

- **Performance Highlights**: 생성된 합성 이마 주름 데이터를 활용한 이마 주름 검증 시스템(FHCVS)에서 검증 정확도가 향상되었습니다.



### DEAR: Depth-Enhanced Action Recognition (https://arxiv.org/abs/2408.15679)
Comments:
          5 pages, 1 figure, 1 table, accepted at Human-inspired Computer Vision, ECCV

- **What's New**: 본 연구는 깊이 맵(depth map)과 RGB 기능을 통합하여 비디오에서의 행동 인식을 개선하는 새로운 접근 방식을 제시합니다. 이 방법은 3D 피처(3D features)를 사용하여 혼란스러운 장면에서도 높은 인식 정확성을 달성합니다.

- **Technical Details**: 이 연구에서는 Side4Video와 VideoMamba 프레임워크를 사용하여 RGB 프레임으로부터 깊이 맵을 생성하고, 두 개의 분리된 브랜치를 통해 이를 처리합니다. 이러한 접근 방식은 RGB과 깊이 정보를 통합하여 공간적 및 시간적 동적을 더 효과적으로 포착합니다. Gated Cross-Attention(GCA)를 활용한 후처리(fusion) 방식으로 각 모달리티 간의 상호작용을 최적화합니다.

- **Performance Highlights**: Our framework achieved 70.8% 정확도, 이는 기존 모델(VideoMamba-S, S4V)보다 각각 4.2%와 0.6% 높은 수준입니다. 특히, 특정 동작에서는 깊이 맵이 10% 가까운 성능 향상을 보여줄 정도로 효과적이었습니다.



### Deep Learning Based Speckle Filtering for Polarimetric SAR Images. Application to Sentinel-1 (https://arxiv.org/abs/2408.15678)
Comments:
          23 pages, 32 figures

- **What's New**: 본 연구는 Polarimetric SAR 이미지에서 스펙클( speckle )을 제거하기 위해 Convolutional Neural Network (CNN)를 기반으로 한 새로운 프레임워크를 제안합니다. 기존의 방법들은 단일 극화(single polarization) 이미지에 국한되어 있었으나, 본 논문은 다중 극화(polarimetric) 이미지에 적용 가능한 기술을 개발했습니다.

- **Technical Details**: 제안된 방법은 복합적인 공분산 행렬(complex covariance matrix)의 reversible transformation을 통해 실수 값의 강도(pixel intensity) 밴드 집합을 생성합니다. 여기에 CNN을 적용하여 스펙클을 제거하고, 변화 탐지(change detection) 전략을 포함하여 비정상적인 영역에서의 잘못된 학습을 방지합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 스펙클 감쇠(speckle reduction)와 해상도( resolution ) 보존 모두에서 우수한 성능을 보여주었습니다. 또한, 필터링된 이미지에서 인공물( artifacts )을 생성하거나 편향(bias)을 도입하지 않음을 확인하여 후속 극화 처리(polarimetric processing)와 활용에 적합함을 입증하였습니다.



### Towards reliable respiratory disease diagnosis based on cough sounds and vision transformers (https://arxiv.org/abs/2408.15667)
- **What's New**: 최근 깊은 학습(deep learning) 기법의 발전으로 호흡기 질환에 대한 기침 소리 데이터를 활용한 진단 분야에서의 성능 향상이 두드러지고 있습니다. 본 연구에서는 다양한 유형의 심층 모델을 평가하는 통합 프레임워크를 제시하며, 특히 ResNet18과 같은 경량 컨볼루션 신경망에서 비전 변환기(vision transformers)에 이르는 모델의 성능을 비교합니다.

- **Technical Details**: 우리는 자기 지도 학습(self-supervised learning) 및 감독 학습(supervised learning)을 기반으로 한 새로운 기침 기반 질병 분류 접근 방식을 제안하며, 이는 대규모 기침 데이터 세트를 사용하여 고정밀 분류를 이루어 냅니다. 특히 ImageNet, 오디오 데이터에 대한 사전 훈련(pre-trained) 모델과 호흡기 소리 데이터에 대한 모델을 포함하여 세 가지 카테고리의 사전 훈련 모델을 비교 평가합니다.

- **Performance Highlights**: 우리의 실험 결과는 COVID-19 진단을 위한 두 개의 벤치마크 데이터 세트와 COPD/non-COPD 분류를 위한 독점 데이터 세트에서 92.5%의 AUROC(Area Under the Receiver Operating Characteristic Curve)를 기록하며, 제안된 접근 방식이 기존 방법들보다 일관되게 우수한 성능을 보임을 보여줍니다.



### Merging and Splitting Diffusion Paths for Semantically Coherent Panoramas (https://arxiv.org/abs/2408.15660)
Comments:
          Accepted at ECCV 2024

- **What's New**: 이 논문은 최근 성과를 나타내는 이미지 생성에 사용되는 확산 모델(Diffusion Models)의 새로운 접근 방식을 제안합니다. 특히, 'Merge-Attend-Diffuse' 연산자를 통해 리얼리즘과 다양성을 유지하면서 시각적 및 의미적으로 일관된 파노라마 이미지를 생성할 수 있는 방법을 탐구합니다.

- **Technical Details**: 이 방법은 여러 파노라마 뷰의 차원에 해당하는 잠재(feature) 특징을 결합하여 이를 하나의 텐서로 평균화하고, 이를 바탕으로 자기 주의(self-attention) 및 교차 주의(cross-attention) 과정을 재구성합니다. 이를 통해 장기 이미지를 전체적으로 모델링하거나 각 뷰의 세부 사항을 살펴볼 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 입력 프로프트(prompt)와 생성된 이미지의 비주얼 품질을 유지하면서도 의미적 일관성을 증가시키는 데 성공했습니다. 이로 인해 실험적으로 도출된 결과들이 사용자 연구와 함께 체계적으로 검증되었습니다.



### TeFF: Tracking-enhanced Forgetting-free Few-shot 3D LiDAR Semantic Segmentation (https://arxiv.org/abs/2408.15657)
- **What's New**: 이 논문은 자율주행에서 3D LiDAR를 이용한 정밀한 주변 인식의 필요성을 강조하며, 미주석(newly emerged) 물체들로 인한 few-shot semantic segmentation 문제를 해결하기 위한 새로운 접근법을 제안합니다.

- **Technical Details**: 본 연구는 시간적 연속성(temporal continuity)을 활용하여 LiDAR 데이터를 통해 pseudo ground truths를 생성하고, 이를 통해 데이터 세트를 증강하여 모델이 새로운 클래스에 적응할 수 있도록 합니다. 또한, LoRA(Low-rank Adaptation) 기법을 통합하여 기존 클래스를 잊지 않도록 하여 기초 클래스와 새로운 클래스 간의 정확도를 유지합니다.

- **Performance Highlights**: SemanticKITTI 데이터셋을 사용한 실험 결과, 본 방법은 기존 few-shot 3D LiDAR semantic segmentation 방법보다 뛰어난 성능을 보이며, 특히 새로운 클래스에서 높은 정확도를 기록했습니다.



### Realigned Softmax Warping for Deep Metric Learning (https://arxiv.org/abs/2408.15656)
Comments:
          Preprint

- **What's New**: 이 논문은 Deep Metric Learning (DML)에서 기존의 손실 함수들이 거리 측정 공간 내에서 유사성과 비유사성을 동시에 조정하는 방식에 집중하는 것입니다. 특히 softmax 연산을 통해 여러 힘을 함께 작용하게 하는 대신 새로운 종류의 손실 함수를 제안하여 Euclidean 도메인에서 최적화할 수 있는 방법을 모색합니다.

- **Technical Details**: 제안된 방법은 softmax의 작용 아래에서 거리 측정 공간을 형성하는 밀집성과 분리성을 제어하기 위해 warping function을 사용합니다. 이는 따로 공간을 왜곡하는 방법으로, 특정 지역에서 밀집성 또는 분리성을 높이거나 감소시키는 것이 가능합니다. 이론적으로 Euclidean 측정을 통해 새로운 손실 함수들을 자세히 탐구하고, 간단한 예제를 제공하며 이를 다양한 metric learning 벤치마크에서 경쟁력 있는 결과를 내는 데 사용합니다.

- **Performance Highlights**: 이 논문에서 제안한 새로운 손실 함수는 기존의 DML 방식보다 우수한 성능을 보여주며, 여러 모범적인 metric learning 벤치마크 테스트에서 state-of-the-art 결과를 달성했습니다. 이는 DML 분야에서 더욱 발전된 손실 함수 디자인을 통해 거둔 성과입니다.



### Online pre-training with long-form videos (https://arxiv.org/abs/2408.15651)
Comments:
          GCCE2024

- **What's New**: 이번 연구에서는 연속적인 비디오 클립을 활용한 온라인 사전 훈련(online pre-training)의 영향을 조사했습니다. 세 가지 방법(Masked Image Modeling, contrastive learning, knowledge distillation)의 성능을 비교한 결과, contrastive learning이 최상의 성능을 보였습니다.

- **Technical Details**: 사전 훈련에 사용된 비디오 데이터셋은 AVA-Actions, HMDB51, UCF101로, 각기 다른 구조의 동작 인식 데이터셋입니다. 연구에서는 Random clip sampling과 Sequential clip sampling 두 가지 접근 방식을 사용하여 비디오 클립을 준비하였습니다.

- **Performance Highlights**: 세 가지 사전 훈련 방법 중 contrastive learning이 다운스트림 동작 인식 작업에서 가장 높은 성능을 기록했습니다. 특히 긴 형식의 비디오에서 학습하는 것이 짧은 비디오의 동작 인식에 유용하다는 것을 시사합니다.



### Leveraging Persistent Homology for Differential Diagnosis of Mild Cognitive Impairmen (https://arxiv.org/abs/2408.15647)
Comments:
          16 pages, 6 figures, 3 tables, accepted at International Conference on Pattern Recognition 2024

- **What's New**: 이번 연구는 Mild Cognitive Impairment (MCI)의 다양한 형식들 간의 융합된 네트워크 분석을 수행하여 뇌 연결성의 변화 및 발견된 topological 변화의 평가를 제안합니다.

- **Technical Details**: 이 연구에서는 두 개의 서로 다른 집단에서 fMRI 시간을 수집하여 3D 벡터의 시퀀스로 변환하고, Betti 지표에 대한 지속성 다이어그램을 계산하며, Wasserstein 거리 메트릭을 사용하여 topological 특성의 차이를 정량화합니다. ROI 특정 주제 상호작용과 주제 특정 ROI 간의 상호작용을 모두 분석하였습니다.

- **Performance Highlights**: 새로운 딥러닝 모델을 통해 ADNI 데이터세트에서 최대 95%의 분류 정확도를 달성하였고, 우리 내부 데이터세트에서는 85%의 정확도를 얻었습니다. MCI 하위 유형의 구분에서도 분류 정확도가 각각 76.5%, 91.1%, 80%에 달했습니다.



### {\mu}gat: Improving Single-Page Document Parsing by Providing Multi-Page Contex (https://arxiv.org/abs/2408.15646)
Comments:
          Accepted at ECCV Workshop "AI4DH: Artificial Intelligence for Digital Humanities"

- **What's New**: 새로운 연구에서는 Regesta Pontificum Romanum이라는 대규모 교황 기록 컬렉션에 초점을 맞추고 있습니다. 기존의 Document Parsing 기술은 단일 페이지 문서에만 초점을 맞추었으나, 본 연구는 다중 페이지 문서에서의 정보를 효과적으로 처리하기 위한 새로운 접근법을 제안합니다.

- **Technical Details**: 이 논문에서 제안하는 μgat(μ multi-page Nougat)은 최근 제안된 Document parsing Nougat 아키텍처의 확장으로, 현재 페이지를 분석하면서 이전 페이지와 다음 페이지의 요소를 동시에 처리할 수 있도록 설계되었습니다. 이는 스캔된 문서에서 유용한 정보를 추출하는 데 도움을 줄 것으로 기대됩니다.

- **Performance Highlights**: 실험 결과, 제안된 접근법은 Regesta Pontificum Romanorum의 경우에서도 우수한 성능을 보였으며, 기존 모델이 단일 페이지 문서에만 국한되었던 한계를 극복할 수 있는 가능성을 제시합니다.



### RIDE: Boosting 3D Object Detection for LiDAR Point Clouds via Rotation-Invariant Analysis (https://arxiv.org/abs/2408.15643)
- **What's New**: 이 논문에서는 LiDAR 포인트 클라우드를 기반으로 하는 3D 객체 탐지를 위한 새로운 접근법 'RIDE'를 제안합니다. 이 방법은 회전에 강한 특징을 설계하고 이를 기존의 탐지기에 효과적으로 통합해 성능을 향상시키고 회전 강인성을 높이는 것을 목표로 합니다.

- **Technical Details**: RIDE는 두 가지 유형의 특징을 추출하는 바이-피쳐 추출기(bi-feature extractor)를 설계합니다: 하나는 회전에 민감하지만 기하학적 정보를 잘 보존하는 객체 인식 피쳐(object-aware features, OAFs)이고, 다른 하나는 회전 불변 특성(rotation-invariant features, RIFs)입니다. 이 두 가지 특징은 서로 보완하여 임의의 회전에 강한 3D 제안을 디코드할 수 있게 합니다. RIDE는 기존의 3D 탐지기와 쉽게 통합 가능하며, 여러 Bi-SA(layer)를 적층하여 기하학적 정보가 풍부하면서도 회전 강인성을 갖춘 특징을 추출합니다.

- **Performance Highlights**: KITTI와 nuScenes 데이터셋에서의 실험 결과, RIDE가 장착된 탐지기는 각각 +5.6%의 mAP(mean average precision) 향상과 53%의 회전 강인성 향상을 보여주었고, nuScenes에서는 +5.1% mAP 향상과 28%의 회전 강인성 개선을 달성했습니다.



### Can SAR improve RSVQA performance? (https://arxiv.org/abs/2408.15642)
Comments:
          6 pages, 4 figures

- **What's New**: 본 연구는 기존의 Remote Sensing Visual Question Answering (RSVQA) 방법에 Synthetic Aperture Radar (SAR) 이미지를 결합하여 성능을 향상시키는 방법을 제안합니다. 일반적으로 Optical 데이터만 사용되었던 RSVQA 분야에서 SAR 데이터를 포함한 연구가 진행된 것입니다.

- **Technical Details**: 연구는 SAR 단독의 분류 결과를 분석하고, SAR과 Optical 데이터를 병합하는 방법론을 모색하며, 최종적으로 SAR 이미지와 Optical 이미지의 조합이 RSVQA 성능에 미치는 영향을 검토합니다. 이를 위해 Multi-label Classification과 Transformer 기반 모델을 활용하여 질문에 대한 자연어 응답을 예측하는 방법을 사용합니다.

- **Performance Highlights**: SAR 모드를 추가함으로써 RSVQA 성능이 향상되었으며, 더 균형 잡힌 데이터셋이 필요하다는 결론을 도출했습니다. 연구 결과는 Optical 이미지 단독 사용 시 대비 SAR 및 Optical 데이터를 결합했을 때 더욱 향상된 성능을 보였습니다.



### MMDRFuse: Distilled Mini-Model with Dynamic Refresh for Multi-Modality Image Fusion (https://arxiv.org/abs/2408.15641)
Comments:
          10 pages, 8 figures, accpeted by ACM International Conference on Multimedia 2024(Oral)

- **What's New**: 최근 Multi-Modality Image Fusion (MMIF) 분야에서 주목받는 새로운 접근법인 Distilled Mini-Model with Dynamic Refresh (MMDRFuse)를 제안하였습니다. 이 모델은 경량화된 구조를 갖추고 있으며, 113개의 학습 가능한 파라미터로 구성되어 있어 효율적인 이미지 융합을 지원합니다.

- **Technical Details**: MMDRFuse는 소스 이미지로부터의 정보를 최대화하기 위해 다각적인 손실 함수를 사용하며, 외부 공간 특징의 일관성을 강조하여 소프트 감독(supervision)을 제공합니다. 또한, 동적 새로 고침 전략을 통해 역사적 파라미터와 현재 감독을 효과적으로 협력시켜 훈련 과정을 최적화합니다.

- **Performance Highlights**: MMDRFuse는 여러 공공 데이터셋을 통해 효율성과 복잡성 면에서 우수한 성능을 보여주었으며, 다중 이미지 융합 작업과 보행자 탐지 다운스트림 응용 분야에서도 뛰어난 결과를 보였습니다.



### Transfer Learning from Simulated to Real Scenes for Monocular 3D Object Detection (https://arxiv.org/abs/2408.15637)
Comments:
          18 pages. Accepted for ECVA European Conference on Computer Vision 2024 (ECCV'24)

- **What's New**: 이 논문은 동적인 도로 상황에서 모노큘러 이미지로부터 3D 객체를 정확하게 탐지하기 위한 두 단계 훈련 전략을 제안합니다. 첫 번째 단계에서는 대규모 합성 데이터셋 RoadSense3D에서 모델을 훈련시키고, 두 번째 단계에서는 다양한 실제 데이터셋을 조합하여 모델을 세밀하게 조정합니다.

- **Technical Details**: Cube R-CNN 모델을 사용하여 전이 학습(transfer learning) 실험을 진행하며, 고도(pitch)와 롤(roll) 정보도 훈련 및 테스트 단계에 포함시킵니다. 논문에서 제안하는 방법은 다수의 도시에서 수집된 실제 데이터셋을 사용하여 모델의 일반화 능력을 임상적으로 향상시키는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: 전이 학습을 통해 TUM Traffic A9 데이터셋에서 평균 정밀도(mean average precision)가 0.26에서 12.76으로, DAIR-V2X-I 데이터셋에서 2.09에서 6.60으로 증가하는 놀라운 성능 향상을 보여줍니다.



### CSAD: Unsupervised Component Segmentation for Logical Anomaly Detection (https://arxiv.org/abs/2408.15628)
- **What's New**: 본 논문에서는 기존의 이상 탐지 방법에 비해 수작업 주석 없이 경량화된 세분화 네트워크를 훈련시키기 위해 기초 모델을 활용한 비지도 세분화 기법을 개발하였습니다. 이로 인해 더 낮은 대기 시간과 더 높은 처리량을 달성하였으며, MVTec LOCO AD 데이터셋에서 95.3%의 AUROC(Area Under Receiver Operating Characteristic) 성능을 기록했습니다. 이는 기존의 SOTA(State-of-the-Art) 방법보다 우수한 성과입니다.

- **Technical Details**: 우리는 기초 모델들을 활용하여 객체 구성 요소에 대한 의미론적 유사 레이블을 생성하고, 이를 기반으로 세분화 네트워크를 반지도 학습(semisupervised learning) 환경에서 훈련시킵니다. 두 가지 세분화 모드, 즉 미세 조정(fine-grained) 모드와 조정된(coarse-grained) 모드를 사용하여 세분화 과정을 수행하며, Patch Histogram 모듈과 LGST(Local-Global Student-Teacher) 모듈을 통합해 기능을 향상시킵니다.

- **Performance Highlights**: 우리의 방법은 기존의 방법들에 비해 우수한 성능과 더 낮은 대기 시간을 제공하며, 특히 논리 및 구조적 이상 탐지에서 최첨단 성능을 달성했습니다. 또한, 세분화 품질과 처리 효율성을 모두 강화하여 산업 이미지에서의 적용 가능성을 높였습니다.



### Can Visual Language Models Replace OCR-Based Visual Question Answering Pipelines in Production? A Case Study in Reta (https://arxiv.org/abs/2408.15626)
- **What's New**: 이 논문은 Visual Question Answering (VQA) 및 Optical Character Recognition (OCR) 작업에 대한 다양한 Vision Language Models (VLMs)의 성능과 한계를 분석합니다. 기존의 다단계 처리 파이프라인 대신 미리 훈련된 단일 단계 VLMs를 사용해 VQA 파이프라인을 대체할 수 있는지에 대한 질문을 다루고 있습니다.

- **Technical Details**: 연구는 Retail-786k 데이터셋을 사용하여 VLM이 광고 제품에 대한 상세한 질문에 답할 수 있는 능력을 조사합니다. 두 개의 상업 모델, GPT-4V와 GPT-4o, 그리고 네 개의 오픈 소스 모델인 InternVL, LLaVA 1.5, LLaVA-NeXT 및 CogAgent를 포함하여 총 400개의 프롬프트 쿼리를 대상으로 실험을 진행했습니다. 실험은 Microsoft Azure를 통해 실행되었으며, 각 모델의 예측은 수동으로 평가되었습니다.

- **Performance Highlights**: 결과에 따르면, 오픈 소스 모델과 상업 모델 간의 큰 성능 차이는 없었습니다. 그러나 VLM의 성능은 작업에 따라 크게 다르며, 대부분의 모델이 제품 브랜드와 가격에 대한 질문에 대해 높은 정확도로 답변할 수 있지만, 특정 제품 이름이나 할인 정보의 식별에서는 실패하는 경향이 나타났습니다. 이는 VLM이 세부적 분류 작업을 해결하는 데 어려움이 있음을 나타냅니다.



### Geometry-guided Feature Learning and Fusion for Indoor Scene Reconstruction (https://arxiv.org/abs/2408.15608)
Comments:
          Accepted by ICCV2023

- **What's New**: 본 논문에서는 3D 장면 재구성을 위한 새로운 기하학적 통합 메커니즘을 제안합니다. 기존 방법이 특징 수준에서만 기하학 정보를 포함한 반면, 본 방법은 특징 학습, 특징 융합, 네트워크 감독의 세 가지 수준에서 3D 기하학을 통합합니다.

- **Technical Details**: 첫 번째로, 기하학 유도 특징 학습(G2FL)을 통해 시각 의존 정보를 포함하는 기하학적 우선 정보를 인코딩합니다. 두 번째로, 기하학 유도 적응형 특징 융합(G2AFF)을 이용하여 여러 뷰에 대해 가중치를 생성합니다. 마지막으로, 2D와 3D 노멀의 일관성을 고려하는 일관된 3D 노멀 손실(C3NL)을 설계하여 지역적인 제약을 추가합니다.

- **Performance Highlights**: ScanNet 데이터셋에서 대규모 실험을 수행한 결과, 기하학적 통합 메커니즘을 적용한 볼륨 기반 방법이 최신 방법보다 정량적 및 정성적 성능에서 우수함을 보여줍니다. 또한, 이 방법은 7-Scenes 및 TUM RGB-D 데이터셋에서 좋은 일반화 성능을 보였습니다.



### Hierarchical Visual Categories Modeling: A Joint Representation Learning and Density Estimation Framework for Out-of-Distribution Detection (https://arxiv.org/abs/2408.15580)
Comments:
          Accepted by ICCV2023

- **What's New**: 이 논문에서는 시각 인식 모델을 위한 out-of-distribution (OOD) 데이터 감지를 위한 새로운 방법론인 hierarchical visual category modeling을 제안합니다. 이 방법은 joint representation learning과 statistical modeling을 통해 in-distribution 데이터와 OOD 데이터를 구분하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 우리의 접근 방식은 각 시각 범주에 대해 Gaussian mixture model(GMM)을 학습하고, Mahalanobis 기반 메트릭을 집계해 in-distribution score function을 설계하는 것입니다. 우리는 OOD 데이터에 대한 일반화 능력을 해치지 않기 위해 어떤 보조 OOD 데이터를 훈련 샘플로 사용하지 않습니다. 기존의 데이터 세트를 여러 하위 그룹으로 나눔으로써 결정 경계를 간소화합니다.

- **Performance Highlights**: 제안된 방법은 ImageNet-1k 데이터를 포함한 7개의 인기 있는 벤치마크에서 실험을 통해 명확하게 기존의 최첨단 알고리즘을 초월하는 성능을 보입니다. 또한, 시각적 표현이 전통적인 방법으로 학습된 특성과 비교했을 때 경쟁력 있는 성능을 나타내어 OOD 샘플 감지의 효율성을 개선하고 있습니다.



### Temporal Attention for Cross-View Sequential Image Localization (https://arxiv.org/abs/2408.15569)
Comments:
          Accepted to IROS 2024

- **What's New**: 이 논문은 기존의 이미지 검색 방식에서 벗어나, 단일 위성 이미지 패치 내에서의 스트리트 뷰 이미지의 미세하고 순차적인 로컬라이제이션(불착 위치 식별) 방법을 제안합니다. 이는 Temporal Attention Module (TAM)을 장착한 모델을 통해 컨텍스트 정보를 이용하여 정확도를 획기적으로 향상시킵니다.

- **Technical Details**: 논문에서는 Cross-View Sequential Image Localization이라는 새로운 작업을 소개하며, 이는 주어진 위성 이미지 패치 내에서 순차적인 지상 뷰 이미지의 위치를 예측하는 것을 포함합니다. 현재 상태와 이전 상태에서 정보를 통합하는 Temporal Attention Module (TAM)을 도입하여 모델의 성능을 향상시킵니다. KITTI-CVL 데이터셋을 순차적인 이미지 세트로 변형하여 모델의 일반화 능력을 입증하였습니다.

- **Performance Highlights**: CVIS 데이터셋에서 평균 및 중앙 로컬라이제이션 오차를 상당히 줄여 현재의 최고 성능을 기록하고 있으며, 교차 뷰 순차 이미지 로컬라이제이션에서 평균 거리 오차를 75.3% 감소시켰습니다.



### TagOOD: A Novel Approach to Out-of-Distribution Detection via Vision-Language Representations and Class Center Learning (https://arxiv.org/abs/2408.15566)
Comments:
          Accepted by ACMMM2024

- **What's New**: 본 논문은 Vision-Language Representation을 활용하여 OOD(Out-of-Distribution) 샘플 탐지를 위한 새로운 접근 방식인 TagOOD를 제안합니다. TagOOD는 전체 이미지에서 객체 특성을 레이블 없이 분리하여 객체의 의미론적 분석에 집중함으로써 OOD 탐지 성능을 개선합니다.

- **Technical Details**: TagOOD는 사전 학습된 태깅 모델을 활용하여 다수의 의미론적 태그를 생성하고, 이미지를 태그에 따라 분할하여 IND(In-Distribution) 데이터의 진정한 의미 영역을 학습합니다. 이어서, 추출된 객체 특성을 기반으로 가벼운 네트워크에서 학습된 클래스 중심점을 생성하고, 코사인 유사도를 통해 OOD 점수를 계산하여 OOD 샘플을 탐지합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋에 대한 실험을 통해 TagOOD가 기존의 OOD 탐지 방법들보다 월등한 성능을 보임을 입증하였으며, 객체 수준의 클래스 중심점 생성과 이미지 특성 분리가 OOD 탐지 성능 향상에 중요한 역할을 한다는 점을 강조합니다.



### Generalization Capabilities of Neural Cellular Automata for Medical Image Segmentation: A Robust and Lightweight Approach (https://arxiv.org/abs/2408.15557)
- **What's New**: 본 논문에서는 U-Net 아키텍처의 세 배 작은 모델(즉, x1000)을 사용하여 의료 영상 분할에서의 성능을 분석합니다. 특히 Neural Cellular Automata(NCA)를 활용하여 일반화 성능을 향상시키는 데 초점을 맞추고 있습니다.

- **Technical Details**: U-Net 모델 대비 약 1000배 적은 파라미터를 사용하는 NCA 아키텍처를 제안하며, 이는 반복적인 과정(iterative process)을 통해 더 넓은 수용 영역(receptive field)을 확보할 수 있습니다. 실험은 Retinal Optical Coherence Tomography (RETOUCH)와 Multi-Disease, Multi-View 이미지 세그멘테이션 작업을 포함하며, 두 데이터셋 모두에서 성능을 분석했습니다.

- **Performance Highlights**: U-Net은 인도메인(in-domain) 데이터에서 약간 더 높은 성능을 보였으나, 아웃오브도메인(out-of-domain) 데이터에서 NCA가 더 적은 성능 저하를 보였습니다. 이는 일반화가 요구되는 이미지 분할 작업에서 NCA의 사용을 고려해야 함을 시사합니다.



### Divide, Conquer and Combine: A Training-Free Framework for High-Resolution Image Perception in Multimodal Large Language Models (https://arxiv.org/abs/2408.15556)
- **What's New**: 이번 논문에서는 MLLM(다중모드 대형 언어 모델)의 HR(고해상도) 이미지 인식 능력을 평가하기 위해 HR-Bench라는 새로운 벤치마크를 소개합니다. 이는 4K 및 8K 해상도의 이미지를 대상으로 하여 기존의 2K 벤치마크 한계를 극복합니다.

- **Technical Details**: HR-Bench는 두 가지 버전(8K 및 4K)으로 구성되어 있으며, HR 이미지를 세분화된 패치로 나누고, 각 패치에 대해 정확한 텍스트 설명을 생성하여 MLLM이 HR 이미지를 보다 효과적으로 이해하도록 돕는 DC$^2$라는 교육 없는 프레임워크를 제안합니다. 이 프레임워크는 Divide(나누기), Conquer(정복하기), Combine(결합하기)라는 세 단계로 진행됩니다.

- **Performance Highlights**: SOTA MLLM의 HR-Bench에서의 정확도는 63%로, 인간의 87% 정확도에 비해 현저히 낮았습니다. DC$^2$ 접근법을 사용했을 때 HR-Bench와 일반 다중모드 벤치마크에서 각각 +6%, +8%의 유의미한 성능 향상이 있었습니다.



### ConsistencyTrack: A Robust Multi-Object Tracker with a Generation Strategy of Consistency Mod (https://arxiv.org/abs/2408.15548)
Comments:
          arXiv admin note: text overlap with arXiv:2308.09905 by other authors

- **What's New**: 본 연구는 비디오 시퀀스에서 여러 대상을 탐지하고 각 대상에 고유한 ID를 할당하는 작업인 Multi-object tracking(MOT)을 위한 새로운 프레임워크인 ConsistencyTrack을 제안합니다. ConsistencyTrack은 샘플링 및 추적을 위한 혁신적인 접근 방식을 도입하여, 노이즈 저항력이 향상되었습니다.

- **Technical Details**: ConsistencyTrack은 물체 탐지 및 추적을 결합한 Joint Detection and Tracking(JDT) 프레임워크로, 퍼지된 경계 상자를 기반으로 denoising diffusion 과정으로 이를 정의합니다. 모델은 인접한 두 프레임 사이의 객체 상자를 해당 진실 박스에서 무작위 분포로 변환한 후, 이 과정을 반대로 되풀이하여 학습합니다. 또한, ODE(Ordinary Differential Equation) 프레임워크를 활용하여 높은 효율성을 갖춘 샘플링을 가능하게 하였습니다.

- **Performance Highlights**: MOT17 및 DanceTrack 데이터셋을 통한 실험 결과, ConsistencyTrack은 다른 비교 모델을 초과하는 성능을 보였으며, 특히 DiffusionTrack보다 추론 속도 및 다른 성능 지표에서 우수한 결과를 나타냈습니다.



### Kangaroo: A Powerful Video-Language Model Supporting Long-context Video Inpu (https://arxiv.org/abs/2408.15542)
- **What's New**: 이번 논문에서는 비디오 데이터 처리에 중점을 둔 강력한 비디오 멀티모달 모델인 Kangaroo를 소개합니다. 현재의 LLMs에서 비디오 데이터를 효과적으로 처리하는 데 있어 어려움을 해결하기 위해 고품질 주석이 있는 대규모 데이터셋을 구축하고, 점진적으로 입력 프레임 수를 늘리는 커리큘럼 트레이닝 전략을 설계하였습니다.

- **Technical Details**: Kangaroo는 총 8B parameters로 비디오 이해를 위한 다양한 벤치마크에서 최첨단 성능을 달성합니다. 데이터 큐레이션 시스템을 통해 영상과 언어의 사전학습 및 지침 튜닝을 위한 고품질 데이터셋을 구축하였고, 최대 64와 160의 입력 프레임 수를 통해 긴 비디오를 효과적으로 처리할 수 있도록 하였습니다.

- **Performance Highlights**: Kangaroo는 다양한 비디오 이해 벤치마크에서 우수한 성능을 보였으며, 특히 긴 비디오에 특화된 벤치마크에서 10B 이상의 파라미터를 가진 다른 모델들을 초월하는 성과를 냈습니다.



### Ray-Distance Volume Rendering for Neural Scene Reconstruction (https://arxiv.org/abs/2408.15524)
Comments:
          Accepted by ECCV2024

- **What's New**: 이 논문은 실내 장면 재구성을 위한 새로운 접근 방식을 제안합니다. 기존의 Signed Distance Function (SDF) 대신 Signed Ray Distance Function (SRDF)를 밀도 함수로 매개화하여, 카메라 레이와 관련된 표면만을 고려한 밀도를 제공합니다.

- **Technical Details**: SRDF는 카메라 레이에서의 표면까지의 최단 거리를 측정하여 주위 물체의 영향을 제거합니다. 본 연구는 SRDF와 SDF 간의 일관성을 보장하기 위한 SRDF-SDF consistency loss를 도입하고, 물리적 가시성을 기반으로 한 자가 지도 학습(Self-supervised learning) 가시성 작업을 통해 3D 기하 구조의 예측을 향상시킵니다.

- **Performance Highlights**: 이 방법은 합성 데이터와 실제 데이터에서 실내 장면 재구성과 뷰 합성(view synthesis) 성능을 개선하였으며, SRDF를 통한 밀도 모델링으로 인해 실제 관측과 더 잘 일치하는 재구성을 생성합니다.



### A Simple Baseline with Single-encoder for Referring Image Segmentation (https://arxiv.org/abs/2408.15521)
Comments:
          ArXiv pre-print

- **What's New**: 이번 논문에서는 Referring Image Segmentation (RIS)를 위한 새로운 단일 인코더 기반 방법인 Shared-RIS를 제안합니다. 기존의 이중 인코더 구조에 비해 단일 인코더인 BEiT-3를 사용하여 데이터 효율성을 극대화하고, 비주얼 픽셀과 텍스트 간의 밀접한 상호작용을 통해 성능 향상을 이루었습니다.

- **Technical Details**: Shared-RIS는 세 가지 주요 구성 요소로 이루어져 있으며, 각각은 Shared Vision-Language Encoder, Shared FPN, Shared Mask Decoder 입니다. 모든 컴포넌트에서 shared self-attention 기술을 활용하여 비주얼과 텍스트 간의 dense fusion을 층별로 수행합니다. 이는 RIS 태스크에서 기대되는 pixel-word level alignment을 지원하며, lightweight한 decoder 모듈 덕분에 모델의 효율성을 높입니다.

- **Performance Highlights**: 제안된 단일 인코더 프레임워크는 RIS 벤치마크 데이터셋에서 기존 이중 인코더 기반 방법보다 뛰어난 성능을 보이며, 높은 효율성과 낮은 계산 비용을 유지합니다. 특히, shared self-attention을 활용하여 다중 모달 간의 정밀한 상호작용을 가능하게 하였습니다.



### Depth-Weighted Detection of Behaviours of Risk in People with Dementia using Cameras (https://arxiv.org/abs/2408.15519)
- **What's New**: 이번 연구에서는 치매 환자의 위험 행동 탐지를 위해 새로운 depth-weighted loss function을 제안하여, 카메라 가까이와 먼 거리의 사건들에 동등한 중요성을 부여하고, 허위 경보를 줄이는 방법을 제시했습니다.

- **Technical Details**: 본 연구에서는 특수 치매 유닛에서 수집된 9명의 치매 환자 데이터를 사용하여, 3개의 카메라를 통해 수집된 데이터를 기반으로 맞춤형 convolutional autoencoder를 훈련했습니다. 제안된 방법은 다음과 같은 주요 구성 요소를 포함합니다: pixel 깊이를 사용한 손실 함수, 비정상적인 활동을 활용한 threshold 결정, 그리고 개인 및 성별에 따른 행동 탐지 성능 분석.

- **Performance Highlights**: 제안된 방법은 AUC(Area Under the Curve) 지표에서 카메라별로 각각 0.852, 0.81, 0.768을 달성하여, 치매 환자들의 위험 행동 탐지에서 상당히 우수한 성능을 보였습니다.



### Continual-learning-based framework for structural damage recognition (https://arxiv.org/abs/2408.15513)
Comments:
          18 pages, 12 figures

- **What's New**: 이번 연구에서는 기존의 convolutional neural network (CNN) 기반의 손상 인식 방식들이 지닌 한계를 극복하기 위해, ResNet-34 구조에 'learning without forgetting' 방식을 통합한 지속적 학습 기반 손상 인식 모델(Continual Learning-based Damage Recognition Model, CLDRM)을 제안합니다.

- **Technical Details**: CLDRM은 Reinforced Concrete (RC) 구조 및 관련 구조 구성 요소의 손상을 인식하기 위해 설계되었습니다. 주된 접근 방식은 지속적 학습 과정을 통해 이전 학습된 작업의 정확도를 유지하면서 새로운 작업을 효과적으로 배우도록 하는 것입니다. 이 모델은 4개의 인식 작업을 위한 세 가지 실험을 수행하여 그 가능성과 효과성을 검증했습니다.

- **Performance Highlights**: 연구 결과, CLDRM은 4개의 지속적 학습 작업에서 예측 시간과 데이터 저장을 각각 약 75% 감소시켰으며, 손상 인식 및 분류에서 높은 정확도를 달성했습니다. 특히, CLDRM은 학습 작업 수가 증가함에 따라 이전에 학습한 작업들에 대한 감소가 적었음을 보여주었습니다.



### RoboSense: Large-scale Dataset and Benchmark for Multi-sensor Low-speed Autonomous Driving (https://arxiv.org/abs/2408.15503)
- **What's New**: 본 연구에서는 자율주행 기술의 발전을 위해 로봇 차량의 근거리 환경 인지를 위한 대규모 다중 모드 데이터셋 RoboSense를 구축하였습니다. 이는 독창적으로 다양한 환경에서 수집된 3D 객체에 대한 인식 및 추적 연구에 기여할 것입니다.

- **Technical Details**: RoboSense 데이터셋은 Camera, LiDAR, 및 Fisheye 센서를 포함하여 총 133K 개의 동기화된 데이터 프레임과 1.4M 개의 3D 바운딩 박스 및 식별자가 포함되어 있습니다. 6개의 주요 장면 클래스에서 216K 개의 궤적이 형성되었습니다.

- **Performance Highlights**: RoboSense는 이전의 KITTI 및 nuScenes 데이터셋보다 270배 및 18배 더 많은 근거리 장애물 주석을 제공합니다. 이는 자율주행 모델이 저속 환경에서 다양한 장애물과 복잡한 조건을 극복하는 데 중요한 역할을 할 것입니다.



### NAS-BNN: Neural Architecture Search for Binary Neural Networks (https://arxiv.org/abs/2408.15484)
Comments:
          23 pages

- **What's New**: 본 논문에서는 고효율의 이진 신경망(Binary Neural Networks, BNNs) 설계를 위한 새로운 신경망 아키텍처 검색 방법(NAS-BNN)을 제안합니다. 이는 기존의 BNN 설계 방법과 비교해 성능 향상을 위해 구축된 검색 공간과 훈련 전략을 포함하고 있습니다.

- **Technical Details**: NAS-BNN은 BNN의 특징을 반영하여 신중히 설계된 검색 공간을 기반으로 세 가지 훈련 전략을 도입하여 성능을 개선합니다. 이 과정에서 기존 MobileNetV1을 기반으로 하여 심층적으로 정보를 유지하고 비효율적인 서브넷을 제거하는 비가역(ND) 제약 조건을 적용합니다. 또한, 공유 가중치의 최적화를 위한 Bi-Transformation과 채널별 가중치 정규화 기법을 사용하여 정보 용량을 극대화합니다.

- **Performance Highlights**: NAS-BNN은 ImageNet에서 57M OPs로 68.20%의 top-1 정확성을 달성하였고, MS COCO 데이터셋에서 370M OPs로 31.6% mAP를 기록하여 이전 BNNs보다 우수한 성능을 보여주었습니다.



### Dynamic Reconstruction from Neuromorphic Data (https://arxiv.org/abs/2408.15465)
- **What's New**: 이번 연구에서는 전통적인 이미지 처리 방식 대신, neuromorphic event data(신경모방 이벤트 데이터)에서 직접적으로 이미지를 재구성하는 최적화 기반 접근 방식을 제안합니다. 이 방법은 초기 밝기 정보(Initial luminosity)에 대한 사전 지식 없이도 가능하다는 점에서 혁신적입니다.

- **Technical Details**: 신경모방 카메라는 각 화소에서 발생하는 빛의 변화만을 비동기적으로 기록합니다. 이 연구는 이러한 event data를 시간적 변화 방정식(ordinary differential equations, ODEs)으로 모델링하여 시간에 따른 밝기 변화를 최적화 문제로 해결합니다. 각 화소의 초기 조건은 기존 방법에서처럼 다른 정보에 의존하지 않고, 시스템 방정식을 통해 포착합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 실제 데이터에서 성공적으로 동적 장면을 재구성하여 높은 정확도와 효율성을 보여주었습니다. 이는 향후 신경모방 센서 데이터를 활용한 실제 환경에서의 응용 가능성을 열어주는 성과입니다.



### Hand1000: Generating Realistic Hands from Text with Only 1,000 Images (https://arxiv.org/abs/2408.15461)
Comments:
          Project page this https URL

- **What's New**: 이 논문에서는 인간 손의 해부학적으로 정확한 이미지를 생성하는 데 어려움이 있는 기존 Text-to-Image 생성 모델의 한계를 극복하기 위해 새로운접근법인 Hand1000을 제안합니다. 이 방법은 단 1,000개의 훈련 샘플을 사용하여 목표 제스처에 대한 현실적인 손 이미지를 생성할 수 있도록 설계되었습니다.

- **Technical Details**: Hand1000은 세 단계로 구성되며 첫 번째 단계에서는 사전 훈련된 손 제스처 인식 모델을 활용하여 손의 해부학적 이해를 향상시키고, 두 번째 단계에서는 텍스트 임베딩을 최적화하여 생성된 손 이미지와 텍스트 설명 간의 정렬을 개선합니다. 마지막으로, 최적화된 임베딩을 사용하여 Stable Diffusion 모델을 미세 조정하여 현실적인 손 이미지를 생성합니다.

- **Performance Highlights**: Hand1000은 기존 모델들에 비해 해부학적으로 정확한 손 이미지를 생성하는 데 있어 현저한 성능 향상을 보여주며, 얼굴, 의상, 색상 등 텍스트의 다른 세부 사항도 충실히 표현합니다.



### Fine-grained length controllable video captioning with ordinal embeddings (https://arxiv.org/abs/2408.15447)
- **What's New**: 이 논문에서는 생성된 캡션의 길이를 조절할 수 있는 비디오 캡셔닝(video captioning) 방법을 제안합니다. 이전 연구들은 길이 제어(level control)에서 몇 가지 수준만 표현할 수 있었으나, 본 연구에서는 좀 더 세부적인 길이 제어를 위한 두 가지 방법을 제시합니다.

- **Technical Details**: 전통적인 임베딩 방법은 원-핫 벡터(one-hot vector)와 임베딩 행렬(embedding matrix)을 이용한 선형 방식입니다. 본 연구에서는 비트 임베딩(bit embedding)과 서열 임베딩(ordinal embedding)을 사용하여 멀티 핫 벡터(multi-hot vectors)로 길이를 표현합니다. 이렇게 생성된 길이 표현은 비선형 MLP를 통해 길이 임베딩(length embedding)으로 변환됩니다.

- **Performance Highlights**: ActivityNet Captions와 Spoken Moments in Time을 사용한 실험에서, 제안된 방법이 생성된 캡션의 길이를 효과적으로 조절함을 보여주었습니다. ICA(Independent Component Analysis) 분석 결과, 길이와 의미가 별개로 학습된다는 것을 입증하여 임베딩 방법의 효과성을 시사합니다.



### HEAD: A Bandwidth-Efficient Cooperative Perception Approach for Heterogeneous Connected and Autonomous Vehicles (https://arxiv.org/abs/2408.15428)
Comments:
          Accepted by ECCV 2024 Workshop

- **What's New**: 이번 연구에서 제안하는 HEAD는 다양한 센서 모달리티를 갖춘 차량 간의 협업 인식을 지원하며, 기존 Late Fusion 기술보다 우수한 인식 성능을 제공하면서도 훨씬 적은 대역폭을 요구합니다.

- **Technical Details**: HEAD 방법은 3D 객체 탐지 네트워크에서 분류 헤드(classification head)와 회귀 헤드(regression head)로부터 특징을 융합하는 기법입니다. 수치적으로 작은 특징 크기에 초점을 맞추고, self-attention 메커니즘과 보완적인 특징 융합 층(complementary feature fusion layer)을 설계하여 데이터 융합 효율성을 극대화했습니다.

- **Performance Highlights**: HEAD는 V2V4Real 및 OPV2V 데이터 세트에서 포괄적인 평가를 통해 기존의 최첨단 중간 특징 융합 방법과 동등한 성능을 보이면서도 대역폭 요구를 현저히 줄일 수 있는 효율적인 융합 방법으로 입증되었습니다.



### Multi-Feature Aggregation in Diffusion Models for Enhanced Face Super-Resolution (https://arxiv.org/abs/2408.15386)
Comments:
          Accepted for presentation at the Conference on Graphics, Patterns and Images (SIBGRAPI) 2024

- **What's New**: 본 연구에서는 저해상도 이미지와 여러 저품질 이미지에서 추출한 얼굴 특성을 결합하여 고해상도 이미지를 생성하는 새로운 알고리즘을 개발했습니다. 기존의 방법과 달리 우리의 접근 방식은 얼굴 속성을 제공하지 않고도 얼굴 특징을 복원할 수 있습니다.

- **Technical Details**: 우리는 Stochastic Differential Equations (SDEs)에 기반한 확산 모델(diffusion models)을 활용하여 Face Recognition을 위한 Feature Aggregation Super-Resolution (FASR) 알고리즘을 제안합니다. 이 방법은 저해상도 이미지와 여러 이미지에서 추출된 얼굴 정보를 결합하여 더 나은 결과를 만들어냅니다.

- **Performance Highlights**: FFHQ 데이터셋을 활용하여 학습한 결과, CelebA 및 Quis-Campi 데이터셋에서 얼굴 인식과 검증 메트릭에서 최고 수준의 성능을 달성했습니다. 특히 1:1 검증 프로토콜에서 Area Under the Curve (AUC)와 1:N 식별 프로토콜의 정확도에서 월등한 성과를 보였습니다.



### CycleGAN with Better Cycles (https://arxiv.org/abs/2408.15374)
Comments:
          Technical Report 2018

- **What's New**: 본 논문에서는 CycleGAN의 cycle consistency loss에 대한 문제점을 식별하고, 이를 해결하기 위해 세 가지 간단한 수정을 제안합니다. 이 방법들은 더욱 현실적인 이미지를 생성하면서도 불필요한 아티팩트를 줄이도록 돕습니다.

- **Technical Details**: CycleGAN은 두 개의 제너레이터(Generator)와 두 개의 판별기(Discriminator)를 사용하는 이미지-투-이미지 번역(framework) 모델입니다. 본 연구에서 제안하는 수정 사항은 L1 손실을 추가하여 제너레이터가 이미지의 일반적인 구조를 회복하도록 요구합니다. 또한, 훈련 초기에는 cycle consistency loss의 가중치를 감소시키면서, 생성된 이미지의 품질에 따라 cycle consistency loss의 가중치를 동적으로 조절하도록 합니다.

- **Performance Highlights**: 수정된 cycle consistency 접근 방식은 훈련 과정에서 더욱 안정적인 결과를 보여주며, 초기 단계의 경우 생성된 이미지의 색상 또는 구조적 변화를 지나치게 억제하지 않으면서 수렴합니다. 이로 인해, CycleGAN은 더 적은 아티팩트로 더 나은 번역 결과를 달성했습니다.



### Handling Geometric Domain Shifts in Semantic Segmentation of Surgical RGB and Hyperspectral Images (https://arxiv.org/abs/2408.15373)
Comments:
          Silvia Seidlitz and Jan Sellner contributed equally

- **What's New**: 이 논문은 기존의 성능 높은 semantic segmentation 모델이 기하학적 도메인 변화(geometric domain shifts)에 직면했을 때의 영향을 분석하고, 'Organ Transplantation'이라는 새로운 데이터 증강 기법을 제안한다.

- **Technical Details**: 제안된 방법은 33마리의 돼지에서 수집된 600개의 RGB 및 hyperspectral imaging (HSI) 큐브를 사용하여 19개 클래스로 주석 처리된 6개의 OOD 데이터셋에서 검증되었다. 기하학적 OOD 데이터에서 SOA 모델의 성능은 RGB 데이터에서 46%, HSI 데이터에서 45%로 감소하였다.

- **Performance Highlights**: 조직의 기하학적 도메인 변화에 대응하는 데 효과적인 'Organ Transplantation' 증강 기법을 통해 RGB 데이터에서 최대 67%, HSI 데이터에서 90%의 성능 향상이 이루어졌으며, 실제 OOD 테스트 데이터에서의 in-distribution 성능 수준에 도달하였다.



### 3D Photon Counting CT Image Super-Resolution Using Conditional Diffusion Mod (https://arxiv.org/abs/2408.15283)
Comments:
          17th International Meeting on Fully 3D Image Reconstruction in Radiology and Nuclear Medicine, Stony Brook, NY, USA, 2023 [arXiv:2310.16846]

- **What's New**: 이 연구는 Denoising Diffusion Probabilistic Models (DDPM)을 사용하여 Photon Counting CT (PCCT) 이미지 해상도를 개선하고자 합니다. 최근 DDPM이 여러 컴퓨터 비전 작업에서 우수한 성능을 보여주었지만, 높은 차원에서 PCCT 슈퍼 해상도에 적용된 예는 없습니다.

- **Technical Details**: DDPM을 조건부 샘플링 방식으로 훈련하기 위해, CatSim을 활용하여 고해상도 CT 스캔에서 현실적인 저해상도 PCCT 이미지를 시뮬레이션합니다. 특히, 3D 작업을 효율적인 2D DDPM으로 분해하고, 모든 3차원에 대한 결과를 결합하여 최종 3D 예측을 수행하는 것을 설계하였습니다.

- **Performance Highlights**: 실험 결과, 본 연구의 DDPM이 고주파 구조를 복원하는 데 있어 기준 모델보다 향상된 결과를 나타내었으며, 현실적인 시뮬레이션과 DDPM 기반 프레임워크가 PCCT 해상도를 개선하는 데 유망함을 시사합니다.



### A Survey of Deep Learning for Group-level Emotion Recognition (https://arxiv.org/abs/2408.15276)
Comments:
          16 pages, 2 figures

- **What's New**: 이 논문은 깊이 있는 심층 학습(Deep Learning) 기법을 이용한 집단 수준의 감정 인식(Group-level Emotion Recognition, GER)에 대한 포괄적인 리뷰를 제공합니다. 기존의 리뷰와는 달리, 깊이 학습 아키텍처에 중점을 두고 현재 상황과 기술적 도전 과제를 설명하며, 집단 감정을 다루는 새로운 분류 체계를 제안합니다.

- **Technical Details**: 이 논문은 이미지 및 비디오 기반의 집단 감정 데이터베이스를 소개하고, 최근 개발된 심층 학습 방법과 이와 관련된 기술적 도전 과제를 논의합니다. 또한, 감정 인식을 위한 데이터 추출의 세 단계를 설명하며, 손으로 설계된 특성(descriptor) 및 신경망(neural networks)을 활용한 얼굴 특징 추출에 대해 조사합니다.

- **Performance Highlights**: 이 논문은 GIS(Ground Image Survey) 분야에서의 최신 기술 비교를 제공하고, 과거 10년간의 최고 성능(methods)의 성능 특성을 분석합니다. 또한, 향후 연구 방안과 응용을 위한 귀중한 통찰을 제공합니다.



### SkillMimic: Learning Reusable Basketball Skills from Demonstrations (https://arxiv.org/abs/2408.15270)
- **What's New**: 본 논문에서는 SkillMimic이라는 데이터를 기반으로 한 새로운 접근 방식을 제안하여 다양한 농구 기술을 습득하는 방법을 탐구합니다. SkillMimic은 인간과 공의 동작을 동시에 모방하여 실시간으로 복잡한 농구 기술을 학습할 수 있는 방식입니다.

- **Technical Details**: SkillMimic은 동일한 하이퍼파라미터를 사용하여 다중 기술을 학습하고, 데이터를 통해 기술의 다양성과 일반화를 향상시키는 통합된 구성 방식을 채택합니다. 두 개의 농구 데이터셋(BallPlay-V와 BallPlay-M)을 사용하여 농구 기술을 학습하고 평가하며, 이를 통해 다양한 드리블, 레이업 및 슈팅 기술을 효과적으로 학습합니다. 또한, Contact Graph Reward (CGR) 구성을 통해 정밀한 접촉 모방을 가능하게 하는 간단하고 일반적인 접촉 모델링 방법을 제안합니다.

- **Performance Highlights**: SkillMimic은 이전 방법들에 비해 농구 기술과 복잡한 작업 학습에서 간단함과 효율성의 측면에서 중요한 이점을 제공합니다. 다양한 농구 기술을 통합된 구성으로 학습하며, 레이업 득점과 같은 복잡한 농구 작업을 수행할 수 있습니다.



### S4DL: Shift-sensitive Spatial-Spectral Disentangling Learning for Hyperspectral Image Unsupervised Domain Adaptation (https://arxiv.org/abs/2408.15263)
- **What's New**: 이 논문에서는 하이퍼스펙트럼 이미지(HSI) 분류를 위한 새로운 방법론인 Shift-Sensitive Spatial-Spectral Disentangling Learning (S4DL)을 제안합니다. S4DL은 도메인 불변 특성과 도메인 특정 특성을 효과적으로 분리하여 다양한 데이터셋에서 성능을 향상시키고자 합니다.

- **Technical Details**: S4DL은 gradient-guided spatial-spectral decomposition 기법을 통해 각각의 채널에서 도메인 정보를 정량화하고, tailored masks를 생성해 도메인 불변 및 특정 채널을 분리하는 방식으로 작동합니다. 또한, shift-sensitive adaptive monitor가 도메인 간 변동을 지속적으로 모니터링하여 모델의 정렬 전략을 동적으로 조정합니다. Reversible feature extractor (RFE)는 저수준 특징도 함께 보존하여 도메인 정보를 유지합니다.

- **Performance Highlights**: 여러 개의 교차 장면 HSI 데이터셋에서 수행된 실험 결과, S4DL은 기존의 최신 UDA 방법들보다 뛰어난 성능을 보였으며, 이를 통해 다양한 도메인 간의 전이 가능성을 향상시켰습니다.



### Transformer-based Neuro-Animator for Qualitative Simulation of Soft Body Movemen (https://arxiv.org/abs/2408.15258)
Comments:
          12 pages, 3 figures

- **What's New**: 이 논문은 최근의 transformer 아키텍처를 활용하여 인공 신경망(ANN)을 통해 인간의 직관적인 물리적 사건의 시뮬레이션 능력을 연구합니다. 특히, 바람에 흔들리는 깃발의 움직임을 시뮬레이션하는 방법에 대해 논의합니다.

- **Technical Details**: visial transformer 모델은 소프트 바디 물체인 깃발의 동작을 

t-n ... t 시각에서의 정보를 바탕으로 t+1 시각의 동작을 예측하도록 훈련됩니다. 이 모델은 2D 플래그의 정점 움직임을 기반으로 한 감독 학습을 통해 훈련되며, 바람의 힘과 중력에 의해 영향을 받는 3D 공간의 깃발 움직임을 포착합니다.

- **Performance Highlights**: 결과적으로, visual transformer 기반 아키텍처는 소프트 바디 물체의 움직임에 대한 시간적 임베딩을 성공적으로 학습하고, 다양한 바람 세기에 따른 깃발의 흔들림을 적절하게 재현하는 시뮬레이션을 생성하는 데 성공했습니다.



### vFusedSeg3D: 3rd Place Solution for 2024 Waymo Open Dataset Challenge in Semantic Segmentation (https://arxiv.org/abs/2408.15254)
- **What's New**: VFusedSeg3D는 VisionRD 팀이 개발한 혁신적인 다중 모달 융합 시스템으로, 카메라 및 LiDAR 데이터를 결합하여 3D 인식의 정확성을 크게 향상시킵니다. 이 시스템은 카메라 이미지의 풍부한 의미(content)와 LiDAR의 정확한 깊이(depth) 센서를 활용하여 환경을 포괄적으로 이해합니다.

- **Technical Details**: VFusedSeg3D는 LiDAR 포인트 클라우드와 이미지의 기하학적 및 의미적 특징(feature)을 결합합니다. 이중형(feature extraction) 아키텍처를 사용하여 DLA34를 이미지 백본으로 활용하고, PTv3을 LiDAR 쪽 핵심 추출 backbone으로 사용합니다. feature fusion을 위한 모듈은 Geometry-based Feature Fusion Module (GFFM)과 Semantic Feature Fusion Module (SFFM)을 사용하여 정밀하고 구체적인 데이터 해석을 수행합니다.

- **Performance Highlights**: VFusedSeg3D는 검증 세트(validation set)에서 72.46%의 최첨단(mean Intersection over Union; mIoU) 정확도를 달성하여, 이전 70.51%에서 크게 향상된 성능을 기록했습니다. 이로 인해 3D 분할(segmentation) 정확성에 대한 새로운 벤치마크를 제시하였습니다.



### TrajFM: A Vehicle Trajectory Foundation Model for Region and Task Transferability (https://arxiv.org/abs/2408.15251)
- **What's New**: TrajFM이라는 차량 궤적 기반 모델을 제안하여 지역 및 작업 이식성을 동시에 개선합니다. 이를 통해 모델은 재학습 없이 다양한 지역과 작업 간 이동할 수 있습니다.

- **Technical Details**: TrajFM은 STRFormer를 주요 학습 모델로 채택하여 차량 궤적의 공간적(spatial), 시간적(temporal), POI(Points of Interest) 모달리티를 통합합니다. 이 모델은 지역 간의 POI 배열 변화에 효과적으로 대응하며, 공간적 특징을 처리하기 위해 학습 가능한 spatio-temporal Rotary position embedding 모듈을 포함하고 있습니다. 작업 이식성을 높이기 위해 궤적 마스킹 및 복원 기법을 도입하여 다양한 작업의 생성 과정을 통합합니다.

- **Performance Highlights**: TrajFM은 두 개의 실제 차량 궤적 데이터 세트를 대상으로 한 실험에서 그 효과성을 입증했습니다. 특정 설정에서 모델이 얼마나 유효하게 지역 및 작업 이식성을 달성하는지를 보여주었습니다.



### Pedestrian Motion Prediction Using Transformer-based Behavior Clustering and Data-Driven Reachability Analysis (https://arxiv.org/abs/2408.15250)
- **What's New**: 이 논문에서는 클러스터링된 역사적 궤적 데이터를 기반으로 미래 보행자 상태를 예측하기 위한 Transformer(변형기) 기반 프레임워크를 제안합니다.

- **Technical Details**: 본 연구에서는 Transformer 인코더와 계층 밀도 기반 클러스터링을 결합하여 보행자 행동의 다양성을 자동으로 식별하고 데이터 기반 도달 가능성 분석을 수행합니다. 이 프레임워크는 궤적 클러스터링을 통해 보행자의 행동 모드를 자동으로 식별하고, 이를 이용하여 더욱 안전하고 효율적인 예측을 가능하게 합니다.

- **Performance Highlights**: 실제 보행자 데이터 세트에서 훈련 및 평가를 수행하여 본 접근 방식이 보행자 움직임 예측의 효과성을 입증했습니다. 이 방법은 데이터 기반 도달 가능성 분석을 통해 보행자 행동 클러스터를 통합하여 보다 안전하면서도 보수적이지 않은 예측을 가능하게 합니다.



### Multi-Slice Spatial Transcriptomics Data Integration Analysis with STG3N (https://arxiv.org/abs/2408.15246)
- **What's New**: 본 논문에서는 Spatially Resolved Transcriptomics (SRT) 데이터의 다중 슬라이스 분석에서 발생하는 배치 효과를 완화하기 위한 새로운 방법론인 STG3Net을 제안했습니다. 이를 통해 기존 기법과 비교해 더 우수한 성능을 달성했습니다.

- **Technical Details**: STG3Net은 masked graph convolutional autoencoders를 백본 모듈로 사용하며, generative adversarial learning을 통합하여 다중 슬라이스 공간 영역 식별 및 배치 보정을 효과적으로 수행합니다. Global Nearest Neighbor (G2N) 방식으로 슬라이스 간의 대표적인 anchor pairs를 선택하여 배치 효과를 저감합니다.

- **Performance Highlights**: STG3Net은 세 가지 서로 다른 SRT 데이터 세트에서 F1LISI 메트릭을 포함한 정확도 및 일관성을 고려하여 평가한 결과, 기존 방법들과 비교하여 가장 뛰어난 성능을 보였으며, 멀티 슬라이스 간의 생물학적 변동성과 연결성을 유지했습니다.



### An Edge AI System Based on FPGA Platform for Railway Fault Detection (https://arxiv.org/abs/2408.15245)
Comments:
          Accepted at the 2024 IEEE 13th Global Conference on Consumer Electronics (GCCE 2024)

- **What's New**: 본 연구는 FPGA(최대 재구성 게이트 배열) 기반의 철도 점검 시스템을 도입하여 효율적이고 신뢰성 있는 레일 결함 탐지를 수행하는 새로운 엣지 AI 시스템을 제안합니다.

- **Technical Details**: 제안된 시스템은 카메라, FPGA 플랫폼, ESP8266 모듈, 그리고 그래픽 사용자 인터페이스가 장착된 컴퓨터로 구성되어 있으며, 카메라를 통해 레일 이미지를 캡처하고 CNN(합성곱 신경망)을 통해 실시간으로 결함을 탐지합니다. 경량화된 신경망을 사용하여 88.9%의 정확도를 달성하며, FPGA의 에너지 효율성은 GPU와 CPU 대비 각각 1.39배, 4.67배 더 높습니다.

- **Performance Highlights**: 실험 결과, 본 시스템은 GPU와 CPU에서의 동일한 네트워크 구현보다 뛰어난 에너지 효율성을 보여주었습니다. 또한, FPGA를 이용한 시스템은 데이터 처리를 신속하게 수행하며, 실시간 분석에 적합한 성능을 발휘합니다.



### CoGen: Learning from Feedback with Coupled Comprehension and Generation (https://arxiv.org/abs/2408.15992)
Comments:
          17 pages, 9 figures

- **What's New**: 이번 연구는 언어 이해(comprehension)와 생성(generation) 능력을 결합하여 사용자와의 상호작용에서 지속적으로 학습하는 방법을 제안합니다. 두 능력을 통합하기 위한 새로운 기술이 도입되었습니다.

- **Technical Details**: 본 연구는 두 플레이어 참조 게임(two-player reference games)에서 진행되며, 수천 번의 사용자 상호작용을 통해 다양한 모델을 배포합니다. 상호작용 피드백 신호(interaction feedback signals)를 이용하여 학습하는 동시에, 이해와 생성 능력을 결합하여 성능을 개선합니다.

- **Performance Highlights**: 이 연구의 결과, 이해-생성 결합이 절대적으로 최대 26% 향상된 성능을 보여주고, 비결합 시스템에 비해 최대 17% 높은 정확도를 기록했습니다. 이러한 결합은 시스템의 언어에도 상당한 질적 영향을 미쳐, 더 인간적인 언어 사용을 가능하게 했습니다.



### Generating Binary Species Range Maps (https://arxiv.org/abs/2408.15956)
- **What's New**: 이번 연구에서는 종 분포 모델(SDM)의 이진 범위 지도를 자동으로 생성하기 위한 최적의 임계값(Threshold)을 식별하는 다양한 접근 방식을 평가했습니다. 특히, 이 논문은 존재 데이터만을 사용하는 경우의 새로운 접근 방식을 제안하고, 기존의 진정한 결석 정보가 필요 없는 기술을 확장하였습니다.

- **Technical Details**: 연구는 서로 다른 이진 범위 추정 및 대규모 세밀한 시각 분류 작업에서 다양한 임계값 설정 기법을 비교하고, 존재 데이터만으로도 효율적인 이진 범위 지도를 생성할 수 있음을 입증합니다. 기존 기술과 비교해 더 나은 성능을 보이는 방법을 구현하였습니다.

- **Performance Highlights**: 결과적으로, 제안된 방법은 기존의 가상의 결석 정보를 사용하는 접근 방식보다 더 효율적이며, 다양한 종의 이진 범위 추정 및 이미지 분류 작업에서 우수한 성능을 보였습니다.



### Auxiliary Input in Training: Incorporating Catheter Features into Deep Learning Models for ECG-Free Dynamic Coronary Roadmapping (https://arxiv.org/abs/2408.15947)
Comments:
          MICCAI 2024

- **What's New**: 본 논문에서는 Auxiliary Input in Training (AIT)이라는 새로운 방법을 소개하여, 심장 위상 매칭(cardiac phase matching) 및 카테터 팁 추적(catheter tip tracking) 작업의 성능을 향상시킵니다. AIT는 카테터 마스크를 보조 입력으로 사용하여 심층 학습(deep learning) 모델의 학습 효율을 높이고, 동시에 baseline 방법들을 초월하는 성과를 보였습니다.

- **Technical Details**: AIT는 심장 위상 매칭 및 카테터 팁 추적에 사용되는 심층 학습 모델에 카테터 마스크를 보조 입력으로 추가하여, 모델이 더 나은 표현을 학습하도록 돕습니다. 이 과정에서 카테터 마스크는 추가적인 입력 채널로 연결되며, 학습 과정 중 점진적으로 제로 행렬로 변환됩니다. 이는 모델이 보조 정보에 대한 의존도를 줄이고, 본래의 입력에 대해 더 잘 일반화할 수 있게 합니다.

- **Performance Highlights**: AIT 방법은 기존 방법들보다 성능이 개선되었으며, 심장 위상 매칭 및 카테터 팁 추적에서 모델의 정확도를 높이는 데 크게 기여했습니다. 이 연구는 심층 학습 모델이 카테터 정보를 효과적으로 활용하여 복잡한 의료 영상을 처리하는 데 도움을 줄 수 있음을 보여줍니다.



### Sigma Flows for Image and Data Labeling and Learning Structured Prediction (https://arxiv.org/abs/2408.15946)
Comments:
          51 pages

- **What's New**: 이 논문은 Riemannian manifold와 Euclidean 이미지 도메인에서 관찰된 데이터의 구조화된 레이블링을 예측하기 위한 sigma flow 모델을 소개합니다. 이 접근법은 Sochen, Kimmel, Malladi에 의해 도입된 Laplace-Beltrami 프레임워크를 이미지 노이즈 제거 및 향상과 결합하고, 저자들이 연구한 assignment flow 접근법을 통합합니다.

- **Technical Details**: sigma flow는 일반화된 조화 에너지의 Riemannian 기울기 흐름으로 나타나며, 이는 비선형 기하학적 PDE에 의해 제어됩니다. 이 PDE는 폐쇄된 Riemannian 도메인 매니폴드에서 통계적 매니폴드로의 조화적인 맵을 결정하며, 정보 기하학에서의 Fisher-Rao 메트릭으로 장비됩니다. sigma flow의 특정 요소는 도메인 매니폴드의 Riemannian 메트릭이 발전하는 상태에 따라 상호 의존한다는 점입니다.

- **Performance Highlights**: 개념 증명 실험은 sigma flow 모델의 표현력과 예측 성능을 입증합니다. transformer 네트워크 아키텍처와 sigma flow의 기하적 통합에 의해 생성된 네트워크 간의 구조적 유사성을 지적하여, 심층 학습(deep learning)과의 연결을 강조하며, 다른 과학적 기계 학습 분야에서의 구조적 예측을 위한 기하학적 설계 원칙의 활용을 자극할 수 있습니다.



### Gen-Swarms: Adapting Deep Generative Models to Swarms of Drones (https://arxiv.org/abs/2408.15899)
- **What's New**: Gen-Swarms는 딥 생성 모델과 반응형 내비게이션 알고리즘을 결합하여 드론 쇼의 자동 생성 방식을 혁신적으로 개선한 방법입니다. 기존의 3D 포인트 클라우드 생성 모델의 한계를 극복하고 드론의 동작을 유도하는 기능을 포함하여 부드러운 궤적과 충돌 방지를 실현합니다.

- **Technical Details**: 이 연구에서는 흐름 매칭(flow matching) 알고리즘을 3D 포인트 클라우드 생성에 최초로 적용하였습니다. 이를 통해, 특정 카테고리의 입력(예: 비행기)에 따라 드론들이 정확하고 다양한 3D 형태를 형성하는 안전하고 부드러운 궤적을 생성합니다.

- **Performance Highlights**: 실험 결과에 따르면, Gen-Swarms는 기존 기준선보다 안전하고 부드러운 드론 궤적을 제공하여 협업 예술 로봇 공학 분야에서 혁신적인 발전을 보여주었습니다. 이는 드론 쇼 생성을 위한 더욱 가능성 있는 솔루션임을 입증합니다.



### SpineMamba: Enhancing 3D Spinal Segmentation in Clinical Imaging through Residual Visual Mamba Layers and Shape Priors (https://arxiv.org/abs/2408.15887)
Comments:
          17 pages, 11 figures

- **What's New**: 이번 연구에서는 3D 임상 의학 이미지를 신뢰성 있게 분할할 수 있는 새로운 방법, SpineMamba를 소개합니다. 이는 척추 질환의 진단 및 치료에 필수적입니다.

- **Technical Details**: SpineMamba 모델은 Residual Visual Mamba Layer를 사용하여 3D 척추 데이터의 심층 의미 표현(deep semantic features)과 장거리 공간 의존성(long-range spatial dependencies)을 효과적으로 모델링합니다. 추가로 spinal shape prior module을 통해 구체적인 해부학적 정보를 추출합니다.

- **Performance Highlights**: SpineMamba는 두 개의 데이터셋에서 비교 실험 결과 기존의 최첨단 모델들보다 탁월한 성능을 보여줍니다. CT 데이터셋에서 평균 Dice 유사성 계수는 94.40에 이르고, MR 데이터셋에서는 86.95입니다. 특히 유명한 nnU-Net 대비 최대 2% 향상된 분할 성능을 기록하였습니다.



### Benchmarking foundation models as feature extractors for weakly-supervised computational pathology (https://arxiv.org/abs/2408.15823)
- **What's New**: 이번 연구에서는 13개의 환자 집단과 6,791명의 환자, 9,493개의 슬라이드를 대상으로 10개의 병리학 기초 모델을 평가했습니다. 이 연구는 외부 집단에서 이러한 기초 모델의 독립적 평가에 초점을 맞추었습니다.

- **Technical Details**: 연구에서는 약한 감독 (weakly-supervised) 작업에 대해 바이오마커, 형태학적 특성, 예후 결과를 관련하여 모델을 평가하였습니다. 비전-언어 기초 모델인 CONCH가 비전 전용 모델에 비해 42%의 작업에서 최고 성능을 기록하였습니다.

- **Performance Highlights**: 상보적 (complementary) 특성을 학습한 다양한 기초 모델들의 앙상블이 CONCH보다 66%의 작업에서 더 뛰어난 성과를 보였습니다. 데이터 다양성이 데이터 양보다 더 중요하다는 점도 발견되었습니다.



### Addressing the challenges of loop detection in agricultural environments (https://arxiv.org/abs/2408.15761)
- **What's New**: 해당 논문에서는 농업 환경에서의 루프 탐지에 특화된 새로운 방법을 제안합니다. 기존의 SLAM(동시 위치 추정 및 지도 작성) 시스템은 실내 및 도시 환경에서 뛰어난 성능을 보였으나, 개방된 농업 환경에서는 지속적인 위치 추정 및 일관된 매핑에 어려움을 겪고 있었습니다. 이를 해결하기 위해 로컬 특징 탐색(Local Feature Search)과 스테레오 기하학적 정제를 기반으로 한 루프 탐지 기술을 개발했습니다.

- **Technical Details**: 제안된 방법은 스테레오 비주얼 정보에서 시작하여 유사도 검색을 진행하고, 이후 시간적 및 기하학적 정제를 수행합니다. 이 과정에서 Bag-of-Words (BoW) 검색을 통해 대략적인 루프 후보를 확보하고, 추가적인 시간적으로 인접한 프레임들을 사용해 최적의 후보를 선정하며, 마지막으로 스테레오 이미지 간의 상대 변환을 추정하여 검증합니다.

- **Performance Highlights**: 제안한 방법은 평균 15cm의 오차로 좋은 루프 탐지를 지속적으로 달성했습니다. 실험을 통해 농업 환경에서의 시각적 로컬라이제이션 향상 가능성을 확인하였으며, 향후 논문은 이러한 방법론이 농업 환경에서의 자율 로봇의 위치 추정을 더욱 개선할 수 있는 방향을 제시합니다.



### G-Style: Stylized Gaussian Splatting (https://arxiv.org/abs/2408.15695)
- **What's New**: 이번 논문에서는 G-Style이라는 새로운 알고리즘을 소개합니다. 이 알고리즘은 Gaussian Splatting을 사용하여 이미지를 3D 장면으로 스타일을 전이하는 방식으로, 기존의 방법들이 가지는 스타일 일관성 문제를 해결하고자 합니다.

- **Technical Details**: G-Style은 세 가지 단계로 이루어집니다: 첫 번째 단계에서는 불필요한 Gaussian을 제거하고, 두 번째 단계에서는 이미지 스타일의 다양한 스케일을 보존하기 위해 설계된 손실 함수들을 결합합니다. 마지막으로, 스타일화 과정에서는 색상의 기울기를 추적하여 필요한 곳에서 Gaussian을 분할하여 세부 사항을 추가합니다.

- **Performance Highlights**: 실험 결과, G-Style은 몇 분 만에 고품질의 스타일화된 장면을 생성하며, 기존 방법들에 비해 품질과 렌더링 속도 모두에서 우수한 성능을 보였습니다.



### A quantitative model of takeover request time budget for conditionally automated driving (https://arxiv.org/abs/2408.15682)
Comments:
          Manuscript: 12 pages, 12 figures, 7 tables

- **What's New**: 자동화 주행 시스템(Automated Driving Systems, ADS)에 대한 최근 연구는 인간 운전자가 Takeover Request(차선을 넘는 요청)에 응답하여 주행을 재개하는 데 필요한 시간 예측에 대한 새로운 접근 방식을 제안합니다. 이 연구에서는 시각적 정보 지원의 유무에 따른 4초, 5초, 6초, 7초의 다양한 시간 예산을 포함한 주행 시나리오를 분석하여 이들 각각의 주행 상황에서 적절한 시간 예산을 제안합니다.

- **Technical Details**: 이 연구는 평균 측면 이탈 거리(average lateral displacement) 등의 성능 측정을 사용하여 3개의 주행 시나리오에서 고정된 시간 예산(7초)과 가변 시간 예산(6초, 5초, 4초)의 적합성을 조사하였습니다. 결과적으로, 7초의 시간 예산은 연구된 두 개의 시나리오에서 적합하다고 평가되었습니다. 연구 결과를 바탕으로, 단순 자극 반응 시간, 주행 경험, 특정 시나리오 요구 사항을 통합하기 위한 수학적 공식을 제안하였으며, 이는 시간 예산을 예측할 수 있게 해줍니다.

- **Performance Highlights**: 제안된 공식을 통해 운전자는 Takeover Maneuvers를 더욱 안전하게 수행할 수 있으며, 시각적 정보 지원이 포함될 경우 Takeover 시간(Takeover Time, TOT)이 증가하여 시간 예산이 더 늘어나는 경향이 있음을 보였습니다. 이 연구는 각 상황에 따라 운전자가 필요한 시간 예산을 조정하자는 취지를 가지고 있으며, 이를 통해 다양한 주행 시나리오에서의 안전성을 개선할 수 있을 것으로 기대됩니다.



### ES-PTAM: Event-based Stereo Parallel Tracking and Mapping (https://arxiv.org/abs/2408.15605)
Comments:
          17 pages, 7 figures, 4 tables, this https URL

- **What's New**: 이번 연구에서는 고속 움직임과 높은 동적 범위 조명 조건에서 모바일 로봇의 공간 인지를 위한 새로운 event-based stereo Visual Odometry(VO) 시스템을 제안합니다. 전통적인 카메라의 한계를 극복하기 위해 이벤트 카메라를 활용하며, 깊이 추정을 위한 correspondence-free mapping 모듈과 카메라 자세를 추정하는 tracking 모듈을 결합하였습니다.

- **Technical Details**: 제안된 시스템은 두 개의 주요 모듈인 tracking과 mapping 모듈로 구성되어 있으며, 각 모듈은 이벤트를 입력으로 받아 서로의 출력을 바탕으로 작동합니다. MC-EMVS(mapping 방법)와 edge alignment 기반의 tracker를 결합하여 높은 정확도의 깊이 맵을 생성하며, 속도와 효율성을 동시에 갖춘 시스템을 구현하였습니다.

- **Performance Highlights**: 풍부한 실제 데이터셋에서 평가한 결과, 제안된 시스템은 기존의 최신 기술들을 초월한 성능을 보였습니다. 예를 들어, RPG 데이터셋에서 45%, DSEC 데이터셋에서 61%, TUM-VIE 데이터셋에서 21%의 경로 오류 감소를 달성하였습니다.



### On the Benefits of Visual Stabilization for Frame- and Event-based Perception (https://arxiv.org/abs/2408.15602)
Comments:
          8 pages, 4 figures, 4 tables, this https URL

- **What's New**: 이 논문은 로봇 애플리케이션에서 시각 기반 인식 시스템의 성능을 향상시키기 위해 새로운 처리를 기반으로 한 카메라 회전 보정 접근 방법을 제안합니다. 이는 이벤트 및 프레임에서 카메라의 회전 운동을 상쇄하는 방법을 다룹니다. 특히, 기계식 안정 장치의 통합이 어렵거나 불가능한 경량 로봇을 위해 설계되었습니다.

- **Technical Details**: 제안된 방법은 카메라의 자세가 미리 알려졌다고 가정하며, 이를 통해 카메라의 선형 속도가 추정되는 데이터 파이프라인에 통합됩니다. 실험은 합성 데이터와 잘 알려진 이벤트 기반 비전 데이터셋을 사용하여 수행됩니다. 본 연구에서의 시각적 인식과 관련하여 두 가지 주요 작업인 카메라 에고-모션 추정과 초점 평면의 특징 추적이 포함됩니다.

- **Performance Highlights**: 실험 결과는 안정화가 특징 추적 및 카메라 에고-모션 추정 정확성을 각각 27.37% 및 34.82% 향상시킴을 보여줍니다. 또한, 카메라의 선형 속도를 계산하는 처리 시간을 최소 25% 줄이는 효과가 있음을 나타냅니다.



### Latent Relationship Mining of Glaucoma Biomarkers: a TRI-LSTM based Deep Learning (https://arxiv.org/abs/2408.15555)
Comments:
          9 pages, 4 images

- **What's New**: 최근 연구에서는 녹내장 분류 및 감지를 위한 딥러닝(deep learning) 방법이 많이 활용되고 있지만, 기존의 기계 학습(machine learning) 모델의 설명 가능성(explainability)이 큰 문제가 되고 있습니다. 본 연구에서는 인지 과학(cognitive science) 개념을 바탕으로 안과 의사들이 녹내장을 판단하는 방식을 연구하였습니다. 우리는 바이오마커(biomarker) 중심의 머신러닝 모델을 통해 계층적 의사결정 시스템을 제안하였습니다.

- **Technical Details**: TRI-LSTM(time series model)은 녹내장 바이오마커 간의 잠재적인 관계를 계산하고 탐색할 수 있는 최초의 모델로, 안과 의사의 진료 과정을 모방하여 운영됩니다. 기존의 연구들과는 달리 이 모델은 다양한 바이오마커의 잠재 관계를 탐구하여 해석 가능성을 크게 향상합니다. 이 연구에서 사용된 데이터는 텍사스 대학교 남서부 의과 대학(UTSW)에서 제공한 것으로, 고해상도 이미지 데이터를 획득할 수 있는 Intailght 장비를 사용했습니다.

- **Performance Highlights**: 실제 데이터셋을 기반으로 한 광범위한 실험을 통해 제안된 TRI-LSTM 모델의 효과성을 입증하였습니다. 이 모델은 다양한 바이오마커 간의 관계를 정량적으로 분석하여 조기 진단을 위한 강력한 도구로 자리잡을 수 있는 가능성을 보여줍니다.



### Avoiding Generative Model Writer's Block With Embedding Nudging (https://arxiv.org/abs/2408.15450)
- **What's New**: 본 연구는 이미지 생성 모델의 출력 제어를 통해 유용성을 개선하는 혁신적인 방법을 제안합니다. 특히, 생성 과정 중 발생할 수 있는 원치 않는 개념을 피하면서도 여전히 출력 생성을 가능하게 합니다.

- **Technical Details**: 본 논문에서는 Latent Diffusion Image Generative Models에 초점을 맞춰, 특정 이미지를 생성하는 것을 방지하면서도 유사한 이미지를 생성할 수 있는 방법을 제안합니다. 이 방법은 불필요한 이미지를 멀리하고 바람직한 스타일로 유도하는 방식으로, 잠재 공간(latent space)을 최적화하여 구현됩니다.

- **Performance Highlights**: 정성적 및 정량적 평가를 통해, 제안된 방법이 메모리제이션(memorization) 문제를 성공적으로 해결하며 기존 모델과 유사한 이미지 품질을 유지함을 입증하였습니다.



### Evaluating Pre-Training Bias on Severe Acute Respiratory Syndrome Datas (https://arxiv.org/abs/2408.15398)
Comments:
          short paper for eurovis, 5 pages

- **What's New**: 이 연구는 브라질의 지역별로 세 가지 선 훈련(bias metrics) 지표를 시각화하여 모델 훈련 전에 데이터의 잠재적 편향을 식별할 수 있는 방법을 제공한다.

- **Technical Details**: 연구에서는 OpenDataSUS의 중증급성호흡기증후군(SRAG) 데이터셋을 사용하여 세 가지 선 훈련(bias) 메트릭을 시각화한다. 이 메트릭은 Class Imbalance, Kullback-Leibler (KL) Divergence, Kolmogorov-Smirnov (KS)를 포함한다. 각 지역에 대해 Random Forest 모델을 훈련시키고, 성능을 평가하고 있다.

- **Performance Highlights**: 모델은 정확도와 F1-Score 성능 지표를 사용하여 성능을 평가하며, 각 지역에 대한 편향을 비교하고 있다. 이 연구의 결과는 데이터의 시각화 기법이 연구자들이 훈련 노력에 앞서 데이터에서의 편향을 식별하는 데 도움이 될 수 있음을 시사한다.



### Panoptic Perception for Autonomous Driving: A Survey (https://arxiv.org/abs/2408.15388)
- **What's New**: 이 논문은 자율주행 기술의 최신 발전 중 하나인 panoptic perception(판옵틱 인식)을 종합적으로 리뷰합니다. 기존의 다양한 인식 작업을 통합하여 차량 주변 환경에 대한 보다 깊이 있는 이해를 제공하는 새로운 접근법을 제시합니다.

- **Technical Details**: 이 논문에서는 panoptic perception의 정의와 이를 통해 수행할 수 있는 다양한 작업들(예: object detection, instance segmentation, semantic segmentation)을 설명합니다. 또한, multi-task network(다중 작업 네트워크)의 구조와 CNNs(합성곱 신경망), transformers(변압기) 및 혼합 모델을 포함한 다양한 아키텍처를 비교 분석합니다.

- **Performance Highlights**: panoptic perception은 기존 방법들에 비해 견고성, 정확성 및 효율성을 크게 개선하는 것으로 나타났습니다. 이는 멀티 태스크 접근 방식을 통해 각 인식 작업 간의 상관관계를 established함으로써 이루어진 것입니다. 또한, 이 연구는 panoptic perception 모델에 대한 포괄적인 자료를 제공하여 후속 연구가 이루어질 기반을 마련합니다.



### Optimizing Lung Cancer Detection in CT Imaging: A Wavelet Multi-Layer Perceptron (WMLP) Approach Enhanced by Dragonfly Algorithm (DA) (https://arxiv.org/abs/2408.15355)
- **What's New**: 본 연구는 CT 스캔 이미지를 통해 폐암을 분류하기 위한 최신 딥러닝 프레임워크를 소개합니다. Canny 엣지 탐지와 웨이블릿 변환을 포함한 이미지 전처리 전략을 통해 특징을 추출하고, Multi-Layer Perceptron(MLP)을 사용하여 분류를 수행합니다. 최적화 과정은 Dragonfly Algorithm(DA)을 통해 진행되어, 이 방법론은 99.82%의 훈련 및 테스트 정확도를 달성하였습니다.

- **Technical Details**: 제안된 접근법은 이미지 전처리, 특징 추출 및 분류를 포함합니다. Canny 엣지 탐지 및 웨이블릿 변환을 통해 이미지 품질을 개선하고, MLP를 통해 특징을 학습합니다. 특히 MLP의 하이퍼파라미터는 DA를 사용하여 최적화됩니다. 웨이블릿 변환(WT)과 MLP는 폐암 진단의 정확성을 높이는 데 매우 중요한 역할을 합니다.

- **Performance Highlights**: 본 연구의 결과는 MLP의 99.82%라는 높은 정확도로, 딥러닝이 폐암 진단에서의 가능성을 강하게 뒷받침하고 있습니다. 향후 연구는 데이터셋 확장, 고급 신경망 아키텍처 탐색 및 새로운 이미지 전처리 기술 통합에 중점을 두어 분류 정확성과 견고성을 더욱 향상시킬 것입니다.



### Parameter-Efficient Quantized Mixture-of-Experts Meets Vision-Language Instruction Tuning for Semiconductor Electron Micrograph Analysis (https://arxiv.org/abs/2408.15305)
Comments:
          Paper published at ICML 2024 Workshop on Foundation Models in the Wild

- **What's New**: 본 논문은 반도체 제조를 위한 소규모 비전-언어 어시스턴트인 sLAVA를 소개합니다. 이 시스템은 전자 현미경 이미지 분석에 중점을 두고 있으며, 데이터 부족 및 전문가 주석이 있는 고품질 데이터 확보 문제를 해결합니다.

- **Technical Details**: sLAVA는 teacher-student paradigm을 사용하여, GPT-4와 같은 기본 비전 언어 모델을 교사로 활용하여 지침을 따르는 다중 모드 데이터를 생성합니다. 이 모델은 커스터마이징된 학습을 통한 전자 현미경 이미지 분석 작업을 지원합니다. 이를 통해 기업들이 자사 데이터로 프레임워크를 세밀하게 조정할 수 있으며, 지적 재산권 보호 또한 가능합니다.

- **Performance Highlights**: 엄격한 실험 결과, sLAVA 프레임워크는 전통적인 방법들을 초월하고, 데이터 변화에 효과적으로 대응하며, 고처리량 스크리닝을 가능하게 함을 입증했습니다.



### NeR-VCP: A Video Content Protection Method Based on Implicit Neural Representation (https://arxiv.org/abs/2408.15281)
- **What's New**: 본 논문에서는 비디오 콘텐츠 보호를 위한 자동 암호화 기술을 제안합니다. 이 기술은 암시적 신경 표현(Implicit Neural Representation, INR)을 기반으로 하며, 수동으로 설계할 필요가 없습니다.

- **Technical Details**: Key-controllable module을 설계하여 암호화 및 복호화의 키 역할을 하며, 발신자가 수신자에게 이 모듈을 사전 배포합니다. 이후 발신자는 암시적 신경망을 이용하여 평문 비디오를 암호화하고, 합법적인 수신자는 배포된 모듈을 사용하여 암호화된 신경망을 복호화합니다.

- **Performance Highlights**: 실험 결과, 비주얼 표현 및 불법 사용자에 대한 인지 불가능성, 암호학적 관점에서의 보안성에 있어 우수한 성능을 보였습니다. 또한 데이터 전송량을 줄이기 위한 모델 압축 기법을 활용하여 데이터 보호를 더욱 효과적으로 수행했습니다.



### Automated Software Tool for Compressing Optical Images with Required Output Quality (https://arxiv.org/abs/2408.15275)
Comments:
          In Proceedings of XIIth intenational conference on CADSM, 2013, pp. 184 187

- **What's New**: 이 논문은 그레이스케일 이미지의 손실 압축(lossy compression)을 위한 자동화된 소프트웨어 도구를 소개합니다.

- **Technical Details**: 이 도구는 선택한 품질 메트릭(metric)에서 제공되는 다양한 코더(coders)를 사용하여 이미지를 압축할 수 있는 기능을 가지고 있으며, 사전 설정된 메트릭 값(preset metric value)을 제공합니다.

- **Performance Highlights**: 논문에서는 도구를 여러 실제 상황에 적용한 사례를 제시하여 그 효과를 보여줍니다.



### Civiverse: A Dataset for Analyzing User Engagement with Open-Source Text-to-Image Models (https://arxiv.org/abs/2408.15261)
- **What's New**: 본 연구는 TTI (Text-to-Image) AI 플랫폼 CivitAI를 분석하여 문화적 관점에서 오픈소스 TTI 프레임워크를 체계적으로 조사합니다. 특히, 사용자 의도와 행동을 파악하기 위해 Civiverse 프롬프트 데이터셋을 소개하며, 이는 수백만 개의 이미지와 관련 메타데이터를 포함합니다.

- **Technical Details**: 대규모 데이터셋 Civiverse 6M을 사용하여 텍스트 프롬프트의 의미적 특성을 분석하였습니다. 이 데이터셋은 사용자가 제공한 프롬프트를 기반으로 생성된 6,546,165개의 이미지 URL과 메타데이터로 구성되어 있습니다. 분석은 사용자와 TTI 모델 간의 상호작용 패턴에 초점을 맞추어 진행되었습니다.

- **Performance Highlights**: 연구 결과, 성적 콘텐츠 생성에 대한 주된 선호가 나타났으며, 의미적 내용의 동질화 경향이 확인되었습니다. 이러한 통찰은 그림을 생성하는 모델 내에서 여성 혐오와 해로운 고정관념이 지속될 가능성을 강조하며, 문화적 다양성이 감소하고 있음을 나타냅니다.



### Emotion Classification from Multi-Channel EEG Signals Using HiSTN: A Hierarchical Graph-based Spatial-Temporal Approach (https://arxiv.org/abs/2408.15255)
Comments:
          Draft

- **What's New**: 이번 연구는 다채널 전기뇌파(EEG) 데이터를 활용한 감정 분류를 위한 매개변수 효율적인 Hierarchical Spatial Temporal Network (HiSTN)를 소개합니다. 이 네트워크는 다양한 추상화 수준에서 구축된 그래프 계층을 통합하여, 더 나은 깊이 있는 특성 추출과 경량 설계를 가능하게 합니다.

- **Technical Details**: HiSTN 모델은 약 1,000개의 매개변수를 가지며, DREAMER 데이터셋의 5분류 과제를 기반으로 주제 의존 테스트에서 96.82%(Valence)와 95.62%(Arousal)의 평균 F1 점수를 달성합니다. 또한, 주제 비의존 설정에서 평균 F1 점수는 Valence 78.34%, Arousal 81.59%입니다. Sequential Top-2 Hit Rate (Seq2HR) 메트릭은 기존 One-Hot 레이블로 훈련한 모델에 비해 예측 품질의 균형을 크게 개선한 점을 강조합니다.

- **Performance Highlights**: 이 모델을 통해 주제 의존 과제에서는 50%, 주제 비의존 과제에서는 30% 이상의 예측 품질 개선이 이루어졌습니다. 다양한 사례 연구와 Ablation 연구를 통해 제안된 모델의 작동 방식과 해석 가능성을 더욱 명확히 하였습니다.



### Machine Learning for Methane Detection and Quantification from Space - A survey (https://arxiv.org/abs/2408.15122)
- **What's New**: 이번 연구에서는 단기 적외선(SWIR) 대역의 메탄(CH4) 감지 센서에 대한 기존 정보를 확장하고, 전통적인 방법과 기계 학습(Machine Learning) 접근법에 대한 최신 동향을 리뷰하였습니다. 특히, 메탄 플룸(segmentation) 및 발생률(emission rate) 추정에 대한 기계 학습 모델의 아키텍처와 데이터 사용에 대해서도 논의합니다.

- **Technical Details**: 기존 전통적 방법은 노동집약적 수작업 조정 과정에 의존하는 반면, ML 접근법은 CNN(Convolutional Neural Network) 기반의 U-net 및 Transformer 아키텍처를 사용하여 더 효율적이고 확장 가능한 감지를 제공합니다. 이 연구는 메탄 감지, 세분화 및 발생률 추정의 세 가지 주제를 다룹니다.

- **Performance Highlights**: 분석 결과, ML 모델이 전통적인 방법보다 성능이 뛰어난 것으로 나타났으며, 특히 대량의 스펙트럼 데이터에서 유용한 정보를 추출하여 보다 정확한 감지를 가능하게 합니다. 전통적인 방법에 비해 시간 효율성과 정확성을 제공하지만, 다양한 데이터와 평가 지표 간의 비교 문제 또한 논의되었습니다.



### CMTA: Cross-Modal Temporal Alignment for Event-guided Video Deblurring (https://arxiv.org/abs/2408.14930)
Comments:
          Accepted in ECCV2024

- **What's New**: 본 연구는 비디오 블러링(video deblurring) 문제를 해결하기 위해 마이크로초(micro-second) 시간 해상도를 가진 이벤트 카메라(event camera)를 활용하는 새로운 접근 방식을 제안합니다. 이를 통해 기존의 프레임 기반 방법의 한계인 심한 모션 블러 상황에서도 효과적으로 성능을 향상시킬 수 있습니다.

- **Technical Details**: 우리는 두 가지 모듈을 제안합니다: 1) Intra-frame feature enhancement는 단일 블러 프레임의 노출 시간 내에서 반복적으로 교차 모달리티 피처를 개선합니다. 2) Inter-frame temporal feature alignment는 특정 프레임에 대한 유용한 장기 시간 정보를 수집하고, 이벤트의 장점을 활용하여 선명한 특징들을 집계합니다. 이 연구에서는 새로운 EVRB 데이터셋을 사용하여 이벤트를 기반으로 한 비디오 디블러링 방법을 평가합니다.

- **Performance Highlights**: 제안된 방법은 합성 및 실제 비디오 블러링 데이터셋에서 기존의 최첨단(frame-based, event-based) 모션 디블러링 방법보다 뛰어난 성능을 보여주었습니다. 이 연구는 이벤트 데이터의 이점을 활용하여 높은 품질의 글로 매핑된 결과를 생성하는 새로운 방법론을 제공합니다.



### Splatt3R: Zero-shot Gaussian Splatting from Uncalibrated Image Pairs (https://arxiv.org/abs/2408.13912)
Comments:
          Our project page can be found at: https://splatt3r.active.vision/

- **What's New**: 이 논문에서는 Pose-free 방식으로 작동하는 Splatt3R라는 새로운 3D 재구성 및 새로운 뷰 합성 방법을 소개합니다. 이 방법은 불균형한 자연 이미지 쌍들로부터 3D Gaussian Splats를 예측할 수 있으며, 카메라 매개변수나 깊이 정보 없이도 작동합니다.

- **Technical Details**: Splatt3R는 MASt3R라는 기초 3D 재구성 방법을 기반으로 하여 3D 구조와 모양을 처리할 수 있도록 확장되었습니다. 이 모델은 3D 점 구름의 기하학적 손실을 최적화한 후 새로운 뷰 합성 목표를 최적화하여 지역 최소값(local minima) 문제를 회피합니다. 또한, 손실 마스킹 전략을 통해 Extrapolated viewpoints에서의 성능을 개선합니다.

- **Performance Highlights**: Splatt3R는 ScanNet++ 데이터셋에서 훈련되었으며, 비보정(un-calibrated) 이미지에 대한 일반화 성능이 뛰어나고, 512x512 해상도에서 4FPS로 씬을 재구성할 수 있습니다. 최종적인 스플랫(splats)은 실시간으로 렌더링할 수 있습니다.



New uploads on arXiv(cs.AI)

### WebPilot: A Versatile and Autonomous Multi-Agent System for Web Task Execution with Strategic Exploration (https://arxiv.org/abs/2408.15978)
- **What's New**: 이번 연구에서는 WebPilot이라고 불리는 자율 다중 에이전트 시스템을 제안합니다. 이 시스템은 Monte Carlo Tree Search (MCTS)를 기반으로 한 이중 최적화 전략을 통해 복잡한 웹 환경에서의 성능을 크게 향상시키는 데 초점을 맞추고 있습니다.

- **Technical Details**: WebPilot은 Global Optimization 단계와 Local Optimization 단계로 구성되어 있습니다. Global Optimization에서는 복잡한 작업을 관리 가능한 하위 작업으로 세분화하여 전체 계획을 생성하고 지속적으로 수정하여 탐색 과정을 집중화합니다. Local Optimization에서는 하위 작업을 수행하기 위해 복잡한 환경에 맞춰 조정된 MCTS를 사용하여 불확실성을 처리하고 불완전한 정보를 관리합니다.

- **Performance Highlights**: 실험 결과, WebPilot은 WebArena에서 SOTA 성능을 달성하며, 다른 동시 기반 트리 검색 방법에 비해 93%의 상대적인 성공률 증가를 기록했습니다. GPT-3.5를 사용할 때도 WebPilot은 29.1%의 성공률을 기록하며 최신 GPT-4 기반의 방법들과 경쟁력을 유지했습니다.



### Atari-GPT: Investigating the Capabilities of Multimodal Large Language Models as Low-Level Policies for Atari Games (https://arxiv.org/abs/2408.15950)
Comments:
          Currently under review

- **What's New**: 본 논문은 멀티모달 대형 언어 모델(LLM)을 로우 레벨 제어기로 활용하는 가능성을 탐구합니다. 특히 Atari 비디오 게임에서의 성능을 새로운 벤치마크로 제시하며, 이 모델들이 기존의 강화 학습(RL) 및 모방 학습(IL) 방법과 어떻게 차별화되는지를 분석합니다.

- **Technical Details**: 연구는 GPT-4V, GPT-4o, Gemini Flash, Claude 3 Haiku와 같은 최첨단 멀티모달 LLM을 사용해 Atari 비디오 게임에서 로우 레벨 정책으로 기능하는 능력을 평가합니다. In-Context Learning(ICL)을 통해 인간의 게임 플레이 예시를 이용해 모델의 맥락적 이해를 높이는 방안도 살펴봅니다.

- **Performance Highlights**: 모델의 성능은 게임 점수, 비주얼 이해, 공간적 추론 및 전략 수립 능력을 포함하여 여러 요소로 측정됩니다. 본 연구의 결과는 멀티모달 LLM들이 복잡한 비주얼 씬을 이해하고 상호작용하며 전략적인 반응을 생성하는데 효과적이라는 것을 보여줍니다. 또한, 이 모델들이 로우 레벨 제어기로서의 가능성을 탐색하며, 애트리 게임의 새로운 기준을 설정합니다.



### Persuasion Games using Large Language Models (https://arxiv.org/abs/2408.15879)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)이 인간과 유사한 텍스트를 이해하고 생성하는 데 탁월한 도구가 되고 있으며, 이러한 도구가 사용자에게 적합한 투자 계획, 신용 카드 및 보험 정책 선택에 도움을 준다는 점을 제시하고 있다.

- **Technical Details**: 주요 요점은 여러 에이전트가 협력하는 복잡한 다중 에이전트 프레임워크를 통해 사용자와의 설득적인 대화를 수행하며, 이 과정에서 정보 검색, 응답 분석, 설득 전략 개발 및 사실 확인을 수행하는 보조 에이전트들이 존재한다는 것이다.

- **Performance Highlights**: 실험을 통해 이 협력 방법론이 LLM의 설득력을 크게 향상시키는 것을 보여주었고, 사용자 저항을 지속적으로 분석해 감정 변화에 대한 판단 및 적절한 사실을 조합해 반박할 수 있는 전략을 개발하였다.



### LogicGame: Benchmarking Rule-Based Reasoning Abilities of Large Language Models (https://arxiv.org/abs/2408.15778)
- **What's New**: 이 논문에서는 LogicGame이라는 새로운 벤치마크를 소개하여 대형 언어 모델(LLMs)의 규칙 이해, 실행, 계획 능력을 종합적으로 평가하고자 합니다.

- **Technical Details**: LogicGame은 다양한 규칙 기반 게임으로 구성되어 있으며, 모델이 유지해야 할 규칙의 시리즈를 통해 문제를 해결해야 합니다. 각 게임 시나리오는 초기 상태와 함께 주어진 규칙을 이해하고 적용하여 최종 목표를 달성하는 것을 목표로 합니다. 평가 과정은 최종 결과뿐 아니라 모델이 거친 중간 단계도 고려하여 전체적인 성능을 평가합니다.

- **Performance Highlights**: LogicGame을 통해 다양한 LLM을 테스트한 결과, 모델들이 규칙 기반 논리 추론 능력에서 주목할 만한 한계를 보여주었습니다. 최고 성능의 LLM조차도 복잡한 추론 과제에서 약 20%의 전체 정확도 및 3단계 과제에서 10% 미만의 성과를 기록했습니다.



### Adaptive Traffic Signal Control Using Reinforcement Learning (https://arxiv.org/abs/2408.15751)
- **What's New**: 최근 도시 지역의 교통 수요가 증가함에 따라 교통 혼잡 문제가 심각해지고 있다. 이 논문은 Reinforcement Learning (RL)을 사용하여 기존의 교통 신호를 최적화하고 이를 통해 교통 혼잡을 해결하는 방안을 제시하고 있다.

- **Technical Details**: 논문에서는 상태를 큐 길이를 나타내는 스칼라로 정의하고 RL 에이전트가 이 단순화된 상태 표현으로부터 효과적으로 학습할 수 있음을 보여준다. 두 가지 RL 알고리즘, 즉 높은 교통량을 가진 방향의 교통 신호를 우선시하는 턴 기반 에이전트와 고정된 사이클에 따라 작동하며, 교통 조건에 따라 신호 주기를 조정하는 시간 기반 에이전트를 개발하였다.

- **Performance Highlights**: 시뮬레이션 결과에 따르면, 두 알고리즘 모두 기존의 교통 신호 제어 시스템보다 뛰어난 성능을 보여주었다. 네 가지 다양한 교통 시나리오에서 실시된 성능 평가에서 두 알고리즘은 효과적으로 교통 흐름을 개선하는 것으로 나타났다.



### Hierarchical Blockmodelling for Knowledge Graphs (https://arxiv.org/abs/2408.15649)
Comments:
          31 pages, 11 figures

- **What's New**: 본 논문에서는 지식 그래프에서의 계층적 엔터티 클러스터링(hierarchical entity clustering)을 위해 확률적 그래픽 모델(probabilistic graphical models), 특히 스토캐스틱 블록모델(stochastic blockmodels)의 사용을 조사합니다.

- **Technical Details**: 이 모델은 그래프를 확률 분포의 집합으로 분해하며, 이 분포의 매개변수를 추론(inference)한 후 랜덤 그래프(random graph)를 생성하기 위한 샘플링(sampling)을 할 수 있습니다. 비모수적(non-parametric) 설정에서 이는 계층 구조의 제약 없이 계층적 클러스터링(hierarchical clustering)의 유도를 가능하게 합니다. 특히, Nested Chinese Restaurant Process와 Stick Breaking Process의 통합을 통해 생성 모델을 제안하여 이를 위한 collapsed Gibbs 샘플링 방식을 도출합니다.

- **Performance Highlights**: 우리 모델은 합성(synthetic) 및 실제(real-world) 데이터셋에서 평가되었고 기준 모델들과 정량적으로 비교되었습니다. 또한 우리 모델은 소규모 환경에서 일관된 클러스터 계층을 유도할 수 있음을 발견하였습니다. 본 연구는 지식 그래프에 대한 스토캐스틱 블록모델의 규모 있는 적용을 위한 첫 단계로, 향후 더 확장 가능한 추론 방식(inference schemes)에 대한 가능성도 제시합니다.



### Trustworthy and Responsible AI for Human-Centric Autonomous Decision-Making Systems (https://arxiv.org/abs/2408.15550)
Comments:
          45 pages, 2 figures

- **What's New**: 이 논문은 인공지능(AI)의 신뢰성과 책임성을 위한 여러 원칙들을 정의하고, AI 모델의 공정성을 제고하며, AI 사용 시 발생할 수 있는 편향 문제를 해결하기 위한 방법들을 제시합니다.

- **Technical Details**: 논문에서는 Trustworthy AI, Responsible AI, Explainable AI의 원칙과 AI 거버넌스, 공정성 평가 지표, 편향 탐지 및 완화 기술에 대한 심층 논의를 진행합니다. AI 시스템의 투명성을 개선하고, AI가 인간의 의사결정에 미치는 영향을 줄이는 것을 목표로 합니다.

- **Performance Highlights**: AI에 대한 신뢰를 구축하기 위해 필요한 설명 가능성과 투명성을 강조하며, 다양한 산업 분야에서 책임감 있는 AI 모델의 적용을 촉진하기 위한 지침 및 연구 기회를 제안합니다.



### TrafficGamer: Reliable and Flexible Traffic Simulation for Safety-Critical Scenarios with Game-Theoretic Oracles (https://arxiv.org/abs/2408.15538)
- **What's New**: TrafficGamer는 멀티 에이전트 게임으로 도로 주행 현상을 이해하고 안전-critical 시나리오의 생성 및 평가를 가능하게 하는 새로운 시뮬레이션 시스템입니다.

- **Technical Details**: TrafficGamer는 게임 이론을 활용하여 여러 차량 간의 복잡한 상호작용을 모델링하며, 진화하는 정책 기반의 Coarse Correlated Equilibrium (CCE) 솔버 알고리즘을 통해 시뮬레이션의 정확성과 유연성을 보장합니다.

- **Performance Highlights**: TrafficGamer는 Argoverse 2 및 Waymo Open Motion Dataset 등 다양한 실제 교통 데이터를 활용하여 생성된 시나리오가 실제 데이터 분포와 통계적으로 일치함을 입증하며, 다양한 안전-critical 시나리오를 효과적으로 시뮬레이션할 수 있습니다.



### Towards Fully Autonomous Research Powered by LLMs: Case Study on Simulations (https://arxiv.org/abs/2408.15512)
Comments:
          For additional code and data, please visit our GitHub repository: this https URL

- **What's New**: 대형 언어 모델(LLM)의 발전으로 자동화된 과학 연구의 새로운 기회를 제시하고, 실험 설계에서 데이터 분석, 보고서 작성까지의 전 과정을 자동화할 수 있는 자율 시뮬레이션 에이전트(ASA)를 개발하였습니다.

- **Technical Details**: ASA는 LLM을 통해 API 통합을 실행하며, 연구 계획(RP)을 입력하면 AI가 전체 연구 과정을 수행합니다. 이러한 방식으로 AI가 완전 자율적으로 시뮬레이션 코드를 작성하고, 데이터를 처리하며, 과학 보고서를 작성합니다. 이 연구에서는 폴리머 체인 구성 시뮬레이션 문제를 사례로 사용하여 LLM의 성능을 평가했습니다.

- **Performance Highlights**: ASA-GPT-4o는 연구 임무를 거의 완벽하게 수행하였으며, 인간의 개입 없이 최대 20 사이클까지 반복 가능한 자동화 가능성을 보여주었습니다. 이 결과는 LLM이 복잡한 과학 연구를 자율적으로 관리할 수 있는 잠재력을 강조합니다.



### What Machine Learning Tells Us About the Mathematical Structure of Concepts (https://arxiv.org/abs/2408.15507)
Comments:
          25 pages, 3 figures

- **What's New**: 이 논문은 철학, 인지 과학, 기계 학습 분야에서 개념을 이해하는 다양한 접근 방식의 연결성을 분석하고 특히 이들의 수학적 특성에 초점을 맞추고 있습니다.

- **Technical Details**: 이 연구는 접근 방식을 Abstractionism, Similarity Approach, Functional Approach, Invariance Approach로 분류하며, 각 프레임워크가 개념을 모델링하는 데 제공하는 수학적 관점을 강조합니다.

- **Performance Highlights**: 이 논문은 철학 이론과 현대 기계 학습 모델 간의 연결 고리를 제시하며, 인간 인지와 인공지능 간의 복잡한 관계를 이해하는 데 기여하는 포괄적인 프레임워크를 제공합니다.



### Pathfinding with Lazy Successor Generation (https://arxiv.org/abs/2408.15443)
Comments:
          14 pages

- **What's New**: 이 논문에서는 단순한 구조로 보이지만, 많은 위치에서 비단순해지는 경로 탐색 문제에 대해 다룹니다. 특히, 제안하는 LaCAS* 알고리즘은 일괄적으로 후속 위치를 생성하지 않고, 검색이 진행됨에 따라 점진적으로 후속 위치를 생성하는 방식으로 설계되었습니다.

- **Technical Details**: 저자들은 k-d tree를 활용하여 k-nearest neighbors search 기법을 적용합니다. LaCAS*는 완전하고 anytime 알고리즘으로, 결국 최적 솔루션에 수렴합니다. 이 알고리즘은 connectivity 정보를 제공하는 oracle이 필요하며, 유클리드 거리(Euclidean distance)를 사용하여 두 위치 간의 거리 계산을 수행합니다.

- **Performance Highlights**: LaCAS* 알고리즘은 복잡한 경로 탐색 문제를 빠르게 해결하는 데 있어 기존의 방법들이 실패한 경우에도 효과적임을 보여 주었습니다. extensive evaluations 결과, LaCAS*는 빠른 해결 속도와 높은 효율성을 가지고 있음을 입증하였습니다.



### On Stateful Value Factorization in Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2408.15381)
Comments:
          22 pages, 9 figures, 4 tables

- **What's New**: 이번 논문은 멀티 에이전트 강화 학습에서의 가치 분해(value factorization) 방법의 이론을 분석하고, 이러한 이론과 실제 적용 간의 불일치 문제를 해결합니다. 특히, 역사 정보 대신 상태(state) 정보를 사용하는 것이 이론적으로 유효함을 제시하면서 새로운 알고리즘인 DuelMIX를 소개합니다.

- **Technical Details**: DuelMIX는 개별 에이전트(level)에서 특정 유틸리티 추정값을 학습하여 성능을 개선하는 새로운 분해 알고리즘입니다. 이 알고리즘은 상태 정보를 사용하더라도 역사 및 상태 값들을 결합하여 최적의 표현력을 달성하는 구조로 개발되었습니다. 연구에서는 Decentralized Partially Observable Markov Decision Processes (Dec-POMDPs) 모델을 사용하여 협력적 멀티 에이전트 작업을 포괄적으로 설명합니다.

- **Performance Highlights**: StarCraft II 미세 관리 및 Box Pushing 작업에 대한 실험 결과, DuelMIX는 이전의 분해 방법들보다 우수한 성능을 보였으며, 샘플 효율성(sample efficiency)에서도 큰 개선을 이루었습니다.



### What Is Required for Empathic AI? It Depends, and Why That Matters for AI Developers and Users (https://arxiv.org/abs/2408.15354)
Comments:
          To appear at the 7th AAAI/ACM Conference on AI, Ethics, and Society, 2024

- **What's New**: AI는 감정을 이해하고 반응하는 능력을 갖추어야 한다는 인식의 증가 속에서, 다양한 응용에 필요한 감정 인식 능력의 차이를 구분하는 프레임워크가 개발되었다. 이 논문에서는 의학 분야에서 세 가지 주요의료 empathic AI 사례를 제시하여 각 경우에 맞는 능력 세트를 논의한다.

- **Technical Details**: 이 논문에서는 인지적 공감(cognitive empathy), 정서적 공감(affective empathy), 동기적 공감(motivated empathy)이라는 세 가지 서로 관련되지만 구분 가능한 공감 현상을 다룬다. 저자들은 AI의 각 응용에 필요한 능력의 조합이 다르며, AI 개발자들이 이러한 차이를 인식하는 것이 중요하다고 강조한다.

- **Performance Highlights**: 의료 질의응답 AI, 노인 및 장애인을 도와주는 AI 보조기기, 환자 돌봄을 위한 AI 제공자 등 다양한 AI 응용이 소개되며, 각 적용 사례의 성공 여부는 공감의 정의에 따라 다를 수 있지만, 공통적으로 특정한 능력이 요구됨을 보여준다.



### Bi-Factorial Preference Optimization: Balancing Safety-Helpfulness in Language Models (https://arxiv.org/abs/2408.15313)
- **What's New**: 이번 연구에서는 Bi-Factorial Preference Optimization (BFPO)이라는 새로운 supervised learning 프레임워크를 제안합니다. 이는 안전성과 유용성을 동시에 고려한 LLM의 fine-tuning을 위한 단일 학습 목표로 변환하여, RLHF에서 발생할 수 있는 안전성과 유용성의 갈등 문제를 해결합니다.

- **Technical Details**: BFPO에서는 labeling function을 통해 안전성과 유용성의 전반적인 선호 순위를 포착하여, 두 가지 목표를 균형 있게 조절합니다. 이 방법은 기존의 RLHF기반 접근법에 비해 인간의 프롬프트나 주석이 필요 없고, 10% 미만의 계산 자원으로도 동일한 안전 수준을 달성할 수 있습니다.

- **Performance Highlights**: BFPO는 안전성과 유용성 모두에서 기존 방법보다 유의미하게 뛰어난 결과를 보였습니다. 특히, 공개 데이터셋을 사용하여 유용성을 유지하면서도 15%의 무해성을 향상시키고, 1.5K의 red teaming 데이터로 13% 향상시키는 성과를 달성했습니다.



### Eagle: Exploring The Design Space for Multimodal LLMs with Mixture of Encoders (https://arxiv.org/abs/2408.15998)
Comments:
          Github: this https URL, HuggingFace: this https URL

- **What's New**: 이번 연구는 여러 비전 인코더를 혼합하여 MLLMs의 성능을 향상시키기 위한 체계적인 탐색을 진행했습니다. 특히, 시각 정보 인코딩의 효과를 비교하고, 비전 전문가의 선택 및 통합 방식에 대한 심층 분석이 포함되어 있습니다.

- **Technical Details**: 비전 전문가의 혼합 설계를 통해 MLLMs의 성능을 높일 수 있는 원리를 발견했습니다. 주요 요소로는 Pre-Alignment를 통해 비전 인코더와 언어 토큰 간의 간극을 줄이는 방식을 도입했습니다. 이 연구는 다양한 비전 인코더의 성능을 벤치마킹하고, 여러 인코더의 효과적인 결합 방식을 규명했습니다.

- **Performance Highlights**: Eagle 모델은 주요 MLLM 벤치마크에서 다른 오픈 소스 모델들을 초월하는 성과를 달성했으며, OCR 및 문서 이해 작업에서 뚜렷한 이점이 있었습니다. 이 연구는 비전 중심의 MLLMs 설계에서 상당한 성과를 거두었음을 보였습니다.



### Mamba or Transformer for Time Series Forecasting? Mixture of Universals (MoU) Is All You Need (https://arxiv.org/abs/2408.15997)
Comments:
          Code at this https URL

- **What's New**: 이번 연구에서는 Mixture of Universals (MoU)라는 새로운 모델을 제안하여 시계열 예측의 단기 및 장기 의존성을 모두 포착하려고 합니다. MoU는 단기 의존성을 개선하는 Mixture of Feature Extractors (MoF)와 장기 의존성을 모델링하는 Mixture of Architectures (MoA)로 구성되어 있습니다.

- **Technical Details**: MoU는 단기 의존성을 위해 MoF를 사용하여 패치(Token) 표현을 개선하며, MoA는 Mamba, FeedForward, Convolution, Self-Attention 아키텍처를 계층적으로 통합하여 장기 의존성을 모델링합니다. MoF는 다양한 컨텍스트를 처리하기 위해 여러 서브 익스트랙터를 활용하여 스파스 활성화를 통해 선택적으로 활성화됩니다. MoA는 Selective State-Space Model (SSM)을 사용하여 주요 의존성을 선택하고 학습하는 초기 레이어를 가지고 있습니다.

- **Performance Highlights**: 실제 데이터셋 7개에 대한 실험 결과 MoU가 대부분의 데이터셋에서 최첨단 성능을 달성했으며, 상대적으로 낮은 계산 비용을 유지합니다.



### Spatio-Temporal Context Prompting for Zero-Shot Action Detection (https://arxiv.org/abs/2408.15996)
- **What's New**: 이번 논문에서는 클립 모델(CLIP)을 활용하여 보지 못한 행동을 감지하는 새로운 방법인 ST-CLIP을 제안합니다. 이는 인물-맥락 상호작용(Person-Context Interaction)을 모델링하여, 기존의 행동 감지 방식을 개선하는 데 초점을 맞추고 있습니다. 특히, 다중 행동 비디오에서 개인의 행동을 인식할 수 있는 방법을 제공합니다.

- **Technical Details**: ST-CLIP은 사전 학습된 시각-언어 모델을 이용하여, 시각 및 텍스트 영역 모두에서 제로샷 스페이쇼-템포럴 액션 디텍션(Zero-Shot Spatio-Temporal Action Detection)을 수행합니다. 이 방법은 파라미터 및 자원을 추가로 필요로 하지 않으며, 멀티 레이어 Context Prompting 모듈과 Interest Token Spotting 메커니즘을 통해 각 개인의 행동과 관련된 맥락 토큰을 식별하고, 이를 활용하여 행동 분류를 정확하게 수행합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법(ST-CLIP)은 J-HMDB와 UCF101-24 데이터셋에서 기존 방법보다 우수한 성과를 거두었으며, AVA 데이터셋에서는 동일한 비디오 내에서 다양한 보지 못한 행동을 개별적으로 감지하는 능력을 입증했습니다. 이러한 결과는 실제 응용에 근접할 뿐만 아니라, 제로샷 학습의 강력한 일반화 능력을 강조합니다.



### CoGen: Learning from Feedback with Coupled Comprehension and Generation (https://arxiv.org/abs/2408.15992)
Comments:
          17 pages, 9 figures

- **What's New**: 이번 연구는 언어 이해(comprehension)와 생성(generation) 능력을 결합하여 사용자와의 상호작용에서 지속적으로 학습하는 방법을 제안합니다. 두 능력을 통합하기 위한 새로운 기술이 도입되었습니다.

- **Technical Details**: 본 연구는 두 플레이어 참조 게임(two-player reference games)에서 진행되며, 수천 번의 사용자 상호작용을 통해 다양한 모델을 배포합니다. 상호작용 피드백 신호(interaction feedback signals)를 이용하여 학습하는 동시에, 이해와 생성 능력을 결합하여 성능을 개선합니다.

- **Performance Highlights**: 이 연구의 결과, 이해-생성 결합이 절대적으로 최대 26% 향상된 성능을 보여주고, 비결합 시스템에 비해 최대 17% 높은 정확도를 기록했습니다. 이러한 결합은 시스템의 언어에도 상당한 질적 영향을 미쳐, 더 인간적인 언어 사용을 가능하게 했습니다.



### In-Context Imitation Learning via Next-Token Prediction (https://arxiv.org/abs/2408.15980)
- **What's New**: 본 논문에서는 In-Context Robot Transformer (ICRT)라는 새로운 모델을 제안합니다. 이 모델은 로봇이 새로운 작업을 수행할 수 있도록 돕는 next-token 예측 모델로, 기존의 정책 매개변수를 업데이트하지 않고도 입력 단계에서 제공되는 맥락 정보를 해석하여 작업을 수행합니다.

- **Technical Details**: ICRT는 센서모터 궤적(sensorimotor trajectories)을 기반으로 autoregressive 예측을 수행하며, 언어 데이터나 보상 함수에 의존하지 않습니다. 로봇의 궤적은 이미지 관찰, 행동 및 상태 튜플로 구성되어 있으며, 이는 새로운 작업을 수행하기 위한 맥락으로 사용됩니다. ICRT는 다중 센서모터 궤적을 학습할 수 있는 긴 맥락 창을 제공합니다.

- **Performance Highlights**: Franka Emika 로봇을 사용한 실험에서 ICRT는 주어진 프롬프트에 따라 새로운 환경 구성에서 지정된 작업을 성공적으로 수행할 수 있음을 보여주었습니다. ICRT는 기존의 next-token 예측 모델보다 월등히 뛰어난 성능을 발휘했으며, unseen tasks에 대한 일반화 능력이 주목할 만합니다.



### Stability of Primal-Dual Gradient Flow Dynamics for Multi-Block Convex Optimization Problems (https://arxiv.org/abs/2408.15969)
Comments:
          31 pages; 4 figures

- **What's New**: 본 논문에서는 복합 볼록 최적화 문제를 위한 프라이멀-듀얼 그라디언트 흐름(dynamics)의 안정성 특성을 조사합니다. 일반화된 합의 제약 조건 아래에서 여러 개의 비부드러운(non-smooth) 항을 포함하는 목적 함수를 다룹니다. 제안된 동역학은 프로시말 증강 라그랑지안(proximal augmented Lagrangian)을 기반으로 하며, 기존의 ADMM보다 대규모 멀티 블록 시나리오에서의 분석 및 구현 도전 과제를 해결하는 효과적인 대안을 제공합니다.

- **Technical Details**: 작성된 알고리즘은 연속적이고 리프시츠(lipschitz) 연속 기울기를 가진 볼록 함수와 효율적으로 계산 가능한 프로시말 연산자(proximal operators)를 사용하는 폐쇄적이고 적절한 볼록(non-differentiable) 함수를 포함하여 다양한 구조적 특성을 활용합니다. 이 논문에서는 우리의 구조적 가정이 기존의 다양한 프라이멀-듀얼 동역학의 (exponential) 안정성을 증명하는 데 요구되는 가정보다 훨씬 약한 것을 보여주며, 전 세계적 (exponential) 수렴 보장을 수립합니다.

- **Performance Highlights**: 제안된 동역학은 병렬 및 분산 시스템에서의 최적화 응용 프로그램에 대한 편리함을 제공하며, 전산 실험을 통해 수행 가능성을 보여줍니다. 기존 동역학의 (linear) 수렴 사항 및 ADMM, EXTRA 알고리즘의 복잡성을 경감시키고, 확장된 적용 가능성을 통해 다양한 제약 조건을 쉽게 통합할 수 있는 점이 특징입니다.



### More Text, Less Point: Towards 3D Data-Efficient Point-Language Understanding (https://arxiv.org/abs/2408.15966)
- **What's New**: 이번 논문에서는 3D 데이터 부족 문제를 해결하기 위해 3D Data-Efficient Point-Language Understanding(3DEPL)이라는 새로운 과제를 제안합니다. GreenPLM 모델을 통해 최소한의 3D 포인트 클라우드 데이터와 텍스트 데이터 쌍으로 강력한 3D 객체 이해를 가능하게 하고자 합니다.

- **Technical Details**: GreenPLM은 CLIP에서 영감을 받아 사전 학습된 포인트 클라우드-텍스트 인코더를 사용하여 3D 포인트 클라우드 공간을 텍스트 공간에 매핑합니다. 6M 개의 자유 텍스트 설명을 생성하고 세 단계의 훈련 전략을 설계하여 LLM이 다양한 모달리티 간의 본질적인 연결을 탐색하도록 돕습니다.

- **Performance Highlights**: GreenPLM은 기존의 최첨단 모델이 사용하는 3D 훈련 데이터의 12%만으로도 우수한 3D 이해력을 달성합니다. 또한, GreenPLM은 텍스트 전용 데이터만으로 경쟁력 있는 성능을 발휘할 수 있습니다.



### Local Descriptors Weighted Adaptive Threshold Filtering For Few-Shot Learning (https://arxiv.org/abs/2408.15924)
- **What's New**: 본 논문에서는 Few-shot learning(소수 샘플 학습)에서의 이미지 분류 성능을 향상시키기 위해 새로운 가중 가변 임계값 필터링(weighted adaptive threshold filtering, WATF) 전략을 제안합니다. 이 방법은 이미지 카테고리에 가장 적합한 지역적 특성을 선택하여 배경 잡음을 효과적으로 걸러낼 수 있도록 설계되었습니다.

- **Technical Details**: WATF는 현재의 작업과 이미지 맥락에 따라 동적으로 조정되어, 이미지 카테고리에 가장 관련성이 높은 지역적 descriptor를 선택합니다. 이를 통해 모델은 카테고리 관련 정보에 더욱 집중하게 되며, 불필요한 배경 지역으로부터 방해를 최소화합니다. 실험은 N-way K-shot 프레임워크에서 진행되었으며, 시각화 실험을 통해 지역적 descriptor의 가중치가 정규 분포를 따른다는 것을 입증하였습니다.

- **Performance Highlights**: 본 방법은 세 가지 widely-used few-shot learning(소수 샘플 학습) 분류 데이터셋에서 최신 기술을 뛰어넘는 성능을 보여주었으며, 특히 CUB-200 데이터셋에서는 몇 가지 최신 transfer learning(전이 학습) 기반의 방법보다도 뛰어난 효과를 보였습니다. 이로 인해 본 방법은 이후 few-shot learning 연구에 상당한 참고 가치가 있을 것으로 기대됩니다.



### Leveraging Open Knowledge for Advancing Task Expertise in Large Language Models (https://arxiv.org/abs/2408.15915)
Comments:
          28 pages, 12 tables, 10 figures

- **What's New**: 이 논문은 특정 도메인 작업에서 대규모 언어 모델(LLMs)의 전문성을 개선하기 위해 몇 가지 인적 주석이 있는 샘플(K-shot)을 사용하는 새로운 접근 방식을 제안합니다. Open knowledge(공개 지식)를 활용하여 모델과 데이터 선택에서의 기존 방법들이 일반적인 능력 향상에 집중한 것에 비해, K-shot 데이터를 사용하여 전문성을 높이고자 합니다.

- **Technical Details**: 우리는 K-shot 학습 설정을 통해 LLM의 작업 전문 지식을 발전시키는 효율적이고 확장 가능한 파이프라인을 개발하였습니다. Mixture-of-Experts (MoE) 시스템을 통해 여러 전문가 간의 상호 보완적인 지식을 최대한 활용하며, K-shot 데이터를 통해 문제 해결 능력을 가진 모델을 선택합니다. 우리가 제안한 방법은 reasoning perplexity(추론 혼란도), exact match accuracy(정확한 매치 정확도), group diversity(그룹 다양성)를 바탕으로 합니다.

- **Performance Highlights**: 광범위한 실험 결과, 우리의 방법이 다양한 작업에서 공개 지식을 활용하는 기존 방법들보다 우수함을 입증하였습니다. K-shot 데이터를 통해 적합한 전문가 모델과 유용한 지시 사항을 선별하는 데 성공했으며, 저비용으로도 실세계에 적용 가능한 작업 전문성을 확보할 수 있었습니다.



### Nexus: Specialization meets Adaptability for Efficiently Training Mixture of Experts (https://arxiv.org/abs/2408.15901)
- **What's New**: 이 논문에서는 Mixture of Experts (MoE) 아키텍처의 새로운 변형인 Nexus를 소개합니다. Nexus는 전문가 임베딩을 도메인 표현에서 학습하여 적응형 라우팅을 제공하여 초기 모델의 전문성을 유지하면서 새로운 작업에 쉽게 적응할 수 있도록 합니다.

- **Technical Details**: Nexus는 기존의 밀집 전문가 모델을 MoE 모델로 업사이클링(upcycling)하여 전문성을 향상시키면서 새로운 도메인을 수용할 수 있는 유연한 특징을 보여줍니다. Nexus의 라우터는 도메인 특정 데이터의 임베딩을 통해 각 도메인에 맞는 전문가 임베딩으로 프로젝션하도록 학습됩니다. 이는 새로운 데이터를 위한 전문가를 독립적으로 훈련하여 MoE 구조에 쉽게 추가할 수 있게 해줍니다.

- **Performance Highlights**: Nexus는 초기 업사이클링에서 기준선에 비해 최대 2.1%의 상대적 향상을 기록했으며, 새로운 전문가를 추가하여 제한된 파인튜닝(finetuning) 데이터로도 18.8%의 향상을 달성하였습니다. 이를 통해 각 사용자가 자신의 요구에 맞춘 MoE 믹스를 지속적으로 구성할 수 있는 오픈 소스 생태계를 가능하게 합니다.



### Airfoil Diffusion: Denoising Diffusion Model For Conditional Airfoil Generation (https://arxiv.org/abs/2408.15898)
Comments:
          12 Pages, 6 figures

- **What's New**: 이번 연구에서는 확산 모델(diffusion model)을 이용하여 데이터 기반으로 에어포일(airfoil) 생성 방법론을 제안합니다. 이 모델은 기존 에어포일 데이터셋을 학습하여 임의의 벡터에서 새로운 에어포일을 생성할 수 있으며, 기계적 성능 지표(예: lift, drag) 또는 기하학적 조건을 바탕으로 하여 조건부로 생성할 수 있습니다.

- **Technical Details**: 연구에서는 UIUC Airfoil Database의 1600개 이상의 에어포일 데이터를 사용하여 확산 모델을 학습합니다. 각 에어포일은 좌표 쌍(x, y)로 표현되며, 모델의 균일한 입력 차원 요구사항을 충족하기 위해 각 샘플을 200개의 포인트로 보간(interpolate)합니다. 이렇게 표준화된 데이터셋을 바탕으로, 생성된 에어포일은 기존의 고정된 디자인 매개변수 없이도 공기역학적 성능을 효과적으로 최적화할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 확산 모델은 현실적인 공기역학적 특성을 가진 에어포일 형태를 효과적으로 생성하며, 전통적인 방법에 비해 효율성 및 유연성이 크게 향상되었습니다. 이를 통해 혁신적인 에어포일 디자인의 발견 가능성이 크게 확대되었습니다.



### A New Method for Cross-Lingual-based Semantic Role Labeling (https://arxiv.org/abs/2408.15896)
- **What's New**: 이번 논문에서는 자연어 처리(Natural Language Processing)에서 중요한 작업인 의미역 레이블링(Semantic Role Labeling)을 위한 새로운 딥 러닝 알고리즘을 제안합니다. 특히, 다국어에서 주석 데이터의 부족 문제를 해결하기 위해 모델 전이(Model Transfer) 방법을 활용하였습니다.

- **Technical Details**: 제안된 알고리즘은 CoNLL2009의 영어 부분과 페르시아어의 의미 역할 코퍼스를 포함하는 데이터셋을 사용하였습니다. 훈련의 효율성을 최적화하기 위해 각 언어에서 교육 데이터의 10%만을 사용하였습니다.

- **Performance Highlights**: 제안된 모델은 기존 Niksirt et al. 모델에 비해 상당한 개선을 보였으며, 단일 언어 모드에서 F1-score에서 2.05% 향상을, 교차 언어 모드에서는 6.23% 향상을 기록했습니다. 이는 제안된 모델이 실제로 더욱 우수함을 나타내며, 특히 여러 언어에 대한 주석 데이터 부족 문제를 해결하는 데 기여할 것으로 기대됩니다.



### Enhancing Intrusion Detection in IoT Environments: An Advanced Ensemble Approach Using Kolmogorov-Arnold Networks (https://arxiv.org/abs/2408.15886)
Comments:
          Accepted to be presented at the 11th International Symposium on Networks, Computers and Communications (ISNCC'24) will be held in Washington DC- USA, from October 22 to 25, 2024. 6 pages and 5 figures

- **What's New**: 최근 IoT(Internet of Things) 환경에서 침입 탐지를 위한 새로운 하이브리드 Intrusion Detection System (IDS)을 제안합니다. 이 시스템은 Kolmogorov-Arnold Networks (KANs)와 XGBoost 알고리즘을 결합하여 보안성을 강화합니다.

- **Technical Details**: 이 제안된 IDS는 KAN의 학습 가능한 활성화 기능을 활용하여 데이터 내의 복잡한 관계를 모델링하고, XGBoost의 강력한 앙상블 학습 기법을 통해 높은 분류 성능과 해석 가능성을 제공합니다. KAN은 매우 복잡한 패턴을 탐지할 수 있는 능력을 가지고 있으며, 나아가 고속 학습을 통해 IoT 환경에서의 이상 탐지를 효과적으로 수행합니다.

- **Performance Highlights**: 실험 결과, 하이브리드 IDS는 공격 탐지 정확도 99% 이상, F1-score, 정밀도(precision), 재현율(recall) 모두 98%를 초과하는 성과를 기록했습니다. 전통적인 Multi-Layer Perceptron (MLP) 네트워크와 비교하여 KAN과 XGBoost의 통합이 IoT 네트워크의 보안을 크게 강화할 수 있음을 보여주었습니다.



### Robust Statistical Scaling of Outlier Scores: Improving the Quality of Outlier Probabilities for Outliers (Extended Version) (https://arxiv.org/abs/2408.15874)
Comments:
          15 pages, 4 figures, accepted for publication in SISAP 2024

- **What's New**: 이 논문은 기존 통계적 스케일링(statistical scaling) 방법이 아웃라이어(outlier)와 인라이어(inlier)에 대해 동일한 품질의 확률(probability)을 제공하지 못한다는 점을 강조하며, 아웃라이어에 대한 확률 품질을 개선하기 위한 강력한 통계적 스케일링(robust statistical scaling) 방법을 제안합니다.

- **Technical Details**: 강력한 통계적 스케일링은 로버스트 추정기(robust estimators)를 사용하여 아웃라이어의 확률을 계산합니다. 이 방법은 기존의 아웃라이어 점수를 변환하여 아웃라이어 확률을 생성하는 여러 가지 변형을 실제 데이터셋과 아웃라이어 탐지 알고리즘에 대해 평가하여 아웃라이어에 대한 확률을 개선하는 것을 목표로 합니다.

- **Performance Highlights**: 이 연구에서는 아웃라이어 점수 변환 방법으로 사용된 강력한 통계적 스케일링이 아웃라이어에 대한 확률의 품질을 향상시킬 수 있음을 실증적으로 보여주며, 특히 의료, 금융 또는 공학과 같이 아웃라이어의 누락이 치명적일 수 있는 분야에서 중요성을 강조합니다.



### GenDDS: Generating Diverse Driving Video Scenarios with Prompt-to-Video Generative Mod (https://arxiv.org/abs/2408.15868)
- **What's New**: 이 연구는 자율주행 훈련을 위한 새로운 데이터 생성 방법인 GenDDS를 제안합니다. GenDDS는 다양한 교통 시나리오를 생성하기 위해 Stable Diffusion XL (SDXL) 모델을 활용하며, 고유한 환경 조건과 희귀한 발생 이벤트를 효과적으로 모델링하는데 중점을 두었습니다.

- **Technical Details**: GenDDS는 텍스트 설명을 기반으로 동영상 시나리오를 생성하는 방법으로, KITTI 데이터셋을 활용하여 LoRA(Low-Rank Adaptation) 모델을 훈련하고, ControlNet을 통해 다양한 주행 시나리오에 대한 제어 신호를 제공합니다. 이 접근 방식은 고해상도의 복잡한 비디오 시퀀스를 생성하는 데 효과적입니다.

- **Performance Highlights**: 실험 결과, GenDDS는 실제 주행 시나리오의 복잡성과 변동성을 잘 포착한 고품질 영상 생성을 보여주었습니다. 이는 자율주행 시스템 훈련을 위한 정교한 데이터 생성에 기여하며, 시뮬레이션 및 검증 목적의 가상 환경 구축에 새로운 가능성을 열어줍니다.



### Retrieval-Augmented Instruction Tuning for Automated Process Engineering Calculations : A Tool-Chaining Problem-Solving Framework with Attributable Reflection (https://arxiv.org/abs/2408.15866)
Comments:
          Accepted for publication at ML4CCE workshop at ECML PKDD 2024. Please find the link: this https URL

- **What's New**: 이 논문은 비즈니스 프로세스 엔지니어링 계산 문제 해결을 위한 새로운 자율 에이전트 프레임워크를 제안합니다. 이 프레임워크는 Retrieval-Augmented Instruction-Tuning (RAIT) 기법을 사용하여 오픈형 소형 코드 언어 모델(SLMs)을 강화합니다.

- **Technical Details**: RAIT 기법은 Retrieval-Augmented Code Generation (RACG)과 instruction-tuning을 결합하여, 자연어 명세로부터 코드를 생성하고 디버깅하며 최적화하는 기능을 제공합니다. 이 프레임워크는 다섯 단계의 작업 흐름을 따르며, 외부 도구와의 조합을 통해 효율적인 코드 생성을 목표로 합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 대규모 상용 모델과 유사한 성능을 보여주며, 화학 공정 및 설계 문제 해결에서 효과적이고 높은 정확성을 유지하고 있음을 입증합니다.



### microYOLO: Towards Single-Shot Object Detection on Microcontrollers (https://arxiv.org/abs/2408.15865)
Comments:
          Published at the ECML PKDD Conference 2023, at the 4th Workshop on IoT, Edge, and Mobile for Embedded Machine Learning

- **What's New**: 이 논문에서는 Cortex-M 기반 마이크로컨트롤러에서 YOLO를 사용한 단일 샷 객체 탐지의 가능성에 대한 결과를 제시합니다. 특히, microYOLO이라는 새로운 아키텍처를 통해 128x128 RGB 이미지를 분류할 때 3.5 FPS의 프레임 속도를 달성했습니다.

- **Technical Details**: microYOLO는 YOLO 아키텍처를 최적화하여 Flash 메모리가 800 KB 미만, RAM이 350 KB 미만인 자원 제한이 있는 플랫폼에서 작동하도록 설계되었습니다. 입력 이미지 해상도를 128x128으로 줄이고, 백본 네트워크의 학습 가능한 파라미터 수를 감소시켰습니다.

- **Performance Highlights**: microYOLO는 인체 탐지에서 평균 27.7%, 차량 탐지에서 평균 12.3%, 냉장고 내 장난감 식료품 탐지에서 평균 56.4%의 mAP0.5 값을 달성했습니다. 이는 데이터셋의 깊이와 복잡성의 차이에 따른 성능 차이를 보여줍니다.



### Knowledge Navigator: LLM-guided Browsing Framework for Exploratory Search in Scientific Literatur (https://arxiv.org/abs/2408.15836)
- **What's New**: 본 연구에서는 과학 문헌의 기하급수적 증가에 대한 대응으로, Knowledge Navigator라는 새로운 시스템을 발표합니다. 이 시스템은 사용자가 넓은 주제(query)에서 검색한 문서를 두 단계 계층으로 조직하고 구조화하여 탐색적인 검색 능력을 향상시킵니다.

- **Technical Details**: Knowledge Navigator는 사용자가 특정 하위 주제(subtopic)에 집중하고 관련 문서를 추가로 검색할 수 있도록 하여 반복적인 검색(iterative search) 및 깊이 있는 지식 발견(deep knowledge discovery)을 가능하게 합니다. 또한, LLM(대형 언어 모델) 기능과 클러스터 기반(cluster-based) 방법을 결합하여 효율적인 탐색(browsing) 방법을 제공합니다.

- **Performance Highlights**: 우리의 접근 방식은 자동 및 수동 평가를 통해 CLUSTREC-COVID 및 SCITOC의 두 가지 새로운 벤치마크(benchmarks)에서 효과성을 입증했습니다. 또한, 우리의 코드, 프롬프트(prompts), 벤치마크는 공개적으로 제공됩니다.



### Object Detection for Vehicle Dashcams using Transformers (https://arxiv.org/abs/2408.15809)
Comments:
          7 Pages, and 6 Figures

- **What's New**: 이 논문에서는 트랜스포머(transformers)를 사용한 새로운 객체 탐지 방법을 제안합니다. 이 방법은 첨단 DEtection TRansformer (DETR)를 기반으로 하여 다양한 주변 조건에서 강력한 성능을 보여줍니다.

- **Technical Details**: 제안된 시스템은 딥 러닝(Deep Learning) 기술을 활용하여 트랜스포머 기반의 객체 탐지 방식을 사용합니다. DETR 모델은 실세계 조건에서 수집된 데이터셋으로 훈련되었으며, mAP(Mean Average Precision) 값 0.95를 달성했습니다. 이는 인간 운전자가 어려움을 겪는 상황에서도 우수한 성능을 발휘합니다.

- **Performance Highlights**: 이 연구의 결과, 제안된 방법을 통해 스마트 대시캠 시스템의 객체 탐지 기능이 현저히 향상되었습니다. 특히, DETR 모델은 다양한 날씨와 조명 조건에서도 효과적으로 작동하며, 차량과 교통 표지판을 정확히 탐지할 수 있음을 입증했습니다.



### ModalityMirror: Improving Audio Classification in Modality Heterogeneity Federated Learning with Multimodal Distillation (https://arxiv.org/abs/2408.15803)
- **What's New**: 이번 연구에서는 Multimodal Federated Learning(다양한 형태의 연합 학습)에서 클라이언트의 modality 이질성을 극복하기 위한 새로운 접근 방식인 ModalityMirror를 소개합니다. 이 방법은 비디오가 없는 상황에서도 오디오 분류 성능을 향상시킬 수 있도록 설계되었습니다.

- **Technical Details**: ModalityMirror는 두 가지 단계로 구성됩니다. 첫 번째는 modality-wise Federated Learning(FL) 단계로, 단일 모달 인코더를 집계합니다. 두 번째는 다양한 모달 클라이언트에 대한 Federated Knowledge Distillation 단계로, 오디오 전용 학생 모델을 교육합니다.

- **Performance Highlights**: 실험 결과, ModalityMirror는 Harmony와 같은 기존 FL 방법과 비교하여 오디오 분류 성능에서 상당한 개선을 보여주며, 특히 정보가 부족한 레이블에 대해 오디오 모델의 성능을 향상시키는 데 효과적입니다.



### Emulating Brain-like Rapid Learning in Neuromorphic Edge Computing (https://arxiv.org/abs/2408.15800)
Comments:
          17 page journal article. Submitted to IOP NCE

- **What's New**: 본 연구는 디지털 뉴로모픽(digital neuromorphic) 기술을 사용하여 두 단계의 학습을 통해 뇌의 신경 및 시냅스 과정을 모방하면서 개인화된 인텔리전스를 효과적으로 구현하는 방법을 제시합니다. 이 접근 방식은 실시간(one-shot) 학습을 가능하게 하여 새로운 클래스의 데이터 학습을 가속화합니다.

- **Technical Details**: 두 단계로 구성된 학습 과정은 (1) 메타 학습(meta-learning) 단계에서 시냅스 가소성(synaptic plasticity) 하이퍼파라미터를 최적화하고, (2) 배포(deployment) 단계에서 최적화된 하이퍼파라미터를 사용하여 현장에서 새로운 데이터를 학습하는 것입니다. 메타 학습은 모델 불변 메타 학습(Model Agnostic Meta Learning, MAML) 알고리즘의 변형을 통해 이루어집니다. 이 방법론은 이벤트 기반(event-driven) 비전 센서 데이터와 Intel Loihi neuromorphic 프로세서에서 적용되어 빠르고 에너지를 효율적으로 학습할 수 있도록 합니다.

- **Performance Highlights**: 이 연구에서 제안된 방법론은 새로운 클래스의 데이터에 대해 실시간으로(one-shot) 학습하는 데 성공하였으며, 이 과정에서 이전의 전달 학습(transfer learning) 방식보다 현저히 개선된 성능을 보여주었습니다. 이를 통해 개인화된 학습 및 적응이 필요한 다양한 상황에 효과적으로 활용될 수 있습니다.



### Evaluating Named Entity Recognition Using Few-Shot Prompting with Large Language Models (https://arxiv.org/abs/2408.15796)
Comments:
          Github repo: this https URL

- **What's New**: 이 논문은 Named Entity Recognition (NER) 작업에 대한 Few-Shot Prompting의 효과 및 최신 대형 언어 모델(Large Language Models, LLMs)의 성능을 평가합니다. 전통적인 NER 시스템은 방대한 수의 레이블이 붙은 데이터셋이 필요하지만, Few-Shot Prompting은 최소한의 예제로 모델이 엔티티를 인식할 수 있도록 도와줍니다.

- **Technical Details**: 본 연구는 NER에서 Few-Shot Learning (FSL) 기법의 실성을 조사합니다. 실험은 GeoEDdA 데이터셋을 사용하여 수행되었으며, LLMs인 GPT-4와 BERT처럼 잘 알려진 모델들과 비교합니다. 실험 결과, LLM은 적은 데이터로 새로운 엔티티 유형 및 도메인에 적응하는 데 강점을 보이지만, 전통적인 완전 감독 방식과 비교할 때 성능 차이가 존재합니다.

- **Performance Highlights**: GPT-3.5보다 개선된 성능을 보인 GPT-4의 span 수준에서는 49%의 정확도를 보여주었고, token 수준에서 미세 평균 precision, recall 및 F1-score를 평가했습니다. 또한, 각 모델의 결과는 매우 다양한 변동성을 보였고, 일부 모델은 필요한 레이블 세트를 이해하지 못하거나 입력 문장을 반복하기만 하는 경우도 있었습니다.



### Easy, Interpretable, Effective: openSMILE for voice deepfake detection (https://arxiv.org/abs/2408.15775)
- **What's New**: 최근 ASVspoof5 데이터셋을 활용한 음성 진위 및 딥페이크 탐지 연구에서, 간단한 오픈SMILE 라이브러리 기반 특징을 활용하여 높은 정확도로 공격을 식별할 수 있음을 입증했습니다. 이 연구에서는 신뢰성 있는 공격 식별을 위해 적은 수의 스칼라 값 특징을 사용했습니다.

- **Technical Details**: 주요 기술 특징으로는 오픈SMILE의 ‘MeanUnvoicedSegmentLength’를 포함하여, 이를 통해 공격 A10의 EER(동등 오류율)을 10.3%로 달성했습니다. 전체 공격에 대해 0.8%의 EER, 평균 15.7 ± 6.0%의 EER도 기록했습니다. 이 연구는 텍스트 음성 변환(TTS) 아키텍처마다 고유한 특성이 있음을 발견했습니다.

- **Performance Highlights**: 이번 연구를 통해 ASVspoof5 데이터셋에서의 공격 분류 성능이 크게 향상되었으며, 특히 모델이 TTS 시스템의 서명이나 지문을 인식하는 데 용이함을 보여주었습니다. 이는 실질적인 음성 안티 스푸핑 모델 이해에 기여할 것입니다.



### A Survey on Evaluation of Multimodal Large Language Models (https://arxiv.org/abs/2408.15769)
- **What's New**: 이 논문은 Multimodal Large Language Models (MLLMs)의 평가 방법을 체계적이고 포괄적으로 검토하고 있습니다. MLLMs는 인간의 지각 및 추론 시스템을 모방하여 다양한 모달리티 인코더와 강력한 Large Language Models (LLMs)을 통합합니다. 이 논문은 기존 MLLM 평가 작업을 능력에 따라 검토하고 분류하여 인식, 인지, 추론 및 신뢰성 등 다양한 분야의 요구를 충족하는 방법을 제안합니다.

- **Technical Details**: MLLM은 LLMs를 '뇌'로 보고 다양한 모달리티 인코더를 감각 기관으로 배치하여 여러 모달리티 정보에 대한 이해와 추론 능력을 제공합니다. MLLM의 평가 단계와 메트릭스를 검토하고 평가 방법론을 다양한 시각에서 제시합니다.

- **Performance Highlights**: MLLMs의 평가 방법에 대한 체계적인 리뷰를 통해 MLLM의 강점과 한계를 정량화할 수 있는 기초를 제공합니다. 또한, MLLMs를 신뢰성 관점에서 탑재할 필요성을 강조하며, 다양한 하위 작업에서의 MLLMs 성능을 평가하는 방법을 제안합니다.



### TCNFormer: Temporal Convolutional Network Former for Short-Term Wind Speed Forecasting (https://arxiv.org/abs/2408.15737)
- **What's New**: 이 연구는 단기(12시간) 풍속 예측을 위해 Temporal Convolutional Network Former(TCNFormer)를 제안합니다. TCNFormer는 Temporal Convolutional Network(TCN)과 transformer encoder를 통합하여 풍속의 시공간 특성을 효과적으로 포착합니다.

- **Technical Details**: TCNFormer은 CT-MSA(causal temporal multi-head self-attention)와 TEA(temporal external attention)라는 두 가지 주의(attention) 메커니즘을 사용합니다. CT-MSA는 입력 단계의 출력이 오직 이전 단계의 데이터에만 의존하도록 하여 인과성을 보장하고, TEA는 다양한 샘플 시퀀스 간의 잠재적인 관계를 탐색합니다. 성능 향상을 위한 Locality 기능도 도입되었습니다.

- **Performance Highlights**: TCNFormer는 1년 간의 NASA POWER 데이터를 사용하여 최신 모델들보다 예측 정확도가 뛰어난 결과를 보였습니다. 이는 실제 풍력 발전 시스템의 응용에서 유망한 성능을 발휘할 가능성을 입증합니다.



### Advanced POD-Based Performance Evaluation of Classifiers Applied to Human Driver Lane Changing Prediction (https://arxiv.org/abs/2408.15722)
Comments:
          Manuscript: 8 pages, 6 figures, 4 tables

- **What's New**: 본 연구에서는 프로세스 파라미터의 영향을 고려한 머신러닝(ML) 분류기의 성능 평가를 위한 수정된 확률 탐지(POD) 접근 방식을 제안한다. 특히, 차량 운전자의 차선 변경 행동 예측을 위한 ML 모델을 사례로 사용하여, 프로세스 파라미터로서 남은 시간을 활용하였다. 이러한 접근 방식은 기존의 단순 hit/miss 접근 방식보다 신뢰성을 높인다.

- **Technical Details**: 수정된 POD 접근 방식을 사용하여 ML 알고리즘의 성능 평가를 수행한다. 이전 연구에서 제시된 ML 분류기는 hit/miss 접근 방식을 통해 결과가 0 또는 1로 구분되지만, 본 연구에서는 각 타임스텝에서 ML 알고리즘에 의해 도출된 차선 변경 확률을 고려하여 최종 결과를 계산한다. 이를 통해 평가가 간소화되고 신뢰성이 향상된다.

- **Performance Highlights**: 제안된 방법의 성능 평가 결과는 기존의 hit/miss 접근 방식 및 â versus a 접근 방식과 비교되며, 제안된 방법이 hit/miss 접근 방식의 신뢰성을 향상시키고 보다 보수적인 행동을 제공함을 보여준다.



### Evaluating Model Robustness Using Adaptive Sparse L0 Regularization (https://arxiv.org/abs/2408.15702)
Comments:
          Accepted by the 20th International Conference on Advanced Data Mining and Applications (ADMA 2024)

- **What's New**: 본 논문에서는 L0 norm을 기반으로 하는 새로운 공격 방법인 Adaptive Sparse and Lightweight Optimization (ASLO)을 제안합니다. 이 방법은 딥 뉴럴 네트워크(DNN)의 강건성 평가를 개선하기 위해 설계되었습니다.

- **Technical Details**: ASLO는 모델 피드백을 기반으로 실시간으로 변동하는 섭동(perturbation) 매개변수를 조정하여 L0 norm 섭동을 최적화합니다. 이는 최소한의 변화로 효과적인 공격을 가능하게 하여 공격의 정확성과 효율성 간의 균형을 유지하게 합니다.

- **Performance Highlights**: ASLO는 다양한 데이터셋과 모델 아키텍처에 걸쳐 우수한 공격 성공률을 유지하며, 더욱 효율적인 저자극 섭동을 생성하는 데 성공합니다.



### G-Style: Stylized Gaussian Splatting (https://arxiv.org/abs/2408.15695)
- **What's New**: 이번 논문에서는 G-Style이라는 새로운 알고리즘을 소개합니다. 이 알고리즘은 Gaussian Splatting을 사용하여 이미지를 3D 장면으로 스타일을 전이하는 방식으로, 기존의 방법들이 가지는 스타일 일관성 문제를 해결하고자 합니다.

- **Technical Details**: G-Style은 세 가지 단계로 이루어집니다: 첫 번째 단계에서는 불필요한 Gaussian을 제거하고, 두 번째 단계에서는 이미지 스타일의 다양한 스케일을 보존하기 위해 설계된 손실 함수들을 결합합니다. 마지막으로, 스타일화 과정에서는 색상의 기울기를 추적하여 필요한 곳에서 Gaussian을 분할하여 세부 사항을 추가합니다.

- **Performance Highlights**: 실험 결과, G-Style은 몇 분 만에 고품질의 스타일화된 장면을 생성하며, 기존 방법들에 비해 품질과 렌더링 속도 모두에서 우수한 성능을 보였습니다.



### An Empirical Study on Self-correcting Large Language Models for Data Science Code Generation (https://arxiv.org/abs/2408.15658)
- **What's New**: 이 논문은 CoT-SelfEvolve라는 새로운 방법론을 제안하며, 이는 LLM(대형 언어 모델)이 자가 수정(Self-Correcting) 프로세스를 통해 코드 품질을 향상시키는 방식을 탐구하고 있습니다. 특히, 실제 프로그래밍 문제에 대한 피드백을 기반으로 한 사고의 연쇄(Chain of Thought)를 활용합니다.

- **Technical Details**: CoT-SelfEvolve 모델은 초기 코드 스니펫을 생성한 후, 이를 평가하여 오류를 식별하고, 피드백을 반영하여 코드를 개선하는 반복적인 과정을 거칩니다. 데이터 과학 코드에 특화되었으며, NumPy와 Pandas와 같은 라이브러리를 포함합니다. 이 모델은 558,402개의 게시글과 972,513개의 관련 댓글로 구성된 데이터셋을 이용하여 개발되었습니다.

- **Performance Highlights**: CoT-SelfEvolve는 DS-1000 데이터셋에 대한 평가 결과, 기존 모델들보다 상당히 우수한 성능을 보였으며, 특히 복잡한 문제를 해결하는 데 있어 정확도가 각 반복마다 현저히 증가했습니다. 이로 인해 코드 생성 초기와 후속 반복 모두에서 상당한 개선이 이루어졌습니다.



### Harnessing the Intrinsic Knowledge of Pretrained Language Models for Challenging Text Classification Settings (https://arxiv.org/abs/2408.15650)
Comments:
          PhD thesis

- **What's New**: 이번 논문에서는 사전 학습된 언어 모델(PLMs)의 본질적인 지식을 활용하여 텍스트 분류의 세 가지 도전적 설정을 탐구합니다. 이 접근법은 오도된 그러나 부정확한 방해물을 선택하고, 보이지 않는 레이블에 대한 모델 일반화를 향상시키며, 대규모 언어 모델의 컨텍스트 학습 프롬프트에 대한 민감성을 해결하는 데 중점을 둡니다.

- **Technical Details**: 연구자들은 PLMs에서 파생된 맥락화된 단어 표현을 기반으로 방해물 생성을 위한 모델을 개발하였고, 도메인에 독립적인 작업 레이블 설명으로 작은 미세 조정 데이터 세트를 생성하였습니다. 이를 통해 모델의 강인성과 성능을 크게 향상시킬 수 있었습니다.

- **Performance Highlights**: 제안된 모델은 인간의 성능에 접근하거나 이를 초과하는 결과를 도출하였으며, 각 모델 구성요소의 중요성에 대한 정량적 분석을 통해 다양한 특징들이 퍼포먼스에 미치는 영향을 평가하였습니다.



### GANs Conditioning Methods: A Survey (https://arxiv.org/abs/2408.15640)
- **What's New**: 최근 Generative Adversarial Networks (GANs)의 발전으로 이를 활용한 다양한 응용 프로그램에서 조건부 생성 모델인 conditional GANs (cGANs)의 필요성이 커지고 있습니다. 이 논문은 cGANs의 다양한 조건화 방법을 검토하고 각 방법의 특성과 이론적 기초를 강조합니다.

- **Technical Details**: cGANs는 생성 과정에서 특정 기준을 따르도록 하는 추가 정보를 통합하여 원래 GAN 아키텍처를 확장합니다. 이 연구는 GAN의 조건화를 위한 여러 방법을 비교하고 평가하는 내용으로 구성되어 있으며, discriminator와 generator에 대한 조건화 접근 방식 각각을 다룹니다. 조건부 생성을 위한 클래스 레이블을 사용한 기법에 집중합니다.

- **Performance Highlights**: cGANs의 여러 구조에 대한 종합적인 분석을 통해 각 방법의 강점과 한계를 파악하고, 다양한 이미지 데이터셋에서 성능을 평가하여 차세대 생성 모델링 연구 및 응용에 대한 통찰력을 제공합니다.



### Structural Optimization of Lightweight Bipedal Robot via SERL (https://arxiv.org/abs/2408.15632)
- **What's New**: 이번 논문에서는 SERL (Structure Evolution Reinforcement Learning) 알고리즘을 소개합니다. 이 알고리즘은 이족 보행 로봇의 최적 구조 설계를 위한 새로운 접근법으로, 강화 학습 (Reinforcement Learning)과 진화 알고리즘 (Evolution Algorithm)을 결합해 다차원 설계 공간에서 최적의 매개 변수 조합을 찾습니다.

- **Technical Details**: SERL 알고리즘을 통해 설계한 이족 보행 로봇 Wow Orin은 구조 최적화에 기반하여 최적 다리 길이를 추정합니다. 이 로봇은 특히 체중과 모터 토크에 맞춰 설계되었으며, 주어진 설계 공간과 작업 조건 내에서 최상의 구조를 최적화하는 데 성공했습니다. 실험적으로 SERL 알고리즘의 효율성을 검증했습니다.

- **Performance Highlights**: WOW Orin은 기존 이족 보행 로봇인 Cassie와 Unitree H1과 비교했을 때 뛰어난 에너지 효율성과 성능을 입증하였습니다. 실험 결과는 SERL 알고리즘의 실제 설계 적용 가능성을 지원하며, 로봇의 성능 향상을 보여줍니다.



### CodeSift: An LLM-Based Reference-Less Framework for Automatic Code Validation (https://arxiv.org/abs/2408.15630)
- **What's New**: CodeSift라는 새로운 코드 검증 프레임워크를 소개하며, LLMs를 활용하여 코드의 실행이나 참조 코드, 인간 피드백 없이도 첫 번째 필터로 활용한다.

- **Technical Details**: CodeSift는 LLMs (Large Language Models)를 기반으로 하여 발생할 수 있는 오류를 줄이고 코드 검증 과정을 간소화한다. 이 프레임워크는 두 가지 프로그래밍 언어를 포함한 세 가지 다양한 데이터셋에서 그 효율성을 평가하였다.

- **Performance Highlights**: CodeSift는 기존의 최첨단 코드 평가 방법들보다 더 뛰어난 성능을 보였으며, 전문가의 내부 테스트 결과, 생성된 출력물이 인간의 선호와 일치함을 확인하였다.



### CBF-LLM: Safe Control for LLM Alignmen (https://arxiv.org/abs/2408.15625)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 정렬을 보장하기 위한 제어 기반의 프레임워크를 제안하며, 사용자에게 바람직한 텍스트 생성을 보장하기 위해 제어 장벽 함수(Control Barrier Function, CBF)를 활용합니다. CBF에 기반한 안전 필터를 적용하여 모델의 출력 생성을 조정하는 방법을 보여줍니다.

- **Technical Details**: 이 프레임워크는 LLM의 출력 형성 과정에서 CBF를 적용하여 사용자가 원하는 바람직한 내용으로 조정하며, Llama 3와 RoBERTa 모델을 활용하여 구현되었습니다. 이 방식은 모델 파라미터를 수정하지 않고 외부 필터를 추가하는 형태로, '학습 없는(learning-free)' 정렬 프레임워크로 작동합니다.

- **Performance Highlights**: 실험 결과, CBF 제어를 통해 사용자가 지정한 정렬 작업에서 필요한 개입의 수가 줄어들어 시스템의 제어 능력과 효율성이 높아졌음을 나타냅니다.



### CGRA4ML: A Framework to Implement Modern Neural Networks for Scientific Edge Computing (https://arxiv.org/abs/2408.15561)
- **What's New**: CGRA4ML은 기존의 hls4ml의 단점을 극복하여, 대규모 신경망 모델을 효율적으로 구현할 수 있는 개방형 모듈식 프레임워크입니다. CGRA4ML은 off-chip 데이터 저장을 지원하며, ResNet, PointNet 및 transformers와 같은 다양한 신경망 아키텍처를 처리할 수 있는 기능을 확장합니다.

- **Technical Details**: CGRA4ML은 Coarse-Grained Reconfigurable Array (CGRA) 아키텍처를 기반으로 하여, 공간 재사용을 촉진하고 modern neural network 기능을 배치하기 위해 off-chip 데이터 저장소를 활용합니다. Python API는 qkeras를 기반으로 하며, SystemVerilog RTL을 생성하여 ASIC 및 FPGA 설계 흐름에 더 적합하도록 설계되었습니다. 이를 통해 CGRA4ML은 다양한 신경망 모델을 프로그래밍, 구성 및 활용할 수 있습니다.

- **Performance Highlights**: CGRA4ML은 ResNet-50, PointNet 및 Particle Transformer 모델을 구현 및 비교하는 데 성공하였으며, 기존 hls4ml로는 불가능했던 더 큰 모델을 확장하여 처리할 수 있는 능력을 보여줍니다. 전력 및 면적비교를 통해 CGRA4ML의 효율성을 입증했습니다.



### An Investigation of Warning Erroneous Chat Translations in Cross-lingual Communication (https://arxiv.org/abs/2408.15543)
- **What's New**: 이 연구에서는 채팅 번역의 복잡성 문제를 해결하기 위해 Multidimensional Quality Metrics for Chat Translation (MQM-Chat)라는 새로운 평가 지표를 제시합니다. 다양한 모델의 실험을 통해 각 모델이 공통적인 오류를 발생시키고, 각 모델의 단점이 다르다는 점을 확인하였습니다.

- **Technical Details**: 연구는 MQM-Chat을 통해 다섯 개의 번역 모델 실험을 수행하였으며, 번역 모델의 불완전함을 인식하고 사용자에게 오류 경고 메시지를 제공함으로써 사용자 경험 개선을 목표로 합니다. 참여자들에게는 가상의 다국어 채팅 시나리오가 제공되며, 번역 오류 시 경고 메시지가 표시됩니다.

- **Performance Highlights**: 설문조사 결과, 경고 메시지는 (1) 다국어 채팅에서 유용하며, (2) 사용자가 채팅 행동을 변경하도록 유도할 수 있는 가능성을 보여줍니다. 이는 차후 채팅 번역에서의 사용자 지원 기능 개발에 중요한 기초 자료가 될 것입니다.



### Kangaroo: A Powerful Video-Language Model Supporting Long-context Video Inpu (https://arxiv.org/abs/2408.15542)
- **What's New**: 이번 논문에서는 비디오 데이터 처리에 중점을 둔 강력한 비디오 멀티모달 모델인 Kangaroo를 소개합니다. 현재의 LLMs에서 비디오 데이터를 효과적으로 처리하는 데 있어 어려움을 해결하기 위해 고품질 주석이 있는 대규모 데이터셋을 구축하고, 점진적으로 입력 프레임 수를 늘리는 커리큘럼 트레이닝 전략을 설계하였습니다.

- **Technical Details**: Kangaroo는 총 8B parameters로 비디오 이해를 위한 다양한 벤치마크에서 최첨단 성능을 달성합니다. 데이터 큐레이션 시스템을 통해 영상과 언어의 사전학습 및 지침 튜닝을 위한 고품질 데이터셋을 구축하였고, 최대 64와 160의 입력 프레임 수를 통해 긴 비디오를 효과적으로 처리할 수 있도록 하였습니다.

- **Performance Highlights**: Kangaroo는 다양한 비디오 이해 벤치마크에서 우수한 성능을 보였으며, 특히 긴 비디오에 특화된 벤치마크에서 10B 이상의 파라미터를 가진 다른 모델들을 초월하는 성과를 냈습니다.



### Improving Thompson Sampling via Information Relaxation for Budgeted Multi-armed Bandits (https://arxiv.org/abs/2408.15535)
Comments:
          accepted

- **What's New**: 이번 연구에서는 자원이 제한된 Bayesian budgeted multi-armed bandit 문제를 다룹니다. 기존의 Budgeted Thompson Sampling (BTS) 방법은 남은 예산 정보를 고려하지 않는 문제점을 해결하기 위해 Information Relaxation Sampling 프레임워크를 적용하였습니다.

- **Technical Details**: 우리는 정보 완화 샘플링(Information Relaxation Sampling) 프레임워크를 통해 전통적인 K-armed bandit 문제를 일반화하여, 예산 제약에 따라 최적화된 랜덤화 알고리즘을 제안합니다. 제안된 알고리즘은 BTS와 유사한 랜덤성을 가지지만, 선택 과정에서 남은 예산에 대해 더 신중하게 최적화를 진행합니다.

- **Performance Highlights**: 이론적 분석 및 시뮬레이션 결과를 통해, 우리의 알고리즘과 성능 기준은 다양한 설정에서 BTS 및 기존 벤치마크보다 점진적인 개선을 보이는 것으로 나타났습니다. 이는 실제 사례에서도 적용되어 그 효과를 입증하고 있습니다.



### LRP4RAG: Detecting Hallucinations in Retrieval-Augmented Generation via Layer-wise Relevance Propagation (https://arxiv.org/abs/2408.15533)
- **What's New**: 이번 논문에서는 RAG (Retrieval-Augmented Generation) 기술을 기반으로 한 hallucination(환각) 탐지 방법인 LRP4RAG를 제안합니다. LRP (Layer-wise Relevance Propagation) 알고리즘을 활용하여 RAG의 출력에 대한 입력의 관련성을 계산하는 새로운 접근 방식을 도입했습니다.

- **Technical Details**: LRP4RAG는 RAG 생성기의 입력과 출력 간의 관련성을 계산하기 위해 LRP를 사용하고, 이를 통해 관련성 행렬을 추가적으로 추출(resampling) 및 처리하는 방법을 포함합니다. 처리된 관련성 데이터는 여러 분류(classifier)를 통해 출력에 환각이 포함되어 있는지를 판단하는 데 사용됩니다.

- **Performance Highlights**: 광범위한 실험 결과, LRP4RAG가 기존의 기준 모델들보다 우수한 성능을 보임을 입증했습니다. 이는 RAG 관련 환각 탐지에 있어 LRP의 사용이 효과적임을 나타냅니다.



### Continual-learning-based framework for structural damage recognition (https://arxiv.org/abs/2408.15513)
Comments:
          18 pages, 12 figures

- **What's New**: 이번 연구에서는 기존의 convolutional neural network (CNN) 기반의 손상 인식 방식들이 지닌 한계를 극복하기 위해, ResNet-34 구조에 'learning without forgetting' 방식을 통합한 지속적 학습 기반 손상 인식 모델(Continual Learning-based Damage Recognition Model, CLDRM)을 제안합니다.

- **Technical Details**: CLDRM은 Reinforced Concrete (RC) 구조 및 관련 구조 구성 요소의 손상을 인식하기 위해 설계되었습니다. 주된 접근 방식은 지속적 학습 과정을 통해 이전 학습된 작업의 정확도를 유지하면서 새로운 작업을 효과적으로 배우도록 하는 것입니다. 이 모델은 4개의 인식 작업을 위한 세 가지 실험을 수행하여 그 가능성과 효과성을 검증했습니다.

- **Performance Highlights**: 연구 결과, CLDRM은 4개의 지속적 학습 작업에서 예측 시간과 데이터 저장을 각각 약 75% 감소시켰으며, 손상 인식 및 분류에서 높은 정확도를 달성했습니다. 특히, CLDRM은 학습 작업 수가 증가함에 따라 이전에 학습한 작업들에 대한 감소가 적었음을 보여주었습니다.



### AeroVerse: UAV-Agent Benchmark Suite for Simulating, Pre-training, Finetuning, and Evaluating Aerospace Embodied World Models (https://arxiv.org/abs/2408.15511)
- **What's New**: 이 논문은 UAV(무인 항공기)의 자율 지능을 위한 첫 번째 대규모 실제 이미지-텍스트 사전 훈련 데이터셋인 AerialAgent-Ego10k를 구성하고, UAV의 자율성을 지원하는 항공 우주 구현 세계 모델을 위한 다섯 가지 다운스트림 작업을 명확히 정의합니다.

- **Technical Details**: 자율 비행 UAV를 위한 항공우주 구현 세계 모델의 개발은 UAV의 인지, 인식 및 행동을 통합하는 것에 중점을 둡니다. 이 논문에서는 5가지 주요 다운스트림 작업, 즉 항공우주 구현 장면 인식(aerospace embodied scene awareness), 공간 추론(spatial reasoning), 내비게이셔널 탐색(navigational exploration), 작업 계획(task planning), 및 동작 결정(motion decision)을 정의하고, 각 작업에 대한 검증 데이터셋을 구성하였습니다. 또한, GPT-4 기반의 SkyAgentEval을 개발하여 결과를 평가합니다.

- **Performance Highlights**: 제안된 AeroVerse 벤치마크는 10개 이상의 2D/3D 비주얼-언어 모델과 여러 평가 지표 및 시뮬레이터를 통합하여 항공우주 구현 지능의 탐색 및 개발을 촉진할 것입니다. 이 연구는 UAV 에이전트 작업에서 2D/3D 비주얼 언어 모델의 잠재력과 한계를 밝혀냅니다.



### Measuring the Reliability of Causal Probing Methods: Tradeoffs, Limitations, and the Plight of Nullifying Interventions (https://arxiv.org/abs/2408.15510)
- **What's New**: 이 연구에서는 기존의 인과적 탐지 기법의 신뢰성을 평가하기 위한 일반적인 경험적 분석 프레임워크를 제안합니다. 특히, 두 가지 주요 속성을 정의하고 정량화하여 인과적 탐지의 효과성을 평가합니다: 완전성(Completeness)과 선택성(Selectivity).

- **Technical Details**: 논문에서는 여러 주요 인과적 탐지 방법론을 대상으로 실험을 수행하고, 이 방법들이 상호 간의 트레이드오프(tradeoff)를 보이며, 한 가지 방법이 두 가지 기준을 동시에 만족할 수 없음을 발견했습니다. 연구 결과는 또한 반증 방법(nullifying interventions)이 대안적 방법(counterfactual interventions)보다 항상 덜 완전하다는 것을 보여주었습니다.

- **Performance Highlights**: 연구에 따르면, 인과적 탐지 방법 간에는 명확한 트레이드오프가 존재하며, 일반적으로 반증 방법은 안정성이 떨어지고 완전성이 낮아 인과적 탐지에서 효과적인 접근이 아닐 수 있음을示했습니다.



### EmoAttack: Utilizing Emotional Voice Conversion for Speech Backdoor Attacks on Deep Speech Classification Models (https://arxiv.org/abs/2408.15508)
Comments:
          Submitted to ICASSP 2025

- **What's New**: 이 논문에서는 감정을 공격 대상으로 삼는 음성 백도어 공격 방법인 EmoAttack을 제안하며, 다양한 감정 기반의 공격이 더욱 효과적임을 보여준다.

- **Technical Details**: EmoAttack은 음성 변환 기술인 Emotional Voice Conversion(EVC)을 활용하여 감정적인 음성 샘플을 생성하고, 이를 통해 백도어 공격을 수행한다. 두 가지 음성 분류 작업에서 테스트를 진행하고, 제안한 방법이 높은 공격 성공률과 효과성을 보인다고 밝혔다.

- **Performance Highlights**: EmoAttack 방법은 키워드 스포팅(keyword spotting, KWS) 및 화자 인증 시스템(speaker verification, SVs)에서 높은 공격 효과성을 기록하며, 감정이 강한 음성이 공격의 주요 타겟으로 적합하다는 결과를 도출하였다.



### RoboSense: Large-scale Dataset and Benchmark for Multi-sensor Low-speed Autonomous Driving (https://arxiv.org/abs/2408.15503)
- **What's New**: 본 연구에서는 자율주행 기술의 발전을 위해 로봇 차량의 근거리 환경 인지를 위한 대규모 다중 모드 데이터셋 RoboSense를 구축하였습니다. 이는 독창적으로 다양한 환경에서 수집된 3D 객체에 대한 인식 및 추적 연구에 기여할 것입니다.

- **Technical Details**: RoboSense 데이터셋은 Camera, LiDAR, 및 Fisheye 센서를 포함하여 총 133K 개의 동기화된 데이터 프레임과 1.4M 개의 3D 바운딩 박스 및 식별자가 포함되어 있습니다. 6개의 주요 장면 클래스에서 216K 개의 궤적이 형성되었습니다.

- **Performance Highlights**: RoboSense는 이전의 KITTI 및 nuScenes 데이터셋보다 270배 및 18배 더 많은 근거리 장애물 주석을 제공합니다. 이는 자율주행 모델이 저속 환경에서 다양한 장애물과 복잡한 조건을 극복하는 데 중요한 역할을 할 것입니다.



### MODULI: Unlocking Preference Generalization via Diffusion Models for Offline Multi-Objective Reinforcement Learning (https://arxiv.org/abs/2408.15501)
Comments:
          23 pages, 7 figures

- **What's New**: 본 논문에서는 Multi-objective Reinforcement Learning (MORL) 분야에서 Offline MORL 접근법을 통해 OOD(Out-Of-Distribution) 환경에서의 기본 정책 생성을 개선하는 방법을 제안합니다. 구체적으로, 고급 확산 모델(Diffusion Models)의 표현력과 일반화 능력을 활용한 MODULI(Multi-objective Diffusion Planner with Sliding Guidance)라는 새로운 알고리즘을 소개합니다.

- **Technical Details**: MODULI는 두 가지 반환 정규화(return normalization) 방법을 도입하고, 슬라이딩 가이던스(sliding guidance) 메커니즘을 통해 OOD 선호도에 대한 일반화 능력을 향상시킵니다. 이를 통해 정책을 미세하게 조정할 수 있으며, 훈련된 슬라이더 어댑터(slider adapter)는 선호도 변화의 방향을 포착하여 ID(In-Distribution) 환경에서 OOD 환경으로의 전환을 가능하게 합니다.

- **Performance Highlights**: D4MORL 벤치마크에서 MODULI는 기존 오프라인 MORL 알고리즘과 비교해 탁월한 성능을 보여주며, 특히 OOD 선호도에 대한 뛰어난 일반화 능력을 입증합니다. 실험 결과, MODULI는 누락된 선호도 영역을 보완하고 Pareto 프런트(Pareto front)를 확장하기 위한 성공적인 방법으로 확인되었습니다.



### Deep Learning to Predict Late-Onset Breast Cancer Metastasis: the Single Hyperparameter Grid Search (SHGS) Strategy for Meta Tuning Concerning Deep Feed-forward Neural Network (https://arxiv.org/abs/2408.15498)
- **What's New**: 이번 연구에서는 유방암 전이를 예측하기 위한 DFNN 모델을 개발하였으며, 이를 위해 Single Hyperparameter Grid Search (SHGS) 전략을 도입하여 하이퍼파라미터 최적화 문제를 해결하였습니다.

- **Technical Details**: SHGS는 8개의 타겟 하이퍼파라미터(epochs, batch size, dropout, L1, L2, learning rate, decay, momentum)에 대해 실험을 수행하였으며, 이를 통해 모델 성능과 하이퍼파라미터 값 간의 관계를 분석했습니다.

- **Performance Highlights**: 실험 결과, 하이퍼파라미터의 최적 값은 데이터셋에 따라 달라지며 다른 하이퍼파라미터 설정의 영향을 받는 것을 보였습니다. 또한, 비용 효율적인 그리드 서치를 위한 하이퍼파라미터 값의 축소 범위를 제시했습니다.



### Remove Symmetries to Control Model Expressivity (https://arxiv.org/abs/2408.15495)
Comments:
          preprint

- **What's New**: 이번 논문에서는 손실 함수에 존재하는 대칭(symmetry)이 신경망의 용량을 줄이고 특징(feature)을 무시하는 두 가지 구체적인 메커니즘을 증명한 후, 이러한 대칭에 의해 발생하는 저용량 상태를 제거하는 'syre'라는 간단하고 이론적으로 정당화된 알고리즘을 제안합니다.

- **Technical Details**: 대칭은 손실 함수에서 저용량 새들 포인트(low-capacity saddle points) 형성을 초래하며, 이는 신경망의 학습을 저해합니다. 제안하는 syre 알고리즘은 이러한 대칭을 수동적으로 제거할 수 있도록 설계되었으며, 대칭에 대한 사전 지식 없이도 구현할 수 있습니다.

- **Performance Highlights**: syre 알고리즘은 기초적인 신경망 문제 해결부터 복잡한 실세계 시나리오에 이르기까지 다양한 분야에서 신경망의 훈련 성능을 향상시키는 것으로 입증되었습니다.



### CTRQNets & LQNets: Continuous Time Recurrent and Liquid Quantum Neural Networks (https://arxiv.org/abs/2408.15462)
- **What's New**: 본 논문에서는 양자 컴퓨팅의 이점을 활용하여 데이터를 보다 효과적으로 분석할 수 있는 새로운 모델 클래스인 Liquid Quantum Neural Network (LQNet)와 Continuous Time Recurrent Quantum Neural Network (CTRQNet)을 제안합니다.

- **Technical Details**: LQNet과 CTRQNet은 기존의 양자 신경망(QNNs)의 정적 구조로 인한 한계를 극복하며, 다양한 데이터 간의 복잡한 관계를 효율적으로 학습할 수 있습니다. 이러한 모델들은 데이터를 통해 학습한 패턴을 넓은 데이터 세트에 적용할 수 있어 다이나믹한 지능을 갖추고 있습니다.

- **Performance Highlights**: LQNet과 CTRQNet은 CIFAR 10 데이터 세트에서 기존의 QNN보다 최대 40%의 정확도 향상을 달성하였으며, 일반적인 QNN보다 더 빠르게 최적 상태에 도달하는 성능을 보입니다.



### Online Event-Triggered Switching for Frequency Control in Power Grids with Variable Inertia (https://arxiv.org/abs/2408.15436)
- **What's New**: 이 논문에서는 가변 관성(time-varying inertia)을 가진 시스템에서 주파수 제어를 위한 Neural Proportional-Integral (Neural-PI) 제어기를 제안하고, 온라인 이벤트 주도 스위칭 알고리즘을 도입하여 최적의 제어기를 선택하는 방법을 제시하였습니다.

- **Technical Details**: 주파수 동역학은 비선형 스위칭 시스템(nonlinear switching system)으로 모델링되며, 각 모드에서 비선형 스윙 방정식(nonlinear swing equations)에 의해 설명됩니다. Neural-PI 구조는 고정 관성 모드에서 주파수 제어를 위해 Exponential Input-to-State Stability (Exp-ISS)를 보장합니다. 온라인 스위칭 알고리즘을 통해 성능에 따라 최적의 Neural-PI 제어기를 동적으로 선택합니다.

- **Performance Highlights**: IEEE 39-bus 시스템에서 시뮬레이션을 수행한 결과 제안된 온라인 스위칭 제어 방법이 안정성 보장과 함께 가변 관성 하에서 주파수 제어 성능을 크게 향상시킴을 확인하였습니다.



### Fast and Modular Autonomy Software for Autonomous Racing Vehicles (https://arxiv.org/abs/2408.15425)
Comments:
          Published in Journal of Field Robotics

- **What's New**: 이 논문은 MIT-Pitt-RW 팀의 Indy Autonomous Challenge (IAC)에서의 자율 레이싱 접근법을 상세히 설명하고 있습니다. 팀은 모듈화된 빠른 에이전트 탐지, 동작 계획 및 제어를 통해 자율성 스택을 개발하였으며, 신속한 경쟁 환경에서의 소프트웨어 성능 분석을 제공합니다.

- **Technical Details**: 자동차 자율 기술의 발전을 위한 세 가지 주요 작업은 Sense, Think, Act입니다. ‘Sense’는 센서를 통해 환경 상태를 측정하며, ‘Think’는 센서 정보를 처리하여 향후 환경을 예측합니다. 마지막으로, ‘Act’는 결정된 행동을 실행합니다. 자율 레이싱 차량 (ARV)은 고속 환경에서 효율적으로 작동해야 하며, 이로 인해 저지연 및 장거리 인식이 필요합니다.

- **Performance Highlights**: IAC의 물리적 시스템 Dallara AV-21 플랫폼에서의 배포 결과 및 성과에 대한 분석을 제공하며, 실행된 기술들이 어떻게 작동했는지에 대한 통찰과 향후 개선 방향을 제시합니다.



### Simultaneous Training of First- and Second-Order Optimizers in Population-Based Reinforcement Learning (https://arxiv.org/abs/2408.15421)
Comments:
          8 pages, 5 figures

- **What's New**: 이 논문에서는 Population-Based Training (PBT) 기법을 활용하여 강화 학습 (Reinforcement Learning, RL)에서 하이퍼파라미터를 동시에 조정하는 새로운 방법론을 제안합니다. 특히, K-FAC와 Adam 두 가지 최적화 기법을 혼합하여 사용함으로써 학습 성능을 10% 향상시키는 결과를 보여줍니다.

- **Technical Details**: PBT는 훈련 과정 동안 하이퍼파라미터를 동적으로 조정하여 학습의 성능 및 안정성을 높이는 방법입니다. 이 연구에서는 TD3 알고리즘을 사용하여 다양한 MuJoCo 환경에서 실험을 수행하였으며, K-FAC 최적화 기법과 Adam을 통합하여 RL 성능을 높였습니다. 이는 첫째로, 단일 인구 내에서 다양한 최적화 기법을 활용하고, 둘째로, K-FAC의 정밀도를 통해 모델의 수렴 속도를 높이는 접근 방식을 택하고 있습니다.

- **Performance Highlights**: 실험 결과, K-FAC와 Adam의 조합을 사용했을 때, Adam만 사용했을 경우보다 전체 성능이 최대 10% 향상되었습니다. 또한, Adam이 간혹 실패하는 환경(예: Swimmer 환경)에서 K-FAC를 혼합한 인구는 더 안정적인 학습 결과를 보이는 등 보다 신뢰할 수 있는 훈련 결과를 제공합니다.



### Intertwined Biases Across Social Media Spheres: Unpacking Correlations in Media Bias Dimensions (https://arxiv.org/abs/2408.15406)
Comments:
          Accepted to ASONAM 2024

- **What's New**: 본 연구는 YouTube와 Reddit에서 수집된 새로운 데이터셋을 소개하며, 다양한 미디어 편향 차원에 대한 자동 주석을 제공합니다. 이 데이터셋은 지난 5년간의 콘텐츠를 포함하며, 정치, 스포츠, 건강, 교육, 오락 등 다섯 개의 주제 도메인에 걸쳐 있습니다.

- **Technical Details**: 본 연구에서는 기계 학습 (machine learning)과 자연어 처리 (natural language processing) 기술을 활용하여 미디어 편향을 식별하고, 여러 편향 차원 간 상관관계를 분석합니다. 데이터셋은 YouTube 댓글을 포함하여 다차원 미디어 편향 인식 저널 데이터세트를 만들기 위해 주석이 달렸습니다.

- **Performance Highlights**: 정치 도메인에서는 편향된 내용의 비율이 다른 도메인에 비해 유의미하게 높은 것으로 나타났습니다. 이 분석을 통해 데이터의 시간적 변화에 따른 편향 차원 간의 상관관계가 동적으로 변동함을 발견했습니다. 이러한 통찰력은 미래의 미디어 편향 탐지 시스템의 주춧돌을 마련합니다.



### A Statistical Framework for Data-dependent Retrieval-Augmented Models (https://arxiv.org/abs/2408.15399)
- **What's New**: 이 논문에서는 retrieval-augmented models (RAMs)의 훈련 및 기본 속성과 성능에 대한 이해를 제공하기 위한 통계적 프레임워크를 제안합니다. RAM은 {	extit{retriever}}와 {	extit{predictor}}라는 두 가지 구성 요소로 이루어져 있어 훈련과 예측 과정에서 중요한 역할을 합니다.

- **Technical Details**: 저자들은 RAM의 end-to-end 훈련을 위한 공식적인 목표 함수를 제시하며, 이는 일반화 및 표현력을 분석하는 데 중요한 역할을 합니다. retriever와 predictor의 상호작용을 포착하고, 데이터 저장소(data-store)의 크기와 RAM의 최종 성능 간의 관계를 명확히 합니다. 또한, excess risk bound를 도출하여 RAM의 성능을 향상시키는 역할을 설명합니다.

- **Performance Highlights**: 제안한 방법은 NaturalQuestions(NQ) 및 TriviaQA와 같은 표준 QA 벤치마크에서 효과적임을 입증하였으며, RAM이 효율적인 성능과 계산 요구 사이의 균형을 잘 맞춘다는 것을 보여줍니다.



### SCAN-Edge: Finding MobileNet-speed Hybrid Networks for Diverse Edge Devices via Hardware-Aware Evolutionary Search (https://arxiv.org/abs/2408.15395)
- **What's New**: 본 논문에서는 다양한 저비용 엣지 디바이스에 최적화된 하이브리드 네트워크를 설계하기 위한 새로운 통합 기반의 하드웨어 인식 신경망 아키텍처 검색(NAS) 프레임워크인 SCAN-Edge를 제안합니다.

- **Technical Details**: SCAN-Edge는 CPU, GPU, 하드웨어 가속기 기반 시스템을 포함한 다양한 엣지 디바이스에 맞춤형 하이브리드 네트워크를 검색하며, 주요 연산인 self-attention, convolution 및 activation을 통합하여 검색합니다. 또한, ECC(complexity), FLOPs 등의 일반적인 모델 복잡도 수치를 기반으로 한 기존의 접근 방식과 달리, 실제 엣지 디바이스에서의 레이턴시를 정확히 반영하는 캘리브레이티드 레이턴시 룩업 테이블(LUTs)을 사용합니다.

- **Performance Highlights**: 실험 결과, SCAN-Edge로 발견된 하이브리드 네트워크는 다양한 엣지 디바이스에서 MobileNetV2의 실제 레이턴시와 일치하며, 이전 접근 방식에 비해 우수한 정확도를 제공합니다.



### Handling Geometric Domain Shifts in Semantic Segmentation of Surgical RGB and Hyperspectral Images (https://arxiv.org/abs/2408.15373)
Comments:
          Silvia Seidlitz and Jan Sellner contributed equally

- **What's New**: 이 논문은 기존의 성능 높은 semantic segmentation 모델이 기하학적 도메인 변화(geometric domain shifts)에 직면했을 때의 영향을 분석하고, 'Organ Transplantation'이라는 새로운 데이터 증강 기법을 제안한다.

- **Technical Details**: 제안된 방법은 33마리의 돼지에서 수집된 600개의 RGB 및 hyperspectral imaging (HSI) 큐브를 사용하여 19개 클래스로 주석 처리된 6개의 OOD 데이터셋에서 검증되었다. 기하학적 OOD 데이터에서 SOA 모델의 성능은 RGB 데이터에서 46%, HSI 데이터에서 45%로 감소하였다.

- **Performance Highlights**: 조직의 기하학적 도메인 변화에 대응하는 데 효과적인 'Organ Transplantation' 증강 기법을 통해 RGB 데이터에서 최대 67%, HSI 데이터에서 90%의 성능 향상이 이루어졌으며, 실제 OOD 테스트 데이터에서의 in-distribution 성능 수준에 도달하였다.



### What makes math problems hard for reinforcement learning: a case study (https://arxiv.org/abs/2408.15332)
Comments:
          39 pages, 18 figures, 1 table

- **What's New**: 본 논문은 Andrews-Curtis 추측에 대한 새로운 알고리즘적 접근 방식을 제안하며, 희귀한 사례에서 불균형적으로 높은 보상을 찾는 문제를 다룹니다. 이 과정에서 저자들은 수학적 문제 해결을 위한 강화 학습(reinforcement learning) 기술의 적용 가능성을 탐구하고 있습니다.

- **Technical Details**: 저자들은 RL 모델이 문제의 난이도를 평가할 수 있도록 설계하고, 이를 통해 문제 분포의 난이도를 학습하여 기존 알고리즘의 자기 개선(self-improvement) 전략을 발전시킵니다. 특별히, 본 연구에서는 Proximal Policy Optimization (PPO) 알고리즘을 사용하며, 강화 학습을 이용한 사례 연구가 포함되어 있습니다. 또한, 고전적인 탐색 알고리즘과 비교하여 기계 학습 성능을 평가합니다.

- **Performance Highlights**: 본 연구의 결과, 강화 학습이 기초 탐색 알고리즘보다 나은 성과를 보였으나, 가장 뛰어난 성능을 발휘한 것은 탐욕적 알고리즘(greedy search)으로 나타났습니다. 이러한 성과는 저렴한 계산 자원으로도 이뤄질 수 있었으며, 이론적 도구와 기술을 통합하여 앞으로의 연구에서 더 큰 발전이 가능함을 시사합니다.



### Parameter-Efficient Quantized Mixture-of-Experts Meets Vision-Language Instruction Tuning for Semiconductor Electron Micrograph Analysis (https://arxiv.org/abs/2408.15305)
Comments:
          Paper published at ICML 2024 Workshop on Foundation Models in the Wild

- **What's New**: 본 논문은 반도체 제조를 위한 소규모 비전-언어 어시스턴트인 sLAVA를 소개합니다. 이 시스템은 전자 현미경 이미지 분석에 중점을 두고 있으며, 데이터 부족 및 전문가 주석이 있는 고품질 데이터 확보 문제를 해결합니다.

- **Technical Details**: sLAVA는 teacher-student paradigm을 사용하여, GPT-4와 같은 기본 비전 언어 모델을 교사로 활용하여 지침을 따르는 다중 모드 데이터를 생성합니다. 이 모델은 커스터마이징된 학습을 통한 전자 현미경 이미지 분석 작업을 지원합니다. 이를 통해 기업들이 자사 데이터로 프레임워크를 세밀하게 조정할 수 있으며, 지적 재산권 보호 또한 가능합니다.

- **Performance Highlights**: 엄격한 실험 결과, sLAVA 프레임워크는 전통적인 방법들을 초월하고, 데이터 변화에 효과적으로 대응하며, 고처리량 스크리닝을 가능하게 함을 입증했습니다.



### The Uniqueness of LLaMA3-70B with Per-Channel Quantization: An Empirical Study (https://arxiv.org/abs/2408.15301)
- **What's New**: LLaMA3-70B 모델은 W8A8(Post-Training Quantization)에서 독특한 정확도 감소 현상을 보이고 있으며, 이는 LLaMA2-70B 및 다른 모델들과는 대조적입니다. 본 연구는 W8A8 기반의 양자화가 LLaMA3-70B 모델의 가중치 분포의 차이로 인해 발생하는 문제를 제시합니다.

- **Technical Details**: 본 연구에서는 LLaMA3-70B 시리즈가 W8A8 양자화에 대해 유일하게 취약하다는 점을 실증적으로 조사하였습니다. 특정 입력 차원의 가중치가 다른 차원에 비해 몇 배 더 큰 경우가 있으며, 이러한 가중치의 분포가 양자화의 정밀도를 저하시킬 수 있습니다. 이를 해결하기 위해, 우리는 고유 가중치가 많은 2-3%의 레이어에 대해 그룹 양자화를 적용하고, 나머지 97%는 채널별 양자화를 유지하는 혼합 전략을 제안합니다.

- **Performance Highlights**: LLaMA3-70B 모델의 평균 정확도는 W8A8을 적용한 후 45.5%에서 73.4%로 증가하여, FP16에 비해 단 0.7%의 차이만 나타났습니다. 이 방법은 보정(calibration)이나 미세 조정(fine-tuning)을 요구하지 않습니다.



### GIFT-SW: Gaussian noise Injected Fine-Tuning of Salient Weights for LLMs (https://arxiv.org/abs/2408.15300)
- **What's New**: 이 논문에서는 Gaussian noise를 주입하여 중요한 가중치 (salient weights)를 업데이트하고 비중요 가중치에는 노이즈를 주입하는 새로운 PEFT 방법인 GIFT-SW를 소개합니다. 기존의 이론을 바탕으로 모든 기존 민감도 지표(sensitivity metric)를 일반화하여 새로운 공식을 도출하였습니다.

- **Technical Details**: GIFT-SW는 주요 가중치에 해당하는 열을 업데이트하면서 비주요 가중치에는 Gaussian 노이즈를 주입하는 방법입니다. 이 방법은 Perturbed Gradient Descent (PGD) 기법을 사용하여 수렴을 안정화하고 과적합(overfitting)을 방지하는데 도움을 줍니다. 저자들은 LLaMA 모델을 통해 GIFT-SW의 성능을 검증하였고, GIFT-SW가 현대 PEFT 방법과 전체 미세조정(full fine-tuning) 대비 우수함을 입증했습니다.

- **Performance Highlights**: 실험 결과에 따르면, GIFT-SW는 총 3%의 가중치만 조정하고도 기존의 최첨단 모델들과 유사한 정확도를 달성하였으며, 컴퓨팅 자원은 10배 더 적게 사용했습니다. 또한, GIFT-SW는 mixed-precision quantization 후 성능 회복에 있어 실질적 장점을 제공합니다.



### TourSynbio: A Multi-Modal Large Model and Agent Framework to Bridge Text and Protein Sequences for Protein Engineering (https://arxiv.org/abs/2408.15299)
- **What's New**: 이번 논문에서는 외부 단백질 인코더 없이 단백질 공학 작업을 위해 특별히 설계된 최초의 다중 모달 대형 모델인 TourSynbio-7B를 소개합니다. 이 모델은 LLM이 단백질을 언어로 이해할 수 있는 능력을 자연스럽게 학습할 수 있음을 보여주고 있습니다.

- **Technical Details**: TourSynbio-7B는 17.46억 토큰의 텍스트 및 단백질 시퀀스 데이터로 구성된 ProteinLMDataset을 사용하여 후속 훈련 및 지침 세분화를 수행했습니다. 박테리아 바이오 합성에 필요한 다양한 단백질 엔지니어링 작업, 예를 들어 변이 분석, 역 접기(protein folding), 단백질 접기 및 시각화를 수행할 수 있는 TourSynbio-Agent를 통해 단백질 공학 작업에 효율성 및 성능을 높였습니다.

- **Performance Highlights**: TourSynbio-7B는 944개의 수동 검증된 다지선다형 질문을 포함하는 ProteinLMBench에서 62.18%의 정확도로 GPT-4를 초월하는 성과를 보였습니다. 또한, 두 개의 웨트랩(case studies)에서 기존의 단백질 엔지니어링 작업 개선 사례를 증명하였습니다.



### YOLO-Stutter: End-to-end Region-Wise Speech Dysfluency Detection (https://arxiv.org/abs/2408.15297)
Comments:
          Interspeech 2024

- **What's New**: 본 논문에서는 비유창한 언어( dysfluent speech ) 검출을 위한 최초의 엔드 투 엔드 방법인 YOLO-Stutter를 제안합니다. 이 방법은 시간을 정확하게 조정하여 비유창한 언어를 탐지합니다.

- **Technical Details**: YOLO-Stutter는 비정상적인 발화-텍스트 정렬을 입력으로 받아, 공간적 특징 집합체(spatial feature aggregator) 및 시간적 의존성 추출기(temporal dependency extractor)를 통해 지역별 경계(boundary)와 클래스(class) 예측을 수행합니다. 또한, VCTK-Stutter 및 VCTK-TTS라는 두 개의 비유창성 데이터셋을 도입하여, 자연스러운 비유창성을 시뮬레이션합니다.

- **Performance Highlights**: YOLO-Stutter는 최소한의 학습 가능 매개변수로도 시뮬레이션된 데이터와 실제 실어증(aphasia) 언어에서 최첨단 성능을 달성했습니다. 이 성과는 38명의 실어증 환자와의 협업을 통해 이루어진 데이터 분석 결과를 통해 입증되었습니다.



### Evaluating the Predictive Features of Person-Centric Knowledge Graph Embeddings: Unfolding Ablation Studies (https://arxiv.org/abs/2408.15294)
Comments:
          Published in the 34th Medical Informatics Europe Conference

- **What's New**: 이 논문에서는 MIMIC-III 데이터셋에서 구조화된 정보와 비구조화된 정보를 모두 활용하여 Graph Neural Network (GNN) 모델의 성능을 체계적으로 평가하는 접근 방식을 제안하고 있습니다. 연구자들은 인구통계학적 변인이나 사회적 요인 등을 포함해 여러 지표의 영향력을 분석하여 재입원 예측을 위한 개인 중심 데이터 모델을 강화하고자 했습니다.

- **Technical Details**: 저자들은 GNN 기반의 예측 모델을 통해 개인 중심의 지식 그래프(PKGs)를 생성하고 분석하였습니다. MIMIC-III 데이터셋을 사용하여 데이터 미선택, 품질 평가, 비구조화 텍스트 마이닝 등의 전처리 단계를 거쳐 개별 환자 입원에 대한 그래프를 작성합니다. 메타 정보가 포함된 PKGs를 통해 인구 통계학적, 임상적 및 사회적 노드를 연결하여 GNN을 훈련시켰습니다.

- **Performance Highlights**: Ablation study(소거 연구)를 통해 사회적 요인 특히 결혼 상태 및 인종의 중요성을 강조하며, 이들 변수의 제거가 모델 성능에 미치는 영향을 확인했습니다. 임상적 파라미터 또한 예측 성능에 중요한 역할을 하며, 특정 질병 및 투약에 대한 통합 정보가 재입원 예측에 중요한 기여를 한다는 사실을 발견했습니다. 전체 질환 정보의 제거가 성능 지표에서 최대 4.13%의 정확도 감소를 초래하였고, 임상 정보가 완전히 제외될 경우 정확도 및 F1 점수에서 각각 최대 18.24%와 8.37%의 성능 저하가 발생했습니다.



### Learning Granularity Representation for Temporal Knowledge Graph Completion (https://arxiv.org/abs/2408.15293)
Comments:
          15 pages. Accepted at ICONIP 2024

- **What's New**: 이번 연구에서는 Temporal Knowledge Graphs (TKGs)의 완전성을 향상시키기 위한 새로운 방법인 Learning Granularity Representation (LGRe)를 제안합니다. LGRe는 시간의 세부 granularity(세분성)를 고려하여 TKG의 결측 정보를 예측합니다.

- **Technical Details**: LGRe는 Granularity Representation Learning (GRL)과 Adaptive Granularity Balancing (AGB)라는 두 가지 주요 구성 요소로 이루어져 있습니다. GRL은 시간 특화된 다층 convolutional neural networks (CNNs)를 사용하여 다양한 granularity 간의 상호작용을 캡처하고, AGB는 시간 의미에 따라 이러한 표현의 가중치를 조정하여 예측의 표현력을 높입니다.

- **Performance Highlights**: 4개의 이벤트 벤치마크에서의 실험 결과, LGRe는 최신 TKG 완전성 방법론에 비해 우수한 성능을 보였으며, 시간 관련 표현 학습에 있어서 그 효과를 입증했습니다.



### A Survey of Deep Learning for Group-level Emotion Recognition (https://arxiv.org/abs/2408.15276)
Comments:
          16 pages, 2 figures

- **What's New**: 이 논문은 깊이 있는 심층 학습(Deep Learning) 기법을 이용한 집단 수준의 감정 인식(Group-level Emotion Recognition, GER)에 대한 포괄적인 리뷰를 제공합니다. 기존의 리뷰와는 달리, 깊이 학습 아키텍처에 중점을 두고 현재 상황과 기술적 도전 과제를 설명하며, 집단 감정을 다루는 새로운 분류 체계를 제안합니다.

- **Technical Details**: 이 논문은 이미지 및 비디오 기반의 집단 감정 데이터베이스를 소개하고, 최근 개발된 심층 학습 방법과 이와 관련된 기술적 도전 과제를 논의합니다. 또한, 감정 인식을 위한 데이터 추출의 세 단계를 설명하며, 손으로 설계된 특성(descriptor) 및 신경망(neural networks)을 활용한 얼굴 특징 추출에 대해 조사합니다.

- **Performance Highlights**: 이 논문은 GIS(Ground Image Survey) 분야에서의 최신 기술 비교를 제공하고, 과거 10년간의 최고 성능(methods)의 성능 특성을 분석합니다. 또한, 향후 연구 방안과 응용을 위한 귀중한 통찰을 제공합니다.



### Anomaly Detection in Time Series of EDFA Pump Currents to Monitor Degeneration Processes using Fuzzy Clustering (https://arxiv.org/abs/2408.15268)
Comments:
          This paper has been accepted to the IEEE International Conference on Machine Learning for Communication and Networking (ICMLCN) 2024

- **What's New**: 이번 논문은 EDFA 시스템에서 펌프 전류 시계열을 위한 새로운 퍼지 클러스터링 기반 이상 탐지 방법을 제안합니다. 이 방법은 엔트로피 분석(Entropy Analysis, EA)과 주성분 분석(Principal Component Analysis, PCA)의 장점을 결합한 변화 탐지 프레임워크(Change Detection Framework, CDF)를 전략적으로 적용합니다.

- **Technical Details**: 제안된 프레임워크에서는 EA를 적용해 특징 선택을 동적으로 수행하며, PCA를 통해 원시 특징 공간에서 특징을 추출합니다. 이후 퍼지 클러스터링 절차를 통해 세 가지 다른 퍼지 클러스터링 방법, 즉 퍼지 클러스터링 알고리즘(Fuzzy Clustering Algorithm), 확률적 클러스터링 알고리즘(Probabilistic Clustering Algorithm), 가능적 클러스터링 알고리즘(Posibilistic Clustering Algorithm)을 평가하여 성능과 일반화를 분석합니다.

- **Performance Highlights**: 제안된 프레임워크는 상업적으로 사용되는 EDFAs의 미리 정의된 알람 시스템과 비교하여 임의의 작업 지점에서 펌프 전류 시계열의 변화를 조기에 감지할 수 있는 혁신적인 기능을 가지고 있으며, 실험 데이터를 사용하여 구현 및 테스트되었습니다. 또한 이 접근 방식은 광섬유 네트워크에 대한 분산 예측 유지보수(Decentralized Predictive Maintenance)를 적용하는 추가 접근 방식을 가능하게 합니다.



### People over trust AI-generated medical responses and view them to be as valid as doctors, despite low accuracy (https://arxiv.org/abs/2408.15266)
- **What's New**: 본 논문은 비전문가가 AI로 생성된 의료 응답을 어떻게 인식하고 평가하는지에 대한 포괄적인 분석을 제공합니다. 300명의 참가자가 온라인 헬스케어 플랫폼에서 의사가 작성한 응답과 일반 언어 모델(large language model, LLM)에 의해 생성된 응답을 평가했습니다.

- **Technical Details**: 연구에서는 150개의 익명 의료 질문과 6개의 의료 분야에 걸친 의사 응답을 수집하고, AI가 생성한 응답의 정확성을 평가하기 위해 4명의 의사가 평가하였습니다. 평가 결과에 따라 두 가지 카테고리(High Accuracy와 Low Accuracy)로 분류했습니다. 참가자들은 AI 생성 응답에 대한 신뢰성, 유효성, 완전성/만족도를 평가했습니다.

- **Performance Highlights**: 참가자들은 AI가 생성한 고 정확도 응답이 의사 응답보다 유효하고 신뢰할 수 있다고 평가했으며, 저 정확도 AI 응답도 유사하게 의사 응답과 비슷하거나 더 나은 성과를 보였습니다. 이로 인해 비전문가들이 부정확한 AI 의료 조언을 신뢰할 경우 잘못된 진단과 해로운 결과를 초래할 수 있음을 강조합니다.



### S4DL: Shift-sensitive Spatial-Spectral Disentangling Learning for Hyperspectral Image Unsupervised Domain Adaptation (https://arxiv.org/abs/2408.15263)
- **What's New**: 이 논문에서는 하이퍼스펙트럼 이미지(HSI) 분류를 위한 새로운 방법론인 Shift-Sensitive Spatial-Spectral Disentangling Learning (S4DL)을 제안합니다. S4DL은 도메인 불변 특성과 도메인 특정 특성을 효과적으로 분리하여 다양한 데이터셋에서 성능을 향상시키고자 합니다.

- **Technical Details**: S4DL은 gradient-guided spatial-spectral decomposition 기법을 통해 각각의 채널에서 도메인 정보를 정량화하고, tailored masks를 생성해 도메인 불변 및 특정 채널을 분리하는 방식으로 작동합니다. 또한, shift-sensitive adaptive monitor가 도메인 간 변동을 지속적으로 모니터링하여 모델의 정렬 전략을 동적으로 조정합니다. Reversible feature extractor (RFE)는 저수준 특징도 함께 보존하여 도메인 정보를 유지합니다.

- **Performance Highlights**: 여러 개의 교차 장면 HSI 데이터셋에서 수행된 실험 결과, S4DL은 기존의 최신 UDA 방법들보다 뛰어난 성능을 보였으며, 이를 통해 다양한 도메인 간의 전이 가능성을 향상시켰습니다.



### Civiverse: A Dataset for Analyzing User Engagement with Open-Source Text-to-Image Models (https://arxiv.org/abs/2408.15261)
- **What's New**: 본 연구는 TTI (Text-to-Image) AI 플랫폼 CivitAI를 분석하여 문화적 관점에서 오픈소스 TTI 프레임워크를 체계적으로 조사합니다. 특히, 사용자 의도와 행동을 파악하기 위해 Civiverse 프롬프트 데이터셋을 소개하며, 이는 수백만 개의 이미지와 관련 메타데이터를 포함합니다.

- **Technical Details**: 대규모 데이터셋 Civiverse 6M을 사용하여 텍스트 프롬프트의 의미적 특성을 분석하였습니다. 이 데이터셋은 사용자가 제공한 프롬프트를 기반으로 생성된 6,546,165개의 이미지 URL과 메타데이터로 구성되어 있습니다. 분석은 사용자와 TTI 모델 간의 상호작용 패턴에 초점을 맞추어 진행되었습니다.

- **Performance Highlights**: 연구 결과, 성적 콘텐츠 생성에 대한 주된 선호가 나타났으며, 의미적 내용의 동질화 경향이 확인되었습니다. 이러한 통찰은 그림을 생성하는 모델 내에서 여성 혐오와 해로운 고정관념이 지속될 가능성을 강조하며, 문화적 다양성이 감소하고 있음을 나타냅니다.



### Transformer-based Neuro-Animator for Qualitative Simulation of Soft Body Movemen (https://arxiv.org/abs/2408.15258)
Comments:
          12 pages, 3 figures

- **What's New**: 이 논문은 최근의 transformer 아키텍처를 활용하여 인공 신경망(ANN)을 통해 인간의 직관적인 물리적 사건의 시뮬레이션 능력을 연구합니다. 특히, 바람에 흔들리는 깃발의 움직임을 시뮬레이션하는 방법에 대해 논의합니다.

- **Technical Details**: visial transformer 모델은 소프트 바디 물체인 깃발의 동작을 

t-n ... t 시각에서의 정보를 바탕으로 t+1 시각의 동작을 예측하도록 훈련됩니다. 이 모델은 2D 플래그의 정점 움직임을 기반으로 한 감독 학습을 통해 훈련되며, 바람의 힘과 중력에 의해 영향을 받는 3D 공간의 깃발 움직임을 포착합니다.

- **Performance Highlights**: 결과적으로, visual transformer 기반 아키텍처는 소프트 바디 물체의 움직임에 대한 시간적 임베딩을 성공적으로 학습하고, 다양한 바람 세기에 따른 깃발의 흔들림을 적절하게 재현하는 시뮬레이션을 생성하는 데 성공했습니다.



### Text classification optimization algorithm based on graph neural network (https://arxiv.org/abs/2408.15257)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2405.17460 by other authors

- **What's New**: 본 논문에서는 그래프 신경망(Graph Neural Networks, GNN)을 활용한 텍스트 분류 최적화 알고리즘을 소개합니다. 이 알고리즘은 적응형 그래프 구성 전략과 효율적인 그래프 컨볼루션(Convolution) 연산을 도입하여 텍스트 분류의 정확도와 효율성을 크게 향상시킵니다.

- **Technical Details**: 전통적인 텍스트 분류 방법은 bag of words 모델이나 TF-IDF 같은 특성 표현에 의존하며, 단어 간의 의미적 연결을 간과합니다. 본 연구에서는 GNN을 통해 비 유클리드 데이터(non-Euclidean data)를 효율적으로 처리할 수 있는 가능성을 탐구하며, 그래프 구조 생성의 복잡성과 모델 학습의 높은 비용 문제를 해결합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 전통적인 접근법 및 기존 GNN 모델보다 여러 공공 데이터셋에서 뛰어난 성능을 보여주며, 텍스트 분류 작업에 대한 우수성과 실현 가능성을 강조합니다.



### Improving Ontology Requirements Engineering with OntoChat and Participatory Prompting (https://arxiv.org/abs/2408.15256)
- **What's New**: 이번 연구에서는 대규모 프로젝트에서 사용자 요구를 효과적으로 수집하기 위해 LLMs(대형 언어 모델)를 활용하는 Ontology Requirements Engineering (ORE)의 새로운 접근 방법인 OntoChat을 소개합니다. 특히 사용자가 챗봇에게 사용자 이야기를 생성하도록 유도하는 과정에서 효과적인 참여 촉구(participatory prompting) 방법을 적용하여 초기 평가에서 드러난 사용자 경험을 개선하고 있습니다.

- **Technical Details**: 연구 팀은 사용자 쿼리를 바탕으로 미리 정의된 프롬프트 템플릿을 개발하여 사용자 이야기를 정제하는 과정에 참여하도록 하였습니다. 이 접근법은 사용자가 LLM을 더 효과적으로 활용할 수 있도록 지원하며, 사용자 이야기를 생성하는 단계에서 페르소나(persona), 목표(goal), 시나리오(scenario), 예제 데이터(example data) 및 데이터 리소스(data resources) 등을 포함한 다양한 요소들을 다룹니다. 이 과정은 사용자가 초기 사용자 이야기를 만들고 피드백을 통해 이를 반복적으로 개선하는 구조로 이루어졌습니다.

- **Performance Highlights**: 사전 정의된 프롬프트 템플릿의 사용을 통해 사용자들은 자신의 요구를 더 명확하고 구체적으로 표현할 수 있었고, 이는 최종 사용자 이야기를 작성하는 데 있어 더 유용하고 영감을 주는 결과를 가져왔습니다. 또한, LLM이 생성한 응답에 대한 참가자들의 평가를 통해 정보의 유용성, 명확성 및 실용성을 높이는 데 기여하였습니다.



### Generative AI on SpectrumNet: An Open Benchmark of Multiband 3D Radio Maps (https://arxiv.org/abs/2408.15252)
Comments:
          30 pages, 15 figures

- **What's New**: SpectrumNet는 다양한 지형 및 기후 정보를 고려한 다중 대역 3차원 라디오 맵 데이터셋으로, 이 분야에서 가장 큰 규모를 자랑합니다. 이 데이터셋은 3개의 공간 차원, 5개의 주파수 대역, 11개의 지형 시나리오 및 3개의 기후 시나리오를 포함하고 있습니다.

- **Technical Details**: SpectrumNet은 30만 개 이상의 라디오 맵 이미지를 포함하고 있으며, 다양한 지형, 날씨 및 물체 재질 정보를 고려하여 생성되었습니다. 이 데이터셋은 3가지 높이 수준에서의 라디오 맵을 포함하는 최초의 3D 데이터셋입니다. 또한, 라디오 맵 구축을 위한 3개의 기준 방법에 대한 SSIM, PSNR, RMSE 성능 평가가 포함되어 있습니다.

- **Performance Highlights**: SpectrumNet 데이터셋은 실험 결과를 통해 다양한 지형, 주파수 및 높이에서 확장 가능한 성능을 입증하였으며, 향후 데이터셋 확장 및 품질 향상 방향에 대한 연구가 논의되었습니다.



### AI-Powered Camera and Sensors for the Rehabilitation Hand Exoskeleton (https://arxiv.org/abs/2408.15248)
- **What's New**: 이 논문은 마비된 사람들의 손 운동을 도와주는 비전 기반의 재활 핸드 엑소스켈레톤에 대한 연구를 소개합니다. 이 디자인은 사용자 교육이 필요 없는 간단한 인터페이스를 목표로 하여 접근 가능한 도구를 만드는 것에 초점을 맞추었습니다.

- **Technical Details**: 이 프로토타입은 상업적으로 이용 가능한 장갑 위에 카메라와 내장 프로세서를 통합하여, 공기를 이용해 손을 열고 닫으며 물체를 잡게 해줍니다. 또한, 가속도계(Accelerometer)를 통해 손 제스처를 감지하고, 사용자가 원할 때 물체를 놓을 수 있도록 설계되었습니다. 이러한 수동 비전 기반 제어(Passive Vision-based Control)는 개별 훈련이 필요없는 EMG(active electromyography) 기반 설계와 차별화됩니다.

- **Performance Highlights**: 이 연구는 비용, 무게 및 전력 소비를 줄여 대량 생산(Mass Implementation)을 가능하게 할 계획을 가지고 있어, 다양한 사용자가 혜택을 받을 수 있는 가능성을 제시합니다.



### AutoGen Studio: A No-Code Developer Tool for Building and Debugging Multi-Agent Systems (https://arxiv.org/abs/2408.15247)
Comments:
          8 pages

- **What's New**: 다중 에이전트 시스템(Multi-agent systems)이 여러 도메인에서 복잡한 문제를 해결하기 위해 협력하는 효과적인 패턴으로 부상하고 있습니다. 이와 같은 시스템을 간편하게 프로토타입, 디버그 및 평가할 수 있도록 해주는 코드 없는 개발 도구인 AUTOGEN STUDIO가 소개되었습니다.

- **Technical Details**: AUTOGEN STUDIO는 LLM 활성화 에이전트를 선언적(JSON 기반) 사양으로 표현할 수 있는 웹 인터페이스와 Python API를 제공합니다. 이 도구는 에이전트 워크플로우 사양을 위한 직관적 드래그 앤 드롭 UI, 워크플로우의 대화형 평가 및 디버깅 도구, 재사용 가능한 에이전트 구성 요소 갤러리를 제공합니다.

- **Performance Highlights**: AUTOGEN STUDIO는 5개월 동안 20만 다운로드를 기록하며 오픈 소스 도구로서의 경험을 바탕으로 다중 에이전트 개발 도구의 새로운 디자인 패턴과 미래 연구 방향을 제시합니다.



### Multi-Slice Spatial Transcriptomics Data Integration Analysis with STG3N (https://arxiv.org/abs/2408.15246)
- **What's New**: 본 논문에서는 Spatially Resolved Transcriptomics (SRT) 데이터의 다중 슬라이스 분석에서 발생하는 배치 효과를 완화하기 위한 새로운 방법론인 STG3Net을 제안했습니다. 이를 통해 기존 기법과 비교해 더 우수한 성능을 달성했습니다.

- **Technical Details**: STG3Net은 masked graph convolutional autoencoders를 백본 모듈로 사용하며, generative adversarial learning을 통합하여 다중 슬라이스 공간 영역 식별 및 배치 보정을 효과적으로 수행합니다. Global Nearest Neighbor (G2N) 방식으로 슬라이스 간의 대표적인 anchor pairs를 선택하여 배치 효과를 저감합니다.

- **Performance Highlights**: STG3Net은 세 가지 서로 다른 SRT 데이터 세트에서 F1LISI 메트릭을 포함한 정확도 및 일관성을 고려하여 평가한 결과, 기존 방법들과 비교하여 가장 뛰어난 성능을 보였으며, 멀티 슬라이스 간의 생물학적 변동성과 연결성을 유지했습니다.



### An Edge AI System Based on FPGA Platform for Railway Fault Detection (https://arxiv.org/abs/2408.15245)
Comments:
          Accepted at the 2024 IEEE 13th Global Conference on Consumer Electronics (GCCE 2024)

- **What's New**: 본 연구는 FPGA(최대 재구성 게이트 배열) 기반의 철도 점검 시스템을 도입하여 효율적이고 신뢰성 있는 레일 결함 탐지를 수행하는 새로운 엣지 AI 시스템을 제안합니다.

- **Technical Details**: 제안된 시스템은 카메라, FPGA 플랫폼, ESP8266 모듈, 그리고 그래픽 사용자 인터페이스가 장착된 컴퓨터로 구성되어 있으며, 카메라를 통해 레일 이미지를 캡처하고 CNN(합성곱 신경망)을 통해 실시간으로 결함을 탐지합니다. 경량화된 신경망을 사용하여 88.9%의 정확도를 달성하며, FPGA의 에너지 효율성은 GPU와 CPU 대비 각각 1.39배, 4.67배 더 높습니다.

- **Performance Highlights**: 실험 결과, 본 시스템은 GPU와 CPU에서의 동일한 네트워크 구현보다 뛰어난 에너지 효율성을 보여주었습니다. 또한, FPGA를 이용한 시스템은 데이터 처리를 신속하게 수행하며, 실시간 분석에 적합한 성능을 발휘합니다.



### Misrepresented Technological Solutions in Imagined Futures: The Origins and Dangers of AI Hype in the Research Community (https://arxiv.org/abs/2408.15244)
Comments:
          Accepted to AIES 2024

- **What's New**: 이 논문은 AI 기술의 과장된 주장 및 그로 인한 위험성을 강조합니다. 연구 커뮤니티가 기술 하이프(hype)의 발전과 확산에 중심적 역할을 하고 있으며, 이에 따른 잘못된 정책과 접근 방식을 제안합니다.

- **Technical Details**: AI hype는 실증적(empirical) 또는 이론적(theoretical) 지원이 결여된 성능 주장을 포함합니다. 이 논문에서는 대규모 언어 모델(LLMs)과 생성적 AI의 능력에 대한 과장된 주장 및 논란을 다룹니다. LLM들이 특정 벤치마크 데이터 세트에서 높은 성능을 보이지만, 일반화 능력은 한계가 있으며, 그 성능이 실제로 삶에 미치는 영향은 신뢰할 수 없습니다.

- **Performance Highlights**: 대부분의 AI 응용 프로그램은 심각한 한계가 있으며, 이로 인해 편향(bias)의 문제가 발생합니다. 예를 들어, LLM들은 주어진 정보에 근거한 잘못된 합리적 결과를 만들어내며, 생성 AI는 스페셜 관계를 잘못 해석하는 등의 문제를 보입니다. 이러한 문제점을 알 수록 AI 기술에 대한 현실적인 기대 수준을 조정할 필요성이 커지고 있습니다.




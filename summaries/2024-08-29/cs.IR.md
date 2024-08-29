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




### SpaceByte: Towards Deleting Tokenization from Large Language Modeling (https://arxiv.org/abs/2404.14408)
Comments: 9+9 pages, 3+1 figures, 2+4 tables

- **What's New**: SpaceByte는 기존의 토크나이제이션(tokenization) 모델의 단점을 극복하고자 개발된 새로운 바이트 레벨 디코더 아키텍처입니다. 이 아키텍처는 바이트 레벨 변환 모델(byte-level Transformer model)을 기반으로 하지만, 중간에 보다 큰 변환 블록(larger transformer blocks)을 삽입하여 성능을 향상시키는 것이 특징입니다. 고정된 훈련 및 추론 계산 예산을 사용하는 경우, SpaceByte는 다른 바이트 레벨 아키텍처를 능가하고 토크나이징된 변환 아키텍처의 성능과 대략 매치합니다.

- **Technical Details**: SpaceByte는 통상의 문자 경계를 나타내는 공백 문자와 같은 특정 바이트 후에만 큰 블록을 적용하는 방식으로 설계되었습니다. 이는 byte를 단어 및 기타 언어 경계와 일치하는 패치로 동적으로 분할하는 간단한 규칙을 사용합니다. 실험 결과, 이러한 변형은 SpaceByte가 다양한 텍스트 모달리티에서 다른 바이트 레벨 아키텍처를 능가하는 데 결정적으로 중요한 것으로 나타났습니다. 예를 들어, 영어 책, LaTeX로 포맷된 arXiv 논문 및 오픈소스 코드 데이터 세트에서 테스트되었습니다.

- **Performance Highlights**: SpaceByte는 고정된 훈련 및 추론 컴퓨트 비용을 고려한 상태에서 바이트 당 비트 크로스 엔트로피(cross entropy measured in bits-per-byte)를 기준으로 성능을 측정하고 있습니다. SpaceByte는 기존의 바이트 레벨 Transformer 및 MegaByte 모델이 서브워드 레벨 Transformer와 동일한 성능을 달성하기 위해 필요한 훈련 FLOPs의 약 10배가 필요한 것으로 나타내고 있습니다.



### RTP-LX: Can LLMs Evaluate Toxicity in Multilingual Scenarios? (https://arxiv.org/abs/2404.14397)
Comments: Work in progress

- **What's New**: 새로운 RTP-LX 데이터세트는 다양한 문화적 맥락에서 독성(toxic) 또는 해로울 수 있는 언어를 판별하는 것을 목표로 하고 있습니다. 이 데이터세트는 28개 언어로 구성되어 있고 인간이 직접 번역하고 주석을 달아 만들었습니다. 이를 통해 S/LLMs(Small/Large Language Models)가 다양한 언어에서 독성 판단을 얼마나 잘 수행하는지를 평가합니다.

- **Technical Details**: RTP-LX는 총 1,000개의 프롬프트(prompt)를 포함하고 있으며, 이는 원래 RTP(Real Toxicity Prompts) 데이터세트에서 유래했습니다. 이 프롬프트들은 인간 번역가와 현지 문화에 정통한 사람들의 참여적 디자인(participatory design) 접근법을 거쳐 특정 문화권에 맞게 재창조되었습니다. 각 텍스트는 문화적 민감성을 반영하여 만들어졌으며, 특히 간접적이지만 해로운 내용(예: 미시적 공격성(microagressions), 편견)을 포함하고 있습니다.

- **Performance Highlights**: 평가된 일곱 S/LLMs는 정확성 면에서는 일반적으로 수용 가능한 점수를 받았으나, 전체적인 독성 판단에서 인간 판정과의 일치도가 낮았습니다. 이 모델들은 문화적 상황에 따라 해를 가할 수 있는 내용을 식별하는 데 어려움을 겪었습니다. 특히 GPT-4 Turbo와 Gemma 7B는 예제를 올바르게 분류한 비율이 높았지만, 섬세하면서도 해로운 내용의 판별에는 여전히 과제가 있음을 확인했습니다.



### PARAMANU-GANITA: Language Model with Mathematical Capabilities (https://arxiv.org/abs/2404.14395)
- **What's New**: 이 논문에서는 수학 분야에 특화된 새로운 자동 회귀(Auto Regressive, AR) 디코더 기반 언어 모델인 'Paramanu-Ganita'를 소개합니다. 이 모델은 208백만 매개변수(parameter)를 가지며, 4096의 문맥 크기(context size)에서 처음부터 사전 훈련(pretrained from scratch)되었습니다. 주요 혁신은 일반 대용량 언어 모델(Large Language Models, LLMs)과 비교하여 매우 적은 매개변수와 훈련 시간으로 뛰어난 수학적 추론 능력을 달성했다는 점입니다.

- **Technical Details**: Paramanu-Ganita는 자체 큐레이션된 수학적 코퍼스(mathematical corpus)에서 사전 훈련되었으며, 단일 NVIDIA A100 PCIE 40GB GPU에서 작동됩니다. 이 모델은 특수하게 고안된 수학 도메인 토크나이저(math domain specialised tokenizer)를 사용하여, 수학 텍스트, 교과서, 강의 노트, 웹 크롤링 데이터 등을 포함합니다.

- **Performance Highlights**: Paramanu-Ganita는 GSM8k 수학 벤치마크에서 전문가 급 LLM인 Minerva 8B와 LLEMMA 7B를 비롯하여 LLaMa, Falcon, PaLM과 같은 일반 대형 모델들을 큰 폭으로 앞질렀습니다. 소형인 모델임에도 불구하고, 거대한 모델들보다 6.4%포인트에서 35.3%포인트까지 더 높은 성능을 보여주며, 효율성과 정확성에서 우수함을 입증했습니다. 이 모델은 23,000 A100 시간이 요구되는 LLEMMA 7B에 비해 단 146 A100 시간만으로 훈련되었으며, 체크포인트 크기도 LLEMMA 7B의 13.5GB에 비해 1GB 미만입니다.

- **Conclusion**: Paramanu-Ganita의 성공은 특정 도메인 맞춤 학습의 잠재력을 보여주며, 수학과 같은 전문 분야에서 작은 모델을 통한 효율적 접근 방식의 중요성을 강조합니다. 이는 거대한 모델과 비교적 적은 계산력을 사용하여도 높은 수준의 추론 능력을 개발할 수 있음을 의미합니다.



### A Survey on Self-Evolution of Large Language Models (https://arxiv.org/abs/2404.14387)
- **What's New**: 이 연구는 대규모 언어 모델(LLMs)의 자기 진화 접근법에 대한 포괄적인 조사를 제시합니다. 인간 경험 학습 과정에서 영감을 받아, 이 새로운 훈련 패러다임은 LLM이 자체적으로 경험을 습득하고, 정제하며, 학습할 수 있게 함으로써 지능의 슈퍼레벨로의 확장 가능성을 제시합니다.

- **Technical Details**: 자기 진화의 개념적 프레임워크는 경험 습득, 경험 정제, 업데이트, 평가의 네 단계로 구성된 반복 주기로 설명됩니다. LLM은 새로운 과제를 통해 경험을 얻고, 이를 정제하여 더 나은 감독 신호를 얻은 다음 모델을 업데이트하고 평가하여 진행 상황을 측정합니다. 이러한 프로세스는 LLM이 주도적으로 학습하고 개선할 수 있는 능력을 부여합니다.

- **Performance Highlights**: 자기 진화를 통한 훈련 방법은 인간의 감독 없이도 LLM이 성능의 한계를 넘어서고 인간을 능가하는 성능을 달성할 수 있는 가능성을 보여줍니다. 예를 들어, DeepMind의 AMIE 시스템과 Microsoft의 WizardLM-2는 자기 진화적 프레임워크를 사용하여 개발되었으며, 각각 주요 진료 의사들보다 높은 진단 정확도를 보여주고 GPT-4의 초기 버전을 능가하는 성능을 달성하였습니다.



### Beyond Scaling: Predicting Patent Approval with Domain-specific  Fine-grained Claim Dependency Graph (https://arxiv.org/abs/2404.14372)
Comments: 17 Pages, Under Review

- **What's New**: 이 논문에서는 특허 승인 예측(patent approval prediction)이라는 특정 영역에 초점을 맞추고 있으며, 대규모 언어 모델들(Large Language Models, LLMs)의 확장이 이 분야에서 기대치만큼의 성능을 발휘하지 못함을 발견했습니다. 대신, 특허 데이터의 복잡한 의존성을 활용하는 도메인 특화 그래프 방법론이 더 효과적임을 제시하고 있습니다. 특히, Fine-grained cLAim depeNdency (FLAN) Graph를 제안하여, 각 클레임의 내부 및 상호 의존성을 그래프 형태로 모델링함으로써 더 나은 예측 성능을 달성하고 있습니다.

- **Technical Details**: 연구팀은 BERT와 같은 기존의 언어 모델을 대형 언어 모델들(LLMs)로 대체하여 실험했으나, 예상과는 달리 성능이 향상되지 않았습니다. 이에 따라, 저자들은 특허 데이터의 구조적 특성과 법적 지식을 깊이 분석하고, 이를 기반으로 FLAN 그래프를 개발했습니다. FLAN 그래프는 모델-비특화적(model-agnostic)이며, 다양한 그래프 모델에 적용 가능합니다. 사용된 그래프 모델로는 GCN(Graph Convolutional Network), GAT(Graph Attention Network), TreeLSTM 등이 있으며, GraphSage를 사용했을 때 가장 높은 성능 향상을 보였습니다.

- **Performance Highlights**: FLAN 그래프를 적용한 그래프 모델들은 모두 기존의 SOTA(state-of-the-art) 모델들을 크게 앞서는 성능을 보였습니다. 특히 GraphSage 모델은 AUC에서 7.4% 향상되고, Macro-F1 점수에서 7.8% 향상되어, 각각 66.04와 58.22의 점수를 달성했습니다. 이러한 결과는 FLAN 그래프가 특허 승인 예측 작업에 매우 유효하다는 것을 입증합니다.



### Better Synthetic Data by Retrieving and Transforming Existing Datasets (https://arxiv.org/abs/2404.14361)
- **What's New**: 새로운 방법인 DataTune을 도입하여, 이는 기존의 공개 데이터셋을 새로운 작업 요구사항에 맞게 자동으로 변환함으로써 NLP 모델을 훈련하는 방식을 개선했습니다. 이 방법은 기존 데이터를 재사용하여 국한된 자원 하에서도 효과적인 모델 훈련이 가능하도록 지원합니다.

- **Technical Details**: DataTune은 적절한 데이터셋을 찾아내어 (dataset retrieval) 그 데이터를 특정 작업에 맞게 변형 (data transformation)하는 과정을 포함합니다. 첫 단계로 HuggingFace Hub에서 관련 데이터셋을 식별하고, 이후 대규모 언어 모델 (LLM, Large Language Model)을 사용하여 데이터셋을 재순위화 (reranking)하여 특정 작업에 최적인 데이터셋을 선택합니다. 마지막으로, 선택된 데이터셋을 대상 작업에 적합하도록 데이터를 변형하고 새로운 학습 데이터셋 𝒟′ (D')을 생성합니다.

- **Performance Highlights**: DataTune을 이용한 모델은 기존 few-shot prompting 방법 대비 평균 5.2 점, 기존 합성 데이터 생성 방법을 사용할 때 대비 평균 8 점 향상을 보였습니다. 특히, 이는 다양한 언어 기반 작업에 대한 성능 개선을 보여주며, 데이터의 다양성과 난이도를 증가시키는 데에 효과적임을 입증합니다.



### Calc-CMU at SemEval-2024 Task 7: Pre-Calc -- Learning to Use the  Calculator Improves Numeracy in Language Models (https://arxiv.org/abs/2404.14355)
Comments: NumEval at SemEval, NAACL 2024

- **What's New**: 이 논문에서는 프리-파인튜닝(pre-finetuning) 목표인 Pre-Calc를 제안하여 언어 모델의 수리 능력을 향상시키는 새로운 접근법을 소개하고 있습니다. Pre-Calc는 언어 모델이 계산기 사용을 학습하도록 하는 것으로, 인코더-온리(encoder-only) 구조와 인코더-디코더(encoder-decoder) 구조 모두에 적용됩니다. 특히, BERT와 RoBERTa를 사용한 인코더-온리 목표와, Flan-T5를 사용한 인코더-디코더 접근 방식의 결과를 다루고 있습니다.

- **Technical Details**: Pre-Calc 방법론은 숫자의 이해와 숫자 조합 방법에 대한 학습을 포함하고 있으며, 이는 두 가지 주요 태스크로 구성됩니다. 첫 번째는 피연산자 식별(Operand Identification)이며, 토큰 수준의 분류 작업입니다. 두 번째는 연산 분류(Operation Classification)으로, 특정 [OP] 토큰을 사용하여 필요한 수학 연산을 분류하는 시퀀스 수준의 분류 작업입니다. MAWPS, SVAMP, AsDiv-A 데이터셋을 활용하여 Pre-Calc 목표 아래에서 모델을 전처리하고, 이를 통해 수리에 중점을 둔 다운스트림(downstream) 작업에서 모델의 성능을 개선합니다.

- **Performance Highlights**: Pre-Calc를 적용한 BERT와 RoBERTa 모델은 NumEval의 모든 정량적 다운스트림 태스크에서 우수한 성능을 보여주며, 특히 RedditNLI와 AWPNLI 하위태스크에서 10점 이상의 성능 향상을 달성했습니다. Flan-T5 모델은 AWPNLI와 같은 계산 집약적인 태스크에서 명시적 연산 수행 능력이 향상되었으나, 텍스트 중심 및 의미(semantics) 작업에서 약간의 성능 저하가 있었습니다.



### Zero-shot Cross-lingual Stance Detection via Adversarial Language  Adaptation (https://arxiv.org/abs/2404.14339)
- **What's New**: 이 논문은 여러 언어에 걸쳐 레이블이 없는 데이터에서 입장 검출을 수행할 수 있는 다양한 언어의 제로-샷 입장 검출 모델을 개발하는 최초의 시도를 소개합니다. 새롭게 제안된 모델인 Multilingual Translation-Augmented BERT (MTAB)는 전사학습(augmented translation)과 적대적 학습(adversarial learning)을 결합하여 효과적인 입장 판별을 가능하게 합니다.

- **Technical Details**: MTAB는 대상 언어에 대한 명시적인 훈련 데이터없이 교차 언어 분류기의 성능을 향상시키기 위해 전사학습을 사용합니다. 이 기술은 또한 적대적 학습을 도입하여 모델의 효능을 더욱 증진시키는 데 사용됩니다. 이 연구에는 영어(English), 독일어(German), 프랑스어(French), 이탈리아어(Italian) 네 가지 언어로 레이블이 지정된 백신에 대한 입장 데이터셋이 사용되었습니다.

- **Performance Highlights**: 실험 결과, MTAB는 강력한 기준 모델(strong baseline model)과 비교하여 더 나은 결과를 보여주며, 모델의 다양한 구성 요소들이 입장 판별 성능 향상에 얼마나 기여했는지를 보여줍니다. 특히, 전사 증강된 데이터와 적대적 학습 구성 요소가 모델 성능 향상에 중요한 역할을 했다는 것을 입증합니다.



### Automated Long Answer Grading with RiceChem Datas (https://arxiv.org/abs/2404.14316)
- **What's New**: 이 논문에서는 교육용 자연어 처리 분야에서 새로운 연구 영역, 자동 장문 답변 채점(Automated Long Answer Grading, ALAG)을 소개합니다. 기존의 자동 짧은 답변 채점(Automated Short Answer Grading, ASAG) 및 자동 에세이 채점(Automated Essay Grading, AEG)과 구분되는 ALAG는 복잡성과 다면성 때문에 고유한 도전을 제시합니다. ALAG 연구를 위해, 대학 화학 과정에서 파생된 RiceChem 데이터셋을 도입하며, 이는 평균 단어 수가 높아 ASAG 데이터셋과 현저한 차이를 보입니다.

- **Technical Details**: 이 논문에서는 새로운 접근 방식을 제안하여 ALAG 문제를 정답 루브릭 구성 요소가 충족되었는지를 검증하는 자연어 추론 모델을 사용하는 루브릭(entailment) 문제로 재정의합니다. 이를 통해 우수한 성능의 전이 학습을 이용할 수 있는데, 특히 MNLI를 활용한 전이 학습이 중요합니다. BERT, RoBERTa, BART 같은 인코더 모델을 미세 조정하여 ALAG 작업에 대한 기준을 설정하고, 이는 전통적 접근법에 비해 뛰어난 성능을 보여줍니다.

- **Performance Highlights**: RiceChem 데이터셋에서 대규모 언어 모델(Large Language Models, LLMs)을 벤치마킹하고 GPT 모델과 비교한 결과, ALAG가 ASAG보다 훨씬 복잡함을 보여주었습니다. 비록 루브릭 기반 접근 방식과 MNLI로부터의 전이 학습을 활용하여 뛰어난 성능을 보였지만, LLM의 성능이 RiceChem에서 낮음은 ALAG 과제의 유의미한 어려움을 강조합니다.



### Self-Supervised Alignment with Mutual Information: Learning to Follow  Principles without Preference Labels (https://arxiv.org/abs/2404.14313)
- **What's New**: 이 논문에서는 새로운 자기감독 학습 접근방식인 SAMI(Self-Supervised Alignment with Mutual Information)를 도입하여 기존 언어모델(LM)을 향상시킵니다. SAMI는 기본 언어 모델이 원칙(Principles)을 따르도록 튜닝하는 새로운 메커니즘으로, 선호 라벨(Preference Labels)이나 시연(Demonstrations) 없이도 모델의 행동 원리를 따르는 방식을 학습할 수 있습니다.

- **Technical Details**: SAMI는 사전 훈련된 LM을 미세조정(Finetuning)하여 주어진 질의에 대한 원칙과 모델 반응 사이의 조건부 상호 정보량(Conditional Mutual Information)을 증가시키는 반복 알고리즘입니다. 이 방법은 원칙 작성자(Principle Writer) 모델을 사용하여 원칙을 작성하고, 이 원칙을 데이터셋의 쿼리와 결합하여 반응을 샘플링 합니다. 그 다음, 상호 정보의 하한을 최적화하여 조건부 상호 정보량을 증가시키는 방식으로 프로세스를 반복합니다.

- **Performance Highlights**: 미스트랄-7b(Mistral-7b) 모델을 사용한 SAMI는 단일 턴 대화에서 초기 사전 훈련된 모델보다 66%에서 77% 사이의 승률을 보여 향상된 성능을 나타냈습니다. 또한, 명령을 따르도록 미세조정된 기준 모델(Mistral-7b-instruct)과 비교했을 때, SAMI 훈련 모델이 55%에서 57%의 승률로 우수한 성능을 보였습니다. 특히, 강력한 사전 훈련 모델(Mixtral-8x7b)을 약한 명령 튜닝 모델(Mistral-7b-instruct)이 작성한 원칙을 사용하여 조정하는 실험에서도 유사한 성능 향상이 관찰되었습니다.



### Marking: Visual Grading with Highlighting Errors and Annotating Missing  Bits (https://arxiv.org/abs/2404.14301)
- **What's New**: 이 논문에서는 '마킹(Marking)'이라는 새로운 평가 작업을 소개합니다. 이 작업은 학생들의 응답을 심층 분석하고 시각적 하이라이트를 통해 피드백을 제공하는 개선된 자동 채점 시스템을 확장하는 것을 목표로 합니다. 기존의 이진 점수 제공 시스템과 달리, '마킹'은 학생 응답의 세그먼트를 정확한지, 부정확한지, 또는 관련 없는지 파악하고, 기준답안(Gold Answers)에서 누락된 내용을 감지합니다. 이를 위해 전문가(Subject Matter Experts)에 의해 특별히 구축된 새 데이터셋을 소개하고, 이 작업을 자연어 추론(Natural Language Inference, NLI) 작업의 확장으로 구성합니다.

- **Technical Details**: 이 연구에서는 BERT와 RoBERTa와 같은 트랜스포머 모델(transformer models)을 사용하여 학생의 응답에서 정답, 부정답, 중립을 판별하는 언어 모델을 훈련시킵니다. '마킹' 프레임워크에서는 기준 답안을 전제(premise)로, 학생의 응답을 가설(hypothesis)로 설정하여 NLI의 원칙을 따릅니다. 실험 설정에서는 e-SNLI 데이터셋을 사용하여 지능적인 훈련 단계를 구현하며, Dual Instance Pairing(DIP) 및 불용어 제거(stopword removal)라는 전처리 단계의 영향을 조사합니다.

- **Performance Highlights**: 실험 결과 DIP와 불용어 제거는 모델 성능에 긍정적인 영향을 미치는 것으로 나타났습니다. 이는 '마킹' 작업의 복잡성과 도전을 강조하며, 자동 채점 시스템을 향상시킬 잠재력을 시사합니다. 해당 연구는 AI 기반 교육 평가 도구에서 새로운 연구 방향을 제시하며, 향후 연구를 위한 명확한 궤적을 설정합니다.



### A Survey on Efficient Inference for Large Language Models (https://arxiv.org/abs/2404.14294)
- **What's New**: 이 논문은 대규모 언어 모델(Large Language Models, LLMs)의 효율적인 추론(inference) 기법에 대한 종합적인 리뷰를 제공합니다. 점점 더 많은 관심을 받고 있는 LLM의 배포와 관련하여, 계산 및 메모리 요구 사항이 높아지는 문제를 해결하기 위한 다양한 최적화 기술이 소개되어 있습니다.

- **Technical Details**: 논문은 큰 모델 크기, 복잡도가 높은 주의(attention) 연산 및 자동 회귀(auto-regressive) 디코딩 방식이 LLM 추론의 비효율성을 초래하는 주요 원인으로 분석합니다. 또한, 현재 문헌을 데이터 수준(data-level), 모델 수준(model-level), 시스템 수준(system-level) 최적화로 구분하는 체계적인 분류(taxonomy)를 제시합니다. 각 부문에서 대표적인 방법론을 실험적으로 비교하여 양적인 통찰을 제공하며, 이는 미래 연구 방향에 대한 유용한 지침을 제공합니다.

- **Performance Highlights**: 이 연구는 다양한 최적화 기술들을 통해 LLM의 추론 성능을 효율적으로 향상시킬 수 있는 방법을 제안합니다. 특히, 모델 양자화(model quantization) 및 서빙 시스템(serving systems)과 같은 주요 분야에서의 실험적 분석을 통해 실용적인 추천 및 지침을 제공하며 이는 LLM 배포의 실제 비용 및 성능 개선에 중요한 기여를 할 수 있을 것으로 기대됩니다.



### What do Transformers Know about Government? (https://arxiv.org/abs/2404.14270)
- **What's New**: 이 논문은 BERT 모델이 어떻게 문장 내에서 구성 요소 간의 '통치(government)' 관계를 인코딩하는지를 조사합니다. 특히, 변형기 전체에서 통치 관계에 대한 정보가 인코딩되어 있으며, 주로 모델의 초기 레이어에서 중요하게 나타납니다. 또한, 핀란드어와 러시아어와 같이 형태학적으로 풍부한 두 언어에 대한 데이터를 사용하여 실험을 수행하였습니다.

- **Technical Details**: BERT 모델의 모든 레이어와 헤드(attention heads)를 사용하여 통치 관계에 대한 정보 인코딩을 조사했습니다. 초기 레이어(5 또는 7 레이어)에서 거의 동일한 정확도로 통치 관계의 존재를 감지할 수 있는 충분한 정보가 있음을 발견하였습니다. 몇 가지 주목할 만한 어텐션 헤드가 통치 관계를 인코딩하고 있으며, 이는 통치 예측을 위해 사용될 수 있습니다. 이러한 발견은 새로운 통치 유형을 발견하고 기존 자원을 확장하는 데 도움이 될 수 있습니다.

- **Performance Highlights**: 프로빙 분류기(probing classifiers)는 통치 동사와 통치 패턴을 감지하는 작업에서 높은 성능을 보였으며, 이는 학습 데이터에서 보지 못한 새로운 패턴을 발견하는 데 사용될 수 있습니다. 이는 언어 학습 자원을 구축하거나 기존 자원을 향상시키는 데 이점을 제공합니다. 연구 결과를 통해 'Government Bank'라는 데이터셋을 공개하여, 실험에 사용된 언어들의 수천 개 단어에 대한 통치 관계를 정의합니다.



### Phi-3 Technical Report: A Highly Capable Language Model Locally on Your  Phon (https://arxiv.org/abs/2404.14219)
Comments: 12 pages

- **What's New**: phi-3-mini는 태블릿과 같은 모바일 기기에서도 배포가 가능할 만큼 작은 크기임에도 불구하고, Mixtral 8x7B 및 GPT-3.5와 같은 대형 모델과 비슷한 수준의 성능을 보여주는 3.8 billion parameters (파라미터)를 가진 언어 모델입니다. phi-3-mini는 3.3 trillion tokens (토큰)의 훈련 데이터를 바탕으로 개발되었고, 최적화된 데이터 세트의 사용을 통해 향상된 성능을 제공합니다.

- **Technical Details**: phi-3-mini 모델은 transformer decoder (트랜스포머 디코더) 아키텍처를 사용하며, LongRope을 통해 컨텍스트 길이를 128K까지 확장할 수 있습니다. 또한, Llama-2와 동일한 블록 구조와 토크나이저를 사용하여, 기존 개발 패키지를 그대로 활용할 수 있는 호환성을 제공합니다. 이 모델은 이미 chat-finetuned 되어 있으며, 메모리 요구사항을 줄이기 위해 4비트로 양자화되어 약 1.8GB의 메모리만을 사용합니다.

- **Performance Highlights**: phi-3-mini는 MMLU에서 69%의 결과를 보여주고, MT-bench에서는 8.38의 점수를 달성하였습니다. 이러한 성능은 초대형 모델들과 견줄만하며, 실제로 iPhone 14에서 원활하게 동작할 정도로 효율적입니다. 추가적으로, 더 큰 모델인 phi-3-small (7B parameters)과 phi-3-medium (14B parameters)도 같은 데이터 세트로 훈련되었고, 예상보다 높은 성능 향상을 보여주었습니다.



### Text-Tuple-Table: Towards Information Integration in Text-to-Table  Generation via Global Tuple Extraction (https://arxiv.org/abs/2404.14215)
- **What's New**: 최근에 대규모 언어 모델(Large Language Models, LLMs)의 출현과 그들이 제공하는 다운스트림 작업들, 예를 들어 텍스트 요약(text summarization)과 텍스트 마이닝(text mining)에 대한 잠재적인 이점으로 인해 텍스트 정보를 간결하고 구조화된 표로 압축하는 작업이 관심을 받고 있습니다. 이 논문에서는 실시간 코멘터리 텍스트를 기반으로 한 경쟁 요약 테이블을 생성하기 위해 개발된 새로운 벤치마크 데이터셋인 LiveSum을 소개합니다.

- **Technical Details**: LiveSum 데이터셋은 텍스트에서 정보를 추출하고 추론 및 통합하는 과정을 필요로 하는 실제 시나리오에서 텍스트-테이블 생성(text-to-table generation) 작업을 위해 설계되었습니다. 이 연구는 $T^3(Text-Tuple-Table)라고 불리는 새로운 파이프라인을 제안하여, 기존 방법론과 데이터셋의 부족을 해결하고 피인용 성능을 향상시킵니다.

- **Performance Highlights**: 실험 결과는 LLM이 이 작업에서 아직도 어려움을 겪고 있지만, 우리의 접근 방식은 명시적인 훈련 없이도 중요한 성능 향상을 제공할 수 있음을 보여줍니다. 또한, $T^3 방법은 여러 다른 텍스트-테이블 데이터셋에서 이전 접근법들을 능가하는 강력한 일반화(generalization) 능력을 보여줍니다.



### EnzChemRED, a rich enzyme chemistry relation extraction datas (https://arxiv.org/abs/2404.14209)
- **What's New**: 이 연구에서는 효소 큐레이션(enzyme curation)을 지원하기 위한 새로운 훈련 및 벤치마킹 데이터셋인 EnzChemRED(Enzyme Chemistry Relation Extraction Dataset)를 소개합니다. 이 데이터셋은 효소와 효소가 촉매하는 화학 반응을 UniProtKB(UniProt Knowledgebase) 및 ChEBI(Chemical Entities of Biological Interest의 온톨로지) 식별자를 사용하여 주석을 단 PubMed의 1,210개의 전문가 큐레이션된 초록을 포함하고 있습니다. 또한, 이 데이터셋은 자연어 처리(Natural Language Processing, NLP) 방법론, 특히 대규모 언어 모델의 개발을 지원할 수 있는 벤치마크 역할을 합니다.

- **Technical Details**: EnzChemRED는 1,210개의 선별된 PubMed 초록들로 구성되어 있고, 효소와 그들이 촉매하는 화학 반응을 UniProtKB와 ChEBI 식별자를 사용하여 주석화하였습니다. 이를 통해 사전 훈련된 언어 모델을 미세조정(fine-tuning)하여 제공하는 자료들은 텍스트에서 단백질과 화학물질의 언급을 식별할 수 있는 능력(Named Entity Recognition, NER)을 상당히 향상시키며, 그들이 참여하는 화학 변환 관계(chemical conversions relations)를 추출하는 능력(Relation Extraction, RE) 또한 향상됩니다.

- **Performance Highlights**: 사전 훈련된 언어 모델들은 EnzChemRED를 사용하여 미세 조정된 후 NER에서 평균 F1 스코어 86.30%, 화학 변환 쌍에 대한 RE에서 평균 F1 스코어 86.66%, 또한 화학 변환 쌍과 연결된 효소에 대한 RE에서 평균 F1 스코어 83.79%를 달성했습니다. 이 연구에서는 EnzChemRED를 사용하여 미세 조정된 가장 효과적인 방법들을 결합하여 PubMed 규모의 초록에서 지식을 추출하는 종단 간 파이프라인(end-to-end pipeline)을 생성하였고, 이를 통해 UniProtKB 및 반응 지식베이스 Rhea를 위한 문헌에서의 효소 기능의 초안 지도를 생성했습니다.



### Swap distance minimization beyond entropy minimization in word order  variation (https://arxiv.org/abs/2404.14192)
- **What's New**: 이 논문에서는 주어(subject), 직접 목적어(direct object), 간접 목적어(indirect object), 동사(verb)와 같은 요소로 구성된 언어 구조의 가능한 모든 순서를 고려합니다 (예를 들어, n=3 또는 n=4). 연구는 성향 최소화(entrophy minimization) 원리와 스왑 거리 최소화(swap distance minimization) 원리에 의해 이러한 순서의 빈도가 제한되는지를 조사합니다. 특히, 스왑 거리 최소화에 대한 새로운 점수인 평균 스왑 거리(average swap distance)를 소개하고 그 이론적 분포를 분석합니다.

- **Technical Details**: 이 논문은 언어구조의 가능한 순서의 엔트로피(entrophy)와 평균 스왑 거리를 정의하고, 이를 통해 언어 구조가 어떻게 최적화될 수 있는지를 탐구합니다. permutohedron과 같은 그래프 이론을 적용하여 인접 요소 간의 스왑에 대한 이론적 배경을 제시하며, 스왑 거리(swap distance)가 어떻게 계산되고, 이것이 언어 처리의 인지적 비용과 어떻게 상관관계가 있는지를 설명합니다. 또한, 빈도 분포에 대한 주사위 굴리기(die rolling experiment)와 무작위 순열(random permutation)의 널 가설(null hypothesis) 하에서의 기대값을 계산하고, 이를 통해 주어진 제약 조건 하에서의 언어 순서 선택의 패턴을 분석합니다.

- **Performance Highlights**: 연구 결과, n=4의 경우 두 원리(성향 최소화와 스왑 거리 최소화) 모두에서 강력한 증거를 발견했으며, n=3에 대해서는 스왑 거리 최소화의 증거는 여전히 발견되었지만, 성향 최소화의 증거는 약간 약해 보입니다. 이는 이 두 원리가 언어 구조의 순서 결정에 미치는 영향을 실증적으로 보여주며, 언어의 진화와 학습 과정에서 이러한 최소화 전략이 어떻게 작용하는지에 대한 중요한 통찰력을 제공합니다.



### SemEval-2024 Task 8: Multidomain, Multimodel and Multilingual  Machine-Generated Text Detection (https://arxiv.org/abs/2404.14183)
Comments: 23 pages, 12 tables

- **What's New**: SemEval-2024 Task 8은 '다언어 및 다영역에서 생성된 다기능 기계 텍스트 감지(multigenerator, multidomain, and multilingual machine-generated text detection)'를 주제로 하여, 인간이 작성한 텍스트와 기계가 생성한 텍스트를 구별하는 세 가지 하위 작업을 포함합니다. 이 특별한 작업은 기계 생성 텍스트(Machine-generated Text, MGT)의 확산이 저널리즘, 교육, 학문 등에서의 잠재적 오용 가능성을 조명하면서 중요성을 더하고 있습니다.

- **Technical Details**: 이 과제는 다음과 같은 세 개의 하위 작업들로 구성됩니다: 1) 단일 언어 및 다중 언어 트랙이 있는 이진 분류 작업(Subtask A), 2) 텍스트가 특정 LLM(Large Language Model)에 의해 생성되었는지를 식별하는 작업(Subtask B), 그리고 3) 텍스트 내에서 인간의 저자에서 기계로의 전환 지점을 식별하는 작업(Subtask C). 각 하위 작업은 대규모 평가 데이터셋에서 성능을 평가하며, 특히 Subtask B와 Subtask C는 기계학습 모델이 사람이 작성한 부분과 기계가 생성한 부분을 구분할 능력을 세밀하게 측정합니다.

- **Performance Highlights**: 각 하위 작업에는 상당한 참여가 있었으며, 특히 Subtask A의 단일 언어 트랙에는 126개 팀이, 다중 언어 트랙에는 59개 팀이 참여했습니다. 최고 성능 시스템은 모든 하위 과제에서 LLM(Large Language Models)을 사용했습니다. 이 과업은 MGT의 탐지와 식별을 위한 자동화된 시스템의 개발을 촉진하고, 인간과 기계가 생성한 텍스트를 정확하게 구분하는 기술을 발전시키는 데 중점을 두고 있습니다.



### Fine-Tuning Large Language Models to Translate: Will a Touch of Noisy  Data in Misaligned Languages Suffice? (https://arxiv.org/abs/2404.14122)
- **What's New**: 본 연구는 기존의 기계 번역(MT)에 관한 연구에서 발견한 내용을 더욱 확장하여 대형 언어 모델(LLMs)의 세밀하게 조정된(SFT) 데이터의 역할을 다루고 있습니다. 특히, 적은 데이터(32개 인스턴스)와 단일 번역 방향을 사용하는 실험을 통해, 이러한 최소한의 설정에서도 LLM이 다양한 언어 방향으로 번역할 수 있는 능력을 획득하였음을 보여줍니다.

- **Technical Details**: 본 논문에서는 LLM을 기반으로 한 세밀한 조정(fine-tuning)을 통해 기계 번역을 위해 준비합니다. 실험은 Llama-2 7B 모델을 사용하며, 훈련 데이터로는 WMT17부터 WMT22까지 다양한 언어 쌍을 포함한 데이터 세트가 사용됩니다. 세밀한 조정에서는 학습률(learning rate) 5e-6, 효과적인 배치 크기(effective batch size) 64, 선형 학습률 스케줄링(linear learning rate scheduling)을 적용하며, Alpaca 프롬프트 템플릿을 사용하여 입력을 형성합니다.

- **Performance Highlights**: 실험 결과는 32개의 데이터 인스턴스만으로 LLM이 11개 언어 방향으로 번역할 수 있음을 입증하였고, 단일 번역 방향에서의 조정만으로도 여러 번역 방향에 대한 능력을 갖출 수 있음을 보여주었습니다. 그러나 영어를 대상(target) 언어로 사용할 경우 작업의 오해로 인해 비영어 번역의 질이 저하될 수 있으며, 입력된 데이터의 질이 결과에 큰 영향을 미칠 수 있음을 발견하였습니다. 또한, 고자원 언어(high-resource languages)에서의 데이터 노이즈는 LLM이 쉽게 과적합(overfit)할 수 있으나, 저자원 언어(low-resource languages)에서는 그 영향이 적은 것으로 나타났습니다.



### Bored to Death: Artificial Intelligence Research Reveals the Role of  Boredom in Suicide Behavior (https://arxiv.org/abs/2404.14057)
- **What's New**: 이 연구는 인공지능(Artificial Intelligence, AI) 기법을 이용하여 자살 행동을 유발하거나 악화시키는 숨겨진 위험 요인을 발견하는 것을 목표로 하였습니다. 특히, 자살 위험의 강력한 예측 요인으로 ‘지루함(boredom)’이 발견되었는데, 이는 기존의 문헌에서는 자살의 독특한 위험 요소로 거의 인식되지 않았습니다.

- **Technical Details**: 연구는 Facebook 게시물 228,052건과 1,006명의 사용자가 완성한 금본위 자살 심각도 평가 척도(Columbia Suicide Severity Rating Scale) 데이터셋을 기반으로 하였습니다. 이 데이터셋은 사전 가설 없이 아래로부터 위로의(bottom-up) 연구 파이프라인을 사용하여 분석되었고, 새로운 데이터셋을 사용한 상위로부터 하위로의(top-down) 분석을 통해 결과를 검증하였습니다. 또한, 이와 관련하여 우울증과 지루함을 측정하는 검증된 척도와 함께 두 번째 데이터셋에서 1,062명의 참가자의 응답을 추가로 분석하였습니다.

- **Performance Highlights**: AI 가이드를 통한 거의 완전 자동화된 연구 파이프라인은 자살 위험을 예측하는 Facebook의 네 가지 주제를 도출하였습니다. 지루함은 우울증을 매개로 자살과 간접적인 관계를 갖는 것으로 나타났으며, Facebook 데이터셋에서도 이와 동등한 매개 관계가 관찰되었습니다. 더 나아가 직접적인 관계 또한 발견되었습니다. 이 결과는 지루함이 우울증과는 관계없이 자살 행동을 촉발할 수 있는 부적응적 '성분(ingredient)'으로, 임상가들이 이 부담스럽고 때로는 존재론적인 경험에 주목해야 한다는 것을 시사합니다.



### Differential contributions of machine learning and statistical analysis  to language and cognitive sciences (https://arxiv.org/abs/2404.14052)
- **What's New**: 이 연구는 언어 과학 및 인지 연구에서 기계 학습(Machine Learning)과 통계 분석(Statistical Analysis)이 어떻게 다른 통찰력을 제공하는지를 보여주고자 합니다. Buckeye Speech Corpus를 사용하여 두 분야가 데이터 기반 연구에서 어떻게 활용되는지를 설명하며, 이는 사회 과학 분야에서의 이러한 기법들의 적용을 보여주는 몇 안 되는 연구 중 하나입니다.

- **Technical Details**: 이 연구에서 사용된 데이터 셋은 Buckeye Speech Corpus로, 언어와 인지 과학 분야의 데이터 기반 연구에 적용되었습니다. 기계 학습과 통계 분석이 각각 어떻게 데이터를 처리하고 해석하는지에 대한 차이점을 비교하고, 이러한 차이가 연구 결과에 어떤 영향을 미치는지 분석합니다. 특히, 기계 학습은 복잡한 패턴을 식별하고 예측 분석을 강화하는 데 사용되는 반면, 통계 분석은 연구 설계, 데이터 수집 및 결과 해석에 필수적인 역할을 합니다.

- **Performance Highlights**: 이 연구는 기계 학습과 통계 분석을 동일한 데이터셋에 적용함으로써, 각 방법론이 데이터를 어떻게 다르게 처리하고 해석하는지에 대한 구체적인 이해를 제공합니다. 이를 통해 언어 과학과 인지 과학 분야에서의 연구 방법론을 개선하고 더 정교하고 효과적인 연구 전략을 개발할 수 있는 기반을 마련했습니다.



### LLMs Know What They Need: Leveraging a Missing Information Guided  Framework to Empower Retrieval-Augmented Generation (https://arxiv.org/abs/2404.14043)
- **What's New**: 이 연구에서는 Retrieval-Augmented Generation (RAG) 의 주요 도전과제에 대응하기 위해, Missing Information Guided Retrieve-Extraction-Solving (MIGRES) 패러다임을 제안하고 있습니다. 이는 누락된 정보를 식별하여 지식 검색을 안내하고 문제 해결을 향상시키기 위한 방법론입니다. 특히 복잡한 멀티홉 쿼리를 처리하고 관련 문서를 효과적으로 검색할 수 있도록 돕는 새로운 접근 방식을 소개하고 있습니다.

- **Technical Details**: MIGRES는 크게 쿼리 생성기(Query Generator), 검색기(Retriever), 지식 필터(Knowledge Filter)로 구성된 검색 모듈과, 유용한 정보를 추출하는 리프 모듈(Leaf module), 질문에 대한 응답 여부를 결정하는 메인 모듈(Main module)로 구성되어 있습니다. 이들은 상호작용하여 검색된 지식에서 필요한 정보를 추출하고, 누락된 정보를 식별함으로써 지속적으로 쿼리를 개선하고, 문제에 대한 답을 찾아갑니다.

- **Performance Highlights**: MIGRES 방법은 다양한 공개 데이터셋에서 실시된 실험을 통해 그 우수성이 입증되었습니다. 특히, 기존 RAG 접근법보다 향상된 성능을 보여줄 뿐만 아니라, 제로샷(Zero-shot) 시나리오에서 평균 95.6%의 정확도로 누락된 정보를 식별할 수 있는 능력을 보여주었습니다. 이는 복잡한 질문에 대응할 수 있는 모델의 전반적인 효율성을 크게 향상시킬 수 있음을 시사합니다.



### Exploring neural oscillations during speech perception via surrogate  gradient spiking neural networks (https://arxiv.org/abs/2404.14024)
- **What's New**: 이 연구에서는 신경 동역학(neural dynamics)을 흉내낼 수 있는 새로운 음성 인식 아키텍처를 제시했습니다. 심층 학습(deep learning) 프레임워크와 호환되며 확장 가능한 이 아키텍처는 중추적인 스파이킹 신경망(central spiking neural network)에서 신경 진동(neural oscillations)의 출현을 이끌어냈습니다. 특히, 이 아키텍처는 피드백 메커니즘(feedback mechanisms)의 억제 역할을 강조하여 신경 활동을 조절하고 동기화하는데 중요하다는 것을 발견했습니다.

- **Technical Details**: 신경 진동의 측정을 위해 사용된 주요 기술은 교차 주파수 결합(cross-frequency couplings, CFC)과 스파이크 빈도 적응(spike frequency adaptation, SFA), 그리고 재귀 연결(recurrent connections)입니다. 이들은 신경망의 각 계층(layer) 내부와 계층 간에서 관찰되었으며, 특히 연설 처리(speech processing) 동안 신경 활동의 동기화와 정보 처리 능력을 향상시키는 데 기여합니다. 이 아키텍처는 음소(phoneme) 시퀀스를 예측하기 위해 TIMIT 데이터셋을 사용하여 전체적으로(end-to-end) 훈련되었습니다.

- **Performance Highlights**: 아키텍처는 여러 계층에서 신경망의 성능을 향상시키며, 특히 4-6 계층에서 성능이 최고점에 도달했습니다. 추가로, 스파이킹 신경망(spiking neural network, SNN)의 스케일을 키우는 것이 가능하며, 최대 1,000개의 뉴런까지 성능이 일관되게 향상됩니다. SFA 및 재귀 연결의 사용은 효과적인 동기화 및 낮은 음소 오류율(phoneme error rate, PER)을 촉진시키는 것으로 나타났습니다.



### Information Re-Organization Improves Reasoning in Large Language Models (https://arxiv.org/abs/2404.13985)
Comments: 10 pages, 3 figures

- **What's New**: 이 논문에서는 대규모 언어 모델(LLM: Large Language Models)의 추론 능력을 개선하기 위해 새로운 접근법, 정보 재구성(InfoRE: Information Re-organization) 방법을 제안합니다. 기존 방법들이 추론 과정의 개선에 집중하는 반면, 이 연구는 문맥 내 논리 관계를 명확히 식별하고 이를 기반으로 추론을 진행하는 방식을 강조합니다.

- **Technical Details**: 연구팀은 문맥 콘텐츠(예: 문서, 단락)를 재구성하여 논리 관계를 파악하고, 이를 추론 과정에 활용합니다. 정보 재구성은 MindMap 구조를 사용하여 문맥 내 암시된 논리 관계와 다단계(multi-hop) 연결을 드러내는 데에 사용됩니다. 이 방식을 통해 LLM은 문맥을 보다 깊이 이해하고, 더 정확하고 신뢰할 수 있는 추론 결과를 도출할 수 있습니다.

- **Performance Highlights**: 기존의 추론 방식 대비 평균 3%의 성능 향상을 보여주며, Llama2-70B, GPT-3.5, GPT-4 모델을 사용한 다양한 문맥 인식 다단계 추론 과제에서 효과를 검증했습니다. 연구 결과는 zero-shot 설정만을 사용하여 이루어졌으며, 이는 제안된 방법이 LLM의 추론 성능을 개선할 잠재력을 보여줍니다.



### Protecting Your LLMs with Information Bottleneck (https://arxiv.org/abs/2404.13968)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)이 유해 콘텐츠를 생성할 수 있는 공격에 취약한 것을 인식하고, 새로운 방어 기법인 'Information Bottleneck Protector (IBProtector)'를 도입하였습니다. IBProtector는 정보 병목(information bottleneck) 원칙에 기반을 두어, 최적화된 또는 수동적 적대적 프롬프트(adversarial prompts)에 의한 탈옥 공격(jailbreaking attacks)을 방지합니다.

- **Technical Details**: IBProtector는 프롬프트를 선택적으로 압축하고 교란시켜, 타겟 LLM이 예상 응답을 제공하도록 필수 정보만을 보존합니다. 이 방어 메커니즘은 가벼우면서도 학습 가능한 추출기(trainable extractor)를 통해 운영되며, 기울기(gradient)가 보이지 않는 상황에서도 LLM과 호환될 수 있도록 설계되었습니다.

- **Performance Highlights**: IBProtector는 응답 품질이나 추론 속도(inference speed)에 크게 영향을 주지 않으면서 현재의 방어 방법보다 탈옥 시도를 완화하는 데에서 뛰어난 성능을 보였습니다. 이 방법은 다양한 공격 방법과 대상 LLM에 걸쳐 효과적이며 적용 가능성을 보여주어, LLM의 보안을 강화하는 새롭고 이전할 수 있는 방어 방법으로서의 잠재력을 강조합니다.



### How Well Can LLMs Echo Us? Evaluating AI Chatbots' Role-Play Ability  with ECHO (https://arxiv.org/abs/2404.13957)
Comments: 9 pages

- **What's New**: 이 연구는 대규모 언어 모델(Large Language Models, LLMs)의 역할극 능력을 평가하기 위해 일반 인물 모방에 초점을 맞춘 새로운 평가 프레임워크 ECHO를 소개하였습니다. 이는 유명 인사나 가상 인물이 아닌 일반인을 모방함으로써 디지털 인간 복제와 비디오 게임 내 비플레이어 캐릭터(NPC)의 발전 가능성을 탐구합니다.

- **Technical Details**: ECHO 프레임워크는 튜링 테스트에 영감을 받아 설계되었습니다. 이는 대상 개인의 지인들을 초대하여 인간과 기계가 생성한 반응을 구분하도록 함으로써 LLM의 역할극 능력을 평가합니다. 이 연구에서는 GPT-3.5와 GPT-4를 기반 모델로 사용하며, OpenAI의 GPTs를 온라인 응용 프로그램으로 활용하여 세 가지 역할극 LLM을 평가하였습니다.

- **Performance Highlights**: GPT-4는 인간 평가자를 가장 효과적으로 속였으며, GPTs는 48.3%의 높은 성공률을 달성하였습니다. 또한 연구는 LLM이 인간 생성 텍스트와 기계 생성 텍스트를 구별할 수 있는지 여부를 조사했는데, GPT-4는 차이를 식별할 수 있었으나 어떤 텍스트가 인간에 의해 생성되었는지 정확히 판단하지 못했습니다.



### Typos that Broke the RAG's Back: Genetic Attack on RAG Pipeline by  Simulating Documents in the Wild via Low-level Perturbations (https://arxiv.org/abs/2404.13948)
Comments: Under Review

- **What's New**: 이 연구에서는 정보 검색을 강화한 생성 모드인 Retrieval-Augmented Generation (RAG)의 견고성을 평가할 때, 주로 간과된 두 가지 측면, 즉 문서의 소음에 대한 취약성과 RAG의 전체적인 평가 방법을 다룹니다. 또한, RAG의 취약점을 폭로하고 시스템 전체 기능성을 소음이 많은 문서에 대해 시험하는 새로운 공격 방법인 Genetic Attack on RAG (GARAG)을 소개합니다.

- **Technical Details**: GARAG는 RAG 시스템의 각 구성 요소의 취약점을 드러내고 전체 시스템 기능성을 소음이 많은 문서에 대해 테스트하기 위해 설계되었습니다. 이를 통해 RAG 시스템이 실제 데이터베이스의 잠재적 위협, 예를 들어 사소한 텍스트 오류에 얼마나 취약한지를 평가합니다. 다양한 retrievers와 Large Language Models (LLMs)를 포함하는 표준 QA 데이터셋에 GARAG를 적용하여 RAG의 견고성을 검증합니다.

- **Performance Highlights**: 실험 결과, GARAG는 높은 공격 성공률을 일관되게 달성했습니다. 이 공격 방법은 각 구성 요소의 성능뿐만 아니라 그들 간의 상호 작용에도 심각한 손상을 입혀, 사소한 텍스트 오류가 실제 세계에서 RAG 시스템에 미칠 수 있는 중대한 위험을 강조합니다. 이러한 결과는 RAG 시스템의 실생활 적용 시 견고성을 향상시킬 필요성을 시사합니다.



### A User-Centric Benchmark for Evaluating Large Language Models (https://arxiv.org/abs/2404.13940)
- **What's New**: LLM(Large Language Models)의 새로운 사용자 중심 평가 방식의 제안에서는, 기존의 모델 능력 중심의 벤치마크와는 달리 실제 사용자의 사용 사례와 의도를 바탕으로 한 데이터셋과 평가 설계를 도입합니다. 이 연구는 전세계 23개국에서 온 712명의 참여자가 제공한 1,863개의 실제 사용 사례를 포함한 'User Reported Scenarios (URS)' 데이터셋을 구축하였고, 다양한 사용자의 의도를 7가지 카테고리로 분류하여 LLM의 효율을 평가합니다.

- **Technical Details**: 본 연구는 다양한 문화적 배경을 포함한 사용자가 제공한 데이터를 기반으로 LLM의 효과를 평가함으로써 국제적 다양성을 고려합니다. 사용자 의도에 따라 분류된 벤처마킹 방식은 사용자가 특정 서비스를 선택할 때 모델 능력에 대한 구체적인 지식이 없는 경우에도 쉽게 접근할 수 있도록 설계되었습니다. 평가는 사용자 의도(사용자가 자체 선택한 의도)에 초점을 맞추어, LLM이 실제 사용자의 요구를 얼마나 잘 충족시키는지를 측정합니다.

- **Performance Highlights**: URS 벤치마크 점수는 다양한 의도를 가진 LLM 상호작용에서 사용자가 보고한 경험과 잘 일치한다고 나타났습니다. 이는 LLM 서비스가 사용자의 주관적 시나리오에 대해 개선이 필요함을 시사하며, 사용자의 선호도와 더 잘 일치하도록 하는 방향으로 나아가야 함을 강조합니다. 또한, 이 연구는 국제적으로 다양한 사용자의 의견을 반영하여 LLM 평가가 전반적으로 더 균형 잡힌 시각을 제공하고 있음을 보여줍니다.



### MARIO Eval: Evaluate Your Math LLM with your Math LLM--A mathematical  dataset evaluation toolk (https://arxiv.org/abs/2404.13925)
- **What's New**: 이 연구에서는 수학적 문제 해결 능력을 평가하기 위한 새로운 도구인 수학 평가 툴킷(Mathematical Evaluation Toolkit)을 소개합니다. 이 툴킷은 파이썬 컴퓨터 대수 시스템(CAS)과 선택적으로 대형 언어 모델(LLM: Large Language Model)을 통합하여 수학적 답변의 채점을 향상시키는 데 중점을 둡니다. 이 툴킷은 두 개의 데이터셋에 대한 수동 주석(annotation)을 통해 검증되었고, 기존의 방법들과 비교하여 더 강력한 평가 결과를 제공함을 보여줍니다. 또한, LLM을 통합할 경우 평가 효능이 더욱 향상됩니다.

- **Technical Details**: 툴킷은 두 단계로 구성되어 있습니다. 첫 번째 단계에서는 예상되는 답변의 유형을 분류하고, 두 번째 단계에서는 예측된 답변과 기대된 답변 사이의 동등성을 평가합니다. 수학적 개념을 기반으로 하는 답변 유형은 다양하며(예: 실수(Real), 복합 구조(Matrix) 등), 이에 대한 평가는 Python 및 SymPy와 연동되어 정의된 유형에 따라 진행됩니다. LLM을 통합함으로써, 답변 유형의 분류와 동등성 판단의 정밀도를 높일 수 있습니다.

- **Performance Highlights**: 툴킷은 MATH와 GaoKao2023-Math-En(GK2023) 데이터셋에 적용되어 테스트되었습니다. 그 결과, LLM을 통합한 하이브리드 접근 방식은 Python 패키지의 수치 정밀도와 LLM의 자연어 처리 능력을 효과적으로 활용하여 약 97%의 높은 정확도를 달성했습니다. 이는 gpt-3.5-turbo와 같은 기존 도구들과 비교하여 뛰어난 성능을 보였습니다. 또한, 다양한 평가 지표(동등성 정확도(equivalence accuracy), 유형 분류 정확도(type classification accuracy), 해결 정확도(solution accuracy))에서 우수한 결과를 나타내 연구 결과의 공정성과 일관성을 높입니다.



### Navigating the Path of Writing: Outline-guided Text Generation with  Large Language Models (https://arxiv.org/abs/2404.13919)
Comments: under review

- **What's New**: 이 연구에서는 '라이팅 패스(Writing Path)'라는 새로운 프레임워크를 제안합니다. 이는 대형 언어 모델(LLMs)을 사용하여 사용자의 의도에 맞는 고품질의 목표 지향적 글쓰기를 가능하게 하기 위한 명시적 아웃라인(outlines)을 사용합니다. 이 프레임워크는 구조화된 글쓰기 계획과 추론 경로에서 영감을 받아, 사용자의 의도를 글쓰기 과정 전반에 걸쳐 포착하고 반영하는 것에 중점을 둡니다.

- **Technical Details**: 라이팅 패스는 메타데이터 준비, 초기 아웃라인 및 타이틀 생성, 정보 탐색, 향상된 아웃라인 생성, 텍스트 작성 등 다섯 단계로 구성됩니다. 사용자의 의도를 반영하기 위해 아웃라인을 생성하고, 추가 정보를 통해 이를 강화합니다. 이 프레임워크는 GPT-3.5-turbo, GPT-4, HyperCLOVA X와 같은 다양한 LLMs와 함께 사용되었습니다.

- **Performance Highlights**: 이 연구의 평가 결과, 라이팅 패스를 사용한 LLMs는 사용자 의도를 더 잘 반영하여 높은 품질의 텍스트를 생성하였습니다. 이는 다양한 도메인에서 구축된 자유 형식 블로그 텍스트 데이터셋을 통해 검증되었으며, LLM과 인간 평가 모두에서 긍정적인 평가를 받았습니다. 이는 중간 아웃라인을 사용하지 않는 기존 방법들보다 품질이 향상된 결과임을 시사합니다.



### Generating Attractive and Authentic Copywriting from Customer Reviews (https://arxiv.org/abs/2404.13906)
- **What's New**: 이 연구는 고객 리뷰를 바탕으로 매력적이고 신뢰할 수 있는 상품 설명(copywriting)을 생성하는 새로운 접근법을 제안하고 있습니다. 기존의 상품 속성에만 의존하던 방식에서 벗어나, 고객 리뷰에서 얻은 실제 사용 경험을 통합하여 정보가 풍부하고 진정성 있는 내용을 생성합니다. 또한 이 연구는 강화 학습(Reinforcement Learning) 기법을 활용하여 설명의 매력(attractiveness), 충실성(faithfulness), 정보 밀도(information density)를 동시에 강화하였습니다.

- **Technical Details**: 연구팀은 시퀀스-투-시퀀스(sequence-to-sequence) 프레임워크에 강화 학습을 접목하여 각각의 보상 모델(reward models)을 구현했습니다. 이 모델들은 설명의 매력, 충실성, 정보 밀도를 각각 측정하고 최적화하는 데 사용됩니다. 특히 매력도를 평가하기 위해 GPT-3.5를 사용하여 문장의 매력을 평가하는 모델을 학습하였으며, 이는 전통적인 이진 분류(binary classification) 방식보다 더 정확하게 사람의 판단과 일치하는 것으로 나타났습니다. 또한, 자연어 추론(Natural Language Inference, NLI) 모델을 이용하여 텍스트의 충실성을 높이는 방법을 채택했습니다.

- **Performance Highlights**: 이 프레임워크는 LLaMA-2-chat-7B 및 GPT-3.5와 같은 대규모 언어 모델들을 포함하여 기존의 모든 기준모델들(baselines)을 성능면에서 능가했습니다. 매력도와 충실성 모두에서 뛰어난 결과를 보여주었고, 마케팅 분야에서 강화 학습을 사용하여 복잡한 문제를 해결할 수 있는 가능성을 보여주었습니다. 뿐만 아니라 연구팀은 GPT-3.5를 사용하여 마케팅 분야 데이터셋을 구축하는데 효과적이라는 결과를 발표했습니다.



### Towards Better Text-to-Image Generation Alignment via Attention  Modulation (https://arxiv.org/abs/2404.13899)
- **What's New**: 텍스트에서 이미지 생성 작업에서 발전된 디퓨전(diffusion) 모델을 사용함에도 불구하고, 복잡한 텍스트 프롬프트를 처리할 때 문제가 발생합니다. 이를 해결하기 위해, 저희는 교육 없이 실행 가능한 단계별 주의력 집중 기법을 제안하였으며, 이는 특히 주의력 분산의 조절을 통해 엔티티 유출(entity leakage)과 속성 정렬(attribute misalignment) 문제를 개선합니다.

- **Technical Details**: 저희가 제안한 방법은 세 가지 주요 요소를 포함합니다: 1) 자기주의(self-attention) 온도 제어 기능을 통한 엔티티 경계 개선, 2) 객체 중심 교차 주의 마스킹(object-focused cross-attention mask)을 사용한 주의력 유출 차단, 3) 단계별 동적 재가중치 전략(phase-wise dynamic reweighting strategy)을 통해 생성 과정의 다양한 단계에서 프롬프트의 서로 다른 의미 구성 요소에 대한 강조를 조정합니다.

- **Performance Highlights**: 실험 결과, 저희 모델은 이미지-텍스트 정렬성에서 뛰어난 성능을 보이며 추가적인 계산 비용 없이 상태-의-기술(state-of-the-art) 결과를 달성합니다. 생성 시간은 기존 모델 SDXL과 거의 차이가 없으며, 다양한 정렬 시나리오에서의 성능이 입증되었습니다.



### VALOR-EVAL: Holistic Coverage and Faithfulness Evaluation of Large  Vision-Language Models (https://arxiv.org/abs/2404.13874)
Comments: Work in process

- **What's New**: 이 연구에서는 큰 시각-언어 모델(Large Vision-Language Models, LVLMs)의 환각 문제에 대해 새로운 다차원 벤치마크(VALOR-Bench)와 평가 프레임워크(VALOR-Eval)를 도입합니다. 이는 객체, 속성(attribute), 관계(relation) 등 다양한 차원에서의 환각을 평가하여 기존의 벤치마크들이 간과했던 세부적인 차이와 어휘의 개방성을 고려한 평가를 가능하게 합니다.

- **Technical Details**: VALOR-Bench는 연관 편향(associative biases)을 기반으로 도전적인 이미지를 선택하여 구성되며, LVLMs의 환각 취약성을 노출시키는데 중점을 둡니다. VALOR-Eval 메트릭은 기존 CHAIR 메트릭을 일반화하고, 대규모 언어 모델(Large Language Model, LLM)-기반의 두 단계 평가 프로세스를 통해 객체, 속성, 관계 차원에서의 개방 어휘 환각을 평가하며, 정보량(coverage)도 함께 고려합니다.

- **Performance Highlights**: 실험 결과, VALOR-Eval은 기존의 방식들보다 인간의 평가와 더 잘 상관관계를 가지며, LVLMs는 정확도를 유지하면서도 정보의 범위를 넓히는 균형을 이루는데 어려움이 있음을 보여줍니다. 특히 GPT-4V(ision) 모델은 다른 모델들에 비해 더 많은 정보를 포함하고 있음에도 불구하고 높은 환각 경향을 보였습니다.



### Context-Enhanced Language Models for Generating Multi-Paper Citations (https://arxiv.org/abs/2404.13865)
Comments: 14 pages, 7 figures, 11th International Conference, BDA 2023, Delhi, India

- **What's New**: 이 연구는 다중 인용 문장(Multi-citation sentences) 생성을 위한 새로운 방법을 제안하고 있습니다. Large Language Models (LLMs)를 활용하여 단일 출처 논문과 다수의 목표 논문을 바탕으로 일관성 있는 단락을 생성합니다. 또한, 멀티-인용 인스턴스를 포함하는 MCG-S2ORC라는 새로운 데이터셋을 소개하며, 지식 그래프(Knowledge graphs)를 활용하여 인용 텍스트 생성 성능을 향상시킵니다.

- **Technical Details**: 이 연구에서는 LLaMA, Alpaca, Vicuna 등 세 가지 LLM을 사용하여 문장 생성 작업에 대한 Fine-tuning을 수행했습니다. Citation Text Generation (CTG) 과제를 수행하기 위해, 연구자들은 출처 문서의 요약과 대상 문서의 요약, 도입부, 결론을 포함하는 지식 그래프를 프롬프트에 통합하였습니다. 이러한 지식 그래프는 연구 논문의 요약에서 추출된 관계를 활용하여, Citation Text의 생성을 보다 정확하고 효과적으로 수행할 수 있도록 도와줍니다.

- **Performance Highlights**: 이 연구는 기존의 단일 인용 생성 모델들에 비해 우수한 성능을 보여주었습니다. Fine-tuned LLMs는 METEOR, Rouge-1, Rouge-2, 및 Rouge-L과 같은 다양한 평가 척도를 사용하여 평가되었고, 지식 그래프를 포함한 프롬프트 사용이 기준 모델들(Baseline models)에 비해 성능 향상을 가져왔다는 것을 경험적으로 관찰하였습니다.



### Understanding the role of FFNs in driving multilingual behaviour in LLMs (https://arxiv.org/abs/2404.13855)
Comments: 10 pages

- **What's New**: 이 연구에서는 대규모 언어 모델(Large Language Models, LLMs)의 다중 언어 처리(multilingual processing) 능력에 대해 탐구합니다. 특히, 연구진은 다중 언어의 행동을 층별로 평가하는 새로운 메트릭(metric)을 도입하고, 다양한 언어를 대상으로 모델 구조의 영향을 분석했습니다. 이는 다층 피드포워드 네트워크(Feed-Forward Networks, FFNs)의 하위 계층에서 차별화된 다언어 처리 패턴을 밝혀내는 데 중점을 둡니다.

- **Technical Details**: 본 논문은 FFN의 하위 층을 패턴 감지기 (detectors)로, 상위 층을 조합기(combinators)로 간주합니다. 각 층에서는 '활성 편평도(activation flatness)'라는 새로운 측정법을 사용하여 다양한 언어에 걸쳐 활동 패턴을 분석하고, 이를 통해 다언어 처리 및 언어별 서브-레이어(language-specific sub-layers)의 구별된 영역을 식별합니다. 연구는 네 가지 다른 크기의 XGLM 모델을 사용하여 실시되었으며, 이 모델들은 동일한 아키텍처를 공유하고 30개 다양한 언어로부터의 500B 토큰으로 훈련되었습니다.

- **Performance Highlights**: 다중 언어성을 분석한 결과, 모델 구조와 층 깊이(layer depth)가 다중 언어 처리 능력에 미치는 상호작용(interaction)을 확인할 수 있었습니다. 특히, '과층화(over-layerization)'라는 현상이 특정 모델 구성에서 발견되었는데, 이는 층을 늘리는 것이 다른 매개 변수의 조정 없이는 모델 성능을 저하시킬 수 있음을 보여줍니다. 또한, 언어 모델이 훈련되는 동안 입력 표현을 예상 출력에 가깝게 점진적으로 변환하는 '증분 예측(incremental prediction)' 과정을 통해 최종 예측이 이루어짐을 설명합니다.



### From LLM to NMT: Advancing Low-Resource Machine Translation with Claud (https://arxiv.org/abs/2404.13813)
Comments: 17 pages, 15 figures

- **What's New**: Anthropic의 크로드(Claude 3 Opus)는 다양한 언어 쌍에서 우수한 기계 번역 성능을 보여주며, 특히 영어를 대상 언어로 하는 번역에서 높은 자원 효율성(resource efficiency)을 보인다. 이전 연구와 비교하여 크로드는 저자원(low-resource) 언어 쌍에 대해서도 강력한 번역 성능을 보이며, 이는 기존 대형 언어 모델(LLM)들과 차별화된다. 또한, 크로드를 활용하여 합성 데이터를 생성하고, 이를 기존 신경 기계 번역(Neural Machine Translation, NMT) 모델에 적용하여 최신 기술 성능을 달성하는 새로운 접근 방식을 제시한다.

- **Technical Details**: 연구팀은 크로드 3 오퍼스 모델을 활용하여 영어를 대상 언어로 하는 다양한 저자원 언어 쌍을 번역하는 실험을 진행했다. 이때 크로드가 데이터 오염(data contamination) 증거를 보였음에도 불구하고, 새롭게 구축한 벤치마크를 통해 기존의 Google Translate와 NLLB-54B 같은 강력한 베이스라인을 초과하는 성능을 보였다. 또한, 번역 능력을 NMT 모델로 압축하여 추출(distillation)하는 기술을 사용해 저비용의 NMT 모델을 개선시킬 수 있음을 보여주었다.

- **Performance Highlights**: 크로드 3 오퍼스는 36개 언어 쌍에 대한 새로운 평가 벤치마크에서 상태 최신 기술(state-of-the-art) 번역 성능을 보였다. 특히, 영어로 번역할 때 다른 LLM들보다 높은 자원 효율성을 보였으며, 심지어 저자원 언어 쌍에 대해서도 NMT 시스템에 비해 우수한 성능을 보여 주었다. Yoruba-English 번역에서는 NLLB-54B 및 Google Translate와 같은 강력한 베이스라인을 만나거나 초과하는 성능을 달성했으며, 이는 합성 데이터와 지식 추출을 활용한 결과이다.



### Lightweight Connective Detection Using Gradient Boosting (https://arxiv.org/abs/2404.13793)
Comments: 7 pages, 2 figures, 5 tables

- **What's New**: 이 작업에서는 경량화된 담론 연결 검출(discourse connective detection) 시스템을 소개합니다. 보통의 저 복잡성 기능에서 훈련된 그래디언트 부스팅(Gradient Boosting)을 활용한 이 접근 방식은 심층 신경망(deep neural networks)에 의존하는 기존 접근 방식의 계산 요구를 피합니다. 그로인해 이 기법은 CPU에서도 시간적 이득을 제공할 수 있음을 보여줍니다.

- **Technical Details**: 기법은 심플하지만 경쟁력 있는 결과를 달성하며, 연결 문구(connectives)를 포함한 담론 구조(discourse structure)의 주요 구성 요소를 감지합니다. 비결은 간단한 언어적 특성(linguistic features)을 사용하여 담론 연결부(discourse connectives, DC)와 비담론 연결부(non-discourse connective, NDC)를 구분하는 것입니다. 또한, 모델은 영어(English)와 터키어(Turkish) 두 가지 언어에서 안정적인 성능을 보여주었으며 이는 다국어 시나리오(multilingual scenario)에서의 강인함을 시사합니다.

- **Performance Highlights**: 이 모델은 국제적인 데이터셋(PDTB 2.0과 터키어 담론 데이터 은행 Turkish Discourse Bank (TDB) 1.0)에 대하여 테스트 되었으며, 상태-아트(state-of-the-art) 모델들과 유사한 결과를 보여줍니다. 경량 연결 검출 모델에서 가장 중요하다고 여겨지는 특징은 동사 기반 특징(verb-based features)입니다. 본 연구에서 사용된 그라디언트 부스팅 방법은 XGBoost를 이용해 구현되었습니다.



### Evaluating Retrieval Quality in Retrieval-Augmented Generation (https://arxiv.org/abs/2404.13781)
- **What's New**: 이 연구에서는 검색 기능이 향상된 생성 모델(RAG: Retrieval-Augment Generation)을 평가하는 새로운 방법인 eRAG을 제안합니다. eRAG은 각각의 문서를 대규모 언어 모델(LLM: Large Language Model)과 개별적으로 결합하여 출력을 생성하고, 이 출력을 이용하여 문서의 관련성을 평가합니다. 이는 전통적인 평가 방식과 다르게, 각 문서의 하류 작업 수행 능력을 통해 문서의 관련성을 직접적으로 표시합니다.

- **Technical Details**: eRAG 방법론은 RAG 시스템 내에서 검색 모델을 평가하기 위해 고안되었습니다. 검색된 각 문서를 LLM에 입력하여 개별적으로 출력을 생성하고, 이 출력을 기반으로 문서의 관련성을 평가합니다. 평가는 다양한 하류 작업 지표(예: 정확도, 완전 일치, ROUGE)를 사용하여 수행됩니다. 또한, 집합 기반 또는 순위 메트릭을 사용하여 각 검색 결과 목록에 대한 단일 평가 점수를 계산합니다.

- **Performance Highlights**: eRAG은 기존 방법들과 비교하여 RAG 시스템의 하류 성능과 더 높은 상관 관계를 보여줍니다. 특히, Kendall의 타우(tau) 상관계수는 0.168에서 0.494의 향상을 보였습니다. 이 평가 방법은 또한 계산적인 이점을 제공하며, GPU 메모리 사용량을 최대 50배까지 줄일 수 있습니다. 이를 통해 연구자들이 이 분야에서 더 효율적으로 연구를 진행할 수 있도록 eRAG의 구현체를 공개하고 있습니다 (https://github.com/alirezasalemi7/eRAG).



### Automated Text Mining of Experimental Methodologies from Biomedical  Literatur (https://arxiv.org/abs/2404.13779)
- **What's New**: 이 연구에서는 생물의학 문헌 분류를 위해 특별히 조정된 'fine-tuned DistilBERT' 모델을 제안하고 있습니다. 이 모델은 BERT 모델의 크기를 40% 감소시키면서도 속도는 60% 향상시켰으며, 기존의 RNN이나 LSTM을 사용하는 전통적인 분류 방법보다 뛰어난 성능을 보여주었습니다.

- **Technical Details**: 이 연구에서는 생물의학 분야에서 효과적인 NLP(Natural Language Processing)를 구현하기 위해 32,000개의 초록(abstract)과 전체 텍스트 기사를 대상으로 사전 훈련된 DistilBERT 모델을 사용하였습니다. 또한, 이 모델은 다양한 메소드로 분류된 20,000개 이상의 코퍼스를 이용하여 미세 조정되었습니다. 이를 통해 의학 문헌의 방법론을 분류하는 데 있어서 낮은 평가 및 훈련 손실을 달성했습니다.

- **Performance Highlights**: fine-tuned DistilBERT 모델은 전통적인 생물의학 문헌 분류 방법과 비교하여 우수한 성능을 보였습니다. 특히, 모델은 미세 조정을 통해 실험 설계 및 실험 방법에 대한 용어를 추출하는 기능을 향상시켜, 관련 기술을 정확하고 신뢰성 있게 식별할 수 있게 되었습니다.



### Using Adaptive Empathetic Responses for Teaching English (https://arxiv.org/abs/2404.13764)
Comments: Accepted to BEA workshop at NAACL 2024

- **What's New**: 이 연구에서는 영어 교육에 공감적 반응을 도입한 챗봇을 제안하고 있습니다. 이는 학습자의 부정적 감정을 탐지하고 그에 따라 적절한 피드백을 제공하는 기능을 통해 학습자의 감정적 지원을 강화하는 새로운 접근 방식입니다.

- **Technical Details**: 이 시스템은 사용자의 오디오를 분석하여 부정적 감정을 감지하고, 감지된 감정에 맞춤형으로 공감적 반응을 제공합니다. 이를 위해 다음과 같은 기술을 사용하였습니다: ChatGPT 자동 프롬프트 최적화(automatic prompt optimization), 최신의 음성 대 텍스트 변환 기술인 Whisper medium, 음성 합성을 위한 SpeechT5. 또한, 사용자의 발화에서 문법적 오류를 정정하는 기능도 포함되어 있습니다.

- **Performance Highlights**: 초기 사용자 연구를 바탕으로, 이 챗봇 시스템은 학습자들에게 긍정적인 인식을 받았으며, 이는 학습자의 장기적인 L2 grit (제2언어 학습에 대한 열정과 끈기)을 증진시킬 잠재력을 시사합니다. 또한 시스템은 공감적인 피드백과 감정 탐지를 위한 모델-기반 접근 방식을 사용하여 부정적 감정을 성공적으로 측정하고 반응하는 것으로 나타났습니다.



### How to Encode Domain Information in Relation Classification (https://arxiv.org/abs/2404.13760)
Comments: Accepted at LREC-COLING 2024

- **What's New**: 이 연구는 도메인 정보를 인코딩하여 관계 분류(Relation Classification, RC) 작업에서 다중 도메인 학습 설정을 탐구합니다. 연구진은 도메인별 데이터 세트를 통합하여 CrossRE 데이터 세트를 확장하고, 새로운 멀티 도메인 학습 기준을 제안하며, 이를 통해 성능을 향상시키는 다양한 방법을 비교합니다.

- **Technical Details**: 이 연구에서는 도메인 정보를 단어 임베딩(word embedding)과 특수 토큰(special tokens)을 사용하여 인코딩하는 두 가지 접근 방식을 비교합니다. 첫 번째는 도메인별 임베딩을 사용하여 입력 인스턴스를 풍부하게 하고, 두 번째는 입력 텍스트에 도메인을 표시하는 특수 토큰을 추가하는 것입니다. 또한, 도메인별 엔티티 유형(예: 음악가, 정치 당)을 사용하여 관계 레이블을 정확하게 식별하는 데 필요한 정보를 제공합니다.

- **Performance Highlights**: 새롭게 제안된 모델은 베이스라인 대비 Macro-F1 스코어에서 2점 이상의 개선을 보였습니다. 특히, 도메인 의존적 관계(예: 'part-of')는 도메인 정보 인코딩시 성능이 크게 향상되었습니다. 하지만 모든 클래스가 동일한 혜택을 받지는 않으며, 도메인 간 유사한 공간을 차지하는 클래스(예: 'physical')는 가장 적은 혜택을 받습니다.



### Embarrassingly Simple Unsupervised Aspect Based Sentiment Tuple  Extraction (https://arxiv.org/abs/2404.13751)
Comments: 4 pages, 4 tables, 3 figures, 2 appendix pages

- **What's New**: 이 연구에서는 감정 분석(Aspect Based Sentiment Analysis, ABSA)의 새로운 접근 방식으로 지도 학습이 아닌 비지도 학습 방식을 제안하여, 주어진 문장 내에서 특정 측면(aspect terms)과 관련된 의견어(opinion terms) 및 감성 극성(sentiment polarity)을 추출합니다. 특히, 비지도 방식을 사용하여 새로운 벤치마크를 설정하고, 이는 레이블이 부족한 도메인에서 효과적일 수 있습니다.

- **Technical Details**: 제안된 방법은 Part-of-Speech (POS) 태거와 도메인에 적응된 단어 임베딩을 사용하여 구현됩니다. 이 연구는 비지도 학습을 통해 감성 어휘 및 해당 측면의 감성 극성을 분류할 수 있는 첫 번째 시도로서, 다양한 벤치마크 데이터셋에서의 실험을 통해 그 효과가 입증되었습니다.

- **Performance Highlights**: 실험 결과, 레이블이 있는 인스턴스의 가용성이 서브태스크(subtasks)에서 약 1.8%의 성능 향상을 보였고, 모델 크기를 증가시킬 경우 전반적인 성능이 약 2% 개선되었습니다. 또한, 이 방법은 도메인 일반화(domain generalizability) 측면에서도 유효성이 관찰되었습니다.



### Trojan Detection in Large Language Models: Insights from The Trojan  Detection Challeng (https://arxiv.org/abs/2404.13660)
- **What's New**: 이 연구에서는 대규모 언어 모델(LLMs)의 트로이 목마 공격을 탐지하고 평가하는 2023 트로이 목마 감지 경쟁(Trojan Detection Competition 2023, TDC2023)의 도전과 통찰력을 탐구합니다. 이 경쟁은 의도하지 않은 트리거(unintended triggers)와 의도한 트리거(intended triggers)를 구분하는 어려움, 트로이 목마를 역공학하는 실현 가능성을 조사했습니다.

- **Technical Details**: 다양한 트로이 목마 탐지 방법의 비교 분석에서 높은 Recall 점수를 달성하는 것이 Reverse-Engineering Attack Success Rate (REASR) 정량을 얻는 것보다 훨씬 더 도전적임이 밝혀졌습니다. 대회의 최고 성과 메소드들은 Recall 점수가 0.16 정도로, 주어진 훈련 데이터(training prefixes)와 유사한 분포에서 문장을 무작위로 샘플링하는 간단한 기준(baseline)과 비슷한 성능을 보였습니다.

- **Performance Highlights**: 이 연구는 트로이 목마가 삽입된 모델에서만 해로운 대상(harmful targets)을 제공했을 때 트로이 목마의 탐지 및 복구 가능성에 대한 의문을 제기합니다. TDC2023은 트로이 목마 탐지의 실현 가능성과 LLM 입력 프롬프트(input prompts) 최적화 기술 개선에 대한 중요한 관찰을 제공하며, LLM의 강건성과 해석 가능성에 대한 추가 연구의 필요성을 강조했습니다.



### PEACH: Pretrained-embedding Explanation Across Contextual and  Hierarchical Structur (https://arxiv.org/abs/2404.13645)
Comments: Accepted at IJCAI 2024

- **What's New**: 이 연구에서는 새로운 트리 기반 설명 기술인 PEACH(Pretrained-embedding Explanation Across Contextual and Hierarchical Structure)를 제안합니다. PEACH는 사전 훈련된 컨텍스트 임베딩(pretrained contextual embeddings)을 사용하여 텍스트 기반 문서가 어떻게 분류되는지를 트리 기반의 인간이 이해할 수 있는 방식으로 설명할 수 있습니다.



### Mixture of LoRA Experts (https://arxiv.org/abs/2404.13628)
Comments: 17 pages, 11 figures

- **What's New**: LoRA (Low-Rank Adaptation)는 대규모 사전 훈련된 모델의 튜닝에 널리 활용되어 왔으며, 그 효과와 효율성 덕분에 가장 일반적인 튜닝 방법 중 하나로 자리 잡았습니다. 본 논문에서는 다양한 하위 작업들(downstream tasks)에 걸쳐 모델의 우수성을 실현하기 위해 여러 LoRA를 통합하는 새로운 접근법인 'Mixture of LoRA Experts (MoLE)'를 제안하고 있습니다. MoLE은 계층적 제어(hierarchical control)와 무제한 분기 선택(unfettered branch selection)을 활용하여 LoRA 융합 성능을 향상시킵니다.

- **Technical Details**: MoLE 방법은 각 계층(layer)에서 훈련된 LoRA의 무게를 조절하는 '계층적 무게 제어(hierarchical weight control)'를 사용합니다. 이는 각 계층을 독립적인 전문가(individual expert)로 보고 게이팅 함수(gating function)를 이용하여 최적의 조합 무게(composition weights)를 학습함으로써 기존의 선형 산술 합성(linear arithmetic composition)의 문제점을 해결합니다. 또한, MoLE은 참조 튜닝 기반 조합(reference tuning-based composition)과는 달리, 여러 훈련된 LoRA를 효과적으로 조합하면서도 계산 비용이 낮습니다.

- **Performance Highlights**: 실험적 평가는 NLP 및 V&L(시각 및 언어) 분야에서 수행되었으며, MoLE이 기존의 LoRA 조합 방법들보다 우수한 성능을 나타내는 것을 입증하였습니다. MoLE은 두 가지 추론 모드(inference modes)를 제공하는데, 첫 번째 모드에서는 학습된 게이팅 기능을 사용하여 각 LoRA의 독특한 특성을 보존합니다. 두 번째 모드에서는 원하지 않는 LoRA를 수동으로 마스킹하고, 재학습 없이 무게를 재계산하고 분배할 수 있습니다. 이 두 모드는 서로 다른 시나리오에 적응할 수 있는 유연성을 제공합니다.



### NegotiationToM: A Benchmark for Stress-testing Machine Theory of Mind on  Negotiation Surrounding (https://arxiv.org/abs/2404.13627)
- **What's New**: 이 논문에서는 협상 상황에서 대화 참여자의 멘탈 상태를 추정하는 데 도움이 되는 새로운 벤치마크, NegotiationToM을 소개합니다. 이 벤치마크는 Belief-Desire-Intention (BDI) 모델에 기반하여, 대규모 언어 모델(Large Language Models, LLMs)이 현실적인 협상 시나리오에서 상대방의 욕구(desires), 믿음(beliefs), 의도(intentions) 같은 다차원적인 멘탈 상태를 얼마나 잘 추적할 수 있는지 평가합니다.

- **Technical Details**: NegotiationToM 벤치마크는 실제 대화를 기반으로 설계되었으며, 참여자들의 멘탈 상태를 순차적으로 추적하는 능력을 측정하기 위해 다양한 라운드의 대화 데이터를 포함합니다. 대화는 참여자들의 욕구, 믿음, 의도를 반영하며, 이를 통해 LLMs의 ToM 능력을 평가합니다.

- **Performance Highlights**: NegotiationToM 벤치마크를 통한 실험 결과, 최신 LLMs는 사람들에 비해 현저하게 낮은 성능을 보였습니다. 심지어 chain-of-thought (CoT) 방법을 사용하였음에도 불구하고, LLMs는 협상 대화에서 상대방의 멘탈 상태를 정확하게 파악하고 추론하는데 큰 어려움을 겪는 것으로 나타났습니다.



### The Branch Not Taken: Predicting Branching in Online Conversations (https://arxiv.org/abs/2404.13613)
- **What's New**: 이 연구에서는 대화의 나무 구조에서 새로운 댓글이 나뭇잎 노드(leaf node)에 달리는 것인지, 아니면 중간 노드(intermediate node)에 새로운 가지(branch)를 생성하는지를 예측하는 새로운 작업인 '가지 예측(branch prediction)'을 제안했습니다. 이를 위해 'GLOBS(Global Branching Score)'라는 딥 뉴럴 네트워크 모델을 제안하여 논의의 분기를 예측합니다. 이 모델은 Reddit 포럼의 대규모 토론 데이터를 사용하여 평가되었습니다.

- **Technical Details**: GLOBS 모델은 BERT 트랜스포머(BERT transformer)를 사용하여 대규모의 대화 트리에서 응답 예측 작업으로 미세 조정(fine-tuned)합니다. 그 후, 트리의 각 노드에 대한 '응답-대응 점수(reply-to scores)'를 생성하고, 이 점수를 나뭇잎 노드와 중간 노드에 대해 별도로 풀링(pooling)한 후, 다른 구조적 및 시간적 특징과 함께 신경 분류기(neural classifier)에 입력합니다. 이를 통해 모델은 각 댓글이 기존 대화에서 새로운 분기를 생성할지 여부를 예측할 수 있습니다.

- **Performance Highlights**: GLOBS는 'Change My View', 'Explain Like I’m Five', 그리고 'Ask Science' 등 세 가지 Reddit 토론 포럼에서 다양한 기존 모델들을 상당히 능가하는 성능을 보여 주었습니다. 모델의 성공에는 구조적, 시간적 및 언어적 특징이 중요하게 기여하며, 또한 분기는 대화 참여자 수가 많고 트리의 초기 단계에서 주로 발생하는 경향이 있음을 발견하였습니다.



### "A good pun is its own reword": Can Large Language Models Understand  Puns? (https://arxiv.org/abs/2404.13599)
- **What's New**: 이 연구에서는 대규모 언어 모델(LLMs)이 유머 생성 및 창작 작문에서 흔히 사용되는 말장난을 어떻게 이해하고 처리하는지에 대한 분석을 새롭게 진행하고 있습니다. 특히, 말장난 인식(pun recognition), 설명(pun explanation), 생성(pun generation) 등 세 가지 주요 작업을 통해 LLM의 말장난 이해 능력을 체계적으로 평가합니다.

- **Technical Details**: 연구팀은 기존의 자동 평가 지표를 활용할 뿐만 아니라 LLMs의 맥락 내 학습(in-context learning) 패러다임에 더 적합한 새로운 평가 방법 및 지표를 도입하였습니다. 이 새로운 지표들은 이전 지표들보다 더 엄격하게 LLM의 말장난 이해 능력을 평가하며 인간의 인식과 더욱 밀접하게 일치합니다.

- **Performance Highlights**: 연구 결과, LLM들이 말장난을 생성할 때 '게으른 말장난 생성(lazy pun generation)' 패턴을 보이는 것이 밝혀졌으며, LLM이 말장난을 이해하는 데 있어 주요하게 직면하는 도전 과제들이 식별되었습니다.



### E-QGen: Educational Lecture Abstract-based Question Generation System (https://arxiv.org/abs/2404.13547)
Comments: IJCAI 2024 Demo Paper

- **What's New**: 이 논문은 강의 요약을 기반으로 학생들의 질문을 생성하는 E-QGen 시스템을 소개합니다. E-QGen은 강의 준비 과정과 관련된 질의응답 세션에서 교육자의 준비 과정을 최적화하는 것을 목표로 합니다. 이 시스템은 교사들이 미리 답변을 준비하고 필요할 때 추가 자료를 제공할 수 있도록 지원할 것으로 기대됩니다.

- **Technical Details**: E-QGen 시스템은 강의의 추상적인 요약을 입력으로 받아 가능한 학생 질문들을 생성합니다. 이 시스템은 교육자들이 학생들의 가능한 질문에 대비하고, 필요한 경우 추가적인 교육 자원을 제공할 수 있도록 돕습니다. 이러한 방식은 강의 준비 과정을 효율적으로 만들고, 학생과 교사 간의 상호작용을 증진시킬 것으로 예상됩니다.

- **Performance Highlights**: E-QGen에 의해 생성된 질문들은 교사들이 미리 답변을 준비하는 데 도움을 주며, 교육 과정에서 학생들의 이해도를 높이는 데 기여할 수 있습니다. 시스템의 성능은 특히 교육자들이 학생들의 다양한 질문에 신속하게 대응하고, 보다 풍부한 교육 경험을 제공하는 데 중점을 둡니다.



### IMO: Greedy Layer-Wise Sparse Representation Learning for  Out-of-Distribution Text Classification with Pre-trained Models (https://arxiv.org/abs/2404.13504)
- **What's New**: 이 연구에서는 'IMO (Invariant features Masks for Out-of-Distribution text classification)'라는 새로운 방법을 제안하여, 텍스트 분류 문제에서 도메인 일반화(domain generalization)를 개선합니다. IMO는 학습 도중 불필요한 특징을 제거하면서 도메인 불변적인 특징을 학습하는 새로운 접근 방식을 사용합니다.

- **Technical Details**: IMO는 토큰 수준에서의 attention module을 통해 예측에 유용한 토큰에 집중하고, 학습 과정에서 sparse mask layers를 통해 불필요한 특징을 제거합니다. 이를 통해 도메인 불변적인 특징(domain-invariant features)과 인과 특징(causal features) 사이의 관계를 이론적으로 설명하며, 대규모 사전 학습된 언어 모델(pre-trained large language models, 이하 LLMs) 사용을 최적화합니다.

- **Performance Highlights**: 실험 결과, IMO는 BART와 결합하여 사용할 경우 ChatGPT와 같은 경쟁 모델보다 우수한 성능을 보이며, 특히 토픽 분류와 감정 극성 분석에서 강력한 베이스라인을 상당히 앞질렀습니다. 또한, 중국어 데이터셋에서 ChatYuan과 결합했을 때도 ChatGPT보다 뛰어난 성능을 보여줍니다. 또한, 학습 데이터의 크기에 따른 OOD 성능이 일관되게 유지되며, 학습 데이터의 양이 적을 때의 성능 저하가 크게 줄어들었습니다.



### Do "English" Named Entity Recognizers Work Well on Global Englishes? (https://arxiv.org/abs/2404.13465)
Comments: EMNLP Findings 2023

- **What's New**: 새롭게 구축된 Worldwide English NER Dataset은 다양한 지역에서 사용되는 저자원 영어 버전의 이름을 인식하기 위해 만들어졌습니다. 이 데이터셋은 전통적으로 사용되는 영어 데이터셋과는 달리, 아메리칸 영어나 브리티시 영어에만 집중되어 있는 기존의 데이터셋들과는 다르게 전 세계적으로 사용되는 다양한 영어 텍스트를 포함합니다.

- **Technical Details**: 이 연구에서는 Flair, SpaCy, Stanza, CoreNLP와 같은 여러 NER 도구를 사용하여 데이터셋에 대한 성능을 평가합니다. 특히, RoBERTa-Large와 ELECTRA-Large와 같은 전이 학습(transfer learning) 모델을 사용하여 다양한 언어 모델의 일반화 능력을 비교 분석하였습니다. 또한, CoNLL 2017의 단어 벡터들과 함께 재학습을 통해 모델의 성능을 높이려 하였습니다. 데이터는 BIOES 형식으로 처리되며, 상세한 범주화를 통해 관련 데이터를 더욱 잘 이해할 수 있도록 구성되었습니다.

- **Performance Highlights**: 기존의 데이터셋(CoNLL, OntoNotes)에서 학습된 모델들은 Worldwide English 데이터셋에 적용했을 때 성능이 크게 저하되는 것으로 나타났습니다. 특히 아프리카와 오세아니아 지역에서 높은 성능 저하를 보였습니다. 반면, 모델을 새롭게 조합한 데이터셋으로 재훈련시키면 기존 데이터셋과 Worldwide 데이터셋 양쪽에서의 성능을 유지하면서 전 세계적 범위에서의 성능이 크게 향상되었음을 확인할 수 있었습니다.



### Fine-Grained Named Entities for Corona News (https://arxiv.org/abs/2404.13439)
Comments: Published at SWAT4HCLS 2023: The 14th International Conference on Semantic Web Applications and Tools for Health Care and Life Sciences

- **What's New**: 이 연구는 코로나 뉴스 기사에서 언급된 엔티티를 체계적으로 태깅하고 추출하기 위해 이름 인식 모델(NER - Named Entity Recognition)을 훈련하고 평가하는 새로운 데이터 주석 파이프라인을 제안합니다. 특히 독일의 뉴스 채널 'Tagesschau'에서 발행된 기사를 대상으로 하여 최신 정보를 반영한 코퍼스를 구축하고, 이를 바탕으로 NER 모델을 훈련 및 테스트합니다.

- **Technical Details**: 이 주석 파이프라인은 은색 씨드(seeds)와 금색 씨드, 그리고 사전 훈련된 NER 모델(OntoNotes)을 활용하여 도메인 특화 엔티티(domain-specific entities) 및 일반 엔티티(generic entities)를 추출합니다. 주요 기술 플랫폼으로는 Flair NLP 프레임워크와 SciBERT가 사용되며, 이러한 모델들은 토큰화(tokenization), 품사 태깅(PoS tagging), 그리고 정확한 문자열 매칭 알고리즘을 통해 특정 엔티티를 식별합니다.

- **Performance Highlights**: Flair와 SciBERT를 활용하여 훈련된 NER 모델들은 정확도 및 적용성 측면에서 높은 성능을 나타냅니다. 테스트 데이터는 도메인 전문가들에 의해 수동으로 주석처리되었으며, 모델의 신뢰성을 확인하기 위해 Fleiss Kappa 값이 0.98로 계산되었습니다. 이를 통해 최신 코로나 바이러스 뉴스 내용의 변화를 식별하고 분석하는데 효과적으로 기여하고 있음을 확인할 수 있습니다.



### Retrieval-Augmented Generation-based Relation Extraction (https://arxiv.org/abs/2404.13397)
Comments: Submitted to Semantic Web Journal. Under Review

- **What's New**: 이 연구에서는 기존의 기계 학습 (ML: Machine Learning) 연계 방법과 대조적으로, 정보 추출 (IE: Information Extraction) 및 관계 추출 (RE: Relation Extraction) 작업을 위한 새로운 접근 방식인 RAG4RE(Retrieved-Augmented Generation-based Relation Extraction)를 제안하였습니다. 이 방법은 대규모 언어 모델 (LLMs: Large Language Models)의 한계를 극복하고, 성능을 개선하기 위해 검색 증강 생성 (RAG: Retrieved-Augmented Generation) 기술을 활용합니다.

- **Technical Details**: RAG4RE는 텍스트에서 엔티티 쌍 (예: 'National Congress of American Indians'와 '1994') 사이의 관계를 식별하는 RE 작업을 수행합니다. 이 접근 방식은 Flan T5, Llama2, Mistral 등 다양한 LLM을 활용하고, 이들의 효과를 TACRED, TACREV, Re-TACRED, SemEval RE 데이터셋을 이용해 평가했습니다. RAG4RE는 관련 정보를 검색하여 LLM의 입력으로 통합함으로써, 모델이 보다 정확한 RE를 수행할 수 있도록 지원합니다.

- **Performance Highlights**: 연구 결과, RAG4RE 접근 방식은 기존의 LLM 기반 RE 접근법들과 비교할 때 실제로 높은 성능을 보였습니다. 특히, TACRED 데이터셋과 그 변형들에서 뛰어난 결과를 보였으며, TACREV 데이터셋에서도 이전 방법론들에 비해 탁월한 성능을 나타냈습니다. 이는 RAG4RE가 RE 작업을 위해 잠재적으로 큰 영향을 미칠 가능성을 보여줍니다.



### Explanation based Bias Decoupling Regularization for Natural Language  Inferenc (https://arxiv.org/abs/2404.13390)
- **What's New**: 이 연구는 자연어 추론(Natural Language Inference, NLI)을 수행하는 트랜스포머(Transformer) 기반 인코더의 편향성 문제를 해결하기 위해 '설명 기반 편향 분리 규제(Explanation based Bias Decoupling Regularization, EBD-Reg)' 방법을 제안합니다. 기존의 탈편향(debiasing) 방법들은 주로 데이터셋의 편향된 샘플을 식별하는데 집중했지만, EBD-Reg는 편향된 구성 요소를 직접적으로 분리하고 키워드를 식별하는 데 중점을 둠으로써, 분포 외 추론(out-of-distribution inference)에서의 성능을 향상시키고자 합니다.

- **Technical Details**: EBD-Reg는 인간의 설명을 사용하여 인코더가 편향과 키워드를 구분하고, 이 두 요소를 분리하며, 두 추론의 조인트 예측 분포를 정렬하는 삼중 병렬 감독(tripartite parallel supervision)을 설정합니다. 연구팀은 또한 각 키워드와 편향의 주 추론 및 부 추론을 강화하기 위해 '적응적 토큰 수준 주의(Adaptive Token-level Attention, ATA)' 기법을 도입했습니다.

- **Performance Highlights**: EBD-Reg는 다양한 트랜스포머 기반 인코더와 쉽게 통합되며, 다수의 실험을 통해 기존의 탈편향 방법들을 상당히 능가하는 분포 외 추론 성능을 보여주었습니다. 인간의 설명을 기반으로 하는 이 접근 방식은 모델이 데이터셋의 개별 샘플의 편향된 부분을 명확히 식별하고 이해할 수 있도록 하여, 더욱 해석 가능한 최종 예측을 가능하게 합니다.



### MahaSQuAD: Bridging Linguistic Divides in Marathi Question-Answering (https://arxiv.org/abs/2404.13364)
Comments: Accepted at the International Conference on Natural Language Processing (ICON 2023)

- **What's New**: 이 연구는 저자원 언어에서 효율적인 QnA(Question and Answering, 질문응답) 데이터셋의 부재를 해결하기 위해 영어 SQuAD(Stanford Question Answering Dataset)를 번역하는 새로운 접근법을 도입했습니다. 새롭게 소개된 MahaSQuAD는 인도의 마라티어(marathi)를 위한 첫 번째 전체 SQuAD 데이터셋으로, 학습 데이터는 118,516개, 검증 데이터는 11,873개, 테스트 데이터는 11,803개의 샘플을 포함하고 있습니다. 또한, 수동 검증을 거친 500개의 금 테스트 세트(gold test set)도 제공합니다.

- **Technical Details**: 연구팀은 번역된 문맥에서 정확한 답변 위치를 찾는 새로운 방법을 개발하여 문장의 의미와 맥락을 유지하는 동시에, SQuAD를 저자원 언어로의 번역 과정에서 발생하는 언어적 뉘앙스와 맥락 유지 문제를 해결했습니다. 이를 통해, 마라티어와 같은 저자원 언어로의 효율적인 질문응답 시스템(QnA system)을 구축할 수 있게 되었습니다.

- **Performance Highlights**: MahaSQuAD 데이터셋과 함께, MahaBERT라는 BERT(Bidirectional Encoder Representations from Transformers) 모델의 마라티어 버전이 Fine-tuning을 거쳐 개발되었습니다. 이 모델은 GitHub 및 Hugging Face 플랫폼에서 공개적으로 공유되어, 연구자 및 개발자가 자유롭게 사용할 수 있습니다. 이러한 새로운 데이터셋과 모델은 마라티어 사용자가 자신의 모국어로 더 정확하고 맥락적으로 관련된 답변을 얻을 수 있게 하여, 정보 접근성을 크게 향상시켰습니다.



### Semantically Corrected Amharic Automatic Speech Recognition (https://arxiv.org/abs/2404.13362)
- **What's New**: 이 논문에서는 아프리카 동부의 주된 언어인 암하라어(Amharic)를 위한 자동 음성 인식(ASR) 도구를 제작하였습니다. 암하라어는 5천만 명 이상이 사용하는 언어로, Ge'ez 스크립트로 쓰여집니다. 이 연구는 기존의 암하라어 ASR 벤치마크가 문장의 의미에 큰 영향을 미치는 단어 경계의 띄어쓰기를 고려하지 않아, 실제 성능 측정이 과대평가됨을 발견했습니다.

- **Technical Details**: 우리는 기존 암하라어 ASR 테스트 데이터셋의 정정된 전사본을 처음으로 공개하고, 원시 ASR 출력을 문법적으로 완전하고 의미론적으로 유의미한 암하라어 문장으로 구성하는 트랜스포머 인코더-디코더(tranformer encoder-decoder) 아키텍처 기반 후처리 접근법을 도입했습니다.

- **Performance Highlights**: 수정된 테스트 데이터셋에서의 실험을 통해, 우리 모델은 암하라어 음성 인식 시스템의 의미론적 정확성을 향상시켰으며, 문자 오류율(Character Error Rate, CER)은 5.5%, 단어 오류율(Word Error Rate, WER)은 23.3%를 달성했습니다.



### Swa Bhasha: Message-Based Singlish to Sinhala Transliteration (https://arxiv.org/abs/2404.13350)
Comments: 6 pages, 6 figures, 2 Tables, Presented at International Conference on Innovations in Info-business and Technology, Colombo, February 2022

- **What's New**: 이 연구는 싱글리쉬(Singlish) 언어의 단어 수준에서 문제를 간소화하여 문제의 복잡성을 줄이는 전사(system transliteration)에 중점을 둡니다. 신규 코딩 시스템과 규칙 기반 접근법을 도입하여 모음이 없어도 싱할라(Sinhala)어와 일치하는 단어를 매핑할 수 있는 새로운 방법을 제시합니다.

- **Technical Details**: 이 시스템은 로마자 영문(Romanized English) 각 글자에 고유한 숫자 코드를 할당하여 각 단어에 대한 독특한 패턴을 구성합니다. 다양한 입력 패턴을 수집하고 분석하여 이와 관련된 고유한 싱글리쉬 패턴을 생성하였습니다. 퍼지 로직(fuzzy logic) 기반 구현법을 사용하였으며, 고유한 숫자 값을 포함하는 코딩된 사전도 구현되었습니다.

- **Performance Highlights**: 예를 들어, 'kiyanna, kianna, kynna, kynn, kiynna'와 같은 다양한 입력이 정확한 싱할라어 단어 'kiyanna'로 매핑되었습니다. 이 결과로 볼 때 'Swa Bhasha' 전사 시스템은 싱글리쉬에서 싱할라어로 문자 메시지를 작성하는 싱할라 사용자의 경험을 향상시킬 수 있는 능력을 가지고 있습니다.



### UnibucLLM: Harnessing LLMs for Automated Prediction of Item Difficulty  and Response Time for Multiple-Choice Questions (https://arxiv.org/abs/2404.13343)
Comments: Accepted at BEA 2024 (NAACL Workshop)

- **What's New**: 이 연구는 BEA 2024 공유 작업에서 사용된 퇴역 USMLE 다지선다형 문제 (MCQs)의 난이도 및 응답 시간 예측을 위해 대규모 언어 모델 (Large Language Models, LLMs)을 기반으로 한 새로운 데이터 증강 방법을 탐구합니다. 연구팀은 Falcon, Meditron, Mistral과 같은 제로-샷 LLM을 활용하여 데이터셋을 증강하는 방식을 제안합니다. 결과적으로 질문 텍스트를 포함하고 LLM 답변의 다양성에서 혜택을 받는 방법이 가장 성공적이라고 평가되어, 의료 면허 시험에서 자동 평가 개선에 LLM의 잠재력을 부각시켰습니다.

- **Technical Details**: 연구팀은 제로-샷 설정에서 LLM을 활용하여 원본 데이터셋에 답변을 추가했으며, 목표 레이블을 정규화하고, 추가적인 기능 조합을 생성하는 등의 여러 전처리 단계를 거쳤습니다. 이 데이터는 트랜스포머 기반 모델 (transformer-based models)을 미세 조정하는데 사용되었습니다. 실험에서는 질문 난이도 예측이 응답 시간 예측보다 더 복잡한 작업임을 발견했습니다. 추가로, 연구팀은 경쟁 후 오버피팅과 잘못된 특징 선택을 해결하기 위한 새로운 모델을 제시하여 기존에 제출된 모델보다 더 나은 결과를 얻었습니다.

- **Performance Highlights**: 연구 결과에 따르면 LLM에 의해 생성된 답변을 데이터셋에 추가하는 것이 러닝 및 응답 시간 예측에 긍정적인 영향을 미쳤습니다. 특히, 질문 텍스트와 LLM 생성 답변을 포함한 모델이 가장 효과적이었습니다. 이러한 방법은 의료 면허 시험의 자동화된 평가 개선에 LLM을 활용할 수 있는 가능성을 시사합니다.



### Beyond Accuracy: Investigating Error Types in GPT-4 Responses to USMLE  Questions (https://arxiv.org/abs/2404.13307)
Comments: 10 pages, 4 figures. Accepted for publication at the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2024)

- **What's New**: 이 연구는 GPT-4가 의학 QA(질문 응답, Question Answering) 작업에서 높은 정확도(86.70%)를 보임에도 불구하고, 그 설명 과정과 추론을 제공하지 않는 점을 개선하여 새로운 오류 분류법과 함께 GPT-4 의학 USMLE 오류(G4UE) 데이터셋을 개발했습니다. 이 데이터셋은 미국 의학 면허 시험(USMLE)에 대한 GPT-4의 올바른 답변 4153개와 잘못된 답변 919개를 포함하며, 각 응답은 평균 258단어로 상세한 설명을 제공합니다.

- **Technical Details**: 추가로 이 데이터셋은 Potato (페이 등, 2022에서 개발된 오픈소스) 주석 도구와 Prolific을 통해 채용된 44명의 의료 전문가에 의해 잘못된 데이터 300점을 세밀하게 주석 처리했습니다. 주석 처리된 데이터는 다중 라벨 스팬으로 오류의 원인을 식별하는 오류 유형 분류 작업을 도입하였습니다. 이 연구는 의료 분야에서 LLMs(Language Large Models)의 부정확성을 이해하고 해결하는 데 기여할 것입니다.

- **Performance Highlights**: 이 연구를 통해 GPT-4의 오답이 상당 부분이 'GPT-4에 의한 합리적인 응답'으로 분류됨을 밝히며, 심지어 훈련된 의료 전문가들 사이에서도 잘못된 옵션으로 이어질 수 있는 설명을 식별하는 도전을 강조합니다. 또한 GPT-4는 2023년 3월의 86.6%에서 2023년 6월에는 82.1%로 정확도가 하락하는 성능 드리프트(Performance Drift) 현상을 보였습니다.



### Evaluating Subword Tokenization: Alien Subword Composition and OOV  Generalization Challeng (https://arxiv.org/abs/2404.13292)
- **What's New**: 이 연구는 기존 언어 모델의 여러 하위 단어 토크나이저(Byte-Pair Encoding, Unigram Language Model)가 형태소 경계를 존중하지 않는 문제를 개선하기 위해 새로운 평가 프레임워크를 제안합니다. UniMorph Labeller 도구를 사용한 '내재적(intrinsic)' 평가와 OOV 일반화 도전(Out-of-Vocabulary Generalization Challenge)를 통한 '외재적(extrinsic)' 평가 방법이 소개되었습니다.

- **Technical Details**: UniMorph Labeller는 하위 단어 토크나이징을 '형태적(morphological)' 혹은 '외계의(alien)'로 분류합니다. 이 도구는 영어 단어 50만 개를 포함하는 개방형 소스 툴이며, 정확도는 98%입니다. 외재적 평가는 텍스트 분류 작업을 포함한 세 가지 새로운 하류(downstream) 작업을 통해 수행되며, 이는 언어 모델의 단어 의미의 의미론적 결합(semantic compositionality)에 대한 일반화 능력을 측정합니다.

- **Performance Highlights**: OOV 일반화 도전 결과에 따르면, 형태적 토크나이징이 외계 토크나이징보다 우수한 일반화 성능을 보였습니다. 이는 하위 단어의 의미론적 구성이 언어 모델의 성능에 중요한 영향을 미칠 수 있음을 시사합니다. 연구는 광범위한 언어 모델에서 실험을 수행했으며(예: ALBERT, BERT, RoBERTa, DeBERTa), 모든 모델에서 유사한 경향을 보였습니다.



### Double Mixture: Towards Continual Event Detection from Speech (https://arxiv.org/abs/2404.13289)
Comments: The first two authors contributed equally to this work

- **What's New**: 이 연구는 음성에서 지속적인 이벤트 감지(Continual Event Detection from Speech, CEDS)라는 새로운 과제를 제안합니다. 이 과제는 음성에 기반한 이벤트를 지속적으로 학습하고 식별하는 것을 목표로 하며, 특히 기존 방법들에서 보았던 치명적인 잊음(catastrophic forgetting) 문제와 음향(acoustic) 이벤트와 의미(sematic) 이벤트의 탈통합(disentanglement)에 중점을 둡니다. 이를 해결하기 위해, 저자들은 'Double Mixture' 방법을 도입하여 전문가의 혼합과 기억의 혼합을 통해 모델의 적응성과 기억 유지 능력을 향상시킵니다.

- **Technical Details**: Double Mixture 방법은 각 태스크에 전용 전문가를 자동으로 할당하는 혼합 전문가(mixture of experts)와 과거의 음성 경험을 재생하는 간단하지만 효과적인 기억 혼합(mixture of memory)을 결합합니다. 이 접근법은 끊임없이 변화하는 데이터 환경에 적응하고 새로운 이벤트 유형을 인식하는 동시에 이전에 학습한 정보를 유지하도록 설계되었습니다.

- **Performance Highlights**: 제안된 Double Mixture 방법은 기존의 최첨단(state-of-the-art) 방법들과 비교하여 최소의 잊음률과 최고의 일반화 능력을 보였습니다. 연구팀은 복잡한 실제 세계 음향 환경에서 다양한 이벤트 조합을 관리하는 데 있어 이 방법의 효과를 입증하였고, 이는 음성 기반 정보 추출의 복잡성을 다시 한번 확인시켜 줍니다.



### ISQA: Informative Factuality Feedback for Scientific Summarization (https://arxiv.org/abs/2404.13246)
Comments: 18 pages, 4 figures

- **What's New**: 이 연구에서는 대규모 언어 모델(LLM: Large Language Model)을 활용한 과학적 요약의 진실성을 향상시키기 위해 'Iterative Factuality Refining on Informative Scientific Question-Answering (ISQA)' 피드백 메커니즘을 제안합니다. 이 메커니즘이 과학 문서 요약 작업에서의 사실에 부합하는 내용의 생성을 강화하고, 비사실적 내용의 생성을 억제하기 위해 긍정적 및 부정적 피드백을 동시에 활용합니다.

- **Technical Details**: ISQA 피드백은 요약 모듈과 피드백 모듈 간의 반복적 상호작용을 통해 요약의 사실성을 향상시킵니다. 피드백 모듈은 문서의 요약에 대해 긍정적(사실적인 내용)과 부정적(비사실적인 내용) 피드백을 제공하며, 이를 기반으로 요약 모듈은 보다 정확한 요약을 생성합니다. 이 과정에는 'evidence-seeking QA' 과제가 도입되어, 요약에서의 증거 문장을 추출하고 이를 사실(Fact)과 비사실(Non-fact)으로 구분합니다.

- **Performance Highlights**: ISQA 메커니즘을 사용한 LLM은 다양한 과학 문서 데이터셋에서 요약 작업의 사실성을 현저히 개선시키는 것으로 나타났습니다. 특히, ISQA는 기존의 일반적인 피드백 방법과 프롬프트 엔지니어링을 적용한 경우보다 우수한 성능을 보여 주었습니다. 이 연구 결과는 크고 작은 LLM들이 고품질의 피드백을 생성할 수 있음을 보여주며, 결과적으로 높은 사실성을 가진 요약 결과를 초래합니다.



### Heterogeneous Subgraph Transformer for Fake News Detection (https://arxiv.org/abs/2404.13192)
- **What's New**: 이 연구는 가짜 뉴스를 효과적으로 탐지하기 위해 뉴스 주제, 엔티티, 콘텐츠 간의 관계에 대한 이질적 그래프(Heterogeneous Graph)를 구성하고, 이를 활용하는 새로운 이질적 부분그래프 변환기(Heterogeneous Subgraph Transformer, HeteroSGT)를 제안합니다. HeteroSGT는 가짜 뉴스가 포함된 부자연스러운 관계의 부분그래프를 탐지 및 분류하고, 뉴스의 진위를 확인하는 데 초점을 맞추고 있습니다.

- **Technical Details**: HeteroSGT는 먼저 사전 학습된 언어 모델을 사용하여 단어 수준과 문장 수준의 의미론을 추출합니다. 연구팀은 뉴스 별로 중심이 되는 부분그래프를 추출하기 위해 재시작(random walk with restart, RWR)을 적용하고, 추출된 부분그래프를 새롭게 제안된 부분그래프 트랜스포머(subgraph Transformer)에 입력하여 진위를 판별합니다.

- **Performance Highlights**: HeteroSGT는 5개의 실제 데이터셋에서 수행된 실험을 통해 다섯 가지 기준 모델을 능가하는 우수한 성능을 보였습니다. 정확도(Accuracy), 매크로-정밀도(macro-precision), 매크로-재현율(macro-recall), 매크로-F1 점수, ROC 곡선에서 모두 높은 성능을 나타내며, 추가적인 사례 연구와 심층 분석을 통해 HeteroSGT의 설계와 주요 구성 요소가 성능 향상에 기여함을 입증하였습니다.



### Beyond Self-Consistency: Ensemble Reasoning Boosts Consistency and  Accuracy of LLMs in Cancer Staging (https://arxiv.org/abs/2404.13149)
Comments: accepted to the 22nd International Conference on Artificial Intelligence in Medicine (AIME'24)

- **What's New**: 최신 연구에서는 대형 언어 모델(Large Language Models, LLMs)을 이용한 앙상블 추론(Ensemble Reasoning, EnsReas) 접근법을 제안하여 암 병기를 결정하는 병리 보고서의 일관성과 성능을 향상시키는 방법을 보여줍니다. 이 방법은 자기일관성(Self-Consistency, SC)을 사용하여 모델 세대의 일관성을 높이는 것을 목표로 합니다.

- **Technical Details**: 이 연구는 Med42-70B, 오픈 액세스 임상 LLM을 사용하여 실제 병리학 보고서에서 병리학적 암 단계를 결정합니다. 지금까지의 접근법과 달리, EnsReas는 초기의 일관되지 않은 예측을 구분하고 다시 평가하여 보다 비용 효율적인 방법을 제공합니다. EnsReas는 자동화된 암 병기 결정을 위해 세 가지 기본 프롬프팅 전략인 zero-shot (ZS), zero-shot chain-of-thought (ZS-CoT) 및 ZS-CoT with self-consistency (ZS-CoT-SC)를 구현하여 기준 성능을 제공합니다.

- **Performance Highlights**: EnsReas는 기존 방법(ZS 및 ZS-CoT)에 비해 예측 성능에서 우수한 결과를 보여주었으며, ZS-CoT-SC에 비해 예측 성능과 일관성 모두에서 뛰어난 성과를 보였습니다. 이는 신뢰성과 신뢰도가 중요한 임상 분야 또는 다른 도메인에서 이 모델의 활용 가능성을 시사합니다.



### Multi Class Depression Detection Through Tweets using Artificial  Intelligenc (https://arxiv.org/abs/2404.13104)
Comments: 33 pages

- **What's New**: 이 연구에서는 트위터(Twitter) 데이터베이스에서 다섯 가지 유형의 우울증(양극성, 주요, 정신병적, 비정형 및 산후)을 예측하며, 우울증의 유형을 나타내는 트윗의 부분을 강조하여 설명 가능한 AI(Explainable AI)를 사용하였습니다. 이는 이전 연구들이 단지 우울증의 감지와 강도만을 초점으로 맞췄던 것과 차별화되는 점입니다.

- **Technical Details**: 양방향 인코더 표현(Bidirectional Encoder Representations from Transformers, BERT)을 특징 추출 및 훈련을 위해 사용하였으며, 기계 학습(Machine Learning)과 심층 학습(Deep Learning) 방법론을 통해 모델을 훈련하였습니다. 트윗 스크래핑을 위해 Apify라는 플랫폼을 사용했으며, 도메인 전문가들에 의해 검증된 어휘들을 사용하여 트윗을 수집하고 레이블링하였습니다.

- **Performance Highlights**: BERT 모델은 특히 뛰어난 성능을 보여 전체적인 정확도가 0.96에 이르렀습니다. 이는 다양한 유형의 우울증을 정확하게 예측할 능력을 입증하며, 이러한 고성능은 설명 가능한 AI를 사용하여 진단의 투명성과 이해도 또한 향상시킵니다.



### Mathify: Evaluating Large Language Models on Mathematical Problem  Solving Tasks (https://arxiv.org/abs/2404.13099)
Comments: 10 pages, 3 figures, NeurIPS 2023 Workshop on Generative AI for Education (GAIED)

- **Newsletter**: [{"What's New": "자연어 처리(Natural Language Processing, NLP) 시스템과 대규모 언어 모델(Large Language Models, LLMs)의 발전은 교육 및 학습 방법론 분야에 새로운 기회를 제공하고 있습니다. 특히, 이 기술의 발전은 수학 문제 해결 영역에서 주목할만한 응용 분야로 자리잡고 있습니다. 본 논문에서는 'MathQuest'라는 새로운 수학 데이터셋을 소개하며, 이를 활용하여 세 가지 대표적인 LLMs (LLaMA-2, WizardMath, MAmmoTH)의 미세 조정 실험을 수행하였습니다."}, {'Technical Details': '이 연구에서 소개하는 MathQuest 데이터셋은 11, 12학년 수준의 NCERT(Mathematics NCERT textbooks) 수학 교과서에서 파생된 다양한 복잡성의 수학 문제를 포함하고 있습니다. 세 가지 모델은 각각 미세 조정(Fine-tuning)을 거쳐 데이터셋 상의 문제 해결 능력을 평가받았으며, 이 중 MAmmoTH-13B 모델이 가장 우수한 성능을 보여주었습니다. 이 결과는 MAmmoTH-13B 모델이 NCERT 수학 문제를 다루는데 있어서 강력하고 신뢰할 수 있는 벤치마크로 자리매김 할 수 있음을 시사합니다.'}, {'Performance Highlights': 'LLaMA-2, WizardMath, MAmmoTH 세 모델을 비교 분석한 결과, MAmmoTH-13B가 가장 높은 정확도로 수학 문제를 해결할 능력을 보여주었습니다. 이 연구는 특히 다단계 계산(Multi-step calculations) 및 복잡한 단어 문제(Complex word problems)를 효과적으로 처리할 수 있는 LLM의 능력을 평가하는 데 중점을 두었습니다. 결과적으로, MAmmoTH-13B는 다양한 수학 개념을 이해하고 적용하는 데 있어 우수한 성과를 나타냈습니다.'}]



### Demystifying Legalese: An Automated Approach for Summarizing and  Analyzing Overlaps in Privacy Policies and Terms of Servic (https://arxiv.org/abs/2404.13087)
- **What's New**: 이 연구는 복잡한 법적 용어가 포함된 이용 약관과 정책 문서를 이해하기 쉽게 요약하고 점수를 매기는 자동화된 도구를 개발하여, 사용자가 정보에 입각한 결정을 내릴 수 있도록 지원합니다. 특히 RoBERTa와 같은 최신 트랜스포머(Transformer) 기반 모델과 전통적인 기계학습 기법을 비교 분석하였으며, GDPR(General Data Protection Regulation 일반 데이터 보호 규정) 문서에서의 개념 중복과 지침 위반 가능성을 식별하여 GDPR 준수의 필요성을 강조합니다.

- **Technical Details**: 본 연구에서는 다중 클래스 텍스트 분류를 사용하여 문서의 주요 개념을 추출하고 이를 점수화하는 작업을 수행하였습니다. 이용된 모델로는 RoBERTa, PrivBERT, 선형 서포트 벡터 머신(SVM), 랜덤 포레스트가 있으며, RoBERTa가 0.74의 F1 점수로 가장 높은 성능을 보였습니다. 데이터 세트는 'Terms of Service; Didn’t Read' (ToS;DR)에서 스크랩된 것을 사용했고, 정책 문서 분석과 개념 분류의 두 가지 주요 작업을 정의하였습니다.

- **Performance Highlights**: RoBERTa 모델은 다른 모델들과 비교하여 가장 높은 F1 점수를 달성하였고, 텍스트의 문맥을 기반으로 문서 유형을 분류하고 주요 개념을 식별하는 능력이 뛰어났습니다. 이는 사용자가 서비스의 주요 조항을 빠르고 정확하게 이해할 수 있도록 하는데 기여할 수 있습니다. 또한, 정책 문서 간의 개념 중복을 측정하고 이를 통해 GDPR 지침 준수 여부를 평가하는데 중요한 기초 자료를 제공합니다.



### TREACLE: Thrifty Reasoning via Context-Aware LLM and Prompt Selection (https://arxiv.org/abs/2404.13082)
- **What's New**: TREACLE (비용 효율적인 추론을 위한 맥락 인식 언어 모델 및 프롬프트 선택, Thrifty Reasoning via Context-Aware LLM and Prompt Selection)은 대규모 언어 모델(LLM) 중에서 비용과 지연 시간 한계 내에서 가장 적합한 모델과 프롬프트를 선택하도록 설계된 새로운 강화 학습 정책입니다. 각 질문의 맥락을 고려하여 정확도와 비용 간의 최적의 균형을 달성하고자 하는 사용자들에게 유용한 도구를 제공합니다.

- **Technical Details**: TREACLE은 질문 텍스트의 임베딩과 이전 응답의 일관성을 고려하여 각 질문에 대해 가장 적합한 LLM과 프롬프트를 선택합니다. 이 시스템은 강화 학습을 사용하여 문제의 맥락, 질문의 난이도, 이전 응답의 일관성 등 다양한 요소를 기반으로 의사결정을 내릴 수 있습니다. 또한, Llama, GPT와 같은 다양한 LLM과 Chain-of-thought (CoT), 도메인 전문가 프롬프트 등 다양한 프롬프트 전략을 평가합니다.

- **Performance Highlights**: TREACLE은 기존의 기준점 대비 최대 85%까지 비용을 절감하면서도 높은 정확도를 유지하는 것으로 나타났습니다. 특히 GSM8K, CSQA, LLC 데이터 세트에서 다양한 LLM과 프롬프트를 사용한 테스트에서 저비용으로 높은 정확도를 달성할 수 있는 능력을 입증하였습니다.



### SuRe: Summarizing Retrievals using Answer Candidates for Open-domain QA  of LLMs (https://arxiv.org/abs/2404.13081)
Comments: Accepted at ICLR 2024

- **What's New**: 이 논문은 대규모 언어 모델 (Large Language Models, LLMs)을 활용하여 검색 기반 질문 응답 (Open Domain Question Answering, ODQA) 시스템의 정확도를 개선하는 새로운 프레임워크인 'Summarized Retrieval' (SuRe)를 제안합니다. 기존의 방법들이 모델의 추가적인 파인 튜닝(fine-tuning)을 요구했다면, SuRe는 특정 지시를 포함한 프롬프팅(prompting)을 통해 검색된 내용을 요약하여 질문에 대한 가장 타당한 답을 예측합니다.

- **Technical Details**: SuRe는 각 후보 답변에 따라 검색된 패시지를 요약하고, 이 요약을 통해 주어진 후보의 타당성 및 정보의 상대적 중요성을 평가하여 가장 가능성 있는 답변을 확인합니다. 이 과정은 모든 LLM 'zero-shot.prompting'에서 수행될 수 있으며, 범용적으로 다양한 검색 방법과 LLM에 적용 가능합니다. 특히, 검색된 패시지의 중요성을 평가하는 데 요약 결과를 추가로 활용할 수 있으며, 모델 및 인간에 의해 선호되는 근거로서의 역할도 수행합니다.

- **Performance Highlights**: 실험 결과, SuRe는 표준 프롬프팅 접근법에 비해 최대 4.6%의 정확도 (Exact Match, EM)와 4.0%의 F1 점수 개선을 보여주었습니다. 또한, ChatGPT와 BM25를 사용한 경우, 다양한 ODQA 데이터 세트에서 평균 4.6%/4.0%의 EM/F1 점수 개선을 보여준 것으로 나타났습니다. 이를 통해 SuRe가 더 정확하고 유용한 질문 응답 시스템을 만드는 데 기여할 수 있음을 입증합니다.



### Relational Graph Convolutional Networks for Sentiment Analysis (https://arxiv.org/abs/2404.13079)
- **What's New**: 이 논문에서는 텍스트 데이터에서 관계 정보를 이해하는 데 RGCNs(Relational Graph Convolutional Networks)의 능력을 활용하는 새로운 접근 방법을 제안합니다. 이를 통해 감정 분석 작업에 대해 더 세밀하고 정확한 분석을 가능하게 하고자 BERT와 RoBERTa와 같은 사전 학습된 언어 모델을 RGCN 구조와 결합하여 사용하였습니다.

- **Technical Details**: RGCNs는 그래프 내의 노드로 표현된 데이터 포인트 간의 의존성을 포착함으로써 해석 가능성과 유연성을 제공합니다. 기존의 GCN(Graph Convolutional Networks)과 달리 RGCNs는 그래프 내의 다양한 유형의 관계를 고려하여 보다 풍부한 관계적 정보를 분석할 수 있습니다. 이 연구에서는 Amazon 및 Digikala 데이터셋의 제품 리뷰를 사용하여 이 접근 방법을 실험하고, 결과를 평가하였습니다.

- **Performance Highlights**: 실험 결과, RGCN 기반 접근 방법은 기존 방법들에 비해 관계 정보를 더 잘 포착하고 감정 분석에서 높은 성능을 보였습니다. 특히, 다양한 감정 또는 정서를 표현하는 텍스트 문서의 분류에서 RGCNs의 강점이 두드러졌습니다.



### Empowering Interdisciplinary Research with BERT-Based Models: An  Approach Through SciBERT-CNN with Topic Modeling (https://arxiv.org/abs/2404.13078)
- **What's New**: 이 논문은 학술 문헌의 추상을 체계적으로 분류하기 위해 SciBERT 모델과 CNN을 결합하여 새로운 접근 방식을 소개합니다. 기존의 여러 레이블(Multi-label) 텍스트 분류 방법들이 의미 관계를 무시하고 클래스의 불균형을 해결하지 못하는 문제를 개선하였습니다.

- **Technical Details**: 이 연구는 Elsevier OA CC-BY 코퍼스를 사용하여 추상(abstract), 본문(body text), 제목(title), 그리고 BERT 토픽 모델링을 통해 얻은 키워드를 포함한 다중 세그먼트 입력 전략을 사용합니다. 과학적 문서의 복잡한 어휘와 구문을 해석하고 분류하는 데 특화된 SciBERT를 미세 조정(fine-tuned)하고, CNN을 통해 특징 추출을 강화하였습니다. 손실 함수와 클래스 가중치(class weights)를 사용하여 클래스 불균형 문제를 해결하였습니다.

- **Performance Highlights**: 모델은 표준 BERT 모델에 비해 불균형 데이터셋에서 F1 점수를 획기적으로 개선하였으며, 여러 학문 분야를 효과적으로 분류할 수 있어 문헌 검토의 효율성을 향상시켰습니다.



### Improving the Capabilities of Large Language Model Based Marketing  Analytics Copilots With Semantic Search And Fine-Tuning (https://arxiv.org/abs/2404.13077)
Comments: 16 pages, 5 figures, presented at the 2nd International Conference on NLP & AI (NLPAI 2024)

- **What's New**: 이 논문은 최근 개발된 대형 언어 모델(Large Language Models, LLMs)인 GPT-4를 사용하여 마케팅 인사이트를 제공하는 새로운 방법을 탐구합니다. 이 모델들이 마케팅 귀속(attribution)과 예산 최적화에 효율적으로 사용될 수 있음을 강조하며, 도메인 특화 질문 응답, SQL 생성 및 표 형식 분석을 중점적으로 다룹니다.

- **Technical Details**: 논문은 의미 검색(semantic search), 프롬프트 엔지니어링(prompt engineering), 그리고 파인튜닝(fine-tuning)의 조합을 통해 LLM의 정확성을 크게 향상시킬 수 있는 방법을 제시합니다. GPT-4와 오픈 소스 모델인 Llama-2-70b를 비롯한 다양한 모델과 임베딩 방법(embedding methods)을 비교 분석합니다.

- **Performance Highlights**: 이 논문은 마케팅 믹스 모델링(marketing mix modeling)과 귀속에 특화된 샘플 사용 사례에서 여러 모델의 성능을 테스트하여 LLMs가 정확하고 신속하게 마케팅 관련 결정을 내릴 수 있도록 지원하는 방법을 시연합니다.



### LLM Evaluators Recognize and Favor Their Own Generations (https://arxiv.org/abs/2404.13076)
- **What's New**: 이 연구는 자가 평가를 수행하는 대형 언어 모델(LLM: Large Language Models)이 자신의 출력물을 다른 LLM이나 인간이 생성한 텍스트보다 높게 평가하는 자기 선호(self-preference) 현상에 대해 집중적으로 분석했습니다. 연구팀은 이러한 자기 선호가 LLM이 자신의 출력물을 식별할 수 있는 자기 인식(self-recognition) 능력과 직접적인 연관이 있는지를 조사했습니다.

- **Technical Details**: 연구팀은 여러 LLM을 사용하여 자기 인식 능력을 측정하고, 이를 기반으로 자기 선호 강도와의 상관관계를 분석했습니다. GPT-4, GPT-3.5 Turbo, Llama 2와 같은 모델들이 사용되었으며, 이들은 fine-tuning(미세 조정)을 통해 자기 인식 능력을 개선할 수 있었습니다. 특히 GPT-4는 자신을 구별하는 정확도가 73.5%에 이르렀고, fine-tuning 후에는 90% 이상의 정확도를 보였습니다.

- **Performance Highlights**: 이 연구에서는 LLM의 자기 인식 능력과 자기 선호 강도 사이에 선형적인 상관관계를 발견했습니다. 즉, LLM이 자신의 출력물을 더 잘 식별할수록 자신의 출력물에 더 높은 점수를 부여하는 경향이 강해진다는 것을 확인했습니다. 이러한 발견은 LLM이 제공하는 평가의 편향을 이해하고 이를 완화하는 전략을 개발하는 데 중요한 정보를 제공합니다.



### Towards Compositionally Generalizable Semantic Parsing in Large Language  Models: A Survey (https://arxiv.org/abs/2404.13074)
- **What's New**: 이 논문은 구성적 일반화의 한계와 방법론을 탐구하면서 대규모 언어 모델(LLMs)이 의미 분석(Semantic Parsing) 작업에 어떻게 적용되는지에 대한 최근 연구를 종합적으로 검토합니다. 특히, 구성적 일반화의 능력 개선을 위한 새로운 접근 방식과 실험 설계가 소개되어 있습니다.

- **Technical Details**: 구성적 일반화(Compositional Generalization)는 모델이 기본 개념(Primitives)만을 보고 복잡한 새로운 조합을 이해하는 능력을 말합니다. 이 연구는 구성적 의미 분석과 대규모 언어 모델 두분야의 접점에서 데이터 구성 요소(Pre-training corpus, fine-tuning corpus)와 모델링 구성 요소(Modeling components: fine-tuning method, prompting strategies 등)를 검토합니다. 의미 분석을 위한 주요 벤치마크 데이터셋인 SCAN, COGS, 및 Freebase에 대해 자세히 설명하며, 이들 데이터셋이 어떻게 구성적 일반화를 정의하고 평가하는지를 설명합니다.

- **Performance Highlights**: 분석은 Transformer 기반 모델, 특히 T5와 BART가 순수 언어 기반 작업에서는 우수한 결과를 보이나, 구성적 일반화 측면에서는 여전히 한계가 있음을 보여줍니다. 대규모 언어 모델은 표준 벤치마크에서 상태-아트(State-of-the-art) 결과를 보이지만, 복잡한 중첩 구조를 많이 포함한 심볼릭 출력에서 일반화 실패 사례가 관찰되었습니다. 이를 극복하기 위한 새로운 데이터 증강 기술, 신경 기호 모델링(Neuro-symbolic modeling), 및 프롬프트 기반 방법론(Prompt-based methods)이 효과적일 수 있음을 제안하고 있습니다.



### Modeling Emotions and Ethics with Large Language Models (https://arxiv.org/abs/2404.13071)
Comments: 9 pages, 4 figures

- **What's New**: 이 논문은 큰 언어 모델(Large Language Models, LLMs)에 인간과 같은 정서와 윤리적 고려를 통합하는 것을 탐구합니다. 저자들은 기본적인 인간의 감정을 모델링하고, 그 감정을 다양한 강도로 재해석하고 표현하기 위해 협력적 LLMs를 사용합니다. 또한, 자기 감독 학습 알고리즘(self-supervised learning algorithm with human feedback, SSHF)을 이용하여 LLMs 안에 잠재적인 윤리적 차원을 내장시키고, 이를 통해 LLMs가 윤리적 지침에 대해 자가 평가하고 조정을 할 수 있도록 합니다.

- **Technical Details**: 논문에서는 8가지 기본적인 인간의 감정을 반대되는 쌍으로 제시하고, 이 감정들을 표현하기 위해 LLMs를 활용하는 방법을 소개합니다. 연구팀은 자기 감독 학습 알고리즘을 사용하여 모델이 윤리적 가이드라인을 자체적으로 평가하고 조정할 수 있도록 구현했습니다. 이를 통해 LLMs는 텍스트 및 이미지 생성을 넘어서 공감적 상호작용과 원칙에 기반한 의사결정까지 가능하게하여 AI 시스템 개발에 새로운 방향을 제시합니다.

- **Performance Highlights**: 이 연구의 방법론과 사례연구는 LLMs가 단순한 텍스트와 이미지 생성을 넘어서 사람들과의 공감적인 상호작용과 윤리적으로 의식 있는 결정을 내릴 수 있는 잠재력을 보여줍니다. LLMs의 자가 평가 및 조정 기능은 이들이 윤리적으로 조화롭고 감정적으로 반향하는 콘텐츠를 생성하는 능력을 향상시켰습니다.



### Evidence from counterfactual tasks supports emergent analogical  reasoning in large language models (https://arxiv.org/abs/2404.13070)
- **What's New**: 이 연구는 대규모 언어 모델(Language Models, LM)의 비유적 추론(analogical reasoning) 능력이 실제로 존재한다는 증거를 제시합니다. 이전 연구와 비교했을 때, 이번 연구는 새로운 '카운터팩추얼(counterfactual)' 작업 유형을 통해 언어 모델의 추론 능력이 훈련 데이터와의 유사성에만 기반한 것이 아니라는 점을 입증하려 시도하였습니다. 특히 GPT-3와 GPT-4 모델이 이와 같은 도전적인 문제에서도 일정 수준의 성능을 보이는 것이 관찰되었습니다.

- **Technical Details**: 연구자들은 모델이 코드 실행(code execution) 능력을 이용해 항목 리스트 내 정확한 인덱싱을 수행할 때 더 나은 성능을 보이는 것을 확인했습니다. 리뷰어들의 비판에도 불구하고, 연구진은 언어 모델 판단의 합리적 추론 능력이 유지된다는 것을 입증하기 위해 반론을 제기하였습니다. 또한 새로운 실험 데이터와 대조를 통해 다양한 유형의 문제 해결 및 패턴 인식에 언어 모델이 어떻게 사용될 수 있는지를 제시합니다.

- **Performance Highlights**: 코드 실행 능력이 강화된 GPT-4는 '카운터팩추얼' 문자열 유추 문제(letter-string analogy problem)를 인간 수준에서 풀어내는 성능을 보였습니다. 이는 단순히 기존 훈련 데이터를 반복하는 능력이 아니라, 새로운 문제 유형(새로운 순서 및 범위 크기가 적용된 알파벳)에 대한 일반화와 추론 능력을 보여주는 결과입니다. 이러한 연구 결과는 대규모 언어 모델이 인간과 유사한 추론 구조를 형성할 수 있는 가능성을 시사합니다.



### Subtle Signs of Scribal Intent in the Voynich Manuscrip (https://arxiv.org/abs/2404.13069)
Comments: Submitted to Histocrypt 2024

- **What's New**: 이 연구는 주목받지 못한 Voynich Manuscript의 'Voynichese' 스크립트 특성에 숨겨진 필사본 의도의 미묘한 신호를 찾는 데 초점을 맞추고 있습니다. 연구자들은 문서 내에서 '단어' 토큰의 통계적 분포와 구조적 기능(구분, 선의 시작과 끝) 및 복잡한 식물 그림과의 인접성이 스크립트의 흐름을 방해하는 방식을 분석하였습니다.

- **Technical Details**: 타이핑된 Voynichese 데이터는 'Zandbergen-Landini' transliteration을 사용하여 분석되었으며 이는 Voynichese 글리프를 나타내기 위해 설계된 여러 알파벳을 사용합니다. 연구진은 일러스트레이션에 의해 생성된 단어 토큰의 위치 변화를 통계적으로 분석하여 글을 쓰는 동안 미적 외관만을 목표로 하였는지, 아니면 실제로 의미 있는 내용을 전달하려 했는지를 이해하려 하였습니다.

- **Performance Highlights**: 연구 결과는 식물 그림이 있는 구간 바로 전후 및 라인의 시작과 끝에서 토큰의 분포에서 통계적으로 유의미한 차이를 발견하였습니다. 이는 특정 구조적 위치에서(specific structural positions) '단어' 토큰의 사용에 있어 의미 있는/의미 없는 내용의 전달 의도가 있었음을 시사합니다.



### Towards Efficient Resume Understanding: A Multi-Granularity Multi-Modal  Pre-Training Approach (https://arxiv.org/abs/2404.13067)
Comments: ICME 2024 Accepted

- **What's New**: 이 연구에서는 이력서의 구조화된 정보를 자동으로 추출하기 위해 새로운 모델인 ERU(Efficient Resume Understanding)를 제안합니다. 이전에는 문서 이해를 위한 단일 모달 전처리 방법에 대한 연구가 대부분이었으나, ERU는 텍스트, 시각, 레이아웃(layout) 정보를 통합하는 멀티-모달(multi-modal) 접근 방식을 채택하여, 이력서의 계층적 관계와 긴 문서의 특성을 더욱 효율적으로 처리합니다.

- **Technical Details**: ERU 모델은 먼저 OCR 도구를 사용하여 각 이력서에서 텍스트, 시각적, 레이아웃 정보를 추출하고 이를 멀티-모달 융합 변환기(transformer)에 통합합니다. 그 후, 세 가지 자체 감독(self-supervised) 태스크를 통해 대량의 비레이블 이력서에서 모델을 사전 훈련(pre-train)합니다. 이 사전 훈련 과정은 가려진 언어 모델(masked language model), 시각적 위치 정렬(visual position alignment), 가려진 부분 예측(masked segment prediction)을 포함합니다. 마지막으로, 멀티-그래뉼러리티(multi-granularity) 시퀀스 라벨링 작업을 통해 모델을 미세 조정(fine-tune)하여 이력서에서 구조화된 정보를 추출합니다.

- **Performance Highlights**: 실제 데이터셋에 대한 광범위한 실험을 통해 ERU 모델의 효과성이 입증되었습니다. 이 모델은 전통적인 규칙 기반 접근법과 다른 신경망 기반 기술들을 능가하는 성능을 보여주었으며, 특히 멀티-모달 정보를 통합한 접근 방식이 이력서의 복잡한 구조와 다양한 형식을 효율적으로 해석하는 데 큰 도움이 되었습니다.



### Leveraging Large Language Model as Simulated Patients for Clinical  Education (https://arxiv.org/abs/2404.13066)
- **What's New**: 이 연구에서는 임상 의료 교육용 가상 환자(VSP; Virtual Simulated Patient)를 구현하기 위해 대규모 언어 모델(LLM; Large Language Models)을 활용하는 새로운 통합 모델 불가지론적 프레임워크인 'CureFun'을 제안합니다. 이는 학생들이 가상 환자와 자연스러운 대화를 나누고, 그 대화를 평가하며, 임상 질문 기술을 향상시킬 수 있도록 제안함으로써, 기존 LLM 기반 챗봇들과 비교했을 때 더욱 진정성 있고 전문적인 대화 흐름을 가능하게 합니다.

- **Technical Details**: CureFun 프레임워크는 LLM의 포괄적인 능력을 활용하여 SP 템플릿 준비, 학생-환자 대화 구현 및 SP 대화 이력 평가를 수행합니다. 주요 기술로는 프롬프트(Prompt) 및 사고의 사슬(Chain-of-Thought)을 사용하여 LLM의 동작을 제어하고, 검색 강화 생성(Retrieval-Augmented Generation, RAG) 형태로 대화 흐름을 향상시키고 제어합니다. 또한, 여러 LLM을 통한 투표로 감독받는 다중 입자성 평가 항목을 생성하여 학생들에게 적절한 점수를 부여하고 제안을 제공합니다.

- **Performance Highlights**: CureFun은 8개 중국어 SP 케이스를 사용하여 평가되었으며, 각 케이스는 다양한 질병과 의료 전문 분야를 포함합니다. 평가 실험에서는 GPT-3.5-turbo, PaLM 등 다수의 LLM을 포함한 6개의 성숙한 LLM 챗봇과 대화를 진행하였습니다. 결과적으로 CureFun은 전통적 SP 평가 체크리스트를 LLM 실행 가능 프로그램으로 자동 변환하는 능력을 향상시켰으며, 다수의 LLM을 결합하여 학생들의 의료 대화를 포괄적이고 신뢰성 있게 평가할 수 있습니다.



### Intellecta Cognitiva: A Comprehensive Dataset for Advancing Academic  Knowledge and Machine Reasoning (https://arxiv.org/abs/2404.13065)
- **What's New**: Intellecta 데이터셋은 현대 언어 모델의 인지 처리 능력을 향상시키기 위해 고안된 혁신적인 합성(synthetic) 데이터셋으로 등장했습니다. 이 데이터셋은 11.53억 토큰으로 구성되어 있으며, 이 중 8.01억 토큰은 합성 데이터고, 3.52억 토큰은 풍부한 교과서 데이터로 구성되어 있습니다. 교육적 서술(narrative)과 고급 추론을 촉진하도록 설계되었으며, Mixtral-8x7B-Instruct-v0.1 모델을 활용하여 복잡한 사고 과정과 교과서 스타일의 상세한 설명을 생성할 수 있습니다.

- **Technical Details**: Intellecta 데이터셋의 설계 목표는 인간이 교과서를 바탕으로 주제를 학습하는 방식을 따라 모델의 일반화를 촉진하고 모델 과적합을 방지하기 위한 데이터의 다양성을 고도화하는 것입니다. 이를 위해 특정 프롬프트 엔지니어링(prompt engineering)이 사용되어 코딩 문제에서 문학 분석에 이르기까지 다양한 데이터 포인트를 배양합니다. 데이터는 초급부터 고급까지 다양한 복잡성 수준을 포함하며, 오픈 소스 원칙에 따라 투명하고 재현 가능한 프로세스를 기반으로 구축되어 윤리적 데이터 큐레이션을 중시하며 편향을 최소화합니다.

- **Performance Highlights**: 이 데이터셋은 고급 사고 및 학문적 능력을 구현할 수 있도록 언어 모델을 훈련하는 데 있어서 다양하고 풍부한 주제들을 포함하여, AI 벤치마크 평가의 유효성을 보장하고, 모델의 효율을 강화하는 데 도움을 주는 데이터 중복 제거 및 윤리적 스크리닝 과정을 포함합니다. 또한 데이터셋은 DBSCAN 클러스터링 메소드를 사용하여 의미론적 유사성에 따라 데이터를 조직하여 언어 모델이 다양한 주제에 걸쳐 미묘한 이해와 생성을 가능하게 합니다.



### "Hey..! This medicine made me sick": Sentiment Analysis of  User-Generated Drug Reviews using Machine Learning Techniques (https://arxiv.org/abs/2404.13057)
- **What's New**: 본 논문에서는 의약품 사용자 리뷰에 대한 감성 분류 시스템을 제안하였습니다. 이 시스템은 BERT, SciBERT, BioBERT와 같은 사전 학습된 언어 모델을 사용하여 특징(Feature)을 추출하고, 이를 다양한 기계 학습(Classification) 분류기, 예를 들어 의사 결정 트리(Decision Trees), SVM(Support Vector Machines), 랜덤 포레스트(Random Forests), 그리고 순환 신경망(Recurrent Neural Networks)에 활용하였습니다.

- **Technical Details**: 이 연구에서는 각각의 감성, 즉 긍정적, 부정적, 중립적인 클래스로 의약품 리뷰를 분류하는 시스템을 개발하였습니다. 사용된 데이터는 공개적으로 이용 가능한 소스에서 수집되었으며, 수동으로 레이블링 및 검증되었습니다. 사전 학습된 언어 모델들은 특징을 잘 포착할 수 있도록 도와주며, 이 특징들은 머신 러닝 분류기들에게 입력되어 감정 분석을 수행합니다.

- **Performance Highlights**: 시스템의 성능은 정밀도(Precision), 재현율(Recall), F1-점수(F1-score)를 통해 측정되었습니다. 결과적으로 제안된 방법론은 의약품에 대한 사람들의 감정을 분석하는 데 유용함을 보여주었으며, 특히 BERT, SciBERT, BioBERT 모델들이 특징 추출에 탁월한 성능을 보였습니다.



### FlowMind: Automatic Workflow Generation with LLMs (https://arxiv.org/abs/2404.13050)
Comments: Published in ACM ICAIF 2023

- **What's New**: 이 논문은 사용자의 예측할 수 없는 요구에 대응하기 위해 로보틱 프로세스 자동화(Robotic Process Automation, RPA)의 한계를 극복하는 새로운 접근법인 FlowMind를 소개합니다. FlowMind는 대규모 언어 모델(Large Language Models, LLMs), 특히 생성적 사전 훈련 변환기(Generative Pretrained Transformer, GPT)를 활용하여 자동 워크플로우 생성 시스템을 구현합니다. 이 시스템은 API(Application Programming Interfaces)를 기반으로 하여 LLM의 '환각' 문제를 완화하고, 민감한 데이터나 코드와의 직접적인 상호작용 없이 정보의 무결성과 기밀성을 보장합니다. 또한, 사용자는 자동 생성된 워크플로우의 고급 설명을 통해 검토 및 피드백을 제공할 수 있습니다.

- **Technical Details**: FlowMind는 LLM을 활용한 프롬프트 디자인의 일반적인 렉처 레시피(generic lecture recipe)를 제안합니다. 이 구조는 LLM에게 제공되는 API와의 상호작용을 통해 워크플로우를 생성하고, 사용자의 피드백을 반영하여 조정할 수 있습니다. 이 시스템은 사용자와의 효율적인 상호작용을 가능하게 하며, 비전문가 사용자도 손쉽게 워크플로우를 검토하고 피드백을 제공할 수 있게 합니다. 또한, NCEN-QA라는 새로운 금융 분야 데이터셋을 도입하여 N-CEN 보고서로부터의 질의응답(Question-Answering, QA) 작업에 대한 워크플로우 생성 시스템의 성능을 평가합니다.

- **Performance Highlights**: FlowMind는 기존의 기준 모델과 FlowMind의 변형 모델을 사용한 실험을 통해 그 효과를 입증하였습니다. 이 연구에서 FlowMind는 NCEN-QA 데이터셋을 사용하여 금융 분야에서의 질의응답 작업을 처리하는 데 있어 뛰어난 성능을 보였습니다. FlowMind의 각 구성 요소의 중요성을 분석하고, 사용자 상호작용 및 피드백이 시스템의 정확성과 적응성을 어떻게 향상시키는지를 보여줍니다.



### A Multimodal Automated Interpretability Agen (https://arxiv.org/abs/2404.14394)
Comments: 25 pages, 13 figures

- **What's New**: 이 논문은 다기능 자동 해석 에이전트(MAIA)를 소개합니다. MAIA는 신경 모델을 사용하여 특징 해석 및 실패 모드 발견과 같은 모델 이해 작업을 자동화하는 시스템입니다. MAIA는 예비 훈련된 시각-언어 모델에 다양한 실험 도구를 장착하여 다른 모델의 하위 구성 요소에 대한 반복적인 실험을 지원하여 그 동작을 설명합니다.

- **Technical Details**: MAIA는 미리 훈련된 시각-언어 모델(GPT-4V 'Vision-Language Model'(VLM))을 기반으로 구축되었으며, 이는 이미지를 직접 처리할 수 있는 기능을 갖추고 있습니다. MAIA는 주어진 설명 작업(예: '레이어 4의 유닛 487의 행동을 설명하라')에 대해 대응하는 'interpretability experiment'를 설계하고, 실험 모듈을 조합하여 쿼리에 답합니다. MAIA의 API는 체계적으로 실험을 조합하여 Python 프로그램으로 변환을 지원하며, 이를 통해 네트워크의 다양한 해석 작업을 수행할 수 있습니다.

- **Performance Highlights**: MAIA는 다양한 훈련된 모델 및 새로운 합성 시각 뉴런 데이터셋을 통해 이미지의 학습 표현에서 특징을 설명할 수 있는 능력을 평가받았습니다. 실제 평가에서 MAIA가 생성한 설명은 전문가 수준의 인간 실험자가 생성한 설명과 비교해 볼 때 매우 비슷한 수준의 예측을 제공하였습니다. 또한, MAIA는 임의의 시스템 행동을 설명하고 설명하는 데 필요한 실험을 자동으로 설계할 수 있는 유연성을 보여주었습니다.



### Graphic Design with Large Multimodal Mod (https://arxiv.org/abs/2404.14368)
- **What's New**: 이 논문에서는 기존의 Graphic Layout Generation (GLG)의 한계를 개선하여 보다 유연하고 실용적인 Hierarchical Layout Generation (HLG)을 소개합니다. HLG는 주어진 디자인 요소들을 순서없이 다루면서도 시각적으로 매력적인 그래픽 구성을 생성할 수 있습니다. 이를 위해 대규모 멀티모달 모델(Large Multimodal Model, LMM)을 기반으로 한 첫 레이아웃 생성 모델인 Graphist를 개발했습니다.

- **Technical Details**: Graphist는 RGB-A 이미지 입력을 받아 각 요소의 좌표, 크기 및 순서를 지정하는 JSON 초안 프로토콜을 출력하는 엔드 투 엔드(end-to-end) 방식으로 구성됩니다. 또한, HLG 작업을 평가하기 위해 Inverse Order Pair Ratio (IOPR) 및 GPT-4V Eval이라는 새로운 평가 메트릭을 도입하였습니다.

- **Performance Highlights**: Graphist는 기존 기술들을 능가하며, HLG 뿐만 아니라 전통적인 GLG 작업에서도 인상적인 성능을 보였습니다. 새로운 평가 메트릭에서도 우수한 성능을 보이며 강력한 벤치마크를 설정하였습니다.



### Detecting and Mitigating Hallucination in Large Vision Language Models  via Fine-Grained AI Feedback (https://arxiv.org/abs/2404.14233)
- **What's New**: 이 연구에서는 대규모 시각 언어 모델(LVLM)에서 발생하는 환각 현상을 세밀하게 감지하고 완화하는 방법을 제안합니다. 이러한 환각 현상은 생성된 텍스트가 주어진 맥락과 일치하지 않는 문제로 LVLM의 활용을 크게 제한합니다. 기존의 연구와 달리, 본 연구는 프로프라이어터리(proprietary) 모델을 통해 생성된 작은 규모의 문장 수준 환각 어노테이션 데이터셋을 활용하여 환각을 감지하고, 감지 후 다시 쓰기(detect-then-rewrite) 파이프라인을 통해 선호 데이터셋을 구축합니다. 또한, 환각의 심각성을 구분하고, 환각 심각성 인식 직접 선호 최적화(Hallucination Severity-Aware Direct Preference Optimization, HSA-DPO)를 도입하여 환각을 완화합니다.

- **Technical Details**: 연구진은 먼저 소규모 문장 수준 환각 어노테이션 데이터셋을 생성하고, 이를 사용하여 환각 감지 모델을 훈련시키며, 주요 환각 유형(객체(Object), 속성(Attribute), 관계(Relationship))을 커버합니다. 이후 감지된 환각을 기반으로 비환각 반응을 생성하여 선호 학습 데이터셋을 구축하는 '감지 후 다시 쓰기' 파이프라인을 제안합니다. HSA-DPO는 환각의 심각성을 선호 학습에 통합하여 중요한 환각을 우선적으로 완화합니다.

- **Performance Highlights**: 실험 결과, 이 방법은 다양한 환각 감지 및 완화 벤치마크에서 효과적임을 입증했습니다. 환각 감지 모델은 MHaluBench에서 GPT-4V 및 Gemini를 상회하는 새로운 최고 성능을 달성했으며, HSA-DPO는 경쟁적인 폐쇄 소스 LVLM보다 AMBER에서 환각률을 36.1% 줄이고, Object HalBench에서는 76.3% 줄였습니다. 이러한 우수한 성능은 HSA-DPO 도입과 세밀한 AI 피드백의 효과를 입증합니다.



### Surveying Attitudinal Alignment Between Large Language Models Vs. Humans  Towards 17 Sustainable Development Goals (https://arxiv.org/abs/2404.13885)
- **What's New**: 이 연구는 대규모 언어 모델(Large Language Models, LLM)이 유엔의 지속 가능한 발전 목표(Sustainable Development Goals, SDGs)에 대한 태도와 이들 목표를 지지하는 정도에서 인간과의 차이를 분석합니다. 특히, 이해와 감정, 문화적 및 지역적 차이, 작업 목표의 변화 및 의사 결정 과정에서 고려되는 요소들을 중심으로 잠재적인 차이점을 조사하였습니다.

- **Technical Details**: LLMs는 텍스트 데이터로부터 학습하여 인간의 언어를 모방하는데 사용되며, GPT-3, GPT-4와 같은 모델은 특히 고급 자연어 처리(Natural Language Processing, NLP) 작업에서 강력한 성능을 보입니다. 이 연구에서는 이러한 모델들이 어떻게 SDGs에 대해 학습하고 이를 어떻게 반영하는지, 그리고 이 과정에서 발생할 수 있는 문제점들을 탐구합니다.

- **Performance Highlights**: LLMs는 특히 데이터가 편향되거나 컨텍스트를 완전히 이해하지 못하는 경우, 사회적 불평등, 인종 차별, 환경 파괴 및 자원 낭비를 심화시킬 수 있는 위험을 수반할 수 있습니다. 이에 대응하기 위해, 연구진은 LLM이 SDGs의 원칙과 목표에 부합하도록 조정하고 규제하는 전략과 권장 사항을 제안합니다.



### EventLens: Leveraging Event-Aware Pretraining and Cross-modal Linking  Enhances Visual Commonsense Reasoning (https://arxiv.org/abs/2404.13847)
- **What's New**: 이 논문에서는 시각적 상식 추론(Visual Commonsense Reasoning, VCR) 태스크를 처리하기 위해 대규모 언어 모델(Large Language Models, LLMs)을 활용하는 새로운 접근법인 EventLens를 제안합니다. EventLens는 사건-인식 사전 훈련(Event-Aware Pretraining)과 크로스-모달 연결(Cross-modal Linking)을 통해 VCR의 성능을 향상시킵니다.

- **Technical Details**: EventLens는 세 가지 주요 요소로 구성됩니다. 첫째, 사건-인식 사전 훈련을 통하여 복잡한 시각적 장면을 이해하고, 이미지 내 객체와 진행 중인 사건 및 등장 인물의 의도를 추론하는 능력을 강화합니다. 둘째, 크로스-모달 로컬 연결 방법을 도입하여 텍스트 토큰과 로컬 객체 특징(local object features) 및 전체 이미지 장면 사이의 상관관계를 강화합니다. 마지막으로, 인스트럭트 스타일의 프롬프트(Instruct-style prompts)와 태스크-특화 어댑터(Task-specific adapters)를 사용하여 사전 훈련과 VCR 태스크 사이의 갭을 좁히고, LLM의 내재된 지식을 새로운 데이터와 더욱 효과적으로 통합합니다.

- **Performance Highlights**: EventLens는 VCR 데이터셋에서 우수한 성능을 보여주었으며, 상대적으로 적은 수의 학습 가능한 매개변수를 사용하여 계산 자원 비용을 절감하였습니다. 성능 개선을 입증하는 아블레이션(ablation) 실험을 통해 EventLens의 효과가 입증되었습니다.



### Filtered Direct Preference Optimization (https://arxiv.org/abs/2404.13846)
- **What's New**: 이 연구에서는 인간의 피드백으로부터 강화학습(RLHF)을 사용하는 언어 모델의 정렬에 있어 텍스트 데이터셋의 품질이 중요한 영향을 미친다는 점을 다루고 있습니다. 특히, 보상 모델 없이 직접적인 선호 최적화(DPO) 방법에 초점을 맞추며, 데이터셋의 품질이 모델 성능에 미치는 영향을 실험적으로 확인하였습니다. 이를 기반으로 보상 모델을 이용해 낮은 품질의 데이터를 걸러내는 새로운 접근법인 필터링된 직접 선호 최적화(fDPO)를 제안합니다.

- **Technical Details**: DPO는 보상 모델(reward model, RM)을 기반으로 하지 않고 선호 데이터만을 사용하여 언어 모델(LM)을 최적화하는 RM-free RLHF 방법입니다. 연구진은 fDPO에서 훈련된 RM을 사용하여 선호 데이터셋 중 품질이 낮은 텍스트를 식별하고 제거합니다. 이 과정은 DPO 동안 데이터셋의 정확성을 높이고 최종 모델 성능을 향상시킵니다.

- **Performance Highlights**: 실험 결과, fDPO는 기존 DPO 대비 향상된 성능을 보여주며, RM을 활용한 접근방식의 잠재적 데이터 효율성의 이점을 실현할 수 있음을 시사합니다. 이 방법은 160M 및 1.4B 크기의 언어 모델에서 테스트되었으며, 지시사항을 따르는 과제에서의 성능이 현저히 향상되었습니다.



### Counterfactual Reasoning Using Predicted Latent Personality Dimensions  for Optimizing Persuasion Outcom (https://arxiv.org/abs/2404.13792)
Comments: 14 pages, 10 figures, Accepted by Persuasive Technology 2024

- **What's New**: 이 논문에서는 특정 사용자의 행동에 대해 효과적으로 설득하기 위하여 사용자의 잠재적 인격 차원(Latent Personality Dimensions, LPDs)을 추적하고 이를 바탕으로 맞춤형 반대 시나리오(counterfactual utterances)를 생성하는 새로운 접근 방식을 제시합니다. 이 방식은 Bi-directional Generative Adversarial Network (BiCoGAN)과 Dialogue-based Personality Prediction Regression (DPPR) 모델을 사용하여 대화 중에 사용자의 상태를 동적으로 조정하고 최적화된 설득 결과를 달성합니다.

- **Technical Details**: 제안된 시스템은 BiCoGAN과 DPPR 모델을 통합하여 반대 시나리오 데이터를 생성하고, 이를 D3QN 모델을 사용하여 시스템 발화의 최적화된 선택을 학습합니다. 이 구조는 사용자의 개별 LPDs를 추정하고, 생성된 반대 시나리오 데이터를 통해 대화의 원활함과 유연성을 제공하며, IMDPs(Individualized Markov Decision Processes)를 활용하여 정책(policy) 학습을 수행합니다.

- **Performance Highlights**: PersuasionForGood 데이터셋을 사용한 실험 결과, 기존 방법론인 단순 BiCoGAN과 비교할 때 본 연구의 접근 방식이 더 높은 누적 보상(cumulative rewards)과 Q-values를 달성했음을 보여줍니다. 이는 반대 시나리오 추론과 LPDs를 활용한 강화 학습 정책 최적화가 온라인 상호작용에서의 설득 효과를 증진시킬 수 있음을 시사합니다.



### Iteratively Prompting Multimodal LLMs to Reproduce Natural and  AI-Generated Images (https://arxiv.org/abs/2404.13784)
- **What's New**: 이 연구는 디지털 이미지 시장에서 AI 생성 이미지 마켓플레이스와 프리미엄 스톡 이미지 제공자 사이의 비용 효율적인 대안을 모색합니다. DALL-E 3, Midjourney와 같은 최신 text-to-image API를 활용하여, 비슷한 이미지를 더 저렴한 비용으로 만들 수 있는 새로운 공격 전략을 제시하였습니다. 이는 디지털 미디어의 무결성과 경제적 고려사항에 대한 새로운 논의를 촉발할 수 있습니다.

- **Technical Details**: 이 연구의 기술적인 내용은 세 부분으로 구성된 공격 전략을 중심으로 합니다. 첫째, large dataset에서 fine-tuned된 CLIP 모델을 사용하여, 관련 키워드와 명명된 개체를 식별하는 multi-label classifier를 통합하였습니다. 둘째, GPT-4V를 사용하여 CLIP 모델과 classifier에서 얻은 정보를 바탕으로 자세한 프롬프트를 생성합니다. 마지막으로, 이 프롬프트를 반복적으로 다듬어 원본 이미지와 유사한 이미지를 생성하였습니다. 추가적으로, Midjourney 플랫폼에서 생성된 약 1천9백만 개의 프롬프트-이미지 쌍으로 구성된 대규모 데이터셋을 수집하여 공격 전략을 개선하고 검증하였습니다.

- **Performance Highlights**: 이 공격 방법은 기존 기준을 능가하는 성능을 보였으며, 자동 메트릭(auto metrics)과 인간 평가(human evaluation)를 통해 검증되었습니다. 특히, 공격자는 대상 이미지와 상당히 유사한 이미지를 개당 단지 $0.23에서 $0.27의 비용으로 생성할 수 있었습니다. 이는 시장에서 통용되는 $3에서 $500 사이의 가격과 비교할 때 현저하게 낮은 비용입니다. 이 결과는 디지털 시장에서의 이미지 생성과 거래 방식에 중대한 영향을 미칠 수 있음을 시사합니다.



### Towards General Conceptual Model Editing via Adversarial Representation  Engineering (https://arxiv.org/abs/2404.13752)
- **What's New**: 이 논문은 대규모 언어 모델(Large Language Models, LLMs)의 복잡한 내부 작동을 이해하기 위한 접근법으로 'Representation Engineering (RepE)'을 소개합니다. 특히, 'Adversarial Representation Engineering (ARE)'이라는 새로운 방법을 도입하여 LLMs의 수정 가능성을 향상시키고, 복잡한 내부 튜닝 없이도 모델을 효율적으로 편집할 수 있는 방법을 제공합니다.

- **Technical Details**: 'Adversarial Representation Engineering (ARE)'은 RepE를 활용해 LLMs를 편집하는 기술입니다. 이 기법은 LLM과 판별 모델을 동시에 훈련시켜, 대조적인 특징 임베딩(Contrastive Feature Embeddings)을 추출하는 두 가지 주요 단계를 거칩니다. ARE는 전체적인 특징 표상(Feature Representations)에 초점을 맞춰, 내부 히든 레이어(Internal Hidden Layers)를 직접 조작하여 LLM의 동작을 수정하고 제어할 수 있습니다.

- **Performance Highlights**: ARE 방법은 다양한 편집 및 검열 작업에서 효과적임을 실험을 통해 입증하였습니다. 예를 들어, 모델을 더 안전하게 조정하거나, 정직한 대답을 유도하는 등의 특정 목표를 달성하면서도 모델의 기본 성능을 저하시키지 않았습니다. 또한 ARE는 TruthfulQA에서 상당히 향상된 정확도를 달성하며, 이는 ARE의 실용성을 강력히 입증하는 결과입니다.



### The Framework of a Design Process Languag (https://arxiv.org/abs/2404.13721)
Comments: PhD dissertation, 1993, Norwegian Institute of Technology

- **What's New**: 이 논문은 디자인 개념 형성 프레임워크에서 디자인을 바라보는 새로운 관점을 개발하고, 디자인 대상과 디자인 과정을 서술하는 언어를 제안합니다. 디자인 과정에서 아직 알려지지 않은 대상을 알려진 개념들에 연관시켜 규정하는 과정을 통해, 디자이너는 이 대상을 정의해 나갑니다.

- **Technical Details**: 디자인 프로세스 언어(Design Process Language, DPL)는 개념을 조합할 때 사용하는 관계의 클래스인 언어적 카테고리를 기반으로 합니다. 이 언어는 과정과 대상을 동일한 일반 체계 내에서 설명하는 데 사용되는 관계를 담고 있으며, 일부는 과정 특유의 관계, 다른 일부는 대상 특유의 관계를 포함하는데 대부분은 과정과 대상 설명 모두에 사용됩니다. 또한 미래, 가능성, 의지, 가상 사건 등을 설명하는 모달(modal) 관계의 구분도 이루어집니다.

- **Performance Highlights**: DPL은 디자인 과정을 묘사하는 데 필요한 관계, 특히 미래나 가능성 같은 요소들을 지원할 수 있는 언어를 구축하는 기반으로서 역할을 합니다. 이는 컴퓨터가 디자인 과정에 더 유용하게 - 더 지능적으로 - 작용하도록 하는 데 사용될 수 있는 언어의 개발을 가능하게 합니다.



### Incorporating Different Verbal Cues to Improve Text-Based  Computer-Delivered Health Messaging (https://arxiv.org/abs/2404.13633)
Comments: PhD thesis - National University of Singapore, November 2023

- **What's New**: 스마트폰의 보급으로 인해 디지털 헬스케어 분야가 확대되었으며, 특히 건강 상태에 대해 사람과 컴퓨터 간의 직접적이고, 실시간적인 의사소통을 도모하기 위해 챗봇(health chatbot)을 활용한 사례가 증가하고 있습니다. 이 연구는 헬스케어 챗봇의 효과적인 대화 스타일(conversational style) 및 메시지 형식(formatting messages)을 찾기 위한 실험을 다루고 있습니다.

- **Technical Details**: 이 논문은 디지털 헬스 메시징의 효과를 높이기 위해 'Computers are Social Actors (CASA)' 이론을 적용하여, 사람이 컴퓨터와 상호작용할 때 인간 대 인간의 상호작용을 모사하도록 설계된 대화 전략과 메시지 형식을 연구합니다. 이 연구는 특히 대화의 내용(conversational content)과 스타일(conversational style), 그리고 사용자의 이전 발화를 참조하는 챗봇의 반응 형식을 중심으로 진행되었습니다.

- **Performance Highlights**: 초기 연구에서는 '대중(crowd)'을 이용하여 다양한 건강 메시지를 생성하고 이를 다양한 대화 스타일로 실험하여, 사용자들의 반응을 분석했습니다. 그 결과, 적절한 대화 스타일과 개인 맞춤형 응답 형식을 사용할 때 사용자의 긍정적인 반응이 증가하는 것으로 나타났습니다.



### Utilizing Deep Learning to Optimize Software Development Processes (https://arxiv.org/abs/2404.13630)
- **What's New**: 이 연구는 소프트웨어 개발 과정에서 딥러닝 기술(deep learning technologies) 응용을 탐구하여, 특히 코드 리뷰 자동화, 오류 예측, 테스트 생성을 통해 코드 품질과 개발 효율을 향상하는데 초점을 맞추고 있습니다. 이는 기존의 전통적 방법들과 비교하여 실질적인 개선을 보여주는 중요한 연구입니다.

- **Technical Details**: 실험 그룹과 대조 그룹(control groups)을 설치하여 딥러닝 도구와 전통적 방법을 각각 적용한 후, 코드 오류율(code error rates)과 프로젝트 완성 시간(project completion times)을 비교하였습니다. 연구는 또한 소프트웨어 개발에서 딥러닝 기술의 최적화 포인트, 방법론, 기술적 도전 과제를 논의합니다.

- **Performance Highlights**: 실험 그룹에서는 코드 오류율이 유의미하게 감소하였으며 프로젝트 완성 시간 또한 대폭 개선되었습니다. 이는 딥러닝 기술이 소프트웨어 개발 과정에 효과적으로 통합될 수 있음을 시사합니다.



### Video sentence grounding with temporally global textual knowledg (https://arxiv.org/abs/2404.13611)
- **What's New**: 이 논문에서는 시각적(visual) 및 언어적(textual) 영역 사이의 도메인 간격을 교량하는 새로운 접근법을 제안합니다. 특히, 영상-질의 쌍에서 파생된 시각적 특징과 의사 질의 특징 사이에 높은 유사성을 달성하기 위해 Pseudo-query Intermediary Network (PIN)를 사용하여 시각적 및 포괄적인 의사 질의 특징 간의 정렬을 개선합니다.

- **Technical Details**: PIN는 대조적 학습(contrastive learning)을 통해 시각적 인코더와 의사 질의 중개 네트워크를 훈련하며, PQ-prompt는 의사 질의의 글로벌 텍스트 지식을 내재화하여 텍스트 인코더의 출력을 정제합니다. 이 프롬트는 멀티 모달 융합 모듈의 학습을 강화하고, 추론 단계에서는 의사 질의를 사용하지 않고 지상 진실 시간적 질의로 대체하여 모델을 안내합니다.

- **Performance Highlights**: 제안하는 방식은 Charades-STA 및 ActivityNet-Captions 데이터셋에서 상당한 향상을 보여 주며 대부분의 평가 지표에서 최첨단(state-of-the-art) 결과를 달성하였습니다.



### Exploring Diverse Methods in Visual Question Answering (https://arxiv.org/abs/2404.13565)
- **What's New**: 이 연구는 GAN(Generative Adversarial Networks), 오토인코더(autoencoders) 및 주의 메커니즘(attention mechanisms)을 사용하여 시각적 질문 응답(VQA)의 개선을 위한 혁신적인 방법을 탐구합니다. VQA 데이터셋을 활용하여 세 가지 전략을 조사했습니다.

- **Technical Details**: 첫 번째 GAN 기반 접근법은 이미지와 질문 입력에 조건을 달아 답변 임베딩을 생성하는 것을 목표로 하며, 복잡한 작업에서 어려움을 겪었습니다. 두 번째로, 오토인코더 기반 기술은 질문과 이미지에 대한 최적의 임베딩 학습에 중점을 두고, 복잡한 질문에 대한 더 나은 능력으로 인해 GAN과 비슷한 결과를 달성했습니다. 마지막으로, MCB(Multimodal Compact Bilinear pooling)을 통합한 주의 메커니즘은 언어 선험적 지식(language priors)과 주의 모델링(attention modeling)을 다루지만 복잡성-성능 간의 트레이드오프가 있습니다.

- **Performance Highlights**: 오토인코더 접근 방식은 복잡한 질문을 처리할 때 GAN에 비해 우수한 성능을 보였으며, 주의 메커니즘을 사용한 MCB는 언어 이해와 이미지 분석을 통합할 수 있는 능력을 개선했지만, 시스템의 전반적인 복잡성이 증가했습니다.



### ChatRetriever: Adapting Large Language Models for Generalized and Robust  Conversational Dense Retrieva (https://arxiv.org/abs/2404.13556)
- **What's New**: 이 논문에서는 대화형 검색(Discussion Search)의 사용자 의도를 정확하게 해석하는 새로운 시스템인 ChatRetriever를 소개합니다. ChatRetriever는 복잡한 대화 세션을 강력하게 표현할 수 있는 LLM(Large Language Models)의 일반화 능력을 활용하여, 향상된 대화형 검색 성능을 제공합니다. 또한, 대조 학습(Contrastive Learning)과 마스크된 지시어 조율(Masked Instruction Tuning)을 결합한 새로운 이중 학습 접근법을 제안합니다.

- **Technical Details**: ChatRetriever는 대화의 복잡함을 이해하고 처리하는 데 초점을 맞춘 대조적 세션-마스킹 지시어 조율(Contrastive Session-Masked Instruction Tuning, CSIT) 방법을 사용합니다. 이를 통해 LLM을 대화형 밀도 검색(Dense Retrieval)에 적합하게 조정할 수 있으며, 세션에서 중요 단어만을 마스킹하여 응답 토큰의 언어 모델링 손실을 계산함으로써 복잡한 세션의 표현 학습을 강화합니다.

- **Performance Highlights**: ChatRetriever는 다양한 대화형 검색 벤치마크에서 기존의 대화형 밀도 검색기보다 월등히 높은 성능을 보여주며, LLM 기반의 대화형 쿼리 재작성 방식과 비슷한 성능을 달성합니다. 특히 CAsT-20과 CAsT-21 벤치마크에서는 각각 6.8% 및 12.2%의 NDCG@3 개선을 보였습니다. 또한, 대화 맥락이 변화하는 상황에서도 높은 강인성을 보여주며, 다양한 대화 상황에서의 일반화 능력이 뛰어남을 입증했습니다.



### Listen Then See: Video Alignment with Speaker Attention (https://arxiv.org/abs/2404.13530)
- **Title**: 고급 답변 생성을 위한 비디오 기반 질의응답 시스템 개선 연구

- **What's New**: 이 연구에서는 사회적 지능을 요구하는 비디오 질의응답(Socially Intelligent Question Answering, SIQA) 작업을 위해, 오디오 모달리티를 브리지로 사용하여 비디오와 언어간의 효율적인 크로스-모달 연결 및 표현 융합 접근 방식을 도입합니다. 이를 통해 기존 기술들이 겪고 있던 언어 모델의 과적합 문제와 비디오 모달리티의 우회 문제를 해결하여 성능을 향상시키고 있습니다.

- **Technical Details**: 본 연구의 기술적 디테일에는 크로스-모달 정렬과 후속 표현 융합 접근법을 통해 비디오, 언어, 오디오 모달리티 간의 상호작용을 강화하는 방법이 포함됩니다. 연구팀은 CLIP(Cross-Lingual Image Pre-trained) 임베딩을 비디오와 언어 모달리티 요소에 융합하고, 결과적인 컨텍스트 임베딩을 언어 공간으로 투영하는 방법을 사용하였습니다.

- **Key Results**: 이 접근 방식을 통해 Social IQ 2.0 데이터셋에서 상태 탐지(state-of-the-art, SOTA) 결과인 82.06% 정확도를 달성하였습니다. 공개된 코드 및 모델은 학계 및 관련 분야 연구자들이 활용할 수 있도록 공유하고 있습니다.



### Parameter Efficient Fine Tuning: A Comprehensive Analysis Across  Applications (https://arxiv.org/abs/2404.13506)
- **What's New**: 이 논문에서는 전통적인 파라미터 튜닝 방법의 한계를 극복하고자 개발된 Parameter Efficient Fine-Tuning (PEFT, 파라미터 효율적인 파인 튜닝) 기술에 대한 체계적인 리뷰를 제공합니다. PEFT 기술은 연산 효율성과 성능의 균형을 맞추기 위해 일부 파라미터만을 선택적으로 업데이트 하는 방법으로, 다양한 도메인에서의 응용을 포함하여 상세하게 다루었습니다.

- **Technical Details**: PEFT 방법은 전체 모델 파라미터를 조정하는 풀 파인 튜닝 (Full Fine-Tuning)과 달리, 서브셋 파라미터 업데이트, 지식 전달 (knowledge distillation), 구조적 중복 활용 등을 통해 효율성을 극대화하도록 설계되었습니다. 특히, Low-rank Linear Subspace ReFT (LoReFT) 기법은 모델의 내부 표현만을 조정하여 파라미터 효율성을 향상시킵니다.

- **Performance Highlights**: LoReFT 기술은 상대적으로 동등하거나 우수한 성능을 제공하였으며, 특히 상식 추론 (commonsense reasoning)에서 LoReFT는 LLaMA-7B와 LLaMA-13B 모델에서 다른 PEFT 방법들보다 뛰어난 결과를 보여주었습니다. 또한, 응용 프로젝트에서는 메디컬 이미징의 정확성 향상, 프로테인 모델링 개선, 코드 리뷰 자동화 및 음성 합성 기술의 발전을 포함하여 다양한 분야에서의 효과를 입증했습니다.



### Intrusion Detection at Scale with the Assistance of a Command-line  Language Mod (https://arxiv.org/abs/2404.13402)
Comments: Accepted by IEEE/IFIP International Conference on Dependable Systems and Networks (DSN), industry track

- **What's New**: 이 논문에서는 대규모 자기 감독(self-supervised) 사전 학습을 통합한 '침입 탐지 시스템(Intrusion Detection System, IDS)'을 소개합니다. 이 시스템은 대규모 데이터의 힘을 이용하여 사용자의 커맨드 라인(command lines)을 기반으로 언어 모델(language model)을 훈련합니다. 이는 많은 기존 IDS가 처리하지 못하는 신규, 제로데이(zero-day) 공격을 포함해 확장성 있는 문제에 대처하기 위한 것입니다.

- **Technical Details**: 이 연구는 사용자의 커맨드 라인을 활용하여 언어 모델을 학습시키고, 이를 기반으로 IDS를 구축합니다. 시스템은 매주 수천만 개의 사용자 커맨드 라인을 학습하여 미래의 공격 및 침입을 탐지합니다. 사전 학습(pre-training)에는 BPE(Byte Pair Encoding) 토크나이저와 트랜스포머(Transformer) 아키텍처가 사용되며, 학습 모델은 마스크 언어 모델링(Masked Language Modeling, MLM) 전략을 사용하여 데이터에서 랜덤하게 마스킹된 토큰을 복원하는 방식으로 학습됩니다.

- **Performance Highlights**: IDS의 효과성은 3000만 개의 훈련 샘플과 1000만 개의 테스트 샘플을 사용하여 검증되었습니다. 모델은 기존 상용 IDS가 놓친 침입을 83% 이상의 정확도로 탐지할 수 있으며, 전체 예측 정확도는 99% 이상으로, 기존 상용 IDS를 대체할 수 있는 경쟁력 있는 성능을 제공합니다.



### Movie101v2: Improved Movie Narration Benchmark (https://arxiv.org/abs/2404.13370)
- **What's New**: 이 연구는 자동 영화 해설 생성의 발전을 위해 새로운 초대형 이중 언어 영화 해설 데이터셋인 Movie101v2를 소개합니다. 이 데이터셋은 오픈 소스의 대형 언어 모델(GPT-4V 포함)을 활용하여 개선되었으며, 양질의 데이터, 명확한 작업 목표, 그리고 세분화된 평가 방식을 통해 보다 효과적인 영화 해설 생성을 목표로 합니다.

- **Technical Details**: 이 데이터셋은 Whisper를 사용하여 영화 오디오를 텍스트로 전사하고, PaddleOCR을 통해 영화 자막의 시각적 단서를 사용하여 케릭터 대사를 식별하고 제거합니다. 그 후 GPT-4를 사용하여 남은 케릭터 대사를 제거하고, GPT-3.5-turbo를 이용하여 텍스트 내 오류를 교정합니다. Movie101v2는 더 많은 영화(총 203편)와 이중 언어 비디오-해설 쌍(46K 쌍)을 포함하고 있습니다.

- **Performance Highlights**: 이전 데이터셋들보다 더 큰 규모와 더 높은 질적 수준을 갖춘 Movie101v2를 사용하여 수행된 초기 실험에서 주요 대형 시각-언어 모델이 영화 클립 내 기본 시각적 사실과 줄거리를 이해하는 데 아직 미흡함을 확인하였습니다. 새로운 평가 프레임워크를 도입하여 L1 (기본 시각적 사실 설명)과 L2 (줄거리 추론 및 해설) 수준에서 내러티브 품질을 분리하여 평가함으로써 모델 개발에 보다 구체적인 피드백을 제공합니다.



### Personalized Wireless Federated Learning for Large Language Models (https://arxiv.org/abs/2404.13238)
Comments: 8 pages, 5 figures

- **What's New**: 이 연구에서는 무선 네트워크에서 자연어 처리(Natural Language Processing, NLP) 작업을 혁신하는 대규모 언어 모델(Large Language Models, LLMs)의 배포에 대한 도전을 탐구하고 있습니다. 특히, 개인화된 연방 학습(Federated Learning, FL)을 통해 이러한 모델을 효율적으로 조정하는 두 가지 새로운 방법, 즉 개인화된 연방 지시 조정(Personalized Federated Instruction Tuning, PFIT)과 개인화된 연방 과제 조정(Personalized Federated Task Tuning, PFTT)을 제안합니다. 이 방법들은 통신 오버헤드를 줄이면서도 모델의 개인화를 유지하는 것을 목표로 합니다.

- **Technical Details**: PFIT는 강화 학습(Reinforcement Learning)을 사용하여 다양한 보상 모델로 로컬 LLM을 세밀하게 조정하여 개인화를 달성합니다. PFTT는 글로벌 어댑터와 로컬 저랭크 조정(Low-Rank Adaptation, LoRA)을 결합하여 로컬 LLM을 협력적으로 미세 조정하며, 로컬 LoRA를 적용함으로써 집계 없이도 개인화를 달성할 수 있습니다. 이러한 방법들은 6G 통신에서 AI와 무선 네트워크의 깊은 통합을 통해 보다 지능적인 서비스를 지원할 수 있는 잠재력을 보여줍니다.

- **Performance Highlights**: 시뮬레이션을 통해 제안된 두 방법의 효과가 입증되었으며, 이는 LLM의 훈련과 fine-tuning 과정에서 데이터 보안 및 개인 정보 보호를 유지하는 동시에 대량의 분산 데이터를 활용할 수 있음을 보여줍니다. 특히, 통신 비용을 절감하면서 개인화된 결과를 제공할 수 있는 능력은 6G 사용자에게 맞춤형 서비스를 제공하는 데 중요합니다.



### The Instruction Hierarchy: Training LLMs to Prioritize Privileged  Instructions (https://arxiv.org/abs/2404.13208)
- **What's New**: 이번 연구에서는 GPT-3.5를 대상으로 수행되었으며, LLM(Language Large Models)이 다양한 우선 순위의 지시사항을 처리할 수 있는 '지시 계층 구조'를 제안합니다. 이 구조를 통해 모델은 상위 우선순위의 지시를 우선적으로 처리하며, 낮은 우선순위의 지시는 무시하거나 거부하는 방식으로 작동합니다. 이는 악의적인 프롬프트 주입(prompt injections)이나 시스템 권한 탈취(jailbreaks)와 같은 공격으로부터 LLM을 보호하는 데 중요한 역할을 합니다.

- **Technical Details**: 연구팀은 '계층적 지시사항 따르기(hierarchical instruction following)' 행동을 실현하기 위한 데이터 생성 방법을 개발했습니다. 이를 위해 합성 데이터 생성(synthetic data generation)과 내용 정제(context distillation) 기술을 사용하여, 서로 양립하는 지시사항(aligned instructions)에 대해서는 원래의 지시사항을 예측하도록, 그리고 서로 양립하지 않는 지시사항(misaligned instructions)에 대해서는 모델이 낮은 수준의 지시사항을 인식하지 못하게 하여 훈련하였습니다. 이렇게 훈련된 모델은 지시 계층에 따라 지시의 우선 순위를 인식하고, 필요한 경우 하위 지시를 무시하거나 거부할 수 있는 능력을 갖추게 됩니다.

- **Performance Highlights**: 새로운 접근 방식은 다양한 벤치마크에서 LLM의 강인성을 크게 향상시켰습니다. 예를 들어, 시스템 풀기(system prompt extraction)에 대한 방어는 63% 향상되었으며, 접근법은 훈련 중에 보지 못한 공격 유형에 대해서도 일반화를 보여주었습니다. 예를 들어, jailbreak에 대한 강인성은 30% 이상 증가했습니다. 비록 일부 '과도한 거부(over-refusals)'—모델이 때때로 해로운 질의를 무시하거나 거부하는 케이스—가 관찰되었지만, 모델의 일반 기능은 영향을 받지 않았으며, 추가적인 데이터 수집을 통해 이 문제를 해결할 수 있을 것으로 기대됩니다.



### A national longitudinal dataset of skills taught in U.S. higher  education curricula (https://arxiv.org/abs/2404.13163)
Comments: 44 pages, 21 figures, 10 tables

- **What's New**: 이 연구는 미국 내 거의 3천 개의 대학 및 대학교에서 수업된 300만 개 이상의 강의 계획서에서 추론된 기술을 포함하는 장기 데이터 세트를 처음으로 제시합니다. 이 데이터를 사용하여 학교에서 교육하는 기술과 미국 노동 통계국(Bureau of Labor Statistics)에 의해 문서화된 노동 시장의 기술 간의 유사성을 비교하고, 성별에 따른 기술 습득 차이, 사회 과학 커리큘럼에서의 기술 변화 추세, 그리고 대학 전공과 졸업생의 급여 차이 간의 상관 관계 등을 분석했습니다.

- **Technical Details**: 자연 언어 처리(Natural Language Processing, NLP) 기법을 사용하여 강의 계획서에서 미 노동부(Department of Labor, DOL)가 직업을 설명하는데 사용하는 상세 업무 활동(Detailed Workplace Activities, DWAs)을 추출하였습니다. 데이터는 강의 계획서에서 학습 목표와 관련된 문장만을 추출하고, 그 외 로지스틱 관련 문장은 제외하는 인간 중심의 접근 방식을 통해 수집되었습니다. 또한, DWAs와 능력 사이의 연관성을 설명하는 새로운 DWA2Ability 매핑을 소개하여, 능력 점수를 종속 변수로 사용하는 회귀 분석을 통해 직업 프로필 간의 매핑을 수행했습니다.

- **Performance Highlights**: 이 데이터셋은 대학에서 교육되는 기술의 광범위한 표현을 제공하며, 교육과 노동 시장 요구의 동조를 평가하는 데 사용됩니다. 각 대학과 학문 분야별로 기술 프로파일을 생성하여 노동력 개발의 문맥에서 기술의 출처에 대한 새로운 연구를 가능하게 하고, 진화하는 노동 수요를 충족하기 위해 고등 교육의 미래를 형성하는 데 도움이 될 수 있는 실행 가능한 통찰을 제공합니다. 또한, Random Forest Regressor를 사용하여 O*NET 능력에 대해서 최대 0.025의 평균 제곱 오차를 달성하는 52개의 모델을 개발하여 높은 정확도를 보여줍니다.




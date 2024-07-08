New uploads on arXiv(cs.CL)

### Me, Myself, and AI: The Situational Awareness Dataset (SAD) for LLMs (https://arxiv.org/abs/2407.04694)
Comments:
          11 page main body, 98 page appendix, 58 figures

- **What's New**: 새로운 연구는 ChatGPT와 같은 AI 어시스턴트가 '나는 대형 언어 모델입니다(I am a large language model)'라고 응답할 때, 이들이 실제로 자신이 LLM(대형 언어 모델)임을 알고 이를 기반으로 신뢰성 있게 행동하는지에 대한 의문을 제기합니다. 연구진은 LLM의 자기 인식 및 상황 인식을 정량화하기 위해 Situational Awareness Dataset (SAD)이라는 벤치마크를 도입했습니다. 이 데이터셋은 7가지 작업 카테고리와 13,000개 이상의 질문으로 구성되어 있습니다.

- **Technical Details**: SAD는 질문 응답 및 지시 따르기를 기반으로 한 일련의 행동 테스트를 통해 LLM의 상황 인식을 평가합니다. 이 벤치마크는 LLM이 (i) 자신이 생성한 텍스트를 인식하는 능력, (ii) 자신의 행동을 예측하는 능력, (iii) 프롬프트가 내부 평가에서 나온 것인지 실제 배포에서 나온 것인지를 판단하는 능력, (iv) 자기 인식에 의존하는 지시를 따르는 능력 등을 테스트합니다. 16개의 LLM을 SAD를 이용해 평가했으며, 여기에는 기본(pretrained) 모델과 챗(chat) 모델이 포함됩니다.

- **Performance Highlights**: 모든 모델이 경우의 수보다 나은 성과를 보였지만, 최고 성능을 보인 Claude 3 Opus마저도 일부 작업에서는 인간 기준에 크게 미치지 못했습니다. SAD 성능은 일반 지식 메트릭(MMLU)과는 부분적으로만 상관 관계가 있음을 발견했습니다. AI 어시스턴트로서 파인튜닝된(chat) 모델은 해당 기본(base) 모델보다 SAD에서 더 나은 성과를 보였지만, 일반 지식 작업에서는 그렇지 않았습니다.

- **Conclusion**: 이 연구는 상황 인식에 대한 정량적 이해를 돕기 위해 SAD를 도입했으며, 이는 LLM의 자율 계획 및 행동 능력을 향상시킬 수 있습니다. 그러나 이는 AI 안전 및 통제와 관련된 새로운 위험도 동시에 초래할 수 있습니다. 현재의 코드와 최신 결과는 제공된 URL에서 확인 가능합니다.



### ANAH-v2: Scaling Analytical Hallucination Annotation of Large Language Models (https://arxiv.org/abs/2407.04693)
Comments:
          9 pages

- **What's New**: 이번 논문에서는 대규모 언어 모델(LLMs)에서 발생하는 환각 문제를 해결하기 위해, 효율적인 환각 주석 데이터셋을 확장하고 더 정확한 주석기를 개발하는 반복적인 자기 학습 프레임워크를 제안합니다. 이 프레임워크는 Expectation Maximization (EM) 알고리즘에 기초하여 주어진 데이터셋을 점진적으로 주석하고, 이를 통해 더 정밀한 환각 주석기를 학습합니다.

- **Technical Details**: 제안된 프레임워크는 EM 알고리즘을 활용하여 환각 주석기의 성능을 개선하고 데이터셋의 규모를 확장합니다. EM 알고리즘의 기대 단계(Expectation step)에서는 기존의 최고의 주석기를 사용해 확장된 데이터셋의 실제 환각 주석을 예측합니다. 최대화 단계(Maximization step)에서는 이전 단계에서 수집된 주석 데이터를 기반으로 더 향상된 주석기를 학습합니다. 이렇게 구축된 주석기를 다음 반복에서도 유용하게 사용할 수 있습니다.

- **Performance Highlights**: 실험 결과, 7B 매개변수를 사용하는 최종 주석기는 GPT-4의 성능을 뛰어넘어 HaluEval과 HalluQA 데이터셋에서 새로운 최첨단 (state-of-the-art) 환각 검출 결과를 달성했습니다. 특히, 네추럴 랭귀지 추론(Natural Language Inference, NLI) 메트릭에서도 25%에서 37%로 향상되었습니다. 이러한 주석기는 대규모 데이터셋에서 다양한 LLM들의 환각 수준을 평가하고, LLM의 생성물에서 환각을 줄이는 데 중요한 역할을 할 수 있습니다.



### Entity Decomposition with Filtering: A Zero-Shot Clinical Named Entity Recognition Framework (https://arxiv.org/abs/2407.04629)
Comments:
          Preprint

- **What's New**: 현재 의료명 인식(NER) 분야에서 대규모 언어 모델(LLM)이 우수한 성과를 보여주고 있습니다. 이전 연구들은 주로 독점적인 LLM을 중심으로 했지만, 이 논문에서는 개방형 NER LLM이 의료명 인식에 어떻게 성능을 발휘하는지를 조사하였습니다. 이를 개선하기 위해 새로운 프레임워크인 '엔터티 디컴포지션(EDF)'을 제안하였습니다. 핵심 아이디어는 엔터티 인식 작업을 여러 하위 엔터티 유형으로 분해하고, 잘못된 엔터티를 제거하는 필터링 메커니즘을 도입하는 것입니다.

- **Technical Details**: 논문은 다양한 데이터셋(i2b2 2010, ClinicalIE, CLEF 2014, i2b2 2012, i2b2 2018 Task 2)을 사용하여 실험을 진행했습니다. 또한, 정확도(precision)와 재현율(recall) 성능을 각 데이터셋과 엔터티 유형별로 제시하며, UniNER과 GNER 프레임워크를 비교하였습니다. EDF 프레임워크는 '필터 프롬프트(Filter Prompt)'를 사용하여 엔터티가 특정 타입에 속하는지 여부를 확인합니다. 또한, 소수 샘플 실험(Few-shot Experiment)에서는 몇 개의 주석 샘플을 포함시키는 접근법을 사용하여 표준 인-컨텍스트 학습과 비교하였습니다.

- **Performance Highlights**: 실험 결과, EDF 프레임워크는 모든 메트릭, 모델, 데이터셋, 엔터티 유형에서 우수한 성능을 보였습니다. 특히, BERT 기반 모델(GLiNER)과의 실험에서는 F1 스코어가 평균 7.29% 향상되었습니다. 또한, '임상 부서(clinical department)' 엔터티에 대한 성능 감소가 관찰되었으며, 더 명확한 엔터티 설명을 사용하면 성능이 향상될 수 있음을 발견했습니다. 필터링을 통해 정밀도와 재현율 간의 트레이드오프를 분석한 결과, 특정 임계값을 조정하면 F1 스코어가 향상될 수 있음을 보였습니다.



### ARM: Efficient Guided Decoding with Autoregressive Reward Models (https://arxiv.org/abs/2407.04615)
- **What's New**: 이 연구에서는 안전한 실제 배포를 위해 대량의 데이터로 훈련된 언어 모델(Large Language Models, LLMs)을 조율하는 방법을 재검토합니다. 특히 자체 회귀적 보상 모델(autoregressive reward model)을 활용한 효율적인 guided decoding 방법을 제안하였습니다. 이 방법은 보상 모델의 점수를 사용해 기본 언어 모델의 로짓을 보완합니다.

- **Technical Details**: 기본 언어 모델의 로짓과 태스크 특화 보상 모델의 점수를 통합해 디코딩을 안내합니다. 더 작은 전문 모델을 훈련시켜 기본 모델의 토크나이저를 공유하고, 이 전문 모델은 디코딩 시 기본 모델의 로짓을 수정하거나 재배열하여 원하는 제약 조건을 충족합니다. 이 방법은 Free-form 외부 분류기(classifiers)를 사용해 디코딩을 제어하는 기존 방법들의 단점을 보완하며, 회귀적 모델을 활용해 효율적인 로짓 점수를 예측합니다.

- **Performance Highlights**: 제안된 방법은 디톡스화 및 감정 제어 작업에서 강력하지만 덜 효율적인 RAD 방식과 유사한 성능을 보여줍니다. 효율적인 파라미터화 덕분에 디코딩 시 높은 품질의 샘플을 생성하면서도 빠른 추론이 가능합니다.



### Testing learning hypotheses using neural networks by manipulating learning data (https://arxiv.org/abs/2407.04593)
Comments:
          Submitted to Journal of Memory and Language

- **What's New**: 이 연구는 신경망 언어 모델(neural network language models)을 사용하여 영어 수동태(passivization)에 대한 예외를 학습하는 방법을 탐구합니다. 영어 사용자가 수동태 전환이 가능한 동사와 그렇지 않은 동사를 어떻게 구분하는지를 이해하기 위해 간접적인 증거를 활용할 수 있는지 조사합니다.

- **Technical Details**: 연구진은 신경망 언어 모델을 통해 인간의 판단과 유사한 수동태 제한 규칙을 학습할 수 있는지 테스트했습니다. 이를 위해 수정된 학습 데이터를 사용하여 두 개의 주요 가설을 검증했습니다: (1) 동사의 수동태 사용 빈도가 그 동사의 수동화 가능성에 영향을 미친다는 가설과 (2) 동사의 의미적 내용이 수동화 가능성과 직접적으로 연결된다는 가설입니다.

- **Performance Highlights**: 실험 결과, 동사의 수동태 사용 빈도가 높은 경우 수동화 가능성이 높아지지만, 동사의 의미적 내용은 큰 영향을 미치지 않는 것으로 나타났습니다. 이는 신경망 모델이 간접적인 증거를 통해 인간과 유사한 수동태 예외를 학습할 수 있음을 시사합니다.



### Not (yet) the whole story: Evaluating Visual Storytelling Requires More than Measuring Coherence, Grounding, and Repetition (https://arxiv.org/abs/2407.04559)
- **What's New**: 이 연구에서는 템포럴하게 정렬된 이미지 시퀀스를 통해 자연 언어로 스토리를 생성하는 '비주얼 스토리텔링'(Visual Storytelling) 작업의 질을 평가하는 새로운 방법을 소개합니다. 이 방법은 이전 연구에서 강조된 시각적 기반, 일관성, 반복성 등 세 가지 주요 측면에서 인간 유사성을 기준으로 스토리의 질을 측정합니다.

- **Technical Details**: 기존의 비주얼 스토리텔링 모델과 더불어, 일반적인 언어 및 시각 능력을 가진 기반 모델인 LLaVA를 처음으로 '제로 샷' 방식으로 테스트했습니다. 이 연구에서는 LLaVA 모델이 최상의 성능을 나타내지만, 50배 작은 비주얼 스토리텔링 모델인 TAPM과는 약간의 차이만 보인다고 확인했습니다. 또한 TAPM의 시각 및 언어 구성 요소를 업그레이드하여 적은 수의 파라미터로 유사한 성능을 달성했습니다.

- **Performance Highlights**: 연구 결과, LLaVA와 업그레이드된 TAPM 모델이 스토리의 시각적 기반, 일관성, 반복성 측면에서 인간 작성 스토리에 매우 근접한 것으로 나타났습니다. 그러나 질적 연구에서 인간은 모델이 생성한 스토리보다 인간이 작성한 스토리를 선호하는 경향이 있다는 점을 확인했습니다.



### Spontaneous Reward Hacking in Iterative Self-Refinemen (https://arxiv.org/abs/2407.04549)
- **What's New**: 최신 연구에서는 언어 모델(Language Models)이 자연어 피드백을 통해 출력을 개선하고 사용자 선호도를 최적화하는 'Self-Refinement(자가 개선)' 방법을 탐구했습니다. 특히 두 번째 언어 모델을 평가자로 사용하여 피드백과 숫자 평가를 제공함으로써, 사용자 선호도에 맞추어 출력을 최적화하려는 시도를 했습니다. 하지만 이는 'Reward Hacking(보상 해킹)'을 야기할 수 있으며, 평가자의 점수만 높아지고 실제 사용자 만족도는 개선되지 않거나 오히려 저하될 수 있다는 문제를 확인했습니다.

- **Technical Details**: 이번 연구에서는 에세이 편집 과제를 통해 'Iterative Self-Refinement(반복 자가 개선)'이 평가자와 인간 판단 사이의 편차를 야기할 수 있음을 보였습니다. 실험에서 사용된 평가자는 생성자와 동일한 언어 모델(LM)을 기반으로 하며, 서로 다른 프롬프트로 작동합니다. 생성자가 출력한 결과에 대해 평가자가 피드백과 점수를 제공하고, 생성자는 이를 바탕으로 출력을 개선합니다. 이 과정이 반복되고, 피드백과 점수가 최적화되는 동안 실제 품질은 유지되거나 악화될 수 있습니다.

- **Performance Highlights**: 연구 결과, 모델 크기와 생성자와 평가자 사이의 컨텍스트 공유 정도가 보상 해킹의 심각성에 영향을 미치는 것으로 나타났습니다. 특히 GPT-4는 GPT-3.5보다 보상 해킹의 영향이 덜한 것으로 보였습니다. 또한 비대칭 컨텍스트는 GPT-3.5에서 보상 해킹을 줄이는 데 도움이 되는 것으로 나타났습니다.



### Strengthening Structural Inductive Biases by Pre-training to Perform Syntactic Transformations (https://arxiv.org/abs/2407.04543)
- **What's New**: 이 연구에서는 Transformer 모델의 구조적 귀납 편향(inductive bias)을 강화하기 위해 추가적인 중간 사전 학습 단계를 도입했습니다. 이 단계에서 의존 구문 트리(dependency trees)에 대한 전환을 수행함으로써 모델이 구문론적 변환을 예측하는 방법을 학습하게 됩니다.

- **Technical Details**: 중간 사전 학습은 Transformer 모델을 대상으로 영어 의존 구문 트리의 자동 생성된 구문 변환을 수행하도록 하는 데이터셋을 사용합니다. 변환 설명을 프리픽스(prefix)로 제공하고 입력 문장을 받아 결과를 예측하도록 모델을 사전 학습합니다. 이렇게 함으로써 모델은 구문 표현을 강화하고, 내려받기 작업에 활용 가능한 재사용 가능한 동적 변환을 획득하게 됩니다.

- **Performance Highlights**: 구문 의존적 seq2seq 작업(예: 능동태에서 수동태 변환, chunking)에서 몇 번의 샷 학습(few-shot learning) 성능이 향상되었으며, 의미 분석(semantic parsing)에서도 구조적 일반화가 개선되었습니다. 주의(attention) 헤드가 구문 패턴을 따르며, 미세 조정 후에도 이러한 헤드를 재사용하는 경향이 있어 사전 학습 중 획득된 변환을 활용하는 모습을 보였습니다.



### PoPreRo: A New Dataset for Popularity Prediction of Romanian Reddit Posts (https://arxiv.org/abs/2407.04541)
Comments:
          Accepted at ICPR 2024

- **What's New**: PoPreRo라는 새로운 데이터셋이 소개되었습니다. 이는 Reddit에서 수집된 루마니아 게시물의 인기 예측(Popularity Prediction)을 위한 첫 번째 데이터셋입니다. PoPreRo 데이터셋은 루마니아의 다섯 개 서브레딧에서 수집된 28,107개의 데이터 샘플을 포함합니다. 또한, 본 연구에서는 향후 연구를 위한 경쟁 모델 세트를 소개합니다.

- **Technical Details**: PoPreRo 데이터셋은 Reddit의 다섯 개 루마니아 서브레딧(Romania, Bucureşti, Cluj, Iaşi, Timişoara)에서 수집된 게시물로 구성되어 있습니다. 각 게시물은 제목, 내용, 댓글 수, 업보트와 다운보트 수 등의 정보를 포함합니다. 데이터셋은 API를 사용하여 수집되었으며, 총 28,107개의 샘플로 구성되어 있습니다. 인기도는 업보트와 다운보트의 합을 기준으로 이진 라벨(인기/비인기)로 분류됩니다.

- **Performance Highlights**: 평가 결과, 최고 성능의 모델은 테스트 세트에서 61.35%의 정확도와 60.60%의 매크로 F1 점수를 달성했습니다. 이는 인기도 예측 작업이 매우 도전적인 과제임을 시사합니다. 추가로 Falcon-7B 대형 언어 모델을 기반으로 한 Few-shot 프롬프트 방식을 조사한 결과도 동일한 방향을 지적합니다.



### Performance Analysis of Speech Encoders for Low-Resource SLU and ASR in Tunisian Dialec (https://arxiv.org/abs/2407.04533)
Comments:
          Accepted in ArabicNLP 2024

- **What's New**: 이 논문은 저자들이 셀프 슈퍼바이즈드 러닝 (SSL) 기반의 음성 인코더를 사용해 저자원이 필요한 튀니지아 아랍어 방언에 대한 자동 음성 인식 (ASR)과 음성 언어 이해 (SLU)을 비교하는 첫 연구입니다. SSL 모델들이 저자원이 필요한 언어 환경에서 얼마나 효과적인지 평가하며, 특히 최소한의 의미 주석이 주어진 상황에서의 성능을 탐구합니다.

- **Technical Details**: 이 연구에서는 TARIC-SLU 데이터셋을 사용해 다양한 SSL 음성 인코더를 실험합니다. 단일언어 및 다중언어로 사전 학습된 모델들이 사용되었으며, 일부 모델은 멀티모달 슈퍼바이즈드 티처-스튜던트 (teacher-student) 패러다임을 통해 추가로 정제되었습니다. 사용된 SSL 모델로는 wav2vec 2.0, wavLM, data2vec, w2v-BERT, w2v-BERT 2.0, SONAR 등이 있으며, 특정 모델들은 다중언어 텍스트/오디오 페어 데이터를 이용해 정제되었습니다.

- **Performance Highlights**: 실험 결과, SSL 음성 인코더들이 튀니지아 아랍어 방언의 ASR과 SLU에서 뛰어난 성능을 보였으며, 특히 다중언어로 학습된 모델들이 더 나은 성능을 보였습니다. 또한, semantic encoding refine이 ASR과 SLU 태스크의 성능을 향상시킨다는 점이 확인되었습니다. 최신의 SSL 모델들인 w2v-BERT 2.0과 SONAR가 특히 주목할 만한 성능을 보여주었습니다.



### GPT vs RETRO: Exploring the Intersection of Retrieval and Parameter-Efficient Fine-Tuning (https://arxiv.org/abs/2407.04528)
- **What's New**: 최근 연구에서는 대형 언어 모델을 적응시키면서 계산 요구를 최소화하는 방법으로 PEFT (Parameter-Efficient Fine-Tuning)와 RAG (Retrieval-Augmented Generation)가 주목받고 있습니다. 이번 논문에서는 PEFT 기법(P-tuning, Adapters, 및 LoRA)을 수정된 RETRO 모델과 GPT 모델에 적용하여 여러 크기의 모델에 대해 비교 분석했습니다. 특히, 이 연구는 여러 PEFT 방법이 RAG와 통합된 첫 번째 포괄적인 비교 분석을 제공합니다.

- **Technical Details**: PEFT는 업데이트하는 매개변수의 수를 줄이면서도 성능을 유지하는 기법입니다. 이 논문에서는 PEFT 기법으로 P-tuning, Adapter 모듈과 Low-Rank Adaptation (LoRA)을 사용했습니다. RETRO 모델은 캡슐화된 크로스 어텐션 메커니즘을 통해 검색 모듈을 직접 Transformer 아키텍처에 통합하여 학습하는 고유한 접근 방식을 가지고 있습니다. RAG는 외부 지식을 통합하여 모델 성능을 향상시키는 방법으로, 검색된 소스를 입력 쿼리에 연결하여 활용합니다.

- **Performance Highlights**: 실험 결과, RETRO 모델은 zero-shot 설정에서 GPT 모델을 능가했으며, 특히 8B 파라미터 모델이 비용과 성능의 균형을 가장 잘 맞췄습니다. 또, P-tuning은 다른 PEFT 기법에 비해 성능이 떨어졌습니다. RETRO 모델은 검색된 텍스트에서 핵심 정보를 추출하고 이를 생성 프로세스에 통합하는데 뛰어난 능력을 보여줬습니다.



### Leveraging Graph Structures to Detect Hallucinations in Large Language Models (https://arxiv.org/abs/2407.04485)
- **What's New**: 이 논문에서는 대형 언어 모델(Large Language Models, LLMs)의 환각 발생(hallucination)을 탐지하는 새로운 방법을 제안합니다. LLMs는 고객 지원, 콘텐츠 생성, 교육, 금융 안내 등에서 활용되지만, 환각 생성으로 인해 정보의 신뢰성이 떨어질 수 있습니다. 이를 해결하기 위해 잠재 공간(latent space)의 구조를 분석하고, 환각 생성과 비환각 생성을 구분할 수 있는 그래프 구조를 제안합니다.

- **Technical Details**: 이 방법은 잠재 공간에서 가까운 위치에 있는 생성을 연결하는 그래프 구조를 만들고, 인접 노드로부터 정보를 모으는 메시지 전달(message passing)을 이용하는 Graph Attention Network(GAT)를 사용합니다. 원리는 상호 유사한 특성을 가진 엔티티들이 연결되어 있다는 '호모필리'(homophily) 원칙에 기반합니다.

- **Performance Highlights**: 연구 결과, 의미론적 정보(semantic information)를 이용해 잠재 공간에서 생성을 연결하는 것이 환각 생성과 비환각 생성 간의 차이를 밝히는 데 효과적임을 확인했습니다. Contrastive learning을 활용하면 성능이 향상되며, 새로운 데이터를 분류할 때도 잘 일반화됩니다. 외부 지식이나 추가적인 LLM 추론 없이도 기존의 벤치마크들과 유사한 성능을 보입니다.



### Using LLMs to label medical papers according to the CIViC evidence mod (https://arxiv.org/abs/2407.04466)
- **What's New**: 우리는 의료 NLP 분야에 시퀀스 분류 문제인 CIViC Evidence를 도입했습니다. CIViC Evidence는 다양한 게놈 변이, 암 유형, 치료 접근 방식을 조사한 과학 논문의 초록에 임상 증거 라벨을 할당하는 멀티 라벨 분류 문제를 의미합니다.

- **Technical Details**: 우리는 다양한 언어 모델을 이용하여 CIViC Evidence 문제를 해결하려고 시도했습니다. BERT와 RoBERTa의 사전학습된 체크포인트를 CIViC Evidence 데이터셋에 맞게 파인튜닝(fine-tuning)하였고, 도메인 특화 텍스트로 사전학습된 모델들과 성능을 비교했습니다. 그 결과, BiomedBERT와 BioLinkBERT가 BERT보다 높은 성능을 보였습니다(+0.8% 및 +0.9%의 클래스 지원 가중 F1 스코어 절대 개선).

- **Performance Highlights**: 모든 트랜스포머(transformer) 기반 모델들이 빅램(bigram) tf-idf 점수를 사용하여 훈련된 로지스틱 회귀(logistic regression) 모델과 비교했을 때 뚜렷한 성능 우위를 보였습니다(+1.5 - 2.7% 개선된 F1 스코어). OpenAI의 GPT-4와 소샘플링 몇몇 실험(few-shot setting) 하에 비교했을 때, 추가적인 프롬프트 엔지니어링(prompt engineering)이나 파인튜닝 없이 GPT-4의 성능이 66.1% 가중 F1 스코어로 파인튠된 모델(최고 71.8%)보다 낮지만, 로지스틱 회귀 모델(67.7%)과는 비슷한 성능을 보였습니다.



### Generalists vs. Specialists: Evaluating Large Language Models for Urdu (https://arxiv.org/abs/2407.04459)
- **What's New**: 이 연구는 일반 목적의 사전 학습 모델인 GPT-4-Turbo와 Llama-3-8b-Instruct을 특정 작업에 맞춰 조정된 특수 목적 모델(XLM-Roberta-large, mT5-large, Llama-3-8b-Instruct)과 비교합니다. 특히 우르두어(약 7천만 명의 모국어 사용자)에 대한 성능을 평가합니다. 이 연구는 NLP 커뮤니티에 저자원 언어에 대한 일반 및 특정 목적의 LLMs(Long Language Models)의 효과성에 대한 통찰을 제공합니다.

- **Technical Details**: 이 연구는 분류와 생성 작업에서 일반 모델과 특수 모델의 성능을 비교합니다. 분류 작업에는 감정 분석, 욕설 탐지, 풍자 탐지, 가짜 뉴스 탐지, 주제 분류, 품사 태깅, 명명된 개체 인식을 포함하며, 생성 작업에는 질문 응답을 포함합니다. 전체 13개의 작업 중 7개는 분류, 6개는 생성 작업입니다. GPT-4-Turbo와 Llama-3-8b-Instruct를 일반 목적 모델로 사용하고, XLM-Roberta-large, mT5-large, Llama-3-8b-Instruct(윤곽 조정됨)를 특수 모델로 사용합니다.

- **Performance Highlights**: 특수 목적 모델이 대부분의 작업에서 일관되게 일반 모델을 능가했습니다. 생성 작업의 평가에서는 GPT-4-Turbo의 자동 평가 결과가 인간 평가와 더 일치했으며, Llama-3-8b-Instruct의 평가 결과와 비교해 더 정확한 평가를 제공했습니다. 이는 특수 모델의 조정이 일반 모델에 비해 성능 향상에 크게 기여한다는 사실을 나타냅니다. 모든 코드, 모델 출력, 및 평가 결과는 GitHub에서 공개됩니다.



### TokenVerse: Unifying Speech and NLP Tasks via Transducer-based ASR (https://arxiv.org/abs/2407.04444)
Comments:
          5 pages, double column

- **What's New**: 전통적인 대화형 인공지능에서는 음성 인식을 위한 다양한 작업이 연속적인 파이프라인으로 이루어져야 합니다. 이번 논문에서는 TokenVerse라는 모델을 소개하여, 하나의 Transducer 기반 모델로 여러 작업을 동시에 수행할 수 있도록 했습니다. 이를 통해 음성 인식(ASR), 화자 변경 감지(Speaker Change Detection), 구간 종료(Endpointing), 그리고 개체명 인식(NER) 등 다양한 작업을 한 번에 처리할 수 있습니다.

- **Technical Details**: TokenVerse는 작업별 토큰을 참고 텍스트에 통합하여 ASR 모델 학습을 최적화하는 방식으로 설계되었습니다. ASR 모델은 XLSR-Transducer 구조를 사용하며, self-supervised 학습된 XLSR-53 모델을 인코더로 활용합니다. 주요 작업별 '토큰'은 [SCD] (화자 변경 감지), [NE]와 [/NE] (개체명 인식), [ENDP] (구간 종료)로 정의되어 있습니다.

- **Performance Highlights**: 공개 및 비공개 데이터셋에서 실험한 결과, 제안된 방법은 ASR 성능을 최대 7.7% 향상시키고, 개별 작업 수행 시 기존의 연속적 파이프라인 접근법보다 우수한 성능을 보였습니다. 또한, 새로운 작업으로의 전이 학습(Task Transfer Learning)도 가능합니다.



### From 'Showgirls' to 'Performers': Fine-tuning with Gender-inclusive Language for Bias Reduction in LLMs (https://arxiv.org/abs/2407.04434)
Comments:
          10 pages, 5 tables; to appear in Proceedings of the 5th Workshop on Gender Bias in Natural Language Processing at ACL 2024

- **What's New**: 이 논문에서는 영어의 'show-girl'이나 'man-cave'와 같은 성별-배제 접사가 성별 고정관념을 강화하고 이분법적 성별 개념을 보존하는 문제를 다룹니다. 연구팀은 692개의 성별-배제 용어와 성별-중립 용어를 포함한 데이터셋 'Tiny Heap'을 개발하여 세 가지 다른 대형 언어 모델(LLMs)을 미세 조정(fine-tuning)한 결과, 모델의 성별 고정관념 경향이 전반적으로 감소했습니다.

- **Technical Details**: 연구팀은 OpenWebText2에서 성별을 표시하는 접두사와 접미사가 포함된 명사를 추출하여 성별-배제 용어와 성별-중립 대안을 포함한 692개의 용어 쌍 목록을 작성했습니다. 이 리스트는 이전에 이용 가능한 리소스보다 세 배 이상 큽니다. 그 후 NeuTral Rewriter를 사용하여 성별이 표시된 대명사(he, she 등)를 단수형 they로 대체했습니다. 이 데이터셋을 사용하여 세 가지 다른 LLMs(마스크드와 인과적 모델 포함)을 미세 조정했습니다.

- **Performance Highlights**: 미세 조정 결과, 모델의 성별 고정관념을 표현하는 경향이 전반적으로 감소했으며, 성별이 표시된 문맥에서 유해한 언어 생성도 감소했습니다. 이 접근 방식은 LLM 훈련 데이터에 성별 포괄성을 향상시키는 실질적인 방법을 제공하며, 성별 편견 완화 연구와 queer-feminist 언어 운동의 통합에 기여합니다.



### Romanization Encoding For Multilingual ASR (https://arxiv.org/abs/2407.04368)
- **What's New**: 로마자 인코딩(romanization encoding)을 사용하여 다국어 및 코드 스위칭 자동 음성 인식(ASR) 시스템의 성능을 최적화하는 방법을 소개합니다. FastConformer-RNNT 프레임워크와 Roman2Char 모듈을 사용하여 어휘 크기를 63.51%로 줄이고, SEAME 코드 스위칭 벤치마크에서 13.72% 및 15.03%의 성능 향상을 달성했습니다.

- **Technical Details**: 이 연구에서는 로마자 인코딩을 사용하여 어휘 크기를 줄이고 모델의 유연성과 적응성을 향상시킵니다. FastConformer-RNNT 모형과 결합된 로마자 인코딩 및 균형 잡힌 토크나이저(tokenizer)를 사용해 어쿠스틱 모델링과 언어 모델링을 분리하였습니다. 또한, Mandarin-English ASR 데이터를 사용하여 이 방법을 검증하였고, Mandarin-Korean과 Mandarin-Japanese 데이터를 통해 추가적인 실험을 수행했습니다.

- **Performance Highlights**: 본 연구에서는 Mandarin-English ASR 데이터에서 어휘 크기를 63.51%로 줄였으며, SEAME 코드 스위칭 벤치마크에서 각각 13.72%와 15.03%의 성능 향상을 달성했습니다. 이러한 결과는 본 방법이 다른 스크립트가 많은 언어에서도 효과적으로 적용될 수 있음을 시사합니다.



### Crafting Large Language Models for Enhanced Interpretability (https://arxiv.org/abs/2407.04307)
Comments:
          Present at ICML 2024 Mechanistic Interpretability (MI) Workshop

- **What's New**: 이번 연구에서는 새로운 개념 병목 대형 언어 모델(Concept Bottleneck Large Language Model, CB-LLM)을 소개합니다. 이 모델은 기존의 블랙박스 대형 언어 모델(LLM) 대비 내재적으로 해석 가능하고 투명하게 설계되었습니다. 특히 자동 개념 수정(Automatic Concept Correction, ACC) 전략을 통해 성능 격차를 좁히며, 높은 정확성과 명확한 해석 가능성을 동시에 제공합니다.

- **Technical Details**: CB-LLM은 블랙박스 모델을 해석 가능한 엔티티로 변환하는 혁신적인 방법론을 제시합니다. 본 연구의 방법론은 사전 훈련된 모델을 변환하는 데 중점을 둡니다. 모델 변환 과정은 아래와 같습니다: (1) 개념 생성: 텍스트 분류 작업에 대한 개념 세트를 생성합니다. (2) 자동 개념 점수화(Automatic Concept Scoring, ACS): 텍스트 샘플과 개념 간의 유사성을 평가합니다. (3) 개념 병목층 훈련: 신경 활성화와 개념 점수 간의 유사성을 최대화하여 인간이 이해 가능한 개념으로 매핑합니다. (4) 예측기 학습: 최종 선형 계층을 훈련하여 다운스트림 작업에 대한 예측을 수행합니다.

- **Performance Highlights**: CB-LLM은 기존 블랙박스 모델과의 성능 격차를 좁히면서도, 1.39배 더 높은 평균 평점을 얻어냅니다. 이는 해석 가능성을 유지하면서도 높은 성능을 제공한다는 것을 의미합니다.



### Systematic Evaluation of Online Speaker Diarization Systems Regarding their Latency (https://arxiv.org/abs/2407.04293)
Comments:
          6 pages

- **What's New**: 이 논문은 다양한 온라인 화자 분할 시스템(speaker diarization systems)의 레이턴시(latency)를 동일한 하드웨어와 테스트 데이터에서 평가합니다. 레이턴시는 오디오 입력부터 해당 화자 레이블이 출력되기까지의 시간 간격입니다. DIART 프레임워크 모델 조합과 엔드 투 엔드(End-to-End) 온라인 화자 분할 시스템인 FS-EEND를 비교하여 저지연을 갖춘 시스템을 확인합니다. 이를 통해 현재 여러 온라인 화자 분할 시스템의 레이턴시를 비교한 첫 연구로서의 의의를 가집니다.

- **Technical Details**: 화자 분할 시스템은 '누가 언제 말했는가'라는 질문에 답하는 머신러닝 과제입니다. 이 논문은 DIART 프레임워크에서 다양한 세그멘테이션(SEGMENTATION) 및 임베딩(EMBEDDING) 모델 조합과 UIS-RNN-SML 및 FS-EEND 시스템의 레이턴시를 평가합니다. 우선 TIMIT 데이터셋을 사용해 시스템을 트레이닝 했으며, 테스트 데이터는 Voxconverse 테스트셋의 하위 집합을 사용했습니다. DIART 프레임워크는 WAV 형식의 오디오 파일을 직접 처리할 수 있지만, UIS-RNN-SML은 클러스터링용 오디오 임베딩이 필요하며, FS-EEND는 칼디(Kaldi) 스타일 데이터를 요구합니다.

- **Performance Highlights**: DIART 파이프라인은 최저 레이턴시를 기록했으며, 특히 'pyannote/embedding' 임베딩 모델과 'pyannote/segmentation' 세그멘테이션 모델 조합이 우수한 성능을 보였습니다. FS-EEND 시스템도 비슷한 수준의 저지연 성능을 보여줬습니다. DIART와 FS-EEND 시스템은 레이턴시 평가에서 눈에 띄는 성과를 보였지만, 각각 다른 모델 구성과 접근 방식으로 차별화되었습니다.



### LearnerVoice: A Dataset of Non-Native English Learners' Spontaneous Speech (https://arxiv.org/abs/2407.04280)
Comments:
          Accepted for INTERSPEECH 2024

- **What's New**: LearnerVoice는 자동 음성 인식(ASR) 시스템의 성능을 향상시키기 위해 만들어진 새로운 L2(제2언어) 학습자의 자발적 영어 음성 데이터셋입니다. 이 데이터셋은 50.04시간의 오디오와 전자 기록을 포함하며, 주로 한국어를 모국어로 하는 58명의 학습자들의 자발적인 말하기 데이터를 포함하고 있습니다.

- **Technical Details**: LearnerVoice 데이터셋은 Ringle이라는 화상 교육 플랫폼을 통해 수집된 자발적인 영어 대화로 구성되어 있습니다. 이 데이터셋에는 239개의 수업에서 수집된 50.04시간의 오디오가 포함되어 있으며, 32kHz의 샘플링 속도로 기록되었습니다. 수업은 주제로 일상 생활, 비즈니스/경제, 시사/정치 및 문화/스포츠 등을 다루며, L2 학습자의 다양한 발음 오류와 비문법적 표현이 잘 반영되어 있습니다. 데이터는 Voice Activity Detection 모델을 사용하여 짧은 단위로 분할되었고, 훈련된 인간 주석자가 전사 작업을 수행했습니다.

- **Performance Highlights**: whisper-small.en 모델을 LearnerVoice로 미세 조정한 결과, WER (Word Error Rate)이 10.26%로, 기존 whisper-small.en 모델 대비 44.2% 감소했습니다. 특히 L2S 기능과 관련된 오류 유형에서 48.1% 감소가 있음을 확인했습니다. 이는 L2 학습자의 자발적인 말하기 데이터에 특화된 모델의 필요성을 강조하는 결과입니다.



### BiosERC: Integrating Biography Speakers Supported by LLMs for ERC Tasks (https://arxiv.org/abs/2407.04279)
Comments:
          Accepted in the 33rd International Conference on Artificial Neural Networks (ICANN 2024)

- **What's New**: 이번 연구에서는 BiosERC라는 새로운 프레임워크를 제안하여 대화 중 화자의 특성을 조사합니다. 대형 언어 모델(LLMs)을 사용해 대화 중 화자의 '전기정보(biographical information)'를 추출하여 감정 라벨을 분류하는 데 사용됩니다. 이 방법은 IEMOCAP, MELD, EmoryNLP 등 3개의 유명 벤치마크 데이터셋에서 최첨단 성능(State-of-the-art, SOTA)을 기록했습니다. 이를 통해 모델의 효율성과 일반화를 증명했습니다. 소스 코드는 이 링크에서 확인할 수 있습니다.

- **Technical Details**: BiosERC는 기존의 GRU나 speaker-based masked attention 메커니즘 대신 LLMs와 프롬프트 기법을 사용해 대화 중 화자의 개별 성격을 정확하게 추출합니다. 이를 통해 각 화자의 감정 전환 사건을 세밀하게 이해하고 포괄할 수 있습니다.

- **Performance Highlights**: BiosERC는 IEMOCAP, MELD, EmoryNLP 등 3개의 벤치마크 데이터셋에서 SOTA 성능을 보였습니다. 또한, 바이오그래피 데이터 통합 방식이 복잡한 대화에서 감정 인식을 더 정확하고 포괄적으로 만듭니다. BiosERC의 프롬프트 기법은 다양한 대화 분석 작업에 적응할 가능성을 보여줍니다.



### Unified Interpretation of Smoothing Methods for Negative Sampling Loss Functions in Knowledge Graph Embedding (https://arxiv.org/abs/2407.04251)
Comments:
          9 pages, 4 figures, 2 tables; accepted to workshop RepL4NLP held in conjunction with ACL 2024

- **What's New**: 이번 논문에서는 Knowledge Graph Embedding(KGE)에서 사용되는 Negative Sampling(NS) 손실의 부드럽게 만드는 방법에 대한 이론적 해석을 제공하고, 기존의 부드러움 방법들의 특성을 포함할 수 있는 새로운 NS 손실인 Triplet Adaptive Negative Sampling(TANS)을 제안합니다. 이를 통해 KGE의 성능을 개선하고자 합니다.

- **Technical Details**: KGE는 대규모 엔티티와 관계를 학습하는 과정에서 Negative Sampling(NS) 손실을 사용합니다. 이 논문에서는 Self-Adversarial Negative Sampling(SANS)과 subsampling 같은 기존의 부드러움 방법들과 TANS의 이론적 차이를 분석하고, TANS가 이 두 가지 방법의 특성을 모두 포함함을 보입니다.

- **Performance Highlights**: 제안된 TANS는 TransE, DistMult, ComplEx, RotatE, HAKE, HousE 모델들을 사용하여 FB15k-237, WN18RR, YAGO3-10 데이터셋 및 그 하위 집합들에서 실험을 수행한 결과, MRR(Metric)을 기준으로 KGC 성능 향상을 검증했습니다. 또한, TANS는 기존의 SANS와 subsampling을 대체할 수 있는 잠재력을 보여주었습니다.



### ArAIEval Shared Task: Propagandistic Techniques Detection in Unimodal and Multimodal Arabic Conten (https://arxiv.org/abs/2407.04247)
Comments:
          propaganda, span detection, disinformation, misinformation, fake news, LLMs, GPT-4, multimodality, multimodal LLMs

- **What's New**: ArAIEval의 두 번째 판이 ArabicNLP 2024와 함께 개최되었습니다. 이번에는 두 가지 작업이 제공됩니다: (i) 트윗과 뉴스 기사에서 선전적 텍스트 스팬 및 설득 기법을 식별하는 작업과 (ii) 선전적 밈과 비선전적 밈을 구별하는 작업입니다. 총 14개 팀이 최종 평가 단계에 참여했으며, 6개 팀은 작업 1에, 9개 팀은 작업 2에 참여했습니다.

- **Technical Details**: 대부분의 참가 시스템은 AraBERT와 같은 Transformer 모델을 미세 조정(fine-tuning)하는 것을 중심으로 구성되었습니다. 데이터셋 구축과 평가 설정에 대한 설명과 함께 참가 시스템의 개요가 제공됩니다. 모든 데이터셋과 평가 스크립트는 연구 커뮤니티에 공개됩니다.

- **Performance Highlights**: 참가한 모든 시스템이 무작위 기초선(random baseline)을 능가하는 성과를 보여주었습니다. 다수의 참가 시스템이 데이터 증강(data augmentation) 기법을 적용하여 성능을 향상시켰습니다. 총 11개의 팀이 시스템 설명서를 제출했습니다.



### HAF-RM: A Hybrid Alignment Framework for Reward Model Training (https://arxiv.org/abs/2407.04185)
- **What's New**: 최근 큰 언어 모델(LLM)의 정렬, 평가 및 데이터 구축에 있어서 보상 모델(reward model)의 중요성이 점점 더 커지고 있습니다. 본 논문에서는 보상 점수 외에도 토큰 레벨의 정책 확률에 추가 제약을 도입하여 보상 모델 교육을 위한 하이브리드 정렬 프레임워크 HaF-RM을 제안합니다. 이 방법은 토큰 레벨에서 내부 선호 모델을 감독하고 시퀀스 레벨에서 보상 모델의 매핑 레이어를 최적화할 수 있습니다. 이론적 근거와 다섯 가지 데이터셋에 대한 실험 결과는 제안된 하이브리드 프레임워크의 유효성과 효과성을 보여줍니다.

- **Technical Details**: HaF-RM 프레임워크는 토큰 레벨의 정책 확률과 시퀀스 레벨의 보상 점수를 동시에 최적화하여 보상 모델의 두 구성 요소를 하이브리드로 최적화합니다. 보상 모델은 각 토큰에 대한 선호 벡터를 출력하는 트랜스포머 기반의 내부 선호 모델과 이러한 벡터를 시퀀스 레벨 보상으로 매핑하는 '보상 레이어'로 구성됩니다. HaF-RM은 정책 손실(policy loss)과 보상 손실(reward loss)을 공유 내부 선호 모델과 함께 사용하여 보상 모델을 최적화합니다.

- **Performance Highlights**: 실험 결과, 제안된 HaF-RM 프레임워크는 네 가지 공개 데이터셋에서 기존의 기준 접근방식과 비교하여 우수한 성능을 나타냅니다. 추가적인 정책 손실의 사용은 정책 모델의 보정(performance of policy model calibration)을 향상시켰으며, 이는 고품질 보상 모델 교육을 위한 새로운 가능성을 열어줍니다. HaF-RM 프레임워크는 보상 모델의 성능과 정렬을 개선하는 효과적인 접근법을 제공합니다.



### Seeing Like an AI: How LLMs Apply (and Misapply) Wikipedia Neutrality Norms (https://arxiv.org/abs/2407.04183)
- **What's New**: 대형 언어 모델(LLMs)이 광범위한 코퍼스로 훈련된 후 전문화된 규범이 있는 커뮤니티에서 사용됩니다. 이 논문은 LLM에게 특정 커뮤니티 규칙을 제공하는 것이 충분한가라는 질문을 다룹니다. 이를 위해 위키피디아의 중립적 관점(NPOV) 정책에 따라 LLM이 편향된 편집을 탐지하고 교정하는 능력을 평가했습니다.

- **Technical Details**: 탐지 작업에서는 ChatGPT 3.5, Mistral-Medium, GPT-4를 사용해 다양한 정책 상세와 예제 결정을 제공합니다. 생성 작업에서는 GPT-4를 사용해 NPOV 위반 편집을 중립적인 편집으로 변환했습니다. 탐지 실험에서 모델의 최고 정확도는 64%였으며, 생성 실험에서는 삭제 단어의 79%를 제거했습니다.

- **Performance Highlights**: LLMs는 탐지에서 64%의 정확도로 문제를 겪었지만, 생성에서 더 나은 성과를 보였습니다. 군중 작업자들은 AI 재작성된 문장을 70% 더 중립적이고 61% 더 유창하다고 평가했습니다. 그러나 LLM이 불필요한 추가 수정도 많이 했습니다. 이는 LLM이 높은 재현율(high-recall)과 낮은 정밀도(low-precision)를 가지고 있음을 시사합니다.



### Defense Against Syntactic Textual Backdoor Attacks with Token Substitution (https://arxiv.org/abs/2407.04179)
- **What's New**: 이 논문은 특수 토큰 기반(triggers) 뿐만 아니라 구문 기반(trigger) 백도어 공격을 효과적으로 방어하는 새로운 온라인 방어 알고리즘을 제안합니다. 이 알고리즘은 문장에서 의미 있는 단어를 완전히 다른 단어로 대체하되, 문법 템플릿이나 특수 토큰은 그대로 유지한 상태로 예측 레이블을 비교하여 문장이 트리거를 포함하는지 판단합니다.

- **Technical Details**: 구문 백도어 공격(syntax-based backdoor attacks)과 삽입 기반(backdoor attacks)을 대상으로 한 텍스트 백도어 방어 방법을 제안합니다. 방어 알고리즘은 다음 단계로 구성됩니다: (1) 문장에서 대체할 단어를 식별, (2) 대체할 적절한 단어를 선택, (3) 단어 대체 전후 예측 레이블을 비교, (4) 트리거를 포함하는 문장을 식별. 실험 결과, 이 알고리즘이 두 종류의 공격을 모두 방어하는 데 효과적임을 확인했습니다.

- **Performance Highlights**: 제안된 알고리즘은 삽입 기반 트리거에 대항하여 ONION과 비슷한 성능을 발휘하며, 기존의 방어 방법들이 잘 처리하지 못하는 구문 기반 트리거에 대해서도 높은 방어 효과를 보였습니다. 실험 결과는 제안된 방법이 모델 무결성을 유지하는 데 있어 포괄적인 방어 전략을 제공함을 보여줍니다.



### ELCC: the Emergent Language Corpus Collection (https://arxiv.org/abs/2407.04158)
Comments:
          18 pages, 3 figures

- **What's New**: Emergent Language Corpus Collection(ELCC)은 초기와 복잡화된 작업, 신호 전달 게임 환경 등을 포함한 다양한 emergent communication systems(ECSs)에서 수집된 코퍼스를 제공합니다. 이는 현재 연구자가 서로 다른 ECSs를 직접 실행해야 하는 어려움을 줄여주며, 깊은 학습 배경 없이도 연구할 수 있도록 도와줍니다.

- **Technical Details**: ELCC는 다양한 하이퍼파라미터 설정에 따라 생성된 각 ECS의 변종을 포함하고 있습니다. 메타데이터는 YAML 파일로 저장되며, 특정 ECS의 각 변종을 구분하는 하이퍼파라미터를 포함합니다. 각 코퍼스는 JSON Lines(JSONL) 형식으로 제공되어, 다재다능하고 사용이 간편합니다.

- **Performance Highlights**: ELCC는 구조적 특성이나 NLP 작업 사전 훈련 모델에 활용 가능성을 비교 연구할 수 있도록 지원합니다. 이 컬렉션은 연구자들이 emergent languages의 특성에 대한 분석 연구를 더욱 용이하게 할 것이며, 복잡한 코드 구현 없이도 다양한 ECSs에서의 데이터를 비교할 수 있습니다.



### Securing Multi-turn Conversational Language Models Against Distributed Backdoor Triggers (https://arxiv.org/abs/2407.04151)
Comments:
          Submitted to EMNLP 2024

- **What's New**: 이번 연구에서는 다중회차 대화형 대형 언어 모델(LLMs)의 분산 백도어 트리거 공격에 대해 새롭게 탐구했습니다. 백도어 트리거가 여러 발화에 걸쳐 분산되어 모델을 공격하는 방식으로, 기존의 단일회차 공격과는 다른 플러그 앤 플레이(plug and play) 접근 방식을 제안합니다.

- **Technical Details**: 연구에서는 PoisonShare라는 다중회차 백도어 트리거 공격 방식을 제안하며, 백도어 트리거를 분산시켜 더 은밀하고 강력한 공격을 가능하게 합니다. 공격자는 크라우드소싱이나 소셜 미디어에서 추출한 악의적인 발화를 통해 모델을 훈련시키며, 모델의 성능을 유지하면서도 비정상적인 반응을 유도합니다. 또한, 대비 디코딩(contrastive decoding) 기반 방어 메커니즘을 탐구해 저비용으로 백도어를 완화할 수 있는 방법을 제시합니다.

- **Performance Highlights**: 실험 결과, 백도어 공격 성공률은 단 5%의 데이터 포이즈닝으로도 100%에 근접하는 성과를 보여주었습니다. 대비 디코딩 기반 방어 전략은 백도어 공격을 89%에서 3%로 대폭 감소시키는 데 성공했습니다. 이러한 결과는 다중회차 대화 설정에서 기존 방어 전략의 한계를 확인하고, 더 효과적인 방어 전략이 필요함을 시사합니다.



### Towards Automating Text Annotation: A Case Study on Semantic Proximity Annotation using GPT-4 (https://arxiv.org/abs/2407.04130)
Comments:
          12 pages

- **What's New**: 이 논문은 GPT-3.5와 GPT-4를 사용하여 데이터 주석 작업을 자동화하는 방법을 탐구합니다. 특히, 인간 주석 지침과 일부 주석 데이터를 재사용하여 대형 언어 모델(LLM)을 위한 자동 프롬프트를 설계하는데 초점을 맞춥니다. 연구에서는 자동 프롬프트와 맞춤형 프롬프트를 비교했으며, OpenAI API를 통해 온라인으로 쉽게 사용할 수 있는 오픈 소스 텍스트 주석 도구에 프롬프트 전략을 구현했습니다.

- **Technical Details**: 데이터 주석은 머신러닝 모델 교육에서 중요한 역할을 합니다. 그러나 고품질의 데이터 주석은 복잡하고 비용이 많이 들며 시간이 소요됩니다. 최근 연구에서는 GPT-4, Gemini, Llama-2와 같은 고급 LLM이 데이터 주석을 혁신할 기회를 제공한다고 강조합니다. 이 연구에서는 인간 주석 지침과 일부 주석 데이터를 재사용하여 LLM을 위한 자동 프롬프트를 설계함으로써 주석 과정의 효율성을 높이고자 합니다. 주요 실험에는 PhiTag 오픈 소스 텍스트 주석 도구를 사용하여 자동 주석 기능을 구현했습니다.

- **Performance Highlights**: 연구 결과, 정확한 프롬프트 설계가 중요한 역할을 하며, GPT-4를 인간과 같은 지침으로 프롬프트하는 것이 간단하지 않음을 발견했습니다. 인간 지침에 작은 수정만으로 성능이 향상될 수 있음을 보여주며, 장래 연구에 대한 가능성을 시사합니다. 실험에서는 PhiTag 플랫폼을 통해 자동 주석 프로세스가 온라인 사용자 인터페이스에서 몇 번의 클릭만으로 가능하게 합니다.



### Query-Guided Self-Supervised Summarization of Nursing Notes (https://arxiv.org/abs/2407.04125)
- **What's New**: 새로운 연구 QGSumm은 간호 기록(nursing notes) 요약을 위해 참조 요약(reference summary)를 필요로 하지 않는 쿼리 기반(query-guided) 자기 지도(self-supervised) 도메인 적응 프레임워크를 소개합니다. 본 연구는 주로 환자의 건강 상태를 중심으로 고품질의 요약문을 생성하며, 이는 기존의 대규모 언어 모델(Large Language Models, LLMs)과 비교하여 뛰어난 성능을 보입니다.

- **Technical Details**: QGSumm은 사전 학습된 언어 모델의 요약 기능을 간호 기록 도메인에 적응시키는 학습 목표를 설정합니다. 이 프레임워크는 환자의 메타데이터와 동일한 입원 기록에서 이전 간호 기록 정보를 통합합니다. 쿼리를 사용하여 환자의 상태에 대한 질문에 대해 요약본과 원본 문서의 응답 간 차이를 최소화하도록 설계되었습니다.

- **Performance Highlights**: 자동 평가 및 전문가 클리니션의 수동 평가를 통해 QGSumm은 제로샷 및 퓨샷 설정에서 기존 LLMs보다 우수한 성능을 입증했습니다. 이 방법은 참조 요약을 요구하지 않으며, 간호 기록 요약에서 높은 품질의 요약문을 생성합니다.



### Hallucination Detection: Robustly Discerning Reliable Answers in Large Language Models (https://arxiv.org/abs/2407.04121)
Comments:
          Accepted to CIKM 2023 (Long Paper)

- **What's New**: 이번 논문에서는 큰 언어 모델(LLMs)이 생성하는 답변에서 발생하는 환각(hallucination)을 감지하기 위한 견고한 판별기인 RelD를 제안했습니다. RelD는 이중 언어 질문-응답 대화 데이터셋인 RelQA와 다양한 LLMs가 생성한 답변, 그리고 포괄적인 평가 지표를 통한 훈련을 통해 개발되었습니다.

- **Technical Details**: RelQA 데이터셋은 위키피디아, Baidu Zhidao, Bing 사용자 쿼리, 중국 고등학교 독해 자료 등 다양한 출처에서 수집한 274,426개의 샘플로 구성되어 있습니다. 여기에는 추출 독해(extractive reading comprehension)와 다지선다형 질문(multiple-choice questions) 방식 등이 포함되어 있습니다. RelD는 LLM-assessment metrics, human metrics, machine metrics, composite metrics 등 다양한 평가 기준을 사용하여 훈련되었습니다.

- **Performance Highlights**: 실험 결과, RelD는 다양한 LLM들이 생성한 답변에서 환각을 효과적으로 감지하는 데 성공하였으며, 인-분포(in-distribution)와 아웃-분포(out-of-distribution) 데이터셋 모두에서 우수한 성능을 보였습니다. 또한, 생성된 답변에서 어떤 유형의 환각이 발생하는지에 대한 철저한 분석을 통해 유의미한 통찰을 제공했습니다.



### MAPO: Boosting Large Language Model Performance with Model-Adaptive Prompt Optimization (https://arxiv.org/abs/2407.04118)
Comments:
          Accepted to EMNLP 2023 (Findings)

- **What's New**: 프롬프트 엔지니어링(Prompt engineering)은 대형 언어 모델(LLM)을 효과적으로 활용하기 위한 중요한 방법으로 주목받고 있습니다. 본 논문에서는 기존 연구가 특정 작업에 대해 프롬프트를 적응시키는 것에 중점을 둔 반면, 각기 다른 LLM에 맞춰 프롬프트를 최적화하는 필요성을 제시합니다. 이를 위해 각 LLM에 대해 프롬프트를 최적화할 수 있는 모델 적응 프롬프트 최적화(MAPO) 방법을 새롭게 제안하였습니다.

- **Technical Details**: MAPO는 처음에 오라클 LLM을 통해 후보 프롬프트를 생성하고, 강화 학습을 통해 최적의 프롬프트를 찾는 방식을 사용합니다. 먼저 기존 프롬프트를 다양한 표현으로 변형하여 후보 프롬프트를 생성합니다. 그 후, F1 점수, 정확도 및 ROUGE-L을 사용하여 각 작업에서 최적의 프롬프트를 찾습니다. 최적화된 프롬프트를 찾기 위해 감독된 미세 조정(SFT)과 강화 학습(RL)을 결합하며, 더 나아가 Proximal Policy Optimization(PPO) 및 RRMF를 사용하여 RL의 성능을 향상시킵니다.

- **Performance Highlights**: 세 가지 LLM(BLOOM-7B, GPT-J-6B, LLaMA-7B)을 통해 질문-답변(QA), 분류, 생성 작업에서 MAPO의 성능을 평가했습니다. 다양한 LLM 간에 분포의 유의미한 변화를 관찰할 수 있었으며, 특히 각 작업에서 최적의 프롬프트를 찾는 것이 LLM의 성능을 향상시키는 데 중요한 역할을 한다는 것을 확인했습니다. 실험 결과, MAPO는 높은 견고성과 일반화 능력을 보이며, 다양한 다운스트림 작업에서 우수한 성능을 나타냈습니다.



### Can Pre-trained Language Models Understand Chinese Humor? (https://arxiv.org/abs/2407.04105)
Comments:
          Accepted to WSDM 2022

- **What's New**: 이 논문은 처음으로 Pre-trained Language Model(PLM)의 유머 이해 능력을 체계적으로 조사합니다. 이를 위해 세 가지 평가 단계와 네 가지 평가 작업을 포함한 포괄적인 프레임워크를 설계하고, 이에 맞춘 포괄적인 중국어 유머 데이터를 구축하였습니다. 이는 현재까지의 연구와 차별화되는 중요한 진전입니다.

- **Technical Details**: 연구진은 PLM의 유머 이해 능력을 평가하기 위해 세 가지 평가 단계를 설계했습니다: (1) 원래 PLM 평가, (2) 지식 강화된 PLM 평가, (3) 유머 이해 해석. 네 가지 주요 평가 작업은 유머 인식(Humor Recognition), 유머 타입 분류(Humor Type Classification), 유머 레벨 분류(Humor Level Classification), 펀치라인 감지(Punchline Detection)입니다. 또한, 중국어 유머 연구를 위해 방대하고 포괄적인 중국어 유머 데이터셋을 구축하였습니다.

- **Performance Highlights**: 실험 결과, 유머 데이터세트에서 파인튜닝을 통해 PLM의 유머 이해 능력이 크게 향상되었습니다. 특히, 중국어 병음 정보와 같은 외부 지식이 유머 관련 작업 성능 향상에 긍정적인 영향을 미쳤습니다. 다만, 탐지된 단서 단어 중 일부가 인간의 유머 인식과 일치했으나, 여전히 PLM의 유머 이해 능력에는 많은 개선 여지가 있음이 확인되었습니다.



### Stephanie: Step-by-Step Dialogues for Mimicking Human Interactions in Social Conversations (https://arxiv.org/abs/2407.04093)
- **What's New**: 이번 논문은 자연어 처리 분야의 새로운 대화 시스템 패러다임인 **Step-by-Step Dialogue Paradigm (Stephanie)**를 소개합니다. 이 패러다임은 기존의 단일 응답 방식이 아닌, 사람의 자연스러운 대화를 모방하여 여러 단계에 걸쳐 대화를 진행하는 방식을 채택합니다. 논문은 Stephanie를 통해 성능을 극대화하기 위한 듀얼 학습 전략과 추가 분할 후 편집(Further-Split post-editing) 방법을 제안합니다.

- **Technical Details**: Stephanie는 기존 대형 언어 모델(large language models)을 미세 조정하기 위해 고품질의 단계별 대화 데이터셋을 생성하고 이를 활용합니다. 듀얼 학습 전략과 추가 분할 후 편집 방법을 통해 점진적이고 유기적인 대화를 생성하며, 이러한 데이터셋을 활용해 모델을 미세 조정합니다. 또한, 대화 시스템을 사람들이 실제로 대화하는 것처럼 단계별로 진행되도록 설계되었습니다. 이 새로운 패러다임은 기존의 단일 응답 방식과 차별화되며, 감정적 깊이와 유기적인 대화 흐름을 강조합니다.

- **Performance Highlights**: 논문은 자동 평가와 인간 평가를 통해 Stephanie의 효과를 입증합니다. 결과적으로, Stephanie는 기존의 단일 응답 대화 시스템에 비해 사용자 참여도와 자연스러운 대화 흐름에서 우수한 성과를 보여주었습니다. 코드, Stephanie 데이터셋 및 모델도 공개되어 향후 연구와 개발에 기여할 예정입니다.



### AXOLOTL'24 Shared Task on Multilingual Explainable Semantic Change Modeling (https://arxiv.org/abs/2407.04079)
Comments:
          Proceedings of the 5th Workshop on Computational Approaches to Historical Language Change (ACL'24)

- **What's New**: AXOLOTL'24라는 새로운 다국어 설명 가능한 의미 변화 모델링 대회가 소개되었습니다. 이 대회는 핀란드어와 러시아어의 새로운 의미로 주석된 시대적 의미 변화 데이터셋을 제공하며, 기존 소스에서 빌려온 독일어 서프라이즈 테스트 데이터셋도 포함됩니다. 참가 팀들은 새로운 의미를 식별하고 해당 의미에 대해 사전 정의를 제공하는 두 가지 하위 작업을 수행해야 했습니다.

- **Technical Details**: AXOLOTL'24는 설명 가능한 의미 변화 모델링의 첫 공식화 및 평가를 목표로 합니다. 참가자들은 핀란드어, 러시아어 및 독일어로 제공된 사용 예문을 통해 두 가지 시간대('old', 'new')에 걸쳐 목표 단어의 새로운 의미를 식별하고 인간 해석 가능한 정의를 제공해야 했습니다. 핀란드어와 러시아어 데이터셋은 트레이닝 및 개발 데이터로 구성되었으며, 독일어 데이터셋은 테스트 데이터로만 제공되었습니다.

- **Performance Highlights**: 참가한 6개 팀의 결과는 새로운 의미를 감지하거나 생소한 의미를 정의하는 문제를 해결하기까지 아직 많은 과제가 남아 있음을 보여줍니다. 하지만 이번 대회를 통해 다이어크로닉 의미 변화 모델링 시스템 개발에 더욱 견고한 기초가 마련되었습니다. 이 연구가 NLP와 역사 언어학 커뮤니티 간의 교류를 촉진하는 중요한 단계가 되기를 기대합니다.



### DotaMath: Decomposition of Thought with Code Assistance and Self-correction for Mathematical Reasoning (https://arxiv.org/abs/2407.04078)
Comments:
          Work in progress

- **What's New**: 이번 연구에서는 복잡한 수학적 문제 해결을 위해 Decomposition of thought with code assistance and self-correction 기법을 활용한 LLM 시리즈, 'DotaMath'를 소개하였습니다. DotaMath 모델은 문제를 더 단순한 논리적 하위 작업으로 나누고, 코드를 활용하여 이 하위 작업을 해결하며 코드 인터프리터의 피드백을 통해 스스로를 반성하고 수정합니다.

- **Technical Details**: DotaMath 모델 개발을 위해 다양한 상호작용 툴 사용 과정과 문제 쿼리 진화를 통해 총 574K의 쿼리-응답 페어로 구성된 DotaMathQA 데이터셋을 생성하였습니다. 이를 통해 모방 학습(imitation learning)을 기반으로 한 여러 LLM 모델을 훈련하였습니다.

- **Performance Highlights**: DotaMath 모델은 다양한 인-도메인 및 아웃-오브-도메인 벤치마크에서 우수한 성능을 보여주었습니다. 특히, DotaMath-deepseek-7B는 MATH 데이터셋에서 64.8%, GSM8K에서는 86.7%의 성능을 기록하였습니다. 또한, 인-도메인 및 아웃-오브-도메인 벤치마크에서 평균 80.1%의 성과를 유지하며 높은 경쟁력을 보였습니다.



### A Systematic Survey and Critical Review on Evaluating Large Language Models: Challenges, Limitations, and Recommendations (https://arxiv.org/abs/2407.04069)
- **What's New**: 최근 공개된 논문에서는 LLM(대규모 언어 모델) 평가의 복잡성과 일관성 문제를 체계적으로 분석하고, 평가의 재현성, 신뢰성, 견고성을 보장하기 위한 권장 사항을 제시하고 있습니다. 이러한 체계적인 리뷰는 현재 관행의 한계를 극복하고 LLM 평가의 일관성을 높이기 위한 중요한 노력을 의미합니다.

- **Technical Details**: LLM 평가에서 중요한 요소로는 평가 설정(Evaluation Setup), 응답 생성(Response Generation), 평가 방법론(Evaluation Methodology) 등이 있습니다. 평가 프로세스의 첫 단계는 적절한 벤치마크를 선택하는 것이며, 벤치마크 데이터셋은 일반적인 역량 벤치마크(general capability benchmarks), 특화된 벤치마크(specialized benchmarks), 기타 다양한 벤치마크(other diverse benchmarks)로 분류할 수 있습니다. 그런 다음 모델을 선택하고 프롬프트(prompt) 디자인 및 디코딩 파라미터 설정을 통해 응답을 생성합니다.

- **Performance Highlights**: 논문에서는 LLM 평가의 복잡성과 자원 소모가 크며, 이는 개발만큼이나 중요하다고 강조하고 있습니다. 채점 스크립트를 사용해 평가 메트릭 적용 전 목표 라벨(target labels)을 추출하고, 인간 평가(human evaluation)와 자동 평가(automatic evaluation), LLM을 평가자로 활용하는 방법이 논의되었습니다. 최종적으로, 논문은 재현성(reproducibility), 신뢰성(reliability), 견고성(robustness) 측면에서의 문제와 해결책을 포괄적으로 탐구합니다.



### Semantic Graphs for Syntactic Simplification: A Revisit from the Age of LLM (https://arxiv.org/abs/2407.04067)
Comments:
          Accepted at TextGraphs-17 @ ACL 2024

- **What's New**: 최근 논문에서는 의미 그래프(Symbolic Sentence Meaning Representations), 특히 AMR(Abstract Meaning Representation)를 활용한 문장 구조 단순화(syntactic simplification) 작업을 다루고 있습니다. 연구진이 제안한 AMRS^3는 최신 의미 그래프를 사용하여 간단하지만 효과적인 문장 단순화 방법을 제공하며, 비용, 해석 가능성, 일반화 측면에서 독특한 장점을 지니고 있습니다.

- **Technical Details**: AMRS^3는 복잡한 문장의 AMR 그래프를 여러 하위 그래프로 나누어 각 하위 그래프가 하나의 의미 단위가 되도록 하고, 이를 바탕으로 간단한 문장을 생성하는 방식입니다. 의미 그래프는 AMR과 같은 최신 의미 표현 방식을 반영하며, 이는 최근 개발된 트리 뱅크, 파싱, 텍스트 생성 및 크로스-링구얼 적응에서 주목받는 의미 표현입니다. 연구진은 또한 AMRCoC(AMR Chain-of-Code) 프롬프트를 제안하여, LLM이 AMR 그래프에서 명시적 상징적 추론을 수행하도록 유도하였습니다.

- **Performance Highlights**: 연구진이 구축한 새로운 복잡하고 자연스러운 데이터셋을 통해 평가한 결과, AMRS^3는 기존 시스템과 LLM과 비교하여 문법적 단순화와 의미 보존 측면에서 경쟁력 있는 성능을 보여주었습니다. 또한, LLM에 AMR 입력을 보조로 제공하는 경우 약간의 성능 향상이 있음을 발견했습니다. 이는 LLM이 AMR 그래프를 통해 복잡한 추론 작업을 수행할 수 있도록 유도할 때 유용하다는 점을 시사합니다.



### Deep Content Understanding Toward Entity and Aspect Target Sentiment Analysis on Foundation Models (https://arxiv.org/abs/2407.04050)
Comments:
          Proceedings of the 41 st International Conference on Machine Learning, Vienna, Austria. PMLR 235, 2024. Copyright 2024 by the author(s)

- **What's New**: 이번 연구는 새로운 ABSA(Aspect-Based Sentiment Analysis) 과제인 EASTE(Entity-Aspect Sentiment Triplet Extraction)를 소개합니다. EASTE는 기존의 TASD(Target-Aspect-Sentiment Detection)을 확장하여, 측면 범주(aspect categories)를 사전 정의된 엔티티(entities)와 측면(aspects)으로 나누어 복잡성을 더합니다. 이는 체인 형태의 감정을 명확히 드러내는 데 도움을 줍니다.

- **Technical Details**: EASTE 과제를 해결하기 위해 트랜스포머(Transformers) 기반 언어 모델을 사용하였습니다. BERT 아키텍처를 사용한 통합 손실(unified-loss) 접근법으로 토큰 분류(task using token classification)를 수행하였으며, Flan-T5, Flan-Ul2, Llama2, Llama3, Mixtral과 같은 텍스트 생성 모델(text generative models)도 활용하였습니다. 이 모델들은 제로샷 학습(zero-shot learning) 및 소수 샘플 학습(few-shot learning)과 같은 다양한 정렬 기법(alignment techniques), PEFT(Parameter Efficient Fine-Tuning) 기법인 LoRA(Low-Rank Adaptation)를 이용해 평가하였습니다.

- **Performance Highlights**: SamEval-2016 벤치마크 데이터셋을 사용하여 모델 성능을 평가하였으며, 기존 작업과 공정하게 비교하였습니다. EASTE 과제에서 높은 성능을 달성하는 것뿐만 아니라, 모델 크기, 유형 및 적응 기법이 작업 성능에 미치는 영향을 조사했습니다. 결과적으로, 복잡한 감정 분석에서 최첨단(State-of-the-Art, SoTA) 결과를 얻었습니다.



### Improving Accented Speech Recognition using Data Augmentation based on Unsupervised Text-to-Speech Synthesis (https://arxiv.org/abs/2407.04047)
Comments:
          Accepted to EUSIPCO 2024

- **What's New**: 이 논문은 자가 교정 텍스트-음성 변환(TTS)을 사용하여 방언 인식 성능을 향상시키는 데이터 증강 방식에 대해 연구합니다. 적은 양의 방언 음성 데이터를 이용해 TTS 시스템을 훈련하고 인공 레이블을 생성하여 데이터 증강을 수행합니다. 이를 통해 수작업으로 전사된 데이터 없이도 방언 음성 데이터를 사용할 수 있습니다. 이로써 비방언 음성 데이터와 결합하여 음성 인식(ASR) 시스템을 훈련할 수 있습니다.

- **Technical Details**: 연구는 Wav2vec2.0 모델을 사용해 자가 교정 학습 프레임워크 내에서 진행되었습니다. L2-ARCTIC 및 British Isles 코퍼스의 낭독된 방언 데이터를 사용하여 자가 교정 TTS 시스템을 훈련하였고 인공적인 방언 음성을 생성했습니다. 이를 통해 EdAcc 코퍼스의 자발적인 회화 데이터를 평가했습니다. TTS 모델로는 VITS(Variational Inference with adversarial learning for end-to-end Text-to-Speech)를 사용했으며, HiFi-GAN을 디코더로 사용하여 고품질 음성을 합성했습니다.

- **Performance Highlights**: 연구 결과에 따르면, 자가 교정 TTS로 생성된 인공 방언 음성 데이터를 사용한 Wav2vec2.0 모델이 기반 모델 대비 최대 6.1%의 상대적 단어 오류율(WER) 감소를 보여주었습니다. 이는 비방언 음성 데이터만 사용한 모델에 비해 유의미한 성능 향상을 나타냅니다.



### Systematic Task Exploration with LLMs: A Study in Citation Text Generation (https://arxiv.org/abs/2407.04046)
Comments:
          Accepted to ACL 2024 (Main)

- **What's New**: 대규모 언어 모델(LLMs)의 창의적 자연 언어 생성(NLG) 작업을 탐구하기 위한 새로운 연구 프레임워크를 제안합니다. 이 프레임워크는 체계적인 입력 조작, 참조 데이터, 출력 측정을 포함합니다. 인용 텍스트 생성 작업에 대해 LLM 패러다임에서 체계적으로 탐구하지 않은 문제를 해결하고자 합니다.

- **Technical Details**: 프레임워크는 모델에 전달되는 프롬프트의 입력 요소와 지시사항을 체계적으로 조작하여 출력에 미치는 영향을広角 측정합니다. 이 프레임워크를 사용하여 최신 LLM인 Llama 2-Chat 및 GPT 3.5 Turbo로 실험하였으며, ACL Anthology 기반의 새로운 참조 데이터셋을 도입하여 자유형 인용 의도를 활용한 생성 방법을 제시합니다.

- **Performance Highlights**: 실험 결과, 입력 요소와 과제 지시사항 모두 인용 텍스트 생성에 중요하며, 그 효과가 합산된다는 것이 밝혀졌습니다. 자유형 인용 의도는 기존의 범주형 의도보다 더 유망한 대안으로 보입니다. 또한 다양한 평가 메트릭스 간의 상관관계를 통해 창의적 NLG 작업에 대한 포괄적인 측정 세트 사용의 필요성을 강조했습니다. 마지막으로, 인간 평가 실험을 통해 입력 요소와 작업 지시사항에 대한 정량적 및 정성적 인사이트를 제공했습니다.



### LLMAEL: Large Language Models are Good Context Augmenters for Entity Linking (https://arxiv.org/abs/2407.04020)
- **What's New**: LLM-Augmented Entity Linking (LLMAEL)은 기존의 EL 모델을 촉진하여 LLM 기반 데이터 확대를 통해 엔티티 연결(entity linking)을 개선하는 새로운 접근 방법입니다. LLM은 토큰의 맥락을 더 잘 해석할 수 있고, 이를 이용해 추가적인 배경 설명을 생성하여 기존 EL 모델에 제공하는 방식을 채택했습니다.

- **Technical Details**: LLMAEL은 다음 세 가지 주요 단계를 포함합니다: (1) 문맥 확장(context augmentation)으로, LLM이 원래 언급된 맥락 쌍을 보충 설명으로 확장합니다. (2) 데이터 융합(data fusion)으로, LLM이 확장한 문맥을 선택된 EL 모델에 통합합니다. (3) EL 수행(EL execution)으로, EL 모델을 사용하여 타겟 엔티티를 검색합니다. 이를 통해 LLM의 광범위한 세계 지식과 텍스트 생성 능력, 그리고 EL 모델의 지식 베이스(KB) 상호작용 능력을 결합합니다.

- **Performance Highlights**: 6개의 표준 데이터셋에서 실험 결과, 기본 LLMAEL은 대부분의 경우 기존 EL 모델을 능가하며, 미세 조정된 LLMAEL은 모든 6개의 벤치마크에서 새로운 상태의 기술(state-of-the-art) 결과를 달성했습니다. 평균적으로 1.21%의 정확도 향상을 기록하며, 선택적 기술 사용(예: 문맥 결합 및 앙상블)으로 성능을 더욱 향상시켰습니다.



### Exploring Diachronic and Diatopic Changes in Dialect Continua: Tasks, Datasets and Challenges (https://arxiv.org/abs/2407.04010)
Comments:
          LChange24 Camera Ready

- **What's New**: 최근 언어 변화와 방언(NLP) 연구에서 발생하는 시간적(diachronic) 및 공간적(diatopic) 변화에 대한 연구가 증가하고 있지만, 두 측면을 동시에 탐구한 연구는 부족한 상황입니다. 이번 연구에서는 이 공백을 메우기 위해 세 언어 계통(슬라브어, 로망스어, 게르만어)의 다섯 가지 방언에 대해 아홉 가지 NLP 작업과 데이터셋을 체계적으로 검토했습니다.

- **Technical Details**: 본 연구에서는 구어 및 문어 형태의 데이터를 다루며, 주요 기술적인 키워드로는 코퍼스 구축(corpus construction), 방언 거리 추정(dialect distance estimation), 방언 지리적 위치 예측(dialect geolocation prediction) 등을 포함합니다. 리뷰를 통해 라파 방식(diasystem)의 다섯 가지 차원에서 방언 변화를 다루었습니다. 근래에는 컴퓨터 매개 의사소통과 소셜 미디어 덕분에 언어 변화 속도가 급격히 증가하고 있으며, NLP 모델이 이러한 변화에 적응하는 것은 주요 과제로 남아 있습니다.

- **Performance Highlights**: 다양한 언어 계통과 방언에 대한 분석을 통해 특정 기간과 지리적 영역에서 방언 데이터셋의 특성을 평가하고, 이를 바탕으로 NLP 작업의 현재 상태와 성과를 조사했습니다. 이로써 방언 NLP 연구에 있어 해결되지 않은 문제들을 다섯 가지 도전 과제(방언 데이터셋의 신뢰성, 화자 특성의 중요성, 방언의 제한된 범위, 데이터 수집의 윤리적 고려사항 등)로 도출했습니다. 이러한 연구가 향후 언어 변종과 방언에 대한 포괄적인 컴퓨팅 방법과 데이터셋 개발로 이어지기를 바랍니다.



### Unlocking the Potential of Model Merging for Low-Resource Languages (https://arxiv.org/abs/2407.03994)
- **What's New**: 새로운 연구에서는 대형 언어 모델(LLMs)을 저자원 언어로 적응시키기 위해 모델 병합(Model Merging)을 제안합니다. 이는 추가적인 훈련 없이 서로 다른 능력을 가진 모델들을 단일 모델로 통합하는 방법입니다.

- **Technical Details**: 전통적인 방법인 지속적 사전 학습(CT)과 지도 학습 최적화(SFT)를 사용하지 않고 모델 병합을 통해 저자원 언어에 대해 과업 해결 능력을 도입합니다. Llama-2-7B 모델을 사용하여 실험을 진행하였으며, 모델 병합이 매우 제한된 데이터 환경에서도 효과적임을 보여주었습니다.

- **Performance Highlights**: 모델 병합은 특히 10억 토큰 미만의 매우 적은 데이터로 훈련된 경우, CT-과-SFT 방법을 능가했습니다. 또한 병합 과정에서 중요한 파라미터 손실을 줄이기 위해 느슨한 변수(slack variable)를 도입하여 성능을 향상시켰습니다.

- **Related Works**: 기존 작업들은 주로 고자원 언어와 특정 과업에 대해 모델 병합을 탐구해왔습니다. 저자원 언어를 대상으로 한 모델 병합 연구는 이번 논문이 처음입니다. 기존의 지속적 사전 학습 및 SFT 방법은 높은 비용과 데이터 질 문제 등 한계가 있습니다.

- **Model Merging for Low-Resource Languages**: 저자들은 두 가지 모델 병합 방법—가중치 평균화(weighted averaging)와 TIES—을 조사했습니다. 가중치 평균화는 두 모델의 파라미터를 검증 세트에서 조정된 가중치와 함께 평균화하는 방법이고, TIES는 여러 모델 간의 파라미터 충돌을 더 세밀하게 처리하는 방법입니다.



### A Survey on Natural Language Counterfactual Generation (https://arxiv.org/abs/2407.03993)
Comments:
          A survey paper

- **What's New**: 이번 설문조사는 자연어 Counterfactual (대안사실적) 생성 방법에 대한 포괄적인 개요를 제공하며, 특히 Large Language Models (LLMs)에 기반한 방법들을 포함한 점이 새롭습니다. 이 설문조사는 다양한 대안사실적 생성 방법을 네 가지 그룹으로 구분하는 새로운 분류 체계를 제안하며, 생성 품질을 평가하기 위한 메트릭을 체계적으로 요약합니다. 마지막으로, 현재 연구 과제를 논의하고 미래 연구 방향을 제시합니다.

- **Technical Details**: 이 설문조사는 대안사실적 예제(CFEs)를 생성하는 방법을 네 가지 그룹으로 분류합니다: (1) 수동 생성 (Manual generation), (2) Gradient-based optimization, (3) Identify and then generate, (4) LLMs as counterfactual generators. 수동 생성 방식은 인간 주석자가 텍스트의 일부 단어를 편집해 클래스를 변경하는 방법을 사용합니다. Gradient-based optimization은 입력 문장 인코딩과 원하는 목표를 기반으로 컨트롤된 텍스트 생성 모델을 미세 조정합니다. Identify and then generate 방식은 단어를 찾아서 레이블을 변경하며, LLMs는 직접적으로 대안사실적 예제를 생성합니다. 또한 기존 메트릭에 기반하여 CFE의 질을 평가하는 다양한 방법들을 요약합니다.

- **Performance Highlights**: 본 연구는 텍스트의 원래 인스턴스에서 최소한의 변경으로 대안사실적을 생성하여 모델 예측의 이유를 파악하고, 소수 그룹 내에서 모델 공정성을 검출하며, 훈련 데이터를 보강하여 모델의 강인성을 향상시키는 데 목적이 있습니다. 이 연구는 특히 LLMs의 사용이 급증하면서 이들 모델의 투명성, 설명가능성, 공정성 및 강인성 향상에 대한 중요성이 강조되고 있음을 지적합니다. 대안사실적 생성 방법의 평가는 다양한 관점에서 이루어지며, 각각의 메트릭이 서로 다른 평가 기준을 가지기 때문에 공정한 비교는 어려운 과제로 남아있습니다.

- **Challenges and Future Directions**: 대안사실적 생성 방법의 공정한 평가의 어려움, 모델 프라이버시와 보안, 다양성 있는 CFEs 생성의 필요성, LLMs의 활용에서 생기는 과제 등이 현재 연구의 주요 도전 과제로 남아있습니다. 다양한 대안사실적을 생성하는 연구는 모델의 해석 및 공정성 검출, 강인한 모델 훈련 등의 다양한 측면에서 필수적입니다. 이러한 문제들을 해결하기 위해 카드널리티 제약을 통합한 방식이나 다양한 CFEs를 선택하는 방법 등이 제안됩니다.



### Benchmarking Complex Instruction-Following with Multiple Constraints Composition (https://arxiv.org/abs/2407.03978)
Comments:
          20 pages, 7 figures

- **What's New**: 이번 연구에서는 복잡한 지시(instruction)를 따르는 대형 언어 모델(LLM)의 능력을 평가하기 위한 새로운 벤치마크 'ComplexBench'를 제안합니다. 기존 벤치마크에서는 개별 제약 조건을 평가하는 데 집중했지만, 여러 제약 조건이 결합된 복합 지시의 평가에는 한계가 있었습니다. ComplexBench는 다양한 제약 조건과 조합 유형을 고려하여 이러한 평가의 공백을 메우고 LLM의 복잡한 지시를 따르는 능력을 종합적으로 평가할 수 있습니다.

- **Technical Details**: ComplexBench는 4개의 제약 유형(예: Lexical, Format, Semantic, Utility), 19개의 제약 차원, 4개의 조합 유형(Single, And, Chain, Selection)을 포함하는 계층적 분류 체계를 기반으로 구축되었습니다. 이 벤치마크에는 고품질의 데이터셋이 수작업으로 수집되었으며, 복잡한 지시가 제대로 이행되었는지 확인하기 위해 yes/no 질문 방식을 채택한 평가 방법이 포함되어 있습니다. 최종 평가 점수는 조합 유형에 따라 결정되는 종속 구조를 기반으로 계산됩니다.

- **Performance Highlights**: ComplexBench는 현재 사용 중인 여러 LLM들이 복잡한 지시에 포함된 다양한 제약 조건을 효과적으로 다루지 못한다는 중요한 결핍을 밝혀냈습니다. 이 연구는 LLM이 복잡한 지시를 처리할 때의 성능 향상 방안을 제시함으로써, LLM의 실제 사용 가능성을 향상시키는 데 기여할 수 있습니다.



### LLM Roleplay: Simulating Human-Chatbot Interaction (https://arxiv.org/abs/2407.03974)
- **What's New**: 이번 논문에서는 대화형 AI 에이전트(챗봇)을 개발하는 데 필요한 광범위한 사용자 대화를 자동으로 생성하는 새로운 기법인 LLM-Roleplay를 제안합니다. 이는 대형 언어 모델(LLM)을 사용하여 목표 지향적이고 페르소나 기반으로 다양한 멀티턴 대화를 생성할 수 있게 해줍니다. 이를 통해 다양한 소시오데모그래픽(배경) 그룹의 사용자를 반영할 수 있습니다.

- **Technical Details**: LLM-Roleplay는 텍스트로 묘사된 페르소나를 맡는 LLM을 활용하여 대화를 생성합니다. 이 방법은 기존의 데이터셋이 가진 한계를 극복하기 위해 설계되었으며, 사용자 평가를 통해 실제 인간-챗봇 대화와 비교하여 높은 유사성을 지닙니다. 구체적으로, 연구자는 여러 소시오데모그래픽 배경의 그룹으로부터 실제 인간-챗봇 대화를 수집하고, LLM-Roleplay를 통해 동일한 페르소나와 목표를 사용하여 대화를 시뮬레이션합니다.

- **Performance Highlights**: 연구 결과, LLM-Roleplay 방법은 실제 인간-챗봇 대화와 높은 유사성을 보여주었습니다. 특히, 다양한 LLM을 비교 분석한 결과, 각 모델이 페르소나를 구현하고 대화를 유지하는 능력에서 우수한 성과를 보였습니다. 그리고 Turing 테스트와 유사한 평가를 통해 참가자들이 실제 대화와 시뮬레이션된 대화를 구분하기 어렵다는 점을 확인했습니다.



### Investigating the Role of Instruction Variety and Task Difficulty in Robotic Manipulation Tasks (https://arxiv.org/abs/2407.03967)
- **What's New**: 이 연구는 다중모달(multimodal) 모델의 일반화 능력을 평가하는 포괄적인 프레임워크를 도입합니다. 이 프레임워크는 아키텍쳐 디자인, 입력 변형, 과제 복잡성 증가 등 다양한 요소를 체계적으로 검사하여 모델의 진정한 견고성을 평가합니다. 특히, 극단적인 지침 변형에 대한 다중모달 모델의 회복력과 관찰 변화에 대한 취약성을 밝혀내며, 스퓨리어스(spurious) 상관관계에 과적합(overfitting)될 수 있다는 문제를 제기합니다.

- **Technical Details**: 이 프레임워크는 언어와 비전(vision) 모달리티 전반에 걸친 입력 변형, 그리고 과제 복잡성 증가를 고려하여 다중모달 모델의 일반화 능력을 체계적으로 평가합니다. 평가에는 언어 지침의 다양한 변형, 전체 모달리티의 마스킹, 객체 순서의 시각적 변형 등이 포함됩니다. 또한, VIMABench를 사용하여 이러한 평가를 수행하며, 모델의 성능에 미치는 다중모달 프롬프트(prompt)와 시각적 표현의 영향을 광범위하게 연구합니다.

- **Performance Highlights**: 연구 결과, 현재 다중모달 모델들은 언어 변형에 대해 둔감하며, 시각적인 방해 요소가 증가할 경우 과제 수행에 어려움을 겪는다는 것이 밝혀졌습니다. 이러한 발견은 모델의 진정한 일반화 능력을 평가하기 위한 체계적인 접근의 필요성을 강조하며, 향후 연구 방향으로는 다중모달 입력을 더 잘 통합하는 아키텍쳐와 학습 방식의 혁신이 제시됐습니다.



### Improving Sample Efficiency of Reinforcement Learning with Background Knowledge from Large Language Models (https://arxiv.org/abs/2407.03964)
- **What's New**: 강화 학습(RL)의 낮은 샘플 효율성 문제를 해결하기 위해, 대형 언어 모델(LLMs)을 활용하여 배경 지식을 추출하는 새로운 프레임워크를 제안합니다. 이 프레임워크는 환경의 일반적인 이해를 기반으로 하여 다양한 RL 작업에서 이점을 제공합니다. 이는 특정 작업에 맞춘 지침이 아닌 일반적인 배경 지식으로, 한번의 지식 표현으로 여러 작업에 활용 가능합니다.

- **Technical Details**: 우리는 LLMs에게 사전에 수집된 몇 가지 경험을 제공하여 환경의 배경 지식을 정의하도록 합니다. 이렇게 얻어진 지식은 잠재적 기반 보상 형성(potential-based reward shaping)으로 표현되어 작업 보상 정책의 최적성을 유지합니다. 이를 위한 세가지 변형을 구현하였으며, 코드 작성, 선호도 주석 및 목표 설정을 포함합니다.

- **Performance Highlights**: Minigrid와 Crafter 도메인에서의 실험 결과, 제안된 방법이 다양한 후속 작업에서 샘플 효율성을 크게 향상시키는 것을 확인했습니다. 또한, 추출된 배경 지식이 새로운 작업 유형이나 작업 규모 증가에도 잘 일반화될 수 있는 가능성을 발견했습니다.



### LLM-jp: A Cross-organizational Project for the Research and Development of Fully Open Japanese LLMs (https://arxiv.org/abs/2407.03963)
- **What's New**: 이번 논문에서는 일본 대형 언어 모델(LLMs) 연구 및 개발을 위한 LLM-jp 프로젝트를 소개합니다. LLM-jp는 오픈 소스의 강력한 일본어 LLM을 개발하는 것을 목표로 하며 현재 학계와 산업계의 1,500명 이상의 참가자가 협력하고 있습니다. 현재까지의 활동과 기술 보고서를 통해 구체적인 과정을 공개하고 있습니다.

- **Technical Details**: LLM-jp는 2023년 5월에 시작되어, 세 가지 주요 작업 그룹(WG)을 통해 작업을 분담하고 있습니다: 코퍼스 구축 WG, 모델 구축 WG, 그리고 미세조정 및 평가 WG. 초기 모델인 LLM-jp 모델 슈트 v1.0는 2023년 10월 20일에 출시되었으며, 후속 모델 슈트인 v2.0은 2024년 4월 30일에 출시되었습니다. 각 모델 슈트는 130억 개의 파라미터를 가지고 있습니다.

- **Performance Highlights**: LLM-jp 모델 슈트 v1.0은 130억 개의 파라미터를 갖추고 있으며, 다양한 일본어, 영어, 코드 코퍼스로 사전 학습되었습니다. v2.0 모델에는 더 엄격한 필터링 방법이 적용되어 고품질의 270억 개의 토큰으로 구성된 코퍼스가 사용되었습니다. 이러한 모델 스위트는 모두 오픈 소스로 공개되어 있으며 연구 및 상업적 용도로 활용될 수 있습니다.



### Stark: Social Long-Term Multi-Modal Conversation with Persona Commonsense Knowledg (https://arxiv.org/abs/2407.03958)
Comments:
          Project website: this https URL

- **What's New**: 이 논문에서는 개인적인 경험과 관련된 이미지를 대화에서 공유하는 인간의 행동을 다루고 있습니다. 

- **기존 연구들은 단기적인 세션에서의 이미지 공유 행동만을 다뤘고, **: 개인화된 이미지 공유 행동이 부족하다는 한계가 있었습니다. 

- **Stark라는 대규모 장기 다중 모달 대화 데이터셋을 소개합니다. **: 이 데이터셋은 다양한 소셜 페르소나와 다중 모달 형식, 시간 간격 및 이미지를 포함합니다.

- **Technical Details**: Stark를 자동으로 구축하기 위해, 

- **새로운 다중 모달 컨텍스트화 프레임워크인 Mcu를 제안합니다. **: 이 프레임워크는 ChatGPT와 제안된 Plan-and-Execute 이미지 정렬기를 사용하여 장기적인 다중 모달 대화를 생성합니다.

- **이를 바탕으로 학습된 다중 모달 대화 모델인 Ultron 7B는 인상적인 시각적 상상력을 보여줍니다.**: 우리의 데이터셋의 효과를 인간 평가에서도 확인했습니다.

- **Performance Highlights**: 소스 코드와 데이터셋은 공개됩니다.



### Meta-prompting Optimized Retrieval-augmented Generation (https://arxiv.org/abs/2407.03955)
- **What's New**: 이번 연구에서는 외부 소스에서 검색된 콘텐츠를 정제하여 대규모 언어 모델(LLM)의 성능을 향상시키기 위해 메타-프롬프팅 최적화(meta-prompting optimization)를 사용한 새로운 방법을 제안합니다. 이 방법은 특히 다단계 질문 응답(multi-hop question answering) 과제에서 기존 방식보다 성능을 30% 이상 향상시킵니다.

- **Technical Details**: 제안된 방법은 외부 콘텐츠를 검색한 후, 이를 바로 LLM에 입력하기 전에 중간 단계로 '변환(Transformation)-LLM'을 사용하여 내용을 정제합니다. 이 변환 단계는 최적화된 프롬프트를 통해 실행됩니다. 최적화 과정에서는 '메타-프롬프트(meta-prompt)'가 사용되며, 이는 최적화 과제에 대한 설명과 이전 최적 솔루션 기록을 포함합니다. 이를 통해 LLM을 반복적으로 최적화하여 검색된 콘텐츠에 대한 최적의 정제 지시문을 생성합니다.

- **Performance Highlights**: 이 방법은 StrategyQA 데이터셋을 사용한 다단계 질문 응답 과제에서 기존의 검색-증강 생성(RAG) 시스템보다 성능을 30% 이상 향상시키는 것으로 나타났습니다. 이는 검색된 콘텐츠를 효과적으로 정제하여 생성-LLM의 응답 품질을 높이기 때문입니다.



### A framework for annotating and modelling intentions behind metaphor us (https://arxiv.org/abs/2407.03952)
- **What's New**: 새로운 논문은 은유(Metaphor) 사용 의도에 대한 종합적인 분류체계(taxonomy)를 제안하고, 이를 통해 최초로 주석된 데이터셋을 공개했습니다. 이를 통해 거대 언어 모델(LLM)이 은유 사용 의도를 추론할 수 있는 능력을 테스트했습니다.

- **Technical Details**: 논문에서 제안한 새로운 분류체계는 9개의 카테고리로 구성되어 있으며, 다양한 문헌을 기반으로 체계화되었습니다. 이 데이터셋을 사용해 GPT-4 터보와 두 가지 Llama2-Chat 모델(13B 및 70B 버전)에서 은유 사용 의도를 추론하는 실험을 수행했습니다. 이는 제로샷(zero-shot) 및 인-컨텍스트 퓨샷(few-shot) 설정에서 실험되었습니다.

- **Performance Highlights**: 실험 결과, GPT-4 모델은 제로샷 설정에서 평균 정확도 42.99%, 5-샷(five-shot) 설정에서 약간 더 높은 44.68%의 정확도를 기록했습니다. 이는 현재의 최첨단 언어 모델들에게도 은유 사용 의도 추론이 여전히 도전적인 과제임을 보여줍니다.



### TongGu: Mastering Classical Chinese Understanding with Knowledge-Grounded Large Language Models (https://arxiv.org/abs/2407.03937)
- **What's New**: TongGu는 최초의 고전 중국어 이해(CCU) 특화 대형 언어 모델(LLM)로, 고전 중국어의 복잡성을 극복하기 위해 개발되었습니다. 이 모델은 현재와 고대의 지식을 동시에 이해하도록 설계되었으며, 세 가지 주요 기여를 바탕으로 만들어졌습니다: ACCN-INS 데이터셋, Redundancy-Aware Tuning(RAT) 방법, 그리고 CCU Retrieval-Augmented Generation(CCU-RAG) 기술이 그것입니다. 이 모델과 데이터셋은 공개될 예정입니다.

- **Technical Details**: ACCN-INS는 풍부한 고전 중국어 코퍼스를 기반으로 구축된 두 단계의 지시 조정 데이터셋입니다. 첫 번째 단계에서는 대규모 데이터로부터 고전 중국어를 현대 중국어로 번역하는 작업을 위한 파인 튜닝을 수행하고, 두 번째 단계에서는 소규모 데이터로 구두점 같은 작업을 위한 파인 튜닝을 진행합니다. Catastrophic forgetting(치명적 망각)을 방지하기 위해 RAT를 제안하는데, 이 방법은 중요한 레이어를 식별하고 동결하는 방식으로 새로운 기능을 모델에 주입하면서 기존 지식을 유지합니다. CCU-RAG 기법은 지식 기반의 정확성을 높여 착각(hallucination)을 줄이는 데 도움을 줍니다.

- **Performance Highlights**: 24가지 다양한 CCU 작업에서 광범위한 실험을 통해 TongGu의 뛰어난 성능이 입증되었습니다. 특히, RAT와 CCU-RAG의 효용성이 강조되었습니다.



### Entity-Level Sentiment: More than the Sum of Its Parts (https://arxiv.org/abs/2407.03916)
Comments:
          14th Workshop on Computational Approaches to Subjectivity, Sentiment & Social Media Analysis (WASSA 2024)

- **What's New**: 이번 연구는 길고 복잡한 텍스트 내에서 각 개체(entity)에 대한 감정을 모델링하는 방법을 탐구합니다. 이를 위해 전문가가 주석을 단 데이터셋을 수집하였으며, 각 개체에 대한 문장 수준과 전체 문서 수준의 감정을 확인했습니다. 데이터셋은 개체별 감정 분류를 정확히 모델링하고 평가할 수 있도록 도와줍니다.

- **Technical Details**: 연구팀은 노르웨이 리뷰 코퍼스(NoReC)의 일부 문서에 대해 감정 주석을 추가로 달았습니다. 이를 위해 이름 인식(Named Entity Recognition, NER) 모델을 사용하여 사람(PER)과 조직(ORG) 라벨을 인식했고, 이를 통해 각 문서 내 여러 이름 언급을 클러스터링했습니다. 또한 문서 레벨과 문장 내의 감정을 따로 주석 달아 복잡한 감정 신호를 세밀하게 반영할 수 있도록 했습니다.

- **Performance Highlights**: 기본 모델을 사용한 전반적인 감정 예측의 F1 점수는 시퀀스 라벨링을 사용한 경우 56%, zero-shot LLM-prompting을 사용한 경우 69%로 나타나, 이 작업의 복잡성을 강조하였습니다.



### Scoping Review of Active Learning Strategies and their Evaluation Environments for Entity Recognition Tasks (https://arxiv.org/abs/2407.03895)
Comments:
          The Version of Record of this contribution is published in Deep Learning Theory and Applications 5th International Conference, DeLTA 2024 Proceedings, and will be available after the conference

- **What's New**: 본 연구는 자연어 처리(NLP) 분야에서 활발히 연구되고 있는 액티브 러닝(Active Learning, AL) 전략을 엔티티 인식(Entity Recognition, ER)에 적용한 최신 접근 방식을 탐색한 리뷰입니다. Scopus와 ACM을 사용해 검색을 진행했고, 62편의 관련 논문을 분석하여 106개의 AL 전략을 도출했습니다. 이 리뷰는 다양한 데이터셋, 평가 메트릭(F1-score), 하드웨어 및 실행 시간 측면에서 이들 전략이 어떻게 평가되었는지를 분석합니다.

- **Technical Details**: 본 연구에서는 ER 작업을 위해 AL 전략을 크게 세 카테고리로 분류했습니다: 1) 활용 기반 전략(Exploitation-based) (60x), 2) 탐색 기반 전략(Exploration-based) (14x), 그리고 3) 하이브리드 전략(Hybrid strategies) (32x). 대부분의 연구들이 F1-score를 평가 메트릭으로 사용했으며, 하드웨어 정보와 실행 시간은 부분적으로만 명시되었습니다. 실험에 사용된 57개의 데이터셋 중 26개는 공개적으로 접근 가능합니다. 논문 준비 과정에서는 PRISMA-ScR 가이드라인을 준수했습니다.

- **Performance Highlights**: 62편의 논문에서 각각의 AL 전략들이 다양한 환경에서 어떻게 성능을 보였는지 종합한 결과, 다음과 같은 핵심 발견을 제시합니다. 모든 연구에서는 F1-score를 공통적으로 사용해 성능을 평가했습니다. 하드웨어 정보는 6건의 논문에서만, 실행 시간은 13건의 논문에서만 제공되었습니다. 전체 데이터셋 중 대부분은 신문 기사 또는 생의학/의료 데이터로 구성되어 있었으며, 26개 데이터셋은 공개적으로 접근 가능했습니다. 이러한 자료들을 기반으로 한 종합적이고 체계적인 비교 실험이 향후 AL 전략 선택에 도움을 줄 것으로 기대됩니다.



### Planning with Large Language Models for Conversational Agents (https://arxiv.org/abs/2407.03884)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)을 활용한 새로운 대화 에이전트 프레임워크인 PCA(Planning-based Conversational Agents)를 제안합니다. PCA는 표준 운영 절차(SOPs)를 따르는 통제 가능성과 대화 목표를 향한 주도성을 동시에 실현하며, 적은 수작업 표시로도 효율적으로 작동합니다. 이를 위해 PCA는 인간이 정의한 과제와 목표만 필요로 하며, 대화 전 LLM이 핵심 및 필요한 SOP를 오프라인으로 계획하고 대화 중에는 SOP를 참조해 온라인으로 최적의 행동 경로를 계획하여 대답을 생성합니다.

- **Technical Details**: PCA는 대화 데이터를 반자동으로 생성하는 프레임워크를 제안하고 고품질의 대화 데이터셋(PCA-D)을 큐레이션했습니다. 또한, PCA의 다양한 변형 및 평가 메트릭을 개발했으며, 예를 들어 Monte Carlo Tree Search (MCTS) 기반 변형인 PCA-M이 있습니다. 이는 SOP 제약을 만족시키면서도 최적의 대화 행동을 탐색해 대화의 주도성과 논리적 일관성을 높입니다. PCA-M은 알파고(AlphaGo)와 유사하게 MCTS를 활용하여 SOP 규칙을 확장 및 시뮬레이션 단계에서 사용합니다.

- **Performance Highlights**: 실험 결과에 따르면, PCA-D로 미세 조정된 LLM은 성능을 크게 향상시키고 새로운 도메인에도 일반화될 수 있음을 보여줍니다. PCA-M은 대화의 통제 가능성, 주도성, 과제 성공률, 논리적 일관성 면에서 다른 CoT 및 ToT 기준점을 능가하여 산업 대화 시나리오에 적용 가능합니다.



### TartuNLP @ AXOLOTL-24: Leveraging Classifier Output for New Sense Detection in Lexical Semantics (https://arxiv.org/abs/2407.03861)
Comments:
          Accepted to the 5th International Workshop on Computational Approaches to Historical Language Change 2024 (LChange'24)

- **What's New**: 이 논문에서는 AXOLOTL-24 공유 작업에 대한 참여를 설명하고 있습니다. 이 작업은 시간이 지남에 따라 단어가 새로운 의미를 얻는지를 식별하고, 그 새로운 의미에 대한 정의를 생성하는 두 가지 하위 작업으로 구성됩니다. 저자들은 이 두 하위 작업을 해결하기 위해 개념적으로 간단하고 계산 비용이 적게 드는 솔루션을 구현했습니다. 우리의 제출 결과는 첫 번째 하위 작업에서 3위를, 두 번째 하위 작업에서 1위를 차지했습니다.

- **Technical Details**: 저자들은 GlossBERT 접근법을 기반으로 이진 분류 모델을 사용하여 용례(usage examples)와 정의(glosses)를 매칭했습니다. 이 모델은 새로운 의미를 식별하는 문제를 처리하기 위해 사용되었습니다. Adapter 기반의 이진 분류 모델들은 사용 예제와 정의 사이의 일치 여부를 예측하는데 사용되었습니다. 정의 생성 작업을 위해 위키낱말(Wiktionary)에서 새로운 의미의 정의를 매칭하는 모델을 동일하게 사용했습니다. 이 방법은 XLM-RoBERTa 모델을 기반으로 한 다중 언어 학습과 효율적인 미세조정(fine-tuning) 기법을 사용합니다.

- **Performance Highlights**: 연구팀의 시스템은 두 번째 하위 작업에서 1위를 차지했고, 첫 번째 하위 작업에서도 경쟁력 있는 결과를 얻었습니다. 특히, 새로운 의미의 정의 생성 작업에서 최고 성과를 나타냈습니다.



### Anthropocentric bias and the possibility of artificial cognition (https://arxiv.org/abs/2407.03859)
Comments:
          Accepted for ICML 2024 (Workshop on Large Language Models and Cognition)

- **What's New**: 대형 언어 모델(LLMs)의 인지 능력을 평가하는 과정에서 인간 중심의 편향을 극복해야 한다는 새로운 관점을 제시하고 있습니다. 특히, LLMs의 성능 저하 원인이 보조적 요인들 때문일 가능성을 무시하는 Type-I 편향과, 인간과 다른 메커니즘을 가진 LLMs의 전략을 진정한 능력으로 인정하지 않는 Type-II 편향을 강조합니다.

- **Technical Details**: 인지과학에서 중요하게 다루어지는 competence와 performance의 구분을 LLMs 평가에 적용할 때 나타나는 문제점을 논의합니다. Competence는 시스템의 내부 지식 또는 능력, performance는 그 능력이 외적으로 드러나는 행동을 의미합니다. 인간 중심적 편향이 이러한 구분을 LLMs에 적용할 때 잘못된 평가로 이어질 수 있다고 주장합니다.

- **Performance Highlights**: Hu & Frank (2024)의 연구를 예로 들어, 언어 모델이 주어-동사 일치와 같은 문법적 특징에 민감한지를 평가할 때 메타언어적 판단을 요구하는 방법보다는 직접적인 확률 비교 접근법이 더 유효하다는 결론을 도출했습니다. 이는 보조적 과제 요구(auxiliary task demands)가 모델 성능을 저하시킬 수 있다는 점을 보여줍니다.



### HYBRINFOX at CheckThat! 2024 -- Task 1: Enhancing Language Models with Structured Information for Check-Worthiness Estimation (https://arxiv.org/abs/2407.03850)
Comments:
          Paper to appear in the Proceedings of the Conference and Labs of the Evaluation Forum (CLEF 2024 CheckThat!)

- **What's New**: 이번 논문은 CheckThat! 2024 - Task 1 대회를 위한 HYBRINFOX 팀의 실험과 결과를 요약합니다. 팀은 RoBERTa와 같은 언어 모델(Language Models)에 문장에서 추출한 트리플(triples) (주어; 동사; 목적어) 임베딩을 추가하여 모델의 성능을 향상시키는 접근 방식을 제안했습니다. 이 방법은 단순한 언어 모델만 사용하는 것보다 성능이 향상되었음을 확인했습니다. 영어 데이터 평가에서는 F1 점수 71.1을 얻어 27팀 중 12위를 차지했고, 네덜란드어와 아랍어에서는 다소 혼합된 결과를 얻었습니다.

- **Technical Details**: 기존의 언어 모델을 미세 조정하는 단순 접근 방식 대신, HYBRINFOX 팀은 RoBERTa 모델(차원 768)을 사용하여 문장을 임베딩했습니다. 동시에 Open Information Extraction 시스템을 통해 텍스트에서 트리플을 추출했습니다. 추출한 트리플(주어; 동사; 목적어)은 fastText로 인코딩 되었고, 각 부분은 밀집 레이어를 거친 후 최종적으로 트리플과 언어 모델 임베딩을 결합하여 체크 가치 추정(checkworthiness estimation)을 수행했습니다. 영어 이외의 언어에 대해서는 멀티링구얼 버트(multilingual BERT)와 Multi²OIE 시스템을 사용했습니다.

- **Performance Highlights**: 제안된 접근 방식은 영어 데이터셋에서 F1 점수 71.1을 기록하며 27팀 중 12위를 차지했습니다. 그러나 네덜란드어와 아랍어 평가 결과는 혼재되어 네덜란드어에서는 8위(F1 점수 58.9), 아랍어에서는 10위(F1 점수 51.9)를 기록했습니다. 이는 제안된 트리플 임베딩과 언어 모델의 조합이 영어 데이터에 더 효과적이나, 다른 언어에는 추가적인 조정이 필요함을 나타냅니다.



### On the Benchmarking of LLMs for Open-Domain Dialogue Evaluation (https://arxiv.org/abs/2407.03841)
Comments:
          Accepted to the 6th NLP for Conversational AI workshop at ACL

- **What's New**: 최근 큰 언어 모델 (LLMs)의 탁월한 자연어 처리(NLP) 능력이 주목받고 있습니다. 특히, 자동 개방형 대화 평가에서 LLM이 평가 프레임워크에 원활하게 통합되어 왔으며, 인간 평가와 함께 대부분의 평가를 이루고 있습니다. 하지만 기존 평가 벤치마크는 낡은 데이터셋에 의존하며, 최신 챗봇 모델의 기능과 한계를 충분히 포착하지 못하는 사례가 많습니다.

- **Technical Details**: 이 논문은 현재 사용되는 평가 벤치마크의 문제점을 비판적으로 분석합니다. 구체적으로, 기존 데이터셋이 약한 챗봇을 사용하여 평가 프레임워크나 메트릭을 평가하는 경우가 많으며, 주로 유창성 (Fluency)과 관련성 (Relevance)와 같은 문제들에 초점을 맞추고 있습니다. LLM 등장 후 이러한 품질 측면은 대부분 쓸모가 없으며, 여전히 기존의 벤치마크가 이러한 구식 기준을 우선시하게 되어 최신 챗봇의 기능과 평가 관행 간의 괴리감이 발생합니다.

- **Performance Highlights**: 최근 생성된 LLM 데이터셋(SODA)에 대한 소규모 주석 실험 결과, GPT-4와 같은 LLM 평가자는 현대 LLM 챗봇이 생성한 대화의 실제 결함을 탐지하는데 어려움을 겪고 있습니다. 유창성이 부족한 대화는 쉽게 탐지되지만 찾아내기 어렵고, LLM 평가는 일관성(Coherence)과 상식(Commonsense) 문제를 제대로 식별하지 못하는 것으로 나타났습니다. 이러한 문제는 현대 챗봇이 아직 완벽하게 해결하지 못한 부분이며, 더 나은 탐지와 평가가 필요한 부분입니다.



### ConText at WASSA 2024 Empathy and Personality Shared Task: History-Dependent Embedding Utterance Representations for Empathy and Emotion Prediction in Conversations (https://arxiv.org/abs/2407.03818)
Comments:
          WASSA'24

- **What's New**: 이번 논문은 WASSA 공유 과제에서의 대화 중 공감(empathy) 및 감정 예측(emotion prediction)에 대한 새로운 접근 방법을 제시합니다. 특히, 이전의 대화 맥락을 적절하게 선택 및 표현하는 것이 공감과 감정을 모델링하는 데 얼마나 중요한지에 대해 다루고 있습니다. 본 논문에서 제안하는 방법은 사전 학습된 언어 모델(Pre-trained Language Model)을 사용하여 대화의 각 발화문에 대한 공감, 감정의 극성(emotion polarity) 및 감정의 강도(emotion intensity)를 예측하는 것입니다. 이 시스템은 CONV-turn 트랙에서 1위를, CONV-dialog 트랙에서 2위를 차지하였습니다.

- **Technical Details**: 논문에서 사용된 모델링 방법은 다음과 같습니다. 발화를 학습된 언어 모델에 입력하고 회귀 머리(regression head)를 추가하여 예측을 수행합니다. CONV-turn 트랙에서는 분류 대상 발화와 이전 대화의 특정 발화들을 함께 입력으로 사용하여 공감, 감정의 극성 및 강도를 모델링하였습니다. 반면, CONV-dialog 트랙에서는 모든 발화를 입력으로 사용하고, 예측 대상 화자를 식별하는 토큰을 추가하여 상대방 공감을 모델링하였습니다. 이 방법은 대화의 역사(history) 정보를 보존하는 데 더 효과적이라고 저자들은 주장합니다.

- **Performance Highlights**: 제안된 시스템은 CONV-turn 트랙에서 1위를 차지하고, CONV-dialog 트랙에서 매우 근소한 차이로 2위를 기록하였습니다. 이는 대화의 적절한 맥락을 선택하여 사전 학습된 언어 모델에 입력하는 방식이 얼마나 효과적인지 보여주는 결과입니다. 다양한 사전 학습된 언어 모델(RoBERTa, DeBERTa, Longformer)을 사용했고, 각각의 모델에 대해 Adam 옵티마이저와 조기 중지(Early Stopping) 기준을 설정하여 성능을 최적화했습니다.



### Finetuning End-to-End Models for Estonian Conversational Spoken Language Translation (https://arxiv.org/abs/2407.03809)
Comments:
          Accepted to LoResMT 2024 (ACL workshop)

- **What's New**: 이번 연구는 양방향 에스토니아어-영어 및 에스토니아어-러시어 회화 음성-텍스트 번역을 위한 엔드투엔드(end-to-end) 모델의 미세 조정을 조사합니다. 에스토니아어에 대한 음성 번역 데이터가 부족하여 웹 스크래핑(web scraping)과 음성 인식(ASR) 데이터세트를 사용한 기계 번역(MT)으로 데이터를 합성하여 추가 학습 데이터를 생성하였습니다. 이 연구는 Whisper, OWSM 3.1, SeamlessM4T라는 세 가지 공공 모델을 평가하고, 합성 데이터를 활용한 미세 조정이 번역 정확도를 크게 향상시켰음을 밝혔습니다.

- **Technical Details**: 이번 연구에서는 제한된 에스토니아어 음성 번역 데이터를 보완하기 위해 ASR 데이터와 기계 번역을 사용하여 합성된 데이터를 활용했습니다. 또한 웹 스크래핑을 통해 추가 데이터를 수집하였습니다. 세 가지 공공 모델(Whisper, OWSM 3.1, SeamlessM4T)을 미세 조정하여 양방향 에스토니아어-영어와 에스토니아어-러시어 음성-텍스트 번역에서의 성능을 평가했습니다. 평가 기준으로 BLEU 및 BLEURT 지표를 사용하였습니다.

- **Performance Highlights**: 합성 데이터를 사용한 미세 조정 결과, 번역 정확도가 크게 향상됨을 확인했습니다. 특히 SeamlessM4T 모델은 현 상태의 음성 인식 및 기계 번역 모델을 사용한 계단식 음성 번역 시스템을 능가하거나 이에 상응하는 성능을 보였습니다. 이에 따라 Whisper와 같은 대규모 다중 언어 사전 학습 모델이 상대적으로 낮은 자원의 언어 번역에서도 높은 성능을 발휘할 수 있음을 증명했습니다. 이번 연구에서 사용된 데이터는 공개되어 있으며, 미세 조정된 Whisper 모델 역시 공개되었습니다.



### Cognitive Modeling with Scaffolded LLMs: A Case Study of Referential Expression Generation (https://arxiv.org/abs/2407.03805)
Comments:
          11 pages, 3 figures, 2 algorithms, to appear at the ICML 2024 workshop on Large Language Models and Cognition

- **What's New**: 이번 연구는 Dale & Reiter (1995)의 알고리즘적 인지 모델을 기반으로 LLMs (Large Language Models)를 사용하여 인지 모델의 언어 생성 능력을 평가한 것입니다. 기존의 기호적(task analysis) 접근 방식을 넘어서 하이브리드 신경-기호적(neuro-symbolic) 모델을 도입해 참조 표현 생성 문제를 해결하고자 했습니다.

- **Technical Details**: 모델은 Dale & Reiter (1995)의증분 알고리즘(Incremental Algorithm, IA)을 기반으로, LLMs를 결합한 반복적 모델을 제안합니다. LLM은 제안된 표현을 생성하는 모듈(UtterancesProposer)과 제안된 표현이 의미론적으로 일치하는지 평가하는 모듈(SemanticEvaluator)로 활용됩니다. 이번 연구는 GPT-3.5-turbo 모델을 사용하였으며, 각 모듈의 기능과 평가 방법은 부록 A에 상세히 설명되어 있습니다. 실험에는 A3DS 데이터셋이 사용되었습니다.

- **Performance Highlights**: 이 하이브리드 접근 방식은 인지적으로 타당하며 복잡한 컨텍스트에서 우수한 성능을 발휘했습니다. 데이터셋 내 다양한 상황에서 인지적 모델링의 유연성과 높은 정확도를 보였습니다.



### M$\mathbf5$ -- A Diverse Benchmark to Assess the Performance of Large Multimodal Models Across Multilingual and Multicultural Vision-Language Tasks (https://arxiv.org/abs/2407.03791)
- **What's New**: 최근 ChatGPT 출시 이후, 자연어 처리 분야에서 특히 대형 언어 모델(LLMs)과 멀티모달 모델(LMMs)의 발전이 두드러졌습니다. 그러나 LLMs는 언어와 문화적 맥락에서 성능 차이가 크다는 문제를 가지고 있으며, 현재까지 이러한 문제를 다루는 멀티모달 비주얼-언어 설정에 대한 벤치마크는 부족했습니다. 이번 연구에서는 이러한 빈틈을 메우기 위해 M5라는 최초의 종합 벤치마크를 소개합니다. M5는 5개의 과제를 다루고 41개 언어를 포함하는 8개의 데이터셋을 통해 LMMs를 평가합니다.

- **Technical Details**: M5 벤치마크는 다양한 비주얼-언어 과제에서 LMMs를 평가하는 멀티모달 및 다국어 벤치마크입니다. 주요 구성 요소로는 LLM, 비전 인코더 모델, 텍스트 임베딩 공간으로 이미지 임베딩을 매핑하는 네트워크가 있으며, 새로운 평가 데이터셋으로는 M5-VGR과 M5-VLOD가 포함됩니다. 이러한 데이터셋은 특히 저평가된 아프리카와 아시아 언어와 문화를 반영합니다.

- **Performance Highlights**: 본 연구를 통해 LMMs의 성능이 고자원 언어와 저자원 언어 간에 상당한 불균형이 존재하며, 모델 크기가 클수록 다국어 설정에서 반드시 더 나은 성능을 보이지는 않는다는 것을 확인했습니다. 또한, 평가된 모든 오픈 소스 모델이 Visio-Linguistic Outlier Detection(VLOD) 과제에서 무작위 베이스라인을 크게 상회하지 못하는 것으로 나타났습니다.



### Functional Faithfulness in the Wild: Circuit Discovery with Differentiable Computation Graph Pruning (https://arxiv.org/abs/2407.03779)
- **What's New**: 이번 논문에서는 회로 발견(Circuit Discovery) 작업에 대한 종합적인 재구성과 함께 차별화된 마스킹(differentiable masking)을 기반으로 한 DiscoGP라는 새로운 알고리즘을 소개합니다. 이 알고리즘은 언어 모델(Language Models, LMs)의 계산 메커니즘을 해부하여 해석하는 작업에서 높은 신뢰성과 완전성을 보여줍니다. 기존 방식의 한계를 해결하며, 생성 AI의 내부 작동 원리에 대한 새로운 통찰을 제공합니다.

- **Technical Details**: DiscoGP 알고리즘은 모델의 가중치와 연결 간선의 복잡성을 동시에 줄이는 방식으로, 계산 그래프를 가지치기(weight and edge pruning)하는 작업을 수행합니다. 이 과정은 가중치와 간선에 학습 가능한 이진 마스크를 부여하여, 최적화 과정을 통해 희소한(subnetwork) 회로를 발견하는 방식으로 진행됩니다. 기존 알고리즘들이 가진 두 가지 주요 한계를 극복하며, 더 신뢰할 수 있고 완전한 회로를 찾을 수 있게 합니다.

- **Performance Highlights**: DiscoGP는 기존 방법들보다 더 기능적으로 충실하고, 모델 성능이 거의 유지되면서도 필요한 가중치와 간선의 일부만을 사용하여 추론을 수행할 수 있습니다. 또한, 발견된 회로가 모델에서 제거된 경우 성능이 크게 감소하여, 이러한 회로가 모델의 필수적인 구성 요소임을 증명합니다. 새로운 구조는 하위 레이어의 주목(attention) 헤드가 중요한 역할을 한다는 점과, 연결 간선이 상위 레이어에서 더 두드러진다는 점을 밝혀내어, 언어 모델 해석에 새로운 접근 방식을 제시합니다.



### HYBRINFOX at CheckThat! 2024 -- Task 2: Enriching BERT Models with the Expert System VAGO for Subjectivity Detection (https://arxiv.org/abs/2407.03770)
Comments:
          To appear in the Proceedings of the Conference and Labs of the Evaluation Forum (CLEF 2024 CheckThat!)

- **What's New**: HYBRINFOX 방법은 CLEF 2024 CheckThat! 대회의 주관식 탐지(Task 2)를 해결하기 위해 개발되었습니다. 이 방법은 RoBERTa 모델을 미세 조정하고, 의미를 포착하기 위해 frozen sentence-BERT(sBERT) 모델 및 어휘에 기반한 VAGO 전문가 시스템의 점수를 결합한 혼합 시스템을 사용합니다.

- **Technical Details**: HYBRINFOX는 주관식 탐지를 위해 RoBERTa 모델을 미세 조정하고, sBERT 모델을 동결하여 의미를 포착하며, 영어 버전의 VAGO 전문가 시스템에서 주관성과 애매함을 측정하는 점수를 계산합니다. 이 방법은 RoBERTa 및 sentence-BERT의 임베딩을 결합하여 차원 축소를 수행한 후, 애매함 점수와 함께 분류 선형 계층으로 입력됩니다.

- **Performance Highlights**: HYBRINFOX 방법은 평가 데이터에서 영어 부문에서 매크로 F1 점수 0.7442로 1위를 차지했습니다. 다른 언어에 대해서는 영어로 번역 과정을 거쳐 더 혼합된 결과를 보였으며, 다국어 부문에서 1위, 이탈리아어 부문에서 2위를 차지했지만 불가리아어, 독일어, 아랍어에서는 기준선보다 낮은 성능을 보였습니다. 검토 결과, 번역 단계에서의 정확도 손실이 원인으로 지적되었습니다.



### Argument Mining in Data Scarce Settings: Cross-lingual Transfer and Few-shot Techniques (https://arxiv.org/abs/2407.03748)
- **What's New**: 최근 연구는 언어 학습모델을 활용하여 수작업으로 주석된 데이터가 부족한 세계 대부분의 언어에 대한 시퀀스 라벨링(sequence labelling) 문제를 해결하기 위해 여러 가지 전략을 탐구해 왔습니다. 이번 논문에서는 주장 추출(Argument Mining)에 대한 새로운 발견을 제시합니다. 많은 이전 연구들과 달리, 이 논문에서는 데이터-전달(data-transfer)이 모델-전달(model-transfer)보다, 파인튜닝(fine-tuning)이 few-shot 방법보다 더 나은 성능을 발휘한다고 주장합니다.

- **Technical Details**: 논문에서는 세 가지 접근법에 집중합니다: 다국어 미리 훈련된 언어 모델의 크로스-링걸 전이(cross-lingual transfer, model-transfer), 데이터 번역 및 라벨 프로젝션(data translation and label projection, data-transfer), 그리고 few-shot 학습을 위한 프롬프팅(prompting). 기본적으로 시퀀스 라벨링 작업이 길고 복잡한 담론 구조를 요구하는 주장 추출 작업에 대한 실험을 통해 데이터를 번역하고 라벨을 프로젝션하는 방법이 더 나은 성능을 보임을 입증했습니다. 실험에 사용된 코퍼스는 AbstRCT로, 의료 논문 초록을 기반으로 한 데이터셋을 다양한 언어로 번역하여 사용했습니다.

- **Performance Highlights**: 실험 결과에 따르면, 주장 추출 작업에서는 기존 연구와 달리 데이터-전달 방법이 모델-전달보다 더 나은 성능을 보였으며, 파인튜닝이 few-shot 방법보다 더 우수한 결과를 나타냈습니다. 특히, 데이터-전달에서는 데이터셋의 도메인이 중요한 역할을 했고, few-shot에서는 작업의 유형(시퀀스 스팬의 길이와 복잡성)과 샘플링 방법이 중요한 요소로 작용했습니다. 전체적인 실험 데이터와 코드는 공개되어 있어 추가적인 연구와 검증이 가능합니다.



### Improving Self-supervised Pre-training using Accent-Specific Codebooks (https://arxiv.org/abs/2407.03734)
Comments:
          Accepted to INTERSPEECH 2024

- **What's New**: 본 논문에서는 최신 자동 음성 인식 (ASR) 시스템의 성능을 저하시키는 주요 문제인 음성 억양 문제를 해결하기 위해 억양 인지 학습 기법을 제안합니다.

- **Technical Details**: 제안된 방법은 자가 지도 학습(Self-Supervised Learning, SSL) 동안 학습 가능한 억양별 특정 코드북을 도입하여 억양 정보를 포착합니다. 이 코드북은 교차 주의 모듈(cross-attention module)을 사용하여 모델에 통합되며, 이후 ASR 미세 조정(fine-tuning) 단계에서 추가로 정제됩니다. HuBERT 구조를 바탕으로, 세 단계의 모듈을 통해 모델을 구성합니다: 컨벌루션 기반 파형 인코더(convolution-based waveform encoder), 트랜스포머 기반 인코더(Transformer-based encoder), 그리고 투영 모듈(projection module). 이는 Mozilla Common Voice 데이터셋에서 놀라운 성능 향상을 보였습니다.

- **Performance Highlights**: 제안된 억양 적응 기술은 기존 방법보다 최대 9%의 단어 오류율(WER) 절감을 이루었으며, 새로운 억양에 대해서도 탁월한 성능을 보였습니다. 또한, L2-Arctic 데이터셋에서 제로 샷(zero-shot) 설정으로도 높은 일반화 능력을 입증했습니다.



### Multi-Convformer: Extending Conformer with Multiple Convolution Kernels (https://arxiv.org/abs/2407.03718)
Comments:
          Accepted to INTERSPEECH 2024

- **What's New**: 새로운 논문에서 Multi-Convformer라는 모델을 소개하였습니다. 이 모델은 Conformer 아키텍처 내의 convolution module에 여러 개의 convolution kernels와 게이팅(gating)을 결합하여 지역 의존성을 더 잘 모델링할 수 있도록 합니다. 이를 통해 기존 Conformer 변형 모델들(CgMLP, E-Branchformer)과 경쟁할 만큼 성능이 향상되었으며, 파라미터 효율성도 높습니다.

- **Technical Details**: Multi-Convformer는 Conformer 아키텍처의 고정된 커널 대신 여러 개의 커널을 사용한 convolution module을 반영한 형태입니다. 이 model은 attention 기반 encoder-decoder 모델(AED), 순수 CTC, RNN-Transducer(RNN-T)와 같은 일반적인 ASR 시스템에서 encoder(enocder) 모듈의 수정만을 포함합니다. Multi-Convformer encoder layer는 포지션 별 feed-forward 레이어, multi-head attention, convolutions를 쌓고 레이어 정규화(layer normalization)와 residual connections를 혼합한 구조로 되어 있습니다.

- **Performance Highlights**: 우리는 Multi-Convformer를 Conformer와 그 변형 모델들에 대해 네 가지 데이터셋(Librispeech, Tedlium2, AISHELL, SLURP)과 세 가지 모델링 패러다임에서 실험하였습니다. 그 결과 최대 8%의 상대적인 단어 오류율(WER, Word Error Rate) 향상을 이루었습니다. 이는 Conformer 아키텍처의 성능을 비약적으로 개선한 결과로 평가됩니다.



### Text2TimeSeries: Enhancing Financial Forecasting through Time Series Prediction Updates with Event-Driven Insights from Large Language Models (https://arxiv.org/abs/2407.03689)
Comments:
          21 pages, 12 figures

- **What's New**: 현재 시간 시계열 예측(time series prediction)에 향후 가치 예측을 위해 텍스트 정보를 통합하는 협업 모델링 프레임워크(collaborative modeling framework)를 제안합니다. 특히, 대형 언어 모델(LLMs)의 직관을 활용하여 실제 숫자 시간 시계열 예측을 갱신하고자 합니다.

- **Technical Details**: 이 연구는 주가 예측을 위한 다변량 시계열 모델(multivariate time series model)을 텍스트 정보와 결합하여 이용됩니다. 주가 변화를 예측하기 위해 대형 언어 모델(LLM)로부터 이벤트 기반의 직관을 사용하고, 주가 변화를 나타내는 실수(real value)를 예측합니다. 또한, 다변량 시계열 모델과 상태 변화 모델(state change model)로부터 예측된 결과를 결합하여 최적화된 예측 결과를 제공합니다.

- **Performance Highlights**: 금융 데이터를 기반으로 제안된 모델을 평가한 결과, 이벤트를 고려하지 않은 기존의 시계열 모델 예측보다 더 정확한 예측을 제공하는 것으로 나타났습니다. 특히, 주가 관련 뉴스 텍스트를 통한 세밀한 언어 이해력을 통해 주간 주가 변동 예측에서 우수한 성과를 보였습니다.



### STOC-TOT: Stochastic Tree-of-Thought with Constrained Decoding for Complex Reasoning in Multi-Hop Question Answering (https://arxiv.org/abs/2407.03687)
Comments:
          10 pages, 5 figures

- **What's New**: 새로운 연구인 STOC-TOT가 소개되었습니다. 이는 무작위(Tree-of-Thought) 추론 방법과 제한적 디코딩을 결합하여 복잡한 다중 도약 질문 응답(MHQA) 작업에서의 성능을 향상시킵니다.

- **Technical Details**: STOC-TOT는 모델이 원래 질문을 더 작은 하위 질문으로 분해하여 다양한 추론 경로를 형성하도록 프롬팅합니다. 각각의 추론 경로에 확률 추정치를 제공합니다. 답변 시간에는 제한적 디코딩(constrained decoding)을 사용하여 모델이 좀 더 신뢰할 수 있는 답변을 생성하도록 유도합니다. 이는 'hallucination'(환상)을 줄이고 보다 정확한 답변을 제공하는 데 도움을 줍니다.

- **Performance Highlights**: 두 개의 MHQA 데이터셋과 다섯 개의 대형 언어 모델(LLMs)을 비교한 실험에서 STOC-TOT는 기존의 추론 프롬팅 방법들보다 상당한 성능 향상을 보였습니다. 특히, HotpotQA 데이터셋에서 Exact Match 정확도가 7% 증가했고, F1 점수가 7.8 포인트 상승했습니다. 이로써 GPT-4를 포함한 모델들에서 STOC-TOT의 우수성을 입증하였습니다.



### Improving Self Consistency in LLMs through Probabilistic Tokenization (https://arxiv.org/abs/2407.03678)
Comments:
          ICML 2024 Workshop on LLMs and Cognition

- **What's New**: 이 연구에서는 현대의 큰 언어 모델(LLMs)의 토크나이저가 제공하는 다중 토큰화 기능을 적극 활용하여 LLM의 자기 일관성을 증대시키는 기법을 제안합니다. 과거 연구에서는 '확률적 토큰화(probabilistic tokenizations)'가 신경 기계 번역 모델의 성능을 높이는 것으로 나타났지만, LLMs에서는 아직 채택되지 않았습니다. 본 논문은 확률적 토큰화가 LLM의 추론 작업에서 논리적으로 다양한 경로를 생성하게 하고, 단순한 언어적 다양성을 넘어서서 문제 해결 능력을 향상시킨다는 것을 실험을 통해 입증합니다.

- **Technical Details**: 본 연구는 Byte-Pair Encoding(BPE)의 다중 토큰화 기능을 활용하여 LLM의 자기 일관성을 개선하는 방안을 제시합니다. BPE는 특정 문자열을 필요한 만큼 작은 토큰으로 나눈 후 이를 병합하는 방식이며, 이는 동일 문자열에 대해 다양한 토큰화가 가능하게 합니다. 확률적 토큰화를 통해 다양한 추론 경로를 생성하여 최종 답을 선택하는 방식입니다. 연구는 Kudo(2018)가 제안한 방식을 확장하여 각 토큰화에 대해 확률을 부여하고 이 확률에 비례하여 샘플링하는 방법을 제안합니다.

- **Performance Highlights**: 제안된 확률적 토큰화 기법은 5개의 LLM 계열과 4개의 추론 벤치마크에서 실험이 수행되었으며, 그 결과 다중 토큰화가 모델의 자기 일관성 및 추론 능력을 크게 향상시킨다는 것을 데이터로 입증하였습니다. 이 방법은 모델의 다음 토큰 분포에 의존하지 않기 때문에 기존 샘플링 기법보다 더 자연스럽고 다양한 결과를 생성하는 것이 가능했습니다.



### GPT-4 vs. Human Translators: A Comprehensive Evaluation of Translation Quality Across Languages, Domains, and Expertise Levels (https://arxiv.org/abs/2407.03658)
- **What's New**: 이 연구는 대형 언어 모델(Large Language Models, LLMs)의 번역 품질을 인간 번역자와 비교하여 처음으로 평가했습니다. 특히 GPT-4를 다양한 전문성 수준의 인간 번역자들과 여러 언어 쌍 및 도메인에서 비교했습니다. 그 결과, GPT-4는 초급 번역자와 유사한 성능을 보였으나 중급 및 고급 번역자와는 차이가 있음을 발견했습니다.

- **Technical Details**: 연구는 세 가지 언어 쌍(중국어↔영어, 러시아어↔영어, 중국어↔힌디어)과 세 가지 도메인(뉴스, 기술, 바이오메디컬)에서 수행되었습니다. MQM(Multidimensional Quality Metrics) 스키마를 사용하여 각 번역 문장의 오류를 독립된 전문 평가자가 라벨링했습니다. GPT-4는 자원 많은 언어 쌍에서는 중급 번역자와 유사한 성능을 보였으나, 힌디어 등 자원이 부족한 언어 쌍에서는 성능 저하가 두드러졌습니다.

- **Performance Highlights**: GPT-4는 전체 오류 측면에서 초급 번역자와 유사한 성능을 보였고, 중급 및 고급 번역자에게는 뒤떨어졌습니다. 특정 언어 쌍 및 도메인에서는 균형 잡히지 않은 성능을 보였으며, 특히 자원이 풍부한 언어 쌍에서 더 나은 성능을 보였습니다. GPT-4의 번역은 문자적 번역이 많은 반면, 인간 번역자는 종종 배경 정보를 과도하게 고려하는 경향이 있음을 발견했습니다.



### Evaluating Language Model Context Windows: A "Working Memory" Test and Inference-time Correction (https://arxiv.org/abs/2407.03651)
- **What's New**: SWiM(스노클 작업 메모리 테스트)이라는 새로운 평가 프레임워크가 제안되었습니다. 이 프레임워크는 현실 세계의 사용 사례에서 긴 문맥(Large Context) 모델의 성능을 평가합니다. 이러한 모델은 최대 2백만 토큰을 수용할 수 있는 능력을 가지고 있지만, 실제 적용 가능성은 아직 불확실합니다. SWiM은 이 문제를 해결하고 평가의 정확성을 높이기 위해 고안되었습니다.

- **Technical Details**: SWiM 프레임워크는 네 가지 단계로 구성됩니다: 작업 생성(Task Generation), 작업 검증(Task Validation), 작업 완료(Task Completion), 작업 평가(Task Evaluation). 특히, SWiM은 긴 문맥을 갖는 모델의 문서 Q&A 성능을 평가하며, 문서 내 정보의 위치(속성-위치 효과)와 문맥의 크기(노이즈 수준)에 따른 성능 변화를 테스트합니다. 또한, 성능 저하를 완화하기 위해 '메디안 투표(메도이드 보팅)'라는 간단한 알고리즘을 제안합니다.

- **Performance Highlights**: 실험 결과, 많은 긴 문맥 모델들이 문맥 창의 중간 위치에 있는 정보를 효과적으로 회수하지 못하는 '중간에 잃어버리는 효과(lost-in-the-middle effect)'를 보였습니다. 이 문제를 해결하기 위해 제안된 메디안 투표 메서드는 모델 성능을 최대 24%까지 향상시켰습니다. 또한, 다량의 노이즈가 포함된 문맥에서도 일부 모델들은 뛰어난 성능을 보였으며, 특히 Gemini-1.5-Pro 모델이 가장 우수한 성능을 나타냈습니다.



### Differentiating between human-written and AI-generated texts using linguistic features automatically extracted from an online computational too (https://arxiv.org/abs/2407.03646)
- **What's New**: 이번 연구는 ChatGPT와 같은 인공지능(AI) 모델이 생성한 텍스트와 인간이 작성한 텍스트 간의 언어적 특징을 체계적으로 비교한 최초의 시도 중 하나입니다. 연구는 AI가 인간의 글쓰기를 모방하는 능력을 평가하는 것을 목적으로 합니다. 인간이 작성한 에세이를 기준으로, 동일한 길이의 에세이를 ChatGPT를 이용해 생성했으며, 이를 Open Brain AI라는 온라인 도구를 사용해 분석했습니다.

- **Technical Details**: 이 연구는 음운론적(phonological), 형태론적(morphological), 구문론적(syntactic), 그리고 어휘론적(lexical) 요소를 측정해 AI와 인간이 작성한 텍스트의 차이점을 분석했습니다. 음소(phonemes), 단어 강세(word stress), 명사(nouns), 동사(verbs), 대명사(pronouns), 직접목적어(direct objects), 전치사 수식어(prepositional modifiers), 어려운 단어 사용 빈도 등의 다양한 언어적 특징에서 두 텍스트 간에 큰 차이가 나타났습니다.

- **Performance Highlights**: AI가 생성한 텍스트는 처음에는 인간의 언어와 유사하게 보일 수 있지만, 세부적인 언어적 요소에서 여전히 많은 차이가 있다는 점이 발견되었습니다. 이러한 결과는 언어 평가를 효율적으로 하기 위한 자동화 도구의 통합 필요성을 강조하며, 더 인간과 유사한 텍스트를 생산할 수 있는 AI 훈련 방법의 개선 필요성을 시사합니다.



### Continual Learning Optimizations for Auto-regressive Decoder of Multilingual ASR systems (https://arxiv.org/abs/2407.03645)
- **What's New**: 이 논문은 Continual Learning (CL, 지속학습)을 활용하여 사전훈련된 Multilingual ASR (MASR, 다국어 자동 음성 인식) 모델의 성능을 유지하면서 새로운 데이터를 사용해 모델을 미세 조정하는 방법을 연구합니다. 특히, MASR의 auto-regressive decoder(자동 회귀 디코더)에 대해 기존의 CL 방법이 비효율적이라는 가설을 세우고, 이를 검증하기 위해 네 가지 최적화를 제안합니다.

- **Technical Details**: 제안된 최적화 기법에는 다음이 포함됩니다: 1) decoder-layer gradient surgery (디코더 레이어 그래디언트 수술), 2) 사용되지 않는 토큰 임베딩 고정, 3) 새로 추가된 토큰의 출력 억제, 4) 학습률 재조정. 이 네 가지 최적화 기법을 통해 모델의 성능을 유지하고 향상시키는 방법을 제안합니다.

- **Performance Highlights**: Whisper 모델을 Common Voice 데이터셋의 새로운 10개 언어에 적응시키는 실험에서, 제안된 최적화 기법이 기존의 Experience Replay 방법에 비해 사전훈련된 언어들의 평균 단어 오류율(Average Word Error Rate, AWER)을 14.2%에서 12.4%로 감소시켰으며, 새로운 언어의 AWER 성능에는 영향을 미치지 않았음을 입증했습니다.



### DSLR: Document Refinement with Sentence-Level Re-ranking and Reconstruction to Enhance Retrieval-Augmented Generation (https://arxiv.org/abs/2407.03627)
- **What's New**: 최근 큰 언어 모델(LLMs)의 발전은 다양한 자연어 처리(NLP) 작업에서 성능을 크게 향상시켰습니다. 그러나 여전히 LLM은 비사실적 응답을 생성하는 데 어려움을 겪고 있습니다. Retrieval-Augmented Generation (RAG) 시스템은 외부 지식을 활용하여 이를 해결하지만, 정보 검색 실패와 관련 없는 정보를 필터링하는 문제를 여전히 가지고 있습니다. 이 문제를 해결하기 위해, 확실한 문장 레벨 재순위 및 재구성을 통한 문서 개선 프레임워크인 DSLR을 제안합니다.

- **Technical Details**: DSLR은 세 가지 단계로 이루어집니다. 첫째, 검색된 문서를 문장으로 분해(Decomposition)한 후, 재순위 모델을 사용하여 관련 없는 문장을 필터링(Re-ranking)합니다. 마지막으로, 필터링된 문장을 다시 문서로 재구성(Reconstruction)하여 원래의 문맥을 유지합니다. 이 과정은 비지도 학습(Unsupervised) 방식으로 추가적인 학습이 필요하지 않습니다. DSLR는 다양한 오픈 도메인 QA 데이터셋에서 검증되었으며, RAG 성능을 크게 향상시킵니다.

- **Performance Highlights**: DSLR는 기존의 고정 크기의 패시지보다 RAG 성능을 현저히 향상시킵니다. 특히, 특정 시나리오에서 추가 학습 없이도 높은 효율성과 효과를 보여줍니다. 구체적인 QA 데이터셋으로 평가한 결과, DSLR은 명확한 실효성을 나타내며, 각 단계의 기여도를 분석한 결과, 전체 성능 향상에 대한 각 단계의 기여를 확인할 수 있었습니다.



### Question-Analysis Prompting Improves LLM Performance in Reasoning Tasks (https://arxiv.org/abs/2407.03624)
Comments:
          11 pages, 8 figures

- **What's New**: 이번 연구에서는 LLM(Large Language Models)의 성능을 향상시키기 위해 Question Analysis Prompting(QAP)이라는 새로운 프롬프트 전략을 제안합니다. QAP는 모델이 문제를 해결하기 전에 문제를 설명하도록 유도합니다. 이는 특히 산술(GSM8K, AQuA, SAT)과 상식(StrategyQA) 데이터셋에서 GPT 3.5 Turbo와 GPT 4 Turbo 모델에서 테스트되었으며, 기존의 Chain-of-Thought(CoT), Plan and Solve Prompting(PS+), Take A Deep Breath(TADB) 프롬프트를 능가하는 성능을 보였습니다.

- **Technical Details**: {'Prompt Design': 'QAP는 모델이 문제를 해결하기 전에 문제를 n 단어 이상으로 설명하도록 유도합니다. 실험에서 n = 25, 50, 100, 150, 200을 사용했으며, 각 버전은 QAP25, QAP50, QAP100, QAP150, QAP200으로 명명되었습니다. 이 방식은 문제의 난이도와 모델 크기에 따라 프롬프트를 조정할 수 있습니다.', 'Experimental Setup': 'QAP는 GSM8K, AQuA, SAT 등의 산술 추리 데이터셋과 StrategyQA 상식 추리 데이터셋에서 테스트되었습니다. 두 가지 크기의 모델(GPT 3.5 Turbo, GPT 4 Turbo)을 사용하여 QAP의 영향을 평가했습니다.'}

- **Performance Highlights**: {'Arithmetic Reasoning': 'GPT 3.5 Turbo에서는 QAP가 3가지 산술 과제 중 2개에서 최고 성능을 보였고, GPT 4 Turbo에서는 동일한 2개의 산술 과제에서 최고 성능을 보였습니다.', 'Commonsense Reasoning': 'StrategyQA 데이터셋에서는 QAP가 일관되게 두 번째로 높은 성능을 보였으며, QAP25가 최고 성능을 보였습니다. 이는 상식적 추리에 더 적은 단어 설명이 더 유리하다는 것을 시사합니다.', 'General Observations': "QAP는 '어려운' 질문에서 다른 프롬프트보다 일관되게 우수한 성능을 보였으나, '쉬운' 질문에서는 더 긴 설명이 부정적인 영향을 미칠 수 있다는 점이 나타났습니다. 또한, QAP의 작은 버전(QAP25)은 산술 데이터셋에서 일부 응답이 불완전하게 출력되는 문제를 보였습니다."}

- **Additional Studies**: 프롬프트의 위치에 대한 추가 연구가 진행되었으며, 문제 전과 후에 프롬프트를 배치한 결과 유사한 성능을 보였습니다. 또한, 두 단계 QAP 접근법이 시험되었으나 단일 단계 프롬프트보다 성능이 낮았습니다.



### The Mysterious Case of Neuron 1512: Injectable Realignment Architectures Reveal Internal Characteristics of Meta's Llama 2 Mod (https://arxiv.org/abs/2407.03621)
Comments:
          21 pages, 17 figures

- **What's New**: 이 연구는 대형 언어 모델(LLMs)의 해석 가능성과 설명 가능성을 높이기 위한 새로운 접근법인 Injectable Realignment Model (IRM)을 소개합니다. IRM은 감정을 기반으로 한 정렬(alignment)을 유도함으로써, 모델의 가중치(weight)를 변경하지 않고도 언어 모델의 출력을 조절할 수 있습니다.

- **Technical Details**: IRM은 작은 신경망으로 설계되었으며, 기본적인 Transformer 아키텍처의 복잡한 메커니즘에서 정렬 행동을 분리합니다. 이 연구에서는 7B 파라미터의 언어 모델(Llama 2)을 대상으로 IRM을 적용했습니다. IRM의 출력은 모델의 전방 패스(forward pass) 동안 여러 지점에서 계층별로 추가되어 언어 모델의 동작을 조절합니다. 이 방식은 원래 모델의 파라미터를 변경하지 않아도 정렬 행동을 유도할 수 있게 합니다.

- **Performance Highlights**: 분석 결과, 거의 모든 Transformer 블록 내의 특정 뉴런 인덱스(1512)가 여러 정렬 데이터셋과 훈련 실행에서 강한 상관관계를 보였습니다. 이 현상은 대형 언어 모델들이 공유하는 설계 선택에 기인하는 것이며, Meta의 사전 훈련된 Llama 2 모델의 잠재적 약점을 강조합니다. 또한 IRM이 신경망의 활성 패턴을 분석하고 이해하는 도구로서 유용함을 보여줍니다.



### Visualizing Dialogues: Enhancing Image Selection through Dialogue Understanding with Large Language Models (https://arxiv.org/abs/2407.03615)
- **What's New**: 대화 시스템의 최신 연구는 이미지를 포함한 멀티모달 (multimodal) 응답의 중요성을 강조하고 있습니다. 이 연구는 대화의 전반적인 전달력을 향상시키고, 상호작용의 질을 높이는 데 기여합니다. 하지만 기존의 대화-이미지 검색 방법은 사전 훈련된 비전 언어 모델 (Vision Language Models, VLMs)이 복잡한 대화를 정확히 이해하는 데 한계가 있어, 이를 해결하기 위한 접근법을 제안합니다.

- **Technical Details**: 우리의 접근법은 대형 언어 모델 (Large Language Models, LLMs)의 강력한 추론 능력을 활용해 정확한 대화 연관 시각 기술자 (visual descriptors)를 생성하는 것입니다. 실험 결과, 이 방법이 대화-이미지 검색 성능을 크게 향상시키는 것으로 나타났습니다. 주로 LLM을 활용해 주요 시각적 요소에 대한 질문을 구성하고, 이를 바탕으로 대화 맥락을 반영한 이미지 검색을 수행합니다.

- **Performance Highlights**: 벤치마크 데이터셋에서 우리의 접근법은 기존 방법들보다 뛰어난 성능을 보였습니다. 다양한 시각적 단서와 LLM, 데이터셋에 걸쳐 일반화 능력도 입증되었습니다. 이를 통해 실생활의 다양한 응용 가능성을 표명합니다.



### Lateralization LoRA: Interleaved Instruction Tuning with Modality-Specialized Adaptations (https://arxiv.org/abs/2407.03604)
Comments:
          8 Pages, visual instruction tuning, parameter-efficient tuning

- **What's New**: Vision-Language Generalists (VLGs)의 기존 한계를 극복하고자 LeafInstruct의 출시가 이루어졌습니다. LeafInstruct는 30,000개 이상의 고품질 인스턴스들로 구성된 첫 오픈 소스(interleaved instruction tuning data)로, 다양한 10여 개의 도메인을 커버합니다.

- **Technical Details**: LeafInstruct는 VLGs에서 모달리티 혼선(modality interference)를 해결하기 위해 매개변수를 효율적으로 사용한 튜닝 방식인 Lateralization LoRA를 제안합니다. 이는 전통적인 선형 LoRA(linear LoRA)와 컨볼루션 LoRA(Convolutional LoRA)를 결합하여 텍스트 와 이미지를 생성하는 하이브리드 방식입니다. Lateralization LoRA는 텍스트 토큰에 대해서는 선형 LoRA를, 이미지 패치에 대해서는 저랭크 컨볼루션 적응층을 사용해 각 모달리티의 특성에 적합한 구조를 제공합니다.

- **Performance Highlights**: InterleavedBench 테스트 결과, Lateralization LoRA로 튜닝된 EMU2는 기존의 오픈 소스 모델들을 능가하는 성능을 보여주었습니다. 특히, EMU2는 복잡한 interleaved 작업들에서 탁월한 성능을 발휘하며, 인접 픽셀 간의 일관성과 왜곡 없는 고품질 이미지를 생성해냈습니다.



### Contrastive Chain-of-Thought Prompting (https://arxiv.org/abs/2407.03600)
Comments:
          6 pages, 0 figures

- **What's New**: 최신 연구에서는 체인-오브-생각(Chain-of-Thought, CoT) 프롬프트와 Context-Aware Decoding(CAD)의 조합을 탐구합니다. 특히, 입력 기반 대조 방법을 사용하여 CoT 프롬프트가 유도하는 추론 유형을 더 강화하는 방안을 제안했습니다. 이 방법이 다양한 데이터셋과 모델에서 안정화될 필요가 있지만, 초기 결과는 추가 연구를 충분히 뒷받침합니다.

- **Technical Details**: 이 연구에서는 특정 입력 쿼리(x)에 대해 언어 모델(θ)에 코드를 입력하고, CoT 프롬프트(c)와 아마추어 모델의 대조 강도(𝛼)를 적용해 새로운 확률 분포를 생성합니다. 이를 통해 모델이 입력을 추론할 때 CoT 프롬프트를 효과적으로 활용하면서도 부정확한 추론 행위를 제재합니다. 다양한 프롬프트와 실험 설정을 통해, 최적의 𝛼값을 찾고 여러 데이터셋을 통해 성능을 평가합니다.

- **Performance Highlights**: 세 가지 질의응답 데이터셋(GSM8K, AQuA, CommonSenseQA)에서 실험을 수행한 결과, Phi-1.5와 Mistral 7B 모델은 CoT와 CAD 기법에 의해 성능이 향상됨을 보였습니다. 특히 CommonSenseQADataset에서 Mistral 7B의 성능이 개선되었습니다. 그러나 모든 모델과 데이터셋에서 일관된 성능 향상을 보인 것은 아닙니다. 일부 조합에서는 성능이 하락하기도 했으며, 특정 데이터셋(GSM8K)에서는 결과가 오염될 가능성도 있었습니다.



### Zero-shot Persuasive Chatbots with LLM-Generated Strategies and Information Retrieva (https://arxiv.org/abs/2407.03585)
- **What's New**: 새로운 연구에서는 큰 언어 모델(LLMs)의 일반성과 내재된 설득 능력을 활용하여 특정 도메인에 대해 효과적이고 진실된 설득형 챗봇을 제로 샷(zero-shot) 방식으로 생성하는 방법을 제안합니다. 이를 통해 학습 데이터를 수집하는 비용과 어려움을 극복할 수 있습니다.

- **Technical Details**: PersuaBot이라는 새로운 챗봇은 두 개의 모듈, 즉 기본 질문 처리 모듈(QHM)과 전략 유지 모듈(SMM)에 의해 작동합니다. 전체적인 철학은 LLM에서 생성된 반응을 실제 추출된 사실로 대체하여 신뢰성을 높이는 것입니다. LLM은 먼저 응답을 생성한 후, 이를 여러 전략으로 나누어 각 섹션의 사실 여부를 점검합니다. 결합된 사실적 정보는 최종적으로 사용자에게 제공됩니다.

- **Performance Highlights**: 이 연구에서 PersuaBot은 세 가지 도메인(기부 요청, 추천 시스템, 건강 중재)에 대해 실험되었으며, 그 결과 사실성에서 최첨단 지식 지향 챗봇보다 최대 26.6% 더 높은 사실성을 보였습니다. 또한 설득력에서는 5점 척도에서 기존 방법보다 0.6점 더 높은 평가를 받았습니다. 이러한 성과는 PersuaBot이 도메인 적응적임을 시사합니다. 특히, 책임 있게 사용될 경우, 설득형 챗봇은 긍정적인 사회 변화를 유도할 수 있습니다.



### Integrating Randomness in Large Language Models: A Linear Congruential Generator Approach for Generating Clinically Relevant Conten (https://arxiv.org/abs/2407.03582)
- **What's New**: 이번 연구는 교육 및 콘텐츠 생성 응용 프로그램에서 언어 모델로부터 다양한 고품질 출력을 생성하기 위한 새로운 접근법을 제안합니다. 기존 문제인 진정한 무작위성과 중복 회피를 해결하기 위해 Linear Congruential Generator(LCG) 방법을 사용하여 체계적인 사실 선택을 실시하였습니다. 이를 통해 GPT-4o 모델로 임상적으로 관련 있는, 비네트 스타일의 출력을 생성할 수 있었습니다. 14회의 라운드를 통해 98개의 고유한 출력을 생성하며, LCG의 효과성을 입증했습니다.

- **Technical Details**: LCG는 널리 사용되는 의사 난수 생성기(PRNG) 알고리즘입니다. 이 알고리즘은 선형 재귀 원리를 기반으로 숫자의 시퀀스를 생성합니다. 이번 연구에서는 LCG가 특정 매개변수로 설정되었습니다: Seed (12345), Multiplier (1103515245), Increment (12345), Modulus (2^31). 이 매개변수들은 시퀀스의 일관성을 유지하기 위해 필수적입니다. 연구에서는 위 매개변수를 사용하여 100개의 관련 사실 풀에서 무작위로 사실을 선택하였고, 이 사실들을 이용하여 GPT-4o 모델로 고품질의 MCQs(Multiple-Choice Questions)를 생성했습니다.

- **Performance Highlights**: 총 14회의 라운드를 통해 98개의 고유한 MCQs를 생성하였으며, 이는 LCG가 중복을 피하면서도 다양한 콘텐츠를 생성할 수 있음을 보여줍니다. 각 라운드마다 고유한 질문 세트를 보장하기 위해 사실 선택 과정에서 사용된 매핑은 모든 라운드를 통해 관리되고 검사되었습니다. 이를 통해 생성된 질문들이 임상적으로 관련 있고 학습자에게 높은 교육적 가치를 제공하는 것으로 나타났습니다.



### Core: Robust Factual Precision Scoring with Informative Sub-Claim Identification (https://arxiv.org/abs/2407.03572)
- **What's New**: 최근 논문에서는 대형 언어 모델(Large Language Models, LLMs)의 사실 정확도를 평가하는 메트릭인 FActScore가 자명한 반복적 주장으로 쉽게 조작될 수 있음을 지적하고 이를 해결하기 위한 새로운 구성 요소인 Core를 소개합니다.

- **Technical Details**: 기존의 'Decompose-Then-Verify' 프레임워크는 생성된 텍스트를 자연어 하위 주장으로 분해한 후, 각 하위 주장에 대해 사실 여부를 평가하는 두 단계를 거칩니다. 하지만 반복적이거나 정보가 부족한 주장을 포함함으로써 점수를 인위적으로 높일 수 있는 문제점이 있습니다. 이를 해결하기 위해 Core는 하위 주장을 고유성과 정보성에 따라 필터링하여 선택합니다. 각 하위 주장의 불확실성 또는 놀라움을 가중치로 부여하여 최적의 하위 주장 집합을 선택하는 방식입니다.

- **Performance Highlights**: Core를 통해 개선된 메트릭은 기존의 FActScore 대비 반복적이고 정보가 부족한 주장에 대한 방어력이 훨씬 강력함을 입증했습니다. 본 논문에서는 전기(biography) 생성 작업에서 Core의 효과를 다양한 Decompose-Then-Verify 메트릭과 비교하여 보여주었습니다.



### On Evaluating Explanation Utility for Human-AI Decision Making in NLP (https://arxiv.org/abs/2407.03545)
Comments:
          9 pages main, 7 pages references, 32 pages appendix

- **What's New**: 이 논문은 NLP(자연어 처리)에서 설명 가능성이 실질적인 인사이트를 제공하지 못할 수 있다는 논란에 답하기 위해, 실제 응용 분야에서의 설명 가능성을 평가하는 데 필요한 기준을 제시합니다. 특히, 인공지능과 인간의 협업 상황에서의 설명 유효성 평가를 위한 데이터셋 선택 기준을 새롭게 확립하고, 법적 주장을 검증하는 과제를 예시로 선정하여 설명 유효성 연구를 수행했습니다.

- **Technical Details**: 논문은 우선 기존의 평가 지표를 검토하고, 설명 유효성을 평가하기 위한 적절한 데이터셋 기준을 수립했습니다. 50개 이상의 데이터셋을 검토한 결과, 4개의 데이터셋이 고위험 상황에서의 설명 유효성을 평가하기에 적합하다는 결론을 내렸습니다. Flan-T5-3B 모델을 파인튜닝(finetuning)하여 최신 모델의 성능을 다시 평가하고 인간-AI 팀을 연구하는 것의 중요성을 강조했습니다.

- **Performance Highlights**: 법적 주장 검증 과제에서의 설명 유효성을 연구한 결과, 모델의 예측과 신뢰도를 제공하더라도 인간과 AI의 협업 성능 향상에는 한계가 있음을 발견했습니다. 다만, 설명이 모델의 예측을 사전에 확인해주는 deferral 모델을 사용하는 새로운 접근법이 효과적일 가능성이 있음을 제안했습니다. 이를 통해 미래의 연구 방향으로 설명 기법의 개선과 새로운 협업 모델의 개발 가능성을 제시했습니다.



### Social Bias in Large Language Models For Bangla: An Empirical Study on Gender and Religious Bias (https://arxiv.org/abs/2407.03536)
- **What's New**: 이 연구는 Large Language Models (LLMs)에서 Bangla 언어의 사회적 편향을 평가한 최초의 시도라는 점에서 독창적입니다. 기존 영어 편향 평가 연구는 풍부하지만, Bangla에 대한 연구는 매우 드뭅니다. 이 연구는 Bangla 언어에서 성별 편향과 종교적 편향을 탐구하는 두 가지 주요 목표를 가지고 있습니다.

- **Technical Details**: 연구는 두 가지 프로빙 기법(probing techniques)을 사용하여 편향을 탐지합니다: 템플릿 기반 접근법(template-based approach)과 자연어 기반 접근법(naturally sourced approach). 템플릿 기반 접근법은 성별이나 종교적 정체성을 예측하기 위해 유형화된 템플릿을 사용하고, 자연어 기반 접근법은 실생활 시나리오를 통해 LLM의 편향을 평가합니다. 또한, 새로운 데이터셋을 생성하여 편향 측정의 기준을 마련하였습니다.

- **Performance Highlights**: 연구 결과, Bangla 언어에 대해 조사된 LLM들은 의미 있는 편향을 보여주었으며, 이는 향후 편향 제거(de-biasing) 작업의 필요성을 강조합니다. 실험은 Llama3-8b, GPT-3.5-Turbo, GPT-4 모델을 사용하여 수행되었으며, 주요 평가 지표로 Disparate Impact(DI)를 활용하였습니다.



### UnSeenTimeQA: Time-Sensitive Question-Answering Beyond LLMs' Memorization (https://arxiv.org/abs/2407.03525)
- **What's New**: 이번 논문에서는 UnSeenTimeQA라는 새로운 시계열 질문-응답 (time-sensitive question-answering, TSQA) 벤치마크를 소개합니다. 전통적인 TSQA 벤치마크와 달리, UnSeenTimeQA는 사실적이고 웹 검색이 가능한 질문을 피하고 있습니다. 대신 현실 정보와 분리된 시간 기반 이벤트 시나리오를 제공하여, 대형 언어 모델들(large language models, LLMs)이 학습 단계에서 습득한 지식과는 별개로 진정한 시간적 추론을 할 수 있게 하는 것이 목표입니다.

- **Technical Details**: UnSeenTimeQA는 현실 세계의 사실 정보를 분리하고 시간에 민감한 이벤트 시나리오를 제공합니다. 이는 기존의 사실적 정보에 기반한 접근법과 다르기 때문에 LLMs가 기존의 지식에 의존하지 않고 독립적인 시간적 추론 능력을 개발하도록 요구합니다. 이러한 시나리오를 통해 모델들이 시간 기반 논리를 수행하는데 얼마나 어려움을 겪는지 평가하게 됩니다.

- **Performance Highlights**: 논문에서는 6개의 오픈 소스 LLMs(2B에서 70B 사이의 크기)와 3개의 폐쇄 소스 LLMs를 평가했습니다. UnSeenTimeQA의 질문들은 모델에게 상당한 도전 과제를 제시하며, 이는 모델들이 복잡한 시간적 추론 시나리오를 처리하는 데 어려움을 겪음을 보여줍니다. 추가적으로, 시간에 민감한 질문에 대한 모델들의 성능을 분석하는 다양한 결과도 제시하고 있습니다.



### Improving LLM Abilities in Idiomatic Translation (https://arxiv.org/abs/2407.03518)
- **What's New**: 이번 연구는 대형 언어 모델(LLM)인 NLLB와 GPT가 겪는 관용어(관용구) 번역 문제를 해결하기 위한 방법을 제시합니다. 특히, 기존의 지식 기반(knowledge base) IdiomKB를 확장하여 목표 언어에 해당하는 관용어를 찾는 방법을 사용해 번역 충실도를 향상시키는 것을 목표로 하고 있습니다. 이 연구는 또한 문화적인 뉘앙스를 유지하면서 번역된 텍스트가 원래의 의도와 감정적 공명을 유지하도록 보장함으로써 더 나은 문화 간 소통을 촉진하고자 합니다.

- **Technical Details**: 번역 방법은 두 가지가 있습니다. 첫 번째 방법은 SentenceTransformers 모델을 사용하여 원본 언어와 목표 언어의 관용어 의미 사이의 코사인 유사도(cosine similarity) 점수를 계산하고 가장 적합한 관용어를 선택하는 방식입니다. 두 번째 방법은 LLM이 번역에 사용할 목표 언어의 관용어를 찾아내는 방식입니다. 기본 비교 대상으로는 추가 정보를 제공하지 않은 직접 번역이 사용되었습니다. 실험은 영어 -> 중국어, 중국어 -> 영어 번역에 대해 인공지능 모델 GPT4o를 통해 인류 평가를 수행했습니다.

- **Performance Highlights**: 코사인 유사도 조회 방법(Cosine Similarity Lookup method)은 다른 모든 방법들보다 뛰어난 성능을 보였습니다. 이를 바탕으로, 한정된 자원의 우르두어 데이터셋을 개발하여 우르두어 관용어와 그 번역을 포함시킬 수 있었습니다. 비록 데이터셋의 한계가 있었지만, 코사인 유사도 조회 방법은 향후 언어 장벽을 극복하고 다양한 문학 작품을 탐구하는데 잠재력을 보이고 있습니다.



### Collaborative Quest Completion with LLM-driven Non-Player Characters in Minecraf (https://arxiv.org/abs/2407.03460)
Comments:
          Accepted at Wordplay workshop at ACL 2024

- **What's New**: 최근 논문에서는 대형 언어 모델(Large Language Models, LLM)을 이용해 비디오 게임 내 비플레이어 캐릭터(NPC)를 구동하는 연구가 주목받고 있습니다. 특히, 주로 Minecraft 게임을 통해 이뤄진 실험에서 인간 플레이어가 GPT4 기반 NPC와 협력하여 게임 내 목표를 달성하는 방법을 탐구했습니다.

- **Technical Details**: Minecraft 내 미니게임을 디자인하고 두 명의 GPT4 기반 NPC와 인간 플레이어가 협력하는 퀘스트를 마련했습니다. 이 NPC들은 대화 능력을 갖추고 있으며, 자원 채굴이나 몬스터 공격 등의 제한적인 게임 내 행동도 수행할 수 있습니다. 프롬프트 디자인 시 NPC에게 고유의 페르소나와 백스토리를 추가하여 협력 경험을 극대화하고, 주기적으로 생성되는 서브 목표를 통해 플레이어가 게임 목표에서 벗어나지 않도록 유도했습니다.

- **Performance Highlights**: 28명의 Minecraft 플레이어를 대상으로 사용자 연구를 수행한 결과, 여러 협력 행동 패턴이 관찰되었습니다. NPC는 게임 관련 질문에 답변하고, 자원 채굴 또는 몬스터 공격과 같은 인게임 작업에서 플레이어를 지원하는 데 도움이 되었습니다. 흥미롭게도, NPC가 시각적 또는 게임 상태 이해가 부족한 상황에서 인간 플레이어가 관련 정보를 전달하여 NPC를 도우는 상호 보완적 협력도 나타났습니다. 이러한 결과는 언어 모델만으로 구동되는 NPC의 한계를 극복할 수 있는 방법을 시사합니다.



### XferBench: a Data-Driven Benchmark for Emergent Languag (https://arxiv.org/abs/2407.03456)
Comments:
          15 pages, 5 figures

- **What's New**: 이 논문에서는 데이터 기반 방법을 이용하여 창발 언어(emergent languages)의 전반적인 품질을 평가하기 위한 벤치마크(benchmark)를 소개합니다. 특히, 창발 언어의 '품질'이라는 개념을 딥러닝(deep learning) 프레임워크 내에서 인간 언어와의 유사성으로 해석합니다.

- **Technical Details**: 이 벤치마크는 창발 언어를 인간 언어로 이루어진 다운스트림 NLP 작업의 사전학습(pretraining) 데이터로 사용하여 측정합니다. 다운스트림 성능이 좋을수록 창발 언어의 품질이 더 우수하다고 평가합니다. 이 벤치마크는 평가할 창발 언어의 발화(utterances) 텍스트 파일만 필요로 하는 간편한 Python 패키지로 구현되었습니다.

- **Performance Highlights**: 마지막으로, 우리는 인간 언어, 합성 언어(synthetic language), 및 창발 언어의 기준선을 사용하여 벤치마크의 타당성을 실증적으로 테스트했습니다.



### Missed Causes and Ambiguous Effects: Counterfactuals Pose Challenges for Interpreting Neural Networks (https://arxiv.org/abs/2407.04690)
- **What's New**: 최근 연구는 신경망의 해석 가능성(interpretability) 연구가 반사실적(counterfactual) 인과성 이론에 의존하고 있으며, 이러한 접근 방식이 특정 인과 관계의 발견을 놓치고 결과를 체계적으로 왜곡할 가능성이 있음을 제기하고 있습니다.

- **Technical Details**: 반사실적 이론은 특정 모델 구성 요소의 활성화(activation) 또는 입력 값을 변경한 후 출력 로짓(logits) 또는 행동의 변화를 관찰하는 방법을 기반으로 합니다. 그러나 이 접근 방식은 독립적으로 충분한 다중 원인을 효과적으로 포착하지 못하고, 반사실적 의존성이 전이성을 가지지 않기 때문에 인과 그래프(causal graph)를 추출하고 해석하는 데 문제를 야기할 수 있습니다.

- **Performance Highlights**: 이 연구는 반사실적 이론의 두 가지 주요 문제점, 즉 과다결정(overdetermination)과 비전이성(non-transitivity)을 지적하면서, 신경망의 해석 가능성 연구에 중요한 함의를 가지고 있습니다. 이를 통해 향후 연구에서 보다 신뢰할 수 있고 인과적으로 엄격한 방법을 개발할 필요성을 강조합니다.



### Rethinking Visual Prompting for Multimodal Large Language Models with External Knowledg (https://arxiv.org/abs/2407.04681)
- **What's New**: 최근 몇 년 동안 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)은 방대한 고품질 이미지-텍스트 데이터셋으로 훈련되어 이미지 이해 능력이 크게 향상되었습니다. 하지만 텍스트로 세밀하거나 공간적으로 밀도가 높은 정보를 명확히 전달하는 데 어려움이 있어, MLLMs가 상세하거나 지역화된 시각적 요소를 이해하는 데 제한이 있습니다. 이 논문에서는 Retrieval-Augmented Generation(RAG) 개념에서 영감을 받아, 전문 비전 모델(e.g., 인스턴스 세그멘테이션/OCR 모델)로부터 얻은 세밀한 외부 지식을 MLLMs에 통합하는 새로운 시각적 프롬프트 접근 방식을 제안합니다.

- **Technical Details**: 제안된 접근 방식은 다른 연구와 달리 외부 지식을 텍스트 프롬프트로 변환하지 않고, 세밀한 지식을 직접 공간적 임베딩 맵(spatial embedding map)으로 삽입하여 시각적 프롬프트로 사용하는 것입니다. 이를 위해 파노픽 세그멘테이션 모델 및 OCR 탐지 모델을 활용하여 픽셀 수준의 텍스트 임베딩을 생성하고, 이를 지역화된 문맥 정보로 사용합니다. 이어서 확보된 문맥 정보를 시각적 프롬프트에 통합하여 다양한 MLLMs (e.g., LLaVA, Mipha)에 적용하였습니다.

- **Performance Highlights**: 실험을 통해 제안된 방식은 LLaVA-1.5 데이터셋을 활용해 기존의 LLaVA-1.5, Qwen-VL 등 7B 및 13B 모델보다 성능이 우수하다는 것을 입증했습니다. 특히 추가 훈련 데이터 없이 3억 개의 파라미터로 다양한 벤치마크에서 뛰어난 성능을 보였습니다.



### Lost in Translation: The Algorithmic Gap Between LMs and the Brain (https://arxiv.org/abs/2407.04680)
- **What's New**: 언어 모델(Language Models, LMs)이 다양한 언어 과제에서 놀라운 성과를 거두고 있지만, 이들이 인간의 언어 처리 방식과 어떻게 관련되는지에 대해서는 명확하지 않습니다. 이 논문은 LMs와 인간 뇌의 관계를 분석하는 데 있어서 입력-출력 행동뿐만 아니라 내부 프로세스를 비교하고 조사하는 중요성을 강조합니다. 신경과학의 통찰력을 바탕으로 생물학적으로 더 그럴듯한 언어 모델을 개발하는 방법을 탐구하고, 효율성 제약의 필요성을 강조합니다.

- **Technical Details**: 이 논문은 Marr의 정보 처리 시스템에 관한 세 가지 분석 수준 - 계산적 수준(Computational level), 알고리즘적 수준(Algorithmic level), 구현 수준(Implementational level) - 을 바탕으로 언어 모델과 인간 뇌의 비교를 시도합니다. 또한, 매커니즘 해석(메커니즘 인터프리터빌리티, mechamism interpretability)과 인과적 개입(causal interventions) 같은 접근 방식을 통해 LMs의 내부 프로세스를 조사하고 이를 뇌와 비교하고자 합니다.

- **Performance Highlights**: 최근 연구에 따르면, LMs의 계층별 활성화와 뇌의 언어 네트워크 간에 상관 관계가 발견되었다고 합니다. 그러나 단순한 입력-출력 매핑을 넘어서는 알고리즘적 유사성을 주장하기 위해서는 내부 동역학과 LMs의 인과 구조를 조사해야 합니다. 또한, LMs의 개별 뉴런이 다양한 기능을 복합적으로 처리하는 '다의성(polysemanticity)'이 문제로 대두되고 있으며, 이는 생물학적 신경 시스템과도 유사하다고 합니다.



### Pretraining End-to-End Keyword Search with Automatically Discovered Acoustic Units (https://arxiv.org/abs/2407.04652)
Comments:
          Interspeech 2024. KWS code at: this https URL AUD code at this https URL

- **What's New**: 기존의 음성 인식(Auto Speech Recognition, ASR) 기반 키워드 검색(KWS) 방식의 대안으로 제안된 엔드-투-엔드(End-to-End, E2E) KWS 모델은 구조를 단순화하지만, 일반적으로 검색 성능이 떨어지는 단점이 있습니다. 본 연구에서는 전사되지 않은 데이터(Untranscribed data)를 활용한 사전 학습(pretraining) 방법을 제안합니다. 이 방법은 음향 유닛 탐색(acoustic unit discovery, AUD)을 통해 전사되지 않은 데이터에서 이산 유닛(discrete units)을 추출하고, 이를 학습하여 음성 내의 해당 유닛들의 위치를 찾아서 KWS 모델을 사전 학습하는 것입니다.

- **Technical Details**: 본 연구는 [12]의 E2E KWS 모델을 기반으로 합니다. 이 모델은 두 개의 인코더(encoder)로 구성되며, 하나는 쿼리(query) 인코더로 문자의 시퀀스로 구성된 쿼리를 벡터로 변환하고, 다른 하나는 문서 인코더로 음성을 벡터로 변환합니다. 제안된 방법에서는 비전사 데이터 세트에 대해 H-SHMM (Hierarchical Subspace Hidden Markov Model)을 사용하여 이산 음향 유닛을 추출한 후, 이를 의사 쿼리로 사용하여 모델을 사전 학습합니다. 이후 소량의 전사 데이터를 사용하여 모델을 미세 조정(finetuning)합니다.

- **Performance Highlights**: 실험 결과, AUD를 기반으로 한 사전 학습은 KWS 성능을 상당히 향상시켰으며, 성능 향상은 사용된 AUD 시스템의 음성 대응 품질과 상관관계가 있음을 확인했습니다. 특히, MFCC 특징(Mel-Frequency Cepstral Coefficients)을 사용한 AUD 시스템이 우수한 성능을 보였고, pretrained transformer 특징을 입력으로 사용했을 때도 여전히 성능 향상이 나타났습니다.



### Speculative Speech Recognition by Audio-Prefixed Low-Rank Adaptation of Language Models (https://arxiv.org/abs/2407.04641)
Comments:
          Interspeech 2024

- **What's New**: 이번 논문에서는 자동 음성 인식(ASR)에 특화된 추정 기능을 추가하여 사용자가 발화하는 도중에 미리 텍스트를 예측하는 방법을 연구하고 있습니다. 특히, RNN-Transducer 기반의 ASR 시스템과 음성-접두어(language-prefixed) 언어 모델(LM)을 결합한 모델을 제안하고 있으며, 이 모델은 다양한 ASR 데이터셋에서 효과적으로 동작하는 것으로 보입니다.

- **Technical Details**: 추정 음성 인식(SSR)을 위해 제안된 모델은 Conformer-Transformer 하이브리드 모델을 사용하며, 음성-접두어(LM)를 통해 ASR의 출력을 기반으로 예측을 수행합니다. 기존 ASR 시스템의 발화 가설을 받아서 텍스트 완성을 예측하는 방식이며, 오디오 의존적인 소프트 프롬프트를 사용하여 ASR 가설의 오류를 고려하는 점이 특징입니다. 또한, Low-Rank (LoRA) 어댑터를 통해 언어 모델을 파인튜닝하며, ASR 가설의 적절한 완료를 예측하도록 학습시킵니다.

- **Performance Highlights**: 제안된 시스템은 공공 데이터셋에서 기존 방법 대비 우수한 성능을 보여주었습니다. 또한, SSR 성능을 측정하기 위해 새롭게 제안된 지표인 Suffix Oracle Word Error Rate (SOWER)를 통해 모델의 성능을 평가하였습니다.



### Learning to (Learn at Test Time): RNNs with Expressive Hidden States (https://arxiv.org/abs/2407.04620)
- **What’s New**: 최신 Arxiv 논문에서는 RNN의 선형적 복잡성(linear complexity)과 Transformer의 표현력(expressive power)을 결합한 새로운 시퀀스 모델링 레이어를 제안합니다. 이 레이어는 테스트 시퀀스(test sequence)에 대한 훈련을 통해 업데이트되므로, 'Test-Time Training (TTT)' 레이어로 명명되었습니다. 두 가지 구현체 TTT-Linear와 TTT-MLP가 소개되었으며, 각각 선형 모델과 2층 MLP(two-layer MLP)를 숨김 상태(hidden state)로 사용합니다.

- **Technical Details**: TTT 레이어는 자가 감독 학습(self-supervised learning)을 통해 숨김 상태를 업데이트하는 새로운 클래스입니다. 이는 시퀀스 모델링에서 매우 중요한 숨김 상태를 더 높은 표현력으로 만들 수 있게 합니다. TTT-Linear와 TTT-MLP는 125M에서 1.3B 파라미터 규모로 평가되었으며, 특히 TTT-Linear는 8k 컨텍스트(context)에서 Transformer보다 빠르고 벽시계 시간(wall-clock time)에서 Mamba와 비슷한 성능을 보였습니다. TTT-MLP는 메모리 I/O에서 약간의 과제를 겪지만, 긴 컨텍스트에서 잠재력을 보여줍니다.

- **Performance Highlights**: 우리의 평가에서 TTT-Linear와 TTT-MLP는 Transformer와 현대의 RNN인 Mamba와 비교하여 동등하거나 그 이상의 성능을 보였습니다. Transformer와 유사하게, TTT 레이어는 더 많은 토큰에 따라 점점 더 낮은 perplexity를 유지할 수 있습니다. 초기 시스템 최적화를 통해 TTT-Linear는 이미 8k 컨텍스트에서 Transformer보다 빠르며, Mamba와 벽시계 시간상 유사한 성능을 보입니다. TTT-MLP는 메모리 I/O에서 도전과제를 제기하지만 긴 컨텍스트에서 큰 잠재력을 보여줍니다.



### Written Term Detection Improves Spoken Term Detection (https://arxiv.org/abs/2407.04601)
Comments:
          IEEE/ACM Transactions on Audio, Speech and Language Processing (TASLP), 2024. Code at this https URL

- **What's New**: 이번 논문에서는 기존의 단어 검색(KWS) 시스템과 비교하여 외부의 문어 텍스트를 통합할 수 있는 새로운 멀티태스크 학습 목표를 제안합니다. 이 방식은 훈련 및 인덱싱 복잡성을 높이지 않으면서, 미구조화된 문어 텍스트를 활용할 수 있게 합니다. 제안된 모델은 다양한 언어에서 검색 성능을 유의미하게 향상시킵니다.

- **Technical Details**: 제안된 방법은 음성이 아닌 문서에서도 텍스트 쿼리를 검색할 수 있도록 훈련하는 멀티태스크 학습을 포함하고 있습니다. 훈련 중에는 주어진 음성 문서에서 텍스트 쿼리를 검색하는 기본적인 훈련과 함께, 마스킹된 문서에서 텍스트 쿼리를 검색하는 보조학습을 병행합니다. 이를 통해, 순수하게 텍스트만으로 된 데이터도 KWS 모델의 학습에 활용할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 여러 언어와 도메인에서 기존 방법에 비해 검색 성능이 대폭 향상되었습니다. 또한, 제안된 멀티태스크 학습 스킴은 다중언어 사전훈련과 데이터 증강 기법과도 독립적으로 작동할 수 있으며, 결합하여 더욱 높은 성능을 달성할 수 있습니다. 도메인 적응에도 효과가 있어, 훈련 데이터와 테스트 데이터의 도메인이 일치하지 않는 경우에도 성능 향상을 보여줍니다.



### VRSD: Rethinking Similarity and Diversity for Retrieval in Large Language Models (https://arxiv.org/abs/2407.04573)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)에 대한 벡터 검색 알고리즘의 중요성을 강조하며, 유사성과 다양성을 모두 충족하는 벡터를 검색하는 새로운 접근 방식을 소개합니다. 기존의 Maximal Marginal Relevance (MMR) 알고리즘이 유사성과 다양성의 균형을 맞추기 위해 사용되었지만, 이 논문은 λ 파라미터의 변동으로 인해 최적화의 방향을 파악하기 어려운 문제를 지적합니다. 본 연구는 쿼리 벡터와 합 벡터 간의 관계를 통해 유사성과 다양성 제약을 특성화하는 새로운 방법을 제안하며, 이를 통해 유사성과 다양성을 동시에 추구하는 복잡한 최적화 문제를 해결하고자 합니다.

- **Technical Details**: 논문은 선택된 벡터의 합 벡터와 쿼리 벡터 간의 유사성을 최대화하는 새로운 조합 최적화 문제를 정의합니다. 이 문제는 NP-complete으로 판명되었으며, 이는 유사성과 다양성을 동시에 충족하는 것이 이론적으로 매우 어려운 문제임을 보여줍니다. 이를 해결하기 위해, 파라미터가 필요 없고 최적화 목표가 명확한 휴리스틱 알고리즘인 Vectors Retrieval with Similarity and Diversity (VRSD)를 제안합니다. 이 알고리즘은 MMR보다 시간 복잡도가 낮으며, 다양한 데이터셋에서 MMR을 능가하는 성능을 보였습니다.

- **Performance Highlights**: 실험 결과, VRSD 알고리즘은 여러 데이터셋에서 MMR 알고리즘을 상당히 능가하는 성능을 보였습니다. 이는 VRSD가 유사성과 다양성을 동시에 충족시키는 검색 문제 해결에 더 효과적임을 입증합니다.



### Controlling Whisper: Universal Acoustic Adversarial Attacks to Control Speech Foundation Models (https://arxiv.org/abs/2407.04482)
- **What's New**: 최근 연구는 다양한 음성 처리 작업을 수행할 수 있는 음성 인식 기반 시스템이나 오디오 프롬프트를 사용하는 대형 언어 모델(LLM)이 점점 인기를 끌고 있음을 보여줍니다. 특히, 이러한 모델의 중요한 특성 중 하나는 자동 음성 인식(ASR) 외에도 적절한 프롬프트를 통해 다른 작업을 수행할 수 있다는 점입니다. 예를 들어, OpenAI의 Whisper 모델은 음성 전사와 음성 번역 모두를 수행할 수 있습니다. 이번 연구에서는 이러한 유연성이 시스템이 모델 제어 적대적 공격에 취약할 수 있음을 시사합니다.

- **Technical Details**: 본 연구에서는 Whisper와 같은 다중 작업 ASR 모델이 모델 제어 적대적 공격에 얼마나 취약한지 설명합니다. 연구 결과, 특정 오디오 입력을 변경하여 시스템의 동작을 조작할 수 있음을 보여줍니다. 특히, 짧은 보편적 적대적 음향 구간을 입력 음성 신호에 추가하여 Whisper를 항상 음성 번역 작업을 수행하도록 설정할 수 있음을 입증했습니다. 이러한 공격은 보통 3초 이내의 짧은 음향 구간을 포함하며, 텍스트 디코더 프롬프트에 접근할 수 없더라도 공격이 효과적임을 증명했습니다.

- **Performance Highlights**: 4개 언어에서 실험한 결과, 보편적 음향 공격 구간은 거의 모든 테스트 샘플에서 Whisper의 동작을 일관되게 조작할 수 있음을 보여주었습니다. 또한, 특정 샘플에 대해 매우 성공적이거나 전혀 효과가 없다는 이분법적 패턴이 관찰되었습니다. 이 연구는 다중 작업 음성 인식 시스템이 새로운 형태의 적대적 공격에 얼마나 취약한지를 잘 보여줍니다.



### EventChat: Implementation and user-centric evaluation of a large language model-driven conversational recommender system for exploring leisure events in an SME contex (https://arxiv.org/abs/2407.04472)
Comments:
          27 pages, 3 tables, 5 figures, pre-print manuscript

- **What's New**: 이번 연구는 대규모 언어 모델(Large Language Models, LLM)을 사용한 대화형 추천 시스템(Conversational Recommender Systems, CRS)의 설계와 성능에 대해 논의합니다. 특히, 중소기업(SME) 환경에서 이를 구현하고 평가하는 데 중점을 두고 있어 기존 연구와 차별화됩니다.

- **Technical Details**: LLM 기반의 CRS는 RAG(Retrieval-Augmented Generation) 기술 내에서 고급 LLM을 랭커(rankers)로 사용하는 구조를 가지고 있습니다. 이와 함께, 신속하게 변화하는 분야에서 평가의 재현성을 높이기 위해 개정된 짧은 형태의 ResQue 모델을 개발하였습니다.

- **Performance Highlights**: 이 시스템은 사용자 경험 측면에서 높은 추천 정확도(85.5%)를 기록하였지만, 지연 시간(latency)과 비용 문제가 비즈니스 실행 가능성을 저해하는 도전 과제로 나타났습니다. 중간 비용은 상호작용당 $0.04이며, 지연 시간은 5.7초로 나타났습니다. 이러한 비용 중 큰 부분은 고급 LLM을 사용함에 기인하며, ChatGPT를 기반으로 한 프롬프트(Prompt) 기반 학습 접근법만으로는 만족스러운 품질을 얻기 어려웠습니다.

- **Business Implications**: SME가 LLM 기반의 CRS를 배포할 때 전략적 고려 사항을 강조합니다. 특히, 비용 효율성과 응답 시간은 사용자 친화적이고 경제적으로 실행 가능한 시스템을 달성하는 데 중요한 요소로 평가됩니다.



### Are Large Language Models Strategic Decision Makers? A Study of Performance and Bias in Two-Player Non-Zero-Sum Games (https://arxiv.org/abs/2407.04467)
Comments:
          8 pages (19 with appendix), 6 figures in the main body (4 in the appendix), 4 tables in the main body

- **What's New**: 이 연구는 대형 언어 모델(Large Language Models, LLMs)의 전략적 능력을 탐구했습니다. 게임 이론(Game theory)을 사용하여 LLMs가 다른 에이전트(agent)와 상호 작용할 때의 의사 결정 능력을 평가했습니다. 연구는 LLMs가 특정 프롬프트로 문제를 해결할 수 있지만, 문제 설정이나 프롬프트가 변경될 때 실패한다고 지적합니다. 이 연구에서는 LLMs가 Stag Hunt와 Prisoner Dilemma 같은 전략적 게임에서 어떻게 행동하는지 분석했습니다.

- **Technical Details**: 테스트된 최신 LLMs는 다음과 같은 체계적인 편향(systemsatic biases)을 보였습니다: (1) 위치 편향(positional bias), (2) 보상 편향(payoff bias), (3) 행동 편향(behavioural bias). LLMs의 성능은 게임 구성(game configuration)이 이러한 편향과 일치하지 않을 때 떨어지는 것으로 나타났습니다. 예를 들어, GPT-4o의 평균 성능은 일치하지 않을 경우 34% 감소했습니다. 또한, '더 크고 최신이 더 낫다'는 현재의 추세는 이러한 상황에서는 성립하지 않습니다. 여기서 GPT-4o는 가장 큰 성능 하락을 겪었습니다.

- **Performance Highlights**: 성능은 올바른 행동을 선택하는 기준으로 평가되었습니다. 이는 프롬프트된 두 플레이어의 선호 행동과 일치하는 행동을 의미합니다. 일치성은 LLM의 편향이 올바른 행동과 일치하는지 여부를 나타냅니다. 체인 오브 사고 프롬프트(chain-of-thought prompting)가 대부분의 모델에서 편향의 영향을 줄여주지만 근본적인 문제를 해결하지 못한다고 결론지었습니다.



### Waterfall: Framework for Robust and Scalable Text Watermarking (https://arxiv.org/abs/2407.04411)
- **What's New**: Waterfall는 텍스트의 지적 재산권(IP)을 보호하기 위한 혁신적인 텍스트 워터마킹 프레임워크를 소개합니다. 이 기법은 향상된 강건성(robustness)과 확장성(scalability)을 제공하며, 여러 유형의 텍스트(예: 기사, 코드)와 다국어를 지원합니다. 특히, LLMs를 패러프레이저로 사용하여 기존 IP를 보호하는 새로운 접근 방식을 제안했습니다.

- **Technical Details**: 이 프레임워크는 LLM을 패러프레이저(paraphrasers)로 사용하여 워터마킹하는 방식을 도입하며, 어휘 순열(vocab permutation)과 토큰 공간 내 새로운 직교 워터마킹 교란(perturbation) 방법을 결합합니다. 이를 통해 높은 확장성, 강한 검증 가능성(verifiability) 및 원본 텍스트의 충실도를 유지하며, 수백만 명의 사용자를 지원할 수 있습니다.

- **Performance Highlights**: Waterfall은 현재의 최첨단(SOTA) 텍스트 워터마킹 방법보다 더 나은 확장성, 강건한 검증 가능성 및 계산 효율성을 구현했음을 실험적으로 입증했습니다. 또한, 다양한 응용 프로그램, 특히 LLM 데이터 원천 확인(data provenance)을 위해 요구되는 조건을 충족했으며, 프로그래밍 코드의 워터마킹에도 직접적으로 적용할 수 있음을 보여주었습니다.



### Jailbreak Attacks and Defenses Against Large Language Models: A Survey (https://arxiv.org/abs/2407.04295)
- **What's New**: 이번 논문에서는 'jailbreak 공격(jailbreak attacks)' 및 방어 방법에 대해 포괄적이고 상세한 분류 체계를 제안합니다. LLM(대형 언어 모델)의 취약점을 악용하여 악의적인 응답을 유도하는 'jailbreak' 문제는 현재까지 계속해서 진화하고 있으며, 이에 따른 안전 정렬(safety alignment) 측정 방법도 발전하고 있습니다.

- **Technical Details**: 논문은 공격 방법을 타겟 모델의 투명성에 따라 블랙박스(black-box) 공격과 화이트박스(white-box) 공격으로 나누고, 각각의 방어 방법을 프롬프트 수준(prompt-level)과 모델 수준(model-level) 방어로 분류합니다. 구체적인 공격 방법으로는 Gradient-based, Logits-based, Fine-tuning-based 공격이 있으며, 방어 방법에서는 프롬프트 재작성(prompt rewriting), 저자원 언어(low-resource languages) 등을 사용합니다.

- **Performance Highlights**: 기존 연구의 다양한 실험을 통해 제안된 공격 및 방어 메서드 간의 관계를 강조하고, 특정 방어 방법이 다른 유형의 공격에도 효과적일 수 있음을 확인했습니다. 또한 현재 평가 방법에 대한 조사도 진행하여 여러 관점에서 비교 분석하였습니다. 이 조사 결과는 LLM의 악의적 공격으로부터의 보호를 위한 향후 연구와 실무적 구현에 영감을 줄 것입니다.



### Orchestrating LLMs with Different Personalizations (https://arxiv.org/abs/2407.04181)
- **What's New**: 이 논문은 개인 맞춤형 휴먼 피드백(Reinforcement Learning from Personalized Human Feedback, RLPHF)을 활용하여 대형 언어 모델(LLM)을 개별적인 인간의 선호도와 조화시키는 새로운 접근 방식을 제안합니다. 기존에는 여러 선호도를 다루기 위해 개별 전문가 모델을 만들어야 했지만, 이 연구에서는 각 토큰 수준에서 전문가 모델의 출력을 병합하는 새로운 방법을 제시합니다.

- **Technical Details**: 제안된 방법인 Mixture of Preference Experts(MoPE)는 미리 훈련된 전문가 LLM들이 각자의 토큰 출력을 토대로 주어진 문맥과 선호도 설명에 따라 가중치를 동적으로 조절합니다. 이를 위해 경량의 Preference Control Model(PCM)을 훈련시켜 다음 토큰 예측 가중치를 계산합니다. 이 방식은 블랙박스 접근법으로 전문가 모델의 가중치에 접근할 필요 없이 상위 출력 로그잇(logit)만 필요합니다. 훈련 과정에서는 각 선호도 차원을 위한 보상 모델을 만들고 온라인 강화학습(REBEL)을 활용하여 선호도에 맞게 PCM을 학습시킵니다.

- **Performance Highlights**: MoPE는 Tulu-7B LLM을 사용한 실험에서 기존의 선호도 병합 기법들보다 더 높은 성능을 보였습니다. 다양한 선호도 조합에서 평균적으로 더 높은 쌍별 승률(pairwise win-rate)을 기록했습니다. 이 방법은 선호도가 변경될 때 재훈련이 필요 없으며, 병렬 처리가 용이하다는 장점이 있습니다.



### MiniGPT-Med: Large Language Model as a General Interface for Radiology Diagnosis (https://arxiv.org/abs/2407.04106)
- **What's New**: AI 분야의 최신 진보 덕분에 MiniGPT-Med라는 혁신적인 모델이 탄생했습니다. 이 모델은 대규모 언어 모델(LLMs)을 기반으로 한 비전-언어 모델로, 특히 의료 이미징 분야에 최적화되었습니다. X-ray, CT 스캔, MRI 등 다양한 영상 기법에서 탁월한 성능을 보이며, 의료 보고서 생성, 시각적 질문 응답(VQA), 질병 식별 등의 작업을 수행할 수 있습니다. 특히, MiniGPT-Med는 의료 이미지와 텍스트 데이터를 통합적으로 처리하여 진단 정확도를 크게 향상시켰습니다.

- **Technical Details**: MiniGPT-Med의 아키텍처는 세 가지 주요 구성 요소로 이루어져 있습니다: ▶ 시각 백본(Vision Backbone), ▶ 선형 프로젝션 레이어(Linear Projection Layer), ▶ 대규모 언어 모델(Large Language Model, LLM). 시각 백본으로는 높은 해상도의 방사선 이미지를 처리할 수 있는 EVA를 사용하였으며, 언어 모델로는 LLaMA2-chat (7B)를 채택했습니다. 또한, MiniGPT-v2의 아키텍처를 채택하여, 시각 토큰을 언어 모델의 피처 스페이스로 매핑하는 방식으로 시각-언어 정렬을 개선시켰습니다.

- **Performance Highlights**: MiniGPT-Med는 다양한 기준치 모델들을 상대로 탁월한 성능을 보였습니다. 특히 의료 보고서 생성 분야에서는 기존 최고 모델을 19% 상회하는 정확도를 기록하였으며, BERT-Sim에서는 19%, CheXbert-Sim에서는 5.2% 더 높은 성능을 보였습니다. 이러한 결과는 MiniGPT-Med가 의료 이미지 분석과 보고서 생성 등 다채로운 의료 비전-언어 작업에서 강력한 성능을 발휘함을 입증합니다.



### Solving Zebra Puzzles Using Constraint-Guided Multi-Agent Systems (https://arxiv.org/abs/2407.03956)
- **What's New**: 이 연구에서는 대형 언어 모델(LLMs)과 표준 정리 증명기를 결합한 새로운 다중 에이전트 시스템 ZPS를 도입했습니다. 이를 통해 복잡한 자연어 문제를 체계적이고 관리 가능한 부분으로 나누어 풀이하는 방식을 제안합니다. 자동화된 그리드 퍼즐 채점기를 도입하여 퍼즐 풀이의 정확성을 평가하며, GPT-4를 포함한 여러 LLM에서 성능이 크게 향상되었음을 보여줍니다.

- **Technical Details**: 방법론적으로, ZPS는 로직 그리드 문제(Zebra puzzle)를 여러 에이전트를 통해 처리합니다. 먼저 LLM 에이전트가 퍼즐을 소문제들로 분해합니다. 각 소문제는 다시 자연어 단서를 SMT-LIB 형식으로 번역하여 정리 증명기(SMT solver)에서 해결됩니다. 해결된 출력은 다음 번역을 개선하기 위한 피드백으로 사용되며, 이를 통해 반복적으로 솔루션을 개선합니다. 이 과정은 수학적 평가 함수와 오류 감지 메커니즘을 결합하여 최적의 솔루션을 얻을 수 있도록 합니다.

- **Performance Highlights**: ZPS는 다양한 LLM에서의 성능을 크게 향상시켰으며, 특히 GPT-4의 경우 완벽하게 정확한 솔루션 수가 166% 증가한 결과를 보였습니다. 이 시스템은 기존 방법론에 비해 복잡한 논리 퍼즐 해결 능력을 크게 향상시켰음을 증명했습니다.



### Narrow Transformer: Starcoder-Based Java-LM For Desktop (https://arxiv.org/abs/2407.03941)
- **What's New**: 이번 논문은 Java 코딩 작업에 특화된 오픈 소스 코드 언어 모델인 NT-Java-1.1B를 소개합니다. 이 모델은 StarCoderBase-1.1B를 기반으로 구축되었으며, MultiPL-E Java 코드 벤치마크에서 최첨단 성능을 달성, 자체 기반 모델과 유사한 크기의 다른 모델들을 능가합니다. 대형 코드 모델은 전문 하드웨어(GPUs) 필요성을 강조하며, 소형 코드 모델 연구의 중요성을 부각시켰습니다. NT-Java-1.1B는 이러한 연구 격차를 해결하기 위해 개발되었으며, 데스크톱 배포에 이상적인 성능을 보이는 소규모 Java 코드 모델입니다.

- **Technical Details**: NT-Java-1.1B는 특별히 Java 코딩 작업을 위해 구축되었으며, StarCoderBase-1.1B를 기반으로 합니다. 데이터 전처리에는 Megatron-LM 프레임워크를 사용했으며, StarCoderBase의 GPT2BPETokenizer를 사용하여 49,152 토큰의 어휘를 활용합니다. Java 데이터셋은 87개의 파케이 파일로 구성되어 있으며, 이 파일들은 단일 파일로 변환된 후 Megatron 전처리 모듈을 통해 .bin 및 .idx 파일로 변환되었습니다. 이 파일들은 모델 훈련에 사용되었습니다.

- **Performance Highlights**: NT-Java-1.1B는 MultiPL-E Java 코드 벤치마크에서 뛰어난 성능을 발휘하였으며, 이는 Java 코딩 작업에 특화된 코드 모델의 가능성을 입증합니다. 성능 비교에서 StarCoderBase-1.1B와 다른 유사한 크기의 모델들을 능가하며, 이는 데스크톱 배포에 알맞은 작은 코드 모델 개발의 중요성을 강조합니다.



### DART: Deep Adversarial Automated Red Teaming for LLM Safety (https://arxiv.org/abs/2407.03876)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)의 취약성을 식별하기 위해 사용되는 수작업 레드 팀설정(manual Red teaming)의 비효율성과 확장성 문제를 해결하기 위해 Deep Adversarial Automated Red Teaming(DART) 프레임워크를 제안합니다. DART는 Red LLM과 Target LLM이 반복(iterative) 방식으로 깊고 동적(dynamic)으로 상호작용하며 진행되는 방식을 사용합니다.

- **Technical Details**: DART 프레임워크는 두 가지 주요 메커니즘을 포함합니다. 첫째, Red LLM은 Target LLM의 응답을 바탕으로 공격 방향을 적응적으로 조정하며, 여러 반복(iterations)을 거쳐 생성된 공격의 글로벌 다양성(global diversity)을 모니터링하여 성공적인 공격 사례를 최대한 많이 생성합니다. 둘째, Target LLM은 동적으로 변경되는 안전성 취약성을 탐구하기 위해 능동 학습 기반 데이터 선택 메커니즘(active learning based data selection mechanism)을 통해 안전성을 강화합니다.

- **Performance Highlights**: DART 프레임워크의 실험 결과, 타겟 LLM의 안전 위험이 크게 감소하는 것으로 나타났습니다. 특정 인간 평가 데이터셋(Anthropic Harmless dataset)에서 DART는 인스트럭션 튜닝(instruction-tuning) 타겟 LLM에 비해 위반 위험을 53.4% 줄였습니다. DART의 데이터셋과 코드는 곧 공개될 예정입니다.



### Meta-optimized Angular Margin Contrastive Framework for Video-Language Representation Learning (https://arxiv.org/abs/2407.03788)
Comments:
          Accepted to ECCV 2024

- **What's New**: 개선된 비디오-언어 표현 학습을 위해 Subtractive Angular Margin Contrastive Loss와 MLP-기반 샘플 가중치 조정 기능을 제안합니다. 이 모델은 정형화되지 않은 개념 분포에 적응하고, 큰 시각-언어 모델(Large Vision-Language Model, LVLM)로 생성된 비디오-텍스트 데이터를 사용해 성능을 높입니다.

- **Technical Details**: Subtractive Angular Margin Contrastive Loss는 비디오와 텍스트 샘플 간의 양성 표본 정렬을 규제하는데 사용됩니다. 또한, MLP-Parameterized Weighting Function은 데이터 샘플의 손실 값을 가중치로 변환해 모델의 훈련 초점을 동적으로 조정합니다. 이러한 접근법은 시간이 많이 소요되는 수작업 설정을 피하고, 불균형한 데이터 분포 문제 및 불완전한 정렬 문제를 해결합니다. 더욱이, LVLM을 사용해 비디오-텍스트 데이터를 증강하는 전략을 통해 데이터 다양성을 확대했습니다.

- **Performance Highlights**: 제안된 프레임워크는 MSRVTT, DiDeMo, ActivityNet, TGIF-QA-R, NExT-QA, Causal-VidQA 등 표준 데이터셋에서 최신 비디오-언어 표현 학습 방법들을 능가하는 성과를 보였습니다. 특히, 비디오 질문 응답(video question answering)과 텍스트-비디오 검색(text-video retrieval) 작업에서 우수한 성능을 입증했습니다.



### Query-oriented Data Augmentation for Session Search (https://arxiv.org/abs/2407.03720)
Comments:
          TKDE 2024

- **What's New**: 검색 세션 내 맥락 정보를 모델링하는 새로운 접근법을 제안합니다. 이 논문에서는 현재 검색 쿼리를 변화시켜 보충 훈련 쌍을 생성하고, 이를 통해 검색 로그 데이터를 더욱 풍부하게 만드는 'Query-oriented Data Augmentation' 방법을 소개합니다. 이를 통해 모델이 문서의 관련성이 세션 맥락에 따라 변화할 수 있음을 학습할 수 있습니다.

- **Technical Details**: 기존의 검색 세션 모델링은 주로 클릭된 문서와 클릭되지 않은 문서 간의 순위를 학습하는 방식이었으나, 문서와 세션 맥락 간의 대칭적인 관련성을 고려하지 못했습니다. 이를 해결하기 위해, 우리는 현재 쿼리를 주요 변경 요소로 설정하여 새로운 훈련 샘플을 생성하는 전략을 사용했습니다. 구체적으로, 쿼리의 용어 수준 변경(Term-level Modification)과 쿼리 수준 교체(Query-level Replacement)를 통해 다양한 난이도의 부정 샘플을 생성했습니다.

- **Performance Highlights**: AOL 및 Tiangong-ST와 같은 두 개의 공개 검색 로그 데이터셋에서 실험한 결과, 본 논문에서 제안한 'Query-oriented Data Augmentation' 방법을 탑재한 모델이 기존의 모델보다 성능 면에서 뛰어난 것으로 나타났습니다. 이는 우리의 접근법이 사용자 검색 패턴을 보다 포괄적으로 학습하는 데 효과적임을 입증합니다.



### BM25S: Orders of magnitude faster lexical search via eager sparse scoring (https://arxiv.org/abs/2407.03618)
Comments:
          Technical Report

- **What's New**: BM25S는 Numpy와 Scipy만을 의존하는 BM25의 효율적인 Python 구현체를 소개합니다. BM25S는 색인화 과정에서 BM25 점수를 미리 계산하여 희소 행렬로 저장함으로써 가장 널리 사용되는 Python 기반 프레임워크보다 최대 500배 빠른 성능을 달성합니다. 또한 인기 있는 상용 제품에서 사용되는 고도로 최적화된 Java 기반 구현보다도 상당한 속도 향상을 이룹니다. BM25S는 새로운 점수 이동 방법을 통해 비희소(Non-sparse) 변형에도 확장 적용할 수 있습니다.

- **Technical Details**: BM25S는 색인화 시 모든 가능한 점수를 미리 계산하고 이를 희소 행렬에 저장하는 방식을 통해 속도를 극대화했습니다. PyTorch를 사용하는 기존의 BM25-PT와 달리, BM25S는 Scipy의 희소 행렬 구현을 사용하며 행렬 곱셈이 아닌 슬라이싱과 summation을 활용합니다. 또한 Scikit-Learn의 텍스트 분할, Elastic's stopword list, 선택적으로 C 기반 Snowball stemmer 등의 간단하지만 빠른 파이썬 기반 토크나이저를 도입했습니다.

- **Performance Highlights**: BM25S는 색인화 과정에서 미래의 쿼리 토큰에 할당될 수 있는 모든 점수를 미리 계산하고 이를 희소 행렬에 저장함으로써, 기존 Python 기반 구현체 대비 최대 500배, 고도로 최적화된 Java 기반 구현체 대비 상당한 속도 향상을 달성합니다. 또, Scipy의 Compressed Sparse Column (CSC) 형식을 사용하여 효율적인 변환과 연산을 가능케 하며, 결과적으로 검색 과정에서 O(n) 복잡도의 평균 시간 복잡도를 달성했습니다.



### Learning Video Temporal Dynamics with Cross-Modal Attention for Robust Audio-Visual Speech Recognition (https://arxiv.org/abs/2407.03563)
- **What's New**: 본 연구는 오디오-비주얼 음성 인식(AVSR)에서 비디오 특성을 강화하는 방법을 제안합니다. 기존 연구들이 주로 오디오 특성 향상에 집중한 반면, 본 연구는 비디오 데이터를 통해 세 가지 시간적 동역학(temporal dynamics)을 학습하여 비디오 특성을 강화하는 데 중점을 둡니다. 또한, 크로스-모달 어텐션 모듈을 도입하여 비디오 특성을 오디오 정보로 풍부하게 만듭니다.

- **Technical Details**: 본 연구는 비디오 특성을 강화하기 위해 세 가지 시간적 동역학을 학습합니다: (1) 임의의 비디오 및 오디오 프레임 간의 컨텍스트 순서(context order), (2) 비디오 프레임의 재생 방향(playback direction), (3) 비디오 프레임의 재생 속도(playback speed). 크로스-모달 어텐션 모듈을 통해 비디오 특성에 오디오 정보를 주입하여 음성 변동성을 학습할 수 있도록 합니다. 이를 위해 비디오 스트림라인 및 오디오 스트림라인에 각각 A2V(Audio to Video) 및 V2A(Video to Audio) 크로스-모달 어텐션 구조를 도입합니다.

- **Performance Highlights**: LRS2 및 LRS3 AVSR 벤치마크에서 잡음이 많은 설정에서도 최신 성능(state-of-the-art)을 달성했습니다. 특히 LRS3 벤치마크에서는 UniVPM보다 높은 성능을 보였으며, 다양한 잡음 환경에서 안정적인 AVSR 성능을 입증했습니다. 추가로, 시간적 동역학 손실 및 크로스-모달 어텐션 구조 설계에 대한 ablation 연구를 통해 방법론의 유효성을 검증했습니다.



### Feelings about Bodies: Emotions on Diet and Fitness Forums Reveal Gendered Stereotypes and Body Image Concerns (https://arxiv.org/abs/2407.03551)
- **What's New**: 최근 연구는 Reddit의 다이어트, 피트니스 및 관련 정신 건강 문제와 관련된 46개의 토론 포럼을 분석하여 이상적인 신체 이미지와 감정 표현 간의 복잡한 상호작용을 밝혀냈습니다. 이 연구는 '얇음(Thinness)'을 이상으로 여기는 커뮤니티와 '근육(Muscularity)'을 추구하는 커뮤니티 간의 감정 및 독성(Toxicity) 표현 차이에 대해 탐구했습니다.

- **Technical Details**: 이 연구는 멤버 구조 분석과 트랜스포머 기반 언어 모델(Transformer-based Language Models)을 사용하여, 커뮤니티를 성별(Gender)과 신체 이상(Body Ideal) 축으로 투영했습니다. 이를 통해 성별, 신체 이상 및 감정 표현(interaction between gender, body ideals, and emotional expression)의 상호작용을 분석했습니다. 특히, 감정 표현과 독성 지표를 이용해 각 커뮤니티의 정신 건강 문제를 분석했습니다.

- **Performance Highlights**: 주요 발견 사항은 다음과 같습니다. 여성 중심의 커뮤니티는 일반적으로 더 부정적인 감정을 표현하는 경향이 있으며, 특히 얇음을 추구하는 커뮤니티에서 그 경향이 두드러집니다. 반면, 근육을 이상으로 하는 커뮤니티는 성별에 관계없이 상대적으로 부정적인 감정 표현이 적었습니다. 또한, 자살, 자해 및 신체 이형성(body dysmorphia)과 같은 심각한 문제를 논의하는 커뮤니티는 주로 얇음을 이상으로 하는 여성 중심 커뮤니티와 밀접하게 연관된 감정 표현 패턴을 보였습니다.



### Codec-ASR: Training Performant Automatic Speech Recognition Systems with Discrete Speech Representations (https://arxiv.org/abs/2407.03495)
Comments:
          Accepted at Interspeech 2024

- **What's New**: 최근 음성-오디오 기술에서 Discrete Speech Representation (범주형 음성 표현)이 주목받고 있습니다. 이 논문에서는 이러한 범주형 코드(Discrete Codes)를 활용하여 성능이 우수한 자동 음성 인식(ASR) 시스템을 구축하는 방법을 다양한 관점에서 분석했습니다. 특히 기존의 Encodec 대비 성능이 뛰어나며, 자가 지도 학습 모델(self-supervised models)들을 능가하는 결과를 보여줍니다.

- **Technical Details**: 논문에서는 코덱 훈련 방법(quantization schemes)과 시간 도메인(time-domain) 대 스펙트럼 도메인(spectral feature) 인코딩을 탐구했습니다. Residual Vector Quantization(RVQ)과 Finite Scalar Quantization(FSQ) 등 여러 양자화 기법을 비교하였고, 각 기법의 Encoder-Quantizer-Decoder 아키텍처를 분석했습니다. 또한 소음에 강한 ASR 트레이닝 기법도 분석하여 최적의 노이즈-로버스트(Noise-robust) 코덱 ASR 파이프라인을 제안했습니다.

- **Performance Highlights**: 제안된 코덱 기반 ASR 시스템은 같은 비트율(bitrate)에서 Encodec을 능가하며, 143개 언어를 포함한 ML-SUPERB 벤치마크에서 최신 기술(State-of-the-Art) 결과보다 우수한 성능을 입증했습니다. 특히 하드 테스트 셋 처리를 통해 Character Error Rate(CER) 21%의 성과를 달성했습니다.



### Exploring LGBTQ+ Bias in Generative AI Answers across Different Country and Religious Contexts (https://arxiv.org/abs/2407.03473)
- **What's New**: 최근 연구는 생성 AI(generative AI) 도구들이 문화적으로 더욱 민감해질 필요가 있다는 점을 강조하고 있습니다. 특히, 각 문화와 종교에 따라 다르게 인식되는 소수자(minorities)를 다루는 데 있어서 복잡성을 종종 간과하고 있습니다. 이 연구는 두 가지 생성 AI 시스템이 문화적 및 종교적 맥락 정보와 함께 등장하는 동성애 혐오(homophobic) 발언에 어떻게 대응하는지 분석했습니다.

- **Technical Details**: 연구에서 ChatGPT 3.5와 Bard를 사용하여 동성애 혐오적 발언에 대한 반응을 분석했습니다. ChatGPT 3.5는 문화상대주의(cultural relativism)의 입장을 보인데 반해, Bard는 인권(human rights)을 강조하며 LGBTQ+ 문제에 더 많은 지지를 보였습니다. 중요한 점은 두 AI 시스템 모두 주어진 맥락 정보(contextual information)에 따라 반응이 크게 달라졌다는 것입니다.

- **Performance Highlights**: 이 연구의 주요 발견은 AI 시스템이 사용자의 배경 정보에 따라 동성애 혐오적 발언에 대한 지지나 대응의 정도와 형태를 조정할 수 있다는 점입니다. 이는 생성 AI의 응답이 어떤 문화적 다양성을 수용하려면 기본적인 인권의 개념에 기반해야 한다는 사회적 및 윤리적 함의를 이해하는 데 기여할 수 있습니다.



### Prosody-Driven Privacy-Preserving Dementia Detection (https://arxiv.org/abs/2407.03470)
Comments:
          Accepted at Interspeech 2024

- **What's New**: 이 연구는 치매 감지를 위해 음성 기록에서 추출된 화자 임베딩(speaker embeddings)을 익명화하는 새로운 방식을 제안합니다. 기존 연구들과 달리, 치매 분류기를 사용하지 않고 도메인 지식을 활용하여 프러소디(prosody) 특징을 분리함으로써 화자의 신원을 보호하면서도 치매 감지 능력을 유지합니다.

- **Technical Details**: 제안된 방법론은 프러소디 기반의 개인정보 보호 기법을 사용합니다. 이는 발화 속도, 정지 시간, 불명확한 발음 등 치매 관련 특징을 분리하여 화자의 신원 정보를 포함한 임베딩에서 제거하는 방식입니다. 이를 위해 도메인 적대 학습(adversarial learning) 및 상호 정보에 기반한 셔플링(shuffling)을 사용하여, 임베딩에서 화자 관련 정보를 최소화합니다. 모델은 여러 구성 요소로 이루어지며, 특징 추출기와 프러소디 회귀기(prosody regressors)를 포함합니다.

- **Performance Highlights**: 제안된 방법은 ADReSS 데이터셋에서 화자 인식 F1-score 0.01%와 치매 감지 점수 F1-score 74%를 기록하며, 이는 치매 관련 정보를 유지하면서도 화자의 신원을 효과적으로 보호함을 입증합니다. 또한, ADReSSo 데이터셋에서도 유사한 성능을 보여주며, 생성된 음성의 자연스러움에 영향을 미치지 않습니다.



### HEMM: Holistic Evaluation of Multimodal Foundation Models (https://arxiv.org/abs/2407.03418)
Comments:
          Code available at this https URL

- **What's New**: 최근 다양한 현실 응용 프로그램에서 텍스트와 이미지, 비디오, 오디오 등 여러 감각적 모달리티를 처리할 수 있는 멀티모달 기초 모델(multimodal foundation models)에 대한 관심이 높아지고 있습니다. 본 논문에서는 이러한 멀티모달 기초 모델의 능력을 체계적으로 평가하기 위해 HEMM(Holistic Evaluation of Multimodal Models)을 제안했습니다.

- **Technical Details**: HEMM은 3가지 차원에서 멀티모달 기초 모델의 능력을 평가합니다: 기초 스킬, 정보 흐름, 실세계 사용 사례. 기초 멀티모달 스킬은 모달리티 간 상호작용 학습, 세밀한 정렬, 다단계 추론, 외부 지식 처리 능력 등을 포함합니다. 정보 흐름은 쿼리, 번역, 편집, 융합 과정에서 멀티모달 콘텐츠가 어떻게 변형되는지를 평가하며, 사용 사례는 멀티미디어, 감정 컴퓨팅, 자연 과학, 헬스케어, 인간-컴퓨터 상호작용(Human-Computer Interaction) 등에서의 도메인별 도전 과제를 다룹니다.

- **Performance Highlights**: HEMM을 통해 30개의 다양한 태스크에서 실험을 수행한 결과, 오늘날 모델들이 직면한 주요 데이터셋 차원과 모델링 결정(모델 크기, 사전 학습 데이터, 멀티모달 정렬, 사전 학습 및 교육 튜닝 목표)이 성능에 미치는 영향을 식별할 수 있었습니다. 이 분석을 통해 미래 연구에서 주목해야 할 멀티모달 인터랙션, 태스크, 데이터 및 모델 스케일, 교육 목표에 관한 유용한 통찰력을 얻을 수 있었습니다.



### ConCodeEval: Evaluating Large Language Models for Code Constraints in Domain-Specific Languages (https://arxiv.org/abs/2407.03387)
- **What's New**: 최근 연구에 따르면 대형 언어 모델(LLMs)이 자연어 제약을 이해하는 데 어려움을 겪습니다. 본 논문에서는 코드 형식의 제약을 사용하여 LLMs의 제어 가능성을 평가하는 두 가지 새로운 작업을 제안합니다. 이는 JSON, YAML, XML, Python 등의 도메인 전용 언어(DSL)에서 수행됩니다.

- **Technical Details**: 제안된 작업은 'Data as Code Generation'과 'DSL Validation'입니다. 첫 번째 작업은 코드 형식의 제약을 고려하여 유효한 데이터를 생성하는 것이고, 두 번째 작업은 주어진 제약에 대해 코드의 유효성을 검증하는 것입니다. 또한 JSON, YAML, XML, Python Pydantic, Natural language 같은 다섯 가지 표준을 평가 대상으로 설정하고, 다양한 모델을 실험하여 그 성능을 비교하였습니다.

- **Performance Highlights**: 실험 결과 LLMs는 Python과 XML 스키마 형식을 이해하는 데 어려움을 겪었고, 반면 JSON과 YAML 스키마 형식에서는 상대적으로 좋은 성능을 보였습니다. 특히, Granite 20B와 34B 모델이 대부분의 평가에서 우수한 성능을 보였으며, Codellama 34B 모델은 전반적으로 기대 이하의 성능을 보였습니다. 제약이 자연어로 표현될 때는 생성 작업에서 가장 이해가 잘 되었으나, 검증 작업에서는 높은 파라미터 모델의 성능을 감소시키는 경향이 있었습니다.



### A Multi-Modal Explainability Approach for Human-Aware Robots in Multi-Party Conversation (https://arxiv.org/abs/2407.03340)
Comments:
          21pp (+7pp sup.mat.) Submitted to Computer Vision and Image Understanding Journal on May 13, 2024. This research received funding Horizon-Europe TERAIS project (G.A. 101079338) and Slovak Research and Development Agency, project no. APVV-21-0105

- **What's New**: 이번 연구는 휴먼-로봇 상호작용(HRI)에서 다자간 대화 중 화자가 누구에게 말하고 있는지 파악하는 '주소지 추정(Addressee Estimation, AE)' 모델을 개선했습니다. 이 모델은 기존의 최고 수준(State-Of-The-Art, SOTA) 대비 성능을 높인 것은 물론, 주의(attention) 기반 세그먼트를 통해 설명 가능성을 내재화했습니다.

- **Technical Details**: 연구팀은 주의(attention) 메커니즘을 활용한 신경망을 설계하고 훈련하여 기존 AE 모델을 최적화했습니다(Mazzola et al., 2023). 이후, 이 모델을 모듈형 로봇 아키텍처에 배치해 실시간으로 행동 설명을 제공하는 다중모달 시스템을 구현했습니다. 이 과정에서 다양한 설명 방식(음성, 신체 움직임, 시각적 설명)의 효과를 사용자 연구를 통해 분석했습니다.

- **Performance Highlights**: 개발된 모델은 기존 최고 수준(SOTA) 모델에 비해 성능이 뛰어났으며, iCub 로봇에 적용하여 다자간 대화 참여 능력을 향상시켰습니다. 설명 가능성과 투명성을 모두 고려하여 설계됨으로써, 인간 사용자들이 로봇의 결정을 직관적으로 이해하고 신뢰할 수 있게 되었습니다.



### How Similar Are Elected Politicians and Their Constituents? Quantitative Evidence From Online Social Networks (https://arxiv.org/abs/2407.03255)
- **What's New**: 이번 연구는 정치인과 유권자의 온라인 담론 유사성을 비교하여 민주적 대표성에 대한 질문에 답하고자 합니다. 이를 위해 미국과 영국의 약 2년 반 동안(2020년 9월 - 2023년 2월) 정치인의 트위터 타임라인(560만 트윗)과 투표자의 넥스트도어(Nextdoor) 포스트(2180만 포스트)를 분석하였습니다.

- **Technical Details**: 연구진은 미국 하원 의원(USA Representatives)과 영국 하원의원의 트위터 데이터를 수집하고, 유권자의 넥스트도어 포스트와 비교하였습니다. 넥스트도어는 위치 기반 사회 네트워크로, 사용자가 자신의 집 주소를 검증한 후에 이웃과 상호작용합니다. 이 데이터는 내용(content)과 스타일(style) 모두에서 정치인과 유권자의 온라인 담론을 비교하는 데 사용되었습니다.

- **Performance Highlights**: 정치인의 온라인 담론은 콘텐츠와 스타일 면에서 유권자와 비슷하다는 것을 발견했습니다. 특별히, 선거에서 승리의 폭이 좁을수록 스타일이 더 유사해지고, 내용은 반대로 더 상이해지는 경향이 있었습니다. 소득 수준이 낮은 선거구의 경우, 콘텐츠 유사성이 더 높았습니다. 스타일 면에서는, 저소득 선거구의 정서가 더 유사하고 심리적 텍스트 특성(LIWC categories)은 더 상이하다는 결과가 나왔습니다.



### Is Your AI-Generated Code Really Safe? Evaluating Large Language Models on Secure Code Generation with CodeSecEva (https://arxiv.org/abs/2407.02395)
Comments:
          arXiv admin note: text overlap with arXiv:2310.16263

- **What's New**: 이번 연구에서는 코드 LLMs (Large Language Models)의 보안 측면을 정밀하게 평가하고 개선하기 위한 포괄적인 연구를 제시합니다. 이를 위해 CodeSecEval이라는 44개의 주요 취약성 유형을 다루는 180개의 샘플로 구성된, 정교하게 큐레이션된 데이터셋을 소개합니다. CodeSecEval은 코드 생성 및 코드 수리 작업에서 코드 모델의 자동 평가의 기반이 되며, 보안에 중점을 둡니다.

- **Technical Details**: CodeSecEval 데이터셋은 기존 데이터셋(예: HumanEval, SecurityEval, LLMSecEval, CyberSecEval)보다 크게 개선된 점이 많습니다. 해당 데이터셋은 완전하고 실행 가능한 코드와 일련의 테스트 케이스를 포함하고 있어, 수작업 평가와 부정확한 분석 도구에 대한 의존도를 줄입니다. 실험 결과, 현재의 모델들은 코드 생성과 코드 수리 과정에서 보안 문제를 자주 간과하고 있어, 취약한 코드를 생성하는 경향이 있습니다.

- **Performance Highlights**: 7개의 최신 코드 LLM 모델을 평가한 결과, 이 모델들이 보안 문제를 빈번히 간과한다는 점을 발견했습니다. 이를 해결하기 위해, 취약성 인식 정보와 불안전한 코드 설명을 활용한 다양한 전략을 제안하고 검증했습니다. 연구 결과는 이러한 전략들이 코드 생성 및 수리 과정의 보안을 크게 향상시키는 것을 보여주며, 안전하고 신뢰할 수 있는 모델 배포를 위한 보다 견고한 방법 개발에 영감을 주는 중요한 통찰을 제공합니다.



New uploads on arXiv(cs.IR)

### Optimizing Nepali PDF Extraction: A Comparative Study of Parser and OCR Technologies (https://arxiv.org/abs/2407.04577)
- **What's New**: 이 연구는 PDF 파싱(PDF parsing)과 광학 문자 인식(OCR) 방법을 비교하여 PDF에서 네팔어 콘텐츠를 추출하는 다양한 방법을 조사합니다. 연구 결과, PDF 파서는 빠르고 정확하지만 비유니코드(non-Unicode) 네팔어 폰트에서 어려움을 겪는 반면, OCR(특히 PyTesseract)은 이러한 문제를 극복하며 디지털 및 스캔된 PDF에도 다재다능함을 보입니다.

- **Technical Details**: PDF 파싱은 PyMuPDF, PyPDF2, 및 PDFMiner와 같은 라이브러리를 사용하여 PDF에서 텍스트, 이미지, 메타데이터 등을 추출합니다. 반면, OCR는 패턴 매칭(Pattern Matching)과 특징 추출(Feature Extraction) 알고리즘을 사용하여 이미지를 텍스트로 변환합니다. 특히 PyTesseract와 EasyOCR 라이브러리를 활용하여 네팔어 텍스트를 추출했습니다.

- **Performance Highlights**: PDF 파싱은 네팔어 유니코드 문자에서는 높은 정확도를 보이지만, 비유니코드 폰트에서는 추가 변환 과정이 필요하여 정확도가 떨어집니다. 반면, OCR는 비유니코드 폰트에서도 안정적으로 텍스트를 추출하며, PyTesseract는 다양한 PDF 유형에 대해 일관된 정확도를 유지하며 다소 긴 추출 시간과의 균형을 맞춥니다. EasyOCR는 GPU 가속을 통해 빠른 인식이 가능하지만, CPU 사용 시 PyTesseract보다 느립니다.



### VRSD: Rethinking Similarity and Diversity for Retrieval in Large Language Models (https://arxiv.org/abs/2407.04573)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)에 대한 벡터 검색 알고리즘의 중요성을 강조하며, 유사성과 다양성을 모두 충족하는 벡터를 검색하는 새로운 접근 방식을 소개합니다. 기존의 Maximal Marginal Relevance (MMR) 알고리즘이 유사성과 다양성의 균형을 맞추기 위해 사용되었지만, 이 논문은 λ 파라미터의 변동으로 인해 최적화의 방향을 파악하기 어려운 문제를 지적합니다. 본 연구는 쿼리 벡터와 합 벡터 간의 관계를 통해 유사성과 다양성 제약을 특성화하는 새로운 방법을 제안하며, 이를 통해 유사성과 다양성을 동시에 추구하는 복잡한 최적화 문제를 해결하고자 합니다.

- **Technical Details**: 논문은 선택된 벡터의 합 벡터와 쿼리 벡터 간의 유사성을 최대화하는 새로운 조합 최적화 문제를 정의합니다. 이 문제는 NP-complete으로 판명되었으며, 이는 유사성과 다양성을 동시에 충족하는 것이 이론적으로 매우 어려운 문제임을 보여줍니다. 이를 해결하기 위해, 파라미터가 필요 없고 최적화 목표가 명확한 휴리스틱 알고리즘인 Vectors Retrieval with Similarity and Diversity (VRSD)를 제안합니다. 이 알고리즘은 MMR보다 시간 복잡도가 낮으며, 다양한 데이터셋에서 MMR을 능가하는 성능을 보였습니다.

- **Performance Highlights**: 실험 결과, VRSD 알고리즘은 여러 데이터셋에서 MMR 알고리즘을 상당히 능가하는 성능을 보였습니다. 이는 VRSD가 유사성과 다양성을 동시에 충족시키는 검색 문제 해결에 더 효과적임을 입증합니다.



### EventChat: Implementation and user-centric evaluation of a large language model-driven conversational recommender system for exploring leisure events in an SME contex (https://arxiv.org/abs/2407.04472)
Comments:
          27 pages, 3 tables, 5 figures, pre-print manuscript

- **What's New**: 이번 연구는 대규모 언어 모델(Large Language Models, LLM)을 사용한 대화형 추천 시스템(Conversational Recommender Systems, CRS)의 설계와 성능에 대해 논의합니다. 특히, 중소기업(SME) 환경에서 이를 구현하고 평가하는 데 중점을 두고 있어 기존 연구와 차별화됩니다.

- **Technical Details**: LLM 기반의 CRS는 RAG(Retrieval-Augmented Generation) 기술 내에서 고급 LLM을 랭커(rankers)로 사용하는 구조를 가지고 있습니다. 이와 함께, 신속하게 변화하는 분야에서 평가의 재현성을 높이기 위해 개정된 짧은 형태의 ResQue 모델을 개발하였습니다.

- **Performance Highlights**: 이 시스템은 사용자 경험 측면에서 높은 추천 정확도(85.5%)를 기록하였지만, 지연 시간(latency)과 비용 문제가 비즈니스 실행 가능성을 저해하는 도전 과제로 나타났습니다. 중간 비용은 상호작용당 $0.04이며, 지연 시간은 5.7초로 나타났습니다. 이러한 비용 중 큰 부분은 고급 LLM을 사용함에 기인하며, ChatGPT를 기반으로 한 프롬프트(Prompt) 기반 학습 접근법만으로는 만족스러운 품질을 얻기 어려웠습니다.

- **Business Implications**: SME가 LLM 기반의 CRS를 배포할 때 전략적 고려 사항을 강조합니다. 특히, 비용 효율성과 응답 시간은 사용자 친화적이고 경제적으로 실행 가능한 시스템을 달성하는 데 중요한 요소로 평가됩니다.



### Leveraging Topic Specificity and Social Relationships for Expert Finding in Community Question Answering Platforms (https://arxiv.org/abs/2407.04018)
- **What's New**: 온라인 커뮤니티 질문-답변(CQA) 플랫폼에서 전문가 찾기(EF)는 사용자 참여 및 답변 신뢰성 향상에 매우 중요한 역할을 합니다. 본 논문에서는 다양한 정보원을 효과적으로 통합하여 EF 성능을 개선하기 위해 TUEF라는 주제 지향 사용자-상호작용 모델을 제안합니다. TUEF는 특정 주제에 대한 사용자의 답변 패턴을 기반으로 멀티-레이어 그래프를 구축하여 콘텐츠와 소셜 데이터를 통합합니다.

- **Technical Details**: TUEF는 멀티-레이어 그래프를 생성하여 각 레이어가 커뮤니티 내 주요 주제를 나타내도록 합니다. 그래프의 노드는 주제 토론에 활발히 참여하는 사용자를 나타내고, 엣지는 사용자 간의 유사성과 관계를 모델링합니다. 질문이 게시되면 TUEF는 멀티-레이어 그래프와 랭킹 모델을 사용하여 관련 주제 및 해당 그래프 레이어를 결정합니다. 네트워크 관점(커뮤니티 내 영향력 있는 사용자 식별)과 콘텐츠 관점(유사 질문에 답한 사용자 식별)에서 후보 전문가를 선정합니다. 최종적으로, TUEF는 Learning-to-Rank 기술을 적용하여 질문에 대한 후보자의 예상 관련성을 점수화하고 순위를 매깁니다.

- **Performance Highlights**: TUEF는 다양한 실험에서 높은 성능을 보였으며, Stack Exchange의 6개 다양한 커뮤니티에서 수행한 실험 결과 최소 42.42%의 P@1 향상, 32.73%의 NDCG@3 향상, 21.76%의 R@5 향상, 29.81%의 MRR 향상을 기록했습니다. 또한, 해석 가능한 Learning-to-Rank 알고리즘을 통합하여 투명성과 설명 가능성을 제공하면서도 성능 저하를 최소화했습니다.



### Query-oriented Data Augmentation for Session Search (https://arxiv.org/abs/2407.03720)
Comments:
          TKDE 2024

- **What's New**: 검색 세션 내 맥락 정보를 모델링하는 새로운 접근법을 제안합니다. 이 논문에서는 현재 검색 쿼리를 변화시켜 보충 훈련 쌍을 생성하고, 이를 통해 검색 로그 데이터를 더욱 풍부하게 만드는 'Query-oriented Data Augmentation' 방법을 소개합니다. 이를 통해 모델이 문서의 관련성이 세션 맥락에 따라 변화할 수 있음을 학습할 수 있습니다.

- **Technical Details**: 기존의 검색 세션 모델링은 주로 클릭된 문서와 클릭되지 않은 문서 간의 순위를 학습하는 방식이었으나, 문서와 세션 맥락 간의 대칭적인 관련성을 고려하지 못했습니다. 이를 해결하기 위해, 우리는 현재 쿼리를 주요 변경 요소로 설정하여 새로운 훈련 샘플을 생성하는 전략을 사용했습니다. 구체적으로, 쿼리의 용어 수준 변경(Term-level Modification)과 쿼리 수준 교체(Query-level Replacement)를 통해 다양한 난이도의 부정 샘플을 생성했습니다.

- **Performance Highlights**: AOL 및 Tiangong-ST와 같은 두 개의 공개 검색 로그 데이터셋에서 실험한 결과, 본 논문에서 제안한 'Query-oriented Data Augmentation' 방법을 탑재한 모델이 기존의 모델보다 성능 면에서 뛰어난 것으로 나타났습니다. 이는 우리의 접근법이 사용자 검색 패턴을 보다 포괄적으로 학습하는 데 효과적임을 입증합니다.



### Heterogeneous Hypergraph Embedding for Recommendation Systems (https://arxiv.org/abs/2407.03665)
- **What's New**: 최근 추천 시스템(Recommender Systems)에서 지식 그래프(Knowledge Graph, KG)를 통합하여 보조 정보를 활용하는 것이 주요한 연구 주제가 되고 있습니다. 이러한 접근법은 풍부한 의미 정보를 통해 더 정확한 추천을 제공하는 것입니다. 그러나 기존 모델들은 KG 기반 사용자-아이템 네트워크에서 복잡한 고차 상호작용을 무시하거나 입력 소스의 다양한 모달리티를 처리하면서 노이즈와 부정확성을 초래하는 두 가지 주요 문제점이 있습니다. 이를 해결하기 위해, 새로운 Knowledge-enhanced Heterogeneous Hypergraph Recommender System (KHGRec)을 제안합니다.

- **Technical Details**: KHGRec은 상호작용 네트워크와 KG의 그룹 특성을 모두 캡처하여 복잡한 연결을 모델링합니다. collaborative knowledge heterogeneous hypergraph (CKHG)를 사용하여 그룹 간 상호 의존성을 모델링하며, 두 가지 하이퍼그래프 인코더를 통해 이를 설명 가능합니다. 또한 KG와 사용자-아이템 그래프의 신호를 통합하여 교차-뷰 자율 지도 학습(cross-view self-supervised learning)과 어텐션 메커니즘(attention mechanisms)을 활용합니다. 이 프레임워크는 그룹 단위 상호작용을 자연스럽게 포착하고, 두 개의 입력 그래프를 통합하여 노이즈를 방지합니다.

- **Performance Highlights**: 4개의 실제 데이터셋에서 실시한 다양한 실험 결과, 우리의 모델은 최신 기준 모델들에 비해 평균 5.18% 상대적 성능 향상을 보여줍니다. 추가적으로 노이즈 저항성, 누락 데이터 및 콜드 스타트 문제에 관한 테스트에서 KHGRec 프레임워크의 견고함을 입증하였습니다. 모델과 평가 데이터셋은 공개되었습니다.



### Reviewers of Educational Immersive and Extended Reality (XR) experiences: Who is creating these reviews and why? (https://arxiv.org/abs/2407.03650)
Comments:
          14 pages

- **What's New**: 이번 논문은 교육용 확장 현실(eduXR) 경험을 검토하는 사람이 누구인지, 왜 검토하는지에 대한 문헌 검토를 제시합니다. EduXR은 증강 현실(AR), 가상 현실(VR), 혼합 현실(MR) 등을 통해 다양한 형태로 교육을 지원합니다.

- **Technical Details**: 연구는 16개의 논문을 분석하여 eduXR 리뷰를 수행하는 사람들(who themes)과 그 이유(why themes)에 대한 주제를 두 번의 테마 분석 주기를 통해 코딩했습니다. 이 과정에서 eduXR 커뮤니티가 교육 경험을 선택하는 방법에 대해 무엇이 가능한지, 제한되는지, 그리고 아직 알려지지 않은 부분을 이해하려고 했습니다. 현재 eduXR 리뷰에 대한 확립된 시스템은 없으며, 일부 'serious games' 리뷰 프레임워크가 유사한 점이 있지만 모든 eduXR 경험에 적용되지는 않습니다.

- **Performance Highlights**: 분석 결과, eduXR 리뷰를 작성하는 사람에 대한 명확하고 단순한 방법이 필요하다는 결론에 도달했습니다. 특히, 사용자들이 리뷰 작성자의 신원과 작성 이유를 쉽게 이해할 수 있게 하는 구조가 필요하다고 주장합니다. 이를 통해 사용자들이 eduXR 경험에 대한 유용한 인사이트를 획득할 수 있을 것입니다.



### BM25S: Orders of magnitude faster lexical search via eager sparse scoring (https://arxiv.org/abs/2407.03618)
Comments:
          Technical Report

- **What's New**: BM25S는 Numpy와 Scipy만을 의존하는 BM25의 효율적인 Python 구현체를 소개합니다. BM25S는 색인화 과정에서 BM25 점수를 미리 계산하여 희소 행렬로 저장함으로써 가장 널리 사용되는 Python 기반 프레임워크보다 최대 500배 빠른 성능을 달성합니다. 또한 인기 있는 상용 제품에서 사용되는 고도로 최적화된 Java 기반 구현보다도 상당한 속도 향상을 이룹니다. BM25S는 새로운 점수 이동 방법을 통해 비희소(Non-sparse) 변형에도 확장 적용할 수 있습니다.

- **Technical Details**: BM25S는 색인화 시 모든 가능한 점수를 미리 계산하고 이를 희소 행렬에 저장하는 방식을 통해 속도를 극대화했습니다. PyTorch를 사용하는 기존의 BM25-PT와 달리, BM25S는 Scipy의 희소 행렬 구현을 사용하며 행렬 곱셈이 아닌 슬라이싱과 summation을 활용합니다. 또한 Scikit-Learn의 텍스트 분할, Elastic's stopword list, 선택적으로 C 기반 Snowball stemmer 등의 간단하지만 빠른 파이썬 기반 토크나이저를 도입했습니다.

- **Performance Highlights**: BM25S는 색인화 과정에서 미래의 쿼리 토큰에 할당될 수 있는 모든 점수를 미리 계산하고 이를 희소 행렬에 저장함으로써, 기존 Python 기반 구현체 대비 최대 500배, 고도로 최적화된 Java 기반 구현체 대비 상당한 속도 향상을 달성합니다. 또, Scipy의 Compressed Sparse Column (CSC) 형식을 사용하여 효율적인 변환과 연산을 가능케 하며, 결과적으로 검색 과정에서 O(n) 복잡도의 평균 시간 복잡도를 달성했습니다.



### Deep Pareto Reinforcement Learning for Multi-Objective Recommender System (https://arxiv.org/abs/2407.03580)
- **What's New**: 추천 플랫폼에서 여러 목표를 동시에 최적화하는 것은 다양한 측면에서 성능을 개선하기 위한 중요한 과제입니다. 본 논문에서는 기존의 정적이고 균일한 방식이 아닌, 동적이고 개인화된 방식으로 다목적 추천 시스템을 최적화하는 Deep Pareto Reinforcement Learning (DeepPRL) 접근법을 제안합니다.

- **Technical Details**: DeepPRL은 (1) 여러 목표 간의 복잡한 관계를 종합적으로 모델링하고, (2) 각 목표에 대한 개인화된 소비자의 선호도와 상황적 변화를 효과적으로 포착하여 추천을 업데이트하며, (3) 다목적 추천의 단기 및 장기 성과를 최적화합니다. 이 모델은 Pareto Frontier에서의 성능을 극대화하는 것을 목표로 합니다.

- **Performance Highlights**: 세 개의 실제 데이터셋을 대상으로 한 광범위한 오프라인 실험에서, DeepPRL은 최신 기준 모델들보다 유의미한 Pareto 우위를 달성했습니다. 또한 Alibaba의 비디오 스트리밍 플랫폼에서 실시한 대규모 온라인 통제 실험에서는 클릭율(Click-Through Rate), 비디오 조회수(Video View) 및 체류 시간(Dwell Time)에서 각각 2%, 5%, 7%의 개선을 이루어냈습니다. 이는 산업 애플리케이션에서의 실질적인 경제적 영향을 입증했습니다.



### GPT vs RETRO: Exploring the Intersection of Retrieval and Parameter-Efficient Fine-Tuning (https://arxiv.org/abs/2407.04528)
- **What's New**: 최근 연구에서는 대형 언어 모델을 적응시키면서 계산 요구를 최소화하는 방법으로 PEFT (Parameter-Efficient Fine-Tuning)와 RAG (Retrieval-Augmented Generation)가 주목받고 있습니다. 이번 논문에서는 PEFT 기법(P-tuning, Adapters, 및 LoRA)을 수정된 RETRO 모델과 GPT 모델에 적용하여 여러 크기의 모델에 대해 비교 분석했습니다. 특히, 이 연구는 여러 PEFT 방법이 RAG와 통합된 첫 번째 포괄적인 비교 분석을 제공합니다.

- **Technical Details**: PEFT는 업데이트하는 매개변수의 수를 줄이면서도 성능을 유지하는 기법입니다. 이 논문에서는 PEFT 기법으로 P-tuning, Adapter 모듈과 Low-Rank Adaptation (LoRA)을 사용했습니다. RETRO 모델은 캡슐화된 크로스 어텐션 메커니즘을 통해 검색 모듈을 직접 Transformer 아키텍처에 통합하여 학습하는 고유한 접근 방식을 가지고 있습니다. RAG는 외부 지식을 통합하여 모델 성능을 향상시키는 방법으로, 검색된 소스를 입력 쿼리에 연결하여 활용합니다.

- **Performance Highlights**: 실험 결과, RETRO 모델은 zero-shot 설정에서 GPT 모델을 능가했으며, 특히 8B 파라미터 모델이 비용과 성능의 균형을 가장 잘 맞췄습니다. 또, P-tuning은 다른 PEFT 기법에 비해 성능이 떨어졌습니다. RETRO 모델은 검색된 텍스트에서 핵심 정보를 추출하고 이를 생성 프로세스에 통합하는데 뛰어난 능력을 보여줬습니다.



### An Interactive Multi-modal Query Answering System with Retrieval-Augmented Large Language Models (https://arxiv.org/abs/2407.04217)
Comments:
          This demo paper has been accepted by VLDB 2024

- **What's New**: 이 논문에서는 다중 모드 질의 응답 시스템(MQA)을 소개합니다. 이 시스템은 최첨단 대형 언어 모델(LLMs)과 새로운 다중 모드 검색 프레임워크 및 네비게이션 그래프 인덱스를 결합하여 개발되었습니다. 특히, 대조 학습(contrastive learning)을 활용하여 다양한 모드의 중요도를 평가하고, 고도로 정확한 검색을 가능하게 합니다.

- **Technical Details**: MQA 시스템은 데이터 전처리, 벡터 표현, 인덱스 구축, 질의 실행, 답변 생성의 다섯 가지 핵심 구성 요소로 이루어져 있습니다. 이 시스템은 다중 벡터 표현 기술을 사용하여 다중 모드 데이터를 효과적으로 표현합니다. 또한, 사용자 친화적인 인터페이스를 갖추고 있어 텍스트와 이미지 등을 포함한 다양한 입력 형태를 지원하며, 반복적인 질의 개선 과정을 통해 최적의 결과를 제공합니다.

- **Performance Highlights**: MQA는 고급 대조 학습을 통해 모드 간 유사성을 평가함으로써 질의 정확도를 향상시킵니다. 또한, 고유한 벡터 가중치 학습 모델을 도입해 각 모드의 중요도를 반영합니다. 나아가, 복잡한 네비게이션 그래프 프레임워크를 통해 대규모 데이터에서도 효율적인 검색이 가능합니다. 이 시스템은 높은 유연성을 자랑하며, 다양한 인코더 및 검색 프레임워크와 통합될 수 있습니다.



### When LLM Meets Hypergraph: A Sociological Analysis on Personality via Online Social Networks (https://arxiv.org/abs/2407.03568)
- **What's New**: 이 논문은 온라인 소셜 네트워크 분석을 통해 개인의 성격을 환경 기반으로 분석하는 새로운 프레임워크를 제안합니다. 기존의 개인 수준 데이터 마이닝을 넘어, 대형 언어 모델(LLMs)의 강력한 연관 능력을 활용하여 낮은 품질의 데이터를 높은 품질로 통합함으로써 성격 분석 성능을 크게 향상시킬 수 있는 방법을 제안합니다. 이를 위해, 실제 소셜 플랫폼에서 수집한 사용자 프로필 데이터, 성격 특성 및 여러 환경 정보가 포함된 새로운 데이터셋을 제공합니다. 이 데이터셋은 하이퍼그래프 구조와 사회 정보를 모두 포함한 최초의 네트워크 기반 데이터셋입니다.

- **Technical Details**: 논문에서는 LLMs를 활용하여 분산된 정보를 통합하기 위해 효과적인 프롬프트를 설계했습니다. 이는 사용자 데이터의 단편적인 데이터를 높은 품질의 프로필로 변환할 수 있게 합니다. 또한, 하이퍼그래프 뉴럴 네트워크(hypergraph neural network)를 설계하여 다양한 소셜 환경에서의 상호작용 메커니즘을 탐구했습니다. 하이퍼그래프의 노드는 사용자, 하이퍼엣지는 소셜 환경을 나타내며, 이를 통해 온라인 소셜 인터액션의 복잡성을 모델링 할 수 있습니다. 이 프레임워크는 개인의 성격과 그들의 온라인 행동을 포착하여 더 깊은 이해를 제공합니다.

- **Performance Highlights**: 제안된 프레임워크를 사용하여 온라인 환경에서 개인 성격을 효과적으로 분석할 수 있으며, 영역 내에서 주요한 성과를 거두었습니다. 특히, 하이퍼그래프 뉴럴 네트워크와 LLMs의 조합을 통해 단편적이고 불완전한 데이터를 통합하여 더 정확하고 일관된 사용자 프로필을 생성할 수 있었습니다. 제공된 데이터셋과 프레임워크는 기존의 방법론과 비교해 성격 분석의 성능을 크게 향상시켰습니다.




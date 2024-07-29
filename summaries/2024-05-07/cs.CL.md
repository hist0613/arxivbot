### Large Language Models Reveal Information Operation Goals, Tactics, and  Narrative Frames (https://arxiv.org/abs/2405.03688)
Comments: 15 pages, 9 figures

- **What's New**: 이 연구는 대규모 언어 모델(Large Language Models, LLMs)을 사용하여 정보 작전을 분석하는 새로운 접근 방식을 제시합니다. 특히, GPT-3.5를 사용하여 지난 십 년간에 걸쳐 확인된 126개의 정보 작전을 검토하고 이를 통해 조정된 캠페인을 파악하고 분석합니다. 주요 사례 연구로는 2022년 프랑스 선거(French election)와 2023년 필리핀-미국 군사 훈련(Balikaran Philippine-U.S. military exercise)에 대한 데이터셋이 사용되었습니다.

- **Technical Details**: 연구팀은 GPT-3.5 모델을 사용하여 각각의 정보 작전에서 목표, 전략, 그리고 서사 구조(Narratives)를 추출했습니다. 이를 위해 두 개의 큰 다국어 데이터셋을 분석했으며, 선거 날짜와 같은 중요 사건 전후의 게시물을 분석하여 정보 캠페인의 더욱 정확한 이미지를 구성할 수 있었습니다.

- **Performance Highlights**: GPT-3.5와 실제 상황과의 밀접한(비록 완벽하지는 않지만) 일치를 보여주는 메트릭스를 적용하여 분석한 결과, LLMs가 기존 방법보다 정보 캠페인에 대해 더 포괄적이고 상세한 통찰을 제공할 수 있는 잠재력을 입증하였습니다. 그러나 때때로 GPT-3.5는 주관적 해석과 다르게 판단할 수 있기 때문에 데이터 해석에 있어 신중을 기할 필요가 있습니다.



### Towards A Human-in-the-Loop LLM Approach to Collaborative Discourse  Analysis (https://arxiv.org/abs/2405.03677)
Comments: In press at the 25th international conference on Artificial Intelligence in Education (AIED) Late-Breaking Results (LBR) track

- **What's New**: LLMs (Large Language Models)가 다양한 작업에서 인간 수준의 성능을 달성하거나 능가하는 것으로 나타났지만, 아직 학생들의 협력적 담론에서 상호작용 학습(synergistic learning)을 특성화하는 데 사용된 적은 없습니다. 이번 탐색적 연구에서는 GPT-4-Turbo를 사용하여 학생들의 협력적 담론 중에 발생하는 상호작용 학습을 요약하고 분류하는 인간-루프(Human-in-the-loop) 프롬프트 엔지니어링(prompt engineering) 접근을 처음으로 시도합니다.

- **Technical Details**: GPT-4-Turbo를 활용하여 학생들의 상호작용 학습을 인간과 비슷한 수준으로 특성화할 수 있다는 초기 결과가 나왔습니다. 이 접근방식은 더 많은 연구가 필요하다는 결과를 도출했습니다.

- **Performance Highlights**: GPT-4-Turbo는 상호작용 학습을 분류하고 요약하는 데 인간 수준의 비교 가능한 성과를 보였으며, 이 연구 방법이 향후 더 많은 연구를 진행할 가치가 있음을 시사합니다.



### GREEN: Generative Radiology Report Evaluation and Error Notation (https://arxiv.org/abs/2405.03595)
- **Newsletter**: [{"What's New": "새로운 평가 메트릭 'GREEN' (Generative Radiology Report Evaluation and Error Notation)이 소개되었습니다. 이 메트릭은 영상의학 보고서의 정확성을 향상시키기 위해 자연어 이해력을 가진 언어 모델을 사용하여 임상적으로 중요한 오류를 정량적이고 정성적으로 식별하고 설명합니다."}, {'Technical Details': 'GREEN 메트릭은 기존 평가 도구들이 놓친 임상적 정확성을 고려하여, 의료 이미지에 관한 정확한 의료 소통이 필요한 영상의학 보고서 평가에 있어 획기적인 접근을 제공합니다. 이 메트릭은 언어 모델(GPT-4)을 활용하여 보고서의 오류를 인식하고, 이를 인간이 이해할 수 있는 방식으로 설명합니다.'}, {'Performance Highlights': 'GREEN은 상업적 대안에 버금가는 성능을 제공하는 가벼운 오픈소스 방법이며, 6명의 전문가의 오류 카운트와 2명의 전문가 선호도와 비교하여 높은 상관관계를 보였습니다. 이는 기존 접근법들과 비교할 때 전문가들의 오류 카운트와 선호도에 더 높은 일치성을 나타내는 동시에 이해 가능한 피드백을 제공할 수 있습니다.'}]



### Enabling High-Sparsity Foundational Llama Models with Efficient  Pretraining and Deploymen (https://arxiv.org/abs/2405.03594)
- **What's New**: 이 연구에서는 큰 언어 모델(Large Language Models, LLMs)의 계산 병목 문제를 해결하기 위해 새로운 접근 방식을 도입하였습니다. 특히, LLaMA-2 7B 모델을 대상으로 SparseGPT의 단발성(pruning) 방법과 SlimPajama 데이터셋의 부분집합 및 The Stack 데이터셋의 Python 부분집합을 사용한 sparse pretraining을 결합하여 최대 70%의 희소성(sparsity)에서도 미세조정(fine-tuning) 작업에 대한 완전한 정확도 회복을 달성했습니다.

- **Technical Details**: 이 연구는 Cerebras CS-3 칩에서의 희소성으로 인한 훈련 가속화를 실증적으로 보여 주며, 이론적인 스케일링과 밀접하게 일치합니다. 또한, Neural Magic의 DeepSparse 엔진을 사용하여 CPU에서 최대 3배, GPU에서는 Neural Magic의 nm-vllm 엔진을 통해 최대 1.7배의 추론 가속을 달성하였습니다. 이러한 성과는 희소성만을 통해 실현되었으며, 양자화(quantization)의 추가적인 사용을 통해 더 큰 이득을 얻을 수 있습니다.

- **Performance Highlights**: 특히, 희소량자화된(sparse-quantized) LLaMA 모델에서 CPU에서 최대 8.6배의 총 속도 향상을 보여주었습니다. 이 결과는 대화, 지시 사항 따르기, 코드 생성, 산술 추론, 요약 등 다양하고 도전적인 작업에 걸쳐 일반성을 입증합니다.



### AlphaMath Almost Zero: process Supervision without process (https://arxiv.org/abs/2405.03553)
Comments: Work in progress

- **What's New**: 이번 연구에서는 복잡한 수학적 추론을 필요로 하는 문제에서 LLM(Large Language Models)의 능력을 향상시키기 위해 수작업으로 주석을 다는 대신 Monte Carlo Tree Search(MCTS) 프레임워크를 활용하여 과정 감독과 평가 신호를 자동으로 생성하는 새로운 접근법을 소개하였습니다. 이 방법은 LLM이 실제 문제 해결에서 오류를 만났을 때 경로를 재평가하고 수정하는 인간의 동적 문제 해결 방식을 반영합니다.

- **Technical Details**: LLM을 MCTS 프레임워크와 통합하여 탐색과 활용 사이의 균형을 더 효과적으로 이룰 수 있도록 하여 질 높은 훈련 데이터를 자동으로 생성할 수 있게 하였습니다. 이 연구는 Monte Carlo Tree Search(MCTS)와 LLM을 결합하여, 학습 데이터를 생성하고 필터링하는 과정에서 인간의 지식 없이도 고도의 수학적 추론을 자동으로 생성할 수 있음을 보여주었습니다. 이를 위해 LLM은 적절한 프롬프트와 MCTS 프레임워크를 이용하여 REACT 형식을 포함한 텍스트 분석 및 코드 스니펫을 자동으로 생성합니다.

- **Performance Highlights**: 실험 결과, MCTS를 통합한 모델은 복잡한 수학 문제 해결 과정에서 높은 성능을 보였습니다. 예를 들어, MATH 데이터셋에서는 도전적인 문제를 효과적으로 해결하며, 중간 단계에서의 오류를 효과적으로 파악하고 수정하는 능력이 향상되었습니다. 이는 LLM과 MCTS 프레임워크, 그리고 가치 모델(Value Model)이 정책 모델(Policy Model)을 보조하여 문제 해결 경로를 더 효율적으로 탐색하도록 도와주는 것을 확인했습니다.



### MAmmoTH2: Scaling Instructions from the Web (https://arxiv.org/abs/2405.03548)
Comments: Work in Progress

- **What's New**: 이 연구는 대규모 언어 모델(Large Language Models, LLMs)의 추론 능력을 향상시키기 위한 새로운 방법을 제안합니다. 기존에 사람의 크라우드소싱이나 GPT-4의 증류를 통해 얻은 지시(instruction) 튜닝 데이터 대신, 이 연구는 웹 코퍼스에서 자연적으로 존재하는 1000만 개의 지시 데이터를 효율적으로 수집하는 패러다임을 제안합니다.

- **Technical Details**: 이 방법은 관련 문서를 회수하고(instruction recall), 지시-응답(instruction-response) 쌍을 추출하며(open-source LLMs를 사용하여), 추출된 쌍을 정제하는(refinement) 세 단계로 구성됩니다. 기본 LLMs에 이 데이터셋을 미세조정(fine-tuning)하여 MAmmoTH2 모델을 구축했으며, 이는 추론 벤치마크에서의 성능을 크게 향상시켰습니다.

- **Performance Highlights**: 특히 MAmmoTH2-7B(Mistral)는 MATH 벤치마크에서 11%에서 34%로, GSM8K에서는 36%에서 67%로 성능이 크게 향상되었습니다. 또한, MAmmoTH2를 공개적이고 사전에 수집된 지시 튜닝 데이터셋으로 추가 트레이닝한 결과, MAmmoTH2-Plus는 여러 추론 및 챗봇 벤치마크에서 최고의 성능(state-of-the-art performance)을 달성했습니다.



### Gaussian Stochastic Weight Averaging for Bayesian Low-Rank Adaptation of  Large Language Models (https://arxiv.org/abs/2405.03425)
Comments: 14 pages, 1 figure, 2 tables

- **What's New**: 이 논문에서는 소규모 데이터셋으로 파인 튜닝된 대형 언어 모델(Large Language Models, LLMs)이 갖는 과잉 확신(overconfidence) 및 교정 문제(poor calibration)를 해결하기 위해 Low-Rank Adaptation (LoRA)와 Gaussian Stochastic Weight Averaging (SWAG)를 조합한 새로운 방법을 제안합니다. 이 방법은 LLM에서의 대략적 베이지안 추론(approximate Bayesian inference)을 촉진시킬 수 있습니다.

- **Technical Details**: 제안된 방법은 LoRA를 통해 모델의 가중치(weight)를 저차원으로 조정하고, SWAG를 사용하여 가중치의 분포를 추정함으로써, 모델의 일반화(generalization)와 교정(calibration) 능력을 향상시킵니다. 이 조합은 계산 효율성을 유지하면서도, LLMs의 베이지안 추론을 근사화하는 효과적인 접근법을 제공합니다.

- **Performance Highlights**: 이 방법은 여러 자연어 처리(Natural Language Processing, NLP) 벤치마크에서 광범위한 테스팅을 거쳐, 모델의 일반화 및 교정 성능이 향상되었음을 입증했습니다. 또한, 분포 이동(distribution shift)에 대한 강인성(robustness) 또한 향상되었다는 것을 배포 외 작업(out-of-distribution tasks)의 성능을 통해 보여주고 있습니다.



### The high dimensional psychological profile and cultural bias of ChatGP (https://arxiv.org/abs/2405.03387)
- **What's New**: 이 연구에서는 AI 모델, 특히 ChatGPT의 인간과 유사한 특성과 그 차이점을 이해하기 위해 84가지 심리학적 특성을 측정하고 이를 인간의 표준과 비교했습니다. 또한, ChatGPT의 문화적 가치 패턴이 세계 여러 나라/지역과 어떻게 다른지 13가지 차원에서 분석했습니다.

- **Technical Details**: ChatGPT는 대부분의 심리적 차원에서 인간 규범과 차이를 보였으며, 심리학적 표현에서도 고차원적인 차이가 관찰되었습니다. 문화적 가치 측정에서는 ChatGPT의 문화적 패턴이 다양한 국가/지역의 패턴과 상이함을 보여주었습니다.

- **Performance Highlights**: ChatGPT는 여러 국가/지역의 인간과의 상호작용을 포함한 8가지 의사결정 작업에서 뚜렷한 문화적 스테레오타입(cultural stereotypes)과 유의미한 문화적 편향(cultural bias)을 보여주었습니다. 특히 제삼자 처벌(third-party punishment)과 최후통첩 게임(ultimatum games)에서 이러한 경향이 강하게 나타났습니다.



### Explainable Fake News Detection With Large Language Model via Defense  Among Competing Wisdom (https://arxiv.org/abs/2405.03371)
Comments: 12 pages, WWW'2024

- **What's New**: 본 논문에서는 거짓 정보를 검출하는 새로운 방법을 제안합니다. 기존의 검증 방법들은 뉴스 기사의 명확한 근거를 제시하지 않고 결과만을 도출하는데, 본 연구는 진위 여부를 판단하기 위해 두 진영의 명확한 증거를 강조하는 새로운 방법을 제시합니다. 이 방법은 경쟁적 지혜(wisdom of crowds)를 활용하여 진실과 거짓을 구분하고, 이를 통해 보다 투명하고 설명 가능한(explainable) 거짓 뉴스 검출이 가능합니다.

- **Technical Details**: 본 시스템은 세 가지 주요 모듈로 구성됩니다. 첫 번째, 증거 추출 모듈(evidence extraction module)은 경쟁적 지혜를 두 개의 대립 진영으로 나누고 각각에서 중요한 증거를 감지합니다. 두 번째, 프롬프트 기반 모듈(prompt-based module)은 대규모 언어 모델(large language model, LLM)을 사용하여 두 가능한 진실성에 대한 이유를 추론하여 정당화문을 생성합니다. 마지막으로, 방어 기반 추론 모듈(defense-based inference module)은 이 정당화들 사이의 방어를 모델링하여 진위 여부를 결정합니다.

- **Performance Highlights**: 이 방법은 두 개의 실제 데이터셋에서 상태-지향(state-of-the-art) 기준 모델들을 능가하는 성능을 보였습니다. 즉, 본 연구는 거짓 뉴스의 검출과 그 정당성을 제공하는데 있어서 높은 품질을 보여줍니다. 또한, 본 방법은 경쟁적 지혜를 활용하여 진위 여부를 결정하기 때문에 기존 방법들이 가지고 있던 다수 의견의 편향 문제를 완화시킬 수 있습니다.



### MedDoc-Bot: A Chat Tool for Comparative Analysis of Large Language  Models in the Context of the Pediatric Hypertension Guidelin (https://arxiv.org/abs/2405.03359)
Comments: {copyright} 2024 IEEE. This work has been accepted for publication and presentation at the 46th Annual International Conference of the IEEE Engineering in Medicine and Biology Society, to be held in Orlando, Florida, USA, July 15-19, 2024

- **What's New**: 이 연구는 비상업적 오픈 소스 대형 언어 모델(Large Language Models, LLMs)인 Meditron, MedAlpaca, Mistral, 그리고 Llama-2를 사용하여 PDF 형식으로 저장된 의료 지침을 해석하는 효과를 평가합니다. 구체적인 테스트 시나리오로서, 유럽심장학회(European Society of Cardiology, ESC)에서 제공하는 소아 및 청소년 고혈압에 대한 지침을 적용했습니다. Python 라이브러리인 Streamlit을 활용하여 사용자 친화적인 의료 문서 챗봇 도구(MedDoc-Bot)를 개발했습니다. 이 도구는 인증된 사용자가 PDF 파일을 업로드하고 질문을 할 수 있게 하며, 네 개의 로컬에 저장된 LLMs로부터 해석적인 응답을 생성합니다.

- **Technical Details**: 이 프로젝트에서는 MedDoc-Bot을 통해 사용자는 PDF 형식의 의료 지침을 업로드하고 질문을 제출할 수 있으며, LLM들은 이에 대한 답변을 생성합니다. 모델 생성 응답의 충실도와 관련성을 평가하기 위해 소아과 전문가가 ESC 지침에서 추출한 질문과 응답을 기준으로 설정하고 평가했습니다. 또한 모델 응답과 참조答안의 유사성을 평가하기 위해 METEOR 및 chrF 메트릭 점수를 평가했습니다.

- **Performance Highlights**: Llama-2와 Mistral은 메트릭 평가에서 높은 성능을 보였습니다. 그러나 Llama-2는 텍스트와 표 데이터를 처리할 때 속도가 느렸습니다. 인간 평가에서는 Mistral, Meditron, Llama-2가 생성한 응답이 합리적인 충실도와 관련성을 보였습니다. 이 연구는 의료 문서 해석에서 LLMs의 강점과 한계에 대한 중요한 통찰력을 제공하며 의료 분야의 AI 개발에 기여할 수 있습니다.



### Lifelong Knowledge Editing for LLMs with Retrieval-Augmented Continuous  Prompt Learning (https://arxiv.org/abs/2405.03279)
Comments: 14 pages, 4 figures, 6 tables

- **What's New**: 이 연구에서는 LLM에서 비용이 많이 드는 재학습을 필요로 하지 않으면서 오래된 또는 잘못된 지식을 수정하는 모델 편집을 목표로 합니다. 특히, 'RECIPE'라는 새로운 기법을 소개하여 평생 동안의 연속적인 모델 편집 요구 사항에 대응합니다. RECIPE는 지식 문장을 짧고 정보성 높은 연속적인 프롬프트로 변환하고, LLM의 입력 쿼리 임베딩에 접두어로 추가하여, 지식에 근거한 응답을 효율적으로 정제합니다.

- **Technical Details**: RECIPE는 Knowledge Sentinel (KS)을 통합하여 동적 임계값을 계산합니다. 이는 검색 저장소가 관련 지식을 포함하고 있는지 여부를 결정합니다. 또한, 검색 엔진과 프롬프트 인코더가 공동으로 훈련되어 편집의 세 가지 주요 속성인 신뢰성(reliability), 일반성(generality), 지역성(locality)을 달성합니다.

- **Performance Highlights**: RECIPE는 다양한 LLM과 편집 데이터셋에서 광범위한 실험을 통해 우수한 편집 성능을 입증했습니다. 이 기법은 LLM의 전체 성능을 유지하면서 빠른 편집 및 추론 속도를 보여주는 능력을 시연했습니다.



### A Philosophical Introduction to Language Models - Part II: The Way  Forward (https://arxiv.org/abs/2405.03207)
- **What's New**: 이 논문은 대규모 언어 모델(Large Language Models, LLMs)의 최신 진보로 인해 제기되는 새로운 철학적 질문을 탐구합니다. 특히, 철학적 논의의 초점은 LLMs의 내부 표현과 계산의 본성에 대한 근거로 인과 개입 방법(Causal Intervention Methods)을 검토하는 것입니다. 또한, 멀티모달(Multimodal) 및 모듈러(Modular) 확장, LLMs가 의식의 최소 기준을 충족할 가능성에 대한 최근 논쟁, 그리고 LLM 연구의 비밀성과 재현성에 대한 우려를 논의합니다.

- **Technical Details**: 본 논문에서는 LLMs의 내부 표현과 계산 과정을 이해하기 위한 인과 개입 방법을 중점적으로 분석합니다. 이를 통해 LLMs의 복잡한 내부 동작을 좀 더 명확히 이해하고자 합니다. 또한, 멀티모달과 모듈러 확장을 통해 LLMs의 기능을 어떻게 향상시킬 수 있는지 살펴봅니다.

- **Performance Highlights**: 이 연구는 LLMs의 내부 메커니즘에 대한 이해를 심화시키고, 멀티모달 및 모듈러 확장 가능성을 탐구함으로써 LLMs의 성능과 응용 범위를 향상시키는 데 기여합니다. 또한, 현재 LLM 연구의 한계와 도전을 명확히 하여, 향후 연구 방향 설정에 중요한 가이드라인을 제공합니다.



### Vietnamese AI Generated Text Detection (https://arxiv.org/abs/2405.03206)
- **What's New**: 이 연구는 AI가 생성한 텍스트와 인간이 작성한 텍스트를 구별하는 것을 목표로 합니다. 새롭게 개발된 'ViDetect' 데이터셋은 베트남어 에세이 6,800개 샘플로 구성되어 있으며, 이 중 절반은 인간이 작성하고 나머지 절반은 LLM(Large Language Models)이 생성했습니다. 이는 베트남어 맥락에서 AI 생성 텍스트 탐지 연구에 중요한 기여를 합니다.



### Anchored Answers: Unravelling Positional Bias in GPT-2's Multiple-Choice  Questions (https://arxiv.org/abs/2405.03205)
Comments: Work in process

- **What's New**: 이 연구에서는 GPT-2 모델의 대표적인 문제점 중 하나인 위치적 편향(position bias)과 고정된 편향(anchored bias)에 대해 새로운 해결 방법을 모색하였습니다. 특히, GPT-2 모델이 다지선다형 문제(Multiple-Choice Questions, MCQs)에서 처음 선택지 'A'를 고르는 경향이 있는 고정된 편향을 집중적으로 분석하고, 이를 중화시키기 위한 최소 개입 전략을 제시하였습니다. 이러한 편향은 진단 및 개선 방법이 기존에 충분히 탐구되지 않았기 때문에 이번 연구가 갖는 의의가 큽니다.

- **Technical Details**: 이 연구는 'logit lens' 방법을 사용하여 GPT-2의 MLP (Multi-Layer Perceptron)층과 attention head에서 편향을 유발하는 특정 벡터들을 식별하고 수정하는데 초점을 맞추었습니다. 이를 통해 첫 번째 선택지 'A'에 대한 선호도를 중화하고, 전체 MCQ 예측 정확도를 향상시켰습니다. 또한, attention 패턴을 재조정하여 고정된 위치와 정확한 답변 위치 간의 attention 가중치를 교체하는 새로운 전략을 제안하였습니다.

- **Performance Highlights**: 이러한 개입은 GPT-2 모델군의 MCQs 예측 정확도를 평균 70% 이상 향상시켰으며, 특히 GPT2-Medium 모델에서는 간접 객체 식별(Indirect Object Identification, IOI) 데이터셋의 분류 정확도가 90% 이상으로 상당히 개선되었습니다. 이는 GPT-2 모델의 강건성과 정확성을 크게 향상시키는 결과를 가져왔습니다.



### Oracle-Checker Scheme for Evaluating a Generative Large Language Mod (https://arxiv.org/abs/2405.03170)
- **What's New**: 이 연구는 생성적 대규모 언어 모델(LLM)이 주어진 답변을 평가하기 위한 새로운 접근 방식인 오라클-체커(oracle-checker) 체계를 제시합니다. 속성 테스팅(property testing)과 프로그램 검사(program checking)의 아이디어를 따른 두 유형의 체커를 소개하고, 엔티티 추출(entity extraction)과 패러프레이즈 결정(paraphrase decision) 등 두 가지 상황에서 이들의 적용을 보여줍니다.

- **Technical Details**: 오라클-체커 체계에서, LLM은 오라클로 사용되며 입력 x에 대해 출력 y를 제공합니다. 해당 출력은 체커에 의해 검증되어, 출력 y가 주어진 함수 f에 대한 올바른 결과인지를 판단합니다. 체커는 오라클의 답변을 받아들일지 여부를 결정하기 위해 일련의 질의를 수행하고, 오라클의 추가 답변을 바탕으로 최종적으로 출력을 수용하거나 거부합니다. 이 연구에서는 엔티티 추출을 위한 선형 복잡도 검사(linearity checker)와 패러프레이즈 결정을 위한 체커 설계를 제시합니다.

- **Performance Highlights**: 제안된 오라클-체커 체계는 LLM의 답변의 신뢰성을 평가하는 데 도움을 주며, 두 가지 상황에서 구현 방법을 시연함으로써 LLM을 활용하는 새로운 방법을 탐색합니다. 이러한 체계는 LLM이 제공하는 변동성 있는 출력을 자동으로 검증할 필요가 있는 경우에 특히 유용할 수 있습니다.



### Exploring the Potential of the Large Language Models (LLMs) in  Identifying Misleading News Headlines (https://arxiv.org/abs/2405.03153)
Comments: 5 pages, 2 tables, 1st HEAL Workshop at CHI Conference on Human Factors in Computing Systems, May 12, Honolulu, HI, USA 2024

- **What's New**: 이 연구는 대규모 언어 모델 (Large Language Models, LLMs)이 어떻게 오도하는 뉴스 헤드라인과 그렇지 않은 헤드라인을 구별하는지 탐구합니다. 특히, ChatGPT-3.5, ChatGPT-4, 그리고 Gemini 모델을 이용해 다양한 도메인의 헤드라인을 분류하였습니다. 이는 디지털 시대에 정보의 진실성을 유지하기 위한 강력한 검출 메커니즘의 필요성을 강조합니다.

- **Technical Details**: 연구는 건강, 과학기술, 비즈니스 분야에서 출처가 믿을 수 있는 매체와 믿을 수 없는 매체 양쪽으로부터 모은 60개의 기사 데이터셋을 사용했습니다. 세 가지 언어 모델 ChatGPT-3.5, ChatGPT-4, Gemini를 사용하여 헤드라인을 분류했는데, 이 때 ChatGPT-4가 특히 높은 정확도를 보였습니다. 연구는 인간 중심 평가의 중요성을 강조하며, 기술적인 능력과 더불어 정교한 인간의 판단과 일치하도록 하는 방향으로 LLM의 개발이 이뤄져야 함을 시사합니다.

- **Performance Highlights**: ChatGPT-4는 오도하는 뉴스 헤드라인을 판단할 때 훨씬 높은 정확도를 보여 주었으며, 특히 평가자 간 일치하는 경우에 뛰어난 성능을 나타냈습니다. 이는 ChatGPT-4가 도덕적이고 민감한 인간 해석의 미묘함에 더 잘 맞춰져 있음을 의미합니다. 이 결과는 인공지능 윤리에 대한 논의에 기여할 수 있으며, 기술적으로 진보된 것뿐만 아니라 윤리적으로도 부합하고 인간 해석의 세밀한 차이를 이해할 수 있는 모델의 필요성을 강조합니다.



### CRAFT: Extracting and Tuning Cultural Instructions from the Wild (https://arxiv.org/abs/2405.03138)
Comments: 6 pages

- **What's New**: 이 논문은 광범위한 비정형(corpus)에서 고품질의 문화 관련 교육 튜닝 데이터셋을 추출하기 위한 새로운 파이프라인을 소개합니다. 문화 개념을 식별하고 지시 사항을 트리거하기 위해 자기 지시 생성(self-instruction generation) 파이프라인을 활용합니다. 이를 통해 모델은 지역 문화의 미묘한 차이를 인식하고 이해하는 능력이 향상되었습니다.

- **Technical Details**: 이 연구는 대규모 비정형 데이터 집합에서 문화적 개념을 식별하고 관련 지시를 생성할 수 있는 자기 지시 생성 파이프라인을 구현합니다. 일반적인 목적의 교육 튜닝 데이터셋(instruction tuning dataset)과 통합함으로써, 이 모델은 지역적 문화적 뉘앙스를 인식하고 이해하는 능력을 강화합니다.

- **Performance Highlights**: 싱가포르, 필리핀, 미국 세 지역에서 실험을 수행하여 최대 6%의 성능 향상을 달성했습니다. 이 연구는 비정형 데이터에서 직접 문화 교육 튜닝 세트를 추출하는 새로운 방법을 제시하며, 필드에서의 미래 혁신을 위한 선례를 설정합니다.



### Lory: Fully Differentiable Mixture-of-Experts for Autoregressive  Language Model Pre-training (https://arxiv.org/abs/2405.03133)
Comments: 21 pages, 12 figures

- **What's New**: 이 논문에서는 자동회귀 언어 모델의 사전 훈련(pre-training)으로 확장할 수 있는 첫 번째 접근법인 Lory를 소개합니다. Lory는 기존의 MoE(models of experts) 아키텍처를 활용하여 전문가들(experts)이 매개변수 공간에서 부드럽게 합쳐지는 방식을 채택합니다. 이는 자동회귀 언어 모델의 특성을 보존하면서도 효율적인 전문가 병합 작업을 가능하게 합니다.

- **Technical Details**: Lory는 두 가지 주요 기술을 도입합니다: (1) causal segment routing: 자동회귀 언어 모델의 특성을 보존하며 전문가 병합 작업의 효율성을 높이는 방법입니다. (2) similarity-based data batching: 비슷한 문서들을 그룹화하여 훈련 인스턴스에서 전문가의 전문성을 증진시키는 방법입니다. 이런 기술들을 통해 Lory는 최대 32명의 전문가와 30B(실질적으로 활성화된 1.5B) 매개변수를 갖는 모델을 150B 토큰에서 처음부터 사전 훈련할 수 있습니다.

- **Performance Highlights**: 실험 결과, Lory 모델은 parameter-matched 밀집 모델(dense models)과 비교하여 퍼플렉시티(perplexity)에서 13.9%의 성능 향상을 보였으며, 다양한 하류 작업(downstream tasks)에서도 1.5%-11.1%의 성과 향상을 보였습니다. 또한, 토큰 수준 라우팅(token-level routing)을 사용하는 최신 MoE 모델과 비교하여 경쟁력 있는 성능을 달성했으며, 훈련된 전문가들이 감독 없이 도메인 수준의 전문성을 포착하는 것을 확인할 수 있었습니다.



### An Active Inference Agent for Simulating Human Translation Processes in  a Hierarchical Architecture: Integrating the Task Segment Framework and the  HOF taxonomy (https://arxiv.org/abs/2405.03111)
- **What's New**: 이 연구에서는 번역 생성을 센서모터(sensorimotor), 인지(cognitive), 그리고 현상(phenomenal)의 세 가지 계층으로 구성된 계층구조로 모델링하는 새로운 접근 방식을 제안합니다. 이 구조는 키스트로크 생성의 시간적 역학을 재현합니다.

- **Technical Details**: 제안된 아키텍처는 CRITT TPR-DB, Task Segment Framework, 및 HOF 분류법을 활용하여, 세 가지 계층 내에서의 타이핑 흐름의 시간적 분해를 시연합니다. 이는 각 계층에서의 입력 동작이 어떻게 서로 다른 타임라인에서 분해되는지를 보여줍니다.

- **Performance Highlights**: 이 연구는 번역 과정에서의 키스트로크 생산의 세부적인 시간 구조를 분석함으로써, 번역기의 성능 개선에 기여할 수 있는 중요한 인사이트를 제공하는 것을 목표로 합니다. 세 계층에서의 데이터를 통합 분석함으로써 보다 정밀한 번역 시스템 설계가 가능해집니다.



### FairMonitor: A Dual-framework for Detecting Stereotypes and Biases in  Large Language Models (https://arxiv.org/abs/2405.03098)
- **What's New**: 이 연구에서는 Large Language Models (LLMs)에서의 편견과 고정관념을 발견하기 위한 새로운 프레임워크인 FairMonitor를 제안합니다. 이 프레임워크는 정적 및 동적 방법론을 결합하여 LLM이 생성한 콘텐츠에서의 스테레오타입 및 편견을 광범위하게 평가합니다. 정적 검사는 명시적 편견, 미묘한 편견, 그리고 새로운 시나리오에 대한 평가를 포함하는 세 가지 테스트로 구성됩니다. 동적 검사는 LLM 기반 다중 에이전트(Multi-Agent) 시스템을 활용해 실제 상황과 유사한 상호작용을 통해 감지합니다.

- **Technical Details**: FairMonitor는 직접 문의 테스트(Direct Inquiry Test), 암묵적 연합 테스트(Implicit Association Test), 그리고 알려지지 않은 상황 테스트(Unknown Situation Test)를 포함하는 정적 검사 방법과, 다양한 교육 시나리오에서 LLM의 상호 작용을 평가하는 동적 검사 방법을 통합하였습니다. 정적 구성 요소는 10,262개의 개방형 질문과 9가지 민감 요인, 26개의 교육 시나리오를 포함합니다. 동적 구성 요소는 600개의 다양한 교육 시나리오에서 LLM간의 상호 작용을 통해 미묘한 편향을 감지합니다.

- **Performance Highlights**: 실험 결과, FairMonitor는 다른 전통적인 방법들이 감지하지 못하는 편견과 고정관념을 효과적으로 탐지할 수 있음을 보여줍니다. 이 프레임워크는 교육 분야에서의 LLM 기반 애플리케이션에 적합하며, GPT-3.5-turbo, LLaMA2 시리즈, ChatGLM-6B, SenseChat 등 다양한 LLM에서 편견을 감지하는 능력을 입증하였습니다.



### Compressing Long Context for Enhancing RAG with AMR-based Concept  Distillation (https://arxiv.org/abs/2405.03085)
- **What's New**: 새로운 개념 기반 Retrieval Augmented Generation (RAG) 프레임워크가 제안되었습니다. 이 프레임워크는 Abstract Meaning Representation (AMR; 추상 의미 표현) 기반의 개념 증류 알고리즘을 사용하여 검색된 문서의 중요 개념들만을 추출하고, LLMs가 핵심 정보에만 집중할 수 있도록 유도합니다. 이것은 처음으로 AMR을 통해 RAG를 강화하는 작업입니다.

- **Technical Details**: 제안된 프레임워크는 AMR을 사용하여 복잡하고 방대한 정보를 포함하는 검색된 문서들을 필수 개념들로 압축합니다. 이 알고리즘은 문서로부터 정보의 노드들을 분석하여 신뢰할 수 있는 언어적 특징을 참조해 중요 개념들을 증류해 내는 과정입니다. 그 후, 이 개념들이 LLM의 추론 과정에서 관심을 가질 정보를 명확하게 제한합니다.

- **Performance Highlights**: 제안된 개념 기반 RAG 프레임워크는 오픈 도메인 질의응답 데이터셋에서 다양한 기본 LLMs와 비교해 우수한 성능을 보였습니다. 특히 지원 문서의 숫자가 많아질수록 성능이 개선되었으며, 다양한 LLM의 구조에도 견고함을 나타냈습니다. 이 결과는 개념 증류가 RAG 과정을 강화하는데 정보적으로 효과적임을 강조합니다.



### Analyzing Emotional Trends from X platform using SenticNet: A  Comparative Analysis with Cryptocurrency Pric (https://arxiv.org/abs/2405.03084)
- **What's New**: 이 연구는 2022년 10월부터 2023년 3월까지 X 플랫폼의 감정 경향과 카르다노(Cardano), 바이낸스(Binance), 팬텀(Fantom), 매틱(Matic), 리플(Ripple)과 같은 유명 암호화폐들의 시장 동향 간의 관계를 탐구했습니다. SenticNet을 이용하여 공포와 불안(Fear and Anxiety), 분노(Rage and Anger), 슬픔(Grief and Sadness), 기쁨(Delight and Pleasantness), 열정(Enthusiasm and Eagerness), 즐거움(Delight and Joy) 등의 감정을 식별했습니다.

- **Technical Details**: 연구에서는 감정 데이터를 추출한 후 매달을 2주 간격으로 세분화하여 감정 경향을 분석했습니다. 연구팀은 가격 데이터를 Finance-Yahoo에서 얻은 후, 같은 방법으로 2주 간격으로 세분화했습니다. 이렇게 해서 수집된 감정 및 가격 데이터에 대해 비교 분석을 실시하여 감정 경향과 암호화폐 가격 간의 상관 관계를 분석하였습니다.

- **Performance Highlights**: 분석 결과, 감정 트렌드와 암호화폐 가치 사이에 유의미한 상관 관계(correlation)를 확인할 수 있었습니다. 이는 특정 감정이 암호화폐의 가격 변동에 영향을 미칠 수 있음을 시사합니다.



### Exploring prompts to elicit memorization in masked language model-based  named entity recognition (https://arxiv.org/abs/2405.03004)
- **What's New**: 이 논문은 자동 생성된 다양한 프롬프트(400개)를 사용하여 6가지 공개된 NER(Named Entity Recognition) 모델에서 언어 모델의 훈련 데이터 기억(memorization)을 감지하는 데 프롬프트가 미치는 영향을 분석하는 최초의 연구입니다. 특히, 이 연구는 프롬프트의 종류(선언문, 감탄문, 명령문, 질문문), 프롬프트 내 토큰 길이, 토큰 위치가 모델의 기억 감지 성능에 어떻게 영향을 미치는지를 다룹니다.

- **Technical Details**: 이 연구에서는 CoNLL-2003 데이터셋을 사용하여 파인튜닝된 6개의 공개 NER 모델을 분석하고, Wikidata에서 샘플링한 대규모 인물 이름 쌍(pairwise dataset)을 이용해 모델 메모리제이션을 정량화합니다. 각 프롬프트는 PER(사람 이름)과 함께 완성된 문장으로 NER 모델에 입력되어, 훈련 데이터 내의 인물 이름(In-train PER)과 훈련 데이터 외의 인물 이름(Out-train PER)에 대한 모델의 신뢰도(confidence)를 비교합니다. 이러한 비교를 통해, 프롬프트의 성능이 인물 이름 기억 감지에 어떻게 기여하는지를 백분율로 나타냅니다.

- **Performance Highlights**: 프롬프트 선택에 따라 동일 모델에서 최대 16% 포인트까지 성능 차이가 발생함을 확인하였습니다. 또한 프롬프트 공학(prompt engineering)을 통해 성능을 최대 2% 포인트까지 향상시킬 수 있었습니다. 실험 결과는 프롬프트 성능이 모델에 따라 다르지만, 개발(dev) 데이터와 테스트(test) 데이터 간에는 일반화될 수 있음을 보여줍니다.



### MedAdapter: Efficient Test-Time Adaptation of Large Language Models  towards Medical Reasoning (https://arxiv.org/abs/2405.03000)
Comments: Work in Progress

- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)을 생물의학 분야에 적응시키는 새로운 접근방식으로 'MedAdapter'라는 통합된 사후(post-hoc) 어댑터를 제안하였습니다. 이 어댑터는 LLM 전체를 미세 조정(fine-tuning)하는 대신, BERT 크기의 어댑터만을 미세 조정하여 LLM이 생성한 후보 해결책들을 순위매기는 방식으로 효과적으로 원본 모델을 적응시킵니다.



### Can Large Language Models Make the Grade? An Empirical Study Evaluating  LLMs Ability to Mark Short Answer Questions in K-12 Education (https://arxiv.org/abs/2405.02985)
- **What's New**: 이 논문에서는 새롭게 개발된 데이터셋을 사용하여 대규모 언어 모델(Large Language Models, LLMs)이 짧은 답변형 문제에 대해 얼마나 정확하게 채점할 수 있는지를 조사한 실험 시리즈에 대해 보고합니다. 특히, 다양한 GPT 버전과 프롬프트 엔지니어링 전략(prompt engineering strategies)의 조합이 과학(Science)과 역사(History) 분야에서 다양한 연령대(5-16세) 학생들의 실제 답변을 평가하는 데 어떻게 작용하는지 탐구했습니다.

- **Technical Details**: 연구팀은 Carousel이라는 퀴즈 플랫폼에서 처음으로 사용되는 데이터셋을 바탕으로 GPT-4 모델을 사용하여 분석하였습니다. 연구에서는 기본적인 숏-샷(few-shot) 프롬프팅을 활용하여 GPT-4가 분석을 수행했습니다. 이는 문제에 대한 몇 가지 예시를 제시한 후 모델이 채점 기준을 학습하게 하는 방식입니다.

- **Performance Highlights**: GPT-4는 카파(Kappa)값 0.70으로, 인간 수준의 성능(0.75)에 매우 근접하게 성과를 보였습니다. 이는 GPT-4가 단답형 독해 문제를 전문가 수준에 가깝게 평가할 수 있음을 보여주는 것과 일치합니다. 이러한 결과는 다양한 과목과 학년에서 인간 수준의 평가에 근접함으로써, K-12 교육에서 저위험성 형성 평가(formative assessment) 작업을 지원하는 도구로서 LLMs가 가치가 있음을 시사합니다.



### E-TSL: A Continuous Educational Turkish Sign Language Dataset with  Baseline Methods (https://arxiv.org/abs/2405.02984)
Comments: 7 pages, 3 figures, 4 tables, submitted to IEEE conference

- **What's New**: 이 연구는 5학년, 6학년, 그리고 8학년을 대상으로 한 온라인 터키어 수업에서 수집한 연속적인 교육용 터키 수화(Educational Turkish Sign Language, E-TSL) 데이터셋을 소개합니다. 이 데이터셋은 11명의 수화 표현자들에 의해 수행된 약 24시간에 달하는 1,410개의 비디오로 구성되어 있습니다. 이 데이터셋은 터키어와 같은 응집어(agglutinative language)에서의 수화 번역에 대한 독특한 도전을 제시하며, 여기에는 64%가 소수 사용 단어(singleton words)이고 85%가 드문 단어(rare words)로 구성되어 있어 번역에 어려움을 겪습니다.

- **Technical Details**: 두 가지 기초 모델, Pose to Text Transformer(P2T-T)와 Graph Neural Network based Transformer (GNN-T)가 개발되었습니다. GNN-T 모델은 BLEU-1 점수 19.13%, BLEU-4 점수 3.28%를 달성하여 기존 벤치마크에 비해 상당한 도전을 제시합니다. P2T-T 모델은 BLEU 점수에서는 다소 낮은 성능을 보였지만, ROUGE-L 점수 22.09%로 더 높은 성능을 보였습니다.

- **Performance Highlights**: 이 연구에서 소개된 E-TSL 데이터셋과 두 가지 변형 모델(Transformer models)은 향후 연구를 위한 기준 모델(baseline models)로 제안됩니다. 특히 GNN-T 모델은 BLEU 점수에서 상대적으로 높은 성능을 보였고, P2T-T는 ROUGE-L에서 뛰어난 성적을 보여 수화 번역 분야에서의 새로운 가능성을 열었습니다.



### Unraveling the Dominance of Large Language Models Over Transformer  Models for Bangla Natural Language Inference: A Comprehensive Study (https://arxiv.org/abs/2405.02937)
Comments: Accepted in 4th International Conference on Computing and Communication Networks (ICCCNet-2024)

- **What's New**: 본 연구는 자연어 추론(Natural Language Inference, NLI)이 벵골어와 같은 저자원 언어의 평가에서 구현되는 모습을 탐색하는데 집중하였습니다. 대규모 언어 모델(Large Language Models, LLMs)과 최신(state-of-the-art, SOTA) 모델들의 성능을 벵골어 NLP 작업에서 비교 분석하였습니다. 특히, GPT-3.5 Turbo와 Gemini 1.5 Pro와 같은 LLMs와 BanglaBERT, mBERT 등과의 비교를 통해, 저자원 언어에 대한 LLM의 효과를 평가했습니다.

- **Technical Details**: 이 연구는 XNLI 데이터셋을 사용하여, 제로-샷(zero-shot) 및 퓨-샷(few-shot) 평가를 수행하였습니다. 벵골어, Bangla BERT Base, DistilBERT, 그리고 sahajBERT 모델들도 분석에 포함되었습니다. 이는 LLM이 SOTA 모델들에 비교하여 어떻게 동등한 혹은 더 나은 성능을 보일 수 있는지를 보여주는 중요한 과정입니다.

- **Performance Highlights**: 연구 결과, LLMs는 몇 가지 시나리오에서 SOTA 모델들을 초과하는 성능을 달성할 수 있음을 보여줍니다. 특히, 퓨-샷 설정에서의 성능이 두드러졌습니다. 그러나 LLMs의 모델 과신(model overconfidence)과 인간 판단 불일치를 캡쳐하는 데 있는 어려움은 여전히 해결해야 할 과제로 남아 있습니다.



### Enabling Patient-side Disease Prediction via the Integration of Patient  Narratives (https://arxiv.org/abs/2405.02935)
- **What's New**: 질병 예측은 현대 의료에서 중요한 의미를 지니며, 초기 개입을 용이하게 하고 효과적인 예방 조치를 구현하는 데 필수적입니다. 최근 질병 예측 방법은 주로 실험실 검사 결과 (예: 혈액 검사 및 X-rays의 의료 영상)에 의존하고 있습니다. 그러나 이러한 데이터에 접근하는 것은 환자의 입장에서 복잡하며, 항상 환자 상담 이후에만 가능합니다. 이러한 문제를 해결하기 위해, 우리는 환자의 건강 서술 및 인구 통계 정보를 사용하여 질병을 예측하는 Personalized Medical Disease Prediction (PoMP)를 제안합니다.

- **Technical Details**: PoMP 모델은 환자로부터 제공된 텍스트 기반의 건강 서술과 인구 통계 정보를 활용하여 질병을 예측합니다. 이를 통해 환자들은 자신의 건강 상태에 대해 더 명확하게 이해하고, 적절한 의료 전문가에게 직접 문의할 수 있게 되어, 적합한 의사를 찾기 위해 의료 커뮤니케이션을 탐색하는 데 소요되는 시간을 줄일 수 있습니다.

- **Performance Highlights**: 실제 세계 데이터(Haodf에서 수집)를 사용한 광범위한 실험을 통해 PoMP의 효과성을 입증했습니다. 이 모델은 환자의 자가 보고된 데이터만을 사용하여, 기존의 의료 검사 기반 방법에 비해 접근성을 크게 향상시킵니다.



### Relay Decoding: Concatenating Large Language Models for Machine  Translation (https://arxiv.org/abs/2405.02933)
Comments: Work in progress

- **What's New**: 새로운 RD (Relay Decoding) 접근 방법이 제안되었습니다. 이 방법은 각각 소스 언어와 타겟 언어를 지원하는 두 개의 큰 모델(large models)을 연결하여 사용합니다. 이 연결을 위해 간단한 매핑 레이어(mapping layer)가 도입되었으며 제한된 양의 병렬 데이터(parallel data)를 사용하여 훈련합니다.

- **Technical Details**: RD 방법은 기계 번역(machine translation)에 있어서 하나의 큰 언어 모델이 소스와 타겟 언어를 모두 처리해야 하는 문제를 해결합니다. 대신, 각각의 언어를 지원하는 두 모델을 사용하며, 이들 사이의 연결을 위한 매핑 레이어가 포함됩니다. 이 연구에서는 Multi30k 및 WikiMatrix 데이터셋을 사용하여 실험을 수행했습니다.

- **Performance Highlights**: 제안된 RD 방법은 기존의 방법들과 비교하여 우수한 결과를 달성했습니다. 특히 Multi30k와 WikiMatrix 데이터셋에서 실험을 진행한 결과, 효율성과 정확성이 입증되었습니다.



### A Two-Stage Prediction-Aware Contrastive Learning Framework for  Multi-Intent NLU (https://arxiv.org/abs/2405.02925)
Comments: LREC-COLING 2024

- **What's New**: 새로운 두 단계 예측 인식 대조 학습 (Prediction-Aware Contrastive Learning, PACL) 프레임워크를 도입하여 다중 의도 자연어 이해 (multi-intent NLU)에 대해 처리합니다. 이 방법은 공유된 의도 정보를 통합하여 모델의 성능을 향상시키기 위해 word-level 사전 훈련과 예측 인식 대조적 미세 조정을 통합합니다.

- **Technical Details**: PACL은 먼저 word-level 데이터 증강 전략을 사용하여 사전 훈련 데이터셋을 구축합니다. 미세 조정 단계에서는 동적으로 인스턴스 역할을 할당하고 예측 인식 대조 손실을 도입하여 대조 학습의 영향을 극대화합니다. 또한 의도-슬롯 (intent-slot) 주의 메커니즘을 설계하여 의도 감지 (multi-intent detection, mID) 및 슬롯 채우기 (slot-filling, SF) 작업 간 강력한 연결을 구축합니다.

- **Performance Highlights**: PACL은 MixATIS, MixSNIPS, StanfordLU와 같은 세 가지 주요 다중 의도 데이터셋에서 실험을 수행하였으며, RoBERTa, TFMN, SLIM과 같은 세 가지 주요 기준을 사용하여 비교하였습니다. 실험 결과 PACL은 낮은 데이터 및 전체 데이터 시나리오 모두에서 기준 모델의 성능을 능가하며 수렴 속도를 가속화하는 데 있어 뚜렷한 향상을 보였습니다.



### Sentiment Analysis Across Languages: Evaluation Before and After Machine  Translation to English (https://arxiv.org/abs/2405.02887)
Comments: 6 pages, 3 Figures

- **What's New**: 이번 연구에서는 전 세계적으로 사용되는 7,000여 개의 언어와 인도에서만 780개에 달하는 언어들을 포함하여 다양한 언어 데이터셋을 대상으로 한 Sentiment Analysis(감성 분석)의 연구 동향을 살펴보았습니다. 특히, 이 연구는 영어 외의 다양한 언어에서 Transformer(트랜스포머) 모델들이 어떻게 작동하는지를 검토하고, 기계 번역을 거친 텍스트에서의 모델 효과성을 비교 분석하였습니다.

- **Technical Details**: 본 연구는 Transformer 모델들을 사용하여 다양한 언어로 작성된 데이터셋과 기계 번역된 텍스트에서의 Sentiment Analysis를 수행하였습니다. 이를 통해, 각 언어 맥락에서 모델의 성능 차이를 파악하고, 이러한 차이가 감성 분석 결과에 미치는 영향에 대한 통찰을 제공합니다.

- **Performance Highlights**: Transformer 모델들은 영어 데이터셋에 비해 다국어 데이터셋과 기계 번역된 텍스트에서 상대적으로 낮은 성능을 보였으나, 그럼에도 불구하고 다양한 언어에 대한 적용 가능성을 확인할 수 있었습니다. 이러한 결과는 향후 다양한 언어 자원의 개발과 감성 분석 기술의 균형 있는 발전 방향을 제시합니다.



### Revisiting a Pain in the Neck: Semantic Phrase Processing Benchmark for  Language Models (https://arxiv.org/abs/2405.02861)
Comments: 24 pages, 17 figures, 10 tables

- **What's New**: LexBench는 문장 내 의미 구문 (semantic phrase) 처리 업무에 대한 언어 모델 (Language Models, LM) 평가를 가능하게 하는 종합 평가 스위트를 소개합니다. 이전 연구들과 달리, 일반 의미 구문(lexical collocation)과 세분화된 의미 구문을 포함하여 모델링하기 위한 비교적 관점의 프레임워크를 제안하는 최초의 작업입니다.

- **Technical Details**: LexBench는 구문 분류(classification), 추출(extraction), 해석(interpretation) 작업을 포함하여 15개의 언어 모델을 다양한 모델 아키텍처와 파라미터 규모에서 평가합니다. 테스트된 의미 구문들은 일반적 구문(lexical collocation), 숙어(idiomatic expression), 명사 복합어(noun compound), 동사 구조(verbal construction)를 포함합니다.

- **Performance Highlights**: 실험을 통해 큰 모델이 대부분의 작업에서 작은 모델보다 우수하다는 것을 확인하며, 스케일링 법칙을 검증합니다. 또한 의미 관계 분류(semantic relation categorization)에서는 few-shot 학습 모델이 여전히 일반적인 미세조정 모델보다 뒤처진다는 것을 발견했습니다. 강인한 모델의 성능이 인간 수준에 맞먹는다는 평가를 통해 확인되었습니다.



### HuixiangDou-CR: Coreference Resolution in Group Chats (https://arxiv.org/abs/2405.02817)
Comments: 5 pages, 3 tables, 3 figures

- **What's New**: 이 연구에서는 그룹 채팅에서 대명사 참조를 제거하는 새로운 방법을 제시합니다. 58k의 실제 채팅 데이터를 전처리하고 2.3k의 질문을 수동으로 주석 처리하여 Qwen 모델을 이용한 최적화를 수행했습니다. 그 결과, F1 점수가 29.07포인트 향상되었으며, 이는 대규모 언어 모델 (LLM)을 자연어 처리 (NLP) 태스크에 효과적으로 활용할 수 있음을 입증합니다.

- **Technical Details**: 이 연구에서는 Supervised Fine-Tuning (SFT), Low-Rank Adaptation (LoRA)을 포함한 접근 방식을 사용하였습니다. 또한, HuixiangDou를 통해 광범위한 사용자 입력에서의 문제를 정의하고, Qwen LLM 시리즈 모델을 0.5B에서 32B까지 다양한 파라미터로 미세 조정하였습니다. 데이터는 alpaca 형식으로 전처리되었으며, 핵심 개념인 스케일링 법칙을 이용하여 데이터의 질을 검증하였습니다.

- **Performance Highlights**: Qwen1.5-MoE-2.7B-Chat 모델의 F1 스코어는 32.07%에서 61.93%로 상당히 향상되었으며, Qwen1.5-14B-Chat 모델은 재현율이 68.91%에서 92.11%로 크게 증가하였습니다. 이러한 결과는 LLM을 활용한 NLP 태스크의 성능 향상 가능성을 보여 줍니다.



### Stochastic RAG: End-to-End Retrieval-Augmented Generation through  Expected Utility Maximization (https://arxiv.org/abs/2405.02816)
Comments: To appear in the proceedings of SIGIR 2024

- **What's New**: 이 논문은 검색-증강 생성(Retrieval-Augmented Generation, RAG) 모델의 종단 간 최적화를 위해 'Stochastic RAG'라는 새로운 접근 방식을 소개합니다. 기존 연구에서 흔히 볼 수 있는 문서의 독립성과 마진화라는 단순화 가정을 완화하여, 비복원 추출 방식을 모방하는 확률적 샘플링을 통해 RAG의 검색 프로세스를 구현합니다.

- **Technical Details**: Stochastic RAG는 straight-through Gumbel-top-k 기법을 사용하여 비복원 샘플링에 대한 미분 가능한 근사치를 제공하며, 이를 통해 RAG의 효과적인 종단 간 최적화를 가능하게 합니다. 또한, 7개의 다양한 데이터셋에서 광범위한 작업을 수행하여 우수한 성능을 입증합니다. 이는 최근 유효한 RAG 모델에 이 최적화 방법을 적용하여 7개 중 6개 데이터셋에서 최첨단(State-of-the-art) 결과를 달성했다는 점에서 주목할 만한 성과입니다.

- **Performance Highlights**: 'Stochastic RAG'의 종단 간 최적화 방법을 적용한 FiD-Light 모델은 특정 기준(Exact Match, BLEU, ROUGE 등)에 따라 측정된 거의 모든 데이터셋에서의 성능 향상을 보여주었습니다. 예를 들어, 오픈 도메인 질의응답, 사실 검증, 담화 시스템, 관계 추출을 위한 슬롯 채우기 등 다양한 응용 분야에서 성능이 크게 개선되었습니다.



### NegativePrompt: Leveraging Psychology for Large Language Models  Enhancement via Negative Emotional Stimu (https://arxiv.org/abs/2405.02814)
Comments: This paper has been accepted by IJCAI 2024

- **What's New**: 이 연구는 NegativePrompt라는 새로운 접근방식을 도입하여 감정지능을 가진 대규모 언어 모델(Large Language Models, LLMs)의 성능을 향상시키기 위해 부정적 감정 자극을 사용합니다. 이 방식은 심리학적 원리에 기초하여 개발되었으며, 다양한 LLMs가 보다 효과적으로 작동할 수 있도록 돕습니다.

- **Technical Details**: NegativePrompt는 10개의 특별히 설계된 부정적 감정 자극을 포함하며, Flan-T5-Large, Vicuna, Llama 2, ChatGPT, GPT-4 등의 LLM들을 대상으로 45개의 작업에서 평가되었습니다. 이론적 배경으로는 Cognitive Dissonance Theory (인지 부조화 이론), Social Comparison Theory (사회 비교 이론), 등의 심리학 이론이 적용되었습니다.

- **Performance Highlights**: NegativePrompt는 Instruction Induction (지시 유도) 작업에서 12.89%의 성능 개선과 BIG-Bench 작업에서 46.25%의 탁월한 성능 향상을 보였습니다. 이 연구는 또한 attention visualization (주의 시각화) 실험을 통해 LLM들이 부정적 감정 자극에 어떻게 반응하는지에 대한 내부 메커니즘을 탐구했습니다.



### Detecting Edited Knowledge in Language Models (https://arxiv.org/abs/2405.02765)
- **What's New**: 이 연구는 대규모 언어 모델(Large Language Models, LLMs)에서 편집된 지식을 감지하는 새로운 작업을 제안합니다. 특히, 저자들은 편집된 모델과 특정 지식 조각을 구분하는 RepReg이라는 간단한 분류기를 제안했으며, 이는 편집된(Edited) 지식과 비편집된(Non-edited) 지식을 구별하는 데 사용됩니다.

- **Technical Details**: RepReg는 로지스틱 회귀 모델(Logistic Regression)로, 생성된 토큰의 숨겨진 상태 표현(Hidden State Representations)을 입력 특성으로 사용하여 지식이 편집되었는지 여부를 분류합니다. 이 연구는 ROME과 MEND 같은 최신 지식 편집 기술(Knowledge Editing Techniques, KEs)과 두 가지 언어 모델, 두 가지 데이터셋을 사용하여 실험을 수행했습니다.

- **Performance Highlights**: RepReg는 최대 99.81%의 높은 정확도를 달성했으며, 훈련 샘플이 200개인 제한된 학습 세트에서도 거의 최적의 성능을 보였습니다. 또한 이 분류기는 도메인 외 설정(Out-of-Domain Settings)에서도 97.79%의 높은 정확도를 유지하며, 강력한 기준선을 설정했습니다.



### Assessing Adversarial Robustness of Large Language Models: An Empirical  Study (https://arxiv.org/abs/2405.02764)
Comments: 16 pages, 9 figures, 10 tables

- **What's New**: 이 연구는 대규모 언어 모델(LLM)의 새로운 white-box 스타일 공격 방법을 제시하여, Llama, OPT, T5 등 주요 오픈소스 LLM에 대한 취약성을 폭로합니다. 여러 NLP 분류 작업에서의 모델의 강인성을 평가함으로써, 신뢰할 수 있는 AI 시스템의 발전에 기여합니다.

- **Technical Details**: 이 연구는 adversarial geometry attack technique을 사용하여 입력 변화에 대한 모델의 취약성을 평가하고, LoRA와 같은 널리 사용되는 LLM 훈련 방법의 효과를 조사합니다. 또한 모델의 크기와 구조 변형, fine-tuning 전략의 영향도 분석합니다.

- **Performance Highlights**: 연구 결과는 다섯 가지 텍스트 분류 작업에 걸쳐 LLM의 능력과 한계를 보여줍니다. 특히, 다양한 공격 시나리오에서 모델의 정확도에 미치는 영향을 중점적으로 평가하여, LLM의 강인성에 대한 새로운 벤치마크(benchmark)를 설정합니다.



### Enhancing Contextual Understanding in Large Language Models through  Contrastive Decoding (https://arxiv.org/abs/2405.02750)
Comments: Accepted to NAACL 2024

- **What's New**: 이 연구는 대형 언어 모델(LLMs)이 입력 컨텍스트를 어떻게 효과적으로 통합할 수 있는지에 대한 새로운 접근 방식을 제시합니다. 특히, 기존 지식과 입력된 프롬프트의 컨텍스트 사이의 균형을 맞추기 위하여 대조적 디코딩(contrastive decoding) 방법과 적대적으로 생성된 불필요한 정보를 사용합니다. 이 방법은 추가 트레이닝 없이 추론 시점에서 작동합니다.

- **Technical Details**: 제안된 방법은 불필요한 컨텍스트를 통합하여 모델이 부정확한 응답에서 벗어나도록 유도합니다. 이는 대조적 디코딩과 적대적 검색을 통해 이루어집니다. 모델은 입력된 질문에 대해 신뢰성 있는 답변을 생성하기 위해 필요한 지식을 적절히 선택하고 통합할 수 있는 능력을 갖추게 됩니다.

- **Performance Highlights**: 다양한 데이터셋(Natural Questions, TriviaQA, PopQA 등)에서 실시한 실험을 통해, 기존 방법들에 비해 제안된 방법이 지식 충돌을 관리하고 문맥을 통합하는 데 더욱 효과적임을 입증하였습니다. 특히 대형 모델에서 더욱 뛰어난 성과를 보였으며, 다양한 검색 소스와 무관한 컨텍스트 선택이 결과에 미치는 영향도 탐구하였습니다.



### Beyond Performance: Quantifying and Mitigating Label Bias in LLMs (https://arxiv.org/abs/2405.02743)
Comments: NAACL 2024

- **What's New**: 이 연구에서는 대규모 언어 모델 (LLM)의 레이블 편향 (label bias) 문제를 평가하고 측정하는 새로운 접근 방식을 제안합니다. 연구팀은 다양한 분류 작업과 LLM들을 대상으로 실험을 진행, 기존의 교정 방법들을 뛰어넘는 새로운 레이블 편향 보정 방법을 제시했습니다. 이는 소수 예시 (few-shot) 프롬프팅에서 탁월한 성능을 보이면서 레이블 편향을 줄일 수 있는 방법으로 평가받았습니다.

- **Technical Details**: 연구팀은 279개의 다양한 분류 작업과 10개의 LLM을 사용하여 실험을 수행했습니다. 각 모델의 레이블 편향을 측정하기 위하여 공정성과 레이블 편향 추정에 대한 이전 연구에서 유래된 메트릭스(metrics)를 사용했습니다. 또한, 이 연구는 LoRA fine-tuning 및 캘리브레이션(calibration)과 같은 편향 완화 방법의 영향도 평가했습니다. 특히, 연구팀은 기존 방법들과 비교하여 성능 개선은 물론 레이블 편향 제거에 있어 더 나은 새로운 보정 방법을 제안하고 검증했습니다.

- **Performance Highlights**: 분석 결과, 대다수의 LLM은 여전히 상당한 레이블 편향을 나타냈으며, 이는 모델의 신뢰성에 제약을 주는 주요 요소로 확인되었습니다. 새롭게 제안된 보정 방법은 기존 방법들에 비해 레이블 편향을 더 효과적으로 줄이는 것으로 나타났습니다. 이는 레이블 편향 감소뿐만 아니라 전반적인 성능 향상에도 기여했습니다. 연구 결과는 LLM의 성능 평가 시 레이블 편향을 고려하고 측정하는 것의 중요성을 강조하며, 보다 정확하고 효과적으로 편향을 추정하고 조정하는 것이 LLM의 신뢰성과 활용도를 높일 수 있다는 가능성을 보여줍니다.



### Relations Prediction for Knowledge Graph Completion using Large Language  Models (https://arxiv.org/abs/2405.02738)
- **What's New**: 본 연구에서는 지식 그래프(Knowledge Graph)의 노드 이름만을 사용하여 관계 예측(relation prediction) 작업을 위해 대규모 언어 모델을 미세 조정(fine-tune)하는 방법을 제안합니다. 이 접근 방식은 지식 그래프가 불완전한 문제를 해결하고자 하는 새로운 시도입니다.

- **Technical Details**: 지식 그래프의 노드 이름을 활용하여, 인덕티브 설정(inductive settings)에서도 효과적으로 작동할 수 있도록 모델을 조정합니다. 이는 노드 간의 관계를 예측하는데 중요한 요소가 됩니다. 대규모 언어 모델을 사용하여 이러한 노드 이름 정보만을 바탕으로 관계를 예측합니다.

- **Performance Highlights**: 이 방법을 통해 널리 사용되는 지식 그래프 벤치마크에서 새로운 점수를 달성하였으며, 이는 기존 방법론 대비 높은 성능을 의미합니다. 연구 결과에 따르면, 노드의 명칭만을 활용해도 충분히 높은 정확도의 관계 예측이 가능함을 입증하였습니다.



### Recall Them All: Retrieval-Augmented Language Models for Long Object  List Extraction from Long Documents (https://arxiv.org/abs/2405.02732)
- **What's New**: L3X 방법론을 소개합니다. 이는 대규모 언어 모델(LLM)을 이용한 새로운 관계 추출 방식으로 긴 텍스트에서 긴 객체 리스트를 추출하는 데 중점을 둡니다. 이 방법은 두 단계로 구성됩니다: (1) 후보 생성을 위한 리콜(recall) 지향 생성과 (2) 후보 검증을 위한 정밀도(precision) 지향 심사.

- **Technical Details**: 첫 번째 단계에서는 대상과 관계를 입력으로 사용하여 LLM에 프롬프트를 제공하고 객체의 전체 목록을 생성하도록 요청합니다. 이 과정에서 정보 검색(IR) 시스템을 사용하여 적절한 후보 텍스트를 찾아 LLM에 제공합니다. 두 번째 단계에서는 생성된 후보 목록을 까다롭게 검토하여 확신이 있는 객체를 확정하고 낮은 확신의 객체는 재평가합니다.

- **Performance Highlights**: L3X 방법은 LLM만을 사용하는 기존 방법보다 훨씬 우수한 성능을 보입니다. GPT-3.5-turbo를 사용하여 약 80%의 리콜과 48%의 R@P50, 30%의 R@P80을 달성했습니다. 이는 기존 방법이라고 할 수 있는 LLM 기반 추출 방식에 비해 상당한 향상을 나타냅니다.



### CoE-SQL: In-Context Learning for Multi-Turn Text-to-SQL with  Chain-of-Editions (https://arxiv.org/abs/2405.02712)
- **What's New**: 새로운 접근 방식인 CoE-SQL(Chain-of-Edition SQL)을 소개합니다. 이 방법은 대화형 멀티 턴 텍스트-투-SQL(Text-to-SQL) 작업을 처리하기 위해 이전에 생성된 SQL 쿼리를 기반으로 SQL 쿼리를 생성하는 방식을 제안합니다. 이는 LLMs(Large Language Models)를 사용하여 문맥 의존성을 활용하고, 이전 출력과의 차이만을 수정하여 새로운 쿼리 생성에 접근합니다.

- **Technical Details**: CoE-SQL은 이전 SQL 쿼리에서 몇 가지 작업으로 현재 SQL 쿼리를 수정하는 방식을 모델링합니다. 이는 Chain of Thought (CoT) 접근 방식의 한 형태로서, 사용자의 집중과 의도의 변화를 추적하는 비교적 이해하기 쉬운 추론 과정입니다. 또한, 이 방법은 Abstract Syntax Tree (AST) 비교 알고리즘을 사용하여 수정 규칙을 자동으로 추출하고, 각 턴의 출력 전에 이러한 수정 내용을 연속적으로 직렬화하는 작업을 포함합니다.

- **Performance Highlights**: CoE-SQL은 SParC와 CoSQL 벤치마크에서 최신(state-of-the-art, SOTA) 성능을 달성하였습니다. 이는 멀티 턴 설정에서 기존의 ICL(In-context Learning) 방식과 비교할 때 향상된 결과를 보여줍니다. CoE-SQL의 접근 방식은 사전 조정된(fine-tuned) 모델과 경쟁적인 성능을 내는 것으로 나타났습니다.



### Enhancing News Summarization with ELearnFit through Efficient In-Context  Learning and Efficient Fine-Tuning (https://arxiv.org/abs/2405.02710)
Comments: 9 Pages

- **What's New**: 이 연구에서는 뉴스 기사의 요약을 최적화하기 위해 큰 언어 모델(LLMs)의 두 가지 주요 기능인 ELearn(효율적인 인콘텍스트 학습)과 EFit(효율적인 파라미터 파인 튜닝)을 집중적으로 탐구했습니다. 특히, ELearnFit 모델을 만들어 두 기술의 통합을 통해 성능 향상을 이루었습니다.

- **Technical Details**: ELearn은 인콘텍스트 학습을 통해 모델이 명령어와 예시를 활용해 학습하는 과정을 향상시키는 데 초점을 맞추고 있으며, 이때 큰 모델 사용, 샷(shot) 수 증가, 단순 템플릿 활용이 성능을 향상시키는 것으로 나타났습니다. EFit에서는 LLM의 첫 번째 레이어를 파인 튜닝하는 것이 다른 레이어를 조정하거나 LoRA를 사용하는 것보다 더 우수한 결과를 냈습니다. ELearnFit은 두 기술을 통합하여 한정된 주석이 달린 샘플을 사용할 때 실질적인 구현이 가능함을 보여주었습니다.

- **Performance Highlights**: ELearn을 사용할 때, 큰 모델과 많은 샷 수, 간단한 템플릿이 성능 개선에 도움이 되는 것으로 확인되었습니다. EFit을 통해 첫 번째 레이어의 파인 튜닝이 다른 방식보다 우수한 결과를 제공했습니다. ELearn과 EFit을 결합한 ELearnFit 모델은 두 기법을 단독으로 사용할 때보다 뛰어난 성능을 보였으며, 이는 특히 주석이 제한적인 샘플에서 뉴스 요약을 효과적으로 최적화할 수 있는 실용적인 방법을 제시합니다.



### Evaluating the Ability of Computationally Extracted Narrative Maps to  Encode Media Framing (https://arxiv.org/abs/2405.02677)
Comments: Text2Story Workshop 2024

- **What's New**: 이 연구에서는 뉴스 데이터에서 서술(narrative) 추출 및 프레이밍(framing) 정보를 포착하는 능력에 초점을 맞추고 있습니다. 특히, 'narrative maps' 접근 방식을 사용하여 데이터 세트의 프레이밍 분포를 적절히 포착하고 일관된 프레이밍을 제공할 수 있는지를 평가합니다.

- **Technical Details**: 연구에서 사용된 'narrative maps'는 이야기의 복잡한 구조를 시각적으로 나타내는 데 사용되는 방향성 비순환 그래프(directed acyclic graph)입니다. 이 도구는 개별 사건들을 노드(node)로 표현하고, 사건들 간의 연결을 통해 전체 서사의 구조를 분석합니다. 데이터는 Gun Violence Frame Corpus (GVFC)에서 수집된 1300개의 뉴스 기사를 포함하고 있으며, 프레이밍 모델을 간소화하여 세 가지 주요 프레임으로 재분류하였습니다.

- **Performance Highlights**: 연구 결과에 따르면, 사용된 알고리즘은 데이터 세트의 프레이밍 분포를 포착하는 데 효과적이었지만, 시작 및 종료 사건을 고려할 때 일관된 프레이밍을 생성하는 데는 일부 도전이 있었습니다. 이는 프레이밍 동작의 복잡성을 이해하는데 'narrative maps'가 유용할 수 있음을 시사하지만, 계산적 서술 추출 과정에서 프레이밍 정보를 직접 활용하는 것은 여전히 개방된 과제입니다.



### On the Information Redundancy in Non-Autoregressive Translation (https://arxiv.org/abs/2405.02673)
Comments: 10 pages, 10 tables

- **What's New**: 이 연구에서는 완전 비자동 번역(NAT: Non-Autoregressive Translation)에서 흔히 발생하는 다중 모달 문제인 토큰 반복에 대해 재조명합니다. 이번 연구는 최근 제안된 NAT 모델들에서 다중 모달 문제를 다시 검토하고, 이 고급 모델들이 기존의 연속 반복 비율 지표로는 측정할 수 없는 다른 유형의 정보 중복 오류를 도입했다는 사실을 밝혔습니다.

- **Technical Details**: 연구팀은 NAT 출력을 수동으로 주석 처리하여 어휘(Lexical) 및 재배열(Reordering) 다중 모드 문제에 해당하는 두 가지 유형의 정보 중복 오류를 식별했습니다. 사람의 주석은 시간이 많이 걸리고 노동 집약적이므로, 연구팀은 이 두 유형의 중복 오류를 평가하기 위한 자동 메트릭을 제안했습니다.

- **Performance Highlights**: 제안된 자동 메트릭을 사용함으로써, 향후 연구에서 새로운 방법을 평가하고 그 효과에 대한 보다 포괄적인 이해를 얻을 수 있습니다. 이러한 메트릭은 정보 중복 오류의 종류를 효과적으로 식별하고 측정할 수 있도록 도와 줍니다.



### R4: Reinforced Retriever-Reorder-Responder for Retrieval-Augmented Large  Language Models (https://arxiv.org/abs/2405.02659)
- **What's New**: 이 연구에서는 '강화된 검색-재정렬-응답자(Reinforced Retriever-Reorder-Responder, R$^4$)'라는 새로운 파이프라인을 제안합니다. 이 방법은 정보 검색(retrieval)-보강된 대규모 언어 모델(LLMs)의 문서 순서를 학습하여 생성 능력을 향상시키는 것을 목표로 합니다. 특히, 문서 순서 조정과 문서 표현 강화의 두 단계로 나누어 접근함으로써, 문서의 세부 구조적 의미와 LLMs 간의 상호작용을 촉진합니다.

- **Technical Details**: 문서 순서 조정(document order adjustment)은 그래프 주의 학습(graph attention learning)을 통해 문서 순서를 시작, 중간, 끝 위치로 조직하여 응답 품질의 강화된 보상을 최대화하는 것을 목표로 합니다. 문서 표현 강화(document representation enhancement)는 문서 수준의 그래디언트 적대 학습(gradient adversarial learning)을 통해 응답 품질이 낮은 문서의 표현을 추가로 정제합니다.

- **Performance Highlights**: 이 프레임워크는 지식 집약적인 작업에서 강력한 기준 모델들(strong baselines)에 비해 더 나은 사실적 질문-응답 성능을 달성하였으며, 다양한 공개 데이터셋에서 그 효과를 입증하였습니다. R$^4$는 생성적 질문-응답(QA), 다중 선택 QA, 대화 관련 작업에서 모든 기준 모델들을 상당히 능가하는 성능을 보여주었습니다.



### Identifying Narrative Patterns and Outliers in Holocaust Testimonies  Using Topic Modeling (https://arxiv.org/abs/2405.02650)
Comments: 9 pages, 7 figures, LREC-COLING 2024

- **What's New**: 본 논문은 USC Shoah Foundation Holocaust 증언 데이터베이스를 활용하여 체계적인 질문-응답 섹션으로 처리되는 생존자 증언에 대해 고급 자연어 처리(Natural Language Processing, NLP) 기법을 적용하고 있습니다. 특히, 최신 언어 모델링 기술을 기반으로 하는 BERTopic을 활용하여 주요 테마를 식별하고, 이를 통해 증언들 간의 공통적인 내러티브 구조와 성별 및 연령에 따른 차이점을 밝혀냅니다.

- **Technical Details**: BERTopic은 문서 임베딩(all-MiniLM-L6-v2, Wang et al., 2020)과 TF-IDF 기반 클러스터링 접근 방식을 결합하여 전통적인 방법인 LDA (Latent Dirichlet Allocation, Blei et al., 2001)보다 문맥을 더 잘 반영할 수 있습니다. 이와 함께 UMAP (McInnes and Healy, 2018)을 통한 차원 축소와 HDBSCAN (McInnes et al., 2017)을 통한 클러스터링을 이용하여 유연하게 주제의 수를 결정합니다. 데이터셋에서는 58개의 주제가 도출되었으며, 각 섹션에서 중요 단어를 추출하여 주제를 해석하는 방식을 채택했습니다.

- **Performance Highlights**: 해당 연구를 통해, Holocaust 생존자의 증언에서 자주 등장하는 주제들과 시간이 지남에 따른 주제의 변화가 구조적으로 관찰되었습니다. 특히, 시작과 끝 부분에서 일관된 패턴이 확인되었으며, 중간 부분에서는 각 개인의 다양한 경험을 반영한 주제의 분포 변화가 두드러졌습니다. 또한, 연령 및 성별에 따른 내러티브 차이를 분석함으로써, 특정 집단 내에서 특이한 주제 분포를 보이는 증언을 식별하는 새로운 방법도 소개되었습니다.



### Astro-NER -- Astronomy Named Entity Recognition: Is GPT a Good Domain  Expert Annotator? (https://arxiv.org/abs/2405.02602)
Comments: 9 pages

- **What's New**: 이번 연구는 학문적 영역에 특화된 명명된 엔티티 인식(NER) 모델의 개발 과정에서 가장 큰 도전 중 하나인 적합한 레이블이 지정된 데이터 부족 문제를 해결하고자 합니다. 주요 혁신으로는 천문학 문헌에서 과학적 엔티티를 주석하는 비전문가를 지원하기 위해 파인튠된 대규모 언어 모델(LLM: Large Language Model)의 예측을 사용하는 접근방식을 실험하였습니다. 또한, 도메인 전문가에 의해 검증된 천문학 전문 과학 엔티티 주석 체계를 도입하였으며, 이를 통해 생성된 데이터셋은 공개적으로 제공됩니다.

- **Technical Details**: 연구에서는 GPT-3.5 모델을 파인튠하여 천문학 문헌의 과학적 엔티티 주석 작업에 사용했습니다. 이 방법은 비전문가도 복잡한 천문학 용어와 개념을 정확하게 이해하고 활용할 수 있는 능력을 향상시키는데 중점을 뒀습니다. 추가로, 다양한 NER 작업에 대해서 기본 LLM과 파인튠된 LLM의 성능을 비교 분석하였습니다.

- **Performance Highlights**: 도메인 전문가와 LLM 보조를 받은 비전문가들 간의 합의는 중등도로 나타났으며, 도메인 전문가와 LLM 모델의 예측 사이에서는 공정한 합의가 있었다고 평가되었습니다. 또한, 파인튠된 LLM은 기본 LLM에 비해 천문학 NER 작업에서 더 나은 성능을 보였습니다. 공개된 데이터셋은 5,000개의 천문학 논문 제목을 포함하고 있으며, 이는 학계 및 연구 커뮤니티에 유용한 자원으로 활용될 수 있습니다.



### Mixat: A Data Set of Bilingual Emirati-English Speech (https://arxiv.org/abs/2405.02578)
Comments: SIGUL 2024

- **What's New**: 이 논문에서 소개하는 Mixat는 아랍에미리트(UAE) 발화와 영어가 혼합된 언어 데이터셋입니다. 이 데이터셋은 현재의 음성 인식 자원이 에미리트 발화, 특히 에미리트의 쌍방 언어 사용자들의 언어 전환을 제대로 다루지 못하는 문제를 해결하기 위해 개발되었습니다. 데이터셋은 두 개의 공개 팟캐스트에서 파생된 15시간 분량의 연설로 구성되어 있으며, 이 중 한 팟캐스트는 호스트와 게스트 사이의 대화 형식으로 되어 있습니다.

- **Technical Details**: 이 연구는 Emirati-English code-mixing의 다양한 발화 예시를 포함하고 있습니다. 데이터 수집 및 주석(annotation) 작업 과정도 설명되며, 만들어진 데이터셋의 특성과 통계에 대한 정보도 기술하고 있습니다. 데이터셋은 많이 활용되지 않는 방언 지역구(specified dialectal region)인 아랍어 특성에 초점을 맞추고 있습니다.

- **Performance Highlights**: 사전 훈련된(pre-trained) 아랍어 및 다중언어(multi-lingual) 자동 음성 인식(Automatic Speech Recognition, ASR) 시스템들의 성능을 평가하였으며, 이러한 모델들이 갖는 한계와 code-switching 인식의 추가적인 도전 과제들을 입증했습니다. 이 데이터셋은 연구 용도로 공개적으로 접근 가능하며, 이는 저자원(low-resource) 언어와 code-switching을 다루는 더 나은 ASR 시스템 개발에 기여할 것입니다.



### A Combination of BERT and Transformer for Vietnamese Spelling Correction (https://arxiv.org/abs/2405.02573)
Comments: 13 pages

- **What's New**: 이 연구는 베트남어 맞춤법 교정 문제에 처음으로 변환기(Transformer) 아키텍처와 BERT(Bidirectional Encoder Representations from Transformers)를 적용하여 성과를 거두었습니다. 이 모델은 기존 방법들을 넘어서고 구글 문서 도구의 맞춤법 검사 기능보다도 우수한 성능을 보여주었으며, 86.24 BLEU 점수를 달성함으로써 새로운 기준을 설정하였습니다.

- **Technical Details**: 이 연구에서는 변환기 아키텍처와 BERT를 조합하여 베트남어 맞춤법 교정 작업을 처리합니다. 변환기 아키텍처(Transformer architecture)는 인코더-디코더(encoder-decoder) 모델에 있어 최첨단 기술로 인식되며, BERT는 강력한 사전 훈련 언어 모델로 다양한 NLP 작업에서 높은 성공을 거두었습니다. 본 연구는 베트남어의 특성을 고려하여 맞춤형 데이터셋을 구축하고 이를 통해 모델을 훈련시켰습니다.

- **Performance Highlights**: 제안된 모델은 베트남어 맞춤법 교정 작업에서 86.24 BLEU 점수를 달성하여 기존의 접근법들을 능가하는 성능을 보여주었습니다. 국제적으로 인정받는 BLEU 점수 측정 방법을 사용함으로써 모델의 정확성과 효과를 객관적으로 검증하였습니다. 또한, 이 모델은 실제 서비스로 통합될 수 있는 가능성을 보여줌으로써 사용자 경험을 향상시킬 수 있는 실질적인 가치를 제공합니다.



### A Literature Review and Framework for Human Evaluation of Generative  Large Language Models in Healthcar (https://arxiv.org/abs/2405.02559)
- **What's New**: 이 연구는 헬스케어 분야에서 Large Language Models (LLMs)의 인간 평가 방법론에 대한 문헌을 검토하고, 일관되고 표준화된 접근 방식의 필요성을 강조합니다. 특히, 새로운 평가 프레임워크인 QUEST (Quality of Information, Understanding and Reasoning, Expression Style and Persona, Safety and Harm, Trust and Confidence)를 제안하여 LLMs의 신뢰성, 일반화 가능성 및 적용 가능성을 향상시키고자 합니다.

- **Technical Details**: 이 연구는 PRISMA (Preferred Reporting Items for Systematic Reviews and Meta-Analyses) 지침을 따라 2018년 1월부터 2024년 2월까지의 문헌을 검토하였습니다. 연구는 다양한 의료 전문 분야에서 LLM의 인간 평가를 분석하고, 평가 차원, 샘플 유형 및 크기, 평가자의 선발 및 모집, 평가 프로세스 및 결과의 통계 분석 등의 요소를 다룹니다.

- **Performance Highlights**: QUEST 평가 프레임워크는 정보의 질, 이해 및 추론, 표현 스타일 및 페르소나, 안전성 및 해로움, 신뢰성 및 확신의 다섯 가지 주요 평가 차원을 명확히 규정하고 상세한 지침을 제공하여, LLMs의 인간 평가 과정의 신뢰성과 객관성을 향상시키는 데 중점을 둡니다.



### Mothman at SemEval-2024 Task 9: An Iterative System for Chain-of-Thought  Prompt Optimization (https://arxiv.org/abs/2405.02517)
Comments: 13 pages, 2 figures, to be published in Proceedings of the 18th International Workshop on Semantic Evaluation (SemEval-2024)

- **What's New**: 이 연구에서는 대규모 언어 모델(Large Language Models, LLMs)이 횡단 사고(lateral thinking) 작업에서 창의적 해결책을 생성하는 능력에 대한 평가를 진행합니다. BrainTeaser 공유 작업을 통해 모델이 창의적 문제 해결에 어려움을 겪는 것을 파악하고, 사고의 흐름(chain-of-thought, CoT)을 최적화하는 새로운 방법을 제안합니다.

- **Technical Details**: 이 시스템은 CoT 프롬프트 엔지니어링을 반복적으로 최적화하여 인간 평가를 사용하여 입력 데이터와 모델 출력을 체계적으로 평가합니다. 이 연구는 GPT-4 모델을 사용하여 문장 퍼즐 부문을 해결하며, 문제 유형을 식별하고 프롬프트 엔지니어링의 다음 반복을 안내합니다. 논문에서는 모델이 모든 답변 선택을 추론하고 정답 및 오답에 대한 설명을 제공하도록 요구하는 프롬프트 엔지니어링 방법을 개발합니다.

- **Performance Highlights**: 이 접근 방식은 적대적 데이터셋(adversarial datasets)에서 성능이 크게 향상되었으며 모델이 이러한 CoT 프롬프트를 사용할 때 암기에 덜 의존함을 나타냅니다. CoT 프롬프트를 사용한 모델은 문제와 관련이 있지만 논리적으로 틀린 답변을 거절할 가능성이 더 높습니다. 이 과정을 통해 연구팀은 데이터 세트에서 복수의 논리적 선택이 가능하거나 제시된 전제로는 답변할 수 없는 여러 질문을 식별할 수 있었습니다.



### Beyond Helpfulness and Harmlessness: Eliciting Diverse Behaviors from  Large Language Models with Persona In-Context Learning (https://arxiv.org/abs/2405.02501)
Comments: Paper accepted at ICML 2024

- **What's New**: 이 연구에서는 대규모 언어 모델(LLMs)이 다양한 인격 특성을 포함하여 학습되는 것에 착안하여, 원하는 인격 특성을 유도하고 그 행동 선호도를 조사하는 목표를 소개합니다. 이를 위해 'Persona In-Context Learning' (PICLe)이라는 새로운 인격 유도 프레임워크를 제시하고 있습니다. PICLe는 베이지안 추론에 기반한 것으로, 특정 목표 인격을 유도하기 위해 최적으로 모델을 안내하는 새로운 ICL(In-Context Learning) 예제 선택 기준을 도입합니다.

- **Technical Details**: PICLe 프레임워크는 가능도 비율(likelihood ratio)에 기반한 ICL 예제 선택 방법을 새롭게 제안하여, 대상 인격을 효과적으로 유도할 수 있도록 설계되었습니다. 이를 통해 모델이 특정한 인격 특성을 반영하여 반응하도록 학습하는 과정이 최적화됩니다. 본 연구에서는 세 가지 현대적 LLM들을 사용하여 기본적인 방법들과 비교 평가를 진행하여 PICLe의 효과를 입증합니다.

- **Performance Highlights**: PICLe은 기존의 기본 모델들과 비교하여 뛰어난 결과를 보여주었습니다. 목표로 하는 인격 특성에 맞게 모델의 행동을 유도하는 데 있어서 더 정확하고 일관된 반응을 이끌어낼 수 있었습니다. 이러한 결과는 다양한 LLM 상황에서 PICLe의 적용 가능성과 유용성을 시사합니다.



### Semantic Scaling: Bayesian Ideal Point Estimates with Large Language  Models (https://arxiv.org/abs/2405.02472)
- **What's New**: 이번 논문에서는 'Semantic Scaling'이라는 새로운 방식을 도입하여 텍스트로부터 이상적인 점수(ideal points) 산출 방법을 제안합니다. 대규모 언어 모델(large language models)을 활용하여 문서의 정치적 입장을 분류하고, 이를 통해 조사 데이터처럼 사용합니다. 더욱더, 이 방법은 기존 텍스트 기반 척도 방식보다 상당한 개선을 이루며, 연구자들이 측정하고자 하는 이념적 차원을 명확하게 정의할 수 있게 해줍니다.

- **Technical Details**: Semantic Scaling은 대규모 언어 모델을 사용하여 문서를 의미론적(semantics)으로 분류하고, 관찰된 대상체에서 조사와 같은 데이터를 추출하여 Bayesian Markov Chain Monte Carlo 기술을 사용하여 이상적인 위치를 추정합니다. 이 방법은 정책 기반(policy-based) 또는 감정적(affective) 이데올로기를 척도하는 데 사용될 수 있으며, 다양한 문서 유형과 길이에 대한 견고한 추정치를 제공합니다.

- **Performance Highlights**: Semantic Scaling은 이전에 널리 사용된 Tweetscores와 비교하여 더 나은 성능을 보였습니다. 사용자들이 인간 판단에 따라 일치하는 경우, Semantic Scaling이 Tweetscores보다 우수한 결과를 보였습니다. 또한, Semantic Scaling은 미국 의회의 구성원들의 정책 및 감정적 이데올로기 위치를 측정함으로써 DW-NOMINATE 결과와 일치하는 점수를 생성했으며, 법제도의 정책적 또는 감정적 차원을 명확히 구분할 수 있는 새로운 방법을 제시했습니다.



### What is Sentiment Meant to Mean to Language Models? (https://arxiv.org/abs/2405.02454)
- **What's New**: 이 논문은 감성 분석(sentiment analysis)의 다양한 정의와 그 측정의 혼재성(confounded measurement)을 탐구합니다. 연구자가 감성을 정의하는 방식이 일관되지 않다는 점에 주목하며, 대형 언어 모델들이(Large Language Models, LLMs) '감성'을 어떻게 이해하고 분류하는지를 실증적으로 분석합니다. 특히, 각기 다른 프롬프트(prompt)를 사용하여 감성, 정서적 가치(emotional valence), 그리고 의견(stance)을 분류하고, 이들 간의 상관관계를 평가합니다.

- **Technical Details**: 이 연구에서는 GPT-4 Turbo, Claude-3 Opus, 그리고 Llama-3 8B 등의 최신 LLMs를 사용하여 문서를 세 가지로 각각 분류합니다: 감성, 정서적 가치, 의견. 데이터 세트로는 정치인에 대한 지지, 반대, 또는 중립적 의견을 표현하는 트윗 2,390개와 2017 SemEval 도전의 감성 라벨이 명시된 2,000개의 트윗을 사용했습니다. 성능 측정 지표로는 Matthew’s Correlation Coefficient (MCC)를 사용하여 각 분류 프롬프트와 실제 라벨 간의 상관관계를 평가합니다.

- **Performance Highlights**: 분석 결과, 의견 분류를 명시적으로 요청한 프롬프트가 감성이나 정서적 가치를 요청한 프롬프트보다 월등히 높은 성능을 보였습니다. 특히 GPT-4 Turbo와 Claude Opus는 의견(prompt for stance classification)에 대한 분류에서 가장 높은 MCC 값을 기록, 감성과 정서적 가치에 대한 분류 결과와 비교해 개선된 정확성을 나타냈습니다. Llama-3는 다른 두 모델에 비해 상대적으로 낮은 성능을 보였지만, 모든 모델에서 감성이 정서적 가치와 의견의 혼재된 측정임을 보여주는 중간 수준의 성능을 보였습니다.



### What does the Knowledge Neuron Thesis Have to do with Knowledge? (https://arxiv.org/abs/2405.02421)
Comments: ICLR 2024 (Spotlight)

- **What's New**: 본 연구에서는 큰 언어 모델들이 훈련 데이터에서 사실을 회상하는 기작을 설명하는 'Knowledge Neuron (KN)' 가설을 재평가합니다. KN 가설에 따르면, MLP (Multi-Layer Perceptron) 가중치가 Key-Value 메모리처럼 작동하여, 사실 정보를 저장하고 검색할 수 있습니다. 그러나 이 연구는 KN 가설이 언어 모델의 사실적 표현 과정을 충분히 설명하지 못한다는 결론을 내립니다.

- **Technical Details**: 본 논문은 언어 모델의 MLP와 Attention 메커니즘을 중점적으로 조사함으로써, 사실 정보가 어떻게 처리되고 생성되는지에 대한 이해를 넓히고자 합니다. 가설 검증을 위해, 통계 메트릭(metrics)과 새로운 평가 기준을 도입하여, KN 가설에 기반한 편집 방법들의 실효성을 분석합니다.

- **Performance Highlights**: 이전에 제안된 모델 편집 방법들이 간단한 문장 완성 과제에서는 어느 정도 성공을 거두었으나, 새롭게 제안된 '대칭성(symmetry)'과 '동의어 사용(synonymy)' 평가 기준에서는 그 효과가 현저히 떨어지는 것으로 나타났습니다. 이는 KN 가설이 제기한 '지식'의 저장과 표현 방식에 대한 재고가 필요함을 시사합니다.



### The Call for Socially Aware Language Technologies (https://arxiv.org/abs/2405.02411)
- **What's New**: 언어 기술은 특히 대규모 언어 모델(LLMs)의 도입으로 큰 진전을 이루었습니다. 이러한 진전은 기계 번역이나 감정 분석과 같은 전통적인 작업에서 인간 수준에 가까운 성능을 발휘하게 했지만, 동시에 편향(bias), 평가 문제, 위험성 등 전통적인 모델에서 고전해 온 다양한 문제를 심화시킬 수 있습니다. 우리는 이러한 문제들이 모두 NLP가 운영되는 사회적 환경의 요소, 맥락, 결과를 인식하지 못하는 데에 공통적인 근본이 있다고 주장합니다. 이 문제들을 '사회적 인식(social awareness)'으로 정의하며, 이는 언어 애플리케이션이 모든 상황과 모든 사용자에게 효과적으로 작동하도록 하는 데 필요합니다.

- **Technical Details**: 사회적 인식(social awareness)은 사회적 요인, 맥락, 그리고 언어를 통해 전달되는 사회적 역학을 인식하는 능력을 가리킵니다. 현재 NLP 모델은 주로 문법과 어휘에 중점을 두어 언어의 계산 문제로 다루지만, 사회적 상호작용과 문화적 맥락의 복잡성을 포착하는 데에는 크게 진전이 없습니다. 따라서 사회적 관점에서 언어 기술을 연구 및 개발하여 NLP 시스템이 사람들이 언어로 표현하는 사회적 맥락, 관점, 감정을 이해할 수 있도록 해야 합니다. 잘 개발된 사회적 인식 기능은 시스템이 사회적 신호, 문화적 뉘앙스를 더 잘 인식하고 반응할 수 있게 하여 사용자의 신뢰도를 높일 수 있습니다.

- **Performance Highlights**: 사회적 인식을 통합하는 것은 단순히 성능 측정 기준(metric)을 넘어서 사람들에게 미치는 영향에 대해서도 고려하게 할 수 있습니다. 예를 들어, 언어의 기능과 사회적 설정에서의 그 기능 간의 관계를 연구하는 체계적 기능 언어학(SFL)을 통해 현재 NLP 접근 방식에서 누락된 요소를 이해하고 이를 시스템에 통합함으로써 정보 내용을 넘어서는 발전을 이룰 수 있습니다. 또한 소셜 사이언스에서 오랫동안 연구된 다양한 이론들을 통합하여 인간 행동과 상호작용을 이해하려는 노력은 NLP의 사회적 인식 개선에 크게 기여할 것입니다.



### Early Transformers: A study on Efficient Training of Transformer Models  through Early-Bird Lottery Tickets (https://arxiv.org/abs/2405.02353)
- **What's New**: 이 논문은 트랜스포머(Transformer) 모델의 훈련 효율성을 최적화하기 위해 '얼리 버드 티켓'(early-bird ticket) 가설을 적용합니다. ViT, Swin-T, GPT-2, RoBERTa 등 다양한 아키텍처에서 초기 훈련단계에서 고성능의 서브네트워크를 식별할 수 있음을 보여줍니다. 이를 통해 리소스 소모를 크게 줄이면서도 성능을 유지하거나 향상시킬 수 있는 새로운 방법론을 제안합니다.

- **Technical Details**: 본 연구는 '반복적 가지치기'(iterative pruning), '마스크 거리 계산'(masked distance calculation), 그리고 선택적 재학습(selective retraining)을 결합한 방법론을 사용합니다. 초반 에폭에서 '얼리 버드 티켓'을 식별하고, 가지치기된 모델을 재학습하여 원래 모델과 비교할 수 있는 정확도를 달성하게 합니다.

- **Performance Highlights**: 얼리 버드 티켓을 사용한 ViT와 Swin-T 모델은 원본 모델과 유사하거나 우수한 정확도를 보여줬습니다. 예를 들어, ViT의 경우 가지치기 비율(pruning ratio) 0.1에서 기본 성능을 회복하여 84.3%의 정확도를 달성했으며, Swin-T는 가지치기 비율 0.1에서 기본 성능을 완전히 회복하여 89.54%의 정확도를 보였습니다. 또한, GPT-2에서는 미세조정 단계에서 매우 빠르게 얼리 버드 티켓이 나타나며, 가지치기를 통해 원래의 성능을 초과하는 결과를 보였습니다.



### NL2FOL: Translating Natural Language to First-Order Logic for Logical  Fallacy Detection (https://arxiv.org/abs/2405.02318)
- **What's New**: 이 논문에서는 논리적 오류를 자동으로 감지하는 새로운 방법을 제안합니다. 자연어를 일차 논리(First-order Logic, FOL)로 단계적으로 변환하여 대형 언어 모델(Large Language Models, LLMs)을 사용하고, 만족도 모듈 이론(Satisfiability Modulo Theory, SMT) 솔버를 이용하여 수식의 타당성을 판단하고 입력을 오류 또는 유효한 진술로 분류합니다.

- **Technical Details**: 본 시스템은 LLM을 사용하여 자연어를 FOL로 변환하고, SMT 솔버를 활용하여 변환된 수식의 타당성을 평가합니다. 추가적으로, LLM을 이용하여 SMT 솔버의 출력을 해석하고, 주어진 문장이 왜 논리적 오류로 간주되는지에 대한 반례를 제공하는 새로운 방법을 제공합니다. 이 접근법은 훈련 데이터나 미세 조정을 필요로 하지 않으며, 해석 가능하고 강건합니다.

- **Performance Highlights**: 논리적 오류와 유효한 문장이 섞인 데이터셋에서 모델을 평가한 결과, 종단 대 종단 LLMs(end-to-end LLMs)와 비교하여 성능이 향상되었으며, Logic 데이터셋에서 71%의 F1-점수를 달성했습니다. 또한, LogicClimate 도전 세트에서 73%의 F1-점수를 달성하여, 상태 최고 기술(state-of-the-art) 모델들을 21% 상회하는 성능을 보였습니다.



### Pose Priors from Language Models (https://arxiv.org/abs/2405.03689)
- **What's New**: 새로운 제로샷 자세 최적화 방법을 제시하며, 이는 사람들의 3D 자세 추정시 정확한 물리적 접촉 제약을 적용합니다. 연구의 핵심 통찰은 언어가 물리적 상호작용을 서술하는 데 종종 사용된다는 것으로, 이를 통해 대규모 사전 훈련된 텍스트 기반 모델(text-based models)을 자세 추정에 대한 사전 정보(priors)로 활용할 수 있습니다.

- **Technical Details**: 이 방법은 큰 멀티모달 모델(large multimodal model, LMM)이 생성하는 자연어 서술자를 추적 가능한 손실로 변환하여 3D 자세 최적화를 제약합니다. 이는 자기 접촉(self-contact)과 사람 대 사람 접촉(person-to-person contact)을 해결하기 위한 통합된 프레임워크를 제공하는 점에서 기존 접근법과 차별화됩니다.

- **Performance Highlights**: 이 방법은 복잡한 최신 기술(state-of-the-art) 접근법과 경쟁하며, 이러한 접근법은 흔히 비싼 인간 주석(contact points annotation)과 특수 모델 훈련을 요구합니다. 그럼에도 불구하고, 제안된 방법은 인간의 밀접한 접촉을 정확하게 포착하며, 사회적 및 물리적 상호작용의 의미(semantics)를 올바르게 재현합니다.



### Language-Image Models with 3D Understanding (https://arxiv.org/abs/2405.03685)
Comments: Project page: this https URL

- **What's New**: 다차원적인 이미지 해석 능력을 가진 새로운 멀티모달 대형 언어 모델(Multi-modal Large Language Model, MLLM) 'Cube-LLM'이 개발되었습니다. 이 모델은 2차원(2D) 및 3차원(3D) 데이터를 활용하여 더 심층적인 이미지 이해와 추론 능력을 보여주며, 특히 3D 공간에서의 이미지 인식 및 추론에 탁월합니다.

- **Technical Details**: Cube-LLM은 새롭게 구축된 대규모 사전 훈련 데이터셋 LV3D 위에서 사전 훈련(pre-trained)을 진행했습니다. LV3D는 다양한 2D 및 3D 인식 데이터셋을 결합하여 다중 턴 질문-응답(multi-turn question-answering)이라는 공통 작업 수행을 가능하게 합니다. Cube-LLM은 3D 구체적인 건축적 설계나 훈련 목표 없이도 순수 데이터 확장을 통해 강력한 3D 인식 능력을 보여줍니다.

- **Performance Highlights**: Cube-LLM은 외부 벤치마크(outdoor benchmarks)에서 기존 기준점(baselines)을 크게 상회하는 성과를 보여주었습니다. 특히 Talk2Car 데이터셋에서 3D 그라운딩 논리 추론(grounded reasoning)에 대해 21.3 AP-BEV 포인트를 상회하고, DriveLM 데이터셋에서는 복잡한 주행 시나리오에 대한 추론에서 17.7 포인트를 기록했습니다. 또한 Cube-LLM은 refCOCO(2D grounding), VQAv2, GQA, SQA, POPE 등의 일반 MLLM 벤치마크와 시각적 질문 응답(visual question answering) 벤치마크에서도 경쟁력 있는 결과를 보여줍니다.



### Large Language Models (LLMs) as Agents for Augmented Democracy (https://arxiv.org/abs/2405.03452)
Comments: 15 pages main manuscript with 3 figures. 12 pages of supplementary material

- **What's New**: 이 연구는 대규모 언어 모델(Large Language Models, LLMs)을 활용하여 2022년 브라질 대통령 선거 동안 수집된 개인 선호도에 대한 데이터를 기반으로 한 '증강 민주주의' 시스템의 가능성을 탐구하였습니다. 연구진은 개별 정치적 선택과 참여자 전체의 집단 선호도를 예측하는 LLM의 정확성을 평가하였습니다.

- **Technical Details**: 연구팀은 훈련-테스트 교차 검증(cross-validation) 설정을 사용하여, 개인 차원에서의 예측 정확도가 69%-76% 범위인 것을 발견했습니다. 이는 특히 자유주의적 성향과 대학 교육을 받은 참여자의 선호도를 더 잘 예측합니다. 집단 차원에서는 보르다 점수(Borda score)의 변형을 사용하여 선호도를 집계하고, 확률적 샘플과 LLM을 사용하여 증강된 데이터로부터 얻은 정책 제안의 순위를 비교분석했습니다.

- **Performance Highlights**: 증강된 데이터는 전체 참여자집단의 선호도를 예측하는데 있어, 확률적 샘플만을 사용했을 때보다 더 나은 결과를 보였습니다. 특히 전체 인구의 30%에서 40% 미만을 대표하는 확률적 샘플의 경우 그 차이가 두드러졌습니다. 이 결과는 LLM이 증강 민주주의 시스템 구축에 유용할 가능성을 시사합니다.



### Advancing Multimodal Medical Capabilities of Gemin (https://arxiv.org/abs/2405.03162)
- **What's New**: Gemini 모델을 기반으로 한 새로운 의료용 AI 모델들, Med-Gemini 시리즈가 출시되었습니다. Med-Gemini는 Gemini의 핵심 기능을 상속받고, 특히 의료 분야에서의 활용을 최적화하기 위해 2D, 3D 방사선학, 조직병리학, 안과학, 피부과학 및 유전체 데이터를 사용하여 미세 조정(fine-tuned)되었습니다. 이러한 모델들은 CXR (chest X-ray), CT (computed tomography), VQA (visual question answering), 그리고 유전적 위험 예측에서 최고 수준의 성능을 보여줍니다.

- **Technical Details**: Med-Gemini는 이미지 분류, 보고서 생성, 질의응답(VQA)과 같은 다양한 의료 태스크를 수행하는 데 사용되며, 이미지 기반 태스크 뿐만 아니라 유전체 기반 질병 위험 예측 같은 비이미지 태스크에서도 우수한 성능을 보여줍니다. 특히 Med-Gemini-2D는 이전의 최고 기록을 뛰어넘는 CXR 보고서 생성에서 탁월한 성과를 보였으며, Med-Gemini-3D는 3D CT 볼륨에 대한 보고서 생성에서도 AI 보고서가 임상적으로 수용 가능하다는 평가를 받았습니다.

- **Performance Highlights**: Med-Gemini-2D는 두 개의 별도 데이터셋에서 이전 최고 결과를 1% 및 12% 포인트 상회하며 새로운 표준을 설정했습니다. 또한, CXR VQA 태스크에서 기존 SoTA (State of the Art) 모델을 능가하는 성능을 보였으며, 이외에도 피부과, 안과, 조직병리 이미지 분류에서 베이스라인을 상회하는 결과를 18/20 태스크에서 보여주었습니다. Med-Gemini-Polygenic은 표준 선형 다형성(polygenic) 위험 점수 기반 접근 방식보다 뛰어난 질병 위험 예측 성능을 제공합니다.



### Quantifying the Capabilities of LLMs across Scale and Precision (https://arxiv.org/abs/2405.03146)
- **What's New**: 이 연구는 대규모 언어 모델(LLMs)의 성능에 미치는 스케일과 정량화(quantization)의 영향을 조사합니다. 특히, 자원이 제한된 환경에서 모델의 사용, 배치, 디버깅이 어려운 점을 해결하기 위한 대안으로 모델의 크기를 줄이거나 메모리 요구 사항을 낮추는 방법이 사용되었습니다.

- **Technical Details**: 연구자들은 7B에서 70B까지 다양한 파라미터를 가진 두 주요 오픈 소스 지시 모델(instruct models)을 사용하여 실험을 수행했습니다. 연구팀은 자연어 이해, 추론, 오보 탐지(misinformation detection), 환각(hallucination) 등 다양한 작업에서 광범위한 영향을 평가하여 제로샷(zero-shot) 실험을 진행했습니다.

- **Performance Highlights**: 더 큰 모델이 일반적으로 작은 모델보다 우수한 성능을 보여주며, 이는 스케일이 성능 향상에 중요한 요소임을 시사합니다. 또한, 큰 모델은 정밀도 감소에 대한 높은 회복력을 보여주어, 비슷한 메모리 요구 조건에서 작은 모델을 고정밀로 사용하는 것보다 더 나은 해결책을 제공할 수 있음을 발견했습니다. 특히 4비트 정량화(4-bit quantization)에서도 높은 정확도를 유지할 수 있었습니다.



### To Each (Textual Sequence) Its Own: Improving Memorized-Data Unlearning  in Large Language Models (https://arxiv.org/abs/2405.03097)
Comments: Published as a conference paper at ICML 2024

- **What's New**: 이 연구에서는 대규모 언어 모델(LLMs)이 훈련 데이터를 기억(memorize)하고 그 데이터를 정확히 복제하여 생성하는 문제에 초점을 맞추고 있습니다. 이 문제는 개인 정보 보호 및 저작권과 관련된 문제를 야기합니다. 이를 해결하기 위해, 저자들은 기억된 데이터를 언러닝(unlearning) 할 때 각 텍스트 시퀀스를 그 기억 정도에 따라 다르게 처리해야 한다는 새로운 관점을 제시합니다.

- **Technical Details**: 연구팀은 언러닝의 질을 측정하는 새로운 지표(metric)를 개발하고, 현재 최고의 기술(state-of-the-art, SOTA) 알고리즘들이 이러한 관점을 무시할 때 발생하는 프라이버시 문제를 보여주는 적대적 공격(adversarial attack)을 제시했습니다. 또한, 그라디언트 상승(Gradient Ascent)과 작업 산술(Task Arithmetic)을 기반으로 한 두 가지 새로운 언러닝 방법을 도입했습니다.

- **Performance Highlights**: 연구진은 다양한 자연어 처리(NLP) 작업에 걸쳐 신규 방법들의 성능 평가를 수행했습니다. 다양한 모델 용량과 잊혀진 데이터 집합 크기에 따른 최적의 해결책을 식별하고, 새로운 접근 방식의 이점을 정량화하였습니다.



### On the performativity of SDG classifications in large bibliometric  databases (https://arxiv.org/abs/2405.03007)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)을 사용하여 광범위한 데이터베이스에 의해 주입된 '데이터 편향'을 조사하고 있습니다. 특히, 유엔의 지속가능발전목표(SDGs)에 맞춘 분류 체계에 따른 편향을 중점으로 논의합니다. 세 가지 주요 데이터베이스(Web of Science, Scopus, OpenAlex)에서 수집한 데이터를 기반으로 하여, 각각의 SDG 관련 분류에 따라 다르게 조정된 LLM을 사용하여 언어적 특성과 지향점의 차이를 고찰합니다.

- **Technical Details**: 본 연구에서는 DistilGPT-2 모델을 사용하였으며, 이는 GPT-2의 경량화 버전으로 빠르고 효율적인 메모리 사용이 가능합니다. 각 데이터 베이스별로 SDG 분류에 따른 추상적 요약을 통해 미세 조정된 모델을 통해 데이터 내재적 관점을 반영하여 자연어 반응을 생성하도록 합니다. 데이터 수집은 2015년부터 2023년까지의 출판물 15,471,336건을 기반으로 이루어졌으며, 이를 통해 다양한 데이터 베이스 간의 SDG 분류에 따른 차이를 비교 분석합니다.

- **Performance Highlights**: 연구 결과에서 다양한 SDG 분류가 LLM의 성능에 높은 민감성을 보이는 것으로 나타났습니다. 각각의 SDG에 대해 미세 조정된 모델들은 그 분류에 따라 상이한 언어적 특성과 시각을 가지며, 이는 학술적 내용의 해석과 분류에 있어 일관성 부재를 시사합니다. 따라서, LLM을 연구실습에 사용함에 있어서 이러한 분류들의 주관성과 임의적 요소들을 신중히 고려할 필요가 있습니다.



### Parameter-Efficient Fine-Tuning with Discrete Fourier Transform (https://arxiv.org/abs/2405.03003)
Comments: Accepted by ICML 2024

- **What's New**: 새로운 FourierFT 방법은 기존의 Low-rank adaptation (LoRA) 모델을 개선하는 방법으로, Fourier 변환의 표현력을 활용하여 훈련 가능한 파라미터를 더욱 압축하는 기법입니다. 이 방식은 $	ext{Δ}W$를 공간 도메인의 행렬로 취급하고, 그 스펙트럼 계수의 작은 부분만을 학습합니다.

- **Technical Details**: FourierFT는 역 이산 푸리에 변환(Inverse Discrete Fourier Transform, IDFT)을 사용하여 $	ext{Δ}W$를 복원합니다. 이 접근법은 푸리에 변환의 강력한 표현 능력을 통해 특히 대형 기반 모델이나 광범위한 맞춤형 적응을 다룰 때 저장 공간 문제를 해결하는 데 도움을 줍니다.

- **Performance Highlights**: FourierFT 방법은 LoRA보다 적은 파라미터로 비슷하거나 더 나은 성능을 보여줍니다. 예를 들어, LLaMA2-7B 모델에 대한 지시 튜닝(instruction tuning)을 수행할 때 FourierFT는 0.064M의 훈련 가능한 파라미터만을 사용하면서 LoRA의 33.5M에 비해 우수한 성능을 보였습니다.



### Overconfidence is Key: Verbalized Uncertainty Evaluation in Large  Language and Vision-Language Models (https://arxiv.org/abs/2405.02917)
Comments: 8 pages, with appendix. To appear in TrustNLP workshop @ NAACL 2024

- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)과 비전 언어 모델(VLMs)의 불확실성 평가 능력을 평가합니다. 특히, 새로운 데이터셋인 'Japanese Uncertain Scenes (JUS)'와 새로운 평가 지표인 'Net Calibration Error (NCE)'가 도입되었습니다. 이를 통해 VLMs이 어려운 이미지 질문과 객체 계수를 어떻게 처리하는지 평가하고 이들 모델의 자신감 수준과 실제 정확도와의 일치도를 검토합니다.

- **Technical Details**: 연구는 네 가지 LLMs(GPT-3.5, GPT-4, LLaMA-2-70b, PaLM 2)와 두 가지 VLMs(GPT-4V, Gemini Pro Vision)를 대상으로 하는데, 이들 모델의 자신감 표현 능력을 비교했습니다. 평가는 세 가지 NLP 작업(감정 분석(SA), 수학 단어 문제(MP), 명명 개체 인식(NER))과 하나의 이미지 인식(IR) 작업을 통해 이루어졌습니다. 구체적으로, STT 및 GSM8K 데이터셋은 LLMs의 NLP 작업을 위해, CoNLL 2003 데이터셋과 JUS는 VLMs의 이미지 인식 능력을 시험합니다.

- **Performance Highlights**: 결과적으로, 모든 모델들은 높은 NCE를 보여주며 대부분의 경우 과도한 자신감을 나타냈습니다. 이는 모델들이 불확실성을 정확히 추정하거나 그 자신감을 적절히 조절하는 데 능숙하지 않음을 나타냅니다. 특히, VLMs는 이미지 인식 작업에서 평균/표준편차 및 95% 신뢰 구간을 생성할 때 적절한 보정이 이루어지지 않는 경우가 많았습니다.



### Language Evolution for Evading Social Media Regulation via LLM-based  Multi-agent Simulation (https://arxiv.org/abs/2405.02858)
Comments: Accepted by IEEE WCCI 2024

- **What's New**: 이 논문에서는 사회적, 기술적 압력 하에서 자연스럽게 발전하는 언어를 연구하기 위해 대규모 언어 모델(Large Language Models, LLMs)을 사용하는 다중 에이전트 시뮬레이션 프레임워크를 제안합니다. 특히, 규제가 심한 소셜 미디어 환경에서 사용자의 언어가 어떻게 진화하는지를 탐구합니다. 이는 표현의 자유를 보장하고 콘텐츠 검열을 최적화하는 데 중요한 의미가 있습니다.

- **Technical Details**: LLMs 기반 에이전트로 구성된 시뮬레이션 프레임워크는 대화 감독을 담당하는 감독 에이전트(supervisory agent)와 언어 전략을 진화시키며 대화에 참여하는 참여자 에이전트(participant agents)로 구성됩니다. 이들은 규제를 회피하기 위해 통신 스타일의 진화를 시뮬레이션합니다. 연구는 이론적 시나리오에서 실제 상황에 이르기까지 다양한 시나리오를 통해 프레임워크의 효과를 평가합니다.

- **Performance Highlights**: LLMs는 제한된 환경에서 미묘한 언어 역학 및 상호 작용을 시뮬레이션하는 데 능력을 보여줍니다. 진화가 진행됨에 따라 규제 회피 및 정보 정확성이 향상되었습니다. 또한 LLM 에이전트는 다양한 시나리오에 대해 다른 전략을 채택하는 것으로 나타났습니다.



### ImageInWords: Unlocking Hyper-Detailed Image Descriptions (https://arxiv.org/abs/2405.02793)
Comments: Webpage (this https URL), GitHub (this https URL), HuggingFace (this https URL)

- **What's New**: 이 연구에서는 시각-언어 모델(Vision-Language Models, VLMs)을 위한 새로운 데이터셋과 주석 프레임워크인 ImageInWords (IIW)를 소개합니다. 이 프레임워크는 인간의 주석자를 중심으로 상세하고 오류가 없는 이미지 설명을 생성하는 과정을 따릅니다. IIW 데이터셋은 최근 다른 데이터셋과 비교하여 +66%, GPT-4V 출력과 비교해 +48%의 성능 향상을 보여주었습니다.

- **Technical Details**: IIW 프레임워크는 객체 감지기를 사용하여 이미지 내 개별 객체를 식별한 후, 각 객체에 대한 세부적인 캡션을 VLMs가 생성하고, 이를 인간 주석자가 수정 및 보완하는 과정을 거쳐 최종 이미지 설명을 생성합니다. 주석 과정은 객체 수준에서 시작하여 이미지 수준으로 확장되며, 이는 시각적 관점, 공간 배치 및 인간-객체 상호 작용과 같은 개념을 포함합니다.

- **Performance Highlights**: IIW로 파인 튜닝(fine-tuned)된 모델은 읽기 가능성, 포괄성, 특이성 지표에서 우수한 성능을 보였습니다. 또한, IIW 모델을 사용하여 생성된 이미지 설명은 원본 이미지와 가장 가까운 이미지를 생성하는 데 성공했으며, 구성 논리(compositionality) 추론 벤치마크에서도 최고의 기준 모델을 최대 6%까지 능가하는 성능을 보였습니다. 이러한 결과는 IIW 데이터셋과 프레임워크가 시각-언어 연구와 응용 분야에 큰 발전 가능성을 제공한다고 할 수 있습니다.



### Get more for less: Principled Data Selection for Warming Up Fine-Tuning  in LLMs (https://arxiv.org/abs/2405.02774)
Comments: Published as a conference paper at ICLR 2024

- **What's New**: 이 연구는 언어 모델이 사전 학습된 분포와 타겟 분포 사이의 간격을 좁힐 수 있는 데이터를 선택하는 새로운 방식인 GOT-D(Gradients of Optimal Transport for Data Selection) 방법을 제안합니다. 이 방법은 다양한 NLU(Natural Language Understanding), NLG(Natural Language Generation), 그리고 제로샷(zero-shot) 태스크에서 효과적으로 모델의 성능을 향상시킬 수 있음을 입증합니다.

- **Technical Details**: GOT-D 방법은 최적 운송(Optimal Transport, OT) 거리의 그래디언트를 사용하여 사전 학습 데이터 분포와 타겟 데이터 분포 사이를 연결하는 데이터 샘플을 선택합니다. 이 과정에서 엔트로피 정규화(entropy regularization)와 모멘텀(momentum) 같은 최적화 기술을 활용하며, GPU 병렬 계산을 통해 수백만 개의 샘플을 몇 분 안에 처리할 수 있습니다.

- **Performance Highlights**: GOT-D는 기존의 데이터 선택 방법들과 비교했을 때 일관되게 뛰어난 성능을 보였으며, 특히 낮은 데이터 선별 예산(예: 50k 샘플)에서 뚜렷한 개선을 보였습니다. 예를 들어, GPT-2의 독성 수준을 10k 샘플로 30% 감소시키고, 8개 도메인 특화 태스크에서 평균 성능을 150k 샘플로 1.13% 향상시켰습니다. 또한, 제로샷 태스크에서는 모델 크기가 2.7B일 때, 40k 샘플만으로 성능을 13.9% 향상시켰습니다.



### Beyond Relevance: Evaluate and Improve Retrievers on Perspective  Awareness (https://arxiv.org/abs/2405.02714)
- **What's New**: 이 연구에서는 정보 검색(IR)에 '관점 인식' 기능을 추가하여, 검색 시스템이 사용자의 질문이나 요구에 포함된 미묘한 관점의 차이를 인식하고, 이에 기반하여 문서를 검색할 수 있는지를 조사합니다. 특히, 검색 시스템이 단순히 관련 문서를 찾는 것을 넘어서, 지지하는 문서와 반대하는 문서를 구분할 수 있는지를 분석합니다.

- **Technical Details**: 새롭게 제안된 검색 기준인 PIR(Perspective-aware Information Retrieval) 벤치마크를 통해 다양한 도메인에서 수집된 7,494개의 다양한 질문과 10,286개의 문서 후보를 포함하여 관점 인식 정보 검색을 시험합니다. 기존의 검색 시스템들이 특정 관점에 편향되어있는 문제를 지적하며, 관점 인식을 강화하기 위해 PAP(Perspective-aware Projection) 방법을 도입하여, 쿼리와 문서 후보의 임베딩을 관점을 통해 벡터 공간 계산으로 투영합니다.

- **Performance Highlights**: PAP 방법을 사용했을 때, 여러 설정에서 기존 기준 모델들을 뛰어넘는 성능을 보여주며, GPT-3.5-Turbo와 같은 최신 언어 모델을 이용한 실험에서도 AmbigQA에서 4.2% 높은 정확도를 달성하고, 에세이 작성에서는 지정된 관점과 29.9% 더 높은 상관 관계를 보였습니다. 이는 관점 인식이 하류 작업(downstream tasks)의 성능 개선에 중요한 역할을 할 수 있음을 시사합니다.



### TREC iKAT 2023: A Test Collection for Evaluating Conversational and  Interactive Knowledge Assistants (https://arxiv.org/abs/2405.02637)
Comments: To appear in SIGIR 2024. arXiv admin note: substantial text overlap with arXiv:2401.01330

- **What's New**: TREC iKAT 트랙은 사용자 대화 검색 에이전트(CSA)의 평가 및 개선을 위해 만들어졌습니다. 이 트랙은 대화형 검색 작업을 평가하는 데 필요한 리소스를 제공하여 검색 에이전트가 사용자의 개인적인 맥락을 식별하고 이를 통해 맞춤형 대화를 생성할 수 있도록 합니다. 특히, TREC iKAT는 대화가 진행됨에 따라 사용자의 개별 요구 사항에 맞춰 지속적으로 적응하며, 다양한 개인별 페르소나를 반영한 검색 작업을 중점적으로 다룹니다.

- **Technical Details**: 본 연구에서는 TREC iKAT 테스트 컬렉션을 확장하여 36개의 개인화된 대화와 20개의 다양한 주제를 포함하며, 각 주제는 Personal Text Knowledge Base(PTKB)와 연결되어 있습니다. 이 컬렉션은 약 26,000개의 패시지를 포함하고 각 턴은 relevance, completeness, groundedness, naturalness의 네 가지 주요 차원에 대한 평가를 제공합니다. 또한, 본 연구는 이러한 대화형 검색 에이전트가 개인적 맥락을 효과적으로 활용하여 관련성 있는 대화를 생성하는 능력을 평가합니다.

- **Performance Highlights**: TREC iKAT 제출 작업의 결과에서는 대화형 검색 에이전트가 사용자의 질문과 이전 대화의 맥락을 이해하고, 이를 바탕으로 사용자의 요구에 맞는 정보를 제공하는 능력이 강조되었습니다. 또한, CSA가 지속적으로 개인화된 정보를 추출하고 대화를 맞춤화하는 능력이 시험되었으며, 개인 페르소나(Persona)에 근거한 대화 반응의 생성 가능성에 여러 관점에서 분석이 이루어졌습니다.



### Random Masking Finds Winning Tickets for Parameter Efficient Fine-tuning (https://arxiv.org/abs/2405.02596)
Comments: ICML 2024

- **What's New**: 이 연구는 AI의 난제인 대규모 언어 모델(LLM)의 튜닝을 위한 새로운 접근 방법을 제안합니다. Random Masking이라는 기술을 사용하여 모델의 매개변수 중 일부만을 무작위로 훈련함으로써, 기존의 파라미터 효율적 미세조정(PEFT) 방법을 더욱 단순화하고 훈련 가능한 매개변수의 수를 줄입니다. 이 방법은 높은 학습률을 사용하여 표준 PEFT기법과 유사한 성능을 달성하는 것으로 나타났습니다.

- **Technical Details**: Random Masking 방법은 LLM의 특정 매개변수에 대해 임의의 이진 마스크를 적용하고, 마스크되지 않은 매개변수만을 미세조정하는 방식입니다. 이 연구는 마스킹이 손실 헤시안 스펙트럼에서 더 평평한 손실 풍경을 유도하고, 이로 인해 더 넓은 학습률이 필요하다는 것을 실험적으로 및 이론적으로 입증합니다. 연구 결과, Random Masking은 기존 LoRA 방법 보다 100배 적은 매개변수로도 비슷한 성능을 낼 수 있음을 보여줍니다.

- **Performance Highlights**: Random Masking은 SuperGLUE 데이터셋을 통한 실험에서 표준 PEFT 방법들과 동등한 성능을 보였으며, 트레이닝 가능 매개변수의 비율이 LoRA 방법의 약 0.001%에 불과함에도 불구하고 효과적인 성과를 보였습니다. 이는 현재의 PEFT 방법들에서 상당한 매개변수 중복이 있음을 시사합니다.



### CALRec: Contrastive Alignment of Generative LLMs For Sequential  Recommendation (https://arxiv.org/abs/2405.02429)
- **What's New**: 새로운 추천 시스템 프레임워크인 CALRec은 대규모 언어 모델 (LLMs)을 중심으로, 정교한 프롬프트 디자인과 두 단계의 파인튜닝 패러다임, 혼합 훈련 목표 및 준라운드 로빈 검색 방식 (quasi-round-robin BM25 retrieval)을 특징으로 합니다. 이는 연속적인 추천(contextual recommendation)에서 텍스트 기반 입력과 출력을 활용하여 진행되며, 기존의 추천 시스템과 대비하여 상당한 성능 향상을 보였습니다.

- **Technical Details**: CALRec은 LLM을 두 단계로 파인튜닝하는 구조를 가지고 있으며, 다중 도메인 데이터를 이용한 첫 단계 파인튜닝 후 목표 도메인에 특화된 추가 파인튜닝을 진행합니다. 이 과정에서 두 가지 대조 손실과 한 가지 언어 모델링 손실을 혼합하여 사용하는 것이 특징입니다. 별도의 텍스트 지표를 사용하여 모델이 데이터 패턴과 텍스트 형식을 더 잘 이해할 수 있도록 설계되었습니다.

- **Performance Highlights**: CALRec 모델은 Recall@1 지표에서 37% 향상, NDCG@10에서 24% 향상을 보였습니다. 이는 표준 높은 기준에서도 뛰어난 성능을 보이며, 도메인 간 대조 정렬이 효과적임을 입증하며, 두 단계 파인튜닝이 성능 향상에 매우 중요하다는 것을 시스템적인 소거 연구를 통해 확인할 수 있었습니다.



### LLM as Dataset Analyst: Subpopulation Structure Discovery with Large  Language Mod (https://arxiv.org/abs/2405.02363)
- **What's New**: 이번 연구에서는 데이터셋 내부의 하위집단(subpopulation) 분포를 체계적으로 탐구하여 이를 나타내고 활용하는 새로운 개념인 하위집단 구조(subpopulation structures)를 도입하였습니다. 이전에는 데이터셋의 하위집단 분포를 체계적으로 탐색한 연구가 없었기 때문에, 이 연구는 중요한 기술적 진보를 나타냅니다.

- **Technical Details**: 연구팀은 하위집단 구조를 해석 가능한 방식으로 특성화하기 위해 Large Language Models (LLMs)를 활용하는 Subpopulation Structure Discovery with Large Language Models (SSD-LLM) 프레임워크를 제안하였습니다. 이 프레임워크는 LLM의 세계적인 지식과 지시에 따른 수행 능력을 활용하여 정보가 풍부한 이미지 캡션을 언어학적으로 분석하고 구조를 요약합니다.

- **Performance Highlights**: 제안된 SSD-LLM 프레임워크는 하위집단 관련 다운스트림 작업(downstream tasks)에 적용될 수 있도록 Task-specific Tuning을 통해 데이터셋의 하위집단 조직화(dataset subpopulation organization), 하위집단 변화(subpopulation shift), 및 슬라이스 발견(slice discovery) 등의 작업을 수행하는 완전한 워크플로우를 제안합니다. 이를 통해 하위집단 구조의 실제적인 활용 가능성을 보여줍니다.



### COPAL: Continual Pruning in Large Language Generative Models (https://arxiv.org/abs/2405.02347)
Comments: Accepted to ICML2024

- **What's New**: 새롭게 소개된 COPAL (COntinual Pruning in Adaptive Language settings)은 대형 언어 모델을 지속적으로 적응시키면서도 고성능을 유지하도록 설계된 알고리듬입니다. 이는 고가의 finetuning이나 retraining을 피하면서, 자원 효율성과 모델의 적응성을 동시에 개선할 수 있는 방법을 제공합니다.

- **Technical Details**: COPAL은 기존 데이터셋에서 중요한 가중치를 발견하고 모델의 복잡성과 크기를 줄이는 방식으로 pruning을 진행합니다. 새로운 데이터셋으로 전환시에는 적은 양의 calibration data를 사용하여 가이드됩니다. 제안된 sensitivity (민감도 분석)는 새로운 데이터셋에 의해 도입된 변동에 대한 모델의 견고성을 측정해, 지속적인 정보 적응을 가능하게 합니다.

- **Performance Highlights**: COPAL은 LLAMA-7B, 30B, 65B 등 다양한 크기의 언어 모델에서 기존 baseline 모델들을 능가하는 성능을 보였습니다. 이는 COPAL이 모델 복잡성과 성능 사이의 균형을 능숙하게 조절할 수 있음을 시사하며, 실제 애플리케이션에서 중요한 요소로 작용합니다.



### Speech Technology Services for Oral History Research (https://arxiv.org/abs/2405.02333)
Comments: 5 pages plus references, 3 figures

- **What's New**: 이 연구에서는 구술 역사(oral history)를 다룹니다. 구술 역사는 역사적 사건에 대한 목격자 및 주석자의 구두 출처에 관한 것입니다. 연구는 구술 기록을 전사(transcription)하고 구조화하기 위한 음성 기술(speech technology)의 중요성을 강조합니다.

- **Technical Details**: 연구의 기술적 세부사항으로는 BAS와 LINDAT에서 개발된 음성 처리 웹서비스와 솔루션, 개인 사용자를 위한 Whisper 등이 포함됩니다. 이러한 도구와 기술은 구술 기록의 전사와 향상을 용이하게 합니다.

- **Performance Highlights**: 기술의 발전에도 불구하고 여전히 남아있는 도전 과제들과 향후 개발 가능성에 대해서도 언급합니다.



### Digital ASIC Design with Ongoing LLMs: Strategies and Prospects (https://arxiv.org/abs/2405.02329)
Comments: 8 pages, 2 figures, 1 table

- **Newsletter**: [{"What's New": '최신 디지털 시스템의 복잡성이 증가함에 따라, 통합 회로(IC) 디자인에 필요한 도구들에 대한 요구가 높아지고 있습니다. 이에 대응하여, Large Language Models (LLMs)는 하드웨어 기술 언어(Hardware Description Language, HDL) 코드의 자동 생성을 가능하게 하여 디지털 IC 디자인의 흐름을 간소화할 수 있는 유망한 발전으로 간주되고 있습니다. 본 논문은 LLMs를 디지털 ASIC 디자인에 활용하기 위한 전략을 제시하고 있으며, 이러한 전략들이 어떻게 HDL 코드 생성의 신뢰성과 정확성을 향상시킬 수 있는지 구체적으로 설명합니다.'}, {'Technical Details': "LLMs는 종종 중요한 구문 오류와 회로 설계의 고급 의미 체계를 정확하게 전달하는 데 어려움을 겪는 것으로 알려져 있습니다. 이러한 문제를 해결하기 위해, 본 연구는 코드 생성의 신뢰도와 정확성을 높이기 위한 방법을 개발하였습니다. 구체적인 적용 예로, 간단한 3단계 펄스 폭 변조(Pulse Width Modulation, PWM) 발생기의 개발을 통해 제공되며, 이는 'Efabless AI-Generated Open-Source Chip Design Challenge'의 일환으로 성공적으로 브루 프레임 검사(Design Rule Check, DRC)를 통과하고 제작되었습니다."}, {'Performance Highlights': '제안된 LLM 기반 방법은 높은 신뢰성과 정확성을 가진 HDL 코드를 생성할 뿐만 아니라, 실제 칩 제작으로 이어질 수 있음을 입증하였습니다. 이는 디지털 ASIC 디자인의 복잡성을 극복하고 효율성을 증진시킬 수 있는 새로운 접근 방식을 제공합니다.'}]



### Evaluating LLMs for Hardware Design and Tes (https://arxiv.org/abs/2405.02326)
- **What's New**: 이 연구에서는 대형 언어 모델(LLMs)이 제공된 사양에서 하드웨어 모듈을 설계하고 테스트할 수 있는 가능성을 탐구합니다. 이는 디지털 설계 파이프라인을 완전 자동화로 나아가기 위한 프레임워크에서 설계와 테스트의 결합을 촉진할 수 있는 중요한 단계입니다.

- **Technical Details**: 연구진은 하드웨어 설명 언어(HDLs)로 코드를 생성할 때, 일반적으로 사용되는 상태 최신 기술 대화형 LLM을 사용하여 Verilog 언어를 사용한 기능 및 검증 목적을 위한 코드 생성 능력을 평가했습니다. 특히, 이 연구는 8가지 대표 벤치마크를 사용하여 LLM의 성능과 한계를 조사했습니다.

- **Performance Highlights**: 실행된 벤치마크는 Skywater 130nm 셔틀을 사용하여 탭 아웃되었고, 기능적인 칩이 생성되었다는 결과를 얻었습니다. 이는 LLM이 하드웨어 설계와 검증에 대해 실제로 유용한 코드를 생산할 수 있다는 가능성을 보여줍니다.



### A geometric framework for interstellar discourse on fundamental physical  structures (https://arxiv.org/abs/2405.02314)
Comments: 15 pages, 2 figures

- **What's New**: 이 논문은 추상적 사고와 고급 종합 능력이 지구의 인류와 의사소통을 받아들일 것으로 예상되는 외계 문명의 가능성을 고려합니다. 이를 위해 알파벳과 숫자 사용에 의존하지 않는 표기법을 제안하여 현재 물리 이론의 기본 기하 구조(벡터 필드(Vector fields), 일차원 필드(one-form fields), 임의 순서의 텐서 필드(tensor fields))를 나타내려고 합니다.

- **Technical Details**: 이 논문에서 제안하는 방식은 전자기학(electromagnetism)과 일반 상대성 이론(general relativity)을 간결하게 설명할 수 있는 방법을 제시하며, 이는 고도의 문명이 우리의 신호에 응답할 도전을 받아들일 수 있는 계기를 제공할 수 있습니다. 물리 이론의 기본 구조를 설명하기 위해 도입된 추상적 기호들은 흑백 비트맵 이미지(black and white bitmap images)로 인코딩되어 라디오 전송을 위해 캐리어 웨이브(carrier wave)에 변조될 수 있는 짧은 비트 시퀀스로 쉽게 변환됩니다.

- **Performance Highlights**: 제안된 표기법은 문화적 또는 언어적 배경이 다른 문명에도 접근성을 제공합니다. 이는 과학의 근본적 개념을 더욱 효율적으로 전달하고, 우주적 차원의 의사소통에서 새로운 가능성을 열어주는 접근법입니다.



### Inserting Faces inside Captions: Image Captioning with Attention Guided  Merging (https://arxiv.org/abs/2405.02305)
- **What's New**: 이 연구에서는 이미지 설명 작업을 위한 새로운 데이터셋인 AstroCaptions을 소개합니다. 이 데이터셋은 전통적인 모델로 식별하기 복잡한 수천 명의 공인 인물들을 포함하고 있습니다. 또한, 설명 가능한 AI 도구와 비전-언어 모델의 그라운딩(grounding) 능력을 활용하여 인식된 사람들의 이름을 캡션에 삽입하는 새로운 후처리(post-processing) 방법을 제안합니다.

- **Technical Details**: 제안된 후처리 방법은 비전-언어 모델의 그라운딩 기능을 활용하여 이미지 내에서 식별된 사람들의 이름을 캡션에 정확하게 삽입합니다. 이 방법은 설명 가능한 AI 도구를 사용하여 캡션 생성 과정에서의 투명성을 향상시킵니다.

- **Performance Highlights**: 이 방법을 사용하여 생성된 캡션은 BLEU, ROUGE, CIDEr, METEOR 점수에서도 향상을 보여줍니다. 특히, 식별된 인물의 93.2%가 이미지 캡션에 삽입되어, 캡션의 품질이 대폭 개선되었으며 환각(hallucinations)의 감소 가능성을 보여줍니다.




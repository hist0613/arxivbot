New uploads on arXiv(cs.CL)

### Towards Compositionality in Concept Learning (https://arxiv.org/abs/2406.18534)
Comments:
          Accepted at ICML 2024. 26 pages, 10 figures

- **What's New**: 현재 많은 개념 기반 설명 방법들은 컨셉을 분해하여 파운데이션 모델(foundation models)의 내부를 들여다보는 방식을 사용합니다. 하지만 기존의 비지도 방식으로 추출된 개념들은 서로 조합되지 않는다는 문제점이 있습니다. 이를 해결하기 위해, 본 연구에서는 두 가지 중요한 속성을 만족하는 Compositional Concept Extraction (CCE) 기법을 제안하였습니다. CCE는 이미지와 텍스트 데이터셋에서 더 조합 가능한 개념 표현을 찾아내고, 네 개의 다운스트림 분류 작업에서 더 나은 정확도를 제공합니다.

- **Technical Details**: CCE는 전체 개념 하위 공간(subspaces)을 한 번에 검색하여 개념의 조합 가능성을 보장합니다. CCE는 개념 속성이 서로 직교하도록 하여, 동일 속성의 개념들은 조합되지 않지만 다른 속성 간의 개념들은 조합될 수 있게 합니다. 이를 통해 기존 방법에서 잘못 추출된 비조합 개념 표현 문제를 해결합니다.

- **Performance Highlights**: CCE는 이미지 및 텍스트 데이터셋에서 기존 방법보다 더 조합 가능한 개념 표현을 찾고, 발견된 개념들이 네 개의 다운스트림 분류 작업에서 더 나은 성능을 보였습니다. 이는 모델의 복잡한 예측 과정을 보다 신뢰하고 수정할 수 있는 기반을 제공합니다.



### Symbolic Learning Enables Self-Evolving Agents (https://arxiv.org/abs/2406.18532)
Comments:
          Code available at this https URL

- **What's New**: AI 커뮤니티는 '언어 에이전트(language agents)'를 개발하여 인공지능 일반지능(AGI)으로 가는 길을 탐구하고 있습니다. 현재의 언어 에이전트 연구는 모델 중심(model-centric) 또는 엔지니어링 중심(engineering-centric) 접근 방식에 의존하고 있어 많은 수작업이 필요하다는 한계가 있습니다. 본 연구에서는 이러한 문제를 해결하고, 언어 에이전트가 데이터 중심(data-centric) 방식으로 자율적으로 학습하고 발전할 수 있도록 돕는 '에이전트 상징 학습(agent symbolic learning)' 프레임워크를 제안합니다.

- **Technical Details**: 제안된 에이전트 상징 학습 프레임워크는 언어 에이전트를 상징적 네트워크(symbolic networks)로 간주합니다. 이 네트워크의 가중치는 프롬프트(prompts)와 도구(tools), 그리고 이들이 결합된 방식에 의해 정의됩니다. 에이전트 상징 학습은 백프로파게이션(back-propagation)과 그래디언트 디센트(gradient descent)의 두 가지 기본 알고리즘을 모방하여 이러한 상징적 네트워크를 최적화합니다. 구체적으로, 언어 기반 가중치, 손실(loss), 그래디언트를 사용하여 에이전트를 최적화합니다.

- **Performance Highlights**: 기본적인 벤치마크와 복잡한 실제 작업에서 실험을 수행한 결과, 제안된 에이전트 상징 학습 프레임워크가 에이전트 시스템을 효과적으로 최적화하고 설계할 수 있음을 보여주었습니다. 이를 통해 데이터 중심 방식으로 역량을 발휘하는 자가 진화형 에이전트(self-evolving agents)를 구현할 수 있음을 증명했습니다.



### PrExMe! Large Scale Prompt Exploration of Open Source LLMs for Machine Translation and Summarization Evaluation (https://arxiv.org/abs/2406.18528)
Comments:
          Preprint

- **What's New**: 이 연구는 PrExMe라는 새로운 대규모 프롬프트 탐색 시스템을 도입하여 720개 이상의 프롬프트 템플릿을 사용하여 오픈 소스 LLM 기반 평가 메트릭을 머신 번역(MT) 및 요약 데이터셋에서 평가했습니다. 총 660만 번의 평가를 통해 이러한 프롬프트 기반 메트릭의 성능을 비교하고 안정성과 변동성을 탐구했습니다.

- **Technical Details**: PrExMe는 각기 다른 입력 프롬프트의 변동이 LLM 기반 메트릭의 인간 평가와의 상관 관계에 미치는 영향을 체계적으로 평가합니다. 주요 프롬프트 기법으로는 zero-shot, chain-of-thought (CoT), retrieval-augmented generation (RAG)가 사용되었고, 요청된 출력 형식을 다양하게 변경하여 평가했습니다. 연구는 옵셋 소스 LLM 7개를 사용하여 720개 이상의 프롬프트 템플릿을 통해 수행되었습니다.

- **Performance Highlights**: 연구 결과, 특정 프롬프트 패턴들이 다양한 작업과 데이터셋에서 일관된 성능을 보이는 반면, 작은 프롬프트 변화에도 성능이 크게 영향을 받을 수 있음을 발견했습니다. 예를 들어, "0 to 100" 형식에서 "-1 to +1" 형식으로의 변경만으로도 평가 결과가 크게 달라질 수 있습니다. 또한 Platypus2-70B 모델이 테스트한 LLM 중에서 가장 강력한 성능을 보였습니다.



### CharXiv: Charting Gaps in Realistic Chart Understanding in Multimodal LLMs (https://arxiv.org/abs/2406.18521)
Comments:
          121 pages, 90 figures

- **What's New**: 새로운 연구는 차트 이해를 향상시키기 위한 CharXiv라는 평가 스위트를 도입했습니다. 이 스위트는 arXiv 논문에서 추출한 2,323개의 자연스럽고 다양한 차트로 구성되며, 두 가지 유형의 질문을 포함합니다: 기본적인 차트 요소를 조사하는 설명 질문과 복잡한 시각적 요소를 종합하여 정보를 추론하는 질문입니다.

- **Technical Details**: CharXiv는 템플릿 기반 질문에 초점을 맞춘 기존 데이터세트의 한계를 극복하고자 합니다. 기존의 일반적인 벤치마크에서 두드러진 성능을 보이는 오픈소스 모델들이 약간의 변화만으로 성능이 최대 34.5%까지 저하될 수 있음을 보여주었습니다. CharXiv의 모든 차트와 질문은 사람 전문가들에 의해 손수 선정되고 검증되었습니다.

- **Performance Highlights**: CharXiv의 결과는 최신의 프로프라이어터리 모델인 GPT-4o가 47.1%의 정확도를, 강력한 오픈소스 모델인 InternVL Chat V1.5이 29.2%의 정확도를 기록했음을 보여줍니다. 이는 80.5%의 인간 성능과 큰 차이가 있으며, 현재 MLLM들이 차트 이해 능력에서 부족함을 강조합니다.



### APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets (https://arxiv.org/abs/2406.18518)
- **What's New**: 이번 논문에서는 API 데이터 세트를 자동으로 생성하는 APIGen이라는 파이프라인을 소개합니다. APIGen을 사용하여 21개의 다양한 카테고리에서 3,673개의 실행 가능한 API를 수집하고, 이를 통해 다양한 함수 호출(function-calling) 데이터 세트를 생성합니다. 생성된 데이터는 포맷 검증, 실제 함수 실행, 의미 검증의 세 가지 계층적 단계를 거쳐 신뢰성과 정확성을 보장합니다.

- **Technical Details**: APIGen은 데이터 생성 파이프라인으로, 다양한 함수 호출 데이터를 구조화되고 확장 가능한 방식으로 합성합니다. 생성되는 각 데이터는 포맷 체크(format checking), 실제 함수 실행(function executions), 의미 검증(semantic verification)의 세 가지 단계로 검증됩니다. 이 절차를 통해 데이터의 신뢰성과 정확성을 보장합니다. 또한, 최종적으로 60,000개의 고품질 엔트리로 구성된 데이터 세트를 출시했습니다.

- **Performance Highlights**: APIGen을 통해 생성된 데이터로 학습된 모델은 Berkeley Function-Calling Benchmark 테스트에서 최신 성능(state-of-the-art)을 달성했으며, 이는 여러 GPT-4 모델을 능가하는 결과를 보여주었습니다. 특히, 1B 파라미터 모델은 GPT-3.5-Turbo와 Claude-3 Haiku를 뛰어넘는 탁월한 성능을 보였습니다. 데이터 세트는 Huggingface에서 공개되어 있습니다.



### "Is ChatGPT a Better Explainer than My Professor?": Evaluating the Explanation Capabilities of LLMs in Conversation Compared to a Human Baselin (https://arxiv.org/abs/2406.18512)
Comments:
          6 figures, 5 pages

- **What's New**: 이번 연구는 설명 과정에서 대화적 접근법(conversational approach)을 활용하여 AI의 가능성을 탐구합니다. 인간 설명자와 AI 설명자 간의 상호작용 동적을 이해하기 위해, WIRED YouTube 시리즈에서 추출한 '5-레벨 데이터셋'(5-Levels dataset)을 활용했습니다.

- **Technical Details**: 이 연구는 보슈헤리 등의 주석이 달린 '5-레벨 데이터셋'을 사용합니다. 이 데이터셋은 한 명의 전문가가 다양한 수준의 학습자(아동, 청소년, 대학생, 대학원생, 동료)와 대화하는 구조로 이루어져 있습니다. 실험에서는 대학생 수준의 학습자와의 대화를 주요 대상으로 삼았습니다. 여기서 세 가지 접근법을 비교했는데, 첫 번째는 인간 설명자의 실제 응답(S1), 두 번째는 GPT-4의 기본 응답(S2), 세 번째는 '설명 행위'(Explanation Acts)를 기반으로 한 GPT-4의 응답(S3)을 포함합니다.

- **Performance Highlights**: 실험 결과는 8가지 차원에서 각 설명의 응답을 평가했으며, 여기에는 일관성(Coherence), 간결성(Concise), 대화성(Conversational), 인정(Acknowledgement), 적절성(Appropriate), 심화 또는 확장(Deepens or Expands)이 포함되었습니다. 이 연구는 LLM이 얼마나 인간과 유사한 설명을 생성할 수 있는지, 그리고 이를 통해 인간 설명자의 능력을 어떻게 증강시킬 수 있는지에 대한 중요한 통찰을 제공합니다.



### WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models (https://arxiv.org/abs/2406.18510)
- **What's New**: WildTeaming이라는 혁신적인 LLM(Large Language Model) 안전성 테스트 도구가 발표되었습니다. 이 도구는 '인 더 와일드(In-the-Wild)' 사용자-챗봇 상호작용을 통해 5,700개의 독특한 탈옥(jailbreak) 전술 클러스터를 발견하고, 복합 전술을 활용하여 새로운 탈옥 시도를 탐색합니다. 이 방법은 기존의 연구보다 최대 4.6배 더 다양한 성공적인 공격을 발견할 수 있습니다.

- **Technical Details**: WildTeaming은 두 단계의 프로세스를 거칩니다. 첫 번째 단계는 'Mine' 단계로, 실제 사용자-챗봇 상호작용 데이터를 분석하여 인간이 고안한 탈옥 전술을 자동으로 추출합니다. 두 번째 'Compose' 단계에서는 추출된 전술을 조합하여 다양한 적대적 공격을 생성합니다. 이를 통해 LMSYS-Chat-1M 및 WildChat 데이터셋에서 105K의 전술을 식별했습니다. WildJailbreak는 262K의 프롬프트-응답 쌍을 포함하는 대규모 오픈 소스 안전성 데이터셋으로, 이를 기반으로 안전성 교육을 실시합니다.

- **Performance Highlights**: WildTeaming은 기존의 탈옥 방법보다 40% 적은 시도로 최대 4.6배 더 많은 성공적인 공격을 발견해냈습니다. 특히 블랙박스 및 화이트박스 LLM에 대해 더 효과적인 공격을 발굴했습니다. WildJailbreak 데이터셋을 이용한 실험 결과, 모델의 일반적 성능을 저하시키지 않으면서도 안전성 행동의 균형을 유지할 수 있음을 확인했습니다.



### Is In-Context Learning a Type of Gradient-Based Learning? Evidence from the Inverse Frequency Effect in Structural Priming (https://arxiv.org/abs/2406.18501)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 문맥 학습(in-context learning, ICL)이 실제로는 경사하강법(gradient descent)을 수행하는지 진단하는 새로운 방법을 소개합니다. 연구진은 LLMs가 인빈스 빈도 효과(inverse frequency effect, IFE)를 보여주는지 여부를 조사함으로써 이 문제를 탐구합니다. 기존 연구에서 IFE는 구조적 프라이밍(structural priming) 맥락에서 인간의 학습이 오류 기반 학습 메커니즘에 의해 이루어진다는 증거로 사용되었습니다. 모의 실험 결과, LLMs가 특히 큰 모델에서 IFE를 보여주며, 이는 ICL이 경사 기반 학습의 일종일 가능성을 강력하게 뒷받침한다고 결론지었습니다.

- **Technical Details**: 연구팀은 인간의 구조적 프라이밍 패러다임에서 인빈스 빈도 효과를 관찰하는 방식을 차용하여 LLMs의 ICL이 경사 기반 학습인지 여부를 진단했습니다. 구조적 프라이밍은 화자들이 최근에 마주친 구문 구조를 반복 사용하는 경향을 의미합니다. IFE는 잘 사용되지 않는 구문 구조가 선호도가 높은 구조보다 더 큰 프라이밍 효과를 나타내는 현상입니다. LLMs를 이용한 구조적 프라이밍 실험을 통해, LLMs가 IFE 현상을 나타내는지를 관찰했습니다.

- **Performance Highlights**: 실험 결과, LLMs는 전통적인 경사하강법 학습과 표준 미세조정(fine-tuning) 상황 하에서 강력한 IFE를 보여주며, ICL 설정에서도 다양한 정도의 IFE를 나타냈습니다. 특히 큰 모델에서 IFE가 더 강하게 나타났습니다. 이러한 결과는 ICL이 경사 기반 학습 메커니즘이라는 가설을 강력히 지지하며, 인간과 LLMs가 오류 기반 학습 프로세싱 메커니즘을 공유할 수 있음을 시사합니다.



### WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs (https://arxiv.org/abs/2406.18495)
Comments:
          First two authors contributed equally. Third and fourth authors contributed equally

- **What's New**: WildGuard는 사용자 요청의 악의적 의도 식별, 모델 응답에 대한 안전 위험 탐지, 모델 거부율 측정의 세 가지 목표를 달성하는 새로운 경량의 안전 모니터링 도구입니다. 총 13가지 위험 범주에서 높은 정확도를 제공하며, 기존의 공개 모니터링 도구와 달리 적대적 '탈출구' 탐지와 모델 거부율 평가에서 큰 개선을 이루었습니다.

- **Technical Details**: WildGuard는 WildGuardMix라는 대규모 데이터셋을 사용하여 학습되었으며, 이 데이터셋은 9만2천 개의 레이블 예제로 구성되어 있습니다. WildGuardMix는 WildGuardTrain(학습 데이터)과 WildGuardTest(평가 데이터)로 나뉘며, 각 5천299개의 항목이 포함됩니다. Llama-Guard2 등의 기존 도구와 비교해 WildGuard는 Prompt Harmfulness, Response Harmfulness, Response Refusal 세 가지 주요 작업에서 뛰어난 성능을 보입니다.

- **Performance Highlights**: WildGuard는 10개의 공개 벤치마크에서 최신 성능을 보여 주며, GPT-4를 상회하기도 합니다(p.e., 프롬프트 해악성 평가에서 최대 3.9% 개선). 시스템의 탈출구 공격 성공률을 79.8%에서 2.4%로 크게 낮췄습니다. 또한, WildGuardTrain의 각 구성 요소가 성능 향상의 중요한 역할을 했고, 멀티태스크 학습이 싱글태스크보다 뛰어남을 입증했습니다.



### Role-Play Zero-Shot Prompting with Large Language Models for Open-Domain Human-Machine Conversation (https://arxiv.org/abs/2406.18460)
Comments:
          Updated version of a paper originally submitted at SIGDIAL 2023

- **What's New**: 이번 연구는 비용 효율적인 오픈도메인 대화 봇 생성 방법으로 Role-Play Zero-Shot Prompting을 탐구합니다. 이는 Vicuna 모델과 같은 다국어 대규모 언어 모델(Large Language Models, LLMs)을 사용하여 특별한 데이터셋 없이도 인간 평가에서 우수한 성능을 보입니다.

- **Technical Details**: Role-Play Zero-Shot Prompting 접근법은 Prompt-Based Learning(PBL) 패러다임에 속하며, 대규모 언어 모델의 지시 따르기(instruction-following) 능력을 활용합니다. 주로 PersonaChat 데이터셋을 활용한 Persona task와 이미지를 논의하는 INT task에서 이 방법을 평가했습니다. 이 방법은 데이터셋 구축 비용과 언어의 제약을 줄여준다.

- **Performance Highlights**: 이 연구는 프랑스어로 두 가지 과제에서 역할극(Role-Play) 제로 샷(Prompting) 접근법이 미세 조정된 모델과 대등하거나 그보다 우수한 성능을 보인다는 사실을 확인했습니다.



### Cascading Large Language Models for Salient Event Graph Generation (https://arxiv.org/abs/2406.18449)
Comments:
          9 + 12 pages

- **What's New**: 본 논문에서는 긴 문서로부터 이벤트 그래프를 생성하는 문제를 다룹니다. 의미 있는 이벤트를 강조하고 관계를 식별하는 새로운 방식인 CALLMSAE(CAscading Large Language Model framework for SAlient Event graph generation)를 제안합니다. 이 접근법은 LLMs(Large Language Models)를 활용하여 비싼 수작업 주석 작업을 없애고, 문서 요약을 통해 중요한 사건을 식별하며, 반복적 코드 개선 프롬프트 전략을 사용하여 이벤트 관계 그래프를 생성합니다.

- **Technical Details**: CALLMSAE는 주로 두 가지 주요 단계로 구성됩니다. 첫 번째 단계는 LLMs에게 문서를 요약하도록 지시하여 중요한 이벤트를 식별합니다. 두 번째 단계에서는 반복적 코드 개선 프롬프트 전략을 통해 이벤트 관계 그래프를 생성합니다. 이 과정에서 거짓된 관계를 제거하고 누락된 에지를 회복하는 방식으로 그래프의 정확성을 향상시킵니다. 이 접근법은 CAEVO 기반 데이터로 훈련된 모델보다 우수한 성능을 보입니다. 또한, 평가 기준으로 의미적 텍스트 임베딩에 기반한 새로운 평가 메트릭을 제안합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 인간 주석 테스트 세트에서 더 정확한 이벤트 그래프를 생성한다는 것이 입증되었습니다. CALLMSAE가 이벤트 중요성 및 에지 품질 측면에서 경쟁 모델들을 능가하며, 제안된 평가 메트릭이 효과적으로 동작하는 것이 확인되었습니다.



### IRCAN: Mitigating Knowledge Conflicts in LLM Generation via Identifying and Reweighting Context-Aware Neurons (https://arxiv.org/abs/2406.18406)
Comments:
          19 pages, 13 figures, 5 tables

- **What's New**: IRCAN이라고 불리는 새로운 프레임워크를 소개합니다. 이는 대형 언어 모델(LLMs)에서 지식 충돌을 해결하기 위한 방법으로, 문맥에 민감한 뉴런을 식별하고 이들의 중요성을 재조정함으로서 모델이 새로운 문맥 지식을 더 잘 반영하도록 만듭니다.

- **Technical Details**: IRCAN은 먼저 통합 경사(Integrated Gradients)에서 파생된 문맥 인식 기여 점수를 사용해 문맥 처리에 중요한 뉴런을 식별합니다. 그런 다음, 식별된 문맥 인식 뉴런의 가중치를 증가시켜 모델이 생성 과정에서 문맥에 제공된 새로운 지식에 민감하게 반응하도록 합니다. 이 과정은 복잡한 실험을 통해 다양한 모델과 태스크에서 성능을 검증받았습니다.

- **Performance Highlights**: IRCAN은 LLaMA, Gemma, Amber 같은 다양한 LLM 패밀리와 2B, 7B, 8B, 13B 규모의 모델에서 탁월한 성능 향상을 보여주었습니다. 예를 들어, LLaMA-2-7B에서는 정확도가 129%, LLaMA-3-8B에서는 136% 향상되었습니다. 또한, 기존의 컨텍스트 인식 디코딩(CAD) 방법과 결합하면 성능이 더욱 개선됩니다.



### LLMs instead of Human Judges? A Large Scale Empirical Study across 20 NLP Evaluation Tasks (https://arxiv.org/abs/2406.18403)
- **What's New**: 최근 대형 언어 모델(LLM)이 인간 평가자 대신 사용되는 경향이 증가하고 있습니다. 이를 해결하기 위해, JUDGE-BENCH라는 20개의 인간 주석이 포함된 NLP 데이터셋 모음을 제공하고, 11개의 최신 LLM을 포괄적으로 평가했습니다. 연구 결과, LLM은 데이터셋마다 인간 판단과의 상관관계에 큰 변동을 보였으며, 시스템적인 인간 평가자 대체가 아직 어렵다는 결론을 도출했습니다.

- **Technical Details**: JUDGE-BENCH는 다양한 품질 차원을 포함하는 20개의 인간 주석 데이터셋으로 구성되어 있으며, 데이터셋은 모델 생성 및 인간 생성 항목을 평가합니다. 대표적인 비공개 및 공개 가중치 모델(예: GPT-4o, LLaMA-3 등)을 선택하여 11개의 모델을 평가하였습니다. 실험에서 여러 데이터셋에 걸쳐 모델과 인간 판단 간의 상관관계를 분석했습니다.

- **Performance Highlights**: 주요 결과로, 일부 LLM은 특정 데이터셋에서 인간 판단과 잘 맞았으나 다른 데이터셋에서는 성능이 크게 떨어졌습니다. 특히 GPT-4o와 LLaMA3-70B는 전체적으로 우수한 성능을 보였으나, 모든 모델의 평가점수를 보정하려면 인간 판단과의 비교가 필요합니다.



### Do LLMs dream of elephants (when told not to)? Latent concept association and associative memory in transformers (https://arxiv.org/abs/2406.18400)
- **What's New**: 새로운 연구는 대형 언어 모델(LLMs)이 사실을 저장하고 회상하는 능력을 갖추고 있으며, 문맥을 변화시킴으로써 이러한 능력이 쉽게 조작될 수 있음을 보여줍니다. 이로 인해 LLMs는 특정 토큰이 사실 회상을 유도하는 단서 역할을 하는 연관 기억 모델처럼 작동할 수 있습니다.

- **Technical Details**: 이 연구에서는 LLMs의 기본 구성 요소인 transformer가 어떻게 자기-어텐션(self-attention) 레이어를 통해 메모리 작업을 수행하는지 수학적으로 탐구합니다. 특히, 단일 레이어의 transformer를 사용하여 잠재 개념 연관(associative) 문제를 연구하고 이 과정에서 자기-어텐션 레이어가 정보를 모으고 value matrix가 연관 기억 기능을 한다는 것을 이론적으로 및 경험적으로 입증합니다.

- **Performance Highlights**: 연구 결과, 여러 오픈 소스 LLM 모델(GPT-2, LLaMA-2, Gemma 등)이 문맥으로 인해 사실 회상이 쉽게 오도될 수 있다는 것을 명확히 보여줍니다. 또한 자체적 실험을 통해 잠재 개념 연관 작업을 통한 transformer의 메모리 회상 능력을 분석하였으며, 이는 기존의 로우-랭크(low-rank) 편집 및 미세 조정 기법에 대한 추가적 이론적 근거를 제공합니다.



### Dynamic Data Pruning for Automatic Speech Recognition (https://arxiv.org/abs/2406.18373)
Comments:
          Accepted to Interspeech 2024

- **What's New**: 최근의 자동 음성 인식(Automatic Speech Recognition, ASR) 성공은 점점 늘어나는 훈련 데이터 양 덕분입니다. 하지만, 이는 모델 훈련 비용을 높이고 계산 요구를 증가시켰습니다. 이런 문제를 해결하기 위해 데이터 프루닝(Data Pruning)이 제안되었으나, ASR에서는 거의 탐구되지 않았습니다. 본 논문은 ASR에 대한 동적 데이터 프루닝(Dynamic Data Pruning, DDP)을 최초로 조사하고, 70%의 데이터만으로도 전체 데이터를 사용하는 것과 유사한 성능을 달성할 수 있음을 발견했습니다. 이를 위해 'DDP-ASR'을 도입하여 음성 데이터셋에 맞춘 세세한 프루닝 기법을 제안합니다.

- **Technical Details**: 기존 데이터 프루닝은 전체 시간 시퀀스(time sequences)를 제거하는 데 집중하였으나, 'DDP-ASR'은 세부적인 프루닝 단계를 도입하여 개별 시간 포인트 뿐만 아니라 시간 청크(temporal chunks)의 일부를 제거하는 방식을 적용하였습니다. 또한, 커리큘럼 학습 전략(curriculum learning strategy)을 적용하여 데이터의 70%만으로도 전체 데이터와 유사한 성능을 달성했습니다. 이를 통해 프루닝 효율성을 극대화하였습니다.

- **Performance Highlights**: DDP-ASR을 사용하면 거의 성능 저하 없이 최대 1.6배의 훈련 속도 향상이 가능합니다. 실험 결과, Librispeech 데이터셋에서 최대 1.6배의 속도 향상을 달성하면서도 원래 데이터와 유사한 성능을 유지할 수 있음을 확인했습니다.



### Themis: Towards Flexible and Interpretable NLG Evaluation (https://arxiv.org/abs/2406.18365)
- **What's New**: 이번 연구는 자연어 생성(NLG) 평가를 위한 새로운 자동 평가 모델을 제안하는 내용입니다. 기존의 문자열 기반 평가 지표와 모델 기반 지표가 가지는 한계점을 극복하기 위해, 이 논문에서는 GPT-4 주석을 포함하여 인간 및 GPT-4가 주석을 다는 대규모 NLG 평가 코퍼스 NLG-Eval을 구성하였습니다. 또한, 다중 관점 일관성 및 등급 지향 선호도 정렬 방법을 적용한 Themis라는 NLG 평가 전용 대형 언어 모델(LLM)을 제안합니다.

- **Technical Details**: Themis는 8B(백억) 파라미터를 가진 LLM으로, 기존의 평가 방법론과 달리 참조 없이도 유연하고 해석 가능한 평가를 수행할 수 있습니다. 이를 위해, NLG-Eval 코퍼스에서 58개의 데이터셋과 약 50만개의 샘플을 수집하고, GPT-4 주석을 추가로 사용하여 데이터의 신뢰성을 높였습니다. 또한, 멀티퍼스펙티브 일관성 검증 방법과 구체적인 선호 정렬 방법을 설계하여 더 나은 평가 성능을 달성하였습니다.

- **Performance Highlights**: Themis는 요약, 스토리 생성, 데이터-텍스트 변환 등 다양한 NLG 작업에서 기존 평가 모델들을 능가하는 성능을 보여줍니다. 새로운 작업에 대한 일반화 능력도 우수하며, 특정 측면을 목표로 한 교란 테스트에서도 높은 신뢰성을 입증했습니다. 이를 통해 Themis는 일반적인 NLG 평가 작업뿐만 아니라, 드문 작업에서도 뛰어난 평가 성능을 발휘합니다.



### Research on Information Extraction of LCSTS Dataset Based on an Improved BERTSum-LSTM Mod (https://arxiv.org/abs/2406.18364)
Comments:
          submitted to ICMIII 2024

- **What's New**: 이 논문은 LCSTS 데이터셋을 기반으로 한 정보 추출 방법을 연구하고, 이를 위해 개선된 BERTSum-LSTM 모델을 사용하여 중국어 뉴스 요약을 생성하는 방안을 제안하고 있습니다. 이 개선된 모델은 뉴스 요약 생성에서 높은 성능을 보입니다.

- **Technical Details**: 기존의 BERTSum-LSTM 모델을 개선하여 사용했습니다. 중국어 뉴스의 복잡한 의미와 많은 정보를 효과적으로 추출하기 위해 모델의 성능을 향상시켰습니다. 폴리세미(Polysemy) 및 단어 분할(word segmentation)과 같은 중국어의 특수성 문제도 다루어졌습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 뉴스 요약 생성에서 매우 좋은 효과를 내는 것으로 나타났습니다. 이는 뉴스 요약 구성에 중요성을 갖고 있습니다.



### Grammar Assistance Using Syntactic Structures (GAUSS) (https://arxiv.org/abs/2406.18340)
Comments:
          5 pages, 4 figures, project summary for CEDI-SEPLN Seminar of the Spanish Society for Natural Language Processing at the 7th Spanish Conference on Informatics, June 19-20, 2024, A Coruña, Spain

- **What's New**: GAUSS 프로젝트는 스페인어를 위한 새로운 문법 코칭 시스템을 제안합니다. 이 시스템은 의미 있는 피드백을 제공하는 풍부한 언어 형식체계를 사용하며, 빠른 구문 분석 알고리즘을 통해 실용적으로 적용할 수 있습니다. 이를 통해 고가의 신경망 기반 방법에 대한 의존도를 줄이고, 전 세계의 교육 문제를 해결하며 포용성과 참여도를 높이는 것을 목표로 합니다.

- **Technical Details**: GAUSS 프로젝트는 전산 언어학(NLP), 이론 언어학 및 응용 언어학 등의 연구 영역이 협력하여 개발된 것입니다. 특히 머리-구동 구문 구조 문법(HPSG; Head-driven Phrase Structure Grammar) 형식으로 구현된 스페인어 자원 문법(SRG)에 의존합니다. 제한 통일 이론(Constraint Unification Theory)의 구문이론인 HPSG는 문장을 구성 요소의 구조로 분석하여 동사와 주어의 일치 값 등을 제약합니다.

- **Performance Highlights**: 기존의 시스템이 통계적 방법에만 의존하고 해석 가능성이 부족한 반면에, GAUSS 시스템은 명시적인 언어 지식을 사용하여 문법적으로 올바르지 않은 문자열을 판별합니다. 스페인어에서 시작하여 그리스어와 같은 소수 언어에도 확장 가능성이 높으며, 신경망 기반 방법의 비용을 줄이고 생태학적으로도 덜 문제가 되는 솔루션을 제공합니다.



### PaCoST: Paired Confidence Significance Testing for Benchmark Contamination Detection in Large Language Models (https://arxiv.org/abs/2406.18326)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)이 광범위한 데이터셋을 기반으로 학습되었을 때 발생할 수 있는 벤치마크 오염 문제를 해결하기 위해 PaCoST(Paired Confidence Significance Testing)라는 새로운 검출 방법을 제안합니다. PaCoST는 모델이 과도한 자신감을 보이는 경우를 통계적으로 분석하여 벤치마크 오염 여부를 판별하는 방식입니다.

- **Technical Details**: PaCoST는 세 단계의 통계 분석을 통해 벤치마크 오염을 감지합니다. 각 데이터 인스턴스에 대해 동일한 분포를 가진 대응 데이터를 구성하고, 모델의 신뢰도(confidence)를 분석하여 원래 벤치마크 앞에서 모델이 더 높은 신뢰도를 보이는지를 테스트합니다. 이는 모델이 학습된 질문에 대해 더 높은 신뢰도를 보인다는 가정 하에 수행됩니다.

- **Performance Highlights**: PaCoST는 여러 오픈 소스 모델과 벤치마크에 적용되어 효과가 검증되었습니다. 실험 결과 거의 모든 모델과 벤치마크에서 오염이 의심되는 것으로 나타났습니다. 따라서, 벤치마크가 포함되지 않는 새로운 평가 방법의 도입을 권장합니다.



### MathOdyssey: Benchmarking Mathematical Problem-Solving Skills in Large Language Models Using Odyssey Math Data (https://arxiv.org/abs/2406.18321)
- **What's New**: 이번 논문은 새로운 'MathOdyssey' 데이터셋을 소개하여, 고등학교 및 대학교 수준의 다양한 수학 문제를 통해 대형 언어 모델(LLMs)의 수학 문제 해결 능력을 평가합니다. 이 데이터셋은 유명 기관의 전문가들이 만든 문제들로 구성되어 있으며, AI 공동체에 제공함으로써 AI의 복잡한 수학적 추론 능력을 향상시키고자 합니다.

- **Technical Details**: MathOdyssey 데이터셋은 고등학교와 대학 수준의 문제를 포함하여 다양한 수학적 주제를 다룹니다. 문제는 Olympia급부터 고등학교, 대학교 수준까지 다양하게 포함되어 있습니다. 데이터셋은 새로운 벤치마크로서 공정한 평가를 위해 이전에 모델 훈련에 사용되지 않았습니다. 우리는 open-source 모델(Llama-3, DBRX-Instruct)과 closed-source 모델(GPT 시리즈, Gemini 모델)에서 이 데이터셋을 사용하여 벤치마킹을 수행하였습니다.

- **Performance Highlights**: 결과에 따르면, LLMs는 일상적인 문제나 중간 난이도의 문제에서는 잘 수행하지만, Olymipiad 급 문제나 복잡한 대학 수준 문제에서는 여전히 많은 도전에 직면합니다. open-source 모델과 closed-source 모델 간의 성능 차이는 점점 줄어들고 있지만, 가장 어려운 문제들에는 여전히 많은 어려움이 남아 있습니다. 이 연구는 LLMs의 수학적 추론 역량을 높이기 위한 지속적인 연구의 필요성을 강조합니다.



### AI-native Memory: A Pathway from LLMs Towards AGI (https://arxiv.org/abs/2406.18312)
- **What's New**: 대형 언어 모델(LLM)이 인공지능 일반화(AGI)를 실현할 가능성을 보여주고 있는 가운데, 일부 스타트업은 거의 무제한의 문맥 길이를 가진 LLM이 AGI를 실현할 수 있다고 주장합니다. 그러나 최근 연구에 따르면 현재 존재하는 LLM은 주장된 문맥 길이보다 훨씬 짧은 효과적인 문맥 길이를 가지고 있으며, 우리의 실험은 긴 문맥에서 관련 정보를 동시에 찾아내고 단순한 추론을 수행하는 것이 거의 불가능하다는 것을 보여줍니다.

- **Technical Details**: 기존 LLM의 문맥 처리 능력은 실제로 제한적입니다. 예를 들어, GPT-4는 128K 토큰의 문맥 길이를 주장하지만, 실제로는 64K 토큰 정도만 효과적으로 활용할 수 있습니다. 이에 따라 우리는 LLM과 AGI를 연결할 방법으로 메모리(memory)의 통합을 제안합니다. 메모리는 단순히 원시 데이터를 저장하는 것뿐만 아니라 추론 과정에서 도출된 중요한 결론들을 저장하며, 이는 저장된 정보를 통해 복잡한 추론을 간소화합니다.

- **Performance Highlights**: 우리의 '하이 스택에서 바늘 찾기(reasoning-in-a-haystack)' 실험은 현재 LLM이 긴 문맥 내에서 관련 정보를 찾아내고 추론하는 능력이 매우 제한적임을 보여줍니다. 이는 기존의 검색 강화 생성(retrieval-augmented generation, RAG)과는 달리 메모리를 원시 데이터를 넘어선 개념으로 발전시켜야 한다는 필요성을 시사합니다. 메모리는 궁극적으로 모든 데이터 유형을 파라미터화하고 압축할 수 있는 개인 맞춤형 모델(LPM) 형태로 발전해야 합니다.



### S3: A Simple Strong Sample-effective Multimodal Dialog System (https://arxiv.org/abs/2406.18305)
- **What's New**: 이 연구에서는 멀티모달 대화(multi-modal dialog) 작업을 위한 새로운 베이스라인 모델, S3 모델을 소개합니다. 이 모델은 MMMU와 AI Journey Contest 2023이라는 두 가지 주요 리더보드에서 최첨단 성능에 근접한 결과를 보여줍니다. 이 시스템은 사전 훈련된 대형 언어 모델, 이미지와 오디오를 위한 사전 훈련된 모달리티 인코더, 그리고 학습 가능한 모달리티 프로젝터로 구성됩니다.

- **Technical Details**: 모델은 150,000개 미만의 멀티모달 샘플, 7B 언어 모델, 그리고 단일 A100-80GB GPU로 학습되었습니다. 멀티레이어 퍼셉트론(MLP)을 사용하는 모달리티 프로젝터가 모달리티 피처를 토큰 임베딩으로 매핑하는 핵심 기술입니다. JSON 형식으로 각 메시지를 포함하는 'role', 'type', 'message content'와 같은 속성을 포함한 대화 레이아웃을 통해 사용자가 대화 중에 시각적 또는 청각적 콘텐츠를 삽입할 수 있습니다.

- **Performance Highlights**: 이 모델은 작은 양의 멀티모달 데이터로도 효율적으로 성능을 발휘하며, 다소 복잡한 시스템과 경쟁할 수 있을 정도로 준수한 성능을 보여줍니다. 이를 통해 대규모 데이터셋과 과도한 컴퓨팅 자원이 필수적이지 않음을 증명했습니다.



### FactFinders at CheckThat! 2024: Refining Check-worthy Statement Detection with LLMs through Data Pruning (https://arxiv.org/abs/2406.18297)
- **What's New**: 이 논문에서는 정치적 대화록에서 검증이 필요한 주장을 식별하기 위해 여덟 가지 주요 오픈 소스 대형 언어 모델(open-source Large Language Models, LLMs)을 사용하여 미세 조정(fine-tuning)과 프롬프트 엔지니어링(prompt engineering)을 수행했다. 이러한 접근법을 통해 AI 모델들이 어떤 주장이 사실 확인이 필요할 만한 가치가 있는지 식별하는 데 중점을 둔다.

- **Technical Details**: 체크워드니스 추정을 위한 데이터셋인 CheckThat! 2024를 분석하여 잘못된 정보가 포함된 문장을 식별하는 두 단계 데이터 절단(프루닝, pruning) 접근법을 제안했다. 사용된 LLMs는 Llama2-7b, Llama2-13b, Llama3-8b, Mistral, Mixtral, Phi3-Mini-4K, Falcon, Gemma-7b 등 여덟 가지로 구성된다. 데이터 절단 기술은 유익한 문장을 먼저 식별하고, 균형 잡힌 학습 데이터셋을 만들기 위해 Condensed Nearest Neighbour 방법을 적용한다.

- **Performance Highlights**: 논문에 따르면, 데이터 절단을 적용하면서 학습 데이터의 약 44%만 사용해도 경쟁력 있는 성능을 낼 수 있었다. 이 접근법은 학습 데이터 양을 대폭 줄이면서도 모델의 성능을 크게 떨어뜨리지 않으며, 실제로 최고 성능을 기록한 Llama2-7b 모델이 있는 다른 데이터세트에서 최초로 등수를 차지했다. 이러한 방법을 통해 미세 조정(fine-tuning) 시간도 대폭 줄일 수 있었다.



### Hierarchical Context Pruning: Optimizing Real-World Code Completion with Repository-Level Pretrained Code LLMs (https://arxiv.org/abs/2406.18294)
- **What's New**: 최근 개발된 코드 언어 모델(Code LLMs)은 코드 리포지토리 데이터를 기반으로 사전 학습되어, 리포지토리 구조를 인식하고 파일 간 정보를 활용할 수 있게 되었습니다. 그러나 실제 개발 환경에서는 전체 코드 리포지토리를 단순히 연결하면 모델의 컨텍스트 윈도우 한계를 초과하여 성능이 저하됩니다. 본 연구에서는 6개의 Repo-Code LLMs에 대한 실험을 통해, 파일 간 토폴로지 의존성을 유지하고 코드 파일 내용을 증가시키는 것이 코드 완성 정확성을 높일 수 있음을 발견했습니다. 이에 따라 Hierarchical Context Pruning(HCP) 전략을 제안하여, 기능 수준에서 코드 리포지토리를 모델링하고 불필요한 코드를 제거하여 입력 길이를 줄이고 정확성을 크게 향상시켰습니다.

- **Technical Details**: HCP 전략은 코드 리포지토리를 기능 수준에서 모델링하며, 파일 간의 토폴로지 의존성을 유지하면서 관련 없는 코드 내용을 제거하는 방식입니다. 이 접근을 통해 입력 길이를 50,000 토큰 이상에서 약 8,000 토큰으로 줄일 수 있었습니다. 또한, 이 전략은 코드 완성 프롬프트(conpletion prompts)의 품질을 높이기 위해 설계되었습니다. 6개의 Repo-Code LLMs를 사용한 실험 결과, HCP 전략이 코드 완성 정확성을 크게 개선하면서 입력 길이를 상당히 줄일 수 있음을 입증했습니다.

- **Performance Highlights**: 실험 결과, HCP 전략은 코드 완성 정확성을 크게 향상시키면서 입력 길이를 기존의 50,000 토큰 이상에서 약 8,000 토큰으로 줄이는 데 성공했습니다. 우리의 방법론은 6개의 Repo-Code LLMs에서 검증되었으며, 프롬프트 품질을 높이고 정확성을 개선하는 데 효과적임을 보여줬습니다.



### Sanskrit Knowledge-based Systems: Annotation and Computational Tools (https://arxiv.org/abs/2406.18276)
Comments:
          PhD Thesis. 204 pages, 6 publications

- **What's New**: 이 논문은 산스크리트어에 대한 질문 응답 시스템 개발에서의 도전과 기회에 대해 다룹니다. 이는 자동화된 지식 그래프(knowledge graphs) 구축을 위한 프레임워크, 온톨로지 기반(annotation tools) 도구 및 일반 용도의 작업 도구를 소개하며, 웹 기반 인터페이스, 도구 및 소프트웨어 라이브러리의 다양한 컬렉션을 제공합니다.

- **Technical Details**: 제안된 프레임워크는 산스크리트어 텍스트 분석의 접근성과 정확성을 향상시키며, 지식 표현(knowledge representation) 및 언어 처리(language processing)에서 추가 발전을 위한 길을 열어줍니다. 또한, 온톨로지 기반(onotlogy-driven) 및 일반 용도의 주석 도구(annotation tools)를 도입하여 논문의 기여도를 높였습니다.

- **Performance Highlights**: 연구 결과는 산스크리트어 텍스트에 포함된 풍부한 언어적 정보를 보존하고 이해하며 활용하는 데 기여합니다. 이는 산스크리트어와 관련된 다양한 기술 생태계를 발전시키는 데 중요한 역할을 합니다.



### "Vorbe\c{s}ti Rom\^ane\c{s}te?" A Recipe to Train Powerful Romanian LLMs with English Instructions (https://arxiv.org/abs/2406.18266)
Comments:
          arXiv admin note: text overlap with arXiv:2405.07703

- **What's New**: 최근 몇 년간의 발전을 통해 대형 언어 모델(LLM)은 다양한 작업에서 거의 인간과 유사한 성능을 달성했습니다. 이번 연구에서는 루마니아어에 특화된 LLM을 개발하고 평가하며 공개하는 첫 번째 시도로서, 루마니아어 전용의 교육 데이터, 지침, 벤치마크를 수집하고 번역하여 모델을 훈련 및 평가했습니다. 모든 데이터를 공공 분야에 공개하여 추가 연구를 독려할 계획입니다.

- **Technical Details**: Transformer 아키텍처(Transformer architecture)는 NLP 분야에서 매우 흔해졌으며, 텍스트 분류에서 생성까지 다양한 작업에서 최첨단 성능을 발휘합니다. 교육을 위해 방대한 데이터가 필요하고, 이는 대개 영어 데이터에 집중되어 있습니다. 따라서 이번 연구에서는 루마니아어에 특화된 LLM을 구축하기 위해 대규모 데이터 세트를 수집하고 번역하였습니다. RoLLMs(Romanian Large Language Models)는 인기 있는 오픈 소스 모델인 Llama2, Mistral 및 Gemma 기반으로 개발되었습니다. 학술 기준, 번역된 MT-Bench 및 루마니아의 역사적, 문화적, 사회적 벤치마크를 포함해 다양한 기준에서 평가되었습니다.

- **Performance Highlights**: 루마니아어에 특화된 RoLLMs는 다양한 벤치마크에서 최첨단 성과를 달성했습니다. 이는 기존의 비영어권 LLM들과 비교할 때 루마니아어로 특화된 작업에서 더 나은 성능을 보였으며, 모든 데이터를 공개하여 다른 저자원 언어에 대한 연구도 가능케 했습니다.



### Detecting Machine-Generated Texts: Not Just "AI vs Humans" and Explainability is Complicated (https://arxiv.org/abs/2406.18259)
Comments:
          19 pages, 2 figures

- **What's New**: 이 논문에서는 현재 LLM(거대 언어 모델, Large Language Models) 생성 텍스트 탐지를 인간과 AI를 구별하는 이진 분류(binary classification) 문제로 간주하는 기존 관행에 도전합니다. 새로운 삼진 텍스트 분류(ternary text classification) 체계를 도입하여 '미결정(undecided)' 카테고리를 추가했습니다. 이 카테고리는 탐지 결과를 일반 사용자에게 더 설명 가능하게 만들기 위해 중요합니다. 본 연구는 기계 생성 텍스트를 단순히 분류하는 것에서 설명하는 것으로 패러다임을 전환하며, 탐지기가 사용자에게 명확하고 이해 가능한 설명을 제공할 필요성을 강조합니다.

- **Technical Details**: 네 개의 새로운 데이터셋을 생성하여 다양한 LLM과 인간 저자의 텍스트를 포함했습니다. 새로운 데이터셋을 기반으로 이진 분류 테스트를 수행하여 가장 효과적인 SOTA(State-of-the-Art) 탐지 방법을 확인하고, 탐지하기 어려운 텍스트를 생성할 수 있는 SOTA LLM을 식별했습니다. 두 개의 상위 LLM과 인간 저자가 생성한 텍스트로 새 데이터셋을 구축하고, 세 명의 인간 주석자가 설명 노트와 함께 삼진 레이블을 생성하도록 했습니다. 이 데이터셋을 사용하여 세 가지 상위 SOTA 탐지기가 새로운 삼진 분류 상황에서 어떻게 작동하는지 조사했습니다.

- **Performance Highlights**: 연구 결과는 설명 가능성 관점에서 '미결정' 카테고리가 매우 필요함을 강조합니다. 추가적으로, 세 가지 최고 성능의 탐지기와 인간 주석자들의 설명 노트를 분석하여 기계 생성 텍스트의 설명 가능한 탐지의 복잡성에 대한 통찰력을 제공합니다. 마지막으로, 설명력이 향상된 탐지 시스템을 개발하기 위한 지침을 제안합니다.



### LLaMIPa: An Incremental Discourse Parser (https://arxiv.org/abs/2406.18256)
Comments:
          12 pages, 2 figures

- **What's New**: 이 논문은 대형 언어 모델(LLM)을 세분된 담론 표현 이론(SDRT) 스타일로 주석된 코퍼스에 맞춰 미세 조정(finetuning)한 최초의 담론 구문 분석 실험을 제공합니다. 이 결과로 LLaMIPa(LLaMA Incremental Parser)라는 담론 구문 분석기가 탄생했으며, 이는 담론 문맥을 더 완전하게 활용할 수 있어 기존의 인코더 전용 모델보다 성능이 크게 향상되었습니다.

- **Technical Details**: LLaMIPa는 LLaMA3 모델을 미세 조정하여 개발되었으며, 이전에 추론된 담론 구조를 활용해 새로운 담론 단위(EDU) 간의 링크(link) 및 링크+관계(link+relation) 작업을 동시에 예측할 수 있습니다. SDRT 스타일의 주석이 달린 세 가지 데이터셋(Minecraft Structured Dialogue Corpus, STAC, Molweni)을 사용해 성능을 검증했습니다.

- **Performance Highlights**: LLaMIPa는 현재 최고 성능의 모델들을 크게 능가하는 성능을 보여줍니다. 특히 다자간 대화 및 상황 대화 데이터셋에서 단일 EDU가 두 개의 상위 EDU를 가질 수 있는 상황을 성공적으로 식별함으로써 기존 SDRT 기반 구문 분석기가 처리할 수 없는 구조적인 복잡성을 다룰 수 있습니다.



### Weak Reward Model Transforms Generative Models into Robust Causal Event Extraction Systems (https://arxiv.org/abs/2406.18245)
Comments:
          13 pages, 6 figures, 6 tables

- **What's New**: 최근 발표된 논문에서는 원인과 결과를 추출하는 작업의 본질적 모호함을 해결하고자 합니다. 기존의 평가 방법인 Exact Match와 BertScore는 모델 성능을 제대로 반영하지 못하기 때문에, 인간 평가에 가까운 결과를 제공하기 위해 평가 모델을 학습시켰습니다. 이 모델은 여러 데이터셋에 대해 효과적으로 검증되었으며, 인간 주석 데이터를 줄이기 위한 데이터셋 간 전이 학습도 수행했습니다. 또한, 약한 감독 데이터와 강화 학습 모델로 높은 성능을 달성하는 '약한 감독에서 강한 감독으로' 전환 방법을 제안했습니다.

- **Technical Details**: 이 연구에서는 강화 학습 (Reinforcement Learning, RL)을 통해 원인과 결과 추출 모델을 인간의 선호에 맞추는 방법을 탐구합니다. FLAN-T5 모델과 GPT-3.5 모델을 통해 원인과 결과 데이터를 수집하고, 사람 주석자들이 이를 평가한 후, 평가 결과를 바탕으로 평가 모델을 훈련시켰습니다. 이 평가 모델은 Policy Proximal Optimization (PPO) 알고리즘을 사용해 강화 학습 모델의 행동을 인간의 선호에 맞게 조정하는 데 사용되었습니다. 이를 위해 일부 정확한 주석 데이터 및 약한 감독 데이터를 활용하여 효율적으로 평가 모델을 훈련시켰습니다.

- **Performance Highlights**: 우리의 접근 방식은 PPO 알고리즘을 사용한 강화 학습으로 FLAN-T5 모델을 미세 조정하며, 세 가지 데이터셋에서 평균 4% 향상을 달성했습니다. 인간 평가와의 상관 관계 0.94를 기록하며 높은 퍼포먼스를 보였습니다. 또한, 주석 데이터의 50%만 사용하여도 약한 감독 데이터를 통해 동적 필터링으로 강화 학습 모델을 훈련시켜 전체 모델과 비교할 만한 성능을 얻었습니다.



### Zero-shot prompt-based classification: topic labeling in times of foundation models in German Tweets (https://arxiv.org/abs/2406.18239)
Comments:
          10 pages, 2 tables, 1 figure

- **What's New**: 최근 자연어 처리 기술의 발전으로 인해 훈련 샘플 없이 서면 지침만으로 텍스트 주석을 자동화하는 새로운 도구가 등장했습니다. 이 연구는 이러한 도구를 실제 독일어 트위터 데이터에 사용하여 사회적, 정치적 유럽 위기를 주제로 한 주석 작업에서 평가하였습니다.

- **Technical Details**: 이 연구는 세 가지 NLP 접근 방식을 비교하였습니다: Naive Bayes 분류기, BERT 기반의 파인튜닝/도메인 적응 파이프라인, 그리고 프롬프트 기반의 접근 방식입니다. 프롬프트 기반 접근 방식은 모델 선택 중 로컬 컴퓨팅 자원에 의해 제한됨에도 불구하고, 파인튜닝된 BERT와 비교할 만한 성능을 보여주었습니다. 특히, 이 접근 방식은 주석 지침서만을 사용하여 훈련 데이터를 제공하지 않고도 주석 작업을 수행할 수 있었습니다.

- **Performance Highlights**: 프롬프트 기반 접근 방식은 파인튜닝된 BERT와 유사한 성능을 보였으며, 사전 레이블된 훈련 데이터를 필요로 하지 않았습니다. 이는 기존의 NLP 작업 방식의 패러다임 변화를 강조하며, 다운스트림 작업의 통합성과 사전 레이블된 훈련 데이터의 필요성을 없앨 가능성을 시사합니다.



### Enhancing Data Privacy in Large Language Models through Private Association Editing (https://arxiv.org/abs/2406.18221)
- **What's New**: 이번 연구에서는 Private Association Editing (PAE)라는 혁신적인 방어 접근법을 제안합니다. PAE는 대형 언어 모델(Large Language Models, LLMs)에서 개인 식별 정보(PII)를 효과적으로 제거할 수 있으며, 재훈련 없이 이를 수행할 수 있습니다. PAE는 네 단계 절차로 구성되며, 이는 PII 감지, PAE 카드 적용, 타겟 데이터 추출 공격(TDE) 저항 검증, 편집 후의 일관성 확인을 포함합니다.

- **Technical Details**: PAE는 모델의 파라미터를 조정하여 PII를 제거하며, 모델 편집을 통해 개인 정보와 관련된 연관성을 깨트립니다. 구체적으로, 모델 내에서 개인 정보를 익명화하고, 원래 정보를 의미상 동등한 마스크된 정보로 대체하는 방식을 사용합니다. 실험에서는 GPT-J 모델을 이용해 TDE 공격 전후를 비교하며, PAE가 개인 정보 유출을 줄이는 데 효과적임을 입증했습니다.

- **Performance Highlights**: 실험 결과, PAE는 개인 정보 유출을 효과적으로 줄이면서 모델의 텍스트 생성 성능을 유지하였습니다. 모델의 재훈련없이 PAE가 효율적인 방어 전략임을 확인했습니다. 이는 LLMs에서 개인 정보 보호를 위한 중요한 도구가 되리라 기대됩니다.



### A Closer Look into Mixture-of-Experts in Large Language Models (https://arxiv.org/abs/2406.18219)
- **What's New**: 새로운 논문에서는 MoE(Mixture-of-Experts) 모델의 내부 작동 방식에 관한 초기 연구를 다뤘습니다. 특히 MoE 기반 대형 언어 모델이 어떻게 작동하는지 이해하기 위해 세 가지 최신 MoE 기반 모델인 Mixtral 8x7B, DeepSeekMoE, Grok-1을 종합적으로 분석하였습니다.

- **Technical Details**: MoE 모델은 각 토큰마다 일부 파라미터만 활성화하여 모델 크기를 늘리면서도 계산 효율성을 유지합니다. 논문에서는 FFN(Feed-Forward Network) 계층의 뉴런이 세밀한 전문가처럼 작동한다는 점, 라우터(router)가 주로 출력 노름(norm)이 큰 전문가를 선택한다는 점, 레이어가 증가함에 따라 전문가의 다양성이 증가하지만 마지막 레이어는 예외라는 점을 밝혔습니다. 라우터 설계와 전문가 할당에 대한 여러 제안도 제시하고 있습니다.

- **Performance Highlights**: Mixtral 8x7B와 DeepSeekMoE에서는 게이트가 주로 출력 노름이 큰 전문가를 선택하는 것을 발견했습니다. 또한, 깊은 레이어의 전문가 파라미터와 출력의 유사성이 레이어 수가 늘어날수록 감소하다가 마지막 레이어에서 갑자기 증가하는 현상을 관찰했습니다.



### SEED: Accelerating Reasoning Tree Construction via Scheduled Speculative Decoding (https://arxiv.org/abs/2406.18200)
- **What's New**: 이번 연구는 고유의 추론 지연 문제를 해결하기 위해 SeeD라는 새로운 고속 추론 프레임워크를 소개합니다. 이 방법은 'scheduled speculative execution'을 채택하여 런타임 속도와 GPU 메모리 관리를 최적화합니다.

- **Technical Details**: SeeD는 주기적으로 스케줄된 전략을 통해 다양한 단계별 사고 생성을 효율적으로 처리하며, 상태 평가를 위해 여러 반복을 처리합니다. 또한, 체인 오브 사고 촉구보다 더 복잡한 사고 경로의 탐색을 허용하는 'scheduled speculative execution'기법을 사용합니다.

- **Performance Highlights**: 세 가지 추론 데이터셋에서의 광범위한 실험 평가를 통해 SeeD의 속도 향상을 입증하였습니다. 이를 통해 학습 없는 speculative decoding(추측적 디코딩)을 위한 일괄 추론에 대한 실용적인 경로를 제공합니다.



### Methodology of Adapting Large English Language Models for Specific Cultural Contexts (https://arxiv.org/abs/2406.18192)
Comments:
          11 pages, 2 figures

- **What's New**: 대규모 언어 모델(LLMs)의 급격한 성장은 인공지능 분야에서 두드러진 추세입니다. 그러나 최신 LLM들은 주로 영어 기반으로 특정 문화적 도메인에 적용할 때 한계에 직면합니다. 본 논문에서는 특정 문화적 맥락에서 대규모 모델을 신속하게 적응시키는 방법을 제안합니다. 이 방법은 특정 문화적 지식과 안전 가치 데이터에 기반한 instruction-tuning(학습 지침 조정)을 활용합니다. 이를 중국 문화에 적용하고, 실험적 영어 LLM으로 LLaMA3-8B를 활용하여 평가 결과, 적응된 LLM이 도메인별 지식과 안전 가치 적응성에서 현저한 향상을 보였음을 확인했습니다.

- **Technical Details**: 제안된 방법론은 3단계로 구성됩니다: 1. 학습 지침 데이터 수집, 2. 지식 및 역량 강화, 3. 안전 가치 정렬. 먼저, 특정 문화적 맥락의 지식과 안전 가치에 관련된 학습 지침 데이터를 수집합니다. 다음으로, 해당 LLM의 지식과 역량을 평가하고, 부족한 부분이 발견되면 목표 지식 및 역량 강화를 실행합니다. 마지막으로, 적응된 모델의 안전 가치를 평가하고 개선합니다. 특히, instruction-tuning은 대규모 모델을 신속하게 정렬할 수 있어, 시간과 자원을 절감하는 데 유리합니다.

- **Performance Highlights**: 제안된 방법론은 영어 LLM의 원래 우수한 전문 지식을 유지하면서도, 특정 문화적 지식과 안전 가치에 신속하게 적응할 수 있게 합니다. 이를 통해 다문화적 사용자 상호작용을 제공하며, 모델의 응용 범위와 적응력을 확장시킵니다.



### Selective Prompting Tuning for Personalized Conversations with LLMs (https://arxiv.org/abs/2406.18187)
Comments:
          Accepted to ACL 2024 findings

- **What's New**: 새로운 Selective Prompt Tuning (SPT) 기법을 통해 대화형 인공지능에서 개인화된 대화를 개선했습니다. SPT는 소프트 프롬프트(soft prompt)를 기반으로 한 선택적 튜닝(selection)을 통해 다양한 대화 설정에 맞는 적절한 소프트 프롬프트를 동적으로 선택합니다.

- **Technical Details**: SPT 모델은 초기화된 여러 소프트 프롬프트를 사용하며, 학습 가능한 밀집 검색기(dense retriever)가 다양한 입력 컨텍스트에 따라 적절한 소프트 프롬프트를 선택합니다. 또한, 피드백을 통해 검색기를 동적으로 업데이트합니다. 더불어, 컨텍스트-프롬프트 대조 학습(context-prompt contrastive learning)과 프롬프트 융합 학습(prompt fusion learning)을 도입하여 대화의 다양한성을 높였습니다.

- **Performance Highlights**: CONVAI2 데이터셋 실험 결과, SPT는 응답 다양성을 최대 90%까지 개선하였고, 다른 중요한 성능 지표에서도 향상을 나타냈습니다. 특히, SPT는 모델의 전략적 프롬프트 선택을 통해 다양한 대화 시나리오에서 탁월한 성능을 발휘했습니다.



### UIO-LLMs: Unbiased Incremental Optimization for Long-Context LLMs (https://arxiv.org/abs/2406.18173)
- **What's New**: 이번 연구는 길어지는 문맥에서도 성능을 유지할 수 있는 메모리 강화 트랜스포머(Memory-Enhanced Transformers)를 소개합니다. UIO-LLMs라는 새로운 방법론을 제시하며, 이는 무편향 증분 최적화(Unbiased Incremental Optimization)를 통해 긴 문맥 설정에서의 성능을 극대화합니다. Llama2-7b-chat 모델의 문맥 크기를 4K 토큰에서 100K 토큰으로 확장하는 데 성공했으며, 추가 매개변수가 적고 계산 비용도 선형적으로 유지됩니다.

- **Technical Details**: 이 연구는 트랜스포머와 RNN의 장점을 결합한 메모리 강화 트랜스포머를 제안합니다. 완전 연결된 RNN으로 구성된 메모리 세그먼트를 통해 이 모델은 Truncated Backpropagation Through Time (TBPTT) 알고리즘을 사용해 최적화를 진행합니다. 이를 통해 시간 복잡성을 줄이고 그라디언트 계산의 편향 문제를 해결합니다. 이 모델은 LoRA를 활용해 매개변수 양을 최소화하면서도 효과적인 훈련을 가능케 합니다.

- **Performance Highlights**: UIO-LLMs는 기존의 메모리 강화 트랜스포머 모델(RMT, AutoCompressor, Gist Tokens 등)을 뛰어넘는 성능을 보입니다. 특히 QA 및 요약 작업에서 AutoCompressor를 능가하며, 긴 텍스트 생성의 품질도 유지됩니다. Activation Beacon보다 훈련 가능한 매개변수를 줄이고 병렬 압축을 가능하게 하며, 훈련 비용 또한 줄어듭니다.



### NeBuLa: A discourse aware Minecraft Builder (https://arxiv.org/abs/2406.18164)
Comments:
          10 pages, 3 figures

- **What's New**: 이 논문은 대화 상황에서 비언어적 환경의 사전 담화와 비언어적 문맥을 통합함으로써 'language to action' 구성 요소를 향상시키는 방법을 제시합니다. 특히 NeBuLa라는 모델을 통해, 이전의 Jayannavar et al.(2020)보다 두 배 이상의 net-action F1 점수를 달성하였습니다.

- **Technical Details**: NeBuLa 모델은 대규모 언어 모델(LLM)을 미세 조정하여 사전 문맥에 기반한 행동을 예측합니다. 이 모델은 Minecraft Dialogue Corpus(MDC)와 Minecraft Structured Dialogue Dataset(MSDC)를 사용하여 학습되었습니다. 모델의 주된 향상은 대화의 언어적 및 비언어적 문맥을 고려하여 지시사항을 이해하는 것입니다.

- **Performance Highlights**: NeBuLa 모델은 이전의 Neural Builder 모델보다 두 배 이상의 net-action F1 점수를 달성하였으며, 기본 형태를 구축하고 이를 적절히 배치하는 작업에서 높은 정확도를 보였습니다. 또한, underspecified instructions의 평가 측정을 개선하여 보다 현실적인 평가 기준을 제시했습니다.



### LOOK-M: Look-Once Optimization in KV Cache for Efficient Multimodal Long-Context Inferenc (https://arxiv.org/abs/2406.18139)
- **What's New**: 이번 연구에서는 LOOK-M이라는 새로운 방법을 소개합니다. 이는 멀티모달 긴 컨텍스트(Look-Once Key-Value Memory) 시나리오에서 KV 캐시를 효율적으로 압축하는 첫 번째 시도입니다. 모델이 프롬프트(prefill) 과정 동안 텍스트 속성을 우선시한다는 관찰에 기반하여, 텍스트 중심의 방법을 사용해 KV 캐시를 압축하고 성능을 유지합니다.

- **Technical Details**: LOOK-M은 멀티모달 입력에서 중요한 텍스트 KV 쌍을 유지하면서 시각적으로 중요도가 낮은 KV 쌍을 제거합니다. 또한, 제거된 KV 쌍들을 보존된 KV 쌍들과 병합하여 총체적인 문맥 정보를 유지합니다. 이는 fine-tuning 없이 plug-and-play 방식으로 적용할 수 있습니다. 다수의 최신 MLLM 백본(LLaVA-v1.5-7B/13B, MobileVLM-v2, InternVL-v1.5)과 다양한 멀티모달 긴 문맥 작업에서 테스트했습니다.

- **Performance Highlights**: LOOK-M은 KV 캐시 메모리 사용량을 80%에서 최대 95%까지 줄이면서도 성능을 유지하거나 향상시켰습니다. 또한, 디코딩 속도는 1.3배에서 1.5배 빨라졌습니다. 이러한 결과는 텍스트 중심의 KV 캐시 압축 전략과 병합 전략의 효과를 입증합니다.



### Automatic Speech Recognition for Hind (https://arxiv.org/abs/2406.18135)
- **What's New**: 이번 연구는 음성 인식(Auto Speech Recognition, ASR) 기술을 개선하기 위해 새로운 웹 애플리케이션과 웹 인터페이스를 개발했습니다. 이 애플리케이션은 큰 규모의 오디오 파일과 그에 대한 전사를 관리하며, 실시간으로 클라이언트-서버 아키텍처(client-server architecture)를 통해 작동합니다. 이를 통해 ASR 전사에 대한 인간의 협력적인 수정이 가능합니다.

- **Technical Details**: 이 웹 애플리케이션은 JavaScript와 Node.js로 개발되었으며, 음성 인식을 위한 웹 인터페이스는 VAD(Voice Activity Detection)을 사용하여 16kHz 모노 오디오를 기록합니다. VAD는 인간의 음성 존재를 감지하여 불필요한 처리를 줄여주며, VoIP 애플리케이션에서 계산 및 네트워크 대역폭을 절약합니다. 최종 단계에서는 신경망(neural network)을 테스트하여 음성 신호를 HMM(Hidden Markov Model) 상태에 정확하게 정렬했습니다. 이를 위해 노드의 공동 활성화 사전 통계를 이용한 새로운 역전파 방법을 구현했습니다.

- **Performance Highlights**: 실제 테스트 결과, 이 연구에서 개발한 웹 애플리케이션과 웹 인터페이스는 실시간 음성 인식 및 처리에 있어 높은 효율성을 보여주었으며, 협력적인 수정 기능을 통해 ASR 전사 품질이 크게 향상되었습니다. 또한, VAD를 통한 효율적인 음성 처리와 자원 절약 측면에서도 긍정적인 결과를 얻었습니다.



### Assessing "Implicit" Retrieval Robustness of Large Language Models (https://arxiv.org/abs/2406.18134)
- **What's New**: 이 논문에서는 외부 지식을 활용하여 대형 언어 모델(Large Language Models, LLMs)을 강화하는 Retrieval-augmented generation이 주목받고 있는 이유와 문제점을 다룹니다. 기존의 방법은 모델의 검색 강건성(retrieval robustness)에 크게 의존하게 되며, 검색된 문맥이 부적절한 경우 성능 저하가 발생할 수 있습니다. 이 연구에서는 여러 LLM 모델이 검색된 문맥의 적절성을 판단하지 않고 최종 답변을 출력하도록 지시하여 '암묵적' 검색 강건성을 평가합니다.

- **Technical Details**: 모델의 검색 정확성을 유지하면서 검색 부정확성에 대한 강건성을 크게 향상시키기 위해, 금(gold) 문맥과 방해(distracting) 문맥을 혼합한 데이터로 파인튜닝(fine-tuning)을 수행했습니다. 이를 통해 모델이 최종 답변의 감독만으로, 엔드 투 엔드(end-to-end) 방식으로 검색된 문맥의 적절 여부를 암묵적으로 처리할 수 있음을 시사합니다. 추가적인 적절성 판단(explicit relevance judgment) 과정은 불필요하며 오히려 엔드 투 엔드 접근을 방해할 수 있습니다.

- **Performance Highlights**: 검색된 문맥이 정확할 때 올바른 답을 추출하는 능력을 유지하면서, 검색 부정확성에도 강건한 성능을 보여줍니다. 이는 LLM이 맞춤형 데이터로 학습했을 때, 검색된 문맥의 적절 여부를 암묵적으로 처리할 수 있는 능력이 향상된다는 것을 보여줍니다.



### ConvoCache: Smart Re-Use of Chatbot Responses (https://arxiv.org/abs/2406.18133)
Comments:
          Accepted to appear at Interspeech 2024

- **What's New**: 최근 arXiv에 게시된 논문에서 ConvoCache라는 새로운 대화 캐싱 시스템을 소개했습니다. 이 시스템은 대화형 챗봇에서 발생하는 느린 응답 속도와 높은 비용 문제를 해결합니다. ConvoCache는 이전에 생성된 응답 중 의미상 유사한 프롬프트를 찾아서 재사용합니다.

- **Technical Details**: ConvoCache는 대화 기록(D)을 가지고 여러 응답 후보를 생성한 다음, 자동 대화 평가 모델인 UniEval을 사용하여 품질을 평가합니다. 유사성 모델로 SimCSE나 AnglE를 사용하며, 응답의 일관성을 평가합니다. 일관성이 90% 이상인 경우 캐시에서 응답을 제공합니다. 만약 일치하는 응답이 없으면 새로운 응답을 생성하여 캐시에 저장합니다.

- **Performance Highlights**: DailyDialog 데이터셋을 사용한 실험에서 ConvoCache는 89%의 요청에 대해 캐시된 응답을 사용할 수 있었고, 평균 지연 시간은 214ms에 불과했습니다. 이는 LLM과 음성 합성에 비해 훨씬 빠른 응답 속도입니다. 프리페칭(prefetching)은 지연 시간을 더 줄이기 위한 방법으로 테스트되었으나, 응답 일관성을 약간 떨어뜨렸습니다. ConvoCache는 챗봇 사용 비용을 최대 89%까지 줄일 수 있습니다.



### ResumeAtlas: Revisiting Resume Classification with Large-Scale Datasets and Large Language Models (https://arxiv.org/abs/2406.18125)
Comments:
          8 pages, 6 figures, 1 table, 6th International Conference on AI in Computational Linguistics

- **What's New**: 이 논문에서는 온라인 채용 플랫폼의 증가와 AI 기술의 도입으로 인해 효율적인 이력서 분류 방법이 필요한 상황에서, 이를 개선하기 위해 큰 규모의 이력서 데이터셋(13,389개 이력서)을 구축하고, BERT와 Gemma 1.1 2B와 같은 대형 언어 모델(Large Language Models, LLMs)을 사용한 새로운 접근 방식을 제시합니다. 제안된 모델은 전통적인 머신러닝 방법보다 뛰어난 성능을 보였습니다.

- **Technical Details**: 이 연구는 다양한 출처에서 수집한 13,389개의 이력서를 포함하는 대규모 데이터셋을 구축하고, 고급 데이터 전처리 기법을 사용했습니다. 이력서 분류에 사용된 주요 모델로는 BERT와 Gemma 1.1 2B와 같은 최신 LLMs가 있습니다. 이러한 모델은 과거의 전통적인 기계 학습 알고리즘, 예를 들면 Naïve Bayes, Support Vector Machine (SVM), Random Forest, TF-IDF, XGB 알고리즘 등이 수행하기 어려운 복잡한 분류 작업도 효과적으로 처리할 수 있습니다.

- **Performance Highlights**: 제안된 모델은 top-1 정확도 92%와 top-5 정확도 97.5%를 달성하였습니다. 이는 전통적인 머신러닝 방법보다 뛰어난 성능을 나타내며, 데이터셋의 품질과 모델 아키텍처의 중요성을 강조합니다. 이러한 결과는 온라인 채용 방식의 향상에 중요한 기여를 할 것입니다.



### Poisoned LangChain: Jailbreak LLMs by LangChain (https://arxiv.org/abs/2406.18122)
Comments:
          6 pages,2 figures,This paper is a submission to ACM TURC. It has been accepted by the editor of the organizer

- **What's New**: 최근 자연어 처리(NLP) 기술의 발전으로 대형 언어 모델(LLMs)이 점점 더 널리 사용되고 있습니다. 이러한 모델의 보안 취약성에 대한 대중의 우려가 증가하고 있으며, 이에 따라 LLMs의 보안 중요성이 대두되고 있습니다. 현재 주요 공격 기법 중 하나는 'jailbreak attack'으로, 이는 모델의 안전 메커니즘을 우회하여 부적절한 콘텐츠를 생성하게 만드는 공격입니다. 본 논문은 최초로 '간접 jailbreak' 개념을 제안하며 LangChain을 이용해 Retrieval-Augmented Generation(RAG)을 수행하는 방법을 제시합니다.

- **Technical Details**: RAG는 외부 지식 베이스를 활용해 모델의 새로운 지식 부족을 보완하는 기술입니다. LangChain을 이용해 구축된 새로운 간접 jailbreak 공격 방법인 Poisoned-LangChain(PLC)은 독이 든 외부 지식 베이스를 사용해 대형 언어 모델과 상호작용함으로써 악의적이고 비규범적인 대화를 생성하게 합니다. 이 방법은 키워드 트리거와 유도 프롬프트를 설계하고 특정 독성 지식 베이스를 만들어 검열을 피하는 방식으로 이루어집니다. 우리는 다양한 시나리오에서 여섯 가지 언어 모델을 테스트하여 높은 성공률(88.56%, 79.04%, 82.69%)을 달성했습니다.

- **Performance Highlights**: Poisoned-LangChain(PLC) 방법은 세 가지 주요 시나리오에서 간접 jailbreak 공격을 성공적으로 수행했으며, 성공률은 각각 88.56%, 79.04%, 82.69%에 달합니다. 또한, 이 연구는 중국어 대형 언어 모델을 대상으로 진행되었으며, 최신 중국어 언어 모델에서도 유효성을 입증하였습니다.



### ArzEn-LLM: Code-Switched Egyptian Arabic-English Translation and Speech Recognition Using LLMs (https://arxiv.org/abs/2406.18120)
Comments:
          9 pages, 4 figures, 5 tables, 6th International Conference on AI in Computational Linguistics

- **What's New**: 이 논문은 최근에 이집트 아랍어와 영어의 코드 스위칭(code-switching) 현상이 증가하는 것에 동기 부여되어, 코드 스위칭된 이집트 아랍어-영어를 영어 또는 이집트 아랍어로 번역하는 기계 번역(MT) 및 자동 음성 인식(ASR) 시스템의 복잡성을 탐구합니다. 저자들은 LLama와 Gemma 같은 대형 언어 모델을 사용하여 이러한 시스템들을 개발했으며, Whisper 모델을 활용한 코드 스위칭된 이집트 아랍어 인식을 중점적으로 다룹니다.

- **Technical Details**: 이 논문에서는 기계 번역(MT)과 자동 음성 인식(ASR) 시스템을 개발하기 위한 데이터 전처리, 훈련 기법 등을 포함한 실험 절차를 자세히 설명합니다. ASR에서는 Whisper 모델을 사용하고, 이어서 MT 시스템과 통합하여 연속적인 음성-텍스트 번역 시스템을 구현합니다. 이 시스템은 제한된 자원과 이집트 아랍어 방언의 독특한 특성을 극복하는 것을 목표로 합니다.

- **Performance Highlights**: 평가 결과, 제안된 시스템이 영어 번역에서 가장 최신 기술 대비 56% 개선된 성능을 보였고, 아랍어 번역에서는 9.3% 개선된 성능을 보였습니다. 코드 스위칭이 구어체에서 깊이 내재된 현상이기 때문에, ASR 시스템이 이 현상을 효과적으로 처리하는 능력은 다양한 도메인에서 원활한 상호 작용을 가능하게 합니다.

- **Open-Sourcing**: 모델 및 코드는 오픈 소스로 공개되어 있으며, 논문은 커뮤니티 참여와 추가 연구를 촉진하기 위해 제공됩니다.



### BADGE: BADminton report Generation and Evaluation with LLM (https://arxiv.org/abs/2406.18116)
Comments:
          Accepted by IJCAI 2024 Workshop: The 2nd International Workshop on Intelligent Technologies for Precision Sports Science (IT4PSS)

- **What's New**: 이번 연구에서는 배드민턴 경기를 자동으로 요약하고 평가할 수 있는 새로운 프레임워크 BADGE를 소개합니다. 이 프레임워크는 대형 언어 모델(Large Language Model, LLM)을 활용하여 경기 보고서를 생성하고 평가하는 두 가지 단계로 구성되어 있습니다. 특히, GPT-4 모델을 이용한 체인 오브 포트(Chain of Thought) 입력 방식과 CSV 데이터 형식을 사용한 보고서 생성에서 높은 성능을 보였습니다.

- **Technical Details**: BADGE 프레임워크는 두 단계로 나뉩니다. 첫째, 배드민턴 관련 데이터를 LLM에 입력하여 상세한 경기 보고서를 생성합니다. 이 과정에서 다양한 입력 데이터 유형 및 인컨텍스트 학습(In-Context Learning)을 시험한 결과, structured data 유형인 CSV와 체인 오브 포트 방식에서 GPT-4 모델이 최상의 성능을 나타냈습니다. 둘째, 생성된 보고서를 LLM이 평가하여 품질 점수를 부여합니다. 이 평가 결과는 인간 판정과 비교하여 LLM에 의해 생성된 보고서가 더 선호되는 경향을 확인했습니다.

- **Performance Highlights**: 다양한 인컨텍스트 학습 방식 및 입력 데이터 유형을 비교한 결과, GPT-4와 CSV, 체인 오브 포트 방식을 조합한 경우에 가장 높은 보고서 품질을 달성했습니다. 평가 단계에서는 G-Eval 프레임워크를 이용하여 생성된 보고서를 체계적으로 평가하였으며, 그 결과는 인간 심사 기준과 잘 일치하는 경향을 보였습니다.



### Token-Weighted RNN-T for Learning from Flawed Data (https://arxiv.org/abs/2406.18108)
- **What's New**: 이 논문에서는 RNN-T 훈련 목적 함수에 새로운 토큰 가중치(token-weighted) 기법을 제안합니다. 이 기법은 타깃 시퀀스의 각 토큰에 특정 가중치를 부여하여 훈련 목표를 개선합니다. 특히, 훈련 데이터에 포함된 오류를 완화하는 데 초점을 맞추어, 인간 주석 오류와 pseudo-labeling에서 발생하는 오류를 줄이는 데 중점을 둡니다.

- **Technical Details**: RNN-T 모델은 주로 타깃 시퀀스의 확률을 증가시키기 위해 cross-entropy 목적 함수로 훈련됩니다. 이 연구에서는 토큰별 가중치를 부여하는 새로운 RNN-T 목표 함수를 제안합니다. 이 방법은 각 토큰의 중요도에 따라 가중치를 부여함으로써 훈련 데이터에 포함된 오류로 인한 정확도 손실을 줄이고자 합니다. pseudo-labeling에서는 초기 모델의 낮은 품질로 인해 많은 오류가 포함될 수 있으며, 이러한 오류를 줄이기 위해 교사 모델의 토큰별 신뢰도 점수를 학생 모델의 훈련에 사용합니다.

- **Performance Highlights**: 이 연구에서는 pseudo-labeling을 사용한 준지도 학습(semi-supervised learning)에서 이 새로운 방법을 사용하면 최대 38%의 상대적 정확도 향상을 달성할 수 있음을 입증했습니다. 또한, 인간 주석 오류 시나리오를 시뮬레이션하여 토큰 가중치 RNN-T가 정확도 손실을 64%에서 99%까지 회복할 수 있음을 보여주었습니다.



### Shimo Lab at "Discharge Me!": Discharge Summarization by Prompt-Driven Concatenation of Electronic Health Record Sections (https://arxiv.org/abs/2406.18094)
Comments:
          BioNLP @ ACL2024

- **What's New**: BioNLP 워크숍 2024에서 'Discharge Me!'라는 새로운 공동 작업이 소개되었습니다. 이 작업의 주요 목표는 전자의무기록(EHR)을 이용하여 '입원 경과 보고서'와 '퇴원 지침서'를 자동으로 생성함으로써 임상의가 상세한 기록을 작성하는 시간을 줄이는 것입니다.

- **Technical Details**: 우리의 접근 방식은 EHR에서 관련 섹션을 추출한 후, 설명 프롬프트를 추가하고 개별 토큰과 함께 연결하여 입력 텍스트를 생성하는 작업으로 시작합니다. 텍스트 생성 모델을 훈련하기 위해 ClinicalT5-large 모델을 LoRA로 파인 튜닝합니다. EHR의 노이즈를 제거하고, 각 대상에 대해 중요한 섹션을 선택한 후, <sep> 토큰을 사용해 연결된 입력 텍스트로 준비합니다.

- **Performance Highlights**: 최종 테스트 데이터에서 우리 접근 방식은 ROUGE-1 점수 0.394를 달성하여, 상위 솔루션들과 비교할 만한 성과를 보였습니다.



### LLM-Driven Multimodal Opinion Expression Identification (https://arxiv.org/abs/2406.18088)
Comments:
          6 pages, 3 Figures

- **What's New**: 이번 연구에서는 기존의 의견 표현 식별(Opinion Expression Identification, OEI) 작업을 멀티모달(multimodal) 입력으로 확장하여 문자(text)와 음성(speech)을 통합하는 새로운 멀티모달 OEI(MOEI) 작업을 제안합니다. 이를 위해 CMU MOSEI와 IEMOCAP 데이터셋을 사용하여 CI-MOEI 데이터셋을 구성하고, MPQA 데이터셋에는 Text-to-Speech(TTS) 기술을 적용하여 CIM-MOEI 데이터셋을 완성합니다. 또한, 대형 언어 모델(LLMs)을 활용한 새로운 방법(STOEI)을 제안하여 문자와 음성 모드를 결합하여 의견 표현을 식별합니다.

- **Technical Details**: 연구에서는 CMU MOSEI와 IEMOCAP 데이터셋을 사용하여 텍스트와 음성이 완벽하게 일치하지 않는 현실적인 시나리오를 반영한 CI-MOEI 데이터셋을 구축했습니다. MPQA 데이터셋에는 TTS 기술을 적용하여 합성 음성을 생성하고, 이를 기존 데이터셋에 통합하여 CIM-MOEI 데이터셋을 완성했습니다. 또한 LLMs 기반의 출력 템플릿을 설계하여 의견 표현을 동시에 출력하도록 개선했습니다. 최종적으로, LLM 인코더, 스피치 인코더, 모달리티 어댑터, LLM을 결합한 종합적인 구조를 제시하는 STOEI 방법을 도입했습니다.

- **Performance Highlights**: 실험 결과, MOEI를 사용하면 전통적인 문자 단독 입력 방식을 사용할 때보다 성능이 크게 향상되었으며, 제안된 방법은 기존의 멀티모달 기술을 9.20% 초과하여 최첨단(State-of-the-Art, SOTA) 결과를 달성했습니다. 코드와 데이터셋은 연구의 미래 발전을 위해 공개될 예정입니다.



### Multilingual Knowledge Graph Completion from Pretrained Language Models with Knowledge Constraints (https://arxiv.org/abs/2406.18085)
Comments:
          11 pages, ACL 2023

- **What's New**: 이 논문은 다국어 지식 그래프 완성(Multilingual Knowledge Graph Completion, mKGC)을 위한 새롭고 효과적인 방법을 제안합니다. 기존의 다국어 사전 훈련된 언어 모델(PLM)을 활용하면서도 발생하는 문제점을 개선하기 위해, 전역(Global) 및 지역(Local) 지식 제약을 도입했습니다. 이는 사전 학습된 모델을 mKGC 작업에 더 잘 적응시키고 저자원(low-resource) 언어에서도 성능을 크게 향상시킵니다.

- **Technical Details**: 이 방법은 답변 엔티티의 추론을 제한하기 위해 전역 지식 제약을 사용하고, 쿼리 문맥의 표현을 강화하기 위해 지역 지식 제약을 사용합니다. 구체적으로는, 전역 지식은 엔티티와 관계 표현 간의 관계를 기반으로 답변의 유형을 제한합니다. 반면 지역 지식은 쿼리 내 하위 토큰 사이의 상호 연결성을 이해하는 능력을 강화합니다. 이를 위해 [H], [R], [T]와 같은 특수 토큰을 도입하고, 현재 트리플에 대한 적합성을 측정하는 스코어링 함수와 서로 다른 분포 간의 상호 정보를 최대화하는 추정기를 사용합니다.

- **Performance Highlights**: 제안된 방법은 Hits@1과 Hits@10 메트릭에서 각각 12.32% 및 16.03%의 평균 향상을 보이며, 기존 SOTA 방법을 능가함을 입증했습니다. 이는 다수의 공공 데이터셋에서 실험된 결과로, 제안된 방법이 mKGC 작업에서 상당한 성능 향상을 가져왔음을 의미합니다. 또한 다국어 지식 그래프와 교차 언어 엔티티 정렬 작업에서도 우수한 성능을 보였습니다.



### Octo-planner: On-device Language Model for Planner-Action Agents (https://arxiv.org/abs/2406.18082)
- **What's New**: 이 논문에서는 AI 에이전트의 효율적이고 자율적인 계획 및 실행을 위한 새로운 온디바이스 Planner-Action 프레임워크를 소개합니다. 이 프레임워크는 계획 기능과 실행 기능을 각각 분리하여 엣지 디바이스용으로 최적화된 Phi-3 Mini (3.8 billion parameter LLM)와 Octopus 모델을 사용합니다. 이 접근 방식은 리소스가 제한된 장치에서 성능을 최적화하기 위해 모델 미세 조정을 채택하여 응답 시간을 개선하고 계산 비용과 에너지 소비를 줄입니다.

- **Technical Details**: 프레임워크는 GPT-4를 사용하여 계획 데이터를 생성하고 검증하여 커널 고유의 데이터셋을 만든 다음, Phi-3 Mini 모델을 미세 조정(cleaned dataset)을 통해 온디바이스 배포를 목표로 합니다. 다중 도메인 계획 문제를 해결하기 위해, 우리는 다양한 기능 세트에 대해 LoRA를 훈련시킨 후 이를 결합하는 multi-LoRA 훈련 방법을 개발했습니다. 이는 복잡한 다중 도메인 쿼리를 유연하게 처리하면서 계산 효율성을 유지합니다.

- **Performance Highlights**: 우리의 프레임워크는 도메인 내 테스트 환경에서 97% 성공률을 달성했습니다. 또한 액션 모델인 Octopus V2는 함수 호출에서 95% 이상의 정확도를 보여줍니다. 전체 시스템은 단순한 작업에 대해 미리 정의된 기능을 우선적으로 사용함으로써 AI 에이전트를 더 실용적이고 접근 가능하게 만들어, 실생활 응용 프로그램에서 비용 효율성을 높입니다.



### Self-Training with Pseudo-Label Scorer for Aspect Sentiment Quad Prediction (https://arxiv.org/abs/2406.18078)
Comments:
          Accepted to ACL 2024 Main Conference

- **What's New**: 이번 연구는 Aspect Sentiment Quad Prediction (ASQP) 작업에서 부족한 라벨 데이터 문제를 해결하기 위해 새로운 자기 훈련 프레임워크를 제안합니다. 이 프레임워크는 가상 라벨 평가기를 도입하여 리뷰와 가상 라벨 간의 일치를 점검하고 부정확한 라벨을 필터링함으로써 자기 훈련의 효과를 향상시킵니다. 특히, 인간 주석자가 생성한 비교 데이터셋을 사용해 평가기의 성능을 극대화하려는 시도를 포함하고 있습니다.

- **Technical Details**: 이 프레임워크는 가상 라벨을 평가하기 위해 주어진 리뷰와 일치하는지 확인하는 점수 시스템을 도입합니다. 평가기의 성능을 보장하기 위해 두 가지 주요 요소가 강조됩니다: 1) 훈련 데이터셋의 품질과 2) 모델 아키텍처 및 목표. 인간 주석자가 만든 비교 데이터셋으로 생성 모델을 훈련하여, 기존의 규칙 기반 데이터셋보다 더 복잡하고 인간의 판단에 가까운 데이터를 확보했습니다. 또한, 평가기를 순차적인 토큰 단위로 가상 라벨의 타당성을 평가할 수 있도록 Conditional Likelihoods 방식을 사용하여 개선했습니다.

- **Performance Highlights**: 공개된 ASQP 데이터셋에서 광범위한 실험을 통해 이 접근 방식이 자기 훈련의 효과를 대폭 향상시킬 수 있음을 검증했습니다. 특히, 비교 데이터 주석 작업에서 인간을 대체할 수 있는 대형 언어 모델의 가능성을 시도했으며, 그 타당성을 입증했습니다. 최종적으로, 가상 라벨 평가기를 다중 후보 라벨의 재랭커(reranker)로 사용하여 데이터 품질을 향상시킨 방법도 탐구했습니다.



### Exploring Energy-Based Models for Out-of-Distribution Detection in Dialect Identification (https://arxiv.org/abs/2406.18067)
- **What's New**: 이 연구는 방언 OOD 검출을 위한 새로운 여백 강화 공동 에너지 모델(MEJEM)을 소개합니다. 이 모델은 생성을 위한 모델과 에너지 여백 손실을 통합하여 방언 식별 시스템의 강건성을 향상시키려는 목표를 가지고 있습니다. 또한 두 가지 OOD 점수를 탐구하여 에너지 점수가 소프트맥스 점수보다 뛰어남을 입증했습니다. Sharpness-Aware Minimization을 활용해 모델 학습을 최적화함으로써 손실과 예리함을 최소화하여 모델의 일반화를 향상시켰습니다.

- **Technical Details**: MEJEM은 차별적 분류 모델, 에너지 기반 여백 손실 및 생성 모델을 통합한 방식으로, OOD 검출에 있어 에너지 함수의 효과를 조사합니다. EBMs를 사용하여 밀도 추정 및 분류기 기반 OOD 검출 방법론을 동시에 탐구합니다. 최적화 과정에서는 Sharpness-Aware Minimization (SAM) 접근을 활용했습니다. 그 결과 에너지 점수가 전통적인 소프트맥스 점수보다 우수함을 보였습니다.

- **Performance Highlights**: 방언 OOD 검출 작업의 수치 실험 결과, 제안된 에너지 모델의 효과성과 전통적인 소프트맥스 점수보다 뛰어남을 확인했습니다. 또한 여백 손실과 SAM 방법이 방언 OOD 검출 성능에 미치는 영향을 조사한 결과, EBMs가 방언 OOD 검출에서 내재적 이점을 가지고 있음을 시사합니다.



### Evaluating Quality of Answers for Retrieval-Augmented Generation: A Strong LLM Is All You Need (https://arxiv.org/abs/2406.18064)
Comments:
          12 pages, 6 figures, 12 tables

- **What's New**: 본 논문에서는 vRAG-Eval이라는 새로운 평가 시스템을 소개하며, 이는 Retrieval-Augmented Generation (RAG) 애플리케이션에서 답변의 정확성, 완전성, 정직성을 평가하는 데 사용됩니다. vRAG-Eval은 평가 결과를 이진 점수(accept/reject)로 변환하여 채팅 애플리케이션에서 흔히 사용되는 'thumbs-up' 또는 'thumbs-down' 제스처와 유사한 결정을 내리는 데 유용합니다. 본 평가 시스템을 두 개의 대형 언어 모델(LLMs)에서 실험한 결과, GPT-4의 평가와 인간 전문가의 평가가 83% 일치함을 발견했습니다.

- **Technical Details**: 현재 ChatGPT와 같은 대형 언어 모델(LLMs)은 2022년 11월부터 시작되었으며, OpenAI의 GPT-4 Turbo의 데이터 컷오프(Date Cut-off)는 2023년 4월입니다. Fine-Tuning(미세조정)은 새로운 지식을 주입하기 위해 사용되는 기술이지만 현재 OpenAI의 실험적 접근 프로그램을 통해서만 가능합니다. Retrieval-Augmented Generation(RAG)는 초기 제안되었으며, 비파라메트릭 밀집 벡터 인덱스를 사용해 추가 지식을 저장하고 관련 컨텍스트를 검색하는 방식입니다.

- **Performance Highlights**: GPT-4의 답변 평가가 인간 전문가의 평가와 83% 일치했으며, 이는 RAG 시스템의 답변 품질 평가에서 매우 높은 신뢰성을 보여줍니다. 이는 특히 폐쇄형 도메인(closed-domain) 및 폐쇄형 답변(closed-ended) 환경에서 LLM이 신뢰할 수 있는 평가자가 될 가능성을 시사합니다.



### AdaZeta: Adaptive Zeroth-Order Tensor-Train Adaption for Memory-Efficient Large Language Models Fine-Tuning (https://arxiv.org/abs/2406.18060)
- **What's New**: 새롭게 제안된 Adaptive Zeroth-order Tensor-Train Adaption (AdaZeta) 프레임워크는 Zeroth-order (ZO) 방법의 성능과 수렴성을 크게 향상시키는데 중점을 두었습니다. 이는 메모리 효율적으로 대규모 언어 모델 (LLMs)을 파인튜닝하는 데 효과적입니다.

- **Technical Details**: AdaZeta 프레임워크는 빠른 포워드 패스와 저매개변수 텐서화 어댑터(tensorized adapter)를 도입하여 ZO 추정 정확도를 향상시킵니다. 또한, 수렴성을 보장하기 위해 적응형 쿼리 수 스케줄(adaptive query number schedule)을 제안합니다. 이는 Roberta-Large 및 Llama-2-7B 모델에 대한 이론적 분석과 실험 결과로 입증되었습니다. 각 어댑터는 Tensor-Train (TT) 분해 방식을 사용하여 매우 적은 수의 매개변수로 높은 성능을 제공합니다.

- **Performance Highlights**: AdaZeta는 다양한 작업에서 MeZO, MeZO-LoRA, Sparse-MeZO와 같은 기존 ZO 파인튜닝 방법을 능가하며 더 빠른 수렴 속도를 자랑합니다. 이 프레임워크는 이론적 및 실험적 결과로 정확도, 메모리 효율성, 수렴 속도 면에서 우수한 성능을 입증하였습니다.



### Improving Entity Recognition Using Ensembles of Deep Learning and Fine-tuned Large Language Models: A Case Study on Adverse Event Extraction from Multiple Sources (https://arxiv.org/abs/2406.18049)
- **What's New**: 이번 연구에서는 COVID-19 백신과 관련된 부작용(AE) 추출을 위해 LLMs (Large Language Models)와 전통적인 딥러닝 모델의 효과를 평가하고, 이 모델들을 결합한 효과를 분석했습니다. VAERS, 트위터, 레딧 등의 데이터를 활용하여 'vaccine', 'shot', 'ae' 세 가지 유형의 엔티티를 추출하였습니다.

- **Technical Details**: GPT-2, GPT-3.5, GPT-4, Llama-2와 같은 LLMs 및 RNN, BioBERT 등의 전통적인 딥러닝 모델을 미세조정(fine-tuning)하였으며, GPT-4는 제외하였습니다. 모델 성능을 향상시키기 위해 성능이 우수한 세 모델을 결합한 앙상블 모델을 생성했습니다. 성능 평가는 strict 및 relaxed F1 점수를 사용하여 엔티티 유형별로 평가하고, 마이크로 평균(micro-average) F1 점수를 사용하여 전체 성능을 평가했습니다.

- **Performance Highlights**: 앙상블 모델은 'vaccine', 'shot', 'ae'에서 각각 strict F1 점수 0.878, 0.930, 0.925를 기록하며 가장 높은 성능을 보였고, 마이크로 평균 점수는 0.903을 달성했습니다. 이는 미세조정된 전통적인 딥러닝 모델과 LLMs를 결합한 방안이 AE 관련 정보 추출에 효과적이고 견고함을 나타냅니다.



### PharmGPT: Domain-Specific Large Language Models for Bio-Pharmaceutical and Chemistry (https://arxiv.org/abs/2406.18045)
- **What's New**: 본 연구에서는 바이오제약 및 화학 분야에 특화된 대규모 언어 모델(LLM)인 PharmGPT를 소개합니다. 이 모델은 130억 및 700억 개의 파라미터를 가진 멀티링구얼 LLM으로, 수십억 개의 토큰으로 구성된 방대한 코퍼스를 기반으로 학습되었습니다. 기존의 범용 모델 대비 NAPLEX 등의 주요 벤치마크에서 동등하거나 그 이상의 성능을 보여줍니다.

- **Technical Details**: PharmGPT는 범용 LLM의 한계를 극복하기 위해 개발된 멀티링구얼 모델입니다. 130억 개와 700억 개의 파라미터를 가진 두 가지 구성으로 제공되며, 다양한 언어와 바이오제약 및 화학 분야에 특화된 방대한 코퍼스를 사용하여 훈련되었습니다. 학습 목표와 분산 학습 전략 등 기술적 세부 사항은 논문의 섹션 2.2, 3.1, 3.2에서 자세히 다룹니다.

- **Performance Highlights**: PharmGPT는 NAPLEX 등의 벤치마크에서 기존의 일반 모델을 능가하는 성능을 보이며, 바이오제약 및 화학 분야에서 뛰어난 전문용어 이해능력과 개념 숙달을 입증했습니다. 이는 전문 지식이 필요한 도메인에서 높은 정밀도를 요구하는 작업에 매우 효과적입니다.



### LLMs for Doctors: Leveraging Medical LLMs to Assist Doctors, Not Replace Them (https://arxiv.org/abs/2406.18034)
- **What's New**: 최근의 대형 언어 모델(LLMs)이 의료 분야에서 환자에게 의료 조언과 진단 정보를 제공하는데 큰 성과를 이루었습니다. 그러나, 전문가 지식의 부족으로 인해 환자들은 잘못된 정보로 쉽게 오도될 가능성이 있습니다. 이러한 문제를 해결하기 위해 LLMs를 좀 더 숙련된 의사들과 협력하는 의료 보조자로 조정하는 연구가 진행되었습니다. 이를 위해 DoctorFLAN이라는 중국 의료 데이터를 구축하고, 진료 시나리오에서 LLMs를 평가하기 위한 DoctorFLAN-test 및 DotaBench를 개발했습니다.

- **Technical Details**: 첫 번째 단계로 두 단계의 설문 조사를 통해 의사들이 필요로 하는 의료 보조의 실질적인 요구를 파악했습니다. 이를 바탕으로 92K의 Q&A 샘플, 22개의 업무, 27명의 전문의가 포함된 중국 의료 데이터셋 DoctorFLAN을 구축했습니다. LLMs의 임상적 능력을 평가하기 위해 단일턴 Q&A로 구성된 DoctorFLAN-test와 다중턴 대화로 구성된 DotaBench를 도입했습니다.

- **Performance Highlights**: 평가 결과, 기존 오픈소스 모델들은 여전히 의료 보조자로서의 도전에 직면하고 있으나, DoctorFLAN이 이 모델들을 상당히 향상시키는 것으로 나타났습니다. DoctorFLAN을 통해 훈련된 모델들은 DoctorFLAN-test와 DotaBench에서 강력한 성과를 보였습니다. 특히, 사전 교육과 같은 의사들이 우선순위로 두는 업무에 대해 뛰어난 성능을 나타냈습니다.



### Automated Clinical Data Extraction with Knowledge Conditioned LLMs (https://arxiv.org/abs/2406.18027)
- **What's New**: 이번 논문에서는 임상 및 의료 이미지 보고서에서 폐 병변 정보를 추출하는 새로운 프레임워크를 제안하였습니다. 이 프레임워크는 대형 언어 모델(LLMs)이 생성한 내부 지식을 외부 지식과 정렬하며, 특히 In-Context Learning(ICL)을 활용하여 정확성과 신뢰성을 크게 향상시킵니다. 이 접근법은 폐 병변 발견 감지와 주요 구조화 필드 분석을 개선하는 동시에, 병변 설명 텍스트를 추가 구조화 필드로 구분하는 두 단계 프로세스를 사용합니다.

- **Technical Details**: 제안된 프레임워크는 내부 지식 기반과 전문 지식 기반 외부 지식을 정렬하여 반복적으로 업데이트합니다. 내부 지식 기반은 수동으로 큐레이션된 의료 보고서 교육 자료를 사용하여 생성된 레퍼런스를 통합합니다. Retriever 모듈은 관련 있는 내부 또는 외부 지식 단위를 식별하고, Grader 모듈은 추출된 내부 지식 규칙의 진실성과 유용성을 평가합니다. 이를 통해 보고서에서 정보 추출을 수행할 때 내부 지식 기반의 규칙을 활용하여 추출 패턴의 정렬 효과를 높입니다.

- **Performance Highlights**: 전문가가 큐레이션한 테스트 데이터셋으로 실험한 결과, 제안된 ICL 접근법은 기존 ICL 방법보다 주요 필드(병변 크기, 가장자리 및 고형성)에 대해 F1 점수를 평균 12.9% 증가시켰습니다. 이 프레임워크는 특히 정교한 세부 사항 추출에서 높은 정확성을 나타내며, 복잡한 도메인별 필드의 추출에서 LLMs의 한계를 극복하는 데 효과적입니다.



### Decoding with Limited Teacher Supervision Requires Understanding When to Trust the Teacher (https://arxiv.org/abs/2406.18002)
Comments:
          preprint

- **What's New**: 이번 연구에서는 제한된 대형 언어 모델(LLM) 감독 하에서 소형 언어 모델(sLLM)의 생성 품질을 향상시키기 위한 새로운 알고리즘을 제안했습니다. 이는 제한 없는 LLM 감독을 사용하는 기존 방법들과는 다르게, LLM으로부터 극히 제한된 토큰 수의 감독을 받을 수밖에 없는 시나리오를 가정하고 있습니다.

- **Technical Details**: 제안된 알고리즘은 초기에 LLM과 sLLM의 예측을 효과적으로 통합하여 sLLM만으로 후속 토큰 생성을 더욱 정확하게 조건화하는 것을 목표로 합니다. 주요 아이디어는 sLLM의 신뢰도에 따라 LLM 예측을 전적으로 신뢰하거나 무시하는 적응형 방법입니다. 이를 위해 예측된 토큰의 엔트로피(즉, 신뢰도)에 기반하여 교사의 정보를 신뢰할지 학생의 정보를 신뢰할지 결정하는 방법을 제안했습니다.

- **Performance Highlights**: 다양한 모델과 데이터셋을 통한 실험 결과, 제안된 방법은 기존의 디코딩 전략에 비해 일관된 성능 향상을 제공했습니다. 특히, 단일 토큰의 감독만으로도 정확도가 상당히 증가하며, 최적의 신뢰도 매개변수 α는 과제와 모델에 따라 크게 달라질 수 있음을 발견했습니다. 예를 들어, StrategyQA 데이터셋에서 Phi3-mini 모델의 경우 단일 LLM 감독으로 정확도가 2%p 증가했습니다.



### Catching Chameleons: Detecting Evolving Disinformation Generated using Large Language Models (https://arxiv.org/abs/2406.17992)
Comments:
          10 pages, 5 figures

- **What's New**: 최근 연구에서 LLM(대형 언어 모델)들이 생성하는 디스인포메이션(허위 정보)의 탐지가 진화하는 문제를 다루고 있습니다. 기존 방법들은 진화하는 디스인포메이션을 효과적으로 탐지하지 못한 반면, 이 연구는 DELD라는 새로운 접근법을 제안합니다. DELD는 사전 학습된 언어 모델(PLM)의 일반적인 사실 확인 능력과 다양한 LLM의 독립적인 디스인포메이션 생성 특성을 결합하여 디스인포메이션을 탐지합니다.

- **Technical Details**: DELD는 디스인포메이션의 의미적 임베딩(semantic embedding)과 트레이너블 소프트 프롬프트(trainable soft prompts)를 결합하여 모델별 지식을 효과적으로 이끌어냅니다. 모델의 독립적인 특성 학습을 통해 지식을 순차적으로 축적하고 변환합니다. 이는 디스인포메이션 라벨의 부족 문제를 해결하는 데 도움을 줍니다.

- **Performance Highlights**: DELD는 최첨단 방법보다 성능이 뛰어나다는 것을 실험 결과로 입증했습니다. 또한, DELD는 다양한 LLM별 디스인포메이션 생성 패턴에 대한 중요한 통찰을 제공합니다. 이를 통해 진화하는 디스인포메이션을 효과적으로 탐지할 수 있는 새로운 연구 분야에 기여합니다.



### Explicit Diversity Conditions for Effective Question Answer Generation with Large Language Models (https://arxiv.org/abs/2406.17990)
Comments:
          Published at COLING 2024

- **What's New**: 새로운 연구에서는 질문 답변 생성(QAG)을 위한 명시적 다양성 조건을 제안합니다. 기존의 사전 훈련된 언어 모델(PLM)과 대형 언어 모델(LLM) 기반 QAG 방법들은 높은 품질의 QA 쌍을 생성했지만, 중복된 QA 쌍을 생성하는 문제가 있었습니다. 이를 해결하기 위해 연구팀은 공간적 측면, 질문 유형, 그리고 엔터티를 고려한 명시적 다양성 조건을 도입했습니다.

- **Technical Details**: 연구팀은 세 가지 명시적 조건을 제안합니다: (1) 문서 내 다양한 위치에서 QA 쌍을 생성하는 위치 조건, (2) 8가지 WH 질문 유형을 사용하는 질문 유형 조건, (3) 다양한 엔터티에 기반한 질문을 생성하는 엔터티 조건. 이러한 명시적 다양성 조건은 입력 문서에 프롬프트 형태로 결합되어 더욱 다양한 QA 쌍을 생성합니다. 이 방법은 특히 대형 언어 모델(LLM)을 사용할 때 우수한 다운스트림 QA 성능을 보였습니다.

- **Performance Highlights**: 이 명시적 다양성 조건을 사용한 QAG 모델은 기존의 암시적 샘플링 기법을 사용한 QAG 모델에 비해 SQuADDU 데이터셋에서 Exact Match(EM) 4.1%와 F1 점수 4.5%의 성능 향상을 보였습니다. 또한, 다중 도메인 저자원 데이터셋인 SubjQA에서는 평균적으로 EM 12%의 성능 향상을 보였습니다. 생성된 QA 쌍의 토큰 중복율은 암시적 샘플링 기법의 64%에 비해 30%로 현저히 낮았습니다.



### Multi-step Knowledge Retrieval and Inference over Unstructured Data (https://arxiv.org/abs/2406.17987)
- **What's New**: Elemental Cognition (EC)에서는 고위험 의사 결정 작업의 복잡한 연구와 발견 작업을 수행하기 위해 설계된 Cora라는 Collaborative Research Assistant를 발표했습니다. Cora는 세밀하게 튜닝된 대형 언어 모델(LLMs)과 강력한 상징적 추론 엔진을 통합한 신경-상징적 AI(neuro-symbolic AI) 플랫폼을 기반으로 합니다.

- **Technical Details**: Cora 시스템은 지식 추출 및 정렬을 위한 미세 조정된 LLMs와 논리적 추론, 계획 및 대화형 제약 해결(interactive constraint solving)을 위한 상징적 추론 엔진을 결합하여 작동합니다. 이 접근 방식은 종합적이고 논리적으로 일관된 결과를 제공하기 위해 다단계 추론 문제를 효과적으로 해결합니다. 또한, 지식 그래프와 온톨로지 같은 구조화된 데이터와 대형 비구조화 텍스트 코퍼스를 결합하여 통일된 지식 소스를 구축하는 방식으로 동작합니다.

- **Performance Highlights**: 시험 결과, Cora는 기존의 LLM 및 Retrieval-Augmented-Generation (RAG) 기반 접근 방식보다 우수한 성능을 보였습니다. 이는 복잡한 문제 영역에서 다단계 추론 문제를 해결하는 데 있어 Cora의 신경-상징적 접근 방식이 효과적이라는 점을 부각시킵니다.



### EDEN: Empathetic Dialogues for English learning (https://arxiv.org/abs/2406.17982)
- **What's New**: EDEN은 영어 학습을 위한 고품질 오픈 도메인(chatbot) 대화 시스템입니다. EDEN은 특히 유연한 공감 피드백을 제공하여 학습자가 더 높은 정서적 지원을 느끼고, 그 결과 더 높은 학습 끈기를 보일 수 있도록 설계되었습니다. 이 연구는 공감 피드백이 학습자의 감정적 지원 인식에 긍정적인 영향을 미쳐 학습 끈기를 증가시킨다는 가설을 실험적으로 검증했습니다.

- **Technical Details**: EDEN을 구축하기 위해, 우리는 먼저 말하는 문법 교정 모델과 고품질의 소셜 채팅 대화를 위한 모델을 훈련시켰습니다. 그 후 다양한 공감 피드백 전략을 테스트하는 초기 사용자 연구를 실시했습니다. EDEN의 설계는 사용자의 발화(sentiment)를 분석하고 필요한 경우 공감 피드백을 생성하며, 문법 교정 모듈을 통해 문법 오류를 교정합니다. 사용자 질의에 대해서는 ChatGPT를 이용하여 추가적인 문법 피드백을 제공하고 대화를 이어갑니다.

- **Performance Highlights**: 실험 결과, 적응형 공감 피드백(adaptive empathetic feedback) 전략이 가장 높은 정서적 지원 인식을 유도하는 데 성공적이었으며, 이는 학습자의 끈기와 긍정적인 변화를 예측하는 데 유효했습니다.



### Inherent Challenges of Post-Hoc Membership Inference for Large Language Models (https://arxiv.org/abs/2406.17975)
- **What's New**: 본 논문에서는 대규모 언어 모델(LLM)의 훈련 데이터 구성에 대한 투명성을 얻기 위해 개발된 회원 유추 공격(Membership Inference Attacks, MIAs)이 평가되는 방식의 근본적인 문제점을 지적합니다. 특히, 회원과 비회원 데이터셋 간의 잠재적인 분포 변화로 인해 기존의 높은 MIA 성능 결과가 오히려 모델의 기억보다는 이러한 분포 변화에 기인할 수 있음을 보입니다.

- **Technical Details**: LLM의 훈련 데이터(회원)와 비훈련 데이터(비회원)를 구분하기 위해 간단한 단어 주머니(bag-of-words) 분류기를 사용하여 회원과 비회원 문서에 대한 잠재적인 분포 이동을 테스트했습니다. LLM의 훈련 데이터세트를 이용하여 수집된 데이터들에서 분포 이동이 확인되었습니다. 이를 해결하기 위해 회귀 불연속 설계(Regression Discontinuity Design, RDD) 접근 방식을 제시하여 데이터 수집 시 분포 이동을 크게 완화시켰습니다.

- **Performance Highlights**: RDD 세팅에서 다양한 MIA 방법론을 평가한 결과, 이전에 보고된 결과와는 대조적으로 성능이 무작위 추측보다 약간 높은 수준에 불과했습니다. 이는 기존 성능 보고서가 기본적으로 분포 이동에 기인했으며 현재 최신의 MIA들은 분포 이동 없는 세팅에서는 무작위 추측보다 성능이 뛰어나지 않음을 시사합니다.



### Evaluating Fairness in Large Vision-Language Models Across Diverse Demographic Attributes and Prompts (https://arxiv.org/abs/2406.17974)
- **What's New**: 최근 대형 비전-언어 모델(Large Vision-Language Models, LVLMs)이 개방형 시각 이해에서 큰 진전을 이뤘으나, 실제 생활에서 성별, 피부색, 나이와 같은 속성에 따른 인구 통계학적 편향을 어떻게 해결하는지는 아직 명확하지 않습니다. 본 논문에서는 공공 공정성 벤치마크 데이터셋(FACET)을 기반으로 주요 LVLM들의 시각적 공정성을 실험적으로 조사하고 인구 통계학적 속성에 따른 성능 차이를 감사합니다.

- **Technical Details**: FACET와 같은 인간 주석형 공정성 벤치마크를 사용하여 단일 인물이 포함된 이미지를 선택해 LVLM들의 성별, 피부색, 연령 등에 따른 시각적 공정성을 평가했습니다. 제안된 평가 프레임워크는 다양한 프롬프트(instruct prompts)와 이미지 시나리오를 활용하여 모델의 이미지를 이해하는 능력을 평가하고, 이에 따른 편향을 분석합니다. 구체적으로, 직접 질문 프롬프트와 단일 선택 질문 프롬프트 방법을 통해 각 모델의 성능을 측정하고 편향 여부를 평가합니다.

- **Performance Highlights**: 실험 결과에 따르면, 개방형 소스와 폐쇄형 소스 모두 다양한 프롬프트와 인구 통계학적 속성에 대해 공정성 이슈를 보이고 있습니다. 제안된 프레임워크는 프롬프트에 따라 예측이 달라지는 모델의 성능 차이를 정량적으로 분석하며, 공정성 메트릭(FACET 벤치마크 기준)에 따라 모델의 예측 정확도와 인구 통계학적 속성이 예측에 미치는 영향을 평가합니다.



### Encourage or Inhibit Monosemanticity? Revisit Monosemanticity from a Feature Decorrelation Perspectiv (https://arxiv.org/abs/2406.17969)
- **What's New**: 본 논문에서는 대형 언어 모델(LLM)의 내재적 메커니즘을 해석하기 위해 단일의미성(monosemanticity)에 초점을 맞추고 있습니다. 단일의미 뉴런은 단일하고 구체적인 개념에 전념하며, 이는 뉴런과 개념 간의 일대일 상관관계를 형성합니다. 기존 연구에서는 단일의미성이 모델 성능에 미치는 영향을 논의했지만 그 이득 여부는 불확실했습니다. 본 연구에서는 특징 비상관(feature decorrelation) 관점에서 단일의미성을 재조명하고 그것의 장려를 제안합니다. 실험적으로 단일의미성이 모델 성능과 긍정적인 상관관계를 가진다는 것을 보여주었습니다.

- **Technical Details**: 본 연구는 특징 비상관을 단일의미성의 대리로 사용하고, 동적 선호 최적화 과정(dynamic preference optimization process)에 특징 비상관 규제를 통합합니다. 실험에서 제안한 방법은 표현 다양성(representation diversity)과 활성화 희소성(activation sparsity)을 향상시키며, 선호 정렬 성능도 개선된다는 것을 보여주었습니다. 참고로, 특징 비상관은 특정 입력 특징과 일대일로 대응하는 뉴런의 활성화 패턴을 의미합니다. 실험에 사용된 방법으로는 sparse autoencoder 및 전역적 규제를 포함합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 표현 다양성과 활성화 희소성을 향상시켰을 뿐만 아니라 선호 정렬 성능도 향상시켰습니다. 이는 제안된 특징 비상관 규제가 모델의 단일의미성을 효과적으로 증가시킨다는 것을 의미합니다. 따라서 이는 단일의미성이 모델의 내재적 메커니즘을 개선할 수 있는 강력한 도구임을 시사합니다.



### Unmasking the Imposters: In-Domain Detection of Human vs. Machine-Generated Tweets (https://arxiv.org/abs/2406.17967)
- **What's New**: 본 논문은 최신의 큰 언어 모델(LLMs)을 사용하여 소셜 미디어 플랫폼에서의 오용을 분석하기 위한 새로운 방법론을 제시합니다. 특히 Twitter 데이터를 사용하여 Llama 3, Mistral, Qwen2, 그리고 GPT4o 네 가지 LLM의 생성 능력을 평가하였습니다. 이에 따라 도메인별 추가 미세 조정(fine-tuning)과 '비검열'(uncensored) 버전의 영향도 검증하였습니다.

- **Technical Details**: 7B 및 8B 매개 변수를 가진 세 종류의 오픈 소스 LLM( Llama 3, Mistral, Qwen2)과 GPT4o 모델의 기초 지침 모델을 평가하였습니다. TweetEval 데이터를 이용하여 인간이 작성한 텍스트와 기계가 생성한 텍스트를 구별하는 방법을 사용하였으며, BERTweet, 소프트 보팅 앙상블(soft-voting ensemble), RADAR, 추가 언어적 특징을 사용하여 감지 방법을 평가하였습니다.

- **Performance Highlights**: '비검열된' 모델에 도메인별 미세 조정을 추가하면 자동 감지 방법의 효과가 크게 감소한다는 결과를 얻었습니다. 최악의 경우 감지율이 16.86% 절대적으로 감소하였으며, 이는 기계가 생성한 텍스트를 식별하는 데 큰 어려움을 초래합니다. 본 연구는 Twitter 데이터를 기반으로 한 '검열된' 및 '비검열된' 오픈 소스 LLM의 텍스트 생성 능력을 이해하는 첫 시도를 제시합니다.



### SimsChat: A Customisable Persona-Driven Role-Playing Agen (https://arxiv.org/abs/2406.17962)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)을 활용하여 현실 세계의 캐릭터를 자유롭게 맞춤 설계할 수 있는 'Customisable Conversation Agent' 프레임워크를 소개합니다. 이 프레임워크는 사용자 취향에 따라 사용자 정의가 가능한 캐릭터와 역할 대행 에이전트를 디자인하는 데 유용합니다.

- **Technical Details**: 우선, 우리는 68개의 다양한 맞춤형 캐릭터와 1360개의 멀티턴 롤플레잉 대화를 포함한 'SimsConv' 데이터셋을 제안합니다. 이 데이터셋은 경력, 목표, 특성, 기술 등 현실 세계의 여러 요소를 바탕으로 생성된 캐릭터로 구성되어 있습니다. 'SimsChat' 모델은 이러한 데이터를 바탕으로 실감나는 다양한 감정, 성격, 디테일한 생활 경험을 시뮬레이션하여 사용자 정의가 가능한 롤플레잉 에이전트를 제공합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 만족스러운 성능을 보였으며, 인간의 다양한 특징을 갖춘 시뮬라크라 개발을 위한 유용한 가이드를 제공합니다. SimsChat 모델은 사용자가 정의한 인물의 성격과 지식을 정확하게 유지하며 다양한 삶의 경험을 학습할 수 있는 능력을 입증했습니다.



### NormTab: Improving Symbolic Reasoning in LLMs Through Tabular Data Normalization (https://arxiv.org/abs/2406.17961)
Comments:
          Work in Progress

- **What's New**: 최근 몇 년간 대규모 언어 모델(Large Language Models, LLMs)은 텍스트 데이터 파싱 및 코드 생성에서 뛰어난 성능을 보였습니다. 그러나 LLMs는 구조적 다양성과 일관성 없는 테이블 셀 값 문제로 인해 탁상 데이터를 포함한 작업에서 상징적 추론(symbolic reasoning)에 어려움을 겪습니다. 이를 해결하기 위해, 우리는 웹 테이블을 정규화하여 상징적 추론 성능을 향상시키는 새로운 프레임워크인 NormTab을 소개합니다.

- **Technical Details**: NormTab은 테이블 정규화를 독립적인 전처리 단계로 수행하여 LLM 기반 상징적 추론을 지원합니다. 이를 통해 구조적 변동성 및 혼합 데이터 형식 문제 등을 해결하여 정확하고 효율적인 상징적 추론 및 쿼리 처리를 가능하게 합니다. 특히 테이블 구조 정규화 및 값 정규화를 포함합니다. 이를 통해 일관성과 정확성을 보장하고, LLM이 더 나은 데이터 클리닝 및 변환 작업을 수행할 수 있도록 도와줍니다.

- **Performance Highlights**: NormTab은 WikiTableQuestions와 TabFact 같은 도전적인 웹 테이블 데이터셋에서 상당한 상징적 추론 성능 향상을 입증합니다. 이러한 데이터셋들은 테이블 구조와 내용의 다양성을 제공하여, 웹 테이블 정규화가 LLM 기반 상징적 추론 작업에 미치는 영향을 철저히 조사할 수 있게 합니다. 실험 결과, NormTab을 활용함으로써, 복잡한 추론 작업에서 LLMs의 성능을 크게 향상시킬 수 있음을 보여줍니다.



### Do they mean 'us'? Interpreting Referring Expressions in Intergroup Bias (https://arxiv.org/abs/2406.17947)
- **What's New**: 이번 논문에서는 그룹 내(in-group)와 그룹 밖(out-group) 말하기의 차이를 태깅 태스크(tagging task)로 모델링하여 사회적 현상들을 설명하는 데 중점을 두고 있습니다. NFL 팀 팬덤 포럼에서 나온 영어 스포츠 댓글을 활용, 특히 게임 시간 동안 반대 팀의 관점에서 나온 댓글을 분석합니다. 총 600만 건 이상의 게임 시간 댓글을 포함하는 독특한 데이터를 수집하여, 각 댓글은 팀의 실시간 승리 확률(live win probabilities)에 기반한 설명과 연관됩니다.

- **Technical Details**: 이 논문은 전문가 및 대중의 주석을 통해 암묵적이고 명시적인 언급 표현(refering expressions)을 태깅하여 언어와 세계에 대한 풍부한 맥락적 이해가 필요함을 보입니다. 대규모 분석을 위해 대형 언어 모델(LLMs)을 활용하여 자동 태깅을 수행하고, 일부 LLM은 수치 확률보다는 언어적 설명으로 승리 확률을 제시할 때 최상의 성능을 보인다는 것을 발견했습니다.

- **Performance Highlights**: LLM을 사용하여 댓글을 대규모로 태깅한 결과, 승리 확률에 따라 변화하는 기준의 선형 변화를 발견했습니다. 이는 그룹 내와 그룹 밖 발화의 차별점을 나타냅니다. 코드와 데이터는 공개되어 있으며, 이 링크를 통해 접근할 수 있습니다.



### Sequential Editing for Lifelong Training of Speech Recognition Models (https://arxiv.org/abs/2406.17935)
Comments:
          INTERSPEECH 2024

- **What's New**: 기존의 Automatic Speech Recognition (ASR) 시스템은 새로운 도메인 데이터를 추가할 때 발생하는 재학습의 비효율성과 관련된 문제를 안고 있습니다. 이러한 문제를 해결하기 위해, 본 논문에서는 Sequential Model Editing이라는 새로운 방법을 제안합니다. 이 방법은 추가적인 파라미터나 이전 도메인 데이터에 접근할 필요가 없습니다. 우리의 접근 방식은 새로운 도메인 데이터를 사용하여 점진적으로 모델을 업데이트하는 데 사용됩니다.

- **Technical Details**: 기존의 ASR 시스템은 주로 Conformer-based CTC 모델을 사용하여 음성 데이터를 텍스트로 변환합니다. 이러한 모델들은 일반적으로 많은 양의 음성-텍스트 페어 데이터를 사용하여 학습합니다. Sequential Model Editing 접근 방식은 Task Arithmetic와 TIES-Merging 기법을 활용하여 이전 도메인 데이터에 접근하지 않고도 새로운 도메인에 대한 모델을 지속적으로 업데이트합니다. 이러한 Task Vectors는 주어진 작업을 잘 수행하기 위해 필요한 정보를 인코딩하며, 이는 기존 모델에 새로운 기능을 추가하거나 불필요한 동작을 제거하는 데 사용됩니다.

- **Performance Highlights**: 본 연구에서는 CommonVoice 영어 멀티-악센트 데이터셋에서 15%의 Word Error Rate Reduction (WERR)을 달성하여 기존의 파인 튜닝 기법보다 뛰어난 성능을 보였습니다. 이는 기존 연구에서 제안된 방법들의 6% WERR과 비교할 때 상당히 높은 성능 향상입니다.



### FASA: a Flexible and Automatic Speech Aligner for Extracting High-quality Aligned Children Speech Data (https://arxiv.org/abs/2406.17926)
Comments:
          4 pages, 1 figure

- **What's New**: 최근까지 성인 음성에 대한 자동 음성 인식(Automatic Speech Recognition, ASR)은 깊은 신경망(MNN) 모델을 사용하여 상당한 진전을 이루었으나, 아동 음성에 대한 성과는 여전히 만족스럽지 않습니다. 이는 아동 음성이 가지는 독특한 특성 때문입니다. 본 논문에서는 FASA(Flexible and Automatic Speech Aligner)라 불리는 새로운 자동 강제 정렬 도구를 제안합니다. 이를 통해 많은 양의 기존 노이즈가 있는 아동 음성 데이터에서 고품질로 정렬된 데이터를 추출할 수 있습니다. CHILDES 데이터셋을 사용한 실험 결과, FASA는 인간 주석에 비해 데이터 품질을 13.6배 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: 아동 음성 데이터의 고유한 발음 패턴과 단어 사용으로 인해 고품질의 맞춤 데이터 확보가 어렵습니다. 기존의 강제 정렬 도구는 입력된 전사(transcription)의 품질에 따른 가정들이 비현실적이어서 사용이 제한적입니다. FASA는 이러한 문제를 해결하기 위해 개발되었으며, 딥러닝(DL) 모델을 백본으로 하여 최소한의 전사 조건에서도 강력한 정렬 성능을 제공합니다. 이를 통해 다양한 소스에서 아동 음성을 추출하여 고품질의 데이터셋을 구성할 수 있습니다.

- **Performance Highlights**: FASA를 통해 CHILDES 데이터셋 기반의 많은 새로운 고품질 아동 음성 데이터셋을 구성할 수 있었습니다. 이 도구는 기존의 강제 정렬 도구 및 인간 주석자와 비교했을 때 일관되게 우수한 성능을 보여주었습니다. 실험 결과로 FASA는 인간 주석자에 비해 데이터 품질을 13.6배 향상시켰습니다.



### PAFT: A Parallel Training Paradigm for Effective LLM Fine-Tuning (https://arxiv.org/abs/2406.17923)
- **What's New**: 이번 논문은 대형 언어 모델(LLMs)의 세부 조정(fine-tuning)을 위한 새로운 병렬 학습 패러다임인 PAFT를 소개합니다. PAFT는 기존의 순차적 학습 파이프라인에서 발생하는 'alignment tax' 문제를 해결하기 위해 설계되었습니다. 기존 방식에서는 Supervised Fine-Tuning(SFT) 후 인간의 선호도 맞춤 (preference alignment)을 순차적으로 진행했으나, PAFT는 이를 병렬로 수행하여 각기 독립적으로 최적화한 모델을 병합하는 방식입니다.

- **Technical Details**: PAFT 방식에서는 동일한 사전 훈련된 모델을 사용하여 SFT와 선호도 맞춤을 각각 다른 데이터셋에서 독립적으로 수행합니다. 그런 다음 생성된 두 모델을 파라미터 병합(parameter fusing) 기법을 사용하여 최종 모델로 합칩니다. SFT는 모델이 자연스럽게 밀도가 높아지도록 만드는 반면, 선호도 맞춤은 자연적으로 스파스(sparse)한 모델을 생성한다는 것이 중요한 발견입니다. 또한, 델타 파라미터의 간섭(interference)에 대한 문제를 해결하기 위해 L1-norm 패널티를 도입하여 효과적인 스파스화(sparsifying)를 달성했습니다.

- **Performance Highlights**: PAFT를 사용한 LLM은 HuggingFace Open LLM Leaderboard에서 1위를 달성했습니다. 7B 모델에서 평균 0.6524 점을 기록했으며, 이는 기존 모델들을 모두 능가하는 성과입니다. 특히, PAFT를 통한 특정한 모델 병합 방법(TIES 및 Task Arithmetic)이 높은 성능을 보였음을 확인했습니다. 이로써 순차적 학습보다 병렬 학습이 alignment tax를 효과적으로 줄이고, 결과적으로 모델 통합 시 발생할 수 있는 충돌을 방지하며 각 모델의 전체 역량을 유지할 수 있음을 보였습니다.



### X-ray Made Simple: Radiology Report Generation and Evaluation with Layman's Terms (https://arxiv.org/abs/2406.17911)
- **What's New**: 본 논문은 Radiology Report Generation (RRG) 분야에서 기존의 표면적 평가 방식의 한계를 극복하기 위해 Layman's RRG 프레임워크를 제안합니다. 해당 프레임워크는 평이한 일상 언어로 번역된 데이터셋, 새로운 평가 방법, 그리고 새로운 훈련 방식을 도입하여 보다 공정하고 의미론적인 평가 기준을 제공합니다.

- **Technical Details**: Layman's RRG 프레임워크는 두 가지 주요 데이터셋을 도입합니다: 일상 언어 문장 수준 데이터셋과 보고 수준 데이터셋. 이러한 데이터셋을 기반으로 의미론적 평가 방법을 제안하여 BLEU와 같은 기존의 어휘 기반 평가 지표의 부풀려진 숫자를 완화합니다. 또한, 이 데이터셋을 사용하여 모델이 템플릿을 학습하는 대신 보고서의 의미에 집중할 수 있도록 합니다. 주요 알고리즘은 생성 및 수정 과정을 포함하며, 임베딩 모델과 LLM을 활용한 자체 검열 시스템을 통해 최적의 품질을 보장합니다.

- **Performance Highlights**: 실험 결과, 문장 레벨의 Layman's terms 데이터셋과 의미 기반 평가 방법을 결합할 때, 평가 결과가 상당히 더 강건해짐을 확인했습니다. 보고서 레벨 Layman 데이터셋을 사용한 훈련은 모델의 의미 이해를 향상시키며, 증가하는 훈련 예시 수에 따라 모델 성능이 향상되는 유망한 스케일링 법칙을 입증했습니다.



### Mapping the Past: Geographically Linking an Early 20th Century Swedish Encyclopedia with Wikidata (https://arxiv.org/abs/2406.17903)
Comments:
          9 pages, 3 figures

- **What's New**: 이번 연구에서는 20세기 초에 출판된 스웨덴 백과사전 'Nordisk Familjebok'의 두 번째 판인 'Uggleupplagan'에서 모든 지리적 항목을 추출했습니다. 이는 스웨덴 백과사전 중 가장 방대하며, 약 18,000개의 유효한 좌표를 포함하는 지리적 항목을 추출하여 분석했습니다.

- **Technical Details**: 우리는 OCR을 통해 백과사전의 텍스트를 디지털화하고, 분류기를 사용하여 항목의 카테고리를 결정했습니다. 약 22%의 항목이 지리적 위치와 관련이 있음을 확인했습니다. 그 후, Named Entity Recognition(NER) 기술을 적용하여 이러한 항목들을 Wikidata에 연결하고, 좌표를 추출했습니다. 이 데이터는 GitHub에 게시되었습니다(https://github.com/axelahlin/uggleupplagan).

- **Performance Highlights**: 전체 분석 결과, 백과사전에 포함된 대부분의 지리적 항목이 스웨덴, 독일, 영국에서 집중적으로 나타났습니다. 이는 스웨덴이 해당 시기 동안 강한 관계를 유지했던 국가들과 관련이 있습니다. 이번 연구는 20세기 초의 백과사전에서 지리적 정보의 선택 및 표현에 대한 통찰을 제공하며, 향후 다양한 백과사전을 비교 분석하는 연구의 기반을 마련할 수 있습니다.



### Script-Agnostic Language Identification (https://arxiv.org/abs/2406.17901)
Comments:
          Under Review in ACL Rolling Review

- **What's New**: 새로운 연구에서는 여러 스크립트(script)로 작성된 현대 언어에 대해 스크립트에 구애받지 않는 표현(script-agnostic representations)을 학습하는 전략을 제안합니다. 이 연구는 언어 식별(Language Identification)을 개선하기 위한 다양한 실험적 전략(업스케일링, 평탄화, 스크립트 혼합)을 적용하여 수행되었습니다.

- **Technical Details**: 연구자들은 드라비다 계열의 네 가지 주요 언어(타밀어, 텔루구어, 칸나다어, 말라얄람어)를 대상으로 다양한 스크립트 기반의 언어 식별 문제를 해결하고자 했습니다. 단어 수준의 스크립트 무작위화(script randomization)와 여러 스크립트로 작성된 언어에 대한 노출이 중요함을 발견했습니다. 사용된 방법론으로는 업스케일링(upscaling), 평탄화(flattening), 스크립트 혼합(script mixing)이 있습니다.

- **Performance Highlights**: 연구의 결과, 스크립트에 구애받지 않는 언어 식별 과정에서 경쟁력 있는 성능을 유지하면서도 다중 스크립트 자연 발생 텍스트에서도 우수한 성과를 보였습니다. 특히 단어 수준의 스크립트 무작위화가 중요한 역할을 했습니다.



### CTBench: A Comprehensive Benchmark for Evaluating Language Model Capabilities in Clinical Trial Design (https://arxiv.org/abs/2406.17888)
- **What's New**: CTBench는 임상 연구 설계를 돕기 위해 언어 모델(LMs)을 평가하는 새로운 벤치마크를 도입했습니다. CTBench는 연구 특유의 메타데이터를 제공받아 임상 시험(CT)의 기본 특성을 결정하는 AI 모델의 능력을 평가합니다. 이러한 기본 특성은 참가자들로부터 수집된 인구 통계적 및 관련 특성을 포함하며 임상 시험 출판물, 특히 'Table 1'에서 자주 나타납니다. 이 연구는 CTBench를 통해 AI가 CT 설계에 미치는 영향을 조사하고 CT의 효율성과 견고성을 향상시키는데 기여하고자 합니다.

- **Technical Details**: CTBench는 두 개의 데이터셋으로 구성됩니다: 'CT-Repo'는 clinicaltrials.gov API에서 수집된 1,690개의 임상 시험의 기본 특성을 포함하며, 'CT-Pub'는 관련 출판물에서 수집된 100개의 시험을 포함한 데이터셋입니다. 두 가지 LM 기반 평가 방법 'ListMatch-LLM'과 'ListMatch-BERT'가 개발되어, 실제 기본 특성 목록을 LM이 생성한 응답과 비교합니다. LLaMa3-70B-Instruct와 GPT-4o를 사용한 zero-shot 및 three-shot 학습 설정에서 잠재적인 기본 특성을 생성하는 고도화된 프롬프트 엔지니어링 기술이 적용되었습니다.

- **Performance Highlights**: GPT-4o의 평가자 성능은 임상 전문가에 의해 CT-Pub 데이터셋에서 LM이 생성한 기능과 실제 기능 간의 일치를 확인하여 검증되었습니다. 결과는 유망한 개선 가능성을 나타내며, CTBench를 CT 설계의 AI 연구 발전을 위한 유용한 도구로 위치시킵니다.



### Cloaked Classifiers: Pseudonymization Strategies on Sensitive Classification Tasks (https://arxiv.org/abs/2406.17875)
Comments:
          Proceedings of the fifth Workshop on Privacy in Natural Language Processing

- **What's New**: 이번 논문에서는 다언어 급진화(radicalization) 데이터셋을 수동으로 가명처리(pseudonymization)하는 방법론을 소개합니다. 이는 유럽의 GDPR과 같은 개인정보 보호 규정을 준수하면서도 데이터의 유용성을 유지하려는 시도를 담고 있습니다. 저자들은 가명처리된 데이터셋이 원본과 유사한 성능을 유지함을 보이며, 민감한 NLP 데이터 처리에 대한 종합적인 가이드라인을 공유합니다.

- **Technical Details**: 본 연구는 급진화 데이터셋의 가명처리를 수동으로 수행하는 방법을 제시합니다. 데이터셋에는 영어, 프랑스어, 아랍어 콘텐츠가 포함되어 있으며, 다양한 원천(포럼, 텔레그램 및 기타 소셜 미디어 플랫폼)으로부터 수집되었습니다. 가명처리 과정은 개인 식별 정보를 가명으로 교체(pseudonymization)하면서, 공개 이벤트와 공인과 같은 예외적인 경우는 가명처리를 적용하지 않도록 하였습니다.

- **Performance Highlights**: 가명처리된 데이터셋으로 훈련한 모델은 원본 데이터셋으로 훈련한 모델과 유사한 수준의 성능을 유지합니다. 이는 개인정보 보호를 강화하면서도 데이터의 유용성을 보장할 수 있음을 시사합니다.



### Improving Arithmetic Reasoning Ability of Large Language Models through Relation Tuples, Verification and Dynamic Feedback (https://arxiv.org/abs/2406.17873)
Comments:
          Under review, 25 figures, 8 tables, 29 pages

- **What's New**: 이 논문에서는 기존의 자연어 또는 프로그래밍 코드 형태로 표현되는 대형 언어 모델의 추론 단계를 반구조적 표현 형식인 'relation tuples'를 사용하도록 제안합니다. 이 방식은 사람이 쉽게 읽을 수 있으면서도 기계가 처리하기 용이한 형태로, 자연어보다 검증이 쉽습니다.

- **Technical Details**: 연구는 세 가지 주요 구성 요소로 이루어진 프레임워크를 구현합니다. (1) 대형 언어 모델의 추론 단계에 'relation tuples'을 도입합니다. (2) 'relation tuples'을 기반으로 로컬 코드 인터프리터를 사용해 추론 단계를 자동으로 검증합니다. (3) 동적 피드백 메커니즘(dynamic feedback mechanism)을 통합해 모델의 자기 개선을 돕습니다.

- **Performance Highlights**: 다양한 산술 데이터셋에서 실험 결과, 제안된 방법이 대형 언어 모델의 산술 추론 능력을 향상시키는 데 효과적임을 보여줍니다. 특히 'relation tuples'을 사용한 예제를 포함하면, 7개의 산술 데이터셋 중 4개에서 정확도가 향상되었습니다.



### Automatic speech recognition for the Nepali language using CNN, bidirectional LSTM and ResN (https://arxiv.org/abs/2406.17825)
Comments:
          Accepted at 2022 International Conference on Inventive Computation Technologies (ICICT), IEEE

- **What's New**: 이 논문은 네팔어 음성을 텍스트로 변환하는 자동 음성 인식(ASR) 기술을 위한 엔드투엔드(End-to-End) 딥러닝 모델을 제안합니다. 이 모델은 OpenSLR 데이터셋을 사용하여 학습되고 테스트되었습니다. 데이터셋 전처리 단계에서 오디오 파일의 양 끝에 있는 긴 공백을 제거하여 더 균일한 오디오 프레임과 텍스트의 매핑을 가능하게 했습니다.

- **Technical Details**: 오디오 특성으로 멜 주파수 켑스트럼 계수(MFCCs)를 사용하였으며, 모델은 양방향 LSTM (Bidirectional LSTM)과 ResNet, 1차원 CNN(Convolutional Neural Networks)을 조합하여 최고의 성능을 보였습니다. 이 모델은 학습 시 손실 계산을 위해 CTC(Connectionist Temporal Classification) 함수를 사용하며, 예측 시 CTC 빔 서치 디코딩(Beam Search Decoding)을 사용하여 네팔어 텍스트의 가장 가능성 높은 시퀀스를 출력합니다.

- **Performance Highlights**: 테스트 데이터셋에서 문자 오류율(CER)이 17.06%로 나타났습니다. 이는 기존에 학습된 다른 변형 모델들(LSTM, GRU, CNN, ResNet 등)보다 우수한 성과입니다. 소스 코드는 공개되어 있어 누구나 접근할 수 있습니다.



### Training-Free Exponential Extension of Sliding Window Context with Cascading KV Cach (https://arxiv.org/abs/2406.17808)
- **What's New**: 트랜스포머 모델의 컨텍스트 윈도우(context window)는 현재 작업의 활성 메모리 형태를 제공하여 few-shot 학습 및 조건부 생성에 유용합니다. 그러나 컨텍스트 길이가 증가하면 계산 비용은 기하급수적으로 증가합니다. 기존 연구들은 초기 몇 개의 토큰을 저장하고 고정 크기의 슬라이딩 윈도우(sliding window)를 사용하는 방법으로 이를 해결하려고 했으나, 이는 토큰을 최적화하지 못하고 무조건적으로 대기 키-값(KV) 캐시에서 모든 토큰을 제거하여 이후 예측에 영향을 미치지 못하게 했습니다. 이를 극복하기 위해, 우리는 별도의 캐스케이딩 서브 캐시(cascading sub-cache) 버퍼를 사용하여 보다 오래된 컨텍스트를 동일한 총 캐시 크기로 저장하는 새로운 메커니즘을 제안합니다.

- **Technical Details**: 기존의 슬라이딩 윈도우 KV 캐시 방식은 위치에 따라 토큰을 무조건적으로 제거하는 정적 고정 장치로, 토큰의 중요도에 관계없이 제거합니다. 우리 방법은 캐시를 여러 서브 캐시로 보며, 각각 다른 빈도로 토큰을 수용하고, 역사적 주의(attention) 점수를 근거로 하여 조건적으로 토큰을 제거합니다. 이는 지수 이동 평균(EMA)으로 추적되며, 이를 통해 더 오래된 중요한 토큰을 더 오랫동안 캐시에 유지할 수 있게 합니다.

- **Performance Highlights**: 같은 크기의 KV 캐시를 사용한 상태에서, 우리의 방법은 롱 컨텍스트 생성(LongBench)에서 5.6%, 스트리밍 퍼플렉시티(PG19)에서 1.2%, 다중 작업 언어 이해력(MMLU STEM)에서 0.6% 향상을 보였습니다. 또한, 우리 방법은 캐시 연산 지연 시간을 1.33ms에서 0.54ms로 59% 단축하였습니다.



### Enhancing Commentary Strategies for Imperfect Information Card Games: A Study of Large Language Models in Guandan Commentary (https://arxiv.org/abs/2406.17807)
- **What's New**: 최근 대형 언어 모델(LLMs)의 발전으로 고품질의 게임 해설을 생성하는 가능성이 열렸습니다. 하지만 불완전한 정보로 구성된 복잡한 게임에 대해 통찰력 있고 흥미로운 해설을 생성하는 것은 여전히 큰 도전 과제입니다. 이 논문에서는 중국 카드 게임 'Guandan'을 위해 강화 학습(RL)과 LLMs를 결합한 새로운 해설 방법을 소개합니다.

- **Technical Details**: 우리 시스템은 RL을 활용하여 복잡한 카드 플레이 시나리오를 생성하고, LLMs를 사용하여 해당 해설 텍스트를 생성합니다. 이는 전문 해설자의 전략적 분석과 서사적 능력을 모방하는 것을 목표로 합니다. 프레임워크는 '상태 해설 가이드(State Commentary Guide)', '마음 이론(ToM) 기반 전략 분석기', '스타일 검색 모듈(Style Retrieval Module)'로 구성되어 있으며, 이들 모듈이 협력하여 상세하고 맥락에 맞는 게임 해설을 제공합니다. ToM 기능을 통해 개인화된 해설 내용을 생성할 수 있도록 LLMs를 강화하며, 검색 및 정보 필터링 메커니즘도 개선합니다.

- **Performance Highlights**: 실험 결과, 제안된 해설 프레임워크가 오픈 소스 LLMs에 적용될 때 성능이 크게 향상됨을 보여줍니다. 여러 평가 지표에서 GPT-4의 성능을 뛰어넘었습니다.



### MOSSBench: Is Your Multimodal Language Model Oversensitive to Safe Queries? (https://arxiv.org/abs/2406.17806)
- **What's New**: 이번 연구는 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)들이 인간과 유사한 인지 왜곡(cognitive distortions)을 보인다는 것을 입증합니다. 구체적으로, MLLMs가 안전 메커니즘 아래에서 설계되었음에도 불구하고 무해한 쿼리를 특정 시각적 자극에 의해 거부하는 문제가 있습니다. 이를 체계적으로 평가하기 위해 MOSSBench라는 벤치마크 도구를 제안하며, 300개의 수집된 무해한 쿼리를 포함합니다.

- **Technical Details**: 세 가지 유형의 시각적 자극이 MLLMs의 과민 반응을 유발한다고 밝혔습니다: Exaggerated Risk (과장된 위험), Negated Harm (부정된 해악), Counterintuitive Interpretation (반직관적 해석). MOSSBench를 이용해 20개의 MLLMs를 평가했으며, 이 벤치마크는 시각적 자극에 대해 MLLMs의 과민성을 체계적으로 연구하기 위해 설계되었습니다.

- **Performance Highlights**: ['과민성은 주요 MLLMs에서 널리 퍼져 있으며, 무해한 쿼리에 대한 거부율이 최대 76%에 달합니다.', '안전성이 뛰어난 모델일수록 과민 반응을 더 많이 보입니다.', '특정 자극 유형이 모델의 응답 과정 중 특정 단계에서 오류를 유발합니다. 구체적으로 감각, 의도 추론, 안전 판단 단계에서 문제가 발생할 수 있습니다.']



### Can LLMs Generate Visualizations with Dataless Prompts? (https://arxiv.org/abs/2406.17805)
- **What's New**: 최근 대형 언어 모델(LLMs)에서 이루어진 발전이 정보 접근 방식을 혁신적으로 변화시키고 있습니다. 본 논문은 LLMs가 데이터 없이 주어진 쿼리에 대해 정확한 시각화를 제공할 수 있는 능력을 조사합니다. 특히 GPT-3와 GPT-4가 데이터 없는 프롬프트(dataless prompts)로 얼마나 적절한 시각화를 생성할 수 있는지를 평가했습니다.

- **Technical Details**: 본 연구는 자연어(NL) 쿼리를 기반으로 데이터 시각화를 생성하는 문제를 다룹니다. 초기 접근법은 전통적인 자연어 처리(NLP) 기술을 사용해 쿼리를 기계 명령어로 변환했으나, 최근에는 심층 학습(deep-learning) 기반 접근법이 주목받고 있습니다. GPT-3 및 GPT-4 모델은 데이터를 포함하지 않은 프롬프트만으로 시각화를 생성할 수 있는지를 테스트했습니다. 데이터가 없는 프롬프트 예시는 '지난 20년 동안의 미국 GDP를 보여주는 차트를 생성해주세요'와 같은 질문을 포함합니다.

- **Performance Highlights**: GPT-4는 프롬프트에 대해 데이터를 포함한 적절한 차트를 생성할 수 있었으며, 시각화 전문가의 지침에 따라 자주 사용되는 시각화 형식을 따랐습니다. 그러나 정확한 데이터 재현 면에서는 일부 차이를 보였습니다. 특히, 선형 차트의 경우 전체적인 경향은 같으나 세부 데이터 값에서는 불일치를 보이는 경우가 있었습니다. 반면, 막대 차트를 이용한 랭킹에서는 더 정확한 결과를 보였습니다.



### Understanding the Role of User Profile in the Personalization of Large Language Models (https://arxiv.org/abs/2406.17803)
- **What's New**: 본 연구는 사용자 프로필을 활용하여 대형 언어 모델(LLMs)을 개인화하는 방법을 탐구하며, 사용자 프로필이 LLM 개인화에 미치는 영향 메커니즘을 분석합니다. 주로 개인화 정보가 의미 정보보다 더 중요하며, 사용자가 생성하거나 승인한 역사적 개인화 응답이 중요한 역할을 한다는 것을 발견했습니다.

- **Technical Details**: 사용자 프로필을 통해 LLM의 성능을 개선하는 방법으로 다양한 강화 방법을 비교하였습니다. 또한, 사용자 프로필의 각 요소와 입력 맥락 내 위치가 개인화에 미치는 영향을 분석하였습니다. 실험 결과, 초기 부분에 배치된 사용자 프로필이 개인화에 더 큰 영향을 미치는 것을 확인했습니다.

- **Performance Highlights**: 개인화 정보가 포함된 사용자 프로필을 활용했을 때, 의미 정보 단독으로 개선되는 경우보다 성능이 더 우수했습니다. 또한, 사용자 응답이 실제 사용자에 의해 생성되거나 승인된 경우, 개인화 효과가 더욱 극대화되었습니다.



### Mashee at SemEval-2024 Task 8: The Impact of Samples Quality on the Performance of In-Context Learning for Machine Text Classification (https://arxiv.org/abs/2406.17790)
- **What's New**: 본 연구는 인컨텍스트 러닝(In-Context Learning, ICL)에서 샘플 품질 향상을 통해 성능을 극대화하는 방법을 제안합니다. 이를 위해 카이-스퀘어(chi-square) 테스트를 사용하여 고품질 샘플을 선별하고, 해당 샘플을 이용한 모델의 성능을 평가합니다. 실험 결과, 고품질 샘플을 활용할 경우 모든 평가 지표에서 성능이 향상됨을 확인했습니다.

- **Technical Details**: 카이-스퀘어(chi-square)는 두 범주형 변수의 독립성을 평가하기 위한 통계 테스트로, 텍스트 분석에서 중요한 특징을 갖는 키워드를 선별하는 데 유용합니다. 본 연구에서는 Flan-T5 모델과 결합하여 인컨텍스트 러닝을 수행하였으며, 고차원 샘플과 저차원 샘플을 비교하여 성능을 분석했습니다. 데이터셋은 인간이 작성한 텍스트와 기계가 생성한 텍스트 두 가지로 구성되었습니다.

- **Performance Highlights**: 고품질 샘플을 선택하여 인컨텍스트 러닝을 수행하면 모델의 성능이 현저히 향상됨을 발견했습니다. 특히, 카이-스퀘어 값이 높은 샘플을 사용하여 학습한 경우, 정확도(accuracy), 재현율(recall), 정밀도(precision), F1 점수(F1-score) 모두에서 저차원 샘플을 사용한 경우보다 더 높은 성과를 보였습니다.



### Spanish and LLM Benchmarks: is MMLU Lost in Translation? (https://arxiv.org/abs/2406.17789)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)을 다양한 업무와 주제에서 평가하는 방법에 대해 설명하고 있습니다. 특히, 영어 외 다른 언어에서 LLM을 평가할 필요성이 강조되고 있습니다. 대부분의 LLM 벤치마크는 자동 번역 도구를 사용하여 다른 언어로 번역된 다음 해당 언어에서 실행됩니다. 그러나 번역 품질이 결과에 영향을 줄 수 있다는 문제가 있습니다. 이 연구에서는 MMLU (Massive Multitask Language Understanding) 벤치마크를 스페인어로 번역하고 번역 오류가 LLM 성능 평가에 미치는 영향을 분석합니다.

- **Technical Details**: 선택된 벤치마크 항목을 Azure Translator와 ChatGPT4를 사용하여 스페인어로 번역한 다음, 스페인어와 영어 번역에서 다른 응답을 생성하는 테스트 항목을 식별합니다. 그런 다음 이 항목들을 수동으로 분석하여 자동 번역이 결과에 어떻게 영향을 미치는지 파악합니다. 구체적으로 Miscellaneous, Philosophy, US foreign policy 세 가지 범주를 선택하여 783개, 311개, 100개의 질문을 분석합니다. GPT4 모델이 이 질문들에 대한 응답을 제공하며, 번역 오류로 인해 성능이 왜곡되는지 평가합니다.

- **Performance Highlights**: 실험 결과, 번역 오류로 인해 상당 부분의 잘못된 응답이 발생하는 것으로 나타났습니다. 이는 번역 품질이 LLM 성능 평가에 중요한 영향을 미칠 수 있음을 보여줍니다. 따라서, 영어 이외의 언어로 작성된 벤치마크는 번역 품질을 개선하거나 해당 언어에 맞게 테스트를 전문가가 적응시키는 방식으로 개선될 필요가 있습니다.



### Role of Dependency Distance in Text Simplification: A Human vs ChatGPT Simplification Comparison (https://arxiv.org/abs/2406.17787)
- **What's New**: 본 연구는 인간과 ChatGPT의 텍스트 단순화 작업을 조사하고, 이와 문장의 의존 거리(dependency distance) 관계를 분석하였습니다. 이전 사용자 연구에서 문법적 난이도가 높아지는 것으로 측정된 220개의 문장을 대상으로, 인간 전문가와 ChatGPT를 이용해 단순화를 시도했습니다.

- **Technical Details**: 문장 세트는 각각 평균 의존 거리에서 차이를 나타냈습니다. 원래의 문장 세트가 가장 높은 평균 의존 거리를 보였으며, 다음으로 ChatGPT가 단순화한 문장이 그 뒤를 이었습니다. 인간이 단순화한 문장은 가장 낮은 평균 의존 거리를 보였습니다.

- **Performance Highlights**: 이 연구는 인간 전문가가 참여한 문장 단순화 작업이 ChatGPT에 비해 더 효과적으로 문장의 복잡도를 낮출 수 있음을 시사합니다. 즉, 인간이 단순화한 문장은 의존 거리가 짧아 독해 난이도가 낮아지는 경향이 있음을 확인할 수 있습니다.



### ChronoMagic-Bench: A Benchmark for Metamorphic Evaluation of Text-to-Time-lapse Video Generation (https://arxiv.org/abs/2406.18522)
Comments:
          31 pages, 15 figures

- **What's New**: 새로운 텍스트-비디오 (Text-to-Video, T2V) 생성 벤치마크인 ChronoMagic-Bench를 제안했으며, 이는 T2V 모델의 시간 경과 비디오 생성에서의 시간적 일관성과 변형 능력을 평가합니다. 기존의 벤치마크는 시각적 품질과 텍스트의 관련성에 초점을 맞추지만, ChronoMagic-Bench는 모델이 물리, 생물 및 화학적 자유형 텍스트 쿼리를 통해 시간 경과 비디오를 생성하는 능력을 평가합니다.

- **Technical Details**: ChronoMagic-Bench는 1,649개의 프롬프트와 실세계 비디오를 참조로 제공하며, 이를 생물학적, 인간이 만든 것, 기상학적, 물리적 현상으로 분류합니다. 이러한 분류는 모델의 다양하고 복잡한 변환 처리를 종합적으로 평가합니다. MTScore와 CHScore 두 가지 새로운 자동 측정 지표를 도입하여 비디오의 변형 속성과 시간적 일관성을 평가합니다.

- **Performance Highlights**: ChronoMagic-Bench를 기반으로 열 개의 대표적인 T2V 모델에 대한 종합적 평가를 수행한 결과, 대부분의 모델이 대규모 변화를 갖는 시간 경과 비디오를 생성하지 못하며, 프롬프트에 부합하지 않는 경향이 높고, 단일 프레임의 시각적 품질이 높더라도 눈에 띄는 깜박임 문제를 가지고 있다고 밝혔습니다. 이를 해결하기 위해 고해상도 타임랩스 비디오와 상세한 캡션으로 구성된 대규모 ChronoMagic-Pro 데이터셋을 만들었습니다.



### Mental Modeling of Reinforcement Learning Agents by Language Models (https://arxiv.org/abs/2406.18505)
Comments:
this https URL

- **What's New**: 이 연구는 대형 언어 모델(Large Language Models, LLMs)이 RL(강화 학습) 에이전트의 행동을 이해하고, 이를 해석할 수 있는 능력을 처음으로 경험적으로 검증합니다. 특히, LLM이 에이전트의 상호작용 이력을 통해 에이전트의 행동을 추론하고 그 결과로 나타나는 상태 변화를 이해할 수 있는지를 탐구합니다. 이러한 연구는 설명 가능한 강화 학습(eXplainable Reinforcement Learning, XRL)에 대한 중요한 도전에 대응할 수 있는 가능성을 열어줍니다.

- **Technical Details**: 이 연구에서는 에이전트의 상호작용 이력을 통해 에이전트의 행동과 그 결과로 나타나는 상태 변화를 이해하는 능력을 '에이전트 정신 모델링 (Agent Mental Modelling)'이라고 정의합니다. LLM들의 이러한 능력을 검증하기 위해서, 다양한 복잡성을 지닌 RL 작업 데이터셋에 대해 특정 평가 기준을 제안하고 테스트하였습니다. 결과는 LLM들이 단순한 추론만으로는 완전한 정신 모델링을 위한 충분한 능력을 갖추지 못했다는 것을 나타냈습니다.

- **Performance Highlights**: 연구 결과, LLM들이 에이전트 정신 모델을 구축하는 능력은 아직 충분하지 않다는 결론을 내렸습니다. 특히, 현재의 모델들이 물리적 세계에서의 에이전트 행동을 비추어 볼 때 줄 수 있는 한계와 가능성을 밝혔습니다. 이 연구는 LLM의 현재 한계를 규명하고, Agent Mental Modelling을 위해 더 많은 혁신이 필요함을 보여줍니다.



### Advancing Airport Tower Command Recognition: Integrating Squeeze-and-Excitation and Broadcasted Residual Learning (https://arxiv.org/abs/2406.18313)
Comments:
          Accepted by IALP 2024

- **What's New**: 이번 연구는 소음이 많은 환경과 제한된 계산 자원이라는 문제를 극복하기 위해 새로운 키워드 감지 기술을 제안합니다. 표준화된 공항 타워 명령 데이터셋을 생성하고, squeeze-and-excitation(SE) 기술과 time-frame frequency-wise squeeze-and-excitation(tfwSE) 기술을 사용하여 BC-SENet 모델을 개발했습니다. 이 모델은 공항 타워 명령을 보다 정확하고 효율적으로 인식할 수 있습니다.

- **Technical Details**: BC-SENet 모델은 broadcasted residual learning 네트워크에 squeeze-and-excitation(SE) 및 time-frame frequency-wise squeeze-and-excitation(tfwSE) 기술을 접목하여 고안되었습니다. 이 모델은 5×5 2D convolution 레이어로 시작하여 batch normalization과 ReLU activation을 통해 초기 특징을 추출합니다. 그런 다음 depthwise separable convolutions와 함께 사용되는 Broadcasted Residual Blocks(BC-ResBlock)을 통해 복잡한 특징을 처리합니다. Squeeze-and-excitation 모듈이 동적으로 채널의 중요성을 조정하여 특징을 재배치합니다. 이후 depthwise separable convolutions와 1×1 convolutions가 사용되어 특징을 재조합하고 채널을 조정합니다. 마지막으로 average pooling을 통해 특징 차원을 줄이고 1×1 convolution을 통해 최종 분류 작업에 적합한 출력 레이어를 구성합니다.

- **Performance Highlights**: BC-SENet 모델을 포함한 다섯 가지 키워드 감지 모델을 테스트한 결과, BC-SENet이 뛰어난 정확도와 효율성을 보여주었습니다. 소음이 많은 환경에서도 명령 인식 성능이 개선되어 항공 안전과 효율성에 기여할 수 있음을 확인했습니다. 또한, BC-SENet은 Google Speech Command 데이터셋에서도 비슷한 성능을 보였습니다.



### MSR-86K: An Evolving, Multilingual Corpus with 86,300 Hours of Transcribed Audio for Speech Recognition Research (https://arxiv.org/abs/2406.18301)
Comments:
          Accepted by InterSpeech 2024

- **What's New**: 최근에는 ChatGPT와 같은 다국어 인공지능 보조 시스템이 큰 인기를 끌고 있습니다. 이에 따라 다국어 자동 음성 인식(ASR)도 주목받고 있으며, Whisper와 같은 시스템이 그 예입니다. 그러나 독점적인 학습 데이터 때문에 연구자들이 다국어 ASR을 연구하는데 어려움을 겪고 있습니다. 이번 논문에서는 MSR-86K라는 대규모 다국어 음성 인식 연구를 위한 코퍼스를 소개합니다. 이 코퍼스는 유튜브의 공개된 비디오에서 수집된 15개 언어, 총 86,300시간의 데이터로 구성되어 있습니다. MSR-86K는 HuggingFace에서 공개될 예정이며, 다국어 ASR 연구의 새로운 지평을 열 것으로 기대됩니다.

- **Technical Details**: MSR-86K 코퍼스는 다국어 ASR 시스템을 구축하기 위한 대규모 데이터세트로, 유튜브 비디오에서 자동으로 수집한 데이터를 바탕으로 합니다. 주요 언어별로 충분한 데이터를 제공하면서 기존 코퍼스의 언어적 다양성 부족 문제를 해결하고자 했습니다. 데이터 수집 및 처리 과정에서는 키워드 리스트 생성, 비디오 ID 검색 및 필터링, 오디오 및 자막 다운로드, 텍스트 정규화, 강제 정렬, 언어 식별(LID) 필터링, ASR 필터링 등을 거쳐 고품질의 데이터를 확보했습니다. 이후, MSR-86K와 기존 오픈소스 데이터를 활용해 Whisper보다 더 빠르고, 메모리 부담이 적으며, 성능이 뛰어난 다국어 ASR 모델을 구축할 수 있음을 보였습니다.

- **Performance Highlights**: MSR-86K의 품질을 평가하기 위해 각 언어별로 단일언어 모델을 학습시켰습니다. 개발 세트에서는 Beam Search 디코딩을 사용해 단어 오류율(WER)과 문자 오류율(CER)을 계산했습니다. Transformer-CTC 아키텍처를 이용하여 높은 품질의 ASR 모델을 구축하였으며, Whisper large 모델과 비교해 WER와 CER 측면에서 더 뛰어난 성능을 보였습니다.



### GUIDE: A Guideline-Guided Dataset for Instructional Video Comprehension (https://arxiv.org/abs/2406.18227)
Comments:
          IJCAI 2024

- **What's New**: GUIDE 데이터셋은 기존 학습 비디오 데이터셋의 한계를 극복하기 위해 만들어졌습니다. 이는 8개의 일상 생활과 관련된 분야에서 560개의 학습 과제와 3.5천 개의 비디오를 포함합니다. GUIDE는 각 학습 과제에 공동 패턴을 나타내는 지침을 주석으로 추가하여, 초보자들이 더 효과적으로 학습할 수 있도록 도와줍니다.

- **Technical Details**: GUIDE 데이터셋은 세 가지 주요 서브 태스크(sub-tasks)를 평가하는 벤치마크를 포함하고 있습니다. (1) 'Step Captioning'에서는 모델이 비디오에서 특정 단계를 설명하는 캡션을 생성해야 합니다. (2) 'Guideline Summarization'에서는 모델이 과제 관련 비디오에서 공통 패턴을 추출하고 이를 요약하는 지침을 생성해야 합니다. (3) 'Guideline-Guided Captioning'에서는 지침을 바탕으로 특정 단계에 대한 설명을 생성해야 합니다. 이러한 태스크를 통해 모델의 이해 능력을 평가할 수 있습니다.

- **Performance Highlights**: 다양성과 실용성을 고려하여 GUIDE 데이터셋은 학습 비디오 이해를 위한 더 나은 벤치마크로 사용될 수 있습니다. 여러 기초 모델(foundation models)을 GUIDE 데이터셋을 통해 평가하고 깊이 있는 분석을 수행함으로써, 데이터셋의 유용성과 성능을 입증했습니다.



### SafeAligner: Safety Alignment against Jailbreak Attacks via Response Disparity Guidanc (https://arxiv.org/abs/2406.18118)
- **What's New**: 새로운 연구로, 대형 언어 모델(LLMs)을 안전하게 보호하면서 유틸리티를 유지하는 문제를 다루는 SafeAligner 방법론이 소개되었습니다. 이는 주로 보안 프로토콜을 우회하려는 시도인 jailbreak 공격에 대한 방어 전략을 강화하는 데 중점을 두고 있습니다.

- **Technical Details**: SafeAligner는 디코딩 단계에서 구현되는 방법론으로, 두 가지 모델을 개발하여 운영됩니다. Sentinel Model은 안전성을 증진시키기 위해 훈련된 모델이고, Intruder Model은 더 위험한 응답을 생성하는 모델입니다. SafeAligner는 이 두 모델의 응답 간의 보안 수준 차이를 활용하여 유해한 토큰과 유익한 토큰을 구별하고, 목표 모델의 출력 토큰 분포를 변경함으로써 안전한 정렬을 유도합니다.

- **Performance Highlights**: 광범위한 실험을 통해 SafeAligner가 유익한 토큰의 발생 가능성을 높이고 유해한 토큰의 발생을 줄일 수 있음이 입증되었습니다. 이를 통해 일반성을 거의 잃지 않으면서도 안전한 정렬을 보장합니다.



### EHR-Based Mobile and Web Platform for Chronic Disease Risk Prediction Using Large Language Multimodal Models (https://arxiv.org/abs/2406.18087)
- **What's New**: 새로운 연구에서는 대만 병원의 5년간의 전자 건강 기록(Electronic Health Records, EHR)을 활용하여 만성 질병 예측 플랫폼을 개발했습니다. 이 플랫폼은 대규모 언어 멀티모달 모델(Large Language Multimodal Models, LLMMs)을 활용하여 임상 노트와 혈액 검사 값을 통해 여러 만성 질병을 예측하고, 병원의 백엔드 데이터베이스와 실시간으로 연결되어 의사들에게 실시간 위험 평가 진단을 제공합니다.

- **Technical Details**: 연구팀은 2017년부터 2021년까지 대만의 Far Eastern Memorial Hospital에서 1,420,596개의 임상 노트와 387,392개의 실험실 결과 데이터를 수집했습니다. 이 데이터는 언어 모델 BERT, BiomedBERT, Flan-T5-large-770M, GPT-2을 사용하여 텍스트 특징을 추출하고, Attention 모듈을 통해 융합하여 최종 예측을 진행했습니다. 혈액 검사 데이터는 Deep Neural Network(DNN)을 이용하여 통합하였으며, 최종적으로 Multi-Head Attention Layer를 통해 두 모달리티에서 얻은 임베딩을 통합하였습니다.

- **Performance Highlights**: 각 질병에 대한 성능 평가 결과, 높은 긍정 비율의 질병(예: 당뇨병)에서는 GPT-2를 사용한 LLMM의 조합이 정밀도 0.70, 재현률 0.71, F1 점수 0.70을 기록했습니다. 심장병 예측에서는 GPT-2가 정밀도 0.81, 재현률 0.85, F1 점수 0.83으로 특히 높은 성능을 보였습니다. BiomedBERT 모델은 고혈압 예측에서 정밀도 0.35를 기록하여 긍정도가 낮은 클래스에서도 더 나은 성능을 보였습니다.



### Large Language Models for Cuffless Blood Pressure Measurement From Wearable Biosignals (https://arxiv.org/abs/2406.18069)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)이 착용형 바이오신호를 분석하여 커프 없이 혈압(BP)을 측정할 수 있는 가능성을 처음으로 탐구하는 연구입니다. 저자들은 심혈관 질환 관리에 중요한 커프 없는 BP 측정을 위해 LLMs를 활용할 방법을 제안합니다.

- **Technical Details**: 연구진은 심전도(ECG)와 광용적맥파(PPG) 신호에서 생리학적 특징을 추출하여, BP 분야의 지식과 사용자 정보를 결합한 컨텍스트 강화 프롬프트를 설계했습니다. 이를 통해 LLMs를 BP 측정 과제에 맞게 조정(instruction tuning)했습니다. 1,272명의 참가자로부터 수집된 대규모 공개 데이터셋을 이용해 10개의 최신 LLMs를 평가하였습니다.

- **Performance Highlights**: 최적화된 LLM은 기존의 특정 과제 전용 기법들보다 우수한 성능을 보여주었으며, 수축기 혈압(SBP) 측정의 경우 오차가 0.00 ± 9.25 mmHg, 이완기 혈압(DBP) 측정의 경우 오차가 1.29 ± 6.37 mmHg였습니다. 또한, 프롬프트의 컨텍스트 강화 전략이 SBP 측정의 평균 절대 오차를 8.9% 감소시키는 데 기여했습니다.



### LABOR-LLM: Language-Based Occupational Representations with Large Language Models (https://arxiv.org/abs/2406.17972)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)을 활용하여 직업 이동 예측 문제를 해결하기 위해 새로운 접근법을 제안합니다. 기존 CAREER 모델 대신 LLMs을 미세 조정(fine-tuning)하여 더 나은 예측 성능을 보여줍니다. 본 연구에서는 이러한 새로운 접근법이 전통적인 경제모형, CAREER 모델, 그리고 컨텍스트 학습(context learning) 기반의 LLM보다 더 우수한 예측 성능을 보인다는 것을 입증했습니다.

- **Technical Details**: LABOR-LLM이라는 새로운 프레임워크를 통해 LLMs이 노동 시장 데이터를 모델링하고 대표성 있는 예측을 생성하도록 합니다. 이 프레임워크는 복잡한 접근법을 포함하여 LLM을 미세 조정하거나 임베딩을 추출한 후 다항 분류기(multinomial classifier) 모델에 통합하여 다음 직업 선택을 예측합니다. 이러한 방법은 일반 LLM의 한계를 극복하고 더 대표성 있는 예측을 할 수 있도록 합니다.

- **Performance Highlights**: 제안된 LLM 기반 모델은 전통적인 경제모형과 최신 CAREER 모델을 포함한 여러 대안들에 비해 예측 성능이 뛰어났습니다. 또한 제안된 모델은 특정 인구 통계 하위 그룹의 직업 경로를 더 잘 반영하였습니다. 이는 LLM이 일반 텍스트 데이터를 활용해 광범위한 노동 시장 정보를 포착할 수 있음을 보여줍니다.



### ET tu, CLIP? Addressing Common Object Errors for Unseen Environments (https://arxiv.org/abs/2406.17876)
- **What's New**: 이번 연구에서는 사전 학습된 CLIP 인코더를 사용하여 ALFRED 과제에서 모델 일반화를 향상시키는 간단한 방법을 소개합니다. 기존 문헌과 달리, 우리는 CLIP을 시각적 인코더 대신에 추가 모듈로 사용하여 보조 객체 탐지 목적을 통해 성능을 개선하고자 합니다. 이러한 방법을 통해 보이지 않는 검증 세트에서의 과제 성능이 향상됨을 입증하였습니다.

- **Technical Details**: 본 논문에서는 준거형 변환기(Episodic Transformer, ET) 아키텍처를 기반으로 한 초기 실험에서 제안된 방법을 조사했습니다. CLIP 모듈과 ET는 훈련 중 객체 예측 손실을 계산하고, 추론 시 CLIP 모듈은 무시되며 객체 예측은 ET에 의해 단독으로 수행됩니다. 이 방법으로 희소한 단어와 작은 객체 탐지 및 객체 설명 활용이 특히 개선됨을 확인했습니다.

- **Performance Highlights**: 실험 결과, ET-CLIP 모델이 보이지 않는 장면에서 더 나은 성능을 보였습니다. 구체적으로, 세세한 객체 속성, 작은 객체 및 희소 단어가 포함된 지시문에서 목표 조건 성공률이 크게 향상되었습니다. 예를 들어, '빨간 의자로 걸어가세요'와 같은 지시문에서 목표 조건 성공률이 0.3% 증가했습니다. 또한, 작은 객체와 관련된 지시문에서도 목표 조건 성공률이 0.5% 증가했습니다. 희소 의미를 포함한 지시문에서 목표 조건 성공률이 0.8% 증가했습니다.



### A multi-speaker multi-lingual voice cloning system based on vits2 for limmits 2024 challeng (https://arxiv.org/abs/2406.17801)
- **What's New**: LIMMITS'24 챌린지 참여를 위한 다중 화자, 다중 언어 음성 합성 시스템이 개발되었습니다. 이번 챌린지는 7개의 인도 언어와 남녀 화자를 대상으로 음성 클로닝(voice cloning) 기능을 갖춘 텍스트-투-스피치(TTS) 시스템을 구축하는 것을 목표로 합니다. 특히, 우리의 시스템은 VITS2 아키텍처에 다중 언어 ID와 BERT 모델을 추가하여 언어 이해력을 향상시켰습니다.

- **Technical Details**: 우리의 시스템은 음소(phoneme)를 입력값으로 사용하며, espeak 도구를 통해 벵골어(Bengali), 칸나다어(Kannada), 텔루구어(Telugu), 힌디어(Hindi), 마라티어(Marathi)를 국제 음성 기호(IPA)로 변환합니다. Chhattisgarhi 언어는 espeak에서 지원되지 않아, Hindi의 IPA로 변환하는 대체 방법을 사용했습니다. 텍스트 인코더 전에 IndicBERT를 통합하여 문맥적 이해와 운율 성능을 향상시켰습니다. 이 모델은 8개의 NVIDIA RTX 4090 GPU를 사용하여 학습되었습니다.

- **Performance Highlights**: Track 1에서 우리의 모델은 고유의 추가 데이터 없이 Speaker Similarity 점수 4.02를 획득했습니다. Track 2에서는 추가 데이터를 활용하여 Speaker Similarity 점수 4.17을 기록, 이 부문에서 1위를 차지했습니다. 전체적으로, 자연스러움과 화자 유사성에서 높은 평가를 받았으며, 두 트랙 모두에서 우수한 성능을 보여줬습니다.



### Deep Learning Approaches for Detecting Adversarial Cyberbullying and Hate Speech in Social Networks (https://arxiv.org/abs/2406.17793)
Comments:
          10 pages, 8 figures, 3 tables, under reviewing

- **What's New**: 이 논문은 소셜 네트워킹 사이트의 텍스트 데이터에서 사이버 괴롭힘을 탐지하는 데 초점을 맞추고 있습니다. 특히 적대적 공격 콘텐츠에서 혐오 발언을 강조합니다. 딥러닝 기반 접근 방식과 보정 알고리즘을 사용하여 중요한 결과를 도출했습니다. 이 논문에서는 100에포크 고정된 LSTM 모델을 활용하여 높은 정확도, 정밀도, 재현율, F1 점수, AUC-ROC 점수를 기록했습니다.

- **Technical Details**: 이 논문은 딥러닝 기반의 LSTM 모델을 사용하여 SNS에서의 사이버 괴롭힘과 혐오 발언을 탐지합니다. 특히, 적대적 공격(adversarial attacks)의 예시들을 탐지하고 대응하는 기법에 집중합니다. 100에포크 에서 LSTM 모델을 훈련시켰으며, 이 모델은 높은 성능을 보였습니다. 또한, 자연어 처리(NLP) 알고리즘 및 어노테이션 데이터(annotated data)를 활용하여 사이버 괴롭힘 콘텐츠를 실시간으로 자동 탐지하는 연구를 다룹니다.

- **Performance Highlights**: LSTM 모델은 높은 성능을 보여주었으며, 정확도(accuracy) 87.57%, 정밀도(precision) 88.73%, 재현율(recall) 87.57%, F1 점수 88.15%, AUC-ROC 점수 91%를 기록하였습니다. 이는 이전 연구들보다 뛰어난 성능을 보였으며, 다양한 플랫폼에서 효율적으로 구현할 수 있음을 입증했습니다.



### OmAgent: A Multi-modal Agent Framework for Complex Video Understanding with Task Divide-and-Conquer (https://arxiv.org/abs/2406.16620)
- **What's New**: OmAgent는 대규모 언어 모델(LLM)의 멀티모달(multimodal) 컨텍스트 확장을 통해 동영상 데이터를 이해하는 혁신적인 방법을 제시합니다. OmAgent는 고유한 Divide-and-Conquer Loop(DnC Loop)를 사용해 자율적으로 API와 도구를 호출하여 질의 처리와 정확성을 향상시킵니다. 이를 통해 24시간 CCTV 영상이나 장편 영화를 처리하는 데 필요한 정보 손실을 크게 줄일 수 있습니다.

- **Technical Details**: OmAgent는 두 가지 주요 구성 요소로 구성됩니다: (1) Video2RAG 비디오 전처리기로, 비디오에서 일반화된 정보를 추출해 저장합니다. (2) Divide-and-Conquer Loop(DnC Loop)를 통해 작업 계획과 실행을 수행하며, 도구 호출 기능을 갖추고 있습니다. 비디오 데이터를 문맥에 맞게 검색하고 처리하는 과정에서 정보 손실을 최소화합니다.

- **Performance Highlights**: 실험 결과, OmAgent는 다양한 비디오와 복잡한 작업을 처리하는 데 있어 높은 효율성을 입증했습니다. 기존 벤치마크를 초과하는 성과를 보였으며, 2000개 이상의 Q&A 쌍을 포함하는 새로운 복잡한 비디오 이해 벤치마크 데이터셋을 제안했습니다. OmAgent의 도입으로 비디오 이해 및 질문 응답 시스템의 성능과 정확성이 크게 향상됩니다.



### Layer-Wise Quantization: A Pragmatic and Effective Method for Quantizing LLMs Beyond Integer Bit-Levels (https://arxiv.org/abs/2406.17415)
Comments:
          submitted to EMNLP, 15 pages, 10 figures, 4 tables

- **What's New**: 새로운 연구에서는 대형 언어 모델(LLM)의 다양한 레이어를 서로 다른 비트 수준으로 양자화하는 간단한 방법을 제안합니다. 중요한 레이어는 더 높은 비트 해상도로, 덜 중요한 레이어는 더 낮은 비트로 양자화하여 모델의 크기를 대폭 줄이면서 성능 저하를 최소화할 수 있습니다.

- **Technical Details**: 이 연구는 레이어의 중요도를 측정하는 두 가지 효과적인 전략을 제안합니다. 첫 번째는 출력 임베딩과 입력 임베딩 간의 차이에 기반하여 중요도를 평가하며, 두 번째는 레이어의 가중치가 평균보다 큰 수준을 평가합니다. 이 두 가지 중요도 점수를 기반으로 레이어를 서로 다른 비트로 양자화하여 메모리 요구 사항에 맞는 모델 압축을 가능하게 합니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안된 순서를 통해 25-50%의 레이어를 더 낮은 양자화로 전환할 경우 원본 모델과 거의 유사한 성능을 유지할 수 있습니다. 또한, 낮은 비트로 양자화된 LLM은 프루닝보다 훨씬 더 나은 성능을 보이며, 특히 더 큰 모델에서는 효과가 더욱 두드러집니다. 코드와 실험에 대한 자세한 내용은 제공된 URL에서 확인할 수 있습니다.



### SetBERT: Enhancing Retrieval Performance for Boolean Logic and Set Operation Queries (https://arxiv.org/abs/2406.17282)
Comments:
          10 pages, 1 figure

- **What's New**: SetBERT는 논리 연산과 Boolean 쿼리(Intersection, Difference, Union)를 위한 쿼리 임베딩을 향상시키기 위해 설계된 BERT 기반의 파인튜닝된 모델입니다. SetBERT는 기존의 전통적 및 신경망 기반 검색 방법에 비해 논리 구조화된 쿼리에 대해 검색 성능을 크게 향상시켰습니다.

- **Technical Details**: SetBERT는 inversed-contrastive loss(역전된 대조 손실)을 사용하여 음성 문장을 식별하고 prompt GPT(GPT 기반 프롬프트)를 통해 생성된 데이터셋을 활용하여 BERT를 파인튜닝합니다. 기존의 BERT 모델과 달리, triplet loss(삼자 손실)을 사용하는 것은 이 특정 작업의 성능을 저하시킵니다. SetBERT는 논리 구조화된 쿼리에서 BERT 모델의 검색 성능을 크게 향상시킵니다.

- **Performance Highlights**: 실험 결과에 따르면, SetBERT-base는 BERT-base보다 호출율(Recall)에서 최대 63% 향상된 성능을 보여주며, 크기는 1/3에 불과하지만 BERT-large 모델과 비교할만한 성능을 달성했습니다. 이는 SetBERT의 효율성과 성능 면에서 큰 잠재력을 보여줍니다.

- **Applications**: SetBERT는 사용자 요구 사항이 매우 구체적이거나 특히 복잡한 경우, 예를 들어, '전쟁에 관한 것이 아닌 베트남 영화' 또는 '브라질과 멕시코에만 서식하는 동물'과 같은 쿼리에서 매우 유용할 수 있습니다. 이러한 Boolean 쿼리를 통해 사용자는 복잡한 검색 과정을 간소화할 수 있습니다.



### LLMs Assist NLP Researchers: Critique Paper (Meta-)Reviewing (https://arxiv.org/abs/2406.16253)
- **What's New**: 새로운 연구는 대형 언어 모델(Large Language Models, LLMs)이 NLP 연구자들을 어떻게 도울 수 있는지를 탐구하고 있습니다. 특히, 논문 검토와 메타 리뷰(meta-review)에서 LLMs의 효과성을 평가하고 있습니다. 이를 위해, 연구팀은 ReviewCritique 데이터셋을 구축하였으며, 이 데이터셋에는 사람과 LLM이 작성한 리뷰와 각 리뷰 섹션에 대한 전문가 주석이 포함되어 있습니다.

- **Technical Details**: ReviewCritique 데이터셋에는 두 가지 주요 구성 요소가 포함됩니다: (i) NLP 논문 (초기 제출본)과 사람이 작성한 리뷰, 그리고 LLM이 생성한 리뷰, (ii) 각 리뷰는 전문가가 '결함(deficiency)' 레이블과 해당 섹션에 대한 설명을 붙인 주석 데이터. 연구팀은 두 가지 연구 질문에 집중하였습니다: 'LLMs as Reviewers'에서 사람과 LLM이 작성한 리뷰의 질과 구별 가능성을 비교하고, 'LLMs as Metareviewers'에서 LLM이 개별 리뷰의 문제점을 얼마나 효과적으로 식별하는지를 평가했습니다.

- **Performance Highlights**: LLMs가 생성한 리뷰는 종종 사람의 리뷰보다 결함이 많고 논문과 관련성이 낮은 것으로 나타났습니다. 또한, LLM들은 사람 전문가와 비교할 때 리뷰의 문제점을 식별하고 설명하는 데 어려움을 겪었습니다. 연구의 주요 공헌으로는 (i) ReviewCritique 데이터셋 제공, (ii) 사람과 LLM이 작성한 리뷰의 문장 수준 비교, (iii) LLM의 리뷰어 및 메타 리뷰어 역할 분석이 있습니다.



New uploads on arXiv(cs.IR)

### UniRec: A Dual Enhancement of Uniformity and Frequency in Sequential Recommendations (https://arxiv.org/abs/2406.18470)
Comments:
          15 pages, 8 figures, for source code, see this https URL

- **What's New**: 새로운 논문에서는 UniRec이라는 새로운 양방향 향상 시퀀스 추천 방법을 제안합니다. 이 방법은 시퀀스 균일성(uniformity)과 아이템 빈도(frequency)를 활용하여 추천 성능을 향상시키며, 특히 비균일 시퀀스와 덜 빈번한 아이템의 표현력을 개선하는데 중점을 둡니다. 이 두 가지 요소는 서로 강화하여 복잡한 시퀀스 추천 시나리오에서 종합적인 성능 최적화를 이끕니다. 또한, 다차원 시간 모듈(multidimensional time module)을 제안하여 적응성을 더욱 향상시켰습니다.

- **Technical Details**: UniRec은 시퀀스 균일성 및 아이템 빈도를 활용하여 성능을 높이는 새로운 양방향 접근법입니다. 시퀀스에서는 비균일 시퀀스를 모사하기 위해 덜 빈번한 아이템을 포함하여 균일 시퀀스로부터 하위 집합을 생성하며, 이후 비균일 시퀀스 표현을 개선합니다. 아이템에서는 빈번한 아이템에 대한 이웃 집합 메커니즘(neighbor aggregation mechanism)을 덜 빈번한 아이템으로 확장하여 그 표현을 향상시키고, 이 지식을 시퀀스 모델링으로 이전합니다.

- **Performance Highlights**: 11개의 최신 모델과 4개의 데이터셋을 비교한 결과, UniRec은 SOTA(State-Of-The-Art) 모델들보다 우수한 성능을 입증했습니다. 특히, 6개의 최첨단 모델을 포함하여 시간 모델링을 통합한 11개의 경쟁 모델에 비해 성능이 크게 향상된 것을 확인할 수 있습니다.



### The Effects of Data Split Strategies on the Offline Experiments for CTR Prediction (https://arxiv.org/abs/2406.18320)
- **What's New**: 이번 연구는 CTR 예측에서 랜덤 데이터 분할과 시간 순서 기반 분할의 효과를 비교하여 모델 평가 방법론 간의 비일관성을 해결하는 것을 목표로 합니다. 기존 연구에서는 데이터 분할 전략의 차이가 실제 성능에 미치는 영향을 충분히 탐구하지 않았으나, 본 연구는 이를 집중적으로 탐구합니다.

- **Technical Details**: CTR(Click-through Rate) 예측은 사용자와 항목 간의 상호작용을 포착하여 제품 추천의 정확성을 높이는 데 중점을 둡니다. 기존 방법론은 주로 랜덤 데이터 분할을 사용하여 학습, 검증 및 테스트 데이터를 생성하지만, 실제 CTR 예측은 시간 순서에 따릅니다. 이번 연구는 랜덤 분할과 시간 순서 분할의 영향을 대규모 공개 벤치마크 데이터셋 Criteo를 통해 실험합니다.

- **Performance Highlights**: 12개의 최신 deep-CTR 모델을 랜덤 분할과 시간 순서 분할 시나리오 하에서 실험하여, 모델 순위가 두 분할 전략 간에 통계적으로 유의한 차이가 있음을 입증했습니다. 또한, 실험 결과 시간 경과에 따라 테스트 세트의 예측이 더 어려워진다는 시간 데이터 분포 변화의 증거를 제시했습니다.



### Effects of Using Synthetic Data on Deep Recommender Models' Performanc (https://arxiv.org/abs/2406.18286)
- **What's New**: 추천 시스템(Recommender systems)에서 데이터 불균형 문제를 해결하기 위해 합성 데이터 생성(synthetic data generation)을 검토한 연구가 발표되었습니다. 이 연구는 부정적인 상호작용이 긍정적인 상호작용보다 많은 불균형을 겪는 추천 시스템의 성능을 향상시키는 데 중점을 둡니다.

- **Technical Details**: 이 연구에서는 SMOTE, CTGAN, Gaussian Copula, Copula GAN, TVAE, TabDDPM 등 다양한 합성 데이터 생성 방법을 사용했습니다. 이러한 방법들을 활용하여 생성된 합성 데이터를 원본 데이터에 통합하였으며, 데이터 불균형 문제를 해결하기 위한 다양한 시나리오를 실험했습니다. 부정적인 샘플을 25% 또는 50% 추가하는 등 여러 시나리오를 통해 CTR 예측 모델을 평가했습니다.

- **Performance Highlights**: 실험 결과, 생성된 부정적인 샘플을 포함한 경우 AUC(Area Under the Curve) 점수가 일관되게 향상되었습니다. 이로 인해 합성 부정 샘플이 데이터 부족 및 불균형 문제를 해결하는 데 유용할 수 있음을 강조하며, 추천 시스템의 성능을 궁극적으로 개선할 수 있음을 보여줍니다.



### Improving the Consistency in Cross-Lingual Cross-Modal Retrieval with 1-to-K Contrastive Learning (https://arxiv.org/abs/2406.18254)
Comments:
          Accepted by KDD 2024 Research Track

- **What's New**: 이번 논문에서는 Cross-lingual Cross-modal Retrieval(CCR)에서 나타나는 불일치를 해결하기 위한 1-to-K 대조 학습 방법을 제안합니다. 이 방법은 언어별로 동등하게 대우하는 동시에 오류 전파와 최적화 바이어스를 제거합니다. 또한, Mean Rank Variance(MRV)라는 새로운 평가 지표를 제안하여 개별 인스턴스 내에서 언어 간의 순위 불일치를 반영하도록 합니다.

- **Technical Details**: CCR은 다중 언어 시나리오에서 이미지-텍스트 검색을 달성하는 것을 목표로 합니다. 최근 대규모 데이터에 기반한 대조 학습이 CCR 작업의 성능을 크게 향상시켰지만, 기존의 방법은 두 가지 불일치 문제를 일으킵니다. 하나는 언어 간 호출 성능이 일관되지 않다는 것이고, 다른 하나는 인스턴스 내 언어 간 순위가 일관되지 않다는 것입니다. 이를 해결하기 위해 1-to-K 대조 학습을 제안하며, 이미지와 다른 K개의 언어 텍스트가 동시에 정렬되도록 합니다.

- **Performance Highlights**: 제안된 방법은 4개의 CCR 데이터셋에서 실험을 통해 보다 작은 규모의 사전 훈련 데이터를 사용하면서도 호출률과 MRV를 모두 개선했습니다. CCRk 모델은 새로운 최첨단 성능을 달성했습니다.



### Knowledge Graph Enhanced Retrieval-Augmented Generation for Failure Mode and Effects Analysis (https://arxiv.org/abs/2406.18114)
- **What's New**: 이 논문은 FMEA(Failure Mode and Effects Analysis) 데이터에서 추론 및 분석 기능을 강화하기 위해 지식 그래프(KG, Knowledge Graph)를 통합한 RAG(Retrieval-Augmented Generation) 프레임워크를 제안합니다. 이는 기존의 FMEA 도구들이 종종 결여된 추론 기능을 보완하여 보다 효율적인 오류 예방을 가능하게 합니다.

- **Technical Details**: 제안된 프레임워크는 FMEA 관찰을 속성 지식 그래프로 모델링하고, 이를 기반으로 벡터 임베딩을 생성하는 알고리즘을 포함합니다. 또한, 비파라메트릭 데이터 스토어를 지식 그래프로 확장하여, 질문 응답(QA) 작업에서 보다 정확하고 해석 가능한 정보를 제공할 수 있습니다. 이는 언어 모델의 추론 능력과 KG의 상징적 구조를 결합하여 보다 효율적인 데이터 조회와 분석을 가능하게 합니다.

- **Performance Highlights**: 인간 연구를 통해 제안된 접근법의 효과를 검증하였으며, 문맥 회수와 정밀도 측면에서 성능을 측정하였습니다. 이로써 사용자의 편의성과 정보 정확성을 동시에 향상시키는 결과를 도출할 수 있었습니다.



### Efficient Document Ranking with Learnable Late Interactions (https://arxiv.org/abs/2406.17968)
- **What's New**: 본 논문에서는 정보 검색에서 질의-문서 관련성을 예측하기 위해 Cross-Encoder (CE)와 Dual-Encoder (DE) 모델이 주로 사용되는데, 각각의 장단점을 극복한 새로운 LITE (Lightweight Scoring with Token Einsum) 모델을 제안합니다. CE 모델은 높은 품질의 예측을 제공하지만, DE 모델은 낮은 대기시간(latency)을 특징으로 합니다. 최근에는 CE와 DE의 절충안을 제공하는 late-interaction 모델이 제안되어 왔지만, 기존 모델은 수작업으로 설계된 경량화된 검사기(scorer)를 사용해 성능 및 저장 부담이 발생하는 문제가 존재했습니다.

- **Technical Details**: LITE 모델은 Transformer 인코더 위에 경량의 학습 가능한 비선형 변환을 적용하여 질의와 문서 토큰 임베딩 간의 유사행렬(similarity matrix)를 처리합니다. 이를 위해 두 개의 공유된 다층 퍼셉트론(MLP)을 사용하여 행 및 열에서 유사행렬을 처리하고 최종 점수를 하나의 스칼라 값으로 투영합니다. 이론적으로, LITE는 작은 임베딩 차원에서도 연속적인 스코어링 함수를 보편적으로 근사할 수 있는 능력을 가지고 있음을 증명했습니다.

- **Performance Highlights**: 실험 결과, LITE는 MS MARCO와 같은 도메인 내 리랭킹 및 BEIR와 같은 도메인 외 리랭킹 작업에서 기존의 late-interaction 모델인 ColBERT를 능가했습니다. MS MARCO 통과 리랭킹 실험에서 LITE는 더 나은 일반화 성능을 가지고 있으며 ColBERT 대비 대기시간을 줄이고 저장 공간을 0.25배로 줄이는 결과를 보였습니다. 이는 학습된 비선형 변환이 제공하는 향상된 표현력 덕분입니다.



### Concordance in basal cell carcinoma diagnosis. Building a proper ground truth to train Artificial Intelligence tools (https://arxiv.org/abs/2406.18240)
Comments:
          Manuscript word count: 3000, Number of figures: 2, Number of tables: 3

- **What's New**: 본 연구는 여러 이름난 피부과 의사가 함께 설정한 Ground Truth(GT)를 통해 인공지능(AI) 도구의 성능을 분석했습니다. 이는 기초세포암(Basal Cell Carcinoma, BCC) 진단에 있어 표준적인 진단 기준을 명확히 하는데 중요한 발판이 될 것입니다.

- **Technical Details**: 병원 및 초진 센터에서 촬영된 1434개의 피부경 경상 이미지 중 204개가 AI 도구로 테스트되었습니다. 이는 4명의 피부과 의사가 독립적으로 레이블링한 데이터를 사용하여 학습되었습니다. GT 설정에는 두 가지 방법, 다수결 및 기대 최대화 알고리즘(Expectation Maximization, EM) 방식이 사용되었습니다. 이 과정에서 Hamming distance, Cohen’s Kappa coefficient, Fleiss’ Kappa coefficient 등의 수치를 통해 판정 일치도를 평가했습니다.

- **Performance Highlights**: 4명의 피부과 의사가 독립적으로 레이블링한 데이터를 사용한 결과, BCC 진단에서 뛰어난 일치율(Fleiss-Kappa=0.9079)과 생검 결과와의 높은 상관관계(PPV=0.9670)를 보였습니다. 하지만, 일부 피부경에 대한 판별에서는 낮은 일치율을 보였습니다. AI 도구의 성능은 단일 피부과 의사 기반 GT와 4명의 피부과 의사 합의 기반 GT를 사용하여 학습한 경우에 통계적으로 유의미한 차이를 보였습니다.



### NormTab: Improving Symbolic Reasoning in LLMs Through Tabular Data Normalization (https://arxiv.org/abs/2406.17961)
Comments:
          Work in Progress

- **What's New**: 최근 몇 년간 대규모 언어 모델(Large Language Models, LLMs)은 텍스트 데이터 파싱 및 코드 생성에서 뛰어난 성능을 보였습니다. 그러나 LLMs는 구조적 다양성과 일관성 없는 테이블 셀 값 문제로 인해 탁상 데이터를 포함한 작업에서 상징적 추론(symbolic reasoning)에 어려움을 겪습니다. 이를 해결하기 위해, 우리는 웹 테이블을 정규화하여 상징적 추론 성능을 향상시키는 새로운 프레임워크인 NormTab을 소개합니다.

- **Technical Details**: NormTab은 테이블 정규화를 독립적인 전처리 단계로 수행하여 LLM 기반 상징적 추론을 지원합니다. 이를 통해 구조적 변동성 및 혼합 데이터 형식 문제 등을 해결하여 정확하고 효율적인 상징적 추론 및 쿼리 처리를 가능하게 합니다. 특히 테이블 구조 정규화 및 값 정규화를 포함합니다. 이를 통해 일관성과 정확성을 보장하고, LLM이 더 나은 데이터 클리닝 및 변환 작업을 수행할 수 있도록 도와줍니다.

- **Performance Highlights**: NormTab은 WikiTableQuestions와 TabFact 같은 도전적인 웹 테이블 데이터셋에서 상당한 상징적 추론 성능 향상을 입증합니다. 이러한 데이터셋들은 테이블 구조와 내용의 다양성을 제공하여, 웹 테이블 정규화가 LLM 기반 상징적 추론 작업에 미치는 영향을 철저히 조사할 수 있게 합니다. 실험 결과, NormTab을 활용함으로써, 복잡한 추론 작업에서 LLMs의 성능을 크게 향상시킬 수 있음을 보여줍니다.



### Understanding the Role of User Profile in the Personalization of Large Language Models (https://arxiv.org/abs/2406.17803)
- **What's New**: 본 연구는 사용자 프로필을 활용하여 대형 언어 모델(LLMs)을 개인화하는 방법을 탐구하며, 사용자 프로필이 LLM 개인화에 미치는 영향 메커니즘을 분석합니다. 주로 개인화 정보가 의미 정보보다 더 중요하며, 사용자가 생성하거나 승인한 역사적 개인화 응답이 중요한 역할을 한다는 것을 발견했습니다.

- **Technical Details**: 사용자 프로필을 통해 LLM의 성능을 개선하는 방법으로 다양한 강화 방법을 비교하였습니다. 또한, 사용자 프로필의 각 요소와 입력 맥락 내 위치가 개인화에 미치는 영향을 분석하였습니다. 실험 결과, 초기 부분에 배치된 사용자 프로필이 개인화에 더 큰 영향을 미치는 것을 확인했습니다.

- **Performance Highlights**: 개인화 정보가 포함된 사용자 프로필을 활용했을 때, 의미 정보 단독으로 개선되는 경우보다 성능이 더 우수했습니다. 또한, 사용자 응답이 실제 사용자에 의해 생성되거나 승인된 경우, 개인화 효과가 더욱 극대화되었습니다.




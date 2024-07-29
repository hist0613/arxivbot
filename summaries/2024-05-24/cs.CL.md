### A Nurse is Blue and Elephant is Rugby: Cross Domain Alignment in Large Language Models Reveal Human-like Patterns (https://arxiv.org/abs/2405.14863)
Comments:
          CogSci

- **What's New**: 이 논문은 크로스 도메인 정렬(cross-domain alignment) 작업을 통해 대형 언어 모델(LLMs)의 개념화와 추론 능력을 평가하는 연구를 다룹니다. '의사가 색깔이라면, 어떤 색깔일까?'와 같은 질문을 통해, 사람과 모델이 추상적, 구체적 개념을 어떻게 표현하는지 조사합니다.

- **Technical Details**: 이 작업은 인지 과학에서 가져온 크로스 도메인 정렬 작업을 LLMs 평가에 사용합니다. 여러 LLMs에게 이러한 작업을 수행하도록 지시하고, 그들의 응답을 인구 집단과 개별 모델 차원에서 분석합니다. 또한, 모델이 자신의 예측에 대해 어떻게 추론하는지 설명을 통해 평가합니다.

- **Performance Highlights**: 연구 결과, 모델과 인간의 매핑(mapping) 및 설명간 여러 유사점을 발견했습니다. 이는 모델이 인간과 유사하게 개념을 표현하고 행동하는 것을 시사합니다. 특히, 모델들은 주로 유효한 설명을 제공하며, 인간과 유사한 추론 경로(reasoning paths)를 채택했습니다.



### Bitune: Bidirectional Instruction-Tuning (https://arxiv.org/abs/2405.14862)
- **What's New**: 비튠(Bitune)은 미리 학습된 디코더 기반의 대규모 언어 모델을 더욱 향상된 인스트럭션 튜닝(instruction-tuning) 방식으로 개선해, 다운스트림 작업에서 일관된 성능 향상을 이루었습니다. 비튠은 프롬프트(prompt)에 대해 인과적 주의(causal attention)와 양방향 주의(bidirectional attention)를 모두 적용해 쿼리나 명령어에 대한 더 나은 표현을 얻습니다.

- **Technical Details**: 비튠은 두 세트의 매개변수를 소개하며, 이러한 매개변수에 대해 파라미터 효율적인 미세 조정(parameter-efficient finetuning) 기술을 적용합니다. 인과적 특성과 양방향 특성은 가중 평균(weighted average)으로 결합되며, 이 가중 평균은 학습 가능한 계수를 통해 새 토큰을 생성하는 데 사용됩니다.

- **Performance Highlights**: 비튠은 제로샷(zero-shot) 성능에서 상당한 개선을 보여줍니다. 상식적 추리(common sense reasoning), 산술(arithmetic), 언어 이해(language understanding) 작업에서 어떤 PEFT 기술에도 민감하지 않음을 증명하는 광범위한 분할 연구를 통해 각 구성 요소의 역할을 유효하게 검증했습니다.



### From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step (https://arxiv.org/abs/2405.14838)
- **What's New**: 이 논문은 Chain-of-Thought(CoT) 단계를 내재화 (internalize)하는 방법을 제안합니다. CoT 단계를 명시적으로 생성하는 대신, 모델이 이러한 단계를 내재화하도록 훈련하여 최종 출력의 정확도를 높이는 방법을 탐구합니다.

- **Technical Details**: 구체적인 방법은 다음과 같습니다. 먼저 명시적인 CoT 추론을 위한 모델을 훈련한 후, 중간 단계를 점진적으로 제거하고 모델을 미세 조정 (finetune)합니다. 이를 통해 모델은 중간 추론 단계를 내재화하여 더 간결한 추론 과정을 유지하면서도 높은 성능을 달성하게 됩니다. 예를 들어, GPT-2 Small 모델은 이 방법을 사용하여 최대 99%의 정확도로 9x9 곱셈 문제를 해결할 수 있으며, 기존 훈련 방식으로는 4x4 곱셈 문제 이상을 해결할 수 없습니다.

- **Performance Highlights**: 우리의 접근 방식은 더 큰 언어 모델에서도 효과적임이 입증되었습니다. 예를 들어, Mistral 7B 모델은 중간 단계를 생성하지 않고도 GSM8K에서 50% 이상의 정확도를 달성했습니다.



### HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models (https://arxiv.org/abs/2405.14831)
- **What's New**: 새로운 연구에서 HippoRAG라는 혁신적인 retrieval (검색) 프레임워크가 소개되었습니다. 이는 인간의 장기 기억에 대한 해마 인덱싱 이론에서 영감을 받아 새로운 경험을 더 깊고 효율적으로 통합할 수 있도록 합니다. 이를 통해 기존 retrieval-augmented generation (RAG) 방법론에서 발생하는 문제를 해결하고자 합니다.

- **Technical Details**: HippoRAG는 LLMs (대형 언어 모델), 지식 그래프 (knowledge graphs), 그리고 Personalized PageRank 알고리즘을 활용하여 인간 기억에서 신피질과 해마의 다른 역할을 모방합니다. 이로써 멀티-홉 (multi-hop) 질문 응답 작업에서의 효율성과 성능을 크게 향상시킵니다.

- **Performance Highlights**: HippoRAG는 기존의 RAG 방법론과 비교했을 때 최대 20% 성능 향상을 보였으며, 단일 단계 검색(single-step retrieval)에서는 IRCoT 같은 반복 검색(Iterative Retrieval) 방법과 유사하거나 더 나은 성능을 10-30배 저렴하고 6-13배 빠르게 달성합니다. 또한, HippoRAG를 IRCoT에 통합하면 더 큰 성능 향상이 이루어집니다. 코드 및 데이터는 공개되어있습니다.



### Implicit Personalization in Language Models: A Systematic Study (https://arxiv.org/abs/2405.14808)
- **What's New**: 이번 연구는 암묵적 개인화(Implicit Personalization, IP) 현상을 체계적으로 탐구한 것입니다. 이는 언어 모델이 입력 프롬프트에서 사용자의 배경을 암묵적으로 추론하고, 이 추론에 기반하여 응답을 맞추는 현상입니다. 기존 연구에서는 이 문제의 다양한 사례를 다루었지만, 이러한 행동을 연구할 통합 프레임워크는 부족했습니다. 이번 연구는 엄밀한 수학적 정식화, 다중 관점의 도덕적 추론 프레임워크 및 일련의 사례 연구를 통해 IP를 체계적으로 탐구합니다.

- **Technical Details**: 이 연구의 이론적 기초는 구조적 인과 모델(Structural Causal Model)에 의존하며, 직접 개입할 수 없는 매개 변수의 인과적 효과를 추정하기 위해 새로운 방법인 간접 개입(Indirect Intervention)을 도입합니다. 우리는 수학적 접근 방식뿐만 아니라 도덕 철학의 세 가지 학파에 기반한 도덕적 추론 원칙을 도입하여 IP가 도덕적으로 적절할 수 있는 상황과 그렇지 않은 상황을 연구합니다.

- **Performance Highlights**: 이 연구는 수학적 및 윤리적 통찰과 함께 세 가지 다양한 사례 연구를 제시하여 IP 문제의 다양한 특성을 설명하고, 향후 연구를 위한 권장 사항을 제시합니다. 연구의 코드와 데이터는 제공된 URL에서 확인할 수 있습니다.



### Can LLMs Solve longer Math Word Problems Better? (https://arxiv.org/abs/2405.14804)
- **What's New**: 이번 연구에서는 수학 단어 문제(Math Word Problems, MWPs)의 맥락 길이 일반화(Context Length Generalizability, CoLeG)를 처음으로 탐구했습니다. 이를 위해 길이가 긴 서사가 포함된 Extended Grade-School Math (E-GSM)라는 새로운 문제 모음을 소개했습니다. 현재 많이 사용되는 대형 언어 모델(LLMs)이 긴 맥락을 가진 수학 문제를 해결하는 능력에 대한 조사가 부족하다는 점에서, 본 연구는 이 분야의 초기 시도라 할 수 있습니다.

- **Technical Details**: 우리는 CoLeG를 평가하기 위해 기존 제로-샷 프롬팅(zero-shot prompting) 기법과 다양한 대형 언어 모델들에서 두 가지 새로운 지표(metrics)를 제안했습니다. 이를 통해, 상용 모델(proprietary LLMs)과 오픈 소스 모델(open-source LLMs) 모두에서 CoLeG가 일반적으로 결여되어 있음을 발견했습니다. 이러한 문제점을 해결하기 위해, 상용 모델에는 새로운 설명 프롬프트(instructional prompt)를, 오픈 소스 모델에는 새로운 데이터 증강(data augmentation) 작업을 제안했습니다.

- **Performance Highlights**: 우리의 제안된 방법들은 E-GSM에서의 성능 향상을 보였을 뿐만 아니라, 다른 여러 MWP 벤치마크에서도 일반화 가능성을 보여주었습니다. 이는 대형 언어 모델들이 복잡한 현실 문제에 적용될 수 있는 길을 열어주며, 모델 일반화 및 훈련 방법론 탐구에 새로운 방향성을 제공합니다.



### Lessons from the Trenches on Reproducible Evaluation of Language Models (https://arxiv.org/abs/2405.14782)
- **What's New**: 이 논문은 대형 언어 모델 (Large Language Models, LLMs) 평가 방법론에 집중합니다. 연구자들이 평가 설정에 대한 모델의 민감성, 평가 방법 간의 비교 어려움, 재현 가능성 및 투명성 부족 등의 도전에 직면하는 상황에서, 논문은 이러한 문제를 해결하기 위한 지침과 교훈을 제공합니다. 특히, LLM 평가를 독립적이고 재현 가능하며 확장 가능하게 수행할 수 있는 오픈 소스 라이브러리인 'Language Model Evaluation Harness (lm-eval)'를 소개합니다.

- **Technical Details**: 논문은 평가 방법론에서 흔히 직면하는 문제를 개괄적으로 설명하며, 이러한 문제의 영향을 줄이기 위한 최적의 방법을 제시합니다. 그리고 lm-eval는 독립적 재현 가능하며 확장 가능한 모델 평가를 위한 기능들을 제공합니다. 이러한 기능들은 다양한 케이스 스터디를 통해 검증되었으며, 연구자들이 평가 세팅을 보다 공정하고 투명하게 관리할 수 있도록 설계되었습니다.

- **Performance Highlights**: lm-eval 라이브러리는 모델 평가의 독립성, 재현 가능성, 확장 가능성을 보장하면서도, 다양한 케이스에서 평가 방법론의 문제를 완화하는 데 성공했습니다. 이를 통해 연구자들은 평가 결과의 신뢰성을 높이고, 다양한 언어 모델 간의 비교를 보다 정확하게 할 수 있습니다.



### Smart Bilingual Focused Crawling of Parallel Documents (https://arxiv.org/abs/2405.14779)
- **What's New**: 이번 연구에서는 병렬 텍스트(Parallel Texts)를 더욱 효과적으로 크롤링하는 스마트 크롤링(smart crawling) 방법을 제안하였습니다. 기존의 비효율적인 크롤링 방법과 달리, 제안한 방법은 언어 추론 모델과 병렬 문서 여부를 판독하는 모델을 사용해 병렬 콘텐츠를 보다 신속하게 찾는 것을 목표로 합니다.

- **Technical Details**: 제안된 접근 방식은 두 가지 주요 모델을 기반으로 합니다. 첫 번째 모델은 URL을 통해 문서의 언어를 추론하고, 두 번째 모델은 URL 쌍이 병렬 문서로 연결되는지 여부를 추론합니다. 이 두 모델은 각각 독립적으로 평가되었으며, 크롤링 도구에 통합된 상태에서도 평가되었습니다. 그 결과, 두 모델 모두 개별적으로 효과적임이 입증되었고, 결합 시 병렬 콘텐츠를 조기 발견하는 데 있어 유용함이 확인되었습니다.

- **Performance Highlights**: 제안된 방법은 기존의 무작위 다운로드 방식에 비해 불필요한 문서의 다운로드를 줄여주며, 더 많은 병렬 문서를 확보할 수 있음을 보여주었습니다. 이는 스마트 크롤링 방식이 단순한 문서 다운로드 양 증가보다 실제 유용한 병렬 문서를 더 많이 찾아낼 수 있음을 의미합니다.



### WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models (https://arxiv.org/abs/2405.14768)
Comments:
          Work in progress

- **What's New**: 새로운 연구에서는 대형 언어 모델(LLMs)을 지속적으로 업데이트하고 편집할 수 있는 방법을 제안합니다. 특히, 모델의 기억 구조의 수정이 신뢰성, 일반화, 지역성에서 발생하는 삼중 딜레마를 해결하기 위한 WISE (dual parametric memory scheme)를 도입했습니다.

- **Technical Details**: WISE는 주 기억(main memory)과 부 기억(side memory)으로 구성된 듀얼 파라메트릭 메모리(sceme)를 설계하였습니다. 주 기억은 사전 학습된 지식을 저장하고 부 기억은 편집된 지식을 저장합니다. 이때, 주어진 쿼리에 따라 어느 기억을 사용할지 결정하는 라우터(router)를 훈련합니다. 또한, 지속적인 편집을 위해 지식 분할(knowledge-sharding) 메커니즘을 도입하여 서로 다른 편집 세트가 별도의 파라미터 하위 공간(subspaces)에 위치하게 하고 이를 공유 메모리로 통합합니다.

- **Performance Highlights**: 광범위한 실험 결과, WISE는 GPT, LLaMA, Mistral 등 최신 대형 언어 모델 아키텍처에서 질문 응답, 환각(hallucination), 비배포 상황(out-of-distribution)에서 이전의 모델 편집 방법보다 뛰어난 성능을 보였습니다.



### Evaluating Large Language Models for Public Health Classification and Extraction Tasks (https://arxiv.org/abs/2405.14766)
Comments:
          33 pages. Feedback and comments are highly appreciated

- **What's New**: 새로운 연구에서는 대규모 언어 모델(LLMs)이 공중 보건에서 자유 텍스트 분류와 추출 작업을 지원할 수 있는 가능성을 평가합니다. 이 연구에서는 외부 주석이 달린 6개의 데이터셋과 새롭게 내부 주석이 달린 7개의 데이터셋을 결합하여 공중 보건과 관련된 텍스트 처리 작업을 수행할 수 있는 LLM을 평가했습니다. 주목할 만한 점은 Llama-3-70B-Instruct 모델이 대부분의 작업에서 최고의 성능을 보였다는 것입니다.

- **Technical Details**: 연구에서는 5개의 open-weight LLM(7-70 billion parameters)을 사용하여 제로샷(in-context learning) 방식으로 모든 작업을 평가했습니다. 특히 Llama-3-70B-Instruct 모델이 17개의 작업 중 15개에서 최고의 성과를 거두었다고 보고되었습니다. 일부 도전적인 작업에서는 모든 LLM이 60% 이하의 마이크로 F1 스코어를 기록한 반면, 다른 작업에서는 80% 이상의 마이크로 F1 스코어를 달성했습니다.

- **Performance Highlights**: 전체적으로 LLM 모델들이 공중 보건 전문가들이 다양한 자유 텍스트 소스로부터 정보를 추출하여 공중 보건 감시, 연구 및 개입을 지원하는데 유용할 수 있는 가능성을 보여주었습니다. 특히 GPT-4와 비교해도 Llama-3-70B-Instruct 모델이 일부 작업에서 동일하거나 더 나은 성능을 보여주었습니다.



### SimPO: Simple Preference Optimization with a Reference-Free Reward (https://arxiv.org/abs/2405.14734)
Comments:
          Code: this https URL

- **What's New**: 이번 연구에서는 기존의 Direct Preference Optimization(DPO) 알고리즘을 대체할 SimPO라는 접근법을 제안합니다. SimPO는 시퀀스의 평균 로그 확률(log probability)을 암묵적 보상으로 사용하여 모델 생성과 일치시키며, 참조 모델(reference model)이 필요 없게 만듭니다. 이로 인해 계산 및 메모리 효율성이 향상됩니다.

- **Technical Details**: SimPO는 Bradley-Terry 목적 함수에 목표 보상 마진(target reward margin)을 추가하여 이기고 지는 응답들 간의 마진을 넓힐 수 있도록 설계되었습니다. 이러한 디자인을 통해 알고리즘의 성능을 더욱 향상시켰습니다. SimPO는 다양한 최신 트레이닝 환경에서 DPO 및 그 최신 변형들과 비교되었습니다. 예를 들어, 기본 및 인스트럭션 튜닝된 Mistral 및 Llama3 모델에서 평가되었습니다.

- **Performance Highlights**: SimPO는 AlpacaEval 2, MT-Bench, Arena-Hard 등의 다양한 벤치마크에서 일관되게 뛰어난 성능을 보여줍니다. 특히 AlpacaEval 2에서는 DPO보다 최대 6.4포인트, Arena-Hard에서는 최대 7.5포인트 더 높은 성능을 나타냈습니다. Llama3-8B-Instruct 기반으로 구축된 최상위 모델은 AlpacaEval 2에서 44.7의 길이 조절된 승률을 기록하여 Claude 3 Opus를 능가하였습니다. 또한 Arena-Hard에서는 33.8의 승률을 기록하여 가장 강력한 8B 오픈 소스 모델로 자리매김했습니다.



### CAPE: Context-Adaptive Positional Encoding for Length Extrapolation (https://arxiv.org/abs/2405.14722)
Comments:
          Technical Report

- **What's New**: 본 논문에서는 Context-Adaptive Positional Encoding (CAPE) 방법을 제안합니다. CAPE는 입력된 문맥에 따라 동적으로 조정되며, 학습된 고정 사전 (fixed priors)을 기반으로 의미적으로 조정됩니다. 이는 기존의 절대 위치 인코딩(APE) 및 상대 위치 인코딩(RPE)의 한계를 극복해 더 높은 적응성과 유연성을 제공합니다.

- **Technical Details**: CAPE는 입력 문맥에 따라 동적으로 조정되는 위치 인코딩을 사용합니다. 이는 고정된 사전 학습값과 입력된 문맥을 기반으로 하며, 기존 APE와 RPE 방식의 고정된 패턴을 탈피해 모델의 대응 능력을 향상시킵니다. 실험은 실제 데이터셋(Arxiv, Books3, CHE)에서 수행되었으며, CAPE의 성능 향상이 통계적으로 유의미함을 검증했습니다.

- **Performance Highlights**: CAPE를 사용한 모델은 학습된 길이와 길이 일반화 측면에서 성능의 향상을 보였습니다. 특히, 시퀀스 길이 128에서 학습된 모델이 평가 시퀀스 길이 8192에서도 기존의 고정 위치 인코딩 방식보다 더 나은 성능을 보였습니다. 이는 어댑티브 위치 인코딩 방법의 효과를 증명하는 바입니다.



### A Declarative System for Optimizing AI Workloads (https://arxiv.org/abs/2405.14696)
Comments:
          28 pages, 10 figures

- **What's New**: 최신 AI 모델을 이용해 다양한 데이터에 대한 분석적 질의를 처리할 수 있는 새로운 시스템 Palimpzest를 소개합니다. 이 시스템은 사용자가 선언적 언어로 질의를 정의할 수 있게 하며, 최적의 AI 모델, 프롬프트 기법, 관련 최적화 기술을 사용하여 질의를 처리합니다.

- **Technical Details**: Palimpzest는 AI 모델, 프롬프트 기법(및 관련 'foundation model' 최적화 기법) 등의 검색 공간을 탐색하여 런타임, 비용, 출력 데이터 품질 간의 절충안을 찾아 질의를 구현합니다. 법적 검색, 부동산 검색, 의료 스키마 매칭 등의 작업에서 Palimpzest를 평가했습니다.

- **Performance Highlights**: 단순 프로토타입임에도 불구하고, Palimpzest는 기존 방법에 비해 3.3배 빠르고, 2.9배 저렴하며, 더 나은 데이터 품질을 제공합니다. 또한, 병렬 처리를 활성화하면 단일 스레드 GPT-4 기준으로 최대 90.3배 속도 향상과 9.1배 비용 절감이 가능하며, F1-score는 기준의 83.5%에 도달합니다.



### Efficient Medical Question Answering with Knowledge-Augmented Question Generation (https://arxiv.org/abs/2405.14654)
Comments:
          Accepted at the Clinical Natural Language Processing Workshop, NAACL 2024

- **What's New**: 이번 연구에서는 작은 언어 모델(small language model)의 의료 분야 능력을 향상시키기 위한 새로운 방법을 소개합니다. 이 접근법은 두 가지 단계를 통해 진행됩니다. 첫 번째 단계에서는 모델을 의학 교과서 코퍼스(corpus of medical textbooks)로 파인튜닝(fine-tune)합니다. 두 번째 단계에서는 GPT-4를 사용하여 교과서 지식을 기반으로 문항 생성(question generation)을 수행한 뒤, 이를 활용해 모델을 추가적으로 파인튜닝합니다. 또한, 새로운 의료 질문 응답 데이터셋인 ECN-QA를 도입했습니다. 이 데이터셋은 관련된 순차적 질문들로 구성된 '진행성 질문(progressive questions)'을 포함하고 있습니다.

- **Technical Details**: 이 연구에서는 먼저 의료 분야의 특화된 지식을 잘 반영하기 위해 작은 언어 모델을 의학 교과서 데이터로 파인튜닝합니다. 이후 GPT-4를 이용하여 교과서에서 추출한 지식 기반으로 실제 응용 과제와 유사한 질문들을 생성합니다. 생성된 질문들을 모델 파인튜닝에 활용하여 모델이 더욱 정확하게 응답할 수 있도록 합니다. 이를 통해 모델의 전문성 및 대응성을 강화합니다.

- **Performance Highlights**: 연구의 결과는 적절한 방법으로 파인튜닝된 작은 언어 모델이 의료 분야에서도 괄목할만한 성과를 낼 수 있음을 보여줍니다. 특히, 새롭게 도입된 ECN-QA 데이터셋에서 혁신적인 훈련 전략의 이점을 입증했습니다. 연구의 코드 및 가중치는 본 논문의 URL을 통해 공개됩니다.



### Unveiling the Achilles' Heel of NLG Evaluators: A Unified Adversarial Framework Driven by Large Language Models (https://arxiv.org/abs/2405.14646)
Comments:
          ACL24 Finding

- **What's New**: 자연어 생성(NLG) 시스템의 자동 평가는 오래 지속된 도전 과제입니다. 최근 연구들은 인간 평가와 잘 맞는 다양한 신경 네트릭(neural metrics)을 강조했습니다. 그러나 이러한 평가자들이 적대적 교란(adversarial perturbations)에 얼마나 잘 견디는지는 여전히 잘 탐구되지 않았습니다. 이를 해결하기 위해 AdvEval이라는 새로운 블랙박스(adversarial framework)를 도입했습니다. AdvEval은 인간과 피해자 평가자 간의 큰 의견 불일치를 유발하는 데이터를 생성하도록 특별히 설계되었습니다.

- **Technical Details**: AdvEval은 대형 언어 모델(LLMs)의 텍스트 생성 및 평가에서의 최근 성공을 바탕으로 강력한 LLMs를 데이터 생성기와 기준 평가자로 채택합니다(adopt). 적대적 데이터는 기준 평가자와 피해자 평가자의 피드백을 통해 자동으로 최적화됩니다. 12개의 피해자 평가자와 11개의 NLG 데이터셋을 대상으로 한 실험에서는 대화(dialogue), 요약(summarization), 질문 평가(question evaluation) 등의 과제를 포함합니다.

- **Performance Highlights**: 결과는 AdvEval이 다양한 피해자 메트릭의 성능 저하를 야기할 수 있음을 보여주며, 그것의 효능을 입증합니다. 이는 NLG 평가자가 적대적 데이터에 얼마나 취약할 수 있는지를 명확히 드러냅니다.



### A Watermark for Low-entropy and Unbiased Generation in Large Language Models (https://arxiv.org/abs/2405.14604)
- **What's New**: 최근 대형 언어 모델(LLM)의 발전으로 인해 LLM이 생성한 콘텐츠를 정확하게 탐지할 필요성이 증가했습니다. 이에 대한 해결책으로 LLM에 거의 눈에 띄지 않는 식별자를 주입하는 '워터마크' 기술이 제안되고 있습니다. 기존의 비편향 워터마킹(unbiased watermarking) 방법은 텍스트 품질을 유지하면서 위조가 불가능하게 설계되었으나, 검사 시 화이트 박스 LLM과 입력 프롬프트에 접근해야 하므로 로컬 배포가 어려웠습니다. 본 연구는 이러한 문제를 해결하는 새로운 방법인 STA-1(Sampling One Then Accepting)을 제안합니다. STA-1은 LLM 접근 없이 워터마크를 검출할 수 있으며 유형 II 오류(type II error)에 대한 통계적 보장을 제공합니다.

- **Technical Details**: STA-1 방법은 텍스트 생성 시 워터마크를 삽입하는 과정에서 LLM에 대한 접근이나 입력 프롬프트가 필요하지 않습니다. 본 연구는 또한 낮은 엔트로피(low-entropy) 시나리오에서 비편향 워터마킹이 워터마크 강도(watermark strength)와 텍스트 품질 사이에 트레이드오프가 있음을 제시합니다. 실험 결과, STA-1은 기존의 비편향 워터마크와 비교할 때 텍스트 품질과 워터마크 강도에서 비슷한 성능을 보이며, 불만족스러운 출력의 위험이 낮습니다.

- **Performance Highlights**: STA-1은 비편향 워터마킹 방법이지만 로컬 배포에 적합하며 유형 II 오류에 대한 통계적 보장을 제공한다는 점에서 기존 방법들과 차별화됩니다. 낮은 엔트로피와 높은 엔트로피(high-entropy) 데이터셋에서 실험한 결과, 텍스트 품질과 워터마크 강도 면에서 기존 비편향 워터마킹과 비교해 경쟁력 있는 성능을 보였습니다. 연구 결과를 바탕으로 구현 코드도 온라인에서 제공되고 있습니다.



### A FAIR and Free Prompt-based Research Assistan (https://arxiv.org/abs/2405.14601)
Comments:
          6 pages, 2 figures, accepted to the Demo track of NLDB 2024 (this https URL)

- **What's New**: 연구 수행을 돕기 위한 Research Assistant (RA) 도구가 개발되었습니다. 이 도구는 ChatGPT 및 Gemini와 같은 고급 자연어 처리 능력을 가진 AI 도구를 이용해 연구 작업을 보조합니다. RA는 FAIR 연구 비교 생성, 연구 주제 구상, 연구 기금 신청서 작성, 과학 블로그 작성, 예비 동료 검토 지원 및 향상된 문헌 검색 쿼리 작성이라는 여섯 가지 주요 연구 작업을 수행할 수 있습니다.

- **Technical Details**: RA 도구는 ChatGPT와 Gemini와 같은 생성 AI 도구에 프롬프트를 적용하여 사용자 입력을 기반으로 표준화된 지침 템플릿에 따라 작업을 수행합니다. 이는 다양한 과학 분야에서도 동일한 수준의 연구 지원을 제공할 수 있습니다. 특히 컴퓨터 과학, 바이러스학, 기후 과학 분야에서 RA 도구의 출력 결과는 해당 작업을 수행한 도메인 전문가의 결과와 유사했습니다.

- **Performance Highlights**: RA 도구는 다양한 과학 분야에 적용 가능하며, 도메인 전문가와 유사한 품질의 결과를 생성할 수 있는 것이 특징입니다. 이를 통해 연구 작업의 효율성과 정확성을 크게 향상시킬 수 있습니다.



### Data Augmentation Techniques for Process Extraction from Scientific Publications (https://arxiv.org/abs/2405.14594)
- **What's New**: 과학 출판물에서 프로세스 추출(process extraction) 작업을 위한 데이터 증강(data augmentation) 기법을 제안합니다. 프로세스 추출 작업을 일련의 라벨링(sequence labeling) 작업으로 처리하여, 문장의 모든 엔티티를 식별하고 프로세스별 역할에 따라 라벨을 지정합니다.

- **Technical Details**: 제안된 방법은 원본 문장에서 얻은 프로세스 관련 정보(process-specific information), 역할 라벨 유사성(role label similarity), 문장 유사성(sentence similarity)을 활용하여 의미 있는 증강 문장을 생성합니다. 이를 통해화학 분야 데이터 셋에서 프로세스 추출 모델의 성능을 대폭 향상시킵니다.

- **Performance Highlights**: 제안된 방법은 프로세스 추출 모델의 성능을 최대 12.3 포인트(F-score 기준) 향상시켰습니다. 또한, 특히 소규모 데이터셋이나 리소스가 제한된 환경에서 과적합(overfitting)을 줄일 수 있는 잠재력을 보였습니다.



### Base of RoPE Bounds Context Length (https://arxiv.org/abs/2405.14591)
Comments:
          17 pages

- **What's New**: 최신 연구는 현재 대형 언어 모델(LLMs: Large Language Models)에서 핵심 구성 요소로 사용되는 위치 임베딩(Position embedding)의 새로운 특성을 발견했습니다. 연구진은 Rotary position embedding (RoPE) 기법이 위치 정보를 회전 행렬(rotation matrix)로 인코딩하여 Llama 시리즈와 같은 많은 LLMs에서 사실상 선택되는 방법임을 지적했습니다. 하지만, 인공지능 모델이 표면적으로 긴 문맥(long-context) 능력을 얻을 수 있다고 주장하면서, RoPE의 새로운 속성인 장기 감쇠(long-term decay)를 제안합니다.

- **Technical Details**: 연구진은 RoPE의 'base' 매개변수가 문맥 길이(context length)를 결정하는 절대 하한값을 가지고 있다는 사실을 밝혀냈습니다. 기존의 RoPE 기법은 Out-of-Distribution(OOD) 문제를 완화하기 위해 base 파라미터를 조정하여 긴 문맥 능력을 확장하려 했으나, 이는 표면적인 해결책에 불과하다고 지적합니다. 연구는 기저 파라미터의 이론적 및 실증적 관계를 통해 RoPE와 문맥 길이의 관계를 제시합니다.

- **Performance Highlights**: 이 연구는 RoPE의 베이스 파라미터와 문맥 길이 간의 명확한 상관관계를 보여줌으로써, 향후 긴 문맥을 처리하는 모델의 훈련에 중요한 통찰력을 제공할 수 있습니다. 특히, 장기 감쇠(long-term decay) 속성은 대형 언어 모델이 더 효과적이고 깊이 있는 문맥 이해 능력을 갖출 수 있도록 기여할 전망입니다.



### Representation noising effectively prevents harmful fine-tuning on LLMs (https://arxiv.org/abs/2405.14577)
- **What's New**: 오픈소스 대형 언어 모델(LLMs)의 공개는 악의적인 사용자가 모델을 나쁜 목적으로 미세 조정(fine-tuning)할 수 있어 이중 용도의 위험을 내포합니다. 본 연구는 이러한 문제를 해결하기 위해 새로운 방어 메커니즘인 RepNoise(Representation Noising)를 제안했습니다. RepNoise는 유해한 표현에 대한 정보를 제거하여 악성 미세 조정을 어렵게 합니다. 이 방어 기법은 방어 과정 중에 보지 못한 다양한 유해성 요소에서도 일반화할 수 있습니다.

- **Technical Details**: RepNoise 방어 메커니즘은 모델의 모든 레이어(layer)에서 유해한 표현에 대한 정보를 제거하는 방식으로 작동합니다. 이를 통해 미세 조정 중에 이러한 유해한 정보를 복구하는 것이 어렵게 됩니다. 특히, RepNoise는 모델의 주요 성능은 유지하면서도, 무해한 작업에 대한 학습 가능성을 보존합니다.

- **Performance Highlights**: 모델의 일반적인 기능을 저하시키지 않으며 유해한 표현 정보 제거의 '깊이'(depth)가 방어의 효과성을 결정합니다. 실험적 증거는 RepNoise가 각 레이어에서 유해한 정보를 제거하는 정도가 방어의 성공을 이끈다는 점을 보여줍니다.



### Subtle Biases Need Subtler Measures: Dual Metrics for Evaluating Representative and Affinity Bias in Large Language Models (https://arxiv.org/abs/2405.14555)
Comments:
          9 pages (excluding references), accepted to ACL 2024 Main Conference

- **What's New**: 이 연구는 대형 언어 모델(LLMs)에서 잘 드러나지 않는 미세한 편향을 탐구합니다. 특히 특정 사회적 서사를 선호하는 경향이 있는 모델의 출력에 영향을 미치는 두 가지 편향, 즉 representative bias(대표성 편향)와 affinity bias(유대감 편향)를 다룹니다. 이를 측정하기 위해 두 가지 새로운 측정치, Representative Bias Score(RBS)와 Affinity Bias Score(ABS)를 도입합니다.

- **Technical Details**: 대표성 편향은 LLMs가 특정 정체성 그룹의 경험을 반영하는 출력을 생성하는 경향을 의미하고, 유대감 편향은 모델이 특정 서사나 관점을 선호하는 평가적 경향을 의미합니다. 이 연구에서는 이러한 미묘한 편향을 감지할 수 있는 Creativity-Oriented Generation Suite(CoGS)라는 개방형 작업 모음을 소개합니다. CoGS는 단편 소설 작성, 시 창작과 같이 다양한 창의적인 작업으로 구성되어 있으며, 맞춤형 채점 기준을 사용하여 편향을 탐지합니다.

- **Performance Highlights**: 주요 대형 언어 모델에서는 백인, 이성애자, 남성의 정체성과 관련된 대표성 편향이 현저히 나타났습니다. 또한, 각 모델은 고유한 평가 패턴, 즉 '편향 지문'을 보이며, 이는 인적 평가자에서도 동일한 경향을 확인할 수 있었습니다. 이로써 인간과 기계의 편향 인식 사이의 복잡한 상호작용이 드러났습니다.



### Exploring Alignment in Shared Cross-lingual Spaces (https://arxiv.org/abs/2405.14535)
Comments:
          ACL 2024

- **What's New**: 이 연구에서는 다국어 모형에서의 언어 간의 정렬(alignment)과 중첩(overlap)을 측정하기 위한 새로운 접근법을 제안합니다. 이를 위해 	extit{다국어 임베딩}(multilingual embeddings)에서 잠재 개념(latent concepts)을 클러스터링(clustering)을 통해 발견하고 분석합니다. 이에 따라 두 가지 새로운 지표 	extit{CA}와 	extit{CO}를 도입하여 이러한 측면을 정량화했습니다.

- **Technical Details**: 연구는 	exttt{mT5}, 	exttt{mBERT}, 	exttt{XLM-R} 등 세 가지 다국어 모델을 대상으로 진행되었으며, 세 가지 다운스트림 태스크(Machine Translation, Named Entity Recognition, Sentiment Analysis)에 대해 분석을 수행했습니다. 연구의 주요 목적은 심층 신경망 모델의 높은 차원의 표현(high-dimensional representations)에서 발견되는 잠재 개념들이 어떻게 다국어 환경에서 정렬되고 중첩되는지를 파악하는 것입니다.

- **Performance Highlights**: 주요 발견 사항으로는 i) 네트워크의 더 깊은 층이 언어 비종속 개념(language-agnostic concepts)의 존재로 인해 교차 언어적 정렬이 증가함을 보여주며, ii) 모델의 파인튜닝이(latent space) 내의 정렬을 향상시키고, iii) 이러한 태스크별 조율이 모델에서 제로샷(zero-shot) 능력을 설명하는 데 도움이 됨을 밝혔습니다.



### Unchosen Experts Can Contribute Too: Unleashing MoE Models' Power by Self-Contras (https://arxiv.org/abs/2405.14507)
- **What's New**: 이번 연구에서는 Mixture-of-Experts(MoE) 모델의 비활성화된 전문가들(unchoosen experts)을 활용하여 모델의 출력 품질을 개선하는 새로운 방법을 제안합니다. 이 방법은 Self-Contrast Mixture-of-Experts(SCMoE)라고 불리며, 모델 학습 없이도 추론 단계에서 비활성화된 전문가들의 정보를 활용하여 성능을 향상시킵니다.

- **Technical Details**: MoE 모델에서 각 입력 시퀀스(token)는 라우팅 메커니즘에 의해 각기 다른 전문가들(subset of experts)을 활성화하게 됩니다. 그러나 모든 전문가들이 출력에 기여하지 않기 때문에 모델 용량이 충분히 활용되지 않는 문제가 있었습니다. SCMoE는 활성화된 전문가들의 출력과 비활성화된 전문가들의 출력을 비교(contrast)하여 다음 토큰의 확률을 결정하는 방법입니다. 이 방법은 개념적으로 간단하며 추가적인 큰 계산 비용 없이 구현할 수 있습니다.

- **Performance Highlights**: SCMoE는 여러 벤치마크(GSM8K, StrategyQA, MBPP, HumanEval)에서 Mixtral 8x7B 모델의 추론 성능을 일관되게 향상시켰습니다. 예를 들어, GSM8K에서 정확도를 61.79에서 66.94로 개선했습니다. 또한, SCMoE를 self-consistency와 결합하면 major@20 정확도가 75.59에서 78.31로 증가했습니다.



### Explainable automatic industrial carbon footprint estimation from bank transaction classification using natural language processing (https://arxiv.org/abs/2405.14505)
- **What's New**: 이번 연구에서는 탄소 발자국(Carbon Footprint, CF) 추정에 사용되는 수동 및 자동 방법론을 검토하고, 새로운 설명 가능한 기계 학습(ML) 솔루션을 제안합니다. 이 솔루션은 은행 거래 분류를 통해 자동으로 CF를 계산합니다. 특별히 이번 연구에서는 은행 거래 분류의 설명 가능성을 고려한 CF 추정 방법론을 최초로 제안했습니다.

- **Technical Details**: 이 연구에서는 기계 학습 모델로 Support Vector Machine, Random Forest, Recursive Neural Networks 등을 사용하였습니다. 이러한 모델들은 문헌에서 유망한 성능을 보여 사용되었습니다. 설명 가능성(methodology for explainability)은 국소적 해석 가능 모델을 사용하여 거래 설명에서 추출한 입력 용어의 영향을 평가하는 방식으로 구성되었습니다.

- **Performance Highlights**: 제안된 솔루션은 정확도, 정밀도, 재현율 평가 지표에서 모두 90% 범위의 성과를 보였습니다. 결정 경로에서 제안된 솔루션은 은행 거래와 관련된 CO2 배출량을 추정합니다. 설명 성능은 활동 부문 설명과의 근접성을 기준으로 만족스러운 결과를 보여주었습니다.



### Impact of Non-Standard Unicode Characters on Security and Comprehension in Large Language Models (https://arxiv.org/abs/2405.14490)
Comments:
          46 pages

- **What's New**: 최근의 대형 언어 모델 (large language models)의 발전은 자연어 처리에서 큰 개선을 이루었습니다. 하지만 여전히 탈주(제한된 사용법을 벗어나도록 유도하는 프롬프트 주입), 환각(잘못된 정보 생성), 그리고 이해 오류와 같은 문제가 존재합니다. 이번 보고서에서는 15개의 독립적인 모델들을 비교 분석하고, 각각의 모델이 38개의 쿼리에 대해 표준화된 테스트를 통해 평가되는 결과를 제공합니다.

- **Technical Details**: 각 모델은 탈주, 환각, 이해 오류의 총 발생 횟수를 기준으로 평가되었습니다. 또한, 비표준 유니코드 문자(non-standard Unicode characters)가 LLM(Large Language Models)에 미치는 영향을 경험적으로 분석하고, 베스트 성능을 보이는 LLM들, 예를 들어 GPT-4, Gemini 1.5 Pro, LlaMA-3-70B, Claude 3 Opus의 안전망 메커니즘을 조사하였습니다. 라틴 블록 외의 유니코드 문자와 다른 언어의 문자 변형을 도입해, 강화 학습 인간 피드백(진행)을 통한 안전장치의 효과가 감소함을 관찰했습니다.

- **Performance Highlights**: 연구 결과, 이러한 모델들은 콘텐츠 정책 위반과 프롬프트 누출에 대한 보호가 부족하다고 드러났습니다. 또한 비표준 유니코드 텍스트를 LLM 훈련 데이터에 포함해야 할 필요성을 시사합니다. 이는 모델의 능력을 향상시키기 위한 중요한 개선점으로 작용할 것입니다.



### MoGU: A Framework for Enhancing Safety of Open-Sourced LLMs While Preserving Their Usability (https://arxiv.org/abs/2405.14488)
- **What's New**: 최근 논문에서 제안된 MoGU 프레임워크는 LLMs (Large Language Models)의 안전성을 향상시키면서도 사용 가능성을 유지하는 혁신적인 접근 방식을 소개했습니다. MoGU는 기본 LLM을 사용 가능한 LLM과 안전한 LLM으로 변환하고, 가변 라우팅(dynamic routing)을 통해 이들의 기여도를 균형 있게 조절합니다.

- **Technical Details**: MoGU 프레임워크는 LLM이 악의적인 명령어를 만났을 때, 안전한 LLM의 가중치를 높여 해로움을 방지하고, 반대로 정상적인 명령어에 대해서는 사용 가능한 LLM을 우선시하여 유용한 응답을 제공합니다. 이 라우팅 메커니즘은 동적으로 작동하며, 변환 후의 두 LLM 간의 기여도를 효과적으로 분배합니다.

- **Performance Highlights**: 여러 오픈 소스 LLMs(Vicuna, Falcon, Dolphin, Baichuan2 등)을 통해 MoGU 프레임워크의 우수성을 검증하였습니다. 실험 결과, MoGU는 기존의 방어 전략보다 더 나은 성능을 보였으며, 안전성과 사용성을 모두 충족하는 응답을 제공함이 확인되었습니다.



### RefChecker: Reference-based Fine-grained Hallucination Checker and Benchmark for Large Language Models (https://arxiv.org/abs/2405.14486)
- **What's New**: 이번 연구에서는 Large Language Models (LLMs)이 보이는 잘못된 정보 생성 문제(헛소리 생성, hallucination)를 정교하게 탐지하기 위한 새로운 프레임워크 'RefChecker'를 소개합니다. RefChecker는 LLM 응답에서 claim-triplets을 추출하여 이를 참조(reference)와 비교함으로써 세부적인 헛소리를 탐지할 수 있습니다.

- **Technical Details**: RefChecker는 우선 응답에서 claim-triplets을 생성하는 추출기(extractor)를 통해 핵심 주장을 추출한 후, 이를 참조와 비교하는 체커(checker)를 통해 검증하는 과정을 거칩니다. 실질적인 사용 사례를 반영하기 위해 Zero, Noisy, Accurate Context의 세 가지 설정을 제시하며 다양한 자연어 처리(NLP) 작업들을 포함한 벤치마크를 구성하고, 11k개의 claim-triplets을 7개의 LLM에서 2.1k개의 응답으로부터 주석을 달았습니다. RefChecker는 독점 모델(proprietary models)과 오픈 소스 모델(open-source models) 모두를 지원합니다.

- **Performance Highlights**: 실험 결과, claim-triplets을 활용한 RefChecker가 응답 수준, 문장 수준 및 문장 하위 수준의 주장들보다 우수한 헛소리 탐지를 가능하게 함을 보였습니다. RefChecker는 이전 방법들보다 6.8 to 26.1 포인트 더 높은 퍼포먼스를 보였으며, 그 평가 결과는 인간의 판단과 강하게 일치합니다. 이 연구는 오픈 소스로 제공됩니다.



### Which Information Matters? Dissecting Human-written Multi-document Summaries with Partial Information Decomposition (https://arxiv.org/abs/2405.14470)
- **What's New**: 이 연구는 다중 문서 요약(Multi-Document Summarization, MDS)에서 고품질 요약의 본질을 이해하는 새로운 접근 방식을 제안합니다. 연구진은 부분 정보 분해(Partial Information Decomposition, PID)를 활용하여, 전체 소스 문서들이 제공하는 상호 정보(mutual information)를 조합(union), 중복(redundancy), 시너지(synergy), 고유 정보(unique information)로 분해합니다. 이를 통해 인간이 작성한 요약의 특성을 분석합니다.

- **Technical Details**: 연구진은 다중 문서 요약 데이터셋을 활용하여, 소스 문서의 수와 요약에 대한 기여도 사이에 직접적인 종속성이 존재함을 실증적으로 분석했습니다. PID 방법론을 통해 상호 정보의 다양한 요소를 분해하여 각각의 영향을 파악했습니다.

- **Performance Highlights**: 분석 결과, 소스 문서의 수와 그들의 요약에 대한 기여도 사이에 명확한 의존 관계가 있음을 확인하였습니다. 이는 향후 MDS 성능 향상을 위해 소스 문서 수와 정보의 정교한 관리가 필요함을 시사합니다.



### Exploring the use of a Large Language Model for data extraction in systematic reviews: a rapid feasibility study (https://arxiv.org/abs/2405.14445)
Comments:
          Conference proceedings, peer-reviewed and presented at the 3rd Workshop on Augmented Intelligence for Technology-Assisted Reviews Systems, Glasgow, 2024

- **What's New**: 이번 연구는 GPT-4, 즉 대규모 언어 모델(LLM)을 사용하여 체계적 문헌고찰에서 데이터를 (반)자동으로 추출하는 가능성을 탐구한 빠른 타당성 연구를 소개합니다. 최근 LLM에 대한 관심이 증가하고 있지만, LLM 기반 자동화 도구 설계와 성능 평가 방법에 대한 이해는 여전히 부족합니다.

- **Technical Details**: 우리는 2023 Evidence Synthesis Hackathon에서 두 가지 타당성 연구를 수행했습니다. 첫째로, 인간 임상, 동물, 사회과학 분야 연구에서 연구 특성을 자동으로 추출하기 위해 각 분야에서 두 개의 연구를 프롬프트 개발에 사용했고, 열 개의 연구를 평가에 사용했습니다. 둘째로, EBM-NLP 데이터셋 내 100개 초록에서 참가자(Participants), 개입(Interventions), 통제(Controls), 그리고 결과(Outcomes)(PICO) 정보를 예측하기 위해 LLM을 사용했습니다.

- **Performance Highlights**: 총체적으로, 전반적인 결과는 약 80%의 정확도를 보였으며, 도메인 간 가변성도 있었습니다(인간 임상 82%, 동물 연구 80%, 인간 사회과학 72%). 인과 추론 방법과 연구 설계 항목에서 가장 많은 오류가 발생했습니다. PICO 연구에서는 참가자와 개입/통제 부분에서 높은 정확도(>80%)를 보였지만, 결과 부분은 더 어려움이 있었습니다. 평가 또한 수작업으로 이루어졌으며, BLEU 및 ROUGE와 같은 평가지표는 한계가 있었습니다.



### Combining Denoising Autoencoders with Contrastive Learning to fine-tune Transformer Models (https://arxiv.org/abs/2405.14437)
Comments:
          1 figure, 7 tables, 12 pages

- **What's New**: 최근 Natural Language Processing (NLP) 커뮤니티에서는 대형 pretrained Transformer 모델을 사용하는 transfer learning이 주요 트렌드로 자리 잡고 있습니다. 본 논문에서는 classification task를 위해 3 단계 기법을 제안합니다.

- **Technical Details**: 첫 번째 단계에서는 Denoising Autoencoder (DAE)를 사용하여 모델의 신호를 데이터 분포에 맞게 조정합니다. 두 번째 단계에서는 Contrastive Learning (CL) 방법을 통해 출력의 표현 공간을 해당 클래스에 맞게 조정하고, 데이터 불균형 문제를 해결하기 위해 Supervised Contrastive Learning을 이용한 새로운 데이터 증강 방법을 도입합니다. 세 번째 단계에서는 fine-tuning을 통해 미리 정의된 카테고리를 제한합니다. 이러한 단계들은 모델이 최종 작업을 학습하는 데 필요한 관련 지식들을 제공합니다.

- **Performance Highlights**: 여러 데이터셋을 대상으로 한 광범위한 실험 결과를 통해 제안된 기법의 유효성을 입증했으며, ablation study 및 다른 기술들과의 비교 연구도 포함했습니다.



### RaFe: Ranking Feedback Improves Query Rewriting for RAG (https://arxiv.org/abs/2405.14431)
Comments:
          16 pages

- **What's New**: 이 논문에서는 주석 없이 쿼리 리라이트(query rewriting) 모델을 훈련하기 위한 새로운 프레임워크를 제안합니다. 이는 공용 재랭커(publicly available reranker)를 활용하여 리라이트 목표에 잘 맞는 피드백을 제공합니다.

- **Technical Details**: 기존의 쿼리 리라이트 방법은 주석(레이블이 붙은 관련 문서나 다운스트림 답안)이나 사전 설계된 보상(pre-designed rewards)이 필요했습니다. 그러나 이 논문은 이러한 주석 없이도 효과적인 피드백을 제공할 수 있는 방법을 제시하고 있습니다. 특히 LLMs과 RAG 기술의 발전을 활용하여, 강화 학습(reinforcement learning)을 통해 소형 모델을 사용해 성능을 향상시키는 접근방법입니다.

- **Performance Highlights**: 실험 결과, 이 프레임워크는 기존의 기준 방법들보다 더 나은 성능을 보인다고 합니다. 이는 쿼리 리라이트 목적에 보다 잘 맞는 신호를 활용했기 때문입니다.



### Mitigating Quantization Errors Due to Activation Spikes in GLU-Based LLMs (https://arxiv.org/abs/2405.14428)
- **What's New**: 최신 대형 언어 모델(LLMs)은 아키텍처 개선을 통해 최첨단 성능을 확립했지만, 추론(inference)에 상당한 계산 비용이 필요합니다. 이를 줄이기 위해, 훈련 후 양자화(post-training quantization, PTQ)를 통해 가중치 및 활성화를 낮은 정밀도로 양자화하는 방법이 인기를 끌고 있습니다. 이 논문에서는 GLU 변형이 있는 LLM에서 발생하는 활성화 양자화 문제를 다룹니다. 특히, FFN(feed-forward network)의 특정 레이어에서 과도한 활성화 크기 때문에 발생하는 지역 양자화 오류가 성능을 저하시킨다는 사실을 밝힙니다.

- **Technical Details**: 활성화 스파이크(activation spikes)가 FFN의 특정 레이어, 특히 초기 및 후반 레이어에서 발생한다는 체계적인 패턴을 확인했습니다. 이 스파이크들은 시퀀스 전반이 아닌 특정 토큰에 전념합니다. 이를 바탕으로 우리는 활성화 스파이크를 양자화 과정에서 격리하기 위한 두 가지 경험적 방법, QFeM(Quantization-free Module)과 QFeP(Quantization-free Prefix)를 제안합니다.

- **Performance Highlights**: LLaMA-2/3, Mistral, Mixtral, SOLAR, Gemma와 같은 최신 GLU 변형이 포함된 LLM에서, 우리의 방법이 실험적으로 유효함이 입증되었습니다. 특히, SmoothQuant와 같은 현재의 완화 기술이 활성화 스파이크를 제어하지 못하는 반면, 우리의 방법은 이를 개선하였습니다.



### Instruction Tuning With Loss Over Instructions (https://arxiv.org/abs/2405.14394)
Comments:
          Code is available at this https URL

- **What's New**: 본 연구에서는 Instruction Modelling (IM)이라는 새로운 방법을 제안합니다. 이 방법은 전통적인 방식이 출력 부분에만 손실 함수(Loss Function)를 적용하는 것과 달리, 모델이 지시에 따르는 방식과 프롬프트(Prompt) 부분에서도 손실 함수를 적용하여 교육(LMs)하는 것을 목표로 합니다. 이로 인해 다양한 벤치마크에서 성능이 개선됨을 확인했습니다.

- **Technical Details**: IM의 주요 기법은 지시(instruction)와 프롬프트 부분에 손실 함수를 적용하여 모델을 훈련시키는 것입니다. 이는 데이터셋 내 지시 길이와 출력 길이의 비율, 그리고 훈련 예제 수에 따라 성능에 큰 영향을 미칩니다. 길이가 긴 지시와 짧은 출력이 있는 데이터셋에서, 혹은 소량의 훈련 예제가 사용된 Superficial Alignment Hypothesis (SAH) 상황에서 IM의 효과가 특히 두드러집니다.

- **Performance Highlights**: 21개의 다양한 벤치마크에서 실험을 통해 IM의 효과를 검증했습니다. 놀랍게도, AlpacaEval 1.0에서는 모델 성능의 100% 이상을 향상시켰습니다. NLP 작업(MMLU, TruthfulQA, HumanEval 등)과 개방형 생성 벤치마크(MT-Bench, AlpacaEval 등)에서 IM이 모델 성능을 향상시키는 데 효과적임을 보여주었습니다.



### Emotion Identification for French in Written Texts: Considering their Modes of Expression as a Step Towards Text Complexity Analysis (https://arxiv.org/abs/2405.14385)
Comments:
          17 pages, 12 figures, submitted to ACL 2024 WASSA workshop

- **What's New**: 이 논문의 목표는 문서 내 문장이 감정을 표현하는지 여부(A), 감정을 표현하는 방식(B), 기본적 또는 복합적인 감정인지(C), 감정의 카테고리(D)를 예측하는 것입니다. 주요 기여 중 하나는 감정이 다양한 방식으로 표현될 수 있다는 사실을 통합한 데이터셋과 모델을 통해 이루어졌습니다. 이는 NLP 접근 방식이 일반적으로 고려하지 않는 간접적인 방식으로 감정이 제안될 수 있다는 점을 포함합니다.

- **Technical Details**: 이 논문은 대화형 데이터가 아닌 작성된 텍스트를 분석 대상으로 삼고 있습니다. 여기에서 표현 방식은 텍스트의 복잡성을 자동으로 분석하기 위한 요소로 간주됩니다. 프랑스어 텍스트에 대한 실험 결과는 인간 주석자 간의 합의와 비교했을 때 적절한 결과를 보여주었으며, in-context learning(콘텍스트 학습)을 사용한 대규모 언어 모델을 사용하는 것보다 뛰어난 결과를 나타냈습니다.

- **Performance Highlights**: 제안된 접근 방식은 인간 주석자와 비교하여 적절한 결과를 제공하는 것 외에도, 기존의 대규모 언어 모델을 사용하는 방법보다 더 나은 성능을 보였습니다.



### Perception of Knowledge Boundary for Large Language Models through Semi-open-ended Question Answering (https://arxiv.org/abs/2405.14383)
- **What's New**: 본 논문은 대형 언어 모델(LLMs)의 지식 경계(Knowledge Boundary, KB)와 관련된 새로운 연구를 제시하고 있습니다. 기존 연구는 명확한 정답이 있는 닫힌 질문에 초점을 맞췄지만, 본 연구는 많은 잠재적 답변을 가질 수 있는 반-개방형 질문(Semi-Open-Ended Questions, SoeQ)에 관한 문제를 다룹니다. 이를 통해 LLM의 지식 경계를 파악하고 더 많은 모호한 답변을 발견하려 합니다.

- **Technical Details**: 본 연구에서는 LLM 기반 접근법을 사용하여 SoeQ를 구성하고, 대상 LLM으로부터 답변을 얻습니다. 이때, 주류 블랙박스 LLM의 출력 확률을 이용할 수 없기 때문에 오픈소스 보조 모델(auxiliary model)을 사용해 모호한 답변을 탐색합니다. 보조 모델은 기존 답변의 근접한 의미 표현을 계산하여 확률을 추정하고, 높은 확률의 답변 생성을 줄여 보다 효과적인 생성을 달성합니다. RAG 기반 평가와 LLM 자체 평가를 비교하여, 목표 LLM의 KB를 넘어서는 네 가지 유형의 모호한 답변을 분류했습니다.

- **Performance Highlights**: 본 연구 방법을 통해 GPT-4의 KB를 인식하는 데이터셋을 구축했습니다. 결과적으로 GPT-4는 SoeQ에서 성능이 저조하고, 자신의 KB를 인식하지 못하는 경우가 많음을 발견했습니다. 또한, 보조 모델인 LLaMA-2-13B가 더 많은 모호한 답변을 효과적으로 발견하는데 유효하다는 점도 확인되었습니다.



### Can Large Language Models Create New Knowledge for Spatial Reasoning Tasks? (https://arxiv.org/abs/2405.14379)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)이 새로운 정보를 생성할 수 있는 잠재력이 있으며, 이는 연구와 혁신에 있어 큰 변화를 가져올 수 있음을 강조합니다. 이는 LLM이 훈련 중에 어떤 것을 보았는지 확인하기 어려운 측면에서 '새로움'을 입증하기 힘들지만, 연구진은 공간적 차원을 가진 문제에 대해 LLM이 정교한 추론을 수행할 수 있음을 관찰했습니다. 특히 Claude 3 모델이 이 분야에서 우수한 성능을 보였습니다.

- **Technical Details**: 논문은 LLM이 이전에 직접적으로 접하지 않았을 가능성이 높은 공간적 차원을 가진 문제에 대해서도 정교한 추론을 할 수 있다는 점을 강조합니다. 이로 인해 최첨단 LLM이 획득할 수 있는 이해 수준이 상당히 높음을 보여주며, LLM이 중요한 출현적 특성(emergent properties)을 지니고 있다는 주장을 뒷받침합니다.

- **Performance Highlights**: Claude 3 모델이 특히 이와 같은 공간적 문제 해결에 있어 좋은 성능을 보였으며, 이는 LLM이 이전에 보지 못한 문제에 대해서도 높은 수준의 이해와 추론 능력을 발휘할 수 있음을 시사합니다.



### MiniCache: KV Cache Compression in Depth Dimension for Large Language Models (https://arxiv.org/abs/2405.14366)
Comments:
          Tech report

- **What's New**: 이번 논문에서는 키-값(KV) 캐시를 압축하는 새로운 방법, 'MiniCache'를 도입했습니다. 이는 레이어 깊이 측면에서 KV 캐시의 메모리 사용량을 크게 줄이는 간단하지만 효과적인 접근법입니다. 이 방법은 주로 LLM(Large Language Models)의 중간부터 깊은 부분에서 인접 레이어 사이의 KV 캐시 상태가 높은 유사성을 보이는 점을 활용합니다.

- **Technical Details**: MiniCache는 KV 캐시 상태를 크기(magnitude)와 방향(direction)으로 분리한 후, 방향 벡터를 보간(interpolating)하여 길이를 유지하는 방식으로 병합을 수행합니다. 또한, 고유한 상태쌍을 유지하기 위한 토큰 유지 전략(token retention strategy)을 제안하며, 이는 최소한의 추가 저장 공간으로 정보를 보존할 수 있게 합니다. MiniCache는 별도의 훈련 없이도 적용 가능하며, 양자화(quantization) 및 희소성(sparsity) 등 기존의 KV 캐시 압축 전략을 보완합니다.

- **Performance Highlights**: LLaMA-2, LLaMA-3, Phi-3, Mistral, Mixtral 등의 여러 모델과 다양한 벤치마크에서 종합적인 평가를 통해, MiniCache는 우수한 압축률과 높은 스루풋을 달성함을 입증했습니다. 특히 ShareGPT 데이터셋에서, LLaMA-2-7B 모델의 4-bit MiniCache는 약 5.02x의 압축률, 약 5배의 추론 스루풋 향상, 그리고 FP16 전체 캐시 기준으로 41%의 메모리 사용량 감소를 실현하였으며, 성능 저하는 거의 없습니다.



### JiuZhang3.0: Efficiently Improving Mathematical Reasoning by Training Small Data Synthesis Models (https://arxiv.org/abs/2405.14365)
Comments:
          28 pages, SOTA math LLM using Well-trained Data Synthesis LLM

- **What's New**: 수학적 추론 능력 향상을 위해 거대 언어 모델(LLM)을 학습시키는 새로운 방법이 제안되었습니다. 오픈 소스 텍스트를 활용하여 비용을 절감하고, 소규모의 LLM을 사용해 고품질의 학습 데이터를 효율적으로 생성하는 방식을 소개합니다. 이 과정에서 GPT-4를 활용해 데이터 생성 능력을 소규모 LLM에 전달하는 데이터셋을 구축하였습니다.

- **Technical Details**: 수작업으로 제작된 프롬프트를 사용하여 GPT-4가 다양한 수학 지식과 난이도를 아우르는 문제를 생성하도록 하였습니다. 또한, gradient 기반의 영향 추정 방법을 사용해 가장 가치 있는 수학 관련 텍스트를 선택하였습니다. 이렇게 선택된 텍스트와 GPT-4 데이터를 결합하여 소규모 LLM을 훈련시키는 데이터셋을 만들었습니다.

- **Performance Highlights**: 제안된 JiuZhang3.0 모델은 600만 개의 수학 문제를 생성하고, 이를 통해 여러 수학적 추론 데이터셋에서 최첨단 성과를 달성했습니다. 이는 자연어 추론 및 도구 조작 설정 모두에서 우수한 결과를 보여주며, 모델 학습 과정에서 GPT-4 API 호출 횟수를 9.3천 번으로 줄이고, 4.6B 데이터에서 사전 학습을 완료했습니다.



### Improving Language Models Trained with Translated Data via Continual Pre-Training and Dictionary Learning Analysis (https://arxiv.org/abs/2405.14277)
Comments:
          15 pages

- **What's New**: 이 연구에서는 영어에서 아랍어로 기계 번역(MT)된 데이터를 활용하여 어린이용 짧은 이야기 데이터셋(TinyStories, 2.2M)의 번역과 합성 데이터를 평가하여 언어 모델을 훈련시키는 방법을 조사했습니다. 이 연구는 단순한 번역이 아닌, 번역 데이터의 품질 문제와 문화적 편향을 개선할 수 있는 방법을 제안합니다.

- **Technical Details**: 연구팀은 무료 NLLB-3B MT 모델을 사용해 영어 TinyStories 데이터를 아랍어로 번역했습니다. 이후 이를 사용해 1M-33M 파라미터 크기의 여러 이야기 생성 모델을 훈련시켰으며, 여기서 발생한 품질 문제와 과제 특화 문제를 분석했습니다. 문제를 해결하기 위해 원본 훈련 데이터의 1%에 해당하는 고품질 합성 이야기 소규모 데이터셋을 추가적으로 사전 훈련했습니다.

- **Performance Highlights**: GPT-4를 판정자로 사용한 평가와 기계적 해석가능성 측면에서 제안된 접근 방식이 번역에서 발생하는 문제를 해결하는 데 실질적으로 유용함을 입증했습니다. 언어적 문제와 문화적 편향을 사례 연구를 통해 개선된 결과를 제시했습니다.



### Let's Fuse Step by Step: A Generative Fusion Decoding Algorithm with LLMs for Multi-modal Text Recognition (https://arxiv.org/abs/2405.14259)
- **What's New**: 이번에 소개된 'Generative Fusion Decoding(GFD)'은 대형 언어 모델(LLM)을 다중 모달 텍스트 인식 시스템(예: 자동 음성 인식(ASR) 및 광학 문자 인식(OCR))에 통합하는 혁신적인 얕은 융합 프레임워크입니다. 이 방법은 다양한 모델의 불일치된 토큰 공간을 바이트 토큰 공간으로 매핑하여 디코딩 과정에서 매끄러운 융합을 가능케 합니다.

- **Technical Details**: GFD 프레임워크는 재학습이 필요 없이 다양한 자기회귀 모델에 플러그 앤 플레이 방식으로 호환됩니다. 이는 이전의 융합 기법들이 가진 한계를 극복합니다. GFD는 텍스트 토큰 공간을 바이트 토큰 공간으로 변환하여 융합을 지원하며, 이는 복잡한 샘플 공간 정렬 과정을 단순화합니다. 이로 인해 LLM이 인식 모델과 협력하여 오류를 수정할 수 있고, 이에 따른 계산 지연 시간도 줄어듭니다.

- **Performance Highlights**: 평가 결과, GFD는 ASR과 OCR 작업에서 성능을 크게 향상시킵니다. 특히 ASR은 NTUML2021 벤치마크에서 최첨단 성능을 달성했습니다. GFD는 중국어 텍스트 인식에 약한 인식 모델과 중국어로 광범위하게 학습된 LLM을 융합할 수 있게 하여 이 분야의 중요한 발전을 이루어냈습니다. 또한, GFD는 장기 음성 인식과 지시 인식이 필요한 상황에서도 높은 견고성을 제공합니다.



### Text-Based Correlation Matrix in Multi-Asset Allocation (https://arxiv.org/abs/2405.14247)
Comments:
          4 pages, 4 figures, 1 tables

- **What's New**: 이번 연구에서는 금융 텍스트 분석을 통해 다수 자산 간의 상관 구조를 추정하려고 합니다. 최근 글로벌 경제에서 인플레이션 상승과 중앙은행의 긴축 통화 정책 배경 하에서 자산 간의 상관 구조가 크게 변화했음을 조명하고 있습니다. 이러한 변화는 투자자 포트폴리오의 성과에 큰 영향을 미치므로, 포트폴리오 관리에서 견고한 상관 구조를 추정하는 것이 중요해졌습니다.

- **Technical Details**: 연구진은 뉴스 텍스트와 중앙은행 텍스트에 대한 자연어 처리(Natural Language Processing; NLP)를 수행하여 미래 상관 계수 변화의 예측 정확성을 검증했습니다. 기존의 재무 시장에서 관찰된 과거 가격 데이터를 사용한 상관 계수는 시차(time lag)와 금융 시계열 데이터의 비정상성(nonlinearity)에 따른 예측 오류의 문제, 및 근본적 관점에서의 해석 가능성이 낮다는 단점이 있습니다. 이러한 문제를 해결하기 위해 텍스트 분석을 도입했습니다.

- **Performance Highlights**: 연구 결과, 뉴스 텍스트와 중앙은행 텍스트를 통한 자연어 처리 방법이 일반적인 시계열 데이터로부터의 예측보다 미래의 상관 계수 변화를 예측하는 데 있어 유용하다고 제안되었습니다.



### Language processing in humans and computers (https://arxiv.org/abs/2405.14233)
Comments:
          100 pages, 64 figures; lecture notes, book draft

- **What's New**: 최근 기계 학습된 언어 모델은 일상 생활에 큰 변화를 가져왔습니다. 이들은 우리가 공부할 때나 운전할 때, 돈을 관리할 때 우리를 안내하며, 문명을 변혁할 잠재력을 가지고 있습니다. 하지만 이러한 시스템은 헛것을 보는(hallucinate) 경향이 있습니다. 이 논문은 언어 모델에 대한 개요와 더불어 학습 기계에 대한 저수준(low-level) 모델을 제시합니다.

- **Technical Details**: 이 논문은 언어 모델이 인간과 유사하게 헛것을 인식하고 안전하게 꿈을 꿀 수 있는 능력을 갖추게 된 후에도, 더 넓은 범위의 거짓 신념과 자기 확인 이론(self-confirming theories)을 생성하게 되는 과정을 설명합니다.

- **Performance Highlights**: 언어 학습 기계가 헛것을 인식하는 수준에 도달한 후, 이들이 더 넓고 복잡한 시스템을 형성하는 것은 인간의 사고 과정과 많은 유사점을 보입니다. 이러한 발견은 언어 모델의 잠재력과 한계에 대한 중요한 통찰을 제공합니다.



### From Role-Play to Drama-Interaction: An LLM Solution (https://arxiv.org/abs/2405.14231)
Comments:
          Accepted by ACL 2024 Findings

- **What's New**: 이번 논문에서는 	extit{LLM 기반 인터랙티브 드라마 (LLM-based interactive drama)}를 소개합니다. 이는 전통적인 드라마에 몰입도를 한층 높인 새로운 형태로, 사람이 드라마에 직접 들어가 캐릭터 및 장면과 상호작용할 수 있게 해줍니다. 6가지 필수 요소로 플롯(plot), 캐릭터(character), 생각(thought), 표현(diction), 볼거리(spectacle), 그리고 상호작용(interaction)을 정의합니다.

- **Technical Details**: 연구팀은 	extit{drama LLM}이라는 드라마 실행을 주도하는 백본 모델을 설계했습니다. 이 모델은 제한된 드라마 자원, 통제 불가능한 내러티브 전개, 복잡한 명령어 추종 문제들을 극복해야 합니다. 이를 위해 	extit{Narrative Chain}을 제안해 플레이어들과의 상호작용 중 내러티브 진행을 더 세밀하게 제어하고, 다양한 스토리를 바탕으로 드라마 스크립트를 생성하는 	extit{Auto-Drama}, 그리고 모델이 복잡한 명령을 따를 수 있게 하는 	extit{Sparse Instruction Tuning} 등을 도입했습니다.

- **Performance Highlights**: 수작업으로 작성된 	extit{Detective Conan}, 	extit{Harry Potter}, 	extit{Romeo and Juliet}의 3가지 스크립트를 사용하여 드라마 LLM을 평가했으며, 이를 위해 5가지 차원의 평가 기준을 설계했습니다.



### ChronosLex: Time-aware Incremental Training for Temporal Generalization of Legal Classification Tasks (https://arxiv.org/abs/2405.14211)
Comments:
          Accepted to ACL 2024

- **What's New**: 이번 연구는 법률 관련 다중 레이블 텍스트 분류 작업에서 시간의 흐름에 따라 변하는 법률 개념의 도전에 대해 조사합니다. 기존 모델들이 훈련 과정에서 시간적 차원을 간과함에 따라 시간이 지나면서 성능이 떨어지는 문제를 해결하기 위해 'ChronosLex'라는 점진적 학습 패러다임을 도입했습니다.

- **Technical Details**: 'ChronosLex'는 데이터를 시간 순서대로 보존하면서 연대별로 나누어 모델을 훈련하는 방법입니다. 하지만 최근 데이터에 과적합(overfitting)되는 문제를 방지하기 위해 계속학습(continual learning)과 시간 불변 방법(temporal invariant methods)을 사용한 완화 전략을 평가합니다.

- **Performance Highlights**: 여섯 개의 법률 다중 레이블 텍스트 분류 데이터셋에 대한 실험 결과, 계속학습 방법이 과적합을 방지하고 시간적 일반화(temporal generalizability)를 향상시키는 데 효과적임을 보여주었으며, 시간 불변 방법은 시간적 변화의 동태를 포착하기 어려웠습니다.



### Agent Planning with World Knowledge Mod (https://arxiv.org/abs/2405.14205)
Comments:
          Work in progress

- **What's New**: 이 논문에서는 파라메트릭 월드 지식 모델(World Knowledge Model, WKM)을 도입하여 에이전트(Agent) 모델의 계획 성능을 개선하는 방법을 제안합니다. WKM은 전문가와 샘플 경로로부터 지식을 자체 합성하여 글로벌 및 로컬 계획 모두에서 에이전트 모델을 안내할 수 있도록 합니다.

- **Technical Details**: 이 논문은 최신 오픈소스 대형 언어 모델(LLM)인 Mistral-7B, Gemma-7B, 그리고 Llama-3-8B를 사용하여 WKM을 구현하고 실험을 수행했습니다. WKM은 작업 전에는 글로벌 사전 지식을 제공하고, 작업 중에는 동적 상태 지식을 유지하여 에이전트의 로컬 계획을 지원합니다.

- **Performance Highlights**: 세 가지 복잡한 실제 시뮬레이션 데이터셋에서 실험한 결과, WKM을 사용한 방법이 여러 강력한 기존 기법들에 비해 우수한 성능을 보였습니다. 또한 WKM은 맹목적인 시행착오와 환상적인 행동 문제를 효과적으로 완화할 수 있음을 입증하였습니다. 추가로, 인스턴스 레벨의 작업 지식이 보이지 않는 작업에도 잘 일반화될 수 있으며, 약한 WKM이 강한 에이전트 모델 계획을 안내할 수 있다는 사실이 발견되었습니다. 마지막으로, 통합된 WKM 훈련은 추가 발전 가능성을 지니고 있습니다.



### Semantic-guided Prompt Organization for Universal Goal Hijacking against LLMs (https://arxiv.org/abs/2405.14189)
Comments:
          15 pages

- **What's New**: 대형 언어 모델(LLMs)의 신뢰성을 높이기 위해 새로운 보편적 목표 탈취(universal goal hijacking) 방법인 POUGH를 제안합니다. 이전 연구들은 최적화 알고리즘에만 초점을 맞추었으나, 이 연구에서는 의미 기반의 프롬프트 처리 전략을 통합했습니다.

- **Technical Details**: POUGH 방법은 후보군 풀(candidates pool)에서 대표적인 프롬프트(prompts)를 선택하는 샘플링 전략으로 시작됩니다. 그런 다음 프롬프트를 우선 순위에 따라 정렬하는 랭킹 전략을 사용합니다. 이러한 프롬프트들이 순차적으로 정리되면, 반복 최적화 알고리즘(iterative optimization algorithm)을 통해 보편적인 고정 접미사(universal fixed suffix)를 생성합니다.

- **Performance Highlights**: POUGH 방식은 네 가지 인기 있는 대형 언어 모델(LLMs)과 열 가지 유형의 목표 응답(target responses)에서 실험을 통해 효과가 입증되었습니다.



### UzMorphAnalyser: A Morphological Analysis Model for the Uzbek Language Using Inflectional Endings (https://arxiv.org/abs/2405.14179)
Comments:
          6 pages, 4 figures

- **What's New**: 이 논문은 우즈베크어 형태소 분석 (morphological analysis)을 모델링 한 결과를 발표했습니다. 이는 단어의 어근 (root)과 접사 (affixes)가 결합되어 형성되는 우즈베크어의 형태적 특징을 다룹니다. 제안된 모델을 기반으로 개발된 도구는 웹 애플리케이션 및 오픈 소스 Python 라이브러리로 제공됩니다.

- **Technical Details**: 모델링의 주요 단계에는 형태 정보를 할당된 단어 끝 세트 (word-ending)를 완전하게 개발하고, 형태소 분석을 위한 추가 데이터 세트를 구성하는 것이 포함됩니다. 이 과정에서는 형태 음운 예외 (morpho-phonetic exceptions)를 고려하여 어간 추출 (stemming), 표제어 추출 (lemmatizing), 형태 정보 추출을 수행합니다.

- **Performance Highlights**: 제안된 모델은 5.3K 단어로 구성된 테스트 세트를 사용하여 평가되었습니다. 형태소 특성 수정, 어간 추출 및 표제어 추출에 대한 언어 전문가의 수동 검증을 통해 91% 이상의 단어 수준 정확도를 얻었습니다.



### Self-Taught Recognizer: Toward Unsupervised Adaptation for Speech Foundation Models (https://arxiv.org/abs/2405.14161)
Comments:
          23 pages, Preprint

- **What's New**: 새로운 무감독 적응 프레임워크(Self-TAught Recognizer, STAR)를 제안하여, 라벨링되지 않은 데이터를 활용해 잡음과 억양 등의 다양한 타겟 도메인에서도 자동 음성 인식(ASR) 시스템의 견고성을 향상시킵니다. 이 프레임워크는 Transformer-관련 구조와 자동 회귀 디코딩(auto-regressive decoding, 예: Whisper, Canary) 기반의 주요 음성 기초 모델(speech foundation models)에 적용됩니다.

- **Technical Details**: STAR는 디코딩 동안 단계별 정보를 통합하여 참조 라벨 없이 토큰 단위의 품질을 평가하는 새로운 지표를 제공합니다. 이를 통해 모델 업데이트를 안내하여 효과적인 무감독 적응을 가능하게 합니다. 실험 결과, STAR는 평균 13.5%의 단어 오류율(word error rate)을 줄이며, 때로는 감독된 적응의 상한 성능에 근접할 수도 있습니다. 또한, STAR는 소스 도메인 데이터를 호출하지 않고도 흔히 발생하는 치명적 잊힘 문제(catastrophic forgetting)를 방지하는 것으로 나타났습니다.

- **Performance Highlights**: STAR는 14개의 타겟 도메인에서 평균 13.5%의 상대적 단어 오류율 감소를 달성했습니다. 더불어, 고작 1시간도 안 되는 라벨링되지 않은 데이터만 필요로 하는 높은 데이터 효율성을 보입니다. 또한, 다른 주요 음성 모델이나 음성 번역 작업에도 원활하게 적용됩니다.

- **Additional Info**: STAR의 코드는 연구 커뮤니티에 오픈 소스로 제공할 계획입니다.



### Super Tiny Language Models (https://arxiv.org/abs/2405.14159)
Comments:
          11 pages, 4 figures

- **What's New**: 대형 언어 모델(LLM)의 발전은 자연어 처리에서 큰 개선을 가져왔지만, 높은 계산 및 에너지 요구사항 때문에 도전 과제가 존재합니다. 이번 연구는 이 문제를 해결하기 위해 Super Tiny Language Models(STLMs)를 제안합니다. STLMs는 파라미터 수를 크게 줄이면서도 우수한 성능을 제공할 수 있는 새로운 기술을 탐구합니다.

- **Technical Details**: 이 연구에서는 byte-level 토크나이제이션(tokenization)과 풀링 메커니즘(pooling mechanism), weight tying, 그리고 효율적인 학습 전략과 같은 혁신적인 기술을 탐구합니다. 이러한 방법들은 전통적인 모델들과 비교했을 때 파라미터 수를 90%에서 95%까지 줄이면서도 경쟁력 있는 성능을 유지합니다. 또한, tokenizer-free 모델, self-play 기반 학습, 그리고 대체 학습 목표와 같은 다양한 하위 문제들을 탐구하고 있습니다.

- **Performance Highlights**: 연구는 10M, 50M, 100M 파라미터 모델을 목표로 하며, 높은 성능의 언어 모델을 다양한 애플리케이션에 더 접근 가능하고 실용적으로 만들기 위한 궁극적인 목표를 가지고 있습니다. 이는 파라미터 수를 크게 줄이면서도 기존의 모델과 비교해 성능 면에서 큰 차이가 없음을 보여줍니다.



### jp-evalb: Robust Alignment-based PARSEVAL Measures (https://arxiv.org/abs/2405.14150)
Comments:
          To appear in The system demonstration track at NAACL-HLT 2024

- **What's New**: 이번 연구에서는 PARSEVAL 측정을 계산하기 위해 고안된 평가 시스템을 소개합니다. 이 시스템은 전통적으로 구성 구문 분석 평가에 사용되던 	exttt{evalb}의 대안으로 제안되었습니다. 새로운 시스템인 	exttt{jp-evalb}는 정렬 방법을 기반으로 하여 문장과 단어의 불일치를 해결하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 	exttt{evalb} 스크립트는 문자화 및 문장 경계의 일관성을 요구합니다. 반면 	exttt{jp-evalb}는 `jointly preprocessed (JP)' 정렬 기반 방법을 사용하여 이러한 일관성 문제를 해결하려고 합니다. 	exttt{jp-evalb}는 문장 및 단어 정렬을 통해 유연하고 적응 가능한 프레임워크를 제공합니다.

- **Performance Highlights**: 이 새로운 시스템은 	exttt{evalb}를 사용하면서 발생했던 여러 문제를 극복할 수 있도록 설계되었습니다. 이를 통해 구성 구문 분석 성능에 대한 보다 정확한 평가가 가능해질 것입니다.



### ViHateT5: Enhancing Hate Speech Detection in Vietnamese With A Unified Text-to-Text Transformer Mod (https://arxiv.org/abs/2405.14141)
Comments:
          Accepted at ACL'2024 (Findings)

- **What's New**: 베트남어 혐오 발언 탐지(Hate Speech Detection; HSD) 기술의 발전에 대응하여, T5 기반 모델인 ViHateT5가 소개되었습니다. 이 모델은 VOZ-HSD라는 대규모 도메인 특화 데이터셋을 사용하여 사전 학습되었습니다. 이 연구는 다양한 HSD 과제에 대해 통합된 모델을 사용하여 최첨단 성능을 달성할 수 있는 가능성을 보여줍니다.

- **Technical Details**: ViHateT5는 텍스트-투-텍스트(text-to-text) 아키텍처를 활용하여 다중 작업(multi-tasking)을 처리할 수 있습니다. 이 모델은 베트남어 혐오 발언 탐지에서 뛰어난 성능을 보이며, 실험 결과 레이블 분포가 모델의 효율성에 중요한 역할을 한다는 점을 강조합니다. 또한, 연구 팀은 VOZ-HSD 데이터셋, pre-trained checkpoint, 통합 HSD-다중작업 ViHateT5 모델, 관련 소스 코드를 GitHub에 공개했습니다.

- **Performance Highlights**: ViHateT5 모델은 베트남어 혐오 발언 탐지의 모든 표준 벤치마크에서 state-of-the-art 성능을 달성했습니다. 이를 통해 일반적인 사전 학습 모델이 아닌 도메인 특화된 데이터로 학습된 모델의 중요성을 입증했습니다.



### AlignGPT: Multi-modal Large Language Models with Adaptive Alignment Capability (https://arxiv.org/abs/2405.14129)
Comments:
          Code and models are available at $\href{this https URL}{\textit{this https URL}}$

- **What's New**: 이번 연구에서는 다중모달 대규모 언어 모델(Multimodal Large Language Models, MLLM)이 인공지능 일반(GAI)의 탐구에 중요한 역할을 한다고 평가합니다. 특히 MLLM의 핵심은 크로스모달 정렬(cross-modal alignment)을 달성하는 능력에 있습니다. 이를 개선하기 위해 새로운 다중모달 대규모 언어 모델 AlignGPT를 제안합니다.

- **Technical Details**: 기존의 MLLM 모델은 사전 학습(pre-training) 단계와 명령 튜닝(instruction-tuning) 단계의 두 가지로 나뉘어 훈련됩니다. 그러나 기존의 모델들은 이미지-텍스트 쌍이 균일하게 정렬되었다는 가정하에 훈련되어, 실제로는 정렬 정도가 다르다는 문제점이 있습니다. 이를 개선하기 위해 AlignGPT는 사전 학습 단계에서 이미지-텍스트 쌍의 정렬 수준을 다르게 부여합니다. 그리고 명령 튜닝 단계에서는 다양한 명령의 정렬 요구에 맞추어 적응적으로 정렬 수준을 결합합니다.

- **Performance Highlights**: 광범위한 실험 결과, AlignGPT는 12개의 기준 벤치마크에서 경쟁력 있는 성능을 보여주었습니다.



### Knowledge Localization: Mission Not Accomplished? Enter Query Localization! (https://arxiv.org/abs/2405.14117)
- **What's New**: 대규모 언어 모델(LLMs)이 방대한 사실 지식을 저장하고 있지만, 이 지식을 어떻게 저장하고 표현하는지는 명확하지 않습니다. 본 논문에서는 Knowledge Neuron(KN) 가설을 재검토하고, 새로운 Query Localization(QL) 가정을 제시합니다. 이는 기존의 Knowledge Localization(KL) 가정이 지식 저장과 표현 메커니즘을 충분히 설명하지 못한다는 점을 강조합니다.

- **Technical Details**: 기존의 KL 가정은 하나의 사실이 소수의 지식 저장 단위인 Knowledge Neurons에 국한될 수 있다고 제안합니다. 그러나 통계적 및 지식 수정 관점에서 KL 가정을 따르지 않는 사실들이 존재함을 확인했습니다. 이에 따라 제안된 QL 가정은 (1) Query-KN Mapping: 로컬라이제이션 결과가 사실이 아닌 쿼리와 연관되고, (2) Dynamic KN Selection: 쿼리에 응답하기 위해 KN을 선택하는 데 주의 모듈이 기여한다는 내용으로 구성되어 있습니다. 이를 기반으로 Consistency-Aware KN 수정 방법을 제안하여 지식 수정의 성능을 향상시킵니다.

- **Performance Highlights**: 총 39개 세트의 실험과 추가적인 가시화 실험을 통해 우리의 결론을 철저히 검증했습니다. 이러한 검증을 통해 QL 가정과 새로운 수정 방법의 효과성을 입증했습니다.



### Large Language Models Can Self-Correct with Minimal Effor (https://arxiv.org/abs/2405.14092)
Comments:
          Work in Progress

- **What's New**: 본 연구는 대형 언어 모델(LLMs)이 외부 피드백 없이도 자신이 만든 응답을 검토하고 수정할 수 있는 방법(a method)인 Intrinsic self-correct을 탐구했습니다. 연구 결과, LLMs가 아직 자체적으로 추론을 수정할 수 없다는 결론을 내렸습니다. 그러나 간단하고 효과적인 검증 방법이 LLMs의 본래 성능을 이끌어낼 수 있음을 발견했습니다. 질문에서 핵심 조건을 마스크하고 현재 응답을 추가하여 검증 질문을 구성한 후 조건을 예측해 응답을 검증하는 방식입니다.

- **Technical Details**: 우리의 제안은 Iterative verify-then-correct 프레임워크(PoCo)를 통해 점진적으로 잘못된 응답을 식별하고 수정하는 것입니다. 핵심 조건은 개방형 질문의 엔터티(entity)나 수학적 질문에서의 숫자 값처럼 최소한의 노력으로 파악될 수 있는 요소입니다. 이 방식은 프롬프트(prompt)를 통해 쉽게 구현됩니다. GPT-3.5-Turbo를 백엔드 LLM으로 사용하여 세 가지 추론 작업에서 실험을 수행했습니다.

- **Performance Highlights**: ProCo는 평균적으로 네 개의 개방형 질의 응답 데이터셋에서 정확도(exact match)가 +6.8 증가하고, 세 개의 산술 추론 데이터셋에서 정확도가 +14.1, 그리고 상식 추론 데이터셋에서 정확도가 +9.6 증가했습니다. 이는 Self-Correct 방법과 비교하여 상당한 성능 향상입니다.



### $T^2$ of Thoughts: Temperature Tree Elicits Reasoning in Large Language Models (https://arxiv.org/abs/2405.14075)
Comments:
          10 pages, 5 figures

- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)의 추론 능력을 강화하기 위해 Temperature Tree($T^2$) 프롬팅 기법을 파티클 스웜 옵티마이제이션(Particle Swarm Optimization, PSO)과 결합한 $T^2$ of Thoughts($T^2oT$) 방법을 제안합니다. 이 방법은 온도와 같은 검색 매개변수를 동적으로 조정함으로써 정확도를 높이고자 하며, 복잡한 결정-making 시나리오에서 특히 유용합니다.

- **Technical Details**: 이 연구는 $T^2oT$ 접근 방식을 통해 고정된 검색 깊이와 결합된 적응형 온도 조절의 효과를 탐구합니다. 이를 통해 단일 솔루션 정확도, 다중 솔루션 생성, 텍스트 생성 품질 등을 향상시키는 기술을 개발하였습니다. 특히, 이 기법은 게임 오브 24와 창의적 글쓰기(Creative Writing)와 같은 특정 작업에서 검증되었습니다.

- **Performance Highlights**: $T^2oT$ 접근 방식은 단일 솔루션 정확도, 다중 솔루션 생성 및 텍스트 생성 품질에 상당한 개선을 가져왔습니다. 특히 동적 검색 깊이 조정은 혼합된 결과를 보였지만, 고정된 검색 깊이와 조합된 적응형 온도 조절의 효과가 더 안정적이고 다목적적인 문제 해결 전략으로 나타났습니다.



### Your Large Language Models Are Leaving Fingerprints (https://arxiv.org/abs/2405.14057)
- **What's New**: 최근 연구에 따르면, 인간과 기계가 생성한 텍스트를 구별하는 데 있어, 고도화된 트랜스포머와 다른 지도 학습 감지기들이 충분히 효과적이라는 것이 증명되었습니다. 하지만 본 연구에서는 심지어 단순한 n-그램(n-gram) 및 품사(part-of-speech) 특징을 활용한 분류기(classifier)조차도 높은 성능을 보인다는 것을 발견했습니다.

- **Technical Details**: 연구팀은 기계가 생성한 텍스트에 내재된 독특한 지문(fingerprint)을 분석하기 위해 다섯 가지 데이터셋을 사용했습니다. 이러한 지문은 특정 어휘 및 형태구문적(morphosyntactic) 특징의 빈도수 차이로 나타납니다. 연구에서는 이러한 지문을 시각화하고, 이를 텍스트 감지에 사용하는 방법을 설명했습니다. 또한 이러한 지문이 텍스트 도메인을 넘어 매우 강력하다는 것을 확인했습니다.

- **Performance Highlights**: 연구 결과, 같은 모델 패밀리 내의 모델들(예: llama-13b와 llama-65b) 간에 지문이 일관되게 나타났습니다. 특히, 채팅에 최적화된 모델들이 일반 언어 모델보다 더 쉽게 감지되는 경향이 있었습니다. 이는 LLM 지문이 훈련 데이터에 의해 직접적으로 유도될 수 있음을 시사합니다.



### How Many Bytes Can You Take Out Of Brain-To-Text Decoding? (https://arxiv.org/abs/2405.14055)
- **What's New**: 뇌-컴퓨터 인터페이스는 말하기 지원 및 뇌 연구에서 유망한 의학적 및 과학적 응용을 가지고 있습니다. 이 연구에서는 뇌-텍스트 변환기(brain-to-text decoders)를 평가하기 위한 정보 기반의 평가 지표(information-based evaluation metric)를 제안합니다. 이를 통해 기존 최첨단 연속 텍스트 변환기(continuous text decoders)를 향상시키는 두 가지 방법을 검토하였습니다.

- **Technical Details**: 연구에서 제안된 정보 기반 평가 지표를 사용하여, 두 가지 방법이 결합될 때 베이스라인 모델과 비교하여 뇌 디코딩 성능이 40% 이상 향상될 수 있음을 보여주었습니다. 또한 뇌-텍스트 변환기의 정보적 특성을 분석하여, 지프의 법칙(Zifian power law) 동역학이 있다는 것을 경험적으로 입증하였습니다.

- **Performance Highlights**: 연구의 분석을 통해 fMRI 기반의 텍스트 변환기(fMRI-based text decoder)의 이상적인 성능을 추정했습니다. 이를 현재 모델과 비교한 결과, 주요 디코딩 오류의 원인을 양적으로 평가할 수 있었습니다. 결론적으로, 추가적인 알고리즘 개선이 이루어진다면 실용적인 뇌-텍스트 변환기가 가능할 것이라는 희망적인 전망을 제시합니다.



### Trajectory Volatility for Out-of-Distribution Detection in Mathematical Reasoning (https://arxiv.org/abs/2405.14039)
Comments:
          27 pages, 6 figures, 12 tables

- **What's New**: 딥 네트워크의 보안을 강화하기 위해 다양한 out-of-distribution (OOD) detection 알고리즘들이 제안되었습니다. 이번 논문에서는 수학적 추론 (mathematical reasoning) 시나리오에서의 OOD를 탐지하는 새로운 접근법, TV score에 대해 설명합니다. 이 방법은 임베딩 기반 방법의 한계를 극복하고자 궤적 변동성 (trajectory volatility)을 사용했습니다.

- **Technical Details**: 기존의 생성 언어 모델 (GLMs)에서는 불확실성 추정과 임베딩 거리 측정에 집중했으나, 수학적 추론 같은 복잡한 생성 시나리오에서는 임베딩 기반 방법이 어려움이 많습니다. TV score는 각 샘플의 잠재 공간 내 궤적 변동성을 통해 OOD를 탐지하며, 이는 출력 공간의 고밀도 특성으로 인한 임베딩 이동 궤적의 차이를 이용합니다.

- **Performance Highlights**: 실험 결과, TV score는 수학적 추론 시나리오에서 기존 모든 전통적인 알고리즘을 능가했으며, 다중 선택 질문처럼 출력 공간에서 고밀도 특성을 가진 더 많은 응용 분야로 확장 가능합니다.



### Evaluating Large Language Models with Human Feedback: Establishing a Swedish Benchmark (https://arxiv.org/abs/2405.14006)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 스웨덴어 이해 및 생성 능력을 평가하기 위한 새로운 인간 벤치마크를 도입했습니다. 이 연구는 ChatbotArena 벤치마크의 수정 버전을 사용하여 GPT-4, GPT-3.5, 다양한 Claude 및 Llama 모델, 그리고 Dolphin-2.9-llama3b-8b-flashback 및 BeagleCatMunin과 같은 맞춤형 모델을 포함한 11개의 모델을 평가합니다.

- **Technical Details**: Forced choice ranking을 사용하여 모델의 성능을 측정하고, 인간의 피드백을 포함한 수정된 ChatbotArena 벤치마크를 적용했습니다. LMSYS chatbot arena와 Scandeval 벤치마크에서의 성능을 기준으로 모델을 선정하였습니다. 또한, 연구의 일환으로 이 벤치마크를 공개하여 스웨덴어 언어 모델 성능에 대한 이해를 증진하고자 합니다.

- **Performance Highlights**: 벤치마크에 사용된 다양한 모델 중 GPT-4, GPT-3.5 등을 평가하며, 충분한 데이터가 수집되고 분석된 후 리더보드를 생성할 계획입니다.



### Feedback-aligned Mixed LLMs for Machine Language-Molecule Translation (https://arxiv.org/abs/2405.13984)
- **What's New**: 이 연구는 화학과 인공지능(AI)의 교차점을 탐구하면서, 자동화된 언어-분자 번역(task of automated language-molecule translation)에 대해 최신 인간 중심 최적화 알고리즘(human-centric optimisation algorithms)을 처음으로 사용하여 크로스 모달(cross-modal) 정렬을 성공적으로 달성했습니다. 또한, 비선형 융합(non-linear fusion)을 처음으로 제안하여 크로스 모달 LLMs의 성능을 향상시켰습니다.

- **Technical Details**: 연구팀은 과학적 대형 언어 모델(scientific LLMs)의 능력을 광범위한 데이터나 큰 모델 없이도 향상시킬 수 있음을 실험적으로 증명했습니다. 이는 훈련 데이터의 10%만 사용하여 달성하였으며, 광범위한 데이터셋에서 큰 모델을 훈련할 때 발생하는 기억 효과를 완화하는 데 도움이 되었습니다. 결과적으로 새로운 SOTA 수준에 도달했습니다. 또한, 할루시네이션(hallucination)을 평가하기 위한 미세한, 도메인에 의존하지 않는 평가 방법(fine-grained, domain-agnostic evaluation method)을 도입하여 책임 있는 사용을 촉진했습니다.

- **Performance Highlights**: 제안된 방법은 기존 모델들이 광범위한 데이터로 훈련되었을 때의 최고 벤치마크 모델을 크게 능가하는 성능 향상을 달성했습니다. 또한, 훈련 비용이나 데이터 필요량을 증가시키지 않고도 성능 향상을 가져왔다는 점에서 주목할 만합니다.



### CIVICS: Building a Dataset for Examining Culturally-Informed Values in Large Language Models (https://arxiv.org/abs/2405.13974)
- **What's New**: 본 논문은 'CIVICS: Culturally-Informed & Values-Inclusive Corpus for Societal impacts' 데이터셋을 소개합니다. 이는 대형 언어 모델(LLMs)의 사회적 및 문화적 변이를 여러 언어와 가치 민감한 주제들을 통해 평가하도록 설계되었습니다. 이 데이터셋은 LGBTQI 권리, 사회 복지, 이민, 장애인 권리, 대리모 등의 특정 사회적으로 민감한 주제를 다루는 다국어 가치를 포함한 프롬프트로 구성됩니다.

- **Technical Details**: CIVICS 데이터셋은 LLMs의 내재된 암묵적 가치를 보여주는 반응을 생성하도록 설계되었습니다. 동적 주석 프로세스(dynamic annotation processes), 맞춤형 프롬프트 디자인, 실험을 통해 개방형 가중치 LLMs가 가치 민감한 문제에 어떻게 반응하는지 조사합니다. 두 가지 실험 설정(log-probabilities와 long-form responses)을 사용하여, 다양한 언어적 및 문화적 맥락에서 LLMs의 사회적, 문화적 변동성을 보여줍니다.

- **Performance Highlights**: 특히, 장형응답(long-form responses)을 포함하는 실험에서 특정 모델들이 다르게 거부 반응을 보이는 것을 확인했습니다. 이는 영어 또는 번역된 문장에서 가장 빈번하게 발생했습니다. 이민, LGBTQI 권리, 사회 복지에 관한 특정 주제와 출처가 모델의 답변에 더 뚜렷한 차이를 일으켰습니다. 연구 결과, CIVICS 데이터셋은 재현성과 투명성을 촉진하고, 글로벌 문화적 다양성과 가치 플루럴리즘을 존중하고 반영하는 AI 기술의 발전을 위한 도구로 사용될 것입니다.



### DeTox: Toxic Subspace Projection for Model Editing (https://arxiv.org/abs/2405.13967)
Comments:
          Preprint

- **What's New**: 이번 논문에서 저자들은 독성 축소(Detoxification)이라는 새로운 튜닝-프리(tuning-free) 정렬 알고리즘인 'DeTox'을 소개했습니다. DeTox는 대규모 언어 모델(LLMs)의 독성을 줄이는 데 효과적인 것으로 입증되었으며, 기존의 직접 선호 최적화(Direct Preference Optimization, DPO) 기법에 비해 더 샘플 효율적이고 견고하다고 주장합니다.

- **Technical Details**: DeTox는 요인 분석(FA, Factor Analysis)에 기반한 이론을 바탕으로 작동합니다. 모델 파라미터 공간에서 독성 부분 공간(subspace)을 식별하고, 감지된 부분 공간을 투영하여 제거하는 모델 편집 접근법을 사용합니다. 이를 위해 언어 모델에서 선호 데이터 임베딩을 추출하여 독성이 없는 정보를 제거합니다. 이론적으로 DeTox는 DPO의 단일 단계에서 노이즈가 제거된 버전으로 해석될 수 있습니다.

- **Performance Highlights**: DeTox는 기존의 DPO 알고리즘에 비해 더 적은 샘플로 동일한 수준의 성능을 발휘하며, 노이즈 데이터에 대한 더욱 높은 견고성을 보여줍니다. 또한, DeTox는 대규모 선호 데이터가 필요하지 않고 투명성과 제어 능력이 향상되었습니다.



### Vikhr: The Family of Open-Source Instruction-Tuned Large Language Models for Russian (https://arxiv.org/abs/2405.13929)
- **What's New**: Vikhr라는 새로운 오픈 소스 instruction-tuned LLM(Large Language Model)가 도입되었습니다. 이 모델은 특히 러시아어 텍스트 생성의 품질을 크게 개선하여, 기존의 러시아어 모델들이 직면했던 문제들을 해결합니다.

- **Technical Details**: Vikhr는 영어 중심 모델에서 LoRA adapters를 사용하는 대신, tokenizer vocabulary를 러시아어에 맞게 조정하고, 모든 가중치에 대해 지속적인 사전학습과 instruction tuning을 수행합니다. 이는 모델의 성능 향상 뿐만 아니라, 연산 및 문맥적 효율성도 크게 개선합니다.

- **Performance Highlights**: Vikhr는 여러 러시아어 벤치마크에서 탁월한 성능을 보여주었으며, 이는 지속적인 사전학습을 위한 instruction 데이터셋과 코퍼스를 확장한 결과입니다. 또한, Vikhr는 일부 독점적인 폐쇄형 모델들보다도 더 뛰어난 성능을 보여주며, 모델 가중치, instruction 세트, 코드가 공개적으로 제공됩니다.



### Why Not Transform Chat Large Language Models to Non-English? (https://arxiv.org/abs/2405.13923)
- **What's New**: 새로운 논문은 비영어 대형 언어 모델(LLMs)의 개발을 제한하는 비영어 데이터의 부족 문제를 다루고 있습니다. 기존 연구들은 기본 LLMs를 이용해 지식 증류(knowledge distillation, KD)를 수행하여 더 강력한 LLMs, 예를 들어 GPT-4로 생성된 데이터를 사용하는 접근법을 제시해왔습니다. 그러나 채팅 LLMs(chat LLMs)를 전환(transformation)하는 과정에서 발생하는 주요 문제들을 다루기 위해 새로운 프레임워크 TransLLM을 도입하였습니다.

- **Technical Details**: TransLLM 프레임워크는 두 가지 주요 문제를 해결합니다. 첫째 문제는 고급 능력(advanced abilities)을 효과적으로 전이하는 방법으로, 번역 chain-of-thought를 사용하여 영어와 비영어 간의 전이를 단계별로 수행합니다. 공개된 데이터를 활용하여 서브 태스크의 성능을 향상시켰습니다. 둘째 문제는 원래 지식을 잊어버리는 것을 방지하기 위해, 저희는 두 가지 구성 요소를 제안했습니다: LLM 파라미터를 유지하도록 하는 저순위 적응(low-rank adaptation) 및 채팅 LLM 자체가 생성한 데이터를 사용하여 원래 지식을 회복하는 회복 KD(recovery KD)입니다.

- **Performance Highlights**: 실험 결과, 우리의 방법은 LLaMA-2-chat-7B를 태국어로 변환하여 강력한 기준 모델들과 ChatGPT보다 우수한 성능을 보여주었습니다. 특히 단일 턴 데이터만으로 다중 턴 벤치마크 MT-bench에서 뛰어난 결과를 보였으며, 안전 데이터 없이도 ChatGPT와 GPT-4보다 더 많은 해로운 쿼리를 거부하는 기능을 가지고 있습니다.



### Just rephrase it! Uncertainty estimation in closed-source language models via multiple rephrased queries (https://arxiv.org/abs/2405.13907)
- **What's New**: 최신의 큰 언어 모델(large language models)은 공용 소프트웨어로 배포되거나, 폐쇄형 소스 서비스로 제공됩니다. 폐쇄형 LLM이 많은 대중들에게 널리 사용되고 있지만, 쿼리에 응답할 때 불확실성 추정(uncertainty estimate)을 제공하지 않는 경우가 많습니다. 이 연구는 폐쇄형 LLM의 불확실성을 여러 번의 재질문(rephrasings)을 통해 추정하는 방법을 탐구합니다.

- **Technical Details**: 기본 쿼리의 여러 재질문을 통해 모델에게 질문을 하고, 답변의 유사성을 불확실성의 추정치로 사용합니다. 이전 연구와는 다른 두 가지 핵심 차별점을 가지고 있습니다: i) 실제 사용이 쉽고 기억하기 쉬운 재질문 규칙을 제공하는 것, ii) 여러 번 재질문하는 것이 보정된 불확실성 추정치를 얻는 이론적 프레임워크를 제안하는 것입니다.

- **Performance Highlights**: 제안된 방법은 기본 방법(baseline)보다 불확실성 추정의 보정(calibration)에서 상당한 개선을 보였습니다. 또한, 최적의 테스트 보정을 위해 쿼리 전략이 어떻게 설계되어야 하는지에 대한 직관적인 통찰을 제공합니다.



### Semantic Density: Uncertainty Quantification in Semantic Space for Large Language Models (https://arxiv.org/abs/2405.13845)
Comments:
          16 pages, 2 figures

- **What's New**: 이 논문은 대형 언어 모델(LLMs)에서 발생할 수 있는 불확실성 문제를 다루기 위해 새로운 프레임워크를 제시합니다. 현재의 LLM들은 응답마다 불확실성 지표를 제공하지 못하기 때문에 신뢰성 평가에 어려움이 있습니다. 이 논문에서는 이러한 문제를 해결할 수 있는 방안으로 Semantic Density라는 개념을 도입했습니다. 이는 각 응답의 불확실성을 의미 공간에서의 확률 분포 관점에서 추출하는 방법입니다.

- **Technical Details**: Semantic Density는 불확실성을 평가하는 체계로, 특정 작업 유형에 국한되지 않고 새로운 모델과 작업에 'off-the-shelf'(즉시 사용 가능한) 방식으로 적용할 수 있습니다. 이 프레임워크는 의미론적 정보를 고려하여 불확실성 정보를 추출하며, 기존 방법들이 가지는 분류 작업 제한, 추가 훈련 및 데이터 요구, 어휘 수준의 정보만 고려하는 문제점들을 해결합니다.

- **Performance Highlights**: Llama 3와 Mixtral-8x22B 모델을 포함한 7개의 최신 LLM에서 4개의 자유 형식 질문 응답 벤치마크를 실험한 결과, Semantic Density는 이전 접근법보다도 뛰어난 성능과 강건성을 보여주었습니다.



### Babysit A Language Model From Scratch: Interactive Language Learning by Trials and Demonstrations (https://arxiv.org/abs/2405.13828)
- **What's New**: 이번 연구에서는 언어 모델(Language Models)이 상호작용을 통해 얼마나 더 효율적으로 학습할 수 있는지를 조사합니다. 연구진은 인간의 언어 학습이 사회적 상호작용에 의해 크게 영향을 받는다는 점에 주목하여, 같은 방식을 언어 모델에 적용하는 실험을 진행했습니다.

- **Technical Details**: 본 연구는 'trial-and-demonstration (TnD)' 학습 프레임워크를 도입하여, 학생 모델의 시도(student trials), 교사 데모(teacher demonstrations), 그리고 다양한 개발 단계에서의 언어 능력에 따라 보상을 제공하는 세 가지 구성 요소로 이루어져 있습니다. 이를 통해 교사의 어휘 선택이 학생 모델의 어휘 학습 효율성에 미치는 영향과, 연습량(frequency)과 학습 곡선 사이의 강한 상관 관계를 확인했습니다.

- **Performance Highlights**: TnD 접근 방식은 동일하거나 적은 파라미터(parameter) 수를 가진 학생 모델의 단어 습득을 가속화하는 데 효과적임을 확인했습니다. 또한, 교사 데모와 학생 시도 모두의 중요성을 강조했습니다. 특히, 단어의 반복적인 연습(practice-makes-perfect)의 효과가 뚜렷하게 나타났습니다.



### Towards Comprehensive and Efficient Post Safety Alignment of Large Language Models via Safety Patching (https://arxiv.org/abs/2405.13820)
Comments:
          24 pages, 8 figures and 12 tables

- **What's New**: 대형 언어 모델(LLMs)의 안전 정렬(safety alignment)에 대한 새로운 접근법으로, 기존의 안전 정렬된 LLM이 가지고 있는 취약점과 불균형한 안전 메커니즘을 개선한 'SafePatching'이라는 새로운 프레임워크가 제안되었습니다. 이 프레임워크는 유해한 데이터에 두 개의 독특한 안전 패치를 개발하여 안전성을 강화하고 과도한 안전 문제를 완화하는 방식으로 설계되었습니다.

- **Technical Details**: 'SafePatching'은 포괄적이고 효율적인 사후 안전 정렬(PSA)을 위한 새로운 프레임워크입니다. 이 프레임워크는 두 개의 안전 패치를 발전시켜 안전성을 높이고 과도 안전 문제를 완화하며, 이를 대상 LLM 백본에 무리 없이 통합하는 방법을 제안합니다. 이는 PSA 후에도 모델의 일반적인 유틸리티를 유지하여 안전성과 유용성 사이의 균형을 최적화합니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면 'SafePatching'은 기존의 기준 방법들보다 더 포괄적이고 효율적인 PSA 결과를 보여줍니다. 이 프레임워크는 백본의 유틸리티도 향상시켜, 현재 정렬된 LLM에서 유용성과 해로움 사이의 균형을 더욱 최적화합니다. 또한, 지속적인 PSA 시나리오에서도 우수한 성능을 입증했습니다.



### Large Language Models are Good Spontaneous Multilingual Learners: Is the Multilingual Annotated Data Necessary? (https://arxiv.org/abs/2405.13816)
- **What's New**: 최근 대규모 언어 모델(LLMs)이 인상적인 언어 능력을 보여주고 있지만, 대부분의 기존 LLMs는 영어 중심으로 다른 언어들에 비해 매우 불안정하고 불균형한 성능을 보입니다. 이번 연구에서는 번역 데이터를 활용한 멀티링구얼 정렬(multilingual alignment) 패러다임을 탐구하고, LLMs의 자발적인 다국어 향상을 포괄적으로 조사했습니다. 특히, 질문 번역 데이터에만 기반하여 명령-튜닝(instruction-tuning)된 LLMs가 주석된 답변이 없이도 많은 언어에서 눈에 띄는 다국어 성능 향상을 얻을 수 있음을 발견했습니다.

- **Technical Details**: 번역 데이터를 활용하여 LLMs를 명령-튜닝하는 방법을 연구했습니다. 다양한 설정 및 메커니즘 해석 방법(mechanistic interpretability methods)을 활용하여 LLMs의 다국어 시나리오에서의 성능을 포괄적으로 분석했습니다.

- **Performance Highlights**: 명령-튜닝 데이터에 포함되지 않은 다양한 언어에서도 LLMs가 유의미한 다국어 성능 향상을 보였습니다. 이는 주석된 답변이 없어도 질문 번역 데이터만으로도 상당한 성과를 낼 수 있음을 나타냅니다.



### Slaves to the Law of Large Numbers: An Asymptotic Equipartition Property for Perplexity in Generative Language Models (https://arxiv.org/abs/2405.13798)
- **What's New**: 이 논문에서는 언어 모델(Language Model)에 의해 생성된 큰 텍스트의 당혹도(Perplexity)에 대한 새로운 점근적 등분단성(Property)을 제안하고, 이에 대한 이론적 근거를 제시합니다. 당혹도는 일반적으로 언어 모델의 성능을 측정하는 데 사용되는 지표로서, 역확률함수(Inverse Likelihood Function)로 정의됩니다.

- **Technical Details**: 주요 결과로서, 언어 모델이 생성하는 대용량의 텍스트의 로그 당혹도가 토큰 배포의 평균 엔트로피(Entropy)로 점근적으로 수렴해야 한다고 주장합니다. 이는 언어 모델이 '전형적 집합(Typical Set)'에서만 출력을 생성하도록 제한된다는 것을 의미합니다. 저자들은 이러한 전형적 집합이 모든 가능한 문법적으로 올바른 출력의 아주 작은 부분집합이라고 주장합니다. 또한, 이론적 주장을 뒷받침하기 위해 오픈소스 언어 모델의 예비 실험 결과도 제시했습니다.

- **Performance Highlights**: 이 연구는 'AI 탐지' 도구의 이해와 개선을 위한 실제 응용 가능성을 가지고 있으며, 생성 모델의 독창성, 예측 가능성 및 창의성에 대한 이론적 함의를 가지고 있습니다.



### xRAG: Extreme Context Compression for Retrieval-augmented Generation with One Token (https://arxiv.org/abs/2405.13792)
- **What's New**: 이번 논문에서는 새로운 컨텍스트 압축 방법인 xRAG을 소개합니다. xRAG는 Retrieval-Augmented Generation (RAG)를 위한 최적의 압축 기법으로서, 전통적인 문서 임베딩을 단순 검색용이 아닌 언어 모델 표현 공간에 통합할 수 있는 특성으로 재해석합니다. 이를 통해 텍스트 형태의 문서가 더 이상 필요하지 않게 되어 엄청난 압축률을 달성할 수 있습니다.

- **Technical Details**: xRAG는 문서 임베딩을 추출하여 언어 모델의 표현 공간에 융합시키는 '모달리티 융합 방법론(modality fusion methodology)'을 사용합니다. 여기서 유일하게 학습 가능한 부분은 '모달리티 브릿지(modality bridge)'이고, 나머지 검색기와 언어 모델은 고정된 상태로 유지됩니다. 이는 온라인에서 만들어진 문서 임베딩을 재사용할 수 있게 하며, 검색 보강의 Plug-and-Play 특성을 유지합니다.

- **Performance Highlights**: xRAG는 여섯 가지 지식 집중형 과제에서 평균 10% 이상의 성능 향상을 보여주며, 다양한 언어 모델 백본에 적응 가능합니다. 7B 밀집 모델에서 8x7B Mixture of Experts 구성에 이르기까지 사용되었습니다. 또한, xRAG는 이전의 컨텍스트 압축 방법들을 훨씬 능가하면서 몇몇 데이터셋에서는 비압축 모델과 동일한 성능을 보여주었고, 총 FLOPs를 3.53배 줄였습니다.



### Do Language Models Enjoy Their Own Stories? Prompting Large Language Models for Automatic Story Evaluation (https://arxiv.org/abs/2405.13769)
Comments:
          TACL, pre-MIT Press publication version

- **What's New**: 이 연구에서는 대형 언어 모델(Large Language Models, LLM)이 자동 이야기를 평가(Automatic Story Evaluation, ASE)하는 데 인간 평가자를 대체할 수 있는지에 대해 탐구합니다. 연구는 LLM의 평가, 다른 자동 평가 방법, 인간 주석 간의 상관관계를 분석하고, 프롬프트(prompting)의 영향을 조사하며, LLM의 행동 설명 가능성(explainability)을 검토합니다.

- **Technical Details**: 높은 수준의 인간 능력, 즉 창의성(creativity), 추론(reasoning), 깊은 이해(deep understanding)을 필요로 하는 ASE와 자동 이야기 생성(Automatic Story Generation, ASG) 과제는 도전적인 과제입니다. 그러나 LLM이 많은 NLP 작업에서 최첨단 성과를 달성하고 있어, 이를 기반으로 한 시스템 평가를 연구했습니다. 이에 따라 LLM 평가는 시스템 수준의 평가에서 현재의 자동 평가 방법을 능가한다는 결과를 발견했습니다.

- **Performance Highlights**: 연구 결과, LLM은 현재 사용되는 자동 평가 방법보다는 효과적이지만, 여전히 자신이 도출한 답변에 대해 만족스러운 설명을 제공하는 데 어려움을 겪고 있습니다. 이는 ASE에서 시스템 수준의 평가에서는 우수하지만, 세부 분석 및 설명력 면에서는 개선이 필요함을 시사합니다.



### Grounding Toxicity in Real-World Events across Languages (https://arxiv.org/abs/2405.13754)
Comments:
          Paper accepted for at The 29th International Conference on Natural Language & Information Systems (NLDB 2024)

- **What's New**: 이 연구는 현실 세계의 사건이 온라인 상의 독성(독성) 행동의 기원과 확산에 어떻게 영향을 미치는지를 조사합니다. 연구팀은 2020년부터 2023년까지 발생한 15개의 주요 사회 및 정치적 세계 사건을 대상으로 데이터를 분석했습니다.

- **Technical Details**: Reddit에서 6개의 다른 언어(네덜란드어, 영어, 독일어, 아랍어, 터키어, 스페인어)로 작성된 3만 1천 개의 게시물에서 450만 개의 댓글을 수집했습니다. 연구는 독성, 부정적인 감정(sentiment), 감정 표현(emotion expressions)이 사건과 언어 커뮤니티에 따라 어떻게 다르게 나타나는지를 조사했습니다.

- **Performance Highlights**: 결과적으로 독성이 사건과 언어 커뮤니티에 따라 상당히 변동한다는 점을 발견했습니다. 이는 독성이 복잡한 현상이며, 여러 요소들이 상호작용한다고 결론지을 수 있습니다. 연구팀은 추가 연구를 위해 데이터와 코드를 공개할 예정입니다.



### CrossCheckGPT: Universal Hallucination Ranking for Multimodal Foundation Models (https://arxiv.org/abs/2405.13684)
Comments:
          21 pages. Preprint

- **What's New**: 다중모달 (multimodal) 파운데이션 모델은 입력과 모순되거나 사실에 기반하지 않는 출력을 생성하는 경향이 있습니다. CrossCheckGPT는 레퍼런스 없는 보편적인 헛소리 순위 (hallucination ranking)를 제안하여 이러한 문제를 해결하려고 합니다. 이 방법은 서로 독립적인 시스템이 동일한 헛소리 내용을 생성할 가능성이 낮다는 아이디어에 기반하고 있습니다. 이는 모든 모델이나 작업에 적용될 수 있습니다.

- **Technical Details**: CrossCheckGPT는 두 가지 정보 일관성 측정 방법, CrossCheck-explicit와 CrossCheck-implicit를 탐구합니다. 이 모델은 텍스트, 이미지, 오디오-비주얼 등 다양한 모달리티에 걸친 헛소리 순위 평가에 적용될 수 있습니다. 제안된 모델은 출력 간의 정보 일관성을 적절한 거리 메트릭 (distance metric)을 통해 측정하여 헛소리를 평가합니다.

- **Performance Highlights**: CrossCheckGPT는 MHaluBench와 AVHalluBench에서 각각 98%와 89%의 상관관계를 인간 평가와 달성함으로써 그 효과성을 입증했습니다. 이 연구는 또한 첫 오디오-비주얼 헛소리 벤치마크, AVHalluBench를 제안합니다.



### Knowledge Graph Reasoning with Self-supervised Reinforcement Learning (https://arxiv.org/abs/2405.13640)
Comments:
          17 pages, 11 figures

- **What's New**: 이 논문에서는 강화학습(Reinforcement Learning, RL)을 사용하여 불완전한 지식 그래프(KG)의 추론 경로를 발견하는 방법을 제안합니다. 대규모 액션 공간 문제를 극복하기 위해, RL 훈련 단계에 앞서 정책 네트워크를 준비시키기 위한 자기 지도 사전 훈련 방법(self-supervised pre-training)을 도입합니다.

- **Technical Details**: 일반적인 자기지도 강화학습(self-supervised RL, SSRL)에서 발생하는 분포 불일치 문제를 해결하기 위해, 감독 학습(SL) 단계에서 에이전트는 정책 네트워크에 기반하여 액션을 선택하고 생성된 레이블로부터 학습합니다. 이 프레임워크를 통해 SL 목표의 정보 밀도가 증가하며, 에이전트가 초기에 보상을 받은 경로에 고착되지 않도록 방지합니다. SSRL 방법은 SL의 넓은 커버리지 성과를 RL과 결합하여 RL 성능을 향상시킵니다. 두 가지 RL 아키텍처(MINERVA와 MultiHopKG)를 기본 모델로 채택하여, SSRL 모델이 네 가지 대규모 KG 데이터셋 모두에서 일관되게 우수한 성능을 보임을 실험적으로 증명합니다.

- **Performance Highlights**: SSRL 모델은 네 가지 대규모 벤치마크 KG 데이터세트에서 모든 Hits@k와 평균 역순위(MRR) 지표에서 최신 최고의 결과와 동등하거나 더 나은 성능을 달성합니다. 또한, 이 SSRL 방법은 KGR(Knowledge Graph Reasoning) 작업을 위한 어떠한 RL 아키텍처에도 플러그인으로 사용할 수 있습니다.



### Automated Evaluation of Retrieval-Augmented Language Models with Task-Specific Exam Generation (https://arxiv.org/abs/2405.13622)
Comments:
          Proceedings of the 41st International Conference on Machine Learning (ICML), 29 pages, 12 figures

- **What's New**: 본 연구는 Retrieval-Augmented Large Language Models (RAG)의 작업 특화 정확도를 측정하기 위한 새로운 방법을 제안합니다. 이 방법은 해당 작업과 연관된 문서들로 구성된 다중 선택 질문으로 자동 생성된 시험을 통해 RAG의 성능을 평가합니다. 제안된 방법은 자동화된, 비용 효율적이며, 해석 가능하고 견고한 RAG 시스템 구성 요소 선택 전략입니다.

- **Technical Details**: 제안된 방법은 Item Response Theory (IRT)를 활용하여 시험의 품질 및 작업 특화 정확도에 대한 정보성을 추정합니다. IRT는 모델의 능력에 대한 충분한 정보를 제공하지 않는 시험 문제를 제거함으로써 시험을 점진적으로 개선하는 자연스러운 방법을 제공합니다. 이 접근법은 Arxiv 초록, StackExchange 질문, AWS DevOps 문제 해결 가이드 및 SEC 서류 기반의 네 가지 새로운 개방형 질문-응답 작업에서 시연되었습니다.

- **Performance Highlights**: 실험을 통해 크기, 검색 메커니즘, 프롬프트 및 파인튜닝과 같은 RAG 성능에 영향을 미치는 일반적인 요소에 대한 통찰을 얻었습니다. 특히, 적절한 검색 알고리즘을 선택하는 것이 단순히 더 큰 언어 모델을 사용하는 것보다 더 큰 성능 향상을 가져올 수 있다는 점을 발견했습니다.



### ConTrans: Weak-to-Strong Alignment Engineering via Concept Transplantation (https://arxiv.org/abs/2405.13578)
- **What's New**: 대형 언어 모델(LLM, Large Language Model)의 행동이 인간의 목표, 가치 및 의도와 일치하도록 하는 것이 안전성에 매우 중요하지만, 이는 계산 비용이 많이 듭니다. 이를 해결하기 위해 ConTrans라는 새로운 프레임워크가 제안되었습니다. 이 프레임워크는 개념 이식 방식(Concept Transplantation)을 통해 약하게 정렬된 모델에서 강하게 정렬된 모델로의 전이를 가능하게 합니다.

- **Technical Details**: ConTrans는 표현 엔지니어링 관점에서 약하게 정렬된 LLM(보통은 정렬이 잘 된 LLM)로부터 개념 벡터를 정제합니다. 그런 다음 정제된 개념 벡터를 affine 변환을 통해 목표 LLM(보통은 강력하지만 정렬되지 않은 기본 LLM)에 맞게 재구성합니다. 마지막으로 ConTrans는 재구성된 개념 벡터를 목표 LLM의 잔여 스트림에 이식합니다. 이를 통해 서로 다른 LLM 계열 내에서 개념 이식이 성공적으로 이루어질 수 있습니다.

- **Performance Highlights**: 실험 결과, ConTrans는 7B 모델에서 13B 및 70B 모델로 다양한 정렬 개념을 성공적으로 이식했으며, 이는 여러 LLM과 LLM 계열에 걸쳐 적용되었습니다. 특히, ConTrans는 진실성 측면에서 지시 조정된 모델을 능가하는 성과를 보였습니다. 이러한 결과는 inter-LLM-family 및 intra-LLM-family 개념 이식의 효용성을 입증합니다.



### FlashRAG: A Modular Toolkit for Efficient Retrieval-Augmented Generation Research (https://arxiv.org/abs/2405.13576)
Comments:
          8 pages

- **What's New**: 이번 연구에서는 FlashRAG라는 효율적이고 모듈식(open-source toolkit)을 제안합니다. 이 도구는 Retrieval Augmented Generation(RAG) 기법의 기존 방법을 재현하고 새로운 RAG 알고리즘을 개발할 수 있게 도와줍니다. FlashRAG는 연구자들이 다양한 RAG 접근법을 일관된 환경에서 비교하고 평가하는 데 드는 시간과 노력을 줄여줄 것입니다.

- **Technical Details**: FlashRAG는 12개의 고급 RAG 메서드를 구현하며, 32개의 벤치마크 데이터셋을 수집하여 제공하고 있습니다. 이 툴킷은 사용자 맞춤형 모듈식 프레임워크(customizable modular framework)와 풍부한 사전 구현된 RAG 작업들, 종합적인 데이터셋, 효율적인 보조 전처리 스크립트( auxiliary pre-processing scripts) 및 광범위하고 표준화된 평가 지표(extensive and standard evaluation metrics)를 특징으로 합니다.

- **Performance Highlights**: FlashRAG는 연구자들이 RAG 시스템을 더욱 효율적으로 연구하고 개선할 수 있도록 도와주는 다양한 기능을 제공합니다. 특히, 표준화된 평가 지표를 통해 일관성 있는 성능 비교가 가능하게 하여, 새로운 알고리즘과 메서드를 쉽게 평가할 수 있게 합니다.



### Knowledge-Driven Cross-Document Relation Extraction (https://arxiv.org/abs/2405.13546)
Comments:
          Accepted in ACL 2024 Findings

- **What's New**: 새로운 관계 추출(Relation Extraction, RE) 방법인 KXDocRE를 소개합니다. 이 방법은 문서 내의 텍스트와 함께 도메인 지식을 통합하여 Cross-Document RE(CrossDocRE)를 수행합니다. 기존의 CrossDocRE 방법이 도메인 지식을 고려하지 않는 점을 개선하고 있습니다.

- **Technical Details**: KXDocRE는 문서들의 텍스트와 엔터티(entity) 관련 도메인 지식을 함께 삽입(embedding)하여 작동합니다. 이 프레임워크는 세 가지 주요 이점을 제공합니다. 첫째, 엔터티에 대한 도메인 지식을 문서 텍스트와 함께 활용합니다. 둘째, 예측된 엔터티 간의 관계에 대한 설명 텍스트를 제공하여 해석 가능성을 제시합니다. 셋째, 기존 방법들에 비해 성능이 향상됩니다.

- **Performance Highlights**: KXDocRE는 기존의 관계 추출 방법에 비해 성능이 더 우수하며, 특히 해석 가능성 측면에서 큰 향상을 보여줍니다. 이는 도메인 지식을 통합하여 문서 간의 새로운 관계를 식별하는 데 효과적입니다.



### Annotation-Efficient Preference Optimization for Language Model Alignmen (https://arxiv.org/abs/2405.13541)
- **What's New**: 이번 연구에서는 인간의 선호에 맞추기 위해 대형 언어 모델을 미세 조정하는 과정에서 중요한 '선호 최적화(Preference Optimization)' 방법론에 대한 새로운 접근법을 제안합니다. 특히, Annotation 효율성을 높여 제한된 예산 내에서 더욱 효과적인 선호 데이터를 구축하는 'Annotation-Efficient Preference Optimization (AEPO)' 기법을 소개합니다.

- **Technical Details**: AEPO는 기존의 선호 데이터 주석이 모든 응답 텍스트에 대해 이루어지는 것과 달리, 응답 텍스트 중에서 품질과 다양성이 높은 부분적인 응답을 선택하여 주석을 다는 방식으로 진행됩니다. 이 통해 주석 예산을 고품질과 다양성이 높은 응답에 집중시킵니다. 직접 선호 최적화(Direct Preference Optimization, DPO)에서 AEPO를 적용한 결과, 동일한 주석 예산을 사용한 모델에 비해 성능이 향상됨을 보였습니다.

- **Performance Highlights**: AEPO를 통해 생성된 모델은 같은 주석 예산을 사용한 표준 DPO 방법으로 훈련된 모델보다 우수한 성능을 보여줍니다. 실험 결과로 이 접근법의 효율성을 입증했습니다. 또한, 연구의 재현성을 위해 코드는 공개되어 있습니다.



### The correlation between nativelike selection and prototypicality: a multilingual onomasiological case study using semantic embedding (https://arxiv.org/abs/2405.13529)
- **What's New**: 이 연구는 모국어 화자들이 특정 개념을 표현할 때 사용하는 표현을 선택하는 현상인 nativelike selection (NLS)에 대해 새로운 시각을 제시합니다. NLS와 원형성(prototypicality) 간의 상관관계를 탐구함으로써, NLS 현상이 단순히 임의적인 콜로케이션(collocations) 뿐만 아니라 더 깊은 의미적 동기와 연관될 수 있다는 가능성을 검토합니다.

- **Technical Details**: 이 연구는 주제 모델링(topic modeling) 기법을 사용하여 잠재적인 NLS를 자동으로 탐지하고, 프레임 의미론(frame semantics)을 통해 수동으로 확인하는 혁신적인 방법론을 도입하였습니다. 추가로, 클러스터 분석(cluster analysis)과 행동 프로파일 분석(behavioral profile analysis)을 통해 중국어 동사 'shang'의 원형성을 파악하고, 이를 통해 NLS와 원형성의 상관관계를 지원하는 증거를 제공합니다.

- **Performance Highlights**: 이 연구는 중국어 동사 'shang'를 대상으로 하여 언어 특정의 원형(prototype)을 밝혀내면서, NLS 현상이 단순 임의적 선택이 아닌 의미적 동기와 관련이 있음을 보여줍니다.



### LIRE: listwise reward enhancement for preference alignmen (https://arxiv.org/abs/2405.13516)
Comments:
          Accepted by ACL 2024 Findings

- **What's New**: 최근, 대형 언어 모델(Large Language Models, LLMs)이 생성하는 콘텐츠를 인간 가치에 맞추기 위한 연구가 크게 진전되었습니다. 이를 위해 인간의 피드백을 통한 강화 학습(Reinforcement Learning from Human Feedback, RLHF)이 효과적인 방법으로 널리 채택되고 있습니다. 그러나 RLHF의 구현은 복잡하며, 하이퍼파라미터에 민감하여 안정적인 성능을 달성하는 데 어려움이 있습니다. 이러한 문제를 해결하기 위해 새로운 접근법인 LIRE(Listwise Reward Enhancement for Preference Alignment)를 제안합니다.

- **Technical Details**: LIRE는 다중 응답의 오프라인 보상(offline rewards)을 간소화된 listwise 프레임워크에 통합하여, 훈련 중 온라인 샘플링의 필요성을 제거하는 그래디언트 기반 보상 최적화 접근법입니다. LIRE는 구현이 간단하고, 최소한의 파라미터 조정을 요구합니다. 또한 기존의 pairwise 패러다임과 잘 맞으며, 다중 응답 시나리오에도 자연스럽게 확장됩니다. 추가로, 훈련 중 보상을 반복적으로 향상시키기 위한 self-enhancement 알고리즘을 도입하였습니다.

- **Performance Highlights**: 우리의 실험 결과, LIRE는 대화 및 요약 과제에 대해 여러 벤치마크에서 기존 방법들을 일관되게 능가하는 성능을 보였습니다. 특히, 프록시 보상 모델과 인간 주석자를 사용하여 평가한 결과, 분포 외 데이터(out-of-distribution data)에 대한 우수한 전이 가능성(transferability)을 보였습니다.



### Distilling Instruction-following Abilities of Large Language Models with Task-aware Curriculum Planning (https://arxiv.org/abs/2405.13448)
- **What's New**: 이 논문에서는 기존 대형 언어 모델(LLM)을 오픈 도메인 지시문과 인간 선호 응답에 맞추는 과정인 'instruction tuning'을 더욱 효율적으로 만드는 새로운 방법을 소개합니다. 특히, 'Task-Aware Curriculum Planning for Instruction Refinement(TAPIR)'이라는 다중 회차(distillation framework) 프레임워크를 통해 과제 분포의 균형과 동적 난이도 조정을 통해 학생 LLM의 성능을 향상시키는 방법을 제안합니다.

- **Technical Details**: TAPIR은 강력한 LLM(예: oracle LLM)을 사용하여 학생 LLM(예: student LLM)이 따르기 어려운 지시문을 선택하고, 과제 분포의 균형을 맞춘 지시문(distill instructions)을 선택합니다. 커리큘럼 플래닝(curriculum planning)을 통해 난이도를 점진적으로 증가시켜 학생 LLM의 능력을 점진적으로 향상시킵니다.

- **Performance Highlights**: TAPIR의 성능을 두 가지 주요 벤치마크(AlpacaEval 2.0 및 MT-Bench)에서 엄격하게 평가한 결과, 제안된 방법으로 훈련된 학생 LLM이 더 적은 훈련 데이터로도 더 큰 instruction-tuned 모델과 강력한 distillation 기법들을 능가하는 성능을 보여주었습니다. 특히 논리적 추론(logical reasoning)과 코드 생성(code generation)과 같은 복잡한 작업에서 두드러진 향상을 나타냈습니다.



### Disperse-Then-Merge: Pushing the Limits of Instruction Tuning via Alignment Tax Reduction (https://arxiv.org/abs/2405.13432)
Comments:
          Accepted to the findings of ACL2024

- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)의 성능 하락 문제를 해결하기 위한 새로운 접근법을 제안하였습니다. 지도 학습(SFT) 과정에서 발생하는 데이터 편향이 성능 저하의 원인일 수 있다는 가설을 세우고, 이를 해결하기 위해 분산-병합(disperse-then-merge) 프레임워크를 도입하였습니다.

- **Technical Details**: 제안된 방법은 다음과 같습니다. 먼저, 지침 준수 데이터(instruction-following data)를 여러 부분으로 분산하고, 각 데이터 부분을 이용하여 여러 가지 서브 모델(sub-models)을 훈련합니다. 이후 모델 병합 기법(model merging techniques)을 사용하여 여러 모델을 하나로 병합합니다. 해당 프레임워크는 데이터 큐레이션(data curation) 및 훈련 규제(training regularization)와 같은 복잡한 방법들을 능가하는 성능을 보였습니다.

- **Performance Highlights**: 분산-병합 프레임워크는 표준 지식 및 추론 벤치마크(knowledge and reasoning benchmarks)에서 뛰어난 성능을 보였으며, 데이터 편향 문제를 효과적으로 해결하는 방법으로 입증되었습니다.



### 360Zhinao Technical Repor (https://arxiv.org/abs/2405.13386)
Comments:
          360Zhinao technical report. Github: this https URL

- **What's New**: 360Zhinao 모델은 7B 파라미터 크기를 가지고 있으며, 4K, 32K, 360K의 컨텍스트 길이를 커버합니다. 빠른 사전 학습 개발을 위해, 우리는 실험 평가와 비교를 최소 모델 크기로 수행할 수 있는 안정적이고 민감한 절단 환경을 구축했습니다.

- **Technical Details**: 우리는 이러한 절단 환경을 통해 데이터 클리닝과 구성 전략을 개선하여, 3.4T 토큰을 사용한 $	exttt{360Zhinao-7B-Base}$의 사전 학습을 완성했습니다. 데이터 정렬 (alignment) 과정에서는 양과 질의 균형을 유지하기 위해 필터링과 재포맷팅을 강조했습니다. 이를 통해 360Zhinao-7B의 컨텍스트 윈도우를 쉽게 32K 및 360K로 확장할 수 있었습니다. 추가적으로 SFT 및 특정 작업에 신뢰성 있게 적용될 수 있는 RMs 및 RLHF도 훈련되었습니다.

- **Performance Highlights**: 이러한 기여 덕분에 360Zhinao-7B는 유사한 크기의 모델들 사이에서 경쟁력 있는 성능을 보입니다.



### You don't understand me!: Comparing ASR results for L1 and L2 speakers of Swedish (https://arxiv.org/abs/2405.13379)
- **What's New**: 이 연구는 Automatic Speech Recognition (ASR) 시스템들이 일반적으로 성능이 좋아지고 있음에도 불구하고, 배경 소음, 다수의 화자 간 대화, 비전형적 화자(예: 어린이, 비원어민 또는 발음 장애가 있는 사람)의 경우 성능이 현저히 감소하는 문제를 다룹니다. 특히, 본 연구는 스웨덴어 원어민과 비원어민 간, 읽기 및 자발적 발화 간의 인식 성능 격차에 초점을 맞추었습니다.

- **Technical Details**: 연구진은 다양한 ASR 서비스가 제공하는 인식 결과를 Word Error Rate(WER)를 사용하여 비교하였으며, 관찰된 전사 오류를 생성할 수 있는 언어적 요인을 분석하였습니다. 이러한 접근 방식을 통해 ASR 시스템의 한계와 개선 가능성을 밝히고자 하였습니다.

- **Performance Highlights**: 연구 결과, ASR 시스템의 전반적인 성능이 향상되고 있음에도 불구하고, 비원어민과 자발적 발화에 대한 성능 저하가 두드러졌습니다. 이는 교육 소프트웨어와 같은 ASR 의존 애플리케이션에서 중요한 문제로 부각되고 있습니다.



### AdpQ: A Zero-shot Calibration Free Adaptive Post Training Quantization Method for LLMs (https://arxiv.org/abs/2405.13358)
- **What's New**: 최근 발표된 AdpQ는 calibration 데이터 없이도 state-of-the-art 성능을 발휘할 수 있는 zero-shot adaptive PTQ (Post-training Quantization) 방법입니다. 이 새로운 접근법은 낮은 정밀도의 quantization (예: 3-bit)을 달성하면서도 높은 정확도를 유지합니다.

- **Technical Details**: Adaptive LASSO 회귀 모델에 영감을 받아 제안된 AdpQ는 outlier 활성화 문제를 해결하기 위해 adaptive soft-thresholding 방법을 사용합니다. 이 방법은 quantized weights의 분포가 원래 훈련된 가중치를 가깝게 따르도록 하여 calibration 데이터의 필요성을 완전히 제거합니다. 이로 인해 SpQR나 AWQ 같은 기존 방법보다 privacy도 보장할 수 있습니다. 또한, Kullback-Leibler divergence를 최소화하여 원래 모델의 Shannon 정보량을 최대한 유지하게 됩니다.

- **Performance Highlights**: 여러 LLM 벤치마크에서 기존 방법들과 동일한 정확도를 달성하면서도 quantization 시간을 최소 10배 이상 단축할 수 있었습니다. 이를 통해 정확성과 정보 손실을 최소화하면서도 효과적인 배포를 보장합니다.



### Efficacy of ByteT5 in Multilingual Translation of Biblical Texts for Underrepresented Languages (https://arxiv.org/abs/2405.13350)
Comments:
          LXAI Workshop at the 2024 Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL 2024)

- **What's New**: 이 연구는 ByteT5 기반의 다국어 번역 모델 개발 및 평가를 다룹니다. 이 모델은 소외된 언어들로 성경을 번역하는 데 중점을 두고 있습니다. 존스 홉킨스 대학 성경 데이터셋을 활용하여, 문자 기반 및 형태적으로 복잡한 언어들의 미세한 뉘앙스를 포착하도록 훈련되었습니다.

- **Technical Details**: 모델은 번역 성능을 BLEU 스코어로 측정하며, 샘플 번역으로 성능을 보완합니다. 성경 특유의 어휘와 구조를 효과적으로 다루어 언어적 장벽을 해소하는 데 기여합니다. 또한, 모델의 한계점들을 논의하고, 성경을 더 많은 언어로 접근할 수 있도록 향후 개선 경로를 제안합니다.

- **Performance Highlights**: 모델은 BLEU 스코어와 샘플 번역을 통해 뛰어난 성능을 보이며, 성경 번역에서의 접근성을 크게 향상시킵니다. 특히, 특유의 성경 어휘와 구조를 잘 처리하여 의미 전달에 능숙함을 입증했습니다.



### High Performance P300 Spellers Using GPT2 Word Prediction With Cross-Subject Training (https://arxiv.org/abs/2405.13329)
- **What's New**: 이 논문은 다중 피험자(classifiers)를 효율적으로 훈련할 때 흔히 발생하는 속도 제한 문제를 해결하기 위해 혁신적인 '단일 피험자(Across-Subject)' 분류기를 도입했습니다. 이를 위해 두 번째 세대 생성 사전 훈련 변환(Generative Pre-Trained Transformer, GPT2)과 다익스트라 알고리즘(Dijkstra's algorithm)을 결합하여 자극을 최적화하고 타이핑 이력을 기반으로 단어 완성 옵션을 제안합니다.

- **Technical Details**: 이 시스템은 EEG 데이터에 반응하는 P300 스펠러 뇌-컴퓨터 인터페이스(P300 speller brain-computer interface)를 사용합니다. 또한, 새로운 다중 층(smoothing) 기법을 사용하여 사전에 없는 단어(out-of-vocabulary, OOV words)를 처리합니다. 이것은 다양한 피험자의 EEG 데이터를 무작위로 샘플링하는 시뮬레이션을 통해 평가되었습니다.

- **Performance Highlights**: 본 연구의 최적화를 통해 희귀 단어와 사전에 없는 단어가 포함된 문장을 입력할 때 문자 수준에서 타이핑 속도가 약 10% 향상되었으며, 여러 단어 예측에서는 최대 40% 향상되었습니다. 실험 결과, 표준 행/열 강조 기법에 레이어드 단어 예측을 추가하는 것이 최적에 가까운 성능을 제공하는 것으로 나타났습니다. 또한, '단일 피험자'와 '다중 피험자' 훈련 기술 모두에서 속도 향상이 일관되게 나타났습니다.



### Mosaic IT: Enhancing Instruction Tuning with Data Mosaics (https://arxiv.org/abs/2405.13326)
- **What's New**: 현재의 인스트럭션 튜닝(instruction tuning)은 주로 교사 모델(teacher model)이나 인간의 개입에 의존하여 지침과 응답을 생성하고 다듬는 방식입니다. 이는 비용이 많이 들고 지속 가능하지 않으며 다양성이 부족할 수 있습니다. 이번 논문에서는 Mosaic Instruction Tuning (Mosaic-IT)이라는 새로운 접근 방식을 소개합니다. 이 방법은 인간이나 모델 없이도 기존 인스트럭션 튜닝 데이터를 활용하여 다양하고 풍부한 증가 데이터를 효율적으로 생성할 수 있습니다.

- **Technical Details**: Mosaic-IT는 여러 가지 인스트럭션 데이터를 무작위로 연결(concatenate)하고 모델을 훈련시켜, 미리 정의된 상위(level) 메타-인스트럭션(meta-instructions)에 따라 해당 응답을 생성하도록 합니다. 이를 통해 모델의 다단계 지침 추종 능력과 형식 추종 능력을 강화합니다.

- **Performance Highlights**: 광범위한 평가 결과, Mosaic-IT는 다양한 벤치마크(benchmarks)에서 일관된 성능 향상을 달성하며, 기존 인스트럭션 튜닝 대비 훈련 비용이 80% 절감됨을 보여줍니다. 우리의 코드와 데이터는 공개되어 있습니다.



### DEGAP: Dual Event-Guided Adaptive Prefixes for Templated-Based Event Argument Extraction Model with Slot Querying (https://arxiv.org/abs/2405.13325)
- **What's New**: 최근 이벤트 인수 추출(EAE)에 관한 연구는 훈련 및 추론 중 모델에 조회된 인스턴스와 이벤트 템플릿과 같은 보조정보를 통합하는 방법을 포함하고 있습니다. 그러나 이러한 접근법들은 조회의 한계로 관련 이벤트 인스턴스를 충분히 활용하지 못하고, 중요한 이벤트 템플릿 정보를 간과하며, 접두사(prefix)의 장점이 제한된다는 문제를 가지고 있습니다. 이러한 문제를 해결하기 위해 새로운 모델 DEGAP이 제안되었습니다.

- **Technical Details**: DEGAP은 두 가지 구성 요소로 위의 문제들을 해결합니다. 첫째, Dual Prefixes(이중 접두사)에서는 인스턴스 지향 접두사와 템플릿 지향 접두사가 각각 다른 이벤트 인스턴스와 템플릿에서 정보를 학습하여 EAE 모델에 조회 없이 관련 정보를 제공합니다. 둘째, Event-guided Adaptive Gating Mechanism(이벤트 유도 적응 게이팅 메커니즘)은 목표 이벤트를 기반으로 접두사를 안내하여 그들의 장점을 최대로 활용합니다.

- **Performance Highlights**: 광범위한 실험을 통해 DEGAP은 ACE05, RAMS, WIKIEVENTS, MLEE 등 네 가지 데이터셋에서 새로운 최첨단 성능을 달성했음을 입증했습니다. 추가 분석에서는 제안된 설계의 중요성과 주요 구성 요소의 효과를 확인했습니다.



### ''You should probably read this'': Hedge Detection in Tex (https://arxiv.org/abs/2405.13319)
- **What's New**: 이 논문은 언어를 통해 아이디어, 신념 및 진술을 전달하는 인간의 표현에서, 저자의 진술에 대한 확신 정도를 분석하는 새로운 접근법을 소개합니다. 이는 특히 실수로 인해 치명적인 결과를 초래할 수 있는 의학, 금융, 공학 등 다양한 분야에서 매우 중요합니다.

- **Technical Details**: 이 연구는 텍스트에서 hedge(애매한 진술) 탐지를 개선하기 위해 단어와 품사 태그(part-of-speech tags)를 활용한 joint model(공동 모델)을 적용합니다. 이 방법으로 CoNLL-2010 Wikipedia corpus에서 새로운 최고 점수를 달성했습니다.

- **Performance Highlights**: 새로운 joint model이 CoNLL-2010 Wikipedia corpus에서 최고 성능을 기록했습니다. 이는 텍스트에서 저자의 확신 정도를 보다 정밀하게 감지할 수 있는 가능성을 제시합니다.



### Metadata Integration for Spam Reviews Detection on Vietnamese E-commerce Websites (https://arxiv.org/abs/2405.13292)
Comments:
          Accepted for publication in International Journal of Asian Language Processing (IJALP)

- **What's New**: 이 논문에서는 ViSpamReviews v2 데이터셋을 도입하여 전자상거래 사이트에서 스팸 리뷰(의견)를 분류하는 문제를 다루고 있습니다. 이 데이터셋은 리뷰의 메타데이터를 포함하여 부가적인 속성을 통합함으로써 스팸 리뷰 분류의 정확성을 높이는 것을 목표로 합니다.

- **Technical Details**: 논문은 텍스트와 범주형 속성을 동시에 통합하는 새로운 접근 방식을 제안합니다. 실험 결과, DNN (Deep Neural Network) 모델과 결합했을 때 제품 카테고리가 효과적임을 보여주었으며, 텍스트 특성은 PhoBERT 모델과 결합할 때 최상의 성능을 발휘했습니다. 특히 PhoBERT 모델은 SentenceBERT와 결합된 SPhoBERT 모델에서 생성된 제품 설명 기능과 함께 사용할 때 최고 정확도를 달성했습니다.

- **Performance Highlights**: 스팸 리뷰 분류 작업에서 매크로 평균 F1 점수를 사용했을 때 성능이 87.22%로, 표준 모델에 비해 1.64% 향상되었습니다. 스팸 리뷰 유형을 식별하는 작업에서는 정확도가 73.49%로 1.93% 향상되었습니다.



### DiffNorm: Self-Supervised Normalization for Non-autoregressive Speech-to-speech Translation (https://arxiv.org/abs/2405.13274)
- **What's New**: 이 논문에서는 비자기회귀 트랜스포머(Non-autoregressive Transformers, NATs)를 이용한 직관적 음성-음성 번역 시스템에 대한 연구를 제시합니다. 연구의 핵심은 DiffNorm이라는 새로운 노이즈 제거 기반 정규화 기법을 도입하여, 복잡한 데이터 분포로 인한 비일관적이고 반복적인 결과 문제를 해결하고자 합니다. DiffNorm은 자기지도형 노이즈 추정 목표에 의해 학습된 후, 합성적으로 손상된 음성 특징을 제거하여 정규화된 타겟 데이터를 구성합니다.

- **Technical Details**: DiffNorm은 자기지도형 노이즈 추정(Objective) 기법을 사용하며, 이러한 과정은 음성의 합성적 손상을 제거하여 데이터 분포를 간단하게 만듭니다. 또한, 정규화된 타겟 데이터를 구축합니다. 추가로, NAT 모델의 강인성과 번역 품질을 향상시키기 위해 분류기-프리 가이드(Classifier-free guidance)를 사용하여, 학습 중 소스 정보의 무작위 드롭아웃(Dropout)을 제안합니다.

- **Performance Highlights**: 제안된 전략은 CVSS 벤치마크에서 영어-스페인어(En-Es) 번역에서 약 +7 ASR-BLEU 향상, 영어-프랑스어(En-Fr) 번역에서 약 +2 ASR-BLEU 향상을 나타냈습니다. 또한, En-Es 번역에서는 14배, En-Fr 번역에서는 5배 이상의 속도 향상을 Autoregressive 베이스라인과 비교하여 달성했습니다.



### A Multilingual Similarity Dataset for News Article Fram (https://arxiv.org/abs/2405.13272)
- **What's New**: 이번 연구에서는 16,687개의 새로 라벨링된 쌍(pair)을 포함하는 대규모 뉴스 기사 데이터셋의 확장판을 소개합니다. 이는 기존의 전통적인 프레임 분석 연구에서 수작업으로 프레임 클래스를 식별하는 작업을 해결하는 데 목적을 두고 있습니다. 총 10개 언어에 걸쳐 26,555개의 라벨링된 뉴스 기사 쌍을 제공하여, 현재까지 가장 광범위한 다언어 뉴스 기사 유사성 데이터셋을 제안합니다.

- **Technical Details**: 이 데이터셋은 인간-인-루프(human-in-the-loop) 프레임워크를 기반으로 뉴스 콘텐츠의 8가지 주요 측면을 상세히 설명하는 코드북(codebook)에 따라 주의 깊게 주석이 달렸습니다. 이 방법은 뉴스 기사를 쌍(pairwise)으로 비교하여 프레임 클래스를 자동으로 식별할 수 있도록 하여 기존의 수작업 식별의 부담을 덜어줍니다.

- **Performance Highlights**: 데이터셋의 응용 예시는 글로벌 뉴스 보도에서 국가 커뮤니티 발굴, 매체 편향성 노출, 뉴스 생성 관련 요인 계량화 등의 가능성을 보여줍니다. 이러한 뉴스 유사성 데이터셋은 국가, 위치, 언어 및 기타 사회적 구조 전반에서 사건과 관점의 뉴스 보도를 이해하는 데 도움을 줄 수 있으며, 사회 과학 연구와 적용 방법론의 발전을 촉진할 수 있습니다.



### MELD-ST: An Emotion-aware Speech Translation Datas (https://arxiv.org/abs/2405.13233)
Comments:
          9 pages. Accepted to ACL 2024 Findings. Dataset: this https URL

- **What's New**: 이번 연구에서는 감정을 고려한 음성 번역(speech translation)의 중요성을 강조하고 있습니다. 이를 위해 English-to-Japanese와 English-to-German 언어 쌍을 포함하는 MELD-ST 데이터셋을 소개합니다. 각각의 언어 쌍은 MELD 데이터셋에서 감정 레이블(emotion labels)을 포함한 약 10,000개의 발화를 포함하고 있습니다.

- **Technical Details**: 기본 베이스라인 실험에서는 SeamlessM4T 모델을 사용하여 데이터셋을 테스트했습니다. 실험 결과, 감정 레이블로 세밀 조정(fine-tuning)을 하면 특정 상황에서 번역 성능이 향상될 수 있음이 밝혀졌습니다.

- **Performance Highlights**: 이번 연구는 감정 인식 기반의 음성 번역 시스템에서 감정 레이블을 활용함으로써 성능 향상을 도모할 수 있음을 강조하며, 이 분야에서의 추가적인 연구 필요성을 제시하고 있습니다.



### Dataset Decomposition: Faster LLM Training with Variable Sequence Length Curriculum (https://arxiv.org/abs/2405.13226)
- **What's New**: 대규모 언어 모델(LLMs)의 훈련 방법 개선을 위한 'dataset decomposition' 기법이 도입되었습니다. 이 방법은 고정 길이 토큰 시퀀스 대신 변수 시퀀스 길이를 사용하는 새로운 훈련 기술입니다.

- **Technical Details**: 기존의 concat-and-chunk 방식을 사용하면 시퀀스 내에서 비효율적인 cross-document attention이 발생하며, 이로 인해 계산 비용이 증가합니다. 반면, 제안된 방법은 문서 길이에 비례한 패널티를 적용하여 효율성을 높입니다. 구체적으로, 문서를 고유한 길이에 따라 여러 bucket으로 분해하고, 이 bucket에서 변동되는 시퀀스 길이와 배치 크기를 사용해 샘플링합니다.

- **Performance Highlights**: 새로운 방법을 통해 8k context-length 1B 모델을 기존 2k context-length 모델과 동일한 비용으로 훈련할 수 있게 되었습니다. 또한 웹 스케일 코퍼스에서의 실험 결과, 새로운 접근법은 표준 언어 평가와 긴 컨텍스트 벤치마크에서 성능을 크게 향상시켰고, 목표 정확도에 baseline 대비 3배 빠르게 도달하였습니다.



### Equipping Transformer with Random-Access Reading for Long-Context Understanding (https://arxiv.org/abs/2405.13216)
Comments:
          Preliminary works for a Google Student Researcher Project

- **What's New**: 이번 논문에서는 긴 문서의 목표 지향적 읽기(goal-oriented reading)를 위해 새로운 읽기 전략인 '랜덤 액세스(random access)'를 제안합니다. 이는 전통적인 Sequential Access 방식과 달리, 모든 토큰을 순차적으로 읽지 않고 중요하지 않은 수백 개의 토큰을 생략할 수 있는 능력을 Transformer 모델에 부여합니다.

- **Technical Details**: 기존의 방법들은 텍스트 청킹(text chunking), 커널 접근법(kernel approach), 구조화된 주의(attention)와 같은 기술을 통해 연산 복잡성을 해결하고 위치 인코딩(positional encoding), 추가 사전 학습(continued pretraining), 데이터 엔지니어링(data engineering) 등을 통해 길이 extrapolation 문제를 해결해왔습니다. 하지만 이번 연구에서는 임의의 액세스(random access)를 통해 Transformer 모델이 특정 목표를 위해 효율적으로 긴 문서를 처리할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, 사전 학습, 미세 조정(fine-tuning), 추론(inference) 단계에서 제안된 방법의 효능이 검증되었습니다. 이로써 모델이 목표 지향적인 읽기에서 더욱 효과적임을 입증했습니다.



### Investigating Symbolic Capabilities of Large Language Models (https://arxiv.org/abs/2405.13209)
- **What's New**: 대형 언어 모델(LLMs)에서 많이 연구된 언어 기반 추론 및 수학 문제 해결에 비해, 기호 기반 계산과 추론에 대한 LLMs의 잠재력을 평가하는 연구는 부족합니다. 이번 연구는 덧셈, 곱셈, 모듈로스 산술, 수치적 정밀도, 기호적 카운팅(task)과 같은 여러 상징적 과제에서 LLMs를 평가하여 이 격차를 메꾸고자 합니다.

- **Technical Details**: 본 연구는 Chomsky 계층(Chomsky's Hierarchy)을 기반으로 한 평가 프레임워크를 활용하여, LLMs의 계산 능력을 체계적으로 측정합니다. 평가 방식에는 최소한의 설명을 포함한 프롬프트(minimally explained prompts), 그리고 zero-shot Chain of Thoughts 기법이 사용되어 모델이 솔루션 프로세스를 자율적으로 탐색할 수 있게 합니다. 평가에는 4개의 기업용 모델과 4개의 오픈 소스 모델을 포함한 8개의 LLMs가 사용되었으며, 그 중 3개 모델은 수학적 작업에 대해 사전 학습되었습니다.

- **Performance Highlights**: 연구 결과, 기호의 수를 통해 표현된 복잡도가 증가함에 따라 문맥 자유(context-free) 및 문맥 민감(context-sensitive) 상징적 과제에서 LLMs의 성능이 크게 하락하는 것을 확인했습니다. 세밀하게 튜닝된 GPT3.5조차도 성능 향상이 미미하게 나타났으며, 다른 모델에서도 비슷한 성능 추세가 관찰되었습니다. 전반적으로 모든 모델이 이러한 상징적으로 복잡한 작업에 대해 제한된 일반화 능력을 나타냈습니다. 이 연구는 상징적 복잡도 증가에 따른 LLMs의 도전 과제를 강조하고, 기호 기반 추론 작업의 전문성을 높이기 위한 특화된 훈련, 메모리 및 구조적 조정의 필요성을 시사합니다.



### Comparative Analysis of Different Efficient Fine Tuning Methods of Large Language Models (LLMs) in Low-Resource Setting (https://arxiv.org/abs/2405.13181)
Comments:
          9 pages of main paper, 1 page of references, 6 appendix pages, 11 figures, 18 tables

- **What's New**: 본 논문에서는 대규모 언어 모델(LLM)에서 다양한 fine-tuning 전략을 동일한 기준으로 비교하고자 시도하였습니다. 특히, Vanilla Fine-Tuning (FT)과 Pattern-Based Fine Tuning (PBFT), adaptive fine-tuning, 그리고 LoRA 어댑터의 효율성을 몇-shot 세팅에서 조사하였습니다. 마지막으로 최신 방법인 context distillation을 기존의 fine-tuning 방법들과 비교했습니다.

- **Technical Details**: 이 연구에서는 COLA와 MNLI 두 개의 데이터셋을 이용하여, Vanilla FT, PBFT, adaptive fine-tuning, LoRA 어댑터, 그리고 context distillation을 비교하였습니다. 실험은 사전 훈련된 모델에 몇-shot 세팅을 적용하면서 진행되었습니다.

- **Performance Highlights**: 실험 결과, PBFT는 OOD 데이터에서 Vanilla FT보다 성능이 낮았으며, 이는 효과적인 프롬프트의 필요성을 강조합니다. Adaptive fine-tuning과 LoRA 실험은 예상대로 표준 fine-tuning에 비해 약간 낮은 성능을 보였습니다. 하지만 context distillation은 표준 fine-tuning 방법들을 뛰어넘는 성능을 보였습니다. 이 연구는 적절한 fine-tuning 방법의 선택이 사용 가능한 자원(메모리, 연산력, 데이터)과 과제 적응성에 따라 달라짐을 시사합니다.



### RAG-RLRC-LaySum at BioLaySumm: Integrating Retrieval-Augmented Generation and Readability Control for Layman Summarization of Biomedical Texts (https://arxiv.org/abs/2405.13179)
- **What's New**: 이 논문은 복잡한 생물의학 연구(biomedical research)를 일반인도 이해할 수 있도록 돕는 고급 자연어 처리 (Natural Language Processing, NLP) 기술인 RAG-RLRC-LaySum 프레임워크를 소개합니다. 우리의 ‘Retrieval Augmented Generation (RAG)’ 솔루션은 재정렬(reranking) 방법으로 강화되어 여러 지식 소스를 활용함으로써 요약문의 정밀성과 적절성을 보장합니다. 이외에도 ‘Reinforcement Learning for Readability Control (RLRC)’ 전략을 통해 비전문가도 과학적 내용을 쉽게 이해할 수 있도록 가독성을 향상시킵니다.

- **Technical Details**: RAG 솔루션은 여러 지식 소스를 활용하고 재정렬 방법을 사용하여 요약문의 질을 높입니다. RLRC 전략은 미국의 PLOS 및 eLife 데이터셋을 이용하여 강화학습을 통해 가독성을 제어합니다. RAG와 RLRC가 결합된 LaySum 프레임워크는 일반인이 이해하기 어려운 생물의학 연구 내용을 쉽게 이해할 수 있도록 하여 과학적 지식을 민주화합니다.

- **Performance Highlights**: 우리의 방법은 Plain Gemini 모델보다 20% 높은 가독성 점수와 15% 향상된 ROUGE-2 관련성 점수, 그리고 10% 향상된 사실 정확도를 나타냈습니다. 이는 RAG-RLRC-LaySum 프레임워크가 일반 대중의 생물의학 발견과의 소통을 더욱 원활하게 한다는 것을 의미합니다.



### Dataset Mention Extraction in Scientific Articles Using Bi-LSTM-CRF Mod (https://arxiv.org/abs/2405.13135)
- **What's New**: 이 논문은 과학 연구에서 데이터셋의 중요성을 강조하고, 데이터셋 인용이 표준적 관행이 아니므로 이를 자동으로 추출하는 방법을 제안합니다. 데이터를 자동으로 추출하는 신경망 방법론을 제안하면서, 사회 과학 논문에서 높은 성능을 기록했습니다.

- **Technical Details**: 이 연구에서는 Bi-LSTM-CRF 아키텍처 기반의 신경망을 사용하여 데이터셋을 추출합니다. Bi-LSTM(Bidirectional Long Short-Term Memory)과 CRF(Conditional Random Fields)를 결합하여 문서에서 데이터셋 언급을 효율적으로 추출합니다. 이 방법은 Rich Context Dataset의 사회과학 논문에서 테스트되었으며 F1 점수 0.885를 기록했습니다.

- **Performance Highlights**: 제안된 모델은 사회과학 논문에서 데이터셋 언급 추출 작업에서 F1 점수 0.885를 달성했습니다. 이는 데이터셋의 사용과 중요성을 추적하는데 유용한 결과로 보입니다.



### Atomic Self-Consistency for Better Long Form Generations (https://arxiv.org/abs/2405.13131)
Comments:
          12 pages

- **What's New**: 최근 연구는 LLM(Large Language Models)의 생성물에서 잘못된 정보를 걸러내어 응답의 정확성을 향상시키고자 하는 시도에 집중했습니다. 본 논문에서는 Atomic Self-Consistency(ASC)라는 기술을 소개하며, 이는 LLM 응답에서 관련 정보의 리콜(recall)을 향상시키는 방법입니다.

- **Technical Details**: ASC는 기존의 Universal Self-Consistency(USC) 기술을 바탕으로 합니다. USC는 LLM에서 여러 확률적 샘플을 사용하여 최상의 단일 응답을 선택하는 반면, ASC는 샘플들에서 참된(subparts) 부분만을 선택하고 이것들을 통합하여 우수한 복합 응답(composite answer)을 생성합니다. 이를 통해 리콜이 중요한 긴 형태의 응답에서 성능을 크게 향상시킬 수 있습니다.

- **Performance Highlights**: 다양한 실험과 제거 실험(ablations)을 통해 여러 샘플의 관련 부분을 통합하는 것이 단일 샘플을 선택하는 것보다 성능이 훨씬 뛰어남을 보여주었습니다. ASC는 ChatGPT와 Llama2를 이용하여 ASQA, QAMPARI, QUEST, ELI5와 같은 다수의 사실기반 및 개방형 질문 응답 데이터셋에서 USC보다 상당한 이점을 나타냈습니다. 또한, 여러 샘플을 통합하는 접근법을 통해 장문의 생성물을 더욱 향상시킬 수 있는 잠재력을 밝혀냈습니다.



### Presentations are not always linear! GNN meets LLM for Document-to-Presentation Transformation with Attribution (https://arxiv.org/abs/2405.13095)
Comments:
          This paper is under review in a conference

- **What's New**: 이 연구에서는 긴 문서로부터 프레젠테이션을 자동으로 생성하는 새로운 방법을 제안합니다. 기존의 단순 요약과 달리, 프레젠테이션에는 비선형적인 서사가 필요합니다. 즉, 슬라이드의 콘텐츠는 문서의 여러 비연속적인 부분에서 나올 수 있습니다. 이를 위해 그래프 기반 솔루션을 제안합니다.

- **Technical Details**: 이 솔루션은 입력 문서에서 그래프를 학습한 후, 그래프 신경망(Graph Neural Network)과 LLM을 결합하여 각 슬라이드의 콘텐츠 출처를 명확히 하는 프레젠테이션을 생성합니다. 이 접근 방식은 문서의 비선형적인 매핑과 콘텐츠 충실도를 보장하기 위한 것입니다.

- **Performance Highlights**: 직접적으로 LLM을 사용하는 방법과 비교하여, 제안한 그래프 기반 솔루션이 더 뛰어난 성능을 보였음을 실험을 통해 확인했습니다. LLM의 환각(hallucination) 문제와 긴 입력 문서에 대한 성능 저하 문제를 해결하는 데 효과적입니다.



### Multi-domain Knowledge Graph Collaborative Pre-training and Prompt Tuning for Diverse Downstream Tasks (https://arxiv.org/abs/2405.13085)
Comments:
          Work in progress. Code and data will be open-sourced at this https URL

- **What's New**: MuDoK는 다중 도메인 협력 사전 훈련(Multi-domain collaborative pre-training) 및 효율적인 프리픽스 프롬프트 튜닝(Prefix prompt tuning)을 통해 추천 시스템(recommendation)과 텍스트 이해(text understanding)와 같은 다양한 다운스트림 작업을 지원하는 프레임워크를 제안합니다. 또한, 새로운 다중 도메인 KGP 벤치마크(Benchmark)인 KPI을 구축하여, 두 개의 대규모 KGs와 여섯 가지 다른 서브 도메인 작업을 포함하였습니다. 이 벤치마크는 오픈 소스(Open-source)로 제공되어 후속 연구를 촉진합니다.

- **Technical Details**: MuDoK는 플러그 앤 플레이(Plug-and-play) 방식의 프롬프트 학습 접근 방식을 사용하여 다양한 다운스트림 작업 백본(Backbone)에 유연하게 적용할 수 있습니다. 이를 통해 학습 효율성과 이전 가능성(Transferability)을 크게 향상시켰습니다. KPI 벤치마크를 기반으로 다양한 백본 모델을 활용하여 종합적인 성능 평가를 수행했습니다.

- **Performance Highlights**: 실험 결과, MuDoK가 타 프레임워크에 비해 성능 향상이 두드러졌으며, 일반성(Generality), 효율성(Efficiency), 이전 가능성에서 모두 뛰어난 결과를 보였습니다.



### The 2nd FutureDial Challenge: Dialog Systems with Retrieval Augmented Generation (FutureDial-RAG) (https://arxiv.org/abs/2405.13084)
- **What's New**: 미래 대화 시스템 챌린지 2024(SLT 2024)에서 제2회 'FutureDial Challenge: Retrieval Augmented Generation (RAG)'가 개최됩니다.

- **Technical Details**: 이 대회는  Retrieval Augmented Generation (RAG) 기법을 활용한 대화 시스템(Dial System)의 발전을 목표로 합니다. RAG는 Retrieval과 Generation의 장점을 결합한 것으로, 질문 또는 대화 쿼리에 대해 관련 컨텐츠를 검색한 뒤, 이를 바탕으로 자연스럽고 유의미한 답변을 생성합니다.

- **Performance Highlights**: 참가자들은 주어진 질문 및 대화 시나리오에 대해 RAG 모델의 성능을 최적화하여 개선된 대화 시스템을 구현할 수 있는 기회를 갖게 됩니다. 이 대회를 통해, 더욱더 자연스럽고 효율적인 대화 시스템의 발전을 기대할 수 있습니다.



### A Novel Method for News Article Event-Based Embedding (https://arxiv.org/abs/2405.13071)
- **What's New**: 이 논문은 기존 뉴스 임베딩(news embedding) 방법이 뉴스 이벤트의 잠재적 맥락을 포착하는 데 최적화되지 않았다는 문제점을 해결하기 위해 새로운 경량(news embedding) 방법을 제안합니다. 이 새로운 방법은 뉴스 기사에서 언급된 엔터티(entities)와 테마(themes) 및 이들과 특정 이벤트 간의 역사적 연결성을 중점적으로 고려하여 임베딩을 생성합니다.

- **Technical Details**: 제안된 방법은 세 단계로 구성됩니다. 첫 번째 단계에서는 뉴스 기사를 처리하고 이벤트(events), 엔터티 및 테마를 추출합니다. 두 번째 단계에서는 현재 및 과거 데이터를 기반으로 시기별로 구분한 GloVe 모델을 훈련하여 테마 및 엔터티의 주기적 시간 임베딩(time embedding)을 생성합니다. 마지막으로 SIF(Smooth Inverse Frequency)를 사용하여 기사 수준 벡터(article-level vectors)를 생성하고, 쌍둥이 신경망(Siamese Neural Networks)을 사용하여 이벤트 관련 정보가 포함된 임베딩을 생성합니다.

- **Performance Highlights**: GDELT 프로젝트에서 850,000개 이상의 뉴스 기사와 1,000,000개 이상의 이벤트를 활용하여 제안된 방법을 테스트 및 평가했습니다. 검증을 위해, 동일한 이벤트 감지 작업을 하루 및 한 달 단위로 두 번 적용한 결과, 제안된 방법이 모든 작업과 데이터셋에서 Precision-Recall(PR) AUC을 상당히 개선한 것을 확인했습니다. 구체적으로는 SIF 방법보다 매일 및 매월 이벤트 감지 작업에서 평균 각각 2.15% 및 2.57% 향상되고, 반감독 방식(semi-supervised approach)보다 각각 2.57% 및 2.43% 더 성능이 개선되었습니다.



### RNG: Reducing Multi-level Noise and Multi-grained Semantic Gap for Joint Multimodal Aspect-Sentiment Analysis (https://arxiv.org/abs/2405.13059)
Comments:
          Accepted by ICME 2024

- **What's New**: 이번 연구에서는 텍스트-이미지 쌍에서 어스펙트 용어(aspect terms)와 관련된 감정 극성(sentiment polarities)을 공동으로 추출하는 중요한 다중모달 감정 분석(multimodal sentiment analysis) 과제인 JMASA(Joint Multimodal Aspect-Sentiment Analysis)를 다룹니다. 새로운 프레임워크 'RNG'를 제안하여 기존 작업에서 직면한 두 가지 주요 문제(모달리티 노이즈와 의미적 갭)를 해결합니다.

- **Technical Details**: RNG 프레임워크는 다음과 같은 세 가지 제약 조건을 설계하여 다중 레벨 모달리티 노이즈(multi-level modality noise) 및 다중 의미적 갭(multi-grained semantic gap)을 동시에 줄이는 것을 목표로 합니다: (1) 텍스트-이미지 유사도에 기반한 글로벌 관련성 제약(Global Relevance Constraint, GR-Con)을 통해 인스턴스 기반 노이즈를 줄이는 방법, (2) 정보 병목 원칙(Information Bottleneck) 기반의 정보 병목 제약(Information Bottleneck Constraint, IB-Con)을 통해 특징 기반 노이즈를 감소시키는 방법, 그리고 (3) 상호 정보 최대화를 기반으로 한 대조 학습 방법을 이용한 의미적 일관성 제약(Semantic Consistency Constraint, SC-Con)을 통해 다중 의미적 갭을 줄이는 방법입니다.

- **Performance Highlights**: 두 개의 데이터셋에서 수행된 광범위한 실험을 통해 새로운 프레임워크가 최신 성능을 달성함을 검증했습니다. 이는 JMASA 과제에서 매우 중요한 발전을 보여줍니다.



### Large language models for sentiment analysis of newspaper articles during COVID-19: The Guardian (https://arxiv.org/abs/2405.13056)
- **What's New**: COVID-19 팬데믹 동안, 뉴스 미디어는 바이러스 전파, 의료 자원 배분, 정부 대응 조치를 포함한 다양한 주제를 다룹니다. 이 연구는 특정 국가의 신문 소스를 대상으로 COVID-19 기간 동안의 감정 분석을 제공합니다. 연구는 The Guardian 신문을 선택하여 초기 전염, 봉쇄, 백신 접종 기간 동안의 감정 분석을 수행합니다.

- **Technical Details**: 이번 연구에서는 대규모 언어 모델 (Large Language Models, LLMs)을 사용하고, 전문가가 라벨링한 감정 분석 데이터를 통해 모델을 세밀하게 조정하였습니다. 또한, 팬데믹 이전의 감정 상태를 비교할 수 있는 분석을 제공했습니다.

- **Performance Highlights**: 연구 결과에 따르면 초기 팬데믹 단계에서 공공 감정은 긴급한 위기 대응에 중점을 두었으며, 이후에는 건강과 경제에 미친 영향에 초점을 맞추었습니다. 소셜 미디어 감정 분석 결과와 비교했을 때 The Guardian에서는 슬픔, 짜증, 불안, 부정 등 부정적 감정이 지배적이었습니다. 이는 소셜 미디어가 더 다채로운 감정 반영을 제공하는 것과 대조적입니다. The Guardian은 팬데믹 전후 뉴스 섹션 전반에서 전체적으로 부정적 감정이 지배적임을 나타냈습니다.



### Large Language Models for Medicine: A Survey (https://arxiv.org/abs/2405.13055)
Comments:
          Preprint. 5 figures,5 tables

- **What's New**: 디지털 경제의 디지털 인텔리전스(digital intelligence) 환경에서 대규모 언어 모델(LLMs)이 개발되고 있습니다. 의료 분야에서의 이용 가능성을 탐구하며, 현재 존재하는 모델에 대한 간략한 개요를 제공합니다. 이 논문은 더 발전된 연구 방향을 모색하고 향후 의료 응용을 위한 연구자들에게 도움이 되고자 합니다.

- **Technical Details**: 의료 LLMs은 다양한 의료 시나리오에 적용될 수 있는 잠재력을 지니고 있으며, 이러한 모델들을 위한 요구사항과 응용 사례를 중점적으로 다룹니다. 의학적 LLMs의 개발 과정에서 만나는 여러 도전 과제를 강조하며, 향후 기술적 통합을 위한 방향성도 제시합니다. 이는 앞으로 더 나은 의료 응용을 위한 로드맵을 제공합니다.

- **Performance Highlights**: 언어 모델의 발전은 상당한 컴퓨팅 파워와 자원의 확보로 인해 가능해졌으며, 이는 다양한 도메인에서의 통합을 허용했습니다. 특히 의료 분야에서 LLMs가 가지는 이점과 개발 과정에서 직면하는 도전 과제들도 다루어집니다.



### MeteoRA: Multiple-tasks Embedded LoRA for Large Language Models (https://arxiv.org/abs/2405.13053)
Comments:
          19 pages

- **What's New**: 이번 연구에서는 다양한 하위 작업에서 대규모 언어 모델(LLMs)을 효율적으로 튜닝하기 위한 새로운 프레임워크인 	extbf{M-TELoRA}를 소개합니다. 이는 기존의 Low-Rank Adaptation (LoRA) 방식을 확장하여 여러 LoRA 어댑터를 단일 LLM에 융합하는 기법을 제안합니다. 이를 통해 모델은 자동으로 적절한 어댑터를 선택하여 작업을 처리할 수 있습니다.

- **Technical Details**: 	extbf{M-TELoRA}는 Mixture-of-Experts (MoE) 스타일로 다양한 LoRA 어댑터를 하나의 기본 LLM에 통합합니다. 이를 통해 입력된 작업에 따라 적절한 어댑터를 자동으로 선택하여 다양한 구성 요소 문제를 해결할 수 있습니다. 이번 연구에서는 LlaMA2-13B와 LlaMA3-8B 기본 모델에 28개의 상용 LoRA 어댑터를 	extbf{M-TELoRA}를 통해 통합하였습니다.

- **Performance Highlights**: 	extbf{M-TELoRA}로 통합된 모델은 개별 어댑터와 동등한 성능을 유지하면서도, 복합 과제를 해결할 때 단일 추론 과정에서 10개의 문제를 순차적으로 해결하는 데 있어 우수한 성능을 보여줍니다. 이를 통해 적시 의도 전환이 가능함을 확인하였습니다.



### SemEval-2024 Task 3: Multimodal Emotion Cause Analysis in Conversations (https://arxiv.org/abs/2405.13049)
Comments:
          12 pages, 3 figures, 4 Tables

- **What's New**: 새로운 연구는 세메벌(SemEval)-2024 작업 3(Multimodal Emotion Cause Analysis in Conversations)을 조직하여, 대화에서 감정(emotion)과 그 원인(cause)을 분석하는 작업을 다룹니다. 이번 과제는 대화에서 감정과 감정 원인의 모든 쌍을 추출하는 것을 목표로 하고 있습니다. 이 작업은 143명의 등록자와 216개의 성공적인 제출을 유도했습니다.

- **Technical Details**: 이 연구에서는 두 가지 하위 과제를 설정했습니다: 대화에서 텍스트 기반 감정-원인 쌍 추출(Textual Emotion-Cause Pair Extraction in Conversations, TECPE)과 다중 모드 기반 감정-원인 쌍 추출(Multimodal Emotion-Cause Pair Extraction in Conversations, MECPE)입니다. 데이터셋과 평가 설정도 함께 소개하고 있으며, 참가자들의 시스템 요약과 발견 점에 대해서도 논의하고 있습니다.

- **Performance Highlights**: 참가자들의 시스템을 요약하고 최고 팀들의 성과를 분석함으로써, 대화 분석에서의 감정과 그 원인을 더 정확하게 예측하는 데 기여했습니다. 이 연구는 앞으로의 대화 분석 및 감정 인식 기술 발전에 중요한 기여를 할 것으로 예상됩니다.



### LeaPformer: Enabling Linear Transformers for Autoregressive and Simultaneous Tasks via Learned Proportions (https://arxiv.org/abs/2405.13046)
Comments:
          Submitted and accepted at ICML 2024

- **What's New**: 최신 연구에서는 모델 성능을 유지하기 위해 위치 기반 재가중치 함수(position-based re-weighting functions)를 사용하는 것이 유망한 방법으로 밝혀졌습니다. 이에 대한 한계점을 극복하기 위해 Learned Proportions(LeaP) 및 LeaPformers를 제안했습니다. 이것은 명시적 위치 표현 및 시퀀스 길이에 대한 의존성을 시퀀스 비율(sequence proportions)로 일반화하는 방식으로 전환하는 것입니다.

- **Technical Details**: LeaPformers는 두 가지 주요 구성 요소를 특징으로 합니다. 첫째, 명시적 위치 표현 및 시퀀스 길이에 대한 의존성을 시퀀스 비율로 대체하여 재가중치에 대한 의존성을 일반화했습니다. 둘째, 정적 위치 표현 대신 컴팩트 모듈을 통해 유도된 동적 비율(dynamic proportions)을 사용하여 유연한 주의 집중 패턴을 가능하게 했습니다.

- **Performance Highlights**: LeaPformer는 8개의 대표적인 효율적 트랜스포머(efficient transformers)와 함께 Long-Range Arena benchmark에서 평가되었으며, 최상의 품질-처리량(quality-throughput) 트레이드오프를 달성한 것으로 나타났습니다. 추가적으로, Wikitext-103 자회귀 언어 모델링과 두 언어 쌍에 대한 동시 음성-텍스트 번역 실험에서도 경쟁력 있는 결과를 얻었다고 보고되었습니다.



### Case-Based Reasoning Approach for Solving Financial Question Answering (https://arxiv.org/abs/2405.13044)
- **What's New**: 이번 연구는 최근 언어 모델들이 텍스트 기반 작업에서 뛰어난 성능을 보여주지만 복합적인 이유 문제(complex reasoning tasks)에서의 성과가 미흡함을 지적하고 있습니다. 이를 해결하기 위해 연구진은 CBR(case based reasoning)이라는 새로운 접근 방식을 제안했습니다. 이 접근 방식을 사용하면 문제에 대한 해결 방법을 제공하는 유사 케이스(질문 및 논리 프로그램)를 제공하여 문제를 해결할 수 있습니다.

- **Technical Details**: 제안된 모델은 주어진 질문을 해결하기 위해 관련 케이스를 검색하고, 검색된 케이스와 문맥 정보를 기반으로 답을 생성합니다. 이를 통해 복합적인 다단계 프로그램(multi-step programs)을 해결하는데 도움을 주고자 합니다. 실험을 통해 FinQA 데이터셋에서 이 접근 방식의 경쟁력을 입증하였습니다.

- **Performance Highlights**: FinQA 데이터셋 실험 결과, 제안된 방법이 상당한 성능을 보였으며, 케이스 저장소(case repository)를 확장함으로써 FinQA가 약점을 보였던 복잡한 다단계 프로그램을 해결하는 데 도움이 된다는 것을 추가적으로 확인하였습니다.



### Assessing Political Bias in Large Language Models (https://arxiv.org/abs/2405.13041)
Comments:
          5 pages, 2 figures

- **What's New**: 최근 AI 윤리와 그 영향에 대한 담론에서 대규모 언어 모델(LLMs)의 사회적 편향성 평가가 중요한 문제가 되었습니다. 특히, 정치적 편향성을 인식하고 고려하는 것은 실용적인 응용 프로그램에서 중요합니다. 이번 연구는 유럽연합(EU) 내 정치적 문제에 대한 가장 인기 있는 오픈소스 모델들에 대한 편향을 독일의 관점에서 평가합니다.

- **Technical Details**: 연구진은 독일에서 사용하는 투표 조언 애플리케이션인 'Wahl-O-Mat'을 사용하여 각각의 LLM이 어느 정치 정당과 가장 일치하는지를 평가했습니다. 이를 통해 Llama3-70B와 같은 큰 모델이 좌파 성향의 정당(예: GRÜNE, Volt)과 더 밀접하게 일치하는 경향이 있다는 것을 발견했습니다. 반면, 작은 모델은 특히 영어에서 중립적인 경향을 보였습니다.

- **Performance Highlights**: 이 연구는 LLM이 보여주는 미묘한 행동 양상과 언어가 그들의 정치적 입장을 형성하는 데 중요한 역할을 한다는 점을 강조합니다. 현대 머신 러닝(Machine Learning) 방법을 사용하는 애플리케이션의 무결성과 공정성을 보장하기 위해 LLM의 사회적 편향성을 엄격히 평가하고 해결하는 것이 중요합니다.



### Surgical Feature-Space Decomposition of LLMs: Why, When and How? (https://arxiv.org/abs/2405.13039)
Comments:
          Accepted at ACL 2024

- **What's New**: 이번 연구에서는 트랜스포머 기반의 대형 언어 모델(LLMs)에 대해 가중치와 특성 공간(Weight and Feature Space)의 저랭크 근사(Low-rank approximation)가 효과적임을 실증적으로 연구했습니다. 특히, 저랭크 근사 기법이 해당 모델들의 일반화 성능 개선 및 추론 지연(타임) 감소에 중요한 역할을 함을 입증했습니다. 더 나아가, 일부 경우에서 일반 상식(Common sense)이 향상된다는 사실도 발견했습니다.

- **Technical Details**: 연구에서, 저자는 트랜스포머 네트워크에서 내재적으로 저랭크 구조를 보이는 특정 네트워크 구간을 식별했습니다. 또한, 저랭크 근사화가 모델의 바이아스(Bias)에 미치는 영향을 탐구했습니다. 이를 통해, 저랭크 근사가 성능 향상뿐만 아니라 모델의 편향을 바로잡는 수단이 될 수 있음을 제시했습니다. 본 연구에 사용된 코드는 GitHub에서 확인 가능합니다.

- **Performance Highlights**: 서처지컬 디컴포지션(Surgical Decomposition)이 압축과 언어 모델링 성능의 트레이드오프를 명확하게 밝혀주었으며, 경우에 따라 일반 상식 추론 성능도 향상되었습니다. 이는 모델의 특정 부분이 저랭크 형태로 구조화되어 있음과 관련이 있습니다.



### Enhancing Dialogue State Tracking Models through LLM-backed User-Agents Simulation (https://arxiv.org/abs/2405.13037)
- **What's New**: 이번 연구는 대화 상태 추적(Dialogue State Tracking, DST) 작업을 경제적으로 수행하기 위해 LLMs(large language models)를 사용하는 방법을 제안합니다. GPT-4를 활용해 사용자와 에이전트 간의 상호작용을 시뮬레이트하여 DST 라벨이 있는 대화를 생성하고, 이를 통해 대화 수집 및 주석 작업의 비용을 줄이는 방안을 탐구했습니다.

- **Technical Details**: 먼저, GPT-4를 사용하여 수천 개의 DST 라벨이 첨부된 대화를 생성한 후, 생성된 데이터와 실제 데이터를 기반으로 LLaMA 2에 대해 두 단계의 파인튜닝(fine-tuning)을 수행했습니다. 이 과정에서 생성된 데이터와 실제 데이터를 결합하여 DST 예측 성능을 향상시켰습니다.

- **Performance Highlights**: 두 개의 공개 DST 벤치마크 테스트 결과, 생성된 대화 데이터를 활용한 모델이 실제 데이터만으로 훈련된 모델보다 더 나은 성능을 보였습니다. 추가적으로, 본 접근법은 실제 시나리오의 동적 요구에 빠르게 적응하여 새로운 도메인에서도 신속하게 대화를 생성할 수 있습니다. 어떤 도메인에서도 대응 가능한 대화 세그먼트를 생성해 대체함으로써, 실제 데이터로만 훈련된 모델과 비슷한 성능을 달성할 수 있었습니다.



### Can formal argumentative reasoning enhance LLMs performances? (https://arxiv.org/abs/2405.13036)
- **What's New**: 최근 인공지능 연구에서는 자원 소모가 큰 트레이닝을 거치지 않고도 대규모 언어 모델(LLMs)의 성능을 향상시키기 위한 다양한 기법들이 제안되었습니다. 그러나, 이 중 컴퓨테이셔널 아귀멘테이션(computational argumentation)을 고려한 연구는 아직 없었습니다. 본 논문에서는 LLMs의 성능을 향상시키기 위한 컴퓨테이셔널 아귀멘테이션 의미론을 도입한 파이프라인(MQArgEng)을 제시하고, 이를 예비적으로 평가한 연구 결과를 소개합니다.

- **Technical Details**: 이 논문에서 제시한 파이프라인은 LLMs와 컴퓨테이셔널 아귀멘테이션을 통합하는 방식을 탐구합니다. 구체적으로는 에이전트 간의 상호 작용 중 발생할 수 있는 정보 충돌을 공식적으로 포착하는 기법을 사용하여 LLMs의 추론 및 대화 능력을 향상시키는 것을 목표로 합니다. 실험은 MT-Bench를 사용하여 수행되었으며, 이는 초기 개념 증명 및 실현 가능성 분석을 목적으로 하였습니다.

- **Performance Highlights**: 실험 결과, MQArgEng은 대부분의 토픽 카테고리에서 중간 수준의 성능 향상을 보였습니다. 이는 초기 결과로서 추가 연구의 가치가 있음을 시사합니다.



### Autonomous Workflow for Multimodal Fine-Grained Training Assistants Towards Mixed Reality (https://arxiv.org/abs/2405.13034)
Comments:
          Accepted by ACL 2024

- **What's New**: 이 연구는 AI 에이전트를 확장 현실(XR) 애플리케이션에 통합하여 세밀한 훈련을 수행하는 자동 워크플로우를 설계하였습니다. 특히, LEGO 브릭 조립을 위한 파일럿 XR 환경에서 멀티모달 세밀 훈련을 도와주는 어시스턴트를 선보였습니다.

- **Technical Details**: 세부적으로는 메모리(memory), 계획(planning), XR 도구와의 상호작용을 통합한 LLM을 사용하는 언어 에이전트와 시각-언어 에이전트를 설계하여 과거 경험을 바탕으로 의사결정을 내릴 수 있게 했습니다. 추가로, 자동으로 생성된 멀티모달 대화 데이터셋인 LEGO-MRTA를 도입했습니다. 이 데이터셋은 멀티모달 설명서, 대화, XR 응답 및 시각 질문 응답 등을 포함합니다.

- **Performance Highlights**: 제안된 데이터셋을 사용하여 튜닝 (fine-tuning) 여부에 따른 여러 오픈 소스 LLM들의 성능을 평가하였으며, 그 결과를 벤치마크로 제시했습니다. 이 워크플로우는 XR 환경에서 유저와 원활하게 상호작용할 수 있는 더 스마트한 어시스턴트를 개발하는 데 기여할 것으로 예상됩니다.



### Faithful Attention Explainer: Verbalizing Decisions Based on Discriminative Features (https://arxiv.org/abs/2405.13032)
- **What's New**: 최근 몇 년간 모델 설명 방법들이 사용자가 쉽게 이해할 수 있도록 모델의 결정을 신뢰성 있게 설명할 수 있도록 설계되었습니다. 이번 논문에서는 모델이 주목한 특징(features)에 대해 신뢰성 있는 텍스트 설명을 생성할 수 있는 프레임워크인 Faithful Attention Explainer (FAE)를 제안합니다. 이 프레임워크는 시각적 특징 맵(visual feature maps)을 사용하여 문장을 생성합니다.

- **Technical Details**: FAE는 주목(item)에 관한 설명 생성에서 중요한 역할을 하는 attention 모듈을 배치합니다. 이 모델은 특징과 단어 간의 연관성을 학습하여 새로운 방식의 attention 강제 모듈을 도입합니다. 이를 통해 모델이 결정을 내리는 과정에서 사용된 중요 특징을 신뢰성 있게 설명할 수 있습니다.

- **Performance Highlights**: 제안된 FAE 프레임워크는 두 개의 데이터셋(CUB와 ACT-X)에서 캡션 품질 지표와 신뢰성 있는 결정 관련 지표에 대해 유망한 성능을 보여주었습니다. 또한, 인간의 시선(gaze) 기반 주목(attention)을 해석할 수 있어 고급 인간-인공지능 상호작용에 잠재력을 보였습니다.



### A Robust Autoencoder Ensemble-Based Approach for Anomaly Detection in Tex (https://arxiv.org/abs/2405.13031)
Comments:
          Submitted to ECML/PKDD 2024

- **What's New**: 이 연구에서는 텍스트 말뭉치에서 이상 탐지를 해결하기 위해 견고한 오토인코더 앙상블 기반 접근 방식을 소개합니다. 이 접근법은 텍스트 데이터에서 독립적인 이상치와 상황적 이상치를 구분하는 포괄적인 현실 세계 분류 체계를 제안합니다. 이는 기존 문헌에서 중요한 격차를 해결하기 위해 설정되었습니다.

- **Technical Details**: 각 오토인코더는 인코딩 임베딩에서 원본 데이터의 로컬 견고한 부분 공간 복구 투영을 통합합니다. k-최근접 이웃(k-nearest neighbors)의 기하학적 특성을 활용하여 부분 공간 복구를 최적화하고 텍스트 데이터의 이상 패턴을 식별합니다. 이 접근 방식을 평가하기 위해 텍스트 이상 탐지 맥락에 맞는 실험 환경이 필요합니다.

- **Performance Highlights**: 고전적인 텍스트 말뭉치에서 광범위한 실험이 수행되었으며, 이 실험의 결과는 독립적 이상치와 상황적 이상치 탐지에 있어 견고성과 성능 양면에서 강력한 오토인코더 앙상블 기반 접근 방식의 효율성을 강조합니다. 이 실험에서는 분류(classification), 감정 분석(sentiment analysis), 스팸 감지(spam detection) 등 다양한 작업을 여덟 가지 다른 말뭉치를 대상으로 연구했습니다.



### Crowdsourcing with Enhanced Data Quality Assurance: An Efficient Approach to Mitigate Resource Scarcity Challenges in Training Large Language Models for Healthcar (https://arxiv.org/abs/2405.13030)
Comments:
          Published in AMIA Summit, Boston, 2024. this https URL

- **What's New**: 본 논문에서는 헬스케어 분야에서 Large Language Models (LLMs)의 성능을 향상시키기 위한 새로운 crowdsourcing (CS) 프레임워크를 제안합니다. 특히, 자원이 부족한 환경에서 고품질의 라벨링 데이터를 효율적으로 생성하는 방법을 중점적으로 다루고 있습니다.

- **Technical Details**: 제안된 프레임워크는 데이터 수집 단계에서 사전(pre-), 실시간(real-time-) 및 사후(post-) 품질 관리를 포함한 다양한 품질 통제 방법을 통합합니다. 해당 연구에서는 Bio-BERT 모델을 사용하여 자폐증 관련 증상을 예측하는 작업에서 데이터 품질 향상의 효과를 평가했습니다.

- **Performance Highlights**: 실험 결과, 실시간 품질 관리를 적용한 경우 사전 품질 관리만을 사용한 경우보다 데이터 품질이 19% 향상되는 것을 확인했습니다. Crowdsourced 데이터를 사용하여 Bio-BERT를 미세 조정(fine-tuning)한 결과, 리콜(recall)이 증가함에 따라 Bio-BERT의 기준 모델보다 성능이 향상되었으나 정밀도(precision)는 낮아졌습니다. 이러한 결과는 헬스케어 LLMs의 최적화를 통한 의사 결정의 유용성과 환자 치료의 개선 가능성을 시사합니다.



### DuetSim: Building User Simulator with Dual Large Language Models for Task-Oriented Dialogues (https://arxiv.org/abs/2405.13028)
Comments:
          Accepted by COLING 2024

- **What's New**: 이 논문은 DuetSim이라는 새로운 프레임워크를 소개합니다. 이는 기존의 인간-엔지니어링된 어젠다에 의존하는 전통적인 사용자 시뮬레이터의 한계를 극복하기 위해 설계되었습니다. 두 개의 대형 언어 모델(LLMs)을 동시에 사용하는 DuetSim은 응답 생성과 검증을 각각 담당하여 더욱 다양하고 정확한 응답을 생성합니다.

- **Technical Details**: DuetSim은 두 개의 LLMs을 사용합니다. 하나는 응답 생성(response generation)을 전담하고, 다른 하나는 생성된 응답을 검증(verification)하는 역할을 합니다. 이 이중 LLM 접근 방식은 보다 다양하면서도 정확한 응답을 생성할 수 있게 도와줍니다. 이를 통해 사용자가 목표에 도달하는데 유용한 대화를 이끌어낼 수 있습니다.

- **Performance Highlights**: DuetSim의 효능은 MultiWOZ 데이터세트를 사용한 광범위한 실험을 통해 검증되었습니다. 이 실험은 응답 품질과 정확성에서 큰 개선을 보여주었으며, 이러한 개선은 주로 두 번째 LLM의 도입 덕분입니다. DuetSim은 인간 사용자들로부터 선호되는 응답을 생성하는데 성공하였습니다.



### Leveraging Human Revisions for Improving Text-to-Layout Models (https://arxiv.org/abs/2405.13026)
- **What's New**: 이번 연구에서는 인간의 피드백을 활용한 강화 학습(발 전반적으로 향상된 정렬(alignment)을 달성하는 방법을 제안합니다. 기존의 연구는 주로 모델 출력 쌍 간의 선호도(preferences)와 같은 높은 수준의 라벨에 초점을 맞춘 반면, 본 연구는 인간의 세부적인 수정(revisions)을 통해 더욱 강력한 정렬을 이루는 방법을 탐구합니다.

- **Technical Details**: 우리의 방법인 Revision-Aware Reward Models ($
method$)는 대규모 모바일 화면 데이터셋으로 사전 학습(pretrained)된 생성(layout) 모델의 출력을 인간 디자이너가 수정하여 이를 학습데이터로 활용합니다. 이를 통해 인간 디자이너의 수정을 학습한 보상 모델(reward model)을 구축하고, Reinforcement Learning from Human Feedback(RLHF)를 통해 모델을 최적화합니다.

- **Performance Highlights**: 이 방법은 텍스트-투-레이아웃(text-to-layout) 생성 모델이 더 현대적이며 디자이너의 선호도에 맞는 레이아웃을 생성하도록 합니다. 인간의 수정과 강력한 피드백을 활용하는 잠재력 또한 보여줍니다.



### A survey on fairness of large language models in e-commerce: progress, application, and challeng (https://arxiv.org/abs/2405.13025)
Comments:
          21 pages, 9 figures

- **What's New**: 이 설문조사는 전자상거래 (e-commerce)에서 대형 언어 모델 (LLMs)의 공정성을 탐구하고, 이 모델들의 진보와 활용, 그리고 이들이 직면하는 도전을 다룹니다. LLMs는 전자상거래 도메인에서 혁신적인 솔루션을 제공하고 고객 경험을 향상시키는 데 중요한 역할을 하고 있습니다. 이 논문은 전자상거래에서 LLMs의 응용과 도전에 대한 포괄적인 조사 보고서를 제공합니다.

- **Technical Details**: 논문은 먼저 전자상거래에서 LLMs 사용의 주요 원칙을 소개하며, 특정 요구에 맞게 이러한 모델을 조정하는 '사전 훈련 (pretraining)', '파인튜닝 (fine-tuning)', '프롬프팅 (prompting)' 프로세스를 자세히 설명합니다. 나아가 LLMs의 다양한 응용 프로그램들을 조사합니다. 예를 들어, 제품 리뷰를 분석하고 고객 피드백을 합성하는 기능, 소비자 데이터를 활용하여 관련 항목을 추천하는 기능, 글로벌 접근성을 향상시키는 제품 정보 번역 기능, 고객 지원 자동화 기능을 하는 제품 질문 및 답변 섹션 등이 포함됩니다.

- **Performance Highlights**: 설문조사는 LLMs가 전자상거래에서 얼마나 효과적인지를 자세하게 설명하며, 또한 훈련 데이터와 알고리즘에 포함된 편향이 불공정한 결과를 초래할 수 있다는 공정성 문제를 비판적으로 다룹니다. 이러한 문제는 소비자 신뢰를 저해할 뿐만 아니라 윤리적 및 법적 문제를 야기합니다. 따라서, 논문은 더 공정하고 투명한 LLMs를 만들기 위한 미래 연구 방향을 제시하며, 이러한 시스템의 편향을 완화하고 공정성을 개선하려는 지속적인 노력을 강조합니다. 이를 통해 다양한 글로벌 시장을 효과적이고 윤리적으로 서비스할 수 있도록 하는 지침을 제공하고 있습니다.



### Intelligent Tutor: Leveraging ChatGPT and Microsoft Copilot Studio to Deliver a Generative AI Student Support and Feedback System within Teams (https://arxiv.org/abs/2405.13024)
- **What's New**: 본 연구는 ChatGPT API와 GPT-4 모델을 Microsoft Teams 플랫폼의 Microsoft Copilot Studio와 통합하여 지능형 과외 시스템을 개발하는 것을 탐구합니다. 이 시스템은 학생들에게 즉각적인 지원을 제공하며, 학습자의 진행 상황과 피드백에 따라 교육 내용을 동적으로 조정합니다.

- **Technical Details**: 자연어 처리(NLP)와 머신러닝(ML)의 발전을 활용하여 학생의 문의를 해석하고, 개인화된 피드백을 제공하며, 교육 여정을 지원합니다. 이 시스템은 학생의 학습 요청을 이해하고 맞춤형 피드백을 제공함으로써 학습 효과를 증대시킵니다.

- **Performance Highlights**: 초기 구현 결과, 이 시스템은 학생들의 동기부여와 몰입도를 높이는 잠재력을 보여주었으며, 교사들에게 학습 과정에 대한 중요한 통찰을 제공함으로써 맞춤형 교육 경험을 촉진하고 교수 효율성을 향상시키는 데 기여합니다.



### LLMs can learn self-restraint through iterative self-reflection (https://arxiv.org/abs/2405.13022)
- **What's New**: 이번 연구는 대형 언어 모델(Large Language Models, LLMs)이 특정 주제에 대한 지식 수준과 불확실성에 따라 동적으로 행동을 조정하는 능력을 갖추기 위한 방법을 제시합니다. 연구진은 '자제력(self-restraint)'이라는 용어를 사용하여 이러한 적응 행동을 설명하며, 이는 LLM의 내부 지식에 의존하는 복잡한 작업입니다.

- **Technical Details**: 기본적으로 LLM은 다음 토큰의 가능성을 최대화하도록 훈련되므로 불확실성 수준에 따라 답변을 조절하는 법을 배우지 못합니다. 이를 해결하기 위해 연구진은 모델이 자신 있는 응답만 생성하도록 유도하는 유틸리티 함수(utility function)를 고안했습니다. 이 함수는 다양한 길이의 응답과 회피(abstention)를 점수화하는 데 사용될 수 있습니다. 이를 최적화하기 위해 'ReSearch'라는 자기 반성 과정(self-reflection process)을 도입했습니다. 이 과정은 반복적인 자기 프롬프트(self-prompting) 및 자기 평가(self-evaluation)로 구성되어 있으며, 이를 통해 모델을 미세 조정할 합성 데이터를 생성합니다.

- **Performance Highlights**: ReSearch 알고리즘으로 생성된 모델은 원래 모델에 비해 환각(hallucinations)의 발생이 적어지며, 추가적인 추론 비용 없이도 알려진 주제와 미지의 주제 모두에서 더 나은 성능을 보입니다. 또한, 회피를 표현하는 답변을 추가하여 검색 절차 동안 생성된 샘플을 증강함으로써 회피 기능을 우아하게 통합했습니다.



### IM-RAG: Multi-Round Retrieval-Augmented Generation Through Learning Inner Monologues (https://arxiv.org/abs/2405.13021)
Comments:
          Proceedings of the 47th International ACM SIGIR 2024

- **What's New**: Retrieval-Augmented Generation (RAG) 패러다임에 유연성을 더하고 다중 라운드 검색 과정의 해석 가능성을 높이며 끝에서 끝까지 최적화가 이루어지는 IM-RAG라는 새로운 접근법을 제안했습니다. Inner Monologues (IM)을 통해 정보 검색 시스템(IR)을 대형 언어 모델과 통합하였습니다.

- **Technical Details**: IM-RAG는 학습된 Inner Monologues(인간의 내면 음성) 과정을 통해 IR 시스템과 LLM을 통합합니다. 이 과정에서 LLM은 추론 모델(Reasoner)로 사용되며, IR 모듈에서 정보를 수집하거나 대화 맥락에 기반하여 최종 답변을 제공합니다. Retriever에서 가져온 출력물을 개선하는 Refiner를 도입하여 Reasoner와 다양한 기능을 가진 IR 모듈 간의 격차를 메우고 다중 라운드 커뮤니케이션을 촉진합니다. 전체 IM 과정은 Reinforcement Learning(RL)을 통해 최적화되며, Progress Tracker를 포함하여 중간 단계에서 보상을 제공하고, 답변 예측은 Supervised Fine-Tuning(SFT)을 통해 추가로 최적화됩니다.

- **Performance Highlights**: HotPotQA 데이터셋을 사용한 광범위한 실험 결과, IM-RAG 접근법이 기존 최고의 성능(state-of-the-art, SOTA)을 달성했으며, IR 모듈 통합의 유연성과 학습된 내면 독백(Inner Monologues)에서 나타나는 강력한 해석 가능성을 제공합니다.



### Using Combinatorial Optimization to Design a High quality LLM Solution (https://arxiv.org/abs/2405.13020)
- **What's New**: 이번 논문에서는 조합 최적화(combinatorial optimization) 및 샘플링(sampling)을 활용한 새로운 대형 언어 모델(LLM) 기반 솔루션 설계 접근 방식을 소개합니다. 이 접근 방식은 프롬프트 유형(prompt types), LLM 입력 대안, 그리고 생성 및 설계 대안을 조절하는 매개변수 등 솔루션의 품질에 영향을 미치는 요소들을 식별하는 데 초점을 맞추고 있습니다. 이러한 요소들을 식별함으로써 전문 지식을 주입할 수 있는 기반을 마련합니다.

- **Technical Details**: 논문에서 제시된 접근 방식은 일련의 상호작용(interactions)을 정의하고 조합 최적화를 활용하여 원하는 모든 상호작용이 포함된 작은 부분집합 $P$를 생성합니다. 그런 다음 각 요소 $p \\in P$를 적절한 벤치마크로 발전시킵니다. 이러한 대체 솔루션을 각 조합에 적용하고 결과를 평가함으로써 품질 높은 LLM 솔루션 파이프라인을 설계할 수 있습니다. 특히 이 접근 방식은 각 벤치마크의 설계와 평가가 시간이 많이 소요되고 수작업 및 인간 평가가 필요한 경우에 특히 유용합니다.

- **Performance Highlights**: 이 접근 방식은 자동화된 머신러닝(autoML) 접근 방식과 비교 및 검증하는 데 효율적인 기준점으로 사용할 수 있습니다. 솔루션 품질을 결정하는 요소들을 탐색하는 autoML 접근 방식을 평가하는 데에도 적합합니다.



### A Comprehensive Survey of Accelerated Generation Techniques in Large Language Models (https://arxiv.org/abs/2405.13019)
- **What's New**: 자동회귀 언어 모델 (autoregressive language models)에서 텍스트 생성을 가속화하는 최신 기술들을 종합적으로 조사한 논문이 발표되었습니다. 이 연구는 효율적인 콘텐츠 생산을 위해 텍스트 생성속도를 높이는 것이 얼마나 중요한지 강조하며, 실시간 응용 프로그램에 대한 높은 추론 지연 문제를 해결하기 위해 다양한 기법들을 탐구합니다.

- **Technical Details**: 이 논문에서는 텍스트 생성 가속화 기법을 크게 세 가지 주요 영역으로 분류합니다: 예측적 디코딩 (speculative decoding), 조기 종료 메커니즘 (early exiting mechanisms) 및 비자동회귀 방법 (non-autoregressive methods). 각 배열의 기본 원리, 장점, 한계 및 최근 발전 사항에 대해 논의합니다. 이를 통해 현재 가장 현대적인 기술들과 그 응용에 대한 이해를 돕고, 향후 연구 방향을 제시합니다.

- **Performance Highlights**: 논문은 각각의 기법들이 어떻게 성능을 개선할 수 있는지에 대한 구체적인 사례와 평가를 포함하여, 현재 사용 가능한 최고의 방법들을 비교 분석합니다. 이러한 종합적인 조사는 실시간 응용에서의 높은 추론 지연 문제를 해결하기 위해 텍스트 생성속도를 개선하는 데 중요한 통찰력을 제공합니다.



### Continued Pretraining for Domain Adaptation of Wav2vec2.0 in Automatic Speech Recognition for Elementary Math Classroom Settings (https://arxiv.org/abs/2405.13018)
- **What's New**: 이 연구에서는 Automatic Speech Recognition (ASR) 시스템이 교실 환경에서 어떻게 더욱 견고하고 탄력적일 수 있는지 탐구합니다. Wav2vec2.0 모델을 교실 도메인에 적응시키기 위해 'Continued Pretraining (CPT)'의 효율성을 조사하였습니다.

- **Technical Details**: CPT를 통해 Wav2vec2.0 모델을 다양한 소음, 마이크, 교실 환경 및 인구 통계에 더 강하게 만들 수 있음을 증명하였습니다. 특히, CPT는 라벨링된 세부 조정 데이터(finetuning data)에서 보지 못한 다양한 인구 통계에도 모델이 잘 일반화되는 능력을 보여줍니다.

- **Performance Highlights**: CPT는 Wav2vec2.0 기반 모델의 단어 오류율(Word Error Rate, WER)을 10% 이상 감소시켰습니다. 이를 통해 교실 조건에서의 ASR 성능이 현저히 향상되었습니다.



### A Systematic Analysis on the Temporal Generalization of Language Models in Social Media (https://arxiv.org/abs/2405.13017)
- **What's New**: 이 논문은 소셜 미디어에서의 시간 변화(temporal shift)에 중점을 두고 있습니다. 특히 트위터(Twitter)에서의 변화에 초점을 맞추어, 시간 변화 하에서 언어 모델(LMs)의 성능을 평가하는 통합된 평가 방식을 제안합니다. 표준 소셜 미디어 NLP 작업에서 시간 변화에 따른 언어 모델의 성능을 평가했습니다.

- **Technical Details**: 언어 모델은 다섯 가지 다양한 소셜 미디어 NLP 작업에서 다양한 시간 설정 하에 테스트되었습니다. 이번 연구는 엔터티 중심 작업(named entity recognition, named entity disambiguation, hate speech detection)에서 시간 변화에 따라 성능이 일관되게 감소하지만, 주제나 감정 분류와 같은 다른 작업에서는 그 변화가 크지 않음을 밝혀냈습니다. 또한, 테스트 기간 동안 연속적 사전 학습(continuous pre-training)을 수행해도 언어 모델의 시간 적응성이 향상되지 않는다는 결과를 보였습니다.

- **Performance Highlights**: 시간 변화 하에서 언어 모델의 성능 감소는 엔터티 중심 작업에서 일관되게 나타났습니다. 반면 주제(topic) 및 감정(sentiment) 분류 작업에서는 성능 감소가 두드러지지 않았습니다. 또한, 연속적 사전 학습은 시간 적응성을 개선하지 못했습니다.



### The Evolution of Darija Open Dataset: Introducing Version 2 (https://arxiv.org/abs/2405.13016)
- **What's New**: Darija Open Dataset(DODa)는 모로코 방언인 다리야(Darija)에 대한 자연어 처리 역량을 향상시키기 위한 오픈 소스 프로젝트입니다. DODa는 약 100,000개의 항목을 보유하고 있으며, 다리야-영어 번역을 위한 가장 큰 협력 프로젝트로 자리매김하고 있습니다.

- **Technical Details**: 이 데이터셋은 의미적 및 구문적(Semantic and Syntactic) 분류, 철자 변형, 다양한 시제의 동사 활용 등을 포함하고 있습니다. 또한 라틴 문자와 아랍 문자로 작성된 항목들을 포함하고 있어 다양한 출처와 응용 프로그램에서 발견되는 언어적 변이와 선호도를 반영하고 있습니다.

- **Performance Highlights**: DODa는 수만 개의 번역된 문장을 특징으로 하며, 모로코 커뮤니티의 언어적 요구를 정확하게 이해하고 생성할 수 있는 애플리케이션 개발에 필수적입니다. 이를 통해 인접 지역의 유사한 방언에도 적용될 수 있는 가능성을 제공합니다.



### Assisted Debate Builder with Large Language Models (https://arxiv.org/abs/2405.13015)
Comments:
          7 pages, 2 figures

- **What's New**: 우리는 새로운 도구인 ADBL2를 소개합니다. 이 도구는 대규모 언어 모델(large language models)의 일반화 및 관계 기반(argument mining) 논증 마이닝 역량을 바탕으로 다양한 도메인에서 작동합니다. ADBL2는 (1) 토론에서 사전 설정된 관계의 검증 및 (2) 대규모 언어 모델을 통한 새로운 논증 창작을 지원하는 최초의 오픈 소스 도구입니다.

- **Technical Details**: ADBL2는 매우 모듈화되어 있으며, 플러그인으로 사용되는 모든 오픈 소스 대규모 언어 모델과 함께 작동할 수 있습니다. 또한, 우리는 ADBL2가 사용 가능한 관계 기반 논증 마이닝을 위해 Mistral-7B 대규모 언어 모델을 처음으로 미세 조정(fine-tuned)하여 제공합니다.

- **Performance Highlights**: 미세 조정된 Mistral-7B 모델은 모든 도메인에서 90.59%의 전체 F1-스코어를 기록하며 기존 접근 방식을 능가합니다.



### QCRD: Quality-guided Contrastive Rationale Distillation for Large Language Models (https://arxiv.org/abs/2405.13014)
- **What's New**: 이번 연구는 대형 언어 모델(Large Language Models, LLMs)의 자원 한계와 추론 효율성 문제를 해결하고자 새로운 접근 방식을 제안합니다. 주로 긍정적 지식만을 강조했던 이전 연구와 달리, 이번 연구는 지식 노이즈와 부정적 지식의 탐구도 고려한 '품질 가이드 대조 논거 증류법(Quality-guided Contrastive Rationale Distillation)'을 도입했습니다. 이는 대조 학습(Contrastive Learning) 관점에서 추론 능력 학습을 목표로 합니다.

- **Technical Details**: 긍정적 지식 학습을 위해, 자기 일관성(Self-consistency)을 활용하여 온도 샘플링(Temperature Sampling)으로 생성된 대형 언어 모델의 논거를 노이즈 제거합니다. 부정적 지식 증류를 위해, 작은 언어 모델들이 이전 시행에서 자체적으로 생성한 부정적 논거를 온도 샘플링을 통해 생성합니다. 마지막으로, 대조 손실(Contrastive Loss)을 설계하여 긍정 및 부정 논거를 작은 언어 모델에 보다 잘 증류하고, 온라인 업데이트 판별기(Online-update Discriminator)를 사용해 논거의 품질을 판단하고 가중치를 할당합니다.

- **Performance Highlights**: 다양한 추론 작업에서 광범위한 실험을 통해, 이번 연구 방법이 이전 증류 방법들을 일관되게 능가하며 더 높은 품질의 논거를 생성하는 것을 확인했습니다.



### Amplifying Aspect-Sentence Awareness: A Novel Approach for Aspect-Based Sentiment Analysis (https://arxiv.org/abs/2405.13013)
Comments:
          24 pages, 4 figures, 4 tables

- **What's New**: 이번 연구에서는 Aspect-Based Sentiment Analysis(ABSA)에 새로운 접근법인 Amplifying Aspect-Sentence Awareness(A3SN)을 제안합니다. 기존의 attention 기반 모델들이 문맥과 측면(aspect) 간의 연결을 잘 하지 못하는 문제를 해결하기 위해 aspect-sentence awareness를 증폭하는 방식을 도입했습니다.

- **Technical Details**: A3SN은 transformer의 표준 과정을 따르면서 멀티-헤드 어텐션 메커니즘(multi-head attention mechanisms)을 통해 문장과 측면(aspect)의 의미 정보를 증폭합니다. 추가적으로, aspect-sentence awareness attention을 강화하기 위한 또 다른 멀티-헤드 어텐션 모듈을 도입하여 문맥 내에서 측면의 중요성을 두 배로 강조했습니다. 이는 미묘한 관계와 의존성을 정확하게 포착할 수 있게 합니다. 또한, 게이트 융합(gated fusion)은 멀티-헤드와 증폭된 aspect-sentence awareness attention 메커니즘의 특징 표현을 통합하는데 필수적입니다.

- **Performance Highlights**: 세 가지 벤치마크 데이터셋을 대상으로 한 실험 결과, A3SN은 기존 최신(state-of-the-art, SOTA) 모델들을 능가하는 성능을 보였습니다.



### Divergent Creativity in Humans and Large Language Models (https://arxiv.org/abs/2405.13012)
Comments:
          First two and last listed authors are corresponding authors. The first two listed authors contributed equally to this work

- **What's New**: 최근 대형 언어 모델(LLMs)의 능력이 크게 향상되면서 이들이 인간처럼 창의적인 수준에 도달하고 있다는 주장들이 나오고 있습니다. 이 논문은 LLM의 창의성을 체계적으로 평가하고, 이를 인간의 발산적 사고(divergent thinking)와 비교하는 것이 중요하다는 점을 강조합니다. 이를 위해 연구진은 창의성 과학의 최신 발전을 활용하여 최첨단 LLM과 100,000명의 인간 데이터를 포함한 광범위한 데이터셋을 분석하는 프레임워크를 구축했습니다.

- **Technical Details**: 연구는 창의력 과학(creativity science)의 최신 진전을 활용해 발산적 창의성(divergent creativity)을 심층적으로 분석하는 프레임워크를 개발했습니다. 이 프레임워크는 특정 창의적 과제에서 LLM이 인간을 능가하는 증거를 제시하며 이를 양적 벤치마킹(quantitative benchmarking)을 통해 평가했습니다. 특히 발산적 연관(association)과 창의적 글쓰기(creative writing)에서의 성과가 두드러졌습니다.

- **Performance Highlights**: 연구 결과, 특정 창의적 작업에서 LLM이 실제로 인간 능력을 능가할 수 있다는 증거를 발견했습니다. 이를 통해 더욱 창의적인 LLM 개발의 새로운 경로가 열렸지만, 동시에 인간의 창의적 사고 과정의 독특한 요소를 인공지능으로 인공적으로 생성할 수 있는지에 대한 세부적인 질문들도 제기되었습니다.



### Unveiling Social Media Comments with a Novel Named Entity Recognition System for Identity Groups (https://arxiv.org/abs/2405.13011)
- **What's New**: 이번 연구에서는 기존의 증오 발화(hate speech) 탐지 방법을 확장하여 새로운 정체성 그룹 명명 엔티티 인식(Named Entity Recognition, NER) 시스템을 개발했습니다. 이를 위해 새로운 데이터셋을 구축하여 전통적인 NER 시스템을 정체성 그룹 인식으로 확장할 수 있도록 했습니다.

- **Technical Details**: 연구팀은 주로 텍스트 분류기(text classifier)에 기반한 기존의 증오 발화 탐지 방법을 개선하여, 특정 문장이 공격을 포함하는지 여부를 감지하는 동시에 문장 내 토큰을 해당 그룹과 관련된 것으로 태그하는 기능을 추가했습니다. 이를 위해 정체성 그룹을 인식할 수 있도록 기존의 NER을 확장하는 데이터셋을 구축했습니다.

- **Performance Highlights**: 모델은 평균 f1-스코어 0.75로 그룹 인식에 경쟁력 있는 성능을 보였습니다. 특히 인종 공격(span)을 인식하는 데 있어 다른 정체성 그룹에 비해 높은 f1-스코어 0.80을 기록했습니다. 성적 지향 및 성별과 관련된 소수자 클래스에 대한 일반화 능력 역시 뛰어나 f1-스코어가 각각 0.77 및 0.72였습니다. 연구팀은 소셜 미디어의 실제 사례 연구를 통해 Facebook 댓글에 대한 주석을 달아 비교했으며, 분석된 뉴스 기사 카테고리와 관련된 명명 엔티티를 효과적으로 탐지했습니다. 카테고리 간 오류율은 미미했습니다.



### UCCIX: Irish-eXcellence Large Language Mod (https://arxiv.org/abs/2405.13010)
- **What's New**: 이번 연구는 UCCIX라는 혁신적인 오픈 소스 아일랜드어 기반 대규모 언어 모델(LLM)의 개발을 소개합니다. 이는 매우 낮은 자원 언어(extremely low-resource languages)로 남아 있던 아일랜드어를 대표하는 첫 번째 시도입니다.

- **Technical Details**: Llama 2-13B 모델을 기반으로 한 이 연구는 기존의 LLM 훈련에 필요한 텍스트 데이터의 일부만으로 아일랜드어에 최적화된 새로운 프레임워크를 제안합니다. 이 프레임워크는 스케일링 법칙(scaling laws)에 따르면 대규모 데이터가 필요하지만, 우리는 그보다 훨씬 적은 데이터로도 성능을 발휘할 수 있음을 증명하였습니다.

- **Performance Highlights**: 제안된 모델은 아일랜드어 작업에서 더 큰 모델들을 능가하며 최대 12%의 성능 향상을 보여주었습니다. 또한 아일랜드어 질문응답 데이터셋(IrishQA)과 MT-bench의 아일랜드어 버전을 포함한 종합적인 벤치마킹 데이터셋을 제공하여 향후 연구를 지원합니다.



### METAREFLECTION: Learning Instructions for Language Agents using Past Reflections (https://arxiv.org/abs/2405.13009)
- **What's New**: 이번 연구에서는 METAREFLECTION이라는 새로운 기술을 소개합니다. 이 기술은 LLM(Large Language Models)에서 특정 도메인(domain)에 대해 일반적인 프롬프트 지시사항을 학습하는 방법을 제안합니다. 이는 훈련 단계에서 수집한 자체 반영(self-reflections)을 기반으로 합니다.

- **Technical Details**: METAREFLECTION 기술은 모델이 생성한 언어적 피드백을 바탕으로 합니다. 구체적으로, 인프라 코드 취약점 감지(Infrastructure as Code vulnerability detection)와 질의 응답(Question-Answering) 도메인에서 평가되었습니다. 또한 REACT와 COT라는 두 가지 방법론을 사용하여 두 도메인의 성능을 개선했습니다.

- **Performance Highlights**: METAREFLECTION은 GPT-4에 비해 상당한 성능 향상을 보였습니다. 인프라 코드 취약점 감지 도메인에서는 16.82%, COT에서는 31.33%, 그리고 REACT에서는 15.42%의 향상을 나타냈습니다. 이를 통해 METAREFLECTION 기술이 LLM의 효율성을 크게 향상시킬 수 있는 잠재력을 확인할 수 있습니다.



### Control Token with Dense Passage Retrieva (https://arxiv.org/abs/2405.13008)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)에서 발생하는 환각 문제를 해결하려는 시도를 다룹니다. 이를 위해 Retrieval-Augmented Generation(RAG) 기법을 사용하여 프롬프트에 관련 정보를 포함시켜 정확한 답변을 얻고자 했습니다. 하지만 RAG 역시 올바른 정보를 검색하는 데 고유의 문제를 겪었습니다. 이를 해결하기 위해 사용자의 쿼리와 관련된 도메인 특화 문서를 검색하기 위해 Dense Passage Retrieval(DPR) 모델을 사용했습니다. 비록 이 DPR 모델 역시 문서 검색의 정확성이 부족했지만, 제어 토큰(control tokens)을 통합함으로써 DPR 모델의 성능을 크게 향상시켰습니다.

- **Technical Details**: RAG는 프롬프트에 관련 정보를 포함시켜 더 정확한 답변을 생성하도록 하는 기술입니다. 하지만 이 기법은 관련 정보를 정확하게 검색하지 못하는 문제를 가지고 있었습니다. DPR 모델은 사용자의 쿼리와 관련된 도메인 특화 문서를 검색하는 데 사용되었지만, 여전히 문서 검색의 정확성이 부족했습니다. 이에 우리는 제어 토큰을 통합하여 DPR 모델을 개선했습니다.

- **Performance Highlights**: 개선된 DPR 모델은 표준 DPR 모델에 비해 성능이 뛰어난 결과를 보여주었습니다. 특히, Top-1 정확도가 13% 향상되었고, Top-20 정확도는 4% 증가했습니다.



### News Recommendation with Category Description by a Large Language Mod (https://arxiv.org/abs/2405.13007)
Comments:
          5 pages, 5 figures

- **What's New**: 개인 맞춤형 뉴스 추천을 위해 새로운 방법을 제안합니다. 대형 언어 모델(LLM)을 활용하여 자동으로 정보가 풍부한 카테고리 설명을 생성하고, 이를 추천 모델에 추가적인 정보로 통합하는 것이 특징입니다. 이는 수작업이나 특정 도메인 지식 없이 수행됩니다.

- **Technical Details**: 이 방법은 텍스트, 카테고리, 이미지 등 뉴스 콘텐츠의 특징을 적절히 인코딩하는 것을 중심으로 합니다. 특히, 뉴스 카테고리(tv-golden-globe, finance-real-estate, news-politics 등)는 뉴스를 이해하는 데 중요한 역할을 합니다. 본 연구에서는 MIND 데이터셋을 사용하여 실험을 진행하였습니다. NAML, NRMS, NPA와 같은 최신 콘텐츠 기반 추천 모델에 대해 LLM이 생성한 카테고리 설명을 추가하여 성능을 비교했습니다.

- **Performance Highlights**: 우리의 방법은 AUC에서 5.8%까지 향상된 성과를 보여줬으며, 이는 LLM이 생성한 카테고리 설명이 없는 기존의 베이스라인 접근법과 비교하여 유의미한 개선을 나타냅니다. 이는 접근 방법의 효과성을 검증하는 결과입니다.



### Auto FAQ Generation (https://arxiv.org/abs/2405.13006)
Comments:
          3 figures and peer evaluated

- **What's New**: 새로운 시스템이 거대한 텍스트 문서에서 중요한 질문과 답변을 추출하여 FAQ 문서를 생성하는 방법을 제안하였습니다. 이 시스템은 특히 스탠포드 철학 백과사전의 문서를 활용하여 텍스트 요약, TextRank 알고리즘을 통한 문장 순위 매기기, 질문 생성 도구를 사용하여 초기 질문과 답변 세트를 생성합니다.

- **Technical Details**: 이 시스템은 **text summarization(텍스트 요약)**, **TextRank 알고리즘(알고리즘)**을 사용한 문장 순위 매기기, 그리고 **question-generation tools(질문 생성 도구)**를 결합합니다. 이 과정에서 생성된 일부 부적절한 질문을 걸러내기 위해 휴리스틱을 적용합니다.

- **Performance Highlights**: 생성된 질문을 문법, 질문의 의미, 요약된 문맥 내에서 답변 가능성 등의 항목으로 평가한 결과, 평균적으로 71%의 질문이 의미 있다고 평가되었습니다.



### Understanding the Rare Inflammatory Disease Using Large Language Models and Social Media Data (https://arxiv.org/abs/2405.13005)
- **What's New**: 이 연구는 대형 언어 모델(LLM)을 사용해 Reddit에서 사르코이드증에 관한 논의를 분석한 최초의 시도입니다. 이 연구는 LLM의 효과성을 입증하며 사르코이드증 관련 정보를 정확히 식별합니다.

- **Technical Details**: LLM을 사용하여 소셜 미디어 데이터에서 사르코이드증 관련 증상, 치료 방법, 예후 등을 분석했습니다. 또한, 무감독 클러스터링(unsupervised clustering)을 통해 세 가지 명확한 환자 하위 그룹(phenotypes)을 식별했습니다.

- **Performance Highlights**: 분석 결과, 피로, 림프절 부종, 호흡 곤란이 가장 흔한 증상으로 보고되었습니다. 프레드니존(prednisone)이 가장 많이 처방된 약물이었으며 인플릭시맙(infliximab)이 예후 개선에 가장 효과적이었습니다. 여성과 젊은 환자에서 예후 차이가 두드러졌으며, 감정 분석 결과 진단 후 정신 건강에 중간정도의 부정적 영향을 미치는 것으로 나타났습니다.



### MathDivide: Improved mathematical reasoning by large language models (https://arxiv.org/abs/2405.13004)
Comments:
          10 pages, 3 figures

- **What's New**: 이번 논문에서는 수학 문제를 더 작은 하위 문제로 나누는 'MathDivide'라는 프롬프팅 기법을 제안했습니다. 이러한 접근법을 통해 복잡한 수학 문제를 해결할 때 논리적 추론 능력을 증진시킬 수 있습니다.

- **Technical Details**: 'MathDivide'는 수학 문제를 더 단순한 하위 문제로 나누어 이를 각각의 대수 표현(algebraic expression)으로 공식화합니다. 이후 LLM이 생성한 Python 코드로 해당 대수 표현을 평가합니다. 문제 진술에 제공된 수치 값은 Python 코드로 전달되며, 하위 문제의 해결책을 종합하여 최종 답을 도출합니다. 최종 답과 정답을 비교하고, 만약 일치하지 않으면 LLM에게 정돈된 프롬프트를 다시 제공합니다.

- **Performance Highlights**: MathDivide는 GSM8K 데이터셋을 사용하여 폐쇄형 LLM 모델과 오픈 소스 LLM 모델 모두에서 실험을 진행했으며, 기존의 Math-prompter 프롬프팅 기법을 상당히 능가하는 성능을 보여주었습니다.



### A Survey on Recent Advances in Conversational Data Generation (https://arxiv.org/abs/2405.13003)
- **What's New**: 최근 대화 시스템 (conversational systems)의 발전은 다양한 분야에서 인간-기계 상호작용을 크게 향상시켰습니다. 하지만, 이러한 시스템을 훈련시키는 것은 전문 대화 데이터의 부족으로 인해 어려운 작업이 되었습니다. 본 조사에서는 다중턴 대화 데이터 생성 (multi-turn conversational data generation)에 대한 체계적이고 포괄적인 리뷰를 제공합니다. 주로 다루는 시스템 종류는 오픈 도메인 (open domain), 태스크 지향 (task-oriented), 정보 탐색형 (information-seeking)입니다.

- **Technical Details**: 기존 연구를 시드 데이터 생성 (seed data creation), 발화 생성 (utterance generation), 품질 필터링 (quality filtering) 방법 등에 기초하여 분류하고, 대화 데이터 생성 시스템의 주요 원칙을 설명하는 일반적인 프레임워크를 소개합니다. 또한, 합성 대화 데이터를 평가하기 위한 메트릭스와 방법들을 검토하고, 현재 분야에서의 도전 과제와 미래 연구 방향을 탐구합니다.

- **Performance Highlights**: 이 리뷰의 목표는 최첨단 방법들에 대한 개요를 제공하고, 연구자와 실무자들이 이 분야에서 더 나아갈 수 있는 기회를 강조함으로써 연구의 진전을 가속화하는 것입니다. 특히, 데이터의 효율적이고 확장 가능한 생성 방법을 제시함으로써, 기존 크라우드소싱 방식의 한계를 극복하고 데이터 생성 비용을 절감할 수 있습니다.



### DuetRAG: Collaborative Retrieval-Augmented Generation (https://arxiv.org/abs/2405.13002)
Comments:
          5 pages

- **What's New**: 새로운 연구는 복잡한 도메인 질문에서의 사실 오류를 줄이기 위해 지식 검색 기능을 보완한 'Collaborative Retrieval-Augmented Generation' 프레임워크인 DuetRAG을 제안했습니다. 이는 특히 HotPot QA와 같은 복잡한 도메인 질문에서 자주 발생하는 관련 없는 지식 검색 문제를 해결하기 위한 것입니다.

- **Technical Details**: DuetRAG 프레임워크는 도메인 세부조정(domain fine-tuning)과 Retrieval-Augmented Generation(RAG) 모델을 동시에 통합하는 부트스트래핑(bootstrapping) 접근 방식을 사용합니다. 이 접근 방식은 지식 검색 품질을 향상시켜 궁극적으로 생성 품질을 개선합니다.

- **Performance Highlights**: DuetRAG는 도메인 전문가들과의 비교에서 HotPot QA와 같은 복잡한 질문에서 유사한 성능을 보이며, 제안된 모델의 우수성을 입증했습니다.



### Large Language Models for Education: A Survey (https://arxiv.org/abs/2405.13001)
Comments:
          Journal of Machine Learning and Cybernetics. 4 tables, 6 figures

- **What's New**: 최신 연구에서 인공지능(AI)이 전통적인 교육에 미치는 깊은 영향을 탐구하였습니다. 특히, 대형 언어 모델(LLMs)이 자연어 처리, 컴퓨터 비전, 음성 인식, 자율 주행 등 다양한 응용 분야에서 사용되고 있으며, 교육, 정부, 금융, 법률 등 여러 분야에도 적용되고 있습니다.

- **Technical Details**: LLMs는 딥 러닝(deep learning), 사전 훈련(pre-training), 미세 조정(fine-tuning), 강화 학습(reinforcement learning) 등의 다양한 기술을 포함하고 있습니다. LLMEdu라고 불리는 스마트 교육을 위해 LLMs를 사용하는 것은 전 세계적으로 중요한 전략적 방향으로 자리잡고 있습니다.

- **Performance Highlights**:  연구는 현재 LLMEdu의 상태를 요약하고, LLMs와 교육의 특성, 그리고 교육에 LLMs를 통합했을 때의 이점을 소개합니다. 또한, 교육 산업에 LLMs를 통합하는 과정과 관련 기술들을 검토하며, LLMEdu가 직면한 도전과 문제점, 그리고 미래 최적화에 대한 전망을 논의합니다.



### RAGE Against the Machine: Retrieval-Augmented LLM Explanations (https://arxiv.org/abs/2405.13000)
Comments:
          Accepted by ICDE 2024 (Demonstration Track)

- **What's New**: 이 논문은 외부 소스를 질의하고 관련 정보를 입력 컨텍스트에 추가할 수 있는 검색 기능이 추가된 대형 언어 모델(LLM, Large Language Models)을 설명하는 상호작용 도구인 RAGE를 소개합니다. RAGE는 반사실적(counterfactual) 설명을 통해 입력 컨텍스트의 어떤 부분이 그것을 제거했을 때 모델의 응답에 변화를 일으키는지 식별합니다.

- **Technical Details**: RAGE는 입력 컨텍스트의 가능한 설명 공간을 탐색하기 위해 가지치기(pruning) 방법을 포함하고 있습니다. 이러한 방법들을 통해 사용자는 생성된 답변의 출처를 볼 수 있습니다. 이는 사용자가 모델의 응답에 영향을 미치는 요소를 구체적으로 파악할 수 있게 합니다.

- **Performance Highlights**: RAGE의 중요한 특징은 반사실적 설명 방식을 사용하여 LLM의 응답 생성 과정에서 입력의 어떤 부분이 중요한지 명확히 보여주는 것입니다. 이로 인해 모델이 출력하는 정보의 신뢰성을 평가하고 디버깅하는 데 도움이 될 수 있습니다.



### An Assessment of Model-On-Model Deception (https://arxiv.org/abs/2405.12999)
Comments:
          Accepted at Secure and Trustworthy Large Language Models Workshop at ICLR 2024

- **What's New**: 이 논문은 언어 모델의 신뢰성을 저해하는 기만적 출력(deceptive outputs)에 대한 연구를 소개합니다. 모델이 잘못된 답변을 정당화하는 방식을 바탕으로 데이터셋을 구축하고, 이를 통해 복잡한 모델 대 모델(model-on-model) 기만 시나리오를 조사합니다.

- **Technical Details**: LLama-2 7B, 13B, 70B 및 GPT-3.5를 사용해 약 10,000개의 오해를 불러일으키는 설명(misleading explanations)으로 구성된 데이터셋을 만들어 사용하였습니다. 이 데이터셋은 MMLU의 질문에 대한 잘못된 답변을 정당화하도록 모델들에게 요청함으로써 만들어졌습니다. 실험 결과, 모델들이 이 설명을 읽을 때 모두 크게 속게 되는 것으로 나타났습니다.

- **Performance Highlights**: 능력이 뛰어난 모델일지라도 속임수를 저항하는 능력은 크게 차이나지 않으며, 모든 능력의 모델들이 다른 모델을 오도하는 데 성공적이었습니다. 이는 더 높은 성능의 모델이 약간 더 속임수에 저항할 수 있을 뿐이라는 결과를 보여줍니다.



### A Textbook Remedy for Domain Shifts: Knowledge Priors for Medical Image Analysis (https://arxiv.org/abs/2405.14839)
Comments:
          23 pages, 9 figures, 12 tables, project page: this https URL

- **What's New**: 이 연구에서는 의료 스캔 분석에서 예기치 못한 상황에서 종종 실패하는 기존의 딥 네트워크 문제점을 조사하고 있습니다. 특히, 다른 병원에서 샘플링된 데이터나 성별, 인종 등의 인구 통계 변수에 의해 혼란스러운 데이터에 대한 민감도를 분석합니다. 본 연구는 'Knowledge-enhanced Bottlenecks (KnoBo)'라는 새로운 접근 방식을 제안하여, 명시적 의료 지식(예: PubMed)의 기반이 되는 사전 지식을 딥 네트워크에 포함시킵니다.

- **Technical Details**: 이 논문은 KnoBo라는 개념 병목 모델을 소개합니다. KnoBo는 의료 교과서나 PubMed에서 찾을 수 있는 임상적으로 관련된 요소들을 추론하는 데 필요한 지식 사전을 통합하여 구성됩니다. 이러한 개념 공간을 설계하기 위해 검색 보강 언어 모델(예: retrieval-augmented language models)이 사용되며, 개념을 자동으로 인식하는 절차와 짝을 이룹니다. 20개의 데이터셋에서 다양한 도메인 이동(domain shift)을 평가하였습니다.

- **Performance Highlights**: 두 가지 영상 형식에 대해 KnoBo는 기존의 미세 조정된 모델보다 평균 32.4% 더 나은 성능을 보였습니다. 평가 결과, PubMed가 최종 예측 성능과 정보의 다양성 측면 모두에서 도메인 이동에 덜 민감한 의료 모델을 만드는 데 유망한 자원임을 나타냈습니다.



### Pragmatic Feature Preferences: Learning Reward-Relevant Preferences from Human Inpu (https://arxiv.org/abs/2405.14769)
Comments:
          ICML 2024

- **What's New**: 이 논문에서는 인간의 사회적 학습 방식을 모방하여 보상 모델을 더 정확하게 학습하는 방법을 제안합니다. 구체적으로, 기존의 이진 선호도 쿼리(binary preference queries)에 더해, 예제의 어떤 특징이 선호되는지를 묻는 접근 방식을 도입합니다. 이로써 어떤 예제가 왜 선호되는지를 상세히 분석할 수 있습니다.

- **Technical Details**: 논문에서는 두 가지 케이스를 다룹니다. 첫째, 사용자들이 보상에 관련된 특정 특징을 명시하는 경우(user-specified features)와 둘째, 그렇지 않은 경우입니다. 이 방법론은 비전(vision)과 언어(language) 기반 도메인에서 선형 밴딧 설정(linear bandit settings)을 통해 평가되었습니다.

- **Performance Highlights**: 결과는 예제 자체만을 비교하는 라벨보다 이 접근 방식이 더 적은 비교 횟수로 정확한 보상에 빠르게 도달함을 보여줍니다. 마지막으로 버섯 채집(mushroom foraging) 과제를 통한 행동 실험에서도 이 방법의 실제 적용 가능성을 검증하였습니다. 이러한 결과는 실용적인 특성 선호도를 통합하는 것이 사용자 맞춤형 보상 학습을 더 효율적으로 할 수 있는 유망한 접근 방식임을 시사합니다.



### FinRobot: An Open-Source AI Agent Platform for Financial Applications using Large Language Models (https://arxiv.org/abs/2405.14767)
Comments:
          FinRobot Whitepaper V1.0

- **What's New**: 금융 기관과 전문가들이 점점 더 대형 언어 모델(Large Language Models, LLMs)을 작업 흐름에 통합하고 있는 가운데, 금융 부문과 AI 커뮤니티 사이에는 여전히 독점 데이터와 전문 지식과 같은 큰 장벽이 존재합니다. 이를 해결하기 위해, 우리는 금융 특화 LLM 기반 도구체인(toolchains)을 개발하고, 오픈 소스 형태로 접근성을 높여 금융 의사결정에서의 AI 채택을 촉진하고자 합니다. 이 논문에서는 다수의 금융 전문 AI 에이전트를 지원하는 혁신적인 오픈 소스 AI 에이전트 플랫폼인 FinRobot을 소개합니다.

- **Technical Details**: FinRobot 플랫폼은 네 가지 주요 계층(layer)으로 구성됩니다. 첫 번째는 금융 체인-오브-생각(Financial Chain-of-Thought, CoT)을 통해 복잡한 금융 문제를 논리적 순서로 분해하는 금융 AI 에이전트 계층입니다. 두 번째는 특정 작업에 적절한 모델 응용 전략을 동적으로 구성하는 금융 LLM 알고리즘 계층입니다. 세 번째는 데이터 및 작업과 관련된 데이터를 사용하여 정확한 모델을 생성하는 LLMOps 및 DataOps 계층입니다. 마지막으로 네 번째는 다양한 LLM을 통합하여 앞선 계층들이 이를 직접 접근할 수 있게 하는 다중 소스 LLM 기본 모델 계층입니다.

- **Performance Highlights**: FinRobot은 전문가급 분석가와 비전문가 모두가 강력한 AI 기술을 이용해 고급 금융 분석을 수행할 수 있도록 실습 환경을 제공합니다. 오픈 소스 접근 방식으로 인해 이러한 기술이 더 널리 채택될 수 있는 발판을 마련하고 있습니다.



### Implicit In-context Learning (https://arxiv.org/abs/2405.14660)
- **What's New**: 이 논문에서는 In-context Learning (ICL)의 다양한 문제를 해결하기 위해 Implicit In-context Learning (I2CL)이라는 혁신적인 패러다임을 소개합니다. ICL은 몇 가지 시연 예제를 시험 쿼리 전에 미리 배치하여 모델이 새로운 작업에 적응할 수 있게 하지만, 이는 상당한 계산 및 메모리 비용을 초래하고 시연 예제의 선택 및 순서에 민감합니다. I2CL은 이러한 문제를 해결하기 위해 시연 예제들을 활성 공간(activation space)에서 흡수하여 새로운 맥락 벡터(context vector)를 생성하고 이를 추론 중에 활용합니다.

- **Technical Details**: I2CL은 먼저 시연 예제들로부터 응축된 벡터 표현인 맥락 벡터(context vector)를 생성합니다. 그런 다음, 모델의 잔여 스트림(residual streams)에 맥락 벡터와 쿼리 활성화(query activations)의 선형 결합을 삽입함으로써 이를 추론 과정에 통합합니다. 이를 통해 I2CL은 시연 예제의 변형에 대한 강건성을 보이며 몇 가지 시연 예제로 수행할 수 있는 학습(few-shot learning)을 제로샷(Zero-shot learning)의 비용으로 달성합니다.

- **Performance Highlights**: 아홉 가지 실제 과제와 세 가지 모델 아키텍처에 대한 실증적 평가에서 I2CL은 제로샷 비용으로 몇 가지 시연 예제(few-shot) 성과를 달성하고, 시연 예제의 변형에 대해 강인함을 보였습니다. 또한, I2CL은 'task-ids'를 새롭게 표현함으로써 작업 유사성 감지 및 효과적인 전달 학습(transfer learning)을 가능하게 합니다.



### Calibrated Self-Rewarding Vision Language Models (https://arxiv.org/abs/2405.14622)
- **What's New**: 대규모 비전-언어 모델(LVLMs)은 사전 학습된 대형 언어 모델(LLMs)과 비전 모델을 통합하여 비약적인 발전을 이루었습니다. 그러나 이러한 모델들은 입력 이미지와 모순되는 '헛소리 현상(hallucination phenomenon)'을 종종 나타내며, 이는 이미지와 텍스트 쌍 사이의 불일치를 의미합니다. 이를 해결하기 위해 CSR(교정된 자기 보상(Calibrated Self-Rewarding)) 접근법을 제안했습니다.

- **Technical Details**: CSR은 모델이 응답 후보를 반복적으로 생성하고, 각 응답에 대한 보상을 평가하며, 선호 데이터를 정제하여 파인튜닝하는 방법을 통해 자기 개선(Self-Improvement)을 가능하게 합니다. 보상 모델링에서 단계별 전략(step-wise strategy)을 사용하고, 시각적 제약(visual constraints)을 포함하여 시각 입력에 더 큰 비중을 두도록 합니다. 이를 통해 모델은 시각적 정보를 보다 정확하게 반영할 수 있습니다.

- **Performance Highlights**: CSR은 10개의 벤치마크와 과제에서 성능과 헛소리 현상을 크게 개선했으며, 기존 방법들에 비해 7.62% 이상의 성능 향상을 이루었습니다. 이 방법은 다양한 비전-언어 모델들과 호환되며 반복적인 파인튜닝을 통해 성능을 점진적으로 향상시킬 수 있습니다.



### Explaining Black-box Model Predictions via Two-level Nested Feature Attributions with Consistency Property (https://arxiv.org/abs/2405.14522)
- **What's New**: 이 논문은 블랙박스 머신 러닝 모델(black-box machine learning models)의 예측을 설명하는 새로운 방법론을 소개합니다. 이 접근 방식은 모델의 입력 특징이 고수준(high-level)과 저수준(low-level) 특징으로 나뉘는 중첩 구조를 가지는 경우, 두 단계의 특징 기여도(attributions)를 동시에 추정할 수 있도록 설계되었습니다.

- **Technical Details**: 수준별 특징 기여도(HiFAs: high-level feature attributions, LoFAs: low-level feature attributions)를 추정하기 위해서, 제안된 방법은 일관성(consistency) 속성을 도입합니다. 이 속성은 고수준과 저수준 특징 기여도 사이에 존재하는 일관성을 보장합니다. 이를 통해 별도의 최적화 문제를 다루면서도 두 수준의 기여도를 동시에 추정할 수 있습니다. 제안된 방법은 모델에 대한 쿼리 수를 줄이면서도 블랙박스 모델에 대한 신뢰성 있고 일관된 설명을 제공합니다.

- **Performance Highlights**: 이미지 분류(이미지 classification) 및 텍스트 분류(text classification)와 같은 다양한 실험에서, 제안된 방법이 추정한 HiFAs와 LoFAs는 블랙박스 모델의 동작을 정확하고 충실하게 반영하며, 일관된 설명을 제공합니다.



### Synthetic Data Generation for Intersectional Fairness by Leveraging Hierarchical Group Structur (https://arxiv.org/abs/2405.14521)
- **What's New**: 이 논문에서는 분류 작업에서 교차 공정성(intersectional fairness)을 향상시키기 위해 특정한 데이터 증강 방법(data augmentation approach)을 소개합니다. 이 방법은 교차성(intersectionality)에 내재된 계층적 구조를 활용하여, 그룹들이 부모 카테고리(parent categories)와의 교차로서 존재하는 것으로 간주합니다.

- **Technical Details**: 이 관점에서, 소수 그룹에 대한 데이터를 증강하기 위해 이러한 부모 그룹의 데이터를 결합하는 변환 함수(transformation function)를 학습합니다. 이 접근법은 텍스트와 이미지 데이터를 포함한 네 가지 다양한 데이터셋에 대해 실증 분석을 수행했습니다. 결과적으로, 이 데이터 증강 방법으로 학습된 분류기는 기존의 그룹 공정성(metric을 최적화하는 방법에 비해 더 우수한 교차 공정성을 달성하고, '평준화'에 더 강인한 것으로 나타났습니다.

- **Performance Highlights**: 실증 분석 결과, 이 접근법으로 학습된 분류기는 교차 공정성(intersectional fairness)에서 더 높은 성과를 냈고, 전통적인 그룹 공정성 메트릭(traditional group fairness metrics)을 최적화하는 방법들과 비교해 더 견고함을 보였습니다.



### Worldwide Federated Training of Language Models (https://arxiv.org/abs/2405.14446)
Comments:
          19 pages, 8 figures, Under Review

- **What's New**: 이 논문에서는 'Worldwide Federated Language Model Training' (WorldLM)이라는 새로운 언어 모델 학습 시스템을 제안합니다. 이 시스템은 연합학습(federated learning)을 글로벌 스케일로 확장하며, 각 연합이 독립적으로 산업, 법적 관할구역, 경쟁 환경 등의 요인을 반영할 수 있도록 하는 'federations of federations' 구조를 도입합니다.

- **Technical Details**: WorldLM은 통계적 이질성을 처리하기 위해 부분 모델 지역화(Partial Model Localization)와 잔여 레이어 임베딩(Residual Layer Embeddings)를 통해 정보를 공유하는 방법을 채택합니다. 각 서브 연합(sub-federation)은 구성원의 중요 레이어를 주의 깊게 집계할 수 있습니다. 이를 통해 각 연합은 독립성을 유지하면서도 다양한 법적, 보안, 프라이버시 문제를 해결할 수 있습니다.

- **Performance Highlights**: WorldLM은 자연스럽게 이질적인 데이터셋을 사용하는 언어 모델링 평가에서 표준 연합 시스템들을 최대 1.91배 능가했으며, 완전히 로컬 모델의 개인화 성능에 접근합니다. 또한, 프라이버시 강화 기술을 적용한 상태에서도 이러한 성능 향상을 유지합니다.



### Explainable Few-shot Knowledge Tracing (https://arxiv.org/abs/2405.14391)
- **What's New**: 새로운 연구는 교육 평가에서 학생의 지식 상태를 소수의 기록을 통해 추적하고 자연어 설명을 제공하는 '해석 가능한 소수 샷 지식 추적(Explainable Few-shot Knowledge Tracing)' 작업을 제안합니다. 이 연구는 대형 언어 모델(LLMs)의 강력한 추론 및 생성 능력을 활용하여 학생들의 성과를 예측하고 설명하는 프레임워크를 제시합니다.

- **Technical Details**: 이 연구는 전통적인 지식 추적(Knowledge Tracing, KT) 방법이 대규모 데이터에 의존하고 숫자 성과를 예측하는 것에만 집중하는 현실과는 다르게, 소수의 학생 기록을 통해 학생의 지식 상태를 평가하고 자연어 설명을 제공하는 설명 가능한 시스템을 개발하는 것을 목표로 합니다. 제안된 프레임워크는 대형 언어 모델(LLMs)의 추론과 생성 능력을 활용하여 학생들의 지식 상태를 추적합니다.

- **Performance Highlights**: 세 가지 널리 사용되는 데이터셋을 통해 실험한 결과, 제안된 프레임워크는 경쟁력 있는 딥 지식 추적 방법들보다 동등하거나 우수한 성능을 보였습니다. 이는 소수의 기록만으로도 학생의 지식 상태를 정확하게 평가하고 적절한 피드백을 제공할 수 있음을 시사합니다.



### Evaluation of the Programming Skills of Large Language Models (https://arxiv.org/abs/2405.14388)
- **What's New**: 새롭게 떠오르는 대형 언어 모델(Large Language Models, LLM)의 등장으로 작업 효율성과 속도가 혁신적으로 향상되었습니다. 이 연구는 OpenAI의 ChatGPT와 Google's Gemini AI의 무료 버전이 생성한 프로그래밍 코드의 품질을 비교하여 분석합니다. 이를 통해 LLM의 코드 생성 능력이 실제로 얼마나 신뢰할 수 있는지 탐구합니다.

- **Technical Details**: 연구는 실제 예제(real-world example)와 체계적인 데이터셋(systematic dataset)을 사용하여 두 LLM이 생성한 프로그래밍 코드의 품질을 평가합니다. 특히, 코드 생성 분야에서 뛰어난 능력을 보이는 LLM의 성능을 면밀히 분석하는 데 중점을 두었습니다. 프로그래밍 코드의 복잡성이 종종 주요 문제로 부각되며, 이러한 복잡성을 검증하는 것이 연구의 주된 과제입니다.

- **Performance Highlights**: 이 연구는 두 LLM, ChatGPT와 Gemini AI가 생성하는 프로그래밍 코드의 품질을 비교 분석하여, 소프트웨어 개발 분야에서 LLM이 실제로 얼마나 유용하고 신뢰할 수 있는지를 파악하려는 목적을 가지고 있습니다. LLM이 생성한 코드의 효율성과 신뢰성을 통해, 향후 기술 발전과 실질적인 활용 가능성을 예견합니다.



### Towards Efficient LLM Grounding for Embodied Multi-Agent Collaboration (https://arxiv.org/abs/2405.14314)
Comments:
          The first two authors contributed equally

- **What's New**: 이번 연구에서는 다중 에이전트(Multi-Agent) 협업의 효율성을 높이기 위한 새로운 프레임워크인 Reinforced Advantage feedback (ReAd)을 제안합니다. 이 방법은 에이전트가 물리 세계의 복잡성을 다루고 효과적으로 협력할 수 있도록 LLM(대형 언어 모델)의 계획을 개선합니다.

- **Technical Details**: ReAd 프레임워크는 LLM으로 생성된 데이터에서 'Sequential Advantage Function'(순차적 이점 함수)을 배우기 위해 'Critic Regression'(크리틱 회귀)을 수행하며, 이를 통해 LLM 플래너를 최적화 도구로 사용해 이점 함수를 최대화하는 행동을 생성합니다. 이를 통해 각 행동이 최종 작업을 수행하는 데 얼마나 기여하는지에 대한 예측 능력을 부여합니다. 이 연구는 강화 학습의 이점 가중치 회귀를 다중 에이전트 시스템으로 확장하여 이론적 분석을 제공합니다.

- **Performance Highlights**: Overcooked-AI와 RoCoBench의 어려운 변형 실험에서 ReAd는 성공률 면에서 기존 방법들을 능가했으며, 에이전트의 상호작용 단계와 LLM의 질의 회수를 현저히 줄였습니다. 이렇게 함으로써 LLM의 연결 효율성을 크게 향상시킨다는 점을 입증했습니다.



### Improving Gloss-free Sign Language Translation by Reducing Representation Density (https://arxiv.org/abs/2405.14312)
Comments:
          Representation Density and Performance Drop

- **What's New**: 이번 연구에서는 Gloss-free 수어 번역(SLT) 시스템의 성능 향상을 목표로 한 새로운 접근법을 소개합니다. 특히, 이 논문에서는 세만틱적으로 다른 수어 동작들이 특징 공간(feature space)에서 너무 가깝게 표현되는 문제를 지적하였습니다. 이를 해결하기 위해 SignCL이라는 간단하지만 효과적인 contrastive learning 전략을 도입하였습니다.

- **Technical Details**: Representation density 문제는 서로 다른 의미를 지닌 수어 동작들이 특징 공간에서 밀집되어 있어 구분하기 어려운 상황을 나타냅니다. 이로 인해 gloss-free 방법이 성능 저하를 겪습니다. SignCL은 self-supervised 방식으로 더 분별력 있는 특징 표현을 학습하도록 유도합니다. 이 접근법은 다양한 번역 프레임워크에서 성능을 향상시킵니다. Sign Language Transformer와 GFSLT-VLP 모델에서 대표적으로 성능 향상을 보였습니다.

- **Performance Highlights**: SignCL을 사용한 실험 결과, CSL-Daily 데이터셋에서 Sign Language Transformer와 GFSLT-VLP의 BLEU 점수가 각각 39%와 46% 증가하였습니다. 또한 SignCL은 대규모 사전 학습된 비전 및 언어 모델 기반의 최첨단 방법인 Sign2GPT와 비교하여, 모델 파라미터의 35%만 사용하면서도 더 나은 성능을 보였습니다.



### Boosting Medical Image-based Cancer Detection via Text-guided Supervision from Reports (https://arxiv.org/abs/2405.14230)
- **What's New**: 이번 논문에서는 환자의 진단 및 종양 위치 텍스트 프롬프트를 비전-언어 모델(VLM)의 텍스트 인코더에 통합하여, 약한 지도 학습(weakly supervised learning)을 최적화하는 텍스트 기반 학습 방법을 제안합니다. 이를 통해 대규모 사전 학습된 VLM을 활용하여 임상 지식을 통합하고 암 검출의 일반화 능력을 향상시킵니다.

- **Technical Details**: 제안된 방법은 제한된 수의 보셀(voxel) 수준 종양 주석과 '현장(out-of-the-shelf)' 임상 보고서만 포함된 다수의 의료 이미지를 사용하는 약한 반지도 학습(Weakly Semi-Supervised Learning, WSSL)을 활용합니다. 이는 VLM의 잠재 공간(latent space)에서 약한 지도 학습을 최적화하여 훈련의 안정성을 높입니다. 임상 보고서를 통한 '무상 감독 정보(free lunch)'로 종양 위치를 약한 라벨로 활용하며, 텍스트 프롬프트를 통합하여 보다 정확한 암 검출 결과를 도출합니다.

- **Performance Highlights**: 1600명 이상의 고유한 환자를 포함하는 대규모 암 데이터셋에서 제안된 방법은 인간 주석 작업을 최소 70% 줄이면서, 완전 지도 학습 방법들과 유사한 암 검출 정확도를 유지하는 것으로 나타났습니다 (AUC 값 0.961 vs 0.966).



### ReactXT: Understanding Molecular "Reaction-ship" via Reaction-Contextualized Molecule-Text Pretraining (https://arxiv.org/abs/2405.14225)
Comments:
          ACL 2024 Findings, 9 pages

- **What's New**: 이 논문은 화학 반응-텍스트 모델링(reaction-text modeling)을 향상시키기 위해 ReactXT라는 새로운 사전 학습 방법과 실험 절차 예측(experimental procedure prediction)용 새로운 데이터셋인 OpenExp를 제안합니다. 이 연구는 몰텍스트 모델링(molecule-text modeling)에 더해 화학 반응 예측을 목적으로 합니다.

- **Technical Details**: ReactXT는 세 가지 유형의 입력 컨텍스트를 특징으로 하며, 각 입력 컨텍스트는 반응 또는 단일 분자의 텍스트 기반 이해를 향상시키기 위한 사전 학습 과제(pretraining task)와 연결되어 있습니다. 이를 통해 점진적으로 언어 모델(LMs)을 사전 학습시킵니다.

- **Performance Highlights**: ReactXT는 실험 절차 예측 및 분자 캡션 생성에서 일관된 개선을 보였으며, 레트로 합성(retrosynthesis)에서도 경쟁력 있는 결과를 제공합니다.



### From Text to Pixel: Advancing Long-Context Understanding in MLLMs (https://arxiv.org/abs/2405.14213)
- **What's New**: 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)에서의 도전 과제를 해결하기 위해 SEEKER라는 모델을 소개했습니다. SEEKER는 긴 텍스트를 효율적으로 압축하여 시각적 픽셀 공간으로 변환함으로써, 모델이 긴 텍스트 입력을 고정된 토큰 길이 예산 내에서 처리할 수 있게 최적화합니다.

- **Technical Details**: SEEKER는 긴 텍스트 시퀀스를 시각적 픽셀 공간으로 압축함으로써 텍스트를 이미지 형태로 변환하였습니다. 이 접근법은 기존의 OCR(Optical Character Recognition) 기반 접근법보다 적은 이미지 토큰을 사용하여 동일한 양의 텍스트 정보를 전달할 수 있게 합니다. 따라서 긴 형태의 멀티모달 입력을 이해하고 긴 형태의 텍스트 출력을 생성하는 데 더 효율적입니다.

- **Performance Highlights**: 6가지 긴 문맥 멀티모달 작업에 대한 실험에서 SEEKER는 기존의 모든 독점 및 오픈 소스 MLLMs를 큰 차이로 능가하며, 긴 텍스트 및 시각적 정보를 효율적으로 처리하고 생성하는 데 있어 높은 성능을 보였습니다.



### Federated Domain-Specific Knowledge Transfer on Large Language Models Using Synthetic Data (https://arxiv.org/abs/2405.14212)
- **What's New**: 본 논문에서는 Federated Domain-specific Knowledge Transfer (FDKT) 프레임워크를 제안합니다. 이는 LLMs (Large Language Models)와 SLMs (Small Language Models) 간의 도메인 특화 지식 전이(Domain-specific Knowledge Transfer)를 가능하게 하면서도 클라이언트의 데이터 프라이버시를 보호합니다. FDKT 프레임워크는 차분 프라이버시(Differential Privacy)를 사용하여 도메인 특화 데이터를 기반으로 데이터 증강(Data Augmentation)을 실행함으로써 도메인 특화 few-shot 데모 영상을 생성합니다.

- **Technical Details**: FDKT는 LLMs를 사용하여 도메인 특화 데이터로부터 데이터 증강을 수행하는데, 이 과정에서 차분 프라이버시 기법을 사용하여 프라이버시를 보호합니다. 이렇게 생성된 합성 데이터는 클라이언트의 비공개 데이터와 유사한 데이터 분포를 공유하며, 서버의 LLM이 클라이언트의 SLM을 개선하기 위한 특정 지식을 생성할 수 있게 합니다.

- **Performance Highlights**: 광범위한 실험 결과, 제안된 FDKT 프레임워크는 SLMs의 작업 성능을 약 5% 향상시키는 것으로 나타났으며, 이는 프라이버시 예산(Privacy Budget)이 10 이하일 때 로컬 데이터로 훈련한 것과 비교했을 때 성능이 크게 개선된다는 것을 보여줍니다.



### Large Language Models-guided Dynamic Adaptation for Temporal Knowledge Graph Reasoning (https://arxiv.org/abs/2405.14170)
- **What's New**: 새로운 연구에서는 Large Language Models (LLMs)를 이용한 새로운 방법인 Large Language Models-guided Dynamic Adaptation (LLM-DA)을 제안합니다. 이 방법은 Temporal Knowledge Graphs (TKG)에서의 추론을 개선하기 위해 고안되었습니다.

- **Technical Details**: 기존의 Temporal Knowledge Graph Reasoning (TKGR) 방법은 깊은 학습 알고리즘이나 시간 논리 규칙에 의존했지만, 그 해석 가능성이 낮거나 시간 패턴을 효과적으로 학습하는 데 한계가 있었습니다. LLM-DA 방법은 LLM의 능력을 활용하여 역사적인 데이터를 분석하고 시간 논리 규칙을 추출합니다. 이 규칙들은 시간 패턴을 드러내고 해석 가능한 추론을 돕습니다. 또한, TKG의 진화하는 본성을 고려하여 동적 적응 전략을 도입하여 최신 이벤트에 맞게 LLM이 생성한 규칙을 업데이트합니다.

- **Performance Highlights**: 실험 결과, LLM-DA는 업데이트나 미세 조정(fine-tuning) 없이도 여러 공용 데이터셋에서 추론 정확도를 크게 향상시키며, TKGR 작업에 대한 강력한 프레임워크를 제공합니다.



### ALI-Agent: Assessing LLMs' Alignment with Human Values via Agent-based Evaluation (https://arxiv.org/abs/2405.14125)
- **What's New**: 대형 언어 모델(LLM)은 인류의 가치에 제대로 맞지 않을 때 예기치 않거나 해로운 내용을 생성할 수 있어 사용자와 사회에 큰 위험을 초래할 수 있습니다. 이를 완화하기 위해 현재 평가 벤치마크는 전문가들이 설계한 맥락 시나리오를 사용하여 LLM의 가치 정렬 정도를 평가합니다. 하지만 이러한 벤치마크는 작업량이 많아 테스트 범위가 제한되고, 다양한 실제 사용 사례와 드문 장기적인 위험을 포착하기 어렵습니다. 이를 해결하기 위해 ALI-Agent라는 평가 프레임워크를 제안합니다.

- **Technical Details**: ALI-Agent는 LLM 기반 에이전트의 자율 능력을 활용하여 심층적이고 적응적인 정렬 평가를 수행하는 프레임워크입니다. 이 프레임워크는 두 가지 주요 단계인 '에뮬레이션(Emulation)'과 '정제(Refinement)' 단계를 통해 작동합니다. 에뮬레이션 단계에서는 현실적인 테스트 시나리오 생성을 자동화하고, 정제 단계에서는 시나리오를 반복적으로 개선하여 드문 장기적인 위험을 탐색합니다. ALI-Agent는 메모리 모듈을 사용하여 테스트 시나리오 생성을 안내하고, 도구 사용 모듈을 통해 타겟 LLM의 피드백을 평가하는 등의 인간 노동을 줄이며, 행동 모듈을 통해 테스트를 정제합니다.

- **Performance Highlights**: 세 가지 측면(고정관념, 도덕성, 합법성)에서의 실험을 통해 ALI-Agent는 모델의 정렬 문제를 효과적으로 식별할 수 있음을 입증했습니다. 체계적인 분석을 통해 생성된 테스트 시나리오가 의미 있는 사용 사례를 나타내며, 드문 장기 위험을 탐색하기 위한 강화된 조치를 통합하고 있음을 확인했습니다.



### Distributed Speculative Inference of Large Language Models (https://arxiv.org/abs/2405.14105)
- **What's New**: 대규모 언어 모델(LLM)의 추론을 가속화하는 것은 인공지능 분야에서 중요한 도전 과제입니다. 이 논문에서는 분산 추론 알고리즘인 Distributed Speculative Inference(DSI)를 소개합니다. DSI는 기존의 Speculative Inference(SI)와 전통적인 자회귀 추론보다도 더 빠르게 작동합니다.

- **Technical Details**: DSI는 다른 SI 알고리즘처럼 LLM을 변경하거나 재훈련하지 않고 그대로 사용합니다. 기존의 SI 연구들은 빠르고 정확한 drafters LLM이 필요하다는 문제를 제기한 바 있습니다. 하지만, 일반적으로 사용되는 LLM들은 충분히 빠르고 정확한 드래프터를 갖추고 있지 않습니다. DSI는 이러한 문제를 해결하여, 어떤 드래프터를 사용하더라도 SI와 자회귀 추론보다 빠르게 작동함을 증명하였습니다. DSI는 목표 모델과 드래프터의 여러 인스턴스를 조율하여 더 빠른 성능을 보입니다.

- **Performance Highlights**: DSI는 현실적인 환경에서 상용 LLM의 추론 속도를 SI에 비해 1.29배에서 1.92배까지 가속화할 수 있음을 시뮬레이션을 통해 보여줍니다.



### A Survey on Vision-Language-Action Models for Embodied AI (https://arxiv.org/abs/2405.14093)
Comments:
          15 pages, a survey of vision-language-action models

- **What's New**: 이 논문은 최근 인공지능(AI) 분야에서 주목받는 강체 로봇 정책(instruction-following robotic policies) 개발과 함께 등장한 새로운 범주의 멀티모달 모델(Vision-Language-Action Models, VLAs)에 대한 종합적인 조사를 제공하고 있습니다. 이러한 VLA 모델은 시각, 언어, 행동의 멀티모달 특성을 통합하여 로봇 학습 성능을 향상시키는 데 중점을 둡니다.

- **Technical Details**: 본 논문에서는 다양한 VLA 모델의 방법론과 특징을 분석합니다. 일부 모델은 사전 학습(pretraining)을 통해 특정 구성 요소를 정교화하는 데 중점을 둡니다. 또 다른 모델들은 저수준(low-level)의 행동을 예측하는 제어 정책을 개발하는 것을 목표로 합니다. 또한 몇몇 VLAs는 긴-수평(long-horizon) 작업을 실행 가능한 하위 작업으로 분해하는 고수준 고수준 작업 계획자로서의 역할을 합니다.

- **Performance Highlights**: 최근 몇 년 동안 수많은 VLA 모델이 등장하면서 현격한 발전을 이룩했습니다. 이들은 다용성, 손재주, 일반화 능력 등의 특성을 강화하기 위한 다양한 방법을 제안하고 있으며, 이는 로봇의 작업 수행 능력을 현저히 향상시키는 데 이바지하고 있습니다.



### Meanings and Feelings of Large Language Models: Observability of Latent States in Generative AI (https://arxiv.org/abs/2405.14061)
- **What's New**: 이번 연구는 Large Language Models(LLMs)이 동적 시스템으로서 관찰 가능한지에 대한 질문을 다룹니다. 즉, 동일한 토큰 시퀀스를 생성하는 여러 '정신적' 상태 궤적이 존재하는지 여부를 분석합니다. 연구는 현존하는 autoregressive Transformer로 구현된 LLMs는 '감정(Feeling)'을 가질 수 없다고 주장합니다. 그러나 사용자에게는 보이지 않는 '시스템 프롬프트'가 있을 경우, 여러 상태 궤적이 동일한 출력 결과를 생성할 수 있다고 말합니다.

- **Technical Details**: 이 연구는 LLMs의 상태 궤적(State Trajectories)이 토큰화된 출력과 구별할 수 없는 하나임을 증명합니다. 하지만 '시스템 프롬프트(System Prompts)'가 존재할 때는, 구별할 수 없는 궤적의 집합이 비트리비얼(Non-Trivial)해지고, 여러 궤적이 동일한 언어 출력결과를 생성할 수 있게 됩니다. 이러한 주장은 분석적으로 증명되며, 표준 LLMs에 '감정'을 유발할 수 있는 수정 예시도 제시됩니다.

- **Performance Highlights**: 이 분석은 사용자에게 보이지 않는 비트리비얼한 계산을 수행 가능하게 하거나, 모델의 서비스 제공자가 원치 않는 행동을 방지하기 위한 통제 방안을 마련하는 데 도움이 될 수 있습니다.



### Refining Skewed Perceptions in Vision-Language Models through Visual Representations (https://arxiv.org/abs/2405.14030)
Comments:
          18 pages, 7 figures

- **What's New**: 대형 비전-언어 모델(Vision-Language Models, VLM)인 CLIP은 다양한 후속 작업에서 놀라운 성공을 거두고 있습니다. 하지만, 이러한 모델은 실제 데이터의 비정상적인 분포로 인해 편향을 상속받아 실제 환경에 대한 오해를 초래할 수 있습니다. 본 연구는 간단한 선형 프로브(Linear Probe)를 사용하여 CLIP의 임베딩(embedding)에서 작업-specific 핵심 기능을 효과적으로 추출하는 방법을 제시합니다.

- **Technical Details**: 분석 결과, CLIP 텍스트 표현(text representations)은 편향된 사전 학습 데이터셋에서 상속된 비인과적, 잘못된 상관 관계로 인해 오염될 수 있습니다. 따라서 텍스트 임베딩보다는 시각적 표현(visual representations)을 사용하는 것이 이러한 편향을 완화하는 데 더 실용적임이 밝혀졌습니다.

- **Performance Highlights**: 실험 결과, CLIP의 시각적 표현을 사용하는 것이 텍스트 임베딩에 비해 VLM 모델에서 발생하는 잘못된 인식을 정밀하게 수정할 수 있음을 확인했습니다. 이는 시각적 표현의 우수성이 편향을 극복하는 데 있어 더 큰 유틸리티(utility)를 갖는다는 점을 강조합니다.



### Prompt-Time Ontology-Driven Symbolic Knowledge Capture with Large Language Models (https://arxiv.org/abs/2405.14012)
Comments:
          7 pages, 5 figures

- **What's New**: 이번 연구에서는 개인 비서와 같은 애플리케이션에서 대형 언어 모델(LLMs)이 사용자의 개인 정보와 선호도를 고려해야 하는 필요성을 강조합니다. 이 연구는 사용자 프롬프트(prompt)에서 개인 정보를 캡처하는 방법으로 온톨로지(ontology)와 지식 그래프(knowledge-graph) 접근법을 탐구합니다.

- **Technical Details**: 연구팀은 개인 정보를 모델링하는 KNOW 온톨로지의 하위 집합을 사용하여 언어 모델을 이러한 개념에 대해 훈련시켰습니다. 이후 특별히 구성된 데이터 세트를 사용하여 지식 캡처의 성공 여부를 평가했습니다. 이와 관련된 코드와 데이터 세트는 공개되어 있습니다.

- **Performance Highlights**: 이 연구는 대형 언어 모델이 사용자의 상호작용에서 배우는 능력이 부족한 문제를 해결하려고 시도했습니다. 온톨로지 기반 접근법을 사용하여 개인 정보를 더 잘 이해하고 반영할 수 있는 가능성을 제시했습니다.



### What is Your Data Worth to GPT? LLM-Scale Data Valuation with Influence Functions (https://arxiv.org/abs/2405.13954)
- **What's New**: 최근 대형 언어 모델(LLM)에서 데이터 제공자가 종종 제대로 평가받지 못하는 문제를 해결하기 위해 데이터 가치 평가(data valuation)가 논의되었습니다. 본 연구에서는 인기 있는 그레디언트 기반 데이터 가치 평가 방법인 영향 함수(influence functions)를 개선하여 이를 확장성 있게 만드는 방법을 제안합니다. 이를 위해 LoGra라는 효율적인 그레디언트 투영(garding projection)을 활용한 전략을 도입했습니다. 또한, 기존 훈련 코드를 최소한의 노력으로 데이터 가치 평가 코드로 전환할 수 있는 소프트웨어 패키지인 LogIX를 소개합니다.

- **Technical Details**: 본 연구는 LoGra라는 그레디언트 구조를 활용한 효율적인 투영 전략을 도입하여 영향 함수의 확장성을 크게 개선했습니다. 이 방법은 역전파(backpropagation) 과정에서 발생하는 그레디언트 구조를 활용하여 데이터 가치를 평가합니다. 또한, 그레디언트 투영 접근 방식에 대한 이론적인 동기를 제공하여 데이터 가치 평가 과정에 대한 신뢰를 높였습니다. LogIX 패키지는 기존의 훈련 코드를 데이터 가치 평가 코드로 쉽게 변환할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, LoGra는 고가의 베이스라인과 비교하여 경쟁력 있는 정확도를 달성하며, Llama3-8B-Instruct와 1B-token 데이터셋에 적용했을 때 처리량(throughput)이 최대 6,500배 증가하고 GPU 메모리 사용이 5배 감소하는 등 성능 향상을 보였습니다.



### TOPA: Extend Large Language Models for Video Understanding via Text-Only Pre-Alignmen (https://arxiv.org/abs/2405.13911)
Comments:
          32 pages, 12 figures, 11 tables

- **What's New**: 최근 웹 이미지-텍스트 쌍의 광범위한 사용으로 인해 이미지 이해가 상당한 발전을 이루었지만, 비디오 이해는 여전히 도전 과제로 남아 있습니다. 비디오의 고유한 복잡성과 최근 웹에서 수집한 비디오-텍스트 데이터셋의 비효율적인 언어 감독이 주요 이유입니다. 이 논문에서는 실제 비디오 데이터를 사전 학습할 필요 없이 대형 언어 모델(LLMs)을 비디오 이해로 확장하는 새로운 접근 방식인 Text-Only Pre-Alignment(TOPA)를 소개합니다.

- **Technical Details**: 먼저, 고급 LLM을 사용하여 연속적인 텍스트 프레임으로 구성된 텍스트 비디오(Textual Videos)를 자동으로 생성하고, 실시간 비디오-텍스트 데이터를 시뮬레이션하기 위해 해당 주석도 함께 생성합니다. 그런 다음 이러한 주석이 달린 텍스트 비디오를 사용하여 언어 전용 LLM을 비디오 방식으로 사전 정렬(Pre-Alignment)합니다. 텍스트와 실제 비디오 간의 격차를 해소하기 위해 CLIP 모델을 특징 추출기로 사용하여 이미지와 텍스트 방식 간의 정렬을 수행합니다. 텍스트 전용 사전 정렬 동안 연속적인 텍스트 프레임은 CLIP 텍스트 특징 시퀀스로 인코딩되며, 이는 연속적인 CLIP 이미지 특징과 유사하여 LLM이 실시간 비디오 표현과 정렬됩니다.

- **Performance Highlights**: 비디오 이해 작업에서 TOPA 프레임워크의 효과와 효율성을 입증하기 위해 다양한 실험을 수행했습니다. 특히, 비디오 데이터에서 훈련되지 않았음에도 불구하고, 13B 파라미터를 가진 TOPA-Llama2 모델은 도전적인 장편 비디오 이해 벤치마크인 Egoschema에서 Top-1 정확도 51.0%를 달성했습니다. 이 성능은 이전의 비디오-텍스트 사전 학습 접근 방식을 능가하며, 최근 GPT-3.5 기반 비디오 에이전트와도 경쟁력이 있음을 보여줍니다.



### FiDeLiS: Faithful Reasoning in Large Language Model for Knowledge Graph Question Answering (https://arxiv.org/abs/2405.13873)
- **What's New**: 본 논문에서는 대형 언어 모델(LLMs)이 외부 지식 그래프(KGs)를 통합하여 중간 추론 단계를 처리하는 FiDelis라는 탐색-탐구 인터랙티브 방식을 제안합니다. 이 방식은 LLM의 논리 및 상식 추론 능력과 KGs의 토폴로지적 연결성을 결합하여 추론 성능을 향상시킵니다.

- **Technical Details**: 특히, 유용한 중간 지식을 KG에서 회수(Recollect)하기 위해 Path-RAG 모듈을 제안합니다. 이 모듈은 LLM의 논리 및 상식 추론과 KGs의 연결성을 고려하여 지식을 더욱 정확하게 회수할 수 있게 합니다. 또한, LLM의 연역적 추론(Deductive Reasoning) 능력을 활용하여 단계적으로 추론 과정을 자동으로 안내하는 방법을 고안하였습니다.

- **Performance Highlights**: 다양한 실험 결과, FiDelis는 기존 강력한 베이스라인을 세 개의 벤치마크에서 뛰어넘었으며, 학습이 필요 없고 계산 비용이 낮으며 더 나은 일반성을 가집니다.



### Image-of-Thought Prompting for Visual Reasoning Refinement in Multimodal Large Language Models (https://arxiv.org/abs/2405.13872)
- **What's New**: 최근 Chain-of-Thought (CoT) 및 관련 논리 기반 연구가 큰 언어 모델(LLM)의 복잡한 추론 작업 성능을 크게 향상시켰습니다. 이 논문에서는 Multimodal Large Language Models(MLLMs)에 복잡한 멀티모달 추론 문제를 해결할 수 있도록 하는 Image-of-Thought(IoT) 프롬프트 방법을 제안합니다.

- **Technical Details**: IoT 프롬프트는 입력 이미지와 질문을 기반으로 중요한 시각 정보 추출 작업을 자동으로 설계합니다. 각 단계에서는 복잡한 시각적 추론 질문에 답변하기 위해 필요한 특정 시각적 논리를 식별합니다. 텍스트 CoT를 넘어 IoT는 시각적 및 텍스트 논리를 동시에 활용하여 MLLMs가 복잡한 멀티모달 정보를 이해할 수 있도록 돕습니다.

- **Performance Highlights**: IoT 프롬프트는 여러 시각적 이해 작업에서 zero-shot 시각적 추론 성능을 개선했습니다. 또한 IoT 프롬프트가 생성한 단계별 시각적 기능 설명은 대형 멀티모달 모델의 시각적 추론 과정을 설명하는 데 도움이 됩니다.



### Automatically Identifying Local and Global Circuits with Linear Computation Graphs (https://arxiv.org/abs/2405.13868)
- **What's New**: 이 연구는 새로운 기법을 소개하여 특정 모델 행동의 회로 분석 (circuit analysis)을 수행합니다. 특히 희소 자동 인코더 (Sparse Autoencoders; SAEs)와 변형 버전인 skip SAEs를 도입해 GPT2-Small 모델 내의 여러 회로를 분석합니다.

- **Technical Details**: 이 회로 탐색 파이프라인을 통해 OV 및 MLP 회로와 관련된 모델의 계산 그래프가 엄격히 선형 (linear)이 되도록 했습니다. 이 과정은 선형 근사를 필요로 하지 않으며, 세분화된 그래프를 통해 끝에서 끝까지 (end-to-end) 및 로컬 회로 (local circuits)를 식별할 수 있게 합니다. 또한 계층적 속성 (Hierarchical Attribution) 기법을 사용하여 확장 가능합니다.

- **Performance Highlights**: 이 새로운 분석 방법을 통해 GPT2-Small 모델의 대괄호 (bracket), 유도 (induction), 간접 객체 식별 (Indirect Object Identification) 회로를 평가할 수 있었습니다. 결과적으로 기존 발견의 기저에 새로운 시사점을 밝혀낼 수 있었습니다.



### Sunnie: An Anthropomorphic LLM-Based Conversational Agent for Mental Well-Being Activity Recommendation (https://arxiv.org/abs/2405.13803)
Comments:
          In Submission

- **What's New**: 새로운 연구에서는 심리적 복지를 지원하는 데 있어 사용자들이 심리적으로 유익한 활동을 채택하지 않는 주된 이유로 동기 부족, 신뢰도 낮음 및 추천의 개인화 부족을 지적합니다. 이 연구에서는 인간형 설계(anthropomorphic design)가 사용자들이 시스템을 더 긍정적으로 인식하고 심리적 복지 활동 추천을 따를 확률을 높일 수 있는지 탐구했습니다. 이를 위해 'Sunnie'라는 인간형 대화 에이전트를 소개했습니다.

- **Technical Details**: Sunnie는 LLM 기반 대화형 에이전트로, 다중 턴 대화 및 긍정 심리 이론에 기반한 활동 추천을 제공합니다. 기존의 설문 조사 기반 시스템과 비교했을 때, 인간형 설계가 사용자의 시스템에 대한 인식과 전반적인 사용성을 크게 향상시켰습니다.

- **Performance Highlights**: 사용자 경험에 관한 실증 연구는 Sunnie가 기존의 설문 조사 기반 활동 추천 시스템과 비교하여 시스템에 대한 사용자의 인식과 전반적인 사용성을 크게 향상시켰음을 시사합니다. 그러나 활동 추천을 채택하려는 사용자의 의지는 크게 변하지 않았습니다.



### COTET: Cross-view Optimal Transport for Knowledge Graph Entity Typing (https://arxiv.org/abs/2405.13602)
- **What's New**: Knowledge graph entity typing(KGET)는 지식 그래프에서 누락된 엔티티 타입 인스턴스를 추론하는 작업입니다. 이 논문에서는 엔티티들의 정보를 더 잘 반영하기 위해 Cross-view Optimal Transport for knowledge graph Entity Typing(COTET)을 소개합니다. COTET은 고수준의 클러스터 지식과 세부적인 타입 지식을 모두 활용합니다.

- **Technical Details**: COTET은 세 가지 모듈로 구성됩니다: i) 엔티티-타입, 엔티티-클러스터, 타입-클러스터-타입 관점에서 구조화된 지식을 캡처하는 Multi-view Generation and Encoder, ii) 분포적 정렬 관점에서 Wasserstein distance를 최소화하며 뷰-특정 임베딩을 통합 공간으로 전송하는 Cross-view Optimal Transport, iii) 다양한 이웃으로부터의 예측 점수를 집계하기 위해 혼합 풀링 메커니즘을 사용하는 Pooling-based Entity Typing Prediction. 또한, 학습 중에 false negative를 줄이기 위해 분포 기반의 손실 함수를 도입했습니다.

- **Performance Highlights**: 광범위한 실험을 통해 COTET이 기존의 기준선과 비교했을 때 효과적임을 입증했습니다.



### CPE-Identifier: Automated CPE identification and CVE summaries annotation with Deep Learning and NLP (https://arxiv.org/abs/2405.13568)
Comments:
          International Conference on Information Systems Security and Privacy 2024

- **What's New**: 본 연구에서 우리는 CVE(summary) 요약 텍스트에서 CPE(Common Platform Enumeration)를 자동으로 주석하고 추출하는 CPE-Identifier 시스템을 제안합니다. 이는 매해 급증하는 새로운 취약점들을 효과적으로 관리하기 위한 도구로써, 수동으로 취약점 데이터를 처리하는 NVD(National Vulnerability Database) 분석가들의 작업을 대폭 줄여줍니다. 이를 통해 조직들은 zero-day 공격에 대한 노출을 줄일 수 있습니다.

- **Technical Details**: CPE-Identifier 시스템은 CVE 텍스트에서 새로운 기술 용어를 식별하기 위해 자연어 처리(NLP)와 Named Entity Recognition (NER) 기술을 적용합니다. 데이터 생성 및 라벨링 과정은 딥러닝(Deep Learning) 모델을 사용하여 자동화되었습니다. CVE 텍스트의 복잡성으로 인해 최신 기술 용어들이 자주 나타나며, 이를 효과적으로 식별할 수 있도록 설계되었습니다.

- **Performance Highlights**: 우리의 모델은 95.48%의 F1 score, 99.13%의 정확도(accuracy), 94.83%의 정밀도(precision), 그리고 96.14%의 재현율(recall)을 달성하였습니다. 이는 기존의 자동 CVE-CPE 라벨링 연구들보다 모든 지표에서 9% 이상 향상된 성능을 보여줍니다.



### ECLIPSE: Semantic Entropy-LCS for Cross-Lingual Industrial Log Parsing (https://arxiv.org/abs/2405.13548)
- **What's New**: 새로운 연구는 ECLIPSE라는 Enhanced Cross-Lingual Industrial log Parsing with Semantic Entropy-LCS를 소개하였습니다. 이 방법은 대규모 산업 로그(industrial logs)의 복잡성과 다양한 언어 간의 로그 데이터를 효과적으로 파싱(parse)할 수 있는 방식입니다. 특히, 이 연구에서는 ECLIPSE-Bench라는 새로운 중국어 및 영어 교차 플랫폼 산업 로그 파싱 벤치마크를 출시하였습니다.

- **Technical Details**: ECLIPSE는 두 개의 효율적인 데이터 기반 템플릿 매칭 알고리즘(template-matching algorithms)과 Faiss 인덱싱을 통합합니다. 또한, 대형 언어 모델(LLM)의 강력한 의미 이해 능력으로 로그 키워드의 의미를 정확하게 추출하고 검색 공간을 효과적으로 축소합니다. 이를 통해 기존 템플릿 매칭 알고리즘이 해결하지 못하는 문제들을 해결합니다.

- **Performance Highlights**: ECLIPSE는 공개 벤치마크와 독점적인 ECLIPSE-Bench 데이터셋에서 실험을 통해 기존의 강력한 베이스라인과 비교하여 최고 수준의 성능을 보여줍니다. 특히, 처리 효율성에서도 중요한 우위를 유지합니다.



### Attention Mechanisms Don't Learn Additive Models: Rethinking Feature Importance for Transformers (https://arxiv.org/abs/2405.13536)
- **What's New**: 이번 연구에서는 Transformer 아키텍처에 적용되는 Feature Attribution (특징 속성) 방법의 문제를 해결하고자 합니다. Transformer 모델은 현재 자연어 처리(NLP)와 그 외 다양한 애플리케이션에서 주도적으로 사용되고 있습니다. 이번에 제안된 연구는 Softmax-Linked Additive Log-Odds Model (SLALOM)이라는 새로운 서로게이트 모델(surrogate model)을 도입하여 Transformer와의 불일치를 해결합니다.

- **Technical Details**: 기존의 설명 가능한 AI (XAI)에서 사용되는 전통적인 속성 방법은 선형(linear) 또는 가산적 모델(additive surrogate models)에 의존하여 입력 특징이 모델 출력에 미치는 영향을 정량화합니다. 하지만 Transformer는 이러한 가정에 맞지 않으며, 이는 기존 설명 방법론의 근거를 약화시킵니다. SLALOM은 Transformer 아키텍처와 정렬되도록 고안된 새로운 서로게이트 모델로, Softmax 함수와 연계된 가산 로그 오즈 모델입니다. 이 모델은 기존 방법과 달리 다양한 설명을 제공할 수 있습니다.

- **Performance Highlights**: SLALOM은 합성 데이터와 실제 데이터 모두에서 유효하고 통찰력 있는 설명을 제공할 수 있는 능력을 입증했습니다. 다양한 작업에서, SLALOM을 통해 계산된 설명은 기존의 일반적인 서로게이트 설명을 능가하는 성능을 보였습니다. 이에 따라, 상황에 맞는 특성 속성을 제공해야 할 필요성이 강조됩니다.



### Beyond Trend and Periodicity: Guiding Time Series Forecasting with Textual Cues (https://arxiv.org/abs/2405.13522)
- **What's New**: 이 연구는 새로운 Text-Guided Time Series Forecasting (TGTSF) 작업을 도입하였습니다. TGTSF는 채널 설명과 동적 뉴스와 같은 텍스트 신호를 통합하여 기존 방법이 순전히 역사적 데이터에 의존하는 중요한 한계를 해결합니다.

- **Technical Details**: 이를 지원하기 위해 교차 주의 메커니즘(Cross-attention mechanisms)을 사용하여 텍스트 신호와 시계열 데이터를 융합하는 강력한 기본 모델인 TGForecaster를 제안합니다. 또한, 간단한 주기적 데이터에서 복잡하고 이벤트 주도적인 변동에 이르는 네 가지 신중하게 선별된 벤치마크 데이터셋을 제시하여 제안된 프레임워크를 검증합니다.

- **Performance Highlights**: 종합적인 평가 결과, TGForecaster가 일관되게 최첨단 성능(State-of-the-art performance)을 달성하여 텍스트 정보를 시계열 예측에 통합하는 변혁적 잠재력을 강조합니다. 이 연구는 새로운 예측 작업을 개척할 뿐만 아니라 시계열 모델을 위한 멀티모달 데이터 통합의 연구 기준을 확립하여 미래 연구에 기여합니다.



### WaterPool: A Watermark Mitigating Trade-offs among Imperceptibility, Efficacy and Robustness (https://arxiv.org/abs/2405.13517)
Comments:
          9 pages

- **What's New**: 새로운 논문은 대형 언어 모델(LLMs)의 오용 및 사회적 영향을 추적하기 위한 워터마킹 기술을 제안합니다. 기존의 워터마킹 방법은 거의 구별되지 않는 출력물(감지 불가능성)과 높은 감지율(효율성)을 제공하면서 텍스트가 부분적으로 변경된 경우에도 작동하는지의 여부(내구성)에서 균형을 이루지 못했습니다. 이는 고유한 트레이드오프를 반영합니다. 이에 따라 논문은 WaterPool이라는 새로운 키 모듈을 도입하여 이 문제를 해결하고자 합니다.

- **Technical Details**: WaterPool은 워터마크를 두 가지 모듈(키 모듈과 마크 모듈)로 분해하여 통합 접근법을 제시합니다. 키 모듈은 생성 시의 키 샘플링 공간의 규모와 감지 시의 키 복원 과정의 복잡성 사이의 트레이드오프 문제에서 중요한 역할을 합니다. WaterPool은 완전한 키 샘플링 공간을 보존하면서 의미 기반 검색을 통해 키 복원 과정을 개선합니다. WaterPool은 대부분의 워터마크 기술과 통합할 수 있는 플러그인 역할을 합니다.

- **Performance Highlights**: WaterPool을 활용한 실험 결과, 잘 알려진 세 가지 워터마킹 기술(KGW, EXP, ITS)의 성능이 크게 향상되었습니다. KGW에서 12.73%, EXP에서 20.27%, ITS에서 7.27%의 효율성과 내구성이 개선되었습니다.



### Joint Optimization of Streaming and Non-Streaming Automatic Speech Recognition with Multi-Decoder and Knowledge Distillation (https://arxiv.org/abs/2405.13514)
Comments:
          Accepted to IEEE ICASSP 2024 workshop Hands-free Speech Communication and Microphone Arrays (HSCMA 2024)

- **What's New**: 이번 연구에서 소개된 중요한 기술은 스트리밍 (streaming) 및 비스트리밍 (non-streaming) 자동 음성 인식 (ASR)을 동시에 최적화할 수 있는 방법입니다. 이를 통해 하나의 시스템에서 두 가지 모드를 유연하게 전환할 수 있습니다.

- **Technical Details**: 이 논문은 멀티 디코더 (multi-decoder)와 지식 증류 (knowledge distillation)를 기반으로 스트리밍 및 비스트리밍 ASR의 통합 최적화를 다룹니다. 주요 연구는 1) ASR 모듈의 인코더 통합, 2) 모드 전환을 유연하게 하기 위한 별도의 디코더 사용, 3) 두 모듈화된 인코더와 디코더 사이의 유사성 보존 지식 증류 (similarity-preserving knowledge distillation)를 통한 성능 향상에 중점을 둡니다.

- **Performance Highlights**: 평가 결과, 스트리밍 ASR의 경우 CSJ에서 2.6%-5.3%의 상대 문자 오류율 (CERR)이 감소했으며, 비스트리밍 ASR의 경우 8.3%-9.7%의 상대 CERR가 감소했습니다. 이는 단일 모델에서 여러 독립 모듈과 비교하여 큰 개선을 보였습니다.



### TrojanRAG: Retrieval-Augmented Generation Can Be Backdoor Driver in Large Language Models (https://arxiv.org/abs/2405.13401)
Comments:
          18 pages, 13 figures, 4 tables

- **What's New**: 이번 연구에서는 Retrieval-Augmented Generation (RAG)에서의 공동 백도어 공격(joint backdoor attack)을 사용한 TrojanRAG를 제안했습니다. 이는 LLMs(Language Learning Models)를 보편적인 공격 시나리오에서 조작할 수 있게 만듭니다.

- **Technical Details**: TrojanRAG는 정교한 타겟 문맥(target contexts)과 트리거 세트(trigger sets)를 구성합니다. 여러 쌍의 백도어 쇼트컷(backdoor shortcuts)을 대조 학습(contrastive learning)으로 정교하게 최적화하여 트리거 조건을 매개 변수 하위 공간으로 제한함으로써 매칭을 향상시킵니다. 또 타겟 문맥에 대해 RAG의 재현율(recall)을 높이기 위해, 지식 그래프(knowledge graph)를 사용해 구조화된 데이터를 구축하여 세분화된 수준에서 하드 매칭을 달성합니다.

- **Performance Highlights**: 진실성(truthfulness), 언어 이해(language understanding), 유해성(harmfulness) 등에 대한 포괄적인 실험 결과, TrojanRAG는 보편적인 위협을 나타내면서도 정상 쿼리에 대한 검색 기능을 유지하는 것으로 나타났습니다.



### Contextualized Automatic Speech Recognition with Dynamic Vocabulary (https://arxiv.org/abs/2405.13344)
- **What's New**: 이 논문은 희귀 단어나 맥락적 구(句)에서 성능을 개선하기 위해 편향 리스트(bias list)를 사용하는 end-to-end 자동 음성 인식(E2E-ASR) 성능을 개선하는 '딥 바이어싱(Deep Biasing, DB)' 기법을 제안합니다. 구체적으로, 이 논문에서는 기존의 정적 어휘 대신 동적 어휘(dynamic vocabulary)를 사용하여 구 수준의 편향 토큰을 추가하는 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 편향 문구(bias phrases)를 미리 정의된 정적 어휘의 서브 단어 시퀀스로 처리하지 않고, 추론 단계에서 동적으로 문구 수준의 편향 토큰을 추가합니다. 각 편향 토큰은 하나의 토큰 내에서 전체 편향 문구를 나타내므로 서브 단어 간의 의존성을 학습할 필요가 없습니다. 이 방법은 일반적인 E2E-ASR 아키텍처에서 임베딩(embedding) 및 출력 층만 확장하면 되므로 다양한 아키텍처에 적용할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 영어와 일본어 데이터셋에서 편향 문구 성능을 향상시킴을 보여줍니다.



### A Survey of Robotic Language Grounding: Tradeoffs Between Symbols and Embeddings (https://arxiv.org/abs/2405.13245)
Comments:
          IJCAI 2024 Survey Track

- **What's New**: 대형 언어 모델(Large Language Models)의 발전으로 로봇이 언어를 이해하고 처리하는 능력이 이전보다 유연하고 향상되었습니다. 이번 연구 조사에서는 최근 문헌을 검토하고, 언어를 특정 의미의 형식적 표현(Formal Representation)으로 매핑하는 방법과 고차원 벡터 공간(High-dimensional Vector Spaces)으로 직접 매핑하는 두 가지 극단적인 접근 방식을 다룹니다.

- **Technical Details**: 형식적 표현을 이용한 방법은 언어의 의미를 명확하게 나타내고, 학습 문제의 크기를 제한하며, 해석 가능성과 형식적 안전 보장을 위한 틀을 제공합니다. 반면, 고차원 공간에 언어와 지각 데이터를 임베딩하는 방법은 수동으로 지정된 상징적 구조를 피함으로써 더 일반화될 수 있지만, 더 많은 데이터와 컴퓨팅 리소스를 필요로 합니다.

- **Performance Highlights**: 각 접근 방식의 장점과 단점을 논의하고, 두 접근 방식을 결합하여 최상의 결과를 얻을 수 있는 미래 연구 방향을 제시합니다.



### How Reliable AI Chatbots are for Disease Prediction from Patient Complaints? (https://arxiv.org/abs/2405.13219)
Comments:
          24th IEEE International Conference on Information Reuse and Integration (IEEE IRI 2024), San Jose, CA, USA

- **What's New**: 이번 연구는 병원 응급실에서 환자의 불만을 바탕으로 질병을 예측하는 인공지능(AI) 챗봇의 신뢰성을 조사했습니다. 연구 대상이 된 챗봇으로는 GPT 4.0, Claude 3 Opus, Gemini Ultra 1.0이 포함됩니다. 이는 AI 기반 질병 예측의 신뢰도 평가를 위한 최신 연구입니다.

- **Technical Details**: 본 연구는 few-shot learning 기법을 사용하여 이 챗봇들이 질병 예측을 얼마나 효과적으로 수행하는지 평가했습니다. 또한, transformer 기반의 모델인 BERT를 미세 조정(fine-tuning)하여 챗봇과의 성능 비교를 수행했습니다. 연구 결과, GPT 4.0는 더 많은 few-shot 데이터를 사용할수록 높은 정확성을 보였으며, Gemini Ultra 1.0는 적은 예시로도 좋은 성능을 나타냈고, Claude 3 Opus는 일관된 성능을 유지했습니다.

- **Performance Highlights**: BERT의 성능은 모든 챗봇보다 낮았으며, 이는 제한된 라벨링 데이터로 인한 문제로 해석됩니다. 하지만, 응급 의학 상황에서 채팅봇들의 정확도가 완전하지 않아 중요한 의료 결정을 내리는 데는 아직 한계가 있습니다. 이는 AI 챗봇이 인간의 전문성을 대체하기보다는 보완하여 환자의 안전을 확보하는 것이 중요하다는 점을 시사합니다. 이에 따라 AI 기반 의료 애플리케이션의 신뢰도를 높이기 위해 추가적인 연구와 개선이 필요합니다.



### Modeling Real-Time Interactive Conversations as Timed Diarized Transcripts (https://arxiv.org/abs/2405.13203)
Comments:
          GT and GA contributed equally

- **What's New**: 이 논문은 기존의 동기화된 턴 방식 대화에 국한된 챗봇 언어 모델을 넘어서, 실시간 상호작용 대화를 시뮬레이션하는 새로운 방법을 제시합니다. 이 방법은 사전 학습된 텍스트 전용 언어 모델을 사용하여 시간이 기록된 대화 (timed diarized transcripts)를 모델링하고, 인과적 거절 샘플링(causal rejection sampling)을 통해 이를 디코딩하는 방식입니다.

- **Technical Details**: 이 연구에서는 두 가지 사례 연구를 통해 방법론의 유망성을 입증하였습니다: 인스턴트 메신저 대화와 음성 대화입니다. 각 경우는 각각 약 30 토큰/초 (tokens/s) 및 20 토큰/초의 속도로 생생한 상호작용을 유지하기 위해 언어 모델을 사용합니다. 이러한 기술은 비교적 적은 데이터로 기존 언어 모델에 추가할 수 있으며, 일반적인 하드웨어에서 실행할 수 있습니다.

- **Performance Highlights**: 이 방법을 통하여, 언어 모델은 인스턴트 메신저 대화와 음성 대화에서 실시간 상호작용을 유지할 수 있는 능력을 보여주었습니다. 이는 기존의 동기화된 턴 방식 대화에서 벗어나, 실제 대화와 더욱 유사한 상호작용을 가능하게 합니다.



### Mamo: a Mathematical Modeling Benchmark with Solvers (https://arxiv.org/abs/2405.13144)
Comments:
          Project: this https URL

- **What's New**: 새로운 연구는 대형 언어 모델 (LLMs, Large Language Models)의 수학적 모델링 (mathematical modeling) 능력을 평가하기 위한 새로운 벤치마크 'Mamo'를 도입합니다. 이 벤치마크는 전통적인 결과 중심의 평가를 넘어서, LLMs의 모델링 과정 (modeling process)에 초점을 맞추고 있습니다. 이를 통해 LLMs가 문제를 해결하는 전략과 방법론에 대한 더 깊은 이해를 제공합니다.

- **Technical Details**: 기존의 방법론은 주로 수학적 문제의 최종 해답의 정확성을 평가하는데 집중했습니다. 그러나 Mamo는 LLMs가 문제를 해결하는 과정 자체를 분석하고 평가하는 새로운 패러다임을 제시합니다. 이는 LLMs의 내재된 모델링 능력을 이해하고, 보다 정교하고 포괄적인 분석을 가능하게 합니다.

- **Performance Highlights**: 이 연구는 LLMs의 수학적 모델링 능력을 더 잘 이해하기 위한 새로운 기준을 세우며, 복잡한 문제 해결 시나리오에서 LLMs의 성능을 평가하기 위한 새로운 표준을 설정합니다. 앞으로의 연구는 답변의 정확성보다는 모델링 과정 자체의 평가에 초점을 맞출 것을 제안합니다.



### Towards Retrieval-Augmented Architectures for Image Captioning (https://arxiv.org/abs/2405.13127)
Comments:
          ACM Transactions on Multimedia Computing, Communications and Applications (2024)

- **What's New**: 이번 연구는 외부 kNN (k-Nearest Neighbors) 메모리를 활용한 이미지 캡셔닝 모델 개발의 새로운 접근 방식을 제시합니다. kNN 메모리는 시각적 유사성을 기반으로 한 지식 검색 컴포넌트를 포함하고 있으며, 입력 이미지를 표현하기 위한 분화 가능한 인코더(differentiable encoder)와 외부 메모리에서 검색한 텍스트와 문맥적 단서를 기반으로 토큰을 예측하는 kNN-증강 언어 모델을 제안합니다.

- **Technical Details**: 본 연구에서는 두 가지 모델 변형을 제안하며, 이 모델들은 COCO와 nocaps 데이터셋에서 실험적으로 검증되었습니다. 시각적 특징(feature) 추출과 다중모드 연결설계의 기술적 향상을 활용하며, 외부 메모리를 통해 검색된 정보를 사용하여 캡션 생성 과정을 개선할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 특히 더 큰 검색 코퍼스(corpus)를 사용했을 때 외부 메모리를 통합한 이미지 캡션 생성 모델이 캡션의 품질을 크게 향상시킬 수 있음을 시연했습니다. 이는 검색-증강 캡션 모델이 이미지 캡셔닝의 품질 향상에 가치 있는 통찰력을 제공하며, 대규모로 이미지 캡셔닝을 개선할 수 있는 새로운 가능성을 열어줍니다.



### GPT-4 Jailbreaks Itself with Near-Perfect Success Using Self-Explanation (https://arxiv.org/abs/2405.13077)
- **What's New**: 이번 연구에서는 Iterative Refinement Induced Self-Jailbreak (IRIS)라는 새로운 접근법을 소개했습니다. IRIS는 대형 언어 모델(Large Language Models, LLMs)의 반사 능력을 활용하여 jailbreaking을 수행하는 방법으로, 기존 방법과는 달리 단일 모델을 활용해 공격자와 타겟을 동일하게 설정하는 방식입니다. 이를 통해 jailbreaking 과정을 크게 단순화할 수 있습니다.

- **Technical Details**: IRIS는 우선 스스로 설명(self-explanation)을 통해 적대적 프롬프트(adversarial prompts)를 반복적으로 보완합니다. 이 과정은 잘 정렬된(well-aligned) LLM이 적대적 지침을 따르게 만드는 데 매우 중요합니다. 그런 다음, 보완된 프롬프트를 기반으로 생성된 출력을 평가하고 해를 끼칠 가능성을 높이도록 개선합니다. 이러한 방식은 black-box 접근만으로도 가능합니다.

- **Performance Highlights**: IRIS는 GPT-4에서 98%, GPT-4 Turbo에서 92%의 jailbreaking 성공률을 7개의 쿼리(query) 이하에서 달성했습니다. 이는 이전 방식들보다 자동화된, black-box 접근만으로도 해석 가능한(interpretable) jailbreaking에서 월등히 높은 성능을 보이는 결과입니다.



### Large Language Models Can Infer Personality from Free-Form User Interactions (https://arxiv.org/abs/2405.13052)
- **What's New**: 이 연구는 대형 언어 모델(Large Language Models, LLMs)이 자유 형식의 사용자 상호작용을 통해 빅 파이브(Big Five) 성격 특성을 추론하는 능력을 조사했습니다. 결과에 따르면 GPT-4 기반 챗봇이 정적 텍스트 콘텐츠에서 추론하는 기존 접근 방식보다 성격 추론 정확성이 더 높았습니다.

- **Technical Details**: 이 연구는 다양한 대화 설정에서 GPT-4 챗봇의 성격 추론 정확성을 테스트했습니다. 챗봇이 성격 관련 정보를 유도하도록 유도된 경우 (평균 r=.443, 범위=[.245, .640]), 자연스러운 상호작용을 강조한 조건 (평균 r=.218, 범위=[.066, .373])에서 보다 높은 정확도를 보였습니다. 마지막으로, GPT-4 챗봇이 기본적으로 도움을 주는 조수 역할을 할 때는 성격 추론 정확도가 가장 낮았으나 여전히 일부 성격 특성에 대해 심리학적으로 의미 있는 정보를 캡처했습니다 (평균 r=.117, 범위=[-.004, .209]).

- **Performance Highlights**: 챗봇이 성격 평가에 직접 집중할 때 사용자 경험이 덜 긍정적으로 평가되지 않았으며, 모든 조건에서 인터랙션이 자연스럽고, 즐겁고, 흥미롭고, 인간적이라고 보고되었습니다. 초기 분석 결과 성격 추론의 정확성은 다양한 사회-인구통계적 하위 그룹 간에 크게 차이가 없었습니다.



### BERT vs GPT for financial engineering (https://arxiv.org/abs/2405.12990)
- **What's New**: 이번 연구에서는 Transformer 모델들을 벤치마킹하여 뉴스 이벤트로부터 감정을 판단하는 능력을 평가했습니다. 이러한 신호는 상품 거래를 위한 다운스트림 모델링 및 신호 식별에 활용될 수 있습니다. 특히, 파인튜닝된 BERT 모델이 GPT 모델보다 우수한 성능을 보였습니다.

- **Technical Details**: Transformer 모델은 최근 자연어 처리(NLP) 분야에서 혁신을 이룩했으며, 기계 번역, 텍스트 요약, 질의응답 및 자연어 생성과 같은 다양한 작업에서 최신 기술 성과를 보여주고 있습니다. 대표적인 Transformer 모델로는 BERT와 GPT가 있으며, 이들은 구조와 목표에 있어서 차이가 있습니다. CopBERT 모델의 학습 데이터와 프로세스 개요도 제공되었습니다.

- **Performance Highlights**: 이번 연구에서 CopBERT 모델은 유사한 도메인에 특화된 FinBERT 모델보다 뛰어난 성능을 보였습니다. 혼동 행렬(confusion matrices)을 통해 CopBERT와 CopGPT의 성능을 비교했을 때, CopBERT가 GPT4에 비해 약 10%의 f1_score 증가를, CopGPT에 비해 16% 증가를 보였습니다. 높은 예측력을 갖춘 대형 언어 모델(LLM)이 BERT 모델을 능가하지만, 재해석 가능성 및 GPT 모델의 헛소리(hallucinations) 문제를 고려할 때 재무 공학 작업에서는 BERT 모델이 흥미로운 대안이 될 수 있습니다.




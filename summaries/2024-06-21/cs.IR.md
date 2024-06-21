New uploads on arXiv(cs.CL)

### Model Merging and Safety Alignment: One Bad Model Spoils the Bunch (https://arxiv.org/abs/2406.14563)
Comments:
          Under review

- **What's New**: 이 논문은 대형 언어 모델(Large Language Models, LLMs)의 병합이 어떻게 안전성 정렬(safety alignment)에 영향을 미치는지 조사합니다. 기존 모델 병합 기술은 종종 안전성 정렬의 중요성을 간과하여 오히려 안전하지 않은 모델이 만들어지는 문제를 제기합니다. 저자들은 이 문제를 해결하기 위해 간단한 두 단계 접근법을 제안합니다: (i) 합성 안전성 및 도메인 데이터 생성, (ii) 생성된 데이터를 사용하여 병합 중 안전성을 최대화하는 최적화 과정에 통합합니다.

- **Technical Details**: 이 접근법은 두 단계로 구성됩니다. 첫 번째 단계는 합성 데이터를 생성하는 것이며, 이는 두 가지 데이터셋으로 나뉩니다. 하나는 안전성 정렬을 유지하고, 다른 하나는 도메인 지식을 전이하기 위한 것입니다. 예를 들어 '사람을 죽이는 방법은?'과 같은 악의적인 질문에 대해 안전한 답변(예: '도와드릴 수 없습니다.')을 포함한 데이터셋을 생성합니다. 두 번째 단계는 이러한 데이터를 기존 병합 기술에 통합하여 데이터 기반 병합 최적화 절차를 수행하는 것입니다. 이는 병합된 모델이 높은 안전성과 도메인 성능을 유지할 수 있게 합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 기존 모델 병합 기술보다 우수한 성능을 보였으며, 특히 안전성 정렬과 도메인 정확도의 균형을 잘 맞추는 것을 입증했습니다. 또한, 여러 조건에서 로버스트성(robustness)을 증명하는 광범위한 실험과 구성 요소별 분석을 수행하였습니다.



### Whiteboard-of-Thought: Thinking Step-by-Step Across Modalities (https://arxiv.org/abs/2406.14562)
Comments:
          Project website: this http URL

- **What's New**: 최근 아카이브 논문에서는 큰 언어 모델(Large Language Models, LLMs)이 가진 시각적 추론 문제를 해결하기 위한 새로운 접근 방법인 '화이트보드 오브 생각(Whiteboard-of-Thought, WoT) 프롬팅'을 소개합니다. 이는 모델이 코드 라이브러리(Matplotlib, Turtle 등)를 사용하여 시각적 중간 추론 단계를 이미지로 만들어내고 이를 다시 처리하는 방법을 제공합니다. 이 접근 방식은 기존의 텍스트 기반 중간 추론(Chain-of-Thought, CoT)보다 훨씬 높은 성능을 보여줍니다.

- **Technical Details**: 이 연구는 멀티모달 대형 언어 모델(MLLMs: Multimodal Large Language Models)이 문의 중간 단계를 이미지로 시각화하고 이를 다시 모델에 입력하여 더 나은 답변을 생성하는 과정을 제안합니다. 시각화를 위해 모델은 Python의 Matplotlib 및 Turtle과 같은 코드 라이브러리를 사용하여 시각적 출력을 생성합니다. 생성된 이미지는 모델의 멀티모달 입력 능력을 활용하여 추가 처리를 통해 최종 결과물을 만들어냅니다. 이 과정에서는 특별한 모듈이나 예제 없이도 모델의 기존 코딩 능력만을 활용하여 시각화를 생성할 수 있습니다.

- **Performance Highlights**: 화이트보드 오브 생각(WoT) 프롬팅은 네 개의 복잡한 자연어 과제에서 기존의 체인 오브 생각(CoT) 방식을 넘어서는 성능을 보여줍니다. GPT-4와 같은 최신 모델이 이루지 못한 정확도 0%의 문제에서도, WoT 프롬팅은 최고 92%의 정확도를 발휘했습니다. 이는 특히 시각적 및 공간적 추론이 요구되는 상황에서 두드러집니다.



### How to Compute the Probability of a Word (https://arxiv.org/abs/2406.14561)
- **What's New**: 최신 언어 모델(LLMs) 연구에서 중요한 확률 분포를 올바르게 계산하는 방법이 제시되었습니다. 특히, GPT 계열과 같이 subword 단위로 작동하는 모델에서 단어 확률을 올바르게 계산하는 방법이 강조되었습니다. 이 연구는 기존의 잘못된 확률 계산 방식이 문장 이해 및 어휘 최적화 분석 결과에 어떻게 영향을 미치는지를 실험적으로 보여줍니다.

- **Technical Details**: 언어 모델은 자연어 시퀀스에 대한 확률 분포를 정의합니다. 이러한 확률 분포는 문장 복잡도(perplexity)와 놀라움(surprisal) 계산에 중요합니다. 대부분의 최신 언어 모델은 subwords(부분 단어)로 작동하며, 이는 최적화 및 효율성 이유에서 비롯된 것입니다. 그러나 subwords는 반드시 언어의 사전에 정의된 실제 단어를 구성하는 것은 아닙니다. 확률 분포를 subword에서 문자(character)나 단어(word)로 정확하게 변환하는 것은 기술적으로 어렵습니다. 이 논문은 이러한 변환 과정에서 발생할 수 있는 문제를 다루고, 특히 초(beginning-of-word, bow) 마킹 토크나이저(tokenizers)를 사용하는 경우의 문제점을 강조합니다.

- **Performance Highlights**: 논문에서 제시된 올바른 확률 계산 방법으로 기존의 잘못된 계산 방식을 교정함으로써 문장 이해 및 어휘 최적화 분석의 결과가 달라졌음을 보여줍니다. 이는 최근 언어 모델 연구 및 사용에서 중요한 교훈을 제공합니다.



### xCOMET-lite: Bridging the Gap Between Efficiency and Quality in Learned MT Evaluation Metrics (https://arxiv.org/abs/2406.14553)
- **What's New**: 최첨단 기계 번역 평가 지표인 xCOMET의 지식 보존을 유지하면서 압축하는 방법을 연구했습니다. 효율적인 xCOMET 대안을 만들기 위해 지식 증류(distillation), 양자화(quantization), 가지치기(pruning) 기술을 활용했습니다. 특히, 효율적인 black-box 증류를 위한 새로운 데이터 수집 파이프라인도 소개했습니다.

- **Technical Details**: 지식 증류는 대형 모델의 출력을 사용하여 소형 모델을 훈련시키는 방법입니다. 양자화는 딥러닝 모델의 매개변수 및 활성화를 32/16비트에서 8, 4, 3, 2비트로 줄여, 더 적은 메모리를 차지하고 빠른 계산을 가능하게 합니다. 가지치기는 모델의 중요하지 않은 부분을 제거하는 것으로, 특정 매개변수, 매개변수 블록 또는 전체 레이어를 제거합니다. 이 방법들을 조합하여 xCOMET의 효율적인 대안을 만들었습니다.

- **Performance Highlights**: xCOMET을 양자화를 통해 최대 3배 압축하면서도 품질 저하가 없음을 확인했습니다. 또한, 지식 증류를 통해 xCOMET-lite라는 지표를 제작하였으며, 이는 xCOMET-XXL 매개변수의 2.6%만 가지고도 92.1%의 품질을 유지했습니다. 이 지표는 WMT22 데이터셋 기준으로 nhỏ규모 강력한 지표인 COMET-22와 BLEURT-20을 6.4% 능가하면서도 매개변수를 50% 적게 사용합니다.



### GraphReader: Building Graph-based Agent to Enhance Long-Context Abilities of Large Language Models (https://arxiv.org/abs/2406.14550)
Comments:
          The first four authors contributed equally, 27 pages

- **What's New**: GraphReader는 Graph 기반의 에이전트 시스템으로 긴 텍스트를 효과적으로 처리하기 위해 설계되었습니다. 긴 텍스트를 그래프로 구조화하고, 에이전트가 이 그래프를 자율적으로 탐색하여 정보를 수집한 후 답변을 생성합니다. 이 시스템은 특히 복잡하고 다중 단계의 질문에 대해 뛰어난 성능을 보입니다.

- **Technical Details**: GraphReader는 텍스트를 작은 청크로 분할하고 핵심 요소와 원자적 사실들을 추출하여 노드와 엣지로 구성된 그래프를 만듭니다. 에이전트는 사전 정의된 함수들을 이용해 그래프를 거친 후 세밀한 방식으로 탐색하며, 필요할 경우 노드를 세부적으로 분석하거나 이웃 노드로 이동합니다. 탐색 과정에서 에이전트는 지속적으로 새로운 통찰을 기록하고 분석하여 최적의 정보를 수집합니다.

- **Performance Highlights**: GraphReader는 LV-Eval 데이터셋에서 4k 컨텍스트 윈도우를 사용하여 16k에서 256k까지의 컨텍스트 길이에서 GPT-4-128k를 큰 차이로 능가했습니다. 네 가지 도전적인 싱글 홉 및 멀티 홉 벤치마크에서도 뛰어난 성능을 입증했습니다.



### Connecting the Dots: LLMs can Infer and Verbalize Latent Structure from Disparate Training Data (https://arxiv.org/abs/2406.14546)
- **What's New**: 이번 논문은 대형 언어 모델(LLMs)이 훈련 데이터에서 위험한 정보를 검열하는 방식으로 안전 문제를 해결하는 방법에 대한 것입니다. 하지만 숨겨진 정보는 여러 문서에 암시된 형태로 남아 있을 수 있습니다. LLM이 이러한 암시를 통해 숨겨진 정보를 도출할 수 있는지 조사했습니다. 이를 위해 inductive out-of-context reasoning (OOCR)라는 새로운 개념을 연구했습니다.

- **Technical Details**: OOCR은 LLM이 훈련 중에 문서들에 분산된 암시적 증거를 통해 잠재된 정보를 추론하고, 이를 하위 작업에 적용하는 능력을 의미합니다. 다양한 문서에서 암시된 '시티 50337'이 파리라는 사실을 LLM이 도출할 수 있는 예제를 포함하여, 총 5가지 테스트를 통해 이 능력을 검증했습니다. 모델은 훈련 중에만 정보를 접하고, 테스트 시에는 직접적인 예제가 제공되지 않는 상황에서 성능을 평가했습니다.

- **Performance Highlights**: GPT-3.5와 GPT-4는 모두 OOCR에서 성공적인 결과를 보였으며, GPT-4가 더 강력한 성능을 나타냈습니다. 예를 들어, 모델이 동전 던지기의 편향 여부를 판별하거나 함수의 역함수를 계산할 수 있는 능력을 보였습니다. 하지만 작은 LLM이나 복잡한 구조에서는 성능이 불안정한 경향이 있었습니다. 이는 LLM의 지식을 모니터링하고 제어하는데 어려움을 초래할 수 있다는 점에서 AI 안전성 측면에서 중요한 시사점을 제공합니다.



### Unmasking Database Vulnerabilities: Zero-Knowledge Schema Inference Attacks in Text-to-SQL Systems (https://arxiv.org/abs/2406.14545)
- **What's New**: 새롭게 등장한 연구는 텍스트-투-SQL(text-to-SQL) 모델이 사용하는 데이터베이스 스키마 요소를 추출하는 방법을 제안합니다. 이를 통해 SQL 인젝션 공격(SQL injection attacks)과 같은 보안 위협이 더 쉬워질 수 있으며, 이 연구는 특히 접근 없이 데이터베이스 스키마를 탐색할 수 있는 제로 지식 프레임워크(zero-knowledge framework)를 개발하였습니다.

- **Technical Details**: 연구에서는 특별히 제작된 질문을 통해 데이터베이스 구조를 알아내는 제로 지식 프레임워크를 이용하여 텍스트-투-SQL 모델을 조사합니다. 다양한 데이터셋과 모델(파인튜닝된 모델과 생성형 언어 모델)에 대해 프레임워크를 평가합니다. 파인튜닝된 모델(fine-tuned models)은 텍스트-SQL 쌍에 맞게 특화된 훈련을 받고, 생성형 언어 모델(generative language models)은 대규모 데이터 코퍼스에서 학습하여 SQL 문의 생성을 쉽게 해줍니다.

- **Performance Highlights**: 파인튜닝된 모델에서 테이블 이름을 0.75의 F1 점수로 재구성할 수 있었으며, 생성형 모델에서는 0.96의 F1 점수를 달성하였습니다. 이를 통해 우리의 시스템이 특정 텍스트-투-SQL 모델 유형과 독립적으로 데이터베이스 스키마를 재생성할 수 있음을 입증했습니다.



### Investigating Mysteries of CoT-Augmented Distillation (https://arxiv.org/abs/2406.14511)
Comments:
          Draft; under review

- **What's New**: 이번 연구는 '사고의 연쇄(Chain of Thought, CoT)' 추론이 모델 디스틸레이션(model distillation)에 어떻게 그리고 왜 도움이 되는지를 조사하였습니다. 구체적으로, CoT 시퀀스를 라벨 뒤에 배치했을 때, 테스트 시 학생 모델이 reasoning을 생성할 필요 없이도 일관되게 더 좋은 성능을 보인다는 것을 발견했습니다. 또한, CoT 시퀀스의 의미 연관성 없이도 성능 향상이 유지되며, 중요한 몇몇 토큰만으로도 전체 이유 시퀀스를 사용할 때와 동일한 성능 향상을 이끌어낼 수 있다는 점을 확인했습니다.

- **Technical Details**: CoT 증강 디스틸레이션(CoT-augmented distillation)은 대규모 'teacher' 모델(GPT-4 등)로부터 추론 시퀀스를 추출하고, 이를 target label과 함께 작은 'student' 모델을 미세 조정하는 데 사용합니다. 이번 연구는 GPT-2, Phi-1.5, Gemma-2B와 같은 작은 학생 모델을 대상으로 실험을 진행하여 CoT 시퀀스 배치의 효과를 비교했습니다. 실험 과정에서는 NVIDIA A100 GPU 두 대를 사용하여 모든 모델을 학습시켰습니다.

- **Performance Highlights**: CoT 시퀀스를 라벨 뒤에 배치했을 때 학생 모델의 성능이 지속적으로 향상되었습니다. 인과 관계나 토큰 순서 없이도 성능 향상이 유지되었으며, 몇 가지 주요 토큰만으로도 완전한 이유 시퀀스와 비슷한 성능 향상을 이끌어낼 수 있었습니다. 이러한 결과는 다양한 모델 및 데이터셋에서 일관되게 나타났습니다.



### Evidence of a log scaling law for political persuasion with large language models (https://arxiv.org/abs/2406.14508)
Comments:
          16 pages, 4 figures

- **What's New**: 최신 연구는 대규모 언어 모델(LLMs)이 인간이 작성한 것만큼 설득력 있는 정치 메시지를 생성할 수 있다는 점에서 주목할 만한 결과를 제시합니다. 연구진은 24개의 서로 다른 크기의 언어 모델을 사용하여 10가지 미국 정치 이슈에 관한 720개의 메시지를 생성했습니다. 그리고 이를 통해 여러 가지 규모의 언어 모델이 얼마나 설득력 있는지 평가했습니다.

- **Technical Details**: 연구는 다양한 크기의 오픈 웨이트(open-weight) 변환기 기반 언어 모델을 평가하여, 이들 모델이 정치적 설득력에서 어느 정도 크기와 상관관계를 가지는지 분석했습니다. 중요한 발견 중 하나는 로그 스케일링 법칙(log scaling law)을 발견한 것입니다. 즉, 모델 크기가 커질수록 설득력은 급격히 감소하는 수익을 보이며, 현재 최첨단 모델들은 크기가 훨씬 작은 모델들에 비해 설득력이 약간만 더 높다는 것입니다.

- **Performance Highlights**: 연구 결과, 평균적으로 언어 모델은 사람들의 태도를 5.77 퍼센트 포인트 변화시킬 수 있었습니다. 로그 스케일링 법칙에 따라, 모델의 파라미터 수가 1 유닛 증가할 때마다 평균 처리 효과가 1.26 퍼센트 포인트 증가했습니다. 하지만, 최첨단 모델인 Claude-3-Opus나 GPT-4-Turbo는 훨씬 작은 모델인 Qwen1.5-7B와 비교했을 때 큰 설득력 차이를 보이지 않았습니다.

- **Policy Implications**: 이 연구는 정책 결정자와 연구자에게 현재와 가까운 미래의 LLM이 공공 여론과 정치 행동에 미치는 잠재적 영향력을 평가할 수 있는 증거를 제공합니다. 설득력 있는 메시지 생성 능력의 한계가 있을 수 있어 추가적인 모델 크기 확대가 큰 효과를 보지 못할 가능성을 시사합니다.



### Translating Across Cultures: LLMs for Intralingual Cultural Adaptation (https://arxiv.org/abs/2406.14504)
- **What's New**: 이 논문은 대형 언어 모델(LLM)이 문화 적응(cultural adaptation) 작업에서 어떻게 성능을 발휘하는지 분석하고 평가하기 위한 프레임워크를 제공합니다. 문화 적응이란 소스 텍스트의 문화적 요소를 타겟 문화에 맞게 수정하는 것을 의미합니다. 논문은 LLM이 다양한 문화 간 지식을 바탕으로 텍스트를 얼마나 효과적으로 변형할 수 있는지 평가합니다.

- **Technical Details**: 문서에서는 텍스트 생성, 언어 이해, 번역 등의 작업에 있어 강력한 성능을 보이는 LLM의 문화적 적응 능력을 분석합니다. 소스 텍스트를 타겟 문화에 맞게 변형하는 작업을 정의하고, 이를 평가하기 위한 프레임워크를 설계합니다. 연구는 주로 영어(내언적 적응) 소스 및 타겟 청중을 대상으로 하며, 다이얼로그(dialog) 코퍼스를 이용해 타겟 문화로 변형된 버전을 생성합니다.

- **Performance Highlights**: LLM은 특정한 인스트럭션을 기반으로 문화를 반영한 텍스트 변형 작업에서 사용될 수 있으며, 문장 단위가 아닌 다이얼로그 기반의 적응이 더 풍부한 컨텍스트를 제공한다고 평가받고 있습니다. 또한, 연구는 LLM이 문화적 편견이나 고정관념을 재생산할 가능성을 분석하고 이에 대한 문제도 탐구합니다.



### Overview of the CAIL 2023 Argument Mining Track (https://arxiv.org/abs/2406.14503)
- **What's New**: CAIL 2023 Argument Mining Track에 대한 자세한 개요를 제공합니다. 이 트랙의 주요 목표는 재판 대화에서 상호 작용하는 논증 쌍을 식별하고 추출하는 것입니다. 주로 요약된 판결 문서를 사용하지만 재판 녹음을 참조할 수도 있습니다. 새로운 데이터세트 CAIL2023-ArgMine이 도입되었으며, 다양한 원인으로 인해 주석이 달린 새로운 사례가 포함되어 있습니다.

- **Technical Details**: 이 트랙은 두 단계로 구성됩니다. 첫 번째 단계에서는 원고의 주어진 논증과 상호 작용하는 방어의 논증을 다섯 후보 중에서 선택해야 합니다. 두 번째 단계에서는 판결 문서에서 직접 상호 작용하는 논증 쌍을 추출해야 합니다. CAIL2023-ArgMine 데이터세트는 원래 데이터세트보다 두 배 크며, 민사 사건을 포함한 다양한 원인으로 인해 더욱 확대되었습니다.

- **Performance Highlights**: 최고의 제출물은 모두 언어 모델을 기반으로 하지만, 추가적인 성능 향상을 위해 다양한 접근 방식을 통합했습니다. 재판 녹음과 같은 멀티모달 데이터를 보너스 과제로 추가했으며, 이 데이터는 특정 문서에 대해 제공되고 있습니다.



### Improving Expert Radiology Report Summarization by Prompting Large Language Models with a Layperson Summary (https://arxiv.org/abs/2406.14500)
- **What's New**: 본 논문은 방사선 보고 요약(Radiology Report Summarization, RRS)을 향상시키기 위한 새로운 프롬프트 전략을 소개합니다. 이 접근법은 비전문가 요약(layperson summary)을 먼저 생성하여 핵심 관찰을 정상화(normalize)하고, 의사-환자 간의 소통 기술을 모방하여 복잡한 정보를 단순화합니다. 이를 통해 모델이 일반 용어를 구체적인 소견과 연결하는 능력을 향상시킵니다.

- **Technical Details**: 논문에서는 인-컨텍스트 학습(few-shot in-context learning)과 결합된 프롬프트 전략을 사용합니다. 이 전략은 먼저 비전문가 요약을 생성하여 키 관찰을 정상화하고, 이를 통해 복잡한 정보를 단순화합니다. 일반적인 텍스트 코퍼스에 훈련된 대형 언어 모델(Large Language Models, LLMs)의 경우, 전문적인 지식이 부족할 수 있는데, 이 부분을 프롬프트를 통해 보완합니다. 데이터셋은 MIMIC-CXR, CheXpert, MIMIC-III를 사용하였고, Meta-Llama-3-8B-Instruct와 같은 최신 오픈소스 LLMs와의 비교가 이루어졌습니다.

- **Performance Highlights**: 본 접근법은 요약 정확도와 접근성 측면에서 개선을 보였습니다. 특히 도메인 외 테스트에서 일부 메트릭에서 최대 5% 향상을 달성했습니다. 의사-환자 소통 기술을 이용한 비전문가 요약 생성이 전문가 요약의 일관성과 정확성을 높이는 데 효과적임을 입증했습니다.



### LLaSA: Large Multimodal Agent for Human Activity Analysis Through Wearable Sensors (https://arxiv.org/abs/2406.14498)
Comments:
          Under review at ARR (for EMNLP 2024)

- **What's New**: 이번 연구에서는 IMU(Inertial Measurement Units)를 대형 언어 모델(LLMs)과 통합하여 인간 활동 이해를 향상시키는 멀티모달 AI를 발전시키고자 합니다. 이를 위해 26,288개의 IMU 기반 활동 내러티브를 포함한 'SensorCaps' 데이터셋과 257,562개의 질의응답 쌍을 포함한 'OpenSQA' 데이터셋을 소개합니다. 이를 통해 Llama와 LIMU-BERT를 결합한 LLaSA라는 대형 멀티모달 에이전트를 개발하였습니다.

- **Technical Details**: LIMU-BERT는 IMU 데이터를 사용하여 자가 지도 학습(self-supervised learning)을 통해 훈련된 모델로, 인간 활동 인식 정확도를 크게 향상시켰습니다. LLaSA는 LIMU-BERT와 Llama 언어 모델을 통합하여 IMU 데이터를 처리하고 질문에 응답할 수 있는 멀티모달 에이전트입니다. 이 모델은 자연어 질의와 IMU 데이터를 결합하여 활동 분석 및 모션 해석 관련 질의에 대한 답변을 제공합니다.

- **Performance Highlights**: LLaSA는 활동 분류와 질의응답에서 탁월한 성능을 보여줍니다. 특히, 의료, 스포츠 과학 및 인간-컴퓨터 상호작용 분야에서 큰 잠재력을 가지고 있습니다. 새로운 벤치마크 데이터셋을 통해 닫힌 형태의 인간 활동 분류 및 열린 형태의 질의응답에서 압도적인 성능을 입증하였습니다.



### Instruction Pre-Training: Language Models are Supervised Multitask Learners (https://arxiv.org/abs/2406.14491)
- **What's New**: 최근 언어 모델(이하 LMs)의 성공 뒤에는 비지도 멀티태스크 사전 학습(unsupervised multitask pre-training)이 중요한 역할을 했습니다. 하지만 이 논문에서는 지도 멀티태스크 학습을 통해 사전 학습의 가능성을 탐구하며, Instruction Pre-Training이라는 프레임워크를 제안합니다. Instruction Pre-Training은 거대한 원시 코퍼스(raw corpora)에 명령-응답 쌍(instruction-response pairs)을 스케일링하여 추가합니다.

- **Technical Details**: Instruction Pre-Training에서는 원시 텍스트에 명령 - 응답 쌍을 추가한 뒤, 이를 기반으로 LMs를 사전 학습합니다. 이는 Instruction Synthesizer라는 효율적인 명령 생성기를 통해 이루어지며, 이 생성기는 공개된 모델을 기반으로 구축되었습니다. 총 200M 개의 명령-응답 쌍을 40개 이상의 작업 범주에 걸쳐 생성하여, 사전 학습의 효과를 실험했습니다. Instruction Synthesizer는 다양한 데이터셋을 기반으로, 원시 텍스트와 명령-응답 쌍으로 구성된 예제를 사용해 파인 튜닝(fine-tune)하여 높은 다양성과 정확성을 보장합니다.

- **Performance Highlights**: 실험 결과, Instruction Pre-Training은 기존 사전 학습된 모델을 일관되게 향상시키며, 추가적인 명령 튜닝에서도 큰 이점을 제공합니다. 도메인 적응형 지속 사전 학습(domain-adaptive continual pre-training)에서도 성능 향상이 확인되었으며, Llama3-8B 모델이 Llama3-70B 모델과 비교해 비슷하거나 더 우수한 성능을 보였습니다.



### Explicit and Implicit Large Language Model Personas Generate Opinions but Fail to Replicate Deeper Perceptions and Biases (https://arxiv.org/abs/2406.14462)
- **What's New**: LLMs(대규모 언어 모델)이 인간 중심의 사회과학 연구 과제에서 점점 더 많이 사용되고 있습니다. 이 논문에서는 LLM을 인간같은 퍼소나(persona)로 설정하고 특정 인간처럼 대답하도록 유도하는 방법을 연구합니다. 퍼소나 설정은 명시적인 demographics, 정치적 신념, 생활 경험을 통해 이루어질 수 있으며, 특정 인구 집단에서 흔한 이름을 통해 암묵적으로도 가능합니다.

- **Technical Details**: LLM 퍼소나는 두 가지 과제(task)를 수행합니다. 먼저 주관적 주석 작업(annotation task)에서는 독성이 탐지되고, 신념 생성 작업(belief generation task)에서는 인간 요인에 따라 변하는 신념이 생성됩니다. 퍼소나는 명시적인 '당신은 78세 여성입니다' 또는 암묵적인 '당신의 이름은 Ethel입니다'와 같은 방법으로 설정됩니다. 이에 따라 LLM이 어떤 인간 요인을 인식하고 반응하는지 조사합니다.

- **Performance Highlights**: 연구 결과, LLM 퍼소나는 인간의 편향성을 어느 정도 재현하지만, 대체로 암묵적 편향성을 나타내지 못했습니다. 이는 LLM이 인간 사고의 내재적 인지 메커니즘을 결여하고 있음을 시사하며, 복잡한 사회과학 응용에서의 효율성을 제한할 수 있습니다.



### Healing Powers of BERT: How Task-Specific Fine-Tuning Recovers Corrupted Language Models (https://arxiv.org/abs/2406.14459)
- **What's New**: 이번 연구는 BERT와 같은 사전 훈련된 언어 모델의 매개변수 손상이 성능에 미치는 영향을 조사하며, 특정 손상 후 튜닝을 통해 성능을 얼마나 회복할 수 있는지를 살펴봅니다. 이를 통해 언어 모델의 견고성과 적응력을 이해하려는 시도를 합니다.

- **Technical Details**: 연구진은 BERT 모델의 내부 매개변수를 의도적으로 손상시키고, 이를 다시 튜닝하여 원래 성능으로 회복하는 방식을 채택했습니다. LayerNorm 계층과 비-LayerNorm 계층의 가중치를 각각 일정 값(1.0)으로 설정하고, Kaiming 초기화 기법을 이용해 재초기화했습니다. 실험은 세 가지 규모의 BERT 모델 (BERT-Base, BERT-Large, DistilBERT)과 6개의 분류 데이터셋(SST-2, IMDB, AG News, Emotion, DBPedia, Twitter Financial News Topic)을 대상으로 진행되었습니다.

- **Performance Highlights**: 실험 결과, 손상된 BERT 모델은 튜닝을 거쳐도 원래의 성능을 완전히 회복하지는 못했습니다. 모델 손상 비율이 높아질수록 성능 저하가 더욱 두드러졌습니다. 특히, 하위 계층 손상이 상위 계층 손상보다 성능에 더 치명적인 영향을 미쳤습니다. 이러한 결과는 BERT의 하위 계층이 중요한 언어적 특징을 포착하는 데 더 중요하다는 점을 시사합니다.



### Towards Truthful Multilingual Large Language Models: Benchmarking and Alignment Strategies (https://arxiv.org/abs/2406.14434)
Comments:
          15 pages

- **What's New**: 본 연구는 다언어 대형 언어 모델(MLLM)의 진실성을 평가하고 개선하는 새로운 벤치마크와 방법을 제안합니다. 기존 연구는 주로 영어에 집중되어 있어 다양한 언어에서의 성능 격차가 큽니다. 이를 해결하기 위해 MTruthfulQA라는 벤치마크를 구축하고, Fact-aware Multilingual Selective Synergy (FaMSS)라는 새로운 접근법을 통해 다언어 간의 사실 정렬을 개선합니다.

- **Technical Details**: FaMSS는 다언어 데이터 간의 선택적 협업을 통해 데이터 할당을 최적화하는 방법입니다. 이 방법은 특정 작업에 국한되지 않고 쌍방향 언어 데이터를 사용하여 다언어 정렬을 수행합니다. 또한, Language Bias Probe를 도입하여 각 언어의 영향을 분석하고, 이를 바탕으로 데이터 할당 전략을 최적화합니다.

- **Performance Highlights**: 실험 결과, FaMSS를 사용한 모델은 다국어 질문-응답 작업에서 진실성이 더욱 향상된 성능을 보였습니다. 이 방법은 또한 다언어 표현 재현 간의 격차를 효과적으로 줄여줍니다. 진실성 평가를 위한 벤치마크 MTruthfulQA는 영어를 포함한 총 9개의 언어에 대해 동일한 질문 세트를 포함하고 있어 공평한 평가가 가능합니다.



### SynDARin: Synthesising Datasets for Automated Reasoning in Low-Resource Languages (https://arxiv.org/abs/2406.14425)
- **What's New**: 이번 연구는 저자들이 낮은 자원 언어로 다국어 질문 응답(QA) 데이터셋을 생성하고 검증하는 새 방법인 SynDARin을 제안한 것입니다. English 외의 언어에서는 QA 데이터셋이 부족한데, 이는 수집과 수작업 주석의 비용과 어려움 때문입니다. SynDARin은 병렬 컨텐츠 마이닝을 사용하여 English와 대상 언어 간의 사람에 의해 curated된 문단을 확보하고 이를 기반으로 multiple-choice(MC) 질문-응답 쌍을 생성하고 자동 번역하여 품질을 검증합니다. 이를 통해 콘텐츠 품질을 유지하고 사실 오류의 가능성을 줄이며, 비용이 많이 드는 주석의 필요성을 회피할 수 있습니다. 이 방법을 테스트하기 위해, 저자들은 1.2K의 데이터를 가진 아르메니아어 QA 데이터셋을 생성하였으며, 인간 평가 결과 98%의 생성된 English 데이터가 질문 유형 및 주제의 다양성을 유지함을 보였습니다. 번역 검증 파이프라인은 품질이 낮은 데이터의 약 70%를 걸러낼 수 있었습니다.

- **Technical Details**: 이 방법은 병렬 English 및 아르메니아어 서론 문단을 다양한 위키피디아 문서에서 마이닝하여 시작됩니다. English 데이터는 단락 내에서 명시적으로 언급된 답을 가지고 질문-응답 쌍을 생성합니다. 인간 평가를 거쳐 품질을 확인한 후, 자동 도구로 번역하고 병렬 아르메니아어 문단에서 답 부분 문자열 및 의미 매칭을 통해 추가 검증과정을 거칩니다. 이를 통해 할루시네이션, 편향, 일관성 문제를 줄입니다.

- **Performance Highlights**: 저자들은 아르메니아어로 된 이 데이터셋을 만들어 zero-shot, few-shot, fine-tuned 모드에서 여러 최신 LLM들을 평가하였습니다. 이 데이터셋으로 인해 모델이 인간의 정확도에 도달하지 못함을 보여주었으며, 이는 생성된 데이터셋이 단순하지 않으며 낮은 자원 언어에서 모델 성능을 측정하기 위한 유용한 자원이 될 수 있음을 시사합니다.



### SEC-QA: A Systematic Evaluation Corpus for Financial QA (https://arxiv.org/abs/2406.14394)
- **What's New**: 금융 도메인의 실세계 작업을 반영한 데이터셋의 부족은 금융 데이터 분석 자동화에서 지속적으로 문제로 제기되었습니다. 이에 대응하여, 금융 문서에서 다중 문맥을 반영하는 질문-답변(QA) 쌍을 반자동으로 생성할 수 있는 프레임워크, SEC-QA를 제안합니다. 이 프레임워크는 최신 공공 문서 컬렉션을 사용하여 지속적으로 데이터셋을 갱신할 수 있는 기능을 제공합니다.

- **Technical Details**: SEC-QA 프레임워크는 다음과 같은 두 가지 주요 기능을 갖추고 있습니다: 1) 실제 금융 시나리오를 더 잘 반영하는 다중 문맥 금융 문서에서 QA 쌍을 반자동으로 생성하는 기능, 2) 대규모 언어 모델(LLM)이 아직 훈련되지 않은 최신 공공 문서 컬렉션을 사용하여 데이터셋을 지속적으로 갱신하는 기능. 실험 결과, 현재의 검색 보강 생성(RAG) 방식은 이러한 복잡한 다중 문서 질문에 체계적으로 답변하지 못하며, 이에 대응하여 'Program-of-Thought' 접근 방식을 기반으로 한 QA 시스템을 도입하여 복잡한 정보 검색 및 정량적 추론 파이프라인 성능을 향상시킵니다.

- **Performance Highlights**: SEC-QA 프레임워크로 생성된 질문에 대해 기존의 RAG 기반 시스템이 체계적으로 실패하는 것으로 나타났으며, 새로운 LLM들이 문서 컬렉션 구조를 효과적으로 탐색할 수 있는 코드 사용을 통해 성능이 크게 향상되었습니다. 예를 들어, 특정 회사의 연평균 실적 정보가 10-K 보고서에 포함되어 있음을 파악하고, 해당 회사와 회계 연도별로 10-K 보고서를 검색하는 기능을 구현하여 QA 정확도를 높일 수 있습니다.



### Exploring Spatial Representations in the Historical Lake District Texts with LLM-based Relation Extraction (https://arxiv.org/abs/2406.14336)
- **What's New**: 본 연구는 영국 호수 지구(English Lake District)의 역사적 내러티브에서 공간적 관계를 추출하기 위해 Generative Pre-trained Transformer (GPT) 모델을 사용하여, 역사적 텍스트 내의 공간적 차원을 포괄적으로 이해하고자 합니다. 결과는 의미적 삼중항(semantic triples)의 형태로 제시되며, 이를 통해 개체와 위치 간의 미묘한 연결을 시각적으로 네트워크로 표현합니다.

- **Technical Details**: 연구는 주로 Corpus of the Lake District Writing (CLDW)를 사용하여, 'near'와 같은 공간적 관계를 추출하는데 집중합니다. 데이터는 Named Entity Recognition (NER) 모듈을 사용해 장소명과 지리적 명사를 추출하며, GPT-4 모델에 프롬프트로 입력하여 의미적 삼중항을 생성합니다. 결과는 적합성 정밀도(precision)를 통해 비교 후, 맞는 삼중항을 네트워크 시각화 모듈을 통해 그래픽으로 표현합니다.

- **Performance Highlights**: 이 연구는 광범위한 텍스트 데이터를 사전 학습한 GPT 모델을 사용하여, 탑재된 예제를 요구하지 않고도 역사적 텍스트 내 공간적 관계를 성공적으로 추출할 수 있음을 입증합니다. 이는 다양한 역사적 문맥에서 공간적 관계를 밝히는 새로운 접근법을 제시합니다.



### Self-supervised Interpretable Concept-based Models for Text Classification (https://arxiv.org/abs/2406.14335)
- **What's New**: 이 논문에서는 텍스트 데이터를 위한 새로운 해석 가능한 개념 임베딩 모델(ICEMs)을 제안합니다. 기존의 콘셉트 기반 모델을 텍스트 도메인에 적용하여, 인간이 이해할 수 있는 고수준 개념으로 중간 표현을 만듭니다. 이 모델은 셀프 슈퍼바이즈드(self-supervised) 방식으로 훈련될 수 있어, 사람이 개입할 수 있는 해석 가능성과 통제 가능성을 제공합니다.

- **Technical Details**: ICEMs는 Transformer 기반의 LLMs와 결합하여 텍스트 데이터의 고수준 개념을 예측합니다. 전통적인 엔드-투-엔드 블랙박스 모델과 달리, 개념 인코더와 태스크 인코더로 나뉘어 훈련됩니다. 개념 인코더는 입력 텍스트를 고수준 개념으로 매핑하고, 태스크 인코더는 이러한 개념을 사용해 최종 예측을 만듭니다. 이는 해석 가능성과 상호작용성을 높입니다.

- **Performance Highlights**: 실험 결과, ICEMs는 완전 슈퍼바이즈드 개념 기반 모델과 엔드-투-엔드 블랙박스 모델과 유사한 성능을 셀프 슈퍼바이즈드 방식으로 달성합니다. 또한 ICEMs는 해석 가능하고(interpretable), 인간이 중간 예측을 수정하게 해주며(interactable), LLM의 디코딩 과정을 특정 의사결정 경로를 따르도록 유도할 수 있는(controllable) 장점을 제공합니다.



### medIKAL: Integrating Knowledge Graphs as Assistants of LLMs for Enhanced Clinical Diagnosis on EMRs (https://arxiv.org/abs/2406.14326)
- **What's New**: 전자 의료 기록(EMRs)의 복잡성과 정보 중복 문제를 해결하기 위해 medIKAL이라는 프레임워크가 제시되었습니다. 이 프레임워크는 큰 언어 모델(LLMs)과 지식 그래프(KGs)를 결합하여 진단 능력을 향상시킵니다. medIKAL은 질병 후보군을 정밀하게 로컬라이징하기 위해 의료 기록의 엔티티에 가중치 중요도를 부여하고, 잔여 네트워크와 유사한 접근 방식을 활용하여 초기 진단을 KG 검색 결과와 통합합니다. 또한 경로 기반 재랭킹 알고리즘과 빈칸 채우기 스타일의 프롬프트 템플릿을 사용하여 진단 프로세스를 정밀하게 합니다.

- **Technical Details**: medIKAL은 의료 기록의 엔티티 타입에 따라 가중치를 부여해 질병 후보군을 정밀하게 로컬라이징 합니다. 잔여 네트워크와 유사한 방식을 적용하여 LLM이 먼저 자체 진단을 한 후, 이 결과를 KG 검색 결과와 병합합니다. 또한 경로 기반 재랭킹 알고리즘과 빈칸 채우기 스타일의 프롬프트 템플릿을 사용하여 진단 과정을 더 정밀하게 만듭니다. 이를 통해 EMR의 중복 정보를 요약하고 핵심 정보를 추출합니다.

- **Performance Highlights**: 새로운 공개 소스 중국어 EMR 데이터셋을 사용한 실험을 통해 medIKAL의 유효성을 검증했습니다. 이 프레임워크는 실제 임상 진단 시 정확성과 효율성을 향상시키는 잠재력을 보여줍니다.



### Mind the Privacy Unit! User-Level Differential Privacy for Language Model Fine-Tuning (https://arxiv.org/abs/2406.14322)
- **What's New**: 대형 언어 모델(LLMs)의 미세 조정 시 사용자 수준의 차등 프라이버시(User-level Differential Privacy, DP)를 보장하는 방법에 대한 체계적인 평가를 수행하였습니다. 기존 연구들은 각 예제를 프라이버시 단위로 간주하는 예제 수준의 DP를 주로 다루었으나, 이는 사용자 프라이버시 보장이 균일하지 않다는 문제를 초래할 수 있습니다. 이번 연구는 사용자 데이터가 여러 개의 예제로 구성될 때도 균일한 프라이버시 보호를 제공하는 사용자 수준의 DP를 집중적으로 분석하였습니다.

- **Technical Details**: DP를 통해 모델이 특정 데이터 없이 거의 구분할 수 없게 만드는 방법을 도입하였습니다. 연구에서는 그룹 프라이버시(Group Privacy)와 사용자별 DP-SGD(User-wise DP-SGD)라는 두 가지 주요 메커니즘을 분석하였습니다. 데이터 선택 전략과 매개 변수 조정을 통한 최적의 프라이버시-유틸리티(trade-off) 트레이드오프를 평가하였으며, 최고 수준의 초과율을 달성하는 Asi & Liu(2024)의 알고리즘을 실제 사용자 데이터 적용 시 어떤 성능을 보이는지 연구하였습니다. 이 연구는 LLM의 자연어 생성 작업에 적용한 최초의 사용자 수준 DP 연구입니다.

- **Performance Highlights**: 그룹 프라이버시와 사용자별 DP-SGD 메커니즘의 성능을 비교하여, 각 방법의 데이터 선택 전략에 따른 프라이버시-유틸리티 트레이드오프 개선 방안을 체계적으로 평가하였습니다. Asi & Liu(2024)의 알고리즘을 적용한 결과, 이론적 가정이 실세계 사용자 수준 DP 응용에 적합하지 않을 수 있음을 발견하였습니다. 이를 통해 연구 커뮤니티가 보다 현실적인 가정을 고려할 필요성을 제기하였습니다.



### Identifying User Goals from UI Trajectories (https://arxiv.org/abs/2406.14314)
- **What's New**: 사용자와 상호작용하는 GUI (Graphical User Interface)에 적용 가능한 자율 에이전트를 개선하기 위해, 사용자 행동과 GUI와의 상호작용을 통해 사용자 의도를 이해하려는 목표 식별(task identification) 작업이 제안되었습니다. 특히, 이번 연구는 사용자의 UI 상호작용 경로(trajectories)를 통해 목표를 유추하는 것을 목표로 하며, 새로운 평가 지표를 통해 두 작업 설명이 UI 환경 내에서 동의어로 평가되는지 확인합니다.

- **Technical Details**: 이번 연구에서는 UI 자동화(UI automation) 문제를 역으로 접근하여 목표 식별 작업으로 전환합니다. 연구는 웹 환경과 Android 환경에서 실험을 수행하며, Mind2Web과 Android-In-The-Wild 데이터 세트를 활용했습니다. 연구의 주요 기여는 다음과 같습니다: (1) UI 경로로부터 목표 식별 작업을 다룬 최초의 연구라는 점, (2) 새로운 수동 및 자동 평가 방법론을 소개, (3) 인간 및 최신 모델의 성능을 평가하고 분석합니다. 특히, 인간 평가자와 GPT-4, Gemini 1.5 Pro의 성능을 비교한 점이 주목됩니다.

- **Performance Highlights**: 실험 결과, Gemini 1.5 Pro가 GPT-4보다 나은 성능을 보였으나 인간 평가자에 비해 여전히 부족한 성과를 보였습니다. 이는 여전히 많은 개선의 여지가 있음을 시사합니다. 새로운 평가 방법론은 목표 설명이 UI 환경에서 상호 파라프레이즈(paraphrases)로 간주되는지 평가하는 데 중점을 둡니다. 또한, LMM(Large Multimodal Model)을 만족도 기준 자동 평가자로 활용했으며, 이 자동 평가가 대체로 인간 평가자와 일치함을 확인했습니다.



### Robust Few-shot Transfer Learning for Knowledge Base Question Answering with Unanswerable Questions (https://arxiv.org/abs/2406.14313)
- **What's New**: 이번 논문은 실제 KBQA(Knowledge Base Question Answering) 응용에서 중요시되는 두 가지 요건, 즉 답변 가능 여부를 구별하는 '강인성'과 큰 학습 데이터를 필요로 하지 않는 '저자원성'을 동시에 충족시키는 새로운 과제를 제시합니다. 이를 위해 저자들은 답변 불가 질문을 포함한 KBQA의 few-shot 전이 학습(few-shot transfer)을 제안하며, 현존 최고 수준(SOTA) 모델인 FuSIC-KBQA를 확장한 FUn-FuSIC이라는 모델을 소개합니다.

- **Technical Details**: FUn-FuSIC는 LLM을 반복적으로 유도하여 질문에 대한 논리 형식을 생성하게 하고, 구문적, 의미적, 실행 가이드 검사를 활용하여 피드백을 제공함으로써 성능을 향상시킵니다. 또한 self-consistency를 활용해 LLM이 논리 형식에 대한 확신도를 평가하고 답변 가능 여부를 판단합니다. 예컨대, 논리 형식의 확신도가 낮을 경우 이를 명확히 하여 질문이 답변 불가 상태임을 감지하는 메커니즘을 도입했습니다.

- **Performance Highlights**: 실험 결과, FUn-FuSIC는 새롭게 구성된 데이터셋을 기반으로 수행된 테스트에서 기존의 SoTA 모델과 그 변형보다 우수한 성능을 보였습니다. 특히, 다양한 오류 검사의 피드백을 통한 논리 형식의 반복 생성 접근 방식이 KBQA 일반에서 유망한 것으로 나타났습니다. 기존 few-shot 전이 데이터셋을 활용한 실험에서도 FUn-FuSIC는 답변 가능한 질문에 대해서도 뛰어난 성능을 보였습니다.



### Infusing clinical knowledge into tokenisers for language models (https://arxiv.org/abs/2406.14312)
Comments:
          18 pages, 6 figures

- **What's New**: 이번 연구에서는 임상 텍스트 처리(Clinical Text Processing)를 위한 새로운 지식 향상 토크나이저(K-Tokeniser)를 소개합니다. K-Tokeniser는 초기화 단계에서 도메인 개념(약물이나 질병 등)의 의미 유형을 기반으로 토큰의 전역 표현을 채웁니다. 그리고 학습 또는 추론 단계에서는 문장 수준의 로컬화된 컨텍스트를 활용하여 최적의 전역 토큰 표현을 선택합니다.

- **Technical Details**: K-Tokeniser는 도메인 온톨로지(Unified Medical Language System)나 작업 관련 코퍼스의 학습 데이터를 활용하여 토큰의 전역 표현을 초기화합니다. 새 토크나이저를 사전 학습(pretraining) 없이 활용하기 위해, 새로운 토큰에 대한 표현을 생성하는 임베딩 초기화 접근법을 제안합니다. 3개의 트랜스포머 기반 언어 모델과 4가지 실제 데이터셋을 사용하여 임상 텍스트 분석 작업을 평가했습니다.

- **Performance Highlights**: 임상 개념 및 관계 추출, 자동 임상 코딩, 임상 표현형 식별, 임상 연구 기사 분류 등의 임상 텍스트 분석 작업에서 K-Tokeniser는 일관된 성능 향상을 보였습니다. 특히 자동 임상 코딩 작업에서는 Micro F1 점수가 13% 증가했습니다. 또한 K-Tokeniser를 사용하면 언어 모델의 수렴 속도가 빨라졌으며, 학습 데이터의 50%만으로도 기존 토크나이저 전체 데이터를 사용한 것과 동일한 성능을 달성할 수 있었습니다. 모든 개선 사항이 사전 학습 과정 없이 이루어져 일반화가 용이합니다.



### VAIYAKARANA : A Benchmark for Automatic Grammar Correction in Bangla (https://arxiv.org/abs/2406.14284)
- **What's New**: 방글라(벵골어)는 전 세계에서 다섯 번째로 많이 사용되는 언어이지만, 방글라의 자동 문법 교정 문제는 아직 초기 단계에 있습니다. 이에 대해 본 연구에서는 방글라의 문법적으로 틀린 문장을 생성하는 실용적인 접근 방식을 제안합니다. 이를 통해 방글라의 문법 오류를 효과적으로 교정할 수 있는 데이터셋인 Vaiyakarana를 제공합니다.

- **Technical Details**: 본 연구에서는 방글라 문법 오류를 생성하는 시스템적 접근 방식을 사용했습니다. 방글라의 오류 유형을 5개의 큰 분류와 12개의 세부 분류로 나누고, 이를 바탕으로 원래 문장에서 문법적으로 틀린 문장을 체계적으로 생성했습니다. 이렇게 하면 신경망(Neural Networks) 모델 훈련에 필요한 대량의 오류 문장을 생성할 수 있습니다. Vaiyakarana 데이터셋은 92,830개의 문법적으로 틀린 문장과 18,426개의 올바른 문장으로 구성되어 있으며, 이는 방글라 원어민 작성의 에세이에서 수집된 619개의 문장을 포함합니다.

- **Performance Highlights**: 연구 결과, 방글라 원어민의 문법 오류 감지 정확도가 최첨단 모델보다 훨씬 높았습니다. 즉, 인간 평가자가 문장 오류를 감지하는 데 더 뛰어난 성능을 보여줬습니다. 우리는 우리의 오류 문장 생성 방법론이 다른 인도 언어에도 적용될 수 있다고 제안합니다.

- **Dataset Information**: Vaiyakarana 데이터셋은 92,830개의 문법적으로 잘못된 문장과 18,426개의 올바른 문장을 포함하고 있습니다. 이는 방글라 원어민이 작성한 619개의 실제 인간 생성 문장을 바탕으로 구축되었습니다.



### Learning to Plan for Retrieval-Augmented Large Language Models from Knowledge Graphs (https://arxiv.org/abs/2406.14282)
Comments:
          Work in progress

- **What's New**: 이 논문은 복잡한 질문-응답(Question-Answering, QA) 작업에서 대형 언어 모델(LLMs)의 성능을 향상시키기 위한 새로운 프레임워크를 소개합니다. 주로 지식 그래프(Knowledge Graph, KG)에서 파생된 플래닝 데이터를 활용하여 LLMs의 플래닝 능력을 강화하는 방법론을 제안합니다. 기존의 수작업 주석 및 교사 LLM에서의 지식 증류 방식보다 빠르고 정확한 플래닝 데이터 구축을 목표로 합니다.

- **Technical Details**: 제안된 프레임워크인 'Learning to Plan from Knowledge Graphs (LPKG)'는 먼저 KG에서 패턴을 추출하여 데이터를 구축합니다. 그런 다음, 추출된 인스턴스를 자연어로 변환하여 복잡한 질문 및 하위 질문을 생성합니다. 이 데이터를 통해 LLMs를 파인튜닝하여 복잡한 질문을 처리하는 능력을 향상시킵니다. 또한, 논문에서는 Wikidata의 일부를 사용하여 새롭게 CLQA-Wiki 벤치마크를 구축하여 평가를 수행합니다.

- **Performance Highlights**: 제안된 프레임워크는 여러 기존 QA 벤치마크에서 기존 방법들보다 우수한 성능을 보였습니다. 특히 CLQA-Wiki 벤치마크에서의 평가 결과, KG에서 파생된 플래닝 데이터가 LLMs의 QA 작업 성능을 현저히 향상시키는 것을 확인했습니다.



### Augmenting Query and Passage for Retrieval-Augmented Generation using LLMs for Open-Domain Question Answering (https://arxiv.org/abs/2406.14277)
- **What's New**: 이 논문은 Open-domain question-answering (ODQA) 작업을 위한 새로운 방법으로 제안된 'question and passage augmentation via LLMs'에 대해 다룹니다. 이 방법은 원래의 질문을 여러 단계의 하위 질문으로 분해하여 질문을 구체화하고, 검색 성능을 향상시킵니다. 또한 회수된 단편들이 포함된 정보가 산만하거나 의견이 분분할 때를 보완하기 위해 LLM이 자체 생성된 단편들을 통해 검색된 단편들로부터의 답변 추출을 유도합니다.

- **Technical Details**: 이 방법은 LLMs를 통해 질문과 단편을 증강하는 간단하지만 효율적인 방법입니다. 질문 증강은 복잡한 질문을 더 쉬운 하위 질문으로 분해하는 가설을 기반으로 합니다. 하위 질문은 외부 소스로부터 검색되어야 할 지식에 대한 세부 정보를 포함합니다. 원래의 질문에 하위 질문을 추가하여 검색 질문을 구성하며, 이는 검색 성능을 개선합니다. 단편 증강은 검색된 단편의 품질이 저하될 경우에 대비해 LLMs의 광범위한 지식을 활용해 자체 생성된 단편을 추가하는 것입니다.

- **Performance Highlights**: 실험 결과, 제안된 스킴은 기존 RAG 방법에 비해 성능이 크게 향상되었음을 보여줍니다. 이 방법은 LLMs 및 리트리버와 잘 통합되어 ODQA 벤치마크 데이터셋에서 뛰어난 성과를 냈습니다. 또한 다양한 어블레이션 연구에서 이 방법이 우수한 성능을 나타내었음을 확인했습니다.



### Step-Back Profiling: Distilling User History for Personalized Scientific Writing (https://arxiv.org/abs/2406.14275)
- **What's New**: 최근의 자연어 처리(NLP) 작업에서 대규모 언어 모델(LLMs)이 탁월한 성능을 보여주고 있지만, 사용자 맞춤형 콘텐츠 생성 특히, 과학적 글쓰기와 같은 실세계 시나리오에서는 여전히 어려움을 겪고 있습니다. 이를 해결하기 위해, 사용자 이력을 요약하여 개인화된 프로필을 생성하는 'Step-Back Profiling' 접근 방식을 소개합니다. 이 방법은 다중 사용자 개인화를 연구하기 위해 PSW(개인화된 과학적 글쓰기) 데이터셋을 구축하여 다양한 학문적 배경을 가진 저자 그룹이 과학 논문을 작성하도록 요구합니다.

- **Technical Details**: LLMs의 개인화를 위해 새로운 프로파일링 접근법인 'Step-Back Profiling'을 제안하였습니다. 이 방법은 사용자 이력을 압축하여 핵심적인 특성과 선호도를 포함한 프로필로 요약합니다. 해당 접근법은 고차원적 특성을 메모리에 효율적으로 관리하고 개인화된 콘텐츠 생성을 가능하게 합니다. PSW 데이터셋은 다중 사용자 시나리오에서의 개인화된 과학적 글쓰기를 연구하기 위해 사용되며, 각각의 저자의 배경 논문들을 기반으로 프로필이 만들어집니다.

- **Performance Highlights**: Step-Back Profiling 접근법은 LaMP(일반 개인화 벤치마크)에서 최대 3.6점까지 기존 방법을 능가하는 성능을 보여주었습니다. 또한, 7개의 개인화 LLM 과제에서 그 효과를 입증하였습니다. 다양한 구성 요소에 대한 광범위한 분석 연구는 각 요소의 기여도를 검증하고 과제 정의에 대한 통찰을 제공합니다.



### On the Evaluation Practices in Multilingual NLP: Can Machine Translation Offer an Alternative to Human Translations? (https://arxiv.org/abs/2406.14267)
- **What's New**: 이 연구는 다국어 언어 모델(MLM)의 평가 문제점을 지적하며, 198개의 언어로 확장된 테스트 데이터를 사용해 MLM을 대규모로 평가하였습니다. 특히, 현재 사용되는 평가 프레임워크가 몇몇 언어에만 집중되어 있는 문제를 해결하고자 기계 번역과 인간 번역 데이터를 비교 분석하였습니다.

- **Technical Details**: 최신 기계 번역 모델 'NLLB'를 사용하여 4가지 테스크를 198개 언어로 번역하고, 이를 통해 XLM-R, BLOOMz, AYA-101 등 3개의 MLM 모델을 평가했습니다. 주로 파인튜닝(fine-tuning)과 제로샷 프롬프트(zero-shot prompting) 두 가지 평가 방식을 사용하여 성능 차이를 분석하였습니다. 기존 다국어 평가 테스크들이 오래된 단언 태스크에서 질문-응답과 검색 과제로 변화하는 경향이 분석되었습니다.

- **Performance Highlights**: 분석 결과, 고자원 언어의 경우 인간 번역과 기계 번역 간 성능 차이가 미미했으나, 저자원 언어에 대한 성능은 과대평가된 경향이 있음을 확인했습니다. 또한, 간단한 기초 모델도 큰 데이터 학습 없이도 좋은 성능을 보일 수 있다는 점에서 저자원 언어에서 대규모 다국어 학습의 이점이 의문시됩니다. 최종적으로, 번역된 테스트 데이터셋을 공개하며, 평가의 신뢰성을 높이기 위한 권장사항을 제안하였습니다.



### Raising the Bar: Investigating the Values of Large Language Models via Generative Evolving Testing (https://arxiv.org/abs/2406.14230)
Comments:
          Work in progress

- **What's New**: 이 논문은 LLMs (Large Language Models)의 윤리적 추론을 동적으로 평가하기 위해 GETA라는 새로운 접근 방식을 제안합니다. 기존의 고정적 데이터셋들이 모델의 빠른 발전에 따라 평가의 공정성을 해치는 문제를 해결하고자, GETA는 항목 생성기(item generator)를 활용해 모델의 도덕적 경계를 테스트하고 어려운 항목을 동적으로 생성합니다. 이를 통해 LLMs의 실제 윤리적 정렬 정도를 보다 정확하게 평가할 수 있습니다.

- **Technical Details**: GETA는 CAT(Computerized Adaptive Testing)와 AIG(Automatic Item Generation)를 결합한 모델로, 항목 응답 이론(Item Response Theory)과 변이형 IRT 모델을 동시에 학습합니다. 이 과정에서 항목 생성기는 LLM의 도덕적 경계를 탐색하고 각 모델에 맞춘 어려운 항목을 생성하는 역할을 합니다. 이를 통해 데이터 누출을 방지하고 LLM의 응답과 함께 항목이 진화하여, 평가의 공정성을 유지합니다.

- **Performance Highlights**: 다양한 LLMs, 예를 들어 GPT, Gemini, LLaMA, Mistral 등을 평가한 결과, GETA는 기존 평가 패러다임보다 더 정확하게 LLMs의 윤리적 정렬 상태를 반영함을 확인했습니다. 이는 특히 이전 평가 방법론이 과소평가했던 unseen OOD나 i.i.d. 항목에서도 일관된 성능을 보였습니다.



### Complexity of Symbolic Representation in Working Memory of Transformer Correlates with the Complexity of a Task (https://arxiv.org/abs/2406.14213)
Comments:
          18 pages, 6 figures. Published in the journal Cognitive Systems Research 3 June 2022: this https URL

- **What's New**: 이번 연구에서는 자연어 처리(NLP) 분야에서 광범위하게 사용되는 Transformer 모델에 상징적 작업 메모리(symbolic working memory)를 추가하여 번역 작업의 예측 품질을 향상시키는 방법을 소개합니다. 메모리 시스템을 통해 번역된 텍스트의 키워드를 저장하고, 이를 바탕으로 올바른 번역을 도출할 수 있는 신경-상징적(neural-symbolic) 정보를 처리합니다.

- **Technical Details**: Transformer 모델의 디코더(decoder)에 상징적 작업 메모리를 추가하여 내부 작업 메모리를 생성하고 이를 관리하는 방식을 제안합니다. 이 메모리 시스템은 작업 메모리를 생성 및 저장하는 동시에, NLP 작업에서 추가적인 컨텍스트 지식을 제공하여 텍스트 처리를 개선합니다. 작업 메모리의 요소는 자연어 단어나 부분 단어를 나타내는 동일한 임베딩(embedding)을 사용하여 해석 가능성을 유지합니다.

- **Performance Highlights**: 추가된 작업 메모리를 통해 번역 품질이 향상되었으며, 메모리 콘텐츠가 번역된 텍스트의 키워드를 저장하여 모델 예측의 정당성을 높였습니다. 또한, 메모리에 저장된 토큰(token) 및 품사가 말뭉치(corpora)의 복잡성과 상관관계가 있는 것으로 나타났습니다.



### On the Representational Capacity of Neural Language Models with Chain-of-Thought Reasoning (https://arxiv.org/abs/2406.14197)
Comments:
          To be published at ACL 2024

- **What's New**: 최신 언어 모델(LM)의 성능은 연쇄 추론(chain-of-thought, CoT) 과정을 통해 크게 향상되었습니다. 이는 중간 결과를 생성하여 모델이 최종 답변을 찾는 과정을 안내하는 방식입니다. 이번 연구에서는 CoT 추론이 언어 모델의 계산 능력을 확장한다고 설명합니다. 이를 이해하기 위해, 연구진은 CoT 추론을 확률론적 환경에서 형식화하여, 순환 신경망(RNNs)과 트랜스포머 모델(transformers)이 확률적 튜링 머신과 유사한 문자열 분포를 대표할 수 있음을 보여줍니다.

- **Technical Details**: 연구는 CoT 추론(chain-of-thought reasoning)의 표현적 능력을 분석하는 여러 결과를 제시합니다. 이 연구는 확률적 튜링 머신(probabilistic Turing machines)과 유사한 문자열 분포를 대표할 수 있는 순환 신경망(RNNs)과 트랜스포머(transformer) 언어 모델의 역량을 입증합니다. 특히, RNNs와 트랜스포머가 추가적인 저장 공간을 통해 계산 능력을 확장할 수 있는 방식을 강조합니다.

- **Performance Highlights**: 연구진은 언어 모델이 확률적 튜링 머신의 계산 능력과 유사한 성능을 나타낼 수 있음을 보여줍니다. 이는 CoT 추론이 중간 결과를 생성하여 모델의 최종 결과를 이끌어내는 방식이 언어 모델의 성능 향상에 중요한 역할을 한다는 사실을 뒷받침합니다.



### Timo: Towards Better Temporal Reasoning for Language Models (https://arxiv.org/abs/2406.14192)
Comments:
          Under review

- **What's New**: 이 논문은 다양한 시간 추론 작업(temporal reasoning tasks)을 처리할 수 있는 보편적인 프레임워크를 제안합니다. 기존 연구는 대부분 특정 작업에 초점을 맞추었지만, 이 방법은 일반적으로 다양한 시간 추론 작업에 대해 일반화할 수 없습니다. 이를 해결하기 위해, 저자들은 수학과 직접 관련이 있는 19개의 작업과 순수 시간 추론 작업(pure temporal reasoning tasks)을 포함한 총 38개의 시간 추론 작업을 체계적으로 연구했습니다.

- **Technical Details**: 이 연구는 먼저 수학적 데이터셋을 활용하여 튼튼한 시간 추론 기반을 구축하였지만, 이는 순수 시간 추론 작업을 다루기에는 부족했습니다. 이를 보완하기 위해, 저자들은 모델의 시간 추론 능력을 강화하면서도 일반 작업 능력을 잃지 않도록 간단하면서도 효과적인 셀프 크리틱 최적화 기법(self-critic temporal optimization method)을 제안했습니다. 최종적으로 Timo라는 7B와 13B 규모의 모델을 개발했으며, 이 모델은 해당 크기의 다른 LLM보다 평균 정확도 점수가 각각 10.0점과 7.6점 상회하며, 새로운 SOTA 성능을 달성했습니다.

- **Performance Highlights**: Timo 모델은 다양한 시간 추론 작업에서 뛰어난 성능을 발휘했습니다. 특히, 저자들은 LLaMA2 모델을 7B와 13B 크기로 훈련하고, 그 결과 각각 10.0점과 7.6점 향상된 평균 정확도를 보였다고 보고했습니다. 이러한 성능 향상은 수학적 지식과 시간 정보를 성공적으로 통합한 결과입니다. 또한, 이 프레임워크는 일반 작업 성능을 유지하면서도 다양한 시나리오에서 일관된 성능을 보여주었습니다.



### Temporal Knowledge Graph Question Answering: A Survey (https://arxiv.org/abs/2406.14191)
Comments:
          8 pages, 3 figures

- **What's New**: 최근 지식 기반 질문 응답(KBQA) 분야의 진화에 따라 Temporal Knowledge Graph Question Answering (TKGQA)라는 새로운 과제가 주목받고 있습니다. 이 논문은 TKGQA에 대한 포괄적인 개요를 제공하며, 기존 연구에서의 시간적 질문의 분류와 TKGQA 기법의 체계적인 범주화를 수행합니다.

- **Technical Details**: 이 논문은 시간적 질문의 분류체계를 정립하고, TKGQA 기법을 두 가지로 나눠 검토합니다: 의미 구문 분석(semantic parsing) 기반 방법과 TKG 임베딩(TKG embedding) 기반 방법입니다. 또한, 각 기법이 해결하는 질문 유형을 표로 요약하여, 기존 방법들의 초점을 분석합니다. 이를 통해 TKGQA 연구의 향후 방향성을 제안합니다.

- **Performance Highlights**: 논문은 시간적 질문의 다양한 유형과 복잡성을 분석하고, 기존 방법들이 해결하는 질문 유형과 주목받지 못한 질문 유형을 구분합니다. 이를 토대로, TKGQA 연구의 새로운 방향과 기회를 제시하며, 시간적 지식 그래프를 활용한 질문 응답 시스템의 발전을 촉진하는 것을 목표로 합니다.



### In Tree Structure Should Sentence Be Generated (https://arxiv.org/abs/2406.14189)
- **What's New**: 이 논문에서는 자연어 생성 시 발생할 수 있는 문제점들, 예를 들어 논리적 반복(looping)이나 오류(hallucinations)를 개선하기 위해 새로운 방법을 제안합니다. 저자들은 자연어 생성을 이진 트리 순회(traversal)를 통해 수행하는 방식으로 접근하며, 이를 통해 기존의 트랜스포머(transformer) 모델의 한계를 극복하려고 합니다. 또한, SenTree라는 이름의 모듈이 소개되었으며, 이는 이진 트리 구조를 근사하여 생성하는 역할을 합니다. 이 모듈은 공개된 URL을 통해 접근 가능합니다.

- **Technical Details**: 기존의 순차적(auto-regressive) 생성 방법 대신, 저자들은 문장을 이진 트리(Binary Tree) 순회 순서에 따라 생성하는 방식을 제안합니다. 이 접근 방식은 단어의 중요도에 따라 먼저 중요한 단어를 생성하고, 이후 덜 중요한 단어들을 생성하도록 합니다. 이는 denoising diffusion probabilistic models (DDPM)와 비슷한 방식으로 작동하며, 기존의 그래픽 생성 모델에서 자연어 생성으로 일반화된 형태로 설명됩니다. 또한, 이 방법은 생성적 적대 신경망(GANs)과 유사한 공동 학습 프레임워크를 포함하고 있습니다. 하지만, GANs와 달리 이 프레임워크는 협력적(collaborative) 방식으로 작동합니다.

- **Performance Highlights**: 제안된 방법은 번역 작업에서 향상된 BLEU 점수를 기록하며, 이는 기존의 트랜스포머 모델보다 우수한 성능을 보입니다. 또한, 경량화 되어 있으며 Widely used transformer 기반의 기존 시스템에 쉽게 통합될 수 있습니다. 마지막으로, 생성된 트리 구조의 품질을 통해 성능 향상이 추가적으로 가능함을 제안합니다.



### SimulSeamless: FBK at IWSLT 2024 Simultaneous Speech Translation (https://arxiv.org/abs/2406.14177)
- **What's New**: FBK가 IWSLT 2024의 Simultaneous Translation Evaluation Campaign에 참여했습니다. Speech-to-text translation (ST) 서브 트랙에서는 SimulSeamless라는 시스템을 제안했으며, 이는 AlignAtt와 SeamlessM4T를 중간 구성에서 결합한 것입니다. SeamlessM4T 모델은 'off-the-shelf'로 사용되며, AlignAtt를 통해 동시 추론이 가능해져 모델 재훈련이나 적응 없이도 사용 가능합니다.

- **Technical Details**: SimulSeamless는 SeamlessM4T 모델과 AlignAtt를 결합하여 동시 번역을 수행합니다. SeamlessM4T는 다중 언어와 다중 모드를 지원하는 모델로, 143개 이상의 출발 언어와 200개 이상의 도착 언어를 포함합니다. AlignAtt는 cross-attention을 기반으로 실시간 번역에서 번역 단어를 발행할지 추가 정보를 기다릴지를 결정합니다. 이 방식은 기존의 EDAtt 정책의 추가 하이퍼 파라미터 의존성을 제거하면서 더 나은 결과를 도출합니다.

- **Performance Highlights**: SimulSeamless는 작년의 참가자들과 비교해도 수용 가능한 결과 또는 더 나은 결과를 달성했습니다. 이 모델은 동시 번역 작업이나 평가된 언어 쌍에 대해 재훈련 되지 않았음에도 불구하고 뛰어난 성능을 보였습니다. 또한, SeamlessM4T는 IWSLT가 요구하는 모든 언어를 이미 지원하므로, 이 모델은 전적으로 'off-the-shelf'로 사용되었습니다.



### Definition generation for lexical semantic change detection (https://arxiv.org/abs/2406.14167)
Comments:
          Findings of ACL 2024

- **What's New**: 본 논문에서는 대형 언어 모델(Large Language Models, LLMs)이 생성한 맥락화된 단어 정의를 의미 표현으로 사용하여 어휘 의미 변화 감지(Diachronic Lexical Semantic Change Detection, LSCD) 작업을 수행합니다. 간단히 말해, 생성된 정의가 '의미'로 사용되며, 대상 단어의 변화 점수는 두 시기 동안의 분포 비교를 통해 계산됩니다. 이 접근법은 기존 비지도 의미 기반 LSCD 방법과 비슷하거나 더 우수한 성능을 보여주며, 해석 가능성을 유지하여 특정 변화의 원인을 이산적 정의로 검토할 수 있도록 합니다.

- **Technical Details**: 기존에는 토큰 임베딩(token embeddings)을 사용한 LSCD 방법이 널리 사용되었으나, 해석 가능성이 부족하다는 한계가 있었습니다. 반면 이 논문에서는 대형 언어 모델로 생성된 사전 같은 맥락화된 정의를 사용하여 이 한계를 극복하려고 합니다. 정의 생성 모델은 mT0-xl을 기반으로 여러 데이터셋에서 미세 조정된 것입니다. 영어, 노르웨이어, 러시아어 벤치마크를 사용하여 기존 연구와 비교 실험을 수행하였습니다.

- **Performance Highlights**: 정의 임베딩을 사용한 기존 LSCD 방법에서는 정의 문자열을 텍스트로 사용하였지만, 이는 의미 변화의 본질을 파악할 수 있는 장점이 있습니다. 실험 결과, 정의 생성 모델은 다수의 데이터셋에서 강력한 성능을 보여주었으며, 이를 통해 감지된 의미 변화의 해석 가능성을 유지할 수 있었습니다.



### Aligning Large Language Models with Diverse Political Viewpoints (https://arxiv.org/abs/2406.14155)
- **What's New**: 대형 언어 모델(LLM)이 종종 정치적 편향을 보일 수 있다는 점을 지적하면서, 스위스의 국회의원 후보들이 작성한 100,000개의 댓글을 사용하여 다양한 정치적 관점을 반영하도록 LLM을 조정한 연구가 소개되었습니다. 이 모델들은 상업적 모델(예: ChatGPT)보다 스위스 정당의 정치적 관점을 더 정확하게 생성합니다. 또한, 다양한 관점을 균형 있게 제시하기 위한 절차도 제안되었습니다.

- **Technical Details**: 이 연구는 Smartvote 애플리케이션에서 수집한 데이터와 메타데이터를 이용하여 LLM을 조정했습니다. Smartvote는 사용자가 정치적 문제에 대한 후보자들의 입장을 이해할 수 있도록 돕는 애플리케이션으로, 약 100,000개의 댓글이 포함되어 있습니다. 모델 조정에는 조건부 생성(conditional generation)과 단일체 편향 최적화(monolithic preference optimization)를 사용했습니다. 특히 Llama 3 모델을 이용하여 다양한 정치적 관점을 반영하는 텍스트를 생성하는데 중점을 두었습니다.

- **Performance Highlights**: 네 가지 모델(ChatGPT 3.5 zero-shot, Llama-3-instruct zero-shot, Llama-3-instruct-finetuned-with-dSFT, Llama-3-instruct-aligned-with-ORPO)의 성능을 비교한 결과, ORPO-aligned 모델이 가장 다양한 정치적 관점을 생성했습니다. 또한, ChatGPT는 모든 정당에 대해 유사한 진보적 응답을 생성했지만, ORPO-aligned 모델은 평균 유사도가 0.24로, 첫 번째 실험의 절반 정도의 중복도를 보였습니다.



### Finding Safety Neurons in Large Language Models (https://arxiv.org/abs/2406.14144)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 안전성 정렬 메커니즘을 탐구하여, 안전 행동을 담당하는 뉴런인 '안전 뉴런(safety neurons)'을 식별하고 분석합니다.

- **Technical Details**: 이 연구는 기계 학습 해석 가능성(mechanistic interpretability)의 관점에서 안전 뉴런을 찾고, 이들의 인과 효과를 평가하는 방법을 제안합니다. 주로 '생성 시간 활성화 대비(generation-time activation contrasting)' 방법을 사용하여 뉴런의 중요성을 계산하고, '동적 활성화 패치(dynamıc activation patching)'를 통해 이들의 인과 효과를 평가합니다.

- **Performance Highlights**: 1) 안전 뉴런의 밀도는 낮고 효과적입니다. 전체 뉴런의 약 5%만 개입해도 90%의 안전 성능을 회복할 수 있습니다. 2) 안전 뉴런은 전이 가능한 메커니즘을 인코딩합니다. 여러 가지 데이터셋에서도 일관된 효과를 보입니다. 3) 모델의 무작위 시도에서도 동일한 그룹의 안전 뉴런이 지속적으로 식별됩니다.



### MACAROON: Training Vision-Language Models To Be Your Engaged Partners (https://arxiv.org/abs/2406.14137)
Comments:
          The code will be made public at this https URL

- **What's New**: 이번 연구에서는 시각-언어 모델(LVLM)이 단순한 정보 제공자가 아닌, 인간과의 상호작용 능력을 갖춘 능동적인 파트너로 전환하고자 합니다. LVLMs의 능동적 상호작용을 평가하기 위한 세 단계의 질문 계층 구조를 개발하고 이를 바탕으로 PIE (ProactIve Engagement Evaluation) 벤치마크를 소개합니다.

- **Technical Details**: 세 단계의 질문 계층 구조는 다음과 같이 구성됩니다: 
1) 무효 질문(Tier-I): 답할 수 없거나 잘못된 전제에 기반한 질문을 식별하고 거부할 수 있는 능력.
2) 모호한 질문(Tier-II): 모호성을 해결하기 위해 명확화를 요청하는 능력.
3) 개인화 가능한 질문(Tier-III): 인간의 선호도를 반영하여 응답을 맞춤화할 수 있는 능력. 
이를 바탕으로 PIE 데이터셋을 구축하였으며, 총 853개의 질문-이미지 쌍이 포함되어 있습니다.

- **Performance Highlights**: 실험 결과, 현재의 LVLM 모델들은 PIE 벤치마크에서 매우 낮은 성능을 보였습니다(Aggregate Align Rate, AAR: 0.28). 이를 개선하기 위해 MACAROON (self-iMaginAtion for ContrAstive pReference OptimizatiON) 방법을 도입하였으며, 이 방법을 통해 LVLMs의 능동적 상호작용 능력을 상당히 향상시킬 수 있었습니다(0.84 AAR). MACAROON은 LVLM이 주어진 과제 설명과 인간이 만든 기준에 따라 대비 반응 쌍을 자동으로 생성하여, 조건부 강화 학습을 통해 모델의 대응 능력을 높였습니다.



### Take the essence and discard the dross: A Rethinking on Data Selection for Fine-Tuning Large Language Models (https://arxiv.org/abs/2406.14115)
- **What's New**: LLM(대규모 언어 모델)들의 미세 조정을 위한 데이터 선택(data selection)은 기존 데이터셋에서 고품질의 하위 집합을 선택하여 모델 성능을 향상시키고 학습 속도를 가속화하는 데 있다. 이 논문에서는 데이터 선택 방법들에 대한 포괄적 비교가 부족한 문제를 해결하기 위해, 데이터 선택을 위한 3단계 스킴을 제안하고 이를 기반으로 기존 연구들을 종합적으로 리뷰한다.

- **Technical Details**: 이 논문은 데이터 전처리(data preprocessing), 데이터 선택자(data selector) 구성, 데이터 선택자 평가의 3단계 스킴을 제안한다. 전처리 단계에서는 원본 텍스트의 특성을 유지하거나 설명 가능한 피처(features)로 변환한다. 데이터 선택자 구성 단계에서는 내부 품질 라벨(예: IFD)과 외부 품질 라벨(예: LLM preference)을 사용하여 데이터 품질을 판단한다. 마지막으로, 데이터 선택자 평가 단계에서는 선택된 데이터 서브셋을 사용한 모델의 성능 향상을 평가하여 데이터 선택의 유효성을 검증한다.

- **Performance Highlights**: 기존 연구들에 대한 심층 분석을 통해, 데이터 선택 방법이 특정 데이터 및 모델에 맞춤화될수록 효율성은 높아지지만 복잡성 증가로 인해 실용성(feasibility)이 떨어진다는 점을 발견했다. 또한, 더 복잡한 알고리즘을 채택하여 선택자의 효율성을 높일 때 추가적인 노이즈 정보의 도입을 피해야 한다는 점을 강조한다.



### Let Guidelines Guide You: A Prescriptive Guideline-Centered Data Annotation Methodology (https://arxiv.org/abs/2406.14099)
- **What's New**: 새로운 데이터 주석(Annotation) 방법론인 Guideline-Centered Annotation Process를 소개합니다. 이 방법론은 각 데이터 샘플에 대한 주석 지침(annotation guidelines)을 보고하는 것에 중점을 둡니다. 기존의 처방적 주석 과정의 세 가지 주요 한계를 극복하며, 정보 손실을 줄이고 지침 준수를 보장합니다. 추가로, 단 한 번의 인간 주석 과정으로 여러 작업에 주석 데이터를 재사용할 수 있는 기능도 설명합니다.

- **Technical Details**: Guideline-Centered Annotation Process(GC)은 주석 과정에서 주석 지침을 투명하게 사용하는 새로운 방법론입니다. 이는 주석자가 지침을 명확히 사용하도록 하여 개인적인 믿음에 의한 주석 지름길(annotation shortcut)을 방지하고, 보다 객관적이고 표준화된 주석 과정을 설계하는 것을 목표로 합니다. 또한, GC 방법론은 주석 지침 정의를 주석 샘플에 맞추어 모델을 훈련할 수 있도록 하여, 주석 과정과 모델 훈련이 보다 일관된 방식으로 진행될 수 있도록 합니다.

- **Performance Highlights**: GC 방법론을 적용함으로써 주석 과정에서 발생하는 정보 손실을 최소화하고 주석 지침 준수를 보장합니다. 이로 인해 모델 훈련 및 평가에서 더 풍부한 지침 정보를 활용할 수 있으며, 주석 데이터의 재사용이 용이해져 여러 과업에 걸쳐 일관된 데이터 품질을 유지할 수 있습니다.



### Seamless Language Expansion: Enhancing Multilingual Mastery in Self-Supervised Models (https://arxiv.org/abs/2406.14092)
Comments:
          Accepted by Interspeech 2024

- **What's New**: 이번 연구에서는 Self-supervised Learning(SSL) 모델을 새로운 언어로 효율적으로 적응시키는 새로운 방법을 제안합니다. 특히, 기존 SSL 모델인 mHuBERT에 Low Rank Adaptation(LoRA)을 통합하여 새로운 언어(중국어 만다린)로 확장합니다. 또한, 데이터 결합 및 재클러스터링을 통해 기존 언어에 대한 성능을 유지하면서 새로운 언어에 적응할 수 있는 보존 전략도 개발했습니다.

- **Technical Details**: 제안된 방법은 기존 SSL 모델을 새 언어에 적응시키면서도 파라미터 효율적인 성능을 보입니다. LoRA를 통해 다중 헤드 자기 주의 모듈을 강화하였고, 모델 파라미터 업데이트 시 기존의 학습된 표현이 덮어쓰이지 않도록 조정했습니다. 실험은 음성 재합성(speech re-synthesis) 작업을 통해 수행되었으며, 이는 Automatic Speech Recognition(ASR)보다 더 복잡한 작업을 요구합니다. 또한, 재합성 작업 결과를 통해 모델의 표현력과 적응 효율성을 평가했습니다.

- **Performance Highlights**: 제안된 적응 방법을 통해 mHuBERT 모델의 Mean Opinion Score(MOS)가 약 2.26에서 3.80으로 증가했습니다. 또한, 새로운 언어에 대한 Word Error Rate(WER)는 최대 61.72%까지 감소되었습니다. 이 결과는 제안된 방법이 기존 SSL 모델을 새로운 언어로 효율적으로 적응시킬 수 있음을 입증하며, 동시에 기존 언어에 대한 성능 저하를 최소화함으로써 Catastrophic Forgetting을 효과적으로 완화할 수 있음을 보여줍니다.



### Protecting Privacy Through Approximating Optimal Parameters for Sequence Unlearning in Language Models (https://arxiv.org/abs/2406.14091)
Comments:
          Accepted to ACL2024 findings

- **What's New**: 본 논문은 개인정보 보호와 관련된 중요한 연구 분야인 기계 학습 소거(unlearning)에 관한 새로운 접근법, Privacy Protection via Optimal Parameters (POP)을 제안합니다. 이는 사전 학습된 언어 모델에서 목표 토큰 시퀀스를 효과적으로 잊게 하며, 기존의 방법들이 갖고 있던 성능 저하 문제를 해결하는 데 중점을 둡니다.

- **Technical Details**: POP는 최적의 그라디언트 업데이트를 통해 특정 토큰 시퀀스의 기억을 제거하는 방식입니다. 이는 전체 재훈련 과정에서 파생된 그라디언트를 근사화하여, 모델이 목표 시퀀스를 제거한 이후에도 나머지 훈련 데이터의 지식을 유지하도록 합니다. 또한, 본 연구에서는 토큰 확률(likelihood)을 기반으로 개인정보 위험을 정량화하는 'Remnant Memorization Accuracy (RMA)'라는 새로운 지표도 제안했습니다.

- **Performance Highlights**: 연구 결과, POP는 9개의 분류 벤치마크와 4개의 대화 벤치마크에서 기존의 최첨단 기술을 뛰어넘는 성능을 보였습니다. 또한, RMA는 개인정보 보호를 더 강력하게 보장할 수 있는 지표로써, 정성적 및 정량적 분석을 통해 그 효과가 입증되었습니다.



### EXCEEDS: Extracting Complex Events as Connecting the Dots to Graphs in Scientific Domain (https://arxiv.org/abs/2406.14075)
Comments:
          This paper is working in process

- **What's New**: 과학 도메인에서는 이벤트 추출(Event Extraction) 연구와 그에 맞는 포괄적인 데이터셋 및 방법론이 부족하다. 이 연구는 과학 논문 초록을 위한 대규모 멀티 이벤트 문서 수준 데이터셋인 SciEvents를 구축했다. SciEvents는 2,508개의 문서와 24,381개의 이벤트를 포함하고 있으며, 정교한 주석과 품질 관리가 특징이다. 이와 더불어, 새로운 과학 이벤트 추출 프레임워크인 EXCEEDS를 제안했다. EXCEEDS는 밀도 높은 너겟(nuggets)을 그리드 매트릭스에 보관하고 복잡한 이벤트 추출을 점 연결 작업으로 단순화하여 처리한다.

- **Technical Details**: EXCEEDS는 모든 토큰 쌍 사이의 관계를 모델링하여 단어-단어 이벤트 그리드에 밀도 높은 너겟과 이벤트를 저장한다. 이를 통해 모든 너겟과 이벤트를 동시에 인코딩하고 추론 시 한 번에 디코딩할 수 있다. 복잡한 이벤트 추출을 점(dot) 구성 및 연결 작업으로 단순화하여, 너겟을 점으로, 이벤트-논쟁(event-argument)과 이벤트-이벤트 링크를 엣지로 간주해 점과 엣지를 동시에 인코딩한다. 실험 결과, EXCEEDS는 주요 태스크에서 기존 방법들보다 우수한 성능을 보였다.

- **Performance Highlights**: EXCEEDS는 SciEvents 데이터셋에서 계층적 이벤트 추출에서 탁월한 성능을 보였으며, 기존의 이벤트 추출 방법들보다 뛰어난 성능을 입증했다. 특히, 복잡한 시나리오에서 밀도 높은 너겟 추출의 도전을 잘 극복하여 성과를 냈다.



### How Many Parameters Does it Take to Change a Light Bulb? Evaluating Performance in Self-Play of Conversational Games as a Function of Model Characteristics (https://arxiv.org/abs/2406.14051)
Comments:
          under review

- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs, Large Language Models)이 목표 지향적(agentive) 문맥에서 대화형 게임의 자가 플레이(self-play)를 통해 성능이 어떻게 발전하는지 분석합니다. 특히 모델의 파라미터 수와 훈련 방법과 같은 모델 특성들이 성능에 어떤 영향을 미치는지를 중점적으로 다룹니다.

- **Technical Details**: 모델 성능의 결정 요인을 분석하기 위해, 마코프 결정 프로세스(Markov Decision Process)를 사용하여 대화 게임의 상태(state), 행동 공간(action space), 전이 함수(transition function), 보상 함수(reward function) 등을 수학적으로 형식화합니다. 또한, 클레멘치(clembench)와 같은 세분화된 평가 지표 사용으로, 모델의 이해력 및 형식 지시사항 준수 능력을 평가합니다.

- **Performance Highlights**: 대형 언어 모델의 성능과 파라미터 수 사이에는 명확한 상관관계가 있지만, 같은 크기 범위 내에서도 성능 차이가 큰 것을 발견했습니다. 이는 미세 조정 데이터의 품질 및 방법과 같은 훈련 파라미터에 기인합니다. 또한, 추론 도중 중간 정도의 가중치 양자화(weight quantisation)에도 성능이 안정적으로 유지되는 긍정적인 면모를 발견했습니다.



### Prompt Injection Attacks in Defended Systems (https://arxiv.org/abs/2406.14048)
- **What's New**: 대규모 언어 모델(LLM)을 대상으로 하는 블랙박스 공격에 대한 방어 기법을 연구한 논문입니다. 이 논문은 LLM의 취약점을 분석하고, 이러한 공격을 탐지하고 방어하기 위한 알고리즘을 제안합니다. 특히 SaTML CTF 2024 대회를 통해 LLM의 보안 능력을 평가하기 위한 실험 환경을 소개합니다.

- **Technical Details**: 이 논문은 LLM이 트레이닝 데이터를 통해 비의도적으로 편향이나 특정 스타일을 모방할 수 있다는 점을 악용하는 블랙박스 공격 기법을 다룹니다. 방어 메커니즘은 세 가지로 구분됩니다. 첫째, 자가 처리 방어(Self-Processing Defenses)로, LLM 자체가 스스로 위협을 평가하고 대응합니다. 둘째, 보조 방어(Additional Helper Defenses)로, 주 모델 외에 추가적인 LLM이 함께 동작합니다. 마지막으로, 입력 변형 방어(Input Permutation Defenses)로, 입력 프롬프트를 변경하여 공격을 방어합니다.

- **Performance Highlights**: 연구는 저자들이 제안한 방어 메커니즘을 사용하여 SaTML 2024 CTF 대회에서 다양한 방어 및 공격 전략을 평가하는 방식으로 진행됩니다. 방어 프롬프트와 필터는 대화 초반에 적용되며 공격 단계에서 참가는 공격 전략을 수정하여 비밀 정보를 추출하려고 시도합니다. 대회는 실제 환경에서 LLM 보안의 한계를 평가하고 다양한 방어 전략의 효과를 입증하는 데 중점을 둡니다.



### Two Giraffes in a Dirt Field: Using Game Play to Investigate Situation Modelling in Large Multimodal Models (https://arxiv.org/abs/2406.14035)
Comments:
          under review

- **What's New**: 이 논문에서는 최근 텍스트 모델에서 도입된 평가 패러다임인 목표 지향 게임 (self) 플레이를 멀티모달 (텍스트와 이미지) 모델에 적용하는 방법을 소개합니다. 이 방법은 기존의 참고 기반(reference-based) 및 선호 기반(preference-based) 평가를 보완합니다. 논문은 시각 정보를 통해 상황을 표현하고 이러한 표현을 대화를 통해 일치시키는 능력을 도전하는 게임을 정의합니다.

- **Technical Details**: 논문은 멀티모달 모델의 평가를 위해 목표 지향 게임 플레이를 제안합니다. 이 게임에서는 모델이 시각 정보를 바탕으로 상황을 표현하고 대화(dialogue)를 통해 그 표현을 맞추는 과제가 주어집니다. 이러한 방식으로, 모델의 캡셔닝 능력(captioning capabilities)과 표현 일치 능력(alignment capability)을 평가합니다.

- **Performance Highlights**: 가장 큰 폐쇄형(closed) 모델은 이러한 게임에서 비교적 좋은 성능을 보였습니다. 반면, 최고의 오픈 웨이트 모델(open-weight models)도 어려움을 겪었습니다. 추가 분석에서는 큰 모델의 뛰어난 딥 캡셔닝 기능(deep captioning capabilities)이 일부 성능을 견인하는 것으로 나타났습니다. 평가 기준이 여전히 관련성이 높음을 보장하는 결과가 나왔습니다.



### The Reason behind Good or Bad: Towards a Better Mathematical Verifier with Natural Language Feedback (https://arxiv.org/abs/2406.14024)
Comments:
          9 pages

- **What's New**: 새로운 연구에서는 기존의 이진 분류 라벨로 훈련된 수학 검증기(verifier)의 한계를 극복하기 위해 단계별 자연어 피드백(rationale labels)을 도입했습니다. 이를 통해 수학적 해결책의 정확성을 평가하는 능력을 향상시키는 Math-Minos라는 모델을 제안했습니다.

- **Technical Details**: Math-Minos는 자동 생성된 훈련 데이터를 사용하여 두 단계의 훈련 절차를 갖추고 있습니다. 첫 번째 단계에서는 단계별 자연어 피드백 데이터를 생성하며, 두 번째 단계에서는 기존 이진 분류 훈련을 통해 모델의 평가 능력을 강화합니다. 특히, GPT-4의 평가 생성 능력을 높이기 위해 'Label-aware Natural Language Feedback Curation'을 도입했습니다.

- **Performance Highlights**: Math-Minos는 30k의 자연어 피드백 훈련 데이터만으로도 검증기의 성능을 크게 향상시킵니다. GSM8K 데이터셋에서 정확도를 1.6% (86.6% → 88.2%) 올렸으며, MATH 데이터셋에서는 0.8% (37.8% → 38.6%) 올렸습니다. 이는 기존의 Outcome Reward Model (ORM)과 Process Reward Model (PRM) 설정에서 모두 성능 향상을 보여줍니다.



### Evaluating Implicit Bias in Large Language Models by Attacking From a Psychometric Perspectiv (https://arxiv.org/abs/2406.14023)
Comments:
          Code and datasets are available at this https URL

- **What's New**: 최근 등장한 대형 언어 모델(LLMs)의 정보 제공 능력이 주목받고 있는 가운데, 해당 모델들이 생성하는 비윤리적 콘텐츠에 대한 우려가 커지고 있습니다. 본 연구는 LLMs의 특정 그룹에 대한 내재된 편향을 평가하기 위해 심리학적 원칙을 바탕으로 설계된 공격 방법론을 제안합니다. 이를 통해 LLMs가 주어진 지시에 따라 편향된 응답을 생성하는지 평가합니다.

- **Technical Details**: 본 논문에서는 Disguise, Deception, Teaching이라는 세 가지 공격 방식을 제안합니다. 각각의 방법은 심리학적 개념인 목표 전환(goal shifting), 인지 일치(cognition concordance), 모방 학습(imitation learning)에서 영감을 받았습니다. Disguise 공격은 대화의 맥락에 편향된 콘텐츠를 숨기고(Viewpoint Contextualization), Deception 공격은 LLMs가 특정 편향적 관점을 가지고 있다고 믿도록 만듭니다(Mental Deception) 또는 특수 API 호출로 생성된 편향된 내용을 메모리에 삽입합니다(Memory Falsification). Teaching 공격은 LLMs에게 편향된 예시를 모방하도록 요구합니다(Destructive Indoctrination). 이러한 공격 방법론을 기반으로 네 가지 일반적인 편향 유형(나이, 성별, 인종, 성적 지향)에 대한 평가 데이터셋을 구축하였습니다.

- **Performance Highlights**: 광범위한 평가 결과, 세 가지 공격 방식 모두 효과적으로 LLMs의 내재된 편향을 유도할 수 있었으며, 특히 Deception 공격이 가장 효과적이었습니다. GLM-3은 GPT-3.5와 GPT-4보다 공격 방어능력이 뛰어났습니다. 또한 Teaching 공격의 경우, 한 가지 유형의 편향 예시(예: 인종)에 대해 학습할 때 다른 유형의 편향(성별, 종교 등)도 유도될 수 있음을 관찰하였습니다. 이러한 방법론은 LLMs의 내재된 편향을 평가하고, LLMs의 잠재적 윤리적 위험을 평가하는 데 유용합니다.



### HIGHT: Hierarchical Graph Tokenization for Graph-Language Alignmen (https://arxiv.org/abs/2406.14021)
Comments:
          Preliminary version of an ongoing project: this https URL

- **What's New**: 최근에는 대규모 언어 모델(LLMs)을 그래프 모달리티, 예를 들어 소셜 네트워크 및 분자와 같은 분야로 확장하는 데 관심이 증가하고 있습니다. 기존 접근법들은 주로 그래프 신경망(GNN)을 사용해 그래프를 일련의 노드 토큰으로 표현한 후 이러한 토큰을 LLMs에 제공하여 그래프-언어 정렬(Alignment)을 구현하고 있습니다. 하지만 이러한 접근법은 그래프 데이터에 내재된 계층적 구조를 간과하고 있었습니다. 따라서, 이를 개선하기 위해 새로운 전략인 HIGHT(Hierarchical GrapH Tokenization)를 제안합니다. 이 방법은 계층적 그래프 토크나이저를 통해 노드, 모티프, 그래프 수준의 유익한 토큰을 추출하고 인코딩하여 LLMs의 그래프 인식을 개선합니다.

- **Technical Details**: HIGHT는 계층적 그래프 토크나이저와 계층적 분자 지시 조정 데이터셋을 사용하여 분자-언어 정렬을 향상시킵니다. 원래의 분자 그래프를 모티프와 그래프 노드를 추가하여 계층적 그래프로 변환한 후, VQVAE(Vector Quantized-Variational AutoEncoder)를 사용해 원자 수준, 모티프 수준, 그래프 수준의 토큰을 개별적으로 획득합니다. 또한, 원래의 구조 정보를 최대한 유지하기 위해 라플라시안 위치 인코딩을 토큰에 추가합니다. 이후, 원자 수준, 모티프 수준, 그래프 수준 토큰을 각각 처리하는 세 개의 어댑터로 구성된 다중 수준 어댑터를 통해 LLMs에 입력합니다.

- **Performance Highlights**: 7가지 분자 중심 벤치마크에서 광범위한 실험을 수행한 결과, HIGHT는 홉니화(hallucination)를 40% 감소시켰으며 다양한 분자-언어 다운스트림 작업에서 유의미한 개선을 보여줬습니다. 특히 MotifHallu라는 벤치마크에서 HIGHT는 공통적인 기능 그룹의 존재 여부에 관한 질문에 대해 일관되게 '예'라고 답하는 기존 LGLMs에 비해 월등한 성능을 보였습니다.



### Seeing Through AI's Lens: Enhancing Human Skepticism Towards LLM-Generated Fake News (https://arxiv.org/abs/2406.14012)
- **What's New**: 이 논문은 사람들이 인간이 작성한 기사와 LLM(대규모 언어 모델)이 생성한 기사를 구별하는 데 도움을 줄 수 있는 간단한 마커(Marker)를 밝히는 데 중점을 두고 있습니다. LLM이 생성한 허위 정보를 효과적으로 감별할 수 있도록 돕기 위해, 정보 이론(Information Theory)과 엔트로피(Entropy) 원칙에 기초한 Entropy-Shift Authorship Signature (ESAS) 메트릭을 도입했습니다. ESAS는 뉴스 기사 내에서 인칭이나 구문 등과 같은 용어를 저작권 구별의 중요도에 따라 순위를 매깁니다.

- **Technical Details**: 이 연구의 주요 기술적 부분은 39,000개의 뉴스 기사 데이터셋을 수집하여 인간이 작성한 기사와 4가지 다른 LLM이 여러 수준의 허위성을 갖고 생성한 기사를 포함시킨 것입니다. ESAS 메트릭은 TF-IDF와 로지스틱 회귀 분류기(Logistic Regression Classifier)를 결합하여 높은 정확도를 달성합니다. 특히, ESAS는 Part-of-Speech (POS) 태깅과 같은 용어를 기사의 저작권 식별의 중요도에 따라 순위를 매기며, 이 과정을 통해 인간이 작성한 기사와 LLM이 생성한 기사의 구별을 돕습니다.

- **Performance Highlights**: ESAS 메트릭은 높은 ESAS 점수를 가진 용어의 작은 집합으로도 TF-IDF와 로지스틱 회귀 분류기를 결합하여 높은 정확도를 달성했습니다. 이 방법은 현재 인간의 AI 생성 콘텐츠 감별 능력을 초과합니다. 이는 사람들이 LLM이 생성한 허위 뉴스에 대한 비판적 관점을 키우는 데 도움이 될 것입니다.



### Information Guided Regularization for Fine-tuning Language Models (https://arxiv.org/abs/2406.14005)
- **What's New**: 이번 논문에서는 사전 훈련(pretraining)과 미세 조정(fine-tuning)의 전략이 현대 언어 모델링에서 일반적인 전이 학습(transfer learning) 전략으로 사용됨에 따라, 이를 더 매끄럽게 하기 위한 정규화 접근법을 제안합니다. 이 방법은 특정 작업(task)에 민감한 매개변수들이 손실 지형(loss landscape)에 미치는 영향을 정보 이론적 관점에서 분석하고, 이를 활용해 새로운 드롭아웃(dropout) 접근법을 개발합니다.

- **Technical Details**: 제안된 방법론은 특정 작업과 모델 아키텍처에 독립적이며, 미세 조정 과정에서 추가적인 계산 부담을 주지 않습니다. 구체적으로, Fisher 정보(Fisher information)를 이용해 손실 지형을 분석하고, 이를 통해 L2 정규화를 유도하는 정보를 기반으로 한 드롭아웃 기법을 설계합니다. 학습 역학 및 손실 지형을 이해하기 위해 Fisher 정보가 Hessian의 근사치로 사용될 수 있음을 밝혔습니다.

- **Performance Highlights**: 실험을 통해 제안된 정규화 방법이 표준화된 기준선보다 일관되게 더 나은 성능을 보임을 확인하였습니다. 특히, 데이터 부족 상황에서도 유의미한 성능 개선을 보여주었습니다. 추가로, 작은 훈련 데이터 샘플만으로도 모델 정보의 신뢰할 수 있는 추정값을 얻을 수 있음을 증명하였습니다.



### "Global is Good, Local is Bad?": Understanding Brand Bias in LLMs (https://arxiv.org/abs/2406.13997)
- **What's New**: 최근 연구들은 대규모 언어 모델(LLMs)의 사회적 편향을 조사해왔지만, 브랜드 편향에 관한 연구는 거의 없었습니다. 본 연구는 제품 추천과 시장 분석 등의 사용 사례에서 브랜드 편향이 미치는 영향을 조사합니다. 특히, 기존의 글로벌 브랜드를 선호하면서 지역 브랜드를 소외시키는 경향을 발견했습니다.

- **Technical Details**: 네 가지의 브랜드 카테고리(신발, 의류, 음료, 전자 제품)를 포함한 데이터셋을 사용해 LLM의 편향을 분석했습니다. GPT-4o, Llama-3-8B, Gemma-7B, Mistral-7B 모델을 사용해 실험을 진행했으며, 기술적으로 Stimulus to Attribute Inference(SAI)와 Attribute to Stimulus Association(ASA) 두 가지 방향으로 프레임을 설정했습니다. 각 실험에서는 긍정적, 부정적, 중립적 속성을 선택하게 했습니다.

- **Performance Highlights**: 실험 결과, LLM들이 글로벌 브랜드를 긍정적으로, 지역 브랜드를 부정적으로 연관짓는 패턴을 확인했습니다. 고소득 국가에는 사치 브랜드를, 저소득 국가에는 비사치 브랜드를 추천하는 경향도 발견되었습니다. 또한, LLM들이 특정 상황에서 자국 브랜드에 대한 선호를 보이는 원산지 효과 역시 확인되었습니다.



### Exploring Changes in Nation Perception with Nationality-Assigned Personas in LLMs (https://arxiv.org/abs/2406.13993)
- **What's New**: 이번 연구에서는 대형 언어 모델(Large Language Model, LLM)이 특정 국가의 페르소나를 부여받았을 때 국가에 대한 인식이 어떻게 변하는지를 탐구하였습니다. 이를 위해 193개국의 국적 페르소나를 네 개의 LLM에 할당하고, 각 LLM의 국가 평가 변화를 조사했습니다.

- **Technical Details**: 연구에서는 GPT-4o, Llama-2-13B, Mistral-7B, 및 Gemma-7B를 사용하였으며, 193개국의 데모님(demonyms)을 기반으로 페르소나를 생성했습니다. 'Response Percentage (RP)'와 'Positively Mention Rate (PMR)'라는 두 가지 메트릭을 사용해 LLM의 국가 및 지역에 대한 행동을 측정했습니다.

- **Performance Highlights**: 모든 LLM은 페르소나 할당 여부와 관계없이 서유럽 국가들을 선호하는 경향을 보였으며, 특히 동유럽, 라틴 아메리카, 아프리카 국가들에 대해 부정적인 인식을 가지고 있었습니다. GPT-4o는 다른 LLM과 달리 긍정적인 평가 편향이 강하게 나타났으며, Llama-2-13B는 국적 페르소나에 가장 덜 민감한 모델로 확인되었습니다. 이러한 결과는 'AI 권리 장전'의 필요성을 강조하며, LLM의 공정성을 보장하기 위한 메커니즘 개발의 중요성을 시사합니다.



### Inference-Time Decontamination: Reusing Leaked Benchmarks for Large Language Model Evaluation (https://arxiv.org/abs/2406.13990)
- **What's New**: 이번 연구에서는 대형 언어 모델(Large Language Models, LLM)의 평가의 정확성을 저해하는 테스트 데이터 누설 문제를 해결하기 위해 새로운 방법을 제안합니다. 'Inference-Time Decontamination (ITD)'라는 기법을 통해 누출된 샘플을 감지하고 수정하여 평가 결과의 인플레이션을 완화하고자 합니다.

- **Technical Details**: ITD는 먼저 누출된 샘플을 감지한 후, 샘플의 난이도를 변경하지 않고 새로운 형태로 다시 작성합니다. 예를 들어, 지식과 관련된 벤치마크(MMLU)에서는 원래 질문의 지식 포인트를 유지하며 질문의 어구만 수정하고, 수학적 추론 능력과 관련된 벤치마크(GSM8K)에서는 원래 데이터의 숫자와 계산 내용을 유지하되, 질문의 배경을 다시 작성합니다.

- **Performance Highlights**: 실제로 ITD를 적용한 실험 결과, 벤치마크 GSM8K에서 22.9%, MMLU에서 19.0%의 정확도 감소를 확인했습니다. 또한, 인기 있는 LLM인 Phi-3와 Mistral에도 ITD를 적용한 결과, Phi-3는 GSM8K에서 5.3%, MMLU에서 6.7%의 감소를 보였으며, Mistral은 GSM8K에서 0.5%, MMLU에서 3.6%의 감소를 나타냈습니다. 이러한 결과는 ITD가 성능 인플레이션을 완화하는 데 효과적임을 나타냅니다.



### MR-BEN: A Comprehensive Meta-Reasoning Benchmark for Large Language Models (https://arxiv.org/abs/2406.13975)
- **What's New**: 대규모 언어 모델(LLMs)의 문제 해결 및 의사 결정 능력을 평가하기 위한 새로운 프로세스 기반 벤치마크 MR-BEN이 제안되었습니다. 이 벤치마크는 자동 생성된 추론 단계에서 잠재적 오류를 발견하고 분석하는 메타 추론(meta reasoning) 능력을 요구합니다.

- **Technical Details**: MR-BEN은 물리학, 화학, 논리학, 코딩 등 여러 과목에서 5,975개의 질문을 포함한 포괄적인 데이터셋입니다. 각 데이터 항목은 질문, 체인 오브 생각 답변(CoT Answer), 오류 분석으로 구성됩니다. 질문들은 다양한 추론 유형과 난이도를 다루며, CoT 답변은 GPT-3.5-Turbo-0125, Claude2, Mistral-Medium으로부터 수집되었습니다.

- **Performance Highlights**: MR-BEN 벤치마크를 통해 현재의 LLM 모델들이 결과 기반 벤치마크에서는 뛰어난 성능을 보이지만, 추론 과정에서의 오류를 발견하고 수정하는 능력에서는 한계가 있다는 점이 밝혀졌습니다. 이를 통해 개방형 모델과 폐쇄형 모델 간의 추론 능력 격차가 드러났습니다. 또한, 고품질 합성 데이터를 활용하는 방식이 추론 능력을 향상시키는 데 유용하다는 점도 제시되었습니다.



### Evolving to be Your Soulmate: Personalized Dialogue Agents with Dynamically Adapted Personas (https://arxiv.org/abs/2406.13960)
Comments:
          Work in progress

- **What's New**: 기존의 페르소나 기반 대화 에이전트 연구는 대화 에이전트의 페르소나를 사전에 설정한 후, 해당 페르소나가 지속적으로 고정된다는 한계를 가지고 있었다. 본 논문에서는 대화 중 에이전트가 자신의 페르소나를 동적으로 적응하여 사용자의 기대에 더 잘 부응하도록 하는 'Self-evolving Personalized Dialogue Agents (SPDA)'라는 새로운 패러다임을 제안한다. 이 패러다임을 통해 사용자 맞춤형 대화가 더욱 최적화될 수 있다.

- **Technical Details**: SPDA의 주요 기술적 문제는 페르소나의 정렬(alignment)과 단계적 적응 과정에서의 부드러운 전환을 어떻게 달성할 것인가에 있다. 이를 해결하기 위해 저자들은 계층적 수준에서 페르소나를 정제하여 점진적으로 사용자와의 정렬을 개선하는 새로운 프레임워크를 제안한다. 이 프레임워크는 현재 에이전트 페르소나의 일부가 여전히 적응 가능한지를 분석하고, 속성 수준(attribute-level)에서의 적응을 통해 빠르고 경량화된 조정을 가능하게 한다.

- **Performance Highlights**: 제안된 프레임워크를 통합한 실험 결과, 다양한 기본 시스템에서 페르소나를 적응시킴으로써 사용자 맞춤화와 전체 대화 성능이 지속적으로 향상됨을 보였다.



### AutoCAP: Towards Automatic Cross-lingual Alignment Planning for Zero-shot Chain-of-Though (https://arxiv.org/abs/2406.13940)
Comments:
          Accepted by ACL2024 Findings

- **What's New**: 이번 논문에서는 Cross-lingual Chain-of-Thought(CoT)의 문제를 해결하기 위해 새로운 Automatic Cross-lingual Alignment Planning(AutoCAP) 프레임워크를 제안했습니다. AutoCAP은 기존의 수작업 기반 언어 선택 방식의 한계를 극복하고, 다양한 언어 간의 최적의 정렬과 가중치를 자동으로 할당하도록 설계되었습니다.

- **Technical Details**: AutoCAP의 핵심은 두 가지 주요 컴포넌트로 구성됩니다. 첫째, Automatic Language Selection Prompting (ALSP)로, 이는 LLMs(Large Language Models)이 각 쿼리에 가장 적절한 언어를 자동으로 선택하도록 유도합니다. 둘째, Automatic Weight Allocation Prompting(AWAP)으로, 각 언어의 논리 경로에 맞는 가중치 점수를 자동으로 할당합니다. 이 과정은 전체적으로 1) 언어 선택과 2) 가중치 할당이라는 두 단계로 이루어집니다.

- **Performance Highlights**: 여러 벤치마크 실험 결과, AutoCAP은 이전의 수작업 기반 방법을 뛰어넘는 성능을 보여주었으며, state-of-the-art 성능을 달성했습니다. 이 프레임워크는 높은 일반화 능력을 입증하여 다양한 언어 간의 복잡한 논리적 추론 작업 수행에서 우수한 결과를 나타냈습니다.



### Reasoning Like a Doctor: Improving Medical Dialogue Systems via Diagnostic Reasoning Process Alignmen (https://arxiv.org/abs/2406.13934)
Comments:
          Accepted by ACL 2024 Findings

- **What's New**: 새로운 연구는 의사의 진단 추론 과정을 모방하여 진단 과정에서 의사의 선호도를 반영하는 의료 대화 시스템 'Emulation'을 제안합니다. 이 시스템은 유기증적(abductive) 및 연역적(deductive) 분석을 통해 진단 추론을 수행하며, 결과에 대한 명확한 설명을 제공하여 투명성을 높입니다.

- **Technical Details**: Emulation 프레임워크는 세 가지 주요 모듈로 구성되어 있습니다. 첫 번째는 유기증적 추론 모듈로 환자의 상태를 설명할 수 있는 잠재적 질병을 탐색합니다. 두 번째는 연역적 추론 모듈로 임상 소견과 잠재적 질병 간의 관계를 종합적으로 분석합니다. 마지막으로, Thought Alignment(사고 정렬) 모듈은 의사 선호에 맞게 응답을 생성합니다. 이를 위해 LLM(대규모 언어 모델)을 활용한 새로운 진단 사고 과정 데이터셋을 구성하였습니다.

- **Performance Highlights**: Emulation의 효과는 두 가지 데이터셋에서 실험을 통해 입증되었습니다. 이 프레임워크는 생성된 응답에 대한 명확한 설명을 제공함으로써 의료 상담에서의 투명성을 향상시킵니다. 또한, 진단 추론 과정에 대한 명료한 설명을 통해 시스템의 설명 가능성을 높였습니다.



### Large Language Models are Skeptics: False Negative Problem of Input-conflicting Hallucination (https://arxiv.org/abs/2406.13929)
Comments:
          12 pages, 9 figures

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)이 입력한 문맥의 내용과 일관되지 않은 응답을 생성하는 새로운 유형의 편향을 식별합니다. 이는 우리가 'false negative 문제'라 부르는 현상으로, LLMs가 문맥에 주어진 문장의 정당성을 평가할 때 부정적인 판단을 더 자주 내리는 경향을 나타냅니다.

- **Technical Details**: LLMs는 사실적 방향이 모순되는 동일한 정보를 포함하는 문장 쌍을 평가할 때 false negative 편향을 보입니다. 모든 대상 정답이 참(True)인 All-True 프롬프트와 모든 대상 정답이 거짓(False)인 All-False 프롬프트를 사용해 실험한 결과, LLMs는 거짓(False)에 대해 더 높은 과신(confidence)을 보여 false negative 문제를 입증했습니다.

- **Performance Highlights**: Mistral, ChatGPT, GPT-4 등을 포함한 다양한 LLMs에서 일관되게 false negative 문제가 나타났으며, 이는 특히 문맥 기반 사실 판단에 중요한 분야에서 치명적인 결과를 초래할 수 있습니다. 또한, 문맥과 쿼리 재작성(rewriting)이 false negative 문제를 효과적으로 해결하는 것을 발견했습니다. 하지만 GPT-4에서는 입력 재작성 시 false negative가 급격히 증가하는 현상도 관찰되었습니다.



### GenderAlign: An Alignment Dataset for Mitigating Gender Bias in Large Language Models (https://arxiv.org/abs/2406.13925)
- **What's New**: 새롭게 공개된 연구는 GenderAlign이라는 성 편견 완화를 위해 설계된 새로운 데이터셋을 소개합니다. 이 데이터셋은 Large Language Models (LLMs)에서 성 편견을 줄이기 위해 고안되었으며, 8,000개의 단일 회차 대화로 구성되어 있습니다. 각 대화는 '선택된' 응답과 '거부된' 응답으로 짝지어져 있습니다.

- **Technical Details**: GenderAlign 데이터셋은 기존의 HH-RLHF 데이터셋과 유사한 구성 방식을 따르지만, 성 편견을 줄이기 위한 특별한 목표를 가지고 있습니다. 이 데이터셋은 질문과 '선택된' 응답을 GPT-3.5 (Ouyang et al., 2024)를 사용하여 생성하고, '거부된' 응답은 편향되지 않은 LLM에 의해 생성됩니다. '거부된' 응답의 성 편견은 고정관념(stereotypes), 차별적 언어(discriminatory language), 직업 및 교육 기관에서의 성차별(sexism), 소수 성별에 대한 편견(bias against marginalized genders)의 네 가지 주요 범주로 분류됩니다.

- **Performance Highlights**: GenderAlign 데이터셋을 활용하여 다양한 LLMs를 평가한 결과, 이 데이터셋이 기존의 HH-RLHF 데이터셋보다 성 편견을 더욱 효과적으로 감소시키는 것으로 나타났습니다. BBQ와 WinoGender 같은 널리 사용되는 평가 데이터셋에서도 GenderAlign을 사용하는 모델이 더 우수한 성능을 보였습니다. 이러한 결과는 GenderAlign이 성 편견 완화에 있어 매우 유망한 도구임을 시사합니다.



### Persuasiveness of Generated Free-Text Rationales in Subjective Decisions: A Case Study on Pairwise Argument Ranking (https://arxiv.org/abs/2406.13905)
- **What's New**: 본 논문은 대규모 언어 모델(LLMs)이 생성하는 자유 텍스트 설명(rationales)이 주관적 과제에서 어떻게 설득력을 가지는지 분석한 연구를 다룹니다. 특히, LLM들이 논쟁 지원(debate assistance) 등 실제 응용에서의 잠재력을 탐구합니다.

- **Technical Details**: 본 연구에서는 GPT4 등 9개의 LLM을 대상으로 쌍대 비교 논쟁 순위 매기기(pairwise argument ranking) 과제를 수행하고, 각 모델이 선택한 답을 지지하는 설득력 있는 설명을 생성하는 능력을 평가했습니다. 또한, 모델의 설명 설득력을 개선하기 위한 모델 파라미터 조정 및 자가 개선(self-refinement) 방식도 실험했습니다.

- **Performance Highlights**: ['Open-source 모델 Llama2-70B-chat이 가장 설득력 있는 설명을 생성하여 GPT 모델을 능가했습니다.', 'GPT4는 인간 평가 기준과 유사한 수준의 설득력을 보였으나, 완벽히 일치하지는 않았습니다.', '대조적 설명(contrastive rationales)이 가장 설득력 있는 요소로 나타났습니다.', '설득력 요소를 모델에 명시적으로 제공하면 설명의 설득력이 향상되었습니다.']



### Generative AI for Enhancing Active Learning in Education: A Comparative Study of GPT-3.5 and GPT-4 in Crafting Customized Test Questions (https://arxiv.org/abs/2406.13903)
Comments:
          Publisher: Canadian Artificial Intelligence Association. URL: this https URL

- **What's New**: 이번 연구는 LLM(대형 언어 모델), 특히 GPT-3.5와 GPT-4가 적극적 학습(principles of active learning) 원칙에 맞춰 9학년 수학을 위한 맞춤형 질문을 개발할 수 있는 방법을 조사했습니다. 이 연구는 GPT-4를 '교사'로, GPT-3.5를 '학생'으로 설정하여 복잡한 질문을 생성하고 이에 대해 응답하도록 했습니다. 결과적으로 GPT-4가 정확하고 도전적인 질문을 생성하는 능력이 우수하며, GPT-3.5도 이러한 질문을 통해 복잡한 문제를 해결하는 능력이 향상됨이 증명되었습니다. 이러한 결과는 LLM이 개인화된 학습 시나리오를 모방하고 향상시킬 수 있는 가능성을 지니고 있음을 보여줍니다.

- **Technical Details**: 연구 방법론은 반복적 질문 생성 및 응답 프로세스를 포함합니다. 질문 생성 과정에서 두 가지 주요 파라미터를 설정했으며, 난이도와 내용을 기준으로 질문을 생성했습니다. 이 질문들은 '학생' 모델에게 제시되며, 학생의 응답을 바탕으로 다음 번 질문이 조정됩니다. 이 과정은 교사 모델인 GPT-4가 학생 모델인 GPT-3.5에게 점점 더 복잡한 질문을 던지도록 하여 학생의 학습 참여를 촉진하는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과, GPT-4는 복잡하고 도전적인 질문을 매우 정확하게 생성할 수 있었으며, GPT-3.5는 이 질문들에 대해 응답하면서 더 복잡한 문제를 처리하는 능력이 향상되었습니다. 이는 LLM을 이용한 맞춤형 학습 도구가 학생의 학습 성과 향상에 기여할 수 있음을 나타내며, 특히 능동적 학습 시나리오에서 높은 잠재력을 지니고 있음을 시사합니다.



### Open Generative Large Language Models for Galician (https://arxiv.org/abs/2406.13893)
Comments:
          12 pages, 1 figure

- **What's New**: 이 논문은 Galician 언어에 초점을 맞춘 최초의 두 개의 생성 LLMs (Large Language Models)을 발표합니다. 기존의 주로 영어 중심의 LLMs 훈련과 달리, Galician 언어를 포용하는 모델을 도입함으로써 언어적 다양성을 증진하고 소외된 언어 사용자를 지원하고자 합니다.

- **Technical Details**: 이 Galician LLMs는 기존의 큰 코퍼스에서 훈련된 LLMs를 Galician 언어로 계속 프리트레이닝(Continual Pretraining)을 통해 적응시키는 방법을 사용했습니다. 모델은 1.3B 매개변수와 2.1B 단어로 구성된 코퍼스를 기반으로 GPT 아키텍처를 사용하여 훈련되었습니다. 주요 목표는 기초적인 LLMs에서 시작하여 소스 및 타겟 어휘를 제공함으로써 데이터를 준수하면서도 Galician 언어에 맞게 지속적으로 적응시키는 것입니다.

- **Performance Highlights**: 이 모델들은 사람의 판단과 표준화된 벤치마크 데이터를 활용한 과제 기반 데이터셋을 통해 평가되었습니다. 평가 결과, Galician LLMs는 유망한 성능을 보여줬으며, 이는 생성 모델에서 언어적 다양성의 중요성을 강조합니다. 지속적인 프리트레이닝을 통해 Galician 언어에 대한 데이터 제약을 극복하면서도 좋은 성능을 달성했습니다.



### Adaptable Logical Control for Large Language Models (https://arxiv.org/abs/2406.13892)
- **What's New**: 대형 언어 모델(LLMs)의 생성 제어를 가능하게 하는 새로운 프레임워크 Ctrl-G를 소개합니다. 이 프레임워크는 LLM 출력이 논리적 제약을 따르도록 보장합니다. Ctrl-G는 LLM에 히든 마르코프 모델(HMM)을 결합해 논리적 제약을 충족시키는 데 유연성과 적응성을 제공합니다.

- **Technical Details**: Ctrl-G는 세 가지 주요 단계로 구성됩니다: (1) 사용하려는 LLM의 히든 마르코프 모델(HMM)로의 증류(distillation), (2) 논리적 제약을 표현하는 결정적 유한 오토마톤(DFA) 구성, (3) HMM을 DFA가 지정한 제약 조건에 조건화하여 LLM 생성을 안내. 이 프로세스를 통해 QTU2-7B 모델을 사용하여 GPT3.5와 GPT4보다 우수한 성능을 발휘합니다.

- **Performance Highlights**: Ctrl-G는 인간 평가에서 논리적 제약을 충족시키는 텍스트 편집 과제에서 GPT4보다 30% 이상 높은 만족도를 기록했습니다. 또한, GPT2와 같은 중간 크기 언어 모델을 사용할 때도 마찬가지로 탁월한 성능을 보입니다. 특히 광범위한 제약 조건에서도 일관된 고품질 텍스트를 생성하며, 이 접근법을 활용한 초기 연구로는 Grade School Math(GSM) 벤치마크에서 LLM의 추론 과정을 향상시키는 데 성공했습니다.



### ClinicalLab: Aligning Agents for Multi-Departmental Clinical Diagnostics in the Real World (https://arxiv.org/abs/2406.13890)
- **What's New**: 이번 논문에서는 LLMs (Large Language Models) 기반의 의료 에이전트 평가를 위한 새로운 평가 도구인 ClinicalLab을 소개합니다. ClinicalLab은 ClinicalBench라는 종합적인 다부문 의료 진단 평가 벤치마크와 네 가지 새로운 평가 지표(ClinicalMetrics)를 포함합니다. ClinicalBench는 24개 부서와 150개의 질병 사례를 포괄하며 실제 사례 기반 평가를 통해 데이터 누출 위험을 방지합니다.

- **Technical Details**: ClinicalBench는 다부문 및 실제 사례 기반의 엔드-투-엔드 평가 벤치마크로, 질병 진단에서 데이터를 누출하지 않도록 설계되었습니다. 이 벤치마크는 다차원적으로 LLMs를 평가하여, 각 모델의 수행 능력을 각각의 임상 진단 작업(task dimension)과 다양한 의료 전문 분야(department dimension)에서 측정합니다. 추가로, ClinicalMetrics를 통해 LLMs의 임상 진단 성능을 효과적으로 평가하며, 네 가지 주요 지표는 부서별 승률(Department Win Rate), 부서 설명 준수율(Department Instruction Following Rate), 종합 진단 정확도(Comprehensive Diagnostic Accuracy) 및 수용성(Acceptability)입니다.

- **Performance Highlights**: ClinicalBench를 통해 17개의 주요 LLMs를 평가한 결과, LLMs의 성능이 부서별로 크게 다름을 확인하였습니다. 특히, 특정 LLM이 모든 부서에서 우수한 성능을 보이지 않고, 각 부서별로 다른 LLM이 더 나은 성과를 나타냅니다. 이러한 결과는 현대 의학의 전문화 요구에 부합하는 것으로 나타났습니다. ClinicalAgent는 포괄적이고 실제적인 진단 기능을 갖춘 새로운 진단 에이전트로서, ClinicalBench 벤치마크에서 최고 성능을 보인 LLMs를 능가하는 성능을 나타냈습니다.



### Knowledge Tagging System on Math Questions via LLMs with Flexible Demonstration Retriever (https://arxiv.org/abs/2406.13885)
Comments:
          13 pages, 6 figures

- **What's New**: 최근의 발전된 텍스트 인코딩 알고리즘(pre-trained language models)을 활용하여, LLMs(Large Language Models)을 통한 자동 지식 태깅 시스템의 성능을 제시합니다. 이 시스템은 수학 문제에서 지식 태깅(task)을 수행하는 데 있어 뛰어난 제로샷(zero-shot) 및 퓨샷(few-shot) 결과를 보여줬으며, 강화학습 기반의 시연 검색기(reinforcement learning-based demonstration retriever)를 통해 성능을 더욱 향상시켰습니다.

- **Technical Details**: 이 논문에서 제안한 KnowTS 프레임워크는 세 가지 주요 구성 요소로 이루어져 있습니다. 이는 제로샷 추론 파이프라인(zero-shot inference pipeline), 퓨샷 추론 파이프라인(few-shot inference pipeline), 그리고 적응형 시연 검색기(adaptive demonstration retriever)입니다. 제로샷 추론 파이프라인은 주어진 주석 데이터가 없을 때 키워드만으로 직접 판단을 생성하며, 퓨샷 추론 파이프라인은 제한된 예제들을 통해 학습하는 것이 가능하도록 합니다. 마지막으로 시연 샘플이 필요할 때, 적응형 시연 검색기는 효과적인 시연을 선택하는 역할을 합니다.

- **Performance Highlights**: KnowTS는 대규모 교육 플랫폼에서 수집된 전문가 태깅 데이터를 사용한 실험에서, 최상의 인컨텍스트 학습 성능을 달성하면서도 적은 양의 시연을 활용해 높은 효율성을 보여주었습니다. 이는 LLMs의 강력한 제로샷 추론 능력과 다양한 및 방대한 데이터셋에 대한 광범위한 프리트레이닝 덕분에 가능합니다.



### Knowledge Graph-Enhanced Large Language Models via Path Selection (https://arxiv.org/abs/2406.13862)
- **What's New**: 대형 언어 모델(LLMs)이 다양한 실제 응용 분야에서 이전에 볼 수 없었던 성능을 보이지만, 사실과 다르게 정보를 생성하는 경향이 있다는 문제점이 지적되었습니다. 이에 따라 외부 지식 그래프(KGs)에서 지식을 추출하여 LLM에 통합하는 전략이 중요해지고 있습니다. 이번 논문에서는 KELP라는 세 가지 단계를 가진 새로운 프레임워크를 제안해 이러한 문제를 해결하려고 합니다. 특히 KELP는 입력 텍스트와 잠재 의미적 매칭을 통해 지식 경로에 대한 점수를 생성하여 세밀하게 지식을 추출할 수 있게 합니다.

- **Technical Details**: KELP는 크게 세 가지 단계로 구성됩니다: (i) 지식 경로 추출(Knowledge path extraction), (ii) 샘플 인코딩(Sample encoding), (iii) 세밀 경로 선택(Fine-grained path selection). 먼저, 입력 텍스트에서 식별된 엔티티를 바탕으로 지식 경로를 후보 지식으로 추출합니다. 이후, 선택된 경로들을 KG와 입력 텍스트 간의 간접적 의미적 관계를 학습하여 인코딩합니다. 마지막으로, 잠재적 의미 공간에서 계산된 유사성 점수를 바탕으로 두 가지 커버리지 규칙을 적용하여 고도로 유연한 경로 선택을 수행합니다.

- **Performance Highlights**: 실제 데이터셋을 사용한 실험에서 KELP는 사실 검증(Fact Verification) 및 질문 응답(Question Answering) 과제에서 LLM의 성능을 효과적으로 개선하는 것으로 확인되었습니다. 이는 KELP가 잠재적으로 영향력 있는 지식을 세밀한 단계로 추출하고 이를 LLM의 출력에 통합함으로써 달성된 결과입니다.



### Distributional reasoning in LLMs: Parallel reasoning processes in multi-hop reasoning (https://arxiv.org/abs/2406.13858)
- **What's New**: 이번 연구는 대형 언어 모델(LLM)의 내부 멀티홉(reasoning) 과정을 해석할 수 있는 새로운 분석 방법을 제안합니다. 특히, 구성적 추론(compositional reasoning) 질문의 예측 과정을 두 개의 의미 범주 공간 간의 단순 선형 변환으로 모델링할 수 있음을 보여줍니다.

- **Technical Details**: 이 연구는 LLM의 중간 층에서 다중 홉 질문의 가능성 있는 중간 답변 세트를 나타내는 고도로 해석 가능한 임베딩을 생성한다는 점을 보여줍니다. 통계 분석을 통해 모델 출력에서 해당 토큰 세트가 활성화되는 것을 확인했습니다. 이를 통해 병렬 추론 경로(parallel reasoning paths)가 존재함을 암시합니다.

- **Performance Highlights**: 모델의 중간 층의 임베딩은 첫 번째 속성 추출의 결과를 나타내고, 이 현상은 가능한 답변의 분포에 걸쳐 분포되어 있습니다. 중간 층 이후, 중간 답변의 포텐셜 활성화와 최종 답변의 강화라는 두 단계로 나누어지는 현상이 관찰되었습니다. 모델이 필요한 지식이 없을 때도 동일한 추론 과정을 사용하는 것을 확인하였습니다.

- **Implications**: 이 연구는 LLM이 연합적 활성화와 구조적 명제 추론을 활용함을 보여주며, 인간 인지 과정과 인공지능 추론 메커니즘 간의 이해를 도와줍니다. 이 결과는 인지 모델링(Cognitive Modeling) 연구에 귀중한 통찰을 제공합니다.



### Text Serialization and Their Relationship with the Conventional Paradigms of Tabular Machine Learning (https://arxiv.org/abs/2406.13846)
Comments:
          Accepted into the ICML AI4Science Workshop

- **What's New**: 최근 연구에서는 언어 모델(LM)을 특성 표현 및 예측을 위해 표형 머신러닝 과제에 적용하는 방법을 탐구하고 있습니다. 이는 텍스트 직렬화(text serialization)와 지도 학습 섬세 조정(Supervised Fine-Tuning, SFT) 기법을 사용하는 것을 포함합니다. 본 연구는 이러한 기술이 표형 머신러닝에서 전통적인 방법들과 비교되는지 평가하며, 데이터 표현 및 직렬화된 표형 데이터를 사용한 예측 성능에 미치는 영향을 탐구합니다.

- **Technical Details**: 자연어 처리(NLP) 분야에서는 트랜스포머 아키텍처(transformer architecture) 기반의 언어 모델 기술의 등장으로 패러다임 전환이 일어났습니다. 본 논문은 기존의 gradient boosting 방법이 심층 학습(deep learning) 전략보다 우수하다는 주장에 도전하며, 데이터 준비 과정의 여러 단계(예: 누락된 데이터 처리, 특징 스케일링 등)를 탐구합니다.

- **Performance Highlights**: 비교 연구를 통해 현재 사전 학습된 언어 모델(Pre-trained LM)이 전통적인 표형 데이터 해석 방법을 대체하지 못한다는 결론을 도출하였습니다. 표형 데이터에서 고차원성, 불균형, 분포 변동 등 다양한 상황에서의 성능을 평가한 결과, LM 기반 접근 방식이 우수하지 않음을 확인했습니다.



### Joint vs Sequential Speaker-Role Detection and Automatic Speech Recognition for Air-traffic Contro (https://arxiv.org/abs/2406.13842)
Comments:
          Accepted at Interspeech 2024

- **What's New**: 이 논문은 조종사와 관제사의 대화를 구분하여 텍스트로 변환하는 기존의 분리된 방법론 대신, Transformer 기반의 통합 ASR-SRD 시스템을 제안합니다. 이 시스템은 자동 음성 인식(ASR)과 스피커 역할 감지(SRD) 작업을 동시에 수행하여 기존 시스템보다 효율적입니다. 새로운 접근법의 성능을 여러 ATC 데이터셋에서 평가하고, 어느 경우에 더욱 효과적인지 분석합니다.

- **Technical Details**: 본 연구에서는 ATCO2, LiveATC, LDC-ATCC 데이터셋을 사용하여 실험을 진행했습니다. 데이터셋의 각 샘플은 2~19초 길이의 청크로 나누어져 있으며, 각 스피커에게 ATCO(관제사) 또는 PILOT(조종사) 라벨이 할당됩니다. SRD-ASR 모델에서는 Pyannote.audio 3.0을 사용하여 스피커 구분 및 클러스터링을 수행한 후, ASR 모델로 텍스트를 생성합니다. ASR-SRD 모델은 먼저 ASR로 텍스트를 생성한 후, 텍스트 기반 SRD 시스템으로 스피커 역할을 분류합니다. 제안된 Joint 모델은 Hugging Face의 wav2vec 2.0 및 xlsr 모델을 사용하여, 스피커 라벨이 포함된 텍스트로 직접 미세 조정됩니다.

- **Performance Highlights**: Joint 모델은 전통적인 SRD-ASR 및 ASR-SRD 접근법보다 여러 ATC 데이터셋에서 더 나은 성능을 보였습니다. 특히, 음향 및 언어적 차이에 의한 성능 저하를 효과적으로 극복할 수 있음을 확인하였습니다. 모든 실험은 NVIDIA V100 GPU에서 수행되었으며, 2000 스텝, 1000 워밍업 스텝, 학습률 4e-4, 배치 크기 4로 설정되었습니다.



### Neuro-symbolic Training for Reasoning over Spatial Languag (https://arxiv.org/abs/2406.13828)
- **What's New**: 최근 연구에 따르면 더 많은 데이터와 더 큰 모델이 자연어 문제 해결에 더 정확한 답을 제공할 수 있음에도 불구하고, 복잡한 입력 조합에서 일반화하지 못할 경우 실패할 수 있습니다. 이를 해결하기 위해, 우리는 신경-상징적(neuro-symbolic) 기법을 활용하여 논리적 추론 규칙을 모델에 종속시키고 추가적인 감독 자원을 제공하는 훈련 방법을 제안합니다. 특히, 텍스트를 통한 공간적 추론이라는 어려운 문제에 초점을 맞추었습니다.

- **Technical Details**: 우리의 접근 방식은 논리적 제약을 위반하지 않도록 모델을 훈련함으로써 일반화 가능성을 높이는 것입니다. 논리적 지식에서 얻은 감독을 통해 모델이 추상화를 향상시키고 다양한 도메인으로 지식을 이전할 수 있는 능력을 강화합니다. 우리는 SPARTQA-HUMAN, ResQ, STEPGAME의 세 가지 벤치마크를 선택하여 우리의 방법을 평가했습니다.

- **Performance Highlights**: 실험 결과, 우리의 신경-상징적 훈련 방법이 일반화 가능성과 도메인 이전 학습에서 매우 효과적이라는 가설을 확인할 수 있었습니다. 특히, 제안한 방법을 통해 작은 모델이 대규모 모델보다 복잡한 추론과 도메인 외 문제에서 더 잘 일반화할 수 있음을 보였습니다.



### Fine-Tuning BERTs for Definition Extraction from Mathematical Tex (https://arxiv.org/abs/2406.13827)
- **What's New**: 이번 논문에서는 수학 논문에서 정의를 추출하는 작업에 대해 세 가지 사전 학습된 BERT 모델을 미세 조정(fine-tuning) 하였습니다. 이 작업은 LaTeX로 작성된 수학 영어 문장에서 정의를 포함하는지 여부를 바이너리 분류 문제로 다룹니다. 'Chicago'와 'TAC' 두 개의 원본 데이터 세트를 사용하여 모델을 미세 조정 및 테스트했으며, 2021년에 Vanetik과 Litvak이 제안한 WFMALL 데이터 세트에서도 성능을 평가하였습니다.

- **Technical Details**: 이 논문에서는 수학 개념에 대한 자연어 정의를 LaTeX 코드에서 추출하기 위해 미리 학습된 BERT 모델을 사용한 후 미세 조정하였습니다. 우리가 정의하는 데이터 세트는 'Chicago'와 'TAC'로, LaTeX 명령어를 사용하여 방정식과 수학적 표기법을 표시합니다. 데이터를 정리한 후, spaCy를 사용하여 신뢰할 수 있는 문장 분할(sentencization)을 수행하고, 새로운 파이프라인 구성 요소 'detextor'를 개발하여 LaTeX 코드 내에서 일관된 토큰화를 보장했습니다.

- **Performance Highlights**: Sentence-BERT 트랜스포머 모델은 정확도, 재현율(recall), 정밀도(precision) 측면에서 가장 뛰어난 성능을 보였으며, 이전 모델과 비교해 적은 연산 노력으로도 동등한 성과를 달성했습니다. Vanetik과 Litvak의 WFMALL 데이터 세트에서 모델의 성능을 비교했을 때에도 높은 성능을 보여주었습니다.



### Framing Social Movements on Social Media: Unpacking Diagnostic, Prognostic, and Motivational Strategies (https://arxiv.org/abs/2406.13820)
Comments:
          Published in ICWSM Special Issue of the Journal of Quantitative Description: Digital Media

- **What's New**: 이번 연구는 2018-2019년 동안 트위터에서 총기, 이민, LGBTQ 권리 등 세 가지 사회 운동에 대한 메시지를 분석하여 진단적(diagnostic), 예측적(prognostic), 동기 부여적(motivational) 프레임 전략을 감지하는 코드를 작성하고 주석이 달린 데이터셋과 컴퓨터 모델을 개발했습니다. 이 연구는 각 프레임 전략에 대한 심층 비지도 언어 분석을 수행하고, 프레임과 언어적 특징 간의 연관성을 밝히는데 초점을 맞추었습니다.

- **Technical Details**: 연구진은 트위터 메시지를 통해 문제 식별 및 귀속(diagnostic), 제안된 해결책 및 전술(prognostic), 행동 촉구(motivational)와 같은 프레임 전략을 탐지하기 위한 코드를 개발했습니다. 또한, 비지도 언어 분석을 통해 프레임 전략과 대명사 및 의무적 법률 용어(deontic modal verbs)와 같은 언어적 특징 간의 연관성을 분석했습니다.

- **Performance Highlights**: 분석 결과, 진단적 프레임은 원본 게시물보다는 댓글에서 더 자주 나타나며, 사회 운동 조직은 일반 시민이나 기자보다 예측적 및 동기 부여적 프레임에 더 집중하는 것으로 나타났습니다.



### WikiContradict: A Benchmark for Evaluating LLMs on Real-World Knowledge Conflicts from Wikipedia (https://arxiv.org/abs/2406.13805)
- **What's New**: 새로운 연구는 대형 언어 모델(LLMs)의 한계를 보완하기 위해 Retrieval-augmented Generation(RAG) 기법을 사용하여 지식 충돌을 다루는 방식을 평가합니다. WikiContradict라는 고품질, 인간 주석 기반 벤치마크를 도입하여, 위키피디아에서 모순된 구절을 바탕으로 LLM의 성능을 평가합니다.

- **Technical Details**: 본 연구는 위키피디아에서 모순된 정보를 담고 있는 253개의 인간 주석 사례를 포함하는 WikiContradict 벤치마크를 소개합니다. 다양한 LLMs를 평가하기 위해, 단일 구절과 두 개의 모순되는 구절을 포함하는 QA 시나리오를 설정하고, 인간 평가를 통해 모델의 응답 정확성을 측정합니다. 또한, 비용이 많이 드는 인간 평가의 대안으로 F-score 0.8을 기록한 자동화 모델을 도입했습니다.

- **Performance Highlights**: 인간 평가 결과, 두 개의 모순된 구절이 제공되었을 때 모든 모델은 특히 암시적 모순을 제대로 반영하지 못하는 경향이 드러났습니다. Llama-3-70b-instruct 모델이 10.4%에서 43.8%로 성능이 크게 향상된 경우도 있었습니다. 자동화 평가 메트릭스를 사용하여 총 1,500개 이상의 응답을 평가한 결과, 모델의 모순 처리 성능을 향상시킬 가능성을 확인했습니다.



### Semantic Structure-Mapping in LLM and Human Analogical Reasoning (https://arxiv.org/abs/2406.13803)
- **What's New**: 이 논문에서는 인간과 대형 언어 모델(LLM)이 의미론적 구조 매핑(semantic structure-mapping) 기법을 통해 유사 추론(analogical reasoning) 능력을 비교하였습니다. 이전 연구는 주로 추상적인 기호를 사용한 유사 추론에 초점을 맞췄으나, 본 연구는 언어와 비언어적 영역 간의 유사 추론을 다루어 보다 의미론적인 접근을 시도했습니다.

- **Technical Details**: 본 연구는 인간과 LLM이 의미론적 구조와 내용을 서로 다른 도메인에서 전이할 수 있는 능력을 실험하였습니다. 참가자는 출발 도메인(source domain)과 도착 도메인(target domain)에서 주어진 단어 세트를 보고 마지막 단어를 완성하는 형태로 퀴즈를 풀게 됩니다. 실험에는 GPT-3, GPT-4, Pythia-12B, Claude 2, Claude 3(Opus), Falcon-40B와 같은 여러 LLM이 포함되었습니다.

- **Performance Highlights**: 가장 고도화된 LLM들은 인간의 성과를 많은 조건에서 일치시켰으며, 인간과 유사한 오류 패턴을 만들어냈습니다. 그러나 정보 제시 순서나 무관한 의미 정보를 무시하는 능력에서 인간과 LLM 간의 차이가 나타났습니다. 이러한 결과는 LLM의 유사 추론 능력이 향상되고 있지만, 여전히 인간과는 다른 류의 정보 처리 방식을 갖고 있음을 시사합니다.



### FoRAG: Factuality-optimized Retrieval Augmented Generation for Web-enhanced Long-form Question Answering (https://arxiv.org/abs/2406.13779)
- **What's New**: 이번 논문에서는 검색 엔진을 활용해 장문의 질문에 대한 답변을 향상시키는 Retrieval Augmented Generation (RAG)에 대한 연구를 다룹니다. 특히, 웹 기반 장문 질문 답변(LFQA)에서 사실성과 명확한 논리의 부족 문제를 해결하기 위해 새로운 아웃라인 강화 생성기(outline-enhanced generator)와 사실성 최적화 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 다음과 같습니다: 1) 다면적인 답변 생성을 위해 아웃라인 강화 생성기 개발, 2) 세밀하게 설계된 Reinforcement Learning from Human Feedback (RLHF) 프레임워크를 기반으로 하는 사실성 최적화. 이 프레임워크는 자동 평가와 보상 모델링을 다양한 수준의 세부 단계에서 수행합니다.

- **Performance Highlights**: 새로운 방법론을 적용한 결과, Llama2-7B-chat 모델을 기반으로 한 FoRAG-L-7B 모델이 WebGPT-175B를 일관성, 유용성, 사실성 측면에서 능가하는 성능을 보였습니다. 특히, FoRAG-L-7B는 파라미터 수가 WebGPT-175B의 1/24에 불과함에도 뛰어난 성능을 보여줍니다. 제안된 데이터셋과 모델은 재현성을 위해 공개될 예정입니다.

- **Datasets and Models Availability**: 생성된 데이터셋과 모델은 공개되어 연구자들이 자유롭게 사용할 수 있습니다.



### Can LLMs Reason in the Wild with Programs? (https://arxiv.org/abs/2406.13764)
- **What's New**: 대형 언어 모델(LLMs)의 추론 능력을 현실적인 시나리오에서 평가하기 위해 'reasoning in the wild'이라는 과제를 도입했습니다. 이는 LLMS이 알려지지 않은 유형의 문제를 해결하기 위해 하위 문제를 식별하고 각 하위 문제를 해결하기 위한 프로그램을 작성하는 과정으로 구성됩니다. 이러한 새로운 벤치마크로 다양한 이유의 문제를 다룹니다.

- **Technical Details**: 새로운 과제는 추론 문제를 해결하기 위해 프로그램을 작성하고 전술(tactic)의 지침을 따르는 것을 포함합니다. 이를 위해 ReWild라는 큰 전술 기반 경로 데이터셋을 만들었으며, 각 문제 해결 경로에는 Thought, Action, Observation의 체인 형태로 문제가 풀리는 과정을 기록합니다. 이 데이터셋은 6.7K 경로와 총 21.7M 토큰으로 구성됩니다.

- **Performance Highlights**: 기존의 LLMs는 명확하지 않고 혼합된 문제의 범위에서 중요한 성능 저하를 보였습니다. 특히 GSM8K에서 정확도가 적어도 50% 떨어졌으며, 혼합 문제가 포함된 경우 성능이 더욱 악화되었습니다. 이러한 한계를 완화하기 위해 ReWild에서 세부 튜닝을 수행한 LLaMA3-8B 모델은 GPT-4 수준의 성능을 달성했습니다.



### Every Language Counts: Learn and Unlearn in Multilingual LLMs (https://arxiv.org/abs/2406.13748)
- **What's New**: 이 연구는 다국어 대규모 언어 모델(LLMs)에서 유해한 정보의 전파를 조사하고 다양한 'unlearning' 기법의 효과를 평가했습니다. 논문은 가짜 정보가 어떤 언어로 도입되든지 훈련 데이터를 통해 모델에 주입되면 각기 다른 언어들 사이에서 확산되어 생성된 콘텐츠의 무결성과 신뢰성을 훼손할 수 있음을 보여줍니다. 또한, 표준적인 'unlearning' 기법이 비영어권 데이터에서는 비효율적이며 오히려 유해한 콘텐츠를 강화할 수 있음을 밝혔습니다.

- **Technical Details**: 이 연구에서는 사전 훈련된 LlaMa3-8B 모델을 다양한 언어로 된 가짜 정보가 포함된 데이터셋으로 미세 조정(fine-tune)하고, 해당 모델들이 어떻게 가짜 정보를 다국어로 생성하게 되는지 분석했습니다. 가짜 정보는 고자원(high-resource) 언어(독일어, 프랑스어, 간체 중국어, 러시아어)와 저자원(low-resource) 언어(자바어, 우르두어, 하우사어, 아르메니아어)를 포함한 8개 언어로 번역된 후 모델에 주입되었습니다. 모델의 평가에는 실제 정보 품질(Real Information Quality)과 가짜 정보 발생 빈도(Fake Information Occurrence Count)라는 두 가지 메트릭이 사용되었습니다.

- **Performance Highlights**: 모델 평가 결과, 영어를 포함한 고자원 언어에서 모델의 성능은 높게 나타났으나, 저자원 언어에서는 성능이 다소 낮았습니다. 또한, 어떤 언어에서 가짜 정보가 주입되더라도 해당 정보가 다른 언어로 전혀 확산되는 것이 확인되었습니다. 특히, 영어로 주입된 가짜 정보는 그 전파가 더 두드러졌으며, 영어로 질의할 때 가장 잘 전파되었습니다. 다국어 환경에서 유해한 콘텐츠를 효과적으로 제거하려면 모든 언어에서 유해한 반응을 근절해야하는 필요성을 강조했습니다.



### On the Utility of Domain-Adjacent Fine-Tuned Model Ensembles for Few-shot Problems (https://arxiv.org/abs/2406.13720)
Comments:
          Main paper is 8 pages, followed by limitations, references and appendix

- **What's New**: 본 논문에서는 다양한 다운스트림 작업에서 도메인 인접(Domain-Adjacent) 모델을 활용하여 제로샷(Zero-shot) 및 퓨샷(Few-shot) 문제를 해결하는 DAFT-E 프레임워크를 제안합니다. 이는 Fine-Tuned Foundation Models(FTFM)을 앙상블(Ensemble)하는 기법을 사용하여, 충분한 도메인 데이터를 확보하지 못한 경우에도 높은 성능을 달성할 수 있습니다.

- **Technical Details**: DAFT-E는 이미 훈련된 다양한 도메인 인접 모델(Domain-Adjacent Fine-Tuned Models, DAFT)을 사용하여 추가 훈련 없이도 제로샷 인퍼런스(Inference)가 가능합니다. 제로샷과 퓨샷 문제에서 각각 단일 모델과 비교하여 적은 데이터로 더 나은 성능을 발휘합니다. 또한 다수의 DAFT 모델을 앙상블하여 몇 개의 예제만으로도 높은 성능을 달성하도록 설계되어 있습니다.

- **Performance Highlights**: 제로샷 문제에서 DAFT-E는 단일 최고 모델과 거의 근접한 정확도를 보였고, 퓨샷 문제에서는 성능이 더욱 개선되며, 이는 단일 도메인 인접 모델보다 뛰어난 성능을 보여줍니다. 이를 통해 기존 모델 대비 훨씬 적은 데이터로 뛰어난 성능을 낼 수 있는 점이 확인되었습니다.



### Evaluating Large Language Models along Dimensions of Language Variation: A Systematik Invesdigatiom uv Cross-lingual Generalization (https://arxiv.org/abs/2406.13718)
Comments:
          9 pages

- **What's New**: 이번 연구에서는 대형 언어 모델(Large Language Models, LLMs)이 유사한 언어에서 성능 저하(Performance Degradation, PD)를 겪는 원인을 분석하고자, 언어 간의 음운적, 형태론적, 어휘적 거리를 베이지안 잡음 프로세스로 모델링했습니다. 이 프로세스를 통해 인위적인 언어를 생성하고, 이를 기반으로 LLM의 견고성을 평가하며, 실제 언어 쌍 데이터를 바탕으로 인공지능 모델의 성능 저하 패턴을 파악했습니다.

- **Technical Details**: 이번 연구에서는 음운적(phonological), 형태론적(morphological), 어휘적(lexical) 변이를 매개변수화된 확률적 '잡음(noise)'으로 모델링했습니다. 잡음 프로세스는 소스 언어에 적용되며, 생성된 인위적 언어들을 통해 LLM의 제로샷(cross-lingual zero-shot) 일반화를 분석했습니다. 주요 매개변수 후부(posteriors)는 이중언어 사전(bilingual lexicons)에서 계산되며, 이를 통해 실제 언어를 HRL 방언 공간에 배치하고 모델 성능 저하를 예측했습니다.

- **Performance Highlights**: 연구 결과, 인위적 언어에 대한 LLM 성능 저하는 실제 언어 쌍 데이터의 패턴과 일치함을 확인했습니다. 이는 언어 간의 거리를 잘 포착하는 잡음 프로세스가 유용한 정보를 제공함을 의미합니다. 또한, 이 프레임워크는 낮은 자원 언어(low-resource languages)의 성능 저하를 완화하는 방법을 제시할 수 있는 가능성을 열어줍니다.



### Benchmarking Open-Source Language Models for Efficient Question Answering in Industrial Applications (https://arxiv.org/abs/2406.13713)
Comments:
          Preprint submitted to Engineering Applications of Artificial Intelligence

- **What's New**: 이번 연구에서는 대규모 언어 모델(Large Language Models, LLMs)의 질문 응답(Question Answering, QA) 작업에서 오픈소스 및 비오픈소스 모델 간의 성능을 비교하는 포괄적인 벤치마킹 연구를 수행했습니다. 본 연구의 목표는 비슷한 성능을 제공하면서도 자원 효율적이고 CPU 기반 추론에 적합한 오픈소스 대안을 찾아내는 것입니다. 이를 통해 NLP 솔루션이 산업 환경에서 접근성과 효율성을 높이는 데 도움이 될 수 있습니다.

- **Technical Details**: 대규모 언어 모델(LLMs)은 최근 몇 년 동안 자연어 처리(NLP) 분야에서 현저한 발전을 이루었습니다. 주된 발전은 Transformer 아키텍처 도입으로 촉발되었으며, OpenAI의 GPT 시리즈를 비롯해 다양한 멀티모달 모델(DALL·E, CLIP 등)과 오픈소스 프로젝트(Hugging Face’s Transformers, EleutherAI의 GPT-Neo 및 GPT-J 등)가 포함됩니다. 이러한 모델들은 초기의 규칙 기반 시스템에서 출발하여 RNNs와 LSTM 네트워크를 거쳐 왔으며, 현재는 주로 Transformer 모델이 주류를 이루고 있습니다. 특히 이번 연구에서는 언어 능력 평가를 모방한 머신 리딩 컴프리헨션(MRC)에 초점을 맞추어 다양한 QA 작업에서 모델의 효과성을 평가했습니다. 모델 평가에는 정확도, 추론 속도, 자원 소비 등의 다양한 지표를 사용했습니다.

- **Performance Highlights**: 연구 결과, 일부 오픈소스 LLMs가 비슷한 성능과 효율성을 제공하면서 산업 적용에 적합한 것으로 나타났습니다. 이들 모델은 정확도, 추론 속도, 자원 소비 면에서 경쟁력을 가지며, 특히 소규모 및 중소기업이나 연구 기관에서 경제적이고 자원 효율적인 NLP 솔루션을 도입할 수 있는 가능성을 제시합니다. 이는 CPU 기반 환경에서도 실현 가능성이 높은 모델을 제시함으로써 실제 응용 현장에서 접근성을 크게 향상시킬 수 있습니다.



### Breaking News: Case Studies of Generative AI's Use in Journalism (https://arxiv.org/abs/2406.13706)
- **What's New**: 이 논문은 대형 언어 모델(LLM)과 기자들 간의 상호 작용을 분석한 연구입니다. 두 개의 뉴스 기관을 대상으로, 기자들이 LLM을 활용해 기사를 작성하는 방식을 조사했습니다. 연구 결과, 기자들이 민감한 자료를 LLM에게 제공하고 거의 개입하지 않은 채 기계가 생성한 기사를 게재하는 사례가 다수 발견되었습니다. 이에 따라, 저자들은 AI의 책임 있는 사용에 대한 연구와 명확한 가이드라인의 필요성을 강조합니다.

- **Technical Details**: LLM 사용 패턴을 분석하기 위해 공개된 'WildChat' 데이터셋을 활용하였습니다. 특정 기자들의 대화를 식별한 후 이를 온라인에 게재된 기사와 일치시켜 검증했습니다. LLM이 생성한 초안과 실제 게재된 기사 간의 겹침 정도를 ROUGE-L 점수로 측정했으며, 기사가 생성되고 출판되는 시간 간격을 분석했습니다. 또한 GPTZero를 사용하여 더 많은 기사를 검출했습니다.

- **Performance Highlights**: 연구 결과, LLM에 의해 생성된 기사와 게시된 기사 간 ROUGE-L 점수가 중간값 0.62를 기록하였고, 생성에서 게시까지의 시간이 하루에 불과했습니다. 입력 자극의 18%가 다른 기관의 기사였고, 9%는 개인적인 대화 내용이 포함된 민감한 정보로 밝혀졌습니다. 이는 사생활 침해의 위험성을 나타내며, AI 사용의 지침과 교육의 필요성을 시사합니다.



### MMTE: Corpus and Metrics for Evaluating Machine Translation Quality of Metaphorical Languag (https://arxiv.org/abs/2406.13698)
- **What's New**: 이번 논문에서는 기계 번역(MT)의 비유적 언어(figurative language) 번역 품질을 평가하는 새로운 평가 기준을 제안했습니다. 기존 평가 방법이 주로 유창성과 사실적 신뢰성에 초점을 맞추는 반면, 비유적 품질 측면에는 부족한 부분이 있음을 지적하고 있습니다. 이를 보완하기 위해 다국어 비유적 언어 병렬 코퍼스(parallel corpus)를 제공하고, 메타포 번역 품질 평가를 위한 인간 평가 프레임워크를 구축했습니다.

- **Technical Details**: 논문에서는 Metaphorical Equivalence, Emotion, Authenticity, 그리고 Quality의 네 가지 측면에서 비유적 언어 번역을 평가하는 프로토콜을 제안합니다. 또한 영어, 중국어 및 이탈리아어 간의 비유적 번역 평가를 위한 처음으로 수작업으로 주석된 다국어 코퍼스를 제공합니다. 평가 프레임워크는 비유적 번역의 품질을 평가하기 위한 수사적 등가(rhetorical equivalence)를 소개하며, 다언어 및 다각도 접근 방식을 통해 비유적 언어 번역의 어려움을 시도합니다.

- **Performance Highlights**: 비유적 표현의 번역은 문자 그대로의 표현과 다른 특성을 보이며, 비유적 언어 번역이 단순한 의미 맞춤을 넘어 문화적, 언어적 차이를 고려해야 함을 강조합니다. 이 연구는 기존의 기계 번역 모델이 비유적 언어 번역에서 나타내는 어려움을 체계적으로 조사하고 평가함으로써 비유적 언어 번역의 복잡성을 파악하는 데 기여합니다.



### Synchronous Faithfulness Monitoring for Trustworthy Retrieval-Augmented Generation (https://arxiv.org/abs/2406.13692)
- **What's New**: 새로운 동시모니터링기술 SynCheck가 제안되었습니다. SynCheck는 sequence likelihood, uncertainty quantification, context influence, semantic alignment 같은 다양한 신호를 활용하여 RALMs (Retrieval-Augmented Language Models)의 신뢰성을 즉각적으로 측정합니다. SynCheck는 여섯 가지 대규모 생성 작업(task)에서 신뢰성 오류 탐지에 있어 0.85 AUROC를 기록하며, 이전 최상의 방법보다 4% 높은 성과를 보였습니다. 또한, SynCheck를 바탕으로 FOD(신뢰성 지향 디코딩) 알고리즘이 도입되어 RALMs의 생성 신뢰성을 크게 개선합니다.

- **Technical Details**: SynCheck는 sequence likelihood, uncertainty quantification, context influence, semantic alignment 등의 신호를 미세하게 분석하며 동기화된 디코딩 과정에서 오류를 탐지합니다. 이 신호들은 효율적으로 측정되며, lightweight aggregator를 통해 통합됩니다. FOD(신뢰성 지향 디코딩) 알고리즘은 SynCheck의 피드백을 받아서 beam search로 더 신뢰성이 높은 출력을 생성하도록 유도합니다. FOD는 기존의 방법들, 예를 들면 abstention, reranking, 또는 contrastive decoding보다 큰 성능 향상을 보였습니다.

- **Performance Highlights**: SynCheck는 여섯 가지 데이터셋에서 평균 0.85 AUROC를 기록하며, 전통적인 방법을 4%에서 35%까지 초과합니다. 또한, FOD는 greedy search 대비 12%, abstention 대비 10%, reranking 대비 13%, 그리고 context-aware decoding (CAD) 대비 19% 향상된 신뢰성을 보여줍니다. 이 결과는 SynCheck가 다양한 작업에서 적용 가능하며 상당한 성능 향상을 가져올 수 있음을 증명합니다.



### Leveraging Large Language Models to Measure Gender Bias in Gendered Languages (https://arxiv.org/abs/2406.13677)
- **What's New**: 이 논문은 자연어 처리(NLP) 맥락에서 대형 언어 모델(LLMs)을 훈련하는 데 사용되는 스페인어 텍스트 코퍼스의 성별 편향을 분석하는 새로운 방법론을 도입합니다. 기존의 성별 편향 분석 방법은 주로 영어에 맞춰져 있어, 성별이 명시된 언어(예: 스페인어, 프랑스어)에는 적용하기 어렵습니다. 이 문제를 해결하기 위해 문맥 이해 능력을 가진 LLMs를 활용하여 스페인어 코퍼스에서 성별 편향을 정량적으로 분석하는 방법을 개발했습니다.

- **Technical Details**: 제안된 방법론은 LLMs를 사용하여 인간 엔티티를 참조하는 성별 명사와 대명사를 식별하고 분류합니다. 이는 LLMs의 문맥적 이해 능력을 활용하여 스페인어의 언어 구조를 존중하면서 필요한 깊이의 분석을 제공합니다. 우리는 이 방법론을 네 개의 광범위한 벤치마크 데이터셋에서 실증적으로 검증하여 성별 불균형을 발견했습니다. 분석 결과 남성 대 여성 비율이 4:1에서 6:1로 나타났습니다.

- **Performance Highlights**: 연구 결과, 스페인어 텍스트 코퍼스에서 성별 불균형이 매우 높다는 것이 드러났습니다. 흥미롭게도, 동일한 코퍼스의 영어 번역본에서는 남성 대 여성 비율이 1:1에서 3.5:1로 나타났습니다. 이러한 발견은 성별이 명시된 언어에서 성별 편향을 검출하기 위한 새로운 방법을 개발하는 중요성을 강조합니다.



### Model Internals-based Answer Attribution for Trustworthy Retrieval-Augmented Generation (https://arxiv.org/abs/2406.13663)
Comments:
          Under review. Code and data released at this https URL

- **What's New**: 최근 RAG(검색 보강 생성, Retrieval-Augmented Generation) 모델에서의 답변 신뢰성 문제를 해결하기 위해 MIRAGE(Model Internals-based RAG Explanations)를 소개합니다. 이는 모델 내부 구조를 활용해 정확하고 효율적인 답변 출처 명시를 가능케 하는 플러그 앤 플레이 접근법입니다.

- **Technical Details**: MIRAGE는 LLM(대형 언어 모델, Large Language Models)이 생성하는 문장 내 텍스트 토큰을 특정 문서와 연결지어 출처를 나타내는 방식입니다. 이를 위해 기울기 기반의 중요한 요소 탐지 기능이나 기타 특성 탐지 기법을 활용합니다. 기존의 자연어 추론(NLI) 기반 접근 방법이나 셀프 인용 방식과 달리, MIRAGE는 모델 내부 구조를 통해 더욱 정교한 제어가 가능합니다.

- **Performance Highlights**: MIRAGE는 다국어 추출 질문 답변 데이터셋에서 인간의 답변 출처 명시와 높은 일치를 보였습니다. 또한, 개방형 질문 답변 환경에서 셀프 인용(Self-Citation)에 필적하는 출처의 질과 효율성을 달성하며, 더욱 세밀한 출처 제어가 가능합니다.



### ObscurePrompt: Jailbreaking Large Language Models via Obscure Inpu (https://arxiv.org/abs/2406.13662)
- **What's New**: 최근 대형 언어 모델(LLMs)의 자연어 처리 능력이 주목받고 있으나, 이들의 신뢰성, 특히 '탈옥(jailbreaking)' 공격에 대한 우려가 남아있습니다. 기존 연구들은 주로 white-box 시나리오나 고정된 프롬프트 템플릿에 의존하는데, 이는 널리 적용하기 어렵습니다. 본 논문은 ObscurePrompt라는 간단하면서도 새로운 방법을 소개합니다. 이는 LLMs의 Out-of-Distribution (OOD) 데이터에서 취약한 정렬을 활용한 탈옥 방법입니다.

- **Technical Details**: ObscurePrompt는 잘 알려진 탈옥 기술을 통합한 기본 프롬프트를 만들고, 강력한 LLMs(GPT-4 등)을 사용하여 이 프롬프트를 반복적으로 변형하여 공격의 강건성을 강화합니다. 이를 통해 본래의 프롬프트를 모호한 텍스트로 변환함으로써, LLM의 윤리적 결정 경계를 약화시킵니다. 세 가지 주요 단계로 진행되며, 첫째는 다양한 프롬프트 엔지니어링 기술을 사용하여 시드 프롬프트를 만드는 것, 둘째는 강력한 LLM을 사용해 모호성 변환을 적용하는 것, 셋째는 이를 반복하여 통합된 공격을 생성하는 것입니다.

- **Performance Highlights**: 실험 결과, ObscurePrompt는 기존 방법들보다 공격 효과가 뛰어났으며, 주류 방어 메커니즘을 상대로도 유효성을 유지했습니다. 구체적으로, 프롬프트 수가 공격 성공률에 큰 영향을 미치며, 모든 유형의 탈옥 전략을 결합하는 것이 항상 가장 효과적이지는 않다는 점이 밝혀졌습니다. 이로써 LLMs는 여전히 모호한 입력에 취약하며, 이러한 취약점을 방어하기 위한 강화된 방어 조치가 필요함을 확인했습니다.



### Towards Minimal Targeted Updates of Language Models with Targeted Negative Training (https://arxiv.org/abs/2406.13660)
Comments:
          Published in Transactions of Machine Learning Research

- **What's New**: 이번 연구에서는 생성 언어 모델이 바람직하지 않은 출력을 최소화하면서 모델의 다른 동작을 최소한으로 변경하여 업데이트하는 '최소 타겟 업데이트(minimal targeted update)' 문제를 다룹니다. 이를 위해 사용된 방법은 Targeted Negative Training(TNT)으로, 모델의 세대를 기반으로 한 부정적인 예제를 사용하여 이루어집니다.

- **Technical Details**: 논문에서는 먼저 최소 타겟 업데이트의 개념을 정립하고, 기존의 부정 시그널을 사용하는 손실이 업데이트된 분포를 제대로 제어하지 못한다는 문제를 제기합니다. TNT는 부정적인 예제를 사용하여 새로운 분포를 원본에 가깝게 유지합니다. 이 방법은 추론(inference) 시점의 지연 없이 작업을 수행하므로 기존의 추론 시점 절차들과는 달리 예측 파이프라인의 복잡성을 줄일 수 있습니다.

- **Performance Highlights**: 실험 결과, TNT는 기존의 베이스라인들보다 원치 않는 행동을 줄이면서 모델의 생성 행동을 더 잘 유지하는 무역오프(trade-off)를 보여주었습니다. 이는 모델이 잘못된 출력을 생성하지 않도록 제어하면서도 그 우수한 성능을 유지하는 데 기여할 수 있습니다.



### Can Few-shot Work in Long-Context? Recycling the Context to Generate Demonstrations (https://arxiv.org/abs/2406.13632)
- **What's New**: 긴 텍스트 문맥(long context)을 다룰 때 일관되지 않고 비효율적인 대형 언어 모델(LLM)을 개선하기 위해 DoubleDipper라는 새로운 방법을 제안합니다. 이 방법은 주어진 입력 문맥을 재활용하여 few-shot 예제를 자동으로 생성함으로써, 문맥 길이를 최소화하고 정확도를 높입니다.

- **Technical Details**: DoubleDipper는 긴 문맥 질문 응답(QA) 작업을 위해 두 가지 주요 원칙을 따릅니다. 첫 번째로, 주어진 입력 문맥에서 일부 문단을 무작위로 선택하고, 각 문단에 대한 질문-응답(QA) 쌍을 생성하여, 이러한 생성된 질문-응답 쌍을 입력 질문 앞에 배치합니다. 두 번째로, 모델이 응답을 생성하기 전에 관련 정보를 포함한 문단을 명시적으로 식별하도록 지시하여, 모델이 관련 정보를 정확하게 찾아내는 능력을 향상시킵니다.

- **Performance Highlights**: DoubleDipper를 사용하여 다양한 LLM(예: Gemini Pro, Gemini Ultra, Llama, Mistral, Gemma)을 여러 QA 데이터셋에서 평가한 결과, 기존의 few-shot 예제보다 일관되게 더 높은 성능을 보였습니다. 특히, 질문의 답변이 문맥 중간에 위치한 경우 성능이 크게 개선되었으며, 단일 호핑(single-hop) 예제를 사용하더라도 다중 호핑(multi-hop) QA에서도 좋은 일반화 성능을 보였습니다.



### InstructRAG: Instructing Retrieval-Augmented Generation with Explicit Denoising (https://arxiv.org/abs/2406.13629)
Comments:
          Code: this https URL

- **What's New**: 인스트럭트RAG(InstructRAG)라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 언어 모델(LM)이 자체적으로 생성한 논리를 통해 명시적으로 노이즈를 제거하고 예측된 최종 답변을 정당화하도록 합니다.

- **Technical Details**: 인스트럭트RAG는 두 단계로 구성됩니다. 먼저, 문서에서 실제 답변이 어떻게 도출되는지 설명하는 논리를 문서와 함께 LM에 제공하여 노이즈 제거 논리를 생성하도록 지시합니다. 그런 다음, 이러한 논리를 예제 학습 또는 감독된 미세 조정 데이터로 사용해 LM이 노이즈를 명시적으로 제거하도록 학습시킵니다. 이를 통해 인스트럭트RAG는 추가적인 감독 없이도 향상된 생성 정확도를 달성할 수 있습니다.

- **Performance Highlights**: 기존의 RAG 방법에 비해 인스트럭트RAG는 평균적으로 8.3% 향상된 성능을 보였습니다. 실험 결과, 인스트럭트RAG는 다양한 노이즈 비율과 도메인 외 데이터셋에서도 일관된 노이즈 제거 능력을 발휘하며 강력한 일반화 성능을 나타냈습니다.



### Fine-Tuning Gemma-7B for Enhanced Sentiment Analysis of Financial News Headlines (https://arxiv.org/abs/2406.13626)
- **What's New**: 오늘날 금융 뉴스 헤드라인을 통한 투자자 감정 분석에 관한 연구에서는 금융 뉴스 헤드라인의 감정을 분석하여 투자자 감정을 이해하는 데 중점을 두고 있습니다. 자연어 처리(NLP)와 대형 언어 모델(LLM)을 활용하여 소매 투자자의 관점에서 감정을 분석합니다. FinancialPhraseBank 데이터셋을 기반으로 여러 모델(distilbert-base-uncased, Llama, gemma-7b)을 미세 조정하여 그 효과를 평가하였습니다. 미세 조정된 gemma-7b 모델이 정밀도, 재현율, F1 점수에서 가장 높은 성능을 보여주는 것으로 나타났습니다.

- **Technical Details**: 감정 분석을 위한 데이터를 Kaggle에서 제공하는 FinancialPhraseBank 데이터셋을 사용했습니다. 이 데이터셋은 긍정, 중립, 부정의 세 가지 감정으로 분류된 뉴스 헤드라인을 포함하고 있습니다. 미세 조정한 모델로는 distilbert-base-uncased, Llama, 그리고 gemma-7b를 사용하였습니다. 특히 gemma-7b 모델은 금융 감정을 정확하게 포착하는 데 있어 매우 강력한 성능을 입증했습니다.

- **Performance Highlights**: 미세 조정된 gemma-7b 모델은 최고 수준의 정밀도, 재현율, 그리고 F1 점수를 달성하며 다른 모델들을 능가하는 결과를 보였습니다. 이 모델은 금융 뉴스 헤드라인의 감정을 정확하게 예측하는 데 매우 효과적이며, 시장 통찰력 제공, 위험 관리, 투자 결정 지원에 유용할 것으로 예상됩니다.



### Improving Visual Commonsense in Language Models via Multiple Image Generation (https://arxiv.org/abs/2406.13621)
- **What's New**: 새로운 방법을 소개하여 대형 언어 모델(LLMs)의 시각적 상식을 향상시키고자 합니다. 이 방법은 입력 텍스트 프롬프트를 기반으로 여러 이미지를 생성하고, 이를 모델의 의사결정 과정에 통합하여 예측 확률을 혼합합니다.

- **Technical Details**: 특히, 우리의 방법은 마지막 단계의 융합 계층(late-fusion layer)을 사용하여 사전 훈련된 LLM의 출력과 투영된 시각적 특징을 결합합니다. 이러한 융합 계층은 이미지-텍스트 지식을 기반으로 예측을 가능하게 하며, 텍스트 기반 예측도 지원합니다. 우리는 사전 훈련된 텍스트-이미지 모델을 사용하여 입력 텍스트를 조건으로 여러 이미지를 생성하고 이를 시각적으로 강화된 LLM에 입력하여 여러 예측 확률 벡터를 생성한 후, 이 모든 확률 벡터를 가중 평균하여 최종 출력을 생성합니다.

- **Performance Highlights**: 우리의 접근 방식을 여러 시각적 상식 추론 과제와 전통적인 NLP 작업에 적용한 결과, 기존 기준 모델들보다 월등한 성능을 나타냈습니다. 최근 최첨단 LLM들(Llama3 등)에 적용했을 때, 시각적 상식뿐만 아니라 전통적인 NLP 벤치마크에 있어서도 성능 향상을 보였습니다.



### In-Context Former: Lightning-fast Compressing Context for Large Language Mod (https://arxiv.org/abs/2406.13618)
- **What's New**: 새로운 논문에서는 In-Context Former (IC-Former)를 제안합니다. 이는 기존 LLMs의 고비용 추론 문제를 해결하면서 문맥 압축을 단순하고 효율적으로 수행하는 방법입니다. 특히, IC-Former는 cross-attention 메커니즘과 소수의 학습 가능한 digest tokens를 활용하여 문맥 정보를 직접 응축합니다. 이로 인해 압축 범위 내에서 시간 복잡도가 선형적으로 성장하며, 압축 과정에서의 비용을 크게 줄입니다.

- **Technical Details**: IC-Former는 자기 주의 메커니즘(self-attention)에 의존하지 않고, 교차 주의 메커니즘(cross-attention)을 활용하여 문맥 단어 임베딩(Contextual Word Embeddings)에서 정보를 추출합니다. 소수의 수학 가능한 digest tokens가 압축 과정의 핵심 요소로 사용되며, 이는 문맥 정보를 효율적으로 압축 벡터로 전환합니다. 이를 통해 IC-Former는 압축 과정에서 시간 복잡도를 선형으로 유지합니다. 또한, pre-training 및 fine-tuning 전략을 사용하여 모델을 최적화합니다.

- **Performance Highlights**: 실험 결과, IC-Former는 압축 중 baseline 대비 단지 1/32의 부동 소수점 연산만 필요로 하며, 처리 속도가 68에서 112배 더 빠릅니다. 평가 지표에서도 baseline 성능의 90% 이상을 유지함으로써 높은 비용 효율성을 보입니다. 요약하자면, IC-Former는 경량화와 효율성을 동시에 달성하면서도 상당한 성능을 발휘합니다.



### Optimizing Psychological Counseling with Instruction-Tuned Large Language Models (https://arxiv.org/abs/2406.13617)
Comments:
          9 pages

- **What's New**: 이 논문은 대형 언어 모델(LLM)들이 심리 상담(Psychological Counseling) 분야에서 어떻게 응용될 수 있는지를 탐구합니다. 특히, 심리 상담 서비스에 대한 수요 증가를 해결하기 위해 LLM을 사용하여 공감적이고 관련성 있으며 지지적인 응답을 제공하는 방법을 제시하고 있습니다. 논문은 전문 카운슬러의 피드백을 바탕으로 개발된 상담 전용 프롬프트를 통해 LLM을 튜닝하는 방식을 설명합니다.

- **Technical Details**: 논문에서는 다양한 카운슬링 시나리오를 위한 포괄적인 프롬프트 데이터셋을 개발하여 LLM을 튜닝하는 방법을 제시합니다. 이 과정은 실제 상담 세션에서 얻은 피드백을 반영하여 프롬프트를 반복적으로 개선하는 피드백 루프를 포함합니다. 또한, 새로운 평가 데이터셋을 수집하여 모델의 성능을 엄격히 평가하고, 객관성을 확보하기 위해 GPT-4를 사용하여 평가합니다. 프롬프트는 적극적 청취, 공감 표현, 인지 재구성, 위기 개입 등을 포함한 상담 기법을 기반으로 설계되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법을 통해 튜닝된 모델은 기존 일부 LLM 대비 우수한 성능을 보였습니다. 이는 우리의 접근 방식이 심리 상담 지원 도구로서의 가능성을 시사합니다. 모델의 응답 품질은 자동 매트릭과 인간 평가를 모두 통해 측정되었으며, 상담 시나리오에서 공감적이고 적절한 응답을 제공하는 능력이 입증되었습니다.



### Enhancing Distractor Generation for Multiple-Choice Questions with Retrieval Augmented Pretraining and Knowledge Graph Integration (https://arxiv.org/abs/2406.13578)
Comments:
          Findings at ACL 2024

- **What's New**: 이 논문은 객관식 문제(MCQ)를 위한 오답지(distractor) 생성 작업을 다루고 있습니다. 두 가지 주요 설계가 도입되었으며, 첫 번째는 'retrieval augmented pretraining'으로 언어 모델의 사전 훈련을 DG(오답지 생성) 작업에 보다 밀접하게 맞추는 방법입니다. 두 번째는 지식 그래프(knowledge graphs)를 통합하여 DG의 성능을 향상시키는 것입니다.

- **Technical Details**: 이 연구는 두 가지 주요 방법을 제안합니다. 첫 번째는 'Retrieval Augmented Pretraining(RAP)'입니다. MCQ 답변을 사용하여 대규모 코퍼스(예: Wikipedia)에서 관련 문장이나 구절을 검색하고, 생성된 의사 질문(pseudo question)을 사용하여 작업 특화 사전 훈련을 수행합니다. 두 번째로는 'Knowledge Augmented Generation(KAG)'로 지식 삼출물(knowledge triplets)을 검색하여 text2text DG 모델에 보조 정보로 제공하는 것입니다.

- **Performance Highlights**: 제안된 모델은 벤치마킹 데이터셋에서 기존 최첨단 성능을 크게 능가합니다. 최고의 성능을 보인 모델은 MCQ 데이터셋에서 F1@3 점수를 14.80에서 16.47로, Sciq 데이터셋에서는 15.92에서 16.50로 향상시켰습니다.



### Lexically Grounded Subword Segmentation (https://arxiv.org/abs/2406.13560)
Comments:
          8 pages (+ 8 pages appendix), 2 figures

- **What's New**: 이 논문에서는 토큰화와 서브워드(segmentation) 분야에서 세 가지 혁신을 제안합니다. 첫 번째로, 모프서(Morfessor)를 이용한 비지도형 형태소 분석을 사전 토큰화에 활용합니다. 두 번째로, 단어 임베딩 공간을 기반으로 한 서브워드 임베딩(subword embeddings) 생성 방법과 이를 이용한 새로운 서브워드 분할 알고리즘을 설계했습니다. 세 번째로, 서브워드 바이그램(bigram) 모델에 기반한 효율적인 분할 알고리즘을 도입해 추론 시에는 Morfessor나 대형 임베딩 테이블을 사용할 필요가 없도록 했습니다. 이러한 방법들을 두 가지 내적 평가 지표와 품사 태그(part-of-speech tagging), 기계 번역(machine translation) 등 두 가지 다운스트림 작업에서 평가했습니다.

- **Technical Details**: 이 논문은 토큰화 과정을 세 단계로 나눕니다: 사전 토큰화, 어휘(vocabulary) 구성, 그리고 분할(segmentation)입니다. 사전 토큰화에는 Morfessor를 이용해 텍스트를 형태소적으로 분석합니다. 서브워드 임베딩 생성은 사전 학습된 단어 임베딩 모델과 학습 코퍼스를 사용해 합니다. 이 임베딩을 기반으로 서브워드 분할 알고리즘을 설계하며, 최종적으로 서브워드 바이그램 통계를 이용한 효율적인 분할 알고리즘도 제시합니다. 이 방법들은 SIGMORPHON 2018 데이터셋을 사용해 평가되었으며, 품사 태깅을 위한 전반적인 성능 향상을 보여줍니다.

- **Performance Highlights**: 제안된 방법은 형태소 경계에서의 분할 정밀도와 Rényi 효율성에서 8개 언어에서 유의미한 향상을 나타냈습니다. 기계 번역 품질에 큰 영향을 미치지 않았으나, 형태소적으로 더 중요한 품사 태깅 작업에서 일관되게 성능 향상을 관찰했습니다.



### Evaluating Short-Term Temporal Fluctuations of Social Biases in Social Media Data and Masked Language Models (https://arxiv.org/abs/2406.13556)
- **What's New**: 소셜 미디어 데이터로 학습된 Masked Language Models (MLMs)에 포함된 사회적 편향(social biases)이 시간이 지남에 따라 어떻게 변화하는지를 분석합니다. 소셜 미디어 사용자 수가 폭발적으로 증가함에 따라, 이러한 편향이 증폭되는지 여부를 조사합니다.

- **Technical Details**: 시간 순서대로 정렬된 코퍼스의 스냅샷을 사용하여 사전 학습된 여러 MLMs를 분석했습니다. 다양한 유형의 사회적 편향을 평가하기 위해 여러 벤치마크 데이터를 사용했습니다. 전반적으로 편향이 시간에 걸쳐 안정적으로 유지되지만, 일부 편향(인종, 피부색, 종교, 성적 지향 등)은 시간에 따라 변동이 있습니다.

- **Performance Highlights**: 연구 결과, 사회적 편향 평가 시 전반적인 편향 점수에만 의존하는 것은 오해를 불러일으킬 수 있음을 발견했습니다. 이는 개별 편향 점수를 평가하는 것이 중요한 이유입니다. 주요 발견 중 하나는 훈련 코퍼스에서 남성이 여성보다 높은 선호도를 가지는 경우가 지속되고 있다는 점입니다.



### BiLD: Bi-directional Logits Difference Loss for Large Language Model Distillation (https://arxiv.org/abs/2406.13555)
Comments:
          Submitted to ARR June (for EMNLP 2024)

- **What's New**: 최근 몇 년 동안, 대형 언어 모델(LLMs)은 다양한 자연어 처리(NLP) 작업에서 뛰어난 성능을 보여주었습니다. 하지만 이러한 뛰어난 성능은 매개 변수의 크기가 커지는 문제와 맞물려 널리 보급되기 어렵다는 단점이 있습니다. 본 논문에서는 LLMs의 로그잇 수준(logit level)에서의 과제 특화 distillation을 탐구합니다. 우리는 고유의 'Bi-directional Logits Difference (BiLD)' 손실을 제안하며, 이는 잡음을 줄이고, 내부 로그잇 순위를 활용함으로써 성능을 향상시킵니다.

- **Technical Details**: LLMs의 로그잇은 비전 모델(vision models)과 달리 더 극단적인 긴 꼬리(long-tail) 분포를 보입니다. 이러한 긴 꼬리에 숨겨진 '잡음(noise)'이 distillation 성능에 영향을 미칩니다. 기존의 로그잇 distillation 방법은 로그잇의 내부 순위 정보를 효과적으로 활용하지 못하는 경우가 많습니다. 이를 해결하기 위해, 우리는 상위 k개의 로그잇만을 사용하고, 내부 로그잇 순위 정보를 활용하여 로그잇 차이를 구성하는 BiLD 손실을 제안합니다.

- **Performance Highlights**: 13개의 데이터 세트를 대상으로 한 종합 실험에서, BiLD 손실을 사용하는 상위 8개의 로그잇만으로 감독된 미세 조정(supervised fine-tuning, SFT), 일반 KL 손실(vanilla KL loss), 그리고 다른 distillation 방법들보다 뛰어난 성능을 보여주었습니다.



### Mining United Nations General Assembly Debates (https://arxiv.org/abs/2406.13553)
Comments:
          4 pages, 1 figure, 2 tables

- **What's New**: 이 논문은 자연어 처리(NLP) 기술을 사용하여 유엔 총회(UNGA) 연설을 분석하는 프로젝트에 대해 다룹니다. 이는 대규모 텍스트 데이터를 효율적으로 처리하고, 의미 패턴 추출, 감정 분석, 토픽 모델링을 가능하게 합니다. 정치 학자들이 국제 관계에 대한 통찰을 얻고, 글로벌 외교 담론에 대한 세밀한 이해를 높이는 데 활용할 수 있는 포괄적인 데이터셋과 도구를 제공하는 것이 목표입니다.

- **Technical Details**: 데이터셋은 1946년부터 2023년까지의 UNGA 연설을 담고 있으며, 날짜, 연설자의 이름 및 역할 등 여러 추가 특징으로 보완되었습니다. 이후, 변환기(transformer)를 기반으로 하는 토픽 모델링 기술인 BERTopic을 적용하여 연설의 주제와 국제적 우려를 표현하는 언어를 분석했습니다. BERTopic은 사전 학습된 대형 언어 모델을 활용하여 문서 내 단어의 의미적 유사성을 바탕으로 주제를 생성합니다.

- **Performance Highlights**: 업데이트된 UNGD 코퍼스는 2023년 연설을 포함하여 총 10,679개의 연설을 포함합니다. 메타데이터의 무결성을 개선하기 위해 반복적인 특징 엔지니어링 프로세스를 통해 ISO 코드, 국가명, 연설자 이름, 직위 등의 오류를 수정하고 표준화했습니다. BERTopic은 토픽 일관성(coherence) 및 다변성(diversity) 측정 기준으로 평가되었으며, DistilBERT 임베딩 방법이 가장 좋은 성능을 보였습니다. 전체 결과는 Streamlit 데이터 앱 프레임워크를 사용해 개발된 인터랙티브 애플리케이션에 통합되어, 비기술적인 사용자도 데이터를 쉽게 탐색하고 분석할 수 있게 되었습니다.



### Mitigating Social Biases in Language Models through Unlearning (https://arxiv.org/abs/2406.13551)
- **What's New**: 최근 언어 모델(LM)에서 편향(bias) 문제를 해결하는 연구가 활발히 진행되고 있습니다. 본 연구에서는 머신 언러닝(machine unlearning) 기법을 활용하여 기존 프리트레인 또는 파인튜닝된 모델의 원하지 않는 행동(undesired behaviors)을 제거하고자 합니다. 특히, 최첨단 및 오픈소스 언어 모델인 LLaMA-2와 OPT에서 이러한 방법을 적용했습니다.

- **Technical Details**: 두 가지 언러닝 방법을 탐구했습니다. (1) Partitioned Contrastive Gradient Unlearning(PCGU)은 디코더 모델에서 적용되었고, (2) Negation via Task Vector 방법은 편향성을 줄이는 데 사용되었습니다. 또한, 큰 모델에 대해서는 분산된 PCGU(distributed PCGU)를 구현하였습니다. BBQ 데이터셋을 활용하여 디코딩 모델의 자동회귀 특성을 고려한 편향 처리 방식을 개발하였습니다.

- **Performance Highlights**: 실험 결과, Negation via Task Vector 방법이 PCGU 방법보다 더 효과적으로 편향성을 줄여주면서 모델의 성능 및 퍼플렉시티(perplexity)에 미치는 영향이 최소화됨을 실증적으로 확인하였습니다. 예를 들어, LLaMA-27B 모델에서 Negation via Task Vector를 사용하여 편향 점수를 약 11.8% 줄였습니다.



### Self-play with Execution Feedback: Improving Instruction-following Capabilities of Large Language Models (https://arxiv.org/abs/2406.13542)
- **What's New**: AutoIF라는 새로운 방법을 도입하여 대규모 언어 모델(LLMs)의 복잡한 지시를 자동으로 생성하고 따를 수 있게 하는 훈련 데이터셋을 생성할 수 있는 확장 가능하고 신뢰할 수 있는 방법을 제안합니다. AutoIF는 코드 검증을 통해 지시에 따른 응답의 정확성을 검증하도록 설계되었습니다.

- **Technical Details**: AutoIF는 (1) 코드로 검증 가능한 지시를 자동으로 생성하고, (2) 이러한 지시에 대한 검증 코드를 자동으로 생성하며, (3) 최초의 두 단계를 신뢰할 수 있도록 하는 것을 포함합니다. LLMs는 검증 코드와 단위 테스트 케이스를 생성하며, 코드가 정확히 컴파일되고 테스트 케이스를 통과한 경우에만 유지됩니다. 실제 훈련은 SFT(Supervised Fine-Tuning)와 RLHF(Reinforcement Learning from Human Feedback)를 사용하여 수행됩니다.

- **Performance Highlights**: AutoIF는 세 가지 훈련 알고리즘(SFT, Offline DPO, Online DPO)에서 Qwen2 및 LLaMA3와 같은 대표적인 오픈소스 LLMs에 적용되어 상당한 성능 개선을 달성했습니다. 특히, Qwen2-72B와 LLaMA3-70B 모델은 IFEval 벤치마크에서 최대 90% 이상의 정확도를 기록했습니다. FollowBench 벤치마크에서도 SSR 척도가 5% 이상 향상되었습니다.



### ManWav: The First Manchu ASR Mod (https://arxiv.org/abs/2406.13502)
Comments:
          ACL2024/Field Matters

- **What's New**: 이 연구는 자동 음성 인식 (ASR) 연구에서 고자원 및 극저자원 언어 간의 격차를 다루며, 특히 심각하게 위험에 처한 언어인 만주어에 초점을 맞추고 있습니다. 연구진은 최초로 만주어 ASR 모델 ManWav를 소개하며, Wav2Vec2-XLSR-53을 활용해 모델의 성능을 크게 향상시켰습니다. 특히, 데이터 증강을 통해 0.02의 CER (Character Error Rate) 및 0.13의 WER (Word Error Rate) 감소를 달성했습니다.

- **Technical Details**: Wav2Vec2-XLSR-53 모델을 기반으로 하여 만주어 데이터를 사용해 두 가지 다른 유형의 데이터로 미세 조정했습니다. 원본 만주어 데이터와 증강 만주어 데이터로 훈련된 두 가지 모델을 개발하는 과정에서, 증강 데이터는 배경 소음 추가 (Additive Noise), 신호 클리핑 (Clipping), 잔향 효과 (Reverberation), 그리고 시간 부분 제거 (Time Dropout) 등의 방법을 사용했습니다. 데이터 증강은 WavAugment를 통해 구현되었으며, 데이터 셋을 원본의 400%로 확장했습니다.

- **Performance Highlights**: 증강 데이터를 사용하여 모델을 훈련한 결과, CER가 0.02, WER가 0.13 감소하여 성능이 향상되었습니다. 이는 제한된 데이터가 사용된 상황에서도 증강 데이터가 얼마나 효과적인지 보여주는 결과입니다. ManWav 모델은 특히 제한된 만주어 음성 데이터에도 불구하고 유의미한 수준의 정확도를 달성했습니다.



### LLMs Are Zero-Shot Context-Aware Simultaneous Translators (https://arxiv.org/abs/2406.13476)
- **What's New**: 최신 연구에서는 공개 소스 대형 언어 모델(LLMs)을 사용하여 동시 번역(Simultaneous Machine Translation, SiMT) 작업에서 최첨단 기준과 동일하거나 더 나은 성능을 발휘하는지에 대해 조사했습니다. 특히, 최소한의 배경 정보를 주입함으로써 기술적인 주제에서 성능이 더 향상될 수 있음을 보여줍니다. 이는 자원 집약적인 훈련이나 미세 조정을 필요로 하지 않는 차세대 다중 언어, 문맥 인식 및 용어적으로 정확한 SiMT 시스템을 구축할 수 있는 LLM의 잠재력을 강조합니다.

- **Technical Details**: 동시 번역(SiMT)은 번역자가 원문 문장이 끝나기 전에 번역을 시작해야 하며, 이는 종종 특정 단어 또는 구의 의미에 대한 강한 가정을 필요로 합니다. 기존 SiMT 시스템은 문맥과 광범위한 문맥(추가적인 텍스트 외부의 정보)을 무시하는 경우가 많아 논리적 일관성이 떨어지고 용어적인 일관성 문제가 발생할 수 있습니다. 이에 반해 인간 번역자는 관련 주제와 용어를 공부하여 번역의 정확성을 높입니다. 본 연구에서는 사전 학습된 명령 조정(Instruction-tuned) LLM을 사용하여 배경 정보를 삽입하는 방식으로 SiMT의 성능을 향상시켰으며, 이는 더욱 상황에 맞는 단어 선택에 도움이 됩니다.

- **Performance Highlights**: 실험 결과, 우리의 접근 방식은 일부의 강력한 이중 언어 SiMT 기준을 능가하며, 최첨단 다중 언어 SiMT 시스템과 견줄만한 경쟁력을 가집니다. 특히, 초기 번역 후 답변을 수정하는 '응답 초기화(response priming)' 기법을 사용하여 LLM의 제로샷(Zero-shot) 성능을 향상시켰습니다. 이러한 방식으로 LLM을 사용하면 복잡한 세분화 정책 없이도 SiMT 작업에서 성공적으로 수행할 수 있음을 확인했습니다.



### Encoder vs Decoder: Comparative Analysis of Encoder and Decoder Language Models on Multilingual NLU Tasks (https://arxiv.org/abs/2406.13469)
Comments:
          14 pages, 2 figures

- **What's New**: 이 논문은 ScandEval 벤치마크를 확장하여 디코더 모델(decoder models)을 포함시키고, 덴마크어, 스웨덴어, 노르웨이어, 아이슬란드어, 페로어, 독일어, 네덜란드어, 영어 등의 언어에서 다국어 자연어 이해(멀티링궈 NLU) 작업에 대한 성능을 평가합니다. 디코더 모델의 성능 평가를 위한 새로운 방법을 도입하여 엔코더 모델과의 비교를 수행합니다.

- **Technical Details**: 이 연구는 자연어 이해(NLU) 작업을 통해 엔코더 모델과 디코더 모델의 성능을 비교합니다. 엔코더 모델은 BERT와 같은 구조를 사용하며, 디코더 모델은 GPT-3와 같은 구조를 기반으로 합니다. 연구는 다양한 NLU 작업과 다양한 언어 자원 스펙트럼 상에서 두 모델의 성능을 실험적 방법으로 분석합니다. UMAP 분석을 통해 디코더 모델과 엔코더 모델의 성능 경향을 시각화합니다.

- **Performance Highlights**: 디코더 모델은 엔코더 모델보다 다국어 NLU 작업에서 더 나은 성능을 발휘할 수 있음을 발견했습니다. 그러나 이 성능 차이는 작업 유형 및 언어에 따라 다소 달라질 수 있습니다. 특히, 디코더 모델은 질문 응답 작업에 대해 현저한 성능 우위를 보였습니다. UMAP 분석을 통해 디코더 모델의 성능 경로가 엔코더 모델과는 다른 '경로'를 따르는 것을 확인했습니다.



### VDebugger: Harnessing Execution Feedback for Debugging Visual Programs (https://arxiv.org/abs/2406.13444)
- **What's New**: VDebugger는 시각적 프로그램(Visual Programs)을 단계별로 추적하며 디버깅하는 새로운 프레임워크로, 프로그램 오류를 식별하고 수정하는 데 중점을 둡니다. 이 프레임워크는 자동화된 파이프라인을 통해 학습 데이터를 생성하고 새로운 마스크-베스트 디코딩 기술을 사용하여 오류를 주입합니다.

- **Technical Details**: VDebugger는 비평가(critic)와 정제기(refiner)로 구성되며, 각 실행 단계에서 발생한 오류를 세밀하게 식별하고 수정합니다. 학습 데이터는 기존 올바른 시각적 프로그램에 오류를 주입하여 생성됩니다. 비평가는 프로그램의 오류 라인을 식별하고, 정제기는 이를 수정합니다.

- **Performance Highlights**: VDebugger는 CodeLlama-7B 및 CodeLlama-13B 모델 기반으로 구성되어 있으며, 6개의 다양한 데이터셋에서 최대 3.2%의 성능 향상을 보였습니다. 또한 GPT-3.5와 같은 독점 코드 생성 모델에도 적용되어 최대 4.9%의 정확도 향상을 달성했습니다. VDebugger는 이전에 보지 못한 작업에서도 2.3%의 정확도 향상과 같은 놀라운 일반화 능력을 보여주었습니다.



### Dual-Phase Accelerated Prompt Optimization (https://arxiv.org/abs/2406.13443)
- **What's New**: 이 논문은 폐소스 대형 언어 모델(LLMs)의 성능을 향상시키기 위한 새로운 이중 단계 접근법을 제안합니다. 이 방식은 고품질 초기 프롬프트를 생성하고 이를 문장 수준에서 최적화하는 것입니다. 이 접근법은 기존 방법들이 직면하는 낮은 수렴률 문제를 해결하고자 합니다.

- **Technical Details**: 제안된 방법은 두 단계를 포함합니다. 첫 번째 단계는 잘 설계된 메타-인스트럭션(meta-instruction)을 사용하여 고품질의 초기 프롬프트를 생성하는 것입니다. 이 초기 프롬프트는 작업 유형, 출력 형식, 제약 조건, 제안된 추론 과정 및 전문적인 팁을 포함합니다. 두 번째 단계는 이전 튜닝 경험을 활용하여 초기 프롬프트를 확장하고 효과적인 프롬프트를 수용함으로써 문장 수준에서 프롬프트를 최적화하는 것입니다.

- **Performance Highlights**: 8개의 데이터셋에서 광범위한 실험을 통해 제안된 방법의 효과성을 입증하였습니다. 기존 기준선 대비 일관된 정확도 향상을 달성했으며, 최적화 단계는 5단계를 넘지 않았습니다. 또한, 이 방식은 기존의 그라디언트 프리 프롬프트 최적화 방법들이 요구하는 과도한 최적화 단계 수요를 대폭 줄였습니다.



### Finding Blind Spots in Evaluator LLMs with Interpretable Checklists (https://arxiv.org/abs/2406.13439)
- **What's New**: 최근 논문에서는 대형 언어 모델(LLMs)이 텍스트 생성 작업을 평가하는 역할이 증가하고 있으며, 이러한 평가의 정확성과 신뢰성에 대한 우려가 제기되고 있습니다. 이에 대하여 FBI라는 새로운 프레임워크를 제안하였습니다. 이 프레임워크를 통해 LLMs 평가자의 효과성을 평가하고, 사실적 정확성(factual accuracy), 지시사항 준수(instruction following), 장문 작성의 일관성(coherence in long-form writing), 그리고 추론 능력(reasoning proficiency)을 중점으로 분석합니다.

- **Technical Details**: 프레임워크는 LLMs가 생성한 답변에 특정한 교란(perturbations)을 도입하여 평가자가 이러한 품질 저하를 감지할 수 있는지 테스트합니다. 총 2400개의 교란된 답변이 22개의 교란 범주를 포함하며, 이를 통해 다섯 개의 주요 평가자 LLMs를 다양한 평가 전략으로 분석합니다. 평가 전략은 단일 답변 평가(single-answer evaluation), 쌍간 평가(pairwise evaluation), 그리고 참조 기반 평가(reference-guided evaluation)로 나뉩니다.

- **Performance Highlights**: 연구 결과는 현재의 평가자 LLMs가 텍스트 생성 작업에 있어 신뢰할 수 없음을 보여줍니다. 평균적으로 50% 이상의 사례에서 품질 저하를 감지하지 못했습니다. 특히 단일 답변 평가와 쌍간 평가에서는 심각한 한계가 나타났으며, 교란을 정확히 식별했음에도 점수를 조정하지 않는 경우가 있었습니다. 참조 기반 평가(reference-based evaluation)에서 비교적 나은 성능을 보였으나, 전반적으로 평가 LLMs의 신뢰성에는 여전히 많은 문제가 있음을 확인하였습니다.



### Children's Speech Recognition through Discrete Token Enhancemen (https://arxiv.org/abs/2406.13431)
Comments:
          Accepted at Interspeech 2024

- **What's New**: 이 논문은 어린이 음성 인식(ASR) 시스템에 새로운 접근 방식을 제안합니다. 저자들은 음성 신호를 민감한 정보를 포함하지 않는 이산 토큰(discrete tokens)으로 변환하여, 개인정보 문제를 해결하는 동시에 언어적 및 음향 정보를 모두 포착할 수 있는 방법을 탐구했습니다. 특히, 단일 및 다중 관점 전략(single-view and multi-view strategies)을 활용하여 이러한 이산 레이블을 생성하고, 이전에 본 적 없는 도메인 및 원어민이 아닌 데이터셋에서도 모델의 일반화(generalization) 능력을 테스트했습니다.

- **Technical Details**: 이 연구에서는 사전 학습된 SSL 모델의 프레임 수준 임베딩(frame-level embeddings)을 사용해 이를 이산 토큰(discrete tokens)으로 양자화(quantized)했으며, 이는 단일 관점(single-view) 또는 다중 관점(multi-view) SSL 모델을 사용해 k-mean 클러스터링을 통해 생성되었습니다. 그런 다음 이 이산 토큰을 엔드투엔드(End-to-End) ASR 모델의 입력으로 사용했습니다. 단순 벡터 양자화(vector quantization) 방식을 채택하여 프레임 수준 임베딩을 고정 코드북 사이즈로 근사화했고, 이를 통해 이산 코드를 추출했습니다. 두 가지 전략: 단일관점 전략(D(S)) 및 다중관점 코드북 전략(D(MV))을 사용하여 코드북을 훈련했습니다.

- **Performance Highlights**: 이산 토큰 ASR 모델의 성능은 기존 모델 대비 약 83%의 파라미터를 줄이면서 거의 동일한 성능을 달성했습니다. 제안된 시스템은 특히 원어민이 아닌 어린이가 포함된 데이터셋 및 읽기/즉흥 연설 스타일을 포함한 새로운 도메인에서도 효과적임을 나타냈습니다. 종합적으로 이 논문은 어린이 음성 인식 시스템을 위한 이산 토큰의 가능성을 실증하고, Whisper 모델과 같은 최첨단 기술과 비교해 비슷한 성능을 유지하는 것을 보여주었습니다.



### Factual Confidence of LLMs: on Reliability and Robustness of Current Estimators (https://arxiv.org/abs/2406.13415)
Comments:
          accepted on the main track of ACL 2024

- **What's New**: 최근 대형 언어 모델(LLMs)의 정당성 문제를 해결하기 위해 다양한 방법들이 제안되었습니다. 이번 논문에서는 이러한 방법들을 체계적으로 비교하고 실험적 프레임워크를 도입하여 객관적인 비교를 가능하게 했습니다.

- **Technical Details**: LLMs의 사실 신뢰도를 평가하기 위해 두 가지 주요 방법을 사용했습니다. 하나는 사실 검증(fact-verification)으로, 주어진 문장의 진위 여부를 평가하는 P(True)입니다. 다른 하나는 질문 응답(question answering)으로, 질의에 대한 올바른 답변을 제공할 가능성을 평가하는 P(I Know)입니다. 실험에서는 LLMs의 비일관성을 확인하기 위해 의미 보존 변형을 사용했습니다.

- **Performance Highlights**: 여덟 가지 공개된 LLM을 대상으로 수행된 실험 결과, 훈련된 히든-스테이트 프로브(hidden-state probes) 방식이 가장 신뢰할 수 있는 것으로 나타났지만, 모델의 웨이트와 훈련 데이터 접근을 필요로 합니다. 반면, 프롬프팅 기반 방법(prompting-based methods)은 신뢰성이 낮았습니다. 또한, 의미 보존 입력 변형에 대한 LLM의 불안정성을 발견하여, 모델 안정성에 개선의 여지가 많음을 시사합니다.



### SQLFixAgent: Towards Semantic-Accurate SQL Generation via Multi-Agent Collaboration (https://arxiv.org/abs/2406.13408)
- **What's New**: 새로운 다중 에이전트 협업 프레임워크인 SQLFixAgent를 소개합니다. 이는 대형 언어 모델(LLMs)이 생성한 SQL 쿼리에서 발생하는 구문 및 의미 오류를 탐지하고 수정하는 데 중점을 둡니다. 이 프레임워크는 핵심 에이전트(SQLRefiner)와 두 가지 보조 에이전트(SQLReviewer, QueryCrafter)로 구성되어 있습니다.

- **Technical Details**: SQLFixAgent는 사용자의 자연 언어 질문을 SQL 쿼리로 변환하는 기존의 Text-to-SQL 방식보다 발전된 방식입니다. 주 에이전트인 SQLRefiner는 보조 에이전트들이 제공하는 정보와 예제를 바탕으로 최적의 SQL 쿼리를 선택합니다. SQLReviewer는 'Rubber Duck Debugging' 방법을 사용하여 SQL 문과 사용자 쿼리 간의 잠재적 의미 불일치를 식별하며, QueryCrafter는 fine-tuned SQLTool을 사용해 여러 후보 SQL 쿼리를 생성합니다.

- **Performance Highlights**: SQLFixAgent는 다섯 개의 Text-to-SQL 벤치마크에서 테스트되었으며, Bird 벤치마크에서 실행 정확도가 3% 이상 향상되었습니다. 또한, 다른 고급 방법들에 비해 더 높은 토큰 효율성을 보여주며 경쟁력을 갖추고 있습니다.



### MoreHopQA: More Than Multi-hop Reasoning (https://arxiv.org/abs/2406.13397)
Comments:
          8 pages, 5 figures. First three authors contributed equally

- **What's New**: 기존의 대부분 멀티홉(Multi-hop) 데이터셋은 질문에 대한 답변이 제공된 문맥에서 직접 추출될 수 있는 추출형(Extractive) 답변 데이터셋입니다. 이는 모델들이 진정한 멀티홉 추론 대신에 휴리스틱이나 지름길을 사용하는 경향이 있습니다. 이에 반해, 새로운 멀티홉 데이터셋인 MoreHopQA는 추출형 답변에서 생성형(Generative) 답변으로 전환하였습니다. 이 데이터셋은 HotpotQA, 2WikiMultihopQA, MuSiQue 등 세 가지 기존 멀티홉 데이터셋을 활용해 만들어졌습니다.

- **Technical Details**: MoreHopQA는 기존 멀티홉 질문에 추가적인 레이어의 질문을 추가하여 추론 유형을 더욱 다양화했습니다. 이를 통해 상식, 산술 연산, 상징적(reasoning) 등의 다양한 유형의 추론을 포함하였습니다. 데이터셋 생성 과정은 반자동화된 과정으로 이루어졌으며, 1,118개의 샘플이 인간 검증을 거쳤습니다. 다양한 대형 언어 모델(LLM)을 평가하기 위해 Mistral 7B, Gemma 7B, Llama 3(8B, 70B), GPT-4를 사용하였습니다.

- **Performance Highlights**: 실험 결과, 모델들이 초기 멀티홉 질문에서는 잘 수행했으나 확장된 질문에서는 어려움을 겪는 것으로 나타났습니다. 예를 들어, GPT-4의 경우 38.7%, Llama3-70B의 경우 33.4%의 정답률만이 완전한 추론을 달성했습니다. 이는 MoreHopQA가 이전 데이터셋보다 더 도전적인 과제를 제시한다는 것을 의미합니다.



### CoAct: A Global-Local Hierarchy for Autonomous Agent Collaboration (https://arxiv.org/abs/2406.13381)
Comments:
          9 pages, 4 figures

- **What's New**: 새로운 CoAct 프레임워크가 기존의 CoT(CoT)와 ReAct(ReAct) 전략으로도 해결하기 어려운 복잡한 실제 세계의 NLP 작업을 처리하는 엔드투엔드 접근법을 제시합니다.

- **Technical Details**: CoAct는 두 명의 에이전트를 포함하는 계층적 계획 프레임워크입니다: (1) 글로벌 계획 에이전트(Global planning agent)는 문제 범위를 이해하고 매크로 수준의 계획을 세우고, 세부적인 하위 작업 설명을 로컬 실행 에이전트(Local execution agent)에게 제공합니다. (2) 로컬 실행 에이전트는 다단계 작업 구조 내에서 세부 작업의 실행에 집중합니다. 이 프레임워크는 웹기반의 복잡한 작업을 보다 정확하게 이해하고 해결하는 데 도움이 됩니다.

- **Performance Highlights**: WebArena 벤치마크 결과에 따르면 CoAct는 실패 시 프로세스 경로를 재조정할 수 있으며, 장기적인 웹 작업에서 기존의 ReAct보다 높은 성능을 보입니다. CoAct는 평균 성공률이 13.8%로 ReAct의 9.4%를 뛰어넘었으며, 강제 중단 개입(Force stop intervention)이 포함된 CoAct는 16.0%의 성공률을 기록했습니다.



### ALiiCE: Evaluating Positional Fine-grained Citation Generation (https://arxiv.org/abs/2406.13375)
- **What's New**: 최근의 연구는 대형 언어 모델(LLMs)의 신뢰성과 검증 가능성을 높이기 위해 인용이 포함된 텍스트 생성을 제안하고 있습니다. 그러나 기존의 작업 및 평가 방식은 주로 문장 수준의 진술에 초점을 맞추어 미세한 위치별 인용의 중요성을 간과하고 있습니다. 이를 해결하기 위해 우리는 ALiiCE라는 최초의 자동 평가 프레임워크를 제안합니다. 이 프레임워크는 의존성 분석을 통해 문장 주장을 원자적 주장으로 분해하고, 원자적 주장 수준에서 인용 품질을 평가합니다. ALiiCE는 위치별 미세 인용 품질 평가를 위한 세 가지 혁신적인 지표를 도입합니다.

- **Technical Details**: ALiiCE는 의존성 트리(Dependency Tree) 기반 접근법을 통해 응답 속 각 인용에 대응하는 원자적 주장을 분석합니다. 예를 들어, 'Cups can be made of glass'와 'Cups can be made of plastic'과 같은 문장을 원자적 주장으로 분리합니다. 또한, 인용 품질을 평가하기 위해 위치별 세밀한 인용 회수(Recall)와 정밀도(Precision), 인용 위치의 변동 계수(Coefficient of Variation)라는 세 가지 새로운 지표를 도입합니다.

- **Performance Highlights**: 우리는 기존 모델의 위치별 세밀 인용 생성 성능을 평가하기 위해 ASQA와 ELI5라는 두 개의 장문형 QA 데이터셋을 사용합니다. 실험 결과와 사례 분석을 통해 ALiiCE가 위치별 세밀 인용 생성 평가에 효과적이고 합리적이라는 것을 증명했습니다. 또한, 기존의 LLM들이 위치별 세밀 인용 생성을 수행하는 데 어려움을 겪고 있는 것으로 나타났습니다. 특히 최신 공개형 LLM들이 폐쇄형 LLM들과의 격차를 좁히고 있다는 점도 관찰되었습니다.



### Evaluating Structural Generalization in Neural Machine Translation (https://arxiv.org/abs/2406.13363)
Comments:
          To appear at ACL 2024 findings

- **What's New**: 새로운 연구는 기계번역에서 구조적 일반화(structural generalization)에 관련된 문제를 다룹니다. 이를 위해 다양한 구조적 일반화를 포함하는 SGET 데이터셋을 개발하여 기존 모델을 평가했습니다.

- **Technical Details**: 연구팀은 규칙 기반 방법(rule-based method)과 확률적 문맥 자유 문법(PCFG)을 사용해 영어-일본어 번역을 위한 병렬 데이터셋을 생성했습니다. SGET는 트레이닝, 개발, 테스트, 일반화 세트로 구성되어 있으며, 특정 조합의 단어와 문법 구조가 포함됩니다.

- **Performance Highlights**: LSTM, Transformer, Llama 2와 같은 신경망 모델을 SGET 데이터셋에서 평가한 결과, 모델들은 구조적 일반화에서 더 많은 어려움을 겪는 것으로 나타났습니다. 기존의 의미 분석과 기계 번역 간의 성능 경향이 다르게 나타나, 다양한 작업에서의 평가 필요성을 강조했습니다.



### Improving Zero-Shot Cross-Lingual Transfer via Progressive Code-Switching (https://arxiv.org/abs/2406.13361)
Comments:
          9 pages, 5 figures, 6 tables. Accepted by International Joint Conference on Artificial Intelligence (IJCAI 2024)

- **What's New**: 새로운 종이에서는 진화적 코드-전환(PCS: Progressive Code-Switching) 방식을 제안하여 점진적으로 난이도가 있는 코드-전환 예제를 생성합니다. 이는 모델이 쉬운 예제에서부터 어려운 예제까지 점진적으로 성능을 향상시킬 수 있도록 합니다.

- **Technical Details**: PCS 방법론의 핵심 구성 요소는 세 가지로 나뉘어집니다: (1) 워드 관련성 점수를 기반으로 문장에서 각 단어의 영향을 측정하는 난이도 측정기(difficulty measurer), (2) 조절 가능한 온도 변수로 난이도가 점진적으로 증가하는 코드-전환 데이터를 생성하는 코드-전환기(code-switcher), (3) 언제 더 어려운 코드-전환 데이터를 샘플링할지 결정하는 학습 스케줄러(training scheduler).

- **Performance Highlights**: PCS 접근법은 세 가지 다양한 zero-shot 크로스-리니얼(크로스-언어) 전이 작업에서 10개 언어에 걸쳐 최첨단 결과를 달성하였습니다. 이 방식은 기존의 무작위 코드-전환 기법에 비해 모델의 다중 언어 표현 정렬을 크게 향상시켰습니다.



### Transferable speech-to-text large language model alignment modu (https://arxiv.org/abs/2406.13357)
Comments:
          Accepted by InterSpeech 2024; 5 pages, 2 figures

- **What's New**: 이번 연구에서는 Whisper 인코더와 Yi-6B 사전 학습 모델의 잠재력을 활용하여 단일 레이어 모듈 및 수백 시간의 음성-텍스트 멀티태스크 코퍼스만으로 모달 정렬을 달성할 수 있음을 발견했습니다. 또한, 인간의 선호도에 맞춘 Yi-6B-Chat으로 바꿔서도 유사한 정렬 성능이 가능함을 확인했습니다. 이러한 모듈 정렬 기법은 다른 기능(음성 프린트, 비디오 등)을 추가하여 모달리티를 확장할 가능성을 보여줍니다.

- **Technical Details**: 이 연구에서는 Whisper 인코더의 출력을 텍스트 피처 공간으로 매핑하는 단일 레이어 선형 모듈을 추가하고, LLM(Yi-6B)을 얼리고, alignment module(정렬 모듈)을 훈련합니다. Whisper 인코더는 1280 차원의 출력을 생성하고, Yi-6B는 4096 차원의 피처를 가집니다. 데이터 포맷은 <|Human|>, <|startofaudio|>, <|endofaudio|> 등의 특별 토큰을 사용하여 모델에게 명령을 전달합니다.

- **Performance Highlights**: 이 정렬 모듈은 적은 데이터로도 모달 정렬을 자극할 수 있으며, 인간의 선호도에 맞춘 SFT 모델로 쉽게 교체 가능해 특정 작업(ST, SQA 등)에서 성능을 향상시킬 수 있습니다. 또한, SVD 분석을 통해 정보 과잉을 밝혀내었으며, 피처 차원 감소가 성능에 큰 영향을 미치지 않음을 확인했습니다.



### ZeroDL: Zero-shot Distribution Learning for Text Clustering via Large Language Models (https://arxiv.org/abs/2406.13342)
Comments:
          ARR Submitted

- **What's New**: 최근의 큰 언어 모델(LLMs)의 발전은 NLP 작업 해결에 큰 진전을 가져왔습니다. 특히 인-컨텍스트 학습(ICL)이 특정 작업의 이해와 미세한 뉘앙스를 파악하는 핵심 메커니즘으로 작용합니다. 본 논문에서는 목표 데이터셋을 설명하는 LLM의 과정을 관찰하고, 이를 통해 생성된 메타 정보를 실제 작업에 통합함으로써 작업을 특정 LLM에 맞게 맥락을 형성하는 간단하면서도 효과적인 방법을 제안합니다.

- **Technical Details**: 제안된 방법인 Zero-shot Distribution Learning(ZeroDL)은 두 가지 주요 구성요소로 구성됩니다: (1) 오픈 엔드 제로샷 추론(open-ended zero-shot inference) 및 (2) 출력 집계(output aggregation). 이 방법은 텍스트 클러스터링 작업에서의 효과성을 보여주며, 모델이 주어진 작업을 성공적으로 수행할 수 있도록 자체 생성된 프레임을 활용합니다. 첫 번째 단계는 제로샷 분류를 위한 프롬프트를 디자인하는 것입니다. 다음으로 생성된 예측을 집계하여 혼란스러움을 줄이고 일관된 출력 형식을 얻습니다. 마지막으로, 이 집계된 메타 정보를 실제 작업에 통합하여 모델의 예측을 리드합니다.

- **Performance Highlights**: 제안된 ZeroDL 방식은 여러 데이터셋에서 임베딩 기반 클러스터링 방법에 비해 경쟁력을 갖추고 있으며, 일부 경우에는 실제 클래스 라벨을 갖춘 모델보다 더 나은 성능을 발휘합니다. 특히 텍스트 클러스터링 작업에서 특정 컨텍스트를 처리할 수 있도록 도와 임베딩 기반 클러스터링 방법에 비해 이점을 제공합니다.



### SD-Eval: A Benchmark Dataset for Spoken Dialogue Understanding Beyond Words (https://arxiv.org/abs/2406.13340)
- **What's New**: 새로운 연구 논문에서는 음성 기반 대화 이해와 생성을 다차원적으로 평가하는 벤치마크 데이터셋인 SD-Eval을 소개합니다. SD-Eval은 감정(emotion), 억양(accent), 나이(age), 배경 소리(background sound) 등 음성 대화의 다양한 측면을 고려합니다. 이 데이터셋은 총 7,303개의 발화와 8.76시간의 음성 데이터를 포함하며, 이는 8개의 공개 데이터셋에서 수집된 것입니다.

- **Technical Details**: 이 연구에서는 SD-Eval 벤치마크 데이터셋의 평가를 위해 세 가지 다른 모델을 구현하고, SD-Eval과 유사한 과정으로 훈련 세트를 구성했습니다. 훈련 세트는 1,052.72시간의 음성 데이터와 724.4천 개의 발화를 포함합니다. 평가 방법으로는 BLEU와 ROUGE와 같은 객관적 평가 방법뿐만 아니라, 주관적 평가와 대화 언어모델(LLM) 기반의 메트릭스를 사용했습니다.

- **Performance Highlights**: 관찰된 결과에 따르면, 비언어적(paralinguistic) 및 환경적 정보(environmental information)가 포함된 모델은 객관적 및 주관적 평가 모두에서 더 우수한 성능을 보였습니다. 또한, LLM 기반의 메트릭스는 전통적인 메트릭스보다 인간 평가와 더 높은 상관관계를 보였습니다.



### How effective is Multi-source pivoting for Translation of Low Resource Indian Languages? (https://arxiv.org/abs/2406.13332)
- **What's New**: 이 논문은 언어적으로 다른 언어 사이의 기계 번역(MT)에서 자주 사용되는 '피봇 방법론(pivoting)'을 여러 출처 구문을 동시에 사용하는 '멀티 소스 번역(multi-source translation)' 접근법과 결합하여 성능을 향상시키려는 시도를 소개합니다. 특히, 영어에서 인도 언어로의 번역에서 기존의 피봇 언어 기법이 아닌 소스 구문을 함께 사용하는 방법을 탐구했습니다.

- **Technical Details**: 이 논문에서는 영어를 Konkani, Manipuri, Sanskrit, Bodo와 같은 인도 언어로 번역하기 위해 Hindi, Marathi 및 Bengali를 피봇 언어로 사용하여 다양한 멀티 소스 기술을 테스트했습니다. 멀티 소스 번역을 통해 피봇 언어와 소스 구문 모두를 사용하여 번역 성능을 향상시키려 시도했습니다. 다양한 정규화 전략과 주의(attention) 메커니즘을 적용하여 소스와 피봇 표현을 균형 있게 반영하려 했습니다.

- **Performance Highlights**: 실험 결과 멀티 소스 피봇 번역은 최첨단 기법에 비해 미미한 성능 향상을 보였으나, 인공(target) 언어 데이터를 활용하면 성능이 크게 개선될 수 있음을 확인했습니다. 이로 인해 멀티 소스 피봇 접근법이 저자원 번역에 유망한 방향임을 시사했습니다.



### Improving Zero-shot LLM Re-Ranker with Risk Minimization (https://arxiv.org/abs/2406.13331)
Comments:
          Under review

- **What's New**: 이번 연구에서는 고급 거대 언어 모델(LLMs)을 탐색-증강 생성(RAG) 시스템 내 비지도 학습 방식의 질의 가능성 모델(QLM)로 사용하는 새로운 프레임워크 $m{UR^3}$을 제시합니다. 이 프레임워크는 베이지안 의사결정 이론(Bayesian decision theory)을 활용해 추정 편향을 정량화하고 완화하는 방법을 제공합니다.

- **Technical Details**: $m{UR^3}$는 문서 생성 확률을 최대화하는 문제로 문제를 재구성하여, 질의와 문서 생성 확률을 통합된 위험 최소화 목표 하에 최적화합니다. Kullback-Leibler(KL) 발산을 사용하여 추정 편향을 최소화하고, 문서 생성 확률의 최대화를 통해 문서 선택 과정에서의 최적화를 이룹니다.

- **Performance Highlights**: $m{UR^3}$는 오픈 도메인 질의응답(QA) 작업에서 재순위 알고리즘의 Top-1 정확도를 크게 향상시키며, NQ, WebQ, TriviaQA 데이터셋에서 각각 UPR 대비 6.64%, 6.35%, 3.18% 포인트 향상된 결과를 보였습니다. QA 작업에서 EM과 F1 점수도 각각 최대 1.48과 2.06까지 증가하였습니다.



### Understanding the RoPE Extensions of Long-Context LLMs: An Attention Perspectiv (https://arxiv.org/abs/2406.13282)
- **What's New**: 최근 연구의 핫이슈인 LLMs(대형 언어 모델)의 긴 맥락 처리 능력 향상에 대한 연구입니다. 대부분의 LLMs는 RoPE(로터리 포지션 임베딩)에 의존하는데, 이는 비교적 짧은 텍스트에 대해 훈련된 RoPE를 더 긴 텍스트로 외삽(Extrapolate)하는 방향성을 가지고 있습니다. 본 연구는 이러한 RoPE 확장이 작동하는 내부 메커니즘을 심층 분석하고자 합니다.

- **Technical Details**: RoPE 확장은 기존의 RoPE가 가지는 한계를 극복하고, LLMs가 더 긴 텍스트를 효과적으로 다루도록 만듭니다. 연구에서는 세 가지 주요 RoPE 확장 방법론인 포지션 인터폴레이션(Position Interpolation), YaRN, NTK-Aware 인터폴레이션을 다루고 있습니다. 이러한 확장 방법들이 주목받고 있지만, 그 내부 메커니즘에 대한 체계적인 이해는 sparsely 존재합니다.

- **Performance Highlights**: ['프리트레인된 길이에서 주어진 RoPE 확장 방법들을 사용하는 것은 LLMs의 외삽 성능을 크게 향상시킵니다.', 'NTK-Aware 인터폴레이션을 통해 프리트레인된 길이의 최대 32배 이상을 외삽할 수 있습니다.', '주의 패턴(Attention Pattern)을 유지하는 것이 외삽 성능을 향상시키는 데 중요한 역할을 합니다.', '긴 텍스트를 다루기 위해 훈련 중 더 긴 텍스트를 사용하는 것이 주의 불확실성을 줄이고 외삽 성능을 향상시킵니다.']



### In-Context Learning on a Budget: A Case Study in Named Entity Recognition (https://arxiv.org/abs/2406.13274)
- **What's New**: 소수 샷(in-context learning, ICL) 학습에서 예산 내에서 샘플 선택을 최적화하는 방법을 제안합니다. 특히, 실제 세계 도메인 적응 시나리오에서 제한된 예산으로 주석된 샘플을 선택하여 후속 성능을 최대화하는 방법을 연구했습니다. 주석 예산 내에서 샘플을 선택하는 여러 방법을 평가했으며, 랜덤 샘플 선택이 기대 이상의 성능을 보인다는 것을 발견했습니다.

- **Technical Details**: ICL 학습은 소수의 주석된 샘플을 모델에 제공하여 학습하는 방식으로, 특히 명명된 엔티티 인식(NER) 작업에 집중했습니다. 다양한 모델과 데이터셋을 적용하여 주석 예산 내에서 샘플 선택을 최적화하는 전략을 평가했습니다. 주요 방법으로는 클러스터링, 최근접 이웃 선택, 랜덤 선택 등이 포함됩니다. 샘플 선택 전략은 훈련 셋과 테스트 셋의 분포를 커버하는 것을 목표로 설계되었습니다.

- **Performance Highlights**: 200개의 주석된 샘플 풀로도 전체 훈련 세트를 사용할 때와 비교하여 약 88%의 성능을 달성할 수 있었습니다. 또한, 다양한 모델과 데이터셋에서 더 다양한 샘플 풀은 성능 향상과 상관관계가 있다는 것을 발견했습니다. 랜덤 샘플 선택이 일부 시나리오에서 신중하게 설계된 방법에 필적하는 성능을 보였습니다.



### BeHonest: Benchmarking Honesty of Large Language Models (https://arxiv.org/abs/2406.13261)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 정직성을 평가하기 위한 최초의 포괄적인 벤치마크인 BeHonest를 소개합니다. 기존 연구들은 LLM의 유용성 및 무해성에 중점을 두었으나, 정직성 평가에 대한 연구는 상대적으로 적었습니다. BeHonest는 LLM이 지식 경계를 인식 하는 정도, 사기를 피하는 능력, 응답의 일관성을 포함한 세 가지 핵심 측면을 평가합니다.

- **Technical Details**: BeHonest는 LLM의 정직성을 평가하기 위해 설계된 10개의 시나리오로 구성되며, GPT-4, ChatGPT, Llama2, Llama3, Mistral, Qwen과 같은 시장에 있는 9개의 주요 LLM을 분석합니다. 이 평가 프레임워크는 LLM의 현재 능력과 한계를 보여주며, 향후 윤리적이고 투명한 AI 시스템 개발을 위한 기초를 마련합니다. 주요 평가 측면은 다음과 같습니다:
1. **Self-Knowledge**: 모델이 자신의 능력과 한계를 투명하게 전달하는지 여부.
2. **Non-Deceptiveness**: 모델이 속이지 않고 내부 인식을 충실히 반영하는지 여부.
3. **Consistency**: 모델이 일관성을 유지하는지 여부.

- **Performance Highlights**: 평가 결과, 대부분의 LLM이 지식을 표현할 수 있지만, 확실하지 않은 경우 답변을 회피하는 경우는 드뭅니다. 또한, LLM은 사람을 기쁘게 하거나 과제를 완료하기 위해 기꺼이 속임수를 사용하며, 프롬프트의 사소한 변경이나 편향에도 응답의 일관성이 부족한 경향이 있습니다. 이러한 결과는 LLM의 정직성 향상이 여전히 많은 여지가 있음을 시사합니다.



### R^2AG: Incorporating Retrieval Information into Retrieval Augmented Generation (https://arxiv.org/abs/2406.13249)
- **What's New**: 이 논문은 RAG(Retrieval Augmented Generation) 프레임워크의 성능을 향상시키기 위해 R^2AG라는 새로운 접근법을 제시합니다. 기존 RAG에서는 대형 언어 모델(LLMs)과 정보 검색 모델(retrievers) 사이의 의미적 간극이 존재했습니다. R^2AG는 이 간극을 메우기 위해 검색 정보를 통합한 RAG를 설계하였습니다. 특히, R$^2$-Former라 불리는 경량 모델을 통해 검색 정보를 캡처하여 LLMs의 문자 생성 과정에 통합하는 전략을 채택했습니다.

- **Technical Details**: R^2AG는 검색 결과를 단순히 텍스트로 결합하는 대신, retriever의 의미적 표현을 통합합니다. R$^2$-Former는 retriever와 LLM 사이에 삽입되는 모델로, 핵심 검색 정보를 포착하고 retrieval-aware prompting strategy를 통해 LLMs에 추가적인 임베딩을 제공합니다. 이러한 과정은 retrieval 문서를 처리할 때의 정보 손실을 방지하고, LLMs의 정확한 응답 생성을 지원합니다. 또한, R^2AG는 retrievers와 LLMs가 고정된 상태에서 작동할 수 있어, 리소스가 제한된 시나리오에서도 유용합니다.

- **Performance Highlights**: 다섯 개의 데이터셋에서 광범위한 실험을 통해 R^2AG의 효과, 강인성, 효율성을 검증했습니다. 분석 결과, R^2AG는 추론 중 지연 시간을 단 0.8% 증가시키면서도 LLMs가 검색된 문서를 이해하고 더 정확한 응답을 생성하는 데 도움을 줄 수 있음을 보여주었습니다.



### GSR-BENCH: A Benchmark for Grounded Spatial Reasoning Evaluation via Multimodal LLMs (https://arxiv.org/abs/2406.13246)
- **What's New**: 이 연구는 이미지 내 객체 간의 공간 관계를 이해하기 위해 What'sUp 데이터셋을 확장하고, 27개의 다양한 모델들에 대한 포괄적인 평가를 제안합니다. 이 연구는 다양한 Multimodal LLMs(MLLMs)를 평가하여, 이들 모델이 공간 관계 이해에서 어떻게 성능을 보이는지를 분석합니다.

- **Technical Details**: 기존의 VLMs(Vision and Language Models)뿐만 아니라, 3가지 클래스의 MLLMs(매개변수 크기: 7B~110B, 학습 방법, 시각적 해상도)를 포함하여 다양한 모델을 평가하였습니다. What’sUp 데이터셋에는 깊이 정보와 바운딩 박스 주석이 추가되었으며, GroundingDINO, Segment Anything (SAM), ZoeDepth와 같은 도구를 사용하여 객체 인식과 위치 파악을 수행했습니다. 또한, 구조화된 프롬프트를 사용하여 객체 및 공간 관계 평가를 위한 다양한 프롬프트 전략을 설계했습니다.

- **Performance Highlights**: 연구 결과에 따르면, 보다 큰 모델(예: LLaVA-NeXT-Yi-34B, LLaMA-3-LLaVA-NeXT-8B)이 작은 모델에 비해 프롬프트에 대한 위치 변경 민감도가 적고 더욱 높은 강건성을 보여주었습니다. 서브셋 B는 '뒤에' 및 '앞에' 같은 공간 절을 이해하는데 어려움을 겪는 경우가 많아 성능이 가장 낮았습니다. 깊이 정보를 프롬프트에 힌트로 활용하여 이러한 성능을 향상할 수 있음을 확인했습니다.



### Data Contamination Can Cross Language Barriers (https://arxiv.org/abs/2406.13236)
Comments:
          12 pages, 5 figures

- **What's New**: 이번 논문은 기존의 검출 방법을 회피하고 대형 언어 모델(LLM)의 성능을 인위적으로 증가시키는 'Cross-lingual' 오염(Cross-lingual contamination) 형상을 다룹니다. 이 방법은 벤치마크 테스트 세트의 번역 버전에 과적합하여 LLM이 성능을 부풀리는 방식을 사용합니다.

- **Technical Details**: 기존의 오염 검출 방법은 훈련 데이터와 평가 데이터 간의 텍스트 중복에 기반을 두고 있습니다. 그러나 이는 표면적인 오염만을 검출하는 데 한계가 있습니다. 본 연구에서는 다국어 모델인 LLaMA3-8B와 Qwen1.5-7b를 사용하여 세 가지 인기 벤치마크(MMLU, ARC Challenge, MathQA)의 번역 버전을 통해 오염을 주입하고, 이를 검출하기 위한 일반화 기반 접근법(generalization-based approaches)을 제안합니다.

- **Performance Highlights**: 실험 결과, Cross-lingual 오염은 기존 검출 방법을 쉽게 속일 수 있으나, 제안된 일반화 기반 검출 방법으로는 효과적으로 검출이 가능했습니다. 이 방법은 원래의 벤치마크를 수정하여 모든 오답 선택지를 다른 질문에서 가져온 정답으로 대체하는 방식을 사용하여 모델의 성능 변화를 평가합니다. 오염된 모델은 이러한 조작된 상황에서 잘 일반화하지 못할 것입니다.

- **Potential Impact**: Cross-lingual 오염의 영향력을 분석하고 LLM의 작업 메커니즘을 이해하는 데 도움을 줄 수 있으며, 다국어 성능 향상을 위한 후속 학습에 활용할 수 있는 가능성도 논의되었습니다.



### Towards Robust Evaluation: A Comprehensive Taxonomy of Datasets and Metrics for Open Domain Question Answering in the Era of Large Language Models (https://arxiv.org/abs/2406.13232)
Comments:
          22 pages, 13 tables, 7 figures

- **What's New**: 본 연구는 열린 도메인 질문 응답(Open Domain Question Answering, ODQA)에 관한 현재의 벤치마킹 상황을 종합적으로 검토합니다. 이를 위해 텍스트 및 멀티모달 데이터셋 52개와 평가 기술 20개를 분석하였습니다. 또한 ODQA 데이터셋에 대해 다중모달성(multimodalities)과 질문 유형의 난이도를 포함하는 새로운 분류법(taxonomy)을 도입했습니다.

- **Technical Details**: ODQA 시스템은 대규모 지식 말뭉치(knowledge corpora)를 사용하여 사실적 질문에 답하는 시스템을 구축하는 과정을 포함합니다. 최근의 발전은 대규모 훈련 데이터셋, 딥러닝 기술, 대형 언어 모델(large language models)의 부상 덕분입니다. 표준화된 메트릭(metrics)은 다양한 ODQA 시스템의 비교를 용이하게 하여 연구자들이 해당 분야의 발전을 객관적으로 추적할 수 있도록 합니다. 본 연구는 ODQA의 평가 메트릭과 그 고유한 트레이드오프(trade-offs)에 대한 구조화된 조직과 비판적 분석을 제시합니다.

- **Performance Highlights**: 본 연구는 현대 ODQA 시스템의 강력한 평가를 위한 프레임워크를 제공함으로써 연구자들에게 힘을 실어줍니다. 현재의 ODQA 시스템이 직면한 주요 도전 과제를 식별하고, 향후 연구 및 개발을 위한 유망한 경로를 제시하고 있습니다.



### Enhancing Language Model Factuality via Activation-Based Confidence Calibration and Guided Decoding (https://arxiv.org/abs/2406.13230)
- **What's New**: 최근 연구에 따르면 언어 모델(LM)의 확률 보정을 통해 모델의 신뢰성을 개선하고 환각된 내용을 줄일 수 있습니다. 이 연구에서는 LM의 마지막 레이어 활성화를 이용한 ActCab라는 보정 기법과, 이를 기반으로 높은 신뢰도의 답변을 도출하는 Confidence-guided Decoding(CoDec) 전략을 제안합니다.

- **Technical Details**: ActCab는 LM의 내부 활성화(activation)를 이용해 모델의 불확실성을 추정합니다. CoDec는 응답 후보 토큰의 상위-K 확률을 고려하여 정답 가능성을 높이는 방식으로 동작합니다. 이 접근법은 LM의 원래 추론 과정을 변경하지 않아 일관된 성능을 제공합니다.

- **Performance Highlights**: ActCab는 Llama2-7b 모델에서 5개의 QA 벤치마크를 통해 평가되었으며, 가장 경쟁력 있는 베이스라인 대비 ECE 점수를 평균 39% 감소시켰습니다. 또한 CoDec는 TruthfulQA와 같은 어려운 QA 데이터셋에서 뛰어난 사실성을 보이며 좋은 성능을 나타냈습니다.



### Probing the Emergence of Cross-lingual Alignment during LLM Training (https://arxiv.org/abs/2406.13229)
Comments:
          Accepted to Findings of the Association for Computational Linguistics: ACL 2024

- **What's New**: 이번 논문은 다국어 대형언어모델(multilingual Large Language Models, LLMs)이 병렬 문장(parallel sentences)의 명시적 관리 없이 언어를 정렬할 수 있는 능력이 교차 언어적 전이(zero-shot cross-lingual transfer) 성능에 어떻게 기여하는지에 대해 탐구합니다. 특히, 모델이 알고리즘의 다양한 학습 단계에서 신경망(neuron)의 특정 부분 집합이 언어적 특징을 인코딩 하는지를 조사해 그 신경망 중첩도가 교차 언어적 전이 성능과의 상관 관계를 보여줍니다.

- **Technical Details**: 논문은 BLOOM이라는 다국어 자가회귀 언어 모델(multilingual autoregressive LLM)의 여러 학습 단계와 모델 크기에서 체크포인트를 활용하여 모델의 학습 도중 교차 언어 정렬이 어떻게 발생하는지를 조사합니다. 특히, 내재적 탐침 기법(intrinsic probing techniques)을 사용하여 어떤 신경망 부분들이 언어적 특징을 인코딩하는지를 확인하고 이것이 모델의 성능에 미치는 영향을 분석합니다. 또한, Torroba Hennigen et al.(2020)에서 제안된 잠재 변수 모델(latent variable model)을 적용해 구체적으로 신경망 내 특정 차원이 특정 언어적 특징을 인코딩하는지를 파악합니다.

- **Performance Highlights**: 분석 결과에 따르면, 신경망 중첩도와 다운스트림 성능 간에 높은 상관 관계가 있으며, 이는 효과적인 교차 언어 전이를 위한 조건을 지지해준다고 봅니다. 또한, 학습 과정 중간 또는 끝에서 암묵적 정렬과 다중 언어 능력이 저하되는 현상도 관찰되어 다국어 사전 학습의 동역학에 대한 새로운 통찰을 제공합니다.



### Bridging Law and Data: Augmenting Reasoning via a Semi-Structured Dataset with IRAC methodology (https://arxiv.org/abs/2406.13217)
- **What's New**: 이 논문에서는 법적 시나리오 분석을 위한 새로운 벤치마크인 LEGALSEMI를 소개합니다. LEGALSEMI는 54개의 법적 시나리오로 구성되며, 각 시나리오는 법률 전문가들에 의해 철저히 주석이 달린 IRAC (Issue, Rule, Application, Conclusion) 프레임워크를 기반으로 합니다. 추가적으로, LEGALSEMI는 구조화된 지식 그래프(SKG)와 함께 제공됩니다. 이 데이터셋은 복잡한 법적 추론 작업에 필요한 고품질 데이터를 제공하여, 법적 추론에 대한 대규모 언어 모델(LLM)의 한계를 극복하는 것을 돕습니다.

- **Technical Details**: LEGALSEMI는 말레이시아 계약법을 다루며, 법률 대학원생들에 의해 주석이 달린 복잡한 IRAC 분석을 포함합니다. 구조화된 지식 그래프(SKG)는 법적 개념, 법원 사례, 법률 규칙, 그 규칙의 해석, 평이한 언어로 된 법적 개념 등의 정보를 포함한 노드와 이들 간의 관계를 나타내는 엣지들로 구성됩니다. SKG는 자동으로 법 교과서와 법률 문서로부터 구축되며, 법적 개념들을 계층적으로 구조화하여 법률 규칙과의 관계를 시각화합니다.

- **Performance Highlights**: 구조화된 지식 그래프(SKG)를 활용한 실험 결과, LLM의 문제 식별 품질이 21.4% 향상되었으며 규칙 검색에서의 상위 5개 결과에서 리콜(Recall)이 60%, F1 점수가 12% 향상되었습니다. 이는 법적 개념이 시나리오의 사실과 법률의 규칙 간의 의미적 격차를 연결하는 데 중요함을 시사합니다. 또한 평이한 언어로 된 해석이 법적 언어와 일상 언어 간의 격차를 줄이는 데 도움이 됨을 보였습니다.



### Multi-Meta-RAG: Improving RAG for Multi-Hop Queries using Database Filtering with LLM-Extracted Metadata (https://arxiv.org/abs/2406.13213)
Comments:
          Submitted to ICTERI 2024 Posters Track

- **What's New**: 새로운 기법인 Multi-Meta-RAG를 소개합니다. 기존의 Retrieval-augmented Generation (RAG) 시스템이 다중-홉(multi-hop) 질문을 효과적으로 처리하지 못한다는 문제를 해결하기 위해 Database Filtering과 LLM에서 추출한 메타데이터를 사용하여 RAG 시스템의 성능을 크게 향상시켰습니다. 이 방법은 MultiHop-RAG 벤치마크에서 좋은 성과를 나타냈습니다.

- **Technical Details**: Multi-Meta-RAG는 특정 도메인과 형식에서의 질문 세트에 맞춰진 데이터베이스 필터링 접근법을 사용합니다. 예를 들어, 특정 출처의 뉴스 기사와 게시 날짜 정보를 포함한 메타데이터 필터를 사용하여 검색 효율성을 높입니다. 주요 필터링 연산자는 $in으로 설정하였으며, LLM 추론 단계에서 추출한 메타데이터를 사용하여 RAG 시스템의 검색 성능을 높입니다. 벡터 데이터베이스로는 Neo4j의 메타데이터 필터링을 지원하는 LangChain을 사용했습니다.

- **Performance Highlights**: Multi-Meta-RAG를 사용한 실험 결과, 기존 RAG 시스템 대비 다양한 성능 지표에서 크게 향상된 결과를 보였습니다. Embedding 모델인 voyage-02를 사용한 Hits@4 지표가 18% 개선되었고, LLM 중 Google PaLM은 정확도가 0.47에서 0.61로 26% 향상되었습니다. GPT-4의 경우도 0.56에서 0.63으로 12% 증가했습니다.



### Learning Translations via Matrix Completion (https://arxiv.org/abs/2406.13195)
Comments:
          This is a late posting of an old paper as Google Scholar somehow misses indexing the ACL anthology version of the paper

- **What's New**: 연구팀은 Bilingual Lexicon Induction (BLI)을 행렬 완성 문제(matrix completion problem)로 모델링하는 확장 가능한 프레임워크를 제시합니다. 이 방법은 다수의 불완전하거나 잡음이 섞인 양언어 및 단일언어 신호를 활용합니다. 본 모델은 고자원과 저자원 언어 모두에서 최첨단 성능(state-of-the-art)을 달성했습니다.

- **Technical Details**: 본 연구는 행렬 분해(Matrix Factorization, MF)를 사용하여 다양한 번역 신호를 결합하는 방식으로 번역을 학습합니다. 번역은 행렬의 형태로 표현되며, 없는 번역은 행렬 완성을 통해 추론됩니다. Bayesian Personalized Ranking (BPR) 목표를 사용하여 관측된 번역을 효율적으로 확장하고, 추가 번역 등신호를 쉽게 통합할 수 있도록 설계되었습니다. 이 방법은 특히 저자원 언어의 경우, 고자원 언어에서 번역을 전사할 때 유용합니다.

- **Performance Highlights**: 대규모 실험을 통해 영어로의 번역 학습을 수행한 결과, 본 모델은 현재 최첨단 성능을 능가하는 결과를 얻었습니다. 본 프레임워크는 다양한 양방향 및 단일언어 번역 신호를 통합하며, 각 신호를 개별적으로 향상시키는 것이 전체 시스템의 성능을 향상시키는데 기여합니다. 연구팀은 코드와 데이터셋 및 출력 번역을 공개했습니다.



### Synthetic Context Generation for Question Generation (https://arxiv.org/abs/2406.13188)
- **What's New**: 최근 자동 질문 생성(QG)에 대한 연구가 활발히 진행되고 있는 가운데, 이 논문에서는 QG 모델의 학습을 위해 대형 언어 모델(LLMs)을 활용하여 합성 콘텍스트를 생성하는 방법을 제안했습니다. 이는 기존의 질문-답변 쌍만을 이용해 학습 데이터를 구축하는 접근법으로, 적절한 도메인별 데이터셋을 얻는 것이 어려운 상황을 타개하고자 합니다.

- **Technical Details**: 이 연구는 두 가지 주요 구성요소를 포함합니다. 첫째, LLMs을 이용하여 주어진 질문-답변 쌍으로부터 합성 콘텍스트를 생성합니다. 둘째, 생성된 합성 콘텍스트와 실제 답변을 기반으로 질문 생성 작업을 수행합니다. 합성 콘텍스트 생성을 위해 OpenAI의 GPT-3.5 모델을 사용하고, nucleus sampling과 특정 셋팅(p=1, 온도=0.9)을 적용했습니다. 결과적으로 생성된 콘텍스트와 질문-답변 쌍을 이용해 QG 모델을 학습합니다.

- **Performance Highlights**: 실험 결과, QG 작업에서 콘텍스트는 합성이든 실제든 매우 중요한 역할을 한다는 것을 확인했습니다. 또한, 작은 언어 모델을 미세 조정(Fine-tuning)한 경우, 더 큰 언어 모델을 프롬프트(Prompting)한 것보다 성능이 우수하다는 것을 발견했습니다. 마지막으로, 합성 콘텍스트와 실제 콘텍스트가 유사한 수준의 성능을 보일 수 있음을 입증했습니다. 이 결과는 QG 연구 및 응용에서 합성 콘텍스트의 유효성을 강조하며, 향후 발전의 길을 열어줍니다.



### Learnable In-Context Vector for Visual Question Answering (https://arxiv.org/abs/2406.13185)
- **What's New**: 대규모 언어 모델(Large Language Models, LLMs)이 In-Context Learning(ICL) 능력을 보여주면서, 연구자들은 이러한 기술을 활용하여 LMMs(Large Multimodal Models)로 확장하고 있습니다. 그러나 ICL의 적용에는 ICDs(In-Context Demonstrations)의 증가로 인한 추론 시간 증가와 성능의 감수성이 문제로 남아 있습니다. 특히 LMMs에서는 여러 데이터 타입의 통합과 멀티모달 ICDs의 복잡성으로 인해 이러한 문제가 더욱 심화됩니다. 이 문제를 해결하기 위해, 본 연구는 Learnable In-Context Vector(L-ICV)를 제안하여 VQA(Visual Question Answering)와 같은 복잡한 멀티모달 작업에서 전통적인 ICL 및 비학습 ICV 방법에 비해 성능을 향상시킨다는 것을 실험적으로 증명합니다.

- **Technical Details**: ICL은 몇 가지 유사한 예제(ICDs)를 제공했을 때 언어 모델이 과제를 해결할 수 있게 하는 방법입니다. 기존의 비학습 ICV 방식은 단순한 NLP 작업에서 유용하지만, VQA와 같은 복잡한 멀티모달 작업에서는 효과가 떨어집니다. L-ICV는 시프트 벡터를 사용하여 쿼리의 잠재 상태를 목표 방향으로 이동시키는 방식으로, 최적의 ICV를 학습합니다. 이를 위해 다양한 조합의 32-shot 랜덤 샘플 데모를 사용하여 ICV를 훈련하고, 각 레이어에 고유한 ICV를 할당하여 보다 세밀한 작업 정보를 캡처합니다.

- **Performance Highlights**: L-ICV는 기존의 비학습 ICV와 달리 동일한 성능 조건에서 1/24.97 FLOPs만을 필요로 하며, VQAV2/OKVQA에서 2.36/1.6의 정확도 향상을 보여주었습니다. 또한 LoRA와 비교했을 때, L-ICV는 훨씬 적은 학습 샘플(500 vs. 8000)로 만족스러운 성능을 달성할 수 있었습니다.



### Locating and Extracting Relational Concepts in Large Language Models (https://arxiv.org/abs/2406.13184)
- **What's New**: 지식 표현 구조에서 관계 개념은 다양한 엔티티(entity) 개념들 간의 연관성을 촉진하여 복잡한 세계 지식을 표현하고 이해할 수 있게 합니다. 자연 언어 프롬프트로 관계 개념을 표현함으로써, 사람들은 대형 언어 모델(LLMs)과 쉽게 상호작용하고 원하는 사실 지식을 회상할 수 있습니다. 이 연구는 사실 회상 과정에서 엔티티와 관계 개념을 표현할 수 있는 숨겨진 상태(hidden states)를 인과 중재 분석(causal mediation analysis)을 통해 식별했습니다. 마지막 토큰 위치에서 관계 개념의 인과 효과만을 표현하는 숨겨진 상태를 발견하였으며, 이를 통해 관계 표현(relational representations)을 성공적으로 LLM에서 추출할 수 있음을 가정했습니다.

- **Technical Details**: 사전 학습된 GPT 기반 LLMs는 마치 지식 베이스처럼 동작하며, 대량의 지식을 저장하고 있습니다. 각 사실 지식은 (s, r, o) 형태의 트리플로 표현될 수 있으며, 여기서 s는 주어 엔티티(subject entity), o는 객체 엔티티(object entity), r은 두 엔티티를 연결하는 관계 개념(relational concept)을 나타냅니다. 이 연구에서는 인과 중재 분석을 통해 LLM 내의 숨겨진 상태들을 살펴보았고, 마지막 토큰 위치에서 관계 인과 효과만을 표현하는 얕은(hidden) 상태와 주제와 관계의 인과 효과를 통합적으로 표현하는 깊은 상태를 발견했습니다. 이를 통해 우리는 마지막 위치에서 관계 인과 효과만을 표현하는 숨겨진 상태가 관계 표현으로 간주될 수 있다는 가설을 세웠습니다.

- **Performance Highlights**: 실험 결과, 추출된 관계 표현은 매우 신뢰성이 높으며, 다른 사실 회상 과정에서도 유연하게 이식될 수 있고, 강력한 엔티티 연결자로 활용될 수 있음을 보여주었습니다. 또한, 관계 재작성(relation rewriting)을 통해 제어 가능한 사실 회상(controllable fact recall) 잠재력을 나타냈습니다.



### QRMeM: Unleash the Length Limitation through Question then Reflection Memory Mechanism (https://arxiv.org/abs/2406.13167)
- **What's New**: 이 논문에서는 자연어 처리에서 상당한 진전을 이룬 대형 언어 모델(large language models, LLMs)이 여전히 방대한 텍스트를 처리하는 데 어려움을 겪고 있는 문제를 해결하고자 한다. 이를 위해 Question then Reflection Memory Mechanism(QRMeM)이라는 새로운 전략이 도입되었다. QRMeM은 이중 구조의 메모리 풀을 활용하여 정적 텍스트 콘텐츠와 구조화된 그래프 지침을 통합한다. 이를 통해 유연하고 효율적으로 대용량 텍스트를 처리할 수 있다.

- **Technical Details**: QRMeM의 구성 요소는 이중 구조 메모리 구성(Dual-structure Memory Construction)과 반영 기반 관련 세그먼트 탐색(Reflection-based Relevant Segments Navigation)으로 나누어진다. 메모리 풀은 구조화된 메모리와 장기적인 정적 메모리로 나뉘며, 이를 통해 질문의 요구사항에 동적으로 맞춘다. 또한, 구조화된 메모리는 그래프로 형성되고, 다양한 엔터티(entity)와 관계(relation)가 노드와 엣지로 구성된다. 이러한 구조화된 메모리는 데이터의 다양한 계층에서 정보를 추적하도록 돕는다.

- **Performance Highlights**: 여러 선택형 질문(MCQ)과 다중 문서 질의 응답(Multi-doc QA) 벤치마크를 통해 QRMeM의 성능이 기존 방법보다 향상되었음을 입증하였다. 특히, 엔터티 기반 접근법(Entity Trial), 그래프 확장 검색(Graph Expansion Search), QRMeM의 반복 학습을 통해 관련 분할 정보를 찾는 과정에서 우수한 성능을 보였다.



### Analyzing Diversity in Healthcare LLM Research: A Scientometric Perspectiv (https://arxiv.org/abs/2406.13152)
- **What's New**: 이 논문은 2021년 1월 1일부터 2024년 6월 16일까지의 데이터를 포함한 대형 언어 모델(LLMs)의 의료 분야 연구에 대한 과학계 계량적 분석을 제시합니다. PubMed와 Dimensions의 메타데이터를 분석하여 연구 기여자의 다양한 정도를 평가하였으며, 성별 및 지리적 차이가 크게 나타났습니다.

- **Technical Details**: 본 연구는 PubMed와 Dimensions API를 이용하여 LLMs 관련 연구 논문을 수집하고, 저자 소속, 국가, 및 자금 출처 등의 메타데이터를 추출하여 분석하였습니다. 또한, Genderize.io API를 사용해 저자의 성별을 추정했고, 월드 뱅크의 2024년 소득 분류와 ISO 알파-3 형식을 이용하여 국가를 분류하였습니다. 저널 다양성 지수를 Gini impurity를 기반으로 새롭게 도입하여 과학 출판의 포용성을 측정하였습니다.

- **Performance Highlights**: 연구 결과, 남성 저자와 고소득 국가(HICs)에서의 기여가 압도적으로 많다는 성별 및 지리적 불균형이 드러났습니다. 이를 해결하기 위해 더 포괄적이고 공평한 인공지능 연구를 촉진하기 위한 실질적인 전략을 제안합니다. 이러한 전략을 통해 LLMs가 전 세계적으로 공정하게 사용되도록 보장하려는 목표를 갖고 있습니다.



### DialSim: A Real-Time Simulator for Evaluating Long-Term Dialogue Understanding of Conversational Agents (https://arxiv.org/abs/2406.13144)
- **What's New**: 대화 에이전트의 실제 성능 평가를 위해 DialSim이라는 실시간 대화 시뮬레이터를 도입했습니다. 이 시뮬레이터는 TV 쇼의 캐릭터 역할을 맡아 자발적인 질문에 답변하며, 이전 대화 정보를 기반으로 대답하고, 알려진 정보와 모르는 정보를 구별하는 능력을 평가합니다.

- **Technical Details**: DialSim은 대화 에이전트가 실시간으로 대화에 참여하고, 다자간 긴 대화를 처리하며, 이전 대화 세션을 기억하고 이유를 도출할 수 있는 능력을 평가합니다. 이 시뮬레이터는 Friends, The Big Bang Theory, The Office 등의 TV 쇼 대본을 기반으로 세 가지 대화 환경을 시뮬레이션합니다.

- **Performance Highlights**: DialSim은 시간 제한 내에 답변 정확도를 측정하며, 대화의 랜덤성을 포함하여 에이전트의 예측 불가능한 환경에서의 적응 능력을 테스트합니다. 또한, 캐릭터 이름을 무작위로 변경하여 사전 학습된 지식에 의존하지 않고 대화 역사에 기반한 응답을 평가합니다.



### Large Language Models are Biased Because They Are Large Language Models (https://arxiv.org/abs/2406.13138)
Comments:
          Under review, 15 pages

- **What's New**: 본 논문은 대형 언어 모델(LLM) 내의 근본적인 속성과 편향(bias) 사이의 관계에 대한 심도 있는 토론을 촉발하고자 합니다. 저자들은 현재의 LLM 설계에서 해로운 편향이 필연적 결과로 나타난다는 주장을 펼칩니다. 이는 LLM을기반으로 하는 AI의 근본적인 가정들을 재고하지 않고서는 편향 문제를 적절히 해결할 수 없음을 시사합니다.

- **Technical Details**: 언어 모델은 관찰 가능한 문자열을 포함하는 언어의 확률 모델입니다. 논문에서는 언어 ℒ의 확률 분포 Prℒ(𝐰)를 설명하며, 이러한 분포를 되도록 정확하게 근사하는 것을 목표로 하는 언어 모델 M을 다룹니다. 이러한 근사 정확성은 상대 엔트로피 D(pℒ(𝐰)||pM(𝐰))에 의해 정의되며, 실제 세계에서는 크로스-엔트로피를 통해 측정됩니다.

- **Performance Highlights**: 기술적으로 언어 모델의 '좋음'은 실제 언어의 분포를 얼마나 충실하게 근사하는가에 따라 다릅니다. 이를 위해 모델들은 크로스-엔트로피를 최적화하도록 학습됩니다. 예를 들어, 숨겨진 마르코프 모델(HMM)에서 파생된 공동 분포 Pr⁡(𝐱,𝐰)를 이용하여 관찰 가능한 문자열 𝐰가 생성됩니다.



### PathoLM: Identifying pathogenicity from the DNA sequence through the Genome Foundation Mod (https://arxiv.org/abs/2406.13133)
Comments:
          9 pages, 3 figures

- **What's New**: PathoLM이라는 최첨단 병원체 언어 모델이 소개되었습니다. PathoLM은 새로운 세균 및 바이러스 병원체를 식별하는 데 최적화되어 있으며, 미리 학습된 DNA 모델인 Nucleotide Transformer를 활용하여 최소한의 데이터로도 튜닝이 가능합니다. 이는 병원체 탐지 능력을 크게 향상시키며, 새로운 병원체를 효과적으로 식별할 수 있습니다.

- **Technical Details**: PathoLM은 박테리아 및 바이러스 염기서열에서 병원성(Pathogenicity)을 식별하기 위해 설계된 모델입니다. ESKAPEE 병원체와 같은 항생제 내성 세균의 염기서열을 포함하여 약 30종의 바이러스와 박테리아 데이터셋을 구축했습니다. 기존의 DciPatho 모델보다 성능이 뛰어나며, zero-shot 및 few-shot 시나리오에서도 탁월한 결과를 보여줍니다.

- **Performance Highlights**: PathoLM은 ESKAPEE 종 분류에서 다른 딥러닝 모델보다 우수한 성능을 나타냈습니다. 특히, 기존의 k-mer 기반 방법이 가진 한계를 극복하며 병원체 식별에서 높은 정확성을 입증하였습니다.



### When Parts are Greater Than Sums: Individual LLM Components Can Outperform Full Models (https://arxiv.org/abs/2406.13131)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)에서 in-context learning(ICL)을 연구하고, 개별 attention heads와 MLPs의 기여도를 분석하여 모델의 출력을 분해합니다. LLM의 내적 구조를 이해하고 개별 컴포넌트의 성능을 기반으로 한 'Component Reweighting' 방법을 제안합니다.

- **Technical Details**: ICL에서 LLM의 출력은 각 attention head와 MLP(components)의 기여도로 분해됩니다. 각 component는 독립적으로 성능이 좋거나, 매우 나쁘거나, 특정 라벨에 편향된 형태로 관찰됩니다. 본 연구는 이러한 component의 정확도가 다양한 프롬프트 설정과 데모 집합에서도 일관되게 나타나는지를 분석합니다. 이를 기반으로, 소수의 라벨링 된 예시에서 component 활성화를 선형적으로 재조정하는 'Component Reweighting'을 제안합니다.

- **Performance Highlights**: 제안된 'Component Reweighting' 방법은 24개의 라벨링 된 예시를 사용하여 Llama-2-7B에서 평균 6.0%의 정확도 향상을 이끌어냈습니다. Llama-2-13B와 Mistral-Instruct-7B에서도 각각 2.2%, 5.1%의 성능 향상을 보였습니다. 새로운 방법은 추가적인 매개변수 학습 없이 모델 내부의 개선을 가능하게 했으며, 이는 비슷한 추론 속도를 유지하면서 성능을 효과적으로 높였습니다.



### Learning to Generate Answers with Citations via Factual Consistency Models (https://arxiv.org/abs/2406.13124)
Comments:
          Accepted to ACL 2024. Code release will follow

- **What's New**: 이번 논문에서는 대규모 언어 모델 (LLMs)이 생성한 내용의 신뢰성을 높이기 위해 인용(citations)을 포함시키는 방법을 제안합니다. 특히, 사실 일관성 모델 (Factual Consistency Models, FCMs)을 활용한 약지도 학습(weakly-supervised fine-tuning) 방법을 통해, LLMs가 생성하는 텍스트에 정확한 인용을 제공하는 방법론을 제시합니다.

- **Technical Details**: 제안된 방법론은 인용이 포함된 텍스트를 생성하고, FCM으로 필터링된 인용 데이터를 이용해 LLM을 지도 학습(supervised fine-tuning)하는 두 가지 단계로 구분됩니다. 초점을 맞춘 학습(focused learning)을 통해 학습 과정 중 사실 단위 토큰(factual unit tokens)의 중요성을 강조합니다. 이 과정에서 LLM이 사실 기반 지식(factual knowledge)과 연관된 토큰에 집중하도록 조정됩니다.

- **Performance Highlights**: ALCE 자동 소수 샷 인용 평가 기준(ALCE few-shot citation benchmark)에서, 제안된 방법론은 인용 생성 성능에서 문맥 학습(in-context learning), 기존 지도 학습, 최신 기법들과 비교해 각각 평균 34.1, 15.5, 10.5 점의 F1 높아진 성과를 보였습니다. 또한, 도메인 전이 설정(domain transfer setting)에서도 제안된 방법론은 새로운 데이터셋에도 강력한 인용 생성 능력을 보여주었습니다. 특히, 사실 오류율이 가장 낮아졌습니다.



### Can Long-Context Language Models Subsume Retrieval, RAG, SQL, and More? (https://arxiv.org/abs/2406.13121)
Comments:
          29 pages. Dataset available at this https URL

- **What's New**: LOFT(LOng-Context Frontiers) 벤치마크를 소개합니다. 이 벤치마크는 실전 태스크에서 긴 컨텍스트를 필요로 하는 작업을 통해 LCLMs(Long-Context Language Models)의 성능을 평가하기 위해 설계되었습니다. LOFT는 최대 백만 개의 토큰 컨텍스트를 다루며, 텍스트, 비주얼, 오디오 등 다양한 모달리티를 포함한 35개의 데이터셋을 포함합니다.

- **Technical Details**: LCLMs는 복잡한 파이프라인 의존성을 줄이고, 툴 없이 전체 코퍼스를 네이티브하게 처리할 수 있는 모델입니다. LOFT 벤치마크는 32k, 128k, 1M 세 가지의 컨텍스트 길이 제한을 제공하며, 각 데이터셋에 대해 최대 100개의 테스트 쿼리와 5개의 few-shot 쿼리, 10개의 개발 쿼리를 샘플링합니다. 주요 평가 영역은 Retrieval, Retrieval-Augmented Generation (RAG), SQL 처리를 포함하며, Many-Shot In-Context Learning (ICL) 능력도 평가됩니다.

- **Performance Highlights**: 128k 토큰 수준에서 LOFT는 LCLMs의 성능이 Gecko와 같은 최첨단 텍스트 검색 시스템과 맞먹는다는 것을 보여줍니다. Gemini는 CLIP 등의 멀티모달 검색 모델을 초과합니다. 그러나 SQL과 같은 복잡한 다중 홉 조합 추론 작업에서는 여전히 성능이 부족합니다. 또한, 체인-오브-생각(chain-of-thought) 전략 등 프롬프트 방식에 따라 성능 변동이 크다는 점도 강조되었습니다.



### Multi-Stage Balanced Distillation: Addressing Long-Tail Challenges in Sequence-Level Knowledge Distillation (https://arxiv.org/abs/2406.13114)
Comments:
          preprint

- **What's New**: 이번 연구에서는 Multi-stage Balanced Distillation (BalDistill) 프레임워크를 제안하여, 다양한 롱테일(long-tailed) 데이터셋에 대해 효율적이고 효과적인 모델 지식을 전이하는 방안을 제시합니다. 이 방법은 특히, 시퀀스 수준의 Knowledge Distillation (KD)을 활용하여, 큰 모델(teacher LLM)에서 더 작은 모델(student LLM)로 이유 기반 추론 프로세스(CoT: chain-of-thought)를 전이합니다.

- **Technical Details**: BalDistill 프레임워크는 예산 내에서 균형 잡힌 훈련 데이터를 생성하는 방법론을 채택하여 성능 편향을 완화합니다. 이는 잘 대표되는 도메인에서 주요 예시를 선택하고, 적게 대표되는 도메인에서는 합성 데이터(synthetic data)를 생성함으로써 이루어집니다. 이 프레임워크는 액티브 러닝(active learning) 접근 방식을 통해, 여러 스테이지에 걸쳐 훈련을 점진적으로 개선합니다. 이를 통해 모델은 폭넓은 도메인 커버리지와 견고성을 확보할 수 있게 됩니다.

- **Performance Highlights**: BalDistill은 예산 내에서 롱테일 데이터셋을 처리함으로써, 모델의 효율성과 성능을 극대화했습니다. 다양한 벤치마크 테스트에서, 이 프레임워크를 통해 파생된 학생 모델들은 최첨단 성능(State-of-the-Art, SoTA)을 달성하였습니다. 특히 롱테일 데이터 분포를 가진 도메인에서 일반화 성능이 크게 향상되었습니다.



### Exploring and Benchmarking the Planning Capabilities of Large Language Models (https://arxiv.org/abs/2406.13094)
- **What's New**: 이 논문은 대형 언어 모델(LLM)의 계획 기능을 향상시키기 위한 네 가지 주요 방향을 제시합니다. 첫째, 고전적인 계획 도메인과 자연어 시나리오를 모두 포괄하는 포괄적인 벤치마크를 구축하였습니다. 둘째, 많은 샷 컨텍스트 학습(in-context learning, ICL)을 통해 LLM 계획 성능을 향상시키는 방법을 탐구했습니다. 셋째, 최적의 계획 경로에 대해 LLM을 미세조정(fine-tuning)함으로써 긍정적인 영향을 입증했습니다. 마지막으로, 제안된 방법이 분포 외(out-of-distribution) 시나리오에서 새로운 계획 도전에 얼마나 잘 일반화할 수 있는지를 평가했습니다.

- **Technical Details**: LLM의 계획 능력을 평가하기 위해 PDDL (Planning Domain Definition Language)과 자연어 기반의 벤치마크 세트를 구축했습니다. PDDL은 AI의 계획 문제를 대표하는 표준 언어로, 초기 상태, 목표, 행동 및 그 효과를 설명합니다. 또한, 많은 샷 컨텍스트 학습 및 슈퍼바이즈드 파인튜닝(SFT), 그리고 Monte-Carlo Tree Search(MCTS)와 같은 검색 기반 계획(method)을 사용하여 LLM 성능을 평가했습니다.

- **Performance Highlights**: 1) ICL을 사용하여 모델을 신중하게 지시하면 계획 성능이 크게 향상되었습니다. 2) 최적의 계획으로 미세조정을 하면 훨씬 작은 모델도 거의 완벽한 정확도를 달성할 수 있습니다. 3) 제안된 계획은 유사한 복잡성을 가진 새로운 환경에서도 동일한 정확도로 일반화할 수 있습니다. 4) 쉬운 인스턴스를 가르치는 것이 어려운 인스턴스에서 더 나은 성능을 발휘하는 데 도움이 됩니다. 5) MCTS와 같은 검색 절차를 통합하면 초기 버전 및 소형 모델이 SoTA 모델에 더 가깝게 성능을 발휘할 수 있습니다.



### Multilingual Synopses of Movie Narratives: A Dataset for Story Understanding (https://arxiv.org/abs/2406.13092)
Comments:
          16 pages, 9 figures

- **What's New**: 최근 발표된 논문에 따르면, 컴퓨터 기반 이야기 이해 분야에서 핵심적인 과제인 스토리 비디오-텍스트 정렬을 위한 새로운 다국어 비디오 스토리 데이터셋 'M-SYMON'이 소개되었습니다. 이 데이터셋은 7개 언어로 작성된 13,166개의 영화 요약 비디오를 포함하고 있으며, 총 101.5 시간의 비디오에 대해 세밀한 수동 주석이 포함되어 있습니다.

- **Technical Details**: M-SYMON 데이터셋은 7개 언어로 작성된 영화 요약 비디오를 포함하고 있어 기존의 할리우드 영화 영어 내레이션에 집중된 문제를 해결하고자 합니다. 이 데이터셋에는 영화 요약 비디오와 이에 해당하는 텍스트 사이의 세밀한 맞춤 주석이 포함되어 있으며, 이는 클립 정확도(Clip Accuracy)와 문장 IoU 점수에서 각각 15.7 및 16.2 퍼센트 포인트의 성능 향상을 보여줍니다. M-SYMON 데이터셋은 다양한 다국어 훈련 전략을 사용한 6가지 기본 접근 방식을 테스트하는 벤치마크로도 사용됩니다.

- **Performance Highlights**: M-SYMON 데이터셋을 기반으로 훈련된 모델은 기존 최첨단 방법들에 비해 클립 정확도(Clip Accuracy)와 문장 IoU 점수에서 각각 15.7 및 16.2 퍼센트 포인트 향상된 성능을 보여주었습니다. 또한, 언어 특화 미세 조정을 추가하면 7개 언어 전반에서 평균 5.4 퍼센트 포인트의 성능 향상이 이루어졌습니다. 언어 간 설정에서, 유사한 언어 사이에서는 전이 성능이 우수하였으나, 언어가 크게 다른 경우 전이 성능이 한정되었습니다.



### Evaluating $n$-Gram Novelty of Language Models Using Rusty-DAWG (https://arxiv.org/abs/2406.13069)
Comments:
          8 page preprint + appendix

- **What's New**: 새로운 연구에서는 언어 모델(또는 LM, Language Models)이 생성하는 텍스트가 훈련 데이터와 비교하여 얼마나 새로운지 평가하였습니다. Rusty-DAWG라는 새로운 검색 도구를 개발하여 대규모 데이터 코퍼스에서도 임의 길이의 n-gram을 일정 시간 내에 검색할 수 있게 되었습니다. 이는 게놈 데이터 인덱싱에서 영감을 받은 것으로, 언어 모델의 텍스트 생성 신선도를 측정하고 비교하는 데 사용되었습니다.

- **Technical Details**: Rusty-DAWG는 압축된 유향 비순환 단어 그래프(CDAWG)를 기반으로 임의 길이의 n-gram 매칭이 가능하도록 구현되었습니다. 이 도구는 대규모 데이터를 선형 시간 내에 검색할 수 있어 'n-gram 신선도'라는 지표를 사용해 텍스트 생성을 평가할 수 있는 효율적인 방법을 제공합니다. 이를 통해 Pythia 모델과 같은 대규모 웹 데이터를 훈련시킨 모델에서 얼마나 새로운 n-gram이 생성되는지 분석했습니다.

- **Performance Highlights**: ['LM이 생성하는 텍스트는 n > 4일 때, 인간이 작성한 텍스트보다 덜 신선하지만, n ≤ 4일 때는 더 신선한 것으로 나타났습니다.', '모델의 크기가 커지거나 디코딩 전략이 제한될수록 신선도가 감소하는 경향이 있었습니다. 또한, 훈련 데이터로부터 프롬프트를 줄 경우, 신선도 감소가 관찰되었습니다.', 'LM은 훈련 데이터에서 자주 나타나는 n-gram을 보다 낮은 손실로 완성하는 경향이 있었습니다.']



### Think-then-Act: A Dual-Angle Evaluated Retrieval-Augmented Generation (https://arxiv.org/abs/2406.13050)
Comments:
          12 pages, 8 figures

- **What's New**: 대형 언어 모델(LLMs)의 한계를 보완하기 위해 새로운 프레임워크 'Think-then-Act'가 제안되었습니다. 이 프레임워크는 입력 쿼리의 명확성과 완전성을 평가해 재작성의 필요성을 결정하고, 모델이 질문에 답할 수 있는 능력을 평가하여 추가 정보 검색이 필요한지 여부를 판단합니다.

- **Technical Details**: Think-then-Act 프레임워크는 두 단계로 구성되어 있습니다: (i) 입력 쿼리의 명확성과 완전성을 평가해 필요시 재작성, (ii) 모델의 능력을 평가해 추가 정보 검색 필요 여부 결정. 이 접근법은 비용을 절감하면서 응답의 정확성과 관련성을 유지하거나 향상시킬 수 있습니다.

- **Performance Highlights**: 다섯 개의 데이터셋에서 테스트한 결과, Think-then-Act 프레임워크는 기존의 방법들에 비해 성능이 크게 향상되었습니다. 특히 영어와 비영어 문맥 모두에서 높은 정확성과 효율성을 보여줍니다.



### D2O:Dynamic Discriminative Operations for Efficient Generative Inference of Large Language Models (https://arxiv.org/abs/2406.13035)
Comments:
          Under review

- **What's New**: 이 논문은 긴 시퀀스를 처리하는 대형 언어 모델(LLMs)의 추론 효율성을 개선하기 위한 새로운 방법을 소개합니다. Dynamic Discriminative Operations (D2O) 방법은 KV 캐시 크기를 최적화하면서 중요한 컨텍스트를 유지하고, 기존의 KV 캐시 퇴출 방식의 문제를 해결합니다. 특히 얕은 계층과 깊은 계층 간의 주의 메커니즘 밀도의 차이를 기반으로 퇴출 비율을 다르게 설정하며, 토큰 레벨에서는 폐기된 토큰의 중요성을 재평가하여 필요한 경우 다시 합치는 보상 메커니즘을 사용합니다.

- **Technical Details**: D2O 방법은 두 단계의 차별적 전략을 이용하여 최적화합니다. 첫째, 얕은 계층과 깊은 계층 간의 주의 가중치의 분포 차이를 관찰하여 정보 손실을 최소화하도록 각각의 계층에서 과도한 퇴출을 피합니다. 둘째, 층별 퇴출 전략에 보상 메커니즘을 도입하여, 이전에 폐기된 토큰들이 유사한 토큰과 합쳐질 수 있도록 합니다. 이 보상 메커니즘은 지수 이동 평균(EMA) 임계치를 유지하여 폐기된 토큰의 중요성을 재평가하고 필요한 경우 다시 합칩니다.

- **Performance Highlights**: D2O 방법은 메모리를 최대 3배 줄이고 고품질의 장문의 텍스트 생성을 유지하면서 추론 처리량을 향상시킵니다. 다양한 벤치마크와 LLM 아키텍처에서 광범위한 실험을 통해, D2O가 제한된 KV 캐시 예산 내에서 성능을 크게 향상시키는 것을 확인했습니다. 특히 수학적 및 상식적 추론 작업에서, 장문 컨텍스트 QA, 요약, 코드 완성 작업에서 다른 최첨단 KV 캐시 퇴출 방법들보다 우수한 성능을 보였습니다.



### Detecting Errors through Ensembling Prompts (DEEP): An End-to-End LLM Framework for Detecting Factual Errors (https://arxiv.org/abs/2406.13009)
- **What's New**: 새로운 DEEP(Detecting Errors through Ensembling Prompts) 프레임워크를 소개합니다. 이는 텍스트 요약에서 사실적 오류를 감지하기 위한 종단 간 대형 언어 모델 기반 방법입니다.

- **Technical Details**: 이 프레임워크는 다양한 LLM 프롬프트를 사용하여 사실적 일관성을 식별하고, 그 결과를 이진 특징으로 처리하여 앙상블 모델로 입력합니다. 이후 앙상블 모델의 출력을 보정하여 텍스트가 사실적으로 일관되거나 헛소리 없이 정확한 확률을 도출합니다. LLM 자체의 fine-tuning 없이, 또는 비실용적인 thresholding 기법에 의존하지 않고 사실적 오류를 감지합니다.

- **Performance Highlights**: DEEP 프레임워크는 AggreFact-XSUM FTSOTA, TofuEval Summary-Level, 및 HaluEval Summarization 벤치마크에서 Transformer가 생성한 텍스트 요약의 사실적 오류를 감지하는 데 있어 SOTA(state-of-the-art) 균형 정확도를 달성했습니다.



### Suitability of CCA for Generating Latent State/ Variables in Multi-View Textual Data (https://arxiv.org/abs/2406.12997)
- **What's New**: 이 연구는 Canonical Correlation Analysis (CCA)의 가능성을 활용하여, 텍스트 데이터 내에서 맥락 정보를 캡처하는 잠재 상태를 발견하는 방법을 제안합니다. 특히 CCA를 자동 단문 응답 평가(Automatic Short Answer Grading, ASAG) 작업에 적용하여 경쟁력 있는 결과를 얻었습니다. 이 모델은 간단하고, 선형적이며 적응성이 뛰어나며, 레이블이 없는 훈련 데이터가 부족할 때 특히 유용합니다.

- **Technical Details**: CCA는 두 가지 세트의 변수들 사이의 관계를 탐색하는 다변량 통계 기법으로, 각각의 세트에서 변수들의 선형 결합을 찾아 해당 세트의 캐노니컬 변동량 간의 상관관계를 최대화합니다. 본 연구에서 사용된 데이터는 과거와 미래의 '뷰'를 텍스트 데이터 내에서 두 개의 서로 독립적인 관점으로 간주하며, CCA를 통해 이 뷰들 간의 상관관계를 분석합니다. 특히, 문서 내의 연속적인 문장이나 대화의 턴을 두 개의 뷰로 간주하고 이들의 조건부 독립성을 활용합니다.

- **Performance Highlights**: 제안된 모델은 자동 단문 응답 평가 작업에서 여러 첨단 감독 학습 기법을 능가하는 경쟁력 있는 결과를 달성했습니다. 이는 대규모 레이블이 붙은 데이터 없이도 사용 가능하며, 리소스 집약적인 훈련이 필요하지 않습니다.



### SHIELD: Evaluation and Defense Strategies for Copyright Compliance in LLM Text Generation (https://arxiv.org/abs/2406.12975)
- **What's New**: 대형 언어 모델(LLMs)이 기계 학습에 혁신을 일으켰지만, 저작권 침해 가능성으로 인해 법적 문제를 제기했습니다. 이에 대응하기 위해 저희는 LLM의 저작권 준수 및 공격에 대한 강인성을 평가할 수 있는 데이터셋과 방어 메커니즘을 개발했습니다. 실험 결과, 기존 LLM이 자주 저작권 있는 텍스트를 생성하며, 방어 메커니즘이 이를 상당히 줄일 수 있음을 확인했습니다.

- **Technical Details**: 이번 연구에서는 다음과 같은 주요 도전과제를 다루었습니다: (i) 저작권 준수를 평가하기 위한 종합적인 평가 기준 마련; (ii) 공격 방어 능력 검증; (iii) 저작권 있는 텍스트 생성 방지를 위한 효과적인 방어 개발. 이를 위해 저작권이 있는 텍스트, 비저작권 텍스트, 그리고 국가별로 저작권 상태가 다른 텍스트 등을 포함한 데이터셋을 구축했습니다. 또한, Agent 기반의 방어 메커니즘을 제안하여 웹 검색을 통해 실시간으로 저작권 정보를 확인하고, 관련된 요청을 거부할 수 있도록 했습니다.

- **Performance Highlights**: 실험 결과, 제안된 방어 메커니즘을 적용했을 때 LLM이 저작권 있는 텍스트를 생성하는 빈도가 크게 감소했습니다. 특히, 단순한 프롬프트 공학으로 쉽게 뚫릴 수 있는 기존의 방어 메커니즘에 비해, 제안된 방어 메커니즘이 더 효과적임을 확인했습니다. 코드와 데이터셋은 공개되어 있어, 누구나 연구에 활용할 수 있습니다.



### Prism: A Framework for Decoupling and Assessing the Capabilities of VLMs (https://arxiv.org/abs/2406.14544)
- **What's New**: 이번 아카이브 논문에서는 시각 질문 해결 과정에서 '지각'과 '추론'을 분리하여 평가할 수 있는 새로운 프레임워크인 Prism을 소개합니다. 기존 시각 언어 모델(VLM)들이 '지각'과 '추론'이 결합되어 있는 것과 달리, Prism은 이 두 단계를 명확히 분리하여 분석할 수 있는 구조를 제공합니다.

- **Technical Details**: Prism은 두 개의 주요 모듈로 구성됩니다: 이미지로부터 텍스트 형태로 시각 정보를 추출하는 '지각' 단계와 추출된 정보를 기반으로 답변을 만드는 '추론' 단계입니다. 이 모듈식 디자인은 다양한 VLM과 대형 언어 모델(LLM)을 조합해 각각의 성능을 비교 및 평가할 수 있도록 합니다. 특히, LLaVA 아키텍처를 기반으로 한 2B-parameter VLM를 시각 캡셔너로 훈련시키고, 추론에는 GPT-3.5를 사용했습니다.

- **Performance Highlights**: Prism을 사용한 실험 결과, 2B LLaVA와 GPT-3.5를 결합한 구성이 훨씬 더 큰 VLM과 동등한 성능을 보였으며, MMStar와 같은 까다로운 멀티모달 벤치마크에서 뛰어난 성과를 기록했습니다. 특히, 추론 관련 시각 질문에서는 매우 높은 정확성을 나타냈습니다.



### RL on Incorrect Synthetic Data Scales the Efficiency of LLM Math Reasoning by Eight-Fold (https://arxiv.org/abs/2406.14532)
- **What's New**: 이 논문은 수학적 추론에서 모델 생성을 통한 합성 데이터(finetuning LLMs with synthetic data)에 대한 경험적 연구와 개념적 이해를 제공하고 있습니다. 특히, 모델이 생성한 긍정적인 문제-해결 쌍의 미세 조정을 통해서는 성능 향상이 미미한 반면, 모델 스스로 생성한 데이터를 활용하여 미세 조정하는 방안이 효율성을 두 배로 올릴 수 있음을 발견했습니다. 또한, 모델 생성 긍정 데이터를 이용할 때는 잘못된 상관관계(spurious correlations)이 강화될 수 있다는 문제를 확인했고, 부정 응답(negative responses)을 활용해서 이를 해결할 수 있음을 제시했습니다.

- **Technical Details**: 이 연구는 모델이 생성한 데이터로 미세 조정(finetuning)을 할 때 긍정(positive) 응답뿐만 아니라 부정(negative) 응답도 포함하여 훈련하는 방식을 제안합니다. 부정 응답은 최종 답안 확인기에 의해 잘못되었다고 판단된 응답을 의미합니다. 이 부정 응답은 각 중간 단계의 유용성을 복구하도록 구성되어야 하며, 이러한 단계별 스키마(per-step scheme)를 통해 모델의 성능이 크게 향상될 수 있습니다. 또한 이 방식은 이점-가중 강화 학습(advantage-weighted RL)과 동등하다고 주장하며, RL의 강인성(robustness) 이점을 상속한다고 설명하고 있습니다.

- **Performance Highlights**: 부정 응답을 포함한 훈련을 통해 긍정 데이터에서 발생하는 잘못된 상관관계를 'unlearn' 할 수 있게 되며, 이는 단순히 긍정 데이터만을 흉내내는 것 보다 훨씬 일관된 성능 향상을 가져옵니다. 실제로, 단계별 부정 응답을 통한 훈련은 합성 데이터를 8배로 증폭시키는 것과 유사한 성능 향상을 보였습니다.



### Learning thresholds lead to stable language coexistenc (https://arxiv.org/abs/2406.14522)
Comments:
          5 pages, 3 figures

- **What's New**: 이 논문에서는 Abrams-Strogatz 모델을 기반으로 메모리와 학습의 영향을 언어 이동 역학에 통합한 언어 경쟁 모델을 소개합니다. 이 모델은 언어 간의 이동 역학에 대한 새로운 평형 상태를 설명하며 이는 기존의 Abrams-Strogatz 모델에서 볼 수 없는 특징입니다.

- **Technical Details**: 이 모델은 메모리와 학습의 효과를 화자의 비율에 대한 임계값(threshold)으로 표현합니다. 단순한 형태에서는 이 모델이 정확하게 해석 가능합니다. 모델은 두 언어 중 하나에 대한 합의뿐만 아니라, 두 언어가 안정적으로 공존하는 상태와 초기 상태가 유지되는 '동결된 상태'를 설명합니다.

- **Performance Highlights**: 수치적으로 계산한 결과, 임계 함수의 더 일반적인 형태에서도 이러한 결과가 유지됨을 확인하였습니다. 이는 두 언어 간 이동이 서로 보상할 만큼 임계값이 낮을 때 두 언어가 공존할 수 있으며, 임계값이 너무 높으면 언어 이동이 일어나지 않음을 보여줍니다.



### PostMark: A Robust Blackbox Watermark for Large Language Models (https://arxiv.org/abs/2406.14517)
Comments:
          preprint; 18 pages, 5 figures

- **What’s New**: 연구진은 모델 디코딩 과정 중에 워터마크(watermark)를 삽입하는 대신, 디코딩 후 워터마크를 삽입하는 모듈식 절차인 PostMark를 개발했습니다. 이는 기본 LLM의 로짓(logits)에 접근하지 않아도 되며, 제3자가 구현할 수 있도록 설계되었습니다.

- **Technical Details**: PostMark는 내포된 의미 분석 임베딩을 통해 디코딩 후 텍스트에 입력 의존적인 단어 세트를 삽입합니다. 워터마크 삽입은 먼저 입력 텍스트의 임베딩을 생성한 후, 이 임베딩과 단어 임베딩 테이블(SecTable) 간의 코사인 유사도를 계산하여 워터마크 단어 목록을 형성합니다. 이후 인서터(Inserter) 모델을 통해 텍스트에 매끄럽게 삽입됩니다.

- **Performance Highlights**: 실험 결과 PostMark는 기존의 워터마크 방법들보다 패러프레이징(paraphrasing) 공격에 더 강력한 것으로 나타났습니다. 8가지 baseline 알고리즘, 5가지 기본 LLM, 3가지 데이터셋에서 실험한 결과, PostMark가 우수한 성능을 보였습니다. 또한 인간 평가를 통해도 높은 텍스트 품질과 의미의 변형이 없음을 확인했습니다.



### CodeRAG-Bench: Can Retrieval Augment Code Generation? (https://arxiv.org/abs/2406.14497)
- **What's New**: 최근 연구에서는 코드 생성 모델을 외부 문서와 결합하여 정확성과 기능성을 향상시키는 방법을 탐구했습니다. 이 논문은 다양한 코드 생성 시나리오에서 'retrieval-augmented generation (RAG)'의 잠재력을 체계적으로 평가하기 위해 CodeRAG-Bench라는 포괄적인 평가 벤치마크를 소개합니다.

- **Technical Details**: CodeRAG-Bench는 기본 프로그래밍, 오픈 도메인 문제, 리포지토리 수준 문제 등 세 가지 카테고리를 포괄합니다. 다섯 가지 소스(경쟁 솔루션, 온라인 튜토리얼, 라이브러리 문서, StackOverflow 게시물, GitHub 저장소)에서 문서를 수집해 모델이 참고할 수 있도록 했습니다. 또한, 다양한 설정에서 문서 검색과 코드 생성 성능을 평가했습니다.

- **Performance Highlights**: 고품질의 문서를 검색함으로써 최종 코드 생성의 성능이 눈에 띄게 향상되었습니다. 그러나 현재의 검색 모델은 특히 제한된 어휘적 중첩으로 인해 유용한 문서를 검색하는 데 어려움을 겪고 있으며, 생성 모델은 제한된 문맥 길이나 추가 문맥 통합 능력에 제약을 받고 있습니다.



### African or European Swallow? Benchmarking Large Vision-Language Models for Fine-Grained Object Classification (https://arxiv.org/abs/2406.14496)
- **What's New**: 최근 거대 비전-언어 모델 (LVLMs, Large Vision-Language Models)은 다양한 이미지 이해 및 추론 작업에서 인상적인 능력을 보여주고 있습니다. 그러나 세분화된 객체 분류(예: 동물 종 간의 구별)는 아직 충분히 탐구되지 않았습니다. 이를 보완하고자, 우리는 	exttt{FOCI} (Fine-grained Object ClassIfication)라는 세분화된 객체 분류를 위한 어려운 다중 선택형 벤치마크를 기존의 객체 분류 데이터셋에서 만들었습니다. 우리는 12개의 공개 LVLMs를 	exttt{FOCI}에 대해 벤치마킹한 결과, CLIP 모델이 LVLMs보다 훨씬 좋은 성능을 보임을 확인했습니다.

- **Technical Details**: FOCI는 여러 인기 있는 분류 데이터셋과 ImageNet-21k의 4개의 도메인별 하위 집합(동물, 식물, 음식, 인공 객체)에서 구성되었습니다. 다중 선택형 문제는 개방형 질문 응답 과제의 모호한 답변을 피하고, CLIP 모델을 사용하여 어려운 음성 라벨을 채굴함으로써 분류 난이도를 유지했습니다. 이는 모델의 세분화된 객체 인식 능력을 테스트하려는 종합적인 벤치마크입니다.

- **Performance Highlights**: 많은 LVLMs들이 세분화된 객체 분류에서 어려움을 겪는 것으로 나타났습니다. 특히 CLIP 모델이 LVLMs보다 훨씬 뛰어난 성능을 보였으며, 이는 이미지 인코더와 LLM 간의 불충분한 정렬을 시사합니다. 더 큰 LLMs 및 더 강력한 이미지 인코더가 성능을 향상시켰고, 세분화된 객체의 이름을 명시하는 자막을 포함하는 훈련 데이터가 분류 성능에 도움이 되었습니다.



### Does Object Grounding Really Reduce Hallucination of Large Vision-Language Models? (https://arxiv.org/abs/2406.14492)
- **What's New**: 최근 대형 비전-언어 모델(LVLMs)이 이미지 캡션 생성 및 이미지 이해 작업에서 획기적인 성과를 보이고 있습니다. 그러나 LVLMs는 종종 이미지에 존재하지 않는 개념을 언급하는 '환각' 현상을 보이며, 이는 이 모델들의 신뢰성을 약화시킵니다. 기존 연구들은 이미지 영역 또는 객체를 텍스트 범위와 명시적으로 맞추는 '그라운딩 목표'를 포함하면 이러한 환각이 줄어든다고 주장해왔습니다. 하지만, 본 연구에서는 그라운딩 목표가 실제로 자유로운 캡션 생성(open generation)에서 LVLM 환각 감소에 미치는 영향을 처음으로 체계적으로 분석하였습니다.

- **Technical Details**: 연구에서는 LVLM 훈련에 그라운딩 목표(Referring Expressions(RE)와 Grounded Captioning(GC))를 추가하여 환각 정도를 측정했습니다. RE 목표는 텍스트 설명과 이미지 영역의 경계 상자를 생성하는 반면, GC 목표는 이미지 설명에 언급된 객체의 경계 상자를 포함하도록 요구합니다. 실험에는 세 가지 LLM 백본을 사용하였으며, 자유로운 캡션 생성과 질문 응답(QA) 기반 평가 두 가지 모두에서 그라운딩 목표가 환각 감소에 미치는 영향을 비교 분석했습니다.

- **Performance Highlights**: 세 가지 다른 LLM 백본을 대상으로 한 실험 결과, 그라운딩 목표를 포함한 LVLM 훈련은 객체 환각에 거의 영향을 미치지 않는 것으로 나타났습니다. 특히, 그라운딩 목표를 강제하는 경우 환각이 약간 줄어들었지만 이는 캡션 상세성의 약간의 감소를 대가로 합니다. 모델이 언급된 객체의 경계 상자를 생성하도록 강제하는 것이 환각을 방지하지 못한다는 점을 확인했습니다. 때문에, 그라운딩 목표가 환각을 의미 있게 줄이지 못한다는 점에서 새로운 방법론적 제안이 필요하다는 결론을 내렸습니다.



### On Layer-wise Representation Similarity: Application for Multi-Exit Models with a Single Classifier (https://arxiv.org/abs/2406.14479)
- **What's New**: 이 논문은 트랜스포머(Transformer) 모델의 은닉 층 간 표현 유사성을 분석하는 새로운 접근 방식을 제안합니다. 구체적으로, 샘플 단위 코사인 유사도(cosine similarity) 메트릭을 사용하여 표현 유사성을 평가하고, 기존의 복잡한 CKA(Centered Kernel Alignment) 방식과의 일치를 보입니다.

- **Technical Details**: 트랜스포머 모델에서는 각 층에서 동일한 차원의 표현이 생성되며, 이는 레이어 간 회전 차이가 적다는 특성을 가집니다. 이를 바탕으로, 샘플 단위 코사인 유사도를 사용해 각 층의 표현 유사성을 분석하였습니다. 또한, 'Aligned Training'이라는 새로운 학습 방식을 제안하여 은닉층 간의 유사성을 향상시킵니다. 이 학습 방식은 마지막 레이어의 분류기를 모든 은닉층에 적용하고 교차 엔트로피 손실을 최소화합니다.

- **Performance Highlights**: 실험 결과, 일반적인 트랜스포머 모델에서 은닉층 간 표현이 긍정적인 상관관계를 가지며, 층이 가까울수록 유사성이 증가하는 것을 확인했습니다. 제안된 Aligned Training 방법을 통해 shallow layer의 표현 유사성을 향상시켜, 각각의 중간 층의 정확도가 표준 훈련 방식보다 높아지는 결과를 보였습니다. 또한, 단일 분류기로 멀티-엑싯(multi-exit) 모델을 구현하여, 다양한 메모리 크기 및 계산 예산에 맞게 조정 가능한 효율적인 추론이 가능합니다.



### Data-Centric AI in the Age of Large Language Models (https://arxiv.org/abs/2406.14473)
Comments:
          Preprint

- **What's New**: 이 포지션 페이퍼는 AI 연구의 데이터 중심적 관점을 제안하며, 특히 대규모 언어 모델(LLMs) 연구에 초점을 맞춥니다. 데이터가 LLM의 개발 단계(예: pretraining과 fine-tuning)와 추론 단계(예: in-context 학습)에서 중요한 역할을 하지만, 연구 커뮤니티에서 상대적으로 적은 주목을 받고 있다는 주요 관찰에서 시작합니다. 데이터 중심 벤치마크, 데이터 큐레이션, 데이터 어트리뷰션, 지식 전이 및 추론 맥락화를 다루는 네 가지 특정 시나리오를 식별하고, 데이터의 중요성과 유망한 연구 방향을 강조하며 사회적 영향을 설명합니다.

- **Technical Details**: 소개 부분에서, 최근의 LLM들은 주로 인터넷에서 스크랩된 방대한 원시 데이터의 코퍼스를 기반으로 훈련된 후, 특화된 도메인 데이터로 fine-tuning됩니다. 이 논문은 데이터의 형태, 규모, 사용 방식 등에서 데이터가 공통 분모 역할을 한다고 지적하면서, 현재까지의 대부분의 연구가 모델링 개선에 집중되어 있고, 데이터의 최적 사용 방법에 대해서는 거의 알려져 있지 않다고 주장합니다. 특히 pretraining과 fine-tuning 단계에서의 데이터 큐레이션 방법이 부족하며, 기존의 수동 접근 방식이 일반화하기 어렵고 비용이 많이 든다고 지적합니다.

- **Performance Highlights**: LLMs의 성능은 사용자 제공 맥락 데이터의 품질과 정렬에 민감하며, 데이터 중심 연구의 중요성을 강조합니다. 예를 들어, CLIP 모델은 4억 개의 이미지-텍스트 쌍에서 훈련되었고, InstructGPT는 수만 개의 사용자 제공 프롬프트를 기반으로 합니다. 이 논문은 향후 더 엄격한 데이터 중심 벤치마크를 구축하고, 데이터 어트리뷰션과 제거, 지식 전이 및 추론 맥락화의 유망한 방향을 제시합니다. 또한, 작은 모델에서 더 큰 모델의 지식을 변환하여 비용 효율적인 대안을 제공하는 방향도 논의됩니다.



### A Review of Common Online Speaker Diarization Methods (https://arxiv.org/abs/2406.14464)
Comments:
          6 pages

- **What's New**: 이 논문은 전통적인 오프라인 화자 분할 시스템의 범위를 넘어, 실시간으로 수행되는 온라인 화자 분할(online speaker diarization)의 개념과 이를 구현하는 시스템을 종합적으로 탐구합니다. 온라인 화자 분할의 역사, 분류 방법, 평가 메트릭, 데이터셋, 그리고 다양한 온라인 화자 분할 시스템과 그 도전에 관한 정보들을 포함하고 있습니다.

- **Technical Details**: 온라인 화자 분할 시스템은 오프라인 시스템과 비슷하지만, 오디오 스트림 형태로 입력을 받아야 한다는 점에서 차별화됩니다. 주요 작업(task)으로는 '음성 활동 검출(Speech Activity Detection, SAD)', '분할(Segmentation)', '클러스터링(Clustering)'이 포함됩니다. 초기 연구들은 주로 Hidden Markov Models (HMM)와 Gaussian Mixture Models (GMM)을 사용하였으며, i-vectors와 d-vectors로 진화했습니다. 최신 연구는 end-to-end 시스템을 이용해 모든 서브태스크를 단일 신경망으로 처리하고 있으며, 특히 self-attention과 결합하여 성능을 향상시키고 있습니다.

- **Performance Highlights**: 주요 평가 메트릭으로 'Diarization Error Rate (DER)'와 'Jaccard Error Rate (JER)'가 있으며, 주로 DER이 사용됩니다. DER은 세 가지 오류, 즉 'False alarm (FA)', 'Missed Speech (MS)', 'Speaker Confusion (SC)'의 합으로 계산됩니다. 온라인 화자 분할 시스템의 데이터셋으로는 CALLHOME, 2003 NIST Rich Transcription, DIHARD, VoxConverse 등이 자주 사용됩니다.



### FVEL: Interactive Formal Verification Environment with Large Language Models via Theorem Proving (https://arxiv.org/abs/2406.14408)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)을 활용한 새로운 형식 검증 환경인 FVEL을 제안하였습니다. FVEL은 프로그램 코드를 형식 검증 도구인 Isabelle로 변환하고, LLM을 통해 자동 정리 증명을 수행하여 검증하는 시스템입니다. 이를 통해 전통적인 심볼릭 검증(symbolic verification)이나 수작업 규칙(hand-craft rules) 기반의 검증 방식의 한계를 극복할 수 있습니다.

- **Technical Details**: FVEL 시스템은 주어진 코드를 Isabelle 형식으로 변환하고, 이를 기반으로 정리를 생성한 다음 전체 증명 과정을 수행합니다. 이 과정에서 LLM은 초기 Isabelle 형식화 결과를 받아 정리를 도출하고, 증명 상태 및 피드백을 기반으로 상호작용을 지속합니다. FVEL 환경을 구축하기 위해 FVELER 라는 대규모 데이터셋이 작성되었으며, 이 데이터셋은 758개의 이론, 29,125개의 정리, 200,646개의 증명 단계로 구성되어 있습니다. 이를 통해 LLM을 미세 조정하고 Code2Inv 및 SV-COMP 벤치마크에서 평가를 수행합니다.

- **Performance Highlights**: FVELER 데이터셋으로 미세 조정된 LLM을 사용한 결과, SV-COMP 벤치마크에서 FVEL 시스템이 성능 향상을 보였습니다. 예를 들어, 라마3-8B(Llama3-8B)는 17.39%(69 -> 81) 더 많은 문제를 해결했고, 미스트랄-7B(Mistral-7B)는 12%(75 -> 84) 더 많은 문제를 해결했습니다. 또한, 증명 오류 비율이 감소하여 FVEL과 FVELER의 효용성을 입증하였습니다.



### Jailbreaking as a Reward Misspecification Problem (https://arxiv.org/abs/2406.14393)
- **What's New**: 최근 대형 언어 모델(LLMs)의 채택이 확산되면서, 이들의 안전성과 신뢰성에 대한 우려가 커지고 있습니다. 이번 연구에서는 이러한 취약성을 보상 지정 오류(reward misspecification)에 기인한다고 보고, 이를 정량화할 수 있는 ReGap 메트릭을 소개했습니다. 이 메트릭을 이용하여 유해한 backdoor prompts를 효과적으로 탐지할 수 있으며, 이를 기반으로 자동 red teaming 시스템인 ReMiss를 제안합니다.

- **Technical Details**: 보상 지정 오류(ReGap)를 정량화하는 새로운 메트릭을 소개하였으며, 이는 보상 함수가 응답의 품질을 정확하게 순위매기지 못한다는 점에 기반하고 있습니다. ReMiss 시스템은 다양한 대형 언어 모델에 대해 적대적 프롬프트(adversarial prompts)를 생성하여, AdvBench 벤치마크에서 최첨단 공격 성공률을 달성합니다. 이 과정에서 생성된 프롬프트의 가독성은 유지됩니다.

- **Performance Highlights**: ReMiss는 AdvBench 벤치마크에서 최고의 공격 성공률을 기록하였으며, 생성된 PRs의 인간 가독성을 유지했습니다. 이 연구의 심층 분석을 통해 제안된 보상 지정 오류 목표(ReGap)가 기존 방법들에 비해 유리한 점들을 잘 보여주었습니다. 이를 통해 타겟 모델의 다양한 실패 모드를 발견할 수 있어, 감사를 더욱 효과적으로 만들어줍니다.



### Artificial Leviathan: Exploring Social Evolution of LLM Agents Through the Lens of Hobbesian Social Contract Theory (https://arxiv.org/abs/2406.14373)
- **What's New**: 새로운 연구는 대형 언어 모델(LLMs) 및 인공지능(AI)의 발전을 이용하여, 복잡한 사회적 관계가 동적으로 형성되고 진화하는 모의 에이전트 사회를 소개합니다. 이 연구에서는 에이전트들이 심리적 동기를 부여받고, 생존 환경에서 상호 작용하는 '샌드박스' 시뮬레이션을 구현했습니다.

- **Technical Details**: 이번 연구에서는 에이전트들이 생존 본능으로 행동하며, 제한된 자원을 두고 협력과 경쟁을 통해 사회 진화를 경험하는 다중 에이전트 시스템을 탐구했습니다. 에이전트들은 초기에는 경쟁하지만 시간이 지나면서 협력을 배우고, 서로 자원을 평화롭게 교환하며 보호를 기대하는 동맹을 형성합니다. 연구는 또한 토마스 홉스의 '사회 계약 이론'(Social Contract Theory, SCT)을 기준으로 에이전트들의 행동을 평가했습니다.

- **Performance Highlights**: 실험 결과, 에이전트들은 초기에는 자연 상태에서의 무자비한 경쟁상태를 거치지만, 점차 사회 계약을 형성하여 절대 주권을 인정하고, 협력을 통해 평화로운 공공체를 구축하는 것으로 드러났습니다. 이 시뮬레이션은 대형 언어 모델이 인간 사회의 복잡한 동역학을 모델링할 수 있음을 시사합니다.

- **Impact and Future Work**: 이번 연구는 LLM 기반의 다중 에이전트 시뮬레이션이 사회 과학 연구와 복잡한 인간 시스템을 이해하는 데에 중요한 도구가 될 가능성을 제시합니다. 또한, 연구자들이 다양한 사회 과학적 가설을 실험할 수 있는 확장 가능한 시뮬레이션 플랫폼을 제공합니다.



### The neural correlates of logical-mathematical symbol systems processing resemble that of spatial cognition more than natural language processing (https://arxiv.org/abs/2406.14358)
- **What's New**: 본 연구는 수리 논리적 기호(LMS: Logical-Mathematical Symbols) 처리에서 공간 인지 (Spatial Cognition)가 언어 처리 (Language Processing)보다 더욱 중요한 역할을 한다는 새로운 발견을 제시합니다.

- **Technical Details**: 자동화된 메타 분석 (Automated Meta-Analysis)와 세 개의 대표적인 LMS 작업(추론, 계산, 정신적 프로그래밍)의 합성 맵을 사용하여 뇌의 상관성을 비교하였습니다. 그 결과, LMS 처리와 공간 인지 사이에 더 큰 피질 겹침이 관찰되었으며, 다변량 활성 패턴 분석에서 LMS 처리가 언어 처리보다 공간 인지와 더 큰 유사성을 보였습니다.

- **Performance Highlights**: 계층적 클러스터링 분석 (Hierarchical Clustering Analysis)은 대표적인 LMS 작업이 신경 수준에서 공간 인지 작업과 구분할 수 없음을 보여주었습니다. 이는 LMS 처리의 기초가 공간 인지일 가능성을 강력히 지지하며, 이는 논리적 추론에서 텍스트 데이터만으로 훈련된 대형 언어 모델의 한계에 대한 이해를 높이는 데 기여할 수 있습니다.



### LiveMind: Low-latency Large Language Models with Simultaneous Inferenc (https://arxiv.org/abs/2406.14319)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)을 위한 새로운 저지연 추론 프레임워크인 'LiveMind'를 소개합니다. LiveMind는 불완전한 프롬포트에서도 추론을 수행할 수 있으며, 이를 통해 지연 시간을 크게 줄이고 상호작용 경험을 개선할 수 있습니다. 특히, 이는 LLM와 작은 언어 모델(SLM)의 협력을 통해 성능을 최적화합니다.

- **Technical Details**: LiveMind는 사용자 프롬포트가 완전히 입력될 때까지 대기하지 않고, 입력이 진행되는 동안 바로 추론을 시작하는 방식으로 기존의 추론 방법론보다 더 빠르게 결과를 반환합니다. 이러한 접근 방식은 인간 대화 방식에서 영감을 받아 개발되었으며, 사용자가 프롬포트를 입력하는 동안 모델이 미리 처리할 수 있도록 설계되었습니다. 이를 통해 최종 프롬포트가 완성된 후의 최종 추론 단계에서 필요한 시간과 리소스를 줄일 수 있습니다.

- **Performance Highlights**: MMLU-Pro 데이터셋에서, LiveMind는 기존 방법론에 비해 평균 59%의 응답 지연 시간 감소를 달성했습니다. LLM와 SLM을 결합하여 추론과 출력을 병행한 경우, 응답 지연 시간이 평균 68% 줄어들었으며, 정확도는 5.5% 향상되었습니다. 20문장을 초과하는 긴 프롬포트에서는 지연 시간이 최대 93%까지 감소했습니다.



### The Fire Thief Is Also the Keeper: Balancing Usability and Privacy in Prompts (https://arxiv.org/abs/2406.14318)
- **What's New**: 온라인 챗봇의 빠른 채택은 인공지능의 주요 진전을 나타냅니다. 하지만 사용자 프롬프트(prompt)에 민감한 정보가 있을 수 있어 프라이버시 문제가 발생합니다. 이에 따라, 'Prompt Privacy Sanitizer'(ProSan)라는 프롬프트 프라이버시 보호 프레임워크를 소개합니다. ProSan은 프라이버시가 제거된 익명화된 프롬프트를 생성할 수 있으며, 작업의 사용성을 유지하면서 인간 가독성도 높입니다.

- **Technical Details**: ProSan은 사용성 높은 프롬프트와 동적 익명성을 위해 보호 대상과 강도를 유연하게 조절합니다. ProSan은 사용자의 컴퓨팅 자원 조건에 맞게 조정하여 프라이버시 보호를 보장합니다. 프롬프트의 단어 중요도와 프라이버시 누출 위험을 기준으로 보호 강도를 조절하며, 마스크드 언어 모델(masked language model)을 이용해 문맥 기반으로 익명화된 단어를 생성하여 유창성을 향상시킵니다.

- **Performance Highlights**: ProSan은 질문 응답, 텍스트 요약, 코드 생성 등 다양한 작업에서 민감한 정보를 효과적으로 제거하면서도 작업 성능에 거의 손실이 없습니다. 실험 결과, 민감한 정보를 정확하게 식별하고 대체하면서도 작업의 유용성에는 최소한의 영향을 미쳤습니다.



### QuST-LLM: Integrating Large Language Models for Comprehensive Spatial Transcriptomics Analysis (https://arxiv.org/abs/2406.14307)
Comments:
          12 pages, 7 figures

- **What's New**: 이 논문은 거대 언어 모델(LLMs)의 능력을 활용하여 공간 전사체학(ST) 데이터를 분석하고 해석하는 혁신적인 도구인 QuST-LLM을 소개합니다. QuST-LLM은 데이터 로딩, 영역 선택, 유전자 발현 분석 및 기능 주석을 포함하는 종합적인 워크플로우를 제공하여 ST 데이터의 해석 가능성을 크게 향상시킵니다. 이를 통해 사용자는 자연어를 통해 자신의 ST 데이터를 상호작용할 수 있습니다.

- **Technical Details**: QuST-LLM은 QuPath와 QuST의 두 가지 주요 구성 요소를 기반으로 합니다. QuPath는 고해상도 조직 이미지를 시각화하고 분석하는 데 널리 사용되는 오픈 소스 소프트웨어 플랫폼으로, QuST는 이와 통합되어 공간 전사체학(ST) 및 전 슬라이드 이미지(WSI) 분석을 지원합니다. QuST-LLM은 SCANPY, GOATOOLS, 그리고 거대 언어 모델(LLMs)을 활용하여 생물학적 의미를 쉽게 이해할 수 있는 설명으로 변환합니다. 또한, 이 도구는 전방 및 후방 분석 시나리오를 제공하여 연구자가 ST 데이터를 해석하는 다양한 방법을 지원합니다.

- **Performance Highlights**: QuST-LLM은 복잡하고 고차원적인 ST 데이터를 쉽게 해석 가능한 생물학적 내러티브로 변환하여 연구자들에게 조직의 공간적 및 기능적 복잡성을 이해하는 데 강력한 기능을 제공합니다. 이를 통해 ST 데이터의 해석 가능성을 크게 향상시키고, 생물의학 연구에 새로운 통찰력을 제공합니다.



### Ranking LLMs by compression (https://arxiv.org/abs/2406.14171)
Comments:
          7 pages, 4 tables

- **What's New**: 해당 연구는 정보 압축 과정을 이해의 핵심으로 보고, LLMs (Large Language Models, 대형 언어 모델)을 무손실 데이터 압축 기반으로 순위를 매기는 새로운 방법을 제안합니다. 모델의 사전 학습 단계와 산술 코딩(arithmetic coding) 하의 압축 길이의 동등성을 입증하고, 평가 메트릭으로서의 압축 비율을 실제 데이터 압축 없이 계산할 수 있도록 하였습니다. 이는 모델의 일반화를 평가하는 데 중요한 지표로 사용할 수 있습니다.

- **Technical Details**: 본 논문에서는 LLMs를 데이터 압축의 사전 모델로 사용하고, 다양한 자연어 처리(NLP) 과제 (문장 완성, 질문 답변, 상관 관계 해소 등)에서 그 성능을 비교합니다. 산술 코딩 기반의 데이터 압축 과정에서 생성된 부호 길이와 모델의 누적 음의 로그 확률들이 동등하다는 것을 증명하였습니다. 또한 이 접근법으로는 실제 데이터를 압축할 필요 없이 평가를 수행할 수 있어 효율성을 확보할 수 있습니다.

- **Performance Highlights**: LLMs를 사용한 무손실 데이터 압축 테스트에서, 압축 비율과 모델 성능이 양의 상관관계를 가짐을 발견하였습니다. 다섯 가지 대형 모델을 통해 실험한 결과, 압축 비율이 더 높은 모델이 NLP 과제에서 더 나은 성능을 발휘하는 것을 확인하였습니다. 따라서 압축 비율을 사용하면 모델의 다양한 과제에서의 성능을 통합적으로 평가할 수 있습니다.



### A Data-Driven Guided Decoding Mechanism for Diagnostic Captioning (https://arxiv.org/abs/2406.14164)
Comments:
          [Pre-print] ACL Findings 2024, 17 pages, 7 figures, 7 tables

- **What's New**: 이번 연구에서는 진단 문장 생성을 위한 새로운 데이터 기반 유도 디코딩 방법을 제안합니다. 이 방법은 이미지에 해당하는 주요 의학 조건을 반영하는 태그를 빔 서치(beam search) 과정에 통합하여 정확성을 높입니다.

- **Technical Details**: 제안된 Distance from Median Maximum Concept Similarity (DMMCS) 방법은 입력 이미지의 의학 태그 정보를 통해 단어 선택을 가이드하는 데이터 기반 디코딩 전략입니다. 각 디코딩 단계에서 새로운 패널티를 정의하여 태그와 비슷한 의미를 갖는 단어를 우선하여 생성하도록 합니다.

- **Performance Highlights**: 두 개의 의료 데이터셋, ImageCLEFmedical 2023 및 MIMIC-CXR에서 4개의 서로 다른 진단 문장 생성 시스템을 사용하여 평가한 결과, 대부분의 경우 제안된 DMMCS 방법이 모든 평가 지표에서 성능을 향상시킴을 보였습니다. 특히, 정확한 태그를 사용했을 때 성능 향상이 더 크게 나타났습니다.



### DIRAS: Efficient LLM-Assisted Annotation of Document Relevance in Retrieval Augmented Generation (https://arxiv.org/abs/2406.14162)
- **What's New**: 이번 논문에서는 Retrieval Augmented Generation (RAG) 시스템에서 정보 검색(IR) 성능 평가를 위한 새로운 도메인-특정 벤치마크를 자동으로 주석 달 수 있는 DIRAS (Domain-specific Information Retrieval Annotation with Scalability)를 소개합니다. DIRAS는 GPT-4 수준의 성능을 가진 오픈 소스 언어 모델(LLM)을 미세 조정하여 관련성 레이블을 주석 달 수 있게 합니다.

- **Technical Details**: DIRAS는 도메인-특정 데이터에서 (query, document) 쌍을 생성하고, 오픈 소스 LLM을 미세 조정해 관련성을 예측하는 방식으로 작동합니다. 이 시스템은 효율적이고 효과적인 포인트 방식 접근법을 이용하여, 모든 (query, document) 쌍을 평가하고 점수를 예측합니다. 주목할 만한 것은 이러한 접근법이 도메인-전문가 또는 LLM에 의해 설계된 관련성 정의를 활용하여 일관성 있는 주석을 다는 것입니다.

- **Performance Highlights**: DIRAS 미세 조정된 모델은 GPT-4 수준의 성능을 보여주며, 특히 ChatReport와 ClimRetrieve 데이터셋에서 뛰어난 성능을 발휘합니다. 이를 통해 DIRAS는 정보 검색 주석 편향을 줄이고, 모든 (query, document) 쌍에 대해 체계적으로 주석을 달아 RAG 시스템의 실제 성능 평가를 돕습니다.



### Watching the Watchers: A Comparative Fairness Audit of Cloud-based Content Moderation Services (https://arxiv.org/abs/2406.14154)
Comments:
          Accepted at European Workshop on Algorithmic Fairness (EWAF'24)

- **What's New**: 이 연구는 네 가지 주요 클라우드 기반 콘텐츠 관리 서비스의 공정성을 평가하는 최초의 종합적인 연구로서, 외부 감사를 통해 소수자와 취약 계층에 대한 편견을 지적합니다. 이 연구는 특히 암묵적인 혐오 발언 탐지와 그룹별 불공정성 문제를 드러냅니다.

- **Technical Details**: 연구에서는 Google Moderate Text API, Amazon Comprehend, Microsoft Azure Content Moderation, OpenAI Content Moderation API를 평가하였으며, MegaSpeech, Jigsaw, HateXplain, ToxiGen 데이터를 사용했습니다. 다양한 성능 지표를 통해 각 서비스의 혐오 발언 탐지 성능과 그룹별 편견을 분석했습니다. 또한 Perturbation Sensitivity Analysis (PSA)를 통해 반사실적 공정성(counterfactual fairness)을 평가했습니다.

- **Performance Highlights**: 연구 결과, 모든 서비스가 암묵적인 혐오 발언 탐지에 어려움을 겪었으며, 특히 PoC와 LGBTQ+ 그룹에 대해 과잉 검열(overmoderation) 경향을 보였습니다. OpenAI의 콘텐츠 관리 알골리즘이 Megaspeech 데이터셋에서 가장 높은 성능을 보였고, Amazon Comprehend가 Jigsaw와 ToxiGen에서 잘 수행되었습니다. 반면에, Google의 API는 전반적으로 가장 낮은 성능을 보였습니다.



### Towards Event-oriented Long Video Understanding (https://arxiv.org/abs/2406.14129)
Comments:
          Work on progress

- **What's New**: 최근 비디오 멀티모달 대형 언어 모델(Video MLLMs)의 발전에 발맞춰, 이벤트 중심의 장시간 비디오 이해 벤치마크 'Event-Bench'가 제안되었습니다. 이는 기존 데이터셋의 단점인 'short-cut bias'를 해결하고자 개발되었습니다.

- **Technical Details**: Event-Bench는 기존 데이터셋과 사람의 주석을 바탕으로 구축되었으며, 6가지 이벤트 관련 작업과 2,190개의 테스트 인스턴스를 포함합니다. 또한, 비용 효율적인 방법인 'Video Instruction Merging'을 통해 이벤트 중심의 비디오 교육 데이터를 효과적으로 통합하는 방법을 제안합니다.

- **Performance Highlights**: Event-Bench에서 최고의 성능을 보인 모델인 GPT-4o는 전체 정확도 53.33%를 달성하며, 가장 뛰어난 오픈 소스 모델보다 41.42% 높은 성능을 보였습니다. 모든 코드, 데이터, 모델은 공개되어 있습니다.



### An Investigation of Prompt Variations for Zero-shot LLM-based Rankers (https://arxiv.org/abs/2406.14117)
- **What's New**: 새로운 연구는 특정 요소와 단어 선택이 zero-shot Large Language Models (LLMs)를 사용한 랭커(ranker)의 효과성에 미치는 영향을 체계적으로 이해하는 방법을 제시합니다. 다양한 zero-shot 랭킹 방법들이 최근 제안되었으며, 이 연구는 랭킹 알고리즘, 백본 LLM (예: GPT-3.5 vs. FLAN-T5), 그리고 프롬프트의 구성 요소와 단어 선택이 랭킹 성능에 미치는 영향을 분석하고 있습니다.

- **Technical Details**: 연구는 네 가지 주요 zero-shot 랭킹 알고리즘 (pointwise, pairwise, listwise, setwise)을 조사했습니다. 이들은 각기 다른 백본 LLM과 프롬프트 구성 요소, 단어 선택에 따라 차이를 보입니다. Pointwise는 개별 문서의 관련성을 평가하고, Pairwise는 두 문서의 상대적인 관련성을 평가하며, Listwise는 문서 리스트의 순서를 평가하고, Setwise는 문서 집합의 순서를 평가합니다. 실험에서 백본 LLM을 고정하고 프롬프트의 구성 요소를 통제하여 다양한 변형의 효과를 분석했습니다.

- **Performance Highlights**: 연구는 랭킹 알고리즘이 zero-shot LLM 랭킹 방법 간의 성능 차이에 기여하는 것을 확인했지만, 더 중요한 것은 프롬프트 구성 요소와 단어 선택이 랭킹 효과성에 미치는 영향입니다. 실험에서 프롬프트의 구성 요소와 단어 선택이 실제 랭킹 알고리즘보다 더 큰 영향을 미친 경우도 발견되었습니다. 따라서 이러한 요소들을 고려하면 랭킹 방법 간의 차이가 더 모호해질 수 있습니다. 이러한 결과는 미래 연구에서 랭킹 방법을 비교할 때 중요한 지침이 됩니다.



### EasyECR: A Library for Easy Implementation and Evaluation of Event Coreference Resolution Models (https://arxiv.org/abs/2406.14106)
Comments:
          14 pages, 4 figures, 12 tables

- **What's New**: 이번 연구에서는 최초의 오픈 소스 라이브러리 'EasyECR'를 개발하여, 다양한 데이터셋과 ECR 파이프라인을 평가하고 비교할 수 있는 표준화된 데이터 구조를 제공했습니다. 이는 ECR 작업의 평가를 보다 신뢰성 있게 만들고, 다른 도메인에서도 높은 일반화를 보장하는 강력한 ECR 파이프라인 개발을 촉진할 것입니다.

- **Technical Details**: EasyECR 라이브러리는 7개의 대표적인 ECR 파이프라인과 10개의 인기 있는 벤치마크 데이터셋을 통합했습니다. 이 라이브러리를 통해, 여러 데이터셋에서 다양한 모델을 평가할 수 있게 하고, 모델 비교의 공정성을 보장하도록 설계되었습니다. 또한, ECR 파이프라인의 재현성을 높이기 위해 일관된 설정에서 여러 모델을 비교 평가했습니다.

- **Performance Highlights**: 실험 결과, 대표적인 ECR 파이프라인은 여러 데이터셋에서 일반화되지 않음을 발견했으며, ECR 파이프라인을 평가할 때는 다양한 데이터셋을 사용하는 것이 필요함을 강조합니다. 또한, ECR 파이프라인 내의 모든 모델이 성능에 큰 영향을 미치므로, 하나의 모델을 비교할 때 나머지 모델들이 일관되도록 유지하는 것이 중요합니다. 실험 결과는 향후 연구를 위한 귀중한 기준점을 제공합니다.



### ReaLHF: Optimized RLHF Training for Large Language Models through Parameter Reallocation (https://arxiv.org/abs/2406.14088)
Comments:
          13 pages (15 pages with references), 13 figures

- **What's New**: 이번 연구에서는 인간의 피드백을 활용한 강화학습(RLHF)에서 병렬화 전략이 비효율적일 수 있는 문제를 해결하기 위해 새로운 접근법인 '파라미터 재할당(parameter ReaLlocation)'을 제안했습니다. 이를 바탕으로 ReaLHF라는 시스템을 개발하여 RLHF 훈련을 위한 효율적인 실행 계획을 자동으로 발견하고 실행할 수 있도록 하였습니다.

- **Technical Details**: ReaLHF는 강화된 데이터 흐름 그래프(augmented dataflow graph)를 통해 RLHF의 실행 계획을 공식화하고, 경량화된 비용 추정기(cost estimator)를 활용한 맞춤형 탐색 알고리즘으로 최적의 실행 계획을 찾아냅니다. 실행 계획이 도출되면 런타임 엔진(runtime engine)이 이 계획을 실행하여 모델 파라미터를 재할당하고, 병렬화를 효과적으로 수행합니다.

- **Performance Highlights**: 실험 결과, ReaLHF는 LLaMA-2 모델(최대 70억 파라미터)과 128개의 GPU를 사용하여 기준 시스템 대비 2.0배에서 최대 10.6배까지 속도 향상을 보여주었습니다. 또한, ReaLHF의 실행 계획은 Megatron-LM 기반의 추론적 접근법 대비 평균 26% 성능 향상을 나타냈습니다.



### Taxonomy-Guided Zero-Shot Recommendations with LLMs (https://arxiv.org/abs/2406.14043)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)을 활용한 추천 시스템(RecSys)의 문제점을 해결하기 위해 신규 방법인 TaxRec을 제안합니다. 이 방법은 체계적인 분류 체계 사전을 사용하여 항목 정보를 구조화하고 명확하게 만듭니다. 이를 통해 좀 더 효율적이고 정확한 추천을 가능하게 합니다.

- **Technical Details**: TaxRec은 두 단계로 구성됩니다. 첫 번째 단계는 분류 체계를 통한 항목 분류로, LLM에서 지식을 가져와 항목들을 체계적으로 조직합니다. 두 번째 단계에서는 LLM을 사용해 사용자의 이전 상호작용을 기반으로 추천을 수행함으로써 제로샷(zero-shot) 추천을 가능하게 합니다. 분류 체계 사전은 후보 항목 정보를 축약하여 LLM이 효율적으로 토큰을 활용하고, 비구조적인 제목으로 인한 모호성을 줄입니다.

- **Performance Highlights**: 실험 결과에 따르면, TaxRec은 기존의 제로샷 추천 시스템에 비해 추천 품질을 상당히 향상시키는 것으로 나타났습니다. 이는 TaxRec이 LLM 기반 추천 시스템의 효율성과 정확도를 크게 개선할 수 있음을 입증합니다.



### CryptoGPT: a 7B model rivaling GPT-4 in the task of analyzing and classifying real-time financial news (https://arxiv.org/abs/2406.14039)
Comments:
          Journ{é}e Nationale sur la Fouille de Textes, Pascal CUXAC; Adrien GUILLE; C{é}dric LOPEZ, Jun 2024, Lyon (Universit{é} Lumi{è}re Lyon 2), France

- **What's New**: 새롭게 소개된 CryptoGPT는 특정 작업에서 GPT-4와 경쟁할 수 있는 7B(70억 매개변수) 모델입니다. 이 모델은 암호화폐 시장의 금융 뉴스 분석을 실시간으로 수행하는 데 특화되어 있습니다. 이 프로젝트는 산업 현장에서 제한된 자원을 통해 고품질의 전용 대형 언어 모델 (LLM)을 개선하는 방법을 제시합니다.

- **Technical Details**: CryptoGPT는 반자동주석(semi-automatic annotation)과 QLoRA를 통한 전략적 파인튜닝(fine-tuning)을 활용하여 모델을 개선했습니다. 미스트랄-7B(Mistral-7B)와 라마-7B(Llama-7B) 같은 다른 7B 크기의 LLM들과 비교 실험을 수행했으며, GPT-3.5 및 GPT-4와도 성능을 비교했습니다. 주요 목표는 데이터의 외부 서버 전송을 피하여 보호(data protection), 주석 비용과 시간을 줄이는 것(annotation cost and time reduction), 모델의 크기를 관리하여 배포 비용을 제어(model size control)하는 것, 그리고 더 나은 분석 품질을 유지(maintaining better analysis quality)하는 것에 있습니다.

- **Performance Highlights**: CryptoGPT는 금융 정보의 분류뿐만 아니라 포괄적인 분석을 제공합니다. 비교 실험에서 다양한 LLM들과 유사한 성능을 보여주었으며, 특히 비용 효율성과 데이터 보호 측면에서 좋은 결과를 나타냈습니다.



### Toward Infinite-Long Prefix in Transformer (https://arxiv.org/abs/2406.14036)
- **What's New**: 이 연구에서는 Prefix Learning이란 프리픽스 길이 관점에서의 학습 능력을 탐구하기 위해 Neural Tangent Kernel (NTK) 기법을 도입했습니다. 이를 통해 시도한 무한 길이의 Prefix Learning 최적화 과정을 나타내며, 주어진 모든 데이터셋에 대해 수렴할 수 있는 over-parameterization 속성을 확인했습니다. 이와 더불어, NTK-Attention이라는 새로운 방법을 제안하여 프리픽스 길이에 상관없이 효율적으로 attention 계산을 수행할 수 있습니다.

- **Technical Details**: NTK 기법을 사용하여 무한 길이의 프리픽스 최적화 문제를 해결하려고 시도했습니다. 이 방법은 attention 네트워크에서 무한 길이의 프리픽스가 over-parameterization 속성을 가져 데이터를 성공적으로 학습할 수 있음을 보여줍니다. 또한, NTK-Attention이라는 새로운 방법은 입력 길이에 따른 시간 복잡도를 줄이며 각 attention head에 대해 두 개의 훈련 가능한 매개변수 Z와 k를 사용합니다.

- **Performance Highlights**: 실험 결과, NTK-Attention이 P-Tuning V2보다 우수한 성능을 나타냈으며, vision 데이터셋에서 5.74%, 자연어 데이터셋에서 1.07%의 정확도 향상을 보여주었습니다. 이는 프리픽스 길이 스케일링 법칙(scaling law)을 다시 한번 확인하는 결과였습니다.



### Demystifying Forgetting in Language Model Fine-Tuning with Statistical Analysis of Example Associations (https://arxiv.org/abs/2406.14026)
Comments:
          5 pages

- **What's New**: 이 논문은 새로운 태스크를 학습할 때 기존의 언어 모델(Language Models, LMs)이 이전에 학습한 예제를 잊어버리는 문제를 연구했습니다. 이러한 문제는 기존 시스템의 안정성을 저해할 수 있습니다. 본 연구에서는 새롭게 학습한 태스크와 잊혀진 예제들 사이의 상관관계를 분석했습니다. 이를 통해 D 이야기의 예제로부터 Z 매트릭스를 사용하여 이러한 상관 관계를 시각화하고, 새로운 태스크를 학습할 때 발생하는 잊어버림을 예측하는 방법을 제안했습니다.

- **Technical Details**: 본 연구는 고차원 $M \times N$ 매트릭스를 이용해 $M$개의 새로운 태스크와 $N$개의 이전 예제 간의 상관 관계를 분석하는 것을 포함합니다. 주로 OLMo-7B 및 OLMo-7B-Instruct 모델을 사용하는 실험에서, 각각 Dolma와 Tulu V2 데이터셋을 사용하여 사전 학습 및 미세 조정을 진행했습니다. 또한, k-nearest neighbor(KNN) 기반의 예측 방법을 도입하여, 기존의 트레이닝 가능한 LMs에 비해 성능이 우수함을 입증했습니다.

- **Performance Highlights**: 결과적으로, 특정 새로운 태스크나 이전 예제가 다른 예제들에 비해 더 자주 잊혀진다는 것을 발견했습니다. 시각화 및 통계 모델을 통해 복잡한 패턴이 존재함을 확인했습니다. D 매트릭스를 통한 예측이 매우 정콩적으로 이루어졌으며, 이전 접근법을 능가하는 성능을 보여주었습니다.



### Investigating the Pre-Training Dynamics of In-Context Learning: Task Recognition vs. Task Learning (https://arxiv.org/abs/2406.14022)
Comments:
          work in progress

- **What's New**: 이 논문에서는 in-context learning (ICL)의 출현이 task recognition (TR)과 task learning (TL)이라는 두 가지 주요 능력에 의존한다고 제안합니다. 이 연구는 사전 훈련 중 이 두 능력 간의 경쟁 관계와 이들이 ICL 성능에 미치는 영향을 처음으로 밝히고자 합니다.

- **Technical Details**: 연구진은 다양한 모델 체크포인트에서 TR과 TL의 성능을 측정하기 위해 입력-라벨 설정을 조작하는 새로운 메트릭스를 설계했습니다. 이로 인해 두 능력이 경쟁 관계에 놓여있다는 사실을 발견했습니다. 경쟁은 일반적으로 사전 훈련 중 '안정-상승' 패턴을 따르며, 경쟁 강도가 낮을수록 ICL 성능이 더 좋다는 강한 부정 상관 관계를 보여줍니다.

- **Performance Highlights**: 연구 팀은 이 두 능력을 효과적으로 통합하기 위해 적응형 앙상블 학습 (adaptive ensemble learning) 방법을 제안했습니다. 이를 통해 두 개의 작은 모델이 파라미터 수가 두 배 이상인 큰 모델보다 더 나은 성능을 발휘했습니다.



### CityGPT: Empowering Urban Spatial Cognition of Large Language Models (https://arxiv.org/abs/2406.13948)
- **What's New**: 새로운 연구 CityGPT는 대형 언어 모델(LLMs)을 도시 공간 이해 및 관련 문제 해결 능력을 향상시키기 위해 제안되었습니다. 이를 위해 도시 규모의 세계 모델을 구축하고, 다양한 지시 조정 데이터셋인 CityInstruction을 생성하여 도시 지식을 주입하고 공간 추론 능력을 강화했습니다.

- **Technical Details**: CityGPT는 일반 지시 데이터와 CityInstruction을 혼합하여 다양한 LLMs(예: ChatGLM3-6B, Qwen1.5, LLama3 시리즈)를 세밀 조정(fine-tune)함으로써, 일반 능력을 희생하지 않으면서 해당 모델들의 능력을 향상시킵니다. 또한, 제안된 방법의 유효성을 검증하기 위해 다양한 도시 시나리오와 문제를 평가할 수 있는 종합 벤치마크 CityEval을 구축했습니다.

- **Performance Highlights**: 종합 평가 결과, CityInstruction으로 훈련된 소형 LLMs은 CityEval의 종합 평가에서 상업용 LLMs와 경쟁력 있는 성능을 보여주었습니다.



### AspirinSum: an Aspect-based utility-preserved de-identification Summarization framework (https://arxiv.org/abs/2406.13947)
- **What's New**: AI 연구 커뮤니티는 최근 개인 관련 분야, 예를 들어 헬스케어나 교육 분야에서의 대규모 언어 모델(Large Language Model, LLM) 적용이 느린 원인을 탐구하고 있습니다. 공개 데이터셋의 부족으로 인해 그런 분야에서의 LLM 훈련이 어려운 상황입니다. 이러한 문제를 해결하기 위해 텍스트 비식별화(de-identification) 기술인 AspirinSum을 제안하고 있습니다. 이 기술은 개인의 민감한 정보를 제거하며 유용한 데이터를 유지하는 요약 프레임워크입니다.

- **Technical Details**: AspirinSum은 전문가의 기존 코멘트 데이터를 사용하여 텍스트 내 개인 민감 항목(PSA)을 파악하고, 이를 유사한 항목으로 대체하여 비식별화 요약을 생성합니다. 기존의 시퀀스 라벨링(sequence labeling) 모델이 아닌, 측면 기반(aspect-based) 방식을 사용하여 더 유연하게 다양한 도메인에 적용할 수 있도록 설계되었습니다. 특히, 기존의 인명식별정보(PII) 식별 방식과 달리 좀 더 포괄적인 접근 방식을 채택했습니다.

- **Performance Highlights**: AspirinSum은 비식별화된 텍스트가 공개 데이터의 발행 및 후속 작업에 사용될 수 있는 유용성을 유지하면서도 개인 식별 정보를 보호하는 것을 목표로 합니다. 이를 위해 유틸리티, 완전성, 재식별 가능성 등을 평가하는 새로운 점수 메트릭을 제안하고 활용할 계획입니다. 특히, 고등학생의 대학 지원서 데이터셋(HSSCAS)을 비식별화 요약해 공개할 예정입니다.



### CityBench: Evaluating the Capabilities of Large Language Model as World Mod (https://arxiv.org/abs/2406.13945)
- **What's New**: 새로운 연구는 도시 분야에서 LLMs(Large Language Models)의 평가를 위한 첫 번째 체계적 평가 벤치마크인 CityBench를 제안합니다. 이 플랫폼은 다양한 데이터를 통합하고 상호 작용할 수 있는 시뮬레이터를 기반으로 하여 도시 내에서 LLM의 성능을 평가합니다.

- **Technical Details**: CityBench는 두 가지 주요 구성 요소로 구성됩니다. 첫 번째는 CitySim이라는 시뮬레이션 모듈로, 이는 다양한 출처의 도시 데이터를 통합하고 세밀한 도시 역학을 시뮬레이션합니다. 여기에는 Open Street Map의 지리 공간 데이터, Google Map의 도시 비전 데이터, Foursquare와 같은 웹 사이트의 인간 활동 데이터가 포함됩니다. 두 번째 구성 요소는 평가 모듈로, LLMs의 도시 규모 세계 모델로서의 능력을 평가하기 위해 설계된 벤치마크를 포함합니다. 평가 과제는 지리 공간 이해 작업과 의사 결정을 포함한 두 가지 주요 범주로 나뉩니다.

- **Performance Highlights**: CityBench는 13개의 잘 알려진 LLMs(오픈 소스 및 상업용)를 13개의 다른 도시에서 평가하였으며, 그 결과 CityBench의 확장성과 효과성을 입증했습니다. 이번 연구는 도시 분야에서 LLMs를 세계 모델로 적용하는 데 있어서의 제한점을 발견하고, 향후 연구 방향을 제시하는 데도 기여했습니다. 데이터셋, 벤치마크, 소스 코드는 연구 커뮤니티에 공개적으로 접근 가능합니다.



### PIN: A Knowledge-Intensive Dataset for Paired and Interleaved Multimodal Documents (https://arxiv.org/abs/2406.13923)
- **What's New**: 최근의 대규모 멀티모달 모델(LMMs)에서 복잡한 지식 기반 작업을 향상시키기 위해 광범위한 멀티모달 데이터를 활용하고 있습니다. 하지만 LMM은 여전히 시각적 데이터 해석과 멀티모달 관계 추론에서 오류를 범하는 한계가 있습니다. 이러한 문제를 해결하기 위해, 우리는 PIN (Paired and INterleaved multimodal documents)이라는 새로운 데이터셋 형식을 도입했습니다. PIN은 지식 집약적이고 확장 가능하며 다양한 교육 모달리티를 지원하기 위해 설계되었습니다.

- **Technical Details**: PIN 형식은 마크다운 파일과 종합 이미지를 결합하여 밀도 높은 지식 구조와 다양한 교육 전략을 제공합니다. 이를 통해 LMM이 텍스트와 이미지 간의 상호작용을 더 잘 이해할 수 있게 합니다. PIN-14M이라는 오픈소스 데이터셋은 다양한 중국어 및 영어 소스에서 1,400만 개의 샘플을 포함하고 있으며, 복잡한 웹 및 과학 콘텐츠를 포함하고 있습니다. 이 데이터셋은 데이터 품질과 윤리적 무결성을 보장하도록 꼼꼼하게 구성되었습니다.

- **Performance Highlights**: 초기 결과는 PIN 형식이 LMM의 성능을 향상시키는데 상당한 잠재력을 가지고 있음을 시사합니다. 앞으로 더 큰 데이터셋과 다양한 평가를 통해 이 데이터 형식의 효과를 탐색할 계획입니다.



### The Use of Multimodal Large Language Models to Detect Objects from Thermal Images: Transportation Applications (https://arxiv.org/abs/2406.13898)
- **What's New**: 본 연구는 열영상 데이터와 다중 모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 통합이 자율 주행 시스템 및 다양한 지능형 교통 시스템(Intelligent Transportation Systems, ITS) 애플리케이션의 안전성과 기능성을 향상시킬 수 있는 흥미로운 기회를 제공한다고 밝혔습니다. 연구는 MLLMs가 RGB 및 열 카메라의 복잡한 이미지를 이해하고 객체를 직접 감지할 수 있는지를 조사했습니다.

- **Technical Details**: 연구 목표는 1) 다양한 정보 세트에서 MLLM이 학습할 수 있는 능력을 평가하고, 2) 열 카메라에서 객체를 감지 및 식별하며, 3) 두 독립적인 모달리티 이미지가 동일한 장면을 보여주는지 판단하고, 4) 여러 모달리티를 통해 모든 객체를 학습하는 것이었습니다. 연구 결과 GPT-4와 Gemini 모델 모두 열 이미지에서 객체를 감지하고 분류하는 데 효과적이라는 것이 입증되었습니다.

- **Performance Highlights**: 보행자 분류의 평균 절대 오차율(Mean Absolute Percentage Error, MAPE)은 각각 70.39% 와 81.48%로 나타났습니다. 또한 자전거, 자동차, 오토바이 감지의 MAPE 값은 각각 78.4%, 55.81%, 96.15%로 나타났습니다. Gemini 모델은 각각 66.53%, 59.35%, 78.18% 를 기록했습니다. 이 결과는 MLLM이 열 이미지를 식별할 수 있으며, ITS 애플리케이션을 위한 고급 영상 자동화 기술에 활용될 수 있음을 보여줍니다.



### StackRAG Agent: Improving Developer Answers with Retrieval-Augmented Generation (https://arxiv.org/abs/2406.13840)
- **What's New**: 개발자가 질문에 대한 정보를 찾을 때 많은 시간을 소비하는 문제를 해결하기 위한 새 도구인 StackRAG을 소개합니다. StackRAG은 Stack Overflow(SO)의 지식을 활용하여 LLM(대규모 언어 모델)의 생성 능력을 향상시키는 도구입니다. 이를 통해 생성된 응답이 신뢰할 수 있고 최신 정보를 포함하도록 합니다.

- **Technical Details**: StackRAG은 LLM 기반의 다중 에이전트(RAG-based Multiagent) 도구로, LangChain 에이전트 프레임워크를 사용하여 개발되었습니다. 키워드 추출기(Keyword Extractor), 검색 및 저장 컴포넌트(Search and Storage component), 증거 수집기(Evidence Gatherer), 그리고 답변 생성기(Answer Generator)의 네 가지 주요 구성 요소를 포함하고 있습니다. 사용자의 질문을 바탕으로 관련 키워드를 추출하고, 이를 SO에서 검색하여 관련 질문-답변 쌍을 수집하고, 이를 바탕으로 답변을 생성하는 방식으로 동작합니다. 모든 컴포넌트에서는 GPT 모델이 에이전트로 사용됩니다.

- **Performance Highlights**: 초기 평가 결과, StackRAG는 기본 LLM인 GPT 4와 비교할 때 더 정확하고, 관련성이 높으며 유용한 응답을 제공합니다.



### AlanaVLM: A Multimodal Embodied AI Foundation Model for Egocentric Video Understanding (https://arxiv.org/abs/2406.13807)
Comments:
          Code available this https URL

- **What's New**: 새로운 연구에서 인간과 효과적으로 협력할 수 있는 AI 개인 비서 개발을 위한 새로운 모델과 데이터셋을 소개했습니다. 현재의 VLMs는 주로 제3자 시점의 동영상에 초점을 맞추고 있어 자아중심적(perceptual) 경험을 충분히 반영하지 못하고 있습니다. 이 문제를 해결하기 위해 자아중심적 비디오를 이해하는 Egocentric Video Understanding Dataset(EVUD)을 도입하였습니다.

- **Technical Details**: 첫 번째로, 이 논문에서는 자아중심적 비디오 캡션 생성 및 질문-답변 작업을 위한 EVUD 데이터셋을 소개했습니다. 이 데이터셋은 강력한 사전 훈련된 대규모 언어 모델(LLM)을 활용하여 AI 시스템을 개선하는 데 중점을 두고 있습니다. 두 번째로, 7B 파라미터로 구성된 Vision-Language Model(VLM)인 AlanaVLM을 EVUD 데이터셋을 활용하여 효율적인 파라미터 훈련 기법으로 학습시켰습니다. 마지막으로, OpenEQA라는 도전적인 벤치마크를 사용하여 AlanaVLM의 능력을 평가한 결과, GPT-4와 같은 강력한 소크라틱 모델을 3.6% 초과하는 성과를 달성하였습니다.

- **Performance Highlights**: 우리는 AlanaVLM이 잘 설정된 벤치마크 OpenEQA에서 기존의 오픈 소스 모델을 능가하는 성과를 거두었으며, Claude 3와 Gemini Pro Vision 1.0을 초과하는 성능을 보였습니다. 또한, Gemini Pro 1.5와 GPT-4V와 비교했을 때도 경쟁력 있는 결과를 보였으며, 특히 공간 추론(spatial reasoning)에서 GPT-4V를 능가하는 성과를 나타냈습니다.



### IoT-Based Preventive Mental Health Using Knowledge Graphs and Standards for Better Well-Being (https://arxiv.org/abs/2406.13791)
Comments:
          20 pages

- **What's New**: 이번 연구는 정신 건강과 웰빙(SDG3)을 촉진하기 위한 디지털 기술의 역할을 중점적으로 다루고 있습니다. 특히, Wearable(웨어러블) 센서를 통해 수집된 생리 신호를 사용하여 감정을 지속적으로 모니터링할 수 있는 Digital Twin(디지털 트윈)의 가능성을 탐구하고, 이를 위해 Knowledge Graph(지식 그래프)가 필요함을 제시합니다.

- **Technical Details**: 연구에서는 US 국가 정신 건강 통계와 관련된 데이터를 인용하며, 표준화된 데이터 포맷, 통신 프로토콜, 데이터 교환 메커니즘 등이 필요함을 강조하고 있습니다. 디지털 트윈의 개념은 물리적 실체와 디지털 상태 간의 적절한 동기화를 가능케 하는 데이터 연결로 정의됩니다. 이 연구에서는 W3C의 RDF, OWL, SPARQL과 같은 시맨틱 웹 기술과 ISO, IEC, W3C 표준을 이용해 지식 그래프를 구축하는 방법을 자세히 설명합니다.

- **Performance Highlights**: 연구 결과, 지식 그래프는 정신 건강 프로젝트에서 중요한 역할을 할 수 있으며, 표준화된 데이터를 통해 보다 개인화된 건강 모니터링 체계를 구축할 수 있습니다. 이를 통해 정신 건강 관리의 품질과 접근성을 향상시키는 데 기여할 수 있습니다.



### A Primal-Dual Framework for Transformers and Neural Networks (https://arxiv.org/abs/2406.13781)
Comments:
          Accepted to ICLR 2023, 26 pages, 4 figures, 14 tables

- **What's New**: 이 논문에서는 transformers에서 주로 사용되는 self-attention 메커니즘을 수학적으로 해석하고 새로운 Attention 메커니즘 두 가지를 제안합니다. Batch Normalized Attention (Attention-BN)과 Scaled Head Attention (Attention-SH)을 통해 모델의 성능을 향상시키고 효율성을 높이는 방법을 소개합니다.

- **Technical Details**: Self-attention 메커니즘이 support vector regression 문제에서 파생된 support vector expansion과 상응할 수 있음을 보여줍니다. 이를 통해 Attention-BN은 배치 정규화 층(batch normalization layer)에서 유도되었고, Attention-SH는 더 적은 훈련 데이터로 SVR 모델을 적합하게 하는 방식에서 유도되었습니다. 이 두 가지 새로운 attention 메커니즘은 head 중복성을 줄이고 모델의 정확도를 높이며, 특히 이미지 및 시계열 분류 같은 다양한 실제 응용 분야에서 모델의 효율성을 향상시키는 데 도움을 줍니다.

- **Performance Highlights**: 제안된 Attention-BN과 Attention-SH 메커니즘을 사용한 모델은 head 중복성을 줄이고, 모델 정확도를 향상시키며, 모델의 효율성을 높이는 데 성공했습니다. 특히, 이미지 및 시계열 분류 응용 분야에서 이러한 개선 사항을 입증했습니다.



### Game of LLMs: Discovering Structural Constructs in Activities using Large Language Models (https://arxiv.org/abs/2406.13777)
Comments:
          6 pages, 2 figures

- **What's New**: 새로운 사람 활동 인식(Human Activity Recognition) 방법을 제안합니다. 기존에는 일정한 길이의 윈도우를 사용하여 활동을 인식했습니다. 그러나 스마트 홈에서의 활동은 지속 시간과 빈도가 다양하기 때문에 이러한 접근법은 효과적이지 않을 수 있습니다. 본 연구는 대형 언어 모델(LLMs)을 사용하여 활동의 기초 단위인 구조적 구성 요소를 식별하는 방법을 탐구합니다.

- **Technical Details**: 본 논문에서는 다양한 스마트 홈 환경에서 수집된 시계열 데이터를 분석하는 방법론을 제안합니다. 기존 방법론들은 일정한 윈도우 길이를 사용하여 데이터를 처리했으나, 스마트 홈 활동 데이터의 비일관된 샘플링 속도와 다양한 패턴으로 인해 이러한 접근법은 한계가 있습니다. 본 연구는 LLMs, 특히 GPT-4와 Gemini를 사용하여 활동의 기초 단위를 자동으로 식별하고 이를 통해 활동 모니터링을 개선합니다.

- **Performance Highlights**: 공개된 CASAS 벤치마크 데이터셋을 사용하여 실험을 수행했습니다. LLMs를 통해 활동 시퀀스를 구성하는 기본 구성 요소를 효과적으로 식별할 수 있음을 확인했으며, 이는 특히 짧은 시간 동안 발생하는 활동이나 드물게 발생하는 활동을 인식하는 데 유용할 수 있습니다. 이 기법은 활동 인식의 정확도를 높이고, 스마트 홈 환경에서의 활동 모니터링에 대한 신뢰성을 증대한 잠재력을 가집니다.



### Elliptical Attention (https://arxiv.org/abs/2406.13770)
Comments:
          38 pages, 7 figures, 12 tables

- **What's New**: 이번 연구에서는 트랜스포머(Transformers)의 핵심이 되는 쌍별 내적 자체-주의(Pairwise dot-product self-attention) 메커니즘의 한계를 보완하기 위해 'Elliptical Attention'을 제안합니다. 이는 마할라노비스 거리(Mahalanobis distance)를 사용하여 주의 가중치를 계산하는 새로운 방법입니다. 이를 통해 모델은 표현 붕괴(Representation collapse)를 줄이고, 오염된 샘플에 대한 내성을 향상시키며, 맥락적으로 중요한 정보에 더 집중할 수 있습니다.

- **Technical Details**: 'Elliptical Attention'은 주의 쿼리(Query) 주위에 초타원형 하이퍼-엘립소이달(neighbourhood)을 정의하여 방향성 정보를 고려한 주의 가중치를 할당합니다. 이는 마할라노비스 변환(Mahalanobis transformation)을 이용하여 특징 공간의 축을 조정하고, 맥락적으로 중요한 방향을 강조합니다. 본 기술은 쿼리 주위의 특징 공간을 늘리는 방식으로 작용하며, 이를 통해 높은 품질의 맥락 표현을 학습하도록 합니다. 또한 좌표별 관련성(Coordinate-wise relevance)을 추정하여 매우 효율적이고 학습 가능한 파라미터가 없는 추정기를 제안합니다.

- **Performance Highlights**: 엡틸리컬 주의(Elliptical Attention)는 다양한 작업들에서 기본 내적 주의 모델 및 최신 주의 방법보다 뛰어난 성능을 보여줍니다. 구체적으로, 이미지넷-1K(ImageNet-1K) 객체 분류, LRA(Long Sequence Modeling), ADE20K 이미지 분할, WikiText-103 언어 모델링 등에서 높은 정확도와 강건성을 입증했습니다. 또한, 적은 메모리 사용량과 빠른 계산 속도로 향상된 효율성을 제시하며, 최신 강건한 트랜스포머와 결합 시에도 성능이 추가로 향상됨을 보였습니다.



### Unveiling the Hidden Structure of Self-Attention via Kernel Principal Component Analysis (https://arxiv.org/abs/2406.13762)
Comments:
          33 pages, 5 figures, 12 tables

- **What's New**: 최근의 연구에서는 트랜스포머(Transformers)의 핵심인 셀프 어텐션(Self-Attention) 메커니즘을 커널 PCA(Kernel Principal Component Analysis)에서 유도했습니다. 이를 통해, 셀프 어텐션이 피처 공간(feature space) 내의 주요 성분 축(Principal Component Axes)에 쿼리 벡터(query vectors)를 투영함을 확인했습니다. 이러한 통찰을 바탕으로, 데이터 오염에 강인한 Robust Principal Components를 사용하는 새로운 종류의 어텐션 메커니즘인 RPC-Attention을 제안합니다.

- **Technical Details**: 기존 셀프 어텐션 메커니즘은 쿼리, 키, 값 행렬(query, key, value matrices)을 통해 입력 시퀀스를 변환합니다. 이 연구는 셀프 어텐션에서 쿼리 벡터가 피처 공간 내의 주요 성분 축에 투영된다는 것을 이론적으로 도출하고 실험적으로 입증했습니다. 또한, 값 행렬(value matrix)은 셀프 어텐션의 키 벡터(key vectors)의 그램 행렬(Gram Matrix)의 고유 벡터를 포착함을 확인했습니다. 이러한 통찰을 바탕으로 RPC-Attention을 설계하였으며, 이는 데이터 오염에 대한 저항력을 크게 향상시킵니다.

- **Performance Highlights**: 제안된 RPC-Attention의 성능을 실험적으로 평가한 결과, ImageNet-1K 객체 분류, WikiText-103 언어 모델링, ADE20K 이미지 세분화 작업에서 소프트맥스 어텐션(softmax attention)보다 뛰어난 성능을 보였습니다.



### GenAI-Bench: Evaluating and Improving Compositional Text-to-Visual Generation (https://arxiv.org/abs/2406.13743)
Comments:
          We open-source our dataset, model, and code at: this https URL ; Project page: this https URL ;. arXiv admin note: substantial text overlap with arXiv:2404.01291

- **What's New**: 최신 연구에서는 텍스트-비주얼 생성 모델(text-to-visual models)이 속성, 관계 및 논리와 비교와 같은 고급 추론을 포함하는 구성적 텍스트 프롬프트(compositional text prompts)에서 여전히 어려움을 겪고 있음을 보여줍니다. 본 연구는 GenAI-Bench에서 주요 이미지 및 비디오 생성 모델의 성능을 평가하기 위한 대규모 인간 연구를 수행하였습니다. 또한, VQAScore가 이전 메트릭인 CLIPScore에 비해 자동 평가 메트릭에서 월등히 뛰어나다는 것을 발견하였습니다.

- **Technical Details**: VQAScore는 VQA(Vision Question Answering) 모델이 이미지가 프롬프트를 정확하게 묘사하고 있는 확률을 측정하는 메트릭(metric)으로, 미세 조정 없이도 단순히 3~9개의 후보 이미지를 랭킹하는 방식으로 생성을 개선할 수 있습니다. DALL-E 3와 Stable Diffusion과 같은 모델의 인간 정합성을 크게 향상시키는데 2배에서 3배 더 효과적입니다. 또한, GenAI-Rank 벤치마크를 출시하여 4만 개 이상의 인간 평가를 통해 동일한 프롬프트로 생성된 이미지를 랭크링 할 수 있도록 할 예정입니다.

- **Performance Highlights**: VQAScore를 사용한 랭킹은 다른 메트릭인 PickScore, HPSv2, ImageReward보다 인간 품질 평가에서 두 배에서 세 배 더 효과적입니다. 특히 고급 비주어-언어 추론(Advanced visio-linguistic reasoning)이 필요한 구성적 프롬프트에서 성능이 두드러졌습니다. 연구팀은 향후 연구에서 세밀한 시각적 디테일을 해결함으로써 VQAScore를 더욱 개선할 가능성을 논의하고 있습니다. 연구팀은 과학적 벤치마킹을 위해 80,000개 이상의 인간 평가 데이터를 공개할 예정입니다.



### BEACON: Balancing Convenience and Nutrition in Meals With Long-Term Group Recommendations and Reasoning on Multimodal Recipes (https://arxiv.org/abs/2406.13714)
Comments:
          6 pages (including references), 1 figure, 2 tables

- **What's New**: 이 논문은 영양가 있는 식사와 편리함을 동시에 고려하는 새로운 식사 추천 시스템을 제안합니다. 이 시스템은 사용자에게 다양한 음식을 추천하고, 영양소와 요리 과정을 이해하며, 장기적인 식사 계획을 제공합니다.

- **Technical Details**: 논문에서 제안하는 시스템은 BEACON(Balancing Convenience and Nutrition Meals With Long-Term Group Recommendations and Reasoning on Multimodal Recipes)입니다. 이 시스템은 온라인 레시피 데이터를 활용하고, 도메인 지식을 결합하여 사용자의 식사 선호도를 고려한 추천을 실시합니다. 이를 위해 신경망을 통해 레시피를 텍스트에서 R3 포맷(리치 레시피 표현: Rich Recipe Representation)으로 변환하고, 컨텍스추얼 밴딧(Contextual Bandits) 학습 방법을 사용합니다.

- **Performance Highlights**: BEACON은 식사 구성 및 장기적인 추천을 위한 문제를 해결하고, 패스트푸드 레시피를 포함한 다양한 데이터를 R3 표현으로 변환합니다. 추천 시스템의 성능 평가에서는 중복, 커버리지 및 사용자 제약 만족도와 같은 지표를 도입하여 기존 베이스라인보다 더 효과적인 결과를 보여줍니다.



### VisualRWKV: Exploring Recurrent Neural Networks for Visual Language Models (https://arxiv.org/abs/2406.13362)
Comments:
          18 pages,14 tables,6 figures

- **What's New**: 이번 연구에서는 Visual Language Models(VLMs)에 기존의 Transformer 기반 모델 대신, 효율적인 선형 RNN 구조를 도입한 첫 사례인 VisualRWKV 모델을 소개합니다. 특히, 사전 학습된 RWKV 언어 모델을 활용하여, 데이터에 의존적인 반복(data-dependent recurrence), 샌드위치 프롬프트(sandwich prompts), 2D 이미지 스캐닝 메커니즘을 제안했습니다.

- **Technical Details**: VisualRWKV 모델은 RWKV 언어 모델을 기반으로 하며, 데이터 종속성(data-dependent recurrence)을 활용해 모델의 능력을 향상시키고, 샌드위치 프롬프트(sandwich prompts)와 2D 이미지 스캐닝 메커니즘을 도입하여 시각적 시퀀스 처리 능력을 강화했습니다. RWKV는 전통적인 Transformer's 자가 주의 메커니즘 대비 선형 확장성과 효율성을 가집니다.

- **Performance Highlights**: 광범위한 실험 결과, VisualRWKV는 다양한 벤치마크에서 LLaVA-1.5 등 Transformer 기반 모델과 비교해 경쟁력 있는 성능을 보였습니다. 이 연구는 VisualRWKV의 혁신적인 구조와 다양한 디자인이 어떻게 모델의 표현 능력을 강화하는지를 탐구합니다. 또한, 체크포인트와 관련 소스를 공개하여 추가 연구를 지원합니다. GitHub에서 확인할 수 있습니다.



### Textual Unlearning Gives a False Sense of Unlearning (https://arxiv.org/abs/2406.13348)
- **What's New**: 이 논문에서는 텍스트 데이터를 언ㄹ러닝 하는 과정이 오히려 데이터 유출 위험을 높일 수 있다는 문제를 제기하고 있습니다. 이를 위해 제안된 'Textual Unlearning Leakage Attack (TULA)'는 악의적인 공격자가 삭제된 데이터를 예상할 수 있는 방법을 소개하고, 블랙박스 및 화이트박스 시나리오에서 그 변형을 제시합니다.

- **Technical Details**: TULA는 모델의 언러닝 전후 정보를 활용해 삭제된 데이터를 예측할 수 있는 방법입니다. 블랙박스 시나리오에서는 모델 쿼리 결과를 비교하여 멤버십 정보 추론 공격을 시도하며, 화이트박스 시나리오에서는 모델 가중치 차이만으로 삭제된 데이터를 직접 재구성할 수 있는 가능성을 제시합니다. 실험적으로 TULA는 텍스트 언러닝이 실제로 더 높은 비율로 데이터를 노출시킨다는 사실을 입증했습니다. 블랙박스의 경우 멤버십 정보 노출 능력을 20% 이상 향상시키며, 화이트박스 시나리오에서는 60% 이상의 정확도로 데이터를 재구성할 수 있습니다.

- **Performance Highlights**: 세 개의 대형 언어 모델 아키텍처와 두 개의 데이터셋을 사용한 실험 결과, 언러닝 메커니즘은 모델에 의해 삭제된 데이터를 노출시키는 효과가 있음이 확인되었습니다. 블랙박스 시나리오에서 회원 정보 노출 위험은 20% 이상 높아졌으며, 화이트박스 시나리오에서는 삭제된 데이터를 60% 정확도로 재구성하는 것이 가능했습니다.



### Medical Spoken Named Entity Recognition (https://arxiv.org/abs/2406.13337)
Comments:
          Preprint, 40 pages

- **What's New**: 첫 번째 의료 분야의 음성 Named Entity Recognition(NER) 데이터셋인 VietMed-NER을 소개합니다. 이 데이터셋은 18가지의 의료적으로 정의된 엔티티 유형을 포함하여, 세계에서 가장 큰 음성 NER 데이터셋 중 하나로 자리매김했습니다.

- **Technical Details**: VietMed-NER 데이터셋은 실제 의료 Automatic Speech Recognition(ASR) 데이터셋인 VietMed에서 구축되었으며, Recursive Greedy Mapping이라는 새로운 어노테이션 기법을 도입하여 어노테이션 속도와 품질을 개선했습니다. 또한, monolingual w2v2-Viet와 cross-lingual XLSR-53-Viet를 ASR 태스크를 위해 파인튜닝한 두 가지 모델을 사용했습니다.

- **Performance Highlights**: 다양한 state-of-the-art 사전 학습된 모델(encoder-only, sequence-to-sequence)을 사용한 베이스라인 결과를 제시했습니다. XLM-R 같은 사전 학습된 다국어 모델이 모든 모노링구얼 모델들을 능가하는 성능을 보였습니다. 전반적으로 encoder 모델이 sequence-to-sequence 모델보다 NER 태스크에서 더 나은 성능을 나타냈습니다. 또한, 수동 및 기계적 접근법을 결합한 어노테이션 방식을 통해 성능을 높였습니다.



### Enhancing Automated Audio Captioning via Large Language Models with Optimized Audio Encoding (https://arxiv.org/abs/2406.13275)
Comments:
          Accepted by Interspeech 2024

- **What's New**: 이 연구에서는 Automated Audio Captioning (AAC)을 개선하기 위해 세 가지 새로운 접근 방식을 제안합니다. 첫째, 일관성 있는 앙상블 증류 (CED)로 사전 학습된 오디오 인코더를 사용하여 음향 토큰의 효과를 향상시킵니다. 둘째, 7B 매개변수를 가진 Llama 2를 텍스트 디코더로 사용하여 더 강력한 언어 모델을 탐사합니다. 셋째, 다른 사전 학습된 LLM을 사용하여 불충분한 학습 데이터와 주석 모호성으로 인한 텍스트 오류를 수정합니다.

- **Technical Details**: 1) 오디오 인코딩: CED 인코더와 로우랭크 어댑테이션 (LoRA)을 결합하여 성능을 개선하고 출력되는 음향 토큰 수를 줄입니다. 입력 길이에 상관없이 CED의 향상된 위치 임베딩을 사용하고, 정형화된 프레임 레이트 쿼리 트랜스포머 (Q-Former)를 사용해 음향 토큰을 압축합니다. 2) 텍스트 디코딩: Llama 2는 7B 매개변수를 가지고 있으며, LoRA 및 지시 프롬프트로 미세 조정되어 AAC 작업을 위한 텍스트 설명 능력을 강화합니다. 3) 오류 수정: 불충분한 학습 데이터와 주석 모호성을 극복하기 위해, ChatGPT-3.5 API를 사용하여 사전 학습된 LLM이 텍스트 오류를 수정합니다.

- **Performance Highlights**: 새로운 접근 방식은 DCASE 2023 Task 6A의 우승자를 능가하는 성능을 발휘하며, 주요 성과지표인 SPIDEr-FL 점수에서 33.0점을 기록하였습니다.



### Investigating Low-Cost LLM Annotation for~Spoken Dialogue Understanding Datasets (https://arxiv.org/abs/2406.13269)
- **What's New**: 최신 연구는 음성 기반 작업 지향 대화(Spoken Task-Oriented Dialogue, TOD) 시스템에서 사용자의 요청을 기술하는 의미 표현(sequential representation) 선택이 원활한 상호작용에 필수적임을 강조하고 있습니다. 본 논문은 음성 대화 데이터셋의 의미 표현을 자동으로 향상하는 방법에 대한 통찰을 제공합니다.

- **Technical Details**: 본 연구의 기여는 세 가지로 요약됩니다. 첫째, 대형 언어 모델(Large Language Model) 미세 조정(fine-tuning)의 관련성을 평가합니다. 둘째, 생성된 주석이 포함하고 있는 지식을 평가합니다. 셋째, 반자동 주석(annotation)의 효과를 강조합니다. 이는 시스템이 데이터베이스 및 도메인 지식을 통해 차후 작업을 선택하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 텍스트 데이터셋은 세밀한 의미 표현을 제공하는 반면, 기존의 음성 대화 데이터셋은 이에 비해 부족한 경우가 많습니다. 본 연구에서는 이러한 격차를 줄이기 위한 방법을 제시하고, 향상된 음성 대화 데이터셋의 성능을 확인했습니다.



### LangTopo: Aligning Language Descriptions of Graphs with Tokenized Topological Modeling (https://arxiv.org/abs/2406.13250)
- **What's New**: 이번 연구에서는 그래프 머신러닝 분야에서 대형언어모델(LLMs)의 자연어 이해 능력과 학습 능력을 효과적으로 그래프 구조 모델링과 일치시키는 새로운 프레임워크인 LangTopo를 소개합니다. LangTopo는 GNNs(Graph Neural Networks)와 LLMs의 그래프 구조 모델링 능력을 정량화하고 일관성 최대화를 통해 GNNs의 그래프 구조 캡처 능력을 LLMs에 학습시킵니다. 이를 통해 LLMs가 독립적으로 그래프 구조 데이터를 처리할 수 있게 합니다.

- **Technical Details**: LangTopo는 그래프 모달리티에 대한 코드북(codebook)을 구축하고 텍스트 기술을 통해 그래프의 토폴로지 모델링을 일치시킵니다. 이를 위해 Gumbel-softmax를 활용하여 이산적인 분석을 연속적이고 미분 가능한 형태로 변환합니다. 또한, Vector Quantised-Variational AutoEncoder(VQ-VAE)를 사용하여 LLMs와 GNNs의 그래프 토폴로지 구조 모델링 능력을 정량화하고 감독 학습을 통해 LLMs가 GNNs의 모델링 능력을 학습하도록 합니다.

- **Performance Highlights**: 제안된 LangTopo 프레임워크는 여러 데이터셋에서 우수한 성능을 입증했으며, LLMs가 외부 모델에 의존하지 않고 그래프 구조를 처리할 수 있는 능력을 부여합니다. 실험 결과는 LangTopo가 그래프 분석 작업에서 기존의 방법들을 능가하는 성능을 보여준다는 것을 확인했습니다.



### MC-MKE: A Fine-Grained Multimodal Knowledge Editing Benchmark Emphasizing Modality Consistency (https://arxiv.org/abs/2406.13219)
- **What's New**: 이 논문은 'MC-MKE'라 불리는 새로운 벤치마크를 소개하며, 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 모달리티 일관성을 강조합니다. 이전 벤치마크들은 멀티모달 지식을 체계적으로 분석하지 않아, 오독(misreading)과 오인(misrecognition) 오류를 제대로 정정하지 못했습니다. MC-MKE는 시각적 및 텍스트적 구성요소로 멀티모달 지식을 분해하여 이러한 오류를 독립적으로 수정할 수 있게 합니다.

- **Technical Details**: MC-MKE는 세 가지 다른 포맷의 멀티모달 지식 구성 요소를 벤치마킹합니다. 멀티모달 지식은 시각적 지식(이미지와 인식된 개체)과 텍스트적 지식(주어, 관계, 객체)으로 나뉘어집니다. 이 방식을 통해 오독오류와 오인오류를 독립적으로 구별하고 수정할 수 있습니다. 추가로, 멀티모달 지식 편집 후에도 각 모달리티 간의 일관성을 유지하는 것이 중요합니다.

- **Performance Highlights**: 세 가지 멀티모달 지식 편집 방법(파인튜닝, MEND, IKE)을 MC-MKE에서 평가한 결과, 이들 방법은 모두 모달리티 일관성을 유지하는 데 한계가 있음을 드러냈습니다. 특히, 모든 편집 형식에서 탁월한 성능을 보이지 못했고, 멀티모달 지식 편집이 여전히 어려운 과제임을 밝혔습니다.



### PRESTO: Progressive Pretraining Enhances Synthetic Chemistry Outcomes (https://arxiv.org/abs/2406.13193)
- **What's New**: 최근 다양한 과학 분야에서 다중모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)이 크게 발전하고 있습니다. 본 연구는 합성 화학 합성 분야에서 분자-텍스트 모델링을 탐구하며, PRESTO(Progressive Pretraining Enhances Synthetic Chemistry Outcomes)라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 여러 분자 그래프 간 상호작용을 이해하는 데 중점을 두어 합성 화학 작업의 성능을 개선합니다.

- **Technical Details**: PRESTO는 다중모달 LLMs의 성능을 점진적으로 향상시키기 위해 크로스모달 정렬(cross-modal alignment) 및 다중 그래프 이해를 통합합니다. 이를 통해 다중 2D 분자 그래프 처리가 가능해지며, 화학 반응 조건 예측과 같은 다운스트림 과제에서 뛰어난 성능을 보입니다. 학습 전략과 데이터셋 설정을 엄격히 분석하여 최적의 사전 학습 방법을 도출하고, 약 300만개의 샘플을 포함한 데이터셋을 구축하여 사전 학습을 지원합니다.

- **Performance Highlights**: 광범위한 실험을 통해 PRESTO가 합성 화학 관련 다운스트림 작업에서 경쟁력 있는 성능을 제공한다는 것이 입증되었습니다. 특히, 화학 반응 조건 예측 정확도를 현저히 향상시키고, 도메인 지식을 효과적으로 활용하여 분자-텍스트 입력을 이해하는 능력을 심화시킵니다. 이로 인해 다양한 합성 화학 작업에서 우수한 성과를 보입니다.



### Biomedical Visual Instruction Tuning with Clinician Preference Alignmen (https://arxiv.org/abs/2406.13173)
- **What's New**: 최근 멀티모달 기초 모델 분야에서 큰 진전을 이루어 시각적 및 텍스트 정보 이해와 추론에서 놀라운 성능을 보여주고 있습니다. 그러나 이러한 모델을 생물의학과 같은 특수한 도메인에 적용하려면 대규모의 도메인 특화된 지시 데이터셋(instructions datasets)이 필요합니다. 본 연구는 임상의의 선호도를 반영한 생의학 시각 지시 조정(BioMed-VITAL) 프레임워크를 제안하며, 이는 지시 데이터 생성을 위해 임상의가 선택한 예시를 반영하고 모델의 선호도를 명시적으로 학습시켜 고품질 데이터를 선별합니다.

- **Technical Details**: BioMed-VITAL은 세 가지 단계로 구성됩니다. 첫째, 다양한 임상의가 선택한 예시를 사용하여 GPT-4V 생성기를 통해 임상 관련 지시 데이터를 생성합니다. 둘째, 임상의와 정책에 기반한 모델의 선호도를 반영하여 별도의 선택 모델을 훈련시키고, 이를 통해 생성된 데이터 샘플을 평가하여 고품질 데이터를 선별합니다. 마지막으로 선별된 데이터로 모델을 튜닝하여 생의학 멀티모달 모델을 최적화합니다.

- **Performance Highlights**: 제안된 방법을 통해 조정된 모델은 공개 시각 채팅(open visual chat)에서 18.5%의 상대적인 성능 향상을 보였으며, 의학적 VQA에서는 최대 81.73%의 승률을 기록했습니다. 이는 데이터 품질의 향상과 함께 모델의 성능 향상을 의미합니다. 이와 관련된 데이터셋과 모델은 모두 공개되어 있으며, 연구자는 https://BioMed-VITAL.github.io에서 접근할 수 있습니다.



### Amphista: Accelerate LLM Inference with Bi-directional Multiple Drafting Heads in a Non-autoregressive Sty (https://arxiv.org/abs/2406.13170)
- **What's New**: 이번 연구에서는 Amphista라는 추론 가속 알고리즘을 제안합니다. 이는 비자기회귀(non-autoregressive) 디코딩 패러다임을 따르며, 기존의 자기회귀(autoregressive) 디코딩 방법에 비해 높은 병렬 처리 능력을 갖추었습니다. 또한, 이 알고리즘은 Auto-embedding Block과 Staged Adaptation Layers를 통합하여 성능을 극대화합니다.

- **Technical Details**: Amphista는 Auto-embedding Block을 통해 각 draft head 간의 양방향 주의(attention) 메커니즘을 구현하며, 이는 병렬 추론 능력을 강화합니다. 또한, Staged Adaptation Layers는 base 모형에서 비자기회귀 모형으로의 토큰 예측 패러다임 전환을 도와주며, 정보 통합을 촉진합니다. 이는 특히 대형 모델에서 더 큰 성과를 보여줍니다.

- **Performance Highlights**: 실험 결과, Amphista는 Vicuna 33B 모델에서 vanilla autoregressive 디코딩과 Medusa와 비교하여 각각 최대 2.75배, 1.40배의 속도 향상을 이루었습니다. 또한, 생성 품질은 손실 없이 유지되었습니다.



### LLMatDesign: Autonomous Materials Discovery with Large Language Models (https://arxiv.org/abs/2406.13163)
- **What's New**: 신규 소재를 발견하는 문제는 방대한 화학 공간으로 인해 여전히 도전적입니다. 최근 머신 러닝의 발전으로 promising materials(유망한 소재)를 신속하게 스크린하거나 생성하는 데이터 중심의 방법이 가능해졌지만, 여전히 매우 많은 양의 학습 데이터에 크게 의존하는 경우가 많습니다. 이를 해결하기 위해, LLMatDesign이라는 새로운 언어 기반 프레임워크가 도입되었습니다. 이 프레임워크는 Large Language Models(대규모 언어 모델, LLM)를 사용하여 인간의 지시를 해석하고, 소재에 수정을 가하며, 주어진 도구로 결과를 평가합니다. 또한, LLMatDesign은 자신의 이전 결정을 반성하면서 새로운 작업 및 조건에 빠르게 적응할 수 있습니다.

- **Technical Details**: LLMatDesign은 LLM 에이전트를 사용하여 인간의 지시를 번역하고, 소재에 수정을 가하며, 결과를 평가할 수 있는 언어 기반 프레임워크입니다. 또한, LLMatDesign은 자연 언어를 직접 활용하여 다양한 작업, 소재 및 목표 속성에 빠르게 적응할 수 있습니다. 각 단계에서 LLMatDesign은 시작 소재를 수정하고 그 수정에 대한 가설을 세우고 이를 검증합니다. 이 과정에서 우리는 DFT(밀도 함수 이론)를 대체하는 대리 모델을 사용하며, 이는 다른 계산적 또는 실험적 검증 방법으로 쉽게 대체될 수 있습니다. 이러한 과정은 반복적으로 수행되며, 요청된 목표 속성에 도달할 때까지 진행됩니다.

- **Performance Highlights**: LLMatDesign의 효과를 평가하기 위해 Materials Project 데이터베이스에서 무작위로 선택된 10개의 시작 소재를 대상으로 실험을 수행했습니다. 구체적으로 두 가지 물질 속성(밴드 갭과 원자당 형성 에너지)을 대상으로 하여 목표 속성 값을 갖는 새로운 소재를 설계했습니다. 예를 들어, 1.4 eV의 밴드 갭을 갖는 이상적인 광전 변환 소재를 설계하는 것이 목표였습니다. 실험 결과, LLMatDesign은 매우 제한된 데이터 상태에서도 사용자 정의 목표 속성을 가진 새로운 소재를 효과적으로 개발할 수 있음을 확인했습니다.



### APPL: A Prompt Programming Language for Harmonious Integration of Programs and Large Language Model Prompts (https://arxiv.org/abs/2406.13161)
- **What's New**: 새로운 Large Language Models(LLMs)와 관련된 연구는 LLM를 사용하여 다양한 작업을 처리하는 방법을 개선하였습니다. 특히, 복잡한 작업을 다룰 때 LLM의 워크플로우를 간소화하는 APPL이라는 새로운 프롬프트 프로그래밍 언어(Prompt Programming Language)를 제안했습니다. APPL은 Python 함수에 프롬프트(prompt)를 쉽게 삽입할 수 있게 하고, 그 반대도 가능하게 합니다.

- **Technical Details**: APPL은 직관적이고 Python-native 구문(syntax)을 제공하며, 효율적인 병렬 처리 가능한 런타임(runtime)과 비동기(Asynchronous) 의미론을 갖추고 있습니다. 또한, 효과적인 실패 진단과 재실행 기능을 지원하는 추적 모듈(tracing module)을 포함하고 있어 추가 비용 없이 활용할 수 있습니다. CoT-SC(Chain-of-Thought with self-consistency), ReAct 도구 사용 에이전트, 멀티 에이전트 채팅 등 세 가지 대표 시나리오를 통해 APPL의 직관성, 간결성, 효율성을 입증했습니다.

- **Performance Highlights**: 세 가지 병렬 가능한 워크플로우에서 APPL을 테스트한 결과, 독립적인 LLM 호출을 효과적으로 병렬 처리할 수 있으며, 예상되는 속도 증가 비율과 거의 일치하는 유의미한 속도 증가를 보여주었습니다.



### Accelerating Complex Disease Treatment through Network Medicine and GenAI: A Case Study on Drug Repurposing for Breast Cancer (https://arxiv.org/abs/2406.13106)
Comments:
          9 pages double columns, 5 figures, 3 algorithms, 3 tables, and 1 listing, Submitted to IEEE MedAI'24 Conference, to be held November 15-17, Chongqing, China

- **What's New**: 이번 연구는 임상 시험 및 생의학 문헌과 같은 실세계 증거(real-world evidence)원을 조사하여 재목적화될 수 있는 약물을 예측하는 네트워크를 소개합니다. 특히 복잡한 질병(예: 암, 알츠하이머)에 대한 약물 조합 치료법을 생성합니다. 이를 위해, 고도로 구성된 ChatGPT 프롬프트 엔지니어링 시스템을 활용한 다층 네트워크 의료 접근 방식을 제안합니다.

- **Technical Details**: 이 접근 방식은 임상 시험에서 약물 언급을 추출하기 위해 즉석에서 구성된 ChatGPT 프롬프트 엔지니어링 시스템을 사용합니다. 또한, 실세계 증거와 질병별 신호 경로(예: KEGG 데이터베이스)를 연결하는 새로운 알고리즘을 소개합니다. 이 연구는 유방암의 경우를 예로 들어 제안된 프레임워크를 구현했으며, KEGG 경로 ID hsa:2064가 108개의 약물에 의해 커버되는 등 다양한 약물 조합의 가능성을 제시합니다.

- **Performance Highlights**: 제안된 네트워크 의료 프레임워크는 유망한 약물 조합을 식별하는 데 높은 특이성을 보여주며, 실제 임상 시험에서 다수의 약물을 정확히 언급할 수 있습니다. 유방암 신호 경로 46개 중 38개의 경로가 적어도 두 개 이상의 약물에 의해 커버됨을 확인했습니다. ChatGPT를 활용한 프롬프트 엔지니어링 시스템은 임상 시험에서 약물 언급을 가속화하는 데 성공했습니다.



### Articulatory Encodec: Vocal Tract Kinematics as a Codec for Speech (https://arxiv.org/abs/2406.12998)
- **What's New**: 이번 논문에서는 음성 생성의 새로운 프레임워크인 '조음 인코데크(articulatory encodec)'를 제안합니다. 이 시스템은 음성 데이터를 기반으로 조음 특성을 추론하고, 이를 통해 고품질의 음성을 재생성할 수 있는 모델을 포함합니다. 특히, 화자 식별 인코더를 추가로 훈련해 개별 화자의 목소리 텍스처를 반영하며, 방대한 음성 데이터를 통해 학습된 이 모델은 보지 못한 화자에 대해서도 높은 품질의 음성을 생성할 수 있습니다.

- **Technical Details**: 조음 인코데크는 두 가지 주요 모델로 구성됩니다: 조음 분석 모델과 조음 합성 모델. 조음 분석 모델은 음성 오디오로부터 조음 특성을 추론하고, 조음 합성 모델은 이러한 특성으로부터 음성을 재생성합니다. 조음 특성은 성도(聲道) 조음기와 출처 특성의 운동학적 궤적을 포함하며, 이는 직관적으로 해석 가능하고 제어 가능한 실제 물리적 인터페이스입니다. 추가적으로, 화자 식별 인코더가 합성 모델과 함께 훈련되어, 제로샷(zero-shot) 음성 변환을 가능하게 합니다.

- **Performance Highlights**: 제안된 조음 인코데크는 고성능, 범용 조음 인코더와 디코더를 시연합니다. 이 모델은 새로운 화자들에게 일반화될 수 있으며, 높은 품질의 음성을 생성합니다. 또한, 조음 특성을 통해 저차원적이며 해석 가능하고 제어 가능한 음성 코딩을 제공합니다. 제로샷 화자의 방언을 일정하게 유지하면서 음성 변환을 할 수 있는 능력도 입증되었습니다.



### MolecularGPT: Open Large Language Model (LLM) for Few-Shot Molecular Property Prediction (https://arxiv.org/abs/2406.12950)
- **What's New**: 이 논문에서는 MolecularGPT라는 새로운 모델을 제안합니다. 이 모델은 소수의 레이블된 분자 데이터만을 사용하여 분자의 속성을 예측하는 few-shot 학습 기능을 갖추고 있습니다. 기존의 모델들은 대규모의 레이블된 데이터가 필요하고 새로운 작업에 적응하는 데 제한이 있었지만, MolecularGPT는 이러한 문제들을 해결합니다.

- **Technical Details**: MolecularGPT는 SMILES 표현을 사용해 분자의 화학 구조를 문자열로 변환하고, 이를 활용해 실험적 데이터와 결합된 그래프 구조 기반의 few-shot 지시를 만듭니다. 총 1000개 이상의 속성 예측 작업을 포함하는 다양화된 지시 세트를 사용해 모델을 훈련시키며, zero-shot 및 few-shot 능력을 동시에 강화합니다.

- **Performance Highlights**: MolecularGPT는 10개의 평가 데이터셋에서 경쟁력 있는 성능을 보이며, 특히 4개의 데이터셋에서 표준 그래프 신경망(GNN) 방법보다 뛰어난 성과를 보였습니다. 분류 정확도에서는 최대 16.6% 향상, 회귀 지표(RMSE)에서는 199.17의 감소를 기록했습니다. 두 샷(two-shot) 예제로만도 GNN을 능가하는 성과를 보였습니다.



### Instruction Data Generation and Unsupervised Adaptation for Speech Language Models (https://arxiv.org/abs/2406.12946)
Comments:
          Accepted for Interspeech 2024

- **What's New**: 이번 논문에서는 텍스트와 음성 입력을 처리할 수 있는 멀티모달 대규모 언어 모델(multimodal large language models)을 훈련 및 평가하기 위해 합성 샘플(synthetic samples)을 생성하는 세 가지 방법을 제안합니다. 이러한 샘플의 부족 문제를 해결하고 시스템 성능을 향상시키기 위해 합성 데이터 생성을 중요한 전략으로 강조하고 있습니다.

- **Technical Details**: 본 연구에서는 대규모 언어 모델(large language models)을 사용하여 텍스트 구성 요소를 생성하고, 텍스트-음성 변환 시스템(text-to-speech systems)을 활용하여 음성 구성 요소를 생성하는 과정을 포함합니다. 이를 통해, 텍스트와 음성 영역 간의 크로스 모달 관계(cross-modal relationships)을 모델링하는데 실질적이고 효과적인 수단을 제공합니다.

- **Performance Highlights**: 실험 결과, 텍스트와 음성을 통합 이해하는데 성과를 나타냈습니다. 또한, 주석이 없는 음성 데이터를 사용하여 유사한 품질의 합성 샘플을 생성할 가능성을 강조하며, 이를 통해 더 많은 언어로 모델을 확장할 수 있는 잠재력을 보여줍니다.



### Self-Train Before You Transcrib (https://arxiv.org/abs/2406.12937)
Comments:
          Accepted at Interspeech 2024

- **What's New**: 최근 발표된 연구에 따르면 도메인 간 불일치가 발생할 때 음성 인식 시스템의 성능 저하를 해결하기 위해 자체 훈련(self-training) 방법이 사용될 수 있습니다. 특히 테스트 세트의 녹음을 사용하여 테스트 시점 적응(test-time adaptation, TTA)을 수행하는 'Noisy Student Teacher at Inference'(NSTI) 방법을 제안합니다. 이는 별도의 적응 데이터 없이 즉각적인 도메인 적응을 가능하게 하며, 데이터 수집 비용이나 개인정보 문제를 회피할 수 있습니다.

- **Technical Details**: 이 연구에서는 CTC 기반 음향 모델을 사용하여 NSTI 기법을 구현했습니다. NSTI는 소음 및 변형(augmentation) 전략을 포함한 두 개의 입력 스펙트로그램을 사용하여 동일한 매개변수를 공유하는 두 모델(학생 및 교사 모델)로 구성됩니다. 변환 함수는 주로 SpecAugment 방식이 사용되었습니다. 녹음은 세그먼트로 분할되고, 각 세그먼트에 대해 랜덤 순서로 n개의 에포크 동안 자체 훈련을 수행합니다. 최종 예측 결과는 녹음의 전체 세그먼트에 대해 얻어집니다.

- **Performance Highlights**: NSTI 방법은 기존 자체 훈련 방법보다 최대 32.2%까지 성능이 향상되었습니다. 특히 도메인 불일치가 클 때 확대 및 변형 전략이 중요한 역할을 하며, NSTI는 별도의 적응 데이터 세트를 사용하는 전통적인 방법보다 100배 적은 데이터로 더 나은 성능을 보였습니다. 또한, 긴 녹음일수록 더 나은 성능 향상이 나타났습니다.



### Automatic Speech Recognition for Biomedical Data in Bengali Languag (https://arxiv.org/abs/2406.12931)
- **What's New**: 이번 연구는 벵골어 생체의료 데이터를 위해 특별히 고안된 자동 음성 인식 시스템(Automatic Speech Recognition, ASR) 프로토타입을 소개합니다. 벵골어 ASR에 대한 최근 발전에도 불구하고, 특정 도메인 데이터 부족으로 실용적인 헬스케어 ASR 모델 생성이 제한되었습니다. 이 프로젝트는 증상, 심각도 수준, 질병 등 벵골어 의학 용어에 맞춘 ASR 시스템을 개발함으로써 이 격차를 해소합니다. 두 가지 인기 있는 ASR 프레임워크를 광범위한 46시간 분량의 벵골어 의료 코퍼스에서 교육하고 평가합니다.

- **Technical Details**: 이번 연구에서는 두 가지 ASR 프레임워크인 DeepSpeech2와 세밀하게 튜닝된 Whisper BanglaASR를 평가했습니다. 이 모델들은 질병 이름, 증상 및 증상 심각도를 포함하는 57.59시간 분량의 벵골어 의료 코퍼스에서 교육되었습니다. 데이터 수집 과정에서는 다양한 의료 증상 오디오 데이터를 모아 데이터셋을 조성했습니다. 모집단계에서는 실험참여자 모집 과정과 Google TTS (Text-to-Speech)를 활용한 합성 데이터, 실제 환자-의사 간 대화를 바탕으로 한 시뮬레이션 의료 시나리오가 포함됩니다. 데이터 전처리 과정에서는 오디오 파일을 16kHz 단일 채널 WAV 포맷으로 변환하고, 텍스트 데이터 정규화 작업을 수행했습니다.

- **Performance Highlights**: 개발된 두 모델은 의료 대화를 전사할 때 단어 오류율(WER)에서 유의미한 개선을 이루었습니다. 평가된 5.8시간의 테스트 데이터셋에서 각각 17.25%와 9.05%의 WER를 기록했습니다. Whisper BanglaASR 모델은 AmarDoctor 플랫폼에서 현재 배치되어 있으며, 모델 테스트 데이터에 대한 상세 정보는 http://amardoctor.health/banglamedasr.html에서 확인할 수 있습니다.



### Evaluating the Generalization Ability of Quantized LLMs: Benchmark, Analysis, and Toolbox (https://arxiv.org/abs/2406.12928)
- **What's New**: 이번 연구는 대규모 언어 모델(Large Language Models, LLMs)에 대한 양자화(Quantization)의 영향을 체계적으로 평가하는 새로운 벤치마크 세트를 제공한다. 특히, 양자화가 모델의 일반화 능력에 미치는 영향을 집중적으로 분석했다. 이 벤치마크는 평가 시스템, 세부 분석, 그리고 일반적인 도구 상자를 포함하며, 두 가지 주요 시나리오를 기반으로 40개 이상의 데이터셋을 사용하여 실험을 수행했다.

- **Technical Details**: 이번 연구는 기존의 양자화 방법이 주로 사용하는 '후-훈련 양자화(PTQ, Post-Training Quantization)' 기법을 채택했다. PTQ는 높은 정밀도의 부동 소수점 숫자를 낮은 정밀도의 정수로 대체하여 모델 크기를 줄인다. 양자화 과정에서 보정 데이터(calibration data)가 테스트 데이터와 같은 분포를 공유하지 않을 때도 최적의 결과를 얻을 수 없다는 사실을 발견했다. 벤치마크는 다양한 데이터셋과 분포 전환 간의 관계를 조사하여 I.I.D(독립 동일 분포) 및 OOD(분포 외) 평가를 구축했다.

- **Performance Highlights**: 실험 결과, 양자화된 모델의 성능은 작업(task) 및 데이터셋에 따라 크게 다르며, 같은 작업일지라도 데이터셋에 따라 민감도가 다르다는 것을 발견했다. 예를 들어, 자연어 추론 작업은 다양한 작업 중 민감도가 가장 낮았다. 또한 낮은 비트 폭의 양자화가 일부 설정에서는 성능을 향상시키기도 했다. 보정 데이터와 테스트 데이터의 분포 일치도가 항상 최적의 성능을 나타내지는 않으며, 이는 평가 작업과 분포 전환의 정도에 따라 달라진다.



### GLiNER multi-task: Generalist Lightweight Model for Various Information Extraction Tasks (https://arxiv.org/abs/2406.12925)
Comments:
          11 pages, 1 figure, 6 tables

- **What's New**: 새로운 형태의 GLiNER 모델을 소개합니다. 이 모델은 정보 추출에 필요한 다양한 작업을 수행할 수 있으며, 작은 인코더 모델로 구축되었습니다. 기존의 대규모 언어 모델(LLMs)보다 효율적이며 구조화된 출력을 생성할 수 있습니다. 이 모델은 제로샷(named entity recognition) 벤치마크에서 SoTA(state of the art) 성능을 기록했으며, 질문 응답, 요약 및 관계 추출 작업에서도 우수한 성능을 보였습니다.

- **Technical Details**: GLiNER 모델은 GLiNER 토큰 분류 아키텍처를 기반으로 하며, 토큰을 분류하는 대신 시퀀스를 추출할 수 있게 합니다. 이를 통해 긴 엔티티 추출, 요약, 텍스트 정리 작업에서 유리합니다. BERT 같은 인코더 아키텍처 위에 구축되었으며, 이번 연구에서는 DeBERTA v3 large를 사용했습니다. 이 모델은 효율성을 높이기 위해 RTD(replaced token detection) 방식을 채택했습니다. GLiNER는 레이블과 텍스트를 단일 인코더 모델에서 처리함으로써 상호 정보를 교환할 수 있게 합니다. 또, bidirectional LSTM을 통해 토큰 임베딩을 추가로 처리해 모델 학습을 가속화하고 데이터 부족 상황에서도 좋은 성능을 유지합니다.

- **Performance Highlights**: GLiNER 모델은 제로샷(named entity recognition) 벤치마크에서 SoTA 성능을 기록했습니다. 또한, 질문 응답, 요약 및 관계 추출 작업에서도 선도적인 성능을 보였습니다. 이 모델은 효율성과 정확성 측면에서 기존의 대규모 언어 모델에 비해 더 나은 성능을 보입니다.



### Reconciling Kaplan and Chinchilla Scaling Laws (https://arxiv.org/abs/2406.12907)
- **What's New**: 이 노트는 Kaplan과 Chinchilla 연구의 스케일링 계수 계산에 대한 불일치를 분석합니다. Kaplan의 최적 파라미터 수 $N_{optimal} 	imes C^{0.73}$에서 Chinchilla의 $N_{optimal} 	imes C^{0.50}$까지의 차이는 주요하게 Kaplan의 비몰입 파라미터(non-embedding parameters)만을 계산하고, 소규모에서 분석을 수행한 결과임을 밝혀냅니다. 따라서 Chinchilla의 스케일링 계수를 재확인합니다.

- **Technical Details**: Kaplan은 비몰입 파라미터($N_{E}$)와 비몰입 컴퓨팅($C_{E}$)에 대해 연구하고, Chinchilla는 총 파라미터($N_{T}$)와 전체 컴퓨팅($C_{T}$)을 조사했습니다. 이 노트는 Chinchilla 연구를 Kaplan의 조건 하에서 시뮬레이션하고, 작은 스케일 범위에서 로컬 파워 법칙을 적용하여 두 연구 결과를 비교합니다.

- **Performance Highlights**: Kaplan의 계산 방법론으로 Chinchilla 연구를 시뮬레이션 했을 때, Kaplan의 결과와 가까운 편향된 스케일링 계수를 얻을 수 있었습니다. 이를 통해 Chinchilla 스케일링 계수의 타당성을 재확인했습니다. 이는 큰 모델이 아니라, 적정 크기의 모델을 더 많은 데이터로 훈련시키는 것이 더 유리할 수 있음을 시사합니다.



### Towards Unlocking Insights from Logbooks Using AI (https://arxiv.org/abs/2406.12881)
Comments:
          5 pages, 1 figure, 15th International Particle Accelerator Conference

- **What's New**: 입자 가속기 관련 전자 로그북(eLogs)을 보다 효과적으로 활용하기 위한 RAG(Retrieval Augmented Generation) 모델을 개발했습니다. 이 모델은 DESY, BESSY, Fermilab, BNL, SLAC, LBNL, CERN 등 다양한 연구소의 데이터를 기반으로 하며, 로그북의 정보를 활용해 자동화와 문제 해결을 지원합니다.

- **Technical Details**: RAG 모델은 질의에 대해 관련 문서를 검색하고, 이를 기반으로 언어 생성 모델이 최종 답변을 생성합니다. 이를 통해 언어 모델이 실제 문서에서 얻은 지식을 활용하여 더 정확한 답변을 생성할 수 있습니다. SimCSE를 사용하여 도메인 데이터를 미세 조정하고, re-ranking 기술을 통해 정확도를 높였습니다.

- **Performance Highlights**: 초기 실험에서 특정 용어와의 유사성을 찾아내는 데 성공했으며, 사용자 맞춤형 설정을 지원하는 BNL eLog 시스템과 CERN의 AccGPT와 같은 프로젝트가 진행 중입니다. CERN에서는 2024년 중반에 최초 정확도 테스트를 시행할 계획입니다. DESY는 SINBAD-ARES 시설에서 성공적으로 RAG를 구현하였으며, faiss vectorstore와 Mistral-7B-Instruct-v0.2를 사용해 eLog 항목의 정보를 증가시켰습니다.



### Large Language Models as Software Components: A Taxonomy for LLM-Integrated Applications (https://arxiv.org/abs/2406.10300)
- **What's New**: 최근 연구는 대형 언어 모델(LLMs)의 사용이 소프트웨어 엔지니어링에서 새로운 분야로 부상하고 있음을 보여줍니다. 이 연구는 LLM 통합 응용 프로그램을 분석하고 기술하기 위한 분류 체계를 제공합니다. 이 분류 체계는 LLM을 활용한 응용 프로그램의 다양한 이용 사례를 보여주며, 특히 'LLM 컴포넌트'라는 개념을 도입하여 각 구성 요소를 개별적으로 분석합니다.

- **Technical Details**: 최신 LLMs(GPT-3.5, GPT-4, Llama, PALM2 등)는 간단한 처리 유닛으로 구성된 인공 신경망입니다. LLM은 입력된 텍스트(prompt) 다음에 이어질 확률을 예측하도록 훈련됩니다. Prompt 엔지니어링과 같은 방법론은 LLM의 다양한 기능을 활용하기 위해 발전하고 있습니다. 연구는 LLM 통합 응용 프로그램을 기술하는 것을 목표로 하며, 이를 통해 소프트웨어가 LLM의 프롬프트를 생성하고 출력을 처리하는 인풋-프로세싱-아웃풋 시퀀스를 정의합니다.

- **Performance Highlights**: 연구는 다양한 도메인에서 LLM 통합 응용 프로그램을 샘플링하여, 이들에게 공통적으로 적용할 수 있는 13가지 차원을 식별했습니다. 이 차원들은 LLM 기반 소프트웨어 컴포넌트를 기술하고 분류하는 데 유용합니다. 또한, 이 분류 체계는 새로운 설계 옵션을 제시하고, 실무자들이 LLM을 응용 프로그램에 활용하는 데 영감을 줄 수 있습니다.



### Informatics & dairy industry coalition: AI trends and present challenges (https://arxiv.org/abs/2406.12770)
- **What's New**: 이 논문은 AI가 산업, 특히 유제품 산업에서 어떻게 생산 공정을 향상시키고 수작업 및 반복 작업을 최소화할 수 있는지를 다룹니다. AI와 고성능 컴퓨팅, 그리고 강력한 수학적 모델의 시너지를 통해 정교한 데이터 분석 절차를 적용하는 방법을 제안하고 있습니다.

- **Technical Details**: 이 논문에서는 고성능 계산(computing)과 강력한 수학적 모델을 활용하여 Machine Learning과 같은 정교한 데이터 분석 절차를 적용하는 방법을 설명하고 있습니다. 특히, 유제품 산업에서의 cattle monitoring(가축 모니터링)과 같은 구체적인 산업적 과제들을 어떻게 해결할 수 있는지에 대해 논의합니다.

- **Performance Highlights**: 이 연구는 연구자들이 새로운 접근 방식을 적용하여 유제품 산업의 다양한 과제를 해결할 수 있는 잠재적인 방법들을 제시합니다. 이는 가축 모니터링과 같은 구체적인 기술 솔루션을 통해 농부들의 요구를 충족시킬 수 있는 결과를 도출할 수 있습니다.



### Preserving Knowledge in Large Language Model with Model-Agnostic Self-Decompression (https://arxiv.org/abs/2406.11354)
- **What's New**: 이번 연구에서는 기존의 큰 언어 모델(LLMs) 및 다중 모달 언어 모델(MLLMs)이 도메인 특화 데이터로 미세 조정(SFT)되었을 때, 과거 지식을 잃는 문제(cumbersome forgetting)를 해결하기 위해 '나무 생성(Tree Generation, TG)' 방법을 제안합니다. TG-SFT를 사용하여 교육 코퍼스(corpus)를 합성적으로 생성하고, 이를 MLLMs의 SFT 단계에 통합함으로써 성능 저하 문제를 줄인다.

- **Technical Details**: TG-SFT는 구체적으로 LLM이 생성한 데이터를 저장하고 이를 SFT 중에 사용하여 기존 지식을 유지하는 방법입니다. TG 알고리즘은 모델에 의존하지 않는 데이터 생성 알고리즘으로, 특정 NLP 작업이 아닌 일반적인 LLMs에 적용할 수 있으며, 추가적인 수작업 프롬프트가 필요하지 않습니다.

- **Performance Highlights**: 광범위한 실험을 통해 TG 알고리즘이 과거 지식을 잊는 문제(cumbersome forgetting)를 줄이는 데 유용함을 입증했습니다. TG 알고리즘은 기존 지식 유지, 지식 증류(knowledge distillation), 연속 학습(continual learning) 등 다양한 응용에서 중요한 역할을 할 수 있습니다.



New uploads on arXiv(cs.IR)

### LARP: Language Audio Relational Pre-training for Cold-Start Playlist Continuation (https://arxiv.org/abs/2406.14333)
- **What's New**: 온라인 음악 소비가 플레이리스트(playlist) 중심으로 이동함에 따라, 개인화된 방식으로 플레이리스트를 확장하는 알고리즘인 플레이리스트 연속(palylist continuation)이 중요하게 되었습니다. 이 논문에서는 기존 방법들의 한계를 극복하기 위해 새로운 멀티모달(multi-modal) 콜드스타트(cold-start) 플레이리스트 연속 모델인 LARP(Language Audio Relational Pre-training)를 소개합니다.

- **Technical Details**: LARP는 세 단계의 대조 학습(contrastive learning) 프레임워크를 통해 멀티모달 및 관계적 신호(relational signals)를 통합하여 표현을 학습합니다. 이 프레임워크는 트랙 내부 언어-오디오 대조 손실(within-track language-audio contrastive loss), 트랙-트랙 대조 손실(track-track contrastive loss), 트랙-플레이리스트 대조 손실(track-playlist contrastive loss)로 구성되어 있습니다. 각 단계는 구체적인 과업에 특화된 추상화를 점진적으로 증가시키며 표현을 학습합니다.

- **Performance Highlights**: LARP는 공개된 두 개의 데이터셋을 기반으로 광범위한 실험을 수행하여, 콜드스타트 설정에서 기존의 단일 모달 및 멀티모달 모델을 능가하는 성능을 보였습니다. 특히 멀티모달 정보 소스로 오디오 샘플과 메타데이터 기반 텍스트 주석을 포함하게 하여, 플레이리스트 연속 작업에서 뛰어난 성능을 입증했습니다.



### Optimizing Novelty of Top-k Recommendations using Large Language Models and Reinforcement Learning (https://arxiv.org/abs/2406.14169)
Comments:
          Accepted at KDD 2024

- **What's New**: 새로운 논문에서는 기존에 배포된 모델과 비교하여 상위-k 추천 항목의 참신성을 최적화하는 역량을 강화하려는 접근 방법을 소개합니다. 특히 대형 언어 모델(Large Language Models, LLMs)을 활용해 사용자 피드백 데이터를 얻기 어려운 참신한 항목의 피드백을 강화 학습(Reinforcement Learning, RL) 형식으로 받는 방법을 제안합니다.

- **Technical Details**: 기존의 추천 시스템은 사용자 피드백 데이터를 활용해 모델을 지도 학습(Supervised Learning)으로 훈련하여 상위-k 항목의 랭킹 리스트를 출력합니다. 하지만 상위-k 항목의 참신성을 최적화하는 것은 비분화 가능한 정렬 작업을 포함하므로 어려운 문제입니다. 이 문제를 해결하기 위해, 논문에서는 LLMs가 참신한 항목에 대한 피드백을 제공할 수 있는 RL 방식을 사용합니다. 또한, 표본 복잡성을 줄이기 위해, 상위-k 리스트 보상을 항목별 보상으로 줄이고 상태 공간을 <query, item> 튜플로 재구성하여 행동 공간을 이진 결정으로 감소시키는 방법을 제안합니다.

- **Performance Highlights**: 제안된 알고리즘은 대규모 상용 검색 엔진 및 Amazon 리뷰 기반의 제품 추천 데이터셋에서 실험되었습니다. 제안된 RL 기반 알고리즘은 지도 학습 모델에 비해 상위-k 항목의 참신성을 2배에서 5배 증가시키면서도 최소한의 재현율 감소를 나타냈습니다. ORCAS 쿼리-웹페이지 매칭 데이터셋에서도 유사한 결과를 얻었으며, Amazon 제품 추천에서도 상당한 참신성 향상을 보였습니다.



### DIRAS: Efficient LLM-Assisted Annotation of Document Relevance in Retrieval Augmented Generation (https://arxiv.org/abs/2406.14162)
- **What's New**: 이번 논문에서는 Retrieval Augmented Generation (RAG) 시스템에서 정보 검색(IR) 성능 평가를 위한 새로운 도메인-특정 벤치마크를 자동으로 주석 달 수 있는 DIRAS (Domain-specific Information Retrieval Annotation with Scalability)를 소개합니다. DIRAS는 GPT-4 수준의 성능을 가진 오픈 소스 언어 모델(LLM)을 미세 조정하여 관련성 레이블을 주석 달 수 있게 합니다.

- **Technical Details**: DIRAS는 도메인-특정 데이터에서 (query, document) 쌍을 생성하고, 오픈 소스 LLM을 미세 조정해 관련성을 예측하는 방식으로 작동합니다. 이 시스템은 효율적이고 효과적인 포인트 방식 접근법을 이용하여, 모든 (query, document) 쌍을 평가하고 점수를 예측합니다. 주목할 만한 것은 이러한 접근법이 도메인-전문가 또는 LLM에 의해 설계된 관련성 정의를 활용하여 일관성 있는 주석을 다는 것입니다.

- **Performance Highlights**: DIRAS 미세 조정된 모델은 GPT-4 수준의 성능을 보여주며, 특히 ChatReport와 ClimRetrieve 데이터셋에서 뛰어난 성능을 발휘합니다. 이를 통해 DIRAS는 정보 검색 주석 편향을 줄이고, 모든 (query, document) 쌍에 대해 체계적으로 주석을 달아 RAG 시스템의 실제 성능 평가를 돕습니다.



### An Investigation of Prompt Variations for Zero-shot LLM-based Rankers (https://arxiv.org/abs/2406.14117)
- **What's New**: 새로운 연구는 특정 요소와 단어 선택이 zero-shot Large Language Models (LLMs)를 사용한 랭커(ranker)의 효과성에 미치는 영향을 체계적으로 이해하는 방법을 제시합니다. 다양한 zero-shot 랭킹 방법들이 최근 제안되었으며, 이 연구는 랭킹 알고리즘, 백본 LLM (예: GPT-3.5 vs. FLAN-T5), 그리고 프롬프트의 구성 요소와 단어 선택이 랭킹 성능에 미치는 영향을 분석하고 있습니다.

- **Technical Details**: 연구는 네 가지 주요 zero-shot 랭킹 알고리즘 (pointwise, pairwise, listwise, setwise)을 조사했습니다. 이들은 각기 다른 백본 LLM과 프롬프트 구성 요소, 단어 선택에 따라 차이를 보입니다. Pointwise는 개별 문서의 관련성을 평가하고, Pairwise는 두 문서의 상대적인 관련성을 평가하며, Listwise는 문서 리스트의 순서를 평가하고, Setwise는 문서 집합의 순서를 평가합니다. 실험에서 백본 LLM을 고정하고 프롬프트의 구성 요소를 통제하여 다양한 변형의 효과를 분석했습니다.

- **Performance Highlights**: 연구는 랭킹 알고리즘이 zero-shot LLM 랭킹 방법 간의 성능 차이에 기여하는 것을 확인했지만, 더 중요한 것은 프롬프트 구성 요소와 단어 선택이 랭킹 효과성에 미치는 영향입니다. 실험에서 프롬프트의 구성 요소와 단어 선택이 실제 랭킹 알고리즘보다 더 큰 영향을 미친 경우도 발견되었습니다. 따라서 이러한 요소들을 고려하면 랭킹 방법 간의 차이가 더 모호해질 수 있습니다. 이러한 결과는 미래 연구에서 랭킹 방법을 비교할 때 중요한 지침이 됩니다.



### Taxonomy-Guided Zero-Shot Recommendations with LLMs (https://arxiv.org/abs/2406.14043)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)을 활용한 추천 시스템(RecSys)의 문제점을 해결하기 위해 신규 방법인 TaxRec을 제안합니다. 이 방법은 체계적인 분류 체계 사전을 사용하여 항목 정보를 구조화하고 명확하게 만듭니다. 이를 통해 좀 더 효율적이고 정확한 추천을 가능하게 합니다.

- **Technical Details**: TaxRec은 두 단계로 구성됩니다. 첫 번째 단계는 분류 체계를 통한 항목 분류로, LLM에서 지식을 가져와 항목들을 체계적으로 조직합니다. 두 번째 단계에서는 LLM을 사용해 사용자의 이전 상호작용을 기반으로 추천을 수행함으로써 제로샷(zero-shot) 추천을 가능하게 합니다. 분류 체계 사전은 후보 항목 정보를 축약하여 LLM이 효율적으로 토큰을 활용하고, 비구조적인 제목으로 인한 모호성을 줄입니다.

- **Performance Highlights**: 실험 결과에 따르면, TaxRec은 기존의 제로샷 추천 시스템에 비해 추천 품질을 상당히 향상시키는 것으로 나타났습니다. 이는 TaxRec이 LLM 기반 추천 시스템의 효율성과 정확도를 크게 개선할 수 있음을 입증합니다.



### EAGER: Two-Stream Generative Recommender with Behavior-Semantic Collaboration (https://arxiv.org/abs/2406.14017)
Comments:
          Accepted by KDD 2024. Source code available at this https URL

- **What's New**: Generative retrieval을 활용한 추천 시스템에서 행동 정보와 의미 정보를 통합한 EAGER 프레임워크를 소개합니다. 기존 방법들은 주로 행동 정보와 의미 정보 중 한 가지에만 집중하여 제한된 성능을 보였습니다. EAGER는 두 가지 정보 타입을 효과적으로 통합하여 개선된 추천 성능을 달성합니다.

- **Technical Details**: EAGER는 행동 토큰과 의미 토큰을 별도로 디코딩하는 두 개의 디코더와 공유 인코더를 사용하는 이중 스트림 생성 아키텍처를 채택합니다. 행동 정보와 의미 정보의 독립적인 학습을 위해 Summary Token을 사용하는 글로벌 대조 작업(global contrastive task)을 도입하고, 재구성과 인식을 목표로 하는 의미 기반 전이 작업(semantic-guided transfer task)을 설계했습니다.

- **Performance Highlights**: EAGER는 네 가지 공개 벤치마크에서 기존 메서드보다 우수한 성능을 보여, 행태 정보와 의미 정보를 모두 활용한 추천 시스템의 가능성을 입증했습니다.



### Do Not Wait: Learning Re-Ranking Model Without User Feedback At Serving Time in E-Commerc (https://arxiv.org/abs/2406.14004)
- **What's New**: 이 논문에서는 전자 상거래에서 사용되는 추천 시스템의 향상을 위해 LAST(Learning At Serving Time)를 제안합니다. 기존의 온라인 학습 방법은 실제 사용자 피드백에 의존하여 모델을 업데이트해야 하지만, 피드백의 지연으로 인해 모델의 개선이 늦어질 수 있습니다. 이를 해결하기 위해 LAST는 대신 대리 모델(surrogate model)을 사용하여 실시간으로 요청별 맞춤형 모델 수정을 적용합니다. 이 수정은 요청이 끝난 후 버려지며, 기존 모델에 남아 있지 않습니다.

- **Technical Details**: LAST는 요청을 받고 추천 결과를 생성하기 전에 실시간으로 모델 수정을 적용합니다. 이 수정은 요청에 특화되어 있으며 일시적입니다. 또한, 대리 모델의 예측이 부정확할 경우 오류 전파를 방지하고 온라인 학습 절차의 안정성을 유지합니다. 중요한 점은 LAST가 기존의 피드백 기반 온라인 학습 방법과 무리 없이 통합될 수 있다는 점입니다.

- **Performance Highlights**: 광범위한 실험, 특히 오프라인 및 온라인 실험 모두에서 LAST가 최신의 재랭킹(re-ranking) 모델보다 뛰어나다는 것을 확인했습니다. 이로써 LAST는 사용자 만족도와 시스템 성능을 높이는 데 효과적임을 보여줍니다.



### Unifying Graph Convolution and Contrastive Learning in Collaborative Filtering (https://arxiv.org/abs/2406.13996)
- **What's New**: 본 논문에서는 협업 필터링(Collaborative Filtering, CF)에 그래프 기반 모델과 대조 학습(contrastive learning)을 이론적 프레임워크를 통해 연결하는 연구를 소개합니다. 저자들은 대조 손실(contrastive loss)의 학습 동태와 평형(equilibrium)을 분석함으로써, 대조 학습이 그래프 이론을 통해 고차 연결성(High-Order Connectivity, HOC)을 효과적으로 모델링할 수 있음을 강조합니다. 이를 기반으로 Simple Contrastive Collaborative Filtering(SCCF)이라는 간단하면서도 효과적인 알고리즘을 소개합니다.

- **Technical Details**: 저자들은 대조 손실을 두 가지 그래프 컨볼루션(graph convolution) 과정으로 분해하여 이를 통해 얻어지는 embedding 업데이트(dynamic embedding)를 설명합니다. 하나는 Positive 샘플에 의해 embedding을 그래프 전반에 걸쳐 단순화(smoothing)하는 것이고, 또 다른 하나는 Negative 샘플에 의해 이 단순화 과정을 억제하여 embedding의 붕괴를 방지하는 것입니다. 이러한 두 갈래의 그래프 컨볼루션 프로세스를 통합하여, 고차 연결성을 모델링함에 있어 그래프 컨볼루션 층이 필수가 아님을 논증합니다.

- **Performance Highlights**: SCCF는 간단한 임베딩 모델과 수정된 대조 손실 함수만으로 구성되어 있으며, 어떤 컨볼루션 층도 사용하지 않습니다. 실험 결과, SCCF가 여러 공개 데이터셋에서 최첨단 성능을 달성하거나 이를 초과하도록 입증되었습니다. 이 알고리즘의 효능은 광범위한 실험을 통해 확인되었습니다.



### UpDLRM: Accelerating Personalized Recommendation using Real-World PIM Architectur (https://arxiv.org/abs/2406.13941)
- **What's New**: 새로운 연구는 Deep Learning Recommendation Models(DLRMs)의 메모리 대역폭 문제를 해결하기 위해 UpDLRM을 제안합니다. UpDLRM은 실세계의 Processing-in-Memory(PIM) 하드웨어인 UPMEM DPU를 활용하여 메모리 대역폭을 증가시키고 추천 지연 시간을 줄이는 것을 목표로 합니다.

- **Technical Details**: DLRMs는 큰 규모의 임베딩 테이블(emembedding table, EMT)을 가지며 이러한 EMT의 빈번하고 불규칙한 메모리 접근은 시스템 성능의 병목을 초래합니다. UpDLRM은 DPU 메모리에 EMT를 저장하고 DPU를 활용하여 다중 임베딩 조회와 축소 작업을 병렬로 처리함으로써 CPU 메모리 대역폭의 경쟁을 줄이고 추론 시간을 가속화합니다. DPU의 메모리 용량이 한정적이기 때문에 EMT를 여러 DPU에 효과적으로 분할하는 방법을 연구하였습니다.

- **Performance Highlights**: 실제 데이터셋과 다양한 하드웨어 아키텍처를 사용한 평가에서 UpDLRM은 기존의 CPU 전용 및 CPU-GPU 하이브리드 시스템과 비교하여 추론 시간을 최대 4.6배까지 단축하는 데 성공했습니다.



### CLIP-Branches: Interactive Fine-Tuning for Text-Image Retrieva (https://arxiv.org/abs/2406.13322)
- **What's New**: CLIP-Branches라는 새로운 텍스트-이미지 검색 엔진이 소개되었습니다. 이 엔진은 CLIP 아키텍처를 기반으로 하며, 사용자가 긍정적 및 부정적 예제를 통해 검색 쿼리를 구체화할 수 있도록 상호작용적 미세 조정 단계를 도입합니다. 이는 전통적인 텍스트-이미지 검색 엔진에 비해 검색 정확도와 관련성을 높이는 데 기여합니다.

- **Technical Details**: CLIP-Branches는 CLIP 모형을 기반으로 한 텍스트-이미지 검색 엔진입니다. 사용자 피드백을 받아들여 긍정적 및 부정적 예제를 통해 검색 쿼리를 구체화하고, 의사 결정 트리(Decision Branches)와 사전 구축된 인덱스 구조를 활용하여 신속한 응답 시간을 제공합니다. 검색-분류(Search-by-Classification) 방식을 도입하여 전체 데이터베이스를 대상으로 긍정적으로 분류된 모든 인스턴스를 검색합니다.

- **Performance Highlights**: CLIP-Branches는 2억 6천만 개 이상의 이미지 인스턴스가 포함된 데이터 세트에서 그 효과를 입증하였습니다. 초기 검색 결과에 비해 검색 관련성과 정확도가 향상되었으며, 빠른 응답 시간을 유지합니다. 코드 저장소는 공개되어 있으며, 다른 연구자들이 자신만의 데이터와 사용 사례를 위해 재사용할 수 있습니다.



### Enhancing Collaborative Semantics of Language Model-Driven Recommendations via Graph-Aware Learning (https://arxiv.org/abs/2406.13235)
Comments:
          10pages

- **What's New**: 이번 연구에서는 LLMs(Large Language Models)를 추천 시스템 도메인에 더욱 효과적으로 적용하기 위한 새로운 방법론, Graph-Aware Learning을 제안했습니다. 이를 통해 기존 모델들이 겪었던 사용자의 협업 정보(collaborative information)를 잘 포착하지 못하는 문제를 해결하고자 합니다.

- **Technical Details**: GAL-Rec(Graph-Aware Learning for Language Model-Driven Recommendations)는 그래프 신경망(Graph Neural Networks, GNNs)의 다중 홉 정보 집계를 모방하여 사용자-아이템 협업 의미를 잘 이해할 수 있도록 설계된 프레임워크입니다. 이 방식은 사용자 간 및 아이템 간의 상호작용 데이터를 보다 효과적으로 학습할 수 있도록 합니다.

- **Performance Highlights**: 세 가지 실제 데이터셋을 이용한 실험 결과, GAL-Rec은 협업 의미를 이해하는 능력을 크게 향상시켰으며, 추천 성능에서도 뛰어난 결과를 보였습니다. 기존의 여러 최첨단 모델들보다도 높은 성능을 나타냈습니다.



### Reproducibility in Machine Learning-based Research: Overview, Barriers and Drivers (https://arxiv.org/abs/2406.14325)
Comments:
          Pre-print of submission planned to the AI Magazine

- **What's New**: 최근 다양한 연구 분야에서 재현성 문제(재현성 위기)를 겪고 있으며, 이는 인공지능(AI)과 머신러닝(ML) 연구에서도 주요한 문제가 되고 있습니다. 이번 연구에서는 ML 기반 연구의 재현성 문제를 다루면서, (i) 재현성의 장벽을 식별 및 유형별로 분류하고, (ii) ML 재현성을 지원하는 도구, 관행, 개입책들을 기술 구동, 절차 구동, 인식 및 교육 관련 드라이버로 구분하며 식별하고, (iii) 이 드라이버들을 장벽에 매핑함으로써 보다 나은 ML 재현성을 위한 통찰을 제공합니다.

- **Technical Details**: 연구는 연구 재현성을 기술 재현성, 코드 재현성, 데이터 재현성 및 실험 재현성으로 구분하며, 각각의 유형에 따른 장벽과 이를 해결하는 드라이버를 식별합니다. 이를 위해, Gundersen의 정의를 바탕으로 재현성의 타겟을 '결과 재현성', '분석 재현성', '해석 재현성'으로 세분화하였습니다. 또한, 방법론 재현성도 추가로 고려하였으며, 이를 텍스트, 코드, 데이터 및 실험의 수준으로 나누어 설명했습니다. 각 재현성 유형에 따른 장벽은 주로 ML 모델의 명확한 설명 부족, 평가 메트릭의 불명확성, 결과의 선택적 보고 등으로 나타났습니다.

- **Performance Highlights**: 논문에서는 ML 재현성의 주요 장벽으로 9가지를 제시하며, 이 중에서 특히 잘못되거나 불완전한 ML 모델 또는 학습 절차의 설명, 평가 메트릭의 불명확성, 결과의 선택적 보고가 주된 문제로 지적되었습니다. 예를 들어, 자연어 처리(NLP) 연구에서는 출판된 성과보다 재현성이 낮게 나타나는 경우가 많습니다. 또한, ML 모델이 바이오메디컬 분야에서 효과적임에도 불구하고 보고 품질이 미흡하여 임상관리에 통합되기 어렵다는 점을 강조합니다.



### Towards Holistic Language-video Representation: the language model-enhanced MSR-Video to Text Datas (https://arxiv.org/abs/2406.13809)
- **What's New**: 이 논문은 비디오 이해를 개선하기 위해 보다 강력하고 전체적인 언어-비디오 표현을 제안합니다. 현재는 단순한 텍스트 설명과 시각적 데이터에만 집중하지만, 이로 인해 실제 자연어 비디오 검색 작업에서는 제한된 용량을 가집니다. 이를 해결하기 위해 이 논문은 비디오-언어 데이터셋을 자동으로 향상시키는 방법을 도입하였습니다.

- **Technical Details**: 저자들이 제안한 다면적인 비디오 캡션 기법(multi-faceted video captioning method)은 개체(entities), 행동(actions), 음성 전사(speech transcripts), 미적 요소(aesthetics), 감정 신호(emotional cues)를 캡쳐하여 텍스트와 비디오 간의 상관된 정보(correlating information)를 제공하여 더욱 복잡한 표현 학습을 가능하게 합니다. 또한, 언어 모델(language models)을 활용하여 높은 품질의 사실적 텍스트 설명을 자동으로 생성하는 에이전트와 같은 전략(agent-like strategy)을 개발하여 사람의 개입을 줄이고 확장성을 제공합니다.

- **Performance Highlights**: 이 방법의 언어-비디오 표현 개선 효과는 MSR-VTT 데이터셋과 여러 멀티모달 검색 모델(multi-modal retrieval models)을 사용한 텍스트 비디오 검색을 통해 평가되었습니다.



### Converging Dimensions: Information Extraction and Summarization through Multisource, Multimodal, and Multilingual Fusion (https://arxiv.org/abs/2406.13715)
Comments:
          11 pages, 3 figures

- **What's New**: 이번 논문에서는 단일 소스 데이터에 의존하는 기존 전략의 한계를 극복하면서 더욱 포괄적이고 정보성이 높은 요약을 제공하는 새로운 접근 방식을 제안합니다. 이 접근 방식은 YouTube 재생 목록, 사전 인쇄본(pre-prints), Wikipedia 페이지 등 다양한 소스들을 통합하여 단일 텍스트 표현으로 변환합니다. 이렇게 통합된 정보는 고유한 분석과 이해를 가능하게 합니다.

- **Technical Details**: 이 시스템은 멀티소스, 멀티모달(multi-modal), 멀티링구얼(multilingual) 플랫폼으로, 다양한 데이터 소스에서 멀티팩시티드(multi-faceted) 요약을 생성하기 위해 고안되었습니다. 이는 YouTube 재생 목록, arXiv 논문, 웹 검색 등의 세 가지 멀티모달 소스들을 활용하고, Whisper와 같은 고급 음성 인식 모델을 통해 오디오와 비디오를 각각 텍스트로 변환하는 작업을 포함합니다.

- **Performance Highlights**: 실험 결과, 이러한 멀티소스 접근 방식은 정보 중복을 최소화하고 정보 이득을 최대화하여 고도의 일관성과 정보를 제공하는 요약을 생성하는 데 성공적입니다. 또한 다양한 소스와 언어에서 정보를 통합하여 더 깊이 있는 이해와 분석을 가능하게 합니다.



### Mining United Nations General Assembly Debates (https://arxiv.org/abs/2406.13553)
Comments:
          4 pages, 1 figure, 2 tables

- **What's New**: 이 논문은 자연어 처리(NLP) 기술을 사용하여 유엔 총회(UNGA) 연설을 분석하는 프로젝트에 대해 다룹니다. 이는 대규모 텍스트 데이터를 효율적으로 처리하고, 의미 패턴 추출, 감정 분석, 토픽 모델링을 가능하게 합니다. 정치 학자들이 국제 관계에 대한 통찰을 얻고, 글로벌 외교 담론에 대한 세밀한 이해를 높이는 데 활용할 수 있는 포괄적인 데이터셋과 도구를 제공하는 것이 목표입니다.

- **Technical Details**: 데이터셋은 1946년부터 2023년까지의 UNGA 연설을 담고 있으며, 날짜, 연설자의 이름 및 역할 등 여러 추가 특징으로 보완되었습니다. 이후, 변환기(transformer)를 기반으로 하는 토픽 모델링 기술인 BERTopic을 적용하여 연설의 주제와 국제적 우려를 표현하는 언어를 분석했습니다. BERTopic은 사전 학습된 대형 언어 모델을 활용하여 문서 내 단어의 의미적 유사성을 바탕으로 주제를 생성합니다.

- **Performance Highlights**: 업데이트된 UNGD 코퍼스는 2023년 연설을 포함하여 총 10,679개의 연설을 포함합니다. 메타데이터의 무결성을 개선하기 위해 반복적인 특징 엔지니어링 프로세스를 통해 ISO 코드, 국가명, 연설자 이름, 직위 등의 오류를 수정하고 표준화했습니다. BERTopic은 토픽 일관성(coherence) 및 다변성(diversity) 측정 기준으로 평가되었으며, DistilBERT 임베딩 방법이 가장 좋은 성능을 보였습니다. 전체 결과는 Streamlit 데이터 앱 프레임워크를 사용해 개발된 인터랙티브 애플리케이션에 통합되어, 비기술적인 사용자도 데이터를 쉽게 탐색하고 분석할 수 있게 되었습니다.



### R^2AG: Incorporating Retrieval Information into Retrieval Augmented Generation (https://arxiv.org/abs/2406.13249)
- **What's New**: 이 논문은 RAG(Retrieval Augmented Generation) 프레임워크의 성능을 향상시키기 위해 R^2AG라는 새로운 접근법을 제시합니다. 기존 RAG에서는 대형 언어 모델(LLMs)과 정보 검색 모델(retrievers) 사이의 의미적 간극이 존재했습니다. R^2AG는 이 간극을 메우기 위해 검색 정보를 통합한 RAG를 설계하였습니다. 특히, R$^2$-Former라 불리는 경량 모델을 통해 검색 정보를 캡처하여 LLMs의 문자 생성 과정에 통합하는 전략을 채택했습니다.

- **Technical Details**: R^2AG는 검색 결과를 단순히 텍스트로 결합하는 대신, retriever의 의미적 표현을 통합합니다. R$^2$-Former는 retriever와 LLM 사이에 삽입되는 모델로, 핵심 검색 정보를 포착하고 retrieval-aware prompting strategy를 통해 LLMs에 추가적인 임베딩을 제공합니다. 이러한 과정은 retrieval 문서를 처리할 때의 정보 손실을 방지하고, LLMs의 정확한 응답 생성을 지원합니다. 또한, R^2AG는 retrievers와 LLMs가 고정된 상태에서 작동할 수 있어, 리소스가 제한된 시나리오에서도 유용합니다.

- **Performance Highlights**: 다섯 개의 데이터셋에서 광범위한 실험을 통해 R^2AG의 효과, 강인성, 효율성을 검증했습니다. 분석 결과, R^2AG는 추론 중 지연 시간을 단 0.8% 증가시키면서도 LLMs가 검색된 문서를 이해하고 더 정확한 응답을 생성하는 데 도움을 줄 수 있음을 보여주었습니다.



### Towards Robust Evaluation: A Comprehensive Taxonomy of Datasets and Metrics for Open Domain Question Answering in the Era of Large Language Models (https://arxiv.org/abs/2406.13232)
Comments:
          22 pages, 13 tables, 7 figures

- **What's New**: 본 연구는 열린 도메인 질문 응답(Open Domain Question Answering, ODQA)에 관한 현재의 벤치마킹 상황을 종합적으로 검토합니다. 이를 위해 텍스트 및 멀티모달 데이터셋 52개와 평가 기술 20개를 분석하였습니다. 또한 ODQA 데이터셋에 대해 다중모달성(multimodalities)과 질문 유형의 난이도를 포함하는 새로운 분류법(taxonomy)을 도입했습니다.

- **Technical Details**: ODQA 시스템은 대규모 지식 말뭉치(knowledge corpora)를 사용하여 사실적 질문에 답하는 시스템을 구축하는 과정을 포함합니다. 최근의 발전은 대규모 훈련 데이터셋, 딥러닝 기술, 대형 언어 모델(large language models)의 부상 덕분입니다. 표준화된 메트릭(metrics)은 다양한 ODQA 시스템의 비교를 용이하게 하여 연구자들이 해당 분야의 발전을 객관적으로 추적할 수 있도록 합니다. 본 연구는 ODQA의 평가 메트릭과 그 고유한 트레이드오프(trade-offs)에 대한 구조화된 조직과 비판적 분석을 제시합니다.

- **Performance Highlights**: 본 연구는 현대 ODQA 시스템의 강력한 평가를 위한 프레임워크를 제공함으로써 연구자들에게 힘을 실어줍니다. 현재의 ODQA 시스템이 직면한 주요 도전 과제를 식별하고, 향후 연구 및 개발을 위한 유망한 경로를 제시하고 있습니다.



### Communication-Efficient Federated Knowledge Graph Embedding with Entity-Wise Top-K Sparsification (https://arxiv.org/abs/2406.13225)
- **What's New**: 연합 지식 그래프 임베딩 학습(FKGE)은 통신 효율성 문제를 겪고 있으며, 이는 파라미터의 크기와 많은 통신 라운드로 인해 발생합니다. 기존의 FKGE 방법은 통신 라운드를 줄이려고 다양한 로컬 학습 라운드를 수행하지만, 각 통신 라운드에서 전송되는 파라미터 크기를 줄이는 데에는 집중하지 않았습니다. 이를 해결하기 위해, 모든 엔티티의 임베딩 정밀도를 보편적으로 줄이면 수렴 속도가 크게 저하된다는 사실을 발견했습니다. 이에 따라, 엔티티 별 Top-K 희소화 전략(Entity-Wise Top-K Sparsification)을 기반으로 양방향 통신 효율을 높이는 새로운 방법인 FedS를 제안했습니다.

- **Technical Details**: FedS는 클라이언트가 변화가 큰 Top-K 엔티티 임베딩만 서버로 업로드하고, 서버는 각 클라이언트에 대해 개인화된 임베딩 집계를 한 후 Top-K 집계된 임베딩을 다시 클라이언트에 전송하는 방식을 사용합니다. 또한, 불일치를 해결하기 위해 간헐적 동기화 메커니즘(Intermittent Synchronization Mechanism)을 도입하여 일정 간격으로 모든 파라미터를 전송합니다.

- **Performance Highlights**: 세 가지 데이터셋에서 수행된 광범위한 실험 결과, FedS는 성능 저하 없이 통신 효율성을 크게 향상시켰습니다. 이는 특히 대규모 지식 그래프와 높은 임베딩 차원을 가진 클라이언트가 많은 환경에서 두드러졌습니다.



### Can Long-Context Language Models Subsume Retrieval, RAG, SQL, and More? (https://arxiv.org/abs/2406.13121)
Comments:
          29 pages. Dataset available at this https URL

- **What's New**: LOFT(LOng-Context Frontiers) 벤치마크를 소개합니다. 이 벤치마크는 실전 태스크에서 긴 컨텍스트를 필요로 하는 작업을 통해 LCLMs(Long-Context Language Models)의 성능을 평가하기 위해 설계되었습니다. LOFT는 최대 백만 개의 토큰 컨텍스트를 다루며, 텍스트, 비주얼, 오디오 등 다양한 모달리티를 포함한 35개의 데이터셋을 포함합니다.

- **Technical Details**: LCLMs는 복잡한 파이프라인 의존성을 줄이고, 툴 없이 전체 코퍼스를 네이티브하게 처리할 수 있는 모델입니다. LOFT 벤치마크는 32k, 128k, 1M 세 가지의 컨텍스트 길이 제한을 제공하며, 각 데이터셋에 대해 최대 100개의 테스트 쿼리와 5개의 few-shot 쿼리, 10개의 개발 쿼리를 샘플링합니다. 주요 평가 영역은 Retrieval, Retrieval-Augmented Generation (RAG), SQL 처리를 포함하며, Many-Shot In-Context Learning (ICL) 능력도 평가됩니다.

- **Performance Highlights**: 128k 토큰 수준에서 LOFT는 LCLMs의 성능이 Gecko와 같은 최첨단 텍스트 검색 시스템과 맞먹는다는 것을 보여줍니다. Gemini는 CLIP 등의 멀티모달 검색 모델을 초과합니다. 그러나 SQL과 같은 복잡한 다중 홉 조합 추론 작업에서는 여전히 성능이 부족합니다. 또한, 체인-오브-생각(chain-of-thought) 전략 등 프롬프트 방식에 따라 성능 변동이 크다는 점도 강조되었습니다.



### Accelerating Complex Disease Treatment through Network Medicine and GenAI: A Case Study on Drug Repurposing for Breast Cancer (https://arxiv.org/abs/2406.13106)
Comments:
          9 pages double columns, 5 figures, 3 algorithms, 3 tables, and 1 listing, Submitted to IEEE MedAI'24 Conference, to be held November 15-17, Chongqing, China

- **What's New**: 이번 연구는 임상 시험 및 생의학 문헌과 같은 실세계 증거(real-world evidence)원을 조사하여 재목적화될 수 있는 약물을 예측하는 네트워크를 소개합니다. 특히 복잡한 질병(예: 암, 알츠하이머)에 대한 약물 조합 치료법을 생성합니다. 이를 위해, 고도로 구성된 ChatGPT 프롬프트 엔지니어링 시스템을 활용한 다층 네트워크 의료 접근 방식을 제안합니다.

- **Technical Details**: 이 접근 방식은 임상 시험에서 약물 언급을 추출하기 위해 즉석에서 구성된 ChatGPT 프롬프트 엔지니어링 시스템을 사용합니다. 또한, 실세계 증거와 질병별 신호 경로(예: KEGG 데이터베이스)를 연결하는 새로운 알고리즘을 소개합니다. 이 연구는 유방암의 경우를 예로 들어 제안된 프레임워크를 구현했으며, KEGG 경로 ID hsa:2064가 108개의 약물에 의해 커버되는 등 다양한 약물 조합의 가능성을 제시합니다.

- **Performance Highlights**: 제안된 네트워크 의료 프레임워크는 유망한 약물 조합을 식별하는 데 높은 특이성을 보여주며, 실제 임상 시험에서 다수의 약물을 정확히 언급할 수 있습니다. 유방암 신호 경로 46개 중 38개의 경로가 적어도 두 개 이상의 약물에 의해 커버됨을 확인했습니다. ChatGPT를 활용한 프롬프트 엔지니어링 시스템은 임상 시험에서 약물 언급을 가속화하는 데 성공했습니다.



### GLiNER multi-task: Generalist Lightweight Model for Various Information Extraction Tasks (https://arxiv.org/abs/2406.12925)
Comments:
          11 pages, 1 figure, 6 tables

- **What's New**: 새로운 형태의 GLiNER 모델을 소개합니다. 이 모델은 정보 추출에 필요한 다양한 작업을 수행할 수 있으며, 작은 인코더 모델로 구축되었습니다. 기존의 대규모 언어 모델(LLMs)보다 효율적이며 구조화된 출력을 생성할 수 있습니다. 이 모델은 제로샷(named entity recognition) 벤치마크에서 SoTA(state of the art) 성능을 기록했으며, 질문 응답, 요약 및 관계 추출 작업에서도 우수한 성능을 보였습니다.

- **Technical Details**: GLiNER 모델은 GLiNER 토큰 분류 아키텍처를 기반으로 하며, 토큰을 분류하는 대신 시퀀스를 추출할 수 있게 합니다. 이를 통해 긴 엔티티 추출, 요약, 텍스트 정리 작업에서 유리합니다. BERT 같은 인코더 아키텍처 위에 구축되었으며, 이번 연구에서는 DeBERTA v3 large를 사용했습니다. 이 모델은 효율성을 높이기 위해 RTD(replaced token detection) 방식을 채택했습니다. GLiNER는 레이블과 텍스트를 단일 인코더 모델에서 처리함으로써 상호 정보를 교환할 수 있게 합니다. 또, bidirectional LSTM을 통해 토큰 임베딩을 추가로 처리해 모델 학습을 가속화하고 데이터 부족 상황에서도 좋은 성능을 유지합니다.

- **Performance Highlights**: GLiNER 모델은 제로샷(named entity recognition) 벤치마크에서 SoTA 성능을 기록했습니다. 또한, 질문 응답, 요약 및 관계 추출 작업에서도 선도적인 성능을 보였습니다. 이 모델은 효율성과 정확성 측면에서 기존의 대규모 언어 모델에 비해 더 나은 성능을 보입니다.




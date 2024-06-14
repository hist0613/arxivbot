### Xwin-LM: Strong and Scalable Alignment Practice for LLMs (https://arxiv.org/abs/2405.20335)
- **What's New**: Xwin-LM은 대규모 언어 모델(LLM)을 위한 포괄적인 정렬 방법론 체계입니다. 이 새로운 체계는 감독된 미세 조정(SFT), 리워드 모델링(RM), 거절 샘플링 미세 조정(RS), 직접 선호 최적화(DPO)와 같은 여러 주요 기술을 포함하고 있습니다.

- **Technical Details**: Xwin-LM은 초기 모델을 고품질의 지침 데이터로 미세 조정한 Xwin-LM-SFT를 통해 시작됩니다. 이후, GPT-4를 이용하여 신중하게 주석된 다중 턴 선호 데이터셋 Xwin-Pair를 수집합니다. Xwin-Pair를 통해 7B, 13B, 70B 파라미터 규모의 리워드 모델 Xwin-RM을 훈련시킵니다. 나아가, 각 프롬프트에 대해 Xwin-LM-SFT가 생성한 64개의 응답을 Xwin-RM이 점수를 매긴 Xwin-Set을 생성합니다. 이를 통해 최고 점수 응답을 이용한 거절 샘플링 미세 조정 Xwin-LM-RS 및 DPO 알고리즘을 사용하여 추가 최적화한 Xwin-LM-DPO를 완성합니다.

- **Performance Highlights**: Xwin-LM의 성능은 AlpacaEval 및 MT-bench 벤치마크 평가에서 일관되고 의미 있는 향상을 보여주었습니다. 특히 초기의 Xwin-LM-SFT 모델은 만족스러운 시작을 보였고, 뒤이어 거절 샘플링 미세 조정과 직접 선호 최적화 단계는 모델 성능을 현저히 향상시켰습니다. Xwin-LM은 모든 Llama2 기반 모델 중에서 최첨단 성능을 달성했습니다.



### CausalQuest: Collecting Natural Causal Questions for AI Agents (https://arxiv.org/abs/2405.20318)
- **What's New**: 인간은 본능적으로 인과 관계를 알고자 합니다. 하지만, 현재 AI 사용 시나리오를 반영하는 자연적인 인과 질문을 포함한 데이터셋이 부족합니다. 이를 해결하기 위해, 소셜 네트워크, 검색 엔진, AI 어시스턴트 등에서 수집된 13,500개의 자연어 인과 질문으로 구성된 CausalQuest 데이터셋을 소개합니다.

- **Technical Details**: 이 연구는 인과 질문(Causal Questions)의 정의를 체계적으로 정립하고, 세부 분류를 위한 택소노미(Taxonomy)를 수립했습니다. 또한, 인간 주석자와 대형 언어 모델(LLM)을 결합하여 데이터를 신중하게 레이블링했습니다. 연구 결과, 인간이 묻는 질문의 42%가 인과적 질문임을 발견했습니다. 이는 주로 주어진 효과의 원인을 이해하려는 질문이 다수를 차지했습니다.

- **Performance Highlights**: CausalQuest 데이터를 활용해 최대 2.85B 파라미터 수의 효율적인 이진 분류기(Binary Classifier)를 훈련시켰으며, F1 점수 0.877이라는 높은 성능을 달성했습니다. 앞으로 이 데이터와 모델을 바탕으로 다양한 연구 방향을 제시하고 있습니다.



### ANAH: Analytical Annotation of Hallucinations in Large Language Models (https://arxiv.org/abs/2405.20315)
Comments:
          Accepted by ACL 2024

- **What's New**: 최신 연구인 ANAH(Analytical Annotation of Hallucinations)는 대규모 언어 모델(LLMs)에서 '환각(hallucination)' 문제를 해결하기 위해 양방향 데이터셋을 소개합니다. 이 데이터셋은 생성형 질문 응답에서 발생하는 환각을 분석적으로 주석 달아 각 답변의 문장을 평가합니다.

- **Technical Details**: ANAH 데이터셋은 700개 이상의 주제를 포괄하는 약 12k 문장 수준의 주석과 4.3k LLM 응답을 포함합니다. 인간 참여형 파이프라인을 통해 구성되었으며, 각 답변 문장은 참조 조각을 검색하고 환각 유형을 판단하며 필요 시에는 환각된 내용을 수정하는 과정을 거칩니다. 데이터셋의 주제는 정치, 역사, 예술, 과학 기술 등 폭넓은 범위에서 선정되었으며, 영어와 중국어로 이루어져 있습니다.

- **Performance Highlights**: 연구 결과, 현재의 공개된 LLM들은 세밀한 환각 주석에서 어려움을 겪고 있으나, ANAH로 훈련된 생성형 주석자는 모든 공개 소스 LLM을 능가하며 GPT-4와 유사한 성능을 보였습니다. 특히, 생성형 주석자는 새로운 질문에 대한 일반화 능력이 뛰어나며, 81.01%의 정확도를 기록하였습니다. 이는 GPT-4의 86.97%에 근접한 성과입니다.



### S3D: A Simple and Cost-Effective Self-Speculative Decoding Scheme for Low-Memory GPUs (https://arxiv.org/abs/2405.20314)
- **What's New**: 최신 연구에서는 Speculative Decoding(SD)의 높은 속도 향상에도 불구하고, 이를 고효율로 사용하는데 필요한 고급 장비와 많은 GPU 메모리가 필요하다는 문제를 해결하기 위해 Skippy Simultaneous Speculative Decoding(S3D)를 제안합니다. S3D는 중간 층 스킵(middle-layer skipping)과 동시에 여러 토큰을 디코딩하는 방식을 채택하여 비용 효율적이면서도 높은 성능을 달성합니다.

- **Technical Details**: Skippy Simultaneous Speculative Decoding(S3D)은 자가 추측 디코딩(self-speculative decoding) 방식으로, 메모리를 절약하면서도 높은 훈련 효율성을 제공합니다. S3D는 중간 층을 스킵하고 동시에 여러 토큰을 예측하는 기능을 포함하고 있습니다. 이러한 방법은 VRAM 비용을 추가하지 않고도 최적의 성능-메모리 비율을 달성할 수 있도록 설계되었습니다.

- **Performance Highlights**: 제안된 S3D 모델은 최근의 개방형 SD 시스템들 중에서도 가장 높은 성능-메모리 비율을 자랑합니다. 또한, 8비트 양자화(quantization) 상태에서 A10G GPU 상에서 이전의 가장 빠른 SD 모델인 EAGLE보다 최대 3.9배 빠른 속도를 보여줍니다. 더 작은 목표 모델인 Phi-3을 기반으로 한 SD 모델은 EAGLE보다 1.4~2배 빠른 디코딩 속도를 가지며, VRAM 소모도 줄입니다.



### Group Robust Preference Optimization in Reward-free RLHF (https://arxiv.org/abs/2405.20304)
Comments:
          Preprint

- **What's New**: 대규모 언어 모델 (Large Language Models, LLMs)을 특정 작업에 맞추기 위해서는 보통 강화 학습(Reinforcement Learning)과 인간 피드백(Human Feedback, RLHF)을 통한 미세 조정이 필요합니다. 그러나 전통적인 RLHF 접근법은 다양한 그룹의 특성과 필요를 반영하지 못하는 'equal' 접근법을 사용합니다. 이를 해결하기 위해, 개별 그룹의 선호도를 보다 잘 반영할 수 있는 새로운 Group Robust Preference Optimization (GRPO) 방법을 제안합니다.

- **Technical Details**: GRPO는 보상 없이 직접 선호도를 최적화하는 기존 메소드(RLH, Reward-Free direct Preference Optimization)에 기반하여 개발되었습니다. 하지만, 기존 접근법과는 달리 GRPO는 그룹 간 최악의 성능을 최대화하는 강건한 정책을 추구합니다. 이를 위해 GRPO는 그룹별 누적 손실이 높은 그룹을 우선적으로 고려하여 다양한 그룹의 중요성을 적응적으로, 순차적으로 가중치를 두어 최적화를 수행합니다. 이론적으로 GRPO의 유효성을 연구하고, log-linear 정책 클래스에 대한 수렴(convergence)을 분석했습니다.

- **Performance Highlights**: 다양한 그룹 기반의 글로벌 의견 데이터를 사용하여 LLM을 GRPO로 미세 조정한 결과, 성능이 가장 낮은 그룹의 성능이 크게 향상되었고, 그룹 간 손실 불균형(loss imbalance)이 감소했으며, 확률 정확성(probability accuracy) 또한 비강건 기준선(non-robust baselines)보다 향상되었습니다.



### Who Writes the Review, Human or AI? (https://arxiv.org/abs/2405.20285)
- **What's New**: 본 연구는 AI가 생성한 텍스트 및 인간이 작성한 텍스트를 식별하는 새로운 방법론을 제안합니다. 특히 책 리뷰에서 AI와 인간 작성 텍스트를 구별하는 데 중점을 둡니다. Vicuna 오픈소스 언어 모델을 사용하여 생성한 AI 리뷰와 실제 책 리뷰 데이터를 활용합니다.

- **Technical Details**: 이 연구는 Transfer Learning(전이 학습)을 활용하여 다양한 주제에서 생성된 텍스트를 식별합니다. 이를 위해 기존의 인간 작성 리뷰 데이터셋을 Kaggle에서 수집하고, Vicuna 모델을 사용해 AI 생성 리뷰를 생성했습니다. 학습 데이터셋은 아마존의 Goodreads 플랫폼에서 수집된 3백만 개의 리뷰로 구성되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법론은 텍스트의 원본 출처를 감지하는 데 있어 96.86%의 정확도를 달성했습니다. 이는 인간 작성 텍스트와 AI 생성 텍스트를 효과적으로 구별할 수 있다는 점을 보여줍니다.



### ROAST: Review-level Opinion Aspect Sentiment Target Joint Detection (https://arxiv.org/abs/2405.20274)
Comments:
          arXiv admin note: text overlap with arXiv:2309.13297

- **What's New**: ABSA(Aspect-Based Sentiment Analysis) 분야가 다중 언어와 도메인 전반에 걸쳐 상당한 발전을 이뤘지만, 여전히 저리소스 언어(low-resource language) 평가의 부족과 문장 수준 분석에 치중하는 단점이 있습니다. 이러한 문제를 해결하기 위해 본 연구에서는 새로운 과제인 ROAST(Review-Level Opinion Aspect Sentiment Target)를 소개합니다. ROAST는 문장 수준과 텍스트 수준의 ABSA 간의 격차를 해소하고 모든 ABSA 구성 요소를 리뷰 수준에서 식별하는 것을 목표로 합니다.

- **Technical Details**: 본 연구는 기존의 데이터셋을 확장하여 다중 언어와 저리소스 언어를 포함한 새로운 주제를 다루면서, ROAST 과제를 지원하는 데 중점을 둡니다. 예를 들어, OATS 데이터셋이 다중 도메인에 걸쳐 있지만 영어 리뷰로만 구성되었던 반면, ROAST는 다양한 언어와 도메인에서 리뷰 데이터를 포함할 예정입니다. 이로써 현실적인 시나리오에서 ABSA 모델의 적용성을 평가할 수 있는 포괄적인 데이터셋을 제공합니다.

- **Performance Highlights**: ROAST는 문장 수준 ABSA의 한계를 극복하고, 복잡한 리뷰의 전체적인 맥락을 이해하는 데 도움을 줄 것입니다. 기존 ABSA 과제들이 리뷰 문장별로 개별 감정을 요약하는 데 중점을 두었던 데 반해, ROAST는 전체 리뷰 맥락 내에서 모든 ABSA 요소의 결합 검출 개념을 도입해 실제 응용 가능성을 높입니다. 이를 통해 다양한 언어와 도메인에서 ABSA 연구를 더욱 확장하고 의미 있는 통찰을 제공할 수 있습니다.



### IsraParlTweet: The Israeli Parliamentary and Twitter Resourc (https://arxiv.org/abs/2405.20269)
Comments:
          Presented at LREC-COLING 2024

- **What's New**: IsraParlTweet는 새로운 히브리어 의회 논의를 수집한 코퍼스입니다. 이 코퍼스는 1992-2023년 사이 Knesset(이스라엘 의회)에서의 토론과 함께 2008-2023년 사이 동일한 의회 구성원이 작성한 Twitter 게시물을 포함하고 있습니다. 총 2억 9450만 개의 히브리어 토큰이 포함되어 있습니다.

- **Technical Details**: 원시 텍스트 이외에도, 이 코퍼스에는 연사 및 Knesset 세션에 대한 종합적인 메타데이터와 몇 가지 언어적 주석이 포함되어 있습니다. 이 데이터를 통해 다양한 정량적 및 정성적 분석을 수행할 수 있습니다.

- **Performance Highlights**: IsraParlTweet는 이스라엘 정치 담론에 대한 귀중한 통찰력을 제공할 수 있으며, 데이터 분석의 폭과 깊이를 크게 넓힐 수 있습니다.



### Auto Arena of LLMs: Automating LLM Evaluations with Agent Peer-battles and Committee Discussions (https://arxiv.org/abs/2405.20267)
- **What's New**: 최신 대규모 언어 모델(LLMs)의 발전에 따라, 신뢰할 수 있는 평가 방법의 필요성이 커지고 있지만, 현존하는 정적 벤치마크는 오염 문제로 신뢰도가 떨어질 수 있습니다. 이를 해결하기 위해 Auto-Arena of LLMs가 제안되었습니다. 이 프레임워크는 LLM 에이전트를 이용해 평가 과정을 자동화하여 인간 평가와 높은 상관성을 나타내며, 신뢰성 있고 효율적인 평가를 제공합니다.

- **Technical Details**: Auto-Arena 프레임워크는 세 가지 주요 단계로 구성됩니다: (1) 질문 생성 단계에서 한 LLM이 사용자처럼 질문을 생성합니다. (2) 두 후보 LLM이 다수 라운드에 걸쳐 질문에 대해 상호평가(peer battle)하며 능력을 발휘합니다. (3) 여러 LLM 판사들로 구성된 위원회가 최종 승자를 결정합니다. 이 과정은 데이터 오염 문제를 줄이고, 여러 모델의 의견을 반영하여 공정성을 확보합니다.

- **Performance Highlights**: 17개의 최신 LLM을 대상으로 한 실험에서, Auto-Arena는 인간 선호도와의 스피어만 상관관계를 4.5% 높여 최신 기술 수준으로 향상시켰습니다. 특유의 peer battle 및 위원회 토론 단계가 인간 평가와의 상관관계 및 일치율을 크게 증가시키는 것이 확인되었습니다. 중국어 예제를 통해 비주류 언어와 도메인에도 쉽게 확장 가능함을 보여주었습니다.



### Evaluating Large Language Model Biases in Persona-Steered Generation (https://arxiv.org/abs/2405.20253)
Comments:
          Accepted to Findings of ACL 2024. Code and data available at this https URL

- **What's New**: 이번 연구는 다양한 측면에서 정의된 사람들의 '페르소나'(persona)를 반영한 텍스트를 생성할 수 있도록 대형 언어 모델(LLMs)을 조종하는 방법을 제시합니다. 특히, 기존 연구에서 자세히 다루지 않은 '불일치 페르소나'(incongruous persona)를 소개하며, 이를 반영하는 모델의 조종 가능성에 대해 조사했습니다. 불일치 페르소나는 하나의 특성이 다른 특성의 가능성을 낮추는 페르소나로 정의됩니다.

- **Technical Details**: 연구팀은 정치, 성별, 인종 등의 다양한 특성과 관련된 미국 국민의 조사 데이터를 사용해 다각적인 페르소나를 구축했습니다. 두 가지 모델군(Llama 2와 Tulu 2)을 사용해 모델이 주어진 페르소나에 맞는 문장을 잘 생성할 수 있는지 평가했습니다. 또한, 인간의 피드백으로 강화 학습(Reinforcement Learning from Human Feedback, RLHF)된 모델을 포함하여 다양한 크기의 모델을 실험에 활용했습니다.

- **Performance Highlights**: 모든 LLM은 불일치 페르소나에 대해 9.7% 낮은 조종 가능성을 보였으며, RLHF로 미세 튜닝된 모델은 더 잘 조종할 수 있었지만, 페르소나의 표현 다양성이 58.2% 감소하는 경향이 있었습니다. 또한, 고정 선택지 질문에 대한 모델의 응답이 개방형 텍스트 생성에서의 조종 가능성을 충분히 예측하지 못했습니다. 마지막으로, GPT-4는 사람처럼 페르소나를 평가하는 데 있어 유사한 정확도를 보였습니다(F1 스코어 96.3%).



### Towards Hierarchical Multi-Agent Workflows for Zero-Shot Prompt Optimization (https://arxiv.org/abs/2405.20252)
- **What's New**: 대형 언어 모델(LLMs)은 다양한 질문에 답할 수 있지만, 그 성능은 프롬프트 디자인에 크게 의존합니다. 이 논문에서는 LLMs가 최적의 프롬프트를 스스로 설계할 수 있도록 하는 새로운 접근 방식인 '계층적 다중 에이전트 워크플로우(Hierarchical Multi-Agent Workflow, HMAW)'를 소개합니다. HMAW는 CEO, 매니저, 워커의 계층적 구조를 통해 프롬프트를 생성하고, 이를 사용해 최종 답변을 제공합니다. 이는 인간의 개입 없이도 다양한 작업에 유연하게 적용될 수 있습니다.

- **Technical Details**: HMAW는 기본적으로 회사의 계층 구조를 모방합니다. 먼저 CEO 역할을 하는 LLM이 전체적인 지침을 생성하고, 매니저 역할을 하는 LLM이 더 구체적인 체크리스트를 작성합니다. 마지막으로 워커 역할을 하는 LLM이 최종 답변을 생성합니다. 이러한 구조는 작업을 분할하여 LLM 각각이 자신에게 할당된 역할에만 집중할 수 있게 만들어줍니다. HMAW는 데이터셋이나 학습이 필요 없으며, 완전히 제로숏(zero-shot)이며 태스크에 독립적인 특성을 가지고 있습니다.

- **Performance Highlights**: 여러 벤치마크 테스트를 통해 HMAW의 효과를 검증한 결과, Mixtral과 결합하여 5개 데이터셋에서 평균 30.7%의 성능 향상을 달성했습니다. 이는 현재 가장 진보된 LLM들보다도 뛰어난 성능을 입증한 것입니다.



### Retrieval Augmented Structured Generation: Business Document Information Extraction As Tool Us (https://arxiv.org/abs/2405.20245)
Comments:
          Accepted by IEEE 7th International Conference on Multimedia Information Processing and Retrieval (MIPR), 2024

- **What's New**: 이 논문에서는 비즈니스 문서 정보 추출 (BDIE)의 새로운 방법론으로 Retrieval Augmented Structured Generation (RASG)을 제안합니다. 이 방법론은 Key-Information Extraction (KIE)와 Line Items Recognition (LIR)에서 최신 성능(SOTA)을 달성합니다. 또한, 새로운 메트릭 클래스인 General Line Items Recognition Metric (GLIRM)을 제안하여 기존의 메트릭보다 실제 사용 사례에 더 부합하도록 하였습니다.

- **Technical Details**: RASG는 4가지 주요 구성 요소로 구성됩니다: (1) In-Context Learning을 활용한 Retrieval Augmented Generation, (2) 정확성을 향상시키는 Supervised Finetuning, (3) Structured Generation을 통한 결과의 파싱 가능성 보장, (4) 레이아웃 정보를 포함하는 Structured Prompting. 또한, BDIE에서 고성능 다중 모달 모델(LMM)보다 대형 언어 모델(LLM)과 RASG의 조합이 실제 상황에서 더 높은 성능을 발휘할 수 있음을 주장합니다.

- **Performance Highlights**: RASG는 최신 성능(SOTA)을 달성하며, 대형 언어 모델(LLM)과의 결합을 통해 기존 다중 모달 모델(LMM)보다 실질적인 이점을 가집니다. 또한, 새로운 메트릭 클래스인 GLIRM은 Line Items Recognition (LIR)에서 더 실질적인 평가 기준을 제공하여 상업적 활용 가능성을 높입니다.



### TS-Align: A Teacher-Student Collaborative Framework for Scalable Iterative Finetuning of Large Language Models (https://arxiv.org/abs/2405.20215)
- **What's New**: TS-Align 프레임워크를 도입하여 자동 페어와이즈 피드백 데이터를 사용해 정책 모델을 파인튜닝합니다. 이는 대규모 teacher 모델과 소규모 student 모델 간의 협업으로 자동적이고 효율적인 데이터 마이닝을 포함하며, 인간의 피드백 없이도 성능을 향상시킬 수 있습니다.

- **Technical Details**: TS-Align 프레임워크는 베이스 슈퍼바이즈드 파인튜닝 정책 모델을 사용하여 다양한 지시에 대한 응답 후보를 생성하고, 소규모 student 리워드 모델이 대량의 라벨링되지 않은 데이터를 빠르게 처리하여 후보 중 선호 쌍을 선택합니다. 선택된 쌍은 강력한 teacher 모델이 재정렬하여 신뢰성을 높이며, 해당 데이터를 사용해 정책 모델이 DPO(Direct Preference Optimization)를 통해 파인튜닝됩니다. 이 과정은 여러 번 반복되며, student 모델이 teacher 모델로부터 새로운 지식을 증류받습니다.

- **Performance Highlights**: 제안된 TS-Align 프레임워크는 7개의 대화 또는 지시 따르기 데이터셋에서 기본 정책 모델에 비해 평균 69.7%의 승률을 기록하였습니다. 또한, teacher 모델의 성능을 student 모델에 효과적으로 증류하여, 최종 student 모델의 강화된 성능을 다른 정책 모델 정렬에 전이할 수 있음을 보여줍니다.



### Jina CLIP: Your CLIP Model Is Also Your Text Retriever (https://arxiv.org/abs/2405.20204)
Comments:
          4 pages, ICML2024 workshop submission

- **What's New**: 초기 CLIP 모델이 텍스트 전용 작업에서 성능이 떨어지는 문제를 해결하기 위해, jina-clip-v1 모델을 제안했습니다. 이 모델은 텍스트-이미지와 텍스트-텍스트 검색 작업 모두에서 뛰어난 성능을 보입니다.

- **Technical Details**: 새로운 멀티태스크 대조 학습법을 도입하여 대규모 이미지-캡션 쌍 및 텍스트 쌍을 동시에 최적화했습니다. 이를 위해 JinaBERT 및 EVA02 아키텍처를 사용했으며, 세 단계의 학습 과정을 통해 모델을 훈련했습니다.

- **Performance Highlights**: jina-clip-v1 모델은 텍스트-이미지 검색에서 EVA-CLIP와 비슷한 성능을, 텍스트 인코더는 MTEB Benchmark에서 유사 모델들과 동등한 성능을 보였습니다.



### TAIA: Large Language Models are Out-of-Distribution Data Learners (https://arxiv.org/abs/2405.20192)
Comments:
          25 pages

- **What's New**: 이번 연구에서는 특정 세부 분야에서 고품질 데이터가 부족한 상황에서도 대형 언어 모델(LLMs)의 성능을 향상시키기 위한 새로운 방법론을 제안합니다. 	rainallInfAttn 방법을 도입하여, 모든 파라미터를 훈련하지만 추론 시에는 어텐션(attention) 파라미터만 사용하는 전략을 제안합니다. 이는 변형기(Transformer) 아키텍처의 재평가를 통해 기존 방법의 한계를 극복하고자 합니다.

- **Technical Details**: 연구진은 변형기(Transformer) 내부의 셀프 어텐션과 피드포워드 네트워크를 분석한 결과, LLM이 특정 과제에 맞게 파인 튜닝될 때, 셀프 어텐션 파라미터만이 주로 긍정적인 기여를 한다는 사실을 밝혀냈습니다. 이에 따라, 	rainallInfAttn 방법은 훈련 시 모든 파라미터를 업데이트하지만 추론 시에는 셀프 어텐션 파라미터만을 사용하는 간단하지만 효과적인 방법을 제시합니다.

- **Performance Highlights**: 이 방법론을 통해 수학, 추론, 지식 이해 등의 다양한 과제에서 LLM의 성능을 크게 향상시켰으며, 특히 데이터 불일치 상황에서도 우수한 성능을 보였습니다. 	rainallInfAttn은 여러 모델 구성 및 파인 튜닝 기법에 걸쳐 실험적으로 검증되었고, 전통적인 파인 튜닝 방법보다 높은 일관성을 유지하는 것으로 나타났습니다.



### Robo-Instruct: Simulator-Augmented Instruction Alignment For Finetuning CodeLLMs (https://arxiv.org/abs/2405.20179)
- **What's New**: Robo-Instruct는 자체 지시(Self-Instruct)의 다양성과 시뮬레이터 기반의 프로그램 검증을 결합하여 소규모 오픈웨이트 모델(open-weight model)을 통해 도메인 특화 로봇 프로그램을 생성하는 새로운 프레임워크입니다. 이는 코드 생성 성능을 향상시켜 GPT-3.5-Turbo 및 Gemini-1.0-Pro와 같은 여러 프로프라이어터리 LLM(performance gap with proprietary LLMs)을 능가하는 성과를 냅니다.

- **Technical Details**: Robo-Instruct는 두 가지 주요 구성 요소를 도입합니다: (1) RoboSim은 도메인 특화 제약을 인코딩하고 자체 지시로 생성된 로봇 프로그램을 유효성 검사하는 작업 무관 시뮬레이터(task-agnostic simulator)입니다. RoboSim는 프로그램을 실행하면서 동적으로 일관된 세계 상태를 합성합니다. (2) InstAlign은 생성된 지시사항과 프로그램이 더 잘 일치하도록 수정하는 지시-프로그램 정렬 절차(instruction-program alignment procedure)입니다.

- **Performance Highlights**: Robo-Instruct를 사용하여 Codellama-Python-7B를 미세 조정(fine-tuning)한 결과, 자체 지시를 사용한 모델보다 13.75% 개선된 성능을 보였고, 68.75%의 pass@1 성능을 달성하여 GPT-3.5-Turbo 및 Gemini-1.0-Pro를 능가했습니다. Codellama-Python-7B 기본 모델과 비교할 때, 28.75% 향상된 성능을 보였습니다.



### InstructionCP: A fast approach to transfer Large Language Models into target languag (https://arxiv.org/abs/2405.20175)
Comments:
          10 pages, 1 figure

- **What's New**: 본 논문에서는 기존의 지속적 예비 학습(Continual Pre-training, CP) 과정에서 모델이 대화 능력을 유지하면서 다른 언어에 적응하도록 하는 새로운 접근 방식인 '명령어 지속적 예비 학습(Instruction Continual Pre-training, InsCP)'을 제안합니다. InsCP는 명령어 태그를 CP 과정에 통합하여 모델이 영어 외의 언어를 학습하면서도 기존의 대화 능력과 강화학습(어서받기 강화학습, RLHF) 능력을 유지하도록 돕습니다.

- **Technical Details**: InsCP는 전통적인 CP와 지도 학습 기반 세부 튜닝(Supervised Fine-Tuning, SFT)의 단계를 하나의 통합된 훈련 과정으로 결합합니다. 구체적으로, 데이터에 특수 명령어 토큰을 추가하여 모델이 목적한 언어 입력에 대해 해당 언어로 응답할 수 있도록 하고, 원래의 RLHF 능력을 통해 공격적인 입력을 처리할 수 있도록 합니다. 실험에서 LLaMA3-instruct 모델을 사용하여 InsCP의 효과를 검증했으며, 주요 평가 지표로는 언어 정렬, 신뢰성 및 지식 벤치마크를 포함합니다.

- **Performance Highlights**: InsCP를 적용한 LLaMA3-instruct 모델은 전통 중국어 입력에 대해 우수한 성능을 보였으며, 영어 입력에 대해서도 적절한 응답을 유지했습니다. 또한, 언어 벤치마크 테스트에서 CP 이전과 이후의 모델 성능이 유사한 수준을 유지했으며, 신뢰성 벤치마크인 TruthfulQA 테스트에서도 일관된 성능을 보였습니다. 특히, InsCP는 단 0.1억 개의 고품질 명령어 데이터 토큰만을 필요로 하여 자원 소비를 크게 줄일 수 있었습니다.



### Reasoning about concepts with LLMs: Inconsistencies abound (https://arxiv.org/abs/2405.20163)
Comments:
          15 pages, 5 figures, 3 tables

- **What's New**: 최근 연구는 대규모 언어 모델(LLMs)이 개념적 일관성을 자주 결여한다는 사실을 밝혀냈습니다. 이와 관련하여 간단한 온톨로지(ontology)를 통해 여러 LLM이 보여주는 개념적 불일치를 드러낼 수 있음을 입증하였습니다.

- **Technical Details**: 연구팀은 전문 지식 그래프(Knowledge Graph, KG)와 온톨로지를 사용하여 개념의 'Is-A' 계층 구조를 분석했습니다. 이를 통해 간단한 지식 그래프를 기반으로 LLM의 불일치를 테스트하고, 추가적인 문맥을 제공함으로써 이러한 불일치를 줄이는 방법을 제안하였습니다.

- **Performance Highlights**: 조사 결과, 공개적으로 사용 가능한 가중치를 가진 여러 LLM이 작은 온톨로지에서도 많은 불일치를 보여주었습니다. 하지만 지식 그래프(KG) 기반의 간단한 프롬프트 전략을 사용하여 이러한 불일치를 줄이고, 도메인 개념의 포괄성을 개선할 수 있음을 증명하였습니다.



### Heidelberg-Boston @ SIGTYP 2024 Shared Task: Enhancing Low-Resource Language Analysis With Character-Aware Hierarchical Transformers (https://arxiv.org/abs/2405.20145)
Comments:
          Accepted for publication at the 6th Workshop on Research in Computational Linguistic Typology and Multilingual NLP (SIGTYP-WS) 2024; 11 pages, 1 figure, 9 tables

- **What's New**: 이번 연구는 SIGTYP 2024 공유 과제의 제한된 하위 과제에 대한 제출 작업을 설명합니다. 특히 13개의 역사적 언어에 대한 PoS 태깅, 형태소 태깅 및 어간화 작업을 중점적으로 다룹니다. 이 연구는 역사적 언어 응용을 위해 제한된 데이터로부터 최초의 우수한 학습 성과를 달성했습니다.

- **Technical Details**: PoS 및 형태소 태깅을 위해 Sun et al. (2023)의 계층적 토큰화 방법을 DeBERTa-V3 아키텍처의 장점과 결합하여 각 문자 데이터를 효율적으로 학습할 수 있도록 하였습니다. 어간화 작업에서는 캐릭터 레벨(character-level) T5 모델의 효과를 입증했습니다. 모델은 제한된 데이터로부터 처음부터 사전 학습(pre-trained from scratch)되었습니다.

- **Performance Highlights**: 제안된 모델은 제한된 하위 과제에서 1위를 차지했으며, 제한되지 않은 작업의 상위 성능 수준에 근접했습니다. 이는 역사적 언어에 대한 제한된 데이터를 활용하여 높은 성과를 달성했음을 의미합니다.



### GNN-RAG: Graph Neural Retrieval for Large Language Model Reasoning (https://arxiv.org/abs/2405.20139)
- **What's New**: 본 논문에서는 GNN-RAG라는 새로운 방법을 소개합니다. 이 방법은 대형 언어 모델 (Large Language Models, LLMs)의 자연어 이해 능력과 그래프 신경망 (Graph Neural Networks, GNNs)의 추론 능력을 결합하여 지식 그래프 기반 질문 응답 (Knowledge Graph Question Answering, KGQA)을 개선하는 것에 중점을 둡니다.

- **Technical Details**: GNN-RAG는 두 단계로 구성됩니다. 첫째, GNN이 밀집된 지식 그래프 서브그래프에서 질문에 대한 답변 후보를 추론합니다. 둘째, 질문 엔티티와 답변 후보를 연결하는 가장 짧은 경로를 추출하여 KG 추론 경로로 사용합니다. 이 추출된 경로는 언어화되어 LLM이 RAG(Retrieval-Augmented Generation) 스타일로 추론할 수 있도록 입력됩니다.

- **Performance Highlights**: 실험 결과, GNN-RAG는 WebQSP와 CWQ의 두 가지 널리 사용되는 KGQA 벤치마크에서 GPT-4와 같은 성능을 능가하거나 동등한 성능을 보여주었습니다. 특히 다중 홉 및 다중 엔티티 질문에서 8.9%에서 15.5% 포인트까지 뛰어난 성능을 보였습니다.



### Language Models Need Inductive Biases to Count Inductively (https://arxiv.org/abs/2405.20131)
- **What's New**: 이 논문은 언어 모델이 '숫자 세기(counting)'라는 기초적인 일반화 작업을 어떻게 수행하는지에 대한 광범위한 실험적 결과를 제공하고 있습니다. 저자들은 RNN, Transformer, State-Space Models, RWKV와 같은 다양한 아키텍처를 사용하여 실험을 진행하였습니다.

- **Technical Details**: 실험은 신중하게 설계된 과제 형식, 보조 과제 및 위치 임베딩(포지셔널 임베딩)을 사용하여 OOD(out-of-domain) 데이터의 일반화 한계를 극복하려고 시도했습니다. RNN은 일반적으로 유도적 숫자 세기 작업을 쉽게 수행하는 반면, Transformer는 위치 임베딩에 의존해야 했습니다. 이는 Transformer의 표현력에 대한 기존의 이해를 재검토할 필요가 있음을 시사합니다.

- **Performance Highlights**: 현대적인 RNN은 병렬 학습을 가능하게 하는 디자인 선택 때문에 전통적인 RNN보다 유도적으로 숫자를 세는 성능이 떨어지는 것으로 나타났습니다.



### Divide-and-Conquer Meets Consensus: Unleashing the Power of Functions in Code Generation (https://arxiv.org/abs/2405.20092)
- **What's New**: FunCoder는 복잡한 요구사항을 가진 프로그램을 효과적으로 생성하기 위한 새롭고 혁신적인 코드 생성 프레임워크입니다. 기존 방법의 한계와 문제점을 해결하기 위해 divide-and-conquer 전략과 functional consensus 메커니즘을 도입했습니다.

- **Technical Details**: FunCoder는 코드를 생성할 때 문제를 작은 단위로 나누고, 각 하위 함수를 계층 구조로 분기하여 보다 복잡한 목표를 달성합니다. divide(분할) 과정에서는 문제를 점차적으로 나눠가며 각 하위 목표를 해결하는 새로운 함수를 소개하고, conquer(정복) 과정에서는 작은 함수들을 통합하여 복잡한 문제를 해결합니다. 이 과정에서 sub-function 간의 오류 전파를 줄이기 위해 기능적 합의(consensus)를 도입하여 프로그램의 신뢰성을 높입니다.

- **Performance Highlights**: FunCoder는 HumanEval, MBPP, xCodeEval 및 MATH와 같은 다양한 코드 생성 벤치마크에서 GPT-3.5와 GPT-4를 사용하여 기존 방법들에 비해 평균 +9.8%의 성능 향상을 보여주었습니다. 또한, StableCode-3b와 같은 소규모 모델에서도 뛰어난 성과를 나타내며, HumanEval에서 GPT-3.5를 +18.6% 초과하고 GPT-4의 성능의 97.7%를 달성했습니다.



### The Fine-Tuning Paradox: Boosting Translation Quality Without Sacrificing LLM Abilities (https://arxiv.org/abs/2405.20089)
Comments:
          Accepted to ACL 2024 (long, main)

- **What's New**: 이번 arxiv 논문에서는 대형 언어 모델(LLMs)의 기계 번역 성능을 개선하기 위해 미세 조정(fine-tuning)을 시도하여 번역 품질의 전반적인 향상을 확인하였으나, 몇 가지 LLM 고유의 특성이 악화되는 현상을 발견하였습니다. 특히, 형식성 조정(formality steering) 및 문서 수준 번역(document-level translation)과 같은 특성이 저하되었으나, 병렬 데이터를 사용하는 미세 조정도 비문학적 번역에서의 성능 향상을 가져왔습니다. 또한, 병렬 데이터와 단일 언어 데이터를 혼합하여 미세 조정을 수행하면 이러한 저하를 방지하면서 전반적인 번역 성능을 향상시킬 수 있다는 것을 보여주었습니다.

- **Technical Details**: 논문에서는 LLaMA와 Falcon 모델을 사용하여 7억에서 650억의 매개변수를 가진 모델을 평가하였습니다. 89K에서 1.4M의 데이터 세트를 사용하여 미세 조정을 수행했으며, 특히 병렬 데이터를 사용할 때는 LLM의 고유한 특성들이 저하되는 경향이 두드러졌습니다. 이를 막기 위해 병렬 데이터와 단일 언어 데이터를 혼합한 미세 조정 전략을 제안하였으며, COMET와 같은 메트릭으로 전반적인 번역 품질을 평가하였습니다.

- **Performance Highlights**: 미세 조정을 통해 전반적인 번역 품질이 향상되었지만, 형식성 조정능력, 기술 번역 능력, 문서 수준에서의 번역 능력 등이 저하되는 결과를 보였습니다. 그럼에도 불구하고 비문학적 번역의 경우는 성능이 개선되었습니다. 새로운 평가 데이터셋 IdiomsInCtx-MT를 도입하여 비문학적 번역 성능을 측정했으며, 이는 처음으로 문맥 내의 관용구 및 그 번역을 포함하는 데이터셋입니다.



### Student Answer Forecasting: Transformer-Driven Answer Choice Prediction for Language Learning (https://arxiv.org/abs/2405.20079)
Comments:
          Accepted as a poster paper at EDM 2024: 17th International Conference on Educational Data Mining in Atlanta, USA

- **What's New**: MCQStudentBert이라는 답안 예측 모델이 발표되었습니다. 이 모델은 대형 언어 모델(LLMs)의 기능을 활용하여 학생의 응답 히스토리 및 질문과 답변의 텍스트를 통합하여 예측을 수행합니다. 이를 통해 교육자들은 모델을 다시 학습하지 않고도 새로운 답변 선택지를 추가하거나 기존 선택지를 제거할 수 있습니다.

- **Technical Details**: MCQStudentBert는 다층 퍼셉트론(MLP), 장단기 기억 네트워크(LSTM), BERT, 그리고 Mistral 7B와 같은 여러 아키텍처를 비교하여 학생 임베딩을 생성합니다. 생성된 임베딩은 세밀 조정된 BERT 답안 예측 메커니즘에 통합됩니다. 이 모델은 학습자의 이전 상호작용 데이터를 사용하여 특정 질문에 대한 답변 선택지를 예측합니다.

- **Performance Highlights**:  모델은 10,499명의 학생 데이터를 사용하여 평가되었으며, 237개의 고유한 질문을 포함하고 있습니다. 연구는 독일어 교과목을 중심으로 이루어졌으며, 이 모델은 기존의 정답 예측 및 전통적인 마스터리 학습 기반 접근법과 비교하여 예측 정확도를 평가하였습니다. 모델은 교육 데이터를 풍요롭게 하고, 모듈화 및 세분화된 지원을 제공하는 데에 기여할 수 있습니다.



### Would I Lie To You? Inference Time Alignment of Language Models using Direct Preference Heads (https://arxiv.org/abs/2405.20053)
- **What's New**: 새로운 Direct Preference Heads (DPH) 프레임워크가 도입되었습니다. 이 프레임워크는 언어 모델(LM)이 언어 생성 헤드의 결과 분포를 직접적으로 변경하지 않고 보조 보상 헤드를 통해 인간의 선호 신호를 학습할 수 있도록 합니다. 이를 통해 RLHF(인간 피드백을 통한 강화 학습)의 장점을 유지하면서도 논리적 추론 능력의 저하나 잘못된 사실을 생성하는 문제를 해결하려 합니다.

- **Technical Details**: DPH는 보상 모델을 학습하여 출력을 평가하고 가장 높은 점수를 받은 출력을 선택하도록 만들어졌습니다. 이를 위해 DPH는 LM의 중간 표현에 의해 조건이 결정된 숨겨진 상태와 이를 변형하는 풀링 함수, 학습 가능한 벡터 wdph(w_dph)을 사용합니다. 보상은 시퀀스의 마지막 토큰에서 출력되는 마지막 트랜스포머 레이어의 출력으로부터 계산됩니다. 실험에서는 세 가지 풀링 함수를 사용하였습니다: (1) 아이덴티티 매핑, (2) 학습 가능한 어파인 투영, (3) Inverted Bottleneck FFN.

- **Performance Highlights**: GLUE, RACE, 및 GPT4All 평가에서 DPH를 사용한 모델이 SFT(Supervised Fine-Tuning) 또는 DPO(Direct Preference Optimization)만 사용한 모델보다 더 높은 점수를 달성했습니다. 이는 DPH가 LM의 출력을 인간 선호도에 더 잘 맞추면서도 추론 능력을 유지하는 데 효과적임을 보여줍니다.



### Improved Out-of-Scope Intent Classification with Dual Encoding and Threshold-based Re-Classification (https://arxiv.org/abs/2405.19967)
- **What's New**: 이번 연구에서는 태스크 지향 대화(task-oriented dialogue) 및 의도 분류(intent classification)에서 아웃 오브 스코프(out-of-scope) 유저 발화를 효과적으로 탐지하기 위한 새로운 방법인 듀얼 인코더 기반 문턱 재분류(Dual Encoder for Threshold-Based Re-Classification, DETER)를 제안합니다. 이 방법은 데이터 분포에 대한 가정을 필요로 하지 않으며 추가적인 후처리 단계 없이 효율적으로 아웃 오브 스코프 의도를 탐지할 수 있습니다.

- **Technical Details**: DETER는 듀얼 텍스트 인코더인 Universal Sentence Encoder(USE)와 Transformer 기반 Denoising AutoEncoder(TSDAE)를 사용하여 사용자 발화 임베딩을 생성합니다. 이를 통해 분기된 신경망 아키텍처에서 분류가 이루어집니다. 또한, 스스로 감독(self-supervision)을 통해 합성(outlier) 아웃라이어를 생성하고, 오픈 도메인 데이터셋에서 아웃 오브 스코프 문구를 반영하여 포괄적인 학습 세트를 구축합니다. 문턱 기반의 재분류 메커니즘을 통해 모델 초기 예측의 정밀도를 높입니다.

- **Performance Highlights**: CLINC-150, Stackoverflow, Banking77 데이터셋에서 DETER의 성능을 평가한 결과, 기존의 벤치마크를 뛰어넘는 성능을 보였습니다. DETER는 CLINC-150과 Stackoverflow에서 알려진 의도에 대해 최대 13%, 미지 의도에 대해 5%의 F1 점수 향상을, Banking77에서는 각각 16%, 24%의 향상을 기록했습니다. 이 모델은 기존의 대규모 모델과 비교해 훨씬 적은 1.5백만 개의 학습 가능 매개변수를 사용하므로, 성능을 저하시키지 않으면서도 계산 효율성과 확장성을 높였습니다.



### Multi-Aspect Controllable Text Generation with Disentangled Counterfactual Augmentation (https://arxiv.org/abs/2405.19958)
Comments:
          Accepted in the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024)

- **What's New**: MAGIC이라는 새로운 다중 측면 제어 텍스트 생성 방법이 소개되었습니다. 이 방법은 속성들 간의 불균형한 상관관계를 해결하기 위해 반사실적 속성 벡터를 활용하여 텍스트를 생성합니다. MAGIC은 속성 상관관계를 강화하여 다중 측면에서 텍스트를 더 잘 제어할 수 있도록 합니다.

- **Technical Details**: MAGIC은 속성 잠재 공간(attribute latent space)에서의 분리(disentanglement) 및 반사실적(feature) 증강(counterfactual augmentation)을 활용합니다. 학습 중에는 속성 상관관계의 균형을 맞추기 위해 반사실적 속성 벡터를 사용하며, 추론 시에는 목표 지향 반사실적 증강(target-guided counterfactual augmentation)으로 속성 상관관계를 강화합니다. MAGIC은 감정(sentiiment), 주제(topic), 독성(detoxification)의 세 측면에서 텍스트를 제어할 수 있습니다.

- **Performance Highlights**: 실험 결과, MAGIC은 불균형한 속성 상관관계 시나리오와 균형잡힌 상황 모두에서 최고 성능을 기록하며 기존 방법들을 능가함을 보여줍니다. MAGIC의 각 모듈의 효과는 추가적인 분석 실험을 통해 입증되었습니다.



### Is In-Context Learning Sufficient for Instruction Following in LLMs? (https://arxiv.org/abs/2405.19874)
Comments:
          Preprint. Code at this https URL

- **What's New**: 최근 논문에서는 In-context learning(ICL)을 이용하여 대형 언어 모델(LLM)이 예시를 통해 무게를 변경하지 않고 배울 수 있는 방법에 대해 다룹니다. Lin et al. (2024)는 베이스 LLM을 정렬하기 위해 세 가지 예시만 사용하는 URIAL 방법을 제안했습니다. 이 논문은 URIAL 방법이 효과적이지만 Instruction fine-tuning(명령 미세 조정)보다 성능이 낮다고 지적합니다. 이를 개선하기 위해 그리디 선택 알고리즘을 사용하여 ICL 예시를 최적화하는 방법을 제안합니다.

- **Technical Details**: ICL은 입력으로 제공된 예시를 통해 모델이 배울 수 있도록 하는 개념으로, 베이스 모델의 무게를 변경하지 않고도 사용자 지시를 따르게 하는 간단하고 유연한 방법입니다. 이번 연구에서는 Lin et al. (2024)에서 제안한 URIAL 전략을 다양한 베이스 모델에서 시험했습니다. 또한, 많은 수의 ICL 예시를 추가하는 방법과 그리디 알고리즘을 사용하여 최적의 예시를 선택하는 방법을 제안했습니다.

- **Performance Highlights**: URIAL 방법은 기본 명령 미세 조정된 모델에 비해 성능이 낮지만, 그리디 알고리즘을 통해 선택된 예시들을 추가하면 성능이 크게 향상됩니다. 이 방법은 Mistral-7B-v0.2와 같은 모델에서 명령 미세 조정된 모델과 유사한 성능을 보이며, MT-Bench 점수도 눈에 띄게 개선되었습니다.



### DevEval: A Manually-Annotated Code Generation Benchmark Aligned with Real-World Code Repositories (https://arxiv.org/abs/2405.19856)
Comments:
          Accepted by the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024). arXiv admin note: substantial text overlap with arXiv:2404.00599, arXiv:2401.06401

- **What's New**: 최근 arxiv에 발표된 논문에서 코딩 능력을 평가하는 새로운 벤치마크, DevEval을 제안했습니다. 기존 벤치마크가 실제 코드 저장소(real-world code repositories)와 잘 맞지 않으며, LLMs(Large Language Models)의 코딩 능력을 평가하기에 충분하지 않다는 문제를 지적합니다. DevEval은 이러한 문제를 해결하기 위해 13명의 개발자가 주석을 달았으며, 117개의 저장소에서 1,874개의 테스트 샘플을 포함하고 있습니다.

- **Technical Details**: DevEval은 여러 면에서 실제 코드 저장소와 정렬되어 있습니다. 코드 분포(code distributions), 의존성 분포(dependency distributions) 등의 측면을 포함하여, 실제 환경과 잘 맞아떨어지도록 설계되었습니다. 특히, DevEval은 자연어 요구사항(requirements), 원본 저장소(original repositories), 참고 코드(reference code), 참고 의존성(reference dependencies) 등의 포괄적인 주석을 포함하고 있습니다. 이를 통해 실제 코딩 과정에서 모델의 성능을 보다 현실적으로 평가할 수 있습니다.

- **Performance Highlights**: DevEval을 이용해 GPT-4, GPT-3.5, StarCoder 2, DeepSeek Coder, CodeLLaMa 등의 8개의 인기 있는 LLMs을 평가했습니다. 예를 들어, GPT-4 터보의 Pass@1 점수는 HumanEval에서 80%였으나, DevEval에서는 53.04%로 낮게 나왔습니다. 이는 LLMs의 실제 코드 저장소에서의 코딩 능력을 보여줍니다. 이러한 평가를 통해 LLMs의 단점을 분석하고 요약했습니다.



### Quest: Query-centric Data Synthesis Approach for Long-context Scaling of Large Language Mod (https://arxiv.org/abs/2405.19846)
- **What's New**: 최근 발표된 연구에서는 초기 훈련 시 제한된 컨텍스트 길이를 가진 대형 언어 모델이 확장된 컨텍스트를 가진 데이터로 추가 훈련되었을 때 더 긴 텍스트를 잘 처리할 수 있다는 사실을 발견했습니다. 이에 따라 효과적인 장문 컨텍스트 데이터를 얻기 위해 Query-centric 데이터 합성 방법인 Quest 방식을 제안하였습니다.

- **Technical Details**: Quest는 유사한 쿼리(query)로 검색된 문서들이 관련성이 높으면서도 중복성이 낮다는 관찰에 기반한 해석 가능한 방법입니다. 이 방법은 스케일러블하여 대량의 장문 컨텍스트 데이터를 구축할 수 있습니다. Quest를 이용하여 128k 컨텍스트 길이까지 장문 컨텍스트 데이터를 합성하였으며, 이는 여러 장문 테스트 벤치마크에서 뛰어난 성과를 나타냈습니다.

- **Performance Highlights**: Quest 방법을 사용하여 합성된 데이터셋은 다른 데이터 합성 방법들보다 훨씬 뛰어난 성과를 나타냈습니다. 또한, 스케일링 법칙 실험을 통해 Quest 방법의 예측 가능성을 검증하였으며, 이를 통해 장문 컨텍스트 모델을 발전시키는 신뢰할 수 있는 솔루션으로 입증되었습니다.



### Improve Student's Reasoning Generalizability through Cascading Decomposed CoTs Distillation (https://arxiv.org/abs/2405.19842)
- **What's New**: 새로운 연구에서는 소형 언어 모델(SLMs)의 추론 능력을 향상시키기 위해 기존의 교사-학생 학습 방법을 개선한 'Cascading Decomposed CoTs Distillation (CasCoD)'을 제안합니다. 이 접근법은 단일 학습 단계를 두 개의 연쇄된 학습 단계로 분해하여 모델이 사전 설정된 답변의 영향을 받지 않고 논리를 학습하도록 합니다.

- **Technical Details**: CasCoD는 기존의 'standard CoTs distillation (Std-CoT)' 방법을 개선합니다. 기존 방법은 교사 모델이 생성한 연쇄적 사고(Chain-of-Thought, CoT) 데이터를 사용해 학생 모델을 미세 조정하지만, CasCoD는 이를 두 개의 학습 단계로 나눕니다. 첫 번째 단계에서 모델은 질문과 이유(rationale)를 입력으로 받고, 두 번째 단계에서 질문과 이유를 조합해 최종 답을 예측합니다. 이를 통해 모델이 질문과 답변 사이의 잘못된 상관관계(spurious correlations)를 피할 수 있도록 합니다.

- **Performance Highlights**: CasCoD는 인도메인(IND) 및 아웃도메인(OOD) 기준 데이터셋 모두에서 기존 방법보다 우수한 성능을 보였습니다. 광범위한 실험에서 CasCoD는 새로운 도메인에서 더 잘 일반화할 수 있는 연쇄적 사고(CoTs)를 생성해 내는 데 성공했습니다. 이는 OOD 작업에서의 모델 성능 향상 뿐만 아니라 다양한 모델 크기와 훈련 데이터 크기에서도 일정한 성능을 보였습니다.



### Just Rewrite It Again: A Post-Processing Method for Enhanced Semantic Similarity and Privacy Preservation of Differentially Private Rewritten Tex (https://arxiv.org/abs/2405.19831)
Comments:
          10 pages, 2 figures, 2 tables. Accepted to ARES 2024 (IWAPS)

- **What's New**: 이번 연구에서는 Natural Language Processing(NLP)에서 Differential Privacy(DP)를 적용하여 텍스트 프라이버시를 보호하기 위해 가해지는 방법을 개선하고자 했습니다. 기존의 DP 텍스트 재작성 기법을 한 단계 더 발전시켜, 재작성된 텍스트를 다시 재작성(Post-Processing)하는 방식을 도입했습니다. 이 방식은 텍스트의 의미를 원본에 더 가깝게 유지하면서도 프라이버시 평가에서 더 나은 결과를 보여줍니다.

- **Technical Details**: 본 연구는 Large Language Models(LLMs)에서 발생할 수 있는 프라이버시 문제를 해결하는 방법으로서 Differential Privacy(DP) 텍스트 재작성 기법을 사용합니다. 최근 연구에서는 DP를 활용해 민감한 데이터를 재작성하여 프라이버시를 보호하는 메커니즘이 제안되고 있습니다. DP는 수학적으로 데이터 처리 시 프라이버시를 보장하는 청사진을 제공합니다. 본 연구에서는 DP 재작성 텍스트를 다시 재작성(post-processing)하여, 원본 텍스트와의 의미적 유사성을 높이고 더 강력한 프라이버시 보장을 제공합니다.

- **Performance Highlights**: 본 연구의 결과로 DP 재작성 텍스트를 한 단계 더 재작성한 후, 의미적 유사성이 원본과 더 가까운 텍스트가 생성되었으며, 프라이버시 평가에서도 기존 방법보다 더 우수한 성능을 보였습니다. 이는 악의적인 공격자에 대한 추가 보호 층을 제공함으로써 DP 텍스트 재작성 기법의 프라이버시 평가 기준을 높였습니다.



### Unsupervised Mutual Learning of Dialogue Discourse Parsing and Topic Segmentation (https://arxiv.org/abs/2405.19799)
- **What's New**: 대형 언어 모델(LLMs)의 발전은 대화 시스템의 기술 발전을 이끌었습니다. 특히, 비즈니스 분야에서는 사용자의 선호도만을 만족시키는 ChatGPT 같은 어시스턴트 모델과 달리, 과제 지향적 대화 시스템은 대화의 각 턴에서 정확한 응답을 제공하고 동시에 과제의 전체 목표를 달성해야 하는 요구 사항이 있습니다. 이에 따라 수사적 구조와 주제 구조를 이해하는 것이 중요해졌습니다. 본 논문에서는 두 구조 간의 상호 학습과 상호 촉진을 가능하게 하는 새로운 비지도 학습 프레임워크를 제안합니다.

- **Technical Details**: 제안된 프레임워크는 수사적 구조와 주제 구조의 글로벌 및 로컬 연결을 활용하여 두 구조 간의 상호 촉진을 가능하게 합니다. 비지도 학습을 기반으로 두 구조를 상호 학습시키기 위해 그래프 신경망(Graph Neural Network, GNN) 모델을 사용하며, 토픽 구조 간의 비인접 담화 단위 사이의 글로벌 구조적 관련성을 보장합니다. 또한, 주제 구조에 수사적 구조를 통합하여 로컬 일관성을 유지합니다. 최종적으로는 두 융합 구조 간의 유사성을 이용하여 상호 학습을 수행합니다.

- **Performance Highlights**: 제안된 방법은 두 가지 대화 수사 데이터셋(STAC 및 Molweni)과 두 가지 대화 주제 데이터셋(Doc2Dial 및 TIAGE)에서 기존의 강력한 기준선보다 뛰어난 성능을 보였습니다. 실험 결과는 제안된 두 가설의 타당성을 입증하고, 방법의 전송 가능성을 보여주었습니다.



### SLM as Guardian: Pioneering AI Safety with Small Language Models (https://arxiv.org/abs/2405.19795)
- **What's New**: 이 논문에서는 보다 작은 LLM(sLLM)을 사용하여 사용자로부터의 유해한 질의를 감지하고 안전한 응답을 생성하는 방법을 제안합니다. 기존 대형 언어 모델(LLM)의 안전성을 강화하는 연구는 훈련 비용 증가와 유용성 저하와 같은 문제를 야기했으나, 저자는 더 작은 모델을 사용한 모듈식 접근 방식을 통해 이를 해결하고자 합니다.

- **Technical Details**: 저자들은 sLLM을 활용해 유해한 질의를 감지하고 이들에 대한 안전한 응답을 생성하는 멀티태스크 학습(multi-task learning) 메커니즘을 제안합니다. 또한, 유해함의 범주를 정리한 분류법과 관련된 다양한 사용 사례를 제시하고, 한국어를 포함한 다양한 언어에서 안전성을 검증하는 데이터를 만들었습니다. 

- **Performance Highlights**: 제안된 방법은 공용 LLM들과 비교했을 때 유해한 질의를 감지하고 안전한 응답을 생성하는 성능 면에서 동등하거나 더 우수함을 입증했습니다. 특히 한국어와 같은 자원이 제한된 언어에서 높은 정확도를 보였습니다.



### PDDLEGO: Iterative Planning in Textual Environments (https://arxiv.org/abs/2405.19793)
Comments:
          In *SEM 2024

- **What's New**: 신규 제안된 PDDLEGO는 부분적으로 관찰된 환경에서 목표를 달성하기 위한 계획 표현을 반복적으로 구성하는 방법론입니다. 이를 통해 다양한 하위 목표를 달성함으로써 점차적으로 환경에 대한 정보를 획득하고, 궁극적으로 최종 목표를 달성합니다.

- **Technical Details**: PDDLEGO는 LLM들과 결합된 신경심볼(neurosymbolic) 접근법을 사용하여 부분적으로 관찰된 환경에서 점진적으로 PDDL 계획 문제 파일을 생성합니다. 초기 관찰과 목표 상태가 주어지면 LLM은 이를 기반으로 초기 문제 파일을 생성시키고, 필요한 정보가 부족할 경우 하위 목표로 대체하여 계획을 찾습니다. 이 과정은 목표 상태를 달성할 때까지 반복됩니다.

- **Performance Highlights**: PDDLEGO는 Coin Collector 시뮬레이션에서 직접 계획을 생성하는 LLM에 비해 계획 효율성이 43% 더 높았습니다. 또한, 더 복잡한 Cooking World 시뮬레이션에서는 직관적인 실행 계획을 생성하지 못하는 LLM들이 4% 성공률을 기록한 것에 비해, PDDLEGO는 98%의 성공률을 달성했습니다.



### From Symbolic Tasks to Code Generation: Diversification Yields Better Task Performers (https://arxiv.org/abs/2405.19787)
- **What's New**: 새로운 연구는 인스턱션 튜닝(instruction tuning)을 통해 대형 언어 모델(LLMs)의 실제 적용성을 높이는 방법을 탐구합니다. Markov 알고리즘을 이용해 인스턱션 데이터를 세밀히 조정하고, 코드 생성 등의 실제 응용 시나리오에 적용했을 때 더 다양한 인스턱션 셋이 성능을 향상시킨다는 결과를 얻었습니다.

- **Technical Details**: 이 연구는 먼저 문자열 교체(string replacements)의 인위적인 실험을 통해 데이터의 다양성이 모델의 일반화(generalization)와 강인성(ROBUSTNESS)에 미치는 영향을 조사합니다.  이 실험은 Markov 알고리즘이라는 튜링 완전(Turing-complete) 모델을 사용하며, 이는 문자열 교체 규칙을 사용하는 전통적인 이론 모델입니다. 이를 통해 다양한 인스턱션 집합이 제공될 경우, 각 작업에 대한 예시가 매우 적음에도 불구하고 일반화된 성능이 나타난다는 결과를 얻었습니다.

- **Performance Highlights**: 다양한 인스턱션 셋을 통해 모델의 성능 및 강인성이 증가한다는 주요 결과를 확인했습니다. 코드 생성 같은 실제 응용에서 코드 관련 작업을 넘어서 다양한 인스턱션을 포함시켰을 때 모델의 성능이 크게 향상되었습니다. 이는 모델이 보다 폭넓은 의미망(semantic space)을 가지게 되어, 지시사항을 이해하고 이를 수행하는 능력이 크게 증가했음을 시사합니다.

- **Related Works**: 기존 연구들은 인스턱션 튜닝을 위한 다양한 데이터셋을 제안했으며, 데이터셋의 크기와 품질이 미치는 영향을 연구했습니다. 일부 연구에서는 데이터셋 간 일관된 형식과 다양한 작업을 혼합하는 방법이 성과를 높일 수 있다는 것을 강조했습니다. 이 연구는 일반 및 도메인 별 인스턱션을 혼합하는 방법이 성과를 극대화할 수 있다는 점에서 기존 연구와 유사합니다.



### Enhancing Consistency and Role-Specific Knowledge Capturing by Rebuilding Fictional Character's Persona (https://arxiv.org/abs/2405.19778)
Comments:
          preprint

- **What's New**: 새로운 연구는 CharacterGPT라는 혁신적인 페르소나 재구성 프레임워크(Character Persona Reconstruction Framework)를 소개합니다. 이는 기존 Assistants API의 한계를 보완하여 문서 기반 언어 모델(Document-based language model)이 일관된 페르소나를 유지할 수 있도록 돕습니다. CharacterGPT는 소설의 요약에서 등장인물의 특성을 추출하고 지속적으로 업데이트하는 Character Persona Training(CPT) 과정을 활용합니다.

- **Technical Details**: CharacterGPT는 소설의 각 장(chapter)에서 등장인물의 총 8가지 특성(Personality, Physical Description, Motivations, Backstory, Emotions, Relationships, Growth and Change, Conflict)을 추출하여 페르소나 문서(Character Persona Document)에 연대순으로 추가합니다. 이 프레임워크는 페르소나 문서의 정보 손실과 연산 비용을 최소화하며, 사용자는 특정 시점의 등장인물과 대화할 수 있습니다. 또한, Big Five Inventory 성격 테스트를 통한 성격 분석 및 단편 소설 생성 실험도 포함되어 있습니다.

- **Performance Highlights**: 광범위한 인간 평가를 통해 CharacterGPT는 페르소나 일관성(Persona Consistency), 제어성(Controllability) 및 역할 고유의 지식 활용(Role-specific knowledge utilization)에서 우수한 성능을 입증했습니다. 7명의 크라우드 워커를 고용하여 5점 리커트 척도를 사용한 6가지 메트릭스 평가 결과, CharacterGPT는 매우 긍정적인 평가를 받았습니다.



### Enhancing Reinforcement Learning with Label-Sensitive Reward for Natural Language Understanding (https://arxiv.org/abs/2405.19763)
Comments:
          Accept at ACL2024 Main

- **What's New**: 최근 인공지능 언어 모델(LLMs)에서 인간 피드백을 통한 강화 학습(RLHF)을 활용한 성능 향상이 두드러지고 있습니다. 그러나 RLHF는 자연어 이해(NLU) 작업에서 최적의 성능을 발휘하지 못하는 문제를 가지고 있습니다. 이를 해결하기 위해 우리는 레이블 민감형 보상(Label-sensitive Reward)을 포함한 새로운 강화 학습 프레임워크(RLLR)를 제안하여 LLM의 NLU 작업 성능을 크게 향상시켰습니다.

- **Technical Details**: 이 연구는 레이블 민감형 쌍을 강화 학습에 도입하여 의미적으로 미묘한 레이블 민감형 특징을 포착하는 방법을 제안합니다. 우선, GPT-4를 활용해 훈련 데이터의 골드 레이블에 해당하는 추론(rationale)을 생성하고, 이를 통해 슈퍼바이즈드 파인 튜닝(SFT) 모델을 훈련시킵니다. 그런 다음, 잘못된 레이블에 대한 추론을 생성하고, 레이블의 정확성을 기준으로 레이블 민감형 쌍을 자동으로 구성하여 보상 모델을 훈련합니다. 최종적으로, Proximal Policy Optimization(PPO)을 통해 정책 모델을 레이블 민감형 보상 모델을 기반으로 최적화합니다.

- **Performance Highlights**: 우리의 RLLR 방법은 8개의 NLU 작업에서 평균 1.54%의 성능 향상을 보여주었으며, RLHF 모델과 비교했을 때 평균 0.69%의 성능 향상을 나타냈습니다. 이는 LLM에 있어서 NLU 작업의 효율성을 크게 개선한 결과로 평가됩니다.



### X-Instruction: Aligning Language Model in Low-resource Languages with Self-curated Cross-lingual Instructions (https://arxiv.org/abs/2405.19744)
Comments:
          ACL 2024. Our codes, data and model weights are available at this https URL

- **What's New**: 최근 대형 언어 모델은 영어와 같은 고자원 언어에서는 훌륭한 성능을 보이지만, 저자원 언어에서는 그 성능이 저조한 경향이 있습니다. 이에 'X-Instruction'이라는 대규모 다국어 학습 데이터를 구축하는 새로운 방법론을 제안하였습니다. 이 방법론은 영어로 된 지시문(instruction)을 기반으로 저자원 언어로 된 응답을 생성하는 형식으로, 기존의 단순 번역 방식을 탈피하여 언어 특유의 지식과 표현을 더 잘 반영하고 있습니다.

- **Technical Details**: 모델 학습을 통해 자연스럽게 생성된 웹 텍스트를 기반으로 영어 지시문을 생성하고, 후보 다국어 학습 샘플을 더 감식하고 다변화하는 자동 파이프라인을 제안합니다. 먼저 모델은 시드 데이터(seed data)를 학습하여 적절한 영어 지시문을 생성하고, 각 반복(iteration)마다 평가기를 이용해 더 나은 다국어 샘플을 뽑아내는 방식입니다. 이처럼 반복적인 개선을 통해 샘플의 품질을 높였습니다.

- **Performance Highlights**: 제안된 X-Instruction 데이터셋으로 튜닝된 모델은 기존 번역 및 증류(distillation) 기반 모델들과 비교하여 응답 품질이 더욱 높으며, ChatGPT와 비슷하거나 그 이상의 성능을 보여주고 있습니다. 10개의 언어 실험에서 32만 개의 고품질 크로스 링귀얼 인스트럭션 튜닝 샘플을 생성하여, 모델이 추가 튜닝 없이도 지시된 언어로 명령을 따를 수 있는 능력을 확인하였습니다.



### PertEval: Unveiling Real Knowledge Capacity of LLMs with Knowledge-Invariant Perturbations (https://arxiv.org/abs/2405.19740)
Comments:
          23 pages, 12 figures, 10 tables

- **What's New**: PertEval이라는 새로운 도구가 도입되었습니다. 이 도구는 LLM(Large Language Model)의 진정한 지식 능력을 평가하기 위해 고안되었으며, 지식에 영향을 미치지 않는 변형(knolwedge-invariant perturbations)을 사용합니다. 기존 벤치마크 테스팅의 한계를 극복하고 LLM의 진정한 능력을 정확하게 측정하는 것을 목표로 합니다.

- **Technical Details**: PertEval는 인간처럼 문장을 다시 표현하는 기술(human-like restatement techniques)을 사용하여 원래의 지식 콘텐츠를 변경하지 않고 지식에 중요하지 않은 세부 사항을 변경합니다. 이를 통해 기존의 고정된 테스트 데이터셋을 동적으로 변환하여 LLM이 여러 테스트 시나리오에 어떻게 반응하는지를 평가합니다. 또한, 원시 데이터와 변형된 데이터에 대한 성능 차이를 분석하여 LLM의 진정한 지식 능력을 평가하는 전환 분석(transition analyses) 기능도 포함하고 있습니다.

- **Performance Highlights**: 6개의 최첨단 LLM이 PertEval을 사용해 재평가되었으며, 이 결과 기존 벤치마크에서의 성능이 과대평가된 것으로 나타났습니다. 예를 들어 GPT-4의 경우 절대적으로 21% 까지 과대평가가 있었으며, 다른 모델들도 유사한 경향을 보였습니다. 또한, LLM의 회상 패턴 분석을 통해 일부 모델들이 올바른 선택지를 암기하며 성능을 부풀렸다는 사실도 밝혀졌습니다. PertEval을 통해 얻은 결과는 LLM의 기존 지식 마스터리에 대한 약점을 드러내고, 모델 개선에 대한 지침을 제공할 수 있습니다.



### Beyond Imitation: Learning Key Reasoning Steps from Dual Chain-of-Thoughts in Reasoning Distillation (https://arxiv.org/abs/2405.19737)
- **What's New**: 대형 언어 모델(LLMs)의 체인-오브-생각(Chain-of-Thought, CoTs) 추론 능력을 소형 언어 모델(SLMs)로 압축하기 위한 새로운 방법, 즉 오류 기반 핵심 추론 단계 증류(EDIT)가 제안되었습니다.

- **Technical Details**: EDIT 기법은 학생 모델(SLMs)이 교사 모델(LLMs)의 데이터에서 얻은 추론 데이터를 단순히 미세 조정(supervised fine-tuning)하는 대신, CoTs에서 중요한 추론 단계를 학습할 수 있도록 설계되었습니다. 이를 위해, 유사한 중간 추론 단계를 가지지만 결론이 다른 이중 CoTs 데이터를 생성하는 특정 프롬프트를 디자인하고, 최소 편집 거리(minimum edit distance) 알고리즘을 적용해 중요한 단계를 식별합니다.

- **Performance Highlights**: 다양한 실험을 통해 EDIT가 기존 방식보다 높은 성능과 일반화 능력을 보여줬으며, 논리적 오류가 지식 또는 수학적 계산 오류보다 유리함을 확인했습니다. 인-도메인(in-domain) 및 아웃-오브-도메인(out-of-domain) 벤치마크 추론 데이터셋에서 높은 품질의 CoTs를 생성할 수 있으며, 정교한 손실 함수(fine-grained loss function)를 활용해 중요한 추론 단계의 가능성을 최적화했습니다.



### SpecDec++: Boosting Speculative Decoding via Adaptive Candidate Lengths (https://arxiv.org/abs/2405.19715)
- **What's New**: SpecDec++는 추론 지연시간을 줄이기 위한 향상된 추측적 디코딩(speculative decoding) 방법입니다. 이 방법은 적응적으로 후보 길이(candidate length)를 조절하여 성능을 최대화합니다. 기존 방법들은 간단한 휴리스틱을 사용하여 후보 길이인 K를 선택했으나, 이 논문에서는 이를 마코프 결정 프로세스(Markov Decision Process, MDP)로 공식화하여 최적 정책을 이끌어냈습니다.

- **Technical Details**: SpecDec++는 후보 토큰 수용 확률을 예측하는 훈련된 수용 예측 헤드를 초안 모델에 추가합니다. MDP 이론에 기반하여, 최소 하나의 토큰 거부 확률이 임계값을 초과할 때 추측을 멈추고 검증을 시작합니다. 훈련 과정에서는 클래스 불균형 문제를 해결하기 위해 가중 바이너리 교차 엔트로피(binary cross-entropy) 손실을 적용하며, BERT에서 사용하는 랜덤 마스킹(random masking) 아이디어를 채택하여 훈련 효율성을 높였습니다.

- **Performance Highlights**: Alpaca 데이터셋에서 SpecDec++는 2.04배 속도 향상을 달성했으며(기존 방법 대비 7.2% 추가 개선), GSM8K와 HumanEval 데이터셋에서는 각각 2.26배(9.4% 개선), 2.23배(11.1% 개선) 속도 향상을 보였습니다. 이 방법은 llama-2-chat 7B & 70B 모델 페어에 적용되어 뛰어난 성능을 입증했습니다.



### Significance of Chain of Thought in Gender Bias Mitigation for English-Dravidian Machine Translation (https://arxiv.org/abs/2405.19701)
- **What's New**: 이번 연구는 텔루구어와 칸나다어 같은 드라비다어에 대한 기계 번역 시스템의 성별 편향을 분석합니다. Google Translate와 ChatGPT를 사용해 성별 굴절이 번역 정확성과 중립성에 미치는 영향을 평가했습니다. 복수형이 편향을 줄일 수 있지만, 개별 중심 문장은 역사적 고정관념 때문에 편향을 유지하는 모습을 보였습니다. Chain of Thought(사고의 연쇄) 처리를 통해 텔루구어에서 80%에서 4%, 칸나다어에서 40%에서 0%로 편향 완화가 이루어졌습니다.

- **Technical Details**: 성별 굴절이 기계 번역에 미치는 영향은 특히 성별이 중요한 문법적 역할을 하는 언어에서 두드러집니다. 텔루구어와 칸나다어에서 성별 굴절과 관련된 오류나 편향이 발생할 수 있으며, 문화적 이해 부족이 이 문제를 심화시킬 수 있습니다. 텔루구어에서 단수 형태는 남성형과 비남성형으로 분류되며, 복수 형태는 인간과 비인간으로 구분됩니다. 칸나다어에서는 남성형, 여성형, 중립형의 명확한 3성(gender) 체계가 있습니다.

- **Performance Highlights**: 구글 번역과 ChatGPT가 영어에서 텔루구어와 칸나다어로 번역할 때 초기 조사 결과, 개별 중심 문장이 성별 편향을 더 많이 보였습니다. 복수형 문장은 상대적으로 중립적이었으며, 구글 번역이 ChatGPT보다 성별 중립성을 더 잘 유지했습니다. Chain of Thought 접근법을 통해 다단계 추론을 적용하여 성별 편향을 줄이는 데 성공했습니다.



### One Token Can Help! Learning Scalable and Pluggable Virtual Tokens for Retrieval-Augmented Large Language Models (https://arxiv.org/abs/2405.19670)
Comments:
          working in progress, repo: this https URL

- **What's New**: 본 논문에서는 리트리벌 기반 생성(RAG)을 위한 새로운 방법인 SPRING을 제안합니다. SPRING은 대형 언어 모델(LLM)의 기존 파라미터를 수정하지 않고, 플러그형 가상 토큰(virtual tokens)의 임베딩만을 학습하여 RAG 성능을 향상시키는 방식입니다. 이를 통해 LLM의 일반 생성 능력을 유지하면서도 성능을 높일 수 있습니다.

- **Technical Details**: SPRING 방법에서는 LLM의 기존 파라미터를 동결(freeze)하고, 추가된 가상 토큰의 임베딩만을 미세 조정합니다. 이를 통해 LLM이 검색된 정보를 효과적으로 이해하고 사용자 입력과의 관련성을 높일 수 있습니다. SPRING은 다양한 훈련 전략을 사용하여 확장성, 유연성 및 일반화를 개선하였으며, 필요에 따라 조정 가능한 가상 토큰의 수를 도입할 수 있습니다.

- **Performance Highlights**: SPRING은 Mistral-7b, LLaMA-2-7b, Phi-3-4b, QWen-1.8b 모델을 대상으로 9개의 질문-응답(QA) 데이터셋에서 실험되었습니다. 그 결과, SPRING은 LLM의 RAG 성능을 크게 향상시키면서도 일반 생성 능력을 유지하는 데 성공했습니다. 특히, 가상 토큰을 추가하는 것이 EM과 F1 점수를 평균적으로 33% 이상, 12% 이상 향상시켰습니다. 또한, SPRING은 플러그 앤 플레이 방식으로 적용 가능하며, 다양한 검색 결과 수에 대해 견고한 성능을 보였습니다.



### PATIENT-{\Psi}: Using Large Language Models to Simulate Patients for Training Mental Health Professionals (https://arxiv.org/abs/2405.19660)
Comments:
          Work in progress

- **What's New**: 이번 연구에서는 정신 건강 훈련을 위해 새로운 환자 시뮬레이션 프레임워크인 PATIENT-Ψ (Patient-Psi)를 제안했습니다. 이 시스템은 대규모 언어 모델(large language models, LLMs)과 인지 행동 치료(cognitive behavior therapy, CBT) 원칙에 기반한 다양한 환자 프로파일을 이용해 가상 환자를 생성합니다.

- **Technical Details**: PATIENT-Ψ는 CBT의 인지 모델을 이용하여 다양한 정신 건강 장애를 가진 환자를 시뮬레이션합니다. 이 시스템은 우울증 및 불안 장애와 같은 문제를 포함하는 고품질의 다양한 환자 인지 모델을 통해 구축되었습니다. 또한, Trainee들이 인지 모델을 구성하는 연습을 할 수 있도록 상호작용 훈련 스킴 PATIENT-Ψ-TRAINER를 제안했습니다. 이 시스템은 환자와의 역할 놀이를 통해 실시간으로 피드백을 제공합니다.

- **Performance Highlights**: 사용자 연구 결과, PATIENT-Ψ-TRAINER를 사용한 훈련이 텍스트북, 비디오, 타인과의 롤플레이 등 기존의 학습 방법에 비해 훨씬 더 높은 숙련도와 자신감을 향상시킵니다. 전문가들은 PATIENT-Ψ가 실제 환자와의 상호작용에 더 가까우며, GPT-4보다 더 높은 유사성을 보인다고 평가했습니다. 또한, 훈련생들은 PATIENT-Ψ-TRAINER를 통해 실제 환자와의 상호작용을 대비하는 데 큰 도움이 된다고 응답했습니다.



### Detecting Hallucinations in Large Language Model Generation: A Token Probability Approach (https://arxiv.org/abs/2405.19648)
Comments:
          ICAI'24 - The 26th Int'l Conf on Artificial Intelligence

- **What's New**: 최신 연구에 따르면, 대규모 언어 모델(LLMs)의 정확하지 않은 출력, 즉 '환상'(hallucinations)을 감지하는 새로운 방법이 소개되었습니다. 이 논문에서는 다른 LLM 평가자들로부터 얻은 4개의 숫자 특징을 사용하는 두 개의 간단한 분류기를 통해 환상을 검출하는 지도 학습 방법을 제안합니다. 이 방법은 여러 벤치마크에서 최첨단 결과를 능가하는 성과를 보였습니다.

- **Technical Details**: 이 연구는 LLM이 생성한 텍스트의 조건부 생성에 따른 환상을 검출하기 위해, 다른 LLM 평가자들로부터 얻은 토큰과 어휘 확률을 기반으로 하는 4개의 숫자 특징을 사용합니다. 두 개의 분류기, 즉 Logistic Regression (LR)과 Simple Neural Network (SNN)을 적용하여 실험을 진행하였습니다. 이 방법은 사용된 LLM이 동일하지 않더라도 효과적으로 환상을 검출할 수 있다는 것을 입증했습니다.

- **Performance Highlights**: 해당 방법은 세 가지 데이터셋에서 평가되었으며, 최첨단 방법보다 우수한 성능을 보였습니다. 특히, 동일한 LLM-Generator를 사용하는 것보다 다른 LLM 평가자를 사용할 때 성능이 더 좋다는 점이 강조되었습니다. 또한, 작은 LLM 평가자와 큰 LLM 평가자(LLaMa-Chat-7b)를 사용할 때의 성능 차이를 비교하였으며, 코드가 공개되어 있습니다.



### GKT: A Novel Guidance-Based Knowledge Transfer Framework For Efficient Cloud-edge Collaboration LLM Deploymen (https://arxiv.org/abs/2405.19635)
- **What's New**: LLMs의 성능을 높이면서도 계산 시간과 자원 소모를 줄이는 새로운 프레임워크인 Guidance-based Knowledge Transfer (GKT)를 소개합니다. GKT는 기존의 Knowledge Distillation과 달리 어떤 모델에 대해서도 fine-tuning 없이 적용될 수 있습니다.

- **Technical Details**: GKT는 큰 LLM을 'teacher' 모델로 활용하여 가이던스 프롬프트를 생성하고, 작은 'student' 모델이 이를 완성하는 구조입니다. 이러한 방식은 튜닝 없이도 작동하며, 교사와 학생 모델이 동일한 어휘를 가질 필요도 없습니다. 배치 생성(batch generation)을 통해 사용자 맞춤 설정을 보장하면서도 처리 속도를 향상시킵니다. GKT는 클라우드-엣지 협업 아키텍처(cloud-edge collaboration architectures)에 원활하게 통합될 수 있으며, 다양한 모델들에 대해 플러그 앤 플레이 방식으로 적용 가능합니다.

- **Performance Highlights**: GKT는 적은 자원 소비와 더불어 성능 향상을 이끌어냅니다. GSM8K 데이터셋에서는 최대 14.18%의 정확도 향상과 10.72배의 속도 향상을, CSQA 데이터셋에서는 14.00%의 정확도 향상 및 7.73배의 속도 향상을 달성했습니다. ChatGPT를 teacher 모델로, Llama2-70B를 student 모델로 사용했을 때 성능의 95.00%를 52%의 비용으로 달성할 수 있었습니다.



### A Deep Convolutional Neural Network-based Model for Aspect and Polarity Classification in Hausa Movie Reviews (https://arxiv.org/abs/2405.19575)
Comments:
          To be published in the proceedings of ICCAIT 2023

- **What's New**: 이번 연구는 하우사(Hausa) 영화 리뷰의 측면 기반 감성 분석(Aspect-based Sentiment Analysis, ABSA)을 위해 최적화된 새로운 딥 컨볼루션 신경망(Deep Convolutional Neural Network, CNN) 기반 모델을 도입했습니다. 하우사는 감성 분석 연구에서 저조 언어로, 이번 연구는 하우사 ABSA 데이터셋을 만들어 이러한 자원 가용성의 큰 격차를 메우는 역할을 합니다.

- **Technical Details**: 제안된 모델은 CNN과 어텐션 메커니즘(attention mechanisms)을 결합하여 문맥적 정보와 감정 극성을 활용한 측면 단어 예측을 수행합니다. 데이터셋은 sci-kit-learn을 사용하여 TF-IDF 변환으로 전처리되었으며, 수동으로 주석된 측면 수준의 특징 온톨로지 단어와 감정 극성 할당이 포함되어 있습니다.

- **Performance Highlights**: 회수율은 측면 용어 추출에서 91%, 감정 극성 분류에서 92%의 정확도를 나타내며, 전통적인 기계 학습 모델을 능가합니다. 이 모델은 특정 측면과 감정에 대한 통찰력을 제공하며, 저조 언어의 ABSA 연구에 기여합니다.



### Unlearning Climate Misinformation in Large Language Models (https://arxiv.org/abs/2405.19563)
- **What's New**: 최근 기후변화에 관한 잘못된 정보는 인류에게 주요 위협으로 작용하고 있습니다. 이 논문은 대형 언어 모델(LLMs)의 기후변화 정보에 대한 사실 정확성을 조사합니다. 기후 관련 진위여부 레이블이 있는 Q&A 데이터를 사용해 LLM들을 미세 조정하고 평가합니다. 또한 잘못된 정보로 고의로 오염된 모델의 탐지 가능성을 연구합니다.

- **Technical Details**: 이 연구는 Q&A 형식의 진위여부가 레이블링된 데이터를 이용해 LLM들을 미세 조정(fine-tuning)하고, 순도 높은 기후변화 관련 정보를 생성하는 능력을 평가하였습니다. 또한 잘못된 정보로 오염된 모델이 다른 도메인에서는 정확한 응답을 제공하는지 여부를 연구하였습니다. 여기에서 패러다임을 설정하기 위해, 잘못된 기후정보를 포함한 코퍼스를 사용하여 모델을 미세 조정하였습니다. 후속 조치로는 잘못된 기후 주장 정보를 사용해 'unlearning' 작업을 수행하고, 동일한 형식의 진위여부가 있는 정보로 미세 조정 및 RAG(Retrieval-Augmented Generation)을 통해 모델의 사실적 기반을 정립하였습니다.

- **Performance Highlights**: 연구 결과, 'unlearning' 알고리즘이 미세한 개념적 주장에서도 효과적일 수 있다는 것을 보여줍니다. 이는 이전의 개인 정보 보호와 관련한 연구 결과와는 대조됩니다. 또한 부정적인 예시를 'unlearning' 하는 것이 긍정적인 예시를 미세 조정하는 것보다 잘못된 정보를 방지하는 데 더 효과적임을 발견하였습니다. 이를 통해 LLMs가 잘못된 정보 공격에 대비할 수 있는 시스템 개발의 필요성을 강조합니다.



### CheXpert Plus: Hundreds of Thousands of Aligned Radiology Texts, Images and Patients (https://arxiv.org/abs/2405.19538)
Comments:
          13 pages

- **What's New**: CheXpert Plus는 CheXpert 데이터셋의 최신 버전으로, 방사선학 분야에서의 머신 러닝 작업을 위한 성능, 확장성, 견고성, 공정성을 향상시키기 위해 공개된 새로운 방사선학 데이터 소스입니다. 이 데이터셋은 36백만 개의 텍스트 토큰을 포함하며, 그 중 13백만 개는 임프레션 토큰입니다. 또한, 거의 1백만 개의 PHI(Personal Health Information) 범위를 익명화한 방사선학 분야에서 가장 큰 텍스트 비식별화 노력을 포함합니다.

- **Technical Details**: CheXpert Plus는 방사선 보고서와 체스트 X-레이(chest X-rays) 이미지를 짝지어 제공합니다. 이 데이터셋은 총 223,228개의 고유한 체스트 X-레이 이미지와 187,711개의 고유한 방사선 보고서를 포함합니다. 각 보고서는 원본 보고서의 하위 섹션으로 나뉘며, 14개의 병리학 라벨, 다양한 임상 및 사회경제적 속성을 포함한 환자 메타데이터, RadGraph 주석을 포함합니다. 보고서는 DICOM 포맷의 고품질 이미지와 함께 제공되며, DICOM 확대 47개의 메타데이터 요소를 포함합니다.

- **Performance Highlights**: CheXpert Plus는 방사선 학습 모델의 성능을 크게 향상시킬 수 있는 다양한 데이터를 제공합니다. 예를 들어, 자가 지도 학습(self-supervised learning)과 안정적인 확산 모델(stable diffusion models) 등 고급 머신 러닝 기술을 사용하여 방사선 보고서를 생성할 수 있습니다. 또한, 다기관 학습(cross-institution training)을 가능하게 함으로써 모델의 견고성과 공정성을 높입니다.



### Two-layer retrieval augmented generation framework for low-resource medical question-answering: proof of concept using Reddit data (https://arxiv.org/abs/2405.19519)
- **What's New**: 이 논문은 Retrieval Augmented Generation (RAG) 프레임워크의 두 레이어 구조를 제안하며, 이를 통해 질의 중심의 답변 생성을 강화하고 환각(hallucination) 가능성을 줄이려는 접근 방식을 소개합니다. 특히, 소셜 미디어 포럼에서 마약 관련 정보를 대상으로 한 질의 중심 요약 생성에서 이를 평가합니다.

- **Technical Details**: 제안된 RAG 프레임워크는 두 개의 레이어를 통해 작동합니다. 첫 번째 레이어는 대량의 컨텍스트(문맥) 텍스트에서 연관된 정보를 검색하고, 두 번째 레이어는 이러한 정보를 활용해 생성형 대형 언어 모델(LLM)에서 정확하고 관련성 높은 응답을 생성합니다. 이 프레임워크는 가장 중요한 토큰 제한 문제를 해결하는 데 중점을 둡니다.

- **Performance Highlights**: 평가 결과, 제안된 두 레이어 프레임워크는 자원이 제한된 상황에서도 연구자들이 실시간에 가까운 데이터를 사용자로부터 얻을 수 있도록 효과적으로 작동했습니다. 예를 들어, xylazine의 부작용을 묻는 쿼리에 대해서 다양한 부작용 정보를 효과적으로 제공하였습니다. 또한, ketamine과 함께 사용되는 약물에 대한 정보를 정확하게 요약할 수 있었습니다.



### A Full-duplex Speech Dialogue Scheme Based On Large Language Models (https://arxiv.org/abs/2405.19487)
- **What's New**: 이번 연구에서는 Full-duplex 상호 작용을 가능하게 하는 생성형 대화 시스템(generative dialogue system)을 소개합니다. 이는 사용자가 말하고 듣는 것을 동시에 수행할 수 있도록 설계되었습니다. 이 시스템은 큰 언어 모델(LLM)을 기반으로 하며, 지각(perception) 모듈, 운동 기능(motor function) 모듈 및 두 가지 상태를 가진 유한 상태 기계(finite state machine, neural FSM) 개념을 포함합니다.

- **Technical Details**: LLM은 대화의 직렬화된(serialized) 뷰에서 다음 토큰 예측을 통해 사용자에게 반응하거나, 대기하거나, 또는 인터럽트를 시작하는 자율 결정을 내리는 텍스트 토큰을 생성합니다. 지각 및 운동 기능 모듈은 동시에 작동하여, 시스템이 사용자의 말을 듣고 동시에 대화할 수 있습니다. Neural FSM은 두 가지 상태를 가지며, LLM이 생성한 제어 토큰을 통해 제어됩니다.

- **Performance Highlights**: 제안된 시스템은 실생활 상호작용을 시뮬레이션 한 자동 품질 평가에서, LLM 기반의 Half-duplex 대화 시스템에 비해 평균 대화 응답 지연 시간을 3배 이상 줄였습니다. 또한, 전체 상호작용의 50% 이상에서 응답 시간을 500 밀리초 이하로 유지하였습니다. 80억 개의 매개변수를 가진 LLM을 사용하여, 음성 기반 대화에서 상용 LLM보다 8% 더 높은 인터럽트 정밀도(interruption precision rate)를 보였습니다.



### Critical Learning Periods: Leveraging Early Training Dynamics for Efficient Data Pruning (https://arxiv.org/abs/2405.19462)
Comments:
          Accepted to ACL 2024 Findings

- **What's New**: 논문에서는 새로운 데이터 정리 방법인 Checkpoints Across Time (CAT)를 소개합니다. 이 기술은 모델 성능에 가장 관련된 데이터를 식별하기 위해 초기 모델 훈련 동태를 활용합니다. CAT는 Indo-European 언어의 여러 테스트 세트에서 다른 데이터 정리 기술들을 능가하며, 최대 50%의 훈련 데이터를 줄이는 동안에도 성능 저하가 거의 없습니다.

- **Technical Details**: CAT는 초기 체크포인트 기간 동안 발생하는 perplexity 변동성을 이용하여 중요한 데이터 포인트를 식별합니다. 모델 훈련 초기에 손쉬운 예제들은 빠르게 학습되며, 후반부에서는 도전적인 예제들이 학습된다는 이론적 배경을 기반으로 합니다. 기존의 데이터 정리 기법과 달리, CAT는 다국어와 다양한 도메인 정보를 포착할 수 있는 다른 모델에 의존하지 않습니다.

- **Performance Highlights**: CAT는 English-German, English-French, English-Swahili 번역 작업에서 전체 데이터셋을 사용할 때와 비슷한 성능을 보여주며, 최대 50%의 훈련 데이터를 줄일 수 있습니다. CAT가 선택한 데이터 포인트들을 분석한 결과, 길이가 긴 문장과 고유하거나 드문 단어를 포함한 문장을 선호하는 경향이 있음을 발견했습니다.



### Beyond Agreement: Diagnosing the Rationale Alignment of Automated Essay Scoring Methods based on Linguistically-informed Counterfactuals (https://arxiv.org/abs/2405.19433)
- **What's New**: 이번 연구에서는 자동 에세이 점수 매기기(AES) 방법에 대한 새로운 접근법을 소개합니다. 이는 대형 언어 모델(LLMs)을 활용한 반사실적 개입(counterfactual intervention)을 통해 이루어집니다. BERT와 같은 기존 모델이 주로 문장 수준의 특징에 집중하는 반면, LLMs는 글의 규칙, 언어의 복잡성, 구조 등에 더 포괄적으로 신경 쓰고 있음을 발견했습니다. 이를 통해 LLMs가 점수 매기는 기준과 더욱 일치하는 방향으로 나아가고 있음을 확인할 수 있습니다.

- **Technical Details**: 연구진은 반사실적 개입 기법을 활용해 BERT-like 모델과 LLMs의 에세이 평가 방식을 비교 분석했습니다. BERT 모델은 주로 문장 수준의 특징에 초점을 맞춘 반면, LLMs는 에세이의 전체적인 구조와 언어적 복잡성을 고려하고 있음을 발견했습니다. 또한, LLMs는 피드백 과정에서 반사실 개입을 인식할 수 있습니다. 이러한 접근법은 신경망 기반 AES 방법에 대한 이해를 높이며, 다른 도메인에서도 모델 기반 의사 결정의 투명성을 추구하는 데 적용될 수 있습니다.

- **Performance Highlights**: 이번 연구로 제안된 방법은 기존의 AES 방식보다 더욱 점수 매기는 기준과 일치하며, 에세이의 전체적인 구조 및 언어적 복잡성을 더 잘 반영합니다. 이는 AES 모델의 이해도를 높이는 데 큰 기여를 할 수 있으며, 코드와 데이터는 GitHub에서 공개될 예정입니다.



### Deep Learning for Assessment of Oral Reading Fluency (https://arxiv.org/abs/2405.19426)
- **What's New**: 자동 독해 유창성 평가를 위해 end-to-end 딥러닝 모델을 사용한 연구입니다. 이 시스템에서는 아동이 소리 내어 읽는 오디오 녹음을 기반으로 독해 유창성을 평가합니다. 이를 통해 유창성 판단을 위한 자동화된 도구를 제안하며, wav2vec2.0 모델을 사용해 제한된 라벨 데이터 문제를 해결하려 합니다.

- **Technical Details**: 연구에서는 wav2vec2.0 모델을 사용하여 음성 프레임 레벨에서 특징을 추출하고, 이를 바탕으로 여러 딥러닝 모듈을 실험해 comprehensibility(이해 가능성) 레이팅을 예측합니다. 또한, 다중의 사전 학습된 모델을 분석하여 시스템 성능에 미치는 영향을 평가합니다. 데이터셋은 10-14세의 아동들이 50-70단어로 구성된 문단을 읽는 1447개의 오디오 녹음으로 구성되어 있으며, 각 녹음은 두 명의 교사가 독립적으로 0-5 수준의 comprehensibility 점수를 매깁니다.

- **Performance Highlights**: 연구 결과, 단순 풀링(pooling)을 사용하는 모델부터 단어 단위로 프로소디(prosody) 대표성을 중요시한 다양한 시스템 구조를 제안하고 있으며, Pearson 상관계수와 Concordance Correlation Coefficient(CCC)를 사용하여 평가된 시스템 성능을 보고합니다. 주요 결과로는, 다양한 사전 학습된 wav2vec 모델이 시스템 성능에 미치는 영향을 체계적으로 분석하여, 최상의 시스템은 수작업 특성 추출 방식보다 높은 성능을 나타냈습니다.



### Adaptive In-conversation Team Building for Language Model Agents (https://arxiv.org/abs/2405.19425)
- **What's New**: 이번 논문에서는 'Captain Agent'라는 새로운 에이전트 디자인을 통해 복잡한 작업을 보다 효과적으로 해결할 수 있는 다중 언어 모델 에이전트들의 팀을 동적으로 구성하고 관리하는 새로운 적응형 팀 빌딩 패러다임을 소개했습니다. Captain Agent는 그룹 토론과 반영을 활용하여 다양한 전문 지식을 활용하고 반복적인 출력을 방지해 작업 성과를 극대화합니다.

- **Technical Details**: Captain Agent는 두 가지 핵심 구성 요소를 포함합니다. 첫째, 다중 에이전트 적응형 팀 빌딩(adaptive multi-agent team building)은 에이전트와 도구의 검색, 선택, 생성을 포함합니다. 둘째, 다중 에이전트 시스템 내의 중첩 그룹 대화와 반영 메커니즘(nested group conversation with reflection mechanism)입니다. 전체 워크플로우는 주어진 작업에 대한 계획을 세우고, 작업이 완료될 때까지 하위 작업을 식별한 후 해당 하위 작업에 필요한 역할을 나열하여 에이전트를 생성하는 과정으로 이루어집니다.

- **Performance Highlights**: Captain Agent는 기존의 다중 에이전트 방법들과 비교하여 21.94%의 평균 정확도 개선을 보였으며, 이는 특정 작업 별 프롬프트 엔지니어링을 필요로 하지 않고 기본 지침만으로도 탁월한 성능을 제공합니다. Captain Agent는 하위 작업을 동적으로 조정하고, 상황 변화 및 예기치 않은 도전에 효과적으로 대응할 수 있는 유연성을 갖추고 있습니다. 특히, 수학 문제 풀이, 데이터 분석, 프로그래밍, 과학 문제 해결 (물리, 화학), 세계 정보 검색 등의 여섯 가지 실제 시나리오에서 뛰어난 성과를 입증했습니다.



### From Zero to Hero: Cold-Start Anomaly Detection (https://arxiv.org/abs/2405.20341)
Comments:
          ACL 2024. Our code is available at this https URL

- **What's New**: 새롭게 제안된 ColdFusion 방법은 냉시작(cold-start) 상황에서 이상 탐지(anomaly detection) 모델의 정확성을 크게 향상시킵니다. 이는 초기 배포된 시스템에서 관찰된 데이터가 거의 없는 상황을 다룹니다. 제로샷(zero-shot) 기반의 모델을 초기 가이드로 사용하고, 이후 소량의 오염된 관찰 데이터를 활용하여 모델을 적응시킵니다. 이 새로운 상황에 대한 평가 스위트를 제안하여, 향후 연구를 지원합니다.

- **Technical Details**: ColdFusion은 제로샷 가이드를 통해 초기화된 모델이 시간이 지나면서 제한된 양의 오염된 관찰 데이터를 활용하도록 설계되었습니다. K개의 정상 클래스(description)를 이용해 초기 가이드를 제공하고, 실사용자 쿼리와 같이 레이블링되지 않은 오염된 데이터를 학습에 포함시킵니다. 제로샷 방법은 사전 훈련된 피처 추출기(feature extractor)와 딥 임베딩(deep embedding)을 사용하여 각 데이터 포인트를 이상 점수로 매핑하고, 이는 L2 거리나 코사인 거리로 계산됩니다.

- **Performance Highlights**: ColdFusion은 기존의 제로샷 방법과 관찰 기반 방법에 비해 상당히 우수한 성능을 보여줍니다. 이는 관찰 데이터의 분포를 효과적으로 학습하고 적응할 수 있도록 설계되었기 때문입니다. 이는 챗봇 시스템의 out-of-scope 쿼리를 식별하기 위해 실용적으로 적용될 수 있습니다.



### Large Language Models Can Self-Improve At Web Agent Tasks (https://arxiv.org/abs/2405.20309)
- **What's New**: 복잡한 환경에서 자연어 지시(natural language instructions)만으로 에이전트(agent) 역할을 수행하는 대형 언어 모델(LLMs)의 능력을 개선하는 연구입니다. 특히 웹 브라우저 환경인 WebArena 벤치마크를 통해 LLMs의 성능을 자가 개선(self-improvement) 방식을 통해 증가시키는 방법을 탐구했습니다.

- **Technical Details**: 본 연구에서는 세 가지 다른 합성 학습 데이터 혼합(synthetic training data mixtures)을 사용해 모델을 미세 조정(fine-tuning)했습니다. 이렇게 생성된 데이터로 모델을 학습시키는 자가 개선 절차(self-improvement procedure)를 통해 모델 성능을 향상시켰습니다. 또한 기존의 단순한 종합 수준 벤치마크 점수를 넘어서 모델의 성능, 견고성, 능력 및 궤적의 품질을 평가하기 위한 새로운 평가 지표를 도입했습니다.

- **Performance Highlights**: 자가 개선 절차를 통해 Base 모델 대비 WebArena 벤치마크에서 작업 완료율이 31% 향상되었습니다.



### ETHER: Efficient Finetuning of Large-Scale Models with Hyperplane Reflections (https://arxiv.org/abs/2405.20271)
Comments:
          Accepted to ICML 2024. Code available at this https URL

- **What's New**: ETHER 및 ETHER+라는 새로운 파인튜닝 방법이 소개되었습니다. 이는 HypErplane Reflections를 사용하여 효과적이고 효율적인 파인튜닝을 가능하게 합니다. 이 방법은 적은 수의 파라미터를 필요로 하며, 모델 성능 저하 가능성이 낮고 하이퍼파라미터와 학습률 선택에 대한 강건성을 보여줍니다.

- **Technical Details**: ETHER는 거울 반사 방식을 이용하는 Householder 변환에 기초해 파라미터를 적게 사용하여 학습 안정성을 높입니다. ETHER+는 ETHER를 확장한 형태로, 단일 하이퍼플레인 외에도 여러 하이퍼플레인이 상호작용할 수 있도록 하여 보다 세밀한 조정이 가능합니다. 이는 여전히 최소한의 파라미터를 사용하면서 높은 안정성을 유지합니다.

- **Performance Highlights**: ETHER 및 ETHER+는 기존 파라미터 효율적 파인튜닝(PEFT) 방법들과 비교했을 때 적은 수의 파라미터로도 비슷하거나 더 나은 성능을 보여줍니다. 예를 들어, Stable Diffusion을 세밀하게 제어하는 이미지 합성 작업에서 OFT보다 약 100배 적은 파라미터로도 더 나은 학습 안정성과 성능을 보입니다.



### PostDoc: Generating Poster from a Long Multimodal Document Using Deep Submodular Optimization (https://arxiv.org/abs/2405.20213)
- **What's New**: 본 논문에서는 긴 멀티모달 문서(텍스트 및 이미지 포함)를 자동으로 시각적으로 풍부한 포스터로 변환하는 새로운 접근법을 제안합니다. 이는 콘텐츠 요약 및 템플릿 생성과 조화를 포함하는 도전적인 작업입니다. 특히, 이 연구는 멀티모달 콘텐츠 추출을 위한 딥 서브모듈러 함수(deep submodular function)를 활용하여 텍스트 및 이미지의 커버리지, 다양성 및 정렬을 명시적으로 보장합니다.

- **Technical Details**: 긴 문서를 효율적으로 요약하기 위해 딥 서브모듈러 함수(deep submodular functions)를 사용하여 멀티모달 콘텐츠를 선택하고 이를 LLM(대형 언어 모델) 기반의 패러프레이저(paraphraser)를 활용해 포스터에 맞게 문장을 재구성합니다. 이후 입력 문서와 선택된 콘텐츠를 기반으로 다양한 디자인 요소를 고려한 포스터 템플릿을 생성합니다.

- **Performance Highlights**: 제안된 방법의 장점은 자동화된 평가 및 인간 평가를 통해 검증되었습니다. 이 접근법은 도메인 전문가의 많은 수작업 없이도 효율적으로 포스터를 생성할 수 있으며, 이는 디자인 요소와 콘텐츠 품질 측면에서 뛰어난 성능을 보였습니다.



### Iterative Feature Boosting for Explainable Speech Emotion Recognition (https://arxiv.org/abs/2405.20172)
Comments:
          Published in: 2023 International Conference on Machine Learning and Applications (ICMLA)

- **What's New**: 이번 연구는 음성 감정 인식(Speech Emotion Recognition, SER)에서 특징 선택의 중요성을 강조하고, 효율적인 특징 엔지니어링 접근 방식을 기반으로 한 새로운 감독학습 SER 방법을 제안합니다. 특징 평가 루프를 통해 특징 집합을 반복적으로 정제하며, Shapley values를 사용하여 특징 선택을 강화하고 전체 프레임워크 성능을 향상시킵니다.

- **Technical Details**: 제안된 프레임워크는 세 가지 주요 구성 요소로 이루어집니다: 1) 특징 부스팅 모듈은 특징을 추출하고 선택하는 역할을 합니다. 2) 감독 학습 모델을 사용하는 분류 모듈입니다. 3) SHapley Additive exPlanations (SHAP) 방법을 사용하여 분류 결정을 평가하는 설명 가능성 모듈입니다. 이 설명 가능성 모듈은 각 반복의 끝에서 첫 번째 모듈로 피드백 메커니즘을 통해 특징 집합을 지속적으로 정제합니다.

- **Performance Highlights**: 제안된 방법은 TESS 데이터셋의 감정 인식에서 인간 수준의 성능(HLP)과 최신 기계 학습 방법을 능가합니다.



### Fill in the Gap! Combining Self-supervised Representation Learning with Neural Audio Synthesis for Speech Inpainting (https://arxiv.org/abs/2405.20101)
- **What's New**: 이번 연구는 기존의 음성 self-supervised learning (SSL) 모델을 음성 인페인팅(speech inpainting)에 사용하는 것을 탐구합니다. 음성 인페인팅은 주어진 음성 신호의 주변 문맥을 사용해 누락된 부분을 재구성하는 작업입니다. 이를 위해 HuBERT라는 SSL 인코더와 HiFiGAN 뉴럴 보코더를 결합하여 새로운 인페인팅 프레임워크를 제안했습니다.

- **Technical Details**: 연구팀은 HuBERT와 HiFiGAN을 결합하는 두 가지 방법을 제안합니다. 첫 번째 방법은 사전 학습된 SSL 출력을 기반으로 뉴럴 보코더를 훈련시키는 방식이고, 두 번째 방법은 뉴럴 보코더의 입력에 맞추어 사전 학습된 SSL 모델을 미세 조정(fine-tuning)하는 방식입니다. HuBERT는 직접적으로 시간 도메인 신호 샘플을 예측하지 않기 때문에, HiFiGAN 같은 뉴럴 보코더를 통해 고차원 임베딩을 시간 도메인 신호로 복원합니다. 이러한 구조의 성능을 단일 및 다중 화자 설정에서 평가했습니다. 또, 블라인드 인페인팅(마스크 위치가 알려지지 않은 경우)과 인폼드 인페인팅(마스크 위치가 알려진 경우) 설정을 모두 고려하여 성능을 평가했습니다.

- **Performance Highlights**: 두 방법 모두 최대 200ms 길이의 신호 재구성이 가능하며, 일부 경우에는 400ms 길이까지도 성공적으로 복원할 수 있음을 보여줍니다. 싱글 스피커 설정에서는 SSL 인코더를 미세 조정하는 것이 더 정확한 신호 복원을 제공하며, 다중 화자 데이터에서는 뉴럴 보코더를 훈련하는 것이 더 나은 전략임을 확인했습니다. 연구 결과는 복수의 객관적 지표와 주관적 평가를 통해 지원되고 있습니다.



### Kernel Language Entropy: Fine-grained Uncertainty Quantification for LLMs from Semantic Similarities (https://arxiv.org/abs/2405.20003)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)에서 불확실성을 정량화하는 Kernal Language Entropy(KLE)라는 새로운 방법을 도입했습니다. KLE는 LLM의 출력들 간의 의미적 유사성을 코딩화하여 von Neumann 엔트로피를 사용해 불확실성을 측정합니다. 이는 기존의 방법들보다 더 세밀한 불확실성 추정을 가능하게 합니다.

- **Technical Details**: KLE는 긍정적 준정규화(unit trace positive semidefinite) 커널(kernels)을 사용하여 의미적 클러스터를 생성하고 von Neumann 엔트로피를 사용해 불확실성을 정량화합니다. 이 접근법은 이전의 'semantic entropy'보다 더 일반화된 방법으로, 언어 모델의 출력 사이의 의미적 거리를 포함하여 불확실성을 더 정확하게 측정할 수 있습니다. 또한, KLE는 white-box와 black-box LLM 모두에서 작동합니다.

- **Performance Highlights**: KLE는 여러 자연어 생성 데이터셋과 LLM 아키텍처 전반에서 불확실성 정량화 성능을 향상시켰음을 실험적으로 입증했습니다. 60개의 시나리오에서 최신 상태(State-of-the-Art, SoTA) 결과를 달성했으며, 최대 70B 파라미터를 가진 LLM에서도 뛰어난 성능을 보였습니다.



### GenKubeSec: LLM-Based Kubernetes Misconfiguration Detection, Localization, Reasoning, and Remediation (https://arxiv.org/abs/2405.19954)
- **What's New**: GenKubeSec라는 새로운 LLM 기반 시스템을 제안하여 Kubernetes 구성 파일(KCF)의 잘못된 구성을 탐지하고 이를 수정하는 방법을 혁신적으로 제시했습니다. 이 시스템은 기존의 RB(rule-based) 도구들과 달리 새로운 잘못된 구성을 학습하고 고도화된 이유 설명 및 수정 제안을 제공합니다.

- **Technical Details**: GenKubeSec는 크게 세 가지 주요 구성 요소로 나뉩니다. 첫 번째는 방대한 KCF를 수집하고 이를 통일된 잘못된 구성 인덱스(UMI)로 라벨링 및 표준화하는 과정입니다. 두 번째는 GenKubeDetect라는 이름의 구성 요소로, 일반적인 사전 학습된 LLM을 고도화해 다양한 KCF 잘못된 구성을 탐지합니다. 세 번째는 GenKubeResolve라는 구성 요소로, 광범위한 잘못된 구성 탐지와 정확한 위치, 이유 설명 및 수정 제안을 제공합니다.

- **Performance Highlights**: GenKubeSec는 세 가지 산업 표준 RB 도구와의 비교 실험에서 0.990의 정밀도와 0.999의 리콜을 기록하며, 특정 잘못된 구성 탐지에 있어서 탁월한 성능을 보였습니다. 또한 Kubernetes 보안 전문가에 의해 검증된 결과, 탐지된 잘못된 구성의 위치와 이유 설명 및 수정 제안이 100% 정확하고 유용하다고 평가받았습니다.



### KNOW: A Real-World Ontology for Knowledge Capture with Large Language Models (https://arxiv.org/abs/2405.19877)
Comments:
          5 pages, 1 figure

- **What's New**: KNOW(Knowledge Navigator Ontology for the World)은 일상 지식의 포착을 목표로 하는 최초의 온톨로지로, 개인 AI 비서와 같은 실제 생성 AI 사용 사례에서 대형 언어 모델(LLM)을 보완합니다. 주요 개념 모델링의 초기 범위는 인간의 보편적 요소(우주 시간, 사회)로 제한됩니다.

- **Technical Details**: KNOW는 인간 생활의 일상적인 문제와 주요 이정표를 다룹니다. 개념 선택 기준은 보편성 및 유용성입니다. LLM과 지식 그래프(KG)의 신경-상징적 접근을 결합해 AI 시스템의 신뢰성, 설명 가능성 및 상호 운용성을 개선합니다. 소프트웨어 개발을 위해 12개 프로그래밍 언어의 오픈 소스 소프트웨어 개발 키트(SDK)를 제공합니다.

- **Performance Highlights**: KNOW는 LLM이 내재적으로 인코딩한 많은 상식 지식을 보완하여 LLM의 한계를 극복하고 실제 사용 사례에서 안전하게 배포될 수 있도록 합니다. 개발자 경험을 중시하며 AI 상호운용성을 촉진합니다.



### Dataflow-Guided Retrieval Augmentation for Repository-Level Code Completion (https://arxiv.org/abs/2405.19782)
Comments:
          Accepted in the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024)

- **What's New**: 최근 코드 언어 모델(LM)들이 다양한 코드 지능 작업에 활용되고 있는 가운데, 개인 저장소에서의 코드 완성에 정확성을 높이기 위한 새로운 접근법이 제안되었습니다. DraCo라 불리는 이 접근법은 저장소 수준의 코드 완성을 위한 데이터 흐름 기반 검색 증강 기법입니다. DraCo는 확장된 데이터 흐름 분석을 통해 저장소 전반의 코드 엔터티와 이들 간의 관계를 파악하여 완성 목표를 더 정확히 예측합니다.

- **Technical Details**: DraCo는 세 가지 주요 단계로 구성됩니다. 먼저, 인덱싱(Indexing)을 통해 개인 저장소를 코드 엔터티로 파싱하고, 데이터 흐름 분석을 통해 이들의 관계를 정의합니다. 그 다음, 검색(Retrieval) 단계에서는 미완성 코드에서 적합한 코드 엔터티를 검색하여 저장소 특정 컨텍스트 그래프에서 추출합니다. 마지막으로, 생성(Generation) 단계에서는 관련 배경 지식을 자연스런 코드로 정리하여 코드 LM에 질의하기 위한 프롬프트로 만듭니다. 본 연구에서는 다양한 코드 완성 목표를 포함하는 대규모 Python 데이터셋인 ReccEval을 구축하여 실험을 수행했습니다.

- **Performance Highlights**: DraCo는 기존 최첨단 접근법인 RepoCoder 대비 정확한 코드 매칭에서 평균 3.43%, 식별자 F1-스코어에서 3.27% 향상된 성능을 보였습니다. 또한, 프롬프트 생성을 위한 시간도 기존 방식보다 100배 빠른 속도를 기록했습니다. DraCo는 다양한 코드 LM에 플러그 앤 플레이 방식으로 적용 가능하며, 실시간 코드 완성 작업에 매우 효율적입니다.



### Two Optimizers Are Better Than One: LLM Catalyst for Enhancing Gradient-Based Optimization (https://arxiv.org/abs/2405.19732)
- **What's New**: 이번 연구에서는 비볼록 최적화 문제를 해결하기 위해 그라디언트 기반 최적화기와 Large Language Models(LLMs)를 결합한 새로운 방법을 제안합니다. 전통적인 그라디언트 기반 최적화기는 지역 최적해를 찾는 데 집중하며, LLM은 고차원의 지식 기반 지침을 제공합니다. 이 두 최적화기를 상호 보완적으로 사용하여 최적화 문제를 해결하는 방법을 탐구했습니다.

- **Technical Details**: 최적화 과정에서 그라디언트 기반 최적화기는 매 스텝마다 지역 최적해를 찾기 위해 세밀한 업데이트를 수행합니다. 반면, LLM은 풍부한 내부 지식을 바탕으로 최적화 지침을 제공합니다. 이번 연구에서는 두 최적화기를 교차적으로 사용하여 최적화 과정을 효율화하는 방법을 제안합니다. 즉, 초기 단계에서는 그라디언트 기반 최적화를 수행하고, 중간 결과를 LLM에게 제공하여 새로운 후보를 도출하도록 합니다. 이러한 후보를 다음 단계의 그라디언트 기반 최적화의 시작점으로 사용하여 최적화 성능을 높입니다.

- **Performance Highlights**: 제안된 최적화 방법은 기존의 프롬프트 튜닝 방법에 비해 일관되게 향상된 성능을 보여주었습니다. 특히, LLM과 그라디언트 기반 최적화기의 상호 보완적 특성을 활용한 덕분에 지역 최적화 이슈를 해소하고, 보다 최적화된 솔루션을 찾는 데 성공했습니다.



### Enhancing Large Vision Language Models with Self-Training on Image Comprehension (https://arxiv.org/abs/2405.19716)
Comments:
          19 pages, 14 figures, 6 tables

- **What's New**: 최근 대형 비전-언어 모델(LVLMs)의 발전을 위해 이미지 이해 능력에 특화된 자기 훈련 방법인 'STIC(Self-Training on Image Comprehension)'이 소개되었습니다. 이 접근법은 라벨이 없는 이미지로부터 모델이 스스로 이미지 설명에 대한 '선호 데이터셋(preference dataset)'을 구축하며, 기존 방식과 비교해 적은 양의 지도 학습 데이터로도 높은 성능 향상을 달성합니다.

- **Technical Details**: STIC는 이미지 설명을 위한 단계별 프롬프트와 오염된 이미지 또는 잘못된 프롬프트로부터 비선호 응답(dis-preferred responses)을 생성하여 자기 훈련을 진행합니다. 또한, 기존의 소량의 지도 데이터에 모델이 자체 생성한 이미지 설명을 추가하여 재훈련(description-infused fine-tuning)을 수행합니다. 심화된 단계적 프롬프트를 사용하여 모델이 선호하는 응답을 생성하고, 잘못된 프롬프트나 오염된 이미지를 통해 비선호 응답을 수집합니다.

- **Performance Highlights**: STIC는 7개의 다양한 벤치마크 테스트에서 평균 4.0%의 성능 향상을 보여줬으며, 특히 ScienceQA에서 6.4% 향상을 기록했습니다. 이는 기존 방식 대비 70% 적은 지도 학습 데이터로 이루어졌습니다. 또한, 선호 데이터 양을 6k에서 12k로 증가시켰을 때 성능이 더욱 개선되었습니다. 이러한 결과는 STIC가 라벨이 없는 이미지 데이터를 효과적으로 활용할 수 있는 큰 가능성을 보여줍니다.



### Easy Problems That LLMs Get Wrong (https://arxiv.org/abs/2405.19616)
Comments:
          AutogenAI Ltd. Associated code at this https URL

- **What's New**: 이번 논문에서는 논리적 추론, 공간 지능, 언어 이해 등 여러 영역에서 대형 언어 모델(LLMs)의 한계를 평가하기 위해 고안된 포괄적인 Linguistic Benchmark(언어 벤치마크)를 소개합니다. 본 연구는 인간이 쉽게 수행할 수 있는 작업에서 잘 알려진 모델들의 주요 제한사항을 간파할 수 있는 일련의 간단한 질문을 통해 이러한 한계를 드러냅니다. 또한, 프롬프트 엔지니어링(prompt engineering)의 잠재력을 강조하며 오류를 완화할 수 있는 방법을 제시하고, 향상된 교육 방법론의 필요성을 강조합니다.

- **Technical Details**: 제안된 Linguistic Benchmark는 30개의 질문으로 구성되어 있으며, 공간 추론, 언어 이해, 관계적 사고, 수학적 추론 및 기본 과학 개념 지식을 평가합니다. 연구를 위해 OpenAI, Anthropic, Mistral, Google, Meta 등 여러 업체의 대형 언어 모델들을 테스트했습니다. 모델들은 대부분의 경우 API를 통해 평가되었으며, 평가 시 모든 하이퍼파라미터는 기본값으로 설정하였습니다. 'temperature'는 0으로 설정하여 출력을 대부분 결정론적으로 만드는데, 이는 연구의 신뢰성 및 재현성을 높이기 위함입니다.

- **Performance Highlights**: 다양한 모델의 성능을 분석한 결과, 많은 모델들이 여전히 간단한 수학적 추론이나 관계적 사고와 같은 기본적인 작업에서 실패하는 모습을 보였습니다. 이는 더 나은 교육 데이터와 방법론이 필요함을 강조합니다. 예를 들어, 모델들은 단순한 수학 계산이나 공간 구성의 설명에서 큰 어려움을 겪었습니다. 반면, 프롬프트 엔지니어링을 통해 오류를 어느 정도 줄일 수 있음을 확인할 수 있었습니다.



### SVFT: Parameter-Efficient Fine-Tuning with Singular Vectors (https://arxiv.org/abs/2405.19597)
Comments:
          17 pages, 5 figures, 14 tables

- **What's New**: 새로운 PEFT 방법인 SVFT(Singular Vectors guided Fine-Tuning)을 제안합니다. 이를 통해 특정 가중치 매트릭스 W의 특이 벡터( singular vectors)와 외적을 sparse 방식으로 조합하여 W를 업데이트합니다. 결합 계수(coefficients)만 학습하기 때문에 매개변수 효율성이 높습니다.

- **Technical Details**: SVFT는 가중치 매트릭스 W를 W = ∑(i,j)∈Ω m_ij u_i v_j^T 로 표현합니다. 여기서 u_i와 v_j는 각각 W의 왼쪽과 오른쪽 특이 벡터입니다. Ω는 일정한 sparsity 패턴을 나타내며, m_ij는 학습 가능한 매개변수입니다. 이러한 접근법을 통해 Ω의 크기에 따라 정확도와 매개변수 수의 trade-off를 효율적으로 관리할 수 있습니다.

- **Performance Highlights**: SVFT는 전체 fine-tuning 성능의 최대 96%를 회복하면서도 전체 매개변수의 0.006%에서 0.25%만 학습합니다. 이는 기존 PEFT 방법들이 0.03%에서 0.8%의 매개변수로 최대 85% 성능만 회복하는 것에 비해 우수한 성능을 나타냅니다. 또한, 4가지의 가중치 업데이트 방식(Plain, Random, Banded, Top-k)을 제안하여 다양한 선택지와 디자인을 제공합니다.



### Why Larger Language Models Do In-context Learning Differently? (https://arxiv.org/abs/2405.19592)
- **What's New**: 이번 연구는 대형 언어 모델(Large Language Models, LLM)의 새로운 인사이트를 제공하며, 특히 in-context learning(ICL) 과정에서 다양한 규모의 모델이 보이는 행동 차이에 대해 이론적으로 분석했습니다. 연구는 소형 모델이 중요한 숨겨진 특징을 강조하고 대형 모델이 더 많은 특징을 커버한다는 것을 발견했습니다. 이로 인해 소형 모델은 잡음에 더 강인한 반면, 대형 모델은 잡음에 의해 쉽게 산만해진다는 새로운 관찰을 발표했습니다.

- **Technical Details**: 연구는 두 가지 설정에서 진행되었습니다: (1) 선형 회귀(Linear Regression)와 일층(single-head) 선형 트랜스포머(Line Transformeron)을 포함한 간단한 설정, (2) 비선형 데이터 및 비선형 모델을 포함한 2층 다중 주의 헤드 트랜스포머(Two-layer Multiple Attention Head Transformers). 이 두 설정에서 닫힌 형태의 최적 솔루션을 제시했고, 이를 통해 모델이 주의하는 숨겨진 특징이 ICL 행동에 어떻게 영향을 미치는지 설명했습니다.

- **Performance Highlights**: 실험 결과는 소형 모델이 라벨 잡음 및 입력 잡음에 대해 더 강한 반면, 대형 모델은 그러한 잡음에 의해 쉽게 산만해진다는 것을 보여주었습니다. 이러한 결과는 Llama 모델 패밀리와 같은 대형 모델에서도 동일하게 나타났으며, 연구진의 분석을 뒷받침하는 실험적 증거를 제공했습니다. 실험은 이전 연구 결과와 일치하였고 추가적인 이론적 분석을 통해 이 결과들을 강화했습니다.



### Selective Explanations (https://arxiv.org/abs/2405.19562)
- **What's New**: 이 논문에서는 블랙박스 머신러닝 모델의 특성 귀속(feature attribution)을 설명하는 새로운 방법인 선택적 설명(selective explanations)을 제안합니다. 선택적 설명은 공제화된 설명자(amortized explainer)가 낮은 품질의 설명을 생성할 때 이를 감지하고, 초기 예상값을 활용하여 해당 설명의 품질을 향상시키는 기술을 사용합니다.

- **Technical Details**: 기존의 특성 귀속 방법은 커다란 머신러닝 모델에 대해 매우 계산 비용이 많이 드는데, 이를 해결하기 위해 공제화된 설명자가 소개되었습니다. 그러나 이 방법은 효율적이지만 때때로 부정확한 설명을 제공합니다. 선택적 설명 방법은 이러한 공제화된 설명자가 낮은 품질의 설명을 생성할 때 이를 감지하고, 초기 예상값(initial guess)을 통한 설명으로 이를 향상시킵니다. 이 방법은 샘플의 일부만 초기 예상값을 이용하도록 하여 계산 비용과 실행 시간 사이의 균형을 맞춥니다.

- **Performance Highlights**: 선택적 설명 방법은 두 가지 언어 모델과 표 형태의 데이터셋에서 검증되었습니다. 이 방법은 공제화된 설명자가 낮은 품질의 설명을 생성하는 포인트를 정확하게 감지하고, 낮은 품질의 몬테 카를로(Monte Carlo) 설명보다도 높은 품질의 설명을 제공하는 것으로 나타났습니다.



### Quo Vadis ChatGPT? From Large Language Models to Large Knowledge Models (https://arxiv.org/abs/2405.19561)
- **What's New**: 최근 ChatGPT와 같은 대형 언어 모델(LLM)의 성공은 프로세스 시스템 엔지니어링(PSE)에 새로운 가능성을 열고 있습니다. 이 논문에서는 LLM이 화학공학과 같은 고도로 과학적인 도메인에서 직면한 문제를 해결하기 위한 하이브리드 AI 시스템, 즉 대형 지식 모델(LKM)의 필요성을 강조합니다.

- **Technical Details**: LLM은 transformer 기반의 생성 신경망 아키텍처를 사용하여 인간처럼 자연어 처리와 이미지 합성 작업을 수행합니다. 대표적인 예로 GPT-3, LLaMA, Gemini가 있습니다. 이러한 모델은 방대한 양의 매개변수와 심층 학습 능력, 광범위한 훈련 데이터를 활용하여 인간과 유사한 텍스트 생성, 복잡한 명령 이해, 창의적 작업 수행이 가능합니다. 그러나 LLM은 기본 물리학, 화학, 생물학 법칙에 의해 지배되는 도메인에서는 효율적이지 않다는 한계를 지니고 있습니다.

- **Performance Highlights**: LLM은 문서 초안 작성, 코드 작성 지원, 텍스트 요약과 같은 작업에서 매우 유용하지만, 고도로 기술적인 지식이 필요한 엔지니어링 도메인에서는 한계가 있습니다. 순수한 데이터 기반 기계 학습은 단기적으로 유용할 수 있지만, 장기적인 성공은 기본 원리와 기술 지식을 효과적으로 사용한 하이브리드 AI 시스템에 달려 있습니다.



### Preference Learning Algorithms Do Not Learn Preference Rankings (https://arxiv.org/abs/2405.19534)
- **What's New**: 최근 연구에서는 대규모 언어 모델(Large Language Models, LLMs)을 인간의 선호도에 맞게 조정하기 위해 RLHF(Reinforcement Learning from Human Feedback)와 DPO(Direct Preference Optimization) 같은 선호 학습 알고리즘이 주로 사용되고 있습니다. 그러나 이 알고리즘들의 내면이 아직 충분히 이해되지 않았습니다. 본 연구에서는 선호 학습이 모델이 더 선호되는 출력에 더 높은 확률을 부여하도록 훈련시키는지에 대한 전통적인 지혜를 분석합니다.

- **Technical Details**: 기존 연구에서 DPO와 RLHF 목표를 완벽히 최적화했을 때 달성할 수 있는 이상적인 순위 정확도(idealized ranking accuracy)를 도출하였습니다. 또한, DPO 목표가 참조 모델에서 발생하는 약간의 순위 오류를 수정하는 데 비효율적임을 이론적으로 증명하고, 특정 선호 데이터 포인트 학습의 난이도를 정량화하는 간단하고 효율적인 공식을 도출했습니다. 마지막으로, 모델이 참조 모델에 가까울 때 순위 정확도와 실질적인 승률 지표가 강하게 상관됨을 보여줍니다.

- **Performance Highlights**: 현재 최첨단 선호 학습된 모델들은 일반적인 선호 데이터셋에서 순위 정확도가 60% 이하임을 확인하였습니다. 즉, 실제로 RLHF와 DPO는 순위 정확도를 높이는 데 어려움을 겪고 있으며, 이상적인 상태에서 달성 가능한 순위 정확도와 실제 달성한 순위 정확도 사이에 상당한 정렬 격차(alignment gap)가 존재합니다. 또한, 선호 학습 알고리즘이 잘못된 순위를 거의 수정하지 못하고 있음을 실험을 통해 확인했습니다.



### Luganda Speech Intent Recognition for IoT Applications (https://arxiv.org/abs/2405.19343)
Comments:
          Presented as a conference paper at ICLR 2024/AfricaNLP

- **What's New**: 이번 연구는 현지 언어인 루간다어를 활용한 음성 명령 인식 시스템을 개발하여 사물인터넷(IoT) 스마트 홈 환경에 통합하는 것을 목표로 합니다. 이 연구를 통해 루간다어 화자가 영어를 몰라도 스마트 홈 장치를 제어할 수 있게 됩니다.

- **Technical Details**: 연구에서는 Mel Frequency Cepstral Coefficients(MFCCs) 를 음향 특징으로 활용하고, Convolutional Neural Network(Conv2D) 아키텍처를 사용하여 음성의 의도를 분류하는 NLP 모델을 Raspberry Pi에 배포합니다. 하드웨어적으로는 Raspberry Pi, Wio Terminal, ESP32 노드를 사용하며, MQTT(Message Queuing Telemetry Transport) 프로토콜을 통해 IoT 기기 간의 통신을 구현합니다. 루간다어 음성 명령 데이터셋을 수집하고 이를 오픈 소스로 제공하였습니다.

- **Performance Highlights**: 모델의 성능을 최적화하고 메모리 사용량을 줄이기 위해 데이터 증강 기법과 모델 양자화 기법을 사용하여 엣지 디바이스에서 실시간으로 작업이 가능하도록 했습니다. 이를 통해 루간다어 화자는 스마트 홈 장치를 효율적으로 제어할 수 있습니다.



### Sonos Voice Control Bias Assessment Dataset: A Methodology for Demographic Bias Assessment in Voice Assistants (https://arxiv.org/abs/2405.19342)
- **What's New**: 최근 음성 비서(voice assistant)가 모든 사용자에게 동일하게 작동하지 않는다는 연구 결과가 발표되었습니다. 그러나 음성 기술의 인구 통계적 강인성(demographic robustness) 에 대한 연구는 여전히 부족합니다. 이를 해결하기 위해 Sonos Voice Control Bias Assessment Dataset이 소개되었습니다. 이 데이터셋은 북미 영어로 된 음악 도메인에서 수집된 음성 비서 요청들로, 총 1,038명의 화자, 166시간, 170,000개 이상의 오디오 샘플, 그리고 9,040개의 고유 라벨링된 전사(transcript) 로 이루어져 있으며, 인구 통계적 다양성이 잘 통제되어 있습니다.

- **Technical Details**: 이 데이터셋은 성별, 연령, 방언 지역(dialectal region), 인종 등의 인구통계학적 요소들을 통제한 음성 비서 요청으로 구성되어 있습니다. 이 데이터셋은 또한 전사 정확도가 아닌 음성 언어 이해(Spoken Language Understanding, SLU) 지표를 활용한 통계적 편향 평가 방법론을 제시합니다. 데이터셋의 타당성을 검증하기 위해 최첨단 자동 음성 인식 및 언어 이해 모델을 사용한 실험을 수행하였으며, 코드는 GitHub에 공개되어 있습니다.

- **Performance Highlights**: 실험 결과 연령, 방언 지역, 인종에 따른 성능 차이가 통계적으로 유의미한 것으로 나타났습니다. 특히 다변량 테스트(multivariate tests)는 방언 지역, 성별 및 연령 간의 혼합 효과를 밝히는 데 중요한 역할을 합니다. 이는 음성 비서 시스템의 인구 통계적 편향을 이해하고 개선하기 위한 중요한 데이터셋과 방법론을 제공합니다.




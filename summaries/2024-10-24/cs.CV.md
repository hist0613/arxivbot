New uploads on arXiv(cs.CL)

### LongRAG: A Dual-Perspective Retrieval-Augmented Generation Paradigm for Long-Context Question Answering (https://arxiv.org/abs/2410.18050)
Comments:
          EMNLP 2024 Main

- **What's New**: 본 연구에서는 LongRAG라는 새로운 RAG 시스템 패러다임을 제안하며, 이 시스템은 long-context (긴 문맥) 질문 답변 (LCQA) 성능을 개선한다. LongRAG는 LLM 기반으로, 정보 추출 및 필터링, 생성 단계에서의 여러 전략을 통해 긴 문맥에서의 복잡한 지식을 이해하는 능력을 향상시킨다.

- **Technical Details**: LongRAG는 하이브리드 리트리버(hybrid retriever), LLM 증강 정보 추출기(LLM-augmented information extractor), Chain of Thought (CoT) 기반 필터(CoT-guided filter), LLM 증강 생성기(LLM-augmented generator)의 네 가지 주요 구성 요소를 포함한다. 이 시스템은 리트리버가 추출한 청크를 세분화하고, 이를 기반으로 LLM이 전역 정보를 잘 학습할 수 있도록 도와준다.

- **Performance Highlights**: LongRAG는 세 가지 멀티 홉 데이터셋(LongBench)에서 광범위한 실험을 통해 기존 long-context LLM(성능 향상 6.94%), 고급 RAG(성능 향상 6.16%), 기본 RAG(성능 향상 17.25%)를 초월하는 성능 향상을 보여주었다. 이 연구는 LongRAG의 구성 요소 및 미세 조정 전략의 효과를 정량적으로 분석하고 있으며, 다양한 도메인에서도 강력한 강건성과 전이 가능성을 보인다고 보고하였다.



### Key Algorithms for Keyphrase Generation: Instruction-Based LLMs for Russian Scientific Keyphrases (https://arxiv.org/abs/2410.18040)
Comments:
          The 12th International Conference on Analysis of Images, Social Networks and Texts (AIST'2024)

- **What's New**: 이번 연구는 러시아어 과학 논문의 요약에서 키워프(Keyword) 생성에 대한 프롬프트 기반 접근 방식을 평가하고, 기존의 수퍼바이즈드(Supervised) 및 언슈퍼바이즈드(Unsupervised) 방법과 비교한 것입니다. 또한, 제로샷(Zero-shot) 및 몇 샷(Few-shot) 프롬프트 방식의 성능을 분석하고, 효과적인 키워프 선택 전략을 제안합니다.

- **Technical Details**: 연구에서는 Math&CS 데이터셋을 사용하였으며, 프롬프트 기반의 키워프 생성 방법을 세 가지 오픈소스 LLM(대형 언어 모델)으로 실시하였습니다. Saiga, Vikhr, Mistral 모델이 사용되었고, 이들 모델은 각각 러시아어에 맞게 특별히 조정되었습니다.

- **Performance Highlights**: 연구 결과, 프롬프트 기반 방법이 기존의 언슈퍼바이즈드 방법과 비교하여 뛰어난 성과를 보였습니다. 인간 평가를 통해 생성된 키워프의 질을 측정하였으며, 단순한 텍스트 프롬프트를 사용하여도 기존 기법보다 우수한 결과를 나타냈습니다.



### MiLoRA: Efficient Mixture of Low-Rank Adaptation for Large Language Models Fine-tuning (https://arxiv.org/abs/2410.18035)
Comments:
          Accepted by EMNLP 2024 Findings. arXiv admin note: substantial text overlap with arXiv:2405.18203

- **What's New**: 이번 논문에서는 Mixture of Low-Rank Adaptation (MiLoRA)라는 새로운 LoRA 변형을 제안하여 다중 테넌트(multi-tenant) 환경에서의 지연(latency)을 크게 줄입니다. MiLoRA는 각 LoRA 모듈을 전문가(expert)로 고려하고 프롬프트 인식(prompt-aware) 라우팅 메커니즘을 사용하여 먼저 토큰 생성을 위한 전문가 라우팅 결과를 계산하고 이후 토큰에 대해 이 결과를 재사용함으로써 지연을 감소시킵니다.

- **Technical Details**: MiLoRA는 LoRA 모듈의 효율적인 사용을 위해 전문가 라우터를 활용하며, 전문가 라우팅 결과를 프롬프트가 주어진 후 첫 번째 새 토큰 생성을 바로 전에 한 번 계산하여, 이후 생성 단계에서는 이 결과를 재사용합니다. 이로 인해 LoRA 모듈과 MOE(Mixture-of-Experts) 라우터 추가로 인한 지연을 최소화합니다.

- **Performance Highlights**: MiLoRA는 일반적인 PEFT 방법들과 비교하여 더 낮은 지연을 보이면서도 다양한 상식 추론 작업, 수학 추론 작업 및 널리 사용되는 LLM 평가 벤치마크에서 일관되게 우수한 성능을 발휘함을 입증하였습니다. 특히, MiLoRA는 비슷한 조정 가능한 파라미터 예산으로 강력한 PEFT 기준선을 지속적으로 초과하는 성능을 보였습니다.



### Cross-lingual Transfer of Reward Models in Multilingual Alignmen (https://arxiv.org/abs/2410.18027)
- **What's New**: 본 논문은 인간 피드백을 통한 강화 학습(Reinforcement Learning with Human Feedback, RLHF)에서 여러 언어에서 훈련된 보상 모델(Reward Model, RM)의 교차 언어 전이 가능성을 조사합니다. 특히, 영어 데이터로 훈련된 RM이 다른 언어의 RM보다 우수한 성능을 보인다는 점을 강조합니다.

- **Technical Details**: 이 연구에서는 영어 RM이 다국어로 사전 훈련된 언어 모델(MLM) 위에 구축되었을 때 강력한 교차 언어 전이를 나타난다는 것을 보여줍니다. 실험 결과, 영어 RM이 Multilingual RewardBench에서 목표 언어 RM보다 평균 3~4% 더 높은 성능을 기록했습니다. 우리는 영어가 MLM의 표현을 잘 보존하며, 여러 언어에 대한 이해도가 강하다는 두 가지 이유로 설명합니다.

- **Performance Highlights**: 최종적으로, 연구 결과는 영어 RM이 다국어 지침 준수 능력을 향상시킬 수 있음을 입증하며, 네 가지 비영어 언어에서 9.5%의 평균 승률 증가를 보여주었습니다. 또한, 전통적인 RM과 비교하여 생성적 RM은 성능이 저조하다는 점도 강조되었습니다.



### Together We Can: Multilingual Automatic Post-Editing for Low-Resource Languages (https://arxiv.org/abs/2410.17973)
Comments:
          Accepted at Findings of EMNLP 2024

- **What's New**: 이 연구는 다국어 Automatic Post-Editing (APE) 시스템이 자원이 적은 Indo-Aryan 언어의 기계 번역 품질을 높일 수 있는 가능성을 탐구합니다. 영어-마라티 및 영어-힌디 두 언어쌍에 초점을 맞춰, 언어적 유사성을 활용하여 강력한 다국어 APE 모델을 개발했습니다.

- **Technical Details**: 연구에서는 품질 예측(Quality Estimation, QE)과 APE의 다중작업 학습(multi-task learning) 프레임워크를 통합하여 성능을 향상시키고 있습니다. 실험 결과, 다국어 APE 모델은 영어-힌디 및 영어-마라티 단일 모델보다 각각 2.5 및 2.39 TER 포인트 높은 성능을 보였으며, 다중작업 학습을 통한 1.29 및 1.44 TER 포인트의 성능 개선을 확인했습니다.

- **Performance Highlights**: 다국어 APE 모델은 단일 쌍 모델에 비해 성능이 높았고, 데이터 증강(data augmentation) 및 도메인 적응(domain adaptation) 기술을 사용하여 추가적인 성능 향상을 이루었습니다. 이 연구에서 생성된 합성 데이터, 코드 및 모델은 공공에 공개할 예정입니다.



### Dependency Graph Parsing as Sequence Labeling (https://arxiv.org/abs/2410.17972)
Comments:
          Accepted at EMNLP-2024

- **What's New**: 본 논문에서는 문법적 의존 구문 분석(syntactic dependency parsing)을 시퀀스 레이블링(sequence labeling) 문제로 변환하기 위해 제안된 다양한 선형화(linearization) 방법의 한계를 높이고, 이를 통해 그래프 기반 표현(graph-based representations)을 지원하는 새로운 접근 방식을 제시합니다.

- **Technical Details**: 기존의 선형화 방법은 재진입성(reentrancy)이나 사이클(cycles)을 처리하지 못하는 한계가 있어, 우리가 정의한 범위의 비한정(unbounded) 및 한정(bounded) 선형화를 통해 그래프 구문 분석(graph parsing)을 태그(task로 변환할 수 있습니다. 이러한 확장은 문제를 해결할 수 있는 도구상자를 확대합니다.

- **Performance Highlights**: 실험 결과, 잘 선택된 인코딩(encoding)을 사용할 경우 시퀀스 레이블링 의존 그래프 파서(dependency graph parsers)는 단순성에도 불구하고 높은 효율성과 거의 최신 기술(state of the art)에 근접한 정확도를 결합할 수 있음을 보여줍니다.



### Zeitenwenden: Detecting changes in the German political discours (https://arxiv.org/abs/2410.17960)
Comments:
          7 pages, 6 figures

- **What's New**: 이번 연구는 1949년 이후 독일 연방 의회의 모든 본회의 세션을 분석하여 정치 담론의 변화를 탐구합니다. 특히, 러시아-우크라이나 전쟁 이후 정치적 논의의 중대한 변곡점을 조명합니다.

- **Technical Details**: 연구에서는 RollingLDA라는 변형된 LDA(잠재 디리클레 할당) 모델을 사용하여 시간 경과에 따른 키워드 변화와 주제를 분석합니다. Pseudo-resampling 기법을 적용하여 문서의 변화를 탐지하고, 특정 시점의 콘텐츠 변화 및 연관 키워드 사용을 비교합니다.

- **Performance Highlights**: 이 분석 접근법은 독일 정치 담론에서 30개의 주요 주제를 효과적으로 분석하며, 전문가들이 정치적 변화에 대한 정성적 결과를 뒷받침할 수 있는 정량적 근거를 제공합니다.



### SimRAG: Self-Improving Retrieval-Augmented Generation for Adapting Large Language Models to Specialized Domains (https://arxiv.org/abs/2410.17952)
Comments:
          Work in Progress

- **What's New**: 이번 논문에서는 SimRAG라는 자가 학습(self-training) 접근 방식을 통해 복잡한 과학 및 의학 분야에 적합한 Retrieval-augmented generation (RAG) 시스템을 제안합니다. 이는 LLM (Large Language Model)이 질문 생성과 질문 응답 능력을 동시에 갖출 수 있도록 하는 방법입니다.

- **Technical Details**: SimRAG는 우선 다양한 데이터 세트에서 LLM을 세부 조정(fine-tuning)한 후, 라벨이 없는 도메인 관련 데이터에서 다양한 질문을 생성하도록 유도하는 방법으로 구성됩니다. 생성된 질문 중 질 높은 예시를 선별할 수 있는 추가적인 필터링 전략도 함께 사용됩니다.

- **Performance Highlights**: 11개 데이터 세트를 활용한 실험 결과, SimRAG는 기존의 기법들에 비해 성능이 1.2%에서 8.6% 향상된 것으로 확인되었습니다.



### ELAICHI: Enhancing Low-resource TTS by Addressing Infrequent and Low-frequency Character Bigrams (https://arxiv.org/abs/2410.17901)
Comments:
          11 pages, 1 figure, 3 tables

- **What's New**: 본 논문은 텍스트-음성 변환(Text-to-Speech, TTS) 기술의 발전을 통해 다른 언어에서의 이해 가능성을 개선하기 위한 세 가지 방법을 제안합니다. 이 방법들은 언어 자원이 부족한 경우에도 효과적으로 적용될 수 있습니다.

- **Technical Details**: 첫째, 관련 언어로부터 고품질 데이터를 활용하여 목표 언어의 TTS 성능을 개선합니다. 둘째, 비스튜디오 환경에서 기록된 저품질 자동 음성 인식(Automatic Speech Recognition, ASR) 데이터를 사용하고, 이를 노이즈 제거 및 음성 향상 모델을 통해 정제합니다. 셋째, 대규모 모델로부터 지식 증류(knowledge distillation)를 적용하여 합성 데이터를 통해 더 강력한 출력을 생성합니다.

- **Performance Highlights**: 히브리어를 대상으로 한 실험에서 제안한 모든 방법이 기존 방법들보다 우수한 성능을 보였습니다. 특히 다국어 훈련이 이해 가능성에서 가장 높은 점수를 기록했으며, 모델의 유연성과 정확성을 입증했습니다.



### Value Residual Learning For Alleviating Attention Concentration In Transformers (https://arxiv.org/abs/2410.17897)
- **What's New**: 이 논문에서는 Attention 집합이 심화 레이어에서 집중화되는 문제를 해결하기 위해 Residual Value를 추가한 ResFormer를 제안합니다. 이를 통해 초기 레이어의 값이 모든 후속 레이어에 직접적으로 접근 가능해지며, 이를 통해 다층 Attention의 집중화를 완화합니다.

- **Technical Details**: ResFormer는 해당 레이어의 Value 벡터와 첫 번째 레이어의 Value 벡터 간의 Residual 연결을 통해 Cross-layer Attention을 근사합니다. 이러한 접근 방식을 통해 깊은 레이어에서도 Attention 분산을 유지하며, SVFormer는 모든 레이어가 첫 번째 레이어의 Value 임베딩을 공유함으로써 KV Cache를 거의 50% 절감합니다.

- **Performance Highlights**: ResFormer는 Vanilla Transformer, DenseFormer, NeuTRENO와 비교하여 훈련 오류 및 다운스트림 작업에서 성능이 뛰어나며, SVFormer는 훈련 속도가 크게 향상되고 길이가 긴 시퀀스에 대해서도 더 나은 성능을 발휘합니다.



### Scaling Diffusion Language Models via Adaptation from Autoregressive Models (https://arxiv.org/abs/2410.17891)
Comments:
          25 pages. Code: this https URL

- **What's New**:  이 논문에서는 기존의 autoregressive (AR) 언어 모델을 변형하여 새로운 diffusion language models (DLMs)를 구축하는 방법을 제안합니다. 기존 DLM들의 한계를 극복하고, 대규모 데이터 세트를 활용한 효과적인 학습 방식을 제공합니다.

- **Technical Details**: Diffusion models는 Markov 프로세스를 통한 잠재 변수를 사용하여 작동합니다. 본 연구에서는 causal masking 방식이 아닌 bi-directional attention masks를 활용하여 DLM을 구축하는 방법을 설명합니다.

- **Performance Highlights**:  실험 결과, DiffuGPT와 DiffuLLaMA는 각각 127M에서 7B 파라미터 규모로, AR 모델보다 높은 성능을 보여줍니다. 특히, DiffuLLaMA는 인-context 학습과 코드 생성, 강력한 infilling 기능을 갖추고 있습니다.



### SpeakGer: A meta-data enriched speech corpus of German state and federal parliaments (https://arxiv.org/abs/2410.17886)
Comments:
          10 pages, 3 figures

- **What's New**: 이 논문에서는 독일의 모든 16개 연방 주의 의회와 독일 연방 의회의 연설 데이터를 포함한 SpeakGer 데이터 세트를 소개합니다. 이 데이터 세트는 1947년부터 2023년까지의 총 10,806,105개의 연설로 구성되어 있으며, 정치적 분석을 위한 메타 정보를 풍부하게 포함하고 있습니다.

- **Technical Details**: SpeakGer 데이터 세트는 청중 반응 정보, 연설자의 정당, 나이, 선거구와 같은 메타 데이터를 포함하여 정치적 경향성 분석을 가능하게 합니다. 데이터는 독일 의회의 공식 웹사이트에서 수집되었고, 또한 OCR(Optical Character Recognition)을 사용하여 문서의 텍스트를 추출하였습니다. Tesseract를 활용하며, 여러 가지 성능 개선 방법도 적용되었습니다.

- **Performance Highlights**: 세 가지 탐색적 분석 결과, 시간에 따른 정당의 주제 비율, 평균 연설자의 나이 변화 분석, COVID-19 관련 연설에 대한 다양한 정당의 감정 분석이 포함됩니다. 이 데이터는 특히 지역 간 연설 비교와 같은 세부적인 정치 연구를 가능하게 합니다.



### Understanding Layer Significance in LLM Alignmen (https://arxiv.org/abs/2410.17875)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)의 정렬 프로세스에서 가장 중요한 층을 식별하기 위한 새로운 접근 방식인 ILA(Important Layers for Alignment)를 제안합니다. 이 방법은 LoRA 알고리즘의 각 가중치 행렬에 대한 이진 마스크를 학습하여 각 층의 중요도를 나타냅니다.

- **Technical Details**: ILA는 층의 중요도를 이진 마스크로 표현하는 것으로, 마스크의 값이 0이면 해당 층의 영향이 미미함을, 1이면 해당 층이 정렬 프로세스에 중요하다는 것을 나타냅니다. 다양한 정렬 데이터셋에서 일관된 중요층 순위를 발견했으며, 25%의 중요하지 않은 층을 동결함으로써 모델의 성능이 향상된다는 것을 입증했습니다.

- **Performance Highlights**: 연구 결과는 가장 중요한 10-30%의 층만을 미세 조정하여도 전체 선형 층을 미세 조정한 것과 유사한 성능을 달성할 수 있음을 보여줍니다. 또한, QLoRA와 결합할 경우 30-75%의 핵심 층만 조정해서도 성능을 유지하거나 향상시킬 수 있으며, 자원 비용을 크게 줄일 수 있음을 강조합니다.



### Understanding When Tree of Thoughts Succeeds: Larger Models Excel in Generation, Not Discrimination (https://arxiv.org/abs/2410.17820)
Comments:
          Code: this http URL

- **What's New**: 이번 논문에서는 Tree of Thoughts (ToT)라는 새로운 추론 전략이 제시되었습니다. 이 전략은 생성기(generator)와 판별기(discriminator)를 활용하여 효율적인 추론 단계를 제안하고 이를 실행할 단계를 결정합니다.

- **Technical Details**: ToT는 Large Language Models (LLMs)에 적용되는 방법으로, Input-Output (IO) prompting 및 Chain-of-Thought (CoT) 추론과 비교했을 때 강력한 성능을 보여줍니다. 연구팀은 생성기와 판별기의 역할을 분리하여 분석하였으며, 생성기가 ToT의 성공에 더 중요한 역할을 한다고 밝혔습니다.

- **Performance Highlights**: 생성기를 확장함으로써 ToT 성능이 눈에 띄게 향상되는 반면, 판별기를 고정한 상태에서 판별기를 확장하는 경우 성과의 향상이 미미했습니다. 다양한 규모의 모델이 유사한 판별 능력을 가지지만, ToT에서 생성 성능은 모델에 따라 크게 차이를 보였습니다.



### OmniFlatten: An End-to-end GPT Model for Seamless Voice Conversation (https://arxiv.org/abs/2410.17799)
Comments:
          Work in progress

- **What's New**: 본 논문에서는 자연스러운 대화를 모델링하기 위한 새로운 End-to-End GPT 기반 모델 OmniFlatten을 소개합니다. 이 모델은 사용자와 시스템 간의 실시간으로 전통적인 turn-based 시스템보다 뛰어난 full-duplex 대화를 가능하게 합니다.

- **Technical Details**: OmniFlatten은 multi-stage 포스트 트레이닝 방법론을 통해 텍스트 기반 LLM의 구조를 변경하지 않고도 speech-text 대화 모델로 발전시킵니다. 데이터 표준화를 위해 flattening 작업을 수행하여 다양한 모달리티와 작업 간의 학습 방식을 통합합니다. 훈련 프로세스는 modality alignment, half-duplex dialogue learning 및 full-duplex dialogue learning의 세 단계로 구성됩니다.

- **Performance Highlights**: OmniFlatten은 ASR과 TTS 작업을 통한 조정 및 훈련을 통해 명확한 대화 품질을 제공합니다. 생성된 대화의 평균 응답 시간은 시스템이 턴을 가질 경우 160ms, 사용자가 턴을 가질 경우 805ms로 나타났습니다. 결과적으로 모델은 모달리티 정렬 및 half-duplex 학습 단계에서 모델의 full-duplex 대화 능력을 개선하는 성과를 보였습니다.



### Leveraging the Domain Adaptation of Retrieval Augmented Generation Models for Question Answering and Reducing Hallucination (https://arxiv.org/abs/2410.17783)
Comments:
          Initial Version fine-tuned on HotelConvQA

- **What's New**: 이 논문은 고객 서비스 분야에서의 대화형 AI 시스템 개발을 위한 Retrieval Augmented Generation (RAG) 아키텍처의 효과성을 연구했습니다. 최근에 도입된 RAG-end2end 모델은 도메인 적응에 대한 성능 개선을 보여주며, 하늘의 환상(hallucinations) 발생을 줄이는 방안들에 대해 학습했습니다.

- **Technical Details**: 연구진은 HotelConvQA라는 새로운 데이터셋을 구축하고, 다양한 RAG 및 RAG 유사 아키텍처의 성능을 평가했습니다. 이 모델들은 도메인 특정 데이터에 대해 미세 조정되어 정확하고 관련성 높은 응답을 생성할 수 있는 능력을 비교했습니다. 또한, 각 RAG 아키텍처에서 도메인 적응이 환상을 줄이는 영향을 평가했습니다.

- **Performance Highlights**: 모델 성능 평가 결과, 도메인 적응을 통해 QA 작업에서 강력한 성능을 입증했습니다. 이는 모든 RAG 아키텍처의 환상 발생을 줄이는데도 긍정적인 영향을 미쳤습니다. 연구 결과는 고객 서비스 분야에서의 RAG 모델 사용이 신뢰성과 정확성을 높이는데 기여할 수 있음을 보여줍니다.



### Latent Structures of Intertextuality in French Fiction (https://arxiv.org/abs/2410.17759)
Comments:
          13 pages, 6 figures. Computational Humanities Research Conference 2024

- **What's New**: 이 논문은 문학 이론에서 중요한 개념인 Intertextuality(상호텍스트성)를 탐구하며, 텍스트를 방대한 Intertextual network(상호텍스트망)의 일부분으로 간주하고 있습니다. 특히, 18세기, 19세기 및 20세기 초의 12,000개 이상의 프랑스 허구 작품들로 구성된 코퍼스를 통해 두 가지 문학 개념인 sub-genres(하위 장르)와 literary canon(문학 정전)의 역할을 평가합니다.

- **Technical Details**: 이 연구는 최신 contextual language models(맥락적 언어 모델)을 활용하여 소설을 인코딩하고 단순한 어휘적 또는 주제적 접근을 초월하는 특징을 캡처하는 방법론을 개발하였습니다. 또한, Gallica의 'Fictions littéraires' 컬렉션을 기반으로 한 19,240개의 문학 작품을 사용하여 Intertextuality를 모델링합니다.

- **Performance Highlights**: 연구 결과, 하위 장르와 정전성이 프랑스 허구 내 텍스트 유사성을 형성하는 데 중요한 역할을 한다는 것을 확인했습니다. 특히, 문학 정전의 영향력은 시간에 걸쳐 지속적이며, 문학 전통의 형성에 중대한 영향을 미침을 발견했습니다.



### Local Contrastive Editing of Gender Stereotypes (https://arxiv.org/abs/2410.17739)
Comments:
          Accepted at EMNLP 2024

- **What's New**: 본 논문에서는 언어 모델(이하 LM)에서 성 고정관념에 대한 편향을 정확하게 로컬화하고 수정할 수 있는 '로컬 대조 편집(Local Contrastive Editing)' 기법을 소개합니다. 이 기법은 참조 모델과 비교하여 목표 모델의 특정 가중치 부분 집합을 조정하는 방법으로, 성 고정관념을 포함한 편향을 제어하는 데 활용됩니다.

- **Technical Details**: 로컬 대조 편집은 두 단계로 구성된 접근 방식입니다. 첫 번째 단계에서는 비구조적 가지치기(unstructured pruning)를 통해 LM에서 성 고정관념을 인코딩하는 개별 가중치를 정확히 찾아냅니다. 두 번째 단계에서는 가중치 보간(weight interpolation)이나 추가 가지치기(pruning)와 같은 다양한 로컬 편집 전략을 활용하여 식별된 가중치를 조정합니다.

- **Performance Highlights**: 실험을 통해 우리는 0.5% 미만의 가중치 부분 집합을 수정하여 측정 가능한 편향을 완화하는 데 성공했음을 입증했습니다. 이러한 로컬 편집 전략들은 모델의 기능성을 유지하면서 성 편향을 유연하게 조정할 수 있는 가능성을 보여줍니다.



### MojoBench: Language Modeling and Benchmarks for Mojo (https://arxiv.org/abs/2410.17736)
- **What's New**: 최근 Modular에서 도입한 Mojo 프로그래밍 언어는 Python에 비해 상당한 속도 향상을 주장하며 과학 커뮤니티에서 큰 관심을 받고 있습니다. 그러나 다양한 프로그래밍 언어에서 코드 Large Language Models (LLMs)의 발전에도 불구하고 Mojo는 이 맥락에서 탐구되지 않았습니다. 이를 해결하기 위해 MojoBench라는 Mojo 코드 생성을 위한 첫 번째 프레임워크를 소개합니다.

- **Technical Details**: MojoBench는 코드 LLM을 Mojo에서 평가하기 위해 설계된 benchmark dataset인 HumanEval-Mojo와 Mojo 코드 생성을 위해 사전 훈련(pretrained) 및 미세 조정(finetuned)된 첫 번째 LLM인 Mojo-Coder를 포함합니다. Mojo-Coder는 5가지 자연어(NLs)를 지원하는 지침을 제공합니다.

- **Performance Highlights**: 우리의 결과는 Mojo-Coder가 GPT-4o 및 Claude-3.5-Sonnet과 같은 주요 모델에 비해 30-35% 성능 향상을 달성했음을 보여줍니다. 또한, 우리는 저대표적인(underrepresented) 및 미존재(unseen) 프로그래밍 언어와 관련된 LLM의 행동에 대한 통찰력을 제공하며, 모델의 적응력 향상을 위한 잠재적 전략을 제시합니다.



### Dialectal and Low Resource Machine Translation for Aromanian (https://arxiv.org/abs/2410.17728)
Comments:
          16 pages, 3 figures, 6 tables, submitted to COLING 2025

- **What's New**: 이 논문에서는 루마니아어, 영어, 그리고 위기 언어인 아로마니아어(romanian, english, aromanian) 간의 번역을 가능하게 하는 신경 기계 번역 시스템을 처음으로 제시합니다. 연구팀은 79,000개의 정제된 문장 쌍으로 구성된 최대 아로마니아-Romanian 이중 언어 코퍼스를 공개합니다.

- **Technical Details**: 연구에서는 아로마니아어와 루마니아어 간의 기계 번역 시스템을 학습하기 위해 다장르 병렬 코퍼스를 구축했습니다. 이를 통해 LaBSE 문장 인코더를 아로마니아어 지원에 맞게 미세 조정하고, NLLB-200 모델을 통해 번역 성능을 비교합니다. 또한, 다양한 instruction-tuned large language models (LLMs)를 사용할 수 있도록 미세 조정했습니다.

- **Performance Highlights**: BLEU 점수는 텍스트의 방향과 장르에 따라 17에서 32까지 다양하며, 이는 아로마니아어와 루마니아어 사이의 번역 성능을 나타냅니다. 연구 결과는 아로마니아어 언어 보존에 기여할 것으로 기대됩니다.



### CogSteer: Cognition-Inspired Selective Layer Intervention for Efficient Semantic Steering in Large Language Models (https://arxiv.org/abs/2410.17714)
- **What's New**: 이번 논문에서는 큰 언어 모델(LLMs)의 해석 가능성을 높이기 위해 눈 움직임 측정치를 사용하여 LLM의 동작을 분석하는 새로운 방법론을 제안합니다. 눈 움직임 연구에서 수집된 데이터를 활용하여 LLM의 다양한 레이어에 걸친 행동을 해석하고, 이를 통해 중간 레이어에서의 의미 조작을 위한 효율적인 방법인 CogSteer를 도입합니다.

- **Technical Details**: 제안된 방법은 LLM의 레이어 속 숨겨진 상태와 눈 움직임 지표를 연관 짓고, 이를 바탕으로 최적의 레이어를 선택하여 의미를 조작합니다. CogSteer 방법은 특히 언어의 독성화(toxification) 및 비독성화(detoxification)의 실험을 통해 효과성을 입증하였고, 실행 시 97%의 계산 자원 절약과 60%의 훈련 시간 절감을 달성합니다. 또한, 암묵적 레이어 대조 개입 방법이 소개되어 안전한 생성 방향으로 의미를 유도합니다.

- **Performance Highlights**: CogSteer 방법은 언어 독성 점수를 87.2%에서 59.1%로 감소시키며, LLaMa2-7B 모델에서는 61.0%로 낮아졌습니다. 이 연구의 결과는 정확하고 신뢰성을 갖춘 모델 배치를 위한 해석 가능성 향상에 기여하며, LLMs의 다양한 모델에 적용될 수 있습니다.



### Beware of Calibration Data for Pruning Large Language Models (https://arxiv.org/abs/2410.17711)
Comments:
          under review

- **What's New**: 최근 연구에서는 대규모 언어 모델(LLMs)의 성능 향상과 배포 비용 절감을 위한 모델 압축 기술, 특히 Post-training pruning에 대한 중요성이 강조되었습니다. 이 논문은 다양한 보정(calibration) 데이터가 pruning 성능에 미치는 영향을 체계적으로 탐구하며, 보정 데이터의 중요성이 고급 pruning 전략 설계보다 더 크다는 점을 발견하였습니다.

- **Technical Details**: Post-training pruning은 반복 학습 없이도 파라미터 중요도를 추정할 수 있는 방법입니다. 이 연구에서는 작은 양의 보정 데이터를 사용하여 파라미터의 중요도를 평가하고, 이를 통해 sparsity가 높은 상황에서도 효과적인 pruning을 가능하게 합니다. 파라미터 중요도 추정에는 inverse Hessian matrix와 L2 노름을 활용하며, 보정 데이터는 훈련 데이터와 유사할수록 성능이 우수하다는 결과를 보여줍니다.

- **Performance Highlights**: DCLM 및 LLaMA-3 모델에서 실험을 수행한 결과, 제안된 보정 데이터 샘플링 방법이 일반적으로 사용되는 보정 데이터보다 우수한 성능을 나타내었으며, 강력한 pruning 기법들과의 호환성을 유지하면서 성능을 크게 향상시켰습니다.



### An Adaptive Framework for Generating Systematic Explanatory Answer in Online Q&A Platforms (https://arxiv.org/abs/2410.17694)
Comments:
          10 pages, 6 figures

- **What's New**: 본 논문에서는 복잡한 질문에 대한 해답 생성을 위한 새로운 프레임워크인 SynthRAG를 제안합니다. 기존의 RAG 모델들이 섬세하게 정리된 답변 제공에 한계가 있음을 지적하고, 정보를 통합하는 체계적 접근 방식을 통해 QA 시스템의 성능을 향상시키는 방법을 설명합니다.

- **Technical Details**: SynthRAG는 적응형 개요 생성(adaptive outline generation), 체계적 정보 생성(systematic information generation), 맞춤형 답변 생성(customized answer generation)의 세 가지 주 단계를 포함합니다. 이 모델은 다양한 질문에 맞춤화된 개요를 생성하고, 각 하위 섹션에 대한 일관된 세부 단락을 생성하여 논리적 구조가 갖춰진 포괄적이고 상세한 응답을 제공합니다.

- **Performance Highlights**: Empirical evaluations에 따르면, SynthRAG는 복잡한 질문을 처리하는 데 있어 기존 RAG 모델보다 뛰어난 성능을 보이며, Zhihu 플랫폼에서의 배포 결과도 그 응답이 사용자의 주목을 받았음을 보여줍니다(평균 5.73개의 업보트). 이는 SynthRAG가 실제 적용에서 큰 영향을 미칠 수 있는 가능성을 나타냅니다.



### Towards a Similarity-adjusted Surprisal Theory (https://arxiv.org/abs/2410.17676)
Comments:
          EMNLP 2024 main conference proceedings

- **What's New**: 본 논문은 기존의 surprisal 이론의 한계를 극복하기 위해 정보 가치를 도입하고, 유사 조정 surprisal을 통해 단어 간의 유사성을 반영한 새로운 예측 변수를 제안합니다.

- **Technical Details**: 유사 조정 surprisal은 Ricotta와 Szeidl의 다양성 지수를 기반으로 하며, 단어의 예측 가능성을 유사한 대안과 비교하여 계산합니다. 이는 기존의 surprisal과 정보 가치 간의 수학적 관계를 드러냅니다.

- **Performance Highlights**: 실험 결과, 유사 조정 surprisal은 특정 데이터셋에서 표준 surprisal 이상으로 예측력을 제공하여, 이해 노력(comprehension effort)의 보완적인 지표로 작용함을 시사합니다.



### Quantifying the Risks of Tool-assisted Rephrasing to Linguistic Diversity (https://arxiv.org/abs/2410.17670)
- **What's New**: 이 논문은 작성 도구와 대규모 언어 모델(LLM)의 사용이 작성된 텍스트의 언어와 어휘에 미치는 영향을 양적으로 분석합니다. 이러한 도구들이 사용자 기반에서 어떻게 텍스트의 다양성을 줄일 수 있는지를 탐구합니다.

- **Technical Details**: 본 연구에서는 Grammarly, Quillbot과 같은 기존의 작성 보조 도구(WATs)와 ChatGPT, GPT-4 등의 LLM을 포함하여 총 8개의 도구로 다양한 텍스트 유형을 재구성했습니다. 텍스트의 길이, 어휘 크기, 의미적 유사성과 같은 여러 지표를 사용해 변화를 측정하여 비교 분석하였습니다. 또한, 텍스트 변화는 문장 수준과 벡터 수준에서 모두 평가되었습니다.

- **Performance Highlights**: 작성 도구(WATs)는 일반적으로 텍스트 길이를 크게 변경하지 않고, 예외적으로 Rephrase 도구는 약간의 연장을 보였습니다. 반면, LLM들은 평균적으로 텍스트를 현저히 단축시켰으며, 이러한 경향은 다양한 텍스트 유형에서 일관되게 나타났습니다. 어휘 크기 측면에서, 대부분의 WATs는 어휘 크기를 줄이는 경향을 보였으며, LLM도 비슷하게 작용하나 좀 더 극단적인 감소를 보였습니다.



### ReflecTool: Towards Reflection-Aware Tool-Augmented Clinical Agents (https://arxiv.org/abs/2410.17657)
Comments:
          20 pages

- **What's New**: 이 논문에서는 ClinicalAgent Bench (CAB)라는 종합적인 의료 에이전트 벤치마크를 소개하며, 18개의 과제와 5개의 주요 차원을 포함한다. 또한, 도메인 특화 도구를 활용하는 ReflecTool이라는 혁신적인 프레임워크를 제안한다.

- **Technical Details**: ReflecTool은 두 단계로 구성된다: 첫 번째는 최적화 단계로, 에이전트가 도구를 사용하여 문제를 해결하고 성공적인 경로를 장기 메모리에 저장한다. 두 번째는 추론 단계로, 에이전트가 장기 메모리에서 비슷한 성공 사례를 검색하여 도구 선택을 최적화한다. 두 가지 검증 방법인 Iterative Refinement와 Candidate Selection을 통해 도구 사용 경험을 개선한다.

- **Performance Highlights**: ClinicalAgent Benchmark에서 ReflecTool은 순수 LLMs보다 10점 이상, 그리고 기존의 에이전트 기반 방법들보다 3점 향상된 성능을 보여주며 복잡한 임상 작업을 해결하는 데 효과적임을 입증했다.



### LMLPA: Language Model Linguistic Personality Assessmen (https://arxiv.org/abs/2410.17632)
- **What's New**: 본 논문은 LLM(Large Language Model)의 언어적 성격을 평가하기 위한 Language Model Linguistic Personality Assessment (LMLPA) 시스템을 도입하여, LLM의 언어 생성 능력을 정량적으로 이해하는 방법을 제공합니다.

- **Technical Details**: LMLPA 시스템은 Big Five Inventory를 기반으로 하여 LLM의 작동 능력과 일치하도록 설계된 성격 평가 설문지를 사용합니다. 이 설문지는 개방형 질문으로 구성되어, AI 레이터가 텍스트 반응으로부터 명확한 성격 특성의 수치 지표를 추출하는 과정을 포함합니다. 또한 주성분 분석(Principal Component Analysis) 및 신뢰도 검증을 통해 LLM의 뚜렷한 성격 특성을 정량적으로 평가하는 방법을 제시합니다.

- **Performance Highlights**: 연구 결과는 LMLPA를 통해 LLM이 뚜렷한 성격 특성을 지닌다는 것을 보여주며, 이는 교육, 제조 등 다양한 분야에서 AI 성격 평가의 정교화 및 활용을 확장하는 데 기여할 것입니다. 예를 들어, LLM의 성격을 평가함으로써 교육 환경에서 어떤 성격 특성이 더 나은 학습 곡선을 제공할 수 있는지를 조사할 수 있습니다.



### Graphusion: A RAG Framework for Knowledge Graph Construction with a Global Perspectiv (https://arxiv.org/abs/2410.17600)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2407.10794

- **What's New**: 이 논문은 자연어에서 지식 그래프(knowledge graph)를 제로샷(zero-shot) 방식으로 구성하는 Graphusion 프레임워크를 소개합니다. 기존 방식들과 달리, 이 방법은 개별 문서에 국한되지 않고 전반적인 지식 통합을 목표로 합니다.

- **Technical Details**: Graphusion의 구성은 세 가지 단계로 이루어집니다: 1단계에서는 토픽 모델링을 통해 시드 엔티티(seed entity)를 추출합니다; 2단계에서는 LLMs를 통해 후보 triplet 추출을 진행합니다; 3단계에서는 통합 모듈을 설계하여 지식의 전반적인 관점을 제공합니다.

- **Performance Highlights**: Graphusion은 엔티티 추출과 관계 인식을 위해 각각 2.92와 2.37의 점수를 기록했습니다. 교육 분야에서 새로운 전문 인증 벤치마크 TutorQA를 통한 9.2%의 정확도 향상도 보여줍니다.



### Cross-model Control: Improving Multiple Large Language Models in One-time Training (https://arxiv.org/abs/2410.17599)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 이 논문은 Cross-model Control (CMC)라는 새로운 방법을 제안합니다. 이는 한 번의 학습으로 여러 개의 대형 언어 모델(LLMs)을 개선할 수 있는 방식으로, 소형 언어 모델을 사용합니다. 저자들은 서로 다른 모델 간의 로짓 변화가 유사하다는 점을 발견하였고, 이를 기반으로 소형 모델이 큰 모델의 로짓 출력을 조절할 수 있도록 하였습니다.

- **Technical Details**: CMC는 소형 언어 모델을 통해 다른 모델의 최적화를 가능하게 합니다. 구체적으로, 저자들은 PM-MinED라는 새로운 토큰 매핑 전략을 제안하여 다양한 어휘를 가진 모델들 사이의 호환성을 확보합니다. 이 방법은 대형 모델과 소형 모델이 함께 학습하도록 하여, 소형 모델이 대형 모델의 로짓 출력을 변경할 수 있게 합니다.

- **Performance Highlights**: 저자들은 CMC의 효과성을 입증하기 위해 광범위한 실험을 수행하였으며, 단 1,500만 개의 파라미터를 가진 소형 언어 모델이 700억 개의 파라미터를 가진 대형 모델에 비슷한 성능 향상을 가져올 수 있음을 보여주었습니다. 이러한 접근법은 모델 소유자가 데이터와 컴퓨팅 자원이 부족할 경우에도 다른 LLM에서의 미세 조정 결과를 활용할 수 있게 합니다.



### MM-Eval: A Multilingual Meta-Evaluation Benchmark for LLM-as-a-Judge and Reward Models (https://arxiv.org/abs/2410.17578)
Comments:
          work in progress

- **What's New**: 이 논문에서는 LLM(대규모 언어 모델)의 평가자로서의 신뢰성을 검증하기 위한 다국어 메타-평가 벤치마크인 MM-Eval을 소개합니다. 기존의 메타-평가 벤치마크가 주로 영어에 국한되었던 것과 달리, MM-Eval은 18개 언어를 지원하여 다양한 언어적 문맥에서 LLM의 효과성을 평가할 수 있도록 설계되었습니다.

- **Technical Details**: MM-Eval는 6개의 카테고리로 구성된 18개 언어에 대한 다국어 메타-평가 벤치마크입니다. 이 벤치마크는 언어적 도전 과제를 포함하여 언어적 환각(language hallucinations)과 같은 다양한 차원을 평가합니다. MM-Eval에서 평가된 12개의 LLM 모델들은 평균 정확도 68.9%로, 낮은 자원 언어에서의 성능 저하가 두드러졌습니다.

- **Performance Highlights**: LLM 평가자들은 전반적으로 개선의 여지가 많으며, 특히 낮은 자원 언어에 대해서는 좋은 응답에 낮은 점수를 부여하고 나쁜 응답에 높은 점수를 부여하는 경향을 보였습니다. 개별 모델의 점수는 다양한 카테고리에서 상이했으며, Skywork-Reward-Gemma-2-27B 모델이 Safety 카테고리에서 가장 높은 성능을 기록했습니다.



### ESpeW: Robust Copyright Protection for LLM-based EaaS via Embedding-Specific Watermark (https://arxiv.org/abs/2410.17552)
- **What's New**: 이 논문에서는 Embeddings as a Service (EaaS)의 저작권 보호를 위해 기존의 물 리의 취약점을 보완한 새로운 임베딩 특정 워터마킹(ESpeW) 방법을 제안합니다.

- **Technical Details**: ESpeW는 각 임베딩에 고유하고 식별 가능한 워터마크를 주입하는 혁신적인 접근 방식을 사용합니다. 이 방법은 워터마크가 서로 상당한 거리를 두도록 설계되어 있으며, 공통 구성 요소를 공유하지 않아 제거를 어렵게 만듭니다. 실험 결과, ESpeW는 다양한 물 리 제거 공격 강도에서도 성공적으로 워터마킹할 수 있습니다.

- **Performance Highlights**: ESpeW는 기존의 워터마킹 대비 저작권 보호가 더욱 강력하며, 네 가지 인기 데이터세트에서 테스트 결과, 유사한 공격 환경에서도 그 효과를 유지하는 것으로 나타났습니다.



### ProtoLens: Advancing Prototype Learning for Fine-Grained Interpretability in Text Classification (https://arxiv.org/abs/2410.17546)
- **What's New**: ProtoLens는 텍스트 분류를 위한 새로운 프로토타입 기반 모델로, 서브 문장 수준의 세밀한 해석 가능성을 제공합니다.

- **Technical Details**: ProtoLens는 Prototype-aware Span Extraction 모듈을 사용하여 학습된 프로토타입과 관련된 텍스트 스팬을 식별하고, Prototype Alignment 메커니즘을 통해 훈련 과정에서 프로토타입이 의미적으로 일관되도록 보장합니다. 이로 인해 해석 가능한 예측을 제공하면서도 경쟁력 있는 정확도를 유지합니다.

- **Performance Highlights**: ProtoLens는 여러 텍스트 분류 벤치마크에서 프로토타입 기반 모델과 비해 해석 불가능한 기준선보다도 더 우수한 성능을 보였습니다.



### Responsible Multilingual Large Language Models: A Survey of Development, Applications, and Societal Impac (https://arxiv.org/abs/2410.17532)
- **What's New**: 다국어 대형 언어 모델(MLLMs)의 발전이 인공지능을 언어적 경계를 넘어 민주화하는 데 있어 중요한 진전을 이루었다. 이 논문은 MLLM을 개발하고 배포하기 위한 포괄적인 종합 프레임워크를 제공함으로써 이론적 기초와 실제 구현 사이의 격차를 해소하고자 한다.

- **Technical Details**: 이 연구는 데이터 전처리에서 배포까지의 실행 가능한 파이프라인을 제시하며, Llama2를 사례 연구로 활용하여 다국어 기능 강화를 위한 최적화 전략을 자세히 설명한다. 여기에는 고자원 언어와 저자원 언어 간의 균형을 맞추기 위한 커리큘럼 학습 접근법, 토큰화 전략 최적화, 효과적인 샘플링 방법이 포함된다.

- **Performance Highlights**: 논문은 고객 서비스, 검색 엔진, 기계 번역 등 실제 애플리케이션을 통해 언어적 다양성을 지원하는 데 있어 중요한 과제가 있음을 밝혀낸다. 또한 88.38%의 세계 언어가 저자원으로 분류되며, 이는 10억 이상의 화자가 영향을 받는다는 점을 강조한다.



### Navigate Complex Physical Worlds via Geometrically Constrained LLM (https://arxiv.org/abs/2410.17529)
- **What's New**: 본 연구는 Large Language Models(LLMs)의 텍스트 지식 기반 물리적 세계 재구성과 구성을 탐구합니다. 모델 성능이 공간 인지 능력에 미치는 영향을 분석하며, 기하학적 규칙과 다계층 그래프, 다중 에이전트 시스템 프레임워크에 기반을 둔 워크플로우를 개발합니다. 이 연구는 LLM이 공간 환경에서 다단계 다목적 기하학적 추론을 수행하는 방법을 분석합니다.

- **Technical Details**: 연구는 LLM의 사전 훈련 지식을 활용하여 복잡한 기하학적 제약 문제를 해결하는 유전 알고리즘을 사용합니다. 기하학적 규약을 통일하여 다계층 그래프 아래에서 공간 구성을 수행하며, LLM간의 정보 일관성을 보장하는 멀티 에이전트 접근법을 도입합니다. GPT-4 모델은 공간 구성 작업에서 우수한 성능을 발휘합니다.

- **Performance Highlights**: 연구 결과, GPT-4는 공간 구성 과제에서 GPT-3.5-turbo보다 뛰어난 성능을 발휘하며, 다계층 그래프 기반 접근법이 LLM의 공간 이해 및 추론 능력을 향상시키는 데 기여합니다. 그래프 데이터베이스를 통해 복잡한 기하학적 관계를 유연하게 관리하며, 3D 오브젝트 생성 과정에서 정보의 일관성과 정확성을 보장합니다.



### Large Language Models Still Exhibit Bias in Long Tex (https://arxiv.org/abs/2410.17519)
Comments:
          22 page, 38 figures, Neurips (SoLaR Workshop)

- **What's New**: 기존의 대형 언어 모델(LLMs)에 대한 공정성 기준은 주로 다중 선택 질문과 같은 간단한 작업에 초점을 맞추고 있었으나, 본 논문은 더 복잡한 시나리오인 긴 텍스트 생성에서 발생할 수 있는 편견을 다룬다는 점에서 새로운 기여를 하고 있습니다.

- **Technical Details**: 우리는 LTF-TEST(Long Text Fairness Test)라는 프레임워크를 소개하여, 에세이 스타일 프롬프트를 통해 LLM의 편견을 평가합니다. LTF-TEST는 14개 주제와 10개 인구 통계 축을 커버하며, 여기에는 성별과 인종이 포함됩니다. 총 11,948개의 샘플을 통해 모델 응답과 그 뒤의 추론을 평가하여, 간단한 응답에서는 감지하기 어려운 미세한 편향을 드러냅니다.

- **Performance Highlights**: 5개의 최신 LLM(GPT-4o 및 LLaMa3 포함)에 대한 평가에서 두 가지 주요 편향 패턴을 발견했습니다. 첫째, 이러한 모델들은 특정 인구 통계 그룹을 선호하는 경향이 있습니다. 둘째, 전통적으로 불리한 그룹에 과도하게 민감하게 반응하며, 종종 다른 그룹을 소홀히 합니다. 이를 해결하기 위해 편향된 프롬프트와 중립적인 응답을 쌍으로 하여 편향을 줄이는 FT-REGARD 방법을 제안하며, 성별 편향을 34.6% 줄이고 BBQ 벤치마크에서 성능을 1.4%포인트 향상시킵니다.



### VoiceTextBlender: Augmenting Large Language Models with Speech Capabilities via Single-Stage Joint Speech-Text Supervised Fine-Tuning (https://arxiv.org/abs/2410.17485)
- **What's New**: 최근 연구에서는 SpeechLMs (Speech Language Models)를 사용하여 음성 기능을 지원하는 대형 언어 모델 (LLMs)을 발전시켰습니다. 이 논문에서는 이전에 단일 발화 음성 기반 질문 응답 (QA)에 초점을 맞춘 SpeechLM을 다중 발화 대화로 확장하며, 복잡한 다단계 최적화 (SFT)를 줄일 수 있는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 우리는 새로운 단일 단계의 공동 음성-텍스트 SFT(Supervised Fine-Tuning) 접근 방식을 제안합니다. 이 방식은 LLM의 저계수 적응 (LoRA)을 기반으로 하며, 텍스트 전용 SFT 데이터와 세 가지 유형의 음성 관련 데이터를 결합합니다: 음성 인식 및 번역, 음성 기반 QA, 혼합 모달 SFT. 이 과정에서 ‘VoiceTextBlender’라는 음성-텍스트 언어 모델을 개발하였습니다.

- **Performance Highlights**: 우리의 3B 모델은 이전의 7B 또는 13B 파라미터의 SpeechLM보다 다양한 음성 벤치마크에서 더 우수한 성능을 보여주었으며, 텍스트 전용 작업의 원래 기능도 유지했습니다. 또한, 새로운 프롬프트와 과제를 효과적으로 처리할 수 있는 emergent abilities(비상 능력)를 보였습니다.



### Is artificial intelligence still intelligence? LLMs generalize to novel adjective-noun pairs, but don't mimic the full human distribution (https://arxiv.org/abs/2410.17482)
Comments:
          9 pages (23 pages with appendix). Accepted to GenBench 2024

- **What's New**: 이 연구는 LLM(대규모 언어 모델)이 주어진 문맥에 따라 형용사-명사 조합의 의미를 이해하고 새로운 조합에도 일반화할 수 있는 능력을 테스트하는 것을 목표로 하고 있습니다.

- **Technical Details**: 저자들은 LLM의 성능을 평가하기 위해 문맥이 없는 상태에서 LLM의 추론을 펼치는 세 가지 방법을 제안하고, LLM이 인간의 판단과 유사한 분포를 보이는지를 평가합니다. 채택된 필자는 여러 크기의 LLM을 비교하고, 그 중에서도 최근 모델이 인간의 성향과 밀접하게 일치하는 것을 발견했습니다.

- **Performance Highlights**: 연구 결과, LLM은 데이터셋의 75%에서 인간과 유사한 추론을 보이나, 높은 변동성이 있는 조합에서는 어려움을 겪었습니다. "인공지능이 여전히 인공지능인가?"라는 질문에 대해 LLM은 대부분의 사람보다 더 긍정적인 평가를 내렸습니다.



### Do Robot Snakes Dream like Electric Sheep? Investigating the Effects of Architectural Inductive Biases on Hallucination (https://arxiv.org/abs/2410.17477)
- **What's New**: 본 논문은 대형 언어 모델(LLM)의 구조적 변화가 허위정보 생성(hallucinations) 경향에 미치는 영향을 폭넓게 분석합니다. 또한, 기존의 연구들이 Transformer 기반 모델에만 집중된 데 반해, 순환 모델(recurrent models)과의 비교를 통해 이러한 문제를 동시에 고려할 필요성을 강조합니다.

- **Technical Details**: 연구에서는 1B에서 70B 규모의 다양한 오픈 소스 LLM을 사용하여 20개의 다양한 허위정보 발생 관련 태스크를 평가하였습니다. 자료는 충실도(faithfulness) 및 사실성(factuality)으로 구분되며, 레이어 설계, 연산 규모 및 교육 방법(instruction-tuning)에 따라 모델 아키텍처의 다양성을 평가합니다.

- **Performance Highlights**: 연구 결과, Transformer 기반 모델과 순환 모델 간에 허위정보 생성 경향에는 큰 차이가 없지만, 특정 태스크에 따라 아키텍처의 특성이 허위정보의 발생 빈도에 영향을 미친다는 점이 드러났습니다. 특히, 순환 모델은 소형 크기에서 더 높은 충실도를 보이며, 교육 시 모델 크기가 증가함에 따라 사실성 허위정보의 의존성이 두드러진다는 점을 강조합니다.



### In Context Learning and Reasoning for Symbolic Regression with Large Language Models (https://arxiv.org/abs/2410.17448)
- **What's New**: 이 연구에서 제안된 방법은 대형 언어 모델(LLMs)이 기호 회귀(symbolic regression) 문제를 해결할 수 있는 가능성을 조사합니다. GPT-4를 활용하여 주어진 데이터에서 수식(expressions)을 생성하고, 이를 외부 Python 도구를 사용하여 최적화하고 평가합니다.

- **Technical Details**: 기호 회귀는 데이터 세트에서 간단하고 정확한 수식을 찾기 위한 머신러닝 방법입니다. 이 연구는 GPT-4를 통해 데이터 분석, 이전의 수식, 그리고 과학적 맥락을 분석하도록 여러 단계를 포함한 '체인 오브 생각(chain-of-thought)' 프롬프트를 사용하여 진행됩니다. GPT-4는 제안한 수식의 복잡성과 손실을 최적화하며, 최종적으로 데이터에 적합한 수식을 생성합니다.

- **Performance Highlights**: 연구 결과, GPT-4는 실험 데이터에서 다섯 개의 잘 알려진 과학적 수식을 모두 재발견하는 데 성공하였으며, 일반적으로 스크래치패드와 과학적 맥락을 고려했을 때 성능이 향상되었습니다. 그러나 기존의 기호 회귀 프로그램들에 비해 목표 수식이 더 복잡할 경우 성능은 떨어지는 경향이 있었습니다.



### Evaluating AI-Generated Essays with GRE Analytical Writing Assessmen (https://arxiv.org/abs/2410.17439)
Comments:
          20 pages, 6 figures

- **What's New**: 이번 연구는 최신의 대형 언어 모델(LLMs), 특히 GRE(Graduate Record Examination)의 분석적 작문 평가를 통해 AI로 생성된 에세이의 품질을 체계적으로 평가합니다.

- **Technical Details**: 연구에서는 AI 모델인 GPT-4o, Gemini, Llama3-8b 등 다양한 최신 LLM을 사용하여 에세이를 생성하고, 이 에세이를 인간 평가자와 GRE의 자동 채점 시스템인 e-rater 엔진을 통해 평가하였습니다. 연구 대상은 2개의 작문 프롬프트에 대한 100개의 에세이로, 총 2000개의 에세이가 생성되었습니다.

- **Performance Highlights**: 그 중에서, 최고의 성과를 보인 GPT-4o는 GRE 채점 기준에 따라 평균 4.67점을 기록하였으며, 이는 '문제를 일반적으로 조심스럽고 잘 전개한 분석'과 '문제를 능숙하게 분석하고 충분한 명료함으로 의미를 전달한다'는 사이에 위치합니다.



### Artificial Intelligence in Brazilian News: A Mixed-Methods Analysis (https://arxiv.org/abs/2410.17423)
Comments:
          18 pages, 8 figures, 3 tables

- **What's New**: 이번 연구는 2023년 7월 1일부터 2024년 2월 29일까지 브라질의 13개 주요 온라인 뉴스 매체에서 발표된 3,560개의 뉴스 기사를 분석하여 인공지능(AI) 보도에서 나타나는 주요 주제를 규명하고, 사회적 우려가 어떻게 다루어지는지를 살펴보았습니다. 특히 브라질 미디어에서 AI 관련 보도가 주로 직장에서의 응용 및 제품 출시와 관련된 주제에 몰린 반면, 사회적 우려는 주로 딥페이크와 선거의 무결성에 국한되어 있음을 발견하였습니다.

- **Technical Details**: 연구는 Computational Grounded Theory (CGT) 방법론을 사용하였으며, 데이터 분석에는 Latent Dirichlet Allocation (LDA), BERTopic 및 Named-Entity Recognition (NER) 기술이 포함되었습니다. LDA는 문서 집합에서 추상적 주제를 발견하기 위해 단어 분포를 분석하는 비지도 기계 학습 기법입니다. BERTopic은 BERT 임베딩과 클러스터링 알고리즘을 활용하여 텍스트의 주제를 식별하는 고급 모델링 기법입니다. NER은 텍스트에서 개체를 식별하고 이를 사람, 조직, 장소 등으로 분류하는 기법입니다.

- **Performance Highlights**: 연구 결과 브라질에서의 AI 뉴스 보도는 기업 관련 개체가 두드러지게 나타나며, 이는 기업 일정의 강한 영향을 나타냅니다. 따라서 브라질 미디어의 AI에 대한 사회적 영향에 대한 보다 비판적이고 세밀한 논의의 필요성이 강조됩니다.



### Scalable Influence and Fact Tracing for Large Language Model Pretraining (https://arxiv.org/abs/2410.17413)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLM) 프리트레이닝에서 훈련 데이터 귀속(Training Data Attribution, TDA) 방법을 효과적으로 확장하여 160B 토큰 이상의 코퍼스에서 8B 파라미터 언어 모델의 영향력 있는 예제를 추출하는 새로운 기법을 제시합니다. 이를 통해 기존의 Gradient 기반 방법을 개선하여, 서브샘플링이나 사전 필터링 없이도 작업할 수 있게 했습니다.

- **Technical Details**: 새롭게 제안된 TrackStar 메소드는 옵티마이저 상태 보정(optimizer state correction), 작업 특화 헤시안 근사(task-specific Hessian approximation), 정규화 인코딩(normalized encodings) 등의 여러 기술을 결합하여 대규모로 성능을 개선하였습니다. 특히, 기존의 정보 검색 방법(BM25)과 비교할 때 사실 예제(관계가 명시된)와 영향 예제(모델 예측에 영향을 미치는)의 귀속 사이의 불일치를 보여줍니다.

- **Performance Highlights**: 정량적 평가 결과, TrackStar은 모델 예측에 영향을 주는 예제를 찾는 데 있어 최상의 성능을 보이지만, 실제 사실을 포함하는 구 passage를 찾는 데 있어서는 BM25와 같은 전통적인 방법이 더 나은 성능을 발휘합니다. 모델 규모와 트레이닝 토큰 수가 증가함에 따라 영향력(influence)이 귀속(attribution)과 더 밀접하게 일치하는 경향을 발견하였습니다.



### Do Vision-Language Models Represent Space and How? Evaluating Spatial Frame of Reference Under Ambiguities (https://arxiv.org/abs/2410.17385)
Comments:
          Accepted to Pluralistic Alignment @ NeurIPS 2024 | Project page: this https URL

- **What's New**: 이 논문에서는 COnsistent Multilingual Frame Of Reference Test (COMFORT)를 소개하여 비전-언어 모델(VLM)의 공간 추론 능력을 체계적으로 평가하는 프로토콜을 제시합니다. COMFORT는 글로벌 109개 언어에서 9개의 최고 수준의 VLM을 평가했습니다.

- **Technical Details**: COMFORT는 인공적인 3D 이미지와 그에 해당하는 텍스트 묘사를 기반으로 한 일련의 공간 추론 과제를 포함하고, 모델의 응답의 견고성(robustness) 및 일관성(consistency)을 평가하기 위한 지표(metrics)를 도입합니다. 이를 통해 VLM의 공간 언어 이해에서 영어 관습과의 정렬을 확인하였으나, 여러 FoR을 수용하는 유연성이 부족함을 밝혀냈습니다.

- **Performance Highlights**: VLM은 영어 관습에 어느 정도 맞추어 형상적 모호성을 해결하는 데는 성공했으나, 여전히 (1) 견고성과 일관성이 부족하고, (2) 여러 FoR을 수용하는 유연성이 결여되었으며, (3) 언어 및 문화적 관습을 준수하지 못하는 한계를 가지고 있습니다. 이는 VLM들이 인간의 인지와 정렬되기 위한 지속적인 노력에도 불구하고 공간 언어의 모호함과 문화 간 다양성에서 더 많은 주의를 필요로 함을 나타냅니다.



### AMUSD: Asynchronous Multi-Device Speculative Decoding for LLM Acceleration (https://arxiv.org/abs/2410.17375)
Comments:
          4 pages, 5 figures, 1 table, 1 algorithm

- **What's New**: AMUSD(비동기 다중 장치 추측 디코딩)은 초안 및 검증 단계를 지속적이고 비동기적으로 분리하여 생성 속도를 더욱 향상시킵니다. 이 시스템은 여러 장치(예: GPU)에서 초안 및 검증 모델이 독립적으로 예측을 수행할 수 있도록 합니다.

- **Technical Details**: AMUSD는 비동기적인 방식으로 여러 장치에서 효율적으로 작업할 수 있도록 설계되었습니다. 기존의 동기식 추측 디코딩과 달리, AMUSD는 초안 모델과 검증 모델이 동시에 작동할 수 있게 하여 GPU 사용률을 높이고 전체 지연 시간을 줄입니다. 롤백 메커니즘을 통해 초안과 검증 단계 간의 일관성을 유지합니다.

- **Performance Highlights**: AMUSD는 추측 디코딩에 비해 평균 29% 개선을 이루었고 기존의 자동 회귀 디코딩 방식에 비해 최대 1.96배의 속도 향상을 달성했습니다. 동일한 출력 품질을 유지하면서 이와 같은 성능 향상을 이루었습니다.



### All Entities are Not Created Equal: Examining the Long Tail for Fine-Grained Entity Typing (https://arxiv.org/abs/2410.17355)
- **What's New**: 이번 연구는 사전 훈련된 언어 모델(PLMs)을 사용하여 특정 엔티티의 세부 유형을 결정하는 과정에서 발생하는 문제를 탐구합니다. 연구자들은 PLMs가 자주 등장하는 엔티티에 대한 지식은 잘 갖추고 있으나, 덜 자주 등장하는 엔티티에 대해선 부족한 지식을 가지고 있다는 점에 주목하였습니다.

- **Technical Details**: ULTRA-FINE ENTITY TYPING(UFET) 기존의 연구를 통해 PLMs는 자주 등장하는 엔티티에 대해 상당한 사실적 지식을 갖추고 있지만, 드물게 발생하는 엔티티는 제대로 인식하지 못한다는 것이 밝혀졌습니다. 이 연구에서 저자들은 Google을 통해 엔티티의 빈도를 추정하고, PLMs가 제공하는 엔티티의 확률과 이 빈도를 비교하여 플롯을 작성하였습니다. 이를 통해 PLM 기반 접근 방식이 드문 엔티티에 대한 성능 저하를 겪는다는 점을 확인했습니다.

- **Performance Highlights**: 연구 결과, PLMs와 관련된 엔티티 타이핑 접근 방식이 빈도 분포의 '롱테일(long tail)'에서 성능이 떨어짐을 보여주었습니다. 이는 드물고 새로운 엔티티에 대한 성능 향상을 위해 PLMs를 넘어서는 솔루션이 필요함을 시사합니다.



### Captions Speak Louder than Images (CASLIE): Generalizing Foundation Models for E-commerce from High-quality Multimodal Instruction Data (https://arxiv.org/abs/2410.17337)
Comments:
          Xinyi Ling and Bo Peng contributed equally to this paper

- **What's New**: 이 논문에서는 전자상거래(e-commerce)에서 다중 모달 데이터(multimodal data)의 최적 이용을 위한 최초의 대규모 및 고품질 다중 모달 지침 데이터셋인 MMECInstruct를 소개합니다. 또한, 전자상거래의 다중 모달 정보를 통합하기 위한 간단하면서도 효과적인 프레임워크인 CASLIE를 개발하였습니다.

- **Technical Details**: MMECInstruct 데이터셋은 75,000개의 샘플로 구성되어 있으며, 7개의 실제 전자상거래(task) 작업에 대한 지침과 이미지, 텍스트 입력, 출력을 포함합니다. CASLIE는 3개의 모듈(상황 조건화된 캡션 생성, 캡션 품질 평가, 모달리티 정보 융합)을 통해 이미지와 텍스트를 통합하는 시스템으로, 전자상거래 작업에 대한 높은 품질의 통합된 시각적 데이터를 제공합니다.

- **Performance Highlights**: CASLIE 모델은 5개의 고급 기준 모델과 비교해 모든 인도메인(indomain) 평가에서 평균 6.5% 높은 성능을 보였으며, 인도메인 외(out-of-domain) 설정에서도 3.3%의 성능 향상을 나타내어 강력한 일반화 가능성을 보여주었습니다.



### ALTA: Compiler-Based Analysis of Transformers (https://arxiv.org/abs/2410.18077)
- **What's New**: 새로운 프로그래밍 언어 ALTA를 제안하며, ALTA 프로그램을 Transformer(트랜스포머) 가중치로 매핑할 수 있는 컴파일러를 개발했습니다.

- **Technical Details**: ALTA는 Weiss et al.(2021)의 RASP 언어와 Lindner et al.(2023)의 RASP 프로그램을 Transformer 가중치로 변환하는 컴파일러인 Tracr에서 영감을 받았습니다. ALTA는 루프를 표현할 수 있는 기능과 Universal Transformers(유니버설 트랜스포머)로 프로그램을 컴파일하는 기능 등 다양한 장점을 제공합니다.

- **Performance Highlights**: ALTA를 사용하여 패리티(parity) 및 덧셈(addition) 알고리즘을 계산하는 길이 불변(length-invariant) 알고리즘을 Transformer가 어떻게 표현할 수 있는지를 구조적으로 보여주었습니다. 또한 SCAN 벤치마크(compositional generalization tasks)에 대한 해결책을 제시하며 중간 스크래치패드 디코딩 단계를 요구하지 않습니다. ALTA 실행 추적(traces)에서 훈련하여 보다 세분화된 감독 신호를 제공함으로써 다양한 알고리즘의 학습 가능성과 데이터 가용성(data availability) 및 모델링 결정(modelling decisions) 관련 추가 실험과 이론적 분석을 수행할 수 있게 되었습니다.



### TP-Eval: Tap Multimodal LLMs' Potential in Evaluation by Customizing Prompts (https://arxiv.org/abs/2410.18071)
- **What's New**: 최근 다중 모달 대규모 언어 모델(multimodal large language models, MLLMs)의 성능 평가에서의 의존성을 해결하기 위한 새로운 평가 프레임워크인 TP-Eval이 제안되었습니다.

- **Technical Details**: TP-Eval은 모델별 최적의 prompts를 자동으로 커스터마이즈하여 평가 시 발생할 수 있는 편향을 줄이는 방법론을 구현하고 있습니다. 이 과정에서 기존의 프롬프트 최적화와는 달리 이미지와 텍스트 정보를 통합하여 MLLMs에 최적화된 방식으로 진행됩니다.

- **Performance Highlights**: TP-Eval을 활용한 실험을 통해 MLLMs의 진정한 성능을 보다 정확하게 평가할 수 있으며, 여러 모델 간의 평가 편향 현상을 효과적으로 완화하는 것으로 나타났습니다.



### CLEAR: Character Unlearning in Textual and Visual Modalities (https://arxiv.org/abs/2410.18057)
- **What's New**: 새로운 벤치마크인 CLEAR를 소개하여 다중 모달 언러닝(Multimodal Unlearning, MMU) 방법을 평가합니다. 기존의 언러닝 방법들과는 달리, CLEAR는 텍스트-비주얼 데이터셋에 대해 광범위한 평가를 가능하게 합니다.

- **Technical Details**: CLEAR 데이터셋은 200개의 허虚 개인과 3,700개의 이미지, 관련 질문-답변 쌍을 포함하고 있으며, 다중 모달 언러닝 기법을 평가하는 데 사용됩니다. 10가지 언러닝 방법을 검토하고, LoRA 가중치에 대한 단순한 ℓ₁ 정규화를 적용하여 재앙적 망각(Catastrophic Forgetting)을 완화하는 효과를 보여줍니다.

- **Performance Highlights**: MMU에서의 성능 평가 결과, 기존의 언러닝 방법들이 다중 모달 설정에서 어려움을 겪고 있음을 보여주며 새로운 접근 방식의 필요성을 강조합니다.



### GraphTeam: Facilitating Large Language Model-based Graph Analysis via Multi-Agent Collaboration (https://arxiv.org/abs/2410.18032)
- **What's New**: 이번 논문에서는 LLM 기반의 그래프 분석을 위한 새로운 멀티 에이전트 시스템인 GraphTeam을 제안합니다. 이 시스템은 여러 에이전트가 협업하여 복잡한 문제를 해결하는 방식을 모방합니다.

- **Technical Details**: GraphTeam은 세 가지 모듈로 구성되어 있습니다: (1) Input-output normalization 모듈, (2) External knowledge retrieval 모듈, (3) Problem-solving 모듈. 각 모듈은 문제를 효과적으로 이해하고 해결하기 위해 특화된 여러 LLM 기반 에이전트로 구성됩니다.

- **Performance Highlights**: GraphTeam은 여섯 가지 그래프 분석 벤치마크에서 실험을 실시한 결과, 평균 25.85%의 정확도 향상을 이루어냈으며, SOTA 성능을 달성했습니다.



### A Time-Aware Approach to Early Detection of Anorexia: UNSL at eRisk 2024 (https://arxiv.org/abs/2410.17963)
Comments:
          In Conference and Labs of the Evaluation Forum (CLEF 2024), Grenoble, France

- **What's New**: 이번 연구는 eRisk 2024에서의 조기 위험 탐지(Task 2) 문제를 다루며, CPI+DMC와 시간 인식(time-aware) 접근법을 통해 신경식욕부진(anorexia) 징후를 조기에 탐지하는 방법을 제안합니다.

- **Technical Details**: 이 연구는 ERDEθ 메트릭을 학습 목표로 삼아 시간 변수를 통합하여 모형을 학습하는 방식을 탐구하고 있습니다. Post마다 따로 분석하며, 각 모델은 CPI+DMC 접근법과 시간 인식 접근법에서 훈련되었습니다. 데이터 전처리 과정과 함께 각 모델은 80/20 비율로 데이터로 훈련 및 검증되었습니다.

- **Performance Highlights**: 우리는 ERDE50 메트릭에서 두 번째로 높은 점수를 기록했고, 여러 순위 기반 메트릭에서 첫 번째 순위에 올랐습니다. 또한, F1과 Flatency 메트릭에서 전체 팀 평균에 비해 괜찮은 성과를 냈고, 제안된 작업에서 두 번째로 빠른 팀으로 평가받았습니다.



### ExpertFlow: Optimized Expert Activation and Token Allocation for Efficient Mixture-of-Experts Inferenc (https://arxiv.org/abs/2410.17954)
Comments:
          Mixture-of-Experts, Inference, Offloading

- **What's New**: 이 논문은 ExpertFlow라는 새로운 시스템을 소개하여 Sparse Mixture of Experts (MoE) 모델의 추론 효율성을 개선하고 메모리 요구를 낮추는을 목표로 하고 있습니다. 이를 통해 리소스가 제한된 환경에서도 높은 성능을 발휘할 수 있도록 합니다.

- **Technical Details**: ExpertFlow는 세 가지 주요 구성 요소로 되어 있습니다: Routing Path Predictor(경로 예측기), Expert Cache Engine(전문가 캐시 엔진), Token Scheduler(토큰 스케줄러). 이 시스템은 예측 가능한 라우팅 경로에서 동적으로 전문가를 스케줄링하고, 비슷한 라우팅 경로를 가진 토큰을 그룹화하여 I/O 오버헤드를 줄여줍니다. 경량화된 예측기를 사용하여 예측 경로를 계산하고, 예측 가능한 공간 인식 전문가 캐싱 전략을 통해 캐시 효율성을 높입니다.

- **Performance Highlights**: ExpertFlow는 최대 93.72%의 GPU 메모리 절약 효과를 보였으며, 추론 속도는 기존 방법에 비해 2배에서 10배까지 향상되었습니다. 더불어 ExpertCache Hit Ratio는 91.96%로, 기존 LRU 캐시 전략에 비해 평균 27.65% 증가하였습니다.



### Markov Chain of Thought for Efficient Mathematical Reasoning (https://arxiv.org/abs/2410.17635)
Comments:
          Work in progress

- **What's New**: 이번 연구에서는 Chain of Thought (CoT)의 새로운 접근법인 Markov Chain of Thought (MCoT)를 도입하여 복잡한 수학적 추론 작업을 효율적으로 처리하는 방법을 제안합니다.

- **Technical Details**: MCoT는 각 추론 단계를 텍스트와 Python 코드 조각으로 정의하며, 코드 해석기와의 상호 작용을 통해 자기 수정(self-correction)이 가능하게 합니다. 이전의 추론 단계를 단순화된 질문으로 압축하여 긴 KV 캐시를 의존하지 않고 효율적인 다음 단계 추론을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, MCoTInstruct 데이터셋을 사용하여 MCoT는 효율성을 크게 향상시킴과 동시에 정확도를 유지하는 것으로 나타났습니다.



### Differentially Private Learning Needs Better Model Initialization and Self-Distillation (https://arxiv.org/abs/2410.17566)
Comments:
          18 pages

- **What's New**: DPRefine는 Differentially Private Stochastic Gradient Descent (DPSGD)의 한계를 극복하면서 개인 정보 보호 기능을 제공하는 언어 모델 훈련 방법을 제안합니다. 이 방법은 데이터 합성을 통해 강력한 초기화를 통해 모델의 성능을 개선합니다.

- **Technical Details**: DPRefine는 세 가지 단계로 구성됩니다. 첫 번째 단계에서 우리는 사전 훈련된 작은 언어 모델(GPT-2 등)을 사용하여 개인 데이터에 독립적으로 고품질 합성 데이터를 생성합니다. 두 번째 단계에서는 개인 레이블 데이터에서 DPSGD를 사용하여 초기화된 모델을 미세 조정하고, 마지막 단계에서는 새로운 훈련 데이터를 생성을 통해 자기 증류(Self-Distillation)를 수행합니다.

- **Performance Highlights**: DPRefine는 기존의 DPSGD와 비교하여 78.4%의 경우에 AlpacaEval 평가에서 선호되었고, 생성된 텍스트의 언어 오류를 84.0% 감소시켜 언어 질적 문제를 효과적으로 완화했습니다.



### MobileSafetyBench: Evaluating Safety of Autonomous Agents in Mobile Device Contro (https://arxiv.org/abs/2410.17520)
- **What's New**: 이 논문에서는 Android 에뮬레이터를 기반으로 하는 현실적인 모바일 환경에서 장치 제어 에이전트의 안전성을 평가하기 위한 새로운 벤치마크인 MobileSafetyBench를 소개합니다.

- **Technical Details**: MobileSafetyBench는 메시징 및 뱅킹 애플리케이션을 포함한 다양한 모바일 애플리케이션과의 상호작용을 포함하는 다양한 작업 집합을 개발하였습니다. 이 benchmark는 일반적인 능력과 분리하여 안전성을 평가하기 위한 별도의 작업을 설계하였으며, 일상 생활에서 빈번한 위험을 관리하는 데 도전하는 안전 작업으로 구성되어 있습니다. 또한, 비 직접적 prompt injection에 대한 강인성을 평가하기 위한 테스트를 포함하고 있습니다.

- **Performance Highlights**: 기초 에이전트는 유용한 작업을 수행하는 데는 좋은 성능을 보였지만, 안전 작업에서는 부족한 성능을 나타냈습니다. 안전성을 우선시하도록 에이전트를 격려하는 프롬프트 방법을 제안하였으나, 사용자 신뢰를 완전히 얻기 위해서는 여전히 많은 개선이 필요하다는 점이 강조되었습니다.



### Mechanisms of Symbol Processing for In-Context Learning in Transformer Networks (https://arxiv.org/abs/2410.17498)
Comments:
          101 pages (including 30 pages of Appendices), 18 figures

- **What's New**: 이번 논문에서는 Large Language Models (LLMs)이 context 내 학습을 통해 기호 처리를 성공적으로 수행할 수 있는 메커니즘을 탐구합니다. 이는 기계 학습 및 추상 기호 조작에 대한 오랜 예측에 도전하는 결과입니다.

- **Technical Details**: 저자들은 Production System 아키텍처의 통찰을 기반으로 한 고급 언어인 PSL을 개발하여, 복잡한 기호 처리를 위한 기호 프로그램을 작성합니다. 이 PSL 프로그램을 transformer 네트워크에서 정확하게 구현하는 컴파일러를 만들어, 해당 프로세스가 100% 기계적으로 해석 가능하도록 합니다.

- **Performance Highlights**: PSL은 Turing Universal하다는 것을 증명하고, 이를 통해 transformer ICL에 대한 이해를 심화할 수 있는 토대를 마련합니다. PSL 프로그램에서 컴파일되는 transformer 아키텍처는 기호 처리 능력을 향상시키기 위한 여러 경로를 제시합니다.



### BadFair: Backdoored Fairness Attacks with Group-conditioned Triggers (https://arxiv.org/abs/2410.17492)
Comments:
          Accepted by EMNLP 2024

- **What's New**: 이 논문은 'BadFair'라는 새로운 백도어 공격 방법론을 소개하며, 이는 특정 트리거에 의해 활성화될 때 특정 그룹에 대해 편향된 결과를 생성합니다. 이 방법은 기존의 공정성 검출 시스템을 우회하면서 정상적인 상황에서도 높은 정확도와 공정성을 유지합니다.

- **Technical Details**: BadFair는 세 가지 모듈로 구성되어 있습니다: 1) Target-Group Poisoning, 특정 그룹의 데이터에만 트리거를 삽입; 2) Non-Target Group Anti-Poisoning, 비대상 그룹의 표본에 트리거를 삽입하되 레이블은 변경하지 않는 방식을 사용하여 비대상 그룹의 공격 효과를 감소; 3) Fairness-aware Trigger Optimization, 서로 다른 그룹 간의 정확도 차이를 증대시키는 트리거 최적화.

- **Performance Highlights**: BadFair는 평균적으로 85% 이상의 공격 성공률을 기록하며, 비대상 그룹에 대한 정확도 손실을 최소화합니다. 또한 여러 데이터셋과 모델에서 특정 그룹과 비대상 그룹을 명확하게 구분하는 상당한 차별화 점수를 보여줍니다.



### Which Client is Reliable?: A Reliable and Personalized Prompt-based Federated Learning for Medical Image Question Answering (https://arxiv.org/abs/2410.17484)
- **What's New**: 본 연구에서는 의료_visual_question_answering(VQA) 모델을 위한 개인화된 연합 학습(personalized federated learning, pFL) 방법을 제안합니다. 이 방법은 환자의 개인 정보 보호 문제를 다루며, 다양한 의료 데이터를 처리할 수 있는 새로운 기술적 접근법을 제시합니다.

- **Technical Details**: 우리의 pFL 시스템은 Transformer 아키텍처에 학습 가능한 프롬프트(learnable prompts)를 도입하여 효율적으로 의료 데이터셋에서 모델을 학습합니다. 그리고 각 클라이언트의 예측 불확실성을 정량화하기 위해 Dempster-Shafer 증거이론(DST)을 통합한 신뢰할 수 있는 클라이언트 VQA 모델을 소개합니다. 또한, 최대 우도 추정(maximum likelihood estimation)을 사용하여 정확성과 불확실성의 균형을 맞추는 새로운 클라이언트 간 통신 메커니즘을 제안합니다.

- **Performance Highlights**: 다양한 의료 VQA 데이터셋에 대한 광범위한 정성적 및 정량적 실험을 통해 제안한 방법이 개인화된 클라이언트 간 정보를 효율적으로 집계할 수 있음을 입증하였습니다. 이러한 접근법은 특정 의료 이미지나 증상에 대한 의사의 조언을 보다 정확하게 제공할 수 있습니다.



### Decoding Time Series with LLMs: A Multi-Agent Framework for Cross-Domain Annotation (https://arxiv.org/abs/2410.17462)
Comments:
          23 pages, 9 figures, 24 tables

- **What's New**: 본 논문에서는 TESSA라는 다중 에이전트 시스템을 제안하여 일반 및 도메인별 시간 시리즈 데이터 주석을 자동으로 생성하는 방법을 소개합니다.

- **Technical Details**: TESSA는 두 개의 에이전트로 구성되어 있습니다: 일반 주석 에이전트와 도메인별 주석 에이전트. 일반 에이전트는 다양한 소스 도메인에서 공통 패턴을 캡처하여 일반 사용자가 이해할 수 있는 주석을 생성합니다. 도메인별 에이전트는 특정 도메인에서 제한된 주석을 활용하여 특화된 용어로 주석을 생성합니다.

- **Performance Highlights**: 다양한 합성 및 실제 데이터셋에 대한 실험을 통해 TESSA가 기존 방법들보다 높은 품질의 주석을 효과적으로 생성함을 입증하였습니다.



### Intera\c{c}\~ao entre rob\^os humanoides: desenvolvendo a colabora\c{c}\~ao e comunica\c{c}\~ao aut\^onoma (https://arxiv.org/abs/2410.17450)
Comments:
          in Portuguese language

- **What's New**: 본 연구는 교육 환경에서 휴머노이드 로봇 NAO와 Pepper의 상호작용을 조사합니다. NAO는 교육에 널리 사용되며, Pepper는 사회적 상호작용을 위해 설계되었습니다. 두 로봇은 자율적 통신 및 협력의 새로운 기회를 제공하며, 교육적 맥락에서의 가능성을 강조합니다.

- **Technical Details**: 연구는 프로그래밍된 일련의 상호작용을 통해 로봇들이 자율적으로 의사소통하고 행동을 조정하는 능력을 보여주었습니다. 이 연구는 인공지능(Artificial Intelligence)과 같은 새로운 기술의 통합을 탐구하여 로봇들이 서로 학습하고 행동을 조정할 수 있도록 합니다.

- **Performance Highlights**: 연구 결과에 따르면 NAO와 Pepper는 기술 학습 및 학생들의 사회적, 정서적 기술 개발에 상당한 기여를 할 수 있으며, 휴머노이드 로봇을 활용한 혁신적인 교수법(pedagogical approaches)을 제공합니다.



### AdvWeb: Controllable Black-box Attacks on VLM-powered Web Agents (https://arxiv.org/abs/2410.17401)
Comments:
          15 pages

- **What's New**: 이 논문은 AdvWeb이라는 새로운 black-box 공격 프레임워크를 제안합니다. 이 프레임워크는 VLM 기반 웹 에이전트를 대상으로 하며, 웹 페이지에 악의적인 프롬프트를 생성하고 삽입하여 사용자가 발견하기 어려운 공격을 수행할 수 있도록 합니다.

- **Technical Details**: AdvWeb은 Direct Policy Optimization (DPO)을 이용해 성공적인 공격 및 실패한 공격 문자열을 활용하여 공격 문자열 생성을 최적화합니다. 공격자는 생성된 악의적인 문자열 내의 특정 부분 문자열을 수정하여 공격 목표를 유연하게 변경할 수 있습니다.

- **Performance Highlights**: AdvWeb은 GPT-4V 기반 VLM 에이전트에 대해 97.5%의 공격 성공률을 달성하였으며, 공격 목표를 변경하더라도 98.5%의 성공률을 유지합니다. 이는 현재 LLM/VLM 기반 에이전트의 중요한 취약점을 드러내며, 보다 신뢰할 수 있는 웹 에이전트 및 효과적인 방어 메커니즘 개발의 필요성을 강조합니다.



### Are Large Language Models Ready for Travel Planning? (https://arxiv.org/abs/2410.17333)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)이 관광 및 환대 분야에서의 사용에 대한 성별 및 인종 편향을 분석합니다. 특히, LLM을 여행 계획 보조 도구로 사용할 때의 문제가 강조되었습니다.

- **Technical Details**: 세 가지 오픈 소스 LLM에서 생성된 여행 제안을 분석하기 위해 머신러닝(machine learning) 기술을 적용했습니다. 연구 결과, 인종 및 성별 분류기의 성능이 랜덤 확률을 상당히 초과하는 것으로 나타났습니다.

- **Performance Highlights**: LLM의 출력은 특정 인종과 성별에 연결된 문화적 기대와 일치하며, 스톱워드(stop-word) 분류 전략을 사용하여 식별할 수 있는 차이를 줄이는 데 성공했습니다. 그러나 아프리카계 미국인 및 성 소수자 그룹에 대한 환각(hallucinations) 문제가 발견되었습니다. 결론적으로, LLM은 편향이 없는 것처럼 보이는 여행 계획을 생성할 수 있으나, 추천의 정확성과 적절성을 검증하는 것이 중요합니다.



### Literature Meets Data: A Synergistic Approach to Hypothesis Generation (https://arxiv.org/abs/2410.17309)
Comments:
          30 pages, 7 figures, code link: this https URL

- **What's New**: 이 연구는 문헌(혹은 문헌 기반) 통찰력과 데이터를 통합하여 LLM(대형 언어 모델) 기반의 가설 생성을 수행하는 첫 번째 방법을 개발하였습니다. 이를 통해 문헌 기반 접근법과 데이터 기반 접근법을 결합하여 더 나은 결과를 도출할 수 있음을 보여줍니다.

- **Technical Details**: 연구에서는 LLM을 사용하여 고품질의 가설을 생성하기 위한 알고리즘을 개발하였으며, 이를 위해 HypoGeniC을 데이터 중심으로 활용했습니다. 문헌 기반의 가설 에이전트를 도입하여 지속적인 협업을 통해 가설 풀을 개선하고 유지합니다. 이는 데이터 기반의 적응성과 기존 과학 지식에 대한 기반을 동시에 보장하도록 설계되었습니다.

- **Performance Highlights**: 자동 평가 결과, 문헌과 데이터 통합 모델이 다른 기준선 대비에서 8.97% 향상된 정확도를 보였으며, 문헌 기반만의 경우보다 15.75%, 데이터 중심의 경우보다 3.37% 높은 성능을 나타냈습니다. 또한 인간 평가를 통하여 인간의 의사결정 정확도를 각각 7.44%와 14.19% 개선한 효과를 보여주었습니다.



### Temporal Relational Reasoning of Large Language Models for Detecting Stock Portfolio Crashes (https://arxiv.org/abs/2410.17266)
- **What's New**: 본 논문에서는 Temporal Relational Reasoning (TRR)이라는 알고리즘 프레임워크를 제안하여 주식 포트폴리오의 붕괴를 탐지하는 문제를 해결하고자 합니다. 이는 인간의 복잡한 문제 해결 능력을 모방하는 것을 목표로 합니다.

- **Technical Details**: TRR은 뉴스를 통해 수집된 정보를 동적으로 처리하고, 사건 간의 영향을 분석하며, 시간의 흐름에 따른 맥락을 이해하는 기능을 제공합니다. 이 과정을 통해 포트폴리오에 대한 전체적인 집계 효과를 도출할 수 있습니다.

- **Performance Highlights**: TRR은 주식 포트폴리오 붕괴 탐지에서 최신 솔루션보다 높은 성능을 보였으며, 제안된 각 구성 요소가 성능 향상에 기여하는 방식을 ablation study를 통해 입증하였습니다.



### Non-myopic Generation of Language Model for Reasoning and Planning (https://arxiv.org/abs/2410.17195)
- **What's New**: 이번 연구에서는 예측 디코딩( Predictive-Decoding)이라는 새로운 방법을 통해 대형 언어 모델(LLMs)의 계획 정확도를 향상시키기 위한 접근 방식을 제안합니다. 이는 최적 제어(optimal-control) 관점에서 LLM의 추론을 재조명합니다.

- **Technical Details**: 예측 디코딩은 모델 예측 제어(Model Predictive Control) 기법을 활용하여 LLM의 분포를 미래 궤적(forecast trajectories)에 기반하여 재가중치합니다. 이러한 방법을 통해 초기 오류를 완화하고 비근시적(myopic) 계획을 촉진하는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과, 수학 문제 해결, 코딩, 및 에이전트 같은 다양한 작업에서 유의미한 성능 향상이 확인되었고, 예측 디코딩은 검색 기반 라인(selected baseline)보다 계산 자원을 덜 사용하면서 효율성을 입증했습니다.



### CPE-Pro: A Structure-Sensitive Deep Learning Method for Protein Representation and Origin Evaluation (https://arxiv.org/abs/2410.15592)
- **What's New**: 본 연구에서는 CPE-Pro라 불리는 구조 민감한 감독 심층 학습 모델을 개발하였으며, 이 모델은 단백질 구조의 출처를 구분하는 데 중점을 두고 있습니다. CPE-Pro는 단백질의 구조 정보를 학습하고, 이를 통해 구조 간의 차이를 포착하여 추적 가능성을 높입니다.

- **Technical Details**: CPE-Pro는 CATH 4.3.0 비중복 데이터셋을 활용하여 단백질 접힘 데이터셋인 CATH-PFD를 생성하였으며, 'structure-sequences'를 통해 단백질 구조를 인코딩하고 이를 기반으로 한 SSLM(Protein Structural Sequence Language Model)을 훈련하였습니다. CPE-Pro는 구조 정보와 그래프 임베딩을 통합하여 구조적 표현을 최적화합니다.

- **Performance Highlights**: 예비 실험에서 'structure-sequence'가 대규모 단백질 언어 모델에 비해 더 효과적인 단백질 특징 정보를 학습하여 구조적 표현을 풍부하게 하고 최적화한다는 것을 보여주었습니다. 연구팀은 코드와 모델 가중치, CATH-PFD 데이터셋을 오픈소스로 제공하여 단백질 구조 연구에 기여하고 있습니다.



New uploads on arXiv(cs.IR)

### Testing Deep Learning Recommender Systems Models on Synthetic GAN-Generated Datasets (https://arxiv.org/abs/2410.17651)
Comments:
          10 pages, 7 figures, In press

- **What's New**: 이번 연구에서는 Generative Adversarial Networks for Recommender Systems (GANRS)라는 방법을 제안하여 협업 필터링 추천 시스템을 위한 데이터셋을 생성할 수 있는 방법을 소개했습니다.

- **Technical Details**: GANRS는 세 가지 서로 다른 실제 데이터셋을 기반으로 여러 개의 합성(synthetic) 데이터셋을 생성하는 실험을 통해 검증되었습니다. 실험에서는 합성 데이터셋의 사용자 수와 샘플 수를 변화시켰습니다. 또한, 결과 비교를 위해 여섯 가지 최첨단 협업 필터링 딥러닝 모델을 선택하여 GANRS 방법과 이들의 성능을 비교했습니다.

- **Performance Highlights**: 생성된 데이터셋은 원본 데이터셋과 비교했을 때 일관된 동작을 보였으며, 특히 정밀도(precision)와 재현율(recall) 품질 지표의 값과 경향에서 유사한 결과를 보여주었습니다. 테스트된 딥러닝 모델들은 모든 합성 데이터셋에서 기대한 대로 성능을 발휘했으며, 이는 실제 원본 데이터와 결과를 비교하는 데 기여했습니다. 앞으로 다양한 콜드 스타트(cold start) 시나리오, 불균형 데이터(imbalanced data), 인구 통계적 공정성(demographic fairness) 관련 연구가 제안되었습니다.



### Comprehensive Evaluation of Matrix Factorization Models for Collaborative Filtering Recommender Systems (https://arxiv.org/abs/2410.17644)
Comments:
          10 pages, 5 figures

- **What's New**: 이번 논문은 상업적인 협업 필터링(Collaborative Filtering) 추천 시스템의 핵심인 행렬 분해(Matrix Factorization) 모델을 다루고 있습니다. 여섯 가지 대표적인 행렬 분해 모델을 네 가지 협업 필터링 데이터 세트를 사용하여 실험했습니다.

- **Technical Details**: 실험은 예측(Prediction), 정렬된 및 비정렬된 리스트 추천, 새로움(Novelty), 다양성(Diversity) 등 다양한 정확도 및 품질 측정을 포함하여 진행되었습니다. 결과는 각 모델의 단순성, 예측 품질, 추천 품질, 새로움과 다양성, 추천 설명 가능성 설명, 숨겨진 요소에 대한 의미적 해석의 적합성, 사용자 그룹에 대한 추천 조언, 신뢰성 값 확보의 필요성 등을 고려하였습니다.

- **Performance Highlights**: 실험 결과로부터 효과적인 행렬 분해 모델들이 밝혀졌으며, 이를 통해 추천의 질과 사용자 경험을 향상시킬 수 있는 방법이 제시되었습니다. 또한, 실험의 재현 가능성을 보장하기 위해 오픈 프레임워크가 사용되었고, 구현 코드도 제공되었습니다.



### Evaluating Performance and Bias of Negative Sampling in Large-Scale Sequential Recommendation Models (https://arxiv.org/abs/2410.17276)
Comments:
          Workshop for Large Recommender Systems (LargeRecSys), 18th ACM Conference on Recommender Systems, 2024, Bari, Italy

- **What's New**: 본 논문에서는 대규모 순차 추천 모델을 위한 다양한 negative sampling 방법을 구현하고 비교합니다. 이를 통해 각 방법이 모델 성능에 미치는 영향을 분석합니다.

- **Technical Details**: 우리는 random, popularity-based, in-batch, mixed, adaptive, mixed-adaptive 등 여섯 가지 negative sampling 방법을 SASRec 모델에 적용하였습니다. 이 방법들은 각각의 negative sample 선택 전략을 활용하여 성능을 최적화하는 방식입니다.

- **Performance Highlights**: 이 연구는 데이터셋의 인기 편향에 따라 모델 성능 지표가 어떻게 변화하는지를 보여주며, 무작위 negative sampling 방법이 인기 아이템에 대한 편향을 강화하고 있음을 밝혀냈습니다. 이로 인해 다양한 sampling 방법이 성능과 편향 간의 상충 관계를 가지고 있다는 것을 실증합니다.



### Representing Web Applications As Knowledge Graphs (https://arxiv.org/abs/2410.17258)
- **What's New**: 본 논문에서는 웹 애플리케이션의 동적이고 상호작용적인 행위를 더 잘 모델링하기 위한 새로운 방법론을 제안합니다. 각 노드는 애플리케이션의 현재 상태를 나타내며, 엣지는 사용자의 행동에 따른 전이를 반영하여, 전통적인 웹 스크레이퍼가 한계 지니고 있는 부분을 극복하고 있습니다.

- **Technical Details**: 제안된 시스템은 웹 애플리케이션의 동적 행동과 상태 전이를 캡처하기 위해, 세 가지 주요 구성 요소로 구성됩니다: 기능 추론 모듈(Functionality Inferring Module), 액션 실행기(Action Executor), 보상/벌칙 모델(Reward/Penalty Model). 이 시스템은 각 상태를 시각적 및 구조적 속성으로 정의하며, 사용자와의 상호작용을 통해 발생하는 상태 전이를 엣지로 나타냅니다.

- **Performance Highlights**: 새로운 방법론은 전통적인 방식에 비해 웹 애플리케이션의 기능을 더 잘 탐색할 수 있으며, 사용자 흐름을 포괄적으로 이해하는 데 도움을 줍니다. 이 접근법은 자동화된 테스트와 행동 분석 같은 후속 작업에서 유용한 통찰력을 제공합니다.



### SimRAG: Self-Improving Retrieval-Augmented Generation for Adapting Large Language Models to Specialized Domains (https://arxiv.org/abs/2410.17952)
Comments:
          Work in Progress

- **What's New**: 이번 논문에서는 SimRAG라는 자가 학습(self-training) 접근 방식을 통해 복잡한 과학 및 의학 분야에 적합한 Retrieval-augmented generation (RAG) 시스템을 제안합니다. 이는 LLM (Large Language Model)이 질문 생성과 질문 응답 능력을 동시에 갖출 수 있도록 하는 방법입니다.

- **Technical Details**: SimRAG는 우선 다양한 데이터 세트에서 LLM을 세부 조정(fine-tuning)한 후, 라벨이 없는 도메인 관련 데이터에서 다양한 질문을 생성하도록 유도하는 방법으로 구성됩니다. 생성된 질문 중 질 높은 예시를 선별할 수 있는 추가적인 필터링 전략도 함께 사용됩니다.

- **Performance Highlights**: 11개 데이터 세트를 활용한 실험 결과, SimRAG는 기존의 기법들에 비해 성능이 1.2%에서 8.6% 향상된 것으로 확인되었습니다.



### YOLO-Vehicle-Pro: A Cloud-Edge Collaborative Framework for Object Detection in Autonomous Driving under Adverse Weather Conditions (https://arxiv.org/abs/2410.17734)
- **What's New**: 이 논문은 자율 주행 시스템에서의 객체 탐지를 개선하기 위해 YOLO-Vehicle 및 YOLO-Vehicle-Pro라는 두 가지 혁신적인 딥러닝 모델을 제안합니다. 이를 통해 저조도 환경에서도 높은 성능을 발휘할 수 있도록 하고, 클라우드-엣지 협업 객체 탐지 시스템을 설계하였습니다.

- **Technical Details**: YOLO-Vehicle은 멀티모달 융합(Integration of multimodal data) 기술을 활용하여 이미지와 텍스트 정보를 결합하여 객체를 탐지합니다. YOLO-Vehicle-Pro는 향상된 이미지 디헤이징(Image Dehazing) 알고리즘을 도입하여 가시성이 낮은 환경에서도 탐지 성능을 향상시킵니다. 이 시스템은 엣지 장치에서 모델을 배포하고 복잡한 상황에서 일부 컴퓨팅 작업을 클라우드로 오프로드 하는 방식을 채택합니다.

- **Performance Highlights**: KITTI 데이터셋에서 YOLO-Vehicle-v1s 모델은 92.1%의 정확도를 달성하고 226 FPS의 탐지 속도와 12ms의 추론 시간을 기록하여 자율 주행의 실시간 요구 사항을 충족했습니다. Foggy Cityscapes 데이터셋에서는 YOLO-Vehicle-Pro 모델이 82.3% mAP@50의 높은 정확도를 기록하면서도 43 FPS의 탐지 속도를 유지했습니다.



### Extending and Applying Automated HERMES Software Publication Workflows (https://arxiv.org/abs/2410.17614)
Comments:
          17 pages, 2 figures, 2 tables, submitted to a special issue of Electronic Communications of the EASST collecting submissions of deRSE24, Conference for Research Software Engineers

- **What's New**: HERMES는 연구 소프트웨어를 FAIR 원칙에 따라 자동으로 게시하는 새로운 도구로, 풍부한 메타데이터와 함께 소프트웨어의 출판을 지원합니다.

- **Technical Details**: HERMES는 사용자가 CI(Continuous Integration) 파이프라인 내에서 소프트웨어 출판 워크플로우를 설정할 수 있도록 돕는 오픈 소스 프로젝트입니다. 이 도구는 소프트웨어 메타데이터를 수집하고 처리하여 출판 프로세스를 자동화합니다. HERMES는 Python으로 작성되었으며, 사용자가 필요에 따라 플러그인을 통해 기능을 확장할 수 있는 아키텍처를 제공합니다.

- **Performance Highlights**: 예비 사례 연구를 통해 HERMES 워크플로우의 실행 가능성과 적용 가능성을 평가하였으며, 결과는 HERMES가 연구 소프트웨어 메타데이터와 인프라의 필요를 충족할 수 있음을 보여줍니다.



### ProveRAG: Provenance-Driven Vulnerability Analysis with Automated Retrieval-Augmented LLMs (https://arxiv.org/abs/2410.17406)
- **What's New**: 본 연구에서는 사이버 보안 분야에서의 LLM(대형 언어 모델) 활용을 개선하기 위해 ProveRAG라는 시스템을 제안합니다. ProveRAG는 CVE(공통 취약점 및 노출)의 신속한 분석을 지원하며, 하여금 웹 데이터의 자동 검색 보강과 검증 가능한 증거를 통해 스스로의 응답을 평가할 수 있도록 설계되었습니다.

- **Technical Details**: ProveRAG는 NVD(국립 취약점 데이터베이스)와 CWE(공통 취약점 나열)를 통해 검증 가능한 출처의 데이터를 교차 참조하여 보안 분석가에게 신뢰할 수 있는 통찰력을 제공합니다. 이 시스템은 자기 비판 메커니즘을 통합하여 LLM의 편향과 오작동을 완화하려고 합니다. 또한, CVE 분석 편리를 위해 요약 기법을 활용하여 커다란 문서 덩어리에 의존하지 않고 대규모 데이터 처리의 효율성을 높입니다.

- **Performance Highlights**: ProveRAG는 2024년 CVE를 대상으로 테스트한 결과, 취약점 분석에서 99%, 완화 전략에서 97%의 정확도를 달성했습니다. 이는 직접 프롬프트(prompt) 방식이나 청킹(chunking) 검색 방식과 비교할 때 더 뛰어난 성능을 보여줍니다. ProveRAG는 사이버 보안의 복잡한 쿼리를 처리하는 데 적합하며, 신뢰할 수 있는 의사 결정을 지원합니다.



### Captions Speak Louder than Images (CASLIE): Generalizing Foundation Models for E-commerce from High-quality Multimodal Instruction Data (https://arxiv.org/abs/2410.17337)
Comments:
          Xinyi Ling and Bo Peng contributed equally to this paper

- **What's New**: 이 논문에서는 전자상거래(e-commerce)에서 다중 모달 데이터(multimodal data)의 최적 이용을 위한 최초의 대규모 및 고품질 다중 모달 지침 데이터셋인 MMECInstruct를 소개합니다. 또한, 전자상거래의 다중 모달 정보를 통합하기 위한 간단하면서도 효과적인 프레임워크인 CASLIE를 개발하였습니다.

- **Technical Details**: MMECInstruct 데이터셋은 75,000개의 샘플로 구성되어 있으며, 7개의 실제 전자상거래(task) 작업에 대한 지침과 이미지, 텍스트 입력, 출력을 포함합니다. CASLIE는 3개의 모듈(상황 조건화된 캡션 생성, 캡션 품질 평가, 모달리티 정보 융합)을 통해 이미지와 텍스트를 통합하는 시스템으로, 전자상거래 작업에 대한 높은 품질의 통합된 시각적 데이터를 제공합니다.

- **Performance Highlights**: CASLIE 모델은 5개의 고급 기준 모델과 비교해 모든 인도메인(indomain) 평가에서 평균 6.5% 높은 성능을 보였으며, 인도메인 외(out-of-domain) 설정에서도 3.3%의 성능 향상을 나타내어 강력한 일반화 가능성을 보여주었습니다.



### Offline Evaluation of Set-Based Text-to-Image Generation (https://arxiv.org/abs/2410.17331)
- **What's New**: 이 논문은 ideation 과정에서 텍스트-이미지(Text-to-Image, TTI) 시스템이 어떻게 지원되는지를 평가하기 위한 새로운 메트릭스를 제안합니다. 기존의 Fréchet Inception Distance(FID)와 같은 분포 유사성 메트릭이 아닌, 사용자 행동 모델을 기반으로 한 오프라인 평가 메트릭스를 개발하였습니다.

- **Technical Details**: 제안된 메트릭스는 사용자가 생성된 이미지 세트를 탐색하고 상호작용하는 방식을 명시적으로 모델링합니다. 이 모델은 정보 검색(Information Retrieval, IR) 기술을 활용하여 grid 레이아웃 및 이미지 다양성을 고려하여 평가합니다. 특히, 상대 순위(expected reciprocal rank)와 순위 편향 정밀도(rank-biased precision)를 적용하였습니다.

- **Performance Highlights**: 세 가지 데이터셋(MS-COCO 캡션, localized narratives 등)을 통해 실험한 결과, 사용자의 브라우징 방식과 이미지의 시각적 두드러짐을 모델링함으로써 인간의 선호도와의 일치성을 높임을 보여주었습니다.



New uploads on arXiv(cs.CV)

### DynamicCity: Large-Scale LiDAR Generation from Dynamic Scenes (https://arxiv.org/abs/2410.18084)
Comments:
          Preprint; 29 pages, 15 figures, 7 tables; Project Page at this https URL

- **What's New**: 이번 논문에서는 DynamicCity라는 새로운 4D LiDAR 생성 프레임워크를 소개하며, 이는 대규모, 고품질의 동적 LiDAR 장면을 생성할 수 있도록 설계되었습니다.

- **Technical Details**: DynamicCity는 두 개의 주요 모델로 구성되어 있습니다: 1) VAE(Variational Autoencoder) 모델, HexPlane을 압축된 4D 표현으로 학습하며, 새로운 Projection Module을 사용하여 4D LiDAR 특징을 여섯 개의 2D 특징 맵으로 효과적으로 압축합니다; 2) HexPlane 생성을 위한 DiT(Diffusion Transformer)-기반의 확산 모델. 또한 Padded Rollout Operation이 제안되어 모든 특징 평면을 정사각형 2D 특징 맵으로 재구성합니다.

- **Performance Highlights**: 많은 실험 결과에 따르면, DynamicCity는 CarlaSC와 Waymo 데이터 세트에서 기존의 4D LiDAR 생성 방법들에 비해 여러 지표에서 현저하게 우수한 성능을 보여주었습니다.



### FreeVS: Generative View Synthesis on Free Driving Trajectory (https://arxiv.org/abs/2410.18079)
Comments:
          Project Page: this https URL

- **What's New**: FreeVS는 기존의 NVS( Novel View Synthesis) 방식의 한계를 극복하고, 새로운 경로에서 카메라 뷰를 합성할 수 있는 혁신적인 접근법을 제안합니다. 이를 통해 실제 주행 장면에서 3D 일관성을 유지하며 정확한 시점 자세를 제어할 수 있습니다.

- **Technical Details**: 이 논문은 pseudo-image representation을 활용하여 3D 장면의 희소하지만 정확한 표현을 생성합니다. 각 기존 뷰에 대해 컬러 포인트 클라우드를 투영하여 pseudo-image를 생성하고, 이를 통해 생성 모델을 훈련하여 novel view를 합성할 수 있습니다.

- **Performance Highlights**: 실험 결과, FreeVS는 Waymo Open Dataset에서 기존 NVS 방법보다 높은 이미지 합성 성능을 보여 주었으며, 새로운 벤치마크에서도 우수한 성능을 발휘하였습니다.



### UnCLe: Unsupervised Continual Learning of Depth Completion (https://arxiv.org/abs/2410.18074)
Comments:
          Preprint

- **What's New**: UnCLe는 비지도 지속 학습(unsupervised continual learning)을 위한 표준화된 벤치마크로, 멀티모달 깊이 추정(multimodal depth estimation) 태스크인 깊이 완성(depth completion)을 평가합니다. 이는 동기화된 RGB 이미지와 희박한 깊이 맵으로부터 밀집(depth map)을 추정하는 어플리케이션을 목표로 합니다.

- **Technical Details**: UnCLe는 비정상(non-stationary) 분포를 시뮬레이션하여 깊이 완성 모델을 세트의 다양한 장면에 적응시키고, 항상 변하는 데이터를 지속적으로 학습하는 시나리오에서 평가합니다. 연구에서는 지속 학습 패러다임에서의 대표적인 방법들을 채택하여 비지도 깊이 완성을 가능하게 했습니다.

- **Performance Highlights**: 우리는 실내 및 실외 데이터셋에서 진행된 모델의 평가에서, 머신이 얼마나 전에 학습한 정보를 잊어버리는지를 정량적으로 측정하였으며, 모델 역전(model inversion)을 추가적인 평가 지표로 도입했습니다. 비지도 깊이 완성학습 분야는 여전히 해결할 수 없는 문제가 많으며, UnCLe은 연구자들이 이 문제를 해결하고 발전시킬 수 있는 플랫폼을 제공합니다.



### WorldSimBench: Towards Video Generation Models as World Simulators (https://arxiv.org/abs/2410.18072)
- **What's New**: 본 연구에서는 World Simulators의 기능성을 분류하는 일계층 구조를 제안하고, Explicit Perceptual Evaluation과 Implicit Manipulative Evaluation을 포함하는 이중 평가 프레임워크인 WorldSimBench를 제안합니다. 이는 기존의 예측 모델 개발과 평가의 한계를 극복하는 데 기여합니다.

- **Technical Details**: WorldSimBench는 HF-Embodied Dataset을 기반으로 하여 시각적 정밀성 및 행동 수준 평가를 포함합니다. 각 차원에서 Visual Quality, Condition Consistency, Embodiment와 같은 평가 기준을 상세히 정의하였으며, 다양한 Embodied 환경에서 발생할 수 있는 인터랙션을 평가합니다.

- **Performance Highlights**: 본 연구는 Open-Ended Embodied Environment, Autonomous Driving, Robot Manipulation의 세 가지 시나리오에서 총 8개의 비디오 생성 모델을 평가하였으며, HF-Embodied Dataset을 통한 효율적인 평가를 통해 비디오 생성 모델의 혁신을 유도할 수 있는 통찰력을 제공하였다.



### TP-Eval: Tap Multimodal LLMs' Potential in Evaluation by Customizing Prompts (https://arxiv.org/abs/2410.18071)
- **What's New**: 최근 다중 모달 대규모 언어 모델(multimodal large language models, MLLMs)의 성능 평가에서의 의존성을 해결하기 위한 새로운 평가 프레임워크인 TP-Eval이 제안되었습니다.

- **Technical Details**: TP-Eval은 모델별 최적의 prompts를 자동으로 커스터마이즈하여 평가 시 발생할 수 있는 편향을 줄이는 방법론을 구현하고 있습니다. 이 과정에서 기존의 프롬프트 최적화와는 달리 이미지와 텍스트 정보를 통합하여 MLLMs에 최적화된 방식으로 진행됩니다.

- **Performance Highlights**: TP-Eval을 활용한 실험을 통해 MLLMs의 진정한 성능을 보다 정확하게 평가할 수 있으며, 여러 모델 간의 평가 편향 현상을 효과적으로 완화하는 것으로 나타났습니다.



### CLEAR: Character Unlearning in Textual and Visual Modalities (https://arxiv.org/abs/2410.18057)
- **What's New**: 새로운 벤치마크인 CLEAR를 소개하여 다중 모달 언러닝(Multimodal Unlearning, MMU) 방법을 평가합니다. 기존의 언러닝 방법들과는 달리, CLEAR는 텍스트-비주얼 데이터셋에 대해 광범위한 평가를 가능하게 합니다.

- **Technical Details**: CLEAR 데이터셋은 200개의 허虚 개인과 3,700개의 이미지, 관련 질문-답변 쌍을 포함하고 있으며, 다중 모달 언러닝 기법을 평가하는 데 사용됩니다. 10가지 언러닝 방법을 검토하고, LoRA 가중치에 대한 단순한 ℓ₁ 정규화를 적용하여 재앙적 망각(Catastrophic Forgetting)을 완화하는 효과를 보여줍니다.

- **Performance Highlights**: MMU에서의 성능 평가 결과, 기존의 언러닝 방법들이 다중 모달 설정에서 어려움을 겪고 있음을 보여주며 새로운 접근 방식의 필요성을 강조합니다.



### In-Pixel Foreground and Contrast Enhancement Circuits with Customizable Mapping (https://arxiv.org/abs/2410.18052)
- **What's New**: 이 논문에서는 혁신적인 인픽셀 대비 향상 회로가 소개됩니다. 이 회로는 픽셀 회로 내에서 직접 이미지 처리를 수행할 수 있습니다.

- **Technical Details**: 회로는 여러 작동 모드에 대해 조정할 수 있으며, 전경 향상 모드에서는 낮은 강도의 배경 픽셀을 거의 0에 가깝게 억제하여 전경을 분리합니다. 대비 향상 모드에서는 전체 이미지의 대비를 개선합니다. 이 기능은 설계 단계와 실시간 모두에서 사용자화(Customization)가 가능하여 다양한 조명 조건에 적응할 수 있습니다. 설계된 픽셀 회로 모델이 개발되어 전체 픽셀 배열에 적용되었습니다.

- **Performance Highlights**: HSPICE를 통해 수행된 시뮬레이션에서는 전경 향상 모드에서 Michelson Contrast Ratio (CR)가 거의 6배 증가함을 보여주었습니다. 이 결과는 다양한 이미징 환경에서도 실시간, 적응형 대비 향상의 잠재력을 나타냅니다.



### Real time anomalies detection on video (https://arxiv.org/abs/2410.18051)
- **What's New**: 이 논문에서는 비디오에서 이상 징후를 실시간으로 감지하기 위한 딥 러닝(Deep Learning) 접근 방식을 제안합니다. 기존 보안 카메라의 사용 방식이 과거 사건을 보여주는 데 그치고 있다는 문제를 해결하고자 하며, CNN(Convolutional Neural Networks)과 LSTM(Long Short-Term Memory) 모델을 결합하여 실시간 분석을 시도합니다.

- **Technical Details**: 제안된 모델은 VGG19라는 CNN 모델을 사용하여 영상에서 특징들을 추출하고, 이를 LSTM 또는 GRU(Gated Recurrent Unit) 모델에 전달하여 시계열 데이터를 분석합니다. 데이터 전처리에는 Zooming, Cropping, Mirroring과 같은 데이터 증강(data augmentation) 기법이 포함되어 있으며, 최종적으로는 binary cross-entropy 함수를 손실 함수로 사용하고 SGD(Stochastic Gradient Descent) 최적화 기법을 적용합니다.

- **Performance Highlights**: 체크한 비디오 데이터셋은 4000개 이상의 비디오로 구성되며, 이상 징후가 있는 장면과 일반 장면을 구분하여 실험하였습니다. 초기 테스트 결과 다른 클래스와 비교했을 때 단순한 Fight/Normal 클래스에서 더욱 향상된 성능을 보였으며, 여러 신경망 구조를 비교 분석하여 최적의 모델을 찾아가는 과정이 진행되었습니다.



### Scalable Ranked Preference Optimization for Text-to-Image Generation (https://arxiv.org/abs/2410.18013)
Comments:
          Project Page: this https URL

- **What's New**: 이 연구에서는 Direct Preference Optimization (DPO) 방법을 활용하여 텍스트-이미지 모델의 효율적인 학습을 위한 대규모 합성 데이터셋을 수집하는 접근 방식을 제안합니다. 이는 수작업 주석이 필요 없는 합성 데이터를 통해 이루어지며, 다양한 보상 모델을 사용하여 이미지에 대한 선호도를 평가합니다.

- **Technical Details**: 합성 선호도 데이터셋(Syn-Pic)을 생성하여 텍스트-이미지(T2I) 모델의 효율성을 높이며, 기존 방법들에서 발생할 수 있는 '보상 해킹'(reward hacking) 문제를 해결하기 위해 여러 보상 모델의 점수를 집계합니다. 또한, RankDPO라는 새로운 DPO 목표를 도입하여 선호도 손실을 할인 누적 이득(Discounted Cumulative Gain, DCG)으로 가중치 부여하여 더욱 향상된 성능을 자랑합니다.

- **Performance Highlights**: 이 방법을 사용하여 SDXL 및 SD3-Medium 모델에서 성능이 눈에 띄게 향상되었으며, 다양한 벤치마크 데이터셋에서 최신 기술 수준의 결과를 달성했습니다. 특히, 기존 데이터셋인 Pick-a-Picv2에 비해 3배 적은 이미지를 사용하여도 우수한 성능을 보여주었습니다.



### Characterization of the multiplicity of solutions for camera pose given two vertically-aligned landmarks and accelerometer (https://arxiv.org/abs/2410.17997)
Comments:
          32 pages, 8 figures

- **What's New**: 이번 연구에서는 중력과 정렬된 좌표 체계에서 두 개의 레이블이 붙은 랜드마크의 이미지로부터 카메라의 위치와 방향을 복원하는 문제를 다룹니다. 기존의 PnP 문제에서 중력 데이터를 추가하여 새로운 변형 문제를 제시합니다.

- **Technical Details**: 이 연구는 세 가지 특이 케이스를证明하며, 각 경우마다 해의 개수를 구체적으로 분류합니다. 특히, 두 랜드마크가 같은 고도에 위치하고 카메라가 다른 고도에 있을 때는 항상 고유한 해가 존재합니다. 또한, 랜드마크가 명시되어 있지 않은 경우에도 여전히 해의 개수는 항상 1 또는 2로 제한됩니다.

- **Performance Highlights**: 실험적으로 소비자 휴대폰에서 구현된 수치 시뮬레이션을 통해, 고유한 해가 존재하는 케이스에서 두 랜드마크가 수평선상에 위치할 때의 결과를 확인했습니다. 이 결과는 드론이나 게임 컨트롤러의 위치 추정에 유용하게 적용될 수 있습니다.



### A Pipeline for Segmenting and Structuring RGB-D Data for Robotics Applications (https://arxiv.org/abs/2410.17988)
- **What's New**: 새로운 RGB-D 데이터 세분화 및 구조화 파이프라인을 제시합니다. 기존의 접근 방식은 단순히 기하학적 정보만 추출했으나, 우리의 방법은 환경에 대한 의미론적 이해를 포함하여 더 발전된 로봇 내비게이션 및 조작 알고리즘을 개발할 수 있는 가능성을 제공합니다.

- **Technical Details**: 이 파이프라인은 RGB-D 데이터를 정확한 의미론적 마스크로 세분화합니다. 이러한 마스크는 캡처된 점 구름을 의미론적으로 구분된 점 구름으로 융합하는 데 사용됩니다. 우리는 Universal Scene Description (USD) 파일 형식을 사용하여 이 정보를 저장합니다. 최신 세분화 알고리즘인 Segment Anything Model 2 (SAM2)와 SegFormer 및 OneFormer를 통해 정확한 의미론적 마스크를 생성하는 하이브리드 접근 방식을 채택했습니다.

- **Performance Highlights**: 우리의 방법은 3D 점 구름 융합 및 메쉬 계산의 처리 속도를 높이며, 재구성된 3D 물체의 기하학적 특성에 대한 가정을 하지 않아서 다운스트림 응용 프로그램이 측정된 기하학을 신뢰할 수 있도록 합니다. 결과적으로, 사람 추적 및 장면 재구성을 통한 효과적인 RGB-D 데이터 분석이 가능해졌습니다.



### Robust Two-View Geometry Estimation with Implicit Differentiation (https://arxiv.org/abs/2410.17983)
Comments:
          IROS 2024 Accepted

- **What's New**: 본 논문에서는 미분 가능한 강인 손실 함수 적합성 기반의 새로운 두 뷰 기하학 추정 프레임워크를 제안합니다. 이 프레임워크는 강인 기본 행렬 추정을 암묵적 레이어로 처리하여 시간에 따른 역전파(backpropagation)를 피하고 수치적 안정성을 크게 향상시킵니다.

- **Technical Details**: 우리는 피쳐 매칭 단계에서의 정보를 최대한 활용하기 위해 매칭 신뢰도에 따라 달라지는 학습 가능한 가중치를 통합합니다. 이로 인해 피쳐 추출(feature extraction), 매칭(matching) 및 두 뷰 기하학 추정이 통합된 end-to-end 학습 파이프라인으로 결합됩니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 야외 및 실내 시나리오의 카메라 포즈 추정(camera pose estimation) 작업에서 고전 및 학습 기반의 최신 기법들보다 훨씬 우수한 성능을 보임을 확인할 수 있었습니다.



### VR-Splatting: Foveated Radiance Field Rendering via 3D Gaussian Splatting and Neural Points (https://arxiv.org/abs/2410.17932)
- **What's New**: 이번 논문에서는 가상 현실(VR) 시스템에 최적화된 새로운 off-axis foveated rendering 기법을 제안합니다. 이 기법은 빠른 렌더링을 가능하게 하는 neural point rendering의 선명한 출력과 3D Gaussian splatting (3DGS)의 부드러운 렌더링을 결합하여 시각적으로 매력적인 VR 경험을 제공합니다.

- **Technical Details**: 제안된 foveated rendering 접근법은 인간 시각 시스템(HVS)의 인지적 한계를 활용합니다. 이를 통해 foveal(중심 시야) 영역에서는 고해상도 이미지를 렌더링하고 peripheral(주변 시야) 영역에서는 해상도를 낮춰서 효과적으로 자원 낭비를 줄입니다. Eye tracking을 통해 사용자가 주시하고 있는 영역을 주의 깊게 처리하여 렌더링 성능을 극대화합니다.

- **Performance Highlights**: 우리 시스템은 실시간 VR 상호 작용에 필요한 성능 기준을 충족하며, 표준 VR-ready 3DGS 구성에 비해 인식된 선명도와 세부 묘사가 증가하여 사용자의 몰입 경험을 향상시킵니다.



### Gaze-Assisted Medical Image Segmentation (https://arxiv.org/abs/2410.17920)
Comments:
          16 pages, 4 figures, Accepted to AIM-FM Workshop @ NeurIPS'24

- **What's New**: 이 논문은 의료 이미지 분할에서 인간 시선(gaze)을 상호작용 입력으로 활용하여 세미-자동화된 분할 수정의 개념을 제안합니다. 특히 Segment Anything Model in Medical Images (MedSAM) 모델을 세미-자동화된 분할 수정 작업에 맞게 미세 조정하였습니다.

- **Technical Details**: 연구에서는 MedSAM을 사용하여 인간의 시선 데이터로부터 CT 스캔에 대한 분할 마스크를 실시간으로 수정하는 방법을 제안합니다. 이 과정에서 gaze 정보(눈의 움직임 데이터)를 입력 프롬프트로 활용하며, 이를 통해 시각적 상호작용을 강화합니다. 실험은 120건의 CT 스캔 이미지를 포함한 공개 WORD 데이터베이스에서 수행되었습니다.

- **Performance Highlights**: gaze 보조 MedSAM의 성능은 기존 최첨단(semi-automated) 분할 모델보다 우수하였으며, 16개의 복부 장기에 대한 평균 Dice 계수는 90.5%로 나타났습니다. 이는 기존 nnUNetV2, ResUNet, 원본 MedSAM 모델의 각각 85.8%, 86.7%, 81.7% 보다 높은 수치입니다.



### Addressing Asynchronicity in Clinical Multimodal Fusion via Individualized Chest X-ray Generation (https://arxiv.org/abs/2410.17918)
Comments:
          Accepted by NeurIPS-24

- **What's New**: DDL-CXR는 EHR과 CXR 간의 비동기성 문제를 해결하고 개인화된 CXR 이미지를 생성하여 임상 예측의 성능을 향상시키는 혁신적인 방법입니다.

- **Technical Details**: DDL-CXR는 잠재적 확산 모델(latent diffusion model)과 변분 오토인코더(variational auto-encoder)를 이용하여 환자 맞춤형 CXR 이미지를 생성합니다. 환자의 이전 CXR 이미지를 참조 이미지로 사용하고 EHR 타임 시리를 인코딩하기 위해 Transformer 모델을 활용하여 질병의 진행 과정을 효과적으로 캡처합니다. 이는 EHR과 CXR 간의 상호작용을 개선합니다.

- **Performance Highlights**: MIMIC 데이터셋을 사용한 실험에서 DDL-CXR는 기존의 방법에 비해 다중 모드 클리닉 예측과 개별 CXR 생성에서 일관되게 뛰어난 성능을 보였습니다.



### A utility-based spatial analysis of residential street-level conditions; A case study of Rotterdam (https://arxiv.org/abs/2410.17880)
- **What's New**: 이 연구는 전통적인 주거 위치 선택 모델에서 간과되었던 지역적 거리 조건을 통합하여, 주택 선택 분석에 있어 거리 수준 이미지(street-level images)의 활용 가능성을 탐구합니다.

- **Technical Details**: 본 연구에서는 컴퓨터 비전(computer vision) 기능을 통합한 이산 선택 모델(discrete choice models)을 활용하여, 로테르담(Rotterdam)에서 거리 수준 조건에서 파생된 유틸리티(utility)의 공간적 분포를 분석합니다. 이 모델은 세멘틱 정규화 레이어(semantic regularisation layer)를 도입하여 설명 가능성을 높이고, 별도의 파이프라인 없이 이미지로부터 정보를 추출하게 합니다.

- **Performance Highlights**: 연구 결과는 거리 수준 조건의 유틸리티가 지역적으로 크게 다르며, 특히 시中心의 높은 부동산 가격이 매력적인 거리 조건에 기인하지 않음을 보여줍니다. 오히려, 문제 지역으로 인식되는 도시 남부 지역의 거리 환경이 더욱 매력적임을 발견하였습니다. 이러한 결과는 도시 계획에 거리 수준 조건을 통합하는 향후 연구를 위한 기초를 마련합니다.



### Blendify -- Python rendering framework for Blender (https://arxiv.org/abs/2410.17858)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 Blender와 함께 통합되어 사용될 수 있는 경량의 Python 기반 프레임워크인 Blendify를 소개합니다. Blendify는 사용자가 Blender의 기본 API를 손쉽게 다룰 수 있게 해주며, 장면 생성 및 렌더링을 위한 고수준 API를 제공합니다.

- **Technical Details**: Blendify는 Blender의 렌더링 엔진과 완벽하게 통합되어, 객체 생성 자동화, 색상 및 재료 연결 자동화, shadow-catcher 객체와 같은 기능을 구현하며, 고품질의 ray-tracing 렌더링 출력을 지원합니다. 코드는 Google Colab과 호환되며, 사용자가 Blender *.blend 파일로 수출 및 수입할 수 있습니다.

- **Performance Highlights**: Blendify의 주요 기능으로는 3D 객체의 효율적인 렌더링을 지원하고, 다양한 재료 정의 및 색상화 기능 제공, point cloud 렌더링 지원 등이 있습니다. 사용자가 카메라를 설정하고 렌더링할 수 있도록 해주며, compeelated한 설치 과정 없이 복잡한 장면 설정을 가능하게 합니다.



### ROCKET-1: Master Open-World Interaction with Visual-Temporal Context Prompting (https://arxiv.org/abs/2410.17856)
- **What's New**: 본 논문은 비전-언어 모델(Vision-Language Models, VLMs)을 활용하여 개방형 세계 환경에서 존재하는 결정을 내리기 위한 도전과제를 다룹니다. 특히, 개체(entities)와 추상 개념(abstract concepts) 간의 원활한 연결이 어려운 문제를 해결하기 위해 'visual-temporal context prompting'라는 새로운 통신 프로토콜을 제안합니다.

- **Technical Details**: 제안된 방법은 객체 분할(object segmentation)을 과거 및 현재 관찰에서 활용하여 정책 모델(policy models)과의 상호작용을 유도합니다. ROCKET-1이라는 저수준 정책(low-level policy)은 시각적 관찰 및 분할 마스크를 연결하여 행동을 예측합니다. 이는 transformer를 사용해 관찰 간의 의존성을 모델링하고, SAM-2를 통해 실시간 객체 추적 기능을 제공합니다.

- **Performance Highlights**: Minecraft 환경에서 진행된 실험을 통해 제안된 방법이 기존 방법으로는 수행할 수 없었던 복잡한 작업들을 성공적으로 완료할 수 있음을 보여주었습니다. 이는 공간 이해(spatial understanding)에 크게 의존하는 창의적인 작업들을 해결할 수 있는 잠재력을 시사합니다.



### TAGE: Trustworthy Attribute Group Editing for Stable Few-shot Image Generation (https://arxiv.org/abs/2410.17855)
Comments:
          Accepted by International Conference on Signal Processing Systems Conference

- **What's New**: TAGE라는 새로운 이미지 생성 네트워크를 소개하며, 세 가지 주요 모듈인 Codebook Learning Module (CLM), Code Prediction Module (CPM), Prompt-driven Semantic Module (PSM)으로 구성되어 있음.

- **Technical Details**: CPM 모듈은 범주 비이성(attribute)-독립 속성의 의미적 차원을 탐구하여 이를 이산 코드북에 캡슐화하고, 주요 요점은 이미지가 속성의 조합으로 이루어져 있다는 것. PSM 모듈은 Transformer 구조에 통합된 의미적 단서를 생성하여 편집하고자 하는 속성에 대한 모델의 이해를 향상시킴.

- **Performance Highlights**: Animal Faces, Flowers, VGGFaces 데이터셋을 활용한 실험 결과, 제안된 방법이 다른 few-shot 이미지 생성 기술에 비해 우수한 성능과 높은 안정성을 보여줌.



### Few-shot NeRF by Adaptive Rendering Loss Regularization (https://arxiv.org/abs/2410.17839)
Comments:
          Accepted by ECCV2024

- **What's New**: 이 논문은 Neural Radiance Field (NeRF)와 관련된 새로운 방법을 제안합니다. 특히, 저자들은 Positional Encoding (PE)의 주파수 규제가 렌더링 손실(rendering loss)과의 불일치가 문제임을 밝혔고, 이를 해결하기 위해 Adaptive Rendering loss regularization을 도입한 AR-NeRF를 발표했습니다.

- **Technical Details**: AR-NeRF는 두 가지 주요 기능을 포함합니다. 첫째, 두 단계로 나뉜 렌더링 감독(rendering supervision)으로 초기 훈련 단계에서는 흐릿한 입력 뷰를 사용하고, 이후 단계에서는 원본 입력 뷰를 사용합니다. 둘째, 불확실성 학습(uncertainty learning)을 기반으로 한 적응형 렌더링 손실 가중치 학습 전략을 도입하여 서로 다른 픽셀 감독에 대한 렌더링 손실 가중치를 동적으로 조정합니다.

- **Performance Highlights**: AR-NeRF는 다양한 데이터 세트에서 최첨단 성능을 달성하며, 특히 객체 수준 및 복잡한 장면에서 우수한 결과를 보여줍니다.



### Exploiting Text-Image Latent Spaces for the Description of Visual Concepts (https://arxiv.org/abs/2410.17832)
Comments:
          19 pages, 7 figures, to be published in ICPR

- **What's New**: 본 연구는 Concept Activation Vectors (CAVs)을 해석하는 데 도움이 되는 텍스트 설명을 제안하여, 인간이 이해할 수 있는 개념의 최신 발견된 집합을 해석하는 방식을 제공합니다. 이는 각 CAV에 대해 가장 관련 있는 이미지를 텍스트-이미지 임베딩에 매핑하여 수행됩니다.

- **Technical Details**: 저자들은 CAV의 시각적 설명을 위해 가장 관련 있는 receptive fields를 활용하여 텍스트 설명을 생성합니다. 이를 통해 이미지의 소음이 줄어들고, 개념을 텍스트로 설명하는 과정에서 인간의 해석 필요성이 최소화됩니다. 또한, k𝑘kitalic_k 가장 높은 등급의 설명으로부터 단일 공통 설명을 도출하여 중복성을 줄입니다.

- **Performance Highlights**: 여러 실험을 통해 CAV 레이블이 주어지지 않은 경우에도 제안된 방법이 정확한 CAV 설명을 제공하며 개념 해석의 도전을 줄인다. 이 방법은 자동적으로 발견된 개념을 보다 효과적으로 해석할 수 있도록 합니다.



### DREB-Net: Dual-stream Restoration Embedding Blur-feature Fusion Network for High-mobility UAV Object Detection (https://arxiv.org/abs/2410.17822)
- **What's New**: 이 논문에서는 고속 이동 UAV(무인 항공기)가 캡처한 흐릿한 이미지에서의 객체 탐지 문제를 해결하기 위한 새로운 알고리즘인 DREB-Net(Dual-stream Restoration Embedding Blur-feature Fusion Network)을 제안합니다. 이 모델은 흐릿한 이미지에 대한 탐지 정확도를 향상시키기 위해 Blurry image Restoration Auxiliary Branch (BRAB)를 포함하며, Multi-level Attention-Guided Feature Fusion (MAGFF) 모듈을 통해 특성 추출을 최적화합니다.

- **Technical Details**: DREB-Net은 두 가지 주요 구성 요소로 이루어져 있습니다. 첫째, 흐릿한 이미지 복원을 위한 보조 브랜치(BRAB)로, 이 브랜치는 훈련 단계에서 흐릿한 이미지를 개선하고 MSE(Mean Squared Error) 및 SSIM(Structural Similarity Index Measure) 손실 함수의 조합을 사용하여 성능 향상을 꾀합니다. 둘째, MAGFF 모듈을 통해 로컬 및 글로벌 어텐션 메커니즘을 활용하여 특징 융합 과정에서 가변 가중치를 적용하며, LFAMM(Learnable Frequency domain Amplitude Modulation Module)을 도입하여 주파수 영역에서 이미지 품질을 향상시킵니다.

- **Performance Highlights**: 실험 결과, DREB-Net은 움직임이 흐릿한 이미지에서도 객체 탐지를 효과적으로 수행할 수 있음을 보여주었으며, 기존 방법들과 비교하여 성능상의 유의미한 향상을 보였습니다. 빠른 처리 속도와 높은 탐지 정확도를 달성함으로써 UAV 이미지 처리 분야에서 널리 적용될 가능성을 가지게 되었습니다.



### EntityCLIP: Entity-Centric Image-Text Matching via Multimodal Attentive Contrastive Learning (https://arxiv.org/abs/2410.17810)
- **What's New**: 최근 이미지-텍스트 매칭 분야에서 주목할 만한 발전이 이루어졌지만, 기존 모델들은 여전히 일반적인 쿼리에 치중하고 있으며 세부 쿼리의 의도를 반영하는 데 어려움을 겪고 있습니다. 이 논문에서는 구체적인 엔티티와 관련된 정보를 포함하는 	extbf{E}ntity-centric 	extbf{I}mage-	extbf{T}ext 	extbf{M}atching (EITM) 문제를 다룬 모델인 EntityCLIP을 제안합니다.

- **Technical Details**: 본 연구는 CLIP 모델을 백본으로 하여 멀티모달 주의 집중 대조 학습(Multimodal Attentive Contrastive Learning) 프레임워크를 개발하였습니다. 이 프레임워크는 LLMs(대형 언어 모델)를 활용해 생성된 해설 텍스트를 통해 엔티티 관련 쿼리와 이미지를 매칭하는 데 사용됩니다. 다중모달 집중 전문가(Multimodal Attentive Experts, MMAE) 모듈은 이 해설 텍스트를 통합하여 엔티티 관련 텍스트와 이미지를 하나의 공유된 의미 공간으로 좁히는 역할을 합니다.

- **Performance Highlights**: N24News, VisualNews, GoodNews 등 세 가지 소셜 미디어 뉴스 벤치마크에서 수행된 광범위한 실험 결과, EntityCLIP은 기존 방법들을 명확한 마진으로 초월하는 성능을 보여주었습니다.



### An Intelligent Agentic System for Complex Image Restoration Problems (https://arxiv.org/abs/2410.17809)
- **What's New**: 이 논문에서는 실제 세계의 이미지 복원(Real-world Image Restoration, IR) 문제를 해결하기 위한 새로운 시스템인 AgenticIR을 제안합니다. 이 시스템은 사람의 문제 해결 방식을 모방하여 지각(Perception), 일정 수립(Scheduling), 실행(Execution), 반성(Reflection), 재조정(Rescheduling)의 다섯 가지 주요 단계를 따릅니다.

- **Technical Details**: AgenticIR은 텍스트 생성을 통해 상호 작용하는 대형 언어 모델(LLMs)과 비전-언어 모델(VLMs)을 활용하여 복잡한 IR 작업을 해결하기 위한 도구 상자를 동적으로 운영합니다. VLM은 이미지 품질 분석을 위해 미세 조정되며, LLM은 단계별로 시스템을 안내하는 추론 기능을 수행합니다. 또한, LLM의 특정 IR 지식 부족을 보완하기 위해 자가 탐색(self-exploration) 방법을 도입하여 복원 결과를 관찰하고 요약하는 능력을 부여합니다.

- **Performance Highlights**: 실험 결과, AgenticIR은 복잡한 IR 과제를 처리하는 데 유망한 가능성을 보여주며, 이는 자동화된 지능적 이미지 처리에 대한 연구의 이정표가 될 것입니다. 이 시스템은 개별 모델을 넘어서 여러 모델의 복합적 상호작용을 관리하고, 동적으로 변화하는 작업을 수행할 수 있는 능력을 가지고 있습니다.



### GenUDC: High Quality 3D Mesh Generation with Unsigned Dual Contouring Representation (https://arxiv.org/abs/2410.17802)
Comments:
          ACMMM 2024, code:this https URL

- **What's New**: 이번 연구에서는 복잡한 구조와 사실적인 표면을 생성하기 위한 3D mesh 생성 프레임워크인 GenUDC를 제안합니다. 이는 Unsigned Dual Contouring (UDC) 방식의 mesh 표현을 활용하여 기존 메소드를 개선하였습니다.

- **Technical Details**: GenUDC는 두 단계의 생성 프로세스를 적용하여 mesh를 생성합니다. 첫 번째 단계에서 rough shape인 face part를 생성하고, 두 번째 단계에서 vertex part를 생성하여 세밀한 형상을 만듭니다. UDC는 정규 격자로 mesh를 이산화하고, face 부분과 vertex 부분으로 나누어 복잡한 구조와 세세한 디테일을 복원합니다.

- **Performance Highlights**: GenUDC는 MeshDiffusion에 비해 속도가 3274% 향상되었으며, 총 메모리 사용량은 13%로 감소했습니다. 다양한 실험 결과, GenUDC는 기존의 mesh 생성 방법들에 비해 우수한 성능을 보였습니다.



### TranSPORTmer: A Holistic Approach to Trajectory Understanding in Multi-Agent Sports (https://arxiv.org/abs/2410.17785)
Comments:
          Accepted to ACCV 2024

- **What's New**: 이 논문은 다중 에이전트 스포츠 시나리오에서 경로 예측, 결측 관측치 보완, 보이지 않는 에이전트 상태 추론 및 다양한 글로벌 상태 분류를 통합할 수 있는 TranSPORTmer라는 혁신적인 텍스트 기반 프레임워크를 소개합니다.

- **Technical Details**: TranSPORTmer는 Set Attention Blocks (SABs)을 활용하여 시간적 동적 및 사회적 상호작용을 효과적으로 캡처하며, 모든 작업은 모양을 감추는 input mask에 의해 조정됩니다. 이 모델은 각 프레임에서 게임 상태 분류를 위해 CLS extra agent를 도입하여 패스, 소유, 제어되지 않는 상태 및 비활성 구간을 포함한 여러 상태를 분류합니다. 또한 불확실성을 반영한 learnable uncertainty mask를 손실 함수에 통합하여 예측 정확도를 높입니다.

- **Performance Highlights**: TranSPORTmer는 축구 및 농구 데이터셋에서 플레이어 예측, 플레이어 예측 보완, 공 추론 및 공 보완에서 기존의 최신 인공지능 모델들을 뛰어넘는 성능을 보였습니다. 특히 공 추론 작업에서는 25%의 성능 향상을 기록하였습니다.



### ADEM-VL: Adaptive and Embedded Fusion for Efficient Vision-Language Tuning (https://arxiv.org/abs/2410.17779)
- **What's New**: 최근 비전-언어(vision-language) 모델의 발전으로 ADEM-VL이라는 새로운 효율적인 접근 방식을 제안합니다. 이는 사전 훈련된 대규모 언어 모델(LLM)을 기반으로 하여 비전 정보를 통합하는 새로운 방법을 사용합니다.

- **Technical Details**: ADEM-VL은 매개변수 없는 교차 주의(cross-attention) 메커니즘을 채택하여 멀티모달 융합에서 유사도를 측정하며, 시각적 특징을 언어 공간에 임베딩하는 것으로 훈련 가능한 매개변수의 수를 크게 줄입니다. 또한 단일 포워드 패스를 통해 멀티스케일 시각적 특징을 생성하는 효율적인 방법을 도입했습니다.

- **Performance Highlights**: ADEM-VL은 ScienceQA 데이터셋에서 평균 94.55%의 정확도를 기록하며 기존 접근 방식을 0.77% 초과하여 향상된 성능을 보여주었으며, 훈련 및 추론 단계에서 각각 15%와 3% 더 빠른 속도를 자랑합니다.



### Quasi-Medial Distance Field (Q-MDF): A Robust Method for Approximating and Discretizing Neural Medial Axis (https://arxiv.org/abs/2410.17774)
- **What's New**: 본 논문에서는 디지털 기하 처리 분야에서 중요한 역할을 하는 medial axis를 효율적으로 계산하기 위한 새로운 방법론을 제안합니다. 기존의 explicit 방법에서 벗어나, signed distance field (SDF)와 medial field (MF)의 관계를 활용하여 implicit 방식으로 medial axis transform (MAT)을 재구성합니다.

- **Technical Details**: 제안된 방법은 solid shape의 SDF와 MF 간의 차이를 통해 unsigned distance field (UDF)를 얻고, 이를 기반으로 compact medial axis transform을 추출합니다. double covering 기법을 이용하여 UDF의 zero level-set에서 medial axis를 추출하며, sharp feature를 가지는 영역에 대한 differentiation을 강화하여 더 정밀한 medial axis를 생성합니다. 이러한 방법은 thorny meshes 및 defect가 있는 point clouds를 처리하는 데 특히 효과적입니다.

- **Performance Highlights**: 실험 결과, 본 연구의 방법은 기존 방법들에 비해 accuracy와 robustness에서 유의미한 개선을 보이며, 복잡한 3D 형상을 효과적으로 재구성할 수 있음을 입증하였습니다.



### AdaDiffSR: Adaptive Region-aware Dynamic Acceleration Diffusion Model for Real-World Image Super-Resolution (https://arxiv.org/abs/2410.17752)
Comments:
          18 pages, 6 figures, ECCV2024 accepted

- **What's New**: 이번 연구에서는 DMs 기반의 SR 방식인 AdaDiffSR을 제안합니다. 이 방식은 각 이미지 영역에 대한 동적 timesteps 샘플링 전략을 이용하여 계산 자원의 활용을 극대화합니다.

- **Technical Details**: AdaDiffSR은 multi-metrics latent entropy module (MMLE)와 dynamic timesteps sampling strategy (DTSS)를 도입하여 denoising 과정 중 정보 이득을 파악하고 동적으로 timesteps를 조정합니다. 또한 progressive feature injection module (PFJ)을 통해 원본 이미지 특성을 받아들이는 과정이 동적으로 조정됩니다.

- **Performance Highlights**: 실험 결과, AdaDiffSR은 기존의 최첨단 DMs 기반 SR 방법보다 비교 가능한 성능을 보이면서도 적은 계산 자원과 짧은 추론 시간을 소모합니다.



### VISAGE: Video Synthesis using Action Graphs for Surgery (https://arxiv.org/abs/2410.17751)
Comments:
          Accepted at MICCAI 2024 Embodied AI and Robotics for HealTHcare (EARTH) Workshop

- **What's New**: 이 논문은 외과 영상 생성(future video generation in laparoscopic surgery)이라는 새로운 작업을 도입하고, 이를 통해 기존 외과 데이터의 한계를 극복하고 다양한 응용 프로그램에 기여할 수 있는 방법을 제시합니다.

- **Technical Details**: 제안된 방법인 VISAGE(VIdeo Synthesis using Action Graphs for Surgery)는 외과 절차의 순차적인 특성을 포착하기 위해 액션 장면 그래프(action scene graphs)의 힘을 활용하며, 확산 모델(diffusion models)을 사용하여 시간적으로 일관된 비디오 시퀀스를 합성(synthesize)합니다. 이 모델은 단일 초기 프레임과 액션 그래프 트리플(action graph triplets)을 기반으로 미래 프레임을 예측합니다.

- **Performance Highlights**: VISAGE는 CholecT50 데이터셋에서 높은 충실도의 비디오를 생성하며, 여러 메트릭과 정성 평가에서 기존 방법들을 초월하는 성능을 보여줍니다.



### Efficient Neural Implicit Representation for 3D Human Reconstruction (https://arxiv.org/abs/2410.17741)
- **What's New**: 이 연구에서는 HumanAvatar라는 혁신적인 접근 방식을 제안하여 단일 RGB 비디오 소스로부터 효율적으로 정확한 인간 아바타를 재구성합니다. 이 시스템은 HuMoR, Instant-NGP, Fast-SNARF라는 세 가지 최신 기술을 통합하여 높은 재구성 충실도 및 속도를 달성합니다.

- **Technical Details**: HumanAvatar는 단일 비디오 입력을 통해 인간의 포즈 파라미터를 추정하고 이를 기반으로 고충실도 아바타를 재구성합니다. HuMoR 모델을 이용해 더 정밀한 인간 포즈를 예측하고, Instant-NGP를 통해 볼륨 렌더링 속도를 높이며, Fast-SNARF를 통해 아바타의 동적 변형을 지원합니다. 포즈에 민감한 공간 감소 기술을 사용하여 렌더링 품질과 계산 효율성을 최적화합니다.

- **Performance Highlights**: HumanAvatar는 고급 재구성 기술과 비교하여 품질이 동등하거나 우수하며, 복잡한 재구성을 수 분 안에 완료할 수 있습니다. 또한, 훈련 속도는 기존의 최첨단 NeRF 기반 모델보다 110배 더 빠르며, 30초 만에 효과적인 시각적 결과를 생성할 수 있습니다.



### Emotion Recognition with Facial Attention and Objective Activation Functions (https://arxiv.org/abs/2410.17740)
- **What's New**: 이 논문에서는 Facial Emotion Recognition (FER) 작업을 수행하기 위해 CNN 비전 기반 모델(VGGNet, ResNet, ResNetV2)에 채널 및 공간 주의 메커니즘(SEN-Net, ECA-Net, CBAM)을 도입하는 효과를 연구합니다. 주의 메커니즘이 기존 모델의 성능을 크게 향상시키고, 다양한 활성화 함수와 결합함으로써 성능을 더욱 증가시키는 방법을 보여줍니다.

- **Technical Details**: 본 연구에서는 VGGNet, ResNet, ResNetV2와 같은 세 가지 CNN 이미징 모델을 기반 모델로 사용하고, 이들에 주의 메커니즘을 추가하여 성능을 향상시킵니다. 각 모델은 노이즈에 대한 저항성과 소실 그래디언트 문제를 처리하는 능력이 뛰어나 FER 문제에 적합합니다. 또한, 다양한 활성화 함수(ReLU와 ELU)를 사용하여 아키텍처의 성능에 미치는 영향을 연구하고, YOLO 기반의 얼굴 감지 방법을 통해 배경 픽셀을 제거합니다.

- **Performance Highlights**: 실험 결과, ELU 활성화 함수는 CK+ 데이터셋에서 ResNet-50 모델에서 최고의 정확도를 기록했습니다. SEN-Net, ECA-Net 및 CBAM과 같은 주의 메커니즘의 도입이 모델의 성능을 개선하고, ECA-Net은 채널 간의 관계를 효율적으로 처리하여 FER 애플리케이션에 적합한 결과를 도출했습니다.



### YOLO-Vehicle-Pro: A Cloud-Edge Collaborative Framework for Object Detection in Autonomous Driving under Adverse Weather Conditions (https://arxiv.org/abs/2410.17734)
- **What's New**: 이 논문은 자율 주행 시스템에서의 객체 탐지를 개선하기 위해 YOLO-Vehicle 및 YOLO-Vehicle-Pro라는 두 가지 혁신적인 딥러닝 모델을 제안합니다. 이를 통해 저조도 환경에서도 높은 성능을 발휘할 수 있도록 하고, 클라우드-엣지 협업 객체 탐지 시스템을 설계하였습니다.

- **Technical Details**: YOLO-Vehicle은 멀티모달 융합(Integration of multimodal data) 기술을 활용하여 이미지와 텍스트 정보를 결합하여 객체를 탐지합니다. YOLO-Vehicle-Pro는 향상된 이미지 디헤이징(Image Dehazing) 알고리즘을 도입하여 가시성이 낮은 환경에서도 탐지 성능을 향상시킵니다. 이 시스템은 엣지 장치에서 모델을 배포하고 복잡한 상황에서 일부 컴퓨팅 작업을 클라우드로 오프로드 하는 방식을 채택합니다.

- **Performance Highlights**: KITTI 데이터셋에서 YOLO-Vehicle-v1s 모델은 92.1%의 정확도를 달성하고 226 FPS의 탐지 속도와 12ms의 추론 시간을 기록하여 자율 주행의 실시간 요구 사항을 충족했습니다. Foggy Cityscapes 데이터셋에서는 YOLO-Vehicle-Pro 모델이 82.3% mAP@50의 높은 정확도를 기록하면서도 43 FPS의 탐지 속도를 유지했습니다.



### YOLOv11: An Overview of the Key Architectural Enhancements (https://arxiv.org/abs/2410.17725)
- **What's New**: YOLOv11은 YOLO 시리즈의 최신 모델로, C3k2 블록 및 SPPF, C2PSA와 같은 혁신적인 아키텍처를 통해 성능을 획기적으로 개선하였습니다. 이 모델은 실시간 물체 탐지 기술의 중요한 진전을 나타내며, 다양한 컴퓨터 비전 (CV) 작업에서의 기능 확장을 지원합니다.

- **Technical Details**: YOLOv11은 C3k2 블록을 포함하는 새로운 백본 구조, SPPF 및 C2PSA 블록을 통하여 이미지 데이터에서 더 나은 특징 추출이 가능합니다. 이 아키텍처는 높은 연산 효율성과 함께 다양한 모델 크기를 지원하여 에지 디바이스로부터 고성능 컴퓨팅 환경까지 다중 응용 프로그램 요구에 적합합니다.

- **Performance Highlights**: YOLOv11은 이전 모델들과 비교하여 평균 정확도 (mAP)와 연산 효율성을 크게 향상시켰습니다. 특히 모델의 파라미터 수와 정확도 사이의 균형을 맞추는 데 중점을 두어 다양한 응용 분야에서 실시간 성능을 최적화했습니다.



### Surgical Scene Segmentation by Transformer With Asymmetric Feature Enhancemen (https://arxiv.org/abs/2410.17642)
- **What's New**: 이번 논문에서는 로봇 보조 복강경 수술의 이해를 위한 수술 장면 분할(surgical scene segmentation) 문제를 해결하기 위한 혁신적인 Transformer 기반 프레임워크인 TAFE(Transformer with Asymmetric Feature Enhancement)를 제안합니다. 이 프레임워크는 로컬 정보의 향상 및 멀티스케일 상호작용 주의(attention) 전략을 활용하여 향상된 피쳐 피라미드(feature pyramid)를 Transformer 인코더에 통합하는 데 초점을 맞추고 있습니다.

- **Technical Details**: TAFE는 두 가지 핵심 구성 요소인 멀티스케일 상호작용 주의(MIA) 브랜치와 비대칭 피쳐 향상(AFE) 모듈로 구성됩니다. MIA는 향상된 피쳐 피라미드를 Transformer 인코더의 임베딩에 주입하여 로컬 및 글로벌 특성 표현을 개선하고, AFE는 해부학(anatomy)과 도구(instruments)의 특정 특징을 모델링하여, 폴리곤 형태의 해부학적 특징에는 대칭 컨볼루션을, 바 또는 튜블 형태의 도구에는 비대칭 컨볼루션을 사용합니다.

- **Performance Highlights**: 제안된 방법은 Endoscapes2023에서 +4.0% mAP, EndoVis2018에서는 +11.3% mIoU의 성능 향상을 보여주며, 섬세한 구조 인식에서도 각각 19.2%와 4.3%의 개선을 달성하여 이전의 최첨단 기법들을 능가합니다.



### MIA-DPO: Multi-Image Augmented Direct Preference Optimization For Large Vision-Language Models (https://arxiv.org/abs/2410.17637)
Comments:
          Project URL: this https URL

- **What's New**: 본 논문에서는 Multi-Image Augmented Direct Preference Optimization (MIA-DPO) 방법을 소개하며, 이는 다중 이미지 입력을 효과적으로 처리할 수 있도록 설계되었습니다.

- **Technical Details**: MIA-DPO는 단일 이미지 데이터를 확장하여 그리드 콜라주(grid collage) 또는 픽-인-픽(pic-in-pic) 형식으로 관련이 없는 이미지를 조합하여 다중 이미지 훈련 데이터의 부족성을 해결합니다. 모델의 주의(attention) 값을 활용하여 잘못 선택된 반응을 확인하고 필터링합니다.

- **Performance Highlights**: MIA-DPO는 여러 다중 이미지 벤치마크에서 기존 방법보다 우수한 성능을 달성하며, LLaVA-v1.5에서는 평균 3.0%, 최근의 InternLM-XC2.5에서는 4.3%의 성능 향상을 이루었습니다. 또한 MIA-DPO는 단일 이미지에 대한 이해 능력에 미치는 영향이 최소화됩니다.



### Bridging the Gaps: Utilizing Unlabeled Face Recognition Datasets to Boost Semi-Supervised Facial Expression Recognition (https://arxiv.org/abs/2410.17622)
- **What's New**: 최근 Facial Expression Recognition (FER) 분야에서 준지도 학습(semi-supervised learning)을 활용한 방법들이 주목받고 있습니다. 이 연구는 주석(label)이 없는 대규모 Face Recognition (FR) 데이터를 활용하여 FER의 성능을 향상시키는 새로운 접근 방식을 제시합니다.

- **Technical Details**: 제안된 방법은 세 단계로 구성됩니다: 첫째, 주석이 없는 대규모 얼굴 이미지를 이용한 face reconstruction pre-training을 수행하여 얼굴의 기하학적 특징과 표현 영역을 학습합니다. 둘째, FaceMix 데이터 증강(data augmentation) 전략을 사용하여 제한된 레이블을 가진 FER 데이터셋에서 두 단계의 미세 조정(fine-tuning)을 진행합니다. 마지막으로, 주석이 없는 FER 이미지에서 반지도 학습을 수행합니다.

- **Performance Highlights**: RAF-DB, AffectNet 및 FERPlus 데이터셋에 대한 실험 결과, 기존 준지도 FER 방법들을 초월하는 성능을 보였으며, 특히 AffectNet에서는 5% and 25%의 훈련 세트를 사용하여 각각 64.02%, 88.23%의 정확도를 달성했습니다. 이는 완전지도 학습을 이용한 기존 최첨단 방법들과 비교할 만한 수준입니다.



### Towards Effective Data-Free Knowledge Distillation via Diverse Diffusion Augmentation (https://arxiv.org/abs/2410.17606)
- **What's New**: 본 연구에서는 데이터 없이 지식 증류(Data-free Knowledge Distillation, DFKD) 기법의 새로운 접근법인 다양한 확산 증강(Diverse Diffusion Augmentation, DDA)을 제안했습니다. 이는 기존의 데이터 합성 방식을 수정하고, 확산 모델을 활용하여 데이터 샘플의 다양성을 높이며, 훈련 데이터를 원활하게 처리할 수 있도록 돕습니다.

- **Technical Details**: DDA는 확산 모델(Diffusion Models)을 활용하여 합성된 데이터의 다양성을 극대화하고, 코사인 유사성(Cosine Similarity) 기법을 통해 품질이 낮은 이미지를 필터링하는 방법을 설명합니다. 이는 데이터의 다양한 스펙트럼을 생성하고, 지식 증류 과정에서의 오류를 줄이는 데 기여합니다.

- **Performance Highlights**: CIFAR-10, CIFAR-100, Tiny-ImageNet 데이터셋에서의 실험 결과, 제안된 DDA 방법은 최신 DFKD 방법들과 비교했을 때 우수한 성능을 보였으며, 다양한 teacher-student 네트워크 구성에서도 큰 이점을 보여주었습니다.



### PlantCamo: Plant Camouflage Detection (https://arxiv.org/abs/2410.17598)
- **What's New**: 이 논문은 식물 위장 탐지(Plant Camouflage Detection, PCD)에 대한 새로운 과제를 제시하고, 이를 해결하기 위한 새로운 데이터셋인 PlantCamo를 도입합니다.

- **Technical Details**: PlantCamo 데이터셋은 58개 객체 카테고리를 포함한 1,250장의 위장이 있는 식물 이미지를 포함하며, 새로운 프레임워크 PCNet을 통해 다중 스케일 글로벌 특징 강화(Multi-scale Global Feature Enhancement)와 정제(Multi-scale Feature Refinement)를 통해 성능을 개선합니다.

- **Performance Highlights**: PCNet은 20개 이상의 최신 COD 모델에 비해 뛰어난 성능을 보여주며, 기존 연구에서 식물 위장 탐지에 대한 중요한 발견을 제공합니다. 이 연구는 세밀한 COD 연구의 빈틈을 메우고, 지능 생태학 연구의 발전을 촉진할 것으로 기대됩니다.



### How to Continually Adapt Text-to-Image Diffusion Models for Flexible Customization? (https://arxiv.org/abs/2410.17594)
Comments:
          Accepted to NeurIPS2024

- **What's New**: 이번 연구에서는 Concept-Incremental text-to-image Diffusion Model (CIDM)을 제안하여 기존의 Custom diffusion models (CDMs)에서 발생하는 catastrophic forgetting과 concept neglect 문제를 해결합니다. 이를 통해 사용자 맞춤형 개념을 지속적으로 학습할 수 있는 새로운 접근법을 제공합니다.

- **Technical Details**: CIDM은 개념 통합 손실(concept consolidation loss)과 탄력적 가중치 집합(elastic weight aggregation) 모듈을 통해 오래된 개념의 catastrophic forgetting을 줄이면서 사용자 제공 조건에 따라 생성된 이미지의 맥락을 조절하는 전략을 개발하였습니다. 종합적으로, CIDM은 계층별 텍스트 임베딩을 활용하여 지역적 특징을 강화하고, 노이즈 추정(region noise estimation)을 통해 생성 이미지의 맥락을 제어합니다.

- **Performance Highlights**: 실험 결과, CIDM은 기존의 CDMs을 초월하는 성능을 보여주었습니다. 이 연구는 사용자가 제공합니다 주어진 조건을 활용하여, 다수의 개인화된 개념을 연속적으로 생성하고, 과거의 개념을 무시하지 않도록 효과적으로 학습할 수 있는 성능을 입증하였습니다.



### Double Banking on Knowledge: Customized Modulation and Prototypes for Multi-Modality Semi-supervised Medical Image Segmentation (https://arxiv.org/abs/2410.17565)
- **What's New**: 본 논문에서는 의료 영상 분할을 위한 Multi-modality (MM) semi-supervised learning (SSL) 접근법인 Double Bank Dual Consistency (DBDC)를 제안합니다. 이 방법은 두 가지 이상 모달리티를 지원하는 통합 네트워크로, 모달리티 특성과 모달리티 불변 특성을 동시에 학습할 수 있는 새로운 구조를 가지고 있습니다.

- **Technical Details**: 제안된 DBDC는 Modality-Level Modulation Bank (MLMB)와 Modality-Level Prototype Bank (MLPB)를 활용하여 각 모달리티에 대한 포괄적인 특성 학습을 수행합니다. 이러한 은행은 Modality Prototype Contrastive Learning (MPCL) 기법을 통해 업데이트되며, Modality Adaptive Weighting (MAW) 기법을 통해 각 모달리티에 따른 학습 가중치를 조정합니다. Dual Consistency (DC) 전략은 이미지와 특성 수준에서 일관성 유지를 강화합니다.

- **Performance Highlights**: 실험 결과, DBDC는 2개에서 4개 모달리티 분할 작업에서 기존의 최첨단 방법들보다 뛰어난 성능을 보였으며, 복잡한 모달리티 등록 및 여러 독립 네트워크의 필요성을 제거하여 실제적 가치와 일반화 능력을 향상시켰습니다.



### OVT-B: A New Large-Scale Benchmark for Open-Vocabulary Multi-Object Tracking (https://arxiv.org/abs/2410.17534)
Comments:
          15 pages, 6 figures, accepted at NeurIPS 2024 Dataset and Benchmark Track

- **What's New**: 이 논문은 새로운 대규모 Open-Vocabulary Multi-Object Tracking Benchmark인 OVT-B를 제안하며, 1,048개의 객체 카테고리와 1,973개의 비디오, 637,608개의 바운딩 박스 주석을 포함하고 있습니다. 이는 기존의 OV-TAO-val 데이터셋에 비해 월등히 큰 규모입니다.

- **Technical Details**: OVT-B는 다양한 객체 카테고리 및 주석 밀도가 뛰어나며, 특정 시나리오(예: out-of-view, 빠른 움직임, 상호 차폐 등)를 포함합니다. 새로운 기본 방법 OVTrack+가 제안되어 모션 정보와 외관 정보의 통합으로 성능을 향상시킵니다.

- **Performance Highlights**: 제안된 OVT-B와 OVTrack+의 실험 결과는 기존의 OVMOT 방법들에 비해 유용성과 효과성을 검증하였습니다. OVT-B는 OVMOT 연구 및 평가에 있어 새로운 플랫폼을 제공합니다.



### Diffusion Priors for Variational Likelihood Estimation and Image Denoising (https://arxiv.org/abs/2410.17521)
Comments:
          Accepted by NeurIPS2024 as Spotlight

- **What's New**: 이 논문은 확산 모형(difussion model)을 활용하여 실제 세계의 구조적이고 신호 의존적인 노이즈를 제거하는 기법을 제안합니다. 기존의 방법들은 단순한 노이즈 유형만을 고려하거나 근사된 후방 추정을 사용하는 데 한계가 있었습니다. 이를 극복하기 위해 논문에서는 적응형 가능도 추정(adaptive likelihood estimation)과 최대 후방 추정(MAP inference)을 도입하였습니다.

- **Technical Details**: 제안된 접근 방안은 독립적이며 비동일하게 분포된 가능도를 사용하여 실제 노이즈 모델을 구성하는 것입니다. 이 방법은 공간적으로 변동하는 노이즈의 특성을 모델링할 수 있게 하며, 변동성의 후방 추정을 변분 베이즈(variational Bayes)로 수행하여 가능도 함수를 정교하게 조정합니다. 또한 로컬 가우시안 합성을 통해 추정된 노이즈 분산을 보정하고, 각 단계별로 적응형으로 가능도를 업데이트하여 최종적으로 소음 제거된 이미지를 생성합니다.

- **Performance Highlights**: 다양한 실제 이미지 데이터셋에 대한 실험을 통해 제안된 방법이 기존의 비지도식 노이즈 제거 방법 및 확산 기반 방법들보다 우수한 성능을 나타내는 것을 확인했습니다. 특히, 저해상도 이미지로 사전 훈련된 확산 모델이 고해상도 이미지 처리에서 효과적인 로컬 확산 프라이어(local diffusion prior)를 제공하는 점을 강조하였습니다.



### PathMoCo: A Novel Framework to Improve Feature Embedding in Self-supervised Contrastive Learning for Histopathological Images (https://arxiv.org/abs/2410.17514)
- **What's New**: 본 논문에서는 역사적 이미지 분석에 특화된 새로운 이미지 증가 방법인 stain reconstruction augmentation(SRA)를 제안합니다. SRA는 MoCo v3 모델과 통합되어 PathMoCo라는 새로운 모델을 형성하며, 이는 다양한 다운스트림 작업에서 MoCo v3를 항상 초월하는 성능을 보여줍니다.

- **Technical Details**: SRA는 H&E 염색된 조직 병리 이미지에 적용되며, RGB 이미지를 Hematoxylin 및 Eosin 채널 이미지로 분해한 후, 각 채널 이미지의 최댓값으로 나눈 정규화를 통해 증강됩니다. PathMoCo는 이러한 SRA를 MoCo v3에 통합하여 일반적인 contrastive loss 외에도 추가적인 contrastive loss 항을 도입합니다.

- **Performance Highlights**: PathMoCo는 여러 공개 데이터 세트에서 실험을 진행하여 MoCo v3에 비해 항상 우수한 성능을 기록했습니다. 또한, 상당히 더 큰 조직 병리 데이터 세트로 미리 학습된 다른 기초 모델들과 동등하거나 그 이상의 성능을 보여줍니다.



### HCDN: A Change Detection Network for Construction Housekeeping Using Feature Fusion and Large Vision Models (https://arxiv.org/abs/2410.17513)
- **What's New**: 본 논문은 건설 현장에서의 안전을 향상시키기 위한 새로운 기술적 접근법을 제안합니다. 특히, 청소 상태(poor housekeeping)를 컴퓨터 비전(computer vision) 기술을 통해 감지하는 데 중점을 두고 있습니다.

- **Technical Details**: HCDN(Housekeeping Change Detection Network)이라는 고급 변화 감지(neural network)를 소개하며, 이 네트워크는 feature fusion module을 통합하여 대규모 비전 모델(large vision model)과 결합되어 있습니다. 또한, 건설 현장의 청소 관리에 초점을 맞춘 새로운 변화 감지 데이터셋인 Housekeeping-CCD를 생성하였습니다.

- **Performance Highlights**: 기존의 방법들과 비교하여 성능이 크게 향상되었으며, 건설 현장에서의 청소 상태 및 안전성을 개선하기 위한 효과적인 도구로 자리잡을 것으로 기대됩니다. 다수의 연구자들이 활용할 수 있도록 소스 코드와 훈련된 모델을 공유합니다.



### PLGS: Robust Panoptic Lifting with 3D Gaussian Splatting (https://arxiv.org/abs/2410.17505)
- **What's New**: 본 논문에서는 기존의 NeRF(Neural Radiance Field) 기반 판옵틱 리프팅(panoptic lifting) 방법의 한계를 극복하기 위해 3D Gaussian Splatting(3DGS) 기술을 이용하여 효율성과 정확성을 동시에 개선하는 PLGS(Panel Gaussian Splatting)이라는 새로운 방법을 제안합니다. PLGS는 노이즈가 있는 2D 세분화 마스크를 기반으로 일관된 3D 세분화 마스크를 생성합니다.

- **Technical Details**: PLGS는 구조화된 판옵틱 인식 3D Gaussian 모델을 구축하여 매끄러움(smoothness)을 포함하고 효과적인 노이즈 감소 전략을 설계합니다. 본 방법은 세멘틱 필드를 구성할 때, 동작 구조(structure from motion)를 초기화하는 대신 신뢰할 수 있는 세멘틱 앵커 포인트를 사용하여 3D Gaussian을 초기화하고, 이 앵커 포인트를 학습 과정에서 부드러운 정규화(smooth regularization)로 활용합니다.

- **Performance Highlights**: 다양한 벤치마크 실험을 통해 PLGS가 이전의 최첨단 방법들보다 판옵틱 세분화(panoptic segmentation) 품질과 속도 면에서 우수함을 입증했습니다. 특히 PLGS는 훈련 시간 및 렌더링 속도에서 상당한 개선을 보여 줍니다.



### Unsupervised Domain Adaptation for Action Recognition via Self-Ensembling and Conditional Embedding Alignmen (https://arxiv.org/abs/2410.17489)
Comments:
          This work has been accepted to the Proceedings of the IEEE International Conference on Data Mining, 2024

- **What's New**: 최근 딥 러닝 기반의 착용형 인간 동작 인식(wHAR) 기술이 복잡한 동작의 캡처 및 분류에서 개선되었습니다. 그러나 전문가 주석 부족과 사용자 변동성으로 인해 채택은 여전히 제한적입니다. 이를 해결하기 위해, 새로운 조인트 최적화 아키텍처인 μDAR을 제안하며, 이는 데이터의 일반화 능력을 개선하고 보다 강력한 목표 레이블 생성을 목적으로 합니다.

- **Technical Details**: μDAR은 (i) 일관성 정규화기, (ii) 시간 집계 기반의 강력한 pseudo-label 생성을 위한 앙상블, (iii) 조건 분포 정렬 모듈 등 세 가지 기능으로 구성됩니다. 이러한 구성 요소들은 각기 다른 변화를 줄이기 위해 결합되어 작동하여 소스 도메인과 타겟 도메인 간의 특징 공간에서의 불일치를 줄이기 위해 kernel-based class-wise conditional maximum mean discrepancy (kCMMD)를 최소화합니다.

- **Performance Highlights**: μDAR은 네 개의 벤치마크 wHAR 데이터셋에서 6개의 최신 UDA 방법에 비해 약 4-12%의 매크로-F1 점수 향상을 달성했습니다. 이는 적은 소스 도메인 샘플로도 강력한 일반화를 가능하게 하며, 타겟 샘플에서도 일관된 pseudo-label 생성을 보장합니다.



### Which Client is Reliable?: A Reliable and Personalized Prompt-based Federated Learning for Medical Image Question Answering (https://arxiv.org/abs/2410.17484)
- **What's New**: 본 연구에서는 의료_visual_question_answering(VQA) 모델을 위한 개인화된 연합 학습(personalized federated learning, pFL) 방법을 제안합니다. 이 방법은 환자의 개인 정보 보호 문제를 다루며, 다양한 의료 데이터를 처리할 수 있는 새로운 기술적 접근법을 제시합니다.

- **Technical Details**: 우리의 pFL 시스템은 Transformer 아키텍처에 학습 가능한 프롬프트(learnable prompts)를 도입하여 효율적으로 의료 데이터셋에서 모델을 학습합니다. 그리고 각 클라이언트의 예측 불확실성을 정량화하기 위해 Dempster-Shafer 증거이론(DST)을 통합한 신뢰할 수 있는 클라이언트 VQA 모델을 소개합니다. 또한, 최대 우도 추정(maximum likelihood estimation)을 사용하여 정확성과 불확실성의 균형을 맞추는 새로운 클라이언트 간 통신 메커니즘을 제안합니다.

- **Performance Highlights**: 다양한 의료 VQA 데이터셋에 대한 광범위한 정성적 및 정량적 실험을 통해 제안한 방법이 개인화된 클라이언트 간 정보를 효율적으로 집계할 수 있음을 입증하였습니다. 이러한 접근법은 특정 의료 이미지나 증상에 대한 의사의 조언을 보다 정확하게 제공할 수 있습니다.



### LongVU: Spatiotemporal Adaptive Compression for Long Video-Language Understanding (https://arxiv.org/abs/2410.17434)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 LongVU라는 새로운 스페이시-템포럴(adaptive spatiotemporal) 압축 메커니즘을 제안하여 긴 비디오의 특성을 효과적으로 처리할 수 있게 합니다. LongVU는 비디오 토큰의 수를 줄이면서도 시각적 세부 정보를 보존하는 데 중점을 두고 있습니다.

- **Technical Details**: LongVU는 DINOv2 특징을 활용하여 높은 유사성을 보이는 중복 프레임을 제거하고, 텍스트 기반 교차 모달 쿼리를 통해 선택적으로 프레임 피처를 줄이며, 각 프레임의 temporal 의존성에 따라 공간적 토큰을 줄입니다. 이 메커니즘은 평균적으로 프레임당 2개의 토큰을 사용하여 1fps로 샘플링된 비디오 입력을 처리할 수 있습니다.

- **Performance Highlights**: LongVU는 VideoMME, MLVU 등 다양한 비디오 이해 벤치마크에서 기존 방법들을 능가하였으며, LLaVA-OneVision과 비교할 때 약 5%의 평균 정확도 향상을 기록했습니다. 또한, 경량 LLM에 기반한 LongVU는 최신 소형 비디오 LLM보다 3.4% 향상된 성능을 보여줍니다.



### SigCLR: Sigmoid Contrastive Learning of Visual Representations (https://arxiv.org/abs/2410.17427)
Comments:
          Neurips 2024 SSL Workshop

- **What's New**: 새로운 연구에서 SigCLR을 제안하며, 이는 시그모이드 손실(sigmoid loss)을 이용하여 이미지 표현을 학습하는 새로운 방법입니다. 이 방법은 SimCLR에서 사용된 크로스 엔트로피 손실(cross-entropy loss)과는 달리, 쌍(pair)별로 작동하고 전역 뷰(global view)가 필요하지 않습니다.

- **Technical Details**: SigCLR은 데이터 인스턴스의 여러 시각적 관점을 서로 비교하여 시그모이드 대조 손실(sigmoid contrastive loss)을 최대화함으로써 표현을 학습합니다. 손실 함수는 모든 이미지 쌍이 이진 분류 문제로 바뀌며, 긍정 관계인 쌍에는 긍정 레이블이, 그 외 쌍에는 부정 레이블이 할당됩니다. 시그모이드 기반 손실은 각 이미지 쌍에 독립적으로 작용하여 전역 정규화(global normalization) 요소를 계산할 필요가 없습니다.

- **Performance Highlights**: SigCLR은 CIFAR-10, CIFAR-100, Tiny ImageNet에서의 실험 결과에서 SimCLR을 초과하며, 최고의 성능 경쟁에서 다른 목표와 비교해도 경쟁력을 갖추고 있습니다. 또한, SigCLR은 소규모 배치 크기(small batch sizes)에서도 뛰어난 성능을 발휘하였습니다.



### Denoise-I2W: Mapping Images to Denoising Words for Accurate Zero-Shot Composed Image Retrieva (https://arxiv.org/abs/2410.17393)
Comments:
          This work was submitted to IJCAI 2024, with a score of weak accept and borderline accept

- **What's New**: 본 논문에서는 Zero-Shot Composed Image Retrieval (ZS-CIR)에서의 정확한 이미지 매핑을 위한 새로운 접근법인 Denoise-I2W를 제안합니다. 이 방법은 의도와 관련 없는 시각 정보를 제거하여 정확한 이미지 검색 성능을 향상시킵니다.

- **Technical Details**: Denoise-I2W는 pseudo-triplet을 구성하여 변별력 있는 매핑을 수행합니다. pseudo-reference 이미지, pseudo-manipulation 텍스트 및 target 이미지를 자동적으로 생성하여, 사전 훈련 매핑 네트워크를 통해 매핑합니다. 이 방법은 model-agnostic하며 추가적인 주석 데이터 없이 기존 ZS-CIR 모델과 통합이 가능합니다.

- **Performance Highlights**: Denoise-I2W를 기존 모델에 통합한 결과, 성능이 1.45%에서 4.17% 개선되었으며, 새로운 최첨단 결과를 달성했습니다. 본 방법은 네 가지 벤치마크 데이터셋에서 강력한 일반화 능력을 보여주는 것으로 나타났습니다.



### Image-aware Evaluation of Generated Medical Reports (https://arxiv.org/abs/2410.17357)
- **What's New**: 이 논문에서는 X-ray 이미지로부터 자동 의료 보고서 생성을 위한 새로운 평가 메트릭, VLScore를 제안합니다. 기존 평가 방법들의 한계를 극복하고자 하며, 텍스트 유사성만을 고려하거나 단일한 임상적 측면에만 집중하는 문제를 해결합니다.

- **Technical Details**: VLScore는 연구자가 공인한 세트에서 오류를 표시한 쌍의 보고서를 평가하여 방사선 전문의의 판단과 여러 강력한 상관관계를 보여줍니다. 이 모델은 이미지와 보고서 간의 유사성을 측정하여 동시 시각-텍스트 공간 내에서의 관계를 기반으로 합니다. 새로운 데이터 세트 또한 제공하여 중요한 수정과 사소한 수정을 구별하도록 설계되었습니다.

- **Performance Highlights**: VLScore는 기존 메트릭들과 비교할 때 방사선 전문의의 판단과 가장 높은 일치를 보이며, 중요한 세부 사항에 민감하면서도 사소한 오류에 대해서는 강인성을 유지합니다.



### Offline Evaluation of Set-Based Text-to-Image Generation (https://arxiv.org/abs/2410.17331)
- **What's New**: 이 논문은 ideation 과정에서 텍스트-이미지(Text-to-Image, TTI) 시스템이 어떻게 지원되는지를 평가하기 위한 새로운 메트릭스를 제안합니다. 기존의 Fréchet Inception Distance(FID)와 같은 분포 유사성 메트릭이 아닌, 사용자 행동 모델을 기반으로 한 오프라인 평가 메트릭스를 개발하였습니다.

- **Technical Details**: 제안된 메트릭스는 사용자가 생성된 이미지 세트를 탐색하고 상호작용하는 방식을 명시적으로 모델링합니다. 이 모델은 정보 검색(Information Retrieval, IR) 기술을 활용하여 grid 레이아웃 및 이미지 다양성을 고려하여 평가합니다. 특히, 상대 순위(expected reciprocal rank)와 순위 편향 정밀도(rank-biased precision)를 적용하였습니다.

- **Performance Highlights**: 세 가지 데이터셋(MS-COCO 캡션, localized narratives 등)을 통해 실험한 결과, 사용자의 브라우징 방식과 이미지의 시각적 두드러짐을 모델링함으로써 인간의 선호도와의 일치성을 높임을 보여주었습니다.



### Automated Quality Control System for Canned Tuna Production using Artificial Vision (https://arxiv.org/abs/2410.17275)
Comments:
          6 pages, 12 figures

- **What's New**: 이 논문은 톤(Tuna) 금속 캔의 결함을 감지하고 분류하기 위한 자동화 제어 시스템을 구현한 내용을 담고 있습니다.

- **Technical Details**: 이 시스템은 컨베이어 벨트(conveyor belt)와 포토일렉트릭 센서(photoelectric sensor)에 의해 작동되는 카메라를 사용하여 시각적 인식을 수행합니다. 로봇 팔(robotic arm)은 금속 캔의 상태에 따라 분류합니다. IoT 시스템을 통해 Industry 4.0 통합이 이루어지며, Mosquitto, Node-RED, InfluxDB, Grafana 등이 사용되었습니다. 결함 감지를 위해 YOLOv5 모델이 사용되며, GPU에서 Google Colab을 통해 학습하여 라벨의 OCR 텍스트 감지를 가능하게 합니다.

- **Performance Highlights**: 결과는 실시간 문제 식별의 효율성, 자원 최적화, 품질 제품 제공을 보여줍니다. 또한 비전 시스템은 품질 관리 작업의 자율성을 기여하여 운영자가 회사 내에서 다른 기능을 수행할 수 있도록 합니다.



### Zero-Shot Vision-and-Language Navigation with Collision Mitigation in Continuous Environmen (https://arxiv.org/abs/2410.17267)
- **What's New**: 본 논문은 Collision Mitigation (VLN-CM) 기능을 갖춘 Zero-Shot Vision-and-Language Navigation을 제안합니다. 기존의 VLN과 달리, 사용자가 훈련 데이터 없이도 자연어 지침을 이용하여 연속적인 환경에서 탐색할 수 있도록 합니다.

- **Technical Details**: VLN-CM은 Attention Spot Predictor (ASP), View Selector (VS), Progress Monitor (PM) 및 Open Map Predictor (OMP)로 구성되어 있습니다. ASP는 Large Language Model (e.g. ChatGPT)을 활용하여 탐색 지침을 특정 주목 지점으로 분해하고, VS는 CLIP 유사도를 사용하여 30도 간격의 파노라마 이미지 중 주목 지점을 포함한 이미지를 선택합니다. PM은 규칙 기반 접근 방식을 사용하여 다음 주목 지점을 결정하고, OMP는 깊이 정보를 사용해 점유 마스크를 예측하여 안전한 이동 거리를 선택합니다.

- **Performance Highlights**: VLN-CM은 VLN-CE 검증 데이터에서 여러 기준선 방법보다 뛰어난 성능을 나타냈으며, OMP는 에이전트의 충돌을 효과적으로 완화하는 데 기여했습니다. SR(성공률)을 0.11로 달성한 반면, Random Agent와 Hand-Crafted Agent는 각각 0.03의 SR을 기록했습니다.



### Federated brain tumor segmentation: an extensive benchmark (https://arxiv.org/abs/2410.17265)
- **What's New**: 최근 연합 학습(federated learning)이 의료 이미지 분석 분야에서 중요성이 대두되고 있습니다. 이는 다수 센터의 데이터를 집계하면서 개인 정보를 보호할 수 있는 특성을 가지고 있습니다.

- **Technical Details**: 다양한 연합 학습 알고리즘을 글로벌(하나의 최종 모델), 개인화(각 기관당 하나의 모델), 하이브리드(기관 클러스터당 하나의 모델)로 분류하고, 2022년 연합 뇌종양 세분화(Federated Brain Tumor Segmentation) 데이터셋에 대한 적용 가능성을 연구했습니다. 일관된 정규화 방법(FedAvg)이 우수한 성능을 보여주지만, 각 카테고리에서 일부 방법이 성능을 약간 향상시킬 수 있음을 확인했습니다.

- **Performance Highlights**: 모델의 편향을 줄이고, 풀링된 데이터셋을 기관 간에 분배하는 다양한 방법을 통한 연합 학습의 동작 방식에 대한 깊은 이해를 제공했습니다. 이 연구는 IID(Independent and Identical Distributed) 설정과 제한된 데이터 설정을 통해 수행됩니다.



### Audio-Driven Emotional 3D Talking-Head Generation (https://arxiv.org/abs/2410.17262)
- **What's New**: 이 논문에서는 높은 충실도의 오디오 기반 비디오 초상화를 정확한 감정 표현으로 합성하는 새로운 시스템 EmoGene을 제안합니다. 저자들은 감정 표현의 생성이 현실적인 대화 머리 생성의 중요한 측면으로 남아있음을 강조하고, 이를 해결하기 위한 세 가지 모듈을 도입합니다.

- **Technical Details**: EmoGene 시스템은 3단계로 구성됩니다: 1) audio-to-motion: VAE 모델을 사용하여 오디오 특징을 중립적 얼굴 랜드마크로 변환합니다. 2) motion-to-emotion: 랜드마크 변형 모델을 통해 중립 랜드마크를 감정 랜드마크로 변환합니다. 3) emotion-to-video: NeRF 모델을 사용하여 높은 충실도의 감정 대화 머리 비디오를 렌더링합니다. 이 시스템은 감정 인식 및 표현의 정확성을 높입니다.

- **Performance Highlights**: EmoGene은 기존 방법들보다 감정 표현 생성을 더 정확하게 수행하며, 높은 그래픽 충실도와 사실성을 유지합니다. 실험 결과, EmoGene은 다양한 감정 표현 및 고충실도의 대화 머리 비디오 생성에서 이전의 작업보다 뛰어난 성능을 보여주었습니다.



### FIPER: Generalizable Factorized Fields for Joint Image Compression and Super-Resolution (https://arxiv.org/abs/2410.18083)
Comments:
          Project page: this https URL

- **What's New**: 본 연구에서는 Super-Resolution (SR)과 이미지 압축을 통합적으로 다룬 **Factorized Fields**라는 새로운 표현 방식을 제안합니다. 이 방식은 두 작업 간의 공유된 원칙에 기초합니다.

- **Technical Details**: 제안된 모델은 Coefficient Backbone과 Basis Swin Transformer를 포함하여 멀티 스케일 시각적 특징과 구조 요소를 정확하게 캡처하기 위해 basis-coefficient 분해 방식을 활용합니다. SR 과정에서 학습된 모듈의 강력한 정보 복원 기능을 이미지 압축 파이프라인에 우선 정보로 사용하여 압축 효율성과 세부 재구성을 향상합니다.

- **Performance Highlights**: 광범위한 실험 결과, 우리의 통합 표현 방식은 Super-Resolution에서 204.4%의 PSNR 개선 및 이미지 압축에서 9.35%의 BD-rate 감소를 달성하며 최신 성능을 입증합니다.



### SPIRE: Synergistic Planning, Imitation, and Reinforcement Learning for Long-Horizon Manipulation (https://arxiv.org/abs/2410.18065)
Comments:
          Conference on Robot Learning (CoRL) 2024

- **What's New**: SPIRE 시스템이 제안되었습니다. 이 시스템은 작업을 작고 학습 가능한 하위 문제로 분해하고, 모방 학습(imitation learning)과 강화 학습(reinforcement learning)을 결합하여 두 기법의 장점을 극대화합니다.

- **Technical Details**: SPIRE는 Task and Motion Planning (TAMP)을 활용하여 각 작업을 지역적인 조작(segment)으로 분해한 다음, 이 조작들을 모방 학습으로 먼저 학습하고, 후에 강화 학습을 통해 조정(finetuning)합니다. 또한, sparse rewards로부터 RL 기반의 조정을 가능하게 하는 여러 가지 전략이 개발되었습니다.

- **Performance Highlights**: SPIRE는 9개의 도전적인 조작 작업에서 평균 성공률 87.8%를 기록하며 이전의 접근 방식에 비해 35%에서 50% 더 높은 성능을 보였습니다. 이 시스템은 인간 시연 데이터의 필요성을 6배 줄이고, 작업 완료 효율성을 두 배로 향상시킵니다.



### A Wavelet Diffusion GAN for Image Super-Resolution (https://arxiv.org/abs/2410.17966)
Comments:
          The paper has been accepted at Italian Workshop on Neural Networks (WIRN) 2024

- **What's New**: 본 연구에서는 시간이 중요한 애플리케이션에서 발생하는 diffusion 모델의 단점을 극복하기 위해 Single-Image Super-Resolution (SISR)을 위한 wavelet 기반의 conditional Diffusion GAN 방식을 제안합니다.

- **Technical Details**: 제안된 방법은 reverse diffusion process에 필요한 timesteps을 줄이기 위해 diffusion GAN 패러다임을 적용하고, Discrete Wavelet Transform (DWT)를 사용하여 차원 축소를 통해 훈련 및 추론 시간을 크게 단축합니다.

- **Performance Highlights**: CelebA-HQ 데이터셋에서 수행된 실험 검증 결과, 제안된 방법은 다른 최신 기술들보다 우수한 성능을 보이며, 높은 화질의 출력을 보장하면서도 시간에 민감한 애플리케이션에서의 inherent drawbacks을 극복하는 데 성공하였습니다.



### Medical Imaging Complexity and its Effects on GAN Performanc (https://arxiv.org/abs/2410.17959)
Comments:
          Accepted to ACCV, Workshop on Generative AI for Synthetic Medical Data

- **What's New**: 이 연구는 의료 이미지 합성을 위한 GAN(Generative Adversarial Network) 훈련에서 샘플 데이터셋의 크기와 생성된 이미지의 충실도 간의 관계를 정량적으로 규명하였습니다.

- **Technical Details**: 이 연구에서는 delentropy라는 이미지 복잡도 측정 기법을 활용하여 훈련 이미지의 복잡도 분포가 GAN 훈련의 난이도를 어떻게 측정할 수 있는지를 탐구했습니다. StyleGAN 3와 SPADE-GAN을 사용하여 다양한 의료 이미징 데이터셋을 기준으로 실험을 진행했습니다.

- **Performance Highlights**: 두 GAN 모두 훈련 세트 크기가 증가함에 따라 일반적인 성능이 개선되었으나, 이미지의 복잡성이 증가할 경우 성능이 저하되는 경향을 보였습니다.



### R-CoT: Reverse Chain-of-Thought Problem Generation for Geometric Reasoning in Large Multimodal Models (https://arxiv.org/abs/2410.17885)
- **What's New**: 이 논문은 기존의 Large Multimodal Models (LMMs)이 겪고 있는 수학 기하학적 추론의 어려움을 해결하기 위해 Reverse Chain-of-Thought (R-CoT) 기하학 문제 생성 파이프라인을 제안합니다. 이 방법은 고품질의 이미지-텍스트 데이터 생성을 통해 기하적 문제의 답변 정확성을 향상시킵니다.

- **Technical Details**: R-CoT 파이프라인은 두 단계로 구성되며, 첫 번째 단계에서는 GeoChain을 통해 고충실도(High-fidelity) 기하학 이미지를 생성하고, 두 번째 단계에서는 Reverse A&Q 방법을 사용하여 설명 기반의 단계적 추론을 통해 질문을 생성합니다. 이로 인해 각 모델에서 새로운 성능 기록을 달성하였고, 2B, 7B, 8B 설정에서 모두 향상된 결과를 보여줍니다.

- **Performance Highlights**: R-CoT-8B는 MathVista와 GeoQA 데이터셋에서 각각 16.6% 및 9.2%의 성과 향상을 보이며, 유료 모델 GPT-4o보다 평균 13% 더 뛰어난 성능을 기록했습니다. 이는 여러 LMM 기초 모델에서 일관된 성능 향상을 나타냅니다.



### CASCRNet: An Atrous Spatial Pyramid Pooling and Shared Channel Residual based Network for Capsule Endoscopy (https://arxiv.org/abs/2410.17863)
Comments:
          8 pages, 4 figures

- **What's New**: 이 논문은 Capsule Vision Challenge 2024에 대한 MISAHUB의 작업을 요약하고, 복잡성과 불균형이 있는 데이터셋을 다루기 위해 CASCRNet이라는 새로운 모델을 제안합니다.

- **Technical Details**: CASCRNet은 Shared Channel Residual (SCR) 블록과 Atrous Spatial Pyramid Pooling (ASPP) 블록을 사용하여 파라미터 효율성과 성능을 개선합니다. 이 모델은 dilated convolutions를 적용하여 이미지 처리 성능을 향상시켰으며, LeakyReLU 활성화 함수를 사용하고 Adam optimizer로 훈련되었습니다. 또한 focal loss를 사용하여 불균형 데이터셋 문제를 해결하였습니다.

- **Performance Highlights**: CASCRNet은 78.5%의 F1 Score와 98.3%의 Mean AUC로 성능 평가에서 우수한 결과를 나타냈으며, 이는 Compact architecture에서 실현되었습니다. 참고로, CASCRNet은 모든 평가 메트릭에서 다른 모델과 비교했을 때 우수한 성능을 보였습니다.



### Att2CPC: Attention-Guided Lossy Attribute Compression of Point Clouds (https://arxiv.org/abs/2410.17823)
- **What's New**: 이 논문에서는 Transformer를 기반으로 한 포인트 클라우드 속성 압축(PCAC) 프레임워크인 Att2CPC를 처음으로 제안합니다. 이 방법은 패턴 인식을 위한 효과적인 External Cross Attention (ECA) 메커니즘을 포함하며, 이는 기하학적 정보와 속성의 관계를 효과적으로 활용합니다.

- **Technical Details**: 본 연구는 포인트 기반 오토인코더 파이프라인을采用하였으며, 기하학 기반의 다중 스케일 표현을 통해 강력한 포인트 클라우드 속성 압축을 실현합니다. 인코딩 과정에서는 여러 다운샘플링을 통해 지역 속성 패턴을 최대한 활용하고, 디코딩 과정에서는 다중 스케일 표현과 제로 패딩 업샘플링 기법을 사용하여 포인트 클라우드의 속성을 점차적으로 재구성합니다.

- **Performance Highlights**: 실험 결과, 제안하는 방법은 Deep-PCAC와 같은 기존의 학습 기반 방법들과 비교했을 때, Y 채널과 YUV 채널에서 각각 평균 1.15 dB 및 2.13 dB의 BD-PSNR 개선 효과를 보였습니다. 따라서 Att2CPC는 기존 방법에 비해 우수한 압축 효율성을 보여줍니다.



### Deep Learning for Active Region Classification: A Systematic Study from Convolutional Neural Networks to Vision Transformers (https://arxiv.org/abs/2410.17816)
- **What's New**: 본 연구에서는 태양 활동 지역의 자동 분류에 대한 심층 학습(deep learning) 기법의 적용 결과를 제시합니다. 특히, Mount Wilson 분류 체계를 기반으로 하는 활동 지역 컷아웃의 분류에 대해 논의하며, 최신 이미지 분류 아키텍처에서의 성능을 평가합니다.

- **Technical Details**: 이 연구는 Convolutional Neural Networks (CNNs)와 Vision Transformers (ViTs)와 같은 최신 심층 학습 아키텍처를 분석하여 활동 지역 분류 작업에 대한 성능을 보고합니다. SOLAR-STORM1 데이터셋을 사용하여, 정량적 평가를 위해 2D convolution을 통한 모델 훈련에 집중하였습니다. 다섯 번의 교차 검증을 통해 모델의 신뢰성을 확보했습니다.

- **Performance Highlights**: 본 연구에서 제안된 메소드는 태양 활동 지역의 자동 분류의 정확성과 신속성을 크게 향상시킬 수 있음을 보여줍니다. 최신의 데이터 보강(data augmentation) 기술과 결합되어, 과적합(overfitting)을 피하면서 일반화된 성능 향상을 이끌어냈습니다.



### Learning Lossless Compression for High Bit-Depth Volumetric Medical Imag (https://arxiv.org/abs/2410.17814)
Comments:
          13 pages

- **What's New**: 이 논문에서는 고정밀 의료 볼륨 이미지 압축을 위한 Bit-Division 기반 Lossless Volumetric Image Compression (BD-LVIC) 프레임워크를 소개합니다. 이 프레임워크는 고비트 심도 데이터를 두 개의 낮은 비트 심도 세그먼트로 분할하여 압축 효율성을 높입니다.

- **Technical Details**: BD-LVIC 프레임워크는 Most Significant Bit-Volume (MSBV)와 Least Significant Bit-Volume (LSBV)로 고비트 심도 볼륨을 나눕니다. MSBV는 볼륨의 주요 정보가 포함되어 있으며, 기존의 코드 방식을 사용하여 효율적인 압축을 이룹니다. LSBV는 복잡한 텍스처 정보를 포함하고 있으며, Transformer 기반 Feature Alignment Module (TFAM)과 Parallel Autoregressive Coding Module (PACM)을 사용하여 공간적 맥락과 특징을 정확하게 정렬 및 융합합니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, BD-LVIC 프레임워크는 여러 의료 이미지 데이터세트에서 최신 손실 없는 압축 성능과 빠른 코딩 속도를 달성하여 의료 이미징 응용 프로그램에서 실용적인 가치를 보여주고 있습니다.



### PGDiffSeg: Prior-Guided Denoising Diffusion Model with Parameter-Shared Attention for Breast Cancer Segmentation (https://arxiv.org/abs/2410.17812)
- **What's New**: 본 논문에서는 유방암 의학적 이미지 분할을 위해 새로운 모델 PGDiffSeg(파라미터 공유 주의 메커니즘을 가진 사전 가이드 확산 제거 모델)를 제안합니다. 이 모델은 가우시안 노이즈로부터 영향을 받은 영역을 정확하게 복구하는데 초점을 맞추고 있습니다.

- **Technical Details**: PGDiffSeg는 두 개의 파이프라인(노이즈 처리 및 의미 정보 처리)을 설계하고, 다층에서 이를 통합하는 파라미터 공유 주의 모듈(PSA)을 제안하여 분할 성능을 향상시킵니다. 이 통합을 통해 다중 레벨에서 의미 세부정보를 포함할 수 있고, 사전 지식을 이용한 가이드 전략을 도입하여 모델의 종양 위치 탐지 능력을 향상시킵니다.

- **Performance Highlights**: 모델의 효과성은 다수의 실험을 통해 입증되었으며, 기존의 최신 방법들보다 우수한 성능을 보였습니다. PGDiffSeg는 유방암 이미지 분할 연구에 적합한 유연한 확산 제거 방법으로 자리매김하고 있습니다.



### Scaling Robot Policy Learning via Zero-Shot Labeling with Foundation Models (https://arxiv.org/abs/2410.17772)
Comments:
          Project Website at this https URL

- **What's New**: NILS(Natural language Instruction Labeling for Scalability)는 인간의 개입 없이 긴 로봇 데이터에서 자연어 지시를 자동으로 라벨링하는 새로운 시스템입니다. 이는 비용과 데이터 품질 문제를 해결합니다.

- **Technical Details**: NILS는 세 단계로 구성됩니다: (1) 장면에서 객체 식별, (2) 객체 중심의 장면 주석 생성, (3) 키 상태 탐지 및 라벨 생성. NILS는 사전 훈련된 비전-언어 모델(Vision-Language Models) 및 대형 언어 모델(Large Language Model)을 사용하여 로봇 행동 데이터를 효과적으로 라벨링합니다.

- **Performance Highlights**: NILS는 115,000개 경로를 라벨링하여 430시간 이상의 로봇 데이터를 처리할 수 있으며, 다른 기존 시스템보다 더 높은 품질의 결과를 제공합니다. 이를 통해 언어 조건 부조작 정책을 훈련할 수 있는 구조화된 데이터셋이 생성됩니다.



### New Insight in Cervical Cancer Diagnosis Using Convolution Neural Network Architectur (https://arxiv.org/abs/2410.17735)
- **What's New**: 이 연구는 자궁경부암(Pap smear) 이미지를 분류하기 위한 CNN(Convolutional Neural Network) 모델에서 최적의 옵티마이저(optimizer)를 선택하는 것이 중요하다는 점을 강조합니다. 여러가지 최적화 기법을 사용하여 Pap smear 이미지를 분류하였습니다.

- **Technical Details**: 이 연구에서는 Stochastic Gradient Descent (SGD), RMSprop, Adam, AdaGrad, AdaDelta, Adamax, Nadam 옵티마이저를 적용하였으며, Resnet-18, Resnet-34, VGG-16의 CNN 아키텍처를 사용했습니다. 각 아키텍처는 전이 학습(transfer learning) 모델을 활용하였습니다.

- **Performance Highlights**: Adamax 옵티마이저는 VGG-16아키텍처에서 72.8%의 정확도를, Resnet-18에서 66.8%의 정확도를 기록하였습니다. Resnet-34는 54.0%로 Nadam보다 0.034% 낮았습니다. 전반적으로 Adamax는 자궁경부암 분류를 위한 CNN에 적합한 옵티마이저로 평가되었습니다.



### Continual Learning on a Data D (https://arxiv.org/abs/2410.17715)
Comments:
          18 pages, 6 figures

- **What's New**: 본 논문은 중요 샘플에서 학습할 때 클래스 증가 학습(Class-Incremental Learning, CIL) 모델의 성능이 증가함을 보여주며, 서로 다른 coreset 선택 방법과 여러 연속 학습 모델의 상호작용을 탐구합니다.

- **Technical Details**: 연구진은 다양한 coreset 선택 기술을 통해 선택된 샘플에서 학습함으로써 CIL 모델의 학습-잊기 동역학을 조사하고, 모델의 안정성과 유연성 간의 균형을 향상시키는 메커니즘에 대한 통찰을 제공합니다. 7개의 CIL 모델을 사용하여 아키텍처 기반, 리플레이 기반, 정규화 기반 및 프롬프트 기반 방식의 학습 전략을 종합적으로 분석합니다.

- **Performance Highlights**: 중요 샘플로부터 학습하는 것이 (i) 점진적 정확성을 증가시키고, (ii) 이전 작업에 대한 지식 유지력을 향상시키며, (iii) 학습된 표현을 정제하는 것으로 나타났습니다. 이 연구는 데이터 중심 접근 방식이 연속 학습에서 유익할 수 있음을 강조합니다.



### Longitudinal Causal Image Synthesis (https://arxiv.org/abs/2410.17691)
- **What's New**: 이번 연구에서는 인과적(longitudinal) 이미지 합성을 위한 새로운 방법인 tabular-visual causal graph (TVCG)를 제안하였습니다. 이는 치매의 대표적인 예로 알츠하이머병(AD) 환자의 뇌 MRI를 포함하여, 환자의 상태를 더 효과적으로 분석할 수 있는 가능성을 보여줍니다.

- **Technical Details**: TVCG는 고차원 이미지와 저차원 표 형 변수 간의 불일치를 해결하고, 임상 데이터를 통한 인과 관계를 명확하게 모델링하기 위해 신경망(Neural Network)과 구조적 인과 모델(Structural Causal Models)을 통합한 방법입니다. 이 모델은 특히 지속적인 시간 예측을 위한 시간 간격을 독립 변수로 포함하여 훈련 데이터를 효과적으로 처리합니다.

- **Performance Highlights**: TVCG는 ADNI 데이터셋을 기반으로 훈련되어 두 개의 추가 AD 데이터셋에서 평가되었으며, 합성된 이미지는 뛰어난 품질을 보이며, 알츠하이머 진행 특성을 효과적으로 설명하는데 기여하였습니다. 이로 인해 임상적 신뢰성과 유용성이 substantiating 되었습니다.



### Deep Generative Models for 3D Medical Image Synthesis (https://arxiv.org/abs/2410.17664)
- **What's New**: 이 논문은 3D 의료 이미지를 생성하기 위한 다양한 Deep Generative Models, 특히 Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), 그리고 Denoising Diffusion Models (DDMs)에 대한 최신 발전을 탐구합니다. 이러한 모델들이 의료 영상 분석 및 진단작업에서의 응용 가능성을 제공합니다.

- **Technical Details**: 본 논문에서는 VAEs, GANs 및 DDMs의 기본 원리와 각 모델의 장단점에 대해 설명하며, 훈련 목표 및 평가 메트릭의 사용에 대해 논의합니다. 특히, 3D 의료 이미지는 복잡한 데이터 특성으로 인해 기존 방법을 조정해야 하며, VAEs는 입력 이미지를 잠재 표현으로 개념화하고 KL-divergence 정규화 항을 적용하여 새로운 샘플을 생성합니다.

- **Performance Highlights**: 이 모델들은 의료 이미지 생성에서 빠른 추론 시간과 다양한 이미지를 생성할 수 있는 능력을 갖추어 있습니다. 그러나, GANs는 상대적으로 품질이 낮은 샘플을 생성하는 문제를 가지고 있어, VQ-VAEs와 같은 변형이 제안되었습니다. 이러한 테스트 사례들은 연구 및 임상에 있어 데이터 부족 문제를 해결하는 데 중요한 기여를 할 것으로 기대됩니다.



### ImDy: Human Inverse Dynamics from Imitated Observations (https://arxiv.org/abs/2410.17610)
Comments:
          Yong-Lu Li and Cewu Lu are the corresponding authors

- **What's New**: 이 논문은 인간의 모션 캡처 데이터를 활용하여 새로운 Inverse Dynamics(ID) 분석 도구인 Imitated Dynamics(ImDy)와 데이터 기반 ID solver인 ImDyS(olver)를 제안합니다. 기존의 비싼 실험 환경에 의존하지 않고, 데이터 기반으로 ID를 분석할 수 있는 새로운 접근 방식을 제공합니다.

- **Technical Details**: 이 연구에서는 최첨단 모션 모방 알고리즘과 물리 시뮬레이터를 활용하여 150시간 이상의 인간 모션을 포함하는 대규모 데이터셋인 ImDy를 생성합니다. ImDyS는 이 데이터셋을 이용하여 ID 및 ground reaction force 추정을 동시에 수행하도록 훈련됩니다. 주요 기술적 요소로는 forward dynamics awareness와 motion plausibility constraints가 포함됩니다.

- **Performance Highlights**: ImDyS는 ImDy에서의 실험과 실제 데이터 기반의 GroundLink 데이터셋을 통해 뛰어난 성능을 보여주었으며, 향후 로봇 공학, 건강 관리, 스포츠 훈련 등의 다양한 분야에 적용될 가능성이 있습니다.



### BlurryScope: a cost-effective and compact scanning microscope for automated HER2 scoring using deep learning on blurry image data (https://arxiv.org/abs/2410.17557)
Comments:
          18 Pages, 6 Figures

- **What's New**: 새로운 초음속 스캐닝 광학현미경 "BlurryScope"를 개발하여 깊이 학습(Deep Learning)을 활용한 자동 검사를 위한 효과적이고 컴팩트한 솔루션을 제시했습니다.

- **Technical Details**: BlurryScope는 느린 움직임으로 인해 생기는 모션 블러(Motion Blur) 이미지 처리 및 자동 병리학(Pathology) 분류를 위한 신경망(Neural Network) 모델과 특수 하드웨어를 통합하여 설계되었습니다. 이 장치는 IHC(면역조직화학) 염색된 유방 조직 섹션의 HER2 점수를 자동으로 분류하기 위해 5 mm/s의 속도로 HER2 염색된 조직 미세 배열(TMA)을 스캔하는 방식으로 작동합니다.

- **Performance Highlights**: 284개의 독창적인 환자 샘플을 사용한 시험 세트에서 4등급(HER2 점수 0, 1+, 2+, 3+) 및 2등급(HER2 점수 0/1+, 2+/3+) 분류에서 각각 79.3% 및 89.7%의 정확도를 달성하였습니다. BlurryScope는 이미지 스캐닝부터 관심 영역의 스티칭 및 크롭, HER2 점수 분류까지 전체 워크플로우를 자동화하여 진단 시간을 절약하고 다양한 임상 환경에서 암 식별 및 분류를 강화할 수 있는 잠재력을 가지고 있습니다.



### Unsupervised Low-dose CT Reconstruction with One-way Conditional Normalizing Flows (https://arxiv.org/abs/2410.17543)
- **What's New**: 이번 논문에서는 저선량 컴퓨터 단층촬영(LDCT)에서 노이즈를 효과적으로 제거하고 고품질 이미지를 복원하기 위한 새로운 조건부 정규화 흐름(CNF) 기반의 비지도 LDCT 반복 복원 알고리즘을 제안합니다. 이 알고리즘은 엄격한 일방향 변환을 사용하여 현재 NFs 기반 방법들이 직면한 문제를 해결합니다.

- **Technical Details**: 제안된 알고리즘은 데이터와 잠재 공간에서의 교차 최적화를 통해 NFs의 정규화 및 생성 가능성을 최대한 활용합니다. 이 방법은 두 방향 변환을 피하고, 고해상도 CT 이미지에서도 쉽게 작동할 수 있도록 조건부 정규화 흐름(CNFs) 훈련 방식을 채택합니다.

- **Performance Highlights**: 다양한 데이터셋에서 실험을 통해, 제안된 방법은 기존의 최첨단 비지도 및 일부 지도 학습된 방법보다 우수한 성능을 보임을 확인했습니다. 이 알고리즘은 빠르고 효율적인 LDCT 복원을 가능하게 하며, 실제 임상 적용에 적합합니다.



### Bilateral Hippocampi Segmentation in Low Field MRIs Using Mutual Feature Learning via Dual-Views (https://arxiv.org/abs/2410.17502)
- **What's New**: 이번 논문은 저자들이 저자들에 의해 제공된 새로운 딥러닝 접근법을 통해 저자들이 더욱 접근하기 쉬운 저자들에 의해 생성된 저자들이 생성하는 이중 뷰 구조를 활용하여 저자들에게 효과적으로 기능을 학습하는 것을 목표로 합니다.

- **Technical Details**: 제안된 모델은 저주파 이미지와 원본 저주파 이미지를 결합하여 양측 해마의 복잡한 구조를 포착합니다. 이 방법은 고주파 마스킹을 통해 구현되며, LISA 2024 저장 MRI 데이터셋을 사용하여 세밀한 실험을 진행했습니다.

- **Performance Highlights**: 우리의 방법은 저자원이 제한된 환경에서 안정적인 해마 분석을 위한 정확하고 신뢰할 수 있는 세분화 결과를 제공합니다.



### Enhancing Multimodal Medical Image Classification using Cross-Graph Modal Contrastive Learning (https://arxiv.org/abs/2410.17494)
- **What's New**: 이 논문은 다중 모달(multi-modal) 의료 이미지 분류를 위한 새로운 Cross-Graph Modal Contrastive Learning (CGMCL) 프레임워크를 제안합니다. 이 모델은 이미지 데이터와 비이미지 데이터(예: 메타 특성)를 통합하여 대칭적 잠재 공간에서 다중 모달 특성을 정렬합니다.

- **Technical Details**: CGMCL은 교차 모달 그래프를 구축하고 대비 학습(contrastive learning)을 활용하여 다양한 모달 데이터를 효과적으로 통합합니다. 또한, GATs(Graph Attention Networks)를 사용하여 노드 임베딩을 통해 모달 간 특성 인코딩을 학습합니다. 이 방법은 두 개의 데이터 세트(파킨슨병 데이터셋과 멜라노마 데이터셋)에서 평가되었습니다.

- **Performance Highlights**: CGMCL은 정확성, 해석 가능성, 및 조기 질병 예측에서 기존의 단일 모달(unimodal) 방법보다 뛰어난 성능을 나타냈습니다. 또한 멜라노마 다중 분류에서도 우수한 성능을 보였습니다.



### GenDP: 3D Semantic Fields for Category-Level Generalizable Diffusion Policy (https://arxiv.org/abs/2410.17488)
Comments:
          Accepted to Conference on Robot Learning (CoRL 2024). Project Page: this https URL

- **What's New**: 이 논문에서는 로봇 조작 작업에서의 일반화 능력을 향상시키기 위해 새로운 프레임워크인 3D semantic fields를 도입했습니다. 이 방법은 복잡한 기하학적 정보와 의미 정보를 명시적으로 통합하여 미지의 객체에 대한 일반화 능력을 향상시킵니다.

- **Technical Details**: 제안된 프레임워크는 3D descriptor fields encoder, semantic fields constructor, action policy의 세 가지 주요 모듈로 구성됩니다. 3D descriptor encoder는 다중 뷰 RGBD 관측값을 입력 받아 고차원 디스크립터를 출력합니다. 이어서 이 디스크립터는 semantic fields constructor로 전달되어 낮은 차원의 semantic fields로 변환되며, 최종적으로 정책은 semantic fields와 포인트 클라우드를 입력 받아 행동을 예측합니다.

- **Performance Highlights**: 이 연구는 8가지 작업에 걸쳐 평가되었으며, 기존 Diffusion Policy의 미지의 인스턴스에서의 성공률을 20%에서 93%로 향상시켰습니다. 이 방법은 기하학적 모호성을 해결하고 미세한 기하학적 세부 사항에 주목하여 보다 강력한 일반화 능력을 보여줍니다.



### AG-SLAM: Active Gaussian Splatting SLAM (https://arxiv.org/abs/2410.17422)
- **What's New**: AG-SLAM은 3D Gaussian Splatting (3DGS)을 활용한 최초의 능동적 SLAM 시스템입니다. 이 시스템은 온라인으로 장면을 재구성하는 데 중점을 둡니다.

- **Technical Details**: AG-SLAM은 정보 이득(maximizing information gain)과 위치 오류의 비용을 최소화하는 목표를 균형 있게 조정하기 위해 Fisher Information을 활용합니다. 이 방법은 Gibson과 Habitat-Matterport 3D 데이터셋에서 검증되었습니다.

- **Performance Highlights**: AG-SLAM은 Active Neural SLAM, ExplORB, UPEN, active-INR 및 frontier-based exploration과 비교하여 다양한 메트릭에서 우수한 재구성 품질을 보여주었습니다.



### Geometric Graph Neural Network Modeling of Human Interactions in Crowded Environments (https://arxiv.org/abs/2410.17409)
Comments:
          \c{opyright} 2024 the authors. This work has been accepted to IFAC for publication under a Creative Commons Licence CC-BY-NC-ND

- **What's New**: 이 연구에서는 심리학적 연구에서 얻은 도메인 지식을 통합하여 보행자 간의 상호작용을 모델링하고 미래 경로를 예측하는 기하학적 그래프 신경망(GNN) 아키텍처를 제안합니다. 기존 연구와 달리, 우리는 보행자의 시야, 이동 방향, 거리 기반 커널 함수를 사용하여 상호작용 이웃을 정의합니다.

- **Technical Details**: 제안된 모델은 Mohamed et al.(2020)에 의해 제안된 Social Spatio-Temporal Graph Neural Network (STGCNN) 아키텍처에 기반하여 기하학적 커널 함수를 접목하고 상호작용 이웃 정의를 정제합니다. 경로 예측 작업에서 우리는 보행자 경로가 이변량 가우시안 분포를 따른다고 가정하고, 평균, 표준 편차 및 상관 계수를 포함한 가우시안 분포의 매개변수를 모델링합니다.

- **Performance Highlights**: 여러 데이터 세트를 통해 실험한 결과, 평균 및 최종 변위 오차 메트릭이 감소함에 따라 예측 정확도가 향상되었습니다. 이 연구는 도메인 지식과 데이터 기반 접근 방식을 통합하여 복잡한 보행자 상호작용 모델링에 효과적이라는 점을 강조합니다.



### Efficient Feature Extraction Using Light-Weight CNN Attention-Based Deep Learning Architectures for Ultrasound Fetal Plane Classification (https://arxiv.org/abs/2410.17396)
Comments:
          Submitted to Computers in Biology and Medicine journal

- **What's New**: 본 연구에서는 경량 인공지능 아키텍처를 제안하여 초음파 태아 이미지의 정확한 특징 추출을 지원하고, 태아 평면 분류(fetal plane classification, FPC) 작업에서 높은 성능을 달성합니다.

- **Technical Details**: EfficientNet에서 사전 훈련된 특징 추출기(backbones)와 주의(attention) 메커니즘을 활용한 CNN-Attention 파이프라인을 설계했습니다. 12,000개의 이미지를 포함하는 대규모 벤치마크 초음파 데이터셋을 사용하여 태아의 주요 평면(예: 뇌, 대퇴골, 흉부 등)을 분류합니다. 이 모델은 40배 적은 학습 가능한 파라미터를 가지고 있어 엣지 디바이스에서도 쉽게 배포할 수 있습니다.

- **Performance Highlights**: 최고 Top-1 정확도 96.25%, Top-2 정확도 99.80% 및 F1-Score 0.9576을 달성했습니다. GradCAM을 이용한 해석 가능성을 제공하여 의료 및 진단 지원에 기여합니다.



### Do Vision-Language Models Represent Space and How? Evaluating Spatial Frame of Reference Under Ambiguities (https://arxiv.org/abs/2410.17385)
Comments:
          Accepted to Pluralistic Alignment @ NeurIPS 2024 | Project page: this https URL

- **What's New**: 이 논문에서는 COnsistent Multilingual Frame Of Reference Test (COMFORT)를 소개하여 비전-언어 모델(VLM)의 공간 추론 능력을 체계적으로 평가하는 프로토콜을 제시합니다. COMFORT는 글로벌 109개 언어에서 9개의 최고 수준의 VLM을 평가했습니다.

- **Technical Details**: COMFORT는 인공적인 3D 이미지와 그에 해당하는 텍스트 묘사를 기반으로 한 일련의 공간 추론 과제를 포함하고, 모델의 응답의 견고성(robustness) 및 일관성(consistency)을 평가하기 위한 지표(metrics)를 도입합니다. 이를 통해 VLM의 공간 언어 이해에서 영어 관습과의 정렬을 확인하였으나, 여러 FoR을 수용하는 유연성이 부족함을 밝혀냈습니다.

- **Performance Highlights**: VLM은 영어 관습에 어느 정도 맞추어 형상적 모호성을 해결하는 데는 성공했으나, 여전히 (1) 견고성과 일관성이 부족하고, (2) 여러 FoR을 수용하는 유연성이 결여되었으며, (3) 언어 및 문화적 관습을 준수하지 못하는 한계를 가지고 있습니다. 이는 VLM들이 인간의 인지와 정렬되기 위한 지속적인 노력에도 불구하고 공간 언어의 모호함과 문화 간 다양성에서 더 많은 주의를 필요로 함을 나타냅니다.



### PtychoFormer: A Transformer-based Model for Ptychographic Phase Retrieva (https://arxiv.org/abs/2410.17377)
Comments:
          20 pages, 12 figures

- **What's New**: 이번 연구에서는 PtychoFormer라는 계층적 transformer 기반 모델을 제안하여 단일 샷에서 데이터 기반의 ptychographic phase retrieval을 수행합니다. 이 모델은 조합된 diffractions 패턴을 처리하여 지역적 추론을 생성하고 이를 연결하여 고품질의 재구성을 제공합니다.

- **Technical Details**: PtychoFormer는 sparse하게 스캔된 diffraction 패턴에 대해 내성을 보이며, extended PtychoFormer(ePF) 하이브리드 접근법을 통해 global phase shifts를 최소화하고 재구성 품질을 크게 향상시키는 데 중점을 두고 있습니다. 이러한 아키텍처는 큰 데이터셋에서 효과적으로 확장될 수 있도록 설계되었습니다.

- **Performance Highlights**: PtychoFormer는 ePIE보다 최대 3600배 빠른 이미징 속도를 제공하며, 재구성 작업에서 이전의 딥러닝 기반 phase retrieval 방법들을 넘어서는 성능을 보였습니다.



### Stool Recognition for Colorectal Cancer Detection through Deep Learning (https://arxiv.org/abs/2410.17288)
Comments:
          21 pages, 28 figures

- **What's New**: 이번 연구에서는 대변에 혈액이 있는지 여부를 확인할 수 있는 대변 인식 신경망(stool recognition neural network)를 제안합니다. 이는 기존의 faecal occult blood test (FOBT)의 불편함을 해결하기 위한 빠르고 간편한 대안입니다.

- **Technical Details**: 본 연구에서는 Generative Adversarial Networks (GANs) (DiffAugment StyleGAN2, DCGAN, Conditional GAN)를 활용하여 제한된 데이터를 보완하기 위해 고충실도(high fidelity) 이미지를 생성했습니다. DiffAugment StyleGAN2로 생성된 이미지는 분류기 훈련 배치에 즉시 추가하여 정확도를 94%로 향상시켰습니다.

- **Performance Highlights**: 이 모델은 모바일 앱 Poolice에 배포되어 사용자가 대변 사진을 찍고 즉각적으로 혈액 여부를 확인할 수 있습니다. 이 앱은 조기 발견을 통한 생명 구제를 목표로 하며, 사용자들이 조기에 치료를 받을 수 있도록 돕습니다.



### Uncovering Regional Defaults from Photorealistic Forests in Text-to-Image Generation with DALL-E 2 (https://arxiv.org/abs/2410.17255)
Comments:
          Accepted by the 16th Conference on Spatial Information Theory (COSIT 2024): this https URL

- **What's New**: 이 논문은 텍스트-이미지(T2I) 생성 모델에서 특정 지역이 과도하게 묘사되는 경향을 분석하고, 이러한 지역 편향을 평가하기 위한 스케일러블한 방법론을 제시합니다.

- **Technical Details**: 종합적으로, 연구는 DALL-E 2라는 최신 T2I 생성 모델을 사용하여 지역 위계(Region Hierarchy)에 기반한 이미지 생성을 수행하고, 교차 수준 유사성(Cross-level Similarity) 비교를 통해 지역 편향을 분석합니다. 두 가지 유사성 측정 지표인 평균 제곱 오차(Mean Squared Error, MSE)와 구조적 유사성 지수(Structural Similarity Index Measure, SSIM)를 사용합니다.

- **Performance Highlights**: 실험 결과, DALL-E 2에서의 지역 편향은 지역 위계의 특정 수준에서 가장 높은 유사성을 보이는 지역에 따라 결정되며, 이는 실제 나무가 잘 자생하는 지역과 일치하지 않을 수 있습니다.



### Exploring Stronger Transformer Representation Learning for Occluded Person Re-Identification (https://arxiv.org/abs/2410.15613)
- **What's New**: 이 논문에서는 사람 재식별(person re-identification) 문제를 해결하기 위한 혁신적인 자기 지도 학습(self-supervised learning)과 지도 학습(supervised learning)을 결합한 변형기(transformer) 기반 프레임워크인 SSSC-TransReID를 제안합니다. 기존의 모델과 달리, 부정 샘플(negative samples)이나 추가적인 사전 학습(pre-training) 없이 특징 표현(feature representation)을 향상시키기 위해 자기 지도 대조 학습(self-supervised contrastive learning) 분기를 설계했습니다.

- **Technical Details**: 이 연구에서는 실제 장면에서의 가림 현상을 시뮬레이션하기 위한 새로운 랜덤 사각형 마스크(random rectangle mask) 전략을 제안하였습니다. 또한, ViT 모델의 특징 학습 능력을 향상시키기 위해, 부정 샘플에 의존하지 않는 자기 지도 대조 학습 분기를 구축하고, ID 태그와 부정 샘플이 없는 자기 지도 대조 학습의 장점을 통합하는 공동 학습 손실 함수를 활용하여 모델의 성능을 향상시켰습니다.

- **Performance Highlights**: 여러 벤치마크 데이터셋에서의 실험 결과, 제안된 모델이 평균 평균 정확도(mean average accuracy, mAP)와 Rank-1 정확도에서 기존의 최첨단 ReID 방법들과 비교하여 큰 차이로 우수한 성능을 보임을 확인하였습니다.



New uploads on arXiv(cs.AI)

### Explaining Bayesian Networks in Natural Language using Factor Arguments. Evaluation in the medical domain (https://arxiv.org/abs/2410.18060)
Comments:
          First Workshop on Explainable Artificial Intelligence for the medical domain - EXPLIMED. THE 27TH EUROPEAN CONFERENCE ON ARTIFICIAL INTELLIGENCE

- **What's New**: 본 논문에서는 Bayesian Network Reasoning을 위한 자연어 설명 생성을 위한 모델을 제안합니다. 이는 관측된 증거를 목표 변수와 연결짓는 인과 관계 그래프인 factor arguments를 통해 이루어집니다.

- **Technical Details**: 우리는 인과 관계의 독립성(factor argument independence)을 정의하는 새로운 개념을 소개하고, 증거 노드에서 시작하여 목표 노드까지의 독립적인 factor arguments의 목록을 생성하는 알고리즘을 제시합니다. 이 접근방식을 바탕으로 Bayesian Reasoning에 대한 자연어 설명을 구축하는 방법론을 구현하였습니다.

- **Performance Highlights**: 의료 분야에서 인간 주도의 평가 연구를 통해 제안된 설명 방법이 기존의 대안 설명 방법보다 Bayesian Network Reasoning을 이해하는 데 훨씬 더 유용하다고 평가되었습니다.



### GraphTeam: Facilitating Large Language Model-based Graph Analysis via Multi-Agent Collaboration (https://arxiv.org/abs/2410.18032)
- **What's New**: 이번 논문에서는 LLM 기반의 그래프 분석을 위한 새로운 멀티 에이전트 시스템인 GraphTeam을 제안합니다. 이 시스템은 여러 에이전트가 협업하여 복잡한 문제를 해결하는 방식을 모방합니다.

- **Technical Details**: GraphTeam은 세 가지 모듈로 구성되어 있습니다: (1) Input-output normalization 모듈, (2) External knowledge retrieval 모듈, (3) Problem-solving 모듈. 각 모듈은 문제를 효과적으로 이해하고 해결하기 위해 특화된 여러 LLM 기반 에이전트로 구성됩니다.

- **Performance Highlights**: GraphTeam은 여섯 가지 그래프 분석 벤치마크에서 실험을 실시한 결과, 평균 25.85%의 정확도 향상을 이루어냈으며, SOTA 성능을 달성했습니다.



### Benchmarking Foundation Models on Exceptional Cases: Dataset Creation and Validation (https://arxiv.org/abs/2410.18001)
Comments:
          EMNLP 2024 Workshop Genbench(this https URL)

- **What's New**: 이 논문은 아울러 Foundation Models (FMs)의 성능을 평가하기 위한 새로운 데이터셋을 제안하고, Graphics Novels, Calligraphy, News Articles, Lyrics 등 다양한 장르를 포함하는 여러 모달리티에서의 reasoning 작업을 다룹니다.

- **Technical Details**: 이 연구에서는 Out-of-Distribution (OOD) reasoning을 정의하고, Chain-of-Thought (CoT) 및 CoT+Few-Shot과 같은 prompt engineering 기법을 제출하여 FMs의 성능을 향상시켰습니다. 실험은 GPT-4o와 Gemini-1.5-pro 모델을 사용하여 진행되었습니다.

- **Performance Highlights**: Graphic Novels 및 Calligraphy 작업에서 FMs의 성능이 저조했지만, Onion과 Not The Onion 작업의 경우 우수한 성과를 기록했습니다. 특히, Infilling 작업에서도 모델이 토큰 예측에서 개선을 보였습니다.



### AI driven health recommender (https://arxiv.org/abs/2410.17991)
- **What's New**: 해당 논문에서는 AI 기술을 활용하여 환자의 증상을 기반으로 질병을 감지하고 적절한 약물 및 예방 조치, 식단, 운동을 추천하는 웹 애플리케이션을 개발했습니다.

- **Technical Details**: 본 웹 애플리케이션은 Machine Learning을 기반으로 하며, Flask 프레임워크를 사용하여 사용자 친화적인 플랫폼을 구축했습니다. 또한, 실시간 데이터(Real-time data)를 이용하여 보다 정확한 정보 제공이 가능하도록 설계되었습니다.

- **Performance Highlights**: 이 애플리케이션은 사용자 증상을 바탕으로 신속하게 질병을 판단하고, 환자가 필요한 정보(약물, 예방책, 식단 및 운동)를 체계적으로 제공함으로써 환자의 치료 과정을 개선할 수 있는 잠재력을 가지고 있습니다.



### ExpertFlow: Optimized Expert Activation and Token Allocation for Efficient Mixture-of-Experts Inferenc (https://arxiv.org/abs/2410.17954)
Comments:
          Mixture-of-Experts, Inference, Offloading

- **What's New**: 이 논문은 ExpertFlow라는 새로운 시스템을 소개하여 Sparse Mixture of Experts (MoE) 모델의 추론 효율성을 개선하고 메모리 요구를 낮추는을 목표로 하고 있습니다. 이를 통해 리소스가 제한된 환경에서도 높은 성능을 발휘할 수 있도록 합니다.

- **Technical Details**: ExpertFlow는 세 가지 주요 구성 요소로 되어 있습니다: Routing Path Predictor(경로 예측기), Expert Cache Engine(전문가 캐시 엔진), Token Scheduler(토큰 스케줄러). 이 시스템은 예측 가능한 라우팅 경로에서 동적으로 전문가를 스케줄링하고, 비슷한 라우팅 경로를 가진 토큰을 그룹화하여 I/O 오버헤드를 줄여줍니다. 경량화된 예측기를 사용하여 예측 경로를 계산하고, 예측 가능한 공간 인식 전문가 캐싱 전략을 통해 캐시 효율성을 높입니다.

- **Performance Highlights**: ExpertFlow는 최대 93.72%의 GPU 메모리 절약 효과를 보였으며, 추론 속도는 기존 방법에 비해 2배에서 10배까지 향상되었습니다. 더불어 ExpertCache Hit Ratio는 91.96%로, 기존 LRU 캐시 전략에 비해 평균 27.65% 증가하였습니다.



### Benchmarking Floworks against OpenAI & Anthropic: A Novel Framework for Enhanced LLM Function Calling (https://arxiv.org/abs/2410.17950)
Comments:
          15 pages for main paper, 21 pages in total including references and appendix, 10 figures

- **What's New**: ThorV2라는 새로운 아키텍처가 도입되어 LLM의 function calling 능력을 크게 향상시켰습니다. 이 모델은 주변 환경에 맞춰 오류를 식별하고 수정하는 접근 방식인 'edge-of-domain modeling'을 채택하고 있습니다.

- **Technical Details**: ThorV2는 Cognitive Enhancement Architecture (CEA)로 설계되었으며, 기존의 포괄적인 지침을 요구하지 않고 에러 수정 기반의 접근 방식을 사용합니다. 이를 통해 Agent-Validator 아키텍처를 통해 API 호출의 정확성을 높이고 있습니다. 또한, Multistep task를 다루기 위한 새로운 Composite Planning 방식을 적용하여 여러 API 호출을 한 번의 단계에서 생성할 수 있습니다.

- **Performance Highlights**: ThorV2는 OpenAI와 Anthropic의 기존 모델들보다 정확도, 신뢰성, 지연 시간, 비용 효율성에서 우수한 성능을 보였습니다. 특히, 복잡한 다단계 작업에서도 월등히 좋은 성능을 발휘하며, 기존 모델에 비해 우수한 신뢰성과 확장성을 자랑합니다.



### Guide for Defense (G4D): Dynamic Guidance for Robust and Balanced Defense in Large Language Models (https://arxiv.org/abs/2410.17922)
- **What's New**: 대규모 언어 모델(LLM)의 안전성을 강화하기 위해, Guide for Defense (G4D)라는 새로운 다중 에이전트 기반 방어 프레임워크를 도입했습니다. 이 프레임워크는 도메인 특화 시나리오에서 악의적인 쿼리에 대한 대응 능력을 향상시키는 것을 목표로 하고 있습니다.

- **Technical Details**: G4D는 세 가지 에이전트를 사용하여 LLM이 생성하는 응답이 핵심 가치와 일치하도록 안내합니다. (1) 의도 탐지기(intent detector)는 사용자 의도를 요약합니다. (2) 질문 재구성기(question paraphraser)는 쿼리를 재구성하여 악의적인 공격을 무력화합니다. (3) 안전 분석기(safety analyzer)는 충분한 맥락을 바탕으로 의도를 평가하여 안전한 응답 작성에 대한 조언을 제공합니다.

- **Performance Highlights**: 실험 결과, G4D는 도메인 특정 jailbreak 공격에 대한 공격 성공률(ASR)을 크게 낮추며, 모델의 일반적인 기능성을 유지하면서 높은 방어 성능을 보여줍니다. 또한, 소규모 LLM을 활용하는 방어 기관은 빠른 추론 속도와 비용 효율성을 제공합니다.



### Leveraging Deep Learning for Time Series Extrinsic Regression in predicting photometric metallicity of Fundamental-mode RR Lyrae Stars (https://arxiv.org/abs/2410.17906)
Comments:
          Sensors 2024, 24(16), 5203; (23 pages)

- **What's New**: 이 연구는 ESA의 Gaia 망원경 데이터셋을 활용하여 fundamental mode (ab-type) RR Lyrae 별의 금속성을 추정하는 새로운 딥러닝(deep learning) 접근 방식을 제시합니다. 기존의 전통적인 분석 방법으로는 이러한 대용량 데이터의 처리에 한계가 있어, 고급 계산 기술의 사용이 요구되었습니다.

- **Technical Details**: 이 연구는 딥러닝 기술, 특히 신경망(neural network) 아키텍처를 통해 RR Lyrae 별의 빛 곡선(light curve) 데이터를 이용한 금속성 예측 모델을 개발했습니다. 모델의 예측 성능은 평균 절대 오차(mean absolute error, MAE) 0.0565, 평균 제곱근 오차(root mean square error, RMSE) 0.0765, 교차 검증에 의한 결정 계수($R^2$) 0.9401 등으로 나타났습니다. 이러한 결과는 딥러닝 모델이 금속성 값 추정에 효과적임을 보여줍니다.

- **Performance Highlights**: 딥러닝 모델이 획득한 성과는 정확한 금속성 추정을 가능하게 했으며, 특히 Gaia 임무(special mission)에서 생성된 대량의 데이터셋을 통해 천문학적 연구의 정확성을 높일 수 있는 잠재력을 보여줍니다. 이 연구를 통해 RR Lyrae 별의 특성을 분석하고, 별 집단과 은하의 진화에 대한 더 깊은 통찰력을 제공할 수 있게 되었습니다.



### R-CoT: Reverse Chain-of-Thought Problem Generation for Geometric Reasoning in Large Multimodal Models (https://arxiv.org/abs/2410.17885)
- **What's New**: 이 논문은 기존의 Large Multimodal Models (LMMs)이 겪고 있는 수학 기하학적 추론의 어려움을 해결하기 위해 Reverse Chain-of-Thought (R-CoT) 기하학 문제 생성 파이프라인을 제안합니다. 이 방법은 고품질의 이미지-텍스트 데이터 생성을 통해 기하적 문제의 답변 정확성을 향상시킵니다.

- **Technical Details**: R-CoT 파이프라인은 두 단계로 구성되며, 첫 번째 단계에서는 GeoChain을 통해 고충실도(High-fidelity) 기하학 이미지를 생성하고, 두 번째 단계에서는 Reverse A&Q 방법을 사용하여 설명 기반의 단계적 추론을 통해 질문을 생성합니다. 이로 인해 각 모델에서 새로운 성능 기록을 달성하였고, 2B, 7B, 8B 설정에서 모두 향상된 결과를 보여줍니다.

- **Performance Highlights**: R-CoT-8B는 MathVista와 GeoQA 데이터셋에서 각각 16.6% 및 9.2%의 성과 향상을 보이며, 유료 모델 GPT-4o보다 평균 13% 더 뛰어난 성능을 기록했습니다. 이는 여러 LMM 기초 모델에서 일관된 성능 향상을 나타냅니다.



### Lightweight Neural App Contro (https://arxiv.org/abs/2410.17883)
- **What's New**: 이번 논문에서는 다양한 Android 앱을 통해 효율적인 상호작용 및 제어를 위한 새로운 모바일 전화 제어 아키텍처인 'App Agents'를 소개합니다. 이를 통해 텍스트 목표 및 이전 모바일 관측치를 입력으로 받아 정확한 행동을 생성하는 라이트웨이트 다중 모달 앱 제어(LiMAC)를 제안합니다.

- **Technical Details**: LiMAC는 Action Transformer(AcT)와 정밀하게 조정된 비전-언어 모델(VLM)을 통합하여 실시간 의사결정 및 작업 실행을 최적화합니다. AcT는 스마트폰 사용자 인터페이스 상태에 따라 필요한 행동 유형을 예측하고, 특정 작업에 대해 VLM을 호출하여 자연어 처리를 수행합니다. 이러한 하이브리드 접근법은 계산 요구 사항을 줄이고 반응 속도를 개선하여 평균 3초의 빠른 실행 시간을 제공합니다.

- **Performance Highlights**: LiMAC는 공개 소스 모바일 제어 데이터셋에서 테스트된 결과, 고급 비전-언어 모델(VLM)인 Florence2 및 Qwen2-VL보다 최대 19% 높은 행동 정확도를 나타내었으며, GPT-4o 기반 모델들보다 최대 42% 더 높은 정확도를 기록했습니다. 실험 결과에 따르면 LiMAC는 작업 실행 시간을 30배 빨라지게 하며, 정확도 또한 40% 향상되었습니다.



### DataTales: A Benchmark for Real-World Intelligent Data Narration (https://arxiv.org/abs/2410.17859)
- **What's New**: DataTales라는 새로운 벤치마크를 소개하여 언어 모델의 데이터 내레이션(data narration) 능력을 평가할 수 있는 기회를 제공합니다. 기존 벤치마크들은 실제 응용 분야에서 요구되는 분석의 복잡성을 충분히 반영하지 못하는 경향이 있습니다.

- **Technical Details**: DataTales는 4.9K개의 재무 보고서와 해당 시장 데이터를 결합하여 언어 모델의 복잡한 데이터 내레이션 분석 능력을 테스트합니다. 이 데이터셋은 광범위한 분석 기법(예: causal analysis, trend analysis)과 전문 용어 파악의 중요성을 강조하며, 저자들은 ChatGPT를 사용하여 문장을 분류하는 방법을 적용했습니다.

- **Performance Highlights**: 현재의 최첨단 모델들을 DataTales에서 평가한 결과, 필요한 정확도와 분석 깊이에 도달하는 데 큰 어려움을 겪고 있음을 확인했습니다. 이는 더 발전된 추론이 가능한 모델 개발의 필요성을 강조합니다.



### RE-tune: Incremental Fine Tuning of Biomedical Vision-Language Models for Multi-label Chest X-ray Classification (https://arxiv.org/abs/2410.17827)
Comments:
          Accepted for publication at Medical Imaging meets NeurIPS (NeurIPS23)

- **What's New**: 본 논문에서는 다중 라벨 흉부 질병 진단을 위한 Incremental Learning 시나리오에서 미세 조정(pre-tuning)된 Multimodal Biomedical Vision-Language 모델(VLMs)에 대한 새로운 접근법 RE-tune을 소개합니다. RE-tune은 백본(backbone)을 동결(freeze)하고, VLM의 이미지 및 텍스트 인코더 위에 단순한 조정기(adaptor)만 훈련합니다. 긍정 및 부정 질병 텍스트 프롬프트(text prompts)를 설계하여 대규모 언어 모델(LLM)의 학습 경로를 조정합니다.

- **Technical Details**: RE-tune은 BioViL이라는 공개된 최첨단 다중 모달 생물 의학 모델을 미세 조정하는 데 초점을 맞추고 있습니다. 모델의 이미지 및 텍스트 인코더를 동결하고, 그 위에 Dense Layer 또는 Multi Layer Perceptron과 같은 간단한 조정기를 추가하여 훈련합니다. 이 방법은 기존의 정보 기억(REmember)과 연속적인 학습을 가능하게 하여, 모델이 새로 추가된 데이터를 학습할 때 이전에 배운 정보를 잃지 않도록 합니다.

- **Performance Highlights**: RE-tune은 다중 라벨 분류에서 높은 정확도를 달성하며, 환자의 프라이버시를 우선시하고 수치 연산 효율성이 뛰어나 실제 의료 환경에서 널리 채택될 수 있을 가능성을 보여줍니다. 실험 결과, Biomedical VLMs가 자연스러운 지속적인 학습자임을 나타내며, 재학습으로 인한 기억 상실을 방지합니다.



### Holon Programming Model -- A Software-Defined Approach for System of Systems (https://arxiv.org/abs/2410.17784)
- **What's New**: 이 논문에서는 시스템 간의 독립성과 자율성을 활용하여 복잡한 시스템의 상호작용을 효과적으로 프로그래밍하고 조정할 수 있는 Holon Programming Model (HPM)을 제안합니다. 이 모델은 재난 관리 시나리오를 통해 실제 적용 사례를 보여줍니다.

- **Technical Details**: HPM은 소프트웨어 정의 시스템(Software-Defined Systems, SDS)의 원칙을 기반으로 하여 고유한 요구사항에 맞게 조정된 전문 프로그래밍 모델이며, 이는 기존의 비동기 모델들을 뛰어넘으며 동적 조합과 프로그램 가능한 상호작용을 통합합니다.

- **Performance Highlights**: HPM을 통해 dynaSoS의 복잡성을 관리할 수 있는 효과적인 프로그래밍 환경을 제공하며, 재난 관리와 같은 실제 시나리오에서 시스템 레벨의 반응성을 증대시킵니다. 이는 보다 나은 상호운용성과 적응성을 가진 SoS를 구현할 수 있게 합니다.



### Evaluating Explanations Through LLMs: Beyond Traditional User Studies (https://arxiv.org/abs/2410.17781)
- **What's New**: AI(인공지능)가 헬스케어와 같은 중요한 분야에 통합됨에 따라, 설명 가능한 AI(Explainable AI, XAI) 도구의 필요성이 커지고 있습니다. 본 논문에서는 대규모 언어 모델(Large Language Models, LLMs)을 활용하여 인적 참여자를 복제하고 XAI 평가 과정의 효율성을 높이는 방안을 탐구합니다.

- **Technical Details**: 우리는 카운터팩추얼(Counterfactual)과 인과적(Causal) 설명의 유용성과 효과성을 비교하는 사용자 연구를 재현하였고, 7개의 최신 LLM(예: Llama 3, Qwen 2, Mistral)과 다양한 실험 설정을 통해 실험하였습니다. LLM의 메모리(memory)와 출력 변동성(output variability)이 인간 반응과의 일치성에 미치는 영향을 조사하였습니다.

- **Performance Highlights**: 실험 결과는 (i) LLM이 원래 연구의 대부분 결론을 재현할 수 있음을 보여주었고, (ii) 다양한 LLM이 결과의 일치 수준이 다르게 나타남과 (iii) 실험적 요인(예: LLM 메모리, 무작위성)이 인간 반응에 대한 일치성에 미치는 영향을 확인했습니다. 이러한 초기 발견은 LLM을 기반으로 하는 자동화된 질적 평가 프레임워크의 개발 가능성을 제시합니다.



### A Data-Driven Odyssey in Solar Vehicles (https://arxiv.org/abs/2410.17712)
- **What's New**: 이 연구는 태양광 차량의 운영에 대한 불확실성을 해소하기 위한 시뮬레이터를 개발하여, 사용자들이 장거리 여행에서 에너지 관리의 중요성을 이해할 수 있도록 돕는다. 구글 맵스 데이터와 날씨 정보를 활용하여 실제 주행 조건을 재현하며, 사용자 맞춤형 속도 입력을 통해 차량 상태를 실시간으로 업데이트한다.

- **Technical Details**: 시뮬레이터는 빠른 프로토타이핑 툴인 Streamlit을 사용하여 구현되었으며, 사용자 친화적인 인터페이스와 데이터 처리의 분리를 통해 유지 관리 및 확장성을 높인다. 주요 단계는 차량 사양 설정, 주행 계획 수립, 주행 정보 출력 및 제어로 구성된다. 태양광 패널의 면적, 차량 무게, 공기 저항 계수 등을 사용자 입력으로 설정하여 에너지 생산 및 소비에 미치는 영향을 확인할 수 있다.

- **Performance Highlights**: 시뮬레이터의 성능은 호주에서 진행된 월드 솔라 챌린지(WSC)의 경로를 사용하여 검증되었으며, 이를 통해 사용자들은 여행 전 에너지 동역학을 모니터링하고 최적의 주행 속도를 선택할 수 있는 기회를 제공받는다. 다양한 속도 정책 시나리오를 탐색하고, 최적의 주행 전략을 추천받을 수 있어 에너지 관리에 대한 이해를 높인다.



### PETAH: Parameter Efficient Task Adaptation for Hybrid Transformers in a resource-limited Contex (https://arxiv.org/abs/2410.17661)
- **What's New**: 본 연구에서는 하이브리드 변환기(Hybrid Transformers) 아키텍처에 대한 새로운 작업 적응 방식인 PETAH(Parameter Efficient Task Adaptation for Hybrid Transformers)를 소개합니다. 이 방법은 여러 다운스트림 작업을 하나의 공유된 변환기 백본을 통해 수행할 수 있도록 하여 효율성을 극대화합니다.

- **Technical Details**: PETAH는 하이브리드 변환기 구조에서 컨볼루션 레이어와 어텐션 레이어를 파라미터 효율적으로 적응시킬 수 있는 방안을 제시합니다. 이를 통해 전통적인 변환기 모델을 초월하는 성능 개선을 이루었습니다. 또한, PETAH 방식과 가지치기(Pruning) 기법을 결합하여 다중 작업을 처리할 수 있는 뛰어난 성능과 저장 친화적인 모델을 생성하였습니다.

- **Performance Highlights**: PETAH로 적응된 하이브리드 모델은 기존의 ViT(비전 변환기) 기반의 작업 적응 기법보다 더 적은 파라미터를 사용하면서도 모바일 하드웨어에서의 효율성이 더 높습니다. 여러 분류 및 비전 작업에 대한 평가에서 PETAH는 다양한 비전 작업을 수행하며 인지 수행 능력을 보여줍니다.



### AutoRNet: Automatically Optimizing Heuristics for Robust Network Design via Large Language Models (https://arxiv.org/abs/2410.17656)
- **What's New**: AutoRNet은 진화 알고리즘(EA)과 대형 언어 모델(LLM)을 통합하여 네트워크 강건성을 위한 효율적인 알고리즘 생성의 새로운 접근 방식을 제안합니다. 이 프레임워크는 수동 설계 및 대규모 레이블 데이터셋의 필요성을 줄이고, 도메인 지식을 활용하여 최적화 전략을 제공합니다.

- **Technical Details**: AutoRNet은 네트워크 최적화 전략(NOS)을 통해 다양한 역할을 수행하는 새로운 확률적 접근법을 설계하며, 적응형 적합성 함수(AFF)를 도입하여 수렴성과 다양성 균형을 맞춥니다. 이 시스템은 복잡한 네트워크 문제를 해결하기 위해 특화된 프롬프트를 생성할 수 있습니다.

- **Performance Highlights**: AutoRNet의 휴리스틱은 희소하고 조밀한 스케일 자유 네트워크에서 평가되어, 현재의 방법들과 비교하여 성능이 향상된 것으로 나타났습니다. 이는 자동화된 설계 솔루션을 통해 네트워크 강건성을 개선하는 데 기여합니다.



### Mapping the Media Landscape: Predicting Factual Reporting and Political Bias Through Web Interactions (https://arxiv.org/abs/2410.17655)
Comments:
          Accepted to CLEF 2024

- **What's New**: 이 논문은 뉴스 출처의 편향(바이어스)을 평가하는 새로운 방법론을 제안하며, 웹 상의 상호작용을 모델링하여 뉴스 매체의 신뢰도를 추정하는 방식을 확장합니다.

- **Technical Details**: 본 연구는 강화 학습(Deep Reinforcement Learning) 전략을 사용하여 뉴스 미디어 하이퍼링크 그래프에서 네 가지 모델의 분류 성능을 평가했습니다. 두 가지 편향 지표인 사실 보도(factual reporting)와 정치적 편향(political bias)을 중심으로 실험을 진행하였으며, CLEF 2023 CheckThat! Lab 챌린지에서 비공식적인 MAE 평가에서 우수한 성과를 나타냈습니다.

- **Performance Highlights**: 소스 미디어 수준에서 편향 지표를 예측하는 데 성공하였으며, CLEF CheckThat! 챌린지에서 새로운 SOTA(State-Of-The-Art) 기록을 세웠습니다. 또한, 사실 보도와 정치적 편향 레이블이 달린 가장 큰 주석 데이터셋을 공개했습니다.



### Markov Chain of Thought for Efficient Mathematical Reasoning (https://arxiv.org/abs/2410.17635)
Comments:
          Work in progress

- **What's New**: 이번 연구에서는 Chain of Thought (CoT)의 새로운 접근법인 Markov Chain of Thought (MCoT)를 도입하여 복잡한 수학적 추론 작업을 효율적으로 처리하는 방법을 제안합니다.

- **Technical Details**: MCoT는 각 추론 단계를 텍스트와 Python 코드 조각으로 정의하며, 코드 해석기와의 상호 작용을 통해 자기 수정(self-correction)이 가능하게 합니다. 이전의 추론 단계를 단순화된 질문으로 압축하여 긴 KV 캐시를 의존하지 않고 효율적인 다음 단계 추론을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, MCoTInstruct 데이터셋을 사용하여 MCoT는 효율성을 크게 향상시킴과 동시에 정확도를 유지하는 것으로 나타났습니다.



### Process Supervision-Guided Policy Optimization for Code Generation (https://arxiv.org/abs/2410.17621)
Comments:
          14 pages, 5 figures

- **What's New**: 이번 연구에서는 Process Reward Model (PRM)을 장착하여 RL 기반 코드 생성의 효율성을 향상시키는 새로운 방법을 제안하고 있습니다. PRM은 코드 생성 중 각 코드 라인의 정확성에 대해 밀집형 피드백을 제공하여 인간 프로그래머가 코드 수정하는 방식을 모방합니다.

- **Technical Details**: PRM은 코드 생성 중 발생하는 각 라인에 대한 피드백을 제공하여 학습 효율성을 향상시킵니다. 다양한 전략을 통해 PRM을 학습시키고 RL 프레임워크와 통합할 방법을 탐구하였으며, PRM을 사용하는 것이 성능을 크게 향상시킴을 발견했습니다. 사용된 데이터셋으로는 LiveCodeBench가 있습니다.

- **Performance Highlights**: 제안된 방법을 통해 인하우스 LLM의 LiveCodeBench 통과율이 28.2%에서 29.8%로 향상되었으며, 내부 벤치마크에서는 31.8%에서 35.8%로 증가했습니다. 특히, 복잡한 코드 생성 작업에 있어 PRM의 통합이 효과적임을 실험 결과에서도 확인했습니다.



### ImDy: Human Inverse Dynamics from Imitated Observations (https://arxiv.org/abs/2410.17610)
Comments:
          Yong-Lu Li and Cewu Lu are the corresponding authors

- **What's New**: 이 논문은 인간의 모션 캡처 데이터를 활용하여 새로운 Inverse Dynamics(ID) 분석 도구인 Imitated Dynamics(ImDy)와 데이터 기반 ID solver인 ImDyS(olver)를 제안합니다. 기존의 비싼 실험 환경에 의존하지 않고, 데이터 기반으로 ID를 분석할 수 있는 새로운 접근 방식을 제공합니다.

- **Technical Details**: 이 연구에서는 최첨단 모션 모방 알고리즘과 물리 시뮬레이터를 활용하여 150시간 이상의 인간 모션을 포함하는 대규모 데이터셋인 ImDy를 생성합니다. ImDyS는 이 데이터셋을 이용하여 ID 및 ground reaction force 추정을 동시에 수행하도록 훈련됩니다. 주요 기술적 요소로는 forward dynamics awareness와 motion plausibility constraints가 포함됩니다.

- **Performance Highlights**: ImDyS는 ImDy에서의 실험과 실제 데이터 기반의 GroundLink 데이터셋을 통해 뛰어난 성능을 보여주었으며, 향후 로봇 공학, 건강 관리, 스포츠 훈련 등의 다양한 분야에 적용될 가능성이 있습니다.



### CLR-Bench: Evaluating Large Language Models in College-level Reasoning (https://arxiv.org/abs/2410.17558)
Comments:
          18 pages, 6 figures, dataset and evaluation framework will be opensourced

- **What's New**: 대규모 언어 모델(LLMs)의 복잡한 대학 수준 추론을 포괄적으로 평가하기 위해 CLR-Bench를 제시했습니다. 이 벤치마크는 5 종류의 질문을 포함한 1,018개의 주제별 질문으로 구성되어 있으며, 각 질문에는 전문가는 물론 상세한 설명이 포함되어 있습니다.

- **Technical Details**: CLR-Bench는 16개의 컴퓨터 과학 및 인공지능 관련 학문 분야에서 엄선된 데이터를 제공합니다. 두 가지 새로운 평가 메트릭인 Q→A(질문→답변) 및 Q→AR(질문→답변 및 근거)를 도입하여 LLMs의 추론 능력을 정량화합니다.

- **Performance Highlights**: 실험 결과, LLMs는 진정한 이해 없이 '추측'하는 경향이 있으며, Q→AR에서 39.00%로 정확도가 낮아지는 경향을 보였습니다. 모델 크기가 항상 더 좋은 성능을 보장하지는 않으며, 작은 모델이 더 높은 Q→AR 성능을 발휘하기도 했습니다.



### FairDgcl: Fairness-aware Recommendation with Dynamic Graph Contrastive Learning (https://arxiv.org/abs/2410.17555)
Comments:
          12 pages, submitted to TKDE

- **What's New**: 본 논문에서는 추천 시스템의 공정성을 향상시키기 위해 FairDgcl이라는 동적 그래프 적대적 대조 학습(Adversarial Contrastive Learning) 프레임워크를 제안합니다. 기존의 데이터 증강(data augmentation) 방법의 한계를 분석하고, 사용자 특성(예: 나이, 성별)에 따른 불공정한 추천 결과를 해결하기 위한 새로운 접근 방식을 제시합니다.

- **Technical Details**: FairDgcl은 대조학습(Contrastive Learning)을 통해 공정한 데이터 증강 전략을 생성하는 적대적 대조 네트워크(Adversarial Contrastive Network)를 개발합니다. view generator와 view discriminator 두 개의 모델을 사용하여 자동으로 공정성을 개선하는 뷰를 생성하며, 깊은 학습(Deep Learning) 및 그래프 신경망(Graph Neural Networks, GNNs) 기술이 통합되어 있습니다. 이는 학습 과정에서 공정성과 정확성을 동시에 극대화할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, FairDgcl은 네 가지 실세계 데이터 세트에서 대부분의 기준 모델보다 공정성과 정확성 모두에서 우수한 성능을 보여주었습니다. 이는 제안된 접근 방식이 추천 시스템의 공정성을 높이는데 효과적임을 입증합니다.



### An Ontology-Enabled Approach For User-Centered and Knowledge-Enabled Explanations of AI Systems (https://arxiv.org/abs/2410.17504)
Comments:
          Doctoral dissertation. Some chapters appeared as individual papers - arXiv:2302.05752 is one such chapters

- **What's New**: 이 논문은 Explainable Artificial Intelligence (설명 가능한 인공지능, XAI)의 발전을 추구하며, 모델 중심의 설명과 사용자 중심의 설명 간의 간극을 메우기 위해 노력합니다. 특히, 문헌에서 파생된 설명 유형을 표현하기 위한 Explanation Ontology (설명 온톨로지, EO)를 설계하고, 임상 환경에서 문맥을 지원하는 질문-응답(QA) 파이프라인을 구현합니다.

- **Technical Details**: 논문에서는 최초로 15가지의 설명 유형을 나타낼 수 있는 Explanation Ontology (EO)를 생성하였으며, 이를 통해 다양한 AI 방법과 데이터 모달리티에서 생성된 설명들을 결합할 수 있는 시스템을 개발하고 있습니다. 전문가들이 필요로 하는 설명의 형태를 여러 차원에서 세분화해 모델링하고, 이를 통한 효과적인 정보 전달을 목표로 하고 있습니다.

- **Performance Highlights**: 지식 증강(Knowledge Augmentation)을 통해 대형 언어 모델의 성능이 개선되었으며, 임상 의사들은 주요 설명에서 실행 가능성(Actionability)을 중점적으로 선호하는 경향을 보였습니다. 추가적으로, 만성 질병 감지 설정에서 유사성 메트릭을 활용하여 다양한 AI 방법에서 나오는 설명을 결합할 계획입니다.



### Learning Fair and Preferable Allocations through Neural Network (https://arxiv.org/abs/2410.17500)
- **What's New**: 이 논문은 공정한 자원의 분배 문제를 해결하기 위해 Neural Round Robin (NRR)이라는 새로운 신경망 메커니즘을 도입하여, 기존의 Round Robin (RR) 방식을 학습하고 강화하는 방법을 제안합니다.

- **Technical Details**: NRR은 RR 메커니즘을 파라미터화하는 최신 신경망으로, SoftRR 알고리즘을 통해 RR의 이산 절차를 미분 가능하게 만들어 원활한 학습을 지원합니다.

- **Performance Highlights**: 실험 결과, NRR는 기존의 기준선보다 예측된 배분 결과의 근접성 및 기타 메트릭에서 우수한 성능을 보여주며, 공정성을 유지하면서 암묵적 할당 메커니즘을 회복할 수 있음을 입증하였습니다.



### Mechanisms of Symbol Processing for In-Context Learning in Transformer Networks (https://arxiv.org/abs/2410.17498)
Comments:
          101 pages (including 30 pages of Appendices), 18 figures

- **What's New**: 이번 논문에서는 Large Language Models (LLMs)이 context 내 학습을 통해 기호 처리를 성공적으로 수행할 수 있는 메커니즘을 탐구합니다. 이는 기계 학습 및 추상 기호 조작에 대한 오랜 예측에 도전하는 결과입니다.

- **Technical Details**: 저자들은 Production System 아키텍처의 통찰을 기반으로 한 고급 언어인 PSL을 개발하여, 복잡한 기호 처리를 위한 기호 프로그램을 작성합니다. 이 PSL 프로그램을 transformer 네트워크에서 정확하게 구현하는 컴파일러를 만들어, 해당 프로세스가 100% 기계적으로 해석 가능하도록 합니다.

- **Performance Highlights**: PSL은 Turing Universal하다는 것을 증명하고, 이를 통해 transformer ICL에 대한 이해를 심화할 수 있는 토대를 마련합니다. PSL 프로그램에서 컴파일되는 transformer 아키텍처는 기호 처리 능력을 향상시키기 위한 여러 경로를 제시합니다.



### AI, Global Governance, and Digital Sovereignty (https://arxiv.org/abs/2410.17481)
Comments:
          21 pages, 2 tables

- **What's New**: 이 논문은 인공지능(AI) 시스템이 국제 문제에서 점차 필수 요소가 되고 있음을 논의하며, 국가들이 AI 기반 글로벌 거버넌스에서 어떻게 힘을 행사하고 디지털 주권을 추구하는지 분석합니다. 필자는 AI가 정부 및 기업에 대해 제공하는 다양한 정치적 힘의 보상을 소개합니다.

- **Technical Details**: AI는 정책 능력과 경쟁력을 강화하는 '활성화 힘'(enabling force)으로 작용합니다. 논문에서는 AI 시스템이 공공 및 민간 글로벌 거버넌스에서의 힘 행사 방식에 대한 분류 체계를 제안하며, 정부와 기업 모두가 AI를 활용하여 힘과 주권을 형성하는 방법을 살펴봅니다.

- **Performance Highlights**: AI 고급화가 국가의 힘을 강화하는 동시에 사적 기업의 권력도 증대시키고 있음을 지적하며, 두 영역 간의 협력 및 경쟁의 역학이 형성되고 있음을 강조합니다. 논문은 AI 시스템의 발전이 인간의 의사결정을 대체하고 있음을 경고하며, 디지털 주권의 개념이 공공과 민간 권력의 얽힘 속에서 어떻게 발전하고 있는지를 설명합니다.



### Decoding Time Series with LLMs: A Multi-Agent Framework for Cross-Domain Annotation (https://arxiv.org/abs/2410.17462)
Comments:
          23 pages, 9 figures, 24 tables

- **What's New**: 본 논문에서는 TESSA라는 다중 에이전트 시스템을 제안하여 일반 및 도메인별 시간 시리즈 데이터 주석을 자동으로 생성하는 방법을 소개합니다.

- **Technical Details**: TESSA는 두 개의 에이전트로 구성되어 있습니다: 일반 주석 에이전트와 도메인별 주석 에이전트. 일반 에이전트는 다양한 소스 도메인에서 공통 패턴을 캡처하여 일반 사용자가 이해할 수 있는 주석을 생성합니다. 도메인별 에이전트는 특정 도메인에서 제한된 주석을 활용하여 특화된 용어로 주석을 생성합니다.

- **Performance Highlights**: 다양한 합성 및 실제 데이터셋에 대한 실험을 통해 TESSA가 기존 방법들보다 높은 품질의 주석을 효과적으로 생성함을 입증하였습니다.



### Revisiting Technical Bias Mitigation Strategies (https://arxiv.org/abs/2410.17433)
- **What's New**: 인공지능(AI) 분야에서 편향(bias) 및 공정성(fairness)을 개선하기 위한 노력들이 주로 기술적 해결책에 집중되어 있으며, 이번 리뷰는 특히 의료(healthcare) 환경에서 기술적 해결책의 실제 제약을 분석합니다.

- **Technical Details**: 리뷰는 편향과 공정성을 정의하는 주체, 사용할 완화(mitigation) 전략의 우선순위, AI 개발 단계에서 해결책의 효과성, 특정 인구(populations)에 대한 고려, 그리고 설계(context) 배경을 포함한 다섯 가지 주요 차원에서 구조적 분석을 제공합니다.

- **Performance Highlights**: 이 연구는 가치 중심(value-sensitive) AI 프레임워크를 제안하여 이해관계자(stakeholders)와의 소통을 강화하고, 편향 및 공정성 완화 솔루션에 그들의 가치를 반영하도록 돕는 방법을 모색합니다. 또한, 연구에서 다룬 제약을 해결하기 위한 실용적인 권장 사항들을 제시합니다.



### Navigating Noisy Feedback: Enhancing Reinforcement Learning with Error-Prone Language Models (https://arxiv.org/abs/2410.17389)
Comments:
          13 pages, 8 figures, The 2024 Conference on Empirical Methods in Natural Language Processing

- **What's New**: 이번 연구에서는 인간 피드백 대신 미리 훈련된 대규모 언어 모델(LLM)을 활용한 강화 학습의 한계를 극복하기 위한 새로운 방법론을 제안합니다. 이 접근법은 LLM이 불확실한 상태에 대해 비정보성 보상을 부여하도록 하여, 정책 학습 중 발생할 수 있는 오류를 완화합니다.

- **Technical Details**: 제안된 방법론은 LLM의 일관성 없는 출력으로 인해 발생하는 불확실성을 반영하여, 강화 학습에서 보상 모델을 개선하는 형식의 피드백 체계를 도입합니다. 연구는 잠재 기반 보상 함수(potential-based shaping function)를 이용해 보상을 정의하며, 이를 통해 에이전트는 비정보성 보상에 의존하면서도 성능을 향상시킬 수 있습니다.

- **Performance Highlights**: 이 방법론은 기존의 기준선에 비해 수렴 속도를 개선하고 정책 수익을 향상시켰으며, 특히 높은 순위 오류를 겪는 경우에도 좋은 성능을 보였습니다. 따라서 복잡한 보상 함수 후처리 없이도 결괏값을 도출할 수 있는 가능성을 보여줍니다.



### DeLLiriuM: A large language model for delirium prediction in the ICU using structured EHR (https://arxiv.org/abs/2410.17363)
- **What's New**: DeLLiriuM은 ICU에서의 섬망 예측을 위한 최초의 LLM(large language model) 기반 도구로, 104,303명의 환자 데이터를 활용하여 개발되었습니다.

- **Technical Details**: 이 연구에서는 ICU 입원 첫 24시간에 수집된 EHR(electronic health records) 데이터를 활용해 섬망 발생 확률을 예측합니다. DeLLiriuM 모델은 195개의 병원에서 수집된 데이터를 기반으로 하였으며, 세 개의 대규모 데이터베이스(eICU Collaborative Research Database, MIMIC-IV, University of Florida Health's Integrated Data Repository)를 사용했습니다.

- **Performance Highlights**: DeLLiriuM의 성능은 AUROC(area under the receiver operating characteristic curve)로 측정되었으며, 외부 검증 세트에서 0.77과 0.84의 값을 기록해 기존의 딥러닝 모델보다 우수함을 입증했습니다. 이는 77,543명의 환자, 194개의 병원에 걸쳐 이루어진 성과입니다.



### FairLoRA: Unpacking Bias Mitigation in Vision Models with Fairness-Driven Low-Rank Adaptation (https://arxiv.org/abs/2410.17358)
- **What's New**: 최근 파라미터 효율적인 미세 조정 방법인 Low Rank Adaptation (LoRA)의 발전은 대규모 기초 모델을 다양한 다운스트림 과업에 효율적으로 적응시킬 수 있는 능력으로 인해 큰 주목을 받고 있습니다. 본 논문에서는 FairLoRA라는 새로운 공정성 특정 정규화를 도입하여 데이터 하위 그룹 간의 성능 불균형을 줄이는 방법을 제안합니다.

- **Technical Details**: FairLoRA는 LoRA로 미세 조정된 모델의 공정성을 개선하기 위해 클래스 당 손실의 분산을 최소화하는 방식으로 설계된 공정성 기반의 정규화 기법입니다. 이 연구에서는 다양한 비전 모델(Bision Models)인 ViT, DiNO, CLIP을 포함하여 분포 이동(distribution shifts)이 있는 시나리오에서 FairLoRA의 효과를 평가합니다.

- **Performance Highlights**: FairLoRA는 기존의 공정성 특정 정규화를 사용한 전체 미세 조정(Full Fine-Tuning, FFT)과 비교하여 대부분의 메트릭에서 동등하거나 더 나은 성과를 보였습니다. 본 연구의 결과는 공정성을 감소시키기 위해 더 높은 랭크가 필요하지 않다는 것을 보여주며, FairLoRA가 다양한 구조와 데이터셋, LoRA 랭크를 기반으로 신뢰성 있는 공정성 개선을 가능하게 함을 나타냅니다.



### Are Large Language Models Ready for Travel Planning? (https://arxiv.org/abs/2410.17333)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)이 관광 및 환대 분야에서의 사용에 대한 성별 및 인종 편향을 분석합니다. 특히, LLM을 여행 계획 보조 도구로 사용할 때의 문제가 강조되었습니다.

- **Technical Details**: 세 가지 오픈 소스 LLM에서 생성된 여행 제안을 분석하기 위해 머신러닝(machine learning) 기술을 적용했습니다. 연구 결과, 인종 및 성별 분류기의 성능이 랜덤 확률을 상당히 초과하는 것으로 나타났습니다.

- **Performance Highlights**: LLM의 출력은 특정 인종과 성별에 연결된 문화적 기대와 일치하며, 스톱워드(stop-word) 분류 전략을 사용하여 식별할 수 있는 차이를 줄이는 데 성공했습니다. 그러나 아프리카계 미국인 및 성 소수자 그룹에 대한 환각(hallucinations) 문제가 발견되었습니다. 결론적으로, LLM은 편향이 없는 것처럼 보이는 여행 계획을 생성할 수 있으나, 추천의 정확성과 적절성을 검증하는 것이 중요합니다.



### Literature Meets Data: A Synergistic Approach to Hypothesis Generation (https://arxiv.org/abs/2410.17309)
Comments:
          30 pages, 7 figures, code link: this https URL

- **What's New**: 이 연구는 문헌(혹은 문헌 기반) 통찰력과 데이터를 통합하여 LLM(대형 언어 모델) 기반의 가설 생성을 수행하는 첫 번째 방법을 개발하였습니다. 이를 통해 문헌 기반 접근법과 데이터 기반 접근법을 결합하여 더 나은 결과를 도출할 수 있음을 보여줍니다.

- **Technical Details**: 연구에서는 LLM을 사용하여 고품질의 가설을 생성하기 위한 알고리즘을 개발하였으며, 이를 위해 HypoGeniC을 데이터 중심으로 활용했습니다. 문헌 기반의 가설 에이전트를 도입하여 지속적인 협업을 통해 가설 풀을 개선하고 유지합니다. 이는 데이터 기반의 적응성과 기존 과학 지식에 대한 기반을 동시에 보장하도록 설계되었습니다.

- **Performance Highlights**: 자동 평가 결과, 문헌과 데이터 통합 모델이 다른 기준선 대비에서 8.97% 향상된 정확도를 보였으며, 문헌 기반만의 경우보다 15.75%, 데이터 중심의 경우보다 3.37% 높은 성능을 나타냈습니다. 또한 인간 평가를 통하여 인간의 의사결정 정확도를 각각 7.44%와 14.19% 개선한 효과를 보여주었습니다.



### Advancements in Visual Language Models for Remote Sensing: Datasets, Capabilities, and Enhancement Techniques (https://arxiv.org/abs/2410.17283)
- **What's New**: 최근 ChatGPT의 성공으로 인해 인공지능(AI)에 대한 관심이 다시 고조되고 있으며, 비주얼 언어 모델(VLM)의 발전이 이 열기를 더욱 고취시키고 있습니다. VLM은 기존의 AI 접근 방식과 달리 다양한 작업을 생성 모델로 설정하여 언어와 시각 정보를 정렬함으로써 보다 도전적인 문제를 처리할 수 있게 해줍니다.

- **Technical Details**: 본 논문은 VLM과 관련된 기초 이론을 리뷰하고, 원격 감지(RS)를 위한 VLM에 의해 구축된 데이터셋과 이들이 다룬 다양한 작업을 요약합니다. 또한 VLM의 핵심 구성 요소에 따라 세 가지 주요 개선 방법으로 분류하여 이들 방법에 대한 상세한 소개와 비교를 제공합니다.

- **Performance Highlights**: VLM은 원격 감지 분야에서 지구물리학 분류, 객체 탐지 및 장면 이해와 같은 작업들에서 상당한 진전을 보여주었습니다. 최근의 VLM 기반 원격 감지 데이터 처리 기술 경향은 기존 방법들이 주로 이미지를 처리하는 데 중점을 두었던 것과 달리 VLM을 활용하여 더 다양한 복합적인 문제를 해결할 수 있는 여지를 제공합니다.



### ALTA: Compiler-Based Analysis of Transformers (https://arxiv.org/abs/2410.18077)
- **What's New**: 새로운 프로그래밍 언어 ALTA를 제안하며, ALTA 프로그램을 Transformer(트랜스포머) 가중치로 매핑할 수 있는 컴파일러를 개발했습니다.

- **Technical Details**: ALTA는 Weiss et al.(2021)의 RASP 언어와 Lindner et al.(2023)의 RASP 프로그램을 Transformer 가중치로 변환하는 컴파일러인 Tracr에서 영감을 받았습니다. ALTA는 루프를 표현할 수 있는 기능과 Universal Transformers(유니버설 트랜스포머)로 프로그램을 컴파일하는 기능 등 다양한 장점을 제공합니다.

- **Performance Highlights**: ALTA를 사용하여 패리티(parity) 및 덧셈(addition) 알고리즘을 계산하는 길이 불변(length-invariant) 알고리즘을 Transformer가 어떻게 표현할 수 있는지를 구조적으로 보여주었습니다. 또한 SCAN 벤치마크(compositional generalization tasks)에 대한 해결책을 제시하며 중간 스크래치패드 디코딩 단계를 요구하지 않습니다. ALTA 실행 추적(traces)에서 훈련하여 보다 세분화된 감독 신호를 제공함으로써 다양한 알고리즘의 학습 가능성과 데이터 가용성(data availability) 및 모델링 결정(modelling decisions) 관련 추가 실험과 이론적 분석을 수행할 수 있게 되었습니다.



### Leveraging Skills from Unlabeled Prior Data for Efficient Online Exploration (https://arxiv.org/abs/2410.18076)
Comments:
          23 pages, 10 figures

- **What's New**: 본 연구에서는 unlabeled prior trajectory 데이터를 활용하여 효율적인 exploration 전략을 학습하는 방법에 대해 설명합니다. 새롭게 제안된 방법인 SUPE(Skills from Unlabeled Prior data for Exploration)는 잠재적으로 미리 학습한 low-level skills를 조합하여 고수준의 정책을 학습합니다.

- **Technical Details**: SUPE는 variational autoencoder (VAE)를 사용하여 low-level skills를 추출하고, optimistic reward model을 이용해 unlabeled trajectories를 pseudo-label링합니다. 이후, 이 transformed data를 online RL에서 off-policy 데이터로 활용하여 더 높은 수준의 exploration policy를 학습합니다.

- **Performance Highlights**: SUPE는 긴 시간 동안의 sparse reward tasks를 해결하는 데 있어 이전의 전략보다 높은 성능을 보여줍니다. 본 연구의 실험 결과에 따르면, SUPE는 unlabeled prior data를 활용하여 sparse reward 신호를 더 빠르게 찾고 학습을 효율적으로 촉진합니다.



### TP-Eval: Tap Multimodal LLMs' Potential in Evaluation by Customizing Prompts (https://arxiv.org/abs/2410.18071)
- **What's New**: 최근 다중 모달 대규모 언어 모델(multimodal large language models, MLLMs)의 성능 평가에서의 의존성을 해결하기 위한 새로운 평가 프레임워크인 TP-Eval이 제안되었습니다.

- **Technical Details**: TP-Eval은 모델별 최적의 prompts를 자동으로 커스터마이즈하여 평가 시 발생할 수 있는 편향을 줄이는 방법론을 구현하고 있습니다. 이 과정에서 기존의 프롬프트 최적화와는 달리 이미지와 텍스트 정보를 통합하여 MLLMs에 최적화된 방식으로 진행됩니다.

- **Performance Highlights**: TP-Eval을 활용한 실험을 통해 MLLMs의 진정한 성능을 보다 정확하게 평가할 수 있으며, 여러 모델 간의 평가 편향 현상을 효과적으로 완화하는 것으로 나타났습니다.



### Training Free Guided Flow Matching with Optimal Contro (https://arxiv.org/abs/2410.18070)
- **What's New**: 본 논문에서는 OC-Flow라는 새로운 프레임워크를 도입하여 최적 제어(optimal control) 이론에 기반한 유도 흐름_matching(guided flow matching)을 수행합니다. 이는 기존의 흐름 모델을 개선하고, 더 복잡한 기하학적 구조인 SO(3)에서의 사용 가능성을 제시합니다.

- **Technical Details**: OC-Flow는 사전 훈련된 ODE 기반 사전(prior)으로부터의 제어 생성을 최적 제어 문제로 공식화하며, 각 단계에서 제어 항을 통해 ODE 경로를 유도합니다. 이 프레임워크는 Euclidean 및 SO(3) 공간 모두에서 유효하며, 이론적 수렴 보장을 제공합니다. 수렴 분석과 함께 OC-Flow의 간단한 알고리즘과 여러 변수를 다룹니다.

- **Performance Highlights**: OC-Flow는 다양한 실험에서 뛰어난 성능을 보였습니다. 텍스트를 기반으로 한 이미지 조작, 조건부 분자 생성을 위한 제어 및 모든 원자가 포함된 펩타이드 설계에서의 효율성을 입증하였고, 이를 통해 과학적 응용에서도 주목할 만한 성과를 거두었습니다.



### Beyond position: how rotary embeddings shape representations and memory in autoregressive transfomers (https://arxiv.org/abs/2410.18067)
- **What's New**: 이 논문은 Rotary Positional Embeddings (RoPE)가 Transformer 모델의 내부 동역학에 미치는 영향을 탐구하며, 특히 RoPE가 토큰 임베딩에 위치 의존적인 회전을 도입하여 모형의 정보 보존 및 시간적 모델링 능력에 어떻게 영향을 미치는지를 분석합니다.

- **Technical Details**: RoPE는 회전 행렬을 사용하여 임베딩에 상대적 위치 의존성을 도입하고, 이를 통해 토큰의 위치 정보를 효과적으로 인코딩합니다. 본 연구는 RoPE의 주파수와 상관된 회전 성질, 비선형 활성화 함수와의 상호작용 및 위상 정렬에 의한 보강적 및 파괴적 간섭 패턴을 조사합니다.

- **Performance Highlights**: RoPE에 의해 유도된 위상 정렬은 활성화를 증폭하고 주의(attention)를 더욱 날카롭게 형성하는 반면, 위상 불일치는 활성화를 약화시키고 위치 패턴에 대한 집중을 방해합니다. 이러한 발견은 모델의 동작에서 주파수 성분의 중요성을 강조하며, 전통적인 분석을 넘어서는 새로운 통찰력을 제공합니다.



### SPIRE: Synergistic Planning, Imitation, and Reinforcement Learning for Long-Horizon Manipulation (https://arxiv.org/abs/2410.18065)
Comments:
          Conference on Robot Learning (CoRL) 2024

- **What's New**: SPIRE 시스템이 제안되었습니다. 이 시스템은 작업을 작고 학습 가능한 하위 문제로 분해하고, 모방 학습(imitation learning)과 강화 학습(reinforcement learning)을 결합하여 두 기법의 장점을 극대화합니다.

- **Technical Details**: SPIRE는 Task and Motion Planning (TAMP)을 활용하여 각 작업을 지역적인 조작(segment)으로 분해한 다음, 이 조작들을 모방 학습으로 먼저 학습하고, 후에 강화 학습을 통해 조정(finetuning)합니다. 또한, sparse rewards로부터 RL 기반의 조정을 가능하게 하는 여러 가지 전략이 개발되었습니다.

- **Performance Highlights**: SPIRE는 9개의 도전적인 조작 작업에서 평균 성공률 87.8%를 기록하며 이전의 접근 방식에 비해 35%에서 50% 더 높은 성능을 보였습니다. 이 시스템은 인간 시연 데이터의 필요성을 6배 줄이고, 작업 완료 효율성을 두 배로 향상시킵니다.



### Key Algorithms for Keyphrase Generation: Instruction-Based LLMs for Russian Scientific Keyphrases (https://arxiv.org/abs/2410.18040)
Comments:
          The 12th International Conference on Analysis of Images, Social Networks and Texts (AIST'2024)

- **What's New**: 이번 연구는 러시아어 과학 논문의 요약에서 키워프(Keyword) 생성에 대한 프롬프트 기반 접근 방식을 평가하고, 기존의 수퍼바이즈드(Supervised) 및 언슈퍼바이즈드(Unsupervised) 방법과 비교한 것입니다. 또한, 제로샷(Zero-shot) 및 몇 샷(Few-shot) 프롬프트 방식의 성능을 분석하고, 효과적인 키워프 선택 전략을 제안합니다.

- **Technical Details**: 연구에서는 Math&CS 데이터셋을 사용하였으며, 프롬프트 기반의 키워프 생성 방법을 세 가지 오픈소스 LLM(대형 언어 모델)으로 실시하였습니다. Saiga, Vikhr, Mistral 모델이 사용되었고, 이들 모델은 각각 러시아어에 맞게 특별히 조정되었습니다.

- **Performance Highlights**: 연구 결과, 프롬프트 기반 방법이 기존의 언슈퍼바이즈드 방법과 비교하여 뛰어난 성과를 보였습니다. 인간 평가를 통해 생성된 키워프의 질을 측정하였으며, 단순한 텍스트 프롬프트를 사용하여도 기존 기법보다 우수한 결과를 나타냈습니다.



### Cross-lingual Transfer of Reward Models in Multilingual Alignmen (https://arxiv.org/abs/2410.18027)
- **What's New**: 본 논문은 인간 피드백을 통한 강화 학습(Reinforcement Learning with Human Feedback, RLHF)에서 여러 언어에서 훈련된 보상 모델(Reward Model, RM)의 교차 언어 전이 가능성을 조사합니다. 특히, 영어 데이터로 훈련된 RM이 다른 언어의 RM보다 우수한 성능을 보인다는 점을 강조합니다.

- **Technical Details**: 이 연구에서는 영어 RM이 다국어로 사전 훈련된 언어 모델(MLM) 위에 구축되었을 때 강력한 교차 언어 전이를 나타난다는 것을 보여줍니다. 실험 결과, 영어 RM이 Multilingual RewardBench에서 목표 언어 RM보다 평균 3~4% 더 높은 성능을 기록했습니다. 우리는 영어가 MLM의 표현을 잘 보존하며, 여러 언어에 대한 이해도가 강하다는 두 가지 이유로 설명합니다.

- **Performance Highlights**: 최종적으로, 연구 결과는 영어 RM이 다국어 지침 준수 능력을 향상시킬 수 있음을 입증하며, 네 가지 비영어 언어에서 9.5%의 평균 승률 증가를 보여주었습니다. 또한, 전통적인 RM과 비교하여 생성적 RM은 성능이 저조하다는 점도 강조되었습니다.



### Federated Transformer: Multi-Party Vertical Federated Learning on Practical Fuzzily Linked Data (https://arxiv.org/abs/2410.17986)
- **What's New**: 이 논문에서는 다수의 당사자가 모호한 식별자를 통해 협업하여 모델을 훈련할 수 있도록 하는 새로운 연합 학습 프레임워크인 Federated Transformer (FeT)를 제안합니다.

- **Technical Details**: FeT는 모호한 식별자를 데이터 표현에 인코딩하고, 다양한 당사자에 분산된 transformer 아키텍처를 사용합니다. 이 과정에서 세 가지 새로운 기술을 도입하여 성능을 향상시킵니다. 또한, 차별적 개인 정보 보호(differential privacy)와 안전한 다자간 계산(secure multi-party computation)을 통합한 다자간 개인 정보 보호 프레임워크를 개발하여 로컬 표현을 보호합니다.

- **Performance Highlights**: FeT는 50개의 당사자 규모에서 기존 모델보다 최대 46% 높은 정확도를 보여줍니다. 또한, 두 당사자 모호 VFL 설정에서도 성능과 개인 정보 보호가 향상되었습니다.



### Dynamic Spectrum Access for Ambient Backscatter Communication-assisted D2D Systems with Quantum Reinforcement Learning (https://arxiv.org/abs/2410.17971)
Comments:
          12 pages, 7 figures

- **What's New**: 이 논문에서는 이동 사용자에 의해 점유되었을 때 D2D 장치가 주변 RF 신호를 역산란(backscatter)하여 데이터를 전송할 수 있게끔 ambient backscatter communication (AmBC) 기술을 D2D 장치에 통합하고, 이를 통해 스펙트럼 접근 정책을 최적화하고자 합니다.

- **Technical Details**: 이 논문은 Markov 결정 프로세스(Markov Decision Process)를 사용하여 사용자 이동성 및 무선 환경으로 인한 시스템 동향과 불확실성을 모델링하며, 심층 Q 학습(deep Q-learning)을 통해 장기 평균 처리량을 극대화하는 최적 스펙트럼 접근 정책을 찾습니다. 또한, 양자 강화 학습(quantum reinforcement learning) 알고리즘을 개발하여 더 빠른 수렴률과 감소된 훈련 복잡성을 달성합니다.

- **Performance Highlights**: 광범위한 시뮬레이션 결과, 제안된 양자 RL 접근 방식은 기존 DRL 방법에 비해 훨씬 더 빠른 수렴 속도와 적은 훈련 파라미터를 통해 D2D 장치의 평균 처리량을 크게 향상시킬 수 있음을 보여줍니다.



### Closed-form merging of parameter-efficient modules for Federated Continual Learning (https://arxiv.org/abs/2410.17961)
- **What's New**: 본 논문은 모델 병합 기술에 대해 다루며, 여러 개의 모델을 통합하여 성능과 확장성을 유지하는 방법을 제안합니다. 기존의 Low-Rank Adaptation(LoRA) 기술을 기반으로, LoRA 모듈을 닫힌 형태(closed form)로 병합하는 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 Low-rank Regression Mean (LoRM)이라 명명되며, 각 LoRA 행렬을 순차적으로 학습하는 교대 최적화(alternating optimization) 전략을 사용합니다. 이로 인해 데이터 분포와 작업(task)에 따라 모델 응답을 조정할 수 있습니다. 주로 Federated Class-Incremental Learning(FCIL) 방식에 중점을 두어 실험을 진행했습니다.

- **Performance Highlights**: LoRM 방법을 적용한 결과, 여러 개의 FCIL 시나리오에서 최첨단(state-of-the-art) 성능을 보여주었으며, 다양한 데이터셋 및 데이터 분포 조건에서도 효과적이라는 것을 확인하였습니다.



### MCUBERT: Memory-Efficient BERT Inference on Commodity Microcontrollers (https://arxiv.org/abs/2410.17957)
Comments:
          ICCAD 2024

- **What's New**: 이 논문에서는 MCUBERT를 제안하여 BERT와 같은 언어 모델을 당량의 마이크로컨트롤러 유닛(MCU)에서 사용할 수 있도록 합니다. 이는 네트워크와 스케줄링의 공동 최적화를 통해 가능해졌습니다. MCUBERT는 경량의 BERT 모델을 상품화된 MCU에서 초기로 지원합니다.

- **Technical Details**: MCUBERT는 임베딩 테이블의 저장 병목 현상을 해결하기 위한 MCU 인식 클러스터 저차원 근사(clustered low-rank approximation) 기반의 두 단계 신경망 아키텍처 탐색(Two-stage Neural Architecture Search, NAS) 알고리즘을 제안합니다. 또한, Transformer 블록의 실행 메모리 사용을 줄이기 위해 미세하게 조정된 스케줄링 전략을 개발했습니다. MCUBERT는 BERT-tiny 및 BERT-mini 모델의 매개변수 크기를 각각 5.7배와 3.0배 줄이고, 실행 메모리는 3.5배와 4.3배 감소시켰습니다.

- **Performance Highlights**: MCUBERT는 1.5배의 지연(latency) 감소를 달성하며, 256KB의 메모리로 512개 이상의 토큰을 동시에 처리할 수 있게 해줍니다. 이는 지금까지 SOTA 인퍼런스 엔진에 비해 성능이 개선된 것입니다.



### SimRAG: Self-Improving Retrieval-Augmented Generation for Adapting Large Language Models to Specialized Domains (https://arxiv.org/abs/2410.17952)
Comments:
          Work in Progress

- **What's New**: 이번 논문에서는 SimRAG라는 자가 학습(self-training) 접근 방식을 통해 복잡한 과학 및 의학 분야에 적합한 Retrieval-augmented generation (RAG) 시스템을 제안합니다. 이는 LLM (Large Language Model)이 질문 생성과 질문 응답 능력을 동시에 갖출 수 있도록 하는 방법입니다.

- **Technical Details**: SimRAG는 우선 다양한 데이터 세트에서 LLM을 세부 조정(fine-tuning)한 후, 라벨이 없는 도메인 관련 데이터에서 다양한 질문을 생성하도록 유도하는 방법으로 구성됩니다. 생성된 질문 중 질 높은 예시를 선별할 수 있는 추가적인 필터링 전략도 함께 사용됩니다.

- **Performance Highlights**: 11개 데이터 세트를 활용한 실험 결과, SimRAG는 기존의 기법들에 비해 성능이 1.2%에서 8.6% 향상된 것으로 확인되었습니다.



### Optimizing Travel Itineraries with AI Algorithms in a Microservices Architecture: Balancing Cost, Time, Preferences, and Sustainability (https://arxiv.org/abs/2410.17943)
Comments:
          18 pages, 6 figures

- **What's New**: 이번 연구는 마이크로서비스 아키텍처(microservices architecture)에서 인공지능(AI) 알고리즘의 구현이 여행 일정 최적화에 미치는 영향을 다룹니다. 비용(cost), 시간(time), 사용자 선호(user preferences), 그리고 환경 지속 가능성(environmental sustainability)을 향상시키는 방식을 제시합니다.

- **Technical Details**: 이 시스템은 비용 예측을 위한 머신러닝 모델(machine learning models), 여정을 최적화하기 위한 유전자 알고리즘(genetic algorithm), 그리고 지속 가능성 검사를 위한 휴리스틱(heuristics)을 사용합니다. 주 평가 요소로는 지연(latency), 사용자 선호 만족도, 비용 및 환경 문제에 대한 고려가 포함됩니다.

- **Performance Highlights**: 실험 결과에 따르면, 1000명의 동시 사용자를 대상으로 평균 응답 시간은 4.5초이며, 사용자 선호도 정확도는 92%에 달합니다. 제공된 여행의 95%가 사용자가 설정한 예산 내에서 이루어졌고, 제안된 여행 계획의 60%는 환경 친화적인 옵션(green options)을 포함하고 있어 전통적인 여행 계획에 비해 평균 15% 낮은 탄소 배출량(carbon emissions)을 기록했습니다. 또한, 시스템은 기능적 가용성(functional availability) 99.9%를 유지하여 요구 사항을 초과해도 서비스 제공이 가능합니다.



### Multi-Continental Healthcare Modelling Using Blockchain-Enabled Federated Learning (https://arxiv.org/abs/2410.17933)
Comments:
          Accepted by IEEE Global Blockchain Conference

- **What's New**: 본 논문에서는 의료 분야에서 데이터 공유의 어려움을 극복하기 위한 프레임워크를 제안합니다. 이 프레임워크는 블록체인 기반의 federated learning을 사용하여 개인 치료 데이터 공유 없이 다중 대륙(유럽, 북미, 아시아)의 데이터를 활용하여 당뇨병의 혈당 관리 모델을 검증합니다.

- **Technical Details**: 제안된 Multi-Continental Glucose Prediction (MCGP) 프레임워크는 각 참여 기관이 민감한 건강 데이터를 직접 공유하지 않고도 모델 훈련에 기여할 수 있도록 설계되었습니다. 각 참여자는 모델 파라미터를 교환하여 건강 데이터의 프라이버시를 유지합니다. 이는 또한 기여자를 유도하고 악의적 행위를 탐지하기 위한 인센티브 메커니즘을 포함합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 제한된 개인 데이터로 훈련된 모델보다 훨씬 높은 예측 정확도를 제공하며, 중앙 집중식 데이터셋의 결과와 유사하거나 더 나은 성능을 보였습니다. 이 연구는 국제적인 의료 프로젝트 협력의 가능성을 열어줍니다.



### Addressing Asynchronicity in Clinical Multimodal Fusion via Individualized Chest X-ray Generation (https://arxiv.org/abs/2410.17918)
Comments:
          Accepted by NeurIPS-24

- **What's New**: DDL-CXR는 EHR과 CXR 간의 비동기성 문제를 해결하고 개인화된 CXR 이미지를 생성하여 임상 예측의 성능을 향상시키는 혁신적인 방법입니다.

- **Technical Details**: DDL-CXR는 잠재적 확산 모델(latent diffusion model)과 변분 오토인코더(variational auto-encoder)를 이용하여 환자 맞춤형 CXR 이미지를 생성합니다. 환자의 이전 CXR 이미지를 참조 이미지로 사용하고 EHR 타임 시리를 인코딩하기 위해 Transformer 모델을 활용하여 질병의 진행 과정을 효과적으로 캡처합니다. 이는 EHR과 CXR 간의 상호작용을 개선합니다.

- **Performance Highlights**: MIMIC 데이터셋을 사용한 실험에서 DDL-CXR는 기존의 방법에 비해 다중 모드 클리닉 예측과 개별 CXR 생성에서 일관되게 뛰어난 성능을 보였습니다.



### Reinforcement Learning under Latent Dynamics: Toward Statistical and Algorithmic Modularity (https://arxiv.org/abs/2410.17904)
- **What's New**: 본 논문은 "일반적인"latent dynamics 하에서 강화 학습의 이해와 기법을 제시합니다. 특히, 높은 차원의 관찰이 있을 때 통계적 요구사항과 알고리즘의 원리를 탐구합니다.

- **Technical Details**: 저자들은 주로 두 가지 기법을 통해 강화 학습을 개선하였습니다. 첫째, latent dynamics에 대한 역사적 관찰(hindsight observations)을 사용할 수 있는 경우와 둘째, 자기 예측적인 latent 모델(self-predictive latent models)을 추정할 수 있는 경우를 다룹니다. 통계적 관점에서, 기존의 기능 근사(function approximation)에 기반한 강화 학습 문제가 복잡한 관찰과 결합될 때 비효율적이라는 부정적 결과를 도출하였습니다.

- **Performance Highlights**: 이 연구는 latent pushforward coverability라는 조건을 식별하여 통계적 계산 가능성을 조건화하며, 여러 관측으로부터 latent MDP를 효과적으로 처리할 수 있는 알고리즘의 맵핑을 제시하였습니다. 이 연구는 latent dynamics 아래의 강화 학습에 대한 통합 이론의 첫 번째 단계로 자리잡고 있습니다.



### Understanding Layer Significance in LLM Alignmen (https://arxiv.org/abs/2410.17875)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)의 정렬 프로세스에서 가장 중요한 층을 식별하기 위한 새로운 접근 방식인 ILA(Important Layers for Alignment)를 제안합니다. 이 방법은 LoRA 알고리즘의 각 가중치 행렬에 대한 이진 마스크를 학습하여 각 층의 중요도를 나타냅니다.

- **Technical Details**: ILA는 층의 중요도를 이진 마스크로 표현하는 것으로, 마스크의 값이 0이면 해당 층의 영향이 미미함을, 1이면 해당 층이 정렬 프로세스에 중요하다는 것을 나타냅니다. 다양한 정렬 데이터셋에서 일관된 중요층 순위를 발견했으며, 25%의 중요하지 않은 층을 동결함으로써 모델의 성능이 향상된다는 것을 입증했습니다.

- **Performance Highlights**: 연구 결과는 가장 중요한 10-30%의 층만을 미세 조정하여도 전체 선형 층을 미세 조정한 것과 유사한 성능을 달성할 수 있음을 보여줍니다. 또한, QLoRA와 결합할 경우 30-75%의 핵심 층만 조정해서도 성능을 유지하거나 향상시킬 수 있으며, 자원 비용을 크게 줄일 수 있음을 강조합니다.



### ROCKET-1: Master Open-World Interaction with Visual-Temporal Context Prompting (https://arxiv.org/abs/2410.17856)
- **What's New**: 본 논문은 비전-언어 모델(Vision-Language Models, VLMs)을 활용하여 개방형 세계 환경에서 존재하는 결정을 내리기 위한 도전과제를 다룹니다. 특히, 개체(entities)와 추상 개념(abstract concepts) 간의 원활한 연결이 어려운 문제를 해결하기 위해 'visual-temporal context prompting'라는 새로운 통신 프로토콜을 제안합니다.

- **Technical Details**: 제안된 방법은 객체 분할(object segmentation)을 과거 및 현재 관찰에서 활용하여 정책 모델(policy models)과의 상호작용을 유도합니다. ROCKET-1이라는 저수준 정책(low-level policy)은 시각적 관찰 및 분할 마스크를 연결하여 행동을 예측합니다. 이는 transformer를 사용해 관찰 간의 의존성을 모델링하고, SAM-2를 통해 실시간 객체 추적 기능을 제공합니다.

- **Performance Highlights**: Minecraft 환경에서 진행된 실험을 통해 제안된 방법이 기존 방법으로는 수행할 수 없었던 복잡한 작업들을 성공적으로 완료할 수 있음을 보여주었습니다. 이는 공간 이해(spatial understanding)에 크게 의존하는 창의적인 작업들을 해결할 수 있는 잠재력을 시사합니다.



### TAGE: Trustworthy Attribute Group Editing for Stable Few-shot Image Generation (https://arxiv.org/abs/2410.17855)
Comments:
          Accepted by International Conference on Signal Processing Systems Conference

- **What's New**: TAGE라는 새로운 이미지 생성 네트워크를 소개하며, 세 가지 주요 모듈인 Codebook Learning Module (CLM), Code Prediction Module (CPM), Prompt-driven Semantic Module (PSM)으로 구성되어 있음.

- **Technical Details**: CPM 모듈은 범주 비이성(attribute)-독립 속성의 의미적 차원을 탐구하여 이를 이산 코드북에 캡슐화하고, 주요 요점은 이미지가 속성의 조합으로 이루어져 있다는 것. PSM 모듈은 Transformer 구조에 통합된 의미적 단서를 생성하여 편집하고자 하는 속성에 대한 모델의 이해를 향상시킴.

- **Performance Highlights**: Animal Faces, Flowers, VGGFaces 데이터셋을 활용한 실험 결과, 제안된 방법이 다른 few-shot 이미지 생성 기술에 비해 우수한 성능과 높은 안정성을 보여줌.



### The Probabilistic Tsetlin Machine: A Novel Approach to Uncertainty Quantification (https://arxiv.org/abs/2410.17851)
Comments:
          12 pages, 5 figures, 6 tables, accepted and presented at ICAAI 2024, London

- **What's New**: 본 연구에서는 기존 Tsetlin 기계(TM)의 불확실성 정량화 문제를 해결하기 위해 확률론적 Tsetlin 기계(Probabilistic Tsetlin Machine, PTM) 프레임워크를 도입합니다. PTM은 각 Tsetlin 자동자(TA)의 상태에 대한 확률을 학습하고, Bayesian 신경망과 유사하게 상태를 샘플링하여 추론을 수행합니다.

- **Technical Details**: PTM은 기존 TM의 피드백 테이블을 사용하여 상태 확률을 업데이트하며, 이는 Type I 및 Type II 피드백을 포함합니다. PTM은 다양한 TM 변형에 적응할 수 있으며, 인공지능(AI) 분야의 불확실성을 효율적으로 정량화할 수 있는 전반적으로 신뢰할 수 있는 접근 방식을 제공합니다.

- **Performance Highlights**: 실험 결과, PTM은 불확실성 정량화에서 우수한 성능을 보았으며, 특히 결정 경계를 구분하고 높은 불확실성 지역을 식별하는 데 있어 효과적입니다. Iris 데이터셋을 통한 다중 클래스 분류에서는 예측 엔트로피(predictive entropy)와 기대 보정 오류(expected calibration error) 측면에서 경쟁력 있는 성능을 입증하였습니다.



### PGDiffSeg: Prior-Guided Denoising Diffusion Model with Parameter-Shared Attention for Breast Cancer Segmentation (https://arxiv.org/abs/2410.17812)
- **What's New**: 본 논문에서는 유방암 의학적 이미지 분할을 위해 새로운 모델 PGDiffSeg(파라미터 공유 주의 메커니즘을 가진 사전 가이드 확산 제거 모델)를 제안합니다. 이 모델은 가우시안 노이즈로부터 영향을 받은 영역을 정확하게 복구하는데 초점을 맞추고 있습니다.

- **Technical Details**: PGDiffSeg는 두 개의 파이프라인(노이즈 처리 및 의미 정보 처리)을 설계하고, 다층에서 이를 통합하는 파라미터 공유 주의 모듈(PSA)을 제안하여 분할 성능을 향상시킵니다. 이 통합을 통해 다중 레벨에서 의미 세부정보를 포함할 수 있고, 사전 지식을 이용한 가이드 전략을 도입하여 모델의 종양 위치 탐지 능력을 향상시킵니다.

- **Performance Highlights**: 모델의 효과성은 다수의 실험을 통해 입증되었으며, 기존의 최신 방법들보다 우수한 성능을 보였습니다. PGDiffSeg는 유방암 이미지 분할 연구에 적합한 유연한 확산 제거 방법으로 자리매김하고 있습니다.



### OmniFlatten: An End-to-end GPT Model for Seamless Voice Conversation (https://arxiv.org/abs/2410.17799)
Comments:
          Work in progress

- **What's New**: 본 논문에서는 자연스러운 대화를 모델링하기 위한 새로운 End-to-End GPT 기반 모델 OmniFlatten을 소개합니다. 이 모델은 사용자와 시스템 간의 실시간으로 전통적인 turn-based 시스템보다 뛰어난 full-duplex 대화를 가능하게 합니다.

- **Technical Details**: OmniFlatten은 multi-stage 포스트 트레이닝 방법론을 통해 텍스트 기반 LLM의 구조를 변경하지 않고도 speech-text 대화 모델로 발전시킵니다. 데이터 표준화를 위해 flattening 작업을 수행하여 다양한 모달리티와 작업 간의 학습 방식을 통합합니다. 훈련 프로세스는 modality alignment, half-duplex dialogue learning 및 full-duplex dialogue learning의 세 단계로 구성됩니다.

- **Performance Highlights**: OmniFlatten은 ASR과 TTS 작업을 통한 조정 및 훈련을 통해 명확한 대화 품질을 제공합니다. 생성된 대화의 평균 응답 시간은 시스템이 턴을 가질 경우 160ms, 사용자가 턴을 가질 경우 805ms로 나타났습니다. 결과적으로 모델은 모달리티 정렬 및 half-duplex 학습 단계에서 모델의 full-duplex 대화 능력을 개선하는 성과를 보였습니다.



### Enhancing Federated Learning Convergence with Dynamic Data Queue and Data Entropy-driven Participant Selection (https://arxiv.org/abs/2410.17792)
Comments:
          The Journal is submitted to IEEE Transactions in the Internet of Things

- **What's New**: 이번 연구에서는 비독립적이고 비동일 분포(non-IID) 데이터의 통계적 복잡성을 해결하기 위한 새로운 접근 방식을 제안합니다. 이를 통해 동적 데이터 큐 기반 연합 학습(DDFL)을 이용하여 데이터 샘플링을 최적화하고 정확도를 크게 향상시킵니다.

- **Technical Details**: DDFL 모델은 각 장치에서 각 훈련 라운드마다 서버에서 보관된 데이터의 일부(10%)를 동적으로 분배하여 훈련합니다. 데이터 엔트로피(Data Entropy) 메트릭스를 활용해 장치 선택과 집계를 개선하며, 유클리드 거리로 측정한 데이터 분포 간의 편향 항(δ_k)을 감소시킵니다.

- **Performance Highlights**: MNIST 데이터셋에서 약 5%, CIFAR-10에서 약 18%, CIFAR-100에서 약 20%의 정확도를 보이며, 기존 최첨단(SOTA) 알고리즘을 능가하는 결과를 얻었습니다.



### Large Language Models Engineer Too Many Simple Features For Tabular Data (https://arxiv.org/abs/2410.17787)
Comments:
          Preprint

- **What's New**: 본 논문은 대형 언어 모델(LLM)의 기능 공학에 대한 편향(bias) 문제를 다루고 있으며, 이를 통해 LLM이 제안하는 자동화된 데이터 과학 기능이 얼마나 효과적인지를 평가합니다.

- **Technical Details**: 이 연구에서는 네 가지 LLM(GPT-4o-mini, Gemini-1.5-flash, Llama3.1-8B, Mistral7B-v0.3)을 대상으로 27개의 표 형식 데이터셋에서 다양한 연산자(Add, GroupByThenMean 등)의 빈도를 분석하여 편향을 평가합니다. 또한, 자동 블랙 박스(feature engineering) 방법인 OpenFE와의 비교를 통해 결과를 도출하였습니다.

- **Performance Highlights**: 결과적으로 LLM은 간단한 연산자(Add)에 더 편향되어 있으며, 이로 인해 예측 성능이 저하될 수 있음을 나타냅니다. 특히, 두 대형 모델인 GPT-4o-mini와 Gemini-1.5-flash는 이러한 경향이 뚜렷하게 나타났으며, Llama3.1-8B와 Mistral7B-v0.3는 상대적으로 덜 영향을 받았습니다. 전반적으로 자동 블랙 박스 방법인 OpenFE가 가장 높은 예측 성능을 발휘했습니다.



### Scaling Robot Policy Learning via Zero-Shot Labeling with Foundation Models (https://arxiv.org/abs/2410.17772)
Comments:
          Project Website at this https URL

- **What's New**: NILS(Natural language Instruction Labeling for Scalability)는 인간의 개입 없이 긴 로봇 데이터에서 자연어 지시를 자동으로 라벨링하는 새로운 시스템입니다. 이는 비용과 데이터 품질 문제를 해결합니다.

- **Technical Details**: NILS는 세 단계로 구성됩니다: (1) 장면에서 객체 식별, (2) 객체 중심의 장면 주석 생성, (3) 키 상태 탐지 및 라벨 생성. NILS는 사전 훈련된 비전-언어 모델(Vision-Language Models) 및 대형 언어 모델(Large Language Model)을 사용하여 로봇 행동 데이터를 효과적으로 라벨링합니다.

- **Performance Highlights**: NILS는 115,000개 경로를 라벨링하여 430시간 이상의 로봇 데이터를 처리할 수 있으며, 다른 기존 시스템보다 더 높은 품질의 결과를 제공합니다. 이를 통해 언어 조건 부조작 정책을 훈련할 수 있는 구조화된 데이터셋이 생성됩니다.



### Beyond Backpropagation: Optimization with Multi-Tangent Forward Gradients (https://arxiv.org/abs/2410.17764)
- **What's New**: 이 논문은 multi-tangent forward gradients에 대한 심층 분석을 제공하고 여러 접선의 forward gradients를 orthogonal projections 기반으로 결합하는 향상된 접근 방식을 소개합니다.

- **Technical Details**: 최근 제안된 forward gradient 기법은 방향 도함수를 사용하여 그래디언트를 근사화하며, 이 논문은 다수의 접선을 활용하는 것이 그래디언트 근사 품질과 최적화 성능을 향상시킬 수 있음을 보여줍니다.

- **Performance Highlights**: 여러 작업에서 접선 수를 증가시키면 근사 품질과 최적화 성능이 개선되는 것을 입증하였으며, 이러한 접근 방식이 최신 아키텍처에 확장될 수 있는 가능성을 보여줍니다.



### Escaping the Forest: Sparse Interpretable Neural Networks for Tabular Data (https://arxiv.org/abs/2410.17758)
- **What's New**: 본 논문은 새로운 접근법을 통해 신경망의 희소성(sparsity)을 효과적으로 도입하고, 이를 통해 생물학적 데이터셋에 대한 성능을 개선하는 방법을 제안합니다. 특히, attention mechanism을 활용하여 tabular 데이터의 중요 특성을 효과적으로 분석할 수 있는 방법을 제시하였습니다.

- **Technical Details**: Sparse TABular NET (sTAB-Net)은 tabular 데이터의 특성 중요도를 포착하기 위해 attention mechanism을 사용하는 신경망입니다. 이 모델은 이전 지식을 활용하여 희소성을 부여할 수 있는 방법을 정의하고, 이로 인해 모델의 투명성과 해석 가능성을 강화합니다. 연구에서는 다양한 데이터셋에서의 성능을 비교하며, decision tree 모델과의 실질적인 비교 결과를 포함하고 있습니다.

- **Performance Highlights**: sTAB-Net은 tree 기반 모델보다 우수한 성능을 보였으며, SHAP와 같은 기존 후처리 방법보다도 더 효율적으로 특성을 식별할 수 있음을 입증하였습니다. 임상 데이터와 같은 복잡한 데이터셋에서 안정적으로 높은 성능을 기록하며, 하이퍼파라미터 최적화 없이도 경쟁력 있는 결과를 나타냈습니다.



### VISAGE: Video Synthesis using Action Graphs for Surgery (https://arxiv.org/abs/2410.17751)
Comments:
          Accepted at MICCAI 2024 Embodied AI and Robotics for HealTHcare (EARTH) Workshop

- **What's New**: 이 논문은 외과 영상 생성(future video generation in laparoscopic surgery)이라는 새로운 작업을 도입하고, 이를 통해 기존 외과 데이터의 한계를 극복하고 다양한 응용 프로그램에 기여할 수 있는 방법을 제시합니다.

- **Technical Details**: 제안된 방법인 VISAGE(VIdeo Synthesis using Action Graphs for Surgery)는 외과 절차의 순차적인 특성을 포착하기 위해 액션 장면 그래프(action scene graphs)의 힘을 활용하며, 확산 모델(diffusion models)을 사용하여 시간적으로 일관된 비디오 시퀀스를 합성(synthesize)합니다. 이 모델은 단일 초기 프레임과 액션 그래프 트리플(action graph triplets)을 기반으로 미래 프레임을 예측합니다.

- **Performance Highlights**: VISAGE는 CholecT50 데이터셋에서 높은 충실도의 비디오를 생성하며, 여러 메트릭과 정성 평가에서 기존 방법들을 초월하는 성능을 보여줍니다.



### Learning Versatile Skills with Curriculum Masking (https://arxiv.org/abs/2410.17744)
Comments:
          NeurIPS 2024 poster, 21 pages, 7 figures

- **What's New**: 이 논문에서는 Offline Reinforcement Learning(RL)에서 다양한 downstream tasks를 위한 유연한 추론을 가능하게 하는 masked prediction의 중요성을 강조하며, CurrMask라는 새로운 curriculum masking pretraining 방식을 제안합니다. 이 방법은 여러 복잡성 수준의 기술 학습을 균형있게 조절하는 데 중점을 두고 있습니다.

- **Technical Details**: CurrMask는 블록 단위의 masking과 curriculum 기반 접근 방식을 통해 pretraining 중 다양한 masking scheme의 순서를 조정합니다. 블록 단위 마스킹 기법은 모델이 지역적 상관 관계보다 전역적 의존성을 우선시하도록 강제합니다. 이러한 접근은 장기 종속성을 효과적으로 캡처하는 데 도움을 줍니다.

- **Performance Highlights**: CurrMask는 zero-shot skill prompting, goal-conditioned planning, 그리고 offline RL 작업에서 경쟁력 있는 fine-tuning 성능을 보여줍니다. 이상의 실험을 통해 CurrMask가 다양한 복잡성을 가진 기술을 점진적으로 습득함을 입증했습니다.



### Emotion Recognition with Facial Attention and Objective Activation Functions (https://arxiv.org/abs/2410.17740)
- **What's New**: 이 논문에서는 Facial Emotion Recognition (FER) 작업을 수행하기 위해 CNN 비전 기반 모델(VGGNet, ResNet, ResNetV2)에 채널 및 공간 주의 메커니즘(SEN-Net, ECA-Net, CBAM)을 도입하는 효과를 연구합니다. 주의 메커니즘이 기존 모델의 성능을 크게 향상시키고, 다양한 활성화 함수와 결합함으로써 성능을 더욱 증가시키는 방법을 보여줍니다.

- **Technical Details**: 본 연구에서는 VGGNet, ResNet, ResNetV2와 같은 세 가지 CNN 이미징 모델을 기반 모델로 사용하고, 이들에 주의 메커니즘을 추가하여 성능을 향상시킵니다. 각 모델은 노이즈에 대한 저항성과 소실 그래디언트 문제를 처리하는 능력이 뛰어나 FER 문제에 적합합니다. 또한, 다양한 활성화 함수(ReLU와 ELU)를 사용하여 아키텍처의 성능에 미치는 영향을 연구하고, YOLO 기반의 얼굴 감지 방법을 통해 배경 픽셀을 제거합니다.

- **Performance Highlights**: 실험 결과, ELU 활성화 함수는 CK+ 데이터셋에서 ResNet-50 모델에서 최고의 정확도를 기록했습니다. SEN-Net, ECA-Net 및 CBAM과 같은 주의 메커니즘의 도입이 모델의 성능을 개선하고, ECA-Net은 채널 간의 관계를 효율적으로 처리하여 FER 애플리케이션에 적합한 결과를 도출했습니다.



### New Insight in Cervical Cancer Diagnosis Using Convolution Neural Network Architectur (https://arxiv.org/abs/2410.17735)
- **What's New**: 이 연구는 자궁경부암(Pap smear) 이미지를 분류하기 위한 CNN(Convolutional Neural Network) 모델에서 최적의 옵티마이저(optimizer)를 선택하는 것이 중요하다는 점을 강조합니다. 여러가지 최적화 기법을 사용하여 Pap smear 이미지를 분류하였습니다.

- **Technical Details**: 이 연구에서는 Stochastic Gradient Descent (SGD), RMSprop, Adam, AdaGrad, AdaDelta, Adamax, Nadam 옵티마이저를 적용하였으며, Resnet-18, Resnet-34, VGG-16의 CNN 아키텍처를 사용했습니다. 각 아키텍처는 전이 학습(transfer learning) 모델을 활용하였습니다.

- **Performance Highlights**: Adamax 옵티마이저는 VGG-16아키텍처에서 72.8%의 정확도를, Resnet-18에서 66.8%의 정확도를 기록하였습니다. Resnet-34는 54.0%로 Nadam보다 0.034% 낮았습니다. 전반적으로 Adamax는 자궁경부암 분류를 위한 CNN에 적합한 옵티마이저로 평가되었습니다.



### FuzzWiz -- Fuzzing Framework for Efficient Hardware Coverag (https://arxiv.org/abs/2410.17732)
- **What's New**: 이번 연구는 하드웨어 검증을 위한 새로운 자동화된 퍼징(fuzzing) 프레임워크인 FuzzWiz를 제안합니다. FuzzWiz는 메타모델링(metamodeling)과 Python을 활용하여 기존의 하드웨어 디자인 검증 과정에서 검증 목표를 신속하게 달성합니다.

- **Technical Details**: FuzzWiz 프레임워크는 RTL(레지스터 전송 수준) 디자인 모듈을 파싱하고 C/C++ 모델로 변환하며, 일반적인 테스트 벤치 및 어설션(assertions)을 포함합니다. 또한, 퍼저(puzzer) 특정 컴파일, 링크, 퍼징 과정을 자동화합니다. 사용자가 선택할 수 있도록 다양한 퍼징 엔진과의 호환성을 보장합니다.

- **Performance Highlights**: OpenTitan 칩의 네 개 IP 블록에서 벤치마킹 결과, 기존의 시뮬레이션 회귀(regression) 기반 접근법보다 약 10배 더 빠른 시간에 약 90%의 커버리지를 달성하는 성과를 보였습니다.



### CogSteer: Cognition-Inspired Selective Layer Intervention for Efficient Semantic Steering in Large Language Models (https://arxiv.org/abs/2410.17714)
- **What's New**: 이번 논문에서는 큰 언어 모델(LLMs)의 해석 가능성을 높이기 위해 눈 움직임 측정치를 사용하여 LLM의 동작을 분석하는 새로운 방법론을 제안합니다. 눈 움직임 연구에서 수집된 데이터를 활용하여 LLM의 다양한 레이어에 걸친 행동을 해석하고, 이를 통해 중간 레이어에서의 의미 조작을 위한 효율적인 방법인 CogSteer를 도입합니다.

- **Technical Details**: 제안된 방법은 LLM의 레이어 속 숨겨진 상태와 눈 움직임 지표를 연관 짓고, 이를 바탕으로 최적의 레이어를 선택하여 의미를 조작합니다. CogSteer 방법은 특히 언어의 독성화(toxification) 및 비독성화(detoxification)의 실험을 통해 효과성을 입증하였고, 실행 시 97%의 계산 자원 절약과 60%의 훈련 시간 절감을 달성합니다. 또한, 암묵적 레이어 대조 개입 방법이 소개되어 안전한 생성 방향으로 의미를 유도합니다.

- **Performance Highlights**: CogSteer 방법은 언어 독성 점수를 87.2%에서 59.1%로 감소시키며, LLaMa2-7B 모델에서는 61.0%로 낮아졌습니다. 이 연구의 결과는 정확하고 신뢰성을 갖춘 모델 배치를 위한 해석 가능성 향상에 기여하며, LLMs의 다양한 모델에 적용될 수 있습니다.



### Beware of Calibration Data for Pruning Large Language Models (https://arxiv.org/abs/2410.17711)
Comments:
          under review

- **What's New**: 최근 연구에서는 대규모 언어 모델(LLMs)의 성능 향상과 배포 비용 절감을 위한 모델 압축 기술, 특히 Post-training pruning에 대한 중요성이 강조되었습니다. 이 논문은 다양한 보정(calibration) 데이터가 pruning 성능에 미치는 영향을 체계적으로 탐구하며, 보정 데이터의 중요성이 고급 pruning 전략 설계보다 더 크다는 점을 발견하였습니다.

- **Technical Details**: Post-training pruning은 반복 학습 없이도 파라미터 중요도를 추정할 수 있는 방법입니다. 이 연구에서는 작은 양의 보정 데이터를 사용하여 파라미터의 중요도를 평가하고, 이를 통해 sparsity가 높은 상황에서도 효과적인 pruning을 가능하게 합니다. 파라미터 중요도 추정에는 inverse Hessian matrix와 L2 노름을 활용하며, 보정 데이터는 훈련 데이터와 유사할수록 성능이 우수하다는 결과를 보여줍니다.

- **Performance Highlights**: DCLM 및 LLaMA-3 모델에서 실험을 수행한 결과, 제안된 보정 데이터 샘플링 방법이 일반적으로 사용되는 보정 데이터보다 우수한 성능을 나타내었으며, 강력한 pruning 기법들과의 호환성을 유지하면서 성능을 크게 향상시켰습니다.



### Scalable Random Feature Latent Variable Models (https://arxiv.org/abs/2410.17700)
- **What's New**: 이번 연구에서는 기존의 Monte Carlo 샘플링 기반 방법 대신, 최적화 기반의 변량 베이지안 추론(Variational Bayesian Inference, VBI) 알고리즘을 통해 RFLVM(Random Feature Latent Variable Models)의 확장 가능성을 개선하고자 합니다. 이 연구에서는 '막대 자르기(stick-breaking)' 구조를 포함한 새로운 방법론인 '블록 좌표 강화 변량 추론(Block Coordinate Descent Variational Inference, BCD-VI)'을 소개하며, 이를 통해 SRFLVMs(Scalable RFLVMs)를 개발하였습니다.

- **Technical Details**: RFLVM은 높은 차원의 데이터를 효과적으로 처리할 수 있는 잠재 변수 모델입니다. 그러나 기존의 Monte Carlo 방법은 많은 계산 비용과 자원 소모로 인해 대규모 데이터셋에서 사용하기 어렵습니다. 이를 해결하기 위해 VBI를 사용하여 posterior inference을 결정론적인 최적화 문제로 변환하였습니다. 또한, stick-breaking 구조를 사용하여 DP(Dirichlet Process)의 명시적 확률 밀도 함수(Probability Density Function)를 도출하였으며, BCD-VI 알고리즘을 통해 비순위 변량을 효율적으로 최적화하였습니다.

- **Performance Highlights**: 제안된 SRFLVMs는 다양한 실제 데이터셋에서 정보가 풍부한 잠재 표현을 생성하고 결측값을 채우는 데 있어 우수한 성능을 보이며, 최신 모델들에 비해 뛰어난 확장성과 계산 효율을 자랑합니다.



### An Adaptive Framework for Generating Systematic Explanatory Answer in Online Q&A Platforms (https://arxiv.org/abs/2410.17694)
Comments:
          10 pages, 6 figures

- **What's New**: 본 논문에서는 복잡한 질문에 대한 해답 생성을 위한 새로운 프레임워크인 SynthRAG를 제안합니다. 기존의 RAG 모델들이 섬세하게 정리된 답변 제공에 한계가 있음을 지적하고, 정보를 통합하는 체계적 접근 방식을 통해 QA 시스템의 성능을 향상시키는 방법을 설명합니다.

- **Technical Details**: SynthRAG는 적응형 개요 생성(adaptive outline generation), 체계적 정보 생성(systematic information generation), 맞춤형 답변 생성(customized answer generation)의 세 가지 주 단계를 포함합니다. 이 모델은 다양한 질문에 맞춤화된 개요를 생성하고, 각 하위 섹션에 대한 일관된 세부 단락을 생성하여 논리적 구조가 갖춰진 포괄적이고 상세한 응답을 제공합니다.

- **Performance Highlights**: Empirical evaluations에 따르면, SynthRAG는 복잡한 질문을 처리하는 데 있어 기존 RAG 모델보다 뛰어난 성능을 보이며, Zhihu 플랫폼에서의 배포 결과도 그 응답이 사용자의 주목을 받았음을 보여줍니다(평균 5.73개의 업보트). 이는 SynthRAG가 실제 적용에서 큰 영향을 미칠 수 있는 가능성을 나타냅니다.



### MIA-DPO: Multi-Image Augmented Direct Preference Optimization For Large Vision-Language Models (https://arxiv.org/abs/2410.17637)
Comments:
          Project URL: this https URL

- **What's New**: 본 논문에서는 Multi-Image Augmented Direct Preference Optimization (MIA-DPO) 방법을 소개하며, 이는 다중 이미지 입력을 효과적으로 처리할 수 있도록 설계되었습니다.

- **Technical Details**: MIA-DPO는 단일 이미지 데이터를 확장하여 그리드 콜라주(grid collage) 또는 픽-인-픽(pic-in-pic) 형식으로 관련이 없는 이미지를 조합하여 다중 이미지 훈련 데이터의 부족성을 해결합니다. 모델의 주의(attention) 값을 활용하여 잘못 선택된 반응을 확인하고 필터링합니다.

- **Performance Highlights**: MIA-DPO는 여러 다중 이미지 벤치마크에서 기존 방법보다 우수한 성능을 달성하며, LLaVA-v1.5에서는 평균 3.0%, 최근의 InternLM-XC2.5에서는 4.3%의 성능 향상을 이루었습니다. 또한 MIA-DPO는 단일 이미지에 대한 이해 능력에 미치는 영향이 최소화됩니다.



### LMLPA: Language Model Linguistic Personality Assessmen (https://arxiv.org/abs/2410.17632)
- **What's New**: 본 논문은 LLM(Large Language Model)의 언어적 성격을 평가하기 위한 Language Model Linguistic Personality Assessment (LMLPA) 시스템을 도입하여, LLM의 언어 생성 능력을 정량적으로 이해하는 방법을 제공합니다.

- **Technical Details**: LMLPA 시스템은 Big Five Inventory를 기반으로 하여 LLM의 작동 능력과 일치하도록 설계된 성격 평가 설문지를 사용합니다. 이 설문지는 개방형 질문으로 구성되어, AI 레이터가 텍스트 반응으로부터 명확한 성격 특성의 수치 지표를 추출하는 과정을 포함합니다. 또한 주성분 분석(Principal Component Analysis) 및 신뢰도 검증을 통해 LLM의 뚜렷한 성격 특성을 정량적으로 평가하는 방법을 제시합니다.

- **Performance Highlights**: 연구 결과는 LMLPA를 통해 LLM이 뚜렷한 성격 특성을 지닌다는 것을 보여주며, 이는 교육, 제조 등 다양한 분야에서 AI 성격 평가의 정교화 및 활용을 확장하는 데 기여할 것입니다. 예를 들어, LLM의 성격을 평가함으로써 교육 환경에서 어떤 성격 특성이 더 나은 학습 곡선을 제공할 수 있는지를 조사할 수 있습니다.



### Graph Signal Adaptive Message Passing (https://arxiv.org/abs/2410.17629)
- **What's New**: 이 논문에서는 Graph Signal Adaptive Message Passing (GSAMP)이라는 새로운 메시지 전송 방법을 제안합니다. GSAMP는 시간에 따라 변하는 그래프 신호에 대해 온라인 예측, 누락된 데이터 보간 및 잡음 제거를 동시에 수행할 수 있는 방법으로, 기존의 그래프 신호 처리 방법과는 다른 접근법을 채택하고 있습니다.

- **Technical Details**: GSAMP는 각 노드에서의 국소화된 계산을 활용하여 고유한 방식으로 시공간 업데이트를 수행합니다. 이 방법은 관측치와 추정치 간의 불일치를 최소화하기 위해 설계된 최적화 문제에서 얻은 적응형 솔루션을 기반으로 하며, Gaussian 및 impulsive noise 조건에서 효과적으로 그래프 신호를 처리합니다.

- **Performance Highlights**: GSAMP는 시간에 따라 변하는 그래프 신호를 다루는 데 있어서 매우 유연하고 견고한 솔루션을 제공합니다. 특정 파라미터 설정 하에 GSAMP는 impulsive noise에서도 높은 강인성을 보이며, 이를 통해 실제 응용에서도 안정적으로 실행될 수 있습니다.



### From PDFs to Structured Data: Utilizing LLM Analysis in Sports Database Managemen (https://arxiv.org/abs/2410.17619)
Comments:
          11 pages, 1 figure

- **What's New**: 이 연구는 대형 언어 모델(Large Language Models, LLMs)이 PDF 문서의 반구조화된 데이터(semistructured data)를 구조화된 형식으로 처리하는 효율성을 평가했습니다. 스페인 스포츠 클럽 데이터베이스(Finnish Sports Clubs Database)를 업데이트하는 데 LLM의 적용 가능성을 분석했습니다.

- **Technical Details**: OpenAI의 GPT-4 및 Anthropic의 Claude 3 Opus 모델을 활용하여 자율적으로 72개의 스포츠 연맹 회원 보고서를 처리하는 AI 지원 접근 방식을 개발하고 평가했습니다. 이 시스템은 90%의 성공률을 기록하며, 72개의 파일 중 65개 파일을 오류 없이 처리하고 7,900개 이상의 데이터 행을 변환했습니다.

- **Performance Highlights**: 초기 개발 시간은 전통적인 수작업 처리와 유사하게 3개월이었으나, 구현된 시스템은 향후 처리 시간을 약 90% 줄일 가능성을 보였습니다. LLM은 반구조화된 데이터 처리 작업을 자동화하는 데 큰 잠재력을 보였으며, AI 자동화와 선택적 인간 감독의 하이브리드 접근 방식을 통해 최적의 결과를 도출했습니다.



### Towards Effective Data-Free Knowledge Distillation via Diverse Diffusion Augmentation (https://arxiv.org/abs/2410.17606)
- **What's New**: 본 연구에서는 데이터 없이 지식 증류(Data-free Knowledge Distillation, DFKD) 기법의 새로운 접근법인 다양한 확산 증강(Diverse Diffusion Augmentation, DDA)을 제안했습니다. 이는 기존의 데이터 합성 방식을 수정하고, 확산 모델을 활용하여 데이터 샘플의 다양성을 높이며, 훈련 데이터를 원활하게 처리할 수 있도록 돕습니다.

- **Technical Details**: DDA는 확산 모델(Diffusion Models)을 활용하여 합성된 데이터의 다양성을 극대화하고, 코사인 유사성(Cosine Similarity) 기법을 통해 품질이 낮은 이미지를 필터링하는 방법을 설명합니다. 이는 데이터의 다양한 스펙트럼을 생성하고, 지식 증류 과정에서의 오류를 줄이는 데 기여합니다.

- **Performance Highlights**: CIFAR-10, CIFAR-100, Tiny-ImageNet 데이터셋에서의 실험 결과, 제안된 DDA 방법은 최신 DFKD 방법들과 비교했을 때 우수한 성능을 보였으며, 다양한 teacher-student 네트워크 구성에서도 큰 이점을 보여주었습니다.



### Integrating Large Language Models for UAV Control in Simulated Environments: A Modular Interaction Approach (https://arxiv.org/abs/2410.17602)
- **What's New**: 이 논문은 LLM(대형 언어 모델)과 UAV(무인 항공기) 기술의 통합이 무인 항공기 제어에 어떻게 새로운 가능성을 열어줄 수 있는지를 탐구하고 있습니다. LLM을 이용하여 UAV는 자연어 명령을 해석하고 응답할 수 있게 되어 사용자에게 보다 직관적인 인터페이스를 제공합니다.

- **Technical Details**: 논문은 LLM을 UAV 제어에 통합하기 위한 다양한 기술적 세부사항을 다루며, LLM의 자연어 처리 기능을 활용하여 UAV의 자율적인 의사결정, 동적인 임무 계획, 상황 인식 및 안전 프로토콜 개선을 도모할 수 있음을 설명합니다. LLM은 API를 통해 UAV와 상호작용하며, GNC(유도 항법 제어)와의 통합을 통해 양방향 정보 수집 및 전송을 지원합니다.

- **Performance Highlights**: LLM과 UAV 통합을 통해 향상된 상황 인식과 자율적인 결정을 바탕으로 UAV가 복잡한 환경에서도 더 빠르게 적응하고 효과적으로 대응할 수 있다는 점이 강조됩니다. 실험적 결과는 LLM 통합이 자율 시스템을 상당히 발전시킬 가능성을 지니고 있음을 보여줍니다.



### Graphusion: A RAG Framework for Knowledge Graph Construction with a Global Perspectiv (https://arxiv.org/abs/2410.17600)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2407.10794

- **What's New**: 이 논문은 자연어에서 지식 그래프(knowledge graph)를 제로샷(zero-shot) 방식으로 구성하는 Graphusion 프레임워크를 소개합니다. 기존 방식들과 달리, 이 방법은 개별 문서에 국한되지 않고 전반적인 지식 통합을 목표로 합니다.

- **Technical Details**: Graphusion의 구성은 세 가지 단계로 이루어집니다: 1단계에서는 토픽 모델링을 통해 시드 엔티티(seed entity)를 추출합니다; 2단계에서는 LLMs를 통해 후보 triplet 추출을 진행합니다; 3단계에서는 통합 모듈을 설계하여 지식의 전반적인 관점을 제공합니다.

- **Performance Highlights**: Graphusion은 엔티티 추출과 관계 인식을 위해 각각 2.92와 2.37의 점수를 기록했습니다. 교육 분야에서 새로운 전문 인증 벤치마크 TutorQA를 통한 9.2%의 정확도 향상도 보여줍니다.



### Challenge on Sound Scene Synthesis: Evaluating Text-to-Audio Generation (https://arxiv.org/abs/2410.17589)
Comments:
          accepted to NeurIPS 2024 Workshop: Audio Imagination

- **What's New**: 이번 논문에서는 2024년 Detection and Classification of Acoustic Scenes and Events의 Sound Scene Synthesis 챌린지를 통해 텍스트에서 오디오로의 생성 관련 문제들을 해결하려고 하였습니다. 특히 Fréchet Audio Distance (FAD) 메트릭과 인간 평가를 결합한 새로운 평가 프로토콜을 제안합니다.

- **Technical Details**: 연구에서는 고품질 오디오 생성, 다양한 소리 생성, 텍스트 캡션 기반의 유연한 조정, 오디오의 카테고리 적합성 및 지각 품질 평가 방안을 중심으로 논의합니다. 데이터 셋은 '전경 소리 (foreground sound)'와 '배경 소리 (background sound)'의 조합으로 구성되며, 모델 성능을 실험하기 위한 평가 구조가 마련되었습니다.

- **Performance Highlights**: 대규모 모델이 일반적으로 뛰어난 성능을 보였지만, 혁신적인 경량화 모델도 괄목할 만한 성과를 나타냈습니다. FAD 메트릭과 인간 평가 간의 강한 상관관계가 확인되어, 새로운 평가 접근법이 효과적임을 입증했습니다.



### Exploring Tokenization Methods for Multitrack Sheet Music Generation (https://arxiv.org/abs/2410.17584)
Comments:
          3 pages, 1 figure, 1 table

- **What's New**: 이 연구에서는 ABC 표기법을 사용한 다중 트랙 악보의 토큰화(tokenization) 방법을 탐구하고, 바 스트림 패칭(bar-stream patching)과 라인 스트림 패칭(line-stream patching)이라는 두 가지 새로운 방법을 도입했습니다.

- **Technical Details**: 본 연구는 기존의 바 패칭(bar patching), 바이트 패칭(byte patching), 바이트 쌍 인코딩(Byte Pair Encoding, BPE) 기법과 비교하여 새로운 패칭 방법의 성능을 평가하였습니다. 바 스트림 패칭은 바를 기준으로 텍스트를 나눈 후, 각 바를 정해진 길이의 패치로 분할합니다. 라인 스트림 패칭은 줄바꿈(line break)을 기준으로 악보를 나누는 방법입니다. 실험은 Tunesformer 아키텍처를 기반으로 하여 진행되었으며, 자동 회귀(character-level) 디코더를 통해 음악 생성 품질을 향상시켰습니다.

- **Performance Highlights**: 실험 결과에 따르면, 바 스트림 패칭은 모든 메트릭에서 가장 우수한 성능을 보였으며, 훈련 및 추론 효율성이 뛰어나고 생성된 결과가 실제 클래식 작곡과 밀접하게 일치하는 것으로 나타났습니다. 바 바이트 패칭, 라인 스트림 패칭은 비교적 짧은 훈련 시간과 빠른 추론 속도를 기록했습니다. BPE는 BPB 평가에서 뛰어난 성능을 보였지만, CLaMP 2 점수에서는 적절한 음악성과의 격차를 보여 회의적인 결과를 낳았습니다.



### Bonsai: Gradient-free Graph Distillation for Node Classification (https://arxiv.org/abs/2410.17579)
- **What's New**: 그래프 증류(Graph distillation)가 GNN(그래프 신경망) 교육의 확장성을 높이기 위해 훈련 데이터 세트를 압축하는 새로운 방법으로 부각되고 있다는 점을 강조합니다. 그러나 기존 기술의 심각한 단점도 발견되었습니다. 이번 연구에서 Bonsai라는 새로운 증류 방법을 제안하며, 이는 메세지 패싱 GNN의 기본 처리 단위인 계산 트리를 바탕으로 합니다.

- **Technical Details**: Bonsai는 기존의 그래프 증류 알고리즘의 한계를 극복하기 위해, 예시(exemplar) 계산 트리를 신중하게 선택하여 데이터 세트를 증류합니다. 이 방법은 모든 계산 트리를 최대한 표현할 수 있도록 설계되어 있으며, 빠른 실행 시간과 모델에 영향을 받지 않는 특징을 가지고 있습니다. 의료적 수학 보장이 적용되어 GNN 아키텍처와 데이터 세트, 매개변수에 강건합니다.

- **Performance Highlights**: Bonsai는 6개의 실제 데이터 세트에서 기존 기법들보다 높은 정확도를 달성했으며, 평균적으로 22배 더 빠른 증류 시간을 기록했습니다. 이러한 성능은 Bonsai의 독창적인 메커니즘 덕분으로, 다른 GNN 아키텍처에 대해서도 뛰어난 강건성을 보입니다.



### Real-time Vehicle-to-Vehicle Communication Based Network Cooperative Control System through Distributed Database and Multimodal Perception: Demonstrated in Crossroads (https://arxiv.org/abs/2410.17576)
Comments:
          ICICT 2024, 18 pages

- **What's New**: 본 논문은 차량-차량(Vehicle-to-Vehicle, V2V) 통신 시스템을 활용한 실시간 네트워크 협력 제어 시스템(VVCCS)을 소개합니다. 이 시스템은 자율 주행의 복잡한 도시 환경에서 교통 계획 및 충돌 회피를 혁신할 수 있도록 설계되었습니다.

- **Technical Details**: VVCCS는 Quanser Car(Qcar) 하드웨어 플랫폼에서 구현되며, 분산 데이터베이스를 개별 자율 차량 및 선택적 중앙 서버에 통합합니다. 시스템은 다중 목표 추적 및 레이더 감지 기능을 갖춘 포괄적인 다중 모달 인식 시스템을 개발하여 복잡한 교차로 환경에서 시연하였습니다.

- **Performance Highlights**: VVCCS는 교차로에서 차량과 보행자를 인식 및 추적하고, 실시간으로 차량 상태 정보를 공유하여 충돌 회피를 위한 최적의 결정을 내립니다. 이 시스템은 통계적으로 안전성을 향상시키는 데 기여할 수 있으며, 도시의 복잡한 환경에서 적용 가능성을 보여주었습니다.



### Differentially Private Learning Needs Better Model Initialization and Self-Distillation (https://arxiv.org/abs/2410.17566)
Comments:
          18 pages

- **What's New**: DPRefine는 Differentially Private Stochastic Gradient Descent (DPSGD)의 한계를 극복하면서 개인 정보 보호 기능을 제공하는 언어 모델 훈련 방법을 제안합니다. 이 방법은 데이터 합성을 통해 강력한 초기화를 통해 모델의 성능을 개선합니다.

- **Technical Details**: DPRefine는 세 가지 단계로 구성됩니다. 첫 번째 단계에서 우리는 사전 훈련된 작은 언어 모델(GPT-2 등)을 사용하여 개인 데이터에 독립적으로 고품질 합성 데이터를 생성합니다. 두 번째 단계에서는 개인 레이블 데이터에서 DPSGD를 사용하여 초기화된 모델을 미세 조정하고, 마지막 단계에서는 새로운 훈련 데이터를 생성을 통해 자기 증류(Self-Distillation)를 수행합니다.

- **Performance Highlights**: DPRefine는 기존의 DPSGD와 비교하여 78.4%의 경우에 AlpacaEval 평가에서 선호되었고, 생성된 텍스트의 언어 오류를 84.0% 감소시켜 언어 질적 문제를 효과적으로 완화했습니다.



### ProtoLens: Advancing Prototype Learning for Fine-Grained Interpretability in Text Classification (https://arxiv.org/abs/2410.17546)
- **What's New**: ProtoLens는 텍스트 분류를 위한 새로운 프로토타입 기반 모델로, 서브 문장 수준의 세밀한 해석 가능성을 제공합니다.

- **Technical Details**: ProtoLens는 Prototype-aware Span Extraction 모듈을 사용하여 학습된 프로토타입과 관련된 텍스트 스팬을 식별하고, Prototype Alignment 메커니즘을 통해 훈련 과정에서 프로토타입이 의미적으로 일관되도록 보장합니다. 이로 인해 해석 가능한 예측을 제공하면서도 경쟁력 있는 정확도를 유지합니다.

- **Performance Highlights**: ProtoLens는 여러 텍스트 분류 벤치마크에서 프로토타입 기반 모델과 비해 해석 불가능한 기준선보다도 더 우수한 성능을 보였습니다.



### Primal-Dual Spectral Representation for Off-policy Evaluation (https://arxiv.org/abs/2410.17538)
Comments:
          29 pages, 5 figures

- **What's New**: 이 논문에서는 오프 폴리시 평가(Off-policy evaluation, OPE) 문제를 다루며, 특히 Distribution Correction Estimation (DICE) 추정기를 개선하기 위해 선형 구조를 도입한 새로운 알고리즘인 SpectralDICE를 제안합니다.

- **Technical Details**: SpectralDICE는 가치 함수(value function)와 상태-행동 전이 연산자의 스펙트럼 분해(spectral decomposition)를 통해 선형적 형태로 표현됩니다. 이를 통해 비선형 비볼록 최적화(non-convex non-concave optimization)를 우회하여 계산 효율성을 높입니다.

- **Performance Highlights**: SpectralDICE는 계산 및 샘플 효율성(sample efficiency)을 제공하며, 이론적으로 엄격한 샘플 복잡도 보장을 제공하는 동시에 다양한 RL 벤치마크에서 철저한 실험 평가를 통해 그 성능을 입증하였습니다.



### Responsible Multilingual Large Language Models: A Survey of Development, Applications, and Societal Impac (https://arxiv.org/abs/2410.17532)
- **What's New**: 다국어 대형 언어 모델(MLLMs)의 발전이 인공지능을 언어적 경계를 넘어 민주화하는 데 있어 중요한 진전을 이루었다. 이 논문은 MLLM을 개발하고 배포하기 위한 포괄적인 종합 프레임워크를 제공함으로써 이론적 기초와 실제 구현 사이의 격차를 해소하고자 한다.

- **Technical Details**: 이 연구는 데이터 전처리에서 배포까지의 실행 가능한 파이프라인을 제시하며, Llama2를 사례 연구로 활용하여 다국어 기능 강화를 위한 최적화 전략을 자세히 설명한다. 여기에는 고자원 언어와 저자원 언어 간의 균형을 맞추기 위한 커리큘럼 학습 접근법, 토큰화 전략 최적화, 효과적인 샘플링 방법이 포함된다.

- **Performance Highlights**: 논문은 고객 서비스, 검색 엔진, 기계 번역 등 실제 애플리케이션을 통해 언어적 다양성을 지원하는 데 있어 중요한 과제가 있음을 밝혀낸다. 또한 88.38%의 세계 언어가 저자원으로 분류되며, 이는 10억 이상의 화자가 영향을 받는다는 점을 강조한다.



### Bridging Swarm Intelligence and Reinforcement Learning (https://arxiv.org/abs/2410.17517)
- **What's New**: 이 논문은 Collective Decision-Making (CDM)과 단일 에이전트 강화 학습 (Reinforcement Learning, RL) 간의 이론적 및 실증적 동등성을 보여줍니다. 이 과정을 통해 'Maynard-Cross Learning'이라는 새로운 RL 업데이트 규칙을 도입합니다.

- **Technical Details**: CDM은 여러 대안 중에서 최적의 선택을 하기 위해 집단이 함께 협력하는 과정입니다. 이 논문에서는 SI와 RL을 연결하는 Replicator Dynamic (RD)을 통해 두 분야의 개념을 연결합니다. 이러한 이론적 장치들은 다중 무장Bandit 문제에서의 RL 적용 시 중요한 통찰을 제공합니다.

- **Performance Highlights**: 이 연구는 기존의 RL 관행인 학습률 조정 및 배칭을 새로운 관점에서 바라봅니다. 또한 다양한 RL 및 집단 실험을 통해 이론적 동등성을 검증하며, SI와 RL 간의 교차 분야에서의 아이디어와 기술 융합을 촉진합니다.



### Time and Frequency Synergy for Source-Free Time-Series Domain Adaptations (https://arxiv.org/abs/2410.17511)
- **What's New**: 본 논문에서는 source-free (소스 없음) time-series (시간 시계열) domain adaptations에 대한 연구가 부족한 가운데, 시간 분야 특성만을 활용하는 기존 접근 방식을 넘어서 frequency components (주파수 구성 요소)를 포함한 Time Frequency Domain Adaptation (TFDA) 방법을 제안합니다.

- **Technical Details**: TFDA는 시간과 주파수 특성을 모두 활용하는 dual branch network (이중 분기 네트워크) 구조로 개발되었습니다. 이 방법에서는 sample group (샘플 그룹)의 예측을 집계하여 신뢰할 수 있는 pseudo-labels (의사 레이블)를 생성하는 neighborhood concept (이웃 개념)를 기반으로 합니다. 또한, contrastive learning (대조 학습) 기법이 적용되어 시간 및 주파수 도메인에서 유효한 neighborhood assumptions (이웃 가정)을 수행합니다. self-distillation strategy (자기 증류 전략)와 uncertainty reduction strategy (불확실성 감소 전략)를 통해 domain shift (도메인 전이) 문제로 인한 불확실성을 완화하고, curriculum learning strategy (커리큘럼 학습 전략)을 통해 noisy pseudo labels (잡음 의사 레이블)에 대응합니다.

- **Performance Highlights**: 실험 결과, TFDA 기법이 기존의 방법들에 비해 benchmark problems (벤치마크 문제)에서 눈에 띄는 성능 향상을 보여 주었습니다.



### Congestion Forecast for Trains with Railroad-Graph-based Semi-Supervised Learning using Sparse Passenger Reports (https://arxiv.org/abs/2410.17510)
Comments:
          Accepted in ACM SIGSPATIAL 2024

- **What's New**: 본 논문은 방향이 요구되는 열차 혼잡 예측을 위해, 승객이 제출한 보고서를 기반으로 한 새로운 접근법인 SURCONFORT를 소개합니다. 이 방법은 반지도 학습( semi-supervised learning )과 철도 네트워크 지향 그래프를 활용하여 데이터 희소성을 해결하는 데 초점을 맞추고 있습니다.

- **Technical Details**: SURCONFORT는 희소한 라벨이 붙은 데이터와 대량의 비라벨 데이터를 활용하는 반지도 학습 방법론을 채택합니다. 또한, 인근 역의 라벨이 붙은 데이터를 통해 비라벨 데이터의 예측을 보완하기 위해 철도 네트워크 그래프를 설계하여 반지도 그래프 정규화( semi-supervised graph regularization )를 적용합니다. 이를 통해 주어진 역, 날짜, 시간에 대한 혼잡 정도를 분류하는 신경망( neural network )을 훈련합니다.

- **Performance Highlights**: SURCONFORT는 최첨단 방법들보다 14.9% 향상된 예측 성능을 보였으며, 데이터 희소성에 효과적으로 대응하여 기존의 지도 학습(supervised learning) 및 그래프 기반 반지도 학습 방법들보다 일관되게 우수한 성능을 보여주었습니다.



### Mitigating Graph Covariate Shift via Score-based Out-of-distribution Augmentation (https://arxiv.org/abs/2410.17506)
Comments:
          17 pages, 5 figures, 4 tables

- **What's New**: 본 논문에서는 그래프 학습에서 훈련 데이터와 테스트 데이터 간의 분포 변화(distribution shift)를 극복하기 위한 새로운 접근 방식을 제안합니다. 특히, 기존의 perturbation 기반 방법들이 안정적인(stable) 및 환경적(environmental) 특징을 정확하게 분리하는 데 의존하는 한계점을 극복하고, 새로운 환경적 특징을 생성하는 score-based 그래프 생성 전략을 사용합니다.

- **Technical Details**: 제안된 방법은 Out-of-Distribution Diffusion Augmentation (OODA) 프레임워크로, 그래프 레이블을 활용하여 안정적인 패턴을 유지하면서 새로운 환경을 탐색할 수 있도록 합니다. Diffusion model을 적용하여, 생성된 그래프의 유효성을 보장하고, unstable한 구조물의 형성을 방지합니다. 또한, 안정적 패턴과 환경적 패턴을 명시적으로 분리할 필요가 없습니다.

- **Performance Highlights**: 광범위한 실험을 통해, OODA는 기존 그래프 OOD 일반화 방법들보다 우수한 성능을 보였습니다. 특히, 합성 데이터(synthetic data), 반인공 데이터(semi-artificial data), 실세계 분자 그래프(real-world molecular graphs), 자연어 감정 분석(natural language sentiment analysis) 데이터셋 등 다양한 과제에서 향상된 결과를 입증하였습니다.



### Unsupervised Domain Adaptation for Action Recognition via Self-Ensembling and Conditional Embedding Alignmen (https://arxiv.org/abs/2410.17489)
Comments:
          This work has been accepted to the Proceedings of the IEEE International Conference on Data Mining, 2024

- **What's New**: 최근 딥 러닝 기반의 착용형 인간 동작 인식(wHAR) 기술이 복잡한 동작의 캡처 및 분류에서 개선되었습니다. 그러나 전문가 주석 부족과 사용자 변동성으로 인해 채택은 여전히 제한적입니다. 이를 해결하기 위해, 새로운 조인트 최적화 아키텍처인 μDAR을 제안하며, 이는 데이터의 일반화 능력을 개선하고 보다 강력한 목표 레이블 생성을 목적으로 합니다.

- **Technical Details**: μDAR은 (i) 일관성 정규화기, (ii) 시간 집계 기반의 강력한 pseudo-label 생성을 위한 앙상블, (iii) 조건 분포 정렬 모듈 등 세 가지 기능으로 구성됩니다. 이러한 구성 요소들은 각기 다른 변화를 줄이기 위해 결합되어 작동하여 소스 도메인과 타겟 도메인 간의 특징 공간에서의 불일치를 줄이기 위해 kernel-based class-wise conditional maximum mean discrepancy (kCMMD)를 최소화합니다.

- **Performance Highlights**: μDAR은 네 개의 벤치마크 wHAR 데이터셋에서 6개의 최신 UDA 방법에 비해 약 4-12%의 매크로-F1 점수 향상을 달성했습니다. 이는 적은 소스 도메인 샘플로도 강력한 일반화를 가능하게 하며, 타겟 샘플에서도 일관된 pseudo-label 생성을 보장합니다.



### Composing Diffusion Policies for Few-shot Learning of Movement Trajectories (https://arxiv.org/abs/2410.17479)
Comments:
          6(+1) pages, 6 figures

- **What's New**: 본 연구에서는 로봇이 신규 스킬을 빠르게 학습할 수 있도록 돕는 새로운 접근법인 Diffusion Score Equilibrium(DSE)를 제안합니다. 이 방법은 기본 정책(prior) 조합을 활용하여 few-shot learning이 가능하도록 합니다.

- **Technical Details**: DSE는 확률적으로 diffusion 정책을 조합하여 몇 개의 시연 데이터 분포에 더 적합하게 모델링합니다. 또한, robot motion의 오류를 평가하기 위한 일반적인 메트릭이 부족한 문제를 해결하기 위해 Maximum Mean Discrepancy on the Forward Kinematics Kernel (MMD-FK)을 제안합니다. 이 메트릭은 로봇의 물리적 연결을 고려하여 두 개의 로봇 모션 분포 간의 거리를 측정합니다.

- **Performance Highlights**: DSE 방법을 사용하면 제공된 시연 수가 늘어남에 따라 30%에서 50%까지 MMD-FK 오류가 감소하는 성과를 보였으며, 이는 이전 정책 조정 방식보다 우수한 성능을 기록했습니다. 또한, 실제 실험을 통해 로봇이 잡지 않은 경로를 학습하는 데 있어서 DSE의 효용성을 입증하였습니다.



### Do Robot Snakes Dream like Electric Sheep? Investigating the Effects of Architectural Inductive Biases on Hallucination (https://arxiv.org/abs/2410.17477)
- **What's New**: 본 논문은 대형 언어 모델(LLM)의 구조적 변화가 허위정보 생성(hallucinations) 경향에 미치는 영향을 폭넓게 분석합니다. 또한, 기존의 연구들이 Transformer 기반 모델에만 집중된 데 반해, 순환 모델(recurrent models)과의 비교를 통해 이러한 문제를 동시에 고려할 필요성을 강조합니다.

- **Technical Details**: 연구에서는 1B에서 70B 규모의 다양한 오픈 소스 LLM을 사용하여 20개의 다양한 허위정보 발생 관련 태스크를 평가하였습니다. 자료는 충실도(faithfulness) 및 사실성(factuality)으로 구분되며, 레이어 설계, 연산 규모 및 교육 방법(instruction-tuning)에 따라 모델 아키텍처의 다양성을 평가합니다.

- **Performance Highlights**: 연구 결과, Transformer 기반 모델과 순환 모델 간에 허위정보 생성 경향에는 큰 차이가 없지만, 특정 태스크에 따라 아키텍처의 특성이 허위정보의 발생 빈도에 영향을 미친다는 점이 드러났습니다. 특히, 순환 모델은 소형 크기에서 더 높은 충실도를 보이며, 교육 시 모델 크기가 증가함에 따라 사실성 허위정보의 의존성이 두드러진다는 점을 강조합니다.



### AdaptoML-UX: An Adaptive User-centered GUI-based AutoML Toolkit for Non-AI Experts and HCI Researchers (https://arxiv.org/abs/2410.17469)
- **What's New**: 본 논문에서는 비전문가가 이해하고 활용할 수 있는 ML 모델 개발을 돕는 새로운 툴킷 AdaptoML-UX를 소개합니다. 이 툴킷은 자동화된 Feature Engineering, 기계 학습 및 Incremental Learning을 통합하여 사용자가 ML 파이프라인을 손쉽게 구축할 수 있도록 합니다.

- **Technical Details**: AdaptoML-UX는 그래픽 사용자 인터페이스(GUI)를 제공하며, 알고리즘 선택, Feature Engineering 및 Hyperparameter Tuning을 자동화하여 복잡한 프로그래밍이나 디버깅 과정을 제거합니다. 또한, 사용자의 행동에 따라 모델을 개인화할 수 있는 Incremental Learning을 지원합니다. 이 프레임워크는 다양한 문제 도메인 및 데이터 세트에 효율적으로 적응합니다.

- **Performance Highlights**: AdaptoML-UX는 유연성과 접근성을 통해 비전문가도 고급 기계 학습 기술을 활용할 수 있도록 하였으며, 사용자 중심의 모델을 개발함으로써 HCI 연구자들이 손쉽게 활용할 수 있도록 지원합니다. 이로 인해 수동 실험의 필요성을 줄이고, 시간 및 자원을 절약할 수 있습니다.



### Data Obfuscation through Latent Space Projection (LSP) for Privacy-Preserving AI Governance: Case Studies in Medical Diagnosis and Finance Fraud Detection (https://arxiv.org/abs/2410.17459)
Comments:
          19 pages, 6 figures, submitted to Conference ICADCML2025

- **What's New**: AI 시스템들이 사회의 중요한 분야에 점점 더 통합됨에 따라 강력한 개인 정보 보호 방법에 대한 수요가 증가하고 있습니다. 본 논문은 AI 거버넌스를 향상시키고 책임 있는 AI 준수를 보장하기 위한 새로운 기술인 Latent Space Projection (LSP)을 소개합니다.

- **Technical Details**: LSP는 머신러닝을 활용하여 민감한 데이터를 라텐트 공간(latent space)으로 투영하여 효과적으로 난독화(obfuscation)하며, 모델 훈련과 추론을 위한 필수 기능을 보존합니다. 전통적인 개인 정보 보호 방법인 차별적 개인 정보 보호(differential privacy)나 동형 암호화(homomorphic encryption)와 달리, LSP는 데이터를 추상적이고 낮은 차원의 형태로 변환하여 데이터 유용성과 개인 정보 보호 간의 섬세한 균형을 유지합니다.

- **Performance Highlights**: LSP는 벤치마크 데이터셋과 두 가지 실제 사례 연구(의료 암 진단 및 금융 사기 분석)에 대한 실험을 통해 그 효과성을 검증했습니다. 결과적으로 LSP는 이미지 분류에서 98.7%의 높은 성능을 달성하며, 민감한 속성 추론에 대해 97.3%의 강력한 개인 정보 보호를 제공합니다. 이는 전통적인 익명화(anonymization) 및 개인 정보 보호 방법들보다 우수한 성과입니다. 또한, LSP는 GDPR, CCPA, HIPAA와 같은 글로벌 AI 거버넌스 프레임워크와의 정렬을 검토하며 공정성(fairness), 투명성(transparency), 책임(accountability)에 기여하는 방안을 강조합니다.



### In Context Learning and Reasoning for Symbolic Regression with Large Language Models (https://arxiv.org/abs/2410.17448)
- **What's New**: 이 연구에서 제안된 방법은 대형 언어 모델(LLMs)이 기호 회귀(symbolic regression) 문제를 해결할 수 있는 가능성을 조사합니다. GPT-4를 활용하여 주어진 데이터에서 수식(expressions)을 생성하고, 이를 외부 Python 도구를 사용하여 최적화하고 평가합니다.

- **Technical Details**: 기호 회귀는 데이터 세트에서 간단하고 정확한 수식을 찾기 위한 머신러닝 방법입니다. 이 연구는 GPT-4를 통해 데이터 분석, 이전의 수식, 그리고 과학적 맥락을 분석하도록 여러 단계를 포함한 '체인 오브 생각(chain-of-thought)' 프롬프트를 사용하여 진행됩니다. GPT-4는 제안한 수식의 복잡성과 손실을 최적화하며, 최종적으로 데이터에 적합한 수식을 생성합니다.

- **Performance Highlights**: 연구 결과, GPT-4는 실험 데이터에서 다섯 개의 잘 알려진 과학적 수식을 모두 재발견하는 데 성공하였으며, 일반적으로 스크래치패드와 과학적 맥락을 고려했을 때 성능이 향상되었습니다. 그러나 기존의 기호 회귀 프로그램들에 비해 목표 수식이 더 복잡할 경우 성능은 떨어지는 경향이 있었습니다.



### Evaluating AI-Generated Essays with GRE Analytical Writing Assessmen (https://arxiv.org/abs/2410.17439)
Comments:
          20 pages, 6 figures

- **What's New**: 이번 연구는 최신의 대형 언어 모델(LLMs), 특히 GRE(Graduate Record Examination)의 분석적 작문 평가를 통해 AI로 생성된 에세이의 품질을 체계적으로 평가합니다.

- **Technical Details**: 연구에서는 AI 모델인 GPT-4o, Gemini, Llama3-8b 등 다양한 최신 LLM을 사용하여 에세이를 생성하고, 이 에세이를 인간 평가자와 GRE의 자동 채점 시스템인 e-rater 엔진을 통해 평가하였습니다. 연구 대상은 2개의 작문 프롬프트에 대한 100개의 에세이로, 총 2000개의 에세이가 생성되었습니다.

- **Performance Highlights**: 그 중에서, 최고의 성과를 보인 GPT-4o는 GRE 채점 기준에 따라 평균 4.67점을 기록하였으며, 이는 '문제를 일반적으로 조심스럽고 잘 전개한 분석'과 '문제를 능숙하게 분석하고 충분한 명료함으로 의미를 전달한다'는 사이에 위치합니다.



### Interpreting Affine Recurrence Learning in GPT-style Transformers (https://arxiv.org/abs/2410.17438)
Comments:
          21 pages, 18 figures

- **What's New**: 본 논문은 GPT 스타일 트랜스포머의 내부 메커니즘, 특히 인컨텍스트 학습(in-context learning, ICL)에 대한 기전적 해석(mechanistic interpretability)을 다룹니다. 연구진은 선형 회귀(recurrences) 예측을 위한 변형기(train-based transformer)를 훈련하여 ICL 문제를 개선하는 방법을 분석했습니다.

- **Technical Details**: 연구는 세 개의 레이어(layer)로 구성된 맞춤형 변형기를 훈련시켜 affine recurrences를 예측하는 것을 목표로 하였으며, 경험적(empirical) 및 이론적(theoretical) 접근 방식을 통해 모델의 내부 작동을 분석했습니다. 논문은 모델이 제로(zeroth) 레이어에서의 복사(copying) 메커니즘을 통해 초기 추정치를 형성하고, 두 번째 레이어에서 음의 유사성 머리(negative similarity heads)를 사용하여 이를 정제(refine)하는 방식에 대해 논의했습니다.

- **Performance Highlights**: 모델은 affine recurrence 문제를 해결하며, 복잡한 수치(numerical tasks) 기능을 ICL을 통해 잘 수행하는 것으로 나타났습니다. 이러한 결과는 트랜스포머의 재귀(recursive) 작업에서의 행동에 대한 더 깊은 이해를 제공하며, AI 정렬(AI alignment)을 개선할 수 있는 잠재적인 방향을 제시합니다.



### Artificial Intelligence in Brazilian News: A Mixed-Methods Analysis (https://arxiv.org/abs/2410.17423)
Comments:
          18 pages, 8 figures, 3 tables

- **What's New**: 이번 연구는 2023년 7월 1일부터 2024년 2월 29일까지 브라질의 13개 주요 온라인 뉴스 매체에서 발표된 3,560개의 뉴스 기사를 분석하여 인공지능(AI) 보도에서 나타나는 주요 주제를 규명하고, 사회적 우려가 어떻게 다루어지는지를 살펴보았습니다. 특히 브라질 미디어에서 AI 관련 보도가 주로 직장에서의 응용 및 제품 출시와 관련된 주제에 몰린 반면, 사회적 우려는 주로 딥페이크와 선거의 무결성에 국한되어 있음을 발견하였습니다.

- **Technical Details**: 연구는 Computational Grounded Theory (CGT) 방법론을 사용하였으며, 데이터 분석에는 Latent Dirichlet Allocation (LDA), BERTopic 및 Named-Entity Recognition (NER) 기술이 포함되었습니다. LDA는 문서 집합에서 추상적 주제를 발견하기 위해 단어 분포를 분석하는 비지도 기계 학습 기법입니다. BERTopic은 BERT 임베딩과 클러스터링 알고리즘을 활용하여 텍스트의 주제를 식별하는 고급 모델링 기법입니다. NER은 텍스트에서 개체를 식별하고 이를 사람, 조직, 장소 등으로 분류하는 기법입니다.

- **Performance Highlights**: 연구 결과 브라질에서의 AI 뉴스 보도는 기업 관련 개체가 두드러지게 나타나며, 이는 기업 일정의 강한 영향을 나타냅니다. 따라서 브라질 미디어의 AI에 대한 사회적 영향에 대한 보다 비판적이고 세밀한 논의의 필요성이 강조됩니다.



### End-to-End Optimization and Learning of Fair Court Schedules (https://arxiv.org/abs/2410.17415)
- **What's New**: 이 논문은 미국 내 형사 재판소의 공정한 사전 심리 일정 조정 시스템을 개발하기 위해 머신 러닝 모델과 최적화 알고리즘을 통합한 새로운 프레임워크를 제안합니다. 이 시스템은 피고의 선호도와 가용성을 균형 있게 고려합니다.

- **Technical Details**: 우선, 이 논문은 피고의 일정 선호도가 종종 불완전하거나 불확실하여 발생하는 형평성 문제를 비선형 정수 프로그램(nonlinear integer program) 형태로 공식화합니다. 이와 함께, 의사 결정 손실 함수(decision loss function)를 직접 최적화하여 피고의 선호에서의 불확실성을 처리하는 머신 러닝과 최적화 기술을 통합합니다.

- **Performance Highlights**: 제안된 프레임워크는 피고의 다양한 인구 통계학적 특성을 반영하여 공정한 법원 일정을 생성하는 데 필요한 하이퍼 파라미터를 직접 조정함으로써 공정성과 효율성을 동시에 강조하고 있습니다.



### Geometric Graph Neural Network Modeling of Human Interactions in Crowded Environments (https://arxiv.org/abs/2410.17409)
Comments:
          \c{opyright} 2024 the authors. This work has been accepted to IFAC for publication under a Creative Commons Licence CC-BY-NC-ND

- **What's New**: 이 연구에서는 심리학적 연구에서 얻은 도메인 지식을 통합하여 보행자 간의 상호작용을 모델링하고 미래 경로를 예측하는 기하학적 그래프 신경망(GNN) 아키텍처를 제안합니다. 기존 연구와 달리, 우리는 보행자의 시야, 이동 방향, 거리 기반 커널 함수를 사용하여 상호작용 이웃을 정의합니다.

- **Technical Details**: 제안된 모델은 Mohamed et al.(2020)에 의해 제안된 Social Spatio-Temporal Graph Neural Network (STGCNN) 아키텍처에 기반하여 기하학적 커널 함수를 접목하고 상호작용 이웃 정의를 정제합니다. 경로 예측 작업에서 우리는 보행자 경로가 이변량 가우시안 분포를 따른다고 가정하고, 평균, 표준 편차 및 상관 계수를 포함한 가우시안 분포의 매개변수를 모델링합니다.

- **Performance Highlights**: 여러 데이터 세트를 통해 실험한 결과, 평균 및 최종 변위 오차 메트릭이 감소함에 따라 예측 정확도가 향상되었습니다. 이 연구는 도메인 지식과 데이터 기반 접근 방식을 통합하여 복잡한 보행자 상호작용 모델링에 효과적이라는 점을 강조합니다.



### Quantum Large Language Models via Tensor Network Disentanglers (https://arxiv.org/abs/2410.17397)
Comments:
          4 pages, 2 figures

- **What's New**: 이 논문에서는 양자 컴퓨팅(quantum computing)과 양자 영감을 받은 기술(quantum-inspired techniques)을 통합하여 대형 언어 모델(Large Language Models, LLMs)의 성능을 향상시키는 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 Self-Attention 및 Multi-layer Perceptron 레이어에서 가중치 행렬(weight matrices)을 두 개의 변량 양자 회로(variational quantum circuits) 및 매트릭스 제품 연산자(Matrix Product Operator, MPO)와 같은 양자 영감 텐서 네트워크(tensor network) 조합으로 대체하는 것을 포함합니다. 이 대체는 텐서 네트워크 분리자(tensor network disentanglers)와 MPO의 적용을 통해 가중치 행렬을 분해하여 기존 LLM의 기능을 재현할 수 있게 합니다.

- **Performance Highlights**: 더 복잡하고 깊은 양자 회로를 통합하고 MPO의 결합 차원(bond dimensions)을 증가시킴으로써, 이 방법은 양자 강화 LLM 내의 추가적인 상관관계를 포착하여 기존 모델보다 더 높은 정확성을 달성하면서도 낮은 메모리 오버헤드를 유지합니다.



### A 10.60 $\mu$W 150 GOPS Mixed-Bit-Width Sparse CNN Accelerator for Life-Threatening Ventricular Arrhythmia Detection (https://arxiv.org/abs/2410.17395)
Comments:
          2 pages, accepted to The 30th Asia and South Pacific Design Automation Conference (ASP-DAC 2025)

- **What's New**: 이 논문은 심실 부정맥(ventricular arrhythmia, VA) 탐지를 가속화하기 위한 초저전력 혼합 비트 너비의 희소 컨볼루션 신경망(accelerator) 설계를 제안합니다.

- **Technical Details**: 제안된 하드웨어 설계는 희소 처리 요소(sparse processing element, SPE) 아키텍처를 사용하며, 측정된 전력 소모는 10.60 μW이며, 성능은 150 GOPS, 진단 정확도는 99.95%입니다. 프로토타입 칩은 TSMC 40nm CMOS 저전력(LP) 공정을 사용하여 제작하였으며, 1D CNN 구조의 8개 레이어를 활용하고, 50% 희소성을 가진 네트워크를 위해 하드웨어 인식 양자화(hardware-aware quantization)와 공진 단축(pruning) 기법이 적용되었습니다.

- **Performance Highlights**: 제안된 CNN 가속기는 35 μs의 추론 시간과 92.35%의 추론 정확도를 달성하며, 최종 진단 정확도는 99.95%입니다. 이 가속기는 최첨단 기술 대비 14.23배 향상된 전력 밀도(0.57 μW/mm²)를 제공하고, 임플란트 또는 착용 가능한 의료 기기에 적합합니다.



### packetLSTM: Dynamic LSTM Framework for Streaming Data with Varying Feature Spac (https://arxiv.org/abs/2410.17394)
- **What's New**: 이 논문은 Streaming 데이터의 변동 입력 특성 공간을 처리하기 위해 새로운 온라인 학습 방법인 packetLSTM을 제안합니다. 이 방법은 각 입력 특성 별로 전용 LSTM이 할당된 동적 구조를 갖추고 있으며, 이는 지속적인 학습과 망각 문제를 완화하는 데 도움을 줍니다.

- **Technical Details**: packetLSTM은 dimension-invariant operator를 통해 정보 집계를 수행하며, 필요한 경우 LSTM을 동적으로 활성화, 비활성화 및 추가할 수 있는 기능을 가집니다. 이는 LSTM이 각 특성을 위한 로컬 정보를 유지하고, 공유 메모리를 통해 글로벌 정보를 통합하는 구조로 이루어져 있습니다.

- **Performance Highlights**: packetLSTM은 다섯 개의 데이터셋에서 최첨단 성능을 기록하였으며, 기존의 HapTransformer와 비교했을 때도 상대적으로 우수한 성능을 보입니다. 또한 packetLSTM의 원리는 GRU 및 기타 RNN 유형으로 확장될 수 있다는 점에서도 의미가 큽니다.



### Episodic Future Thinking Mechanism for Multi-agent Reinforcement Learning (https://arxiv.org/abs/2410.17373)
Comments:
          NeurIPS 2024 (Web: this https URL)

- **What's New**: 본 연구는 동물에서 관찰된 인지 과정을 기반으로 강화 학습(RL) 에이전트에 에피소드 미래 사고(EFT) 메커니즘을 도입했습니다. 이 메커니즘은 다양한 성격의 에이전트를 고려하여 다중 에이전트 상호작용에서 최적의 결정을 내릴 수 있도록 지원합니다.

- **Technical Details**: 제안된 EFT 메커니즘은 다중 캐릭터 정책(multi-character policy)과 캐릭터 추론 모듈(character inference module)로 구성됩니다. 캐릭터는 보상 구성 요소의 다양한 가중치 조합으로 정의되며, 에이전트는 타겟 에이전트의 관찰-행동 궤적을 수집하고 이를 통해 캐릭터를 추론합니다. 이 메커니즘은 Multi-agent Partially Observable Markov Decision Process (MA-POMDP) 프레임워크를 사용하여 모델링됩니다.

- **Performance Highlights**: 다양한 주행 특성을 가진 다중 에이전트 자율 주행 시나리오에서 EFT 메커니즘을 평가한 결과, 정확한 캐릭터 추론이 기존 다중 에이전트 솔루션보다 더 높은 보상을 가져온다는 것을 보여주었습니다. 사회의 캐릭터 다양성이 어떻든 간에 제안된 메커니즘은 집단 보상을 향상시키는 효과가 있음이 확인되었습니다.



### EEG-DIF: Early Warning of Epileptic Seizures through Generative Diffusion Model-based Multi-channel EEG Signals Forecasting (https://arxiv.org/abs/2410.17343)
Comments:
          9 pages, 4 figures, 3 tables, accepted by ACM BCB 2024

- **What's New**: 이번 연구에서는 다채널 EEG 신호의 미래 경향을 예측할 수 있는 새로운 알고리즘인 EEG-DIF를 제안합니다. 이 알고리즘은 다채널 EEG 신호의 시간-공간(Spatio-Temporal) 정보를 효과적으로 표현하고 학습할 수 있는 방법을 제공합니다.

- **Technical Details**: EEG-DIF 알고리즘은 다채널 EEG 신호 예측 작업을 이미지 완성(Image Completion) 작업으로 변환하여 시간-공간 상관관계와 미래 발달 패턴을 종합적으로 표현합니다. 이 연구는 공개된 간질(Epilepsy) EEG 데이터 세트를 활용하여 EEG-DIF를 구성하고 검증합니다.

- **Performance Highlights**: EEG-DIF는 다채널 EEG 신호의 미래 경향을 동시에 정확하게 예측할 수 있으며, 생성된 EEG 데이터에 기반한 간질 발작 조기 경고 정확도가 0.89에 달합니다. 이는 임상 진단 과정의 최적화 및 향상에 기여할 수 있습니다.



### Captions Speak Louder than Images (CASLIE): Generalizing Foundation Models for E-commerce from High-quality Multimodal Instruction Data (https://arxiv.org/abs/2410.17337)
Comments:
          Xinyi Ling and Bo Peng contributed equally to this paper

- **What's New**: 이 논문에서는 전자상거래(e-commerce)에서 다중 모달 데이터(multimodal data)의 최적 이용을 위한 최초의 대규모 및 고품질 다중 모달 지침 데이터셋인 MMECInstruct를 소개합니다. 또한, 전자상거래의 다중 모달 정보를 통합하기 위한 간단하면서도 효과적인 프레임워크인 CASLIE를 개발하였습니다.

- **Technical Details**: MMECInstruct 데이터셋은 75,000개의 샘플로 구성되어 있으며, 7개의 실제 전자상거래(task) 작업에 대한 지침과 이미지, 텍스트 입력, 출력을 포함합니다. CASLIE는 3개의 모듈(상황 조건화된 캡션 생성, 캡션 품질 평가, 모달리티 정보 융합)을 통해 이미지와 텍스트를 통합하는 시스템으로, 전자상거래 작업에 대한 높은 품질의 통합된 시각적 데이터를 제공합니다.

- **Performance Highlights**: CASLIE 모델은 5개의 고급 기준 모델과 비교해 모든 인도메인(indomain) 평가에서 평균 6.5% 높은 성능을 보였으며, 인도메인 외(out-of-domain) 설정에서도 3.3%의 성능 향상을 나타내어 강력한 일반화 가능성을 보여주었습니다.



### A Comprehensive Survey and Classification of Evaluation Criteria for Trustworthy Artificial Intelligenc (https://arxiv.org/abs/2410.17281)
Comments:
          This work has been accepted for publication in AI and Ethics

- **What's New**: 이 논문은 신뢰할 수 있는 인공지능(Trustworthy Artificial Intelligence, TAI)에 대한 평가 기준의 문헌을 체계적으로 검토하여, TAI의 EU 7대 원칙에 초점을 맞추고 있습니다. 논문은 현재의 평가 기준을 분석하고, 이를 EU TAI 원칙에 매핑하며, 각 원칙에 대한 새로운 분류 체계를 제안합니다.

- **Technical Details**: 시스템적 문헌 검토( systematic literature review )를 바탕으로 TAI 평가 기준을 정리하였으며, 공정성(fairness) 원칙에 대한 기존의 평가 기준을 상세히 분석하고, 나머지 원칙에 대한 시작점을 제시합니다. 이 논문에서는 또한 다수의 TAI 원칙 사의 trade-off에 대한 논의 역시 포함되어 있습니다.

- **Performance Highlights**: 이 연구는 TAI 평가를 위한 표준화된 기준 필요성을 강조하며, AI 시스템의 신뢰성을 측정하기 위한 거버넌스 프레임워크의 필요성을 부각시킵니다. 또한, ISO/IEC 42001 표준이 EU AI 법률 준수에 중요한 역할을 할 가능성을 제시하고 있습니다.



### Automated Quality Control System for Canned Tuna Production using Artificial Vision (https://arxiv.org/abs/2410.17275)
Comments:
          6 pages, 12 figures

- **What's New**: 이 논문은 톤(Tuna) 금속 캔의 결함을 감지하고 분류하기 위한 자동화 제어 시스템을 구현한 내용을 담고 있습니다.

- **Technical Details**: 이 시스템은 컨베이어 벨트(conveyor belt)와 포토일렉트릭 센서(photoelectric sensor)에 의해 작동되는 카메라를 사용하여 시각적 인식을 수행합니다. 로봇 팔(robotic arm)은 금속 캔의 상태에 따라 분류합니다. IoT 시스템을 통해 Industry 4.0 통합이 이루어지며, Mosquitto, Node-RED, InfluxDB, Grafana 등이 사용되었습니다. 결함 감지를 위해 YOLOv5 모델이 사용되며, GPU에서 Google Colab을 통해 학습하여 라벨의 OCR 텍스트 감지를 가능하게 합니다.

- **Performance Highlights**: 결과는 실시간 문제 식별의 효율성, 자원 최적화, 품질 제품 제공을 보여줍니다. 또한 비전 시스템은 품질 관리 작업의 자율성을 기여하여 운영자가 회사 내에서 다른 기능을 수행할 수 있도록 합니다.



### Military Applications of Machine Learning: A Bibliometric Perspectiv (https://arxiv.org/abs/2410.17272)
- **What's New**: 이 논문은 군사 조직에 적용된 머신러닝 아키텍처 모델을 제시하고 있습니다. 군 개인정보 보호를 위해 비군사 소스를 사용하였고, 2021년까지의 데이터로 진행된 문헌 분석을 포함합니다.

- **Technical Details**: 이 연구는 머신러닝과 관련된 기존 연구를 조사하며, SciMat, Excel, VosViewer 도구를 활용하여 연구 방법론을 설명합니다. 데이터 마이닝(data mining), 전처리(preprocessing), 클러스터 정규화(cluster normalization) 및 전략적 다이어그램(strategic diagram) 분석을 포함합니다.

- **Performance Highlights**: 결과적으로, 머신러닝이 군사 환경의 대량 데이터 분석에 어떻게 활용될 수 있는지를 보여주며, 의사결정 지원을 위한 중요 영역과 최신 발전을 다루고 있습니다.



### Legal Theory for Pluralistic Alignmen (https://arxiv.org/abs/2410.17271)
- **What's New**: 본 논문은 법 이론이 AI의 조화(alignment) 문제 해결에 기여할 수 있는 방법을 제안합니다. 특히, 다원성(pluralism)과 구체화(specification)와 관련된 문제에서 법과 AI를 연결지어 설명하고 있습니다.

- **Technical Details**: 논문에서는 법이 일반적인 규칙을 구체적인 사례(case)의 적용을 통해 정의하는 방식을 설명합니다. 이는 법이 시간에 따라 일반 규칙을 구체화하는 과정에서 일어나는 상호작용을 통해 향후 법 해석의 예측 가능성을 높이고, 다원적 관점을 유지하는 것을 강조합니다. 또한, Kundu 외의 연구를 통해 모델이 규칙을 해석하는 방식의 혼란을 지적하고, Hart와 Sunstein의 법적 개념을 통해 AI의 조화 문제를 논의합니다.

- **Performance Highlights**: 이 연구는 법의 기존 해결책이 AI 조화에 어떻게 적용될 수 있는지를 설명하면서, AI 시스템이 다원적인 관점을 수용하고 보장할 수 있는 방법론을 제공하고 있습니다. 법 해석의 접목을 통해, AI의 출력 결과에 대한 일관성과 예측 가능성을 높이고 개인의 선택 자유를 존중하는 데 기여할 수 있는 방안을 제안합니다.



### FairFML: Fair Federated Machine Learning with a Case Study on Reducing Gender Disparities in Cardiac Arrest Outcome Prediction (https://arxiv.org/abs/2410.17269)
- **What's New**: 이번 연구에서는 Fair Federated Machine Learning (FairFML)이라는 새로운 접근 방식을 제시하여, 여러 기관간의 의료 데이터 협업에서 알고리즘 편향을 줄이면서도 환자의 프라이버시를 보호하는 방법을 제안합니다.

- **Technical Details**: FairFML은 모델에 구애받지 않는 솔루션으로, 제안된 프레임워크는 연합 학습(federated learning, FL) 모델의 공정성을 향상시키는 데 기여합니다. 연구에서는 성별 편차 해소를 목표로 심정지 예측 임상 사례를 통해 FairFML의 효과를 입증하였습니다.

- **Performance Highlights**: FairFML 프레임워크는 중앙집중식 모델과 비교하여 최대 65%까지 공정성을 개선하며, 성능에 있어서도 로컬 및 중앙집중식 모델과 유사한 결과를 보여주었습니다. 이는 수신기 작동 특성(receiver operating characteristic) 분석을 통해 측정되었습니다.



### SPikE-SSM: A Sparse, Precise, and Efficient Spiking State Space Model for Long Sequences Learning (https://arxiv.org/abs/2410.17268)
Comments:
          23 pages, 5 figures

- **What's New**: 최근 논문에서는 새로운 스파이킹 상태 공간 모델(SNN)인 SPikE-SSM을 제안하여 긴 시퀀스 학습의 효율성을 높이고자 하였습니다. 이 모델은 생물학적 해석 가능성을 살리면서도 고속 추론이 가능하도록 설계되었습니다.

- **Technical Details**: SPikE-SSM은 두 가지 주요 혁신을 포함하고 있습니다. 첫째, 병렬 처리 성능을 높이기 위해 경계 압축 전략(Parallel Max-Min Boundary Compression, PMBC)을 도입하였습니다. 둘째, 리셋-비재발 메커니즘을 갖춘 정교한 LIF(Leaky Integrate-and-Fire) 뉴런 모델을 제안하여 생물학적으로 해석 가능한 동적 계산을 가능하게 하였습니다. 이를 통해 하이퍼파라미터를 효율적으로 훈련할 수 있습니다.

- **Performance Highlights**: SPikE-SSM은 긴 시퀀스 학습을 위한 다양한 벤치마크 실험에서 그 효과와 견고성을 검증하였으며, WikiText-103 대규모 언어 데이터셋에서도 성능을 입증하였습니다. 결과적으로 SPikE-SSM은 스파이킹 뉴런을 활용한 효율적인 긴 시퀀스 학습을 가능하게 하며, 전반적인 정확도와 효율성을 극대화합니다.



### Zero-Shot Vision-and-Language Navigation with Collision Mitigation in Continuous Environmen (https://arxiv.org/abs/2410.17267)
- **What's New**: 본 논문은 Collision Mitigation (VLN-CM) 기능을 갖춘 Zero-Shot Vision-and-Language Navigation을 제안합니다. 기존의 VLN과 달리, 사용자가 훈련 데이터 없이도 자연어 지침을 이용하여 연속적인 환경에서 탐색할 수 있도록 합니다.

- **Technical Details**: VLN-CM은 Attention Spot Predictor (ASP), View Selector (VS), Progress Monitor (PM) 및 Open Map Predictor (OMP)로 구성되어 있습니다. ASP는 Large Language Model (e.g. ChatGPT)을 활용하여 탐색 지침을 특정 주목 지점으로 분해하고, VS는 CLIP 유사도를 사용하여 30도 간격의 파노라마 이미지 중 주목 지점을 포함한 이미지를 선택합니다. PM은 규칙 기반 접근 방식을 사용하여 다음 주목 지점을 결정하고, OMP는 깊이 정보를 사용해 점유 마스크를 예측하여 안전한 이동 거리를 선택합니다.

- **Performance Highlights**: VLN-CM은 VLN-CE 검증 데이터에서 여러 기준선 방법보다 뛰어난 성능을 나타냈으며, OMP는 에이전트의 충돌을 효과적으로 완화하는 데 기여했습니다. SR(성공률)을 0.11로 달성한 반면, Random Agent와 Hand-Crafted Agent는 각각 0.03의 SR을 기록했습니다.



### Temporal Relational Reasoning of Large Language Models for Detecting Stock Portfolio Crashes (https://arxiv.org/abs/2410.17266)
- **What's New**: 본 논문에서는 Temporal Relational Reasoning (TRR)이라는 알고리즘 프레임워크를 제안하여 주식 포트폴리오의 붕괴를 탐지하는 문제를 해결하고자 합니다. 이는 인간의 복잡한 문제 해결 능력을 모방하는 것을 목표로 합니다.

- **Technical Details**: TRR은 뉴스를 통해 수집된 정보를 동적으로 처리하고, 사건 간의 영향을 분석하며, 시간의 흐름에 따른 맥락을 이해하는 기능을 제공합니다. 이 과정을 통해 포트폴리오에 대한 전체적인 집계 효과를 도출할 수 있습니다.

- **Performance Highlights**: TRR은 주식 포트폴리오 붕괴 탐지에서 최신 솔루션보다 높은 성능을 보였으며, 제안된 각 구성 요소가 성능 향상에 기여하는 방식을 ablation study를 통해 입증하였습니다.



### Federated brain tumor segmentation: an extensive benchmark (https://arxiv.org/abs/2410.17265)
- **What's New**: 최근 연합 학습(federated learning)이 의료 이미지 분석 분야에서 중요성이 대두되고 있습니다. 이는 다수 센터의 데이터를 집계하면서 개인 정보를 보호할 수 있는 특성을 가지고 있습니다.

- **Technical Details**: 다양한 연합 학습 알고리즘을 글로벌(하나의 최종 모델), 개인화(각 기관당 하나의 모델), 하이브리드(기관 클러스터당 하나의 모델)로 분류하고, 2022년 연합 뇌종양 세분화(Federated Brain Tumor Segmentation) 데이터셋에 대한 적용 가능성을 연구했습니다. 일관된 정규화 방법(FedAvg)이 우수한 성능을 보여주지만, 각 카테고리에서 일부 방법이 성능을 약간 향상시킬 수 있음을 확인했습니다.

- **Performance Highlights**: 모델의 편향을 줄이고, 풀링된 데이터셋을 기관 간에 분배하는 다양한 방법을 통한 연합 학습의 동작 방식에 대한 깊은 이해를 제공했습니다. 이 연구는 IID(Independent and Identical Distributed) 설정과 제한된 데이터 설정을 통해 수행됩니다.



### Audio-Driven Emotional 3D Talking-Head Generation (https://arxiv.org/abs/2410.17262)
- **What's New**: 이 논문에서는 높은 충실도의 오디오 기반 비디오 초상화를 정확한 감정 표현으로 합성하는 새로운 시스템 EmoGene을 제안합니다. 저자들은 감정 표현의 생성이 현실적인 대화 머리 생성의 중요한 측면으로 남아있음을 강조하고, 이를 해결하기 위한 세 가지 모듈을 도입합니다.

- **Technical Details**: EmoGene 시스템은 3단계로 구성됩니다: 1) audio-to-motion: VAE 모델을 사용하여 오디오 특징을 중립적 얼굴 랜드마크로 변환합니다. 2) motion-to-emotion: 랜드마크 변형 모델을 통해 중립 랜드마크를 감정 랜드마크로 변환합니다. 3) emotion-to-video: NeRF 모델을 사용하여 높은 충실도의 감정 대화 머리 비디오를 렌더링합니다. 이 시스템은 감정 인식 및 표현의 정확성을 높입니다.

- **Performance Highlights**: EmoGene은 기존 방법들보다 감정 표현 생성을 더 정확하게 수행하며, 높은 그래픽 충실도와 사실성을 유지합니다. 실험 결과, EmoGene은 다양한 감정 표현 및 고충실도의 대화 머리 비디오 생성에서 이전의 작업보다 뛰어난 성능을 보여주었습니다.



### Masked Autoencoder with Swin Transformer Network for Mitigating Electrode Shift in HD-EMG-based Gesture Recognition (https://arxiv.org/abs/2410.17261)
- **What's New**: 본 논문은 Masked Autoencoder with Swin Transformer (MAST) 프레임워크를 제안하여, HD-sEMG 채널의 마스킹된 부분에 대한 학습을 수행하고 전극 이동에 대한 강인성을 높이는 방법을 다룹니다.

- **Technical Details**: MAST 프레임워크는 랜덤 블록 마스킹(random block masking), 시간 마스킹(temporal masking), 센서별 랜덤 마스킹(sensor-wise random masking), 멀티스케일 마스킹(multi-scale masking)과 같은 4가지 마스킹 전략을 결합하여 HD-sEMG의 잠재 표현(latent representations)을 학습합니다. 이 구조는 Swin-Unet 아키텍처를 기반으로 하며, 시간 도메인(time-domain), 주파수 도메인(frequency-domain), 그리고 크기 기반(magnitude-based) 특징을 동시에 캡처하여 HD-sEMG 신호의 포괄적인 이해를 제공합니다.

- **Performance Highlights**: 제안된 MAST 프레임워크는 다른 최신 기법들과 비교하여 현저히 높은 세션 간 정확도를 보이며, 계산 요구 사항이 낮아 실제적인 솔루션으로 자리 잡을 가능성이 높습니다.



### Representing Web Applications As Knowledge Graphs (https://arxiv.org/abs/2410.17258)
- **What's New**: 본 논문에서는 웹 애플리케이션의 동적이고 상호작용적인 행위를 더 잘 모델링하기 위한 새로운 방법론을 제안합니다. 각 노드는 애플리케이션의 현재 상태를 나타내며, 엣지는 사용자의 행동에 따른 전이를 반영하여, 전통적인 웹 스크레이퍼가 한계 지니고 있는 부분을 극복하고 있습니다.

- **Technical Details**: 제안된 시스템은 웹 애플리케이션의 동적 행동과 상태 전이를 캡처하기 위해, 세 가지 주요 구성 요소로 구성됩니다: 기능 추론 모듈(Functionality Inferring Module), 액션 실행기(Action Executor), 보상/벌칙 모델(Reward/Penalty Model). 이 시스템은 각 상태를 시각적 및 구조적 속성으로 정의하며, 사용자와의 상호작용을 통해 발생하는 상태 전이를 엣지로 나타냅니다.

- **Performance Highlights**: 새로운 방법론은 전통적인 방식에 비해 웹 애플리케이션의 기능을 더 잘 탐색할 수 있으며, 사용자 흐름을 포괄적으로 이해하는 데 도움을 줍니다. 이 접근법은 자동화된 테스트와 행동 분석 같은 후속 작업에서 유용한 통찰력을 제공합니다.



### Code-Driven Law NO, Normware SI! (https://arxiv.org/abs/2410.17257)
Comments:
          First version of the paper presented at CRCL 2022

- **What's New**: 이 논문은 '노름웨어(normware)'라는 개념을 도입하여 소프트웨어와 하드웨어 외에 인공지능 기기의 해석과 설계에 있어 명시적인 추가 관점을 제공하고, 기술 시스템과 인간 제도 간의 상호작용을 더 잘 연구하고 설계할 수 있는 방법을 모색합니다.

- **Technical Details**: 논문은 코드를 기반으로 한 법(code-driven law), 데이터 기반의 법(data-driven law), 텍스트 기반의 법(text-driven law)의 차이를 재조명하고, 각 법의 본질적 차이를 강조하였습니다. '노름웨어'는 법적 규범과 기대를 규제하는 기능을 하는 인공지능 및 기술 시스템의 아티팩트로서 세 가지 주요 기능을 식별합니다: 행동, 자격, 그리고 기대를 규제하는 기능입니다.

- **Performance Highlights**: 이 논문은 기술적인 영역과 관련된 주제를 다루지만, 주요 기여는 이론적인 것입니다. '노름웨어'를 통해 계산 시스템의 규제 및 기술적 중재 디자인을 위한 새로운 이해의 기초를 제시하며, 궁극적으로 사회-기술적 관점에서 넓은 범위의 법적 결정 과정에서 더 나은 신뢰성과 적절성을 달성하기 위한 방법론을 제시합니다.



### Self-Evolving Multi-Agent Collaboration Networks for Software Developmen (https://arxiv.org/abs/2410.16946)
Comments:
          25 pages

- **What's New**: EvoMAC는 MAC 네트워크를 위한 새로운 자가 진화 패러다임으로, 소프트웨어 개발 작업에서 기능 수준을 넘어서 복잡한 소프트웨어 수준의 개발을 지원합니다.

- **Technical Details**: EvoMAC는 MAC 네트워크의 출력을 검증하여 텍스트 기반 환경 피드를 얻고, 고유의 텍스트 백 프로퍼게이션 알고리즘을 통해 네트워크를 업데이트하여 에이전트와 연결을 반복적으로 적응시키는 기능을 가지고 있습니다. 또한 rSDE-Bench라는 요구 사항 지향 소프트웨어 개발 벤치마크를 제안하여 53개 코딩 작업과 616개의 요구 사항을 포함하고 있습니다.

- **Performance Highlights**: EvoMAC는 rSDE-Bench와 HumanEval에서 각각 26.48%, 34.78%, 6.10% 향상된 성능을 보여주며, rSDE-Bench의 자동 평가가 인간 평가와 대략 99.22%의 일관성을 가진다는 것을 입증했습니다.



### Non-myopic Generation of Language Model for Reasoning and Planning (https://arxiv.org/abs/2410.17195)
- **What's New**: 이번 연구에서는 예측 디코딩( Predictive-Decoding)이라는 새로운 방법을 통해 대형 언어 모델(LLMs)의 계획 정확도를 향상시키기 위한 접근 방식을 제안합니다. 이는 최적 제어(optimal-control) 관점에서 LLM의 추론을 재조명합니다.

- **Technical Details**: 예측 디코딩은 모델 예측 제어(Model Predictive Control) 기법을 활용하여 LLM의 분포를 미래 궤적(forecast trajectories)에 기반하여 재가중치합니다. 이러한 방법을 통해 초기 오류를 완화하고 비근시적(myopic) 계획을 촉진하는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과, 수학 문제 해결, 코딩, 및 에이전트 같은 다양한 작업에서 유의미한 성능 향상이 확인되었고, 예측 디코딩은 검색 기반 라인(selected baseline)보다 계산 자원을 덜 사용하면서 효율성을 입증했습니다.




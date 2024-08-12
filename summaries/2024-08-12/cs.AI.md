New uploads on arXiv(cs.CL)

### TaSL: Task Skill Localization and Consolidation for Language Model Continual Learning (https://arxiv.org/abs/2408.05200)
Comments:
          Extension of ACL 2024 paper titled: Continual Dialog State Tracking via Task Skill Localization and Consolidation

- **What's New**: 본 논문에서는 언어 모델 지속 학습(CL, Continual Learning)을 위한 새로운 프레임워크인 'Task Skill Localization and Consolidation (TaSL)'을 제시합니다. TaSL은 기존 방식의 단점을 극복하고 지식 전이(KT, Knowledge Transfer)를 향상시키기 위해 다음과 같은 핵심 기능을 도입했습니다:

* **Skill Unit**: 모델을 기능 단위(Skill Unit)로 분할하여 더 세밀한 제어를 가능하게 합니다. 각 Skill Unit은 특정 작업에 관련된 지식을 담고 있습니다.
* **Group-wise Skill Localization**: 각 작업에 대한 Skill Unit의 중요성 분포를 파악하는 기법입니다. 이를 통해 작업 특정 지식과 공유 지식을 구분할 수 있습니다.
* **Skill Consolidation**: 작업 특정 지식은 보존하고, 공유 지식은 통합하는 전략을 통해 지식 전이를 촉진하고 망각을 방지합니다.

TaSL은 기존 지식을 유지하면서 새로운 작업에서 우수한 성능을 달성하는 데 효과적입니다. 또한, LoRA와 같은 PEFT 방법에 적용 가능하며 메모리 리플레이와의 통합을 통해 성능을 더욱 향상시킬 수 있습니다.



### Separating Style from Substance: Enhancing Cross-Genre Authorship Attribution through Data Selection and Presentation (https://arxiv.org/abs/2408.05192)
- **What's New**: 이 연구는 두 문서가 같은 저자에 의해 작성되었는지 여부를 판단하는 과제를 다룹니다. 특히, 이 연구는 두 문서가 서로 다른 주제(예: 야구 vs. 정치) 또는 다른 장르(예: 블로그 게시물 vs. 학술 논문)에 대해 쓰여진 경우 이 문제가 더욱 어려워진다는 사실에 주목합니다. 이 문제는 저자가 서로 다른 주제에 대해 작성한 실제 세계의 학습 데이터가 부족하고 장르 간 데이터가 부족하기 때문에 더욱 복잡해집니다. 본 논문에서는 모델이 저자 식별에 주제 정보에 의존하는 것을 방지하고, 주제에 관계없이 스타일을 더 강력하게 반영하는 정보를 통합하도록 설계된 데이터 선택 방법과 새로운 학습 커리큘럼을 제안합니다. 이러한 개선을 통해 장르 간 저자 식별 정확도가 평균 62.7% 향상되었으며, 장르별 조건에서 16.6% 향상되었습니다.



### Deep-change at AXOLOTL-24: Orchestrating WSD and WSI Models for Semantic Change Modeling (https://arxiv.org/abs/2408.05184)
- **What's New**: 이 논문은 AXOLOTL-24 공유 작업에서 의미 변화 모델링의 첫 번째 하위 작업에 대한 솔루션을 설명합니다. 이 하위 작업의 목표는 새 시대의 다의어 단어의 사용을 이전 시대의 단어의 의미와 단어의 새로 얻은 의미를 나타내는 클러스터 사이에 분산하는 것입니다. 저자는 이 작업을 해결하는 세 가지 새로운 방법을 제안하고 실험했습니다. 저자의 방법은 첫 번째 하위 작업의 공식 지표 모두에 따라 최첨단 결과를 달성했습니다. 또한 저자는 제공된 의미 정의 중 어느 것으로도 설명되지 않는 단어 사용인지 여부를 판단할 수 있는 모델을 개발했습니다. 이 모델은 저자의 방법 중 하나의 구성 요소 역할을 하지만 잠재적으로 자체적으로 유용할 수 있습니다.



### A Hybrid RAG System with Comprehensive Enhancement on Complex Reasoning (https://arxiv.org/abs/2408.05141)
Comments:
          Technical report for 3rd prize in Task 1 of Meta CRAG KDD Cup 2024

- **What's New**: 본 논문에서는 Retrieval-Augmented Generation (RAG) 시스템을 소개합니다. 이 시스템은 외부 지식베이스를 통합하여 대규모 언어 모델(LLM)의 정확성을 향상시키고 환각(hallucination)을 줄이는 데 도움이 됩니다. 시스템은 웹 페이지의 텍스트 조각과 테이블을 개선하고, 환각을 줄이기 위해 속성 예측기를 추가하며, LLM 지식 추출기와 지식 그래프 추출기를 수행하고, 마지막으로 모든 참조를 사용하여 추론 전략을 구축합니다.

- **Technical Details**: 제안된 RAG 시스템은 다음과 같은 4가지 단계로 구성됩니다:
1) **사전 검색** 단계: 외부 지식 소스를 색인화하고 쿼리를 조작하며 데이터를 수정하는 단계입니다. 특히, 향후 검색 실행의 효율성을 높이기 위해 외부 지식 소스를 컴파일하고 색인화하고, 외부 데이터 분포와 일치하도록 쿼리를 개선하고, 추가 추론을 위한 관련 지식을 포괄하는 일관된 표현을 생성하기 위해 데이터 소스를 수정합니다.
2) **검색** 단계: 쿼리와 외부 지식베이스 사이의 유사성을 측정하여 관련 텍스트 문서를 검색하는 단계입니다. 코사인 유사도와 같은 지표를 사용하여 검색을 수행합니다.
3) **사후 검색** 단계: 검색된 문서에서 추가 정보를 추출하고 처리하여 LLM이 추론 및 생성 작업에 더 잘 활용할 수 있도록 하는 단계입니다.
4) **생성** 단계: LLM이 쿼리와 검색된 문서를 사용하여 답변을 생성하는 단계입니다.

- **Performance Highlights**: 본 논문에서 제안된 RAG 시스템은 CRAG 데이터 세트에서 평가되었습니다. 시스템은 Task 1에서 3위를 차지했으며 Task 2에서는 7,777개의 질문 유형 중 5,555개 유형에서 1위를 차지했습니다. 지역 평가와 온라인 평가 모두에서 시스템이 복잡한 추론 능력을 크게 향상시킨다는 것을 보여줍니다. 지역 평가에서 시스템은 기준 모델에 비해 정확도를 크게 향상시키고 오류율을 감소시켜 점수가 상당히 증가했습니다. 한편, 시스템은 온라인 평가에서 뛰어난 결과를 달성하여 제안된 시스템의 성능과 일반화 능력을 보여줍니다.



### How Well Do LLMs Identify Cultural Unity in Diversity? (https://arxiv.org/abs/2408.05102)
Comments:
          COLM 2024

- **What's New**: 본 논문에서는 대규모 언어 모델(LLMs)의 문화적 통합(cultural unity)에 대한 이해 능력을 평가하기 위한 새로운 벤치마크 데이터셋인 CUNIT을 소개합니다. CUNIT은 10개국의 285개 전통 문화 특정 개념을 기반으로 1,425개의 평가 예제로 구성되어 있습니다. 이 연구는 지리 문화적 다양성(geo-cultural diversity)에 대한 모델의 민감성에 초점을 맞춘 기존 연구와 달리, 문화적으로 중심적인 개념 간의 연관성(cultural-centered associations)을 파악하는 LLMs의 능력에 초점을 맞춥니다.

- **Technical Details**: CUNIT은 각 개념에 대한 문화적 관련성 있는 기능(cultural-relevant features)을 체계적으로 수동 주석 처리하여 교차 문화적 개념 쌍 간의 문화적 연관성(cultural association)을 계산합니다. 이 데이터셋을 기반으로 LLMs의 높은 연관성을 가진 교차 문화적 개념 쌍을 식별하는 능력을 평가하기 위한 대조적 매칭 작업(contrastive matching task)을 설계했습니다. 이 연구에서는 3가지 인기 있는 프롬프팅 전략을 사용하여 3가지 강력한 LLM을 평가했습니다. 또한, LLMs에 문화적 개념의 추출된 특징을 모두 제공하거나 전혀 제공하지 않는 두 가지 설정에서 CUNIT을 사용하여 평가를 진행했습니다.

- **Performance Highlights**: 흥미롭게도, 의복 개념에 대한 국가 간 문화적 연관성은 음식에 대한 문화적 연관성과 크게 다릅니다. 분석 결과, LLMs는 인간과 비교하여 개념 간 교차 문화적 연관성을 포착하는 데 여전히 제한적이라는 사실을 보여줍니다. 또한, 지리 문화적 근접성(geo-cultural proximity)은 교차 문화적 연관성을 포착하는 모델 성능에 약한 영향을 미치는 것으로 나타났습니다.



### MooER: LLM-based Speech Recognition and Translation Models from Moore Threads (https://arxiv.org/abs/2408.05101)
- **What's New**: MooER, a novel LLM-based large-scale automatic speech recognition (ASR) / automatic speech translation (AST) model developed by Moore Threads, is presented. This model is trained on a 5000h pseudo-labeled dataset, demonstrating comparable performance to open source models trained on hundreds of thousands of hours of labeled data.  MooER outperforms other open source Speech LLMs, achieving a BLEU score of 25.2 on the Covost2 Zh2en testset.  It is noteworthy that MooER is the first speech large-scale model to utilize domestically produced GPUs for training and inference, showcasing its potential for industrial application.



### Unlocking Decoding-time Controllability: Gradient-Free Multi-Objective Alignment with Contrastive Prompts (https://arxiv.org/abs/2408.05094)
- **What's New**: MCA (Multi-objective Contrastive Alignment), a new method for multi-objective alignment of large language models (LLMs), is proposed. MCA focuses on achieving control over various alignment objectives (e.g., helpfulness, harmlessness, honesty) without requiring additional model training. Unlike prior methods that rely on training numerous models or modifying model parameters, MCA utilizes contrastive decoding with expert and adversarial prompts.  MCA contrasts the responses generated by these prompts at decoding time, allowing users to dynamically control the alignment objectives.

- **Technical Details**: MCA constructs expert and adversarial prompts for each alignment objective. These prompts are derived from responses with maximum and minimum rewards for the respective objective. The language model's predictions from these prompts are contrasted in the logit space, enabling users to adjust the balance between objectives by manipulating the contrast weight. This approach provides a gradient-free solution that requires no model parameter updates and offers flexibility in incorporating new alignment objectives during decoding.

- **Performance Highlights**: Empirical results demonstrate MCA's superior performance in balancing multiple alignment objectives, achieving a well-distributed Pareto front. This approach outperforms existing methods, offering a more efficient and adaptable way to control LLMs for personalized preferences.



### Order Matters in Hallucination: Reasoning Order as Benchmark and Reflexive Prompting for Large-Language-Models (https://arxiv.org/abs/2408.05093)
Comments:
          7 pages, submitted to AAAI25

- **What's New**: 이 논문은 대규모 언어 모델(LLM)의 일관성을 평가하기 위한 새로운 벤치마크 방법인 **Reasoning Order as Benchmark**를 제시합니다. 이 벤치마크는 LLM이 먼저 답변을 생성한 다음 추론 과정을 제공하는 방식과 먼저 추론 과정을 생성한 다음 결론을 제공하는 방식의 차이를 비교하여 모델의 추론 논리의 일관성을 반영합니다. 또한, 이러한 문제를 완화하기 위해 **Reflexive Prompting**이라는 새로운 프롬프트 전략을 소개합니다. Reflexive Prompting은 먼저 답변 우선 프롬프트와 논리 우선 프롬프트를 사용하여 두 개의 잠재적으로 다른 답변을 얻습니다. 그런 다음 두 답변을 분석하여 최종 답변을 도출합니다. 실험 결과는 이 방법이 다양한 데이터 세트와 LLM 아키텍처에서 추론 작업의 정확도를 향상시키는 것을 보여줍니다. 또한, 다른 프롬프트 전략을 사용하여 달성한 정확도는 Reasoning Order as Benchmark 테스트 결과와 강한 상관 관계가 있어 이 벤치마크 방법의 유용성을 입증합니다.



### Generating novel experimental hypotheses from language models: A case study on cross-dative generalization (https://arxiv.org/abs/2408.05086)
- **What's New**: This research uses neural network language models (LMs) as simulated learners to explore hypotheses about how children learn dative constructions in English. This approach provides a way to systematically investigate the vast space of possible cues that children could use to learn which verbs can be used in different dative structures.  Specifically, it aims to derive novel hypotheses from LMs that can be tested with real children.



### RT-Surv: Improving Mortality Prediction After Radiotherapy with Large Language Model Structuring of Large-Scale Unstructured Electronic Health Records (https://arxiv.org/abs/2408.05074)
Comments:
          23 pages, 2 tables, 4 figures

- **What's New**: 본 연구는 방사선 치료(RT)의 효과적인 환자 선별을 위한 새로운 접근 방식을 제시합니다. 기존의 생존 예측 모델은 구조화된 데이터에 의존하여 정확성이 제한적이었습니다. 이 연구는 대규모 언어 모델(LLM)을 활용하여 비구조화된 전자 건강 기록(EHR) 데이터를 구조화하고, 포괄적인 임상 정보 통합을 통해 생존 예측 정확도를 향상시키는 가능성을 탐구합니다.



### Examining the Behavior of LLM Architectures Within the Framework of Standardized National Exams in Braz (https://arxiv.org/abs/2408.05035)
Comments:
          Accepted at the Seventh AAAI/ACM Conference on AI, Ethics and Society (AIES 2024). 14 pages, 4 figures

- **What's New**: This research uses the Brazilian ENEM (Exame Nacional do Ensino Médio) dataset to analyze the biases of large language models (LLMs) in Portuguese. It compares the performance of GPT-3.5, GPT-4, and MariTalk, a Portuguese-trained model, to human test takers, considering their socioeconomic status (SES).

- **Technical Details**: The study analyzes the LLMs' performance on both multiple-choice questions (Math, Humanities, Natural Sciences, Languages) and essay writing. It also examines the linguistic complexity of the generated essays.

- **Performance Highlights**: The study finds no significant biases in the LLMs' performance on the multiple-choice questions, suggesting that their performance is more influenced by their overall accuracy than by specific SES groups. However, the generated essays show significant differences from human essays, with the LLMs exhibiting distinct word choices and syntactic complexities. The findings indicate that LLMs in Brazilian Portuguese do not represent any specific human group and are significantly different from human performance.

- **Limitations**: The study focuses solely on Brazilian Portuguese and doesn't generalize to other languages. The research examines biases based on SES but does not explore other potential biases, such as gender or racial biases.

- **Future Directions**: The research highlights the need for further studies on LLM biases in different languages and cultural contexts.  Exploring the potential impacts of  LLM biases on various applications, particularly in education and social settings, is crucial. 



### Investigating a Benchmark for Training-set free Evaluation of Linguistic Capabilities in Machine Reading Comprehension (https://arxiv.org/abs/2408.05023)
- **What's New**: 본 논문은 기존 MRC(Machine Reading Comprehension) 평가 방식의 한계를 지적하고, 인공적으로 생성된 챌린지 세트를 사용하여 모델의 언어 이해 능력을 평가하는 새로운 프레임워크를 제시합니다. 특히, 챌린지 세트 데이터를 활용하여 MRC 모델의 언어적 능력을 평가하는 방식을 심층적으로 분석합니다.



### ProFuser: Progressive Fusion of Large Language Models (https://arxiv.org/abs/2408.04998)
- **What's New**: 본 논문에서는 여러 대규모 언어 모델(LLM)의 장점을 결합하여 더욱 강력하고 다재다능한 모델을 구축하는 새로운 방법을 제시합니다. 기존의 모델 융합 방법들은 주로 teacher-forcing 설정에서 지도 학습(ground truth)에 대한 교차 엔트로피(cross entropy)를 사용하여 모델의 장점을 평가하는 데 중점을 두었습니다. 그러나 이러한 방법은 모델의 장점에 대한 제한적인 통찰력만을 제공할 수 있습니다. 본 논문에서는 훈련 모드와 추론 모드를 모두 포함하는 새로운 방법을 제시하여 모델 융합 프로세스를 개선합니다. 제안된 방법은 훈련 중에 교차 엔트로피를 통해 모델의 장점을 평가할 뿐만 아니라 추론 출력을 고려하여 보다 포괄적인 평가를 제공합니다. 두 모드를 효과적으로 결합하기 위해 본 논문에서는 추론 모드에서 훈련 모드로 점진적으로 전환하는 ProFuser를 도입합니다. ProFuser의 효과를 검증하기 위해 vicuna-7b-v1.5, Llama-2-7b-chat, mpt-7b-8k-chat을 포함한 세 가지 모델을 융합하고 기준 방법과 비교하여 지식, 추론 및 안전성 측면에서 향상된 성능을 보여줍니다.



### Get Confused Cautiously: Textual Sequence Memorization Erasure with Selective Entropy Maximization (https://arxiv.org/abs/2408.04983)
Comments:
          15 pages, 7 figures

- **What's New**: 이 논문은 **엔트로피 극대화 선택적 최적화(EMSO)**라는 새로운 프레임워크를 제안하여 LLM의 텍스트 시퀀스 메모리(TSM)를 지우는 동시에 모델 유용성을 보존합니다.

- **Technical Details**: EMSO는 LLM에서 메모리 된 텍스트 시퀀스를 제거하는 새로운 방법입니다. 기존의 방법들은 대규모 메모리 된 샘플을 지우는 데 실패하거나 모델 유용성을 심각하게 저해했습니다. EMSO는 엔트로피 극대화와 선택적 최적화를 결합하여 이 문제를 해결합니다. 엔트로피 극대화 손실을 통해 모델이 더 다양한 출력을 생성하도록 장려하고, 선택적 최적화는 엔트로피 극대화에 가장 큰 영향을 미치는 가중치만 업데이트하여 모델의 유용성을 보존합니다.  EMSO는 **대조적 기울기 메트릭**을 사용하여 **가장 영향력 있는 가중치**를 찾아내 TSM 지우기에 사용합니다.

- **Performance Highlights**: 실험 결과, EMSO는 다양한 모델 규모와 메트릭에 걸쳐 **대규모 잊기 데이터 세트**에서 정보 유출과 모델 유용성 간의 최상의 절충안을 달성하는 것으로 나타났습니다. 또한 EMSO는 **기존 방법보다 안정적인 최적화 프로세스**를 가지고 있으며, **모델 유용성을 더 잘 보존**하는 것으로 나타났습니다.

- **Additional Information**: EMSO는 **참조 모델 없이 작동**하기 때문에 개인정보 보호 문제를 일으키지 않습니다. 또한, **유지 데이터 세트가 필요하지 않아** 실용적입니다. 이 논문은 **TSM 지우기와 모델 유용성** 간의 균형을 맞추는 데 중요한 기여를 할 것으로 기대됩니다.



### \textit{re}CSE: Portable Reshaping Features for Sentence Embedding in Self-supervised Contrastive Learning (https://arxiv.org/abs/2408.04975)
- **What's New**: This paper introduces “reCSE”, a novel self-supervised contrastive learning framework for sentence representation. Unlike existing methods that rely on discrete data augmentation, reCSE focuses on “feature reshaping” to enhance sentence understanding without requiring supplementary samples. This approach aims to address the issues of representation polarity and high GPU memory consumption associated with conventional data augmentation techniques.



### Generalisation First, Memorisation Second? Memorisation Localisation for Natural Language Classification Tasks (https://arxiv.org/abs/2408.04965)
Comments:
          Published in ACL Findings 2024; 19 pages total (9 in the main paper, 4 pages with limitations, acknowledgments and references, 6 pages with appendices)

- **What's New**: 본 논문에서는 12개의 자연어 분류 작업에서 레이어별 메모리 국소화(memorisation localisation)를 수행하여 딥 신경망에서 메모리가 어떻게 국소화되는지에 대한 통찰력을 제공합니다. 기존 연구와 달리 메모리 국소화가 특정 레이어에 국한되지 않고 점진적인 과정임을 밝혀냈으며, 메모리 국소화는 작업 의존적(task-dependent)임을 보여줍니다. 또한, 일반화 우선, 메모리 후순(generalisation first, memorisation second) 가설에 대한 미묘한 차이점을 제시합니다.



### HybridRAG: Integrating Knowledge Graphs and Vector Retrieval Augmented Generation for Efficient Information Extraction (https://arxiv.org/abs/2408.04948)
Comments:
          9 pages, 2 figures, 5 tables

- **What's New**: This paper introduces a novel approach called **HybridRAG**, which combines the strengths of **VectorRAG** (using vector databases) and **GraphRAG** (using Knowledge Graphs) for enhancing Question-Answer (Q&A) systems in the context of financial document analysis.

- **Technical Details**: HybridRAG leverages both vector databases and knowledge graphs to retrieve relevant information for a given query. It addresses the limitations of traditional VectorRAG and GraphRAG techniques in financial document analysis, particularly the challenges posed by domain-specific terminology, complex document formats, and the need for accurate context retrieval.

- **Performance Highlights**: Experiments conducted on a dataset of financial earning call transcripts demonstrate that HybridRAG outperforms both traditional VectorRAG and GraphRAG individually. This improvement is observed in both the retrieval and generation stages, leading to higher retrieval accuracy and more contextually relevant answers.

- **Applications**: This technique has broad applicability beyond the financial domain, making it potentially valuable for various information extraction tasks across different industries.



### Quantitative Information Extraction from Humanitarian Documents (https://arxiv.org/abs/2408.04941)
- **What's New**: This paper introduces a new dataset for quantitative information extraction in humanitarian documents, along with a custom NLP pipeline for extracting quantities, units, and context. This dataset is annotated with detailed information about the quantities, including units, modifiers, and relevant events, and it aims to improve NLP tools for humanitarian crisis response and decision-making.



### Surveying the Landscape of Image Captioning Evaluation: A Comprehensive Taxonomy and Novel Ensemble Method (https://arxiv.org/abs/2408.04909)
- **What's New**: 본 연구는 이미지 캡셔닝 평가 지표 70개 이상을 조사하여 최초의 분류 체계를 제시합니다.  이 연구는 300개 이상의 논문에 사용된 지표들을 분석하여 다양한 지표들이 존재함에도 불구하고 실제로 많이 사용되는 지표는 5개뿐이며, 이러한 지표들은 사람의 평가와 약한 상관관계를 갖는다는 점을 밝혀냈습니다.  본 연구에서는 다양한 지표들을 활용하여 사람의 평가와 가장 높은 상관관계를 보이는 EnsembEval이라는 새로운 앙상블 평가 지표를 제안합니다.  이를 통해 다양한 지표들을 활용하는 것이 이미지 캡셔닝 모델 평가의 정확성을 높이는 데 중요하다는 점을 보여줍니다.



### Towards a Generative Approach for Emotion Detection and Reasoning (https://arxiv.org/abs/2408.04906)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)을 이용하여 텍스트에서 감정을 감지하고 감정적 추론을 수행하는 새로운 제너레이티브 접근 방식을 소개합니다. 기존의 제로샷 감정 감지 방법은 텍스트 함축 모델에 의존하여 입력 텍스트에 가장 적합한 감정 레이블을 선택하는 반면, 이 논문은 감정 분석을 제너레이티브 질문 답변(QA) 작업으로 재구성하는 접근 방식을 제시합니다. 이 접근 방식은 감정 감지 질문에 단계별로 답변할 수 있는 관련 맥락 또는 배경 지식을 생성하는 두 단계 방법론을 사용합니다. 이 논문은 텍스트에 대한 감정 감지와 감정적 추론 작업을 함께 해결하기 위해 제너레이티브 접근 방식을 사용하는 최초의 연구입니다.



### GlitchProber: Advancing Effective Detection and Mitigation of Glitch Tokens in Large Language Models (https://arxiv.org/abs/2408.04905)
- **What's New**: 본 논문에서는 '글리치 토큰'(glitch token)이라 불리는 특수 토큰의 영향을 조사하고, 탐지 및 완화 기술을 제시합니다. 글리치 토큰은 LLM(대규모 언어 모델)의 어휘 공간에서 발견되는 비정상적인 토큰으로, 모델의 입력에 포함될 경우 잘못되거나 무관한 결과, 또는 심지어 유해한 결과를 초래할 수 있습니다. 본 연구에서는 LLM의 내부 메커니즘에 미치는 글리치 토큰의 영향에 대한 이해를 심화시키고, 효과적인 탐지 및 완화 기법을 제안합니다.

- **Technical Details**: 본 연구는 LLM의 중간층에서 나타나는 주의 패턴(attention patterns)과 동적 정보(dynamic information) 분포의 차이를 통해 글리치 토큰과 정상 토큰을 구별합니다. 이러한 분석을 기반으로, GlitchProber라는 새로운 도구를 개발했습니다. GlitchProber는 소규모 샘플링, 주성분 분석(PCA: Principal Component Analysis) 기반의 특징 추출, 그리고 간단한 분류기를 사용하여 효율적으로 글리치 토큰을 탐지하고 완화합니다. GlitchProber는 또한 비정상적인 모델 중간층 값을 수정하여 글리치 토큰의 부정적 영향을 완화합니다.

- **Performance Highlights**: GlitchProber는 5개의 주요 오픈소스 LLM을 대상으로 평가한 결과, 기존 방법에 비해 더 높은 효율성, 정확도, 재현율을 보였습니다. GlitchProber는 평균 F1 점수 0.86과 평균 복구율 50.06%를 기록했습니다. GlitchProber는 글리치 토큰의 문제를 해결하는 새로운 방식을 제시하며, 더욱 강력하고 해석 가능한 LLM에 대한 향후 연구에 영감을 줄 수 있습니다.



### Communicate to Play: Pragmatic Reasoning for Efficient Cross-Cultural Communication in Codenames (https://arxiv.org/abs/2408.04900)
- **What's New**: This research introduces RSA+C3, a new method for cross-cultural communication in AI systems, particularly for resolving pragmatic failures due to differing common ground. This method is evaluated in the context of the Codenames Duet game.



### Unsupervised Episode Detection for Large-Scale News Events (https://arxiv.org/abs/2408.04873)
- **What's New**: 이 논문은 뉴스 기사 모음에서 에피소드를 식별하는 새로운 작업인 '에피소드 감지'를 소개합니다. 에피소드는 특정 시간과 장소에서 행동을 수행하는 핵심 엔터티(예: '시위자', '경찰')의 일관성 있는 클러스터를 설명합니다. 또한 에피소드는 특정 주요 이벤트 아래의 더 큰 에피소드 그룹의 중요한 부분입니다.

- **Technical Details**: EpiMine이라는 새로운 에피소드 감지 프레임워크가 소개되었습니다. EpiMine은 (1) 가장 두드러지고 주요 이벤트와 관련된 용어와 세그먼트를 자동으로 식별하고, (2) 차별적 용어 조합의 변화를 통해 추정된 자연적 에피소드 분할을 기반으로 기사에서 후보 에피소드를 결정하고, (3) 후보 에피소드에 대한 대규모 언어 모델 기반 추론을 사용하여 최종 에피소드 클러스터를 개선하고 형성합니다.

- **Performance Highlights**: EpiMine은 세 가지 다양한 실제 세계 이벤트 데이터 세트에서 모든 메트릭에서 평균 59.2% 증가하여 모든 기준선을 능가합니다.

- **Datasets**: EpiMine은 30개의 글로벌 주요 이벤트를 반영하는 세 가지 새로운 데이터 세트를 구축했습니다. 이 데이터 세트는 각 주요 이벤트가 구별 가능한 에피소드를 포함하도록 보장하는 이 작업에 대한 대규모 주요 이벤트 특정 뉴스 코퍼스가 존재하지 않기 때문에 다양한 실제 세계 주제를 반영합니다.



### SCOI: Syntax-augmented Coverage-based In-context Example Selection for Machine Translation (https://arxiv.org/abs/2408.04872)
Comments:
          16 pages, 2 figures, 14 tables

- **What's New**: 본 논문에서는 기계 번역(MT)을 위한 인 컨텍스트 학습(ICL)에서 더 나은 인 컨텍스트 예제를 선택하기 위해 구문 지식을 도입하는 새로운 전략, 즉 구문 증강 커버리지 기반 인 컨텍스트 예제 선택(SCOI)을 제안합니다. 기존의 단어 일치를 넘어 심층 구문 구조를 활용하여 세트 수준의 구문 커버리지를 측정합니다. 구문 커버리지는 단순화된 트리-투-폴리노미얼 알고리즘을 사용하여 폴리노미얼 항의 커버리지를 계산하고, 어휘 커버리지는 단어 중복을 사용하여 측정합니다. 또한 두 가지 커버리지 측정 방식을 결합하는 대체 선택 방식을 고안하여 구문 및 어휘 정보를 활용합니다.

- **Technical Details**: SCOI는 구문 커버리지를 측정하기 위해 트리-투-폴리노미얼 알고리즘을 단순화하여 대규모 MT 데이터 세트에서 실용적으로 실행할 수 있도록 했습니다. 단순화된 알고리즘을 사용하여 의존성 트리를 폴리노미얼로 변환하고 폴리노미얼 항의 벡터 표현의 커버리지를 계산합니다. 세트 수준의 어휘 커버리지를 측정하기 위해 단어 중복 비율을 고려합니다. 그런 다음 어휘 및 구문 지식을 활용하는 대체 전략을 설계합니다.

- **Performance Highlights**: 6개의 번역 방향(독일어, 프랑스어, 러시아어에서 영어로, 그리고 영어에서 독일어, 프랑스어, 러시아어로)에 대한 실험 결과, SCOI는 모든 학습 없는 방법 중에서 4개의 번역 방향에서 가장 높은 COMET 점수를, 그리고 가장 높은 평균 COMET 점수를 얻었습니다. 특히, 러시아어-영어 및 영어-러시아어 번역에서 SCOI는 Alpaca를 사용할 때 학습 기반 CTQ Scorer보다 더 나은 성능을 보였습니다.



### MSG-Chart: Multimodal Scene Graph for ChartQA (https://arxiv.org/abs/2408.04852)
Comments:
          Accpeted by CIKM Short 2024

- **What's New**: This paper proposes a novel multimodal scene graph approach for ChartQA. This graph representation, encompassing both visual and textual aspects, enhances the understanding of chart elements' relationships and patterns.

- **Technical Details**: The proposed model comprises two graphs: a visual graph capturing spatial relations based on visual features and a textual graph representing semantic knowledge using textual features. The visual graph utilizes a fully connected structure with edge weights determined by the Euclidean distance between objects, prioritizing closer neighbors. Each visual node is initialized with the mean of hidden states from corresponding image patches. The textual graph consists of label nodes and OCR nodes, connected based on chart semantics. Label nodes represent object labels, while OCR nodes capture OCR text extracted from non-shape objects. Connections are established based on chart structure, such as x/y axis titles to labels, x/y axis labels to chart elements, and legend labels to markers. Each node is initialized with the mean of BERT embeddings of its corresponding text.

- **Performance Highlights**: Experiments demonstrate that incorporating the proposed graph module improves performance on public benchmarks, ChartQA and OpenCQA, surpassing previous approaches. This indicates that the multimodal scene graph effectively enhances the understanding of chart elements' structure and semantics, leading to better question-answering accuracy.



### Ensemble BERT: A student social network text sentiment classification model based on ensemble learning and BERT architectur (https://arxiv.org/abs/2408.04849)
- **What's New**: 본 논문은 중학생의 정신 건강 평가를 위해 BERT 기반의 앙상블 학습 네트워크를 새롭게 제안합니다. 이 네트워크는 다수의 분류기를 통합하여 모델 성능을 향상시키는 개념을 활용합니다. 다양한 BERT 기반 학습기를 훈련시킨 후, 다수결 투표 방식을 사용하여 이들을 결합했습니다. 중국 Weibo에서 중학생들의 소셜 네트워크 텍스트 데이터를 수집하여 중학생 소셜 네트워크 텍스트의 감정 경향 분류 작업에 적용했습니다. 



### FUSE-ing Language Models: Zero-Shot Adapter Discovery for Prompt Optimization Across Tokenizers (https://arxiv.org/abs/2408.04816)
Comments:
          Published as a Conference Paper at COLM 2024; 10 Pages; this https URL

- **What's New**: 본 연구에서는 FUSE (Flexible Unification of Semantic Embeddings)를 제안합니다. FUSE는 토크나이저가 다르더라도, 모델의 텍스트 임베딩 공간을 다른 모델의 임베딩 공간으로 매핑하는 저렴한 어댑터 레이어를 근사하는 방법입니다. 연구팀은 모델의 임베딩 공간을 3차 텐서로 표현하여, 서로 다른 토크나이저에 의해 분리된 의미적 임베딩을 정렬하는 방법을 제시하며, 이를 통해 다른 모델의 임베딩 공간에 대한 한 모델의 출력 기울기를 근사합니다.



### Hybrid Student-Teacher Large Language Model Refinement for Cancer Toxicity Symptom Extraction (https://arxiv.org/abs/2408.04775)
- **What's New**: 본 연구는 컴팩트한 대규모 언어 모델(LLM)을 사용하여 암 독성 증상 추출을 위한 새로운 반복적 개선 접근 방식을 제시합니다. 학생-교사 아키텍처를 활용하여, 학생 모델 (Zephyr-7b-beta 및 Phi3-mini-128)과 교사 모델(GPT-4o)을 사용하여 프롬프트 개선, 검색 기반 생성 (RAG), 미세 조정 전략 중에서 동적으로 선택합니다. 294개의 임상 노트를 대상으로 한 실험 결과, 이 접근 방식의 효과를 확인했습니다.



### Understanding the Performance and Estimating the Cost of LLM Fine-Tuning (https://arxiv.org/abs/2408.04693)
Comments:
          10 pages, conference

- **What's New**: 본 논문은 제한된 GPU 자원으로 특정 작업에 대하여 대규모 언어 모델(LLM)을 효율적으로 특화시키는 방법인 LLM 미세 조정의 성능을 특성화합니다. 특히, GPU 하나에서의 성능과 실행 시간 성능을 이해하기 위해,  Sparse Mixture of Experts (MoE) 기반 LLM 미세 조정에 대한 연구를 진행합니다.

- **Technical Details**: 본 논문은 두 가지 MoE 모델 (Mixtral-8x7B, BlackMamba-630M/2.8B)과 두 가지 도메인 특정 데이터 세트 (commonsense_15k, Math_14k)를 사용하여 LLM 미세 조정을 평가합니다. 이 연구는 밀집 및 희소 MoE 모델의 학습 효율과 런타임 특성을 비교하며, 이는 최대 배치 크기, 실행 시간 분류, 종단 간 처리량, GPU 하드웨어 사용률, 부하 분산을 포함합니다.

- **Performance Highlights**: 결과는 다음과 같은 주요 통찰력을 제공합니다. 1) 미세 조정은 10회 미만의 epoch 내에 달성할 수 있으며, 전문가의 하위 집합을 활성화하는 희소 MoE 모델은 밀집형 모델과 동일하게 학습할 수 있습니다. 2) MoE 계층은 LLM 미세 조정에서 가장 많은 실행 시간을 차지하며 MoE 계층 성능을 최적화하는 것은 LLM 미세 조정의 전반적인 비용을 개선하는 데 중요합니다. 3) 희소 MoE 모델은 더 큰 배치 크기를 지원하여 종단 간 처리량을 향상시킵니다. 4) 배치 크기가 증가함에 따라 워크로드는 계산 제한이 됩니다. 5) 희소 모델의 미세 조정은 더 많은 부하 불균형으로 이어집니다. 이러한 통찰력을 바탕으로 본 논문은 모델 크기, 데이터 세트 크기 및 GPU 아키텍처를 기반으로 LLM 미세 조정 비용을 추정하는 분석 모델을 제시합니다.

- **Analytical Model**: 본 논문은 GPU 메모리를 고려하여 최대 배치 크기를 추정하고 미세 조정 처리량을 계산합니다. 실험 결과를 통해 이 처리량이 검증되었으며, RMSE는 0.55 미만입니다. 추정된 처리량을 사용하여 이 모델은 다양한 클라우드 제공업체에 대한 미세 조정 비용을 계산합니다.



### Improving Relational Database Interactions with Large Language Models: Column Descriptions and Their Impact on Text-to-SQL Performanc (https://arxiv.org/abs/2408.04691)
- **What's New**: 이 논문은 관계형 데이터베이스(Relational database)의 설명력이 부족한 컬럼 묘사(Column description) 문제를 해결하기 위해 대규모 언어 모델(LLM)을 활용하여 정보가 풍부한 컬럼 묘사를 자동으로 생성하는 방법을 제시합니다. 이는 사람과 Text-to-SQL 모델 모두에게 데이터베이스의 이해도를 높여줍니다.

- **Technical Details**: 본 연구에서는 BIRD-Bench 개발 셋을 기반으로 LLM과 사람의 수정을 거쳐 컬럼 묘사가 포함된 새로운 데이터셋인 ColSQL을 만들었습니다. 다양한 모델을 평가한 결과, GPT-4o와 Command R+가 고품질 컬럼 묘사를 생성하는 데 뛰어난 성능을 보였습니다. LLM을 심판(judge)으로 활용하여 모델 성능을 평가하는 방법도 시도했지만, 사람의 평가와 일치하지 않았으며 더 많은 연구가 필요합니다. 컬럼 묘사를 추가하면 특히 정보가 부족한 컬럼에서 Text-to-SQL 작업의 정확도를 크게 향상시키는 것으로 나타났습니다.

- **Performance Highlights**: LLM은 컬럼 묘사 생성에 매우 효과적이며, 특히 정보가 부족한 컬럼에서 Text-to-SQL 정확도를 향상시킵니다. GPT-4o와 Command R+는 고품질 컬럼 묘사를 생성하는 데 탁월한 성능을 보여주었습니다. 이 연구는 LLM이 데이터베이스 사용성을 개선하는 데 도움이 되는 상세한 메타데이터(Metadata) 생성 도구임을 입증합니다.



### Multi-Turn Context Jailbreak Attack on Large Language Models From First Principles (https://arxiv.org/abs/2408.04686)
- **What's New**: This paper proposes a new multi-turn jailbreak attack method called Context Fusion Attack (CFA) that leverages contextual scenarios to effectively bypass LLM security measures. This method dynamically integrates the target into contextual scenarios, concealing malicious intent by replacing harmful key terms with innocuous ones.



### ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities (https://arxiv.org/abs/2408.04682)
- **What's New**: ToolSandbox는 상태 유지(stateful), 대화형(conversational) 도구 사용 벤치마크로, 상태 유지 도구 간의 암시적 의존성(implicit state dependencies), LLM 시뮬레이션 사용자(LLM simulated user), 역동적인(dynamic) 평가 전략을 포함합니다. 이를 통해 도구 사용 LLM 기능의 포괄적인 평가를 제공합니다.



### Conversational AI Powered by Large Language Models Amplifies False Memories in Witness Interviews (https://arxiv.org/abs/2408.04681)
- **What's New**: 이 연구는 AI가 인간의 거짓 기억 형성에 미치는 영향을 조사했습니다. 특히, AI 대화 시스템(챗봇)과의 상호 작용을 통해 거짓 기억이 유발되는 과정을 살펴봤으며, 이는 범죄 현장 증인 인터뷰를 시뮬레이션한 것입니다. 연구에서는 4가지 조건, 즉 통제 조건, 설문 조사 기반 조건, 사전 설정된 챗봇 조건, 그리고 대규모 언어 모델(LLM) 기반 생성형 챗봇 조건을 비교 분석했습니다.



### Dynamic Fog Computing for Enhanced LLM Execution in Medical Applications (https://arxiv.org/abs/2408.04680)
- **What's New**: This paper presents **SpeziLLM**, an open-source framework for executing Large Language Models (LLMs) in a decentralized fog computing architecture, addressing the privacy, trust, and cost concerns associated with centralized cloud-based LLM platforms in healthcare.



### Towards Linguistic Neural Representation Learning and Sentence Retrieval from Electroencephalogram Recordings (https://arxiv.org/abs/2408.04679)
- **What's New**: 이 논문에서는 뇌파 신호를 문장으로 변환하는 새로운 접근 방식인 **EEG-to-Text Retrieval (ETER)**를 제안합니다. ETER는 **과도한 사전 훈련된 LLM 디코더에 대한 의존성을 제거**하고 **EEG 인코더가 텍스트 읽기 EEG 데이터에서 학습한 의미 정보를 평가**할 수 있도록 합니다. 이를 위해 **마스크 콘트라스트 학습 손실**을 사용하여 컨포머 기반 EEG 인코더를 훈련하여 의미론적 EEG 표현을 학습합니다.



### CREST: Effectively Compacting a Datastore For Retrieval-Based Speculative Decoding (https://arxiv.org/abs/2408.04678)
- **What's New**: CREST (Compact Retrieval-Based Speculative Decoding) is a new approach for efficient speculative decoding in large language models (LLMs). It enhances the existing REST (Retrieval-Based Speculative Decoding) method by redesigning the datastore structure to effectively 'compact' it while maintaining or even improving performance.

- **Technical Details**: CREST decouples n-grams from each other in the datastore, allowing selective storage of the most frequent and smallest n-grams. This compaction strategy leads to reduced storage space and surprisingly, improved acceptance length.

- **Performance Highlights**: CREST outperforms REST in terms of storage efficiency and performance. It achieves a 10.6-13.5x reduction in storage space compared to REST while achieving a 16.5-17.1% higher acceptance length using the same storage space on HumanEval and MT Bench benchmarks.

- **Advantages**: CREST's key advantages include:
- **Reduced storage space:** Storing only a subset of n-grams significantly reduces the storage requirement for the datastore.
- **Improved performance:** By focusing on the most common and smallest n-grams, CREST achieves a higher acceptance length and faster drafting time complexity.

- **Comparison with REST**: CREST improves upon REST by addressing the issue of unbounded datastore growth. It allows for efficient compaction without sacrificing performance, making it a more practical and scalable approach for speculative decoding.



### ACL Ready: RAG Based Assistant for the ACL Checklis (https://arxiv.org/abs/2408.04675)
- **What's New**: ACLReady라는 도구는 ACL (Association for Computational Linguistics) 책임감 있는 NLP 연구 체크리스트를 작성하는 데 도움을 주는 Retrieval-Augmented Language Model (RAG) 기반 애플리케이션입니다. 이 도구는 저자들이 자신의 연구에 대한 깊은 생각을 할 수 있도록 돕고 체크리스트에 대한 답변을 생성하는 데 도움을 줄 수 있습니다.



### AutoFAIR : Automatic Data FAIRification via Machine Reading (https://arxiv.org/abs/2408.04673)
- **What's New**: 본 논문은 자동화된 FAIR 데이터 처리를 위한 새로운 아키텍처인 AutoFAIR를 제안합니다. AutoFAIR는 웹 페이지에서 메타데이터를 자동으로 추출하고 표준화하는 데 집중하여 데이터 찾기, 접근, 상호 운용성 및 재사용성을 향상시킵니다. AutoFAIR는 Web Reader와 FAIR Alignment라는 두 가지 주요 구성 요소를 통합하여 데이터 FAIR화를 자동화합니다.



### Prompt and Prejudic (https://arxiv.org/abs/2408.04671)
Comments:
          Accepted at ECCV workshop FAILED

- **What's New**: 이 논문은 큰 언어 모델(LLM, Large Language Models)과 비전 언어 모델(VLM, Vision Language Models)에서 윤리적 의사 결정 작업을 수행할 때 이름을 사용하는 것이 미치는 영향을 조사합니다. 이 연구는 윤리적으로 주석이 달린 텍스트 시나리오에 이름을 추가하여 모델 출력에서 인구 통계적 편견을 드러내는 접근 방식을 제안합니다. 이 연구에는 다양한 성별과 민족적 배경을 대표하는 300개 이상의 이름 목록이 포함되어 있으며, 수천 개의 도덕적 시나리오에서 테스트되었습니다. 연구팀은 사회 과학의 감사 방법론을 따르면서 인기 있는 LLM/VLM을 포함한 자세한 분석을 제안하여 이러한 시스템의 편견을 인식하고 완화하는 것이 중요하다는 점을 강조함으로써 책임감 있는 AI 분야에 기여합니다. 또한 연구팀은 실제 시나리오 벤치마크(PSB, Pratical Scenarios Benchmark)라는 새로운 벤치마크를 소개합니다. PSB는 일상적인 의사 결정 시나리오에서 성별 또는 인구 통계적 편견과 관련된 편견이 있는지 여부를 평가하고 LLM이 합리적인 결정을 내리기 위해 사용될 수 있는 실제 시나리오(예: 모기지 또는 보험 부여)를 평가하도록 설계되었습니다. 이 벤치마크를 통해 LLM과 VLM의 실제 적용에서 발생할 수 있는 위험과 편견을 강조하면서 다양한 인구 통계적 범주에 걸쳐 모델 동작을 포괄적으로 비교할 수 있습니다.



### Forecasting Live Chat Intent from Browsing History (https://arxiv.org/abs/2408.04668)
Comments:
          CIKM 2024

- **What's New**: This paper proposes a two-stage approach to predict user intent based on their browsing history on online shopping websites. The first stage classifies the browsing history into high-level intent categories using fine-tuned Transformers. The second stage then uses a large language model (LLM) to generate detailed user intents based on the browsing history and the predicted intent class.



### LLM Stability: A detailed analysis with some surprises (https://arxiv.org/abs/2408.04667)
- **What's New**: 본 연구는 대규모 언어 모델(LLM)의 안정성을 정량화하여 분석했습니다. 동일한 입력과 결정론적 하이퍼파라미터를 사용해도 결과가 다르게 나타나는 LLM의 특징을 심층적으로 살펴보았습니다. 특히, LLM이 출력 수준에서는 거의 결정론적이지 않지만, 구문 분석된 출력/답변 수준에서는 훨씬 더 결정론적이라는 사실을 밝혔습니다. 또한, LLM 정확도 변화는 정규 분포를 따르지 않으며, 안정성은 작업에 따라 다르게 나타난다는 점도 발견했습니다.



### LLMs are Not Just Next Token Predictors (https://arxiv.org/abs/2408.04666)
- **What's New**: 이 논문은 거대 언어 모델(LLM)을 단순히 다음 토큰 예측기로만 보는 시각에 대한 비판을 제기합니다. LLM은 다음 토큰 예측을 목표로 하는 확률적 경사 하강법(stochastic gradient descent)을 통해 언어를 학습하는 통계적 모델이지만, 이러한 예측 능력만으로 LLM을 설명하는 것은 부족하며, LLM의 작동 방식과 능력을 제대로 이해하기 위해서는 더 포괄적인 관점이 필요하다고 주장합니다. 이러한 주장을 뒷받침하기 위해 논문에서는 유전자의 관점에서 진화와 발달을 설명하려는 생물학 연구 프로그램에 대한 비유를 사용합니다.



### LLM-based MOFs Synthesis Condition Extraction using Few-Shot Demonstrations (https://arxiv.org/abs/2408.04665)
- **What's New**: 본 논문은 기존의 제로 샷 학습(zero-shot learning) 방식 대신 퓨샷 인 컨텍스트 학습(few-shot in-context learning) 방식을 활용하여 MOFs 합성 조건 추출을 개선한 연구를 제시합니다. 이를 통해 MOFs 합성 조건을 더 정확하고 효율적으로 추출할 수 있으며, 이는 새로운 MOFs 설계 및 발견에 중요한 역할을 할 것으로 기대됩니다.

- **Technical Details**: 본 연구는 다음과 같은 기술적 특징을 가지고 있습니다:  
 1. 인간-AI 협업 데이터 큐레이션 프로세스를 통해 고품질 기반 진실(ground-truth) 데모를 확보합니다. 
 2. BM25 알고리즘 기반의 검색 증강 생성(RAG) 기법을 활용하여 각 MOF 추출에 적합한 퓨샷 데모를 선택적으로 사용합니다. 
 3. LLM의 성능을 향상시키기 위해 다양한 지식(retrieval된 합성 조건, 숫자/텍스트 형식 제약, 퓨샷 데모)을 통합합니다. 
 4. 대규모 합성 추출을 위한 확장성을 고려하여 핵심 문단 검출, 데모 풀 크기 조절, LLM 기반 공동 참조 해결(co-reference resolution) 등의 기술을 적용합니다.

- **Performance Highlights**: 본 논문에서 제시된 퓨샷 방식은 제로 샷 LLM 방식보다 14.8% 더 높은 F1 성능을 달성했습니다. 또한 실제 MOFs 합성-구조 추론 실험에서 퓨샷 방식은 기존 방식보다 29.4% 향상된 성능을 보여주었습니다. 이는 퓨샷 인 컨텍스트 학습이 MOFs 합성 조건 추출에 효과적임을 증명합니다.



### Mitigating Hallucinations in Large Vision-Language Models (LVLMs) via Language-Contrastive Decoding (LCD) (https://arxiv.org/abs/2408.04664)
- **What's New**: 이 논문은 대규모 비전-언어 모델(LVLM)에서 발생하는 객체 환각(object hallucination) 문제를 완화하기 위해 새로운 디코딩 알고리즘인 언어 대조 디코딩(LCD: Language Contrastive Decoding)을 제안합니다. LCD는 LVLM 출력을 조정하여 객체 환각을 줄이는 데 효과적이며, 특히 LLM의 확률 분포 신뢰도 수준을 기반으로 하는 동적 가중치 메커니즘을 사용하여 LLM의 언어 편향을 줄이는 데 중점을 둡니다.



### Dopamin: Transformer-based Comment Classifiers through Domain Post-Training and Multi-level Layer Aggregation (https://arxiv.org/abs/2408.04663)
Comments:
          Accepted at The 3rd Intl. Workshop on NL-based Software Engineering, 2024

- **What's New**: Dopamin은 코드 주석 분류를 위한 새로운 Transformer 기반 도구로, 다양한 프로그래밍 언어에서 다양한 주석 유형에 대한 도메인 사후 훈련 (domain post-training)을 통해 언어 간 지식 전이 (knowledge transfer)를 실현하고 계층적 집계 (Hierarchical aggregation, HSUM) 전략을 활용하여 주석 표현을 향상시켜 보다 정확하고 관련성 있는 분류를 제공합니다.

- **Technical Details**: Dopamin은 CodeBERT 모델을 기반으로 하며, 다양한 프로그래밍 언어(Java, Python, Pharo)에서 수집된 다양한 주석 유형에 대한 사후 훈련을 통해 언어 간 지식 전이를 수행합니다. 또한 HSUM 전략을 적용하여 BERT 모델의 상위 계층(layers)을 결합하여 입력 텍스트의 의미 정보를 더 풍부하게 표현합니다. 이러한 기술은 Dopamin이 다양한 프로그래밍 언어의 미묘한 차이를 파악하여 더욱 정확한 주석 분류를 가능하게 합니다.

- **Performance Highlights**: NLBSE'24 Tool Competition 데이터셋에서 Dopamin은 STACC 기준 모델보다 평균 F1 점수가 3% 높은 0.74를 달성하며 우수한 성능을 보여주었습니다. 이는 Dopamin이 코드 주석 분류 작업에 있어서 효과적인 도구임을 입증합니다.



### Citekit: A Modular Toolkit for Large Language Model Citation Generation (https://arxiv.org/abs/2408.04662)
Comments:
          7 pages, 13 figures

- **What's New**: Citekit, an open-source and modular toolkit for LLM (Large Language Model) citation generation in Question Answering (QA) tasks, is introduced. It aims to standardize and facilitate the comparison of different citation generation methods, promoting reproducibility and the development of new approaches.

- **Technical Details**: Citekit consists of four main modules: Input, Generation Module, Enhancing Module, and Evaluator.  Input handles data loading and prompt generation, Generation Module houses LLMs for response generation with citations, Enhancing Module includes components for assessment, retrieval, planning, and editing, and Evaluator incorporates metrics for evaluating the pipeline's output. The toolkit is highly extensible, allowing users to customize modules and connections for specific tasks.

- **Performance Highlights**: Citekit enables a comprehensive evaluation of 11 state-of-the-art citation generation baselines on Llama3 and GPT-4o, highlighting the strengths and weaknesses of different modules in improving answer accuracy and citation quality. The analysis led to the development of a new method, self-RAG Snippet, which achieves a balance between answer accuracy and citation quality by combining effective components.



### MaterioMiner -- An ontology-based text mining dataset for extraction of process-structure-property entities (https://arxiv.org/abs/2408.04661)
- **What's New**: 본 논문은 **재료 역학(Materials mechanics)** 분야를 위한 새롭게 구축된 **온톨로지(Ontology)**와 **데이터셋(Dataset)**을 소개합니다. 이 데이터셋은 재료의 **조성(Composition)**, **처리 과정(Processing)**, **실험(Experimentation)**, **결함 분포(Defect distribution)**, **특성(Property)** 등의 세부 정보를 비정형 텍스트 데이터에서 추출하기 위한 목적으로 설계되었습니다. 특히, **재료 피로(Materials fatigue)**에 대한 심층적인 정보를 담고 있습니다. 또한 **명명된 엔터티 인식(Named-entity recognition, NER)** 모델의 훈련 및 벤치마킹을 위한 핵심적인 역할을 합니다.



### XMainframe: A Large Language Model for Mainframe Modernization (https://arxiv.org/abs/2408.04660)
- **What's New**: 이 연구는 레거시 메인프레임 시스템의 관리 및 현대화를 위한 혁신적인 도구로서 XMainframe라는 대규모 언어 모델(LLM)을 소개합니다. XMainframe은 메인프레임 시스템과 COBOL 코드베이스에 대한 지식을 갖추도록 특별히 설계되었으며, 고품질의 훈련 데이터 세트를 생성하는 광범위한 데이터 수집 파이프라인을 통해 훈련되었습니다. 또한, 연구에서는 메인프레임 지식 평가를 위한 종합적인 벤치마크인 MainframeBench도 제시합니다. MainframeBench는 다지선다형 질문, 질의응답 및 COBOL 코드 요약을 포함합니다.



### Winning Amazon KDD Cup'24 (https://arxiv.org/abs/2408.04658)
- **What's New**: This paper presents the winning solution for all five tracks of the Amazon KDD Cup 2024 Multi Task Online Shopping Challenge for LLMs, focused on building a useful online shopping assistant.

- **Technical Details**: The solution involves fine-tuning the Qwen2-72B-Instruct model on a custom training dataset created by processing multiple public datasets and utilizing Large Language Models (LLMs) for data augmentation.  Key techniques include:

* **Wise-ft** to mitigate distribution shifts during training.
* **Ensemble of LoRA adapters** within a single model to improve performance.
* **Logits Processors** to constrain model output to relevant tokens for specific tasks.
* **4-bit Quantization** and **vLLM** to optimize inference time and resource usage.

- **Performance Highlights**: The solution achieved first place in each individual track of the competition and secured the overall first place in the Amazon KDD Cup 2024. It showcases the effectiveness of fine-tuning and data augmentation techniques for building a robust and efficient LLM-based shopping assistant.



### Towards Semantic Markup of Mathematical Documents via User Interaction (https://arxiv.org/abs/2408.04656)
Comments:
          Submitted to the CICM 2024 conference, due to be published in Volume 14960 of Springer's Lecture Notes in Computer Science

- **What's New**: 이 논문은 LaTeX 수식을 의미적으로 마크업(semantic markup)하기 위한 새로운 접근 방식을 제시합니다. 특히, 기존 sTeX 매크로 정의에서 문법을 (반)자동으로 생성하고 이를 사용하여 수학 공식을 구문 분석하는 방법을 소개합니다. 이는 LaTeX 사용자들이 sTeX로 쉽게 전환할 수 있도록 돕는 것을 목표로 합니다.



### Strong and weak alignment of large language models with human values (https://arxiv.org/abs/2408.04655)
- **What's New**: 본 논문에서는 AI 시스템이 인간의 가치와 일치하도록 하는 것에 대해 두 가지 개념, **강한 정렬(strong alignment)**과 **약한 정렬(weak alignment)**을 구분합니다. **강한 정렬**은 AI가 인간의 의도를 이해하고 예측하고, 행동의 결과를 예측하여 인간의 가치를 위협하는 상황을 인식할 수 있어야 한다는 것을 의미합니다. 반면 **약한 정렬**은 AI가 인간의 가치를 완전히 이해하지 못하더라도, 주어진 상황에서 인간의 가치에 부합하는 행동을 하도록 훈련하는 것을 의미합니다.



### Batching BPE Tokenization Merges (https://arxiv.org/abs/2408.04653)
Comments:
          8 pages, 5 figures, 1 code block

- **What's New**: BatchBPE, an open-source pure Python implementation of batching Byte Pair Encoding (BPE) algorithm for tokenizer training, is presented. This technique enables faster and more memory-efficient training on a standard laptop. BatchBPE allows experimenting with different tokenization strategies by preprocessing stop words and ignoring infrequent text chunks.

- **Technical Details**: BatchBPE takes advantage of the power-law distribution of text chunks in a dataset by representing it as a dictionary mapping text chunks to their frequencies. This significantly reduces memory usage and processing time.  It introduces two features for preprocessing: stop word removal and frequency cutoff.  Stop word removal excludes common words from the token merging process, while frequency cutoff discards text chunks appearing below a certain frequency.  BatchBPE enables safe batch merging of token pairs by defining a set of non-interfering merges. This significantly speeds up vocabulary building. 

- **Performance Highlights**: The effectiveness of BatchBPE is demonstrated through experiments with the FineWeb-Edu dataset.  These experiments reveal the impact of stop word preprocessing and frequency cutoff on the resulting encoded text lengths. The results show that both techniques can have a significant impact on the final tokenization results.



### Leveraging Large Language Models with Chain-of-Thought and Prompt Engineering for Traffic Crash Severity Analysis and Inferenc (https://arxiv.org/abs/2408.04652)
Comments:
          20 pages, 12 figures, 3 tables

- **What's New**: 이 연구는 최첨단 LLM(Large Language Model)인 GPT-3.5-turbo, LLaMA3-8B, LLaMA3-70B를 사용하여 교통사고 심각도 추론을 분류 작업으로 수행합니다. 도메인 지식을 포함한 사전 구축된 템플릿을 사용하여 교통사고 표 데이터에서 텍스트 설명을 생성하고, 사고 원인을 분석하고 심각도를 추론하기 위해 CoT(Chain-of-Thought) 추론을 통합합니다. 또한 사고 심각도 추론을 위해 특별히 설계된 프롬프트 엔지니어링의 영향을 조사합니다.



### Knowledge AI: Fine-tuning NLP Models for Facilitating Scientific Knowledge Extraction and Understanding (https://arxiv.org/abs/2408.04651)
Comments:
          11 pages

- **What's New**: 본 프로젝트는 특정 도메인에서 과학적 지식을 이해하고 추출하는 데 있어 대규모 언어 모델(LLM)의 효과를 조사하여 지식 AI라는 딥 러닝 프레임워크를 만듭니다. 이 프레임워크의 일환으로, 사전 훈련된 모델을 사용하고 과학 도메인의 데이터셋에서 미세 조정합니다. 모델은 네 가지 핵심적인 자연어 처리(NLP) 작업, 즉 요약, 텍스트 생성, 질문 답변 및 명명된 엔터티 인식에 적응됩니다. 결과는 도메인별 미세 조정이 각 작업에서 모델 성능을 크게 향상시켜 과학적 맥락에 대한 적용성을 높인다는 것을 나타냅니다. 이러한 적응을 통해 비전문가는 타겟 과학 분야 내에서 정보를 효율적으로 쿼리하고 추출할 수 있으며, 미세 조정된 LLM이 과학 분야에서 지식 발견을 위한 도구로서의 잠재력을 보여줍니다.



### Building Trust in Mental Health Chatbots: Safety Metrics and LLM-Based Evaluation Tools (https://arxiv.org/abs/2408.04650)
- **What's New**: 본 연구는 접근성, 인간과 유사한 상호 작용, 맥락 인식 지원 등으로 인해 점점 더 인기를 얻고 있는 정신 건강 챗봇의 안전성과 신뢰성을 보장하기 위한 평가 프레임워크를 개발하고 검증했습니다. 연구진은 챗봇 응답에 대한 100개의 벤치마크 질문 및 이상적인 답변과 5개의 가이드라인 질문으로 구성된 평가 프레임워크를 만들었습니다. 정신 건강 전문가가 검증한 이 프레임워크는 GPT-3.5-turbo 기반 챗봇에서 테스트되었습니다. 자동 평가 방법에는 대규모 언어 모델(LLM) 기반 점수, 실시간 데이터를 사용하는 에이전틱 접근 방식, 챗봇 응답을 실제 기준과 비교하기 위한 임베딩 모델이 포함되었습니다.



### Chain of Stance: Stance Detection with Large Language Models (https://arxiv.org/abs/2408.04649)
- **What's New**: This paper introduces a novel prompting method called "Chain of Stance" (CoS) for stance detection using large language models (LLMs). Unlike existing methods that solely focus on fine-tuning LLMs, CoS decomposes the stance detection process into a sequence of intermediate reasoning steps, allowing LLMs to act as expert stance detectors.

- **Technical Details**: CoS leverages the encyclopedic knowledge of LLMs by prompting them with a series of questions related to the context, sentiment, and opinion expressed in the text. This chain of assertions ultimately leads to the final stance prediction. The method draws inspiration from the Chain-of-Thought (CoT) prompting paradigm used in mathematical reasoning tasks.

- **Performance Highlights**: Extensive experiments were conducted using four state-of-the-art LLMs (Mistral-7B, Qwen 1.5-7B, LLaMA 3-8B, LLaMA 2-7B) on the SemEval 2016 dataset. CoS achieved state-of-the-art results with an F1 score of 79.84 in the few-shot setting and 76.43 in the zero-shot setting, outperforming other baselines.



### PLUGH: A Benchmark for Spatial Understanding and Reasoning in Large Language Models (https://arxiv.org/abs/2408.04648)
Comments:
          Wordplay Workshop @ ACL 2024

- **What's New**: PLUGH (**P**layable **L**anguage **U**nderstanding **G**raph **H**andling) is introduced, a new benchmark for assessing LLMs' spatial understanding and reasoning skills. This benchmark comprises 5 tasks based on 125 input texts extracted from 48 text-based games, totaling 61 distinct spatial graphs.



### Distinguishing Chatbot from Human (https://arxiv.org/abs/2408.04647)
- **What's New**: 이 연구는 챗봇이 생성한 텍스트를 인간이 작성한 텍스트와 구별하는 새로운 방법을 제시합니다. 이를 위해, 75만 개 이상의 인간이 작성한 단락과 각각에 해당하는 챗봇이 생성한 단락으로 구성된 새로운 데이터셋을 개발했습니다. 이 데이터셋을 기반으로, 기계 학습 기술을 활용하여 텍스트의 출처 (인간 또는 챗봇)를 판별하는 모델을 개발했습니다.



### Efficacy of Large Language Models in Systematic Reviews (https://arxiv.org/abs/2408.04646)
- **What's New**: 본 연구는 ESG 요소와 재무 성과 간의 관계에 대한 체계적 검토를 통해 대규모 언어 모델(LLM)의 기존 문헌 해석 효과를 조사합니다. 주된 목표는 LLM이 ESG 관련 논문 집합에 대한 체계적 검토를 재현할 수 있는지 평가하는 것입니다.



### Evaluating the Impact of Advanced LLM Techniques on AI-Lecture Tutors for a Robotics Cours (https://arxiv.org/abs/2408.04645)
Comments:
          The article is an extended version of a paper presented at the International Workshop on AI in Education and Educational Research (AIEER) at ECAI-2024 (27th European Conference on Artificial Intelligence)

- **What's New**: 본 연구는 대학 강의를 위한 인공지능 튜터로서 대규모 언어 모델(LLM)의 성능을 평가했습니다. 특히, 프롬프트 엔지니어링, 검색 강화 생성(RAG), 미세 조정과 같은 다양한 고급 기술을 활용했습니다. BLEU-4, ROUGE, BERTScore와 같은 일반적인 유사성 지표를 사용하여 모델과 적용된 기술을 평가했으며, 도움 유용성과 신뢰성에 대한 소규모 인간 평가를 보완했습니다. 연구 결과, RAG와 프롬프트 엔지니어링의 결합은 모델 응답을 크게 향상시키고 더 나은 사실적 답변을 생성하는 것으로 나타났습니다. 교육 환경에서 RAG는 모델 입력을 추가 정보와 자료로 풍부하게 하는 것을 기반으로 하기 때문에 이상적인 기술로 보입니다. 일반적으로 대학 강좌에 이미 존재합니다. 반면에 미세 조정은 여전히 강력한 전문가 모델을 만들 수 있지만 과적합 위험이 있습니다. 본 연구는 또한 LLM의 성능을 어떻게 측정하고 현재 측정 방식이 정확성 또는 관련성을 얼마나 잘 나타내는지에 대해 질문합니다. 연구팀은 유사성 지표에서 높은 상관관계와 대부분의 지표에서 짧은 응답에 대한 편향을 발견했습니다. 전반적으로 본 연구는 교육 환경에서 LLM을 통합하는 잠재력과 과제를 모두 지적하며, 균형 잡힌 훈련 접근 방식과 고급 평가 프레임워크의 필요성을 시사합니다.



### Risks, Causes, and Mitigations of Widespread Deployments of Large Language Models (LLMs): A Survey (https://arxiv.org/abs/2408.04643)
Comments:
          Accepted to 2nd International Conference on Artificial Intelligence, Blockchain, and Internet of Things (AIBThings-2024), September 07-08, 2024, Michigan, USA

- **What's New**: 이 논문은 대규모 언어 모델(LLMs)의 광범위한 사용과 관련된 위험, 원인 및 완화 전략을 포괄적으로 조사한 설문 조사를 제공합니다. 이 설문 조사는 특정 LLM과 관련된 위험을 파악하고, 관련 위험을 정의하고, 그 원인을 식별하고, 잠재적인 해결책을 제안합니다. 또한 이 논문은 LLM과 관련된 더 넓은 범위의 과제를 다루고 그 원인과 완화 전략을 자세히 설명합니다.



### GPT-3 Powered Information Extraction for Building Robust Knowledge Bases (https://arxiv.org/abs/2408.04641)
- **What's New**: 이 논문은 GPT-3을 이용하여 지식베이스 구축을 위한 새로운 정보 추출 방법을 제안합니다. 제안된 방법은 비정형 텍스트에서 관련 엔티티와 관계를 추출하여 구조화된 정보를 추출하는 데 따르는 어려움을 해결하려고 시도합니다.

- **Technical Details**: 이 연구는 GPT-3의 컨텍스트 학습(in-context learning)을 사용하여 지식베이스 구축을 위한 정보 추출을 수행합니다. 이 방법은 구조화된 프롬프트(structured prompt)를 생성하고, k-최근접 이웃 모듈(k-nearest neighbor module)을 사용하며, NER 및 RE를 위한 로짓 편향(logit bias)과 컨텍스트 보정(contextual calibration)을 통합합니다.

- **Performance Highlights**: 실험 결과는 GPT-3가 텍스트에서 관련성 있고 정확한 정보를 효율적이고 정확하게 추출할 수 있음을 보여줍니다. 따라서 지식베이스 구축의 정확성과 생산성을 높일 수 있습니다. 또한, 제안된 방법이 기존의 최첨단 정보 추출 기술과 비교하여 경쟁력 있는 결과를 얻을 수 있음을 보여줍니다. 컨텍스트 학습을 사용하여 제한된 수의 샘플만으로도 경쟁력 있는 결과를 얻을 수 있어, 데이터 주석 및 엔지니어링 비용을 상당히 절감할 수 있습니다. 또한, 생물 의학 정보를 추출하여 실제 환경에서의 실용성을 입증했습니다.



### LLMs for Enhanced Agricultural Meteorological Recommendations (https://arxiv.org/abs/2408.04640)
Comments:
          10 pages

- **What's New**: 이 논문은 농업 기상 권장 사항을 개선하기 위해 대규모 언어 모델(LLM)과 프롬프트 엔지니어링을 결합한 새로운 접근 방식을 제시합니다. 이 방법은 ChatGPT, Claude2 및 GPT-4에서 구현된 다중 라운드 프롬프트 프레임워크를 사용하여 업데이트된 데이터와 피드백을 통해 권장 사항을 반복적으로 개선합니다.



### Abstractive summarization from Audio Transcription (https://arxiv.org/abs/2408.04639)
Comments:
          36 pages, Master's thesis, 14 figures

- **What's New**: 본 논문은 기존의 대규모 언어 모델을 특정 작업에 효과적으로 미세 조정하기 위한 기술인 LoRA와 양자화를 활용하여 E2E (end to end) 오디오 요약 모델을 제안합니다. 또한, 이러한 방법론을 오디오 요약 문제에 적용하여 효과성을 분석하고, 적용 가능성에 대한 결론을 도출합니다.



### Affective Computing in the Era of Large Language Models: A Survey from the NLP Perspectiv (https://arxiv.org/abs/2408.04638)
- **What's New**: 본 논문은 대규모 언어 모델(LLM) 시대의 감성 컴퓨팅(AC)에 대한 NLP 관점에서의 포괄적인 개요를 제공합니다. LLM이 AC 분야에 가져오는 새로운 기회와 과제를 분석하고 요약합니다. 기존 AC 연구는 주로 특정 작업을 위한 사전 훈련된 언어 모델(PLM)의 미세 조정에 의존했습니다. 그러나 LLM은 인맥스 컨텍스트 학습(in-context learning), 상식 추론(common sense reasoning), 고급 시퀀스 생성(advanced sequence generation)과 같은 기능을 통해 AC의 패러다임 전환을 이끌어냅니다. 이 논문은 LLM을 AC 작업에 적용하기 위한 지침 조정(Instruction Tuning)과 프롬프트 엔지니어링(Prompt Engineering)의 핵심 기술을 살펴봅니다.



### APE: Active Learning-based Tooling for Finding Informative Few-shot Examples for LLM-based Entity Matching (https://arxiv.org/abs/2408.04637)
Comments:
          3 pages, Proceedings of the Fifth Workshop on Data Science with Human-in-the-Loop (DaSH 2024)

- **What's New**: APE (Active Prompt Engineering)는 능동 학습 기법을 활용하여 프롬프트를 개선하는 도구입니다. APE는 대규모 언어 모델(LLM)의 성능을 높이기 위해 가장 유익한 몇 가지 예시를 찾는 데 도움이 되는 도구입니다. APE는 사용자와 LLM API와 상호 작용하여 반복적으로 가장 유익한 예시를 찾고, 사용자의 피드백을 통해 프롬프트를 개선합니다.



### Preserving Privacy in Large Language Models: A Survey on Current Threats and Solutions (https://arxiv.org/abs/2408.05212)
Comments:
          GitHub repository: this https URL

- **What's New**: 본 논문은 대규모 언어 모델(LLM)의 개인 정보 보호 위협을 포괄적으로 분석하고, 훈련 데이터, 모델 자체 및 추론 단계에서 개인 정보 보호를 강화하는 다양한 솔루션을 제시합니다. 특히, 훈련 데이터에서 개인 정보 식별 가능 정보(PII)가 유출될 위험을 강조하며, 기존의 개인 정보 보호 공격 연구를 검토하고, 훈련 데이터 익명화, 차등 프라이버시 적용 및 훈련 후 기계적 잊음(unlearning)과 같은 다양한 해결책을 제시합니다. 즉,  LLM의 훈련 및 배포 전반에 걸쳐 개인 정보 보호 메커니즘을 통합하는 방법을 제시합니다.



### VITA: Towards Open-Source Interactive Omni Multimodal LLM (https://arxiv.org/abs/2408.05211)
Comments:
          Project Page: this https URL

- **What's New**: VITA는 영상, 이미지, 텍스트, 오디오를 동시에 처리하고 분석할 수 있는 최초의 오픈소스 멀티모달 대규모 언어 모델(MLLM)입니다. VITA는 멀티모달 대화형 경험을 제공하는 기능을 갖추고 있으며, 영어와 중국어를 모두 지원합니다.



### Evaluating the capability of large language models to personalize science texts for diverse middle-school-age learners (https://arxiv.org/abs/2408.05204)
Comments:
          20 pages, 3 figures

- **What's New**: 본 논문은 GPT-4를 활용하여 중학생을 위한 과학 교육 텍스트를 개인화하는 효과를 평가한 최초의 무작위 통제 실험(n=23) 중 하나를 제시합니다. 연구는 GPT-4를 사용하여 학생의 학습 선호도를 프로파일링하고, 개인화된 텍스트를 생성하는 방법을 시험했습니다. 실험군 학생들은 GPT-4를 사용하여 자신의 학습 선호도에 맞게 재작성된 텍스트를 제공받았고, 대조군 학생들은 선호도와 반대되는 방향으로 재작성된 텍스트를 제공받았습니다. 결과적으로 Mann-Whitney U 검정 결과 학생들은 자신의 프로파일에 맞는 재작성된 텍스트를 유의미하게 더 선호하는 것으로 나타났습니다 (p=.059). 이 연구는 GPT-4가 다양한 학습자의 선호도에 맞게 교육 콘텐츠를 효과적으로 해석하고 조정할 수 있음을 시사하며, PL 기술의 중요한 진전을 의미합니다. 이 연구의 한계와 교육에서 인공 지능을 사용하는 데 대한 윤리적 고려 사항도 논의됩니다.



### Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2 (https://arxiv.org/abs/2408.05147)
Comments:
          12 main text pages, and 14 pages of acknowledgements, references and appendices

- **What's New**: Gemma Scope, a comprehensive suite of sparse autoencoders (SAEs) trained on all layers and sub-layers of Gemma 2 2B, 9B, and 27B models, is released. This open-source resource aims to accelerate research in safety and interpretability of language models. It contains over 400 SAEs with over 30 million learned features, trained using a significant amount of computational resources.



### MIDI-to-Tab: Guitar Tablature Inference via Masked Language Modeling (https://arxiv.org/abs/2408.05024)
Comments:
          Reviewed pre-print accepted for publication at ISMIR 2024

- **What's New**: 본 논문은 기존의 제약 기반 동적 프로그래밍 방식에서 벗어나 딥 러닝 기반의 솔루션을 제시하여 기타 탭 음악을 자동으로 생성하는 문제를 해결합니다. 특히, 마스크 언어 모델링 패러다임(masked language modeling paradigm)을 적용한 인코더-디코더 트랜스포머 모델을 사용하여 노트를 스트링에 할당하는 방식을 채택합니다.



### mPLUG-Owl3: Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models (https://arxiv.org/abs/2408.04840)
- **What's New**: mPLUG-Owl3는 긴 이미지 시퀀스를 이해하는 데 뛰어난 성능을 보여주는 새로운 다중 모달 대규모 언어 모델(MLLM)입니다. 기존 모델과 달리, mPLUG-Owl3는 검색된 이미지-텍스트 지식, 혼합된 이미지-텍스트, 그리고 긴 비디오를 포함하는 시나리오에서 긴 이미지 시퀀스 이해를 향상시킵니다. 특히, mPLUG-Owl3는 긴 다중 이미지 시나리오를 처리하기 위해 비전과 언어를 공통의 언어 기반 의미 공간에 효율적으로 통합하는 새로운 하이퍼 어텐션 블록을 제안합니다.



### Survey: Transformer-based Models in Data Modality Conversion (https://arxiv.org/abs/2408.04723)
Comments:
          Submitted to ACM Computing Surveys (CSUR)

- **What's New**: This paper provides a comprehensive review of Transformer-based (TB) models applied to modality conversion, focusing on text, vision, and speech. It systematically summarizes various TB models for converting data between these three modalities, including text-to-speech, speech-to-text, image-to-text, and more, highlighting the versatility and scalability of transformers in advancing AI-driven content generation and understanding.



New uploads on arXiv(cs.IR)

### A GNN Model with Adaptive Weights for Session-Based Recommendation Systems (https://arxiv.org/abs/2408.05051)
Comments:
          7 pages, 7 tables, 2 figures, and 3 equations

- **What's New**: This paper proposes a novel approach to enhance session-based recommendations (SBRs) by introducing an adaptive weighting mechanism to the SR-GNN model. This mechanism assigns varying importance to items within a session based on side information, leading to more accurate predictions and addressing the cold start problem in SBRs.



### Early Exit Strategies for Approximate k-NN Search in Dense Retrieva (https://arxiv.org/abs/2408.04981)
Comments:
          6 pages, published at CIKM 2024

- **What's New**: 이 논문은 효율성을 높이면서 효과는 크게 저하시키지 않는  A-kNN (Approximate k-Nearest Neighbors) 검색을 위한 새로운 방식을 제시합니다.  특히, '인내심(patience)' 개념을 도입하여 클러스터 탐색을 조기에 종료하는 기술을 소개합니다. 또한, 초기 몇 개의 클러스터에서 최근접 이웃을 찾을 수 있는 쿼리를 먼저 식별하고, 인내심 기반 전략 또는 다른 최신 전략을 사용하여 탐색 범위를 조정하는 캐스케이드 방식을 제안합니다.  



### Relevance Filtering for Embedding-based Retrieva (https://arxiv.org/abs/2408.04887)
Comments:
          8 pages, 3 figures, CIKM 2024

- **What's New**: 이 논문에서는 임베딩 기반 검색에서 관련성 필터링을 위한 새로운 'Cosine Adapter'라는 컴포넌트를 소개합니다. 이는 쿼리에 따른 매핑 함수를 통해 원시 코사인 유사성 점수를 해석 가능한 점수로 변환하고, 이렇게 매핑된 점수에 대한 글로벌 임계값을 적용하여 관련성이 낮은 결과를 제거하는 방식입니다.

- **Technical Details**: Cosine Adapter는 쿼리 의존적인 매핑 함수를 사용하여 원시 코사인 유사성 점수를 해석 가능한 관련성 점수로 변환합니다. 이는 쿼리 난이도에 대한 맥락적 인식을 가능하게 합니다. 이러한 관련성 매핑 기법은 점수에 대한 명확한 확률적 해석을 제공하여 투명성과 해석 가능성을 향상시키고, 효율성이 높다는 장점이 있습니다.

- **Performance Highlights**: MS MARCO 및 Walmart 제품 검색 데이터 세트에 대한 광범위한 오프라인 벤치마킹과 실시간 온라인 테스트를 통해 다양한 쿼리 및 응용 분야에서 향상된 정밀도가 입증되었습니다. 이 접근 방식은 검색 시스템의 정밀도를 크게 향상시키는 동시에 약간의 재현율 손실만 발생한다는 점에서 주목할 만합니다.



### Enhancing Relevance of Embedding-based Retrieval at Walmar (https://arxiv.org/abs/2408.04884)
Comments:
          8 pages, 3 figures, CIKM 2024

- **What's New**: 본 논문은 Walmart.com의 검색 시스템에 사용되는 임베딩 기반 검색(EBR) 모델의 정확성을 향상시키기 위한 여러 가지 새로운 접근 방식을 제시합니다. EBR은 사용자 검색 쿼리와 상품 사이의 어휘 차이를 해소하는 데 효과적인 검색 방법입니다. 그러나 Walmart.com의 초기 EBR 시스템은 정확도 향상과 장바구니 추가율 증가를 가져왔지만, 여전히 정확도 저하 사례가 관찰되었습니다. 이러한 저하의 주요 원인으로는 훈련 데이터의 오탐, 오류, 오타 처리 불능 등이 있습니다. 본 논문은 이러한 문제를 해결하기 위해 훈련 데이터의 노이즈를 제거하고 EBR 모델을 개선하는 새로운 방식을 제시합니다.



### 3DLNews: A Three-decade Dataset of US Local News Articles (https://arxiv.org/abs/2408.04716)
Comments:
          This is a technical report for a resource paper accepted at CIKM 2024

- **What's New**: 3DLNews는 1996년부터 2024년까지 미국의 지역 뉴스 기사를 담은 새로운 데이터셋입니다. 이 데이터셋은 50개 주의 14,000개 이상의 지역 신문, TV 및 라디오 방송국에서 거의 100만 개의 URL(HTML 텍스트 포함)을 포함하고 있으며 미국 지역 뉴스 환경의 광범위한 스냅샷을 제공합니다. 데이터셋은 Google 및 Twitter 검색 결과를 스크래핑하여 수집되었습니다.

- **Technical Details**: 3DLNews 데이터셋은 Google 및 Twitter 검색 결과를 스크래핑하여 수집되었습니다. 3DLNews 데이터셋은 Google 및 Twitter 검색 결과를 스크래핑하여 수집되었습니다. 이 데이터셋은 Google 및 Twitter 검색 결과를 스크래핑하여 수집되었습니다. 데이터셋은 Google 및 Twitter 검색 결과를 스크래핑하여 수집되었습니다.

- **Performance Highlights**: 3DLNews는 50개 주의 14,000개 이상의 지역 신문, TV 및 라디오 방송국에서 거의 100만 개의 URL(HTML 텍스트 포함)을 포함하고 있으며 미국 지역 뉴스 환경의 광범위한 스냅샷을 제공합니다. 데이터셋은 Google 및 Twitter 검색 결과를 스크래핑하여 수집되었습니다. 3DLNews 데이터셋은 Google 및 Twitter 검색 결과를 스크래핑하여 수집되었습니다. 3DLNews 데이터셋은 Google 및 Twitter 검색 결과를 스크래핑하여 수집되었습니다.

- **Applications**: 3DLNews는 지역 뉴스의 국가화, 미디어 편향 및 지역 뉴스 사막 분석, 커뮤니티 이해를 포함한 4가지 유스 케이스를 설명하여 유용성을 입증했습니다.



### A Hybrid RAG System with Comprehensive Enhancement on Complex Reasoning (https://arxiv.org/abs/2408.05141)
Comments:
          Technical report for 3rd prize in Task 1 of Meta CRAG KDD Cup 2024

- **What's New**: 본 논문에서는 Retrieval-Augmented Generation (RAG) 시스템을 소개합니다. 이 시스템은 외부 지식베이스를 통합하여 대규모 언어 모델(LLM)의 정확성을 향상시키고 환각(hallucination)을 줄이는 데 도움이 됩니다. 시스템은 웹 페이지의 텍스트 조각과 테이블을 개선하고, 환각을 줄이기 위해 속성 예측기를 추가하며, LLM 지식 추출기와 지식 그래프 추출기를 수행하고, 마지막으로 모든 참조를 사용하여 추론 전략을 구축합니다.

- **Technical Details**: 제안된 RAG 시스템은 다음과 같은 4가지 단계로 구성됩니다:
1) **사전 검색** 단계: 외부 지식 소스를 색인화하고 쿼리를 조작하며 데이터를 수정하는 단계입니다. 특히, 향후 검색 실행의 효율성을 높이기 위해 외부 지식 소스를 컴파일하고 색인화하고, 외부 데이터 분포와 일치하도록 쿼리를 개선하고, 추가 추론을 위한 관련 지식을 포괄하는 일관된 표현을 생성하기 위해 데이터 소스를 수정합니다.
2) **검색** 단계: 쿼리와 외부 지식베이스 사이의 유사성을 측정하여 관련 텍스트 문서를 검색하는 단계입니다. 코사인 유사도와 같은 지표를 사용하여 검색을 수행합니다.
3) **사후 검색** 단계: 검색된 문서에서 추가 정보를 추출하고 처리하여 LLM이 추론 및 생성 작업에 더 잘 활용할 수 있도록 하는 단계입니다.
4) **생성** 단계: LLM이 쿼리와 검색된 문서를 사용하여 답변을 생성하는 단계입니다.

- **Performance Highlights**: 본 논문에서 제안된 RAG 시스템은 CRAG 데이터 세트에서 평가되었습니다. 시스템은 Task 1에서 3위를 차지했으며 Task 2에서는 7,777개의 질문 유형 중 5,555개 유형에서 1위를 차지했습니다. 지역 평가와 온라인 평가 모두에서 시스템이 복잡한 추론 능력을 크게 향상시킨다는 것을 보여줍니다. 지역 평가에서 시스템은 기준 모델에 비해 정확도를 크게 향상시키고 오류율을 감소시켜 점수가 상당히 증가했습니다. 한편, 시스템은 온라인 평가에서 뛰어난 결과를 달성하여 제안된 시스템의 성능과 일반화 능력을 보여줍니다.



### MIDI-to-Tab: Guitar Tablature Inference via Masked Language Modeling (https://arxiv.org/abs/2408.05024)
Comments:
          Reviewed pre-print accepted for publication at ISMIR 2024

- **What's New**: 본 논문은 기존의 제약 기반 동적 프로그래밍 방식에서 벗어나 딥 러닝 기반의 솔루션을 제시하여 기타 탭 음악을 자동으로 생성하는 문제를 해결합니다. 특히, 마스크 언어 모델링 패러다임(masked language modeling paradigm)을 적용한 인코더-디코더 트랜스포머 모델을 사용하여 노트를 스트링에 할당하는 방식을 채택합니다.



### Hybrid Student-Teacher Large Language Model Refinement for Cancer Toxicity Symptom Extraction (https://arxiv.org/abs/2408.04775)
- **What's New**: 본 연구는 컴팩트한 대규모 언어 모델(LLM)을 사용하여 암 독성 증상 추출을 위한 새로운 반복적 개선 접근 방식을 제시합니다. 학생-교사 아키텍처를 활용하여, 학생 모델 (Zephyr-7b-beta 및 Phi3-mini-128)과 교사 모델(GPT-4o)을 사용하여 프롬프트 개선, 검색 기반 생성 (RAG), 미세 조정 전략 중에서 동적으로 선택합니다. 294개의 임상 노트를 대상으로 한 실험 결과, 이 접근 방식의 효과를 확인했습니다.



### ACL Ready: RAG Based Assistant for the ACL Checklis (https://arxiv.org/abs/2408.04675)
- **What's New**: ACLReady라는 도구는 ACL (Association for Computational Linguistics) 책임감 있는 NLP 연구 체크리스트를 작성하는 데 도움을 주는 Retrieval-Augmented Language Model (RAG) 기반 애플리케이션입니다. 이 도구는 저자들이 자신의 연구에 대한 깊은 생각을 할 수 있도록 돕고 체크리스트에 대한 답변을 생성하는 데 도움을 줄 수 있습니다.



### Towards Semantic Markup of Mathematical Documents via User Interaction (https://arxiv.org/abs/2408.04656)
Comments:
          Submitted to the CICM 2024 conference, due to be published in Volume 14960 of Springer's Lecture Notes in Computer Science

- **What's New**: 이 논문은 LaTeX 수식을 의미적으로 마크업(semantic markup)하기 위한 새로운 접근 방식을 제시합니다. 특히, 기존 sTeX 매크로 정의에서 문법을 (반)자동으로 생성하고 이를 사용하여 수학 공식을 구문 분석하는 방법을 소개합니다. 이는 LaTeX 사용자들이 sTeX로 쉽게 전환할 수 있도록 돕는 것을 목표로 합니다.



### PLUGH: A Benchmark for Spatial Understanding and Reasoning in Large Language Models (https://arxiv.org/abs/2408.04648)
Comments:
          Wordplay Workshop @ ACL 2024

- **What's New**: PLUGH (**P**layable **L**anguage **U**nderstanding **G**raph **H**andling) is introduced, a new benchmark for assessing LLMs' spatial understanding and reasoning skills. This benchmark comprises 5 tasks based on 125 input texts extracted from 48 text-based games, totaling 61 distinct spatial graphs.



### Abstractive summarization from Audio Transcription (https://arxiv.org/abs/2408.04639)
Comments:
          36 pages, Master's thesis, 14 figures

- **What's New**: 본 논문은 기존의 대규모 언어 모델을 특정 작업에 효과적으로 미세 조정하기 위한 기술인 LoRA와 양자화를 활용하여 E2E (end to end) 오디오 요약 모델을 제안합니다. 또한, 이러한 방법론을 오디오 요약 문제에 적용하여 효과성을 분석하고, 적용 가능성에 대한 결론을 도출합니다.



New uploads on arXiv(cs.CV)

### VITA: Towards Open-Source Interactive Omni Multimodal LLM (https://arxiv.org/abs/2408.05211)
Comments:
          Project Page: this https URL

- **What's New**: VITA는 영상, 이미지, 텍스트, 오디오를 동시에 처리하고 분석할 수 있는 최초의 오픈소스 멀티모달 대규모 언어 모델(MLLM)입니다. VITA는 멀티모달 대화형 경험을 제공하는 기능을 갖추고 있으며, 영어와 중국어를 모두 지원합니다.



### Multi-Garment Customized Model Generation (https://arxiv.org/abs/2408.05206)
- **What's New**: 이 논문에서는 여러 옷 조합 이미지 합성이라는 탐험되지 않은 과제를 해결하기 위해 잠재적 확산 모델(LDM) 기반의 통합 프레임워크인 다중 의상 맞춤 모델 생성을 소개합니다. 이 방법은 다양한 텍스트 프롬프트에 따라 여러 목표 의상을 입은 맞춤 모델을 생성하는 데 중점을 둡니다. 주요 과제는 각 의상의 복잡한 질감을 유지하면서 옷을 입은 모델의 자연스러운 외관을 유지하는 동시에 서로 다른 의상의 정보가 서로 간섭하지 않도록 하는 것입니다.



### Kalman-Inspired Feature Propagation for Video Face Super-Resolution (https://arxiv.org/abs/2408.05205)
Comments:
          Accepted by ECCV 2024. Project page: this https URL

- **What's New**: 본 논문에서는 영상 얼굴 초해상도(VFSR)를 위한 새로운 프레임워크인 칼만-영감 기능 전파(KEEP)를 소개합니다. KEEP은 칼만 필터링 원리를 기반으로 이전 프레임에서 얻은 정보를 사용하여 현재 프레임의 복원 과정을 안내하고 조절하여 시간 경과에 걸쳐 안정적인 얼굴 사전을 유지합니다.



### Cross-Domain Learning for Video Anomaly Detection with Limited Supervision (https://arxiv.org/abs/2408.05191)
- **What's New**: 이 연구는 비디오 이상 탐지(VAD, Video Anomaly Detection)에서 약한 지도 학습(weakly-supervised learning)을 사용하여 도메인 간 일반화(cross-domain generalization)를 향상시키는 새로운 프레임워크를 제시합니다. 특히, 외부 데이터(external data)의 예측 편향(prediction bias)을 추정하고 예측 불확실성(prediction uncertainty)을 이용하여 적응적으로 최소화함으로써 외부 데이터를 훈련에 통합하는 방법을 제안합니다.



### EasyInv: Toward Fast and Better DDIM Inversion (https://arxiv.org/abs/2408.05159)
Comments:
          8 pages

- **What's New**: 이 논문은 EasyInv를 소개하는데, 이는 기존의 반복적인 최적화 방법의 비효율성과 성능 제한을 해결하여 DDIM Inversion(역변환) 분야를 크게 발전시키는 간단하면서도 새로운 접근 방식입니다. EasyInv의 핵심은 Inversion(역변환) 노이즈를 근사화하기 위한 개선된 전략으로, 이는 Inversion(역변환) 프로세스의 정확성과 신뢰성을 높이는 데 중요합니다. 원래 이미지에 대한 풍부한 정보를 담고 있는 초기 잠재 상태를 우선시함으로써 EasyInv는 노이즈 항목의 반복적인 개선을 피합니다. 대신 이전 시간 단계의 잠재 상태를 현재 상태와 체계적으로 집계하여 초기 잠재 상태의 영향을 증가시키고 노이즈의 영향을 완화합니다. EasyInv는 모델의 정밀도가 제한적이거나 컴퓨팅 리소스가 부족한 조건에서 특히 기존 DDIM Inversion(역변환) 접근 방식과 동등하거나 능가하는 결과를 제공할 수 있음을 보여줍니다. 동시에 EasyInv는 기존의 반복적인 최적화 기술에 비해 추론 효율성을 약 3배 향상시킵니다.



### PriPHiT: Privacy-Preserving Hierarchical Training of Deep Neural Networks (https://arxiv.org/abs/2408.05092)
Comments:
          16 pages, 16 figures, 6 tables

- **What's New**: This paper introduces Privacy-Preserving Hierarchical Training (PriPHiT), a method for training deep neural networks on edge-cloud systems while preserving privacy of sensitive data. It utilizes adversarial early exits at the edge to suppress sensitive information before transmission to the cloud, ensuring only task-relevant information is shared. The method also incorporates noise addition for differential privacy guarantee.



### Loc4Plan: Locating Before Planning for Outdoor Vision and Language Navigation (https://arxiv.org/abs/2408.05090)
Comments:
          arXiv admin note: text overlap with arXiv:2203.13838 by other authors

- **What's New**: This paper introduces **Loc4Plan**, a novel framework for **outdoor Vision and Language Navigation (VLN)**. Loc4Plan emphasizes the importance of **spatial localization** before action planning, drawing inspiration from human navigation behavior.



### UNIC: Universal Classification Models via Multi-teacher Distillation (https://arxiv.org/abs/2408.05088)
Comments:
          To be presented at ECCV 2024

- **What's New**: 본 논문은 다양한 분류 작업에서 더 강력한 일반화를 목표로 여러 보완적인 사전 훈련된 모델에서 정보를 취할 수 있는 고유한 인코더를 학습하는 방법을 제시합니다. 이를 위해 다중 교사 증류(multi-teacher distillation) 방법을 제안합니다. 먼저 여러 강력한 교사 모델을 활용한 기본 증류 방법을 자세히 분석하고, 그 결과를 바탕으로 기본 증류 설정을 개선하는 방법을 단계적으로 제시합니다.



### PreciseControl: Enhancing Text-To-Image Diffusion Models with Fine-Grained Attribute Contro (https://arxiv.org/abs/2408.05083)
Comments:
          ECCV 2024, Project page: this https URL

- **What's New**: 이 논문은 StyleGAN 모델의 disentangled 𝒲+limit-from𝒲\mathcal{W+}caligraphic_W + 공간을 사용하여 T2I 모델을 조건화하는 새로운 방법을 제안하여, 얼굴 특징을 정밀하게 조작하면서 T2I 모델의 기존 텍스트 기반 제어를 유지할 수 있도록 합니다. 이를 통해 얼굴 이미지의 정확한 역변환 및 속성 보존이 가능해졌고, 미소, 나이 등 세밀한 얼굴 속성을 부드럽게 편집할 수 있습니다.



### DeepInteraction++: Multi-Modality Interaction for Autonomous Driving (https://arxiv.org/abs/2408.05075)
Comments:
          Journal extension of NeurIPS 2022. arXiv admin note: text overlap with arXiv:2208.11112

- **What's New**: This paper proposes a novel **modality interaction strategy** called **DeepInteraction++** for autonomous driving tasks, aiming to overcome the limitations of traditional **multi-modal fusion** approaches. Instead of merging modality-specific representations into a single representation, DeepInteraction++ maintains and leverages individual modality strengths throughout the perception pipeline.

- **Technical Details**: DeepInteraction++ utilizes a **dual-stream Transformer** architecture for encoding, enabling **inter-modal representational interaction** and **intra-modal representational learning** simultaneously. The encoder employs specialized attention mechanisms like **deformable attention** for flexible receptive fields and **LiDAR-guided cross-plane polar ray attention** to propagate dense semantic information from the visual to the LiDAR representation. The decoder, equipped with a **multi-modal predictive interaction** module, refines predictions by iteratively aggregating information from individual modality representations. The framework also incorporates **grouped sparse attention** for enhanced efficiency.

- **Performance Highlights**: DeepInteraction++ demonstrates superior performance on both **3D object detection** and **end-to-end autonomous driving tasks**, surpassing prior art methods on the nuScenes dataset. It effectively combines object-centric information with dense environment representation, offering a versatile solution for various autonomous driving applications. The paper also presents comprehensive ablation studies to illustrate the effectiveness of different design choices and the scalability of the framework.



### Livestock Fish Larvae Counting using DETR and YOLO based Deep Networks (https://arxiv.org/abs/2408.05032)
- **What's New**: 이 논문은 양식업에서 물고기 유충 계산을 자동화하기 위한 혁신적인 접근 방식을 제시합니다. 이 연구는 데이터 수집 요구 사항이 적은 새롭게 주석 처리된 이미지 데이터 세트를 사용하여 컨볼루션 신경망(CNN)과 트랜스포머(transformer)를 포함한 다양한 규모의 네 가지 신경망 아키텍처를 평가합니다.



### Collaborative Static-Dynamic Teaching: A Semi-Supervised Framework for Stripe-Like Space Target Detection (https://arxiv.org/abs/2408.05029)
- **What's New**: 이 논문은 우주 상황 인식을 위한 스트라이프 모양 우주 표적 감지(SSTD)를 위한 혁신적인 협업 정적-동적 교사(CSDT) 반지도 학습(SSL) 프레임워크를 소개합니다. CSDT는 정적 및 동적 교사 모델과 학생 모델로 구성되어 있으며, 스트라이프 모양 특징을 학습하는 효과적인 방법입니다. 이 프레임워크는 적응형 가짜 라벨링(APL) 전략을 사용하여 초기 정적 교습에서 적응형 협업 교습으로 전환하여 학생 모델의 학습을 안내합니다. 지수 이동 평균(EMA) 메커니즘은 학생 모델을 통해 동적 교사 모델에 새로운 스트라이프 모양 지식을 제공하여 가짜 라벨의 품질을 지속적으로 향상시키는 긍정적 피드백 루프를 만듭니다. 또한 이 논문은 CSDT SSL 학습 프레임워크 내에서 다양한 스트라이프 모양 특징을 추출하도록 설계된 다중 스케일 이중 경로 합성곱(MDPC) 블록과 특징 맵 가중치 주의(FMWA) 블록을 갖춘 새로운 SSTD 네트워크인 MSSA-Net을 소개합니다.



### RadarPillars: Efficient Object Detection from 4D Radar Point Clouds (https://arxiv.org/abs/2408.05020)
Comments:
          This paper has been accepted at IEEE Intelligent Transportation Systems Conference (ITSC), 2024

- **What's New**: This paper introduces RadarPillars, a novel 3D object detection network specifically designed for 4D radar data. It effectively addresses the limitations of existing methods that were originally developed for LiDAR data, leading to significant performance improvements in 4D radar object detection. The key contributions of RadarPillars include: 

1. **Enhanced Velocity Information Utilization:** The paper proposes a method to decompose radial velocity data, providing richer features to the network and significantly improving detection performance.
2. **Adapting to Radar Sparsity:** RadarPillars leverages the pillar representation for efficient real-time processing, taking advantage of the sparsity inherent in 4D radar data.
3. **PillarAttention for Feature Extraction:**  A novel self-attention layer called PillarAttention is introduced. This layer treats each pillar as a token, enabling efficient and effective feature extraction.
4. **Scaling for Sparse Radar Data:** The paper demonstrates how network scaling can be optimized to accommodate the sparsity of radar data, leading to enhanced performance and reduced parameter count.



### Instruction Tuning-free Visual Token Complement for Multimodal LLMs (https://arxiv.org/abs/2408.05019)
Comments:
          Accepted by ECCV2024 (20pages)

- **What's New**: 본 논문은 이미지에서 시각적 정보를 더 잘 포착하고 이해할 수 있도록 멀티모달 LLM (MLLM)을 개선하는 새로운 프레임워크인 Visual Token Complement (VTC)를 제안합니다. VTC는 이미지-텍스트 쌍 없이도, 즉 추가 학습 없이 시각적 토큰을 보완하여 MLLM의 성능을 향상시키는 데 초점을 맞춥니다.

- **Technical Details**: VTC는 텍스트-이미지 생성을 활용하여 텍스트와 관련 없는 시각적 특징을 식별하고, 이를 통해 보완적인 시각적 토큰을 생성하여 원본 이미지의 시각적 정보를 풍부하게 합니다. 또한, 추가적인 학습 없이 시각적 선택기를 반복적으로 사용하여 시각 정보를 추출하는 반복적인 전략을 디자인합니다.

- **Performance Highlights**: 실험 결과, VTC는 여러 멀티모달 벤치마크에서 기존 방법보다 뛰어난 성능을 보여주었습니다. 특히, 시각적 대화(visual dialogue) 작업에서 InstructBLIP보다 45% 개선된 성능을 보였으며, 보완적인 시각적 토큰의 시각화를 통해 보완된 의미를 해석할 수 있는 도구를 제공합니다.



### DreamCouple: Exploring High Quality Text-to-3D Generation Via Rectified Flow (https://arxiv.org/abs/2408.05008)
Comments:
          Tech Report

- **What's New**: 이 논문은 3D 모델 생성을 위한 Score Distillation Sampling(SDS) 방법을 rectified flow 기반 확산 모델에 적용한 최초의 연구입니다. 기존 SDS 방법은 DDPM과 DDIM 모델에서 주로 사용되었지만, rectified flow 모델의 특징을 활용하여 SDS를 적용하면 오버 스무딩 문제가 발생할 수 있습니다. 이 논문은 rectified flow 기반 SDS를 위한 새로운 프레임워크인 RcSDS와 오버 스무딩 문제를 해결하기 위한 DreamCouple 방법을 제안합니다.



### DAFT-GAN: Dual Affine Transformation Generative Adversarial Network for Text-Guided Image Inpainting (https://arxiv.org/abs/2408.04962)
Comments:
          ACM MM'2024. 9 pages, 3 tables, 9 figures

- **What's New**: 본 논문은 텍스트 기반 이미지 인페인팅을 위한 새로운 모델인 DAFT-GAN(Dual Affine Transformation Generative Adversarial Network)을 제안합니다. DAFT-GAN은 각 디코딩 블록에서 텍스트와 이미지 피처를 점진적으로 결합하기 위해 두 개의 어파인 변환 네트워크를 통합합니다. 또한, 마스크 이미지의 손상된 영역과 손상되지 않은 영역을 별도로 인코딩하여 미세한 이미지 생성을 위한 손상되지 않은 피처의 정보 누출을 최소화합니다.



### In Defense of Lazy Visual Grounding for Open-Vocabulary Semantic Segmentation (https://arxiv.org/abs/2408.04961)
Comments:
          Accepted to ECCV 2024

- **What's New**: LaVG (Lazy Visual Grounding)는 기존의 오픈 보캐뷸러리 세맨틱 세그멘테이션(OVSeg) 방법과 달리, 이미지의 시각적 단위를 먼저 식별한 다음 해당 단위에 텍스트를 할당하는 새로운 두 단계 접근 방식을 제시합니다. 기존 방법들은 이미지의 픽셀을 텍스트로 분류하는 방식으로, 객체 수준의 이해 없이 수행되는 반면, LaVG는 객체 수준의 이해를 기반으로 합니다. 특히, LaVG는 텍스트 정보 없이도 이미지의 시각적 정보만으로 객체를 구분할 수 있다는 점을 강조하며, Normalized Cut 알고리즘을 이용하여 이미지를 여러 개의 객체 마스크로 분할합니다. 이후 각 마스크에 해당하는 텍스트를 할당하여 최종적인 세그멘테이션 결과를 얻습니다.

- **Technical Details**: LaVG는 먼저 DINO (self-supervised vision transformer)의 출력을 이용하여 Normalized Cut 알고리즘을 반복적으로 적용하여 이미지를 객체 마스크로 분할합니다. 이 과정을 Panoptic Cut이라고 합니다. Panoptic Cut이 완료되면, 각 마스크에 해당하는 텍스트를 할당합니다. 텍스트 할당은 객체 마스크를 나타내는 벡터와 텍스트를 나타내는 벡터 간의 유사도를 계산하여 수행됩니다. LaVG는 추가적인 학습 없이도 우수한 성능을 보이며, 픽셀 기반의 방법에 비해 더 정확한 객체 경계를 생성합니다. LaVG의 주요 특징은 다음과 같습니다: - 텍스트 없이 시각적 정보만으로 객체를 식별하는 능력 - Panoptic Cut을 통한 효율적인 객체 마스크 생성 - 추가적인 학습 없이도 우수한 성능 달성 - 객체 경계를 정확하게 식별하는 능력

- **Performance Highlights**: LaVG는 Pascal VOC, Pascal Context, COCO-object, COCO-stuff, ADE 20K와 같은 다섯 개의 공개 데이터 세트에서 기존의 OVSeg 모델보다 뛰어난 성능을 보였습니다. LaVG는 특히 객체 경계를 더 정확하게 식별할 수 있다는 장점이 있으며, 이는 시각적으로 더욱 매력적인 세그멘테이션 결과를 제공합니다. 또한, LaVG는 학습이 필요하지 않기 때문에 계산 비용이 매우 적습니다.



### Surgical-VQLA++: Adversarial Contrastive Learning for Calibrated Robust Visual Question-Localized Answering in Robotic Surgery (https://arxiv.org/abs/2408.04958)
Comments:
          Accepted by Information Fusion. Code and data availability: this https URL

- **What's New**: This paper proposes a new Surgical-VQLA++ framework that aims to improve the performance and robustness of existing Visual Question Localized-Answering (VQLA) models in the domain of robotic surgery. The key contribution of the paper is the introduction of a novel Calibrated Co-Attention Gated Vision-Language (C^2G-ViL) embedding module that effectively aligns multimodal representations, allowing for more accurate and robust localization and answering. Furthermore, the paper introduces an adversarial contrastive training strategy to enhance the model's robustness to noise and image corruption, which is common in medical data. Lastly, the paper expands existing endoscopic datasets, EndoVis-18-VQLA and EndoVis-17-VQLA, to create larger and more comprehensive datasets for VQLA tasks.

- **Technical Details**: The core of the Surgical-VQLA++ framework is the C^2G-ViL embedding module, which consists of three key components: 1) Multimodal Collaborated Calibration, which aligns and normalizes multimodal representations; 2) Global Contextual Calibration, which enhances robustness by capturing subtle feature perturbations; and 3) Adversarial Contrastive Training, which further boosts model performance and robustness by learning from adversarial examples. The framework uses DeiT backbone for deep feature learning. To improve the training process, the paper investigates different combinations of loss functions to achieve multi-task convergence.

- **Performance Highlights**: The proposed Surgical-VQLA++ framework outperforms existing models in terms of both accuracy and robustness. It achieves an improvement of 1.12% in accuracy and 1.55% in mIoU in overall performance compared to the second-best model. In robustness tests, it further exceeds the second-best by 1.64% in accuracy and 1.70% in mIoU, demonstrating the effectiveness of the proposed method. The model also achieves an inference speed of 150.6 FPS, which is efficient for real-time applications.

- **Datasets**: The paper presents two comprehensive surgical datasets for VQLA tasks, EndoVis-18-VQLA and EndoVis-17-VQLA, based on public EndoVis18 and EndoVis17 datasets. These datasets have been expanded to include a total of 17,269 QA pairs covering contents such as surgical organs, instruments, actions, and instrument locations. Each pair also comes with a corresponding bounding box to localize the answer.



### LLaVA-VSD: Large Language-and-Vision Assistant for Visual Spatial Description (https://arxiv.org/abs/2408.04957)
- **What's New**: LLaVA-VSD는 이미지 내 객체 간 공간 관계를 설명하는 텍스트를 생성하는 새로운 모델입니다. 기존 VSRC (Visual Spatial Relationship Classification) 방법은 이미지에서 두 객체 간의 공간 관계만 출력하는데, LLaVA-VSD는 세계 지식 (world knowledge)을 활용하고 일반적인 언어 능력을 갖추고 있습니다. 이를 통해, LLaVA-VSD는 이미지에서 객체 간 관계에 대한 질문에 답변하는 등 다중 모드 대화 능력을 선보입니다.



### Capsule Vision 2024 Challenge: Multi-Class Abnormality Classification for Video Capsule Endoscopy (https://arxiv.org/abs/2408.04940)
Comments:
          6 pages

- **What's New**: Capsule Vision 2024 Challenge는 비디오 캡슐 내시경 (VCE) 영상에서 다중 클래스 이상 상태를 분류하는 AI 모델 개발을 위한 국제 경진 대회입니다. 이 대회는 오스트리아 크렘스의 다뉴브 사립 대학교 의학부 의료 영상 분석 및 인공지능 연구 센터 (MIAAI)와 인도 첸나이에 있는 인도 정보 기술, 디자인 및 제조 연구소 (IIITDM)가 주관하는 제 9회 컴퓨터 비전 및 이미지 처리 국제 학술 대회 (CVIP 2024)와 함께 개최됩니다. 특히, VCE 영상에서 이상 상태를 자동으로 분류하는 AI 기반 모델을 개발하고 평가할 기회를 제공하며, 소화기내과 전문의의 업무 부담을 줄이고 진단 정확도를 높이는 데 목표를 두고 있습니다.



### UAV-Enhanced Combination to Application: Comprehensive Analysis and Benchmarking of a Human Detection Dataset for Disaster Scenarios (https://arxiv.org/abs/2408.04922)
Comments:
          This Paper is accepted for 27th International Conference on Pattern Recognition (ICPR 2024)

- **What's New**: 본 논문에서는 드론을 이용한 재난 구조 작업에서 인간 탐지를 위한 새로운 데이터셋인 'C2A' 데이터셋을 소개합니다. 기존 데이터셋의 부족함을 해결하기 위해 실제 재난 현장 사진에 다양한 인간 자세를 합성하여 만들어졌습니다.

- **Technical Details**: C2A 데이터셋은 실제 드론으로 촬영된 재난 현장 이미지에 인간 자세 이미지를 합성하여 생성됩니다. 이를 통해 다양한 재난 상황에서 인간 탐지 모델 학습을 가능하게 합니다. 또한, 폐쇄된 인간, 즉 부분적으로 가려진 인간의 탐지를 위해 데이터셋에 가려짐 요소를 추가하여 모델 학습의 난이도를 높였습니다. C2A 데이터셋은 다양한 인간 자세와 재난 현장 정보를 포함하여 재난 상황의 심각성을 평가하는 데 도움이 됩니다.

- **Performance Highlights**: 실험 결과 C2A 데이터셋으로 미세 조정된 모델은 일반적인 항공 데이터셋으로 사전 훈련된 모델보다 뛰어난 성능을 보였습니다. 또한, C2A 데이터셋을 일반적인 인간 데이터셋과 결합하면 다양한 시나리오에서 최적의 성능과 일반화 성능을 달성할 수 있음을 보여줍니다.

- **Contributions**: 본 연구의 기여는 다음과 같습니다.
- 드론을 이용한 재난 구조 작업에서 인간 탐지 모델 학습을 위한 새로운 데이터셋 C2A를 소개합니다.
- 다양한 재난 상황을 반영한 C2A 데이터셋 생성 파이프라인을 개발합니다.
- C2A 데이터셋은 다양한 인간 자세와 재난 현장 정보를 포함하여 재난 상황의 심각성을 평가하는 데 도움이 됩니다.
- 실험 결과 C2A 데이터셋은 인간 탐지 모델의 성능을 향상시키고 운영상의 실현 가능성을 높이는 데 기여합니다.

- **Limitations**: 본 연구에서 생성된 데이터셋은 실제 재난 현장과 완벽히 동일하지 않다는 점에 유의해야 합니다.

- **Future Directions**: 본 연구는 앞으로 드론을 이용한 재난 구조 작업에서 인간 탐지 모델의 성능을 더욱 향상시키기 위해 노력할 것입니다. 특히, 실제 재난 현장에서 수집된 데이터를 활용하여 더욱 현실적인 데이터셋을 생성하고 모델 학습을 위한 다양한 방법을 연구할 것입니다.



### Avoid Wasted Annotation Costs in Open-set Active Learning with Pre-trained Vision-Language Mod (https://arxiv.org/abs/2408.04917)
- **What's New**: 이 논문은 기존의 액티브 러닝(AL) 방법론이 오픈셋 데이터에서 겪는 한계점을 해결하기 위해 CLIP 기반의 새로운 데이터 선택 전략인 CLIPNAL을 제안합니다. CLIPNAL은 오픈셋 데이터에서 불필요한 주석 비용을 최소화하면서 모델 성능을 향상시키는 데 중점을 둡니다.



### GuidedNet: Semi-Supervised Multi-Organ Segmentation via Labeled Data Guide Unlabeled Data (https://arxiv.org/abs/2408.04914)
Comments:
          Accepted by ACM MM2024, 10 pages, 5 figures

- **What's New**: 본 논문에서는 Semi-supervised multi-organ medical image segmentation (반지도 학습 기반 다중 장기 의료 영상 분할)을 위한 새로운 방법인 GuidedNet을 제안합니다. GuidedNet은 Labeled data (레이블된 데이터)에서 얻은 지식을 활용하여 Unlabeled data (레이블되지 않은 데이터) 학습을 안내하는 것을 목표로 합니다. 핵심 개념은 Feature space (특징 공간)에서 서로 가까운 Labeled data와 Unlabeled data의 Voxel features (복셀 특징)는 같은 클래스에 속할 가능성이 높다는 것입니다. 이를 기반으로 3D Consistent Gaussian Mixture Model (3D-CGMM)을 설계하여 Labeled data의 feature distribution (특징 분포)을 활용하여 생성된 Pseudo-labels (의사 레이블)을 수정합니다. 또한, Labeled data에서 얻은 사전 지식을 활용하여 Unlabeled data 학습을 안내하여 작고 복잡한 장기의 분할 정확도를 향상시키는 Knowledge Transfer Cross Pseudo Supervision (KT-CPS) 전략을 도입합니다.



### Clustering-friendly Representation Learning for Enhancing Salient Features (https://arxiv.org/abs/2408.04891)
Comments:
          12 pages, 6 figures, 28th Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD2024)

- **What's New**: 본 논문은 비지도 이미지 클러스터링을 위한 표현 학습(Representation Learning) 방법인 **'대조적 IDFD(cIDFD)'**를 제안합니다. cIDFD는 중요한 특징(Feature)을 강화하여 클러스터링 성능을 향상시키는 방법입니다. cIDFD는 클러스터링 친화적인 대조 학습 방법을 확장하고, 참조 데이터셋(Reference Dataset)을 활용하여 중요한 특징과 중요하지 않은 특징을 분리하는 대조적 분석(Contrastive Analysis) 방식을 손실 함수 설계에 통합했습니다.



### ProxyCLIP: Proxy Attention Improves CLIP for Open-Vocabulary Segmentation (https://arxiv.org/abs/2408.04883)
Comments:
          Accepted to ECCV 2024. Code available at this https URL

- **What's New**: 본 논문은 오픈-보캐뷸러리 의미론적 분할(Open-vocabulary semantic segmentation)을 위한 새로운 프레임워크인 ProxyCLIP을 소개합니다. ProxyCLIP은 CLIP과 VFM(Vision Foundation Models)의 강점을 결합하여 개선된 성능을 제공합니다. ProxyCLIP은 VFM의 공간적 특징 대응을 활용하여 CLIP을 강화하는데, 이를 통해 VFM의 강력한 지역 일관성을 유지하면서 CLIP의 뛰어난 제로-샷 전이 능력을 보존합니다. ProxyCLIP은 또한 적응형 정규화 및 마스킹 전략을 사용하여 다양한 VFM에 적응할 수 있습니다. ProxyCLIP은 훈련이 필요 없는 방법으로, 8개의 벤치마크에서 평균 mIoU를 40.3에서 44.4로 크게 향상시켜 오픈-보캐뷸러리 분할 작업에서 공간적 정확성과 의미론적 풍부함 사이의 간극을 줄이는 뛰어난 효과를 보여줍니다.



### On the Element-Wise Representation and Reasoning in Zero-Shot Image Recognition: A Systematic Survey (https://arxiv.org/abs/2408.04879)
Comments:
          24 pages, 7 figures

- **What's New**: This paper presents a comprehensive overview of element-wise learning techniques in Zero-Shot Image Recognition (ZSIR), covering object recognition, compositional recognition, and foundation model-based open-world recognition. It analyzes the strengths and weaknesses of different approaches and provides a unified taxonomy, technical details, and benchmark datasets. The paper also highlights the applications of element-wise ZSIR and explores future research directions.

- **Technical Details**: Element-wise learning in ZSIR leverages the ability of models to learn and reason about new concepts by decomposing them into their basic elements (attributes, parts, functions) and combining them to represent novel concepts. This approach aims to mimic human cognition, where we recognize things by understanding their components and their relationships.

- **Performance Highlights**: The paper does not specifically discuss performance highlights, as it focuses on providing a comprehensive overview of the field. However, it does mention the advantages of ZSIR over other approaches like few-shot and one-shot learning, particularly in terms of reducing data requirements.

- **Applications**: Element-wise ZSIR has various applications, including object recognition, image retrieval, visual question answering, and scene classification. It can also be used for analyzing complex visual scenes and understanding the relationships between objects and their attributes.

- **Future Directions**: Future research in element-wise ZSIR can focus on improving the robustness of models to noisy data and adversarial examples, enhancing the ability to reason with complex relationships between elements, and exploring the potential of combining element-wise approaches with other ZSIR techniques.



### ChatGPT Meets Iris Biometrics (https://arxiv.org/abs/2408.04868)
Comments:
          Published at IJCB 2024

- **What's New**: 이 연구는 ChatGPT-4와 같은 대규모 언어 모델(LLM)의 능력을 활용하여 안면 인식보다 덜 흔하고 전문적인 분야인 홍채 인식 분야에서의 잠재력을 탐구합니다. 이 연구는 ChatGPT와 같은 AI 도구가 홍채 이미지를 얼마나 잘 이해하고 분석할 수 있는지 조사합니다. 영점 학습 접근 방식(zero-shot learning approach)을 사용한 일련의 주의 깊게 설계된 실험을 통해, 다양한 데이터 세트, 프레젠테이션 공격(presentation attacks), 안경과 같은 가리움(occlusion), 기타 실제 환경 변형을 포함한 다양한 까다로운 조건에서 ChatGPT-4의 기능을 평가했습니다. 이 연구 결과는 ChatGPT-4의 뛰어난 적응력과 정확성을 보여주며, 홍채 인식에 대한 메이크업과 같은 미묘한 효과를 감지하면서 독특한 홍채 특징을 식별하는 능숙함을 보여줍니다. 구글의 AI 모델인 Gemini Advanced와의 비교 분석은 복잡한 홍채 분석 작업에서 ChatGPT-4의 더 나은 성능과 사용자 경험을 강조했습니다. 이 연구는 전문적인 생체 인식 애플리케이션에 대한 LLM의 사용을 검증할 뿐만 아니라 생체 인식 데이터에서 중요한 통찰력을 얻기 위해 미묘한 쿼리 프레이밍(query framing) 및 상호 작용 디자인의 중요성을 강조합니다. 이 연구 결과는 미래 연구와 더욱 적응력이 뛰어나고 효율적이며 강력하고 대화형 생체 인식 보안 솔루션 개발을 위한 유망한 길을 제시합니다.



### mPLUG-Owl3: Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models (https://arxiv.org/abs/2408.04840)
- **What's New**: mPLUG-Owl3는 긴 이미지 시퀀스를 이해하는 데 뛰어난 성능을 보여주는 새로운 다중 모달 대규모 언어 모델(MLLM)입니다. 기존 모델과 달리, mPLUG-Owl3는 검색된 이미지-텍스트 지식, 혼합된 이미지-텍스트, 그리고 긴 비디오를 포함하는 시나리오에서 긴 이미지 시퀀스 이해를 향상시킵니다. 특히, mPLUG-Owl3는 긴 다중 이미지 시나리오를 처리하기 위해 비전과 언어를 공통의 언어 기반 의미 공간에 효율적으로 통합하는 새로운 하이퍼 어텐션 블록을 제안합니다.



### Self-augmented Gaussian Splatting with Structure-aware Masks for Sparse-view 3D Reconstruction (https://arxiv.org/abs/2408.04831)
- **What's New**: 이 논문에서는 희소 뷰 3D 재구성을 위한 자체 증강된 거칠기에서 미세한 가우시안 스플래팅(Gaussian Splatting) 프레임워크를 제안합니다. 이 프레임워크는 구조 인식 마스크(structure-aware mask)를 사용하여 구조 정보를 활용하고 3D 기하 증강과 지각적 뷰 증강을 통해 더 나은 재구성 결과를 얻습니다.

- **Technical Details**: 제안된 방법은 희소 뷰 입력으로부터 기본적인 3D 표현을 얻기 위해 거친 가우시안 모델을 사용하는 것으로 시작합니다. 그런 다음, 미세한 가우시안 네트워크를 개발하여 3D 기하 증강과 지각적 뷰 증강을 통해 출력의 일관성 있고 상세한 표현을 향상시킵니다. 훈련 중에는 구조 인식 마스킹 전략을 설계하여 희소 입력 및 노이즈에 대한 모델의 견고성을 더욱 향상시킵니다.

- **Performance Highlights**: MipNeRF360 및 OmniObject3D 데이터 세트에 대한 실험 결과는 제안된 방법이 희소 입력 뷰에 대해 지각적 품질과 효율성 모두에서 최첨단 성능을 달성함을 보여줍니다.



### One Shot is Enough for Sequential Infrared Small Target Segmentation (https://arxiv.org/abs/2408.04823)
- **What's New**: 본 논문은 단일 샷 (one-shot) 학습 방식을 이용하여 적은 데이터로도 적외선 소형 표적 시퀀스 (infrared small target sequence) 분할을 수행하는 새로운 방법을 제시합니다. 이 방법은 기존의 대규모 모델인 Segment Anything Model (SAM)을 활용하여 적외선 소형 표적 시퀀스 분할 문제에 적용합니다. 특히, 하나의 주석이 달린 프레임 (annotated frame)만을 참조로 사용하여 시퀀스 내 다른 프레임의 소형 표적을 정확하게 분할합니다. 이러한 새로운 방법은 기존의 다중 샷 (many-shot) 방식의 학습에 비해 뛰어난 성능을 보여줍니다.



### Rethinking Multiple Instance Learning: Developing an Instance-Level Classifier via Weakly-Supervised Self-Training (https://arxiv.org/abs/2408.04813)
- **What's New**: 본 논문은 기존의 MIL(Multiple Instance Learning) 문제를 semi-supervised instance classification 문제로 재정의하여, 모든 labeled 및 unlabeled instances를 활용하여 더 나은 분류기를 학습할 수 있도록 합니다. 이를 위해 weakly-supervised self-training 방법을 제안하며, 이는 positive bag labels를 활용하여 pseudo labels에 대한 global 및 local constraint를 적용하여 pseudo labels의 degeneration을 방지하고 classifier가 hard positive instances를 학습하도록 합니다. 이러한 global 및 local constraint를 통해 pseudo labels는 true labels에 점차적으로 근접하게 됩니다.



### UniBench: Visual Reasoning Requires Rethinking Vision-Language Beyond Scaling (https://arxiv.org/abs/2408.04810)
- **What's New**: UniBench는 50개 이상의 비전-언어 모델(VLM) 벤치마크를 통합한 구현체입니다. 벤치마크는 개체 인식, 공간 인식, 계산 등 다양한 기능을 망라합니다. UniBench는 VLM의 진행 상황을 체계적으로 평가하기 위한 도구를 제공합니다.



### Hyper-YOLO: When Visual Object Detection Meets Hypergraph Computation (https://arxiv.org/abs/2408.04804)
- **What's New**: Hyper-YOLO는 기존 YOLO 모델의 한계를 극복하고 시각적 특징 간 복잡한 고차 상관관계를 포착하기 위해 하이퍼그래프 계산을 통합한 새로운 객체 감지 방법을 소개합니다. Hyper-YOLO는 특징 추출을 위한 백본에 MANet(Mixed Aggregation Network)을 통합하고, 하이퍼그래프 기반 크로스 레벨 및 크로스 포지션 표현 네트워크인 HyperC2Net을 도입하여 고차 상호 작용을 통해 5개의 스케일에서 특징을 융합합니다. Hyper-YOLO는 고차 정보 인식 능력을 향상시켜 객체 감지 성능을 개선합니다.



### FewShotNeRF: Meta-Learning-based Novel View Synthesis for Rapid Scene-Specific Adaptation (https://arxiv.org/abs/2408.04803)
- **What's New**: 본 논문은 제한된 멀티뷰 이미지를 통해 실제 객체의 새로운 뷰를 생성하는 문제를 해결하기 위한 FewShotNeRF라는 새로운 접근 방식을 제시합니다. 이 방법은 메타 학습(Meta-Learning)을 활용하여 최적의 초기화(Initialization)를 얻어 특정 장면에 대한 신경 복사장(NeRF, Neural Radiance Field)의 빠른 적응을 가능하게 합니다. 메타 학습 과정은 카테고리 내에 공유되는 기하학 및 텍스처를 포착하는 데 중점을 두며, 이는 가중치 초기화(Weight Initialization)에 내장됩니다. 이러한 접근 방식은 NeRF의 학습 과정을 가속화하고, 최근의 위치 인코딩(Positional Encoding) 발전을 활용하여 NeRF를 장면에 맞추는 데 필요한 시간을 줄임으로써 메타 학습의 내부 루프 최적화를 가속화합니다. 특히, 이 방법은 다양한 카테고리에 대한 강력한 3D 사전 정보를 구축하기 위해 많은 3D 장면에서 메타 학습을 가능하게 합니다. 3D 오픈 소스 데이터 세트에 있는 공통 객체(Common Objects in 3D)에 대한 광범위한 평가를 통해 객체의 고품질 새로운 뷰를 생성하는 데 있어 메타 학습의 효과와 잠재력을 실증적으로 보여줍니다.



### SOD-YOLOv8 -- Enhancing YOLOv8 for Small Object Detection in Traffic Scenes (https://arxiv.org/abs/2408.04786)
Comments:
          15 pages, 14 figures

- **What's New**: 본 논문에서는 소형 객체 탐지를 위한 새로운 모델인 **SOD-YOLOv8** (Small Object Detection YOLOv8)을 제안합니다. SOD-YOLOv8은 YOLOv8의 **GFPN (Generalized Feature Pyramid Network)**을 개선하여 다양한 수준의 특징을 통합하여 소형 객체 탐지 정확도를 향상시킵니다. 특히, YOLOv8에 **네 번째 탐지 계층**을 추가하여 고해상도 공간 정보를 효과적으로 활용하고, **C2f-EMA (Efficient Multi-Scale Attention Module)** 모듈에 EMA Attention (Efficient Multi-Scale Attention Module)을 통합하여 특징 추출을 개선합니다. 또한, 기존의 CIoU (Complete Intersection over Union)를 대체하는 **PIoU (Powerful-IoU)**를 도입하여 중간 품질의 앵커 박스에 집중하고 예측된 바운딩 박스와 실제 바운딩 박스 모서리 차이에 대한 페널티를 추가하여 계산을 단순화하고 수렴 속도를 높이며 탐지 정확도를 향상시킵니다.



### BRAT: Bonus oRthogonAl Token for Architecture Agnostic Textual Inversion (https://arxiv.org/abs/2408.04785)
- **What's New**: 이 논문에서는 기존의 UNet(U-Net) 기반 텍스트 인버전(Textual Inversion)에 대한 대안으로 비전 트랜스포머(Vision Transformer)를 활용하는 새로운 방법을 제안합니다. 또한, UNet에 의존하지 않는 새로운 최적화 전략인 BRAT(Bonus Token)를 소개하여 텍스트 인버전의 성능을 향상시킵니다. BRAT는 새로운 토큰을 추가하고 직교성(orthogonality)을 강제함으로써, 소스 이미지에 대한 충실도를 높이고 비전 트랜스포머를 사용하여 프롬프트(prompt)에 대한 일관성을 향상시킵니다.



### Data-Driven Pixel Control: Challenges and Prospects (https://arxiv.org/abs/2408.04767)
Comments:
          Accepted to the Conference on Dynamic Data-Driven Applications Systems (DDDAS2024)

- **What's New**: 이 논문은 픽셀 단위의 동적 감지 (dynamic sensing)와 비디오 단위의 컴퓨터 비전 분석을 결합한 데이터 기반 시스템을 연구하며, 감지 전단 (sensor front-end)과 계산 후단 (computational back-end) 간의 데이터 이동을 최소화하는 피드백 제어 루프 (feedback control loop)를 제안합니다. 이 시스템은 객체 감지 및 추적 정확도를 저하시키지 않으면서도 대역폭을 줄이고 에너지 효율을 높입니다.



### Novel adaptation of video segmentation to 3D MRI: efficient zero-shot knee segmentation with SAM2 (https://arxiv.org/abs/2408.04762)
- **What's New**: 본 연구는 3D 무릎 MRI 이미지의 0-샷(zero-shot) 단일 프롬프트 분할을 위한 새로운 방법을 소개합니다. 이 방법은 일반적인 목적 분할 모델인 Segment Anything Model 2 (SAM2)를 활용하여 3D 의료 볼륨의 슬라이스를 개별 비디오 프레임으로 처리하여 SAM2의 고급 기능을 활용하여 움직임 및 공간 인식 예측을 생성합니다. 이러한 접근 방식은 추가 훈련이나 미세 조정 없이도 0-샷 분할 작업을 효율적으로 수행할 수 있으며, 단일 프롬프트만으로 무릎 MRI 스캔의 구조를 정확하게 구분할 수 있습니다. 연구 결과, SAM2는 OAI-ZIB 데이터셋에서 3D 무릎 뼈 분할에 대해 높은 정확도를 달성하여 경골에 대한 Dice 유사도 계수(DSC)가 0.9643에 도달했습니다.

- **Technical Details**: 본 연구는 Segment Anything Model 2 (SAM2)를 기반으로 3D 무릎 MRI 이미지를 슬라이스 단위로 비디오 프레임으로 처리하여 움직임 및 공간 정보를 활용하는 새로운 0-샷 단일 프롬프트 분할 방법을 제시합니다. SAM2는 프롬프트 기반 영상 분할 모델로 이미지 및 비디오에 적용 가능하며, 이전 프레임의 정보를 기억하여 맥락을 유지하고 정확성을 높입니다. 본 연구에서는 점 프롬프트 방식을 사용하여 무릎 뼈의 중심 위치를 지정하여 분할 작업을 수행합니다. 또한 SAM1과 SAM2의 성능을 비교 분석하여 SAM2의 효율성과 정확성을 검증했습니다.

- **Performance Highlights**: OAI-ZIB 데이터셋을 사용한 실험 결과, SAM2는 경골에 대해 0.9643의 Dice 유사도 계수(DSC)를 달성하여 높은 정확도를 보였습니다. 또한 다양한 SAM2 모델 크기와 프롬프트 방식에 대한 실험 결과를 제시하고, SAM1과의 비교 결과를 통해 SAM2의 우수성을 확인했습니다. 특히, SAM2는 단일 프롬프트만으로도 높은 정확도를 달성하여 의료 이미지 분석의 효율성을 향상시키는 가능성을 보여줍니다.



### Weak-Annotation of HAR Datasets using Vision Foundation Models (https://arxiv.org/abs/2408.05169)
Comments:
          8 pages, 3 figures, accepted at ISWC'24: International Symposium on Wearable Computers, Oct, 2024

- **What's New**: 본 논문은 인체 활동 인식(HAR) 분야에서 wearable 데이터 annotation의 효율성을 높이기 위한 새로운 방법을 제안합니다. 기존의 HAR 데이터셋은 데이터 양과 다양성 측면에서 부족했으며, wearable 데이터 annotation은 수작업으로 이루어져 시간과 노력이 많이 소요되는 작업이었습니다. 이 연구는 CLIP과 같은 비전 기반 모델을 활용하여 wearable 데이터를 시각적으로 표현하고 군집화하여 annotation 작업을 간소화하는 새로운 annotation 파이프라인을 제안합니다. 이를 통해 소량의 데이터만 사람이 annotation 하는 것만으로도 높은 정확도를 달성할 수 있습니다.

- **Technical Details**: 본 연구에서는 pretrained vision foundation model (CLIP)을 사용하여 wearable 데이터를 시각적으로 표현하고 이를 통해 활동을 군집화합니다. 각 군집의 대표 클립 (centroid clip)만 사람이 annotation 하면 나머지 데이터는 자동으로 annotation 됩니다. 이 방법은 기존의 fully-supervised 학습과 비슷한 성능을 보이며 annotation 작업을 상당히 줄일 수 있습니다.

- **Performance Highlights**: 본 연구의 annotation 파이프라인은 3가지 HAR 데이터셋에서 평균 90%에 가까운 정확도를 달성했습니다. 또한, weakly annotated 데이터를 사용하여 fully-supervised deep learning classifier와 동등한 성능을 보였습니다.



### Modeling Electromagnetic Signal Injection Attacks on Camera-based Smart Systems: Applications and Mitigation (https://arxiv.org/abs/2408.05124)
Comments:
          13 pages, 10 figures, 4 tables

- **What's New**: 이 연구는 카메라 기반 스마트 시스템에서 전자기 신호 주입 공격을 효과적으로 모의하는 새로운 방법을 제시합니다. 이 방법은 실제 공격 없이도 다양한 응용 프로그램에 대한 공격의 영향을 평가할 수 있도록 하여 시스템의 취약성을 효과적으로 분석하고 방어 메커니즘을 개발하는 데 도움이 됩니다. 또한, 이 연구는 공격에 대한 모델의 견고성을 향상시키는 적대적 훈련(adversarial training)의 효과를 보여주는 사례 연구를 통해 시스템의 보안 강화 방안을 제시합니다.



### Beyond the Eye: A Relational Model for Early Dementia Detection Using Retinal OCTA Images (https://arxiv.org/abs/2408.05117)
- **What's New**: 본 논문에서는 망막 OCTA 영상을 사용하여 조기 발병 알츠하이머병(EOAD) 및 경도 인지 장애(MCI)를 구별하는 혁신적인 폴라넷+ 모델을 소개합니다. 폴라넷+는 임상의에게 친숙한 ETDRS 그리드 분석을 구현하기 위해 OCTA 영상을 카르테시안 좌표에서 극좌표로 매핑하는 방식을 사용합니다. 또한, 이 논문은 다중 뷰 모듈을 도입하여 3차원을 따라 영상을 직렬화하고 분석하여 포괄적인 임상적으로 유용한 정보 추출을 가능하게 합니다. 마지막으로, 시퀀스 임베딩을 그래프로 추상화하여 탐지 작업을 일반적인 그래프 분류 문제로 변환합니다. 다중 뷰 모듈 이후에는 지역 관계 모듈을 적용하여 서브 영역 간의 관계를 파악합니다. 이러한 지역 관계 분석은 알려진 눈-뇌 연결을 검증하고 새로운 차별적 패턴을 밝혀냅니다.



### Multi-dimensional Parameter Space Exploration for Streamline-specific Tractography (https://arxiv.org/abs/2408.05056)
Comments:
          Accepted at MICCAI 2024 International Workshop on Computational Diffusion MRI

- **What's New**: 본 논문은 기존 트랙토그래피(tractography) 알고리즘의 한계를 극복하기 위해 스트림라인 특정 매개변수(SSP, streamline-specific parameters)를 이용한 새로운 접근 방식을 제시합니다. 기존 알고리즘은 일반적인 매개변수를 사용하여 모든 트랙토그래피에 적용하지만, SSP는 각 스트림라인에 맞춤형 매개변수를 사용하여 더 정확하고 효율적인 트랙토그래피를 수행할 수 있도록 합니다.



### Integrating Edge Information into Ground Truth for the Segmentation of the Optic Disc and Cup from Fundus Images (https://arxiv.org/abs/2408.05052)
- **What's New**: 이 논문은 안구 디스크와 컵의 에지를 학습함으로써 안구 디스크 및 컵 분할 정확도를 향상시키는 새로운 방법을 제안합니다. 기존 U-Net 기반 모델들은 종종 과도한 분할 또는 부족한 분할을 하는 문제가 있었는데, 이는 에지를 학습하지 않아서 발생했습니다. 본 논문에서는 2D Laplacian 필터를 사용하여 기존의 안구 디스크 및 컵 지표에서 에지를 추출하여 에지 지표를 추가로 학습합니다. 그 결과, U-Net 모델의 Dice 및 Hausdorff 거리 지표가 개선되었습니다.



### Benchmarking Conventional and Learned Video Codecs with a Low-Delay Configuration (https://arxiv.org/abs/2408.05042)
- **What's New**: 이 논문은 저지연 환경에서 최첨단 기존 및 학습 기반 비디오 코딩 방법의 비교 연구를 수행합니다. 연구 대상은 MPEG H.266/VVC VTM, JVET ECM, AOM AV1 libaom, AOM AVM, DCVC-DC 및 DCVC-FM입니다. MPEG 및 AOM 공통 테스트 조건에 따라 저지연 모드로 평가되었습니다. 평가 결과 JVET ECM 코덱이 테스트된 모든 코덱 중 가장 우수한 코딩 성능을 제공하며, AOM AVM 대비 평균 BD-레이트에서 16.1% (PSNR 기반) 개선, DCVC-FM 대비 11.0% 개선을 보였습니다. 또한 DCVC-DC 및 DCVC-FM과 같은 학습 기반 비디오 코덱은 배경 움직임이 큰 테스트 콘텐츠에 대해 일관성 없는 성능을 보였습니다.



### XNN: Paradigm Shift in Mitigating Identity Leakage within Cloud-Enabled Deep Learning (https://arxiv.org/abs/2408.04974)
- **What's New**: XNN과 XNN-d는 클라우드 기반 딥 러닝에서 개인 정보 유출 문제를 해결하기 위한 새로운 방법론입니다. XNN은 훈련 단계에서 랜덤 순열(random permutation)과 행렬 곱셈(matrix multiplication)을 사용하여 특징 맵(feature map)을 난독화하여 개인 정보 유출을 방지하는 동시에 훈련 정확도를 유지합니다. XNN-d는 추론 단계에서 적대적 학습(adversarial training)을 사용하여 생성적 적대적 노이즈(generative adversarial noise)를 통합하여 개인 식별 정보 추출을 위한 블랙박스 공격을 방어합니다. 또한, 증류된 얼굴 인식 네트워크(distilled face recognition network)를 사용하여 난독화된 특징을 처리하여 정확한 식별을 보장합니다.



### Model Debiasing by Learnable Data Augmentation (https://arxiv.org/abs/2408.04955)
- **What's New**: 이 논문은 지도 학습 (supervised learning) 시나리오에서 흔히 사용되는 편향 (bias) 제거 기술과 달리, 편향이 알려지지 않은 비지도 학습 (unsupervised learning) 환경에서 편향된 데이터로부터 학습하는 새로운 방법을 제시합니다. 이는 실제 상황에서 더욱 현실적이며 도전적인 문제입니다. 이 방법은 데이터 증강 (data augmentation) 전략을 사용하여 모델이 편향된 데이터에 대한 일반화 능력을 향상시킵니다.



### CROCODILE: Causality aids RObustness via COntrastive DIsentangled LEarning (https://arxiv.org/abs/2408.04949)
Comments:
          MICCAI 2024 UNSURE Workshop, Accepted for presentation, Submitted Manuscript Version, 10 pages

- **What's New**: 이 논문에서는 CROCODILE 프레임워크를 소개하며, 인과관계 분석 도구를 사용하여 특징 분리 (feature disentanglement), 대조 학습 손실 (contrastive learning losses), 그리고 사전 지식 주입을 통해 도메인 이동 (domain shift)에 대한 모델의 견고성을 향상시키는 방법을 보여줍니다. 이를 통해 모델은 가짜 상관관계 (spurious correlations)에 대한 의존도를 줄이고 이미지에서 예측으로 이어지는 메커니즘을 더 잘 학습하며, 분포 외 (out-of-distribution, OOD) 데이터에서 기준 모델 성능을 능가합니다.



### Surveying the Landscape of Image Captioning Evaluation: A Comprehensive Taxonomy and Novel Ensemble Method (https://arxiv.org/abs/2408.04909)
- **What's New**: 본 연구는 이미지 캡셔닝 평가 지표 70개 이상을 조사하여 최초의 분류 체계를 제시합니다.  이 연구는 300개 이상의 논문에 사용된 지표들을 분석하여 다양한 지표들이 존재함에도 불구하고 실제로 많이 사용되는 지표는 5개뿐이며, 이러한 지표들은 사람의 평가와 약한 상관관계를 갖는다는 점을 밝혀냈습니다.  본 연구에서는 다양한 지표들을 활용하여 사람의 평가와 가장 높은 상관관계를 보이는 EnsembEval이라는 새로운 앙상블 평가 지표를 제안합니다.  이를 통해 다양한 지표들을 활용하는 것이 이미지 캡셔닝 모델 평가의 정확성을 높이는 데 중요하다는 점을 보여줍니다.



### MSG-Chart: Multimodal Scene Graph for ChartQA (https://arxiv.org/abs/2408.04852)
Comments:
          Accpeted by CIKM Short 2024

- **What's New**: This paper proposes a novel multimodal scene graph approach for ChartQA. This graph representation, encompassing both visual and textual aspects, enhances the understanding of chart elements' relationships and patterns.

- **Technical Details**: The proposed model comprises two graphs: a visual graph capturing spatial relations based on visual features and a textual graph representing semantic knowledge using textual features. The visual graph utilizes a fully connected structure with edge weights determined by the Euclidean distance between objects, prioritizing closer neighbors. Each visual node is initialized with the mean of hidden states from corresponding image patches. The textual graph consists of label nodes and OCR nodes, connected based on chart semantics. Label nodes represent object labels, while OCR nodes capture OCR text extracted from non-shape objects. Connections are established based on chart structure, such as x/y axis titles to labels, x/y axis labels to chart elements, and legend labels to markers. Each node is initialized with the mean of BERT embeddings of its corresponding text.

- **Performance Highlights**: Experiments demonstrate that incorporating the proposed graph module improves performance on public benchmarks, ChartQA and OpenCQA, surpassing previous approaches. This indicates that the multimodal scene graph effectively enhances the understanding of chart elements' structure and semantics, leading to better question-answering accuracy.



### Adversarially Robust Industrial Anomaly Detection Through Diffusion Mod (https://arxiv.org/abs/2408.04839)
- **What's New**: This paper introduces a new method for adversarially robust anomaly detection called **AdvRAD** (Adversarially Robust Anomaly Detection). It uses a diffusion model to simultaneously detect anomalies and purify adversarial noise, addressing the shortcomings of existing methods that suffer from high anomaly miss rates due to their inability to differentiate between anomaly signals and adversarial perturbations.

- **Technical Details**: AdvRAD leverages the denoising capability of diffusion models to remove adversarial perturbations while preserving anomaly signals. It achieves this by integrating the anomaly detection and adversarial purification processes within the diffusion model itself, allowing it to perform both functions simultaneously.

- **Performance Highlights**: Extensive experiments on benchmark datasets like MVTec AD, ViSA, and BTAD show that AdvRAD exhibits outstanding (certified) adversarial robustness while maintaining strong anomaly detection performance on par with state-of-the-art methods. It outperforms existing robust anomaly detection methods, demonstrating its effectiveness in mitigating adversarial attacks and maintaining accurate anomaly detection in real-world industrial settings.



### Geo-UNet: A Geometrically Constrained Neural Framework for Clinical-Grade Lumen Segmentation in Intravascular Ultrasound (https://arxiv.org/abs/2408.04826)
Comments:
          Accepted into the 15th workshop on Machine Learning in Medical Imaging at MICCAI 2024. (* indicates equal contribution)

- **What's New**: This paper presents a new neural framework called Geo-UNet for lumen segmentation in venous intravascular ultrasound (v-IVUS) images. Geo-UNet leverages the radial geometry of IVUS imaging for accurate lumen boundary estimation, addressing limitations of previous segmentation networks like UNet.



### On the Geometry of Deep Learning (https://arxiv.org/abs/2408.04809)
- **What's New**: 본 논문은 딥 러닝의 수학적 기초에서 진전을 이룰 수 있는 유망한 분야 중 하나인 딥 네트워크와 아핀 스플라인(affine spline)(다차원 연속적 조각 선형 함수)에 의한 함수 근사(function approximation) 간의 연결성을 살펴봅니다. 특히, 딥 네트워크의 아핀 스플라인 매핑(affine spline mapping)의 기하학적 특성(geometrical properties)을 이해하는 데 초점을 맞추어 지난 10년간의 연구를 개괄적으로 살펴봅니다. 특히, 입력 공간을 어떻게 테셀레이션(tessellation)(분할)하는지에 대한 연구를 살펴봅니다. 아핀 스플라인 연결과 기하학적 관점은 딥 네트워크의 내부 작동 방식을 이해하고 분석하며 개선하는 강력한 도구를 제공합니다.



### Improved Robustness for Deep Learning-based Segmentation of Multi-Center Myocardial Perfusion MRI Datasets Using Data Adaptive Uncertainty-guided Space-time Analysis (https://arxiv.org/abs/2408.04805)
Comments:
          Accepted for publication in JCMR, 2024

- **What's New**: 본 연구는 다중 센터 심장 자기 공명 영상(MRI) 데이터 세트의 자동 분석을 위한 딥 러닝 기반 방법론인 DAUGS(Deep-Learning-Assisted Uncertainty Guided Segmentation)를 제안합니다. 이 방법론은 딥 신경망(DNN)의 불확실성을 이용하여 다양한 센터, 소프트웨어 및 하드웨어 환경에서도 안정적인 결과를 제공합니다. 특히, 기존 방법론에 비해 외부 데이터 세트에서 더 나은 성능을 보여주고 있으며, 심장 MRI 데이터 세트의 분할에서의 불확실성을 줄이는 데 기여할 수 있습니다.



### Deep Learning-based Unsupervised Domain Adaptation via a Unified Model for Prostate Lesion Detection Using Multisite Bi-parametric MRI Datasets (https://arxiv.org/abs/2408.04777)
Comments:
          Accept at Radiology: Artificial Intelligence. Journal reference and external DOI will be added once published

- **What's New**: 본 연구는 다양한 b-값을 사용하여 수집된 다중 기관 전립선 병변 검출에서 지도 학습 모델의 성능을 향상시키는 유망하고 신뢰할 수 있는 전략으로 통합 모델을 사용한 확산 가중 영상(DWI) 기반 UDA(Unsupervised Domain Adaptation)를 제안합니다.



### Segmentation of Mental Foramen in Orthopantomographs: A Deep Learning Approach (https://arxiv.org/abs/2408.04763)
Comments:
          9 pages

- **What's New**: 이 논문은 심층 학습 모델을 사용하여 파노라마 방사선 사진에서 정신 구멍을 정확하게 탐지하고 분할하는 새로운 방법을 제시합니다. 이 연구에서는 원형 및 정사각형 마스크를 사용하여 다양한 분할 모델을 훈련하고 평가했습니다.  다양한 지표를 사용하여 모델의 효과를 평가했습니다.



### Embodied Uncertainty-Aware Object Segmentation (https://arxiv.org/abs/2408.04760)
Comments:
          IROS 2024

- **What's New**: 이 논문에서는 불확실성 인식 객체 인스턴스 분할(UncOS)을 소개하고, 이것이 신체화된 상호 작용 분할에 유용하다는 것을 보여줍니다. 로봇 인식의 불확실성을 다루기 위해 객체 분할의 가설 분포를 생성하는 방법을 제안합니다. 대규모 사전 훈련된 모델을 여러 번 쿼리하여 신뢰도 추정치와 함께 영역 요인화된 분할 가설 집합을 얻습니다. 이 과정은 보이지 않는 객체 분할 문제에서 최첨단 성능을 달성하는 분할 결과를 생성할 수 있습니다. 출력은 또한 모호성을 줄이기 위해 장면을 방해하기 위한 로봇 작업 선택을 위한 믿음 기반 프로세스의 입력으로 사용될 수 있습니다. 실제 로봇 실험을 통해 이 방법의 효과를 보여줍니다.



### Mini-Monkey: Multi-Scale Adaptive Cropping for Multimodal Large Language Models (https://arxiv.org/abs/2408.02034)
- **What's New**: Mini-Monkey, a lightweight multimodal large language model (MLLM) that utilizes a multi-scale adaptive cropping strategy (MSAC) to address the "sawtooth effect" caused by traditional cropping methods in image processing for MLLMs.



New uploads on arXiv(cs.AI)

### Application of Unsupervised Artificial Neural Network (ANN) Self_Organizing Map (SOM) in Identifying Main Car Sales Factors (https://arxiv.org/abs/2408.05110)
- **What's New**: 이 연구는 퍼지 델파이 기법(fuzzy Delphi technique)과 자기조직화 지도(self-organizing map, SOM)를 이용하여 이란 자동차 고객의 구매 의사 결정에 영향을 미치는 요인을 분석했습니다. 퍼지 도구를 사용하여 연구의 현실성을 높였으며, MATLAB 소프트웨어를 사용하여 네트워크를 개발 및 훈련했습니다.



### On the use of neurosymbolic AI for defending against cyber attacks (https://arxiv.org/abs/2408.04996)
Comments:
          Accepted to 18th International Conference on Neural-Symbolic Learning and Reasoning

- **What's New**: 본 논문은 사이버 보안 분야에서 뉴로심볼릭 AI(NeSy)의 사용 가능성을 제시합니다. NeSy는 연결주의 AI와 기호주의 AI를 결합하여 사이버 공격 탐지 및 대응에 사용됩니다. 기존의 AI 기반 사이버 보안 시스템이 겪는 한계점을 분석하고, NeSy가 이러한 문제들을 해결할 수 있는 잠재력을 강조합니다.

- **Technical Details**: 논문은 사이버 보안 운영 센터(SOC)에서 NeSy가 어떻게 활용될 수 있는지 설명하고, 구체적인 사용 사례들을 제시합니다. 또한, NeSy가 기존의 AI 방식들보다 어떤 이점을 제공하는지 기술적인 측면에서 살펴봅니다. 특히, 탐지 모델 학습, 상황 인식, 위협 정보 활용 및 응답 등의 핵심 기능에 초점을 맞추어 NeSy의 효용성을 강조합니다. 또한, 실제 환경에서 NeSy를 적용할 수 있는 가능성을 보여주기 위한 두 가지 개념 증명(proof-of-concept) 실험 결과를 제시합니다.

- **Performance Highlights**: 본 논문은 NeSy가 기존 AI 시스템이 겪는 한계점을 해결할 수 있는 잠재력이 있음을 보여줍니다. 특히, 실제 환경에서 발생하는 노이즈와 데이터 부족 문제를 효과적으로 해결할 수 있으며, 다양한 유형의 정보를 효율적으로 통합하고 추론하여 상황 인식을 향상시킬 수 있다는 장점을 제시합니다.

- **Challenges Faced**: 논문은 AI 기반 사이버 보안 시스템이 직면하는 주요 과제들을 분석합니다. 특히, 실제 환경에서 AI 모델의 정확도를 높이는 것, 작은 데이터셋으로 학습하는 것, 위협 정보를 모델에 효과적으로 통합하는 것 등을 주요 과제로 제시합니다.

- **Use Cases**: 논문은 NeSy가 사이버 보안에서 활용될 수 있는 구체적인 사용 사례들을 제시합니다. 특히, 탐지, 상황 인식, 위협 분석, 대응 등의 다양한 분야에서 NeSy가 기존의 AI 방식들을 뛰어넘는 성능을 제공할 수 있음을 보여줍니다.



### Knowledge Base Embeddings: Semantics and Theoretical Properties (https://arxiv.org/abs/2408.04913)
Comments:
          This is an extended version of a paper appearing at the 21st International Conference on Principles of Knowledge Representation and Reasoning (KR 2024). 17 pages

- **What's New**: 이 논문은 지식 그래프 임베딩(Knowledge Graph Embedding) 분야의 최근 발전인 지식 베이스 임베딩(Knowledge Base Embedding)에 대한 연구를 다룹니다. 지식 베이스 임베딩은 단순히 사실을 벡터 공간에 매핑하는 것뿐만 아니라 관련 개념적 지식을 고려하도록 모델을 제약하는 것을 목표로 합니다. 특히, 논문은 설명 논리(Description Logic)로 표현된 지식 베이스를 벡터 공간에 임베딩하는 최신 방법을 기하학적 기반 의미론(Geometric-based Semantics) 관점에서 조사합니다. 논문은 다양한 이론적 특성을 식별하고, 구체적인 임베딩 방법들이 이러한 이론적 틀에 어떻게 적합하는지 살펴봅니다.



### Unleashing Artificial Cognition: Integrating Multiple AI Systems (https://arxiv.org/abs/2408.04910)
- **What's New**: 본 연구에서는 인공지능의 인지 능력을 향상시키기 위해 언어 모델과 쿼리 분석 기법을 융합한 혁신적인 시스템을 소개합니다. 이 시스템은 체스 엔진과 언어 모델을 통합하여 체스 게임에서 수를 예측하고 전략적인 설명을 제공합니다. 또한, 검색 가능한 답변 생성을 통해 벡터 데이터베이스를 활용하여 OpenSI AI 시스템은 의사 결정 과정을 명확히 하여 원시 계산과 인간과 유사한 이해 간의 간극을 해소합니다. 이 연구는 체스를 시연 환경으로 선택하여 접근 방식의 다양성을 강조합니다. 체스 외에도 이 시스템은 의료 진단에서 금융 예측에 이르기까지 다양한 분야에 적용될 수 있는 가능성을 보여줍니다.



### Axiomatic Characterisations of Sample-based Explainers (https://arxiv.org/abs/2408.04903)
- **What's New**: 본 논문은 블랙박스 분류기의 의사 결정을 설명하는 기능 기반 설명기를 자세히 분석합니다. 설명기가 만족해야 하는 바람직한 속성을 제시하고, 이들 간의 관계를 살펴보며, 일부 속성의 비호환성을 강조합니다. 또한 두 가지 핵심 속성을 만족하는 설명기의 전체 계열을 식별하고, 이 계열의 인스턴스는 약한 귀납적 설명이라고 불리는 충분한 이유를 제공합니다. 이 논문은 호환 가능한 속성의 하위 집합을 만족하는 다양한 하위 계열을 밝혀냅니다. 특히, 설명의 존재와 글로벌 일관성을 보장하는 최초의 설명기(광범위한 계열)를 소개합니다. 또한 이 계열의 몇 가지 인스턴스를 논의하며, 여기에는 반박할 수 없는 설명기와 대리 모델을 기반으로 한 설명을 다항식 시간 내에 찾을 수 있는 대리 설명기가 포함됩니다.



### AI for operational methane emitter monitoring from spac (https://arxiv.org/abs/2408.04745)
- **What's New**: MARS-S2L, a 새로운 AI 기반 메탄 배출 모니터링 시스템이 개발되었으며, 이 시스템은 유엔 환경 프로그램의 국제 메탄 배출 관측소 (IMEO)에서 운영 중입니다. 이 시스템은 Sentinel-2 및 Landsat 위성 이미지를 사용하여 전 세계 메탄 배출 사건을 자동으로 감지하고 보고하는 기능을 갖추고 있습니다.



### Preserving Privacy in Large Language Models: A Survey on Current Threats and Solutions (https://arxiv.org/abs/2408.05212)
Comments:
          GitHub repository: this https URL

- **What's New**: 본 논문은 대규모 언어 모델(LLM)의 개인 정보 보호 위협을 포괄적으로 분석하고, 훈련 데이터, 모델 자체 및 추론 단계에서 개인 정보 보호를 강화하는 다양한 솔루션을 제시합니다. 특히, 훈련 데이터에서 개인 정보 식별 가능 정보(PII)가 유출될 위험을 강조하며, 기존의 개인 정보 보호 공격 연구를 검토하고, 훈련 데이터 익명화, 차등 프라이버시 적용 및 훈련 후 기계적 잊음(unlearning)과 같은 다양한 해결책을 제시합니다. 즉,  LLM의 훈련 및 배포 전반에 걸쳐 개인 정보 보호 메커니즘을 통합하는 방법을 제시합니다.



### VITA: Towards Open-Source Interactive Omni Multimodal LLM (https://arxiv.org/abs/2408.05211)
Comments:
          Project Page: this https URL

- **What's New**: VITA는 영상, 이미지, 텍스트, 오디오를 동시에 처리하고 분석할 수 있는 최초의 오픈소스 멀티모달 대규모 언어 모델(MLLM)입니다. VITA는 멀티모달 대화형 경험을 제공하는 기능을 갖추고 있으며, 영어와 중국어를 모두 지원합니다.



### TaSL: Task Skill Localization and Consolidation for Language Model Continual Learning (https://arxiv.org/abs/2408.05200)
Comments:
          Extension of ACL 2024 paper titled: Continual Dialog State Tracking via Task Skill Localization and Consolidation

- **What's New**: 본 논문에서는 언어 모델 지속 학습(CL, Continual Learning)을 위한 새로운 프레임워크인 'Task Skill Localization and Consolidation (TaSL)'을 제시합니다. TaSL은 기존 방식의 단점을 극복하고 지식 전이(KT, Knowledge Transfer)를 향상시키기 위해 다음과 같은 핵심 기능을 도입했습니다:

* **Skill Unit**: 모델을 기능 단위(Skill Unit)로 분할하여 더 세밀한 제어를 가능하게 합니다. 각 Skill Unit은 특정 작업에 관련된 지식을 담고 있습니다.
* **Group-wise Skill Localization**: 각 작업에 대한 Skill Unit의 중요성 분포를 파악하는 기법입니다. 이를 통해 작업 특정 지식과 공유 지식을 구분할 수 있습니다.
* **Skill Consolidation**: 작업 특정 지식은 보존하고, 공유 지식은 통합하는 전략을 통해 지식 전이를 촉진하고 망각을 방지합니다.

TaSL은 기존 지식을 유지하면서 새로운 작업에서 우수한 성능을 달성하는 데 효과적입니다. 또한, LoRA와 같은 PEFT 방법에 적용 가능하며 메모리 리플레이와의 통합을 통해 성능을 더욱 향상시킬 수 있습니다.



### HistoKernel: Whole Slide Image Level Maximum Mean Discrepancy Kernels for Pan-Cancer Predictive Modelling (https://arxiv.org/abs/2408.05195)
Comments:
          28 pages, 5 figures, 1 Table. Preprint for article in review at Nature Machine Intelligence

- **What's New**: HistoKernel, a novel Maximum Mean Discrepancy (MMD) kernel that measures distributional similarity between WSIs (Whole Slide Images) for enhanced prediction performance on downstream prediction tasks, is introduced.

- **Technical Details**: HistoKernel quantifies distributional differences or similarities between WSIs, facilitating downstream tasks like visualization, clustering, regression, classification, and survival analysis. It is unsupervised, requiring no target labels for computation, and seamlessly integrates multi-modal data.

- **Performance Highlights**: HistoKernel demonstrates effectiveness in various CPath tasks: 
- **WSI Retrieval:** Outperforms state-of-the-art RetCCL with a 12% higher mMV@5 (majority vote at top five search results) macro-average.
- **Drug Sensitivity Regression:**  Outperforms SlideGraph∞ with statistically significantly higher Spearman correlation coefficient for 94% of compounds. 
- **Point Mutation Classification:** Achieves impressive performance. 
- **Survival Analysis:** Effectively integrates multi-modal data for survival prediction.



### Meta-Learning Guided Label Noise Distillation for Robust Signal Modulation Classification (https://arxiv.org/abs/2408.05151)
Comments:
          8 pages, 7 figures

- **What's New**: 본 논문에서는 사물 인터넷 (IoT)의 물리적 계층 위협에 대응하는 효과적인 방법인 자동 변조 분류 (AMC)에 대한 새로운 연구 결과를 제시합니다. 특히, 실제 환경에서 자주 발생하는 레이블 오류 문제를 해결하기 위해, 메타 러닝 기반 레이블 노이즈 증류 (label noise distillation) 기법을 제안합니다. 이 방법은 딥 러닝 네트워크 (DNN)의 성능과 안정성을 향상시키는 데 중요한 역할을 합니다.



### AttackER: Towards Enhancing Cyber-Attack Attribution with a Named Entity Recognition Datas (https://arxiv.org/abs/2408.05149)
Comments:
          Submitted to WISE 2024

- **What's New**: 이 논문은 사이버 공격 속성(attribution)을 위한 새로운 데이터셋인 AttackER를 소개합니다. 기존의 사이버 보안 NER 데이터셋과 달리 AttackER는 다양한 컨텍스트를 포함하는 풍부한 주석을 제공하여 사이버 공격 속성 정보를 추출하는 데 도움이 됩니다.

- **Technical Details**: AttackER는 STIX 2.1 프레임워크를 기반으로 하며 18가지 유형의 엔티티를 식별합니다. 또한, 이 논문은 AttackER 데이터셋에서 NER 작업을 위해 Huggingface(HF) 및 spaCy와 같은 트랜스포머 기반 모델 학습과 GPT-3.5, Llama-2, Mistral-7B와 같은 미세 조정된 LLM(Large Language Models)을 사용했습니다.

- **Performance Highlights**: 실험 결과는 특히 LLM(Large Language Models)에서 AttackER의 효과성을 보여주었습니다. 특히, LLMs는 AttackER 데이터셋의 NER 작업에서 특정 프롬프트 템플릿을 사용한 지침 미세 조정을 통해 성능을 향상시켰습니다.

- **Availability**: AttackER 데이터셋 및 Huggingface 트랜스포머 기반 모델은 공개적으로 사용 가능합니다. 이 연구는 사이버 공격 속성을 위한 NER 작업에서 미세 조정된 LLMs의 사용을 탐구한 최초의 연구입니다.



### Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2 (https://arxiv.org/abs/2408.05147)
Comments:
          12 main text pages, and 14 pages of acknowledgements, references and appendices

- **What's New**: Gemma Scope, a comprehensive suite of sparse autoencoders (SAEs) trained on all layers and sub-layers of Gemma 2 2B, 9B, and 27B models, is released. This open-source resource aims to accelerate research in safety and interpretability of language models. It contains over 400 SAEs with over 30 million learned features, trained using a significant amount of computational resources.



### Cautious Calibration in Binary Classification (https://arxiv.org/abs/2408.05120)
Comments:
          Accepted to ECAI 2024

- **What's New**: 이 논문에서는 이진 분류에서 **신중한 보정 (Cautious Calibration)**이라는 새로운 개념을 소개합니다. 신중한 보정은 예측 확률에 대한 **낮은 신뢰도 (Underconfidence)**를 의도적으로 제공하여 예측 확률을 보정하는 방법입니다. 이는 특히 **높은 위험 상황 (High-risk Scenario)**에서 중요합니다. 높은 위험 상황에서 과도한 신뢰는 엄청난 비용으로 이어질 수 있기 때문에, **각 예측 확률 (Predicted Probability)**은 **과대 추정 (Overestimation)**보다는 **과소 추정 (Underestimation)**에 치우쳐야 합니다.



### Beyond the Eye: A Relational Model for Early Dementia Detection Using Retinal OCTA Images (https://arxiv.org/abs/2408.05117)
- **What's New**: 본 논문에서는 망막 OCTA 영상을 사용하여 조기 발병 알츠하이머병(EOAD) 및 경도 인지 장애(MCI)를 구별하는 혁신적인 폴라넷+ 모델을 소개합니다. 폴라넷+는 임상의에게 친숙한 ETDRS 그리드 분석을 구현하기 위해 OCTA 영상을 카르테시안 좌표에서 극좌표로 매핑하는 방식을 사용합니다. 또한, 이 논문은 다중 뷰 모듈을 도입하여 3차원을 따라 영상을 직렬화하고 분석하여 포괄적인 임상적으로 유용한 정보 추출을 가능하게 합니다. 마지막으로, 시퀀스 임베딩을 그래프로 추상화하여 탐지 작업을 일반적인 그래프 분류 문제로 변환합니다. 다중 뷰 모듈 이후에는 지역 관계 모듈을 적용하여 서브 영역 간의 관계를 파악합니다. 이러한 지역 관계 분석은 알려진 눈-뇌 연결을 검증하고 새로운 차별적 패턴을 밝혀냅니다.



### Semantic Successive Refinement: A Generative AI-aided Semantic Communication Framework (https://arxiv.org/abs/2408.05112)
- **What's New**: This paper introduces a novel **Generative AI Semantic Communication (GSC)** system for single-user and multi-user scenarios, leveraging deep generative models to enhance semantic communication performance, especially in low SNR environments.



### MooER: LLM-based Speech Recognition and Translation Models from Moore Threads (https://arxiv.org/abs/2408.05101)
- **What's New**: MooER, a novel LLM-based large-scale automatic speech recognition (ASR) / automatic speech translation (AST) model developed by Moore Threads, is presented. This model is trained on a 5000h pseudo-labeled dataset, demonstrating comparable performance to open source models trained on hundreds of thousands of hours of labeled data.  MooER outperforms other open source Speech LLMs, achieving a BLEU score of 25.2 on the Covost2 Zh2en testset.  It is noteworthy that MooER is the first speech large-scale model to utilize domestically produced GPUs for training and inference, showcasing its potential for industrial application.



### AI-driven Java Performance Testing: Balancing Result Quality with Testing Tim (https://arxiv.org/abs/2408.05100)
Comments:
          Accepted for publication in The 39th IEEE/ACM International Conference on Automated Software Engineering (ASE '24)

- **What's New**: This paper proposes an AI-based framework for dynamically stopping warm-up iterations in Java performance testing. The framework leverages Time Series Classification (TSC) models to predict the end of the warm-up phase during test execution, aiming to enhance the cost-effectiveness of testing.



### Overcoming the Limitations of Layer Synchronization in Spiking Neural Networks (https://arxiv.org/abs/2408.05098)
- **What's New**: 본 논문은 기존의 레이어 동기화 (layer synchronization) 방식 대신, 신경망에서 모든 뉴런이 비동기적으로 (asynchronous) 작동하는 '네트워크 비동기 (network asynchrony)'를 제안합니다. 이 방식은 뉴런이 레이어 간 동기화 없이 독립적으로 활동하고 신호를 전달할 수 있어, 기존 방식에 비해 지연 시간과 에너지 효율성을 향상시킬 수 있습니다. 하지만 이 새로운 방식은 기존의 레이어 동기화 기반으로 학습된 모델의 성능 저하를 야기할 수 있습니다. 이 문제를 해결하기 위해 논문은 비동기 처리에 적합한 모델 학습을 위한 새로운 역전파 기반 훈련 방법인 '비레이어 역전파 (unlayered backprop)'를 제시합니다.



### Hyperbolic Learning with Multimodal Large Language Models (https://arxiv.org/abs/2408.05097)
Comments:
          ECCV 2024 - Beyond Euclidean Workshop

- **What's New**: This paper introduces a novel large-scale hyperbolic vision-language model (VLM) based on the BLIP-2 architecture, which significantly expands the application of hyperbolic embeddings in VLMs.  This model leverages the hierarchical properties of hyperbolic space to enhance representational power and improve reasoning capabilities.

- **Technical Details**: The paper proposes a stable hyperbolic training method for BLIP-2.  This method overcomes the challenges of scaling multi-modal hyperbolic models by orders of magnitude and allows the model to achieve performance comparable to its Euclidean counterpart, while maintaining stability during training.

- **Performance Highlights**: The hyperbolic BLIP-2 model shows comparable performance to the Euclidean baseline on retrieval tasks. The hyperbolic embeddings also provide a measure of uncertainty through their radii, offering insights into the model's confidence levels.

- **Contributions**: This paper presents the first large-scale hyperbolic VLM.  It demonstrates that hyperbolic embeddings can capture uncertainty information and achieve performance comparable to Euclidean baselines. The paper provides detailed analyses of the different embedding spaces, shedding light on the operation of current VLMs and effective representation in hyperbolic space.



### Order Matters in Hallucination: Reasoning Order as Benchmark and Reflexive Prompting for Large-Language-Models (https://arxiv.org/abs/2408.05093)
Comments:
          7 pages, submitted to AAAI25

- **What's New**: 이 논문은 대규모 언어 모델(LLM)의 일관성을 평가하기 위한 새로운 벤치마크 방법인 **Reasoning Order as Benchmark**를 제시합니다. 이 벤치마크는 LLM이 먼저 답변을 생성한 다음 추론 과정을 제공하는 방식과 먼저 추론 과정을 생성한 다음 결론을 제공하는 방식의 차이를 비교하여 모델의 추론 논리의 일관성을 반영합니다. 또한, 이러한 문제를 완화하기 위해 **Reflexive Prompting**이라는 새로운 프롬프트 전략을 소개합니다. Reflexive Prompting은 먼저 답변 우선 프롬프트와 논리 우선 프롬프트를 사용하여 두 개의 잠재적으로 다른 답변을 얻습니다. 그런 다음 두 답변을 분석하여 최종 답변을 도출합니다. 실험 결과는 이 방법이 다양한 데이터 세트와 LLM 아키텍처에서 추론 작업의 정확도를 향상시키는 것을 보여줍니다. 또한, 다른 프롬프트 전략을 사용하여 달성한 정확도는 Reasoning Order as Benchmark 테스트 결과와 강한 상관 관계가 있어 이 벤치마크 방법의 유용성을 입증합니다.



### Generating novel experimental hypotheses from language models: A case study on cross-dative generalization (https://arxiv.org/abs/2408.05086)
- **What's New**: This research uses neural network language models (LMs) as simulated learners to explore hypotheses about how children learn dative constructions in English. This approach provides a way to systematically investigate the vast space of possible cues that children could use to learn which verbs can be used in different dative structures.  Specifically, it aims to derive novel hypotheses from LMs that can be tested with real children.



### Generalizing Few Data to Unseen Domains Flexibly Based on Label Smoothing Integrated with Distributionally Robust Optimization (https://arxiv.org/abs/2408.05082)
- **What's New**: 본 논문은 기존의 레이블 스무딩(Label Smoothing, LS) 기술을 분포적으로 강건한 최적화(Distributionally Robust Optimization, DRO) 프레임워크에 통합하여 'DRO-LS' 라는 새로운 이단계 문제를 제안합니다.  DRO-LS는 LS를 이용하여 기존 데이터를 변형하여 새로운 샘플을 생성하고, 이후 생성된 샘플과 기존 샘플을 사용하여 DNN을 훈련합니다.  DRO-LS는 기존 데이터를 다양한 시나리오에 따라 변형하여 실제 데이터 분포를 벗어난 영역까지 확장할 수 있습니다.

- **Technical Details**: LS는 원-핫 인코딩된 레이블을 균일한 레이블 벡터와 혼합하여 과적합을 방지하는 효과적인 정규화 기법입니다.  하지만 LS는 기존 데이터의 분포를 고려하지 않고 레이블에만 집중합니다.  본 논문에서는 DRO를 LS에 통합하여 DNN 훈련 시 기존 데이터 분포를 새로운 도메인으로 유연하게 이동시키는 방법을 제시합니다.  특히 DRO와 LS를 통합하면 LS의 정규화 효과를 DNN 파라미터에 대한 정규화 항으로 확장할 수 있음을 증명했습니다.  이 정규화 항을 이용하여 기존 데이터를 새로운 도메인으로 이동시키고 새로운 데이터를 생성할 수 있습니다.  또한 이러한 발견을 구현하고 DNN을 훈련하기 위해 근사적 경사 하강 반복 레이블 스무딩 알고리즘(GI-LS)를 제안했습니다.  GI-LS는 여러 하이퍼파라미터를 포함하기 때문에 베이지안 최적화(Bayesian Optimization, BO)를 사용하여 상대적으로 최적의 하이퍼파라미터 조합을 찾는 방법을 고려했습니다.

- **Performance Highlights**: 본 논문은 소규모 이상 감지 분류 작업을 사례로 사용하여 GI-LS를 평가했습니다.  결과는 GI-LS가 뛰어난 성능을 보여줌을 명확히 증명합니다.



### RT-Surv: Improving Mortality Prediction After Radiotherapy with Large Language Model Structuring of Large-Scale Unstructured Electronic Health Records (https://arxiv.org/abs/2408.05074)
Comments:
          23 pages, 2 tables, 4 figures

- **What's New**: 본 연구는 방사선 치료(RT)의 효과적인 환자 선별을 위한 새로운 접근 방식을 제시합니다. 기존의 생존 예측 모델은 구조화된 데이터에 의존하여 정확성이 제한적이었습니다. 이 연구는 대규모 언어 모델(LLM)을 활용하여 비구조화된 전자 건강 기록(EHR) 데이터를 구조화하고, 포괄적인 임상 정보 통합을 통해 생존 예측 정확도를 향상시키는 가능성을 탐구합니다.



### A Jailbroken GenAI Model Can Cause Substantial Harm: GenAI-powered Applications are Vulnerable to PromptWares (https://arxiv.org/abs/2408.05061)
Comments:
          Website, see this https URL

- **What's New**: 이 논문은 PromptWare라는 새로운 유형의 공격을 소개하며, 이 공격은 GenAI(Generative AI) 모델의 동작 방식을 응용 프로그램을 지원하는 것에서 공격하는 것으로 바꿉니다. PromptWare는 사용자 입력을 이용하여 GenAI 모델을 탈옥시키고, GenAI 기반 응용 프로그램의 맥락 내에서 악의적인 활동을 강제로 수행합니다.



### GLEAMS: Bridging the Gap Between Local and Global Explanations (https://arxiv.org/abs/2408.05060)
- **What's New**: GLEAMS (Global & Local ExplainAbility of black-box Models through Space partitioning) is a novel post-hoc interpretability method that provides both local and global explanations for black-box models. GLEAMS aims to create a global surrogate model that captures the overall shape of the original black-box model while maintaining a simple local structure. This allows for simultaneous extraction of local explanations for any instance and global feature importance.



### SELD-Mamba: Selective State-Space Model for Sound Event Localization and Detection with Source Distance Estimation (https://arxiv.org/abs/2408.05057)
- **What's New**: SELD-Mamba는 효율적인 음향 이벤트 위치 및 검출(SELD) 모델로서, 선형 복잡도를 유지하면서 다양한 맥락 정보를 포착하는 Mamba(선택적 상태 공간 모델)를 사용합니다. SELD-Mamba는 EINV2(Event-Independent Network V2)를 기반으로 하며, Conformer 블록을 양방향 Mamba 블록으로 대체하여 성능을 향상시킵니다. 또한, SELD-Mamba는 음향 이벤트 검출(SED) 및 도착 방향(DoA) 추정 손실에 중점을 둔 1단계 훈련과 SDE(Source Distance Estimation) 손실을 재도입하는 2단계 훈련을 통해 더욱 개선된 성능을 달성합니다.

- **Technical Details**: SELD-Mamba는 EINV2 아키텍처를 기반으로 하며, Conformer 블록을 BMamba(Bidirectional Mamba) 블록으로 대체하여 선형 복잡도를 유지하면서 맥락 정보를 효과적으로 포착합니다. Mamba는 SSM(State Space Model)의 변형으로서 입력에 따라 동적으로 정보를 선택적으로 처리하는 기능을 제공합니다. SELD-Mamba는 2단계 훈련 방식을 사용합니다. 1단계에서는 SED 및 DoA 추정 작업에 집중하여 손실을 최소화하고, 2단계에서는 SDE 손실을 재도입하여 모든 작업의 성능을 최적화합니다.

- **Performance Highlights**: 2024년 DCASE Challenge Task3 데이터셋에서 SELD-Mamba는 EINV2보다 우수한 성능을 달성했습니다. SELD-Mamba는 적은 파라미터와 계산 복잡도로도 높은 성능을 보여주며, SSM을 SELD에 적용한 최초의 연구입니다.



### A GNN Model with Adaptive Weights for Session-Based Recommendation Systems (https://arxiv.org/abs/2408.05051)
Comments:
          7 pages, 7 tables, 2 figures, and 3 equations

- **What's New**: This paper proposes a novel approach to enhance session-based recommendations (SBRs) by introducing an adaptive weighting mechanism to the SR-GNN model. This mechanism assigns varying importance to items within a session based on side information, leading to more accurate predictions and addressing the cold start problem in SBRs.



### Rag and Roll: An End-to-End Evaluation of Indirect Prompt Manipulations in LLM-based Application Frameworks (https://arxiv.org/abs/2408.05025)
- **What's New**: This research explores the security vulnerabilities of Retrieval Augmented Generation (RAG) systems, a technique that enhances LLMs by retrieving external knowledge to generate responses. It focuses on indirect prompt manipulation attacks, where malicious documents are injected into the knowledge base to manipulate the model's output.



### Enhancing the Code Debugging Ability of LLMs via Communicative Agent Based Data Refinemen (https://arxiv.org/abs/2408.05006)
- **What's New**: This paper introduces DebugEval, a comprehensive benchmark for evaluating Large Language Models (LLMs) in debugging tasks, surpassing existing benchmarks by incorporating four tasks: BUG Localization, BUG Identification, Code Review, and Code Repair. Additionally, it proposes MASTER, a communicative agent-based data refinement framework that generates high-quality code debugging data for supervised fine-tuning. MASTER utilizes a Code Quizzer to generate diverse problems, a Code Learner to evaluate their value, and a Code Teacher to provide detailed solutions for incorrect problems. This framework aims to enhance the debugging ability of LLMs by providing them with refined training data.



### ProFuser: Progressive Fusion of Large Language Models (https://arxiv.org/abs/2408.04998)
- **What's New**: 본 논문에서는 여러 대규모 언어 모델(LLM)의 장점을 결합하여 더욱 강력하고 다재다능한 모델을 구축하는 새로운 방법을 제시합니다. 기존의 모델 융합 방법들은 주로 teacher-forcing 설정에서 지도 학습(ground truth)에 대한 교차 엔트로피(cross entropy)를 사용하여 모델의 장점을 평가하는 데 중점을 두었습니다. 그러나 이러한 방법은 모델의 장점에 대한 제한적인 통찰력만을 제공할 수 있습니다. 본 논문에서는 훈련 모드와 추론 모드를 모두 포함하는 새로운 방법을 제시하여 모델 융합 프로세스를 개선합니다. 제안된 방법은 훈련 중에 교차 엔트로피를 통해 모델의 장점을 평가할 뿐만 아니라 추론 출력을 고려하여 보다 포괄적인 평가를 제공합니다. 두 모드를 효과적으로 결합하기 위해 본 논문에서는 추론 모드에서 훈련 모드로 점진적으로 전환하는 ProFuser를 도입합니다. ProFuser의 효과를 검증하기 위해 vicuna-7b-v1.5, Llama-2-7b-chat, mpt-7b-8k-chat을 포함한 세 가지 모델을 융합하고 기준 방법과 비교하여 지식, 추론 및 안전성 측면에서 향상된 성능을 보여줍니다.



### LLaVA-VSD: Large Language-and-Vision Assistant for Visual Spatial Description (https://arxiv.org/abs/2408.04957)
- **What's New**: LLaVA-VSD는 이미지 내 객체 간 공간 관계를 설명하는 텍스트를 생성하는 새로운 모델입니다. 기존 VSRC (Visual Spatial Relationship Classification) 방법은 이미지에서 두 객체 간의 공간 관계만 출력하는데, LLaVA-VSD는 세계 지식 (world knowledge)을 활용하고 일반적인 언어 능력을 갖추고 있습니다. 이를 통해, LLaVA-VSD는 이미지에서 객체 간 관계에 대한 질문에 답변하는 등 다중 모드 대화 능력을 선보입니다.



### CROCODILE: Causality aids RObustness via COntrastive DIsentangled LEarning (https://arxiv.org/abs/2408.04949)
Comments:
          MICCAI 2024 UNSURE Workshop, Accepted for presentation, Submitted Manuscript Version, 10 pages

- **What's New**: 이 논문에서는 CROCODILE 프레임워크를 소개하며, 인과관계 분석 도구를 사용하여 특징 분리 (feature disentanglement), 대조 학습 손실 (contrastive learning losses), 그리고 사전 지식 주입을 통해 도메인 이동 (domain shift)에 대한 모델의 견고성을 향상시키는 방법을 보여줍니다. 이를 통해 모델은 가짜 상관관계 (spurious correlations)에 대한 의존도를 줄이고 이미지에서 예측으로 이어지는 메커니즘을 더 잘 학습하며, 분포 외 (out-of-distribution, OOD) 데이터에서 기준 모델 성능을 능가합니다.



### UAV-Enhanced Combination to Application: Comprehensive Analysis and Benchmarking of a Human Detection Dataset for Disaster Scenarios (https://arxiv.org/abs/2408.04922)
Comments:
          This Paper is accepted for 27th International Conference on Pattern Recognition (ICPR 2024)

- **What's New**: 본 논문에서는 드론을 이용한 재난 구조 작업에서 인간 탐지를 위한 새로운 데이터셋인 'C2A' 데이터셋을 소개합니다. 기존 데이터셋의 부족함을 해결하기 위해 실제 재난 현장 사진에 다양한 인간 자세를 합성하여 만들어졌습니다.

- **Technical Details**: C2A 데이터셋은 실제 드론으로 촬영된 재난 현장 이미지에 인간 자세 이미지를 합성하여 생성됩니다. 이를 통해 다양한 재난 상황에서 인간 탐지 모델 학습을 가능하게 합니다. 또한, 폐쇄된 인간, 즉 부분적으로 가려진 인간의 탐지를 위해 데이터셋에 가려짐 요소를 추가하여 모델 학습의 난이도를 높였습니다. C2A 데이터셋은 다양한 인간 자세와 재난 현장 정보를 포함하여 재난 상황의 심각성을 평가하는 데 도움이 됩니다.

- **Performance Highlights**: 실험 결과 C2A 데이터셋으로 미세 조정된 모델은 일반적인 항공 데이터셋으로 사전 훈련된 모델보다 뛰어난 성능을 보였습니다. 또한, C2A 데이터셋을 일반적인 인간 데이터셋과 결합하면 다양한 시나리오에서 최적의 성능과 일반화 성능을 달성할 수 있음을 보여줍니다.

- **Contributions**: 본 연구의 기여는 다음과 같습니다.
- 드론을 이용한 재난 구조 작업에서 인간 탐지 모델 학습을 위한 새로운 데이터셋 C2A를 소개합니다.
- 다양한 재난 상황을 반영한 C2A 데이터셋 생성 파이프라인을 개발합니다.
- C2A 데이터셋은 다양한 인간 자세와 재난 현장 정보를 포함하여 재난 상황의 심각성을 평가하는 데 도움이 됩니다.
- 실험 결과 C2A 데이터셋은 인간 탐지 모델의 성능을 향상시키고 운영상의 실현 가능성을 높이는 데 기여합니다.

- **Limitations**: 본 연구에서 생성된 데이터셋은 실제 재난 현장과 완벽히 동일하지 않다는 점에 유의해야 합니다.

- **Future Directions**: 본 연구는 앞으로 드론을 이용한 재난 구조 작업에서 인간 탐지 모델의 성능을 더욱 향상시키기 위해 노력할 것입니다. 특히, 실제 재난 현장에서 수집된 데이터를 활용하여 더욱 현실적인 데이터셋을 생성하고 모델 학습을 위한 다양한 방법을 연구할 것입니다.



### Avoid Wasted Annotation Costs in Open-set Active Learning with Pre-trained Vision-Language Mod (https://arxiv.org/abs/2408.04917)
- **What's New**: 이 논문은 기존의 액티브 러닝(AL) 방법론이 오픈셋 데이터에서 겪는 한계점을 해결하기 위해 CLIP 기반의 새로운 데이터 선택 전략인 CLIPNAL을 제안합니다. CLIPNAL은 오픈셋 데이터에서 불필요한 주석 비용을 최소화하면서 모델 성능을 향상시키는 데 중점을 둡니다.



### Towards a Generative Approach for Emotion Detection and Reasoning (https://arxiv.org/abs/2408.04906)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)을 이용하여 텍스트에서 감정을 감지하고 감정적 추론을 수행하는 새로운 제너레이티브 접근 방식을 소개합니다. 기존의 제로샷 감정 감지 방법은 텍스트 함축 모델에 의존하여 입력 텍스트에 가장 적합한 감정 레이블을 선택하는 반면, 이 논문은 감정 분석을 제너레이티브 질문 답변(QA) 작업으로 재구성하는 접근 방식을 제시합니다. 이 접근 방식은 감정 감지 질문에 단계별로 답변할 수 있는 관련 맥락 또는 배경 지식을 생성하는 두 단계 방법론을 사용합니다. 이 논문은 텍스트에 대한 감정 감지와 감정적 추론 작업을 함께 해결하기 위해 제너레이티브 접근 방식을 사용하는 최초의 연구입니다.



### GlitchProber: Advancing Effective Detection and Mitigation of Glitch Tokens in Large Language Models (https://arxiv.org/abs/2408.04905)
- **What's New**: 본 논문에서는 '글리치 토큰'(glitch token)이라 불리는 특수 토큰의 영향을 조사하고, 탐지 및 완화 기술을 제시합니다. 글리치 토큰은 LLM(대규모 언어 모델)의 어휘 공간에서 발견되는 비정상적인 토큰으로, 모델의 입력에 포함될 경우 잘못되거나 무관한 결과, 또는 심지어 유해한 결과를 초래할 수 있습니다. 본 연구에서는 LLM의 내부 메커니즘에 미치는 글리치 토큰의 영향에 대한 이해를 심화시키고, 효과적인 탐지 및 완화 기법을 제안합니다.

- **Technical Details**: 본 연구는 LLM의 중간층에서 나타나는 주의 패턴(attention patterns)과 동적 정보(dynamic information) 분포의 차이를 통해 글리치 토큰과 정상 토큰을 구별합니다. 이러한 분석을 기반으로, GlitchProber라는 새로운 도구를 개발했습니다. GlitchProber는 소규모 샘플링, 주성분 분석(PCA: Principal Component Analysis) 기반의 특징 추출, 그리고 간단한 분류기를 사용하여 효율적으로 글리치 토큰을 탐지하고 완화합니다. GlitchProber는 또한 비정상적인 모델 중간층 값을 수정하여 글리치 토큰의 부정적 영향을 완화합니다.

- **Performance Highlights**: GlitchProber는 5개의 주요 오픈소스 LLM을 대상으로 평가한 결과, 기존 방법에 비해 더 높은 효율성, 정확도, 재현율을 보였습니다. GlitchProber는 평균 F1 점수 0.86과 평균 복구율 50.06%를 기록했습니다. GlitchProber는 글리치 토큰의 문제를 해결하는 새로운 방식을 제시하며, 더욱 강력하고 해석 가능한 LLM에 대한 향후 연구에 영감을 줄 수 있습니다.



### Better Not to Propagate: Understanding Edge Uncertainty and Over-smoothing in Signed Graph Neural Networks (https://arxiv.org/abs/2408.04895)
- **What's New**: 기존 Graph Neural Network (GNN)은 네트워크의 동종성 (homophily)에 의존하여 실제 세계의 이종성 (heterophily) 시나리오에서 과도한 평활화 (over-smoothing)로 인해 성능 저하가 발생할 수 있습니다. 본 논문에서는 노드 분리 (separability) 및 에지 오류 비율 (edge error ratio)을 추정하는 새로운 방법을 제안하고, 훈련 중에 블록형 (blocked) 전파와 서명형 (signed) 전파 사이를 동적으로 선택합니다. 이 연구는 블록형 전파가 높은 에지 오류 비율에서 서명형 전파보다 더 효과적일 수 있음을 입증합니다. 이는 동종성 및 이종성 그래프 모두에서 성능을 향상시킵니다.



### ConfusedPilot: Compromising Enterprise Information Integrity and Confidentiality with Copilot for Microsoft 365 (https://arxiv.org/abs/2408.04870)
- **What's New**: 이 논문은 RAG(Retrieval Augmented Generation) 시스템에서 발견된 새로운 보안 취약점인 ConfusedPilot에 대해 소개합니다. ConfusedPilot는 RAG 기반 시스템, 특히 Microsoft 365의 Copilot에서 발생할 수 있는 정보 무결성 및 기밀성 위반을 야기하는 취약점입니다. 이 취약점은 공격자가 악의적인 문서를 공유하여 Copilot을 속여 잘못된 정보를 제공하도록 유도하여 기업의 의사 결정 과정을 방해할 수 있습니다.



### Ensemble BERT: A student social network text sentiment classification model based on ensemble learning and BERT architectur (https://arxiv.org/abs/2408.04849)
- **What's New**: 본 논문은 중학생의 정신 건강 평가를 위해 BERT 기반의 앙상블 학습 네트워크를 새롭게 제안합니다. 이 네트워크는 다수의 분류기를 통합하여 모델 성능을 향상시키는 개념을 활용합니다. 다양한 BERT 기반 학습기를 훈련시킨 후, 다수결 투표 방식을 사용하여 이들을 결합했습니다. 중국 Weibo에서 중학생들의 소셜 네트워크 텍스트 데이터를 수집하여 중학생 소셜 네트워크 텍스트의 감정 경향 분류 작업에 적용했습니다. 



### UGrid: An Efficient-And-Rigorous Neural Multigrid Solver for Linear PDEs (https://arxiv.org/abs/2408.04846)
Comments:
          Proceedings of the 41st International Conference on Machine Learning, Vienna, Austria. PMLR 235, 2024

- **What's New**: 본 논문은 선형 편미분 방정식 (PDEs)을 위한 수학적으로 엄격한 신경망 솔버인 UGrid를 제시합니다. UGrid는 U-Net과 멀티그리드 (MultiGrid)를 통합하여 수렴성 및 정확성을 수학적으로 증명하고 다양한 입력 기하/값과 여러 PDE 공식에 대한 높은 수치 정확도와 강력한 일반화 능력을 보여줍니다. 또한 기존 손실 함수보다 안정성과 더 큰 해 공간을 제공하는 새로운 잔차 손실 지표를 고안하여 비지도 학습을 가능하게 합니다.

- **Technical Details**: UGrid는 멀티그리드 V-사이클 (V-cycle)의 구조를 기반으로 하며 멀티그리드 솔버의 기능을 학습합니다. 컨볼루션 연산자를 개선하여 임의의 경계 조건과 여러 미분 스텐실 (differential stencil)을 통합합니다. 또한 반복 업데이트 규칙과 멀티그리드 V-사이클을 CNN (Convolutional Neural Network) 구조로 변환합니다.

- **Performance Highlights**: UGrid는 다양한 선형 PDE에 대한 멀티그리드 연산자를 학습하고 훈련 단계에서 볼 수 없는 복잡한 기하학과 위상의 임의 경계 조건을 가진 PDE를 수치적으로 해결할 수 있음을 실험적으로 보여줍니다. 광범위한 실험과 종합적인 평가를 통해 주장된 장점을 모두 검증하고 제안된 방법이 최첨단 (SOTA) 성능을 능가한다는 것을 확인했습니다.



### Counterfactual Explanations with Probabilistic Guarantees on their Robustness to Model Chang (https://arxiv.org/abs/2408.04842)
- **What's New**: This paper proposes a novel method called **BetaRCE** for generating **counterfactual explanations (CFEs)** that are robust to model changes. BetaRCE is a **post-hoc method** that can be applied to any existing CFE generation method to improve its robustness. It provides **probabilistic guarantees** on the CFE's validity, allowing users to specify the desired level of robustness.



### Kolmogorov-Arnold Network for Online Reinforcement Learning (https://arxiv.org/abs/2408.04841)
Comments:
          Paper accepted at 24th International Conference on Control, Automation and Systems

- **What's New**: 이 논문은 강화 학습(Reinforcement Learning, RL) 알고리즘에서 콜모고로프-아놀드 네트워크(Kolmogorov-Arnold Networks, KANs)를 함수 근사기(function approximator)로 사용하는 방법을 제안합니다. 특히, 근사 정책 최적화(Proximal Policy Optimization, PPO) 알고리즘에 KANs를 적용하여 MLP 기반 PPO와 성능을 비교합니다. 이는 기존의 다층 퍼셉트론(Multi-Layer Perceptrons, MLPs) 대신 KANs를 사용하여 강화 학습 모델의 효율성을 높일 수 있는 가능성을 보여줍니다.



### mPLUG-Owl3: Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models (https://arxiv.org/abs/2408.04840)
- **What's New**: mPLUG-Owl3는 긴 이미지 시퀀스를 이해하는 데 뛰어난 성능을 보여주는 새로운 다중 모달 대규모 언어 모델(MLLM)입니다. 기존 모델과 달리, mPLUG-Owl3는 검색된 이미지-텍스트 지식, 혼합된 이미지-텍스트, 그리고 긴 비디오를 포함하는 시나리오에서 긴 이미지 시퀀스 이해를 향상시킵니다. 특히, mPLUG-Owl3는 긴 다중 이미지 시나리오를 처리하기 위해 비전과 언어를 공통의 언어 기반 의미 공간에 효율적으로 통합하는 새로운 하이퍼 어텐션 블록을 제안합니다.



### Self-augmented Gaussian Splatting with Structure-aware Masks for Sparse-view 3D Reconstruction (https://arxiv.org/abs/2408.04831)
- **What's New**: 이 논문에서는 희소 뷰 3D 재구성을 위한 자체 증강된 거칠기에서 미세한 가우시안 스플래팅(Gaussian Splatting) 프레임워크를 제안합니다. 이 프레임워크는 구조 인식 마스크(structure-aware mask)를 사용하여 구조 정보를 활용하고 3D 기하 증강과 지각적 뷰 증강을 통해 더 나은 재구성 결과를 얻습니다.

- **Technical Details**: 제안된 방법은 희소 뷰 입력으로부터 기본적인 3D 표현을 얻기 위해 거친 가우시안 모델을 사용하는 것으로 시작합니다. 그런 다음, 미세한 가우시안 네트워크를 개발하여 3D 기하 증강과 지각적 뷰 증강을 통해 출력의 일관성 있고 상세한 표현을 향상시킵니다. 훈련 중에는 구조 인식 마스킹 전략을 설계하여 희소 입력 및 노이즈에 대한 모델의 견고성을 더욱 향상시킵니다.

- **Performance Highlights**: MipNeRF360 및 OmniObject3D 데이터 세트에 대한 실험 결과는 제안된 방법이 희소 입력 뷰에 대해 지각적 품질과 효율성 모두에서 최첨단 성능을 달성함을 보여줍니다.



### Performance Prediction of Hub-Based Swarms (https://arxiv.org/abs/2408.04822)
- **What's New**: 본 논문에서는 벌집 구조의 집단 (hub-based colony)에서 일어나는 복잡한 의사 결정 과정을 이해하기 위해 그래프 기반 표현과 그래프 인코더를 사용하여 집단 상태의 저차원 표현을 생성하는 새로운 방법을 제시합니다. 특히, 많은 에이전트가 존재하는 시나리오에서도 효과적으로 사용 가능한 방법을 제시합니다.

- **Technical Details**: 본 논문은 집단 상태를 그래프의 노드로 표현하고, 집단 상태의 변화를 확률적 전이로 표현하여 마르코프 체인을 형성합니다. 이를 통해 집단 성능과 다른 특징을 예측할 수 있습니다. 하지만, 에이전트 수가 증가하면 노드, 노드 특징, 에지의 수가 빠르게 증가하여 계산 복잡성이 커지는 문제가 있습니다. 이 논문에서는 저차원 그래프 임베딩을 사용하여 이 문제를 해결합니다. 저차원 임베딩은 그래프 인코더를 사용하여 생성되며, 집단 행동에 대한 통찰력을 제공합니다.

- **Performance Highlights**: 본 논문에서는 저차원 임베딩을 활용하여 두 가지 실험을 통해 그 효용성을 입증했습니다. 첫째, 매우 작은 문제에서 최적의 위치를 선택할 확률에 따라 집단 상태를 군집화하는 데 사용할 수 있음을 보여줍니다. 둘째, 그래프 인코더를 사용하여 저차원 임베딩을 학습했을 때 구조화된 집단 궤적이 나타나고, 이러한 궤적은 군집 성능을 예측하는 데 사용할 수 있는 정보를 포함하고 있음을 보여줍니다.



### Natural Language Outlines for Code: Literate Programming in the LLM Era (https://arxiv.org/abs/2408.04820)
- **What's New**: 이 논문은 소프트웨어 개발 과정 전반에 걸쳐 개발자에게 AI 지원을 제공하기 위한 새로운 방식 및 인터페이스로서 자연어 개요(NL outline)를 사용하는 것을 제안합니다. 코드 함수의 NL 개요는 간결한 산문으로 작성된 여러 문장으로 구성되며, 코드를 분할하고 리터럴 프로그래밍 스타일로 주요 아이디어를 요약합니다. 중요한 것은, 최신 LLMs가 실제로 정확하고 고품질의 NL 개요를 생성할 수 있다는 점입니다. 또한 NL 개요는 코드와 NL 간의 양방향 동기화를 가능하게 하여 한쪽의 변경 사항이 다른 쪽에 자동으로 반영됩니다. NL 개요의 다양한 사용 사례를 논의합니다. NL 개요는 코드와 차이의 이해 및 탐색 속도를 높이고, 코드 유지 관리를 간소화하고, 코드 검색을 강화하고, 코드 생성을 안내하고, 그 외에도 여러 가지 역할을 수행할 수 있습니다. 그런 다음 NL 개요 생성을 위한 여러 LLM 프롬프팅 기법을 제안하고 비교하여 전문 개발자가 개요 품질을 평가하도록 합니다. 마지막으로 코드 검토 및 악성 코드 탐지라는 어려운 작업을 향한 NL 개요 적용에 대한 두 가지 사례 연구를 제시합니다.



### Performance Metric for Multiple Anomaly Score Distributions with Discrete Severity Levels (https://arxiv.org/abs/2408.04817)
Comments:
          accepted as a work-in-progress paper at the 2024 Annual Conference of the IEEE Industrial Electronics Society (IECON)

- **What's New**: 스마트 공장의 부상으로 자동화된 유지보수의 필요성이 높아졌으며, 이상 데이터가 부족한 환경에서 정상 데이터 기반 이상 감지가 특히 효과적임이 입증되었습니다. 훈련 중 이상 데이터를 필요로 하지 않는 이 방법은 연구자들이 이상 감지뿐만 아니라 이상 점수를 사용하여 심각도 수준을 분류하는 데 중점을 두도록 했습니다. 그러나 수신기 조작 특성 곡선 아래 영역(AUROC)과 같은 기존 성능 지표는 이상 점수를 기반으로 심각도 수준을 분류하는 모델의 성능을 효과적으로 반영하지 못합니다. 이러한 제한 사항을 해결하기 위해, 우리는 심각도 수준 차이의 패널티를 AUROC와 결합한 수신기 조작 특성 곡선 아래 영역의 가중 합계(WS-AUROC)를 제안합니다. 다양한 패널티 할당 방법을 사용하여 다양한 실험을 수행했습니다: 심각도 수준 차이에 관계없이 균일한 패널티, 심각도 수준 인덱스 차이를 기반으로 한 패널티, 이상을 유발하는 실제 물리적 수량을 기반으로 한 패널티. 후자의 방법이 가장 민감했습니다. 또한 분포의 명확한 분리를 달성하고 WS-AUROC 및 AUROC 지표에서 압축 모델을 능가하는 이상 감지기를 제안합니다.



### A Collaborative PIM Computing Optimization Framework for Multi-Tenant DNN (https://arxiv.org/abs/2408.04812)
- **What's New**: 본 논문에서는 여러 텐넌트(tenant)의 DNN을 ReRAM 기반 PIM (Processing-in-Memory) 설계에 효율적으로 배포하기 위한 새로운 프레임워크를 제안합니다. 제안된 프레임워크는 텐넌트 수준에서 PIM 하드웨어를 반복적으로 분할하여 리소스 경쟁 문제를 해결하고, 영역 집약적(area-intensive) 연산자를 처리하기 위해 연산자 수준에서 세분화된 재구성된 처리 파이프라인을 구축합니다.



### h4rm3l: A Dynamic Benchmark of Composable Jailbreak Attacks for LLM Safety Assessmen (https://arxiv.org/abs/2408.04811)
- **What's New**: 이 논문은 컴퓨터 프로그램 합성을 이용하여 다양한 새로운 제한 해제 공격을 생성하는 새로운 벤치마크(benchmark)를 제안합니다. 이 벤치마크는 정적 데이터셋이 아닌 동적 데이터셋을 사용하기 때문에 끊임없이 진화하고 있는 제한 해제 공격에 대해 더 효과적으로 대처할 수 있습니다. 또한, 이 논문은 이러한 공격을 공식적으로 표현할 수 있는 도메인 특정 언어(DSL)를 개발하였습니다. 이 DSL은 제한 해제 공격을 파라미터화된 문자열 변환 함수(string transformation function)의 조합으로 표현합니다.



### UniBench: Visual Reasoning Requires Rethinking Vision-Language Beyond Scaling (https://arxiv.org/abs/2408.04810)
- **What's New**: UniBench는 50개 이상의 비전-언어 모델(VLM) 벤치마크를 통합한 구현체입니다. 벤치마크는 개체 인식, 공간 인식, 계산 등 다양한 기능을 망라합니다. UniBench는 VLM의 진행 상황을 체계적으로 평가하기 위한 도구를 제공합니다.



### On the Geometry of Deep Learning (https://arxiv.org/abs/2408.04809)
- **What's New**: 본 논문은 딥 러닝의 수학적 기초에서 진전을 이룰 수 있는 유망한 분야 중 하나인 딥 네트워크와 아핀 스플라인(affine spline)(다차원 연속적 조각 선형 함수)에 의한 함수 근사(function approximation) 간의 연결성을 살펴봅니다. 특히, 딥 네트워크의 아핀 스플라인 매핑(affine spline mapping)의 기하학적 특성(geometrical properties)을 이해하는 데 초점을 맞추어 지난 10년간의 연구를 개괄적으로 살펴봅니다. 특히, 입력 공간을 어떻게 테셀레이션(tessellation)(분할)하는지에 대한 연구를 살펴봅니다. 아핀 스플라인 연결과 기하학적 관점은 딥 네트워크의 내부 작동 방식을 이해하고 분석하며 개선하는 강력한 도구를 제공합니다.



### AI and Machine Learning Driven Indoor Localization and Navigation with Mobile Embedded Systems (https://arxiv.org/abs/2408.04797)
- **What's New**: 실내 내비게이션은 사람, 자율 주행 차량, 드론, 로봇 등의 실내 공간에서 추적 및 위치 확인을 돕는 기본 기술입니다. 건물, 지하 공간 및 밀집된 도시 환경에서 GPS 신호가 도달하지 못하는 경우가 많기 때문에, 실내 내비게이션 솔루션은 일반적으로 모바일 임베디드 시스템의 센서와 유비쿼터스 무선 신호(예: WiFi)를 사용하여 추적 및 위치 확인을 수행합니다. 이 기사는 최첨단 실내 내비게이션 솔루션이 직면한 여러 과제를 개괄적으로 설명한 다음, 모바일 임베디드 시스템에 배포된 AI 알고리즘이 이러한 과제를 극복하는 방법을 설명합니다.



### AI Consciousness and Public Perceptions: Four Futures (https://arxiv.org/abs/2408.04771)
- **What's New**: 본 논문은 첨단 인공지능 시스템(AIs)의 윤리적 지위에 대한 논의를 통해 발생할 수 있는 위험을 분석하며, 이러한 위험은 오용, 사고, 통제력 상실 등 기존의 AI 위험과 동일한 중요성을 지니며 유사한 시기에 발생할 수 있음을 강조합니다. 특히, 미래 첨단 AI 시스템의 의식(consciousness) 여부와 사회가 AI의 의식을 믿는 정도를 분석하여 4가지 시나리오(참/거짓 양성, 참/거짓 음성)를 제시하고 각 시나리오에 따른 위험을 평가합니다. 본 논문은 AI 고통(suffering), 인간의 권한 축소(disempowerment), 지정학적 불안정(geopolitical instability), 인간의 타락(depravity) 등 4가지 주요 위험을 식별하고 각 시나리오에 대한 위험 평가를 제공합니다. 분석 결과, AI가 비 의식적이라고 잘못 믿는 경우(false negative)가 가장 심각한 위험이며, AI가 의식적이라고 잘못 믿는 경우(false positive)가 그 뒤를 잇는 것으로 나타났습니다.



### Data-Driven Pixel Control: Challenges and Prospects (https://arxiv.org/abs/2408.04767)
Comments:
          Accepted to the Conference on Dynamic Data-Driven Applications Systems (DDDAS2024)

- **What's New**: 이 논문은 픽셀 단위의 동적 감지 (dynamic sensing)와 비디오 단위의 컴퓨터 비전 분석을 결합한 데이터 기반 시스템을 연구하며, 감지 전단 (sensor front-end)과 계산 후단 (computational back-end) 간의 데이터 이동을 최소화하는 피드백 제어 루프 (feedback control loop)를 제안합니다. 이 시스템은 객체 감지 및 추적 정확도를 저하시키지 않으면서도 대역폭을 줄이고 에너지 효율을 높입니다.



### Embodied Uncertainty-Aware Object Segmentation (https://arxiv.org/abs/2408.04760)
Comments:
          IROS 2024

- **What's New**: 이 논문에서는 불확실성 인식 객체 인스턴스 분할(UncOS)을 소개하고, 이것이 신체화된 상호 작용 분할에 유용하다는 것을 보여줍니다. 로봇 인식의 불확실성을 다루기 위해 객체 분할의 가설 분포를 생성하는 방법을 제안합니다. 대규모 사전 훈련된 모델을 여러 번 쿼리하여 신뢰도 추정치와 함께 영역 요인화된 분할 가설 집합을 얻습니다. 이 과정은 보이지 않는 객체 분할 문제에서 최첨단 성능을 달성하는 분할 결과를 생성할 수 있습니다. 출력은 또한 모호성을 줄이기 위해 장면을 방해하기 위한 로봇 작업 선택을 위한 믿음 기반 프로세스의 입력으로 사용될 수 있습니다. 실제 로봇 실험을 통해 이 방법의 효과를 보여줍니다.



### More Questions than Answers? Lessons from Integrating Explainable AI into a Cyber-AI Too (https://arxiv.org/abs/2408.04746)
Comments:
          ACM CHI 2024 Workshop on Human-Centered Explainable AI (HCXAI)

- **What's New**: 본 연구는 사이버보안 분석가를 위한 도메인 특화 워크플로우에 XAI를 구현하는 과정에서 나타나는 관찰 결과와 어려움을 공유합니다. 특히, 소스 코드 분류에 XAI를 활용하는 사례 연구를 통해 정확한 평가와 신속성이 중요한 분야에서 XAI의 적용 가능성을 살펴봅니다. 연구팀은 최첨단 핵심 설명 기술(예: SHAP 또는 LIME)이 AI 전문 지식이 부족한 사람들이 해석할 때는 효용성이 떨어진다는 사실을 발견했습니다. 이는 해당 기술들이 비 전문가를 위해 개발되었음에도 불구하고 사용자에게 전달되는 과정에서 의미가 왜곡될 수 있음을 의미합니다. 또한, 연구팀은 널리 사용되는 XAI 기술들이 사후적(post hoc)이며 설명이 국소적인(localized) 경향이 있어 실시간 인간-AI 워크플로우에서는 제한적인 통찰력을 제공한다는 사실을 발견했습니다. 대신 사이버 분석가는 워크플로우를 최소한으로 방해하면서 쉽게 이해할 수 있는 고수준의 설명을 필요로 합니다. 연구팀은 실질적이고 효과적인 XAI에서 해결되지 않은 문제점을 제시하고, 대규모 언어 모델(LLM)과 같은 신기술이 이러한 문제를 완화할 수 있는 방안을 제시합니다.



### Survey: Transformer-based Models in Data Modality Conversion (https://arxiv.org/abs/2408.04723)
Comments:
          Submitted to ACM Computing Surveys (CSUR)

- **What's New**: This paper provides a comprehensive review of Transformer-based (TB) models applied to modality conversion, focusing on text, vision, and speech. It systematically summarizes various TB models for converting data between these three modalities, including text-to-speech, speech-to-text, image-to-text, and more, highlighting the versatility and scalability of transformers in advancing AI-driven content generation and understanding.



### DyGMamba: Efficiently Modeling Long-Term Temporal Dependency on Continuous-Time Dynamic Graphs with State Space Models (https://arxiv.org/abs/2408.04713)
Comments:
          Preprint. Work on progress

- **What's New**: This paper introduces DyGMamba, a new model for continuous-time dynamic graph (CTDG) representation learning that leverages state space models (SSMs) to effectively handle long-term temporal dependencies.



### MulliVC: Multi-lingual Voice Conversion With Cycle Consistency (https://arxiv.org/abs/2408.04708)
- **What's New**: 이 논문은 MulliVC라고 불리는 새로운 다국어 음성 변환 시스템을 제안하며, 이는 타겟 스피커의 음색을 유지하면서 원본 언어의 프로소디를 보존하는 것을 목표로 합니다. 기존의 다국어 음성 변환 방법과 달리, MulliVC는 동일한 스피커의 다국어 데이터가 필요하지 않습니다. 대신, MulliVC는 타겟 스피커의 음색 정보를 학습하기 위해 다국어 데이터 없이도 사이클 훈련 전략을 활용합니다.



### Understanding the Performance and Estimating the Cost of LLM Fine-Tuning (https://arxiv.org/abs/2408.04693)
Comments:
          10 pages, conference

- **What's New**: 본 논문은 제한된 GPU 자원으로 특정 작업에 대하여 대규모 언어 모델(LLM)을 효율적으로 특화시키는 방법인 LLM 미세 조정의 성능을 특성화합니다. 특히, GPU 하나에서의 성능과 실행 시간 성능을 이해하기 위해,  Sparse Mixture of Experts (MoE) 기반 LLM 미세 조정에 대한 연구를 진행합니다.

- **Technical Details**: 본 논문은 두 가지 MoE 모델 (Mixtral-8x7B, BlackMamba-630M/2.8B)과 두 가지 도메인 특정 데이터 세트 (commonsense_15k, Math_14k)를 사용하여 LLM 미세 조정을 평가합니다. 이 연구는 밀집 및 희소 MoE 모델의 학습 효율과 런타임 특성을 비교하며, 이는 최대 배치 크기, 실행 시간 분류, 종단 간 처리량, GPU 하드웨어 사용률, 부하 분산을 포함합니다.

- **Performance Highlights**: 결과는 다음과 같은 주요 통찰력을 제공합니다. 1) 미세 조정은 10회 미만의 epoch 내에 달성할 수 있으며, 전문가의 하위 집합을 활성화하는 희소 MoE 모델은 밀집형 모델과 동일하게 학습할 수 있습니다. 2) MoE 계층은 LLM 미세 조정에서 가장 많은 실행 시간을 차지하며 MoE 계층 성능을 최적화하는 것은 LLM 미세 조정의 전반적인 비용을 개선하는 데 중요합니다. 3) 희소 MoE 모델은 더 큰 배치 크기를 지원하여 종단 간 처리량을 향상시킵니다. 4) 배치 크기가 증가함에 따라 워크로드는 계산 제한이 됩니다. 5) 희소 모델의 미세 조정은 더 많은 부하 불균형으로 이어집니다. 이러한 통찰력을 바탕으로 본 논문은 모델 크기, 데이터 세트 크기 및 GPU 아키텍처를 기반으로 LLM 미세 조정 비용을 추정하는 분석 모델을 제시합니다.

- **Analytical Model**: 본 논문은 GPU 메모리를 고려하여 최대 배치 크기를 추정하고 미세 조정 처리량을 계산합니다. 실험 결과를 통해 이 처리량이 검증되었으며, RMSE는 0.55 미만입니다. 추정된 처리량을 사용하여 이 모델은 다양한 클라우드 제공업체에 대한 미세 조정 비용을 계산합니다.



### Exploring Scalability in Large-Scale Time Series in DeepVATS framework (https://arxiv.org/abs/2408.04692)
Comments:
          Admitted pending publication in Lecture Notes in Network and Systems (LNNS) series (Springer). Code available at this https URL

- **What's New**: DeepVATS는 대용량 시계열 데이터 분석을 위한 시각적 분석 도구로, 딥러닝(DL)과 시각적 분석(VA)을 결합하여 사용자 친화적인 인터페이스를 제공합니다. 이 도구는 데이터 세트 로딩, 모델 학습, 임베딩 추출, 시각화 및 클러스터링 등의 기능을 통합하여 대규모 시계열 데이터의 패턴 및 트렌드를 효과적으로 파악하고 분석할 수 있도록 지원합니다.

- **Technical Details**: DeepVATS는 Deep Learning(DL) 모듈, Storage 모듈, Visual Analytics(VA) 모듈의 세 가지 주요 모듈로 구성됩니다.

* **Deep Learning(DL) 모듈:** R로 구현된 DL 모듈은 데이터 세트 로딩 및 저장, 모델 학습 및 임베딩 추출 등을 담당합니다. DL 모듈은 Weights & Biases 시스템을 사용하여 데이터를 저장하고 관리합니다.
* **Storage 모듈:** Weights & Biases 시스템을 사용하여 데이터를 저장하고 관리합니다.
* **Visual Analytics(VA) 모듈:** R Shiny 애플리케이션을 기반으로 구축된 VA 모듈은 임베딩 공간의 투영 및 클러스터링 파라미터를 조정할 수 있는 인터랙티브 플롯을 제공합니다. VA 모듈은 임베딩 및 시계열 데이터를 동시에 시각화하여 사용자에게 상호 작용을 통해 데이터를 탐색할 수 있는 기능을 제공합니다.

- **Performance Highlights**: DeepVATS는 대규모 시계열 데이터 세트에 대한 확장성 분석을 수행하여 성능을 개선했습니다. 특히, Monash 벤치마크의 Solar Power 데이터 세트를 사용하여 시간 범위를 변경하면서 실행 시간을 측정하고 분석했습니다. 분석 결과, 다음과 같은 성능 문제가 발견되었으며 개선 방안을 모색했습니다.

* **UMAP 구현의 안정성 문제:** cuml UMAP 구현의 내부 문제로 인해 안정성이 떨어지는 문제가 발생했습니다. 향후 분석을 위해 다른 대안을 모색할 필요가 있습니다.
* **R과 Python 간 통신 오버헤드:** reticulate를 사용한 R과 Python 간 통신으로 인해 추가적인 시간이 소요되었습니다. 따라서 R 변수 대신 pickled 파일을 사용하여 성능을 향상시키는 방안을 고려하고 있습니다.
* **Shiny의 반응성으로 인한 추가 실행 단계:** Shiny의 반응성으로 인해 추가적인 실행 단계가 발생했습니다. R Shiny 캐시를 효율적으로 활용하여 이 문제를 해결할 수 있는 방법을 모색하고 있습니다.



### Improving Relational Database Interactions with Large Language Models: Column Descriptions and Their Impact on Text-to-SQL Performanc (https://arxiv.org/abs/2408.04691)
- **What's New**: 이 논문은 관계형 데이터베이스(Relational database)의 설명력이 부족한 컬럼 묘사(Column description) 문제를 해결하기 위해 대규모 언어 모델(LLM)을 활용하여 정보가 풍부한 컬럼 묘사를 자동으로 생성하는 방법을 제시합니다. 이는 사람과 Text-to-SQL 모델 모두에게 데이터베이스의 이해도를 높여줍니다.

- **Technical Details**: 본 연구에서는 BIRD-Bench 개발 셋을 기반으로 LLM과 사람의 수정을 거쳐 컬럼 묘사가 포함된 새로운 데이터셋인 ColSQL을 만들었습니다. 다양한 모델을 평가한 결과, GPT-4o와 Command R+가 고품질 컬럼 묘사를 생성하는 데 뛰어난 성능을 보였습니다. LLM을 심판(judge)으로 활용하여 모델 성능을 평가하는 방법도 시도했지만, 사람의 평가와 일치하지 않았으며 더 많은 연구가 필요합니다. 컬럼 묘사를 추가하면 특히 정보가 부족한 컬럼에서 Text-to-SQL 작업의 정확도를 크게 향상시키는 것으로 나타났습니다.

- **Performance Highlights**: LLM은 컬럼 묘사 생성에 매우 효과적이며, 특히 정보가 부족한 컬럼에서 Text-to-SQL 정확도를 향상시킵니다. GPT-4o와 Command R+는 고품질 컬럼 묘사를 생성하는 데 탁월한 성능을 보여주었습니다. 이 연구는 LLM이 데이터베이스 사용성을 개선하는 데 도움이 되는 상세한 메타데이터(Metadata) 생성 도구임을 입증합니다.



### Design of a Quality Management System based on the EU Artificial Intelligence Ac (https://arxiv.org/abs/2408.04689)
- **What's New**: 본 논문은 유럽연합 (EU)의 인공지능 (AI) 법안 (EU AIA)의 규정을 기반으로 고위험 AI 시스템의 검증 및 문서화를 위한 품질 관리 시스템 (QMS)을 제안합니다. 특히, 대규모 언어 모델 (LLM)과 같은 범용 AI (GPAI) 모델을 포함한 AI 시스템을 위한 QMS의 설계 및 아키텍처를 제공하며, 위험 관리 시스템 (RMS)을 포함한 프로토타입을 구현하고 평가합니다.



### Multi-Turn Context Jailbreak Attack on Large Language Models From First Principles (https://arxiv.org/abs/2408.04686)
- **What's New**: This paper proposes a new multi-turn jailbreak attack method called Context Fusion Attack (CFA) that leverages contextual scenarios to effectively bypass LLM security measures. This method dynamically integrates the target into contextual scenarios, concealing malicious intent by replacing harmful key terms with innocuous ones.



### Eliminating Backdoors in Neural Code Models via Trigger Inversion (https://arxiv.org/abs/2408.04683)
Comments:
          Under review

- **What's New**: EliBadCode라는 새로운 백도어 방어 기법을 제안하여 코드 이해를 위한 신경망 코드 모델(NCM)의 백도어를 제거합니다. 이 기술은  Trigger Inversion(트리거 반전) 기법을 사용하여 백도어를 제거합니다.  EliBadCode는 트리거 반전을 위한 검색 공간을 줄이고 효율성을 높이는 트리거 토큰 필터링 기능을 포함합니다. 또한, 트리거 삽입 위치를 식별하여 트리거 반전 중에 적대적 방해 요소를 줄이는 기술을 사용합니다. 또한,  EliBadCode는 효율적인 반전 트리거 생성을 위해  Greedy Coordinate Gradient(GCG) 알고리즘을 사용하며 트리거 고정 메커니즘을 통해 반전 트리거를 정제합니다. 마지막으로 모델 재학습을 통해 백도어를 제거합니다.



### ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities (https://arxiv.org/abs/2408.04682)
- **What's New**: ToolSandbox는 상태 유지(stateful), 대화형(conversational) 도구 사용 벤치마크로, 상태 유지 도구 간의 암시적 의존성(implicit state dependencies), LLM 시뮬레이션 사용자(LLM simulated user), 역동적인(dynamic) 평가 전략을 포함합니다. 이를 통해 도구 사용 LLM 기능의 포괄적인 평가를 제공합니다.



### Conversational AI Powered by Large Language Models Amplifies False Memories in Witness Interviews (https://arxiv.org/abs/2408.04681)
- **What's New**: 이 연구는 AI가 인간의 거짓 기억 형성에 미치는 영향을 조사했습니다. 특히, AI 대화 시스템(챗봇)과의 상호 작용을 통해 거짓 기억이 유발되는 과정을 살펴봤으며, 이는 범죄 현장 증인 인터뷰를 시뮬레이션한 것입니다. 연구에서는 4가지 조건, 즉 통제 조건, 설문 조사 기반 조건, 사전 설정된 챗봇 조건, 그리고 대규모 언어 모델(LLM) 기반 생성형 챗봇 조건을 비교 분석했습니다.



### Dynamic Fog Computing for Enhanced LLM Execution in Medical Applications (https://arxiv.org/abs/2408.04680)
- **What's New**: This paper presents **SpeziLLM**, an open-source framework for executing Large Language Models (LLMs) in a decentralized fog computing architecture, addressing the privacy, trust, and cost concerns associated with centralized cloud-based LLM platforms in healthcare.



### Towards Linguistic Neural Representation Learning and Sentence Retrieval from Electroencephalogram Recordings (https://arxiv.org/abs/2408.04679)
- **What's New**: 이 논문에서는 뇌파 신호를 문장으로 변환하는 새로운 접근 방식인 **EEG-to-Text Retrieval (ETER)**를 제안합니다. ETER는 **과도한 사전 훈련된 LLM 디코더에 대한 의존성을 제거**하고 **EEG 인코더가 텍스트 읽기 EEG 데이터에서 학습한 의미 정보를 평가**할 수 있도록 합니다. 이를 위해 **마스크 콘트라스트 학습 손실**을 사용하여 컨포머 기반 EEG 인코더를 훈련하여 의미론적 EEG 표현을 학습합니다.



### CREST: Effectively Compacting a Datastore For Retrieval-Based Speculative Decoding (https://arxiv.org/abs/2408.04678)
- **What's New**: CREST (Compact Retrieval-Based Speculative Decoding) is a new approach for efficient speculative decoding in large language models (LLMs). It enhances the existing REST (Retrieval-Based Speculative Decoding) method by redesigning the datastore structure to effectively 'compact' it while maintaining or even improving performance.

- **Technical Details**: CREST decouples n-grams from each other in the datastore, allowing selective storage of the most frequent and smallest n-grams. This compaction strategy leads to reduced storage space and surprisingly, improved acceptance length.

- **Performance Highlights**: CREST outperforms REST in terms of storage efficiency and performance. It achieves a 10.6-13.5x reduction in storage space compared to REST while achieving a 16.5-17.1% higher acceptance length using the same storage space on HumanEval and MT Bench benchmarks.

- **Advantages**: CREST's key advantages include:
- **Reduced storage space:** Storing only a subset of n-grams significantly reduces the storage requirement for the datastore.
- **Improved performance:** By focusing on the most common and smallest n-grams, CREST achieves a higher acceptance length and faster drafting time complexity.

- **Comparison with REST**: CREST improves upon REST by addressing the issue of unbounded datastore growth. It allows for efficient compaction without sacrificing performance, making it a more practical and scalable approach for speculative decoding.



### ACL Ready: RAG Based Assistant for the ACL Checklis (https://arxiv.org/abs/2408.04675)
- **What's New**: ACLReady라는 도구는 ACL (Association for Computational Linguistics) 책임감 있는 NLP 연구 체크리스트를 작성하는 데 도움을 주는 Retrieval-Augmented Language Model (RAG) 기반 애플리케이션입니다. 이 도구는 저자들이 자신의 연구에 대한 깊은 생각을 할 수 있도록 돕고 체크리스트에 대한 답변을 생성하는 데 도움을 줄 수 있습니다.



### AutoFAIR : Automatic Data FAIRification via Machine Reading (https://arxiv.org/abs/2408.04673)
- **What's New**: 본 논문은 자동화된 FAIR 데이터 처리를 위한 새로운 아키텍처인 AutoFAIR를 제안합니다. AutoFAIR는 웹 페이지에서 메타데이터를 자동으로 추출하고 표준화하는 데 집중하여 데이터 찾기, 접근, 상호 운용성 및 재사용성을 향상시킵니다. AutoFAIR는 Web Reader와 FAIR Alignment라는 두 가지 주요 구성 요소를 통합하여 데이터 FAIR화를 자동화합니다.



### Prompt and Prejudic (https://arxiv.org/abs/2408.04671)
Comments:
          Accepted at ECCV workshop FAILED

- **What's New**: 이 논문은 큰 언어 모델(LLM, Large Language Models)과 비전 언어 모델(VLM, Vision Language Models)에서 윤리적 의사 결정 작업을 수행할 때 이름을 사용하는 것이 미치는 영향을 조사합니다. 이 연구는 윤리적으로 주석이 달린 텍스트 시나리오에 이름을 추가하여 모델 출력에서 인구 통계적 편견을 드러내는 접근 방식을 제안합니다. 이 연구에는 다양한 성별과 민족적 배경을 대표하는 300개 이상의 이름 목록이 포함되어 있으며, 수천 개의 도덕적 시나리오에서 테스트되었습니다. 연구팀은 사회 과학의 감사 방법론을 따르면서 인기 있는 LLM/VLM을 포함한 자세한 분석을 제안하여 이러한 시스템의 편견을 인식하고 완화하는 것이 중요하다는 점을 강조함으로써 책임감 있는 AI 분야에 기여합니다. 또한 연구팀은 실제 시나리오 벤치마크(PSB, Pratical Scenarios Benchmark)라는 새로운 벤치마크를 소개합니다. PSB는 일상적인 의사 결정 시나리오에서 성별 또는 인구 통계적 편견과 관련된 편견이 있는지 여부를 평가하고 LLM이 합리적인 결정을 내리기 위해 사용될 수 있는 실제 시나리오(예: 모기지 또는 보험 부여)를 평가하도록 설계되었습니다. 이 벤치마크를 통해 LLM과 VLM의 실제 적용에서 발생할 수 있는 위험과 편견을 강조하면서 다양한 인구 통계적 범주에 걸쳐 모델 동작을 포괄적으로 비교할 수 있습니다.



### Forecasting Live Chat Intent from Browsing History (https://arxiv.org/abs/2408.04668)
Comments:
          CIKM 2024

- **What's New**: This paper proposes a two-stage approach to predict user intent based on their browsing history on online shopping websites. The first stage classifies the browsing history into high-level intent categories using fine-tuned Transformers. The second stage then uses a large language model (LLM) to generate detailed user intents based on the browsing history and the predicted intent class.



### LLM Stability: A detailed analysis with some surprises (https://arxiv.org/abs/2408.04667)
- **What's New**: 본 연구는 대규모 언어 모델(LLM)의 안정성을 정량화하여 분석했습니다. 동일한 입력과 결정론적 하이퍼파라미터를 사용해도 결과가 다르게 나타나는 LLM의 특징을 심층적으로 살펴보았습니다. 특히, LLM이 출력 수준에서는 거의 결정론적이지 않지만, 구문 분석된 출력/답변 수준에서는 훨씬 더 결정론적이라는 사실을 밝혔습니다. 또한, LLM 정확도 변화는 정규 분포를 따르지 않으며, 안정성은 작업에 따라 다르게 나타난다는 점도 발견했습니다.



### LLMs are Not Just Next Token Predictors (https://arxiv.org/abs/2408.04666)
- **What's New**: 이 논문은 거대 언어 모델(LLM)을 단순히 다음 토큰 예측기로만 보는 시각에 대한 비판을 제기합니다. LLM은 다음 토큰 예측을 목표로 하는 확률적 경사 하강법(stochastic gradient descent)을 통해 언어를 학습하는 통계적 모델이지만, 이러한 예측 능력만으로 LLM을 설명하는 것은 부족하며, LLM의 작동 방식과 능력을 제대로 이해하기 위해서는 더 포괄적인 관점이 필요하다고 주장합니다. 이러한 주장을 뒷받침하기 위해 논문에서는 유전자의 관점에서 진화와 발달을 설명하려는 생물학 연구 프로그램에 대한 비유를 사용합니다.



### LLM-based MOFs Synthesis Condition Extraction using Few-Shot Demonstrations (https://arxiv.org/abs/2408.04665)
- **What's New**: 본 논문은 기존의 제로 샷 학습(zero-shot learning) 방식 대신 퓨샷 인 컨텍스트 학습(few-shot in-context learning) 방식을 활용하여 MOFs 합성 조건 추출을 개선한 연구를 제시합니다. 이를 통해 MOFs 합성 조건을 더 정확하고 효율적으로 추출할 수 있으며, 이는 새로운 MOFs 설계 및 발견에 중요한 역할을 할 것으로 기대됩니다.

- **Technical Details**: 본 연구는 다음과 같은 기술적 특징을 가지고 있습니다:  
 1. 인간-AI 협업 데이터 큐레이션 프로세스를 통해 고품질 기반 진실(ground-truth) 데모를 확보합니다. 
 2. BM25 알고리즘 기반의 검색 증강 생성(RAG) 기법을 활용하여 각 MOF 추출에 적합한 퓨샷 데모를 선택적으로 사용합니다. 
 3. LLM의 성능을 향상시키기 위해 다양한 지식(retrieval된 합성 조건, 숫자/텍스트 형식 제약, 퓨샷 데모)을 통합합니다. 
 4. 대규모 합성 추출을 위한 확장성을 고려하여 핵심 문단 검출, 데모 풀 크기 조절, LLM 기반 공동 참조 해결(co-reference resolution) 등의 기술을 적용합니다.

- **Performance Highlights**: 본 논문에서 제시된 퓨샷 방식은 제로 샷 LLM 방식보다 14.8% 더 높은 F1 성능을 달성했습니다. 또한 실제 MOFs 합성-구조 추론 실험에서 퓨샷 방식은 기존 방식보다 29.4% 향상된 성능을 보여주었습니다. 이는 퓨샷 인 컨텍스트 학습이 MOFs 합성 조건 추출에 효과적임을 증명합니다.



### Mitigating Hallucinations in Large Vision-Language Models (LVLMs) via Language-Contrastive Decoding (LCD) (https://arxiv.org/abs/2408.04664)
- **What's New**: 이 논문은 대규모 비전-언어 모델(LVLM)에서 발생하는 객체 환각(object hallucination) 문제를 완화하기 위해 새로운 디코딩 알고리즘인 언어 대조 디코딩(LCD: Language Contrastive Decoding)을 제안합니다. LCD는 LVLM 출력을 조정하여 객체 환각을 줄이는 데 효과적이며, 특히 LLM의 확률 분포 신뢰도 수준을 기반으로 하는 동적 가중치 메커니즘을 사용하여 LLM의 언어 편향을 줄이는 데 중점을 둡니다.



### Dopamin: Transformer-based Comment Classifiers through Domain Post-Training and Multi-level Layer Aggregation (https://arxiv.org/abs/2408.04663)
Comments:
          Accepted at The 3rd Intl. Workshop on NL-based Software Engineering, 2024

- **What's New**: Dopamin은 코드 주석 분류를 위한 새로운 Transformer 기반 도구로, 다양한 프로그래밍 언어에서 다양한 주석 유형에 대한 도메인 사후 훈련 (domain post-training)을 통해 언어 간 지식 전이 (knowledge transfer)를 실현하고 계층적 집계 (Hierarchical aggregation, HSUM) 전략을 활용하여 주석 표현을 향상시켜 보다 정확하고 관련성 있는 분류를 제공합니다.

- **Technical Details**: Dopamin은 CodeBERT 모델을 기반으로 하며, 다양한 프로그래밍 언어(Java, Python, Pharo)에서 수집된 다양한 주석 유형에 대한 사후 훈련을 통해 언어 간 지식 전이를 수행합니다. 또한 HSUM 전략을 적용하여 BERT 모델의 상위 계층(layers)을 결합하여 입력 텍스트의 의미 정보를 더 풍부하게 표현합니다. 이러한 기술은 Dopamin이 다양한 프로그래밍 언어의 미묘한 차이를 파악하여 더욱 정확한 주석 분류를 가능하게 합니다.

- **Performance Highlights**: NLBSE'24 Tool Competition 데이터셋에서 Dopamin은 STACC 기준 모델보다 평균 F1 점수가 3% 높은 0.74를 달성하며 우수한 성능을 보여주었습니다. 이는 Dopamin이 코드 주석 분류 작업에 있어서 효과적인 도구임을 입증합니다.



### Citekit: A Modular Toolkit for Large Language Model Citation Generation (https://arxiv.org/abs/2408.04662)
Comments:
          7 pages, 13 figures

- **What's New**: Citekit, an open-source and modular toolkit for LLM (Large Language Model) citation generation in Question Answering (QA) tasks, is introduced. It aims to standardize and facilitate the comparison of different citation generation methods, promoting reproducibility and the development of new approaches.

- **Technical Details**: Citekit consists of four main modules: Input, Generation Module, Enhancing Module, and Evaluator.  Input handles data loading and prompt generation, Generation Module houses LLMs for response generation with citations, Enhancing Module includes components for assessment, retrieval, planning, and editing, and Evaluator incorporates metrics for evaluating the pipeline's output. The toolkit is highly extensible, allowing users to customize modules and connections for specific tasks.

- **Performance Highlights**: Citekit enables a comprehensive evaluation of 11 state-of-the-art citation generation baselines on Llama3 and GPT-4o, highlighting the strengths and weaknesses of different modules in improving answer accuracy and citation quality. The analysis led to the development of a new method, self-RAG Snippet, which achieves a balance between answer accuracy and citation quality by combining effective components.



### XMainframe: A Large Language Model for Mainframe Modernization (https://arxiv.org/abs/2408.04660)
- **What's New**: 이 연구는 레거시 메인프레임 시스템의 관리 및 현대화를 위한 혁신적인 도구로서 XMainframe라는 대규모 언어 모델(LLM)을 소개합니다. XMainframe은 메인프레임 시스템과 COBOL 코드베이스에 대한 지식을 갖추도록 특별히 설계되었으며, 고품질의 훈련 데이터 세트를 생성하는 광범위한 데이터 수집 파이프라인을 통해 훈련되었습니다. 또한, 연구에서는 메인프레임 지식 평가를 위한 종합적인 벤치마크인 MainframeBench도 제시합니다. MainframeBench는 다지선다형 질문, 질의응답 및 COBOL 코드 요약을 포함합니다.



### Winning Amazon KDD Cup'24 (https://arxiv.org/abs/2408.04658)
- **What's New**: This paper presents the winning solution for all five tracks of the Amazon KDD Cup 2024 Multi Task Online Shopping Challenge for LLMs, focused on building a useful online shopping assistant.

- **Technical Details**: The solution involves fine-tuning the Qwen2-72B-Instruct model on a custom training dataset created by processing multiple public datasets and utilizing Large Language Models (LLMs) for data augmentation.  Key techniques include:

* **Wise-ft** to mitigate distribution shifts during training.
* **Ensemble of LoRA adapters** within a single model to improve performance.
* **Logits Processors** to constrain model output to relevant tokens for specific tasks.
* **4-bit Quantization** and **vLLM** to optimize inference time and resource usage.

- **Performance Highlights**: The solution achieved first place in each individual track of the competition and secured the overall first place in the Amazon KDD Cup 2024. It showcases the effectiveness of fine-tuning and data augmentation techniques for building a robust and efficient LLM-based shopping assistant.



### Strong and weak alignment of large language models with human values (https://arxiv.org/abs/2408.04655)
- **What's New**: 본 논문에서는 AI 시스템이 인간의 가치와 일치하도록 하는 것에 대해 두 가지 개념, **강한 정렬(strong alignment)**과 **약한 정렬(weak alignment)**을 구분합니다. **강한 정렬**은 AI가 인간의 의도를 이해하고 예측하고, 행동의 결과를 예측하여 인간의 가치를 위협하는 상황을 인식할 수 있어야 한다는 것을 의미합니다. 반면 **약한 정렬**은 AI가 인간의 가치를 완전히 이해하지 못하더라도, 주어진 상황에서 인간의 가치에 부합하는 행동을 하도록 훈련하는 것을 의미합니다.



### Batching BPE Tokenization Merges (https://arxiv.org/abs/2408.04653)
Comments:
          8 pages, 5 figures, 1 code block

- **What's New**: BatchBPE, an open-source pure Python implementation of batching Byte Pair Encoding (BPE) algorithm for tokenizer training, is presented. This technique enables faster and more memory-efficient training on a standard laptop. BatchBPE allows experimenting with different tokenization strategies by preprocessing stop words and ignoring infrequent text chunks.

- **Technical Details**: BatchBPE takes advantage of the power-law distribution of text chunks in a dataset by representing it as a dictionary mapping text chunks to their frequencies. This significantly reduces memory usage and processing time.  It introduces two features for preprocessing: stop word removal and frequency cutoff.  Stop word removal excludes common words from the token merging process, while frequency cutoff discards text chunks appearing below a certain frequency.  BatchBPE enables safe batch merging of token pairs by defining a set of non-interfering merges. This significantly speeds up vocabulary building. 

- **Performance Highlights**: The effectiveness of BatchBPE is demonstrated through experiments with the FineWeb-Edu dataset.  These experiments reveal the impact of stop word preprocessing and frequency cutoff on the resulting encoded text lengths. The results show that both techniques can have a significant impact on the final tokenization results.



### Leveraging Large Language Models with Chain-of-Thought and Prompt Engineering for Traffic Crash Severity Analysis and Inferenc (https://arxiv.org/abs/2408.04652)
Comments:
          20 pages, 12 figures, 3 tables

- **What's New**: 이 연구는 최첨단 LLM(Large Language Model)인 GPT-3.5-turbo, LLaMA3-8B, LLaMA3-70B를 사용하여 교통사고 심각도 추론을 분류 작업으로 수행합니다. 도메인 지식을 포함한 사전 구축된 템플릿을 사용하여 교통사고 표 데이터에서 텍스트 설명을 생성하고, 사고 원인을 분석하고 심각도를 추론하기 위해 CoT(Chain-of-Thought) 추론을 통합합니다. 또한 사고 심각도 추론을 위해 특별히 설계된 프롬프트 엔지니어링의 영향을 조사합니다.



### Knowledge AI: Fine-tuning NLP Models for Facilitating Scientific Knowledge Extraction and Understanding (https://arxiv.org/abs/2408.04651)
Comments:
          11 pages

- **What's New**: 본 프로젝트는 특정 도메인에서 과학적 지식을 이해하고 추출하는 데 있어 대규모 언어 모델(LLM)의 효과를 조사하여 지식 AI라는 딥 러닝 프레임워크를 만듭니다. 이 프레임워크의 일환으로, 사전 훈련된 모델을 사용하고 과학 도메인의 데이터셋에서 미세 조정합니다. 모델은 네 가지 핵심적인 자연어 처리(NLP) 작업, 즉 요약, 텍스트 생성, 질문 답변 및 명명된 엔터티 인식에 적응됩니다. 결과는 도메인별 미세 조정이 각 작업에서 모델 성능을 크게 향상시켜 과학적 맥락에 대한 적용성을 높인다는 것을 나타냅니다. 이러한 적응을 통해 비전문가는 타겟 과학 분야 내에서 정보를 효율적으로 쿼리하고 추출할 수 있으며, 미세 조정된 LLM이 과학 분야에서 지식 발견을 위한 도구로서의 잠재력을 보여줍니다.



### Building Trust in Mental Health Chatbots: Safety Metrics and LLM-Based Evaluation Tools (https://arxiv.org/abs/2408.04650)
- **What's New**: 본 연구는 접근성, 인간과 유사한 상호 작용, 맥락 인식 지원 등으로 인해 점점 더 인기를 얻고 있는 정신 건강 챗봇의 안전성과 신뢰성을 보장하기 위한 평가 프레임워크를 개발하고 검증했습니다. 연구진은 챗봇 응답에 대한 100개의 벤치마크 질문 및 이상적인 답변과 5개의 가이드라인 질문으로 구성된 평가 프레임워크를 만들었습니다. 정신 건강 전문가가 검증한 이 프레임워크는 GPT-3.5-turbo 기반 챗봇에서 테스트되었습니다. 자동 평가 방법에는 대규모 언어 모델(LLM) 기반 점수, 실시간 데이터를 사용하는 에이전틱 접근 방식, 챗봇 응답을 실제 기준과 비교하기 위한 임베딩 모델이 포함되었습니다.



### Chain of Stance: Stance Detection with Large Language Models (https://arxiv.org/abs/2408.04649)
- **What's New**: This paper introduces a novel prompting method called "Chain of Stance" (CoS) for stance detection using large language models (LLMs). Unlike existing methods that solely focus on fine-tuning LLMs, CoS decomposes the stance detection process into a sequence of intermediate reasoning steps, allowing LLMs to act as expert stance detectors.

- **Technical Details**: CoS leverages the encyclopedic knowledge of LLMs by prompting them with a series of questions related to the context, sentiment, and opinion expressed in the text. This chain of assertions ultimately leads to the final stance prediction. The method draws inspiration from the Chain-of-Thought (CoT) prompting paradigm used in mathematical reasoning tasks.

- **Performance Highlights**: Extensive experiments were conducted using four state-of-the-art LLMs (Mistral-7B, Qwen 1.5-7B, LLaMA 3-8B, LLaMA 2-7B) on the SemEval 2016 dataset. CoS achieved state-of-the-art results with an F1 score of 79.84 in the few-shot setting and 76.43 in the zero-shot setting, outperforming other baselines.



### PLUGH: A Benchmark for Spatial Understanding and Reasoning in Large Language Models (https://arxiv.org/abs/2408.04648)
Comments:
          Wordplay Workshop @ ACL 2024

- **What's New**: PLUGH (**P**layable **L**anguage **U**nderstanding **G**raph **H**andling) is introduced, a new benchmark for assessing LLMs' spatial understanding and reasoning skills. This benchmark comprises 5 tasks based on 125 input texts extracted from 48 text-based games, totaling 61 distinct spatial graphs.



### Evaluating the Impact of Advanced LLM Techniques on AI-Lecture Tutors for a Robotics Cours (https://arxiv.org/abs/2408.04645)
Comments:
          The article is an extended version of a paper presented at the International Workshop on AI in Education and Educational Research (AIEER) at ECAI-2024 (27th European Conference on Artificial Intelligence)

- **What's New**: 본 연구는 대학 강의를 위한 인공지능 튜터로서 대규모 언어 모델(LLM)의 성능을 평가했습니다. 특히, 프롬프트 엔지니어링, 검색 강화 생성(RAG), 미세 조정과 같은 다양한 고급 기술을 활용했습니다. BLEU-4, ROUGE, BERTScore와 같은 일반적인 유사성 지표를 사용하여 모델과 적용된 기술을 평가했으며, 도움 유용성과 신뢰성에 대한 소규모 인간 평가를 보완했습니다. 연구 결과, RAG와 프롬프트 엔지니어링의 결합은 모델 응답을 크게 향상시키고 더 나은 사실적 답변을 생성하는 것으로 나타났습니다. 교육 환경에서 RAG는 모델 입력을 추가 정보와 자료로 풍부하게 하는 것을 기반으로 하기 때문에 이상적인 기술로 보입니다. 일반적으로 대학 강좌에 이미 존재합니다. 반면에 미세 조정은 여전히 강력한 전문가 모델을 만들 수 있지만 과적합 위험이 있습니다. 본 연구는 또한 LLM의 성능을 어떻게 측정하고 현재 측정 방식이 정확성 또는 관련성을 얼마나 잘 나타내는지에 대해 질문합니다. 연구팀은 유사성 지표에서 높은 상관관계와 대부분의 지표에서 짧은 응답에 대한 편향을 발견했습니다. 전반적으로 본 연구는 교육 환경에서 LLM을 통합하는 잠재력과 과제를 모두 지적하며, 균형 잡힌 훈련 접근 방식과 고급 평가 프레임워크의 필요성을 시사합니다.



### GPT-3 Powered Information Extraction for Building Robust Knowledge Bases (https://arxiv.org/abs/2408.04641)
- **What's New**: 이 논문은 GPT-3을 이용하여 지식베이스 구축을 위한 새로운 정보 추출 방법을 제안합니다. 제안된 방법은 비정형 텍스트에서 관련 엔티티와 관계를 추출하여 구조화된 정보를 추출하는 데 따르는 어려움을 해결하려고 시도합니다.

- **Technical Details**: 이 연구는 GPT-3의 컨텍스트 학습(in-context learning)을 사용하여 지식베이스 구축을 위한 정보 추출을 수행합니다. 이 방법은 구조화된 프롬프트(structured prompt)를 생성하고, k-최근접 이웃 모듈(k-nearest neighbor module)을 사용하며, NER 및 RE를 위한 로짓 편향(logit bias)과 컨텍스트 보정(contextual calibration)을 통합합니다.

- **Performance Highlights**: 실험 결과는 GPT-3가 텍스트에서 관련성 있고 정확한 정보를 효율적이고 정확하게 추출할 수 있음을 보여줍니다. 따라서 지식베이스 구축의 정확성과 생산성을 높일 수 있습니다. 또한, 제안된 방법이 기존의 최첨단 정보 추출 기술과 비교하여 경쟁력 있는 결과를 얻을 수 있음을 보여줍니다. 컨텍스트 학습을 사용하여 제한된 수의 샘플만으로도 경쟁력 있는 결과를 얻을 수 있어, 데이터 주석 및 엔지니어링 비용을 상당히 절감할 수 있습니다. 또한, 생물 의학 정보를 추출하여 실제 환경에서의 실용성을 입증했습니다.




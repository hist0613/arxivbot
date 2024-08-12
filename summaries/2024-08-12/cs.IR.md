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




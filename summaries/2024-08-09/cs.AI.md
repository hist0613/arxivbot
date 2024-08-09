New uploads on arXiv(cs.CL)

### Arctic-TILT. Business Document Understanding at Sub-Billion Sca (https://arxiv.org/abs/2408.04632)
- **What's New**: Arctic-TILT: a lightweight document understanding (DU) model that achieves comparable accuracy to models 1000 times its size, enabling cost-effective and efficient processing of visually rich documents. It excels on various benchmarks such as MP-DocVQA, DUDE, and ArXiv-Lay, demonstrating its versatility and performance. Arctic-TILT also provides reliable confidence scores and fast inference, making it suitable for large-scale and time-sensitive applications in enterprise environments.

- **Technical Details**: Arctic-TILT builds upon the TILT encoder-decoder model, introducing a novel modality fusion technique that integrates textual and visual information within each transformer block, inspired by tensor product representations. It incorporates attention sparsity to handle longer documents efficiently by restricting attention calculations to a local neighborhood.  Arctic-TILT is optimized for deployment on single GPUs, enabling efficient processing of documents with up to 400k tokens.

- **Performance Highlights**: Arctic-TILT demonstrates state-of-the-art results on seven diverse document understanding benchmarks, including MP-DocVQA, DUDE, Kleister, ArXiv-Lay, and PubMed-Lay. It achieves comparable accuracy to much larger models, making it a compelling choice for cost-efficient document understanding tasks.



### LogogramNLP: Comparing Visual and Textual Representations of Ancient Logographic Writing Systems for NLP (https://arxiv.org/abs/2408.04628)
- **What's New**: 본 논문은 고대 상형 문자 언어(Logographic language)를 위한 NLP 분석을 위한 새로운 벤치마크인 **LogogramNLP**를 소개합니다. 이 벤치마크는 4개의 고대 문자 체계(Linear A, 이집트 상형 문자, 설형 문자, 죽간 문자)에 대한 전사 및 시각 데이터셋과 분류, 번역, 구문 분석과 같은 작업을 위한 주석을 제공합니다. 또한 최근 시각 및 텍스트 인코딩 전략을 백본으로 사용하는 시스템을 비교합니다. 연구 결과는 시각적 표현이 특정 작업에서 텍스트 표현보다 성능이 뛰어나다는 것을 보여줍니다. 이는 시각적 처리 파이프라인이 NLP 기반 분석을 위한 많은 양의 상형 문자 언어 문화 유산 데이터를 활용할 수 있음을 시사합니다.



### Better Alignment with Instruction Back-and-Forth Translation (https://arxiv.org/abs/2408.04614)
- **What's New**: 본 논문에서는 대규모 언어 모델(LLM) 정렬을 위해 세계 지식에 기반한 고품질 합성 데이터를 구축하는 새로운 방법인 **명령어 쌍방향 번역(instruction back-and-forth translation)**을 제안합니다. 웹 코퍼스에서 문서를 사용하여 Li et al.(2023a)가 제안한 쌍방향 번역 방식을 사용하여 합성 명령어를 생성하고 큐레이션하며, 초기 문서를 기반으로 응답을 다시 작성하여 품질을 향상시킵니다. 생성된 (쌍방향 번역 명령어, 다시 작성된 응답) 쌍으로 미세 조정하면 AlpacaEval에서 Humpback, ShareGPT, Open Orca, Alpaca-GPT4, Self-instruct와 같은 다른 일반적인 명령어 데이터셋을 사용하는 것보다 더 높은 승률을 보입니다. 또한 LLM을 사용하여 응답을 다시 작성하면 직접 증류보다 성능이 뛰어나고, 두 가지 생성된 텍스트 분포는 임베딩 공간에서 상당한 차이를 보입니다. 추가 분석을 통해 쌍방향 번역 명령어가 다른 합성 명령어 소스보다 품질이 높고, 응답은 증류에서 얻은 응답보다 다양하고 복잡하다는 것을 보여줍니다. 전반적으로 쌍방향 번역은 웹에서 발견되는 정보의 다양성과 양을 활용하면서도 효과적인 정렬에 필요한 응답 품질을 보장하여 두 세계의 장점을 결합한다는 것을 발견했습니다.



### Code-switching in text and speech reveals information-theoretic audience design (https://arxiv.org/abs/2408.04596)
Comments:
          Submitted to Journal of Memory and Language on 7 June 2024

- **What's New**: 이 연구는 언어 모델을 사용하여 코드 전환(code-switching)에 영향을 미치는 요인을 조사합니다. 특히, 코드 전환이 단순히 화자 중심(speaker-driven)인지 아니면 청취자 중심(audience-driven)인지 밝히고자 합니다. 연구자들은 중국어-영어 이중언어 온라인 포럼 게시물과 자발적인 중국어-영어 발화 녹음을 분석하여 코드 전환이 정보 부하(information load)가 높은 지점에서 발생하는 경향이 있다는 기존 연구 결과를 재확인했습니다. 그러나 더 놀라운 결과는 영어로 된 발화의 정보 부하가 의미적으로 동일한 중국어 대안보다 더 높다는 사실입니다. 이는 코드 전환이 청취자의 주의를 요구하는 상황에서 발생할 수 있다는 것을 시사합니다. 즉, 코드 전환은 단순히 화자의 언어적 용이성 때문만이 아니라 청취자에게 메시지를 더 명확하게 전달하고자 하는 목적으로 사용될 수 있습니다.

- **Technical Details**: 본 연구는 이중언어 온라인 포럼 게시물과 자발적인 발화 녹음을 분석하여 언어 모델을 사용하여 코드 전환 지점에서 영어와 중국어의 정보 부하를 비교했습니다. 정보 부하 측정은 서프라이절(surprisal)을 기반으로 했습니다. 서프라이절은 특정 단어나 구문이 주어진 문맥에서 얼마나 예상 밖인지를 나타내는 지표입니다. 본 연구에서는 중국어-영어 이중언어 데이터 세트에서 코드 전환 구문과 해당하는 중국어 번역을 분석하여 영어 코드 전환 구문의 서프라이절이 중국어 번역보다 높은지 확인했습니다.

- **Performance Highlights**: 연구 결과, 영어 코드 전환 구문의 서프라이절이 중국어 번역보다 높게 나타났습니다. 이는 코드 전환이 단순히 화자의 언어적 용이성 때문이 아니라 청취자의 주의를 요구하는 상황에서 발생할 수 있다는 것을 시사합니다. 또한, 이러한 패턴은 글쓰기와 말하기라는 두 가지 다른 의사소통 방식에서 모두 확인되었습니다. 따라서 이 연구는 코드 전환이 정보 부하와 연관되어 있으며, 이러한 연관성은 청취자에게 더 명확한 메시지를 전달하려는 목적으로 나타날 수 있다는 것을 보여줍니다.



### Towards Resilient and Efficient LLMs: A Comparative Study of Efficiency, Performance, and Adversarial Robustness (https://arxiv.org/abs/2408.04585)
- **What's New**: 이 연구는 Transformer++ [15], GLA Transformer [11], MatMul-Free LM [12]와 같은 주목할만한 언어 모델에 대한 효율성, 성능 및 적대적 견고성 사이의 상충 관계를 분석하는 새로운 프레임워크를 제시합니다. 특히, 이 연구는 이러한 모델의 적대적 견고성을 광범위하게 조사하는 데 초점을 맞추어 실제 적용에 대한 중요한 통찰력을 제공합니다. 이 연구는 GLUE와 AdvGLUE 데이터셋을 사용하여 세 가지 모델을 다양한 공격 수준에서 비교 평가합니다.

- **Technical Details**: 이 연구는 E-P-R Trade-off Evaluation Framework을 소개하여 LLM의 효율성, 성능 및 적대적 견고성을 평가합니다. 이 프레임워크는 사전 훈련된 LLM을 다양한 언어 작업에 미세 조정하고, Word-level, Sentence-level, Human-level perturbation을 포함한 다양한 적대적 공격을 수행하여 적대적 예제를 생성합니다. 그런 다음 미세 조정된 모델은 원본 및 적대적 샘플을 사용하여 성능과 적대적 견고성을 평가합니다. 또한 각 LLM의 미세 조정 또는 추론 효율성을 평가하여 LLM 기능과 효율성, 성능, 적대적 견고성 사이의 상충 관계에 대한 포괄적인 이해를 제공합니다.

- **Performance Highlights**: GLA Transformer와 MatMul-Free LM은 GLUE 작업에서 Transformer++와 비교하여 높은 효율성과 비교적 우수한 성능을 달성합니다. 또한 GLA Transformer는 모든 공격 수준에서 뛰어난 견고성을 보이는 반면, MatMul-Free LM은 Word-level 공격에 대한 견고성이 뛰어나고 Sentence-level 및 Human-level 공격에 대한 견고성은 Transformer++와 동일합니다. 이러한 결과는 효율적인 아키텍처가 효율성, 성능 및 적대적 견고성 간에 적절한 균형을 이룰 수 있는 가능성을 보여주며, 리소스 제약과 적대적 공격에 대한 회복력이 중요한 응용 프로그램에 귀중한 통찰력을 제공합니다.



### Learning Fine-Grained Grounded Citations for Attributed Large Language Models (https://arxiv.org/abs/2408.04568)
Comments:
          Accepted by ACL 2024 Findings

- **What's New**: 이 논문은 대규모 언어 모델(LLM)이 생성한 텍스트에 인용을 추가하여 환각을 줄이고 검증 가능성을 높이는 새로운 프레임워크인 FRONT를 소개합니다. FRONT는 모델 출력을 세부적인 인용구에 기반으로 하여, 인용의 질을 향상시킬 뿐만 아니라 사용자가 세부적인 검증을 수행할 수 있도록 지원합니다.



### Conversational Prompt Engineering (https://arxiv.org/abs/2408.04560)
- **What's New**: Conversational Prompt Engineering (CPE) is introduced, a user-friendly approach that aids users in constructing personalized prompts for their specific tasks without requiring labeled data or initial prompts.

- **Technical Details**: CPE utilizes a chat model to interact with users, helping them articulate their output preferences and integrating these into the prompt. The process involves two stages: first, the model analyzes user-provided unlabeled data to generate data-driven questions and utilize user responses to shape the initial instruction. Second, the model shares the outputs generated by the instruction and uses user feedback to further refine the instruction and the outputs. The final result is a few-shot prompt, where the outputs approved by the user serve as few-shot examples.

- **Performance Highlights**: A user study on summarization tasks demonstrates the effectiveness of CPE in creating personalized, high-performing prompts. The results indicate that the zero-shot prompt obtained by CPE is comparable to its few-shot counterpart, highlighting significant time savings for repetitive tasks with large text volumes.



### Bias-Aware Low-Rank Adaptation: Mitigating Catastrophic Inheritance of Large Language Models (https://arxiv.org/abs/2408.04556)
Comments:
          Work in progress

- **What's New**: BA-LoRA (Bias-Aware Low-Rank Adaptation) is a novel Parameter-Efficient Fine-Tuning (PEFT) method for mitigating bias propagation during fine-tuning of Large Language Models (LLMs). It aims to improve the consistency, diversity, and generalization capabilities of LLMs while addressing the issue of 'Catastrophic Inheritance' from pre-training data.



### Moly\'e: A Corpus-based Approach to Language Contact in Colonial Franc (https://arxiv.org/abs/2408.04554)
Comments:
          8 main pages and 3 pages of references

- **What's New**: 이 연구는 초기 근대 시대에 발전한 여러 크리올어(Creole languages)가 유럽 언어의 유전적 후손인지 여부에 대한 논쟁에 새로운 시각을 제시합니다. 이는 중간 형태의 증거 부족으로 인해 뜨거운 논쟁거리였습니다. 이 연구는 400년에 걸쳐 유럽의 세 가지 언어 변이 유형의 전형적인 표현과 프랑스 기반 크리올어의 초기 증거를 결합한 새로운 개방형 말뭉치인 Molyé 말뭉치를 소개합니다. 이는 유럽의 접촉 상황과 크리올어 사용 지역(전 식민지) 사이의 연속성에 대한 미래 연구를 촉진하기 위한 것입니다.



### MemeMind at ArAIEval Shared Task: Spotting Persuasive Spans in Arabic Text with Persuasion Techniques Identification (https://arxiv.org/abs/2408.04540)
- **What's New**: 이 논문은 트윗과 뉴스 단락에서 아랍어 텍스트의 선전적 범위(propagandistic spans)와 설득 기법(persuasion techniques)을 감지하는 데 중점을 둡니다. 데이터셋의 각 항목은 텍스트 샘플과 해당 텍스트 내에서 선전 기법의 시작 및 끝 위치를 나타내는 레이블을 포함합니다. 레이블이 지정된 범위 내의 토큰은 특정 선전 기법에 따라 "B"(Begin), "I"(Inside), "O"로 지정되었습니다. 어텐션 마스크(attention masks)를 사용하여 각 범위의 길이를 균일하게 만들고 제공된 레이블을 기반으로 각 토큰에 BIO 태그를 할당했습니다. 그런 다음 아랍어 텍스트 토큰화 및 임베딩(embeddings)을 위한 AraBERT-base 사전 훈련된 모델을 토큰 분류(token classification) 계층과 함께 사용하여 선전 기법을 식별했습니다. 훈련 과정에는 2단계 미세 조정(fine-tuning) 접근 방식이 포함됩니다. 먼저 몇 에포크 동안 분류 계층만 훈련하고, 전체 모델 미세 조정을 통해 모든 매개변수를 업데이트합니다. 이 방법론을 통해 모델은 사전 훈련된 AraBERT 모델에서 캡처된 지식을 활용하면서 선전 감지 작업의 특정 특성에 적응할 수 있습니다. 제안된 접근 방식은 F1 점수 0.2774를 달성하여 Task 1 리더보드에서 3위를 차지했습니다.



### Compromesso! Italian Many-Shot Jailbreaks Undermine the Safety of Large Language Models (https://arxiv.org/abs/2408.04522)
Comments:
          Accepted at ACL 2024 (Student Research Workshop)

- **What's New**: 이 연구는 이탈리아어를 사용하는 대규모 언어 모델(LLM)의 안전성을 조사하여 영어 이외 언어에서 LLM의 안전성에 대한 이해를 넓히는 데 기여합니다.

- **Technical Details**: 이 연구에서는 이탈리아어로 된 418개의 불안전한 질문과 답변 쌍으로 구성된 새로운 데이터셋을 제시합니다. 이 데이터셋을 사용하여 6개의 오픈-웨이트 LLM에서 많은 샷(many-shot) 탈옥(jailbreaking)에 대한 효과를 평가하고, 다양한 샷 수에 대한 불안전한 응답의 확률 변화를 분석합니다.

- **Performance Highlights**: 실험 결과, 모든 모델에서 많은 샷 탈옥이 불안전한 행동을 유발하는 것으로 나타났으며, 샷의 수가 증가함에 따라 불안전한 행동이 증가하는 경향을 보였습니다. 1샷에서 평균 68%였던 불안전한 응답 비율은 32샷에서 84%로 증가했습니다. 이 결과는 다국어 안전 프로토콜의 필요성을 강조합니다.



### Can LLMs Beat Humans in Debating? A Dynamic Multi-agent Framework for Competitive Deba (https://arxiv.org/abs/2408.04472)
Comments:
          9 pages, 3 figures

- **What's New**: 이 논문은 대규모 언어 모델(LLM)을 기반으로 하는 새로운 멀티 에이전트 프레임워크인 Agent4Debate를 소개합니다. Agent4Debate는 경쟁 토론에서 LLM의 능력을 향상시키기 위해 설계되었으며, 인간의 토론 준비 및 실행 행동에서 영감을 받았습니다. 이 프레임워크는 서로 다른 특수화된 에이전트(Searcher, Analyzer, Writer, Reviewer) 간의 협업을 통해 토론 과정 전반에 걸쳐 초기 조사 및 논거 형성부터 반박 및 요약에 이르기까지 다양한 단계를 다룹니다.



### Crowd Intelligence for Early Misinformation Prediction on Social Media (https://arxiv.org/abs/2408.04463)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: CROWDSHIELD, a new method for early misinformation prediction, leverages crowd intelligence, specifically user responses to a post, to assess its credibility.  It analyzes both the stances (support/denial) and the claims made in replies to a post, going beyond just explicit stances.



### AcrosticSleuth: Probabilistic Identification and Ranking of Acrostics in Multilingual Corpora (https://arxiv.org/abs/2408.04427)
- **What's New**: 이 논문은 AcrosticSleuth라는 새로운 도구를 소개합니다. 이 도구는 텍스트에서 숨겨진 메시지(acrostic)를 자동으로 찾아내어, 그 숨겨진 메시지가 우연히 발생했을 가능성이 아닌, 의도적으로 삽입되었을 가능성을 계산하여 순위를 매깁니다. AcrosticSleuth는 텍스트에서 숨겨진 메시지를 찾는 기존의 수작업 방식을 대체하여 효율성을 높이고, 객관적인 분석을 가능하게 합니다.



### Recognizing Emotion Regulation Strategies from Human Behavior with Large Language Models (https://arxiv.org/abs/2408.04420)
Comments:
          Accepted to ACII'24

- **What's New**: This paper presents a novel approach to automatically classifying emotion regulation strategies, a key aspect of affective computing that has been largely unexplored.  The researchers fine-tune large language models (LLMs) such as Llama2-7B and Gemma on prompts generated from the Deep corpus, which contains recordings of human behavior in shame-inducing situations. This approach outperforms previous methods based on Bayesian Networks, achieving a high accuracy (0.84) without requiring post-interaction interviews.



### Enhancing Robustness of Retrieval-Augmented Language Models with In-Context Learning (https://arxiv.org/abs/2408.04414)
Comments:
          10 pages, 2 figures

- **What's New**: 이 연구는 오픈 도메인 질의응답(QA)에서 RALM(Retrieval-Augmented Language Model)의 추론 능력을 향상시키는 컨텍스트 내 학습(In-context Learning) 기반 접근 방식을 소개합니다. 이 방법은 RALM이 검색된 컨텍스트에서 불가능한 상황과 모순되는 정보를 식별하는 능력을 향상시키기 위해 MRC(Machine Reading Comprehension) 데모(cases라고 함)를 통합합니다.  이를 통해 불완전한 검색 시나리오에서 RALM의 견고성을 높입니다.



### Exploring Reasoning Biases in Large Language Models Through Syllogism: Insights from the NeuBAROCO Datas (https://arxiv.org/abs/2408.04403)
Comments:
          To appear in Findings of the Association for Computational Linguistics: ACL 2024

- **What's New**: This paper evaluates the logical reasoning abilities of Large Language Models (LLMs) using the NeuBAROCO dataset, a manually constructed syllogism dataset in both English and Japanese.  This dataset is based on psychological experiments designed to assess human reasoning capabilities using various forms of syllogisms.



### Automated Educational Question Generation at Different Bloom's Skill Levels using Large Language Models: Strategies and Evaluation (https://arxiv.org/abs/2408.04394)
- **What's New**: 이 연구는 5가지 최첨단 대규모 언어 모델(LLM)을 사용하여 블룸의 분류 체계에 따른 다양한 인지 수준의 질문을 생성하는 능력을 조사했습니다. 연구는 LLM이 적절한 정보를 제공받았을 때 다양한 인지 수준의 관련성 있고 고품질의 교육 질문을 생성할 수 있음을 시사하지만, 고려된 5가지 LLM의 성능에는 상당한 차이가 있습니다. 또한 자동 평가는 인간 평가와 동일하지 않다는 것을 보여줍니다.



### Open-domain Implicit Format Control for Large Language Model Generation (https://arxiv.org/abs/2408.04392)
Comments:
          6 pages

- **What's New**: 이 논문은 대규모 언어 모델(LLM)의 제어된 텍스트 생성을 위한 새로운 프레임워크를 제시합니다. 이 프레임워크는 사용자가 제공한 단일 샷(one-shot) 질문 답변(QA) 쌍을 활용하여 모델이 개방형 도메인(open-domain) 형식 제약 사항을 따르도록 합니다. 기존 방식과 달리 규칙 기반 오토마타를 사용한 제약된 디코딩 또는 수동으로 만든 형식 지침을 이용한 미세 조정은 개방형 도메인 요구 사항에 어려움을 겪었습니다.  이 논문은 이러한 문제를 해결하기 위해 단일 샷 QA 쌍을 사용한 새로운 프레임워크를 제안합니다.



### Overview of the NLPCC 2024 Shared Task on Chinese Metaphor Generation (https://arxiv.org/abs/2408.04378)
- **What's New**: NLPCC 2024에서 개최된 중국어 은유 생성 공유 과제의 결과를 발표합니다. 이 과제는 두 가지 하위 작업으로 나뉘며, 1) 은유 생성: 주제(TENOR), 근거(GROUND), 운반체(VEHICLE)로 구성된 튜플에서 은유를 생성하는 작업. 2) 은유 구성 요소 식별: 은유 문장에서 가장 적합한 TENOR, GROUND, VEHICLE을 추출하는 작업. 이 공유 과제는 총 4개의 참여팀을 유치했습니다.

- **Technical Details**: 이 공유 과제는 중국어 은유 생성 및 이해를 위한 혁신적인 모델 개발과 고품질 데이터 세트 구축을 촉진하는 데 초점을 맞추고 있습니다. 두 하위 작업은 모두 대규모 언어 모델(LLM) 또는 규칙 기반 방법을 사용하여 수행됩니다. 은유 생성 작업은 LLM이 제공된 TENOR, GROUND, VEHICLE을 사용하여 은유 문장을 생성하는 작업이며, 은유 구성 요소 식별 작업은 LLM이 은유 문장에서 각 구성 요소를 식별하는 작업입니다. 이 작업은 중국어 NLP 코퍼스(corpus)와 벤치 마크(benchmark)의 발전과 더불어 진행되고 있습니다.

- **Performance Highlights**: 공유 과제에는 총 4개의 팀이 참가하여 32개의 제출물을 제출했습니다. 각 팀은 서로 다른 접근 방식을 사용하여 두 하위 작업을 수행했습니다. KangGreen팀은 중국어 텍스트 생성 능력으로 유명한 Yi-1.5-9b-chat 모델을 사용했습니다. Tencent팀은 Transformer 기반 모델을 사용했으며, Meta팀은 규칙 기반 시스템과 LLM을 결합했습니다.

- **Datasets**: 데이터 세트는 Shao et al. [14]의 연구에서 파생된 은유 훈련 데이터 세트와 경쟁을 위해 사용되는 자체적으로 큐레이팅된 중국어 은유 문장으로 구성됩니다. 자체 큐레이팅된 데이터 세트는 검색 엔진과 GPT-4를 통해 수집됩니다. 총 1,500개의 은유 문장이 수집되었으며, GPT-4를 사용하여 카테고리화되었습니다. 각 데이터 세트는 500개의 질문으로 구성된 세 부분으로 나뉘며, 검증을 위해 500개의 객관식 예제와 정답이 제공됩니다.

- **Evaluation**: 공유 과제의 평가는 두 부분으로 나뉩니다. 첫 번째 부분은 500개의 질문이 포함된 Test A이며, 정답은 제공되지 않습니다. 두 번째 부분은 500개의 질문이 포함된 Test B이며, 참가자에게는 공개되지 않습니다. 참가자의 제출 코드를 사용하여 이러한 숨겨진 질문을 평가합니다.

- **Future Directions**: 이 공유 과제는 중국어 은유 생성 및 이해 연구의 발전에 기여할 것으로 예상됩니다. 앞으로 더 많은 팀이 참여하고 더욱 혁신적인 모델이 개발될 것으로 기대됩니다.



### Analyzing Consumer Reviews for Understanding Drivers of Hotels Ratings: An Indian Perspectiv (https://arxiv.org/abs/2408.04369)
Comments:
          This is the pre-print of the paper that was accepted for oral presentation and publication in the proceedings of IEEE ICCCNT 2024 which was organized as IIT Mandi, India from June 24 to 28, 2024. The paper is 5 pages long and it contains 4 figures and 6 tables. The is not the final version of the paper

- **What's New**: 본 연구는 인도 호텔에 대한 고객 리뷰를 분석하여 최종 평점에 영향을 미치는 주요 요소를 파악하는 데 중점을 둡니다. 웹 스크래핑(web scraping)을 통해 데이터를 수집하고, 잠재 디리클레 할당(Latent Dirichlet Allocation, LDA)을 사용하여 주제 추출(topic extraction)을 수행하며, 측면별 감정 분석(aspect-specific sentiment analysis)을 통해 측면별 감정 매핑(sentiment mapping)을 수행합니다. 마지막으로 랜덤 포레스트(Random Forest)를 활용하여 사용자의 최종 평점을 예측하는 데 있어 측면의 중요성을 파악합니다.



### Enhancing Journalism with AI: A Study of Contextualized Image Captioning for News Articles using LLMs and LMMs (https://arxiv.org/abs/2408.04331)
- **What's New**: 본 연구는 대규모 언어 모델(LLM)과 대규모 멀티모달 모델(LMM)이 뉴스 기사에 첨부된 이미지에 대한 문맥화된 캡션을 생성하여 저널리즘 실무를 지원할 수 있는 방법을 탐구합니다. 저널리즘 분야에서 AI를 통합하는 것은 뉴스 보도의 품질과 효율성을 향상시키는 데 있어 독특한 과제와 기회를 제시합니다.



### Trans-Tokenization and Cross-lingual Vocabulary Transfers: Language Adaptation of LLMs for Low-Resource NLP (https://arxiv.org/abs/2408.04303)
Comments:
          Accepted at COLM 2024

- **What's New**: 이 연구는 저자원 언어 모델을 위한 혁신적인 **교차 언어 어휘 전이**(cross-lingual vocabulary transfer) 전략인 **트랜스 토크나이제이션**(trans-tokenization)을 소개합니다. 이 전략은 고자원 언어 모델을 낮은 자원을 가진 언어로 효율적으로 적응시키는 데 초점을 맞춥니다. 기존의 어휘 전이 방식의 한계를 극복하고 다양한 언어 간의 모델 적응 가능성을 확장하는 것을 목표로 합니다.



### Are Social Sentiments Inherent in LLMs? An Empirical Study on Extraction of Inter-demographic Sentiments (https://arxiv.org/abs/2408.04293)
- **What's New**: 본 연구는 대규모 언어 모델(LLM)에서 특정 사회 집단 간의 감정을 추출하고 검증하는 새로운 방법을 제시합니다. LLM은 대량의 텍스트 데이터를 학습하여 사회적 상식과 편견을 포함한 무의식적인 인간 지식과 감정을 습득하는 것으로 알려져 있지만, 다양한 LLM에서 특정 사회 집단의 감정이 얼마나 포착될 수 있는지는 불분명했습니다. 이 연구는 국적, 종교, 인종/민족으로 정의된 사회 집단에 초점을 맞추어 LLM에서 사회 집단 간의 감정을 추출하고 검증했습니다.



### EMTeC: A Corpus of Eye Movements on Machine-Generated Texts (https://arxiv.org/abs/2408.04289)
- **What's New**: This paper introduces the Eye Movements on Machine-Generated Texts Corpus (EMTeC), a new corpus of eye-tracking data collected from 107 native English speakers reading machine-generated texts.

- **Technical Details**: EMTeC features eye movement data at all processing stages, including raw coordinate data, fixation sequences, and reading measures. It also includes the corrected versions of fixation sequences to account for calibration drift and the internal workings of the language models used to generate the texts. The corpus includes stimuli annotated for various linguistic features, both at the text and word levels.

- **Performance Highlights**: This corpus is expected to be used in various applications, including investigating reading behavior on machine-generated text, evaluating the impact of different decoding strategies, analyzing reading behavior across different text types, and developing new pre-processing, data filtering, and drift correction algorithms. It can also be used for enhancing the cognitive interpretability of language models, assessing the predictive power of surprisal and entropy for human reading times, and developing new data-driven models of human eye movements.



### LLM-DetectAIve: a Tool for Fine-Grained Machine-Generated Text Detection (https://arxiv.org/abs/2408.04284)
- **What's New**: 이 논문은 기존의 텍스트 생성 감지 시스템과 달리 텍스트를 **네 가지 카테고리** (인간 작성, 기계 생성, 기계 작성 기계 수정, 인간 작성 기계 수정)로 분류하는 **LLM-DetectAIve** 시스템을 제안합니다. 이는 AI가 텍스트 생성에 개입하는 **정도**를 파악하여 교육, 학술 분야에서의 공정성 및 신뢰성을 높이는 데 도움을 줍니다.



### LaDiMo: Layer-wise Distillation Inspired MoEfier (https://arxiv.org/abs/2408.04278)
Comments:
          21 pages, 10 figures

- **What's New**: 이 연구에서는 기존의 Transformer 기반 모델을 최소한의 추가 학습 비용으로 MoE 모델로 효율적으로 변환하는 새로운 알고리즘인 LaDiMo를 제안합니다. LaDiMo는 계층별 전문가 구성과 라우팅 정책 결정의 두 단계로 구성됩니다. 지식 증류(Knowledge Distillation) 개념을 활용하여 모델을 압축하고 빠르게 성능을 회복합니다. 또한, 라우팅 가중치 분포를 프로파일링하고 정확도와 지연 시간의 균형을 맞추는 계층별 정책을 결정하여 추론 효율성을 최적화하는 적응형 라우터를 개발합니다. LLaMA2-7B 모델을 100K 토큰만 사용하여 MoE 모델로 변환하여 활성화된 파라미터를 20% 이상 줄이면서 정확도를 유지하는 방법의 효과를 보여줍니다. 이 방법은 MoE 모델을 구축하고 배포하기 위한 유연하고 효율적인 솔루션을 제공합니다.



### Analysis of Argument Structure Constructions in the Large Language Model BER (https://arxiv.org/abs/2408.04270)
Comments:
          arXiv admin note: text overlap with arXiv:2408.03062

- **What's New**: 본 연구는 기존 LSTM 분석을 확장하여 BERT가 Argument Structure Construction(ASC)을 어떻게 처리하고 표현하는지 조사합니다. 4가지 ASC 유형 (타동사, 이중 타동사, 원인-운동, 결과적)에 걸쳐 2,000개 문장의 데이터셋을 사용하여 12개 층에 걸쳐 BERT의 토큰 임베딩을 분석했습니다. MDS와 t-SNE를 사용한 시각화와 GDV(Generalized Discrimination Value)에 의해 정량화된 클러스터링을 사용했습니다. 피드포워드 분류기(프로브)는 임베딩에서 구문 범주를 예측했습니다. CLS 토큰 임베딩은 2-4 층에서 가장 잘 클러스터링되었고, 중간 층에서는 감소했으며, 최종 층에서는 약간 증가했습니다. DET 및 SUBJ 임베딩은 중간 층에서 일관된 클러스터링을 보였으며, VERB 임베딩은 1층에서 12층으로 클러스터링이 증가했고, OBJ 임베딩은 10층에서 최고점에 도달했습니다. 프로브 정확도는 1층에서 낮은 구문 정보를 나타냈으며, 2층 이후로 90% 이상의 정확도를 보여 GDV 클러스터링을 넘어선 잠재적 구문 정보를 드러냈습니다. 어텐션 가중치에 대한 FDR(Fisher Discriminant Ratio) 분석은 OBJ 토큰이 VERB 및 DET 토큰에 이어 ASC를 구분하는 데 중요했음을 보여주었습니다. SUBJ, CLS 및 SEP 토큰은 FDR 점수가 무시할 만했습니다. 본 연구는 BERT의 언어 구문의 계층 처리 및 LSTM과의 차이점을 강조합니다. 미래 연구는 이러한 발견을 신경 영상 데이터와 비교하여 ASC 처리의 신경 상관 관계를 이해하는 데 중점을 둘 것입니다. 본 연구는 신경 언어 모델이 인간 두뇌의 언어 처리를 모방할 수 있는 잠재력을 강조하고, 언어 이해를 기반으로 하는 계산 및 신경 메커니즘에 대한 통찰력을 제공합니다.



### EfficientRAG: Efficient Retriever for Multi-Hop Question Answering (https://arxiv.org/abs/2408.04259)
Comments:
          20 pages, 4 figures

- **What's New**: This paper introduces EfficientRAG, an effective retriever for multi-hop question answering that iteratively generates new queries without relying on multiple LLM calls.

- **Technical Details**: EfficientRAG consists of two components: a Labeler & Tagger and a Filter. The Labeler annotates relevant tokens in retrieved chunks, while the Tagger determines whether the chunk is helpful or irrelevant. The Filter constructs new queries for subsequent retrieval rounds by replacing unknown parts with labeled tokens. This approach avoids the need for LLM calls during query generation.

- **Performance Highlights**: Experimental results show that EfficientRAG outperforms existing RAG methods on three open-domain multi-hop question-answering datasets (HotpotQA, 2Wiki-multihop (2WikiMQA), and MuSiQue). The study also highlights the efficiency of EfficientRAG in terms of chunk retrieval compared to one-time and iterative query decomposition approaches.

- **Training Details**: EfficientRAG is trained on synthetic data generated by LLMs through multi-hop question decomposition, token labeling, next-hop question filtering, and negative sampling.

- **Data Generation Process**: The synthetic data generation involves prompting LLMs to decompose multi-hop questions into single-hop questions, label important words in chunks, generate next-hop questions, and select negative samples.

- **Architecture**: EfficientRAG utilizes an auto-encoder language model for token embedding and a fully connected layer for classification.

- **Key Contributions**: EfficientRAG introduces an efficient iterative retrieval method for multi-hop question answering by leveraging a small model for query generation, reducing reliance on LLMs and improving efficiency.



### Explicating the Implicit: Argument Detection Beyond Sentence Boundaries (https://arxiv.org/abs/2408.04246)
Comments:
          9 pages, ACL 2024

- **What's New**: 이 논문은 기존의 문장 수준의 작업으로 모델링된 술어 단어의 의미적 인수를 감지하는 것을 문장 경계를 넘어 의미적 관계를 포착하기 위해 텍스트적 추론을 통해 문제를 재구성합니다. 이 방법은 통과에서 추론할 수 있는지 여부를 테스트하기 위해 먼저 간단하고 독립적인 명제로 인코딩한 다음 통과에 대한 추론을 테스트하여 전체 통과에서 일부 의미적 관계를 추론할 수 있는지 여부를 테스트합니다. 이 방법은 데이터 세트 부족으로 인해 일반적으로 부재하는 직접적인 감독을 요구하지 않고 대신 기존 NLI 및 문장 수준 SRL 리소스를 기반으로 합니다.



### Learning to Rewrite: Generalized LLM-Generated Text Detection (https://arxiv.org/abs/2408.04237)
- **What's New**: 본 논문은 LLM(Large Language Model)이 생성한 텍스트를 감지하기 위해  LLM을 훈련하는 새로운 방법인 L2R(Learning to Rewrite)을 제안합니다. L2R은 LLM이 인간이 작성한 텍스트를 다시 작성할 때는 최대한 많은 수정을 가하고, LLM이 생성한 텍스트를 다시 작성할 때는 최소한의 수정을 가하도록 훈련합니다. 이를 통해  LLM이 생성한 텍스트와 인간이 작성한 텍스트를 구별하는 더욱 명확하고 일반화 가능한 편집 거리 차이를 생성할 수 있습니다.



### Evaluating Language Model Math Reasoning via Grounding in Educational Curricula (https://arxiv.org/abs/2408.04226)
Comments:
          30 pages, 23 figures

- **What's New**: 본 논문은 언어 모델(LM)의 수학 능력을 평가하는 새로운 방식을 제시하며, LM이 수학 내용으로 가능하게 되는 기술과 개념을 구별할 수 있는지 여부를 조사합니다. 본 논문에서는 Achieve the Core(ATC)에서 제공되는 385개의 세분화된 K-12 수학 기술 및 개념 설명(표준)으로 구성된 데이터 세트와 이러한 표준으로 레이블된 9,900개의 문제를 포함하는 MathFish라는 두 개의 데이터 세트를 제공합니다. 경험이 풍부한 교사들과 협력하여, LM이 문제에 연결된 표준을 태깅하고 검증하는 데 어려움을 겪고 있으며, 대신 실제 값에 가깝지만 미묘한 차이가 있는 레이블을 예측한다는 사실을 발견했습니다. 또한, LM이 종종 프롬프트에 설명된 표준과 완전히 일치하지 않는 문제를 생성한다는 사실도 보여줍니다. 마지막으로, GSM8k의 문제를 수학 표준을 사용하여 분류하여 모델이 특정 문제를 다른 문제보다 푸는 데 어려움을 겪는 이유를 더 잘 이해할 수 있도록 합니다.



### Diffusion Guided Language Modeling (https://arxiv.org/abs/2408.04220)
Comments:
          ACL Findings 2024

- **What's New**: This paper presents a novel framework, Diffusion Guided Language Modeling (DGLM), which combines the fluency of auto-regressive language models with the flexibility of continuous diffusion models for controllable text generation. DGLM uses a diffusion network to generate soft prompts (semantic proposals) that guide an auto-regressive model to generate text with desired properties.

- **Technical Details**: DGLM uses a diffusion model to generate continuous semantic proposals that act as soft prompts for an auto-regressive language model. During pre-training, the language decoder is conditioned on embedded representations of the ground truth continuation, teaching it to interpret the semantic proposals. During inference, the diffusion model generates its own proposal guided by a simple classifier to ensure desired attributes. The decoder then uses these proposals to generate fluent text with the desired attributes.

- **Performance Highlights**: DGLM outperforms existing plug-and-play methods in controllable text generation across various benchmark datasets. Controlling a new attribute only requires training a simple logistic regression classifier, making it highly flexible for different use cases.

- **Advantages**: DGLM offers the following advantages:
 - Decoupled model training and attribute control.
 - Easy control of new attributes using a simple logistic regression classifier.
 - High effectiveness and outperformance of existing plug-and-play methods.

- **Keywords**: Diffusion Models, Autoregressive Language Models, Controllable Text Generation, Plug-and-Play Guidance



### Simplifying Translations for Children: Iterative Simplification Considering Age of Acquisition with LLMs (https://arxiv.org/abs/2408.04217)
Comments:
          Findings of ACL 2024

- **What's New**: 본 연구는 사용자의 언어 수준에 맞춰 번역의 난이도를 조절하는 새로운 방법을 제시합니다. 특히, 어린이들이 번역 내용을 제대로 이해하지 못할 수 있는 문제를 해결하기 위해 노력합니다. 이 방법은 번역된 문장에서 높은 AoA(Age of Acquisition, 습득 연령) 단어를 더 간단한 단어로 대체하는 기술을 사용합니다. 이를 위해 대규모 언어 모델(LLM)을 사용하여 번역 문장, 원문, 대체 대상 단어를 입력으로 받아 처리합니다.

- **Technical Details**: 본 연구는 Simple English Wikipedia를 활용하여 백 번역(back-translation) 기법을 적용하여 새로운 벤치마크 데이터 세트를 구축했습니다. 이 데이터 세트는 번역된 문장과 해당하는 간단한 참조 문장으로 구성됩니다. 제안된 방법은 LLM을 활용하여 고 AoA 단어를 반복적으로 저 AoA 단어로 대체하는 기술을 사용합니다. 이 과정은 원문 맥락을 고려하여 대상 단어뿐만 아니라 주변 단어까지 함께 변경하여 문장 전체를 간단하게 만드는 효과를 냅니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존의 다른 방법들(MUSS, AoA-constrained NMT, LLM-based translation)에 비해 BLEU 점수가 가장 높았고, 동시에 단순화 품질도 유지했습니다. 특히, 높은 AoA 단어를 낮은 AoA 단어로 효과적으로 대체하는 것을 확인했습니다. 본 연구는 AoA를 기반으로 한 텍스트 단순화를 위한 데이터 세트를 자동으로 생성하는 방법을 제시했고, LLM을 활용한 단순화 기술이 기존 방법보다 번역과 단순화 측면에서 모두 뛰어난 성능을 보이는 것을 입증했습니다.



### Attention Mechanism and Context Modeling System for Text Mining Machine Translation (https://arxiv.org/abs/2408.04216)
- **What's New**: 본 논문은 Transformer 패러다임을 기반으로 하는 새로운 아키텍처 스키마를 제시하며, K-means 클러스터링 알고리즘을 혁신적으로 통합하여 스키마의 맥락 이해 능력을 향상시킵니다. Transformer 모델은 병렬 컴퓨팅 성능과 멀티 헤드 어텐션 메커니즘 덕분에 기계 번역 작업에서 우수한 성능을 보여줍니다. 그러나 복잡한 언어 구조를 다룰 때 맥락적 모호성을 경험하거나 지역적 특징을 무시할 수 있습니다. 이러한 제약을 해결하기 위해 본 논문은 입력 텍스트의 어휘 및 관용구를 계층화하는 데 사용되는 K-Means 알고리즘을 통합하여 언어의 지역적 구조와 맥락적 지능을 더 잘 식별하고 보존할 수 있도록 합니다. 이러한 조합의 장점은 K-Means가 텍스트에서 번역 품질과 직접 관련될 수 있는 토픽 또는 개념 영역을 자동으로 발견할 수 있다는 것입니다. 따라서 본 논문에서 고안된 스키마는 Transformer 이전에 K-Means를 준비 단계로 사용하고 멀티 헤드 어텐션 가중치를 재조정하여 유사한 의미나 기능을 가진 어휘 및 관용구를 구별하는 데 도움을 줍니다. 이는 스키마가 위치적 지능에만 집중하는 것이 아니라 학습 단계 동안 이러한 클러스터에 내재된 맥락적 지능에 더 큰 관심을 기울이도록 보장합니다.



### MMREC: LLM Based Multi-Modal Recommender System (https://arxiv.org/abs/2408.04211)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)과 딥러닝 기법을 활용하여 추천 시스템을 개선하는 새로운 접근 방식을 제시합니다. 이 프레임워크는 다중 모달 정보 처리와 통합된 잠재 공간 표현을 통해 추천의 정확성과 관련성을 향상시키는 것을 목표로 합니다.

- **Technical Details**: 이 프레임워크는 LLM을 통해 텍스트와 이미지와 같은 다중 모달 정보를 효율적으로 추출하고 통합하며, 다양한 모달리티를 잠재 공간에 통합하여 순위 모델의 학습 과정을 단순화합니다. 이 연구에서는 LLM이 추천 컨텍스트에서 자연어 데이터를 더 잘 이해하고 활용하여 이전 방법의 한계를 해결할 수 있는 잠재력을 살펴봅니다.

- **Performance Highlights**: 실험 결과는 다중 모달 정보를 사용할 때 모델의 차별적 성능이 향상됨을 보여줍니다. 특히 불균형 데이터셋의 경우 오탐률을 개선하는 데 효과적입니다. 이 연구는 LLM과 다중 모달 데이터 통합이 더욱 개인화되고 맥락적으로 관련성이 높은 추천을 만드는 데 기여할 수 있음을 보여줌으로써 진화하는 추천 시스템 분야에 기여합니다.



### wav2graph: A Framework for Supervised Learning Knowledge Graph from Speech (https://arxiv.org/abs/2408.04174)
Comments:
          Preprint, 32 pages

- **What's New**: wav2graph는 음성 데이터에서 지식 그래프(KG)를 학습하기 위한 최초의 프레임워크로, 지식 추출을 자동화하고 음성 기반 정보를 다양한 AI 애플리케이션에 통합합니다. 이 프레임워크는 음성 인식(ASR) 전사본을 사용하여 KG를 구축하고, 최첨단 그래프 신경망(GNN)을 통해 노드 분류 및 링크 예측을 수행합니다.



### mbrs: A Library for Minimum Bayes Risk Decoding (https://arxiv.org/abs/2408.04167)
- **What's New**: 이 논문은 MBR(Minimum Bayes Risk, 최소 베이즈 위험) 디코딩을 위한 mbrs라는 오픈 소스 라이브러리를 소개합니다. 이 라이브러리는 다양한 지표와 알고리즘을 구현하여 연구자와 개발자가 MBR 디코딩 방법을 비교하고 새로운 방법을 개발할 수 있도록 돕습니다.



### Semantics or spelling? Probing contextual word embeddings with orthographic nois (https://arxiv.org/abs/2408.04162)
- **What's New**: 이 연구는 사전 훈련된 언어 모델(PLM)에서 생성된 문맥적 단어 임베딩(CWE)의 견고성을 조사하며, 이러한 임베딩이 단어 수준 의미를 제대로 포착하지 못할 수 있음을 밝혀냈습니다. 즉, PLM은 입력 데이터의 미세한 철자 오류에 매우 민감하며, 이는 특히 단일 토큰으로 표현되는 단어에서 더욱 두드러지게 나타납니다.



### UNLEARN Efficient Removal of Knowledge in Large Language Models (https://arxiv.org/abs/2408.04140)
Comments:
          11 pages, 2 Figures

- **What's New**: This paper introduces **UNLEARN**, a novel algorithm for removing knowledge from Large Language Models (LLMs) without affecting related knowledge. UNLEARN leverages subspace methods to identify and target specific knowledge for removal.

- **Technical Details**: UNLEARN works in three main steps: 1. **Subspace Identification**: training a task-dependent matrix for each layer to capture the subspace of that specific knowledge. 2. **Subspace Discrimination**: using a Gram-Schmidt process to orthogonalize subspaces and prevent information loss in similar tasks. 3. **Task Removal**: subtracting the modified task matrix to eliminate the targeted knowledge.

- **Performance Highlights**: UNLEARN achieves **96% forgetting** on the target task while maintaining performance on dissimilar tasks within 2.5% of the original model. It also demonstrates **80% forgetting** on similar tasks while preserving performance within 10%. The paper also introduces **LEARN**, a dual method for targeted knowledge addition that achieves similar performance to fine-tuning techniques like LoRA.

- **Limitations**: UNLEARN's effectiveness might be limited by the presence of highly overlapping tasks, where separating subspaces can be challenging.

- **Future Directions**: The paper highlights potential for future research in areas such as: 1.  Improving the efficiency of subspace identification and discrimination. 2. Investigating the impact of UNLEARN on various LLM architectures and tasks. 3. Exploring applications of UNLEARN for privacy-preserving LLM deployment.



### Enhancing Healthcare through Large Language Models: A Study on Medical Question Answering (https://arxiv.org/abs/2408.04138)
Comments:
          received by IEEE ICPICS

- **What's New**: 본 논문은 MedQuAD 의료 질문 답변 데이터 세트에 대한 다양한 LLM (Large Language Model)을 훈련하고 가장 효과적인 모델을 식별하는 연구를 소개합니다. 이 연구는 Sentence-t5와 Mistral 7B의 조합이 0.762의 정밀도 점수를 달성하여 우수한 성능을 보였음을 밝혀냈습니다.

- **Technical Details**: 본 논문은 Gemma 2b + LoRA, Phi-2, Sentence-t5 + Mistral 7B를 포함한 세 가지 모델의 훈련과 미세 조정(fine-tuning)을 자세히 설명합니다. 모델은 데이터 전처리, 프롬프트 생성, 미세 조정 등 엄격한 프로세스를 거쳤습니다.

- **Performance Highlights**: Sentence-t5 + Mistral 7B 모델은 최고의 정밀도를 달성하여 환자 교육 및 지원을 향상시키는 데 유망한 후보임을 입증했습니다. 이 모델은 고급 사전 훈련 기술, 강력한 아키텍처, 효과적인 프롬프트 생성 방법론을 통해 향상된 기능을 제공합니다.



### Can Rule-Based Insights Enhance LLMs for Radiology Report Classification? Introducing the RadPrompt Methodology (https://arxiv.org/abs/2408.04121)
Comments:
          Accepted at BioNLP, ACL 2024

- **What's New**: 이 연구는 **RadPert** (룰 기반 시스템)과 **RadPrompt** (멀티턴 프롬프팅 전략)을 소개하며, 이를 통해 의료 이미지 분석에서 **원격 감독**(distant supervision)을 개선하고 **대규모 언어 모델**(LLM)의 성능을 향상시키는 새로운 방법을 제시합니다.

- **Technical Details**: RadPert는 **RadGraph 지식 그래프**와 함께 **불확실성 인식 정보 스키마**를 통합하여 룰 기반 시스템의 견고성을 높였습니다. RadPrompt는 RadPert를 활용하여 **LLM의 제로 샷 예측 능력**을 강화하는 **멀티턴 프롬프팅 전략**입니다.

- **Performance Highlights**: RadPert는 기존 룰 기반 SOTA인 **CheXpert**를 능가하는 성능을 보였습니다. 또한, RadPrompt는 **GPT-4 Turbo**와 **RadPert** 모두를 뛰어넘는 성능을 달성하여 LLM과 룰 기반 모델의 상호작용 가능성을 입증했습니다.



### Zero-shot Factual Consistency Evaluation Across Domains (https://arxiv.org/abs/2408.04114)
- **What's New**: 이 논문은 텍스트 생성 시스템에서 사실 일관성을 평가하는 새로운 방법을 제시합니다. 자연어 추론(NLI), 요약 평가, 사실 검증, 사실 일관성 평가 등 다양한 작업을 통합하여 다양한 영역에서 소스-타겟 쌍의 사실 일관성을 평가할 수 있는 모델을 학습시킵니다. 22개의 데이터셋으로 구성된 포괄적인 벤치마크 세트에 대해 8가지 기준과 엄격하게 비교 평가하여, 다양한 작업, 영역 및 문서 길이를 포함합니다. 결과는 제시된 방법이 이러한 이종 벤치마크에서 최첨단 성능을 달성하면서 효율성 문제를 해결하고 도메인 간 일반화를 달성함을 보여줍니다.



### Human Speech Perception in Noise: Can Large Language Models Paraphrase to Improve It? (https://arxiv.org/abs/2408.04029)
Comments:
          Accepted at HuCLLM @ ACL 2024

- **What's New**: 본 연구는 대규모 언어 모델(LLM)을 활용하여 소음 환경에서 인간의 음성 인식을 개선하기 위한 새로운 과제인 '소음 속 음성 인식 개선을 위한 문장 바꿔 말하기'(PI-SPiN)를 제시합니다. 기존의 문장 바꿔 말하기는 의미는 유지하면서 표현 방식을 바꾸는 데 초점을 맞추었지만, PI-SPiN은 의미를 유지하면서 소음 환경에서 음성으로 인식하기 더 쉬운 문장을 생성하는 데 목표를 두고 있습니다.



### Improving Large Language Model (LLM) fidelity through context-aware grounding: A systematic approach to reliability and veracity (https://arxiv.org/abs/2408.04023)
Comments:
          14 pages

- **What's New**: 본 논문에서는 자연어 처리(NLP) 애플리케이션에서 점점 더 정교하고 널리 사용되는 대규모 언어 모델(LLM)의 견고성, 신뢰성 및 인간 가치와의 정렬을 보장하는 데 초점을 맞춘 새로운 텍스트 모델의 문맥 기반 접근 방식을 제시합니다. 이 접근 방식은 특히 문맥 표현 단계에 중점을 두어 포괄적인 문맥 인식 방법론을 통해 모델의 신뢰성과 윤리적 정렬을 향상시키는 것을 목표로 합니다. 상황, 문화 및 윤리적 문맥을 기계 판독 가능한 형식으로 명시적으로 캡처하고 표현함으로써 모델의 동작을 이러한 문맥 내에서 고정하는 기반을 마련합니다. 이 접근 방식은 온톨로지, 의미 웹 기술 및 논리 기반 공식과 같은 지식 표현 및 추론 기술을 활용합니다. 연구자들은 실제 텍스트 데이터 세트에서 프레임워크를 평가하여 높은 정확성을 유지하면서 모델 성능, 공정성 및 인간의 기대와의 정렬을 향상시키는 데 효과적임을 입증했습니다. 또한, 연구자들은 문맥 인식 인코딩, 문맥 인식 학습, 해석 가능성 및 설명 가능성, 지속적인 모니터링 및 적응을 포함한 프레임워크의 다른 핵심 구성 요소에 대해 논의합니다. 이 연구는 책임감 있는 AI에 대한 점점 더 많은 작업에 기여하여 더 신뢰할 수 있고, 신뢰할 수 있으며, 윤리적으로 정렬된 언어 모델을 개발하는 실용적인 접근 방식을 제공합니다. 연구 결과는 문맥 이해가 가장 중요한 의료, 법률 시스템 및 사회 서비스와 같은 민감한 분야에서 LLM의 배포에 중요한 의미를 갖습니다.



### Image-to-LaTeX Converter for Mathematical Formulas and Tex (https://arxiv.org/abs/2408.04015)
Comments:
          4 pages

- **What's New**: 이 연구는 이미지에서 LaTeX 코드를 생성하는 비전 인코더-디코더 모델을 훈련합니다. 이 모델은 컴퓨터 생성 이미지로 학습된 Swin Transformer 인코더와 GPT-2 디코더로 구성된 기본 모델과 손으로 쓴 수식으로 미세 조정된 LoRA(Low-Rank Adaptation) 모델을 포함합니다.



### Impacts of Anthropomorphizing Large Language Models in Learning Environments (https://arxiv.org/abs/2408.03945)
Comments:
          Presented at Affective Computing Pre-Conference at ISRE 2024

- **What's New**: 본 연구는 교육 환경에서 사용되는 대규모 언어 모델(LLM)의 의인화(anthropomorphism)가 학습자에게 미치는 영향, 특히 학습 결과에 대한 감정적 영향을 조사합니다. LLM 기반 챗봇이 학습 도구로 점점 더 많이 사용되면서, LLM 기반 챗봇에 대한 의인화가 학습자의 감정에 미치는 영향을 이해하는 것이 중요합니다.



### Transformer Explainer: Interactive Learning of Text-Generative Models (https://arxiv.org/abs/2408.04619)
Comments:
          To be presented at IEEE VIS 2024

- **What's New**: Transformer Explainer는 비전문가가 트랜스포머(Transformer) 모델을 이해할 수 있도록 돕는 새로운 웹 기반 시각화 도구입니다. GPT-2 모델을 사용하여 트랜스포머 아키텍처에 대한 높은 수준의 개요와 수학적 연산 및 모델 구조의 저수준 세부 사항을 볼 수 있습니다. 사용자는 자신의 텍스트 입력을 사용하여 실시간으로 GPT-2 모델을 실행하고 모델이 다음 토큰을 예측하는 방식을 관찰할 수 있습니다.



### SCENE: Evaluating Explainable AI Techniques Using Soft Counterfactuals (https://arxiv.org/abs/2408.04575)
Comments:
          10 pages, 5 tables

- **What's New**: SCENE (Soft Counterfactual Evaluation for Natural language Explainability)는 자연어 처리 (NLP) 작업에서 AI 모델의 투명성과 책임성을 높이기 위한 새로운 평가 방법입니다. SCENE은 대규모 언어 모델 (LLM)을 활용하여 제로 샷 방식으로 Soft Counterfactual 설명을 생성합니다. SCENE은 토큰 기반 치환에 초점을 맞춰 광범위한 미세 조정 없이 문맥적으로 적절하고 의미적으로 의미있는 Soft Counterfactuals를 생성합니다. SCENE은 Validitysoft와 Csoft 지표를 채택하여 텍스트 분류 작업에서 모델 독립적인 XAI 방법의 효율성을 평가합니다. SCENE은 CNN, RNN 및 BERT 아키텍처에 적용되어 다양한 XAI 기법의 강점과 한계에 대한 귀중한 통찰력을 제공합니다.



### Articulatory Configurations across Genders and Periods in French Radio and TV archives (https://arxiv.org/abs/2408.04519)
Comments:
          accepted to InterSpeech 2024, Kos Island, Greece keywords : acoustic to articulatory inversion, diachrony, gender, French, media

- **What's New**: 이 논문은 1955년부터 2015년까지 60년 동안의 프랑스 미디어 아카이브를 기반으로 한 통시적 코퍼스를 사용하여 성별과 시대에 따른 발성 구조 변화를 분석합니다. 음향 매개변수에서 발성 매개변수로의 역변환을 통해 이러한 변화를 연구합니다. 자동 전사와 강제 정렬을 통해 각 모음의 중심 프레임을 추출하고, 성별과 연령 카테고리에 걸쳐 1,000명 이상의 화자로부터 100만 개가 넘는 프레임을 확보했습니다. 이러한 모음 프레임에서 추출된 포먼트(formants)를 사용하여 마에다의 발성 모델 매개변수를 추정했습니다. 본 연구는 성별과 시대에 따른 발성 변화의 영향을 조사하고, 특히 마에다 모델의 발성 길이와 관련된 두 가지 매개변수인 후두 위치(여성의 경우 더 높음)와 입술 돌출(남성의 경우 더 돌출됨)에 주목합니다. 성별 간 음성 품질에 대한 영향을 논의하고, 시대에 따른 효과는 성별과 독립적이라는 것을 밝힙니다. 따라서 여성의 음성이 시간이 지남에 따라 낮아졌다는 주장은 뒷받침되지 않습니다.



### Simulating Articulatory Trajectories with Phonological Feature Interpolation (https://arxiv.org/abs/2408.04363)
Comments:
          accepted at Interspeech 2024

- **What's New**: 이 연구는 음성 인식과 생성 사이의 상호작용 루프를 기반으로 하는 완전한 음성 학습 계산 모델을 구축하는 첫 번째 단계로, 의사 운동 명령(pseudo-motor command)과 조음 궤적(articulatory trajectory) 사이의 순방향 매핑(forward mapping)을 조사합니다. 생성적 음운론(generative phonology)과 조음적 음운론(articulatory phonology)에 기반한 두 가지 음운적 특징 집합(phonological feature set)이 사용되어 음성 목표 시퀀스(phonetic target sequence)를 인코딩합니다. 각 특징 공간에서 부드러운 궤적을 생성하기 위해 다양한 보간 기법(interpolation technique)이 비교되며, 공동 발음(co-articulation) 효과를 포착하기 위해 목표 값(target value)과 타이밍(timing)을 최적화할 수 있습니다. 이 연구는 생성적 음운론에 기반한 확장된 특징 집합(extended feature set)과 선형 보간 기법(linear interpolation technique)을 사용하여 생성된 궤적의 선형 투영(linear projection)과 다중 화자(multi-speaker) 전자기 조음 측정법(EMA, electromagnetic articulography) 녹음 데이터 사이의 피어슨 상관관계(Pearson correlation)를 보고합니다. 0.67의 상관관계가 얻어졌습니다. 이 연구 결과는 생물학적 운동(biological motion) 역학에 대한 이해에 대한 의미를 논의합니다.



### HydraFormer: One Encoder For All Subsampling Rates (https://arxiv.org/abs/2408.04325)
Comments:
          accepted by ICME 2024

- **What's New**: 본 논문은 다양한 하위 샘플링 비율에 적응 가능한 새로운 음성 인식 모델인 HydraFormer를 제안합니다. 기존의 고정된 하위 샘플링 비율을 사용하는 모델과 달리, HydraFormer는 여러 하위 샘플링 비율을 동시에 처리하여, 다양한 상황에 효율적으로 대응할 수 있습니다. HydraFormer는 훈련 및 배포 비용을 크게 줄이면서도, 높은 인식 성능을 유지합니다. 또한, HydraFormer는 다양한 초기화 조건에서도 안정적인 성능을 보여주며, 사전 훈련된 단일 하위 샘플링 비율 음성 인식 모델에서 쉽게 미세 조정이 가능합니다.



### Incorporating Spatial Awareness in Data-Driven Gesture Generation for Virtual Agents (https://arxiv.org/abs/2408.04127)
- **What's New**: This paper proposes a novel synthetic dataset that augments an existing co-speech gesture dataset with multimodal referring expressions, enabling the generation of more natural gestures that are context-aware and spatially grounded.

- **Technical Details**: The dataset combines an existing co-speech gesture dataset containing beat gestures with a pointing gesture dataset. It utilizes GPT-4 to generate simple demonstrative referring expressions for the pointing gesture dataset and utilizes a TTS engine to synthesize speech segments aligned with the gestures.

- **Performance Highlights**: The resulting dataset enables the training of conversational embodied agents that can interpret and respond to both verbal and environmental cues. This represents a critical step toward creating more natural and engaging virtual agents that can interact with users in a more realistic and context-aware manner.



### Patchview: LLM-Powered Worldbuilding with Generative Dust and Magnet Visualization (https://arxiv.org/abs/2408.04112)
Comments:
          Accepted to UIST2024

- **What's New**: 이 논문은 Patchview를 소개합니다. Patchview는 사용자가 스토리 개념과 요소들을 자석과 먼지의 물리적 은유를 통해 상호작용할 수 있도록 하여 세계 구축을 시각적으로 돕는 맞춤형 LLM 기반 시스템입니다. Patchview에서 요소는 시각적으로 드래그되어 관련성이 높은 개념에 가까워지므로 이해를 돕습니다. 사용자는 또한 언어적으로 모호한 개념으로 생성을 조종할 수 있습니다. 사용자가 LLM의 시각화 및 생성에 동의하지 않으면 요소를 재배치하여 수정할 수 있습니다. 이러한 수정은 LLM의 미래 행동을 사용자의 인식에 맞추는 데 사용할 수 있습니다.



### Tree Attention: Topology-aware Decoding for Long-Context Attention on GPU clusters (https://arxiv.org/abs/2408.04093)
- **What's New**: 이 논문은 셀프 어텐션(self-attention)의 이론적 기반을 밝혀내고 베이지안(Bayesian) 해석을 제공하며, 홉필드 네트워크(Hopfield Networks)와 같은 에너지 기반 모델(energy-based models)과의 연관성을 강조하는 스칼라 에너지 함수(scalar energy function)를 도출합니다. 이를 통해 셀프 어텐션의 기울기(gradient)를 효율적으로 계산할 수 있는 트리 어텐션(Tree Attention) 알고리즘을 개발했습니다.

- **Technical Details**: 셀프 어텐션을 에너지 함수의 기울기(gradient)로 계산하는 새로운 방법을 제시합니다. 이 에너지 함수는 셀프 어텐션 블록의 계산을 효율적으로 병렬화하고 트리 감소(tree reduction) 토폴로지(topology)를 활용할 수 있도록 설계되었습니다. 또한, 이 논문은 셀프 어텐션을 수행하는 데 필요한 계산량을 줄이는 트리 어텐션(Tree Attention) 알고리즘을 개발했습니다.

- **Performance Highlights**: 본 논문의 접근 방식은 여러 장치에서 긴 시퀀스 길이(sequence length)를 가진 데이터를 디코딩(decoding)할 때, 링 어텐션(Ring Attention)과 같은 다른 방법보다 상당히 빠른 속도(최대 8배 빠름)를 제공하며, 통신량(communication volume)과 피크 메모리 사용량(peak memory usage)도 크게 감소시킵니다.



### Fairness in Large Language Models in Three Hours (https://arxiv.org/abs/2408.00992)
- **What's New**: 이 튜토리얼은 대규모 언어 모델(LLMs)에서의 공정성 문제에 대한 체계적인 개요를 제공하며, 실제 사례 연구에서 시작하여 편향의 원인을 분석하고 LLM의 공정성을 평가하고 촉진하기 위한 전략 및 알고리즘을 요약합니다. LLM에서 편향을 평가하기 위한 도구 및 데이터셋에 대한 정보도 제공하며, 이 분야의 현재 연구 과제와 미래 방향을 논의합니다.

- **Technical Details**: 이 튜토리얼은 LLM의 기본적인 이해, LLM 훈련 과정에서 발생하는 편향의 원인, LLM의 공정성을 측정하고 평가하는 다양한 전략(예: 인구 통계적 표현, 고정관념 연관성, 반팩트 공정성, 성능 불균형)을 다룹니다. 또한 LLM의 공정성을 개선하기 위한 방법(예: 전처리, 훈련 중, 프로세스 내, 후처리)을 소개하며, LLM 편향 평가를 위한 도구 및 데이터셋에 대한 정보도 제공합니다.

- **Performance Highlights**: 이 튜토리얼은 LLM의 공정성 문제를 해결하기 위한 최신 연구 동향을 포괄적으로 다루는 데 중점을 둡니다. 실제 사례 연구를 통해 논의를 심화시키고 참여자들의 참여를 유도하며, LLM의 공정성 평가 및 개선을 위한 구체적인 방법과 도구를 제공합니다.

- **Key Topics**: ['Fairness in LLMs', 'Bias in LLMs', 'Bias Detection', 'Bias Mitigation', 'Fairness Evaluation', 'Datasets for Fairness', 'Tools for Fairness', 'Challenges in Fairness', 'Future Directions in Fairness']



New uploads on arXiv(cs.IR)

### Pairing Clustered Inverted Indexes with kNN Graphs for Fast Approximate Retrieval over Learned Sparse Representations (https://arxiv.org/abs/2408.04443)
- **What's New**: SeismicWave, an enhanced version of the efficient sparse retrieval algorithm, Seismic, is introduced. SeismicWave improves on Seismic by incorporating two key innovations: 1) traversing blocks in order of their potential, instead of arbitrarily; and 2) expanding the retrieved document list with neighbors found through an offline k-regular nearest neighbor graph, leveraging the clustering hypothesis.



### Judgment2vec: Apply Graph Analytics to Searching and Recommendation of Similar Judgments (https://arxiv.org/abs/2408.04382)
Comments:
          5 pages, 7 figures, 2 tables

- **What's New**: 본 연구는 대만 법원 판결문 유사성 분석 자동화 시스템 개발을 목표로 합니다. 이를 위해 전문가들이 '골든 스탠다드'로 분류한 판결문 데이터셋을 활용하여 '전문가 유사성 점수'를 도출하고, '사건-조항' 관계 기반 지식 그래프를 구축하여 'Node2Vec 유사성 점수'를 산출합니다.

- **Technical Details**: 본 연구는 '코사인 유사성'과 'Node2Vec 유사성' 두 가지 유사성 지표를 비교 분석합니다. 특히 'Node2Vec' 알고리즘을 활용하여 판결문 간의 관계를 추론하고 지식 그래프를 구축하는 기술이 핵심입니다. 이는 기존의 '공법 체계' 기반 연구에서 벗어나 '대륙법 체계' 판결문의 유사성 분석에 적용 가능성을 보여줍니다.

- **Performance Highlights**: 본 연구는 판결문 유사성 분석 자동화를 통해 법률 전문가의 수고를 줄이고, 관련 정보 검색 및 추천 서비스 개발 가능성을 제시합니다. 또한 전문가 평가를 통해 알고리즘 기반 유사성과 인간의 직관적 유사성 간의 차이와 연관성을 분석하여, 법률 분야 AI 기술 개발에 기여할 수 있습니다.



### Understanding and Modeling Job Marketplace with Pretrained Language Models (https://arxiv.org/abs/2408.04381)
Comments:
          accepted by CIKM'24 applied research track

- **What's New**: PLM4Job, a novel job marketplace foundation model, is proposed to effectively leverage pretrained language models (PLMs) for understanding and modeling job market data. This approach aims to overcome the limitations of existing graph neural network (GNN) methods, which lack deep understanding of textual features and heterogeneous relationships within the job marketplace.



### Mitigating Exposure Bias in Online Learning to Rank Recommendation: A Novel Reward Model for Cascading Bandits (https://arxiv.org/abs/2408.04332)
- **What's New**: 이 논문은 온라인 추천 시스템에서 **선형 캐스케이드 밴딧**(Linear Cascading Bandit) 알고리즘을 사용하는 **노출 편향**(Exposure Bias) 문제를 해결하기 위한 새로운 보상 모델을 제안합니다. **노출 편향**은 특정 아이템이 반복적으로 추천되면서 다른 아이템은 제대로 추천되지 않는 현상을 말합니다. 이로 인해 사용자는 제한된 아이템만 접하게 되고, 시스템은 다양한 아이템의 가치를 제대로 평가하지 못합니다.



### Pairwise Judgment Formulation for Semantic Embedding Model in Web Search (https://arxiv.org/abs/2408.04197)
- **What's New**: 이 논문은 웹 검색을 위한 의미 임베딩 모델(SEM) 훈련을 위한 효과적인 쌍방향 판단(pairwise judgment) 생성 전략을 심층적으로 조사한 최초의 연구입니다. SEM은 쌍방향 아키텍처 기반의 신경망으로, 정보 검색 및 자연어 처리 분야에서 주목을 받고 있습니다. 이 논문에서는 기존 쌍방향 학습 순위(LTR) 분야에서 널리 사용되는 쌍방향 판단 공식화 전략이 SEM 훈련에 항상 효과적인 것은 아니라는 사실을 발견했습니다. 이 연구는 대규모 상업 검색 엔진의 쿼리 로그와 클릭 활동을 기반으로 한 광범위한 실험을 통해 SEM에 효과적인 전략을 보여주었으며, LTR의 원자적 휴리스틱(예: Clicked > Skipped)과 비교하여 하이브리드 휴리스틱(예: Clicked > Non-Clicked)의 장점을 강조했습니다.

- **Technical Details**: 이 연구에서는 SEM 훈련을 위한 쌍방향 판단 공식화를 위한 다양한 전략을 조사했습니다. 연구 결과, 기존 LTR에서 널리 사용되는 쌍방향 판단 전략은 SEM 훈련에 항상 효과적이지 않음을 밝혀냈습니다. 특히, LTR에서 거의 사용되지 않는 Clicked>Non-Examined 전략이 SEM에 가장 효과적임을 발견했습니다. 또한 Clicked>Non-Clicked 전략이 실질적인 장점을 가지고 있음을 보여주었습니다. 이 연구는 SEM의 아키텍처와 훈련 과정을 자세히 설명하고 있으며, 특히 힌지 손실(hinge loss) 함수를 이용한 쌍방향 학습 방식을 강조합니다.

- **Performance Highlights**: 이 연구에서는 대규모 상업 검색 엔진의 쿼리 로그를 사용하여 실험을 수행했으며, 다양한 쌍방향 판단 전략의 성능을 비교 분석했습니다. 그 결과, LTR에서 일반적으로 사용되는 전략보다 Clicked>Non-Examined 전략이 SEM 훈련에 더 효과적임을 확인했습니다. 또한 Clicked>Non-Clicked 전략이 SEM 훈련에 실질적인 장점을 제공한다는 점을 확인했습니다. 이러한 결과는 의미 임베딩 모델 훈련을 위한 효과적인 쌍방향 판단 공식화 전략을 찾는 것이 중요하며, 더 나아가 실무에서 의미 임베딩 모델의 성능을 향상시키기 위한 방향을 제시합니다.



### Enhanced Semantic Graph Based Approach With Sentiment Analysis For User Interest Retrieval From Social Sites (https://arxiv.org/abs/2408.04395)
Comments:
          This research was conducted as part of Master Thesis in Computer Science by the first author at HITEC University Taxila

- **What's New**: 본 논문은 사용자 제작 텍스트 (예: 트윗)를 분석하여 사용자의 관심사를 식별하고 분석하는 새로운 방식을 제시합니다. 기존의 설문 조사나 평점 시스템에 의존하는 방식과 달리, 사용자의 관심사를 파악하기 위해 소셜 미디어 콘텐츠를 직접 활용합니다. 특히, 다목적 자동 주제 색인 알고리즘 (Muai)을 사용하여 사용자 생성 텍스트에서 키워드를 추출하고, 이를 기반으로 의미 그래프를 생성합니다. 의미 그래프는 키워드를 노드로, 키워드 간 의미적 연결을 엣지로 표현하여 사용자의 관심사를 시각화하고 분석합니다. 또한, Zemanta와 같은 웹 서비스 및 DBpedia를 활용하여 키워드 추출 정확도를 높이고, 감성 분석을 통해 제품에 대한 긍정적/부정적 평가를 파악하는 기능을 추가합니다.



### MM-Forecast: A Multimodal Approach to Temporal Event Forecasting with Large Language Models (https://arxiv.org/abs/2408.04388)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)을 이용한 다중 모달 시간적 이벤트 예측의 새로운 문제를 연구하며, 특히 이미지를 사용한 시간적 이벤트 예측은 덜 연구되어 왔다는 점을 강조합니다. 이 연구는 이미지가 시간적 이벤트 예측에 어떻게 도움이 될 수 있는지, 그리고 이미지를 LLM 기반 예측 프레임워크에 통합하는 방법에 대해 탐구합니다.



### Scalable Transformer for High Dimensional Multivariate Time Series Forecasting (https://arxiv.org/abs/2408.04245)
- **What's New**: 이 논문은 고차원 다변량 시계열(MTS) 예측을 위한 새로운 스케일 가능한 트랜스포머 모델인 STHD(Scalable Transformer for High-Dimensional Multivariate Time Series Forecasting)를 제안합니다. STHD는 기존의 채널 종속 모델이 고차원 데이터에서 저조한 성능을 보이는 문제를 해결하기 위해 고안되었습니다.

- **Technical Details**: STHD는 세 가지 핵심 구성 요소로 구성됩니다:

1. **관계 행렬 희소성(Relation Matrix Sparsity):** 관련 없는 시계열로 인한 노이즈를 제한하고 메모리 문제를 완화합니다.
2. **ReIndex:** 더 유연한 배치 크기 설정을 가능하게 하고 훈련 데이터의 다양성을 높이는 훈련 전략입니다.
3. **트랜스포머(Transformer):** 2차원 입력을 처리하고 채널 간 종속성을 포착합니다.

STHD는 DeepGraph를 사용하여 고차원 MTS 데이터에서 채널 간의 관계를 희소화함으로써 효율성을 높입니다.

- **Performance Highlights**: 실험 결과 STHD는 범죄, 위키피디아 인물, 교통과 같은 세 가지 고차원 데이터셋에서 기존 방법보다 뛰어난 성능을 보였습니다.



### Enhanced Traffic Flow Prediction with Multi-Segment Fusion Tensor Graph Convolutional Networks (https://arxiv.org/abs/2408.04232)
- **What's New**: 본 논문에서는 복잡한 교통 네트워크 내의 공간-시간적 의존성을 포착하는 데 제한이 있는 기존의 교통량 예측 모델을 개선하기 위해 다중 세그먼트 융합 텐서 그래프 합성곱 네트워크(MS-FTGCN)를 제안합니다.  MS-FTGCN은 다음과 같은 세 가지 아이디어를 기반으로 합니다. a) 텐서 M-곱을 기반으로 하는 통합된 공간-시간적 그래프 합성곱 프레임워크를 구축하여 공간-시간 패턴을 동시에 포착합니다. b) 각각 교통량의 다중 시간적 특성을 모델링하기 위해 시간별, 일별, 주별 구성 요소를 통합합니다. c) 주의 메커니즘을 통해 세 구성 요소의 출력을 융합하여 최종 교통량 예측 결과를 얻습니다.



### MMREC: LLM Based Multi-Modal Recommender System (https://arxiv.org/abs/2408.04211)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)과 딥러닝 기법을 활용하여 추천 시스템을 개선하는 새로운 접근 방식을 제시합니다. 이 프레임워크는 다중 모달 정보 처리와 통합된 잠재 공간 표현을 통해 추천의 정확성과 관련성을 향상시키는 것을 목표로 합니다.

- **Technical Details**: 이 프레임워크는 LLM을 통해 텍스트와 이미지와 같은 다중 모달 정보를 효율적으로 추출하고 통합하며, 다양한 모달리티를 잠재 공간에 통합하여 순위 모델의 학습 과정을 단순화합니다. 이 연구에서는 LLM이 추천 컨텍스트에서 자연어 데이터를 더 잘 이해하고 활용하여 이전 방법의 한계를 해결할 수 있는 잠재력을 살펴봅니다.

- **Performance Highlights**: 실험 결과는 다중 모달 정보를 사용할 때 모델의 차별적 성능이 향상됨을 보여줍니다. 특히 불균형 데이터셋의 경우 오탐률을 개선하는 데 효과적입니다. 이 연구는 LLM과 다중 모달 데이터 통합이 더욱 개인화되고 맥락적으로 관련성이 높은 추천을 만드는 데 기여할 수 있음을 보여줌으로써 진화하는 추천 시스템 분야에 기여합니다.



### wav2graph: A Framework for Supervised Learning Knowledge Graph from Speech (https://arxiv.org/abs/2408.04174)
Comments:
          Preprint, 32 pages

- **What's New**: wav2graph는 음성 데이터에서 지식 그래프(KG)를 학습하기 위한 최초의 프레임워크로, 지식 추출을 자동화하고 음성 기반 정보를 다양한 AI 애플리케이션에 통합합니다. 이 프레임워크는 음성 인식(ASR) 전사본을 사용하여 KG를 구축하고, 최첨단 그래프 신경망(GNN)을 통해 노드 분류 및 링크 예측을 수행합니다.



New uploads on arXiv(cs.CV)

### LiDAR-Event Stereo Fusion with Hallucinations (https://arxiv.org/abs/2408.04633)
Comments:
          ECCV 2024. Code: this https URL - Project Page: this https URL

- **What's New**: This paper proposes a novel approach to address the limitations of event-based stereo matching, a technique used to estimate depth from neuromorphic cameras. The approach involves integrating a stereo event camera with a fixed-frequency active sensor, like a LiDAR, to acquire sparse depth measurements, overcoming the limitations caused by motion and untextured regions. The authors introduce two hallucination techniques: Virtual Stack Hallucination (VSH) and Back-in-Time Hallucination (BTH), to generate fictitious events in the event streams, thereby compensating for the lack of information in areas lacking brightness changes. These methods effectively improve the accuracy of event-based stereo systems, surpassing existing fusion methods from the RGB stereo literature.



### Puppet-Master: Scaling Interactive Video Generation as a Motion Prior for Part-Level Dynamics (https://arxiv.org/abs/2408.04631)
Comments:
          Project page: this https URL

- **What's New**: Puppet-Master라는 새로운 인터랙티브 비디오 생성 모델을 소개합니다. 이 모델은 부분 레벨 (part-level) 동역학에 대한 모션 (motion) 사전으로 사용될 수 있습니다. 테스트 시간에, 단일 이미지와 희소한 모션 궤적 세트 (즉, 드래그)가 주어지면, Puppet-Master는 주어진 드래그 상호 작용에 충실한 사실적인 부분 레벨 모션을 묘사하는 비디오를 합성할 수 있습니다. 이것은 대규모 사전 훈련된 비디오 확산 모델을 미세 조정하여 달성되며, 이를 위해 드래깅 컨트롤을 효과적으로 주입할 수 있는 새로운 조건화 아키텍처를 제안합니다. 더 중요한 것은, 널리 채택된 공간 주의 모듈을 대체할 수 있는 all-to-first 주의 메커니즘을 소개하는데, 이는 기존 모델의 외관 및 배경 문제를 해결하여 생성 품질을 크게 향상시킵니다. 다른 모션 조건 비디오 생성기는 야생 비디오로 훈련되고 대부분 전체 객체를 이동하는 반면, Puppet-Master는 Objaverse-Animation-HQ라는 새로운 데이터셋으로 훈련됩니다. 이 데이터셋은 큐레이션된 부분 레벨 모션 클립을 제공합니다. 최적이 아닌 애니메이션을 자동으로 필터링하고 의미 있는 모션 궤적을 합성 렌더링에 추가하는 전략을 제안합니다. Puppet-Master는 다양한 범주의 실제 이미지로 잘 일반화되며 실제 세계 벤치마크에서 제로 샷 방식으로 기존 방법을 능가합니다. 자세한 결과는 프로젝트 페이지를 참조하십시오: this http URL.



### Enhanced Prototypical Part Network (EPPNet) For Explainable Image Classification Via Prototypes (https://arxiv.org/abs/2408.04606)
Comments:
          Accepted at the International Conference on Image Processing (ICIP), IEEE (2024), we will update the new version after published through IEEE

- **What's New**: 이 논문은 이미지 분류를 위한 딥 뉴럴 네트워크 아키텍처인 'Enhanced Prototypical Part Network (EPPNet)'을 소개합니다. EPPNet은 강력한 성능을 달성하면서 동시에 분류 결과를 설명할 수 있는 관련 프로토타입(prototype)을 발견합니다. EPPNet은 더 관련성 있고 사람이 이해하기 쉬운 프로토타입을 발견하는 데 도움이 되는 새로운 클러스터 손실(cluster loss)을 도입합니다. 또한 발견된 프로토타입을 기반으로 결과의 설명 가능성을 평가하기 위한 충실도 점수(faithfulness score)를 소개합니다. 이 점수는 학습된 프로토타입의 관련성뿐만 아니라 모델의 성능도 고려합니다.



### Fall Detection for Industrial Setups Using YOLOv8 Variants (https://arxiv.org/abs/2408.04605)
- **What's New**: 이 논문은 산업 환경에서의 낙상 감지 시스템을 개발했으며, YOLOv8 모델과 데이터 증강 파이프라인을 사용하여 낙상 감지 정확도를 향상시켰습니다. 특히, YOLOv8m 모델은 연산 효율성과 감지 성능의 균형을 이루어 50% IoU(Intersection over Union)에서 mAP(mean Average Precision) 0.971을 달성했습니다. 이는 '낙상 감지'와 '인간의 움직임' 두 가지 범주 모두에서 높은 정확도를 보여줍니다.



### Towards High-resolution 3D Anomaly Detection via Group-Level Feature Contrastive Learning (https://arxiv.org/abs/2408.04604)
Comments:
          ACMMM24, 12 pages, 5 figures

- **What's New**: 이 논문은 고해상도 포인트 클라우드(HRPCD) 이상 탐지(AD)를 위한 새로운 그룹 수준 기능 기반 네트워크인 Group3AD를 제안합니다. Group3AD는 HRPCD 데이터에서 이상을 효과적으로 탐지하기 위해 특징 공간에서 더 균일하고 정렬된 특징 분포를 생성하는 Intercluster Uniformity Network(IUN) 및 Intracluster Alignment Network(IAN)을 사용합니다. 또한, Group3AD는 잠재적인 이상 영역의 픽셀 밀도를 향상시키고 탐지 성능을 향상시키기 위해 기하학적 정보를 기반으로 하는 적응형 그룹 센터 선택(AGCS)을 사용합니다.



### Improving Network Interpretability via Explanation Consistency Evaluation (https://arxiv.org/abs/2408.04600)
Comments:
          To appear in IEEE Transactions on Multimedia

- **What's New**: 이 논문은 딥러닝 모델의 해석성(interpretability)을 향상시키는 새로운 프레임워크를 제안합니다. 이 프레임워크는 '설명 일관성(explanation consistency)'이라는 새로운 지표를 도입하여 모델의 학습 과정을 개선합니다. 설명 일관성은 모델이 생성한 설명이 원본 이미지와 의미적으로 유사한 적대적 예제(adversarial example)에 대해 얼마나 일관성 있는지를 측정합니다. 이를 통해 모델은 설명 일관성이 낮은(즉, 설명이 일관성이 없는) 학습 샘플에 더 많은 주의를 기울이고, 모델의 예측 성능과 해석성을 모두 향상시킬 수 있습니다.



### Img-Diff: Contrastive Data Synthesis for Multimodal Large Language Models (https://arxiv.org/abs/2408.04594)
Comments:
          14 pages, 9 figures, 7 tables

- **What's New**: Img-Diff, a novel dataset designed for enhancing fine-grained image recognition in Multimodal Large Language Models (MLLMs), is introduced. This dataset leverages insights from contrastive learning and image difference captioning, focusing on identifying object differences between similar images.

- **Technical Details**: The dataset is created using Stable-Diffusion-XL and advanced image editing techniques. It involves generating pairs of similar images with object replacements, highlighting specific differences. This process utilizes a Difference Area Generator for object differences identification and a Difference Captions Generator for detailed descriptions of those differences. The methodology emphasizes identifying 'object replacement' samples, resulting in a relatively small but high-quality dataset.

- **Performance Highlights**: Fine-tuning state-of-the-art (SOTA) MLLMs like MGM-7B using Img-Diff leads to significant performance improvements in image difference and Visual Question Answering (VQA) tasks. Notably, the trained models surpass SOTA models like GPT-4V and Gemini on the MMVP benchmark. The dataset showcases diversity, quality, and robustness, demonstrating effectiveness in bolstering MLLMs' capabilities in image difference recognition and fine-grained image analysis.

- **Code and Dataset Availability**: The code and dataset are publicly available at (URL provided in the paper) to encourage further research and advancements in multimodal data synthesis and enhancement of MLLM capabilities for image understanding.



### SAM 2 in Robotic Surgery: An Empirical Evaluation for Robustness and Generalization in Surgical Video Segmentation (https://arxiv.org/abs/2408.04593)
Comments:
          Empirical study. Previous work "SAM Meets Robotic Surgery" is accessible at: arXiv:2308.07156

- **What's New**: 이 논문은 최근 출시된 **Segment Anything Model (SAM) 2** 의 수술 영상 분할(segmentation) 성능과 견고성(robustness)에 대한 연구 결과를 제시합니다. SAM 2는 이미지와 비디오에서 **상호 작용적 분할(interactive segmentation)** 에 있어 뛰어난 성능을 보여주었으며, **메모리 메커니즘(memory mechanism)** 과 **마스크 디코더(mask decoder)** 를 사용하여 비디오 추적(video tracking)과 객체 폐색(object occlusion) 문제를 효과적으로 해결했습니다.

- **Technical Details**: 본 연구는 MICCAI EndoVis 2017과 2018 데이터셋을 사용하여 SAM 2의 **영점 분할(zero-shot segmentation)** 성능을 평가했습니다. 이미지 데이터의 경우 **1점(one-point)** 또는 **바운딩 박스(bounding box)** 를 사용하여 프롬프트를 입력하고, 비디오 데이터의 경우 첫 번째 프레임에 1점 프롬프트를 적용했습니다. 또한 이미지 압축, 노이즈, 블러, 폐색 등의 **실제 세계 데이터 오류(real-world corruption)** 에 대한 견고성을 분석했습니다.

- **Performance Highlights**: 결과적으로 SAM 2는 기존 SAM 모델보다 뛰어난 성능을 보여주었으며, 특히 바운딩 박스 프롬프트를 사용할 경우 기존 **최첨단(state-of-the-art, SOTA)** 방법을 능가했습니다. 1점 프롬프트를 사용했을 때도 기존 **프롬프트 없는 SOTA(unprompted SOTA)** 방법에 근접하거나 뛰어넘는 성능을 보였습니다. SAM 2는 또한 기존 SAM 모델보다 **추론 속도(inference speed)** 가 2배 이상 빨라졌으며, 다양한 데이터 오류에 대한 견고성도 향상되었습니다. 다만 일부 경계나 영역에서는 여전히 개선의 여지가 있습니다.

- **Limitations**: 본 연구에서는 1점 프롬프트를 사용했을 때, 비디오 분할의 성능이 이미지 분할보다 다소 떨어지는 것을 확인했습니다. 이는 비디오 분할은 첫 번째 프레임에만 프롬프트를 사용하는 반면, 이미지 분할은 모든 프레임에서 프롬프트를 적용하기 때문입니다.

- **Future Directions**: 본 연구 결과는 SAM 2가 수술 영상 분할과 추적에 잠재력을 가지고 있음을 시사합니다. 향후 연구에서는 **자동화된 프롬프트 생성(automated prompt generation)** 기술을 활용하여 SAM 2의 적용 범위를 확대하고, **실제 수술 환경(real-world surgical settings)** 에서의 성능을 검증할 필요가 있습니다.



### HiLo: A Learning Framework for Generalized Category Discovery Robust to Domain Shifts (https://arxiv.org/abs/2408.04591)
Comments:
          39 pages, 9 figures, 26 tables

- **What's New**: 이 논문은 '일반화된 카테고리 발견 (Generalized Category Discovery, GCD)' 작업에 도메인 이동 (domain shift) 문제를 추가하여 기존 연구와 차별화합니다. 기존 GCD 연구는 라벨이 없는 데이터가 라벨이 있는 데이터와 동일한 도메인에서만 온다고 가정했지만, 이 논문은 라벨이 없는 데이터가 다른 도메인에서도 올 수 있다는 현실적인 상황을 고려합니다. 이러한 상황을 해결하기 위해 'HiLo' 네트워크라는 새로운 방법을 제안합니다. HiLo 네트워크는 고수준 의미론적 특징(High-level semantic features)과 저수준 도메인 특징(Low-level domain features)을 추출하고, 이 두 특징 사이의 상호 정보(mutual information)를 최소화하는 방식으로 동작합니다.



### SAM2-Adapter: Evaluating & Adapting Segment Anything 2 in Downstream Tasks: Camouflage, Shadow, Medical Image Segmentation, and Mor (https://arxiv.org/abs/2408.04579)
Comments:
          arXiv admin note: text overlap with arXiv:2304.09148

- **What's New**: 이 논문은 SAM2-Adapter를 소개하며, 이는 기존의 SAM2 모델의 한계를 극복하고 의료 이미지 분할, 위장된 객체 탐지, 그림자 탐지 등의 특정 하위 작업에서 최첨단(SOTA) 결과를 달성하기 위한 새로운 어댑터입니다. SAM2-Adapter는 SAM-Adapter의 강점을 기반으로 하여 다양한 애플리케이션에 대한 향상된 일반화 및 구성 가능성을 제공합니다.



### Sketch2Scene: Automatic Generation of Interactive 3D Game Scenes from User's Casual Sketches (https://arxiv.org/abs/2408.04567)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문은 사용자의 간단한 스케치만으로 상호 작용할 수 있고 플레이 가능한 3D 게임 장면을 자동으로 생성하는 새로운 딥 러닝 기반 방식을 제안합니다. 이 방법은 3D 장면 생성을 위해 훈련 데이터 부족 문제를 해결하기 위해 사전 훈련된 2D 잡음 제거 확산 모델을 활용하여 컨셉 가이드로 사용할 2D 장면 이미지를 생성합니다. 이 논문은 Sketch2Scene이라는 파이프라인을 제안하는데, 이 파이프라인은 사용자가 그린 스케치와 텍스트 프롬프트를 입력으로 사용하여 사실적이고 상호 작용할 수 있는 가상 환경을 자동으로 생성합니다. Sketch2Scene은 사용자가 그린 스케치를 활용하여 대규모 오픈 월드 야외 장면 생성의 한계를 효과적으로 해결합니다. 이 논문은 사용자가 제공한 간단한 스케치를 통해 컨트롤 가능하고 2D 등각 투영 이미지 생성 파이프라인, 새로운 데이터 세트에서 단계별 잡음 제거 확산으로 훈련된 기본 지형 채우기 모델, 학습 기반 구성 3D 장면 이해 모듈, 위 장면 이해 모듈에서 얻은 장면 매개변수를 사용하여 상호 작용하는 3D 장면을 렌더링하는 절차적 생성 파이프라인을 제안합니다.



### Depth Any Canopy: Leveraging Depth Foundation Models for Canopy Height Estimation (https://arxiv.org/abs/2408.04523)
Comments:
          Accepted at ECCV 2024 CV4E Workshop

- **What's New**: This paper introduces Depth Any Canopy (DAC), a novel method for estimating global tree canopy height by leveraging the capabilities of pre-trained monocular depth estimation foundation models.

- **Technical Details**: DAC utilizes Depth Anything v2, a state-of-the-art monocular depth estimation model, and fine-tunes it for canopy height estimation using aerial imagery. This approach minimizes the need for extensive training on remote sensing data, making it efficient and cost-effective.

- **Performance Highlights**: DAC demonstrates superior or comparable performance compared to existing methods, requiring significantly fewer computational resources and achieving an estimated carbon footprint of 0.14 kgCO2. It achieves these results with minimal compute and parameters, making it a practical solution for large-scale canopy height estimation.

- **Datasets**: The study utilizes high-resolution multi-spectral and hyperspectral aerial imagery along with LiDAR data obtained from the National Ecological Observatory Network (NEON) catalog. This data provides detailed information on vegetation, land cover, and water bodies, aiding in canopy height estimation.



### Saliency Detection in Educational Videos: Analyzing the Performance of Current Models, Identifying Limitations and Advancement Directions (https://arxiv.org/abs/2408.04515)
- **What's New**: 이 논문은 교육용 비디오에 대한 최첨단 샐리언시 감지(saliency detection) 접근 방식을 평가하여 교육적 맥락에서 이러한 모델의 성능과 한계를 분석합니다. 교육용 비디오에서 샐리언시 감지에 대한 연구가 부족한 상황을 감안하여, 본 연구는 4가지 주요 샐리언시 감지 모델을 교육용 비디오에 적용하고 일반적인 (비 교육용) 데이터셋에 대한 모델의 복제 능력을 조사합니다. 또한 교육용 비디오에 대한 모델의 일반화 능력을 평가하고, 실패 사례와 개선 가능성을 분석합니다.



### Towards Synergistic Deep Learning Models for Volumetric Cirrhotic Liver Segmentation in MRIs (https://arxiv.org/abs/2408.04491)
- **What's New**: 이 연구는 복잡한 특징 상호 작용을 모델링하고 다양한 데이터셋에서 일반화하기 위해 새로운 시너지 이론을 제안합니다. 이 이론은 상호 보완적인 잠재 공간을 활용하여 특징 상호 작용을 향상시키는 것을 목표로 합니다. 이 연구는 nnSynergyNet3D 아키텍처를 소개하며, 이 아키텍처는 3D 볼륨에 대한 연속적 및 이산적 잠재 공간을 통합합니다.



### SegXAL: Explainable Active Learning for Semantic Segmentation in Driving Scene Scenarios (https://arxiv.org/abs/2408.04482)
Comments:
          17 pages, 7 figures. To appear in the proceedings of the 27th International Conference on Pattern Recognition (ICPR), 01-05 December, 2024, Kolkata, India

- **What's New**: 이 논문은 자동 주행 시나리오에서의 의미론적 분할(semantic segmentation)을 위해 'SegXAL'이라는 새로운 설명 가능한 능동 학습(Explainable Active Learning, XAL) 모델을 제안합니다. SegXAL은 효율적으로 라벨이 지정되지 않은 데이터를 활용하고, 사람의 개입(oracle)을 통해 가장 정보가 많은 데이터를 선택적으로 라벨링하여 훈련 데이터를 효율적으로 구축하는 데 중점을 둡니다. 또한, 모델의 의사 결정 과정을 해석 가능하게 함으로써 사람과 AI 간의 협업 지능을 강화합니다.



### LumiGauss: High-Fidelity Outdoor Relighting with 2D Gaussian Splatting (https://arxiv.org/abs/2408.04474)
Comments:
          Includes video files in src

- **What's New**: LumiGauss는 2D Gaussian Splatting을 사용하여 3D 장면 재구성 및 환경 조명을 수행하는 새로운 기술입니다. LumiGauss는 고품질 장면 재구성을 제공하고 새로운 환경 맵 아래에서 사실적인 조명 합성을 가능하게 합니다. 특히, LumiGauss는 구면 조화 특성을 활용하여 야외 장면에서 흔히 볼 수 있는 그림자의 품질을 향상시키는 방법을 제안합니다.



### What could go wrong? Discovering and describing failure modes in computer vision (https://arxiv.org/abs/2408.04471)
- **What's New**: 이 연구는 컴퓨터 비전 모델의 오류 모드를 자연어로 설명하고 예측하는 새로운 접근 방식을 제안합니다.  특히, Language-Based Error Explainability (LBEE) 문제를 공식화하고, 이를 위한 새로운 지표들을 소개합니다. 또한, 비전-언어 임베딩 공간에서 작동하는 새로운 방법들을 제안하여 훈련 중에 보지 못한 객체나 시각적으로 불리한 조건으로 인한 모델 오류를 설명합니다.

- **Technical Details**: 이 연구는 모델의 성능이 좋거나 나쁜 샘플을 구분하기 위해 불확실성 추정을 사용하고, 특정 시각적 조건과 관련된 어려운 샘플을 그룹화하기 위해 공동 비전-언어 임베딩 공간에서 클러스터링을 수행합니다. 모델이 어려움을 겪는 클러스터와 잘 수행하는 클러스터의 설명을 대조하여 각 클러스터에 고유한 문장을 식별합니다.

- **Performance Highlights**: 제안된 방법은 분류 및 분할 작업에 대한 광범위한 실험을 통해 검증되었습니다. 이를 통해 모델의 실패 이유를 만족스럽게 설명하고, 각 오류 모드 내에서 이미지를 잘 설명하는 문장을 생성하는 것으로 나타났습니다.



### Deep Learning for identifying systolic complexes in SCG traces: a cross-dataset analysis (https://arxiv.org/abs/2408.04439)
- **What's New**: 이 연구는 심장 활동 분석에 사용되는 기존 ECG의 대안으로 떠오르는 세이스모카디오그램(SCG) 신호의 시스톨릭 복합체(systolic complex) 검출을 위한 딥러닝 기반 솔루션을 제안합니다. 특히, 이 연구는 다양한 데이터 세트에서 수집된 SCG 데이터를 사용하여 실제 환경에서의 도메인 쉬프트(domain shift) 문제를 해결하는 데 중점을 둡니다. 즉, 제어된 환경에서 수집된 데이터로 훈련된 모델이 실제 환경에서 수집된 데이터에 대해서는 효과적이지 않을 수 있음을 인식하고, 이러한 문제를 해결하기 위해 모델 개인화(personalization) 방법을 적용합니다. 또한, 가속도계(accelerometer)와 자이로스코프(gyroscope) 데이터를 결합하여 다중 채널 접근 방식(multi-channels approach)을 사용하는 장점을 보여줍니다.

- **Technical Details**: 이 연구에서는 U-Net이라는 딥러닝 모델을 사용하여 SCG 신호에서 시스톨릭 복합체를 검출합니다. U-Net은 이미지 분할(semantic segmentation) 문제에 널리 사용되는 모델이며, SCG 신호를 입력으로 받아 각 샘플이 시스톨릭 복합체에 속할 확률을 출력합니다. 이 연구에서는 CEBS, MEC, BioPoli 세 가지 데이터 세트를 사용하여 실험을 진행합니다. CEBS와 MEC는 제어된 환경에서 수집된 데이터를 포함하는 반면, BioPoli는 실제 환경에서 수집된 24시간 데이터를 포함합니다. 데이터 세트 간의 차이점을 고려하여 도메인 쉬프트 문제를 해결하기 위해 모델 개인화(personalization) 방법을 적용합니다. 또한, 가속도계와 자이로스코프 데이터를 결합하여 다중 채널 접근 방식을 사용하여 성능을 향상시킵니다.

- **Performance Highlights**: 이 연구는 다양한 데이터 세트에서 수집된 SCG 데이터를 사용하여 실제 환경에서의 도메인 쉬프트 문제를 해결하는 데 효과적인 딥러닝 모델을 제시합니다. 또한, 모델 개인화(personalization) 방법을 적용하여 성능을 향상시키고, 다중 채널 접근 방식의 장점을 보여줍니다. 이 연구는 SCG 신호를 이용한 심장 활동 분석 분야의 발전에 기여할 것으로 기대됩니다.



### A Review of 3D Reconstruction Techniques for Deformable Tissues in Robotic Surgery (https://arxiv.org/abs/2408.04426)
Comments:
          To appear in MICCAI 2024 EARTH Workshop. Code availability: this https URL

- **What's New**: 본 논문은 수술 장면을 재구성하기 위한 최첨단(SOTA) 접근 방식을 분석하고 평가하며, 이러한 기술의 발전으로 실시간 고품질 재구성이 가능해짐을 보여줍니다. 특히, NeRF 기반 기법과 Gaussian Splatting 기반 기법의 장단점을 비교 분석하고, 수술 장면 재구성에 적용했을 때의 차이점을 분석합니다. 또한, EndoNeRF, EndoSurf, LerPlane 및 4D-GS의 네 가지 방법을 구현하고, 세 가지 데이터셋에서 성능을 비교 평가합니다.



### MultiViPerFrOG: A Globally Optimized Multi-Viewpoint Perception Framework for Camera Motion and Tissue Deformation (https://arxiv.org/abs/2408.04367)
- **What's New**: This paper proposes a novel multi-viewpoint global optimization framework called MultiViPerFrOG for 3D deformable scene reconstruction using multiple depth cameras. This framework combines low-level perception modules (data association, depth, relative scene flow) with kinematic and scene-modeling priors to jointly estimate camera motions and absolute scene flow, addressing the challenges of ill-posed problems arising from single-viewpoint deformable scene reconstruction.



### AggSS: An Aggregated Self-Supervised Approach for Class-Incremental Learning (https://arxiv.org/abs/2408.04347)
Comments:
          Accepted in BMVC 2024

- **What's New**: 본 논문은 이미지 회전을 이용한 자기 지도 학습(self-supervised learning)이 다양한 클래스 증분 학습(class-incremental learning) 패러다임에 미치는 영향을 조사합니다. 각 이미지에 미리 정의된 회전을 적용하여 새로운 클래스로 간주하고 학습하는 방식을 제시하며, 이를 Aggregated Self-Supervision (AggSS)라고 합니다. AggSS는 딥 신경망의 주의(attention)를 객체의 본질적인 특징(intrinsic object features)에 집중시켜 강력한 특징 학습(robust feature learning)을 가능하게 합니다. AggSS는 클래스 증분 학습 프레임워크에 쉽게 통합할 수 있는 플러그 앤 플레이(plug-and-play) 모듈이며, 다양한 클래스 증분 학습 방식의 성능을 향상시키는 강력한 특징 학습 기능을 제공합니다. CIFAR-100 및 ImageNet-Subset과 같은 표준 증분 학습 데이터셋에서 수행된 광범위한 실험을 통해 AggSS가 이러한 패러다임에서 성능을 향상시키는 데 중요한 역할을 한다는 사실을 보여줍니다.



### Multi-Scale and Detail-Enhanced Segment Anything Model for Salient Object Detection (https://arxiv.org/abs/2408.04326)
Comments:
          This work is accepted by ACM MM2024

- **What's New**: 본 논문은 MDSAM(Multi-scale and Detail-Enhanced SAM)이라는 새로운 SOD(Salient Object Detection) 프레임워크를 제안합니다. 이 프레임워크는 SAM(Segment Anything Model)을 SOD 작업에 적용하기 위해 다중 스케일 및 디테일 강화 정보를 사용합니다. MDSAM은 SAM의 성능을 향상시키기 위해 LMSA(Lightweight Multi-Scale Adapter), MLFM(Multi-Level Fusion Module), DEM(Detail Enhancement Module)의 세 가지 모듈을 활용합니다.  LMSA는 SAM이 훈련 파라미터를 최소화하면서 다중 스케일 정보를 학습하도록 합니다. MLFM은 SAM의 인코더에서 다양한 레벨의 특징을 추출하고 융합합니다. DEM은 이미지의 디테일과 가장자리를 통합하여 정확하고 세밀한 결과를 생성합니다.



### Respiratory Subtraction for Pulmonary Microwave Ablation Evaluation (https://arxiv.org/abs/2408.04299)
- **What's New**: 이 논문은 폐암 수술 후 폐암 절제술의 효과를 평가하기 위한 새로운 방법인 '호흡 보정'을 제안합니다. 호흡 보정은 수술 전과 수술 후 영상 간의 차이를 계산하여 절제 영역의 범위를 시각적으로 보여주고, 수술 효과를 더 정확하게 평가하는 것을 목표로 합니다. 또한 절제 효과를 정량적으로 평가할 수 있는 새로운 지표인 '절제 효과 지표 (AES)'를 제시합니다. 이 방법은 수술 후 장기 추적 관찰 없이 수술 효과를 평가하는 데 도움이 될 수 있습니다.



### Dual-branch PolSAR Image Classification Based on GraphMAE and Local Feature Extraction (https://arxiv.org/abs/2408.04294)
- **What's New**: 본 논문에서는 제한된 레이블로 된 PolSAR 영상 분류 문제를 해결하기 위해 생성적 자기지도 학습(Generative Self-Supervised Learning) 기반의 이중 분기(Dual-Branch) 분류 모델을 제안합니다. 제안된 모델은 그래프 마스크 오토 인코더(Graph Masked Autoencoder) 기반의 슈퍼 픽셀 분기(Superpixel Branch)와 CNN 기반의 픽셀 분기(Pixel Branch)로 구성됩니다. 슈퍼 픽셀 분기는 그래프 마스크 오토 인코더를 사용하여 슈퍼 픽셀 수준의 편광 표현을 학습하고, 픽셀 분기는 더욱 세밀한 분류 결과를 얻기 위해 픽셀 수준의 특징을 학습합니다. 마지막으로 두 분기의 특징을 융합하여 분류를 수행합니다.



### Evaluating Modern Approaches in 3D Scene Reconstruction: NeRF vs Gaussian-Based Methods (https://arxiv.org/abs/2408.04268)
Comments:
          Accepted by 2024 6th International Conference on Data-driven Optimization of Complex Systems

- **What's New**: 이 논문은 3D 장면 재구성(reconstruction) 분야에서 Neural Radiance Fields(NeRF)와 Gaussian 기반 방법들을 최신 SLAM 시스템과 비교 분석합니다. Replica 및 ScanNet 데이터셋을 사용하여 추적 정확도, 매핑 정확도, 뷰 합성 측면에서 성능을 평가합니다. NeRF는 뷰 합성에서 탁월한 성능을 보여주지만 처리 속도가 느립니다. 반면 Gaussian 기반 방법은 빠른 처리 속도와 높은 표현력을 제공하지만 장면 완성(scene completion)에는 부족합니다. NICE-SLAM과 SplaTAM과 같은 최신 방법은 전역 최적화(global optimization)와 루프 클로저(loop closure) 기술을 통해 ORB-SLAM2와 같은 기존 프레임워크보다 견고성이 뛰어나며 동적이고 복잡한 환경에서도 우수한 성능을 보여줍니다. 이 비교 분석은 이론적 연구와 실제 적용 사이의 간극을 메워 3D 장면 재구성 분야의 미래 발전을 위한 통찰력을 제공합니다.



### CoBooM: Codebook Guided Bootstrapping for Medical Image Representation Learning (https://arxiv.org/abs/2408.04262)
Comments:
          Accepted in MICCAI 2024

- **What's New**: 본 논문은 의료 이미지 분석을 위한 새로운 자기 지도 학습(Self-Supervised Learning, SSL) 프레임워크인 CoBooM을 제안합니다. CoBooM은 코드북(Codebook)을 통합하여 의료 이미지의 해부학적 유사성을 활용하여 일반화된 특징을 학습하는 데 중점을 둡니다. 기존의 SSL 방법들은 의료 이미지에서 나타나는 다양한 해부학적 구조 간의 유사성을 충분히 고려하지 못했지만, CoBooM은 코드북을 통해 유사한 해부학적 특징을 동일하거나 유사한 코드와 연결하고, 서로 다른 특징을 구별되는 코드와 연결함으로써 구조화된 학습 과정을 제공합니다. 이를 통해 모델은 의료 이미지에 흔히 나타나는 패턴과 구조를 더 잘 인식하고 학습할 수 있습니다.



### Unveiling Hidden Visual Information: A Reconstruction Attack Against Adversarial Visual Information Hiding (https://arxiv.org/abs/2408.04261)
Comments:
          12 pages

- **What's New**: 이 논문은 암호화된 이미지에 대한 데이터 재구성(DR) 공격을 수행하여 적대적 예제 기반 이미지 암호화의 보안 취약점을 조사합니다. 대표적인 이미지 암호화 방법은 적대적 시각 정보 숨기기(AVIH)로, 이미지 인식 작업에 사용되는 갤러리 데이터 세트를 보호하기 위해 유형 I 적대적 예제 학습을 사용합니다. AVIH 방법에서 유형 I 적대적 예제 접근 방식은 완전히 다르게 보이지만 기계가 원본으로 인식하는 이미지를 만듭니다. 또한 AVIH 방법은 미리 정의된 개인 키 생성 모델을 사용하여 암호화된 이미지를 원래 형태로 복원할 수 있습니다. 최상의 보안을 위해 각 이미지에 고유한 키를 할당하는 것이 좋지만 저장 용량 제한으로 인해 일부 이미지가 동일한 키 모델을 공유해야 할 수 있습니다. 이는 AVIH에 대한 중요한 보안 문제를 제기합니다. 즉, DR 공격으로 손상되지 않고 동일한 키 모델을 안전하게 공유할 수 있는 이미지는 몇 개입니까? 이 질문에 답하기 위해 저자는 (1) 생성적 적대적 손실과 (2) 증강된 ID 손실을 통합하여 AVIH 암호화 방법에 대한 이중 전략 DR 공격을 소개합니다. 이는 DR이 과적합(overfitting)되는 것을 방지합니다. 이는 기계 학습에서 발생하는 문제와 유사합니다. 숫자 결과는 이미지 인식 및 재 식별 벤치마크를 통해 이 접근 방식을 검증하여 제안된 전략이 재구성된 이미지의 품질을 크게 향상시켜 더 적은 키 공유 암호화 이미지를 필요로 함을 보여줍니다. 결과를 재현하기 위한 소스 코드는 곧 제공될 예정입니다.



### UHNet: An Ultra-Lightweight and High-Speed Edge Detection Network (https://arxiv.org/abs/2408.04258)
- **What's New**: 이 논문은 의료 이미지 처리에서 경계선 감지를 위한 초경량 모델인 UHNet을 제안합니다. UHNet은 42.3k 매개변수, 166 FPS의 속도, 0.79G FLOPs만 사용하며, 기존의 무거운 모델들과 동등한 수준의 성능을 보입니다. 특히, UHNet은 전처리 비용이 전혀 들지 않고,  최소한의 매개변수로 높은 처리 속도를 보장합니다.



### InstantStyleGaussian: Efficient Art Style Transfer with 3D Gaussian Splatting (https://arxiv.org/abs/2408.04249)
- **What's New**: 본 논문은 3D Gaussian Splatting (3DGS) 장면 표현을 기반으로 하는 혁신적인 3D 스타일 전이 기법인 InstantStyleGaussian을 소개합니다. 이 방법은 대상 스타일 이미지를 입력으로 받아 빠르게 새로운 3D GS 장면을 생성합니다. InstantStyleGaussian은 사전 재구성된 GS 장면에서 작동하며, 확산 모델(diffusion model)과 개선된 반복적 데이터셋 업데이트 전략을 결합합니다. 확산 모델을 사용하여 대상 스타일 이미지를 생성하고, 이러한 새 이미지를 학습 데이터셋에 추가한 후, 이 데이터셋을 사용하여 GS 장면을 반복적으로 업데이트하고 최적화합니다. 광범위한 실험 결과를 통해 InstantStyleGaussian이 고품질의 스타일화된 장면을 보장하는 동시에 스타일 전이 속도와 일관성 측면에서 상당한 이점을 제공함을 보여줍니다.



### MU-MAE: Multimodal Masked Autoencoders-Based One-Shot Learning (https://arxiv.org/abs/2408.04243)
Comments:
          IEEE MIPR 2024

- **What's New**: 본 논문에서는 멀티모달 마스크 오토인코더를 기반으로 한 원샷 학습(Mu-MAE)이라는 새로운 접근 방식을 제안하여 멀티모달 데이터의 어노테이션 비용을 크게 줄이고, 외부 데이터나 사전 학습된 모델에 의존하지 않고도 효율적인 멀티모달 모델 사전 훈련을 수행합니다.



### LLDif: Diffusion Models for Low-light Emotion Recognition (https://arxiv.org/abs/2408.04235)
Comments:
          Accepted by ICPR2024

- **What's New**: LLDif, a novel diffusion-based facial expression recognition (FER) framework for extremely low-light (LL) environments is introduced. It addresses the challenges posed by low-brightness and reduced contrast in such environments by using a two-stage training process.



### Cross-View Meets Diffusion: Aerial Image Synthesis with Geometry and Text Guidanc (https://arxiv.org/abs/2408.04224)
- **What's New**: 본 논문은 지상 이미지에서 사실적인 항공 이미지를 생성하는 새로운 기하학적 보존 지상-항공(G2A) 이미지 합성(GPG2A) 모델을 제안합니다. GPG2A는 지상 이미지에서 조류의 눈으로 보는 (BEV, Bird's Eye View) 분할(BEV 레이아웃 맵이라고 함)을 예측하는 첫 번째 단계와 예측된 BEV 레이아웃 맵과 지상 이미지의 텍스트 설명에서 항공 이미지를 합성하는 두 번째 단계로 구성됩니다. 모델을 훈련하기 위해 VIGOR에 새로 수집된 항공 이미지, 지도 및 텍스트 설명을 추가한 새로운 다중 모드 크로스 뷰 데이터 세트인 VIGORv2를 제시합니다.  GPG2A가 기존 모델보다 기하학적으로 보존된 항공 이미지를 더 잘 합성한다는 것을 광범위한 실험을 통해 보여줍니다. 또한 크로스 뷰 지리 위치 지정 및 스케치 기반 지역 검색에 대한 두 가지 애플리케이션을 제시하여 GPG2A의 효율성을 추가로 검증합니다. 코드와 데이터는 공개적으로 제공될 예정입니다.



### VideoQA in the Era of LLMs: An Empirical Study (https://arxiv.org/abs/2408.04223)
Comments:
          Preprint. Under Review

- **What's New**: This paper provides a comprehensive analysis of Video-LLMs' (Video Large Language Models) performance in Video Question Answering (VideoQA). It investigates their strengths, weaknesses, and failure modes, offering valuable insights for developing more human-like video understanding and question-answering systems.

- **Technical Details**: The research utilizes a series of adversarial probes to test Video-LLMs in various areas, including temporal understanding, visual grounding, multimodal VQA reasoning, robustness, and generalization.  These probes were designed to adjust the original VideoQA data or settings and analyze the models' performance before and after these modifications.  For comparison, the study also includes the behavior of SOTA (State-of-the-Art) non-LLM methods that fine-tune small language models like BERT and RoBERTa.

- **Performance Highlights**: While Video-LLMs excel in standard VideoQA scenarios, showing strong correlations with contextual cues and generating plausible responses to questions about various video content, the analysis reveals significant limitations:

*   **Temporal Understanding:** Video-LLMs struggle with temporal reasoning, particularly in understanding the order of video content. They are often outperformed by non-LLM methods in this area.
*   **Visual Grounding:**  While Video-LLMs significantly outperform non-LLM methods in answering video questions, their visual grounding abilities are relatively weak. This suggests their success is largely driven by language priors and spurious vision-text correlations.
*   **Multimodal VQA Reasoning:** Video-LLMs excel at exploiting shortcuts in candidate answers for multi-choice QA, which indicates their limited capacity for faithful reasoning from video questions to correct answers.
*   **Robustness:** Video-LLMs show surprising insensitivity to video perturbations (e.g., shuffling frames) but are unexpectedly sensitive to simple variations in question wording, especially in open-ended QA.
*   **Generalization:** Video-LLMs tend to favor high-frequency answers for open-ended questions and may not generalize effectively across different question types or datasets.

- **Findings Summary**: The findings highlight the impressive capabilities of Video-LLMs in standard VideoQA tasks, but also reveal critical shortcomings in robustness, interpretability, and their reliance on spurious correlations. This underscores the urgent need for rationales in the development of future Video-LLMs to ensure more trustworthy and human-like video understanding.



### Connective Viewpoints of Signal-to-Noise Diffusion Models (https://arxiv.org/abs/2408.04221)
- **What's New**: 이 논문은 S2N(Signal-to-Noise) 확산 모델에 대한 통합적인 관점을 제공하여 다양한 관점을 연결하고 새로운 관점을 탐구합니다. 특히, 노이즈 스케줄러(noise scheduler)의 역할을 SNR(Signal-to-Noise Ratio) 관점과 정보 이론과의 연결성을 통해 분석합니다. 이 프레임워크를 기반으로, 논문은 추론 과정의 성능을 향상시키기 위한 일반화된 역방향 방정식을 개발합니다. 또한, Non-Markovian 연속 변분 확산 모델(Continuous Variational Diffusion Model)을 개발하여 전방 분포를 정확하게 유도합니다.



### Medical Graph RAG: Towards Safe Medical Large Language Model via Graph Retrieval-Augmented Generation (https://arxiv.org/abs/2408.04187)
- **What's New**: MedGraphRAG: 새로운 그래프 기반 RAG 프레임워크를 소개합니다. 특히 의료 분야를 위해 설계되었으며, 대규모 언어 모델(LLM)의 기능을 향상시키고 증거 기반의 결과를 생성하여 개인 의료 데이터를 처리할 때 안전성과 신뢰성을 향상시킵니다.



### MultiColor: Image Colorization by Learning from Multiple Color Spaces (https://arxiv.org/abs/2408.04172)
- **What's New**: 이 논문은 이미지 컬러라이제이션(colorization)에서 다양한 컬러 공간(color space)의 장점을 활용하는 새로운 학습 기반 방법인 MultiColor를 제시합니다. 기존 방식은 단일 컬러 공간에 의존하여 컬러라이제이션 결과에 편향(bias)이 발생할 수 있지만, MultiColor는 RGB, HSV, CIE-Lab 등 다양한 컬러 공간을 활용하여 더 풍부하고 사실적인 컬러라이제이션 결과를 얻을 수 있습니다.

- **Technical Details**: MultiColor는 여러 컬러 공간을 위한 개별 컬러라이제이션 모듈을 사용합니다. 각 모듈은 트랜스포머 디코더(transformer decoder)를 사용하여 컬러 쿼리 임베딩(color query embedding)을 정제하고, 컬러 매퍼(color mapper)를 사용하여 임베딩과 의미적 특징(semantic features)을 이용해 컬러 채널(color channel)을 예측합니다. 이렇게 예측된 여러 컬러 공간의 컬러 채널을 사용하여 컬러 공간 보완 네트워크(color space complementary network)가 설계되고, 이를 통해 다양한 컬러 정보를 활용하여 자연스럽고 합리적인 컬러라이제이션 이미지를 생성합니다.

- **Performance Highlights**: ImageNet, COCO-Stuff, ADE20K 등 다양한 데이터셋에서 실험을 통해 기존 최첨단 기술보다 뛰어난 성능을 보여줍니다. 특히 복잡한 컬러 상호 작용(color interactions)이 있는 콘텐츠에 대해서도 더 나은 컬러라이제이션 결과를 제공합니다.



### Rotation center identification based on geometric relationships for rotary motion deblurring (https://arxiv.org/abs/2408.04171)
- **What's New**: This paper proposes a geometric-based method to accurately determine the rotation center for non-blind rotary motion deblurring (RMD) in assembled imaging systems. Unlike existing algorithm-based methods, this approach leverages the fixed position of the rotation center in such systems, leading to a more robust and precise identification. The theoretical error is analyzed to be less than 0.5 pixels along a single axis.



### M2EF-NNs: Multimodal Multi-instance Evidence Fusion Neural Networks for Cancer Survival Prediction (https://arxiv.org/abs/2408.04170)
- **What's New**: 본 연구에서는 암 생존 예측을 위한 새로운 신경망 모델인 M2EF-NNs를 제안합니다. 이 모델은 다중 모달 및 다중 인스턴스 증거 융합 기술을 활용하여 암 생존 예측 정확도를 높입니다. 특히, 이미지의 글로벌 정보를 포착하기 위해 사전 훈련된 비전 트랜스포머(ViT) 모델을 사용하여 병리학적 이미지의 패치 특징 임베딩을 얻습니다. 그런 다음, 유전체 임베딩을 쿼리로 사용하는 다중 모달 어텐션 모듈을 도입하여 유전체 및 병리학적 이미지 간의 공동 어텐션 매핑을 학습하여 다중 모달 정보의 초기 상호 작용 융합을 달성하고 상관 관계를 더 잘 포착합니다. 이 연구는 암 생존 예측에 덤스터-셰이퍼 증거 이론(DST)을 적용한 첫 번째 사례입니다. 처리된 다중 모달 특징을 사용하여 클래스 확률 분포를 매개변수화하고 주관적 논리를 도입하여 서로 다른 모달과 관련된 불확실성을 추정합니다. 덤스터-셰이퍼 이론과 결합하여 다중 모달 융합 후 클래스 확률의 가중치를 동적으로 조정하여 신뢰할 수 있는 생존 예측을 달성할 수 있습니다.



### Decorrelating Structure via Adapters Makes Ensemble Learning Practical for Semi-supervised Learning (https://arxiv.org/abs/2408.04150)
- **What's New**: 본 논문에서는 Decorrelating Structure via Adapters (DSA)라는 경량화, 손실 함수 없는 아키텍처 독립적 앙상블 학습 방법을 제안합니다. DSA는 구조적으로 다양한 어댑터를 활용하여 다양한 시각적 작업에 대한 여러 예측 헤드의 상관 관계를 제거합니다. DSA는 추가적인 정규화 또는 손실 없이 다양한 아키텍처에 쉽게 확장 가능합니다. 이론적 분석에 따르면 DSA는 단일 헤드 기반 방법보다 낮은 편향과 분산을 갖습니다.



### ComKD-CLIP: Comprehensive Knowledge Distillation for Contrastive Language-Image Pre-traning Mod (https://arxiv.org/abs/2408.04145)
Comments:
          first submit

- **What's New**: 본 논문은 ComKD-CLIP이라는 새로운 방법을 제안합니다. ComKD-CLIP은 큰 CLIP 모델의 지식을 작은 모델에 효과적으로 전달하는 새로운 지식 증류 방법입니다. 이를 통해 작은 모델의 성능을 획기적으로 향상시켜 컴퓨팅 자원이 제한적인 환경에서도 CLIP 모델을 효과적으로 활용할 수 있도록 돕습니다.

- **Technical Details**: ComKD-CLIP은 이미지 특징 정렬(IFAlign)과 교육적 주의(EduAttention)라는 두 가지 핵심 메커니즘을 포함합니다. IFAlign은 학생 모델이 추출한 이미지 특징이 교사 모델이 추출한 이미지 특징과 일치하도록 합니다. 이는 학생 모델이 교사 모델이 이미지 특징을 추출하는 방식에 대한 지식을 습득할 수 있도록 합니다. EduAttention은 교사 모델이 추출한 텍스트 특징과 학생 모델이 추출한 이미지 특징 사이의 상관관계를 분석합니다. 이를 통해 학생 모델은 교사 모델이 텍스트와 이미지 특징을 통합하는 방식을 이해하고 모방하여 다중 모드 정보를 더 잘 이해할 수 있도록 합니다. 또한 ComKD-CLIP은 IFAlign과 EduAttention에서 증류된 지식을 교사 모델의 특징 융합 결과를 활용하여 개선합니다. 이를 통해 학생 모델이 교사 모델의 지식을 더 정확하게 흡수할 수 있습니다.

- **Performance Highlights**: 본 논문에서 제시된 ComKD-CLIP은 다양한 인식 데이터셋에서 기존의 최첨단 방법보다 뛰어난 성능을 보여주었습니다. 11개의 다양한 데이터셋에서 실시된 실험 결과 ComKD-CLIP은 8개의 데이터셋에서 가장 우수한 성능을 보였습니다. 이는 ComKD-CLIP이 다양한 상황에서 효과적이고 강력한 지식 증류 방법임을 입증합니다.



### Integrated Dynamic Phenological Feature for Remote Sensing Image Land Cover Change Detection (https://arxiv.org/abs/2408.04144)
- **What's New**: This paper introduces the InPhea model for remote sensing image change detection (CD) that effectively addresses the challenge of differentiating actual changes from pseudo-changes caused by phenological characteristics in natural areas. This model leverages joint phenological features to improve accuracy in complex scenes.



### PaveCap: The First Multimodal Framework for Comprehensive Pavement Condition Assessment with Dense Captioning and PCI Estimation (https://arxiv.org/abs/2408.04110)
- **What's New**: 본 연구에서는 포장 상태 평가를 위한 최초의 다중 모달 접근 방식을 소개합니다. 이는 정량적인 포장 상태 지수(PCI, Pavement Condition Index) 예측과 질적 설명을 모두 제공합니다. 연구진은 자동 포장 상태 평가를 위한 새로운 프레임워크인 PaveCap을 소개합니다. 이 프레임워크는 단일 샷 PCI 추정 네트워크와 밀집 캡션 네트워크의 두 가지 주요 부분으로 구성됩니다. PCI 추정 네트워크는 YOLOv8을 이용하여 객체 감지(object detection), SAM(Segment Anything Model)을 이용하여 제로 샷 분할(zero-shot segmentation)을 수행하며, 4층 합성곱 신경망을 통해 PCI를 예측합니다. 밀집 캡션 네트워크는 YOLOv8 백본, 트랜스포머 인코더-디코더 아키텍처, 합성곱 피드 포워드 모듈을 사용하여 포장 상태에 대한 상세한 설명을 생성합니다. 이러한 네트워크를 학습 및 평가하기 위해 연구진은 바운딩 박스 주석, 텍스트 주석, PCI 값이 포함된 포장 데이터 세트를 개발했습니다. PCI 추정 네트워크의 결과는 예측된 PCI와 실제 PCI 간의 강력한 양의 상관 관계(0.70)를 보여주었으며, 이는 조건 평가 자동화의 효율성을 입증합니다. 또한 밀집 캡션 네트워크는 높은 BLEU(0.7445), GLEU(0.5893), METEOR(0.7252) 점수로 입증된 정확한 포장 상태 설명을 생성했습니다. 또한 밀집 캡션 모델은 복잡한 시나리오를 잘 처리했으며, 심지어 정답 데이터의 오류를 일부 수정했습니다. 여기서 개발된 프레임워크는 포장 유지 관리 분야에서 인프라 관리 및 의사 결정을 크게 개선할 수 있습니다.



### Decoding Visual Sentiment of Political Imagery (https://arxiv.org/abs/2408.04103)
- **What's New**: 본 연구에서는 시청자들이 시각적 감정에 대한 의견이 일치하지 않을 때, 시각적 감정을 정의하는 새로운 방법을 제시합니다. 특히, 정치적 성향과 같은 사회적 분열이 감정 라벨링에 큰 영향을 미친다는 점을 인지하여, 이러한 분열을 반영하는 데이터 세트를 개발했습니다. 그런 다음 다양한 이념적 관점에서 시각적 감정을 예측하는 딥 러닝 멀티태스크 멀티클래스 모델을 훈련했습니다. 이 방법을 이민 관련 이미지에 적용하여 민주당과 공화당의 관점을 모두 포착했습니다. 라벨링 및 모델 훈련 과정에 다양한 관점을 통합함으로써, 이 연구는 라벨 모호성의 한계를 해결하고 시각적 감정 예측의 정확성을 향상시켰습니다. 전반적으로 본 연구는 인간이 생성한 감정을 더 정확하게 반영하는 분류기를 만드는 데 중점을 두어, 시각적 감정 해독의 패러다임 전환을 옹호합니다.



### ArtVLM: Attribute Recognition Through Vision-Based Prefix Language Modeling (https://arxiv.org/abs/2408.04102)
Comments:
          Accepted at ECCV 2024

- **What's New**: 이 논문은 대규모 이미지-텍스트 기반 모델을 사용하여 시각적 속성 인식 문제를 해결하는 새로운 접근 방식을 제시합니다. 이는 '생성적 검색'(generative retrieval)이라는 새로운 개념을 도입하여 기존의 '대조 검색'(contrastive retrieval)의 단점을 극복합니다. 생성적 검색은 시각적 속성과 객체의 관계를 조건부 확률 그래프로 모델링하고, 이를 바탕으로 시각적 속성을 인식하는 문제를 언어 모델링 문제로 변환합니다. 이를 통해 대규모 사전 훈련된 비전-언어 모델(VLM)의 지식을 효과적으로 활용하여 이미지에서 시각적 속성을 인식하는 성능을 향상시킵니다. 특히, 이미지에서 인식해야 할 각 속성에 대해, 이미지 내 객체와 속성의 관계를 나타내는 짧은 문장을 생성할 확률을 계산합니다. 이러한 접근 방식은 문장 내 객체와 속성의 순서와 의존성을 고려하여 대조 검색과 비교하여 더욱 정확한 결과를 제공합니다.



### PushPull-Net: Inhibition-driven ResNet robust to image corruptions (https://arxiv.org/abs/2408.04077)
Comments:
          Accepted at ICPR 2024, code available at this https URL

- **What's New**: 본 논문은 ResNet 아키텍처의 첫 번째 레이어에 **PushPull-Conv**라는 새로운 계산 유닛을 소개합니다. 이 유닛은 시각 피질에서 관찰되는 **위상 반전 억제 (anti-phase inhibition)** 현상에서 영감을 받았습니다. PushPull-Conv는 기존의 합성곱 계층을 재정의하여 학습 가능한 **푸시 커널 (push kernel)**과 그에 상응하는 **풀 커널 (pull kernel)** 쌍을 구현합니다. 푸시 커널은 기존의 합성곱과 유사하게 특정 자극에 반응하도록 학습하지만, 풀 커널은 동일한 자극에 대해 반대되는 대비로 반응합니다. 이 구성은 자극 선택성을 향상시키고 선호하는 자극이 없는 영역에서 반응을 효과적으로 억제합니다. 이 효과는 푸시 및 풀 커널이 해당 영역에서 비슷한 크기의 응답을 생성하여 서로 상쇄시키기 때문입니다.



### AEye: A Visualization Tool for Image Datasets (https://arxiv.org/abs/2408.04072)
Comments:
          Accepted at IEEE VIS 2024

- **What's New**: AEye는 이미지 데이터셋을 시각화하고 탐색하기 위한 새로운 도구입니다. AEye는 대규모 이미지 데이터셋을 이해하기 쉽게 구성하고 시각화하기 위해 CLIP(Contrastive Language-Image Pretraining) 임베딩 공간을 활용합니다. 이미지는 CLIP 임베딩을 기반으로 2차원 평면에 배치되어 사용자는 직관적으로 탐색할 수 있습니다. AEye는 또한 이미지 및 텍스트 검색 기능을 제공하여 데이터셋을 탐색할 수 있는 기능을 제공합니다.



### Task-oriented Sequential Grounding in 3D Scenes (https://arxiv.org/abs/2408.04034)
Comments:
          website: this https URL

- **What's New**: 새로운 3D 시퀀셜 그라운딩(Sequential Grounding) 작업 제안 및 대규모 데이터셋 SG3D 공개



### HiRISE: High-Resolution Image Scaling for Edge ML via In-Sensor Compression and Selective ROI (https://arxiv.org/abs/2408.03956)
Comments:
          10 pages, 8 figures

- **What's New**: HiRISE, a high-resolution image scaling system for edge ML is proposed to address the challenges of high-resolution images in tiny IoT devices, especially for applications like face recognition that require rich features. HiRISE leverages in-sensor image scaling to significantly reduce peak memory requirements, data transfer, and energy consumption. It achieves up to 17.7x reduction in data transfer and energy consumption.



### Histopathology image embedding based on foundation models features aggregation for patient treatment response prediction (https://arxiv.org/abs/2408.03954)
Comments:
          Accepted at MICCAI 2024 workshop MOVI

- **What's New**: 본 논문에서는 확산성 대세포 림프종 (DLBCL) 환자의 치료 반응을 예측하기 위해 새로운 방법론을 제안합니다. 이 방법은 여러 기초 모델을 패치 기반 특징 추출기로 사용하여 이미지의 국소적 표현을 얻고, 주의 기반 다중 인스턴스 학습 (MIL)을 사용하여 이러한 국소적 표현을 집계하여 이미지의 전역적 표현을 얻습니다.

- **Technical Details**: 이 방법은 여러 기초 모델을 특징 추출기로 사용하여 이미지의 패치 기반 국소적 표현을 추출하고, 이를 주의 기반 MIL을 사용하여 집계하여 이미지의 전역적 표현을 얻습니다. 기초 모델은 이미지넷(ImageNet) 사전 훈련된 모델보다 뛰어난 성능을 보이며, 조직 특성을 보다 잘 나타내는 것을 보여줍니다.

- **Performance Highlights**: 152명의 환자 데이터 세트에 대한 실험 연구에서 제안된 방법론의 유망한 결과를 보여주었습니다. 특히, 기초 모델을 사용하는 것이 기존 이미지넷 사전 훈련된 모델에 비해 치료 반응 예측 작업에서 유리하다는 것을 강조했습니다.

- **Keywords**: Diffuse Large B-Cell Lymphoma (DLBCL), Whole Slide Images (WSI), Foundation Models, Multiple Instance Learning (MIL), Attention-based Aggregation



### Taxonomy Driven Fast Adversarial Training (https://arxiv.org/abs/2408.03944)
Comments:
          This paper is accepted by AAAI

- **What's New**: 본 논문은 단일 단계 적대적 학습(single-step adversarial training, AT)에서 발생하는 심각한 과적합(catastrophic overfitting, CO) 현상을 해결하기 위해, 적대적 예제(adversarial examples, AEs) 분류 체계를 제안합니다. 이 분류 체계는 다양한 AEs가 학습에 미치는 영향을 분석하여 CO가 발생하는 근본적인 원인을 밝혀냅니다. 이를 기반으로, 논문은 적대적 예제 분류 체계 기반의 빠른 적대적 학습(Taxonomy Driven fast Adversarial Training, TDAT)을 제안합니다. TDAT는 동적 레이블 완화(dynamic label relaxation), 배치 모멘텀 초기화(batch momentum initialization), 분류 기반 손실 함수(taxonomy driven loss function)를 통합하여 AT의 효율성을 크게 향상시킵니다.

- **Technical Details**: TDAT는 다음과 같은 세 가지 주요 요소를 통합합니다. 
1. **동적 레이블 완화(dynamic label relaxation)**: 학습 과정에서 AEs의 분류 변화를 고려하여 레이블을 유연하게 조정합니다. 
2. **배치 모멘텀 초기화(batch momentum initialization)**: 초기화 단계에서 모멘텀을 적용하여 AEs에 대한 과적합을 방지합니다. 
3. **분류 기반 손실 함수(taxonomy driven loss function)**: AEs의 분류 특성을 반영하여 손실 함수를 재정의합니다.

- **Performance Highlights**: 실험 결과, TDAT는 기존의 단일 단계 및 다단계 AT 방식에 비해 우수한 성능을 보여줍니다. 특히 CIFAR-10, CIFAR-100, Tiny ImageNet, ImageNet-100 데이터셋에서 PGD10 공격에 대한 견고성이 각각 1.59%, 1.62%, 0.71%, 1.26% 향상되었습니다. 또한 TDAT는 다른 공격에도 최첨단 성능을 달성합니다.



### Arctic-TILT. Business Document Understanding at Sub-Billion Sca (https://arxiv.org/abs/2408.04632)
- **What's New**: Arctic-TILT: a lightweight document understanding (DU) model that achieves comparable accuracy to models 1000 times its size, enabling cost-effective and efficient processing of visually rich documents. It excels on various benchmarks such as MP-DocVQA, DUDE, and ArXiv-Lay, demonstrating its versatility and performance. Arctic-TILT also provides reliable confidence scores and fast inference, making it suitable for large-scale and time-sensitive applications in enterprise environments.

- **Technical Details**: Arctic-TILT builds upon the TILT encoder-decoder model, introducing a novel modality fusion technique that integrates textual and visual information within each transformer block, inspired by tensor product representations. It incorporates attention sparsity to handle longer documents efficiently by restricting attention calculations to a local neighborhood.  Arctic-TILT is optimized for deployment on single GPUs, enabling efficient processing of documents with up to 400k tokens.

- **Performance Highlights**: Arctic-TILT demonstrates state-of-the-art results on seven diverse document understanding benchmarks, including MP-DocVQA, DUDE, Kleister, ArXiv-Lay, and PubMed-Lay. It achieves comparable accuracy to much larger models, making it a compelling choice for cost-efficient document understanding tasks.



### LogogramNLP: Comparing Visual and Textual Representations of Ancient Logographic Writing Systems for NLP (https://arxiv.org/abs/2408.04628)
- **What's New**: 본 논문은 고대 상형 문자 언어(Logographic language)를 위한 NLP 분석을 위한 새로운 벤치마크인 **LogogramNLP**를 소개합니다. 이 벤치마크는 4개의 고대 문자 체계(Linear A, 이집트 상형 문자, 설형 문자, 죽간 문자)에 대한 전사 및 시각 데이터셋과 분류, 번역, 구문 분석과 같은 작업을 위한 주석을 제공합니다. 또한 최근 시각 및 텍스트 인코딩 전략을 백본으로 사용하는 시스템을 비교합니다. 연구 결과는 시각적 표현이 특정 작업에서 텍스트 표현보다 성능이 뛰어나다는 것을 보여줍니다. 이는 시각적 처리 파이프라인이 NLP 기반 분석을 위한 많은 양의 상형 문자 언어 문화 유산 데이터를 활용할 수 있음을 시사합니다.



### Quantifying the Impact of Population Shift Across Age and Sex for Abdominal Organ Segmentation (https://arxiv.org/abs/2408.04610)
Comments:
          This paper has been accepted for publication by the MICCAI 2024 Fairness of AI in Medical Imaging (FAIMI) Workshop

- **What's New**: 이 논문은 의료 영상 분할 모델의 성능에 대한 인구 변화(population shift)의 영향을 처음으로 연구한 논문이며, 특히 복부 CT 영상에서 나이와 성별에 따른 영향을 조사했습니다. 또한, 새로운 지표인 "성능 차이(performance gap)"를 도입하여 인구 변화가 각 하위 그룹(subgroup)에 미치는 최대 영향을 정량화했습니다.



### Sampling for View Synthesis: From Local Light Field Fusion to Neural Radiance Fields and Beyond (https://arxiv.org/abs/2408.04586)
Comments:
          Article written for Frontiers of Science Award, International Congress on Basic Science, 2024

- **What's New**: 이 논문은 'Local Light Field Fusion (LLFF)' 알고리즘을 소개합니다. 이 알고리즘은 불규칙적인 샘플링된 뷰 그리드에서 실제 장면의 새로운 뷰를 합성하는 방법을 제공합니다. LLFF는 각 샘플링된 뷰를 다중 평면 이미지 표현 (multiplane image scene representation)을 통해 로컬 라이트 필드로 확장하고, 이웃한 로컬 라이트 필드를 블렌딩하여 새로운 뷰를 렌더링합니다. LLFF는 기존의 플레놉틱 샘플링 이론을 확장하여 특정 장면을 얼마나 밀도 있게 샘플링해야 하는지 정확하게 지정하는 경계를 유도합니다.



### Clutter Classification Using Deep Learning in Multiple Stages (https://arxiv.org/abs/2408.04407)
Comments:
          SoutheastCon 2024

- **What's New**: 이 논문은 위성 이미지를 이용하여 딥러닝 기반 환경 장애물(clutter) 유형 자동 인식 기술을 제안합니다. 이 기술은 무선 통신에서의 신호 전파 손실(path loss) 예측 정확도를 향상시키는 데 활용될 수 있습니다.



### Detecting Car Speed using Object Detection and Depth Estimation: A Deep Learning Framework (https://arxiv.org/abs/2408.04360)
Comments:
          This is the pre-print of the paper which was accepted for oral presentation and publication in the proceedings of IEEE CONIT 2024, organized at Pune from June 21 to 23, 2024. The paper is 6 pages long and it contains 11 figures and 1 table. This is not the final version of the paper

- **What's New**: 본 논문은 휴대폰이나 웨어러블 카메라와 같은 휴대용 장치를 사용하여 딥러닝 프레임워크를 통해 차량 속도를 추정하는 새로운 방법을 제시합니다. 이 시스템은 기존의 LIDAR 또는 레이더 기반 속도 추정 장치를 사용하지 않고도 차량 속도를 추정할 수 있습니다. 이는 교통 경찰이 휴대용 장치를 사용하여 과속 차량을 효과적으로 단속할 수 있게 해줄 수 있습니다.



### Enhancing Journalism with AI: A Study of Contextualized Image Captioning for News Articles using LLMs and LMMs (https://arxiv.org/abs/2408.04331)
- **What's New**: 본 연구는 대규모 언어 모델(LLM)과 대규모 멀티모달 모델(LMM)이 뉴스 기사에 첨부된 이미지에 대한 문맥화된 캡션을 생성하여 저널리즘 실무를 지원할 수 있는 방법을 탐구합니다. 저널리즘 분야에서 AI를 통합하는 것은 뉴스 보도의 품질과 효율성을 향상시키는 데 있어 독특한 과제와 기회를 제시합니다.



### Deep Transfer Learning for Kidney Cancer Diagnosis (https://arxiv.org/abs/2408.04318)
Comments:
          32 pages, 8 figures and 8 tables

- **What's New**: 본 논문은 딥러닝 기반의 전이 학습(TL) 프레임워크를 이용한 신장암 진단에 대한 최초의 종합적인 분석을 제시합니다. 이는 전이 학습(TL)이 신장암 진단에 어떻게 활용될 수 있는지, 그리고 이 분야의 현재 과제와 미래 전망을 파악하는 데 도움이 됩니다. 또한 다양한 프레임워크의 장단점을 분석하고, 미래 연구 방향을 제시합니다.



### An Explainable Non-local Network for COVID-19 Diagnosis (https://arxiv.org/abs/2408.04300)
- **What's New**: 이 논문에서는 COVID-19, 일반 폐렴 및 정상을 포함한 CT 이미지를 분류하기 위해 새로운 딥 레지듀얼 3D 어텐션 논로컬 네트워크(NL-RAN)를 제안하여 빠르고 설명 가능한 COVID-19 진단을 수행합니다. 이 연구에서는 전반적인 학습을 달성할 수 있는 딥 레지듀얼 3D 어텐션 논로컬 네트워크를 구축했습니다. 네트워크에는 전역 정보를 포착하기 위한 논로컬 모듈이 포함되어 있으며, 3D 어텐션 모듈은 병변의 세부 사항에 집중하여 3D 폐 CT를 직접 분석하고 분류 결과를 출력할 수 있도록 했습니다. 어텐션 모듈의 출력은 히트 맵으로 사용되어 모델의 해석 가능성을 높일 수 있습니다.

- **Technical Details**: NL-RAN은 3D 레지듀얼 네트워크를 기반으로 하며, 3D 어텐션 모듈과 논로컬 모듈을 통합하여 병변의 세부 사항에 집중하고 전역 정보를 포착하여 성능을 향상시킵니다. 3D 어텐션 모듈은 병변의 세부 사항에 집중하여 더 정확한 분류를 가능하게 하고, 논로컬 모듈은 이미지의 전체적인 맥락을 이해하여 이미지 내 병변 간의 관계를 학습합니다. 또한, 어텐션 맵을 히트 맵으로 사용하여 모델의 해석 가능성을 높였습니다.

- **Performance Highlights**: 4079개의 3D CT 스캔을 사용하여 모델을 학습했으며, NL-RAN은 AUC 0.9903, 정밀도 0.9473, F1-스코어 0.9462를 달성하여 비교 대상인 다른 분류 방법을 능가했습니다. 또한, 어텐션 모듈에서 출력된 히트 맵은 CAM에서 출력된 히트 맵보다 더 명확하게 감염된 영역을 보여주었습니다. 이는 NL-RAN이 빠르고 정확한 COVID-19 진단을 가능하게 하며, 의료 분야에서 유용하게 활용될 수 있음을 의미합니다.



### Efficient and Accurate Pneumonia Detection Using a Novel Multi-Scale Transformer Approach (https://arxiv.org/abs/2408.04290)
- **What's New**: 이 연구는 폐렴 진단을 위한 새로운 딥러닝 기반 접근 방식을 제시합니다. 이 방법은 트랜스포머 기반 어텐션 메커니즘과 딥러닝을 결합하여 흉부 X선 영상에서 폐렴을 감지하는 능력을 향상시킵니다. 특히, 일반적인 트랜스포머에 비해 매개변수 수가 적으면서도 성능은 유지하는 새로운 트랜스포머 모듈을 소개합니다.



### SG-JND: Semantic-Guided Just Noticeable Distortion Predictor For Image Compression (https://arxiv.org/abs/2408.04273)
Comments:
          Accepted by ICIP 2024

- **What's New**: 이 논문에서는 주의할 만한 왜곡(JND, Just Noticeable Distortion) 예측을 위해 의미 정보를 활용하는 새로운 Semantic-Guided JND(SG-JND) 네트워크를 제안합니다. SG-JND는 이미지의 의미 수준 패치를 추출하는 이미지 전처리 모듈, 크로스 스케일 어텐션 레이어를 사용하여 다층 기능을 추출하는 기능 추출 모듈, 추출된 기능을 최종 JND 값으로 회귀하는 JND 예측 모듈의 세 가지 필수 모듈로 구성됩니다. 기존의 JND 예측 방법은 픽셀 수준 또는 서브 밴드 수준의 특징에만 의존했지만, SG-JND는 이미지 내용의 영향을 포착할 수 있다는 장점이 있습니다. 특히, SG-JND는 고수준의 의미 정보를 활용하여 다층 기능을 융합하여 의미를 인식하는 JND 기능 추출을 달성합니다.



### Physical prior guided cooperative learning framework for joint turbulence degradation estimation and infrared video restoration (https://arxiv.org/abs/2408.04227)
Comments:
          21

- **What's New**: 본 논문에서는 대기 난류 강도 추정과 적외선 이미지 복원을 동시에 개선하는 P2GCL(Physical Prior Guided Cooperative Learning) 프레임워크를 소개합니다. P2GCL은 TMNet과 TRNet이라는 두 모델 간의 순환적 협력을 기반으로 합니다. TMNet은 난류 강도를 측정하여 굴절률 구조 상수(Cn2)를 물리적 prior로 출력하고, TRNet은 Cn2를 기반으로 적외선 이미지 시퀀스 복원을 수행하며 복원된 이미지를 TMNet에 다시 제공하여 측정 정확도를 향상시킵니다. 물리적 이론과 훈련 과정을 일치시키기 위해 Cn2-guided frequency 손실 함수와 물리적 제약 손실이 새롭게 도입되었습니다.



### Is SAM 2 Better than SAM in Medical Image Segmentation? (https://arxiv.org/abs/2408.04212)
- **What's New**: 이 연구는 최근 발표된 Segment Anything Model 2 (SAM 2) 모델이 의료 이미지 분할에서 원래의 SAM 모델보다 더 나은 성능을 보이는지 평가합니다. SAM 2는 비디오 분할 기능을 추가한 SAM의 업데이트 버전입니다. 이 연구는 SAM과 SAM 2의 성능을 여러 의료 영상 데이터 세트에서 비교하여 2D 의료 이미지 분할에서 SAM 2가 SAM을 능가하는지 평가합니다.



### pyBregMan: A Python library for Bregman Manifolds (https://arxiv.org/abs/2408.04175)
Comments:
          28 pages

- **What's New**: pyBregMan이라는 Python 라이브러리 (Python library)는 Bregman manifold (Bregman 매니폴드)에 대한 일반적인 연산을 구현하고 정보 과학에서 사용되는 여러 가지 일반적인 Bregman manifold를 구현합니다.  pyBregMan은 Legendre-Fenchel duality (르장드르-펜쉘 듀얼리티) 개념을 핵심으로 하여 쌍대 포텐셜 함수 (dual potential function)와 쌍대 Bregman divergence (dual Bregman 발산)를 생성합니다.  또한 카테고리/다항 분포 (categorical/multinomial distributions)와 다변수 정규 분포 (multivariate normal distributions)의 Fisher-Rao manifold도 구현합니다.



### Efficient Single Image Super-Resolution with Entropy Attention and Receptive Field Augmentation (https://arxiv.org/abs/2408.04158)
Comments:
          Accepted to ACM MM 2024

- **What's New**: 본 논문은 효율적인 싱글 이미지 슈퍼 해상도 (ESISR)를 위한 새로운 Entropy Attention and Receptive Field Augmentation (EARFA) 모델을 제안합니다. EARFA 모델은 정보 이론적 관점에서 모델의 연산 오버헤드를 줄이고 효과적인 수용 필드를 확장하는 것을 목표로 합니다. 특히, EA는 중간 특징의 엔트로피를 높여 후속 추론을 위한 정보 입력을 늘리는 데 집중합니다. SLKA는 채널 이동을 활용하여 모델의 수용 필드를 확장하고 계층적 특징의 다양성을 증진시킵니다. 두 방법 모두 복잡한 연산을 포함하지 않아 모델 추론 속도를 크게 향상시킵니다.



### Can Rule-Based Insights Enhance LLMs for Radiology Report Classification? Introducing the RadPrompt Methodology (https://arxiv.org/abs/2408.04121)
Comments:
          Accepted at BioNLP, ACL 2024

- **What's New**: 이 연구는 **RadPert** (룰 기반 시스템)과 **RadPrompt** (멀티턴 프롬프팅 전략)을 소개하며, 이를 통해 의료 이미지 분석에서 **원격 감독**(distant supervision)을 개선하고 **대규모 언어 모델**(LLM)의 성능을 향상시키는 새로운 방법을 제시합니다.

- **Technical Details**: RadPert는 **RadGraph 지식 그래프**와 함께 **불확실성 인식 정보 스키마**를 통합하여 룰 기반 시스템의 견고성을 높였습니다. RadPrompt는 RadPert를 활용하여 **LLM의 제로 샷 예측 능력**을 강화하는 **멀티턴 프롬프팅 전략**입니다.

- **Performance Highlights**: RadPert는 기존 룰 기반 SOTA인 **CheXpert**를 능가하는 성능을 보였습니다. 또한, RadPrompt는 **GPT-4 Turbo**와 **RadPert** 모두를 뛰어넘는 성능을 달성하여 LLM과 룰 기반 모델의 상호작용 가능성을 입증했습니다.



### The Quest for Early Detection of Retinal Disease: 3D CycleGAN-based Translation of Optical Coherence Tomography into Confocal Microscopy (https://arxiv.org/abs/2408.04091)
Comments:
          30 pages, 11 figures, 5 tables

- **What's New**: 본 논문은 광간섭 단층촬영(OCT) 이미지를 시체 해부 현미경 이미지로 변환하는 3차원 CycleGAN 기반의 새로운 프레임워크를 제안합니다. 이는 OCT와 시체 해부 현미경의 장점을 결합하여 망막 영상의 진단 및 모니터링 기능을 향상시키는 것을 목표로 합니다. 특히, OCT의 3차원 정보를 시체 해부 현미경의 풍부하고 상세한 색상 영역으로 변환하는 최초의 시도입니다.



### Multi-scale structural complexity as a quantitative measure of visual complexity (https://arxiv.org/abs/2408.04076)
Comments:
          16 pages, 11 figures, 2 tables

- **What's New**: This paper introduces a new measure for visual complexity called "Multi-Scale Structural Complexity" (MSSC), which quantifies complexity based on the dissimilarity between different scales in the hierarchical organization of an image. It departs from traditional measures that focus on informational complexity, which primarily consider randomness, and instead captures the structural complexity that humans intuitively perceive.



### Do Sharpness-based Optimizers Improve Generalization in Medical Image Analysis? (https://arxiv.org/abs/2408.04065)
- **What's New**: 이 논문은 의료 영상에서 딥러닝 모델의 일반화 성능을 향상시키는 최근의 **sharpness-based** (날카로움 기반) 최적화 방법들을 검토하고, 특히 유방 초음파 영상에서 이러한 방법들의 효과를 평가합니다. 특히, **Sharpness-Aware Minimization (SAM)**과 그 변형들을 중점적으로 살펴보며 실제 의료 영상에 대한 적용 가능성을 연구합니다.



### Image-to-LaTeX Converter for Mathematical Formulas and Tex (https://arxiv.org/abs/2408.04015)
Comments:
          4 pages

- **What's New**: 이 연구는 이미지에서 LaTeX 코드를 생성하는 비전 인코더-디코더 모델을 훈련합니다. 이 모델은 컴퓨터 생성 이미지로 학습된 Swin Transformer 인코더와 GPT-2 디코더로 구성된 기본 모델과 손으로 쓴 수식으로 미세 조정된 LoRA(Low-Rank Adaptation) 모델을 포함합니다.



New uploads on arXiv(cs.AI)

### SCENE: Evaluating Explainable AI Techniques Using Soft Counterfactuals (https://arxiv.org/abs/2408.04575)
Comments:
          10 pages, 5 tables

- **What's New**: SCENE (Soft Counterfactual Evaluation for Natural language Explainability)는 자연어 처리 (NLP) 작업에서 AI 모델의 투명성과 책임성을 높이기 위한 새로운 평가 방법입니다. SCENE은 대규모 언어 모델 (LLM)을 활용하여 제로 샷 방식으로 Soft Counterfactual 설명을 생성합니다. SCENE은 토큰 기반 치환에 초점을 맞춰 광범위한 미세 조정 없이 문맥적으로 적절하고 의미적으로 의미있는 Soft Counterfactuals를 생성합니다. SCENE은 Validitysoft와 Csoft 지표를 채택하여 텍스트 분류 작업에서 모델 독립적인 XAI 방법의 효율성을 평가합니다. SCENE은 CNN, RNN 및 BERT 아키텍처에 적용되어 다양한 XAI 기법의 강점과 한계에 대한 귀중한 통찰력을 제공합니다.



### Reasoning about Study Regulations in Answer Set Programming (https://arxiv.org/abs/2408.04528)
Comments:
          To appear in Theory and Practise of Logic Programming

- **What's New**: 본 논문은 대학 학습 규정을 자동화하여 관리자, 교수, 학생 등 다양한 이해관계자를 위한 시스템을 제안합니다. 특히, 포츠담 대학교의 다양한 학습 프로그램 분석을 기반으로 학습 규정의 기본 원칙을 정의하고, 이를 통해 합법적인 학습 계획의 특징을 밝힙니다. 또한, Answer Set Programming (ASP)를 사용하여 학습 규정을 인코딩하고, 이를 바탕으로 학습 계획을 생성하는 방법을 제시합니다. 나아가, 이러한 접근 방식을 통해 학습 계획을 탐색할 수 있는 사용자 인터페이스를 구축할 수 있습니다.



### RiskAwareBench: Towards Evaluating Physical Risk Awareness for High-level Planning of LLM-based Embodied Agents (https://arxiv.org/abs/2408.04449)
- **What's New**: 본 논문은 LLM 기반의 로봇이 현실 세계에서 물리적 위험을 인식하고 완화할 수 있는 능력을 평가하는 자동화된 프레임워크인 **RiskAwareBench**를 제안합니다. 기존의 안전 벤치마크는 주로 언어 기반의 위험에 초점을 맞춘 반면, RiskAwareBench는 **물리적 위험**에 초점을 맞추어 로봇이 수행하는 작업 계획에서 잠재적인 위험을 식별하고 완화할 수 있는지 평가합니다.



### Non-maximizing policies that fulfill multi-criterion aspirations in expectation (https://arxiv.org/abs/2408.04385)
Comments:
          16 pages main text + 4 pages supplement. Accepted for Algorithmic Decision Theory 2024

- **What's New**: 본 논문은 기존의 단일 보상 함수를 최대화하는 대신 다중 평가 지표(evaluation metrics)와 목표 집합(aspiration set)을 사용하여 에이전트를 설계하는 새로운 방법을 제시합니다. 에이전트의 목표는 다중 평가 지표의 기댓값 벡터가 목표 집합 안에 들어오도록 하는 것입니다.



### KnowPC: Knowledge-Driven Programmatic Reinforcement Learning for Zero-shot Coordination (https://arxiv.org/abs/2408.04336)
- **What's New**: 이 논문은 제로 샷 협업(ZSC)을 위한 새로운 접근 방식인 지식 기반 프로그래밍 강화 학습(KnowPC)를 제안합니다. KnowPC는 에이전트의 정책을 해석 가능한 프로그램으로 표현하여 기존의 블랙박스 신경망에 비해 일반화 능력을 향상시킵니다. 특히, KnowPC는 프로그램 구조, 조건부 원시 함수(primitive), 액션 원시 함수(primitive)를 포함하는 도메인 특정 언어(DSL)를 정의합니다. 또한, 효율적인 프로그램 검색을 위해 추출기와 추론기를 통합합니다. 추출기는 다중 에이전트 상호 작용 경로에서 환경 전환 지식을 발견하고, 추론기는 전환 지식을 기반으로 각 액션 원시 함수의 전제 조건을 추론합니다.



### MMRole: A Comprehensive Framework for Developing and Evaluating Multimodal Role-Playing Agents (https://arxiv.org/abs/2408.04203)
- **What's New**: 이 논문은 텍스트 기반의 역할 수행 에이전트 (RPAs) 의 한계를 극복하고, 시각 및 텍스트를 통합하는 멀티모달 역할 수행 에이전트 (MRPAs) 라는 새로운 개념을 제시합니다.  MRPAs는 이미지와 관련된 대화를 통해 특정 캐릭터를 모방하며, 인간 사용자 또는 다른 캐릭터와 상호 작용할 수 있습니다.

- **Technical Details**: MMRole 이라는 포괄적인 프레임워크는 MRPAs의 개발 및 평가를 위한 솔루션을 제시하며,  이는 대규모 고품질 데이터셋 (MMRole-Data) 과 강력한 평가 방법 (MMRole-Eval) 을 포함합니다. MMRole-Data 는 85 개의 캐릭터, 11,000 개의 이미지, 14,000 개의 대화로 구성되어 있으며, MRPAs를 위한 훈련 및 테스트 샘플을 제공합니다. MMRole-Eval 은  MRPAs 의 기본적인 대화 능력, 멀티모달 이해 능력, 역할 수행 능력 등 세 가지 차원에서 8 가지 지표를 사용하여 MRPAs를 평가합니다.  특히, MMRole-Eval 은 MRPAs의 성능을 비교하기 위해 사전에 구축된 기준 데이터를 기반으로 보상 모델을 학습시켜 MRPAs를 평가하는 방식을 채택합니다.

- **Performance Highlights**: MMRole-Agent 는 MMRole 프레임워크를 기반으로 개발된 최초의 특수 MRPA이며, 일반 대화 멀티모달 모델과 비교하여 향상된 성능을 보여줍니다.  이러한 결과는 멀티모달 이해 능력 및 역할 수행 일관성 향상에 대한 필요성을 강조하며, 향후 MRPA 개발의 중요한 과제를 제시합니다.



### Perceive, Reflect, and Plan: Designing LLM Agent for Goal-Directed City Navigation without Instructions (https://arxiv.org/abs/2408.04168)
- **What's New**: 이 논문은 도시 탐색 환경에서 랜드마크를 기준으로 목적지 위치를 언어로 설명하는 새로운 AI 에이전트 탐색 방식을 제시합니다. 에이전트는 주변 환경을 관찰하고 랜드마크와 도로 네트워크 연결을 인식하여 명시적인 지침 없이 목표 지점까지 이동할 수 있습니다. 이는 에이전트가 스스로 위치를 파악하고 복잡한 도시 환경의 공간적 표현을 획득해야 하며, 랜드마크가 보이지 않는 경우도 고려해야 하기 때문에 매우 어려운 과제입니다.



### Digital Avatars: Framework Development and Their Evaluation (https://arxiv.org/abs/2408.04068)
Comments:
          This work was presented during the IJCAI 2024 conference proceedings for demonstrations

- **What's New**: This paper presents a novel 'show don't tell' prompting strategy for creating AI-driven digital avatars with enhanced humor, authenticity, and favorability. It introduces Crowd Vote, an adaptation of Crowd Score for evaluating avatar responses based on these qualities. Additionally, a visualization pipeline is developed to showcase the quality of the generated avatar responses, featuring an end-to-end AI-driven framework.

- **Technical Details**: The 'show don't tell' strategy involves providing few-shot examples to the LLM for learning from directly, instead of relying solely on instructions. This strategy focuses on incorporating elements of humor and entertainment into the prompts. An end-to-end AI pipeline is developed, integrating Speech-to-Text (STT), Text-to-Speech (TTS), and Talking Face Synthesis (TFS) technologies for generating realistic avatar responses. Crowd Vote is introduced as a novel evaluation tool that leverages LLMs with assigned personalities to judge the humor, authenticity, and favorability of avatar responses. The paper highlights the use of Donald Trump and Joe Biden avatars as a demonstration.

- **Performance Highlights**: Experiments show that the 'show don't tell' prompting strategy significantly improves humor and interest in avatar responses compared to baseline LLMs and Character.ai. Crowd Vote results demonstrate the superior performance of the proposed strategy, with the prompted LLM receiving a higher percentage of votes for humor compared to the other candidates. The avatar responses are shown to be more humorous, imaginative, and engaging than those generated by other methods. Additionally, the avatar responses for Donald Trump and Joe Biden are found to be more authentic and favorable than their real-world counterparts in certain scenarios.



### NAVINACT: Combining Navigation and Imitation Learning for Bootstrapping Reinforcement Learning (https://arxiv.org/abs/2408.04054)
Comments:
          16 pages, 10 figures

- **What's New**: 본 논문은 로봇 조작 작업을 위한 새로운 학습 프레임워크인 NAVINACT를 제안합니다. NAVINACT는 로봇이 고전적인 모션 플래닝 기반 탐색과 강화 학습 중 어떤 것을 사용할지 동적으로 선택합니다. 또한 탐색 효율성을 높이기 위해 모방 학습 데이터를 사용하여 초기 학습을 수행합니다.



### Puppet-Master: Scaling Interactive Video Generation as a Motion Prior for Part-Level Dynamics (https://arxiv.org/abs/2408.04631)
Comments:
          Project page: this https URL

- **What's New**: Puppet-Master라는 새로운 인터랙티브 비디오 생성 모델을 소개합니다. 이 모델은 부분 레벨 (part-level) 동역학에 대한 모션 (motion) 사전으로 사용될 수 있습니다. 테스트 시간에, 단일 이미지와 희소한 모션 궤적 세트 (즉, 드래그)가 주어지면, Puppet-Master는 주어진 드래그 상호 작용에 충실한 사실적인 부분 레벨 모션을 묘사하는 비디오를 합성할 수 있습니다. 이것은 대규모 사전 훈련된 비디오 확산 모델을 미세 조정하여 달성되며, 이를 위해 드래깅 컨트롤을 효과적으로 주입할 수 있는 새로운 조건화 아키텍처를 제안합니다. 더 중요한 것은, 널리 채택된 공간 주의 모듈을 대체할 수 있는 all-to-first 주의 메커니즘을 소개하는데, 이는 기존 모델의 외관 및 배경 문제를 해결하여 생성 품질을 크게 향상시킵니다. 다른 모션 조건 비디오 생성기는 야생 비디오로 훈련되고 대부분 전체 객체를 이동하는 반면, Puppet-Master는 Objaverse-Animation-HQ라는 새로운 데이터셋으로 훈련됩니다. 이 데이터셋은 큐레이션된 부분 레벨 모션 클립을 제공합니다. 최적이 아닌 애니메이션을 자동으로 필터링하고 의미 있는 모션 궤적을 합성 렌더링에 추가하는 전략을 제안합니다. Puppet-Master는 다양한 범주의 실제 이미지로 잘 일반화되며 실제 세계 벤치마크에서 제로 샷 방식으로 기존 방법을 능가합니다. 자세한 결과는 프로젝트 페이지를 참조하십시오: this http URL.



### LogogramNLP: Comparing Visual and Textual Representations of Ancient Logographic Writing Systems for NLP (https://arxiv.org/abs/2408.04628)
- **What's New**: 본 논문은 고대 상형 문자 언어(Logographic language)를 위한 NLP 분석을 위한 새로운 벤치마크인 **LogogramNLP**를 소개합니다. 이 벤치마크는 4개의 고대 문자 체계(Linear A, 이집트 상형 문자, 설형 문자, 죽간 문자)에 대한 전사 및 시각 데이터셋과 분류, 번역, 구문 분석과 같은 작업을 위한 주석을 제공합니다. 또한 최근 시각 및 텍스트 인코딩 전략을 백본으로 사용하는 시스템을 비교합니다. 연구 결과는 시각적 표현이 특정 작업에서 텍스트 표현보다 성능이 뛰어나다는 것을 보여줍니다. 이는 시각적 처리 파이프라인이 NLP 기반 분석을 위한 많은 양의 상형 문자 언어 문화 유산 데이터를 활용할 수 있음을 시사합니다.



### Transformer Explainer: Interactive Learning of Text-Generative Models (https://arxiv.org/abs/2408.04619)
Comments:
          To be presented at IEEE VIS 2024

- **What's New**: Transformer Explainer는 비전문가가 트랜스포머(Transformer) 모델을 이해할 수 있도록 돕는 새로운 웹 기반 시각화 도구입니다. GPT-2 모델을 사용하여 트랜스포머 아키텍처에 대한 높은 수준의 개요와 수학적 연산 및 모델 구조의 저수준 세부 사항을 볼 수 있습니다. 사용자는 자신의 텍스트 입력을 사용하여 실시간으로 GPT-2 모델을 실행하고 모델이 다음 토큰을 예측하는 방식을 관찰할 수 있습니다.



### Better Alignment with Instruction Back-and-Forth Translation (https://arxiv.org/abs/2408.04614)
- **What's New**: 본 논문에서는 대규모 언어 모델(LLM) 정렬을 위해 세계 지식에 기반한 고품질 합성 데이터를 구축하는 새로운 방법인 **명령어 쌍방향 번역(instruction back-and-forth translation)**을 제안합니다. 웹 코퍼스에서 문서를 사용하여 Li et al.(2023a)가 제안한 쌍방향 번역 방식을 사용하여 합성 명령어를 생성하고 큐레이션하며, 초기 문서를 기반으로 응답을 다시 작성하여 품질을 향상시킵니다. 생성된 (쌍방향 번역 명령어, 다시 작성된 응답) 쌍으로 미세 조정하면 AlpacaEval에서 Humpback, ShareGPT, Open Orca, Alpaca-GPT4, Self-instruct와 같은 다른 일반적인 명령어 데이터셋을 사용하는 것보다 더 높은 승률을 보입니다. 또한 LLM을 사용하여 응답을 다시 작성하면 직접 증류보다 성능이 뛰어나고, 두 가지 생성된 텍스트 분포는 임베딩 공간에서 상당한 차이를 보입니다. 추가 분석을 통해 쌍방향 번역 명령어가 다른 합성 명령어 소스보다 품질이 높고, 응답은 증류에서 얻은 응답보다 다양하고 복잡하다는 것을 보여줍니다. 전반적으로 쌍방향 번역은 웹에서 발견되는 정보의 다양성과 양을 활용하면서도 효과적인 정렬에 필요한 응답 품질을 보장하여 두 세계의 장점을 결합한다는 것을 발견했습니다.



### Inference with the Upper Confidence Bound Algorithm (https://arxiv.org/abs/2408.04595)
Comments:
          17 pages, 1 figure

- **What's New**: 이 논문에서는 UCB 알고리즘이 멀티암 밴딧 문제에서 어떻게 작동하는지, 그리고 이것이 후속 추론 작업에 어떤 영향을 주는지 분석합니다. 특히 UCB 알고리즘이 어떻게 데이터 수집 과정에서 안정성을 유지하고, 이로 인해 각 암의 샘플 평균이 점근적으로 정규 분포를 따르는지 설명합니다. 또한, 암의 개수가 증가하는 상황에서 UCB 알고리즘의 안정성을 분석하고, 특정 조건에서 암이 안정적으로 작동하며 최적의 암 수가 많아지는 것을 증명합니다. (stability property, asymptotic normality, number of arms, near-optimal arms)



### Img-Diff: Contrastive Data Synthesis for Multimodal Large Language Models (https://arxiv.org/abs/2408.04594)
Comments:
          14 pages, 9 figures, 7 tables

- **What's New**: Img-Diff, a novel dataset designed for enhancing fine-grained image recognition in Multimodal Large Language Models (MLLMs), is introduced. This dataset leverages insights from contrastive learning and image difference captioning, focusing on identifying object differences between similar images.

- **Technical Details**: The dataset is created using Stable-Diffusion-XL and advanced image editing techniques. It involves generating pairs of similar images with object replacements, highlighting specific differences. This process utilizes a Difference Area Generator for object differences identification and a Difference Captions Generator for detailed descriptions of those differences. The methodology emphasizes identifying 'object replacement' samples, resulting in a relatively small but high-quality dataset.

- **Performance Highlights**: Fine-tuning state-of-the-art (SOTA) MLLMs like MGM-7B using Img-Diff leads to significant performance improvements in image difference and Visual Question Answering (VQA) tasks. Notably, the trained models surpass SOTA models like GPT-4V and Gemini on the MMVP benchmark. The dataset showcases diversity, quality, and robustness, demonstrating effectiveness in bolstering MLLMs' capabilities in image difference recognition and fine-grained image analysis.

- **Code and Dataset Availability**: The code and dataset are publicly available at (URL provided in the paper) to encourage further research and advancements in multimodal data synthesis and enhancement of MLLM capabilities for image understanding.



### HiLo: A Learning Framework for Generalized Category Discovery Robust to Domain Shifts (https://arxiv.org/abs/2408.04591)
Comments:
          39 pages, 9 figures, 26 tables

- **What's New**: 이 논문은 '일반화된 카테고리 발견 (Generalized Category Discovery, GCD)' 작업에 도메인 이동 (domain shift) 문제를 추가하여 기존 연구와 차별화합니다. 기존 GCD 연구는 라벨이 없는 데이터가 라벨이 있는 데이터와 동일한 도메인에서만 온다고 가정했지만, 이 논문은 라벨이 없는 데이터가 다른 도메인에서도 올 수 있다는 현실적인 상황을 고려합니다. 이러한 상황을 해결하기 위해 'HiLo' 네트워크라는 새로운 방법을 제안합니다. HiLo 네트워크는 고수준 의미론적 특징(High-level semantic features)과 저수준 도메인 특징(Low-level domain features)을 추출하고, 이 두 특징 사이의 상호 정보(mutual information)를 최소화하는 방식으로 동작합니다.



### Sampling for View Synthesis: From Local Light Field Fusion to Neural Radiance Fields and Beyond (https://arxiv.org/abs/2408.04586)
Comments:
          Article written for Frontiers of Science Award, International Congress on Basic Science, 2024

- **What's New**: 이 논문은 'Local Light Field Fusion (LLFF)' 알고리즘을 소개합니다. 이 알고리즘은 불규칙적인 샘플링된 뷰 그리드에서 실제 장면의 새로운 뷰를 합성하는 방법을 제공합니다. LLFF는 각 샘플링된 뷰를 다중 평면 이미지 표현 (multiplane image scene representation)을 통해 로컬 라이트 필드로 확장하고, 이웃한 로컬 라이트 필드를 블렌딩하여 새로운 뷰를 렌더링합니다. LLFF는 기존의 플레놉틱 샘플링 이론을 확장하여 특정 장면을 얼마나 밀도 있게 샘플링해야 하는지 정확하게 지정하는 경계를 유도합니다.



### Unveiling the Power of Sparse Neural Networks for Feature Selection (https://arxiv.org/abs/2408.04583)
- **What's New**: 본 논문은 Sparse Neural Networks (SNNs)를 이용한 특징 선택 (feature selection)에 대한 포괄적인 분석을 제공합니다. 특히, Dynamic Sparse Training (DST) 알고리즘을 사용하여 SNNs를 훈련하는 과정에서 발생하는 여러 측면들을 심층적으로 살펴봅니다. 또한, SNNs의 특징을 고려한 새로운 특징 중요도 지표를 제시하며, 이 지표는 기존 방법들보다 SNNs 내에서 특징의 관련성을 더 잘 파악할 수 있다는 장점을 지닙니다.



### Learning Fine-Grained Grounded Citations for Attributed Large Language Models (https://arxiv.org/abs/2408.04568)
Comments:
          Accepted by ACL 2024 Findings

- **What's New**: 이 논문은 대규모 언어 모델(LLM)이 생성한 텍스트에 인용을 추가하여 환각을 줄이고 검증 가능성을 높이는 새로운 프레임워크인 FRONT를 소개합니다. FRONT는 모델 출력을 세부적인 인용구에 기반으로 하여, 인용의 질을 향상시킬 뿐만 아니라 사용자가 세부적인 검증을 수행할 수 있도록 지원합니다.



### Synchronous Multi-modal Semantic CommunicationSystem with Packet-level Coding (https://arxiv.org/abs/2408.04535)
Comments:
          12 pages, 9 figures

- **What's New**: 이 논문은 다중 모드 의미 통신 시스템에서 시간 및 의미 영역 동기화와 패킷 수준 오류 정정 문제를 해결하는 새로운 접근 방식을 제안합니다. 특히, 3DMM 계수와 텍스트를 의미로 전송하여 동기화된 다중 모드 의미 통신 시스템(SyncSC)을 제안하며, PacSC라는 패킷 수준 오류 정정(FEC) 방법을 도입하여 손실 네트워크 환경에서도 높은 품질의 동기화 전송을 실현합니다. 또한, BERT 기반의 TextPC라는 텍스트 패킷 손실 숨기기 모듈을 도입하여 기존 FEC 방식의 성능을 개선합니다.

- **Technical Details**: **3DMM 계수 및 텍스트 의미 표현:**  얼굴 영상은 3D Morphable Model (3DMM) 계수로 표현되고 음성은 텍스트로 변환되어 의미를 나타냅니다. 
**SyncSC:** 시간 및 의미 영역 동기화를 위해 3DMM 계수와 텍스트는 타임스탬프를 함께 전송됩니다. 
**PacSC (Packet-Level FEC):** 비디오 의미 패킷에 대한 오류 정정을 위해 MAE (Masked Autoencoders) 기반의 패킷 수준 의미 코딩 방법을 사용합니다. 
**TextPC (Text Packet Loss Concealment):**  BERT (Bidirectional Encoder Representations from Transformers)를 기반으로 텍스트 패킷 손실을 숨기는 모듈을 개발하여 전통적인 FEC 방식의 성능을 개선합니다.

- **Performance Highlights**: 시뮬레이션 결과에 따르면, 제안된 SyncSC는 기존 방법에 비해 성능이 우수하며, 손실 네트워크에서도 뛰어난 품질의 동기화 전송을 실현합니다. 특히, 텍스트 패킷 손실률이 높은 환경에서 TextPC는 기존 RS 코딩에 비해 BLEU와 의미 유사성 측면에서 0.1의 성능 향상을 보입니다.



### Towards Synergistic Deep Learning Models for Volumetric Cirrhotic Liver Segmentation in MRIs (https://arxiv.org/abs/2408.04491)
- **What's New**: 이 연구는 복잡한 특징 상호 작용을 모델링하고 다양한 데이터셋에서 일반화하기 위해 새로운 시너지 이론을 제안합니다. 이 이론은 상호 보완적인 잠재 공간을 활용하여 특징 상호 작용을 향상시키는 것을 목표로 합니다. 이 연구는 nnSynergyNet3D 아키텍처를 소개하며, 이 아키텍처는 3D 볼륨에 대한 연속적 및 이산적 잠재 공간을 통합합니다.



### Statistical Framework for Clustering MU-MIMO Wireless via Second Order Statistics (https://arxiv.org/abs/2408.04484)
- **What's New**: 본 논문은 무선 사용자의 채널 공분산 행렬 간의 거리를 분석하여 무선 사용자를 군집화하는 방법을 연구합니다. 특히, 샘플 수와 관측 크기가 동일한 속도로 무한대로 커지는 경우 일관성을 유지하는 다중 샘플 공분산 행렬(SCM) 간의 로그-유클리드 거리 추정기를 고려합니다. 본 논문은 다중 사용자 MIMO(MU-MIMO) 무선 통신 시스템에서 실제 조건에서 클러스터링 알고리즘 성능을 정확하게 예측할 수 있는 통계적 프레임워크를 개발했습니다. 특히, 두 개의 샘플 공분산 행렬에서 계산된 로그-유클리드 거리의 일관성 있는 추정기의 점근적 가우시안성을 확립하는 중심 극한 정리를 제시합니다.



### SegXAL: Explainable Active Learning for Semantic Segmentation in Driving Scene Scenarios (https://arxiv.org/abs/2408.04482)
Comments:
          17 pages, 7 figures. To appear in the proceedings of the 27th International Conference on Pattern Recognition (ICPR), 01-05 December, 2024, Kolkata, India

- **What's New**: 이 논문은 자동 주행 시나리오에서의 의미론적 분할(semantic segmentation)을 위해 'SegXAL'이라는 새로운 설명 가능한 능동 학습(Explainable Active Learning, XAL) 모델을 제안합니다. SegXAL은 효율적으로 라벨이 지정되지 않은 데이터를 활용하고, 사람의 개입(oracle)을 통해 가장 정보가 많은 데이터를 선택적으로 라벨링하여 훈련 데이터를 효율적으로 구축하는 데 중점을 둡니다. 또한, 모델의 의사 결정 과정을 해석 가능하게 함으로써 사람과 AI 간의 협업 지능을 강화합니다.



### FedAD-Bench: A Unified Benchmark for Federated Unsupervised Anomaly Detection in Tabular Data (https://arxiv.org/abs/2408.04442)
Comments:
          8 pages, 1 figure

- **What's New**: This paper introduces FedAD-Bench, a unified benchmark for evaluating unsupervised anomaly detection algorithms in federated learning (FL) environments. It systematically analyzes the performance of recent deep learning anomaly detection models under federated settings, which were typically assessed solely in centralized settings.

- **Technical Details**: FedAD-Bench incorporates diverse datasets and metrics for a holistic evaluation. It focuses on deep learning anomaly detection techniques for tabular data within FL frameworks, addressing the gap in existing research. Key features include support for FL, redesigned data splitting with anomalies excluded from the training set, and a unified set of evaluation metrics to provide robust and unbiased results.

- **Performance Highlights**: The paper highlights key challenges in FL anomaly detection, such as model aggregation inefficiencies and metric unreliability. It also presents insights into FL's regularization effects, showing scenarios where it outperforms centralized approaches due to its inherent ability to mitigate overfitting.



### Enhancing Robustness of Retrieval-Augmented Language Models with In-Context Learning (https://arxiv.org/abs/2408.04414)
Comments:
          10 pages, 2 figures

- **What's New**: 이 연구는 오픈 도메인 질의응답(QA)에서 RALM(Retrieval-Augmented Language Model)의 추론 능력을 향상시키는 컨텍스트 내 학습(In-context Learning) 기반 접근 방식을 소개합니다. 이 방법은 RALM이 검색된 컨텍스트에서 불가능한 상황과 모순되는 정보를 식별하는 능력을 향상시키기 위해 MRC(Machine Reading Comprehension) 데모(cases라고 함)를 통합합니다.  이를 통해 불완전한 검색 시나리오에서 RALM의 견고성을 높입니다.



### Probabilistic energy forecasting through quantile regression in reproducing kernel Hilbert spaces (https://arxiv.org/abs/2408.04405)
Comments:
          12 pages, {Owner/Author | ACM} {2024}. This is the author's version of the work. It is posted here for your personal use. Not for redistribution. The definitive Version of Record will published in this https URL

- **What's New**: This paper explores a non-parametric method, **kernel quantile regression (KQR)**, for **probabilistic energy demand forecasting**, focusing on the DACH region (Germany, Austria, and Switzerland).



### Exploring Reasoning Biases in Large Language Models Through Syllogism: Insights from the NeuBAROCO Datas (https://arxiv.org/abs/2408.04403)
Comments:
          To appear in Findings of the Association for Computational Linguistics: ACL 2024

- **What's New**: This paper evaluates the logical reasoning abilities of Large Language Models (LLMs) using the NeuBAROCO dataset, a manually constructed syllogism dataset in both English and Japanese.  This dataset is based on psychological experiments designed to assess human reasoning capabilities using various forms of syllogisms.



### DIVE: Subgraph Disagreement for Graph Out-of-Distribution Generalization (https://arxiv.org/abs/2408.04400)
- **What's New**: 이 논문은 그래프 기계 학습에서 분포 외(OOD) 일반화 문제를 해결합니다. 기존 그래프 학습 알고리즘은 훈련 및 테스트 데이터 간 균일 분포를 가정하지만, 실제 시나리오에서는 이러한 가정이 성립하지 않아 최적의 성능을 내지 못합니다. 이 문제는 SGD를 통해 훈련된 신경망의 본질적인 단순성 편향(simplicity bias)으로 인해 더욱 악화됩니다. 이는 단순한 특징을 더 복잡하지만 예측 능력이 동일하거나 더 뛰어난 특징보다 선호하는 경향으로 이어집니다. 이러한 편향으로 인해 이미지 인식, 자연어 이해 및 그래프 분류와 같은 다양한 작업에서 OOD 성능에 영향을 미치는 잘못된 상관 관계에 의존하게 됩니다.  이 논문에서는 DIVE라는 새로운 방법을 제안합니다. DIVE는 다양한 모델을 훈련하여 모든 레이블 예측 서브그래프에 집중하여 서브그래프 마스크(subgraph mask)에서 분산을 유도하고 단순 구조 패턴에만 집중하는 모델의 한계를 극복합니다.  DIVE는 모델 간에 추출된 서브그래프의 겹침을 처벌하는 정규화를 사용하여 서로 다른 모델이 고유한 구조적 패턴에 집중하도록 유도합니다.  강력한 OOD 성능을 위한 모델 선택은 검증 정확도를 통해 수행됩니다.  GOOD 벤치마크의 네 개 데이터 세트와 DrugOOD 벤치마크의 한 개 데이터 세트에 대해 테스트한 결과, DIVE는 기존 방법보다 상당한 개선을 보였으며, 단순성 편향을 효과적으로 해결하고 그래프 기계 학습에서 일반화를 향상시킵니다.



### Automated Educational Question Generation at Different Bloom's Skill Levels using Large Language Models: Strategies and Evaluation (https://arxiv.org/abs/2408.04394)
- **What's New**: 이 연구는 5가지 최첨단 대규모 언어 모델(LLM)을 사용하여 블룸의 분류 체계에 따른 다양한 인지 수준의 질문을 생성하는 능력을 조사했습니다. 연구는 LLM이 적절한 정보를 제공받았을 때 다양한 인지 수준의 관련성 있고 고품질의 교육 질문을 생성할 수 있음을 시사하지만, 고려된 5가지 LLM의 성능에는 상당한 차이가 있습니다. 또한 자동 평가는 인간 평가와 동일하지 않다는 것을 보여줍니다.



### MM-Forecast: A Multimodal Approach to Temporal Event Forecasting with Large Language Models (https://arxiv.org/abs/2408.04388)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)을 이용한 다중 모달 시간적 이벤트 예측의 새로운 문제를 연구하며, 특히 이미지를 사용한 시간적 이벤트 예측은 덜 연구되어 왔다는 점을 강조합니다. 이 연구는 이미지가 시간적 이벤트 예측에 어떻게 도움이 될 수 있는지, 그리고 이미지를 LLM 기반 예측 프레임워크에 통합하는 방법에 대해 탐구합니다.



### Judgment2vec: Apply Graph Analytics to Searching and Recommendation of Similar Judgments (https://arxiv.org/abs/2408.04382)
Comments:
          5 pages, 7 figures, 2 tables

- **What's New**: 본 연구는 대만 법원 판결문 유사성 분석 자동화 시스템 개발을 목표로 합니다. 이를 위해 전문가들이 '골든 스탠다드'로 분류한 판결문 데이터셋을 활용하여 '전문가 유사성 점수'를 도출하고, '사건-조항' 관계 기반 지식 그래프를 구축하여 'Node2Vec 유사성 점수'를 산출합니다.

- **Technical Details**: 본 연구는 '코사인 유사성'과 'Node2Vec 유사성' 두 가지 유사성 지표를 비교 분석합니다. 특히 'Node2Vec' 알고리즘을 활용하여 판결문 간의 관계를 추론하고 지식 그래프를 구축하는 기술이 핵심입니다. 이는 기존의 '공법 체계' 기반 연구에서 벗어나 '대륙법 체계' 판결문의 유사성 분석에 적용 가능성을 보여줍니다.

- **Performance Highlights**: 본 연구는 판결문 유사성 분석 자동화를 통해 법률 전문가의 수고를 줄이고, 관련 정보 검색 및 추천 서비스 개발 가능성을 제시합니다. 또한 전문가 평가를 통해 알고리즘 기반 유사성과 인간의 직관적 유사성 간의 차이와 연관성을 분석하여, 법률 분야 AI 기술 개발에 기여할 수 있습니다.



### Anomaly Prediction: A Novel Approach with Explicit Delay and Horizon (https://arxiv.org/abs/2408.04377)
- **What's New**: 본 논문은 시간 시계열 데이터의 이상 현상 예측을 위한 새로운 접근 방식을 소개하며, 예측 결과에 시간 정보를 직접 통합하여 기존의 단순 이상 현상 감지 방식의 한계를 극복합니다. 이는 이상 현상의 지연 시간(delay time)과 예측 범위(horizon)를 정확하게 예측할 수 있도록 하여 실제 상황에서 더 유용한 정보를 제공합니다. 또한, 이러한 접근 방식을 평가하기 위해 특별히 설계된 합성 데이터셋을 제시하고 최첨단 방법들을 사용하여 포괄적인 실험을 수행합니다.



### Optimal Layout-Aware CNOT Circuit Synthesis with Qubit Permutation (https://arxiv.org/abs/2408.04349)
Comments:
          9 pages, 12 tables

- **What's New**: 이 논문은 양자 회로의 노이즈 감소에 중요한 역할을 하는 CNOT 최적화에 대한 새로운 접근 방식을 제시합니다. 기존의 휴리스틱 및 정확한 방법과 달리, 이 논문은 큐비트 순열(permutation)을 허용하고 레이아웃 제한 사항을 처리하는 보다 복잡한 최적 합성 변형을 조사합니다. 이러한 문제는 계획(Planning), SAT 및 QBF로 인코딩됩니다. CNOT 게이트 수와 회로 깊이(circuit depth)를 모두 최적화합니다. 실험적 평가를 위해 표준 T-게이트 최적화 벤치마크(benchmark)를 고려하고 CNOT 서브 회로를 최적화합니다. 큐비트 순열을 허용하면 CNOT 수는 최대 56%, 회로 깊이는 최대 46%까지 추가로 줄일 수 있습니다. 레이아웃 제한 사항 하에서 최적으로 매핑된 회로의 경우, CNOT 수는 최대 17%, CNOT 깊이는 최대 19%까지 감소하는 것으로 나타났습니다.



### Towards Explainable Network Intrusion Detection using Large Language Models (https://arxiv.org/abs/2408.04342)
- **What's New**: 본 논문에서는 대규모 언어 모델(LLM)을 네트워크 침입 탐지 시스템(NIDS)으로 활용하는 가능성을 살펴봅니다. 기존 NIDS는 인공적으로 생성된 벤치마크 데이터셋에 의존하여 실제 환경에서의 성능이 저조했지만, LLM은 방대한 사전 학습 데이터를 기반으로 실제 네트워크 환경에 더 잘 적응할 수 있을 것으로 기대됩니다. 따라서 GPT-4 및 LLama3 모델을 기존 아키텍처와 트랜스포머 기반 모델과 비교하여 인공 데이터셋에 의존하지 않고 LLM의 사전 학습된 지식만으로 악성 네트워크 흐름(NetFlows)을 탐지하는 능력을 평가합니다.

- **Technical Details**: LLM의 네트워크 침입 탐지 능력을 평가하기 위해 GPT-4와 LLama3 모델을 사용하고, Zero-shot learning과 Fine-tuning을 수행합니다. 이러한 모델을 NF-UNSW-NB15-v2 및 NF-CSE-CIC-IDS2018-v2 데이터셋에서 평가하며, 악성 네트워크 흐름을 탐지하고 그 이유를 설명하는 능력을 살펴봅니다.

- **Performance Highlights**: LLM은 정확한 공격 탐지에는 어려움을 겪지만, NIDS에서 설명 가능성을 높이고 위협 대응을 지원하는 보완적인 역할을 할 수 있는 잠재력이 있습니다. 특히, Retrieval Augmented Generation(RAG) 및 함수 호출 기능과 통합될 경우, 설명 제공 및 위협 대응에 유용하게 활용될 수 있습니다.



### Learning with Digital Agents: An Analysis based on the Activity Theory (https://arxiv.org/abs/2408.04304)
Comments:
          Authors manuscript accepted for publication in Journal of Management Information Systems

- **What's New**: 본 논문에서는 교육 환경에서 사용되는 대화형 인공지능 에이전트, 즉 교육적 에이전트(pedagogical agent)와의 상호작용을 활동 이론(activity theory) 관점에서 분석하는 모델을 제시합니다. 이 모델은 학습 활동을 포괄적으로 이해하고, 교육적 에이전트의 다양한 특징과 학습자 특성이 학습 결과에 미치는 영향을 분석하는 데 도움이 됩니다.  



### Tackling Noisy Clients in Federated Learning with End-to-end Label Correction (https://arxiv.org/abs/2408.04301)
Comments:
          To appear in ACM CIKM'24 full research paper track

- **What's New**: 본 논문에서는 분산 환경에서 데이터 품질 문제 해결을 위한 새로운 연합 학습 (Federated Learning, FL) 프레임워크인 FedELC를 제안합니다. FedELC는 특히 각 클라이언트의 데이터에 존재하는 복잡한 라벨 노이즈 (label noise)를 해결하기 위한 두 단계 접근 방식을 활용합니다. 첫 번째 단계에서는 높은 노이즈 비율을 가진 클라이언트들을 식별하고, 두 번째 단계에서는 이러한 클라이언트들의 데이터 라벨을 교정하는 작업을 수행합니다. 특히, 두 번째 단계에서는 백프로퍼게이션 (back propagation)을 통해 노이즈가 있는 클라이언트 데이터의 가능한 진실 레이블 분포를 학습하는 엔드투엔드 라벨 교정 프레임워크를 사용합니다. 즉, FedELC는 노이즈가 있는 클라이언트 데이터의 품질을 개선하여 연합 학습 모델의 성능을 향상시키는 데 초점을 맞춥니다.



### Assigning Credit with Partial Reward Decoupling in Multi-Agent Proximal Policy Optimization (https://arxiv.org/abs/2408.04295)
Comments:
          20 pages, 5 figures, 12 tables, Reinforcement Learning Journal and Reinforcement Learning Conference 2024

- **What's New**: 본 논문에서는 다중 에이전트 근접 정책 최적화 (MAPPO)의 성능을 개선하는 새로운 다중 에이전트 강화 학습 알고리즘을 제안합니다. 이 알고리즘은 부분 보상 분리 (PRD) 기술을 활용하여 에이전트 간의 신용 할당 문제를 완화합니다. PRD는 학습된 주의 메커니즘을 통해 각 에이전트의 학습 업데이트에 관련된 팀원들을 추정하여, 큰 에이전트 그룹을 더 작고 관리하기 쉬운 하위 그룹으로 동적으로 분해합니다. 이는 각 에이전트가 자신에게 영향을 미치지 않는 팀원들로부터 분리됨으로써 신용 할당을 간소화하는 효과를 가져옵니다. PRD-MAPPO는 StarCraft II를 포함한 다양한 다중 에이전트 작업에서 MAPPO와 다른 최첨단 방법보다 데이터 효율성과 점근적 성능이 훨씬 뛰어납니다. 또한, PRD-MAPPO의 공유 보상 환경에 적용할 수 있는 버전을 제안하여 MAPPO보다 성능 향상을 보여줍니다.



### AI-Driven Chatbot for Intrusion Detection in Edge Networks: Enhancing Cybersecurity with Ethical User Consen (https://arxiv.org/abs/2408.04281)
- **What's New**: 본 논문은 엣지 네트워크 보안, 특히 침입 탐지 분야에서 챗봇의 잠재력을 탐구합니다. 챗봇을 활용하여 엣지 네트워크에서 침입 탐지 시스템을 구현하여 네트워크 보안을 강화하고자 합니다.



### Unveiling Hidden Visual Information: A Reconstruction Attack Against Adversarial Visual Information Hiding (https://arxiv.org/abs/2408.04261)
Comments:
          12 pages

- **What's New**: 이 논문은 암호화된 이미지에 대한 데이터 재구성(DR) 공격을 수행하여 적대적 예제 기반 이미지 암호화의 보안 취약점을 조사합니다. 대표적인 이미지 암호화 방법은 적대적 시각 정보 숨기기(AVIH)로, 이미지 인식 작업에 사용되는 갤러리 데이터 세트를 보호하기 위해 유형 I 적대적 예제 학습을 사용합니다. AVIH 방법에서 유형 I 적대적 예제 접근 방식은 완전히 다르게 보이지만 기계가 원본으로 인식하는 이미지를 만듭니다. 또한 AVIH 방법은 미리 정의된 개인 키 생성 모델을 사용하여 암호화된 이미지를 원래 형태로 복원할 수 있습니다. 최상의 보안을 위해 각 이미지에 고유한 키를 할당하는 것이 좋지만 저장 용량 제한으로 인해 일부 이미지가 동일한 키 모델을 공유해야 할 수 있습니다. 이는 AVIH에 대한 중요한 보안 문제를 제기합니다. 즉, DR 공격으로 손상되지 않고 동일한 키 모델을 안전하게 공유할 수 있는 이미지는 몇 개입니까? 이 질문에 답하기 위해 저자는 (1) 생성적 적대적 손실과 (2) 증강된 ID 손실을 통합하여 AVIH 암호화 방법에 대한 이중 전략 DR 공격을 소개합니다. 이는 DR이 과적합(overfitting)되는 것을 방지합니다. 이는 기계 학습에서 발생하는 문제와 유사합니다. 숫자 결과는 이미지 인식 및 재 식별 벤치마크를 통해 이 접근 방식을 검증하여 제안된 전략이 재구성된 이미지의 품질을 크게 향상시켜 더 적은 키 공유 암호화 이미지를 필요로 함을 보여줍니다. 결과를 재현하기 위한 소스 코드는 곧 제공될 예정입니다.



### EfficientRAG: Efficient Retriever for Multi-Hop Question Answering (https://arxiv.org/abs/2408.04259)
Comments:
          20 pages, 4 figures

- **What's New**: This paper introduces EfficientRAG, an effective retriever for multi-hop question answering that iteratively generates new queries without relying on multiple LLM calls.

- **Technical Details**: EfficientRAG consists of two components: a Labeler & Tagger and a Filter. The Labeler annotates relevant tokens in retrieved chunks, while the Tagger determines whether the chunk is helpful or irrelevant. The Filter constructs new queries for subsequent retrieval rounds by replacing unknown parts with labeled tokens. This approach avoids the need for LLM calls during query generation.

- **Performance Highlights**: Experimental results show that EfficientRAG outperforms existing RAG methods on three open-domain multi-hop question-answering datasets (HotpotQA, 2Wiki-multihop (2WikiMQA), and MuSiQue). The study also highlights the efficiency of EfficientRAG in terms of chunk retrieval compared to one-time and iterative query decomposition approaches.

- **Training Details**: EfficientRAG is trained on synthetic data generated by LLMs through multi-hop question decomposition, token labeling, next-hop question filtering, and negative sampling.

- **Data Generation Process**: The synthetic data generation involves prompting LLMs to decompose multi-hop questions into single-hop questions, label important words in chunks, generate next-hop questions, and select negative samples.

- **Architecture**: EfficientRAG utilizes an auto-encoder language model for token embedding and a fully connected layer for classification.

- **Key Contributions**: EfficientRAG introduces an efficient iterative retrieval method for multi-hop question answering by leveraging a small model for query generation, reducing reliance on LLMs and improving efficiency.



### Scalable Transformer for High Dimensional Multivariate Time Series Forecasting (https://arxiv.org/abs/2408.04245)
- **What's New**: 이 논문은 고차원 다변량 시계열(MTS) 예측을 위한 새로운 스케일 가능한 트랜스포머 모델인 STHD(Scalable Transformer for High-Dimensional Multivariate Time Series Forecasting)를 제안합니다. STHD는 기존의 채널 종속 모델이 고차원 데이터에서 저조한 성능을 보이는 문제를 해결하기 위해 고안되었습니다.

- **Technical Details**: STHD는 세 가지 핵심 구성 요소로 구성됩니다:

1. **관계 행렬 희소성(Relation Matrix Sparsity):** 관련 없는 시계열로 인한 노이즈를 제한하고 메모리 문제를 완화합니다.
2. **ReIndex:** 더 유연한 배치 크기 설정을 가능하게 하고 훈련 데이터의 다양성을 높이는 훈련 전략입니다.
3. **트랜스포머(Transformer):** 2차원 입력을 처리하고 채널 간 종속성을 포착합니다.

STHD는 DeepGraph를 사용하여 고차원 MTS 데이터에서 채널 간의 관계를 희소화함으로써 효율성을 높입니다.

- **Performance Highlights**: 실험 결과 STHD는 범죄, 위키피디아 인물, 교통과 같은 세 가지 고차원 데이터셋에서 기존 방법보다 뛰어난 성능을 보였습니다.



### The Ungrounded Alignment Problem (https://arxiv.org/abs/2408.04242)
Comments:
          7 pages, plus references and appendix

- **What's New**: 본 논문에서는 "Ungrounded Alignment Problem"이라는 새로운 문제를 제시하고, 이 문제를 해결하기 위한 방법을 제시합니다. 이 문제는 레이블 없이 특정 추상적 패턴(예: 'fnord')을 인식하는 시스템을 만드는 것과 관련이 있습니다. 즉, 시스템은 학습 중에 어떤 이미지가 어떤 레이블에 해당하는지 알지 못한 채로, 이미지 시퀀스에서 특정 패턴을 식별할 수 있어야 합니다.  이는 로봇에게 쓰레기 줍기 등 특정 행동을 학습시키는 데 필요한 핵심적인 문제입니다.



### Cluster-Wide Task Slowdown Detection in Cloud System (https://arxiv.org/abs/2408.04236)
Comments:
          This paper has been accepted by KDD2024

- **What's New**: 본 논문은 클러스터 수준의 느린 작업 감지를 위한 새로운 방법론인 SORN을 제시하며, 복잡한 주기성을 갖는 시간 시리즈에서 저진폭 하위주기를 정확하게 재구성하는 데 어려움을 겪는 표준 어텐션 메커니즘의 문제점을 분석합니다.

- **Technical Details**: SORN은 복합 주기성을 포착하기 위한 스키밍 어텐션 메커니즘, 클러스터 수준의 느린 작업을 제외한 비-느린 변동을 재구성하는 신경 최적 수송 모듈, 그리고 훈련 데이터의 이상치 오염을 완화하는 까다로운 손실 함수 등 3가지 혁신적인 메커니즘을 포함하고 있습니다. 특히 스키밍 어텐션은 진폭이 다른 하위 주기를 순차적으로 스키밍하여 원본 시리즈에서 제거하고 남은 시리즈에서 반복적으로 재구성합니다.

- **Performance Highlights**: SORN은 실제 산업 데이터 세트에 대한 광범위한 실험을 통해 최첨단 방법을 능가하는 성능을 입증했습니다. 특히 F1 점수 측면에서 우수한 성능을 보였습니다.



### Probabilistic Circuits for Cumulative Distribution Functions (https://arxiv.org/abs/2408.04229)
- **What's New**: 이 논문에서는 확률 회로(PC)가 누적 분포 함수(CDF)를 계산하도록 확장하여 기존의 확률 질량/밀도 함수(PMF/PDF)를 계산하는 방식을 보완합니다. CDF는 모든 실수 값 다변량 확률 분포에 대해 존재하고, 머신 러닝과 통계에서 폭넓게 사용되기 때문에 중요합니다. 이 논문은 이진 변수, 유한 이산 변수, 연속 변수에 대한 CDF를 계산하는 PC를 분석하고, 이러한 변수에 대해 CDF와 PMF/PDF가 서로 변환 가능함을 보여줍니다.



### VideoQA in the Era of LLMs: An Empirical Study (https://arxiv.org/abs/2408.04223)
Comments:
          Preprint. Under Review

- **What's New**: This paper provides a comprehensive analysis of Video-LLMs' (Video Large Language Models) performance in Video Question Answering (VideoQA). It investigates their strengths, weaknesses, and failure modes, offering valuable insights for developing more human-like video understanding and question-answering systems.

- **Technical Details**: The research utilizes a series of adversarial probes to test Video-LLMs in various areas, including temporal understanding, visual grounding, multimodal VQA reasoning, robustness, and generalization.  These probes were designed to adjust the original VideoQA data or settings and analyze the models' performance before and after these modifications.  For comparison, the study also includes the behavior of SOTA (State-of-the-Art) non-LLM methods that fine-tune small language models like BERT and RoBERTa.

- **Performance Highlights**: While Video-LLMs excel in standard VideoQA scenarios, showing strong correlations with contextual cues and generating plausible responses to questions about various video content, the analysis reveals significant limitations:

*   **Temporal Understanding:** Video-LLMs struggle with temporal reasoning, particularly in understanding the order of video content. They are often outperformed by non-LLM methods in this area.
*   **Visual Grounding:**  While Video-LLMs significantly outperform non-LLM methods in answering video questions, their visual grounding abilities are relatively weak. This suggests their success is largely driven by language priors and spurious vision-text correlations.
*   **Multimodal VQA Reasoning:** Video-LLMs excel at exploiting shortcuts in candidate answers for multi-choice QA, which indicates their limited capacity for faithful reasoning from video questions to correct answers.
*   **Robustness:** Video-LLMs show surprising insensitivity to video perturbations (e.g., shuffling frames) but are unexpectedly sensitive to simple variations in question wording, especially in open-ended QA.
*   **Generalization:** Video-LLMs tend to favor high-frequency answers for open-ended questions and may not generalize effectively across different question types or datasets.

- **Findings Summary**: The findings highlight the impressive capabilities of Video-LLMs in standard VideoQA tasks, but also reveal critical shortcomings in robustness, interpretability, and their reliance on spurious correlations. This underscores the urgent need for rationales in the development of future Video-LLMs to ensure more trustworthy and human-like video understanding.



### Connective Viewpoints of Signal-to-Noise Diffusion Models (https://arxiv.org/abs/2408.04221)
- **What's New**: 이 논문은 S2N(Signal-to-Noise) 확산 모델에 대한 통합적인 관점을 제공하여 다양한 관점을 연결하고 새로운 관점을 탐구합니다. 특히, 노이즈 스케줄러(noise scheduler)의 역할을 SNR(Signal-to-Noise Ratio) 관점과 정보 이론과의 연결성을 통해 분석합니다. 이 프레임워크를 기반으로, 논문은 추론 과정의 성능을 향상시키기 위한 일반화된 역방향 방정식을 개발합니다. 또한, Non-Markovian 연속 변분 확산 모델(Continuous Variational Diffusion Model)을 개발하여 전방 분포를 정확하게 유도합니다.



### Attention Mechanism and Context Modeling System for Text Mining Machine Translation (https://arxiv.org/abs/2408.04216)
- **What's New**: 본 논문은 Transformer 패러다임을 기반으로 하는 새로운 아키텍처 스키마를 제시하며, K-means 클러스터링 알고리즘을 혁신적으로 통합하여 스키마의 맥락 이해 능력을 향상시킵니다. Transformer 모델은 병렬 컴퓨팅 성능과 멀티 헤드 어텐션 메커니즘 덕분에 기계 번역 작업에서 우수한 성능을 보여줍니다. 그러나 복잡한 언어 구조를 다룰 때 맥락적 모호성을 경험하거나 지역적 특징을 무시할 수 있습니다. 이러한 제약을 해결하기 위해 본 논문은 입력 텍스트의 어휘 및 관용구를 계층화하는 데 사용되는 K-Means 알고리즘을 통합하여 언어의 지역적 구조와 맥락적 지능을 더 잘 식별하고 보존할 수 있도록 합니다. 이러한 조합의 장점은 K-Means가 텍스트에서 번역 품질과 직접 관련될 수 있는 토픽 또는 개념 영역을 자동으로 발견할 수 있다는 것입니다. 따라서 본 논문에서 고안된 스키마는 Transformer 이전에 K-Means를 준비 단계로 사용하고 멀티 헤드 어텐션 가중치를 재조정하여 유사한 의미나 기능을 가진 어휘 및 관용구를 구별하는 데 도움을 줍니다. 이는 스키마가 위치적 지능에만 집중하는 것이 아니라 학습 단계 동안 이러한 클러스터에 내재된 맥락적 지능에 더 큰 관심을 기울이도록 보장합니다.



### Pairwise Judgment Formulation for Semantic Embedding Model in Web Search (https://arxiv.org/abs/2408.04197)
- **What's New**: 이 논문은 웹 검색을 위한 의미 임베딩 모델(SEM) 훈련을 위한 효과적인 쌍방향 판단(pairwise judgment) 생성 전략을 심층적으로 조사한 최초의 연구입니다. SEM은 쌍방향 아키텍처 기반의 신경망으로, 정보 검색 및 자연어 처리 분야에서 주목을 받고 있습니다. 이 논문에서는 기존 쌍방향 학습 순위(LTR) 분야에서 널리 사용되는 쌍방향 판단 공식화 전략이 SEM 훈련에 항상 효과적인 것은 아니라는 사실을 발견했습니다. 이 연구는 대규모 상업 검색 엔진의 쿼리 로그와 클릭 활동을 기반으로 한 광범위한 실험을 통해 SEM에 효과적인 전략을 보여주었으며, LTR의 원자적 휴리스틱(예: Clicked > Skipped)과 비교하여 하이브리드 휴리스틱(예: Clicked > Non-Clicked)의 장점을 강조했습니다.

- **Technical Details**: 이 연구에서는 SEM 훈련을 위한 쌍방향 판단 공식화를 위한 다양한 전략을 조사했습니다. 연구 결과, 기존 LTR에서 널리 사용되는 쌍방향 판단 전략은 SEM 훈련에 항상 효과적이지 않음을 밝혀냈습니다. 특히, LTR에서 거의 사용되지 않는 Clicked>Non-Examined 전략이 SEM에 가장 효과적임을 발견했습니다. 또한 Clicked>Non-Clicked 전략이 실질적인 장점을 가지고 있음을 보여주었습니다. 이 연구는 SEM의 아키텍처와 훈련 과정을 자세히 설명하고 있으며, 특히 힌지 손실(hinge loss) 함수를 이용한 쌍방향 학습 방식을 강조합니다.

- **Performance Highlights**: 이 연구에서는 대규모 상업 검색 엔진의 쿼리 로그를 사용하여 실험을 수행했으며, 다양한 쌍방향 판단 전략의 성능을 비교 분석했습니다. 그 결과, LTR에서 일반적으로 사용되는 전략보다 Clicked>Non-Examined 전략이 SEM 훈련에 더 효과적임을 확인했습니다. 또한 Clicked>Non-Clicked 전략이 SEM 훈련에 실질적인 장점을 제공한다는 점을 확인했습니다. 이러한 결과는 의미 임베딩 모델 훈련을 위한 효과적인 쌍방향 판단 공식화 전략을 찾는 것이 중요하며, 더 나아가 실무에서 의미 임베딩 모델의 성능을 향상시키기 위한 방향을 제시합니다.



### Uncertainty-Aware Crime Prediction With Spatial Temporal Multivariate Graph Neural Networks (https://arxiv.org/abs/2408.04193)
- **What's New**: 본 논문은 도시 범죄 예측을 위한 새로운 프레임워크인 STMGNN-ZINB를 제안합니다. STMGNN-ZINB는 Zero-Inflated Negative Binomial (ZINB) 분포를 활용하여 범죄 데이터의 희소성 문제를 해결하고 확률적 범죄 예측을 수행합니다. 이는 기존 범죄 예측 모델들이 가우시안 분포를 가정하여 범죄 데이터의 희소성과 과분산 패턴을 제대로 다루지 못했던 문제점을 해결합니다.



### Listwise Reward Estimation for Offline Preference-based Reinforcement Learning (https://arxiv.org/abs/2408.04190)
Comments:
          21 pages, ICML 2024

- **What's New**: 이 논문은 오프라인 기반의 선호도 기반 강화 학습(PbRL)을 위한 새로운 방법인 Listwise Reward Estimation (LiRE)를 제안합니다. LiRE는 기존의 삼항 선호도 피드백을 사용하면서도, 궤적의 순위 목록(RLT)을 구축하여 2차 선호도 정보를 활용합니다. 이를 통해 더 정확한 보상 함수를 추정하고 기존의 오프라인 PbRL 방법보다 뛰어난 성능을 보여줍니다.



### EdgeShield: A Universal and Efficient Edge Computing Framework for Robust AI (https://arxiv.org/abs/2408.04181)
- **What's New**: This paper proposes an edge computing framework for detecting adversarial patch attacks on AI systems. The framework uses a lightweight, attention-based detection model deployed on edge devices, allowing real-time detection and preventing contaminated data from being sent to the cloud for further processing.

- **Technical Details**: The detection model utilizes shallow layers from a conventional neural network (e.g., VGG-16). It employs an attention-based methodology, generating an attention map from the activation map of a specific layer. The attention map is calculated using a simple formula that identifies abnormal regions caused by adversarial patches.

- **Performance Highlights**: The proposed framework achieved an impressive 97.43% F-score in detecting adversarial attacks across five different neural networks. Compared to previous methods, it offers significantly reduced computational complexity and cost, making it suitable for real-time deployment on edge devices.



### wav2graph: A Framework for Supervised Learning Knowledge Graph from Speech (https://arxiv.org/abs/2408.04174)
Comments:
          Preprint, 32 pages

- **What's New**: wav2graph는 음성 데이터에서 지식 그래프(KG)를 학습하기 위한 최초의 프레임워크로, 지식 추출을 자동화하고 음성 기반 정보를 다양한 AI 애플리케이션에 통합합니다. 이 프레임워크는 음성 인식(ASR) 전사본을 사용하여 KG를 구축하고, 최첨단 그래프 신경망(GNN)을 통해 노드 분류 및 링크 예측을 수행합니다.



### The Data Addition Dilemma (https://arxiv.org/abs/2408.04154)
Comments:
          Machine Learning For Health Care 2024 (MLHC)

- **What's New**: This paper identifies the "Data Addition Dilemma" in healthcare machine learning, where adding more data from diverse sources can sometimes worsen model performance, fairness, and worst-subgroup performance. This arises from a trade-off between model performance improvements due to data scaling and model deterioration due to distribution shift (data coming from different sources).



### UNLEARN Efficient Removal of Knowledge in Large Language Models (https://arxiv.org/abs/2408.04140)
Comments:
          11 pages, 2 Figures

- **What's New**: This paper introduces **UNLEARN**, a novel algorithm for removing knowledge from Large Language Models (LLMs) without affecting related knowledge. UNLEARN leverages subspace methods to identify and target specific knowledge for removal.

- **Technical Details**: UNLEARN works in three main steps: 1. **Subspace Identification**: training a task-dependent matrix for each layer to capture the subspace of that specific knowledge. 2. **Subspace Discrimination**: using a Gram-Schmidt process to orthogonalize subspaces and prevent information loss in similar tasks. 3. **Task Removal**: subtracting the modified task matrix to eliminate the targeted knowledge.

- **Performance Highlights**: UNLEARN achieves **96% forgetting** on the target task while maintaining performance on dissimilar tasks within 2.5% of the original model. It also demonstrates **80% forgetting** on similar tasks while preserving performance within 10%. The paper also introduces **LEARN**, a dual method for targeted knowledge addition that achieves similar performance to fine-tuning techniques like LoRA.

- **Limitations**: UNLEARN's effectiveness might be limited by the presence of highly overlapping tasks, where separating subspaces can be challenging.

- **Future Directions**: The paper highlights potential for future research in areas such as: 1.  Improving the efficiency of subspace identification and discrimination. 2. Investigating the impact of UNLEARN on various LLM architectures and tasks. 3. Exploring applications of UNLEARN for privacy-preserving LLM deployment.



### Enhancing Healthcare through Large Language Models: A Study on Medical Question Answering (https://arxiv.org/abs/2408.04138)
Comments:
          received by IEEE ICPICS

- **What's New**: 본 논문은 MedQuAD 의료 질문 답변 데이터 세트에 대한 다양한 LLM (Large Language Model)을 훈련하고 가장 효과적인 모델을 식별하는 연구를 소개합니다. 이 연구는 Sentence-t5와 Mistral 7B의 조합이 0.762의 정밀도 점수를 달성하여 우수한 성능을 보였음을 밝혀냈습니다.

- **Technical Details**: 본 논문은 Gemma 2b + LoRA, Phi-2, Sentence-t5 + Mistral 7B를 포함한 세 가지 모델의 훈련과 미세 조정(fine-tuning)을 자세히 설명합니다. 모델은 데이터 전처리, 프롬프트 생성, 미세 조정 등 엄격한 프로세스를 거쳤습니다.

- **Performance Highlights**: Sentence-t5 + Mistral 7B 모델은 최고의 정밀도를 달성하여 환자 교육 및 지원을 향상시키는 데 유망한 후보임을 입증했습니다. 이 모델은 고급 사전 훈련 기술, 강력한 아키텍처, 효과적인 프롬프트 생성 방법론을 통해 향상된 기능을 제공합니다.



### Can Rule-Based Insights Enhance LLMs for Radiology Report Classification? Introducing the RadPrompt Methodology (https://arxiv.org/abs/2408.04121)
Comments:
          Accepted at BioNLP, ACL 2024

- **What's New**: 이 연구는 **RadPert** (룰 기반 시스템)과 **RadPrompt** (멀티턴 프롬프팅 전략)을 소개하며, 이를 통해 의료 이미지 분석에서 **원격 감독**(distant supervision)을 개선하고 **대규모 언어 모델**(LLM)의 성능을 향상시키는 새로운 방법을 제시합니다.

- **Technical Details**: RadPert는 **RadGraph 지식 그래프**와 함께 **불확실성 인식 정보 스키마**를 통합하여 룰 기반 시스템의 견고성을 높였습니다. RadPrompt는 RadPert를 활용하여 **LLM의 제로 샷 예측 능력**을 강화하는 **멀티턴 프롬프팅 전략**입니다.

- **Performance Highlights**: RadPert는 기존 룰 기반 SOTA인 **CheXpert**를 능가하는 성능을 보였습니다. 또한, RadPrompt는 **GPT-4 Turbo**와 **RadPert** 모두를 뛰어넘는 성능을 달성하여 LLM과 룰 기반 모델의 상호작용 가능성을 입증했습니다.



### Patchview: LLM-Powered Worldbuilding with Generative Dust and Magnet Visualization (https://arxiv.org/abs/2408.04112)
Comments:
          Accepted to UIST2024

- **What's New**: 이 논문은 Patchview를 소개합니다. Patchview는 사용자가 스토리 개념과 요소들을 자석과 먼지의 물리적 은유를 통해 상호작용할 수 있도록 하여 세계 구축을 시각적으로 돕는 맞춤형 LLM 기반 시스템입니다. Patchview에서 요소는 시각적으로 드래그되어 관련성이 높은 개념에 가까워지므로 이해를 돕습니다. 사용자는 또한 언어적으로 모호한 개념으로 생성을 조종할 수 있습니다. 사용자가 LLM의 시각화 및 생성에 동의하지 않으면 요소를 재배치하여 수정할 수 있습니다. 이러한 수정은 LLM의 미래 행동을 사용자의 인식에 맞추는 데 사용할 수 있습니다.



### Hardware-Assisted Virtualization of Neural Processing Units for Cloud Platforms (https://arxiv.org/abs/2408.04104)
Comments:
          Accepted to MICRO'24

- **What's New**: TCloud는 클라우드 플랫폼에서 NPU (Neural Processing Units)의 가상화를 가능하게 하는 새로운 하드웨어 지원 시스템 가상화 프레임워크입니다. TCloud는 다중 테넌트 ML 서비스를 위한 효율적인 자원 공유를 위해 NPU를 가상화하여 리소스 활용도를 극대화하고 적절한 서비스 품질을 유지합니다. TCloud는 멀티 테넌트 ML 서비스의 다양한 리소스 요구 사항을 충족시키기 위해 vNPU, 리소스 할당기, ISA 확장을 포함한 3가지 주요 구성 요소를 포함합니다.



### ArtVLM: Attribute Recognition Through Vision-Based Prefix Language Modeling (https://arxiv.org/abs/2408.04102)
Comments:
          Accepted at ECCV 2024

- **What's New**: 이 논문은 대규모 이미지-텍스트 기반 모델을 사용하여 시각적 속성 인식 문제를 해결하는 새로운 접근 방식을 제시합니다. 이는 '생성적 검색'(generative retrieval)이라는 새로운 개념을 도입하여 기존의 '대조 검색'(contrastive retrieval)의 단점을 극복합니다. 생성적 검색은 시각적 속성과 객체의 관계를 조건부 확률 그래프로 모델링하고, 이를 바탕으로 시각적 속성을 인식하는 문제를 언어 모델링 문제로 변환합니다. 이를 통해 대규모 사전 훈련된 비전-언어 모델(VLM)의 지식을 효과적으로 활용하여 이미지에서 시각적 속성을 인식하는 성능을 향상시킵니다. 특히, 이미지에서 인식해야 할 각 속성에 대해, 이미지 내 객체와 속성의 관계를 나타내는 짧은 문장을 생성할 확률을 계산합니다. 이러한 접근 방식은 문장 내 객체와 속성의 순서와 의존성을 고려하여 대조 검색과 비교하여 더욱 정확한 결과를 제공합니다.



### AEye: A Visualization Tool for Image Datasets (https://arxiv.org/abs/2408.04072)
Comments:
          Accepted at IEEE VIS 2024

- **What's New**: AEye는 이미지 데이터셋을 시각화하고 탐색하기 위한 새로운 도구입니다. AEye는 대규모 이미지 데이터셋을 이해하기 쉽게 구성하고 시각화하기 위해 CLIP(Contrastive Language-Image Pretraining) 임베딩 공간을 활용합니다. 이미지는 CLIP 임베딩을 기반으로 2차원 평면에 배치되어 사용자는 직관적으로 탐색할 수 있습니다. AEye는 또한 이미지 및 텍스트 검색 기능을 제공하여 데이터셋을 탐색할 수 있는 기능을 제공합니다.



### PowerPM: Foundation Model for Power Systems (https://arxiv.org/abs/2408.04057)
Comments:
          23 pages, 5 figures, 8 tables

- **What's New**: PowerPM, a foundation model for power systems, is proposed to address the challenge of modeling Electricity Time Series (ETS) data with complex hierarchical structures, temporal dependencies, and diverse consumption patterns. PowerPM leverages a novel self-supervised pre-training framework combining masked ETS modeling and dual-view contrastive learning, enhancing its ability to capture temporal dependencies within ETS windows and discrepancies across them.

- **Technical Details**: PowerPM comprises a temporal encoder and a hierarchical encoder. The temporal encoder utilizes Transformer encoders to capture temporal dependencies, incorporating exogenous variables for robustness. The hierarchical encoder employs R-GCN to model the correlation between hierarchy levels, integrating micro and macro information for effective ETS modeling.

- **Performance Highlights**: PowerPM achieves state-of-the-art (SOTA) performance on diverse downstream tasks within a private dataset after pre-training on massive ETS data. It demonstrates strong generalization ability across various tasks and domains when transferred to public datasets. Ablation studies and few-shot experiments further validate the model's effectiveness.

- **Deployment**: PowerPM has been successfully deployed in Zhejiang Power Grid, generating substantial economic benefits.

- **Advantages**: PowerPM is the first model to simultaneously consider temporal dependency and hierarchical dependency in ETS data. It leverages a novel self-supervised pre-training framework, enabling the model to learn robust and generic representations.

- **Applications**: PowerPM is designed for various applications in power systems, including demand-side management, grid stability, and consumer behavior analysis.

- **Keywords**: Electricity Time Series (ETS), Foundation Model, PowerPM, Self-Supervised Learning, Masked ETS Modeling, Dual-View Contrastive Learning, Temporal Dependency, Hierarchical Dependency, Transformer, R-GCN, Pre-training, Fine-tuning, Downstream Tasks, Generalization Ability



### Machine Learning-Based Reward-Driven Tuning of Scanning Probe Microscopy: Towards Fully Automated Microscopy (https://arxiv.org/abs/2408.04055)
Comments:
          20 pages, 6 figures

- **What's New**: 이 논문은 탭핑 모드(tapping mode) 스캐닝 프로브 현미경(SPM)의 자동화된 최적화를 위한 보상 기반 워크플로우를 소개합니다. 이 워크플로우는 이미지 품질을 측정하고 인간 운영자가 사용하는 의사 결정 논리를 모방하여 다양한 채널(channels)과 물리적 및 경험적 지식을 사용하여 보상 함수(reward function)를 정의합니다.



### Learning Rate-Free Reinforcement Learning: A Case for Model Selection with Non-Stationary Objectives (https://arxiv.org/abs/2408.04046)
Comments:
          RLC 2024 Workshop on Failure Modes of Sequential Decision-Making in Practice

- **What's New**: 이 논문은 강화 학습(RL) 알고리즘에서 최적의 학습률(learning rate)을 찾는 데 모델 선택(model selection) 방법을 활용하는 새로운 프레임워크를 제시합니다. 이 프레임워크는 RL 알고리즘이나 최적화 방법과 독립적으로 동작하며, 보상 피드백(reward feedback)을 기반으로 학습률을 조정하여 '학습률 없는 강화 학습'을 구현합니다.



### Multimodal Gender Fairness in Depression Prediction: Insights on Data from the USA & China (https://arxiv.org/abs/2408.04026)
Comments:
          9 Pages, 7 Tables. To be published and indexed in the IEEE Xplore Digital Library under the ACII 2024 Workshop Proceedings

- **What's New**: 본 논문은 문화적 차이와 성별 간의 우울증 발현에 대한 멀티모달 기계 학습(ML) 모델의 공정성을 평가하는 최초의 연구입니다. 이 연구는 미국과 중국의 두 개의 데이터 세트를 사용하여 수행되었습니다.



### Improving Large Language Model (LLM) fidelity through context-aware grounding: A systematic approach to reliability and veracity (https://arxiv.org/abs/2408.04023)
Comments:
          14 pages

- **What's New**: 본 논문에서는 자연어 처리(NLP) 애플리케이션에서 점점 더 정교하고 널리 사용되는 대규모 언어 모델(LLM)의 견고성, 신뢰성 및 인간 가치와의 정렬을 보장하는 데 초점을 맞춘 새로운 텍스트 모델의 문맥 기반 접근 방식을 제시합니다. 이 접근 방식은 특히 문맥 표현 단계에 중점을 두어 포괄적인 문맥 인식 방법론을 통해 모델의 신뢰성과 윤리적 정렬을 향상시키는 것을 목표로 합니다. 상황, 문화 및 윤리적 문맥을 기계 판독 가능한 형식으로 명시적으로 캡처하고 표현함으로써 모델의 동작을 이러한 문맥 내에서 고정하는 기반을 마련합니다. 이 접근 방식은 온톨로지, 의미 웹 기술 및 논리 기반 공식과 같은 지식 표현 및 추론 기술을 활용합니다. 연구자들은 실제 텍스트 데이터 세트에서 프레임워크를 평가하여 높은 정확성을 유지하면서 모델 성능, 공정성 및 인간의 기대와의 정렬을 향상시키는 데 효과적임을 입증했습니다. 또한, 연구자들은 문맥 인식 인코딩, 문맥 인식 학습, 해석 가능성 및 설명 가능성, 지속적인 모니터링 및 적응을 포함한 프레임워크의 다른 핵심 구성 요소에 대해 논의합니다. 이 연구는 책임감 있는 AI에 대한 점점 더 많은 작업에 기여하여 더 신뢰할 수 있고, 신뢰할 수 있으며, 윤리적으로 정렬된 언어 모델을 개발하는 실용적인 접근 방식을 제공합니다. 연구 결과는 문맥 이해가 가장 중요한 의료, 법률 시스템 및 사회 서비스와 같은 민감한 분야에서 LLM의 배포에 중요한 의미를 갖습니다.



### Learning from Noisy Labels for Long-tailed Data via Optimal Transpor (https://arxiv.org/abs/2408.03977)
- **What's New**: 이 논문은 실제 데이터셋에서 흔히 발생하는 긴 꼬리 분포(long-tailed distribution)와 잡음 레이블(noisy labels)을 동시에 다루는 새로운 방법을 제시합니다. 이 방법은 잡음 레이블과 긴 꼬리 분포로 인한 불확실성을 완화하기 위해 손실-거리 교차 선택 모듈(loss-distance cross-selection module)을 도입합니다. 또한, 긴 꼬리 분포로 인한 샘플 부족 문제를 완화하고 의사 레이블(pseudo-labels)의 품질을 향상시키기 위해 최적 수송(optimal transport) 전략을 사용하여 의사 레이블을 생성합니다.



### Enhancing Output Diversity Improves Conjugate Gradient-based Adversarial Attacks (https://arxiv.org/abs/2408.03972)
Comments:
          ICPRAI2024

- **What's New**: 본 논문에서는 'Rescaling-ACG(ReACG)'라는 새로운 적대적 공격 방법을 제안합니다. ReACG는 기존의 Auto Conjugate Gradient (ACG) 공격 방식을 개선하여 두 연속적인 탐색 지점 간 거리를 늘려 모델 출력의 다양성을 증가시킵니다. 이를 통해 더욱 강력한 적대적 예제를 생성하고,  딥러닝 모델의 취약성을 더 효과적으로 평가할 수 있습니다.

- **Technical Details**: ReACG는 ACG의 탐색 방향(search direction)과 단계 크기(step size) 제어를 자동으로 조정하여 두 연속적인 탐색 지점 간 거리를 늘리는 방식으로 작동합니다. 특히, ReACG는 그래디언트와 켤레 그래디언트(conjugate gradient)의 비율을 기반으로 계수 β(k)superscript𝛽𝑘eta^{(k)}italic_β start_POSTSUPERSCRIPT ( italic_k ) end_POSTSUPERSCRIPT 를 조정하고, Optuna를 사용한 다목적 최적화(multi-objective optimization)를 통해 단계 크기를 제어합니다.

- **Performance Highlights**: ReACG는 CIFAR-10, CIFAR-100, ImageNet 등 다양한 데이터셋에서 훈련된 30개의 강력한 모델에 대한 실험 결과, 기존의 SOTA 공격 방식인 APGD와 ACG보다 더 높은 공격 성능을 보였습니다. 특히 ImageNet 모델에 대해서는 모든 모델에서 가장 높은 공격 성능을 보였습니다.  ReACG의 성능 향상은 두 연속적인 탐색 지점 간 거리 증가와 CTC 다양성 증가로 이어지는 것으로 분석되었습니다. 이는 적대적 공격 분야에서 새로운 가능성을 제시하는 결과입니다.



### Telecom Foundation Models: Applications, Challenges, and Future Trends (https://arxiv.org/abs/2408.03964)
- **What's New**: 이 논문은 통신 네트워크의 복잡성 증가에 대응하여 기존 AI 모델의 한계를 극복하고 다양한 통신 시나리오와 애플리케이션을 지원하기 위해 **Foundation Models (FMs)** 를 통신 네트워크에 적용하는 가능성을 제시합니다. 특히, **Telecom FMs (TFMs)** 개발을 위한 개념적 프로세스를 제시하고 네트워크 구성, 운영 및 유지 보수를 위한 특수 TFM 오케스트레이션의 새로운 기회를 논의합니다. 또한 TFM 개발 및 배포와 관련된 제한 사항과 과제를 분석합니다.



### A self-adaptive system of systems architecture to enable its ad-hoc scalability: Unmanned Vehicle Fleet -- Mission Control Center Case study (https://arxiv.org/abs/2408.03963)
Comments:
          2023 7th International Conference on Intelligent Systems, Metaheuristics & Swarm Intelligence (ISMSI 2023)

- **What's New**: 본 논문은 시스템 오브 시스템(SoS)의 **확장성(scalability)** 문제에 초점을 맞추어, 특히 무인 차량 함대(UVF)를 실제 예시로 다룹니다. 특히, **임무 변화(mission changes)**, **범위 확장(range extensions)**, **UV 고장(UV failures)**과 같은 불확실성을 고려하여 시스템의 **자체 적응성(self-adaptiveness)**을 강조합니다. 이를 통해 임무 제어 센터(MCC)는 성능 기준에 따라 UVF의 규모를 자동으로 조정하거나 운영자가 수동으로 조정할 수 있습니다.



### EcoFollower: An Environment-Friendly Car Following Model Considering Fuel Consumption (https://arxiv.org/abs/2408.03950)
- **What's New**: 본 연구에서는 차량 추종 시나리오에서 연료 소비를 최적화하기 위해 강화 학습(RL)을 사용하여 개발된 새로운 친환경 차량 추종 모델인 EcoFollower를 소개합니다. EcoFollower는 차량 추종 시나리오에서 연료 소비를 최적화하기 위해 강화 학습(RL)을 사용하여 개발된 새로운 친환경 차량 추종 모델입니다.



### A Survey of AI Relianc (https://arxiv.org/abs/2408.03948)
- **What's New**: 본 논문은 AI 시스템에 대한 인간의 의존도(AI reliance)에 대한 연구의 현황과 미래 방향을 제시합니다. 특히, AI 조언에 대한 인간의 의존도에 영향을 미치는 요인, 외부 타당성, 의존도 측정 방법, 시간에 따른 의존도 변화 등에 대한 연구의 부족함을 지적합니다. 또한, 생성형 AI (generative AI) 출력에 대한 의존도와 다중 사용자 상황에서의 의존도 연구를 미래 연구 방향으로 제시합니다. 



### Prompting for products: Investigating design space exploration strategies for text-to-image generative models (https://arxiv.org/abs/2408.03946)
Comments:
          12 pages, 7 figures

- **What's New**: 이 연구는 텍스트-이미지 생성 AI 모델을 사용하여 제품 디자인 공간 탐색을 위한 전략을 실험적으로 조사합니다. 특히, 사용자의 편집 방식(전역 및 로컬)과 관련된 다양한 요소(시간 소요, 프롬프트 길이, 단일 vs. 다중 기준 프롬프트, 프롬프트의 목표 지향성)가 제품 디자인 목표(실현 가능성, 참신성, 미학)에 미치는 영향을 분석합니다.



### Impacts of Anthropomorphizing Large Language Models in Learning Environments (https://arxiv.org/abs/2408.03945)
Comments:
          Presented at Affective Computing Pre-Conference at ISRE 2024

- **What's New**: 본 연구는 교육 환경에서 사용되는 대규모 언어 모델(LLM)의 의인화(anthropomorphism)가 학습자에게 미치는 영향, 특히 학습 결과에 대한 감정적 영향을 조사합니다. LLM 기반 챗봇이 학습 도구로 점점 더 많이 사용되면서, LLM 기반 챗봇에 대한 의인화가 학습자의 감정에 미치는 영향을 이해하는 것이 중요합니다.



### Building Machines that Learn and Think with Peop (https://arxiv.org/abs/2408.03943)
- **What's New**: This paper proposes a new vision for designing AI thought partners, focusing on collaborative cognition, where humans and AI work together as partners rather than just tools for thought. It challenges the traditional approach of scaling foundation models and suggests a new path based on explicit models of the task, the world, and the human, drawing insights from cognitive psychology and behavioral sciences.

- **Technical Details**: The paper highlights the importance of building explicit models of the task, world, and human within AI systems. This contrasts with current approaches that primarily rely on large language models (LLMs) trained on massive datasets, often without explicit reasoning about the world or human cognition. It emphasizes the role of probabilistic programming, goal-directed search, and structured representations for building AI systems that can reason about other agents, including humans.

- **Performance Highlights**: The paper explores several domains where AI thought partners could be particularly valuable, including programming, embodied assistance, storytelling, and medicine. It highlights the challenges of building such partners, including the need to understand human intentions, beliefs, and limitations, as well as the need for the AI system to be transparent and reliable. It argues that effective thought partners must build models of the human and the world to collaborate effectively.



### A Comparative Visual Analytics Framework for Evaluating Evolutionary Processes in Multi-objective Optimization (https://arxiv.org/abs/2308.05640)
Comments:
          Accepted by IEEE VIS 2023 (will appear in IEEE TVCG)

- **What's New**: 본 논문은 여러 다중 목표 최적화 알고리즘(EMO)의 비교 분석을 위해 시각 분석 프레임워크를 제시합니다. EMO 알고리즘은 일반적으로 블랙 박스로 취급되어 내부 진화 과정에 대한 상세 분석과 비교가 어렵습니다. 이 프레임워크는 다양한 분석 작업을 다루고 진화의 중간 세대와 해답 집합의 비교 분석을 지원하기 위해 다면적인 시각화 설계를 구축하여 이 문제를 해결합니다.




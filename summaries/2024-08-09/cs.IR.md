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




### Timeline-based Sentence Decomposition with In-Context Learning for Temporal Fact Extraction (https://arxiv.org/abs/2405.10288)
Comments:
          Accepted to ACL2024 main conference

- **What's New**: 이 논문에서는 자연어 텍스트로부터 시간적 사실을 추출하는 새로운 방법을 제안하고 있습니다. 특히, 문장에서 시간과 사실 간의 대응 관계를 설정하는 도전 과제에 초점을 맞추고 있으며, 이를 해결하기 위해 대형 언어 모델(LLM)의 상황 학습 능력을 적용한 타임라인 기반 문장 분해 전략을 제안합니다.

- **Technical Details**: 새로운 방식인 TSDRE(Time-based Sentence Decomposition using LLMs for Relation Extraction)는 LLM의 분해 능력을 전통적인 작은 사전 학습 언어 모델(PLM)로 미세 조정하는 방식과 결합합니다. 또한, 복합 시간 표현이 포함된 19,148개의 문장으로 구성된 ComplexTRED 데이터셋을 구축하여 평가를 수행했습니다.

- **Performance Highlights**: TSDRE 방법은 최신 HyperRED-Temporal 및 ComplexTRED 데이터셋에서 최첨단 결과를 달성했습니다. 이 방법은 복잡한 시간 관련 서술이 포함된 문장에서 시간적 사실을 효과적으로 추출하며, 전통적인 방법을 능가하는 성능을 보였습니다.



### Revisiting OPRO: The Limitations of Small-Scale LLMs as Optimizers (https://arxiv.org/abs/2405.10276)
- **What's New**: 이번 연구는 비교적 소규모 LLM(Large Language Models)인 LLaMa-2 및 Mistral 7B 모델을 대상으로 OPRO(Optimization by PROmpting) 접근법의 효과를 재평가합니다. 연구 결과, 소규모 LLM에서는 OPRO의 최적화 능력이 제한적이라는 것을 확인하였으며, 이는 모델의 추론 능력에 한계가 있기 때문입니다.

- **Technical Details**: OPRO는 LLM을 최적화 도구로 활용하여 지시문(instruction)을 최적화하려는 접근법입니다. 본 연구에서는 소규모 LLM인 LLaMa-2-7B, LLaMa-2-13B, LLaMa-2-70B 및 Mistral 7B를 대상으로 실험을 진행하였고, 모델의 최적화 능력을 평가했습니다. 실험은 주로 GSM8K 데이터셋(초등학교 수학 문제)로 수행되었습니다.

- **Performance Highlights**: 실험 결과, 대규모 LLM인 Gemini-Pro 모델은 기존 결과와 일치하게 OPRO의 효용성을 입증하였지만, 소규모 LLM인 LLaMa-2 및 Mistral 7B 모델은 Zero-shot-CoT 및 Few-shot-CoT 기준보다 성능이 떨어졌습니다. 특히 소규모 LLM에서는 명확한 지시문을 제공하는 것이 최적화된 지시문을 생성하는 데 더 효과적이었습니다.



### Keep It Private: Unsupervised Privatization of Online Tex (https://arxiv.org/abs/2405.10260)
Comments:
          17 pages, 6 figures

- **What's New**: 이번 연구에서는 자동 텍스트 프라이버시 보호 프레임워크를 도입하여, 강화학습(reinforcement learning)을 통해 대형 언어 모델을 미세 조정(fine-tune)하여 저자의 신원을 감추면서도 내용의 의미와 자연스러움을 유지하는 리라이트(rewrite)를 생성합니다. 특히, 68,000명의 사용자가 작성한 Reddit 포스트를 대상으로 대규모 테스트를 수행하여 성능을 평가하였습니다.

- **Technical Details**: 우리의 자동 텍스트 프라이버시 보호 모델은 대형 언어 모델을 활용하며, 강화학습을 통해 저자 식별을 피하면서도 원본 텍스트의 의미를 보존하는 방향으로 텍스트를 다시 작성합니다. 이 과정에서 자동화된 메트릭과 인간 평가 양쪽 모두에서 높은 텍스트 품질을 유지하며, 다양한 저자 식별 공격을 성공적으로 회피합니다.

- **Performance Highlights**: 평가 결과, 제안된 방법은 여러 자동 저자 식별 모델에 대하여 높은 정확도를 유지하면서도 저자를 효과적으로 숨기는 데 성공하였습니다. 특히, 자동화된 메트릭과 인간 평가 모두에서 높은 품질의 텍스트를 생산하며, 저자 식별 모델을 가장 잘 속이는 것으로 나타났습니다.



### A Systematic Evaluation of Large Language Models for Natural Language Generation Tasks (https://arxiv.org/abs/2405.10251)
Comments:
          CCL2023

- **What's New**: 이 논문은 대형 언어 모델(LLM)의 자연어 생성(NLG) 작업 성능을 종합적으로 평가합니다. 기존 연구에서는 상식 추론, 수학적 추론, 코드 생성 등 다양한 영역에서 LLM을 평가해 왔으나, NLG 작업에 대한 성능을 구체적으로 조사한 연구는 드뭅니다. 이 논문에서는 ChatGPT, ChatGLM, T5 기반 모델, LLaMA 기반 모델, Pythia 기반 모델 등을 대상으로 영어와 중국어 데이터셋을 활용해 대화 생성(Dialogue Generation)과 텍스트 요약(Text Summarization) 작업에 대해 평가합니다.

- **Technical Details**: 자연어 생성 작업은 특정 의사소통 목표를 달성하기 위해 자연어 텍스트를 생성하는 과정입니다. 이 논문은 텍스트-텍스트 형식의 NLG 스타일에 중점을 두고 있습니다. NLG 작업은 시퀀스 y를 최적화하는 것을 목표로 하며, 다음 토큰 y_t의 조건부 확률을 기반으로 합니다. 이를 평가하기 위해 DailyDialog, PersonaChat, EmpatheticDialogue, LCCC 등의 대화 생성 데이터셋과, CNN/DailyMail 등의 텍스트 요약 데이터셋을 사용하였습니다.

- **Performance Highlights**: 논문은 다양한 아키텍처와 규모의 LLM을 비교 분석하여, 자동화된 결과와 상세한 분석을 제공하고 있습니다. 예를 들어, LLaMA-13B 모델은 GPT-3보다 작은 규모에도 불구하고 대부분의 벤치마크에서 뛰어난 성능을 보였습니다. 이러한 결과는 모델의 크기뿐만 아니라 훈련 데이터의 질과 양이 성능에 중요한 역할을 한다는 점을 시사합니다.



### CPsyExam: A Chinese Benchmark for Evaluating Psychology using Examinations (https://arxiv.org/abs/2405.10212)
- **What's New**: 이번 논문에서는 새로운 심리학 평가 벤치마크인 CPsyExam을 소개합니다. CPsyExam은 중국어 시험 문제들로 구성되었으며, 심리학 지식과 사례 분석을 별도로 우선시하여 실제 시나리오에 심리학 지식을 적용할 수 있는 능력을 강조합니다. 22,000개 이상의 문제들에서 4,000개의 문제를 선별하여 다양한 사례 분석 기법을 포함하는 균형 잡힌 데이터셋을 구성했습니다.

- **Technical Details**: CPsyExam은 세 가지 종류의 질문들로 구성됩니다: 다지선다형 문제(Multiple-choice Question Answering, MCQA), 다중 응답형 문제(Multiple-response Question Answering, MRQA), 그리고 일반 질문 응답(QA)입니다. 심리학 분야의 특수성을 고려하여 CPsyExam은 지식(KG)과 사례 분석(CA) 두 부분으로 나누어집니다. KG 부분은 전문가 시험에서 가져온 심리학 개념을 폭넓게 다루는 질문들로 구성되며, CA 부분은 상담 중 필요한 방법론, 진단 및 치료에 중점을 둔 사례 지향적 질문들로 구성됩니다.

- **Performance Highlights**: 최근의 일반 도메인 언어 모델(LLM)과 심리학 특화 LLM을 CPsyExam을 통해 비교한 결과, 기본 모델에 비해 정교하게 조정된 모델들이 심리학 지식을 이해하는 면에서 미미한 향상 또는 향상이 없음을 보였습니다. 일부 경우에서는 사례를 분석하는 능력이 오히려 저하되기도 했습니다. 이것은 LLM이 심리학 지식을 마스터하고 이를 사례 분석에 적용하는 데 아직 개선의 여지가 많음을 시사합니다. CPsyExam은 LLM의 심리학 이해를 발전시키는 데 중요한 벤치마크로서 기능할 것입니다.



### Hierarchical Attention Graph for Scientific Document Summarization in Global and Local Lev (https://arxiv.org/abs/2405.10202)
Comments:
          Accepted to NAACL 2024 Findings

- **What's New**: 과학 문서 요약의 도전 과제를 해결하기 위해 HAESum이 제안되었습니다. HAESum은 과학 문서의 계층적 담화 구조를 기반으로 한 그래프 신경망을 활용하여 문서를 로컬 및 글로벌 방식으로 모델링합니다. 이 접근법은 문장 내 관계를 학습하기 위해 로컬 이종 그래프를 사용하며, 고차원의 문장 간 관계를 더욱 향상시키기 위해 새로운 하이퍼그래프 셀프 어텐션(hypergraph self-attention) 계층을 도입했습니다.

- **Technical Details**: HAESum은 먼저 단어와 문장의 관계를 나타내는 로컬 이종 그래프를 구성하고 문장 표현을 문장 내 수준에서 업데이트합니다. 그 후, 로컬 문장 표현을 하이퍼그래프 셀프 어텐션 계층에 전달하여 셀프 어텐션 메커니즘을 통해 문장 간 관계를 완전히 캡처합니다. 이 과정에서 노드와 엣지 사이의 관계를 업데이트하여 문서의 로컬 및 글로벌 계층 정보를 포함하는 표현을 생성합니다.

- **Performance Highlights**: 두 개의 벤치마크 데이터셋에서 HAESum의 효과를 검증한 결과, 이 접근법이 강력한 기준 모델들에 비해 우수한 성능을 보여주었습니다. 특히, 계층적 구조를 고려하는 것이 긴 과학 문서를 모델링하는 데 중요함을 실험적으로 입증했습니다.



### LFED: A Literary Fiction Evaluation Dataset for Large Language Models (https://arxiv.org/abs/2405.10166)
- **What's New**: 이번 논문에서는 LFED(Literary Fiction Evaluation Dataset)라는 새로운 데이터셋을 제안하여, 대형 언어 모델(LLMs)의 긴 소설 이해 및 추론 능력을 평가하고자 합니다. LFED는 약 95개의 중국어 원본 또는 번역 소설을 포함하며, 1,304개의 질문을 작성하여 모델의 성능을 평가합니다.

- **Technical Details**: 데이터셋은 8가지 질문 유형을 정의하고, 각 질문 유형별로 문자열 선택 질문을 구성합니다. 문항 생성은 크라우드소싱을 통해 이루어졌으며, 모델들은 제로샷(Zero-shot) 및 몇 샷(Few-shot) 설정에서 평가되었습니다. 각 소설은 캐릭터 수, 출판 연도 등 특성에 따라 분석되었습니다.

- **Performance Highlights**: 최신 대형 언어모델(LLMs)을 여러 가지 실험을 통해 평가한 결과, 긴 소설 질문에 효과적으로 답변하는 데 큰 어려움이 있음을 확인했습니다. 예를 들어, ChatGPT는 제로샷 설정에서 57.08%의 정확도를 기록했습니다.



### Speaker Verification in Agent-Generated Conversations (https://arxiv.org/abs/2405.10150)
- **What's New**: 최근 대형 언어 모델(LLM)의 성공은 다양한 화자들의 특징과 스타일에 맞춘 역할 수행 대화 에이전트를 개발하는 데 큰 관심을 불러일으켰습니다. 이러한 연구는 일반 및 특수 목적 대화 작업 모두를 향상시키기 위함입니다. 하지만 화자에 맞춘 생성된 발화(personalized utterances)의 능력은 충분히 연구되지 않았습니다. 이를 해결하기 위해, 본 연구는 에이전트 생성 대화에서 화자 검증(speaker verification)이라는 새로운 평가 과제를 소개합니다. 이는 두 발화 집합이 동일한 화자로부터 나온 것인지 검증하는 것을 목표로 합니다.

- **Technical Details**: 우리는 수천 명의 화자와 그들의 발화를 포함한 대규모 데이터셋(collection)을 수집하였으며, 실험 설정 하에 화자 검증 모델(speaker verification models)을 개발하고 평가했습니다. 특히, 화자 검증은 신규 화자(즉, 훈련 데이터에서 보지 못한 화자)의 발화를 포함할 수 있으며, 이는 역할 수행 에이전트(role-playing agents)가 동일한 스타일 및 개인 특성을 보유하고 있는지 평가하는 중요한 접근법입니다. 우리는 스타일 임베딩(style embedding), 저작자 검증(authorship verification) 등의 방법을 이용하여 실험을 설계했고, 대화 맥락의 난이도에 따라 Base, Hard, Harder 세 개의 레벨로 나누어 평가했습니다.

- **Performance Highlights**: 우리의 실험 결과는 비전문가나 ChatGPT 모두 화자 검증을 정확하게 수행하지 못한다는 점에서 현재의 평가 방법이 가진 한계와 도전 과제를 부각했습니다. 반면, 세밀히 조정한(fine-tuned) 모델은 화자를 효과적으로 구별하는 능력을 보여주었습니다. 이러한 모델을 통해 생성된 발화가 실제 화자의 스타일과 얼마나 일치하는지 평가하는 시뮬레이션 점수(Simulation Score)와 발화들 간의 구별을 평가하는 구별 점수(Distinction Score)를 도입했습니다. 이 분석을 통해 LLM 기반 역할 수행 모델은 목표 화자의 스타일이나 개인 특성을 정확히 모방하는데 어려움을 겪고 있음을 확인했습니다.



### PL-MTEB: Polish Massive Text Embedding Benchmark (https://arxiv.org/abs/2405.10138)
Comments:
          10 pages, 6 tables, 1 figure

- **What's New**: 이 논문에서는 폴란드어 텍스트 임베딩(Polish Text Embedding)에 관한 종합적인 벤치마크인 PL-MTEB를 소개합니다. PL-MTEB는 5가지 유형의 28개의 다양한 NLP 과제로 구성되어 있으며, 폴란드 NLP 커뮤니티에서 이전에 사용되었던 데이터셋을 기반으로 과제를 적응시켰습니다. 또한 과학 연구 논문의 제목과 초록으로 구성된 새로운 데이터셋 PLSC(Polish Library of Science Corpus)를 만들어 두 가지 새로운 클러스터링 과제의 기초로 활용했습니다.

- **Technical Details**: PL-MTEB는 텍스트 임베딩 모델을 평가하기 위해 두 개의 모듈로 구성되어 있습니다. 첫 번째 모듈은 평가를 위한 설정 및 실행을 담당하며, 두 번째 모듈은 폴란드어에 특화된 과제 모음과 평가 방법을 포함한 확장된 MTEB 버전입니다. 이 벤치마크는 분류(Classification), 클러스터링(Clustering), 쌍 분류(Pair Classification), 검색(Retrieval), 의미적 텍스트 유사성(Semantic Textual Similarity) 등의 5가지 과제 유형을 포함합니다. 각 과제 유형마다 주요 평가 메트릭이 다릅니다. 분류 과제에서는 정확도(Accuracy), 클러스터링에서는 V-측정(V-measure), 쌍 분류에서는 평균 정밀도 점수(Average Precision Score), 검색에서는 정규화 할인 누적 이득(Normalized Discounted Cumulative Gain, nDCG@10), 의미적 텍스트 유사성에서는 스피어만 상관(Spearman Correlation)을 사용합니다.

- **Performance Highlights**: PL-MTEB를 이용해 15개의 공개된 모델(폴란드어 모델 8개 및 다국어 모델 7개)을 평가했으며, 개별 과제와 각 과제 유형 및 전체 벤치마크에 대한 종합적인 결과를 수집했습니다. 해당 모델들의 성능을 비교하고 평가 결과를 바탕으로 폴란드어 텍스트 임베딩을 위한 최적의 모델을 선택하는 데 도움이 될 수 있습니다. 또한, PL-MTEB는 공개 소스 코드와 함께 제공되어 누구나 실험을 재현하고 추가 연구를 진행할 수 있습니다.



### Turkronicles: Diachronic Resources for the Fast Evolving Turkish Languag (https://arxiv.org/abs/2405.10133)
- **What's New**: 터키어는 지난 한 세기 동안 정부의 개입으로 인해 많은 변화를 겪었습니다. 이번 연구에서는 터키 공화국이 설립된 1923년 이후 터키어의 변화를 조사하고자 합니다. 이를 위해 'Turkronicles'라는 터키어의 역사적 변화를 분석할 수 있는 코퍼스를 소개합니다. 또한 기존의 그랜드 내셔널 어셈블리 코퍼스를 확장하여 다양한 연대를 포함시켰습니다.

- **Technical Details**: Turkronicles 코퍼스는 터키의 공식 관보(Official Gazette of Türkiye)에서 추출한 45,375개의 문서로 구성되어 있으며, 정부 정책의 영향을 받은 언어적 변화를 분석하는 데 중요한 자원입니다. 코퍼스에는 총 842M단어와 211K개의 고유 단어가 포함되어 있습니다. 또한, 이 코퍼스와 기존 코퍼스를 결합하여 터키어 어휘와 문법 규칙의 변화에 대한 두 가지 주요 연구 질문에 답하려고 합니다. 첫째, 1920년대 이후 터키어 어휘가 어떻게 변했는가? 둘째, 1920년대 이후 문법 규칙이 어떻게 변했는가?

- **Performance Highlights**: 분석 결과, 두 시기 간의 어휘는 시간이 지남에 따라 더 많이 달라졌으며, 새로운 터키어 단어가 이전의 단어를 대체하는 경향이 발견되었습니다. 예를 들어, 'kitab' 대신 'kitap'과 같은 단어로 바뀌었습니다. 또한 걸프렉스(circumflex)의 사용이 눈에 띄게 감소했고, '-b'와 '-d'로 끝나는 단어는 각각 '-p'와 '-t'로 대체되었습니다.



### StyloAI: Distinguishing AI-Generated Content with Stylometric Analysis (https://arxiv.org/abs/2405.10129)
Comments:
          25th International Conference on Artificial on Artificial Intelligence in Education(AIED 2024)

- **What's New**: 새로운 연구 StyloAI가 발표되어 AI가 작성한 텍스트를 구별하는 새로운 방법을 제안합니다. StyloAI는 31개의 스타일로메트릭(stylometric) 특징(features)을 사용하여 Random Forest 분류기를 통해 다중 도메인 데이터셋에서 AI 생성 텍스트를 식별하며, AuTexTification 데이터셋에서 81%, Education 데이터셋에서 98%의 정확도를 달성했습니다.

- **Technical Details**: StyloAI 모델은 31개의 스타일로메트릭 특징을 사용하여 텍스트의 작가를 식별합니다. 이 특징은 어휘적 다양성(lexical diversity), 구문 복잡성(syntactic complexity), 감정 및 주관성(sentiment and subjectivity), 가독성(readability), 고유명사(named entities), 고유성과 다양성(uniqueness and variety)으로 분류됩니다. 또한, 이 모델은 Google Colab 플랫폼에서 Scikit-Learn 라이브러리를 사용해 머신 러닝 분류기를 훈련시켰고, 5-폴드 교차 검증을 통해 신뢰할 수 있는 결과를 도출했습니다.

- **Performance Highlights**: 제안된 StyloAI 모델은 두 가지 데이터셋에서 각각 81%와 98%의 정확도를 달성하며, 현재의 최신 모델들을 능가하는 성능을 보였습니다. 이는 다양한 분야에서 AI가 작성한 텍스트와 인간이 작성한 텍스트를 구분하는 데 있어 큰 발전을 의미합니다.



### Red Teaming Language Models for Contradictory Dialogues (https://arxiv.org/abs/2405.10128)
Comments:
          18 pages, 5 figures

- **What's New**: 이번 연구에서는 대화 중 자가 모순(self-contradiction)을 감지하고 수정하는 새로운 대화 처리 작업을 제안했습니다. 이 작업은 맥락 충실도(context faithfulness) 및 대화 이해(dialogue comprehension)에 대한 연구에서 영감을 받아, 모순을 탐지하고 이해하기 위해서는 상세한 설명이 필요함을 강조합니다. 이를 위해 연구진은 모순적 대화 데이터셋을 개발하였으며, 이 데이터셋은 자가 모순이 포함된 대화와 해당 모순의 위치와 세부 사항을 설명하는 라벨로 구성되어 있습니다.

- **Technical Details**: 대화 모순 탐지 및 수정 작업을 수행하기 위해 Red Teaming 프레임워크를 제안합니다. 이 프레임워크는 모순을 탐지하고 설명을 제공한 후, 해당 설명을 사용하여 모순적 내용을 수정합니다. 데이터셋은 ChatGPT와 Wikipedia에서 수집된 12,000개 이상의 대화로 구성되어 있으며, 이중 6,000개 이상의 대화는 한 가지 이상의 모순적 맥락을 포함합니다. 대화는 15개의 일상 대화 주제와 700개 이상의 특정 주제를 포함하여 다양성을 확보했습니다.

- **Performance Highlights**: 제안된 Red Teaming 프레임워크는 모순 탐지 정확도와 설명 유효성 측면에서 강력한 베이스라인 모델을 뛰어넘어 두 배 이상의 성능을 보여주었습니다. 또한, 대화에서 논리적 불일치를 수정하는 능력도 입증되었습니다. 이를 통해, 대화형 AI의 논리적 일관성 문제를 해결하여 사용자 경험을 개선할 수 있음을 강조합니다.



### Distilling Implicit Multimodal Knowledge into LLMs for Zero-Resource Dialogue Generation (https://arxiv.org/abs/2405.10121)
Comments:
          Under Review

- **What's New**: 이 논문은 대형 언어 모델(Large Language Models, LLMs)에서 멀티모달(multimodal) 지식을 통합하여 제로 리소스(zero-resource) 상황에서도 풍부한 대화 생성을 가능하게 하는 'Visual Implicit Knowledge Distillation Framework (VIKDF)'를 제안합니다. VIKDF는 암묵적 시각 지식을 추출하고 이를 LLM에 통합하여 효과적인 대화 생성을 목표로 합니다.

- **Technical Details**: VIKDF는 두 주요 단계로 구성됩니다. 첫째, 지식 증류 단계에서는 'Implicit Query Transformer'를 사용하여 이미지-텍스트 쌍에서 암묵적 시각 지식을 추출하고 이를 지식 벡터로 인코딩합니다. 둘째, 지식 통합 단계에서는 'Bidirectional Variational Information Fusion' 기술을 사용하여 추출된 지식 벡터를 LLM에 자연스럽게 통합합니다. 이 과정은 텍스트 문맥과 암묵적 시각 지식 간의 상호 정보를 최대화하는 방식을 채택합니다.

- **Performance Highlights**: 두 개의 대화 데이터셋(이미지-챗 데이터셋과 Reddit 대화 데이터셋)에서 실험한 결과, VIKDF는 기존 최신 모델보다 높은 품질의 대화를 생성하는 데 성공했습니다. 자동 평가 및 사람 평가를 통해 VIKDF가 문맥적으로 풍부하고 일관된 대화를 생성하는 능력을 입증했습니다.



### SynthesizRR: Generating Diverse Datasets with Retrieval Augmentation (https://arxiv.org/abs/2405.10040)
- **What's New**: 이번 연구에서는 데이터셋의 다양성 문제를 해결하기 위해 LLM을 사용하여 새로운 'Synthesize by Retrieval and Refinement (SynthesizRR)' 기법을 제안합니다. 이 방법은 기존의 Few-shot 프롬팅(few-shot prompting) 방식을 개선하여, 데이터셋 합성을 위한 콘텐츠 소싱(content sourcing) 단계에서 검색 증강(retrieval augmentation)을 활용합니다.

- **Technical Details**: SynthesizRR는 처음에 입력 예제를 검색 쿼리로 사용하여 대규모 도메인 특화 말뭉치(domain-specific corpus)에서 여러 문서를 검색합니다. 그런 다음 일반 LLM을 사용하여 검색된 각 문서에 대해 작업을 반전(task inversion)시킵니다. 이를 통해 각각의 프롬프트에 독특한 검색된 문서를 사용함으로써, 더 다양한 예제를 생성하고, 실제 세계 엔터티와 주장을 더 많이 포함한 데이터셋을 만듭니다.

- **Performance Highlights**: 실험 결과, SynthesizRR는 Six text classification tasks에서 FewGen보다 데이터의 다양성과 인간이 작성한 텍스트와의 유사성 면에서 큰 향상을 보였습니다. 또한 몇몇 다중 클래스 작업(multiclass tasks)에서 학생 분류기(student classifier)의 성능이 크게 향상되었습니다. SynthesizRR가 생성한 예제는 클래스 분리 가능성(class-separability)의 측면에서도 뛰어났습니다. 그리고 여섯 가지 기존 합성 접근 방식과의 비교에서도 SynthesizRR가 높은 데이터 다양성과 학생 모델의 정확도를 보여주었습니다.



### Listen Again and Choose the Right Answer: A New Paradigm for Automatic Speech Recognition with Large Language Models (https://arxiv.org/abs/2405.10025)
Comments:
          14 pages, Accepted by ACL 2024

- **What's New**: 최근 대형 언어 모델(LLMs)의 발전은 자동 음성 인식(ASR)의 생성적 오류 수정(GER)에 큰 도움이 되었습니다. GER은 디코딩된 N-best 가설로부터 최종 정확한 전사를 예측하는데 초점을 맞춥니다. 최신 연구에서는 'ClozeGER'라는 새로운 패러다임을 제안하여 ASR의 GER을 개선하고자 합니다.

- **Technical Details**: ClozeGER는 멀티모달 언어 모델, 흔히 SpeechGPT로 알려진 LLM을 도입하여 원본 음성을 추가 입력으로 받아들입니다. 이렇게 함으로써 오류 수정 결과의 신뢰성을 높입니다. 다음으로, GER을 클로즈 테스트(cloze test) 형식으로 재구성하여 입력 정보의 중복성을 제거하고, 명확한 지침을 통해 수정 작업을 단순화합니다. 이 방법은 로짓 교정(logits calibration) 기술을 통해 구현됩니다. 마지막으로, 간단한 후처리 단계가 추가되어 N-best 목록 전반에 걸친 동일한 부분의 오류를 수정합니다.

- **Performance Highlights**: 실험 결과, ClozeGER는 기존 GER 방법론을 9개의 유명한 ASR 데이터셋에서 능가하는 성과를 보여주었습니다.



### FinTextQA: A Dataset for Long-form Financial Question Answering (https://arxiv.org/abs/2405.09980)
- **What's New**: 새로운 금융 도메인 장문형 질문 응답(LFQA) 데이터셋인 FinTextQA가 소개되었습니다. 이 데이터셋은 금융 교과서 및 정부 웹사이트에서 추출된 1,262개의 고품질, 출처 명시된 질문-응답 쌍으로 구성되어 있어 질문 유형 및 맥락의 다양성을 확보하고 있습니다.

- **Technical Details**: FinTextQA는 Retrieval-Augmented Generation (RAG) 기반의 LFQA 시스템을 개발했습니다. 이 시스템은 임베더(embedder), 리트리버(retriever), 리랭커(reranker), 생성기(generator)로 구성되어 있습니다. 다양한 모델 구성의 성능을 평가하기 위해 사람 순위, 자동화된 지표, GPT-4 점수 등의 다면 평가 접근 방식을 사용했습니다.

- **Performance Highlights**: 모든 비교된 생성기(generator) 중에서 Baichuan2-7B는 정확도 점수 면에서 GPT-3.5-turbo에 근접한 성능을 보였으며, 가장 효과적인 시스템 구성은 Ada2 임베더, Automated Merged Retrieval 리트리버, Bge-Reranker-Base 리랭커, Baichuan2-7B 생성기였습니다. 또한, 문맥 길이가 특정 임계값에 도달한 후에는 모델들이 노이즈에 덜 민감하게 반응하는 것으로 나타났습니다.



### Mitigating Text Toxicity with Counterfactual Generation (https://arxiv.org/abs/2405.09948)
- **What's New**: 이번 연구에서는 eXplainable AI(XAI)의 반사실적 생성 방법을 활용하여 텍스트 독성(taxicity)을 감지하고 완화하는 새로운 접근법을 제안합니다. 이 접근법은 기존의 방식이 초기의 비독성 의미를 보존하지 못하는 문제를 해결하고자 합니다.

- **Technical Details**: 연구에서는 Local Feature Importance(LFI)와 반사실적 생성(counterfactual generation) 기법을 사용하여 독성 분류기에서 독성을 타겟팅하고 완화합니다. 특히, TIGTEC 모델을 활용한 CF-DetoxTIGTEC 방법을 제안하여 독성 텍스트를 다시 작성하여 독성을 제거하면서도 원래의 의미를 최대한 보존합니다. 이 방법은 GPT-3, LaMDA와 같은 대형 언어 모델(LLM)에 독성 데이터가 포함되는 문제를 해결하고자 합니다.

- **Performance Highlights**: 세 가지 데이터셋에서 자동 및 인간 평가를 통해 CF-DetoxTIGTEC 방법이 기존의 MaRCo 및 CondBERT 방법보다 더 높은 성능을 보여줍니다. 이는 독성 완화, 내용 보존 및 텍스트의 개연성 측면에서 높은 평가를 받았음을 의미합니다. 실험 결과 독성 완화에서 경쟁력 있는 성능을 확인하였고, 인간 평가에서도 긍정적인 평가를 받았습니다.



### SciQAG: A Framework for Auto-Generated Scientific Question Answering Dataset with Fine-grained Evaluation (https://arxiv.org/abs/2405.09939)
- **What's New**: 본 연구에서는 과학 문헌을 기반으로 자동으로 질문-답변(QA) 쌍을 생성하고 평가하는 프레임워크인 SciQAG를 소개했습니다. 오픈 소스 대형 언어 모델(LLM)을 미세 조정하여 과학 논문에서 96만 개의 과학 QA 쌍을 생성하고, 생성된 QA 쌍의 품질을 평가하기 위한 5차원 척도(metric)를 제안했습니다.

- **Technical Details**: SciQAG 프레임워크는 3단계로 구성됩니다. 첫째, Seed QA 단계에서는 무작위로 선택된 과학 논문 123편에서 GPT-4를 사용해 QA 쌍을 생성하고, 도메인 전문가가 효과적인 프롬프트를 설계합니다. 둘째, QA Generator 단계에서는 미세 조정된 모델을 사용해 다양한 과학 논문에서 QA 쌍을 대량으로 생성합니다. 셋째, QA Evaluator 단계에서는 다른 LLM을 사용해 생성된 QA 쌍을 5가지 기준(관련성, 문헌 비의존성, 완성도, 정확성, 합리성)에 따라 평가합니다. 이 프레임워크는 유연하며 선택적 미세 조정 단계를 포함합니다.

- **Performance Highlights**: LLM 기반 평가 결과, 생성된 QA 쌍은 5가지 기준에서 평균 점수 3점 만점에 2.5점을 꾸준히 기록했습니다. 이는 SciQAG 프레임워크가 논문에서 핵심 지식을 고품질 QA 쌍으로 대량 추출할 수 있다는 것을 의미합니다. 생성된 데이터셋은 과학적 발견을 위한 대형 모델을 훈련시키는 데 사용될 수 있으며, 모델 붕괴를 막기 위해 정확하고 도메인 특정 지식을 제공합니다.



### DEBATE: Devil's Advocate-Based Assessment and Text Evaluation (https://arxiv.org/abs/2405.09935)
- **What's New**: 최근 자연어 생성(NLG) 모델의 발전으로 인해 기계 생성 텍스트의 품질 평가가 더욱 중요해졌습니다. 기존의 LLM 기반 평가 방식이 단일 에이전트를 사용해 효율성을 저해하는 편향을 갖고 있다고 주장하며, 이를 해결하기 위해 '디베이트(DEBATE)'라는 다중 에이전트 점수 시스템을 도입했습니다. 이 시스템은 'Devil's Advocate' 개념을 이용하여 에이전트 간의 비판적 토론을 통해 편향을 줄여줍니다.

- **Technical Details**: DEBATE는 세 명의 에이전트로 구성된 다중 에이전트 평가 프레임워크입니다. 이 세 에이전트는 지휘관(Commander), 점수 계산자(Scorer), 비평가(Critic) 역할을 맡습니다. 지휘관은 토론을 주도하며, 점수 계산자는 주어진 작업에 대해 점수를 계산하고, 비평가는 점수 계산자의 결과를 비판하는 'Devil's Advocate' 역할을 합니다. 이 시스템은 템플릿, 다중 에이전트 프레임워크, 비평가의 비판적 역할 등 세 가지 주요 요소로 구성됩니다.

- **Performance Highlights**: DEBATE는 SummEval과 TopicalChat이라는 두 개의 메타 평가 벤치마크에서 기존 최고 성능을 크게 능가하였습니다. 특히, DEBATE는 SummEval에서 G-Eval보다 6.4% 높은 ρ(상관계수)와 12.5% 높은 τ(스피어만 순위 상관계수)를 기록했습니다. TopicalChat에서도 유사한 성능 향상을 보였습니다.



### TransMI: A Framework to Create Strong Baselines from Multilingual Pretrained Language Models for Transliterated Data (https://arxiv.org/abs/2405.09913)
Comments:
          preprint

- **What's New**: 이 논문은 다른 스크립트(script)를 사용하는 관련 언어를 공통 스크립트로 음역(transliteration)하여 크로스링구얼 트랜스퍼(crosslingual transfer)의 성능을 향상시키는 새로운 프레임워크인 'Transliterate-Merge-Initialize (TransMI)'를 제안합니다. 기존의 mPLMs(multilingual pretrained language models)를 이용하여 새로운 서브워드를 생성하고 음역 데이터를 효과적으로 처리할 수 있도록 합니다.

- **Technical Details**: TransMI는 세 가지 단계로 구성됩니다. 첫째, mPLM의 어휘를 공통 스크립트(라틴 스크립트)로 음역(transliterate)합니다. 둘째, 새로 얻어진 서브워드를 원래의 토크나이저(tokenizer)와 병합(merge)합니다. 마지막으로, 새로 추가된 서브워드의 임베딩(embedding)을 초기화(initialize)합니다. 이 프레임워크는 기존의 mPLM과 그에 수반되는 토크나이저를 활용하여 음역 데이터를 처리하는 모델을 구축하는 간단하면서도 효과적인 방법을 제공합니다.

- **Performance Highlights**: 세 가지 최신 mPLM에 TransMI를 적용한 결과, 비음역 데이터를 처리하는 능력을 유지하면서 음역 데이터에 대해서도 높은 성능을 발휘했습니다. 실험 결과, 다양한 모델과 작업에서 3%에서 34%에 이르는 일관된 성능 향상이 있었습니다. 음역 데이터 세트와 비음역 데이터 세트 모두에서 평가를 수행한 결과, TransMI로 조작된 모델이 음역 데이터에서 기존 mPLM보다 높은 성능을 기록했습니다.



### IGOT: Information Gain Optimized Tokenizer on Domain Adaptive Pretraining (https://arxiv.org/abs/2405.09857)
- **What's New**: 본 논문에서는 고도화된 도메인 적응 훈련을 위해 Information Gain Optimized Tokenizer (이하 IGOT) 기법을 제안합니다. 이는 특화된 토큰 세트를 분석하고 정보 이득(Information Gain)을 통해 새로운 도메인-특화 토크나이저를 구축하여 다운스트림 작업 데이터를 계속 학습시키는 방식입니다.

- **Technical Details**: IGOT는 다운스트림 작업에서 사용되는 특수 토큰과 그 정보 이득을 분석하여 새로운 하위 집합을 구성하는 휴리스틱 함수 겸 토크나이저입니다. 이 방법을 통해 일반적인 데이터 수집과 파인 튜닝만으로는 얻기 힘든 성능 향상을 달성할 수 있습니다. 수학적 방법과 실험을 통해 IGOT의 우수성을 확인하였습니다. 특히, 분류 시스템에서 정보 이득은 피처의 중요성을 나타내며, 높은 정보 이득일수록 중요도가 더 크다는 점을 이용합니다.

- **Performance Highlights**: IGOT를 LLaMA-7B 모델에 적용한 결과, 11.9%의 토큰 절약, 12.2%의 훈련 시간 절약, 5.8%의 최대 GPU VRAM 사용량 절약을 달성하였습니다. 또한 T5 모델과 결합했을 때 최대 31.5%의 훈련 시간 절약이 가능했습니다. 이로 인해 일반적인 생성형 AI를 특화된 도메인에 보다 효율적으로 적용할 수 있게 되었습니다.



### On the relevance of pre-neural approaches in natural language processing pedagogy (https://arxiv.org/abs/2405.09854)
Comments:
          Under review at Teaching NLP workshop at ACL 2024; 8 pages

- **What's New**: 최근의 자연어 처리(NLP) 교육과정에서 Transformer 모델과 비신경(pre-neural) 접근법을 어떻게 균형있게 다루고 있는지를 비교한 연구가 발표되었습니다. 이 연구는 호주와 인도의 두 대학에서 제공되는 NLP 입문 강의를 비교하며, 비신경 접근법이 여전히 학생들의 직관적인 이해를 돕는 가치가 있음을 주장합니다.

- **Technical Details**: NLP 입문 강의에서 Transformer와 비신경 접근법의 균형을 어떻게 잡을 수 있는지에 대해 조사합니다. 주요 논점은 (a) 다른 NLP 교육 기관들이 어떻게 가르치고 있는지, (b) 연구진이 자신들의 강의에서 어떻게 다루고 있는지입니다. 연구는 초기 경력의 교육자와 20년 이상의 경험을 가진 베테랑 교육자가 각각 운영하는 두 NLP 강의를 비교하며, 비신경 접근법이 심층 학습 접근법과의 비교를 통해 문제 해결 능력 향상과 개념 이해에 어떻게 기여하는지를 설명합니다.

- **Performance Highlights**: 기존 연구와 달리, 이 논문은 두 교육자가 운영하는 NLP 강의의 비교를 통해 비신경 접근법의 현재 교육적 역할과 가치를 강조합니다. 두 강의 모두 학생들에게 문제를 해결하고 심층 학습 접근법을 이해하는 데 있어서 비신경 접근법이 중요한 역할을 한다고 보고합니다. 특히, 비신경 접근법이 Transformer 기반 모델을 이해하는 데 도움이 된다고 설명하며, 이러한 접근법의 교육적 유용성을 강조합니다.



### Enhancing Semantics in Multimodal Chain of Thought via Soft Negative Sampling (https://arxiv.org/abs/2405.09848)
Comments:
          Accepted by LREC-COLING 2024

- **What's New**: 이 연구는 복잡한 추론이 필요한 문제에서 체인 오브 사고(Chain of Thought, CoT)의 효과성을 입증했습니다. 특히 이런 문제들은 텍스트와 멀티모달(multimodal) 요소가 모두 포함된 경우가 많습니다. 이 연구는 부정적 합리화(negative rationales) 생성 방법을 개선하여 멀티모달 CoT에서 발생하는 환각(hallucination) 문제를 완화하는 것을 목표로 하고 있습니다.

- **Technical Details**: 본 연구는 Soft Negative Sampling (SNSE-CoT)을 사용하여 환각 문제를 완화하는 새로운 합리화 생성 방법을 제안합니다. 이 방법에서는 기존의 정상적인 포지티브(positive) 및 네거티브(negative) 샘플만을 포함하는 전통적인 대조 학습 프레임워크에 비슷한 텍스트지만 다른 의미를 가진 부정적 샘플을 추가하여 사용합니다. 특수 손실 함수인 역이중마진 손실(Bidirectional Margin Loss, BML)을 적용하여 이러한 부정적 샘플을 대조 학습에 도입합니다.

- **Performance Highlights**: ScienceQA 데이터셋에서 광범위한 실험을 통해 제안된 방법의 유효성을 검증했습니다. 실험 결과, 제안된 SNSE-CoT 모델이 기존의 다른 모델들보다 대부분의 멀티모달 CoT 범주에서 우수한 성능을 보였습니다. 연구 코드와 데이터는 추가 연구를 위해 공개되었습니다.



### Chameleon: Mixed-Modal Early-Fusion Foundation Models (https://arxiv.org/abs/2405.09818)
- **What's New**: Chameleon은 텍스트와 이미지를 임의의 순서로 이해 및 생성할 수 있는 초기 결합 토큰 기반의 혼합 모달 모델(family of early-fusion token-based mixed-modal models)입니다. 이 모델은 시각적 질문 응답(visual question answering), 이미지 캡션링(image captioning), 텍스트 생성(text generation), 이미지 생성(image generation) 및 장기 혼합 모달 생성(long-form mixed-modal generation)과 같은 다양한 작업을 수행할 수 있습니다. 특히 이미지 캡션링에서는 최첨단 성능을 보여주고, 텍스트 전용 작업에서도 Llama-2를 능가하며 Mixtral 8x7B 및 Gemini-Pro와 경쟁할 수 있는 성능을 발휘합니다.

- **Technical Details**: Chameleon은 이미지와 텍스트 내용을 모두 토큰화하여 동일한 변환기(transformer) 아키텍처 내에서 처리합니다. 이를 통해 별도의 이미지/텍스트 인코더가 필요 없으며, 모든 모달리티를 공유 표현 공간(shared representational space)으로 투영합니다. 이를 위해 쿼리 키 정규화(query-key normalization) 및 레이어 정규화(layer norms)의 위치 수정과 같은 변형 아키텍처와 새로운 훈련 기법을 도입하여 초기 결합 설정에서 안정적인 훈련을 가능하게 했습니다. 또한, BPE 토크나이저를 사용하여 텍스트 및 이미지 토큰을 65,536개의 어휘 크기로 압축하여 처리합니다.

- **Performance Highlights**: 시각적 질문 응답 및 이미지 캡션링 벤치마크에서 Chameleon-34B는 Flamingo, IDEFICS, Llava-1.5 모델을 능가하는 성능을 보여주었습니다. 텍스트 전용 벤치마크에서는 Mixtral 8x7B 및 Gemini-Pro와 동등한 성능을 유지하며, 혼합 모달리티 장기 생성 및 추론에서는 GPT-4V를 능가하는 능력을 보였습니다. 인간 평가 실험에서도 Gemini-Pro 대비 60.4%, GPT-4V 대비 51.6%의 선호율을 보여주면서 높은 성능을 입증했습니다.



### SecureLLM: Using Compositionality to Build Provably Secure Language Models for Private, Sensitive, and Secret Data (https://arxiv.org/abs/2405.09805)
- **What's New**: 이번 연구는 민감한 정보를 다루는 환경에서 사용할 수 있는 LLM(대형 언어 모델)을 안전하게 구축하는 새로운 방법을 제안합니다. 기존 방식들은 악의적인 사용자나 결과를 보호하려 했지만, 민감한 데이터에 대해 충분히 안전하지 않았습니다. 이와 달리 SecureLLM은 접근 보안(access security)과 미세 조정(fine-tuning) 방법을 결합하여 사용자의 권한에 따라 모델을 구성합니다.

- **Technical Details**: 각 데이터 사일로(data silo)는 별도의 미세 조정을 거치며, 사용자는 허가된 미세 조정만 접근할 수 있습니다. 이 모델은 다양한 사일로의 교차 영역(compositional tasks)에서 작동해야 하며, 새로운 SQL 데이터베이스의 레이아웃을 학습하여 자연어를 SQL로 번역하는 능력을 제공합니다. 이를 위해 모델의 가중치 일부만 업데이트하고, 필요에 따라 해당 부분을 분리하여 저장하는 방법을 사용합니다.

- **Performance Highlights**: SecureLLM은 LoraHub 등의 기존 방법보다 높은 성능을 보입니다. 예를 들어, A와 B 사일로에 접근 권한이 있는 사용자는 C와 관련된 주제를 피하면서 A와 B의 미세 조정을 결합하여 정확한 응답을 생성할 수 있습니다. 기존의 미세 조정 방법들이 이러한 구성 과제에서 실패하는 반면, SecureLLM은 보안성을 유지하면서도 높은 성능을 발휘합니다.



### Optimization Techniques for Sentiment Analysis Based on LLM (GPT-3) (https://arxiv.org/abs/2405.09770)
- **What's New**: 이 논문은 GPT-3와 같은 대규모 사전 학습된 언어 모델을 기반으로 감정 분석 최적화 기술을 탐구하여 모델 성능을 향상시키고 자연어 처리(NLP) 기술의 발전을 촉진하는 것을 목표로 하고 있습니다. 기존 방식을 대체하는 GPT-3 및 Fine-tuning 기법(fine-tuning techniques)을 소개하고, 이를 감정 분석(sentiment analysis)에 어떻게 적용하는지 설명합니다.

- **Technical Details**: 논문에서는 감정 분석의 중요성과 전통적인 방법들의 한계를 먼저 설명한 후, GPT-3 모델의 세부 사항과 Fine-tuning 기법에 대해 다룹니다. Fine-tuning 기법은 이미 학습된 대규모 언어 모델을 특정 작업에 맞게 조정하는 과정을 의미합니다. 이러한 방식으로 GPT-3의 성능을 최적화하여 감정 분석 작업에서의 효율성을 극대화할 수 있습니다.

- **Performance Highlights**: 실험 결과에 따르면 Fine-tuning 기법을 활용한 GPT-3 모델이 감정 분석 작업에서 우수한 성능을 보였습니다. 이는 대규모 언어 모델을 활용한 감정 분석 연구에 중요한 참고 자료를 제공하며, 향후 연구 방향에 큰 기여를 할 수 있을 것으로 기대됩니다.



### Unsupervised Extractive Dialogue Summarization in Hyperdimensional Spac (https://arxiv.org/abs/2405.09765)
Comments:
          ICASSP 2024

- **What's New**: HyperSum이라는 새로운 추출적 요약 프레임워크가 소개되었습니다. 이 프레임워크는 전통적인 어휘적 요약의 효율성과 현대 신경 접근법의 정확성을 동시에 잡아내는 것이 특징입니다. HyperSum은 고차원의 벡터를 무작위 초기화할 때 발생하는 '차원 축복'을 이용하여 대표적이고 효율적인 문장 임베딩을 생성합니다. 무작위로 초기화된 벡터가 매우 높은 차원에서 가짜 직교성을 나타내며, 이를 활용해 문장을 요약합니다.

- **Technical Details**: HyperSum은 두 단계로 구성됩니다: (1) 문장 임베딩 구축 및 (2) 요약 추출. 먼저, 각 문장을 고차원 벡터로 변환하여 대표성을 확보하고, 클러스터링 후 중심점을 선택하여 요약 문장을 추출합니다. 구체적으로, 고차원 컴퓨팅(Hyperdimensional Computing, HDC)을 사용하여 문장을 표현하는데, 이는 무작위로 초기화된 벡터들이 고차원에서 나타내는 가짜 직교성을 이용합니다. 각 단어는 고차원 벡터로 인코딩되고, 문장은 이러한 단어 벡터의 결합을 통해 구축됩니다. 최종적으로, k-메도이드 알고리즘을 사용해 문장의 중심 임베딩을 추출하여 요약합니다.

- **Performance Highlights**: HyperSum은 CPU 환경에서도 기존의 신경 요약 모델 대비 10배에서 100배 더 빠른 성능을 자랑합니다. 또한, 현대의 최첨단 요약 모델과 비교하여 더 높은 요약 정확도와 신뢰성을 보입니다. 이를 통해 효율성과 정확성을 모두 만족시키는 요약 시스템을 구현했습니다.



### Many Hands Make Light Work: Task-Oriented Dialogue System with Module-Based Mixture-of-Experts (https://arxiv.org/abs/2405.09744)
- **What's New**: 이 연구는 'Soft Mixture-of-Expert Task-Oriented Dialogue system (SMETOD)'을 제안합니다. 이 시스템은 대화 시스템의 성능을 향상시키기 위해 다양한 출력 생성을 위한 전문가 집합(Mixture-of-Experts, MoEs) 접근 방식을 사용합니다. SMETOD는 학습과 추론 비용을 낮추면서 모델 용량을 확장할 수 있는 혁신적인 방법을 제공합니다.

- **Technical Details**: SMETOD는 'Soft Mixture-of-Experts (Soft-MoE)' 아키텍처를 채택하여 각각의 입력 토큰에 전문가를 부드럽게 할당합니다. 이는 파라미터 효율성이 높은 미세 조정(fine-tuning) 효율성을 유지하면서 모델 용량을 확장할 수 있습니다. 이 시스템은 T5-small과 T5-base를 백본 PLM으로 사용하여 MultiWOZ 및 NLU 데이터셋에서 평가되었습니다.

- **Performance Highlights**: 실험 결과, SMETOD는 의도 예측(intent prediction), 대화 상태 추적(dialogue state tracking), 대화 응답 생성(dialogue response generation)에서 대부분의 평가 기준에서 최첨단 성능을 달성했습니다. 특히 추론 비용과 문제 해결의 정확성 면에서 기존 강력한 기준선 모델들과 비교하여 큰 이점을 보였습니다.



### An Analysis of Sentential Neighbors in Implicit Discourse Relation Prediction (https://arxiv.org/abs/2405.09735)
- **What's New**: 본 연구에서는 문장 관계 예측 작업에서 컨텍스트를 활용하는 세 가지 새로운 방법을 제안합니다: (1) Direct Neighbors (DN), (2) Expanded Window Neighbors (EWN), (3) Part-Smart Random Neighbors (PSRN). 주된 발견은 한 개의 담화 단위를 넘어선 컨텍스트의 포함이 담화 관계 분류 작업에 해롭다는 것입니다.

- **Technical Details**: 본 연구는 Penn Discourse TreeBank (PDTB)을 사용하며, DistilBERT 모델을 기반으로 합니다. 직접 이웃 (DN) 모델은 최소 한 개 이상의 직접 이웃 문장을 가진 암묵적 문장 페어를 포함합니다. 확장 윈도우 이웃 (EWN) 모델은 동일한 문서 내의 가장 가까운 이전 이웃 문장과 가장 가까운 이후 문장을 추가합니다. 파트-스마트 랜덤 이웃 (PSRN) 모델은 동일한 문서 내의 랜덤 이전 이웃 문장과 랜덤 이후 문장을 추가합니다. 각 모델은 Hugging Face 하이퍼튜닝 파이프라인을 통해 DistilBERT로 훈련되었습니다.

- **Performance Highlights**: 전체적으로 Baseline 모델이 제안된 모든 컨텍스트 윈도우 전략들을 능가했습니다. PSRN과 EWN (N=2) 모델이 Baseline과 가장 경쟁적인 성능을 보였습니다. PSRN 모델은 정확도에서 58%를 기록하였고, EWN (N=2) 모델은 57.5%를 기록했습니다. 에포크 10에서 F1 점수는 EWN (N=2)이 57.2%, PSRN이 57%로 거의 차이가 없었습니다.



### SCI 3.0: A Web-based Schema Curation Interface for Graphical Event Representations (https://arxiv.org/abs/2405.09733)
- **What's New**: 새로운 논문에서는 자연어 처리(NLP)에서의 복잡한 이벤트 구조를 다루는 'Schema Curation Interface 3.0 (SCI 3.0)'에 대해 소개합니다. SCI 3.0은 사건 스키마(event schema)를 실시간으로 편집할 수 있는 웹 애플리케이션으로, 하위 이벤트, 엔티티 및 관계를 편리하게 추가, 제거, 또는 편집할 수 있습니다.

- **Technical Details**: SCI 3.0은 React.js, Flask, Cytoscape.js를 사용해 구축된 웹 애플리케이션이며, 이벤트 스키마의 계층 구조를 시각화하고 편집할 수 있는 기능을 제공합니다. 사용자는 JSON 데이터 형식을 사용하여 사건 스키마를 업로드하고, 인터페이스 내에서 직접 편집할 수 있습니다. 시스템은 이벤트 스켈레톤 구성, 이벤트 확장, 이벤트-이벤트 관계 검증의 세 단계를 포함한 알고리즘을 사용하여 초기 스키마 라이브러리를 자동으로 생성합니다.

- **Performance Highlights**: SCI 3.0은 이벤트 라이브러리의 수동 큐레이션을 통해 초기 자동 생성된 스키마의 품질과 범위를 향상시킵니다. 주요 기능으로는 그래프의 요소를 쉽게 추가 및 편집할 수 있는 기능, 직관적인 우클릭 메뉴, 확장 가능한 작업 공간 등이 포함되어 있습니다. 또한 사용자는 엔티티 및 관계를 추가하고 이벤트 사이의 시간 관계를 나타내는 'outlink'를 생성할 수 있습니다.



### Spectral Editing of Activations for Large Language Model Alignmen (https://arxiv.org/abs/2405.09719)
- **What's New**: 대규모 언어 모델(LLMs)이 발생시키는 부정확하거나 편향된 콘텐츠를 줄이기 위해 새로운 추론 시간 편집 방법, 즉 SEA(Spectral Editing of Activations)를 제안합니다. 이 방법은 입력 표현을 긍정적 예시(e.g., 진실된 내용)와의 최대 공분산을 가지는 방향으로 투영하며, 부정적 예시(e.g., 환각된 내용)와의 공분산을 최소화합니다. 또한, 비선형 편집을 위해 특징 함수(feature function)를 확장 적용합니다.

- **Technical Details**: SEA 방법은 산업적 비용이 많이 드는 반복 최적화를 필요로 하지 않으며, 클로즈드-폼 스펙트럴 분해를 통해 편집 투영을 찾습니다. LLM의 중립적, 부정적, 긍정적인 활성화 간의 공분산 행렬에 대한 특이값 분해(SVD)를 사용하여 편집 투영을 얻습니다. 비선형 편집을 위해서는 가역적 비선형 특징 함수를 사용합니다.

- **Performance Highlights**: SEA 방법을 사용한 실험에서 LLM의 진실성과 공정성 점수를 향상시킴을 보여줍니다. 예를 들어, LLaMA-2-chat 모델에 SEA를 적용했을 때 TruthfulQA 점수가 36.96에서 39.41로 향상되었으며, BBQ 데이터셋에서의 정확도가 43.02에서 56.17로 증가했습니다. 총 6개의 다양한 크기와 아키텍처를 가진 LLM에서 일관된 성능 향상을 보였습니다. 25개의 시연만을 사용해도 모델의 진실성과 공정성에서 눈에 띄는 개선이 나타났습니다.



### Simulating Policy Impacts: Developing a Generative Scenario Writing Method to Evaluate the Perceived Effects of Regulation (https://arxiv.org/abs/2405.09679)
Comments:
          Currently under review. 10 pages

- **What's New**: 이번 연구에서는 대형 언어 모델(Large Language Models, LLMs)을 이용해 특정한 정책이 부정적 영향을 완화하는 효능을 평가하는 방법을 개발했습니다. GPT-4를 사용하여, 정책 도입 전후의 시나리오를 생성하고 이를 인간의 영향을 바탕으로 측정 가능한 지표로 번역하는 방식을 채택했습니다. 이미 확립된 생성적 AI의 미디어 환경에서의 영향을 바탕으로 시나리오를 생성하고, EU AI Act의 제50조 정보 공개 관련 법안에 의해 완화된 시나리오와 그렇지 않은 시나리오 쌍을 비교 연구했습니다.

- **Technical Details**: 연구에서는 GPT-4를 사용해 다양한 시나리오를 작성하고, 제50조의 영향을 반영한 재작성된 시나리오를 비교하였습니다. 234명의 참가자를 대상으로 한 사용자 연구를 통해 시나리오의 질을 평가하고, 네 가지 위험 평가 기준(Severity, Plausibility, Magnitude, Specificity to vulnerable populations)에 따라 시나리오를 평가했습니다.

- **Performance Highlights**: 연구 결과, 제50조 정보 공개 법안은 노동 및 웰빙 분야에서의 해악 완화에는 효과적이었으나 사회적 결속 및 보안 분야에서는 효과가 미미한 것으로 나타났습니다. 이를 통해 정책이 다양한 영향 측면에서 얼마나 효과적인지를 평가할 수 있는 방법론의 유효성을 입증했습니다. 본 연구 방법론은 정책 입안자나 연구자가 다양한 정책의 잠재적 유용성을 탐구할 수 있는 도구로 활용될 수 있을 것으로 기대됩니다.



### Elements of World Knowledge (EWOK): A cognition-inspired framework for evaluating basic world knowledge in language models (https://arxiv.org/abs/2405.09605)
Comments:
          21 pages (11 main), 7 figures. Authors Anna Ivanova, Aalok Sathe, Benjamin Lipkin contributed equally

- **What's New**: 새로운 연구인 Ewok (Elements of World Knowledge)은 언어 모델에서 세계 모델링(world modeling) 능력을 평가하는 프레임워크를 제안합니다. Ewok은 사회적 상호작용부터 공간적 관계에 이르는 여러 지식 영역의 특정 개념을 대상으로 하여 적절한/부적절한 문맥과 텍스트를 매칭하는 능력을 시험합니다. Ewok-CORE-1.0이라는 데이터셋도 함께 소개되었습니다.

- **Technical Details**: Ewok 프레임워크는 기본적인 인간 세계 지식을 구성하는 여러 도메인을 포함합니다. 각 도메인은 특정 개념과 항목 템플릿, 모듈식 구성 요소를 포함하여 타겟 문장과 문맥의 정합성을 평가합니다. 템플릿은 여러 번 사용할 수 있도록 설계되었으며, 데이터셋 생성 파이프라인이 포함되어 있습니다. Ewok-CORE-1.0 데이터셋은 11개 세계 지식 도메인을 아우르는 4,374개의 항목으로 구성됩니다.

- **Performance Highlights**: 1.3B~70B 파라미터 크기의 20개 대형 언어 모델(LLM)을 평가한 결과, 모든 모델의 성능이 인간 성능보다 낮았습니다. 도메인에 따라 성능 차이가 크게 나타났으며, 단순한 사례에서도 큰 모델들이 실패하는 경우가 많았습니다. 이 연구는 LLM의 세계 모델링 능력에 대한 타겟 연구의 풍부한 가능성을 노출시킵니다.



### How Far Are We From AGI (https://arxiv.org/abs/2405.10313)
- **What's New**: 이 논문은 AGI(Artificial General Intelligence)의 정의, 목표 및 개발 경로에 대한 종합적인 논의를 다루며, AGI에 얼마나 가까워졌는지와 이를 실현하기 위한 전략을 탐구합니다. 기존 AI 발전의 제한점을 인식하고, AGI의 실현을 위해 필요한 정렬 기술을 논의하여, 책임감 있는 접근 방식을 제안합니다.

- **Technical Details**: AGI를 위한 능력 프레임워크를 내부(Internal), 인터페이스(Interface), 시스템(System) 차원으로 통합하여 설명합니다. AGI 시스템의 인지, 감정, 실행 기능 등을 포함하는 네 가지 주요 구성 요소로 나누어, 지각(Perception), 기억(Memory), 추론(Reasoning), 메타 인지(Metacognition)를 다룹니다. 더 나아가 AGI 실현을 위한 정렬 절차의 중요성을 강조하고, AGI의 평가 프레임워크 및 개발 로드맵을 제시합니다.

- **Performance Highlights**: AGI의 실현이 점점 더 가까워지고 있음을 강조하며, 여러 분야에서 AGI 통합의 도전 과제와 잠재적인 경로를 제시합니다. AGI가 사회적, 윤리적 함의와 함께 어떻게 발전해야 하는지를 탐구하고, 현재와 미래의 AGI 내부 구성 요소의 상태를 요약합니다.



### Fine-Tuning Large Vision-Language Models as Decision-Making Agents via Reinforcement Learning (https://arxiv.org/abs/2405.10292)
- **What's New**: 이 논문에서는 다중 단계의 목표 지향적 작업에서 이상적인 의사결정 에이전트를 효율적으로 학습하기 위해 강화 학습(RL)을 활용한 Large Vision-Language Models(LVM)을 미세 조정하는 알고리즘 프레임워크를 제안합니다. 이 프레임워크는 작업 설명을 제공한 후 체인 오브 생각(Chain-of-Thought, CoT) 추론을 생성해 중간 추론 단계를 탐색하고 최종 텍스트 기반 액션으로 연결합니다.

- **Technical Details**: 제안된 프레임워크는 작업 설명 프롬프트를 VLM에 제공해 CoT 추론을 생성하고, 이를 통해 중간 추론 단계를 효율적으로 탐색합니다. 이는 텍스트 기반 액션으로 변환된 후 환경과 상호작용하며, 목표 지향적 보상을 획득합니다. 마지막으로 이 보상을 사용해 RL로 전체 VLM을 미세 조정합니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안된 프레임워크는 다양한 작업에서 VLM 에이전트의 의사결정 능력을 향상시킵니다. 특히, 7b 모델을 사용해 GPT4-V 및 Gemini와 같은 상업 모델을 능가하는 성능을 보였습니다. 또한 CoT 추론이 성능 향상의 핵심 요소로 작용하며, CoT 추론을 제거할 경우 전반적인 성능이 크게 감소하는 것으로 나타났습니다.



### A Tale of Two Languages: Large-Vocabulary Continuous Sign Language Recognition from Spoken Language Supervision (https://arxiv.org/abs/2405.10266)
- **What's New**: 이번 연구의 주요 목표는 대규모 어휘 연속 수어 인식(CSLR)과 수어 검색입니다. 이를 위해, 다중 과제 Transformer 모델인 CSLR2를 도입하였습니다. 이 모델은 서명 시퀀스를 받아들여 수어와 음성 언어 텍스트 간의 공동 임베딩 공간에 출력을 제공합니다. 새로운 평가를 위해 새롭게 수집된 데이터셋 주석을 공개할 예정입니다.

- **Technical Details**: CSLR2 모델은 Transformer 구조를 기반으로 하며, 서명 비디오 시퀀스를 수용하여 서명과 음성 언어 간의 공동 임베딩 공간에 출력합니다. 모델은 두 가지 과제를 동시에 학습하여 각 과제의 성능을 향상시킵니다. 대규모 데이터셋인 BOBSL에서 추출한 약한 감독 데이터와 혼잡한 감독 데이터를 활용하여 학습을 진행합니다.

- **Performance Highlights**: CSLR2 모델은 이전의 최첨단 기술을 뛰어넘어 대규모 어휘 연속 수어 인식과 수어 검색 과제에서 높은 성능을 보여줍니다. 특히 공동 임베딩 공간을 통해 두 과제의 상호 성능 향상을 도모했습니다. 또한, 6시간 분량의 테스트 비디오에 대해 새롭게 수집된 연속 서명-레벨 주석을 통해 평가를 진행하였습니다.



### Words as Trigger Points in Social Media Discussions (https://arxiv.org/abs/2405.10213)
- **What's New**: 이번 연구는 Mau, Lux, Westheuser(2023)가 도입한 트리거 포인트(Trigger Points) 개념을 대규모 소셜 미디어 데이터를 활용하여 체계적으로 분석한 첫 번째 사례입니다. 연구는 2020년부터 2022년까지 Reddit에서의 온라인 토론을 분석하여 트리거 포인트가 사용자 참여와 적대감을 어떻게 유발하는지 조사합니다.

- **Technical Details**: 트리거 포인트는 개개인의 사회적 행동과 기대가 도전받는 순간을 가리키며, 사람들이 이러한 트리거에 대해 감정적으로 강하게 반응합니다. 연구는 'Rwanda', 'Brexit', 'NHS', 'Vaccine', 'Feminism' 등 영국 정치 관련 단어들을 트리거 포인트로 삼아 총 1억 개 이상의 Reddit 게시물을 수집하였습니다. 텍스트 및 자연어 처리(NLP) 도구를 활용하여 이러한 단어들이 얼마나 많은 부정적, 정서적, 유해한 반응을 유발하는지 분석하였습니다.

- **Performance Highlights**: 연구 결과에 따르면 트리거 단어들은 높은 수준의 사용자 참여를 촉발하며, 이는 논쟁성 증가, 부정적 감정, 분노, 혐오 발언의 증가로 측정되었습니다. 트리거 단어들이 포함된 게시물은 더 많은 메시지와 애니모시티(animosity)를 유발하며, 이는 온라인 토론의 분열과 극단화를 초래할 수 있습니다.



### Building a Luganda Text-to-Speech Model From Crowdsourced Data (https://arxiv.org/abs/2405.10211)
Comments:
          Presented at the AfricaNLP workshop at ICLR 2024

- **What's New**: 이 논문은 Luganda 언어의 Text-to-Speech (TTS) 모델 성능을 개선하기 위해, 기존의 다중 화자 데이터를 사용하는 방식에서 더 나아가, 유사한 억양을 가진 여섯 명의 여성 화자의 데이터로 훈련하는 방법을 제안합니다. 이는 기존의 Common Voice 데이터의 저품질 문제를 보완하고, 다양한 억양 문제를 해결하기 위한 것입니다.

- **Technical Details**: 데이터 전처리 과정에서 녹음의 시작과 끝 부분의 침묵을 제거하고, 사전 훈련된 음성 향상 모델을 사용하여 배경 소음을 줄였습니다. 또한, 비침해적 자가 감독 Mean Opinion Score (MOS) 추정 모델을 사용하여 추정 MOS가 3.5 이상인 녹음을 필터링했습니다. 이 방법을 통해 훈련 데이터의 품질을 향상시켰습니다.

- **Performance Highlights**: 주관적 MOS 평가에서, 제안된 모델은 3.55의 높은 MOS 점수를 기록했으며, 이는 기존 모델의 2.5 MOS보다 상당히 향상된 결과입니다. 또한, 단일 화자(3.13 MOS)나 두 명의 화자(3.22 MOS)로 훈련된 모델보다 더 자연스러운 음성이 생성되었습니다. 이는 유사한 억양을 가진 다중 화자의 데이터를 사용하여 TTS 품질을 향상시킬 수 있음을 보여줍니다.



### MarkLLM: An Open-Source Toolkit for LLM Watermarking (https://arxiv.org/abs/2405.10051)
Comments:
          16 pages, 5 figures, 6 tables

- **What's New**: LLM 워터마킹 분야 발전을 지원하기 위해 MarkLLM이라는 오픈 소스 툴킷(toolkit)이 도입되었습니다. 이 툴킷은 LLM 워터마킹 알고리즘을 구현하고 평가할 수 있는 통합되고 확장 가능한 프레임워크를 제공하며, 사용자 친화적인 인터페이스도 포함되어 있습니다.

- **Technical Details**: MarkLLM은 두 가지 주요 알고리즘 가족인 KGW와 Christ 가족의 아홉 가지 구체적인 워터마킹 알고리즘을 지원합니다. KGW는 LLM에서 생성된 logits을 수정해 워터마킹 하는 방식으로, Christ는 텍스트 생성 과정에서 샘플링을 변경하는 방법을 사용합니다. 또한, MarkLLM은 워터마킹 메커니즘을 시각화할 수 있는 기능도 제공하여 이해도를 높이고자 합니다.

- **Performance Highlights**: MarkLLM은 탐지성(detectability), 강건성(robustness), 텍스트 품질에 미치는 영향 등 세 가지 중요한 관점에서 평가할 수 있는 12개의 도구를 포함하고 있습니다. 또한, 사용자 정의가 가능한 두 가지 자동 평가 파이프라인을 통해 포괄적인 평가를 지원합니다.



### Natural Language Can Help Bridge the Sim2Real Gap (https://arxiv.org/abs/2405.10020)
Comments:
          To appear in RSS 2024

- **What's New**: 이 연구에서는 언어적 설명을 통해 시뮬레이터에서 실제 환경으로의 이미지 기반 로봇 정책(이미지-조건 정책: image-conditioned policy)을 전이하는 새로운 접근 방식을 제안합니다. 이 접근법은 유사한 언어 설명으로 라벨링된 이미지가 유사한 작업 행동을 유도할 수 있음을 발견했습니다. 이를 통해, 시뮬레이터와 실제 환경 간의 시각적 격차를 줄여줍니다.

- **Technical Details**: 제안된 방법은 이미지 인코더(image encoder)를 사전 훈련하여 시뮬레이터 및 실제 이미지를 자연어 설명으로 예측하게 합니다. 이 후, 이 사전 훈련된 이미지 인코더를 기반으로 하여 실제 환경에서 소량의 데모와 시뮬레이터 데이터를 함께 사용하여 IL 정책을 학습합니다. 주요한 기술적 기법에는 LLM(beta)의 사전 훈련된 임베딩(embedging)을 활용하여 시멘틱 유사성을 측정하는 것입니다.

- **Performance Highlights**: Lang4Sim2Real프레임워크는 기존의 sim2real 방법 및 CLIP과 R3M과 같은 비전-언어 사전 훈련 기법을 큰 폭으로 능가했습니다. 몇 번의 실세계 데모로 긴 경로 멀티 스텝 작업을 수행하는 설정에서 성능이 25~40% 향상되었습니다. 특히, 이 연구는 시멘틱한 비전-언어 기반 학습이 sim2real 전이의 샘플 효율성과 성능을 향상시킬 수 있음을 최초로 입증했습니다.



### Zero-Shot Hierarchical Classification on the Common Procurement Vocabulary Taxonomy (https://arxiv.org/abs/2405.09983)
Comments:
          Full-length version of the short paper accepted at COMPSAC 2024

- **What's New**: 유럽 연합이 공공 입찰을 분류하기 위해 개발한 'Common Procurement Vocabulary (CPV)'의 정확성을 높이기 위한 방법을 제안합니다. 특히, CPV 레이블이 없는 공공 입찰 데이터를 분류하여 보다 정밀한 정보를 고객에게 추천할 수 있도록 하는 것을 목표로 합니다.

- **Technical Details**: 이 연구는 사전 학습된 언어 모델인 *bi-directional Pre-trained Language Model (PLM)*을 활용한 *zero-shot* 접근 방법을 제안합니다. 이 모델은 레이블 설명과 레이블 분류 체계를 기반으로 작동하며, 교육 데이터로 SpazioDati s.r.l에서 수집한 이탈리아의 25년간의 공공 계약 데이터를 사용합니다. 구조는 8자리 코드와 제어 코드로 구성된 CPV 레이블을 다루며, 이는 24개의 다른 언어로 표현되어 있습니다.

- **Performance Highlights**: 제안된 모델은 저빈도 클래스 분류에서 세 가지 다른 기준보다 더 좋은 성능을 보였습니다. 특히, 이전에 본 적이 없는 클래스도 예측할 수 있는 능력을 보여줬습니다. 구체적으로, 이 시스템은 데이터베이스의 누락된 값을 채우고 유사한 기록 내 오류를 줄여주는 데 도움을 줍니다.



### "Hunt Takes Hare": Theming Games Through Game-Word Vector Translation (https://arxiv.org/abs/2405.09893)
Comments:
          7 pages, PCG Workshop at FDG 2024

- **What's New**: 이번 연구에서는 게임의 로그 데이터로부터 게임의 역학 구조를 모델링하는 최근 방법인 게임 임베딩(game embeddings)과 언어의 의미 정보를 모델링하는 워드 임베딩(word embeddings)을 연결하는 기술을 제시합니다. 두 가지 접근 방식을 설명하고, 게임 개념을 다른 테마로 번역하는 언어적 번역을 향상시키는 증거를 보여줍니다. 이는 게임의 테마적 요소를 이해하고 조작하는 AI 시스템의 새로운 가능성을 열어주는 초석입니다.

- **Technical Details**: 워드 임베딩은 자연어 처리에서 자주 사용되는 n차원 벡터로, 유사한 의미를 가진 단어들이 벡터 공간에서 가깝게 나타나는 것을 목표로 합니다. 가장 널리 알려진 알고리즘으로는 Word2Vec이 있습니다. 게임 임베딩은 특정 게임의 로그 데이터를 기반으로 벡터 공간을 학습하여 게임의 구조, 동력학 및 전략에 대한 고급 정보를 추출합니다. 이 연구에서는 주어진 체스 데이터와 영어 텍스트 데이터를 사용하여 게임 임베딩과 워드 임베딩 공간을 연결하고, 새로운 테마와 의미를 찾기 위한 변환을 수행합니다.

- **Performance Highlights**: 연구에서는 워드 임베딩 및 게임 임베딩을 결합하여 게임의 테마를 다른 의미 공간으로 번역하는 초기 결과를 제시합니다. 이를 통해 AI가 게임의 동력학을 실세계 아이디어와 연결하는 가능성을 탐색하였습니다. 초기 결과는 유의미한 결과를 보여주었으며, 이는 게임 설계와 튜토리얼화, 콘텐츠 생성 맥락화 등의 다양한 애플리케이션에 유용할 수 있습니다.



### MediSyn: Text-Guided Diffusion Models for Broad Medical 2D and 3D Image Synthesis (https://arxiv.org/abs/2405.09806)
- **What's New**: 최신 연구에서는 MediSyn이라는 텍스트 유도 잠재 확산 모델(latent diffusion models, LDMs)을 통해 다양한 고품질 의료 2D 및 3D 이미지를 생성할 수 있는 방법을 소개합니다. 이 모델은 텍스트 프롬프트에 따라 이미지를 생성하여 의료 데이터의 희소성을 해결하고 개인정보 보호를 보장하는 자원을 생성하는 목표를 가지고 있습니다.

- **Technical Details**: MediSyn은 텍스트 유도 잠재 확산 모델(Instruction-tuned text-guided latent diffusion models)의 쌍으로 구성되어 있습니다. 각 모델은 500만개 이상의 이미지-캡션 쌍과 10만개 이상의 비디오-캡션 쌍으로 훈련되었습니다. 이 연구에서는 Würstchen v2 아키텍처를 사용하여 이미지와 비디오를 처리하며, 이는 기존 모델보다 훈련 시간이 크게 단축됩니다. GPU 훈련 시간을 줄이기 위해 공간 압축(spatial compression)을 활용하고, 세 가지 단계의 모델 파이프라인을 통해 이미지를 인코딩(encodings)하고 텍스트로 조건화된 디노이징을 수행합니다.

- **Performance Highlights**: MediSyn 모델은 다양한 의료 이미지와 비디오 생성에서 뛰어난 성능을 보입니다. 표준 메트릭을 통해 생성된 출력물들의 품질을 평가한 결과, 기존 방법보다 향상된 품질을 나타냈습니다.



### Many-Shot In-Context Learning in Multimodal Foundation Models (https://arxiv.org/abs/2405.09798)
- **What's New**: 최근 연구는 멀티모달 기초 모델(multimodal foundation models)에서 많은 예시를 포함한 in-context learning(ICL) 능력을 평가했습니다. GPT-4o와 Gemini 1.5 Pro를 10개의 다양한 도메인과 이미지 분류 작업에서 벤치마킹한 결과, 예시 수가 많을수록 성능이 크게 개선되는 것을 확인했습니다.

- **Technical Details**: 이 연구에서는 자연 이미지, 의료 이미지, 원격 감지, 분자 이미지 등 여러 도메인에서의 멀티클래스, 멀티라벨, 세분화된 분류 작업을 포함한 10개의 데이터 집합을 사용했습니다. 로컬에 있는 예제를 수십 개에서 최대 2,000개까지 제공하는 실험을 통해 Gemini 1.5 Pro가 로그 선형(log-linear) 성능 향상을 보였으며, GPT-4o는 덜 안정적이었습니다. 또한, 대규모 컨텍스트 윈도우의 활용으로 인해 다수의 쿼리를 한 번의 API 호출로 배치(Batching)하면 성능이 향상되었습니다.

- **Performance Highlights**: Gemini 1.5 Pro는 대부분의 데이터셋에서 높은 데이터 효율성을 보이며, 성능이 계속해서 향상되었습니다. 다수의 쿼리를 배치하는 방식은 비용 절감과 지연 시간을 크게 줄였고, 제로 샷(Zero-shot) 설정에서도 성능이 향상되는 결과를 보였습니다.



### SOK-Bench: A Situated Video Reasoning Benchmark with Aligned Open-World Knowledg (https://arxiv.org/abs/2405.09713)
Comments:
          CVPR

- **What's New**: 이번 연구는 SOK-Bench라는 새로운 벤치마크를 제안합니다. SOK-Bench는 비디오에서 관찰된 상황 지식과 시각적으로 보이지 않는 오픈월드 지식을 통합하여 문제 해결을 요구하는 44K개의 질문과 10K개의 상황을 포함하고 있습니다. 이 새로운 벤치마크는 기존의 비디오 추론 벤치마크들의 한계를 해결하고자 합니다.

- **Technical Details**: SOK-Bench 벤치마크는 자동화되고 확장 가능한 생성 방법을 통해 질문-답변 쌍, 지식 그래프 및 추론 과정을 생성합니다. 이를 위해, LLMs(대형 언어 모델)와 MLLMs(멀티모달 대형 언어 모델)의 상호작용을 사용하여 여러 차례의 대화로 이루어진 학습 과정을 통해 태스크를 생성합니다. 동적, 오픈월드 및 구조화된 맥락 지식을 이해하고 적용하기 위한 질문과 답변을 자동으로 생성하고, 이를 수동으로 검토하여 품질을 보장합니다.

- **Performance Highlights**: 주요 LLMs 및 VideoLLMs를 사용하여 제안된 벤치마크에서 테스트한 결과, 기존 모델들은 여전히 동영상에서의 상황 기반 오픈 지식을 사용하는 추론에 대해 부족한 성과를 보였습니다. 이는 해당 데이터셋의 연구 가치를 입증하는 결과입니다. 우리의 실험은 다양한 설정에서 LLMs 및 MLLMs를 평가하고, 생성 과정에 대한 포괄적인 비교와 분석을 제공하였습니다.



### STAR: A Benchmark for Situated Reasoning in Real-World Videos (https://arxiv.org/abs/2405.09711)
Comments:
          NeurIPS

- **What's New**: 실세계 상황에서의 추론 능력을 평가하기 위한 새로운 벤치마크 'STAR Benchmark'가 소개되었습니다. 이 벤치마크는 실질적인 동영상에서 추출한 상황 추상화 및 논리 기반 질문 응답을 중심으로 이루어져 있습니다. STAR 벤치마크는 사람들의 행동 또는 상호작용과 관련된 네 가지 유형의 질문(상호작용, 순서, 예측, 가능성)을 포함합니다.

- **Technical Details**: STAR 벤치마크는 추출된 기본 엔터티(entities)와 관계를 연결하는 하이퍼 그래프(hyper-graphs)를 통해 실세계 영상의 상황을 나타냅니다. 각 질문의 답변 논리는 상황 하이퍼 그래프에 기반한 기능 프로그램으로 표현됩니다. 진단 신경-상징 모델(NS-SR)을 제안하여 시각 인식, 상황 추상화, 언어 이해 및 기능적 추론을 분리할 수 있습니다.

- **Performance Highlights**: 다양한 기존 비디오 추론 모델을 STAR 벤치마크에서 평가한 결과, 대부분의 모델들이 이 도전적인 과제에서 어려움을 겪고 있다는 것이 밝혀졌습니다. 또한, 제안된 NS-SR 모델은 구조화된 상황 그래프와 동적 단서를 활용하여 상징적 추론을 수행함으로써 주요 도전에 대한 심층 분석을 제공합니다.



### LoRA Learns Less and Forgets Less (https://arxiv.org/abs/2405.09673)
- **What's New**: 이번 연구는 Low-Rank Adaptation (LoRA)와 전체 파인튜닝(Full Finetuning)의 성능을 두 가지 목표 도메인(프로그래밍과 수학)에서 비교합니다. 독특하게도, 연구는 두 가지 데이터 레짐(Instruction Finetuning과 Continued Pretraining)을 고려합니다. 결과적으로 대부분의 설정에서 LoRA가 전체 파인튜닝보다 성능이 떨어지는 것으로 나타났습니다. 그러나 LoRA는 기본 모델의 성능 유지 측면에서 더 우수한 형태의 regularization(정규화)을 보여줍니다.

- **Technical Details**: LoRA는 수억 개의 파라미터를 가진 대형 언어 모델(LLM)을 효율적으로 파인튜닝하는 데 사용되는 방법입니다. 이 방식은 특정 가중치 행렬에 대한 저순위(low-rank) perturbations(교란)을 학습합니다. 본 연구에서는 Llama-2 7B 및 13B 모델을 대상으로 코드와 수학 도메인에서 LoRA와 전체 파인튜닝을 비교했습니다. 연구에 사용된 데이터셋은 Magicoder-Evol-Instruct-110K, MetaMathQA, StarCoder-Python, OpenWebMath입니다.

- **Performance Highlights**: 코드 도메인에서는 LoRA가 전체 파인튜닝보다 현저히 성능이 떨어지지만, 수학 도메인에서는 그 격차가 더 작은 편입니다. LoRA는 기본 모델의 성능을 더 잘 유지하고, 다양한 일반화학습 성취도(learning-forgetting tradeoff)를 보여줍니다. 이는 LoRA가 다른 정규화 기법(예: 드롭아웃, 웨이트 디케이)보다 더 강력한 정규화를 제공하기 때문입니다. 또한, LoRA는 하이퍼파라미터(예: 학습률, 타겟 모듈, 랭크)에 더 민감한 것으로 나타났습니다.



### Unveiling Hallucination in Text, Image, Video, and Audio Foundation Models: A Comprehensive Review (https://arxiv.org/abs/2405.09589)
- **What's New**: 최근 발전된 대규모 기초 모델(FMs)이 텍스트, 이미지, 비디오, 오디오 분야에서 다양한 작업을 처리할 수 있는 놀라운 능력을 보여줬습니다. 그러나 이러한 모델들은 고위험 응용 분야에서 헛소리를 생성하는 경향이 있으며, 이는 실제 환경에서의 널리 사용을 방해하는 주요 요인으로 작용합니다. 종합적인 설문 논문이 최근의 발전을 종합적으로 분석하여 이러한 문제를 식별하고 완화하는 방법을 제시하였습니다.

- **Technical Details**: 헛소리는 FMs이 현실적이지만 맥락이나 사실적 기반이 부족한 콘텐츠를 생성하는 문제를 의미합니다. 텍스트뿐만 아니라 이미지, 비디오, 오디오 영역에서도 이러한 문제가 발생할 수 있습니다. 트레이닝 데이터의 편향, 최신 정보 접근의 제한, 또는 모델의 근본적인 한계들이 헛소리의 원인으로 작용할 수 있습니다. 이를 해결하기 위한 전략으로는 도메인 특화 데이터로 모델을 미세 조정하거나 다양한 트레이닝 데이터를 활용하여 모델의 견고성을 높이는 방법 등이 있습니다.

- **Performance Highlights**: 헛소리 문제를 해결하기 위한 다양한 검출 및 완화 전략이 제안되었습니다. 텍스트, 이미지, 비디오, 오디오 각 영역에서 헛소리 문제를 연구한 결과를 종합적으로 요약하였으며, 최근 발전 상황을 도표와 표를 통해 명확하게 정리하였습니다. 이 논문은 연구자와 개발자에게 중요 자원을 제공하며 견고한 AI 솔루션 개발을 돕는 역할을 합니다.



### OpenLLM-Ro -- Technical Report on Open-source Romanian LLMs (https://arxiv.org/abs/2405.07703)
- **What's New**: 최근 대형 언어 모델(LLMs)은 다양한 작업에서 거의 인간과 유사한 성능을 달성했습니다. 이 문서는 첫 번째 루마니아어 전문 LLM을 훈련하고 평가하는 접근 방식을 제시합니다. 우리의 주요 기여는 오픈소스 Llama 2 모델을 기반으로 첫 번째 기초 루마니아어 LLM(RoLlama)을 출시하고, 번역된 대화 및 지침 세트를 통해 RoLlama2-7b-Instruct 및 RoLlama2-7b-Chat 모델을 구축한 것입니다.

- **Technical Details**: 훈련 데이터는 CulturaX 멀티링구얼 데이터셋과 번역된 대화 및 지침 데이터셋에서 수집되었습니다. 기초 모델 훈련은 CulturaX 데이터셋을 사용하여 계속적인 사전 훈련으로 수행되었습니다. Chat 모델 훈련은 UltraChat 및 OpenAssistant에서 번역된 대화와 지침을 사용하여 이루어졌습니다. 훈련에는 고정된 학습률과 AdamW 옵티마이저가 사용되었으며, 각각의 최대 시퀀스 길이에 따라 적합한 배치 크기를 선택했습니다.

- **Performance Highlights**: 루마니아어 작업에서의 성능 평가를 위해 기존 테스트 스위트를 확장하고 적응시켜 다양한 벤치마크에 대해 실험을 수행했습니다. LLM의 품질을 판단하기 위한 인기 있는 벤치마크를 사용했으며, 그 결과 루마니아어 사용자를 위한 연구와 애플리케이션 개발에 중요한 기여를 하였습니다.




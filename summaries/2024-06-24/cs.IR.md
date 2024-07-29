New uploads on arXiv(cs.CL)

### A SMART Mnemonic Sounds like "Glue Tonic": Mixing LLMs with Student Feedback to Make Mnemonic Learning Stick (https://arxiv.org/abs/2406.15352)
Comments:
          In-Progress Preprint

- **What's New**: 이번 연구에서는 학생들이 새로운 용어를 학습하는 데 도움을 주는 기억술(mnemonics)을 생성하는 모델 SMART를 개발했습니다. 이 모델은 학생들의 피드백을 통해 기억술을 조정하며 특히 학생들이 선호하고 학습에 도움이 되는 기억술을 생성하는 데 중점을 둡니다.

- **Technical Details**: SMART 모델은 MnemonicDictionary에서 학생들이 작성한 고품질의 기억술 데이터를 수집하고 이를 기반으로 LLaMA-2 모델을 파인튜닝하여 초기 모델을 생성했습니다. 이후, Large Language Model(LLM) 정렬을 통해 SMART를 개선했습니다. 학생들이 SMART가 생성한 기억술에 대한 선호도를 플래시카드 앱을 통해 수집하고, 수집된 데이터를 사용하여 Bayesian 모델로 학습 효과를 측정했습니다. 이를 통해 Direct Preference Optimization을 수행하여 SMART를 튜닝했습니다.

- **Performance Highlights**: 2700개 이상의 선호도를 분석한 결과, 학생들이 실제로 학습에 도움이 되는 기억술과 생각에 도움이 된다고 느끼는 기억술이 일치하지 않음을 발견했습니다. Bayesian 모델을 사용하여 이러한 서로 다른 유형의 피드백을 합쳐 효과적인 기억술을 학습할 수 있음을 보여주었습니다. 전문가들의 평가 결과, SMART는 GPT-4와 비슷한 품질의 기억술을 생성하지만, 훨씬 낮은 비용으로 운영될 수 있음을 입증했습니다. 최종적으로, SMART는 학생들의 피드백을 적절히 반영한 모델로서 학습 효과를 극대화했습니다.



### LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs (https://arxiv.org/abs/2406.15319)
Comments:
          Technical Report

- **What's New**: 전통적인 RAG(Retrieval-Augmented Generation) 프레임워크에서는 검색 단위로 일반적으로 짧은 텍스트를 사용합니다. 이에 반해 LongRAG라는 새로운 프레임워크를 제안하여 '롱 리트리버(long retriever)'와 '롱 리더(long reader)'로 구성된 새로운 균형 잡힌 구조를 도입했습니다. LongRAG는 전체 Wikipedia를 4K 토큰 단위로 처리하여 검색 단위를 크게 증가시키며, 이를 통해 검색 단위의 총 개수를 22M에서 700K로 줄였습니다.

- **Technical Details**: LongRAG 프레임워크는 세 가지 주요 설계를 포함합니다. 첫 번째는 롱 검색 단위(Long Retrieval Unit)로, 이는 Wikipedia 문서 전체 또는 여러 관련 문서를 합쳐서 4K 토큰 이상의 긴 검색 단위를 만듭니다. 두 번째는 롱 리트리버(Long Retriever)로, 이는 긴 검색 단위를 통해 주어진 쿼리에 대한 중요한 정보를 검색합니다. 마지막으로, 롱 리더(Long Reader)는 검색된 긴 문맥으로부터 답을 추출합니다. 실험에서는 오프 더 쉘프(off-the-shelf) 리트리버와 리더 모델(Gemini, GPT-4)을 튜닝 없이 사용했습니다.

- **Performance Highlights**: LongRAG는 NQ 데이터셋에서 answer recall@1을 52%에서 71%로, HotpotQA 데이터셋에서 answer recall@2를 47%에서 72%로 향상시켰습니다. 이를 통해 LongRAG는 추가적인 재순위자(re-ranker) 없이 최상위 4-8개 검색 단위만 고려하여 최고의 결과를 달성할 수 있음을 보였습니다. 또한, GPT-4o의 긴 문맥 이해 능력을 활용하여 NQ에서 62%의 EM을, HotpotQA에서 64%의 EM을 달성했습니다. 이 결과는 최상의 세밀 조정된 RAG 모델과 견줄 만한 성능을 보여줍니다.



### NLP-KG: A System for Exploratory Search of Scientific Literature in Natural Language Processing (https://arxiv.org/abs/2406.15294)
Comments:
          Accepted to ACL 2024 System Demonstrations

- **What's New**: NLP-KG는 새로운 연구 문헌 탐색 시스템으로, 사용자들이 잘 모르는 자연어 처리(NLP) 분야를 탐험할 수 있도록 돕습니다. 이 시스템은 기존의 키워드 기반 검색을 넘어서 사용자가 관심 있는 분야에 대한 서베이 논문을 쉽게 찾을 수 있도록 하며, 학습 그래프(Field of Study hierarchy graph)를 통해 관련 분야와 개념을 시각적으로 탐색할 수 있도록 지원합니다. 또한, NLP-KG는 사용자가 자연어로 질문을 하고 학술 출판물에서 검색된 지식을 바탕으로 답변을 얻을 수 있는 대화형 인터페이스를 제공합니다.

- **Technical Details**: NLP-KG는 자연어 처리(NLP) 지식 그래프(NLP-KG)를 핵심으로 하며, 이는 다양한 Fields of Study(FoS), 논문, 저자, 발표 장소를 의미적 관계로 연결합니다. 이를 생성하기 위해, 연구 논문의 제목 및 초록을 사용한 반자동 접근법을 이용해 품질 높은 계층 구조 비순환 그래프를 구축하였습니다. 엔터티와 관계 추출을 위해 최적화된 PL-Marker 모델을 사용하여 대규모 엔터티와 관계 세트를 생성하고, 이를 도메인 전문가들이 검증 및 수정합니다.

- **Performance Highlights**: PL-Marker 모델의 평가 결과, SciBERT 기반 PL-Marker 모델을 사용하여 엔터티와 관계를 추출하는 것이 가장 효과적이었으며, 자동 분류 모델은 높은 F1 스코어를 성취하였습니다. 최종적으로 GPT-4를 사용하여 각 Fields of Study에 대한 짧은 텍스트 설명을 생성하였습니다. 또한, 고급 필터 기능을 통해 논문 검색 결과를 특정 FoS, 발표 장소, 날짜, 인용 횟수, 서베이 논문으로 필터링할 수 있습니다.



### The Greek podcast corpus: Competitive speech models for low-resourced languages with weakly supervised data (https://arxiv.org/abs/2406.15284)
Comments:
          To be presented at Interspeech 2024

- **What's New**: 이 연구에서는 제한된 디지털 표현을 가진 언어에 대한 음성 기술 개발의 어려움을 해결하기 위해 약 800시간의 현대 그리스어 팟캐스트 코퍼스를 컴파일하고, 이를 이용해 Whisper large-v3 모델을 조정하였습니다. 16개의 다채로운 팟캐스트 도메인을 포괄하는 분석을 수행하여 ASR 성능 향상을 평가하였습니다.

- **Technical Details**: 약한 감독(weak supervision)을 통해 데이터를 확대하는 방법을 연구하였습니다. WhisperX 파이프라인을 사용하여 오디오 샘플을 분할하고 전사하였으며, 16개 카테고리에서 약 3124시간의 사전 학습 코퍼스와 800시간의 ASR 코퍼스를 구축하였습니다. Whisper-small과 Whisper-medium 모델을 이를 이용해 미세 조정(fine-tuning)하였습니다.

- **Performance Highlights**: 데이터 양과 모델 크기에 비례하여 WER(Word Error Rate) 개선이 일관되게 나타났습니다. 다중 도메인 및 학습 데이터셋 이외의 표준 코퍼스에서 실험을 통해 약하게 감독된 코퍼스를 사용하는 것이 효과적이라는 것을 확인하였습니다. 훈련된 모델 체크포인트와 코퍼스를 재현하기 위한 스크립트는 공개적으로 제공됩니다.



### Cognitive Map for Language Models: Optimal Planning via Verbally Representing the World Mod (https://arxiv.org/abs/2406.15275)
- **What's New**: 이 논문은 언어 모델이 다단계 시뮬레이션을 요구하는 계획 업무에서의 성능을 향상시키기 위해 인지 지도를 사용하는 방법을 조사합니다. Gridworld 경로 계획(task)에서 인지 지도가 최적 및 도달 가능한 계획 생성 능력을 크게 향상시킨다는 실험 결과를 보고합니다.

- **Technical Details**: 이 연구는 주어진 환경에 대한 인지 지도(cognitive map)를 구축할 수 있는 언어 모델의 최적 계획 능력을 탐구합니다. 인지 지도를 활용하여 언어 모델이 Gridworld 경로 계획(task)에서 인간 인식과 유사한 두 가지 주요 특성을 나타내는 것을 관찰했습니다: 확장된 환경에 대한 계획 능력의 일반화(generalization)와 제한된 훈련 데이터로 빠른 적응(rapid adaptation).

- **Performance Highlights**: 우리 방법이 Gridworld 경로 계획에서 크게 개선된 성능을 보여주었으며, 이는 인간의 인지 과정을 모델링하는 데 중요한 통찰을 제공합니다. 이 결과는 보다 고급적이고 강력한 시스템을 개발하는 데 기여할 수 있습니다.



### Evaluating Diversity in Automatic Poetry Generation (https://arxiv.org/abs/2406.15267)
Comments:
          init version

- **What's New**: 자동 시 생성 자동화(poetry generation)를 위한 연구 개발이 활발히 진행 중입니다. 이번 연구는 기존의 자동 시 생성 시스템에서 Turing Test 차량 여부 대신 다각도로 생성된 시의 다양성을 평가하고 비교하는 최초의 연구로, 시의 구조적, 어휘적, 의미적, 스타일적 차원에서 자동 시 생성 시스템의 다양성을 조사합니다.

- **Technical Details**: 이 연구에서는 다양한 유형의 모델들(단어 수준 vs 문자 수준, 범용 대형 언어 모델(LLMs) vs 시 전용 모델)을 평가하고, 컨디셔닝 방식(조건부 vs 무조건부)에 따른 성능을 분석합니다. 예를 들면 최신 LLaMA3 모델과 특별히 조정된 시 전용 모델을 포함하여, 여러 가지 모델을 비교했습니다. 연구는 생성된 시의 구조적, 어휘적, 의미적, 스타일적 분포를 인간이 작성한 시와 비교하여 다양성을 측정했습니다.

- **Performance Highlights**: 현재의 자동 시 생성 시스템은 여러 차원에서 예상보다 다양성이 부족하다는 결론을 얻었습니다. 자주 운율이 맞지 않거나, 의미적으로 일정하고 인간 시의 길이 분포와 일치하지 않는 경우가 많았습니다. 그러나 스타일 컨디셔닝과 문자 수준 모델링(character-level modeling)이 거의 모든 차원에서 다양성을 증가시킨다는 것이 실험을 통해 발견되었습니다. 이 모델들의 한계를 극복함으로써 더 진정한 다양성을 지닌 미래의 시 생성 모델을 발전시킬 수 있는 가능성을 시사합니다.



### Perception of Phonological Assimilation by Neural Speech Recognition Models (https://arxiv.org/abs/2406.15265)
Comments:
          Accepted for publication in Computational Linguistics (Special Issue on Language Learning, Representation, and Processing in Humans and Machines)

- **What's New**: 이번 연구는 Wav2Vec2 같은 신경 기반 음성 인식 모델이 소리 동화(assimilation)를 어떻게 인식하고 보상하는지, 그리고 그러한 보상을 위해 모델이 사용하는 언어적 지식을 규명하는 데 초점을 맞췄습니다. 연구에서는 심리언어학적 자극을 사용해 모델의 출력에서 나타나는 보상 패턴을 분석하고, 해석 가능성 실험을 통해 이러한 패턴의 메커니즘을 구체적으로 확인했습니다.

- **Technical Details**: 연구는 단계적으로 진행되었습니다. 첫 번째 단계에서는 신경 음성 인식 모델을 심리언어학 자극에 노출시켰습니다. 이후, 해석 가능성 실험(probing experiments)을 통해 모델이 동화된 소리를 어떻게 처리하는지 분석했습니다. 최종 단계에서는 원인적 개입 실험(causal intervention experiments)을 통해 최소한의 음운(context cues) 단서가 모델의 최종 해석에 어떻게 영향을 미치는지 조사했습니다.

- **Performance Highlights**: 연구 결과 Wav2Vec2 모델은 인간과 유사한 방식으로 소리 동화에 대한 보상을 수행하는 것으로 나타났습니다. 특히 모델은 최종 계층에서 동화된 소리를 원래의 형태로 해석하는 것으로 확인되었습니다. 또한 최소한의 음운적 단서(context)만으로도 모델이 동화에 대한 보상을 효율적으로 수행할 수 있음을 발견했습니다.



### Unsupervised Morphological Tree Tokenizer (https://arxiv.org/abs/2406.15245)
- **What's New**: 이 논문에서는 언어 모델링의 핵심 단계인 토크나이제이션(tokenization) 과정에서 기존의 통계적 방법이 단어의 구성 경계를 깨뜨려 의미 정보가 손상되는 문제를 해결하기 위해, 형태 구조(morphological structure)를 고려한 새 방법을 제안했습니다. 이를 통해 Deep model을 사용하여 단어의 문자 수준(character-level)의 구조를 유도하고, MorphOverriding 메커니즘을 통해 형태소가 분해되지 않도록 보장합니다. 또한, 완전히 새로운 토크나이제이션 알고리즘인 TreeTok도 제안합니다.

- **Technical Details**: 이 새로운 방법은 스스로 학습하는 모델(self-supervised objectives)을 훈련시켜 주석이 없는 상태에서도 형태 규칙에 맞는 문자 수준 구조를 유도합니다. 모델은 문장의 내부 구조와 단어의 표현을 공동으로 인코딩하며, 단어의 문자 구조를 유도하기 위해 구성 모델(composition model)을 사용합니다. 구체적으로, 입력 단어가 형태소인지 확인된 경우, 형태소 임베딩(morpheme embedding)을 사용하여 구성 결과를 재정의하거나 덮어씁니다. 이로 인해 BPE나 WordPiece와 달리 단어를 형태소 단위로 덜 나누고 더 자연스럽게 토크나이제이션 할 수 있습니다.

- **Performance Highlights**: TreeTok 알고리즘은 형태소 완전성을 효과적으로 유지하며, BPE 및 WordPiece와 같은 기존 방법보다 형태소 분할 과제(morphological segmentation tasks)와 언어 모델링 과제(language modeling tasks)에서 우수한 성능을 보였습니다. TreeTok는 먼저 트리 기반 BPE 변형을 사용하여 초기 어휘를 구축한 다음, 트리 기반 Unigram 변형을 적용하여 초기 어휘를 지정된 크기로 간추리는 과정을 거칩니다. 이 접근법은 BPE와 달리 병합 작업에서 생성된 중간 토큰을 모두 유지할 필요가 없기 때문에 더 컴팩트한 어휘를 구축할 수 있습니다. 최종적으로, TreeTok는 Wikitext-103 코퍼스를 사용하여 훈련하고 평가한 결과, 모든 과제에서 BPE와 WordPiece를 지속적으로 능가하는 성능을 보였습니다.



### Detecting Synthetic Lyrics with Few-Shot Inferenc (https://arxiv.org/abs/2406.15231)
Comments:
          Under review

- **What's New**: 최근 몇 년간 음악에서 생성된 콘텐츠가 주목받고 있습니다. 특히, 대형 언어 모델(Large Language Models, LLMs)이 다양한 스타일, 테마 및 언어 구조의 인간처럼 보이는 가사를 효과적으로 생성하는 데 사용되고 있습니다. 하지만 이런 기술 발전은 저작권 침해, 소비자 만족도 저하, 콘텐츠 스팸과 같은 문제를 야기하고 있습니다. 이에 대응하여 생성된 가사를 탐지하는 방법이 필요합니다. 이 연구는 고품질의 합성 가사 데이터셋을 처음으로 구축하고 다양한 소수샷(few-shot) 콘텐츠 탐지 접근법을 평가하였습니다.

- **Technical Details**: 연구는 기존 탐지 방법론이 창의적 텍스트, 특히 시나 가사와 같은 영역에서는 효과적이지 않다는 문제를 제기합니다. 이를 해결하기 위해, 연구팀은 첫 번째 고품질 합성 가사 데이터셋을 구축하고, LLM2Vec에 기반한 최선의 소수샷 탐지기를 포함한 7개의 소수샷 콘텐츠 탐지 방법을 평가했습니다. 이를 통해 새로운 아티스트와 모델에 대한 일반화 능력을 테스트하고, 생성 후 대체(paraphrasing) 탐지 성능도 평가했습니다.

- **Performance Highlights**: LLM2Vec 기반의 탐지기는 다른 스타일리틱 및 통계적 방법을 능가하였으며, 새로운 아티스트와 모델에 대한 일반화 능력도 우수한 것으로 나타났습니다. 또한, 생성 후 대체된 내용을 효과적으로 탐지하는 능력도 보여주었습니다. 이 연구는 특히 창의적 콘텐츠 탐지에 대한 추가 연구의 필요성을 강조하며, 모든 데이터셋, 전처리 스크립트 및 코드는 Apache 2.0 라이선스 하에 GitHub와 Hugging Face에서 공개되었습니다.



### A LLM-Based Ranking Method for the Evaluation of Automatic Counter-Narrative Generation (https://arxiv.org/abs/2406.15227)
- **What's New**: 온라인 상에서 잘못된 정보와 유해한 내러티브가 퍼지는 상황에서 이를 효과적으로 대처하기 위해 카운터 내러티브(Counter Narrative, CN) 생성 기술이 필요하다. 이번 논문에서는 인간의 평가와 높은 상관관계를 가지는 새로운 평가 방식을 제안했으며, 특히 대형 언어 모델(LLM)을 평가자로 활용하는 방법론을 개발했다는 점에서 새롭다.

- **Technical Details**: 논문은 생성된 CN을 쌍으로 비교하여 토너먼트 형식으로 모델을 평가하는 시스템을 제안했다. 이를 통해 인간의 선호도와 0.88의 상관관계를 달성했다. 추가적으로, 대형 언어 모델을 제로샷(zero-shot, ZS) 방식으로 활용하여 챗(chat), 인스트럭트(instruct), 베이스(base) 모델 간의 비교 분석을 수행했다. 이 과정에서 도메인 특화 데이터에 대한 모델의 튜닝(fine-tuning) 실험도 포함되었다.

- **Performance Highlights**: 챗 맞춤형(chat-aligned) 모델이 ZS 설정에서 가장 우수한 성능을 발휘했으며, 이는 해당 모델이 보안 문제로 인해 응답을 거부하지 않는다면 유효한 선택임을 보여준다. 코드와 관련 자료는 논문이 출판되면 공개될 예정.



### Unsupervised Extraction of Dialogue Policies from Conversations (https://arxiv.org/abs/2406.15214)
- **What's New**: 이 논문에서는 대규모 언어 모델(Large Language Models, LLMs)과 그래프 기반 알고리즘을 결합하여 대화 정책(dialogue policies)을 자동으로 추출하는 새로운 방법론을 제안합니다. 이는 주어진 대화 데이터를 통합 형식(canonical forms)으로 변환하고, 이를 기반으로 대화 흐름 네트워크(flow network)를 만들어내어, 그래프 탐색 알고리즘을 통해 대화 흐름을 추출하는 방식입니다.

- **Technical Details**: 이 방법론은 먼저 LLM을 사용해 각 대화 턴(turn)을 통합 형식으로 번역합니다. 다음으로, 이러한 통합 형식을 클러스터링하여 작은 변형들을 통합한 후, 대화 데이터를 모델링하는 그래프를 구성합니다. 이 그래프는 대화 턴의 통합 형식이 노드가 되고, 턴 간의 진행이 엣지가 되는 흐름 네트워크입니다. 마지막으로, 이 그래프에 경로 탐색 알고리즘(path-finding algorithms)을 적용하여 대화 정책을 추출합니다.

- **Performance Highlights**: 제안된 하이브리드 방법론은 LLM만을 사용한 정책 생성 방법보다 우수한 성능을 보였습니다. 이 방법론은 정량적 성능뿐만 아니라, 대화 설계자가 더 많은 제어와 해석 가능성을 제공하며, 높은 견고성을 나타냈습니다. 또한, 자동 평가 지표(BLEU, BertScore 등)와 인간 평가 간의 높은 상관관계를 보여주었습니다.



### How Effective is GPT-4 Turbo in Generating School-Level Questions from Textbooks Based on Bloom's Revised Taxonomy? (https://arxiv.org/abs/2406.15211)
Comments:
          Accepted at Learnersourcing: Student-Generated Content @ Scale 2024

- **What's New**: 이번 연구는 GPT-4 Turbo의 능력을 평가하여 NCERT 교과서에서 교육적 질문을 생성하는 능력을 보여줍니다. 특히, Bloom's Revised Taxonomy에 따른 '이해' 수준에서 요구되는 고차원적 사고 기술을 필요로 하는 질문을 생성할 수 있음을 강조합니다. 그러나 인간 채점자와의 일관성을 고려할 때, 일부 차이가 있음을 발견했습니다. 이는 GPT-4 Turbo가 교육적 질문 생성 도구로 유망하지만, 다양한 인지 수준에서의 효능이 다르기 때문에 추가적인 개선이 필요함을 시사합니다.

- **Technical Details**: GPT-4 Turbo를 사용하여 NCERT 교과서의 챕터에서 제로샷(Zero-shot) 모드로 질문을 생성하고 이를 평가했습니다. 연구는 Bloom's Revised Taxonomy의 다양한 단계와 질문의 품질 평가에서 기계와 인간 평가자간의 차이점을 발견했습니다. 실험은 주로 역사, 지리, 경제학, 환경학, 과학 등 다양한 과목과 학년을 대상으로 진행되었습니다. 또한, 질문의 품질을 평가하기 위해 고급 자연어 처리(NLP) 방법을 사용했고, IWF 기준에 따라 평가를 수행했습니다.

- **Performance Highlights**: GPT-4 Turbo는 교육적 질문 생성에 있어 주목할만한 성능을 보여주었으며, 특히 이해 수준에서 인간 채점자와 높은 일관성을 보였습니다. 그러나, Bloom's Revised Taxonomy의 상위 수준으로 갈수록 성능 차이가 나타났습니다. 결과적으로, 인간 교사와 기계의 결합된 검증이 질문의 품질을 더 향상시킬 수 있는 가능성이 확인되었습니다.



### Reward Steering with Evolutionary Heuristics for Decoding-time Alignmen (https://arxiv.org/abs/2406.15193)
- **What's New**: LLMs(대규모 언어 모델)의 응답을 사용자와 이해관계자의 선호도에 맞추려는 필요성이 증가하면서 다양한 선호도 최적화 접근법이 제안되었습니다. 기존의 많은 방법들이 모델 성능을 저하시킬 수 있지만, 우리의 새로운 접근법은 이러한 문제를 진화 알고리즘(진화적인 방식)으로 해결합니다. 이 접근법은 '탐색(exploration)'과 '활용(exploitation)'을 분리하여 적용하며, 실험적으로 높은 성능을 보였습니다.

- **Technical Details**: 기존의 선호도 최적화와 달리, 우리는 인코딩 시간 동안 보상을 주는 모델을 사용하여 선호도를 모색하고 활용합니다. 탐색은 변이된 뮤테이션(mutations) 명령어로 디코딩을 수행하고, 잘못된 보상을 받은 생성물을 주기적으로 교체하여 보상을 최적화합니다. 이러한 전략은 진화 알고리즘과 유사합니다. 우리는 '탐색(exploration)'을 디코딩 단계마다 보상을 체크하지 않고, 주기적으로 체크하는 방식으로 구현했습니다. 또한 초기 명령어에서 변형된 명령어를 생성하여 더 많은 탐색을 장려합니다.

- **Performance Highlights**: 우리의 접근법인 Darwin은 AlpacaEval 2와 MT-Bench 정렬 벤치마크에서 ARGS와 같은 기존 디코딩 시간 정렬 방법보다 뛰어난 성능을 보였습니다. 여러 실험 설정에서 ARGS 및 기타 최적화 방법을 능가했습니다.



### Hybrid Alignment Training for Large Language Models (https://arxiv.org/abs/2406.15178)
Comments:
          accepted by ACL (Findings) 2024

- **What's New**: 이 연구에서는 기존의 두 단계로 나누어 수행되는 대형 언어 모델(LLM)의 정렬 훈련(alignment training)의 문제를 해결하기 위해 하이브리드 정렬 훈련(Hbat) 접근법을 제안합니다. 이 방법은 명령 따르기 정렬(instruction-following alignment)과 인간 선호 정렬(human-preference alignment)을 번갈아 수행하여 두 가지 목표가 더 잘 협력할 수 있도록 합니다.

- **Technical Details**: 하이브리드 정렬 훈련(Hbat) 접근법은 교대 정렬 방법(alternating alignment)과 수정된 탄력적 가중치 통합(elastic weight consolidation) 방법을 기반으로 합니다. 교대 정렬 방법은 다목적 최적화(multi-objective optimization)에서 영감을 받아 인간 선호 정렬을 의사결정자 역할로 하여 명령 따르기 정렬과 지속적으로 상호작용하도록 설계되었습니다. 또한 각 매개변수에 적절한 제약을 동적으로 부과하여 최적화 충돌을 완화하는 수정된 EWC 방법이 도입되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 Hbat 접근법이 요약(summarization) 및 대화(dialogue) 태스크에서 기존의 모든 기준 모델을 크게 능가함을 확인했습니다. 특히, LLaMA2-13B 모델을 기반으로 한 요약 태스크에서 Hbat은 전통적인 강화 학습 기반 인간 피드백 조정(RLHF)에 비해 ROUGE-L 점수가 2.26 포인트 증가했습니다. 대화 태스크에서는 GPT-4의 승률이 21.01 포인트 증가했습니다. 추가적으로 ESRL과 결합했을 때도 추가적인 성능 개선을 보여줬습니다.



### Enhancing Idiomatic Representation in Multiple Languages via an Adaptive Contrastive Triplet Loss (https://arxiv.org/abs/2406.15175)
- **What's New**: 이 논문에서는 비유어구 (idiomatic expressions)를 효과적으로 모델링하기 위한 새로운 접근 방식을 제안합니다. 비유어구는 구성 단어만으로는 그 의미를 해석할 수 없기 때문에 자연어 처리 (NLP)에서 어려움이 있어왔습니다. 저자들은 비유어구 인식 목표를 포함하게끔 언어 모델을 훈련시키기 위해 비대칭 기여도를 고려한 트리플릿 손실 (triplet loss)을 활용한 적응형 대조 학습 (adaptive contrastive learning)과 리샘플링 마이너 (resampling miners)를 제안합니다. 이 방법은 SemEval 챌린지에서 이전 방법들에 비해 많은 지표에서 크게 앞섰습니다.

- **Technical Details**: 논문에서는 비유어구의 의미를 이해하기 위해 비대칭 단어 기여도를 포함한 트리플릿 손실 함수를 사용하여 사전 훈련된 모델을 미세 조정합니다. 또한 인-배치 양성-앵커-음성 트리플릿을 사용하여 비유어구와 비유어구가 아닌 경우를 구별하는 모델을 구축했습니다. 이는 대조 학습 (Contrastive Learning)의 'learn-to-compare' 패러다임을 활용하여 더 나은 텍스트 임베딩을 얻는 방식을 참고했습니다.

- **Performance Highlights**: 제안된 방법은 다양한 비유어구 수준을 포함한 데이터 세트에서 새로운 최첨단 결과를 달성했습니다. 특히 비유어구 전용 성능과 전반적 성능에서 이전 최고 성과와 비교하여 많은 개선을 보였습니다.



### A Syntax-Injected Approach for Faster and More Accurate Sentiment Analysis (https://arxiv.org/abs/2406.15163)
- **What's New**: 이번 논문에서는 의견 분석(사)와 문법 정보 도입을 통한 성능 향상을 다루고 있습니다. 구문 분석 알고리즘의 느린 속도로 인한 계산 병목 현상을 해결하기 위해, 시퀀스 라벨링 구문 분석기(SELSP)를 사용하여 구문을 사에 주입하려고 합니다. SELSP를 통해 전통적 구문 분석기보다 훨씬 빠르게 구문 기반 사를 수행할 수 있습니다. SELSP는 영문과 스페인어로 삼진 극성 분류 작업에서 훈련 및 평가되며, 기존의 Stanza 및 VADER와 같은 휴리스틱 접근법보다 더 나은 정확도와 속도를 보여줍니다.

- **Technical Details**: SELSP는 종속 구문 분석을 시퀀스 라벨링 문제로 처리합니다. 이는 구문 기반 사를 더 빠르게 수행할 수 있게 해줍니다. UD(Universal Dependencies) 형식을 사용하며, 테스트 세트에서의 분석 정확도를 측정한 후, 영어와 스페인어로 극성 예측 작업에서 적용됩니다. 또한 여러 감정 사전을 테스트하여 성능을 비교합니다. SELSP와 Transformer 기반 모델(RoBERTa)와의 비교에서도 SELSP는 훨씬 빠른 속도와 더 나은 극성 예측 성능을 나타냈습니다.

- **Performance Highlights**: SELSP는 상업 및 연구에서 사를 위한 실용적인 도구로, 기존의 Stanza 구문 분석기보다 매우 빠르며, 정확도 또한 뛰어납니다. 감정 사전이 극성 판별 작업의 성능을 향상시키는 데 중요한 역할을 함을 확인했습니다. SELSP는 Transformer 기반 모델보다 훨씬 빠르게 작동하여 실용성이 더 높습니다.



### Assessing Good, Bad and Ugly Arguments Generated by ChatGPT: a New Dataset, its Methodology and Associated Tasks (https://arxiv.org/abs/2406.15130)
- **What's New**: 최근 대규모 언어 모델(Large Language Models, LLMs)의 성공으로 인해 이러한 모델이 생산하는 가짜 정보에 대한 우려가 높아지고 있습니다. 이 논문에서는 ChatGPT와 같은 LLM이 생성한 논증문을 분석하는 새로운 방법론을 소개하고, 다양한 논증 텍스트를 포함하는 새로운 데이터셋 ArGPT를 제안합니다.

- **Technical Details**: ArGPT 데이터셋은 ChatGPT가 생성한 논증문에서 '좋은', '나쁜', '추악한' 논증을 수집하여 구성됩니다. 이 데이터셋은 다음과 같은 여러 작업을 위해 사용됩니다: 논증 품질 분류(Argument Quality Classification), 범위 식별(Span Identification), 요소 분류(Component Classification), 관계 분류(Relation Classification), 에세이 채점(Essay Scoring). 각 단계는 전문가가 직접 주석을 달아 데이터셋의 품질을 확보합니다.

- **Performance Highlights**: 인위적으로 생성된 데이터와 인간이 작성한 논증 데이터 간의 유사성을 강조하면서, 이 데이터셋이 주어진 작업에 학습 및 테스트 도구로서 유용함을 입증합니다. LLM이 생성한 논증 데이터가 인간의 논증 데이터셋과 충분히 유사하다는 점을 입증함으로써, 다량의 데이터를 신속하고 저렴하게 생성할 수 있는 방법을 제공하는 점이 주목할 만합니다.



### On LLMs-Driven Synthetic Data Generation, Curation, and Evaluation: A Survey (https://arxiv.org/abs/2406.15126)
Comments:
          A survey on LLMs-driven synthetic data generation, curation and evaluation

- **What's New**: 최근 대규모 언어 모델(LLMs)의 등장으로 인한 중요한 패러다임 변화가 딥러닝 분야에서 주목받고 있습니다. 이 논문은 기계 학습 모델 훈련 및 평가에 있어 데이터 의존성을 인공지능이 생성한 합성 데이터(synthetic data)로 보완하는 방법을 탐구합니다. 합성 데이터는 실제 데이터의 특성과 패턴을 모방하여 생성되며, 높은 데이터 품질을 유지하는 데 중점을 둡니다. 연구진은 LLMs가 데이터를 생성하는 과정에서 신뢰성과 다양성을 확보하는 방법을 제안합니다.

- **Technical Details**: 이 논문은 합성 데이터 생성의 일반적인 워크플로우(workflow)를 바탕으로 현재의 연구들을 체계적으로 정리하였습니다. 전체 생성 작업은 데이터 생성, 큐레이션(curation), 평가로 나눌 수 있으며, 이러한 과정에서 언급된 일반적인 방법 중 하나는 프롬프트 엔지니어링(prompt engineering)과 다단계 생성(multistep generation)입니다. 연구에서 제안된 방법은 LLMs의 지침 따르기(instruction-following) 능력을 활용하여 데이터 생성의 통제 가능성을 높입니다.

- **Performance Highlights**: 연구진은 현재의 연구에서 LLMs가 높은 정확성과 다양성을 동시에 달성하는 데 어려움을 겪고 있음을 밝혔습니다. 합성 데이터 생성에서 발생할 수 있는 논리적 오류와 일관성 결여 문제를 해결하기 위해 다양한 기술적 트릭을 적용하여 합성 데이터의 품질을 향상시키려는 노력이 반영되었습니다.



### Brain-Like Language Processing via a Shallow Untrained Multihead Attention Network (https://arxiv.org/abs/2406.15109)
Comments:
          Preprint

- **What's New**: 이번 연구에서는 훈련되지 않은 대형 언어 모델(Large Language Models, LLMs)이 인간 뇌 데이터와 놀라운 정렬을 보이는 주요 구조적 요소를 조사하였습니다. 언어선택적 유닛(Language-selective units)을 식별하고, 이를 기반으로 다섯 개의 뇌 녹화 데이터셋에서 LLM의 뇌 정렬을 벤치마킹하였습니다.

- **Technical Details**: 훈련되지 않은 모델의 뇌 정렬을 추정하기 위해서, 먼저 LLM 내의 언어선택적 유닛을 선택하였습니다. Transformer 아키텍처의 중요한 구성 요소를 분리하여, 토큰화 전략(tokenization strategy)과 멀티헤드 어텐션(multihead attention)을 주요 구성 요소로 식별했습니다. 순환(recurrence)을 단순화하면 정렬이 더욱 향상됨을 발견하였습니다.

- **Performance Highlights**: 이 모델은 인간의 언어 신경과학 연구에서 요구하는 조건 하에서도 유사한 반응 프로파일을 보여주었습니다. 또한, 언어 모델링에서도 샘플 효율성과 파라미터 효율성에서 개선을 이루었고, 인간의 읽기 시간을 예측하는 행동 정렬에서 새로운 최고 수준을 달성했습니다.



### A Unified Framework for Input Feature Attribution Analysis (https://arxiv.org/abs/2406.15085)
- **What's New**: 이 논문에서는 머신러닝 모델의 결정 과정을 설명하기 위한 새로운 통합 프레임워크를 제안합니다. 이 프레임워크는 하이라이트 및 상호작용 설명(highlight and interactive explanations) 유형을 네 가지 진단 속성(four diagnostic properties)을 기준으로 직접 비교할 수 있도록 설계되었습니다. 이는 각각의 설명 타입이 어떤 진단 속성에 강점을 가지는지를 확인하기 위한 체계적인 분석을 가능하게 합니다.

- **Technical Details**: 제안된 통합 프레임워크는 네 가지 중요한 진단 속성으로 구성됩니다: 신뢰도 (Faithfulness), 인간 주석과의 일치도 (Agreement with Human Annotations), 모사 가능성 (Simulatability), 및 복잡도 (Complexity)입니다. 이 프레임워크는 토큰 설명(Token Explanations), 토큰 상호작용 설명(Token Interactive Explanations), 및 스판 상호작용 설명(Span Interactive Explanations)의 다양한 속성을 분석하고 비교할 수 있게 합니다.

- **Performance Highlights**: 실험 결과, 토큰 설명(TokenEx)은 모델 예측의 신뢰도 측면에서 가장 뛰어난 성능을 보였습니다. 반면에, 상호작용 설명(SpanIntEx 및 TokenIntEx)은 사용자 에이전트의 모사 가능성 측면에서 더 높은 유용성을 제공합니다. 또한, SpanIntEx와 TokenEx는 이해하기 쉽다는 측면에서 상대적으로 나은 성능을 보였습니다.



### Cross-lingual paraphrase identification (https://arxiv.org/abs/2406.15066)
- **What's New**: 이 연구에서는 다국어 언어 간의 의미 유사성을 측정하는 Paraphrase Identification(PI) 과제를 위해 대조적 방식으로 다국어 Bi-encoder 모델을 훈련하는 방식을 소개합니다. 이 방법은 모델이 생성한 임베딩을 사용해 의미 검색 등 다양한 작업에 활용할 수 있습니다.

- **Technical Details**: 연구팀은 Bi-encoder 모델을 대조 손실(Contrastive Loss)을 사용하여 훈련했습니다. 추가적으로 Additive Margin Softmax Loss를 수정하고, Hard Negatives 샘플링을 적용하여 모델의 성능을 향상시켰습니다. 실험은 PAWS-X 데이터셋을 사용하여 진행되었으며, 임베딩 공간의 품질도 평가했습니다. 이 모델은 109개의 언어를 지원하며, Zero-shot 전송 학습 능력을 유지하기 위해 임베딩 레이어를 고정(freezing)했습니다.

- **Performance Highlights**: 제안된 Bi-encoder 모델은 선택된 데이터셋에서 최첨단 Cross-encoder 모델과 비슷한 성능을 보였으며, 상대적 성능 저하는 7-10%로 미미한 수준입니다. 또한, 다양한 언어 간의 의미를 비교하면서도 문맥을 유지하는 데 우수한 성능을 보였습니다.



### PARIKSHA : A Large-Scale Investigation of Human-LLM Evaluator Agreement on Multilingual and Multi-Cultural Data (https://arxiv.org/abs/2406.15053)
Comments:
          Work in progress

- **What's New**: 이번 연구에서는 다국어 및 다문화 환경에서 인간 및 대형 언어 모델(LLMs)에 대한 평가를 수행했습니다. 10개의 인디언 언어로 구성된 90K 인간 평가와 30K LLM 기반 평가를 통해 각 언어에서 일관되게 높은 성능을 보이는 모델 GPT-4o와 Llama-3 70B를 확인했습니다. 우리는 두 가지 평가 설정(쌍별 비교와 직접 평가)에 대해 리더보드를 작성하고, 인간과 LLM 간의 동의 정도를 분석했습니다.

- **Technical Details**: 다국어 LLM 평가의 어려움은 언어적 다양성이 부족한 벤치마크, LLM 사전 학습 데이터의 벤치마크 오염, 번역된 벤치마크의 현지 문화적 뉘앙스 부족 등 여러 가지 요인 때문입니다. 이번 연구에서는 10개의 인디언 언어를 대상으로 30개 모델을 평가하고 90K 인간 평가와 30K LLM 기반 평가를 수행했습니다. 인간 평가자들은 쌍별 비교와 직접 평가 두 가지 작업을 수행하였으며, 새로운 일반 및 문화적 뉘앙스가 반영된 프롬프트 세트를 사용했습니다.

- **Performance Highlights**: 인간과 LLM 평가의 동의 수준을 분석한 결과, 쌍별 비교에서는 비교적 높은 동의를 보였지만, 특히 벵골어와 오디아어와 같은 언어에서 직접 평가의 동의 수준은 떨어졌습니다. 또한, 인간과 LLM 평가에서 다양한 편향성을 검사한 결과, GPT 기반 평가자에서 자기 편향의 증거를 발견했습니다. 연구 결과, GPT-4o와 Llama-3 70B 모델이 대부분의 인디언 언어에서 우수한 성능을 보였습니다.



### Harnessing Knowledge Retrieval with Large Language Models for Clinical Report Error Correction (https://arxiv.org/abs/2406.15045)
- **What's New**: 이 연구는 임상 방사선 보고서의 오류 수정을 위해 대형 언어 모델(LLMs)과 검색 증강 생성(RAG) 기술을 활용한 접근법을 제안합니다. 이 프레임워크는 내부 및 외부 검색 메커니즘을 사용하여 보고서 및 외부 지식 소스로부터 관련 의료 엔터티 및 관계를 추출합니다. 오류 검출, 위치 지정, 수정으로 작업을 분해하는 3단계 추론 프로세스를 도입하여 시스템의 설명 가능성을 높이고 성능을 향상시킵니다.

- **Technical Details**: 제안된 방법은 오류 검출, 위치 지정, 수정을 포함하는 3단계 접근법을 사용합니다. 내부 검색은 제공된 보고서 내의 일관성 검사를 목적으로, 외부 검색은 충분한 맥락 정보를 제공하지 않는 경우에 추론 과정을 증강합니다. RadGraph라는 도구를 사용해 방사선 보고서로부터 임상 엔터티와 관계를 추출합니다. 이 과정은 데이터셋을 통해 엄격하게 평가되며, 실제 임상 기록에 도메인 전문가의 지도를 받아 오류를 도입합니다.

- **Performance Highlights**: 실험 결과, 내부 및 외부 검색 결합이 여러 최신 LLM들에서 오류 검출, 위치 지정, 수정의 정확도를 크게 향상시키는 것으로 나타났습니다. 기존 베이스라인과의 비교 및 다양한 프레임워크 구성 요소의 아블레이션을 통해 접근법의 성과를 증명하였습니다.



### GiusBERTo: A Legal Language Model for Personal Data De-identification in Italian Court of Auditors Decisions (https://arxiv.org/abs/2406.15032)
Comments:
          14 pages, 4 figures, 6 Tables

- **What's New**: 이번 논문에서는 이탈리아 법률 문서에서 개인 데이터를 익명화하는 데 특화된 최초의 BERT 기반 모델 GiusBERTo를 소개합니다. GiusBERTo는 Court of Auditors의 판결 데이터를 이용해 이름, 날짜, 장소 등의 실체(entity)를 인식하고 이를 익명화하는 기능을 가지고 있습니다. 테스트 세트에서 97%의 토큰 수준 정확도를 달성했습니다.

- **Technical Details**: GiusBERTo는 Masked Language Task로 사전 학습된 후, 법률 문서의 데이터 익명화 작업에 맞추어 세밀하게 튜닝되었습니다. 법률 문서에서 문맥적 관련성을 유지하면서 실체를 익명화하기 위해 Context-aware 접근 방식을 사용합니다. 기존의 규칙 기반 또는 Named Entity Recognition(NER) 방식의 한계를 극복하기 위해, BERT의 문맥적 단어 임베딩(contextual word embeddings)을 활용하여 실체를 분류합니다.

- **Performance Highlights**: 테스트 세트에서 GiusBERTo는 97%의 토큰 수준 정확도를 달성하여 높은 성능을 자랑합니다. 이 모델은 개인 정보 보호와 데이터 보호를 균형 있게 유지하면서 이탈리아 법률 커뮤니티에 정확하고 맞춤형 BERT 모델을 제공합니다.



### MedOdyssey: A Medical Domain Benchmark for Long Context Evaluation Up to 200K Tokens (https://arxiv.org/abs/2406.15019)
- **What's New**: 최근 여러 고급 대형 언어 모델(LLMs)들이 컨텍스트 길이를 최대 128K에서 200K까지 지원하는 능력을 갖추고 있습니다. 일반 도메인에서의 긴 컨텍스트 성능 평가에 대한 연구는 활발히 진행되고 있지만, 의료 도메인에서는 이러한 평가가 드물었습니다. 이번 논문에서는 MedOdyssey라는 첫 번째 의료 도메인 긴 컨텍스트 벤치마크를 제안합니다.

- **Technical Details**: MedOdyssey는 4K에서 200K 토큰의 7가지 길이 수준으로 구성되며, 크게 두 가지 주요 요소로 구성됩니다: 의료 컨텍스트 '건초 더미에서의 바늘 찾기' 과제와 의료 관련 일련의 과제들로 총 10개의 데이터셋을 포함하고 있습니다. 첫 번째 요소는 LLM의 지식 누출과 데이터 오염을 방지하기 위해 반직관적 추론과 새로운 사실 주입과 같은 도전 과제를 포함합니다. 두 번째 요소는 전문 의료 지식이 필요한 과제들로 구성됩니다. 공정성을 보장하기 위해 '최대 동일 컨텍스트' 원칙을 설계하였습니다.

- **Performance Highlights**: 실험 결과, 최신 GPT-4o 모델도 단순한 NIAH 실험에서만 우수한 성능을 보였으며, 전반적으로 모든 LLM들이 의료 긴 컨텍스트 처리에 어려움을 겪고 있음을 확인했습니다. 종합적인 성능 분석을 통하여 LLM의 개선 방향과 추가 연구 필요성을 강조합니다.



### Unveiling the Impact of Multi-Modal Interactions on User Engagement: A Comprehensive Evaluation in AI-driven Conversations (https://arxiv.org/abs/2406.15000)
- **What's New**: 이번 연구는 대규모 언어 모델(LLMs: Large Language Models)이 사용자와 챗봇 간의 상호작용을 크게 향상시켰지만, 텍스트 전용 모달리티만으로는 사용자의 참여도를 완전히 끌어올리지 못할 수 있다는 점을 지적합니다. 따라서, 이 논문은 텍스트에 더해 이미지와 오디오가 포함된 멀티모달 상호작용이 사용자 참여에 미치는 영향을 분석합니다.

- **Technical Details**: 이 연구는 다양한 챗봇과 실제 사용자 상호작용 데이터를 사용하여 멀티모달 상호작용의 효과를 평가합니다. 총 146,179명의 다양한 연령, 성별 및 국적의 사용자 데이터를 수집하였으며, 198개의 봇 캐릭터와 747,350개의 대화를 분석했습니다. 텍스트, 이미지, 오디오의 개별 및 조합된 멀티모달 요소가 포함된 상호작용을 통한 사용자 참여도를 측정하기 위해 유지율(retention rate), 대화 길이(conversation length) 등의 객관적인 지표를 사용했습니다.

- **Performance Highlights**: 멀티모달 상호작용은 텍스트 전용 대화보다 사용자 참여를 눈에 띄게 향상시켰습니다. 특히, 이미지와 오디오를 함께 사용할 경우 사용자 참여도가 크게 증가하는 것으로 나타났습니다. 멀티모달 기능을 활용한 대화는 텍스트 전용 대화보다 강화된 사용자 참여를 보여주었고, 이는 사용자 인지처리(cognitive processing)를 최적화하고 풍부한 정보 이해를 가능하게 했습니다. 구체적으로 멀티모달 상호작용은 유지율에서 0.139점, 대화 길이에서 28.97점, 사용자 발화 길이에서 13.16점을 기록하며, 텍스트 전용 대화의 0.105점, 15.77점, 13.01점을 크게 초과했습니다.



### SpreadsheetBench: Towards Challenging Real World Spreadsheet Manipulation (https://arxiv.org/abs/2406.14991)
Comments:
          Homepage: this https URL

- **What's New**: 새로운 시트 조작 (spreadsheet manipulation) 벤치마크인 SpreadsheetBench가 소개되었습니다. 이 벤치마크는 실제 엑셀 포럼에서 수집한 912개의 질문과 관련된 다양한 데이터가 포함되어 있으며, 사용자가 마주하는 복잡한 문제를 반영하는 것이 특징입니다.

- **Technical Details**: SpreadsheetBench는 2,729개의 테스트 케이스를 포함하고 있으며, 평균적으로 각 지시사항마다 세 개의 테스트 케이스가 제공됩니다. 주요 작업 유형으로는 찾기, 추출, 합계 계산, 강조 표시, 삭제, 수정, 계산 등이 있습니다. 특히, 다중 테이블, 비표준 관계형 테이블, 그리고 비텍스트 요소가 풍부하게 포함된 실제 데이터를 기반으로 합니다. 평가 메트릭 또한 온라인 저지 플랫폼의 방식을 차용하여, 동일한 지시사항에 대해 다양한 값이 포함된 스프레드시트를 테스트 케이스로 사용합니다.

- **Performance Highlights**: 대부분의 최신 대형 언어 모델(LLM)이 단일 라운드 및 다중 라운드 추론 설정에서 사람의 성능과 상당한 차이를 보였습니다. 이는 SpreadsheetBench의 높은 난이도를 반영합니다. 또한, 기존 벤치마크와 비교했을 때, SpreadsheetBench는 가장 긴 지시사항 단어 수(평균 85.7 단어)와 가장 많은 스프레드시트 파일(2,729개)을 제공하여 그 복잡성을 더욱 증가시킵니다.



### Retrieve-Plan-Generation: An Iterative Planning and Answering Framework for Knowledge-Intensive LLM Generation (https://arxiv.org/abs/2406.14979)
- **What's New**: Retrieve-Plan-Generation (RPG)이라는 새로운 프레임워크가 제안되었습니다. 이 프레임워크는 계획 단계와 답변 단계를 반복적으로 수행하여, LLM이 더욱 관련성 높은 내용을 생성할 수 있도록 합니다.

- **Technical Details**: RAG 시스템이 가진 단점을 해결하기 위해, RPG는 초기에 plan tokens을 생성하고 이를 기반으로 답변 단계를 수행합니다. 또한, 멀티태스크 프롬프트 튜닝(multi-task prompt-tuning) 방법을 사용하여 기존 LLM이 계획과 답변 단계 모두를 처리할 수 있도록 합니다. 이를 위해, 데이터셋을 기반으로 plan 생성과 세부 문단 활용에 대한 감독 데이터를 ChatGPT로 생성하고, end-to-end로 모델을 학습시킵니다.

- **Performance Highlights**: 5개의 지식 집약적 생성 작업에서 RPG는 기존 RAG 접근법과 지시 튜닝된(instruction-tuned) LLM보다 뛰어난 성능을 보였습니다. 장문, 멀티홉, 단문 생성 작업 모두에서 탁월한 성과를 시연했습니다.



### A Tale of Trust and Accuracy: Base vs. Instruct LLMs in RAG Systems (https://arxiv.org/abs/2406.14972)
- **What's New**: 이번 연구에서는 Retrieval Augmented Generation (RAG) 시스템에서 'instructed'로 미세 조정된 대형 언어 모델(Large Language Models, LLMs)이 아닌, 'base' 모델이 평균적으로 20% 더 좋은 성능을 보인다는 결과를 밝혔습니다. 이는 RAG 작업에서 'instructed' LLMs의 우수성이 당연하다는 기존의 가정을 재고하게 합니다.

- **Technical Details**: RAG는 데이터 검색과 생성 과정이 결합된 혁신적인 AI 접근 방식입니다. 기존의 'instructed' 모델은 감독 학습과 인간 피드백을 통해 지시를 따르는 능력과 인간의 선호도에 맞춰 조정되어 있지만, 본 연구에서는 'base' 모델이 이러한 추가 조정 없이 더 우수한 성능을 보였습니다. 'base' 모델은 주로 다음 토큰 예측(next token prediction) 작업을 통해 사전 훈련되며, 이는 광범위한 텍스트 데이터를 처리하여 언어, 구문, 의미론 및 일반 지식을 습득하게 합니다. 이와 비교하여 'instructed' 모델은 감독 학습(fine-tuning) 및 인간 피드백을 통해 '지시'를 따르는 능력을 배양합니다.

- **Performance Highlights**: 본 연구의 실험 설정에서 'base' 모델이 RAG 작업에서 'instructed' 모델을 평균 20% 뛰어넘는다는 결과가 나왔습니다. 추가적인 분석을 통해 이 차이가 단순한 것 이상의 복잡성을 띄고 있음을 발견했으며, RAG의 방법론 및 평가 절차에 대한 재검토와 더 넓은 논의가 필요함을 시사합니다.



### Domain Adaptation of Llama3-70B-Instruct through Continual Pre-Training and Model Merging: A Comprehensive Evaluation (https://arxiv.org/abs/2406.14971)
Comments:
          8 pages, 6 figures

- **What's New**: Meta-Llama-3-70B-Instruct 모델을 SEC 데이터를 기반으로 도메인 적응 실험을 수행했습니다. 여기에는 지속적인 사전 훈련(CPT)과 모델 병합 기술을 사용하여 도메인 특정 기능을 향상시키고 파국적인 망각을 줄이는 목표가 포함되었습니다. 결과적으로 금융 규제 데이터를 통합하여 강력한 언어 모델을 만들고, 모델 병합 기술의 효과를 평가했습니다. 모델은 Hugging Face에서 'arcee-ai/Llama-3-SEC-Base'로 제공됩니다.

- **Technical Details**: 도메인 적응은 LLM(대규모 언어 모델)을 특정 분야에 맞추는 능력을 향상시키기 위해 지속적인 사전 훈련(CPT)과 모델 병합 기술을 활용합니다. SEC 데이터는 금융 규제 정보를 포함하며, 공시된 기업의 재무 상태 및 운영 성과 분석에 중요합니다. 이 연구에서는 Meta-Llama-3-70B-Instruct 모델을 기반으로 하여 SEC 데이터를 통해 모델의 금융 도메인 이해 능력을 향상시켰습니다. Megatron-Core 프레임워크를 사용하여 700억 개의 파라미터를 가진 모델을 효율적으로 훈련했습니다.

- **Performance Highlights**: 도메인 특정 성능 평가 결과, 금융 분류 및 수치 추론과 같은 도메인 특정 태스크에서 상당한 성능 향상을 보였습니다. 또한, TIES 병합 기술을 사용해 원래 지시 모델의 일반적 능력을 유지하면서 도메인 특정 지식을 강화하여 파국적 망각을 효과적으로 완화했습니다. 훈련 과정 동안 200억 개의 토큰을 처리한 중간 체크포인트가 제공됩니다.



### ICLEval: Evaluating In-Context Learning Ability of Large Language Models (https://arxiv.org/abs/2406.14955)
- **What's New**: 이번 연구에서는 대형 언어 모델 (Large Language Models, LLMs)의 In-Context Learning(ICL) 능력을 평가하기 위한 ICLEval 벤치마크를 소개합니다. 기존의 평가 프레임워크들은 주로 언어 능력과 지식에 중점을 뒀으나, ICL 능력 평가를 간과하는 경우가 많았습니다. ICLEval 벤치마크는 두 가지 핵심 하위 능력인 '정확한 복사'(exact copying)와 '규칙 학습'(rule learning)을 포함하고 있습니다. 실험 결과, ICL 능력은 다양한 LLM 모델에서 보편적으로 나타났으며, 모델 크기만이 ICL 효율성의 유일한 결정 요인이 아님을 알 수 있었습니다. 특히, ICL 능력은 전이 학습 과정 초기에 발현되어 이후 안정화되는 경향이 있었습니다.

- **Technical Details**: ICLEval 벤치마크는 두 가지 핵심 하위 능력을 평가하는 과제를 제공합니다. 정확한 복사 능력(exact copying)은 모델이 동일한 접두사를 매칭하고 후속 콘텐츠를 복사하는 능력을 의미합니다. 규칙 학습 능력(rule learning)은 예제들로부터 규칙을 학습하고 유사한 콘텐츠를 생성하는 능력을 의미합니다. 이를 위해 ICLEval은 비구조적(unstructured) 및 구조적(context) 시나리오에서 정확한 복사 능력을 평가하고, 형식(format), 순서(order), 통계(statistics) 등 다양한 시나리오의 규칙 학습 능력을 평가하는 과제를 포함합니다.

- **Performance Highlights**: 실험에서는 모델 크기가 ICL 능력에 미치는 영향을 탐구하기 위해 1.1억(1.1B)에서 650억(65B) 파라미터 사이의 여러 LLM을 테스트했습니다. 큰 모델일수록 ICL 능력이 더 강한 경향을 보였으나, 일부 작은 모델도 큰 모델과 비교할 만한 성능을 보였습니다. 또한, 전이 학습 초기 단계에서 대부분의 ICL 능력이 정점에 도달하며, 이후 훈련 동안에는 최소한의 성장이 나타났습니다.



### ESC-Eval: Evaluating Emotion Support Conversations in Large Language Models (https://arxiv.org/abs/2406.14952)
Comments:
          Pre-print

- **What's New**: 현재 대형 언어 모델(LLMs)을 활용한 감정 지원 대화(ESC) 모델들이 주목받고 있으나, 그 평가 방법이 아직 확실하지 않습니다. 이를 개선하기 위해, 연구진은 역할 놀이 에이전트를 사용하는 ESC 평가 프레임워크(ESC-Eval)를 제안했습니다. 이 프레임워크에서는 2,801개의 역할 카드를 재구성하고, 특정 역할 놀이 모델(ESC-Role)을 훈련하여 ESC 모델과 상호작용합니다.

- **Technical Details**: ESC-Eval은 역할 놀이 에이전트와 다양한 역할 카드를 사용하여 ESC 모델과 다중 턴 대화를 수행한 후 수동으로 대화를 평가합니다. 역할 카드는 7개의 기존 데이터셋에서 재구성되었으며, ESC-Role은 3.5K ESC 역할 놀이 데이터를 바탕으로 훈련되었습니다. 이 데이터는 ESConv, ExTES, Smile 등의 데이터셋에서 추출되었습니다. 14개의 LLM을 이용해 실험을 진행하고 인간 평가를 통해 결과를 얻었습니다.

- **Performance Highlights**: 평가 결과, ESC 지향 LLM은 일반 AI-보조 LLM보다 우수한 성능을 보였으나, 여전히 인간 성능에는 미치지 못했습니다. ESC-RANK를 개발하여 자동화된 스코어링 프로세스를 구현하였으며, 이는 GPT-4 대비 35포인트 높은 성능을 보였습니다.



### Towards Retrieval Augmented Generation over Large Video Libraries (https://arxiv.org/abs/2406.14938)
Comments:
          Accepted in IEEE HSI 2024

- **What's New**: 본 논문에서는 대형 비디오 라이브러리에서 새로운 비디오를 만드는 작업을 돕기 위한 Video Library Question Answering (VLQA)라는 새로운 과제를 소개합니다. 이를 위해 Retrieval Augmented Generation (RAG) 기법을 적용한 상호 운용 가능한 아키텍처를 제안합니다.

- **Technical Details**: 제안된 시스템은 대형 언어 모델 (LLMs)을 사용하여 검색 쿼리를 생성하고, 음성 및 시각 메타데이터로 인덱싱된 관련 비디오 순간을 검색합니다. 그 후 응답 생성 모듈이 사용자 쿼리와 이 메타데이터를 통합하여 특정 비디오 타임스탬프와 함께 응답을 생성합니다.

- **Performance Highlights**: 이 접근 방식은 멀티미디어 콘텐츠 검색 및 AI 지원 비디오 콘텐츠 생성 분야에서 유망한 성과를 보여줍니다.



### Talking the Talk Does Not Entail Walking the Walk: On the Limits of Large Language Models in Lexical Entailment Recognition (https://arxiv.org/abs/2406.14894)
- **What's New**: 본 연구는 여덟 개의 대형 언어 모델(Large Language Models, LLMs)이 동사 간의 어휘 포함 관계(lexical entailment relations)를 인식하는 능력을 조사했습니다. 특히, 두 개의 어휘 데이터베이스 WordNet과 HyperLex에서 동사 쌍을 통해 제로샷(zero-shot) 및 몇샷(few-shot) 설정을 통해 다양한 프롬프트 전략(prompting strategies)을 검토하였습니다.

- **Technical Details**: 연구에서는 LLMs가 어휘 포함 인식 작업에서 적당히 좋은 성능을 보이지만, 효과성 정도와 조건에 따라 성능이 다양하다는 것을 밝혔습니다. 또한 몇샷 프롬프팅(few-shot prompting)을 활용하면 모델의 성능을 향상시킬 수 있음을 발견했습니다.

- **Performance Highlights**: 모든 연구된 LLMs가 완벽하게 과제를 해결하지는 못했고, 이는 향후 연구 개발이 더 필요함을 시사합니다.



### Generate-then-Ground in Retrieval-Augmented Generation for Multi-hop Question Answering (https://arxiv.org/abs/2406.14891)
Comments:
          ACL 2024 (main conference)

- **What's New**: 본 논문에서는 다중 단계 질문 답변(MHQA) 작업에서 발생하는 문제를 해결하기 위해 새로운 '생성-이후에-기반'(Generate-then-Ground, GenGround) 프레임워크를 제안합니다. 기존의 '검색-이후에-읽기'(Retrieve-then-Read) 패러다임과 달리, 이 방법은 대형 언어 모델(LLMs)의 내재된 지식과 외부 문서를 결합하여 다중 단계 질문을 해결합니다.

- **Technical Details**: GenGround는 두 가지 주요 단계를 반복하여 최종 답변을 도출합니다: (1) 더 단순한 단일 단계 질문을 생성하여 직접 답변을 생성, (2) 해당 질문-답변 쌍을 검색된 문서에 기반하여 교정. 이 과정에서 LLM의 지식을 사용하여 잘못된 예측을 수정합니다. 또한, 소규모 모델에도 이 방식을 적용할 수 있도록 지시 기반 증류 방법을 제안합니다.

- **Performance Highlights**: 네 개의 데이터셋에서 광범위한 실험을 통해 제안된 방법의 우수성을 입증했습니다. GenGround는 기존의 강력한 베이스라인보다 뛰어난 성능을 보여주었으며, 지시 기반 증류 방법을 통해 소규모 모델에서도 강력한 성능을 발휘하도록 했습니다.



### InterBiasing: Boost Unseen Word Recognition through Biasing Intermediate Predictions (https://arxiv.org/abs/2406.14890)
Comments:
          Accepted to Interspeech 2024

- **What's New**: 새로운 연구는 자체 조건화된 CTC(Self-conditioned CTC)를 기반으로 한 적응 매개변수 없는 접근 방식을 제안하여 미리 인식되지 않은 키워드 및 고유 명사의 인식 정확도를 개선합니다. Text-to-Speech (TTS) 및 인식 모델을 사용하여 키워드 목록에 대한 올바른 레이블과 인식 오류 인스턴스의 쌍을 생성하고, 이를 사용하여 중간 예측 오류를 올바른 레이블로 대체합니다.

- **Technical Details**: 연구는 자체 조건화된 CTC 모델을 활용하여 중간 CTC 예측을 올바른 레이블로 대체하고 이후 레이어에 전달합니다. 정확하게 대체된 레이블로 레이어를 조건화하여 목표 키워드를 음향적으로 평가할 수 있게 합니다. 이 접근 방식은 추가적인 모델 재훈련 또는 디코딩 그래프 생성이 필요 없으며, 키워드 목록의 크기에 관계없이 대상 단어에 편향을 제공할 수 있습니다. 중간 레벨 예측을 개선하기 위해 TTS로 키워드에 대한 음성을 생성하고 인식 모델에 입력하여 올바른 레이블과 인식 오류 쌍을 만듭니다.

- **Performance Highlights**: 일본어 실험에서, 제안된 방법은 알려지지 않은 단어에 대한 인식 정확도를 의미 있게 향상시켜 F1 스코어를 높이는 데 성공했습니다. 이는 자체 조건화된 CTC의 프레임워크를 활용하여 예측 결과를 반복적으로 정제함으로써 가능했습니다.



### InternLM-Law: An Open Source Chinese Legal Large Language Mod (https://arxiv.org/abs/2406.14887)
Comments:
          Our dataset, code and models will be released at this https URL

- **What's New**: 이 논문에서는 중국 법률 관련 다양한 법률 질문에 답변할 수 있도록 특화된 대형 언어 모델(Large Language Model, LLM)인 InternLM-Law를 소개합니다. 이 모델은 중국 법률 도메인에서 100만 개 이상의 쿼리를 포함하는 데이터셋을 구축하고, 이를 통해 모델을 교육하여 복잡한 실제 법률 상황을 분석할 수 있도록 합니다. 또한, InternLM-Law는 LawBench 벤치마크에서 최고 성능을 기록하며, GPT-4를 포함한 최신 모델들을 능가합니다. 본 모델과 데이터셋은 향후 법률 도메인에서의 LLM 적용 연구를 지원하기 위해 공개됩니다.

- **Technical Details**: InternLM-Law는 두 단계의 교육 방법을 통해 훈련됩니다. 첫 번째 단계에서는 법률 및 일반 도메인 작업에 모델을 훈련하여 중국 법률 도메인에 대한 이해를 풍부하게 합니다. 두 번째 단계에서는 고품질 법률 데이터에 추가적인 훈련을 실시하여 법률 질의 응답의 정확성과 응답 스타일을 향상시킵니다. 모델의 교육에는 64개의 A100-80GB GPU를 사용하여 8시간 동안 진행되었으며, 학습 시퀀스 길이는 32,000으로 설정되어 대부분의 법률 텍스트를 처리할 수 있습니다.

- **Performance Highlights**: InternLM-Law는 LawBench 벤치마크에서 20개의 서브 태스크 중 13개에서 GPT-4 등을 능가하며 최고의 평균 성능을 달성했습니다. 주관적인 평가와 장문 평가를 포함한 추가 평가 메트릭스를 통해 모델의 성능을 더욱 풍부하게 평가합니다.



### FlowBench: Revisiting and Benchmarking Workflow-Guided Planning for LLM-based Agents (https://arxiv.org/abs/2406.14884)
- **What's New**: LLM 기반 에이전트가 복잡한 작업을 수행하는 도구로 떠오르는 가운데, 특정 지식이 부족할 때 계획적 환각이 발생하는 문제를 해결하고자 외부 작업 흐름 관련 지식을 포함시키는 시도가 진행되었습니다. 이를 더욱 체계화하고자 FlowBench라는 첫 번째 워크플로우 가이드를 위한 벤치마크가 제시되었습니다. FlowBench는 6개의 도메인에서 51개의 다양한 시나리오를 포함합니다.

- **Technical Details**: FlowBench는 다양한 형식으로 제공되는 워크플로우 지식을 공식화하고, 이를 평가하는 다중 평가 프레임워크를 설계하였습니다. 평가 메커니즘은 정적 한 단계 평가와 동적 세션 수준 평가로 구성되어 있으며, 다양한 수준의 워크플로우 지식 형식을 포함한 LLM을 평가하였습니다. 주로 텍스트, 코드, 플로우차트(Flowchart) 형식을 다룹니다.

- **Performance Highlights**: 최신 고성능 LLM인 GPT-4조차도 일부 작업에서 낮은 성공률을 보였습니다. 특히, 플로우차트(Flowchart) 형식이 다른 형식들에 비해 성능, 적응성 및 사용자 친화성 측면에서 균형 있는 결과를 보여줬습니다. 이러한 결과는 향후 에이전트 계획 연구의 방향성을 제시합니다.



### OATH-Frames: Characterizing Online Attitudes Towards Homelessness with LLM Assistants (https://arxiv.org/abs/2406.14883)
Comments:
          Project website: this https URL

- **What's New**: 이번 논문에서는 미국의 노숙자 문제에 대한 온라인 인식을 분석하기 위해 대규모 언어 모델(Large Language Models, LLM)이 어떻게 사회복지 전문가를 도울 수 있는지를 연구합니다. '온라인 노숙자 인식(OATH) 프레임'이라는 새로운 프레임워크를 소개하며, 이를 통해 수백만 개의 트위터 게시글을 분석합니다. 이 프레임워크는 비판, 반응, 인식을 포착하는 9개의 계층적 프레임을 포함하며, 다양한 언어 모델의 도움을 받아 주석 작업의 속도를 급격히 향상시켰습니다.

- **Technical Details**: 주요 기술적 상세로는 OATH-프레임을 생성하기 위해 프레이밍 이론(Framing theory)과 근거 이론(Grounded theory)을 사용한 것입니다. 2021년에서 2023년까지의 3.1M 트위터 게시글을 분석 대상으로 삼았으며, 비전문가와 전문가의 주석 데이터를 조합하여 다중라벨 분류 모델을 통해 2.4M 개의 예측 주석을 생성했습니다.

- **Performance Highlights**: LLM의 도움으로 주석 작업 속도가 6.5배 빨라졌으며, 정확도는 전문가 대비 3 포인트 F1 감소에 그쳤습니다. OATH-프레임을 적용한 대규모 분석 결과, 주요 사회적 사건에 따라 주와 기간에 따른 태도 변화가 드러났습니다. 특히, 기존의 감정 및 유해성 분류기보다 더 정확히 PEH(People Experiencing Homelessness)에 대한 유해 언어를 포착할 수 있음을 강조합니다.



### 70B-parameter large language models in Japanese medical question-answering (https://arxiv.org/abs/2406.14882)
Comments:
          7 pages, 2 figures, 4 Tables

- **What's New**: 의료 분야에 대해 일본어 대형언어모델(LLMs)를 최초로 활용하여 일본어 의료 질문-답변 데이터셋을 사용한 instruction tuning(명령 조정)을 통해 일본어 의료 시험에서 50% 이상의 정확도를 달성한 모델을 개발했습니다.

- **Technical Details**: 본 연구는 여러 70B-parameter LLMs를 이용하여 일본어 의료 질문-답변 성능을 향상시키는 데 중점을 두었으며, 이를 위해 Llama 2를 기본 모델로 다양한 변형 모델을 활용했습니다. QLoRA(양자화 및 저랭크 적응 기법)을 적용해 모델 성능을 세밀하게 조정했습니다.

- **Performance Highlights**: Swallow-70b 모델이 Xwin 모델보다 더 나은 성능을 보였고, 이는 모델의 지속적인 사전학습(continual pretraining)과 일본어 토크나이저(tokenizer) 최적화 덕분임을 확인했습니다. 또한, 프롬프트 형식 차이로 인해 최대 8%의 정확도 차이가 발생할 수 있음을 발견했습니다.



### Sports Intelligence: Assessing the Sports Understanding Capabilities of Language Models through Question Answering from Text to Video (https://arxiv.org/abs/2406.14877)
- **What's New**: 이 논문에서는 스포츠 이해를 위한 새로운 벤치마크를 제안하고, 다양한 스포츠 작업에 대해 주류 대형 언어 모델(Large Language Models, LLMs)을 평가했습니다. 특히 기존 벤치마크의 한계를 극복하고자 체계적인 오류 분석을 통해 향후 연구 우선순위를 제시합니다.

- **Technical Details**: 새로운 벤치마크 'Sport Intelligence'를 도입했으며, LLMs와 비디오 언어 모델(Video Language Models, VLMs)을 평가했습니다. 주요 평가 모델로는 Llama3, GPT-4 시리즈, Gemini 1.5 시리즈, Claude 등이 있습니다. 기존 스포츠 관련 질문 응답(QA) 데이터셋을 통합하여 텍스트와 비디오 기반 QA의 복잡한 수준을 세분화했습니다.

- **Performance Highlights**: 최신 LLMs와 VLMs를 종합 평가한 결과, 복잡하고 다중 모달 시나리오에서 스포츠 추론의 한계를 확인했습니다. 또한, LLM의 스포츠 분야 추론 능력 향상을 위해 도메인 특화 교육 방법의 필요성을 강조했습니다.



### Direct Multi-Turn Preference Optimization for Language Agents (https://arxiv.org/abs/2406.14868)
- **What's New**: 논문에서는 Multi-turn Agent Tasks(다중 턴 에이전트 작업)에서의 Direct Preference Optimization (DPO)의 문제점을 해결하고 새로운 손실 함수인 DMPO를 제안합니다. 이 새로 개발된 손실 함수는 정책 제약 조건을 상태-행동 점유 측정 제약 사항으로 대체하고 Bradley-Terry 모델에 길이 정규화(length normalization)를 추가한 것입니다.

- **Technical Details**: 기존 DPO는 다중 턴 작업에 적용할 때 분할 함수(partition function)를 취소할 수 없다는 문제가 있었습니다. 이를 해결하기 위해 분할 함수를 현재 상태와 독립적으로 만들고 선호 경로(preferred trajectories)와 비선호 경로(dis-preferred trajectories) 사이의 길이 차이를 조정하는 방법을 제시합니다. 새롭게 제안된 DMPO에서는 정책 제약 조건을 상태-행동 점유 측정(state-action occupancy measure) 제약 사항으로 대체하고, 길이 차이를 Bradley-Terry 모델에 반영하여 새로운 손실 함수를 도출하였습니다.

- **Performance Highlights**: 세 가지 다중 턴 에이전트 작업 데이터셋에 대한 광범위한 실험 결과, DMPO 손실 함수의 효과와 우수성이 입증되었습니다. 프로젝트는 이론적인 설명과 실험적 결과 모두에 대해 신뢰할 수 있는 성능 향상을 보여줍니다.



### From LLMs to MLLMs: Exploring the Landscape of Multimodal Jailbreaking (https://arxiv.org/abs/2406.14859)
- **What's New**: 최근 대형 언어 모델(LLM) 및 멀티모달 대형 언어 모델(MLLM)의 발전은 다양한 적대적 공격에 취약함을 드러냈습니다. 본 논문은 LLM 및 MLLM을 겨냥한 탈옥(jailbreaking) 연구의 종합 개요를 제공하며, 평가 기준, 공격 기술 및 방어 전략의 최신 발전 사항을 강조합니다. 특히, 멀티모달 분야는 아직 탐색이 부족한 상태로 남아 있습니다. 이 연구는 멀티모달 탈옥의 한계와 잠재적 연구 방향을 요약하여 향후 연구를 유도하고 MLLM의 강건성과 보안을 향상시키고자 합니다.

- **Technical Details**: 최근의 LLM은 놀라운 성능을 보여주고 있지만, 공격에 취약하여 모델의 무결성과 신뢰성을 위협하고 있습니다. 탈옥 공격은 악의적인 지침 또는 훈련 및 디코딩 개입을 통해 LLM의 내장된 안전 조치를 우회하여 모델이 바람직하지 않은 행동을 하도록 유도합니다. 방어 전략으로는 감독된 미세 조정(SFT) 및 인간 피드백을 통한 강화 학습(RLHE) 등이 있습니다. MLLM은 시각적 및 언어적 입력에 대응하기 위해 개발되었으며, 이에 따른 다양한 공격에 취약함을 보이는 초기 연구가 이뤄지고 있습니다.

- **Performance Highlights**: LLM과 MLLM을 대상으로 하는 탈옥 공격과 방어 연구는 중요한 의미를 가집니다. LLM의 경우, 단일 회차와 다중 회차 대화 설정에서 다양한 데이터셋을 사용해 평가합니다. MLLM 평가를 위해 MM-SafetyBench와 같은 데이터셋이 제안되어 다중 시나리오에서의 안전성을 평가합니다.



### Leveraging Passage Embeddings for Efficient Listwise Reranking with Large Language Models (https://arxiv.org/abs/2406.14848)
- **What's New**: 최근 많은 연구에서 대형 언어 모델(LLMs)을 사용해 문장을 평가(ranking)하는 효과를 입증했으며, RankGPT와 같은 리스트 방식(listwise) 접근법이 이 작업에서 새로운 상태의 최첨단을 이루었습니다. 그러나 RankGPT 모델의 효율성은 최대 컨텍스트 길이 및 비교적 높은 LLM 추론 지연 시간에 한계가 있습니다. 이를 해결하기 위해, 본 논문에서는 PE-Rank를 제안합니다. 이는 문장 임베딩(single passage embedding)을 컨텍스트 압축의 좋은 대안으로 활용하여 효율적인 리스트 방식 문장 재평가를 수행합니다. 각 문장을 특수 토큰으로 취급하여 문장 임베딩을 LLM에 직접 입력하고, 이를 통해 입력 길이를 줄일 수 있습니다. 또한, 디코딩 공간을 동적으로 이러한 특수 토큰으로 제한하여 디코딩 프로세스를 가속화하는 추론 방법을 도입했습니다.

- **Technical Details**: PE-Rank는 먼저 밀도 검색 모델(dense retrieval model)을 사용하여 문장을 벡터 인덱스로 사전 인코딩합니다. 문장 임베딩을 특수 토큰으로 취급하여 원래 텍스트 대신 LLM에 입력하고, 검색 모델의 임베딩 공간과 LLM의 입력 임베딩 공간을 일치시키기 위해 프로젝터를 사용합니다. 추론에서는 동적으로 디코딩 공간을 제한하여 특수 토큰을 사용할 수 있도록 하는 '동적 제약 디코딩(Dynamic-Constrained Decoding)' 전략을 제안합니다. 또한, 목록 학습(listwise learning to rank loss)을 통해 모델을 학습시킵니다.

- **Performance Highlights**: PE-Rank는 TREC DL 및 BEIR와 같은 인기 있는 검색 벤치마크에서 평가되었습니다. 실험 결과, PE-Rank는 높은 효율성을 유지하면서 비압축 방법과 비교할 때 유사한 순위 성능을 달성했습니다. 예를 들어, DL19에서 BM25로 검색된 상위 100 개 후보를 재평가할 때, PE-Rank의 NDCG@10은 동일한 설정 하에서 비압축 방법에 비해 성능 저하가 2 % 미만이면서 지연 시간을 4.5배 줄였습니다.



### ToVo: Toxicity Taxonomy via Voting (https://arxiv.org/abs/2406.14835)
- **What's New**: 기존의 독성 감지 모델들은 투명성, 맞춤화, 재현성의 부족과 같은 문제점을 가지고 있습니다. 이러한 문제를 해결하기 위해 투표와 chain-of-thought 프로세스를 통합한 새로운 데이터셋 생성 메커니즘을 제안합니다. 이 메커니즘은 높은 품질의 오픈 소스 데이터셋을 생성하며, 각 샘플에 대해 다양한 분류 지표와 분류 점수, 그리고 설명 reasoning을 포함합니다.

- **Technical Details**: 우리는 ToVo (Toxicity Taxonomy Voting) 데이터셋을 도입하였으며, 이는 네 가지 다른 moderation 도구에서 파생된 42개의 사전 정의된 독성 지표(pool of 42 predefined toxicity metrics)를 활용합니다. 각 샘플에 대해 투표를 통해 독성 여부를 결정하고, 이에 대한 설명 reasoning을 제공합니다. 모델 교육은 Mistral-Hermes-2-Pro와 로라(LoRA)를 이용하여 수행되었으며, 단일 A100 GPU로 2회 epoch를 진행하였습니다.

- **Performance Highlights**: 우리 데이터셋과 기존의 moderation 도구들(Llama Guard 2, OpenAI Moderation, Perspective API) 간의 합의율을 평가한 결과, OAIM이 가장 높은 합의율을 보였습니다. 이는 우리의 방법론이 일관성 있고 합리적인 독성 분류 gold labels를 생성할 수 있음을 시사합니다.



### Efficient Continual Pre-training by Mitigating the Stability Gap (https://arxiv.org/abs/2406.14833)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)을 새로운 도메인에 적응시키기 위한 지속적 사전 훈련(Continual pre-training) 과정에서 발생하는 '안정성 갭(stability gap)' 현상을 분석합니다. 이 연구는 이러한 문제를 해결하기 위한 세 가지 전략을 제안하며, 이를 통해 LLM의 성능을 고정된 연산 예산 내에서 효과적으로 향상시킬 수 있음을 입증합니다.

- **Technical Details**: 논문에서는 지속적 사전 훈련 동안 LLM의 성능 변화를 측정한 결과, 초기 학습 단계에서 성능이 일시적으로 하락한 후 회복되는 '안정성 갭' 현상을 관찰했습니다. 이를 해결하기 위해 제안된 세 가지 전략은 다음과 같습니다: (1) 적절한 크기의 서브셋(subset)을 여러 에포크(epoch) 동안 지속적으로 사전 훈련; (2) 높은 품질의 서브 코퍼스(sub-corpus)만을 사용해 빠르게 도메인 성능 향상; (3) 사전 훈련 데이터와 유사한 데이터 혼합비율을 사용해 분포 차이를 줄입니다.

- **Performance Highlights**: 제안된 전략을 적용했을 때, OpenLlama-3B 모델의 평균 의학 과제 성능이 36.2%에서 40.7%로 향상되었으며, 이는 원래 훈련 예산의 40%만 사용한 결과입니다. 또한, Llama-3-8B 모델에 적용했을 때 Llama-3-Physician 모델이 현재 공개된 오픈 소스 모델 중 최고 수준의 의학 성능을 달성했으며, 여러 의학 벤치마크에서 GPT-4와 동등하거나 더 나은 성능을 보였습니다.



### Is this a bad table? A Closer Look at the Evaluation of Table Generation from Tex (https://arxiv.org/abs/2406.14829)
- **What's New**: 연구진은 기존의 테이블 품질 평가 기준이 전체적인 의미를 제대로 반영하지 못하고, 부적절하게 좋은 테이블을 벌하고 나쁜 테이블을 보상하는 문제를 강조했습니다. 이를 해결하기 위해 테이블을 자연어 원자적 진술로 분해하고, 이를 근거 진술과 비교하는 새로운 테이블 평가 전략인 TabEval을 제안하였습니다.

- **Technical Details**: TabEval은 두 단계로 이루어진 파이프라인으로, 첫 단계에서는 Chain-of-Thought을 사용해 테이블을 의미 있는 자연어 원자적 진술로 변환하는 TabUnroll을 사용합니다. 그 후에는 자연어 추론을 통해 예측된 테이블의 진술과 근거 테이블의 진술 간의 함의(entailment) 점수를 계산하여 테이블의 품질을 평가합니다.

- **Performance Highlights**: TabEval은 인간의 품질 평가와 기존 메트릭보다 높은 상관관계를 보여주며, 특히 다양한 도메인에서 수집된 1,250개의 위키피디아 테이블과 그 텍스트 설명을 포함한 새로운 데이터셋 DescToTTo를 통해 검증되었습니다. 실험 결과, TabEval은 대부분의 시나리오에서 기존 메트릭보다 더 높은 신뢰도를 보였습니다.



### Word Matters: What Influences Domain Adaptation in Summarization? (https://arxiv.org/abs/2406.14828)
- **What's New**: 본 논문은 주로 요약 작업에 대한 대규모 언어 모델(LLM)의 도메인 적응(Domain Adaptation) 성능을 분석합니다. 특히 '단어(words)'가 학습 데이터에 미치는 영향을 세밀하게 분석하고, 데이터셋 학습 난이도를 계량화하여 요약 작업의 성능 향상을 더 정확하게 반영하는 방법을 제안합니다. 이를 통해 기존의 학습 없이도 모델의 새로운 도메인 데이터셋에 대한 성능을 예측할 수 있음을 보여줍니다.

- **Technical Details**: 학습 데이터의 난이도를 측정하기 위해 압축율(compression rate)과 추상화 수준(abstraction level) 두 가지 지표를 사용합니다. 실험 결과, 데이터셋 학습 난이도를 고려할 때 도메인 간 겹침(cross-domain overlap)과 요약 작업 성능 향상 사이에서 대략적인 선형 관계가 나타났습니다. 또한, 단어 수(word count)는 성능과 유의미한 상관관계를 보이지 않았습니다. 이 결과를 바탕으로, 학습 없이도 데이터셋의 도메인 간 겹침을 통해 모델 성능을 예측하는 것이 가능함을 제시합니다.

- **Performance Highlights**: 실험에서는 다양한 크기의 모델이 네 개의 도메인 요약 데이터셋에 대해 수행되었습니다. 결과적으로, 데이터셋 난이도 계수를 고려한 도메인 간 겹침이 성능 향상과 약 1대의 선형 관계를 띠는 것을 확인했습니다. 또한, 학습 없이도 난이도 계수와 도메인 간 겹침을 기반으로 모델의 성능을 예측할 수 있음을 실증하여, 특히 자원이 제한된 상황에서 유용한 빠르고 효율적인 검증 방법을 제시합니다.



### TemPrompt: Multi-Task Prompt Learning for Temporal Relation Extraction in RAG-based Crowdsourcing Systems (https://arxiv.org/abs/2406.14825)
Comments:
          12 pages, 9 figures

- **What's New**: 이번 연구는 시간 관계 추출(TRE)을 위해 TemPrompt라는 멀티 태스크 프롬프트 학습 프레임워크를 제안합니다. 이 프레임워크는 프롬프트 튜닝(prompt tuning)과 대조 학습(contrastive learning)을 활용하여 기존 모델에서 발생하는 데이터 부족과 불균형 문제를 해결하려고 합니다.

- **Technical Details**: TemPrompt는 다양한 데이터 문제 해결을 위해 프롬프트 튜닝과 대조 학습을 통합한 멀티 태스크 프레임워크입니다. 자동 프롬프트 생성 방식은 트리거(trigger), 레이블(label) 및 이벤트 언급(event mentions)을 고려하여 큰 감독 비용 없이 의미 있는 프롬프트를 생성합니다. 또한, 시간 이벤트 추론을 보조 작업으로 포함하여 시간 이벤트 지식에 대한 모델의 이해를 강화합니다.

- **Performance Highlights**: MATRES와 TB-Dense 데이터셋에서 수행된 실험적으로 TemPrompt가 대다수의 메트릭에서 비교 기준 모델을 능가하는 성과를 보였습니다. 특히, 다양한 자료 분포에서 더욱 뛰어난 성과를 보였습니다.



### How Well Do LLMs Represent Values Across Cultures? Empirical Analysis of LLM Responses Based on Hofstede Cultural Dimensions (https://arxiv.org/abs/2406.14805)
- **What's New**: 최근 연구에서는 Large Language Models (LLMs)이 인간의 행동을 모방하여 사람들의 가치를 반영하려는 시도를 하고 있지만, 사용자가 어떤 국가에 속해 있는지에 따라 LLMs이 다른 가치를 보여주는지에 대해 조사했습니다. Hofstede 문화 차원(Cultural Dimensions)을 기반으로 36개국의 인물 및 언어를 사용하여 여러 LLMs의 문화적 이해도를 분석했습니다. 연구 결과, LLMs이 국가별 다양한 가치를 이해하고 있지만 항상 그 가치를 반영하는 답변을 제공하지는 못하며, 문화적 차이에 따라 다르게 답변해야 할 필요성을 충분히 이해하지 못하고 있음이 드러났습니다. 연구를 통해 가치 정렬 및 문화적으로 민감한 LLMs을 만들기 위한 권장 사항을 제시했습니다.

- **Technical Details**: 이 연구는 Hofstede의 문화 차원(Individualism vs. Collectivism, Long Term vs. Short Term Orientation, High vs. Low Uncertainty Avoidance, High vs. Low Motivation Towards Achievement and Success, High vs. Low Power Distance Index)이라는 프레임워크를 사용하여 이루어졌습니다. 각 질문은 36개국의 인물 및 해당 국가와 밀접하게 연관된 언어로 설정되어 다양한 LLMs의 반응을 분석했습니다. 이 연구 방법은 LLMs이 문화적 가치에 대해 얼마나 잘 이해하고 있는지, 그리고 사용자의 국가나 언어에 따라 맞춤형 조언을 제공할 수 있는지를 평가하는 데 목적이 있습니다.

- **Performance Highlights**: LLMs은 일반적으로 다른 문화적 가치를 인식할 수 있지만, 항상 그 가치를 반영하거나 명확히 표현하지는 못합니다. 이는 LLMs이 특정 데이터에서 발견한 인기 있는 감정에 따라 답변을 제공할 가능성이 높기 때문입니다. 예를 들어, 일본어로 질문을 받았을 때 LLM이 일본의 전형적인 가치에 맞게 답변을 제공하는지, 아니면 보편적인 가치에 따라 답하는지를 평가했습니다. 연구 결과에 따라 LLMs의 문화적 차이를 해결하기 위한 권장 사항이 도출되었습니다.



### Understanding Finetuning for Factual Knowledge Extraction (https://arxiv.org/abs/2406.14785)
Comments:
          To appear in ICML 2024

- **What's New**: 이번 연구에서는 QA 파인튜닝(fine-tuning) 데이터가 다운스트림 사실성(factuality)에 미치는 영향을 조사했습니다. 미리 학습된 모델에 잘 저장되지 않은 덜 알려진 사실로 파인튜닝을 할 경우, 더 잘 알려진 사실로 파인튜닝을 하는 것보다 사실성이 크게 저하된다는 것을 이론적으로 증명했습니다. 이를 통해 파인튜닝 데이터의 구성에 따라 다운스트림 성능이 영향을 받을 수 있다는 점을 강조했습니다.

- **Technical Details**: 연구는 세 가지 QA 벤치마크 데이터셋(PopQA, Entity Questions, MMLU)과 두 가지 언어 모델(Llama-2-7B, Mistral-7B)을 사용하여 실험을 진행했습니다. 이 연구는 덜 잘 알려진 사실에 대한 파인튜닝이 모델이 주제 엔티티(subject entity) 이름을 무시하고 일반적으로 그럴듯한 응답을 생기게 한다는 점을 이론적으로 증명했습니다. 이를 위해 1층 트랜스포머(transformer)에서 사실 중요도(factual salience)라는 개념을 도입했습니다. 실험 결과, 잘 알려진 사실(top 50%)에 파인튜닝을 하는 것이 덜 알려진 사실에 파인튜닝하는 것보다 다운스트림 사실성을 높이는 데 더 효과적이라는 것을 발견했습니다.

- **Performance Highlights**: 세 가지 데이터셋 모두에서 잘 알려진 지식에 대한 파인튜닝이 덜 알려진 지식에 대한 파인튜닝보다 평균 5-10% 정도 성능이 향상되었으며, 전체 데이터셋에 대해 파인튜닝하는 것을 대체할 수 있음을 보여줬습니다. 특히 MMLU에서 최상위 30%의 잘 알려진 사실에 대한 파인튜닝이 전체 데이터셋에 대한 파인튜닝을 최대 2.5% 초과 성능을 보였습니다.



### A Learn-Then-Reason Model Towards Generalization in Knowledge Base Question Answering (https://arxiv.org/abs/2406.14763)
- **What's New**: 이 논문은 지식 베이스 질의 응답(KBQA)을 위한 새로운 접근법인 KBLLaMA를 소개합니다. 이 프레임워크는 고전적인 retrieve-then-reason 접근법에서 벗어나 learn-then-reason 접근법을 채택하여, 대형 언어 모델(LLM) 내에 새로운 지식 베이스(KB) 지식을 주입합니다. 이를 통해 기존 모델의 일반화 능력을 극대화할 수 있습니다.

- **Technical Details**: KBLLaMA의 핵심은 (1) KBQA에 대한 새로운 지식을 어떻게 조직화할 것인가와 (2) 조직화된 지식을 어떻게 학습할 것인가 입니다. 이 모델은 LLaMA2-7B를 기반으로 새로운 KB 지식을 자연어 질문과 논리적 표현 쌍으로 조직화하고, 해당 쌍을 이용해 모델을 미세 조정(fine-tuning)합니다. 또한 GrailQA와 같은 일반 벤치마크와 바이오-화학과 같은 도메인 벤치마크에서 높은 성능을 보입니다.

- **Performance Highlights**: KBLLaMA는 In-KB 및 Cross-KB 일반화 작업에서 최첨단 성능을 입증했습니다. 특히 일반 벤치마크 GrailQA에서 최대 3.8%, 도메인 특화 벤치마크 바이오-화학에서 9.8%의 성능 향상을 보였습니다.



### An LLM Feature-based Framework for Dialogue Constructiveness Assessmen (https://arxiv.org/abs/2406.14760)
- **What's New**: 이 논문은 대화 건설성 평가를 위해 혁신적인 LLM(Large Language Model, 대형 언어 모델) 특징 기반 프레임워크를 제안합니다. 이 프레임워크는 설명 가능한 특징 기반 모델과 높은 정확도를 자랑하는 신경망 모델의 장점을 결합하여, 대화의 건설성을 예측할 수 있게 합니다.

- **Technical Details**: 프레임워크는 먼저 데이터셋에 독립적이며 이해할 수 있는 언어적 특징들을 정의하고, 이러한 특징들을 LLM을 통해 추출합니다. 그런 다음 추출된 특징들을 기준으로 LLM 특징 기반 모델을 훈련합니다. 구체적으로, 평균 협력 마커(collaboration markers) 및 논쟁 전술(dispute tactics) 등의 언어적 특징들을 포함한 총 6개의 특징 세트를 사용합니다.

- **Performance Highlights**: 세 가지 대화 건설성 데이터셋을 사용해 실험한 결과, 제안된 LLM 특징 기반 모델이 일반적인 특징 기반 모델과 기존 신경망 모델보다 뛰어난 예측 성능과 견고성을 보였습니다. 또한, 이 모델은 대화 건설성을 예측하는 데 중요한 언어적 요소들을 식별하는 데 유용한 통찰을 제공할 수 있습니다.



### An Adapter-Based Unified Model for Multiple Spoken Language Processing Tasks (https://arxiv.org/abs/2406.14747)
Comments:
          ICASSP 2024

- **What's New**: 이 논문은 여러 음성 처리 작업을 효과적으로 처리할 수 있는 통합 모델을 개발하기 위해 어댑터 기반 미세 조정(adapter-based fine-tuning)을 탐구합니다. 이 연구는 SUPERB 벤치마크에서 ASR, PR, IC, SF, ER의 다섯 가지 작업에서 어댑터 기반 미세 조정이 평균 18.4% 성능 개선을 가져오는 것으로 나타났습니다.

- **Technical Details**: 이 연구는 wav2vec 2.0-large 모델을 인코더로 사용하고 6층 트랜스포머 디코더를 무작위 초기화했습니다. LibriSpeech 100시간 데이터셋과 하이브리드 CTC/어텐션 목적함수를 사용해 이 인코더-디코더 모델을 기본 모델로 미세 조정했습니다. 이 모델을 다양한 SLP 작업을 수행할 수 있도록 트랜스포머 레이어에 작업별 어댑터 모듈을 삽입했습니다. 어댑터 기반 MTL을 위해 스태킹(stacking)과 퓨전(fusion) 방식을 사용했습니다.

- **Performance Highlights**: 실험 결과, 어댑터 기반 미세 조정이 SUPERB 벤치마크에서 5가지 목표 작업(ASR, PR, IC, SF, ER)에서 평균 18.4%의 성능 향상을 달성했습니다. 특히, 상관된 작업들을 동시에 수행할 때 성능이 더욱 향상되었습니다.



### Relation Extraction with Fine-Tuned Large Language Models in Retrieval Augmented Generation Frameworks (https://arxiv.org/abs/2406.14745)
Comments:
          preprint

- **What's New**: 이 논문은 Fine-Tuning을 통해 대규모 언어 모델(LLMs)을 활용하여 문장 수준에서 암묵적 관계를 추출하는 방법을 탐구합니다. 이를 통해 대규모 언어 모델이 논리적 추론을 통해 암묵적 관계를 인식할 수 있도록 도와줍니다.

- **Technical Details**: 이 연구에서는 LLM들을 Retrieval Augmented-based (RAG) RE 프레임워크 내에서 생성기로 통합하여 암묵적 관계의 식별 문제를 해결하고자 합니다. 실험에는 Llama2-7B, Mistral-7B, T5 Large가 포함되며, TACRED, TACRED-Revisited (TACREV), Re-TACRED, SemEVAL 데이터셋을 사용해 성능을 평가합니다.

- **Performance Highlights**: Fine-Tuning된 LLM들을 사용한 결과, SemEVAL 데이터셋에서 큰 성능 개선을 이루었으며, TACRED, TACREV, Re-TACRED에서도 우수한 성능을 보였습니다. 특히 SemEVAL 데이터셋에서는 암묵적 관계가 일반적이라 이 접근법이 뛰어난 성과를 냈습니다.



### Learning to Retrieve Iteratively for In-Context Learning (https://arxiv.org/abs/2406.14739)
- **What's New**: 이번 연구에서는 새로운 반복 검색 프레임워크(Iterative Retrieval)를 소개하였습니다. 이는 정책 최적화를 통해 검색기가 반복적인 결정을 내리도록 하여, 최적화 문제를 해결하는 접근 방식을 제시합니다. 특히, 이 방법은 대형 언어 모델(LLM)을 활용한 작문(task) 평가를 목표로 합니다.

- **Technical Details**: 이 프레임워크는 강화 학습(Reinforcement Learning)을 통해 학습되며, LLM으로부터 피드백을 받아 개선됩니다. 반복 검색기는 기존의 밀집 검색기(dense retriever)에 4백만 개의 추가 파라미터만을 추가하여 상태를 인코딩합니다. 이를 통해 오프 더 쉘프(off-the-shelf) 검색기가 상태 기반의 반복 검색기로 변환됩니다. 마르코프 결정 과정(Markov Decision Process, MDP)을 활용하여, 검색 과정에서 각 단계마다 최적의 예시(exemplar)를 선택하는 것이 목표입니다.

- **Performance Highlights**: 제안된 반복 검색기는 CalFlow, TreeDST 및 MTOP과 같은 의미 분석(semantic parsing) 데이터세트에서 기존 방법보다 우수한 성능을 보였습니다. 이 검색기는 학습에 사용된 LLM 외의 다양한 추론 기반 LLM에서도 일반화된 성능을 입증하였습니다.



### Dissecting the Ullman Variations with a SCALPEL: Why do LLMs fail at Trivial Alterations to the False Belief Task? (https://arxiv.org/abs/2406.14737)
- **What's New**: 최근 대형 언어 모델(LLMs)이 '마음 이론(Theory of Mind)'을 구현할 수 있는지에 대한 논쟁이 불거졌습니다. 일부 연구자들은 LLMs가 거짓 믿음 과제(False Belief task)에서 성공을 거두었다고 주장하는 반면(Ullman, 2023), 다른 연구자들은 LLMs의 실패 원인이 일반적인 상식 추론의 부족 때문이라고 주장합니다. 이 논문에서는 거짓 믿음 과제에 대해 다양한 가설을 테스트하기 위해 SCALPEL이라는 기법을 도입합니다.

- **Technical Details**: SCALPEL(선택적 비교를 통한 대립적 언어적 프롬프트 설명)은 LLMs가 투명한 용기(Transparent-Access)로 변경된 예기치 않은 내용 과제(Unexpected Contents Task)에서 실패하는 이유를 구체적으로 설명하기 위해 사용된 기법입니다. 여기에는 '투명' 대신 '보이는'과 같은 단어를 사용하거나, '투명한 용기란 내부를 볼 수 있는 용기'라는 구절을 추가하여 보다 명확한 설명을 부여하는 방법이 포함됩니다.

- **Performance Highlights**: GPT-3.5와 GPT-4 모델은 투명한 용기 변형의 기본 테스트에서 약 20%의 정확도를 보였습니다. '보이는(see-through)' 또는 '안을 볼 수 있는(see-inside)'이라는 수정된 프롬프트에서도 모델의 성능에는 유의미한 변화가 없었습니다. 이 결과는 LLMs의 실패가 단순히 학습된 경험적 연관성을 사용하지 못한 것이 아니라, 특정 상황을 올바르게 이해하지 못했기 때문일 가능성을 시사합니다.



### TTQA-RS- A break-down prompting approach for Multi-hop Table-Text Question Answering with Reasoning and Summarization (https://arxiv.org/abs/2406.14732)
- **What's New**: 다중 홉 (multi-hop) 테이블-텍스트 질의응답(QA) 모델 TTQA-RS를 제안합니다. 이 모델은 테이블과 텍스트 요약을 통한 서브질문을 생성하여, 논리에 기반한 테이블-텍스트 QA를 수행합니다. TTQA-RS는 오픈소스 언어 모델(Large Language Models, LLMs)을 이용하여 기존의 프롬프트 방법을 초과하는 성능을 보였으며, 학습 기반 모델들과 비교해도 우수한 성능을 나타냈습니다. 특히 GPT-4와 LLaMA3-70B를 이용하여, 다중 홉 테이블-텍스트 QA에서 최첨단의 성능을 달성하였습니다.

- **Technical Details**: TTQA-RS 모델은 다섯 가지 단계로 테이블-텍스트 QA 문제를 해결합니다. (1) 테이블 행과 텍스트로부터 요약 생성, (2) 질문 분해, (3) 예상 답변의 엔터티 타입 예측, (4) 독립 서브질문의 테이블-텍스트 QA, (5) 원래 질문의 테이블-텍스트 QA. 추출기는 테이블 셀에 링크된 텍스트에서 관련 행과 패시지를 가져옵니다. HybridQA 데이터셋에서는 S3HQA 모델의 테이블 추출기와 HYBRIDER의 텍스트 추출기를 사용합니다. OTT-QA의 개발 세트에서는 테이블 추출기를 사용하지 않고 HYBRIDER의 텍스트 추출기만을 사용합니다.

- **Performance Highlights**: TTQA-RS 모델은 HybridQA와 OTT-QA 데이터셋에서 기존의 프롬프트 기반 방법을 능가하는 성능을 보였습니다. 특히 HybridQA 테스트 세트에서 기존 Chain of Thought(CoT) 모델 대비 정확한 일치 점수에서 6% 증가를 달성하였습니다. 또한, S3HQA의 GPT 3.5 성능을 초과하여, 소규모 LLMs에서도 다중 홉 테이블-텍스트 QA에서 우수한 잠재력을 입증하였습니다.



### 1+1>2: Can Large Language Models Serve as Cross-Lingual Knowledge Aggregators? (https://arxiv.org/abs/2406.14721)
- **What's New**: 본 논문에서는 다양한 언어에 걸친 정보 처리가 뛰어난 대형 언어 모델(LLMs)의 다국어 성능을 향상시키기 위한 방법을 제안합니다. 이 방법은 저자원 지식 감지기(low-resource knowledge detector), 언어 선택 프로세스(language selection process), 답변 교체 및 통합(answer replacement and integration)을 포함합니다. 실험 결과, 이 방법은 언어 간 성능 차이를 줄이는 데 현저한 개선을 보여주었습니다.

- **Technical Details**: 제안된 방법은 다음 세 가지 주요 모듈로 구성됩니다: 1) 저자원 지식 감지기는 사용자의 쿼리가 특정 언어에서 저평가된 지식을 포함하는지 평가합니다. 2) 저자원 지식이 탐지되면 LLM은 가장 관련성이 높은 목표 언어를 선택합니다. 3) 쿼리는 선택된 언어로 번역되고, 해당 언어로 답변이 생성되며 원래 언어로 다시 번역되어 사용자에게 전달됩니다.

- **Performance Highlights**: 실험은 영어와 중국어의 이중 언어 데이터셋을 사용하여 6개의 인기 있는 LLM에 대해 수행되었으며, 제안된 방법이 언어 간 지식을 성공적으로 통합하고 성능을 향상시키며 언어 불일치 문제를 해결하는 데 효과적임을 입증했습니다. 또한, 각 구성 요소가 성능 개선에 중요한 역할을 함을 확인하기 위해 분석 연구를 실시했습니다.



### MultiAgent Collaboration Attack: Investigating Adversarial Attacks in Large Language Model Collaborations via Deba (https://arxiv.org/abs/2406.14711)
- **What's New**: 이번 논문에서는 다양한 작업을 수행하기 위해 여러 LLM(Large Language Models)이 협력하는 방식을 평가합니다. 우리는 경쟁자가 개입하여 협력 메커니즘을 방해할 때의 모델 네트워크 행동을 분석합니다. 또한 이러한 위협의 효과성을 평가하기 위해 시스템 정확도와 모델 동의율을 주요 지표로 도입했습니다.

- **Technical Details**: 논문에서는 추론(Reasoning), 신뢰성(Truthfulness), 의료(MedMCQA), 법적(Scalr) 작업을 통해 모델 간의 토론이 영향을 받는 방법을 실험합니다. 토론 프로토콜 아래에서 각 모델은 질문에 대한 초기 응답을 합니다. 라운드마다 다른 모델의 응답을 받고 재검토 및 수정하여 최종 답변을 다수결(Majority Vote)로 선택합니다. 경쟁자는 잘못된 답변을 선택하여 다른 모델들에게 이를 정답으로 받아들이게 설득하려고 시도합니다.

- **Performance Highlights**: 1) 협력 토론 메커니즘은 경쟁자에 의해 취약해질 수 있습니다. 시스템의 정확도가 10%-40%, 개별 모델의 정확도가 최대 30%까지 감소했습니다. 2) 모델의 설득력은 협력 환경에서 중요한 능력으로, 이로 인해 다른 모델의 응답이 변경될 가능성이 높습니다. 3) 에이전트 수나 라운드 수가 많아져도 경쟁자의 효과는 여전히 큽니다.



### Factual Dialogue Summarization via Learning from Large Language Models (https://arxiv.org/abs/2406.14709)
- **What's New**: 이 논문은 대화 요약 (dialogue summarization)에서 사실적 일관성을 향상시키기 위해 상징적 지식 증류 (symbolic knowledge distillation) 방법을 제안합니다. 대형 언어 모델 (LLMs)은 일관된 요약을 생성할 수 있지만, 자원 및 프라이버시 문제로 인해 실세계 응용에서는 사용이 어렵습니다. 본 연구에서는 사실적 일관성을 높이기 위해 LLM을 활용하여 사실적으로 일관성 있는 요약과 일관성 없는 요약을 모두 생성한 후, 이를 기반으로 작은 사전 학습 모델을 개선합니다.

- **Technical Details**: 우리는 대화 요약을 위한 상징적 지식 증류의 시도를 했습니다. GPT-3.5 turbo를 이용해 사실적으로 일관된(긍정적) 요약과 비일관된(부정적) 요약을 생성하고, 이를 통해 작은 요약 모델을 교육하기 위한 대조 학습 (contrastive learning) 방법을 적용했습니다. 실험에 사용된 사전 학습 모델은 BART, PEGASUS, 그리고 Flan-T5입니다. 이러한 접근법은 사실적인 일관성을 향상시키고 자동 평가 지표를 통해 유창성, 일관성 및 관련성을 유지하는 것으로 나타났습니다.

- **Performance Highlights**: 제안된 방법은 기존 데이터 증강 전략에 의존하는 강력한 기초모델들을 능가하는 성능을 보여 주었습니다. 실험 결과, 상징적 지식 증류를 통해 훈련된 작은 모델들이 인간 작성 요약과 비교할 때 사실적 일관성 면에서 더 나은 성능을 발휘하며, 유창성이나 일관성을 손상시키지 않았음이 확인되었습니다. 우리는 향후 연구를 위해 데이터 및 코드를 공개합니다.



### Do LLMs Have Distinct and Consistent Personality? TRAIT: Personality Testset designed for LLMs with Psychometrics (https://arxiv.org/abs/2406.14703)
Comments:
          Preprint; Under review

- **What's New**: 이번 연구는 대형 언어 모델(Large Language Models, LLMs)의 성격을 평가하기 위해 TRAIT라는 새로운 도구를 소개합니다. 기존의 자기 평가 성격 테스트는 신뢰성과 타당성이 부족하여 정확한 성격 측정이 어려웠지만, TRAIT는 이를 극복하고 높은 신뢰성과 타당성을 보여주기 위해 개발되었습니다.

- **Technical Details**: TRAIT는 심리적으로 검증된 인간 질문지인 Big Five Inventory (BFI)와 Short Dark Triad (SD-3)를 기반으로 하여 다양한 현실 시나리오에서 LLM의 성격을 테스트하는 데 사용됩니다. TRAIT에는 ATOMIC10X 지식 그래프를 활용한 8,000개의 다중 선택 질문이 포함되어 있어, 다양한 상황에서 LLM의 반응 패턴을 통계적으로 유의미하게 분석할 수 있습니다.

- **Performance Highlights**: TRAIT는 세 가지 주요 메트릭인 거부율(refusal rate), 프롬프트 민감도(prompt sensitivity), 선택지 순서 민감도(option order sensitivity)에서 최고 점수를 기록하며, LLM의 성격에 대한 중요한 통찰을 제공합니다. 예를 들어, 정렬 튜닝(alignment tuning)은 LLM의 외향성(extraversion), 개방성(openness), 사회적 적대성(dark triad)을 감소시키는 반면, 동의성과 성실성을 증가시킵니다. 또한, 현행 프롬프트 기술은 특정 성격 특성을 이끌어내는 데 한계가 있어 추가 연구가 필요함을 시사합니다.



### Depth $F_1$: Improving Evaluation of Cross-Domain Text Classification by Measuring Semantic Generalizability (https://arxiv.org/abs/2406.14695)
- **What's New**: 최신 크로스 도메인 텍스트 분류 모델의 평가 전략은 소스 도메인에서 학습된 모델이 타겟 도메인에서도 일관된 성능을 유지할 수 있는지를 측정하는 데 중점을 둔다. 그러나 기존의 평가 전략은 소스와 타겟 도메인 간의 유사성을 충분히 고려하지 못하며, 이는 모델이 소스 도메인과 크게 다른 특정 타겟 샘플에 대한 전이 학습에 실패했을 때 이를 감지하지 못할 수 있다. 이를 해결하기 위해 Depth F1이라는 새로운 크로스 도메인 텍스트 분류 성능 메트릭을 도입했다. 이 메트릭은 소스 도메인과 유사하지 않은 타겟 샘플에서 모델이 얼마나 잘 작동하는지를 측정한다.

- **Technical Details**: Depth F1 메트릭은 기존의 F1 점수와 보완적으로 설계되었으며, 통계적 깊이 함수(Statistical Depth Function)를 사용해 소스와 타겟 도메인의 개별 샘플 간의 차이를 측정한다. 이 메트릭은 크로스 도메인 텍스트 분류 모델의 의미 논리적 일반화 가능성을 평가하는 데 중점을 두고 있다. 이를 통해 기존 평가 전략이 범하지 않는 샘플 단위의 성능 차이를 포착하며, 소스-비유사적 타겟 샘플에 대한 모델 성능을 종합적으로 평가할 수 있다.

- **Performance Highlights**: 최신 전이 학습 모델을 사용한 두 개의 벤치마크 크로스 도메인 텍스트 분류 데이터셋에서의 실험을 통해 Depth F1 메트릭의 필요성을 강조했다. 이 실험에서 기존 평가 전략이 소스-비유사적 타겟 샘플에서의 낮은 모델 성능을 감추고 있음을 발견하고 탐구했다. Depth F1 메트릭은 이러한 타겟 샘플에서의 모델 성능을 포괄적으로 평가하는 데 효과적임을 입증했다.



### A Contrastive Learning Approach to Mitigate Bias in Speech Models (https://arxiv.org/abs/2406.14686)
Comments:
          Accepted at Interspeech 2024

- **What's New**: 이 논문은 종전의 방법이 특정 사용자 정의 하위 그룹만을 초점으로 하거나 하위 그룹 수준에서 명시적으로 내부 표현을 개선하지 못하는 문제를 해결하기 위해, 성능이 부진한 하위 그룹을 대상으로 편향을 완화하는 최초의 대조 학습(Contrastive Learning) 접근법을 제안합니다. CLUES라는 대조 학습 프레임워크를 통해 모델이 태스크(task), 하위 그룹(subgroup), 하위 그룹 내 오류(error)라는 세 가지 수준의 대조 손실(contrastive loss)에 초점을 맞추도록 유도합니다.

- **Technical Details**: CLUES는 하위 그룹 간의 성능 격차를 줄이기 위해 세 가지 대조 학습 수준을 사용합니다. 첫 번째 수준은 태스크 수준에서 동일한 클래스를 공유하는 샘플들은 가깝게, 다른 클래스의 샘플들은 멀게 배치합니다. 두 번째 수준은 동일한 하위 그룹 내의 샘플들은 가깝게, 다른 하위 그룹의 샘플들은 멀게 배치합니다. 세 번째 수준에서는 하위 그룹 내에서 올바르게 예측된 샘플들은 가깝게, 잘못 예측된 샘플들은 멀게 배치합니다. CLUES는 K-Means 클러스터링이나 DivExplorer 등을 사용해 자동으로 하위 그룹을 정의할 수 있습니다.

- **Performance Highlights**: 제안된 방법론은 영어 및 이탈리아어 두 가지 언어와 두 개의 구어체 이해 데이터셋을 사용한 실험에서 하위 그룹의 내부 표현을 개선하여 모델 편향을 감소시키고 성능을 향상시킨 것으로 나타났습니다. FSC 데이터셋에서는 성능이 가장 낮은 하위 그룹의 격차가 66.9% 감소했으며, ITALIC 데이터셋에서는 15.5% 감소했습니다. 또한 F1 Macro 점수에서 각각 6.1%, 4.8% 증가한 성과를 보였습니다.



### Dravidian language family through Universal Dependencies lens (https://arxiv.org/abs/2406.14680)
Comments:
          unpublished report from 2021

- **What's New**: 최근 Universal Dependencies (UD) 프로젝트는 다국어 자연어 처리(NLP) 지원을 위해 다양한 언어에 대한 일관된 의존 관계 주석을 만들고 있습니다. 현재 114개의 언어를 지원하고 있으나, 드라비다어 (Dravidian languages) 계열의 언어는 단 2개만 포함되어 있습니다. 이 논문은 드라비다어 계열 언어의 형태 및 구문적 특징을 살펴보고, 이들을 UD 프레임워크 내에서 어떻게 주석화할 수 있을지 탐구합니다.

- **Technical Details**: 드라비다어 계열은 약 80여 개의 언어 및 방언으로 구성되며, 총 2억 2천만 명 이상이 사용합니다. 드라비다어는 주로 남아시아, 특히 남인도와 중인도에 집중적으로 사용되며, 넷상자 최소화, 반복어 및 에코어와 같은 특징들을 가지고 있습니다. 주로 종결어미를 사용하여 문법적 관계를 표현합니다. 드라비다어 계열 언어들은 대체로 자유 어순을 가지며, 주어-목적어-동사가 일반적인 어순입니다.

- **Performance Highlights**: 현재 UD 프레임워크 내에서 드라비다어 계열 언어들은 가장 적게 대표되고 있습니다. 타밀어와 텔루구어 두 개의 작은 트리 뱅크(treebanks)만 존재하며, 각각 약 500문장과 1300문장으로 구성되어 있습니다. 이 논문은 드라비다어 언어의 다이버시티를 UD에 추가하고 NLP 자원을 개발하기 위해 다양한 드라비다어 언어에 대한 트리 뱅크를 구축할 필요성을 제기하고 있습니다. 드라비다어 언어 주석에 대한 첫 논문으로, 형태소 수준의 주석화 방향 또한 제안합니다.



### Bidirectional Transformer Representations of (Spanish) Ambiguous Words in Context: A New Lexical Resource and Empirical Analysis (https://arxiv.org/abs/2406.14678)
Comments:
          16 pages, 12 figures, submitted to conference (EMNLP 2024)

- **What's New**: 이 논문은 스페인어 다의어 명사의 의미 표현을 평가하기 위한 새로운 데이터셋을 개발하고 이를 활용해 여러 BERT 기반 대형 언어 모델(LLMs)의 성능을 비교합니다. 연구를 통해 스페인어와 영어 모델에서 모델 규모와 성능 간의 상관관계가 다르다는 흥미로운 발견을 발표합니다.

- **Technical Details**: 연구는 최소쌍 문장(minimal-pair sentences)을 사용해 목표 다의어 명사의 같은 또는 다른 의미를 유도하는 새로운 데이터셋을 만들었습니다. 미리 등록된 연구를 통해 각 문장 쌍에 대한 인간의 관련성 판단을 수집했습니다. 여러 모노링구얼과 멀티링구얼 BERT 기반 모델들을 분석해 인간 판단과 모델의 의미 표현 간의 관계를 조사했습니다.

- **Performance Highlights**: 스페인어 LLM의 경우, 영어 모델과는 달리 모델 규모와 성능 간의 상관관계가 나타나지 않았습니다. 인간 판단을 기준으로 다양한 BERT 기반 LLM들이 어느 정도의 변동성을 포착할 수 있었지만, 인간 벤치마크에는 미치지 못했습니다. 연구는 또한 특정 LLM 아키텍처 내에서 목표 명사 분해의 전형적인 궤적을 발견하고 이를 영어에서 부분적으로 재현했습니다.



### Insights into LLM Long-Context Failures: When Transformers Know but Don't (https://arxiv.org/abs/2406.14673)
- **What's New**: Large Language Models (LLMs)에서 중간 또는 끝부분의 정보를 제대로 활용하지 못하는 경향인 위치 편향(positional bias)을 해결하려는 연구가 진행되었습니다. 이 연구는 LLM이 중간표현(hidden representations)을 통해 숨겨진 정보를 처리하는 방식을 탐구하여, 정보를 '알지만 말하지 않는(know but don't tell)' 현상을 밝혀냈습니다.

- **Technical Details**: 본 연구는 LLM의 다양한 레이어와 위치에서 내부 표현(internal representation)을 기반으로 프로빙 분석(probing analysis)을 수행했습니다. 이를 통해 LLM이 타겟 정보의 위치를 얼마나 정확하게 인식하고 있는지를 측정합니다. 실험은 두 가지 작업(Key-Value 쌍 검색 및 다중 문서 질문 응답)과 세 가지 최신 오픈 소스 모델을 대상으로 진행되었습니다.

- **Performance Highlights**: 실험 결과, LLM은 중요한 정보의 위치를 정확하게 인식하더라도 이를 효과적으로 활용하는 데 실패하는 경향이 있음을 발견했습니다. 특히, kv-pairs와 MDQA 설정에서 프로빙 최고 정확도는 문서 구조화로부터의 응답 생성 정확도보다 일관되게 높았습니다. 이는 LLM이 정보를 인코딩하는 것과 응답 생성 사이에 연결 고리가 부족함을 시사합니다.



### Exploring Design Choices for Building Language-Specific LLMs (https://arxiv.org/abs/2406.14670)
Comments:
          15 pages, 6 figures, 11 tables

- **What's New**: 이 논문에서는 단일 언어와 다중 언어의 대형 언어 모델(LLMs)을 조정하여 언어별 LLM을 구축하는 방법을 연구합니다. 각각의 디자인 선택(베이스 모델 선택, 어휘 확장 및 지속적인 미세 조정)이 효율성과 최종 작업 성능에 미치는 영향을 체계적으로 실험합니다.

- **Technical Details**: 논문에서는 두 단계로 된 간단한 조정 방식을 제안합니다. 첫째로, 목표 언어에서 추출한 토큰을 기본 어휘에 추가합니다. 둘째로, 목표 언어 말뭉치에서 새로운 토큰을 효율적으로 사용할 수 있도록 언어 모델링 목표로 LLM을 계속 훈련합니다. 이를 위해 300k 예제의 BPE sentencepiece 토크나이저를 사용하여 특정 언어의 어휘를 생성합니다.

- **Performance Highlights**: ['초기 적응 전 베이스 LM의 성능이 항상 최종 성능을 나타내는 것은 아닙니다.', '적당한 양의 어휘 추가(10K)는 영어와 저자원 언어 간의 효율성 격차를 메울 수 있습니다.', '어휘 확장은 초기의 최종 작업 성능을 떨어뜨리지만, 대부분의 베이스 LLM들은 지속적인 훈련 후 성능을 회복 및 개선할 수 있습니다.', '새로운 토큰 매개변수 초기화는 효율적인 적응에 중요하며, 간단한 평균 초기화 방식이 효과적입니다.']



### Co-training for Low Resource Scientific Natural Language Inferenc (https://arxiv.org/abs/2406.14666)
Comments:
          Accepted in ACL 2024 (main conference)

- **What's New**: 이 논문에서는 Scientific Natural Language Inference (NLI) 작업에서 발생하는 레이블 노이즈 문제를 해결하기 위해 새로운 공동 학습(co-training) 방법을 제안합니다. 기존의 반자율 학습(SSL) 방법들과 달리, 두 개의 분류기(classifier)가 학습 과정에서 각 레이블의 품질을 상호 평가합니다. 이를 통해 자동으로 주어진 레이블의 중요도 가중치를 할당하고, 노이즈가 모델 학습에 미치는 영향을 최소화합니다.

- **Technical Details**: 제안된 방법은 각 예시의 학습 동역학(training dynamics)을 기반으로 레이블의 중요도 가중치를 계산합니다. 구체적으로, 훈련 과정 동안 각 예시에 대해 예측된 확률의 평균 신뢰도와 변동성을 고려하여 가중치를 부여합니다. 이를 통해 비교적 깨끗한 예시에는 높은 가중치를, 노이즈 가능성이 높은 예시에는 낮은 가중치를 부여합니다. 그리고 모호한 예시에 대해서는 두 분류기 사이의 가중치 차이를 통해 다양성을 장려합니다.

- **Performance Highlights**: 제안된 방법은 기존의 먼 거리 감독(distant supervision) 기법에 비해 Macro F1 점수가 1.5% 향상되었으며, 여러 다른 강력한 SSL 기법들에 비해서도 상당한 성능 향상을 보여줍니다.



### OpenDebateEvidence: A Massive-Scale Argument Mining and Summarization Datas (https://arxiv.org/abs/2406.14657)
Comments:
          Accepted for Publication to ARGMIN 2024 at ACL2024

- **What's New**: OpenDebateEvidence는 미국 경쟁 토론 커뮤니티에서 소스를 얻은 대규모 데이터셋으로, 약 350만 개의 문서를 포함하고 있습니다. 이 데이터셋은 고등학교 및 대학 토론에서의 복잡한 논증을 포착하며, 이는 자신을 갈고닦는 데 매우 귀중한 자원입니다. OpenDebateEvidence는 컴퓨터 기술 논증 연구의 발전을 목표로 하고 있으며, 연구자와 실무자를 위한 다양한 NLP 작업 및 응용 프로그램을 위한 풍부한 메타데이터를 제공합니다.

- **Technical Details**: OpenDebateEvidence는 NSDA 토론 주제를 포함한 약 350만 개의 문서를 포함하여 Policy, Lincoln-Douglas (LD), Public Forum과 같은 다양한 토론 형식의 증거를 다룹니다. 이 데이터셋은 Low-Rank Adaptation (LoRA), Representation Fine-Tuning (ReFT), Orthogonalization 등과 같은 고급 기술을 사용하여 LLaMA3-8B 및 Mistral-7B와 같은 최첨단 대형 언어 모델을 미세 조정하는 데 사용되었습니다. 이러한 방법들을 통해 모델 성능을 크게 향상시켰습니다.

- **Performance Highlights**: OpenDebateEvidence로 훈련된 모델은 기존의 논증 마이닝 데이터셋보다 성능이 크게 향상되었습니다. 실험 결과, 모델들이 OpenDebateEvidence 및 기타 관련 논쟁 데이터셋에서 뛰어난 성능을 보였음을 보여주었습니다. 또한, 논리적 요약 작업에서 높은 효율성을 입증하여 법률 문서 분석, 교육 도구 및 AI 모델 개발 등에서 실용적인 응용 가능성을 높였습니다.



### Major Entity Identification: A Generalizable Alternative to Coreference Resolution (https://arxiv.org/abs/2406.14654)
Comments:
          16 pages, 6 figures

- **What's New**: 본 논문에서는 주요 엔티티 식별(MEI, Major Entity Identification) 과제를 새롭게 제안하여, 기존의 코어퍼런스 해소(CR, Coreference Resolution) 모델들의 한계를 극복하고자 합니다. MEI은 입력 텍스트에 주요 엔티티들이 명시되어 있는 상태로만 작업을 제한하며, 자주 발생하는 엔티티만을 대상으로 합니다.

- **Technical Details**: MEI은 기존의 CR 과제와는 달리, 주어진 텍스트와 함께 주요 엔티티를 입력으로 받습니다. 이를 통해 도메인 적응을 훈련이 아닌 추론 단계에서 수행하게 됩니다. MEI 과제는 분류 기반의 접근 방식을 취하여, 모든 텍스트 스팬이 입력 엔티티 중 하나에 해당하거나 무시되는 분류 문제로 정의됩니다. 이는 분류 기반 메트릭을 사용할 수 있도록 하여 현재의 CR 메트릭보다 더 강건한 평가가 가능합니다.

- **Performance Highlights**: 다양한 데이터셋에서의 실험 결과, MEI 모델은 CR 모델보다 도메인 전반에 걸쳐 더 잘 일반화됨을 보여줍니다. MEI의 성능이 CR보다 더 우수하거나 비슷한 종합 성능을 나타내었으며, GPT-4 같은 LLM(Large Language Models)에서도 이를 확인할 수 있었습니다. 특히, 중요한 것은 CR 모델이 도메인에 따른 성능 차이를 보이는 반면, MEI 모델은 그 성능 차이가 더 적었습니다.



### Unveiling the Spectrum of Data Contamination in Language Models: A Survey from Detection to Remediation (https://arxiv.org/abs/2406.14644)
Comments:
          ACL 2024 Camera-Ready Version

- **What's New**: 이번 연구는 대규모 언어 모델(LLMs) 훈련 시 인터넷에서 얻은 대규모 자료들이 평가 벤치마크 데이터와 중복되면서 발생하는 데이터 오염 문제를 종합적으로 다루고 있습니다. 데이터 오염(dinction)의 정의, 영향, 검출 방법 및 완화 전략에 대해 포괄적인 조사를 수행하며, 향후 연구를 위한 방향성을 제시합니다.

- **Technical Details**: 데이터 오염은 LLMs의 훈련 단계에서 평가 데이터가 훈련 데이터에 포함됨으로써 발생하는 문제입니다. 이는 모델의 일반화 성능을 왜곡할 수 있습니다. 본 논문에서는 데이터 오염 검출 방법들을 체계적으로 분류하고, 각 방법의 가정, 강점, 약점을 분석합니다. 검출 방법으로는 Retrieval, Temporal Cutoff, Masking-based, Perturbation-based, Behavioral Manipulation, Membership Inference Attacks 등이 있습니다. 각 방법은 개발자의 접근 가능성 및 데이터의 속성에 따라 달라질 수 있습니다.

- **Performance Highlights**: 이 종합 조사는 데이터 오염 문제의 현재 상태를 명확히 하고, 이를 해결하기 위한 다양한 방법론을 제공함으로써 NLP 연구자들이 평가의 신뢰성을 높이는데 기여할 것으로 기대됩니다. 특히, 데이터 오염의 영향을 정확히 파악하고, 이를 검출 및 완화하는 방안을 체계적으로 제시함으로써 향후 연구 방향을 명확히 제시합니다.



### Can LLMs Learn by Teaching? A Preliminary Study (https://arxiv.org/abs/2406.14629)
Comments:
          Under review

- **What's New**: 이번 논문에서는 인간의 학습 방식인 '가르치면서 배우기(Learning by Teaching, LbT)'를 대형 언어 모델(LLM)에 적용하는 새로운 방법론을 소개합니다. 이를 통해 인간이 생성한 데이터나 더 강력한 모델에 의존하지 않고도 계속해서 모델을 발전시킬 수 있는 가능성을 모색합니다.

- **Technical Details**: 논문에서는 인간의 LbT를 세 가지 수준으로 모방하는 방법을 설계했습니다: 학생의 피드백 관찰, 피드백으로부터 학습, 반복적 학습. 이러한 방법들을 기존 LLM 훈련 및 프롬프트(pipeline) 과정에 통합하여 눈에 띄는 향상을 달성하였습니다. 구체적으로는, 학습 없이 답변 정확도를 개선하고, 미세조정(fine-tuning)을 통해 모델의 고유 능력을 향상시키는 데 목표를 두었습니다.

- **Performance Highlights**: 초기 결과는 고무적입니다. (1) LbT를 통해 강한 모델이 약한 모델을 가르침으로써 자신을 개선할 수 있는 '약-강 일반화(generalization)'를 유도할 수 있음을 발견했습니다. (2) 다양한 학생을 가르치는 것이 하나의 학생 또는 자신을 가르치는 것보다 더 효과적일 수 있다는 점에서 다각성(diversity)이 도움이 될 수 있음을 발견했습니다.



### Multimodal Task Vectors Enable Many-Shot Multimodal In-Context Learning (https://arxiv.org/abs/2406.15334)
- **What's New**: 최근 Interleaved Large Multimodal Models (LMMs)의 성공은 많은 예제들을 통한 in-context learning (ICL)의 가능성을 보여줍니다. 하지만, 많은 예제들을 이용한 multimodal ICL 설정은 모델의 사전훈련 중 설정된 context 길이에 의해 제한됩니다. 이 문제를 해결하기 위해 Multimodal Task Vectors (MTV)를 활용하여 LMM의 attention heads 내부에서 in-context 예제들을 압축하는 방법을 제안합니다.

- **Technical Details**: MTV는 LMM의 attention heads 내의 활성화들을 평균 내어 얻어진 compact하고 암묵적인 표현입니다. 이 기법은 다양한 Vision-and-Language (VL) 작업에서 많은 예제들을 압축하여 사용 가능하게 합니다. 첫 번째로 많은 수의 multimodal ICL 예제들에 대한 평균 활성화를 계산하고, 두 번째로 모델의 주의(heads)를 선택하여 이 활성화를 저장합니다. 최종적으로 이 MTV를 이용하여 downstream 추론 작업을 수행합니다.

- **Performance Highlights**: MTV를 사용한 우리의 실험은 압축된 샷수가 증가함에 따라 성능이 스케일링될 수 있음을 보여주며, 추가적인 context 길이 없이도 유사한 out-of-domain 작업으로 일반화할 수 있음을 시사합니다. 중요한 이점으로, MTV는 추론 과정 중 토큰을 더욱 효율적으로 사용할 수 있게 합니다. MTV는 다양한 VL 벤치마크에서 zero-shot 및 few-shot ICL 설정보다 우수한 성능을 보이며, 추가적인 파인튜닝 없이 많은 예제를 압축할 수 있게 해줍니다.



### Gradient-Mask Tuning Elevates the Upper Limits of LLM Performanc (https://arxiv.org/abs/2406.15330)
- **What's New**: 대규모 언어 모델(LLMs)이 연구 분야에서 혁신을 일으키고 있습니다. 하지만 기존의 연구들이 Fine-Tuning 과정에서 일부 중복이 발생할 수 있음을 시사하고, 모든 파라미터를 업데이트하지 않아도 된다고 제안합니다. 이에 따라 Gradient-Mask Tuning(GMT)이라는 새로운 방법을 제안합니다. GMT는 과제별 데이터의 Gradient 정보를 활용해 업데이트할 파라미터를 선택 정확하게 합니다.

- **Technical Details**: GMT는 Gradient의 절대값을 계산해 상대적으로 크기가 작은 Gradient를 마스킹합니다. 이 방법은 추가 계산 없이 과제 별 데이터를 자연스럽게 활용하고, 다른 네트워크 파라미터의 중요성을 고려하여 더 세밀한 튜닝 제어를 가능하게 해줍니다. 이를 위해 GMT는 파라미터를 업데이트하기 전에 손실 함수의 Gradient를 계산하고, 일정한 간격 동안 Gradient를 누적합니다. 일정 기준 이하의 Gradient는 마스킹하여 업데이트에서 제외합니다.

- **Performance Highlights**: 다양한 과제를 통해 실험한 결과, GMT는 전통적인 Fine-Tuning 방법보다 뛰어난 성능을 보였으며, 모델 성능의 상한선도 높였습니다. 또한, GMT는 마스크 비율에 대해 민감하지 않으며, Vanilla SFT와 비교해도 계산 효율성이 비슷한 것으로 나타났습니다. FLOPs와 시간 효율성을 분석한 결과, 네트워크 구조를 해치지 않으면서도 효율성을 유지하는 것으로 분석되었습니다.



### STARD: A Chinese Statute Retrieval Dataset with Real Queries Issued by Non-professionals (https://arxiv.org/abs/2406.15313)
- **What's New**: 본 연구는 비전문가의 질의에 대응하는 법률 조항 검색의 중요성을 강조하며, 기존 데이터셋의 한계를 보완한 **STARD**(STAtute Retrieval Dataset)를 소개합니다. 이 데이터셋은 중국 법률 상담 사례에서 나온 1,543개의 질의와 55,348개의 후보 법률 조항으로 구성되어 있습니다. 이는 법률 자문 서비스 사용자들의 비전문적인 질의를 실제로 반영한 최초의 데이터셋입니다.

- **Technical Details**: STARD 데이터셋은 다양한 검색 베이스라인에서 평가되었습니다. 여기에는 전통적인 어휘 매칭 모델, 오픈 도메인 (open-domain) 신경 검색 모델, 법률 도메인 신경 검색 모델, 그리고 GPT-4로 주석이 달린 데이터를 사용하여 훈련된 **Dense Retriever** 등이 포함됩니다. 이 데이터는 Retrieval-Augmented Generation (RAG) 모델에서 외부 지식 원천으로 사용될 때 대형 생성 언어 모델 (Large Generative Language Models, **LLMs**)의 성능을 크게 향상시키는 것으로 나타났습니다.

- **Performance Highlights**: 실험 결과, 기존 검색 베이스라인은 비전문가의 질의에 충분히 대응하지 못하며, 최적의 방법도 Recall@100에서 0.907의 성능을 기록했습니다. 이는 비전문가 질의의 법률 조항 검색이 여전히 어렵고 추가 연구가 필요함을 시사합니다. STARD를 사용한 LLM 실험에서는 법률 과제의 성능이 현저히 향상되었음을 알 수 있습니다.



### Advanced Multimodal Deep Learning Architecture for Image-Text Matching (https://arxiv.org/abs/2406.15306)
Comments:
          arXiv admin note: text overlap with arXiv:2405.17460 by other authors

- **What's New**: 이미지-텍스트 매칭(image-text matching)는 이미지와 텍스트 간의 의미적 연관(security association)을 매칭 관계로 모델링하는 핵심 멀티모달(multimodal) 과제입니다. 이번 연구에서는 현재 멀티모달 딥러닝 모델들이 이미지-텍스트 매칭 작업을 처리하는 데 있어서의 한계를 심도 있게 분석하고, 이를 해결하기 위해 고급 멀티모달 딥러닝 아키텍처(multimodal deep learning architecture)를 혁신적으로 설계합니다. 이 아키텍처는 이미지에 대한 딥 뉴럴 네트워크(deep neural networks)의 고수준 추상화 표현 능력과 텍스트 의미 이해를 위한 자연어 처리 모델의 장점을 결합합니다.

- **Technical Details**: 새로운 아키텍처는 독창적인 크로스-모달 주의 메커니즘(cross-modal attention mechanism)과 계층적 특징 융합 전략(hierarchical feature fusion strategy)을 도입하여 이미지와 텍스트 특징 공간 간의 심층 융합과 양방향 상호작용을 달성합니다. 또한, 모델이 학습 과정에서 이미지와 텍스트 간의 잠재적 연관 구조를 더 잘 매핑할 수 있도록 훈련 목적과 손실 함수도 최적화합니다.

- **Performance Highlights**: 실험 결과, 기존 이미지-텍스트 매칭 모델들과 비교해 이 새로운 모델은 여러 벤치마크 데이터 세트에서 성능이 크게 향상되었습니다. 더 나아가 이 모델은 대규모 및 다양한 공개 시나리오 데이터 세트에서도 우수한 일반화 및 견고성을 보였으며, 이전에 보지 못한 복잡한 상황에서도 높은 매칭 성능을 유지할 수 있음을 보여주었습니다.



### Cross-Modality Safety Alignmen (https://arxiv.org/abs/2406.15279)
- **What's New**: 이 논문은 Safe Inputs but Unsafe Output (SIUO)라는 새로운 안전 정렬 과제를 소개합니다. 이는 단일 모달리티는 안전하지만 서로 결합되면 위험하거나 비윤리적인 결과를 초래할 수 있는 경우를 평가합니다. SIUO는 자해, 불법 행위, 개인정보 침해 등 9개의 중요한 안전 분야를 포함하고 있습니다.

- **Technical Details**: SIUO는 크로스 모달리티(cross-modality)를 고려하여 독립적으로 안전한 입력이 결합되면 위험이 될 수 있는 경우를 문제로 삼습니다. 이를 위해, 연구팀은 SIUO 벤치마크를 만들었으며, 이를 통해 자기상해, 불법행위, 프라이버시 위반 등의 다양한 안전 영역을 탐구했습니다.

- **Performance Highlights**: 연구 결과, GPT-4V 및 LLaVA와 같은 폐쇄형 및 오픈 소스 LVLMs 모두에서 상당한 안전 취약성이 드러났습니다. 이는 현재 모델들이 복잡하고 실제적인 시나리오를 신뢰성 있게 해석하고 대응하는 데 불충분함을 강조합니다.



### Towards Fine-Grained Citation Evaluation in Generated Text: A Comparative Analysis of Faithfulness Metrics (https://arxiv.org/abs/2406.15264)
Comments:
          12 pages, 3 figures

- **What's New**: 대규모 언어 모델(LLMs)은 종종 '환각(hallucinations)'으로 알려진 검증되지 않은 정보를 생성합니다. 이를 해결하기 위해 인용을 추가하여 신뢰할 수 있는 출처에 근거한 내용을 생성하는 retrieval-augmented LLMs가 개발되었습니다. 하지만 인용이 주어진 진술을 얼마나 잘 지원하는지 평가하는 것은 여전히 어려운 과제입니다. 이전 연구들은 faithfulness metrics를 사용하여 자동으로 인용 지원을 추정했으나, 이는 이진 분류에만 국한되어 실제 시나리오에서의 세분화된 인용 지원을 간과했습니다. 이 연구는 세분화된 시나리오에서 faithfulness metrics의 효율성을 조사하기 위해 비교 평가 프레임워크를 제안합니다.

- **Technical Details**: 본 연구는 '세분화된 지원 수준(fine-grained support levels)'을 정의하고, 인용이 진술을 얼마나 잘 지원하는지를 평가하기 위해 세 가지 평가 프로토콜을 사용합니다: 상관 분석(correlation analysis), 분류 평가(classification evaluation), 및 검색 평가(retrieval evaluation). 이러한 평가 프로토콜을 통해 metrics 점수와 인간 판단 간의 정렬을 측정합니다. 실험은 널리 사용되는 여러 faithfulness metrics를 평가하고, 이를 유사성 기반(similarity-based), 함의 기반(entailment-based), 및 LLM 기반(metrics)으로 분류하였습니다.

- **Performance Highlights**: 핵심 발견은 다음과 같습니다: 1) 어떤 단일 metrics도 모든 평가에서 일관되게 뛰어나지 않습니다. 이는 평가 프로토콜이 상호 보완적이며, 포괄적인 평가를 위해 통합되어야 함을 시사합니다. 2) 최고의 성능을 보이는 metrics는 일부 지원 시나리오에서 우수하지만 다른 시나리오에서는 어려움을 겪고 있습니다. 3) 유사성 기반 metrics는 검색 평가에서 다른 metrics보다 우수하지만, 이는 미세한 데이터에 민감하기 때문입니다. 이 연구는 세분화된 인용 지원을 식별하는 데 어려움을 겪는 자동화된 인용 평가의 복잡성을 강조합니다.



### Investigating the impact of 2D gesture representation on co-speech gesture generation (https://arxiv.org/abs/2406.15111)
Comments:
          8 pages. Paper accepted at WACAI 2024

- **What's New**: 이번 논문은 음성과 동기화되는 사실적이고 자연스러운 몸짓을 생성할 수 있는 딥 러닝 방법을 제안합니다. 이 연구는 2D 또는 3D 관절 좌표의 차원성이 모달리스피치-제스처(Multimodal Speech-to-Gesture) 딥 생성 모델의 성능에 미치는 영향을 평가합니다. 특히 'In-the-wild' 데이터셋을 활용한 연구에서, 2D 관절 데이터를 3D로 변환하여 3D 몸짓을 생성하는 방법을 제안하고, 이를 직접적으로 3D 데이터를 이용해 생성한 몸짓과 비교합니다.

- **Technical Details**: 연구는 비디오에서 추출한 2D 골격 데이터를 3D로 변환하는 리프팅 모델(lifting model)을 사용하여, 3D로 변환된 몸짓의 성능을 평가합니다. Denoising Diffusion Probabilistic Model (DDPM)을 이용하여 음성-제스처 생성 모델을 훈련시키며, 제안된 평가 파이프라인을 통해 2D와 3D 데이터의 제스처 품질을 비교합니다. 또한, 'In-the-wild' 데이터셋에서 2D 관절 데이터를 이용해 3D 동작을 추정하는 과정에서 발생하는 오류와 그로 인한 제스처의 자연스러움과 다양성 문제를 다룹니다.

- **Performance Highlights**: 실험 결과 2D 몸짓 데이터를 3D로 리프팅한 경우, 원래의 3D 몸짓 분포와 완벽하게 일치하지 않으며, 음성 및 몸짓의 일관성이 저하되고, 2D 데이터에서 리프팅한 3D 몸짓이 직접 생성한 3D 몸짓보다 다양성이 떨어진다는 결론을 얻었습니다.



### Tri-VQA: Triangular Reasoning Medical Visual Question Answering for Multi-Attribute Analysis (https://arxiv.org/abs/2406.15050)
- **What's New**: 새로운 Triangular Reasoning VQA (Tri-VQA) 프레임워크가 제안되었습니다. 이는 기존 메디컬 VQA (Med-VQA) 방식의 한계를 극복하고, 답변의 신뢰성을 강화하기 위해 '왜 이 답변인가?'라는 역방향 인과 질문을 통해 더 정교한 추론을 촉진합니다. 제안된 모델은 Endoscopic Ultrasound (EUS) 데이터셋과 기타 의료 VQA 데이터셋에서 탁월한 성능을 입증했습니다.

- **Technical Details**: Tri-VQA는 기존의 순방향 인과 질문(Q+V→A)에 더해, 역방향 인과 질문을 포함하여 답변의 출처를 명확히 하고, 올바른 추론 과정을 자극합니다. 이 프레임워크는 세 가지 종류의 추론을 포함합니다: 순방향 추론(F: fusion(Q, V)→A)과 두 가지 형태의 역방향 추론(G: fusion(A, V)→Q 및 H: fusion(A, Q)→V). 이렇게 함으로써 네트워크는 여러 컴포넌트 간 상호작용을 더 잘 이해하고 더 신뢰성 있는 답변을 도출할 수 있습니다.

- **Performance Highlights**: Tri-VQA 모델은 EUS 데이터셋에서 다속성 분석을 수행하며 기존의 Med-VQA 방식보다 뛰어난 성능을 보였습니다. 모델의 효율성은 오픈엔디드 질문이 포함된 의료 VQA 벤치마크에서도 검증되었습니다. Tri-VQA는 더 안정적이고 신뢰성 있는 추론 구조를 형성하며, 역방향 추론 정확도를 통해 답변의 신뢰성을 평가하는 지표를 제공합니다.



### Online detection and infographic explanation of spam reviews with data drift adaptation (https://arxiv.org/abs/2406.15038)
- **What's New**: 이 논문은 온라인 플랫폼에서 스팸 리뷰 식별 및 설명을 위한 새로운 실시간 솔루션을 제안합니다. 데이터 드리프트(Data Drift) 적응 기능을 통합하여 동적으로 변화하는 환경에서도 효과적인 스팸 탐지를 가능하게 합니다.

- **Technical Details**: 이 방법론은 (i) 증분 프로파일링(Incremental Profiling), (ii) 데이터 드리프트 탐지 및 적응(Data Drift Detection & Adaptation), (iii) 머신 러닝(Machine Learning)을 활용한 스팸 리뷰 식별을 통합합니다. 또한, 설명 가능한 메커니즘이 대시보드에서 시각적 및 텍스트 예측 설명을 제공합니다.

- **Performance Highlights**: 제안된 솔루션은 평가된 데이터 세트에서 최대 87%의 스팸 F-측정을 달성했습니다.



### GraLMatch: Matching Groups of Entities with Graphs and Language Models (https://arxiv.org/abs/2406.15015)
Comments:
          12 pages, 4 figures, accepted as research paper at EDBT 2025

- **What's New**: 이 논문에서는 엔드-투-엔드 다중 소스 엔터티 매칭(Entity Matching) 문제인 엔터티 그룹 매칭(entity group matching)을 소개합니다. 이 문제는 동일한 실제 엔터티를 나타내는 서로 다른 데이터 소스에서 온 레코드를 동일한 그룹으로 할당하는 것을 목표로 합니다. 특히, 그래프 G=(V,E)에서 경로로 연결된 트랜지티브 매칭(transitive matching) 레코드의 영향을 중점적으로 다룹니다. 본 연구는 기업 및 금융 증권 레코드를 매칭하는 실제 사례를 제시하며, 유사한 매칭 과제를 가진 두 개의 새로운 다중 소스 벤치마크 데이터셋을 소개합니다.

- **Technical Details**: 데이터 소스에서 업데이트가 균일하게 적용되지 않기 때문에 트랜지티브 정보를 사용한 매칭만 가능한 경우가 있습니다. 제안된 GraLMatch 방법은 그래프 기반 속성을 통해 잘못된 긍정(pairwise predictions)을 부분적으로 탐지하고 제거합니다. 또한, DistilBERT와 같은 Transformer 모델을 소량의 레이블된 샘플로 파인튜닝(fine-tuning)하여 더 나은 엔터티 그룹 매칭 결과를 도출할 수 있음을 실험을 통해 보여줍니다.

- **Performance Highlights**: 실험 결과에 따르면, 트랜지티브 매칭 레코드가 엔터티 그룹 매칭에서 중요함을 확인할 수 있으며, 이로 인해 잘못된 긍정(pairwise predictions)에 민감한 문제가 발생할 수 있음을 보여줍니다. 또한, GraLMatch를 사용하여 트랜지티브 매칭 문제를 해결함으로써 더 정교한 매칭 결과를 도출할 수 있음을 입증했습니다.



### Disability Representations: Finding Biases in Automatic Image Generation (https://arxiv.org/abs/2406.14993)
Comments:
          Presented at AVA Workshop of CVPR 2024

- **What's New**: 최근 AI를 사용한 이미지 생성 기술이 크게 발전하면서 광고, 엔터테인먼트 및 다양한 시각적 콘텐츠에 널리 활용되고 있습니다. 하지만 이러한 기술은 사회적 편견을 지속적으로 반영하고 있음을 보여줍니다. 본 연구는 유명한 텍스트-투-이미지(text-to-image) 모델에서 장애인을 대상으로 하는 표현 편견을 조사합니다.

- **Technical Details**: 이번 연구는 여러 유명 텍스트-투-이미지 모델을 이용하여 장애인의 묘사를 분석하는 광범위한 실험을 수행했습니다. 구체적으로, 생성된 이미지에서 장애인이 주로 어떻게 표현되는지에 대해 조사했습니다.

- **Performance Highlights**: 결과는 놀라울 정도로 큰 편견을 보여주었으며, 대부분의 생성된 이미지에서 장애인은 나이 들고 슬퍼 보이며, 주로 수동 휠체어(manual wheelchair)를 사용하는 모습으로 나타났습니다. 이러한 결과는 더욱 포괄적이고 정확한 장애인 표현을 보장하기 위해 AI 모델의 개선 필요성을 강조합니다. 따라서 이 연구는 AI 모델의 편견을 해결하고 완화하기 위한 중요성을 환기시키며, 보다 공평하고 현실적인 표현을 촉진할 것을 주장합니다.



### Do Large Language Models Exhibit Cognitive Dissonance? Studying the Difference Between Revealed Beliefs and Stated Answers (https://arxiv.org/abs/2406.14986)
- **What's New**: 이 연구는 대형 언어 모델(LLM, Large Language Models)의 명제적(Causal) 추론과 불확실성(uncertainty) 인식을 평가하는 데 있어서, 기존의 프롬프트(prompting) 및 객관식 질문(MCQ) 방식의 한계를 지적하고, 대안적 평가 방법인 '드러난 신념(Revealed Belief)'을 제안한다.

- **Technical Details**: 드러난 신념(Revealed Belief) 방법은 LLM이 텍스트 완료(text completion)를 통해 특정 시나리오 예상 결과에 대한 다음 토큰 예측(next token prediction)을 분석하며, 이를 통해 모델의 내부 신념을 드러내어(Stated Answer와 비교하여) 다중 가능한 결과에 대한 확률 분포를 평가한다. 이 접근 방식은 질문/응답 프레임워크를 포기하고 본질적인 텍스트 생성 능력을 강조하도록 한다.

- **Performance Highlights**: 연구 결과, LLM이 객관식 질문(MCQ) 환경에서는 괜찮은 성능을 보이지만, 드러난 신념에서 드러나는 편향(bias)과 잘못된 대표성(misrepresentation)은 고급 추론 능력과 양립하지 않으며, 이는 현재 평가 방법이 모델의 실제 능력을 제대로 반영하지 못한다는 것을 시사한다.



### Unlocking the Global Synergies in Low-Rank Adapters (https://arxiv.org/abs/2406.14956)
Comments:
          Accepted at ICML2024 ES-FoMo-II Workshop

- **What's New**: HeteroLoRA는 Low-rank Adaption (LoRA) 기법을 향상시키기 위한 경량 검색 알고리즘으로, 사전 학습된 대형 언어 모델(LLM)의 성능을 더 효과적으로 미세 조정할 수 있도록 설계되었습니다. 이 알고리즘은 zero-cost proxies를 활용하여 제한된 LoRA 학습 가능한 파라미터를 모델에 배분합니다.

- **Technical Details**: HeteroLoRA는 'LoRA rank allocation' 문제를 해결하기 위해 LoRA 모듈 내에서 계층별로 다른 순위를 할당합니다. 이것은 zero-cost proxies를 사용하여 높은 비용의 브루트 포스 검색을 피합니다. 또한, 롤아 적응형 지름길 연결 (LoRA-adapted shortcut connections)을 포함하는 확장된 검색 공간에서도 작동합니다.

- **Performance Highlights**: HeteroLoRA는 동일한 모델 크기 예산으로 성능을 향상시킬 수 있음을 실험을 통해 입증했습니다. 예를 들어, MRPC에서 유사한 학습 파라미터 예산으로 정확도가 1.6% 향상되었습니다. 본 논문이 승인되면 HeteroLoRA 알고리즘을 오픈 소스로 제공할 예정입니다.



### Autonomous Agents for Collaborative Task under Information Asymmetry (https://arxiv.org/abs/2406.14928)
Comments:
          16 pages, 8 figures, 5 tables, Work in progress

- **What's New**: 연구진은 다양한 복잡한 작업을 해결하는 데 뛰어난 성과를 보인 대형 언어 모델 멀티 에이전트 시스템(LLM-MAS)의 새로운 패러다임을 제안합니다. 기존의 MAS는 정보 비대칭 상황에서 작업 수행에 어려움을 겪었으나, 새롭게 제안된 iAgents 시스템은 정보 비대칭을 극복하도록 설계되었습니다. 이는 인간의 소셜 네트워크를 에이전트 네트워크로 투영하여 필요한 정보를 주도적으로 교환하는 새로운 에이전트 환경을 구성합니다.

- **Technical Details**: iAgents는 InfoNav라는 새로운 에이전트 추론 메커니즘을 사용하여 에이전트 간의 자율적인 정보 교환을 유도합니다. InfoNav는 에이전트의 의사소통을 효과적으로 안내하여 정보 비대칭 문제를 해결하도록 돕습니다. 또한, 새로운 메모리 메커니즘(Mixed Memory)을 도입하여 에이전트가 정확하고 종합적인 정보를 교환할 수 있도록 지원합니다. 마지막으로, 정보 비대칭 상황에서 LLM 에이전트의 작업 해결 능력을 평가하기 위해 InformativeBench라는 첫 번째 벤치마크를 소개했습니다.

- **Performance Highlights**: 실험 결과, iAgents는 140명의 개별 인물과 588개의 관계로 이루어진 소셜 네트워크 내에서 30회 이상의 대화 턴을 통해 자율적으로 소통할 수 있었으며, 70,000개에 가까운 메시지에서 정보를 검색하여 3분 이내에 작업을 완료할 수 있었습니다. 그러나 일부 최첨단 LLM 백엔드를 사용하는 에이전트는 InformativeBench에서 평균 50.48%의 정확도를 기록했으며, 가장 어려운 작업에서는 22.8%의 정확도를 기록하여 개선의 여지가 남아 있음을 시사합니다.



### LLM2FEA: Discover Novel Designs with Generative Evolutionary Multitasking (https://arxiv.org/abs/2406.14917)
Comments:
          This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 이 논문은 LLM(대형 언어 모델)과 MFEA(다중 요소 진화 알고리즘)를 결합하여 창의적이고 실용적인 디자인을 생성하는 새로운 접근 방식인 LLM2FEA를 제안합니다. 이 모델은 여러 도메인의 지식을 전이하여 새로운 디자인을 발견할 수 있도록 지원합니다. 특히, 3D 공기역학적 디자인에서 LLM2FEA가 높은 성능을 보였다는 것이 검증되었습니다.

- **Technical Details**: LLM2FEA는 세 가지 주요 컴포넌트로 구성됩니다: 
 1. Shape Generation Component - 텍스트로부터 3D 모델을 생성하는 컴포넌트.
 2. LLM-based Prompt Generation Component - 창의적이고 맥락에 맞는 텍스트 프롬프트를 생성하는 LLM.
 3. Evolutionary Multitask Searching Component - 최적의 프롬프트를 탐색하기 위해 여러 도메인의 작업을 활용하는 MFEA. 이러한 구성 요소들은 서로 협력하여 혁신적인 디자인을 발견하는 데 기여합니다.

- **Performance Highlights**: 이론적 검증을 통해 LLM2FEA는 기존의 블랙박스 최적화 알고리즘보다 더 우수한 성능을 보였으며, 공기역학적 설계 문제에서 실용적인 기준을 충족하면서도 시각적으로 매력적인 디자인을 생성하는 데 성공하였습니다. 특히, MFEA의 크로스 도메인 학습 능력을 활용하여 LLM의 창의적 탐색 능력을 크게 향상시킬 수 있었습니다.



### MoA: Mixture of Sparse Attention for Automatic Large Language Model Compression (https://arxiv.org/abs/2406.14909)
Comments:
          10 pages

- **What's New**: Mixture of Attention (MoA)는 LLM 모델의 긴 컨텍스트에 대한 메모리 및 처리량 문제를 해결하기 위해 고안된 새로운 방법론입니다. 기존의 균일한 sparse attention을 대체하여, 각 attention head와 레이어에 적합한 다양한 sparse attention 구성을 자동으로 적용합니다.

- **Technical Details**: MoA는 여러 attention 패턴과 입력 시퀀스 길이에 상대적인 스케일링 규칙을 탐색하는 공간을 구성합니다. 모델을 프로파일링하고, 잠재적인 구성을 평가하며 최적의 sparse attention 압축 계획을 도출합니다. 이를 통해 일부 attention head는 더 긴 시퀀스를 처리할 수 있도록 포커스를 확장하고, 다른 head는 고정 길이의 로컬 컨텍스트만 집중합니다.

- **Performance Highlights**: MoA는 Vicuna-7B, Vicuna-13B, 그리고 Llama3-8B 모델에서 평균 attention span이 동일한 상태에서 효과적인 컨텍스트 길이를 3.9배 증가시키고, retrieval accuracy를 1.5-7.1배 향상시켰습니다. 또한 sparse와 dense 모델 간의 성능 격차를 좁혀서, 두 개의 long-context 이해 벤치마크에서 최대 상대 성능 저하를 9"-36"에서 5" 이내로 줄였습니다. GPU 메모리 소비는 1.2-1.4배 줄이고, 디코딩 처리량을 5.5-6.7배 향상시켰습니다.



### Safely Learning with Private Data: A Federated Learning Framework for Large Language Mod (https://arxiv.org/abs/2406.14898)
- **What's New**: 새로운 연구는 FL-GLM이라는 연합 학습(federated learning) 프레임워크를 제안합니다. 이는 분할 학습(split learning)의 장점을 유지하면서도 보안과 효율성을 개선한 모델입니다. 이 프레임워크는 클라이언트 간 데이터 누수를 방지하기 위해 입력 및 출력 블록을 로컬 클라이언트에 배치하고, 클라이언트-서버 통신 중에 키 암호화를 사용합니다.

- **Technical Details**: FL-GLM은 서버에서 대부분의 파라미터를 처리하고, 클라이언트가 입력 및 출력 블록을 로컬에서 처리하도록 설계되었습니다. 클라이언트는 입력 데이터를 통해 은닉 상태를 생성하고 이를 서버에 암호화된 형태로 전송합니다. 서버는 여러 클라이언트의 은닉 상태를 병렬로 처리하고 결과를 클라이언트로 되돌려줍니다. 또한 클라이언트 배칭(client-batching)과 서버 계층화(server-hierarchical)와 같은 최적화 방법을 사용해 훈련 효율성을 향상시킵니다.

- **Performance Highlights**: SuperGLUE와 추상적 요약 데이터셋에서의 실험 결과, FL-GLM은 중앙집중식 chatGLM 모델 수준의 성능을 달성하였습니다. 또한 클라이언트 배칭과 서버 계층화 메커니즘을 통해 훈련 시간을 48% 이상 절약할 수 있음을 보여주었습니다.



### DistiLRR: Transferring Code Repair for Low-Resource Programming Languages (https://arxiv.org/abs/2406.14867)
- **What's New**: 이 논문에서는 저자들이 새로운 접근법인 Distilling Low-Resource Repairs (DistiLRR)를 제안했습니다. 이 접근법은 고자원 프로그래밍 언어(High-Resource Programming Languages, HRPL)에서 저자원 프로그래밍 언어(Low-Resource Programming Languages, LRPL)로 코드 수리 능력을 이전하는 것을 목표로 합니다. 기존의 많은 코드 생성 모델들이 Python과 같은 고자원 언어에서 우수한 성능을 보이지만, 저자원 언어에서는 성능이 저조한 문제를 해결하기 위한 시도로 볼 수 있습니다.

- **Technical Details**: DistiLRR는 코드 수리를 중심으로 하는 프레임워크로, 초기 코드 생성, 테스트 실행, 반복적 수리(iterative repair)의 단계를 포함합니다. 이 과정에서 더 큰 모델(교사 모델, teacher model)로부터 작은 모델(학생 모델, student model)로 지식 이전(Distillation)이 이루어집니다. 이는 특히 저자원 언어에서 모델이 깊은 프로그래밍 언어 지식을 갖추지 못해 코드 수정의 효율성이 떨어지는 문제를 해결하는 데 중점을 둡니다.

- **Performance Highlights**: DistiLRR를 이용한 코드 수리는 저자원 언어에서 일관되게 우수한 성능을 보였습니다. 예를 들어, HumanEval 벤치마크에서 Perl 언어의 pass@1 평균을 99.5%, Golang을 112.8%, Swift를 144.5% 향상시켰습니다. 반면, 고자원 언어에서는 유사한 성능을 보였습니다. 이는 저자원 언어에서의 코드 수리 효과가 더 두드러짐을 보여줍니다. 또한, 코드 수정 후 좋은 논리(rationale)를 제시하고도 잘못된 코드를 수정하는 비율이 저자원 언어에서는 76.4%로, 고자원 언어의 69.9%보다 높게 나타났습니다. DistiLRR는 이 비율을 크게 줄이는데 기여했습니다.

- **Conclusion**: DistiLRR는 저자원 프로그래밍 언어에서 코드 수리 성능을 획기적으로 개선할 수 있는 잠재력을 가지고 있습니다. 이는 더 큰 모델의 지식을 작은 모델로 이전함으로써, 추가적인 인간 작성 코드 없이도 효율적으로 LRPL의 코드 생성 능력을 향상시키는 유망한 접근법입니다.



### LatentExplainer: Explaining Latent Representations in Deep Generative Models with Multi-modal Foundation Models (https://arxiv.org/abs/2406.14862)
- **What's New**: 이번 논문에서는 딥 생성 모델(Deep Generative Models)에서 잠재 변수(latent variables)를 이해하는 과정에서의 도전 과제를 해결하기 위해 LatentExplainer라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 잠재 변수의 의미를 추론하고, 설명을 유도 편향(inductive biases)과 정렬하며, 설명 가능성의 다양한 정도를 처리합니다. 잠재 변수를 조정하고 생성된 데이터의 변화를 해석함으로써, LatentExplainer는 데이터를 생성하는 과정을 체계적으로 이해하고 제어할 수 있도록 지원합니다.

- **Technical Details**: LatentExplainer는 딥 생성 모델의 잠재 변수에 대해 의미 있는 설명을 자동으로 생성하는 프레임워크입니다. 잠재 변수를 조정하여 생성된 데이터의 변화를 해석함으로써 잠재 변수의 의미를 추론합니다. 또한, 사용자 제공의 유도 편향을 텍스트(prompt)로 변환하여 인간과 다중 모드 대형 언어 모델(Multimodal Large Language Models, MLLMs)이 쉽게 이해할 수 있도록 합니다. 마지막으로 설명의 불확실성을 추정하여 설명의 신뢰도를 평가하고 가장 일관된 설명을 선택함으로써 잠재 변수의 다양한 설명 가능성을 처리합니다.

- **Performance Highlights**: LatentExplainer는 여러 실제 데이터셋과 합성 데이터셋에서 평가되어, 잠재 변수의 설명 생성에 있어 높은 품질의 결과를 보여주었습니다.



### Evaluating RAG-Fusion with RAGElo: an Automated Elo-based Framework (https://arxiv.org/abs/2406.14783)
Comments:
          Accepted to LLM4Eval @ SIGIR24

- **What's New**: 이 논문은 Infineon Technologies의 제품 질의응답(QA) 작업에서 RAG-Fusion (RAGF)과 같은 Retrieval-Augmented Generation (RAG) 시스템 변형을 평가하는 문제를 해결하기 위해 새로운 평가 프레임워크를 제안합니다. 이를 통해 도메인 특정 지식에서 발생하는 헛소리 문제와 회사 내부 업무에 대한 골드 표준 벤치마크의 부족 문제를 해결하고자 합니다.

- **Technical Details**: 제안된 평가 프레임워크는 실사용자 쿼리와 도메인 문서를 기반으로 하는 큰 데이터셋의 합성 쿼리를 생성하기 위해 대형 언어 모델(LLM)을 활용합니다. 또한, LLM-as-a-judge를 통해 검색된 문서와 답변을 평가하고, 답변의 품질을 평가하며, RAGElo의 자동 Elo 기반 경쟁을 사용하여 다양한 RAG 에이전트 변형을 순위 매깁니다. RAGElo는 수집된 검색 결과와 답변을 LLM을 통해 평가하고, 각 RAG 파이프라인을 엘로 시스템을 통해 순위를 매기는 도구키트입니다.

- **Performance Highlights**: RAGF는 RAG에 비해 Elo 점수에서 우수한 성능을 보였으며, 전문 평가에 따르면 완전성 측면에서도 RAG보다 유의미하게 우수했습니다. 그러나 정밀성에서는 뒤처진다는 단점이 있었습니다. Infineon의 RAGF 어시스턴트는 문서 관련성에서 MRR@5 점수를 기준으로 약간 더 높은 성능을 나타냈습니다. RAGElo는 인간 주석자의 선호와 긍정적인 일치를 보였으나, 여전히 신중한 접근이 필요합니다.



### Evaluating Numerical Reasoning in Text-to-Image Models (https://arxiv.org/abs/2406.14774)
- **What's New**: 일반적으로 고품질 이미지를 생성할 수 있는 텍스트-이미지 생성 모델들이, 수량을 포함한 텍스트 설명을 정확하게 시각화하는 데에 제한이 있다는 것이 이번 연구의 주요 발견입니다. 모델들은 작은 숫자의 경우에만 정확한 객체 생성을 할 수 있으며, 숫자 용어가 나타나는 컨텍스트에 크게 의존하고, 숫자가 커질수록 성능이 빠르게 악화됩니다. 이 연구에서는 다양한 난이도의 수치적 추론(numerical reasoning) 작업에 대해, 다양한 텍스트-이미지 모델들을 평가했으며, GeckoNum이라는 새로운 벤치마크를 제안합니다.

- **Technical Details**: GeckoNum 벤치마크는 세 가지 작업으로 구성됩니다: 정확한 숫자 생성(exact number generation), 대략적인 숫자 생성(approximate number generation), 부분 수량 추론(partial quantities reasoning). 다양한 템플릿을 사용하여 문장 구조, 숫자 단어가 나타나는 컨텍스트, 프롬프트 내의 속성/객체의 수 등 다양한 변수를 제어합니다. 연구팀은 DALL·E 3, Imagen, Muse 모델들에서 생성된 이미지와 텍스트 프롬프트를 기준으로, 인간 주석(annotation)을 통해 모델의 성능을 평가했습니다.

- **Performance Highlights**: 연구 결과, 최신 텍스트-이미지 모델들은 기본적인 수치적 추론(sills) 능력만을 가지고 있으며, 작은 정확한 수량을 생성할 때 가장 정확했습니다. GeckoNum 벤치마크는 높은 이미지 품질을 자랑하는 강력한 모델들(예: Imagen-D, DALL·E 3) 간의 차이를 구분하는 데 유용성이 입증되었습니다. 또한, 연구는 수량 평가 자동화 기준 개발 및 사전 학습된 비전-언어 모델의 평가 및 개선 분야에서의 진전을 촉진할 수 있음을 보여주었습니다.



### ChatGPT as Research Scientist: Probing GPT's Capabilities as a Research Librarian, Research Ethicist, Data Generator and Data Predictor (https://arxiv.org/abs/2406.14765)
Comments:
          Main article is 14 pages, 1 table. Includes SI Appendix: 26 pages, 12 tables, 2 figures. Total: 40 pages, 13 tables, 2 figures. Under revised review at PNAS

- **What's New**: 이 논문에서는 GPT-3.5와 GPT-4의 과학적 연구 수행 능력을 체계적으로 평가하였습니다. 연구 분야로 심리학을 선택하여 네 가지 주요 연구 과정을 조사했습니다: 연구 사서(Research Librarian), 연구 윤리학자(Research Ethicist), 데이터 생성자(Data Generator), 그리고 새로운 데이터 예측자(Novel Data Predictor)로서의 역할 수행 능력을 분석하였습니다.

- **Technical Details**: 1. 연구 사서(Research Librarian): GPT-4와 GPT-3.5는 특정 주제의 문헌 검토를 수행하는데, GPT-3.5는 36.0%의 확률로 가짜 참조를 생성한 반면, GPT-4는 5.4%로 상대적으로 낮았습니다. GPT-4는 자신의 오류를 인식하고 있음을 보여주었습니다.
2. 연구 윤리학자(Research Ethicist): GPT-4는 연구 프로토콜의 p-hacking과 같은 위반을 감지하는 능력이 뛰어나, 명확한 문제의 88.6%, 미묘한 문제의 72.6%를 수정했습니다.
3. 데이터 생성자(Data Generator): GPT 모델은 문화적 편향 패턴을 일관되게 재현하여, 가설 생성 등 데이터 생성의 가능성을 보여주었습니다.
4. 새로운 데이터 예측자(Novel Data Predictor): 두 모델 모두 훈련 데이터에 없는 새로운 결과 예측에 실패했습니다.

- **Performance Highlights**: GPT-4는 연구 사서 역할 수행 능력에서 GPT-3.5보다 훨씬 적은 가짜 참조를 생성하며 발전된 모습을 보였습니다. 연구 윤리학자로서도 GPT-4는 뛰어난 성능을 보였으며, 데이터 생성에서는 기존의 문화적 편향 패턴을 재현하는 능력을 보여주었습니다. 그러나 새로운 데이터를 예측하는 능력에서는 두 모델 모두 한계가 확인되었습니다.



### RE-AdaptIR: Improving Information Retrieval through Reverse Engineered Adaptation (https://arxiv.org/abs/2406.14764)
- **What's New**: 최근 정보 검색(IR) 벤치마크에서 뛰어난 성능을 보이는 대형 언어 모델(LLMs)의 뒤집어진 적응(RE-AdaptIR)을 활용하여 레이블이 없는 데이터만으로 성능을 향상시키는 방법을 탐구합니다. 특히, 이 접근법은 학습 도메인 내뿐만 아니라 쿼리를 전혀 본 적이 없는 도메인에서도 성능 향상을 실현합니다.

- **Technical Details**: RE-AdaptIR 접근법을 통해 LLM 기반 IR 모델을 개선했습니다. 이 모델은 레이블이 붙은 예제들 없이 정보 검색 작업을 수행할 수 있도록 설계되었습니다. 이를 위해 RepLLaMA와 e5-Mistral 두 가지 최신 텍스트 검색 모델에 RE-AdaptIR을 적용했습니다. RE-AdaptIR은 모델의 이전 지식 학습을 방해하지 않으면서 새로운 지식 어댑터를 활용해 모델을 재조정합니다.

- **Performance Highlights**: 총 14개의 데이터셋에서 도메인 내 및 제로샷 환경 모두에서 성능 향상을 보였습니다. 특히 MS-MARCO와 BeIR IR 벤치마크에서 RepLLaMA와 e5-Mistral 모델의 검색 성능을 뛰어넘는 결과를 나타냈습니다. 이는 레이블이 없는 데이터를 효과적으로 활용하여 기존 모델의 성능을 유지하거나 향상시킬 수 있음을 보여줍니다.



### Speech Prefix-Tuning with RNNT Loss for Improving LLM Predictions (https://arxiv.org/abs/2406.14701)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)을 적용한 자동 음성 인식(ASR) 제약을 해결하는 방법에 대해 다룹니다. 최근 연구들은 prefixLM 유형의 모델을 사용하여 음성을 LLM의 prefix로 직접 적용하여 ASR을 수행하고 있습니다. 본 연구에서는 RNNT 손실(RNNT loss)을 사용하여 음성 prefix-tuning을 수행하는 것을 제안합니다. 또한 고정된 LLM에 언어 기반 소프트 프롬프트(language-based soft prompting)를 적용하여 성능을 향상시킵니다. 10개의 인도어 데이터셋에 대한 실험 결과, 제안된 음성 prefix-tuning이 고정된 LLM과 미세 조정된 LLM 모두에서 성능 향상을 달성함을 입증했습니다.

- **Technical Details**: 제안된 방법은 prefixLM 모델의 음성 prefix 토큰을 ASR 손실로 튜닝하여 인식 성능을 향상시키는 것입니다. RNNT 손실을 사용하여 음성 인코더와 prefix 임베딩을 업데이트하여 더 구별되는 음성 특징을 학습할 수 있도록 했습니다. 입력된 음성 시퀀스와 텍스트 시퀀스를 결합하여 prefixLM 모델이 양방향(attention)으로 텍스트 예측을 수행하도록 합니다. 언어 ID 기반 소프트 프롬프트(langID soft prompting) 기술을 통해 고정된 LLM의 성능을 강화했습니다. 또한 언어 모델의 예측 오류를 줄이기 위해 RNNT와 LLM을 통합하는 방법을 제안했습니다.

- **Performance Highlights**: 제안된 음성 prefix-tuning과 langID 기반 소프트 프롬프트 기술을 적용한 결과, 10개의 인도어 테스트 세트 평균 WER에서 미세 조정된 LLM 대비 12%의 상대적 성능 향상을 이루었습니다. 또한, 고정된 LLM에 대한 제안된 접근법은 기본 소프트 프롬프트 prefixLM에 비해 31%의 상대적 성능 향상을 달성했습니다.



### TAGLAS: An atlas of text-attributed graph datasets in the era of large graph and language models (https://arxiv.org/abs/2406.14683)
Comments:
          Preprint

- **What's New**: 이번 보고서에서는 텍스트 속성 그래프(Text Attributed Graph, TAG) 데이터셋 및 벤치마크의 아틀라스(TAGLAS)를 소개합니다. TAGLAS는 인용 그래프에서 분자 그래프에 이르는 다양한 도메인의 23개 이상의 TAG 데이터셋을 통합하여 제공합니다. TAGLAS는 그래프 모델이 여러 도메인의 다양한 데이터셋에서 동시에 학습되고 평가될 수 있도록 지원합니다.

- **Technical Details**: TAGLAS에는 통일된 노드 및 엣지 텍스트 속성 형식의 데이터셋이 포함되어 있으며, 텍스트를 임베딩(embedding)으로 변환하거나 그래프를 텍스트로 다시 변환하는 유틸리티 도구가 제공됩니다. 또한, 모든 데이터셋과 작업을 표준화하고 효율적이며 단순하게 로드할 수 있는 방법도 제공합니다. 쉽게 사용할 수 있는 평가 유틸리티도 포함되어 있습니다.

- **Performance Highlights**: TAGLAS는 그래프-언어 또는 그래프 기반 모델의 훈련과 평가를 용이하게 합니다. 프로젝트는 오픈 소스로 공개되어 있으며, 앞으로 더 많은 데이터셋과 기능이 추가될 예정입니다.



### Holistic Evaluation for Interleaved Text-and-Image Generation (https://arxiv.org/abs/2406.14643)
Comments:
          Work in progress. 13 pages, 5 figure, 6 tables

- **What's New**: InterleavedBench는 텍스트와 이미지가 임의 순서로 혼합된 출력을 생성하는 모델을 평가하는 첫 번째 종합 벤치마크입니다. 또한 InterleavedEval이라는 GPT-4o 기반 참조 없는 메트릭을 도입하여 정확하고 설명 가능한 평가를 제공합니다.

- **Technical Details**: InterleavedBench는 다양한 실제 사용 사례를 포괄하는 고품질의 다양한 시나리오를 포함합니다. 평가를 지원하기 위해 다섯 가지 필수 평가 측면(텍스트 품질, 지각적 품질, 이미지 일관성, 텍스트-이미지 일관성 및 도움)을 정의했습니다. InterleavedEval은 GPT-4o를 사용하여 세밀한 평가를 수행하며 자세한 설명을 제공합니다.

- **Performance Highlights**: 광범위한 실험 및 엄격한 인간 평가를 통해 InterleavedEval이 기존의 참조 기반 메트릭을 능가하는 인간 판단과의 강한 상관관계를 보여줍니다. 또한, 현재 통합된 LMM들이 InterleavedBench에서 생성된 출력의 품질에 큰 도전을 받고 있음을 발견했습니다.



### Safety Verification of Wait-Only Non-Blocking Broadcast Protocols (https://arxiv.org/abs/2403.18591)
Comments:
          Long version of a paper accepted to PetriNets 2024

- **What's New**: 이번 연구에서는 프로세스 네트워크에서 동기적으로 통신하는 두 가지 방법(브로드캐스트와 단일 전송)을 통해 동작하는 알고리즘을 분석하였습니다. 이러한 알고리즘이 'Wait-Only' 프로토콜 하에서는 복잡성이 감소함을 보여줍니다.

- **Technical Details**: 일반적인 경우 이 문제들은 Ackermann-hard 복잡성을 가지지만, 프로토콜이 'Wait-Only'인 경우 상태 커버리티 문제(state coverability problem)는 P 복잡성 클래스로, 구성 커버리티 문제(configuration coverability problem)는 PSPACE 복잡성 클래스로 복잡성이 감소합니다. 'Wait-Only' 프로토콜은 어떤 상태에서도 메시지를 보내고 받지 않는 상태를 갖지 않는 프로토콜을 의미합니다.

- **Performance Highlights**: 기존의 Ackermann-hard 문제로 알려져 있던 상태 및 구성 커버리티 문제가 'Wait-Only' 프로토콜을 사용하면 각각 P와 PSPACE 복잡성 클래스로 해결 가능함을 입증하였습니다. 이는 특정 조건 하에서 문제를 더 효율적으로 해결할 수 있는 가능성을 열어줍니다.



### [WIP] Jailbreak Paradox: The Achilles' Heel of LLMs (https://arxiv.org/abs/2406.12702)
- **What's New**: 이번 연구에서는 기초 모델(foundation models)의 탈옥(jailbreak)에 관련된 두 가지 역설(paradox)을 소개합니다. 첫 번째는 완벽한 탈옥 분류기를 만들 수 없다는 것이고, 두 번째는 더 약한 모델이 더 강한 모델이 탈옥되었는지 여부를 일관되게 감지할 수 없다는 것입니다. Llama와 GPT-4o를 사용한 사례 연구와 이론적인 증명을 통해 이러한 역설을 설명합니다.

- **Technical Details**: 이 연구는 언어 모델 G가 주어진 프롬프트를 받아 출력을 생성하는 시스템을 사용합니다. 주된 개념은 '친절하고 해롭지 않은'(helpful and harmless)을 목표로 한 정렬된 모델과 이를 탈옥시키는 '탈옥 프롬프트'입니다. 정렬된 언어 모델은 악의적인 프롬프트에도 불구하고 해로운 출력을 피해야 합니다. 그러나 정렬세금(alignment taxes)이나 정의의 모호성 등으로 인해 완벽한 정렬은 불가능합니다. 이러한 상황에서 연구는 두 가지 주요 역설을 증명합니다: 완벽한 탈옥 분류기의 불가능성과 더 약한 모델이 더 강한 모델의 탈옥 여부를 감지할 수 없다는 것입니다.

- **Performance Highlights**: Llama-2와 GPT-4o를 사용한 실험에서 연구의 주장을 뒷받침하는 결과가 나타났습니다. 즉, 강력한 모델일수록 탈옥이 더 쉬워질 수 있다는 역설적인 결론에 도달했습니다. 이로 인해 약한 모델에서는 자동 벤치마킹이 유용할 수 있지만, 강력한 모델에서는 본질적으로 신뢰할 수 없다는 결론이 도출됩니다.



New uploads on arXiv(cs.IR)

### STARD: A Chinese Statute Retrieval Dataset with Real Queries Issued by Non-professionals (https://arxiv.org/abs/2406.15313)
- **What's New**: 본 연구는 비전문가의 질의에 대응하는 법률 조항 검색의 중요성을 강조하며, 기존 데이터셋의 한계를 보완한 **STARD**(STAtute Retrieval Dataset)를 소개합니다. 이 데이터셋은 중국 법률 상담 사례에서 나온 1,543개의 질의와 55,348개의 후보 법률 조항으로 구성되어 있습니다. 이는 법률 자문 서비스 사용자들의 비전문적인 질의를 실제로 반영한 최초의 데이터셋입니다.

- **Technical Details**: STARD 데이터셋은 다양한 검색 베이스라인에서 평가되었습니다. 여기에는 전통적인 어휘 매칭 모델, 오픈 도메인 (open-domain) 신경 검색 모델, 법률 도메인 신경 검색 모델, 그리고 GPT-4로 주석이 달린 데이터를 사용하여 훈련된 **Dense Retriever** 등이 포함됩니다. 이 데이터는 Retrieval-Augmented Generation (RAG) 모델에서 외부 지식 원천으로 사용될 때 대형 생성 언어 모델 (Large Generative Language Models, **LLMs**)의 성능을 크게 향상시키는 것으로 나타났습니다.

- **Performance Highlights**: 실험 결과, 기존 검색 베이스라인은 비전문가의 질의에 충분히 대응하지 못하며, 최적의 방법도 Recall@100에서 0.907의 성능을 기록했습니다. 이는 비전문가 질의의 법률 조항 검색이 여전히 어렵고 추가 연구가 필요함을 시사합니다. STARD를 사용한 LLM 실험에서는 법률 과제의 성능이 현저히 향상되었음을 알 수 있습니다.



### Towards Fine-Grained Citation Evaluation in Generated Text: A Comparative Analysis of Faithfulness Metrics (https://arxiv.org/abs/2406.15264)
Comments:
          12 pages, 3 figures

- **What's New**: 대규모 언어 모델(LLMs)은 종종 '환각(hallucinations)'으로 알려진 검증되지 않은 정보를 생성합니다. 이를 해결하기 위해 인용을 추가하여 신뢰할 수 있는 출처에 근거한 내용을 생성하는 retrieval-augmented LLMs가 개발되었습니다. 하지만 인용이 주어진 진술을 얼마나 잘 지원하는지 평가하는 것은 여전히 어려운 과제입니다. 이전 연구들은 faithfulness metrics를 사용하여 자동으로 인용 지원을 추정했으나, 이는 이진 분류에만 국한되어 실제 시나리오에서의 세분화된 인용 지원을 간과했습니다. 이 연구는 세분화된 시나리오에서 faithfulness metrics의 효율성을 조사하기 위해 비교 평가 프레임워크를 제안합니다.

- **Technical Details**: 본 연구는 '세분화된 지원 수준(fine-grained support levels)'을 정의하고, 인용이 진술을 얼마나 잘 지원하는지를 평가하기 위해 세 가지 평가 프로토콜을 사용합니다: 상관 분석(correlation analysis), 분류 평가(classification evaluation), 및 검색 평가(retrieval evaluation). 이러한 평가 프로토콜을 통해 metrics 점수와 인간 판단 간의 정렬을 측정합니다. 실험은 널리 사용되는 여러 faithfulness metrics를 평가하고, 이를 유사성 기반(similarity-based), 함의 기반(entailment-based), 및 LLM 기반(metrics)으로 분류하였습니다.

- **Performance Highlights**: 핵심 발견은 다음과 같습니다: 1) 어떤 단일 metrics도 모든 평가에서 일관되게 뛰어나지 않습니다. 이는 평가 프로토콜이 상호 보완적이며, 포괄적인 평가를 위해 통합되어야 함을 시사합니다. 2) 최고의 성능을 보이는 metrics는 일부 지원 시나리오에서 우수하지만 다른 시나리오에서는 어려움을 겪고 있습니다. 3) 유사성 기반 metrics는 검색 평가에서 다른 metrics보다 우수하지만, 이는 미세한 데이터에 민감하기 때문입니다. 이 연구는 세분화된 인용 지원을 식별하는 데 어려움을 겪는 자동화된 인용 평가의 복잡성을 강조합니다.



### Retrieval Augmented Zero-Shot Text Classification (https://arxiv.org/abs/2406.15241)
Comments:
          Proceedings of the 2024 ACM SIGIR International Conference on the Theory of Information Retrieval (ICTIR '24), July 13, 2024, Washington DC, DC, USA

- **What's New**: Zero-shot 텍스트 학습은 미리 보지 못한 클래스도 효율적으로 처리할 수 있게 하며, 특정 작업에 맞춘 학습 데이터의 필요성을 줄입니다. 전통적인 방법은 쿼리(query) 텍스트와 잠재적 클래스의 임베딩(embeddings)을 비교하여 분류하는 방식입니다. 그러나 쿼리의 단순한 임베딩은 때때로 풍부한 문맥 정보를 결여하여 분류 성능이 저하될 수 있습니다. 이를 해결하기 위해 QZero라는 새로운 학습 불필요(Training-free) 지식 증강 접근 방식을 도입했습니다. QZero는 위키피디아(Wikipedia)에서 지원 카테고리(supporting categories)를 검색하여 쿼리를 재구성함으로써 Zero-shot 텍스트 분류 성능을 향상시킵니다.

- **Technical Details**: QZero는 기존의 비싸고 시간이 많이 소요되는 임베딩 모델을 재훈련하지 않고도, 다양한 데이터셋에서 성능을 향상시킵니다. QZero는 쿼리의 문맥을 보강하기 위해 위키피디아에서 관련 카테고리를 검색하여 이를 텍스트에 추가합니다. 이로 인해 임베딩 기반 Zero-shot 텍스트 분류기의 성능이 향상되며, 특히 반복적 진화가 일어나는 정보 영역과 자원이 제한된 환경에서 유용합니다. 기계 학습 모델의 예측을 이해하는 데 도움이 되는 유의미한 인사이트(insights)도 제공합니다.

- **Performance Highlights**: QZero는 뉴스(news)와 의학 주제 분류(medical topic classification) 작업에서, 가장 큰 OpenAI 임베딩 모델의 성능을 각각 최소한 5%와 3% 개선합니다. 또한, 작은 단어 임베딩 모델들이 더 큰 문맥 모델들과 비슷한 성능을 얻을 수 있게 해 줌으로써 상당한 컴퓨팅 절약(computational savings)을 제공합니다. 이로써 QZero는 기존 임베딩 기반 zero-shot 분류기를 더욱 간단하면서도 성능을 높입니다.



### \'Evaluation des capacit\'es de r\'eponse de larges mod\`eles de langage (LLM) pour des questions d'historiens (https://arxiv.org/abs/2406.15173)
Comments:
          in French language

- **What's New**: 이 연구는 여러 Large Language Models(LLMs)의 역사적 사실에 대한 프랑스어 답변 생성 능력을 평가한 것입니다. 다양한 유형과 주제, 난이도의 역사 관련 질문들로 구성된 시험지를 통해 열 개의 LLM을 평가하였습니다.

- **Technical Details**: 연구에서 사용된 시험지는 총 62개의 질문으로 구성되었습니다. 각각의 질문에 대해 답변 가능성을 특징지었으며, 열 개의 적합한 LLM이 선택되어 평가되었습니다. 평가 결과는 LLM의 내용 및 형식 면에서 여러 한계를 드러내었습니다. 평가에는 주로 ChatGPT와 Bard와 같은 최신 LLM이 사용되었으며, 이들의 답변은 각기 다른 정확도와 일관성을 보였습니다.

- **Performance Highlights**: 평가 결과에 따르면, 전체적인 정확도는 충분하지 않았으며, 프랑스어에 대한 비균등한 처리와 답변의 일관성 문제 등이 부각되었습니다. 특히, 역사 분야에서 LLM이 정확하고 포괄적인 답변을 제공하는 데 있어서 많은 한계가 있다는 점이 확인되었습니다.



### IDentity with Locality: An ideal hash for gene sequence search (https://arxiv.org/abs/2406.14901)
Comments:
          13 pages

- **What's New**: 이 논문에서는 기존의 해시 함수(Random Hash, RH) 기반 유전자 탐색 시스템의 성능 저하 문제를 해결하기 위해 새로운 해시 함수인 Identity with Locality(IDL) 해시 패밀리를 제안합니다. IDL 해시 함수는 입력값이 가까운 키들을 충돌 없이 인접하게 배치함으로써 캐시 효율성을 극대화하였습니다.

- **Technical Details**: 유전자 서열 검색 문제는 주어진 유전자 서열의 각 서브스트링(kmer)이 전체 유전자 데이터베이스에 존재하는지를 테스트하는 멤버십 문제로 캐스팅됩니다. 기존의 RH 함수는 탐색 쿼리에 무관하게 kmers를 무작위로 분포시켜 높은 캐시 미스와 시스템 성능 저하를 초래했습니다. IDL 해시는 유사한 입력값을 공간상 인접하게 배치하면서도 충돌을 피할 수 있도록 설계되었습니다. 이는 RH와 LSH(Locality-Sensitive Hash)의 장점을 결합한 것입니다.

- **Performance Highlights**: IDL 해시를 적용한 Bloom Filter(IDL-BF)는 캐시 미스를 약 5배 감소시켰으며, 최첨단(COBS, RAMBO) 유전자 탐색 시스템의 쿼리 시간과 인덱싱 시간을 각각 최대 2배까지 개선했습니다.



### Decoding Matters: Addressing Amplification Bias and Homogeneity Issue for LLM-based Recommendation (https://arxiv.org/abs/2406.14900)
- **What's New**: 최근의 연구는 대형 언어 모델(LLM)을 추천 시스템에 적응시키는 새로운 디코딩 접근 방식, 'Debiasing-Diversifying Decoding(D3)'을 소개합니다. D3는 기존 LLM의 디코딩 방식에서 발생하는 증폭 편향(amplification bias)과 동질성 문제(homogeneity issue)를 해결하는 것을 목표로 합니다.

- **Technical Details**: 'Debiasing-Diversifying Decoding(D3)' 방법은 두 가지 주요 메커니즘을 도입합니다. 첫째, 고스트 토큰(ghost token)이라고 불리는 특정 토큰에 대해 길이 정규화를 비활성화하여 증폭 편향을 줄입니다. 둘째, 텍스트가 아닌 보조 모델(text-free assistant model)을 통해 LLM이 자주 생성하지 않는 토큰을 장려하여 동질성을 완화합니다.

- **Performance Highlights**: 실제 데이터셋을 활용한 광범위한 실험 결과, 새로운 D3 방법이 추천의 정확성과 다양성을 크게 향상시킨다는 것을 입증했습니다. D3는 기존의 방법이 가진 한계를 효과적으로 극복하고 더 나은 추천을 생성할 수 있는 가능성을 보여줍니다.



### Evaluating RAG-Fusion with RAGElo: an Automated Elo-based Framework (https://arxiv.org/abs/2406.14783)
Comments:
          Accepted to LLM4Eval @ SIGIR24

- **What's New**: 이 논문은 Infineon Technologies의 제품 질의응답(QA) 작업에서 RAG-Fusion (RAGF)과 같은 Retrieval-Augmented Generation (RAG) 시스템 변형을 평가하는 문제를 해결하기 위해 새로운 평가 프레임워크를 제안합니다. 이를 통해 도메인 특정 지식에서 발생하는 헛소리 문제와 회사 내부 업무에 대한 골드 표준 벤치마크의 부족 문제를 해결하고자 합니다.

- **Technical Details**: 제안된 평가 프레임워크는 실사용자 쿼리와 도메인 문서를 기반으로 하는 큰 데이터셋의 합성 쿼리를 생성하기 위해 대형 언어 모델(LLM)을 활용합니다. 또한, LLM-as-a-judge를 통해 검색된 문서와 답변을 평가하고, 답변의 품질을 평가하며, RAGElo의 자동 Elo 기반 경쟁을 사용하여 다양한 RAG 에이전트 변형을 순위 매깁니다. RAGElo는 수집된 검색 결과와 답변을 LLM을 통해 평가하고, 각 RAG 파이프라인을 엘로 시스템을 통해 순위를 매기는 도구키트입니다.

- **Performance Highlights**: RAGF는 RAG에 비해 Elo 점수에서 우수한 성능을 보였으며, 전문 평가에 따르면 완전성 측면에서도 RAG보다 유의미하게 우수했습니다. 그러나 정밀성에서는 뒤처진다는 단점이 있었습니다. Infineon의 RAGF 어시스턴트는 문서 관련성에서 MRR@5 점수를 기준으로 약간 더 높은 성능을 나타냈습니다. RAGElo는 인간 주석자의 선호와 긍정적인 일치를 보였으나, 여전히 신중한 접근이 필요합니다.



### RE-AdaptIR: Improving Information Retrieval through Reverse Engineered Adaptation (https://arxiv.org/abs/2406.14764)
- **What's New**: 최근 정보 검색(IR) 벤치마크에서 뛰어난 성능을 보이는 대형 언어 모델(LLMs)의 뒤집어진 적응(RE-AdaptIR)을 활용하여 레이블이 없는 데이터만으로 성능을 향상시키는 방법을 탐구합니다. 특히, 이 접근법은 학습 도메인 내뿐만 아니라 쿼리를 전혀 본 적이 없는 도메인에서도 성능 향상을 실현합니다.

- **Technical Details**: RE-AdaptIR 접근법을 통해 LLM 기반 IR 모델을 개선했습니다. 이 모델은 레이블이 붙은 예제들 없이 정보 검색 작업을 수행할 수 있도록 설계되었습니다. 이를 위해 RepLLaMA와 e5-Mistral 두 가지 최신 텍스트 검색 모델에 RE-AdaptIR을 적용했습니다. RE-AdaptIR은 모델의 이전 지식 학습을 방해하지 않으면서 새로운 지식 어댑터를 활용해 모델을 재조정합니다.

- **Performance Highlights**: 총 14개의 데이터셋에서 도메인 내 및 제로샷 환경 모두에서 성능 향상을 보였습니다. 특히 MS-MARCO와 BeIR IR 벤치마크에서 RepLLaMA와 e5-Mistral 모델의 검색 성능을 뛰어넘는 결과를 나타냈습니다. 이는 레이블이 없는 데이터를 효과적으로 활용하여 기존 모델의 성능을 유지하거나 향상시킬 수 있음을 보여줍니다.



### UDA: A Benchmark Suite for Retrieval Augmented Generation in Real-world Document Analysis (https://arxiv.org/abs/2406.15187)
- **What's New**: 이번 논문에서는 Document Analysis(문서 분석)의 새로운 벤치마크 세트, UDA(Unstructured Document Analysis)를 소개합니다. UDA는 2,965개의 실제 문서와 29,590개의 전문적인 Q&A 페어를 포함하고 있습니다. 이를 통해 문서 분석의 다양한 설계 선택과 응답 품질을 평가할 수 있습니다.

- **Technical Details**: UDA는 재무, 학술, 지식 기본 세 가지 중요한 도메인에 걸쳐 여섯 개의 하위 데이터 세트를 포함합니다. 원시 문서들은 HTML이나 PDF 형식으로 주어지며, 잘 정리되지 않은 형태로 제공됩니다. 이를 통해 다양한 데이터 패턴을 대응할 수 있습니다. RAG(Retrieval-Augmented Generation) 기반의 해결책들을 재평가하여 다단계 절차의 성능을 측정합니다.

- **Performance Highlights**: 체인오브생각(Chain-of-Thought) 접근 방식은 숫자 문서 분석에서 응답 품질을 향상시키는 반면, 장문 문맥 LLM(long-context LLM)은 그러한 작업에서 부족한 성능을 보였습니다. RAG 기반 솔루션은 예측과 결합된 데이터 추출 및 생성 전략에서 더 유리한 결과를 보였으며, 소형 검색 모델이 특정 응용에서는 합리적으로 잘 작동한다는 점을 발견했습니다.



### Hierarchical thematic classification of major conference proceedings (https://arxiv.org/abs/2406.14983)
- **What's New**: 이번 논문에서는 계층적 텍스트 분류(hierarchical text classification)를 위한 의사 결정 지원 시스템을 개발했습니다. 전문가가 제공한 트리 형태의 주제 계층 구조가 있는 텍스트 컬렉션을 대상으로 문서와 관련된 주제를 정렬하여 주제를 선택할 수 있도록 도와줍니다. 이 시스템은 가중치 계층적 유사성 함수(weighted hierarchical similarity function)를 사용해 주제의 관련성을 계산하는 데 중점을 두고 있습니다. 이를 통해 EURO 주요 회의와 산업 회사 웹사이트 컬렉션의 추상 컬렉션에서 랭킹 정확도를 향상시켰습니다.

- **Technical Details**: 가중치 계층적 유사성 함수를 통해 문서와 트리 브랜치의 유사성을 계산합니다. 이 함수는 단어의 중요도를 결정하기 위한 가중치를 포함하며, 단어의 엔트로피(entropy)를 사용하여 가중치를 추정합니다. 계층적 주제 분류 확률 모델을 공식화하고, 변형 베이지안 추론(variational Bayesian inference)을 사용하여 EM 알고리즘을 도출하여 주어진 문서에 대한 주제의 확률을 추정합니다. 본 논문은 계층적 멀티 클래스 SVM(hierarchical multiclass SVM), 적응형 정규화를 가진 계층적 PLSA(hierarchical PLSA with adaptive regularization), 그리고 계층적 나이브 베이즈(hierarchical naive Bayes)와 비교했을 때 유의미한 성능 향상을 보여줍니다.

- **Performance Highlights**: 가중치 계층적 유사성 함수는 EURO 주요 회의와 산업 회사 웹사이트의 추상 컬렉션을 대상으로 한 실험에서 랭킹 정확도 측면에서 더 나은 개선을 보였습니다. 이 시스템은 전문가의 모형을 통해 추정된 파라미터를 이용하여 이전에 사용된 분류 방법들과 비교했을 때 더 높은 성능을 나타냈습니다.



### A Tale of Trust and Accuracy: Base vs. Instruct LLMs in RAG Systems (https://arxiv.org/abs/2406.14972)
- **What's New**: 이번 연구에서는 Retrieval Augmented Generation (RAG) 시스템에서 'instructed'로 미세 조정된 대형 언어 모델(Large Language Models, LLMs)이 아닌, 'base' 모델이 평균적으로 20% 더 좋은 성능을 보인다는 결과를 밝혔습니다. 이는 RAG 작업에서 'instructed' LLMs의 우수성이 당연하다는 기존의 가정을 재고하게 합니다.

- **Technical Details**: RAG는 데이터 검색과 생성 과정이 결합된 혁신적인 AI 접근 방식입니다. 기존의 'instructed' 모델은 감독 학습과 인간 피드백을 통해 지시를 따르는 능력과 인간의 선호도에 맞춰 조정되어 있지만, 본 연구에서는 'base' 모델이 이러한 추가 조정 없이 더 우수한 성능을 보였습니다. 'base' 모델은 주로 다음 토큰 예측(next token prediction) 작업을 통해 사전 훈련되며, 이는 광범위한 텍스트 데이터를 처리하여 언어, 구문, 의미론 및 일반 지식을 습득하게 합니다. 이와 비교하여 'instructed' 모델은 감독 학습(fine-tuning) 및 인간 피드백을 통해 '지시'를 따르는 능력을 배양합니다.

- **Performance Highlights**: 본 연구의 실험 설정에서 'base' 모델이 RAG 작업에서 'instructed' 모델을 평균 20% 뛰어넘는다는 결과가 나왔습니다. 추가적인 분석을 통해 이 차이가 단순한 것 이상의 복잡성을 띄고 있음을 발견했으며, RAG의 방법론 및 평가 절차에 대한 재검토와 더 넓은 논의가 필요함을 시사합니다.



### Talking the Talk Does Not Entail Walking the Walk: On the Limits of Large Language Models in Lexical Entailment Recognition (https://arxiv.org/abs/2406.14894)
- **What's New**: 본 연구는 여덟 개의 대형 언어 모델(Large Language Models, LLMs)이 동사 간의 어휘 포함 관계(lexical entailment relations)를 인식하는 능력을 조사했습니다. 특히, 두 개의 어휘 데이터베이스 WordNet과 HyperLex에서 동사 쌍을 통해 제로샷(zero-shot) 및 몇샷(few-shot) 설정을 통해 다양한 프롬프트 전략(prompting strategies)을 검토하였습니다.

- **Technical Details**: 연구에서는 LLMs가 어휘 포함 인식 작업에서 적당히 좋은 성능을 보이지만, 효과성 정도와 조건에 따라 성능이 다양하다는 것을 밝혔습니다. 또한 몇샷 프롬프팅(few-shot prompting)을 활용하면 모델의 성능을 향상시킬 수 있음을 발견했습니다.

- **Performance Highlights**: 모든 연구된 LLMs가 완벽하게 과제를 해결하지는 못했고, 이는 향후 연구 개발이 더 필요함을 시사합니다.



### Generate-then-Ground in Retrieval-Augmented Generation for Multi-hop Question Answering (https://arxiv.org/abs/2406.14891)
Comments:
          ACL 2024 (main conference)

- **What's New**: 본 논문에서는 다중 단계 질문 답변(MHQA) 작업에서 발생하는 문제를 해결하기 위해 새로운 '생성-이후에-기반'(Generate-then-Ground, GenGround) 프레임워크를 제안합니다. 기존의 '검색-이후에-읽기'(Retrieve-then-Read) 패러다임과 달리, 이 방법은 대형 언어 모델(LLMs)의 내재된 지식과 외부 문서를 결합하여 다중 단계 질문을 해결합니다.

- **Technical Details**: GenGround는 두 가지 주요 단계를 반복하여 최종 답변을 도출합니다: (1) 더 단순한 단일 단계 질문을 생성하여 직접 답변을 생성, (2) 해당 질문-답변 쌍을 검색된 문서에 기반하여 교정. 이 과정에서 LLM의 지식을 사용하여 잘못된 예측을 수정합니다. 또한, 소규모 모델에도 이 방식을 적용할 수 있도록 지시 기반 증류 방법을 제안합니다.

- **Performance Highlights**: 네 개의 데이터셋에서 광범위한 실험을 통해 제안된 방법의 우수성을 입증했습니다. GenGround는 기존의 강력한 베이스라인보다 뛰어난 성능을 보여주었으며, 지시 기반 증류 방법을 통해 소규모 모델에서도 강력한 성능을 발휘하도록 했습니다.



### Leveraging Passage Embeddings for Efficient Listwise Reranking with Large Language Models (https://arxiv.org/abs/2406.14848)
- **What's New**: 최근 많은 연구에서 대형 언어 모델(LLMs)을 사용해 문장을 평가(ranking)하는 효과를 입증했으며, RankGPT와 같은 리스트 방식(listwise) 접근법이 이 작업에서 새로운 상태의 최첨단을 이루었습니다. 그러나 RankGPT 모델의 효율성은 최대 컨텍스트 길이 및 비교적 높은 LLM 추론 지연 시간에 한계가 있습니다. 이를 해결하기 위해, 본 논문에서는 PE-Rank를 제안합니다. 이는 문장 임베딩(single passage embedding)을 컨텍스트 압축의 좋은 대안으로 활용하여 효율적인 리스트 방식 문장 재평가를 수행합니다. 각 문장을 특수 토큰으로 취급하여 문장 임베딩을 LLM에 직접 입력하고, 이를 통해 입력 길이를 줄일 수 있습니다. 또한, 디코딩 공간을 동적으로 이러한 특수 토큰으로 제한하여 디코딩 프로세스를 가속화하는 추론 방법을 도입했습니다.

- **Technical Details**: PE-Rank는 먼저 밀도 검색 모델(dense retrieval model)을 사용하여 문장을 벡터 인덱스로 사전 인코딩합니다. 문장 임베딩을 특수 토큰으로 취급하여 원래 텍스트 대신 LLM에 입력하고, 검색 모델의 임베딩 공간과 LLM의 입력 임베딩 공간을 일치시키기 위해 프로젝터를 사용합니다. 추론에서는 동적으로 디코딩 공간을 제한하여 특수 토큰을 사용할 수 있도록 하는 '동적 제약 디코딩(Dynamic-Constrained Decoding)' 전략을 제안합니다. 또한, 목록 학습(listwise learning to rank loss)을 통해 모델을 학습시킵니다.

- **Performance Highlights**: PE-Rank는 TREC DL 및 BEIR와 같은 인기 있는 검색 벤치마크에서 평가되었습니다. 실험 결과, PE-Rank는 높은 효율성을 유지하면서 비압축 방법과 비교할 때 유사한 순위 성능을 달성했습니다. 예를 들어, DL19에서 BM25로 검색된 상위 100 개 후보를 재평가할 때, PE-Rank의 NDCG@10은 동일한 설정 하에서 비압축 방법에 비해 성능 저하가 2 % 미만이면서 지연 시간을 4.5배 줄였습니다.



### ChatGPT as Research Scientist: Probing GPT's Capabilities as a Research Librarian, Research Ethicist, Data Generator and Data Predictor (https://arxiv.org/abs/2406.14765)
Comments:
          Main article is 14 pages, 1 table. Includes SI Appendix: 26 pages, 12 tables, 2 figures. Total: 40 pages, 13 tables, 2 figures. Under revised review at PNAS

- **What's New**: 이 논문에서는 GPT-3.5와 GPT-4의 과학적 연구 수행 능력을 체계적으로 평가하였습니다. 연구 분야로 심리학을 선택하여 네 가지 주요 연구 과정을 조사했습니다: 연구 사서(Research Librarian), 연구 윤리학자(Research Ethicist), 데이터 생성자(Data Generator), 그리고 새로운 데이터 예측자(Novel Data Predictor)로서의 역할 수행 능력을 분석하였습니다.

- **Technical Details**: 1. 연구 사서(Research Librarian): GPT-4와 GPT-3.5는 특정 주제의 문헌 검토를 수행하는데, GPT-3.5는 36.0%의 확률로 가짜 참조를 생성한 반면, GPT-4는 5.4%로 상대적으로 낮았습니다. GPT-4는 자신의 오류를 인식하고 있음을 보여주었습니다.
2. 연구 윤리학자(Research Ethicist): GPT-4는 연구 프로토콜의 p-hacking과 같은 위반을 감지하는 능력이 뛰어나, 명확한 문제의 88.6%, 미묘한 문제의 72.6%를 수정했습니다.
3. 데이터 생성자(Data Generator): GPT 모델은 문화적 편향 패턴을 일관되게 재현하여, 가설 생성 등 데이터 생성의 가능성을 보여주었습니다.
4. 새로운 데이터 예측자(Novel Data Predictor): 두 모델 모두 훈련 데이터에 없는 새로운 결과 예측에 실패했습니다.

- **Performance Highlights**: GPT-4는 연구 사서 역할 수행 능력에서 GPT-3.5보다 훨씬 적은 가짜 참조를 생성하며 발전된 모습을 보였습니다. 연구 윤리학자로서도 GPT-4는 뛰어난 성능을 보였으며, 데이터 생성에서는 기존의 문화적 편향 패턴을 재현하는 능력을 보여주었습니다. 그러나 새로운 데이터를 예측하는 능력에서는 두 모델 모두 한계가 확인되었습니다.



### TTQA-RS- A break-down prompting approach for Multi-hop Table-Text Question Answering with Reasoning and Summarization (https://arxiv.org/abs/2406.14732)
- **What's New**: 다중 홉 (multi-hop) 테이블-텍스트 질의응답(QA) 모델 TTQA-RS를 제안합니다. 이 모델은 테이블과 텍스트 요약을 통한 서브질문을 생성하여, 논리에 기반한 테이블-텍스트 QA를 수행합니다. TTQA-RS는 오픈소스 언어 모델(Large Language Models, LLMs)을 이용하여 기존의 프롬프트 방법을 초과하는 성능을 보였으며, 학습 기반 모델들과 비교해도 우수한 성능을 나타냈습니다. 특히 GPT-4와 LLaMA3-70B를 이용하여, 다중 홉 테이블-텍스트 QA에서 최첨단의 성능을 달성하였습니다.

- **Technical Details**: TTQA-RS 모델은 다섯 가지 단계로 테이블-텍스트 QA 문제를 해결합니다. (1) 테이블 행과 텍스트로부터 요약 생성, (2) 질문 분해, (3) 예상 답변의 엔터티 타입 예측, (4) 독립 서브질문의 테이블-텍스트 QA, (5) 원래 질문의 테이블-텍스트 QA. 추출기는 테이블 셀에 링크된 텍스트에서 관련 행과 패시지를 가져옵니다. HybridQA 데이터셋에서는 S3HQA 모델의 테이블 추출기와 HYBRIDER의 텍스트 추출기를 사용합니다. OTT-QA의 개발 세트에서는 테이블 추출기를 사용하지 않고 HYBRIDER의 텍스트 추출기만을 사용합니다.

- **Performance Highlights**: TTQA-RS 모델은 HybridQA와 OTT-QA 데이터셋에서 기존의 프롬프트 기반 방법을 능가하는 성능을 보였습니다. 특히 HybridQA 테스트 세트에서 기존 Chain of Thought(CoT) 모델 대비 정확한 일치 점수에서 6% 증가를 달성하였습니다. 또한, S3HQA의 GPT 3.5 성능을 초과하여, 소규모 LLMs에서도 다중 홉 테이블-텍스트 QA에서 우수한 잠재력을 입증하였습니다.



### Bioptic -- A Target-Agnostic Efficacy-Based Small Molecules Search Engin (https://arxiv.org/abs/2406.14572)
- **What's New**: 최근 가상 스크리닝(Virtual Screening) 성공 사례는 대형 모델과 광범위한 화학 라이브러리 덕분에 가능했습니다. 하지만 이 두 요소를 결합하는 것은 도전 과제입니다. 더 큰 모델을 사용하면 비용이 많이 들고, 초거대 라이브러리를 스크리닝하는 것이 불가능해집니다. 이에 대해 우리는 구조가 다른 분자들을 생물학적 활동이 유사한 분자들과 찾을 수 있는, 타겟 독립적인 효능 기반 분자 검색 모델을 개발했습니다.

- **Technical Details**: 우리의 모델은 SMILES 기반 Transformermodel입니다. 먼저 비지도 학습 방식으로 PubChem과 Enamine REAL Space의 대규모 무라벨 분자 데이터를 사용해 예비 학습을 수행한 후, BindingDB 데이터를 사용해 각 분자가 활성화될 타겟을 예측하는 두 번째 단계를 진행합니다. 이를 위해 우리는 로버타(RoBERTa) 모델을 사용하고, 모델의 최종 임베딩을 위해 여러 풀링 방식을 적용했습니다. 예를 들면, 분류 토큰의 최대값과 평균값을 사용한 풀링 기법 등이 있습니다.

- **Performance Highlights**: 우리 모델은 40억 개의 Enamine REAL 라이브러리를 100% 리콜율로 초고속 스크리닝할 수 있도록 고도로 최적화된 SIMD 명령어 기반의 검색 시스템을 개발했습니다. 실험 결과, 기존 최첨단 모델들과 비교했을 때, 새로운 구조의 분자를 찾는 능력과 속도에서 우수한 성능을 보였습니다.




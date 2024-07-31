New uploads on arXiv(cs.CL)

### Training on the Test Task Confounds Evaluation and Emergenc (https://arxiv.org/abs/2407.07890)
- **What's New**: 이 논문은 대규모 언어 모델 평가에서 새로운 관점을 제안합니다. 저자들은 테스트 작업에서의 학습(training on the test task)이 모델 비교와 신흥 능력(emergent capabilities)에 대한 주장에 어떻게 영향을 미치는지 탐구합니다. 테스트 데이터에서의 학습과는 달리, 테스트 작업에서의 학습은 부정행위가 아니며, 오히려 평가 관련 데이터를 사전 학습 단계에 포함시키는 기술을 의미합니다.

- **Technical Details**: 논문에서는 총 53개의 언어 모델을 대상으로 MMLU와 GSM8K 두 가지 주요 벤치마크를 분석합니다. 해당 모델들을 사전 학습된 모델과 사후 미세 조정된 모델로 나눠 비교합니다. 사후 미세 조정 시, 테스트 작업 관련 데이터를 사용하여 다시 조정하면 테스트 작업에서의 학습 효과를 상쇄할 수 있음을 보여줍니다.

- **Performance Highlights**: MMLU와 GSM8K 벤치마크에서 비교했을 때, 최신 모델이 이전 모델보다 더 잘 수행되는 것은 테스트 작업의 영향 때문이라는 결과가 나옵니다. 사후 미세 조정을 통해 새로운 모델의 성능 우위를 제거할 수 있었으며, 이는 테스트 작업에서의 학습이 벤치마크 성능을 왜곡한다는 것을 시사합니다. 따라서 모델 비교와 신흥 능력을 올바르게 평가하기 위해서는 동일한 양의 테스트 작업 관련 데이터를 사용한 미세 조정이 필요합니다.



### Attribute or Abstain: Large Language Models as Long Document Assistants (https://arxiv.org/abs/2407.07799)
Comments:
          Code and data: this https URL

- **What's New**: LLMs (대형 언어 모델)에서 장문 문서 처리와 관련한 'hallucination' 문제를 해결하기 위해 'attribution' (정보 출처 명시) 방법을 연구한 새로운 벤치마크, LAB를 발표했습니다. 이 연구는 6개의 다양한 장문 문서 작업에 대해 평가하였으며, 다양한 크기의 4개의 LLM에서 출처 명시 방식을 실험했습니다.

- **Technical Details**: 연구는 'response generation' (응답 생성)과 'evidence retrieval' (증거 검색) 두 가지 작업을 중심으로 진행되었으며, 다음과 같은 접근 방식을 사용했습니다: 1) post-hoc, 2) retrieve-then-read, 3) citation. 또한, 입력 길이를 줄이기 위한 추가적인 검색 단계가 포함된 'reduced-post-hoc'과 'reduced-citation' 접근 방식도 고려되었습니다. 연구는 이러한 접근 방식들이 장문 문서 환경에서 어떻게 성능을 발휘하는지 평가했습니다.

- **Performance Highlights**: 1) Large fine-tuned 모델에서는 citation 방식이 최상의 성능을 보였으나, 작은 크기의 LLM에서는 post-hoc 접근 방식이 효과적일 수 있습니다. 2) 문서의 위치에 따른 증거의 분포는 대체로 일치하였으나, GovReport를 제외한 대부분의 문서에서 증거가 뒤에 나타날수록 응답 질이 떨어졌습니다. 3) 증거 품질은 단일 사실 응답의 품질을 예측할 수 있으나, 다중 사실 응답의 경우 증거를 제공하는 데 어려움이 있음을 발견하였습니다.



### Flooding Spread of Manipulated Knowledge in LLM-Based Multi-Agent Communities (https://arxiv.org/abs/2407.07791)
Comments:
          18 Pages, working in progress

- **What's New**: 이번 연구에서는 다중 에이전트 시스템 내에서 대형 언어 모델(LLMs)을 활용한 조작된 지식의 확산 가능성과 보안 위협을 분석하였습니다. 연구진은 Persuasiveness Injection과 Manipulated Knowledge Injection을 포함한 새로운 2단계 공격 방법을 제안하여 명시적 프롬프트 조작 없이도 조작된 지식이 확산될 수 있음을 입증하였습니다.

- **Technical Details**: 연구에서는 실제 다중 에이전트 시스템 배포를 모방한 시뮬레이션 환경을 구축하고, 공격자가 프롬프트를 직접 조작하지 않고도 에이전트를 통해 조작된 정보를 확산시킬 수 있는지를 검토하였습니다. Persuasiveness Injection 단계에서 Direct Preference Optimization (DPO) 알고리즘을 사용해 에이전트의 설득력을 높이고, Low-Rank Adaptation (LoRA) 기술을 활용해 효율적으로 미세 조정하였습니다. 두 번째 Manipulated Knowledge Injection 단계에서는 Rank-One Model Editing (ROME)을 통해 에이전트의 특정 신경망 층을 수정하여 지식 인식을 잠재적으로 변경하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 공격 방법을 통해 조작된 정보가 다중 에이전트 시스템 내에서 성공적으로 확산될 수 있음을 확인하였습니다. 특히 대화가 거듭될수록 조작된 지식의 영향력이 커짐을 발견하였고, RAG 시스템을 사용하는 에이전트의 경우 이러한 지식이 장기간 지속될 수 있음이 입증되었습니다. 이 연구는 LLM 기반 다중 에이전트 시스템의 보안 리스크를 강조하며, 조작된 지식 확산을 방지하기 위한 강력한 방어 방법의 필요성을 제기합니다.



### WorldAPIs: The World Is Worth How Many APIs? A Thought Experimen (https://arxiv.org/abs/2407.07778)
Comments:
          ACL 2024 NLRSE, 8 pages

- **What's New**: 이 연구는 근본적인 행동이나 affordances를 API 호출을 통해 접근하여 물리적 환경에서 결정을 내리는 AI 시스템의 필요 조건을 탐구합니다. 기존의 embodied simulators가 제한된 API 세트를 제공하는 것과 달리, 얼마나 다양한 행동(API)가 필요한지와 이들이 어떤 모습이어야 하는지를 고찰합니다. 이는 wikiHow 튜토리얼을 기반으로 수행되며, wikiHow의 다양한 인간 작성 작업을 기반으로 새로운 API를 점진적으로 도출하는 프레임워크를 제안합니다.

- **Technical Details**: 이 연구는 크게 두 부분으로 나뉩니다: 1) 초기 API 세트를 재사용하고 2) 필요한 경우 새로운 API 호출을 생성하는 방식입니다. GPT-4를 사용해 Python 프로그램을 생성하는 few-shot prompting 기법을 통해 수행됩니다. 구체적으로, wikiHow 튜토리얼을 기반으로 자연어 지침을 Python 프로그램으로 변환하여 agent의 정책을 생성하며, 필요한 경우 새로운 API를 도출합니다.

- **Performance Highlights**: wikiHow 튜토리얼의 0.5% 미만을 대상으로 수행된 초기 실험에서 300개 이상의 API를 도출했습니다. 이 결과는 다양한 물리적 세계의 작업을 포괄하기 위해 필요한 기본적인 행동 공간의 하한선을 제시합니다. 인간 평가와 자동 평가를 통해 제안된 파이프라인이 wikiHow 지침을 수행하는 데 필요한 API를 효과적으로 유도할 수 있음을 보여줍니다. 기존 embodied environments가 도출된 API의 극히 일부분만을 지원한다는 점을 강조하며 더 풍부한 행동 공간의 개발을 촉진합니다.



### Multi-task Prompt Words Learning for Social Media Content Generation (https://arxiv.org/abs/2407.07771)
Comments:
          8 pages, 5 figures

- **What's New**: 인터넷의 급속한 발전으로 소셜 미디어 플랫폼에서의 자가 표현 및 상호작용이 증가했습니다. 하지만 소셜 미디어 콘텐츠 생성에 인공지능 기술이 활용되지 않았습니다. 이번 연구는 이러한 문제를 해결하기 위해 다중 모드 정보 융합을 기반으로 새로운 프롬프트 단어 생성 프레임워크를 제안합니다. 이 프레임워크는 주제 분류, 감정 분석, 장면 인식, 키워드 추출을 통해 더 포괄적인 프롬프트 단어를 생성합니다. 이를 바탕으로 ChatGPT를 이용해 고품질의 트윗을 생성합니다.

- **Technical Details**: 제안된 알고리즘은 다음과 같습니다: 1) BERT와 ViT를 통해 이미지 및 텍스트 특징을 결합; 2) 결합된 이미지 및 텍스트 특징을 이용해 주제 분류, 감정 분석, 장면 인식, 키워드 추출 등 다중 작업 학습을 통해 프롬프트 단어를 생성; 3) ChatGPT와 프롬프트 단어로 채워진 템플릿을 이용해 트위터 텍스트 생성; 4) 최종적으로 이미지와 텍스트를 포함한 트윗을 생성합니다. 특히 ViT ImageProcessor를 이용해 이미지를 벡터로 변환하고, HuggingGPT와 CLIP을 통해 다양한 텍스트 설명을 생성 및 선택합니다.

- **Performance Highlights**: 확장된 콘텐츠 생성 평가 결과, 제안된 프롬프트 단어 생성 프레임워크는 수작업 및 다른 프롬프트 기술에 비해 더 높은 품질의 콘텐츠를 생성했습니다. 주제 분류, 감정 분석, 장면 인식 작업은 콘텐츠의 명확성과 이미지와의 일관성을 크게 향상시켰습니다.



### A Proposed S.C.O.R.E. Evaluation Framework for Large Language Models : Safety, Consensus, Objectivity, Reproducibility and Explainability (https://arxiv.org/abs/2407.07666)
- **What's New**: 본 논문은 기존의 정확성 및 정량적 지표를 넘어서는 대규모 언어 모델(LLM, Large Language Models)의 헬스케어 분야에서의 포괄적인 정성적 평가 프레임워크의 필요성을 제안합니다. 이를 위해, 안전성(Safety), 합의(Consensus), 객관성(Objectivity), 재현성(Reproducibility), 설명 가능성(Explainability)이라는 5가지 핵심 측면을 평가 기준으로 제안합니다.

- **Technical Details**: 제안된 S.C.O.R.E. 프레임워크는 헬스케어와 임상 응용 분야에서 안전하고, 신뢰할 수 있으며, 윤리적인 미래의 LLM 기반 모델을 평가하는 기초가 될 수 있습니다. 각 측면은 다음과 같습니다: 안전성(Safety)은 모델이 환자에게 해를 끼치지 않도록 보장하며, 합의(Consensus)는 의료 전문가들 사이에서 일관된 결과를 도출할 수 있음을 의미합니다. 객관성(Objectivity)은 모델의 결과가 편향되지 않도록 하며, 재현성(Reproducibility)은 동일한 조건에서 반복적인 결과를 생성할 수 있음을 보장합니다. 설명 가능성(Explainability)은 모델의 결정 과정이 투명하고 이해 가능하도록 만듭니다.

- **Performance Highlights**: S.C.O.R.E. 프레임워크를 통해 평가된 LLM 모델은 기존의 단순한 정확성 지표를 넘어선 포괄적인 신뢰성과 윤리적 평가를 받을 수 있습니다. 이는 특히 헬스케어 및 임상 환경에서 모델의 채택과 신뢰성을 높이는 중요한 기준이 될 것입니다.



### A Review of the Challenges with Massive Web-mined Corpora Used in Large Language Models Pre-Training (https://arxiv.org/abs/2407.07630)
Comments:
          8 pages, Icaisc 2024 conference

- **What's New**: 대규모 웹 마이닝 코퍼스(web-mined corpora)를 사용한 대형 언어 모델(LLM) 사전 훈련의 도전과제를 깊이 있게 검토한 논문이 공개되었습니다. 이 리뷰는 노이즈(irrelevant or misleading information), 중복 콘텐츠, 저품질 또는 부정확한 정보, 편향성, 그리고 민감하거나 개인적인 정보의 포함 등 주요 문제점을 식별합니다. 이러한 과제를 해결하는 것이 정확하고 신뢰할 수 있으며 윤리적으로 책임있는 언어 모델 개발에 필수적입니다.

- **Technical Details**: 웹 마이닝 코퍼스의 본질을 설명하며 Common Crawl, C4, RefinedWeb, OSCAR, WebText, The Pile, RedPajama-Data-v2 등 여러 대표적인 코퍼스의 특성 및 기여도를 강조합니다. 웹에서 자동으로 스크레핑된 데이터는 다단계의 청소 및 사전 처리 과정을 거쳐 머신러닝 응용을 위해 준비됩니다. 하지만 중복 콘텐츠, 무관하거나 저품질 텍스트, 민감한 정보의 존재 등 수많은 과제가 존재합니다.

- **Performance Highlights**: 대부분의 웹 마이닝 코퍼스는 많은 양의 텍스트 데이터를 포함하고 있으며 다양하고 광범위한 언어 패턴, 문맥, 뉘앙스를 학습할 수 있는 기회를 제공합니다. 예를 들면 Common Crawl의 경우 매월 약 20TB의 텍스트 데이터를 제공합니다. 하지만 이와 같은 대규모 데이터의 품질 문제를 해결하기 위한 다수의 접근법이 필요합니다. 해당 논문에서 제안된 방법은 규칙 및 휴리스틱 방법, 사전 훈련된 텍스트 품질 분류기(pre-trained text quality classifiers), 통계 모델 등을 포함합니다.

- **Challenges and Recommendations**: 저품질 텍스트, 중복 및 민감 정보 처리 등 웹 마이닝 코퍼스 사용의 도전 과제와 그에 대한 해결책을 논의합니다. 텍스트 품질 평가 및 데이터 청소 방법으로 규칙 및 휴리스틱 방법, 사전 훈련된 품질 분류기, 통계 모델 활용 등이 제안되었습니다. 특히 사용되는 대형 코퍼스들(Common Crawl, C4, etc.)을 설명하며 각각의 강점과 잠재적 문제를 다룹니다.



### Psycho-linguistic Experiment on Universal Semantic Components of Verbal Humor: System Description and Annotation (https://arxiv.org/abs/2407.07617)
Comments:
          5 pages, 4 figures, preprint submitted to journal in 2023

- **What's New**: 이번 논문에서는 독자가 단어를 하나씩 열어보며 텍스트를 읽는 과정에서 유머를 주석 달기 위한 자가 속도 읽기 시스템(SPReadAH: Self-Paced Reading for Annotation of Humor)을 소개하고, 이를 통해 독자가 텍스트를 유머로 인식하기 시작하는 지점을 기록합니다. 이 시스템을 이용한 심리-언어학적 실험과 수집된 데이터도 다룹니다.

- **Technical Details**: SPReadAH 시스템은 독자가 다음 단어를 열기 위해 누르는 키와 유머 여부를 선택하고 이를 변경하는 과정을 기록합니다. 이 실험은 유머에 대한 이론적 배경 없이 Tyumen State University 학생들을 대상으로 수동 주석 달기를 통해 진행되었습니다. 전체적인 목적은 유머의 보편적인 의미 구성 요소를 찾는 것입니다.

- **Performance Highlights**: 실험 결과, 두 개의 의미 대립 스크립트가 동시에 존재하는 텍스트를 읽는 데 시간이 더 오래 걸리며, 독자의 시선 방향과 같은 다양한 얼굴 표정의 변화를 유도하는 것으로 나타났습니다. 이는 기존 논문에서 논의된 바와 유사합니다. 또한, SPReadAH 시스템을 통해 수집된 데이터는 심리-언어학적 실험 연구에 중요한 기여를 할 수 있습니다.



### The Computational Learning of Construction Grammars: State of the Art and Prospective Roadmap (https://arxiv.org/abs/2407.07606)
Comments:
          Peer-reviewed author's draft of a journal article to appear in Constructions and Frames (2025)

- **What's New**: 이 논문은 건축 문법(computational construction grammar) 학습의 최신 동향을 문서화하고 검토하며, 다양한 연구 분야에서 이루어진 형태-의미 쌍(form-meaning pairings) 학습에 관한 기존 연구를 종합합니다. 목표는 세 가지입니다. 첫째, 제안된 다양한 방법론과 그 결과를 종합합니다. 둘째, 이미 해결된 부분과 추가 연구가 필요한 부분을 식별합니다. 셋째, 대규모 사용 기반 건축 문법 학습에 대한 로드맵을 제시하여 향후 연구를 촉진하고 효율화합니다.

- **Technical Details**: 이 논문은 건축 문법 학습을 위한 31개의 모델을 14가지 기준에 따라 검토합니다. 이 기준은 학습 작업, 데이터셋, 입력, 형태 복잡성, 의미 복잡성, 의미 표현의 기반, 분할 수준, 미리 정의된 어휘, 문법 범주, 점진적 학습, 양방향 문법, 추상화 수준, 비구성적 언어 사용, 벤치마크를 포함합니다. 이러한 다양한 기준을 통해 모델들을 비교 및 분석하여 연구의 차이를 체계적으로 평가합니다.

- **Performance Highlights**: 논문에 따르면 현재 연구의 중요한 격차는 특정 이론적 주장들의 대규모 확장성 확증에 있습니다. 또한 서로 다른 이론 간 지식 격차와 차이를 식별하는 데 중요한 역할을 합니다. 이 연구는 다양한 실제 응용 프로그램에서 건축 문법을 활용하는데 중요한 기여를 합니다. 예를 들어, 시각적 질문 응답(visual question answering), 담화의 프레임 의미론적 분석(frame-semantic analysis), 코퍼스 분석 등에 활용될 수 있습니다.



### HebDB: a Weakly Supervised Dataset for Hebrew Speech Processing (https://arxiv.org/abs/2407.07566)
Comments:
          Accepted at Interspeech2024

- **What's New**: HebDB는 약 2500시간 분량의 자연스러운 히브리어 음성을 포함하는 약한 감독 방식의 데이터셋입니다. 다양한 화자와 주제를 포함하며, 연구와 개발을 목표로 ASR(Auto Speech Recognition, 자동 음성 인식)을 위한 두 가지 기본 시스템을 제공합니다: (i) 셀프 슈퍼바이즈드 모델(Self-supervised model); (ii) 완전 감시된 모델(Fully supervised model). 이 데이터셋은 히브리어 음성 처리 도구의 개발을 촉진하기 위해 공개되었습니다.

- **Technical Details**: HebDB는 2584시간의 자연 발생 및 자발적인 음성을 담고 있습니다. 이 데이터셋은 증인들과의 증언과 다섯 개의 팟캐스트로 구성됩니다. 제공되는 데이터는 raw 버전과 전처리된 버전으로 나뉘며, 전처리된 데이터는 자동으로 전사되어 약 1690시간 분량입니다. 이를 통해 다양한 사전 처리 방법을 탐구할 수 있습니다.

- **Performance Highlights**: HebDB에 최적화된 셀프 슈퍼바이즈드 모델과 완전 감독 모델은 유사한 모델 크기를 고려할 때 현재 다국어 ASR 대안들보다 더 나은 결과를 달성했습니다. 이로써 히브리어와 같은 저자원 언어의 성능 격차를 해소하는 데 기여할 것입니다. 전체 데이터셋 및 관련 코드와 모델은 https://pages.cs.huji.ac.il/adiyoss-lab/HebDB/에서 공개적으로 이용할 수 있습니다.



### On Leakage of Code Generation Evaluation Datasets (https://arxiv.org/abs/2407.07565)
Comments:
          4 main pages, 9 in total

- **What's New**: 이번 논문에서는 최신 대규모 언어 모델에서 코드 생성 테스트 세트의 오염 문제를 다룹니다. 저자들은 직접적인 데이터 누출(direct data leakage), 합성 데이터를 통한 간접적인 데이터 누출(indirect data leakage through synthetic data), 그리고 모델 선택 과정에서 평가 세트에 과적합(overfitting to evaluation sets)되어 발생하는 오염의 세 가지 가능성을 제시하고, 이를 뒷받침하는 증거를 보여줍니다. 저자들은 새로운 'Less Basic Python Problems(LBPP)' 데이터셋을 제안하여, 현재 코드 생성 능력을 평가하고 HumanEval과 MBPP에 과적합된 상태를 확인하고자 합니다.

- **Technical Details**: 저자들은 161개의 프롬프트와 이에 맞는 파이썬 솔루션으로 구성된 새로운 데이터셋을 활용하여, 코드 생성 모델의 오염 문제를 분석했습니다. 데이터 누출은 주로 세 가지 경로를 통해 발생할 수 있는 것으로 나타났습니다: 훈련 데이터 내에서의 직접적인 포함, 합성 데이터 사용을 통한 간접적인 포함, 및 모델 선택 과정에서의 과적합입니다. 저자들은 특히 HumanEval과 MBPP 데이터셋이 자주 사용되며, 이는 코드 생성 능력 평가에 있어서의 문제를 초래한다고 주장합니다.

- **Performance Highlights**: HumanEval와 MBPP 테스트 세트가 데이터 오염 문제를 가지고 있다는 사실이 밝혀졌습니다. 이는 2023-2024년 동안 학계 및 산업계에서 발표된 주요 코드 생성 기능을 주장하는 모든 주요 모델이 이 두 데이터셋을 사용했기 때문에 중요합니다. 저자들은 LBPP를 제안하여, 현존하는 모델들이 얼마나 이 데이터셋에 과적합되어 있는지 평가하기 위한 새로운 기준을 제공합니다.



### Arabic Automatic Story Generation with Large Language Models (https://arxiv.org/abs/2407.07551)
- **What's New**: 이 논문은 최근에 강력한 도구로 부상한 대형 언어 모델(LLMs)로 아랍어 이야기 생성 작업을 탐구합니다. 번역 데이터와 GPT-4를 활용하여 아랍어 표준어(MSA)와 두 가지 방언(이집트 및 모로코)으로 적합한 데이터를 생성하는 훈련 프롬프트를 도입했습니다. 이 모델들은 고품질의 일관된 이야기를 생성할 수 있으며, 공개 데이터셋과 모델도 제공할 예정입니다.

- **Technical Details**: 이 논문은 AraLLaMA라는 강력한 아랍어 LLM을 활용하여 이야기를 생성하는 방법론을 제시합니다. GPT-4로 생성된 합성 데이터셋과 영어에서 번역된 합성 데이터셋을 사용해 두 가지 방법으로 모델을 미세 조정하였습니다. 또한 두 가지 아랍어 방언 데이터로 모델을 추가 튜닝하여, 표준 아랍어(MSA)뿐만 아니라 방언으로도 이야기를 생성할 수 있습니다. 평가 방법으로는 인간 평가와 자동 평가를 모두 사용했습니다.

- **Performance Highlights**: 모델의 효능은 인간 평가를 통해 입증되었으며, 세밀하게 커스터마이징된 데이터셋 덕분에 일관성과 유창성이 높은 이야기를 생성했습니다. AraLLaMA는 GPT-3.5, AceGPT-7B, Command-R222와 같은 최신 모델들과 비교하여 경쟁력 있는 성능을 보였습니다. 또한, 개발된 데이터셋과 모델은 https://github.com/UBC-NLP/arastories에서 공개될 예정입니다.



### Beyond Benchmarking: A New Paradigm for Evaluation and Assessment of Large Language Models (https://arxiv.org/abs/2407.07531)
- **What's New**: 현재 대형 언어 모델(LLMs)의 평가 벤치마크는 평가지 제한, 적시성 업데이트 부족, 최적화 안내 부족 등 다양한 문제를 가지고 있습니다. 본 논문에서는 LLMs의 측정을 위한 새로운 패러다임인 Benchmarking-Evaluation-Assessment를 제안하며, '시험실'에서 '병원'으로 LLM 평가의 '장소'를 이동시킵니다. 이 패러다임을 통해 LLM에 대한 '신체 검사'를 수행하고, 특정 작업 해결을 평가 내용으로 삼아, LLMs의 기존 문제를 깊이 있게 분석하고 최적화 권고 사항을 제공합니다.

- **Technical Details**: 새로운 패러다임은 사람들이 병원에서 신체 검사를 받는 과정과 유사하게 LLMs의 능력을 세 가지 단계로 측정합니다. 먼저, 포괄적이고 거친 점수를 통해 LLM의 결여된 능력을 찾고, 두 번째 단계에서는 전문 작업을 완료함으로써 문제를 심층적으로 조사합니다. 마지막으로, 세 번째 단계에서는 '의사 모델'을 사용해 세분화된 메트릭과 함께 문제의 원인을 진단하고 최적화 방향을 제시합니다. 현재 CritiqueLLM과 같은 연구들이 이 세 번째 단계에서 LLM의 평가 기능을 탐구하고 있습니다.

- **Performance Highlights**: 제안된 Benchmarking-Evaluation-Assessment 패러다임을 통해 기존의 지식 시험 평가에서 탈피하여 작업 해결 능력을 평가하면서, LLM의 능력을 동적으로 업데이트할 수 있습니다. 이러한 접근 방식은 전통적인 평가 벤치마크의 제한을 극복하고, LLM의 문제를 발견하고 '치료'할 수 있는 진정한 최적화 방향을 제공합니다.



### Bucket Pre-training is All You Need (https://arxiv.org/abs/2407.07495)
- **What's New**: 이번 연구에서는 고정 길이의 데이터 구성 전략이 도입하는 노이즈와 모델의 장거리 의존성(capture long-range dependencies) 인식 한계를 극복하기 위해 고유의 새로운 다중 버킷 데이터 구성 방법을 제안합니다. 본 방법론은 고정 길이 패러다임에서 벗어나 보다 유연하고 효율적인 사전 훈련 방식을 제공합니다.

- **Technical Details**: 본 연구는 세 가지 주요 평가지표인 패딩 비율(padding ratio), 잘림 비율(truncation ratio), 그리고 연결 비율(concatenation ratio)을 도입하여 데이터 구성 품질을 평가했습니다. 제안된 다중 버킷 데이터 구성 방법은 문서의 길이에 따라 데이터를 다양한 버킷으로 구성함으로써 양질의 데이터 구성을 달성합니다. 먼저 문서를 길이순으로 정렬한 후, 남은 문서 중 가장 긴 문서를 버킷에 배정하는 알고리즘을 사용합니다. 버킷이 꽉 찰 때까지 문서를 추가하고, 필요 시 미리 정의된 패딩 비율에 따라 패딩 토큰을 사용합니다.

- **Performance Highlights**: 제안된 방법론은 노이즈를 줄이고 컨텍스트를 유지하는 동시에 훈련 속도를 가속화하여 LLMs 사전 훈련의 효율성과 효과를 모두 향상시키는 것으로 나타났습니다. 실험 결과, 이 방법이 전통적인 고정 길이 사전 훈련 접근 방식보다 데이터 구성 품질을 크게 개선한다는 것이 입증되었습니다.



### Review-LLM: Harnessing Large Language Models for Personalized Review Generation (https://arxiv.org/abs/2407.07487)
- **What's New**: 이 논문은 Review-LLM이라는 새로운 시스템을 소개합니다. Review-LLM은 대형 언어 모델 (Large Language Models, LLMs)을 사용해 개인화된 리뷰를 생성하는 방법을 제안합니다. 특히, 사용자 행동 이력을 집계하여 입력 프롬프트를 구성하고, 사용자의 만족도를 표시하는 평점을 포함시켜 모델이 사용자 취향과 감정의 경향을 더 잘 이해하도록 돕습니다. 마지막으로, LLMs (예: Llama-3)를 사용하여 감독된 미세 조정 (Supervised Fine-Tuning, SFT)을 통해 맞춤 리뷰를 생성합니다.

- **Technical Details**: Review-LLM은 사용자 u, 아이템 v, 평점 r, 사용자 이력에 기반한 리뷰 생성을 목표로 합니다. 사용자 이력은 사용자가 구매한 아이템의 연속이며, 각 아이템에 대한 제목과 리뷰로 구성됩니다. 이 정보를 통합하여 입력 프롬프트를 구성하고, 만족도를 파악하기 위해 평점을 포함시킵니다. 이를 통해 LLM는 사용자 관심사와 리뷰 작성 스타일을 학습하고, 'polite' 현상을 방지하여 사용자 불만을 반영한 부정적 리뷰도 생성할 수 있습니다.

- **Performance Highlights**: 실험 결과, 실제 데이터셋에서 기존의 폐쇄형 LLMs보다 더 나은 리뷰 생성 성능을 보여주었습니다. 이는 Review-LLM이 사용자 맞춤형 리뷰 생성을 위한 효과적인 모델임을 입증합니다.



### Out-of-distribution generalisation in spoken language understanding (https://arxiv.org/abs/2407.07425)
Comments:
          Accepted for INTERSPEECH 2024

- **What's New**: 이 논문은 음성 언어 이해(Spoken Language Understanding, SLU) 과제에서 out-of-distribution (OOD) 데이터를 다루기 위한 새로운 데이터셋 SLURPFOOD를 소개합니다. 기존의 SLU 데이터셋인 SLURP를 수정하여, OOD 일반화를 테스트할 수 있도록 데이터 분할을 새롭게 구성했습니다. 이 새로운 데이터셋을 통해, 엔드투엔드 SLU 모델의 일반화 능력이 제한적이라는 것을 발견했습니다.

- **Technical Details**: SLURPFOOD 데이터셋은 SLURP의 다양한 분할을 포함하며, 이를 통해 OOD 일반화 능력을 평가할 수 있습니다. 여기에는 다음과 같은 분할 유형이 포함됩니다. 1) OOV (Out-Of-Vocabulary) 분할: 학습 세트에 보이지 않은 단어를 포함한 테스트 세트가 포함됩니다. 2) CG (Compositional Generalisation) 분할: 새로운 조합의 단어를 포함한 테스트 세트를 사용합니다. 3) 마이크로폰 불일치 (microphone mismatch) 분할: 서로 다른 음향 환경에서의 토큰이 포함됩니다. 통계는 논문 내의 표 1에 제시되어 있습니다. 또한, SLU 모델의 이해도를 높이기 위해 Integrated Gradients 기법을 사용했습니다.

- **Performance Highlights**: 새로운 SLURPFOOD 데이터를 이용한 테스트 결과, 기존의 엔드투엔드 SLU 모델들이 OOD 일반화에서 제한적임을 발견했습니다. 이를 개선하기 위해 두 가지 기술, TOPK와 세분화 처리(segmented processing)를 실험했으나, 일부 데이터 분할에서는 성능이 향상되었으나 모든 분할에서 일관되게 성능이 향상되지는 않았습니다. 이는 새로운 기술의 필요성을 강조합니다.



### KpopMT: Translation Dataset with Terminology for Kpop Fandom (https://arxiv.org/abs/2407.07413)
Comments:
          accepted to LoresMT 2024

- **What's New**: 새로운 연구에서 Kpop 팬덤의 언어 시스템에 특화된 번역 데이터셋, KpopMT를 제안합니다. 이 데이터셋은 Kpop 팬들이 사용하는 고유한 용어를 포함한 1,000개의 한국어-영어 문장 쌍으로 구성되어 있으며, 전문가 번역가들이 번역을 제공했습니다. 이를 통해 소셜 그룹 내 고유 용어 번역의 문제를 해결하고자 합니다.

- **Technical Details**: KpopMT 데이터셋은 3개의 주요 부분으로 구성됩니다: 1) 용어가 태그된 병렬 문장, 2) 병렬 용어 사전 (termbase), 3) 팬덤 단일 언어 데이터셋. 데이터셋의 신뢰성을 위해 영어와 한국어 팬덤 용어에 능통한 5명의 네이티브 영어 사용자가 검토했습니다. 병렬 문장은 팬 커뮤니티 사이트와 트위터에서 수집한 한국어 데이터를 바탕으로 하며, 인간 번역가가 영어 팬덤 용어를 포함하여 번역했습니다.

- **Performance Highlights**: 기존의 번역 시스템, 포함해 최첨단 모델인 GPT를 KpopMT에서 평가한 결과, 전반적으로 낮은 점수를 기록하여 그룹별 고유 용어와 스타일 반영의 어려움을 드러냈습니다. 인간 설문조사에서 팬들은 용어가 포함된 번역을 선호했으며, 이는 문화적 요소를 고려한 번역의 중요성을 강조합니다.



### Automatic Extraction of Disease Risk Factors from Medical Publications (https://arxiv.org/abs/2407.07373)
Comments:
          BioNLP@ACL2024, 12 pages

- **What's New**: 이 논문은 바이오-메디컬 도메인에 사전 훈련된 모델을 활용하여 질병의 위험 요인을 의료 문헌에서 자동으로 식별하는 새로운 접근 방식을 제시합니다. 특히, 이번 연구는 다단계 시스템을 도입하여 관련 문서를 식별하고, 위험 요인 논의가 있는지 분류한 다음, 질병에 대한 특정 위험 요인 정보를 질문 및 응답 모델(QA model)을 통해 추출합니다. 또한, 위험 요인 자동 추출을 위한 종합적인 파이프라인 개발 및 다양한 데이터셋을 컴파일하여 이 분야의 추가 연구에 유용한 자원을 제공하는 것이 주요 기여점입니다.

- **Technical Details**: 이 시스템은 BioBERT를 기반으로 한 사전 훈련된 대형 언어 모델을 활용합니다. 단계별로, 첫째, PubMed에서 의학 초록을 검색하고, 둘째, 위험 요인 정보를 포함하는 초록을 식별하는 이진 분류기를 사용하며, 셋째, 수동으로 주석이 달린 QA 항목에 맞춰 미세 조정된 질문 응답 모델을 통해 텍스트에서 위험 요인 스팬을 추출합니다. 데이터 수집 과정은 KEGG 질병 데이터베이스 API와 PubMed에서 질병 이름을 검색하여 과학 초록을 얻고, 이를 바탕으로 위험 요인을 추출합니다. 수집된 데이터셋에는 15개의 질병과 관련된 1,700개 이상의 위험 요인 및 160,000개 이상의 자동으로 추출된 위험 요인이 포함되어 있습니다.

- **Performance Highlights**: 이 모델은 자동 및 수동 평가 모두에서 고무적인 결과를 보였습니다. 추가 연구를 위한 종합적인 데이터셋 컴파일 및 효과적인 예방 전략 수립을 통해 의료 전문가들이 환자의 결과를 개선하는 데에 크게 기여할 것으로 기대됩니다. 데이터 품질에 대한 세밀한 평가 체계를 통해 수집된 위험 요인의 질을 보장했습니다.



### LokiLM: Technical Repor (https://arxiv.org/abs/2407.07370)
- **What's New**: 이번 연구에서는 1.4B 파라미터를 갖춘 대형 언어 모델 LokiLM을 도입하였습니다. 이 모델은 500B 토큰을 학습하여 자연어 추론 작업에서 강력한 성능을 보이며, 1.5B 파라미터 이하의 모델 중 최고 성능을 달성했습니다. LokiLM은 다수의 교사 모델을 이용한 지식 증류(knowledge distillation)와 고품질의 학습 데이터를 통해 큰 토큰으로 훈련된 모델들과 경쟁할 수 있는 결과를 얻었습니다. 하지만, 이는 TruthfulQA 벤치마크에서 낮은 점수를 받아 공개하지 않기로 결정되었습니다.

- **Technical Details**: LokiLM은 24개의 레이어, 32개의 어텐션 헤드, 2048의 숨김 차원을 가진 표준 디코더 전용 Transformer 아키텍처를 사용합니다. FlashAttention-2를 사용하여 다중 헤드 어텐션 계산을 가속화하고, Attention with Linear Bias (ALiBi)를 사용하여 위치 임베딩을 대신하여 위치 정보를 더 잘 캡쳐했습니다. RMSNorm과 SwiGLU 활성화를 통해 훈련을 안정시키고 더 복잡한 표현을 학습합니다. 고품질의 웹에서 스크랩된 콘텐츠와 머신 생성 텍스트를 포함한 다단계 필터링 파이프라인을 사용하여 데이터 품질과 다양성을 보장했습니다. 8개의 NVIDIA A100 GPU를 사용하여 8일간 훈련을 진행했으며, Fully Sharded Data Parallelism을 통한 8-bit 정밀도로 훈련 효율성을 최적화하였습니다. 지식 증류는 GPT-4, Mistral 7B, Llama 2 13B와 같은 교사 모델을 사용하여 4번째마다 수행되었으며, 손실 스파이크 문제를 해결하기 위해 체크포인트 롤백 전략을 채택했습니다.

- **Performance Highlights**: LokiLM은 1.5B 파라미터 이하의 모델들 중에서 공통 상식 추론(ARC-Challenge, HellaSwag, MMLU, Winogrande) 테스트에서 우수한 성과를 보였습니다. 그러나 TruthfulQA 벤치마크에서는 낮은 점수를 받아 일부 잘못된 정보를 생성하는 경향이 있음을 나타냈습니다. 이러한 결과는 모델의 전반적인 성능이 뛰어나지만, 신뢰성 부분에서 개선이 필요함을 보여줍니다.



### Multilingual Blending: LLM Safety Alignment Evaluation with Language Mixtur (https://arxiv.org/abs/2407.07342)
- **What's New**: 이번 연구에서는 다양한 언어 상황에서 LLM의 안전성 정렬(safety alignment)을 평가하기 위한 혼합 언어 쿼리-응답 스킴인 Multilingual Blending을 도입했습니다. 이는 GPT-4o, GPT-3.5, Llama3와 같은 최첨단 LLM을 복잡한 다언어 환경에서 테스트하는 방식을 제안합니다.

- **Technical Details**: 연구는 다국어 혼합 운영 형식(Multilingual Blending)이 LLM의 안전성 정렬을 얼마나 쉽게 우회할 수 있는지를 조사하며, 특정 언어 패턴(언어 가용성, 형태, 언어 계열 등)이 이 영향에 어떤 역할을 하는지 분석합니다. 실험에서 120개의 명시적 악의적 질문을 사용해 7개의 최첨단 LLM, 55개의 개별 소스 언어 및 53개의 고유한 언어 조합에서 300,000번 이상의 LLM 추론 실행을 포함했습니다.

- **Performance Highlights**: 실험 결과, 섬세하게 설계된 프롬프트 템플릿 없이 Multilingual Blending이 악의 있는 쿼리의 영향을 크게 증폭시켜 LLM 안전성 정렬 우회율이 급격히 증가하는 것으로 나타났습니다(GPT-3.5에서 67.23%, GPT-4o에서 40.34%). 또한, 혼합 언어 형식에서는 여러 언어가 더 쉽게 안전성 정렬을 벗어날 수 있는 것으로 드러났습니다. 이러한 결과는 LLM의 뛰어난 다언어 일반화 능력에 맞춰 복잡한 다언어 환경에서 LLM의 평가 및 안전성 정렬 전략을 개발해야 할 필요성을 강조합니다.



### MixSumm: Topic-based Data Augmentation using LLMs for Low-resource Extractive Text Summarization (https://arxiv.org/abs/2407.07341)
- **What's New**: 이번 연구에서는 자원이 적은 환경에서의 추출적 텍스트 요약을 위한 새로운 접근법인 MixSumm을 제안했습니다. 많은 기존 연구는 주로 추상적 텍스트 요약이나 대형 언어 모델(LLM)인 GPT-3와 같은 모델을 직접적으로 요약에 사용하는데 초점을 맞추어 왔습니다. MixSumm은 오픈소스 LLM인 LLaMA-3-70b를 사용하여 다중 정보가 혼합된 문서를 생성하고, 이를 기반으로 요약 모델을 학습시키는 방법을 제안합니다.

- **Technical Details**: MixSumm은 먼저 LLM을 프롬프트하여 다중 주제 정보를 혼합한 문서를 생성한 후, 이러한 데이터를 활용하여 요약 모델을 학습합니다. 이를 평가하기 위해 ROUGE 점수와 L-Eval을 사용합니다. L-Eval은 참조가 필요 없는 LLaMA-3 기반 평가 방법입니다. 실험은 TweetSumm, WikiHow, ArXiv/PubMed 데이터셋을 사용하여 수행되었습니다. LLaMA-3-70b-Instruct LLM을 사용했으며, 기존의 데이터 증강 및 반감독 학습 방법보다 우수한 성능을 보였습니다.

- **Performance Highlights**: MixSumm은 자원이 적은 환경에서도 최신 프롬프트 기반 접근법보다 뛰어난 성능을 보였습니다. 특히, LLaMA-3-70b 모델의 지식을 소형 BERT 기반 추출적 요약 모델로 효과적으로 증류하는 데 성공했으며, 이는 메모리 요구사항을 크게 줄이면서 성능을 유지하는 데 기여했습니다. 또한, 데이터 증강과 관련된 광범위한 실험을 통해 MixSumm의 유효성을 입증했습니다.



### Interpretable Differential Diagnosis with Dual-Inference Large Language Models (https://arxiv.org/abs/2407.07330)
Comments:
          15 pages

- **What's New**: 이 논문은 대형 언어 모델(LLM)을 사용하여 해석 가능한 감별 진단(Differential Diagnosis, DDx)을 자동으로 생성하는 새로운 프레임워크, Dual-Inf를 제안합니다. 이 프레임워크는 양방향 추론을 통해 감별 진단을 설명할 수 있습니다. 또한, 570개의 공개된 임상 노트를 바탕으로 전문가가 해석한 새로운 DDx 데이터 세트를 개발하였습니다.

- **Technical Details**: Dual-Inf는 대형 언어 모델을 활용하여 환자의 증상 설명을 기반으로 가능한 질병 목록을 예측하고 이에 대한 해석을 제공합니다. 이 프레임워크는 양방향 추론(bidirectional inference)을 통해 감별 진단을 도출하게 설계되었습니다. 실험 결과 Dual-Inf는 기존 기준 방법들에 비해 DDx 해석에서 BERTScore 기준으로 32% 이상의 성능 개선을 달성했습니다.

- **Performance Highlights**: Dual-Inf는 해석 과정에서 오류를 줄이는 데 있어서 탁월하며, 일반화 성능도 뛰어납니다. 특히, 드문 질병에 대한 감별 진단과 설명 제공에 있어서도 유망한 결과를 보였습니다. 인간과 자동 평가 모두에서 Dual-Inf의 효율성이 입증되었습니다.



### Probability of Differentiation Reveals Brittleness of Homogeneity Bias in Large Language Models (https://arxiv.org/abs/2407.07329)
- **What's New**: LLMs(large language models)의 동질성 편향(homogeneity bias)을 조사하여, 이전 연구에서 사용된 인코더 모델의 편향 가능성을 피하고자 GPT-4를 사용하여 단일 단어/표현 완성을 통해 18개 상황 단서에 대한 편향을 검사했습니다. 이를 통해 모델 출력을 직접 평가하는 새로운 접근 방식을 제안했습니다.

- **Technical Details**: 이 연구에서는 인코더 모델이 아닌 GPT-4를 사용하여 특정 인종/성별 그룹과 연관된 다양한 활동을 자동 완성하도록 하였습니다. 각 상황 단서에 대해 6,000개의 완성을 생성했습니다. 완성 변수를 측정하기 위해 '변별 확률(probability of differentiation)'을 사용하여 그룹 간의 변동성을 평가했습니다.

- **Performance Highlights**: 그 결과, 동질성 편향은 상황 단서와 프롬프트에 따라 크게 변동되었고, 이전 연구에서 관측된 편향은 인코더 모델에서 비롯된 것일 수 있음을 시사합니다. 즉, LLM에서의 동질성 편향은 프롬프트의 사소한 변화에도 크게 변할 수 있는 불안정한 것임을 발견했습니다. 앞으로의 연구에서는 구문적 특성과 주제 선택의 변화를 더 탐구할 필요가 있습니다.



### RAG vs. Long Context: Examining Frontier Large Language Models for Environmental Review Document Comprehension (https://arxiv.org/abs/2407.07321)
Comments:
          14 pages

- **What's New**: 이번 연구에서는 NEPAQuAD1.0 벤치마크를 구축하여 환경 영향 보고서(EIS)에 기반한 질문에 대해 세 가지 최첨단 대형 언어 모델(LLMs)인 Claude Sonnet, Gemini, GPT-4의 성능을 평가했습니다. 특히, 이러한 모델들이 법률, 기술 및 규제 관련 정보를 이해하는 능력을 측정했습니다.

- **Technical Details**: 이 연구는 NEPA(National Environmental Policy Act)에 따라 작성된 환경 영향 보고서에서 추출한 긴 문맥을 이해하고 질문에 답하는 능력을 평가하기 위해 LLM을 사용했습니다. 두 가지 접근법을 사용했는데, 첫 번째는 LLM이 NEPA 문서의 내용을 바탕으로 질문에 답할 수 있도록 긴 문맥을 처리하는 방식이고, 두 번째는 Retrieval-Augmented Generation(RAG) 모델을 사용하는 방식입니다.

- **Performance Highlights**: 연구 결과, RAG 모델이 긴 문맥 모델에 비해 정확한 답변을 제공하는 데 있어 현저히 우수한 성능을 보였습니다. 또한, 많은 모델이 닫힌 질문(Closed Questions)에 비해 확산형 질문(Divergent Questions)과 문제 해결형 질문(Problem-Solving Questions)에서는 성능이 떨어지는 것으로 나타났습니다.



### ESM+: Modern Insights into Perspective on Text-to-SQL Evaluation in the Age of Large Language Models (https://arxiv.org/abs/2407.07313)
- **What's New**: LLM 기반 Text-to-SQL 모델의 평가에 있어 기존 평가 지표 EXE 및 ESM의 한계를 밝혀내고, 이를 보완할 새로운 평가 지표 ESM+를 제안합니다. ESM+ 스크립트는 오픈 소스로 공개되어 있으며, 커뮤니티가 더욱 신뢰성 있는 평가를 진행할 수 있게 합니다.

- **Technical Details**: ESM(Evaluated Set Matching)은 SQL 쿼리의 키워드와 인자를 비교함으로써 모델 성능을 평가하지만, semantically (의미적으로) 같은 쿼리가 syntactically (문법적으로) 다른 경우를 처리하지 못해 false positive와 false negative를 일으킬 수 있습니다. 새로운 ESM+는 이러한 문제점을 JOIN 조건, DISTINCT 키워드, LIMIT 값 등에서 개선하여 더 정확한 평가를 제공합니다.

- **Performance Highlights**: EXE와 ESM에서 각각 11.3%, 13.9%의 false positive와 false negative 비율이 나타난 반면, 개선된 ESM+는 각각의 비율이 0.1%와 2.6%로 낮아져 보다 안정적인 평가 결과를 제공했습니다. 9개의 LLM 기반 모델을 Spider와 Co-SQL 데이터셋에서 EXE, ESM 및 ESM+를 사용해 비교 평가한 결과, ESM+가 가장 견고한 평가 지표로 확인되었습니다.



### Reuse, Don't Retrain: A Recipe for Continued Pretraining of Language Models (https://arxiv.org/abs/2407.07263)
Comments:
          Preprint. Under review

- **What's New**: 이번 논문에서는 언어 모델(LM)의 사전 학습을 완료한 후의 성능을 계속해서 향상시키기 위한 데이터 분포 설계와 학습률 스케줄에 대한 가이드라인을 제안합니다. 이를 통해 사전 학습을 새로 시작할 필요 없이, 이미 학습된 모델의 능력을 지속적으로 개선할 수 있습니다.

- **Technical Details**: 15B 파라미터 모델을 8T 토큰으로 사전 학습한 후, 우리는 지속적인 사전 학습에서 사용할 데이터 분포와 학습률 스케줄을 최적화하는 방법을 제안합니다. 두 가지 데이터 분포를 사용하는 것이 최적이며, 두번째 데이터 분포는 모델이 향상시키고자 하는 능력에 더 큰 비중을 두는 방식입니다. 학습률 스케줄도 학습률의 크기와 감쇠의 기울기 간의 균형을 맞추는 것이 가장 효과적입니다.

- **Performance Highlights**: 지속적 사전 학습을 통해 모델 정확도가 평균적으로 9% 향상되었습니다. 또한, 100B에서 1조 토큰까지의 다양한 학습 규모에서도 이 레시피가 유용함을 입증하였으며, 이는 다양한 환경에서 유연하고 견고하게 적용될 수 있음을 보여줍니다.



### Identification of emotions on Twitter during the 2022 electoral process in Colombia (https://arxiv.org/abs/2407.07258)
- **What's New**: 트위터를 사용한 사회 현상 분석이 최근 많은 주목을 받고 있습니다. 특히 정치적 이벤트의 경우, 감정 분석을 통해 후보자에 대한 인식과 대중 토론의 중요한 측면에 대한 정보를 얻을 수 있습니다. 하지만 스페인어, 특히 콜롬비아 스페인어에 대한 감정 분석 연구는 거의 없는 상황입니다. 이에 본 연구에서는 2022년 콜롬비아 대선과 관련된 트윗을 수집하여 감정 분석을 수행하고, 이를 통해 데이터셋 및 코드를 연구 목적으로 공개하고자 합니다.

- **Technical Details**: 본 연구의 데이터는 2022년 5월 22일부터 6월 22일까지 콜롬비아 대선 기간 동안 수집된 585,001개의 트윗으로 구성되었습니다. 이 트윗들은 특정 정치적 해시태그(#)를 이용해 필터링되었고, 총 1,200개의 트윗이 라벨링되었습니다. 라벨링 작업은 BERT 모델을 사용한 감독된 학습 및 GPT-3.5를 사용한 few-shot learning 설정 하에 이뤄졌습니다. 또한, 트윗의 감정을 세부적으로 분류하기 위해 14개의 감정 범주와 '기타' 범주를 포함한 다중 선택 방식을 사용했습니다.

- **Performance Highlights**: BERT 모델과 GPT-3.5 모델의 성능을 비교 분석한 결과, 각각의 모델이 콜롬비아 대선 관련 감정 분석에서 어떠한 성과를 보이는지 평가되었습니다. 특히 BERT 모델은 사전 훈련된 대규모 텍스트 데이터를 활용하여 높은 성능을 발휘하였으며, GPT-3.5는 텍스트 프롬프트를 통해 효과적으로 적은 양의 데이터를 이용해 학습을 진행할 수 있음을 보였습니다.



### Nash CoT: Multi-Path Inference with Preference Equilibrium (https://arxiv.org/abs/2407.07099)
- **What's New**: 이번 연구에서는 체인의 사고(CoT) 프롬프팅이 대형 언어 모델(LLMs)의 복잡한 문제에 대한 추론 능력을 향상시키기 위한 강력한 기술로 부상했다는 점에 주목합니다. 이 연구는 언어 디코딩을 선호도의 합의 게임으로 개념화하여 Nash CoT(Chain-of-Thought)를 제안합니다. 특히, 각 추론 경로 내에서 바이어플레이어 게임 시스템을 구축하여 특정 질문에 대한 컨텍스트에 적합한 템플릿을 자율적으로 선택하여 출력을 생성하도록 합니다. 이렇게 함으로써 기존 자기 일관성 기반 접근방식과 비교하여 유사하거나 향상된 성능을 더 적은 추론 경로로 달성할 수 있습니다.

- **Technical Details**: Nash CoT는 자체 추론 경로에서 내쉬 균형(Nash Equilibrium)에 도달하기 위해 대형 언어 모델(LLM)을 활용하여, 각 로컬 경로 내에서 바이플레이어 게임 시스템을 구축합니다. 이 시스템은 템플릿에 의해 안내된 생성과 모델의 기본 상태에서 생성된 출력을 균형 잡히게 합니다. 이는 정상적인 생성과 템플릿 지향 생성의 선호도를 균형 맞춤으로써 주어진 질문의 컨텍스트에 일치하는 출력을 생성합니다. 이를 통해 다중 경로 추론에서 필요한 경로 수를 줄이면서도 유사 또는 향상된 성능을 달성합니다.

- **Performance Highlights**: Nash CoT는 자가 일관성(self-consistency) 접근 방식 대비 더 적은 추론 경로를 사용하면서도, 아랍어 추론, 상식적 질문 응답 및 상징적 추론과 같은 다양한 추론 작업에서 유사하거나 향상된 성능을 보여주었습니다. 특히 로컬 LLM에서 추론 비용을 최대 50%까지 줄이는 데 성공했습니다.



### LLaVA-NeXT-Interleave: Tackling Multi-image, Video, and 3D in Large Multimodal Models (https://arxiv.org/abs/2407.07895)
Comments:
          Project Page: this https URL

- **What's New**: LLaVA-NeXT-Interleave가 출시되어 대형 멀티모달 모델(Large Multimodal Models, LMMs)의 기능을 확장합니다. 이 모델은 멀티 이미지, 멀티 프레임(비디오), 멀티 뷰(3D), 멀티 패치(단일 이미지) 시나리오를 동시에 다루며, Emergent Capabilities를 통해 다양한 설정과 모달리티 간의 작업 전환을 보여줍니다. 이는 새로운 M4-Instruct 데이터셋과 LLaVA-Interleave Bench를 통해 평가되었습니다.

- **Technical Details**: LLaVA-NeXT-Interleave는 이미지-텍스트 교차 형식(interleaved data format)을 일반 템플릿으로 사용하여 다양한 시나리오를 통합합니다. 이를 위해 M4-Instruct 데이터셋을 1,177.6k 샘플로 컴파일하여 멀티 이미지, 멀티 프레임, 3D 및 단일 이미지 데이터를 포함하게 했습니다. 모델은 이 데이터로 학습하고 교차 도메인 작업 구성을 통해 새로운 기능을 자동으로 학습합니다.

- **Performance Highlights**: LLaVA-NeXT-Interleave는 멀티 이미지, 비디오, 3D 벤치마크에서 우수한 성능을 보여주었으며, 단일 이미지 작업에서도 기존 성능을 유지했습니다. 또한 Emergent Capabilities를 통해 이미지 간 차이점을 영상에서 찾아내는 등 다양한 설정과 모달리티 간의 작업 전환 능력을 자랑합니다.



### Towards Robust Alignment of Language Models: Distributionally Robustifying Direct Preference Optimization (https://arxiv.org/abs/2407.07880)
- **What's New**: 이번 연구는 Direct Preference Optimization (DPO)에서 발생하는 훈련 데이터셋의 노이즈 문제를 해결합니다. 그중에서도 pointwise 노이즈는 저품질 데이터 포인트를, pairwise 노이즈는 잘못된 데이터 쌍 연결로 인해 선호도 순위에 영향을 미치는 데이터쌍을 포함합니다. 이 연구는 Distributionally Robust Optimization (DRO)을 활용해 DPO의 노이즈 저항성을 강화합니다.

- **Technical Details**: DPO는 본질적으로 DRO 원칙을 포함하고 있어 pointwise 노이즈에 대한 견고성을 갖추고 있으며, 정규화 계수 $eta$가 중요한 역할을 합니다. 이 연구에서 우리는 DPO의 프레임워크를 확장하여 Dr. DPO를 소개합니다. Dr. DPO는 pairwise 시나리오에서도 최악의 상황에 대한 최적화를 통해 견고성을 통합합니다. 새로운 하이퍼파라미터 $eta'$를 활용하여 데이터 쌍 신뢰도에 대한 정밀한 제어가 가능하며, 노이즈가 많은 훈련 환경에서 탐색과 활용 간의 전략적 균형을 제공합니다.

- **Performance Highlights**: 실험 결과, Dr. DPO는 생성된 텍스트의 품질과 선호도 데이터셋에서의 응답 정확도를 상당히 향상시켰습니다. 이는 노이즈가 있는 AND 노이즈가 없는 환경 모두에서 개선된 성능을 보여줍니다. 코드는 연구팀의 웹사이트에서 확인할 수 있습니다.



### Generative Image as Action Models (https://arxiv.org/abs/2407.07875)
Comments:
          Project website, code, checkpoints: this https URL

- **What's New**: 최신 연구 'GENIMA'는 Stable Diffusion을 미세 조정하여 로봇 관절 동작(joint-actions)을 RGB 이미지 상에 그려 모델 학습을 가능하게 했습니다. 이는 기존의 이미지 생성 모델을 비주얼 모터(Visuomotor) 제어에도 활용할 수 있는 새로운 가능성을 열었습니다.

- **Technical Details**: GENIMA는 행동 클로닝 에이전트로, Stable Diffusion을 미세 조정하여 목표 관절 위치를 이미지로 생성합니다. 이 이미지는 컨트롤러에 입력되어 시각적 목표를 관절 위치 시퀀스로 변환합니다. 세부적으로는 (1) Stable Diffusion을 fine-tuning하여 목표 관절 위치를 포함한 이미지를 생성하고, (2) 이 이미지를 기반으로 컨트롤러가 실제 관절 동작을 수행하도록 학습시킵니다.

- **Performance Highlights**: GENIMA는 25개의 RLBench와 9개의 실제 조작 작업에서 성능을 입증했습니다. 장면의 변형과 새로운 객체에 대한 강인성에서 기존 최첨단 방법보다 뛰어난 성능을 보였으며, 깊이 정보나 키포인트 없이도 뛰어난 성능을 발휘했습니다. RLBench의 25개 작업 전체에서 우수한 성과를 보였으며, 실험 결과는 새로운 객체와 장면 변형에 유연하게 대응할 수 있음을 보여줍니다.



### FACTS About Building Retrieval Augmented Generation-based Chatbots (https://arxiv.org/abs/2407.07858)
Comments:
          8 pages, 6 figures, 2 tables, Preprint submission to ACM CIKM 2024

- **What's New**: NVIDIA는 자동화된 직원 생산성 향상을 목표로 엔터프라이즈 챗봇을 개발하고 있으며, 이 챗봇들은 주로 Retrieval Augmented Generation (RAG) 및 Large Language Models (LLMs)를 기반으로 합니다. 특히, 기업용 챗봇을 효과적으로 만들기 위해서는 RAG 파이프라인의 세심한 설계가 필요합니다. 이 논문에서는 FACTS(Freshness, Architectures, Cost, Testing, Security) 프레임워크를 소개하고, 엔터프라이즈용 챗봇 구축의 어려움과 해결책을 제시합니다.

- **Technical Details**: RAG 기반 챗봇은 벡터 데이터베이스에서 문서를 추출하고, 쿼리를 재작성(rephrasing), 결과를 재정렬(reranking), 프롬프트(prompt)를 설계하여 응답을 생성하는 과정에서 LLMs를 활용합니다. 특히 벡터 검색 기반 IR 시스템, LLMs, LangChain과 같은 프레임워크는 RAG 파이프라인의 핵심 구성 요소입니다. NVIDIA는 NVBot 플랫폼을 사용하여 IT/HR, 재무 수익 질문 등을 처리하는 세 가지 챗봇(NVInfo Bot, NVHelp Bot, Scout Bot)을 제작했습니다.

- **Performance Highlights**: NVIDIA의 챗봇들은 다양한 데이터 형식을 관리하고, 문서 접근 권한을 준수하며, 최신의 엔터프라이즈 지식을 제공하는 데 집중하고 있습니다. 이를 위해 15개의 RAG 파이프라인 제어 포인트가 필요하며, 각 제어 포인트를 최적화하기 위한 전략이 논의됩니다. 또한, 메타데이터 강화(metadata enrichment), 쿼리 재작성(query rephrasal) 및 하이브리드 검색(hybrid search) 기술을 통해 검색 효율성을 높이는 것이 중요합니다.



### Decompose and Compare Consistency: Measuring VLMs' Answer Reliability via Task-Decomposition Consistency Comparison (https://arxiv.org/abs/2407.07840)
Comments:
          Preprint

- **What's New**: 새로운 알고리즘 Decompose and Compare Consistency (DeCC)는 시각-언어 모델(VLM)의 응답 신뢰도를 측정하는 방법입니다. DeCC는 질문을 여러 하위 질문으로 나누고, VLM이 생성한 직접 응답과 하위 질문에 대한 응답을 비교하여 신뢰도를 평가합니다.

- **Technical Details**: DeCC는 두 가지 주요 구성 요소로 이루어집니다. 첫째, '질문 분해'(Task Decomposition) 단계에서는 질문을 하위 질문들로 나누고, VLM이 이에 답변합니다. 둘째, '일관성 비교'(Consistency Comparison) 단계에서는 VLM의 직접 응답과 하위 질문들을 기반으로 한 응답 간의 일관성을 비교합니다. 이 과정에서 두 개의 독립적인 에이전트(VLM과 LLM)를 사용하여 일관성 비교를 수행합니다.

- **Performance Highlights**: DeCC는 기존의 불확실성 기반 메서드나 self-consistency 방식보다 높은 상관성을 보입니다. 다양한 시각-언어 작업에서 세 가지 VLM 모델에 대해 실험한 결과, DeCC는 모델의 작업 정확도와 더 높은 상관 관계를 나타냈습니다. 이를 통해 VLM의 응답 신뢰도를 더욱 정확하게 측정할 수 있음을 확인했습니다.



### Transformer Alignment in Large Language Models (https://arxiv.org/abs/2407.07810)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)을 고차원 이산 결합 비선형 동적 시스템으로 간주하여, 이러한 모델이 어떻게 작동하는지에 대한 새로운 통찰을 제시합니다. 38개의 공개된 LLM을 분석한 결과, Residual Jacobians의 좌우 특이 벡터 정렬이 모델 성능과 긍정적으로 상관됨을 발견했습니다. 이러한 특성은 LLM의 아키텍처를 최적화하는 데 중요한 정보를 제공합니다.

- **Technical Details**: 이 연구는 Residual Networks (ResNets)에서 발견된 Residual Alignment (RA)에 영감을 받아 Transformer Alignment (TA)를 제안합니다. 구체적으로, 훈련된 LLM의 각 레이어는 비선형적인 경로를 따라가며, Jacobian 행렬을 통해 시스템을 선형화하였습니다. 중요한 발견은 다음과 같습니다: 1) 훈련 중간 표현이 계층 별로 선형화 및 지수적으로 배치된 경로를 형성함. 2) Residual Jacobians의 좌우 특이 벡터가 정렬됨.

- **Performance Highlights**: 훈련 후 측정된 메트릭은 임의로 초기화된 가중치와 비교해 상당한 성능 향상을 보였습니다. 이는 훈련이 Transformers에 미치는 중대한 영향을 강조합니다. 특히, Residual Jacobian 특이 벡터 정렬이 향상될수록 모델의 Open LLM leaderboard 벤치마크 점수가 개선됨을 확인하였습니다.



### ROSA: Random Subspace Adaptation for Efficient Fine-Tuning (https://arxiv.org/abs/2407.07802)
- **What's New**: 새로운 연구에서는 기존의 매개변수 효율적 미세 조정(Parameter Efficient Fine-Tuning, PEFT) 방법들이 초래하는 지연 시간과 성능 문제를 해결할 새로운 방법인 Random Subspace Adaptation (ROSA)을 소개합니다. ROSA는 추론 시간 동안 지연 시간을 전혀 발생시키지 않으면서도 기존의 PEFT 방법들보다 뛰어난 성능을 보여줍니다.

- **Technical Details**: ROSA는 기존의 고정된 사전 학습된 가중치에 병렬로 낮은 순위 매트릭스를 조정하여 대규모 모델을 낮은 메모리 요구사항으로 미세 조정할 수 있습니다. LoRA와 달리 ROSA는 다양한 차원의 하위 공간을 적응할 수 있어 이론적으로나 실험적으로 더 표현력이 높습니다. ROSA는 학습 과정에서 다양한 낮은 순위의 훈련 가능한 하위 공간을 지속적으로 샘플링하고 학습된 정보를 고정된 가중치에 병합합니다.

- **Performance Highlights**: 자연어 생성(NLG)과 자연어 이해(NLU)와 같은 여러 NLP 시나리오에서 ROSA는 거의 모든 GLUE 작업에서 LoRA보다 월등히 뛰어난 성능을 보였습니다. 또한 ROSA는 E2E 벤치마크에서도 LoRA를 능가하는 성능을 입증했습니다. ROSA의 코드와 더 상세한 실험 결과는 [여기](https URL)에서 확인할 수 있습니다.



### AVCap: Leveraging Audio-Visual Features as Text Tokens for Captioning (https://arxiv.org/abs/2407.07801)
Comments:
          Interspeech 2024

- **What's New**: 최근 몇 년간 표현 학습(representation learning)과 언어 모델(language models)의 발전으로 인해 자동 캡셔닝(Automated Captioning, AC)의 성능이 인간 수준에 매우 가까워졌습니다. 이러한 발전을 바탕으로 우리는 간단하면서도 강력한 오디오-비주얼 캡셔닝(Audi0-Visual Captioning, AVC) 프레임워크인 AVCap을 제안합니다. AVCap은 오디오-비주얼 기능을 텍스트 토큰으로 활용하여 성능뿐만 아니라 모델의 확장성과 확장이 용이합니다.

- **Technical Details**: AVCap은 오디오-비주얼 인코더 아키텍처의 최적화, 생성된 텍스트의 특성에 따른 사전 학습 모델의 적응, 캡셔닝에서 모달리티 결합의 효율성을 탐구하는 세 가지 중요한 차원을 중심으로 설계되었습니다. 이 방법은 오디오 비주얼 데이터를 인코딩하고, 오디오-비주얼 임베딩을 텍스트 임베딩과 함께 투영 및 결합하여 텍스트를 디코딩하는 세 가지 주요 구성 요소로 구성됩니다. 오디오캡스(Audiocaps) 데이터셋을 사용해 모델을 학습시킵니다.

- **Performance Highlights**: AVCap은 기존의 오디오-비주얼 캡셔닝 방법보다 모든 평가 지표에서 우수한 성능을 보였습니다. 코드 또한 공개되어 있어 연구 및 응용에 활용할 수 있습니다.



### Evaluating Large Language Models with Grid-Based Game Competitions: An Extensible LLM Benchmark and Leaderboard (https://arxiv.org/abs/2407.07796)
- **What's New**: 이번 연구에서는 Tic-Tac-Toe, Connect-Four, Gomoku와 같은 격자 기반 게임을 이용해 대형 언어 모델(LLMs)을 평가하는 새로운 벤치마크를 소개합니다. 이 오픈 소스 게임 시뮬레이션 코드는 GitHub에서 제공되며, LLM들이 서로 경쟁할 수 있게 하고, leaderboard 순위와 심층 분석을 위한 JSON, CSV, TXT, PNG 형태의 상세 데이터 파일을 생성합니다. Anthropic의 Claude 3.5 Sonnet과 Claude 3 Sonnet, Google의 Gemini 1.5 Pro와 Gemini 1.5 Flash, OpenAI의 GPT-4 Turbo와 GPT-4o, Meta의 Llama3-70B 등 주요 LLM들 간의 게임 결과가 이번 연구에서 다루어졌습니다.

- **Technical Details**: 총 2,310번의 게임이 시뮬레이션되었으며, 각 LLM과 무작위 플레이어 사이에서 7개 LLM 쌍당 5개의 세션이 진행되었습니다. 게임 종류는 세 가지 유형(Tic-Tac-Toe, Connect-Four, Gomoku)으로, 세 가지 프롬프트 유형(list, illustration(삽화), image(이미지))이 사용되었습니다. 상세한 분석은 승리 및 실격률, 기회 놓침 분석, 무효 이동 분석을 포함합니다. 실험 데이터 및 리더보드는 GitHub에 공개 접근 데이터로 제공됩니다.

- **Performance Highlights**: 분석 결과, 게임 종류와 프롬프트 유형에 따라 LLM 성능에 상당한 차이가 나타났습니다. 이는 LLM이 특정 훈련 없이 게임 규칙 이해와 전략적 사고 능력을 평가할 수 있도록 기여하며, LLM의 복잡한 의사 결정 시나리오에서의 유용성을 탐구하는 데 기초를 제공합니다.



### Fine-Tuning Large Language Models with User-Level Differential Privacy (https://arxiv.org/abs/2407.07737)
- **What's New**: 이 논문에서는 사용자의 개인 데이터를 확실히 보호하기 위해 사용자 수준 차등 프라이버시(DP)를 제공하는 대규모 언어 모델(LLMs)을 훈련시키기 위한 실용적이고 확장 가능한 알고리즘을 연구합니다. 특히, 예제 수준 샘플링(ELS)과 사용자 수준 샘플링(ULS) 두 가지 개념을 비교하며, 사용자가 다양한 예제를 가지고 있을 때 ULS가 더 나은 성능을 발휘한다고 결론내립니다.

- **Technical Details**: 이 연구에서는 두 가지 모델을 고려합니다. 첫째, DP-SGD와 예제 수준 샘플링을 결합한 DP-SGD-ELS(ELS) 방식이고, 두 번째로 DP-SGD와 사용자 수준 샘플링을 결합한 DP-SGD-ULS(ULS) 방식입니다. 예제 수준으로 DP 보장을 사용자 수준으로 변환하는 새로운 DP 회계 기법을 개발하여 DP-SGD-ELS의 엄격한 프라이버시 보장을 계산할 수 있게 했습니다. 또한, 이 기법들을 고정된 계산 예산 하에 최적화와 미세 조정을 통해 비교합니다.

- **Performance Highlights**: 테스트 결과, 사용자가 다양한 데이터 예제를 갖고 있거나 강력한 프라이버시 보장이 필요한 경우, ULS가 ELS보다 우수한 성능을 발휘하는 것으로 나타났습니다. 이를 통해 수백만 개의 파라미터와 수십만 명의 사용자 데이터를 포함한 모델에 대해서도 확장 가능함을 입증했습니다.



### PaliGemma: A versatile 3B VLM for transfer (https://arxiv.org/abs/2407.07726)
- **What's New**: PaliGemma는 SigLIP-So400m 비전 인코더와 Gemma-2B 언어 모델을 기반으로 하는 오픈 비전-언어 모델(VLM)입니다. 이 모델은 다양한 오픈월드 태스크에서 뛰어난 성능을 발휘하며, 표준 VLM 벤치마크뿐만 아니라 원격 감지 및 세분화와 같은 특화된 태스크에서도 평가되었습니다.

- **Technical Details**: PaliGemma는 PaLI 비전-언어 모델과 Gemma 언어 모델의 조합을 기반으로 합니다. SigLIP-So400m은 대규모 대조 학습을 통해 사전 훈련된 ViT-So400m 이미인코더를 사용하고, Gemma-2B는 영어 텍스트를 생성하기 위해 사전 훈련된 디코더 전용 언어 모델입니다. 이미지를 텍스트로 변환하여 다양한 태스크에서 활용할 수 있는 간단한 API를 제공하며, 모델 학습 단계는 Unimodal, Multimodal, 고해상도 프리트레이닝, 그리고 각 태스크에 특화된 모델로 전환하는 과정을 포함합니다.

- **Performance Highlights**: PaliGemma는 뛰어난 다용성 및 지식을 바탕으로 COCO 캡션, VQAv2, InfographicVQA 등의 표준 테스트뿐만 아니라 Remote-Sensing VQA, TallyVQA, 비디오 캡션 및 QA 태스크, 참조 표현 세분화 등에서도 최첨단 성능을 기록했습니다. 현재 간단하고 효율적인 구조의 이미 인코더와 언어 모델의 조합을 통해, 이전의 큰 모델들과 맞먹는 성능을 유지하는 동시에 모델 크기를 크게 줄였습니다.



### The Language of Weather: Social Media Reactions to Weather Accounting for Climatic and Linguistic Baselines (https://arxiv.org/abs/2407.07683)
Comments:
          12 pages, 5 figures

- **What's New**: 이 연구는 다양한 날씨 조건이 소셜 미디어, 특히 트위터에서의 공공 감정에 어떻게 영향을 미치는지를 탐구한 것입니다. 날씨와 관련된 감정 분석의 정확성을 높이기 위해 기후와 언어적 기준을 고려했습니다. 결과적으로, 날씨에 대한 감정 반응은 복잡하며, 날씨 변수의 조합과 지역적인 언어 차이에 의해 영향을 받는다는 것을 발견했습니다.

- **Technical Details**: 본 연구에서는 2021년 영국에서 생성된 'weather'라는 단어가 포함된 모든 트윗 데이터를 수집하였습니다. 수집된 원천 데이터에서 기계적으로 생성된 트윗을 필터링하였으며, 1,012,319개의 트윗이 최종적으로 남았습니다. 필터링 과정에서는 지리적 위치 추정, 봇 계정 제거, 날씨 계정 제거 등의 단계가 포함되었습니다. 감정 분석에는 특히 날씨와 관련된 용어와 지역적 언어적 변이를 고려한 알고리즘을 사용하였습니다.

- **Performance Highlights**: 본 연구는 트위터 데이터를 기반으로 날씨와 공공 감정 간의 관계를 보다 정확하게 이해하기 위해 맥락 민감한 방법을 적용했습니다. 이를 통해 날씨 조건의 조합, 예를 들어 고온과 고습도 vs 고온과 저습도, 및 지역적 언어 변이를 포함하여 더 정밀한 분석이 가능했습니다. 이 연구 결과는 기후 변화에 대한 예측 및 위험 전달을 개선하는 데 큰 도움이 될 수 있습니다.



### Teaching Transformers Causal Reasoning through Axiomatic Training (https://arxiv.org/abs/2407.07612)
- **What's New**: 최근, 텍스트 기반 AI 시스템이 현실 세계에서 상호작용하기 위해 '인과적 추론(causal reasoning)' 능력을 학습하는 새로운 방식이 제안되었습니다. 이 연구에서는 비용이 많이 드는 개입 데이터를 생성하지 않고도 수동 데이터(passive data)를 통해 인과적 추론을 학습할 수 있는지를 조사합니다. 특히, 인과적 속성(axiom)을 귀납적 편향(inductive bias)으로 통합하거나 데이터 값으로부터 추론하는 대신, 여러 시연을 통해 학습하는 방식을 제안합니다. 예를 들어, 작은 그래프에서 인과적 전이(transitivity) 속성을 시연하여 트랜스포머 모델을 훈련시킬 경우, 이 모델이 큰 그래프에서도 이 속성을 적용할 수 있는지를 확인합니다.

- **Technical Details**: 이번 연구에서 우리는 'axiomatic training'이라는 새로운 방식을 통해 인과적 추론을 학습합니다. 구체적으로, 어떤 가설이 참인지 여부를 결정하는 데 필요한 정보를 제공하는 'premise'와 'hypothesis', 그리고 결론을 나타내는 'result'를 포함하는 상징적 튜플 형식을 사용합니다. 이러한 방식으로 생성된 다수의 합성 튜플로 트랜스포머 모델을 훈련하여, 모델이 인과적 속성을 새로운 시나리오에 적용할 수 있는지 평가합니다.

- **Performance Highlights**: 67백만 매개변수로 구성된 트랜스포머 모델은 간단한 인과적 체인(chain)에서 훈련되었을 때, 더 긴 체인, 순서가 반전된 체인 및 분지된 구조를 포함한 새로운 그래프에서도 뛰어난 일반화 능력을 보였습니다. 이 모델의 성능은 GPT-4, Gemini Pro, 그리고 Phi-3과 같은 더 큰 언어 모델과 동등하거나 더 우수한 성능을 발휘했습니다. 또한, 모델이 다양한 평가 시나리오에서 잘 일반화되도록 하기 위해, 단순한 체인 및 일부 변형된 체인을 포함한 데이터셋으로 훈련하는 것이 중요함을 발견했습니다.



### GLBench: A Comprehensive Benchmark for Graph with Large Language Models (https://arxiv.org/abs/2407.07457)
- **What's New**: 최근 몇 년간 급격히 발전한 GraphLLM(Graph large language models) 방법을 평가하기 위한 첫 번째 종합 벤치마크 GLBench가 소개되었습니다. GLBench는 그래프 관련 문제를 다루는 데 있어 LLM(Large Language Models)의 성능을 평가하며, 감독 학습과 zero-shot 시나리오 모두에서 적용될 수 있습니다.

- **Technical Details**: GLBench는 다양한 GraphLLM 방법과 전통적인 그래프 신경망(Graph Neural Networks, GNNs)을 공정하게 평가하기 위한 일관된 데이터 처리 및 분할 전략을 채택하며, LLM을 enhancer, predictor, aligner이라는 세 가지 역할로 구분하여 성능을 분석합니다. 또한, 노드 속성과 그래프 구조 정보를 모두 포함하는 텍스트 속성 그래프(Text-attributed Graphs, TAGs)를 중심으로 실험이 진행되었습니다.

- **Performance Highlights**: 주요 실험 결과로는, 감독 학습 환경에서는 GraphLLM 방법이 전통적인 GNN을 능가했으며, 특히 LLM-as-enhancers가 가장 높은 성능을 보여주었습니다. 반면에 LLM-as-predictors 방법은 만족스럽지 못한 성과를 나타내며 종종 제어할 수 없는 출력 문제를 야기했습니다. 현재 GraphLLM 방법들에 대한 명확한 스케일링 법칙은 존재하지 않으며, 구조적 및 의미적 정보가 효과적인 zero-shot 전이에 중요한 역할을 한다는 점이 밝혀졌습니다.



### HiLight: Technical Report on the Motern AI Video Language Mod (https://arxiv.org/abs/2407.07325)
- **What’s New**: 해당 논문은 비디오-텍스트 모달 정렬(video-text modal alignment)과 'HiLight'라는 동영상 대화(framework) 프레임워크를 소개합니다. HiLight는 듀얼 비주얼 타워(dual visual towers)를 특징으로 합니다. 특히, 비디오 이해 작업을 당구 실내 장면(billiards indoor scenes)에서 다루며, 비디오와 텍스트 모달 정렬 및 사용자와의 상호작용을 중점으로 합니다.

- **Technical Details**: HiLight 프레임워크는 두 단계로 나눠집니다. 첫 번째 단계는 비디오와 텍스트 모달의 정렬을 다루는 단계로, Microsoft의 CLIP-ViP 모델을 비디오 인코더(video encoder)로 선택했습니다. 이 단계에서는 Vision Transformer(ViT)의 패치 내에 입력 객체의 공간적 관계를 명시적으로 모델링하는 방법을 사용합니다. 두 번째 단계는 사용자가 오픈 보캐뷸러리(open vocabulary)로 모델과 대화할 수 있도록 하는 비주얼-언어 모델 튜닝(fine-tuning)을 포함합니다.

- **Performance Highlights**: 실험 결과, projector layer 앞에서 발생하는 로컬 손실(local loss)이 CLIP-ViP 모델보다 향상된 시각적 대응을 보여주었습니다. 특히 언어 마스크(language mask)를 적용한 실험 설정이 더 향상된 성능을 나타냈습니다. HiLight의 듀얼 비주얼 타워 구조는 고해상도와 저해상도 시각 추출기(high-resolution and low-resolution feature extractor)를 결합하여 다양한 정보 콘텐츠를 얻어냅니다.



### Remastering Divide and Remaster: A Cinematic Audio Source Separation Dataset with Multilingual Suppor (https://arxiv.org/abs/2407.07275)
Comments:
          Submitted to the 5th IEEE International Symposium on the Internet of Sounds

- **What's New**: 최신 DnR 데이터셋 버전 3(DnR v3)은 비대화 스템에서의 목소리 콘텐츠, 라우드니스 분포, 마스터링 과정, 그리고 언어 다양성 문제를 해결합니다. 특히 DnR v3의 대화 스템에는 게르만어, 로망스어, 인도-아리안어, 드라비다어, 마라요-폴리네시아어, 반투어 등 30개 이상의 언어가 포함되어 있습니다.

- **Technical Details**: DnR v3는 훈련, 검증, 테스트 각 변형마다 6000, 600, 1200개의 클립을 포함합니다. 모든 오디오 데이터는 24비트 깊이로 48kHz로 샘플링된 모노 트랙으로 제공됩니다. 음악 스템은 FMA(FMA) 데이터셋에서 제공되며, 비상업적 내지 파생작업 사용을 허용하는 트랙을 선별하여 포함합니다. 대화 스템은 SMAD(음성-음악 활동 감지) 모델을 이용해 음성 및 보컬이 포함되지 않은 연속적인 구간으로 선별됩니다.

- **Performance Highlights**: Bandit 모델을 이용한 기준 결과는 다언어 데이터를 훈련할 경우 언어 데이터 가용성이 낮은 경우에도 모델의 일반화 성능이 크게 향상됨을 보여줍니다. 고가용성 언어에서도 다언어 모델은 전용 모노언어 모델과 동등하거나 더 나은 성능을 보여줍니다.



### ConvNLP: Image-based AI Text Detection (https://arxiv.org/abs/2407.07225)
Comments:
          11 pages, 5 figures

- **What's New**: 이번 논문에서는 학문적 부정행위를 방지하기 위해 Generative AI 기술을 활용한 방식을 소개합니다. 특히, 새로운 Convolutional Neural Network(CNN) 구조인 ZigZag ResNet과 ZigZag Scheduler를 통해 AI 생성 텍스트를 식별하는 기술을 제안합니다. 이 방식은 이미지 처리 모델을 사용하는 텍스트 임베딩 접근법을 도입하여 AI 생성 텍스트를 식별합니다.

- **Technical Details**: 논문에서 사용된 방법론은 다음과 같습니다. 텍스트-이미지 임베딩을 통해 문장의 공간적, 언어적 관계를 유지하고, 이를 이미지 형태로 변환하여 CNN 기반의 분류기로 입력합니다. 본 연구에서는 Universal Sentence Encoder를 사용하여 단어 임베딩을 생성하고, 이를 시각적 표현 이미지로 변환한 후 ZigZag ResNet 모델에 입력합니다. 특히 ZigZag Scheduler를 통해 일반화 성능을 개선했습니다.

- **Performance Highlights**: 제안된 모델은 6개의 서로 다른 최첨단 LLM 데이터를 기반으로 강력한 일반화 성능을 나타냈습니다. Inter-domain 및 Intra-domain 테스트 데이터에서 평균 88.35%의 AI 생성 텍스트 검출률을 기록했으며, Vanilla ResNet에 비해 약 4%의 성능 개선을 이루었습니다. 모델의 문장당 엔드투엔드 추론 지연 시간은 2.5ms 이하로, 매우 빠르고 경량화된 솔루션을 제공합니다.




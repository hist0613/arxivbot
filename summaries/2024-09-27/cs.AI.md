New uploads on arXiv(cs.CL)

### Open-World Evaluation for Retrieving Diverse Perspectives (https://arxiv.org/abs/2409.18110)
- **What's New**: 새로운 연구 Benchmark for Retrieval Diversity for Subjective questions (BeRDS)을 소개하여, 여러 관점을 포괄하는 문서 셋을 검색하는 과정을 분석하였습니다.

- **Technical Details**: 이 연구는 복잡한 질문에 대한 다양한 관점을 검색하는 방식을 다룹니다. 단순한 문자열 일치에 의존하지 않고 언어 모델 기반의 자동 평가자를 사용하여 문서가 특정 관점을 포함하는지 판단합니다. 세 가지 유형의 말뭉치(Wikipedia, 웹 스냅샷, 검색 엔진 결과 활용)를 조합한 성능을 평가합니다.

- **Performance Highlights**: 기존의 검색 시스템은 다양한 관점을 포함한 문서 세트를 33.74%의 경우에만 검색할 수 있었습니다. 쿼리 확장 및 다양성 중심의 재정렬 방법을 적용하여 성능 향상을 관찰하였으며, 이러한 접근법들이 밀집 검색기와 결합된 경우 더욱 강력한 결과를 보였습니다.



### Unveiling the Role of Pretraining in Direct Speech Translation (https://arxiv.org/abs/2409.18044)
Comments:
          EMNLP 2024

- **What's New**: 본 연구에서는 직접 음성-텍스트 번역 시스템의 훈련 역학을 처음으로 분석하여, 사전훈련(pretraining)의 역할과 효과적인 훈련 방법을 제안합니다. 구체적으로, 기존의 사전훈련된 인코더를 사용하는 시스템과 무작위로 초기화된 모델을 비교했으며, 무작위 초기화된 모델이 훈련 초기에 인코더의 정보를 제대로 활용하지 못하는 경향이 있음을 확인했습니다.

- **Technical Details**: 본 연구는 Transformer(Vaswani et al., 2017) 아키텍처에서의 디코더 교차-어텐션 메커니즘을 수정하여 훈련 초기 단계에서 소스 정보를 더 잘 통합하도록 하였습니다. ALTI+ 해석 가능성 방법을 활용하여 훈련 데이터의 소스 기여(source contribution)를 분석하고, 이를 통해두 가지 훈련 전략의 효과를 비교합니다. 이로써 음성 번역 모델이 처음 몇 번의 업데이트 동안 인코더의 출력을 조기에 활용하도록 유도합니다.

- **Performance Highlights**: 변경된 구조를 통해, 무작위 초기화된 모델이 사전훈련된 모델과 유사한 성능을 달성하면서도 훈련 시간을 줄일 수 있음을 보여주었습니다. 무작위 초기화된 모델은 최초 30k 업데이트 후에야 안정적인 수준의 소스 기여를 이루는 반면, 사전훈련된 모델은 6k 업데이트 후 안정성에 도달합니다. 이는 음성 번역을 위한 인코더의 훈련 과정이 문장 번역에 비해 더 복잡하다는 점을 시사합니다.



### Automated Detection and Analysis of Power Words in Persuasive Text Using Natural Language Processing (https://arxiv.org/abs/2409.18033)
- **What's New**: 본 연구에서는 마케팅, 정치 및 동기 부여 글쓰기 분야에서 중요한 역할을 하는 power words(파워 워드)를 자동으로 탐지하고 분석하는 방법론을 제안합니다.

- **Technical Details**: Python의 TextBlob 라이브러리와 사용자 정의 어휘를 사용하여 설득력 있는 텍스트 내에서 power words의 존재 및 빈도를 식별합니다.

- **Performance Highlights**: 다양한 도메인에서 다양한 데이터 세트를 조사하여 power words의 효과에 대한 통찰력을 제공합니다. 이는 콘텐츠 제작자, 광고주 및 정책 입안자에게 실용적인 응용 프로그램을 제공합니다.



### DARE: Diverse Visual Question Answering with Robustness Evaluation (https://arxiv.org/abs/2409.18023)
- **What's New**: 이 연구에서는 Vision Language Models (VLMs)의 성능을 평가하기 위해 DARE(Diverse Visual Question Answering with Robustness Evaluation)라고 하는 새로운 VQA 벤치마크를 소개합니다. DARE는 다양한 카테고리의 질문을 포함하며, 모델의 강건성(robustness)을 평가하기 위해 여러 변화를 도입합니다.

- **Technical Details**: DARE는 다음과 같은 네 가지 강건성 요소를 통해 평가됩니다: 1) 안내문의 변형, 2) 대답 옵션 세트의 변형, 3) 출력 형식의 변형, 4) 정답 개수의 변형. 이러한 다양한 평가 요소는 VLMs의 훈련 과정에서 학습한 편향을 드러내는 데 도움이 됩니다. DARE는 다섯 가지 카테고리에 걸쳐 복잡한 시나리오를 포괄하고 각 카테고리에 대해 세밀하게 구축된 평가 항목들을 포함합니다.

- **Performance Highlights**: 최신 VLM 모델들은 여전히 조건부 카운팅(conditional counting) 및 공간적 추론(spatial reasoning)과 같은 '인간에게는 간단한' 비전 이해 작업에서 어려움을 겪고 있으며, 특히 정답 옵션에 대한 변형에 대해 강건하지 못합니다. VLM들은 변형된 질문에서도 일관되게 성능을 발휘하지 못하며, LLaVA 1.6 및 Idefics2는 정답을 하나만 정확하게 표시하는 경향이 있습니다. DARE는 모델의 성능을 더 잘 이해할 수 있는 기회를 제공하며, 이 연구는 VLM의 향후 발전에 기여할 것으로 기대됩니다.



### Multilingual Evaluation of Long Context Retrieval and Reasoning (https://arxiv.org/abs/2409.18006)
Comments:
          Under review

- **What's New**: 이번 연구는 다국어 환경에서의 LLM(대형 언어 모델) 성능을 분석하고, 특히 길이 있는 문맥과 여러 개의 숨겨진 목표 문장을 다루는 능력에 초점을 맞췄습니다.

- **Technical Details**: 연구에서는 영어, 베트남어, 인도네시아어, 스와힐리어, 소말리어의 5개 언어에서 여러 가지 LLM을 평가했습니다. Gemini-1.5와 GPT-4o 모델이 단일 목표 문장을 처리할 때 영어에서 96%의 정확도를 보였으나, 소말리어에서는 36%로 떨어졌습니다. 세 개의 목표 문장을 사용할 경우 영어는 40%, 소말리어는 0%로 성능이 급락했습니다.

- **Performance Highlights**: 모델 성능은 문맥 길이가 증가하고 자원 수준이 낮은 언어로 이동할수록 급격히 감소했습니다. 추론 작업은 모든 언어에 대해 검색 작업보다 더 어려운 것으로 나타났고, 다국어 상황에서 '바늘 찾기' 평가에서도 모델의 한계가 드러났습니다.



### BEATS: Optimizing LLM Mathematical Capabilities with BackVerify and Adaptive Disambiguate based Efficient Tree Search (https://arxiv.org/abs/2409.17972)
- **What's New**: 본 연구는 대규모 언어 모델(LLMs)이 수학 문제 해결 능력을 향상시키기 위한 새로운 방법인 BEATS를 제안합니다. 이 방법은 모델이 문제를 단계별로 해결하도록 유도하는 적절히 설계된 프롬프트(prompt)를 활용하며, 생성된 답변의 정확성을 검증하기 위한 새로운 역검증(back-verification) 기술을 도입합니다.

- **Technical Details**: BEATS는 모델이 문제를 반복적으로 재작성(rewrite)하고, 한 단계씩 진전을 이루며, 이전 단계를 바탕으로 답변을 생성하도록 유도합니다. 또한, 기존의 투표 기반 검증 방식 대신 질문과 정답을 모델에 재제출하여 정답의 정확성을 판단하는 역검증 방식을 적용합니다. 마지막으로, 가지치기(pruning) 트리 탐색을 통해 검색 시간을 최적화하면서도 성과를 높입니다.

- **Performance Highlights**: BEATS 방법은 Qwen2-7B-Instruct 모델을 기반으로 할 때 MATH 데이터셋에서 점수를 36.94에서 61.52로 개선시켰으며, 이는 GPT-4의 42.5를 초월한 성과입니다. 추가적으로 MATH, GSM8K, SVAMP, SimulEq, NumGLUE 등 여러 데이터셋에서도 경쟁력 있는 결과를 달성하였습니다.



### The Hard Positive Truth about Vision-Language Compositionality (https://arxiv.org/abs/2409.17958)
Comments:
          ECCV 2024

- **What's New**: 이 논문에서는 CLIP와 같은 최첨단 비전-언어 모델의 조합 가능성(compositionality) 부족 문제를 다룹니다. 기존 벤치마크에서 힘들었던 점은 이러한 모델들이 하드 네거티브(hard negative)를 사용하여 개선된 성능을 과장해왔다는 것입니다. 이번 연구는 하드 포지티브(hard positive)를 포함한 대상에서 CLIP의 성능이 12.9% 감소하는 반면, 인간은 99%의 정확도를 보인다는 주장을 제기합니다.

- **Technical Details**: 저자들은 112,382개의 하드 네거티브 및 하드 포지티브 캡션으로 평가 데이터셋을 구축하였습니다. CLIP을 하드 네거티브로 파인튜닝(finetuning)할 경우 성능이 최대 38.7%까지 감소했으며, 하드 포지티브가 포함될 때의 효과를 분석하여 조합 가능한 성능을 개선할 수 있음을 밝혀냈습니다.

- **Performance Highlights**: CLIP 모델은 하드 네거티브로 훈련했을 때, 기존 벤치마크에서 성능이 개선되었음에도 불구하고, 하드 포지티브에 대한 성능이 동시에 저하되었습니다. 반면, 하드 네거티브와 하드 포지티브를 동시에 사용하여 훈련했을 때, 두 가지 모두 성능 개선이 이루어졌습니다. 이러한 연구는 조합 가능성에 대한 새로운 차원을 탐구하며 향후 연구 방향을 제시합니다.



### On Translating Technical Terminology: A Translation Workflow for Machine-Translated Acronyms (https://arxiv.org/abs/2409.17943)
Comments:
          AMTA 2024 - The Association for Machine Translation in the Americas organizes biennial conferences devoted to researchers, commercial users, governmental and NGO users

- **What's New**: 이번 논문에서는 기계 번역(Machine Translation, MT) 시스템에서 약어의 모호성 제거(acronym disambiguation)를 제안함으로써, 약어 번역의 정확성을 높이고자 하는 새로운 접근 방식을 소개합니다. 또한 새로운 약어 말뭉치(corpus)를 공개하고, 이를 기반으로 한 검색 기반 임계값(thresholding) 알고리즘을 실험하여 기존의 Google Translate와 OpusMT보다 약 10% 더 나은 성능을 보였습니다.

- **Technical Details**: 기계 번역 시스템의 약어 번역 정확도를 향상시키기 위해, 4단계의 고레벨 프로세스를 제안하였습니다. 이 프로세스는 (1) Google Translate를 사용하여 FR-EN 번역 수행, (2) 영어 장기형(long form, LF)과 단기형(short form, SF) 추출, (3) AB3P 툴을 사용한 약어 가설 생성, (4) 검색 기법을 통한 가설 검증 및 평가입니다. 이 방법들은 텍스트에서 사용된 기술 용어의 신뢰성을 향상시키는데 기여하고자 합니다.

- **Performance Highlights**: Google Translate와 OpusMT와 비교하여, 제안하는 임계값 알고리즘은 약 10%의 성능 향상을 보여주었습니다. 우리의 연구에서는 전문 번역사들이 자주 직면하는 용어 오류를 감소시키는 방안을 제시하고 있으며, 이를 통해 MT 시스템에서의 기술 용어 번역의 적합성을 증대시키는 데 기여할 것입니다.



### Predicting Anchored Text from Translation Memories for Machine Translation Using Deep Learning Methods (https://arxiv.org/abs/2409.17939)
Comments:
          AMTA 2024 - The Association for Machine Translation in the Americas organizes biennial conferences devoted to researchers, commercial users, governmental and NGO users

- **What's New**: 본 논문은 번역 메모리(Translation Memoires, TMs)와 컴퓨터 보조 번역(Computer-Aided Translation, CAT) 도구에서의 구문 정정 기법인 퍼지 매치 리페어(Fuzzy-Match Repair, FMR) 기술을 발전시키는 데 집중하고 있습니다. 특히, 기존의 기계 번역(Machine Translation, MT) 기법 대신 Word2Vec, BERT, 그리고 GPT-4와 같은 머신 러닝(machine learning) 기반 접근 방식을 사용하여 고정된 단어(anchor word) 번역의 정확성을 향상시킬 수 있는 방법을 제시합니다.

- **Technical Details**: 이 연구는 번역 시스템에서 고정된 단어의 번역을 개선하기 위해 네 가지 기술을 실험했습니다: (1) Neural Machine Translation(NMT), (2) BERT 기반 구현, (3) Word2Vec, (4) OpenAI GPT-4. 특히, 고정된 단어는 두 개의 단어 사이에 위치하며, 연속적인 단어의 가방(CBOW) 패러다임을 따른다고 설명합니다. 내재된 문맥(window) 내에서 각 단어에 가중치를 부여하여 주변 단어가 예측에 미치는 영향을 극대화할 수 있음을 강조하고 있습니다.

- **Performance Highlights**: 실험 결과, Word2Vec, BERT 및 GPT-4는 프랑스어에서 영어로의 번역에 있어 기존의 Neural Machine Translation 시스템보다 유사하거나 더 나은 성능을 보였습니다. 특히, 각 접근 방식이 고정된 단어 번역에서 성공적으로 작동하는 경우를 다루고 있습니다.



### The Lou Dataset -- Exploring the Impact of Gender-Fair Language in German Text Classification (https://arxiv.org/abs/2409.17929)
- **What's New**: 이 연구는 성 평등 언어(gender-fair language)의 효과를 평가하기 위해 최초의 고품질 독일어 텍스트 분류 데이터셋인 Lou를 소개합니다. 이 데이터셋은 성 평등 언어가 언어 모델에 미치는 영향을 체계적으로 분석하는 데 중점을 두고 개발되었습니다.

- **Technical Details**: Lou 데이터셋은 3.6k개의 개정 사례를 포함하며, 7개의 분류 작업(예: stance detection, toxicity classification)을 지원합니다. 이 연구는 성 평등 언어가 언어 모델의 예측에 미치는 영향을 조사하고 특정 개정 전략의 유효성을 평가합니다. 성 평등 언어는 기존의 성별 편향과 관련된 연구를 확장하는 중요한 기회를 제공합니다.

- **Performance Highlights**: 실험 결과, 성 평등 언어는 예측에 상당한 영향을 미치며, 레이블 플립(label flips)과 불확실성 감소가 관찰되었습니다. 특히, 성 평등 언어는 낮은 레이어에서 언어 모델이 개정된 사례를 처리하는 방식에 영향을 주어 예측의 변동성을 초래합니다. 본 연구는 독일어뿐만 아니라 다른 언어에서도 비슷한 패턴이 나타날 가능성을 제시합니다.



### Pioneering Reliable Assessment in Text-to-Image Knowledge Editing: Leveraging a Fine-Grained Dataset and an Innovative Criterion (https://arxiv.org/abs/2409.17928)
Comments:
          EMNLP24 Findings

- **What's New**: 이번 연구에서는 Text-to-Image (T2I) diffusion 모델의 지식 편집을 위한 새로운 프레임워크를 제안합니다. 특히, 새로운 데이터셋 CAKE를 제작하고, 평가지표를 개선하는 adaptive CLIP threshold를 도입하며, Memory-based Prompt Editing (MPE) 접근 방식을 통해 효과적인 지식 업데이트를 구현합니다.

- **Technical Details**: T2I 모델의 편집 성능을 평가하기 위해 CAKE라는 데이터셋에서 paraphrase와 다중 객체 테스트를 포함하여 보다 정밀한 평가를 가능하게 합니다. 또한, 기존의 이진 분류 기반의 평가 방식에서 벗어나 이미지가 목표 사실과 '충분히' 유사한지를 측정하는 adaptive CLIP threshold라는 새로운 기준을 제안합니다. 이와 함께, MPE는 외부의 메모리에 모든 사실 편집을 저장하여 입출력 프롬프트의 오래된 부분을 수정합니다. 이를 통해 MPE는 기존 모델 편집기보다 뛰어난 성과를 보여줍니다.

- **Performance Highlights**: MPE 접근법은 기존의 모델 편집 기법보다 전반적인 성능과 적용 가능성에서 더 우수한 결과를 보였습니다. 연구 결과는 T2I 지식 편집 방법의 신뢰할 수 있는 평가를 촉진할 것으로 예상됩니다.



### Atlas-Chat: Adapting Large Language Models for Low-Resource Moroccan Arabic Dialec (https://arxiv.org/abs/2409.17912)
- **What's New**: 본 논문에서는 모로코 아랍어, 즉 다리자(Darija)에 맞춰 특별히 개발된 최초의 대형 언어 모델 집합인 Atlas-Chat을 소개합니다. 다리자 언어 자원과 지침 데이터를 통합하여 구축하였으며, 신규 데이터셋을 수동 및 합성적으로 생성했습니다. 이를 통해 우리의 모델이 다리자 지침을 따르고 표준 NLP 작업을 수행하는 데 탁월한 능력을 보입니다.

- **Technical Details**: Atlas-Chat 모델은 9B와 2B 파라미터로 구성되며, 다리자 전용 지침 데이터셋으로 정밀하게 조정(fine-tuned)되었습니다. 이 모델들은 DaijaMMLU benchmark에서 13B 모델을 13% 초과하는 성능 향상을 보여주며, 실험적 결과를 통해 다양한 fine-tuning 전략과 기본 모델 선택에 대한 최적 구성을 실험적으로 분석했습니다.

- **Performance Highlights**: Atlas-Chat 모델은 LLaMa, Jais 및 AceGPT와 같은 기존 아랍어 전문 LLM보다 뛰어난 성능을 발휘하며, 자동화된 메트릭과 시뮬레이션된 승률을 기준으로 평가되었습니다. 이로써, 자원이 부족한 언어 변종에 대한 지침 조정의 설계 방법론을 제공하고, 모든 자원을 공개 접근 가능하게 하여 해당 연구의 포괄성을 강조합니다.



### EMMA-500: Enhancing Massively Multilingual Adaptation of Large Language Models (https://arxiv.org/abs/2409.17892)
- **What's New**: 이번 연구에서는 546개의 언어로 작성된 텍스트를 기반으로 한 대규모 다국어 언어 모델 EMMA-500을 소개합니다. EMMA-500은 저자원 언어의 언어 범위를 개선하기 위해 설계되었습니다. 이를 위해 MaLA 코퍼스를 구성하여 다양하고 포괄적인 데이터 세트를 포함시켰습니다.

- **Technical Details**: EMMA-500은 Llama 2 7B 모델을 기반으로 하여 계속적인 사전 훈련(cont continual pre-training)을 수행하였으며, 이는 저자원 언어에 대한 언어 능력을 확장하는 데 효과적입니다. 이상적으로 구성된 MaLA 코퍼스는 939개의 언어를 포함하며, 74억 개의 공백으로 구분된 토큰을 포함하고 있습니다.

- **Performance Highlights**: EMMA-500 모델은 상반기 후속 결과에서 Llama 2 기반 모델과 다국어 기준선에 비해 두드러진 성과를 보여주었습니다. 모형 크기가 4.5B에서 13B 사이의 모델 중 EMMA-500은 최저의 부정 로그 우도(negative log-likelihood)를 기록하였으며, 일상 추론(common sense reasoning), 기계 번역(machine translation), 개방형 생성(open-ended generation) 작업에서도 우수한 성과를 보였습니다.



### PEDRO: Parameter-Efficient Fine-tuning with Prompt DEpenDent Representation MOdification (https://arxiv.org/abs/2409.17834)
Comments:
          arXiv admin note: text overlap with arXiv:2405.18203

- **What's New**: 이 논문에서는 새로운 PEFT 방법론인 PEDRO(Prompt dEpenDent Representation mOdification)를 소개합니다. PEDRO는 각 Transformer 레이어에 경량의 벡터 생성기(Vector Generator)를 통합하여 입력 프롬프트에 따라 변수를 생성합니다. 이 변수를 통해 LLM의 내부 표현을 수정함으로써 LLM의 동작을 조정합니다.

- **Technical Details**: PEDRO는 Transformer 레이어에서 수정 벡터를 직접 생성하는 메커니즘을 포함하고 있습니다. 이 과정에서 Vector Generator는 입력 프롬프트의 숨겨진 상태를 입력으로 받아 수정 벡터를 출력합니다. 이 구조는 구조상 경량화되어 있으며 다양한 작업에서 PEDRO의 효과를 입증하기 위해 광범위한 실험을 수행했습니다.

- **Performance Highlights**: PEDRO는 비슷한 수의 조정 가능한 파라미터를 사용할 때 최근의 PEFT 벤치마크를 초과하는 성능을 보이며, 단일 백본 다중 테넌트 배포 모델에서는 LoRA보다 더 높은 효율성을 나타냅니다. 이는 산업적인 응용 가능성을 더욱 높이는 결과로 이어집니다.



### BeanCounter: A low-toxicity, large-scale, and open dataset of business-oriented tex (https://arxiv.org/abs/2409.17827)
- **What's New**: 이 논문은 비즈니스 관련 공개 정보를 기반으로 한 159B 토큰의 새로운 데이터셋인 BeanCounter를 소개합니다. 이 데이터셋은 기존의 웹 기반 데이터셋보다 사실적이고, 품질이 높은 동시에 독성이 적은 컨텐츠를 제공합니다.

- **Technical Details**: BeanCounter는 공공 도메인 비즈니스 정보에서 추출된 데이터로, 다른 일반적으로 사용되는 데이터셋과는 최소 0.1%만 겹친다고 보고됩니다. 개인 식별 정보의 포함, 품질 저하 및 편향 등의 문제를 다루며, 모든 항목에 정확한 타임스탬프를 제공합니다.

- **Performance Highlights**: BeanCounter를 지속적으로 재훈련한 두 개의 LLM(대형 언어 모델)을 평가한 결과, 독성 생성이 18-33% 감소했으며, 재무 영역 내 성능이 개선되었습니다. BeanCounter는 저독성과 높은 품질의 도메인별 데이터의 새로운 출처로서, 다중 억 파라미터 LLM 교육에 충분한 규모를 자랑합니다.



### Inference-Time Language Model Alignment via Integrated Value Guidanc (https://arxiv.org/abs/2409.17819)
Comments:
          EMNLP 2024 Findings

- **What's New**: 본 논문에서는 대형 언어 모델(Large Language Models, LLMs) 조정의 복잡성을 피하면서도 효율적으로 인간의 선호에 부합할 수 있게 하는 새로운 방법인 통합 가치 안내(Integrated Value Guidance, IVG)를 소개합니다. IVG는 암묵적(implicit) 및 명시적(explicit) 가치 함수(value function)를 사용하여 언어 모델의 디코딩(decoding)을 유도합니다.

- **Technical Details**: IVG는 두 가지 형태의 가치 함수를 결합합니다. 암묵적 가치 함수는 각 토큰(token)별 샘플링에 적용되고, 명시적 가치 함수는 청크(chunk) 단위의 빔 탐색(beam search)에 사용됩니다. IVG는 다양한 작업에서의 효과를 검증하며, 전통적인 방법들을 능가합니다. 특히, IVG는 gpt2 기반의 가치 함수로부터의 유도 덕분에 감정 생성과 요약 작업에서 성능을 크게 향상했습니다.

- **Performance Highlights**: IVG는 AlpacaEval 2.0과 같은 어려운 지침 따르기 벤치마크에서, 전문가 튜닝된 모델과 상용 모델 모두에서 길이 제어된 승률이 크게 향상되는 것을 보여줍니다. 예를 들어, Mistral-7B-Instruct-v0.2 모델은 19.51%에서 26.51%로, Mixtral-8x7B-Instruct-v0.1 모델은 25.58%에서 33.75%로 증가했습니다.



### Self-supervised Preference Optimization: Enhance Your Language Model with Preference Degree Awareness (https://arxiv.org/abs/2409.17791)
Comments:
          Accepted at EMNLP 2024 Findings

- **What's New**: 최근에는 대형 언어 모델(LLM)의 보상 모델을 인간 피드백(Human Feedback) 기반 강화 학습(Reinforcement Learning from Human Feedback, RLHF) 방법으로 대체하려는 관심이 증가하고 있습니다. 이 연구에서는 Self-supervised Preference Optimization(SPO) 프레임워크를 제안하여, LLM이 인간의 선호도를 더 잘 이해하고 조정할 수 있도록 합니다.

- **Technical Details**: 본 논문에서는 SPO라는 새로운 자가 감독(preference optimization) 프레임워크를 제안합니다. 이 방법은 LLM의 출력에서 중요한 내용을 선택적으로 제거하여 다양한 선호도 정도의 응답을 생성합니다. 이러한 응답은 자기 감독 모듈로 분류되어 주 손실 함수와 결합되어 LLM을 동시에 최적화합니다.

- **Performance Highlights**: 실험 결과, SPO는 기존의 선호 최적화 방법들과 통합되어 LLM의 성능을 유의미하게 향상시킬 수 있으며, 두 가지 다양한 데이터셋에서 최첨단 결과를 달성했습니다. LLM이 선호도 정도를 구분하는 능력을 높이는 것이 여러 작업에서 성능 향상에 기여한다는 것을 보여주었습니다.



### Faithfulness and the Notion of Adversarial Sensitivity in NLP Explanations (https://arxiv.org/abs/2409.17774)
Comments:
          Accepted as a Full Paper at EMNLP 2024 Workshop BlackBoxNLP

- **What's New**: 본 논문에서는 설명 가능한 AI의 신뢰성을 평가하기 위한 새로운 접근법인 'Adversarial Sensitivity'를 소개합니다. 이 방식은 모델이 적대적 공격을 받을 때 설명자의 반응에 중점을 둡니다.

- **Technical Details**: Adversarial Sensitivity는 신뢰성을 평가하는 지표로, 적대적 입력 변화에 대한 설명자의 민감도를 포착합니다. 이를 통해 신뢰성이 기존 평가 기술의 중대한 한계를 극복하고, 기존의 설명 메커니즘의 정확성을 정량화합니다.

- **Performance Highlights**: 연구팀은 세 개의 텍스트 분류 데이터셋에 대해 여섯 개의 최첨단 후속 설명기에서 제안된 신뢰성 테스트를 수행하고, 인기 있는 설명기 테스트와의 (비)일관성을 보고하였습니다.



### Integrating Hierarchical Semantic into Iterative Generation Model for Entailment Tree Explanation (https://arxiv.org/abs/2409.17757)
- **What's New**: 본 논문은 Controller-Generator 프레임워크(HiSCG)를 기반으로 문장의 계층적 의미(Hierarchical Semantics)를 통합하여 신뢰할 수 있는 설명(Explanation)을 생성하는 새로운 아키텍처를 제안합니다. 이 방법은 동일 계층 및 인접 계층 간의 문장 간 계층적 의미를 처음으로 고려하여 설명의 향상을 이끌어냅니다.

- **Technical Details**: HiSCG 아키텍처는 세 가지 주요 구성 요소로 나뉩니다: 계층적 의미 인코더(Hierarchical Semantic Encoder), 선택 컨트롤러(Selection Controller), 중간 생성 모듈(Intermediate Generation Module). 이 구조는 계층적 연결을 통해 관련 사실을 클러스터링하고, 이러한 사실을 조합하여 결론을 생성하는 과정을 최적화합니다. 각 모듈은 계층적 정보의 활용을 극대화하도록 설계되었습니다.

- **Performance Highlights**: 제안된 방법은 EntailmentBank 데이터셋의 세 가지 설정에서 기존 다른 방법들과 비교하여 동등한 성능을 달성했으며, 두 개의 도메인 외 데이터셋에서 일반화 능력도 입증되었습니다.



### MIO: A Foundation Model on Multimodal Tokens (https://arxiv.org/abs/2409.17692)
Comments:
          Technical Report. Codes and models will be available soon

- **What's New**: MIO라는 새로운 기초 모델이 등장했습니다. 이는 음성, 텍스트, 이미지 및 비디오를 이해하고 생성할 수 있는 멀티모달 토큰 기반의 모델로서, end-to-end 및 autoregressive 방식으로 작동합니다. MIO는 기존의 모델들이 갖고 있지 않았던 all-to-all 방식으로의 이해와 생성을 지원하며, 기존의 비공식 모델들(GPT-4o 등)을 대체할 수 있습니다.

- **Technical Details**: MIO는 causal multimodal modeling에 기반해 4단계의 훈련 과정을 거칩니다: (1) alignment pre-training, (2) interleaved pre-training, (3) speech-enhanced pre-training, (4) 다양한 텍스트, 비주얼, 음성 작업에 대한 포괄적인 감독 하에 fine-tuning을 수행합니다. MIO는 discrete multimodal tokens을 사용하여 학습되며, 이는 대조 손실(contrastive loss)과 재구성 손실(reconstruction loss) 기법을 통해 semantical representation과 low-level features를 포착합니다.

- **Performance Highlights**: MIO는 이전의 dual-modal 및 any-to-any 모델들, 심지어 modality-specific baselines와 비교해서 경쟁력 있는 성능을 보이며, interleaved video-text generation, 시각적 사고의 연쇄(chain-of-visual-thought reasoning), 시각적 가이드라인 생성, 이미지 편집 기능 등 고급 기능을 구현합니다.



### Zero- and Few-shot Named Entity Recognition and Text Expansion in Medication Prescriptions using ChatGP (https://arxiv.org/abs/2409.17683)
- **What's New**: 이번 연구는 ChatGPT 3.5를 사용하여 약물 처방의 데이터 통합 및 해석을 자동화하여 사용자와 기계 모두에게 이해하기 쉬운 형식으로 제공하는 방법을 제시합니다. 이는 자유 텍스트 형태의 약물 진술에서 의미 있는 정보를 구조화하고 확장할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 연구는 Named-Entity Recognition (NER) 및 Text Expansion (EX) 기술을 활용하였습니다. NER은 약물 이름, 농도, 단위, 투여 경로 및 복용 지침을 식별하며, EX는 이를 명확하고 구체적인 형태로 변환합니다. 최적화된 프롬프트를 사용해 NER의 평균 F1 점수는 0.94에 도달하였고, EX는 0.87에 도달하였습니다.

- **Performance Highlights**: 이 연구는 ChatGPT를 사용한 NER 및 EX 작업에서 우수한 성능을 입증하였습니다. 특히, few-shot 학습 접근법을 통해 잘못된 정보를 생성하는 현상(hallucination)을 방지할 수 있었으며, 이는 약물 안전 데이터 처리 시 중요합니다.



### Cross-lingual Human-Preference Alignment for Neural Machine Translation with Direct Quality Optimization (https://arxiv.org/abs/2409.17673)
Comments:
          17 pages, 1 figure

- **What's New**: 이 논문에서는 일반적인 기초 모델을 특정 작업에 맞추기 위해 인간의 피드백(Feedback)으로부터 강화 학습(Reinforcement Learning from Human Feedback, RLHF) 및 직접 선호 최적화(Direct Preference Optimization, DPO)와 같은 작업 정렬 알고리즘을 적용하는 방법을 제시합니다. 특히, 신경 기계 번역(Neural Machine Translation, NMT)에 작업 정렬을 적용하여 데이터의 불일치를 해결하는 접근방식을 소개합니다.

- **Technical Details**: 직접 품질 최적화(Direct Quality Optimization, DQO)라는 DPO의 변형을 도입하며, 이 방법은 인간의 선호를 대리하는 사전 훈련된 번역 품질 추정 모델을 활용합니다. DQO는 자동 척도(BLEU, COMET 등) 및 인간 평가를 통해 번역 품질의 향상을 검증하였습니다.

- **Performance Highlights**: DQO는 다국어 모델에 적용했을 때, 성능을 증가시키고 인력 선호를 향상하며, 모델 출력 분포와 훈련 데이터 분포 간의 거리를 늘리는 결과를 보여줍니다. 특히, DQO에 사용된 데이터에 포함되지 않은 언어와 언어 계통에서도 성능 향상이 관찰되었으며, 이는 일반적인 번역 작업에서 요구되는 행동의 전이 학습(Transfer Learning) 외에도 언어 특정적 언어적 특징들에 대한 향상이 있음을 시사합니다.



### Efficient In-Domain Question Answering for Resource-Constrained Environments (https://arxiv.org/abs/2409.17648)
Comments:
          6 pages, 2 tables

- **What's New**: 이 연구에서는 Retrieval Augmented Fine Tuning (RAFT) 및 Parameter-Efficient Fine Tuning (PEFT) 기법인 Low-Rank Adaptation (LoRA)을 결합하여 자원 제한 환경에서 질문 답변 (QA) 성능을 개선하는 새로운 접근 방식인 CRAFT를 제안합니다.

- **Technical Details**: RAFT는 정보 검색을 결합하여 LLM이 관련 콘텐츠를 기반으로 질문에 효과적으로 답할 수 있게 해주는 기법입니다. LoRA는 훈련된 경량 어댑터를 사용하여 모델을 효율적으로 미세 조정합니다. CRAFT는 RAFT와 LoRA를 결합하여 미세 조정 및 저장 요구 사항을 줄이면서도 RAG 성능을 유지합니다.

- **Performance Highlights**: CRAFT는 7-8억 개의 파라미터를 가진 LLM에서 빠른 추론 시간과 메모리 효율성을 제공하며, 자원 제한 환경에서도 기계 학습 모델의 실용적 적용을 가능하게 합니다.



### T3: A Novel Zero-shot Transfer Learning Framework Iteratively Training on an Assistant Task for a Target Task (https://arxiv.org/abs/2409.17640)
- **What's New**: 이 논문에서는 T3이라 약칭되는 새로운 제로-샷 (zero-shot) 전이 학습 프레임워크를 제안하여, 긴 텍스트 요약(Long Text Summarization) 작업을 위한 효과적인 솔루션을 제공합니다. 이 프레임워크는 보조 작업(assistant task)과 목표 작업(target task) 간의 관계를 활용하여 LLM을 학습시킵니다.

- **Technical Details**: T3는 보조 작업으로써 질문 답변(Question Answering, QA)을 활용하며, 이를 통해 긴 텍스트 요약 작업에 대한 LLM을 훈련합니다. 이 과정에서 QA는 풍부한 공개 데이터 세트를 제공하고, 질문-답변 쌍을 통해 더 많은 개체와 관계를 포함할 수 있어 요약의 질적 성장을 도모합니다. 또한, 질문 생성(Question Generation, QG)을 통해 두 작업 간의 문맥적 특성을 이해하게 됩니다.

- **Performance Highlights**: T3는 BBC summary, NarraSum, FairytaleQA, NLQuAD 데이터셋에서 3개의 기준 LLM에 비해 ROUGE에서 최대 14%, BLEU에서 35%, Factscore에서 16%의 향상치를 보여주며 그 효과성을 입증했습니다.



### ZALM3: Zero-Shot Enhancement of Vision-Language Alignment via In-Context Information in Multi-Turn Multimodal Medical Dialogu (https://arxiv.org/abs/2409.17610)
- **What's New**: 이 논문에서는 ZALM3를 제안하여 다중 회의(Multi-turn) 다중 모달(Multimodal) 의료 대화에서 비전-언어 정합성(Vision-Language Alignment)을 향상시키는 Zero-shot 접근 방식을 소개합니다. 환자가 제공하는 저품질 이미지와 텍스트 간의 관계를 개선하기 위해 LLM을 활용하여 이전 대화 맥락에서 키워드를 요약하고 시각적 영역을 추출합니다.

- **Technical Details**: ZALM3는 이미지 이전의 텍스트 대화에서 정보를 추출하여 관련 지역(Regions of Interest, RoIs)을 확인하는 데 LLM과 비주얼 그라우딩 모델을 활용합니다. 이 접근 방식은 추가적인 주석 작업이나 모델 구조 변경 없이도 비전-언어 정합성을 개선할 수 있는 특징이 있습니다.

- **Performance Highlights**: 세 가지 다른 임상 부서에서 실시한 실험 결과, ZALM3는 통계적으로 유의한 효과를 입증하며, 기존의 간단한 승패 평가 기준에 대한 보다 정량적인 결과를 제공하는 새로운 평가 메트릭을 개발하였습니다.



### Deep CLAS: Deep Contextual Listen, Attend and Sp (https://arxiv.org/abs/2409.17603)
Comments:
          Accepted by NCMMSC 2022

- **What's New**: 이번 연구에서는 Automatic Speech Recognition (ASR)에서 드문 단어의 인식률을 개선할 수 있는 Contextual-LAS (CLAS)의 발전형인 deep CLAS를 제안합니다. 기존 CLAS의 한계를 보완하기 위한 다양한 개선사항을 도입했습니다.

- **Technical Details**: deep CLAS는 문맥 정보를 보다 효율적으로 사용하기 위해 바이어스 손실(bias loss)을 도입합니다. 이 모델은 바이어스 어텐션(bias attention) 점수를 개선하기 위해 바이어스 어텐션의 쿼리를 보강하고, 문맥 정보는 LSTM 대신 Conformer로 인코딩합니다. 또한, 문구 수준 인코딩에서 문자 수준 인코딩으로 변경하여 세밀한 문맥 정보를 얻습니다.

- **Performance Highlights**: 공식 AISHELL-1 및 AISHELL-NER 데이터셋에서 실험한 결과, deep CLAS는 CLAS 기준 모델에 비해 명명 엔티티 인식(named entity recognition) 장면에서 상대적인 Recall 65.78%와 F1-score 53.49% 향상된 성능을 보였습니다.



### DualCoTs: Dual Chain-of-Thoughts Prompting for Sentiment Lexicon Expansion of Idioms (https://arxiv.org/abs/2409.17588)
- **What's New**: 이 논문에서는 관용구(idiom) 감정 분석을 위한 감정 어휘 확장을 자동으로 수행하는 혁신적인 접근 방식이 제안됩니다. 주로 대형 언어 모델(large language models)의 Chain-of-Thought prompting 기법을 활용하여 기존 자원을 통합하고 EmoIdiomE라는 새로운 감정 관용구 어휘 확장 데이터셋을 구축했습니다.

- **Technical Details**: 이 연구에서는 Dual Chain-of-Thoughts (DualCoTs) 방법을 설계하여 언어학적 및 심리언어학적 통찰을 결합함으로써 관용구의 감정 어휘를 자동으로 확장하는 방법을 제시합니다. DualCoTs는 문자적 체인(literal chain)과 어원적 체인(etymological chain)이라는 두 가지 체인으로 구성되어 있습니다. 이 두 체인을 통해 관용구의 감정 예측을 수행하며, 효과적인 감정 어휘 확장을 가능하게 합니다.

- **Performance Highlights**: DualCoTs 방법의 실험 결과, 중국어와 영어 모두에서 관용구 감정 어휘 확장에 효과적임을 입증했습니다. 연구팀은 EmoIdiomE 데이터셋을 통해 기존의 감정 어휘 확장의 한계를 극복하고, 감정 분석 및 감정 어휘 확장 작업에서 높은 정확도를 달성함을 보여주었습니다.



### Leveraging Annotator Disagreement for Text Classification (https://arxiv.org/abs/2409.17577)
- **What's New**: 이 논문은 다수의 주석자에 의해 주석된 데이터셋에서 단일 주석만을 활용하는 기존의 관행을 뛰어넘어, 주석자 간의 의견 불일치를 모델 훈련에 활용할 수 있는 세 가지 전략을 제안하고 비교합니다. 이를 통해 모델의 성능을 향상시킬 수 있는 가능성을 탐구합니다.

- **Technical Details**: 제안된 방법은 확률 기반 다중 레이블 (multi-label) 접근법, 앙상블 시스템 (ensemble system), 그리고 지시 조율 (instruction tuning) 방법입니다. 두 가지 텍스트 분류 작업(증오 발언 감지 및 대화 내 폭력 감지)을 사용하여 이 세 가지 방법의 효과를 평가하고, 다중 레이블 모델과 단일 레이블 모델 간의 성능을 온라인 설문조사를 통해 비교했습니다.

- **Performance Highlights**: 증오 발언 감지에서 다중 레이블 방법이 가장 뛰어난 성능을 보였고, 대화 내 폭력 감지에서는 지시 조율 방법이 최고 성과를 기록했습니다. 다중 레이블 모델의 출력은 단일 레이블 모델보다 텍스트를 더 잘 대표하는 것으로 평가되었습니다.



### Modulated Intervention Preference Optimization (MIPO): Keey the Easy, Refine the Difficu (https://arxiv.org/abs/2409.17545)
Comments:
          8pages, submitted to AAAI 2025

- **What's New**: 이번 연구에서는 Modulated Intervention Preference Optimization (MIPO)이라는 새로운 선호 최적화 알고리즘을 제안합니다. MIPO는 주어진 데이터와의 정렬 정도에 따라 참조 모델의 개입 수준을 조절하여 이전 방법들의 한계를 극복합니다.

- **Technical Details**: MIPO는 주어진 데이터가 참조 모델과 얼마나 잘 정렬되어 있는지를 평가하기 위해 평균 로그 가능성(average log likelihood)을 사용합니다. 이 값을 기반으로 MIPO는 정책 모델의 훈련 목표를 조정하여 잘 정렬되지 않은 데이터 쌍에 대해 더 많은 훈련을 수행할 수 있도록 합니다.

- **Performance Highlights**: MIPO는 Alpaca Eval 2.0 및 MT-Bench를 활용한 실험에서 DPO보다 일관되게 우수한 성능을 보였습니다. Llama3-8B-Instruct 모델의 경우 DPO보다 약 9점 (+36.07%), Mistral-7B-Base 모델에서는 약 8점 (+54.24%) 향상된 결과를 나타냈습니다. MIPO는 다양한 실험 환경에서 가장 뛰어난 성능을 달성했습니다.



### Logic-of-Thought: Injecting Logic into Contexts for Full Reasoning in Large Language Models (https://arxiv.org/abs/2409.17539)
Comments:
          20 pages

- **What's New**: 이번 논문은 Logic-of-Thought (LoT) 프로밍 기법을 제안하여 기존의 신경상징적 방법에서 발생하는 정보 손실 문제를 해결하고, LLM의 논리적 추론 능력을 향상시키고자 합니다.

- **Technical Details**: LoT는 입력 문맥에서 명제(propositions)와 논리적 표현(logical expressions)을 추출하여, 이들을 논리적 추론 법칙(logical reasoning laws)에 따라 확장한 뒤, 확장된 논리적 표현을 자연어로 변환하여 LLM의 입력 프롬프트에 추가적인 보강제로 활용합니다. LoT는 기존의 다양한 프롬프트 기법과 호환되도록 설계되었습니다.

- **Performance Highlights**: LoT는 CoT의 ReClor 데이터셋에서 성능을 +4.35% 향상시켰고, CoT-SC는 LogiQA에서 +5% 증가, ToT는 ProofWriter 데이터셋에서 +8% 향상된 성과를 보여주었습니다.



### MUSE: Integrating Multi-Knowledge for Knowledge Graph Completion (https://arxiv.org/abs/2409.17536)
Comments:
          arXiv admin note: text overlap with arXiv:2408.05283

- **What's New**: 이 논문에서는 Knowledge Graph Completion (KGC) 문제를 해결하기 위해 MUSE라는 지식 인식 추론 모델을 제안합니다. MUSE는 결측 관계를 예측하기 위해 다중 지식 표현 학습 메커니즘을 설계하였으며, BERT 조정, Context Message Passing, Relational Path Aggregation을 포함한 세 가지 모듈을 활용합니다.

- **Technical Details**: MUSE는 세 가지 병렬 구성 요소로 이루어진 맞춤형 임베딩 공간을 개발합니다: 1) Prior Knowledge Learning을 통해 BERT를 미세 조정하여 triplet의 의미 표현을 강화합니다; 2) Context Message Passing을 통해 KG의 문맥적 메시지를 강화하고; 3) Relational Path Aggregation을 통해 head entity에서 tail entity까지의 경로 표현을 강화합니다.

- **Performance Highlights**: MUSE는 NELL995 데이터셋에서 H@1 지표가 5.50% 향상되고, MRR 지표가 4.20% 향상되는 성과를 보여주었습니다. 또한, WN18과 WN18RR 데이터셋에서 H@3 수치가 1.00을 달성했습니다.



### Data Proportion Detection for Optimized Data Management for Large Language Models (https://arxiv.org/abs/2409.17527)
- **What's New**: 이 논문은 데이터 비율 탐지(data proportion detection)라는 새로운 주제를 소개하며, LLM의 생성 출력 분석을 통해 사전 학습 데이터 비율을 자동으로 추정할 수 있는 방법을 제안합니다.

- **Technical Details**: 이 논문은 데이터 비율 탐지 문제에 대한 이론적 증명과 실제 알고리즘, 초기 실험 결과를 제시합니다. 데이터 비율 탐지는 특정한 원래 데이터에 대한 사전 지식 없이 모델이 사용하는 사전 학습 데이터의 비율을 파악하는 데 중점을 둡니다.

- **Performance Highlights**: 미리 준비된 데이터 비율 탐지 알고리즘을 통해 초기 실험을 진행했으며, 이는 데이터 비율 탐지의 기준선을 수립하고 향후 연구의 기초로 삼습니다.



### Reducing and Exploiting Data Augmentation Noise through Meta Reweighting Contrastive Learning for Text Classification (https://arxiv.org/abs/2409.17474)
Comments:
          IEEE BigData 2021

- **What's New**: 이 논문은 Meta Reweighting Contrastive (MRCo) 모델을 제안하여 데이터 증강(sample)의 품질을 고려하여 기존의 딥러닝 모델의 성능을 향상시키고자 합니다. 이 프레임워크는 메타 학습(meta learning)과 대조 학습(contrastive learning)을 결합하여 증강된 샘플의 가중치 정보를 활용합니다.

- **Technical Details**: MRCo 모델은 두 가지 최적화 루프를 가지며, 내부 루프는 다운스트림 작업을 위한 주 모듈이 재가중된 손실(loss)로 학습합니다. 외부 루프는 메타 재가중화 모듈이 증강 샘플에 적절한 가중치를 할당합니다. 대조 학습을 이용해 저품질 샘플과 고품질 샘플 간의 차이를 증대시키고 있다.

- **Performance Highlights**: 실험 결과, 본 프레임워크는 Text-CNN 인코더에서 평균 1.6%, 최대 4.3%의 개선율을 보였고, RoBERTa-base 인코더에서는 평균 1.4%, 최대 4.4%의 개선을 기록했습니다. 이는 7개의 GLUE 벤치마크 데이터셋을 기준으로 하였습니다.



### Autoregressive Multi-trait Essay Scoring via Reinforcement Learning with Scoring-aware Multiple Rewards (https://arxiv.org/abs/2409.17472)
Comments:
          EMNLP 2024

- **What's New**: SaMRL(Scoring-aware Multi-reward Reinforcement Learning) 방법론을 제안하여 다중 특성 자동 에세이 평가(multi-trait automated essay scoring)에서 Quadratic Weighted Kappa(QWK) 기반 보상을 통합하고, 평균 제곱 오차(MSE) 페널티를 적용하여 실제 평가 방식을 훈련 과정에 포함시킴으로써 모델의 성능을 향상시킵니다.

- **Technical Details**: 기존의 QWK는 비미분 가능성으로 인해 신경망 학습에 직접 사용되지 못하나, SaMRL은 bi-directional QWK와 MSE 페널티를 통해 다중 특성 평가의 복잡한 측정 방식을 훈련에 유효하게 통합하고, 오토 회귀 점수 생성 프레임워크를 적용해 토큰 생성 확률로 강력한 다중 특성 점수를 예측합니다.

- **Performance Highlights**: ASAP 및 ASAP++ 데이터셋을 통한 광범위한 실험 결과, SaMRL은 기존 강력한 기준선에 비해 각 특성과 프롬프트에서 점수 향상을 이끌어내며, 특히 넓은 점수 범위를 가진 프롬프트에서 기존 RL 사용의 한계를 극복한 점이 두드러집니다.



### What is the social benefit of hate speech detection research? A Systematic Review (https://arxiv.org/abs/2409.17467)
Comments:
          Accepted to the 3rd Workshop on NLP for Positive Impact

- **What's New**: 본 연구는 혐오 발언 탐지(NLP) 시스템의 사회적 영향력을 높이기 위한 윤리적 프레임워크의 필요성을 강조합니다. 연구진은 48개의 혐오 발언 탐지 시스템과 관련된 37개의 논문을 검토하여 현재의 연구 실천과 최선의 실천 간의 간극을 보여줍니다.

- **Technical Details**: 연구는 혐오 발언 탐지 시스템이 데이터 세트 수집과 준비, 특징 엔지니어링, 모델 훈련 및 모델 평가 등의 유사한 워크플로우를 따르며, 데이터의 품질 및 다양성이 시스템 성능에 미치는 영향을 논의합니다. 또한, AI 연구의 책임감 있는 혁신을 위한 새로운 프레임워크를 제안합니다.

- **Performance Highlights**: 연구 결과, 기존의 혐오 발언 탐지 시스템은 사회적 편향성을 내포하고 있으며, 이러한 편향성이 취약한 커뮤니티에 추가적인 불이익을 초래할 수 있음을 나타내었습니다. 따라서, NLP 연구자들은 비단 모델 성능 외에도 시스템의 사회적 영향을 고려해야 할 필요가 강조됩니다.



### Navigating the Shortcut Maze: A Comprehensive Analysis of Shortcut Learning in Text Classification by Language Models (https://arxiv.org/abs/2409.17455)
- **What's New**: 본 연구에서는 언어 모델(당초 LMs)에서 무시되었던 복잡한 대체 요인들이 모델의 신뢰성에 미치는 영향을 분석합니다. 저자들은 다각적으로 단축키(shortcut)를 정의하고 이를 발생, 스타일, 개념으로 분류하여 연구합니다.

- **Technical Details**: 연구에서는 BERT, Llama, SOTA 강인한 모델 등의 다양한 언어 모델에 대해 대체 요인에 대한 저항력과 취약성을 체계적으로 분석합니다. 그 결과, BERT는 모든 유형의 대체 요인에 취약하다는 것을 발견했습니다.

- **Performance Highlights**: 대형 모델의 크기 증가가 항상 Robustness를 보장하지 않으며, 강인한 모델들이 때때로 LLM보다 더 우수한 강인성을 나타냅니다. 모든 모델이 모든 유형의 대체 요인에 대해 강인하지 않다는 것이 드러났습니다.



### Enhancing Financial Sentiment Analysis with Expert-Designed Hin (https://arxiv.org/abs/2409.17448)
- **What's New**: 이 논문은 재무 소셜 미디어 게시물에서 감정 분석을 향상시키는 데 있어 전문가가 설계한 힌트의 역할을 조사합니다.

- **Technical Details**: 대규모 언어 모델(LLMs)의 관점 수용(perspective-taking) 능력을 분석하고, 숫자의 중요성을 강조하는 전문가 설계 힌트를 적용하여 성능 향상을 이루었습니다. 사용된 데이터셋은 Fin-SoMe이며, 10,000개의 트윗을 포함하고 있습니다. 실험에 포함된 LLMs는 PaLM 2, Gemini Pro, GPT-3.5, GPT-4입니다.

- **Performance Highlights**: 전문가 설계 힌트를 포함했을 때 모든 LLM에서 감정 분석 성능이 일관되게 향상되었으며, 특히 숫자와 관련된 게시물에서 두드러진 성과를 보였습니다. 기본적인 프롬프트( Simple Prompt)와 비교했을 때, 전문가의 힌트가 큰 효과를 발휘함을 보여주었습니다.



### HDFlow: Enhancing LLM Complex Problem-Solving with Hybrid Thinking and Dynamic Workflows (https://arxiv.org/abs/2409.17433)
Comments:
          27 pages, 5 figures

- **What's New**: 이 논문에서는 Large Language Models (LLMs)의 복잡한 추론 문제 해결을 위한 새로운 프레임워크인 HDFlow를 소개합니다. 이 프레임워크는 빠른 사고와 느린 사고를 적응적으로 결합하여, 복잡한 문제를 더 잘 해결할 수 있도록 합니다.

- **Technical Details**: HDFlow는 두 가지 주요 구성 요소로 이루어져 있습니다. 첫째, 'Dynamic Workflow'라는 새로운 느린 추론 접근법은 복잡한 문제를 관리 가능한 하위 작업으로 자동 분해합니다. 둘째, 'Hybrid Thinking'은 문제의 복잡성에 따라 빠른 사고와 느린 사고를 동적으로 결합하는 일반적인 프레임워크입니다. 이를 통해 모델은 문제의 복잡성에 따라 적절한 사고 모드를 선택합니다.

- **Performance Highlights**: 실험 결과, HDFlow의 느린 사고 방식 및 동적 워크플로우가 Chain-of-Thought (CoT) 전략보다 평균 22.4% 더 높은 정확도를 보였고, Hybrid Thinking은 네 개의 벤치마크 데이터 세트 중 세 개에서 최고의 정확도를 기록했습니다. 또한 하이브리드 사고를 통한 미세 조정 방법이 오픈 소스 언어 모델의 복잡한 추론 능력을 크게 향상시켰습니다.



### On Extending Direct Preference Optimization to Accommodate Ties (https://arxiv.org/abs/2409.17431)
Comments:
          24 pages

- **What's New**: 이 논문에서는 DPO(Direct Preference Optimization)의 두 가지 변형을 도입하여 쌍별 비교에서 동점(tie) 가능성을 명시적으로 모델링하는 방법을 제시합니다. 즉, 통상적인 Bradley-Terry 모델을 Rao-Kupper와 Davidson의 확장 모델로 대체하여 동점 확률을 명시적으로 포함시킵니다.

- **Technical Details**: DPO는 쌍으로 구성된 옵션 데이터(yw ≻ yl)를 통해 명확한 선호를 요구합니다. 본 연구는 Rao-Kupper 및 Davidson의 모델을 적용하여 동점 사례를 데이터셋에 추가하였고, 이는 작업 성능 저하 없이 실현되었습니다. KL-divergence 측정치를 통해 참조 정책에 대한 정규화가 강화됨을 확인했습니다.

- **Performance Highlights**: 실험 결과, DPO 변형에 동점 데이터를 통합함으로써 성능 저하 없이 훨씬 더 강한 정규화를 달성하였고, 원래 형태의 DPO에서도 유사한 개선 효과를 발견하였습니다. 이러한 발견은 단순히 동점을 제외하는 대신 선호 최적화에서 동점 쌍을 포함할 수 있는 방법을 공고히 합니다.



### Discovering the Gems in Early Layers: Accelerating Long-Context LLMs with 1000x Input Token Reduction (https://arxiv.org/abs/2409.17422)
- **What's New**: 이 연구에서는 긴 컨텍스트 입력을 처리하기 위해 LLM(Large Language Model)의 추론을 가속화하고 GPU 메모리 소모를 줄이는 새로운 접근 방식인 GemFilter를 소개합니다.

- **Technical Details**: GemFilter는 LLM의 초기 레이어를 필터로 사용하여 입력 토큰을 선택하고 압축하는 알고리즘으로, 이에 따라 추후 처리할 컨텍스트 길이를 대폭 줄입니다. 이 방법은 2.4$	imes$ 속도 향상과 30
gpu(%) GPU 메모리 사용량 감소를 달성하였습니다.

- **Performance Highlights**: GemFilter는 Needle in a Haystack 벤치마크에서 기존의 표준 어텐션 및 SnapKV를 능가하며, LongBench 챌린지에서는 SnapKV/H2O와 유사한 성능을 보여줍니다.



### Pre-Finetuning with Impact Duration Awareness for Stock Movement Prediction (https://arxiv.org/abs/2409.17419)
Comments:
          NTCIR-18 FinArg-2 Dataset

- **What's New**: 이 연구는 투자자 의견에 기반한 뉴스 이벤트의 영향 지속 시간을 추정하기 위한 새로운 데이터 세트인 Impact Duration Estimation Dataset (IDED)을 소개하며, 사전 미세 조정(pre-finetuning) 작업을 통해 주식 이동 예측 성능을 향상시키는 방법을 제시합니다. 기존 연구에서 간과된 시간적 정보의 중요성을 강조합니다.

- **Technical Details**: IDED 데이터 세트는 투자자 의견을 포함하고 있으며, 8,760개의 게시물에 대한 영향 지속 시간이 1주일 이내, 1주일 이상, 불확실로 분류되었습니다. 또한, 여러 대형 사전 학습 언어 모델(BERT-Chinese, Multilingual-BERT 등)을 사용하여 IDED로 사전 미세 조정한 결과, 모든 모델에서 성능 향상이 관찰되었습니다. SRLP(Semantic Role Labeling Pooling) 기법과 함께 StockNet 및 HAN 모델과 비교 분석하였습니다.

- **Performance Highlights**: 모든 사전 학습 언어 모델은 SRLP의 성능을 초과하지 못했으며, IDED 적용이 주식 이동 예측에서 성능을 향상시킨다는 것이 입증되었습니다. IDED-Mengzi-Fin 모델이 최상의 성능을 기록하였으며, 이는 해당 모델이 재무 문서로 사전 학습되어 도메인 특화의 이점을 지니기 때문입니다.



### Enhancing Investment Opinion Ranking through Argument-Based Sentiment Analysis (https://arxiv.org/abs/2409.17417)
- **What's New**: 본 연구는 전문가 및 아마추어 투자자의 관점을 모두 고려하여 효과적인 추천 시스템을 위한 이중 접근 방식인 'argument mining' 기술을 도입했습니다. 이를 통해 유망한 투자 의견을 식별하고 분석하는 새로운 방법을 제시합니다.

- **Technical Details**: 연구는 가격 목표(price target)를 사용하여 의견의 강도를 평가하고, 투자 의견을 평가하기 위해 역사적 데이터를 사용하여 모델을 훈련합니다. 또한, 'argument mining' 기법을 사용하여 의견을 명제(premise)와 주장(claim)으로 분해하고 이들의 관계를 분석합니다. 이 과정에서 분석가의 가격 목표를 이용하여 주장의 강도를 수치화합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 보다 높은 수익 잠재력을 가진 의견을 식별하는 능력을 검증하였으며, 추천된 의견과 투자자 행동 간의 관계를 분석하여 리스크(risk) 분석에도 기여하였습니다. 또한, 전문가의 의견 변화와 거래 패턴에 대한 포괄적인 분석이 이루어졌습니다.



### From Deception to Detection: The Dual Roles of Large Language Models in Fake News (https://arxiv.org/abs/2409.17416)
- **What's New**: 최근 연구에서는 다양한 Large Language Models (LLMs)가 가짜 뉴스를 생성할 수 있는 능력과 이러한 모델들이 가짜 뉴스를 감지하는 성능을 비교했습니다. 이는 7개의 LLM을 분석한 최초의 연구로, 각 모델의 편향 및 안전성 준수 여부를 평가하였습니다.

- **Technical Details**: 연구는 LLM의 가짜 뉴스 생성과 감지 등 두 가지 주요 단계를 중점적으로 다루고 있습니다. LLM들은 다양한 편향을 담은 가짜 뉴스를 생성할 수 있으며, 종종 인간이 작성한 내용보다 탐지하기 어렵습니다. 또한, LLM에서 제공하는 설명의 유용성도 평가되었습니다.

- **Performance Highlights**: 결과적으로, 크기가 큰 LLM들이 더 나은 가짜 뉴스 탐지 능력을 보였으며, 일부 모델은 안전 프로토콜을 엄격히 준수하여 편향된 내용을 생성하지 않았습니다. 반면에, 다른 모델은 여러 편향을 포함한 가짜 뉴스를 쉽게 생성할 수 있었고, LLM이 생성한 가짜 뉴스는 일반적으로 인간이 작성한 것보다 탐지될 가능성이 낮다는 사실이 밝혀졌습니다.



### Severity Prediction in Mental Health: LLM-based Creation, Analysis, Evaluation of a Novel Multilingual Datas (https://arxiv.org/abs/2409.17397)
- **What's New**: 이 연구는 다양한 비영어권 언어에서 대규모 언어 모델(LLMs)의 효과성을 평가하기 위한 다국어 정신 건강 데이터셋을 개발하였습니다. 이 데이터셋은 영어에서 그리스어, 터키어, 프랑스어, 포르투갈어, 독일어, 핀란드어로 번역된 사용자 생성 콘텐츠로 구성되어 있습니다. 이는 모델의 성능을 여러 언어에서 종합적으로 평가할 수 있도록 합니다.

- **Technical Details**: 이 연구에서는 GPT와 Llama를 사용하여 여러 언어로 번역된 동일한 데이터셋에서 정신 건강 상태의 심각성을 정밀하게 예측하는 성능을 비교하였습니다. 연구 결과, 언어별로 상당한 성능 차이를 관찰했으며 이는 다국어 정신 건강 지원의 복잡성을 강조합니다. 대상 데이터셋은 다음 링크에서 공개되어 있습니다: https://github.com/y3nk0/multilingual-mental-severity-prediction.

- **Performance Highlights**: 이 연구는 다국어 정신 건강 조건의 심각성을 탐지하는 LLM의 능력에 대한 포괄적인 분석을 제공하며, 특정 언어에서의 성과 차이가 어떻게 발생하는지를 조명합니다. 또한, LLM을 의료 환경에서 사용할 때의 잠재적인 오진 위험에 대해 경고합니다. 이 연구의 접근 방식은 다국어 작업의 비용 절감 측면에서도 중요한 이점을 제공합니다.



### Scaling Behavior for Large Language Models regarding Numeral Systems: An Example using Pythia (https://arxiv.org/abs/2409.17391)
Comments:
          EMNLP 2024 Findings

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 수치 연산 성능에 영향을 미치는 숫자 체계의 차이를 조사했습니다. 특히, 10진수(base 10) 시스템과 100진수(base 100), 1000진수(base 1000) 시스템의 학습 효율성을 비교하여 10진수 시스템이 보편적으로 더 데이터 효율적임을 보여주었습니다.

- **Technical Details**: 본 연구에서는 LLMs의 수치 연산을 이해하기 위해 서로 다른 숫자 체계(10진수, 100진수, 1000진수)에 대해 실험을 설계했습니다. 중요한 실험 차원으로는 숫자 체계, 데이터 규모, 모델 크기 및 훈련 방식( from-scratch 또는 fine-tuning)이 있습니다. 실험 결과, 10진수 시스템은 훈련에서 일관되게 더 높은 데이터 효율성을 보였으며, 고빈도 토큰이 높은 학습 결과를 낳는 것으로 분석되었습니다.

- **Performance Highlights**: 10진수 시스템은 100진수나 1000진수 시스템에 비해 데이터 규모와 모델 규모에 상관없이 일관되게 더 우수한 성능을 보였습니다. 특히, 10진수 시스템은 숫자 인식 및 연산에서 더 효과적인 패턴을 보여주었으며, 100진수 및 1000진수 시스템은 토큰 수준 인식과 연산에서 어려움을 겪는 것으로 나타났습니다.



### data2lang2vec: Data Driven Typological Features Completion (https://arxiv.org/abs/2409.17373)
Comments:
          9 pages, 11 figures

- **What's New**: 본 연구는 다국어 자연어 처리(NLP)의 모델 적응성을 향상시키기 위해 언어 유형론 데이터베이스를 활용한 새로운 접근법을 제안합니다. 특히, lang2vec 툴킷의 한계를 극복하기 위해 텍스트 데이터를 사용하여 누락된 특성 예측을 수행합니다.

- **Technical Details**: 우리는 1749개 언어에 대해 70% 이상의 정확도를 달성한 다국어 품사(POS) 태거를 소개하며, 외부 통계적 특성과 다양한 머신러닝 알고리즘을 실험합니다. 각 언어의 언어족, 위키피디아 크기, 인간 수 등을 포함한 다수의 특징들을 사용하여 예측을 수행합니다.

- **Performance Highlights**: 우리는 제안된 방법이 기존의 방법보다 두 가지 설정에서 모두 성능이 우수함을 보여주며, 텍스트 기반 접근 방식을 통해 lang2vec 데이터의 전체 특성 목록을 완성하는 데 기여했습니다.



### Internalizing ASR with Implicit Chain of Thought for Efficient Speech-to-Speech Conversational LLM (https://arxiv.org/abs/2409.17353)
- **What's New**: 본 논문에서는 ASR(Automatic Speech Recognition) 및 TTS(Text-to-Speech) 기능을 내재화하여 음성 기반 LLM(Large Language Models)의 대화 능력을 강화하는 새로운 방법을 제안합니다. 기존의 ASR-TTS 파이프라인에서 발생하던 지연을 줄이고, 모델의 음성 이해능력을 향상시키는 것이 주요 목표입니다.

- **Technical Details**: 저자들은 ICoT(Implicit Chain of Thought) 기술을 활용하여 훈련된 Speech LLM에서 ASR 기능을 내재화하는 방법론을 확립했습니다. 이 과정에서 말의 전사 및 응답 생성을 위한 명시적 단계 없이도 음성 대화를 가능하게 합니다. 이러한 접근법은 대화 경험을 Improve하며, 이전의 CoT(Chain of Thought) 방법론을 오디오 및 음성 도메인에 적용한 사례로서 의미가 있습니다.

- **Performance Highlights**: 연구 결과, 기존 ASR-텍스트 기반 방법 대비 14.5%의 지연 감소를 달성했으며, 새로운 대화 모델인 AnyGPT를 통해 약 660,000개의 대화 쌍이 포함된 대규모 합성 대화 데이터셋을 구축하여 연구에 기여했습니다. 모델의 효율성을 개선하고 보다 자연스러운 음성 상호작용이 가능하도록 했습니다.



### How Transliterations Improve Crosslingual Alignmen (https://arxiv.org/abs/2409.17326)
Comments:
          preprint

- **What's New**: 이 논문은 다국어 사전 훈련된 언어 모델(mPLMs)의 크로스링구얼 정렬(alignment)을 개선하기 위한 능동적인 방법으로, 단순한 변환(transliteration) 데이터 추가가 어떻게 성능 향상에 기여하는지 명확히 평가하려는 첫 시도를 다룹니다.

- **Technical Details**: 크로스링구얼 정렬의 정의를 정립하고, 문장 표현(sentence representations)을 기준으로 네 가지 유사성(similarity) 유형을 정의합니다. 실험은 폴란드어-우크라이나어, 힌디어-우르두어 쌍에 대해 진행되며, 데이터 정렬을 평가하기 위해 다양한 설정 하에 여러 모델을 훈련합니다. 실험 결과, 단독으로 추가된 변환 데이터가 유사성을 높이며, 보조 정렬 목표(auxiliary alignment objectives)는 매칭된 쌍을 구별하는 데 도움이 되는 것으로 나타났습니다.

- **Performance Highlights**: 전반적으로 변환 데이터 사용이 모든 유사성 유형을 증가시키지만, 정렬이 항상 다운스트림 성능을 향상시키는 것은 아니라는 점이 드러났습니다. 이러한 결과는 크로스링구얼 정렬과 성능 간의 더 깊은 연구가 필요함을 시사합니다.



### BabyLlama-2: Ensemble-Distilled Models Consistently Outperform Teachers With Limited Data (https://arxiv.org/abs/2409.17312)
Comments:
          9 pages, 3 figures, 5 tables, submitted to the BabyLM Challenge (CoNLL 2024 Shared Task)

- **What's New**: BabyLlama-2는 3억 4500만 개의 매개변수를 가진 모델로, 두 개의 교사 모델로부터 1000만 개의 단어로 학습한 후 BabyLM 대회에 제출된 것이다. 결과적으로 BabyLlama-2는 BLiMP 및 SuperGLUE 벤치마크에서 동일한 데이터 혼합을 사용하는 1000만 및 1억 단어 데이터셋으로 학습된 기준 모델을 능가하였다.

- **Technical Details**: BabyLlama-2는 345M의 decoder-only 모델로, 9.5M 단어로 사전 학습되었다. 하이퍼파라미터 최적화가 광범위하게 수행되었으며, 최적의 하이퍼파라미터 선택이 교사 모델의 우수한 성능에 기인하지 않음을 증명하였다. 지식 증류(knowledge distillation) 기법을 사용하여 샘플 효율성을 높이는 데 중점을 두었다.

- **Performance Highlights**: BabyLlama-2는 기존 모델보다 더 적은 데이터로도 더 높은 성능을 달성하였다. 특히, BLiMP 과제에서 제로 샷(zero-shot) 성능과 모델의 테스트 손실(test loss) 사이에 상관관계를 찾았으며, 이는 모델 성능을 향상시키기 위한 지식 증류의 가능성을 강조한다.



### Plurals: A System for Guiding LLMs Via Simulated Social Ensembles (https://arxiv.org/abs/2409.17213)
- **What's New**: 최근 논의에서 언어 모델들이 특정 관점을 선호한다는 우려가 제기되었습니다. 이에 대한 해결책으로 '어디서도 바라보지 않는 시각'을 추구하는 것이 아니라 다양한 관점을 활용하는 방안을 제안합니다. 'Plurals'라는 시스템과 Python 라이브러리를 소개하게 되었습니다.

- **Technical Details**: Plurals는 다양한 관점을 반영하여 신뢰할 수 있는 소셜 에셈블을 생성하는 시스템으로, 사용자 맞춤형 구조 내에서 '대리인'(Agents)들이 심의(deliberation)를 진행하고, 보고자'(Moderators)가 이를 감독합니다. 이 시스템은 미국 정부의 데이터셋과 통합되어 국가적으로 대표적인 페르소나를 생성하면서 민주적 심의 이론에 영감을 받은 형식으로 사용자 정의가 가능합니다.

- **Performance Highlights**: 여섯 가지 사례 연구를 통해 이론적 일관성과 효용성(efficacy)을 보여주었고, 세 가지 무작위 실험에서는 생성된 산출물이 관련 청중의 온라인 샘플과 잘 맞아떨어짐을 발견했습니다. Plurals는 사용자가 시뮬레이션된 사회적 집단을 생성할 수 있도록 도와주며, 초기 연구 결과 또한 제시됩니다.



### An Effective, Robust and Fairness-aware Hate Speech Detection Framework (https://arxiv.org/abs/2409.17191)
Comments:
          IEEE BigData 2021

- **What's New**: 이 논문은 온라인 소셜 네트워크에서의 증오 발언을 효과적으로 탐지하기 위한 새로운 프레임워크를 제안합니다. 데이터 부족, 모델의 불확실성 추정, 악의적 공격에 대한 강인성 향상 및 공정성 처리의 한계를 극복하고자 합니다.

- **Technical Details**: 제안된 프레임워크는 Bidirectional Quaternion-Quasi-LSTM (BiQQLSTM) 계층을 포함하여 효과성과 효율성을 균형 있게 맞추고, 세 가지 플랫폼에서 수집한 다섯 개 데이터셋을 결합하여 일반화된 모델을 구축합니다. 데이터 증강 기법을 사용해 다양한 공격 및 텍스트 조작에 강한 모델을 만드는 데 주력합니다.

- **Performance Highlights**: 실험 결과, 제안한 모델은 공격이 없는 시나리오에서 8개의 최첨단 방법들보다 5.5% 향상된 성능을 보여주며, 다양한 공격 시나리오에서도 최고 기준 대비 3.1% 향상된 성능을 기록하여 모델의 효과성과 강인성을 입증했습니다.



### Fully automatic extraction of morphological traits from the Web: utopia or reality? (https://arxiv.org/abs/2409.17179)
- **What's New**: 이 논문은 최근의 대형 언어 모델(LLMs)을 활용하여 비구조적인 텍스트에서 식물의 형태학적 특성 정보를 자동으로 수집하고 처리하는 메커니즘을 제안합니다. 이를 통해 전문가가 수년간 수집해야 하는 복잡한 특성 정보를 손쉽게 구축할 수 있습니다.

- **Technical Details**: 제안된 방법론은 다음과 같은 세 가지 입력을 요구합니다: (i) 관심 있는 종의 목록, (ii) 관심 있는 특성 목록, 그리고 (iii) 각 특성이 가질 수 있는 모든 가능한 값의 목록. 이 프로세스는 검색 엔진 API를 사용하여 관련 URL를 가져오고, 이 URL에서 텍스트 콘텐츠를 다운로드합니다. 이후 NLP 모델을 통해 설명 문장을 판별하고, LLM을 사용하여 기술적 특성 값을 추출합니다.

- **Performance Highlights**: 이 방식으로 3개의 수작업으로 작성된 종-특성 행렬을 자동으로 복제하는 평가를 실시했습니다. 결과적으로 75% 이상의 F1-score를 달성하며, 50% 이상의 종-특성 쌍의 값을 찾는 데 성공했습니다. 이는 비구조적 온라인 텍스트로부터 구조화된 특성 데이터베이스를 대규모로 생성하는 것이 현재 가능하다는 점을 보여줍니다.



### CSCE: Boosting LLM Reasoning by Simultaneous Enhancing of Casual Significance and Consistency (https://arxiv.org/abs/2409.17174)
- **What's New**: 현재의 언어 모델(LLMs)에서 Chain-Based(체인 기반) 접근 방식에 의존하지 않고, 인과적 중요성(causal significance)과 일관성(consistency)을 동시에 고려할 수 있는 비체인 기반의 새로운 추론 프레임워크인 CSCE(Causal Significance and Consistency Enhancer)를 제안합니다.

- **Technical Details**: CSCE는 Treatment Effect(치료 효과) 평가를 활용하여 LLM의 손실 함수(loss function)를 맞춤화하고, 인과적 중요성과 일관성을 두 가지 측면에서 향상시킴으로써 인과관계를 정확히 파악하고 다양한 상황에서 견고하고 일관된 성능을 유지하도록 합니다. 이 프레임워크는 최대한의 추론 효율성을 위해 전체 추론 과정을 한 번에 출력합니다.

- **Performance Highlights**: CSCE 방법은 Blocksworld, GSM8K, Hanoi Tower 데이터셋에서 Chain-Based 방법들을 초월하여 높은 성공률과 빠른 처리 속도를 기록하며, 비체인 기반 방법이 LLM의 추론 작업을 완료하는 데에도 기여할 수 있음을 입증했습니다.



### A Multiple-Fill-in-the-Blank Exam Approach for Enhancing Zero-Resource Hallucination Detection in Large Language Models (https://arxiv.org/abs/2409.17173)
Comments:
          20 pages

- **What's New**: 본 논문에서는 이야기가 변경되는 문제를 해결하기 위해 다중 객관식 채우기 시험(multiple-fill-in-the-blank exam) 접근 방식을 포함한 새로운 환각(hallucination) 감지 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 원본 텍스트에서 여러 객체를 마스킹한 후 각 시험 응답이 원본 스토리라인과 일치하도록 LLM을 반복적으로 도와줍니다. 이 과정에서 발생하는 환각 정도를 채점하여 평가합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존 방법(SCGP)보다 우수한 성능을 보이며, SCGP와의 앙상블에서도 현저한 성능 향상을 나타냅니다.



### What Would You Ask When You First Saw $a^2+b^2=c^2$? Evaluating LLM on Curiosity-Driven Questioning (https://arxiv.org/abs/2409.17172)
- **What's New**: 본 논문에서는 대규모 언어 모델(LLM)의 새로운 지식 습득 가능성을 평가하기 위한 혁신적인 평가 프레임워크를 제안합니다. 이 프레임워크는 LLM이 과학적 지식을 소개하는 진술에 대한 질문을 생성하도록 유도하여, 처음 접하는 사람처럼 호기심을 가지고 질문하는 방식으로 평가합니다.

- **Technical Details**: 제안된 평가 방법은 호기심 기반 질문 생성(CDQG) 과제로, LLM이 처음 접하는 진술을 상상하며 즉각적으로 떠오르는 질문을 만들어내도록 프롬프트합니다. 생성된 질문은 관련성(relevance), 일관성(coherence), 다양성(diversity) 세 가지 주요 지표로 평가되며, 심리학 문헌에 뿌리를 두고 있습니다. 여러 모델의 성능을 비교 평가하기 위해 물리학, 화학 및 수학 분야의 1101개의 다양한 난이도의 진술로 구성된 합성 데이터셋을 수집했습니다.

- **Performance Highlights**: GPT-4와 Mistral 8x7b와 같은 대형 모델이 일관성 있고 관련성이 높은 질문을 잘 생성하는 반면, 크기가 작은 Phi-2 모델은 동등하거나 그 이상으로 효과적임을 발견했습니다. 이는 모델의 크기만으로 지식 습득 가능성을 판단할 수 없음을 나타냅니다. 연구 결과, 제안된 프레임워크는 LLM의 질문 생성 능력을 새로운 관점에서 평가할 수 있는 기회를 제공합니다.



### Cross-Domain Content Generation with Domain-Specific Small Language Models (https://arxiv.org/abs/2409.17171)
Comments:
          15 pages

- **What's New**: 이 연구는 여러 개별 데이터셋에서의 작은 언어 모델을 활용한 도메인 특정 콘텐츠 생성에 대한 새로운 접근 방식을 탐색합니다. 특히, 두 개의 서로 다른 도메인인 이야기 (story)와 레시피 (recipe)에 대한 모델을 비교하였습니다.

- **Technical Details**: 모델을 각각의 데이터셋에 대해 개별적으로 훈련시키는 방식이 사용되었으며, 사용자 정의 토크나이저 (custom tokenizer)를 적용하여 생성 품질을 크게 향상시켰습니다. 또한 Low-Rank Adaptation (LoRA)나 일반적인 파인튜닝 (fine-tuning) 방법으로 단일 모델을 두 도메인에 적용하려고 시도했지만 유의미한 결과를 얻지 못했습니다. 특히, 전체 파인튜닝(full fine-tuning)중 모델의 가중치 동결 없이 진행 시, 치명적 망각 (catastrophic forgetting)이 발생하였습니다.

- **Performance Highlights**: 지식 확장 전략 (knowledge expansion strategy)을 통해 모델이 이야기와 레시피를 요청에 따라 생성할 수 있도록 하였으며, 이는 서로 다른 데이터셋 간의 의미 있는 출력을 유지하도록 하였습니다. 연구 결과는 고정된 레이어를 가진 지식 확장이 작은 언어 모델이 다양한 도메인에서 콘텐츠를 생성하는 데 효과적인 방법임을 보여줍니다.



### REAL: Response Embedding-based Alignment for LLMs (https://arxiv.org/abs/2409.17169)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)과 인간의 선호를 일치시키기 위한 효율적인 데이터 선택 전략을 제안합니다. 기존 알고리즘의 한계를 극복하기 위해 의미 있는 응답 쌍을 선택하는 방법론을 개발하였습니다.

- **Technical Details**: 제안된 방법은 HH-RLHF 데이터셋에서 유사하지 않은 응답 쌍을 선택하여 LLM의 직접적인 정렬을 개선하며 라벨링 오류를 감소시키는 것을 목표로 합니다. 코사인 유사성을 기반으로 응답 쌍의 유용성을 평가하여 높은 품질의 학습 데이터를 형성합니다.

- **Performance Highlights**: 실험 결과, 유사하지 않은 응답 쌍을 사용한 모델이 대화 작업에서 최상의 승률을 기록하였으며, 라벨러의 작업을 최대 65%까지 절감하는 효율성을 보여주었습니다.



### BERTScoreVisualizer: A Web Tool for Understanding Simplified Text Evaluation with BERTScor (https://arxiv.org/abs/2409.17160)
- **What's New**: BERTScoreVisualizer는 BERTScore 메트릭의 토큰 매칭 정보를 시각화하여 텍스트 단순화 시스템의 품질 분석을 향상시키는 웹 애플리케이션입니다.

- **Technical Details**: 이 도구는 BERTScore의 정밀도, 재현율, F1 점수 외에 참조 텍스트와 후보 텍스트 간의 토큰 매칭 정보를 시각적으로 제공합니다. React 웹 애플리케이션으로 프론트엔드를 구현하고, Flask를 사용해 백엔드를 구성하여 BERT 추론 및 BERTScore 알고리즘을 처리합니다.

- **Performance Highlights**: BERTScoreVisualizer의 독창적인 기능은 각 토큰의 재현율 및 정밀도 점수를 팝업으로 표시하고, 매칭되지 않은 토큰을 강조하여 더 간편한 분석을 제공합니다. 이로 인해 텍스트 품질을 평가하는 데 있어 귀중한 인사이트를 제공합니다.



### Infer Human's Intentions Before Following Natural Language Instructions (https://arxiv.org/abs/2409.18073)
- **What's New**: 이 논문에서는 Ambiguous natural language instructions를 효과적으로 수행하기 위한 새로운 프레임워크인 Follow Instructions with Social and Embodied Reasoning (FISER)를 제안합니다. 이 프레임워크는 사람의 목표와 의도를 추론하는 단계를 명시적으로 포함합니다. 이는 AI 에이전트가 인간의 자연어 명령을 이해하고 협력적 작업을 더 잘 수행하도록 돕습니다.

- **Technical Details**: FISER는 두 가지 주요 구성 요소인 social reasoning과 embodied reasoning을 사용하여 모델이 인간의 의도를 명시적으로 추론할 수 있도록 합니다. 먼저 social reasoning을 통해 인간이 요청하는 하위 작업(sub-task)을 예측한 후, 이러한 지침을 로봇이 이해할 수 있는 작업으로 변환하는 Embodied reasoning 단계로 진행합니다. 또한, FISER는 계획 인식(plan recognition) 단계를 추가하여 인간의 전반적인 계획을 추론하는 데 도움을 줍니다.

- **Performance Highlights**: FISER 모델은 HandMeThat(HMT) 벤치마크에서 64.5%의 성공률을 기록하며, 이전의 end-to-end 접근법을 초월한 성능을 보여줍니다. 또한, FISER는 체인 오브 토트(Chain-of-Thought) 방식으로 GPT-4를 기반으로 한 강력한 기준과 비교했을 때도 우수한 결과를 도출했습니다. 이 결과는 인간의 의도에 대한 중간 추론을 명시적으로 수행하는 것이 AI의 성능을 개선하는 데 효과적임을 입증합니다.



### IFCap: Image-like Retrieval and Frequency-based Entity Filtering for Zero-shot Captioning (https://arxiv.org/abs/2409.18046)
Comments:
          Accepted to EMNLP 2024

- **What's New**: 최근 이미지 캡셔닝(image captioning) 분야에서 이미지-텍스트 데이터 쌍의 한계를 극복하기 위해 텍스트-전용(training) 교육 방법이 탐색되고 있습니다. 본 논문에서는 텍스트 데이터와 이미지 데이터 간의 모달리티 차이를 완화하기 위한 새로운 접근 방식으로 'Image-like Retrieval'을 제안합니다.

- **Technical Details**: 제안된 방법인 IFCap($\textbf{I}$mage-like Retrieval과 $\textbf{F}$requency-based Entity Filtering for Zero-shot $\textbf{Cap}$tioning)는 효율적인 이미지 캡셔닝을 위한 통합 프레임워크로, Fusion Module을 통해 검색된 캡션과 입력 특성을 통합하여 캡션 품질을 향상시킵니다. 또한 Frequency-based Entity Filtering 기술을 도입하여 더 나은 캡션 품질을 제공합니다.

- **Performance Highlights**: 광범위한 실험 결과, IFCap은 기존의 텍스트-전용 훈련 기반 제로샷 캡셔닝(zero-shot captioning) 방식에 비해 이미지 캡셔닝과 비디오 캡셔닝 모두에서 state-of-the-art 성능을 기록하는 것으로 나타났습니다.



### EMOVA: Empowering Language Models to See, Hear and Speak with Vivid Emotions (https://arxiv.org/abs/2409.18042)
Comments:
          Project Page: this https URL

- **What's New**: EMOVA(EMotionally Omni-present Voice Assistant)는 음성 대화에서 다양한 감정과 톤을 지원하며, 최첨단 비전-언어(vision-language) 및 음성(speech) 능력을 결합한 최초의 옴니모달(omni-modal) LLM입니다. 이 모델은 기존 비전-언어 및 음성-언어 모델의 한계를 극복하고, 실시간 대화의 요구를 충족합니다.

- **Technical Details**: EMOVA는 연속 비전 인코더와 의미-음향 분리된 음성 토크나이저를 사용하여 음성 이해 및 생성을 위한 엔드투엔드(end-to-end) 아키텍처를 갖추고 있습니다. 이를 통해 입력 음성의 의미적 내용과 음향적 스타일을 분리하여 다양한 음성 스타일 조절을 지원합니다.

- **Performance Highlights**: EMOVA는 첫 번째로 비전-언어 및 음성 벤치마크에서 최첨단 성능을 달성하였으며, 공공 데이터셋을 활용하여 옴니모달 정렬을 효율적으로 수행하고, 생생한 감정을 담은 음성 대화를 지원합니다.



### Compositional Hardness of Code in Large Language Models -- A Probabilistic Perspectiv (https://arxiv.org/abs/2409.18028)
- **What's New**: 본 연구는 큰 언어 모델(LLM)이 동일한 컨텍스트 내에서 여러 하위 작업을 수행하는 능력에 한계가 있음을 지적하고, 이를 극복하기 위해 멀티 에이전트 시스템을 활용한 문제 해결 방안을 제시합니다.

- **Technical Details**: 본 연구에서는 생성 복잡도(Generation Complexity)라는 지표로 LLM이 정확한 솔루션을 샘플링하기 위해 필요한 생성 횟수를 정량화 하며, 복합 코딩 문제에서 LLM과 멀티 에이전트 시스템 간의 성능 차이를 분석합니다. LLM을 오토회귀 모델로 모델링하고, 두 개의 다른 문제를 독립적으로 해결하는 방법을 논의합니다.

- **Performance Highlights**: 실험적으로, Llama 3 모델을 사용하여 복합 코드 문제에 대한 생성 복잡도의 기하급수적 차이를 입증하였으며, 이는 동일한 컨텍스트 내에서 문제를 해결하기에 LLM이 더 어려움을 겪는 다는 것을 보여줍니다.



### An Adversarial Perspective on Machine Unlearning for AI Safety (https://arxiv.org/abs/2409.18025)
- **What's New**: 이번 연구는 기존의 unlearning(언러닝) 방법이 안전성 훈련(safety training)에서의 위험한 정보 제거를 효과적으로 대체할 수 있는지를 조명합니다. 연구자들은 unlearning이 단순히 정보를 숨기는데 그치고, 위험한 지식이 여전히 회복될 수 있음을 보여줍니다.

- **Technical Details**: 이 논문은 RMU(Residual Memory Unlearning)와 같은 상태-of-the-art unlearning 방법을 평가하며, WMDP(WMD Probing) 벤치마크를 사용하여 안전성 훈련과 비교합니다. 기존 보고된 jailbreak 방법들은 unlearning에 무효로 간주되었으나, 세심하게 적용될 경우 여전히 효과적일 수 있음을 발견하였습니다. 이들은 특정 활성 공간 방향을 제거하거나 무관한 예제를 가진 finetuning을 통해 원래의 성능을 복구할 수 있는 방법을 제시합니다.

- **Performance Highlights**: 연구 결과, unlearning 방법은 특정 공격에 보다 강해 보이지만, 안전성 훈련에 사용된 기법으로 쉽게 우회될 수 있습니다. 특히, GCG와 같은 jailbreak 기법은 손실 함수를 약간 변경함으로써 의미 있는 정확도를 회복할 수 있음을 보여주었습니다.



### Extracting Affect Aggregates from Longitudinal Social Media Data with Temporal Adapters for Large Language Models (https://arxiv.org/abs/2409.17990)
Comments:
          Code available at this https URL

- **What's New**: 이번 연구에서는 사회 미디어 데이터의 종단적 분석을 위한 도구로, 시간적으로 정렬된 Large Language Models (LLMs)을 제안합니다. 이 연구는 사용자 생성 데이터를 기반으로 Temporal Adapters를 Llama 3 8B에 맞춰 조정하여 긴 시간에 걸쳐 감정과 태도를 추출합니다.

- **Technical Details**: Temporal Adapters를 통해 Llama 3 8B 모델을 조정하고, 24,000명의 영국 Twitter 사용자로부터 2019년 11월부터 2020년 6월까지의 데이터를 기반으로 데이터를 수집합니다. 조정된 모델에 여러 설문지로부터 프롬프트를 제공하여 토큰 확률로부터 longitudinal affect aggregates를 추출합니다.

- **Performance Highlights**: 우리의 추정치는 영국 성인 집단의 설문 데이터와 강한 긍정적 상관관계를 보였으며, 전통적인 분류 모델을 사용했을 때와 일관성 있는 결과를 보여줍니다. 이 방법은 기존의 LLM을 피드백해서 사용자 생성 데이터에 맞추도록 조정하여 감정 집합의 변화를 연구하는 데 유용합니다.



### Weak-To-Strong Backdoor Attacks for LLMs with Contrastive Knowledge Distillation (https://arxiv.org/abs/2409.17946)
- **What's New**: 이번 연구에서는 Parameter-Efficient Fine-Tuning (PEFT)을 사용하는 대규모 언어 모델(LLMs)에 대한 백도어 공격의 효율성을 검증하고, 이를 개선하기 위한 새로운 알고리즘인 W2SAttack(Weak-to-Strong Attack)을 제안합니다.

- **Technical Details**: W2SAttack은 약한 모델에서 강한 모델로 백도어 특징을 전이하는 대조적 지식 증류(constractive knowledge distillation) 기반의 알고리즘입니다. 이 알고리즘은 소규모 언어 모델을 사용하여 완전 파라미터 미세 조정을 통해 백도어를 임베드하고 이를 교사 모델로 활용하여 대규모 학생 모델로 전이합니다. 이 과정에서 정보를 최소화하여 학생 모델이 목표 레이블과 트리거 간의 정렬을 최적화하도록 합니다.

- **Performance Highlights**: W2SAttack은 여러 언어 모델과 백도어 공격 알고리즘을 대상으로 한 실험 결과에서 100%에 가까운 성공률을 기록했습니다. 이는 PEFT를 사용한 기존 백도어 공격의 성능을 크게 개선한 결과입니다.



### Unveiling the Potential of Graph Neural Networks in SME Credit Risk Assessmen (https://arxiv.org/abs/2409.17909)
- **What's New**: 본 논문은 그래프 신경망 (Graph Neural Network)을 활용한 기업 신용 위험 평가 모델을 제안하며, 기업 재무 지표 간의 내재적 연결을 통합합니다.

- **Technical Details**: 29개의 기업 재무 데이터 지표를 선택하고 각 지표를 정점 (vertex)으로 추상화했습니다. 유사도 행렬 (similarity matrix)을 구성하고 최대 신장 트리 알고리즘 (maximum spanning tree algorithm)을 사용하여 기업의 그래프 구조 매핑을 수행했습니다. 매핑된 그래프의 표현 학습 단계에서 그래프 신경망 모델을 구축하여 32차원의 임베딩 표현을 얻었습니다. 세 가지 GraphSAGE 연산을 수행하고 Pool 연산을 통해 결과를 집계했습니다.

- **Performance Highlights**: 실제 기업 데이터에 대한 실험 결과, 제안된 모델은 다단계 신용 수준 추정 작업을 효과적으로 수행하며, ROC 및 기타 평가 기준에 따라 모델의 분류 효과가 중요하고 안정성 (robustness)도 뛰어나며, 다양한 지표 데이터 간의 내재적 연결을 심층적으로 표현합니다.



### Revisiting Acoustic Similarity in Emotional Speech and Music via Self-Supervised Representations (https://arxiv.org/abs/2409.17899)
- **What's New**: 이 연구는 음성 감정 인식(SER)과 음악 감정 인식(MER) 간의 지식을 전이할 수 있는 가능성을 탐구하고, 자가 감독 학습(Self-Supervised Learning, SSL) 모델에서 추출된 공통의 음향 특징을 활용하여 감정 인식 성능을 향상시키는 방법을 제시합니다.

- **Technical Details**: 이 연구에서는 RAVDESS 데이터셋을 사용하여 SSL 모델의 층별 행동을 분석하고, 두 단계의 미세 조정을 통해 SER과 MER 간의 도메인 적응 방법을 비교합니다. 또한, Frechet 오디오 거리(Frechet audio distance)를 사용하여 감정별로 음성 및 음악의 음향 유사성을 평가합니다. 세 가지 SSL 모델(Wav2Vec 2.0, HuBERT, MERT)을 적용하여 각 모델의 음성 및 음악 데이터에서의 성능을 조사합니다.

- **Performance Highlights**: 연구 결과, SER과 MER에서 SSL 모델이 공통된 음향 특징을 잘 포착하긴 하지만, 각기 다른 감정에 따라 그 행동이 다르게 나타납니다. 또한, 효율적인 매개변수 기반 미세 조정을 통해 서로의 지식을 활용하여 SER과 MER 성능을 향상할 수 있음을 보여줍니다.



### Implementing a Nordic-Baltic Federated Health Data Network: a case repor (https://arxiv.org/abs/2409.17865)
Comments:
          24 pages (including appendices), 1 figure

- **What's New**: 이 논문에서는 북유럽-발트해 지역에서 건강 데이터의 2차 사용을 촉진하기 위해 5개국 6개 기관으로 구성된 연합 건강 데이터 네트워크(federated health data network)를 개발하는 과정에서 얻은 초기 경험을 공유합니다.

- **Technical Details**: 이 연구는 혼합 방법(mixed-method approach)을 사용하여 실험 설계(experimental design)와 실행 과학(implementation science)을 결합하여 네트워크 구현에 영향을 미치는 요소를 평가했습니다. 실험 결과, 중앙 집중식 시뮬레이션(centralized simulation)과 비교할 때, 네트워크는 성능 저하(performance degradation) 없이 기능한다는 것을 발견했습니다.

- **Performance Highlights**: 다학제적 접근 방식(interdisciplinary approaches)을 활용하면 이러한 협력 네트워크(collaborative networks)를 구축하는 데 따른 도전 과제를 해결할 수 있는 잠재력이 있지만, 규제 환경이 불확실하고 상당한 운영 비용(operational costs)이 발생하는 것이 문제로 지적되었습니다.



### SECURE: Semantics-aware Embodied Conversation under Unawareness for Lifelong Robot Learning (https://arxiv.org/abs/2409.17755)
Comments:
          10 pages,4 figures, 2 tables

- **What's New**: 본 연구는 로봇이 작업 수행에 필요한 개념을 인식하지 못한 상황에서 진행되는 상호작용적 작업 학습(interactive task learning) 문제를 다루며, 이를 해결하기 위한 새로운 프레임워크 SECURE를 제안합니다.

- **Technical Details**: SECURE는 상징적 추론(symbolic reasoning)과 신경 기초(neural grounding)를 결합하여 로봇이 대화 중에 새로운 개념을 인식하고 학습할 수 있도록 설계되었습니다. 이 프레임워크는 서울대학교의 대학원 연구팀이 개발하였으며, 기존의 기계 학습 모델이 가지는 한계를 극복하고 지속적인 개념 학습을 가능하게 합니다.

- **Performance Highlights**: SECURE를 이용한 로봇은 이전에 알지 못했던 개념을 효과적으로 학습하고, 그로 인해 무의식적 인식 상태에서도 작업 재배치 문제를 성공적으로 해결할 수 있음을 보여주었습니다. 이러한 결과는 로봇이 대화의 의미 논리(semantics)적 결과를 활용하여 보다 효과적으로 학습할 수 있음을 입증합니다.



### Are Transformers in Pre-trained LM A Good ASR Encoder? An Empirical Study (https://arxiv.org/abs/2409.17750)
Comments:
          8pages

- **What's New**: 이번 연구에서는 사전 훈련된 언어 모델(PLM) 내에서 변환기(transformers)의 효능을 탐구하며, 이들이 자동 음성 인식(ASR) 시스템의 인코더로서 어떻게 재활용되는지를 논의합니다. 변환기들은 텍스트 기반의 데이터로 훈련되었음에도 불구하고, 음성 데이터에서도 효과적으로 기능을 수행할 수 있다는 가설을 세우고 있습니다.

- **Technical Details**: 우리의 ASR 모델은 Connectionist Temporal Classification (CTC) 기반의 인코더 전용 아키텍처를 채택합니다. 입력 오디오 피쳐 시퀀스와 이에 대응하는 타겟 레이블 시퀀스를 사용하여 CTC 손실 함수가 정의되며, Qwen 모델의 변환기들이 인코더의 설정에서 핵심 역할을 합니다.

- **Performance Highlights**: 실험 결과, PLM에서 파생된 변환기를 사용한 모델들은 CER 및 WER에서 다수의 ASR 작업에서 유의미한 성능 향상을 보여주었으며, 특히 심도 있는 의미 이해가 중요한 시나리오에서 성능을 크게 개선할 수 있었음이 발견되었습니다.



### Few-shot Pairwise Rank Prompting: An Effective Non-Parametric Retrieval Mod (https://arxiv.org/abs/2409.17745)
Comments:
          Accepted to EMNLP 2024

- **What's New**: 이번 연구에서는 기존의 감독 학습 방식의 복잡성을 줄이고, 대규모 언어 모델(LLM) 기반의 간단한 순위 모델을 제안하여 제로샷(zero-shot) 환경에서도 효과적인 성능 향상을 이끌어냈습니다.

- **Technical Details**: 제안된 방법은 쿼리와 문서 쌍을 기준으로 유사한 쿼리들의 선호 예시를 활용하여, 쿼리와 문서의 쌍에 대한 상대적 선호 순서를 예측합니다. 이는 대규모 언어 모델의 언어 처리 능력을 이용한 몇 샷(few-shot) 프롬프트를 통해 이루어집니다. 이를 통해 기존의 제로샷 모델보다 지속적으로 성능이 개선되었음을 보였습니다.

- **Performance Highlights**: 제안된 모델은 TREC DL과 BEIR 서브셋 벤치마크에서 제로샷 기준선보다 일관된 향상을 보였으며, 복잡한 훈련 파이프라인 없이 감독 모델과 유사한 성능을 달성했습니다. 또한, MS-MARCO의 예시 쿼리를 이용한 실험에서 TREC Covid과 SciFact 테스트 컬렉션에 대한 아웃 오브 도메인 일반화의 가능성을 보여주었습니다.



### Digital Twin Ecosystem for Oncology Clinical Operations (https://arxiv.org/abs/2409.17650)
Comments:
          Pre Print

- **What's New**: 이 논문은 인공지능(AI)과 디지털 트윈(Digital Twin) 기술을 활용하여 종양학(oncology) 분야의 임상 운영을 혁신적으로 향상시키기 위한 새로운 디지털 트윈 프레임워크를 소개합니다. 여러 전문 디지털 트윈을 통합하여 각 환자에 대한 개인화된 치료를 가능하게 하며, 이는 기존의 데이터와 NCCN 가이드라인에 기반하여 클리닉 추천을 제공합니다.

- **Technical Details**: 이 프레임워크는 Medical Necessity Twin, Care Navigator Twin, Clinical History Twin과 같은 여러 개의 전문 디지털 트윈을 활용하여 환자 맞춤형 치료와 워크플로우 효율성을 높입니다. 각 디지털 트윈은 실시간 데이터 교환을 통해 작동하며, 이를 통해 환자의 유일한 데이터에 기반한 개인화된 care path를 생성하여 ACA 데이터를 통합합니다. 이 시스템은 또한 여러 지식 기반을 통합하여 복잡한 상호작용을 시뮬레이션하는 멀티 에이전트 시스템(Multi-Agent Systems)은 의사결정 지원 및 care coordination을 강화할 수 있습니다.

- **Performance Highlights**: 이 논문에서 제시한 사례 연구는 다양한 에이전트들이 어떻게 협력하여 워크플로우를 간소화하고, 적시의 임상 추천을 제공할 수 있는지를 보여줍니다. 디지털 트윈 기술과 AI의 결합으로 인해 임상 결정의 정확성과 환자 맞춤형 치료의 효율성이 크게 향상됩니다.



### On the Implicit Relation Between Low-Rank Adaptation and Differential Privacy (https://arxiv.org/abs/2409.17538)
- **What's New**: 이번 논문에서는 자연어 처리에서의 저차원(adaptation) 접근법이 데이터 프라이버시(data privacy)와 어떻게 연결되는지를 제시합니다. 특히, LoRA(Lo-Rank Adaptation)와 FLoRA(Fully Low-Rank Adaptation)의 방법이 미치는 영향을 분석하여, 이들이 데이터 민감도를 고려한 저차원 적응과 유사하다는 것을 보여줍니다.

- **Technical Details**: LoRA와 FLoRA는 언어 모델 언어를 특정 작업에 적응시키기 위해 몇 개의 레이어에 훈련 가능한 저차원 분해 매트릭스(adapter)를 통합하여, 사전 훈련된 모델의 가중치를 고정한 상태에서 사용됩니다. 이 접근법은 전통적인 매개변수 조정(full fine-tuning) 방식에 비해 필요한 훈련 가능한 매개변数의 수를 상당히 줄입니다. 연구진은 또한 저차원 적응이 DPSGD(Differentially Private Stochastic Gradient Descent)와 근본적으로 유사하다는 것을 입증하며, 가우시안 분포(Gaussian distribution)와의 변동성(variance)도 분석합니다.

- **Performance Highlights**: 연구의 주요 기여는 다음과 같습니다: 1) LoRA/FLoRA로 저차원 적응을 수행하는 것이 어댑터의 배치 그라디언트(batch gradients)에 무작위 노이즈를 주입하는 것과 동등하다는 것을 보여줍니다. 2) 주입된 노이즈의 분산을 찾아 노이즈가 입력 수와 저차원 적응의 순위(rank)가 증가함에 따라 가우시안 분포에 가까워지게 됨을 증명합니다. 3) 저차원 적응의 동역학은 DP 완전 조정(DP full fine-tuning) 어댑터와 매우 유사함을 입증하며, 이러한 저차원 적응이 데이터 프라이버시를 제공할 수 있는 가능성을 제시합니다.



### When A Man Says He Is Pregnant: ERP Evidence for A Rational Account of Speaker-contextualized Language Comprehension (https://arxiv.org/abs/2409.17525)
- **What's New**: 이번 연구는 발화와 화자 간의 맥락을 이해하는 과정에서 나타나는 두 가지의 서로 다른 ERP (event-related potential) 효과인 N400과 P600에 대해 다루고 있습니다. 화자와 메시지 간의 불일치 상황에서 신경생리학적 반응을 분석했습니다.

- **Technical Details**: 연구에 참여한 64명의 참가자를 대상으로 한 실험에서, 말의 의미를 이해하기 위해서 사회적 고정관념을 위반하는 경우에는 N400 효과가 발생하고, 생물학적 지식을 위반하는 경우에는 P600 효과가 발생함을 발견했습니다.

- **Performance Highlights**: 참가자의 개성 중 개방성(openness) 성향에 따라 사회적 N400 효과는 감소했지만, 생물학적 P600 효과는 여전히 강력하게 나타났습니다. 이러한 결과는 기존의 연구에서 나타난 경량한 불일치를 해소하는 데 기여합니다.



### Comparing Unidirectional, Bidirectional, and Word2vec Models for Discovering Vulnerabilities in Compiled Lifted Cod (https://arxiv.org/abs/2409.17513)
Comments:
          6 pages, 2 figures

- **What's New**: 이 연구는 LLVM 코드에서 GPT 모델을 처음부터 훈련하여 생성된 임베딩을 사용해 특정 취약성(예: buffer overflows)을 찾는 결과를 제공합니다. 또한 단방향 변환기(GPT-2)와 양방향 변환기(BERT, RoBERTa) 및 비변환기 기반 임베딩 모델(Skip-Gram, Continuous Bag of Words)의 영향을 비교하고 있습니다.

- **Technical Details**: 연구는 먼저 NIST SARD Juliet 데이터 세트에서 코드 샘플을 선택한 후, RetDec 도구를 사용하여 LLVM으로 변환한 다음 사전 처리를 진행하였습니다. 이후 CWE-121 샘플을 사용하여 GPT-2 모델의 임베딩을 생성하고, 이러한 임베딩을 LSTM 신경망 훈련에 사용하였습니다.

- **Performance Highlights**: GPT-2로부터 생성된 임베딩은 BERT 및 RoBERTa의 양방향 모델보다 뛰어난 성능을 보여주었으며, 92.5%의 정확도와 89.7%의 F1 점수를 기록했습니다. 또한 SGD 옵티마이저가 Adam보다 우수한 성능을 나타냈습니다.



### HaloScope: Harnessing Unlabeled LLM Generations for Hallucination Detection (https://arxiv.org/abs/2409.17504)
Comments:
          NeurIPS 2024 Spotlight

- **What's New**: 이번 논문에서는 HaloScope라는 새로운 학습 프레임워크를 소개합니다. 이 프레임워크는 레이블이 없는 LLM 생성물을 활용하여 hallucination을 탐지하는 데 중점을 두고 있습니다. 기존의 신뢰성 분류기 학습의 주요 어려움인 레이블이 붙은 데이터 부족 문제를 해결하기 위한 접근법을 제시합니다.

- **Technical Details**: HaloScope는 자동화된 membership estimation score를 통해 레이블이 없는 데이터에서 진실한 생성물과 허위 생성물을 구분합니다. 이 프레임워크는 LLM의 잠재적 표현(latent representations)을 활용하여 허위 진술과 관련된 서브스페이스를 식별하고, 이를 통해 이진 진실성 분류기를 훈련할 수 있도록 합니다.

- **Performance Highlights**: HaloScope는 다양한 데이터셋에 걸쳐 hallucination 탐지 성능을 향상시켰으며, TruthfulQA 벤치마크에서 기존의 최첨단 방법 대비 10.69% (AUROC) 향상된 정확도를 기록했습니다. 이로 인해 HaloScope의 실용성과 유연성이 강화되었습니다.



### MaskLLM: Learnable Semi-Structured Sparsity for Large Language Models (https://arxiv.org/abs/2409.17481)
Comments:
          NeurIPS 2024 Spotlight

- **What's New**: 이번 연구는 MaskLLM이라는 새로운 learnable pruning (학습 가능한 가지치기) 방법을 제안합니다. 이 방법은 LLM (대규모 언어 모델)의 Semi-structured (반구조적) sparsity (희소성)을 통해 추론 시 계산 비용을 줄이는 데 초점을 맞추고 있습니다.

- **Technical Details**: MaskLLM은 Gumbel Softmax sampling (검벨 소프트맥스 샘플링)을 활용하여 N:M 패턴을 학습 가능한 분포로 모델링합니다. 이 방법은 N개의 비제로 값과 M개의 매개변수 사이의 관계를 설정하여 대규모 데이터 집합에서 end-to-end training (엔드투엔드 훈련)을 수행할 수 있게 합니다.

- **Performance Highlights**: MaskLLM은 LLaMA-2, Nemotron-4, GPT-3와 같은 여러 LLM을 사용하여 2:4 sparsity 설정 하에 평가되었습니다. 기존의 최첨단 방법에 비해 PPL (perplexity) 측정에서 유의미한 개선을 보였으며, 특히 SparseGPT의 10.42에 비해 MaskLLM은 6.72의 PPL을 달성하였습니다. 이 결과는 MaskLLM이 대규모 모델에서도 효과적으로 고품질 마스크를 학습할 수 있음을 나타냅니다.



### RED QUEEN: Safeguarding Large Language Models against Concealed Multi-Turn Jailbreaking (https://arxiv.org/abs/2409.17458)
- **What's New**: 이 논문에서는 새로운 jailbreak 공격 방법인 Red Queen Attack을 제안합니다. 이 방법은 다단계(멀티턴) 시나리오를 활용하여 악의적인 의도를 숨기고, 14가지 유해 카테고리에서 56,000개의 멀티턴 공격 데이터를 생성합니다.

- **Technical Details**: Red Queen Attack은 다양한 직업과 관계에 기반하여 40개의 시나리오를 구성하며, GPT-4o, Llama3-70B 등 4개의 대표적인 LLM 가족에 대해 87.62% 및 75.4%의 높은 성공률을 기록했습니다. Red Queen Guard라는 완화 전략도 제안하여, 공격 성공률을 1% 미만으로 줄입니다.

- **Performance Highlights**: 모든 LLM이 Red Queen Attack에 취약하며, 대형 모델일수록 공격에 더 민감하다는 사실이 발견되었습니다. 이 연구는 56k 멀티턴 공격 데이터셋과 함께 LLM의 안전성을 높이는 방법을 제시합니다.



### Description-based Controllable Text-to-Speech with Cross-Lingual Voice Contro (https://arxiv.org/abs/2409.17452)
Comments:
          Submitted to ICASSP 2025

- **What's New**: 이번 연구에서는 음성 생성을 위한 새로운 설명 기반의 제어 가능한 TTS 방법을 제안합니다. 이 방법은 교차 언어 제어 기능을 갖추고 있으며, 대상 언어에서 오디오-설명 쌍 데이터의 부족 문제를 해결하기 위해 다국어 간의 연관성을 활용합니다.

- **Technical Details**: 이 TTS 프레임워크는 세 가지 컴포넌트로 구성됩니다: NANSY++ 모델, TTS 음향 모델, 설명 제어 모델. NANSY++ 모델은 자가 학습(self-supervised learning)으로 페이지 분리된 음성 표현을 학습하며, 이를 통해 TTS 모델의 조건적 특징으로 사용되는 음조(timbre)와 스타일(style) 임베딩을 공유합니다. 또한, 설명 제어 모델은 입력된 텍스트 설명을 음조 및 스타일 임베딩으로 변환합니다.

- **Performance Highlights**: 영어와 일본어 TTS 실험 결과, 제안한 방법이 두 언어 모두에서 높은 자연스러움(naturalness)과 제어 가능성(controllability)을 달성하는 것으로 나타났습니다. 특히, 일본어 오디오-설명 쌍 데이터가 없어도 기존 시스템보다 개선된 pitch 및 speaking speed 제어 능력을 보여주었습니다.



### Post-hoc Reward Calibration: A Case Study on Length Bias (https://arxiv.org/abs/2409.17407)
Comments:
          Preprint

- **What's New**: 이 논문은 Large Language Models (LLMs)의 보상 모델 (Reward Model, RM)의 편향을 교정하는 데 도움을 주는 'Post-hoc Reward Calibration' 개념을 소개합니다. 이 접근법은 추가 데이터나 훈련 없이도 성능 향상을 가능하게 합니다.

- **Technical Details**: 편향된 보상을 분해하여 잠재적인 진짜 보상과 특정 특성에 의존하는 편향 항으로 나누는 방법론을 제시합니다. 이를 위해 Locally Weighted Regression 기법을 사용하여 편향을 추정하고 제거하는 방식을 채택하였습니다.

- **Performance Highlights**: 본 연구는 세 가지 실험 설정에서 다음과 같은 성과 향상을 입증하였습니다: 1) RewardBench 데이터셋의 33개 RM에서 평균 3.11 성능 향상, 2) AlpacaEval 벤치마크에 기반한 GPT-4 평가 및 인간 선호도와의 RM 순위 개선, 3) 다양한 LLM-RM 조합에서 RLHF 과정의 Length-Controlled 승률 향상.



### Navigating the Nuances: A Fine-grained Evaluation of Vision-Language Navigation (https://arxiv.org/abs/2409.17313)
Comments:
          EMNLP 2024 Findings; project page: this https URL

- **What's New**: 이 연구는 Vision-Language Navigation (VLN) 작업을 위한 새로운 평가 프레임워크를 제안합니다. 이 프레임워크는 다양한 지침 범주에 대한 현재 모델을 더 세분화된 수준에서 진단하는 것을 목표로 합니다. 특히, context-free grammar (CFG)를 기반으로 한 구조에서 VLN 작업의 지침 카테고리를 설계하고, 이를 Large-Language Models (LLMs)의 도움으로 반자동으로 구성합니다.

- **Technical Details**: 제안된 평가 프레임워크는 atom instruction, 즉 VLN 지침의 기본 행동에 집중합니다. CFG를 사용하여 지침의 구조를 체계적으로 구성하고 5개의 주요 지침 범주(방향 변화, 수직 이동, 랜드마크 인식, 영역 인식, 숫자 이해)를 정의합니다. 이 데이터를 활용하여 평가 데이터셋 NavNuances를 생성하고, 이를 통해 다양한 모델의 성능을 평가하며 문제가 드러나는 경우가 많았습니다.

- **Performance Highlights**: 실험 결과, 모델 간 성능 차이와 일반적인 문제점이 드러났습니다. LLM에 의해 강화된 제로샷 제어 에이전트가 전통적인 감독 학습 모델보다 방향 변화와 랜드마크 인식에서 더 높은 성능을 보였으며, 반면 기존 감독 접근 방식은 선택적 편향으로 인해 원자 개념 변화에 적응하는 데 어려움을 겪었습니다. 이러한 분석은 VLN 방식의 향후 발전에 중요한 통찰력을 제공합니다.



### On the Vulnerability of Applying Retrieval-Augmented Generation within Knowledge-Intensive Application Domains (https://arxiv.org/abs/2409.17275)
- **What's New**: 이 논문은 Retrieval-Augmented Generation (RAG) 시스템의 적대적 강인성(adversarial robustness)을 조사하였습니다. 특히, 의학 분야의 Q&A를 대상으로 한 '보편적인 중독 공격(universal poisoning attacks)'의 취약성을 분석하고 새로운 탐지 기반(defense) 방어 체계를 개발하였습니다.

- **Technical Details**: RAG 시스템은 외부 데이터에서 중요한 정보를 검색(retrieve)하고 이를 LLM의 생성 과정에 통합하는 두 가지 단계를 포함합니다. 이 연구에서는 225개 다양한 설정에서 RAG의 검색 시스템이 개인 식별 정보(PII)와 같은 다양한 타겟 정보를 포함하는 중독 문서에 취약하다는 것을 입증하였습니다. 중독된 문서는 쿼리의 임베딩과 높은 유사성을 유지함으로써 정확히 검색될 수 있음을 발견하였습니다.

- **Performance Highlights**: 제안한 방어 방법은 다양한 Q&A 도메인에서 뛰어난 탐지율(detection rates)을 일관되게 달성하며, 기존의 방어 방법에 비해 훨씬 효과적임을 보여주었습니다. 실험에 따르면 거의 모든 공격에 대해 일관되게 높은 탐지 성공률을 보였습니다.



### Proof of Thought : Neurosymbolic Program Synthesis allows Robust and Interpretable Reasoning (https://arxiv.org/abs/2409.17270)
- **What's New**: 이 연구에서는 LLM(대형 언어 모델)의 출력 신뢰성과 투명성을 높이기 위해 'Proof of Thought'라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 LLM이 생성한 아이디어와 형식 논리 검증을 연결합니다.

- **Technical Details**: Proof of Thought는 사용자 친화적인 개념을 사용하는 JSON 기반의 도메인 특화 언어(DSL)로, 이를 통해 LLM의 출력을 1차 논리(First Order Logic) 구조로 변환하는 커스텀 인터프리터를 사용합니다. 이 방법은 논리적 구조와 인간이 이해할 수 있는 개념 사이의 균형을 이루도록 설계되었습니다.

- **Performance Highlights**: Proof of Thought는 StrategyQA 및 새로운 멀티모달 추론 작업에서 효과를 입증하였으며, 개방형 시나리오에서 향상된 성능을 보였습니다. 이는 AI 시스템의 책임성을 다루고, 높은 위험 도메인에서 인간의 관여를 위한 기초를 설정하는 데 기여합니다.



### StressPrompt: Does Stress Impact Large Language Models and Human Performance Similarly? (https://arxiv.org/abs/2409.17167)
Comments:
          11 pages, 9 figures

- **What's New**: 이번 연구는 Large Language Models (LLMs)가 인간과 유사한 스트레스 반응을 보이는지를 탐구하고, 다양한 스트레스 유도 프롬프트 하에서의 성능 변화를 평가합니다. 이는 스트레스의 심리적 원리를 바탕으로 한 새로운 종류의 프롬프트 세트, StressPrompt를 통해 이루어졌습니다.

- **Technical Details**: 연구에서는 심리학적 이론에 기반하여 설계된 100개의 프롬프트를 개발하였으며, 이는 각각 다른 수준의 스트레스를 유도하도록 설계되었습니다. 또한, LLM의 내부 상태와 성능에 미치는 스트레스의 영향을 측정하기 위한 '스트레스 스캐너'를 도입했습니다.

- **Performance Highlights**: 연구 결과, LLM은 중간 수준의 스트레스 하에서 최적의 성능을 보이며, 낮은 스트레스와 높은 스트레스 모두에서 성능이 저하되는 것으로 나타났습니다. 이는 Yerkes-Dodson 법칙에 부합하며, 고객 서비스, 의료, 응급 대응과 같은 실세계 시나리오에서 AI 시스템의 성능 유지의 중요성을 시사합니다.



### TCSinger: Zero-Shot Singing Voice Synthesis with Style Transfer and Multi-Level Style Contro (https://arxiv.org/abs/2409.15977)
Comments:
          Accepted by EMNLP 2024

- **What's New**: 본 논문에서는 TCSinger를 소개합니다. TCSinger는 스타일 전이(style transfer) 및 스타일 제어(style control)를 지원하며, 다양한 언어의 음성 및 노래 스타일에 대한 제로샷(Zero-shot) 노래 목소리 합성(SVS) 모델입니다. 이 모델은 개인화된 제어가 가능한 노래 합성을 가능하게 합니다.

- **Technical Details**: TCSinger는 세 가지 주요 모듈로 구성됩니다: 1) clustering style encoder는 클러스터 벡터 양자화(clustering vector quantization) 모델을 사용하여 스타일 정보를 안정적으로 축소하여 잠재 공간(latent space)에 압축합니다. 2) Style and Duration Language Model (S&D-LM)은 스타일 정보와 음소 지속시간(phoneme duration)을 함께 예측하여 두 가지 작업 모두에 도움을 줍니다. 3) style adaptive decoder는 멜 스타일 적응(normalization) 방법을 이용하여 풍부한 스타일 세부정보가 포함된 노래 목소리를 생성합니다.

- **Performance Highlights**: TCSinger의 실험 결과는 합성 품질(synthesis quality), 가수 유사성(singer similarity), 스타일 제어 가능성과 같은 지표에서 모든 기준 모델을 초과하여 성능을 입증했습니다. 여러 작업에서의 성과에는 제로샷 스타일 전이, 다단계 스타일 제어, 교차 언어 스타일 전이 및 음성-노래 스타일 전이가 포함됩니다.



New uploads on arXiv(cs.IR)

### Report on the Workshop on Simulations for Information Access (Sim4IA 2024) at SIGIR 2024 (https://arxiv.org/abs/2409.18024)
Comments:
          Preprint of a SIGIR Forum submission for Vol. 58 No. 2 - December 2024

- **What's New**: 2024년 SIGIR에서 열린 Sim4IA 워크숍에서는 사용자 시뮬레이션(user simulation)의 중요성을 강조하며, 온라인 및 오프라인 평가의 간극을 메울 가능성과 정보 접근을 위한 사용자 시뮬레이션 공유 작업의 조직 문제를 다루었습니다.

- **Technical Details**: 워크숍에서는 사용자 시뮬레이션을 기반으로 한 정보 접근 시스템 평가에 관한 연구자 및 전문가들이 모였으며, 개인화된 정보 접근과 사용자 모델링(user modeling)에 대한 논의가 이루어졌습니다. Gabriella Pasi와 Martin Mladenov의 기조 강연에서는 사용자 경험을 활용한 개인화의 필요성과 사용자 시뮬레이션을 A/B 테스트를 대체할 수 있는 가능성에 대해 논의했습니다.

- **Performance Highlights**: Sim4IA 워크숍은 25명의 참가자가 참여하여 활발한 논의가 이루어진 상호작용 중심의 행사였으며, 참가자들은 짧은 강연과 패널 토론, 브레이크 아웃 세션을 통해 향후 연구 주제에 대한 심화 논의를 진행했습니다.



### Enhancing Tourism Recommender Systems for Sustainable City Trips Using Retrieval-Augmented Generation (https://arxiv.org/abs/2409.18003)
Comments:
          Accepted at the RecSoGood 2024 Workshop co-located with the 18th ACM Conference on Recommender Systems (RecSys 2024)

- **What's New**: 이 논문은 지속 가능한 관광을 고려하여 관광 추천 시스템(Tourism Recommender Systems, TRS)의 새로운 접근 방식을 제안합니다. 이 시스템은 대규모 언어 모델(Large Language Models, LLMs)과 수정된 Retrieval-Augmented Generation(RAG) 파이프라인을 통합하여 사용자 맞춤형 추천을 개선하고 있습니다.

- **Technical Details**: 전통적인 RAG 시스템에 도시의 인기와 계절 수요 기반의 지속 가능성 지표(Sustainability Metric)를 포함하여, Sustainability Augmented Reranking(SAR)라는 수정을 통해 추천의 지속 가능성을 보장합니다. 이 시스템은 사용자 쿼리에 대한 자연어 응답을 기반으로 유럽의 지속 가능한 도시를 추천하도록 설계되었습니다.

- **Performance Highlights**: Llama-3.1-Instruct-8B와 Mistral-Instruct-7B와 같은 인기 있는 오픈 소스 LLM을 사용하여 평가한 결과, SAR이 적용된 방식이 대부분의 메트릭에서 기준선(baseline)과 일치하거나 성능이 우수함을 보여, TRS의 지속 가능성을 통합하는 것의 이점을 강조합니다.



### A Multimodal Single-Branch Embedding Network for Recommendation in Cold-Start and Missing Modality Scenarios (https://arxiv.org/abs/2409.17864)
Comments:
          Accepted at 18th ACM Conference on Recommender Systems (RecSys '24)

- **What's New**: 이 논문은 추천 시스템에서 콜드 스타트(cold-start) 문제를 해결하기 위한 새로운 방법으로, 멀티모달(single-branch) 임베딩 네트워크를 제안합니다. 이를 통해 다양한 모달리티에 기반한 추천 성능을 개선하고자 합니다.

- **Technical Details**: 제안된 방법은 SiBraR(Single-Branch embedding network for Recommendation)로 불리며, 여러 모달리티 데이터를 공유하는 단일 브랜치 네트워크를 통해 처리합니다. SiBraR는 상호작용 데이터와 여러 형태의 사이드 정보를 동일한 네트워크에서 인코딩하여 모달리티 간의 간극(modality gap)을 줄입니다.

- **Performance Highlights**: 대규모 추천 데이터셋에서 실시한 실험 결과, SiBraR는 콜드 스타트 과제에서 기존의 CF 알고리즘과 최신 콘텐츠 기반 추천 시스템(Content-Based Recommender Systems)보다 유의미하게 우수한 성능을 보였습니다.



### Value Identification in Multistakeholder Recommender Systems for Humanities and Historical Research: The Case of the Digital Archive Monasterium.n (https://arxiv.org/abs/2409.17769)
Comments:
          To be presented at: NORMalize 2024: The Second Workshop on the Normative Design and Evaluation of Recommender Systems, October 18, 2024, co-located with the ACM Conference on Recommender Systems 2024 (RecSys 2024), Bari, Italy

- **What's New**: 본 논문은 인문학 및 역사 연구 분야에서 추천 시스템(Recommender Systems, RecSys)의 활용 가능성을 탐구합니다. 특히 Monasterium.net이라는 디지털 아카이브를 중심으로 다양한 이해관계자(stakeholders)의 가치(hostakeholder values)를 식별하고, 이들의 상충하는 요구를 이해하여 추천 시스템의 효용성을 높이는 방안을 제시합니다.

- **Technical Details**: 추천 시스템은 과거의 사용 행동을 분석하여 사용자에게 관련 콘텐츠를 제안하는 기술입니다. 본 논문에서는 인문학과 디지털 인문학(Digital Humanities, DH)의 맥락에서 추천 시스템의 적용이 적절히 이루어지지 않았음을 언급하며, 특히 법적 문서와 문화유산 데이터의 집합 및 전파에 유용할 수 있는 가능성을 강조합니다. Monasterium.net은 약 65만 개의 차터(charters)를 포함한 디지털 아카이브로, 현재 기계 학습(machine learning) 파이프라인의 통합을 통해 사용자 경험을 향상시키고 있습니다.

- **Performance Highlights**: 현재 Monasterium.net은 유럽과 북미에서 매월 약 4,000회의 방문을 받고 있으며, 3,000명의 가입 사용자가 데이터 생성 및 주석 작성 기능에 접근하고 있습니다. 추천 시스템의 도입은 다차원적이고 다양한 데이터를 효과적으로 필터링 및 계층화함으로써 관련 문서의 검색 효율성을 개선할 것으로 기대합니다.



### Few-shot Pairwise Rank Prompting: An Effective Non-Parametric Retrieval Mod (https://arxiv.org/abs/2409.17745)
Comments:
          Accepted to EMNLP 2024

- **What's New**: 이번 연구에서는 기존의 감독 학습 방식의 복잡성을 줄이고, 대규모 언어 모델(LLM) 기반의 간단한 순위 모델을 제안하여 제로샷(zero-shot) 환경에서도 효과적인 성능 향상을 이끌어냈습니다.

- **Technical Details**: 제안된 방법은 쿼리와 문서 쌍을 기준으로 유사한 쿼리들의 선호 예시를 활용하여, 쿼리와 문서의 쌍에 대한 상대적 선호 순서를 예측합니다. 이는 대규모 언어 모델의 언어 처리 능력을 이용한 몇 샷(few-shot) 프롬프트를 통해 이루어집니다. 이를 통해 기존의 제로샷 모델보다 지속적으로 성능이 개선되었음을 보였습니다.

- **Performance Highlights**: 제안된 모델은 TREC DL과 BEIR 서브셋 벤치마크에서 제로샷 기준선보다 일관된 향상을 보였으며, 복잡한 훈련 파이프라인 없이 감독 모델과 유사한 성능을 달성했습니다. 또한, MS-MARCO의 예시 쿼리를 이용한 실험에서 TREC Covid과 SciFact 테스트 컬렉션에 대한 아웃 오브 도메인 일반화의 가능성을 보여주었습니다.



### Autoregressive Generation Strategies for Top-K Sequential Recommendations (https://arxiv.org/abs/2409.17730)
- **What's New**: 이 논문에서는 사용자의 미래 상호작용을 예측하는 Top-K 순차 추천(task)에서 생성적 transformer 기반 모델의 적용 가능성을 탐구합니다. 특히, 일반적으로 사용되는 오토회귀 생성 전략인 greedy decoding, beam search, temperature sampling을 비교하고, 새로운 Reciprocal Rank Aggregation(RRA) 및 Relevance Aggregation(RA) 전략을 제안합니다.

- **Technical Details**: 제안된 접근법은 GPT-2 모델을 사용하여 장기 예측을 수행하는 Top-K 추천 작업에 적용되었습니다. 새로운 multi-sequence 생성 및 집합 방법으로서, 여러 개의 시퀀스를 생성하고 이를 집계하여 최종 추천 목록을 만듭니다. 또한, 다음 항목 예측 작업을 위해 훈련된 모델에서 유용한 성능을 보여줍니다.

- **Performance Highlights**: 제안된 방식은 전통적인 Top-K 예측 방법과 단일 시퀀스 오토회귀 생성 전략에 비해 긴 시간 범위에서 성능을 향상시킵니다. 실험 결과, 제안된 생성 전략이 추천의 성능을 크게 향상시키고 오류 누적을 줄여주는 데 기여함을 보여줍니다.



### Efficient Pointwise-Pairwise Learning-to-Rank for News Recommendation (https://arxiv.org/abs/2409.17711)
- **What's New**: 이 논문에서는 사전 훈련된 언어 모델(PLM)을 기반으로 하는 뉴스 추천의 새로운 프레임워크를 제안합니다. 이 프레임워크는 점수 기반의 pointwise(포인트와이즈) 접근법과 쌍 비교(pairwise) 접근법을 통합하여 규모 문제를 해결합니다.

- **Technical Details**: 제안된 알고리즘인 GLIMPSE는 뉴스 추천을 위해 multi-task(다중 작업) 훈련을 수행합니다. GLIMPSE는 단일 텍스트 생성 작업으로 두 가지 목표(관련성 예측 및 선호도 예측)를 결합하여 최종 순위를 도출합니다. 이 과정에서 Right-To-Left (RTL) 패스를 통해 adjacently(인접하게) 비교를 수행합니다.

- **Performance Highlights**: 광범위한 실험을 통해 MIND와 Adressa 뉴스 추천 데이터셋에서 최신 방법들과 비교하여 성능이 향상된 것을 보여주었습니다.



### Enhancing Structured-Data Retrieval with GraphRAG: Soccer Data Case Study (https://arxiv.org/abs/2409.17580)
- **What's New**: Structured-GraphRAG는 복잡한 데이터셋에서 정보 검색의 정확성과 관련성을 향상시키기 위해 설계된 새로운 프레임워크입니다. 이 방법은 전통적인 데이터 검색 방법의 한계를 극복하고, 구조화된 데이터셋에서 자연어 질의를 통한 정보 검색을 지원합니다.

- **Technical Details**: Structured-GraphRAG는 여러 개의 지식 그래프(knowledge graph)를 활용하여 데이터 간의 복잡한 관계를 포착합니다. 이를 통해 더욱 세밀하고 포괄적인 정보 검색이 가능하며, 결과의 신뢰성을 높이고 언어 모델의 출력을 구조화된 형태로 바탕으로 답변을 제공합니다.

- **Performance Highlights**: Structured-GraphRAG는 전통적인 검색 보완 생성 기법과 비교하여 쿼리 처리 효율성을 크게 향상시켰으며, 응답 시간을 단축시켰습니다. 실험의 초점이 축구 데이터에 맞춰져 있지만, 이 프레임워크는 다양한 구조화된 데이터셋에 널리 적용될 수 있어 데이터 분석 및 언어 모델 응용 프로그램의 향상된 도구로 기능합니다.



### Improving the Shortest Plank: Vulnerability-Aware Adversarial Training for Robust Recommender System (https://arxiv.org/abs/2409.17476)
- **What's New**: 이번 연구에서는 추천 시스템의 취약성을 고려한 새로운 방법인 Vulnerability-aware Adversarial Training (VAT)을 제안합니다. 이 방법은 사용자의 적합도에 따라 공격에 대한 보호를 조정함으로써 추천 품질을 유지합니다.

- **Technical Details**: VAT는 사용자의 적합도를 기반으로 취약성을 추정하는 기능을 구현하고, 이를 바탕으로 사용자별로 적응적인 크기의 perturbations를 적용합니다. 이는 기존의 동일한 크기의 perturbations가 아닌, 사용자 맞춤형으로 이루어져 공격의 성공률을 낮추고 추천 품질을 높입니다.

- **Performance Highlights**: VAT는 다양한 추천 모델 및 여러 유형의 공격에 대해 평균 공격 성공률을 21.53% 감소시키면서도, 추천 성능은 12.36% 향상시키는 것으로 확인되었습니다.



### Towards More Relevant Product Search Ranking Via Large Language Models: An Empirical Study (https://arxiv.org/abs/2409.17460)
Comments:
          To be published in CIKM 2024 GenAIECommerce Workshop

- **What's New**: 이 논문은 e-commerce 제품 검색 순위를 최적화하기 위한 새로운 접근 방식을 제안합니다. 특히, Large Language Models (LLMs)를 사용하여 랭킹 관련성을 콘텐츠 기반(content-based)과 참여 기반(engagement-based)으로 분해하고, 이를 모델 학습 과정에서 효율적으로 활용하는 방법을 다룹니다.

- **Technical Details**: 논문은 Learning-to-Rank (LTR) 프레임워크를 기반으로 하며, Walmart.com의 고객 검색 트래픽 데이터를 통해 모델을 학습합니다. 이 과정에서 콘텐츠 기반 관련성과 참여 기반 관련성으로 랭킹 관련성을 나누고, LLM을 사용하여 두 가지 관련성을 균형 있게 고려한 레이블을 생성합니다. 특히, Mistral 7B 모델을 활용하여 인간 평가 데이터를 바탕으로 콘텐츠 기반 관련성 점수를 미세 조정하는 과정을 포함합니다.

- **Performance Highlights**: 제안된 모델은 콘텐츠 기반과 참여 기반 관련성을 균형 있게 반영하여 검색 결과의 유용성을 개선하며, 온라인 테스트와 오프라인 평가를 통해 그 성능을 입증했습니다. 이는 e-commerce 제품 검색랭킹에 LLM을 통합하는 보다 효과적이고 균형 잡힌 모델 디자인을 제시합니다.



### Long or Short or Both? An Exploration on Lookback Time Windows of Behavioral Features in Product Search Ranking (https://arxiv.org/abs/2409.17456)
Comments:
          Published in ACM SIGIR Workshop on eCommerce 2024

- **What's New**: 본 논문에서는 전자 상거래에서 고객의 쇼핑 행동 특성이 상품 검색 순위 모델에 미치는 영향을 연구했습니다. 특히, (query, product) 레벨의 행동 특성을 집계할 때 사용되는 lookback time window의 효과를 조사하며, 이러한 역사적 행동 특성을 통합하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 행동 특성의 집계 방법에 대한 실증적 연구를 통해, 긴 시간 창 (long time window)과 짧은 시간 창 (short time window)의 장단점을 분석했습니다. 이를 바탕으로, 두 가지 행동 특성을 효과적으로 모델에 통합하는 방법론을 제시했고, Walmart.com의 온라인 A/B 테스트를 통해 이를 검증했습니다.

- **Performance Highlights**: A/B 테스트 결과, 짧은 행동 특성이 Food 및 Consumables 카테고리에서 중요한 역할을 하며, 반대로 ETS와 같은 다이나믹한 환경에서는 짧은 창이 더 유용하다는 것을 확인했습니다. 그러나 두 가지 유형의 행동 특성을 통합한 모델은 전반적으로 성능이 저하되는 경향을 보였습니다.



### Minimizing Live Experiments in Recommender Systems: User Simulation to Evaluate Preference Elicitation Policies (https://arxiv.org/abs/2409.17436)
- **What's New**: 이 연구에서는 YouTube Music 플랫폼의 신규 사용자를 위한 온보딩 프로세스를 평가하는 데 있어, 시뮬레이션 방법론을 개발하여 A/B 테스트를 대체하고 비용을 줄이는 방안을 제시합니다.

- **Technical Details**: 연구의 핵심은 반사적이고 강건한 사용자 행동 모델을 개발하여, 시뮬레이션 서비스를 통해 실시간 실험을 수행하지 않고도 알고리즘의 성능 예측을 가능하게 한다는 것입니다. 이를 위해 순환 신경망(recurrent neural networks)과 트랜스포머(transformers)를 사용하여 사용자 모델을 생성하였습니다.

- **Performance Highlights**: 본 연구는 새로운 PE(preference elicitation) 방법을 통해 신규 사용자가 특정 음악 아티스트에 대한 선호도를 동적으로 질문받는 성과를 보여주었으며, 이러한 프로세스는 실시간 데이터와 연결된 시뮬레이션을 통해 A/B 테스트의 요구 사항을 줄일 수 있음을 시사합니다.



### Results of the Big ANN: NeurIPS'23 competition (https://arxiv.org/abs/2409.17424)
Comments:
          Code: this https URL

- **What's New**: 2023 Big ANN Challenge는 NeurIPS 2023에서 개최되어 Approximate Nearest Neighbor(ANN) 검색의 최신 진행을 목표로 했습니다. 이전에는 전통적인 ANN 검색을 대규모로 확장하는 데 중점을 두었지만, 이번 대회는 필터링 검색, 분포 밖 데이터(out-of-distribution data), 희소(sparse) 및 스트리밍 변형을 다루었습니다.

- **Technical Details**: 이번 대회에서는 필터링된 검색, 분포 밖 데이터, 희소 벡터, 스트리밍 시나리오와 같은 네 가지 트랙이 구성되었습니다. 각 트랙은 데이터베이스로부터 색인을 구축해야 하며, 데이터셋은 공공 클라우드 저장소에서 다운로드할 수 있는 형태로 제공되었습니다. 모든 참가자는 제한된 컴퓨팅 자원으로 새로운 표준 데이터셋에서 제출된 혁신적인 솔루션을 평가받았습니다.

- **Performance Highlights**: 참가 팀들은 업계 표준 기준에 비해 검색 정확성 및 효율성에서 유의미한 개선을 보였으며, 특히 학계 및 산업 팀들로부터 많은 주목할 만한 기여가 있었습니다.



### Enhancing Recommendation with Denoising Auxiliary Task (https://arxiv.org/abs/2409.17402)
- **What's New**: 본 연구는 사용자의 이력(interaction sequences)에서 발생하는 잡음(noise)이 추천 시스템에 미치는 영향을 다루고 있으며, 이를 개선하기 위한 새로운 방법인 자가 감독(Auto-supervised) 보조 작업 결합 훈련(Auxiliary Task Joint Training, ATJT)을 제안합니다.

- **Technical Details**: ATJT 방법은 원본 이력을 바탕으로 랜덤 대체를 통해 인위적으로 생성된 잡음이 포함된 시퀀스를 학습하여 모델의 성능을 향상시키기 위한 것입니다. 잡음 인식 모델과 추천 모델을 난이도에 따라 조정된 가중치로 훈련하여 잡음이 포함된 시퀀스로부터 적절한 학습을 이끌어냅니다.

- **Performance Highlights**: ATJT 방법은 세 개의 데이터셋에서 일관된 기본 모델을 사용하여 실험한 결과, 모델의 추천 성능을 개선하는 데 효과적임을 입증했습니다.



### VectorSearch: Enhancing Document Retrieval with Semantic Embeddings and Optimized Search (https://arxiv.org/abs/2409.17383)
Comments:
          10 pages, 14 figures

- **What's New**: 이 논문은 고차원 데이터 검색의 정확성을 높이기 위해 'VectorSearch'라는 새로운 알고리즘을 제안합니다. 이 시스템은 고급 언어 모델과 다중 벡터 인덱싱 기술을 통합하여 텍스트 데이터의 의미적 관계를 더 잘 파악할 수 있도록 설계되었습니다.

- **Technical Details**: VectorSearch는 데이터의 다차원 임베딩(embeddings)을 효율적으로 검색하는 하이브리드(document retrieval framework) 시스템입니다. HNSWlib 및 FAISS와 같은 최적화 기법을 사용하여 대규모 데이터셋을 효과적으로 관리하고, 복잡한 쿼리 처리 기능을 통해 고급 검색 작업을 지원합니다. 또한, 시스템은 클러스터 환경에서 동적으로 변화하는 데이터셋을 처리할 수 있는 메커니즘을 포함하고 있습니다.

- **Performance Highlights**: 실제 데이터셋을 기반으로 한 실험 결과, VectorSearch는 기존의 기준 메트릭을 초월하여 대규모 검색 작업에서 뛰어난 성능을 보였습니다. 이는 높은 차원의 데이터에서도 저지연성 검색 결과를 제공함으로써 정보 검색의 정확성을 획기적으로 향상시킨 것을 나타냅니다.



### Mamba for Scalable and Efficient Personalized Recommendations (https://arxiv.org/abs/2409.17165)
Comments:
          8 pages, 6 figures, 2 tables

- **What's New**: 이번 연구에서는 개인화 추천 시스템에서 표 형식(tabular data) 데이터를 처리하기 위해 FT-Mamba(Feature Tokenizer + Mamba)라는 새로운 하이브리드 모델을 제안합니다. 이 모델은 FT-Transformer 아키텍처 내의 Transformer 레이어를 Mamba 레이어로 대체하여, 계산 복잡성을 줄이고 효율성을 향상시킵니다.

- **Technical Details**: FT-Mamba 모델은 Mamba 아키텍처를 활용하여, 표 형식 데이터를 시퀀스 형태로 변환하고, Mamba 레이어를 통해 처리합니다. Mamba는 State Space Model(SSM)의 효율성을 개선하며, 시퀀스의 길이에 대해 선형 복잡도(𝒪(L))를 제공합니다. 이는 일반적인 Transformer의 제곱 복잡도(𝒪(L²))를 극복할 수 있게 합니다. 이 모델은 Three datasets (Spotify, H&M, Vaccine messaging)에서 평가되었습니다.

- **Performance Highlights**: FT-Mamba는 기존의 Transformer 기반 모델에 비해 계산 효율성이 향상되었으며, 정확도 같은 주요 추천 지표에서 성능을 유지하거나 초과했습니다. FT-Mamba는 사용자 정보를 인코딩하는 Two-Tower 구조를 사용하여, 개인화 추천 시스템을 위한 확장 가능하고 효과적인 솔루션을 제공합니다.



### Open-World Evaluation for Retrieving Diverse Perspectives (https://arxiv.org/abs/2409.18110)
- **What's New**: 새로운 연구 Benchmark for Retrieval Diversity for Subjective questions (BeRDS)을 소개하여, 여러 관점을 포괄하는 문서 셋을 검색하는 과정을 분석하였습니다.

- **Technical Details**: 이 연구는 복잡한 질문에 대한 다양한 관점을 검색하는 방식을 다룹니다. 단순한 문자열 일치에 의존하지 않고 언어 모델 기반의 자동 평가자를 사용하여 문서가 특정 관점을 포함하는지 판단합니다. 세 가지 유형의 말뭉치(Wikipedia, 웹 스냅샷, 검색 엔진 결과 활용)를 조합한 성능을 평가합니다.

- **Performance Highlights**: 기존의 검색 시스템은 다양한 관점을 포함한 문서 세트를 33.74%의 경우에만 검색할 수 있었습니다. 쿼리 확장 및 다양성 중심의 재정렬 방법을 적용하여 성능 향상을 관찰하였으며, 이러한 접근법들이 밀집 검색기와 결합된 경우 더욱 강력한 결과를 보였습니다.



### Revisit Anything: Visual Place Recognition via Image Segment Retrieva (https://arxiv.org/abs/2409.18049)
Comments:
          Presented at ECCV 2024; Includes supplementary; 29 pages; 8 figures

- **What's New**: 이번 연구에서는 Embodied agents가 시각적으로 장소를 인식하고 이동하는 데 있어 중요한 문제를 다루었습니다. 전체 이미지를 사용하는 기존 방법 대신, 이미지의 '세그먼트'를 인코딩하고 검색하는 새로운 접근 방식을 제안합니다. 이를 통해 SuperSegment라는 새로운 이미지 표현을 생성하여 장소 인식을 향상시킵니다.

- **Technical Details**: 제안된 SegVLAD는 Open-set 이미지 분할(open-set image segmentation)을 통해 이미지를 의미있는 요소(entities)로 분해합니다. 각 아이템은 SuperSegments로 연결되어 구조화됩니다. 새로 제안된 Feature aggregation 방법을 사용하여 이 SuperSegments를 효율적으로 컴팩트한 벡터 표현으로 인코딩합니다. SegVLAD는 다양한 벤치마크 데이터셋에서 기존의 방법보다 높은 인식 리콜을 기록했습니다.

- **Performance Highlights**: SegVLAD는 다양한 VPR 벤치마크 데이터셋에서 최첨단 성능을 달성했습니다. IOU 기반 필터링을 통해 중복성을 줄이고 스토리지를 절약하며, 전체 이미지 기반 검색보다 더욱 뛰어난 성능을 보입니다. 연구 결과, SegVLAD는 이미지 인코더의 특정 작업에 관계없이 적용 가능하고, 객체 인스턴스 검색(object instance retrieval) 과제를 평가하여 '무언가를 재방문(revisit anything)'할 수 있는 잠재력을 보여주었습니다.



### On the Vulnerability of Applying Retrieval-Augmented Generation within Knowledge-Intensive Application Domains (https://arxiv.org/abs/2409.17275)
- **What's New**: 이 논문은 Retrieval-Augmented Generation (RAG) 시스템의 적대적 강인성(adversarial robustness)을 조사하였습니다. 특히, 의학 분야의 Q&A를 대상으로 한 '보편적인 중독 공격(universal poisoning attacks)'의 취약성을 분석하고 새로운 탐지 기반(defense) 방어 체계를 개발하였습니다.

- **Technical Details**: RAG 시스템은 외부 데이터에서 중요한 정보를 검색(retrieve)하고 이를 LLM의 생성 과정에 통합하는 두 가지 단계를 포함합니다. 이 연구에서는 225개 다양한 설정에서 RAG의 검색 시스템이 개인 식별 정보(PII)와 같은 다양한 타겟 정보를 포함하는 중독 문서에 취약하다는 것을 입증하였습니다. 중독된 문서는 쿼리의 임베딩과 높은 유사성을 유지함으로써 정확히 검색될 수 있음을 발견하였습니다.

- **Performance Highlights**: 제안한 방어 방법은 다양한 Q&A 도메인에서 뛰어난 탐지율(detection rates)을 일관되게 달성하며, 기존의 방어 방법에 비해 훨씬 효과적임을 보여주었습니다. 실험에 따르면 거의 모든 공격에 대해 일관되게 높은 탐지 성공률을 보였습니다.



New uploads on arXiv(cs.CV)

### FlowTurbo: Towards Real-time Flow-Based Image Generation with Velocity Refiner (https://arxiv.org/abs/2409.18128)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 이 논문에서는 흐름 기반 생성 모델의 샘플링 속도를 가속화하면서 샘플링 품질을 향상시키는 FlowTurbo라는 새로운 프레임워크를 제안합니다.

- **Technical Details**: FlowTurbo는 샘플링 중 속도 예측기가 안정적이라는 사실을 이용하여 경량화된 속도 정제기를 통해 속도를 추정합니다. 추가로, pseudo corrector 및 sample-aware compilation 기법을 통해 추론 시간을 단축합니다. 이 프레임워크는 다단계 샘플링 패러다임을 변경하지 않으며 이미지 편집이나 인페인팅과 같은 다양한 작업에 효과적으로 적용할 수 있습니다.

- **Performance Highlights**: FlowTurbo를 다양한 흐름 기반 모델에 통합함으로써 클래스 조건 생성에서 53.1%~58.3%의 가속 비율을 달성하였으며, 텍스트-이미지 생성에서는 29.8%~38.5%의 가속을 이뤘습니다. ImageNet에서 FlowTurbo는 100 (ms / img) 당 FID 2.12, 38 (ms / img)에서 FID 3.93을 기록하며 실시간 이미지 생성을 가능하게 했습니다.



### EgoLM: Multi-Modal Language Model of Egocentric Motions (https://arxiv.org/abs/2409.18127)
Comments:
          Project Page: this https URL

- **What's New**: 최근 웨어러블 장치의 보급으로, 인공지능(AI)의 맥락적 이해를 위한 에고센트릭(egocentric) 동작 학습이 필수적입니다. 이 논문에서는 다중 모달(multi-modal) 입력(예: 에고센트릭 비디오와 모션 센서)을 통해 에고센트릭 동작을 추적하고 이해하는 새로운 프레임워크인 EgoLM을 제시합니다.

- **Technical Details**: EgoLM은 두 가지 주요 작업인 에고센트릭 모션 추적(egocentric motion tracking)과 이해(understanding)를 통합하여 다루며, 이를 위해 대규모 언어 모델(large language model, LLM)을 활용하여 에고센트릭 모션과 자연어의 결합 분포(joint distribution)를 모델링합니다. 또한, 모션 센서 데이터와 에고센트릭 비디오를 결합하여 서로 다른 입력을 통합하고, 다중 작업(multi-task) 교육을 통해 효과를 극대화합니다.

- **Performance Highlights**: EgoLM을 대규모 인체 모션 데이터셋인 Nymeria에서 실험한 결과, 기존 모션 추적 및 이해 방법과 비교하여 가장 우수한 성능을 보였습니다. 이 프레임워크는 에고센트릭 학습에 대한 통합적인 접근 방식을 제시하며, 특히 에고센트릭 비디오와 희소(유한) 센서 데이터를 결합한 새로운 설정이 주목받고 있습니다.



### LLaVA-3D: A Simple yet Effective Pathway to Empowering LMMs with 3D-awareness (https://arxiv.org/abs/2409.18125)
Comments:
          Project page: this https URL

- **What's New**: 최근 Large Multimodal Models (LMMs)의 발전이 2D 시각 이해 작업에서의 능력을 크게 향상시켰습니다. 그러나 3D 장면 이해를 위한 3D 인식 모델 개발은 대규모 3D 시각-언어 데이터세트의 부족과 강력한 3D 인코더의 결여로 제약을 받아왔습니다. 본 논문에서 소개된 LLaVA-3D 프레임워크는 2D 이해 능력을 유지하면서 3D 장면 이해를 효율적으로 적응시킬 수 있는 혁신적인 접근 방식입니다.

- **Technical Details**: LLaVA-3D는 간단하지만 효과적인 3D Patch 표현을 도입하여 2D CLIP 패치 특징을 3D 공간의 해당 위치와 연결합니다. 이 방법은 2D LMM에 3D 패치를 통합하고 2D 및 3D 비전-언어 명령 튜닝을 공동으로 수행하여 2D 이미지 이해와 3D 장면 이해를 위한 통합 아키텍처를 생성합니다. 이 모델은 3D 비전-언어 데이터세트에서 훈련된 후, 기존 3D LMM보다 3.5배 빠르게 수렴하여 고도로 효율적인 성능을 보입니다.

- **Performance Highlights**: LLaVA-3D는 3D 캡션 생성, 3D 질문 응답, 3D 그라운딩 등의 다양한 3D 작업에서 최첨단 성능을 달성하였으며, 기존 3D LMM보다 훈련 시간과 에폭 수가 현저히 적습니다. 또한 LLaVA와 비교할 때 2D 이미지 이해 및 언어 대화 능력 또한 유사하게 유지됩니다.



### Lotus: Diffusion-based Visual Foundation Model for High-quality Dense Prediction (https://arxiv.org/abs/2409.18124)
Comments:
          Project page: this https URL

- **What's New**: 본 논문에서는 기존의 diffusion 모델이 dense prediction 작업에 적합하지 않다는 문제를 인식하고, 새로운 접근법인 Lotus 모델을 제안하며 기존 모델 대비 효율성과 성능이 월등히 향상된 결과를 보여줍니다.

- **Technical Details**: Lotus는 noise 예측 대신 직접적인 주석 예측을 통해 모델의 해석을 개선하며, 하나의 단계로 단순화된 diffusion 과정을 도입하여 훈련과 추론 속도를 크게 향상시킵니다. 또한, 'detail preserver'라는 새로운 튜닝 전략을 통해 세밀한 예측을 더 정확하게 수행할 수 있게 합니다.

- **Performance Highlights**: Lotus는 zero-shot monocular depth 및 surface normal estimation에서 SoTA (state-of-the-art) 성능을 달성하였으며, 기존 디퓨전 기반 방법들에 비해 수백 배 빨라 효율성을 극대화했습니다. 이 모델은 59K의 훈련 샘플로도 놀라운 성능을 발휘합니다.



### Multi-View and Multi-Scale Alignment for Contrastive Language-Image Pre-training in Mammography (https://arxiv.org/abs/2409.18119)
Comments:
          This work is also the basis of the overall best solution for the MICCAI 2024 CXR-LT Challenge

- **What's New**: 이 연구는 의료 영상 분야에서 Contrastive Language-Image Pre-training (CLIP) 모델의 초기 적응을 유방 촬영술에 적용하고, 데이터 부족과 고해상도 이미지의 세부 사항 강조를 해결하기 위한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 제안하는 방법은 Multi-view와 Multi-scale Alignment (MaMA)를 기반으로 하며, 각 관점에서 정보를 활용하여 multi-view mammography의 특징을 동시에 정렬하는 시스템을 구축합니다. 또한, clinical report 부족 문제를 해결하기 위해 template-based report construction 방식을 개발하고, parameter-efficient fine-tuning 기법을 적용합니다.

- **Performance Highlights**: EMBED 및 RSNA-Mammo의 대규모 실제 유방촬영 데이터셋에서 3개의 다른 작업에 대해 기존 방법보다 우수한 성능을 보여주었으며, 모델 크기의 52%만으로도 뛰어난 성과를 달성했습니다.



### EdgeRunner: Auto-regressive Auto-encoder for Artistic Mesh Generation (https://arxiv.org/abs/2409.18114)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문에서는 고품질 3D 메쉬 생성을 위한 자동 회귀 자동 인코더(ArAE) 모델을 제안합니다. ArAE 모델은 4,000개 면(면)을 가진 메쉬를 $512^3$의 공간 해상도로 생성할 수 있으며, 새로운 메쉬 토큰화 알고리즘을 도입하여 훈련 효율성을 크게 향상시킵니다.

- **Technical Details**: 제안된 방법인 EdgeRunner는 50%의 시퀀스 길이 압축과 토큰 간의 장거리 의존성 감소를 통해 훈련 효율성을 크게 개선합니다. ArAE는 가변 길이의 삼각형 메쉬를 고정 길이의 잠재 벡터로 압축하며, 이 잠재 공간은 다른 모달리티에 조건화된 잠재 확산 모델을 훈련하는 데 사용될 수 있습니다.

- **Performance Highlights**: 실험 결과, EdgeRunner는 4,000개의 면을 가진 다양한 고품질 아티스틱 메쉬를 생성할 수 있으며, 이전 방법에 비해 두 배 긴 시퀀스와 네 배 높은 해상도를 자랑합니다. 이는 포인트 클라우드 또는 단일 뷰 이미지에서 생성된 메쉬에 대해 뛰어난 일반화 및 강인성을 보여줍니다.



### E.T. Bench: Towards Open-Ended Event-Level Video-Language Understanding (https://arxiv.org/abs/2409.18111)
Comments:
          Accepted to NeurIPS 2024 Datasets and Benchmarks Track

- **What's New**: 최근의 Video Large Language Models (Video-LLMs)와 더불어 E.T. Bench (Event-Level & Time-Sensitive Video Understanding Benchmark)의 도입으로 비디오 이해의 새로운 기준이 세워졌습니다. 이 벤치마크는 현재 존재하는 평가 방식의 한계를 극복하고, 이벤트 레벨의 비디오 이해를 위한 고유한 과제를 제시합니다.

- **Technical Details**: E.T. Bench는 7,300개의 샘플과 12개의 과제로 구성된 대규모 벤치마크로, 8개 도메인에서 7,000개의 비디오(총 251.4시간)를 포함합니다. 이 벤치마크는 3단계 작업 분류법으로 범주화되어 있으며, 참조(referring), 그라운딩(grounding), 밀집 캡션(dense captioning), 복잡한 이해(complex understanding)라는 4가지 주요 기능을 통해 비디오를 평가합니다.

- **Performance Highlights**: 기존 비디오 QA 벤치마크에서 선도적인 모델들은 E.T. Bench에서의 세밀한 과제를 해결하는 데 어려움을 겪었습니다. 특히, E.T. Chat이라는 새로운 모델이 제안되었으며, 이는 타임스탬프 예측을 임베딩 매칭 문제로 재정의함으로써 우수한 성능을 입증했습니다. E.T. Instruct 164K라는 맞춤형 데이터 세트 또한 다중 이벤트와 시간 민감 시나리오를 위해 개발되었습니다.



### Find Rhinos without Finding Rhinos: Active Learning with Multimodal Imagery of South African Rhino Habitats (https://arxiv.org/abs/2409.18104)
Comments:
          9 pages, 9 figures, IJCAI 2023 Special Track on AI for Good

- **What's New**: 이 연구는 코뿔소 보호를 위한 새로운 접근 방식을 제안하고 있으며, 코뿔소의 집단 배변 장소를 지도화하는 것을 통해 이들의 공간 행동에 대한 정보를 수집할 수 있는 방법을 모색하고 있다.

- **Technical Details**: 이 논문은 원거리에서 감지된 열 감지(thermal), RGB, LiDAR 이미지를 사용하여 코뿔소 배변 장소를 탐지하는 분류기를 구축하고, 수동적 및 능동적 학습 설정에서 이를 수행한다. 특히, 기존 능동 학습 방법들이 지극히 불균형한 데이터셋의 문제로 성능이 저하되는 문제를 해결하기 위해, MultimodAL이라고 불리는 새로운 능동 학습 시스템을 설계하였다.

- **Performance Highlights**: 제안된 방법은 94% 감소된 라벨 수로 수동적 학습 모델과 경쟁력을 갖추며, 유사한 크기의 데이터셋에 적용할 경우 라벨링 시간에서 76시간 이상을 절약할 수 있는 것으로 나타났다. 또한, 연구 결과 배변 장소가 무작위 분포가 아니라 클러스터 형태로 밀집해 있다는 점을 발견하여, 이에 따라 기동대(rangers) 활동의 효율성을 높이기 위한 방안을 제시하고 있다.



### AI-Powered Augmented Reality for Satellite Assembly, Integration and Tes (https://arxiv.org/abs/2409.18101)
- **What's New**: 이번 논문은 인공지능(AI)과 증강현실(AR)의 통합을 통해 위성 조립, 통합 및 테스트(AIT) 프로세스를 혁신하려는 유럽우주청(ESA)의 프로젝트 "AI for AR in Satellite AIT"에 대해 설명합니다. 이 시스템은 Microsoft HoloLens 2를 사용하여 기술자에게 실시간으로 맥락에 맞는 지침과 피드백을 제공합니다.

- **Technical Details**: AI4AR 시스템은 컴퓨터와 AR 헤드셋으로 구성되어 있으며, 객체 감지 및 추적, 6D 포즈 추정과 OCR을 포함한 다양한 컴퓨터 비전 알고리즘을 이용합니다. 특히, 6D 포즈 모델 훈련에 합성 데이터를 사용하는 접근법이 독창적이며, 이는 AIT 프로세스의 복잡한 환경에서 유용합니다.

- **Performance Highlights**: AI 모델의 정확도가 70%를 넘었고, 객체 감지 모델은 95% 이상의 정확도를 기록했습니다. 자동 주석화 기능을 가진 Segmented Anything Model for Automatic Labelling(SAMAL)을 통해 실제 데이터의 주석화 속도가 수동 주석화보다 최대 20배 빨라졌습니다.



### Self-supervised Pretraining for Cardiovascular Magnetic Resonance Cine Segmentation (https://arxiv.org/abs/2409.18100)
Comments:
          Accepted to Data Engineering in Medical Imaging (DEMI) Workshop at MICCAI 2024

- **What's New**: 이번 연구는 Self-supervised pretraining (SSP) 방법이 Cardiovascular Magnetic Resonance (CMR) short-axis cine segmentation에서 어떻게 효용성을 가지는지를 평가하고 있으며, 대규모 무라벨 데이터셋을 활용하는 가능성을 탐구합니다.

- **Technical Details**: 본 연구에서는 SimCLR, positional contrastive learning, DINO 및 masked image modeling (MIM) 등 4가지 SSP 방법을 사용하여 296명의 연구대상에서 총 90,618개의 2D 슬라이스를 통한 무라벨 프리트레이닝을 진행했습니다. 이후 여러 수의 레이블이 있는 데이터를 통해 각 SSP 방법으로 2D 모델의 파인튜닝을 수행하여 baseline 모델과 성능을 비교했습니다.

- **Performance Highlights**: 결과적으로, 레이블 데이터가 충분하지 않은 경우, MIM을 사용한 SSP는 0.86의 DSC를 기록하며 무라벨로부터 학습한 모델보다 개선된 결과를 보였습니다. 반면에, 충분한 레이블 데이터가 있을 경우 SSP는 성능 향상을 보이지 않았습니다.



### EfficientCrackNet: A Lightweight Model for Crack Segmentation (https://arxiv.org/abs/2409.18099)
- **What's New**: 본 연구에서는 Convolutional Neural Networks (CNNs)와 transform architecture를 결합한 EfficientCrackNet이라는 경량 하이브리드 모델을 제안합니다. 이 모델은 정밀한 균열(segmentation) 감지를 위해 설계되었으며, 높은 정확도와 낮은 계산 비용을 동시에 제공합니다.

- **Technical Details**: EfficientCrackNet 모델은 depthwise separable convolutions (DSC)와 MobileViT block을 통합하여 전역 및 지역 특성을 캡처합니다. Edge Extraction Method (EEM)를 사용하여 사전 훈련 없이도 효율적인 균열 가장자리 감지를 수행하며, Ultra-Lightweight Subspace Attention Module (ULSAM)을 통해 특성 추출을 향상합니다.

- **Performance Highlights**: 세 개의 벤치마크 데이터셋(Crack500, DeepCrack, GAPs384)을 기반으로 한 광범위한 실험에 따르면, EfficientCrackNet은 단 0.26M의 파라미터와 0.483 FLOPs(G)만으로 기존의 경량 모델들보다 우수한 성능을 보였습니다. 이 모델은 정확성과 계산 효율성 간의 최적의 균형을 제공하여 실제 균열 분할 작업에 강력하고 적응 가능한 솔루션을 제공합니다.



### DiffSSC: Semantic LiDAR Scan Completion using Denoising Diffusion Probabilistic Models (https://arxiv.org/abs/2409.18092)
Comments:
          Under review

- **What's New**: 이 논문에서는 자율 주행 차량을 위한 Semantic Scene Completion (SSC) 작업을 다루고 있으며, LiDAR(빛 감지 및 범위 측정) 센서로부터 얻는 드문 드로우 포인트 클라우드의 빈 공간과 폐쇄된 지역을 예측할 수 있는 방안을 제시합니다.

- **Technical Details**: 제안된 DiffSSC 방법론은 Denoising Diffusion Probabilistic Models (DDPM)을 기반으로 하여 점진적인 노이즈 제거를 통해 LiDAR 점 클라우드를 처리합니다. 이 과정에서는 기하학 및 의미적 정보가 동시에 모델링되며, 로컬과 글로벌 정규화 손실을 통해 학습 과정을 안정화합니다.

- **Performance Highlights**: 제안된 방법은 자율 주행 데이터셋에서 기존의 최첨단 SSC 방법보다 우수한 성능을 보이며, LiDAR 점 클라우드 처리에서 메모리 사용량 및 양자화 오류를 줄이는 데 성공했습니다.



### Stable Video Portraits (https://arxiv.org/abs/2409.18083)
Comments:
          Accepted at ECCV 2024, Project: this https URL

- **What's New**: 본 논문에서는 2D와 3D를 혼합한 새로운 생성 방법인 SVP(Stable Video Portraits)를 제시합니다. 이 방법은 대규모 사전 훈련된 텍스트-투-이미지 모델(Stable Diffusion)과 3D Morphable Models (3DMM)을 활용하여 대화하는 얼굴의 포토리얼리스틱한 비디오를 생성합니다.

- **Technical Details**: SVP는 2D 안정적 확산 모델에 대한 개인화된 미세 조정을 통해 3DMM 시퀀스를 조건으로 제공하고, 시간적 노이즈 제거 절차를 도입하여 비디오 모델로 전환합니다. 이 결과로, 3DMM 기반 제어를 통해 3D 형태의 아바타 이미지를 생성하고, 테스트 시 텍스트에 정의된 유명인사로 아바타의 얼굴을 변형할 수 있는 기능을 포함합니다.

- **Performance Highlights**: SVP 방법은 최신 모노큘러 헤드 아바타 방법보다 우수한 성능을 보이고 있으며, 고충실도의 인체 얼굴 이미지를 세밀하게 재구성하는 능력에서 특히 뛰어납니다. 8분 이상의 비디오 시퀀스를 포함한 포트레이트 아바타 데이터셋도 제공됩니다.



### FreeEdit: Mask-free Reference-based Image Editing with Multi-modal Instruction (https://arxiv.org/abs/2409.18071)
Comments:
          14 pages, 14 figures, project website: this https URL

- **What's New**: 본 논문에서는 사용자가 지정한 시각적 개념을 기반으로 한 이미지 편집을 가능하게 하는 FreeEdit라는 새로운 접근법을 제안합니다. 이 방법은 사용자가 제공한 언어 명령을 통해 참조 이미지를 효과적으로 재현할 수 있도록 합니다.

- **Technical Details**: FreeEdit는 다중 모달 instruction encoder를 활용하여 언어 명령을 인코딩하고, 이를 기반으로 편집 과정을 안내합니다. Decoupled Residual ReferAttention (DRRA) 모듈을 통해 참조 이미지에서 추출한 세밀한 특성을 이미지 편집 과정에 효과적으로 통합합니다. 또한, FreeBench라는 고품질 데이터셋을 구축하였으며, 이는 원본 이미지, 편집 후 이미지 및 참조 이미지를 포함하여 다양한 편집 태스크를 지원합니다.

- **Performance Highlights**: FreeEdit는 객체 추가 및 교체와 같은 참고 기반 이미지 편집 작업에서 기존 방법보다 우수한 성능을 보이며, 수동 편집 마스크를 필요로 하지 않아 사용의 편리함을 크게 향상시킵니다. 실험을 통해 얻은 결과는 FreeEdit가 고품질 제로샷(zero-shot) 편집을 달성하고, 편리한 언어 명령으로 사용자 요구를 충족할 수 있음을 보여줍니다.



### LightAvatar: Efficient Head Avatar as Dynamic Neural Light Field (https://arxiv.org/abs/2409.18057)
Comments:
          Appear in ECCV'24 CADL Workshop. Code: this https URL

- **What's New**: 본 논문에서는 Neural Light Fields (NeLFs)를 기반으로 한 최초의 헤드 아바타 모델인 LightAvatar를 제안합니다. 이 모델은 메쉬(mesh)나 볼륨 렌더링(volume rendering)을 사용하지 않으면서도 고품질 이미지를 효율적으로 렌더링할 수 있습니다.

- **Technical Details**: LightAvatar는 3DMM 파라미터와 카메라 포즈를 입력으로 받아 단일 네트워크의 사전 통과(forward pass)를 통해 이미지를 렌더링합니다. 이를 통해 NeRF의 수백 번의 네트워크 통과를 줄여 렌더링 속도를 크게 향상시켰습니다. 또한, 지식 증류(distillation)를 활용하여 학습 안정성을 높이고, 실 데이터에서의 적합 오류를 보정하기 위한 워핑 필드 네트워크를 도입하였습니다.

- **Performance Highlights**: LightAvatar는 상업용 GPU (RTX3090)에서 512x512 해상도로 174.1 FPS의 성능을 보여주며, 기존 NeRF 기반 아바타들보다 훨씬 빠른 속도로 고품질 이미지를 생성할 수 있습니다.



### Visual Data Diagnosis and Debiasing with Concept Graphs (https://arxiv.org/abs/2409.18055)
- **What's New**: CONBIAS는 비주얼 데이터셋의 Concept co-occurrence Biases를 진단하고 완화하기 위해 개발된 새로운 프레임워크입니다. 이는 비주얼 데이터셋을 지식 그래프(knowledge graph)로 표현하여 편향된 개념의 동시 발생을 분석하고 이를 통해 데이터셋의 불균형을 파악합니다.

- **Technical Details**: CONBIAS 프레임워크는 세 가지 주요 단계로 이루어져 있습니다: (1) Concept Graph Construction: 데이터셋에서 개념의 지식 그래프를 구축합니다. (2) Concept Diagnosis: 생성된 지식 그래프를 분석하여 개념 불균형을 진단합니다. (3) Concept Debiasing: 그래프 클리크(clique)를 사용해 불균형한 개념 조합을 샘플링하고 이에 대한 이미지를 생성하여 데이터셋을 보완합니다. 이 과정에서 대규모 언어 모델의 의존성을 줄였습니다.

- **Performance Highlights**: CONBIAS를 기반으로 한 데이터 증강(data augmentation) 방법이 여러 데이터셋에서 일반화 성능을 향상시키는 데 성공적임을 보여줍니다. 기존의 최첨단 방법들과 비교했을 때, 균형 잡힌 개념 분포에 기반한 데이터 증강이 분류기의 전반적인 성능을 개선시킴을 실험을 통해 입증하였습니다.



### Revisit Anything: Visual Place Recognition via Image Segment Retrieva (https://arxiv.org/abs/2409.18049)
Comments:
          Presented at ECCV 2024; Includes supplementary; 29 pages; 8 figures

- **What's New**: 이번 연구에서는 Embodied agents가 시각적으로 장소를 인식하고 이동하는 데 있어 중요한 문제를 다루었습니다. 전체 이미지를 사용하는 기존 방법 대신, 이미지의 '세그먼트'를 인코딩하고 검색하는 새로운 접근 방식을 제안합니다. 이를 통해 SuperSegment라는 새로운 이미지 표현을 생성하여 장소 인식을 향상시킵니다.

- **Technical Details**: 제안된 SegVLAD는 Open-set 이미지 분할(open-set image segmentation)을 통해 이미지를 의미있는 요소(entities)로 분해합니다. 각 아이템은 SuperSegments로 연결되어 구조화됩니다. 새로 제안된 Feature aggregation 방법을 사용하여 이 SuperSegments를 효율적으로 컴팩트한 벡터 표현으로 인코딩합니다. SegVLAD는 다양한 벤치마크 데이터셋에서 기존의 방법보다 높은 인식 리콜을 기록했습니다.

- **Performance Highlights**: SegVLAD는 다양한 VPR 벤치마크 데이터셋에서 최첨단 성능을 달성했습니다. IOU 기반 필터링을 통해 중복성을 줄이고 스토리지를 절약하며, 전체 이미지 기반 검색보다 더욱 뛰어난 성능을 보입니다. 연구 결과, SegVLAD는 이미지 인코더의 특정 작업에 관계없이 적용 가능하고, 객체 인스턴스 검색(object instance retrieval) 과제를 평가하여 '무언가를 재방문(revisit anything)'할 수 있는 잠재력을 보여주었습니다.



### IFCap: Image-like Retrieval and Frequency-based Entity Filtering for Zero-shot Captioning (https://arxiv.org/abs/2409.18046)
Comments:
          Accepted to EMNLP 2024

- **What's New**: 최근 이미지 캡셔닝(image captioning) 분야에서 이미지-텍스트 데이터 쌍의 한계를 극복하기 위해 텍스트-전용(training) 교육 방법이 탐색되고 있습니다. 본 논문에서는 텍스트 데이터와 이미지 데이터 간의 모달리티 차이를 완화하기 위한 새로운 접근 방식으로 'Image-like Retrieval'을 제안합니다.

- **Technical Details**: 제안된 방법인 IFCap($\textbf{I}$mage-like Retrieval과 $\textbf{F}$requency-based Entity Filtering for Zero-shot $\textbf{Cap}$tioning)는 효율적인 이미지 캡셔닝을 위한 통합 프레임워크로, Fusion Module을 통해 검색된 캡션과 입력 특성을 통합하여 캡션 품질을 향상시킵니다. 또한 Frequency-based Entity Filtering 기술을 도입하여 더 나은 캡션 품질을 제공합니다.

- **Performance Highlights**: 광범위한 실험 결과, IFCap은 기존의 텍스트-전용 훈련 기반 제로샷 캡셔닝(zero-shot captioning) 방식에 비해 이미지 캡셔닝과 비디오 캡셔닝 모두에서 state-of-the-art 성능을 기록하는 것으로 나타났습니다.



### EMOVA: Empowering Language Models to See, Hear and Speak with Vivid Emotions (https://arxiv.org/abs/2409.18042)
Comments:
          Project Page: this https URL

- **What's New**: EMOVA(EMotionally Omni-present Voice Assistant)는 음성 대화에서 다양한 감정과 톤을 지원하며, 최첨단 비전-언어(vision-language) 및 음성(speech) 능력을 결합한 최초의 옴니모달(omni-modal) LLM입니다. 이 모델은 기존 비전-언어 및 음성-언어 모델의 한계를 극복하고, 실시간 대화의 요구를 충족합니다.

- **Technical Details**: EMOVA는 연속 비전 인코더와 의미-음향 분리된 음성 토크나이저를 사용하여 음성 이해 및 생성을 위한 엔드투엔드(end-to-end) 아키텍처를 갖추고 있습니다. 이를 통해 입력 음성의 의미적 내용과 음향적 스타일을 분리하여 다양한 음성 스타일 조절을 지원합니다.

- **Performance Highlights**: EMOVA는 첫 번째로 비전-언어 및 음성 벤치마크에서 최첨단 성능을 달성하였으며, 공공 데이터셋을 활용하여 옴니모달 정렬을 효율적으로 수행하고, 생생한 감정을 담은 음성 대화를 지원합니다.



### ReliOcc: Towards Reliable Semantic Occupancy Prediction via Uncertainty Learning (https://arxiv.org/abs/2409.18026)
Comments:
          Technical report. Work in progress

- **What's New**: 본 연구에서는 카메라 기반의 시맨틱 점유 예측(semantic occupancy prediction) 모델의 신뢰성(reliability)을 평가하고, 이를 개선하기 위한 새로운 방법인 ReliOcc를 제안합니다. 지금까지 LiDAR(라이다)와 비교하여 카메라 모델의 정확도는 개선되었지만, 신뢰성 측면에서는 여전히 큰 격차가 존재합니다.

- **Technical Details**: ReliOcc는 기존의 모델에 플러그 앤 플레이(plug-and-play) 방식으로 통합될 수 있는 방법으로, 개별 복셀(voxel)의 하이브리드 불확실성(hybrid uncertainty)과 샘플링 기반의 노이즈를 혼합 학습(mix-up learning)을 통해 결합합니다. 또한, 오프라인 모드에서 모델 신뢰성을 강화하기 위한 불확실성 인식 보정 전략(uncertainty-aware calibration strategy)을 제공합니다.

- **Performance Highlights**: 다양한 실험 설정에서 ReliOcc는 모델의 신뢰성을 상당히 향상시키면서 정의 기하학 및 시맨틱 예측의 정확성을 유지하는 데 성공하였습니다. 특히, 제안된 방법은 센서 고장 및 도메인 외 노이즈(out of domain noises)에도 강인성을 보였습니다.



### Transferring disentangled representations: bridging the gap between synthetic and real images (https://arxiv.org/abs/2409.18017)
- **What's New**: 이 연구는 합성 데이터(synthetic data)를 사용하여 실제 데이터에 적용 가능한 일반 목적의 분리 표현(disentangled representation)을 학습할 가능성을 탐구합니다. 이 과정에서 성능을 향상시키기 위한 미세 조정(fine-tuning)의 효과와 전이(transfer) 후 보존되는 분리 특성에 대해 논의합니다.

- **Technical Details**: 본 연구에서는 OMES (Overlap Multiple Encoding Scores)라는 새로운 간섭 기반(intervention-based) 지표를 제안하여, 표현에서 인코딩된 요소들의 품질을 측정합니다. 이는 기존의 분류기(classifier) 의존적인 방법들과는 달리, 분류기 없는(intervention-based) 접근을 채택하여 요소의 분포를 분석하며, 데이터를 쌍으로 매칭하여 단일 요소만 다르게 하여 평가합니다.

- **Performance Highlights**: 연구 결과, 합성 데이터에서 학습된 표현을 실제 데이터로 전이할 수 있는 가능성이 있으며, 일부 분리 수준이 효과적임을 보여줍니다. 특히, 다양한 (Source, Target) 쌍에 대한 실험을 통해 학습된 DR의 표현력이 잘 평가됨을 확인하였습니다.



### InterNet: Unsupervised Cross-modal Homography Estimation Based on Interleaved Modality Transfer and Self-supervised Homography Prediction (https://arxiv.org/abs/2409.17993)
- **What's New**: 본 논문에서는 InterNet이라는 새로운 비지도 방식의 교차 모달 호모그래피(모드의 전환) 추정 프레임워크를 제안합니다. 이 방법은 모달리티 전이(modality transfer)와 자기 감독(self-supervised) 호모그래피 예측을 기반으로 하며, 혁신적인 상호 최적화(interleaved optimization) 프레임워크를 도입하여 두 구성 요소를 순차적으로 개선하는 방식을 사용합니다.

- **Technical Details**: InterNet은 서로 다른 모달리티의 이미지 간의 호모그래피를 추정하기 위해 설계되었습니다. 이 네트워크는 모달리티 전이 모듈과 호모그래피 추정 모듈로 구성되어 있으며, 두 모듈을 교차로 훈련하여 모달리티 간의 격차를 줄이는 방식으로 동작합니다. 또한, 정밀 호모그래피 특징 손실(fine-grained homography feature loss)을 도입하여 두 모듈 간의 상호작용을 강화하고, 간단하지만 효과적인 증류 훈련(distillation training) 기법을 적용해 모델 파라미터를 줄이고 교차 도메인 일반화 능력을 향상시킵니다.

- **Performance Highlights**: 실험 결과, InterNet은 다양한 데이터셋에서 최신 비지도 방식 중에서 최상의 성능을 달성하며, MHN과 LocalTrans와 같은 많은 지도 방식보다 더 높은 성능을 기록했습니다. InterNet은 GoogleMap과 WHU-OPT-SAR 데이터셋에서 각각 MHN보다 54.3% 및 47.4%, LocalTrans보다 61.8% 및 85.8% 낮은 평균 코너 오차(MACEs)를 달성하였습니다.



### Deblur e-NeRF: NeRF from Motion-Blurred Events under High-speed or Low-light Conditions (https://arxiv.org/abs/2409.17988)
Comments:
          Accepted to ECCV 2024. Project website is accessible at this https URL. arXiv admin note: text overlap with arXiv:2006.07722 by other authors

- **What's New**: 본 연구에서는 이벤트 카메라의 Motion Blur를 효과적으로 처리할 수 있는 새로운 방법인 Deblur e-NeRF를 제안합니다. 이 방법은 고속 또는 저조도 조건에서 생성된 Motion-blurred 이벤트를 기반으로 최소 블러의 Neural Radiance Fields (NeRF)를 직접 복원하는 것을 목표로 합니다.

- **Technical Details**: Deblur e-NeRF의 핵심 구성 요소는 임의의 속도와 조명 조건에서 이벤트 Motion Blur를 고려하는 물리적으로 정확한 픽셀 대역폭 모델입니다. 이 모델은 이벤트 생성 모델의 일부로 통합되어 블러가 최소화된 NeRF 복원을 지원하며, 새로운 임계값 정규화된 총 변동 손실(threshold-normalized total variation loss)을 도입하여 큰 텍스처 없는 패치를 더 잘 정규화합니다.

- **Performance Highlights**: 실제로 검증된 실험과 현실적으로 시뮬레이션된 시퀀스를 통해 Deblur e-NeRF의 효과성을 확인하였습니다. 더불어, 연구에 사용된 코드, 이벤트 시뮬레이터 및 합성 이벤트 데이터 세트는 오픈 소스로 제공됩니다.



### LLM4Brain: Training a Large Language Model for Brain Video Understanding (https://arxiv.org/abs/2409.17987)
Comments:
          ECCV2024 Workshop

- **What's New**: 본 연구에서는 LLM(대형 언어 모델)을 기반으로 한 접근 방식으로, 비디오 자극으로 유발된 fMRI(기능적 MRI) 신호로부터 시각-의미 정보를 재구성하는 방법을 소개합니다. 특히, 우리는 fMRI 인코더에 적응기를 장착하여 뇌 반응을 비디오 자극과 정렬된 잠재 표현으로 변환합니다.

- **Technical Details**: 제안된 방법은 두 단계의 훈련 프로세스를 사용합니다: Stage I은 fMRI와 비디오 데이터 간의 교차 모드 정렬을 배우고, Stage II는 감독된 지침 파인튜닝입니다. 첫 번째 단계에서는 CLIP 손실을 사용하여 fMRI 및 비디오 임베딩 간의 정렬을 학습합니다.

- **Performance Highlights**: 우리는 제안된 방법이 다양한 정량적 의미 메트릭을 사용하여 우수한 결과를 달성했으며, 진실 정보와의 유사성을 나타낸다는 것을 강조합니다. 또한, 우리의 방법은 다양한 자극 및 개인에 대해 좋은 일반화 능력을 보여줍니다.



### BlinkTrack: Feature Tracking over 100 FPS via Events and Images (https://arxiv.org/abs/2409.17981)
- **What's New**: BlinkTrack는 이벤트 데이터와 RGB 이미지를 통합하여 고주파수 피쳐 트래킹을 위한 새로운 프레임워크입니다. 이 방법은 Kalman 필터를 학습 기반 프레임워크로 확장하여 동기화되지 않은 잠재적 오류를 해결합니다.

- **Technical Details**: BlinkTrack은 이벤트 모듈과 이미지 모듈로 구성되어 있으며, 학습 가능 Kalman 필터를 사용합니다. 이 구조는 기존의 단일 모달리티 트래커를 개선하고 모달리티 간의 비동기 데이터 융합을 지원합니다.

- **Performance Highlights**: BlinkTrack은 기존 이벤트 기반 방법보다 성능이 크게 향상되었으며, 이벤트 데이터가 전처리된 경우 100 FPS 이상, 다중 모달리티 데이터가 포함될 경우 80 FPS 이상으로 작동합니다.



### HydraViT: Stacking Heads for a Scalable V (https://arxiv.org/abs/2409.17978)
- **What's New**: 새로운 HydraViT 접근 방식은 다중 크기의 Vision Transformers (ViTs) 모델을 학습하고 저장하는 필요성을 없애면서, 스케일러블한 ViT를 가능하게 합니다.

- **Technical Details**: HydraViT는 Multi-head Attention (MHA) 메커니즘을 기반으로 하여, 다양한 하드웨어 환경에 적응할 수 있도록 임베딩 차원과 MHA의 헤드 수를 동적으로 조정합니다. 이 방식은 최대 10개의 서브 네트워크를 생성할 수 있습니다.

- **Performance Highlights**: HydraViT는 ImageNet-1K 데이터셋에서 동일한 GMACs와 처리량을 기준으로, 기존 모델 대비 최대 7 p.p. 높은 정확성을 달성하며, 이는 다양한 하드웨어 환경에서 특히 유용합니다.



### Cross-Modality Attack Boosted by Gradient-Evolutionary Multiform Optimization (https://arxiv.org/abs/2409.17977)
- **What's New**: 최근의 적대적 공격(adversarial attack) 연구에서, 중첩된 이미지 모달리티(infrared, thermal, RGB 이미지 등) 간에 적대적 공격의 전이성을 다룬 연구가 부족했던 부분을 보완하는 새로운 'Multiform Attack' 전략을 제안합니다.

- **Technical Details**: 제안하는 듀얼 레이어 최적화 프레임워크는 'gradient-evolution'을 기반으로 하여 다양한 모달리티 간의 효과적인 노이즈 전이를 촉진합니다. 첫 번째 층은 각 모달리티 내에서 범용적인 방해를 생성하며, 두 번째 최적화에서는 진화 알고리즘을 사용해 서로 다른 모달리티 간의 공유 방해를 찾습니다. 이러한 방법을 통해 우리는 협력적이고 포괄적인 보안 접근 방식을 실현합니다.

- **Performance Highlights**: 다양한 이질적 데이터셋에서의 광범위한 테스트를 통해 'Multiform Attack'이 기존의 기술들보다 월등한 성능과 강인성을 보인 것을 입증했습니다. 이는 교차 모달 적대적 공격의 전이성을 개선하여 복잡한 다중 모달 시스템의 보안 취약성을 이해하는 데 새로운 시각을 제공합니다.



### CNCA: Toward Customizable and Natural Generation of Adversarial Camouflage for Vehicle Detectors (https://arxiv.org/abs/2409.17963)
- **What's New**: 본 연구는 사용자 정의 가능한 자연스러운 적대적 위장(Adversarial Camouflage) 생성을 위해 사전 훈련된 diffusion 모델을 활용하는 새로운 접근 방식을 제안합니다. 이를 통해 이전의 고정관념과 달리 어렵고 신경망이 쉽게 식별할 수 있었던 적대적 위장으로부터 벗어날 수 있습니다.

- **Technical Details**: 이 방법은 사용자가 제공하는 텍스트 프롬프트를 기반으로 최적의 텍스쳐 이미지를 생성하고, 적대적 특성과 원래의 텍스트 프롬프트 기능을 결합하여 diffusion 모델의 조건부 입력을 형성합니다. 이 과정에서 클리핑 전략을 도입하여 자연스러움과 공격 성능 간의 균형을 유지합니다.

- **Performance Highlights**: 우리는 광범위한 실험을 통해 제안된 방법이 기존 최첨단 기법들보다 자연스럽고 커스터마이즈 가능한 위장을 생성할 수 있고, 공격 성능에서도 경쟁력을 가진다는 것을 입증하였습니다. 코드도 공개되었습니다.



### Spatial Hierarchy and Temporal Attention Guided Cross Masking for Self-supervised Skeleton-based Action Recognition (https://arxiv.org/abs/2409.17951)
Comments:
          12 pages,6 figures,IEEE Trans

- **What's New**: 본 논문에서는 스켈레톤 기반 행동 인식을 위한 새로운 프레임워크인 계층적 및 어텐션 기반 교차 마스킹 프레임워크(HA-CM)를 제안합니다. HA-CM은 시간적 및 공간적 관점에서 스켈레톤 시퀀스에 마스킹을 적용하여 단일 마스킹 방법의 고유한 편향을 완화합니다.

- **Technical Details**: HA-CM은 하이퍼볼릭 공간을 활용하여 고차원 스켈레톤의 계층 구조를 유지하고, 참여 조인트의 하위 계층을 마스킹 기준으로 사용합니다. 시간 흐름에서는 조인트의 글로벌 어텐션을 활용하여 마스킹 기법을 개선하고, 교차 대비 손실을 손실 함수에 도입해 인스턴스 수준의 특징 학습을 향상시킵니다.

- **Performance Highlights**: HA-CM은 세 개의 공개 대규모 데이터세트인 NTU-60, NTU-120, PKU-MMD에서 효율성과 보편성을 입증했습니다.



### Perturb, Attend, Detect and Localize (PADL): Robust Proactive Image Defens (https://arxiv.org/abs/2409.17941)
- **What's New**: 본 논문에서는 PADL(Proactive Attack Detection and Localization)이라는 혁신적인 솔루션을 제안하여 이미지 조작 탐지 및 위치 지정을 위한 새로운 접근 방식을 구현하였습니다. 이 방법은 이미지에 대한 특정한 방해 요소를 생성하는 대칭 인코딩 및 디코딩 스킴을 기반으로 하여, 기존 방법들의 한계를 극복합니다.

- **Technical Details**: PADL은 transformer 아키텍처의 cross-attention 메커니즘을 활용해, 이미지에 특화된 방해 요소를 생성하는데, 이 과정은 코드화 및 복호화 모듈로 구성된 대칭적 구조에서 이루어집니다.  또한 새로운 손실 함수를 통해 방해 요소의 다양성을 유지하여 성능을 향상시킵니다.

- **Performance Highlights**: PADL은 다양한 이미지 생성 모델에 대해 우수한 일반화 성능을 보이며, StarGANv2, BlendGAN, DiffAE 등 여러 미지의 모델에 적용 가능함을 입증합니다. 또한 새로운 평가 프로토콜을 통해 탐지 정확도에 따라 위치 지정 성능을 공정하게 평가하여 실제 시나리오를 더 잘 반영합니다.



### Neural Light Spheres for Implicit Image Stitching and View Synthesis (https://arxiv.org/abs/2409.17924)
Comments:
          Project site: this https URL

- **What's New**: 이번 연구에서는 네트워크 기반의 구형 신경 광선 필드 모델을 통해 파노라마 이미지를 효과적으로 스티칭(stitching)하고 재렌더링하는 방법을 제안합니다. 이 모델은 카메라 경로와 고해상도 장면 재구성을 동시에 추정하여, 다양한 경로로 촬영된 파노라마 비디오를 처리할 수 있습니다.

- **Technical Details**: 연구에서 제안한 모델은 깊이 시차(depth parallax), 시점 의존 조명(view-dependent lighting), 지역 장면의 움직임 및 색상 변화를 처리할 수 있는 구성 요소로 나뉘어 있으며, 모델 크기는 장면당 80MB에 1080p 해상도로 50 FPS의 실시간 렌더링을 지원합니다. 추가로, Android 기반의 데이터 수집 도구를 제공하여 카메라 및 시스템 메타데이터와 함께 RAW 이미지 배열을 녹화할 수 있습니다.

- **Performance Highlights**: 전통적인 이미지 스티칭 및 방사선 필드(radiance field) 방법과 비교하여 향상된 재구성 품질을 입증하였으며, 장면의 움직임 및 비이상적인 촬영 설정에 대해 훨씬 높은 내성을 보여줍니다. 50개의 실내 및 실외 핸드헬드 파노라마 장면이 포함된 데이터셋도 공개되었습니다.



### Resolving Multi-Condition Confusion for Finetuning-Free Personalized Image Generation (https://arxiv.org/abs/2409.17920)
- **What's New**: 이번 논문은 여러 개의 참조 이미지를 사용할 때 발생하는 객체 혼동 문제를 해결하기 위해 새로운 가중 병합(weighted-merge) 방법을 제안합니다. 이 방법은 diffusion model에서 잠재 이미지 특성과 목표 객체와의 관련성을 조사하여 참조 이미지 특성을 적절하게 통합합니다.

- **Technical Details**: 제안된 가중 병합 방법은 참조 이미지의 특성을 해당 객체에 맞게 병합하며, 이는 각각의 잠재 이미지 특성의 위치에 따라 달라지는 가중치를 이용합니다. 또한, 여러 객체를 포함한 데이터셋에서 기존의 사전 훈련된 모델에 통합되며, 노이즈 추가 실험을 통해 객체 관련성 예측의 유효성이 검증되었습니다.

- **Performance Highlights**: 이 방법은 11백만 개의 이미지로 구성된 SA-1B 데이터셋을 활용하여 모델을 훈련한 후, Concept101 및 DreamBooth 데이터셋에서 다중 객체 개인화 이미지 생성의 최첨단 성능을 달성했습니다. 또한 단일 객체 개인화 이미지 생성에서도 성능이 크게 향상되었습니다.



### WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians (https://arxiv.org/abs/2409.17917)
- **What's New**: 이번 논문에서는 3D 장면에서의 스타일 전이에 대한 새로운 접근 방식을 제시합니다. 기존의 2D 이미지 스타일 전이 기술에 비해 3D 장면에서의 기하학을 보다 정확하게 복제할 수 있는 방법을 모색하였습니다.

- **Technical Details**: 연구팀은 명시적인 Gaussian Splatting (GS) 표현을 활용하여 스타일 장면과 콘텐츠 장면 간의 Gaussian 분포를 Earth Mover's Distance (EMD)를 사용하여 직접 일치시킵니다. 이를 통해 공간의 부드러움을 유지할 수 있도록 엔트로피 정규화 Wasserstein-2 거리도 도입하였습니다. 그리고 장면 스타일화 문제를 더 작은 단위로 분해하여 효율성을 높였습니다.

- **Performance Highlights**: 제안한 WaSt-3D 방법은 스타일 장면의 세부 정보를 콘텐츠 장면에 충실하게 전이함으로써 고해상도 3D 스타일화를 가능하게 합니다. 이 방법은 다양한 콘텐츠 및 스타일 장면에서도 일관되게 성능을 발휘하며, 어떠한 학습 없이 최적화 기반 기법만 사용하여 결과를 도출합니다.



### LKA-ReID:Vehicle Re-Identification with Large Kernel Attention (https://arxiv.org/abs/2409.17908)
Comments:
          The paper is under consideration at 2025 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2025)

- **What's New**: 지능형 교통 시스템과 스마트 시티 인프라의 빠른 발전에 따라 차량 재식별 기술(vehicle Re-ID)이 중요한 연구 분야로 부상하고 있습니다. 이 논문에서는 큰 커널 주의 메커니즘(large kernel attention, LKA)을 활용한 LKA-ReID를 제안하며, 이로써 차량의 글로벌 및 로컬 피처를 보다 포괄적으로 추출하고 inter-class 간 유사성을 해결하고자 합니다.

- **Technical Details**: LKA-ReID는 네 개의 브랜치로 구성된 네트워크를 사용하여 각 브랜치가 2048 차원의 피처를 생성합니다. LKA 모듈을 통해 차량의 글로벌 및 로컬 특성을 추출하고, HCA(hybrid channel attention) 모듈을 사용하여 채널과 공간 정보를 결합하여 배경 및 방해 정보를 제거하고 중요한 특성을 강조합니다.

- **Performance Highlights**: VeRi-776 데이터셋에서 LKA-ReID의 효과를 실험했으며, mAP(micro Average Precision)는 86.65%, Rank-1은 98.03%에 도달하여 경쟁력 있는 성능을 입증했습니다.



### Self-supervised Monocular Depth Estimation with Large Kernel Attention (https://arxiv.org/abs/2409.17895)
Comments:
          The paper is under consideration at 2025 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2025)

- **What's New**: 이 논문에서는 Self-Supervised Monocular Depth Estimation을 위한 새로운 네트워크를 제안합니다. 특히, 큰 커널 어텐션(large kernel attention)을 기반으로 한 디코더를 통해 깊이 추정에서의 성능을 개선합니다.

- **Technical Details**: 제안한 네트워크는 인코더-디코더 아키텍처를 채택하며, 인코더에서는 HRNet18을 사용하여 다중 스케일 특성을 유지합니다. 디코더는 LKA(large kernel attention)와 업샘플링 모듈을 통해 관측된 이미지를 세밀하게 복원합니다. 이 과정에서 2D 이미지 구조를 손상시키지 않으면서도 채널 적응성을 유지합니다.

- **Performance Highlights**: KITTI 데이터셋의 실험에서 AbsRel = 0.095, SqRel = 0.620, RMSE = 4.148, RMSElog = 0.169, δ<1.25 = 90.7의 성능을 달성했습니다.



### Upper-Body Pose-based Gaze Estimation for Privacy-Preserving 3D Gaze Target Detection (https://arxiv.org/abs/2409.17886)
Comments:
          Accepted in the T-CAP workshop at ECCV 2024

- **What's New**: 이 논문은 사람의 상체 자세와 깊이 지도(depth map)를 활용하여 3D 시선 방향을 추출하고, 이를 통해 주목하는 대상을 예측하는 새로운 접근 방식을 제시합니다. 또한, 얼굴 이미지를 필요로 하지 않고도 주목 대상을 탐지하는 방법을 통해 프라이버시를 보호할 수 있습니다.

- **Technical Details**: 논문에서 제안하는 방법은 상체 스켈레톤과 깊이 지도를 사용하여 3D 시선 추정 및 주목 대상을 탐지하는 다중 단계 또는 엔드-투-엔드 파이프라인을 활용합니다. 각 단계에서 상체 자세 피처 및 장면 깊이 지도에서 추출한 컨볼루션 피처를 조합하여 실시간으로 높은 정확도의 3D 시선 벡터를 생성합니다.

- **Performance Highlights**: 제안된 방법은 GFIE 데이터셋에서 최첨단 결과를 달성하여 3D 시선 목표 탐지 분야에서의 새로운 기준을 마련했습니다. 이 방법은 개인의 얼굴 정보를 사용하지 않고도 정확한 시선 추정과 주목 대상 탐지가 가능하여 특히 개인의 프라이버시를 유지하는 측면에서 중요한 혁신을 제공합니다.



### Self-Distilled Depth Refinement with Noisy Poisson Fusion (https://arxiv.org/abs/2409.17880)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 본 연구는 Depth Refinement(심도 정제)의 새로운 프레임워크인 Self-distilled Depth Refinement (SDDR)을 제안합니다. 이 모델은 저해상도 심도 예측 결과를 고해상도로 변환하며, 모호한 에지(boundaries)와 잡음을 줄이고자 합니다.

- **Technical Details**: SDDR은 노이즈가 있는 Poisson fusion 문제로 심도 정제를 모델링합니다. 이는 두 가지 노이즈: local inconsistency noise와 edge deformation noise를 고려하여 심도 예측 오류를 분리합니다. SDDR의 주요 구성 요소는 depth edge representation과 edge-based guidance입니다. SDDR은 coarse-to-fine self-distillation을 통해 pseudo-labels로서 low-noise depth edge representations를 생성합니다.

- **Performance Highlights**: SDDR은 Middlebury2021, Multiscopic, Hypersim 및 자연 장면에서의 DIML와 DIODE 데이터셋에서 모두 우수한 성능을 달성했습니다. 이전 방법들보다 심도 정확도와 에지 품질이 현저히 개선되었으며 efficiency(효율성) 또한 높아졌습니다. 또한 SDDR이 생성한 정확한 depth edge labels는 다른 모델 학습에 활용되어 성능을 향상시킬 수 있음을 보여주었습니다.



### A New Dataset for Monocular Depth Estimation Under Viewpoint Shifts (https://arxiv.org/abs/2409.17851)
Comments:
          17 pages, 5 figures. Accepted at ECCV 2024 2nd Workshop on Vision-Centric Autonomous Driving (VCAD)

- **What's New**: 본 논문에서는 모노큘라 깊이 추정(Depth Estimation)과 관련하여 카메라 위치와 방향이 모델에 미치는 영향을 정량화할 수 있는 새로운 데이터셋과 평가 방법론을 소개합니다. 이 접근법은 고가의 lidar 센서를 사용할 필요 없이 객체 감지 및 호모그래피 추정을 기반으로 한 새로운 'ground truth' 전략을 제안합니다.

- **Technical Details**: 제안된 방법은 호모그래피( Homography)와 객체 감지를 결합하여 도로 장면에서 모노큘라 깊이 추정의 오류를 측정하는 'ground truth' 소스로 사용합니다. 이 연구는 여러 알려진 카메라 설정에서 기록된 다양한 비디오 시퀀스를 수집하고 이를 사용하여 모던 깊이 추정 모델의 다양한 뷰포인트 변화에 대한 강건성을 평가합니다. 데이터셋은 kaggle.com/datasets/aurelpjetri/viewpointdepth에서 액세스할 수 있습니다.

- **Performance Highlights**: 본 연구의 결과는 3D 장면 이해를 위한 깊이 추정이 다양한 뷰포인트에서 어떻게 영향을 받는지를 정량화합니다. 실험 결과, 기존 lidar 'ground truth'를 사용한 평가 결과와 유사한 결과를 보여 대안적 접근법의 유효성을 입증합니다.



### Unsupervised Learning Based Multi-Scale Exposure Fusion (https://arxiv.org/abs/2409.17830)
Comments:
          11 pages

- **What's New**: 이 논문은 비지도 학습 기반의 다중 스케일 노출 융합 알고리즘(ULMEF)를 제안합니다. 기존의 알고리즘과는 달리, 이미지 세트를 융합하는 것뿐만 아니라, 같은 HDR 장면에서 다른 노출의 이미지를 이용하여 새로운 손실 함수(loss functions)를 정의합니다.

- **Technical Details**: ULMEF는 비지도 학습 방법을 사용하며, 다중 스케일 주의 모듈(multi-scale attention module)을 포함하여 장면 깊이와 지역 대비(local contrast)를 효과적으로 보존합니다. 새로운 손실 함수는 융합할 이미지 집합과 비축점(이와 같이) LDR 이미지 집합을 분리하여 정의하여, 훈련 효율을 향상시킵니다.

- **Performance Highlights**: 제안된 ULMEF 알고리즘은 다양한 데이터셋에서 실험 결과 다른 최신의 노출 융합 알고리즘들보다 우수한 성능을 보였습니다. 이는 이미지 융합 품질을 현저히 향상시키며, 기존 알고리즘들이 남기는 밝기 질서 역전 아티팩트(brightness order reversal artifacts)를 방지합니다.



### Kendall's $\tau$ Coefficient for Logits Distillation (https://arxiv.org/abs/2409.17823)
- **What's New**: 본 논문에서는 KL divergence의 제약으로 인한 학생 모델의 비효율적 최적화를 해결하기 위해 Rank-Kendall Knowledge Distillation(RKKD)라는 새로운 플러그 앤 플레이(ranking loss) 기법을 제안합니다. RKKD는 학생의 로짓(logits)에서 채널 값의 순서를 제약하여 작은 값 채널들에도 집중할 수 있도록 합니다.

- **Technical Details**: RKKD는 Kendall의 τ𝜏	au 계수를 기반으로 하여 학생 모델의 로짓 내에서 값의 순서를 제약하는 랭크(loss) 기능을 제공합니다. 이 랭크 제약은 상위 채널의 올바른 클래스를 최상위 순위로 구속하여 KL divergence와 작업 손실(task loss)의 최적화 방향을 일치시키는 데 도움을 줍니다. 저자는 다양한 차별적 형태의 Kendall 계수를 탐구합니다.

- **Performance Highlights**: CIFAR-100 및 ImageNet 데이터셋에서 광범위한 실험이 수행되었으며, RKKD는 다양한 지식 증류(baseline) 설정에서 성능을 향상시켰습니다. RKKD는 기존 지식 증류 방법에 플러그 앤 플레이 방식으로 추가되어 다양한 교사-학생 아키텍처 조합에서 전반적인 개선을 보여 주었습니다.



### Cascade Prompt Learning for Vision-Language Model Adaptation (https://arxiv.org/abs/2409.17805)
Comments:
          ECCV2024

- **What's New**: 이번 연구에서는 Vision-Language Models (VLMs)인 CLIP의 성능 향상을 위한 Cascade Prompt Learning (CasPL) 프레임워크를 제안합니다. 이 프레임워크는 일반적인 지식과 특정 지식을 동시에 캡처할 수 있는 새로운 학습 패러다임을 제공합니다.

- **Technical Details**: CasPL은 두 가지의 학습 가능한 프롬프트로 구성됩니다. 첫 번째는 도메인 일반 지식을 추출하는 'boosting prompt'이며, 두 번째는 다운스트림 태스크를 미세 조정하는 'adapting prompt'입니다. 이 두 단계는 서로 연결되어 단계적으로 최적화됩니다. CasPL은 기존의 다른 프롬프트 학습 방법들과 쉽게 통합할 수 있는 플러그 앤 플레이 모듈입니다.

- **Performance Highlights**: CasPL은 PromptSRC 방법과 비교했을 때, 기본 클래스(Base Classes)에서 평균 1.85%, 새로운 클래스(Novel Classes)에서 3.44%, 그리고 조화 평균(Harmonic Mean)에서 2.72%의 성능 향상을 보여주었습니다.



### Reblurring-Guided Single Image Defocus Deblurring: A Learning Framework with Misaligned Training Pairs (https://arxiv.org/abs/2409.17792)
Comments:
          The source code and dataset are available at this https URL

- **What's New**: 이번 논문은 misaligned training pairs를 사용하여 단일 이미지 defocus deblurring을 위한 reblurring-guided 학습 프레임워크를 제안합니다. 새로운 SDD 데이터셋을 수집하여 이러한 방법의 유효성을 검증하고 있습니다.

- **Technical Details**: 이 프레임워크는 baseline defocus deblurring 네트워크를 구축하고, reblurring 모듈을 사용하여 입력 흐림 이미지와 deblurred 이미지 간의 공간적 일관성을 유지합니다. 이 모듈은 isotropic blur kernels을 재구성하고 pseudo defocus blur map을 통해 training triplets를 형성합니다.

- **Performance Highlights**: 제안된 방법은 기존의 state-of-the-art 방법들과 비교하여 향상된 성능을 보이며, SDD 데이터셋을 통해 학습된 모델을 평가할 수 있습니다.



### Taming Diffusion Prior for Image Super-Resolution with Domain Shift SDEs (https://arxiv.org/abs/2409.17778)
Comments:
          This paper is accepted by NeurIPS 2024

- **What's New**: 본 연구에서는 기존의 Diffusion 모델을 활용하여 이미지 초해상도(SR) 문제를 해결하는 DoSSR 모델을 제안하며, 저해상도(Low-Resolution, LR) 이미지를 시작점으로 하는 접근 방식을 통해 효율성을 크게 향상시켰습니다.

- **Technical Details**: DoSSR 모델은 도메인 이동 방정식(domain shift equation)을 기반으로 하여 기존의 Diffusion 모델과 통합됩니다. 이 과정은 확률적 미분 방정식(Stochastic Differential Equations, SDEs)으로 연속적인 형태로 전환되어 높은 샘플링 효율성을 달성합니다.

- **Performance Highlights**: 실험 결과, DoSSR 모델은 합성 및 실제 데이터셋에서 최신 기술 대비 뛰어난 성능을 보여주었으며, 단 5회의 샘플링 단계만으로도 상대적으로 5-7배 빠른 속도를 제공합니다.



### Harnessing Shared Relations via Multimodal Mixup Contrastive Learning for Multimodal Classification (https://arxiv.org/abs/2409.17777)
Comments:
          RK and RS contributed equally to this work, 20 Pages, 8 Figures, 9 Tables

- **What's New**: 본 논문에서는 M3CoL(Multimodal Mixup Contrastive Learning) 접근 방식을 제안하여, 다양한 모달리티 간의 미묘한 공유 관계를 포착할 수 있음을 보여줍니다. M3CoL은 모달리티 간의 혼합 샘플을 정렬하여 강력한 표현을 학습하는 Mixup 기반의 대조 손실을 활용합니다.

- **Technical Details**: M3CoL은 이미지-텍스트 데이터셋(N24News, ROSMAP, BRCA, Food-101)에서 광범위한 실험을 통해 공유 모달리티 관계를 효과적으로 포착하고, 다양한 도메인에서 일반화 능력을 발휘합니다. 이는 fusion module과 unimodal prediction modules로 구성된 프레임워크에 기반하였으며, Mixup 기반의 대조 손실을 통해 보조 감독을 강화합니다.

- **Performance Highlights**: M3CoL은 N24News, ROSMAP, BRCA 데이터셋에서 최첨단 방법들을 초월하는 성능을 보이며, Food-101에서는 유사한 성능을 달성했습니다. 이를 통해 공유 관계 학습의 중요성이 강조되며, 강력한 다중 모달 학습을 위한 새로운 연구 방향을 열어갑니다.



### UNICORN: A Deep Learning Model for Integrating Multi-Stain Data in Histopathology (https://arxiv.org/abs/2409.17775)
- **What's New**: 이번 연구에서는 다채로운 염색을 이용한 조직병리학 이미지의 통합을 위해 다중 모달리티(transformer) 모델인 UNICORN을 제안하였습니다. 이 모델은 학습 및 추론 중 결측값을 처리할 수 있으며, 상이한 염색 방법을 통해 아테롬성 동맥경화증의 중증도를 예측합니다.

- **Technical Details**: UNICORN은 다단계(end-to-end) 훈련이 가능한 transformer 아키텍처로, 256x256 px 크기의 패치로 분할된 WSIs를 처리합니다. 개별적으로 처리된 도메인 전문 모듈들이 각각의 염색에서 고유한 특징을 학습하고, 집계 전문가 모듈이 다양한 염색 간의 상호작용을 학습하여 정보 통합을 수행합니다.

- **Performance Highlights**: UNICORN은 4,000개의 다중 염색 전체 슬라이드 이미지(WSIs)를 평가하여 0.67의 분류 정확도를 달성하였고, 이는 현재 최고 수준의 타 모델들을 초월하는 성능입니다. 모델은 다양한 염색 방법을 통한 조직의 특징을 효과적으로 식별하고, 질병 진행 모델링에서도 높은 성능을 나타냈습니다.



### Confidence intervals uncovered: Are we ready for real-world medical imaging AI? (https://arxiv.org/abs/2409.17763)
Comments:
          Paper accepted at MICCAI 2024 conference

- **What's New**: 이 논문은 의료 영상(segmentation) 분야에서 AI 성능 변동성을 평가되지 않는 문제를 다루고 있습니다. 2023년 MICCAI에서 발표된 논문 221편을 분석한 결과, 50% 이상의 논문이 성능 변동성을 평가하지 않고, 단 0.5%의 논문만이 신뢰 구간(confidence intervals, CIs)을 보고했습니다. 이는 기존 논문들이 임상 적용을 위한 충분한 근거를 제공하지 않음을 지적합니다.

- **Technical Details**: 연구에서는 segmentation 논문에서 보고되지 않은 표준 편차(standard deviation, SD)를 평균 Dice 유사도 계수(Dice similarity coefficient, DSC)의 2차 다항식 함수를 통해 근사할 수 있음을 보여줍니다. 이를 바탕으로 2023년 MICCAI segmentation 논문의 평균 DSC 주변에 95% CIs를 재구성하였고, 그 중간 CI 폭은 0.03으로 첫 번째와 두 번째 순위 방법 간의 중간 성능 격차보다 세 배 더 큽니다.

- **Performance Highlights**: 60% 이상의 논문에서 두 번째 순위 방법의 평균 성능이 첫 번째 순위 방법의 신뢰 구간 내에 포함되었으며, 이는 현재 보고된 성능이 실제 임상에서의 가능성을 충분히 뒷받침하지 않음을 의미합니다.



### Text Image Generation for Low-Resource Languages with Dual Translation Learning (https://arxiv.org/abs/2409.17747)
Comments:
          23 pages, 11 figures

- **What's New**: 이 연구에서는 고자원 언어에서의 실제 텍스트 이미지 스타일을 모방하여 저자원 언어의 텍스트 이미지를 생성하는 새로운 접근 방식을 제안합니다. 이를 통해 저자원 언어의 장면 텍스트 인식 성능을 향상시키고자 합니다.

- **Technical Details**: 제안된 방법은 'synthetic'(합성)과 'real'(실제)라는 이진 상태에 따라 조건화된 diffusion model(확산 모델)을 이용합니다. 이 모델은 텍스트 이미지를 합성과 실제 이미지로 변환하는 이중 번역 작업(Dual Translation Learning, DTL)에 대한 훈련을 포함하여 텍스트 인식 모델의 성능을 극대화하는 데 중요한 역할을 합니다. 또한 Fidelity-Diversity Balancing Guidance 및 Fidelity Enhancement Guidance와 같은 두 가지 지침 기술을 도입하여 생성된 텍스트 이미지의 정확성과 다양성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크를 통해 생성된 텍스트 이미지는 저자원 언어의 장면 텍스트 인식 모델의 성능을 크게 개선할 수 있음을 보여주었습니다.



### AnyLogo: Symbiotic Subject-Driven Diffusion System with Gemini Status (https://arxiv.org/abs/2409.17740)
Comments:
          13 pages, 12 figures

- **What's New**: 이번 연구에서는 AnyLogo라는 새로운 제로샷 지역 맞춤형 모델을 소개하고 있습니다. 이 모델은 복잡한 설정을 없애고, 뛰어난 세부 일관성을 가지고 있습니다.

- **Technical Details**: AnyLogo는 대칭적 확산 시스템을 기반으로 하며, 단일 디노이징 모델 내에서 엄격한 서명 추출과 창의적 콘텐츠 생성이 체계적으로 재활용됩니다. 이는 주제 전달 효율성을 향상시킵니다.

- **Performance Highlights**: AnyLogo는 약 1K의 고품질 쌍을 포함한 로고 수준 커스터마이징 벤치마크에서 실험을 수행하여 방법의 효과와 실용성을 입증했습니다.



### Neural Implicit Representation for Highly Dynamic LiDAR Mapping and Odometry (https://arxiv.org/abs/2409.17729)
- **What's New**: 이 논문은 NeRF-LOAM을 기반으로 하여 동적 객체를 포함한 외부 환경의 3D 재구성을 개선하는 새로운 방법을 제안합니다. 특히, 동적 전경과 정적 배경을 분리하여 정적 배경만으로 밀집 3D 지도를 만드는 방법론을 소개합니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 구성 요소로 나뉩니다. 첫 번째는 정적 배경과 동적 전경을 분리하는 것입니다. 동적인 요소를 매핑 과정에서 제외함으로써 밀집 3D 지도를 생성합니다. 두 번째 구성 요소는 다중 해상도 표현을 지원하기 위해 옥트리(Octree) 구조를 확장하는 것입니다. Fourier feature encoding을 통해 샘플링된 포인트에 고주파 정보를 캡처하여 재구성 결과를 개선합니다.

- **Performance Highlights**: 다양한 데이터셋에서 수행한 평가 결과, 제안된 방법이 현재 최첨단 접근 방식들보다 더 경쟁력 있는 결과를 달성하는 것으로 나타났습니다. 또한, 동적 객체 제거 및 구멍 메우기에서 뛰어난 성능을 보여줍니다.



### AlterMOMA: Fusion Redundancy Pruning for Camera-LiDAR Fusion Models with Alternative Modality Masking (https://arxiv.org/abs/2409.17728)
Comments:
          17 pages, 3 figures, Accepted by NeurIPS 2024

- **What's New**: 본 논문에서는 카메라-라이다(Camera-LiDAR) 융합 모델의 효율성을 높이기 위해 새로운 가지치기 프레임워크인 AlterMOMA를 제안합니다.

- **Technical Details**: AlterMOMA는 각 모달리티에 대해 대체 마스킹을 사용하고 중요도 평가 함수인 AlterEva를 통해 중복된 파라미터를 식별합니다. 이 과정에서는 한 모달리티 파라미터가 활성화 및 비활성화될 때의 손실 변화를 관찰하여 중복 파라미터를 식별합니다.

- **Performance Highlights**: AlterMOMA는 nuScenes 및 KITTI 데이터셋의 다양한 3D 자율 주행 작업에서 기존의 가지치기 방법보다 뛰어난 성능을 보이며, 최첨단 성능을 달성하였습니다.



### Scene Understanding in Pick-and-Place Tasks: Analyzing Transformations Between Initial and Final Scenes (https://arxiv.org/abs/2409.17720)
Comments:
          Conference Paper, ICEE 2024, 7 pages, 5 figures

- **What's New**: 본 논문은 로봇이 사람과 함께 작업할 수 있도록 환경을 이해할 수 있는 로봇 시스템의 개발을 목표로 한다. 초기 및 최종 장면의 이미지를 통해 픽 앤 플레이스(pick and place) 작업을 감지하기 위한 데이터셋을 수집하고 YOLOv5 네트워크를 사용해 객체를 탐지하여, 픽 앤 플레이스 작업을 도출하는 두 가지 방법을 제안한다. 

- **Technical Details**: 제안된 방법 중 첫 번째는 기하학적 방법으로, 두 장면에서 객체의 움직임을 추적하여 바운딩 박스(bounding boxes)의 교차점에 기반하여 작업을 감지한다. 두 번째는 CNN(Convolutional Neural Network) 기반 방법으로, 교차된 바운딩 박스를 가진 객체를 분류하여 객체 간의 공간적 관계를 이해한다. VGG16 백본을 사용하는 CNN 기반 방법이 기하학적 방법보다 약 12% 높은 성능을 보였다.

- **Performance Highlights**: CNN 기반 방법은 특정 상황에서 84.3%의 성공률을 기록하며, 전반적인 성능에서 기하학적 방법보다 약 12% 높은 점수를 기록하였다.



### Behaviour4All: in-the-wild Facial Behaviour Analysis Toolk (https://arxiv.org/abs/2409.17717)
- **What's New**: Behavior4All은 야외 환경에서 얼굴 행동 분석을 위한 포괄적이고 오픈 소스인 도구 모음입니다. 얼굴 위치 탐지, 감정 점수 추정, 기본 표정 인식 및 행동 단위 탐지를 하나의 프레임워크 내에서 통합합니다.

- **Technical Details**: Behavior4All은 CPU 버전과 GPU 가속 버전으로 제공되며, 12개의 대규모 야외 데이터셋을 활용하여 500만 장 이상의 이미지로 테스트되었습니다. 이 도구는 비포함 주석을 다루기 위해 분포 일치 및 레이블 공동 주석을 활용하는 새로운 프레임워크를 도입하여, 관련성에 대한 사전 지식을 인코딩합니다. 또한, FaceLocalizationNet과 FacebehaviourNet의 두 가지 주요 구성 요소로 구성되어 있습니다.

- **Performance Highlights**: Behavior4All은 AUC, 기본 표정 인식, VA 추정 및 AU 탐지를 포함한 모든 데이터베이스와 작업에서 기존 도구 및 최첨단 기술을 초월하는 성능과 공정성을 보여주었습니다. 또한, 다른 도구들보다 1.9배 이상의 처리 속도로 작동합니다.



### MoGenTS: Motion Generation based on Spatial-Temporal Joint Modeling (https://arxiv.org/abs/2409.17686)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 본 연구는 개별 관절을 벡터로 양자화하여 모션 생성을 간소화하고, 공간-시간적 구조를 유지하며, 2D 토큰 맵을 생성하는 새로운 방법을 제안합니다. 이는 이전 방법들이 전체 포즈를 양자화하면서 발생했던 문제들을 해결하는 방향으로 발전했습니다.

- **Technical Details**: 우리는 2D 모션 양자화를 기반으로 한 공간-시간 모델링 프레임워크를 구축하였습니다. 여기서는 2D 조인트 VQVAE, 시간-공간 2D 마스킹 기술 및 공간-시간 2D 주의(attention) 메커니즘을 결합하여 2D 토큰들 간의 신호를 효과적으로 활용합니다. 각각의 관절은 개별 코드로 양자화되어 2D 토큰을 형성하며, 2D 컨볼루션 네트워크를 통해 효율적인 특성 추출이 이루어집니다.

- **Performance Highlights**: 우리의 방법은 HumanML3D 데이터셋에서 FID를 26.6% 감소시키고, KIT-ML 데이터셋에서는 29.9% 감소시키는 성능 향상을 보였습니다. 이는 이전의 최첨단 방법(SOTA) 대비 매우 우수한 결과입니다.



### Dark Miner: Defend against unsafe generation for text-to-image diffusion models (https://arxiv.org/abs/2409.17682)
- **What's New**: 이번 논문은 Text-to-image diffusion 모델의 안전성 문제를 해결하기 위해 새로운 방법인 Dark Miner를 제안합니다. 기존 방법들이 훈련 단계에서 보지 못한 텍스트에 대한 안전한 생성을 보장하지 못했던 문제를 분석하고, 안전 개념을 효과적으로 지우기 위해 반복되는 3단계 과정(채굴, 검증, 회피)을 포함한 DM2 방식에 대해 설명합니다.

- **Technical Details**: Dark Miner는 unsafe 개념의 최대 생성 확률을 가진 임베딩을 채굴하는 'mining', 이를 검증하는 'verifying', 그리고 안전한 텍스트에 조건화된 생성 확률로 수정하는 'circumventing'의 세 가지 단계를 반복적으로 수행합니다. 이 과정에서 DM2는 unsafe 생성의 총 확률을_EFFECTIVELY_ 줄이는 방안을 모색합니다.

- **Performance Highlights**: 실험 결과, DM2는 6개의 최신 방법과 비교했을 때 전반적인 지우기 성능과 공격 방어 성능에서 우수한 결과를 보였으며, 4개의 최신 공격에 대해서도 높은 방어 효과를 유지했습니다. 이러한 성능은 DM2가 일반 텍스트에 조건화된 생성 능력을 보존하면서 이루어졌습니다.



### Event-based Stereo Depth Estimation: A Survey (https://arxiv.org/abs/2409.17680)
Comments:
          28 pages, 20 figures, 7 tables

- **What's New**: 이 연구는 기존의 연구들에서 다루어지지 않았던 스테레오 데이터셋에 대한 포괄적인 개요를 제공합니다. 또한, 딥러닝(DL) 방법과 스테레오 데이터셋에 대한 광범위한 리뷰를 최초로 수행하고, 새로운 벤치마크를 생성하기 위한 실용적인 제안도 포함하고 있습니다.

- **Technical Details**: 이 논문은 이벤트 카메라(event cameras)의 깊이 추정(depth estimation)과 관련하여 스테레오(Stereo) 기술을 다룹니다. 이벤트 카메라는 퍼픽셀(pixel) 밝기 변화(brightness changes)를 비동기적으로 감지하는 생체 모방 센서로, 높은 시간 해상도(temporal resolution)와 높은 동적 범위(dynamic range)를 가지고 있습니다. 이러한 특성으로 인해 스테레오 매칭(stereo matching)에서의 성능이 이점이 있습니다.

- **Performance Highlights**: 이 연구는 스테레오 깊이 추정 분야에서 딥러닝 기반 접근 방식의 발전을 조명하고 있으며, 정확도(accuracy)와 효율성(efficiency) 측면에서 최적 성능을 달성하는 데 여전히 많은 도전이 있다고 밝혔습니다. 이벤트 기반 컴퓨팅(event-based computing)에서의 주요 장점과 도전 과제(challenges)에 대해서도 논의하고 있습니다.



### EM-Net: Efficient Channel and Frequency Learning with Mamba for 3D Medical Image Segmentation (https://arxiv.org/abs/2409.17675)
Comments:
          10 pages, 3 figures, accepted by MICCAI 2024

- **What's New**: 본 논문에서는 EM-Net이라는 새로운 3D 의료 이미지 분할 모델을 소개합니다. EM-Net은 Mamba 기반 아키텍처를 활용하여 지역 간의 상호작용을 효율적으로 캡처하며, 다양한 스케일에서의 특징 학습을 조화롭게 하는 주파수 도메인 학습을 활용하여 훈련 속도를 가속화할 수 있습니다.

- **Technical Details**: EM-Net은 채널 압축 강화 Mamba 블록(CSRM 블록)과 효율적인 주파수 도메인 학습(EFL) 레이어를 채택하여 다중 스케일의 특징을 추출하며, Mamba 기술을 통해 메모리 비용을 줄이면서 분할 성능을 향상시킵니다. 이 프레임워크는 효율적인 인코더-디코더 구조를 기반으로 하며, 3D 입력 데이터를 패치로 분할하고, 각각의 패치에서 중요한 특징을 학습합니다.

- **Performance Highlights**: EM-Net은 두 개의 도전적인 다중 장기 데이터셋에서 테스트되었으며, 기존 최첨단(SOTA) 모델보다 2배 빠른 훈련 속도를 제공하며, 파라미터 크기가 거의 절반인 상황에서 더 나은 분할 정확도를 보여주었습니다.



### Self-Supervised Learning of Deviation in Latent Representation for Co-speech Gesture Video Generation (https://arxiv.org/abs/2409.17674)
Comments:
          5 pages, 5 figures, conference

- **What's New**: 이번 연구에서는 공동 언어(코스피치) 제스처 생성을 위한 혁신적인 접근 방식을 제안하며, 이는 자가 감독(self-supervised) 표현과 픽셀 수준의 모션 변화를 결합합니다. 기존의 방법이 점 수준(point-level) 모션 변환에 주로 집중했던 것에 비해, 본 연구는 손 제스처 생성의 질을 높이는데 집중하고 있습니다.

- **Technical Details**: 제안된 방법은 자가 감독 모듈을 이용하여 풍경(세부) 변화를 생성합니다. 이 모듈은 라텐트(잠재) 변화 추출기(latent deviation extractor), 워핑 계산기(warping calculator), 라텐트 변화 디코더(latent deviation decoder)로 구성됩니다. 이 시스템은 발화자의 음성 및 소스 이미지 입력을 통해 자연스럽고 동기화된 제스처 비디오를 생성합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 생성된 비디오의 질을 향상시켰으며, FGD, DIV, FVD에서 2.7%에서 4.5% 개선, PSNR에서 8.1%, SSIM에서 2.5% 개선을 보였습니다. 이는 현재의 최신 기법보다 뛰어난 성과를 보여줍니다.



### Leveraging Anthropometric Measurements to Improve Human Mesh Estimation and Ensure Consistent Body Shapes (https://arxiv.org/abs/2409.17671)
- **What's New**: 이 연구는 SOTA (State of the Art) 인간 메시 추정(Human Mesh Estimation, HME) 모델이 비디오의 각 프레임마다 다르게 몸체 모양을 출력하는 문제를 해결하기 위해, 인체 측정(anthropometric measurements)을 이용하여 각각의 인간에게 맞는 일관된 기본 몸체 모양을 생성하는 모델 A2B를 제안합니다.

- **Technical Details**: A2B 모델은 인체 측정을 HME의 몸체 모양 파라미터로 변환하는 머신 러닝 모델입니다. 이 모델은 HME 데이터셋의 GT(ground truth) 데이터의 불일치를 밝혀내고, SMPL-X 모델의 몸체 모양 파라미터와 인체 측정을 연결하는 다양한 모델을 생성하고 평가하여 일관된 메시를 생성합니다. 추가적으로, 조인트 회전을 보완하기 위해 역기구학(inverse kinematics, IK)을 적용합니다.

- **Performance Highlights**: A2B 모델을 통해 HME 모델의 성능이 크게 향상되어, ASPset 및 fit3D와 같은 도전적인 데이터셋에서 MPJPE(Mean Per Joint Position Error)를 30mm 이상 낮출 수 있음을 보여줍니다. 제안된 방법은 SOTA HME 모델보다 더 높은 정확도로 HME를 구현할 수 있습니다.



### MECD: Unlocking Multi-Event Causal Discovery in Video Reasoning (https://arxiv.org/abs/2409.17647)
Comments:
          Accepted at NeurIPS 2024 as a spotlight paper

- **What's New**: 새로운 과제인 Multi-Event Causal Discovery (MECD)는 장기 비디오에서 사건 간의 인과 관계를 발견하는 것을 목표로 합니다. 기존의 질문-답변(Question-Answering) 패러다임에 한정된 비디오 추론 과제를 넘어 다양한 이벤트에 대한 포괄적이고 구조화된 인과 분석을 제공합니다.

- **Technical Details**: 이 연구에서는 Granger Causality 방법에서 영감을 받아 새로운 프레임워크를 제안하여 효율적인 마스크 기반 이벤트 예측 모델을 사용하고, Event Granger Test를 수행합니다. 이를 통해 조건부 인과 추론 기법인 front-door adjustment와 counterfactual inference를 통합하여 인과성 혼동과 허위 인과성 문제를 해결합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 다중 이벤트 비디오에서 인과 관계를 예측하는 데 있어 GPT-4o 및 VideoLLaVA보다 각각 5.7% 및 4.1% 더 우수한 성능을 보였습니다.



### P4Q: Learning to Prompt for Quantization in Visual-language Models (https://arxiv.org/abs/2409.17634)
- **What's New**: 이 논문에서는 Vision-Language Models (VLMs) 의 양자화(quantization)와 미세 조정(fine-tuning)을 통합하는 새로운 방법인 'Prompt for Quantization'(P4Q)을 제안합니다. 이 방법은 PTQ(Post-Training Quantization) 모델의 인식 성능을 향상시키기 위해 경량 아키텍처를 설계합니다.

- **Technical Details**: P4Q는 이미지 특징과 텍스트 특징 간의 격차를 줄이기 위해 학습 가능한 프롬프트(prompt)를 사용하여 텍스트 표현을 재구성하고, 저비트(low-bit) 어댑터(QAdapter)를 사용하여 이미지와 텍스트 특징 분포를 재조정하는 방법입니다. 또한 코사인 유사도 예측을 기반으로 하는 증류 손실(distillation loss)을 도입하여 확장성 있는 증류를 수행합니다.

- **Performance Highlights**: P4Q 방법은 이전 연구보다 뛰어난 성능을 보이며, ImageNet 데이터셋에서 8비트 P4Q가 CLIP-ViT/B-32을 4배 압축하면서도 Top-1 정확도 66.94%를 기록하였습니다. 이는 전체 정밀(full-precision) 모델보다 2.24% 향상된 결과입니다.



### Hand-object reconstruction via interaction-aware graph attention mechanism (https://arxiv.org/abs/2409.17629)
Comments:
          7 pages, Accepted by ICIP 2024

- **What's New**: 이 연구에서는 손과 객체의 상호작용을 고려한 새로운 그래프 주의(attention) 메커니즘을 제안합니다. 기존 방법론이 그래프의 엣지 연관성을 충분히 활용하지 못했던 점을 극복하기 위해, 공통 관계 엣지(common relation edges)와 주의 유도 엣지(attention-guided edges)를 도입하여 물리적 타당성을 향상시키기 위한 그래프 기반 개선 방법을 소개합니다.

- **Technical Details**: 제안된 접근 방식은 상호작용 인식을 위한 그래프 주의 메커니즘을 통해 손과 객체의 메쉬(mesh)를 이미지에서 추정하여, 두 가지 타이프의 엣지(즉, intra-class와 inter-class 노드 간의 연결)를 사용하여 밀접하게 상관된 노드 간의 관계를 설정합니다. 이 연구는 ObMan와 DexYCB 데이터셋을 활용하여 제안된 방법의 효과성을 평가했습니다.

- **Performance Highlights**: 실험 결과, 손과 객체 간의 물리적 타당성이 개선되었음을 확인했습니다. 정량적 및 정성적으로 관찰된 결과는 제안된 상호작용 인식 그래프 주의 메커니즘이 손과 객체 포즈 추정을 통해 물리적 타당성을 획기적으로 향상시킴을 보여줍니다.



### Appearance Blur-driven AutoEncoder and Motion-guided Memory Module for Video Anomaly Detection (https://arxiv.org/abs/2409.17608)
Comments:
          13 pages, 11 figures

- **What's New**: 비디오 이상 탐지(VAD)의 새로운 방법으로, 우리는 제로샷 학습을 통한 교차 데이터셋 검증을 가능하게 하는 모션 가이드를 통한 메모리 모듈을 제안합니다. 이 방법은 Gaussian 블러를 적용한 이미지로부터 비정상적으로 흐릿한 특성을 인식합니다.

- **Technical Details**: 본 연구에서는 Gaussian 블러를 raw appearance 이미지에 추가하여 global pseudo-anomaly를 생성하고, multi-scale residual channel attention(MRCA)을 통해 정상 샘플의 이 pseudo-anomaly를 복원합니다. 수업 단계에서 모션 특성을 기록하여 메모리 아이템을 추출하고, 테스트 단계에서 이 정보를 활용해 정상성과 비정상 동작의 간극을 확대합니다.

- **Performance Highlights**: 세 가지 벤치마크 데이터셋에 대한 광범위한 실험에서 제안된 방법이 경쟁력 있는 성능을 발휘함을 입증했습니다. 특히, 테스트 중 적응 없이도 강력한 성능을 달성하며 교차 데이터셋 검증에서 우수함을 보여줍니다.



### Good Data Is All Imitation Learning Needs (https://arxiv.org/abs/2409.17605)
- **What's New**: 이 논문에서는 Autonomous/Automated Driving Systems (ADS)에서 기존의 teacher-student 모델, imitation learning, behavior cloning의 한계를 극복하기 위해 Counterfactual Explanations (CFEs)를 새로운 데이터 증강 기법으로 도입하였습니다.

- **Technical Details**: CFEs는 최소한의 입력 수정으로 결정 경계 근처에 있는 학습 샘플을 생성하여 수집된 데이터에 희귀한 사건을 포함시킴으로써 전문가 운전자의 전략을 더 포괄적으로 표현할 수 있게 해줍니다. 이 논문은 CARLA 시뮬레이터에서 CF-Driver를 통해 실험하며 SOTA 수준의 성과를 입증하였습니다.

- **Performance Highlights**: CF-Driver는 드라이빙 점수 84.2를 기록하며 이전 최고 모델보다 15.02% 향상된 성과를 달성하며, CFEs를 통해 드라이빙 전략의 개선을 보여줍니다. 또한, 연구 개발을 위해 생성된 데이터셋은 공개하여 후속 연구를 촉진할 계획입니다.



### TA-Cleaner: A Fine-grained Text Alignment Backdoor Defense Strategy for Multimodal Contrastive Learning (https://arxiv.org/abs/2409.17601)
- **What's New**: 이 논문에서는 CLIP와 같은 대규모 사전학습 모델이 데이터 중독된 백도어 공격에 취약하다는 점을 지적하며, 이를 해결하기 위한 새로운 방어 기법 TA-Cleaner를 제안합니다.

- **Technical Details**: TA-Cleaner는 CleanCLIP의 한계를 보완하기 위해 세밀한 텍스트 정렬 기법을 적용합니다. 매 에포크마다 긍정 및 부정 서브텍스트를 생성하고 이를 이미지와 정렬하여 텍스트 자기 감독(self-supervision)을 강화함으로써 백도어 트리거의 특징 연결을 차단합니다.

- **Performance Highlights**: 특히 BadCLIP과 같은 새로운 공격 기법에 대해 TA-Cleaner는 CleanCLIP보다 Top-1 ASR을 52.02%, Top-10 ASR을 63.88% 감소시키며 뛰어난 방어 성능을 보임을 보여주었습니다.



### Unifying Dimensions: A Linear Adaptive Approach to Lightweight Image Super-Resolution (https://arxiv.org/abs/2409.17597)
- **What's New**: 본 논문에서는 기존의 Window-based Transformer의 높은 계산 복잡성 문제를 해결하기 위해 linear focal separable attention (FSA)와 dual-branch 구조를 결합한 Linear Adaptive Mixer Network (LAMNet)를 제안했습니다. 이를 통해 경량화된 모델링이 가능해졌으며, inference 속도가 개선되었습니다.

- **Technical Details**: LAMNet는 ConvNet의 장점을 살려 adaptive spatial aggregation 기능을 갖춘 convolution 기반 구조입니다. 구체적으로, FSA는 2D 가중치 행렬을 희소화하면서도 높은 차원의 정보와 복잡한 관계를 모형화할 수 있도록 설계되었으며, Channel Selective Mixer (CSM)와 정보 교환 모듈 (IEM)도 포함되어 있습니다. DGFN 구조를 통해 spatial gating 구현 시 채널 정보의 다변성을 보장합니다.

- **Performance Highlights**: 실험 결과, LAMNet는 기존의 SA 기반 Transformer 방법들보다 우수한 성능을 달성하면서도 CNN만큼의 계산 효율성을 유지하며, inference 시간을 최대 3배 단축하는 성과를 보여주었습니다.



### Improving Fast Adversarial Training via Self-Knowledge Guidanc (https://arxiv.org/abs/2409.17589)
Comments:
          13 pages

- **What's New**: 이 논문에서는 Fast Adversarial Training (FAT)에서 발생하는 불균형 문제를 체계적으로 조사하고, 이를 해결하는 Self-Knowledge Guided FAT (SKG-FAT) 방법론을 제안합니다. SKG-FAT는 각 클래스의 학습 상태에 따라 차별화된 정규화 가중치를 할당하고, 학습 정확도에 따라 레이블을 동적으로 조정함으로써/adversarial robustness를 증가시키는 데 초점을 맞춥니다.

- **Technical Details**: FAT의 기존 방법들은 모든 훈련 데이터를 균일하게 최적화하는 전략을 사용하여/imbalanced optimization을 초래합니다. 본 연구에서는 클래스 간의 성능 격차를 드러내고, 이를 해소하기 위해/self-knowledge guided regularization과/self-knowledge guided label relaxation을 도입합니다. SKG-FAT는 자연적으로 생성되는 지식을 활용하여 adversarial robustness를 증대시킬 수 있습니다.

- **Performance Highlights**: SKG-FAT는 네 가지 표준 데이터셋에 대한 광범위한 실험을 통해/adversarial robustness를 향상시키면서도 경쟁력 있는 clean accuracy를 유지하며, 최신 방법들과 비교해 우수한 성능을 보였습니다.



### ID$^3$: Identity-Preserving-yet-Diversified Diffusion Models for Synthetic Face Recognition (https://arxiv.org/abs/2409.17576)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 본 논문에서는 심층 학습을 기반으로 한 합성 얼굴 인식(SFR) 모델인 ID³를 소개합니다. 이 모델은 개인 정보 보호를 고려하여 실제 얼굴 데이터의 분포를 모사하는 합성 얼굴 데이터셋을 생성하는 데 중점을 두고 있습니다.

- **Technical Details**: ID³ 모델은 ID-preserving loss를 활용하여 서로 다른 얼굴 속성을 지니면서도 동일한 개체의 정체성을 유지하는 얼굴 이미지를 생성합니다. 이론적으로는 제안한 손실 함수를 최소화하는 것이 조정된 조건부 로그 가능성의 하한을 최대화하는 것과 동등하다는 것을 보여줍니다.

- **Performance Highlights**: ID³는 다섯 가지 난이도 높은 벤치마크에서 기존의 최첨단 SFR 방법들 대비 약 2.4%의 성능 향상을 보였습니다.



### Flexiffusion: Segment-wise Neural Architecture Search for Flexible Denoising Schedu (https://arxiv.org/abs/2409.17566)
- **What's New**: Flexiffusion은 신속한 생성 과정을 최적화하기 위해 생성 단계와 네트워크 구조를 동시에 조정하는 새롭고 훈련이 필요하지 않은 Neural Architecture Search(NAS) 패러다임을 도입했습니다.

- **Technical Details**: Flexiffusion은 생성 과정을 등간격 단계로 나누고, 각 단계는 전면 단계(full step), 부분 단계(partial step), 무효 단계(null step)로 구성됩니다. 이 방법은 캐시 메커니즘을 사용해 모델 재훈련 없이 효율적인 탐색을 가능하게 합니다.

- **Performance Highlights**: Flexiffusion의 모델은 LDM-4-G의 경우 2.6배, Stable Diffusion V1.5에서는 5.1배의 속도 향상을 기록했습니다. 여러 데이터셋에서의 실험 결과 Flexiffusion이 이미지 생성 속도와 품질을 효과적으로 개선함을 확인했습니다.



### Pixel-Space Post-Training of Latent Diffusion Models (https://arxiv.org/abs/2409.17565)
- **What's New**: 이번 논문에서는 Latent Diffusion Models (LDMs)의 한계를 극복하기 위한 새로운 접근 방법을 제안합니다. LDMs는 이미지 생성 분야에서 큰 발전을 이루었지만, 고주파 세부 사항과 복잡한 구성 생성에서 여전히 어려움이 있음을 지적합니다.

- **Technical Details**: LDMs가 고주파 세부 사항을 잘 생성하지 못하는 이유는 기존 학습이 보통 $8 	imes 8$의 낮은 공간 해상도에서 이뤄지기 때문이라고 가설합니다. 이를 해결하기 위해 포스트 트레이닝 과정에서 pixel-space supervision을 추가하는 방법을 제안하였습니다.

- **Performance Highlights**: 실험 결과, pixel-space objective를 추가함으로써 최첨단 DiT transformer와 U-Net diffusion 모델의 supervised quality fine-tuning 및 preference-based post-training에서 시각적 품질 및 결함 지표가 크게 향상되었으며, 동일한 텍스트 정렬 품질을 유지했습니다.



### General Compression Framework for Efficient Transformer Object Tracking (https://arxiv.org/abs/2409.17564)
- **What's New**: 본 논문에서는 CompressTracker라는 새로운 모델 압축 프레임워크를 제안하여, 사전 훈련된 추적 모델의 사이즈를 최소한의 성능 저하로 경량화하는 방법을 소개합니다. 이 방법은 Transformer 기반의 객체 추적에서의 효율성을 향상시키기 위한 단계 구분 전략을 특징으로 합니다.

- **Technical Details**: CompressTracker는 사전 훈련된 교사 모델의 변환기 레이어를 여러 개의 단계로 나누어 각 단계에서 학생 모델이 교사 모델의 행동을 모방하도록 학습합니다. 또한, 교사 모델의 특정 단계를 학생 모델의 특정 단계와 랜덤하게 교체하여 훈련하는 교체 훈련(replacement training) 기법을 도입하였습니다. 이러한 방법은 학습 과정에서 예측 가이던스(prediction guidance)와 단계 별 특성 모방(stage-wise feature mimicking)을 추가하여 학생 모델의 성능을 향상시킵니다.

- **Performance Highlights**: CompressTracker-4는 OSTrack에서 압축되어 4개의 Transformer 레이어를 사용하며, LaSOT에서 약 96%의 성능을 유지(66.1% AUC)하면서 2.17배 속도 향상을 달성합니다. 이 모델은 훈련 시간도 20시간과 같이 단순화된 과정으로 줄였습니다.



### Dynamic Subframe Splitting and Spatio-Temporal Motion Entangled Sparse Attention for RGB-E Tracking (https://arxiv.org/abs/2409.17560)
Comments:
          15 pages, 8 figures, conference

- **What's New**: 이 논문에서는 동적 이벤트 서브프레임 분할 전략(Dynamic Event Subframe Splitting, DES)을 제안하여 이벤트 스트림을 더 미세한 이벤트 클러스터로 나누고, 이에 따라 상대적으로 느린 속도의 객체 추적이 가능하도록 합니다. 또한, 이러한 접근을 통해 시간적 정보를 최대한 활용합니다.

- **Technical Details**: 제안한 방법은 이벤트 기반 희소 주의(attention) 메커니즘(Event-based Sparse Attention, ESA)를 설계하여 시공간(spatial and temporal) 차원에서의 이벤트 특징 간 상호작용을 강화합니다. 동적 이벤트 서브프레임 분할 전략(DES)은 이벤트 스트림을 다수의 동적 서브프레임으로 나누어 모션 정보를 보존합니다. 또한, 시공간 모션 얽힘 추출기(Spatio-Temporal Motion Entanglement Extractor, STME)는 이벤트 기반 희소 주의 메커니즘을 결합하여 다양한 시간적 속성의 이동 정보를 추출합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 FE240 및 COESOT 데이터셋에서 기존 최첨단 방법보다 뛰어난 성능을 보여주었으며, 이벤트 데이터 처리를 위한 효과적인 방법을 제공합니다.



### Triple Point Masking (https://arxiv.org/abs/2409.17547)
- **What's New**: 본 논문에서는 제한된 데이터 환경에서 기존 3D 마스크 학습 방법의 성능 한계를 극복하기 위한 새로운 Triple Point Masking (TPM) 기법을 소개합니다. 이 기법은 다중 마스크 학습을 위한 확장 가능한 프레임워크로, 3D 포인트 클라우드 데이터를 위한 masked autoencoder의 사전 학습을 지원합니다.

- **Technical Details**: TPM은 기본 모델에 두 가지 추가 마스크 선택지(중간 마스크 및 낮은 마스크)를 통합하여 객체 복구 과정을 다양한 방법으로 표현할 수 있도록 설계되었습니다. 이 과정에서 고차원 마스킹 기법의 한계를 극복하고, 다양한 3D 객체에 대한 여러 표현을 고려하여 더 유연하고 정확한 완성 능력을 제공합니다. 또한, SVM 기반의 최적 가중치 선택 모듈을 통해 다운스트림 네트워크에 최적의 가중치를 적용하여 선형 정확성을 극대화합니다.

- **Performance Highlights**: TPM을 탑재한 네 가지 기본 모델은 다양한 다운스트림 작업에서 전반적인 성능 향상을 달성했습니다. 예를 들어, Point-MAE는 TPM 적용 후 사전 학습 및 미세 조정 단계에서 각각 1.1% 및 1.4%의 추가 성과를 보였습니다.



### CAMOT: Camera Angle-aware Multi-Object Tracking (https://arxiv.org/abs/2409.17533)
- **What's New**: 본 논문은 Multi-Object Tracking (MOT)을 위한 간단한 카메라 각도 추정기인 CAMOT를 제안합니다. 이 방법은 occlusion과 부정확한 거리 추정을 해결하는 데 중점을 둡니다.

- **Technical Details**: CAMOT는 비디오 프레임 내의 여러 객체가 평면에 위치해 있다는 가정을 바탕으로 객체 탐지를 통해 카메라 각도를 추정합니다. 이 방법은 각 객체의 깊이를 제공하여 pseudo-3D MOT를 가능하게 하고, 다양한 2D MOT 방법에 플러그인 형태로 적용될 수 있습니다.

- **Performance Highlights**: CAMOT를 ByteTrack에 적용했을 때 MOT17에서 63.8% HOTA, 80.6% MOTA, 78.5% IDF1를 기록하며, 기존의 깊이 추정기들보다 계산 비용이 훨씬 낮고 속도는 24.92 FPS에 달합니다.



### SimVG: A Simple Framework for Visual Grounding with Decoupled Multi-modal Fusion (https://arxiv.org/abs/2409.17531)
Comments:
          21pages, 11figures, NeurIPS2024

- **What's New**: 이번 연구에서는 Visual Grounding (VG) 문제를 해결하기 위한 새로운 프레임워크 SimVG를 제안합니다. 기존의 복잡한 모듈이나 아키텍처 대신, SimVG는 멀티모달 전이 학습 모델을 활용하여 시각-언어 기능 융합을 하위 작업으로부터 분리합니다.

- **Technical Details**: SimVG는 기존의 멀티모달 모델을 기반으로 하며, 객체 토큰(Object Tokens)을 포함하여 하위 작업(Task)과 사전 학습(Pre-training) 작업을 깊게 통합하는 방식으로 설계되었습니다. 동적 가중치 균형 증류(DWBD) 방법을 사용하여 다중 브랜치 동기식 학습 과정에서 간단한 브랜치의 표현력을 향상시킵니다. 이 브랜치는 경량 MLP로 구성되어, 구조를 단순화하고 추론 속도를 개선합니다.

- **Performance Highlights**: SimVG는 RefCOCO/+/g, ReferIt, Flickr30K 등 총 6개의 VG 데이터셋에서 실험을 실시한 결과, 효율성과 수렴 속도에서 개선을 이루었으며, 새로운 최첨단 성능을 달성했습니다. 특히, 단일 RTX 3090 GPU에서 RefCOCO/+/g 데이터셋에 대해 12시간 훈련하여 이룬 성과가 주목할 만합니다.



### Drone Stereo Vision for Radiata Pine Branch Detection and Distance Measurement: Integrating SGBM and Segmentation Models (https://arxiv.org/abs/2409.17526)
- **What's New**: 이 연구에서는 드론 기반의 가지치기 시스템을 개발하여, 전통적인 수동 가지치기의 안전 위험을 해결하고자 합니다. 이 시스템은 전문적인 가지치기 도구와 스테레오 비전 카메라를 활용하여 가지를 정확하게 감지하고 자를 수 있는 기능을 제공합니다.

- **Technical Details**: 시스템은 YOLO와 Mask R-CNN을 포함한 딥 러닝 알고리즘을 사용하여 가지 감지를 정확하게 수행하며, Semi-Global Matching 알고리즘을 통합하여 신뢰성 있는 거리 추정을 가능하게 합니다. 이를 통해 드론은 가지의 위치를 정밀하게 파악하고 효율적인 가지치기를 수행할 수 있습니다.

- **Performance Highlights**: 실험 결과에 따르면, YOLO와 SGBM의 결합 구현이 드론이 가지를 정확하게 감지하고 드론으로부터의 거리를 측정하는 데 성공적이라는 것을 보여줍니다. 이 연구는 가지치기 작업의 안전성과 효율성을 향상시키는 한편, 농업 및 임업 관행의 자동화를 위한 드론 기술의 발전에 중요한 기여를 합니다.



### JoyType: A Robust Design for Multilingual Visual Text Creation (https://arxiv.org/abs/2409.17524)
Comments:
          Under Review at AAAI 2025

- **What's New**: 이번 논문에서는 멀티링구얼 비주얼 텍스트 생성을 위한 JoyType이라는 새로운 접근법을 소개합니다. JoyType은 이미지 생성 과정에서 텍스트의 글꼴 스타일을 유지하도록 설계되었습니다.

- **Technical Details**: JoyType는 1 백만 개의 데이터 쌍으로 구성된 훈련 데이터셋, JoyType-1M을 기반으로 합니다. 각 데이터 쌍은 이미지, 해당 설명 및 이미지 내 글꼴 스타일에 대한 글리프 지침을 포함합니다. Font ControlNet을 개발하여 글꼴 스타일 정보를 추출하고 이미지 생성을 조정하며, 멀티 레이어 OCR 인식 손실(multi-layer OCR-aware loss)을 도입하여 작은 글꼴을 생성하는 모델의 능력을 강화합니다.

- **Performance Highlights**: JoyType은 기존 최첨단 방법들에 비해 현저하게 우수한 성능을 보였습니다. HuggingFace 및 CivitAI와 같은 다른 안정적인 확산 모델과 함께 다양한 스타일의 이미지를 생성하는 플러그인으로 기능할 수 있습니다.



### EAGLE: Egocentric AGgregated Language-video Engin (https://arxiv.org/abs/2409.17523)
Comments:
          Accepted by ACMMM 24

- **What's New**: EAGLE (Egocentric AGgregated Language-video Engine) 모델과 EAGLE-400K 데이터셋을 소개하여, 다양한 egocentric video 이해 작업을 통합하는 단일 프레임워크를 제공합니다.

- **Technical Details**: EAGLE-400K는 400K개의 다양한 샘플로 구성된 대규모 instruction-tuning 데이터셋으로, 활동 인식부터 절차 지식 학습까지 다양한 작업을 향상시킵니다. EAGLE는 공간적(spatial) 및 시간적(temporal) 정보를 효과적으로 캡처할 수 있는 강력한 비디오 멀티모달 대형 언어 모델(MLLM)입니다.

- **Performance Highlights**: 광범위한 실험을 통해 EAGLE는 기존 모델보다 우수한 성능을 발휘하며, 개별 작업에 대한 이해와 비디오 내용에 대한 전체적인 해석을 균형 있게 수행할 수 있는 능력을 강조합니다.



### SCOMatch: Alleviating Overtrusting in Open-set Semi-supervised Learning (https://arxiv.org/abs/2409.17512)
Comments:
          ECCV 2024 accepted

- **What's New**: 이번 논문에서는 Open-set Semi-Supervised Learning (OSSL)의 새로운 방법인 SCOMatch를 제안합니다. 이 방법은 기존 OSSL이 직면하는 과도한 신뢰의 문제를 해결하며, OOD(Out-of-Distribution) 샘플을 새로운 클래스(functional class)로 취급하여 더욱 효과적인 학습을 진행합니다.

- **Technical Details**: SCOMatch는 두 가지 주요 전략을 통해 OOD 샘플을 신뢰할 수 있는 레이블 데이터로 선택하고(Memory Queue), 기존 SSL 작업에 새로운 (K+1)-class SSL을 통합합니다. 이 과정에서 OOD 메모리 큐와 업데이트 전략을 사용하고, Close-set 및 Open-set self-training을 동시에 수행하게 됩니다.

- **Performance Highlights**: SCOMatch는 TinyImageNet 데이터셋에서 Close-set 정확도를 13.4% 향상시키는 등 여러 벤치마크에서 기존 OSSL 방법들을 뛰어넘는 성능을 보여 줍니다. 이는 포괄적인 실험과 시각화를 통해 각 구성 요소의 효과가 입증되었습니다.



### Uni-Med: A Unified Medical Generalist Foundation Model For Multi-Task Learning Via Connector-MoE (https://arxiv.org/abs/2409.17508)
- **What's New**: 이 논문은 다양한 시각 및 언어 작업을 위한 일반-purpose 인터페이스로서, 의료 분야의 multi-task learning을 위한 통합된 Multi-modal large language model (MLLM)을 제안합니다. 특히, 이전의 연구들이 처리하지 않았던 connector 문제를 해결하고자 합니다. 향후 Uni-Med는 의학 분야에서의 새로운 시도를 이루어낼 것입니다.

- **Technical Details**: Uni-Med는 범용 시각 feature extraction 모듈, connector mixture-of-experts (CMoE) 모듈, 그리고 LLM으로 구성된 의료 전문 기초 모델입니다. CMoE는 잘 설계된 라우터를 활용하여 projection experts의 혼합을 통해 connector에서 발생하는 문제를 해결합니다. 이 접근 방식을 통해 6개의 의료 관련 작업을 수행할 수 있습니다: 질문 응답(question answering), 시각 질문 응답(visual question answering), 보고서 생성(report generation), 지칭 표현 이해(referring expression comprehension), 지칭 표현 생성(referring expression generation) 및 이미지 분류(image classification).

- **Performance Highlights**: Uni-Med는 connector에서의 multi-task 간섭을 해결하려는 첫 번째 노력으로, 다양한 구성에서 CMoE를 도입하여 평균 8%의 성능 향상을 validates합니다. 이전의 최신 의료 MLLM과 비교할 때, Uni-Med는 다양한 작업에서 경쟁력 있거나 더 우수한 평가 지표를 달성하였습니다.



### Learning Quantized Adaptive Conditions for Diffusion Models (https://arxiv.org/abs/2409.17487)
- **What's New**: 이 논문에서는 낮은 함수 평가 수(NFE)로 고품질 이미지를 생성하는 데 있어 ODE(Ordinary Differential Equation) 경로의 곡률을 줄이기 위한 새로운 접근 방식을 제안합니다. 적응 가능한 조건을 활용하여 경로 곡률을 효과적으로 감소시키며, 추가적인 훈련 매개변수는 1%에 불과합니다.

- **Technical Details**: 제안된 방법은 경로의 교차 정도를 줄이기 위해 적응형으로 학습된 양자화 조건을 사용합니다. 이것은 경로 재배치 없이도 가능하며, 실제로 샘플링을 가속화할 수 있습니다. 기존의 방법들은 최적화 과정을 통해 경로의 교차를 줄이려고 했으나, 이 과정에서 마진 노이즈 분포를 유지하는 데 어려움이 있었습니다. 이러한 문제를 해결하기 위해 제안된 방법이 사용됩니다.

- **Performance Highlights**: CIFAR-10에서 6 NFE로 5.14 FID를 달성하였으며, FFHQ 64x64에서 6.91 FID, AFHQv2에서 3.10 FID를 기록합니다. 이는 기존의 확산 모델에 비해 우수한 성능을 보여줍니다.



### Revisiting Deep Ensemble Uncertainty for Enhanced Medical Anomaly Detection (https://arxiv.org/abs/2409.17485)
Comments:
          Early accepted by MICCAI2024

- **What's New**: 의료 이상 탐지(Anomaly Detection) 방법론에 있어 새로운 접근 방식인 Diversified Dual-space Uncertainty Estimation(D2UE)을 제안하며, 이 방법은 Redundancy-Aware Repulsion(RAR) 및 Dual-Space Uncertainty(DSU)를 통해 정상 샘플에 대한 동의와 이상 샘플에 대한 이견을 균형있게 조정합니다.

- **Technical Details**: D2UE 프레임워크는 다수의 Autoencoder 구조를 가진 N개의 학습자로 구성되며, 이들은 정상 샘플에 대해 동의하고 이상 샘플에 대해 이견을 나타내도록 설계되었습니다. RAR는 여러 학습자가 보다 다양한 feature space에서 학습하도록 유도하며, DSU는 입력과 출력 공간에서의 불확실성을 결합하여 이상 지역을 강조합니다.

- **Performance Highlights**: 다양한 백본을 가진 5개의 의료 벤치마크에 대해 종합 평가를 실시했으며, 실험 결과 D2UE가 기존의 최첨단 방법들보다 우수한 성능을 보였고, 각 구성 요소의 효과iveness도 입증되었습니다.



### TFS-NeRF: Template-Free NeRF for Semantic 3D Reconstruction of Dynamic Scen (https://arxiv.org/abs/2409.17459)
Comments:
          Accepted in NeuRIPS 2024

- **What's New**: TFS-NeRF는 복잡한 상호작용을 가진 동적 장면에 대한 템플릿 프리(Template-free) 3D 시맨틱(semantic) NeRF를 도입하며, 희소 또는 단일 시점 RGB 비디오로부터 학습합니다. 이 방법은 기존 LBS (Linear Blend Skinning) 방식보다 훈련 시간을 단축시키고, 각 개체의 동작을 비텐트(Given) 분리합니다.

- **Technical Details**: 본 프레임워크는 INN(Invertible Neural Network)를 사용하여 LBS 예측을 단순화하고, 개별 개체의 스킨닝 가중치를 최적화하여 각각의 Signed Distance Field(SDF)를 정확하게 생성합니다. 세부적으로, 우리는 의미 기반(ray sampling)으로 샘플링하여 각 개체의 독립적인 변환을 학습합니다.

- **Performance Highlights**: 광범위한 실험을 통해 본 방법이 복잡한 상호작용 속에서 변형 가능한 물체와 비변형 가능한 물체 모두에 대해 높은 품질의 재구성을 생성하며, 기존 방법들에 비해 훈련 효율성도 향상되었다는 것을 보여주었습니다.



### CadVLM: Bridging Language and Vision in the Generation of Parametric CAD Sketches (https://arxiv.org/abs/2409.17457)
- **What's New**: 이 논문은 Parametric Computer-Aided Design (CAD) 분야에서 CAD 생성 작업을 위한 새로운 Vision Language 모델인 CadVLM을 제안합니다. 이는 기존의 CAD 모델링 방법의 한계를 극복하고, 스케치 이미지와 텍스트를 결합한 멀티모달 접근법을 적용하였습니다.

- **Technical Details**: CadVLM은 사전 학습된 모델을 활용하여 엔지니어링 스케치를 효과적으로 조작할 수 있는 엔드 투 엔드 모델입니다. 이 모델은 스케치 원시 시퀀스 및 스케치 이미지를 통합하여 CAD 자동완성(CAD autocompletion) 및 CAD 자동 제약(CAD autoconstraint)과 같은 다양한 CAD 스케치 생성 작업에서 뛰어난 성능을 발휘합니다.

- **Performance Highlights**: CadVLM은 SketchGraphs 데이터셋에서 CAD 자동완성 및 CAD 자동 제약 작업에서 우수한 성능을 보였으며, Entity Accuracy, Sketch Accuracy, CAD F1 score로 평가된 새로운 평가 지표를 소개하였습니다.



### AgMTR: Agent Mining Transformer for Few-shot Segmentation in Remote Sensing (https://arxiv.org/abs/2409.17453)
Comments:
          accepted to IJCV

- **What's New**: 본 연구에서는 원거리 센싱 시나리오에서의 일부샷 세분화(Few-shot Segmentation, FSS)를 위해 새로운 에이전트 마이닝 변환기(Agent Mining Transformer, AgMTR)를 제안한다. AgMTR는 피사체의 픽셀 수준 세분화 문제를 해결하기 위해 지역 인식 에이전트를 활용하여 의미적 상관관계를 구축한다.

- **Technical Details**: AgMTR는 에이전트 학습 인코더(Agent Learning Encoder, ALE), 에이전트 집계 디코더(Agent Aggregation Decoder, AAD), 의미 정렬 디코더(Semantic Alignment Decoder, SAD)의 세 가지 구성 요소로 구성된다. ALE는 지원 픽셀에서 클래스별 의미를 효율적으로 마이닝하고, AAD는 레이블이 없는 이미지로부터 유용한 의미를 탐색한다. SAD는 쿼리 이미지와 에이전트 간의 의미적 일관성을 촉진하여 세분화를 돕는다.

- **Performance Highlights**: iSAID 원거리 센싱 벤치마크에서 제안된 AgMTR 방법은 최첨단 성능을 달성하였으며, PASCAL-5i와 COCO-20i와 같은 자연 시나리오에서도 경쟁력 있는 결과를 유지하였다.



### Rejection Sampling IMLE: Designing Priors for Better Few-Shot Image Synthesis (https://arxiv.org/abs/2409.17439)
- **What's New**: 최근 연구는 한정된 훈련 데이터로 딥 생성 모델을 학습하는 분야에서의 진전을 보여주고 있습니다. 본 논문에서는 Implicit Maximum Likelihood Estimation (IMLE) 기술을 보다 발전시켜, 시험 시 데이타와 훈련 시 데이터 간의 불일치를 해결하는 RS-IMLE라는 새로운 접근법을 제안합니다.

- **Technical Details**: 기존 IMLE 방식은 훈련 시 선택한 잠재 코드와 검사 시 샘플링되는 잠재 코드 간에 불일치가 발생하는 문제는 유지했습니다. RS-IMLE는 훈련에 사용되는 사전 분포를 조정하여 이러한 불일치를 극복하고, 훈련 데이터와 더욱 유사한 분포를 가진 샘플을 선택하게끔 합니다.

- **Performance Highlights**: RS-IMLE를 사용한 결과, 아홉 개의 소수 데이터 이미지 세트에서 GAN 및 기존 IMLE 기반 방법들과 비교하여 평균 45.9%의 FID 가치를 감소시키며, 이미지 생성 품질이 크게 향상되었습니다.



### HazeSpace2M: A Dataset for Haze Aware Single Image Dehazing (https://arxiv.org/abs/2409.17432)
Comments:
          Accepted by ACM Multimedia 2024

- **What's New**: 본 연구에서는 HazeSpace2M 데이터셋을 소개합니다. 이 데이터셋은 200만 개 이상의 이미지로 구성되어 있으며, 각기 다른 형태의 안개를 분류하여 디하징(dehazing) 효과를 향상시키기 위해 설계되었습니다.

- **Technical Details**: HazeSpace2M 데이터셋은 10가지 안개 강도 레벨과 다양한 장면(Fog, Cloud, Environmental Haze (EH))을 포함하고 있습니다. 본 연구에서는 먼저 안개 유형을 분류한 후, 특정 유형에 대한 디하징 기법을 적용하는 접근 방식을 사용하였습니다. 전통적인 방법과는 다르게, 이 기술은 안개 유형에 따른 디하징을 적용하여 실제 환경에서의 이미지 선명도를 개선합니다.

- **Performance Highlights**: ResNet50과 AlexNet을 사용한 벤치마킹 결과, 기존 합성 데이터셋에 대비하여 각각 92.75% 및 92.50%의 정확도를 달성했습니다. 그러나 HazeSpace2M 데이터셋을 테스트했을 때, 모델들은 각각 80%와 70%의 정확도에 그쳤습니다. 추가 실험 결과, 안개 유형 분류 후 전문 디하저를 사용하면 PSNR에서 2.41%, SSIM에서 17.14%, MSE에서 10.2%의 성능 향상을 얻을 수 있었습니다. 또한, SOTA 모델을 통해 우리 프레임워크를 적용하면 성능이 크게 개선됨을 알 수 있었습니다.



### AgRegNet: A Deep Regression Network for Flower and Fruit Density Estimation, Localization, and Counting in Orchards (https://arxiv.org/abs/2409.17400)
- **What's New**: 이 논문에서는 농업 산업의 노동력과 비용 문제를 해결하기 위한 자동화된 꽃과 과일 밀도 추정 기술을 제안합니다. 특히, AgRegNet이라는 딥 회귀 기반 네트워크를 사용하여, 탐지(Object Detection)나 다각형 주석(Polygon Annotation) 없이 꽃과 과일의 밀도, 개수, 위치를 추정합니다.

- **Technical Details**: AgRegNet은 U-Net 아키텍처에서 영감을 받아 개발된 U자형 네트워크로, 인코더-디코더 스킵 커넥션을 포함하고 있습니다. ConvNeXt-T를 수정한 구조로 특징을 추출하며, 포인트 주석(Point Annotation) 정보를 바탕으로 학습되고, 세그멘테이션 정보 및 주의 모듈(Attention Modules)을 활용하여 relevant한 꽃과 과일의 특징을 강조합니다.

- **Performance Highlights**: 실험 결과, AgRegNet은 구조적 유사도 지수(Structural Similarity Index, SSIM), 백분율 평균 절대 오차(percentage Mean Absolute Error, pMAE), 평균 평균 정밀도(mean Average Precision, mAP) 측면에서 높은 정확도를 달성했습니다. 특히, 꽃 이미지의 SSIM은 0.938, pMAE는 13.7%, mAP는 0.81이며, 과일 이미지의 경우 SSIM은 0.910, pMAE는 5.6%, mAP는 0.93으로 나타났습니다.



### The Overfocusing Bias of Convolutional Neural Networks: A Saliency-Guided Regularization Approach (https://arxiv.org/abs/2409.17370)
- **What's New**: 본 논문에서는 Neural Networks(신경망)가 훈련 데이터가 제한적일 때 특정 이미지 영역에 초점을 맞추는 경향을 설명하고, 이러한 경향을 개선하기 위한 새로운 정규화 방법인 Saliency Guided Dropout(SGDrop)를 제안합니다.

- **Technical Details**: SGDrop은 attribution methods(어트리뷰션 방법)을 사용하여 훈련 중 가장 중요한 특징들의 영향을 줄이고, 신경망이 입력 이미지의 다양한 영역에 주의를 분산시키도록 유도합니다. SGDrop의 구현은 훈련 과정에서 중요한 특징을 선택적으로 삭제하는 방식으로 진행됩니다.

- **Performance Highlights**: 여러 비주얼 분류 벤치마크에서 SGDrop을 적용한 모델은 더 넓은 어트리뷰션과 신경 활동을 보여주었으며, 이는 입력 이미지의 전체적인 관점을 반영합니다. 또한, SGDrop을 통해 일반화 성능이 향상되는 것을 확인할 수 있었습니다.



### Improving satellite imagery segmentation using multiple Sentinel-2 revisits (https://arxiv.org/abs/2409.17363)
- **What's New**: 최근 원격 탐사(data) 데이터의 분석은 대규모 및 다양한 데이터 셋에서 사전 훈련된 공유 모델을 사용하는 컴퓨터 비전의 기술을 차용함으로써 큰 혜택을 보고 있습니다. 본 논문에서는 여러 번 촬영된 동일 위치의 이미지에 대한 특성을 반영하여 사전 훈련된 원격 탐사 모델의 미세 조정(fine-tuning)에서 이러한 재방문(revisit)의 최적 사용 방안을 탐구합니다.

- **Technical Details**: 연구의 초점은 기후 변화 완화(climate change mitigation)와 관련된 전력 변전소(segmentation) 문제로, 여러 다중 시간 입력(multi-temporal input) 방식과 다양한 모델 아키텍처를 통해 이미지를 조합하는 방법을 테스트했습니다. 실험 결과, 모델의 잠재 공간(latent space)에서 여러 번의 재방문 이미지를 결합하는 것이 다른 방법보다 우수하다는 것을 발견했습니다. 또한, SWIN Transformer 기반 아키텍처가 U-net 및 ViT 기반 모델보다 더 높은 성능을 보였습니다.

- **Performance Highlights**: 이 연구는 재방문 데이터를 활용한 전력 변전소 분할(task)에서 성능 향상이 두드러졌으며, 그 결과는 별도의 건물 밀도 추정(task)에서도 일반화 가능성을 검증했습니다. 전체적으로, 재방문을 사용한 단순하면서도 효과적인 접근 방법이 원격 탐사 커뮤니티에 귀중한 통찰을 제공함을 확인했습니다.



### A vision-based framework for human behavior understanding in industrial assembly lines (https://arxiv.org/abs/2409.17356)
- **What's New**: 본 논문은 자동차 도어 제조와 관련하여 산업 조립 라인에서 인간 행동을 캡처하고 이해하기 위한 비전 기반 프레임워크를 도입합니다. 이 프레임워크는 고급 컴퓨터 비전 기술을 활용하여 작업자의 위치와 3D 자세를 추정하고, 작업 자세, 행동 및 작업 진행 상황을 분석합니다. 주요 기여는 CarDA 데이터셋의 도입으로, 이는 인간 자세 및 행동 분석을 지원하기 위해 현실적인 환경에서 캡처한 도메인 관련 조립 동작을 포함합니다.

- **Technical Details**: 이 프레임워크는 상태-of-the-art 방식의 인간 자세 추정(human pose estimation), 인체공학적 자세 평가(ergonomic postural evaluation), 인간 행동 모니터링(human action monitoring)을 사용하여, 조립 과정 동안 인간 활동의 신체적·인체공학적·운영 측면을 모니터링하고 평가하는 데 중점을 둡니다. CarDA 데이터셋은 실시간 평가를 가능하게 하며, 실제 산업 설정에서 적용성과 효과성을 보장합니다.

- **Performance Highlights**: 실험 결과는 제안된 접근 방식이 작업자의 자세를 분류하는 데 효과적임을 보여주며, 조립 작업 진행 상황을 모니터링하는 데 강력한 성능을 발휘합니다.



### SeaSplat: Representing Underwater Scenes with 3D Gaussian Splatting and a Physically Grounded Image Formation Mod (https://arxiv.org/abs/2409.17345)
Comments:
          Project page here: this https URL

- **What's New**: SeaSplat은 3D radiance fields의 최근 발전을 활용하여 실시간 수중 장면 렌더링을 가능하게 하는 방법입니다. 이 방법은 물체와 수중 환경의 시각적 특성을 고려한 물리적으로 기반한 수중 이미지 형성 모델을 통해 3D Gaussian Splatting (3DGS)과 결합됩니다.

- **Technical Details**: SeaSplat은 물질의 매개변수와 기본 3D 표현을 동시에 학습함으로써 장면의 진정한 색상을 복원하고, 장면의 기하학을 더 정확하게 추정합니다. 이 방법은 SeaThru-NeRF 데이터셋에서 수집된 실제 수중 장면에도 적용되며, 시뮬레이션으로 저하된 실제 장면에서도 효과적입니다.

- **Performance Highlights**: SeaSplat을 통해 수중 매질이 있는 장면에서 새로운 시점에서의 렌더링 성능이 향상되었으며, 장면의 진정한 색상을 복원하고 렌더링을 매질의 존재 없이 복원할 수 있었습니다. 또한, 수중 이미지 형성이 장면 구조 학습을 도우며, 깊이 지도 품질을 향상시킵니다.



### Energy-Efficient & Real-Time Computer Vision with Intelligent Skipping via Reconfigurable CMOS Image Sensors (https://arxiv.org/abs/2409.17341)
Comments:
          Under review

- **What's New**: 이 논문은 CMOS 이미지 센서(CMOS image sensor, CIS) 시스템을 재설계하여 에너지 효율성을 개선하는 새로운 방법을 제시합니다. 이 시스템은 비 사건성이 있는 영역이나 행을 선택적으로 생략하여 에너지를 절약하며, 이는 센서의 읽기 단계에서 수행됩니다.

- **Technical Details**: 제안된 시스템은 비디오 프레임 과정에서 에너지를 절감할 수 있도록 고안된 커스텀 형식의 재구성 가능한 CIS를 사용합니다. 새로운 마스킹 알고리즘이 실시간으로 생략 과정을 지능적으로 안내하여, 자율주행 및 증강/가상 현실 응용에 적합합니다. 이 시스템은 또한 애플리케이션 필요에 따라 표준 모드로도 작동할 수 있습니다. 하드웨어 알고리즘 협업 프레임워크에서 BDD100K 및 ImageNetVID 기반의 객체 탐지와 OpenEDS 기반의 시선 추정에서 평가하여 최대 53%의 에너지 비용 절감 효과와 SOTA 정확도를 달성했습니다.

- **Performance Highlights**: 제안된 알고리즘-하드웨어 공동 설계 프레임워크는 자율 주행 및 AR/VR 응용에 대해 46% (행 단위 생략), 53% (행 단위 생략), 52% (영역 단위 생략) 에너지 효율성을 달성하며 SOTA 정확도를 유지합니다.



### Block Expanded DINORET: Adapting Natural Domain Foundation Models for Retinal Imaging Without Catastrophic Forgetting (https://arxiv.org/abs/2409.17332)
Comments:
this http URL, C. Merk and M. Buob contributed equally as shared-first authors. D. Cabrera DeBuc, M. D. Becker and G. M. Somfai contributed equally as senior authors for this work

- **What's New**: 이 연구에서는 자기 지도 학습(self-supervised learning)과 파라미터 효율적인 미세 조정(parameter-efficient fine-tuning)을 이용하여 망막 이미징(retinal imaging) 작업을 위한 두 가지 새로운 기초 모델인 DINORET와 BE DINORET을 개발하였습니다.

- **Technical Details**: DINOv2 비전 트랜스포머(vision transformer)를 적응하여 망막 이미징 분류 작업에 활용하였으며, 두 모델 모두 공개된 색상 안저 사진(color fundus photographs)을 사용하여 개발 및 미세 조정을 진행하였습니다. 우리는 블록 확장(block expansion)이라는 새로운 도메인 적응(domain adaptation) 전략을 도입하였습니다.

- **Performance Highlights**: DINORET과 BE DINORET은 망막 이미징 작업에서 경쟁력 있는 성능을 보여주었고, 블록 확장 모델이 대부분의 데이터 세트에서 최고 점수를 기록했습니다. 특히 DINORET과 BE DINORET은 데이터 효율성 측면에서 RETFound을 초과하며, 블록 확장이 재학습 시의 재난적 망각(catastrophic forgetting)을 성공적으로 완화했다는 점이 주목할 만합니다.



### ChatCam: Empowering Camera Control through Conversational AI (https://arxiv.org/abs/2409.17331)
Comments:
          Paper accepted to NeurIPS 2024

- **What's New**: 이 연구에서는 사용자가 자연어로 카메라를 제어할 수 있는 새로운 시스템 ChatCam을 소개합니다. 이를 통해 영상 제작 과정에서의 기술적 장벽이 낮아지고, 전문적인 촬영 기법을 쉽게 적용할 수 있습니다.

- **Technical Details**: ChatCam은 사용자의 요청을 이해하고, CineGPT라는 GPT 기반의 오토회귀 모델 및 Anchor Determinator를 활용하여 카메라 경로를 생성합니다. 이 모델은 텍스트-경로 쌍 데이터셋에 기반하여 훈련되어 텍스트 조건에 따른 경로 생성을 가능하게 합니다.

- **Performance Highlights**: 실험을 통해 ChatCam의 카메라 운영에 대한 복잡한 지시를 해석하고 실행하는 능력을 입증하였으며, 실제 제작 환경에서의 가능성이 높음을 보여주었습니다.



### VL4AD: Vision-Language Models Improve Pixel-wise Anomaly Detection (https://arxiv.org/abs/2409.17330)
Comments:
          27 pages, 9 figures, to be published in ECCV 2024 2nd Workshop on Vision-Centric Autonomous Driving (VCAD)

- **What's New**: 이 논문에서는 기존의 anomaly segmentation 방법의 한계를 극복하기 위해 Vision-Language (VL) 인코더를 통합한 새로운 접근법을 제안합니다. 이로 인해 비정상 클래스를 더 잘 인식할 수 있게 됩니다.

- **Technical Details**: 제안된 VL4AD 모델은 max-logit prompt ensembling 및 class-merging 전략을 포함하여 텍스트 프롬프트를 통해 데이터 및 훈련이 필요 없는 비정상 감지를 가능하게 합니다. 이를 통해 VL 모델의 일반화된 지식을 활용합니다.

- **Performance Highlights**: VL4AD 모델은 널리 사용되는 벤치마크 데이터셋에서 경쟁력 있는 성능을 보여주며, 픽셀 단위의 비정상 탐지에서 의의 있는 발전을 이룹니다.



### Bi-TTA: Bidirectional Test-Time Adapter for Remote Physiological Measuremen (https://arxiv.org/abs/2409.17316)
Comments:
          Project page: this https URL

- **What's New**: 본 연구에서는 Test-Time Adaptation (TTA) 기법을 rPPG(원격 광체적맥파측정) 분야에 처음으로 도입했습니다. 이는 사전 학습된 모델이 추론( inference ) 중에 목표 도메인에 적응할 수 있도록 하여, 개인 정보 보호 문제로 인해 소스 데이터나 라벨이 필요하지 않게 만듭니다.

- **Technical Details**: Bi-TTA는 두 가지 전문가 지식 기반의 자가 감독( self-supervised ) 형태를 활용하여 rPPG 모델을 조정합니다. 여기에는 '전향 적응'(prospective adaptation) 모듈과 '후향 안정화'(retrospective stabilization) 모듈이 포함되어 있습니다. 전향 적응 모듈은 불필요한 도메인 노이즈를 제거하여 안정성을 높이며, 후향 안정화 모듈은 모델 파라미터를 동적으로 강화하여 과적합이나 기억 상실을 방지합니다.

- **Performance Highlights**: Bi-TTA는 기존의 TTA 알고리즘과 비교하여 비약적으로 향상된 적응 능력을 보여주었으며, 실험 결과에서 우수한 성능을 입증했습니다. 대규모 벤치마크가 마련되었으며, 이는 향후 rPPG 연구에 기여할 것입니다.



### Navigating the Nuances: A Fine-grained Evaluation of Vision-Language Navigation (https://arxiv.org/abs/2409.17313)
Comments:
          EMNLP 2024 Findings; project page: this https URL

- **What's New**: 이 연구는 Vision-Language Navigation (VLN) 작업을 위한 새로운 평가 프레임워크를 제안합니다. 이 프레임워크는 다양한 지침 범주에 대한 현재 모델을 더 세분화된 수준에서 진단하는 것을 목표로 합니다. 특히, context-free grammar (CFG)를 기반으로 한 구조에서 VLN 작업의 지침 카테고리를 설계하고, 이를 Large-Language Models (LLMs)의 도움으로 반자동으로 구성합니다.

- **Technical Details**: 제안된 평가 프레임워크는 atom instruction, 즉 VLN 지침의 기본 행동에 집중합니다. CFG를 사용하여 지침의 구조를 체계적으로 구성하고 5개의 주요 지침 범주(방향 변화, 수직 이동, 랜드마크 인식, 영역 인식, 숫자 이해)를 정의합니다. 이 데이터를 활용하여 평가 데이터셋 NavNuances를 생성하고, 이를 통해 다양한 모델의 성능을 평가하며 문제가 드러나는 경우가 많았습니다.

- **Performance Highlights**: 실험 결과, 모델 간 성능 차이와 일반적인 문제점이 드러났습니다. LLM에 의해 강화된 제로샷 제어 에이전트가 전통적인 감독 학습 모델보다 방향 변화와 랜드마크 인식에서 더 높은 성능을 보였으며, 반면 기존 감독 접근 방식은 선택적 편향으로 인해 원자 개념 변화에 적응하는 데 어려움을 겪었습니다. 이러한 분석은 VLN 방식의 향후 발전에 중요한 통찰력을 제공합니다.



### Disco4D: Disentangled 4D Human Generation and Animation from a Single Imag (https://arxiv.org/abs/2409.17280)
- **What's New**: 이번 논문에서 제안하는 Disco4D는 단일 이미지에서 4D 사람 모델과 애니메이션을 생성하는 새로운 Gaussian Splatting 프레임워크입니다. Disco4D는 기존 방식과 달리 SMPL-X 모델을 사용하여 인체와 의류를 효과적으로 분리하여 생성 세부 정보와 유연성을 크게 향상시킵니다.

- **Technical Details**: Disco4D는 SMPL-X 모델을 사용하여 인체를 모델링하고 Gaussian 모델을 통해 의류를 독립적으로 모델링합니다. 주요 기술 혁신으로는 1) 의류 Gaussians의 효율적인 학습과 적합, 2) Diffusion 모델을 사용하여 시각적으로 보이지 않는 부분을 모델링, 3) 각 의류 Gaussian에 대한 아이덴티티 인코딩 학습 등이 있습니다.

- **Performance Highlights**: Disco4D는 4D 인간 생성 및 애니메이션 작업에서 우수한 성능을 입증했습니다. 의류와 신체의 독립적인 재구성 덕분에 정밀한 카테고리화 및 추출이 가능하며, 다양한 편집 기능을 지원합니다. 또한, SMPL-X 모델을 기반으로 한 애니메이션 기능 향상 덕분에 복잡한 몸동작에 대한 의류의 반응을 세밀하게 조정할 수 있습니다.



### Walker: Self-supervised Multiple Object Tracking by Walking on Temporal Appearance Graphs (https://arxiv.org/abs/2409.17221)
Comments:
          ECCV 2024

- **What's New**: Walker는 희소한 bounding box 주석과 추적 레이블 없이 비디오에서 학습하는 첫 번째 self-supervised tracker입니다. 이는 다수의 객체 추적(MOT) 방법에서 필수적인 주석 작업을 크게 줄여줍니다.

- **Technical Details**: Walker는 quasi-dense temporal object appearance graph를 설계하고, 그래프에서 랜덤 워크를 최적화하여 인스턴스 유사성을 학습하는 새로운 multi-positive contrastive objective를 제안합니다. 이 알고리즘은 그래프 내 인스턴스 간의 상호 배타적인 연결 속성을 강화하여 MOT를 위한 학습된 토폴로지를 최적화합니다.

- **Performance Highlights**: Walker는 MOT17, DanceTrack, BDD100K에서 경쟁력 있는 성능을 달성하였고, 이전의 self-supervised trackers를 초월하며 주석 요구 사항을 400배까지 줄이면서도 우수한 결과를 보였습니다.



### Neural Network Architecture Search Enabled Wide-Deep Learning (NAS-WD) for Spatially Heterogenous Property Awared Chicken Woody Breast Classification and Hardness Regression (https://arxiv.org/abs/2409.17210)
- **What's New**: 최근 몇 년간 빠른 성장률과 높은 육계 수확량을 위한 집중적인 유전자 선택으로 인해 전 세계 가금류 산업은 '우디 브레스트(woody breast)'라는 어려운 문제에 직면하고 있습니다. 본 연구에서는 hyperspectral imaging (HSI)과 machine learning 알고리즘을 결합해 우디 브레스트 상태를 비침습적이고 객관적이며 높은 처리량으로 평가할 수 있는 방법을 제시합니다.

- **Technical Details**: 본 연구는 250개의 생닭 가슴살 필렛(sample)을 수집하여 정상, 경증, 중증으로 분류하였고, HSI 처리 모델 설계 시 공간적으로 이질적인 경도 분포를 고려했습니다. 이 연구에서는 HSI를 통해 WB 수준을 분류하고 샘플 경도 데이터와의 상관관계를 찾기 위한 회귀(regression) 모델을 구축했습니다. 신경망 구조 검색(neural architecture search, NAS)을 통해 NAS-WD라는 광범위-깊이 신경망 모델을 개발하였으며, NAS를 통해 자동으로 네트워크 구조와 하이퍼파라미터를 최적화했습니다.

- **Performance Highlights**: NAS-WD는 95%의 전반적인 정확도로 세 가지 WB 수준을 분류할 수 있으며, 기존의 전통적인 머신 러닝 모델보다 성능이 우수합니다. 스펙트럼 데이터와 경도 간의 회귀 상관관계는 0.75로, 전통적인 회귀 모델보다 더 높은 성능을 보여줍니다.



### 2024 BRAVO Challenge Track 1 1st Place Report: Evaluating Robustness of Vision Foundation Models for Semantic Segmentation (https://arxiv.org/abs/2409.17208)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2409.15107

- **What's New**: 2024 BRAVO Challenge의 Track 1에서 Cityscapes 데이터셋으로 훈련된 모델을 제시하며, 다양한 out-of-distribution 데이터셋에서의 강건성을 평가합니다. 이 연구는 Vision Foundation Models (VFM)을 활용하여 DINOv2에 간단한 segmentation decoder를 붙여 전체 모델을 fine-tuning하여 우수한 성능을 보여줍니다.

- **Technical Details**: 이 연구에서는 DINOv2 VFM을 사용하여 semantic segmentation을 위한 pre-trained 모델을 fine-tuning합니다. 기본 구성에서 간단한 linear decoder를 사용하여 patch-level features를 segmentation logits로 변환합니다. 다양한 모델 크기, patch 크기, pre-training 전략 및 segmentation decoders를 실험하여 우리의 접근 방식의 효과를 평가합니다.

- **Performance Highlights**: 우리는 기존의 복잡한 모델들을 능가하여 이 챌린지에서 1위를 달성했습니다. 우리의 간단한 접근 방식이 어떻게 specialist 모델들보다 더 나은 성능을 내는지를 입증하며, 향후 연구에서 관심을 끌 수 있는 새로운 관찰도 제시합니다.



### AACLiteNet: A Lightweight Model for Detection of Fine-Grained Abdominal Aortic Calcification (https://arxiv.org/abs/2409.17203)
Comments:
          10 pages including references

- **What's New**: 이 연구에서는 경량화된 딥러닝 모델인 'AACLiteNet'을 제안하여, 고해상도의 Abdominal Aortic Calcification (AAC) 점수를 효과적으로 예측할 수 있도록 하였습니다. 기존의 자동화 모델들에 비해 정확도가 높고 메모리 사용량이 적어, 휴대용 컴퓨팅 장치에서 실용적으로 사용될 수 있는 특징이 있습니다.

- **Technical Details**: AACLiteNet은 경량화된 Convolutional Neural Network (CNN)과 효율적인 글로벌 주의 메커니즘을 통합하여 설계되었습니다. 이 모델은 소규모 데이터셋에서 학습할 수 있도록 최적화되어 있으며, 누적 및 세분화된 AAC 점수를 단일 헤드로 예측할 수 있습니다. 네트워크는 여러 단계의 2D convolution 작업을 통해 복잡한 특징 표현을 학습합니다. 특히, Depthwise Convolution (DWC) 기법을 채택하여 계산 비용과 메모리 사용량을 줄였습니다.

- **Performance Highlights**: AACLiteNet은 이전 모델에 비해 85.94%의 높아진 평균 정확도를 기록했으며, 이는 이전 최상의 모델인 81.98%에 비해 개선된 수치입니다. 또한, 계산 비용은 19.88배, 메모리 사용량은 2.26배 감소하여, 휴대용 기기에서 구현 가능함을 보여주었습니다.



### Cross Dataset Analysis and Network Architecture Repair for Autonomous Car Lane Detection (https://arxiv.org/abs/2409.17158)
- **What's New**: 본 연구에서는 자율주행 차량의 차선 인식 애플리케이션을 위한 교차 데이터세트 분석과 신경망 아키텍처 수리를 수행합니다. 제공된 아키텍처인 ERFCondLaneNet은 복잡한 형상의 차선을 탐지하는 데 어려움을 겪는 기존의 CondLaneNet을 개선한 것입니다.

- **Technical Details**: 이 연구에서 제안하는 ERFCondLaneNet은 CondLaneNet [liu_condlanenet_2021]과 ERFNet [romera_erfnet_2018]을 통합하여 만들어졌으며, Transfer Learning (TL) 프로토콜을 사용하여 두 가지 주요 차선 탐지 벤치마크인 CULane과 CurveLanes에서 테스트되었습니다. 이 기술은 성능을 유지하면서도 33% 적은 특징을 사용하여 모델 크기를 46% 줄였습니다.

- **Performance Highlights**: ERFCondLaneNet은 ResnetCondLaneNet과 비슷한 성능을 보이며, 이는 복잡한 지형을 가진 차선 탐지에서  충분한 정확성을 유지합니다. 학습 과정에서 기존 모델보다 적은 데이터로도 우수한 결과를 보여줍니다.



### An Art-centric perspective on AI-based content moderation of nudity (https://arxiv.org/abs/2409.17156)
Comments:
          To be published at the AI4VA (AI for Visual Arts) Workshop and Challenges at ECCV 2024

- **What's New**: 이번 연구에서는 예술적 나체 이미지에 대한 알고리즘 필터링 알고리즘의 성능을 분석하고, 성별 및 스타일 편향과 같은 기술적 한계를 발견했습니다. 또한, 예술적 나체 이미지 분류의 개선을 위해 다중 모달 제로샷 классифика는 방안을 제안하였습니다.

- **Technical Details**: 세 개의 공개된 NSFW (Not-Safe-For-Work) 이미지 분류 알고리즘을 사용하여 140개 이상의 예술적 나체 이미지의 성능을 비교했습니다. 우리는 최신 다중 모달 심층학습 모델(CLIp)을 활용하여 예술적 나체 분류를 향상시킬 것을 제안합니다.

- **Performance Highlights**: 실험 결과, 알고리즘은 예술적 나체 이미지를 음란물로 잘못 분류하는 등 높은 오검출률과 오탐률을 보였습니다. 제안된 다중 모달 접근법을 통해 성능이 크게 개선된다는 것을 확인했습니다.



### Robot See Robot Do: Imitating Articulated Object Manipulation with Monocular 4D Reconstruction (https://arxiv.org/abs/2409.18121)
Comments:
          CoRL 2024, Project page: this https URL

- **What's New**: 이 논문에서는 Robot See Robot Do (RSRD)라는 방법을 개발하여, 단일 모노큘러 RGB 인간 시연을 통해 가동 가능한 객체 조작을 모방할 수 있도록 로봇에게 학습 능력을 제공합니다. 이 방법은 단일 정적 다중 뷰 객체 스캔을 기반으로 하여, 로봇이 객체의 부품 움직임을 복구하고 이를 통해 객체의 경로를 모사할 수 있도록 합니다.

- **Technical Details**: RSRD는 4D Differentiable Part Models (4D-DPM)라는 방법을 제안하여, 모노큘러 비디오에서 3D 부품 움직임을 회복합니다. 이 접근 방식은 기능 필드를 활용하여 이터레이티브 최적화를 수행하며, 이는 기하학적 규제기를 사용하여 단일 비디오로부터 3D 움직임을 복구할 수 있게 합니다.

- **Performance Highlights**: RSRD는 10회의 실험에서 9개의 객체에서 평균 87%의 성공률을 기록했으며, 총 90회의 실험에서 최종 성공률이 60%에 도달했습니다. 이 모든 결과는 대규모 사전 훈련된 비전 모델로부터 얻은 기능 필드만을 통해 이루어졌습니다.



### EvMAPPER: High Altitude Orthomapping with Event Cameras (https://arxiv.org/abs/2409.18120)
Comments:
          7 pages, 7 figures

- **What's New**: 이번 연구는 이벤트 카메라를 사용하여 전통적인 CMOS 카메라의 한계를 극복하고, 어려운 조명 조건에서도 높은 해상도의 orthomosaic(정사영상)를 생성할 수 있는 새로운 접근법을 제시합니다.

- **Technical Details**: 연구팀은 고해상도의 동기화된 데이터 수집을 위한 하드웨어 및 소프트웨어 아키텍처를 개발하였으며, 이를 통해 이벤트 카메라 데이터를 기존의 orthomosaic 생성 도구와 통합하는 방법을 제안했습니다. 또한, 고속 비행 중의 어려운 조명 조건에서 촬영한 고해상도 이벤트 카메라 데이터셋을 공개했습니다.

- **Performance Highlights**: 이벤트 카메라를 사용한 orthomosaic 생성은 조명 조건에 구애받지 않으며, 기존의 RGB 카메라를 활용한 결과와 비교하여 좋은 성능을 보였습니다. 이러한 접근 방식은 향후 UAV(무인 항공기) 기반의 고해상도 이미지 생성에 중요한 방향성을 제시합니다.



### MALPOLON: A Framework for Deep Species Distribution Modeling (https://arxiv.org/abs/2409.18102)
- **What's New**: MALPOLON은 딥 종 분포 모델(Deep-SDM) 훈련 및 추론을 지원하는 새로운 프레임워크입니다. 사용자가 Python 언어에 대한 일반적인 지식만으로도 딥 러닝 방식의 SDM을 시험해볼 수 있도록 설계되었습니다.

- **Technical Details**: 이 프레임워크는 Python으로 작성되었으며 PyTorch 라이브러리를 기반으로 합니다. 모듈성이 뛰어나고, 사용자 맞춤형 데이터셋에 대한 신경망 훈련을 위한 버튼 클릭 예제가 제공됩니다. YAML 기반의 설정과 병렬 컴퓨팅, 다중 GPU 활용이 가능합니다.

- **Performance Highlights**: MALPOLON은 접근성을 높이고 성능 확장성을 지원하기 위해 open-source로 제공되며, GitHub와 PyPi에 배포되었습니다. 다양한 시나리오에서의 사용 예제 및 광범위한 문서화가 이루어져 있습니다.



### SKT: Integrating State-Aware Keypoint Trajectories with Vision-Language Models for Robotic Garment Manipulation (https://arxiv.org/abs/2409.18082)
- **What's New**: 이 연구에서는 다양한 의류 소목(garment categories)에 대해 단일 모델로 키포인트(keypoint) 예측을 개선하기 위해 비전-언어 모델(vision-language models, VLM)을 사용하는 통합 접근 방식을 제안합니다. 본 연구는 의류의 다양한 변형 상태를 관리하는 데 도움을 줄 수 있는 새로운 접근법을 제시합니다.

- **Technical Details**: 제안된 방법은 상태 인식 쌍 키포인트 생성(State-aware paired keypoint formation) 기법을 활용하여 다양한 의류 환경에 잘 일반화되는 상태 인식 키포인트 궤적(State-aware Keypoint Trajectories)을 생성합니다. 이를 통해 비주얼 시그널과 텍스트 설명을 함께 해석할 수 있으며, 고급 물리 시뮬레이터를 이용한 대규모 합성 데이터셋을 통해 훈련됩니다.

- **Performance Highlights**: 실험 결과, VLM 기반 방법이 키포인트 감지 정확성과 작업 성공률을 획기적으로 향상시켜 주목받았습니다. 이 연구는 VLM을 활용하여 장기적으로 홈 자동화 및 보조 로봇 분야에서의 폭넓은 응용 가능성을 제시하고 있습니다.



### PhoCoLens: Photorealistic and Consistent Reconstruction in Lensless Imaging (https://arxiv.org/abs/2409.17996)
Comments:
          NeurIPS 2024 Spotlight

- **What's New**: 이 논문에서는 기존의 렌즈 기반 시스템보다 크기, 무게 및 비용 측면에서 이점이 있는 렌즈 없는 카메라의 이미지 재구성을 개선하기 위해 새로운 두 단계 접근 방식을 소개합니다. 이 접근 방식은 일관성과 포토리얼리즘을 동시에 달성하는 데 중점을 두고 있습니다.

- **Technical Details**: 첫 번째 단계에서는 시공간 변동 표시 함수(Variation of Point Spread Function, PSF)에 적응하는 공간적으로 변하는 복원(deconvolution) 방법을 사용하여 저주파(low-frequency) 콘텐츠를 정확하게 재구성합니다. 두 번째 단계에서는 사전 훈련된 확산(diffusion) 모델에서 생성적 사전(Generative Prior)을 통합하여 고주파(high-frequency) 세부사항을 복원하며, 첫 번째 단계에서 복원된 저주파 콘텐츠에 조건을 부여하여 포토리얼리즘을 높입니다.

- **Performance Highlights**: 우리의 방법은 기존의 방법들과 비교하여 데이터 충실도(data fidelity)와 시각적 품질(visual quality) 사이의 우수한 균형을 달성했습니다. PhlatCam과 DiffuserCam의 두 가지 렌즈 없는 시스템에서 성능을 입증하며, 시각적 개선을 여러 평가 지표를 통해 보여주고 있습니다.



### The Hard Positive Truth about Vision-Language Compositionality (https://arxiv.org/abs/2409.17958)
Comments:
          ECCV 2024

- **What's New**: 이 논문에서는 CLIP와 같은 최첨단 비전-언어 모델의 조합 가능성(compositionality) 부족 문제를 다룹니다. 기존 벤치마크에서 힘들었던 점은 이러한 모델들이 하드 네거티브(hard negative)를 사용하여 개선된 성능을 과장해왔다는 것입니다. 이번 연구는 하드 포지티브(hard positive)를 포함한 대상에서 CLIP의 성능이 12.9% 감소하는 반면, 인간은 99%의 정확도를 보인다는 주장을 제기합니다.

- **Technical Details**: 저자들은 112,382개의 하드 네거티브 및 하드 포지티브 캡션으로 평가 데이터셋을 구축하였습니다. CLIP을 하드 네거티브로 파인튜닝(finetuning)할 경우 성능이 최대 38.7%까지 감소했으며, 하드 포지티브가 포함될 때의 효과를 분석하여 조합 가능한 성능을 개선할 수 있음을 밝혀냈습니다.

- **Performance Highlights**: CLIP 모델은 하드 네거티브로 훈련했을 때, 기존 벤치마크에서 성능이 개선되었음에도 불구하고, 하드 포지티브에 대한 성능이 동시에 저하되었습니다. 반면, 하드 네거티브와 하드 포지티브를 동시에 사용하여 훈련했을 때, 두 가지 모두 성능 개선이 이루어졌습니다. 이러한 연구는 조합 가능성에 대한 새로운 차원을 탐구하며 향후 연구 방향을 제시합니다.



### Visualization of Age Distributions as Elements of Medical Data-Stories (https://arxiv.org/abs/2409.17854)
Comments:
          11 pages, 7 figures

- **What's New**: 본 연구는 의학적 내러티브 시각화에서 질병의 연령 분포를 효과적으로 표현하기 위한 방법을 탐구합니다. 특히, 피코그램(pictogram) 변형의 효과를 비교 평가하여, 정보 이해도 및 미적 요소를 향상시키는 방법을 제시합니다.

- **Technical Details**: 이 연구는 3가지 피코그램 변형(바 형태의 피코그램, 스택 아래 피코그램, 주석)이 있는 18개의 시각화를 분석하였습니다. 총 72명의 참가자와 3명의 전문가 리뷰를 통해 평가를 진행했으며, 디자인 선택 기준(comprehension, aesthetics, engagement, and memorability)에 따라 결과를 도출하였습니다.

- **Performance Highlights**: 주석을 사용한 피코그램이 정보 이해도와 미적 요소에서 가장 효과적이었으나, 전통적인 바 차트는 참여도에서 선호되었습니다. 다양한 시각화 변형을 통한 사용자 기억력의 향상도 기록되었습니다.



### CASPFormer: Trajectory Prediction from BEV Images with Deformable Attention (https://arxiv.org/abs/2409.17790)
Comments:
          Under Review at ICPR 2024, Kolkata

- **What's New**: 이 논문에서는 고해상도 (High Definition, HD) 맵에 의존하지 않고 Bird-Eye-View (BEV) 이미지를 기반으로 다중 모드 모션 예측을 수행할 수 있는 Context Aware Scene Prediction Transformer (CASPFormer)를 제안합니다. 이는 자율 주행 및 운전 보조 시스템에 클라우드 서비스의 확장성을 제공할 수 있는 혁신적 방법입니다.

- **Technical Details**: CASPFormer는 래스터화된 BEV 이미지를 사용하여 다중 모드 벡터화된 궤적을 생성합니다. 이 시스템은 기존의 인식 모듈과 통합할 수 있으며, 포스트 프로세싱 없이 벡터화된 궤적을 직접 디코딩합니다. 디포머블 (Deformable) 어텐션 방식으로 궤적을 반복적으로 디코딩하며, 이는 컴퓨팅 효율성을 높이고 중요한 공간 위치에 초점을 맞출 수 있도록 합니다. 또한, 학습 가능한 모드 쿼리를 통합하여 다수의 씬-일관적인 궤적을 생성할 때 '모드 붕괴' 문제를 해결합니다.

- **Performance Highlights**: 우리의 모델은 nuScenes 데이터셋에서 평가되었으며, 여러 메트릭에서 최신 기술 수준의 성능을 달성했습니다. 특히, 다중 궤적 예측에서의 높은 정확도와 효율성을 보이며, 기존의 HD 맵 기반 방법들 대비 비용 효율적이고 확장 가능한 해결책을 제안합니다.



### LGFN: Lightweight Light Field Image Super-Resolution using Local Convolution Modulation and Global Attention Feature Extraction (https://arxiv.org/abs/2409.17759)
Comments:
          10 pages, 5 figures

- **What's New**: 본 논문에서는 가벼운 LF 이미지 초해상도(SR)를 위한 모델인 LGFN을 제안합니다. 이 모델은 서로 다른 관점의 로컬 및 글로벌 특성과 다양한 채널의 특성을 통합하여 성능을 향상시킵니다.

- **Technical Details**: LGFN은 CNN 기반의 특성 추출 모듈인 DGCE를 사용하여 로컬 특성을 추출하고, ESAM과 ECAM을 통해 글로벌 및 채널 특성을 효과적으로 추출합니다. 본 모델은 0.45M의 파라미터 수와 19.33G의 FLOPs를 가지며, 경쟁력 있는 성과를 달성하였습니다.

- **Performance Highlights**: LGFN 모델은 NTIRE2024 Light Field Super Resolution Challenge에서 Track 2 Fidelity & Efficiency 부문에서 2위, Track 1 Fidelity 부문에서 7위를 기록하였습니다.



### Robotic-CLIP: Fine-tuning CLIP on Action Data for Robotic Applications (https://arxiv.org/abs/2409.17727)
Comments:
          7 pages

- **What's New**: 본 논문에서는 Robotic-CLIP을 소개하여 로봇의 인식 능력을 강화하고자 합니다. Robotic-CLIP은 CLIP 모델을 기초로 하여 동작 데이터인 309,433개의 비디오(~740만 프레임)를 사용해 대조 학습(contrastive learning)을 통해 Fine-tuning되었습니다.

- **Technical Details**: Robotic-CLIP은 텍스트 지시와 비디오 프레임 간의 의미론적 정렬(semantic alignment)을 수행할 뿐만 아니라 비디오 내 동작을 효과적으로 포착하고 강조합니다. 두 개의 서로 다른 프레임을 사용하여 동작 관계를 더 잘 이해할 수 있도록 설계되었습니다.

- **Performance Highlights**: Robotic-CLIP은 다양한 언어 기반 로봇 작업에서 다른 CLIP 기반 모델들보다 뛰어난 성능을 보입니다. 실제 그리핑(grasping) 응용 프로그램에서도 실용적인 효과를 보여주었습니다.



### Explanation Bottleneck Models (https://arxiv.org/abs/2409.17663)
Comments:
          13 pages, 4 figures

- **What's New**: 이 논문은 사전 정의된 개념 세트에 의존하지 않고 입력에서 텍스트 설명을 생성할 수 있는 새로운 해석 가능한 심층 신경망 모델인 설명 병목 모델(XBMs)을 제안합니다.

- **Technical Details**: XBMs는 입력 데이터에서 텍스트 설명을 생성하고, 이를 기반으로 최종 작업 예측을 수행하는 모델입니다. XBMs는 사전 학습된 비전-언어 인코더-디코더 모델을 활용하여 입력 데이터에 나타난 개념을 포착합니다. 훈련 과정 중 '설명 증류(explanation distillation)' 기술을 사용하여 분류기의 성능과 텍스트 설명의 품질을 모두 확보합니다.

- **Performance Highlights**: 실험 결과, XBMs는 기존의 개념 병목 모델(CBMs)과 비교하여 더욱 유의미하고 자연스러운 언어 설명을 제공하며, 블랙박스 기준선 모델에 필적하는 성능을 달성하고 있습니다. 특히, XBMs는 테스트 정확도에서 CBMs을 크게 초월합니다.



### Provable Performance Guarantees of Copy Detection Patterns (https://arxiv.org/abs/2409.17649)
- **What's New**: 이번 연구에서는 Copy Detection Patterns (CDPs)의 인증 기술에 대한 새로운 이론적 프레임워크를 제시하여, 다양한 기준을 통해 CDP의 성능을 보장하는 방법을 모색합니다. 이를 통해 기존의 단순한 매트릭스에서 벗어나 보다 정교한 기법을 도입할 수 있게 됩니다.

- **Technical Details**: CDPs는 고해상도 인쇄나 레이저 각인 기술로 생성된 무작위 이진 패턴으로, 물리적 패턴과 디지털 템플릿을 비교하여 인증이 이루어집니다. 본 논문은 Hamming distance, cross-entropy, Neymann-Pearson likelihood ratio와 같은 통계적 테스트를 기반으로 한 세 가지 품질 측정 기준을 제안합니다.

- **Performance Highlights**: 새로운 접근 방식으로 CDP 인증의 신뢰성과 효과성을 향상시키기 위한 이론적 발견을 제공합니다. 특히, 복잡한 공격 기법에 대응하기 위해 다양한 결정 전략 및 융합 규칙을 고려하여 최적화 방법을 제시합니다.



### Diversity-Driven Synthesis: Enhancing Dataset Distillation through Directed Weight Adjustmen (https://arxiv.org/abs/2409.17612)
- **What's New**: 본 논문은 데이터 세트를 압축하면서도 핵심 특징을 보존하는 방법, 특히 데이터셋 디스틸레이션(dataset distillation)에 대한 새로운 접근 방식을 제안하고 있습니다. 기존 방법들이 각 합성 데이터 인스턴스를 개별적으로 생성하는 방식에 제한되어 있었다면, 이번 연구는 다양성을 높이기 위한 동적 및 방향성 가중치 조정 기술을 도입하여 합성 과정의 대표성과 다양성을 극대화합니다.

- **Technical Details**: 연구에서는 Batch Normalization (BN) 손실 내의 분산 정규화기(variance regularizer)가 합성 데이터의 다양성을 보장하는 핵심 요소임을 밝혔습니다. 반면, 평균 정규화기(mean regularizer)는 기대와 달리 다양성을 제약하는 역할을 하고 있습니다. 또한, 원본 데이터셋에서 단 하나의 감독(source of supervision) 역할을 하는 teacher 모델의 가중치를 동적으로 조정하는 메커니즘이 도입되어 있습니다.

- **Performance Highlights**: CIFAR, Tiny-ImageNet, ImageNet-1K 등 다양한 데이터셋을 대상으로 한 실험에서 제안된 방법이 높은 성능을 보였으며, 최소한의 계산 비용(<0.1%)으로도 매우 다양한 합성 데이터셋을 생성할 수 있음을 보여주었습니다. 이 연구는 데이터셋 디스틸레이션의 효율성을 높이는 데 기여할 것입니다.



### ZALM3: Zero-Shot Enhancement of Vision-Language Alignment via In-Context Information in Multi-Turn Multimodal Medical Dialogu (https://arxiv.org/abs/2409.17610)
- **What's New**: 이 논문에서는 ZALM3를 제안하여 다중 회의(Multi-turn) 다중 모달(Multimodal) 의료 대화에서 비전-언어 정합성(Vision-Language Alignment)을 향상시키는 Zero-shot 접근 방식을 소개합니다. 환자가 제공하는 저품질 이미지와 텍스트 간의 관계를 개선하기 위해 LLM을 활용하여 이전 대화 맥락에서 키워드를 요약하고 시각적 영역을 추출합니다.

- **Technical Details**: ZALM3는 이미지 이전의 텍스트 대화에서 정보를 추출하여 관련 지역(Regions of Interest, RoIs)을 확인하는 데 LLM과 비주얼 그라우딩 모델을 활용합니다. 이 접근 방식은 추가적인 주석 작업이나 모델 구조 변경 없이도 비전-언어 정합성을 개선할 수 있는 특징이 있습니다.

- **Performance Highlights**: 세 가지 다른 임상 부서에서 실시한 실험 결과, ZALM3는 통계적으로 유의한 효과를 입증하며, 기존의 간단한 승패 평가 기준에 대한 보다 정량적인 결과를 제공하는 새로운 평가 메트릭을 개발하였습니다.



### Let the Quantum Creep In: Designing Quantum Neural Network Models by Gradually Swapping Out Classical Components (https://arxiv.org/abs/2409.17583)
Comments:
          50 pages (including Appendix), many figures, accepted as a poster on QTML2024. Code available at this https URL

- **What's New**: 이 논문에서는 양자 신경망(Quantum Neural Network, QNN)의 구조적 한계를 극복하기 위해 고전적 신경망과 양자 신경망 사이의 점진적인 전환 전략, HybridNet을 제안합니다. 이는 정보 흐름을 유지하면서 고전적 신경망 레이어를 점진적으로 양자 레이어로 대체하는 프레임워크를 제공합니다.

- **Technical Details**: 제안된 HybridNet은 고전적 모델에서 양자 모델로의 전환을 통해 양자 구성이 신경망의 성능에 미치는 영향을 보다 면밀히 분석합니다. 연구에서는 FlippedQuanv3x3라는 새로운 양자 커널과 데이터 재업로드 회로(Data Reuploading Circuit)를 도입하여 고전적 선형 레이어와 동일한 입력 및 출력을 공유하는 양자 레이어를 구현합니다.

- **Performance Highlights**: MNIST, FashionMNIST, CIFAR-10 데이터셋에 대한 수치 실험을 통해 양자 구성요소의 체계적인 도입이 성능 변화에 미치는 영향을 분석했습니다. 연구 결과, 기존의 QNN 모델보다 더 효과적인 성능을 발휘할 수 있음을 발견하였습니다.



### Advancing Open-Set Domain Generalization Using Evidential Bi-Level Hardest Domain Scheduler (https://arxiv.org/abs/2409.17555)
Comments:
          Accepted to NeurIPS 2024. The source code will be available at this https URL

- **What's New**: 본 논문은 Open-Set Domain Generalization (OSDG) 문제를 다루며, 기존의 정해진 도메인 스케줄러와 비교하여 적응형 도메인 스케줄러의 효과를 제안합니다. 이를 통해 동적 환경에서의 데이터 배틀링에 대한 새로운 접근 방식을 제시합니다.

- **Technical Details**: 논문에서 제안하는 EBiL-HaDS (Evidential Bi-Level Hardest Domain Scheduler)는 도메인 신뢰도를 측정하는 방법을 사용하여 도메인 간의 프레젠테이션 순서를 동적으로 조정합니다. 이 방법은 follower network를 활용해 신뢰도를 평가하고, bi-level 최적화 기법을 사용하여 학습합니다.

- **Performance Highlights**: 실험 결과, EBiL-HaDS는 PACS, DigitsDG 및 OfficeHome 데이터셋에서 OSDG 성능을 크게 향상시키며, 기존의 임의적이거나 연속적인 도메인 스케줄링 방법보다 더 효과적인 성능 개선을 보였습니다.



### Robotic Environmental State Recognition with Pre-Trained Vision-Language Models and Black-Box Optimization (https://arxiv.org/abs/2409.17519)
Comments:
          Accepted at Advanced Robotics, website - this https URL

- **What's New**: 이번 연구에서는 로봇이 다양한 환경에서 자율적으로 탐색하고 작동하기 위해 필요로 하는 환경 상태 인식을 위한 새로운 방법을 제안합니다. 특히, 사전 훈련된 대규모 Vision-Language Models (VLMs)를 활용하여 환경 상태를 통합적으로 인식할 수 있는 방법을 개발하였습니다.

- **Technical Details**: VLM을 사용하여 Visual Question Answering (VQA) 및 Image-to-Text Retrieval (ITR) 작업을 수행합니다. 이를 통해 로봇은 문이 열려 있는지, 물이 흐르고 있는지와 같은 다양한 환경 상태를 인식할 수 있습니다. 또한, 블랙박스 최적화를 통해 적절한 텍스트를 선택하는 방식으로 인식 정확도를 향상시킬 수 있습니다.

- **Performance Highlights**: 실험을 통해 제안된 방법의 효과성을 입증하였고, Fetch라는 모바일 로봇에서 인식 행동에 적용하였습니다. 이 방법은 다양한 상태 인식을 가능하게 하며, 여러 개의 모델과 프로그램을 준비할 필요 없이 소스 코드 및 컴퓨터 자원의 관리를 용이하게 해줍니다.



### NeuroPath: A Neural Pathway Transformer for Joining the Dots of Human Connectomes (https://arxiv.org/abs/2409.17510)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 이 논문은 신경 이미지 데이터에서 구조적 연결성(Structural Connectivity, SC)과 기능적 연결성(Functional Connectivity, FC) 간의 결합 메커니즘에 대한 연구를 소개합니다. 특히 'NeuroPath'라는 새로운 생물학적 영감을 받은 딥 모델을 제안하여, SC와 FC의 쌍으로부터 복잡한 신경 구조의 특징 표현을 발견하고 인지 행동의 예측 및 질병 진단에 활용할 수 있습니다.

- **Technical Details**: NeuroPath 모델은 고차원 토폴로지의 표현 학습 문제로 구성된 SC-FC 결합 메커니즘을 다루며, 다중 헤드 자기 주의(multi-head self-attention) 메커니즘을 사용하여 SC와 FC의 쌍 그래프에서 다중 모달 특징 표현을 캡처합니다. 이를 통해 우리는 SC의 다양한 경로들이 FC를 지원하는 방식(예: cyclic loop)을 이해할 수 있게 됩니다.

- **Performance Highlights**: NeuroPath 지표는 HCP(인간 연결체 프로젝트) 및 UK Biobank와 같은 대규모 공개 데이터셋에서 검증되었으며, 기존의 최첨단 성능을 초과하여 인지 상태 예측과 질병 위험 진단에서 뛰어난 잠재력을 보였습니다.



### Shape-intensity knowledge distillation for robust medical image segmentation (https://arxiv.org/abs/2409.17503)
- **What's New**: 이 논문에서는 의료 이미지 분할을 위한 새로운 접근 방식을 제안합니다. 이는 shape-intensity prior 정보를 분할 네트워크에 통합하여 정확한 분할 결과를 얻는 것을 목표로 하고 있습니다.

- **Technical Details**: 제안된 방법은 teacher network에서 class-wise 평균화된 훈련 이미지를 사용하여 shape-intensity 정보를 추출한 후, knowledge distillation을 통해 이 정보를 student network에 전파합니다. 이 과정에서 student network는 추가적인 계산 비용 없이 shape-intensity 정보를 효과적으로 학습합니다.

- **Performance Highlights**: 다섯 가지 의료 이미지 분할 작업에서 실험한 결과, 제안된 Shape-Intensity Knowledge Distillation (SIKD)은 기존의 여러 baseline 모델들을 일관되게 개선하였으며, 특히 cross-dataset 일반화 능력이 향상되었습니다.



### Global-Local Medical SAM Adaptor Based on Full Adaption (https://arxiv.org/abs/2409.17486)
- **What's New**: 최근 시각 언어 모델(visual language models)인 Segment Anything Model (SAM)의 발전이 보편적인 의미 분할(universal semantic segmentation) 분야에서 큰 혁신을 가져왔습니다. 특히 Medical SAM adaptor (Med-SA)를 통해 의료 이미지 분할에 많은 도움을 주었습니다. 하지만 Med-SA는 부분적 적응(partial adaption) 방식으로 SAM을 미세 조정(fine-tunes)하여 개선의 여지가 있습니다.

- **Technical Details**: 이 논문에서는 전체 적응(full adaption)이 가능한 새로운 글로벌 의료 SAM 어댑터(GMed-SA)를 제안합니다. GMed-SA는 SAM을 전 세계적으로 적응할 수 있도록 설계되었습니다. 또한 GMed-SA와 Med-SA를 결합하여 글로벌-로컬 의료 SAM 어댑터(GLMed-SA)를 제안하며, SAM을 글로벌과 로컬 모두에 적응시킵니다.

- **Performance Highlights**: 우리는 도전적인 공개 2D 흑색종(segmentation dataset) 분할 데이터셋에서 GLMed-SA의 광범위한 실험을 진행했습니다. 결과는 GLMed-SA가 다양한 평가 메트릭(evaluation metrics)에서 여러 최첨단 의미 분할 방법들보다 뛰어난 성능을 발휘함을 보여주었으며, 우리의 방법의 우수성을 입증하였습니다.



### Study of Subjective and Objective Quality in Super-Resolution Enhanced Broadcast Images on a Novel SR-IQA Datas (https://arxiv.org/abs/2409.17451)
Comments:
          This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 이 연구에서는 저해상도 방송 콘텐츠의 슈퍼 해상도(Super-Resolution, SR) 이미지를 평가하기 위한 새로운 이미지 품질 평가(IQA) 데이터셋인 SREB(Super-Resolution Enhanced Broadcasting contents) 데이터셋을 소개합니다. 기존 데이터셋들과 달리 SREB는 원본 이미지의 고유 해상도를 그대로 사용하며, SR 이미지의 왜곡과 개선을 모두 고려하고 있습니다.

- **Technical Details**: SREB 데이터셋은 낮은 품질의 방송 콘텐츠 원본 이미지와 해당 SR 이미지, 그리고 품질 점수를 포함합니다. 주관적 품질 테스트를 수행하여 평균 의견 점수(Mean Opinion Score, MOS)를 도출하였으며, 다양한 SR 방법에 대한 주관적 품질 영향 요인을 분석하였습니다. 기존의 IQA 메트릭스의 성능을 비교 분석하여, 딥러닝 기반의 메트릭스가 더 우수한 성능을 보였음을 확인했습니다.

- **Performance Highlights**: SREB 데이터셋을 기반으로 한 연구 결과, 현재 IQA 메트릭스의 한계가 드러났으며, SR 이미지의 인식 품질을 더 잘 반영하는 IQA 메트릭스의 필요성이 강조되었습니다. 또한, 51명의 참여자로부터 420개의 주관적 품질 점수가 수집되어 실제 환경에서의 SR 이미지 평가를 지원합니다.



### Transient Adversarial 3D Projection Attacks on Object Detection in Autonomous Driving (https://arxiv.org/abs/2409.17403)
Comments:
          20 pages, 7 figures, SmartSP 2024

- **What's New**: 이 논문에서는 자율 주행 시나리오에서 물체 탐지를 타겟으로 한 새로운 적대적 3D 프로젝션 공격을 제안합니다. 기존의 고정된 적대적 패턴과 달리, 이 새로운 유형의 공격은 3D 표면에서의 일시적 수정과 같은 유연성을 제공합니다.

- **Technical Details**: 이 공격은 최적화 문제로 구성되며, 색상 매핑(color mapping) 및 기하변환 모델(geometric transformation models)을 결합하여 설계되었습니다. 특히, Thin Plate Spine (TPS) 알고리즘을 사용하여 2D 이미지를 3D 표면에 효과적으로 변형합니다.

- **Performance Highlights**: 실험 결과, YOLOv3 및 Mask R-CNN을 기준으로 한 공격의 성공률이 낮은 조도 조건에서 최대 100%에 달하는 것으로 나타났습니다. 이는 실제 자율주행 상황에서의 치명적인 결과를 초래할 수 있는 공격의 효과를 강조합니다.



### Data-efficient Trajectory Prediction via Coreset Selection (https://arxiv.org/abs/2409.17385)
- **What's New**: 이 논문에서는 복잡한 주행 시나리오(시나리오)에서 데이터 부족 문제와 과대표현된 주행 시나리오로 인한 데이터 중복 문제를 완화하기 위해 새로운 데이터 효율적인 훈련 방법인 'coreset selection'을 제안합니다. 이 방법은 다양한 난이도의 시나리오 간 비율을 조절하면서 중요 데이터를 선택하여, 훈련 성능을 유지하면서 데이터 용량을 줄입니다.

- **Technical Details**: 이 방법은 데이터셋의 난이도 수준에 따라 데이터를 그룹화하고, 각 샘플에 대한 하위 모듈러 이득(submodular gain)을 계산하여 가장 가치 있는 데이터를 선택합니다. 두 가지 선택 방법인 'Fixed Selection'과 'Balanced Selection'을 통해 데이터 분포를 조절하며, 특히 Balnced Selection 방법은 복잡한 시나리오에서 유리한 성과를 보여줍니다. 또한, coresets는 일반화 능력이 뛰어나고 다양한 모델에 대해 테스트되었습니다.

- **Performance Highlights**: Fixed Selection 방법을 이용한 coresets는 전체 데이터셋의 50%만으로도 성능 저하 없이 비슷한 결과를 보여주었으며, Balanced Selection 방법은 더 복잡한 주행 시나리오에서 현저한 성과를 기록했습니다. 선택된 coresets는 SOTA 모델인 HPNet에서도 유사한 성능을 발휘하여, 모델의 다양한 난이도 시나리오에서의 일반화 능력을 강화했습니다.



### Optical Lens Attack on Deep Learning Based Monocular Depth Estimation (https://arxiv.org/abs/2409.17376)
Comments:
          26 pages, 13 figures, SecureComm 2024

- **What's New**: 이 논문에서는 자율 주행 시스템에서 사용되는 단안 깊이 추정(MDE) 알고리즘의 보안 위험을 조사하고, LensAttack이라는 새로운 물리적 공격 기법을 제안합니다. 이 공격은 카메라에 광학 렌즈를 전략적으로 배치하여 물체의 깊이 인식을 조작합니다.

- **Technical Details**: LensAttack은 두 가지 공격 형식을 포함합니다: 오목 렌즈 공격(concave lens attack)과 볼록 렌즈 공격(convex lens attack)으로, 각기 다른 렌즈를 사용하여 잘못된 깊이 인식을 유도합니다. 두 렌즈의 배치는 물체의 크기를 변경하여 깊이 추정 결과에 영향을 미치며, 이를 수학적 모델로 구축하여 다양한 공격 매개변수를 고려합니다.

- **Performance Highlights**: 시뮬레이션 및 실험을 통해 LensAttack의 효과성을 입증했으며, 세 가지 최신 MDE 모델에 대해 11.48%와 29.84%의 평균 에러율을 기록했습니다. 이러한 결과는 자율 주행 시스템의 깊이 추정 정확도에 대한 LensAttack의 중대한 영향을 강조합니다.



### Implicit Neural Representations for Simultaneous Reduction and Continuous Reconstruction of Multi-Altitude Climate Data (https://arxiv.org/abs/2409.17367)
Comments:
          arXiv admin note: text overlap with arXiv:2401.16936

- **What's New**: 이번 논문에서는 다중 고도 풍속 데이터 분석 및 저장을 위한 심층 학습 프레임워크인 GEI-LIIF를 제안합니다. 이 프레임워크는 차원 축소(dimensionality reduction), 교차 모드 예측(cross-modal prediction), 슈퍼 해상도(super-resolution)를 동시에 지원하여 기존의 방법론보다 우수한 성능을 보입니다.

- **Technical Details**: GEI-LIIF는 고해상도 데이터의 효과적인 회복을 위해 임시 신경망(implicit neural networks)을 사용합니다. 이 접근 방식은 고해상도 풍속 데이터를 저해상도로 축소한 후, 연속적인 슈퍼 해상도 표현을 학습하는 방식으로 작동합니다. 구체적으로, 별도의 입력 모드에 구애받지 않는 모드 특정 저차원 표현을 학습하기 위한 새로운 잠재 손실 함수(latent loss function)를 제안합니다.

- **Performance Highlights**: 제안된 방법은 슈퍼 해상도 품질 및 압축 효율성(compression efficiency) 면에서 기존의 방법들을 초월한다고 입증되었습니다. 실험 결과, 기후 변화 분석 및 풍력 에너지 최적화에 유용한 다중 모드 형태로 풍속 패턴을 추정할 수 있는 가능성을 보여줍니다.



### Multi-scale decomposition of sea surface height snapshots using machine learning (https://arxiv.org/abs/2409.17354)
- **What's New**: 이번 연구는 고해상도 Sea Surface Height (SSH) 데이터를 사용하여 균형 운동 (BM)과 비균형 운동 (UBM)으로 SSH를 분해하는 새로운 기법을 제안합니다. 특히, ZCA 흰색화 (whitening) 기술과 데이터 증강 (data augmentation)을 통해 다양한 공간 스케일에서의 분해 문제를 해결하려고 합니다.

- **Technical Details**: 연구에서는 ZCA 변환을 통해 UBM을 처리하기 전에 흰색화하여 여러 스케일에서의 정보 증가를 꾀하며, 전통적인 방법에 비해 훈련 안정성 및 계산 효율성을 개선합니다. 입력 SSH 데이터는 Agulhas retroflection 지역의 고해상도 글로벌 해양 시뮬레이션 데이터에서 가져왔고, 다양한 데이터를 처리하기 위해 회전 증강 및 합성 샘플 생성을 사용했습니다.

- **Performance Highlights**: 연구 결과, 제안된 기법은 기존의 딥러닝 모델에 비해 다중 스케일 데이터 처리에서 더 나은 성능을 보여주었으며, BM과 UBM의 정확한 분해를 가능하게 했습니다. 특히, ZCA 흰색화 기법이 훈련 안정성과 모델의 일반화 능력에 긍정적인 영향을 미쳤음을 증명했습니다.



### An Integrated Deep Learning Framework for Effective Brain Tumor Localization, Segmentation, and Classification from Magnetic Resonance Images (https://arxiv.org/abs/2409.17273)
Comments:
          36 pages, 27 figures, 5 tables

- **What's New**: 본 연구는 자기공명영상(MRI)을 바탕으로 뇌종양의 조기 진단을 위한 딥러닝(DL) 프레임워크를 제안합니다. 특히 이 연구에서는 뇌신경에서 발생하는 다양한 형태의 종양을 정확하게 국소화(localize)하고 분할(segment)하며 등급을 분류(classify) 할 수 있는 방법론을 다룹니다.

- **Technical Details**: 링크넷(LinkNet) 프레임워크를 VGG19에서 영감을 받은 인코더 아키텍처로 개선하여 멀티모달(multi-modal) 종양 특징 추출을 향상시켰으며, 공간 및 그래프 주의 메커니즘(spatial and graph attention mechanisms)을 통해 특징 강조 및 상호 특징 관계를 정제합니다. 이후, 세레즈넷101(SeResNet101) CNN 모델을 인코더 백본으로 통합하여 종양을 분할하였으며, 이로 인해 96% IoU 점수를 달성하였습니다. 분할된 종양을 분류하기 위해 세레즈넷152(SeResNet152) 특징 추출기와 적응형 부스팅(classifier)을 결합하여 98.53%의 정확도를 실현하였습니다.

- **Performance Highlights**: 제안된 모델들은 뚜렷한 성과를 보이며, 의료 AI의 발전을 통해 조기 진단을 가능하게 하고 환자에 대한 보다 정확한 치료 옵션을 제공할 가능성을 지니고 있습니다.



### AIM 2024 Challenge on Efficient Video Super-Resolution for AV1 Compressed Conten (https://arxiv.org/abs/2409.17256)
Comments:
          European Conference on Computer Vision (ECCV) 2024 - Advances in Image Manipulation (AIM)

- **What's New**: 본 연구에서는 비디오 초해상도(Video Super-Resolution, VSR) 문제를 해결하기 위한 새로운 실시간 프레임워크를 제안합니다. 기존의 VSR 방법론들은 높은 계산 요구사항으로 인해 저FPS 및 낮은 전력 효율을 초래하는데, 이 연구에서는 퍼포먼스를 높이면서도 실제 사용 환경에 적합한 최적화를 제공합니다.

- **Technical Details**: 제안된 두 가지 애플리케이션(540p에서 4K로의 확대 및 360p에서 1080p로의 확대) 각각에 대해 4K 해상도의 고품질 테스트 세트를 사용하여 최신 비디오 코덱인 AV1을 통해 비디오를 압축했습니다. 모든 제안된 방법은 각 프레임을 독립적으로 처리하여 효율성을 높입니다.

- **Performance Highlights**: 제안된 방법들이 기존의 침 인터폴레이션 기법들보다 VMAF 및 PSNR 개선을 이루었으며, 대부분의 솔루션은 150K 이하의 파라미터를 가지고 250 GMACs 미만의 연산량으로 24-30FPS의 실시간 초해상도 처리가 가능합니다.



### MODEL&CO: Exoplanet detection in angular differential imaging by learning across multiple observations (https://arxiv.org/abs/2409.17178)
- **What's New**: 이 연구에서는 태양계 외부 행성(exoplanets)의 직접 이미징(direct imaging)에서 발생하는 신호 간섭(nuisance model)을 구축하기 위한 새로운 방법을 제안합니다. 기존의 관측에 의존하는 방법이 아닌, 슈퍼바이즈드 딥 러닝(supervised deep learning) 기술을 활용하여 다수의 관측 아카이브를 통해 모델링하는 접근 방식을 사용합니다.

- **Technical Details**: 제안된 접근법은 신호를 재구성하는 과제로 변환하고 데이터의 두 가지 보완적 표현으로부터 간섭의 구조를 캡처합니다. 기존의 참조 차별 이미징(reference differential imaging) 접근 방식과 달리, 제안된 모델은 고차 비선형적이며 명시적인 이미지 간 유사성 측정 및 차감 과정을 사용하지 않습니다. 또한 학습 가능한 공간 특성의 통계적 모델링을 포함하여 탐지 민감도(detection sensitivity)와 이질적 데이터에 대한 강인성을 향상시킵니다.

- **Performance Highlights**: VLT/SPHERE 기기의 여러 데이터 세트에 이 알고리즘을 적용한 결과, PACO 알고리즘과 비교하여 우수한 정밀도-재현율(precision-recall) 균형을 보여줍니다. 특히 ADI에 의해 유도되는 다양성이 가장 제한적일 때, 제안된 접근 방식이 다수의 관측을 통해 정보를 학습할 수 있는 능력을 지원합니다.



### Gaussian Deja-vu: Creating Controllable 3D Gaussian Head-Avatars with Enhanced Generalization and Personalization Abilities (https://arxiv.org/abs/2409.16147)
Comments:
          11 pages, Accepted by WACV 2025 in Round 1

- **What's New**: 최근 3D Gaussian Splatting (3DGS)의 발전으로 3D 헤드 아바타 모델링의 잠재력이 크게 향상되었습니다. 본 논문에서는 'Gaussian Déjà-vu' 프레임워크를 통해 보다 신속하게 개인화된 3DGS 기반 헤드 아바타를 생성하는 방법을 제안합니다.

- **Technical Details**: Gaussian Déjà-vu 프레임워크는 일반화된 3D 아바타 모델을 훈련한 후, 단일 이미지와 단안 비디오를 통해 개인화하는 과정을 포함합니다. 초기 3D Gaussian 헤드 모델은 대규모 2D 이미지 데이터셋에서 훈련되었습니다. 제안된 방법은 learnable expression-aware rectification blendmaps를 통해 3D Gaussian의 초기값을 수정합니다.

- **Performance Highlights**: 제안된 방법은 기존 3D Gaussian 헤드 아바타 제작 방법에 비해 포토리얼리스틱 품질이 우수하며, 훈련 시간 소비를 최소 1/4로 줄여 몇 분 안에 아바타를 생성합니다.



### Vision-Language Models Assisted Unsupervised Video Anomaly Detection (https://arxiv.org/abs/2409.14109)
- **What's New**: 본 논문에서는 VLAVAD (Video-Language Models Assisted Anomaly Detection)라는 새로운 비디오 이상 탐지 방법을 제안합니다. 이는 이전의 방법들의 한계를 극복하고 대규모 언어 모델의 추론 능력을 활용합니다.

- **Technical Details**: VLAVAD는 Selective-Prompt Adapter (SPA)를 사용하여 의미 공간을 선택하고, Sequence State Space Module (S3M)을 통해 의미 특징의 시간적 일관성을 감지합니다. 고차원 비주얼 특징을 저차원 의미 특징으로 매핑함으로써 비지도 학습에서의 해석 가능성을 크게 향상시킵니다.

- **Performance Highlights**: 제안된 방법은 ShanghaiTech 데이터셋에서의 비디오 이상 탐지 성능이 SOTA(State of the Art)에 도달하여, 탐지하기 어려운 이상을 효과적으로 처리합니다.



New uploads on arXiv(cs.AI)

### Infer Human's Intentions Before Following Natural Language Instructions (https://arxiv.org/abs/2409.18073)
- **What's New**: 이 논문에서는 Ambiguous natural language instructions를 효과적으로 수행하기 위한 새로운 프레임워크인 Follow Instructions with Social and Embodied Reasoning (FISER)를 제안합니다. 이 프레임워크는 사람의 목표와 의도를 추론하는 단계를 명시적으로 포함합니다. 이는 AI 에이전트가 인간의 자연어 명령을 이해하고 협력적 작업을 더 잘 수행하도록 돕습니다.

- **Technical Details**: FISER는 두 가지 주요 구성 요소인 social reasoning과 embodied reasoning을 사용하여 모델이 인간의 의도를 명시적으로 추론할 수 있도록 합니다. 먼저 social reasoning을 통해 인간이 요청하는 하위 작업(sub-task)을 예측한 후, 이러한 지침을 로봇이 이해할 수 있는 작업으로 변환하는 Embodied reasoning 단계로 진행합니다. 또한, FISER는 계획 인식(plan recognition) 단계를 추가하여 인간의 전반적인 계획을 추론하는 데 도움을 줍니다.

- **Performance Highlights**: FISER 모델은 HandMeThat(HMT) 벤치마크에서 64.5%의 성공률을 기록하며, 이전의 end-to-end 접근법을 초월한 성능을 보여줍니다. 또한, FISER는 체인 오브 토트(Chain-of-Thought) 방식으로 GPT-4를 기반으로 한 강력한 기준과 비교했을 때도 우수한 결과를 도출했습니다. 이 결과는 인간의 의도에 대한 중간 추론을 명시적으로 수행하는 것이 AI의 성능을 개선하는 데 효과적임을 입증합니다.



### Explaining Explaining (https://arxiv.org/abs/2409.18052)
- **What's New**: 본 논문은 기계 학습 시스템의 블랙 박스 문제를 해결하기 위해 설명가능한 인공지능(XAI) 및 인간 중심의 설명가능한 인공지능(HCXAI) 접근 방식을 결합하는 하이브리드(cognitive agents) 방법을 제안합니다.

- **Technical Details**: 제안된 접근 방식은 지식 기반 인프라를 활용하며, 상황에 따라 기계 학습을 통해 획득한 데이터를 보조적으로 사용합니다. 이로 인해 인공지능 시스템이 사용자에게 더욱 필요한 설명을 제공할 수 있습니다.

- **Performance Highlights**: 시뮬레이션된 로봇 팀이 사용자의 지시에 따라 협력적인 검색 작업을 수행하는 시연 시스템의 예를 통해 이러한 에이전트의 설명 가능성을 보여줍니다.



### Compositional Hardness of Code in Large Language Models -- A Probabilistic Perspectiv (https://arxiv.org/abs/2409.18028)
- **What's New**: 본 연구는 큰 언어 모델(LLM)이 동일한 컨텍스트 내에서 여러 하위 작업을 수행하는 능력에 한계가 있음을 지적하고, 이를 극복하기 위해 멀티 에이전트 시스템을 활용한 문제 해결 방안을 제시합니다.

- **Technical Details**: 본 연구에서는 생성 복잡도(Generation Complexity)라는 지표로 LLM이 정확한 솔루션을 샘플링하기 위해 필요한 생성 횟수를 정량화 하며, 복합 코딩 문제에서 LLM과 멀티 에이전트 시스템 간의 성능 차이를 분석합니다. LLM을 오토회귀 모델로 모델링하고, 두 개의 다른 문제를 독립적으로 해결하는 방법을 논의합니다.

- **Performance Highlights**: 실험적으로, Llama 3 모델을 사용하여 복합 코드 문제에 대한 생성 복잡도의 기하급수적 차이를 입증하였으며, 이는 동일한 컨텍스트 내에서 문제를 해결하기에 LLM이 더 어려움을 겪는 다는 것을 보여줍니다.



### Role-RL: Online Long-Context Processing with Role Reinforcement Learning for Distinct LLMs in Their Optimal Roles (https://arxiv.org/abs/2409.18014)
- **What's New**: 이번 논문에서는 자동 뉴스 보도, 라이브 전자상거래 및 바이럴 짧은 동영상과 같이 다양한 스트리밍 미디어에서 정보를 수신하고 정리하는 과정에서 발생하는 무제한 길이의 문서를 처리하기 위한 새로운 패러다임, 즉 Online Long-context Processing (OLP)을 제안합니다.

- **Technical Details**: 제안된 OLP는 다양한 LLM (Large Language Models)을 효율적으로 활용할 수 있는 방법론으로, Role Reinforcement Learning (Role-RL) 기법을 통해 여러 LLM을 성능에 맞게 OLP 파이프라인 내에서 자동으로 배치하는 메커니즘을 제공합니다. 이 프레임워크는 각 모델의 성능에 따라 역할을 분담하므로 최적의 성능을 달성할 수 있습니다.

- **Performance Highlights**: OLP-MINI 데이터셋에 대한 광범위한 실험 결과, Role-RL 프레임워크를 적용한 OLP는 평균 재현율(Recall Rate) 93.2%를 달성하며, LLM 비용을 79.4% 절감할 수 있음을 보여주었습니다.



### CRoP: Context-wise Robust Static Human-Sensing Personalization (https://arxiv.org/abs/2409.17994)
Comments:
          31 pages, 10 figues and 13 tables

- **What's New**: 이 연구에서는 CRoP라는 새로운 정적 개인화 접근 방식을 소개합니다. 이는 사전 학습된 모델을 활용하고, 모델 프루닝을 통해 개인화 및 일반화 성능을 최적화합니다.

- **Technical Details**: CRoP는 오프라인에서 구입한 사전 훈련 모델을 활용하여, 개별 사용자를 위해 개발된 모델의 일반적인 특성을 고려합니다. 이는 Gradient Inner Product 분석 및 다양한 데이터 세트를 통해 설계 선택을 정당화합니다.

- **Performance Highlights**: CRoP는 네 가지 인간 감지 데이터 세트에 대해 개인화 효과성과 사용자 내 강건성을 보여주었습니다. 이를 통해 임상 환경에서 노출이 적은 데이터로도 모델의 성능을 극대화할 수 있습니다.



### Enhancing elusive clues in knowledge learning by contrasting attention of language models (https://arxiv.org/abs/2409.17954)
Comments:
          7 pages and 17 figures

- **What's New**: 이 논문은 대규모 언어 모델들이 지식 학습의 효율성을 높이는 방법에 대한 새로운 접근법을 제안합니다. 특히, 언어 모델의 주의(attention) 가중치를 비교하여 중요한 힌트를 식별하고, 이를 데이터 augmentation 방식으로 활용하여 성능을 향상시키는 방법을 소개합니다.

- **Technical Details**: 이 연구에서는 대규모 언어 모델과 소규모 언어 모델 간의 주의 가중치 차이를 분석하여, 작은 모델들이 간과하기 쉬운 중요한 단서를 포착합니다. 이 단서들은 `token-dropout data augmentation` 기법을 통해 강조되며, 사실 기억력에서 성능 향상을 보여줍니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 소규모 및 대규모 모델 모두에서 효과적으로 사실 기반의 지식 학습 성능을 향상시켰습니다. 기존의 데이터 augmentation 방법들과 비교할 때, 새로운 접근법이 모든 모델에 걸쳐 우수한 성능을 보였습니다.



### Navigation in a simplified Urban Flow through Deep Reinforcement Learning (https://arxiv.org/abs/2409.17922)
- **What's New**: 최근 도심 환경에서의 UAV(무인 항공기) 사용 증가로 인해 에너지 효율성과 소음 감소를 위한 새로운 비행 계획 최적화 전략이 필요해졌습니다. 이 논문은 DRL(Deep Reinforcement Learning) 알고리즘을 개발하여 UAV의 자율 내비게이션을 가능하게 하는 방법을 제안합니다.

- **Technical Details**: 논문에서는 두 차원 유동장(flow field)에서 UAV의 비행 경로를 최적화하기 위해 PPO+LSTM 셀을 사용하여 알고리즘을 구현하였습니다. 이 방식은 단순한 PPO 및 TD3 알고리즘에 비해 비행 경로 최적화와 상관하여 상당한 개선을 보여주었습니다.

- **Performance Highlights**: PPO+LSTM로 훈련된 정책은 98.7% 의 성공률(SR)과 0.1% 의 충돌률(CR)을 기록하였으며, 이는 기존의 PPO(75.6% SR, 18.6% CR)와 TD3(77.4% SR, 14.5% CR) 알고리즘을 능가하는 성과입니다.



### Learning to Love Edge Cases in Formative Math Assessment: Using the AMMORE Dataset and Chain-of-Thought Prompting to Improve Grading Accuracy (https://arxiv.org/abs/2409.17904)
- **What's New**: 이번 논문에서는 아프리카 여러 국가의 학생들이 사용하는 학습 플랫폼 Rori에서 수집한 53,000개의 수학 개방형 질문-답변 쌍으로 이루어진 새로운 데이터세트 AMMORE를 소개합니다.

- **Technical Details**: AMMORE 데이터세트는 학생들의 수학 성취도를 연구하기 위한 중요한 자원으로, 두 가지 실험을 통해 대규모 언어 모델(LLM)을 사용하여 학생의 도전적인 답변을 채점하는 방법을 평가했습니다. 실험 1에서는 zero-shot, few-shot 및 chain-of-thought prompting을 포함한 다양한 LLM 기반 접근 방식을 사용하여 규칙 기반 분류기가 정확하게 채점하지 못한 학생 답변의 1%를 채점했습니다. 실험 2에서는 최상의 LLM 기반 접근 방식에서 생성된 점수를 Bayesian Knowledge Tracing (BKT) 모델에 전달하여 학생의 특정 수업 마스터리를 추정했습니다.

- **Performance Highlights**: 체인 오브 씽킹(prompting)은 가장 잘 수행된 접근 방식으로, 엣지 케이스의 92%를 정확하게 채점하여 채점 정확도를 98.7%에서 99.9%로 향상시켰습니다. LLM 체인 오브 씽킹 접근 방식을 사용했을 때, 학생의 마스터리 상태를 잘못 분류한 비율이 6.9%에서 2.6%로 감소하여 모델의 정확성 향상이 학생 마스터리 추정에 중요한 변화를 가져올 수 있음을 보여줍니다.



### DarkSAM: Fooling Segment Anything Model to Segment Nothing (https://arxiv.org/abs/2409.17874)
Comments:
          This paper has been accepted by the 38th Annual Conference on Neural Information Processing Systems (NeurIPS'24)

- **What's New**: 본 논문에서는 Segment Anything Model (SAM)에 대한 최초의 프롬프트-프리(universal attack framework)가 제안되었습니다. 새로운 공격 방법인 DarkSAM은 semantic decoupling 기반의 공간 공격과 texture distortion 기반의 주파수 공격을 포함합니다.

- **Technical Details**: DarkSAM은 SAM의 출력을 전경(foreground)과 배경(background)으로 나눈 후, 그림의 semantic blueprint를 이용하여 공격 목표를 설정합니다. 공간 영역에서는 이미지의 전경과 배경의 의미를 방해하여 SAM을 혼란스럽게 하고, 주파수 영역에서는 고주파 성분(texture information)을 왜곡하여 공격 효과를 강화합니다.

- **Performance Highlights**: 실험 결과, DarkSAM은 다양한 데이터셋에서 SAM 및 그 변형 모델(HQ-SAM 및 PerSAM)을 대상으로 높은 공격 성공률과 전이 가능성을 보여주었습니다.



### Detecting and Measuring Confounding Using Causal Mechanism Shifts (https://arxiv.org/abs/2409.17840)
- **What's New**: 본 논문은 관측 데이터에서의 혼란 변수(confounding variable)를 탐지하고 측정하는 포괄적인 접근 방식이 제안됩니다. 기존 연구들이 비관측 혼란 변수의 존재를 간과한 반면, 우리는 이를 해결하기 위해 다양한 방법론을 마련했습니다.

- **Technical Details**: (i) 변수 집합 간의 혼란을 탐지하고 측정하기 위한 정의와 방법, (ii) 관측된 혼란과 비관측 혼란 효과를 분리하며, (iii) 서로 다른 변수 집합 간의 혼란 편향의 상대 강도를 이해하는 방법론을 포함합니다. 이 연구는 파라메트릭(parmetric) 또는 인과 충분성(causal sufficiency) 가정을 완화함으로써 데이터에서 혼란을 체계적으로 탐구하는 최초의 연구로, 여러 맥락(context)에서 얻은 데이터를 활용합니다.

- **Performance Highlights**: 실험 결과는 이론적 분석을 뒷받침하며, 제안된 혼란 측정의 유용성을 강조합니다. 여러 환경에서의 데이터 활용을 통해 혼란 탐지 및 측정의 효과를 입증했습니다.



### DREAMS: A python framework to train deep learning models with model card reporting for medical and health applications (https://arxiv.org/abs/2409.17815)
- **What's New**: EEG (Electroencephalography) 데이터 분석을 위해 설계된 포괄적인 딥러닝 프레임워크, DREAMS (Deep REport for AI ModelS) 를 소개합니다. 이 프레임워크는 모델 훈련 및 결과 리포트를 위한 기능을 제공하여 의료 연구자와 개발자에게 투명하고 책임 있는 AI 모델을 지원합니다.

- **Technical Details**: DREAMS는 Python 패키지로, 모델 카드 (model cards)를 생성하여 ML (Machine Learning) 및 DL (Deep Learning) 모델의 성능 특성을 문서화하는 포괄적인 프레임워크입니다. 이 프레임워크는 데이터 수집, 전처리, 모델 훈련 및 평가를 포함한 모든 프로세스를 체계적으로 기록합니다. 내부적으로는 EDA (Exploratory Data Analysis)와 같이 데이터 분석을 시각화하는 과정을 포함하며, YAML 파일을 통해 모든 정보에 대한 경로를 정의합니다.

- **Performance Highlights**: DREAMS는 EEG 데이터 분석의 투명성과 윤리적 사용을 촉진하며, 기계 학습 모델의 문서화된 성능 지표를 통해 연구자와 임상의가 효과적으로 모델을 이용할 수 있도록 돕습니다. 특히, EEG 데이터 특정 감정 분류에서 세 가지 주요 범주: 부정적, 중립적, 긍정적 감정을 분류하는 예시를 통해 그 효용성을 보여줍니다.



### Ophthalmic Biomarker Detection with Parallel Prediction of Transformer and Convolutional Architectur (https://arxiv.org/abs/2409.17788)
Comments:
          5 pages

- **What's New**: 본 논문에서는 Optical Coherence Tomography (OCT) 이미지를 이용한 안과 바이오 마커(blog detection) 탐지에서 새로운 접근법을 제시합니다. Convolutional Neural Network (CNN)와 Vision Transformer의 앙상블을 활용하여 보다 정밀한 분석을 가능하게 합니다.

- **Technical Details**: 이 방법은 CNN의 지역적 특징 추출 능력과 Transformer의 전역적 특징 추출 능력을 결합하여 최적의 결과를 도출합니다. OLIVES 데이터셋을 사용하여 OCT 이미지에서 6개의 주요 바이오 마커를 탐지하였으며, 이는 기존 방식에 비해 성능이 크게 향상되었습니다.

- **Performance Highlights**: 평가 지표로 사용된 매크로 평균 F1 스코어(macro averaged F1 score)가 데이터셋에서 유의미한 개선을 보였습니다.



### The application of GPT-4 in grading design university students' assignment and providing feedback: An exploratory study (https://arxiv.org/abs/2409.17698)
Comments:
          25 pages, 5 figures

- **What's New**: 이 연구는 GPT-4가 디자인 대학 학생들의 과제를 효과적으로 채점하고 유용한 피드백을 제공할 수 있는지를 조사합니다.

- **Technical Details**: 연구는 iterative research approach (반복적 연구 접근 방식)을 사용하여 Custom GPT를 개발했습니다. 연구 결과, GPT와 인간 채점자 간의 inter-reliability (상호 신뢰성)가 교육자들에 의해 일반적으로 수용되는 수준에 도달했습니다.

- **Performance Highlights**: GPT의 채점 intra-reliability (내부 신뢰성)은 0.65에서 0.78 사이로 나타났습니다. 이는 적절한 지침을 제공할 경우 일관된 결과를 제공함을 의미하며, 일관성과 비교 가능성은 교육 평가의 신뢰성을 보장하는 두 가지 주요 규칙입니다.



### Artificial Data Point Generation in Clustered Latent Space for Small Medical Datasets (https://arxiv.org/abs/2409.17685)
Comments:
          8 pages, 2 figures

- **What's New**: 이 연구는 작은 의료 데이터셋에서 분류 성능을 향상시키기 위한 합성 데이터 생성 방법인 AGCL(Artificial Data Point Generation in Clustered Latent Space)를 소개합니다. AGCL은 K-평균 클러스터링을 기반으로 하여 클래스 표현이 뚜렷한 클러스터에서 합성 데이터 포인트를 생성하는 방식으로 작동합니다.

- **Technical Details**: AGCL 프레임워크는 특징 추출, K-평균 클러스터링, 클래스 분리를 위한 클러스터 평가 및 각 클러스터의 매개변수를 기반으로 정규 분포에서 합성 데이터 포인트를 생성하는 과정을 포함합니다. 이 방법은 파킨슨병 스크리닝을 위한 얼굴 표정 데이터에 적용되어 여러 머신러닝 분류기에서 평가되었습니다.

- **Performance Highlights**: AGCL은 기본선(GN) 및 kNNMTD와 비교하여 분류 정확도를 유의미하게 향상시켰으며, 서로 다른 감정에 대한 다수결 방식의 교차 검증에서 90.90%의 정확도를 기록했습니다. AGCL은 궁극적으로 83.33%의 테스트 정확도를 달성하며, 작은 데이터셋 증대에 효과적임을 입증했습니다.



### Explanation Bottleneck Models (https://arxiv.org/abs/2409.17663)
Comments:
          13 pages, 4 figures

- **What's New**: 이 논문은 사전 정의된 개념 세트에 의존하지 않고 입력에서 텍스트 설명을 생성할 수 있는 새로운 해석 가능한 심층 신경망 모델인 설명 병목 모델(XBMs)을 제안합니다.

- **Technical Details**: XBMs는 입력 데이터에서 텍스트 설명을 생성하고, 이를 기반으로 최종 작업 예측을 수행하는 모델입니다. XBMs는 사전 학습된 비전-언어 인코더-디코더 모델을 활용하여 입력 데이터에 나타난 개념을 포착합니다. 훈련 과정 중 '설명 증류(explanation distillation)' 기술을 사용하여 분류기의 성능과 텍스트 설명의 품질을 모두 확보합니다.

- **Performance Highlights**: 실험 결과, XBMs는 기존의 개념 병목 모델(CBMs)과 비교하여 더욱 유의미하고 자연스러운 언어 설명을 제공하며, 블랙박스 기준선 모델에 필적하는 성능을 달성하고 있습니다. 특히, XBMs는 테스트 정확도에서 CBMs을 크게 초월합니다.



### A Fuzzy-based Approach to Predict Human Interaction by Functional Near-Infrared Spectroscopy (https://arxiv.org/abs/2409.17661)
- **What's New**: 이번 논문은 심리 연구에서 신경 모델의 해석 가능성과 효율성을 높이기 위해 Fuzzy Attention Layer라는 새로운 Fuzzy 기반 주의 메커니즘을 소개합니다. 이 메커니즘은 Transformer Encoder 모델에 통합되어, 기능적 근적외선 분광법(fNIRS)으로 캡처한 신경 신호를 통해 복잡한 심리 현상을 분석할 수 있도록 합니다.

- **Technical Details**: Fuzzy Attention Layer는 퍼지 집합 이론(Fuzzy set theory)과 Fuzzy 신경망(Fuzzy neural networks), Transformer 시퀀스 모델링을 결합하여 신경 활동 패턴을 학습합니다. 이 계층은 전통적인 dot-product attention보다 향상된 모델 성능을 제공하며, 신경 데이터 집합에서의 해석 가능성을 높입니다.

- **Performance Highlights**: 실험 결과, Fuzzy Attention Layer는 손을 잡고 소통하는 참여자의 fNIRS 데이터를 분석한 결과, 해석 가능한 신경 활동 패턴을 학습함과 동시에 모델 성능을 향상시켰습니다. 이를 통해 인간 간의 터치, 감정 교환과 같은 사회적 행동의 미묘한 복잡성을 해독할 수 있는 잠재력을 보여주었습니다.



### Hierarchical End-to-End Autonomous Driving: Integrating BEV Perception with Deep Reinforcement Learning (https://arxiv.org/abs/2409.17659)
- **What's New**: 이 논문은 Deep Reinforcement Learning (DRL) 기반의 end-to-end 자율주행 프레임워크를 제안하며, Bird's-Eye-View (BEV) 표현을 활용하여 주행 환경을 통합적으로 이해할 수 있도록 합니다.

- **Technical Details**: 이 시스템은 서로 다른 방향을 향한 카메라의 입력을 통합하여 BEV 표현을 구성하고, BEV 데이터를 통해 관련 특징을 추출하는 신경망 모듈을 설계하여 DRL 에이전트에 입력합니다. 이를 통해 추출된 특성은 DRL 에이전트가 주행 전략을 직접 학습할 수 있도록 합니다.

- **Performance Highlights**: 우리의 방법은 자율주행 제어 작업에서 최신 기술보다 성능을 크게 개선하여 충돌 비율을 20% 줄이는 결과를 보여주었습니다.



### FactorSim: Generative Simulation via Factorized Representation (https://arxiv.org/abs/2409.17652)
Comments:
          neurips 2024, project website: this https URL

- **What's New**: 새로운 접근 방식인 FACTORSIM을 소개하며, 텍스트 입력에서 전체 시뮬레이션 코드를 생성하여 에이전트를 훈련할 수 있는 가능성을 제시합니다.

- **Technical Details**: FactorSim은 각 단계에서 필요한 문맥을 줄이기 위해 구조적 모듈성에 기반한 부분적으로 관찰 가능한 마르코프 결정 과정(POMDP) 표현을 활용합니다. 이 과정에서 Model-View-Controller(MVC) 소프트웨어 디자인 패턴을 사용하여 시뮬레이션 생성을 구조화합니다.

- **Performance Highlights**: FACTORSIM은 기존 방법들에 비해 시뮬레이션 생성의 정확도와 제로샷 전이 능력, 인간 평가에서 우수한 성능을 보여주며 로봇 작업 생성에서도 효과적임을 입증했습니다.



### Digital Twin Ecosystem for Oncology Clinical Operations (https://arxiv.org/abs/2409.17650)
Comments:
          Pre Print

- **What's New**: 이 논문은 인공지능(AI)과 디지털 트윈(Digital Twin) 기술을 활용하여 종양학(oncology) 분야의 임상 운영을 혁신적으로 향상시키기 위한 새로운 디지털 트윈 프레임워크를 소개합니다. 여러 전문 디지털 트윈을 통합하여 각 환자에 대한 개인화된 치료를 가능하게 하며, 이는 기존의 데이터와 NCCN 가이드라인에 기반하여 클리닉 추천을 제공합니다.

- **Technical Details**: 이 프레임워크는 Medical Necessity Twin, Care Navigator Twin, Clinical History Twin과 같은 여러 개의 전문 디지털 트윈을 활용하여 환자 맞춤형 치료와 워크플로우 효율성을 높입니다. 각 디지털 트윈은 실시간 데이터 교환을 통해 작동하며, 이를 통해 환자의 유일한 데이터에 기반한 개인화된 care path를 생성하여 ACA 데이터를 통합합니다. 이 시스템은 또한 여러 지식 기반을 통합하여 복잡한 상호작용을 시뮬레이션하는 멀티 에이전트 시스템(Multi-Agent Systems)은 의사결정 지원 및 care coordination을 강화할 수 있습니다.

- **Performance Highlights**: 이 논문에서 제시한 사례 연구는 다양한 에이전트들이 어떻게 협력하여 워크플로우를 간소화하고, 적시의 임상 추천을 제공할 수 있는지를 보여줍니다. 디지털 트윈 기술과 AI의 결합으로 인해 임상 결정의 정확성과 환자 맞춤형 치료의 효율성이 크게 향상됩니다.



### AI Delegates with a Dual Focus: Ensuring Privacy and Strategic Self-Disclosur (https://arxiv.org/abs/2409.17642)
- **What's New**: 이 논문은 AI delegate의 사용자 선호도를 조사한 파일럿 연구를 기반으로 개인 정보 보호와 자발적 공개(self-disclosure) 사이의 균형을 맞추는 새로운 시스템을 제안합니다.

- **Technical Details**: 이 AI delegate 시스템은 대화의 맥락, 관계의 성격, 양쪽 당사자의 편안함 수준을 고려하여 적절한 정보 공개 전략을 선택합니다. 이는 다중 에이전트 프레임워크(multi-agent framework)를 기반으로 하여 대화 목표를 평가하고, 사회적 규범 및 맥락 정보를 바탕으로 대화 전략을 조정합니다.

- **Performance Highlights**: 사용자 연구 결과, 제안된 AI delegate는 다양한 사회적 상호작용에서 개인 정보를 전략적으로 보호하고, LLM과 인간 평가자 간의 일치를 통해 사회적 목표 달성을 위한 적절한 자발적 공개 행동을 보여줍니다.



### Dirichlet-Based Coarse-to-Fine Example Selection For Open-Set Annotation (https://arxiv.org/abs/2409.17607)
- **What's New**: 이번 논문에서는 오픈셋 노이즈가 포함된 데이터에 대해 기존의 Active Learning (AL) 기법의 한계를 극복하기 위해, Dirichlet 기반의 Coarse-to-Fine Example Selection (DCFS) 전략을 제안합니다. 특히, 기존 softmax 기반의 예측에서 발생하는 불안정성을 해결하고, 알려진 클래스와 미지의 클래스를 효과적으로 구별할 수 있는 방법을 제시합니다.

- **Technical Details**: 이 방법은 simplex 기반의 evidential deep learning (EDL)을 활용하여 softmax의 translation invariance를 제거하고, 데이터 및 분포 불확실성을 동시에 고려합니다. 또한, 두 개의 분류 헤드를 통해 모델의 불일치성을 활용하여 강력한 알려진 클래스 예제를 식별하며, 불확실성을 결합하여 두 단계의 전략으로 가장 정보량이 많은 예제를 선택합니다.

- **Performance Highlights**: DCFS는 다양한 openness ratio 데이터셋에서 실험을 진행하였고, 기존 방법들에 비해 뛰어난 예측 정확도를 기록하였으며, 복잡한 오픈셋 환경에서 정보량이 많은 예제를 효과적으로 선택하여 모델 성능을 현저히 향상시키며, state-of-the-art 성능을 달성하였습니다.



### A Scalable Data-Driven Framework for Systematic Analysis of SEC 10-K Filings Using Large Language Models (https://arxiv.org/abs/2409.17581)
Comments:
          10 pages, 7 figures

- **What's New**: 본 연구는 NYSE에 상장된 다수의 기업 성과를 신속하고 비용 효율적으로 평가하고 비교하는 새로운 데이터 기반 접근법을 제안합니다. 이 접근법은 SEC 10-K 보고서를 기반으로 기업의 성과를 체계적으로 분석하고 점수를 매기는 대규모 언어 모델(LLM) 사용을 포함합니다.

- **Technical Details**: SEC 10-K filings에서 데이터를 추출하고 전처리하기 위한 자동화된 시스템을 도입하여 각 섹션을 식별하고 핵심 정보를 분리합니다. 이 데이터는 Cohere의 Command-R+ LLM에 공급되어 다양한 성과 지표에 대한 정량적 평가를 생성합니다.

- **Performance Highlights**: 제안된 시스템은 기업 성과를 연도별로 비교할 수 있는 인터랙티브 GUI를 통해 시각화합니다. 이를 통해 사용자들은 기업의 전략적 변화와 성과 개선을 시간 경과에 따라 쉽게 평가하고 비교할 수 있습니다.



### Showing Many Labels in Multi-label Classification Models: An Empirical Study of Adversarial Examples (https://arxiv.org/abs/2409.17568)
Comments:
          14 pages

- **What's New**: 딥 뉴럴 네트워크(Deep Neural Networks, DNNs)의 발전과 함께 다중 레이블(multi-label) 도메인에서도 적대적 예제(adversarial examples)의 영향을 연구하였습니다. 본 연구는 '많은 레이블 표시(Showing Many Labels)'라는 새로운 유형의 공격을 소개합니다.

- **Technical Details**: 이 공격의 목표는 분류기(classifier) 결과에서 포함된 레이블 수를 극대화하는 것입니다. 실험에서는 9개의 공격 알고리즘(attack algorithms)을 선택하고 '많은 레이블 표시' 상황에서 성능을 평가하였습니다. 8개의 알고리즘은 다중 클래스(multi-class) 환경에서 다중 레이블 환경으로 적응되었고, 나머지 하나는 기존 다중 레이블 환경을 위해 특별히 설계되었습니다.

- **Performance Highlights**: ML-LIW 및 ML-GCN 모델을 선택하여 VOC2007, VOC2012, NUS-WIDE, COCO의 4가지 인기 다중 레이블 데이터셋에서 학습하였습니다. 실험 결과 '많은 레이블 표시' 하에서는 반복 공격(iterative attacks)이 일회성 공격(one-step attacks)보다 현저히 더 높은 성공률(success rate)을 보였습니다. 또한, 데이터셋에 있는 모든 레이블을 표시하는 것이 가능함을 보여주었습니다.



### Just say what you want: only-prompting self-rewarding online preference optimization (https://arxiv.org/abs/2409.17534)
- **What's New**: 이 논문에서는 Reinforcement Learning from Human Feedback (RLHF)의 온라인 자기 보상 정렬 방법(self-rewarding alignment methods)을 다루고 있습니다. 기존의 자기 보상 접근 방식이 판단 능력에 의존하는 반면, 우리는 새로운 프롬프트 기반의 자기 보상 온라인 알고리즘을 제안합니다. 이 알고리즘은 판단 능력 없이도 선호 데이터셋을 생성할 수 있습니다.

- **Technical Details**: 우리는 긍정적 및 부정적 예시 간의 최적성 최적화(optimality gap)를 세밀하게 조정할 수 있는 방식을 채택하였으며, 훈련 후반부에 더 많은 난이도가 있는 부정적 예시(hard negatives)를 생성하여 모델이 미묘한 인간의 선호를 더 잘 포착할 수 있도록 돕습니다. 실험은 Mistral-7B와 Mistral-Instruct-7B 두 가지 기본 모델을 바탕으로 수행되었습니다.

- **Performance Highlights**: 우리의 방법은 AlpacaEval 2.0에서 34.5%의 Length-controlled Win Rates를 달성하며 여러 기준 방법들보다 종합적으로 우수한 성능을 보였습니다.



### Functional Classification of Spiking Signal Data Using Artificial Intelligence Techniques: A Review (https://arxiv.org/abs/2409.17516)
Comments:
          8 figures, 32 pages

- **What's New**: 이번 논문은 인지과학 및 컴퓨터 과학 간의 교차점을 강조하며, 신경 세포 행동의 분석에서 spike의 중요성을 다룹니다. 특히 인공지능(AI)의 도움을 받아 spike를 효과적으로 분류하는 방법에 대한 최근의 연구 결과를 검토합니다.

- **Technical Details**: 본 논문에서는 spike classification(스파이크 분류)의 전처리(preprocessing), 분류(classification), 평가(evaluation) 세 가지 주요 구성 요소를 중심으로 AI의 역할을 탐구합니다. 전통적인 수작업(spike classification)과 비교하여 머신러닝(machine learning) 및 딥러닝(deep learning) 접근법을 통한 신호 데이터의 자동화된 처리의 중요성을 설명합니다.

- **Performance Highlights**: 현재까지의 연구들은 AI를 활용한 스파이크 분류의 정확성을 높일 수 있는 잠재력을 보여줍니다. 효율적인 알고리즘의 필요성도 강조하며, 향후 연구에서 더욱 발전된 방법론과 문제 해결을 위한 방향을 제시합니다.



### From News to Forecast: Integrating Event Analysis in LLM-Based Time Series Forecasting with Reflection (https://arxiv.org/abs/2409.17515)
Comments:
          This paper has been accepted for NeurIPS 2024

- **What's New**: 이 논문은 시간 시계열 예측을 향상하기 위해 Large Language Models (LLMs)과 Generative Agents를 사용하는 새로운 접근 방식을 소개합니다. 이 방법은 뉴스 콘텐츠와 시간 시계열 변동을 정렬하여 여러 사회적 사건을 예측 모델에 적응적으로 통합합니다.

- **Technical Details**: 우리는 LLM 기반 에이전트를 사용하여 관련 없는 뉴스를 반복적으로 걸러내고, 인간과 유사한 추론 및 반성을 통해 예측을 평가합니다. 이를 통해 예기치 않은 사건 및 사회적 행동의 변화와 같은 복잡한 사건을 분석하고 뉴스의 선택 논리 및 에이전트의 출력을 지속적으로 개선합니다.

- **Performance Highlights**: 결과는 예측 정확도가 상당히 향상되었음을 보여주며, 비구조적 뉴스 데이터를 효과적으로 활용하여 시간 시계열 예측의 패러다임 변화를 제안합니다. 다양한 도메인(예: 금융, 에너지, 교통, 비트코인)에서 더 높은 예측 정확도를 달성했습니다.



### GLinSAT: The General Linear Satisfiability Neural Network Layer By Accelerated Gradient Descen (https://arxiv.org/abs/2409.17500)
- **What's New**: 본 논문에서는 신경망(Neural Network) 출력이 특정 제약 조건을 만족하도록 하는 새로운 접근 방식을 제안합니다. 특히, 일반적인 선형 제약 조건을 만족시키는 GLinSAT 레이어를 개발하여, 효율적인 배치 처리와 GPU에서의 고속 연산을 가능하게 하였습니다.

- **Technical Details**: GLinSAT는 엔트로피 정규화(Entropy Regularization)를 추가한 선형 계획 문제(Linear Programming)로 신경망 출력 투영 문제를 공식화합니다. 이 문제는 쌍대성 정리(Duality Theorem)에 의해 Lipschitz 연속 기울기를 가진 비제약 볼록 최적화 문제(Unconstrained Convex Optimization Problem)로 변환됩니다. GLinSAT은 이 문제를 해결하기 위해 가속화된 기울기 하강 알고리즘(Accelerated Gradient Descent Algorithm)을 사용하여 구현되었습니다.

- **Performance Highlights**: 실험 결과, GLinSAT는 제약이 포함된 외판원 문제(Constrained Traveling Salesman Problems), 아웃라이어가 포함된 부분 그래프 매칭(Partial Graph Matching with Outliers), 예측 포트폴리오 할당(Predictive Portfolio Allocation) 및 전력 시스템 단위 확립(Power System Unit Commitment)에서 기존의 만족도 레이어들에 비해 우수한 성능을 보여주었습니다.



### Human Mobility Modeling with Limited Information via Large Language Models (https://arxiv.org/abs/2409.17495)
- **What's New**: 이 연구는 기존의 데이터 기반 인간 이동 모델링의 한계를 극복하기 위해 Large Language Model(LLM)을 활용한 새로운 인간 이동 모델링 프레임워크를 제안합니다. 이 접근법은 고품질의 이동 데이터에 대한 의존도를 크게 줄이고, 기본적인 사회-인구통계적 정보만으로 개인의 일상 이동 패턴을 생성할 수 있도록 합니다.

- **Technical Details**: 제안된 프레임워크는 개인의 사회-인구통계적 정보에 기반해 활동의 순서를 생성하는 'activity chain' 개념을 사용합니다. LLM의 추론 및 논리적 사고 능력을 활용하여, 공공 자원의 고차원 통계와 구조화된 가이드라인을 활용하여 에이전트의 이동 패턴을 모델링합니다. Jensen-Shannon Divergence(JSD) 측정에서 0.011로 NHTS 데이터셋과 SCAG의 ABM 모델과 비교할 때 유망한 결과를 보였습니다.

- **Performance Highlights**: 이 프레임워크는 지역에 따라 다양한 인간 이동 패턴을 효과적으로 모델링할 수 있음을 보여주었으며, LLM의 활용을 통해 데이터 수집의 어려움을 극복할 수 있는 가능성을 제시합니다. 이는 교통 수요 모델링과 다중 모형, 다중 규모의 교통 시뮬레이션을 지원할 수 있는 토대를 제공합니다.



### Global-Local Medical SAM Adaptor Based on Full Adaption (https://arxiv.org/abs/2409.17486)
- **What's New**: 최근 시각 언어 모델(visual language models)인 Segment Anything Model (SAM)의 발전이 보편적인 의미 분할(universal semantic segmentation) 분야에서 큰 혁신을 가져왔습니다. 특히 Medical SAM adaptor (Med-SA)를 통해 의료 이미지 분할에 많은 도움을 주었습니다. 하지만 Med-SA는 부분적 적응(partial adaption) 방식으로 SAM을 미세 조정(fine-tunes)하여 개선의 여지가 있습니다.

- **Technical Details**: 이 논문에서는 전체 적응(full adaption)이 가능한 새로운 글로벌 의료 SAM 어댑터(GMed-SA)를 제안합니다. GMed-SA는 SAM을 전 세계적으로 적응할 수 있도록 설계되었습니다. 또한 GMed-SA와 Med-SA를 결합하여 글로벌-로컬 의료 SAM 어댑터(GLMed-SA)를 제안하며, SAM을 글로벌과 로컬 모두에 적응시킵니다.

- **Performance Highlights**: 우리는 도전적인 공개 2D 흑색종(segmentation dataset) 분할 데이터셋에서 GLMed-SA의 광범위한 실험을 진행했습니다. 결과는 GLMed-SA가 다양한 평가 메트릭(evaluation metrics)에서 여러 최첨단 의미 분할 방법들보다 뛰어난 성능을 발휘함을 보여주었으며, 우리의 방법의 우수성을 입증하였습니다.



### MaskLLM: Learnable Semi-Structured Sparsity for Large Language Models (https://arxiv.org/abs/2409.17481)
Comments:
          NeurIPS 2024 Spotlight

- **What's New**: 이번 연구는 MaskLLM이라는 새로운 learnable pruning (학습 가능한 가지치기) 방법을 제안합니다. 이 방법은 LLM (대규모 언어 모델)의 Semi-structured (반구조적) sparsity (희소성)을 통해 추론 시 계산 비용을 줄이는 데 초점을 맞추고 있습니다.

- **Technical Details**: MaskLLM은 Gumbel Softmax sampling (검벨 소프트맥스 샘플링)을 활용하여 N:M 패턴을 학습 가능한 분포로 모델링합니다. 이 방법은 N개의 비제로 값과 M개의 매개변수 사이의 관계를 설정하여 대규모 데이터 집합에서 end-to-end training (엔드투엔드 훈련)을 수행할 수 있게 합니다.

- **Performance Highlights**: MaskLLM은 LLaMA-2, Nemotron-4, GPT-3와 같은 여러 LLM을 사용하여 2:4 sparsity 설정 하에 평가되었습니다. 기존의 최첨단 방법에 비해 PPL (perplexity) 측정에서 유의미한 개선을 보였으며, 특히 SparseGPT의 10.42에 비해 MaskLLM은 6.72의 PPL을 달성하였습니다. 이 결과는 MaskLLM이 대규모 모델에서도 효과적으로 고품질 마스크를 학습할 수 있음을 나타냅니다.



### What Would Happen Next? Predicting Consequences from An Event Causality Graph (https://arxiv.org/abs/2409.17480)
- **What's New**: 본 논문은 사건 스크립트 체인(event script chain) 대신 사건 인과 그래프(Event Causality Graph, ECG)를 활용하여 사건 예측의 정확성을 높이는 새로운 Causality Graph Event Prediction (CGEP) 작업을 제안합니다. 이를 위해 Semantic Enhanced Distance-sensitive Graph Prompt Learning (SeDGPL) 모델을 제시하며, 이는 세 가지 모듈로 구성되어 있습니다.

- **Technical Details**: SeDGPL 모델은 (1) 거리 민감 그래프 선형화(Distance-sensitive Graph Linearization, DsGL) 모듈을 통해 ECG를 PLM(Pre-trained Language Model)의 입력으로 변환합니다. (2) 이벤트 강화 인과성 인코딩(Event-Enriched Causality Encoding, EeCE) 모듈은 이벤트의 맥락적 의미와 그래프 스키마 정보를 통합하여 이벤트 표현을 풍부하게 합니다. (3) 의미적 대조 사건 예측(Semantic Contrast Event Prediction, ScEP) 모듈은 다양한 후보 사건 사이에서 이벤트 표현을 강화하고, 대조 학습 프레임워크를 통해 결과 사건을 예측합니다.

- **Performance Highlights**: 실험 결과, SeDGPL 모델이 기존 경쟁 모델보다 CGEP 작업에서 우수한 성능을 발휘함을 입증하였으며, ECG 기반의 사건 예측이 사건 스크립트 체인 기반 예측보다 더 합리적임을 확인했습니다.



### A Time Series is Worth Five Experts: Heterogeneous Mixture of Experts for Traffic Flow Prediction (https://arxiv.org/abs/2409.17440)
Comments:
          20 pages, 4 figures

- **What's New**: 이번 연구에서는 교통 흐름 예측을 위해 변수 중심(variable-centric) 및 선행 지식 중심(prior knowledge-centric) 모델링 기법을 도입한 TITAN이라는 이질적(expert) 혼합 모델을 제안합니다. TITAN은 3개의 시퀀스 중심(expert)과 1개의 변수 중심(expert), 1개의 선행 지식 중심(expert)으로 구성되어 있습니다.

- **Technical Details**: TITAN 모델은 세 가지 서로 다른 전문가 유형으로 구성됩니다: 1) 시퀀스 중심 예측 전문가, 2) 변수 중심 예측 전문가, 3) 선행 지식 중심 리더 전문가. 저희는 전문가 간의 지식 정렬을 위해 저순위 행렬(low-rank matrix)을 활용하며, 이를 통해 다양한 백본 네트워크를 통합하여 MoE 프레임워크에서 효과적으로 inductive bias를 줄이는 방향으로 설계되었습니다.

- **Performance Highlights**: TITAN 모델은 METR-LA 및 PEMS-BAY라는 두 개의 공공 교통 네트워크 데이터 세트에서 평가되어, 기존의 최첨단(SOTA) 모델에 비해 4.37%에서 11.53%의 성능 향상을 달성하였습니다. 평균적으로 9%의 개선을 보였습니다.



### Exploring the Use of ChatGPT for a Systematic Literature Review: a Design-Based Research (https://arxiv.org/abs/2409.17426)
Comments:
          21 pages, 13 figures, 2 tables

- **What's New**: 본 연구는 교육, 배우기, 교수 및 연구 등 여러 교육적 맥락에서 사용되고 있는 ChatGPT의 시스템적 문헌 리뷰(SSR) 수행 가능성을 탐구합니다. 또한, ChatGPT를 사용하여 이전에 발표된 SSR을 분석하고 비교하여 그 차이를 발견했습니다.

- **Technical Details**: 이 연구는 디자인 기반 접근법을 사용하여 33개 논문에 대한 SSR을 ChatGPT로 수행했습니다. ChatGPT는 문헌을 분석하기 위해 세부적이고 정확한 프롬프트가 필요하지만, 신뢰성과 유효성을 향상시키기 위한 연구자의 전략도 중요합니다.

- **Performance Highlights**: 연구 결과, ChatGPT는 SSR을 수행할 수 있는 능력이 있지만 제한 사항도 존재함을 확인했습니다. 연구자들은 SSR 수행 시 ChatGPT를 사용하는 데 필요한 가이드라인을 제시했습니다.



### Exploring Semantic Clustering in Deep Reinforcement Learning for Video Games (https://arxiv.org/abs/2409.17411)
- **What's New**: 이 논문에서는 비디오 게임을 위한 딥 강화 학습(Deep Reinforcement Learning, DRL)의 의미 클러스터링(semaitc clustering) 특성을 조사하여 DRL의 내부 동역학을 이해하고 그 해석 가능성을 발전시킵니다.

- **Technical Details**: 제안된 새로운 DRL 아키텍처는 의미 클러스터링 모듈을 통합하여 특징 차원 축소(feature dimensionality reduction) 및 온라인 클러스터링(online clustering)을 지원합니다. 이 모듈은 DRL 훈련 파이프라인에 원활하게 통합되어 기존의 t-SNE 기반 분석 방법에서 보였던 불안정성 문제를 해결하고, 광범위한 수동 주석(manual annotation)의 필요성을 제거합니다.

- **Performance Highlights**: 실험을 통해 제안된 모듈 및 DRL의 의미 클러스터링 특성의 효과성을 검증하였으며, 이를 기반으로 정책(policy)의 계층 구조(hierarchical structure) 및 특징 공간(feature space) 내의 의미 분포(semantic distribution)를 이해하는 데 도움이 되는 새로운 분석 방법을 소개합니다.



### Post-hoc Reward Calibration: A Case Study on Length Bias (https://arxiv.org/abs/2409.17407)
Comments:
          Preprint

- **What's New**: 이 논문은 Large Language Models (LLMs)의 보상 모델 (Reward Model, RM)의 편향을 교정하는 데 도움을 주는 'Post-hoc Reward Calibration' 개념을 소개합니다. 이 접근법은 추가 데이터나 훈련 없이도 성능 향상을 가능하게 합니다.

- **Technical Details**: 편향된 보상을 분해하여 잠재적인 진짜 보상과 특정 특성에 의존하는 편향 항으로 나누는 방법론을 제시합니다. 이를 위해 Locally Weighted Regression 기법을 사용하여 편향을 추정하고 제거하는 방식을 채택하였습니다.

- **Performance Highlights**: 본 연구는 세 가지 실험 설정에서 다음과 같은 성과 향상을 입증하였습니다: 1) RewardBench 데이터셋의 33개 RM에서 평균 3.11 성능 향상, 2) AlpacaEval 벤치마크에 기반한 GPT-4 평가 및 인간 선호도와의 RM 순위 개선, 3) 다양한 LLM-RM 조합에서 RLHF 과정의 Length-Controlled 승률 향상.



### AI Enabled Neutron Flux Measurement and Virtual Calibration in Boiling Water Reactors (https://arxiv.org/abs/2409.17405)
- **What's New**: 이 논문에서는 고온수로(Cooling Water Reactor)에서의 전력 분포를 정확하게 측정하기 위한 새로운 접근법을 제시합니다. 특히, 두 가지 딥러닝(DL) 모델인 SurrogateNet과 LPRMNet을 통해 오프라인과 온라인 전력 분포 간의 편향을 줄이고, 안전하고 경제적인 Reload Core 설계를 가능하게 합니다.

- **Technical Details**: SurrogateNet은 다른 LPRM의 판독값을 이용해 특정 LPRM의 판독값을 예측하고, LPRMNet은 원자로의 상태 변수를 기반으로 LPRM 값을 예측합니다. 두 모델 모두 작동 주파수는 약 1Hz이며, LPRM 시스템은 핵심 신호를 제공합니다. DNN 아키텍처들은 각각 1%와 3%의 테스트 오류를 보여줍니다.

- **Performance Highlights**: 모델의 응용으로는 LPRM이 우회되거나 오류가 발생했을 때의 가상 센서 기능, 연속 캘리브레이션 간의 가상 캘리브레이션, LPRM의 종말 수명(EOL) 결정의 높은 정확성 등이 포함됩니다. 이로 인해 오프라인과 예측된 전력 분포 간의 편향이 감소하게 됩니다.



### Search for Efficient Large Language Models (https://arxiv.org/abs/2409.17372)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 본 논문은 Large Language Models (LLMs)의 최적 아키텍처를 찾기 위한 트레이닝이 필요 없는 아키텍처 검색 프레임워크를 제안합니다. 이 방법은 기존 LLM의 기본 강점을 유지하면서도 추론 속도를 향상시키는 최적의 서브넷을 식별하는 것을 목표로 합니다.

- **Technical Details**: 새로운 아키텍처 검색 방법은 초기 아키텍처의 가중치 중요성을 기반으로 하여 초기 아키텍처를 설정하고, 진화 기반(Path-based) 알고리즘을 통해 효율적인 서브넷을 글로벌하게 검색합니다. 최종적으로, 선택된 서브넷의 가중치를 조정하기 위해 ADMM(Alternating Direction Method of Multipliers) 기반의 재형성 알고리즘을 도입합니다.

- **Performance Highlights**: 제안된 방법은 여러 기준점에서 SOTA 구조적 가지치기(SOTA structured pruning) 작업들을 초월하는 성능을 나타내며, GPU 메모리 사용량을 줄이고 추론 속도를 크게 향상시킵니다. 예를 들어, LLaMA-7B 모델에서는 perplexity가 10.21로, LLM-Pruner의 38.27보다 우수한 성과를 보였습니다.



### A Hybrid Quantum-Classical AI-Based Detection Strategy for Generative Adversarial Network-Based Deepfake Attacks on an Autonomous Vehicle Traffic Sign Classification System (https://arxiv.org/abs/2409.17311)
- **What's New**: 이번 연구에서는 자율주행차량(AV)의 교통신호 인식 시스템을 속이기 위한 딥페이크(deepfake) 공격 방법을 제시했습니다. 딥페이크 기술이 악의적인 공격에서 어떻게 활용될 수 있는지를 탐구하였습니다.

- **Technical Details**: 연구진은 생성적 적대 신경망(Generative Adversarial Network, GAN)을 기반으로 한 딥페이크 공격을 설계하였습니다. 하이브리드 양자-고전적 신경망(hybrid quantum-classical neural networks, NNs)을 활용하여 교통신호 이미지의 특징을 양자 상태로 암호화하는 방식으로 메모리 요구량을 감소시켰습니다.

- **Performance Highlights**: 하이브리드 딥페이크 탐지 접근법은 실제 및 딥페이크 교통신호 이미지에 대해 여러 베이스라인 고전적 합성곱 신경망(classical convolutional NNs)과 비교 평가한 결과, 대부분의 경우 베이스라인보다 유사하거나 높은 성능을 나타냈고, 가장 얕은 고전적 합성곱 NN의 메모리 사용량의 1/3 이하에서 실행될 수 있음을 보여주었습니다.



### Proof of Thought : Neurosymbolic Program Synthesis allows Robust and Interpretable Reasoning (https://arxiv.org/abs/2409.17270)
- **What's New**: 이 연구에서는 LLM(대형 언어 모델)의 출력 신뢰성과 투명성을 높이기 위해 'Proof of Thought'라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 LLM이 생성한 아이디어와 형식 논리 검증을 연결합니다.

- **Technical Details**: Proof of Thought는 사용자 친화적인 개념을 사용하는 JSON 기반의 도메인 특화 언어(DSL)로, 이를 통해 LLM의 출력을 1차 논리(First Order Logic) 구조로 변환하는 커스텀 인터프리터를 사용합니다. 이 방법은 논리적 구조와 인간이 이해할 수 있는 개념 사이의 균형을 이루도록 설계되었습니다.

- **Performance Highlights**: Proof of Thought는 StrategyQA 및 새로운 멀티모달 추론 작업에서 효과를 입증하였으며, 개방형 시나리오에서 향상된 성능을 보였습니다. 이는 AI 시스템의 책임성을 다루고, 높은 위험 도메인에서 인간의 관여를 위한 기초를 설정하는 데 기여합니다.



### AAPM: Large Language Model Agent-based Asset Pricing Models (https://arxiv.org/abs/2409.17266)
- **What's New**: 이 연구에서는 LLM 에이전트를 활용한 새로운 자산 가격 책정 접근 방식인 LLM Agent-based Asset Pricing Models (AAPM)을 제안합니다. 이 방법은 정성적인 투자 분석과 정량적인 경제적 요인을 결합하여 자산의 초과 수익을 예측하는 데 중점을 둡니다.

- **Technical Details**: AAPM 모델은 최신 뉴스 분석을 반복적으로 수행하는 LLM 에이전트를 사용하며, 이전 분석 보고서와 책, 백과사전, 저널 등을 포함하는 지식 기반을 지원합니다. 이 모델은 분석 보고서와 수동 요인을 결합하여 향후 초과 자산 수익을 예측합니다.

- **Performance Highlights**: 실험 결과, AAPM은 머신러닝 기반 자산 가격 책정 기준을 초월하여 샤프 비율을 9.6% 향상시키고 자산 가격 오류에 대한 평균 |α|를 10.8% 개선했습니다.



### Collaborative Comic Generation: Integrating Visual Narrative Theories with AI Models for Enhanced Creativity (https://arxiv.org/abs/2409.17263)
Comments:
          This paper has been accepted for oral presentation at CREAI2024, ECAI, 2024. However, the author's attendance is currently uncertain due to visa issues

- **What's New**: 이 연구에서는 컴퓨터 생성 모델을 활용하여 만화 생성 과정의 효율성을 향상시키기 위해 새로운 시각적 내러티브 생성 시스템을 제안합니다. 이 시스템은 인간의 창의성과 AI 모델을 결합하여 만화 콘텐츠 제작을 지원하는 협업 플랫폼을 제공합니다.

- **Technical Details**: 시스템은 다중 AI 모델을 활용하여 저자들이 생성 과정을 사용자 지정할 수 있는 사람-중심(‘human-in-the-loop’) 워크플로우로 구조화되어 있습니다. 주요 구성 요소로는 그래픽 사용자 인터페이스(GUI), 모델 컨테이너, 이미지 시퀀스 모델, 생성기, 렌더러 등 여섯 개의 모듈이 있습니다.

- **Performance Highlights**: 이 시스템은 스토리 요소가 잘 반영된 이미지 시퀀스를 생성하며, 저자들이 부분적으로 수정할 수 있도록 하여 만화 제작을 더 유연하고 효율적으로 만듭니다. 또한 드라마틱한 이야기 전개와 캐릭터 일관성을 보장합니다.



### Multi-View and Multi-Scale Alignment for Contrastive Language-Image Pre-training in Mammography (https://arxiv.org/abs/2409.18119)
Comments:
          This work is also the basis of the overall best solution for the MICCAI 2024 CXR-LT Challenge

- **What's New**: 이 연구는 의료 영상 분야에서 Contrastive Language-Image Pre-training (CLIP) 모델의 초기 적응을 유방 촬영술에 적용하고, 데이터 부족과 고해상도 이미지의 세부 사항 강조를 해결하기 위한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 제안하는 방법은 Multi-view와 Multi-scale Alignment (MaMA)를 기반으로 하며, 각 관점에서 정보를 활용하여 multi-view mammography의 특징을 동시에 정렬하는 시스템을 구축합니다. 또한, clinical report 부족 문제를 해결하기 위해 template-based report construction 방식을 개발하고, parameter-efficient fine-tuning 기법을 적용합니다.

- **Performance Highlights**: EMBED 및 RSNA-Mammo의 대규모 실제 유방촬영 데이터셋에서 3개의 다른 작업에 대해 기존 방법보다 우수한 성능을 보여주었으며, 모델 크기의 52%만으로도 뛰어난 성과를 달성했습니다.



### Find Rhinos without Finding Rhinos: Active Learning with Multimodal Imagery of South African Rhino Habitats (https://arxiv.org/abs/2409.18104)
Comments:
          9 pages, 9 figures, IJCAI 2023 Special Track on AI for Good

- **What's New**: 이 연구는 코뿔소 보호를 위한 새로운 접근 방식을 제안하고 있으며, 코뿔소의 집단 배변 장소를 지도화하는 것을 통해 이들의 공간 행동에 대한 정보를 수집할 수 있는 방법을 모색하고 있다.

- **Technical Details**: 이 논문은 원거리에서 감지된 열 감지(thermal), RGB, LiDAR 이미지를 사용하여 코뿔소 배변 장소를 탐지하는 분류기를 구축하고, 수동적 및 능동적 학습 설정에서 이를 수행한다. 특히, 기존 능동 학습 방법들이 지극히 불균형한 데이터셋의 문제로 성능이 저하되는 문제를 해결하기 위해, MultimodAL이라고 불리는 새로운 능동 학습 시스템을 설계하였다.

- **Performance Highlights**: 제안된 방법은 94% 감소된 라벨 수로 수동적 학습 모델과 경쟁력을 갖추며, 유사한 크기의 데이터셋에 적용할 경우 라벨링 시간에서 76시간 이상을 절약할 수 있는 것으로 나타났다. 또한, 연구 결과 배변 장소가 무작위 분포가 아니라 클러스터 형태로 밀집해 있다는 점을 발견하여, 이에 따라 기동대(rangers) 활동의 효율성을 높이기 위한 방안을 제시하고 있다.



### AI-Powered Augmented Reality for Satellite Assembly, Integration and Tes (https://arxiv.org/abs/2409.18101)
- **What's New**: 이번 논문은 인공지능(AI)과 증강현실(AR)의 통합을 통해 위성 조립, 통합 및 테스트(AIT) 프로세스를 혁신하려는 유럽우주청(ESA)의 프로젝트 "AI for AR in Satellite AIT"에 대해 설명합니다. 이 시스템은 Microsoft HoloLens 2를 사용하여 기술자에게 실시간으로 맥락에 맞는 지침과 피드백을 제공합니다.

- **Technical Details**: AI4AR 시스템은 컴퓨터와 AR 헤드셋으로 구성되어 있으며, 객체 감지 및 추적, 6D 포즈 추정과 OCR을 포함한 다양한 컴퓨터 비전 알고리즘을 이용합니다. 특히, 6D 포즈 모델 훈련에 합성 데이터를 사용하는 접근법이 독창적이며, 이는 AIT 프로세스의 복잡한 환경에서 유용합니다.

- **Performance Highlights**: AI 모델의 정확도가 70%를 넘었고, 객체 감지 모델은 95% 이상의 정확도를 기록했습니다. 자동 주석화 기능을 가진 Segmented Anything Model for Automatic Labelling(SAMAL)을 통해 실제 데이터의 주석화 속도가 수동 주석화보다 최대 20배 빨라졌습니다.



### EfficientCrackNet: A Lightweight Model for Crack Segmentation (https://arxiv.org/abs/2409.18099)
- **What's New**: 본 연구에서는 Convolutional Neural Networks (CNNs)와 transform architecture를 결합한 EfficientCrackNet이라는 경량 하이브리드 모델을 제안합니다. 이 모델은 정밀한 균열(segmentation) 감지를 위해 설계되었으며, 높은 정확도와 낮은 계산 비용을 동시에 제공합니다.

- **Technical Details**: EfficientCrackNet 모델은 depthwise separable convolutions (DSC)와 MobileViT block을 통합하여 전역 및 지역 특성을 캡처합니다. Edge Extraction Method (EEM)를 사용하여 사전 훈련 없이도 효율적인 균열 가장자리 감지를 수행하며, Ultra-Lightweight Subspace Attention Module (ULSAM)을 통해 특성 추출을 향상합니다.

- **Performance Highlights**: 세 개의 벤치마크 데이터셋(Crack500, DeepCrack, GAPs384)을 기반으로 한 광범위한 실험에 따르면, EfficientCrackNet은 단 0.26M의 파라미터와 0.483 FLOPs(G)만으로 기존의 경량 모델들보다 우수한 성능을 보였습니다. 이 모델은 정확성과 계산 효율성 간의 최적의 균형을 제공하여 실제 균열 분할 작업에 강력하고 적응 가능한 솔루션을 제공합니다.



### DiffSSC: Semantic LiDAR Scan Completion using Denoising Diffusion Probabilistic Models (https://arxiv.org/abs/2409.18092)
Comments:
          Under review

- **What's New**: 이 논문에서는 자율 주행 차량을 위한 Semantic Scene Completion (SSC) 작업을 다루고 있으며, LiDAR(빛 감지 및 범위 측정) 센서로부터 얻는 드문 드로우 포인트 클라우드의 빈 공간과 폐쇄된 지역을 예측할 수 있는 방안을 제시합니다.

- **Technical Details**: 제안된 DiffSSC 방법론은 Denoising Diffusion Probabilistic Models (DDPM)을 기반으로 하여 점진적인 노이즈 제거를 통해 LiDAR 점 클라우드를 처리합니다. 이 과정에서는 기하학 및 의미적 정보가 동시에 모델링되며, 로컬과 글로벌 정규화 손실을 통해 학습 과정을 안정화합니다.

- **Performance Highlights**: 제안된 방법은 자율 주행 데이터셋에서 기존의 최첨단 SSC 방법보다 우수한 성능을 보이며, LiDAR 점 클라우드 처리에서 메모리 사용량 및 양자화 오류를 줄이는 데 성공했습니다.



### GSON: A Group-based Social Navigation Framework with Large Multimodal Mod (https://arxiv.org/abs/2409.18084)
- **What's New**: 이번 연구에서는 GSON이라고 명명된 그룹 기반 사회 내비게이션 프레임워크를 소개합니다. 이 프레임워크는 모바일 로봇이 주변의 사회 집단을 인식하고 활용할 수 있도록 지원하여, 사회적 맥락을 고려한 내비게이션을 가능하게 합니다.

- **Technical Details**: GSON은 Large Multimodal Model (LMM)의 시각적 추론 능력을 활용하여 보행자 간의 사회적 관계를 제로샷(Zero-shot)으로 추출하는 시각적 프롬프팅 기법을 적용합니다. 로봇은 2D LiDAR와 RGB 카메라를 통해 주변 환경을 감지하고, 사회 집단 추정 모듈과 사회적으로 인식된 계획 모듈을 통해 경로를 계획합니다. GSON은 글로벌 경로 계획(global path planning)과 로컬 동작 계획(local motion planning) 간의 중간 수준 계획(mid-level planning)을 통해 사회적 구조를 유지하며 움직임을 조율합니다.

- **Performance Highlights**: GSON 프레임워크는 다양한 사회적 상호작용 시나리오에서 강화된 로봇 내비게이션 성능을 보여주며, 기존의 기준선 방법들보다 사회 구조를 최소한으로 방해하며 내비게이션할 수 있는 효과성을 입증하였습니다.



### SKT: Integrating State-Aware Keypoint Trajectories with Vision-Language Models for Robotic Garment Manipulation (https://arxiv.org/abs/2409.18082)
- **What's New**: 이 연구에서는 다양한 의류 소목(garment categories)에 대해 단일 모델로 키포인트(keypoint) 예측을 개선하기 위해 비전-언어 모델(vision-language models, VLM)을 사용하는 통합 접근 방식을 제안합니다. 본 연구는 의류의 다양한 변형 상태를 관리하는 데 도움을 줄 수 있는 새로운 접근법을 제시합니다.

- **Technical Details**: 제안된 방법은 상태 인식 쌍 키포인트 생성(State-aware paired keypoint formation) 기법을 활용하여 다양한 의류 환경에 잘 일반화되는 상태 인식 키포인트 궤적(State-aware Keypoint Trajectories)을 생성합니다. 이를 통해 비주얼 시그널과 텍스트 설명을 함께 해석할 수 있으며, 고급 물리 시뮬레이터를 이용한 대규모 합성 데이터셋을 통해 훈련됩니다.

- **Performance Highlights**: 실험 결과, VLM 기반 방법이 키포인트 감지 정확성과 작업 성공률을 획기적으로 향상시켜 주목받았습니다. 이 연구는 VLM을 활용하여 장기적으로 홈 자동화 및 보조 로봇 분야에서의 폭넓은 응용 가능성을 제시하고 있습니다.



### FreeEdit: Mask-free Reference-based Image Editing with Multi-modal Instruction (https://arxiv.org/abs/2409.18071)
Comments:
          14 pages, 14 figures, project website: this https URL

- **What's New**: 본 논문에서는 사용자가 지정한 시각적 개념을 기반으로 한 이미지 편집을 가능하게 하는 FreeEdit라는 새로운 접근법을 제안합니다. 이 방법은 사용자가 제공한 언어 명령을 통해 참조 이미지를 효과적으로 재현할 수 있도록 합니다.

- **Technical Details**: FreeEdit는 다중 모달 instruction encoder를 활용하여 언어 명령을 인코딩하고, 이를 기반으로 편집 과정을 안내합니다. Decoupled Residual ReferAttention (DRRA) 모듈을 통해 참조 이미지에서 추출한 세밀한 특성을 이미지 편집 과정에 효과적으로 통합합니다. 또한, FreeBench라는 고품질 데이터셋을 구축하였으며, 이는 원본 이미지, 편집 후 이미지 및 참조 이미지를 포함하여 다양한 편집 태스크를 지원합니다.

- **Performance Highlights**: FreeEdit는 객체 추가 및 교체와 같은 참고 기반 이미지 편집 작업에서 기존 방법보다 우수한 성능을 보이며, 수동 편집 마스크를 필요로 하지 않아 사용의 편리함을 크게 향상시킵니다. 실험을 통해 얻은 결과는 FreeEdit가 고품질 제로샷(zero-shot) 편집을 달성하고, 편리한 언어 명령으로 사용자 요구를 충족할 수 있음을 보여줍니다.



### Visual Data Diagnosis and Debiasing with Concept Graphs (https://arxiv.org/abs/2409.18055)
- **What's New**: CONBIAS는 비주얼 데이터셋의 Concept co-occurrence Biases를 진단하고 완화하기 위해 개발된 새로운 프레임워크입니다. 이는 비주얼 데이터셋을 지식 그래프(knowledge graph)로 표현하여 편향된 개념의 동시 발생을 분석하고 이를 통해 데이터셋의 불균형을 파악합니다.

- **Technical Details**: CONBIAS 프레임워크는 세 가지 주요 단계로 이루어져 있습니다: (1) Concept Graph Construction: 데이터셋에서 개념의 지식 그래프를 구축합니다. (2) Concept Diagnosis: 생성된 지식 그래프를 분석하여 개념 불균형을 진단합니다. (3) Concept Debiasing: 그래프 클리크(clique)를 사용해 불균형한 개념 조합을 샘플링하고 이에 대한 이미지를 생성하여 데이터셋을 보완합니다. 이 과정에서 대규모 언어 모델의 의존성을 줄였습니다.

- **Performance Highlights**: CONBIAS를 기반으로 한 데이터 증강(data augmentation) 방법이 여러 데이터셋에서 일반화 성능을 향상시키는 데 성공적임을 보여줍니다. 기존의 최첨단 방법들과 비교했을 때, 균형 잡힌 개념 분포에 기반한 데이터 증강이 분류기의 전반적인 성능을 개선시킴을 실험을 통해 입증하였습니다.



### DualAD: Dual-Layer Planning for Reasoning in Autonomous Driving (https://arxiv.org/abs/2409.18053)
Comments:
          Autonomous Driving, Large Language Models (LLMs), Human Reasoning, Critical Scenario

- **What's New**: 새로운 자율주행 프레임워크인 DualAD가 소개되었습니다. 본 프레임워크는 인간의 추론을 모방하여 자율주행 시스템의 성능을 향상시키는 데 중점을 두고 있습니다. DualAD는 하위 계층에서는 규칙 기반의 동작 계획자(rule-based motion planner)가 작동하고, 상위 계층에서는 규칙 기반의 텍스트 인코더(rule-based text encoder)가 주행 시나리오를 텍스트 설명으로 변환한 후, 이를 대형 언어 모델(LLM)이 처리하여 주행 결정을 내립니다.

- **Technical Details**: DualAD는 두 개의 레이어로 구성됩니다. 첫 번째 레이어는 기본적인 주행 작업을 수행하는 규칙 기반의 동작 계획자이며, 두 번째 레이어는 주행 시나리오를 텍스트로 변환하고 이를 LLM을 통해 처리합니다. 특히, 잠재적 위험이 탐지되면 상위 레이어가 하위 레이어의 결정을 수정하여 인간의 사고 방식을 모방합니다.

- **Performance Highlights**: DualAD는 제로샷(zero-shot)으로 학습된 LLM을 활용하여 일반적인 규칙 기반 동작 계획자보다 뛰어난 성능을 보여주었습니다. 특히, 텍스트 인코더의 효과적인 활용이 모델의 시나리오 이해도를 크게 향상시켰으며, 통합된 DualAD 모델은 강력한 LLM이 추가될수록 성능이 개선됨을 알 수 있었습니다.



### Revisit Anything: Visual Place Recognition via Image Segment Retrieva (https://arxiv.org/abs/2409.18049)
Comments:
          Presented at ECCV 2024; Includes supplementary; 29 pages; 8 figures

- **What's New**: 이번 연구에서는 Embodied agents가 시각적으로 장소를 인식하고 이동하는 데 있어 중요한 문제를 다루었습니다. 전체 이미지를 사용하는 기존 방법 대신, 이미지의 '세그먼트'를 인코딩하고 검색하는 새로운 접근 방식을 제안합니다. 이를 통해 SuperSegment라는 새로운 이미지 표현을 생성하여 장소 인식을 향상시킵니다.

- **Technical Details**: 제안된 SegVLAD는 Open-set 이미지 분할(open-set image segmentation)을 통해 이미지를 의미있는 요소(entities)로 분해합니다. 각 아이템은 SuperSegments로 연결되어 구조화됩니다. 새로 제안된 Feature aggregation 방법을 사용하여 이 SuperSegments를 효율적으로 컴팩트한 벡터 표현으로 인코딩합니다. SegVLAD는 다양한 벤치마크 데이터셋에서 기존의 방법보다 높은 인식 리콜을 기록했습니다.

- **Performance Highlights**: SegVLAD는 다양한 VPR 벤치마크 데이터셋에서 최첨단 성능을 달성했습니다. IOU 기반 필터링을 통해 중복성을 줄이고 스토리지를 절약하며, 전체 이미지 기반 검색보다 더욱 뛰어난 성능을 보입니다. 연구 결과, SegVLAD는 이미지 인코더의 특정 작업에 관계없이 적용 가능하고, 객체 인스턴스 검색(object instance retrieval) 과제를 평가하여 '무언가를 재방문(revisit anything)'할 수 있는 잠재력을 보여주었습니다.



### HARMONIC: Cognitive and Control Collaboration in Human-Robotic Teams (https://arxiv.org/abs/2409.18047)
Comments:
          Submitted to ICRA 2025 Conference, Atlanta, GA, USA

- **What's New**: 이 논문은 다중 로봇 계획 및 협력에 대한 새로운 접근 방식을 선보입니다. 로봇과 인간이 팀을 이루어 작업할 때 인지 전략을 포함하여 메타인지, 자연어 통신 및 설명 가능성을 통합한 시스템을 설명합니다.

- **Technical Details**: HARMONIC 아키텍처를 사용하여 인지 및 제어 기능을 팀 전반에 걸쳐 유연하게 통합합니다. 유연한 중앙집중식 및 분산 방식의 계획 접근 방식을 활용하여 로봇이 목표, 계획, 태도를 인식하고 그들의 행동과 결정에 대한 설명을 제공할 수 있도록 합니다.

- **Performance Highlights**: 이 연구는 이질적인 로봇 팀(UGV 및 드론 포함)이 인간과 함께 공동 검색 작업을 수행하는 시뮬레이션 실험 결과를 통해 복잡한 실제 시나리오를 처리하는 로봇 팀의 능력을 보여줍니다.



### IFCap: Image-like Retrieval and Frequency-based Entity Filtering for Zero-shot Captioning (https://arxiv.org/abs/2409.18046)
Comments:
          Accepted to EMNLP 2024

- **What's New**: 최근 이미지 캡셔닝(image captioning) 분야에서 이미지-텍스트 데이터 쌍의 한계를 극복하기 위해 텍스트-전용(training) 교육 방법이 탐색되고 있습니다. 본 논문에서는 텍스트 데이터와 이미지 데이터 간의 모달리티 차이를 완화하기 위한 새로운 접근 방식으로 'Image-like Retrieval'을 제안합니다.

- **Technical Details**: 제안된 방법인 IFCap($\textbf{I}$mage-like Retrieval과 $\textbf{F}$requency-based Entity Filtering for Zero-shot $\textbf{Cap}$tioning)는 효율적인 이미지 캡셔닝을 위한 통합 프레임워크로, Fusion Module을 통해 검색된 캡션과 입력 특성을 통합하여 캡션 품질을 향상시킵니다. 또한 Frequency-based Entity Filtering 기술을 도입하여 더 나은 캡션 품질을 제공합니다.

- **Performance Highlights**: 광범위한 실험 결과, IFCap은 기존의 텍스트-전용 훈련 기반 제로샷 캡셔닝(zero-shot captioning) 방식에 비해 이미지 캡셔닝과 비디오 캡셔닝 모두에서 state-of-the-art 성능을 기록하는 것으로 나타났습니다.



### HARMONIC: A Framework for Explanatory Cognitive Robots (https://arxiv.org/abs/2409.18037)
Comments:
          Accepted for presentation at ICRA@40. 23-26 September 2024, Rotterdam, Netherlands

- **What's New**: HARMONIC 프레임워크는 일반 목적의 로봇을 신뢰할 수 있는 팀원으로 변환하여 복잡한 의사결정, 자연어 소통 및 인간 수준의 설명을 가능하게 하는 시스템입니다.

- **Technical Details**: HARMONIC은 전략적(cognitive) 레이어와 전술적(robot) 레이어로 구성된 이중 제어 아키텍처를 가지고 있으며, 이 두 레이어는 독립적으로 상호작용합니다. 전략적 레이어는 주의 관리, 지각 해석 및 유틸리티 기반 의사결정 모듈을 포함하고, 전술적 레이어는 센서 입력 처리 및 로봇 제어를 담당합니다.

- **Performance Highlights**: 초기 구현에서는 인간-로봇 팀이 아파트 환경에서 분실된 열쇠를 찾는 시뮬레이션 작업을 수행합니다. HARMONIC 기반의 로봇인 UGV와 드론이 인간과 협력하여 검색 매개변수를 설정하고 전략적 계획을 선택 및 실행합니다.



### An Adversarial Perspective on Machine Unlearning for AI Safety (https://arxiv.org/abs/2409.18025)
- **What's New**: 이번 연구는 기존의 unlearning(언러닝) 방법이 안전성 훈련(safety training)에서의 위험한 정보 제거를 효과적으로 대체할 수 있는지를 조명합니다. 연구자들은 unlearning이 단순히 정보를 숨기는데 그치고, 위험한 지식이 여전히 회복될 수 있음을 보여줍니다.

- **Technical Details**: 이 논문은 RMU(Residual Memory Unlearning)와 같은 상태-of-the-art unlearning 방법을 평가하며, WMDP(WMD Probing) 벤치마크를 사용하여 안전성 훈련과 비교합니다. 기존 보고된 jailbreak 방법들은 unlearning에 무효로 간주되었으나, 세심하게 적용될 경우 여전히 효과적일 수 있음을 발견하였습니다. 이들은 특정 활성 공간 방향을 제거하거나 무관한 예제를 가진 finetuning을 통해 원래의 성능을 복구할 수 있는 방법을 제시합니다.

- **Performance Highlights**: 연구 결과, unlearning 방법은 특정 공격에 보다 강해 보이지만, 안전성 훈련에 사용된 기법으로 쉽게 우회될 수 있습니다. 특히, GCG와 같은 jailbreak 기법은 손실 함수를 약간 변경함으로써 의미 있는 정확도를 회복할 수 있음을 보여주었습니다.



### Transferring disentangled representations: bridging the gap between synthetic and real images (https://arxiv.org/abs/2409.18017)
- **What's New**: 이 연구는 합성 데이터(synthetic data)를 사용하여 실제 데이터에 적용 가능한 일반 목적의 분리 표현(disentangled representation)을 학습할 가능성을 탐구합니다. 이 과정에서 성능을 향상시키기 위한 미세 조정(fine-tuning)의 효과와 전이(transfer) 후 보존되는 분리 특성에 대해 논의합니다.

- **Technical Details**: 본 연구에서는 OMES (Overlap Multiple Encoding Scores)라는 새로운 간섭 기반(intervention-based) 지표를 제안하여, 표현에서 인코딩된 요소들의 품질을 측정합니다. 이는 기존의 분류기(classifier) 의존적인 방법들과는 달리, 분류기 없는(intervention-based) 접근을 채택하여 요소의 분포를 분석하며, 데이터를 쌍으로 매칭하여 단일 요소만 다르게 하여 평가합니다.

- **Performance Highlights**: 연구 결과, 합성 데이터에서 학습된 표현을 실제 데이터로 전이할 수 있는 가능성이 있으며, 일부 분리 수준이 효과적임을 보여줍니다. 특히, 다양한 (Source, Target) 쌍에 대한 실험을 통해 학습된 DR의 표현력이 잘 평가됨을 확인하였습니다.



### Control Industrial Automation System with Large Language Models (https://arxiv.org/abs/2409.18009)
- **What's New**: 이 논문은 대형 언어 모델(LLM)을 산업 자동화 시스템에 통합하기 위한 새로운 프레임워크를 제안합니다.

- **Technical Details**: 프레임워크는 산업 업무를 위한 에이전트 시스템, 구조화된 프로프트(prompting) 방법 및 사건 기반(event-driven) 정보 모델링 메커니즘을 포함하고 있으며, 이는 LLM 추론을 위한 실시간 데이터를 제공합니다.

- **Performance Highlights**: 제안된 방법은 LLM이 정보 해석, 생산 계획 생성 및 자동화 시스템의 작업 제어를 가능하게 하여 산업 자동화 시스템의 적응력을 높이고 자연어를 통한 더 직관적인 인간-기계 상호 작용을 지원합니다.



### Joint Localization and Planning using Diffusion (https://arxiv.org/abs/2409.17995)
Comments:
          7 pages, 9 figures. Submitted to ICRA 2025, under review

- **What's New**: 이번 연구에서는 로봇 내비게이션 문제 해결을 위한 새로운 Diffusion 모델을 제안합니다. 이는 전방향 로컬라이제이션(global localization)과 경로 계획(path planning), 두 가지 프로세스를 통합하여 임의의 2D 환경에서의 탐색을 가능하게 합니다.

- **Technical Details**: 제안된 모델은 LIDAR 스캔, 장애물 맵 및 목표 위치에 의존하여 충돌 없는 경로를 생성합니다. 연구는 SE(2) 공간에서의 확산을 구현하며, 장애물과 센서 관측치에 따라 디노이징 과정이 어떻게 조정되는지를 설명합니다.

- **Performance Highlights**: 실험 결과, 제안된 조건화 기법이 훈련 환경과 상당히 다른 외관의 현실적인 맵에 대한 일반화를 가능하게 하며, 불확실한 솔루션을 정확히 설명할 수 있음을 보여주었습니다. 또한, 실제 환경에서의 실시간 경로 계획 및 제어를 위한 모델의 활용 가능성을 시연했습니다.



### HydraViT: Stacking Heads for a Scalable V (https://arxiv.org/abs/2409.17978)
- **What's New**: 새로운 HydraViT 접근 방식은 다중 크기의 Vision Transformers (ViTs) 모델을 학습하고 저장하는 필요성을 없애면서, 스케일러블한 ViT를 가능하게 합니다.

- **Technical Details**: HydraViT는 Multi-head Attention (MHA) 메커니즘을 기반으로 하여, 다양한 하드웨어 환경에 적응할 수 있도록 임베딩 차원과 MHA의 헤드 수를 동적으로 조정합니다. 이 방식은 최대 10개의 서브 네트워크를 생성할 수 있습니다.

- **Performance Highlights**: HydraViT는 ImageNet-1K 데이터셋에서 동일한 GMACs와 처리량을 기준으로, 기존 모델 대비 최대 7 p.p. 높은 정확성을 달성하며, 이는 다양한 하드웨어 환경에서 특히 유용합니다.



### Weak-To-Strong Backdoor Attacks for LLMs with Contrastive Knowledge Distillation (https://arxiv.org/abs/2409.17946)
- **What's New**: 이번 연구에서는 Parameter-Efficient Fine-Tuning (PEFT)을 사용하는 대규모 언어 모델(LLMs)에 대한 백도어 공격의 효율성을 검증하고, 이를 개선하기 위한 새로운 알고리즘인 W2SAttack(Weak-to-Strong Attack)을 제안합니다.

- **Technical Details**: W2SAttack은 약한 모델에서 강한 모델로 백도어 특징을 전이하는 대조적 지식 증류(constractive knowledge distillation) 기반의 알고리즘입니다. 이 알고리즘은 소규모 언어 모델을 사용하여 완전 파라미터 미세 조정을 통해 백도어를 임베드하고 이를 교사 모델로 활용하여 대규모 학생 모델로 전이합니다. 이 과정에서 정보를 최소화하여 학생 모델이 목표 레이블과 트리거 간의 정렬을 최적화하도록 합니다.

- **Performance Highlights**: W2SAttack은 여러 언어 모델과 백도어 공격 알고리즘을 대상으로 한 실험 결과에서 100%에 가까운 성공률을 기록했습니다. 이는 PEFT를 사용한 기존 백도어 공격의 성능을 크게 개선한 결과입니다.



### On Translating Technical Terminology: A Translation Workflow for Machine-Translated Acronyms (https://arxiv.org/abs/2409.17943)
Comments:
          AMTA 2024 - The Association for Machine Translation in the Americas organizes biennial conferences devoted to researchers, commercial users, governmental and NGO users

- **What's New**: 이번 논문에서는 기계 번역(Machine Translation, MT) 시스템에서 약어의 모호성 제거(acronym disambiguation)를 제안함으로써, 약어 번역의 정확성을 높이고자 하는 새로운 접근 방식을 소개합니다. 또한 새로운 약어 말뭉치(corpus)를 공개하고, 이를 기반으로 한 검색 기반 임계값(thresholding) 알고리즘을 실험하여 기존의 Google Translate와 OpusMT보다 약 10% 더 나은 성능을 보였습니다.

- **Technical Details**: 기계 번역 시스템의 약어 번역 정확도를 향상시키기 위해, 4단계의 고레벨 프로세스를 제안하였습니다. 이 프로세스는 (1) Google Translate를 사용하여 FR-EN 번역 수행, (2) 영어 장기형(long form, LF)과 단기형(short form, SF) 추출, (3) AB3P 툴을 사용한 약어 가설 생성, (4) 검색 기법을 통한 가설 검증 및 평가입니다. 이 방법들은 텍스트에서 사용된 기술 용어의 신뢰성을 향상시키는데 기여하고자 합니다.

- **Performance Highlights**: Google Translate와 OpusMT와 비교하여, 제안하는 임계값 알고리즘은 약 10%의 성능 향상을 보여주었습니다. 우리의 연구에서는 전문 번역사들이 자주 직면하는 용어 오류를 감소시키는 방안을 제시하고 있으며, 이를 통해 MT 시스템에서의 기술 용어 번역의 적합성을 증대시키는 데 기여할 것입니다.



### Predicting Anchored Text from Translation Memories for Machine Translation Using Deep Learning Methods (https://arxiv.org/abs/2409.17939)
Comments:
          AMTA 2024 - The Association for Machine Translation in the Americas organizes biennial conferences devoted to researchers, commercial users, governmental and NGO users

- **What's New**: 본 논문은 번역 메모리(Translation Memoires, TMs)와 컴퓨터 보조 번역(Computer-Aided Translation, CAT) 도구에서의 구문 정정 기법인 퍼지 매치 리페어(Fuzzy-Match Repair, FMR) 기술을 발전시키는 데 집중하고 있습니다. 특히, 기존의 기계 번역(Machine Translation, MT) 기법 대신 Word2Vec, BERT, 그리고 GPT-4와 같은 머신 러닝(machine learning) 기반 접근 방식을 사용하여 고정된 단어(anchor word) 번역의 정확성을 향상시킬 수 있는 방법을 제시합니다.

- **Technical Details**: 이 연구는 번역 시스템에서 고정된 단어의 번역을 개선하기 위해 네 가지 기술을 실험했습니다: (1) Neural Machine Translation(NMT), (2) BERT 기반 구현, (3) Word2Vec, (4) OpenAI GPT-4. 특히, 고정된 단어는 두 개의 단어 사이에 위치하며, 연속적인 단어의 가방(CBOW) 패러다임을 따른다고 설명합니다. 내재된 문맥(window) 내에서 각 단어에 가중치를 부여하여 주변 단어가 예측에 미치는 영향을 극대화할 수 있음을 강조하고 있습니다.

- **Performance Highlights**: 실험 결과, Word2Vec, BERT 및 GPT-4는 프랑스어에서 영어로의 번역에 있어 기존의 Neural Machine Translation 시스템보다 유사하거나 더 나은 성능을 보였습니다. 특히, 각 접근 방식이 고정된 단어 번역에서 성공적으로 작동하는 경우를 다루고 있습니다.



### Intelligent Energy Management: Remaining Useful Life Prediction and Charging Automation System Comprised of Deep Learning and the Internet of Things (https://arxiv.org/abs/2409.17931)
- **What's New**: 이번 연구는 배터리의 Remaining Useful Life (RUL)을 예측하기 위한 머신 러닝(Machine Learning) 기반 모델의 개발을 목표로 합니다. 또한 IoT(Internet of Things) 개념을 활용하여 충전 시스템을 자동화하고 결함을 관리하는 방법을 제시합니다.

- **Technical Details**: 연구에서는 catboost, Multi-Layer Perceptron (MLP), Gated Recurrent Unit (GRU) 모델 및 혼합 모델을 개발하여 차량의 RUL을 세 가지 클래스로 분류할 수 있는 성능을 보여줍니다. 이 데이터는 tkinter GUI를 통해 입력되고 pyserial 백엔드를 사용하여 Esp-32 마이크로컨트롤러에 연계되어 충전 및 방전 작업을 가능하게 합니다.

- **Performance Highlights**: 모델들은 99% 이상의 정확도로 RUL을 분류할 수 있으며, Blynk IoT 플랫폼을 사용해 다양한 차량 파라미터 간의 관계를 나타내는 그래프를 시뮬레이션합니다. 또한 자동화된 충전 및 에너지 절약 메커니즘을 위한 릴레이 기반 트리거링도 가능합니다.



### Pioneering Reliable Assessment in Text-to-Image Knowledge Editing: Leveraging a Fine-Grained Dataset and an Innovative Criterion (https://arxiv.org/abs/2409.17928)
Comments:
          EMNLP24 Findings

- **What's New**: 이번 연구에서는 Text-to-Image (T2I) diffusion 모델의 지식 편집을 위한 새로운 프레임워크를 제안합니다. 특히, 새로운 데이터셋 CAKE를 제작하고, 평가지표를 개선하는 adaptive CLIP threshold를 도입하며, Memory-based Prompt Editing (MPE) 접근 방식을 통해 효과적인 지식 업데이트를 구현합니다.

- **Technical Details**: T2I 모델의 편집 성능을 평가하기 위해 CAKE라는 데이터셋에서 paraphrase와 다중 객체 테스트를 포함하여 보다 정밀한 평가를 가능하게 합니다. 또한, 기존의 이진 분류 기반의 평가 방식에서 벗어나 이미지가 목표 사실과 '충분히' 유사한지를 측정하는 adaptive CLIP threshold라는 새로운 기준을 제안합니다. 이와 함께, MPE는 외부의 메모리에 모든 사실 편집을 저장하여 입출력 프롬프트의 오래된 부분을 수정합니다. 이를 통해 MPE는 기존 모델 편집기보다 뛰어난 성과를 보여줍니다.

- **Performance Highlights**: MPE 접근법은 기존의 모델 편집 기법보다 전반적인 성능과 적용 가능성에서 더 우수한 결과를 보였습니다. 연구 결과는 T2I 지식 편집 방법의 신뢰할 수 있는 평가를 촉진할 것으로 예상됩니다.



### PhantomLiDAR: Cross-modality Signal Injection Attacks against LiDAR (https://arxiv.org/abs/2409.17907)
- **What's New**: 이 논문은 LiDAR(빛 탐지 및 거리 측정) 시스템의 신뢰성을 위협하는 새로운 공격 벡터인 크로스 모디얼리티 신호 주입 공격(Cross-modality signal injection attacks)의 가능성을 조사합니다. 특히, 의도적인 전자기 간섭(IEMI)을 생성하여 LiDAR 출력을 조작하는 방법을 제시합니다.

- **Technical Details**: PhantomLiDAR 공격을 통해 LiDAR 시스템의 내부 센서와 모듈을 대상으로 하여 Points Interference, Points Injection, Points Removal, LiDAR Power-Off 등의 공격 방식을 구현하였습니다. 이 과정에서 EM 공격 장치를 사용하여 다양한 주파수 대역에서 취약점을 실험적으로 검색하고, 고유한 평가 방법을 통해 효과성을 검증했습니다.

- **Performance Highlights**: PhantomLiDAR는 시뮬레이션 및 실제 실험을 통해 최대 16,000개의 가짜 포인트를 주입할 수 있는 능력을 입증했습니다. 공격 거리 5미터에서도 목표물 숨기기가 가능하며, SOTA 기반 레이저 공격보다 5배 더 많은 포인트를 주입할 수 있는 성과를 달성했습니다.



### Revisiting Acoustic Similarity in Emotional Speech and Music via Self-Supervised Representations (https://arxiv.org/abs/2409.17899)
- **What's New**: 이 연구는 음성 감정 인식(SER)과 음악 감정 인식(MER) 간의 지식을 전이할 수 있는 가능성을 탐구하고, 자가 감독 학습(Self-Supervised Learning, SSL) 모델에서 추출된 공통의 음향 특징을 활용하여 감정 인식 성능을 향상시키는 방법을 제시합니다.

- **Technical Details**: 이 연구에서는 RAVDESS 데이터셋을 사용하여 SSL 모델의 층별 행동을 분석하고, 두 단계의 미세 조정을 통해 SER과 MER 간의 도메인 적응 방법을 비교합니다. 또한, Frechet 오디오 거리(Frechet audio distance)를 사용하여 감정별로 음성 및 음악의 음향 유사성을 평가합니다. 세 가지 SSL 모델(Wav2Vec 2.0, HuBERT, MERT)을 적용하여 각 모델의 음성 및 음악 데이터에서의 성능을 조사합니다.

- **Performance Highlights**: 연구 결과, SER과 MER에서 SSL 모델이 공통된 음향 특징을 잘 포착하긴 하지만, 각기 다른 감정에 따라 그 행동이 다르게 나타납니다. 또한, 효율적인 매개변수 기반 미세 조정을 통해 서로의 지식을 활용하여 SER과 MER 성능을 향상할 수 있음을 보여줍니다.



### Why Companies "Democratise" Artificial Intelligence: The Case of Open Source Software Donations (https://arxiv.org/abs/2409.17876)
Comments:
          30 pages, 1 figure, 5 tables

- **What's New**: 이 연구는 인공지능(AI) 민주화의 상업적 유인을 이해하는 것을 목표로 하며, 43개의 AI 오픈 소스 소프트웨어(OSS) 기부 사례를 분석하여 그에 대한 분류 체계를 제시합니다. 이를 통해 AI 민주화의 다양한 이점이 상업적 이익과 연결되어 있음을 강조합니다.

- **Technical Details**: 이 연구는 혼합 연구 방법(mixed-methods approach)을 사용하여 기업의 AI OSS 기부에 대한 상업적 유인을 조사합니다. 연구에서는 기부 전에 제출된 기술 발표, 기부 후 블로그 글, 설문조사, 반구조화된 인터뷰를 분석하고,OSS 프로젝트의 거버넌스 및 통제 권한을 민주화하는 것의 중요성에 대해 설명합니다.

- **Performance Highlights**: AI OSS 기부는 외부 기여자를 유치하고 개발 비용을 절감하며 산업 표준에 영향을 미치는 등 다운스트림 목표를 위한 구조적 촉진제 역할을 합니다. 개인 개발자의 하향식(bottom-up) 유인이 AI 민주화에 있어 중요함을 강조하며, 다른 AI 민주화 노력에 대한 유인을 이해하는 데 도움이 되는 틀과 도구를 제공합니다.



### Efficient Arbitrary Precision Acceleration for Large Language Models on GPU Tensor Cores (https://arxiv.org/abs/2409.17870)
- **What's New**: 해당 논문은 대형 언어 모델(LLM)의 효율적인 추론을 위한 새로운 가속화 방안을 제안합니다. 특히, 이 논문에서는 대칭 양자화(symmetric quantization)를 지원하는 새롭고 효율적인 데이터 형식인 bipolar-INT를 소개합니다.

- **Technical Details**: 제안된 방법은 주로 세 가지로 구성됩니다: (1) bipolar-INT 데이터 형식 도입; (2) 비트 수준에서 행렬을 분해하고 복원하는 방법으로 임의 정밀도(matrix multiplication에서 비트 수준 분해 및 복구) 행렬 곱셈(arbitrary precision MatMul) 스킴 구현; (3) 효율적인 행렬 전처리(matrix preprocessing) 방법 도입 및 데이터 회복 지향의 메모리 관리(memory management) 시스템 설계로, 이를 통해 GPU Tensor Core 활용을 극대화합니다.

- **Performance Highlights**: 실험 결과, 제안된 스킴은 NVIDIA의 CUTLASS와 비교할 때 행렬 곱셈에서 최대 13배의 속도 향상을 달성하며, LLM에 통합 시 추론 속도에서 최대 6.7배의 가속을 실현합니다. 이로 인해 LLM의 추론 효율이 크게 향상되어 더 넓고 반응성이 뛰어난 LLM 응용을 가능하게 합니다.



### Implementing a Nordic-Baltic Federated Health Data Network: a case repor (https://arxiv.org/abs/2409.17865)
Comments:
          24 pages (including appendices), 1 figure

- **What's New**: 이 논문에서는 북유럽-발트해 지역에서 건강 데이터의 2차 사용을 촉진하기 위해 5개국 6개 기관으로 구성된 연합 건강 데이터 네트워크(federated health data network)를 개발하는 과정에서 얻은 초기 경험을 공유합니다.

- **Technical Details**: 이 연구는 혼합 방법(mixed-method approach)을 사용하여 실험 설계(experimental design)와 실행 과학(implementation science)을 결합하여 네트워크 구현에 영향을 미치는 요소를 평가했습니다. 실험 결과, 중앙 집중식 시뮬레이션(centralized simulation)과 비교할 때, 네트워크는 성능 저하(performance degradation) 없이 기능한다는 것을 발견했습니다.

- **Performance Highlights**: 다학제적 접근 방식(interdisciplinary approaches)을 활용하면 이러한 협력 네트워크(collaborative networks)를 구축하는 데 따른 도전 과제를 해결할 수 있는 잠재력이 있지만, 규제 환경이 불확실하고 상당한 운영 비용(operational costs)이 발생하는 것이 문제로 지적되었습니다.



### A Multimodal Single-Branch Embedding Network for Recommendation in Cold-Start and Missing Modality Scenarios (https://arxiv.org/abs/2409.17864)
Comments:
          Accepted at 18th ACM Conference on Recommender Systems (RecSys '24)

- **What's New**: 이 논문은 추천 시스템에서 콜드 스타트(cold-start) 문제를 해결하기 위한 새로운 방법으로, 멀티모달(single-branch) 임베딩 네트워크를 제안합니다. 이를 통해 다양한 모달리티에 기반한 추천 성능을 개선하고자 합니다.

- **Technical Details**: 제안된 방법은 SiBraR(Single-Branch embedding network for Recommendation)로 불리며, 여러 모달리티 데이터를 공유하는 단일 브랜치 네트워크를 통해 처리합니다. SiBraR는 상호작용 데이터와 여러 형태의 사이드 정보를 동일한 네트워크에서 인코딩하여 모달리티 간의 간극(modality gap)을 줄입니다.

- **Performance Highlights**: 대규모 추천 데이터셋에서 실시한 실험 결과, SiBraR는 콜드 스타트 과제에서 기존의 CF 알고리즘과 최신 콘텐츠 기반 추천 시스템(Content-Based Recommender Systems)보다 유의미하게 우수한 성능을 보였습니다.



### Machine Learning-based vs Deep Learning-based Anomaly Detection in Multivariate Time Series for Spacecraft Attitude Sensors (https://arxiv.org/abs/2409.17841)
Comments:
          Accepted for the ESA SPAICE Conference 2024

- **What's New**: 이번 연구는 우주선의 태세 센서에서 발생하는 다변량 시계열의 고착 값(stuck value) 감지를 위한 두 가지 AI 기반 접근 방식을 분석하고, 전통적인 임계값 검사에서의 한계를 극복하기 위한 혁신적인 기술을 제시합니다.

- **Technical Details**: 연구에서는 머신 러닝(ML)의 XGBoost 알고리즘과 심층 학습(DL) 방법인 다채널 합성곱 신경망(CNN)을 사용하여 다변량 시간 시계열 데이터에서 고착 값을 탐지합니다. 두 방법의 해석 가능성과 일반화 가능성을 논의하며, 특히 고착 값이 발생한 신호에서 AI 기반 FDIR 기능의 효과성을 강조합니다.

- **Performance Highlights**: XGBoost는 해석 가능성을 제공합니다. CNN은 빠른 처리 속도와 더 적은 파라미터로 성능을 극대화하며, 일반적인 우주선의 제한된 계산 자원에도 잘 적응합니다. 두 접근 방식은 AI 기반 FDIR의 성능 향상에 기여하며, 특히 고착 값 감지의 정확성을 높이는 데 효과적입니다.



### Language Models as Zero-shot Lossless Gradient Compressors: Towards General Neural Parameter Prior Models (https://arxiv.org/abs/2409.17836)
Comments:
          To appear in NeurIPS 2024

- **What's New**: 본 논문에서는 신경망 기울기(gradient)에 대한 통계적 사전 모델이 오랫동안 간과되어왔음을 지적하며, 대규모 언어 모델(LLM)이 제로샷(zero-shot) 설정에서 기울기 사전으로 작용할 수 있는 가능성을 제시합니다. 이를 통해 효율적인 기울기 압축 방법인 LM-GC를 개발했습니다.

- **Technical Details**: LM-GC는 LLM과 산술 부호화(arithmetic coding)를 통합하여 기울기를 텍스트와 유사한 형식으로 변환합니다. 이 방법은 기울기의 구조와 LLM이 인식 가능한 기호 사이의 정렬을 유지하며, LLM의 토큰 효율성을 최대 38배 향상시키는 특징이 있습니다.

- **Performance Highlights**: LM-GC는 기존의 손실 없는 압축 방법보다 10%에서 최대 17.2% 더 우수한 압축률을 나타내며, 다양한 데이터세트 및 아키텍처에서 실험을 통해 입증되었습니다. 또한, 본 방법은 양자화 및 희소화와 같은 손실 압축 기법과도 호환 가능성을 보였습니다.



### Inference-Time Language Model Alignment via Integrated Value Guidanc (https://arxiv.org/abs/2409.17819)
Comments:
          EMNLP 2024 Findings

- **What's New**: 본 논문에서는 대형 언어 모델(Large Language Models, LLMs) 조정의 복잡성을 피하면서도 효율적으로 인간의 선호에 부합할 수 있게 하는 새로운 방법인 통합 가치 안내(Integrated Value Guidance, IVG)를 소개합니다. IVG는 암묵적(implicit) 및 명시적(explicit) 가치 함수(value function)를 사용하여 언어 모델의 디코딩(decoding)을 유도합니다.

- **Technical Details**: IVG는 두 가지 형태의 가치 함수를 결합합니다. 암묵적 가치 함수는 각 토큰(token)별 샘플링에 적용되고, 명시적 가치 함수는 청크(chunk) 단위의 빔 탐색(beam search)에 사용됩니다. IVG는 다양한 작업에서의 효과를 검증하며, 전통적인 방법들을 능가합니다. 특히, IVG는 gpt2 기반의 가치 함수로부터의 유도 덕분에 감정 생성과 요약 작업에서 성능을 크게 향상했습니다.

- **Performance Highlights**: IVG는 AlpacaEval 2.0과 같은 어려운 지침 따르기 벤치마크에서, 전문가 튜닝된 모델과 상용 모델 모두에서 길이 제어된 승률이 크게 향상되는 것을 보여줍니다. 예를 들어, Mistral-7B-Instruct-v0.2 모델은 19.51%에서 26.51%로, Mixtral-8x7B-Instruct-v0.1 모델은 25.58%에서 33.75%로 증가했습니다.



### Self-supervised Preference Optimization: Enhance Your Language Model with Preference Degree Awareness (https://arxiv.org/abs/2409.17791)
Comments:
          Accepted at EMNLP 2024 Findings

- **What's New**: 최근에는 대형 언어 모델(LLM)의 보상 모델을 인간 피드백(Human Feedback) 기반 강화 학습(Reinforcement Learning from Human Feedback, RLHF) 방법으로 대체하려는 관심이 증가하고 있습니다. 이 연구에서는 Self-supervised Preference Optimization(SPO) 프레임워크를 제안하여, LLM이 인간의 선호도를 더 잘 이해하고 조정할 수 있도록 합니다.

- **Technical Details**: 본 논문에서는 SPO라는 새로운 자가 감독(preference optimization) 프레임워크를 제안합니다. 이 방법은 LLM의 출력에서 중요한 내용을 선택적으로 제거하여 다양한 선호도 정도의 응답을 생성합니다. 이러한 응답은 자기 감독 모듈로 분류되어 주 손실 함수와 결합되어 LLM을 동시에 최적화합니다.

- **Performance Highlights**: 실험 결과, SPO는 기존의 선호 최적화 방법들과 통합되어 LLM의 성능을 유의미하게 향상시킬 수 있으며, 두 가지 다양한 데이터셋에서 최첨단 결과를 달성했습니다. LLM이 선호도 정도를 구분하는 능력을 높이는 것이 여러 작업에서 성능 향상에 기여한다는 것을 보여주었습니다.



### Harnessing Shared Relations via Multimodal Mixup Contrastive Learning for Multimodal Classification (https://arxiv.org/abs/2409.17777)
Comments:
          RK and RS contributed equally to this work, 20 Pages, 8 Figures, 9 Tables

- **What's New**: 본 논문에서는 M3CoL(Multimodal Mixup Contrastive Learning) 접근 방식을 제안하여, 다양한 모달리티 간의 미묘한 공유 관계를 포착할 수 있음을 보여줍니다. M3CoL은 모달리티 간의 혼합 샘플을 정렬하여 강력한 표현을 학습하는 Mixup 기반의 대조 손실을 활용합니다.

- **Technical Details**: M3CoL은 이미지-텍스트 데이터셋(N24News, ROSMAP, BRCA, Food-101)에서 광범위한 실험을 통해 공유 모달리티 관계를 효과적으로 포착하고, 다양한 도메인에서 일반화 능력을 발휘합니다. 이는 fusion module과 unimodal prediction modules로 구성된 프레임워크에 기반하였으며, Mixup 기반의 대조 손실을 통해 보조 감독을 강화합니다.

- **Performance Highlights**: M3CoL은 N24News, ROSMAP, BRCA 데이터셋에서 최첨단 방법들을 초월하는 성능을 보이며, Food-101에서는 유사한 성능을 달성했습니다. 이를 통해 공유 관계 학습의 중요성이 강조되며, 강력한 다중 모달 학습을 위한 새로운 연구 방향을 열어갑니다.



### Faithfulness and the Notion of Adversarial Sensitivity in NLP Explanations (https://arxiv.org/abs/2409.17774)
Comments:
          Accepted as a Full Paper at EMNLP 2024 Workshop BlackBoxNLP

- **What's New**: 본 논문에서는 설명 가능한 AI의 신뢰성을 평가하기 위한 새로운 접근법인 'Adversarial Sensitivity'를 소개합니다. 이 방식은 모델이 적대적 공격을 받을 때 설명자의 반응에 중점을 둡니다.

- **Technical Details**: Adversarial Sensitivity는 신뢰성을 평가하는 지표로, 적대적 입력 변화에 대한 설명자의 민감도를 포착합니다. 이를 통해 신뢰성이 기존 평가 기술의 중대한 한계를 극복하고, 기존의 설명 메커니즘의 정확성을 정량화합니다.

- **Performance Highlights**: 연구팀은 세 개의 텍스트 분류 데이터셋에 대해 여섯 개의 최첨단 후속 설명기에서 제안된 신뢰성 테스트를 수행하고, 인기 있는 설명기 테스트와의 (비)일관성을 보고하였습니다.



### Federated Learning under Attack: Improving Gradient Inversion for Batch of Images (https://arxiv.org/abs/2409.17767)
Comments:
          5 pages, 7 figures

- **What's New**: 이 논문은 Federated Learning(FL) 시스템의 공격에 대한 새로운 접근 방식, Deep Leakage from Gradients with Feedback Blending (DLG-FB)을 제안합니다. 이 방법은 이미지 배치(batch) 내의 공간적 상관관계를 활용하여 공격 성능을 개선합니다.

- **Technical Details**: DLG-FB는 이미 성공적으로 재구성된 이미지들을 결합하여 공격 초기 입력으로 사용합니다. 기존의 공격 방법들은 매번 무작위 데이터를 초기화했으나, DLG-FB는 재건된 이미지를 혼합하여 더 나은 출발점을 제공합니다. 이로 인해 공격 성공률이 19.18% 향상되었고, 공격당한 이미지당 반복 횟수가 48.82% 감소했습니다.

- **Performance Highlights**: 실험 결과, DLG-FB는 이미지 재구성의 정확성과 효율성을 크게 향상시켰습니다. 특히 다수의 이미지를 목표로 할수록 성능이 더욱 두드러졌습니다.



### Confidence intervals uncovered: Are we ready for real-world medical imaging AI? (https://arxiv.org/abs/2409.17763)
Comments:
          Paper accepted at MICCAI 2024 conference

- **What's New**: 이 논문은 의료 영상(segmentation) 분야에서 AI 성능 변동성을 평가되지 않는 문제를 다루고 있습니다. 2023년 MICCAI에서 발표된 논문 221편을 분석한 결과, 50% 이상의 논문이 성능 변동성을 평가하지 않고, 단 0.5%의 논문만이 신뢰 구간(confidence intervals, CIs)을 보고했습니다. 이는 기존 논문들이 임상 적용을 위한 충분한 근거를 제공하지 않음을 지적합니다.

- **Technical Details**: 연구에서는 segmentation 논문에서 보고되지 않은 표준 편차(standard deviation, SD)를 평균 Dice 유사도 계수(Dice similarity coefficient, DSC)의 2차 다항식 함수를 통해 근사할 수 있음을 보여줍니다. 이를 바탕으로 2023년 MICCAI segmentation 논문의 평균 DSC 주변에 95% CIs를 재구성하였고, 그 중간 CI 폭은 0.03으로 첫 번째와 두 번째 순위 방법 간의 중간 성능 격차보다 세 배 더 큽니다.

- **Performance Highlights**: 60% 이상의 논문에서 두 번째 순위 방법의 평균 성능이 첫 번째 순위 방법의 신뢰 구간 내에 포함되었으며, 이는 현재 보고된 성능이 실제 임상에서의 가능성을 충분히 뒷받침하지 않음을 의미합니다.



### Integrating Hierarchical Semantic into Iterative Generation Model for Entailment Tree Explanation (https://arxiv.org/abs/2409.17757)
- **What's New**: 본 논문은 Controller-Generator 프레임워크(HiSCG)를 기반으로 문장의 계층적 의미(Hierarchical Semantics)를 통합하여 신뢰할 수 있는 설명(Explanation)을 생성하는 새로운 아키텍처를 제안합니다. 이 방법은 동일 계층 및 인접 계층 간의 문장 간 계층적 의미를 처음으로 고려하여 설명의 향상을 이끌어냅니다.

- **Technical Details**: HiSCG 아키텍처는 세 가지 주요 구성 요소로 나뉩니다: 계층적 의미 인코더(Hierarchical Semantic Encoder), 선택 컨트롤러(Selection Controller), 중간 생성 모듈(Intermediate Generation Module). 이 구조는 계층적 연결을 통해 관련 사실을 클러스터링하고, 이러한 사실을 조합하여 결론을 생성하는 과정을 최적화합니다. 각 모듈은 계층적 정보의 활용을 극대화하도록 설계되었습니다.

- **Performance Highlights**: 제안된 방법은 EntailmentBank 데이터셋의 세 가지 설정에서 기존 다른 방법들과 비교하여 동등한 성능을 달성했으며, 두 개의 도메인 외 데이터셋에서 일반화 능력도 입증되었습니다.



### SECURE: Semantics-aware Embodied Conversation under Unawareness for Lifelong Robot Learning (https://arxiv.org/abs/2409.17755)
Comments:
          10 pages,4 figures, 2 tables

- **What's New**: 본 연구는 로봇이 작업 수행에 필요한 개념을 인식하지 못한 상황에서 진행되는 상호작용적 작업 학습(interactive task learning) 문제를 다루며, 이를 해결하기 위한 새로운 프레임워크 SECURE를 제안합니다.

- **Technical Details**: SECURE는 상징적 추론(symbolic reasoning)과 신경 기초(neural grounding)를 결합하여 로봇이 대화 중에 새로운 개념을 인식하고 학습할 수 있도록 설계되었습니다. 이 프레임워크는 서울대학교의 대학원 연구팀이 개발하였으며, 기존의 기계 학습 모델이 가지는 한계를 극복하고 지속적인 개념 학습을 가능하게 합니다.

- **Performance Highlights**: SECURE를 이용한 로봇은 이전에 알지 못했던 개념을 효과적으로 학습하고, 그로 인해 무의식적 인식 상태에서도 작업 재배치 문제를 성공적으로 해결할 수 있음을 보여주었습니다. 이러한 결과는 로봇이 대화의 의미 논리(semantics)적 결과를 활용하여 보다 효과적으로 학습할 수 있음을 입증합니다.



### Byzantine-Robust Aggregation for Securing Decentralized Federated Learning (https://arxiv.org/abs/2409.17754)
Comments:
          18 pages, 7 figures, 1 table

- **What's New**: 본 논문은 분산된 환경에서의 Decentralized Federated Learning(DFL)을 위한 새로운 Byzantine-robust aggregation 알고리즘 WFAgg를 제안합니다. 이는 중앙 서버 없이도 보안을 강화하는데 기여합니다.

- **Technical Details**: WFAgg 알고리즘은 다수의 필터를 사용하여 Byzantine 공격을 분석하고 완화하는 기능을 갖추고 있으며, 동적 분산 토폴로지에서의 강력한 견고성을 제공합니다. 이를 통해 중앙 집중식 Byzantine-robust aggregation 알고리즘과 비교하여 다양한 Byzantine 공격 시나리오에서도 높은 모델 정확성과 수렴성을 유지할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 알고리즘 WFAgg는 여러 중앙 집중식 Byzantine-robust aggregation 알고리즘(예: Multi-Krum, Clustering)과 비교하여 이미지 분류 문제에서 더 높은 정확도를 기록하며, 중앙 집중식 시스템보다 우수한 모델 일관성 결과를 보여줍니다.



### AlterMOMA: Fusion Redundancy Pruning for Camera-LiDAR Fusion Models with Alternative Modality Masking (https://arxiv.org/abs/2409.17728)
Comments:
          17 pages, 3 figures, Accepted by NeurIPS 2024

- **What's New**: 본 논문에서는 카메라-라이다(Camera-LiDAR) 융합 모델의 효율성을 높이기 위해 새로운 가지치기 프레임워크인 AlterMOMA를 제안합니다.

- **Technical Details**: AlterMOMA는 각 모달리티에 대해 대체 마스킹을 사용하고 중요도 평가 함수인 AlterEva를 통해 중복된 파라미터를 식별합니다. 이 과정에서는 한 모달리티 파라미터가 활성화 및 비활성화될 때의 손실 변화를 관찰하여 중복 파라미터를 식별합니다.

- **Performance Highlights**: AlterMOMA는 nuScenes 및 KITTI 데이터셋의 다양한 3D 자율 주행 작업에서 기존의 가지치기 방법보다 뛰어난 성능을 보이며, 최첨단 성능을 달성하였습니다.



### Episodic Memory Verbalization using Hierarchical Representations of Life-Long Robot Experienc (https://arxiv.org/abs/2409.17702)
Comments:
          Code, data and demo videos at this https URL

- **What's New**: 이 논문에서는 로봇 경험을 효과적으로 언어로 표현하기 위해 대규모 사전 훈련된 모델을 활용하는 새로운 접근 방식을 소개합니다. 특히, 긴 경험의 흐름을 요약하고 질의 응답을 수행하는 데 중점을 두고 있으며, 기존의 규칙 기반 시스템이나 세분화된 심층 모델에 대한 의존도를 줄였습니다.

- **Technical Details**: 이 연구에서는 에피소드 기억(Episodic Memory, EM)에서 파생된 계층적 트리 구조를 구성하여 로봇의 경험 흐름을 저장합니다. 로우 레벨 수준에는 원시 지각(raw perception) 및 감각 데이터가 포함되고, 더 높은 수준에서는 사건이 자연어 개념으로 추상화됩니다. 사용자 질의에 대한 답변을 위해 대형 언어 모델(Large Language Model, LLM)을 사용하여 EM에서 필요한 정보를 동적으로 탐색하게 됩니다.

- **Performance Highlights**: H-Emv 시스템은 시뮬레이션된 가정용 로봇 데이터와 실제 세계의 로봇 녹음 데이터를 활용하여 테스트되었으며, 매우 긴 역사 데이터에 대해 효율적으로 확장할 수 있음을 보여주었습니다. 이 시스템은 여러 시간의 시뮬레이션 데이터와 6시간 이상의 실제 인간 영상에서 뛰어난 성능을 발휘하였습니다.



### MoJE: Mixture of Jailbreak Experts, Naive Tabular Classifiers as Guard for Prompt Attacks (https://arxiv.org/abs/2409.17699)
- **What's New**: 이 논문은 Large Language Models (LLMs)의 보안을 강화하기 위한 새로운 방어 메커니즘인 MoJE (Mixture of Jailbreak Expert)를 제안합니다. 기존의 guardrails 방법들이 갖는 한계를 극복하고, 탈옥(jailbreak) 공격 탐지의 정확도와 효율성을 동시에 향상시키는 구조입니다.

- **Technical Details**: MoJE는 간단한 언어 통계 기법을 활용하여, 다양한 토큰화(tokenization) 전략이나 n-그램(n-gram) 특징 추출을 통해 탈옥 공격을 탐지하고 필터링합니다. 이 구조는 기존의 state-of-the-art guardrails에 비해 공격 탐지 정확도, 대기 시간(latency), 처리량(throughput) 면에서 우수한 성능을 보입니다. 또한, 모듈러 특성 덕분에 새로운 공격에 대한 방어 모델이나 OOD(out-of-distribution) 데이터셋을 포함하도록 쉽게 확장 가능합니다.

- **Performance Highlights**: MoJE는 여러 데이터셋에서 유해 내용 탐지에서 기존의 ProtectAI 및 Llama-Guard와 같은 최첨단 솔루션을 초월하며, 탈옥 공격에 대한 저항력이 뛰어납니다. 특히, 90%의 탈옥 공격을 탐지하면서도 정상 프롬프트(benign prompts)에 대한 정확도를 유지하여 LLMs의 보안을 크게 강화합니다.



### MIO: A Foundation Model on Multimodal Tokens (https://arxiv.org/abs/2409.17692)
Comments:
          Technical Report. Codes and models will be available soon

- **What's New**: MIO라는 새로운 기초 모델이 등장했습니다. 이는 음성, 텍스트, 이미지 및 비디오를 이해하고 생성할 수 있는 멀티모달 토큰 기반의 모델로서, end-to-end 및 autoregressive 방식으로 작동합니다. MIO는 기존의 모델들이 갖고 있지 않았던 all-to-all 방식으로의 이해와 생성을 지원하며, 기존의 비공식 모델들(GPT-4o 등)을 대체할 수 있습니다.

- **Technical Details**: MIO는 causal multimodal modeling에 기반해 4단계의 훈련 과정을 거칩니다: (1) alignment pre-training, (2) interleaved pre-training, (3) speech-enhanced pre-training, (4) 다양한 텍스트, 비주얼, 음성 작업에 대한 포괄적인 감독 하에 fine-tuning을 수행합니다. MIO는 discrete multimodal tokens을 사용하여 학습되며, 이는 대조 손실(contrastive loss)과 재구성 손실(reconstruction loss) 기법을 통해 semantical representation과 low-level features를 포착합니다.

- **Performance Highlights**: MIO는 이전의 dual-modal 및 any-to-any 모델들, 심지어 modality-specific baselines와 비교해서 경쟁력 있는 성능을 보이며, interleaved video-text generation, 시각적 사고의 연쇄(chain-of-visual-thought reasoning), 시각적 가이드라인 생성, 이미지 편집 기능 등 고급 기능을 구현합니다.



### Efficient Bias Mitigation Without Privileged Information (https://arxiv.org/abs/2409.17691)
Comments:
          Accepted at the 18th European Conference on Computer Vision (ECCV 2024) as an Oral presentation

- **What's New**: 본 논문에서는 Bias Mitigation을 위한 Targeted Augmentations (TAB)라는 새로운 프레임워크를 제안합니다. 이 방법은 그룹 레이블 없이도 훈련 세트를 재조정하여 바이어스를 감소시키는 효과적인 방법으로, 기존의 방법들보다 뛰어난 성능을 보입니다.

- **Technical Details**: TAB는 하이퍼파라미터 최적화가 필요 없는 간단한 비지도 학습 기반의 바이어스 완화 파이프라인입니다. 이는 보조 모델의 전체 훈련 이력을 활용하여 spurious samples를 식별하고, 그룹 균형 훈련 세트를 생성합니다.

- **Performance Highlights**: TAB는 기존의 비지도 방법들보다 worst-group 성능을 향상시키며 전체 정확도를 유지합니다. 이 방법은 다양한 실제 데이터 세트에 쉽게 적용할 수 있으며, 그룹 정보나 모델 선택 없이도 성능 개선을 이룰 수 있습니다.



### Graph Edit Distance with General Costs Using Neural Set Divergenc (https://arxiv.org/abs/2409.17687)
Comments:
          Published at NeurIPS 2024

- **What's New**: GRAPHEDX는 서로 다른 비용의 edit operations를 명시적으로 고려하여 Graph Edit Distance (GED)를 추정하는 새로운 신경망 모델입니다. 이를 통해 여러 종류의 edit 작업에 대해 보다 정확한 추정을 가능합니다.

- **Technical Details**: GRAPHEDX는 네 가지 edit 작업(edge deletion, edge addition, node deletion, node addition)에 대한 비용을 포함하는 quadratic assignment problem (QAP)으로 GED를 모델링합니다. 각 그래프는 노드와 엣지의 embedding으로 표현되며, Gumbel-Sinkhorn permutation generator를 통해 노드와 엣지 간의 정렬을 학습합니다.

- **Performance Highlights**: 여러 데이터 세트에서 진행된 실험 결과, GRAPHEDX는 예측 오류 측면에서 최신의 방법들과 휴리스틱을 일관되게 초월하는 성능을 보였습니다.



### Preserving logical and functional dependencies in synthetic tabular data (https://arxiv.org/abs/2409.17684)
Comments:
          Submitted to Pattern Recognition Journal

- **What's New**: 이 논문에서는 기존의 기능적 의존성(functioanal dependencies) 외에도 속성 간의 논리적 의존성(logical dependencies)을 도입하였습니다. 이와 함께, 테이블 데이터에서 논리적 의존성을 수량적으로 평가할 수 있는 새로운 방법을 제시합니다.

- **Technical Details**: 새롭게 제안된 방법을 사용하여 여러 최신 합성 데이터 생성 알고리즘을 비교하고, 공개 데이터 세트에서 논리적 및 기능적 의존성을 보존하는 능력을 시험합니다. 이러한 연구는 기능적 의존성을 완전히 보존하는 합성 테이블 데이터 생성 알고리즘의 한계를 밝혀냅니다.

- **Performance Highlights**: 본 연구에서는 특정 합성 데이터 생성 모델들이 속성 간의 논리적 의존성을 잘 보존할 수 있음을 보여주며, 향후 작업에 특화된 합성 테이블 데이터 생성 모델 개발의 필요성과 기회를 제시합니다.



### Zero- and Few-shot Named Entity Recognition and Text Expansion in Medication Prescriptions using ChatGP (https://arxiv.org/abs/2409.17683)
- **What's New**: 이번 연구는 ChatGPT 3.5를 사용하여 약물 처방의 데이터 통합 및 해석을 자동화하여 사용자와 기계 모두에게 이해하기 쉬운 형식으로 제공하는 방법을 제시합니다. 이는 자유 텍스트 형태의 약물 진술에서 의미 있는 정보를 구조화하고 확장할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 연구는 Named-Entity Recognition (NER) 및 Text Expansion (EX) 기술을 활용하였습니다. NER은 약물 이름, 농도, 단위, 투여 경로 및 복용 지침을 식별하며, EX는 이를 명확하고 구체적인 형태로 변환합니다. 최적화된 프롬프트를 사용해 NER의 평균 F1 점수는 0.94에 도달하였고, EX는 0.87에 도달하였습니다.

- **Performance Highlights**: 이 연구는 ChatGPT를 사용한 NER 및 EX 작업에서 우수한 성능을 입증하였습니다. 특히, few-shot 학습 접근법을 통해 잘못된 정보를 생성하는 현상(hallucination)을 방지할 수 있었으며, 이는 약물 안전 데이터 처리 시 중요합니다.



### Prototype based Masked Audio Model for Self-Supervised Learning of Sound Event Detection (https://arxiv.org/abs/2409.17656)
Comments:
          Submitted to ICASSP2025; The code for this paper will be available at this https URL after the paper is accepted

- **What's New**: 본 논문은 라벨이 없는 데이터를 더 효과적으로 활용하기 위해 프로토타입 기반 마스킹 오디오 모델(Prototype based Masked Audio Model, PMAM) 알고리즘을 제안합니다. 이는 음향 사건 탐지(Sound Event Detection, SED)에서 자가 지도 표현 학습을 위한 새로운 접근 방법으로, Gaussian mixture model(GMM)을 사용한 의미론적으로 풍부한 프레임 수준의 의사 레이블을 생성합니다.

- **Technical Details**: PMAM은 두 개의 주요 구성 요소로 이루어져 있으며, 여기에는 인코더 네트워크와 컨텍스트 네트워크가 포함됩니다. 인코더 네트워크는 스펙트로그램에서 프레임 수준의 잠재 표현(latent embeddings)을 추출하고, 컨텍스트 네트워크는 마스킹된 오디오 모델 작업을 통해 사운드 사건의 시간적 의존성을 모델링합니다. 트랜스포머 및 CNN 구조를 채택하고, 프로토타입 기반의 이진 교차 엔트로피 손실을 사용하여 다양한 프로토타입에 대한 독립적인 예측을 가능하게 합니다.

- **Performance Highlights**: DESED 작업을 이용한 유사 실험에서 PMAM은 62.5%의 PSDS1 점수를 기록하며 기존 최첨단 모델을 뛰어넘는 성능을 보여주었습니다. 이는 제안된 기술의 우수성을 입증합니다.



### AssistantX: An LLM-Powered Proactive Assistant in Collaborative Human-Populated Environmen (https://arxiv.org/abs/2409.17655)
Comments:
          6 pages, 8 figures, 4 tables

- **What's New**: 최근 연구들은 인간이 밀집한 환경에서의 지능형 비서에 대한 수요 증가에 대응하여 자율 로봇 시스템에 중점을 두고 있습니다. 특히, Large Language Models (LLMs)의 발전이 이러한 시스템을 개선할 새로운 길을 열었습니다.

- **Technical Details**: 본 연구에서는 AssistantX라는 LLM 기반의 능동 비서를 소개하며, 물리적 사무 환경에서 자율적으로 작동하도록 설계되었습니다. 새로운 다중 에이전트 아키텍처인 PPDR4X를 활용하여 복잡한 상황에서도 효과적으로 작업을 수행할 수 있습니다.

- **Performance Highlights**: AssistantX는 명확한 지시에 반응하고, 메모리에서 추가 정보를 능동적으로 검색하며, 타 팀원과 협력하여 작업의 성공적인 완료를 위해 노력하는 등 강력한 성능을 보여줍니다.



### T3: A Novel Zero-shot Transfer Learning Framework Iteratively Training on an Assistant Task for a Target Task (https://arxiv.org/abs/2409.17640)
- **What's New**: 이 논문에서는 T3이라 약칭되는 새로운 제로-샷 (zero-shot) 전이 학습 프레임워크를 제안하여, 긴 텍스트 요약(Long Text Summarization) 작업을 위한 효과적인 솔루션을 제공합니다. 이 프레임워크는 보조 작업(assistant task)과 목표 작업(target task) 간의 관계를 활용하여 LLM을 학습시킵니다.

- **Technical Details**: T3는 보조 작업으로써 질문 답변(Question Answering, QA)을 활용하며, 이를 통해 긴 텍스트 요약 작업에 대한 LLM을 훈련합니다. 이 과정에서 QA는 풍부한 공개 데이터 세트를 제공하고, 질문-답변 쌍을 통해 더 많은 개체와 관계를 포함할 수 있어 요약의 질적 성장을 도모합니다. 또한, 질문 생성(Question Generation, QG)을 통해 두 작업 간의 문맥적 특성을 이해하게 됩니다.

- **Performance Highlights**: T3는 BBC summary, NarraSum, FairytaleQA, NLQuAD 데이터셋에서 3개의 기준 LLM에 비해 ROUGE에서 최대 14%, BLEU에서 35%, Factscore에서 16%의 향상치를 보여주며 그 효과성을 입증했습니다.



### P4Q: Learning to Prompt for Quantization in Visual-language Models (https://arxiv.org/abs/2409.17634)
- **What's New**: 이 논문에서는 Vision-Language Models (VLMs) 의 양자화(quantization)와 미세 조정(fine-tuning)을 통합하는 새로운 방법인 'Prompt for Quantization'(P4Q)을 제안합니다. 이 방법은 PTQ(Post-Training Quantization) 모델의 인식 성능을 향상시키기 위해 경량 아키텍처를 설계합니다.

- **Technical Details**: P4Q는 이미지 특징과 텍스트 특징 간의 격차를 줄이기 위해 학습 가능한 프롬프트(prompt)를 사용하여 텍스트 표현을 재구성하고, 저비트(low-bit) 어댑터(QAdapter)를 사용하여 이미지와 텍스트 특징 분포를 재조정하는 방법입니다. 또한 코사인 유사도 예측을 기반으로 하는 증류 손실(distillation loss)을 도입하여 확장성 있는 증류를 수행합니다.

- **Performance Highlights**: P4Q 방법은 이전 연구보다 뛰어난 성능을 보이며, ImageNet 데이터셋에서 8비트 P4Q가 CLIP-ViT/B-32을 4배 압축하면서도 Top-1 정확도 66.94%를 기록하였습니다. 이는 전체 정밀(full-precision) 모델보다 2.24% 향상된 결과입니다.



### Hand-object reconstruction via interaction-aware graph attention mechanism (https://arxiv.org/abs/2409.17629)
Comments:
          7 pages, Accepted by ICIP 2024

- **What's New**: 이 연구에서는 손과 객체의 상호작용을 고려한 새로운 그래프 주의(attention) 메커니즘을 제안합니다. 기존 방법론이 그래프의 엣지 연관성을 충분히 활용하지 못했던 점을 극복하기 위해, 공통 관계 엣지(common relation edges)와 주의 유도 엣지(attention-guided edges)를 도입하여 물리적 타당성을 향상시키기 위한 그래프 기반 개선 방법을 소개합니다.

- **Technical Details**: 제안된 접근 방식은 상호작용 인식을 위한 그래프 주의 메커니즘을 통해 손과 객체의 메쉬(mesh)를 이미지에서 추정하여, 두 가지 타이프의 엣지(즉, intra-class와 inter-class 노드 간의 연결)를 사용하여 밀접하게 상관된 노드 간의 관계를 설정합니다. 이 연구는 ObMan와 DexYCB 데이터셋을 활용하여 제안된 방법의 효과성을 평가했습니다.

- **Performance Highlights**: 실험 결과, 손과 객체 간의 물리적 타당성이 개선되었음을 확인했습니다. 정량적 및 정성적으로 관찰된 결과는 제안된 상호작용 인식 그래프 주의 메커니즘이 손과 객체 포즈 추정을 통해 물리적 타당성을 획기적으로 향상시킴을 보여줍니다.



### Neural P$^3$M: A Long-Range Interaction Modeling Enhancer for Geometric GNNs (https://arxiv.org/abs/2409.17622)
Comments:
          Published as a conference paper at NeurIPS 2024

- **What's New**: Neural P$^3$M은 기하학적 그래프 신경망(GNN)의 한계를 극복하기 위해 망 점(mesh points)과 원자(Atoms)를 함께 통합한 새로운 프레임워크입니다. 이를 통해 대규모 분자 시스템에서의 장거리 상호작용을 효과적으로 캡처할 수 있습니다.

- **Technical Details**: Neural P$^3$M은 전통적인 수학적 연산을 훈련 가능한 방식으로 재구성하여 원자와 메쉬 스케일에서 단거리와 장거리 상호작용을 포착합니다. 기존의 LSRM 및 Ewald MP 방법과 비교하여 유연성과 계산 효율성을 높였습니다. 인접한 원자 및 메쉬 간의 정보 교환을 포함하여 장거리 항(terms)의 수식을 가능하게 합니다.

- **Performance Highlights**: Neural P$^3$M은 MD22 데이터셋에서 최첨단 성능을 달성하며, OE62 데이터셋에서는 평균 22% 개선된 에너지 평균 절대 오차(MAE)를 기록하였습니다. 여러 기하학적 GNN과 통합하여 Ag 및 MD22 벤치마크에서 상당한 개선을 이루었습니다.



### Open Digital Rights Enforcement Framework (ODRE): from descriptive to enforceable policies (https://arxiv.org/abs/2409.17602)
Comments:
          20 pages, 3 Figures, Submitted to Computers & Security journal

- **What's New**: 이 논문에서는 Open Digital Rights Language (ODRL) 정책에 집행 기능을 추가하기 위한 Open Digital Rights Enforcement (ODRE) 프레임워크를 소개합니다. ODRE는 정책을 기술할 수 있는 새로운 접근 방식을 제공하며, 동적 데이터 처리 및 함수 평가가 가능하도록 다양한 언어와 결합합니다.

- **Technical Details**: ODRE 프레임워크는 ODRL에서 기술된 정책을 집행할 수 있도록 적응된 알고리즘을 포함합니다. 이 알고리즘은 정책에 선언된 데이터 사용 제약이 충족되었는지를 검증하고, 필요시 정책이 명시한 작업을 호출합니다. 알고리즘은 동적 데이터 처리 메커니즘을 구현하여 개인 정보 유출을 방지합니다.

- **Performance Highlights**: ODRE의 두 가지 오픈 소스 구현체 (Python과 Java)를 통해 여러 실험을 진행하였으며, 구현의 성능과 확장성 특징을 보여주는 긍정적인 결과가 도출되었습니다. 각 구현은 24개의 단위 시험을 통과하며, 정책을 집행하는 데 소요된 시간을 보고합니다.



### TA-Cleaner: A Fine-grained Text Alignment Backdoor Defense Strategy for Multimodal Contrastive Learning (https://arxiv.org/abs/2409.17601)
- **What's New**: 이 논문에서는 CLIP와 같은 대규모 사전학습 모델이 데이터 중독된 백도어 공격에 취약하다는 점을 지적하며, 이를 해결하기 위한 새로운 방어 기법 TA-Cleaner를 제안합니다.

- **Technical Details**: TA-Cleaner는 CleanCLIP의 한계를 보완하기 위해 세밀한 텍스트 정렬 기법을 적용합니다. 매 에포크마다 긍정 및 부정 서브텍스트를 생성하고 이를 이미지와 정렬하여 텍스트 자기 감독(self-supervision)을 강화함으로써 백도어 트리거의 특징 연결을 차단합니다.

- **Performance Highlights**: 특히 BadCLIP과 같은 새로운 공격 기법에 대해 TA-Cleaner는 CleanCLIP보다 Top-1 ASR을 52.02%, Top-10 ASR을 63.88% 감소시키며 뛰어난 방어 성능을 보임을 보여주었습니다.



### Subjective and Objective Quality-of-Experience Evaluation Study for Live Video Streaming (https://arxiv.org/abs/2409.17596)
Comments:
          14 pages, 5 figures

- **What's New**: 최근 라이브 비디오 스트리밍에 대한 QoE(Quality of Experience) 평가를 위해 새로운 데이터셋인 TaoLive QoE가 발표되었습니다. 이 데이터셋은 실시간 방송에서 수집된 42개의 소스 비디오와 다양한 스트리밍 왜곡으로 인해 손상된 1,155개의 왜곡 비디오로 구성되어 있습니다.

- **Technical Details**: TaoLive QoE 데이터셋을 기반으로 주관적 및 객관적 QoE 평가를 실시하였으며, 실시간 콘텐츠에 대한 기존 QoE 모델의 한계를 강조했습니다. 특히, 새로운 end-to-end QoE 평가 모델인 TAO-QoE는 다중 스케일의 의미(feature)와 광학 흐름(optical flow) 기반 모션 특징을 통합하여 QoE 점수를 예측하는 혁신적인 접근 방식을 제안합니다.

- **Performance Highlights**: TAO-QoE는 기존 QoE 모델들보다 라이브 비디오 콘텐츠에 대해 더 정확한 평가 성능을 보여주며, 통계적 QoS 특성에 대한 의존성을 제거하여 실시간 방송 시나리오에서 우수한 성과를 달성하는 것을 목표로 하고 있습니다.



### Deep Manifold Part 1: Anatomy of Neural Network Manifold (https://arxiv.org/abs/2409.17592)
- **What's New**: 이 논문은 Neural Network (신경망)의 수학적 프레임워크인 Deep Manifold를 개발하여, 신경망의 수치 계산 특성과 학습 가능성을 탐구합니다. 주요 발견 사항으로는 신경망이 거의 무한한 자유도와 깊이에 따른 지수적인 학습 능력을 갖고 있다는 것을 들 수 있습니다.

- **Technical Details**: Neural network learning space (신경망 학습 공간)와 deep manifold space (딥 매니폴드 공간)라는 두 가지 개념을 새롭게 정의하였습니다. 또한, neural network intrinsic pathway (신경망 내재 경로) 및 fixed point (고정 점)라는 새로운 개념을 소개하였습니다. 고차 비선형(High-order non-linearity)을 다루는 Numerical Manifold Method를 통해 신경망의 해부학적 구조를 연구합니다.

- **Performance Highlights**: Deep Manifold는 LLM (Large Language Model) 훈련 과정에서 부정적 시간(Negative time)의 중요성을 강조하며, 데이터의 위치 임베딩을 통한 암묵적 시간 인코딩이 LLM의 훈련 성능에 미치는 영향을 분석합니다. 연구 결과, 신경망 모델의 학습 능력이 비선형성을 어떻게 극복하는지에 대한 통찰력을 얻을 수 있었습니다.



### Improving Fast Adversarial Training via Self-Knowledge Guidanc (https://arxiv.org/abs/2409.17589)
Comments:
          13 pages

- **What's New**: 이 논문에서는 Fast Adversarial Training (FAT)에서 발생하는 불균형 문제를 체계적으로 조사하고, 이를 해결하는 Self-Knowledge Guided FAT (SKG-FAT) 방법론을 제안합니다. SKG-FAT는 각 클래스의 학습 상태에 따라 차별화된 정규화 가중치를 할당하고, 학습 정확도에 따라 레이블을 동적으로 조정함으로써/adversarial robustness를 증가시키는 데 초점을 맞춥니다.

- **Technical Details**: FAT의 기존 방법들은 모든 훈련 데이터를 균일하게 최적화하는 전략을 사용하여/imbalanced optimization을 초래합니다. 본 연구에서는 클래스 간의 성능 격차를 드러내고, 이를 해소하기 위해/self-knowledge guided regularization과/self-knowledge guided label relaxation을 도입합니다. SKG-FAT는 자연적으로 생성되는 지식을 활용하여 adversarial robustness를 증대시킬 수 있습니다.

- **Performance Highlights**: SKG-FAT는 네 가지 표준 데이터셋에 대한 광범위한 실험을 통해/adversarial robustness를 향상시키면서도 경쟁력 있는 clean accuracy를 유지하며, 최신 방법들과 비교해 우수한 성능을 보였습니다.



### Multimodal Banking Dataset: Understanding Client Needs through Event Sequences (https://arxiv.org/abs/2409.17587)
- **What's New**: 이 논문은 산업 규모의 공개 멀티모달 은행 데이터셋인 Multimodal Banking Dataset (MBD)를 소개합니다. 이 데이터셋은 150만 개 이상의 법인 고객 데이터를 포함하고 있으며, 다양한 소스에서 수집된 대량의 시퀀스 정보를 제공합니다.

- **Technical Details**: MBD는 9억 5천만 건의 은행 거래, 10억 건의 지리적 이벤트, 500만 개의 기술 지원 대화 임베딩, 4개의 은행 제품에 대한 월별 집계 구매를 포함합니다. 데이터는 클라이언트 개인 정보 보호를 위해 적절히 익명화되어 있습니다. 이 데이터셋은 두 가지 비즈니스 태스크(종합 캠페인 및 클라이언트 매칭)에 대한 기준을 제공합니다.

- **Performance Highlights**: MBD를 사용하여 다중 모달 베이스라인이 단일 모달 기법보다 각 태스크에 대해 우수함을 입증하는 수치 결과를 제공합니다. 이 데이터셋은 향후 이벤트 시퀀스에 대한 대규모 및 다중 모달 알고리즘 개발을 촉진할 수 있는 새로운 관점을 열어줄 잠재력을 가지고 있습니다.



### Let the Quantum Creep In: Designing Quantum Neural Network Models by Gradually Swapping Out Classical Components (https://arxiv.org/abs/2409.17583)
Comments:
          50 pages (including Appendix), many figures, accepted as a poster on QTML2024. Code available at this https URL

- **What's New**: 이 논문에서는 양자 신경망(Quantum Neural Network, QNN)의 구조적 한계를 극복하기 위해 고전적 신경망과 양자 신경망 사이의 점진적인 전환 전략, HybridNet을 제안합니다. 이는 정보 흐름을 유지하면서 고전적 신경망 레이어를 점진적으로 양자 레이어로 대체하는 프레임워크를 제공합니다.

- **Technical Details**: 제안된 HybridNet은 고전적 모델에서 양자 모델로의 전환을 통해 양자 구성이 신경망의 성능에 미치는 영향을 보다 면밀히 분석합니다. 연구에서는 FlippedQuanv3x3라는 새로운 양자 커널과 데이터 재업로드 회로(Data Reuploading Circuit)를 도입하여 고전적 선형 레이어와 동일한 입력 및 출력을 공유하는 양자 레이어를 구현합니다.

- **Performance Highlights**: MNIST, FashionMNIST, CIFAR-10 데이터셋에 대한 수치 실험을 통해 양자 구성요소의 체계적인 도입이 성능 변화에 미치는 영향을 분석했습니다. 연구 결과, 기존의 QNN 모델보다 더 효과적인 성능을 발휘할 수 있음을 발견하였습니다.



### Enhancing Structured-Data Retrieval with GraphRAG: Soccer Data Case Study (https://arxiv.org/abs/2409.17580)
- **What's New**: Structured-GraphRAG는 복잡한 데이터셋에서 정보 검색의 정확성과 관련성을 향상시키기 위해 설계된 새로운 프레임워크입니다. 이 방법은 전통적인 데이터 검색 방법의 한계를 극복하고, 구조화된 데이터셋에서 자연어 질의를 통한 정보 검색을 지원합니다.

- **Technical Details**: Structured-GraphRAG는 여러 개의 지식 그래프(knowledge graph)를 활용하여 데이터 간의 복잡한 관계를 포착합니다. 이를 통해 더욱 세밀하고 포괄적인 정보 검색이 가능하며, 결과의 신뢰성을 높이고 언어 모델의 출력을 구조화된 형태로 바탕으로 답변을 제공합니다.

- **Performance Highlights**: Structured-GraphRAG는 전통적인 검색 보완 생성 기법과 비교하여 쿼리 처리 효율성을 크게 향상시켰으며, 응답 시간을 단축시켰습니다. 실험의 초점이 축구 데이터에 맞춰져 있지만, 이 프레임워크는 다양한 구조화된 데이터셋에 널리 적용될 수 있어 데이터 분석 및 언어 모델 응용 프로그램의 향상된 도구로 기능합니다.



### Dr. GPT in Campus Counseling: Understanding Higher Education Students' Opinions on LLM-assisted Mental Health Services (https://arxiv.org/abs/2409.17572)
Comments:
          5 pages

- **What's New**: 이번 연구는 대학생들이 AI 응용 프로그램, 특히 Large Language Models (LLMs)를 통해 정신 건강을 향상시킬 수 있는 방안에 대한 인식을 조사했습니다. 파일럿 인터뷰를 통해 다섯 가지 시나리오에서 LLMs의 사용에 대한 학생들의 의견을 탐색했습니다.

- **Technical Details**: 대학생들의 정신 건강에 대한 LLMs의 잠재력은 초기 스크리닝(Initial Screening) 및 후속 치료(Follow-up Care) 시나리오에서 특히 높게 평가되었습니다. 연구는 LLM의 맞춤형 상호작용과 정기적인 체크인 기능이 학생들에게 매우 유용하다는 것을 보여줍니다. 여러 연구는 LLMs가 치료적 대화와 데이터 검색을 통해 정신 건강 개입을 강화하는 데 기여할 수 있음을 시사합니다.

- **Performance Highlights**: 참여자들은 LLM이 초기 스크리닝에서 정신 건강 문제를 보다 효과적으로 표현할 수 있게 해주며, 후속 치료에서는 치료 계획의 이행을 지원하는 데 긍정적인 역할을 할 수 있다고 인식하였습니다. 그러나 LLM이 감정적 지원을 제공하는 데 있어 한계가 있음을 우려하는 목소리도 있었습니다.



### Pixel-Space Post-Training of Latent Diffusion Models (https://arxiv.org/abs/2409.17565)
- **What's New**: 이번 논문에서는 Latent Diffusion Models (LDMs)의 한계를 극복하기 위한 새로운 접근 방법을 제안합니다. LDMs는 이미지 생성 분야에서 큰 발전을 이루었지만, 고주파 세부 사항과 복잡한 구성 생성에서 여전히 어려움이 있음을 지적합니다.

- **Technical Details**: LDMs가 고주파 세부 사항을 잘 생성하지 못하는 이유는 기존 학습이 보통 $8 	imes 8$의 낮은 공간 해상도에서 이뤄지기 때문이라고 가설합니다. 이를 해결하기 위해 포스트 트레이닝 과정에서 pixel-space supervision을 추가하는 방법을 제안하였습니다.

- **Performance Highlights**: 실험 결과, pixel-space objective를 추가함으로써 최첨단 DiT transformer와 U-Net diffusion 모델의 supervised quality fine-tuning 및 preference-based post-training에서 시각적 품질 및 결함 지표가 크게 향상되었으며, 동일한 텍스트 정렬 품질을 유지했습니다.



### Triple Point Masking (https://arxiv.org/abs/2409.17547)
- **What's New**: 본 논문에서는 제한된 데이터 환경에서 기존 3D 마스크 학습 방법의 성능 한계를 극복하기 위한 새로운 Triple Point Masking (TPM) 기법을 소개합니다. 이 기법은 다중 마스크 학습을 위한 확장 가능한 프레임워크로, 3D 포인트 클라우드 데이터를 위한 masked autoencoder의 사전 학습을 지원합니다.

- **Technical Details**: TPM은 기본 모델에 두 가지 추가 마스크 선택지(중간 마스크 및 낮은 마스크)를 통합하여 객체 복구 과정을 다양한 방법으로 표현할 수 있도록 설계되었습니다. 이 과정에서 고차원 마스킹 기법의 한계를 극복하고, 다양한 3D 객체에 대한 여러 표현을 고려하여 더 유연하고 정확한 완성 능력을 제공합니다. 또한, SVM 기반의 최적 가중치 선택 모듈을 통해 다운스트림 네트워크에 최적의 가중치를 적용하여 선형 정확성을 극대화합니다.

- **Performance Highlights**: TPM을 탑재한 네 가지 기본 모델은 다양한 다운스트림 작업에서 전반적인 성능 향상을 달성했습니다. 예를 들어, Point-MAE는 TPM 적용 후 사전 학습 및 미세 조정 단계에서 각각 1.1% 및 1.4%의 추가 성과를 보였습니다.



### Modulated Intervention Preference Optimization (MIPO): Keey the Easy, Refine the Difficu (https://arxiv.org/abs/2409.17545)
Comments:
          8pages, submitted to AAAI 2025

- **What's New**: 이번 연구에서는 Modulated Intervention Preference Optimization (MIPO)이라는 새로운 선호 최적화 알고리즘을 제안합니다. MIPO는 주어진 데이터와의 정렬 정도에 따라 참조 모델의 개입 수준을 조절하여 이전 방법들의 한계를 극복합니다.

- **Technical Details**: MIPO는 주어진 데이터가 참조 모델과 얼마나 잘 정렬되어 있는지를 평가하기 위해 평균 로그 가능성(average log likelihood)을 사용합니다. 이 값을 기반으로 MIPO는 정책 모델의 훈련 목표를 조정하여 잘 정렬되지 않은 데이터 쌍에 대해 더 많은 훈련을 수행할 수 있도록 합니다.

- **Performance Highlights**: MIPO는 Alpaca Eval 2.0 및 MT-Bench를 활용한 실험에서 DPO보다 일관되게 우수한 성능을 보였습니다. Llama3-8B-Instruct 모델의 경우 DPO보다 약 9점 (+36.07%), Mistral-7B-Base 모델에서는 약 8점 (+54.24%) 향상된 결과를 나타냈습니다. MIPO는 다양한 실험 환경에서 가장 뛰어난 성능을 달성했습니다.



### On the Implicit Relation Between Low-Rank Adaptation and Differential Privacy (https://arxiv.org/abs/2409.17538)
- **What's New**: 이번 논문에서는 자연어 처리에서의 저차원(adaptation) 접근법이 데이터 프라이버시(data privacy)와 어떻게 연결되는지를 제시합니다. 특히, LoRA(Lo-Rank Adaptation)와 FLoRA(Fully Low-Rank Adaptation)의 방법이 미치는 영향을 분석하여, 이들이 데이터 민감도를 고려한 저차원 적응과 유사하다는 것을 보여줍니다.

- **Technical Details**: LoRA와 FLoRA는 언어 모델 언어를 특정 작업에 적응시키기 위해 몇 개의 레이어에 훈련 가능한 저차원 분해 매트릭스(adapter)를 통합하여, 사전 훈련된 모델의 가중치를 고정한 상태에서 사용됩니다. 이 접근법은 전통적인 매개변수 조정(full fine-tuning) 방식에 비해 필요한 훈련 가능한 매개변数의 수를 상당히 줄입니다. 연구진은 또한 저차원 적응이 DPSGD(Differentially Private Stochastic Gradient Descent)와 근본적으로 유사하다는 것을 입증하며, 가우시안 분포(Gaussian distribution)와의 변동성(variance)도 분석합니다.

- **Performance Highlights**: 연구의 주요 기여는 다음과 같습니다: 1) LoRA/FLoRA로 저차원 적응을 수행하는 것이 어댑터의 배치 그라디언트(batch gradients)에 무작위 노이즈를 주입하는 것과 동등하다는 것을 보여줍니다. 2) 주입된 노이즈의 분산을 찾아 노이즈가 입력 수와 저차원 적응의 순위(rank)가 증가함에 따라 가우시안 분포에 가까워지게 됨을 증명합니다. 3) 저차원 적응의 동역학은 DP 완전 조정(DP full fine-tuning) 어댑터와 매우 유사함을 입증하며, 이러한 저차원 적응이 데이터 프라이버시를 제공할 수 있는 가능성을 제시합니다.



### SimVG: A Simple Framework for Visual Grounding with Decoupled Multi-modal Fusion (https://arxiv.org/abs/2409.17531)
Comments:
          21pages, 11figures, NeurIPS2024

- **What's New**: 이번 연구에서는 Visual Grounding (VG) 문제를 해결하기 위한 새로운 프레임워크 SimVG를 제안합니다. 기존의 복잡한 모듈이나 아키텍처 대신, SimVG는 멀티모달 전이 학습 모델을 활용하여 시각-언어 기능 융합을 하위 작업으로부터 분리합니다.

- **Technical Details**: SimVG는 기존의 멀티모달 모델을 기반으로 하며, 객체 토큰(Object Tokens)을 포함하여 하위 작업(Task)과 사전 학습(Pre-training) 작업을 깊게 통합하는 방식으로 설계되었습니다. 동적 가중치 균형 증류(DWBD) 방법을 사용하여 다중 브랜치 동기식 학습 과정에서 간단한 브랜치의 표현력을 향상시킵니다. 이 브랜치는 경량 MLP로 구성되어, 구조를 단순화하고 추론 속도를 개선합니다.

- **Performance Highlights**: SimVG는 RefCOCO/+/g, ReferIt, Flickr30K 등 총 6개의 VG 데이터셋에서 실험을 실시한 결과, 효율성과 수렴 속도에서 개선을 이루었으며, 새로운 최첨단 성능을 달성했습니다. 특히, 단일 RTX 3090 GPU에서 RefCOCO/+/g 데이터셋에 대해 12시간 훈련하여 이룬 성과가 주목할 만합니다.



### Drone Stereo Vision for Radiata Pine Branch Detection and Distance Measurement: Integrating SGBM and Segmentation Models (https://arxiv.org/abs/2409.17526)
- **What's New**: 이 연구에서는 드론 기반의 가지치기 시스템을 개발하여, 전통적인 수동 가지치기의 안전 위험을 해결하고자 합니다. 이 시스템은 전문적인 가지치기 도구와 스테레오 비전 카메라를 활용하여 가지를 정확하게 감지하고 자를 수 있는 기능을 제공합니다.

- **Technical Details**: 시스템은 YOLO와 Mask R-CNN을 포함한 딥 러닝 알고리즘을 사용하여 가지 감지를 정확하게 수행하며, Semi-Global Matching 알고리즘을 통합하여 신뢰성 있는 거리 추정을 가능하게 합니다. 이를 통해 드론은 가지의 위치를 정밀하게 파악하고 효율적인 가지치기를 수행할 수 있습니다.

- **Performance Highlights**: 실험 결과에 따르면, YOLO와 SGBM의 결합 구현이 드론이 가지를 정확하게 감지하고 드론으로부터의 거리를 측정하는 데 성공적이라는 것을 보여줍니다. 이 연구는 가지치기 작업의 안전성과 효율성을 향상시키는 한편, 농업 및 임업 관행의 자동화를 위한 드론 기술의 발전에 중요한 기여를 합니다.



### EAGLE: Egocentric AGgregated Language-video Engin (https://arxiv.org/abs/2409.17523)
Comments:
          Accepted by ACMMM 24

- **What's New**: EAGLE (Egocentric AGgregated Language-video Engine) 모델과 EAGLE-400K 데이터셋을 소개하여, 다양한 egocentric video 이해 작업을 통합하는 단일 프레임워크를 제공합니다.

- **Technical Details**: EAGLE-400K는 400K개의 다양한 샘플로 구성된 대규모 instruction-tuning 데이터셋으로, 활동 인식부터 절차 지식 학습까지 다양한 작업을 향상시킵니다. EAGLE는 공간적(spatial) 및 시간적(temporal) 정보를 효과적으로 캡처할 수 있는 강력한 비디오 멀티모달 대형 언어 모델(MLLM)입니다.

- **Performance Highlights**: 광범위한 실험을 통해 EAGLE는 기존 모델보다 우수한 성능을 발휘하며, 개별 작업에 대한 이해와 비디오 내용에 대한 전체적인 해석을 균형 있게 수행할 수 있는 능력을 강조합니다.



### Robotic Environmental State Recognition with Pre-Trained Vision-Language Models and Black-Box Optimization (https://arxiv.org/abs/2409.17519)
Comments:
          Accepted at Advanced Robotics, website - this https URL

- **What's New**: 이번 연구에서는 로봇이 다양한 환경에서 자율적으로 탐색하고 작동하기 위해 필요로 하는 환경 상태 인식을 위한 새로운 방법을 제안합니다. 특히, 사전 훈련된 대규모 Vision-Language Models (VLMs)를 활용하여 환경 상태를 통합적으로 인식할 수 있는 방법을 개발하였습니다.

- **Technical Details**: VLM을 사용하여 Visual Question Answering (VQA) 및 Image-to-Text Retrieval (ITR) 작업을 수행합니다. 이를 통해 로봇은 문이 열려 있는지, 물이 흐르고 있는지와 같은 다양한 환경 상태를 인식할 수 있습니다. 또한, 블랙박스 최적화를 통해 적절한 텍스트를 선택하는 방식으로 인식 정확도를 향상시킬 수 있습니다.

- **Performance Highlights**: 실험을 통해 제안된 방법의 효과성을 입증하였고, Fetch라는 모바일 로봇에서 인식 행동에 적용하였습니다. 이 방법은 다양한 상태 인식을 가능하게 하며, 여러 개의 모델과 프로그램을 준비할 필요 없이 소스 코드 및 컴퓨터 자원의 관리를 용이하게 해줍니다.



### Multi-Designated Detector Watermarking for Language Models (https://arxiv.org/abs/2409.17518)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)에 대한 다중 지정 탐지기 수조(MDDW; multi-designated detector watermarking) 기술을 제안합니다. 이 기술은 모델 제공자가 두 가지 주요 특성을 가진 수조를 생성할 수 있게 합니다: (i) 특정 여러 지정 탐지기만 워터마크를 식별할 수 있으며, (ii) 일반 사용자에게는 출력 품질이 눈에 띄게 저하되지 않습니다.

- **Technical Details**: MDDW의 보안 정의를 형식화하고, 다중 지정 검증기 서명(MDVS; multi-designated verifier signatures)을 사용하여 MDDW를 구축하기 위한 프레임워크를 제공합니다. MDDW는 모델 제공자, 지정 탐지기, 사용자라는 세 가지 역할을 포함하며, MDDW 스킴은 설정 알고리즘, 모델 제공자를 위한 키 생성 알고리즘, 지정 탐지기를 위한 키 생성 알고리즘, 워터마킹 알고리즘 및 탐지 알고리즘으로 구성됩니다.

- **Performance Highlights**: MDDW 스킴의 구현은 기존 방법보다 향상된 기능과 유연성을 강조하며, 만족스러운 성능 지표를 보입니다. 실험 평가 결과 MDDW의 적용 가능성을 보여줍니다.



### Dataset Distillation-based Hybrid Federated Learning on Non-IID Data (https://arxiv.org/abs/2409.17517)
- **What's New**: 이번 연구에서는 비독립적이고 동일 분포하지 않은(Non-IID) 데이터로 인한 라벨 분포 왜곡(label distribution skew) 문제를 해결하기 위해 HFLDD라는 새로운 하이브리드 연합 학습 프레임워크를 제안합니다.

- **Technical Details**: HFLDD는 클라이언트를 이질적인 클러스터로 분할하며, 각 클러스터 내의 데이터 라벨은 비균형하지만 클러스터 간에는 균형을 이룹니다. 클러스터 헤더는 해당 클러스터의 원거리 데이터를 수집하고 서버와 협업하여 모델 학습을 수행합니다. 이 과정은 기존의 IID 데이터에서 수행되는 전통적인 연합 학습과 유사하여 Non-IID 데이터의 영향을 효과적으로 감소시킵니다.

- **Performance Highlights**: 실험 결과에 따르면, 데이터 라벨이 심각하게 불균형할 때 HFLDD가 Baseline 방법들에 비해 테스트 정확도(test accuracy)와 통신 비용(communication cost) 모두에서 우수한 성능을 보였습니다.



### NeuroPath: A Neural Pathway Transformer for Joining the Dots of Human Connectomes (https://arxiv.org/abs/2409.17510)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 이 논문은 신경 이미지 데이터에서 구조적 연결성(Structural Connectivity, SC)과 기능적 연결성(Functional Connectivity, FC) 간의 결합 메커니즘에 대한 연구를 소개합니다. 특히 'NeuroPath'라는 새로운 생물학적 영감을 받은 딥 모델을 제안하여, SC와 FC의 쌍으로부터 복잡한 신경 구조의 특징 표현을 발견하고 인지 행동의 예측 및 질병 진단에 활용할 수 있습니다.

- **Technical Details**: NeuroPath 모델은 고차원 토폴로지의 표현 학습 문제로 구성된 SC-FC 결합 메커니즘을 다루며, 다중 헤드 자기 주의(multi-head self-attention) 메커니즘을 사용하여 SC와 FC의 쌍 그래프에서 다중 모달 특징 표현을 캡처합니다. 이를 통해 우리는 SC의 다양한 경로들이 FC를 지원하는 방식(예: cyclic loop)을 이해할 수 있게 됩니다.

- **Performance Highlights**: NeuroPath 지표는 HCP(인간 연결체 프로젝트) 및 UK Biobank와 같은 대규모 공개 데이터셋에서 검증되었으며, 기존의 최첨단 성능을 초과하여 인지 상태 예측과 질병 위험 진단에서 뛰어난 잠재력을 보였습니다.



### Uni-Med: A Unified Medical Generalist Foundation Model For Multi-Task Learning Via Connector-MoE (https://arxiv.org/abs/2409.17508)
- **What's New**: 이 논문은 다양한 시각 및 언어 작업을 위한 일반-purpose 인터페이스로서, 의료 분야의 multi-task learning을 위한 통합된 Multi-modal large language model (MLLM)을 제안합니다. 특히, 이전의 연구들이 처리하지 않았던 connector 문제를 해결하고자 합니다. 향후 Uni-Med는 의학 분야에서의 새로운 시도를 이루어낼 것입니다.

- **Technical Details**: Uni-Med는 범용 시각 feature extraction 모듈, connector mixture-of-experts (CMoE) 모듈, 그리고 LLM으로 구성된 의료 전문 기초 모델입니다. CMoE는 잘 설계된 라우터를 활용하여 projection experts의 혼합을 통해 connector에서 발생하는 문제를 해결합니다. 이 접근 방식을 통해 6개의 의료 관련 작업을 수행할 수 있습니다: 질문 응답(question answering), 시각 질문 응답(visual question answering), 보고서 생성(report generation), 지칭 표현 이해(referring expression comprehension), 지칭 표현 생성(referring expression generation) 및 이미지 분류(image classification).

- **Performance Highlights**: Uni-Med는 connector에서의 multi-task 간섭을 해결하려는 첫 번째 노력으로, 다양한 구성에서 CMoE를 도입하여 평균 8%의 성능 향상을 validates합니다. 이전의 최신 의료 MLLM과 비교할 때, Uni-Med는 다양한 작업에서 경쟁력 있거나 더 우수한 평가 지표를 달성하였습니다.



### Autoregressive Multi-trait Essay Scoring via Reinforcement Learning with Scoring-aware Multiple Rewards (https://arxiv.org/abs/2409.17472)
Comments:
          EMNLP 2024

- **What's New**: SaMRL(Scoring-aware Multi-reward Reinforcement Learning) 방법론을 제안하여 다중 특성 자동 에세이 평가(multi-trait automated essay scoring)에서 Quadratic Weighted Kappa(QWK) 기반 보상을 통합하고, 평균 제곱 오차(MSE) 페널티를 적용하여 실제 평가 방식을 훈련 과정에 포함시킴으로써 모델의 성능을 향상시킵니다.

- **Technical Details**: 기존의 QWK는 비미분 가능성으로 인해 신경망 학습에 직접 사용되지 못하나, SaMRL은 bi-directional QWK와 MSE 페널티를 통해 다중 특성 평가의 복잡한 측정 방식을 훈련에 유효하게 통합하고, 오토 회귀 점수 생성 프레임워크를 적용해 토큰 생성 확률로 강력한 다중 특성 점수를 예측합니다.

- **Performance Highlights**: ASAP 및 ASAP++ 데이터셋을 통한 광범위한 실험 결과, SaMRL은 기존 강력한 기준선에 비해 각 특성과 프롬프트에서 점수 향상을 이끌어내며, 특히 넓은 점수 범위를 가진 프롬프트에서 기존 RL 사용의 한계를 극복한 점이 두드러집니다.



### Adjusting Regression Models for Conditional Uncertainty Calibration (https://arxiv.org/abs/2409.17466)
Comments:
          Machine Learning Special Issue on Uncertainty Quantification

- **What's New**: 이 논문에서는 분할(conformal prediction) 일치 결과의 조건부 커버리지(conditional coverage)를 개선하기 위한 새로운 회귀(Regression) 알고리즘을 제안합니다. 기존의 방법들이 조건부 커버리지 보장을 제공하지 못하는 문제를 해결하고자 조건부 커버리지와 명목(marginal) 커버리지 사이의 미커버리지(miscoverage) 갭을 제어하는 세부 목표를 설정했습니다.

- **Technical Details**: 제안된 알고리즘은 기본적으로 분할(conformal prediction) 절차를 적용한 후의 회귀 함수 최적화를 통해 조건부 커버리지를 향상시키고, Kolmogorov-Smirnov 거리와의 연결고리를 구축합니다. 구체적으로는 미커버리지 갭에 대한 상한을 설정하고, 이를 제어하기 위한 끝에서 끝(end-to-end) 알고리즘을 제안합니다.

- **Performance Highlights**: 이 방법론은 합성 데이터(synthetic datasets) 및 실제 데이터(real-world datasets)에서 실험적으로 유효성을 입증했으며, 기존의 방법들과 비교했을 때 조건부 커버리지의 개선 효과를 확인할 수 있었습니다.



### CadVLM: Bridging Language and Vision in the Generation of Parametric CAD Sketches (https://arxiv.org/abs/2409.17457)
- **What's New**: 이 논문은 Parametric Computer-Aided Design (CAD) 분야에서 CAD 생성 작업을 위한 새로운 Vision Language 모델인 CadVLM을 제안합니다. 이는 기존의 CAD 모델링 방법의 한계를 극복하고, 스케치 이미지와 텍스트를 결합한 멀티모달 접근법을 적용하였습니다.

- **Technical Details**: CadVLM은 사전 학습된 모델을 활용하여 엔지니어링 스케치를 효과적으로 조작할 수 있는 엔드 투 엔드 모델입니다. 이 모델은 스케치 원시 시퀀스 및 스케치 이미지를 통합하여 CAD 자동완성(CAD autocompletion) 및 CAD 자동 제약(CAD autoconstraint)과 같은 다양한 CAD 스케치 생성 작업에서 뛰어난 성능을 발휘합니다.

- **Performance Highlights**: CadVLM은 SketchGraphs 데이터셋에서 CAD 자동완성 및 CAD 자동 제약 작업에서 우수한 성능을 보였으며, Entity Accuracy, Sketch Accuracy, CAD F1 score로 평가된 새로운 평가 지표를 소개하였습니다.



### HDFlow: Enhancing LLM Complex Problem-Solving with Hybrid Thinking and Dynamic Workflows (https://arxiv.org/abs/2409.17433)
Comments:
          27 pages, 5 figures

- **What's New**: 이 논문에서는 Large Language Models (LLMs)의 복잡한 추론 문제 해결을 위한 새로운 프레임워크인 HDFlow를 소개합니다. 이 프레임워크는 빠른 사고와 느린 사고를 적응적으로 결합하여, 복잡한 문제를 더 잘 해결할 수 있도록 합니다.

- **Technical Details**: HDFlow는 두 가지 주요 구성 요소로 이루어져 있습니다. 첫째, 'Dynamic Workflow'라는 새로운 느린 추론 접근법은 복잡한 문제를 관리 가능한 하위 작업으로 자동 분해합니다. 둘째, 'Hybrid Thinking'은 문제의 복잡성에 따라 빠른 사고와 느린 사고를 동적으로 결합하는 일반적인 프레임워크입니다. 이를 통해 모델은 문제의 복잡성에 따라 적절한 사고 모드를 선택합니다.

- **Performance Highlights**: 실험 결과, HDFlow의 느린 사고 방식 및 동적 워크플로우가 Chain-of-Thought (CoT) 전략보다 평균 22.4% 더 높은 정확도를 보였고, Hybrid Thinking은 네 개의 벤치마크 데이터 세트 중 세 개에서 최고의 정확도를 기록했습니다. 또한 하이브리드 사고를 통한 미세 조정 방법이 오픈 소스 언어 모델의 복잡한 추론 능력을 크게 향상시켰습니다.



### Discovering the Gems in Early Layers: Accelerating Long-Context LLMs with 1000x Input Token Reduction (https://arxiv.org/abs/2409.17422)
- **What's New**: 이 연구에서는 긴 컨텍스트 입력을 처리하기 위해 LLM(Large Language Model)의 추론을 가속화하고 GPU 메모리 소모를 줄이는 새로운 접근 방식인 GemFilter를 소개합니다.

- **Technical Details**: GemFilter는 LLM의 초기 레이어를 필터로 사용하여 입력 토큰을 선택하고 압축하는 알고리즘으로, 이에 따라 추후 처리할 컨텍스트 길이를 대폭 줄입니다. 이 방법은 2.4$	imes$ 속도 향상과 30
gpu(%) GPU 메모리 사용량 감소를 달성하였습니다.

- **Performance Highlights**: GemFilter는 Needle in a Haystack 벤치마크에서 기존의 표준 어텐션 및 SnapKV를 능가하며, LongBench 챌린지에서는 SnapKV/H2O와 유사한 성능을 보여줍니다.



### Solar Active Regions Emergence Prediction Using Long Short-Term Memory Networks (https://arxiv.org/abs/2409.17421)
Comments:
          20 pages, 8 figures, 5 tables, under review at the AAS Astrophysical Journal

- **What's New**: 이 연구에서는 Long Short-Term Memory (LSTM) 모델을 개발하여 태양 표면에서 활성 영역(Active Regions, AR)의 형성을 예측합니다. 이 모델은 Solar Dynamics Observatory (SDO) Helioseismic and Magnetic Imager (HMI)로부터 얻은 Doppler shift velocity, continuum intensity, magnetic field observations를 활용하여 시간 시계열 데이터셋을 생성했습니다.

- **Technical Details**: LSTM 모델은 acústic power 및 magnetic flux의 변화를 포착하는 기능이 있으며, 이 연구에서는 61개의 신생 AR에 대한 데이터를 수집하여 12시간 전 continuum intensity를 예측하기 위해 훈련되었습니다. 연구에 사용된 데이터는 활성 영역의 출현 전후의 상태를 포함합니다.

- **Performance Highlights**: 모델 8은 실험 설정에서 모든 테스트 활성 영역의 출현 예측에 성공하였고, AR11726, AR13165 및 AR13179의 경우 각각 10, 29, 5시간 전에 예측했습니다. 이 모델의 RMSE(평균 제곱근 오차) 값은 태양 디스크의 활성 및 조용한 지역 모두에 대해 평균 0.11로, ML 기반의 태양 AR 예측의 기초를 마련했습니다.



### From Deception to Detection: The Dual Roles of Large Language Models in Fake News (https://arxiv.org/abs/2409.17416)
- **What's New**: 최근 연구에서는 다양한 Large Language Models (LLMs)가 가짜 뉴스를 생성할 수 있는 능력과 이러한 모델들이 가짜 뉴스를 감지하는 성능을 비교했습니다. 이는 7개의 LLM을 분석한 최초의 연구로, 각 모델의 편향 및 안전성 준수 여부를 평가하였습니다.

- **Technical Details**: 연구는 LLM의 가짜 뉴스 생성과 감지 등 두 가지 주요 단계를 중점적으로 다루고 있습니다. LLM들은 다양한 편향을 담은 가짜 뉴스를 생성할 수 있으며, 종종 인간이 작성한 내용보다 탐지하기 어렵습니다. 또한, LLM에서 제공하는 설명의 유용성도 평가되었습니다.

- **Performance Highlights**: 결과적으로, 크기가 큰 LLM들이 더 나은 가짜 뉴스 탐지 능력을 보였으며, 일부 모델은 안전 프로토콜을 엄격히 준수하여 편향된 내용을 생성하지 않았습니다. 반면에, 다른 모델은 여러 편향을 포함한 가짜 뉴스를 쉽게 생성할 수 있었고, LLM이 생성한 가짜 뉴스는 일반적으로 인간이 작성한 것보다 탐지될 가능성이 낮다는 사실이 밝혀졌습니다.



### Sociotechnical Approach to Enterprise Generative Artificial Intelligence (E-GenAI) (https://arxiv.org/abs/2409.17408)
- **What's New**: 이 논문에서는 비즈니스 생태계를 사회기술적(sociotechnical) 접근으로 특성화하는 새로운 방법론을 제시합니다. 특히 SCM, ERP, CRM 플랫폼을 통해 제공자(Providers), 기업(Enterprise), 고객(Customers) 간의 관계에 중점을 두고 있습니다.

- **Technical Details**: OID 모델을 통해 비즈니스 인텔리전스(Business Intelligence, BI), 퍼지 로직(Fuzzy Logic, FL), 발명 문제 해결 이론(TRIZ)을 통합하고, OIDK 모델을 통해 지식 관리(Knowledge Management, KM) 및 불완전 지식 관리(Imperfect Knowledge Management, IKM)를 조율합니다. 또한 E-GenAI 비즈니스 생태계는 SCM, ERP, CRM의 GenAI 기반 플랫폼과 BI, FL, TRIZ, KM, IKM의 GenAI 기반 플랫폼을 통합하여 대형 언어 모델(Large Language Models, LLMs)을 정렬합니다.

- **Performance Highlights**: 마지막으로, LLM의 역학을 이해하기 위해 유한 자동자(finite automata)를 활용하여 팔로워(Followers)와 팔로우이(Followees) 간의 관계를 모델링합니다. 이를 통해 소셜 미디어 플랫폼에서 사용자 특성을 식별할 수 있는 LLM 구축을 촉진합니다.



### Transient Adversarial 3D Projection Attacks on Object Detection in Autonomous Driving (https://arxiv.org/abs/2409.17403)
Comments:
          20 pages, 7 figures, SmartSP 2024

- **What's New**: 이 논문에서는 자율 주행 시나리오에서 물체 탐지를 타겟으로 한 새로운 적대적 3D 프로젝션 공격을 제안합니다. 기존의 고정된 적대적 패턴과 달리, 이 새로운 유형의 공격은 3D 표면에서의 일시적 수정과 같은 유연성을 제공합니다.

- **Technical Details**: 이 공격은 최적화 문제로 구성되며, 색상 매핑(color mapping) 및 기하변환 모델(geometric transformation models)을 결합하여 설계되었습니다. 특히, Thin Plate Spine (TPS) 알고리즘을 사용하여 2D 이미지를 3D 표면에 효과적으로 변형합니다.

- **Performance Highlights**: 실험 결과, YOLOv3 및 Mask R-CNN을 기준으로 한 공격의 성공률이 낮은 조도 조건에서 최대 100%에 달하는 것으로 나타났습니다. 이는 실제 자율주행 상황에서의 치명적인 결과를 초래할 수 있는 공격의 효과를 강조합니다.



### Enhancing Recommendation with Denoising Auxiliary Task (https://arxiv.org/abs/2409.17402)
- **What's New**: 본 연구는 사용자의 이력(interaction sequences)에서 발생하는 잡음(noise)이 추천 시스템에 미치는 영향을 다루고 있으며, 이를 개선하기 위한 새로운 방법인 자가 감독(Auto-supervised) 보조 작업 결합 훈련(Auxiliary Task Joint Training, ATJT)을 제안합니다.

- **Technical Details**: ATJT 방법은 원본 이력을 바탕으로 랜덤 대체를 통해 인위적으로 생성된 잡음이 포함된 시퀀스를 학습하여 모델의 성능을 향상시키기 위한 것입니다. 잡음 인식 모델과 추천 모델을 난이도에 따라 조정된 가중치로 훈련하여 잡음이 포함된 시퀀스로부터 적절한 학습을 이끌어냅니다.

- **Performance Highlights**: ATJT 방법은 세 개의 데이터셋에서 일관된 기본 모델을 사용하여 실험한 결과, 모델의 추천 성능을 개선하는 데 효과적임을 입증했습니다.



### AgRegNet: A Deep Regression Network for Flower and Fruit Density Estimation, Localization, and Counting in Orchards (https://arxiv.org/abs/2409.17400)
- **What's New**: 이 논문에서는 농업 산업의 노동력과 비용 문제를 해결하기 위한 자동화된 꽃과 과일 밀도 추정 기술을 제안합니다. 특히, AgRegNet이라는 딥 회귀 기반 네트워크를 사용하여, 탐지(Object Detection)나 다각형 주석(Polygon Annotation) 없이 꽃과 과일의 밀도, 개수, 위치를 추정합니다.

- **Technical Details**: AgRegNet은 U-Net 아키텍처에서 영감을 받아 개발된 U자형 네트워크로, 인코더-디코더 스킵 커넥션을 포함하고 있습니다. ConvNeXt-T를 수정한 구조로 특징을 추출하며, 포인트 주석(Point Annotation) 정보를 바탕으로 학습되고, 세그멘테이션 정보 및 주의 모듈(Attention Modules)을 활용하여 relevant한 꽃과 과일의 특징을 강조합니다.

- **Performance Highlights**: 실험 결과, AgRegNet은 구조적 유사도 지수(Structural Similarity Index, SSIM), 백분율 평균 절대 오차(percentage Mean Absolute Error, pMAE), 평균 평균 정밀도(mean Average Precision, mAP) 측면에서 높은 정확도를 달성했습니다. 특히, 꽃 이미지의 SSIM은 0.938, pMAE는 13.7%, mAP는 0.81이며, 과일 이미지의 경우 SSIM은 0.910, pMAE는 5.6%, mAP는 0.93으로 나타났습니다.



### Beyond Redundancy: Information-aware Unsupervised Multiplex Graph Structure Learning (https://arxiv.org/abs/2409.17386)
Comments:
          Appear in NeurIPS 2024

- **What's New**: 이 논문에서는 여러 가지 간선 유형에 대한 노드 표현을 수동 라벨링 없이 학습하는 비지도 다중 그래프 학습(UMGL)에 초점을 두고 있습니다. 특히, 그래프 구조의 신뢰성을 고려하지 않았던 기존 연구의 한계를 지적하며, 다양한 그래프에서 노이즈를 제거하고 과제 관련 정보를 보존하는 새로운 비지도 학습 방법을 제안합니다.

- **Technical Details**: 제안된 프레임워크인 정보 인식 비지도 다중 그래프 융합(InfoMGF)은 그래프 구조 정제를 활용하여 불필요한 노이즈를 제거하고, 동시에 공유된 과제 관련 정보와 고유한 과제 관련 정보를 최대화합니다. 이 프레임워크는 비지도 학습 방식으로 다중 그래프에서 융합 그래프를 학습합니다.

- **Performance Highlights**: 다양한 다운스트림 작업에 대해 여러 기준선과 비교하여 InfoMGF의 우수한 성능과 강건성을 입증합니다. 특히, 비지도 방법임에도 불구하고 기존의 정교한 감독 방법보다 더 나은 성능을 보였습니다.



### Data-efficient Trajectory Prediction via Coreset Selection (https://arxiv.org/abs/2409.17385)
- **What's New**: 이 논문에서는 복잡한 주행 시나리오(시나리오)에서 데이터 부족 문제와 과대표현된 주행 시나리오로 인한 데이터 중복 문제를 완화하기 위해 새로운 데이터 효율적인 훈련 방법인 'coreset selection'을 제안합니다. 이 방법은 다양한 난이도의 시나리오 간 비율을 조절하면서 중요 데이터를 선택하여, 훈련 성능을 유지하면서 데이터 용량을 줄입니다.

- **Technical Details**: 이 방법은 데이터셋의 난이도 수준에 따라 데이터를 그룹화하고, 각 샘플에 대한 하위 모듈러 이득(submodular gain)을 계산하여 가장 가치 있는 데이터를 선택합니다. 두 가지 선택 방법인 'Fixed Selection'과 'Balanced Selection'을 통해 데이터 분포를 조절하며, 특히 Balnced Selection 방법은 복잡한 시나리오에서 유리한 성과를 보여줍니다. 또한, coresets는 일반화 능력이 뛰어나고 다양한 모델에 대해 테스트되었습니다.

- **Performance Highlights**: Fixed Selection 방법을 이용한 coresets는 전체 데이터셋의 50%만으로도 성능 저하 없이 비슷한 결과를 보여주었으며, Balanced Selection 방법은 더 복잡한 주행 시나리오에서 현저한 성과를 기록했습니다. 선택된 coresets는 SOTA 모델인 HPNet에서도 유사한 성능을 발휘하여, 모델의 다양한 난이도 시나리오에서의 일반화 능력을 강화했습니다.



### VectorSearch: Enhancing Document Retrieval with Semantic Embeddings and Optimized Search (https://arxiv.org/abs/2409.17383)
Comments:
          10 pages, 14 figures

- **What's New**: 이 논문은 고차원 데이터 검색의 정확성을 높이기 위해 'VectorSearch'라는 새로운 알고리즘을 제안합니다. 이 시스템은 고급 언어 모델과 다중 벡터 인덱싱 기술을 통합하여 텍스트 데이터의 의미적 관계를 더 잘 파악할 수 있도록 설계되었습니다.

- **Technical Details**: VectorSearch는 데이터의 다차원 임베딩(embeddings)을 효율적으로 검색하는 하이브리드(document retrieval framework) 시스템입니다. HNSWlib 및 FAISS와 같은 최적화 기법을 사용하여 대규모 데이터셋을 효과적으로 관리하고, 복잡한 쿼리 처리 기능을 통해 고급 검색 작업을 지원합니다. 또한, 시스템은 클러스터 환경에서 동적으로 변화하는 데이터셋을 처리할 수 있는 메커니즘을 포함하고 있습니다.

- **Performance Highlights**: 실제 데이터셋을 기반으로 한 실험 결과, VectorSearch는 기존의 기준 메트릭을 초월하여 대규모 검색 작업에서 뛰어난 성능을 보였습니다. 이는 높은 차원의 데이터에서도 저지연성 검색 결과를 제공함으로써 정보 검색의 정확성을 획기적으로 향상시킨 것을 나타냅니다.



### Tesla's Autopilot: Ethics and Tragedy (https://arxiv.org/abs/2409.17380)
- **What's New**: 이번 사례 연구는 Tesla의 Autopilot과 관련된 사건에서 윤리적 결과를 다루고 있습니다. Tesla Motors의 도덕적 책임과 함께 자율 주행 기술의 윤리적 도전에 대한 넓은 평가를 강조합니다.

- **Technical Details**: 사례 연구는 7단계 윤리적 의사결정 프로세스를 사용하여 Tesla Motors의 도덕적 책임을 분석합니다. 사용자 행동, 시스템 제한, 규제적 함의가 포함되며 자율 주행 기술에 대한 윤리적 고려를 일반적으로 탐구합니다.

- **Performance Highlights**: 사고 분석을 통해 자율 주행 기술의 도입 과정에서 발생할 수 있는 도덕적 딜레마를 강조하며, Tesla가 도로에서 인간과 기계가 공유하는 미래에 맞춰 도덕적 원칙과 법적 시스템을 정렬하는 것이 중요하다고 강조합니다.



### The Overfocusing Bias of Convolutional Neural Networks: A Saliency-Guided Regularization Approach (https://arxiv.org/abs/2409.17370)
- **What's New**: 본 논문에서는 Neural Networks(신경망)가 훈련 데이터가 제한적일 때 특정 이미지 영역에 초점을 맞추는 경향을 설명하고, 이러한 경향을 개선하기 위한 새로운 정규화 방법인 Saliency Guided Dropout(SGDrop)를 제안합니다.

- **Technical Details**: SGDrop은 attribution methods(어트리뷰션 방법)을 사용하여 훈련 중 가장 중요한 특징들의 영향을 줄이고, 신경망이 입력 이미지의 다양한 영역에 주의를 분산시키도록 유도합니다. SGDrop의 구현은 훈련 과정에서 중요한 특징을 선택적으로 삭제하는 방식으로 진행됩니다.

- **Performance Highlights**: 여러 비주얼 분류 벤치마크에서 SGDrop을 적용한 모델은 더 넓은 어트리뷰션과 신경 활동을 보여주었으며, 이는 입력 이미지의 전체적인 관점을 반영합니다. 또한, SGDrop을 통해 일반화 성능이 향상되는 것을 확인할 수 있었습니다.



### Koopman-driven grip force prediction through EMG sensing (https://arxiv.org/abs/2409.17340)
Comments:
          11 pages, 8 figures, journal

- **What's New**: 본 연구는 단일 sEMG 센서 쌍을 이용하여 중간 잡기(grasping)에서의 힘 추정치를 정확하게 도출하는 방법론을 개발하였습니다. 기존 연구들이 정확한 예측을 위해 많은 수의 센서를 요구하는 데 비해, 우리가 제안하는 방법론은 최소한의 센서를 사용하여도 높은 신뢰도를 달성할 수 있도록 설계되었습니다.

- **Technical Details**: 연구 결과, 공차(variance) 없는 sEMG 신호와 그립 힘(grip force) 사이에 높은 피크 상관관계를 달성했으며, 데이터 기반의 Koopman operator를 활용하여 실시간 움켜잡기 힘의 추정 및 단기 예측을 진행하였습니다. 또한, 약 30 ms 내에 0.5초 sEMG 신호 배치를 처리하고 예측하는 고속 알고리즘이 고안되었습니다.

- **Performance Highlights**: 추정된 그립 힘의 wMAPE(weighted Mean Absolute Percentage Error)는 약 5.5%였으며, 0.5초 예측 시에도 wMAPE가 약 17.9%에 달했습니다. 전극 위치에 대한 민감도 분석을 통해 정확한 배치에 대한 요구 조건이 비현저하다는 결과를 도출하였고, 이 연구는 전통적인 방법론에 비해 실시간 적용 가능성을 증가시켰습니다.



### The Technology of Outrage: Bias in Artificial Intelligenc (https://arxiv.org/abs/2409.17336)
Comments:
          Distribution Statement A. Approved for public release; distribution is unlimited

- **What's New**: 본 연구는 알고리즘이 사람을 대체할 수 있다는 통념과 알고리즘이 편향될 수 없다는 주장에 대해 심층적으로 논의하고, 알고리즘 편향에 대한 감정적인 반응의 세 가지 형태를 진단합니다.

- **Technical Details**: 편향(bias)이라는 용어의 모호함을 해소하고, 지능적 시스템에 대한 새로운 감사(audit) 방법을 개발하며, 이러한 시스템에 특정 기능을 구축하는 등 AI 커뮤니티가 취할 수 있는 세 가지 실용적인 접근법을 제안합니다.

- **Performance Highlights**: AI가 인간의 행동을 모방하며, 이러한 시스템이 인간의 편견을 반영함에 따라 public의 우려가 증가하고 있습니다. 이 연구는 복잡한 수학적 모델을 통한 알고리즘 편향 문제 해결에 기여할 방법을 제시합니다.



### Block Expanded DINORET: Adapting Natural Domain Foundation Models for Retinal Imaging Without Catastrophic Forgetting (https://arxiv.org/abs/2409.17332)
Comments:
this http URL, C. Merk and M. Buob contributed equally as shared-first authors. D. Cabrera DeBuc, M. D. Becker and G. M. Somfai contributed equally as senior authors for this work

- **What's New**: 이 연구에서는 자기 지도 학습(self-supervised learning)과 파라미터 효율적인 미세 조정(parameter-efficient fine-tuning)을 이용하여 망막 이미징(retinal imaging) 작업을 위한 두 가지 새로운 기초 모델인 DINORET와 BE DINORET을 개발하였습니다.

- **Technical Details**: DINOv2 비전 트랜스포머(vision transformer)를 적응하여 망막 이미징 분류 작업에 활용하였으며, 두 모델 모두 공개된 색상 안저 사진(color fundus photographs)을 사용하여 개발 및 미세 조정을 진행하였습니다. 우리는 블록 확장(block expansion)이라는 새로운 도메인 적응(domain adaptation) 전략을 도입하였습니다.

- **Performance Highlights**: DINORET과 BE DINORET은 망막 이미징 작업에서 경쟁력 있는 성능을 보여주었고, 블록 확장 모델이 대부분의 데이터 세트에서 최고 점수를 기록했습니다. 특히 DINORET과 BE DINORET은 데이터 효율성 측면에서 RETFound을 초과하며, 블록 확장이 재학습 시의 재난적 망각(catastrophic forgetting)을 성공적으로 완화했다는 점이 주목할 만합니다.



### KIPPS: Knowledge infusion in Privacy Preserving Synthetic Data Generation (https://arxiv.org/abs/2409.17315)
- **What's New**: 이 논문은 KIPPS라는 새로운 프레임워크를 제안하여, Generative Deep Learning 모델에 Domain 및 Regulatory Knowledge를 포함시킴으로써 Privacy Preserving Synthetic 데이터 생성을 개선합니다.

- **Technical Details**: KIPPS는 Generative 모델의 학습 과정에서 속성 값에 대한 추가 맥락과 도메인 제약을 강화하는 방법으로, 데이터를 합성하는 모델의 수용성을 높입니다. 이는 주로 Cybersecurity와 Healthcare와 같은 전문화된 도메인에서 사용됩니다.

- **Performance Highlights**: KIPPS 모델은 실제 데이터셋을 사용하여 프라이버시 보호와 데이터 정확도 간의 균형을 유지하는 효과를 보여줍니다. 모델은 최신 프라이버시 공격에 대해 회복력이 뛰어나며, 다운스트림 작업에서 원본 데이터와 유사한 정보를 유지합니다.



### Navigating the Nuances: A Fine-grained Evaluation of Vision-Language Navigation (https://arxiv.org/abs/2409.17313)
Comments:
          EMNLP 2024 Findings; project page: this https URL

- **What's New**: 이 연구는 Vision-Language Navigation (VLN) 작업을 위한 새로운 평가 프레임워크를 제안합니다. 이 프레임워크는 다양한 지침 범주에 대한 현재 모델을 더 세분화된 수준에서 진단하는 것을 목표로 합니다. 특히, context-free grammar (CFG)를 기반으로 한 구조에서 VLN 작업의 지침 카테고리를 설계하고, 이를 Large-Language Models (LLMs)의 도움으로 반자동으로 구성합니다.

- **Technical Details**: 제안된 평가 프레임워크는 atom instruction, 즉 VLN 지침의 기본 행동에 집중합니다. CFG를 사용하여 지침의 구조를 체계적으로 구성하고 5개의 주요 지침 범주(방향 변화, 수직 이동, 랜드마크 인식, 영역 인식, 숫자 이해)를 정의합니다. 이 데이터를 활용하여 평가 데이터셋 NavNuances를 생성하고, 이를 통해 다양한 모델의 성능을 평가하며 문제가 드러나는 경우가 많았습니다.

- **Performance Highlights**: 실험 결과, 모델 간 성능 차이와 일반적인 문제점이 드러났습니다. LLM에 의해 강화된 제로샷 제어 에이전트가 전통적인 감독 학습 모델보다 방향 변화와 랜드마크 인식에서 더 높은 성능을 보였으며, 반면 기존 감독 접근 방식은 선택적 편향으로 인해 원자 개념 변화에 적응하는 데 어려움을 겪었습니다. 이러한 분석은 VLN 방식의 향후 발전에 중요한 통찰력을 제공합니다.



### Neural Network Plasticity and Loss Sharpness (https://arxiv.org/abs/2409.17300)
- **What's New**: 최신 연구는 비정상적(non-stationary) 환경에서의 연속 학습(Continual Learning)에서 플라스틱성 손실(plasticity loss)과 손실 경량(sharpness) 간의 관계를 조사하였다. 이 논문에서는 이러한 손실을 줄이기 위한 샤프니스 정규화(sharpness regularization) 기법의 사용을 제안한다.

- **Technical Details**: 연속 학습 모델은 시간이 지남에 따라 변화하는 데이터 흐름을 학습하는 모델로, 신경망(neural network)의 예측이 다른 작업에 적응할 수 있는 능력을 요구한다. 본 연구는 손실의 헤세 행렬(Hessian matrix) 최대 고유값(maximal eigenvalue)을 통해 손실 경량을 정량화하며, 이는 네트워크가 처리하는 작업 수에 따라 증가한다는 기존 연구를 바탕으로 한다. 샤프니스 정규화 기법이 사용되며, 이는 네트워크가 더 '평평한(flatter)' 최소값을 찾도록 유도한다.

- **Performance Highlights**: 실험 결과, 샤프니스 정규화 기법은 플라스틱성 손실 감소에 유의미한 영향을 미치지 않는 것으로 나타났다. 이는 이러한 기술이 연속 학습 설정에서의 성능 유지에 있어 한계가 있음을 시사한다.



### SpoofCeleb: Speech Deepfake Detection and SASV In The Wild (https://arxiv.org/abs/2409.17285)
Comments:
          9 pages, 2 figures, 8 tables

- **What's New**: SpoofCeleb 데이터셋은 Speech Deepfake Detection (SDD) 및 Spoofing-robust Automatic Speaker Verification (SASV) 연구를 위해 설계되었으며, 1,251명의 독특한 화자들로부터 250만 개가 넘는 발화 데이터를 포함합니다.

- **Technical Details**: 이 데이터셋은 VoxCeleb1에서 자동 생성된 텍스트 음성 변환(Text-To-Speech, TTS) 시스템을 기반으로 하여 자연스러운 환경에서 수집된 발화로 구성되어 있습니다. 품질이 우수한 합성 음성의 악성 사용을 방지하기 위한 연구를 지원합니다.

- **Performance Highlights**: 정확한 평가 프로토콜이 포함된 학습, 검증 및 평가 세트로 잘 구분되어 있으며, SDD 및 SASV 작업에 대한 기준 성능을 제시하여 연구의 효율성을 높입니다.



### Memory Networks: Towards Fully Biologically Plausible Learning (https://arxiv.org/abs/2409.17282)
Comments:
          2024

- **What's New**: 이번 연구에서 제안하는 Memory Network(메모리 네트워크)는 생물학적 원리에 영감을 받아, 기존 딥러닝 모델에서 사용하는 backpropagation(역전파) 및 convolutions(합성곱)을 회피하고, 단일 패스(single pass)로 작동하는 새로운 접근 방식을 제공합니다. 이로 인해 빠르고 효율적인 학습이 가능해지며, 데이터에 대한 최소한의 노출로도 빠르게 적응할 수 있는 뇌의 능력을 모방합니다.

- **Technical Details**: Memory Network는 입력된 데이터를 인코딩하여 각 뉴런이 새 입력과 같은 레이블을 가진 이전 입력의 평균 표현 사이의 유사성에 기반하여 업데이트되는 방식으로 작동합니다. 이는 전통적인 큰 규모의 CNN(합성곱 신경망)과는 달리, 지역적 플라스틱성(local plasticity) 메커니즘을 활용하여 학습하며, 이는 생물학적 과정과 더욱 밀접하게 일치합니다.

- **Performance Highlights**: 실험 결과, Memory Network는 MNIST와 같은 간단한 데이터셋에서 효율적이고 생물학적으로 그럴듯한 학습을 달성하며 강력한 성능을 보였습니다. 그러나 CIFAR10과 같은 더 복잡한 데이터셋에서는 추가적인 개선이 필요함을 나타내어, 생물학적 과정에 근접하면서도 계산 효율성을 유지할 수 있는 새로운 알고리즘과 기법 개발의 필요성을 강조했습니다.



### On the Vulnerability of Applying Retrieval-Augmented Generation within Knowledge-Intensive Application Domains (https://arxiv.org/abs/2409.17275)
- **What's New**: 이 논문은 Retrieval-Augmented Generation (RAG) 시스템의 적대적 강인성(adversarial robustness)을 조사하였습니다. 특히, 의학 분야의 Q&A를 대상으로 한 '보편적인 중독 공격(universal poisoning attacks)'의 취약성을 분석하고 새로운 탐지 기반(defense) 방어 체계를 개발하였습니다.

- **Technical Details**: RAG 시스템은 외부 데이터에서 중요한 정보를 검색(retrieve)하고 이를 LLM의 생성 과정에 통합하는 두 가지 단계를 포함합니다. 이 연구에서는 225개 다양한 설정에서 RAG의 검색 시스템이 개인 식별 정보(PII)와 같은 다양한 타겟 정보를 포함하는 중독 문서에 취약하다는 것을 입증하였습니다. 중독된 문서는 쿼리의 임베딩과 높은 유사성을 유지함으로써 정확히 검색될 수 있음을 발견하였습니다.

- **Performance Highlights**: 제안한 방어 방법은 다양한 Q&A 도메인에서 뛰어난 탐지율(detection rates)을 일관되게 달성하며, 기존의 방어 방법에 비해 훨씬 효과적임을 보여주었습니다. 실험에 따르면 거의 모든 공격에 대해 일관되게 높은 탐지 성공률을 보였습니다.



### Model aggregation: minimizing empirical variance outperforms minimizing empirical error (https://arxiv.org/abs/2409.17267)
Comments:
          The code in this paper is available for download at this https URL

- **What's New**: 이 논문은 다양한 모델의 예측을 하나의 더 정확한 출력으로 집계하는 데이터 기반 프레임워크를 제안합니다. 이 집계 접근 방식은 각 모델의 강점을 활용하여 전체 정확도를 높이며, 비침해적이고 모델 불가지론적(model-agnostic)입니다.

- **Technical Details**: 제안된 집계 방법에는 최소 오류 집계(Minimal Error Aggregation, MEA)와 최소 분산 집계(Minimal Variance Aggregation, MVA)가 포함됩니다. MEA는 집계의 예측 오류를 최소화하는 반면, MVA는 분산을 최소화합니다. MEVA(Minimal Empirical Variance Aggregation)는 모델 오류를 추정하여 집계를 구성하며, MEEA(Minimal Empirical Error Aggregation)와 비교하여 데이터를 기준으로 한 추정에서 일관성 있게 더 우수한 성능을 발휘합니다.

- **Performance Highlights**: 제안된 MEVA 기법은 데이터 과학 작업 및 오퍼레이터 학습 과제에서 검증되었으며, 모든 사례에서 직접 오류 최소화 방법보다 우수한 성능을 보였습니다. 이는 MEVA가 개별 모델보다 더 강력하고 효과적인 집계 모델을 제공함을 시사합니다.



### Disk2Planet: A Robust and Automated Machine Learning Tool for Parameter Inference in Disk-Planet Systems (https://arxiv.org/abs/2409.17228)
Comments:
          Accepted to ApJ

- **What's New**: Disk2Planet라는 기계 학습 기반 도구를 소개합니다. 이 도구는 관측된 원시 행성계(disk-planet) 구조에서 핵심 매개변수를 추론합니다.

- **Technical Details**: Disk2Planet은 2D 밀도(density) 및 속도(velocity) 맵의 형태로 입력된 원반 구조를 바탕으로, Shakura–Sunyaev 점성(viscosity), 원반 비율(aspect ratio), 행성-별 질량 비율(planet-star mass ratio), 행성의 반지름(radius) 및 방위각(azimuth) 같은 매개변수를 출력합니다. 이 도구는 CMA-ES라는 복잡한 최적화 문제를 위한 진화 알고리즘과, 원반-행성 상호작용의 예측을 위해 설계된 PPDONet이라는 신경망(neural network)을 통합했습니다. 전체 자동화된 시스템으로, Nvidia A100 GPU에서 3분 이내에 하나의 시스템의 매개변수를 검색할 수 있습니다.

- **Performance Highlights**: Disk2Planet은 0.001에서 0.01 정도의 정확도로 매개변수를 추론할 수 있으며, 결측치(missing data)와 다양한 노이즈(noise) 수준을 처리할 수 있는 강력성을 입증했습니다.



### Data-Centric AI Governance: Addressing the Limitations of Model-Focused Policies (https://arxiv.org/abs/2409.17216)
- **What's New**: 현재의 강력한 AI 능력에 대한 규제는 "기초" (foundation) 또는 "최전선" (frontier) 모델에 너무 좁게 초점을 맞추고 있으며, 이러한 용어의 모호성과 불일치로 인해 거버넌스 노력의 기초가 불안정하다는 점을 강조합니다.

- **Technical Details**: 이 논문은 데이터셋의 크기와 내용이 모델의 성능에 미치는 영향을 평가하는 데 필수적인 요소라는 것을 보여주며, 상대적으로 "작은" 모델조차도 충분히 특정한 데이터셋에 노출될 경우 동등한 결과를 달성할 수 있음을 설명합니다. 논의는 데이터 사용의 중요성을 간과하는 동안 정책 논쟁이 발생하는 웃음을 다룹니다.

- **Performance Highlights**: 과도한 사후 규제의 위험성을 강조하고, 능력을 신중하게 정량적으로 평가할 수 있는 경로를 제시하여 규제 환경을 단순화할 수 있는 가능성을 보여줍니다.



### Plurals: A System for Guiding LLMs Via Simulated Social Ensembles (https://arxiv.org/abs/2409.17213)
- **What's New**: 최근 논의에서 언어 모델들이 특정 관점을 선호한다는 우려가 제기되었습니다. 이에 대한 해결책으로 '어디서도 바라보지 않는 시각'을 추구하는 것이 아니라 다양한 관점을 활용하는 방안을 제안합니다. 'Plurals'라는 시스템과 Python 라이브러리를 소개하게 되었습니다.

- **Technical Details**: Plurals는 다양한 관점을 반영하여 신뢰할 수 있는 소셜 에셈블을 생성하는 시스템으로, 사용자 맞춤형 구조 내에서 '대리인'(Agents)들이 심의(deliberation)를 진행하고, 보고자'(Moderators)가 이를 감독합니다. 이 시스템은 미국 정부의 데이터셋과 통합되어 국가적으로 대표적인 페르소나를 생성하면서 민주적 심의 이론에 영감을 받은 형식으로 사용자 정의가 가능합니다.

- **Performance Highlights**: 여섯 가지 사례 연구를 통해 이론적 일관성과 효용성(efficacy)을 보여주었고, 세 가지 무작위 실험에서는 생성된 산출물이 관련 청중의 온라인 샘플과 잘 맞아떨어짐을 발견했습니다. Plurals는 사용자가 시뮬레이션된 사회적 집단을 생성할 수 있도록 도와주며, 초기 연구 결과 또한 제시됩니다.



### 2024 BRAVO Challenge Track 1 1st Place Report: Evaluating Robustness of Vision Foundation Models for Semantic Segmentation (https://arxiv.org/abs/2409.17208)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2409.15107

- **What's New**: 2024 BRAVO Challenge의 Track 1에서 Cityscapes 데이터셋으로 훈련된 모델을 제시하며, 다양한 out-of-distribution 데이터셋에서의 강건성을 평가합니다. 이 연구는 Vision Foundation Models (VFM)을 활용하여 DINOv2에 간단한 segmentation decoder를 붙여 전체 모델을 fine-tuning하여 우수한 성능을 보여줍니다.

- **Technical Details**: 이 연구에서는 DINOv2 VFM을 사용하여 semantic segmentation을 위한 pre-trained 모델을 fine-tuning합니다. 기본 구성에서 간단한 linear decoder를 사용하여 patch-level features를 segmentation logits로 변환합니다. 다양한 모델 크기, patch 크기, pre-training 전략 및 segmentation decoders를 실험하여 우리의 접근 방식의 효과를 평가합니다.

- **Performance Highlights**: 우리는 기존의 복잡한 모델들을 능가하여 이 챌린지에서 1위를 달성했습니다. 우리의 간단한 접근 방식이 어떻게 specialist 모델들보다 더 나은 성능을 내는지를 입증하며, 향후 연구에서 관심을 끌 수 있는 새로운 관찰도 제시합니다.



### Enhancing Guardrails for Safe and Secure Healthcare AI (https://arxiv.org/abs/2409.17190)
- **What's New**: 이번 논문에서는 헬스케어 AI에 내재된 독특한 안전 및 보안 문제를 조사하고, 의료 분야에 적합한 안전망 개선 방안을 제안합니다. 특히, 비임상 환경에서 발생할 수 있는 환각(hallucination), 잘못된 정보(misinformation)의 확산, 사실 정확성의 필요성을 강조합니다.

- **Technical Details**: 이 연구는 NVIDIA NeMo Guardrails와 Llama Guard의 통합 방법을 제안합니다. NeMo Guardrails는 사용자의 프롬프트를 벡터 표현으로 변환하고, Llama Guard는 악의적인 우회(jailbreaking)를 방지하여 의료 AI 시스템의 무결성을 보장합니다. 두 프레임워크를 통합하여 의료 AI 시스템의 리스크를 줄이기 위한 방법론을 제시합니다.

- **Performance Highlights**: 이 통합 접근 방식은 여러 의료 데이터 세트를 통해 평가되었으며, 정확하고 신뢰성 있는 헬스케어 AI의 사용을 보장하여 환자 안전을 향상시키는 데 기여할 것이 기대됩니다. 이 연구는 의료 AI의 널리 보급되는(application) 가능성을 높이는 데 중점을 두고 있습니다.



### Transfer learning for financial data predictions: a systematic review (https://arxiv.org/abs/2409.17183)
Comments:
          43 pages, 5 tables, 1 figure

- **What's New**: 이 논문은 전이 학습(Transfer Learning) 방법론을 금융 시장 예측에 적용하는 데 중점을 두고 있으며, 기존에는 신경망(neural network) 아키텍처에 주로 집중된 리뷰가 많았던 점을 지적합니다.

- **Technical Details**: 금융 시계열 데이터는 노이즈(noise)와 뉴스에 취약하며, 전통적인 통계 방법론은 선형성(linearity) 및 정규성(normality) 가정을 바탕으로 하였기 때문에 비선형적(non-linear) 특성을 잘 설명하지 못합니다. 신경망은 금융 가격 예측에 있어 주요한 머신 러닝 도구로 자리잡고 있으며, 전이 학습은 출발하는 작업(source task)에서 목표 작업(target task)으로 지식을 전이하는 방법으로, 금융 예측에 있어 유용한 도구가 될 수 있습니다.

- **Performance Highlights**: 전이 학습 방법론은 향후 주식 시장 예측의 도전과제 및 잠재적 미래 방향성을 탐구하는 데 중요한 역할을 할 것으로 기대됩니다.



### Fully automatic extraction of morphological traits from the Web: utopia or reality? (https://arxiv.org/abs/2409.17179)
- **What's New**: 이 논문은 최근의 대형 언어 모델(LLMs)을 활용하여 비구조적인 텍스트에서 식물의 형태학적 특성 정보를 자동으로 수집하고 처리하는 메커니즘을 제안합니다. 이를 통해 전문가가 수년간 수집해야 하는 복잡한 특성 정보를 손쉽게 구축할 수 있습니다.

- **Technical Details**: 제안된 방법론은 다음과 같은 세 가지 입력을 요구합니다: (i) 관심 있는 종의 목록, (ii) 관심 있는 특성 목록, 그리고 (iii) 각 특성이 가질 수 있는 모든 가능한 값의 목록. 이 프로세스는 검색 엔진 API를 사용하여 관련 URL를 가져오고, 이 URL에서 텍스트 콘텐츠를 다운로드합니다. 이후 NLP 모델을 통해 설명 문장을 판별하고, LLM을 사용하여 기술적 특성 값을 추출합니다.

- **Performance Highlights**: 이 방식으로 3개의 수작업으로 작성된 종-특성 행렬을 자동으로 복제하는 평가를 실시했습니다. 결과적으로 75% 이상의 F1-score를 달성하며, 50% 이상의 종-특성 쌍의 값을 찾는 데 성공했습니다. 이는 비구조적 온라인 텍스트로부터 구조화된 특성 데이터베이스를 대규모로 생성하는 것이 현재 가능하다는 점을 보여줍니다.



### CSCE: Boosting LLM Reasoning by Simultaneous Enhancing of Casual Significance and Consistency (https://arxiv.org/abs/2409.17174)
- **What's New**: 현재의 언어 모델(LLMs)에서 Chain-Based(체인 기반) 접근 방식에 의존하지 않고, 인과적 중요성(causal significance)과 일관성(consistency)을 동시에 고려할 수 있는 비체인 기반의 새로운 추론 프레임워크인 CSCE(Causal Significance and Consistency Enhancer)를 제안합니다.

- **Technical Details**: CSCE는 Treatment Effect(치료 효과) 평가를 활용하여 LLM의 손실 함수(loss function)를 맞춤화하고, 인과적 중요성과 일관성을 두 가지 측면에서 향상시킴으로써 인과관계를 정확히 파악하고 다양한 상황에서 견고하고 일관된 성능을 유지하도록 합니다. 이 프레임워크는 최대한의 추론 효율성을 위해 전체 추론 과정을 한 번에 출력합니다.

- **Performance Highlights**: CSCE 방법은 Blocksworld, GSM8K, Hanoi Tower 데이터셋에서 Chain-Based 방법들을 초월하여 높은 성공률과 빠른 처리 속도를 기록하며, 비체인 기반 방법이 LLM의 추론 작업을 완료하는 데에도 기여할 수 있음을 입증했습니다.



### A Multiple-Fill-in-the-Blank Exam Approach for Enhancing Zero-Resource Hallucination Detection in Large Language Models (https://arxiv.org/abs/2409.17173)
Comments:
          20 pages

- **What's New**: 본 논문에서는 이야기가 변경되는 문제를 해결하기 위해 다중 객관식 채우기 시험(multiple-fill-in-the-blank exam) 접근 방식을 포함한 새로운 환각(hallucination) 감지 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 원본 텍스트에서 여러 객체를 마스킹한 후 각 시험 응답이 원본 스토리라인과 일치하도록 LLM을 반복적으로 도와줍니다. 이 과정에서 발생하는 환각 정도를 채점하여 평가합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존 방법(SCGP)보다 우수한 성능을 보이며, SCGP와의 앙상블에서도 현저한 성능 향상을 나타냅니다.



### What Would You Ask When You First Saw $a^2+b^2=c^2$? Evaluating LLM on Curiosity-Driven Questioning (https://arxiv.org/abs/2409.17172)
- **What's New**: 본 논문에서는 대규모 언어 모델(LLM)의 새로운 지식 습득 가능성을 평가하기 위한 혁신적인 평가 프레임워크를 제안합니다. 이 프레임워크는 LLM이 과학적 지식을 소개하는 진술에 대한 질문을 생성하도록 유도하여, 처음 접하는 사람처럼 호기심을 가지고 질문하는 방식으로 평가합니다.

- **Technical Details**: 제안된 평가 방법은 호기심 기반 질문 생성(CDQG) 과제로, LLM이 처음 접하는 진술을 상상하며 즉각적으로 떠오르는 질문을 만들어내도록 프롬프트합니다. 생성된 질문은 관련성(relevance), 일관성(coherence), 다양성(diversity) 세 가지 주요 지표로 평가되며, 심리학 문헌에 뿌리를 두고 있습니다. 여러 모델의 성능을 비교 평가하기 위해 물리학, 화학 및 수학 분야의 1101개의 다양한 난이도의 진술로 구성된 합성 데이터셋을 수집했습니다.

- **Performance Highlights**: GPT-4와 Mistral 8x7b와 같은 대형 모델이 일관성 있고 관련성이 높은 질문을 잘 생성하는 반면, 크기가 작은 Phi-2 모델은 동등하거나 그 이상으로 효과적임을 발견했습니다. 이는 모델의 크기만으로 지식 습득 가능성을 판단할 수 없음을 나타냅니다. 연구 결과, 제안된 프레임워크는 LLM의 질문 생성 능력을 새로운 관점에서 평가할 수 있는 기회를 제공합니다.



### Cross-Domain Content Generation with Domain-Specific Small Language Models (https://arxiv.org/abs/2409.17171)
Comments:
          15 pages

- **What's New**: 이 연구는 여러 개별 데이터셋에서의 작은 언어 모델을 활용한 도메인 특정 콘텐츠 생성에 대한 새로운 접근 방식을 탐색합니다. 특히, 두 개의 서로 다른 도메인인 이야기 (story)와 레시피 (recipe)에 대한 모델을 비교하였습니다.

- **Technical Details**: 모델을 각각의 데이터셋에 대해 개별적으로 훈련시키는 방식이 사용되었으며, 사용자 정의 토크나이저 (custom tokenizer)를 적용하여 생성 품질을 크게 향상시켰습니다. 또한 Low-Rank Adaptation (LoRA)나 일반적인 파인튜닝 (fine-tuning) 방법으로 단일 모델을 두 도메인에 적용하려고 시도했지만 유의미한 결과를 얻지 못했습니다. 특히, 전체 파인튜닝(full fine-tuning)중 모델의 가중치 동결 없이 진행 시, 치명적 망각 (catastrophic forgetting)이 발생하였습니다.

- **Performance Highlights**: 지식 확장 전략 (knowledge expansion strategy)을 통해 모델이 이야기와 레시피를 요청에 따라 생성할 수 있도록 하였으며, 이는 서로 다른 데이터셋 간의 의미 있는 출력을 유지하도록 하였습니다. 연구 결과는 고정된 레이어를 가진 지식 확장이 작은 언어 모델이 다양한 도메인에서 콘텐츠를 생성하는 데 효과적인 방법임을 보여줍니다.



### REAL: Response Embedding-based Alignment for LLMs (https://arxiv.org/abs/2409.17169)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)과 인간의 선호를 일치시키기 위한 효율적인 데이터 선택 전략을 제안합니다. 기존 알고리즘의 한계를 극복하기 위해 의미 있는 응답 쌍을 선택하는 방법론을 개발하였습니다.

- **Technical Details**: 제안된 방법은 HH-RLHF 데이터셋에서 유사하지 않은 응답 쌍을 선택하여 LLM의 직접적인 정렬을 개선하며 라벨링 오류를 감소시키는 것을 목표로 합니다. 코사인 유사성을 기반으로 응답 쌍의 유용성을 평가하여 높은 품질의 학습 데이터를 형성합니다.

- **Performance Highlights**: 실험 결과, 유사하지 않은 응답 쌍을 사용한 모델이 대화 작업에서 최상의 승률을 기록하였으며, 라벨러의 작업을 최대 65%까지 절감하는 효율성을 보여주었습니다.



### StressPrompt: Does Stress Impact Large Language Models and Human Performance Similarly? (https://arxiv.org/abs/2409.17167)
Comments:
          11 pages, 9 figures

- **What's New**: 이번 연구는 Large Language Models (LLMs)가 인간과 유사한 스트레스 반응을 보이는지를 탐구하고, 다양한 스트레스 유도 프롬프트 하에서의 성능 변화를 평가합니다. 이는 스트레스의 심리적 원리를 바탕으로 한 새로운 종류의 프롬프트 세트, StressPrompt를 통해 이루어졌습니다.

- **Technical Details**: 연구에서는 심리학적 이론에 기반하여 설계된 100개의 프롬프트를 개발하였으며, 이는 각각 다른 수준의 스트레스를 유도하도록 설계되었습니다. 또한, LLM의 내부 상태와 성능에 미치는 스트레스의 영향을 측정하기 위한 '스트레스 스캐너'를 도입했습니다.

- **Performance Highlights**: 연구 결과, LLM은 중간 수준의 스트레스 하에서 최적의 성능을 보이며, 낮은 스트레스와 높은 스트레스 모두에서 성능이 저하되는 것으로 나타났습니다. 이는 Yerkes-Dodson 법칙에 부합하며, 고객 서비스, 의료, 응급 대응과 같은 실세계 시나리오에서 AI 시스템의 성능 유지의 중요성을 시사합니다.



### ScriptSmith: A Unified LLM Framework for Enhancing IT Operations via Automated Bash Script Generation, Assessment, and Refinemen (https://arxiv.org/abs/2409.17166)
Comments:
          Under Review

- **What's New**: 본 논문에서는 사이트 신뢰성 엔지니어링(SRE) 분야에서 발생하는 이슈를 보다 효율적으로 관리하고 해결하기 위한 혁신적인 접근 방식을 제시합니다. 대규모 언어 모델(LLMs)을 활용하여 Bash 스크립트 생성, 평가 및 개선을 자동화하는 방법을 통해 SRE 팀의 생산성을 향상시키고자 합니다.

- **Technical Details**: 이 연구에서는 CodeSift 데이터셋(100개 작업)과 InterCode 데이터셋(153개 작업)을 활용하여 LLMs가 스크립트를 자동으로 평가하고 개선할 수 있는지 실험하였습니다. 이를 통해 실행 환경에 의존하지 않는 프레임워크 'ScriptSmith'를 개발하여, 추천된 행동 단계에 대한 올바른 Bash 스크립트를 생성하는 데 중점을 두었습니다. 해당 프레임워크는 과거 작업의 카탈로그에서 일치하는 스크립트를 찾고, 없다면 새로운 스크립트를 동적으로 생성합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 스크립트 생성에서 7-10%의 전체적인 개선을 보였으며, Llama3과 Gemini 모델을 바탕으로 한 Bash 데이터셋 평가에서도 유의미한 정확도를 기록하였습니다. 특히, Llama3_70B 모델은 CodeSift 데이터셋에서 75%의 정확도를 보였으며, Gemini1.5_Pro 모델 역시 우수한 성능을 보여주었습니다.



### Cross Dataset Analysis and Network Architecture Repair for Autonomous Car Lane Detection (https://arxiv.org/abs/2409.17158)
- **What's New**: 본 연구에서는 자율주행 차량의 차선 인식 애플리케이션을 위한 교차 데이터세트 분석과 신경망 아키텍처 수리를 수행합니다. 제공된 아키텍처인 ERFCondLaneNet은 복잡한 형상의 차선을 탐지하는 데 어려움을 겪는 기존의 CondLaneNet을 개선한 것입니다.

- **Technical Details**: 이 연구에서 제안하는 ERFCondLaneNet은 CondLaneNet [liu_condlanenet_2021]과 ERFNet [romera_erfnet_2018]을 통합하여 만들어졌으며, Transfer Learning (TL) 프로토콜을 사용하여 두 가지 주요 차선 탐지 벤치마크인 CULane과 CurveLanes에서 테스트되었습니다. 이 기술은 성능을 유지하면서도 33% 적은 특징을 사용하여 모델 크기를 46% 줄였습니다.

- **Performance Highlights**: ERFCondLaneNet은 ResnetCondLaneNet과 비슷한 성능을 보이며, 이는 복잡한 지형을 가진 차선 탐지에서  충분한 정확성을 유지합니다. 학습 과정에서 기존 모델보다 적은 데이터로도 우수한 결과를 보여줍니다.



### Confident Teacher, Confident Student? A Novel User Study Design for Investigating the Didactic Potential of Explanations and their Impact on Uncertainty (https://arxiv.org/abs/2409.17157)
Comments:
          15 pages, 5 figures, 1 table, presented at ECML 2024, AIMLAI Workshop, Vilnius

- **What's New**: 이 연구는 Explainable Artificial Intelligence (XAI)의 평가를 위한 실험 디자인을 제안하며, 1200명의 참가자를 대상으로 AI와의 협력 설정에서 설명이 인간 성과에 미치는 영향을 조사합니다.

- **Technical Details**: 이 연구에서는 복잡한 생물 분류의 시각적 주석 작업에서 XAI의 잠재력을 평가하였고, 사용자가 기계의 예측을 보여줄 때와 설명을 추가했을 때의 차이를 분석했습니다. 또한, 사용자의 주석이 AI 도움 후에도 유의미하게 개선되지 않았음을 발견했습니다.

- **Performance Highlights**: 사용자는 AI 지원으로 주석 정확도가 높아졌지만, 모델의 예측을 보여주는 것과 설명을 제공하는 것의 효과는 유의미한 차이가 없었습니다. 또한, 사용자가 잘못된 예측을 반복하는 부정적 효과가 발견되었습니다.




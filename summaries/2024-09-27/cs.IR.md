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




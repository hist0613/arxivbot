New uploads on arXiv(cs.CL)

### Recycled Attention: Efficient inference for long-context language models (https://arxiv.org/abs/2411.05787)
- **What's New**: 새로운 접근법, Recycled Attention을 제안하여 긴 컨텍스트 입력에 대한 inference 시간을 줄이는 동시에 모델 성능을 유지합니다. 이전의 Token에 대한 주의 패턴을 재활용하여 주의를 부분적으로 수행하고, 현재 decoding 단계와 관련있는 Token을 유연하게 선택합니다.

- **Technical Details**: Recycled Attention은 전체 KV 캐시를 유지하면서, 동적으로 구성된 더 작은 KV 캐시에 대해 주의 계산을 수행합니다. 이 방법은 두 가지 모드로 생성이 가능합니다: 전체 KV 캐시에서 주의를 수행하는 생성과 Token의 하위 집합에 대해 주의를 수행하는 생성입니다. 주요 아이디어는 인접한 Token들이 이전 Token에 대한 주의를 공유한다는 것입니다.

- **Performance Highlights**: Recycled Attention을 사용하여 LlaMa-3.1-8B 및 QWEN-2-7B와 같은 긴 컨텍스트 LLM에서 기존의 StreamingLLM 및 H2O보다 2배 이상의 성능 향상과 같은 속도 향상을 달성했습니다. 성능 개선은 작업 요구 사항에 따라 지역 및 비지역 컨텍스트에 유연하게 주의할 수 있는 Recycled Attention의 능력에 기인합니다.



### ASL STEM Wiki: Dataset and Benchmark for Interpreting STEM Articles (https://arxiv.org/abs/2411.05783)
Comments:
          Accepted to EMNLP 2024

- **What's New**: ASL STEM Wiki는 영어로 된 254개의 Wikipedia 기사를 아메리칸 수화(ASL)로 해석한 병렬 말뭉치로, 청각 장애인(DHH) 학생들이 STEM 교육을 더 쉽게 접근할 수 있도록 돕기 위한 첫 번째 연속 수화 데이터셋입니다.

- **Technical Details**: 이 데이터셋은 37명의 인증된 통역사가 ASL로 해석한 64,266개의 문장과 300시간 이상의 ASL 비디오로 구성되어 있으며, STEM 관련 5개 주제에 걸쳐 진행되었습니다. 데이터셋은 일반적인 영어 문장을 ASL 비디오와 매칭하여 수화 연산 모델에 새로운 도전과제를 제시합니다.

- **Performance Highlights**: 이 논문에서 제안한 모델은 STEM 콘텐츠의 손가락 철자가 자주 사용되는 문제를 해결하고자 하며, 손가락 철자 탐지 정확도를 47% 개선하는 contrastive learning 기법을 사용하였습니다. 결과적으로, ASL 교육 자료 접근성을 높이고, 기술적인 ASL 수화 사용을 촉진하는 도구 개발을 위한 도약의 기회를 제공합니다.



### Using Language Models to Disambiguate Lexical Choices in Translation (https://arxiv.org/abs/2411.05781)
Comments:
          Accepted to EMNLP 2024

- **What's New**: 본 연구에서는 단어가 여러 변형으로 번역되는 경우의 모호함을 해결하기 위해, DTAiLS라는 데이터셋을 구축했습니다. 이 데이터셋은 9개의 언어를 포함하여 1,377개의 문장 쌍으로 구성되어 있으며, 다양한 언어 간 개념 변형을 포함합니다.

- **Technical Details**: DTAiLS(Translation with Ambiguity in Lexical Selection)는 단어 선택의 맥락을 이해하기 위한 데이터셋입니다. 본 논문에서는 LLM(대형 언어 모델)과 NMT(신경 기계 번역) 시스템의 성능을 평가하였고, GPT-4 모델이 67%에서 85%의 정확도를 달성했습니다. LLM을 통해 생성된 규칙을 제공함으로써 더 낮은 성능의 모델도 정확도를 개선할 수 있음을 보여주었습니다.

- **Performance Highlights**: 최고 성능을 기록한 GPT-4는 여러 언어에서 67%에서 85%의 정확도를 보여주었으며, 자가 생성된 규칙을 활용했을 때 모든 LLM의 성능이 향상되었습니다. 또한, GPT-4의 규칙을 활용하여 오픈 웨이트 LLM이 NMT 시스템에 비해 성능 격차를 줄일 수 있음을 시사합니다.



### Quantitative Assessment of Intersectional Empathetic Bias and Understanding (https://arxiv.org/abs/2411.05777)
- **What's New**: 본 논문은  현재의 공감(complex empathy) 정의의 모호함으로 인해 공감 평가 방법의 비효율성을 비판하고, 공감의 심리적 기원에 가깝게 공감을 구체적으로 측정할 수 있는 평가 프레임워크를 제안합니다.

- **Technical Details**: 제안된 JaEm-ST 프레임워크는 공감을 두 가지 차원, 즉 인지적 공감(Cognitive Empathy, CE)과 정서적 공감(Affective Empathy, AE)으로 나누어 정의하며, 각 차원에 대한 측정 방법 및 평가 절차를 포함합니다. 평가 데이터 세트는 마스킹된 템플릿을 통해 생성되며, 다양한 사회적 편견(social biases)을 적용하여 대화 에이전트와의 공감 이해도를 계산합니다.

- **Performance Highlights**: 초기 평가 샘플에서 모델 간의 공감 이해도 차이는 크지 않았으나, 모델의 추론(chain of reasoning) 과정에서 prompt의 미세한 변화에 따른 유의미한 차이를 보였습니다. 이는 향후 공감 평가 샘플 구성 및 통계적 방법론에 대한 연구의 기초가 될 것입니다.



### Fact or Fiction? Can LLMs be Reliable Annotators for Political Truths? (https://arxiv.org/abs/2411.05775)
Comments:
          Accepted at Socially Responsible Language Modelling Research (SoLaR) Workshop at NeurIPS 2024

- **What's New**: 이 연구는 오픈 소스 대형 언어 모델(LLMs)을 정치적 사실성을 측정하기 위한 신뢰할 수 있는 주석자로 활용하는 방법을 탐구합니다. 이를 통해 전통적인 사실 확인 방법의 한계를 극복하고, 언론에 대한 투명성과 신뢰를 높이고자 합니다.

- **Technical Details**: 정치 편향(political bias)에 대한 분석을 바탕으로, LLMs를 사용하여 정치 관련 뉴스 기사를 사실적으로 정확한 것과 부정확한 것으로 이분화하는 주석 작업을 수행합니다. 또한, LLM 주석자의 정확성을 평가하기 위해 LLM을 평가자로 활용하는 방법론을 제시합니다.

- **Performance Highlights**: 실험 결과, LLM이 생성한 주석은 인간 주석자와 높은 일치를 보였으며, LLM 기반 평가에서는 강력한 성능을 나타내어 사실 확인 과정에서 효과적인 대안임을 입증하였습니다.



### FinDVer: Explainable Claim Verification over Long and Hybrid-Content Financial Documents (https://arxiv.org/abs/2411.05764)
Comments:
          EMNLP 2024

- **What's New**: 이번 논문에서는 FinDVer라는 새로운 벤치마크를 소개합니다. 이는 LLM(대형 언어 모델)의 설명 가능한 청구 검증 기능을 평가하기 위해 특별히 설계된 포괄적인 기준입니다. 유효한 정보 추출, 수치 추론, 지식 집약적 추론의 세 가지 하위 집합으로 나뉘어 있으며, 총 2,400개의 전문가 주석 샘플이 포함되어 있습니다.

- **Technical Details**: FinDVer는 금융 문서에서의 청구 검증을 위한 최초의 맥락 기반 벤치마크로, 두 가지 설정인 길이가 긴 문맥(long-context)과 검색 보강 생성(RAG) 설정에서 다양한 LLM을 평가합니다. 저자들은 9개 기관의 16개 모델을 테스트하였고, 각 예제는 세부적인 증거 지원과 단계별 추론 과정을 제공하는 주석으로 뒷받침됩니다.

- **Performance Highlights**: 실험 결과, 현재 가장 성능이 뛰어난 LLM인 GPT-4o조차도 인간 전문가보다 상당히 뒤처지는 것으로 나타났습니다(76.2% 대 93.3%). 이러한 결과는 LLM이 금융 문서의 복잡성을 이해하고 처리하는 데 여전히 많은 도전 과제가 있음을 보여줍니다.



### Multi-hop Evidence Pursuit Meets the Web: Team Papelo at FEVER 2024 (https://arxiv.org/abs/2411.05762)
Comments:
          To appear in the Seventh FEVER Workshop at EMNLP 2024

- **What's New**: 이번 연구에서는 대형 언어 모델(large language models, LLMs)과 현대 검색 엔진(search engines)을 통합하여 온라인에서의 잘못된 정보(disinformation)와 사실(fact)을 효과적으로 분리하는 자동화된 시스템을 제안합니다.

- **Technical Details**: 연구에서 제안된 접근법은 멀티 홉 증거 추구 전략(multi-hop evidence pursuit strategy)을 채택하여 입력 주장(claim) 기반의 초기 질문을 생성하고, 검색 후 답변을 공식화하며, 추가 질문을 반복적으로 생성하여 부족한 증거를 추구하는 방식으로 구성되어 있습니다. 이 시스템은 FEVER 2024(AVeriTeC) 공유 과제에서 효과를 입증하였습니다.

- **Performance Highlights**: 제안된 방법은 모든 질문을 한 번에 생성하는 전략에 비해 라벨 정확도(label accuracy)가 0.045 높고, AVeriTeC 점수는 0.155 높았습니다. 최종적으로 개발 세트(dev set)에서 0.510의 AVeriTeC 점수를 달성하였으며, 테스트 세트(test set)에서는 0.477의 점수를 기록했습니다.



### Asterisk*: Keep it Simp (https://arxiv.org/abs/2411.05691)
- **What's New**: 이 논문에서는 Asterisk라는 소형 GPT 기반의 텍스트 임베딩 생성 모델을 설명합니다. 이 모델은 두 개의 레이어, 두 개의 어텐션 헤드, 256 차원의 임베딩을 포함한 미니멀리스트 아키텍처(minimalist architecture)를 가지고 있으며, 대형 사전 학습(pre-trained) 모델로부터 지식 증류(Knowledge Distillation)를 활용하여 모델 크기와 성능 간의 균형을 탐색합니다.

- **Technical Details**: Asterisk 모델은 256차원 임베딩 공간과 2개의 Transformer 레이어, 각 레이어 당 2개의 어텐션 헤드를 사용합니다. 총 14,019,584개의 파라미터로 구성되어 있으며, 임베딩 레이어는 토큰 임베딩과 위치 임베딩을 결합하여 초기화되었습니다. 이 모델은 주로 MSE(Mean Squared Error)와 코사인 유사성(Cosine Similarity)을 결합하여 지식 증류 과정을 구현하였고, OpenAI의 text-embedding-3-small을 교사 모델로 선택하여 사용했습니다.

- **Performance Highlights**: Asterisk 모델은 다양한 분류 작업에서 실험적으로 중간 수준의 성능을 보였으며, 사전 훈련된 큰 모델들과 비교하여 특정 작업에서는 성능이 유사하거나 뛰어난 결과를 나타냈습니다. 특히 Fully-Connected 네트워크를 통해 1000개의 샘플 훈련만으로도 신뢰할 수 있는 분류 성능을 달성하였고, MTEB 리더보드에서는 MassiveIntentClassification(1위)와 AmazonReviewsClassification(2위) 작업에서 성과를 거두었습니다.



### Unmasking the Limits of Large Language Models: A Systematic Evaluation of Masked Text Processing Ability through MskQA and MskCa (https://arxiv.org/abs/2411.05665)
Comments:
          16 pages

- **What's New**: 이 논문은 Large Language Models (LLMs)의 한계를 밝히고, 마스킹된 텍스트 처리 능력을 평가하는 새로운 두 가지 작업을 제시합니다: MskQA(Mask Question-Answering)와 MskCal(Masked Calculations).

- **Technical Details**: MskQA는 마스킹된 질문-응답 데이터셋에서의 추론 능력을 측정하고, MskCal은 마스킹된 수치 계산에서의 수리적 추론을 평가합니다. 연구 결과, GPT-4o가 4o-mini보다 일관되게 우수한 성과를 보였으며, 이는 주로 의미적 단서(semantic cues)에 의존하는 경향을 보여줍니다.

- **Performance Highlights**: GPT-4o는 MskCal 작업에서 특히 우수한 수리적 추론 능력을 발휘하며, ‘solid masking’ 조건하에서는 성능이 크게 떨어지는 반면, ‘partial lifting’ 조건에서는 성능이 상대적으로 유지되는 것을 확인했습니다.



### Evaluating Large Language Model Capability in Vietnamese Fact-Checking Data Generation (https://arxiv.org/abs/2411.05641)
- **What's New**: 본 논문은 베트남어와 같은 자원이 제한된 언어에서 대형 언어 모델(Large Language Models, LLMs)을 활용한 자동 데이터 생성에 대한 연구를 수행했습니다.

- **Technical Details**: LLMs의 정보 종합 능력을 평가하기 위해 여러 증거 문장에서 주장을 종합하는 사실 확인 데이터(fact-checking data)를 생성하는 과정을 다룹니다. 이를 위해 간단한 프롬프트(prompt) 기법을 활용한 자동 데이터 구축 프로세스를 개발하고 생성된 데이터의 품질을 개선하기 위한 여러 방법을 모색합니다.

- **Performance Highlights**: 실험 결과와 수동 평가(manual evaluations)에서 데이터의 품질은 미세 조정(fine-tuning) 기술을 통해 상당히 향상되었으나, LLMs가 생성한 데이터 품질은 여전히 인간이 생성한 데이터와는 차이가 있음을 보여주었습니다.



### Assessing Open-Source Large Language Models on Argumentation Mining Subtasks (https://arxiv.org/abs/2411.05639)
- **What's New**: 이 연구는 네 개의 오픈소스 대형 언어 모델(LLMs)인 Mistral 7B, Mixtral8x7B, Llama2 7B, Llama3 8B의 논증 추출(Argumentation Mining, AM) 능력을 탐구합니다. 이들은 제로샷(zero-shot) 및 몇 샷(few-shot) 시나리오에서 실험되며, 두 가지 하위 작업에 대한 성능을 평가합니다.

- **Technical Details**: 논증 추출에는 두 가지 주요 하위 작업이 포함되며, 첫째는 논증 담론 단위 분류(Argumentative Discourse Unit Classification, ADUC)이고, 둘째는 논증 관계 분류(Argumentative Relation Classification, ARC)입니다. ADUC는 주장의 유형을 식별하고, ARC는 논증 단위 간의 관계를 정의합니다. 이 연구에서는 세 가지 주요 데이터 셋(AMT1, AMT2, PE)을 사용하여 실험합니다.

- **Performance Highlights**: 결과적으로, Ne LLMs는 다양한 데이터셋과 설정에서 기존의 모델들에 비해 뛰어난 성능을 보이며, 논증 구성 요소의 식별 정확도를 크게 향상시켰습니다. 이러한 결과는 오픈소스 LLM이 계산적 논증 작업에 효과적으로 활용될 수 있다는 것을 시사합니다.



### Impact of Fake News on Social Media Towards Public Users of Different Age Groups (https://arxiv.org/abs/2411.05638)
- **What's New**: 이 연구는 가짜 뉴스가 다양한 연령대의 소셜 미디어 사용자에게 미치는 영향을 조사하고, 머신 러닝 (ML)과 인공지능 (AI)이 잘못된 정보의 확산을 줄이는 데 어떻게 도움이 되는지를 분석했습니다.

- **Technical Details**: 이 논문은 Kaggle 데이터셋을 사용하여 가짜 뉴스를 식별하고 분류하는 다양한 머신 러닝 모델의 효능을 평가했습니다. 평가한 모델은 랜덤 포레스트 (Random Forest), 서포트 벡터 머신 (Support Vector Machine, SVM), 신경망 (Neural Networks), 로지스틱 회귀 (Logistic Regression)로, SVM과 신경망이 각각 93.29%와 93.69%의 정확도로 다른 모델들보다 우수한 성능을 보였습니다.

- **Performance Highlights**: 노인 집단의 비판적 분석 능력이 저조하여 잘못된 정보에 더 쉽게 영향을 받는다는 점이 강조되었습니다. 자연어 처리 (Natural Language Processing, NLP)와 딥러닝 접근 방식이 잘못된 뉴스 탐지의 정확도를 개선할 수 있는 잠재력이 있으며, AI와 ML 모델의 편견과 AI가 생성한 정보 식별의 어려움이 여전히 주요 문제로 남아 있습니다.



### Evaluating and Adapting Large Language Models to Represent Folktales in Low-Resource Languages (https://arxiv.org/abs/2411.05593)
- **What's New**: 이 논문은 민속학 분야를 위한 새로운 접근 방식을 제시합니다. 특히 아일랜드어와 스코틀랜드 게일어의 데이터를 활용하여 대규모 언어 모델(LLMs)의 저자원 언어에 대한 분류 성능을 평가하고 개선하는 방법을 탐구합니다.

- **Technical Details**: 논문에서는 아일랜드어와 스코틀랜드 게일어의 민속 이야기를 기반으로 두 가지 분류 작업을 설정했습니다. 첫 번째는 민속 이야기의 유형을 예측하는 것이고, 두 번째는 이야기꾼의 성별을 예측하는 것입니다. 성별 분포는 아일랜드 데이터에서 남성이 83.6%, 게일 데이터에서 83.2%로 편향되어 있습니다. 또한, LLM의 기본 모델을 사용하여 저자원 언어를 어떻게 잘 표현하는지를 평가하며, 도메인 적응(domain adaptation)과 최대 시퀀스 길이의 증가가 성능 향상에 미치는 영향을 실험합니다.

- **Performance Highlights**: 기본 모델 또한 SVM과 같은 비맥락적 특징을 활용한 모델과 비교해 우수한 성능을 보였으며, 변형된 모델을 통해 긴 시퀀스를 처리하고 민속 이야기에 대한 지속적인 사전 훈련을 통한 향상된 성능을 확인했습니다.



### Assessing the Answerability of Queries in Retrieval-Augmented Code Generation (https://arxiv.org/abs/2411.05547)
- **What's New**: 이 연구는 Retrieval-augmented Code Generation (RaCG) 환경에서 사용자의 쿼리에 대한 답변 가능성을 평가하는 새로운 작업을 제안합니다. 이는 LLM이 적절한 응답을 제공할 수 있는지를 먼저 판단하게끔 하는 접근방법입니다.

- **Technical Details**: RaCG에서 사용자의 쿼리가 답변 가능한지를 평가하는 작업을 수행하며, 이를 위해 Retrieval-augmented Code Generability Evaluation (RaCGEval)이라는 기준 데이터세트를 구축하였습니다. 이 데이터세트는 각각의 입력 프롬프트가 주어진 사용자 쿼리와 관련된 API 정보를 기반으로 해서 올바른 코드가 유도될 수 있는지를 결정합니다.

- **Performance Highlights**: 실험 결과, 이 작업은 46.7%라는 낮은 성능 결과를 보였으며, 세 가지 분류 문제(답변 가능성, 부분 답변 가능성, 답변 불가능성)로 나뉘어 분류됩니다. 또한 성능 개선을 위한 다양한 방법에 대해서도 논의합니다.



### How Good is Your Wikipedia? (https://arxiv.org/abs/2411.05527)
- **What's New**: 이 논문은 비영어권에서 위키피디아의 데이터 품질에 대한 비판적인 고찰을 다루고 있으며, 특히 저자원이 부족한 언어에 대해 데이터 품질 필터링 기법의 효과를 분석합니다.

- **Technical Details**: 주요 내용으로는 여러 필터링 기법을 적용하여 위키피디아에서 발견되는 문제점들을 드러내며, 비영어권 위키피디아에서 중복 기사 및 한 줄짜리 기사 비율이 높다는 점을 밝혔습니다. 이 연구는 패턴 인식 및 내용 기반의 품질 필터를 활용한 품질 정제 기법을 논의합니다.

- **Performance Highlights**: 연구 결과에 따르면, 데이터 품질 필터링은 성능을 저하시킴이 없이 자원 효율적인 훈련을 가능하게 하며, 특히 저자원이 부족한 언어의 경우 더욱 뚜렷한 개선 효과를 보여줍니다.



### LBPE: Long-token-first Tokenization to Improve Large Language Models (https://arxiv.org/abs/2411.05504)
Comments:
          arXiv admin note: text overlap with arXiv:2404.17808

- **What's New**: 이 논문에서는 기존의 Byte Pair Encoding (BPE) 방식의 한계를 극복하기 위해 LBPE(Long-Byte Pair Encoding)이라는 새로운 접근 방식을 제안합니다. LBPE는 긴 토큰을 우선적으로 인코딩하여 짧은 토큰과의 불균형 학습 문제를 해결하려고 합니다.

- **Technical Details**: LBPE는 토큰 길이의 역 순위를 기반으로 토큰을 병합하는 인코딩 알고리즘을 사용하여, 긴 토큰의 빈도수를 높이고 최종 토큰 표현에서 이를 반영합니다. 기존 BPE 방식에 따르면, 빈도가 높은 짧은 토큰이 먼저 병합되어 긴 토큰의 학습이 어려워지는 문제가 발생했습니다. LBPE는 이러한 문제를 해결하기 위해 긴 토큰 우선 인코딩을 적용합니다.

- **Performance Highlights**: 다양한 언어 모델링 작업에서의 실험 결과, LBPE는 기존 BPE보다 일관되게 우수한 성능을 보였습니다. 특히, 연속적인 사전 훈련에서도 LBPE의 사용이 권장되며, 기존 BPE 수정 방식과도 통합할 수 있어 추가적인 개선이 가능합니다.



### KyrgyzNLP: Challenges, Progress, and Futur (https://arxiv.org/abs/2411.05503)
Comments:
          Keynote talk at the 12th International Conference on Analysis of Images, Social Networks and Texts (AIST-2024)

- **What's New**: 이 논문은 현재 자연어 처리(NLP) 분야에서 부족한 자원을 가진 키르기스어(kyrgyz tili)의 상황을 조명합니다. 인간 평가와 원어민이 만든 주석 데이터셋의 중요성을 강조하며, 특히 자동 평가가 부족한 LRL(less-resourced languages)에서의 신뢰성 있는 NLP 성능의 필요성을 설명합니다.

- **Technical Details**: 키르기스어는 'Scraping By'라는 평가를 받으며, 이는 디지털 도구, 데이터셋 및 모델의 심각한 부족을 나타냅니다. 이 연구에서는 자원 부족, 방언 다양성, 복합적인 형태론의 복잡성 등의 NLP 상황을 다루며, NLP 도구 개발을 위해 원어민 및 언어 전문가의 참여가 필수적임을 강조합니다.

- **Performance Highlights**: 기술 발전의 필요성에 대한 인식을 높이고, 정부 및 민간 분야의 지원을 강조합니다. 또한, 데이터 품질 및 구체성이 NLP 성능 향상에 중요한 요소임을 보여주며, 키르기스어 NLP 분야에서의 연구 주제 및 자원 개발을 위한 로드맵을 제안합니다.



### Supporting Automated Fact-checking across Topics: Similarity-driven Gradual Topic Learning for Claim Detection (https://arxiv.org/abs/2411.05460)
- **What's New**: 이 연구에서는 아랍어에 대한 사실 검증용 주장(selecting check-worthy claims)을 선택하는 새로운 도메인 적응 프레임워크를 제안합니다. 이는 다양한 주제에 대해 신뢰성 있는 주장을 검증하는 데 도움을 주며, 매일 발생하는 사건을 반영한 실제 시나리오를 모사합니다.

- **Technical Details**: Gradual Topic Learning (GTL) 모델이 단계적 학습을 통해 목표 주제에 대한 체크 가치가 있는 주장을 강조하는 방식을 제안합니다. 또한, Similarity-driven Gradual Topic Learning (SGTL) 모델은 목표 주제에 대한 유사성 기반 전략과 점진적 학습을 통합합니다. 이러한 두 모델은 새로운 주제에 효율적으로 적응할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 제안한 모델은 14개의 주제 중 11개에서 최첨단(baseline) 모델에 비해 전반적으로 성능이 향상됨을 보여주었습니다.



### VISTA: Visual Integrated System for Tailored Automation in Math Problem Generation Using LLM (https://arxiv.org/abs/2411.05423)
Comments:
          Accepted at NeurIPS 2024 Workshop on Large Foundation Models for Educational Assessment (FM-Assess)

- **What's New**: 새로운 연구에서는 Large Language Models (LLMs)를 활용하여 수학 교육에서 복잡한 시각적 보조 도구를 자동으로 생성하는 멀티 에이전트 프레임워크를 소개합니다. 이 프레임워크는 수학 문제와 관련된 시각 보조 도구를 정확하고 일관되게 생성하도록 설계되었습니다.

- **Technical Details**: 이 연구에서는 7개의 특수화된 에이전트로 구성된 시스템을 설계하여 문제 생성 및 시각화 작업을 세분화하였습니다. 각 에이전트는 Numeric Calculator, Geometry Validator, Function Validator, Visualizer, Code Executor, Math Question Generator, Math Summarizer로 구성되며, 각 에이전트는 특정 역할에 맞춰 문제를 해결합니다.

- **Performance Highlights**: 시스템은 Geometry와 Function 문제 유형에서 기존의 기본 LLM보다 텍스트의 일관성, 관련성 및 유사성을 크게 향상시켰으며, 수학적 정확성을 유지하는 데 주목할 만한 성과를 보였습니다. 이를 통해 교육자들이 수학 교육에서 시각적 보조 도구를 보다 효과적으로 활용할 수 있게 되었습니다.



### Gap-Filling Prompting Enhances Code-Assisted Mathematical Reasoning (https://arxiv.org/abs/2411.05407)
- **What's New**: 이 논문에서는 작은 언어 모델(SLMs)의 문제 해결 능력을 향상시키기 위한 새로운 전략인 Gap-Filling Prompting (GFP)을 소개합니다. 이 방법은 질문의 간극을 식별하고 이를 보완하기 위한 힌트를 제공하여 최종 코드 솔루션을 생성하는 두 단계 프로세스를 기반으로 합니다.

- **Technical Details**: GFP는 두 단계로 구성됩니다: 첫 번째 단계에서 SLM은 질문의 간극을 메우기 위한 힌트를 생성하고, 두 번째 단계에서는 이 힌트를 질문에 추가하여 최종적인 Python 코드 솔루션을 생성합니다. 이 연구에서는 데이터 합성을 위해 OpenAI의 GPT-4 모델을 사용하였고, 힌트 생성에는 Flan-T5 모델, 코드 생성을 위해 CodeT5 모델을 사용했습니다.

- **Performance Highlights**: 실험 결과, GFP를 적용한 SLM은 두 개의 벤치마크 데이터셋에서 수학적 추론 능력이 크게 향상되었음을 보여주었습니다. 이는 기존의 방법들과 비교하여 더 낮은 계산 오류를 기록했습니다.



### Benchmarking Distributional Alignment of Large Language Models (https://arxiv.org/abs/2411.05403)
- **What's New**: 이 논문은 언어 모델(LLMs)이 특정 인구 집단의 의견 분포를 얼마나 잘 모사할 수 있는지에 대한 불확실성을 다루고, 이를 위한 벤치마크를 제공하여 세 가지 주요 변수를 탐구합니다: 질문 도메인, 스티어링 방법, 그리고 분포 표현 방법입니다.

- **Technical Details**: 저자들은 LLM의 분포 정렬(distributional alignment)을 평가하기 위해 NYT Book Opinions라는 새로운 데이터세트를 수집하고, 모델이 특정 집단의 의견 분포에 맞추기 위해 어떻게 스티어링 되어야 하는지를 연구합니다. 이 작업은 모델이 의견 분포를 어떻게 제시하는지에 따라 다르며, 기존의 log-probabilities 기반 방식이 LLM의 성능을 과소평가할 수 있음을 발견했습니다.

- **Performance Highlights**: 연구에 따르면, LLM은 텍스트 기반 형식에서 (예: 'JSON으로 분포 반환') 의견 분포를 보다 정확하게 추정할 수 있으며, 정치 및 문화적 가치를 넘어서 비문화적 의견(예: 책 선호도)의 정렬 및 스티어링에서 상당한 격차가 존재합니다. 이로 인해 LLM의 인간 의견 시뮬레이션 능력을 향상시킬 기회가 부각됩니다.



### Towards Low-Resource Harmful Meme Detection with LMM Agents (https://arxiv.org/abs/2411.05383)
Comments:
          EMNLP 2024

- **What's New**: 이 논문에서는 적은 리소스 환경에서의 유해한 밈 탐지를 위한 새로운 에이전시 기반 프레임워크, LoReHM을 제안합니다. 이는 소수의 주석이 있는 샘플을 이용한 외부 및 내부 분석을 결합하여 밈의 유해성을 효과적으로 탐지하고자 합니다.

- **Technical Details**: 이 연구는 Large Multimodal Models (LMMs)을 활용하여 유해한 밈의 특성을 파악하기 위해 외부 유사 밈을 검색하고, 단기적인 경험을 축적할 수 있는 내적인 학습 과정을 모사하여 정확한 유해성 추론을 가능하게 합니다. 최종적으로, 외부 및 내부 접근 방식을 결합하여 LMM 에이전트가 유해한 콘텐츠를 탐지하는 능력을 향상시킵니다.

- **Performance Highlights**: 세 개의 밈 데이터셋에서 수행된 실험 결과, 제안된 접근 방식이 적은 리소스 환경에서 유해한 밈 탐지 작업에서 최신 기법들보다 우수한 성능을 보였습니다.



### Word reuse and combination support efficient communication of emerging concepts (https://arxiv.org/abs/2411.05379)
Comments:
          Published in Proceedings of the National Academy of Sciences

- **What's New**: 본 연구에서는 단어 재사용(word reuse)과 단어 조합(word combination)이라는 두 개의 사전적 동의어를 통한 새로운 개념의 생성 과정을 다루며, 이러한 두 가지 전략이 효율적인 커뮤니케이션을 위한 기본적인 균형(tradeoff)에 의해 제한된다는 정보 이론적 관점을 제시합니다.

- **Technical Details**: 이 논문은 단어 재사용과 조합이 의사소통의 효율성을 극대화하려는 압력에 의해 형성된다는 이론적 제안을 공식화합니다. 특히, 단어 재사용은 평균 단어 형태의 길이를 보존하며 정밀성을 감소시키고, 단어 조합은 더 많은 정보를 제공하지만 단어 길이를 증가시킨다는 것을 제안하였습니다. 연구는 영어, 프랑스어 및 핀란드어의 재사용 항목과 합성어에 대한 대규모 데이터 세트를 사용하여 이 주장을 검증하였습니다.

- **Performance Highlights**: 이 연구의 결과는 역사적으로 나타난 재사용 항목과 합성어가 가상의 방법보다 더 높은 수준의 의사소통 효율성을 달성한다는 것을 보여줍니다. 더불어, 문자 그대로의 재사용 항목과 합성어는 비문자 그대로(non-literal) 항목보다 더 효율적이라는 것을 입증하였습니다.



### Ev2R: Evaluating Evidence Retrieval in Automated Fact-Checking (https://arxiv.org/abs/2411.05375)
Comments:
          10 pages

- **What's New**: 본 논문에서는 Automated Fact-Checking (AFC)을 위한 새로운 평가 프레임워크인 Ev2R을 소개합니다. Ev2R은 증거 평가를 위한 세 가지 접근 방식인 reference-based, proxy-reference, reference-less 평가 방식으로 구성되어 있습니다.

- **Technical Details**: Ev2R 프레임워크의 세 가지 평가자 그룹은 (1) reference-based 평가자: 참조 증거와 비교하여 검색된 증거를 평가, (2) proxy-reference 평가자: 시스템이 예측한 평결을 기준으로 증거를 평가, (3) reference-less 평가자: 입력 클레임만을 기반으로 증거를 평가하는 방식입니다. 구체적으로 LLMs를 활용하여 증거를 원자적 사실로 분해한 후 평가합니다.

- **Performance Highlights**: Ev2R의 reference-based 평가자는 전통적인 메트릭보다 인간 평가와 높은 상관관계를 보였습니다. Gemini 기반 평가자는 검색된 증거가 참조 증거를 얼마나 포함하고 있는지를 잘 평가한 반면, GPT-4 기반 평가자는 평결 일치에서 더 좋은 성능을 보였습니다. 이 결과는 Ev2R 프레임워크가 더욱 정확하고 강력한 증거 평가를 가능하게 함을 시사합니다.



### Dynamic-SUPERB Phase-2: A Collaboratively Expanding Benchmark for Measuring the Capabilities of Spoken Language Models with 180 Tasks (https://arxiv.org/abs/2411.05361)
- **What's New**: 본 논문에서는 다양한 자연어 지시를 이해하는 보편적인 음성 언어 모델 개발을 위한 평가 기준으로 Dynamic-SUPERB Phase-2를 새롭게 제시합니다. 이 기준은 180개의 광범위한 과제를 포함하여 음성과 오디오 평가를 위한 최대 기준을 자랑합니다.

- **Technical Details**: Dynamic-SUPERB Phase-2는 125개의 새로운 과제로 구성되어 있으며, 음성, 음악 및 환경 오디오 등 다양한 작업을 포함합니다. 첫 번째 단계에서 주로 분류 작업만 다루었던 것이 점차적으로 회귀 및 시퀀스 생성 작업을 포함한 폭넓은 평가 기능을 제공합니다.

- **Performance Highlights**: 평가 결과, SALMONN-13B는 영어 ASR에서 우수한 성능을 보였으나, 현재 모델들이 일반화에 실패하고 있는 상황입니다. WavLLM은 감정 인식에서는 높은 정확도를 기록했지만, 여전히 많은 과제를 처리하는 데는 혁신이 필요합니다. 이 연구 결과는 다양한 데이터로 학습하는 것이 도메인 간 성능 향상에 기여할 수 있음을 보여줍니다.



### Reasoning Robustness of LLMs to Adversarial Typographical Errors (https://arxiv.org/abs/2411.05345)
- **What's New**: 본 연구에서는 Chain-of-Thought (CoT) 프롬프트를 사용하는 대형 언어 모델(Large Language Models, LLMs)의 추론 강건성에 대한 연구를 진행했습니다. 특히, 사용자 쿼리에서 발생할 수 있는 오타(typographical errors)의 영향에 주목했습니다. 따라서, Adversarial Typo Attack (ATA) 알고리즘을 설계하여, 쿼리에서 중요한 단어의 오타를 반복적으로 샘플링하고 성공 가능성이 높은 수정을 선택하는 과정을 통해 LLM의 오타에 대한 민감성을 입증하였습니다.

- **Technical Details**: ATA(Adversarial Typo Attack) 알고리즘은 입력에서 중요한 토큰을 추출하고, 각 선택 단어에 대해 타이핑 오류를 샘플링한 후 수정된 입력의 손실을 평가하여 최적 후보를 보존하는 방식으로 작동합니다. 이 과정에서, 모델이 잘못된 답을 생성하도록 유도하는 단순한 타이핑 오류를 포함한 다양한 수정 방식을 적용하였습니다. 연구에서는 R^2ATA 벤치마크를 개발하여 GSM8K, BBH, MMLU의 세 가지 추론 데이터셋을 사용하여 LLM의 추론 강건성을 평가했습니다.

- **Performance Highlights**: 실험 결과, Mistral-7B-Instruct 모델의 경우 단일 문자 수정으로 정확도가 43.7%에서 38.6%로 감소하였고, 8자리 수정으로 성능이 19.2%로 추가 하락했습니다. R2ATA 평가에서 고급 모델들 역시 서로 다른 취약성을 보였으며, 예를 들어 Vicuna-33B-chat의 경우, GSM8K에서 38.2%에서 26.4%로, BBH에서는 52.1%에서 42.5%로, MMLU에서는 59.2%에서 51.5%로 성능이 감소했습니다.



### Improving Multi-Domain Task-Oriented Dialogue System with Offline Reinforcement Learning (https://arxiv.org/abs/2411.05340)
- **What's New**: 이번 논문은 사전 훈련된 GPT-2 모델을 활용하여 사용자 정의 작업을 수행하는 대화형 시스템(TOD)을 제안합니다. 주요 특징은 감독 학습(Supervised Learning)과 강화 학습(Reinforcement Learning)을 결합하여, 성공률(Success Rate)과 BLEU 점수를 기반으로 보상 함수를 최적화하여 모델의 성능을 향상시키는 것입니다.

- **Technical Details**: 제안된 TOD 시스템은 사전 훈련된 대형 언어 모델인 GPT-2를 기반으로 하며, 감독 학습과 강화 학습을 통해 최적화되었습니다. 비가역적 보상 함수를 사용하여 대화 결과의 성공률과 BLEU 점수를 가중 평균하여 보상을 계산합니다. 모델은 사용자 발화, 신념 상태(Belief State), 시스템 액트(System Act), 시스템 응답(System Response)으로 구성된 대화 세션 수준에서 미세 조정(Fine-tuning)됩니다.

- **Performance Highlights**: MultiWOZ2.1 데이터셋에서 실험 결과, 제안한 모델은 기준 모델에 비해 정보 비율(Inform Rate)을 1.60% 증가시키고 성공률(Success Rate)을 3.17% 향상시켰습니다.



### SciDQA: A Deep Reading Comprehension Dataset over Scientific Papers (https://arxiv.org/abs/2411.05338)
Comments:
          18 pages, Accepted to EMNLP 2024

- **What's New**: SciDQA라는 새로운 데이터셋이 소개되었습니다. 이 데이터셋은 자연어 처리(NLP) 분야에서의 과학 논문의 깊은 이해를 요구하는 질의응답(Question-Answering, QA) 작업을 목적으로 만들어졌습니다. 총 2,937개의 QA 쌍으로 구성되어 있으며, 과학 논문에 대한 도메인 전문가의 피어 리뷰에서 질문을 추출하고 논문 저자가 답변을 제공하여 그 질을 담보합니다.

- **Technical Details**: SciDQA는 기계 학습(ML) 도메인에 특화된 과학 논문을 위한 데이터셋입니다. 데이터셋은 OpenReview 플랫폼에서의 피어 리뷰를 통해 수집하며, 긴 형태의 질문과 답변 쌍으로 구성되어 있습니다. 질문은 도표, 표, 방정식, 그리고 보충 자료를 포함하는 내용까지 요구하며, 다문서 추론이 필요합니다. 데이터셋의 구축 과정에서는 주제 전문가에 의한 수작업 주석이 포함되어 있습니다.

- **Performance Highlights**: 여러 오픈소스 및 상용 LLM에 대한 평가 결과, 과학적 텍스트 이해 능력에서 상당한 성능 차이가 나타났습니다. 많은 LLM이 제시된 질문에 대해 정확하고 사실적인 답변을 생성하는 데 어려움을 겪었습니다. 데이터셋과 코드는 GitHub에 공개되어 있으며, 다양한 실험 구성에 대한 성과가 포함되어 있습니다.



### SpecHub: Provable Acceleration to Multi-Draft Speculative Decoding (https://arxiv.org/abs/2411.05289)
Comments:
          EMNLP 2024 (Main)

- **What's New**: 대규모 언어 모델(LLMs)의 인퍼런스 속도를 개선하기 위한 새로운 방법, SpecHub를 소개합니다. 기존 방식인 Recursive Rejection Sampling (RRS)의 한계를 극복하고, 다수의 토큰 초안을 효율적으로 검증하는 혁신적인 기법입니다.

- **Technical Details**: SpecHub는 Multi-Draft Speculative Decoding (MDSD)의 새로운 샘플링-검증 방법으로, Optimal Transport 문제를 간소화하여 선형 프로그래밍 모델(Linear Programming model)로 변환합니다. 이를 통해 계산 복잡성을 입증적으로 줄이고 희소 분포(sparse distribution)를 활용하여 높은 확률의 토큰 시퀀스에 계산을 집중시킵니다.

- **Performance Highlights**: SpecHub는 RRS와 비교하여 매 스텝 당 0.05-0.27, RRS without replacement에 비해 0.02-0.16 더 많은 토큰을 생성하며, 기존 방법에 비해 1-5%의 두 번째 초안 수락률 증가를 기록합니다. 또한, 동일한 배치 효율성을 달성하기 위해 다른 방법의 절반에 해당하는 노드를 가진 트리를 사용합니다.



### Fox-1 Technical Repor (https://arxiv.org/abs/2411.05281)
Comments:
          Base model is available at this https URL and the instruction-tuned version is available at this https URL

- **What's New**: Fox-1은 새로운 훈련 커리큘럼 모델링을 도입한 소형 언어 모델(SLM) 시리즈로, 3조 개의 토큰과 5억 개의 지침 데이터로 사전 훈련 및 미세 조정되었습니다. 이 모델은 그룹화된 쿼리 어텐션(Grouped Query Attention, GQA)과 깊은 레이어 구조를 가지고 있어 성능과 효율성을 향상시킵니다.

- **Technical Details**: Fox-1-1.6B 모델은 훈련 데이터를 3단계 커리큘럼으로 구분하여 진행하며, 2K-8K 시퀀스 길이를 처리합니다. 데이터는 오픈 소스의 다양한 출처에서 수집된 고품질 데이터를 포함하고 있습니다. 이 모델은 텐서 오페라(TensorOpera) AI 플랫폼과 훌리 지픈(Facing)에서 공개되며, Apache 2.0 라이센스 하에 이용 가능합니다.

- **Performance Highlights**: Fox-1은 StableLM-2-1.6B, Gemma-2B와 같은 여러 벤치마크에서 경쟁력 있는 성능을 보여주며 빠른 추론 속도와 처리량을 자랑합니다.



### Seeing Through the Fog: A Cost-Effectiveness Analysis of Hallucination Detection Systems (https://arxiv.org/abs/2411.05270)
Comments:
          18 pags, 13 figures, 2 tables

- **What's New**: 이번 논문은 AI를 위한 hallucination detection 시스템의 비교 분석을 제공합니다. 주로 Large Language Models (LLMs)에 대한 자동 요약 및 질문 응답 작업을 중점적으로 다루고 있습니다.

- **Technical Details**: 다양한 hallucination detection 시스템을 진단 오즈 비율 (diagnostic odds ratio, DOR)과 비용 효율성 (cost-effectiveness) 메트릭스를 사용하여 평가하였습니다. 고급 모델이 더 나은 성능을 보일 수 있지만, 비용이 상당히 증가함을 보여주고 있습니다.

- **Performance Highlights**: 이상적인 hallucination detection 시스템은 다양한 모델 크기에서 성능을 유지해야 함을 입증하였으며, 특정 응용 프로그램 요구 사항과 자원 제약에 맞는 시스템 선택의 중요성을 강조하고 있습니다.



### What talking you?: Translating Code-Mixed Messaging Texts to English (https://arxiv.org/abs/2411.05253)
- **What's New**: 이 연구는 코드 혼합 언어인 Singlish(싱가포르식 영어)를 표준 영어로 번역하는 데 초점을 맞추고 있으며, 코드 혼합 언어의 번역 및 언어 감지를 위한 최근의 LLM(대형 언어 모델) 활용 방안을 조사합니다. 이를 통해 더 넓은 이해를 가능하게 하고 감정 분석과 같은 애플리케이션에 도움을 주고자 합니다.

- **Technical Details**: 이 연구는 Singlish 문장의 언어 감지 및 번역을 위한 다단계 프롬프트 스키마를 설계하였으며, 총 5개의 LLM(Mistral-7B, LLaMA-3.1, Gemma-2, Qwen-2.5, Phi-3.1)을 사용하여 분석했습니다. 특히, LLM을 사용할 때 코드 혼합 특성이 포함된 언어 문장을 처리할 수 있도록 설계되었습니다. 데이터셋은 SMS에서 수집된 300개의 간단한 메시지로 구성되어 있으며, 이 메시지들은 비공식적인 대화를 반영합니다.

- **Performance Highlights**: 연구 결과, LLM들은 코드 혼합 언어에 대한 언어 감지와 번역 작업에서 모두 낮은 성능을 보였습니다. 이는 다국어를 포함한 문장의 번역 과정에서 발생하는 여러 가지 도전 과제를 나타내며, 코드를 혼합한 언어의 번역에서의 어려움을 강조합니다.



### Abstract2Appendix: Academic Reviews Enhance LLM Long-Context Capabilities (https://arxiv.org/abs/2411.05232)
Comments:
          We share our latest dataset on this https URL

- **What's New**: 이 연구는 대형 언어 모델(LLM)의 긴 문맥 처리를 향상시키기 위해 고품질 학술 피어 리뷰 데이터를 활용한 효과성을 탐구합니다. 연구 결과, Direct Preference Optimization (DPO) 방법이 Supervised Fine-Tuning (SFT) 방법에 비해 우수하고 데이터 효율성이 높음을 입증했습니다.

- **Technical Details**: 본 연구에서는 2000개의 PDF로부터 파생된 데이터셋을 사용하여 DPO와 SFT 방식을 비교했습니다. DPO를 통해 모델의 긴 텍스트 이해 능력을 효과적으로 향상시킬 수 있음을 보여주었습니다. DPO 실험에서 모델 성능 개선을 위해 GPT-4를 사용하여 피어 리뷰들을 집계하였고, 피어 리뷰 데이터 사용을 통해 모델의 추가적인 감독 신호를 제공하는 방법을 모색했습니다.

- **Performance Highlights**: DPO로 미세 조정된 phi-3-mini-128k 모델은 phi-3-mini-4B-128k 모델보다 평균 4.04 점 향상된 결과를 보였고, Qasper 벤치마크에서 2.6% 증가를 기록했습니다. 이는 제한된 데이터만으로도 긴 문맥 읽기 능력을 크게 개선할 수 있음을 나타냅니다.



### CHATTER: A Character Attribution Dataset for Narrative Understanding (https://arxiv.org/abs/2411.05227)
Comments:
          submitted to NAACL 2025

- **What's New**: 이 논문은 캐릭터의 특성을 식별하고 설명하며 상호작용하는 데 중점을 둔 컴퓨터 내러티브 이해(computational narrative understanding)에 대한 연구로, 새로운 Chatter 데이터셋을 소개합니다. 이 데이터셋은 2998 캐릭터와 13324 특성, 660 개의 영화에서 88148 캐릭터-특성 쌍을 포괄하며, 캐릭터의 특성을 평가할 수 있는 강력한 벤치마크를 제공합니다.

- **Technical Details**: 논문은 내러티브의 4가지 주요 요소인 캐릭터, 특성, 사건 및 관계를 규명하며, 캐릭터 이해(character understanding) 작업을 다양한 방법으로 운영화하는 접근법을 탐구합니다. Chatter 데이터셋은 TVTropes에서 정의된 캐릭터 트로프를 사용하여 생성되었으며, 특정 캐릭터가 해당 트로프를 구현했는지를 이진 분류 작업으로 모델링합니다. ChatterEval로 불리는 검증 하위 집합은 인간 주석을 사용하여 평가 벤치마크를 제공합니다.

- **Performance Highlights**: 지금까지 수집된 데이터셋은 LLM(대형 언어 모델)의 제로샷 성능을 비교하기 위한 데이터로 사용되어, Chatter 데이터셋이 캐릭터 속성 작업에 대한 효과적인 학습 세트로 적합하다는 것을 평가합니다. 이는 언어 모델의 장기 컨텍스트 모델링 능력을 평가하는 데 중요한 역할을 하고 있습니다.



### Beyond the Numbers: Transparency in Relation Extraction Benchmark Creation and Leaderboards (https://arxiv.org/abs/2411.05224)
Comments:
          This paper was accepted at the GenBench workshop at EMNLP2024

- **What's New**: 이 논문은 자연어 처리(NLP)에서 벤치마크(benchmark)와 리더보드(leaderboard)의 투명성을 조사하며, 특히 관계 추출(relation extraction, RE) 작업에 중점을 둡니다. 기존의 RE 벤치마크는 데이터 출처, 상호 주석자 합의(inter-annotator agreement), 데이터셋 인스턴스 선택을 위한 알고리즘, 데이터셋 불균형과 같은 잠재적 편향에 대한 정보가 부족한 경향이 있습니다.

- **Technical Details**: RE 벤치마크는 보통 F1-score와 같은 집계 메트릭을 기준으로 시스템을 평가하는 리더보드를 사용하지만, 이러한 메트릭을 넘어서 성능을 자세히 분석하지 않으면 모델의 진정한 일반화(generalization) 능력을 가릴 수 있습니다. 이 논문에서는 TACRED와 NYT 데이터셋의 자료를 바탕으로, 관계 추출 분야에서의 투명성이 결여된 상태가 어떻게 모델 평가에 영향을 미치는지를 분석합니다.

- **Performance Highlights**: 이 논문의 분석 결과, 널리 사용되는 RE 벤치마크인 TACRED와 NYT는 매우 불균형하고, 노이즈 레이블(noisy labels)을 포함하고 있는 것으로 나타났습니다. 또한, 클래스 기반 성능 메트릭의 부재는 다양한 관계 타입이 있는 데이터셋에서 모델 성능을 정확하게 반영하지 못함을 보여줍니다. 결국, RE의 진행 상황을 보고할 때 이러한 한계를 고려해야 합니다.



### STAND-Guard: A Small Task-Adaptive Content Moderation Mod (https://arxiv.org/abs/2411.05214)
Comments:
          20 pages, 1 figure

- **What's New**: STAND-GUARD 모델은 소규모 언어 모델(SLMs)을 사용하여 새로운 콘텐츠 조정 작업에 신속하게 적응할 수 있도록 설계되었습니다. 이 모델은 주어진 여러 콘텐츠 조정 작업에서 지시 튜닝(instruct tuning)을 수행하여 더욱 친숙한 콘텐츠 안전성을 보장합니다.

- **Technical Details**: 이 논문은 STAND-GUARD라는 소형 작업 적응 콘텐츠 조정 모델을 소개하며, 이는 새로운 작업에 대해 신속하게 적응할 수 있도록 설계되었습니다. 크로스-테스크 파인튜닝(cross-task fine-tuning)을 통해 소규모 언어 모델이 보이지 않는(out-of-distribution) 콘텐츠 조정 작업에서 효과적으로 작동할 수 있는 가능성을 보여줍니다. 실험 결과 STAND-GUARD는 40개 이상의 공개 데이터 세트에서 GPT-3.5-Turbo와 동등한 성능을 보였습니다.

- **Performance Highlights**: STAND-GUARD는 비공식적인 영어 이항 분류 작업에서 GPT-4-Turbo와 유사한 결과를 달성했습니다. 이 모델은 다양한 데이터 세트에서 OUT-OF-DISTRIBUTION 및 IN-DISTRIBUTION 작업 모두에서 뛰어난 성능을 발휘합니다.



### CodeLutra: Boosting LLM Code Generation via Preference-Guided Refinemen (https://arxiv.org/abs/2411.05199)
Comments:
          18 pages, 4 figures

- **What's New**: CodeLutra는 코드 생성에서 성능이 저조한 LLM(대형 언어 모델)을 개선하기 위한 새로운 프레임워크로, 성공적인 코드와 실패한 코드 생성 시도의 데이터를 활용합니다. 이를 통해 기존의 데이터 및 모델를 최대한 활용하여 성능 차이를 줄이도록 설계되었습니다.

- **Technical Details**: CodeLutra는 전통적인 Fine-tuning 관행과는 달리, 성공적 및 실패한 코드 생성을 비교하고 선호도를 최대화하는 반복적 학습 메커니즘을 채택합니다. 이 과정에서 모델은 올바른 코드를 생성하려는 경향을 강화하고, 코드 품질에 대한 이해를 지속적으로 개선합니다.

- **Performance Highlights**: CodeLutra를 적용한 결과, Llama-3-8B 모델이 76.6%의 실행 정확도를 달성하여 GPT-4의 74.4%를 초과했습니다. 특히, 단 500개의 샘플을 사용하여 정확도가 28.2%에서 48.6%로 향상되며 GPT-4와의 성능 차이가 줄어드는 것을 확인했습니다.



### Explaining Mixtures of Sources in News Articles (https://arxiv.org/abs/2411.05192)
Comments:
          9 pages

- **What's New**: 이 연구에서는 인간 작가의 글쓰기 전 계획 단계를 이해하는 것이 중요하다는 점을 강조하며, 뉴스에서 출처 선택(Source-Selection)을 하나의 사례 연구로 설정하여 장기 형식 기사 생성을 평가합니다.

- **Technical Details**: 연구자는 저널리스트와 협력하여 기존의 다섯 가지 출처 선택 스키마를 수정하고 세 가지 새로운 스키마를 도입하였습니다. 베이지안(Bayesian) 잠재변수 모델링에서 영감을 받아, 스토리의 근본적인 계획(혹은 스키마)을 선택하는 메트릭트를 개발하였습니다. 두 가지 주요 스키마(stance, social affiliation)가 대부분의 문서에서 출처 계획을 잘 설명하는 것으로 나타났습니다.

- **Performance Highlights**: 연구자가 수집한 400만 개의 뉴스 기사의 주석이 추가된 데이터셋인 NewsSources를 रिलीज했습니다. 기사의 제목만으로도 적절한 스키마를 예측할 수 있으며, 90,000개의 뉴스 기사에 대한 분석 결과, 47%의 문장이 출처와 연결될 수 있음을 발견했습니다.



### ImpScore: A Learnable Metric For Quantifying The Implicitness Level of Languag (https://arxiv.org/abs/2411.05172)
- **What's New**: 이번 논문은 자연어 처리(Natural Language Processing, NLP) 시스템에서의 암묵적 언어의 중요성을 강조하며, 이를 정량화할 수 있는 새로운 메트릭인 ImpScore를 도입합니다. 이 메트릭은 외부 기준을 사용하지 않고도 언어의 암묵적 수준을 측정할 수 있는 정량적 도구로써, 기존의 주관적인 평가 기준을 보완합니다.

- **Technical Details**: ImpScore는 전통적인 언어학의 원리를 기반으로 정의된 메트릭으로, 의미(semantic)와 화용(pragmatic) 해석 간의 차이를 통해 암묵성을 측정합니다. 이 메트릭은 112,580개의 (암묵적 문장, 명시적 문장) 쌍을 사용하여 학습된 해석 가능한 회귀 모델을 통해 제안되었으며, 문장 간 암묵적 수준을 비교할 수 있는 추가적인 지표도 포함합니다. ImpScore는 짝 비교 학습(pairwise contrastive learning)을 통해 훈련되었습니다.

- **Performance Highlights**: ImpScore는 사용자 연구를 통해 OOD(Out-of-Distribution) 데이터에서의 성능을 검증하였고, 인간 평가와의 상관성 또한 높은 것으로 나타났습니다. 또한, 증오 발언 탐지(hate speech detection) 데이터셋에 적용한 결과, 현재의 대형 언어 모델들이 매우 암묵적인 내용을 이해하는 데 있어 제한적임을 확인했습니다. 이는 모델의 전반적인 성능은 우수하더라도, 심화된 암묵적 내용의 처리에 있어 병목 현상을 나타냅니다.



### Findings of the IWSLT 2024 Evaluation Campaign (https://arxiv.org/abs/2411.05088)
Comments:
          IWSLT 2024; 59 pages

- **What's New**: 이번 논문은 21회 IWSLT 컨퍼런스에서 조직된 공유 작업에 대해 보고하며, 7가지 음성 언어 번역의 과제를 다루고 있습니다. 다루는 과제는 실시간 및 오프라인 번역, 자동 자막 작성 및 더빙, 음성-음성 번역, 방언 및 저자원 음성 번역, 인도 언어 번역 등입니다.

- **Technical Details**: 18개 팀이 참여하여 26개의 시스템 논문을 제출했으며, 산업과 학계에서 번역 기술에 대한 관심이 꾸준히 증가하고 있음을 나타냅니다. 이번 공유 작업은 실시간 번역(Simultaneous Translation), 오프라인 번역(Offline Translation), 자동 자막 부여(Automatic Subtitling), 음성-음성 번역(Speech-to-Speech Translation), 다이얼렉트(Dialect), 저자원 언어(Low-resource Speech Translation), 인도 언어(Indic Languages) 등 다양한 주제를 포함합니다.

- **Performance Highlights**: 이번 컨퍼런스에서의 참여자 수의 증가와 다양한 설정에서의 도전 과제들은 음성 언어 번역 분야의 성장 가능성을 보여주고 있으며, 산업과 학계 간의 활발한 협력이 이루어지고 있음을 증명합니다.



### FineTuneBench: How well do commercial fine-tuning APIs infuse knowledge into LLMs? (https://arxiv.org/abs/2411.05059)
- **What's New**: 이 연구에서는 FineTuneBench라는 새로운 평가 프레임워크 및 데이터셋을 소개하여 상업적 LLM(대형 언어 모델) 미세 조정 API가 새로운 정보와 업데이트된 지식을 성공적으로 학습할 수 있는지를 분석합니다.

- **Technical Details**: FineTuneBench 데이터셋은 뉴스, 허구의 인물, 의료 지침, 코드의 네 가지 영역에서 625개의 학습 질문과 1,075개의 테스트 질문으로 구성되어 있습니다. 연구진은 GPT-4o, GPT-3.5 Turbo, Gemini 1.5 Pro 등의 LLM 5개를 활용하여 미세 조정 서비스를 평가했습니다. 이 연구는 새로운 정보를 학습하는 LLM의 능력에 대해 체계적인 평가를 제공합니다.

- **Performance Highlights**: 모델들의 평균 일반화 정확도는 37%로, 새로운 정보를 효과적으로 학습하는 데에는 상당한 한계가 있음을 보여주었습니다. 기존 지식을 업데이트할 때는 평균 19%의 정확도만을 기록하며, 특히 Gemini 1.5 시리즈는 새로운 지식을 학습하거나 기존 지식을 업데이트하는 데에 실패했습니다. GPT-4o mini가 새로운 지식을 주입하고 지식을 업데이트하는 데 가장 효과적인 모델로 판단되었습니다.



### A Brief History of Named Entity Recognition (https://arxiv.org/abs/2411.05057)
Comments:
          Survey done in 2020

- **What's New**: 최근 NER(Named Entity Recognition) 분야의 발전과 함께, 딥 러닝 기반의 NER 시스템의 성능이 크게 향상되었습니다. 이 연구에서는 NER 기술의 진화 과정을 조사하며, 감독 학습(supervised learning)에서 비감독 학습(unsupervised learning) 방법으로 변화하는 과정과 이들 시스템의 성능을 비교합니다.

- **Technical Details**: NER 과정은 이름 있는 개체 인식(NER), 이름 있는 개체의 애매성 제거(NED), 이름 있는 개체 연결(NEL)로 세 단계로 나눌 수 있습니다. 딥 러닝 모델은 주로 Recurrent Neural Networks (RNNs)를 이용하여 새로운 데이터에 대한 일반화 성능이 향상되었습니다. 최근에서는 액티브 러닝(active learning), 반감독 학습(semi-supervised learning), 비감독 학습(unsupervised learning) 기술 또한 탐구되고 있습니다.

- **Performance Highlights**: 다양한 데이터 세트와 평가 지표를 통해 NER의 성능을 비교 분석하였고, CoNLL-2002 및 CoNLL-2003 데이터 세트에서 유의미한 결과를 도출했습니다. 비감독 학습 방법이 딥 러닝 모델과 유사한 수준의 성과를 내는 것으로 나타났으며, 특정 NER 툴이 광범위하게 사용되고 있음이 강조되었습니다.



### FMEA Builder: Expert Guided Text Generation for Equipment Maintenanc (https://arxiv.org/abs/2411.05054)
Comments:
          4 pages, 2 figures. AI for Critical Infrastructure Workshop @ IJCAI 2024

- **What's New**: 이 논문에서는 산업 장비와 관련된 구조화된 문서, 특히 Failure Mode and Effects Analysis (FMEA) 생성에 대한 새로운 AI 시스템을 제안합니다. 이 시스템은 대규모 언어 모델을 활용하여 FMEA 문서를 신속하고 전문가가 감독하는 방식으로 생성할 수 있도록 지원합니다.

- **Technical Details**: 제안된 시스템의 주요 기술적 세부 사항은 단편화된 생성 문제를 해결하기 위해 동적 예시 선택을 통한 Prompting(프롬프트) 접근 방식을 사용하고, LLMs(대규모 언어 모델)의 답변 일관성, 컨텍스트 학습 및 동적 관련 예시 선택 같은 기술을 적용하는 것입니다. 이 시스템은 주제 전문가의 지식을 활용하여 FMEA 문서의 각 섹션을 생성할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, DFSP(Dynamic Few Shot Prompting) 방법이 FMEA의 주요 구성 요소인 장비 경계와 실패 위치를 생성하는 데 있어 가장 높은 품질을 보였으며, LLMs의 성능이 증가하는 경향을 보였습니다. 전문가의 감독 하에 생성된 결과물은 신뢰성 엔지니어에게 중요한 역할을 하며, 전체적인 사용자 피드백은 긍정적이었습니다.



### Selecting Between BERT and GPT for Text Classification in Political Science Research (https://arxiv.org/abs/2411.05050)
Comments:
          28 pages, 5 figures, 7 tables

- **What's New**: 이 연구는 GPT 기반 모델과 프롬프트 엔지니어링을 활용하여 데이터가 부족한 상황에서 텍스트 분류 작업에 대한 새로운 대안을 제시합니다. 기존에 일반적으로 사용되던 BERT 모델과의 성능 비교를 통해 GPT 모델 사용의 잠재력을 탐구합니다.

- **Technical Details**: 이 연구에서는 다양한 분류 작업에 걸쳐 BERT 기반 모델과 GPT 기반 모델의 성능을 평가하기 위한 일련의 실험을 수행하였습니다. 실험은 데이터 샘플 수가 적고 복잡성이 다른 분류 작업을 포함하며, 특히 1,000개 이하의 샘플에서 GPT 모델의 제로샷(zero-shot) 및 몇 샷(few-shot) 학습이 BERT 모델과 비교되었음을 강조합니다.

- **Performance Highlights**: 결과에 따르면, GPT 모델을 이용한 제로샷 및 몇 샷 학습은 초기 연구 탐색에 적합하나, 일반적으로 BERT의 파인 튜닝과 동등하거나 부족한 성능을 보였습니다. BERT 모델은 높은 훈련 세트에 도달할 때 더 우수한 성능을 발휘하는 것으로 나타났습니다.



### ProverbEval: Exploring LLM Evaluation Challenges for Low-resource Language Understanding (https://arxiv.org/abs/2411.05049)
- **What's New**: 저자는 저자들은 LLM(대형 언어 모델) 평가의 새로운 벤치마크인 ProverbEval을 소개합니다. 이는 저자 원어의 속담을 기반으로 하여 저자 자원 부족 언어에 대한 이해를 평가하는데 중점을 둡니다.

- **Technical Details**: ProverbEval은 다양한 LLM을 벤치마킹하며, 다중 선택 과제에서 답변 선택의 제시 순서에 따라 최대 50%의 성능 변화를 관찰했습니다. 원어 속담 설명이 속담 생성과 같은 작업을 개선하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 모노링궐(monolingual) 평가가 크로스링궐(cross-lingual) 평가보다 일관되게 높은 성능을 보이며, LLM 평가 벤치마크를 설정할 때 선택지의 순서, 프롬프트 언어의 선택, 작업의 다양성 및 생성 작업에 특별한 주의를 기울여야 한다고 주장합니다.



### Leveraging LLMs to Enable Natural Language Search on Go-to-market Platforms (https://arxiv.org/abs/2411.05048)
Comments:
          11 pages, 5 figures

- **What's New**: 이 논문은 Zoominfo 제품을 위한 자연어 쿼리(queries) 처리 솔루션을 제안하고, 대규모 언어 모델(LLMs)을 활용하여 검색 필드를 생성하고 이를 쿼리로 변환하는 방법을 평가합니다. 이 과정은 복잡한 메타데이터를 요구하는 기존의 고급 검색 방식의 단점을 극복하려는 시도입니다.

- **Technical Details**: 제안된 솔루션은 LLM을 사용하여 자연어 쿼리를 처리하며, 중간 검색 필드를 생성하고 이를 JSON 형식으로 변환하여 최종 검색 서비스를 위한 쿼리로 전환합니다. 이 과정에서는 여러 가지 고급 프롬프트 엔지니어링 기법(techniques)과 체계적 메시지(system message),few-shot prompting, 체인 오브 씽킹(chain-of-thought reasoning) 기법을 활용했습니다.

- **Performance Highlights**: 가장 정확한 모델은 Anthropic의 Claude 3.5 Sonnet으로, 쿼리당 평균 정확도(accuracy) 97%를 기록했습니다. 부가적으로, 작은 LLM 모델들 또한 유사한 결과를 보여 주었으며, 특히 Llama3-8B-Instruct 모델은 감독된 미세조정(supervised fine-tuning) 과정을 통해 성능을 향상시켰습니다.



### PhoneLM:an Efficient and Capable Small Language Model Family through Principled Pre-training (https://arxiv.org/abs/2411.05046)
- **What's New**: 이 연구는 작은 언어 모델(SLM)의 설계에서 하드웨어 특성을 고려하지 않던 기존의 방법에서 벗어나, 사전 학습(pre-training) 전에 최적의 런타임 효율성을 추구하는 새로운 원칙을 제시합니다. 이를 바탕으로 PhoneLM이라는 SLM 패밀리를 개발하였으며, 현재 0.5B 및 1.5B 변종이 포함되어 있습니다.

- **Technical Details**: PhoneLM은 스마트폰 하드웨어(예: Qualcomm Snapdragon SoC)를 위해 설계된 사전 학습 및 지시 모델로, 다양한 모델 변형을 포함합니다. SLM 아키텍처는 다층 구조와 피드 포워드 네트워크의 활성화 함수(activation function)와 같은 하이퍼파라미터를 포함하며, 특히 ReLU 활성화를 사용합니다. 연구에서는 100M 및 200M 파라미터 모델을 가지고 다양한 설정에서 런타임 속도(inference speed)와 손실(loss)을 비교했습니다.

- **Performance Highlights**: PhoneLM은 1.5B 모델이 Xiaomi 14에서 58 tokens/second의 속도로 실행되어 대안 모델보다 1.2배 빠르며, 654 tokens/second의 속도를 NPU에서 달성합니다. 또한, 7개의 대표 벤치마크에서 평균 67.3%의 정확도를 기록하며, 기존 비공식 데이터셋에서 훈련된 SLM보다 더 나은 언어 능력을 보입니다. 이 모든 자료는 완전 공개되어 있어 재현 가능성과 투명성을 제공합니다.



### Performance-Guided LLM Knowledge Distillation for Efficient Text Classification at Sca (https://arxiv.org/abs/2411.05045)
Comments:
          Published in EMNLP 2024

- **What's New**: 이 논문은 성능 중심의 지식 증류(Performance-Guided Knowledge Distillation, PGKD)라는 새로운 방식을 소개하며, 이는 대형 언어 모델(Large Language Models, LLMs)의 지식을 보다 작고 효율적인 모델로 전달하기 위한 방법이다. PGKD는 LLM과 학생 모델 간의 능동 학습 루틴을 수립하여, LLM이 새로운 훈련 데이터를 지속적으로 생성하게 한다.

- **Technical Details**: PGKD는 훈련 중 학생 모델의 성과를 기반으로 하여 LLM이 훈련 데이터를 동적으로 생성하는 과정을 포함한다. 이 방법은 향후 다중 클래스 분류 문제를 해결하기 위해 설계되었으며, 주간 평가 체크(Gradual Evaluation Checks)를 통해 학생 모델의 성과를 지속적으로 모니터링하고 최적화 방향을 제시한다.

- **Performance Highlights**: PGKD를 이용해 미세 조정된 모델은 기존의 BERT-base 모델과 다른 전통적 지식 증류 방법보다 다중 클래스 분류 데이터세트에서 성능이 우수하였으며, 추론 속도는 최대 130배 빠르고 비용은 25배 저렴한 것으로 나타났다.



### Improving Radiology Report Conciseness and Structure via Local Large Language Models (https://arxiv.org/abs/2411.05042)
- **What's New**: 본 연구에서는 방사선 보고서의 간결성과 구조적 조직을 개선하여 진단 결과를 더 효과적으로 전달하는 방법을 제시합니다. 특히, 해부학적 영역에 따라 정보를 정리하는 템플릿 방식을 통해 의사들이 관련 정보를 빠르게 찾을 수 있도록 합니다.

- **Technical Details**: 대규모 언어 모델(LLMs)인 Mixtral, Mistral, Llama를 활용하여 간결하고 잘 정리된 보고서를 생성합니다. 특히, Mixtral 모델이 다른 모델에 비해 특정 포맷 요구사항을 잘 준수하므로 주로 이에 초점을 맞추었습니다. LangChain 프레임워크를 활용해 다섯 가지 다양한 프롬프트(Prompting) 전략을 적용하여 방사선 보고서의 일관된 구조를 유지합니다.

- **Performance Highlights**: 새로운 메트릭인 간결성 비율(Conciseness Percentage, CP) 점수를 도입하여 보고서의 간결성을 평가합니다. 814개의 방사선 보고서를 분석한 결과, LLM이 먼저 보고서를 압축한 다음 특정 지침에 따라 내용을 구조화하도록 지시하는 방식이 가장 효과적이라는 것을 발견했습니다. 우리의 연구는 오픈소스이고 로컬에서 배포된 LLM이 방사선 보고서의 간결성과 구조를 상당히 개선할 수 있음을 보여줍니다.



### Bottom-Up and Top-Down Analysis of Values, Agendas, and Observations in Corpora and LLMs (https://arxiv.org/abs/2411.05040)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)이 생성하는 텍스트에서 사회문화적 가치들을 추출하고 평가하기 위한 자동화된 접근 방식을 제시합니다. 특히, 다양한 관점과 가치의 충돌과 공명을 자동으로 분석하여 다원적 가치 정렬(pluralistic value alignment)을 이해하고 관리하는 것을 목표로 합니다.

- **Technical Details**: 연구팀은 상향식(bottom-up) 및 하향식(top-down) 접근 방식을 결합하여 LLM의 출력에서 다수의 가치를 식별하고 측정하는 새로운 방법론을 제안합니다. 상향식 분석은 텍스트로부터 이질적인 가치 주장을 추출하고, 하향식 분석은 기존의 가치 목록을 이용하여 LLM의 출력을 평가하는 방식입니다. 이를 통해 가치의 정렬 및 다원성을 분석하는 네트워크를 구축했습니다.

- **Performance Highlights**: 하향식 가치 분석에서는 F1 점수 0.97의 높은 정확도를 기록하였으며, 상향식 가치 추출은 인간 주석자와 유사한 성능을 보였습니다. 이를 통해 연구팀은 LLM의 출력에서 나타나는 가치의 일관성과 충돌을 효과적으로 평가할 수 있음을 입증하였습니다.



### YouTube Comments Decoded: Leveraging LLMs for Low Resource Language Classification (https://arxiv.org/abs/2411.05039)
Comments:
          Accepted at FIRE 2024 (Track: Sarcasm Identification of Dravidian Languages Tamil & Malayalam (DravidianCodeMix))

- **What's New**: 본 연구는 Tamil-English 및 Malayalam-English 언어 쌍에서 코드 혼합(text blending)된 세팅에서의 풍자(sarcasm) 및 감정(sentiment) 분석을 위한 새로운 gold standard corpus를 소개합니다. 이는 NLP(Natural Language Processing) 분야에 중요한 발전을 제공합니다.

- **Technical Details**: 연구에서는 GPT-3.5 Turbo 모델을 활용하여 코드 혼합된 텍스트의 풍자 및 감정 감지를 수행합니다. 각 텍스트는 감정 극성(sentiment polarity)에 따라 주석이 달리며, 라벨 불균형(class imbalance) 문제를 해결하기 위해 데이터셋을 구성합니다.

- **Performance Highlights**: Tamil 언어에서 매크로 F1 점수 0.61을 기록하며 9위에 올랐고, Malayalam 언어에서는 0.50 점수로 13위를 기록하였습니다. 이는 Tamil의 풍자가 더 많은 특성을 반영하고 있었음을 보여줍니다.



### Towards Interpreting Language Models: A Case Study in Multi-Hop Reasoning (https://arxiv.org/abs/2411.05037)
Comments:
          University of Chicago, Computer Science, Master of Science Dissertation

- **What's New**: 이 논문에서는 다중 추론(multi-hop reasoning) 질문에 대한 모델의 성능을 개선하기 위해 특정 메모리 주입(memory injections) 메커니즘을 제안합니다. 이 방법은 언어 모델의 주의(attention) 헤드에 대한 목표 지향적인 수정을 제공하여 그들의 추론 오류를 식별하고 수정하는 것을 목표로 합니다.

- **Technical Details**: 제안된 방법은 먼저 GPT-2 모델의 단일 및 다중 추론 프롬프트에 대한 층(layer)별 활성화를 분석하여, 추론 과정에서 언어 모델에 관련된 프롬프트 específicos 정보(정보의 종류를 '기억'이라고 명명)를 주입할 수 있는 메커니즘을 제공합니다. 주의 헤드에서 적절히 위치에서 정보를 주입함으로써 다중 추론 작업에서 성공적인 다음 토큰의 확률을 최대 424%까지 높일 수 있음을 보였습니다. 이를 통해 'Attention Lens'라는 도구를 개발하여 주의 헤드의 출력을 사람에게 이해할 수 있는 형식으로 해석합니다.

- **Performance Highlights**: 이 방법을 통해 모델의 다중 추론 성능이 상당히 향상되었습니다. 메모리 주입을 통해 모델의 예측이 개선되었으며, 424%까지 성공적인 다음 토큰의 확률이 향상되었습니다. 이런 성과는 특정 주의 헤드의 작은 부분 집합이 모델의 예측에 상당한 영향을 미칠 수 있음을 나타냅니다.



### From Word Vectors to Multimodal Embeddings: Techniques, Applications, and Future Directions For Large Language Models (https://arxiv.org/abs/2411.05036)
Comments:
          21 pages

- **What's New**: 이 논문은 자연어 처리(NLP)에서 단어 임베딩(Word Embedding)과 언어 모델의 발전에 대해 논의하며, sparse representation에서 dense embeddings로의 전환을 다룹니다. 최신 모델인 ELMo, BERT, GPT 등을 통해 문맥에 따른 단어의 의미를 반영하는 발전을 강조합니다.

- **Technical Details**: 논문은 distributional hypothesis와 contextual similarity 같은 기초 개념을 설명하며, one-hot encoding에서 Word2Vec, GloVe, fastText 등으로의 발전을 다룹니다. 또한 문서 임베딩과 생성 주제 모델에 대해서도 논의하여, 비전(vision), 로보틱스(robotics), 인지 과학(cognitive science) 같은 다양한 분야에서의 활용을 소개합니다.

- **Performance Highlights**: 이 논문은 단어 임베딩 모델들이 성능을 개선하기 위해 문맥 정보를 통합하는 방법을 분석하고, 향후 연구 방향으로 스케일러블 훈련 기술, 향상된 해석 가능성 및 비텍스트 모달리티(non-textual modalities)에 대한 강력한 기반 마련의 필요성을 강조합니다.



### On-Device Emoji Classifier Trained with GPT-based Data Augmentation for a Mobile Keyboard (https://arxiv.org/abs/2411.05031)
Comments:
          8 pages

- **What's New**: 이번 연구에서는 스마트폰 사용자를 위한 온디바이스(온 기기) 이모지 분류기를 제안합니다. 이 시스템은 MobileBert를 기반으로 하며, 낮은 메모리와 지연 시간 제약을 고려하여 SwiftKey용으로 최적화되었습니다.

- **Technical Details**: 이모지 분류기는 비대칭 데이터셋 문제를 해결하기 위해 GPT를 사용하여 각 이모지 클래스에 대한 태그를 생성하고, 이를 통해 인공적으로 문장을 만들어 이모지 라벨을 할당합니다. 이를 통해 데이터를 보강하고, 사용자 이력을 고려하여 출력 이모지의 우선순위를 재조정하여 이모지 예측 정확도를 높입니다.

- **Performance Highlights**: 제안된 이모지 분류기는 드문 이모지와 이모지 참여도를 특히 향상시키며, 사용자 맞춤형 예측 결과를 제공합니다. 실제 사용자 테스트 결과, 이 클래스화 기술이 효과적으로 작용함을 확인했습니다.



### Deep Learning and Machine Learning -- Natural Language Processing: From Theory to Application (https://arxiv.org/abs/2411.05026)
Comments:
          255 pages

- **What's New**: 본 논문은 자연어 처리(NLP)와 대규모 언어 모델(LLMs)의 역할에 중점을 두고 기계 학습, 심층 학습 및 인공지능의 교차점을 탐구합니다. NLP는 의료부터 금융에 이르기까지 다양한 분야에서 AI 혁신을 주도하고 있으며, 토큰화(Tokenization), 텍스트 분류(Text Classification), 개체 인식(Entity Recognition)과 같은 기술들이 필수적입니다.

- **Technical Details**: 이 논문에서는 고급 데이터 전처리(Data Preprocessing) 기술과 변환기 기반 모델을 구현하기 위한 Hugging Face와 같은 프레임워크의 사용을 다루며, 다국어 데이터 처리, 편향(Bias) 감소, 모델 강건성(Model Robustness) 확보와 같은 주요 도전 과제를 강조합니다. 또한, 데이터 처리 및 모델 미세 조정(Model Fine-tuning)의 주요 측면을 다루며 효과적이고 윤리적으로 건전한 AI 솔루션 배포에 대한 통찰력을 제공합니다.

- **Performance Highlights**: NLP는 헬스케어, 금융, 법률, 교육, 고객 서비스, 미디어 및 엔터테인먼트, 전자상거래, 정부 및 공공 정책, 과학 연구, 사회적 선 및 인도적 노력 등 다양한 산업에서 혁신적인 역할을 하고 있습니다. 예를 들어, NLP는 전자 건강 기록(EHR)에서 중요한 환자 정보를 추출하여 임상 의사 결정 지원 시스템을 개선하고, 고객 피드백을 분석하여 마케팅 전략을 가이드하며, 계약 분석 및 컴플라이언스 모니터링에서 법률 문제를 자동화하는 데 널리 사용됩니다.



### LLMs as Research Tools: A Large Scale Survey of Researchers' Usage and Perceptions (https://arxiv.org/abs/2411.05025)
Comments:
          30 pages, 5 figures

- **What's New**: 본 연구는 816명의 승인된 연구 논문 저자들을 대상으로 한 대규모 조사를 통해 연구 커뮤니티가 대형 언어 모델(LLMs)을 연구 도구로서 어떻게 활용하는지와 이를 어떻게 인식하는지를 분석했습니다.

- **Technical Details**: 응답자들은 LLMs를 정보 탐색, 편집, 아이디어 구상, 직접 작성, 데이터 정리 및 분석, 데이터 생성 등의 다양한 작업에 활용하고 있다고 보고했습니다. 전반적으로 81%의 연구자들이 자신의 연구 작업에 LLMs를 사용하고 있으며, 특히 비백인, 비영어 원어민, 주니어 연구자들이 더 많이 사용하고 높은 이점을 보고했습니다.

- **Performance Highlights**: LLMs를 사용하고 있는 연구자들은 정보 탐색과 편집의 작업에서 가장 빈번하게 사용하고 있으며, 데이터 분석 및 생성에서 상대적으로 적게 사용하고 있습니다. 여성과 비바이너리 연구자들은 더 큰 윤리적 우려를 표명했으며, 전체적으로 연구자들은 영리 기관보다 비영리 또는 오픈소스 모델을 더 선호했습니다.



### Multimodal Quantum Natural Language Processing: A Novel Framework for using Quantum Methods to Analyse Real Data (https://arxiv.org/abs/2411.05023)
Comments:
          This thesis, awarded a distinction by the Department of Computer Science at University College London, was successfully defended by the author in September 2024 in partial fulfillment of the requirements for an MSc in Emerging Digital Technologies

- **What's New**: 이 논문은 양자 컴퓨팅을 언어의 구성(compositionality) 모델링에 적용하는 데 중심을 두고 있으며, 특히 이미지와 텍스트 데이터를 통합하여 다중 양자 자연어 처리(Multimodal Quantum Natural Language Processing, MQNLP) 영역을 발전시키려는 시도를 다룹니다.

- **Technical Details**: 이 연구는 Lambeq 툴킷을 활용하여 이미지-텍스트 양자 회로를 설계하고, 구문 기반 모델(DisCoCat 및 TreeReader)과 비구문 기반 모델의 성능을 평가하는 비교 분석을 포함합니다. 연구는 구문 구조를 이해하고 설명하는 데 초점을 두며, 군집 모델, bag-of-words 모델, 그리고 순차 모델의 성능을 비교합니다.

- **Performance Highlights**: 구문 기반 모델이 특히 우수한 성과를 보였으며, 이는 언어의 구성 구조를 고려했기 때문입니다. 본 연구의 결과는 양자 컴퓨팅 방법이 언어 모델링의 효율성을 높이고 향후 기술 발전을 이끄는 잠재력을 가지고 있음을 강조합니다.



### LLMs as Method Actors: A Model for Prompt Engineering and Architectur (https://arxiv.org/abs/2411.05778)
- **What's New**: 이 논문은 LLM (Large Language Model) 프롬프트 엔지니어링 및 아키텍처 설계를 위한 정신 모델로 'Method Actors'를 도입합니다. 이 모델에서는 LLM을 배우고, 프롬프트를 대본과 신호로, LLM의 응답을 공연으로 간주합니다.

- **Technical Details**: 새로운 접근 방식인 'Method Actors'는 Connections라는 NYT 단어 퍼즐 게임에서 LLM의 성능을 향상시키기 위해 적용됩니다. 기본 접근 방식(vanilla)에서 27%의 퍼즐을 해결하고, Chain of Thoughts 접근 방식에서는 41%를 해결하는 반면, 'Method Actor' 접근 방식은 86%의 퍼즐을 해결합니다. 또한 OpenAI의 새로운 모델인 o1-preview는 다중 API 호출을 통해 100%의 퍼즐을 해결할 수 있습니다.

- **Performance Highlights**: GPT-4o를 사용한 초기 실험에서, 'Method Actor' 접근 방식은 78%의 퍼즐을 해결하고 41%를 완벽하게 해결합니다. 수정된 접근 방식은 86%의 퍼즐을 해결하고 50%를 완벽하게 해결했습니다. o1-preview의 경우, 'Method Actor' 프롬프트 구조를 적용하면 완벽하게 해결하는 퍼즐 비율이 76%에서 87%로 증가했습니다.



### End-to-End Navigation with Vision Language Models: Transforming Spatial Reasoning into Question-Answering (https://arxiv.org/abs/2411.05755)
- **What's New**: VLMnav는 Vision-Language Model(VLM)을 활용하여 엔드 투 엔드 기억 내비게이션 정책으로 전환할 수 있는 새로운 프레임워크를 제안합니다. 기존의 분리된 접근 방식 대신 VLM을 사용하여 직접 행동을 선택하며, 특정 탐색 데이터 없이도 제로샷(zero-shot) 방식으로 정책을 구현할 수 있다는 점이 혁신적입니다.

- **Technical Details**: VLMnav는 목표를 언어나 이미지로 입력 받고, RGB-D 이미지와 자세(pose) 정보를 기반으로 로봇의 회전 및 이동 명령을 출력합니다. 이 시스템은 탐색 문제를 VLM이 잘 수행하는 질문 응답 형식으로 변환하여, 탐색과 장애물 회피를 명확하게 이해하도록 설계된 새로운 프롬프트 전략을 사용합니다. 2D voxel mapping을 통해 탐사된 영역과 탐사되지 않은 영역을 구분하여, 충돌 방지 및 탐색을 최적화합니다.

- **Performance Highlights**: VLMnav는 기존의 프롬프트 방식 대비 더 나은 내비게이션 성능을 보였으며, 탐색 퍼포먼스 평가에서 통계적으로 유의미한 결과를 창출했습니다. 다양한 구성 요소에 대한 분해 실험을 통해 설계 결정의 영향을 분석하고, 전반적으로 향상된 내비게이션 효과를 입증했습니다.



### FisherMask: Enhancing Neural Network Labeling Efficiency in Image Classification Using Fisher Information (https://arxiv.org/abs/2411.05752)
- **What's New**: 이 논문에서는 Fisher 정보에 기반한 액티브 러닝 방법인 FisherMask를 제안합니다. FisherMask는 네트워크의 필수 파라미터를 식별하기 위해 Fisher 정보를 활용하여 효과적으로 중요한 샘플을 선택합니다. 이 방법은 방대한 레이블된 데이터에 대한 의존성을 줄이면서도 모델 성능을 유지할 수 있는 전략을 제공합니다.

- **Technical Details**: FisherMask는 Fisher 정보를 이용하여 고유한 네트워크 마스크를 구축하고, 이 마스크는 가장 높은 Fisher 정보 값을 가진 k개의 가중치를 선택하여 형성됩니다. 또한, 성능 평가는 CIFAR-10 및 FashionMNIST와 같은 다양한 데이터셋에서 수행되었으며, 특히 불균형 데이터셋에서 유의미한 성능 향상이 나타났습니다.

- **Performance Highlights**: FisherMask는 기존 최첨단 방법들보다 뛰어난 성능을 보이며, 레이블링 효율성을 크게 향상시킵니다. 이 방법은 모든 데이터셋에서 테스트 되었으며, 액티브 러닝 파이프라인에서 모델의 성능 특성을 더 잘 이해할 수 있는 유용한 통찰력을 제공합니다.



### Aioli: A Unified Optimization Framework for Language Model Data Mixing (https://arxiv.org/abs/2411.05735)
- **What's New**: 이 논문에서는 언어 모델의 성능이 데이터 그룹의 최적 혼합 비율을 파악하는 데 의존한다는 점을 강조하며, 다양한 방법의 효율성을 비교했습니다. 특히, 기존 방법들이 단순한 stratified sampling 기준선보다 높은 평균 테스트 perplexity 성능을 보이지 못한 점을 발견했습니다.

- **Technical Details**: 저자들은 Linear Mixing Optimization (LMO)이라는 통합 최적화 프레임워크를 제안하였으며, 이를 통해 다양한 혼합 방법들의 기본 가정을 분석했습니다. 혼합 법칙(mixing law)의 매개변수 조정에서의 오류가 기존 방법들의 성능 불일치를 초래한다는 것을 입증했습니다.

- **Performance Highlights**: 새롭게 제안된 온라인 데이터 혼합 방법 Aioli는 6개의 데이터셋에서 모든 경우에 걸쳐 본 논문의 기존 방법보다 평균 0.28 포인트의 테스트 perplexity 향상을 보였습니다. 또한, Aioli는 계산자원이 제한된 상황에서도 혼합 비율을 동적으로 조정하여 기존 방법들보다 최대 12.01 포인트 이상 개선된 성능을 보여줍니다.



### Image2Text2Image: A Novel Framework for Label-Free Evaluation of Image-to-Text Generation with Text-to-Image Diffusion Models (https://arxiv.org/abs/2411.05706)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2408.01723

- **What's New**: 자동 생성된 이미지 설명의 품질을 평가하기 위한 새로운 프레임워크인 Image2Text2Image를 제안합니다. 이 프레임워크는 텍스트-이미지 생성 시스템인 Stable Diffusion이나 DALL-E를 활용하여 이미지 캡셔닝 모델을 평가합니다.

- **Technical Details**: 제안된 프레임워크는 입력 이미지를 선택한 이미지 캡셔닝 모델로 처리하여 텍스트 설명을 생성하고, 이를 통해 생성된 설명으로 새로운 이미지를 생성합니다. 이 과정에서 원본 이미지와 생성된 이미지의 특징을 추출하여 코사인 유사도(Cosine Similarity) 메트릭을 사용하여 유사성을 측정합니다. 고유의 코사인 유사도 점수는 생성된 설명이 원본 이미지를 얼마나 잘 표현했는지를 나타냅니다.

- **Performance Highlights**: 이 프레임워크는 기존의 사람 주석이 필요 없는 자동화된 평가 방법을 통해 이미지 캡셔닝 모델을 효과적으로 평가할 수 있음을 보여줍니다. 인상적인 실험 결과와 인간 평가를 통해 제안된 프레임워크의 효능이 확인되었습니다.



### An Early FIRST Reproduction and Improvements to Single-Token Decoding for Fast Listwise Reranking (https://arxiv.org/abs/2411.05508)
- **What's New**: FIRST는 listwise reranking(리스트와이즈 리랭킹) 문제를 해결하기 위해 learning-to-rank(학습 기반 순위) 목표를 포함하고 첫 번째 생성된 토큰의 logits(로짓)만을 활용하여 지연 시간을 효과적으로 줄이는 새로운 접근법입니다.

- **Technical Details**: 기존의 LLM reranker(대형 언어 모델 리랭커)는 전체 문서 식별자의 순열을 생성해야 하는 반면, FIRST는 첫 번째 토큰의 logits을 바탕으로 재정렬을 수행합니다. 이 접근은 inference latency(추론 지연)을 21%-42% 줄이며, 다양한 backbone models(백본 모델)에 적용하여 효과성을 높였습니다.

- **Performance Highlights**: FIRST는 기존 LLM reranker 보다 효율적이면서도 효과적인 작업을 수행하며, 21%-42%의 지연 시간 개선을 기록했습니다. 실험 결과, single-token logits(단일 토큰 로짓)만을 사용하더라도 out-of-domain(도메인 외) 리랭킹 품질은 유지되며, 기존 LM training(언어 모델 훈련)이 zero-shot single-token reranking(제로샷 단일 토큰 리랭킹) 능력을 향상시키는 가운데, LM pre-training(언어 모델 사전 훈련)이 FIRST 목표의 fine-tuning(미세 조정)을 저해할 수 있음을 발견했습니다.



### EUREKHA: Enhancing User Representation for Key Hackers Identification in Underground Forums (https://arxiv.org/abs/2411.05479)
Comments:
          Accepted at IEEE Trustcom 2024

- **What's New**: 이 논문은 사이버 범죄 활동을 위한 하위 포럼에서 주요 해커를 식별하는 새로운 방법인 EUREKHA를 제안합니다. EUREKHA는 사용자를 텍스트 시퀀스로 모델링하고, 이를 통해 해커 식별의 정확성을 크게 향상시킵니다.

- **Technical Details**: EUREKHA는 사용자의 메타데이터, 스레드 및 응답을 통합하여 텍스트 시퀀스로 변환한 후, 대형 언어 모델(LLM)을 사용하여 도메인 적응과 특징 추출을 수행합니다. 추출된 특징은 그래프 신경망(GNN)에 입력되어 사용자 구조 관계를 모델링합니다. 또한, BERTopic을 사용하여 사용자 생성 콘텐츠에서 개인화된 주제를 추출합니다.

- **Performance Highlights**: EUREKHA는 기존 방법들에 비해 약 6%의 정확도와 10%의 F1 점수 향상을 달성하였으며, 해커 포럼 데이터 세트에서 테스트되었습니다. 또한, 코드를 오픈 소스로 제공하여 연구 커뮤니티의 접근성을 높였습니다.



### WorkflowLLM: Enhancing Workflow Orchestration Capability of Large Language Models (https://arxiv.org/abs/2411.05451)
- **What's New**: 본 논문은 WorkflowLLM이라는 데이터 중심의 프레임워크를 제안하여 LLMs의 workflow orchestration 능력을 향상시키기 위한 새로운 접근 방식을 소개합니다. 이 프레임워크는 106,763개의 샘플과 1,503개의 API를 포함하는 대규모 fine-tuning 데이터셋인 WorkflowBench를 기반으로 하며, 기존의 LLM들이 가지고 있던 제한점을 극복할 수 있는 방안을 모색합니다.

- **Technical Details**: WorkflowLLM은 데이터 수집, 쿼리 확장, workflow 생성의 세 가지 단계로 구성됩니다. 첫 번째 단계에서는 Apple Shortcuts와 RoutineHub로부터 실제 workflow 데이터를 수집하고 이를 Python 스타일 코드로 변환하여 처리합니다. 두 번째 단계에서는 ChatGPT를 활용하여 더 다양한 작업 쿼리를 생성하여 workflow의 다양성을 높입니다. 마지막 단계에서는 수집된 데이터를 기반으로 학습한 annotator 모델을 사용하여 쿼리에 대한 workflows를 생성합니다. 이러한 과정을 통해 Llama-3.1-8B 모델을 fine-tuning하여 WorkflowLlama를 생성합니다.

- **Performance Highlights**: 실험 결과, WorkflowLlama는 기존의 모든 기준선 모델들, 특히 GPT-4o를 포함해, unseen instructions 및 unseen APIs 설정하에서 강력한 workflow orchestration 능력을 보여주었으며, T-Eval 벤치마크에서 out-of-distribution 상황에서도 뛰어난 일반화 능력을 입증하였습니다. F1 plan score는 77.5%로 높게 기록되었습니다.



### Learning the rules of peptide self-assembly through data mining with large language models (https://arxiv.org/abs/2411.05421)
- **What's New**: 이 연구에서는 펩타이드 자가 조립 행동에 대한 실험적 데이터셋을 수집하고, 이를 통해 머신 러닝 모델을 훈련하여 자가 조립 단계(Peptide Assembly Phase)를 예측합니다. 특히, 기존의 문헌 데이터베이스에서 1,000개 이상의 실험 데이터를 수집하여 자가 조립의 기본 규칙을 이해하는 데 기여하고자 하였습니다.

- **Technical Details**: 제안된 연구에서는 SAPdb 데이터베이스에서 문헌 채굴(literature mining)을 통해 1,012개의 데이터를 수집하고, 이를 기반으로 다양한 머신 러닝 알고리즘을 훈련합니다. 다수의 ML 알고리즘(예: Random Forest, GNN, Transformer 기반 모델)을 비교하여 최적의 성능을 발휘하는 모델을 선정하고, 각 모델을 사용하여 자가 조립 단계를 분류합니다. 또한, OpenAI API를 통해 GPT-3.5 Turbo 모델을 미세 조정하여 정보 추출 성능을 향상시킵니다.

- **Performance Highlights**: 수집된 데이터셋을 활용한 머신 러닝 모델은 자가 조립 단계 분류에서 80% 이상의 높은 정확도를 달성하였으며, 직관적인 데이터 처리를 통해 실험 효율성을 극대화하는 데 기여합니다. Fine-tuned GPT 모델은 기존 모델 대비 학술 출판물에서 정보 추출에서 뛰어난 성능을 나타내며, 자가 조립 펩타이드 후보를 탐색할 때 실험 작업을 안내하는 데 유용하며, 바이오 재료, 센서 및 촉매와 같은 다양한 응용 분야에서의 새로운 구조 접근을 용이하게 합니다.



### Revisiting the Robustness of Watermarking to Paraphrasing Attacks (https://arxiv.org/abs/2411.05277)
Comments:
          EMNLP 2024

- **What's New**: 본 연구는 언어 모델(LM)에서 생성된 텍스트의 감지를 위한 수조 방지 기법의 효과성을 강조합니다. 기존의 수조 방지 기법들은 높은 강인성을 주장하지만, 실제로는 역설계(reverse-engineering)가 용이하다는 점을 보여줍니다.

- **Technical Details**: 연구에서는 언어 모델의 출력에서 수조 신호를 포함시키는 방법을 논의하며, 이를 임의로 선택된 토큰 집합(그린 리스트)으로 부각시키는 기법을 설명합니다. 이 연구는 특정 수의 생성 결과만으로도 파라프레이징 공격(paraphrasing attacks)의 효과를 극대화할 수 있음을 나타냅니다.

- **Performance Highlights**: 200K개의 수조된 토큰을 사용할 경우, 그린 리스트의 정확도를 0.8 이상의 F1 점수를 기록할 수 있으며, 이를 통해 파라프레이징을 통해 수조 탐지율을 10% 이하로 떨어트릴 수 있습니다. 이러한 결과는 수조 방지 알고리즘의 강인성에 대한 우려를 제기합니다.



### Decoding Report Generators: A Cyclic Vision-Language Adapter for Counterfactual Explanations (https://arxiv.org/abs/2411.05261)
- **What's New**: 보고서 생성 모델에서 생성된 텍스트의 해석 가능성을 향상시키기 위한 혁신적인 접근 방식이 소개되었다. 이 방법은 사이클 텍스트 조작(cyclic text manipulation)과 시각적 비교(visual comparison)를 활용하여 원본 콘텐츠의 특징을 식별하고 설명한다.

- **Technical Details**: 새롭게 제안된 접근 방식은 Cyclic Vision-Language Adapters (CVLA)를 활용하여 설명의 생성 및 이미지 편집을 수행한다. counterfactual explanations를 통해 원본 이미지와 대조하여 수정된 이미지와의 비교를 통해 해석을 제공하며, 이는 모델에 구애받지 않는 방식으로 이루어진다.

- **Performance Highlights**: 이 연구의 방법론은 다양한 현재 보고서 생성 모델에서 적용 가능하며, 생성된 보고서의 신뢰성을 평가하는 데 기여할 것으로 기대된다.



### Evaluating GPT-4 at Grading Handwritten Solutions in Math Exams (https://arxiv.org/abs/2411.05231)
- **What's New**: 최근 generative artificial intelligence (AI)의 발전은 개방형 학생 응답을 정확하게 점수 매기는 데 가능성을 보여주었습니다. 이 연구에서는 특히 GPT-4o와 같은 최신 multi-modal AI 모델을 활용하여 대학 수준의 수학 시험에 대한 손글씨 응답을 자동으로 평가하는 방법을 제안합니다.

- **Technical Details**: 이 연구에서는 OpenAI의 GPT-4o 모델을 사용하여 실제 학생의 손글씨 시험 응답을 평가합니다. 평가 방법에는 3가지 prompting 방식이 포함되어 있으며(N - 문맥 없음, C - 정답 제공, CR - 정답과 루브릭 제공), 다양한 매트릭스를 통해 AI 모델의 점수 예측과 인간 채점자의 실제 점수를 비교합니다.

- **Performance Highlights**: GPT-4o 모델은 루브릭을 제공할 때 전반적인 점수 일치도가 개선되지만, 여전히 정확도는 너무 낮은 수준으로 실질적인 적용이 어렵습니다. 특히, 학생의 서술 방식 이해에 어려움이 있으며, 여러 요인으로 인해 점수 부여에서 많은 문제가 발생하고 있습니다.



### Alopex: A Computational Framework for Enabling On-Device Function Calls with LLMs (https://arxiv.org/abs/2411.05209)
- **What's New**: Alopex는 Fox LLM을 사용하여 모바일 기기에서 정확한 기능 호출을 가능하게 하는 새로운 프레임워크를 제안합니다. 이 접근 방식은 고품질 훈련 데이터를 생성하는 논리 기반 방법과 기능 호출 데이터의 혼합 전략을 통합하여 성능을 향상시킵니다.

- **Technical Details**: Alopex는 'description-question-output' 포맷을 사용하여 LLM 미세 조정을 최적화하며, 기능 정보 누출의 위험을 감소시킵니다. 또한, 데이터 혼합 전략을 통해 기능 호출 훈련 중의 catastrophic forgetting을 완화합니다.

- **Performance Highlights**: 실험 결과 Alopex는 기능 호출 정확도를 향상시키고 기존 LLM에서 관찰되는 catastrophic forgetting 현상을 유의미하게 줄였습니다. 이는 LLM의 기능 호출 능력을 통합하는 강력한 해결책을 제공합니다.



### Toward Cultural Interpretability: A Linguistic Anthropological Framework for Describing and Evaluating Large Language Models (LLMs) (https://arxiv.org/abs/2411.05200)
Comments:
          Accepted for publication in Big Data & Society, November 2, 2024

- **What's New**: 이 논문은 언어 인류학(linguistic anthropology)과 머신 러닝(machine learning, ML)의 새로운 통합을 제안하며, 언어의 기반과 언어 기술의 사회적 책임에 대한 관심이 융합되는 지점을 탐구합니다.

- **Technical Details**: 연구에서는 LLM(대형 언어 모델) 기반 챗봇과 인간 사용자의 대화를 분석하여, 문화적 해석 가능성(cultural interpretability, CI)이라는 새로운 연구 분야를 제시합니다. CI는 인간-컴퓨터 상호작용의 발화적 인터페이스에서 인간 사용자가 AI 챗봇과 함께 의미를 생성하는 방식을 중심으로, 언어와 문화 간의 역동적인 관계를 강조합니다.

- **Performance Highlights**: CI는 LLM이 언어와 문화의 관계를 내부적으로 어떻게 '표현'하는지를 조사하며, (1) 언어 인류학의 오랜 질문에 대한 통찰을 제공하고, (2) 모델 개발자와 인터페이스 디자이너가 언어 모델과 다양한 스타일의 화자 및 문화적으로 다양한 언어 커뮤니티 사이의 가치 정렬을 개선할 수 있도록 돕습니다. 논문은 상대성(relativity), 변이(variation), 지시성(indexicality)이라는 세 가지 중요한 연구 축을 제안합니다.



### On Erroneous Agreements of CLIP Image Embeddings (https://arxiv.org/abs/2411.05195)
Comments:
          18 pages, 4 figures

- **What's New**: 본 연구에서는 Vision-Language Models (VLMs)의 시각적 추론에서 발생하는 오류의 주 원인이 항상 잘못된 합의(erroneous agreements) 때문이 아님을 보여줍니다. LLaVA-1.5-7B 모델은 CLIP 이미지 인코더를 사용하면서도 퀘리와 관련된 시각적 정보를 추출할 수 있음을 밝혔습니다.

- **Technical Details**: CLIP 이미지 인코더는 시각적으로 구별되는 이미지들이 높은 코사인 유사도(cosine similarity)로 애매하게 인코딩되는 문제를 가지고 있습니다. 그러나 LLaVA-1.5-7B는 이러한 CLIP 이미지 임베딩에서 추출 가능한 정보를 이용해 더 나은 성능을 보입니다. 이를 위해 M3ID(Multi-Modal Mutual-Information Decoding)와 같은 대체 디코딩 알고리즘을 사용하여 시각적 입력에 더 많은 주의를 기울일 수 있게 했습니다.

- **Performance Highlights**: LLaVA-1.5-7B는 What'sUp 벤치마크에서 100%에 가까운 정확도로 작업을 수행하였고, MMVP 벤치마크에서도 CLIP 기반 모델보다 우수한 성능을 나타냈습니다. 전체적으로, CLIP 이미지 인코더를 개선하는 것도 중요하지만, 고정된 이미지를 사용하는 모델에서도 정보를 더 잘 추출하고 활용하는 전략을 적용하는 여지가 남아 있음을 보여주고 있습니다.



### Interactive Dialogue Agents via Reinforcement Learning on Hindsight Regenerations (https://arxiv.org/abs/2411.05194)
Comments:
          23 pages, 5 figures

- **What's New**: 본 논문에서는 대화 에이전트가 대화를 효과적으로 이끌 수 있는 능력을 향상시키기 위한 새로운 방법을 제안합니다. 특히, 기존 데이터에 대한 후행(regeneration)을 통해 대화 에이전트를 훈련시키는 방법을 소개하며, 이는 특히 정신 건강 지원 및 자선 기부 요청과 같은 복잡한 대화 작업에 적용됩니다.

- **Technical Details**: 대화 에이전트는 오프라인 강화 학습(offline reinforcement learning, RL)을 사용하여 훈련됩니다. 기존 비효율적인 대화 데이터를 개선하고 새로운 대화 전략을 학습하기 위해, 사후에 생성된 합성 데이터(synthetic data)를 추가하여 적절한 행동을 나타내는 다양한 대화 전략을 포착합니다.

- **Performance Highlights**: 실제 사용자를 대상으로 한 연구 결과, 제안된 방법이 기존 최첨단 대화 에이전트에 비해 효과성, 자연스러움 및 유용성 측면에서 크게 우수함을 보여주었습니다.



### Q-SFT: Q-Learning for Language Models via Supervised Fine-Tuning (https://arxiv.org/abs/2411.05193)
Comments:
          16 pages, 4 figures

- **What's New**: 이번 연구에서는 기존의 가치 기반 강화 학습(value-based reinforcement learning, RL) 알고리즘의 한계를 극복하기 위한 새로운 오프라인 RL 알고리즘을 제안합니다. 이 알고리즘은 Q-learning을 수정된 감독 세부 조정(supervised fine-tuning, SFT) 문제로 간주하며, 이를 통해 언어 모델의 사전 학습(pretraining) 효과를 충분히 활용할 수 있습니다.

- **Technical Details**: 제안된 알고리즘은 최대 우도(maximum likelihood) 목표에 가중치를 추가하여, 잔여 정책(behavior policy) 대신 보수적으로 가치 함수를 추정하는 확률을 학습합니다. 이를 통해 비안정적인 가치 학습(regression) 목표를 피하고, 대규모 사전 학습에서 발생한 초기 우도(likelihood)를 직접 활용할 수 있습니다.

- **Performance Highlights**: 실험을 통해 LLM(대형 언어 모델)과 VLM(비전-언어 모델)의 다양한 작업에서 제안된 알고리즘의 효과를 입증했습니다. 자연어 대화, 로봇 조작 및 네비게이션 등 여러 과제에서 기존의 감독 세부 조정 방법을 초월하는 성능을 보였습니다.



### Watermarking Language Models through Language Models (https://arxiv.org/abs/2411.05091)
- **What's New**: 이 논문은 언어 모델을 통한 워터마킹(watermarking)을 위한 새로운 프레임워크를 제시합니다. 제안된 접근 방식은 다중 모델(multimodal) 설정을 활용하여 언어 모델이 생성한 프롬프트를 통해 워터마킹 지침을 생성하고, 생성된 컨텐츠에 워터마크를 삽입하며, 이러한 워터마크의 존재를 검증하는 모델들을 포함합니다.

- **Technical Details**: 논문에서는 ChatGPT와 Mistral을 프롬프트 생성 및 워터마킹 모델로 사용하고, 분류기(classifier) 모델을 통해 감지 정확도를 평가했습니다. 이 프레임워크는 다양한 구성에서 95%의 ChatGPT 감지 정확도와 88.79%의 Mistral 감지 정확도를 달성했습니다.

- **Performance Highlights**: 제안된 프레임워크는 다른 언어 모델 아키텍처에서 워터마킹 전략의 유효성과 적응성을 검증합니다. 이러한 결과는 콘텐츠 획득, 저작권 보호, 모델 인증 등의 응용 분야에서 유망한 가능성을 나타냅니다.



### PadChest-GR: A Bilingual Chest X-ray Dataset for Grounded Radiology Report Generation (https://arxiv.org/abs/2411.05085)
- **What's New**: 이번 연구에서는 Chest X-ray(CXR) 이미지를 위한 최초의 손으로 주석이 달린 데이터셋인 PadChest-GR(도구화 보고서)의 개발을 발표합니다. 이 데이터셋은 각 개별 발견을 기술하는 문장의 완전한 목록을 제공하여 GRRG(기반 방사선 보고서 생성) 모델 훈련에 기여합니다.

- **Technical Details**: PadChest-GR 데이터셋은 영어와 스페인어로 된 4,555개의 CXR 연구를 포함하고 있으며, 여기에는 3,099개의 비정상 및 1,456개의 정상 케이스가 포함되어 있습니다. 각 긍정 발견 문장은 두 개의 독립적인 바운딩 박스 세트와 관련이 있으며, 발견 유형, 위치, 진행 상태에 대한 범주형 레이블이 함께 제공됩니다. 이는 의료 이미지와 생성된 텍스트의 이해 및 해석을 위한 최초의 수작업으로 구성된 데이터셋입니다.

- **Performance Highlights**: PadChest-GR은 의료 AI 모델의 검증을 돕고, 이해도를 높이며, 임상의와 환자 간의 상호작용을 촉진하는 데 중요한 기초를 제공하고 있습니다. 이 데이터셋은 방사선 이미지를 수집하여 의료 AI 모델의 개발 및 평가에 유용한 자료로 활용될 것입니다.



### Precision or Recall? An Analysis of Image Captions for Training Text-to-Image Generation Mod (https://arxiv.org/abs/2411.05079)
Comments:
          EMNLP 2024 Findings. Code: this https URL

- **What's New**: 본 논문은 텍스트-이미지 모델 훈련에서 캡션(기술명령)의 정밀도(precision)와 재현율(recall)의 중요성을 분석하고, 합성(합성된) 캡션을 생성하기 위해 Large Vision Language Models(LVLMs)를 활용한 새로운 접근 방식을 소개합니다.

- **Technical Details**: 연구에서는 Dense Caption Dataset을 사용하여 이미지 캡션의 정밀도와 재현율이 텍스트-이미지(T2I) 모델의 조합 능력에 미치는 영향을 체계적으로 평가하였습니다. Human-annotated 캡션과 LVLM으로 생성된 캡션을 활용하여 T2I 모델을 훈련시키고, 이 모델들이 조합 능력에서 어떤 성능을 보이는지 분석하였습니다.

- **Performance Highlights**: 이 연구의 결과는 캡션의 정밀도가 성능에 더 큰 영향을 미친다는 것을 강조합니다. LVLM을 사용해 생성된 합성 캡션으로 훈련된 T2I 모델은 Human-annotated 캡션으로 훈련된 모델과 유사한 성능을 보였으며, 특히 정밀도가 중요한 역할을 한다는 결론에 도달하였습니다.



### A Guide to Misinformation Detection Datasets (https://arxiv.org/abs/2411.05060)
- **What's New**: 본 논문은 기존의 허위 정보(misinformation) 데이터셋에 대한 가장 포괄적인 조사를 제공하며, 총 75개의 데이터셋을 정리하고 분석했습니다. 특히, 36개의 주장(claims) 데이터셋의 품질을 평가하여, 고품질 데이터의 선별을 위한 지침을 제시합니다.

- **Technical Details**: 저자들은 표기(label) 품질, 키워드 기반 상관관계(spurious correlations), 정치적 편향(political bias)에 대한 평가를 수행하였으며, 이러한 요소들이 잘못된 예측을 초래할 수 있음을 강조합니다. 또한, GPT-4를 활용하여 현대적인 기준과 베이스라인을 설정하였고, 정확도(accuracy)와 F1 스코어가 더 이상 충분하지 않음을 지적합니다.

- **Performance Highlights**: 본 연구는 75개 데이터셋에 대한 종합적인 신뢰도와 질을 평가하고, 안정적인 구조의 데이터 및 분석을 통해 허위 정보 탐지 연구를 개선하기 위한 로드맵을 제시합니다. 데이터셋의 다양한 주제 및 사례를 포함하며, 초기의 새로운 평가 기준을 제안하고 있습니다.



### Mitigating Privacy Risks in LLM Embeddings from Embedding Inversion (https://arxiv.org/abs/2411.05034)
- **What's New**: 이 논문에서는 embedding inversion 공격에 대한 방어 메커니즘인 Eguard를 소개합니다. 이 방법은 transformer 기반의 projection network와 text mutual information 최적화를 사용하여 embedding의 개인 정보를 보호합니다.

- **Technical Details**: Eguard는 embedding 공간을 안전한 embedding 공간으로 투영하여 민감한 특징을 분리합니다. 이 과정에서 autoencoder를 활용하여 텍스트와 embedding 간의 상호 정보를 계산하고, 다중 작업 최적화 메커니즘을 도입하여 방어와 기능 간의 균형을 유지합니다.

- **Performance Highlights**: Eguard는 embedding inversion 공격으로부터 95% 이상의 토큰을 보호하며, 4가지 하류 작업에서 98% 이상의 원래 embedding과의 일관성을 유지합니다. 또한, Eguard는 embedding perturbations, 보지 않은 학습 시나리오 및 적응 공격에 대한 강력한 방어 성능을 보여줍니다.



### BhasaAnuvaad: A Speech Translation Dataset for 13 Indian Languages (https://arxiv.org/abs/2411.04699)
Comments:
          Work in Progress

- **What's New**: 본 논문에서는 인도 언어를 위한 자동 음성 번역 (Automatic Speech Translation, AST) 시스템의 미비한 데이터 세트를 해결하기 위해 BhasaAnuvaad라는 대규모 데이터 세트를 소개합니다. 이는 22개 공식 언어 중 13개 언어와 영어를 포함하며, 44,400시간 이상의 음성과 1,700만 텍스트 세그먼트를 포함하고 있습니다.

- **Technical Details**: BhasaAnuvaad 데이터 세트는 (1) 기존 자원에서의 큐레이션된 데이터 세트, (2) 대규모 웹 마이닝, (3) 합성 데이터 생성의 세 가지 주요 카테고리로 구성됩니다. 이 데이터 세트는 읽은 음성(FLUERS 데이터 세트 사용)과 자발적 음성을 평가하기 위한 새로운 벤치마크인 Indic-Spontaneous-Synth를 사용하여 평가됩니다.

- **Performance Highlights**: 기존의 AST 시스템은 읽은 음성에서는 적절한 성능을 보이나 자발적 음성에서는 현저하게 성능이 저하됨을 보여줍니다. 이 연구는 자발적 언어의 복잡성을 반영하기 위해 더 많은 데이터 세트와 평가가 필요하다는 것을 강조합니다.



New uploads on arXiv(cs.IR)

### Harnessing High-Level Song Descriptors towards Natural Language-Based Music Recommendation (https://arxiv.org/abs/2411.05649)
- **What's New**: 이 논문은 자연어 기반의 음악 추천 시스템에서 언어 모델(Language Models, LMs)의 효과성을 분석합니다. 특히, 사용자 자연어 설명과 장르, 기분, 청취 맥락과 같은 항목의 고수준(descriptors) 설명이 조화를 이루도록 하는 접근을 제안합니다.

- **Technical Details**: 추천 작업을 밀집 검색(dense retrieval) 문제로 공식화하며, LMs가 특정 작업과 도메인에 점차 익숙해지면서 효과를 평가합니다. 또한, 이 연구는 고수준 설명을 기반으로 하는 추천 모듈을 개발하여, 저수준의 오디오 또는 메타데이터와는 구별되는 방식으로 추천 성과를 향상시킵니다.

- **Performance Highlights**: 구현된 LMs는 사전 훈련(pre-trained) 상태에서 성능이 떨어지나, 점진적으로 텍스트 유사성 및 다중 도메인 쿼리 검색에 대한 미세 조정(fine-tuning)을 거치면서 성능이 개선됨을 확인했습니다. 더 나아가, 모델과 데이터셋은 음악 관련 설명을 검색하는 데도 적합하다는 것을 밝혔습니다.



### Why These Documents? Explainable Generative Retrieval with Hierarchical Category Paths (https://arxiv.org/abs/2411.05572)
- **What's New**: 본 논문에서는 기존의 generative retrieval 기법의 한계를 극복하기 위해 Hierarchical Category Path-Enhanced Generative Retrieval(HyPE)를 제안합니다. HyPE는 docid를 직접 디코딩하기 전에 계층적인 카테고리 경로(hierarchical category paths)를 단계별로 생성하여 설명 가능성을 향상시킵니다.

- **Technical Details**: HyPE는 외부의 고품질 의미적 계층(semantic hierarchy)을 바탕으로 카테고리 경로를 구성하고, 각 문서에 적합한 후보 경로를 선택하기 위해 LLM(Large Language Models)을 활용합니다. 경로를 포함한 데이터셋으로 generative retrieval 모델을 최적화하고, 추론 단계에서는 경로 인식 재순위(path-aware reranking) 전략을 사용하여 다양한 주제 정보를 집계합니다.

- **Performance Highlights**: HyPE는 높은 설명 가능성을 제공할 뿐만 아니라 문서 검색(task in document retrieval) 성능을 개선하는데 성공했습니다. 다양한 docid 형식에 적용 가능하며, 다른 generative retrieval 시스템에 쉽게 통합될 수 있는 다재다능한 프레임워크로 평가받고 있습니다.



### An Early FIRST Reproduction and Improvements to Single-Token Decoding for Fast Listwise Reranking (https://arxiv.org/abs/2411.05508)
- **What's New**: FIRST는 listwise reranking(리스트와이즈 리랭킹) 문제를 해결하기 위해 learning-to-rank(학습 기반 순위) 목표를 포함하고 첫 번째 생성된 토큰의 logits(로짓)만을 활용하여 지연 시간을 효과적으로 줄이는 새로운 접근법입니다.

- **Technical Details**: 기존의 LLM reranker(대형 언어 모델 리랭커)는 전체 문서 식별자의 순열을 생성해야 하는 반면, FIRST는 첫 번째 토큰의 logits을 바탕으로 재정렬을 수행합니다. 이 접근은 inference latency(추론 지연)을 21%-42% 줄이며, 다양한 backbone models(백본 모델)에 적용하여 효과성을 높였습니다.

- **Performance Highlights**: FIRST는 기존 LLM reranker 보다 효율적이면서도 효과적인 작업을 수행하며, 21%-42%의 지연 시간 개선을 기록했습니다. 실험 결과, single-token logits(단일 토큰 로짓)만을 사용하더라도 out-of-domain(도메인 외) 리랭킹 품질은 유지되며, 기존 LM training(언어 모델 훈련)이 zero-shot single-token reranking(제로샷 단일 토큰 리랭킹) 능력을 향상시키는 가운데, LM pre-training(언어 모델 사전 훈련)이 FIRST 목표의 fine-tuning(미세 조정)을 저해할 수 있음을 발견했습니다.



### IntellBot: Retrieval Augmented LLM Chatbot for Cyber Threat Knowledge Delivery (https://arxiv.org/abs/2411.05442)
- **What's New**: 이 논문은 최신 사이버 보안 챗봇인 IntellBot을 개발한 내용을 소개합니다. IntellBot은 인공지능(Artificial Intelligence) 및 자연어 처리(Natural Language Processing) 기술을 바탕으로 정보 조회와 사이버 위협 지식을 제공하는 고급 챗봇입니다.

- **Technical Details**: IntellBot은 대규모 언어 모델(Large Language Model) 및 Langchain을 활용하여 구축되었습니다. 이 챗봇은 2,447개의 PDF 문서, 7,989개의 악성 파일 해시 정보, 2,959개의 URL 세부사항과 같은 다양한 데이터 소스로부터 지식베이스를 수집하여 정보를 제공합니다.

- **Performance Highlights**: IntellBot의 성능은 BERT 점수 0.8 이상과 코사인 유사도 점수 0.8에서 1까지의 범위를 통해 평가되었습니다. 또한 RAGAS를 사용하여 모든 평가 메트릭이 0.77 이상의 점수를 기록하여 시스템의 효과성을 강조하였습니다.



### Ev2R: Evaluating Evidence Retrieval in Automated Fact-Checking (https://arxiv.org/abs/2411.05375)
Comments:
          10 pages

- **What's New**: 본 논문에서는 Automated Fact-Checking (AFC)을 위한 새로운 평가 프레임워크인 Ev2R을 소개합니다. Ev2R은 증거 평가를 위한 세 가지 접근 방식인 reference-based, proxy-reference, reference-less 평가 방식으로 구성되어 있습니다.

- **Technical Details**: Ev2R 프레임워크의 세 가지 평가자 그룹은 (1) reference-based 평가자: 참조 증거와 비교하여 검색된 증거를 평가, (2) proxy-reference 평가자: 시스템이 예측한 평결을 기준으로 증거를 평가, (3) reference-less 평가자: 입력 클레임만을 기반으로 증거를 평가하는 방식입니다. 구체적으로 LLMs를 활용하여 증거를 원자적 사실로 분해한 후 평가합니다.

- **Performance Highlights**: Ev2R의 reference-based 평가자는 전통적인 메트릭보다 인간 평가와 높은 상관관계를 보였습니다. Gemini 기반 평가자는 검색된 증거가 참조 증거를 얼마나 포함하고 있는지를 잘 평가한 반면, GPT-4 기반 평가자는 평결 일치에서 더 좋은 성능을 보였습니다. 이 결과는 Ev2R 프레임워크가 더욱 정확하고 강력한 증거 평가를 가능하게 함을 시사합니다.



### Improving Multi-Domain Task-Oriented Dialogue System with Offline Reinforcement Learning (https://arxiv.org/abs/2411.05340)
- **What's New**: 이번 논문은 사전 훈련된 GPT-2 모델을 활용하여 사용자 정의 작업을 수행하는 대화형 시스템(TOD)을 제안합니다. 주요 특징은 감독 학습(Supervised Learning)과 강화 학습(Reinforcement Learning)을 결합하여, 성공률(Success Rate)과 BLEU 점수를 기반으로 보상 함수를 최적화하여 모델의 성능을 향상시키는 것입니다.

- **Technical Details**: 제안된 TOD 시스템은 사전 훈련된 대형 언어 모델인 GPT-2를 기반으로 하며, 감독 학습과 강화 학습을 통해 최적화되었습니다. 비가역적 보상 함수를 사용하여 대화 결과의 성공률과 BLEU 점수를 가중 평균하여 보상을 계산합니다. 모델은 사용자 발화, 신념 상태(Belief State), 시스템 액트(System Act), 시스템 응답(System Response)으로 구성된 대화 세션 수준에서 미세 조정(Fine-tuning)됩니다.

- **Performance Highlights**: MultiWOZ2.1 데이터셋에서 실험 결과, 제안한 모델은 기준 모델에 비해 정보 비율(Inform Rate)을 1.60% 증가시키고 성공률(Success Rate)을 3.17% 향상시켰습니다.



### FineTuneBench: How well do commercial fine-tuning APIs infuse knowledge into LLMs? (https://arxiv.org/abs/2411.05059)
- **What's New**: 이 연구에서는 FineTuneBench라는 새로운 평가 프레임워크 및 데이터셋을 소개하여 상업적 LLM(대형 언어 모델) 미세 조정 API가 새로운 정보와 업데이트된 지식을 성공적으로 학습할 수 있는지를 분석합니다.

- **Technical Details**: FineTuneBench 데이터셋은 뉴스, 허구의 인물, 의료 지침, 코드의 네 가지 영역에서 625개의 학습 질문과 1,075개의 테스트 질문으로 구성되어 있습니다. 연구진은 GPT-4o, GPT-3.5 Turbo, Gemini 1.5 Pro 등의 LLM 5개를 활용하여 미세 조정 서비스를 평가했습니다. 이 연구는 새로운 정보를 학습하는 LLM의 능력에 대해 체계적인 평가를 제공합니다.

- **Performance Highlights**: 모델들의 평균 일반화 정확도는 37%로, 새로운 정보를 효과적으로 학습하는 데에는 상당한 한계가 있음을 보여주었습니다. 기존 지식을 업데이트할 때는 평균 19%의 정확도만을 기록하며, 특히 Gemini 1.5 시리즈는 새로운 지식을 학습하거나 기존 지식을 업데이트하는 데에 실패했습니다. GPT-4o mini가 새로운 지식을 주입하고 지식을 업데이트하는 데 가장 효과적인 모델로 판단되었습니다.



### Leveraging LLMs to Enable Natural Language Search on Go-to-market Platforms (https://arxiv.org/abs/2411.05048)
Comments:
          11 pages, 5 figures

- **What's New**: 이 논문은 Zoominfo 제품을 위한 자연어 쿼리(queries) 처리 솔루션을 제안하고, 대규모 언어 모델(LLMs)을 활용하여 검색 필드를 생성하고 이를 쿼리로 변환하는 방법을 평가합니다. 이 과정은 복잡한 메타데이터를 요구하는 기존의 고급 검색 방식의 단점을 극복하려는 시도입니다.

- **Technical Details**: 제안된 솔루션은 LLM을 사용하여 자연어 쿼리를 처리하며, 중간 검색 필드를 생성하고 이를 JSON 형식으로 변환하여 최종 검색 서비스를 위한 쿼리로 전환합니다. 이 과정에서는 여러 가지 고급 프롬프트 엔지니어링 기법(techniques)과 체계적 메시지(system message),few-shot prompting, 체인 오브 씽킹(chain-of-thought reasoning) 기법을 활용했습니다.

- **Performance Highlights**: 가장 정확한 모델은 Anthropic의 Claude 3.5 Sonnet으로, 쿼리당 평균 정확도(accuracy) 97%를 기록했습니다. 부가적으로, 작은 LLM 모델들 또한 유사한 결과를 보여 주었으며, 특히 Llama3-8B-Instruct 모델은 감독된 미세조정(supervised fine-tuning) 과정을 통해 성능을 향상시켰습니다.



New uploads on arXiv(cs.CV)

### GazeSearch: Radiology Findings Search Benchmark (https://arxiv.org/abs/2411.05780)
Comments:
          Aceepted WACV 2025

- **What's New**: 이 논문에서는 방사선학에 특화된 시선 추적 데이터셋인 GazeSearch를 제안하고, 이를 통해 방사선과 의료 이미징의 해석 가능성과 정확성을 개선하고자 합니다. 또한, ChestSearch라는 스캔 경로 예측 모델을 소개하여 현재의 알고리즘 성능을 평가합니다.

- **Technical Details**: GazeSearch는 기존의 시선 추적 데이터(EGD, REFLACX)를 가공하여 검사 결과에 주목하는 데이터셋으로 변환합니다. ChestSearch 모델은 Transformer 기반으로 설계되었으며, 자기 지도 학습 및 쿼리 메커니즘을 활용하여 중요한 시선 데이터를 선택합니다. 이 모델은 특정 데이터를 예측하는 다중 태스크 처리 기능을 갖추고 있습니다.

- **Performance Highlights**: GazeSearch 데이터셋을 기반으로 ChestSearch 모델의 성능을 평가한 결과, 기존의 시선 추적 예측 모델들보다 뛰어난 성능을 보였으며, 방사선학 분야에서 비주얼 서치의 최신 발전을 보여줍니다.



### Curriculum Learning for Few-Shot Domain Adaptation in CT-based Airway Tree Segmentation (https://arxiv.org/abs/2411.05779)
Comments:
          Under review for 22nd IEEE International Symposium on Biomedical Imaging (ISBI), Houston, TX, USA

- **What's New**: 이 논문에서는 심층 학습(Deep Learning) 기술을 활용하여 흉부 CT 스캔에서 기도를 자동으로 분할하는 문제를 해결하기 위해 교육 과정 학습(Curriculum Learning)을 통합한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 훈련 데이터를 CT 스캔과 이에 해당하는 정답 트리 피처에서 얻은 복잡도 점수에 따라 배치로 분배하여 모델을 훈련시키며, 적은 샷 도메인 적응(few-shot domain adaptation)을 적용합니다. 이는 전체 미세 조정(Fine-Tuning) 데이터 세트를 수동으로 주석을 달기 어렵거나 비용이 높은 상황을 겨냥합니다.

- **Performance Highlights**: ATM22와 AIIB23 두 개의 대규모 공개 코호트에서 교육 과정 학습을 이용한 전훈련(Source domain) 및 적은 샷 미세 조정(Target domain)을 통해 높은 성능을 보였습니다. 그러나 전통적인 부트스트래핑 scoring function 사용 시 또는 올바른 스캔 순서를 사용하지 않을 시 발생할 수 있는 잠재적 부정적인 효과에 대한 통찰도 제공합니다.



### WavShadow: Wavelet Based Shadow Segmentation and Remova (https://arxiv.org/abs/2411.05747)
Comments:
          ICVGIPâ€™24, December 2024, Bangaluru, India

- **What's New**: 본 연구는 ShadowFormer 모델을 Masked Autoencoder (MAE) 사전 정보와 Fast Fourier Convolution (FFC) 블록을 통합하여 성능을 크게 향상시킨 새로운 접근 방식을 소개합니다. 이 방법은 더 빠른 수렴 속도와 뛰어난 성능 개선을 이루었습니다.

- **Technical Details**: 본 연구는 다음과 같은 기술적 혁신을 포함합니다: (1) Places2 데이터셋에서 훈련된 MAE 사전 정보를 사용하여 맥락 이해를 개선, (2) Haar wavelet 특징을 채택하여 엣지 검출 및 다중 스케일 분석을 향상, (3) 강력한 그림자 분할을 위한 수정된 SAM Adapter의 구현. 또한, MAE는 Places2 데이터셋에서 사전 훈련된 모델을 사용하여 그림자를 제거하기 위한 파이프라인에 통합되었습니다.

- **Performance Highlights**: DESOBA 데이터셋에서 수행한 실험 결과, 본 방법은 기존 ShadowFormer 모델보다 약 200 epochs 덜 소요되며, 수렴 속도와 그림자 제거 품질 모두에서 현저한 개선을 나타냈습니다. 특히, 복잡한 실제 환경에서 더욱 우수한 결과를 보였습니다.



### StdGEN: Semantic-Decomposed 3D Character Generation from Single Images (https://arxiv.org/abs/2411.05738)
Comments:
          13 pages, 10 figures

- **What's New**: StdGEN은 단일 이미지에서 의미론적으로 분해된 고품질 3D 캐릭터를 생성하는 혁신적인 파이프라인입니다. 이 시스템은 짧은 시간 안에 복잡한 3D 캐릭터를 생성할 수 있으며, 분해 가능성을 제공합니다.

- **Technical Details**: 핵심 기술인 S-LRM(Semantic-aware Large Reconstruction Model)은 다중 시점 이미지에서 기하학, 색상 및 의미를 동시적으로 재구성하는 트랜스포머 기반 모델입니다. Differentiable multi-layer semantic surface extraction 기법을 통해 하이브리드 임플리시트 필드에서 메쉬를 획득하고, 고급 멀티 뷰 확산 모델 및 반복적인 멀티 레이어 표면 정제 모듈이 통합되어 있습니다.

- **Performance Highlights**: StdGEN은 기존의 기준을 크게 초과하는 성능을 입증했으며, 특히 3D 애니 캐릭터 생성에서 최첨단 성능을 달성하여 여러 다운스트림 응용 프로그램에 유용한 유연한 커스터마이징을 가능하게 합니다.



### Poze: Sports Technique Feedback under Data Constraints (https://arxiv.org/abs/2411.05734)
- **What's New**: Poze는 전문가 코치의 통찰력을 모방하여 인체의 동작에 피드백을 제공하는 혁신적인 비디오 처리 프레임워크입니다. 이 시스템은 최소한의 데이터로 작동하도록 최적화되었습니다.

- **Technical Details**: Poze는 포즈 추정(pose estimation)과 시퀀스 비교(sequence comparison)를 결합하여 작동하며, 동적 시간 정렬(Dynamic Time Warping, DTW)을 사용하여 비디오에서 포즈 시퀀스를 추출하고 전처리합니다. 이러한 프로세스를 통해 각 관절의 평균 오류(μj)와 분산(σj)을 계산하여 최적의 기술 수행을 일반화합니다.

- **Performance Highlights**: Poze는 최신 비전-언어 모델(Vision-Language Models)인 GPT4V 및 LLaVAv1.6 7b보다 각각 70% 및 196%의 정확도 향상을 달성하여, 자원 집약적인 VQA 프레임워크에 비해 데이터 효율성을 보여주고 있습니다.



### PEP-GS: Perceptually-Enhanced Precise Structured 3D Gaussians for View-Adaptive Rendering (https://arxiv.org/abs/2411.05731)
- **What's New**: PEP-GS는 구조화된 3D Gaussian splatting을 위한 새로운 프레임워크로, 시각적 일관성 및 복잡한 뷰 의존 효과를 처리하는 세 가지 주요 혁신을 도입합니다.

- **Technical Details**: 1. Local-Enhanced Multi-head Self-Attention (LEMSA) 메커니즘을 통해 구형 조화 함수(spherical harmonics)를 대체하고, 더 정밀한 뷰 의존 색상 디코딩을 구현합니다. 2. Kolmogorov-Arnold Networks (KAN)를 활용하여 Gaussian 불투명도(opacity)와 공분산(covariance) 함수의 최적화를 통해 해석 가능성과 정확성을 높입니다. 3. Neural Laplacian Pyramid Decomposition (NLPD)를 도입하여 다양한 뷰에서의 지각적 유사성을 향상시킵니다.

- **Performance Highlights**: PEP-GS는 다양한 데이터셋에서 평가했으며, 기존의 최신 방법들에 비해 복잡한 조명과 기하학적 세부사항에서도 특히 뛰어난 성능을 보여줍니다.



### STARS: Sensor-agnostic Transformer Architecture for Remote Sensing (https://arxiv.org/abs/2411.05714)
- **What's New**: 본 논문에서는 센서에 독립적인 스펙트럼 변환기(Spectral Transformer)를 제안하며, 이를 통해 다양한 스펙트럼 기초 모델을 구축하는 기초로 삼고자 한다. 연구팀은 센서 메타데이터를 활용하여 어떤 스펙트럼 기기에서 얻은 스펙트럼을 공통 표현으로 인코딩하는 범용 스펙트라 표현(USR, Universal Spectral Representation)을 소개한다.

- **Technical Details**: 제안된 아키텍처는 세 가지 모듈로 구성된다: (i) 범용 스펙트라 표현(USR), (ii) 스펙트럼 변환기 인코더, (iii) 연산자 이론 기반 디코더. 이 방법들은 센서에 독립적인 모델 학습을 위해 강력한 데이터 증강 전략과 결합된다. 또한 자가 지도식 복원 작업을 통해 저해상도 스펙트럼을 높은 해상도 스펙트럼으로 복원하는 방법을 제안한다.

- **Performance Highlights**: 모델을 통해 훈련 과정에서 보지 못한 센서로부터 유도된 스펙트럼 서명(latent representation)을 비교함으로써, 제안한 방법이 다양한 스펙트럼 데이터에서 중요한 정보를 신뢰성 있게 추출할 수 있음을 보여준다. 또한 이러한 성능을 통해 스펙트럼 데이터의 다양성을 활용할 수 있는 기초 모델의 교육을 위한 새로운 방향을 제시한다.



### Image2Text2Image: A Novel Framework for Label-Free Evaluation of Image-to-Text Generation with Text-to-Image Diffusion Models (https://arxiv.org/abs/2411.05706)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2408.01723

- **What's New**: 자동 생성된 이미지 설명의 품질을 평가하기 위한 새로운 프레임워크인 Image2Text2Image를 제안합니다. 이 프레임워크는 텍스트-이미지 생성 시스템인 Stable Diffusion이나 DALL-E를 활용하여 이미지 캡셔닝 모델을 평가합니다.

- **Technical Details**: 제안된 프레임워크는 입력 이미지를 선택한 이미지 캡셔닝 모델로 처리하여 텍스트 설명을 생성하고, 이를 통해 생성된 설명으로 새로운 이미지를 생성합니다. 이 과정에서 원본 이미지와 생성된 이미지의 특징을 추출하여 코사인 유사도(Cosine Similarity) 메트릭을 사용하여 유사성을 측정합니다. 고유의 코사인 유사도 점수는 생성된 설명이 원본 이미지를 얼마나 잘 표현했는지를 나타냅니다.

- **Performance Highlights**: 이 프레임워크는 기존의 사람 주석이 필요 없는 자동화된 평가 방법을 통해 이미지 캡셔닝 모델을 효과적으로 평가할 수 있음을 보여줍니다. 인상적인 실험 결과와 인간 평가를 통해 제안된 프레임워크의 효능이 확인되었습니다.



### Image inpainting enhancement by replacing the original mask with a self-attended region from the input imag (https://arxiv.org/abs/2411.05705)
- **What's New**: 본 논문은 이미지 인페인팅(image inpainting) 작업을 위한 새로운 딥 러닝 기반의 전처리(pre-processing) 방법론을 소개합니다. 이 접근 방식은 Vision Transformer (ViT)를 활용하여 마스킹된 픽셀 값을 ViT가 생성한 값으로 대체하는 방식을 사용합니다.

- **Technical Details**: 우리의 방법론은 주의(attention) 매트릭스 내의 다양한 시각적 패치를 활용하여 구별 가능한 공간적 특징(discriminative spatial features)을 포착하는 것을 목표로 합니다. 기존의 인페인팅 작업에서 이러한 전처리 모델이 제안된 것은 이번이 처음입니다.

- **Performance Highlights**: 우리의 전처리 기술이 인페인팅 성능 향상에 효과적임을 보여준 실험 결과를 제시하며, 네 개의 공공 데이터셋(public datasets)과 네 개의 표준 모델 간의 성능 비교를 통해 일반화(generalization) 능력을 평가했습니다.



### Visual-TCAV: Concept-based Attribution and Saliency Maps for Post-hoc Explainability in Image Classification (https://arxiv.org/abs/2411.05698)
Comments:
          Preprint currently under review

- **What's New**: 이 논문에서는 CNN (Convolutional Neural Networks)의 이미지 분류를 위한 새로운 설명 가능성 프레임워크인 Visual-TCAV를 소개합니다. 이 방법은 기존의 saliency 방법과 개념 기반 접근 방식을 통합하여, CNN의 예측에 대한 개념의 기여도를 숫자로 평가하는 동시에, 어떤 이미지에서 어떤 개념이 인식되었는지를 시각적으로 설명합니다.

- **Technical Details**: Visual-TCAV는 Concept Activation Vectors (CAVs)를 사용하여 네트워크가 개념을 인식하는 위치를 보여주는 saliency map을 생성합니다. 또한, Integrated Gradients의 일반화를 통해 이러한 개념이 특정 클래스 출력에 얼마나 중요한지를 추정할 수 있습니다. 이 프레임워크는 CNN 아키텍처에 적용 가능하며, 각 시각적 설명은 네트워크가 선택한 개념을 인식한 위치와 그 개념의 중요도를 포함합니다.

- **Performance Highlights**: Visual-TCAV는 기존의 TCAV와 비교하여 지역적(local) 및 글로벌(global) 설명 가능성을 모두 제공할 수 있으며, 실험을 통해 그 유효성이 확인되었습니다. 본 방법은 CNN 모델의 다양한 레이어에 적용할 수 있으며, 사용자 정의 개념을 통합하여 더 풍부한 설명을 제공합니다.



### Autoregressive Adaptive Hypergraph Transformer for Skeleton-based Activity Recognition (https://arxiv.org/abs/2411.05692)
Comments:
          Accepted to WACV 2025

- **What's New**: Autoregressive Adaptive HyperGraph Transformer (AutoregAd-HGformer) 모델을 제안하여 skeleton-based action recognition에서 효과적인 feature representation을 제공한다는 점이 주목할 만하다.

- **Technical Details**: 모델은 in-phase와 out-phase hypergraph 생성을 위한 고유한 변환기 아키텍처를 포함하며, 이 방법을 통해 skeleton embedding에 대한 다양한 action-dependent feature를 탐색한다. 성능 향상을 위해 hybrid learning (supervised, self-supervised)을 적용하며, spatiotemporal와 channel 차원에서의 action-dependent feature를 강조한다.

- **Performance Highlights**: Extensive experimental results 및 ablation study를 통해, AutoregAd-HGformer 모델이 NTU RGB+D, NTU RGB+D 120 및 NW-UCLA 데이터셋에서 최신 hypergraph 아키텍처보다 우수함을 입증하였다.



### Tell What You Hear From What You See -- Video to Audio Generation Through Tex (https://arxiv.org/abs/2411.05679)
Comments:
          NeurIPS 2024

- **What's New**: 본 연구에서는 비디오와 텍스트 프롬프트를 입력받아 오디오 및 텍스트 설명을 생성하는 다중 모달 생성 프레임워크인 VATT를 제안합니다. 이 모델은 비디오의 맥락을 보완하는 텍스트를 통해 오디오 생성 과정을 세밀하게 조정 가능하게 하며, 비디오에 대한 오디오 캡션을 생성하여 적절한 오디오를 제안할 수 있습니다.

- **Technical Details**: VATT에는 VATT Converter와 VATT Audio라는 두 가지 모듈이 포함되어 있습니다. VATT Converter는 미세 조정된 LLM(대형 언어 모델)으로, 비디오 특징을 LLM 벡터 공간에 매핑하는 프로젝션 레이어를 갖추고 있습니다. VATT Audio는 변환기로, 시각적 프레임 및 선택적 텍스트 프롬프트에서 오디오 토큰을 생성하는 병렬 디코딩을 통해 운영됩니다. 생성된 오디오 토큰은 사전 훈련된 뉴럴 코덱을 통해 웨이브폼으로 변환됩니다.

- **Performance Highlights**: 실험 결과에 따르면, VATT는 기존 방법 대비 경쟁력 있는 성능을 보이며, 오디오 캡션이 제공되지 않을 때에도 향상된 결과를 도출합니다. 오디오 캡션을 제공할 경우 KLD 점수가 1.41로 가장 낮은 성과를 기록했습니다. 주관적 연구에서는 VATT Audio에서 생성된 오디오가 기존 방법에 비해 높은 선호도를 나타냈습니다.



### Online-LoRA: Task-free Online Continual Learning via Low Rank Adaptation (https://arxiv.org/abs/2411.05663)
Comments:
          WACV 2025

- **What's New**: 이 논문에서는 작업 경계가 정의되지 않은 상황에서 온라인 지속 학습(Online Continual Learning, OCL)에서 발생하는 치명적인 망각(Catastrophic Forgetting) 문제를 해결하기 위해 'Online-LoRA'라는 새로운 프레임워크를 제안합니다.

- **Technical Details**: Online-LoRA는 사전 학습된 Vision Transformer (ViT) 모델을 실시간으로 미세 조정(finetuning)하여 리허설 버퍼의 한계와 사전 학습 모델의 성능 이점을 활용합니다. 이 접근법은 온라인 가중치 정규화 전략을 통해 중요한 모델 매개변수를 식별하고 통합하며, 손실 값의 훈련 동력을 활용하여 데이터 분포의 변화 인식을 자동화합니다.

- **Performance Highlights**: 다양한 작업 없는 OCL 시나리오와 기준 데이터 세트(CIFAR-100, ImageNet-R, ImageNet-S, CUB-200, CORe50)에서 진행된 광범위한 실험 결과, Online-LoRA는 다양한 ViT 아키텍처에 robust하게 적응할 수 있으며, 기존의 SOTA 방법들보다 더 나은 성능을 보임을 보여주었습니다.



### Video RWKV:Video Action Recognition Based RWKV (https://arxiv.org/abs/2411.05636)
- **What's New**: 본 논문에서는 기존의 CNN 및 Transformer 기반 비디오 이해 방법의 높은 계산 비용과 장거리 의존성 문제를 해결하기 위해 LSTM CrossRWKV (LCR) 프레임워크를 제안합니다. LCR은 비디오 이해 작업을 위한 공간-시간(spatiotemporal) 표현 학습을 위해 설계되었으며, 새로운 Cross RWKV 게이트를 통해 현재 프레임의 엣지 정보와 과거 특징 간의 상호작용을 원활히 하여 주제를 보다 집중적으로 처리합니다.

- **Technical Details**: LCR은 선형 복잡성을 가지며, 현재 프레임의 엣지 특징과 과거 특징을 결합하여 동적 공간-시간 컨텍스트 모델링을 제공하는 Cross RWKV 게이트를 포함합니다. LCR은 향상된 LSTM 반복 실행 메커니즘을 통해 비디오 처리를 위한 장기 기억을 저장하며, 엣지 정보는 LSTM의 잊기 게이트 역할을 하여 장기 기억을 안내합니다.

- **Performance Highlights**: LSTM-CrossRWKV는 Kinetics-400, Sometingsometing-V2, Jester의 세 가지 데이터 세트에서 뛰어난 성능을 발휘하며, 비디오 이해를 위한 새로운 기준을 제공합니다. 이 모델의 코드는 공개되어 있어 누구나 활용할 수 있습니다.



### SynDroneVision: A Synthetic Dataset for Image-Based Drone Detection (https://arxiv.org/abs/2411.05633)
Comments:
          Accepted at the 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)

- **What's New**: 이번 논문에서는 감시 애플리케이션을 위한 RGB 기반 드론 탐지에 특화된 합성 데이터 세트인 SynDroneVision을 소개하고 있습니다. 합성 데이터를 활용함으로써 실제 데이터 수집의 시간과 비용 부담을 크게 줄일 수 있습니다.

- **Technical Details**: SynDroneVision 데이터 세트는 다양한 배경, 조명 조건, 드론 모델을 포함하여, 깊이 학습(deep learning) 알고리즘을 위한 포괄적인 훈련 기반을 제공합니다. 이 데이터 세트는 YOLO(You Only Look Once) 모델의 최신 버전들과 비교하여 효과성을 평가하였습니다.

- **Performance Highlights**: SynDroneVision은 모델 성능과 견고성에서 주목할 만한 향상을 이끌어냈으며, 실제 데이터 수집의 시간과 비용을 상당히 절감할 수 있음을 보여주었습니다.



### A Two-Step Concept-Based Approach for Enhanced Interpretability and Trust in Skin Lesion Diagnosis (https://arxiv.org/abs/2411.05609)
Comments:
          Preprint submitted for review

- **What's New**: 본 연구에서는 기존 Concept Bottleneck Models (CBMs)의 데이터 주석 부담 및 해석 가능성 부족 문제를 해결하기 위해 새로운 두 단계 접근 방식을 제안합니다. 이 방법은 pretrained Vision Language Model (VLM)과 Large Language Model (LLM)을 활용하여 임상 개념을 자동으로 예측하고 질병 진단을 생성합니다.

- **Technical Details**: 첫 번째 단계에서는 pretrained VLM을 사용하여 임상 개념의 존재를 예측합니다. 두 번째 단계에서는 예측된 개념을 맞춤형 프롬프트에 통합하여 LLM에 질병 진단을 요청합니다. 이 과정은 추가 훈련 없이 이루어지며, 새로운 개념의 추가 시 재훈련이 필요하지 않습니다.

- **Performance Highlights**: 이 방법은 세 가지 피부 병변 데이터셋에서 전통적인 CBMs 및 최첨단 설명 가능한 방법들보다 우수한 성능을 보였습니다. 추가 훈련 없이 몇 개의 주석된 예시만으로 최종 진단 클래스를 제공하여 해석 가능성과 성능을 동시에 개선하였습니다.



### Efficient Audio-Visual Fusion for Video Classification (https://arxiv.org/abs/2411.05603)
Comments:
          CVMP Short Paper

- **What's New**: Attend-Fusion은 비디오 분류 작업에서 오디오( audio )와 비주얼( visual ) 데이터를 효율적으로 융합하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 이 방법은 compact model architecture를 유지하면서 두 가지 모달리티를 동시에 활용하는 문제를 해결합니다. 또한, YouTube-8M 데이터셋에서의 실험을 통해 성능을 검증했습니다.

- **Performance Highlights**: Attend-Fusion은 더욱 큰 baseline 모델들에 비해 모델 복잡성을 크게 줄이면서도 경쟁력 있는 성능을 보여줍니다.



### Predicting Stroke through Retinal Graphs and Multimodal Self-supervised Learning (https://arxiv.org/abs/2411.05597)
Comments:
          Accepted as oral paper at ML-CDS workshop, MICCAI 2024

- **What's New**: 본 연구는 효율적인 망막 이미지 표현과 임상 정보를 결합하여 심혈관 건강에 대한 포괄적인 개요를 포착하는 새로운 접근 방식을 제안합니다. 이는 망막 이미지에서 유도된 혈관 그래프를 사용하여 대규모 다중 모드 데이터를 활용한 신뢰할 수 있는 뇌졸중 예측 모델을 개발하는 것을 포함합니다.

- **Technical Details**: 제안된 프레임워크는 여러 모드의 데이터를 통합하는 대조적 학습(contrastive learning) 방법을 기반으로 하며, 망막 사진(fundus photographs)의 세 가지 모듈을 통해 특징 임베딩을 추출합니다. 이미지 데이터에는 ResNet을 기반으로 한 모듈과 혈관 그래프 표현을 통한 그래프 신경망(Graph Neural Networks, GNNs)이 사용됩니다.

- **Performance Highlights**: 본 프레임워크는 감독 학습(supervised learning)에서 자가 감독 학습(self-supervised learning)으로 전환되는 과정에서 AUROC 점수가 3.78% 향상되었으며, 그래프 수준의 표현 방식이 이미지 인코더(image encoders)보다 우수한 성능을 보여줍니다. 이 결과는 망막 이미지를 활용한 심혈관 질환 예측의 비용 효율적인 개선을 입증합니다.



### Open-set object detection: towards unified problem formulation and benchmarking (https://arxiv.org/abs/2411.05564)
Comments:
          Accepted at ECCV 2024 Workshop: "The 3rd Workshop for Out-of-Distribution Generalization in Computer Vision Foundation Models"

- **What's New**: 이 연구에서는 OpenSet Object Detection(OSOD) 관련 여러 접근 방식의 평가를 통합하기 위한 새로운 벤치마크(OpenImagesRoad)를 소개하고, 기존의 편향된 데이터 세트와 평가 지표를 해결하는 데 초점을 맞추고 있습니다. 이를 통해 OSOD에 대한 명확한 문제 정의와 일관된 평가가 가능해집니다.

- **Technical Details**: 연구에서는 다양한 unknown object detection 접근 방식을 논의하며, VOC와 COCO 데이터 세트를 기반으로 하는 통합된 평가 방법론을 제안합니다. 새로운 OpenImagesRoad 벤치마크는 명확한 계층적 객체 정의와 새로운 평가 지표를 제공합니다. 또한 최근 자가 감독 방식(self-supervised)으로 학습된 Vision Transformers(DINOv2)를 활용하여 pseudo-labeling 기반의 OSOD를 개선하는 OW-DETR++ 모델을 제안합니다.

- **Performance Highlights**: 본 연구에서 제안한 벤치마크에서 state-of-the-art(OSOT) 방법들의 성능을 광범위하게 평가하였으며, OW-DETR++ 모델은 기존 pseudo-labeling 방법들 중에서 가장 우수한 성능을 기록했습니다. 이를 통해 OSOD 전략의 효과 및 경계에 대한 새로운 통찰을 제공합니다.



### Training objective drives the consistency of representational similarity across datasets (https://arxiv.org/abs/2411.05561)
Comments:
          26 pages

- **What's New**: 최근의 기초 모델들이 다운스트림 작업 성능에 따라 공유 표현 공간으로 수렴하고 있다는 새로운 가설인 Platonic Representation Hypothesis를 제시합니다. 이는 데이터 모달리티와 훈련 목표와는 무관하게 발표됩니다.

- **Technical Details**: 연구에서는 CKA와 RSA와 같은 유사도 측정 방법을 사용하여 모델 표현의 일관성을 측정하는 체계적인 방법을 제안했습니다. 실험 결과, 자기 지도 시각 모델이 이미지 분류 모델이나 이미지-텍스트 모델보다 다른 데이터셋에서의 쌍별 유사도를 더 잘 일반화하는 것으로 나타났습니다. 또한, 모델의 과제 행동과 표현 유사성 간의 관계는 데이터셋 의존적이라 밝혀졌습니다.

- **Performance Highlights**: 이 연구를 통해 쌍별 표현 유사도가 모델의 작업 성능 차이와 강한 상관관계를 보임을 발견하였으며, 단일 도메인 데이터셋에서 이러한 경향이 가장 뚜렷하게 나타났습니다.



### A Nerf-Based Color Consistency Method for Remote Sensing Images (https://arxiv.org/abs/2411.05557)
Comments:
          4 pages, 4 figures, The International Geoscience and Remote Sensing Symposium (IGARSS2023)

- **What's New**: 이번 연구에서는 계절, 조명 및 대기 조건에 따라 서로 다른 이미지를 통합할 때 발생하는 문제를 해결하기 위해 NeRF(Neural Radiance Fields) 기반의 색상 일관성 보정 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 다중 뷰 이미지의 특징을 암시적 표현을 통해 엮어 결과 이미지를 생성하고, 이를 통해 새로운 시점의 융합 이미지를 재조명합니다. 실험에는 Superview-1 위성 이미지와 UAV(불특정 비행체) 이미지를 사용하였으며, 이는 큰 범위와 시간 차이가 있는 데이터입니다.

- **Performance Highlights**: 실험 결과, 제안된 방법으로 생성된 합성 이미지는 뛰어난 시각적 효과를 가지며, 이미지 가장자리에서 부드러운 색상 전환을 보여주었습니다.



### DeepArUco++: Improved detection of square fiducial markers in challenging lighting conditions (https://arxiv.org/abs/2411.05552)
- **What's New**: 본 연구에서는 DeepArUco++라는 새로운 딥러닝 기반 프레임워크를 제안하여, 조명이 어려운 환경에서도 fiducial markers를 효과적으로 탐지하고 디코딩하는 방법을 개발했습니다.

- **Technical Details**: DeepArUco++는 Convolutional Neural Networks의 강점을 활용하여, 여러 단계의 Neural Network 모델을 사용하는 파이프라인 구조를 가지고 있습니다. 이 시스템은 마커 탐지, 코너 보정, 마커 디코딩의 세 가지 주요 단계로 구성됩니다. 또한, 다양한 모델을 훈련시키기 위한 합성 데이터 생성 방법과 조명 조건이 어려운 상황에서 테스트할 수 있는 실제 ArUco 마커 데이터셋을 제공합니다.

- **Performance Highlights**: 개발된 방법은 기존의 최신 방법들과 비교했을 때, 성능이 뛰어나며 특히 어려운 조명 조건에서도 경쟁력을 발휘합니다.



### Towards Lifelong Few-Shot Customization of Text-to-Image Diffusion (https://arxiv.org/abs/2411.05544)
- **What's New**: 이번 연구는 Lifelong Few-Shot Diffusion (LFS-Diffusion) 방법을 제안하여 텍스트-이미지 확산 모델을 지속적으로 조정하면서 이전 지식을 보존할 수 있도록 합니다.

- **Technical Details**: 우리는 Relevant Concepts Forgetting (RCF)와 Previous Concepts Forgetting (PCF) 문제를 파악하였고, 이를 해결하기 위해 데이터 없는 knowledge distillation 전략과 In-Context Generation (ICGen) 패러다임을 개발하였습니다. ICGen은 입력 비전 컨텍스트에 기반하여 모델의 성능을 향상시키는 방법입니다.

- **Performance Highlights**: 제안된 방법은 CustomConcept101 및 DreamBooth 데이터셋에서 기존 방법보다 뛰어난 성능을 보여주며, 고품질의 이미지를 생성할 수 있음을 입증하였습니다.



### Alignment of 3D woodblock geometrical models and 2D orthographic projection imag (https://arxiv.org/abs/2411.05524)
- **What's New**: 이번 논문은 베트남의 문화유산 보존을 위한 3D 목판 문자 모델과 2D 정투영 이미지 간의 정합(Registration) 품질을 향상시키는 통합 이미지 처리 알고리즘을 제안합니다. 이 방법은 3D 문자 모델의 평면을 결정하고, 이 평면을 2D 인쇄 이미지 평면과 정렬하기 위한 변환 행렬을 설정하는 과정을 포함합니다.

- **Technical Details**: 제안된 방법은 다음 세 단계로 구성됩니다: 1) 전체 품질의 3D 포인트 클라우드 데이터에서 투영 평면을 식별하여 고해상도 깊이 맵 이미지 생성, 2) 깊이 맵 이미지와 2D 컬러 이미지 정규화, 3) 2D 이진 문자 정렬 알고리즘으로 사용되는 방향 정렬 방법 적용. 이를 통해 문자 모양과 스트로크가 정확하게 위치하는 것을 보장합니다.

- **Performance Highlights**: 실험 결과는 대규모 한자-놈 문자 데이터셋에 대한 정렬 최적화의 중요성을 강조합니다. 제안된 밀도 기반 및 구조 기반 방법의 조합은 향상된 정합 성능을 보여주며, 디지털 유산 보존을 위한 효과적인 정규화 방안을 제공합니다.



### Towards Scalable Foundation Models for Digital Dermatology (https://arxiv.org/abs/2411.05514)
Comments:
          Findings paper presented at Machine Learning for Health (ML4H) symposium 2024, December 15-16, 2024, Vancouver, Canada, 11 pages

- **What's New**: 이번 연구는 피부과(digital dermatology)에서의 데이터 부족 문제를 해결하기 위해 도메인-특화된 기초 모델(domain-specific foundation models)을 활용하는 방안을 제안하고 있습니다. 특히, 240,000장이 넘는 피부과 이미지를 사용하여 self-supervised learning (SSL) 기법을 적용하였고, 훈련된 모델을 공개하여 임상의들이 활용할 수 있도록 하였습니다.

- **Technical Details**: 모델 훈련에는 두 가지 아키텍처, 즉 CNN(Convolutional Neural Network) 및 ViT(Vision Transformer)를 사용하였으며, 각각 ResNet-50과 ViT-Tiny를 채택했습니다. 이 모델들은 임상에서의 제한된 자원 환경에서도 효율적으로 사용할 수 있도록 설계되었습니다. 연구에서는 총 12개의 진단 관련 다운스트림 작업(downstream tasks)을 통해 모델 성능을 평가했습니다.

- **Performance Highlights**: 모델의 성능은 일반적인 목적의 모델보다 우수할 뿐만 아니라, 50배 큰 모델에 근접하는 수준으로, 임상 진단 작업에서의 활용 가능성을 높이고 있습니다. 연구 결과는 리소스가 제한된 환경에서도 적용 가능한 효율적인 모델 개발에 기여할 것으로 기대됩니다.



### FGGP: Fixed-Rate Gradient-First Gradual Pruning (https://arxiv.org/abs/2411.05500)
- **What's New**: 최근 딥러닝 모델의 크기가 증가하고 계산 자원의 수요가 커짐에 따라, 정확도를 유지하면서 신경망을 가지치기(pruning)하는 방법에 대한 관심이 높아지고 있습니다. 본 연구에서는 그라디언트 우선의 크기 선택 전략을 소개하고, 이러한 접근법이 고정비율(subselection criterion) 기준을 통해 높은 성과를 낸다는 것을 보여줍니다.

- **Technical Details**: 이 연구에서는 CIFAR-10 데이터셋을 사용하여 VGG-19와 ResNet-50 네트워크를 기초로 하여 90%, 95%, 98%의 희소성(sparsity)에 대한 가지치기를 수행하였습니다. 제안한 고정비율 그라디언트 우선 가지치기(FGGP) 접근법은 대부분의 실험 설정에서 기존의 최첨단 기술보다 우수한 성능을 보였습니다.

- **Performance Highlights**: FGGP는 타 기술 대비 높은 순위를 기록하며, 경우에 따라 밀집 네트워크의 결과 상한선을 초과하는 성과를 달성하였습니다. 이는 신경망의 파라미터를 가지치기하는 방식에서 단계별 선택 프로세스의 질이 결과에 얼마나 중요한지를 입증합니다.



### Tightly-Coupled, Speed-aided Monocular Visual-Inertial Localization in Topological Map (https://arxiv.org/abs/2411.05497)
- **What's New**: 이 논문은 Topological map을 이용한 차량 속도 보조 단안 경량 시각 관성 로컬라이제이션을 위한 새로운 알고리즘을 제안합니다. 이는 GPS나 LiDAR와 같은 비싼 센서에 의존하는 기존 방법의 한계를 극복하는 것을 목표로 합니다.

- **Technical Details**: 제안된 시스템은 LiDAR 포인트 클라우드로부터 오프라인에서 생성된 topological map을 사용하며, 여기에는 깊이 이미지(depth images), 강도 이미지(intensity images), 그리고 해당 카메라 자세가 포함됩니다. 이 지도는 현재 카메라 이미지와 저장된 topological 이미지 간의 대응 매칭을 통해 실시간 로컬라이제이션에 사용됩니다. 시스템은 Iterated Error State Kalman Filter (IESKF)를 사용하여 포즈 추정(pose estimation)을 최적화하며, 이미지 간의 대응과 차량 속도 측정을 통합하여 정확도를 향상시킵니다.

- **Performance Highlights**: 오픈 데이터셋과 실제 수집 데이터(예: 터널과 같은 도전적인 시나리오)를 사용한 실험 결과는 topological map 생성 및 로컬라이제이션 작업에서 제안된 알고리즘의 우수한 성능을 보여줍니다.



### Improving image synthesis with diffusion-negative sampling (https://arxiv.org/abs/2411.05473)
- **What's New**: 본 논문은 Diffusion Models (DMs)에서의 image generation에서 negative prompt의 중요성과 이를 개선하기 위한 diffusion-negative prompting (DNP) 전략을 제안합니다. DNP는 사용자가 제공한 prompt에 따라 DMs가 이해하는 부정적인 이미지 개념을 시각화할 수 있도록 합니다.

- **Technical Details**: DNP는 diffusion-negative sampling (DNS)을 기반으로 하여 주어진 텍스트 prompt에 대한 least compliant 이미지를 샘플링합니다. 이 과정을 통해 자동적으로 부정적인 prompt n*를 생성할 수 있으며, 이를 DMs에 입력으로 사용하여 이미지 품질을 향상시킵니다.

- **Performance Highlights**: 실험 결과 DNP는 quantitive metrics (예: CLIP scores)와 human evaluatorsによる 주관적인 평가 모두에서 image quality와 prompt compliance를 개선함을 보여주었습니다. DNP는 특히 다양한 DMs과 그 변형들에 대해 높은 성능을 발휘하며, 기존의 부정적인 prompting 방법에 비해 우수한 결과를 도출했습니다.



### POC-SLT: Partial Object Completion with SDF Latent Transformers (https://arxiv.org/abs/2411.05419)
- **What's New**: 이 논문에서는 부분 관측을 통해 3D 형상을 완성하는 새로운 방법을 제안합니다. 이는 Signed Distance Fields (SDF) 상의 잠재 공간에서 작동하는 트랜스포머를 이용하여 이루어집니다.

- **Technical Details**: 제안된 방법은 고해상도 패치로 나누어진 SDF를 활용하고, Variational Autoencoder (VAE)를 통해 학습된 부드러운 잠재 공간 인코딩을 기반으로 합니다. 효율적인 마스크된 오토인코더 트랜스포머를 사용하여 부분 시퀀스를 완전한 형태로 변환합니다.

- **Performance Highlights**: ShapeNet 및 ABC 데이터세트에서 광범위하게 평가된 결과, 제안된 POC-SLT 아키텍처는 여러 최신 기법들과 비교하여 3D 형상 완성에서 질적, 정량적으로 상당한 개선을 보여주었습니다.



### AuthFormer: Adaptive Multimodal biometric authentication transformer for middle-aged and elderly peop (https://arxiv.org/abs/2411.05395)
- **What's New**: 이 논문에서는 노인 사용자에게 적합한 적응형 멀티모달 생체 인증 모델인 AuthFormer를 제안합니다.

- **Technical Details**: AuthFormer 모델은 LUTBIO 멀티모달 생체 데이터베이스를 기반으로 하며, 교차 주의 메커니즘(cross-attention mechanism)과 Gated Residual Network (GRN)를 통합하여 노인의 생리적 변화를 잘 반영할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과 AuthFormer는 99.73%의 정확도를 달성하며, 인코더에서 최적의 성능을 발휘하기 위해 단 두 개의 레이어만을 사용합니다.



### From Transparent to Opaque: Rethinking Neural Implicit Surfaces with $\alpha$-NeuS (https://arxiv.org/abs/2411.05362)
- **What's New**: 이 논문은 $eta$-NeuS라는 새로운 방법을 제안하여 얇고 투명한 객체와 불투명한 객체를 동시에 재구성하는 방법을 소개합니다. 기존의 방법들은 주로 불투명한 표면에만 집중하였으며, 투명한 물체의 복잡한 조명 효과는 해결하지 못했습니다.

- **Technical Details**: α-NeuS는 신경 볼륨 렌더링(neural volumetric rendering) 과정에서 투명 표면이 학습된 거리 필드에서 로컬 극값(local extreme values)을 유도한다는 관찰을 활용합니다. 이 방법은 거리 필드의 절대값을 취하여 전통적인 등표면 추출 알고리즘(marching cubes)의 한계를 극복하고, 투명한 물체와 불투명한 물체의 표면을 동시에 추출할 수 있도록 최적화된 방법을 개발합니다.

- **Performance Highlights**: 실험 결과, α-NeuS는 5개의 실제 장면과 5개의 합성 장면에서 투명하고 불투명한 객체를 효과적으로 재구성했음을 보여주었습니다. 이로써, 제안된 방법은 다양한 현실 세계 장면에서의 유용성과 효과성을 입증합니다.



### Agricultural Landscape Understanding At Country-Sca (https://arxiv.org/abs/2411.05359)
Comments:
          34 pages, 7 tables, 15 figs

- **What's New**: 이 연구는 인도 지역에서 농업 경관을 디지털화하는 혁신적인 접근 방식을 제시합니다. 높은 해상도의 위성 이미지를 사용하여 국가 규모의 다중 클래스 팬옵틱(segmentation) 세분화 결과물을 생성한 최초의 사례입니다.

- **Technical Details**: 연구는 빨강, 초록, 파랑(RGB) 채널의 위성 이미지를 이용한 다중 클래스 팬옵틱(segmentation) 문제로 농업 경관 이해를 공식적으로 모델링합니다. U-Net 아키텍처를 활용하여 개별 필드, 나무, 물체 경계를 픽셀 수준에서 분류합니다.

- **Performance Highlights**: 본 연구는 소규모 농장을 강조하며, 면적 151.7M 헥타르에 걸쳐 수백만 개의 소규모 농장 필드와 수백만 개의 소규모 관개 구조물을 성공적으로 식별했습니다. 또한, 엄격한 현장 검증을 통해 모델의 신뢰성을 보장합니다.



### Enhancing Visual Classification using Comparative Descriptors (https://arxiv.org/abs/2411.05357)
Comments:
          Accepted to WACV 2025. Main paper with 8 pages

- **What's New**: 이 논문에서는 semantically similar classes를 강조하여 시각 분류 성능을 향상시키기 위해 새로운 comparative descriptors 개념을 제안합니다. 이 방법은 subtle differences에 대한 모델의 어려움을 해결하고, 고유한 특성을 명확히 하여 분류 정확성을 높입니다.

- **Technical Details**: 제안된 방법은 두 단계로 이루어져 있습니다: (1) LLM을 사용하여 특정 class에 대한 comparative descriptors를 생성합니다. 이때, semantically similar classes를 미리 식별하고, (2) filtering process를 통해 가장 유사한 descriptors만 보존합니다. 이 과정에서, CLIP embedding space에서의 이미지 임베딩과 가까운 descriptors를 선택하여 성능을 향상시킵니다.

- **Performance Highlights**: 이 접근법은 다양한 데이터셋에서 VLMs, 특히 CLIP의 이미지 분류 성능을 유의미하게 향상시켰으며, interpretability를 보존합니다. 제안된 comparative descriptors와 filtering process를 적용한 결과, top-1 accuracy와 top-5 accuracy 간의 실질적인 차이를 줄이고, subtle inter-class differences를 해결하는 데 성공했습니다.



### A Quality-Centric Framework for Generic Deepfake Detection (https://arxiv.org/abs/2411.05335)
- **What's New**: 본 논문은 딥페이크 탐지에서의 일반화 문제를 해결하기 위해 훈련 데이터의 위조 품질을 활용합니다. 다양한 위조 품질의 딥페이크가 혼합된 데이터로 탐지기를 교육하면 탐지기의 일반화 성능이 저하될 수 있다는 점을 지적합니다. 이를 해결하기 위해 새로운 품질 중심의 프레임워크를 제안하고, 이를 통해 저품질 데이터를 증강하는 방법과 학습 속도를 조절하는 전략을 구현합니다.

- **Technical Details**: 제안하는 프레임워크는 품질 평가기(Quality Evaluator), 저품질 데이터 증강 모듈, 그리고 학습 속도 조절 전략으로 구성됩니다. 위조 품질 점수(Forgery Quality Score, FQS)를 정적으로(예: ArcFace를 사용) 그리고 동적으로(모델의 피드백 사용) 평가하여 얻어진 FQS를 기반으로 훈련 샘플을 선택합니다. 저품질 샘플에 대해서는 주파수 데이터 증강(Frequency Data Augmentation, FreDA) 기법을 적용하여 위조 흔적을 줄이고 현실감을 개선하는 방법을 활용합니다.

- **Performance Highlights**: 실험 결과, 제안하는 방법은 다양한 딥페이크 탐지기의 범위 내 및 교차 데이터셋 성능을 크게 향상시킬 수 있음이 입증되었습니다. 여러 유명한 평가 데이터셋에서 기존 방법보다 약 10% 성능 개선을 달성하며, 손쉬운 샘플부터 어려운 샘플로 단계적으로 학습하도록 모델을 유도하는 커리큘럼 학습(curriculum learning)을 효과적으로 적용하였습니다.



### A Real-time Face Mask Detection and Social Distancing System for COVID-19 using Attention-InceptionV3 Mod (https://arxiv.org/abs/2411.05312)
- **What's New**: COVID-19의 전파를 최소화하기 위해, 얼굴 마스크 착용 및 6피트(약 1.83미터) 거리 유지 상태를 점검할 수 있는 시스템을 개발했습니다.

- **Technical Details**: 이 시스템은 커스터마이즈된 Attention-Inceptionv3 모델을 사용하여 마스크 착용 여부와 거리 유지 여부를 식별합니다. 두 가지 데이터셋을 사용하여 총 10,800장의 이미지(마스크 착용과 미착용 포함)를 학습하였습니다.

- **Performance Highlights**: 학습 정확도는 98%에 도달하였고, 검증 정확도는 99.5%였습니다. 시스템의 정밀도는 약 98.2%이며, 초당 프레임 속도(FPS)는 25.0입니다. 이를 통해 고위험 지역을 효과적으로 식별할 수 있습니다.



### ZOPP: A Framework of Zero-shot Offboard Panoptic Perception for Autonomous Driving (https://arxiv.org/abs/2411.05311)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 이번 논문에서는 자율 주행(AD) 장면을 위한 새로운 다중 모드 제로샷 오프보드 팬옵틱 지각(ZOPP) 프레임워크를 제안합니다. ZOPP는 비전 기초 모델(vision foundation models)의 능력과 포인트 클라우드(point clouds)에서 유도된 3D 표현을 통합하여 인간 수준의 인식 능력을 목표로 하며, 다양한 인식 작업을 지원하는 통일된 프레임워크를 제공합니다.

- **Technical Details**: ZOPP는 SAM-Track을 다중 시점 이미지에 확장하여 2D 감지를 통해 객체 추적 및 인스턴스 분할을 수행합니다. 포인트 클라우드와 다중 시점 이미지 간의 정렬된 대응을 기반으로 하여 각 3D 포인트에 대한 강력한 의미 및 인스턴스 분할을 제공합니다. 또한, ZOPP는 신경 렌더링(neural rendering)을 사용하여 복원된 장면에서 3D 점유 예측을 수행합니다. 최종적으로 4D 점유 흐름을 출력합니다.

- **Performance Highlights**: ZOPP는 Waymo 오픈 데이터셋에서 다양한 인식 작업에 대해 실험을 수행하였으며, 2D/3D 의미 및 팬옵틱 분할, 2D/3D 감지 및 추적, 4D 점유 흐름 예측을 포함하여 우수한 성능을 입증했습니다. 특히 ZOPP는 소규모 및 원거리 객체에 대한 개방형 감지 기능을 통합하여 중요한 의미를 가집니다.



### Revisiting Network Perturbation for Semi-Supervised Semantic Segmentation (https://arxiv.org/abs/2411.05307)
Comments:
          Accepted by PRCV2024

- **What's New**: 이번 연구에서는 반지도 식별(Semi-supervised semantic segmentation, SSS)에서 네트워크 perturbation (perturbation)을 효과적으로 통합하는 새로운 접근법인 MLPMatch를 제안합니다.

- **Technical Details**: MLPMatch는 Deep Neural Network (DNN)의 특정 레이어를 무작위로 비활성화(deactivate)하여 네트워크 perturbation을 수행하고, 이는 단일 네트워크에서 쉽게 구현될 수 있습니다. 또한, 레이블이 있는 데이터와 레이블이 없는 데이터 모두에 대해 volatile learning 과정을 도입하여 성능을 극대화합니다.

- **Performance Highlights**: MLPMatch는 Pascal VOC와 Cityscapes 데이터셋에서 최첨단(State-of-the-art) 성능을 달성하였으며, 기존의 방법들과 비교하여 더욱 효율적이고 간단한 접근법으로 평가받고 있습니다.



### SimpleBEV: Improved LiDAR-Camera Fusion Architecture for 3D Object Detection (https://arxiv.org/abs/2411.05292)
- **What's New**: 본 연구에서는 LiDAR(라이더)와 카메라 정보를 통합하여 자율주행 시스템의 3D 객체 탐지 성능을 향상시키는 단순하면서도 효과적인 새로운 융합 프레임워크인 SimpleBEV를 제안합니다.

- **Technical Details**: SimpleBEV는 BEV(Bird's Eye View) 기반으로 LiDAR와 카메라 기능을 통합합니다. 카메라 기반 깊이 추정을 위해 2단계 캐스케이드 네트워크를 사용하고, LiDAR 포인트에서 파생된 깊이 정보를 통해 깊이 결과를 수정합니다. 3D 객체 탐지를 위한 보조 분기를 도입하여 카메라 정보의 활용성을 높이고, 멀티 스케일 희소 합성곱 특징을 융합하여 LiDAR 특징 추출기를 개선합니다.

- **Performance Highlights**: 실험 결과, 제안하는 방법은 nuScenes 데이터 세트에서 77.6%의 NDS 정확도를 달성하였으며, 이는 3D 객체 탐지 분야에서 타의 추종을 불허하는 성능을 보여줍니다. 모델 앙상블과 테스트 타임 증강을 통해 최고의 NDS 점수를 기록했습니다.



### Cancer-Net SCa-Synth: An Open Access Synthetically Generated 2D Skin Lesion Dataset for Skin Cancer Classification (https://arxiv.org/abs/2411.05269)
- **What's New**: 미국에서 피부암은 가장 흔하게 진단되는 암으로, 조기에 발견하지 않으면 심각한 합병증의 위험이 있어 주요 공공 건강 문제로 인식되고 있습니다. 최신 연구는 피부암 분류를 위한 합성 데이터셋 Cancer-Net SCa-Synth를 공개하며, 이는 Stable Diffusion과 DreamBooth 기술을 활용하여 제작되었습니다.

- **Technical Details**: Cancer-Net SCa-Synth는 피부암 분류를 위한 2D 피부 병변 데이터셋으로, 총 10,000개의 이미지를 포함하고 있습니다. 이 데이터셋은 ISIC 2020 테스트 세트에 대한 학습을 통해 얻은 성능 향상을 증명합니다. Stable Diffusion 모델과 DreamBooth 트레이너를 이용해 두 가지 피부암 클래스에 대해 각각 학습이 이루어졌으며, 여기에서 합성 이미지는 300개의 무작위 샘플을 기반으로 생성되었습니다.

- **Performance Highlights**: Cancer-Net SCa-Synth와 ISIC 2020 훈련 세트를 결합하여 훈련했을 때, 공개 점수에서 0.09, 개인 점수에서 0.04 이상의 성능 향상이 발견되었습니다. 이는 단독으로만 사용된 데이터셋들로부터 훈련한 경우보다 성능이 향상된 결과입니다.



### Decoding Report Generators: A Cyclic Vision-Language Adapter for Counterfactual Explanations (https://arxiv.org/abs/2411.05261)
- **What's New**: 보고서 생성 모델에서 생성된 텍스트의 해석 가능성을 향상시키기 위한 혁신적인 접근 방식이 소개되었다. 이 방법은 사이클 텍스트 조작(cyclic text manipulation)과 시각적 비교(visual comparison)를 활용하여 원본 콘텐츠의 특징을 식별하고 설명한다.

- **Technical Details**: 새롭게 제안된 접근 방식은 Cyclic Vision-Language Adapters (CVLA)를 활용하여 설명의 생성 및 이미지 편집을 수행한다. counterfactual explanations를 통해 원본 이미지와 대조하여 수정된 이미지와의 비교를 통해 해석을 제공하며, 이는 모델에 구애받지 않는 방식으로 이루어진다.

- **Performance Highlights**: 이 연구의 방법론은 다양한 현재 보고서 생성 모델에서 적용 가능하며, 생성된 보고서의 신뢰성을 평가하는 데 기여할 것으로 기대된다.



### Hierarchical Visual Feature Aggregation for OCR-Free Document Understanding (https://arxiv.org/abs/2411.05254)
Comments:
          NeurIPS 2024

- **What's New**: 이 논문은 OCR(Optical Character Recognition) 없이 문서를 이해하는 새로운 프레임워크를 제안합니다. 이 프레임워크는 미리 훈련된 다중 모달 대형 언어 모델(MLLMs)을 기반으로 하며, 여러 글꼴 크기를 효과적으로 처리하기 위해 다중 스케일 비주얼 기능을 활용합니다.

- **Technical Details**: 제안된 방법은 계층적 비주얼 특징 집계(Hierarchical Visual Feature Aggregation, HVFA) 모듈을 도입하여 입력 토큰 수를 줄이고, 입력 문서 이미지의 크기에 따라 시각적 특징을 조정하여 정보 손실과 효율성 간의 균형을 맞춥니다. 또한, 입력 텍스트의 상대 위치를 예측하는 새로운 지침 조정 작업을 통해 모델의 텍스트 읽기 능력을 향상시킵니다.

- **Performance Highlights**: 종합적인 실험 결과, 제안된 접근 방식은 다양한 문서 이해 작업에서 기존 OCR 기반 모델보다 뛰어난 성능을 보여주었습니다. 이 연구는 다양한 문서 이미지 크기와 글꼴을 효과적으로 처리할 수 있는 가능성을 증명하고 있습니다.



### Breaking The Ice: Video Segmentation for Close-Range Ice-Covered Waters (https://arxiv.org/abs/2411.05225)
- **What's New**: 이 논문에서는 북극 해양의 급격한 얼음 감소 문제를 해결하기 위해 자동화된 데이터 기반의 탐색 솔루션을 제안합니다. 특히 선박에서 수집한 광학 데이터로 얼음 상태를 평가하는 기계 학습 접근 방식을 채택하였습니다.

- **Technical Details**: 이 연구에서는 946개의 이미지로 구성된 세밀하게 주석이 달린 데이터셋을 소개하고, 반수동(region-based) 주석 기술을 사용했습니다. 제안된 비디오 세분화 모델인 UPerFlow는 SegFlow 아키텍처를 발전시켜, 6채널 ResNet 인코더 및 각 이미지에 대해 두 개의 UPerNet 기반 세분화 디코더를 포함하고 있습니다. 또한, PWCNet을 광학 흐름 인코더로 사용하며, 양방향 흐름 특성을 통합하는 교차 연결(cross-connections)을 적용하여 잠재 정보의 손실 없이 처리합니다.

- **Performance Highlights**: 제안된 아키텍처는 오클루전(occluded) 영역에서 기준 이미지 세분화 네트워크보다 평균 38% 더 우수한 성능을 보이며, 어려운 북극 환경에서 비디오 세분화의 견고성을 입증합니다.



### Generalizable Single-Source Cross-modality Medical Image Segmentation via Invariant Causal Mechanisms (https://arxiv.org/abs/2411.05223)
Comments:
          WACV 2025

- **What's New**: 이 논문에서는 단일 소스 도메인에서 학습한 모델이 보이지 않는 타겟 도메인에서 잘 일반화될 수 있도록 하는 단일 소스 도메인 일반화(SDG)를 제안하며, 의학 이미지 분할의 교차 모달리티 상황에 맞춘 방법론을 소개합니다.

- **Technical Details**: 본 연구에서는 직관적으로 다양한 이미징 스타일을 시뮬레이션하기 위해 통제된 diffusion models(DMs)를 활용하는 방법을 제안하며, 'intervention-augmentation equivariant' 원칙에 기초하여 다차원 스타일 변수를 포괄적으로 변화시킵니다. 이를 통해 의료 이미지 분할에서 교차 모달리티 문제를 해결하는 접근 방식을 제시합니다.

- **Performance Highlights**: 본 논문에서 제안하는 접근법은 세 가지 다른 해부학적 구조와 이미징 모달리티를 테스트한 결과, 기존의 SDG 방법들보다 일관되게 우수한 성능을 발휘하며, 이는 의료 영상 기술에 실질적인 기여를 할 것으로 기대됩니다.



### Don't Look Twice: Faster Video Transformers with Run-Length Tokenization (https://arxiv.org/abs/2411.05222)
Comments:
          16 pages, 6 figures. Accepted to NeurIPS 2024 (spotlight)

- **What's New**: 이번 논문에서는 비디오 트랜스포머의 처리 속도를 높이는 새로운 방법인 Run-Length Tokenization (RLT)을 제안합니다. RLT는 반복적으로 나타나는 패치를 효율적으로 찾아 제거하고, 이를 단일 패치와 위치 인코딩으로 대체하는 방식으로 동작합니다. 이는 기존의 방법들이 가진 조정 필요성이나 성능 손실을 피하면서, 토큰 수를 줄이는 혁신적인 접근입니다.

- **Technical Details**: RLT는 비디오의 스패치(Spatiotemporal) 패치를 표기하는 기존 방식 대신, 시간에 따라 반복되는 패치의 연속(run)을 찾아 이를 제거한 후, 남은 토큰의 변동 길이를 나타내는 정보를 추가합니다. 이는 데이터 압축을 위한 Run-Length Encoding에서 영감을 받았으며, 훈련 없이도 모델의 처리량을 35% 증가시키고, 정확도는 0.1% 떨어뜨리는 수준입니다.

- **Performance Highlights**: RLT를 사용하면 비디오 트랜스포머의 훈련 시간을 30% 단축시키면서도 성능은 기준 모델과 동일하게 유지됩니다. 또한, 훈련 시 30 FPS에서 100% 이상의 속도 향상을 이뤄내고, 긴 비디오 데이터셋에서는 토큰 수를 최대 80%까지 줄일 수 있습니다.



### Anticipatory Understanding of Resilient Agriculture to Clima (https://arxiv.org/abs/2411.05219)
- **What's New**: 이번 연구에서는 기후 변화와 지정학적 사건으로 인해 증가하는 식량 불안정 문제를 해결하기 위해 원격 감지(geospatial sensing), 딥 러닝(deep learning), 작물 수확량 모델링(crop yield modeling) 및 인과 모델링(causal modeling)을 결합한 식량 안전성 핫스팟 발굴 프레임워크를 제안합니다.

- **Technical Details**: 북인도의 밀 생산 중심지에서 원격 감지 및 딥 러닝을 이용하여 밀 농장 식별을 위한 정량적 분석을 제공합니다. 기후 변화에 따른 작물 수확량의 영향을 WOFOST 도구를 통해 모델링하고, 식량 배급 시스템의 주요 요인을 파악합니다. 또한, 시스템 다이내믹스 모델을 활용하여 식량 불안정성을 식별하는 방법을 제안합니다.

- **Performance Highlights**: 이 연구의 결과로, 북인도의 밀 농장을 효과적으로 식별하고, 기후 시나리오에 따른 작물 생산을 예측할 수 있는 시스템을 개발하였습니다. 이는 인도 및 다른 지역의 식량 시스템의 회복력을 강화하는 데 기여할 수 있는 기초 자료를 제공합니다.



### Interpretable Measurement of CNN Deep Feature Density using Copula and the Generalized Characteristic Function (https://arxiv.org/abs/2411.05183)
- **What's New**: 이 논문에서는 Convolutional Neural Networks (CNN)의 깊은 특성의 확률 밀도 함수(Probability Density Function, PDF)를 측정하기 위한 새로운 경험적 접근 방식을 제안합니다. 이 접근 방식은 CNN의 특징 이해와 이상 탐지(anomaly detection) 개선에 기여할 수 있습니다.

- **Technical Details**: 이 연구는 copula 분석(copula analysis)과 직교 모멘트 방법(Method of Orthogonal Moments, MOM)을 결합하여 다변량 깊은 특성 PDF의 일반화된 특징 함수(Generalized Characteristic Function, GCF)를 직접 측정합니다. 연구 결과, CNN의 비음수 깊은 특징들은 Gaussian 분포로 잘 근사되지 않으며, 네트워크가 깊어질수록 이러한 특징들은 지수 분포에 점점 가까워집니다. 또한, 깊은 특징들은 깊이가 증가할수록 독립적이 되어 가지만, 극단값 표현 사이에서는 강한 의존성(상관관계 혹은 반상관관계)을 보이는 것으로 나타났습니다.

- **Performance Highlights**: 이 연구는 CNN 특징의 확률 밀도를 측정하는 데 있어 기존의 가정(parametric assumptions)이나 선형성 가정을 사용하지 않으면서도 높은 정확도를 유지할 수 있는 방법론을 제시합니다. 또한, 극단값에 대해 높은 상관관계를 보이는 특징들이 컴퓨터 비전 탐지의 중요한 신호이며, 이는 앞으로의 특징 밀도 분석 방법론에 기초자료를 제공할 것으로 기대됩니다.



### Precision or Recall? An Analysis of Image Captions for Training Text-to-Image Generation Mod (https://arxiv.org/abs/2411.05079)
Comments:
          EMNLP 2024 Findings. Code: this https URL

- **What's New**: 본 논문은 텍스트-이미지 모델 훈련에서 캡션(기술명령)의 정밀도(precision)와 재현율(recall)의 중요성을 분석하고, 합성(합성된) 캡션을 생성하기 위해 Large Vision Language Models(LVLMs)를 활용한 새로운 접근 방식을 소개합니다.

- **Technical Details**: 연구에서는 Dense Caption Dataset을 사용하여 이미지 캡션의 정밀도와 재현율이 텍스트-이미지(T2I) 모델의 조합 능력에 미치는 영향을 체계적으로 평가하였습니다. Human-annotated 캡션과 LVLM으로 생성된 캡션을 활용하여 T2I 모델을 훈련시키고, 이 모델들이 조합 능력에서 어떤 성능을 보이는지 분석하였습니다.

- **Performance Highlights**: 이 연구의 결과는 캡션의 정밀도가 성능에 더 큰 영향을 미친다는 것을 강조합니다. LVLM을 사용해 생성된 합성 캡션으로 훈련된 T2I 모델은 Human-annotated 캡션으로 훈련된 모델과 유사한 성능을 보였으며, 특히 정밀도가 중요한 역할을 한다는 결론에 도달하였습니다.



### Multi-language Video Subtitle Dataset for Image-based Text Recognition (https://arxiv.org/abs/2411.05043)
Comments:
          12 pages, 5 figures

- **What's New**: 다중 언어 비디오 자막 데이터셋(Multi-language Video Subtitle Dataset)은 텍스트 인식을 지원하기 위해 설계된 포괄적인 컬렉션입니다. 이 데이터셋은 온라인 플랫폼에서 출처를 찾은 24개의 비디오에서 추출한 4,224개의 자막 이미지를 포함하고 있습니다.

- **Technical Details**: 데이터셋에는 태국어 자음, 모음, 성조 기호, 구두점, 숫자, 로마 문자 및 아라비아 숫자를 포함한 157개의 고유 문자가 포함되어 있어 복잡한 배경에서 텍스트 인식의 도전 과제를 해결하기 위한 자원으로 활용될 수 있습니다. 또한, 이미지 내 텍스트의 길이, 글꼴 및 배치에서의 다양성은 심층 학습 모델의 개발 및 평가를 위한 가치 있는 자원을 제공합니다.

- **Performance Highlights**: 이 데이터셋은 비디오 콘텐츠에서의 정확한 텍스트 전사(text transcription)를 촉진하며, 텍스트 인식 시스템의 계산 효율성을 개선하는 기초를 제공합니다. 따라서 인공지능(artificial intelligence), 심층 학습(deep learning), 컴퓨터 비전(computer vision), 패턴 인식(pattern recognition) 등 다양한 컴퓨터 과학 분야에서 연구 및 혁신의 발전을 이끌 잠재력을 가지고 있습니다.



### Ultrasound-Based AI for COVID-19 Detection: A Comprehensive Review of Public and Private Lung Ultrasound Datasets and Studies (https://arxiv.org/abs/2411.05029)
- **What's New**: 이 문서는 COVID-19의 진단 및 예측을 위한 AI 기반 연구에서 폐 초음파(ultrasound) 기술의 활용을 다룬 포괄적인 검토입니다. 저자들은 공공 및 민간 LUS 데이터셋을 정리하고, 분석된 연구 결과를 표 형식으로 제시하였습니다. 이 연구는 특히 아동 및 임산부에 대한 진단 임상에 큰 잠재력을 보여줍니다.

- **Technical Details**: 저자들은 총 60개의 관련 논문을 검토하였으며, 그 중 41개는 공개 데이터셋을, 나머지 19개는 개인 데이터셋을 사용했습니다. AI 모델, 데이터 전처리 방법(data preprocessing methods), 교차 검증 기술(cross-validation techniques), 평가 메트릭스(evaluation metrics) 등을 체계적으로 분석하고 정리하였습니다. 또한, AI 기반 COVID-19 검출 및 분석에서 사용되는 선호되는 초음파 전처리 및 증강 기법(augmentation techniques)에 대한 리뷰도 포함되어 있습니다.

- **Performance Highlights**: COVID-19를 검출하기 위한 초음파 기반 AI 연구는 임상 현장에서 큰 잠재력을 보여주며, 특히 아동과 임산부를 위한 진단에 유용하게 활용될 수 있습니다. 이 연구는 인공지능 모델의 성능을 개선하기 위한 다양한 접근 방식과 데이터셋의 활용을 비교할 수 있는 유용한 자원을 제공합니다.



### Leveraging Transfer Learning and Multiple Instance Learning for HER2 Automatic Scoring of H\&E Whole Slide Images (https://arxiv.org/abs/2411.05028)
- **What's New**: 이번 연구에서는 유방암 환자의 HER2 스코어링을 위한 자동화 모델 개발이 딥러닝 전이 학습(transfer learning)과 다중 인스턴스 학습(multiple-instance learning, MIL)의 조합을 통해 어떻게 향상될 수 있는지를 보여주었습니다. 특히, Hematoxylin and Eosin (H&E) 이미지에서 사전 훈련된 모델(embedding models)이 다른 유형의 이미지보다 일관되게 높은 성능을 나타냈습니다.

- **Technical Details**: 연구는 H&E 이미지, 면역 조직 화학(Immunohistochemistry, IHC) 이미지 및 비의료 이미지를 소스 데이터셋으로 설정하여 각각의 분류 작업을 시행했습니다. 또한, MIL 프레임워크와 주의(attention) 메커니즘을 활용하여 패치(patch) 기반의 주목도를 통해 HER2 양성 영역을 시각적으로 강조할 수 있는 방법론을 제안했습니다. AlexNet CNN 구조를 사용하여 각 소스 작업에서 사전 훈련을 실시하였고, 이를 통해 생성된 피쳐 벡터(feature vector)는 주의 레이어(attention layer)로 전달되어 가방(bag) 수준의 피쳐 벡터가 작성되었습니다.

- **Performance Highlights**: 연구 결과, H&E 이미지를 기반으로 사전 훈련된 모델은 평균 AUC-ROC 값이 0.622로 나타났으며, HER2 스코어에 대해 $0.59-0.80$의 성능을 보이는 것으로 확인되었습니다. 이 연구는 세 가지 다양한 소스 타입에서의 사전 훈련이 모델의 성능에 미치는 영향을 정량화하고, K개의 훈련 세트를 통해 각각의 모델 성능 변동을 계산하여 95% 신뢰 구간을 산출했습니다.



### Generative Artificial Intelligence Meets Synthetic Aperture Radar: A Survey (https://arxiv.org/abs/2411.05027)
- **What's New**: 이 논문은 SAR(Synthetic Aperture Radar) 이미지 해석의 생성적 인공지능(Generative AI, GenAI) 기술 활용 가능성을 탐구하며, 이 두 분야 간의 접목을 체계적으로 분석합니다.

- **Technical Details**: 이 연구에서는 SAR 데이터의 수량 및 품질 문제를 해결하기 위해, 최신 GenAI 모델을 사용한 데이터 생성 기반 애플리케이션을 분석하고, SAR 관련 모델의 기본 구조와 변형을 검토합니다. 특히, GenAI와 해석 가능한 모델을 결합한 하이브리드 모델링 방법을 제안합니다.

- **Performance Highlights**: SAR 이미지 해석에 있어 많은 제한이 있지만, GenAI 기술은 고품질 및 다양한 SAR 데이터 생성 가능성을 열어주며, multi-view SAR 이미지 생성, optical-to-SAR 번역, SAR 이미지 조합과 같은 특정 응용 사례를 중점적으로 다룹니다.



### ASL STEM Wiki: Dataset and Benchmark for Interpreting STEM Articles (https://arxiv.org/abs/2411.05783)
Comments:
          Accepted to EMNLP 2024

- **What's New**: ASL STEM Wiki는 영어로 된 254개의 Wikipedia 기사를 아메리칸 수화(ASL)로 해석한 병렬 말뭉치로, 청각 장애인(DHH) 학생들이 STEM 교육을 더 쉽게 접근할 수 있도록 돕기 위한 첫 번째 연속 수화 데이터셋입니다.

- **Technical Details**: 이 데이터셋은 37명의 인증된 통역사가 ASL로 해석한 64,266개의 문장과 300시간 이상의 ASL 비디오로 구성되어 있으며, STEM 관련 5개 주제에 걸쳐 진행되었습니다. 데이터셋은 일반적인 영어 문장을 ASL 비디오와 매칭하여 수화 연산 모델에 새로운 도전과제를 제시합니다.

- **Performance Highlights**: 이 논문에서 제안한 모델은 STEM 콘텐츠의 손가락 철자가 자주 사용되는 문제를 해결하고자 하며, 손가락 철자 탐지 정확도를 47% 개선하는 contrastive learning 기법을 사용하였습니다. 결과적으로, ASL 교육 자료 접근성을 높이고, 기술적인 ASL 수화 사용을 촉진하는 도구 개발을 위한 도약의 기회를 제공합니다.



### Sketched Equivariant Imaging Regularization and Deep Internal Learning for Inverse Problems (https://arxiv.org/abs/2411.05771)
- **What's New**: 이 논문에서는 기존의 Equivariant Imaging (EI) 기반 무감독 훈련 방식의 비효율성을 줄이기 위해 스케치된 EI 정규화 기법을 제안합니다. 이 기법은 무작위 스케치 기술을 활용하여 고차원 응용에서의 계산 효율성을 개선합니다.

- **Technical Details**: 제안된 Sketched Equivariant Deep Image Prior (DIP) 프레임워크는 단일 이미지 기반 및 작업 적응형 복원에 효율적으로 적용할 수 있으며, EI 정규화의 차원 축소를 통한 가속화를 통해 계산 성능을 대폭 향상시킵니다.

- **Performance Highlights**: X-ray CT 이미지 복원 작업에서의 연구 결과, 제안된 방법이 기존 EI 기반 방법에 비해 수량적 계산 가속화를 달성하며, 테스트 시 네트워크 적응이 가능함을 보여줍니다.



### End-to-End Navigation with Vision Language Models: Transforming Spatial Reasoning into Question-Answering (https://arxiv.org/abs/2411.05755)
- **What's New**: VLMnav는 Vision-Language Model(VLM)을 활용하여 엔드 투 엔드 기억 내비게이션 정책으로 전환할 수 있는 새로운 프레임워크를 제안합니다. 기존의 분리된 접근 방식 대신 VLM을 사용하여 직접 행동을 선택하며, 특정 탐색 데이터 없이도 제로샷(zero-shot) 방식으로 정책을 구현할 수 있다는 점이 혁신적입니다.

- **Technical Details**: VLMnav는 목표를 언어나 이미지로 입력 받고, RGB-D 이미지와 자세(pose) 정보를 기반으로 로봇의 회전 및 이동 명령을 출력합니다. 이 시스템은 탐색 문제를 VLM이 잘 수행하는 질문 응답 형식으로 변환하여, 탐색과 장애물 회피를 명확하게 이해하도록 설계된 새로운 프롬프트 전략을 사용합니다. 2D voxel mapping을 통해 탐사된 영역과 탐사되지 않은 영역을 구분하여, 충돌 방지 및 탐색을 최적화합니다.

- **Performance Highlights**: VLMnav는 기존의 프롬프트 방식 대비 더 나은 내비게이션 성능을 보였으며, 탐색 퍼포먼스 평가에서 통계적으로 유의미한 결과를 창출했습니다. 다양한 구성 요소에 대한 분해 실험을 통해 설계 결정의 영향을 분석하고, 전반적으로 향상된 내비게이션 효과를 입증했습니다.



### FisherMask: Enhancing Neural Network Labeling Efficiency in Image Classification Using Fisher Information (https://arxiv.org/abs/2411.05752)
- **What's New**: 이 논문에서는 Fisher 정보에 기반한 액티브 러닝 방법인 FisherMask를 제안합니다. FisherMask는 네트워크의 필수 파라미터를 식별하기 위해 Fisher 정보를 활용하여 효과적으로 중요한 샘플을 선택합니다. 이 방법은 방대한 레이블된 데이터에 대한 의존성을 줄이면서도 모델 성능을 유지할 수 있는 전략을 제공합니다.

- **Technical Details**: FisherMask는 Fisher 정보를 이용하여 고유한 네트워크 마스크를 구축하고, 이 마스크는 가장 높은 Fisher 정보 값을 가진 k개의 가중치를 선택하여 형성됩니다. 또한, 성능 평가는 CIFAR-10 및 FashionMNIST와 같은 다양한 데이터셋에서 수행되었으며, 특히 불균형 데이터셋에서 유의미한 성능 향상이 나타났습니다.

- **Performance Highlights**: FisherMask는 기존 최첨단 방법들보다 뛰어난 성능을 보이며, 레이블링 효율성을 크게 향상시킵니다. 이 방법은 모든 데이터셋에서 테스트 되었으며, 액티브 러닝 파이프라인에서 모델의 성능 특성을 더 잘 이해할 수 있는 유용한 통찰력을 제공합니다.



### Scaling Laws for Task-Optimized Models of the Primate Visual Ventral Stream (https://arxiv.org/abs/2411.05712)
Comments:
          9 pages for the main paper, 20 pages in total. 6 main figures and 10 supplementary figures. Code, model weights, and benchmark results can be accessed at this https URL

- **What's New**: 이번 연구는 인공지능 신경망 모델의 확장(scale)이 동물의 시각 처리에서 핵심 객체 인식(코어 오브젝트 인식: CORE)과 신경 반응 패턴과 어떻게 관련되는지를 분석합니다. 600개 이상의 모델을 체계적으로 평가하여 모델 아키텍처와 데이터 세트의 크기에 따른 뇌 정렬(brain alignment)의 영향을 조사했습니다.

- **Technical Details**: 저자들은 다양한 아키텍처의 모델을 훈련하고, V1, V2, V4, IT 및 CORE 행동과 같은 벤치마크에서 모델의 성능을 평가하여 스케일링 법칙(scaling laws)을 도출하였습니다. 결과적으로, 행동 정렬(behavioral alignment)은 계속 개선되는 반면, 뇌 정렬은 포화 상태에 이른다는 중요한 발견이 있었습니다.

- **Performance Highlights**: 작은 모델들은 샘플 수가 적을 때 낮은 정렬 성능을 보였지만, 고차원 시각 영역에서는 더 큰 모델이 뇌 정렬 성능에 있어서 이점을 지닙니다. 저자들은 더 나은 성능을 위해 데이터 샘플에 더 많은 계산 리소스를 할당하는 것이 필요하다고 제안합니다.



### Do Histopathological Foundation Models Eliminate Batch Effects? A Comparative Study (https://arxiv.org/abs/2411.05489)
Comments:
          Accepted to AIM-FM Workshop @ NeurIPS'24

- **What's New**: 이 연구는 최신 histopathological foundation models의 batch effects(배치 효과)를 체계적으로 검토했습니다. 기존 연구에서는 이러한 모델이 데이터 편향에 면역적일 것이라는 가설이 제기되었으나, 본 연구에서는 여전히 병원 고유의 시그니처가 포함되어 있어 모델의 편향된 예측을 초래할 수 있음을 입증했습니다.

- **Technical Details**: 연구에서는 여러 개의 histopathology 데이터셋(TCGA-LUSC-5, CAMELYON16)을 활용하여 foundation models에서 추출한 feature embeddings(특징 임베딩)을 분석했습니다. 또한, 일반적으로 사용되는 stain normalization(염색 정규화) 방법이 배치 효과를 충분히 감소시키지 않음을 확인했습니다. 모델의 성능에 따라 더욱 높은 출처 예측 정확도를 보였습니다.

- **Performance Highlights**: 기존 모델들이 여전히 병원별 시그니처의 영향을 받으며, 이는 다운스트림 예측 작업에 편향을 초래하는 것으로 나타났습니다. 따라서, 이 연구는 medical foundation models의 평가에 대한 새로운 관점을 제시하며, 더 강력한 사전 훈련 전략과 다운스트림 예측기 개발을 위한 길을 열었습니다.



### Comparative Study of Probabilistic Atlas and Deep Learning Approaches for Automatic Brain Tissue Segmentation from MRI Using N4 Bias Field Correction and Anisotropic Diffusion Pre-processing Techniques (https://arxiv.org/abs/2411.05456)
- **What's New**: 이 논문은 MRI 이미지에서 뇌 조직을 자동으로 분할하는 데 대한 최신 연구를 제공합니다. 특히, 전통적인 통계적 방법과 현대의 딥러닝 접근법 간의 비교 분석을 수행하고 N4 Bias Field Correction 및 Anisotropic Diffusion과 같은 전처리 기법을 적용한 다양한 세분화 모델의 성능을 조사합니다.

- **Technical Details**: 연구에서는 Probabilistic ATLAS, U-Net, nnU-Net, LinkNet과 같은 다양한 세분화 모델을 사용하여 IBSR18 데이터셋에서 하얀질 (White Matter, WM), 회색질 (Gray Matter, GM), 및 뇌척수액 (Cerebrospinal Fluid, CSF)을 분할했습니다. nnU-Net 모델이 평균 Dice Coefficient (0.937 ± 0.012)에서 가장 뛰어난 성능을 보였으며, 2D nnU-Net 모델은 평균 Hausdorff Distance (5.005 ± 0.343 mm) 및 평균 Absolute Volumetric Difference (3.695 ± 2.931 mm)에서 가장 낮은 점수를 기록했습니다.

- **Performance Highlights**: 본 연구는 N4 Bias Field Correction 및 Anisotropic Diffusion 전처리 기법과 결합된 nnU-Net 모델의 우수성을 강조합니다. 이 모델은 MRI 데이터에서 뇌 조직의 정확한 분할에 있어 뛰어난 성과를 보여주며, GitHub를 통해 구현된 코드를 공개하였습니다.



### VISTA: Visual Integrated System for Tailored Automation in Math Problem Generation Using LLM (https://arxiv.org/abs/2411.05423)
Comments:
          Accepted at NeurIPS 2024 Workshop on Large Foundation Models for Educational Assessment (FM-Assess)

- **What's New**: 새로운 연구에서는 Large Language Models (LLMs)를 활용하여 수학 교육에서 복잡한 시각적 보조 도구를 자동으로 생성하는 멀티 에이전트 프레임워크를 소개합니다. 이 프레임워크는 수학 문제와 관련된 시각 보조 도구를 정확하고 일관되게 생성하도록 설계되었습니다.

- **Technical Details**: 이 연구에서는 7개의 특수화된 에이전트로 구성된 시스템을 설계하여 문제 생성 및 시각화 작업을 세분화하였습니다. 각 에이전트는 Numeric Calculator, Geometry Validator, Function Validator, Visualizer, Code Executor, Math Question Generator, Math Summarizer로 구성되며, 각 에이전트는 특정 역할에 맞춰 문제를 해결합니다.

- **Performance Highlights**: 시스템은 Geometry와 Function 문제 유형에서 기존의 기본 LLM보다 텍스트의 일관성, 관련성 및 유사성을 크게 향상시켰으며, 수학적 정확성을 유지하는 데 주목할 만한 성과를 보였습니다. 이를 통해 교육자들이 수학 교육에서 시각적 보조 도구를 보다 효과적으로 활용할 수 있게 되었습니다.



### WeatherGFM: Learning A Weather Generalist Foundation Model via In-context Learning (https://arxiv.org/abs/2411.05420)
- **What's New**: 본 논문에서는 WeatherGFM이라는 첫 번째의 일반화된 날씨 기반 모델을 소개합니다. 이 모델은 단일 모델 내에서 다양한 날씨 이해 작업을 통합하여 처리 가능하도록 설계되었습니다.

- **Technical Details**: WeatherGFM은 날씨 이해 작업에 대한 단일화된 표현 및 정의를 처음으로 통합하고, 단일, 다중, 시간 모달리티를 관리하기 위한 날씨 프롬프트 포맷을 고안합니다. 또한, 통합된 날씨 이해 작업의 교육을 위해 시각적 프롬프트 기반의 질문-응답 패러다임을 채택하였습니다.

- **Performance Highlights**: 광범위한 실험 결과, WeatherGFM은 날씨 예보, 초해상도, 날씨 이미지 변환, 후처리 등 최대 10개의 날씨 이해 작업을 효과적으로 처리할 수 있으며, 보지 못한 작업에 대한 일반화 능력을 보여줍니다.



### Advancing Meteorological Forecasting: AI-based Approach to Synoptic Weather Map Analysis (https://arxiv.org/abs/2411.05384)
- **What's New**: 이 연구는 기상 예측의 정확성을 높이기 위한 새로운 전처리 방법 및 합성곱 오토인코더(Convolutional Autoencoder) 모델을 제안합니다. 이 모델은 과거의 기상 패턴과 현재의 대기 조건을 비교 분석하여 기상 예측의 효율성을 향상시킬 수 있는 잠재력을 가지고 있습니다.

- **Technical Details**: 연구는 VQ-VQE와 같은 비지도 학습(un_supervised learning) 모델과 VGG16, VGG19, Xception, InceptionV3, ResNet50과 같은 지도 학습(supervised learning) 모델을 포함합니다. 모델의 성능은 유사성 비교를 위한 지표로 코사인 유사도(cosine similarity)를 사용하여, 과거 기상 패턴을 정확하게 식별할 수 있도록 최적화 과정이 이루어졌습니다.

- **Performance Highlights**: 연구 결과, 제안된 모델은 기존의 사전 학습(pretrained) 모델들과 비교할 때 과거 기상 데이터의 식별에서 우수한 성능을 보였으나, 유사성을 판별하는 데 한계도 존재했습니다. 이 모델은 기상학자들이 정보를 더욱 신속하고 정밀하게 분석할 수 있도록 도와줍니다.



### SASWISE-UE: Segmentation and Synthesis with Interpretable Scalable Ensembles for Uncertainty Estimation (https://arxiv.org/abs/2411.05324)
Comments:
          16 pages, 12 figures, 5 tables

- **What's New**: 이 논문은 의료 딥러닝 모델의 해석 가능성을 향상시키는 효율적인 서브 모델 앙상블 프레임워크를 소개합니다. 이를 통해 모델 출력의 신뢰성을 평가할 수 있는 불확실성 맵을 생성합니다.

- **Technical Details**: SASWISE라는 효율적인 앙상블 방법을 제안하며, U-Net 및 UNETR 모델을 활용하여 CT 체적 분할(segmentation) 및 MR-CT 합성을 위한 데이터셋에서 테스트하였습니다. 모형 가족을 훈련하여 단일 체크포인트에서 다양한 모델을 생성할 수 있는 전략을 개발하였습니다.

- **Performance Highlights**: CT 체적 분할에서 평균 Dice 계수가 0.814, MR-CT 합성에서는 89.43 HU에서 88.17 HU로 개선되었습니다. 손상 및 언더샘플링 데이터에서도 불확실성과 오류 간의 상관관계를 유지하는 강인성을 보여줍니다.



### Rate-aware Compression for NeRF-based Volumetric Video (https://arxiv.org/abs/2411.05322)
Comments:
          Accepted by ACM MM 2024 (Oral)

- **What's New**: 본 논문에서는 3D 볼륨 비디오 기술의 데이터 압축 효율성을 높이기 위해 학습 단계에서 직접적으로 압축된 Neural Radiance Fields (NeRF) 표현을 학습하는 새로운 접근 방식을 제안합니다. 기존의 방법들이 주로 훈련 후 압축을 수행했던 것과 달리, 본 연구는 훈련 단계에서 비트레이트를 추정하고 압축 전략을 통합합니다.

- **Technical Details**: 압축 프레임워크는 두 가지 주요 기술로 구성됩니다. 첫째, 잔여 정보(Residual Information)를 학습하여 이전 프레임을 기반으로 하는 예측 모델링을 통해 NeRF 표현의 엔트로피를 효과적으로 줄입니다. 둘째, 학습 가능한 양자화 단계(Quantization Step)를 포함한 적응형 양자화 전략을 도입하여 다양한 장소와 스케일에서 더 나은 재구성 품질을 유지합니다.

- **Performance Highlights**: 실험 결과, HumanRF 및 ReRF 데이터셋에서 기존의 최첨단 방법인 TeTriRF에 비해 각각 -80% 및 -60%의 BD-rate 감소를 달성하였습니다. 본 접근법은 동일한 재구성 품질에서 80% 이상의 비트 전송률을 절약할 수 있음을 보여줍니다.



### Adaptive Whole-Body PET Image Denoising Using 3D Diffusion Models with ControlN (https://arxiv.org/abs/2411.05302)
- **What's New**: 이번 연구는 Positron Emission Tomography (PET) 이미징을 위한 새로운 3D ControlNet 기반의 denoising 방법을 제안합니다. 이 방법은 다양한 임상 환경에서의 PET 이미지 denoising 작업에 적응할 수 있도록 개발되었습니다.

- **Technical Details**: 3D Denoising Diffusion Probabilistic Model (DDPM)을 대규모 고품질 정상 용량 PET 이미지 데이터셋으로 미리 학습하였으며, 이후 소규모의 페어링된 저용량 및 정상 용량 PET 이미지로 fine-tuning을 진행했습니다. 또한, ControlNet 아키텍처를 이용하여 원시 PET 이미지 공간에서 직접 작동하며, 공간적 제어를 통해 세밀한 조정이 가능한 방식으로 설계되었습니다.

- **Performance Highlights**: 실험 결과 제안된 방법이 기존의 가장 우수한 PET 이미지 denoising 방법들과 비교하여 시각적 품질과 정량적 메트릭 모두에서 성능이 우수함을 보여주었습니다. 이 연구는 다양한 획득 프로토콜에 걸쳐 PET 이미지 denoising에 대한 잠재력을 강조하며, 플러그 앤 플레이 접근 방식으로 알려진 유연성을 제공합니다.



### Image Decomposition: Theory, Numerical Schemes, and Performance Evaluation (https://arxiv.org/abs/2411.05265)
- **What's New**: 이 논문은 구조(structures), 텍스처(textures) 및 노이즈(noise)를 분리할 수 있는 여러 이미지 분해 모델에 대해 설명합니다. Meyer의 연구를 기반으로 한 새로운 세 부분 모델이 소개됩니다.

- **Technical Details**: 논문에서는 조화된 총 변동(total variation) 접근 방식을 다양한 적응형 기능 공간(functional spaces)과 결합한 여러 이미지 분해 모델을 설명합니다. 여기에는 Besov 및 Contourlet 공간과 같은 공간이 포함되며, 특히 주기적(tangible) 오실레이팅 함수 공간이 언급됩니다. 성능 평가를 위해 구조, 텍스처 및 노이즈 참조 이미지를 별도로 생성하여 테스트 이미지를 만들고 분해 알고리즘(output of the decomposition algorithms)의 품질을 평가하기 위한 메트릭(metrics)을 정의합니다.

- **Performance Highlights**: 새로운 세 부분 모델은 contourlet soft thresholding 기반으로 이전 알고리즘보다 향상된 결과를 보여줍니다. 이 논문은 다양한 분해 모델의 성능을 요약하고, 향후 연구 방향에 대해 논의합니다.



### On Erroneous Agreements of CLIP Image Embeddings (https://arxiv.org/abs/2411.05195)
Comments:
          18 pages, 4 figures

- **What's New**: 본 연구에서는 Vision-Language Models (VLMs)의 시각적 추론에서 발생하는 오류의 주 원인이 항상 잘못된 합의(erroneous agreements) 때문이 아님을 보여줍니다. LLaVA-1.5-7B 모델은 CLIP 이미지 인코더를 사용하면서도 퀘리와 관련된 시각적 정보를 추출할 수 있음을 밝혔습니다.

- **Technical Details**: CLIP 이미지 인코더는 시각적으로 구별되는 이미지들이 높은 코사인 유사도(cosine similarity)로 애매하게 인코딩되는 문제를 가지고 있습니다. 그러나 LLaVA-1.5-7B는 이러한 CLIP 이미지 임베딩에서 추출 가능한 정보를 이용해 더 나은 성능을 보입니다. 이를 위해 M3ID(Multi-Modal Mutual-Information Decoding)와 같은 대체 디코딩 알고리즘을 사용하여 시각적 입력에 더 많은 주의를 기울일 수 있게 했습니다.

- **Performance Highlights**: LLaVA-1.5-7B는 What'sUp 벤치마크에서 100%에 가까운 정확도로 작업을 수행하였고, MMVP 벤치마크에서도 CLIP 기반 모델보다 우수한 성능을 나타냈습니다. 전체적으로, CLIP 이미지 인코더를 개선하는 것도 중요하지만, 고정된 이미지를 사용하는 모델에서도 정보를 더 잘 추출하고 활용하는 전략을 적용하는 여지가 남아 있음을 보여주고 있습니다.



### AGE2HIE: Transfer Learning from Brain Age to Predicting Neurocognitive Outcome for Infant Brain Injury (https://arxiv.org/abs/2411.05188)
Comments:
          Submitted to ISBI 2025

- **What's New**: 이번 연구에서는 Hypoxic-Ischemic Encephalopathy (HIE)와 관련된 신경인지 결과를 예측하기 위한 새로운 딥러닝 모델 AGE2HIE를 제안합니다. AGE2HIE는 건강한 뇌 MRI로부터 학습한 지식을 HIE 환자로 전이하여 신경인지 결과를 예측할 수 있도록 설계되었습니다.

- **Technical Details**: AGE2HIE는 다음의 여러 단계에서 작동합니다: (a) 건강한 뇌 MRI를 사용한 뇌 나이 추정에서 HIE 환자의 신경인지 결과 예측으로의 작업 간 전이, (b) 0-97세에서 영아(0-2주)로의 나이 간 전이, (c) 3D T1-weighted MRI로부터 3D 확산 MRI로의 모달리티 간 전이, (d) 건강한 대조군에서 HIE 환자까지의 건강 상태 간 전이를 포함합니다.

- **Performance Highlights**: AGE2HIE는 기존의 방법에 비해 예측 정확성을 3%에서 5%까지 개선할 수 있으며, 이는 다양한 사이트에서의 모델 일반화 또한 포함됩니다. 예를 들어, 크로스 사이트 검증에서 5% 성능 개선이 확인되었습니다.



### PadChest-GR: A Bilingual Chest X-ray Dataset for Grounded Radiology Report Generation (https://arxiv.org/abs/2411.05085)
- **What's New**: 이번 연구에서는 Chest X-ray(CXR) 이미지를 위한 최초의 손으로 주석이 달린 데이터셋인 PadChest-GR(도구화 보고서)의 개발을 발표합니다. 이 데이터셋은 각 개별 발견을 기술하는 문장의 완전한 목록을 제공하여 GRRG(기반 방사선 보고서 생성) 모델 훈련에 기여합니다.

- **Technical Details**: PadChest-GR 데이터셋은 영어와 스페인어로 된 4,555개의 CXR 연구를 포함하고 있으며, 여기에는 3,099개의 비정상 및 1,456개의 정상 케이스가 포함되어 있습니다. 각 긍정 발견 문장은 두 개의 독립적인 바운딩 박스 세트와 관련이 있으며, 발견 유형, 위치, 진행 상태에 대한 범주형 레이블이 함께 제공됩니다. 이는 의료 이미지와 생성된 텍스트의 이해 및 해석을 위한 최초의 수작업으로 구성된 데이터셋입니다.

- **Performance Highlights**: PadChest-GR은 의료 AI 모델의 검증을 돕고, 이해도를 높이며, 임상의와 환자 간의 상호작용을 촉진하는 데 중요한 기초를 제공하고 있습니다. 이 데이터셋은 방사선 이미지를 수집하여 의료 AI 모델의 개발 및 평가에 유용한 자료로 활용될 것입니다.



### EAP4EMSIG -- Experiment Automation Pipeline for Event-Driven Microscopy to Smart Microfluidic Single-Cells Analysis (https://arxiv.org/abs/2411.05030)
Comments:
          Proceedings - 34. Workshop Computational Intelligence

- **What's New**: 본 논문에서는 Microfluidic Live-Cell Imaging (MLCI)에서의 데이터 분석을 자동화하고 실시간 이벤트 분류 정확성을 높이기 위한 Experiment Automation Pipeline for Event-Driven Microscopy to Smart Microfluidic Single-Cells Analysis (EAP4EMSIG)라는 새로운 시스템을 소개합니다.

- **Technical Details**: EAP4EMSIG는 8개의 주요 모듈로 구성되어 있으며, Real-time segmentation 모듈의 초기 제로샷 결과를 제시합니다. 연구에서는 4개의 State-Of-The-Art (SOTA) 분할 방법을 평가하였고, Omnipose가 Panoptic Quality (PQ) 점수 0.9336로 가장 높은 성능을 보여주었으며, Contour Proposal Network (CPN)이 185 ms의 가장 빠른 추론 시간을 기록하였습니다.

- **Performance Highlights**: Omnipose는 PQ 점수 0.9336으로 매우 높은 성능을 보였고, CPN은 0.8575의 PQ 점수와 함께 가장 빠른 추론 시간을 기록했습니다. Segment Anything 모델은 이번 사용 사례에 부적합한 것으로 관찰되었습니다.



### SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models (https://arxiv.org/abs/2411.05007)
Comments:
          Quantization Library: this https URL Inference Engine: this https URL Website: this https URL Demo: this https URL Blog: this https URL

- **What's New**: 본 논문에서는 Diffusion 모델의 메모리 사용량과 지연 시간을 줄이기 위해 weights와 activations를 4비트로 양자화하는 새로운 방법인 SVDQuant를 제안합니다. 이 방법은 기존의 post-training quantization 방법들이 가진 한계를 극복하는 데 중점을 두고 있습니다.

- **Technical Details**: SVDQuant는 outliers를 흡수하기 위해 low-rank branch를 도입합니다. 이 과정에서 activation에서 weight로 outliers를 이동시키고, Singular Value Decomposition (SVD)을 통해 low-rank branch를 운영하여 고정밀도로 quantization을 진행합니다. 또한, Nunchaku라는 inference engine을 통해 low-bit branch와 low-rank branch의 커널을 융합하여 메모리 접근을 최적화합니다.

- **Performance Highlights**: 실험 결과, 12B FLUX.1 모델에 대해 메모리 사용량을 3.5배 줄이고, 4비트 weights만 양자화한 기준에 비해 3.0배의 속도 향상을 달성했습니다. 이 성능은 16GB RTX 4090 GPU에서 수치를 측정했습니다.



New uploads on arXiv(cs.AI)

### LLMs as Method Actors: A Model for Prompt Engineering and Architectur (https://arxiv.org/abs/2411.05778)
- **What's New**: 이 논문은 LLM (Large Language Model) 프롬프트 엔지니어링 및 아키텍처 설계를 위한 정신 모델로 'Method Actors'를 도입합니다. 이 모델에서는 LLM을 배우고, 프롬프트를 대본과 신호로, LLM의 응답을 공연으로 간주합니다.

- **Technical Details**: 새로운 접근 방식인 'Method Actors'는 Connections라는 NYT 단어 퍼즐 게임에서 LLM의 성능을 향상시키기 위해 적용됩니다. 기본 접근 방식(vanilla)에서 27%의 퍼즐을 해결하고, Chain of Thoughts 접근 방식에서는 41%를 해결하는 반면, 'Method Actor' 접근 방식은 86%의 퍼즐을 해결합니다. 또한 OpenAI의 새로운 모델인 o1-preview는 다중 API 호출을 통해 100%의 퍼즐을 해결할 수 있습니다.

- **Performance Highlights**: GPT-4o를 사용한 초기 실험에서, 'Method Actor' 접근 방식은 78%의 퍼즐을 해결하고 41%를 완벽하게 해결합니다. 수정된 접근 방식은 86%의 퍼즐을 해결하고 50%를 완벽하게 해결했습니다. o1-preview의 경우, 'Method Actor' 프롬프트 구조를 적용하면 완벽하게 해결하는 퍼즐 비율이 76%에서 87%로 증가했습니다.



### Solving 7x7 Killall-Go with Seki Databas (https://arxiv.org/abs/2411.05565)
Comments:
          Accepted by the Computers and Games conference (CG 2024)

- **What's New**: 이번 논문은 7x7 Killall-Go 게임의 효율적인 해결을 위한 seki 패턴 인식 기법을 제안합니다. 저자들은 사전 정의된 크기까지 모든 seki 패턴을 열거하고 이들을 seki 테이블에 저장함으로써 게임 해결 과정에서 seki를 더욱 신속하게 인식하는 방법을 발표하였습니다.

- **Technical Details**: 이 연구에서는 seki 패턴을 인식하기 위해 Niu et al.의 알고리즘을 사용하여 7x7 Killall-Go 게임의 해를 개선하는 방식으로 seki 데이터베이스를 생성합니다. 실험 결과, 하루 종일 해결할 수 없었던 게임 위치가 seki 테이블을 이용하여 단 482초 만에 해결되었습니다. 또한, 일반적인 위치에서도 검색 시간과 노드 수가 각각 10%에서 20%까지 개선됨을 확인하였습니다.

- **Performance Highlights**: Killall-Go의 경우, seki 데이터베이스의 추가로 인해 기존보다 훨씬 더 단축된 시간으로 문제를 해결할 수 있으며, 이는 고전적인 게임 문제 해결에서의 효율성을 극대화하는 데 기여합니다. 실제로, seki를 인식함으로써 탐색 깊이를 크게 줄일 수 있음을 보여줍니다.



### Enhancing Cluster Resilience: LLM-agent Based Autonomous Intelligent Cluster Diagnosis System and Evaluation Framework (https://arxiv.org/abs/2411.05349)
Comments:
          10 pages

- **What's New**: 최근 대형 언어 모델(LLMs)과 관련 기술인 Retrieval-Augmented Generation (RAG) 및 Diagram of Thought (DoT)의 발전 덕분에 자율 진단 및 문제 해결이 가능한 지능형 시스템이 개발되었습니다. 이 시스템은 AI 클러스터 내의 문제를 자율적으로 진단하고 해결할 수 있도록 설계되었습니다.

- **Technical Details**: LLM 에이전트 시스템은 클러스터 진단을 위한 전문 지식 기반, 개선된 LLM 알고리즘 및 실제 환경에서의 실용적인 배포 전략을 포함합니다. 이 시스템은 150개의 다양한 질문으로 구성된 벤치마크를 통해 성능을 평가하며, 자동화된 탐지 소프트웨어를 사용하는 기존 방법보다 훨씬 빠르고 정확하게 성능 문제를 감지하고 수정할 수 있습니다.

- **Performance Highlights**: 실험 결과, LLM 에이전트 시스템은 성능 저하를 인지하기 전에 문제를 탐지하고 교정 조치를 취할 수 있는 능력을 갖추고 있어, 엔진 리소스를 더 복잡하고 가치 있는 작업에 집중할 수 있도록 합니다. 예를 들어, 한 GPU의 주파수가 낮게 조정된 상황에서 이 시스템은 몇 분 이내에 문제를 해결하는 반면, 전통적인 방법은 한 시간 이상 소요될 수 있습니다.



### LLM-PySC2: Starcraft II learning environment for Large Language Models (https://arxiv.org/abs/2411.05348)
- **What's New**: LLM-PySC2는 StarCraft II Learning Environment에서 파생된 새로운 환경으로, 대형 언어 모델(LLM)을 위한 의사결정 방법론의 연구를 지원합니다. 이 환경은 StarCraft II의 전체 액션 스페이스, 멀티 모달 옵저베이션 인터페이스, 구조화된 게임 지식 데이터베이스를 제공하며, 다양한 LLM과 무결하게 연결되어 있습니다.

- **Technical Details**: LLM-PySC2는 PySC2 모듈을 기반으로 하며, 에이전트에게 포괄적인 관찰 정보와 확장된 액션 스페이스를 제공합니다. 이 환경은 멀티 에이전트 연구를 지원하기 위해 포인트 투 포인트 및 도메인 통신이 가능한 다중 에이전트 프레임워크를 구축했습니다.

- **Performance Highlights**: 실험 결과, LLM의 적절한 파라미터가 필요하지만, 추론 능력이 향상된다고 해서 의사 결정을 잘하는 것은 아닙니다. LLM이 자율적으로 환경에서 학습할 수 있도록 하는 것이 중요하며, LLM-PySC2 환경은 LLM의 교육 방법 개선에 기여할 것으로 기대됩니다.



### A Taxonomy of AgentOps for Enabling Observability of Foundation Model based Agents (https://arxiv.org/abs/2411.05285)
Comments:
          19 pages, 9 figures

- **What's New**: 이 연구는 AI 자동화와 기초 모델 기반 자율 에이전트의 발전에 따른 AgentOps 플랫폼의 필요성을 강조하며, 신뢰할 수 있는 AI 에이전트를 구축하기 위한 관찰성과 추적 가능성의 역할을 탐구합니다.

- **Technical Details**: 본 연구에서는 AgentOps 에코시스템의 관련 도구들을 빠르게 검토하였고, 에이전트 생산 생애 주기 전반에 걸쳐 필요한 관찰성 데이터 및 추적 가능한 아티팩트를 제안합니다. AgentOps 플랫폼은 Agentic 시스템의 운영 관리를 위한 개발, 평가, 테스트, 배포 및 모니터링을 아우르는 DevOps/MLOps 유사 엔드 투 엔드 플랫폼입니다.

- **Performance Highlights**: 현재의 AgentOps 경관을 체계적으로 정리하고, 자율 에이전트 시스템의 신뢰성을 향상시키기 위해 관찰성 및 추적 가능성의 중요성을 강조하여 신뢰성 높은 AI 에이전트를 구축하는 데 기여할 것으로 기대됩니다.



### Minimal Conditions for Beneficial Neighbourhood Search and Local Descen (https://arxiv.org/abs/2411.05263)
- **What's New**: 이 논문은 유용한 지역 검색을 지원하는 이웃의 특성을 조사합니다. 특히 이웃의 지역성(neighbourhood locality)과 최적에 대한 비용 확률의 감소가 탐색이 이웃 사이에서 개선된 솔루션을 찾을 가능성을 더 높인다는 주장을 최초로 증명합니다.

- **Technical Details**: 이 논문에서는 local blind descent이라는 무작위 검색(blind search)과 지역적 하강(local descent)의 조합을 탐색하며, 지역적 하강이 주어진 목표 비용(target cost)보다 낮은 비용에 도달할 기대 단계 수가 무작위 검색보다 적다는 조건을 제시하고 있습니다. 또한, 비용이 최적에 가까워질수록 지역적 하강으로 전환해야 한다는 내용도 포함하고 있습니다.

- **Performance Highlights**: 실험 결과에 따르면, local blind descent는 주어진 목표 비용 t에 대해 초기 비용을 줄여가면서 지역적 하강으로 전환하는 것이 바람직하다는 것을 보여줍니다. 이는 기존의 무작위 검색보다 개선된 성능을 보입니다.



### Alopex: A Computational Framework for Enabling On-Device Function Calls with LLMs (https://arxiv.org/abs/2411.05209)
- **What's New**: Alopex는 Fox LLM을 사용하여 모바일 기기에서 정확한 기능 호출을 가능하게 하는 새로운 프레임워크를 제안합니다. 이 접근 방식은 고품질 훈련 데이터를 생성하는 논리 기반 방법과 기능 호출 데이터의 혼합 전략을 통합하여 성능을 향상시킵니다.

- **Technical Details**: Alopex는 'description-question-output' 포맷을 사용하여 LLM 미세 조정을 최적화하며, 기능 정보 누출의 위험을 감소시킵니다. 또한, 데이터 혼합 전략을 통해 기능 호출 훈련 중의 catastrophic forgetting을 완화합니다.

- **Performance Highlights**: 실험 결과 Alopex는 기능 호출 정확도를 향상시키고 기존 LLM에서 관찰되는 catastrophic forgetting 현상을 유의미하게 줄였습니다. 이는 LLM의 기능 호출 능력을 통합하는 강력한 해결책을 제공합니다.



### Explainable AI through a Democratic Lens: DhondtXAI for Proportional Feature Importance Using the D'Hondt Method (https://arxiv.org/abs/2411.05196)
- **What's New**: 이번 연구는 D'Hondt 투표 원칙을 활용하여 인공지능 모델의 해석 가능성을 증대시키는 DhondtXAI 방법론을 제시합니다. 이 방법론은 기계 학습에서의 feature importance 해석을 쉽게 하기 위해 자원 할당 개념을 적용합니다.

- **Technical Details**: DhondtXAI는 결정 트리 기반 알고리즘을 중심으로 feature importance를 평가하는 구조적 방법론입니다. 사용자가 설정하는 매개변수는 총 투표 수, 의석 수, 제외할 feature, feature 앨리언스, 임계값(Threshold) 등을 포함하며, 이는 유연하고 통찰력 있는 분석을 가능하게 합니다.

- **Performance Highlights**: SHAP와 DhondtXAI를 비교하여 CatBoost 및 XGBoost 모델에서 유방암 및 당뇨병 예측에 대한 feature attribution의 효과성을 평가했습니다. DhondtXAI는 해석 가능성을 증가시키기 위해 연합 형성과 임계값을 활용하며, 이는 AI 모델에서 feature importance를 이해하는 데에 기여할 수 있음을 보여줍니다.



### Discern-XR: An Online Classifier for Metaverse Network Traffic (https://arxiv.org/abs/2411.05184)
- **What's New**: 이 논문에서는 Internet Service Providers (ISP) 및 라우터 제조업체가 Metaverse 서비스의 품질을 향상시키는 데 도움을 주기 위해 전용 Metaverse 네트워크 트래픽 분류기인 Discern-XR를 설계했습니다.

- **Technical Details**: Discern-XR는 세분화된 학습(segmented learning) 기법을 활용하여 Frame Vector Representation (FVR) 알고리즘과 Frame Identification Algorithm (FIA)를 통해 네트워크 데이터에서 중요한 프레임 관련 통계를 추출합니다. 또한, Augmentation, Aggregation, and Retention Online Training (A2R-OT) 알고리즘을 통해 온라인 학습 방법론을 사용한 정확한 분류 모델을 찾습니다.

- **Performance Highlights**: Discern-XR는 최신 분류기보다 7% 우수한 성능을 보이며, 훈련 효율성을 향상시키고 잘못된 음성(rate of false-negatives)을 줄입니다. 결과적으로, Metaverse 네트워크 트래픽 분류의 발전을 이끌며 최첨단 솔루션으로 자리잡았습니다.



### PadChest-GR: A Bilingual Chest X-ray Dataset for Grounded Radiology Report Generation (https://arxiv.org/abs/2411.05085)
- **What's New**: 이번 연구에서는 Chest X-ray(CXR) 이미지를 위한 최초의 손으로 주석이 달린 데이터셋인 PadChest-GR(도구화 보고서)의 개발을 발표합니다. 이 데이터셋은 각 개별 발견을 기술하는 문장의 완전한 목록을 제공하여 GRRG(기반 방사선 보고서 생성) 모델 훈련에 기여합니다.

- **Technical Details**: PadChest-GR 데이터셋은 영어와 스페인어로 된 4,555개의 CXR 연구를 포함하고 있으며, 여기에는 3,099개의 비정상 및 1,456개의 정상 케이스가 포함되어 있습니다. 각 긍정 발견 문장은 두 개의 독립적인 바운딩 박스 세트와 관련이 있으며, 발견 유형, 위치, 진행 상태에 대한 범주형 레이블이 함께 제공됩니다. 이는 의료 이미지와 생성된 텍스트의 이해 및 해석을 위한 최초의 수작업으로 구성된 데이터셋입니다.

- **Performance Highlights**: PadChest-GR은 의료 AI 모델의 검증을 돕고, 이해도를 높이며, 임상의와 환자 간의 상호작용을 촉진하는 데 중요한 기초를 제공하고 있습니다. 이 데이터셋은 방사선 이미지를 수집하여 의료 AI 모델의 개발 및 평가에 유용한 자료로 활용될 것입니다.



### Deep Heuristic Learning for Real-Time Urban Pathfinding (https://arxiv.org/abs/2411.05044)
- **What's New**: 이 논문은 전통적인 휴리스틱 기반 (heuristic-based) 알고리즘을 심층 학습 모델 (deep learning models)로 변환하여 실시간 컨텍스트 데이터(traffic 및 weather conditions)를 활용하는 새로운 도시 경로 탐색 방법을 소개합니다. 두 가지 방법을 제안합니다: 현재 환경 조건에 따라 경로를 동적으로 조정하는 향상된 A* 알고리즘과 과거 및 실시간 데이터를 사용해 다음 최적 경로를 예측하는 신경망 모델입니다.

- **Technical Details**: 연구에서는 MLP, GRU, LSTM, Autoencoders, Transformers와 같은 여러 심층 학습 모델을 체계적으로 비교하였으며, 베를린의 시뮬레이션된 도시 환경에서 평가되었습니다. 두 가지 접근법 중 신경망 모델이 전통적인 방법에 비해 최대 40%의 여행 시간을 단축시키며, 향상된 A* 알고리즘은 34%의 개선을 달성했습니다. 이러한 결과는 심층 학습이 실시간으로 도시 내비게이션을 최적화하는 잠재력을 보여줍니다.

- **Performance Highlights**: 제안된 모델을 통해 경로 탐색 효율성이 크게 향상되었으며, 특히 심층 학습 모델이 전통적인 방법 보다 효과적임을 입증하였습니다. 실시간 데이터에 대한 적응성을 통해 동적 도시 환경에서의 경로 최적화에 기여하고 있습니다.



### Towards Probabilistic Planning of Explanations for Robot Navigation (https://arxiv.org/abs/2411.05022)
- **What's New**: 이 논문에서는 로봇 경로 계획 과정에 사용자 중심 디자인 원칙을 통합하는 새로운 접근 방식을 소개합니다. 이는 사용자의 설명 선호도를 확률적으로 모델링하여 로봇의 내비게이션을 위한 자동 설명 계획을 위한 확률적 프레임워크를 제안합니다.

- **Technical Details**: 이 연구는 Explainable AI Planning (XAIP)에 중점을 두고, 로봇의 설명을 생성하는 과정을 확률적 계획 문제로 개념화합니다. RDDL(Relational Dynamic Influence Diagram Language)을 사용하여 로봇의 행동에 대한 인간의 설명 선호를 모델링하는데, 여기서 설명의 구성, 타이밍 및 형식 등에 대한 속성을 고려합니다.

- **Performance Highlights**: 이 접근 방식은 사람들이 특정 유형의 설명을 요구하는 것을 예측함으로써 로봇 경로 계획의 투명성을 향상시키고, 다양한 사용자 요구에 적응할 수 있는 가능성을 보여줍니다. 이는 인간-로봇 상호작용(HRI)의 신뢰성과 효율성을 증대시킬 수 있는 것으로 기대됩니다.



### ASL STEM Wiki: Dataset and Benchmark for Interpreting STEM Articles (https://arxiv.org/abs/2411.05783)
Comments:
          Accepted to EMNLP 2024

- **What's New**: ASL STEM Wiki는 영어로 된 254개의 Wikipedia 기사를 아메리칸 수화(ASL)로 해석한 병렬 말뭉치로, 청각 장애인(DHH) 학생들이 STEM 교육을 더 쉽게 접근할 수 있도록 돕기 위한 첫 번째 연속 수화 데이터셋입니다.

- **Technical Details**: 이 데이터셋은 37명의 인증된 통역사가 ASL로 해석한 64,266개의 문장과 300시간 이상의 ASL 비디오로 구성되어 있으며, STEM 관련 5개 주제에 걸쳐 진행되었습니다. 데이터셋은 일반적인 영어 문장을 ASL 비디오와 매칭하여 수화 연산 모델에 새로운 도전과제를 제시합니다.

- **Performance Highlights**: 이 논문에서 제안한 모델은 STEM 콘텐츠의 손가락 철자가 자주 사용되는 문제를 해결하고자 하며, 손가락 철자 탐지 정확도를 47% 개선하는 contrastive learning 기법을 사용하였습니다. 결과적으로, ASL 교육 자료 접근성을 높이고, 기술적인 ASL 수화 사용을 촉진하는 도구 개발을 위한 도약의 기회를 제공합니다.



### Using Language Models to Disambiguate Lexical Choices in Translation (https://arxiv.org/abs/2411.05781)
Comments:
          Accepted to EMNLP 2024

- **What's New**: 본 연구에서는 단어가 여러 변형으로 번역되는 경우의 모호함을 해결하기 위해, DTAiLS라는 데이터셋을 구축했습니다. 이 데이터셋은 9개의 언어를 포함하여 1,377개의 문장 쌍으로 구성되어 있으며, 다양한 언어 간 개념 변형을 포함합니다.

- **Technical Details**: DTAiLS(Translation with Ambiguity in Lexical Selection)는 단어 선택의 맥락을 이해하기 위한 데이터셋입니다. 본 논문에서는 LLM(대형 언어 모델)과 NMT(신경 기계 번역) 시스템의 성능을 평가하였고, GPT-4 모델이 67%에서 85%의 정확도를 달성했습니다. LLM을 통해 생성된 규칙을 제공함으로써 더 낮은 성능의 모델도 정확도를 개선할 수 있음을 보여주었습니다.

- **Performance Highlights**: 최고 성능을 기록한 GPT-4는 여러 언어에서 67%에서 85%의 정확도를 보여주었으며, 자가 생성된 규칙을 활용했을 때 모든 LLM의 성능이 향상되었습니다. 또한, GPT-4의 규칙을 활용하여 오픈 웨이트 LLM이 NMT 시스템에 비해 성능 격차를 줄일 수 있음을 시사합니다.



### GazeSearch: Radiology Findings Search Benchmark (https://arxiv.org/abs/2411.05780)
Comments:
          Aceepted WACV 2025

- **What's New**: 이 논문에서는 방사선학에 특화된 시선 추적 데이터셋인 GazeSearch를 제안하고, 이를 통해 방사선과 의료 이미징의 해석 가능성과 정확성을 개선하고자 합니다. 또한, ChestSearch라는 스캔 경로 예측 모델을 소개하여 현재의 알고리즘 성능을 평가합니다.

- **Technical Details**: GazeSearch는 기존의 시선 추적 데이터(EGD, REFLACX)를 가공하여 검사 결과에 주목하는 데이터셋으로 변환합니다. ChestSearch 모델은 Transformer 기반으로 설계되었으며, 자기 지도 학습 및 쿼리 메커니즘을 활용하여 중요한 시선 데이터를 선택합니다. 이 모델은 특정 데이터를 예측하는 다중 태스크 처리 기능을 갖추고 있습니다.

- **Performance Highlights**: GazeSearch 데이터셋을 기반으로 ChestSearch 모델의 성능을 평가한 결과, 기존의 시선 추적 예측 모델들보다 뛰어난 성능을 보였으며, 방사선학 분야에서 비주얼 서치의 최신 발전을 보여줍니다.



### Quantitative Assessment of Intersectional Empathetic Bias and Understanding (https://arxiv.org/abs/2411.05777)
- **What's New**: 본 논문은  현재의 공감(complex empathy) 정의의 모호함으로 인해 공감 평가 방법의 비효율성을 비판하고, 공감의 심리적 기원에 가깝게 공감을 구체적으로 측정할 수 있는 평가 프레임워크를 제안합니다.

- **Technical Details**: 제안된 JaEm-ST 프레임워크는 공감을 두 가지 차원, 즉 인지적 공감(Cognitive Empathy, CE)과 정서적 공감(Affective Empathy, AE)으로 나누어 정의하며, 각 차원에 대한 측정 방법 및 평가 절차를 포함합니다. 평가 데이터 세트는 마스킹된 템플릿을 통해 생성되며, 다양한 사회적 편견(social biases)을 적용하여 대화 에이전트와의 공감 이해도를 계산합니다.

- **Performance Highlights**: 초기 평가 샘플에서 모델 간의 공감 이해도 차이는 크지 않았으나, 모델의 추론(chain of reasoning) 과정에서 prompt의 미세한 변화에 따른 유의미한 차이를 보였습니다. 이는 향후 공감 평가 샘플 구성 및 통계적 방법론에 대한 연구의 기초가 될 것입니다.



### Fact or Fiction? Can LLMs be Reliable Annotators for Political Truths? (https://arxiv.org/abs/2411.05775)
Comments:
          Accepted at Socially Responsible Language Modelling Research (SoLaR) Workshop at NeurIPS 2024

- **What's New**: 이 연구는 오픈 소스 대형 언어 모델(LLMs)을 정치적 사실성을 측정하기 위한 신뢰할 수 있는 주석자로 활용하는 방법을 탐구합니다. 이를 통해 전통적인 사실 확인 방법의 한계를 극복하고, 언론에 대한 투명성과 신뢰를 높이고자 합니다.

- **Technical Details**: 정치 편향(political bias)에 대한 분석을 바탕으로, LLMs를 사용하여 정치 관련 뉴스 기사를 사실적으로 정확한 것과 부정확한 것으로 이분화하는 주석 작업을 수행합니다. 또한, LLM 주석자의 정확성을 평가하기 위해 LLM을 평가자로 활용하는 방법론을 제시합니다.

- **Performance Highlights**: 실험 결과, LLM이 생성한 주석은 인간 주석자와 높은 일치를 보였으며, LLM 기반 평가에서는 강력한 성능을 나타내어 사실 확인 과정에서 효과적인 대안임을 입증하였습니다.



### On Differentially Private String Distances (https://arxiv.org/abs/2411.05750)
- **What's New**: 이 논문에서는 Hamming 거리와 edit 거리 계산을 위한 새로운 차등 프라이버시(differentially private, DP) 데이터 구조를 제안합니다. 이 데이터 구조는 비밀성을 보장하면서도 높은 성능을 제공한다는 점에서 혁신적입니다.

- **Technical Details**: 논문에서는 데이터베이스에 포함된 비트 문자열의 거리 추정 문제를 다룹니다. Hamming 거리의 경우, 쿼리를 처리하는 데 걸리는 시간은 O(mk+n)이며, 각 추정값은 실제 거리와 최대 O(k/e^(ε/log k))의 차이를 가집니다. Edit 거리의 경우, 쿼리 처리 시간은 O(mk^2+n)이며, 추정값의 차이는 O(k/e^(ε/(log k log n)))로 제한됩니다.

- **Performance Highlights**: 제안된 데이터 구조는 ε-DP를 유지하며, moderate한 k값에 대해 서브라인 리어 쿼리 작업을 지원합니다. 이는 효율성을 고려한 차별화된 접근 방식을 보여줍니다.



### Multi-Dimensional Reconfigurable, Physically Composable Hybrid Diffractive Optical Neural Network (https://arxiv.org/abs/2411.05748)
Comments:
          7 pages

- **What's New**: 새로운 연구에서는 다차원 재구성이 가능한 혼합 회절 광 신경망 시스템(MDR-HDONN)을 소개하여, 고정된 광학 구조의 한계를 극복하고 출력의 유연성과 적응성을 향상 시켰습니다.

- **Technical Details**: MDR-HDONN은 통합 광학 및 포토닉 디자인을 채택하여 고정된 제작된 광학 하드웨어를 재사용하며, 차별 가능한 학습을 통해 시스템 변수를 최적화합니다. 이는 시스템의 운용 가능성과 기능의 범위를 크게 확장하는 데 기여합니다.

- **Performance Highlights**: MDR-HDONN은 다양한 작업 적응에서 디지털에 상응하는 정확도를 보이며, 74배 빠른 속도와 194배 낮은 에너지를 소비합니다. 이전의 DONN과 비교했을 때 교육 속도는 5배 빨라졌으며, 이는 혼합 광학/포토닉 AI 컴퓨팅의 새로운 패러다임을 열어주는 데 기여합니다.



### Continuous-Time Analysis of Adaptive Optimization and Normalization (https://arxiv.org/abs/2411.05746)
- **What's New**: 이번 연구에서는 현대 딥러닝에서 중요한 구성 요소인 Adam 및 AdamW 최적화 알고리즘의 지속적인 시간(formulation) 모델을 제시하고, 이 모델을 통해 이러한 최적화 알고리즘의 훈련 동역학(training dynamics)을 분석합니다. 특히, 하이퍼파라미터(hyperparameters)의 최적 선택과 구조적 결정(architectural decisions)에 대한 보다 깊은 이해를 제공합니다.

- **Technical Details**: Adam 및 AdamW 최적화 알고리즘의 지속적인 시간 공식화는 쌍곡선 차분 방정식으로 표현되며, 이를 통해 Adam의 하이퍼파라미터(eta, \\gamma)의 안정적 지역을 이론적으로 도출합니다. 또한, scale-invariant 아키텍처의 메타 적응 효과(meta-adaptive effect)를 밝혀내어 최적화기 $k$-Adam을 일반화하는 과정을 설명합니다.

- **Performance Highlights**: 실험적으로, 제안된 하이퍼파라미터 지역 외부에서 파라미터 업데이트의 불안정한 지수 성장(exponential growth)을 확인함으로써 이론 예측을 실증적으로 검증하였습니다. $2$-Adam optimizer는 정상화(normalization) 절차를 k회 적용하여 Adam과 AdamW의 성능을 통합하는 새로운 접근 방식을 제시합니다.



### Topology-aware Reinforcement Feature Space Reconstruction for Graph Data (https://arxiv.org/abs/2411.05742)
- **What's New**: 이번 연구에서는 그래프 데이터의 특성 공간(feature space) 재구성을 자동화하고 최적화하는 새로운 접근법을 제시합니다. 기존의 수작업적인 피처 변환 및 선택 기법과는 달리, 그래프 데이터의 고유한 위상 구조(topological structure)를 고려하여 혁신적인 해법을 제공합니다.

- **Technical Details**: 이 논문에서는 topology-aware reinforcement learning을 활용하여 그래프 데이터의 피처 공간을 재구성하는 방식을 도입합니다. 주된 구성 요소로는 핵심 서브그래프(core subgraphs) 추출, 그래프 신경망(graph neural network; GNN) 사용, 그리고 세 가지 계층적 강화 에이전트가 포함됩니다. 이 접근법은 피처 생성의 반복적인 과정을 통해 의미 있는 피처를 효과적으로 생성합니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안된 방법의 효과성과 효율성이 입증되었습니다. 특히, 위상 정보를 포함함으로써 피처 공간의 재구성을 최적화할 수 있으며, 이는 내려가는 머신러닝 작업(downstream ML tasks)의 성능 향상으로 이어집니다.



### Aioli: A Unified Optimization Framework for Language Model Data Mixing (https://arxiv.org/abs/2411.05735)
- **What's New**: 이 논문에서는 언어 모델의 성능이 데이터 그룹의 최적 혼합 비율을 파악하는 데 의존한다는 점을 강조하며, 다양한 방법의 효율성을 비교했습니다. 특히, 기존 방법들이 단순한 stratified sampling 기준선보다 높은 평균 테스트 perplexity 성능을 보이지 못한 점을 발견했습니다.

- **Technical Details**: 저자들은 Linear Mixing Optimization (LMO)이라는 통합 최적화 프레임워크를 제안하였으며, 이를 통해 다양한 혼합 방법들의 기본 가정을 분석했습니다. 혼합 법칙(mixing law)의 매개변수 조정에서의 오류가 기존 방법들의 성능 불일치를 초래한다는 것을 입증했습니다.

- **Performance Highlights**: 새롭게 제안된 온라인 데이터 혼합 방법 Aioli는 6개의 데이터셋에서 모든 경우에 걸쳐 본 논문의 기존 방법보다 평균 0.28 포인트의 테스트 perplexity 향상을 보였습니다. 또한, Aioli는 계산자원이 제한된 상황에서도 혼합 비율을 동적으로 조정하여 기존 방법들보다 최대 12.01 포인트 이상 개선된 성능을 보여줍니다.



### A Retrospective on the Robot Air Hockey Challenge: Benchmarking Robust, Reliable, and Safe Learning Techniques for Real-world Robotics (https://arxiv.org/abs/2411.05718)
Comments:
          Accept at NeurIPS 2024 Dataset and Benchmark Track

- **What's New**: 2023 NeurIPS에서 열린 Robot Air Hockey Challenge는 로봇학습을 위한 새로운 벤치마크로, 기존의 단순한 시뮬레이션 기반 테스트와는 달리 실제 로봇 환경에서의 적용 가능성을 중점적으로 다루고 있습니다.

- **Technical Details**: 이 대회는 시뮬레이션과 실제 환경 간의 간극(sim-to-real gap) 문제, 안전성 문제, 데이터 부족 등의 실제 로봇 문제를 해결하기 위한 방법론을 탐구합니다. Kuka LBR IIWA 14 로봇을 사용하여 공기 하키 디자인을 구현하였고, MuJoCo 시뮬레이터를 통해 제어 전략을 평가함으로써 다양한 환경 요인들을 반영했습니다.

- **Performance Highlights**: 학습 기반 접근 방식과 기존 지식을 결합한 솔루션이 데이터만 의존하는 방식보다 뛰어난 성능을 보였으며, 최상의 성과를 올린 에이전트들이 성공적인 실제 공기 하키 배치 사례를 수립했습니다.



### Visual-TCAV: Concept-based Attribution and Saliency Maps for Post-hoc Explainability in Image Classification (https://arxiv.org/abs/2411.05698)
Comments:
          Preprint currently under review

- **What's New**: 이 논문에서는 CNN (Convolutional Neural Networks)의 이미지 분류를 위한 새로운 설명 가능성 프레임워크인 Visual-TCAV를 소개합니다. 이 방법은 기존의 saliency 방법과 개념 기반 접근 방식을 통합하여, CNN의 예측에 대한 개념의 기여도를 숫자로 평가하는 동시에, 어떤 이미지에서 어떤 개념이 인식되었는지를 시각적으로 설명합니다.

- **Technical Details**: Visual-TCAV는 Concept Activation Vectors (CAVs)를 사용하여 네트워크가 개념을 인식하는 위치를 보여주는 saliency map을 생성합니다. 또한, Integrated Gradients의 일반화를 통해 이러한 개념이 특정 클래스 출력에 얼마나 중요한지를 추정할 수 있습니다. 이 프레임워크는 CNN 아키텍처에 적용 가능하며, 각 시각적 설명은 네트워크가 선택한 개념을 인식한 위치와 그 개념의 중요도를 포함합니다.

- **Performance Highlights**: Visual-TCAV는 기존의 TCAV와 비교하여 지역적(local) 및 글로벌(global) 설명 가능성을 모두 제공할 수 있으며, 실험을 통해 그 유효성이 확인되었습니다. 본 방법은 CNN 모델의 다양한 레이어에 적용할 수 있으며, 사용자 정의 개념을 통합하여 더 풍부한 설명을 제공합니다.



### Asterisk*: Keep it Simp (https://arxiv.org/abs/2411.05691)
- **What's New**: 이 논문에서는 Asterisk라는 소형 GPT 기반의 텍스트 임베딩 생성 모델을 설명합니다. 이 모델은 두 개의 레이어, 두 개의 어텐션 헤드, 256 차원의 임베딩을 포함한 미니멀리스트 아키텍처(minimalist architecture)를 가지고 있으며, 대형 사전 학습(pre-trained) 모델로부터 지식 증류(Knowledge Distillation)를 활용하여 모델 크기와 성능 간의 균형을 탐색합니다.

- **Technical Details**: Asterisk 모델은 256차원 임베딩 공간과 2개의 Transformer 레이어, 각 레이어 당 2개의 어텐션 헤드를 사용합니다. 총 14,019,584개의 파라미터로 구성되어 있으며, 임베딩 레이어는 토큰 임베딩과 위치 임베딩을 결합하여 초기화되었습니다. 이 모델은 주로 MSE(Mean Squared Error)와 코사인 유사성(Cosine Similarity)을 결합하여 지식 증류 과정을 구현하였고, OpenAI의 text-embedding-3-small을 교사 모델로 선택하여 사용했습니다.

- **Performance Highlights**: Asterisk 모델은 다양한 분류 작업에서 실험적으로 중간 수준의 성능을 보였으며, 사전 훈련된 큰 모델들과 비교하여 특정 작업에서는 성능이 유사하거나 뛰어난 결과를 나타냈습니다. 특히 Fully-Connected 네트워크를 통해 1000개의 샘플 훈련만으로도 신뢰할 수 있는 분류 성능을 달성하였고, MTEB 리더보드에서는 MassiveIntentClassification(1위)와 AmazonReviewsClassification(2위) 작업에서 성과를 거두었습니다.



### Data-Driven Distributed Common Operational Picture from Heterogeneous Platforms using Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2411.05683)
Comments:
          29th International Command and Control Research & Technology Symposium

- **What's New**: 본 연구에서는 군사 작전에서 상황 인식을 강화하고 "전쟁의 안개"를 완화하기 위한 차세대 다중 에이전트 학습 프레임워크를 소개합니다. 이 프레임워크는 에이전트와 인간 간의 자율적이고 안전한 통신을 가능하게 하여 실시간으로 해석 가능한 Common Operational Picture (COP)를 형성할 수 있도록 합니다.

- **Technical Details**: 각 에이전트는 자신의 인식 및 행동을 압축된 벡터로 인코딩하여 전송하고, 이를 통해 전투 상황에서의 모든 에이전트(우호 및 적군)의 현재 상태를 포함하는 COP를 생성합니다. Deep Reinforcement Learning (DRL)을 사용하여 COP 모델과 에이전트의 행동 선택 정책을 공동으로 훈련하였습니다.

- **Performance Highlights**: Starcraft-2 시뮬레이션 환경에서 실험적 검증을 수행하였으며, COP의 정확도는 5% 미만의 오류를 보였고, 다양한 적대적 조건에서도 정책이 견고성을 유지함을 보고했습니다. 연구 결과는 자율적인 COP 형성 방법, 분산 예측을 통한 향상된 회복력, COP 모델과 다중 에이전트 RL 정책의 공동 훈련을 포함하여, 이종 무인 플랫폼의 효과적인 제어를 촉진하는 적응적이고 회복력이 있는 Command and Control (C2) 시스템을 발전시키는 방향으로 나아갑니다.



### Tell What You Hear From What You See -- Video to Audio Generation Through Tex (https://arxiv.org/abs/2411.05679)
Comments:
          NeurIPS 2024

- **What's New**: 본 연구에서는 비디오와 텍스트 프롬프트를 입력받아 오디오 및 텍스트 설명을 생성하는 다중 모달 생성 프레임워크인 VATT를 제안합니다. 이 모델은 비디오의 맥락을 보완하는 텍스트를 통해 오디오 생성 과정을 세밀하게 조정 가능하게 하며, 비디오에 대한 오디오 캡션을 생성하여 적절한 오디오를 제안할 수 있습니다.

- **Technical Details**: VATT에는 VATT Converter와 VATT Audio라는 두 가지 모듈이 포함되어 있습니다. VATT Converter는 미세 조정된 LLM(대형 언어 모델)으로, 비디오 특징을 LLM 벡터 공간에 매핑하는 프로젝션 레이어를 갖추고 있습니다. VATT Audio는 변환기로, 시각적 프레임 및 선택적 텍스트 프롬프트에서 오디오 토큰을 생성하는 병렬 디코딩을 통해 운영됩니다. 생성된 오디오 토큰은 사전 훈련된 뉴럴 코덱을 통해 웨이브폼으로 변환됩니다.

- **Performance Highlights**: 실험 결과에 따르면, VATT는 기존 방법 대비 경쟁력 있는 성능을 보이며, 오디오 캡션이 제공되지 않을 때에도 향상된 결과를 도출합니다. 오디오 캡션을 제공할 경우 KLD 점수가 1.41로 가장 낮은 성과를 기록했습니다. 주관적 연구에서는 VATT Audio에서 생성된 오디오가 기존 방법에 비해 높은 선호도를 나타냈습니다.



### Improving Molecular Graph Generation with Flow Matching and Optimal Transpor (https://arxiv.org/abs/2411.05676)
- **What's New**: GGFlow는 분자 그래프 생성을 위한 첫 번째 이산 흐름 매칭 생성 모델로, 최적 운송(optimal transport)을 활용하여 샘플링 효율성과 훈련 안정성을 향상시킵니다. 이 모델은 화학 결합 간의 직접적인 통신을 가능하게 하는 가장자리 증강 그래프 변환기(edge-augmented graph transformer)를 통합하여 생성 작업을 개선합니다.

- **Technical Details**: GGFlow는 이산 흐름 매칭 기법과 최적 운송을 활용하여 분자 그래프 생성을 위한 샘플링 효율성과 훈련 안정성을 개선하는 모델입니다. 이 모델은 그래프의 희소성(sparsity)과 순열 불변성(permutation invariance)을 유지하면서 화학 결합(info flash) 생성을 위한 트랜스포머를 포함합니다. 또한, 목표 속성을 가진 분자를 디자인하기 위해 강화 학습(reinforcement learning)을 이용한 목표 지향(guided) 세대 프레임워크가 도입되었습니다.

- **Performance Highlights**: GGFlow는 무조건 조건 및 조건부 분자 생성 작업에서 최첨단 성능을 보여주며 기존 방법들을 일관되게 초월합니다. 이 모델은 적은 추론 단계(few inference steps)에서도 뛰어난 성과를 발휘하며, 다양한 그래프 유형과 복잡성에 대해 개선된 결과를 보여줍니다.



### The influence of persona and conversational task on social interactions with a LLM-controlled embodied conversational agen (https://arxiv.org/abs/2411.05653)
Comments:
          11 pages, 5 figures

- **What's New**: 이번 연구는 Large Language Models (LLMs)가 가상 현실(Virtual Reality) 환경에서 사용자와 대화할 때, LLM의 성격(예: 외향적 또는 내향적)에 따른 상호작용의 영향을 분석하였습니다.

- **Technical Details**: 46명의 참가자가 외향적 또는 내향적으로 조작된 가상 에이전트와 세 가지 대화 과제(가벼운 대화, 지식 테스트, 설득)에서 상호작용했습니다. 사회적 평가는 평가 점수를 통해, 정서적 경험과 사실감은 등급으로 측정되었습니다. 상호작용 참여는 참가자의 발언 수 및 대화 전환 수로 정량화되었습니다.

- **Performance Highlights**: 외향적인 에이전트는 내향적인 에이전트에 비해 긍정적인 평가를 받았으며, 더 쾌적한 경험과 높은 참여도를 유도했습니다. 또한, 외향적인 에이전트가 더 사실적으로 평가되었습니다. 도움을 요청하는 경향에는 개인의 성격이 영향을 주지 않았으나, LLM의 도움을 받을 때 참가자들은 답변에 대해 더 자신감을 느꼈습니다.



### SynDroneVision: A Synthetic Dataset for Image-Based Drone Detection (https://arxiv.org/abs/2411.05633)
Comments:
          Accepted at the 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)

- **What's New**: 이번 논문에서는 감시 애플리케이션을 위한 RGB 기반 드론 탐지에 특화된 합성 데이터 세트인 SynDroneVision을 소개하고 있습니다. 합성 데이터를 활용함으로써 실제 데이터 수집의 시간과 비용 부담을 크게 줄일 수 있습니다.

- **Technical Details**: SynDroneVision 데이터 세트는 다양한 배경, 조명 조건, 드론 모델을 포함하여, 깊이 학습(deep learning) 알고리즘을 위한 포괄적인 훈련 기반을 제공합니다. 이 데이터 세트는 YOLO(You Only Look Once) 모델의 최신 버전들과 비교하여 효과성을 평가하였습니다.

- **Performance Highlights**: SynDroneVision은 모델 성능과 견고성에서 주목할 만한 향상을 이끌어냈으며, 실제 데이터 수집의 시간과 비용을 상당히 절감할 수 있음을 보여주었습니다.



### Knowledge Distillation Neural Network for Predicting Car-following Behaviour of Human-driven and Autonomous Vehicles (https://arxiv.org/abs/2411.05618)
Comments:
          27th IEEE International Conference on Intelligent Transportation Systems

- **What's New**: 본 연구는 자율주행차(Autonomous vehicles, AV)와 인간이 운전하는 차량(Human-driven vehicles, HDV)의 혼합 교통 시나리오에서 차량 추적 행동(car-following behaviour)을 분석하여 교통 효율성 및 도로 안전성을 개선하는 데 중점을 둡니다. 기존 모델들과 비교하여 Knowledge Distillation Neural Network (KDNN) 모델을 도입하여 차량 추적 행동의 속도를 예측합니다.

- **Technical Details**: 이 연구는 Waymo의 차량 궤적 데이터셋을 사용하여 HDV-AV, AV-HDV, HDV-HDV의 세 가지 차량 쌍에 대한 차량 추적 행동을 분석합니다. KDNN 모델은 복잡한 신경망 모델의 지식을 경량화된 모델로 증류하여 기존의 LSTM 및 MLP와의 성능 비교를 통해 높은 예측 정확도를 유지하면서도 낮은 연산 요구 사항을 충족합니다.

- **Performance Highlights**: KDNN 모델은 최소 시간-충돌(Time-to-Collision, TTC) 측정에서 충돌 방지 성능이 우수하며, LSTM 및 MLP와 비교했을 때 예측 정확도가 유사하거나 더 뛰어난 결과를 보이고 있습니다. 이 모델은 자율주행차 및 드라이빙 시뮬레이터와 같은 자원 제약 환경에서 효율적인 연산이 가능합니다.



### Acceleration for Deep Reinforcement Learning using Parallel and Distributed Computing: A Survey (https://arxiv.org/abs/2411.05614)
Comments:
          This paper has been accepted by ACM Computing Surveys

- **What's New**: 이 논문은 최근 몇 년간 인공지능 분야에서 중요한 발전을 가져온 Deep Reinforcement Learning (DRL)의 훈련을 가속화하기 위한 방법론을 포괄적으로 조사한 내용을 담고 있습니다. 특히, 병렬 및 분산 컴퓨팅(parallel and distributed computing)을 기반으로 한 DRL 훈련 가속화 방법에 대한 통계적 분류와 신기술 동향을 다룹니다.

- **Technical Details**: DRL은 인공지능(AI) 연구자들에 의해 널리 탐구되고 있는 강력한 머신러닝(parallel and distributed computing) 패러다임입니다. 이 논문에서는 시스템 아키텍처, 시뮬레이션 병렬성(simulation parallelism), 계산 병렬성(computing parallelism), 분산 동기화 메커니즘(distributed synchronization mechanisms), 심층 진화 강화 학습(deep evolutionary reinforcement learning) 등의 다양한 기술적 세부정보를 분석합니다.

- **Performance Highlights**: 논문에서는 현재의 16개 오픈 소스 라이브러리(open-source libraries) 및 플랫폼을 비교하고, 이러한 시스템들이 병렬 및 분산 DRL을 구현하는 데 어떻게 기여하는지를 분석합니다. 연구자들이 제안한 기술들을 실제로 어떻게 적용할 수 있는지를 조명하며, 향후 연구 방향에 대한 통찰을 제공합니다.



### Expectation vs. Reality: Towards Verification of Psychological Games (https://arxiv.org/abs/2411.05599)
- **What's New**: 본 논문은 심리적 게임(Psychological Games, PGs)의 적용 가능성을 컴퓨터 과학 문제로 확장하며, PRISM-games라는 정형 검증 도구를 이용하여 PGs를 해결하는 방법을 제안합니다.

- **Technical Details**: PGs에서 플레이어의 효용은 실제로 발생하는 것뿐만 아니라 플레이어가 기대했던 사건에도 의존합니다. 이 논문은 이러한 게임을 모델링하고 이를 PRISM-games에 구현하는 방법을 다룹니다. 또한 분석에 있어 구체적인 도전 과제를 강조합니다.

- **Performance Highlights**: 서로 다른 사례 연구를 통해 인간 행동이 교통 시나리오에 미치는 영향을 포함한 PGs의 유용성을 입증합니다.



### Tangled Program Graphs as an alternative to DRL-based control algorithms for UAVs (https://arxiv.org/abs/2411.05586)
Comments:
          The papers was accepted for the 2024 Signal Processing: Algorithms, Architectures, Arrangements, and Applications (SPA) conference in Poznan, Poland

- **What's New**: 이번 연구는 Tangled Program Graphs (TPGs)를 이용해 자율 비행기 제어에서 심층 강화 학습 (Deep Reinforcement Learning, DRL)의 대안으로 사용하며, LiDAR 데이터를 처리하는 최초의 시도를 제시합니다.

- **Technical Details**: TPGs는 환경 데이터와 에이전트의 행동 간의 관계를 정의하는 방향 그래프입니다. 이를 통해 TPG 기반 모델은 대규모 데이터나 집중적인 계산 자원 없이 상태 및 행동 공간의 미세한 차이를 효과적으로 캡처합니다. 이 구조는 해석 가능하고 효율적인 학습 과정을 촉진하며, TPG 모델의 훈련은 강화 학습 프레임워크 내에서 수행될 수 있습니다.

- **Performance Highlights**: 실험 결과는 TPGs가 자율 비행기 제어 과제에서 DRL과 비교했을 때 유망한 성과를 보여주었습니다. 연구에서는 LiDAR 센서를 사용하여 미지의 환경을 탐색하는 상황에서 TPG를 적용하여 DRL의 한계를 극복하려는 노력을 하였습니다.



### Open-set object detection: towards unified problem formulation and benchmarking (https://arxiv.org/abs/2411.05564)
Comments:
          Accepted at ECCV 2024 Workshop: "The 3rd Workshop for Out-of-Distribution Generalization in Computer Vision Foundation Models"

- **What's New**: 이 연구에서는 OpenSet Object Detection(OSOD) 관련 여러 접근 방식의 평가를 통합하기 위한 새로운 벤치마크(OpenImagesRoad)를 소개하고, 기존의 편향된 데이터 세트와 평가 지표를 해결하는 데 초점을 맞추고 있습니다. 이를 통해 OSOD에 대한 명확한 문제 정의와 일관된 평가가 가능해집니다.

- **Technical Details**: 연구에서는 다양한 unknown object detection 접근 방식을 논의하며, VOC와 COCO 데이터 세트를 기반으로 하는 통합된 평가 방법론을 제안합니다. 새로운 OpenImagesRoad 벤치마크는 명확한 계층적 객체 정의와 새로운 평가 지표를 제공합니다. 또한 최근 자가 감독 방식(self-supervised)으로 학습된 Vision Transformers(DINOv2)를 활용하여 pseudo-labeling 기반의 OSOD를 개선하는 OW-DETR++ 모델을 제안합니다.

- **Performance Highlights**: 본 연구에서 제안한 벤치마크에서 state-of-the-art(OSOT) 방법들의 성능을 광범위하게 평가하였으며, OW-DETR++ 모델은 기존 pseudo-labeling 방법들 중에서 가장 우수한 성능을 기록했습니다. 이를 통해 OSOD 전략의 효과 및 경계에 대한 새로운 통찰을 제공합니다.



### Training objective drives the consistency of representational similarity across datasets (https://arxiv.org/abs/2411.05561)
Comments:
          26 pages

- **What's New**: 최근의 기초 모델들이 다운스트림 작업 성능에 따라 공유 표현 공간으로 수렴하고 있다는 새로운 가설인 Platonic Representation Hypothesis를 제시합니다. 이는 데이터 모달리티와 훈련 목표와는 무관하게 발표됩니다.

- **Technical Details**: 연구에서는 CKA와 RSA와 같은 유사도 측정 방법을 사용하여 모델 표현의 일관성을 측정하는 체계적인 방법을 제안했습니다. 실험 결과, 자기 지도 시각 모델이 이미지 분류 모델이나 이미지-텍스트 모델보다 다른 데이터셋에서의 쌍별 유사도를 더 잘 일반화하는 것으로 나타났습니다. 또한, 모델의 과제 행동과 표현 유사성 간의 관계는 데이터셋 의존적이라 밝혀졌습니다.

- **Performance Highlights**: 이 연구를 통해 쌍별 표현 유사도가 모델의 작업 성능 차이와 강한 상관관계를 보임을 발견하였으며, 단일 도메인 데이터셋에서 이러한 경향이 가장 뚜렷하게 나타났습니다.



### A Nerf-Based Color Consistency Method for Remote Sensing Images (https://arxiv.org/abs/2411.05557)
Comments:
          4 pages, 4 figures, The International Geoscience and Remote Sensing Symposium (IGARSS2023)

- **What's New**: 이번 연구에서는 계절, 조명 및 대기 조건에 따라 서로 다른 이미지를 통합할 때 발생하는 문제를 해결하기 위해 NeRF(Neural Radiance Fields) 기반의 색상 일관성 보정 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 다중 뷰 이미지의 특징을 암시적 표현을 통해 엮어 결과 이미지를 생성하고, 이를 통해 새로운 시점의 융합 이미지를 재조명합니다. 실험에는 Superview-1 위성 이미지와 UAV(불특정 비행체) 이미지를 사용하였으며, 이는 큰 범위와 시간 차이가 있는 데이터입니다.

- **Performance Highlights**: 실험 결과, 제안된 방법으로 생성된 합성 이미지는 뛰어난 시각적 효과를 가지며, 이미지 가장자리에서 부드러운 색상 전환을 보여주었습니다.



### CRepair: CVAE-based Automatic Vulnerability Repair Technology (https://arxiv.org/abs/2411.05540)
- **What's New**: 이 논문은 CVAE(Conditional Variational Autoencoder)를 기반으로 한 자동 취약점 수리 기술인 CRepair를 제안합니다. CRepair는 보안 취약점을 공략하는 시스템 코드의 문제를 해결하는 데 초점을 맞추고 있습니다.

- **Technical Details**: CRepair는 프롬프트 기반 방식으로 취약점 데이터를 전처리하고, 인과 추론 기법을 적용하여 취약점 특성 데이터를 확률 분포로 매핑합니다. 또한, 다중 샘플 특징 융합(Multi-sample Feature Fusion)을 통해 다양한 취약점 특성 정보를 포착합니다. 마지막으로 조건적 제어를 통해 수리 프로세스를 안내합니다.

- **Performance Highlights**: CRepair는 기존의 벤치마크 모델을 능가하여 52%의 완벽한 수정률을 달성했습니다. 이 접근 방법은 여러 관점에서의 효과성을 입증하며, AI 기반의 코드 취약점 수리 및 응용 가능성이 뛰어난 것으로 나타났습니다.



### SM3-Text-to-Query: Synthetic Multi-Model Medical Text-to-Query Benchmark (https://arxiv.org/abs/2411.05521)
Comments:
          NeurIPS 2024 Track Datasets and Benchmarks

- **What's New**: 이 논문에서 소개하는 SM3-Text-to-Query는 여러 데이터베이스 모델을 기반으로 한 최초의 의료 Text-to-Query 벤치마크입니다. Synthea의 합성 환자 데이터를 활용하여 SNOMED-CT 분류법에 따른 매우 다양한 표본 세트를 제공합니다.

- **Technical Details**: SM3-Text-to-Query는 PostgreSQL, MongoDB, Neo4j, GraphDB (RDF)를 포함한 세 가지 데이터베이스 모델을 사용하며, 각 모델에 대해 SQL, MQL, Cypher, SPARQL의 네 개의 쿼리 언어를 통한 평가를 가능하게 합니다. 본 데이터셋은 408개의 템플릿 질문을 포함하며, 총 10K 쌍의 자연어 질문/쿼리 쌍을 생성합니다.

- **Performance Highlights**: SM3-Text-to-Query는 다양한 ICL 접근 방식을 통해 여러 LLM의 성능을 평가하고, 데이터베이스 모델과 쿼리 언어 간의 trade-off를 조명합니다. 이 벤치마크는 향후 쿼리 언어 또는 실제 표준 기반 환자 데이터베이스로 쉽게 확장 가능합니다.



### Towards Scalable Foundation Models for Digital Dermatology (https://arxiv.org/abs/2411.05514)
Comments:
          Findings paper presented at Machine Learning for Health (ML4H) symposium 2024, December 15-16, 2024, Vancouver, Canada, 11 pages

- **What's New**: 이번 연구는 피부과(digital dermatology)에서의 데이터 부족 문제를 해결하기 위해 도메인-특화된 기초 모델(domain-specific foundation models)을 활용하는 방안을 제안하고 있습니다. 특히, 240,000장이 넘는 피부과 이미지를 사용하여 self-supervised learning (SSL) 기법을 적용하였고, 훈련된 모델을 공개하여 임상의들이 활용할 수 있도록 하였습니다.

- **Technical Details**: 모델 훈련에는 두 가지 아키텍처, 즉 CNN(Convolutional Neural Network) 및 ViT(Vision Transformer)를 사용하였으며, 각각 ResNet-50과 ViT-Tiny를 채택했습니다. 이 모델들은 임상에서의 제한된 자원 환경에서도 효율적으로 사용할 수 있도록 설계되었습니다. 연구에서는 총 12개의 진단 관련 다운스트림 작업(downstream tasks)을 통해 모델 성능을 평가했습니다.

- **Performance Highlights**: 모델의 성능은 일반적인 목적의 모델보다 우수할 뿐만 아니라, 50배 큰 모델에 근접하는 수준으로, 임상 진단 작업에서의 활용 가능성을 높이고 있습니다. 연구 결과는 리소스가 제한된 환경에서도 적용 가능한 효율적인 모델 개발에 기여할 것으로 기대됩니다.



### WorkflowLLM: Enhancing Workflow Orchestration Capability of Large Language Models (https://arxiv.org/abs/2411.05451)
- **What's New**: 본 논문은 WorkflowLLM이라는 데이터 중심의 프레임워크를 제안하여 LLMs의 workflow orchestration 능력을 향상시키기 위한 새로운 접근 방식을 소개합니다. 이 프레임워크는 106,763개의 샘플과 1,503개의 API를 포함하는 대규모 fine-tuning 데이터셋인 WorkflowBench를 기반으로 하며, 기존의 LLM들이 가지고 있던 제한점을 극복할 수 있는 방안을 모색합니다.

- **Technical Details**: WorkflowLLM은 데이터 수집, 쿼리 확장, workflow 생성의 세 가지 단계로 구성됩니다. 첫 번째 단계에서는 Apple Shortcuts와 RoutineHub로부터 실제 workflow 데이터를 수집하고 이를 Python 스타일 코드로 변환하여 처리합니다. 두 번째 단계에서는 ChatGPT를 활용하여 더 다양한 작업 쿼리를 생성하여 workflow의 다양성을 높입니다. 마지막 단계에서는 수집된 데이터를 기반으로 학습한 annotator 모델을 사용하여 쿼리에 대한 workflows를 생성합니다. 이러한 과정을 통해 Llama-3.1-8B 모델을 fine-tuning하여 WorkflowLlama를 생성합니다.

- **Performance Highlights**: 실험 결과, WorkflowLlama는 기존의 모든 기준선 모델들, 특히 GPT-4o를 포함해, unseen instructions 및 unseen APIs 설정하에서 강력한 workflow orchestration 능력을 보여주었으며, T-Eval 벤치마크에서 out-of-distribution 상황에서도 뛰어난 일반화 능력을 입증하였습니다. F1 plan score는 77.5%로 높게 기록되었습니다.



### ICE-T: A Multi-Faceted Concept for Teaching Machine Learning (https://arxiv.org/abs/2411.05424)
Comments:
          Accepted and presented at the 17th International Conference on Informatics in Schools (ISSEP 2024)

- **What's New**: 이번 논문은 인공지능(AI)과 머신 러닝(ML) 교육을 위해 다양한 플랫폼, 도구 및 게임을 활용하는 새로운 접근법을 소개합니다. 특히, ML을 교수할 때의 교육적 원칙과 이를 토대로 한 ICE-T라는 새로운 개념에 대해 설명합니다.

- **Technical Details**: ICE-T는 Intermodal transfer, Computational thinking, Explanatory thinking의 세 가지 요소로 구성된 다면적(멀티-팩터) 개념입니다. 이 개념은 교육자들이 ML 교육을 향상시키기 위해 사용할 수 있는 구조화된 접근 방식을 제공합니다. 또한 기존의 교수 도구들에서 이 요소들이 어떻게 구현되고 있는지를 평가합니다.

- **Performance Highlights**: 디지털 게임 기반 학습이 학생들의 동기 부여와 참여를 높이고, 인지 및 정서적 발달을 촉진하여 학습 효율성을 향상시킨다는 이전의 연구들을 언급하며, 학습 플랫폼의 부족한 부분을 보완하는 새로운 관점을 제안합니다.



### VISTA: Visual Integrated System for Tailored Automation in Math Problem Generation Using LLM (https://arxiv.org/abs/2411.05423)
Comments:
          Accepted at NeurIPS 2024 Workshop on Large Foundation Models for Educational Assessment (FM-Assess)

- **What's New**: 새로운 연구에서는 Large Language Models (LLMs)를 활용하여 수학 교육에서 복잡한 시각적 보조 도구를 자동으로 생성하는 멀티 에이전트 프레임워크를 소개합니다. 이 프레임워크는 수학 문제와 관련된 시각 보조 도구를 정확하고 일관되게 생성하도록 설계되었습니다.

- **Technical Details**: 이 연구에서는 7개의 특수화된 에이전트로 구성된 시스템을 설계하여 문제 생성 및 시각화 작업을 세분화하였습니다. 각 에이전트는 Numeric Calculator, Geometry Validator, Function Validator, Visualizer, Code Executor, Math Question Generator, Math Summarizer로 구성되며, 각 에이전트는 특정 역할에 맞춰 문제를 해결합니다.

- **Performance Highlights**: 시스템은 Geometry와 Function 문제 유형에서 기존의 기본 LLM보다 텍스트의 일관성, 관련성 및 유사성을 크게 향상시켰으며, 수학적 정확성을 유지하는 데 주목할 만한 성과를 보였습니다. 이를 통해 교육자들이 수학 교육에서 시각적 보조 도구를 보다 효과적으로 활용할 수 있게 되었습니다.



### Learning the rules of peptide self-assembly through data mining with large language models (https://arxiv.org/abs/2411.05421)
- **What's New**: 이 연구에서는 펩타이드 자가 조립 행동에 대한 실험적 데이터셋을 수집하고, 이를 통해 머신 러닝 모델을 훈련하여 자가 조립 단계(Peptide Assembly Phase)를 예측합니다. 특히, 기존의 문헌 데이터베이스에서 1,000개 이상의 실험 데이터를 수집하여 자가 조립의 기본 규칙을 이해하는 데 기여하고자 하였습니다.

- **Technical Details**: 제안된 연구에서는 SAPdb 데이터베이스에서 문헌 채굴(literature mining)을 통해 1,012개의 데이터를 수집하고, 이를 기반으로 다양한 머신 러닝 알고리즘을 훈련합니다. 다수의 ML 알고리즘(예: Random Forest, GNN, Transformer 기반 모델)을 비교하여 최적의 성능을 발휘하는 모델을 선정하고, 각 모델을 사용하여 자가 조립 단계를 분류합니다. 또한, OpenAI API를 통해 GPT-3.5 Turbo 모델을 미세 조정하여 정보 추출 성능을 향상시킵니다.

- **Performance Highlights**: 수집된 데이터셋을 활용한 머신 러닝 모델은 자가 조립 단계 분류에서 80% 이상의 높은 정확도를 달성하였으며, 직관적인 데이터 처리를 통해 실험 효율성을 극대화하는 데 기여합니다. Fine-tuned GPT 모델은 기존 모델 대비 학술 출판물에서 정보 추출에서 뛰어난 성능을 나타내며, 자가 조립 펩타이드 후보를 탐색할 때 실험 작업을 안내하는 데 유용하며, 바이오 재료, 센서 및 촉매와 같은 다양한 응용 분야에서의 새로운 구조 접근을 용이하게 합니다.



### WeatherGFM: Learning A Weather Generalist Foundation Model via In-context Learning (https://arxiv.org/abs/2411.05420)
- **What's New**: 본 논문에서는 WeatherGFM이라는 첫 번째의 일반화된 날씨 기반 모델을 소개합니다. 이 모델은 단일 모델 내에서 다양한 날씨 이해 작업을 통합하여 처리 가능하도록 설계되었습니다.

- **Technical Details**: WeatherGFM은 날씨 이해 작업에 대한 단일화된 표현 및 정의를 처음으로 통합하고, 단일, 다중, 시간 모달리티를 관리하기 위한 날씨 프롬프트 포맷을 고안합니다. 또한, 통합된 날씨 이해 작업의 교육을 위해 시각적 프롬프트 기반의 질문-응답 패러다임을 채택하였습니다.

- **Performance Highlights**: 광범위한 실험 결과, WeatherGFM은 날씨 예보, 초해상도, 날씨 이미지 변환, 후처리 등 최대 10개의 날씨 이해 작업을 효과적으로 처리할 수 있으며, 보지 못한 작업에 대한 일반화 능력을 보여줍니다.



### Web Archives Metadata Generation with GPT-4o: Challenges and Insights (https://arxiv.org/abs/2411.05409)
- **What's New**: 이번 연구는 GPT-4o를 활용하여 웹 아카이브 싱가포르의 메타데이터 생성을 자동화하는 방법을 탐구하였습니다. 연구의 주요 목표는 비용 효율성과 효율성을 높이고, 112개의 WARC(Web ARChive) 파일을 처리하여 메타데이터 생성 비용을 99.9% 절감하는 성과를 달성한 것입니다.

- **Technical Details**: 연구는 Prompt Engineering 기법을 사용하여 제목과 요약을 생성하였으며, 생성된 메타데이터의 질을 Levenshtein Distance 및 BERTScore 같은 내부 평가 방법과 McNemar's test를 통해 외부 평가하였습니다. 결과적으로, 자동 생성된 메타데이터는 상당한 비용 절감을 제공했지만, 사람에 의해 작성된 메타데이터가 여전히 질적인 측면에서 우위를 점하고 있음을 확인하였습니다.

- **Performance Highlights**: 이 연구는 Web Archiving 분야에서 LLMs(대형 언어 모델)의 통합 가능성을 한층 발전시켰으며, 데이터 필터링 개선 및 개인정보 보호 문제를 다루기 위한 향후 작업 방향을 제시하였습니다. 연구 결과를 통해 LLMs는 인간 카탈로거를 대체하기보다는 보완하는 역할을 할 것으로 보입니다.



### Benchmarking Distributional Alignment of Large Language Models (https://arxiv.org/abs/2411.05403)
- **What's New**: 이 논문은 언어 모델(LLMs)이 특정 인구 집단의 의견 분포를 얼마나 잘 모사할 수 있는지에 대한 불확실성을 다루고, 이를 위한 벤치마크를 제공하여 세 가지 주요 변수를 탐구합니다: 질문 도메인, 스티어링 방법, 그리고 분포 표현 방법입니다.

- **Technical Details**: 저자들은 LLM의 분포 정렬(distributional alignment)을 평가하기 위해 NYT Book Opinions라는 새로운 데이터세트를 수집하고, 모델이 특정 집단의 의견 분포에 맞추기 위해 어떻게 스티어링 되어야 하는지를 연구합니다. 이 작업은 모델이 의견 분포를 어떻게 제시하는지에 따라 다르며, 기존의 log-probabilities 기반 방식이 LLM의 성능을 과소평가할 수 있음을 발견했습니다.

- **Performance Highlights**: 연구에 따르면, LLM은 텍스트 기반 형식에서 (예: 'JSON으로 분포 반환') 의견 분포를 보다 정확하게 추정할 수 있으며, 정치 및 문화적 가치를 넘어서 비문화적 의견(예: 책 선호도)의 정렬 및 스티어링에서 상당한 격차가 존재합니다. 이로 인해 LLM의 인간 의견 시뮬레이션 능력을 향상시킬 기회가 부각됩니다.



### Advancing Meteorological Forecasting: AI-based Approach to Synoptic Weather Map Analysis (https://arxiv.org/abs/2411.05384)
- **What's New**: 이 연구는 기상 예측의 정확성을 높이기 위한 새로운 전처리 방법 및 합성곱 오토인코더(Convolutional Autoencoder) 모델을 제안합니다. 이 모델은 과거의 기상 패턴과 현재의 대기 조건을 비교 분석하여 기상 예측의 효율성을 향상시킬 수 있는 잠재력을 가지고 있습니다.

- **Technical Details**: 연구는 VQ-VQE와 같은 비지도 학습(un_supervised learning) 모델과 VGG16, VGG19, Xception, InceptionV3, ResNet50과 같은 지도 학습(supervised learning) 모델을 포함합니다. 모델의 성능은 유사성 비교를 위한 지표로 코사인 유사도(cosine similarity)를 사용하여, 과거 기상 패턴을 정확하게 식별할 수 있도록 최적화 과정이 이루어졌습니다.

- **Performance Highlights**: 연구 결과, 제안된 모델은 기존의 사전 학습(pretrained) 모델들과 비교할 때 과거 기상 데이터의 식별에서 우수한 성능을 보였으나, 유사성을 판별하는 데 한계도 존재했습니다. 이 모델은 기상학자들이 정보를 더욱 신속하고 정밀하게 분석할 수 있도록 도와줍니다.



### Ev2R: Evaluating Evidence Retrieval in Automated Fact-Checking (https://arxiv.org/abs/2411.05375)
Comments:
          10 pages

- **What's New**: 본 논문에서는 Automated Fact-Checking (AFC)을 위한 새로운 평가 프레임워크인 Ev2R을 소개합니다. Ev2R은 증거 평가를 위한 세 가지 접근 방식인 reference-based, proxy-reference, reference-less 평가 방식으로 구성되어 있습니다.

- **Technical Details**: Ev2R 프레임워크의 세 가지 평가자 그룹은 (1) reference-based 평가자: 참조 증거와 비교하여 검색된 증거를 평가, (2) proxy-reference 평가자: 시스템이 예측한 평결을 기준으로 증거를 평가, (3) reference-less 평가자: 입력 클레임만을 기반으로 증거를 평가하는 방식입니다. 구체적으로 LLMs를 활용하여 증거를 원자적 사실로 분해한 후 평가합니다.

- **Performance Highlights**: Ev2R의 reference-based 평가자는 전통적인 메트릭보다 인간 평가와 높은 상관관계를 보였습니다. Gemini 기반 평가자는 검색된 증거가 참조 증거를 얼마나 포함하고 있는지를 잘 평가한 반면, GPT-4 기반 평가자는 평결 일치에서 더 좋은 성능을 보였습니다. 이 결과는 Ev2R 프레임워크가 더욱 정확하고 강력한 증거 평가를 가능하게 함을 시사합니다.



### Agricultural Landscape Understanding At Country-Sca (https://arxiv.org/abs/2411.05359)
Comments:
          34 pages, 7 tables, 15 figs

- **What's New**: 이 연구는 인도 지역에서 농업 경관을 디지털화하는 혁신적인 접근 방식을 제시합니다. 높은 해상도의 위성 이미지를 사용하여 국가 규모의 다중 클래스 팬옵틱(segmentation) 세분화 결과물을 생성한 최초의 사례입니다.

- **Technical Details**: 연구는 빨강, 초록, 파랑(RGB) 채널의 위성 이미지를 이용한 다중 클래스 팬옵틱(segmentation) 문제로 농업 경관 이해를 공식적으로 모델링합니다. U-Net 아키텍처를 활용하여 개별 필드, 나무, 물체 경계를 픽셀 수준에서 분류합니다.

- **Performance Highlights**: 본 연구는 소규모 농장을 강조하며, 면적 151.7M 헥타르에 걸쳐 수백만 개의 소규모 농장 필드와 수백만 개의 소규모 관개 구조물을 성공적으로 식별했습니다. 또한, 엄격한 현장 검증을 통해 모델의 신뢰성을 보장합니다.



### Controlling Grokking with Nonlinearity and Data Symmetry (https://arxiv.org/abs/2411.05353)
Comments:
          15 pages, 14 figures

- **What's New**: 이 논문은 신경망에서 모듈러 산술(modular arithmetic)의 grokking 행동을 활성화 함수(activation function)의 프로필 수정과 모델의 깊이(depth) 및 너비(width)를 변경하여 제어할 수 있음을 보여줍니다.

- **Technical Details**: 마지막 신경망(layer) 층의 가중치(weights)의 짝수 PCA 투영(even PCA projections)과 홀수 투영(odd projections)을 플로팅하여 비선형성(nonlinearity)을 증가시키면서 패턴이 더욱 고른 형태로 변하는 것을 발견했습니다. 이러한 패턴은 P가 비소수(nonprime)일 때 P를 인수분해(factor)하는 데 사용될 수 있습니다.

- **Performance Highlights**: 신경망의 일반화 능력(generalization ability)은 층 가중치의 엔트로피(entropy)로부터 유도되며, 비선형성의 정도는 최종 층의 뉴런 가중치(local entropy) 간의 상관관계(correlations)와 관련이 있습니다.



### Reasoning Robustness of LLMs to Adversarial Typographical Errors (https://arxiv.org/abs/2411.05345)
- **What's New**: 본 연구에서는 Chain-of-Thought (CoT) 프롬프트를 사용하는 대형 언어 모델(Large Language Models, LLMs)의 추론 강건성에 대한 연구를 진행했습니다. 특히, 사용자 쿼리에서 발생할 수 있는 오타(typographical errors)의 영향에 주목했습니다. 따라서, Adversarial Typo Attack (ATA) 알고리즘을 설계하여, 쿼리에서 중요한 단어의 오타를 반복적으로 샘플링하고 성공 가능성이 높은 수정을 선택하는 과정을 통해 LLM의 오타에 대한 민감성을 입증하였습니다.

- **Technical Details**: ATA(Adversarial Typo Attack) 알고리즘은 입력에서 중요한 토큰을 추출하고, 각 선택 단어에 대해 타이핑 오류를 샘플링한 후 수정된 입력의 손실을 평가하여 최적 후보를 보존하는 방식으로 작동합니다. 이 과정에서, 모델이 잘못된 답을 생성하도록 유도하는 단순한 타이핑 오류를 포함한 다양한 수정 방식을 적용하였습니다. 연구에서는 R^2ATA 벤치마크를 개발하여 GSM8K, BBH, MMLU의 세 가지 추론 데이터셋을 사용하여 LLM의 추론 강건성을 평가했습니다.

- **Performance Highlights**: 실험 결과, Mistral-7B-Instruct 모델의 경우 단일 문자 수정으로 정확도가 43.7%에서 38.6%로 감소하였고, 8자리 수정으로 성능이 19.2%로 추가 하락했습니다. R2ATA 평가에서 고급 모델들 역시 서로 다른 취약성을 보였으며, 예를 들어 Vicuna-33B-chat의 경우, GSM8K에서 38.2%에서 26.4%로, BBH에서는 52.1%에서 42.5%로, MMLU에서는 59.2%에서 51.5%로 성능이 감소했습니다.



### Improving Multi-Domain Task-Oriented Dialogue System with Offline Reinforcement Learning (https://arxiv.org/abs/2411.05340)
- **What's New**: 이번 논문은 사전 훈련된 GPT-2 모델을 활용하여 사용자 정의 작업을 수행하는 대화형 시스템(TOD)을 제안합니다. 주요 특징은 감독 학습(Supervised Learning)과 강화 학습(Reinforcement Learning)을 결합하여, 성공률(Success Rate)과 BLEU 점수를 기반으로 보상 함수를 최적화하여 모델의 성능을 향상시키는 것입니다.

- **Technical Details**: 제안된 TOD 시스템은 사전 훈련된 대형 언어 모델인 GPT-2를 기반으로 하며, 감독 학습과 강화 학습을 통해 최적화되었습니다. 비가역적 보상 함수를 사용하여 대화 결과의 성공률과 BLEU 점수를 가중 평균하여 보상을 계산합니다. 모델은 사용자 발화, 신념 상태(Belief State), 시스템 액트(System Act), 시스템 응답(System Response)으로 구성된 대화 세션 수준에서 미세 조정(Fine-tuning)됩니다.

- **Performance Highlights**: MultiWOZ2.1 데이터셋에서 실험 결과, 제안한 모델은 기준 모델에 비해 정보 비율(Inform Rate)을 1.60% 증가시키고 성공률(Success Rate)을 3.17% 향상시켰습니다.



### Inversion-based Latent Bayesian Optimization (https://arxiv.org/abs/2411.05330)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 본 논문에서는 Latent Bayesian Optimization (LBO) 방법의 문제점을 해결하기 위해 새로운 모듈인 Inversion-based Latent Bayesian Optimization (InvBO)를 제안합니다. 주로 'misalignment problem'과 trust region anchor selection의 한계를 다룹니다.

- **Technical Details**: InvBO는 두 가지 주요 구성 요소로 구성됩니다: inversion method와 potential-aware trust region anchor selection. Inversion method는 주어진 타겟 데이터를 완전히 재구성하는 latent code를 검색하며, potential-aware anchor selection은 optimization 과정의 향상을 고려하여 trust region의 중심을 선택합니다.

- **Performance Highlights**: 실험 결과, InvBO는 9개의 실제 벤치마크(예: molecule design, arithmetic expression fitting)의 성능에서 기존 방법에 비해 큰 성능 향상을 보이며, 최신 기술 수준(state-of-the-art)을 달성했습니다.



### Exploring the Alignment Landscape: LLMs and Geometric Deep Models in Protein Representation (https://arxiv.org/abs/2411.05316)
Comments:
          24 pages, 9 figures

- **What's New**: 이번 연구는 단백질 분야에서 LLM(대형 언어 모델)과 GDM(기하학적 딥 모델) 간의 다중 모드 표현 정렬을 탐구하고, 이를 통해 단백질 관련 MLLM(다중 모드 대형 언어 모델)의 성능을 향상시키는 방법을 제안합니다.

- **Technical Details**: 연구에서는 Gemma2-2B, LLaMa3.1-8B, LLaMa3.1-70B 등 세 가지 최신 LLM과 GearNet, GVP, ScanNet, GAT 등 네 가지 단백질 전문 GDM을 평가했습니다. 또한, 두 가지 층 프로젝션 헤드를 사용하고 LLM을 단백질 특화 데이터로 세부 조정(fine-tuning)하면 정렬 품질이 크게 향상된다는 것을 발견했습니다.

- **Performance Highlights**: 정렬 품질을 향상시키는 여러 전략을 제안하며, GDM이 그래프 및 3D 구조 정보를 통합할수록 LLM과의 정렬 성능이 개선되고, 더 큰 LLM이 더 나은 정렬 능력을 보여주며, 단백질의 희귀성이 정렬 성능에 큰 영향을 미친다는 주요 결과를 도출했습니다.



### Revisiting Network Perturbation for Semi-Supervised Semantic Segmentation (https://arxiv.org/abs/2411.05307)
Comments:
          Accepted by PRCV2024

- **What's New**: 이번 연구에서는 반지도 식별(Semi-supervised semantic segmentation, SSS)에서 네트워크 perturbation (perturbation)을 효과적으로 통합하는 새로운 접근법인 MLPMatch를 제안합니다.

- **Technical Details**: MLPMatch는 Deep Neural Network (DNN)의 특정 레이어를 무작위로 비활성화(deactivate)하여 네트워크 perturbation을 수행하고, 이는 단일 네트워크에서 쉽게 구현될 수 있습니다. 또한, 레이블이 있는 데이터와 레이블이 없는 데이터 모두에 대해 volatile learning 과정을 도입하여 성능을 극대화합니다.

- **Performance Highlights**: MLPMatch는 Pascal VOC와 Cityscapes 데이터셋에서 최첨단(State-of-the-art) 성능을 달성하였으며, 기존의 방법들과 비교하여 더욱 효율적이고 간단한 접근법으로 평가받고 있습니다.



### On Training of Kolmogorov-Arnold Networks (https://arxiv.org/abs/2411.05296)
Comments:
          7 pages, 6 figures

- **What's New**: 최근 Kolmogorov-Arnold Networks (KAN)은 다층 Perceptron 아키텍처에 대한 유연한 대안으로 소개되었습니다. 이 논문에서는 여러 KAN 아키텍처의 훈련 동역학을 분석하고 이를 해당하는 MLP 형태와 비교합니다.

- **Technical Details**: KAN 아키텍처는 B-spline을 사용하여 입력 데이터에 적합하면서 웨이브릿 개념을 기반으로 하며, HSIC Bottleneck과 같은 백 프로파게이션을 사용하지 않는 방법으로 훈련됩니다. 우리는 다양한 초기화 관리 기법과 최적화 기법, 학습률을 바탕으로 여러 가지 조합을 실험했습니다.

- **Performance Highlights**: 테스트 정확도에 따라 KAN은 고차원 데이터셋에서 MLP 아키텍처에 대한 효과적인 대안으로 간주되며, 파라미터 효율성에서 약간 더 우수하나 훈련 동역학에서는 더 불안정한 경향이 있음을 발견했습니다. 논문에서는 KAN 모델의 훈련 안정성을 향상시키기 위한 추천 사항도 제공합니다.



### SimpleBEV: Improved LiDAR-Camera Fusion Architecture for 3D Object Detection (https://arxiv.org/abs/2411.05292)
- **What's New**: 본 연구에서는 LiDAR(라이더)와 카메라 정보를 통합하여 자율주행 시스템의 3D 객체 탐지 성능을 향상시키는 단순하면서도 효과적인 새로운 융합 프레임워크인 SimpleBEV를 제안합니다.

- **Technical Details**: SimpleBEV는 BEV(Bird's Eye View) 기반으로 LiDAR와 카메라 기능을 통합합니다. 카메라 기반 깊이 추정을 위해 2단계 캐스케이드 네트워크를 사용하고, LiDAR 포인트에서 파생된 깊이 정보를 통해 깊이 결과를 수정합니다. 3D 객체 탐지를 위한 보조 분기를 도입하여 카메라 정보의 활용성을 높이고, 멀티 스케일 희소 합성곱 특징을 융합하여 LiDAR 특징 추출기를 개선합니다.

- **Performance Highlights**: 실험 결과, 제안하는 방법은 nuScenes 데이터 세트에서 77.6%의 NDS 정확도를 달성하였으며, 이는 3D 객체 탐지 분야에서 타의 추종을 불허하는 성능을 보여줍니다. 모델 앙상블과 테스트 타임 증강을 통해 최고의 NDS 점수를 기록했습니다.



### SpecHub: Provable Acceleration to Multi-Draft Speculative Decoding (https://arxiv.org/abs/2411.05289)
Comments:
          EMNLP 2024 (Main)

- **What's New**: 대규모 언어 모델(LLMs)의 인퍼런스 속도를 개선하기 위한 새로운 방법, SpecHub를 소개합니다. 기존 방식인 Recursive Rejection Sampling (RRS)의 한계를 극복하고, 다수의 토큰 초안을 효율적으로 검증하는 혁신적인 기법입니다.

- **Technical Details**: SpecHub는 Multi-Draft Speculative Decoding (MDSD)의 새로운 샘플링-검증 방법으로, Optimal Transport 문제를 간소화하여 선형 프로그래밍 모델(Linear Programming model)로 변환합니다. 이를 통해 계산 복잡성을 입증적으로 줄이고 희소 분포(sparse distribution)를 활용하여 높은 확률의 토큰 시퀀스에 계산을 집중시킵니다.

- **Performance Highlights**: SpecHub는 RRS와 비교하여 매 스텝 당 0.05-0.27, RRS without replacement에 비해 0.02-0.16 더 많은 토큰을 생성하며, 기존 방법에 비해 1-5%의 두 번째 초안 수락률 증가를 기록합니다. 또한, 동일한 배치 효율성을 달성하기 위해 다른 방법의 절반에 해당하는 노드를 가진 트리를 사용합니다.



### MicroScopiQ: Accelerating Foundational Models through Outlier-Aware Microscaling Quantization (https://arxiv.org/abs/2411.05282)
Comments:
          Under review

- **What's New**: 본 논문에서는 MicroScopiQ라는 새로운 공동 설계 기법을 제안합니다. 이는 pruning과 outlier-aware quantization을 결합하여 높은 정확도와 메모리 및 하드웨어 효율성을 달성하는 방안을 제시합니다.

- **Technical Details**: MicroScopiQ는 outlier를 더 높은 정밀도로 유지하면서 중요하지 않은 일부 가중치를 pruning합니다. 새로운 NoC(네트워크 온 칩) 아키텍처인 ReCoN을 통해 하드웨어의 복잡성을 효율적으로 추상화하고, 다양한 비트 정밀도를 지원하는 가속기 아키텍처를 설계합니다. MicroScaling 데이터 형식을 활용하여 아웃라이어 가중치를 정량화하는 방식도 포함되어 있습니다.

- **Performance Highlights**: MicroScopiQ는 다양한 정량화 설정에서 기존 기법 대비 3배 향상된 추론 성능을 달성하고, 에너지를 2배 절감하여 SoTA(SOTA: State of the Art) 정량화 성능을 달성합니다.



### Fox-1 Technical Repor (https://arxiv.org/abs/2411.05281)
Comments:
          Base model is available at this https URL and the instruction-tuned version is available at this https URL

- **What's New**: Fox-1은 새로운 훈련 커리큘럼 모델링을 도입한 소형 언어 모델(SLM) 시리즈로, 3조 개의 토큰과 5억 개의 지침 데이터로 사전 훈련 및 미세 조정되었습니다. 이 모델은 그룹화된 쿼리 어텐션(Grouped Query Attention, GQA)과 깊은 레이어 구조를 가지고 있어 성능과 효율성을 향상시킵니다.

- **Technical Details**: Fox-1-1.6B 모델은 훈련 데이터를 3단계 커리큘럼으로 구분하여 진행하며, 2K-8K 시퀀스 길이를 처리합니다. 데이터는 오픈 소스의 다양한 출처에서 수집된 고품질 데이터를 포함하고 있습니다. 이 모델은 텐서 오페라(TensorOpera) AI 플랫폼과 훌리 지픈(Facing)에서 공개되며, Apache 2.0 라이센스 하에 이용 가능합니다.

- **Performance Highlights**: Fox-1은 StableLM-2-1.6B, Gemma-2B와 같은 여러 벤치마크에서 경쟁력 있는 성능을 보여주며 빠른 추론 속도와 처리량을 자랑합니다.



### Real-World Offline Reinforcement Learning from Vision Language Model Feedback (https://arxiv.org/abs/2411.05273)
Comments:
          7 pages. Accepted at the LangRob Workshop 2024 @ CoRL, 2024

- **What's New**: 새로운 시스템을 제안하여, Vision-Language Model (VLM)의 선호 피드백을 사용하여 오프라인 데이터셋의 보상 레이블을 자동으로 생성할 수 있도록 했습니다. 이 방법을 통해 라벨이 없는 서브 최적 오프라인 데이터셋에서도 효과적으로 정책(policy)을 학습할 수 있게 됩니다.

- **Technical Details**: 정확한 보상 레이블을 생성하기 위해, Vision-Language Model (VLM) 피드백을 활용한 기존 연구인 RL-VLM-F를 기반으로 하여, 주어진 오프라인 데이터셋으로부터 선호 데이터셋을 생성합니다. 이후 생성된 데이터셋을 통해 보상 함수를 학습하고, 이를 이용하여 주어진 데이터셋에 레이블을 붙입니다. 이렇게 레이블이 붙은 데이터셋은 기존의 오프라인 강화 학습 프레임워크를 통해 제어 정책을 학습하는 데 활용됩니다.

- **Performance Highlights**: 제안한 시스템은 복잡한 현실 세계의 로봇 보조 의상 착용 작업에 적용되었으며, 비최적 오프라인 데이터셋에서 효과적인 보상 함수와 정책을 학습하여 기존의 Behavior Cloning (BC) 및 Inverse Reinforcement Learning (IRL) 방법들보다 우수한 성능을 보였습니다.



### Seeing Through the Fog: A Cost-Effectiveness Analysis of Hallucination Detection Systems (https://arxiv.org/abs/2411.05270)
Comments:
          18 pags, 13 figures, 2 tables

- **What's New**: 이번 논문은 AI를 위한 hallucination detection 시스템의 비교 분석을 제공합니다. 주로 Large Language Models (LLMs)에 대한 자동 요약 및 질문 응답 작업을 중점적으로 다루고 있습니다.

- **Technical Details**: 다양한 hallucination detection 시스템을 진단 오즈 비율 (diagnostic odds ratio, DOR)과 비용 효율성 (cost-effectiveness) 메트릭스를 사용하여 평가하였습니다. 고급 모델이 더 나은 성능을 보일 수 있지만, 비용이 상당히 증가함을 보여주고 있습니다.

- **Performance Highlights**: 이상적인 hallucination detection 시스템은 다양한 모델 크기에서 성능을 유지해야 함을 입증하였으며, 특정 응용 프로그램 요구 사항과 자원 제약에 맞는 시스템 선택의 중요성을 강조하고 있습니다.



### Decoding Report Generators: A Cyclic Vision-Language Adapter for Counterfactual Explanations (https://arxiv.org/abs/2411.05261)
- **What's New**: 보고서 생성 모델에서 생성된 텍스트의 해석 가능성을 향상시키기 위한 혁신적인 접근 방식이 소개되었다. 이 방법은 사이클 텍스트 조작(cyclic text manipulation)과 시각적 비교(visual comparison)를 활용하여 원본 콘텐츠의 특징을 식별하고 설명한다.

- **Technical Details**: 새롭게 제안된 접근 방식은 Cyclic Vision-Language Adapters (CVLA)를 활용하여 설명의 생성 및 이미지 편집을 수행한다. counterfactual explanations를 통해 원본 이미지와 대조하여 수정된 이미지와의 비교를 통해 해석을 제공하며, 이는 모델에 구애받지 않는 방식으로 이루어진다.

- **Performance Highlights**: 이 연구의 방법론은 다양한 현재 보고서 생성 모델에서 적용 가능하며, 생성된 보고서의 신뢰성을 평가하는 데 기여할 것으로 기대된다.



### QuanCrypt-FL: Quantized Homomorphic Encryption with Pruning for Secure Federated Learning (https://arxiv.org/abs/2411.05260)
- **What's New**: 이 논문은 QuanCrypt-FL이라는 새로운 알고리즘을 제안하며, 이는 저비트 양자화(low-bit quantization)와 가지치기(pruning) 기법을 결합하여 훈련 중 공격으로부터의 보호를 강화하는 동시에 계산 비용을 크게 줄이는 것을 목표로 합니다.

- **Technical Details**: QuanCrypt-FL은 동적 층별 클리핑(mean-based clipping) 기법을 사용하여 양자화 중 발생할 수 있는 오버플로우 문제를 완화하고, 전체 모델 업데이트를 안전하게 암호화하여 전달하는 방법을 채택합니다. 이를 통해 훈련 시간과 통신 비용을 줄이고, 모델의 정확성에 미치는 영향을 최소화합니다.

- **Performance Highlights**: QuanCrypt-FL은 MNIST, CIFAR-10, CIFAR-100 데이터셋에서 기존 방법보다 우수한 성능을 발휘했습니다. 또한, BatchCrypt와 비교했을 때 최대 9배 빠른 암호화 속도, 16배 빠른 복호화 속도, 1.5배 더 빠른 추론 속도를 달성하며 훈련 시간이 최대 3배 단축됨을 입증하였습니다.



### Abstract2Appendix: Academic Reviews Enhance LLM Long-Context Capabilities (https://arxiv.org/abs/2411.05232)
Comments:
          We share our latest dataset on this https URL

- **What's New**: 이 연구는 대형 언어 모델(LLM)의 긴 문맥 처리를 향상시키기 위해 고품질 학술 피어 리뷰 데이터를 활용한 효과성을 탐구합니다. 연구 결과, Direct Preference Optimization (DPO) 방법이 Supervised Fine-Tuning (SFT) 방법에 비해 우수하고 데이터 효율성이 높음을 입증했습니다.

- **Technical Details**: 본 연구에서는 2000개의 PDF로부터 파생된 데이터셋을 사용하여 DPO와 SFT 방식을 비교했습니다. DPO를 통해 모델의 긴 텍스트 이해 능력을 효과적으로 향상시킬 수 있음을 보여주었습니다. DPO 실험에서 모델 성능 개선을 위해 GPT-4를 사용하여 피어 리뷰들을 집계하였고, 피어 리뷰 데이터 사용을 통해 모델의 추가적인 감독 신호를 제공하는 방법을 모색했습니다.

- **Performance Highlights**: DPO로 미세 조정된 phi-3-mini-128k 모델은 phi-3-mini-4B-128k 모델보다 평균 4.04 점 향상된 결과를 보였고, Qasper 벤치마크에서 2.6% 증가를 기록했습니다. 이는 제한된 데이터만으로도 긴 문맥 읽기 능력을 크게 개선할 수 있음을 나타냅니다.



### Maximizing User Connectivity in AI-Enabled Multi-UAV Networks: A Distributed Strategy Generalized to Arbitrary User Distributions (https://arxiv.org/abs/2411.05205)
- **What's New**: 이 연구에서는 Multi-Unmanned Aerial Vehicle (MUN) 환경에서 임의의 사용자 분포(User Distribution, UD)에 대해 적응할 수 있는 깊은 강화 학습(Deep Reinforcement Learning, DRL) 방법론을 제안합니다. 특히, UAV의 사용자 연결성 극대화 문제를 다루고 있으며, 기존의 연구와 달리 알려지지 않은 환경에서도 효과적으로 동작할 수 있는 전략을 수립합니다.

- **Technical Details**: 본 연구는 시간을 결합한 조합 비선형 비볼록 최적화 문제로 정의된다. 이를 해결하기 위해 ResNet 기반의 CNN을 활용한 다중 에이전트 CNN 강화 깊은 Q 학습(MA-CDQL) 알고리즘을 제안합니다. 이 알고리즘은 사용자 분포의 고차원 특징을 실시간으로 분석하여 정책 네트워크에 중요한 입력으로 사용합니다. 추가적으로, RAW UD를 연속 밀도 맵으로 변환하는 히트맵 알고리즘을 개발하여 학습 효율성을 높이고 로컬 최적에 빠지는 것을 방지합니다.

- **Performance Highlights**: 시뮬레이션 결과, 제안된 MA-CDQL 알고리즘은 K-means 방법과 비교하여 사용자 연결성을 극대화하는 데 효과적임을 입증하였습니다. UD 히트맵을 활용하여 최적의 성능을 달성하며, 이를 통해 MUN이 알려지지 않은 환경에서도 뛰어난 유연성과 적응성을 갖출 수 있음을 보여줍니다.



### Interactive Dialogue Agents via Reinforcement Learning on Hindsight Regenerations (https://arxiv.org/abs/2411.05194)
Comments:
          23 pages, 5 figures

- **What's New**: 본 논문에서는 대화 에이전트가 대화를 효과적으로 이끌 수 있는 능력을 향상시키기 위한 새로운 방법을 제안합니다. 특히, 기존 데이터에 대한 후행(regeneration)을 통해 대화 에이전트를 훈련시키는 방법을 소개하며, 이는 특히 정신 건강 지원 및 자선 기부 요청과 같은 복잡한 대화 작업에 적용됩니다.

- **Technical Details**: 대화 에이전트는 오프라인 강화 학습(offline reinforcement learning, RL)을 사용하여 훈련됩니다. 기존 비효율적인 대화 데이터를 개선하고 새로운 대화 전략을 학습하기 위해, 사후에 생성된 합성 데이터(synthetic data)를 추가하여 적절한 행동을 나타내는 다양한 대화 전략을 포착합니다.

- **Performance Highlights**: 실제 사용자를 대상으로 한 연구 결과, 제안된 방법이 기존 최첨단 대화 에이전트에 비해 효과성, 자연스러움 및 유용성 측면에서 크게 우수함을 보여주었습니다.



### Q-SFT: Q-Learning for Language Models via Supervised Fine-Tuning (https://arxiv.org/abs/2411.05193)
Comments:
          16 pages, 4 figures

- **What's New**: 이번 연구에서는 기존의 가치 기반 강화 학습(value-based reinforcement learning, RL) 알고리즘의 한계를 극복하기 위한 새로운 오프라인 RL 알고리즘을 제안합니다. 이 알고리즘은 Q-learning을 수정된 감독 세부 조정(supervised fine-tuning, SFT) 문제로 간주하며, 이를 통해 언어 모델의 사전 학습(pretraining) 효과를 충분히 활용할 수 있습니다.

- **Technical Details**: 제안된 알고리즘은 최대 우도(maximum likelihood) 목표에 가중치를 추가하여, 잔여 정책(behavior policy) 대신 보수적으로 가치 함수를 추정하는 확률을 학습합니다. 이를 통해 비안정적인 가치 학습(regression) 목표를 피하고, 대규모 사전 학습에서 발생한 초기 우도(likelihood)를 직접 활용할 수 있습니다.

- **Performance Highlights**: 실험을 통해 LLM(대형 언어 모델)과 VLM(비전-언어 모델)의 다양한 작업에서 제안된 알고리즘의 효과를 입증했습니다. 자연어 대화, 로봇 조작 및 네비게이션 등 여러 과제에서 기존의 감독 세부 조정 방법을 초월하는 성능을 보였습니다.



### Explaining Mixtures of Sources in News Articles (https://arxiv.org/abs/2411.05192)
Comments:
          9 pages

- **What's New**: 이 연구에서는 인간 작가의 글쓰기 전 계획 단계를 이해하는 것이 중요하다는 점을 강조하며, 뉴스에서 출처 선택(Source-Selection)을 하나의 사례 연구로 설정하여 장기 형식 기사 생성을 평가합니다.

- **Technical Details**: 연구자는 저널리스트와 협력하여 기존의 다섯 가지 출처 선택 스키마를 수정하고 세 가지 새로운 스키마를 도입하였습니다. 베이지안(Bayesian) 잠재변수 모델링에서 영감을 받아, 스토리의 근본적인 계획(혹은 스키마)을 선택하는 메트릭트를 개발하였습니다. 두 가지 주요 스키마(stance, social affiliation)가 대부분의 문서에서 출처 계획을 잘 설명하는 것으로 나타났습니다.

- **Performance Highlights**: 연구자가 수집한 400만 개의 뉴스 기사의 주석이 추가된 데이터셋인 NewsSources를 रिलीज했습니다. 기사의 제목만으로도 적절한 스키마를 예측할 수 있으며, 90,000개의 뉴스 기사에 대한 분석 결과, 47%의 문장이 출처와 연결될 수 있음을 발견했습니다.



### Inverse Transition Learning: Learning Dynamics from Demonstrations (https://arxiv.org/abs/2411.05174)
- **What's New**: 본 논문에서는 오프라인 모델 기반 강화 학습(context of offline model-based reinforcement learning) 환경에서 near-optimal expert trajectories를 통해 상태 전이 동역학 T^*의 추정 문제를 다루고 있으며, Inverse Transition Learning(ITL)이라는 새로운 제약 기반(method)이 개발되었습니다.

- **Technical Details**: Inverse Transition Learning(ITL)은 전문가의 행동 궤적이 불완전한 데이터 샘플 시성을 띄고 있다는 사실을 특징으로 활용하며, Bayesian 접근법을 통합하여 T^*의 추정을 개선합니다. 이 방법은 MCE(Maximum Causal Entropy) 기반의 기존 접근 방식에 비해 더 빠르고 신뢰할 수 있습니다.

- **Performance Highlights**: 이 연구는 인공 환경(synthetic environments) 및 실제 의료 상황(real healthcare scenarios)에서의 효과를 입증하였습니다. ICU(중환자실) 환자 관리를 포함한 테스트는 의사 결정 개선의 상당한 향상을 보였으며, posterior는 전이가 성공할 시기를 예측하는 데 유용함을 나타냈습니다.



### FineTuneBench: How well do commercial fine-tuning APIs infuse knowledge into LLMs? (https://arxiv.org/abs/2411.05059)
- **What's New**: 이 연구에서는 FineTuneBench라는 새로운 평가 프레임워크 및 데이터셋을 소개하여 상업적 LLM(대형 언어 모델) 미세 조정 API가 새로운 정보와 업데이트된 지식을 성공적으로 학습할 수 있는지를 분석합니다.

- **Technical Details**: FineTuneBench 데이터셋은 뉴스, 허구의 인물, 의료 지침, 코드의 네 가지 영역에서 625개의 학습 질문과 1,075개의 테스트 질문으로 구성되어 있습니다. 연구진은 GPT-4o, GPT-3.5 Turbo, Gemini 1.5 Pro 등의 LLM 5개를 활용하여 미세 조정 서비스를 평가했습니다. 이 연구는 새로운 정보를 학습하는 LLM의 능력에 대해 체계적인 평가를 제공합니다.

- **Performance Highlights**: 모델들의 평균 일반화 정확도는 37%로, 새로운 정보를 효과적으로 학습하는 데에는 상당한 한계가 있음을 보여주었습니다. 기존 지식을 업데이트할 때는 평균 19%의 정확도만을 기록하며, 특히 Gemini 1.5 시리즈는 새로운 지식을 학습하거나 기존 지식을 업데이트하는 데에 실패했습니다. GPT-4o mini가 새로운 지식을 주입하고 지식을 업데이트하는 데 가장 효과적인 모델로 판단되었습니다.



### Seeing is Deceiving: Exploitation of Visual Pathways in Multi-Modal Language Models (https://arxiv.org/abs/2411.05056)
- **What's New**: 이 논문은 Multi-Modal Language Models (MLLMs)에서 시각적 입력과 텍스트 입력을 악용하는 다양한 공격 전략을 리뷰합니다. 특히, VLATTACK, HADES, Collaborative Multimodal Adversarial Attack (Co-Attack)와 같은 고급 공격 방법론을 소개하며, 이러한 공격이 고안된 모델들의 신뢰성을 어떻게 저하시키는지를 다룹니다.

- **Technical Details**: MLLMs는 비전 인코더(vision encoders)와 언어 모델(language models)을 결합하여 시각 및 언어 데이터의 통합을 시도합니다. 논문에서는 다양한 공격 모드가 어떻게 데이터를 변형하고, 기존의 보안 메커니즘을 우회할 수 있는지를 설명합니다. 주요 기술적 접근으로는 Cross-Attention 메커니즘과 Feature Fusion이 포함되어 있습니다.

- **Performance Highlights**: 실험을 통해 다양한 유형의 공격이 MLLMs의 성능을 어떻게 저해할 수 있는지를 분석하였으며, SmoothVLM 프레임워크, 픽셀 단위 무작위화(pixel-wise randomization), MirrorCheck와 같은 방어 메커니즘이 갖는 강점과 한계를 논의합니다. 이를 통해 더 안전하고 신뢰할 수 있는 MLLM 시스템의 다음 단계에 대한 통찰을 제시합니다.



### Integrating Large Language Models for Genetic Variant Classification (https://arxiv.org/abs/2411.05055)
Comments:
          21 pages, 7 figures

- **What's New**: 본 연구에서는 변이 예측에 있어 최신 Large Language Models (LLMs), 특히 GPN-MSA, ESM1b 및 AlphaMissense를 통합하여 변이의 병원성을 예측하는 새로운 접근 방식을 제시합니다. 이 모델들은 DNA 및 단백질 서열 데이터와 구조적 통찰력을 활용하여 변이 분류를 위한 포괄적인 분석 프레임워크를 형성합니다.

- **Technical Details**: GPN-MSA는 MSA (Multiple Sequence Alignment) 데이터를 기반으로 한 DNA 언어 모델로, 100종의 종에서 수집된 진화 정보를 활용해 병원성 점수를 예측합니다. ESM1b는 단백질 언어 모델로, 20가지의 아미노산 변이에 대한 병원성을 예측하며, AlphaMissense는 단백질 서열로부터 구조를 예측한 후 병원성 예측을 수행합니다. 이들 모델을 통합하여 보다 정확하고 포괄적인 변이 병원성 예측 도구를 개발하였습니다.

- **Performance Highlights**: 연구 결과, 통합 모델들이 기존의 최첨단 도구에 비해 해석이 모호하고 임상적으로 불확실한 변이를 처리하는 데 있어 상당한 개선을 보여주었으며, ProteinGym 및 ClinVar 데이터셋에서 새로운 성능 기준을 설정했습니다. 이 결과는 진단 프로세스의 품질을 향상시키고 개인 맞춤형 의학의 경계를 확장할 수 있는 잠재력을 가지고 있습니다.



### FMEA Builder: Expert Guided Text Generation for Equipment Maintenanc (https://arxiv.org/abs/2411.05054)
Comments:
          4 pages, 2 figures. AI for Critical Infrastructure Workshop @ IJCAI 2024

- **What's New**: 이 논문에서는 산업 장비와 관련된 구조화된 문서, 특히 Failure Mode and Effects Analysis (FMEA) 생성에 대한 새로운 AI 시스템을 제안합니다. 이 시스템은 대규모 언어 모델을 활용하여 FMEA 문서를 신속하고 전문가가 감독하는 방식으로 생성할 수 있도록 지원합니다.

- **Technical Details**: 제안된 시스템의 주요 기술적 세부 사항은 단편화된 생성 문제를 해결하기 위해 동적 예시 선택을 통한 Prompting(프롬프트) 접근 방식을 사용하고, LLMs(대규모 언어 모델)의 답변 일관성, 컨텍스트 학습 및 동적 관련 예시 선택 같은 기술을 적용하는 것입니다. 이 시스템은 주제 전문가의 지식을 활용하여 FMEA 문서의 각 섹션을 생성할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, DFSP(Dynamic Few Shot Prompting) 방법이 FMEA의 주요 구성 요소인 장비 경계와 실패 위치를 생성하는 데 있어 가장 높은 품질을 보였으며, LLMs의 성능이 증가하는 경향을 보였습니다. 전문가의 감독 하에 생성된 결과물은 신뢰성 엔지니어에게 중요한 역할을 하며, 전체적인 사용자 피드백은 긍정적이었습니다.



### Intellectual Property Protection for Deep Learning Model and Dataset Intelligenc (https://arxiv.org/abs/2411.05051)
- **What's New**: 본 논문은 딥러닝 모델과 데이터셋의 지적재산권(IP) 보호에 대한 포괄적인 리뷰를 제시합니다. 최근 대형 언어 모델(LLM)과 같은 최신 연구의 발전을 바탕으로, 고급 모델의 보호와 관련된 다양한 방법론과 이론적 배경을 다룹니다.

- **Technical Details**: 이 연구는 모델 지능과 데이터셋 지능의 IP 보호를 모두 포괄하며, IP 보호(IPP) 방법론을 반응적(reactive) 및 능동적(proactive) 관점에서 평가합니다. 또한, 분산 학습 환경에서의 IPP의 도전 과제를 다루며, 각기 다른 공격 유형에 대해 분석합니다.

- **Performance Highlights**: 논문에서는 고급 IP 보호 방법의 한계 및 각 방법론의 강점과 약점을 체계적으로 분석하며, 새로운 연구 방향과 실제 응용 가능성을 제시하고 있습니다. 또한, IPP에 관련된 측정 지표를 구분하여 종합적으로 제시합니다.



### Selecting Between BERT and GPT for Text Classification in Political Science Research (https://arxiv.org/abs/2411.05050)
Comments:
          28 pages, 5 figures, 7 tables

- **What's New**: 이 연구는 GPT 기반 모델과 프롬프트 엔지니어링을 활용하여 데이터가 부족한 상황에서 텍스트 분류 작업에 대한 새로운 대안을 제시합니다. 기존에 일반적으로 사용되던 BERT 모델과의 성능 비교를 통해 GPT 모델 사용의 잠재력을 탐구합니다.

- **Technical Details**: 이 연구에서는 다양한 분류 작업에 걸쳐 BERT 기반 모델과 GPT 기반 모델의 성능을 평가하기 위한 일련의 실험을 수행하였습니다. 실험은 데이터 샘플 수가 적고 복잡성이 다른 분류 작업을 포함하며, 특히 1,000개 이하의 샘플에서 GPT 모델의 제로샷(zero-shot) 및 몇 샷(few-shot) 학습이 BERT 모델과 비교되었음을 강조합니다.

- **Performance Highlights**: 결과에 따르면, GPT 모델을 이용한 제로샷 및 몇 샷 학습은 초기 연구 탐색에 적합하나, 일반적으로 BERT의 파인 튜닝과 동등하거나 부족한 성능을 보였습니다. BERT 모델은 높은 훈련 세트에 도달할 때 더 우수한 성능을 발휘하는 것으로 나타났습니다.



### Leveraging LLMs to Enable Natural Language Search on Go-to-market Platforms (https://arxiv.org/abs/2411.05048)
Comments:
          11 pages, 5 figures

- **What's New**: 이 논문은 Zoominfo 제품을 위한 자연어 쿼리(queries) 처리 솔루션을 제안하고, 대규모 언어 모델(LLMs)을 활용하여 검색 필드를 생성하고 이를 쿼리로 변환하는 방법을 평가합니다. 이 과정은 복잡한 메타데이터를 요구하는 기존의 고급 검색 방식의 단점을 극복하려는 시도입니다.

- **Technical Details**: 제안된 솔루션은 LLM을 사용하여 자연어 쿼리를 처리하며, 중간 검색 필드를 생성하고 이를 JSON 형식으로 변환하여 최종 검색 서비스를 위한 쿼리로 전환합니다. 이 과정에서는 여러 가지 고급 프롬프트 엔지니어링 기법(techniques)과 체계적 메시지(system message),few-shot prompting, 체인 오브 씽킹(chain-of-thought reasoning) 기법을 활용했습니다.

- **Performance Highlights**: 가장 정확한 모델은 Anthropic의 Claude 3.5 Sonnet으로, 쿼리당 평균 정확도(accuracy) 97%를 기록했습니다. 부가적으로, 작은 LLM 모델들 또한 유사한 결과를 보여 주었으며, 특히 Llama3-8B-Instruct 모델은 감독된 미세조정(supervised fine-tuning) 과정을 통해 성능을 향상시켰습니다.



### PhoneLM:an Efficient and Capable Small Language Model Family through Principled Pre-training (https://arxiv.org/abs/2411.05046)
- **What's New**: 이 연구는 작은 언어 모델(SLM)의 설계에서 하드웨어 특성을 고려하지 않던 기존의 방법에서 벗어나, 사전 학습(pre-training) 전에 최적의 런타임 효율성을 추구하는 새로운 원칙을 제시합니다. 이를 바탕으로 PhoneLM이라는 SLM 패밀리를 개발하였으며, 현재 0.5B 및 1.5B 변종이 포함되어 있습니다.

- **Technical Details**: PhoneLM은 스마트폰 하드웨어(예: Qualcomm Snapdragon SoC)를 위해 설계된 사전 학습 및 지시 모델로, 다양한 모델 변형을 포함합니다. SLM 아키텍처는 다층 구조와 피드 포워드 네트워크의 활성화 함수(activation function)와 같은 하이퍼파라미터를 포함하며, 특히 ReLU 활성화를 사용합니다. 연구에서는 100M 및 200M 파라미터 모델을 가지고 다양한 설정에서 런타임 속도(inference speed)와 손실(loss)을 비교했습니다.

- **Performance Highlights**: PhoneLM은 1.5B 모델이 Xiaomi 14에서 58 tokens/second의 속도로 실행되어 대안 모델보다 1.2배 빠르며, 654 tokens/second의 속도를 NPU에서 달성합니다. 또한, 7개의 대표 벤치마크에서 평균 67.3%의 정확도를 기록하며, 기존 비공식 데이터셋에서 훈련된 SLM보다 더 나은 언어 능력을 보입니다. 이 모든 자료는 완전 공개되어 있어 재현 가능성과 투명성을 제공합니다.



### Multi-language Video Subtitle Dataset for Image-based Text Recognition (https://arxiv.org/abs/2411.05043)
Comments:
          12 pages, 5 figures

- **What's New**: 다중 언어 비디오 자막 데이터셋(Multi-language Video Subtitle Dataset)은 텍스트 인식을 지원하기 위해 설계된 포괄적인 컬렉션입니다. 이 데이터셋은 온라인 플랫폼에서 출처를 찾은 24개의 비디오에서 추출한 4,224개의 자막 이미지를 포함하고 있습니다.

- **Technical Details**: 데이터셋에는 태국어 자음, 모음, 성조 기호, 구두점, 숫자, 로마 문자 및 아라비아 숫자를 포함한 157개의 고유 문자가 포함되어 있어 복잡한 배경에서 텍스트 인식의 도전 과제를 해결하기 위한 자원으로 활용될 수 있습니다. 또한, 이미지 내 텍스트의 길이, 글꼴 및 배치에서의 다양성은 심층 학습 모델의 개발 및 평가를 위한 가치 있는 자원을 제공합니다.

- **Performance Highlights**: 이 데이터셋은 비디오 콘텐츠에서의 정확한 텍스트 전사(text transcription)를 촉진하며, 텍스트 인식 시스템의 계산 효율성을 개선하는 기초를 제공합니다. 따라서 인공지능(artificial intelligence), 심층 학습(deep learning), 컴퓨터 비전(computer vision), 패턴 인식(pattern recognition) 등 다양한 컴퓨터 과학 분야에서 연구 및 혁신의 발전을 이끌 잠재력을 가지고 있습니다.



### Improving Radiology Report Conciseness and Structure via Local Large Language Models (https://arxiv.org/abs/2411.05042)
- **What's New**: 본 연구에서는 방사선 보고서의 간결성과 구조적 조직을 개선하여 진단 결과를 더 효과적으로 전달하는 방법을 제시합니다. 특히, 해부학적 영역에 따라 정보를 정리하는 템플릿 방식을 통해 의사들이 관련 정보를 빠르게 찾을 수 있도록 합니다.

- **Technical Details**: 대규모 언어 모델(LLMs)인 Mixtral, Mistral, Llama를 활용하여 간결하고 잘 정리된 보고서를 생성합니다. 특히, Mixtral 모델이 다른 모델에 비해 특정 포맷 요구사항을 잘 준수하므로 주로 이에 초점을 맞추었습니다. LangChain 프레임워크를 활용해 다섯 가지 다양한 프롬프트(Prompting) 전략을 적용하여 방사선 보고서의 일관된 구조를 유지합니다.

- **Performance Highlights**: 새로운 메트릭인 간결성 비율(Conciseness Percentage, CP) 점수를 도입하여 보고서의 간결성을 평가합니다. 814개의 방사선 보고서를 분석한 결과, LLM이 먼저 보고서를 압축한 다음 특정 지침에 따라 내용을 구조화하도록 지시하는 방식이 가장 효과적이라는 것을 발견했습니다. 우리의 연구는 오픈소스이고 로컬에서 배포된 LLM이 방사선 보고서의 간결성과 구조를 상당히 개선할 수 있음을 보여줍니다.



### Bottom-Up and Top-Down Analysis of Values, Agendas, and Observations in Corpora and LLMs (https://arxiv.org/abs/2411.05040)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)이 생성하는 텍스트에서 사회문화적 가치들을 추출하고 평가하기 위한 자동화된 접근 방식을 제시합니다. 특히, 다양한 관점과 가치의 충돌과 공명을 자동으로 분석하여 다원적 가치 정렬(pluralistic value alignment)을 이해하고 관리하는 것을 목표로 합니다.

- **Technical Details**: 연구팀은 상향식(bottom-up) 및 하향식(top-down) 접근 방식을 결합하여 LLM의 출력에서 다수의 가치를 식별하고 측정하는 새로운 방법론을 제안합니다. 상향식 분석은 텍스트로부터 이질적인 가치 주장을 추출하고, 하향식 분석은 기존의 가치 목록을 이용하여 LLM의 출력을 평가하는 방식입니다. 이를 통해 가치의 정렬 및 다원성을 분석하는 네트워크를 구축했습니다.

- **Performance Highlights**: 하향식 가치 분석에서는 F1 점수 0.97의 높은 정확도를 기록하였으며, 상향식 가치 추출은 인간 주석자와 유사한 성능을 보였습니다. 이를 통해 연구팀은 LLM의 출력에서 나타나는 가치의 일관성과 충돌을 효과적으로 평가할 수 있음을 입증하였습니다.



### YouTube Comments Decoded: Leveraging LLMs for Low Resource Language Classification (https://arxiv.org/abs/2411.05039)
Comments:
          Accepted at FIRE 2024 (Track: Sarcasm Identification of Dravidian Languages Tamil & Malayalam (DravidianCodeMix))

- **What's New**: 본 연구는 Tamil-English 및 Malayalam-English 언어 쌍에서 코드 혼합(text blending)된 세팅에서의 풍자(sarcasm) 및 감정(sentiment) 분석을 위한 새로운 gold standard corpus를 소개합니다. 이는 NLP(Natural Language Processing) 분야에 중요한 발전을 제공합니다.

- **Technical Details**: 연구에서는 GPT-3.5 Turbo 모델을 활용하여 코드 혼합된 텍스트의 풍자 및 감정 감지를 수행합니다. 각 텍스트는 감정 극성(sentiment polarity)에 따라 주석이 달리며, 라벨 불균형(class imbalance) 문제를 해결하기 위해 데이터셋을 구성합니다.

- **Performance Highlights**: Tamil 언어에서 매크로 F1 점수 0.61을 기록하며 9위에 올랐고, Malayalam 언어에서는 0.50 점수로 13위를 기록하였습니다. 이는 Tamil의 풍자가 더 많은 특성을 반영하고 있었음을 보여줍니다.



### Towards Interpreting Language Models: A Case Study in Multi-Hop Reasoning (https://arxiv.org/abs/2411.05037)
Comments:
          University of Chicago, Computer Science, Master of Science Dissertation

- **What's New**: 이 논문에서는 다중 추론(multi-hop reasoning) 질문에 대한 모델의 성능을 개선하기 위해 특정 메모리 주입(memory injections) 메커니즘을 제안합니다. 이 방법은 언어 모델의 주의(attention) 헤드에 대한 목표 지향적인 수정을 제공하여 그들의 추론 오류를 식별하고 수정하는 것을 목표로 합니다.

- **Technical Details**: 제안된 방법은 먼저 GPT-2 모델의 단일 및 다중 추론 프롬프트에 대한 층(layer)별 활성화를 분석하여, 추론 과정에서 언어 모델에 관련된 프롬프트 específicos 정보(정보의 종류를 '기억'이라고 명명)를 주입할 수 있는 메커니즘을 제공합니다. 주의 헤드에서 적절히 위치에서 정보를 주입함으로써 다중 추론 작업에서 성공적인 다음 토큰의 확률을 최대 424%까지 높일 수 있음을 보였습니다. 이를 통해 'Attention Lens'라는 도구를 개발하여 주의 헤드의 출력을 사람에게 이해할 수 있는 형식으로 해석합니다.

- **Performance Highlights**: 이 방법을 통해 모델의 다중 추론 성능이 상당히 향상되었습니다. 메모리 주입을 통해 모델의 예측이 개선되었으며, 424%까지 성공적인 다음 토큰의 확률이 향상되었습니다. 이런 성과는 특정 주의 헤드의 작은 부분 집합이 모델의 예측에 상당한 영향을 미칠 수 있음을 나타냅니다.



### Mitigating Privacy Risks in LLM Embeddings from Embedding Inversion (https://arxiv.org/abs/2411.05034)
- **What's New**: 이 논문에서는 embedding inversion 공격에 대한 방어 메커니즘인 Eguard를 소개합니다. 이 방법은 transformer 기반의 projection network와 text mutual information 최적화를 사용하여 embedding의 개인 정보를 보호합니다.

- **Technical Details**: Eguard는 embedding 공간을 안전한 embedding 공간으로 투영하여 민감한 특징을 분리합니다. 이 과정에서 autoencoder를 활용하여 텍스트와 embedding 간의 상호 정보를 계산하고, 다중 작업 최적화 메커니즘을 도입하여 방어와 기능 간의 균형을 유지합니다.

- **Performance Highlights**: Eguard는 embedding inversion 공격으로부터 95% 이상의 토큰을 보호하며, 4가지 하류 작업에서 98% 이상의 원래 embedding과의 일관성을 유지합니다. 또한, Eguard는 embedding perturbations, 보지 않은 학습 시나리오 및 적응 공격에 대한 강력한 방어 성능을 보여줍니다.



### Ultrasound-Based AI for COVID-19 Detection: A Comprehensive Review of Public and Private Lung Ultrasound Datasets and Studies (https://arxiv.org/abs/2411.05029)
- **What's New**: 이 문서는 COVID-19의 진단 및 예측을 위한 AI 기반 연구에서 폐 초음파(ultrasound) 기술의 활용을 다룬 포괄적인 검토입니다. 저자들은 공공 및 민간 LUS 데이터셋을 정리하고, 분석된 연구 결과를 표 형식으로 제시하였습니다. 이 연구는 특히 아동 및 임산부에 대한 진단 임상에 큰 잠재력을 보여줍니다.

- **Technical Details**: 저자들은 총 60개의 관련 논문을 검토하였으며, 그 중 41개는 공개 데이터셋을, 나머지 19개는 개인 데이터셋을 사용했습니다. AI 모델, 데이터 전처리 방법(data preprocessing methods), 교차 검증 기술(cross-validation techniques), 평가 메트릭스(evaluation metrics) 등을 체계적으로 분석하고 정리하였습니다. 또한, AI 기반 COVID-19 검출 및 분석에서 사용되는 선호되는 초음파 전처리 및 증강 기법(augmentation techniques)에 대한 리뷰도 포함되어 있습니다.

- **Performance Highlights**: COVID-19를 검출하기 위한 초음파 기반 AI 연구는 임상 현장에서 큰 잠재력을 보여주며, 특히 아동과 임산부를 위한 진단에 유용하게 활용될 수 있습니다. 이 연구는 인공지능 모델의 성능을 개선하기 위한 다양한 접근 방식과 데이터셋의 활용을 비교할 수 있는 유용한 자원을 제공합니다.



### Leveraging Transfer Learning and Multiple Instance Learning for HER2 Automatic Scoring of H\&E Whole Slide Images (https://arxiv.org/abs/2411.05028)
- **What's New**: 이번 연구에서는 유방암 환자의 HER2 스코어링을 위한 자동화 모델 개발이 딥러닝 전이 학습(transfer learning)과 다중 인스턴스 학습(multiple-instance learning, MIL)의 조합을 통해 어떻게 향상될 수 있는지를 보여주었습니다. 특히, Hematoxylin and Eosin (H&E) 이미지에서 사전 훈련된 모델(embedding models)이 다른 유형의 이미지보다 일관되게 높은 성능을 나타냈습니다.

- **Technical Details**: 연구는 H&E 이미지, 면역 조직 화학(Immunohistochemistry, IHC) 이미지 및 비의료 이미지를 소스 데이터셋으로 설정하여 각각의 분류 작업을 시행했습니다. 또한, MIL 프레임워크와 주의(attention) 메커니즘을 활용하여 패치(patch) 기반의 주목도를 통해 HER2 양성 영역을 시각적으로 강조할 수 있는 방법론을 제안했습니다. AlexNet CNN 구조를 사용하여 각 소스 작업에서 사전 훈련을 실시하였고, 이를 통해 생성된 피쳐 벡터(feature vector)는 주의 레이어(attention layer)로 전달되어 가방(bag) 수준의 피쳐 벡터가 작성되었습니다.

- **Performance Highlights**: 연구 결과, H&E 이미지를 기반으로 사전 훈련된 모델은 평균 AUC-ROC 값이 0.622로 나타났으며, HER2 스코어에 대해 $0.59-0.80$의 성능을 보이는 것으로 확인되었습니다. 이 연구는 세 가지 다양한 소스 타입에서의 사전 훈련이 모델의 성능에 미치는 영향을 정량화하고, K개의 훈련 세트를 통해 각각의 모델 성능 변동을 계산하여 95% 신뢰 구간을 산출했습니다.



### Generative Artificial Intelligence Meets Synthetic Aperture Radar: A Survey (https://arxiv.org/abs/2411.05027)
- **What's New**: 이 논문은 SAR(Synthetic Aperture Radar) 이미지 해석의 생성적 인공지능(Generative AI, GenAI) 기술 활용 가능성을 탐구하며, 이 두 분야 간의 접목을 체계적으로 분석합니다.

- **Technical Details**: 이 연구에서는 SAR 데이터의 수량 및 품질 문제를 해결하기 위해, 최신 GenAI 모델을 사용한 데이터 생성 기반 애플리케이션을 분석하고, SAR 관련 모델의 기본 구조와 변형을 검토합니다. 특히, GenAI와 해석 가능한 모델을 결합한 하이브리드 모델링 방법을 제안합니다.

- **Performance Highlights**: SAR 이미지 해석에 있어 많은 제한이 있지만, GenAI 기술은 고품질 및 다양한 SAR 데이터 생성 가능성을 열어주며, multi-view SAR 이미지 생성, optical-to-SAR 번역, SAR 이미지 조합과 같은 특정 응용 사례를 중점적으로 다룹니다.



### LLMs as Research Tools: A Large Scale Survey of Researchers' Usage and Perceptions (https://arxiv.org/abs/2411.05025)
Comments:
          30 pages, 5 figures

- **What's New**: 본 연구는 816명의 승인된 연구 논문 저자들을 대상으로 한 대규모 조사를 통해 연구 커뮤니티가 대형 언어 모델(LLMs)을 연구 도구로서 어떻게 활용하는지와 이를 어떻게 인식하는지를 분석했습니다.

- **Technical Details**: 응답자들은 LLMs를 정보 탐색, 편집, 아이디어 구상, 직접 작성, 데이터 정리 및 분석, 데이터 생성 등의 다양한 작업에 활용하고 있다고 보고했습니다. 전반적으로 81%의 연구자들이 자신의 연구 작업에 LLMs를 사용하고 있으며, 특히 비백인, 비영어 원어민, 주니어 연구자들이 더 많이 사용하고 높은 이점을 보고했습니다.

- **Performance Highlights**: LLMs를 사용하고 있는 연구자들은 정보 탐색과 편집의 작업에서 가장 빈번하게 사용하고 있으며, 데이터 분석 및 생성에서 상대적으로 적게 사용하고 있습니다. 여성과 비바이너리 연구자들은 더 큰 윤리적 우려를 표명했으며, 전체적으로 연구자들은 영리 기관보다 비영리 또는 오픈소스 모델을 더 선호했습니다.



### Enhancing literature review with LLM and NLP methods. Algorithmic trading cas (https://arxiv.org/abs/2411.05013)
- **What's New**: 이 연구는 알고리즘 거래 분야의 지식을 분석하고 정리하기 위해 machine learning 알고리즘을 활용했습니다. 136 백만 개의 연구 논문 데이터셋을 필터링하여 1956년부터 2020년 1분기까지 발표된 14,342개의 관련 논문을 식별했습니다. 이들은 전통적인 키워드 기반 알고리즘과 최신 topic modeling 방법을 비교하여 알고리즘 거래의 다양한 접근 방식과 주제의 인기 및 진화를 평가했습니다.

- **Technical Details**: 연구는 자연어 처리(Natural Language Processing, NLP)를 통해 지식을 자동으로 추출할 수 있는 유용성을 입증하고, ChatGPT와 같은 최신 대형 언어 모델(Large Language Models, LLMs)의 새로운 가능성을 강조합니다. 연구는 알고리즘 거래에 대한 연구 기사가 전반적인 출판 수 증가 속도보다 빠르게 증가하고 있음을 발견했습니다. 머신 러닝 모델은 최근 몇 년간 가장 인기 있는 방법으로 자리 잡았습니다.

- **Performance Highlights**: 본 연구는 복잡한 질문을 해결하기 위해 작업을 더 작은 구성 요소로 분해하고 추론 단계를 포함함으로써 알고리즘 거래 방법론에 대한 깊은 이해를 도왔습니다. LLMs를 이용한 자동 문서 리뷰 및 데이터셋 개선의 효율성을 보여주며, 시간에 따른 자산 클래스, 시간 지평선 및 모델의 인기 변화를 상세히 분석했습니다.



### Scattered Forest Search: Smarter Code Space Exploration with LLMs (https://arxiv.org/abs/2411.05010)
- **What's New**: 본 연구는 코드 생성(cod generation)을 위한 LLM(대형 언어 모델) 추론(inference) 확장을 위한 새로운 접근 방식을 제안합니다. 코드 생성을 블랙박스 최적화 문제로 설정하고, 해결책의 다양성을 높이기 위해 Scattered Forest Search(SFS)라는 최적화 기법을 도입합니다.

- **Technical Details**: SFS는 입력 프롬프트를 동적으로 변경하여 다양한 출력(output)을 생성하는 Scattering 기법과, 초기화된 다양한 랜덤 시드를 사용하여 탐색 범위를 넓히는 Foresting 기법을 포함합니다. 또한 Ant Colony Optimization과 Particle Swarm Optimization에서 영감을 받아 Scouting 기법이 추가되어, 검색 단계에서 긍정적이거나 부정적인 결과를 공유함으로써 탐색(exploration)과 활용(exploitation)을 개선합니다.

- **Performance Highlights**: HumanEval+에서 67.1%, HumanEval에서는 87.2%의 pass@1 비율을 기록하며, 기존 최첨단 기술 대비 각각 8.6%와 4.3%의 성능 향상을 달성했습니다. 또한, 코드 컨테스트, Leetcode 등의 다양한 벤치마크에서 기존 방법에 비해 더 빠른 정확한 솔루션을 발견할 수 있음을 보여주었습니다.



New uploads on arXiv(cs.LG)

### Tract-RLFormer: A Tract-Specific RL policy based Decoder-only Transformer Network (https://arxiv.org/abs/2411.05757)
- **What's New**: 새로운 논문에서는 Tract-RLFormer라는 네트워크를 제안하여 지도 학습(supervised learning)과 강화 학습(reinforcement learning)을 결합하여 섬유 측정의 정확도와 일반화 능력을 향상시키고 있습니다. 이 네트워크는 기존의 전통적인 분할(segmentation) 프로세스를 생략하며, 특정 트랙에 대해 직접적으로 형상을 생성합니다.

- **Technical Details**: Tract-RLFormer는 두 단계 정책(refinement) 개선 과정을 통해 네트워크가 흰 물질(white matter) 경로를 보다 정확하게 매핑할 수 있도록 합니다. 이를 위해 Spherical Harmonics Coefficients(SHC), Fiber Orientation Distribution Functions(fODF) 및 섬유 직선(Fiber peaks) 등의 전처리된 데이터를 이용하여 정책을 학습합니다. 모델은 TD3(Twin-Delayed Deep-Deterministic Policy Gradient) 및 SAC(Soft Actor-Critic) 알고리즘을 사용하여 트랙을 생성하는 정책을 학습합니다.

- **Performance Highlights**: Tract-RLFormer는 TractoInferno, HCP, ISMRM2015와 같은 다양한 데이터 세트에서 엄격한 검증을 통해 기존의 방법보다 더 나은 성능을 보였으며, 흰 물질 경로에 대한 정확한 매핑 능력을 입증했습니다.



### FisherMask: Enhancing Neural Network Labeling Efficiency in Image Classification Using Fisher Information (https://arxiv.org/abs/2411.05752)
- **What's New**: 이 논문에서는 Fisher 정보에 기반한 액티브 러닝 방법인 FisherMask를 제안합니다. FisherMask는 네트워크의 필수 파라미터를 식별하기 위해 Fisher 정보를 활용하여 효과적으로 중요한 샘플을 선택합니다. 이 방법은 방대한 레이블된 데이터에 대한 의존성을 줄이면서도 모델 성능을 유지할 수 있는 전략을 제공합니다.

- **Technical Details**: FisherMask는 Fisher 정보를 이용하여 고유한 네트워크 마스크를 구축하고, 이 마스크는 가장 높은 Fisher 정보 값을 가진 k개의 가중치를 선택하여 형성됩니다. 또한, 성능 평가는 CIFAR-10 및 FashionMNIST와 같은 다양한 데이터셋에서 수행되었으며, 특히 불균형 데이터셋에서 유의미한 성능 향상이 나타났습니다.

- **Performance Highlights**: FisherMask는 기존 최첨단 방법들보다 뛰어난 성능을 보이며, 레이블링 효율성을 크게 향상시킵니다. 이 방법은 모든 데이터셋에서 테스트 되었으며, 액티브 러닝 파이프라인에서 모델의 성능 특성을 더 잘 이해할 수 있는 유용한 통찰력을 제공합니다.



### Continuous-Time Analysis of Adaptive Optimization and Normalization (https://arxiv.org/abs/2411.05746)
- **What's New**: 이번 연구에서는 현대 딥러닝에서 중요한 구성 요소인 Adam 및 AdamW 최적화 알고리즘의 지속적인 시간(formulation) 모델을 제시하고, 이 모델을 통해 이러한 최적화 알고리즘의 훈련 동역학(training dynamics)을 분석합니다. 특히, 하이퍼파라미터(hyperparameters)의 최적 선택과 구조적 결정(architectural decisions)에 대한 보다 깊은 이해를 제공합니다.

- **Technical Details**: Adam 및 AdamW 최적화 알고리즘의 지속적인 시간 공식화는 쌍곡선 차분 방정식으로 표현되며, 이를 통해 Adam의 하이퍼파라미터(eta, \\gamma)의 안정적 지역을 이론적으로 도출합니다. 또한, scale-invariant 아키텍처의 메타 적응 효과(meta-adaptive effect)를 밝혀내어 최적화기 $k$-Adam을 일반화하는 과정을 설명합니다.

- **Performance Highlights**: 실험적으로, 제안된 하이퍼파라미터 지역 외부에서 파라미터 업데이트의 불안정한 지수 성장(exponential growth)을 확인함으로써 이론 예측을 실증적으로 검증하였습니다. $2$-Adam optimizer는 정상화(normalization) 절차를 k회 적용하여 Adam과 AdamW의 성능을 통합하는 새로운 접근 방식을 제시합니다.



### Free Record-Level Privacy Risk Evaluation Through Artifact-Based Methods (https://arxiv.org/abs/2411.05743)
- **What's New**: 회원 추론 공격 (Membership Inference Attacks, MIA)은 머신 러닝 모델의 프라이버시 위험성을 평가하는 데에 널리 사용되는 기법입니다. 본 논문은 기존의 고비용 해법 대신, 훈련 중 이용 가능한 아티팩트(artifacts)를 활용하여 위험한 샘플을 식별하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 본 연구에서 제안하는 Loss Trace IQR (LT-IQR) 메서드는 훈련 중 각 샘플의 손실 추적을 분석하여 취약한 데이터 샘플을 식별합니다. 우리 방법은 CIFAR10 데이터셋에서의 실험을 통해 검증되었고, LiRA와 같은 고급 MIA와 동등한 정밀도를 달성했습니다. LT-IQR은 샘플 레벨 손실의 집계 점수를 간단하지만 견고하게 계산합니다.

- **Performance Highlights**: LT-IQR 메서드는 10^(-3)에서 정의된 취약한 샘플에 대해 61% Precision@k=1%를 달성했습니다. 이는 LT-IQR이 고성능의 그림자 모델 기반 MIA와 유사한 성능을 가지면서도, 극히 저렴한 계산 비용으로 구현 가능하다는 것을 보여줍니다. 또한, 다른 손실 집계 방법들과 비교하여 LT-IQR이 우수함을 입증하였습니다.



### Topology-aware Reinforcement Feature Space Reconstruction for Graph Data (https://arxiv.org/abs/2411.05742)
- **What's New**: 이번 연구에서는 그래프 데이터의 특성 공간(feature space) 재구성을 자동화하고 최적화하는 새로운 접근법을 제시합니다. 기존의 수작업적인 피처 변환 및 선택 기법과는 달리, 그래프 데이터의 고유한 위상 구조(topological structure)를 고려하여 혁신적인 해법을 제공합니다.

- **Technical Details**: 이 논문에서는 topology-aware reinforcement learning을 활용하여 그래프 데이터의 피처 공간을 재구성하는 방식을 도입합니다. 주된 구성 요소로는 핵심 서브그래프(core subgraphs) 추출, 그래프 신경망(graph neural network; GNN) 사용, 그리고 세 가지 계층적 강화 에이전트가 포함됩니다. 이 접근법은 피처 생성의 반복적인 과정을 통해 의미 있는 피처를 효과적으로 생성합니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안된 방법의 효과성과 효율성이 입증되었습니다. 특히, 위상 정보를 포함함으로써 피처 공간의 재구성을 최적화할 수 있으며, 이는 내려가는 머신러닝 작업(downstream ML tasks)의 성능 향상으로 이어집니다.



### Aioli: A Unified Optimization Framework for Language Model Data Mixing (https://arxiv.org/abs/2411.05735)
- **What's New**: 이 논문에서는 언어 모델의 성능이 데이터 그룹의 최적 혼합 비율을 파악하는 데 의존한다는 점을 강조하며, 다양한 방법의 효율성을 비교했습니다. 특히, 기존 방법들이 단순한 stratified sampling 기준선보다 높은 평균 테스트 perplexity 성능을 보이지 못한 점을 발견했습니다.

- **Technical Details**: 저자들은 Linear Mixing Optimization (LMO)이라는 통합 최적화 프레임워크를 제안하였으며, 이를 통해 다양한 혼합 방법들의 기본 가정을 분석했습니다. 혼합 법칙(mixing law)의 매개변수 조정에서의 오류가 기존 방법들의 성능 불일치를 초래한다는 것을 입증했습니다.

- **Performance Highlights**: 새롭게 제안된 온라인 데이터 혼합 방법 Aioli는 6개의 데이터셋에서 모든 경우에 걸쳐 본 논문의 기존 방법보다 평균 0.28 포인트의 테스트 perplexity 향상을 보였습니다. 또한, Aioli는 계산자원이 제한된 상황에서도 혼합 비율을 동적으로 조정하여 기존 방법들보다 최대 12.01 포인트 이상 개선된 성능을 보여줍니다.



### Differential Privacy Under Class Imbalance: Methods and Empirical Insights (https://arxiv.org/abs/2411.05733)
Comments:
          14 pages

- **What's New**: 본 연구에서는 데이터의 클래스 비율이 불균형한 상황에서 차등 개인정보 보호(차등 개인정보 보호, Differential Privacy) 기술을 적용하는 알고리즘적 솔루션을 제시합니다. 기존의 불균형 학습 기법을 차등 개인 정보 보호 프레임워크로 확장할 때 발생하는 도전과제와 이를 해결하기 위한 다양한 접근 방식이 다루어집니다.

- **Technical Details**: 연구는 불균형 데이터 문제 해결을 위한 두 가지 주요 접근법을 제시합니다. 첫 번째는 전처리(pre-processing) 기법으로, 원본 데이터셋을 비공식적으로 증대시키거나 불균형을 줄이는 방법이며 SMOTE(부샘플링, Synthetic Minority Over-sampling Technique)와 같은 기법이 포함됩니다. 두 번째 접근법은 학습 알고리즘을 조정하는 인처리(in-processing) 기법으로, 모델 백개팅(model bagging)이나 클래스 가중치(class-weighted) 손실 최소화 기법이 이에 해당합니다.

- **Performance Highlights**: 실험 결과, 프라이빗 가중치 경험적 위험 최소화(private weighted ERM) 모델이 비공식 모델에 비해 불균형 분류를 위한 성능이 평균적으로 개선되었으나, 딥 러닝 모델은 소규모 및 중간 규모 개인 정보 보호 불균형 분류 작업에서 불리한 결과를 보였습니다.



### Graph-Dictionary Signal Model for Sparse Representations of Multivariate Data (https://arxiv.org/abs/2411.05729)
- **What's New**: 이번 연구에서는 변수들 간의 복잡한 관계를 포착하기 위한 새로운 Graph-Dictionary 신호 모델을 제안합니다. 이 모델은 데이터 분포의 관계를 표현하기 위해 유한한 그래프 집합을 사용하여 Laplacian의 가중치 합으로 특징화합니다.

- **Technical Details**: 제안된 프레임워크는 관측된 데이터로부터 그래프 사전 표현을 추론하는 과정을 포함하며, 학습 문제 해결을 위한 primal-dual splitting 알고리즘의 이진 선형 일반화를 사용합니다. 이 새로운 수식화는 신호 속성뿐 아니라 기저 그래프 및 그 계수에 대한 사전 지식을 포함할 수 있습니다.

- **Performance Highlights**: 이 방법은 여러 합성 설정에서 신호로부터 그래프를 재구성하는 능력을 보여주며, 이전 기준보다 우수한 성능을 보입니다. 또한, 뇌 활동 데이터의 motor imagery decoding 작업에서 그래프-사전 표현을 활용해 상상한 운동을 분류하는 정확도가 기존의 여러 특징에 의존하는 표준 방법보다 더 뛰어난 결과를 낳았습니다.



### Scaling Laws for Task-Optimized Models of the Primate Visual Ventral Stream (https://arxiv.org/abs/2411.05712)
Comments:
          9 pages for the main paper, 20 pages in total. 6 main figures and 10 supplementary figures. Code, model weights, and benchmark results can be accessed at this https URL

- **What's New**: 이번 연구는 인공지능 신경망 모델의 확장(scale)이 동물의 시각 처리에서 핵심 객체 인식(코어 오브젝트 인식: CORE)과 신경 반응 패턴과 어떻게 관련되는지를 분석합니다. 600개 이상의 모델을 체계적으로 평가하여 모델 아키텍처와 데이터 세트의 크기에 따른 뇌 정렬(brain alignment)의 영향을 조사했습니다.

- **Technical Details**: 저자들은 다양한 아키텍처의 모델을 훈련하고, V1, V2, V4, IT 및 CORE 행동과 같은 벤치마크에서 모델의 성능을 평가하여 스케일링 법칙(scaling laws)을 도출하였습니다. 결과적으로, 행동 정렬(behavioral alignment)은 계속 개선되는 반면, 뇌 정렬은 포화 상태에 이른다는 중요한 발견이 있었습니다.

- **Performance Highlights**: 작은 모델들은 샘플 수가 적을 때 낮은 정렬 성능을 보였지만, 고차원 시각 영역에서는 더 큰 모델이 뇌 정렬 성능에 있어서 이점을 지닙니다. 저자들은 더 나은 성능을 위해 데이터 샘플에 더 많은 계산 리소스를 할당하는 것이 필요하다고 제안합니다.



### Sample and Computationally Efficient Robust Learning of Gaussian Single-Index Models (https://arxiv.org/abs/2411.05708)
- **What's New**: 이번 연구는 Gaussian 분포 하에서 L^2_2 손실을 기준으로 한 단일 색인 모델(Single-Index Model, SIM)의 학습을 다루고 있습니다. 특히, 적대적 레이블 노이즈(agnostic model)에 따른 효율적 학습 알고리즘을 제안합니다.

- **Technical Details**: 제안된 알고리즘은 O(OPT)+ε의 L^2_2 오류를 달성하며, 알고리즘의 샘플 복잡도는 \tilde{O}(d^{\lceil k^{\ast}/2 \rceil}+d/\epsilon)입니다. 이 때, k^{\ast}는 고유한 정보 지수를 나타내며, 이전에 알려진 하한과 거의 일치하는 샘플 경계로 설정됩니다.

- **Performance Highlights**: 제안된 알고리즘은 기존의 방법들보다 강력한 가정을 필요로 하지 않으며, 단순히 비모노톤 및 리프시츠(link function) 링크 함수에 대해서도 성능을 보장합니다. 이는 적대적 조건에서도 안정적인 학습 결과를 나타냅니다.



### YOSO: You-Only-Sample-Once via Compressed Sensing for Graph Neural Network Training (https://arxiv.org/abs/2411.05693)
- **What's New**: 본 논문에서는 YOSO (You-Only-Sample-Once)라는 새로운 샘플링 알고리즘을 제안하여 GNN (Graph Neural Networks) 훈련 시간을 크게 단축시키면서도 예측 정확도를 유지하도록 설계되었습니다.

- **Technical Details**: YOSO는 압축 샘플링 (Compressed Sensing, CS) 기반의 샘플링 및 재구성 프레임워크를 도입합니다. 입력 계층에서 노드를 한 번 샘플링하고, 각 에포크마다 출력 계층에서 손실 없는 재구성을 진행합니다. 이 과정에서 YOSO는 전통적인 압축 샘플링 방법에서 발생하는 비싼 계산을 피하며, 모든 노드가 참여하는 것과 동등한 높은 확률의 정확성을 보장합니다.

- **Performance Highlights**: 실험 결과에 따르면 YOSO는 노드 분류 및 링크 예측 작업에서 평균 75%의 훈련 시간을 단축하면서도 상위 성능 기준선과 동등한 정확도를 유지하는 것으로 나타났습니다. 이는 YOSO가 GNN의 전체 훈련 과정을 크게 효율화했음을 보여줍니다.



### Improving Molecular Graph Generation with Flow Matching and Optimal Transpor (https://arxiv.org/abs/2411.05676)
- **What's New**: GGFlow는 분자 그래프 생성을 위한 첫 번째 이산 흐름 매칭 생성 모델로, 최적 운송(optimal transport)을 활용하여 샘플링 효율성과 훈련 안정성을 향상시킵니다. 이 모델은 화학 결합 간의 직접적인 통신을 가능하게 하는 가장자리 증강 그래프 변환기(edge-augmented graph transformer)를 통합하여 생성 작업을 개선합니다.

- **Technical Details**: GGFlow는 이산 흐름 매칭 기법과 최적 운송을 활용하여 분자 그래프 생성을 위한 샘플링 효율성과 훈련 안정성을 개선하는 모델입니다. 이 모델은 그래프의 희소성(sparsity)과 순열 불변성(permutation invariance)을 유지하면서 화학 결합(info flash) 생성을 위한 트랜스포머를 포함합니다. 또한, 목표 속성을 가진 분자를 디자인하기 위해 강화 학습(reinforcement learning)을 이용한 목표 지향(guided) 세대 프레임워크가 도입되었습니다.

- **Performance Highlights**: GGFlow는 무조건 조건 및 조건부 분자 생성 작업에서 최첨단 성능을 보여주며 기존 방법들을 일관되게 초월합니다. 이 모델은 적은 추론 단계(few inference steps)에서도 뛰어난 성과를 발휘하며, 다양한 그래프 유형과 복잡성에 대해 개선된 결과를 보여줍니다.



### Enhancing Model Fairness and Accuracy with Similarity Networks: A Methodological Approach (https://arxiv.org/abs/2411.05648)
Comments:
          7 pages, 4 figures

- **What's New**: 본 논문에서는 데이터셋 특성으로 인해 기계 학습 작업에서 발생하는 편향을 탐색하기 위한 혁신적인 접근 방식을 제안합니다. 데이터 형식에 따라 인스턴스를 유사성 특징 공간(Similarity Feature Space)으로 매핑하는 다양한 기술을 사용할 수 있습니다. 이 방법은 쌍(pairwise) 유사성의 해상도를 조정하여 데이터셋 분류의 복잡성(classification complexity)과 모델 공정성(fairness) 간의 관계에 대한 명확한 통찰을 제공합니다.

- **Technical Details**: 본 연구에서는 Gower Distance(𝐆𝐃), Natural Language Processing(NLP), word2vec(단어 임베딩 기법) 등 다양한 거리 측정 기법과 데이터 처리 기법을 사용하여 유사성 네트워크(Similarity Network)를 구축합니다. 이를 위해 Scaled Exponential Kernel 및 Random Walk Kernel을 사용하여 유사성 네트워크의 링크 가중치를 조정함으로써 정확성과 공정성 간의 효과적인 균형을 달성합니다.

- **Performance Highlights**: 실험 결과, 제안한 유사성 네트워크 방법론은 데이터셋에 따른 편향을 줄이고 공정성을 보장하는 기계 학습 모델의 적용 가능성을 입증했습니다. 분류(classification), 데이터 임퓨테이션(data imputation), 데이터 증강(data augmentation) 등 다양한 하위 작업에서 뛰어난 성능을 보여 주었습니다.



### WHALE: Towards Generalizable and Scalable World Models for Embodied Decision-making (https://arxiv.org/abs/2411.05619)
- **What's New**: WHALE(행동 조건화 및 재추적 롤아웃 학습이 결합된 세계 모델)는 결정-making을 지원하기 위해 일반화와 불확실성 추정에서의 두 가지 주요 도전을 해결하는 프레임워크입니다.

- **Technical Details**: WHALE 프레임워크는 행동-conditioning 및 retracing-rollout이라는 두 가지 주요 기술로 구성되어 있으며, 이는 모든 신경망 아키텍처와 결합이 가능합니다. Whale-ST(공간-시간 트랜스포머 기반 세계 모델)는 향상된 일반화를 제공하고, Whale-X는 Open X-Embodiment 데이터셋에서 970K 경로를 기반으로 414M 파라미터를 가지는 세계 모델입니다.

- **Performance Highlights**: 실험 결과, Whale-ST는 가치 추정 정확도와 비디오 생성 충실도 모두에서 기존 세계 모델 학습 방법보다 우수한 성능을 보였으며, Whale-X는 실제 조작 시나리오에서 뛰어난 확장성과 강력한 일반화를 보여주었습니다.



### Knowledge Distillation Neural Network for Predicting Car-following Behaviour of Human-driven and Autonomous Vehicles (https://arxiv.org/abs/2411.05618)
Comments:
          27th IEEE International Conference on Intelligent Transportation Systems

- **What's New**: 본 연구는 자율주행차(Autonomous vehicles, AV)와 인간이 운전하는 차량(Human-driven vehicles, HDV)의 혼합 교통 시나리오에서 차량 추적 행동(car-following behaviour)을 분석하여 교통 효율성 및 도로 안전성을 개선하는 데 중점을 둡니다. 기존 모델들과 비교하여 Knowledge Distillation Neural Network (KDNN) 모델을 도입하여 차량 추적 행동의 속도를 예측합니다.

- **Technical Details**: 이 연구는 Waymo의 차량 궤적 데이터셋을 사용하여 HDV-AV, AV-HDV, HDV-HDV의 세 가지 차량 쌍에 대한 차량 추적 행동을 분석합니다. KDNN 모델은 복잡한 신경망 모델의 지식을 경량화된 모델로 증류하여 기존의 LSTM 및 MLP와의 성능 비교를 통해 높은 예측 정확도를 유지하면서도 낮은 연산 요구 사항을 충족합니다.

- **Performance Highlights**: KDNN 모델은 최소 시간-충돌(Time-to-Collision, TTC) 측정에서 충돌 방지 성능이 우수하며, LSTM 및 MLP와 비교했을 때 예측 정확도가 유사하거나 더 뛰어난 결과를 보이고 있습니다. 이 모델은 자율주행차 및 드라이빙 시뮬레이터와 같은 자원 제약 환경에서 효율적인 연산이 가능합니다.



### Acceleration for Deep Reinforcement Learning using Parallel and Distributed Computing: A Survey (https://arxiv.org/abs/2411.05614)
Comments:
          This paper has been accepted by ACM Computing Surveys

- **What's New**: 이 논문은 최근 몇 년간 인공지능 분야에서 중요한 발전을 가져온 Deep Reinforcement Learning (DRL)의 훈련을 가속화하기 위한 방법론을 포괄적으로 조사한 내용을 담고 있습니다. 특히, 병렬 및 분산 컴퓨팅(parallel and distributed computing)을 기반으로 한 DRL 훈련 가속화 방법에 대한 통계적 분류와 신기술 동향을 다룹니다.

- **Technical Details**: DRL은 인공지능(AI) 연구자들에 의해 널리 탐구되고 있는 강력한 머신러닝(parallel and distributed computing) 패러다임입니다. 이 논문에서는 시스템 아키텍처, 시뮬레이션 병렬성(simulation parallelism), 계산 병렬성(computing parallelism), 분산 동기화 메커니즘(distributed synchronization mechanisms), 심층 진화 강화 학습(deep evolutionary reinforcement learning) 등의 다양한 기술적 세부정보를 분석합니다.

- **Performance Highlights**: 논문에서는 현재의 16개 오픈 소스 라이브러리(open-source libraries) 및 플랫폼을 비교하고, 이러한 시스템들이 병렬 및 분산 DRL을 구현하는 데 어떻게 기여하는지를 분석합니다. 연구자들이 제안한 기술들을 실제로 어떻게 적용할 수 있는지를 조명하며, 향후 연구 방향에 대한 통찰을 제공합니다.



### Machine learning-driven Anomaly Detection and Forecasting for Euclid Space Telescope Operations (https://arxiv.org/abs/2411.05596)
Comments:
          Presented at IAC 2024

- **What's New**: 본 연구는 Euclid 우주 망원경의Telemetry(텔레메트리)와 과학 데이터에서의 이상 탐지(Anomaly Detection) 문제를 해결하기 위해 머신러닝을 활용한 새로운 접근법을 제시합니다.

- **Technical Details**: 연구에서는 2024년 2월부터 8월까지 Euclid의 텔레메트리에서 11개의 온도 파라미터와 35개의 공변량(Covariates)을 분석합니다. 과거 값을 기반으로 온도를 예측하기 위해 예측 XGBoost 모델을 사용하며, 예측으로부터의 이탈을 이상치로 감지합니다. 또한, 두 번째 XGBoost 모델은 공변량으로부터의 이상치를 예측하여 온도 이상치와의 관계를 포착합니다. 마지막으로 SHAP(Shapley Additive Explanations)를 이용하여 공변량과의 상호작용을 분석합니다.

- **Performance Highlights**: 이 접근법을 통해 복잡한 파라미터 관계를 신속하게 자동 분석할 수 있으며, 머신러닝을 통해 텔레메트리 모니터링을 개선할 수 있는 가능성을 보여줍니다. 이는 유사 데이터 문제를 가진 다른 우주 미션에도 확장 가능한 솔루션을 제공합니다.



### Towards Active Flow Control Strategies Through Deep Reinforcement Learning (https://arxiv.org/abs/2411.05536)
Comments:
          ECOMMAS 2024 conference proceeding paper

- **What's New**: 이 논문은 항공역학에서의 항력(drag)을 감소시키기 위해 능동 유동 제어(active flow control, AFC)용으로 심층 강화 학습(deep reinforcement learning, DRL) 프레임워크를 제시합니다. 3D 실린더에서 Re=100 조건으로 테스트하여 9.32%의 항력 감소와 78.4%의 양력 변동 감소를 달성했습니다.

- **Technical Details**: DRL 프레임워크는 Computational Fluid Dynamics (CFD) 해석기를 DRL 모델과 통합하여 효율적인 데이터 통신을 위한 인메모리 데이터베이스(redis database)를 사용합니다. 이를 통해 DRL 학습을 위한 빠른 경험 데이터 수집이 가능합니다. DRL에서 두 개의 주요 개체가 있으며, 환경은 CFD 시뮬레이션이고, 에이전트는 상태에 따라 가능한 행동의 확률 분포를 예측하는 신경망(neural network)입니다.

- **Performance Highlights**: 이 연구에서 제안된 AFC-DRL 프레임워크는 복잡한 유체 역학 문제에 적용될 가능성을 보여주며, 이전 연구에 비해 보다 현실적인 산업 시나리오에 적용할 수 있는 중요한 발전을 나타냅니다.



### Do Histopathological Foundation Models Eliminate Batch Effects? A Comparative Study (https://arxiv.org/abs/2411.05489)
Comments:
          Accepted to AIM-FM Workshop @ NeurIPS'24

- **What's New**: 이 연구는 최신 histopathological foundation models의 batch effects(배치 효과)를 체계적으로 검토했습니다. 기존 연구에서는 이러한 모델이 데이터 편향에 면역적일 것이라는 가설이 제기되었으나, 본 연구에서는 여전히 병원 고유의 시그니처가 포함되어 있어 모델의 편향된 예측을 초래할 수 있음을 입증했습니다.

- **Technical Details**: 연구에서는 여러 개의 histopathology 데이터셋(TCGA-LUSC-5, CAMELYON16)을 활용하여 foundation models에서 추출한 feature embeddings(특징 임베딩)을 분석했습니다. 또한, 일반적으로 사용되는 stain normalization(염색 정규화) 방법이 배치 효과를 충분히 감소시키지 않음을 확인했습니다. 모델의 성능에 따라 더욱 높은 출처 예측 정확도를 보였습니다.

- **Performance Highlights**: 기존 모델들이 여전히 병원별 시그니처의 영향을 받으며, 이는 다운스트림 예측 작업에 편향을 초래하는 것으로 나타났습니다. 따라서, 이 연구는 medical foundation models의 평가에 대한 새로운 관점을 제시하며, 더 강력한 사전 훈련 전략과 다운스트림 예측기 개발을 위한 길을 열었습니다.



### The Limits of Differential Privacy in Online Learning (https://arxiv.org/abs/2411.05483)
- **What's New**: 이번 연구에서는 differential privacy (DP)의 fundamental limits와 online learning 알고리즘에서 DP의 적용을 살펴보았습니다. 연구의 핵심은 no DP, pure DP, approximate DP의 세 가지 제약 조건을 구분하는 것입니다.

- **Technical Details**: 연구에서는 approximately differentially private 상태에서 online learning이 가능한 반면, pure DP에서는 adaptive adversaries에 대해 온라인 학습이 불가능하다는 것을 보여주었습니다. 또한, 모든 private 온라인 학습자는 거의 모든 hypothesis class에 대해 무한한 실수를 범해야 한다는 것을 입증하였습니다. 이는 이전 결과를 일반화한 것입니다.

- **Performance Highlights**: 연구 결과, 모든 private online learning 알고리즘은 약 Ω(log T)의 실수를 범해야 하며, 특정한 hypothesis class의 경우, DP의 여러 조건하에서도 finiteness가 성립하지 않을 수 있음을 보여줍니다. 이는 DP 환경에서 private learning이 non-private learning에 비해 어렵다는 것을 강조합니다.



### Bridging the Gap between Learning and Inference for Diffusion-Based Molecule Generation (https://arxiv.org/abs/2411.05472)
Comments:
          14 pages, 5 figures

- **What's New**: Diffusion 모델을 활용하여 분자의 생성 과정에서의 exposure bias 문제를 해결하기 위해 GapDiff라는 새로운 훈련 프레임워크를 제안했습니다. 이 프레임워크는 모델 예측으로 생성된 상태를 확률적으로 정답으로 사용하여 데이터 분포의 차이를 줄이고 생성된 분자의 친화도를 향상시키는 것을 목표로 하고 있습니다.

- **Technical Details**: GapDiff는 모델 예측에 기반한 pseudo molecule 추정 방법을 도입하고, 훈련 과정에서 모델 예측 노이즈를 포함하는 적응 샘플링 전략을 구현합니다. 실험은 CrossDocked2020 데이터셋에서 수행되었으며, 3D 분자 구조와 단백질 결합 친화도에서 향상된 성능을 보여주었습니다.

- **Performance Highlights**: GapDiff를 사용하여 생성된 분자는 기존 기법에 비해 우수한 3D 구조 표현과 단백질 결합 친화도를 나타내며, 새로운 SOTA docking score를 달성했습니다.



### Generalization, Expressivity, and Universality of Graph Neural Networks on Attributed Graphs (https://arxiv.org/abs/2411.05464)
- **What's New**: 본 연구에서는 속성이 있는 그래프(attributed graphs)에서 그래프 신경망(GNNs)의 보편성과 일반화를 분석합니다. 새로운 pseudometric을 제안하여 GNNs의 세밀한 표현력을 설명하고, 보편 근사 이론(universal approximation theorem)과 그래프에 대한 GNN의 일반화 경계를 제시합니다.

- **Technical Details**: 제안하는 pseudometric은 computation trees의 계층적 최적 운송(hierarchical optimal transport)을 통해 속성이 있는 그래프의 구조 유사성을 계산합니다. 이를 통해 MPNNs(Message Passing Neural Networks)는 점을 분리할 수 있으며, Lipschitz 연속성을 유지합니다. 이 연구는 GNNs의 일반화 및 보편 근사에 대한 연속적인 함수에 대한 이론적 기초를 제공합니다.

- **Performance Highlights**: MPNNs는 속성이 있는 그래프의 스페이스에서 어떤 함수도 근사할 수 있으며, 데이터 분포에 대해서 가정할 필요 없이 일반화 경계를 제공합니다. 새로운 pseudometric은 MPNN의 출력 변화(output perturbations)와 상관관계를 보이며, 이를 통해 안정성을 판단할 수 있습니다.



### WeatherGFM: Learning A Weather Generalist Foundation Model via In-context Learning (https://arxiv.org/abs/2411.05420)
- **What's New**: 본 논문에서는 WeatherGFM이라는 첫 번째의 일반화된 날씨 기반 모델을 소개합니다. 이 모델은 단일 모델 내에서 다양한 날씨 이해 작업을 통합하여 처리 가능하도록 설계되었습니다.

- **Technical Details**: WeatherGFM은 날씨 이해 작업에 대한 단일화된 표현 및 정의를 처음으로 통합하고, 단일, 다중, 시간 모달리티를 관리하기 위한 날씨 프롬프트 포맷을 고안합니다. 또한, 통합된 날씨 이해 작업의 교육을 위해 시각적 프롬프트 기반의 질문-응답 패러다임을 채택하였습니다.

- **Performance Highlights**: 광범위한 실험 결과, WeatherGFM은 날씨 예보, 초해상도, 날씨 이미지 변환, 후처리 등 최대 10개의 날씨 이해 작업을 효과적으로 처리할 수 있으며, 보지 못한 작업에 대한 일반화 능력을 보여줍니다.



### Post-Hoc Robustness Enhancement in Graph Neural Networks with Conditional Random Fields (https://arxiv.org/abs/2411.05399)
- **What's New**: 본 연구는 GNN의 추론 단계에서 강건성을 높이기 위한 새로운 접근법인 RobustCRF를 제안합니다. 이는 Conditional Random Field를 사용하여 통계적 관계 학습에 기반한 포스트-호크(post-hoc) 방법론입니다.

- **Technical Details**: RobustCRF는 모델-불가지론적(model-agnostic)이며, 기본 모델 아키텍처에 대한 사전 지식이 필요하지 않습니다. 이 방법은 이웃 포인트 간의 유사한 예측을 보존하는 것에 중점을 두고 있으며, 다양한 GNN 모델에 적용 가능합니다.

- **Performance Highlights**: RobustCRF의 유효성을 다양한 노드 분류 데이터셋을 통해 검증하였으며, 여러 모델에서 실험적으로 그 효과를 입증하였습니다. 이는 특히 전이 학습된 모델의 강건성을 향상시키는 데 기여할 것으로 기대됩니다.



### Advancing Meteorological Forecasting: AI-based Approach to Synoptic Weather Map Analysis (https://arxiv.org/abs/2411.05384)
- **What's New**: 이 연구는 기상 예측의 정확성을 높이기 위한 새로운 전처리 방법 및 합성곱 오토인코더(Convolutional Autoencoder) 모델을 제안합니다. 이 모델은 과거의 기상 패턴과 현재의 대기 조건을 비교 분석하여 기상 예측의 효율성을 향상시킬 수 있는 잠재력을 가지고 있습니다.

- **Technical Details**: 연구는 VQ-VQE와 같은 비지도 학습(un_supervised learning) 모델과 VGG16, VGG19, Xception, InceptionV3, ResNet50과 같은 지도 학습(supervised learning) 모델을 포함합니다. 모델의 성능은 유사성 비교를 위한 지표로 코사인 유사도(cosine similarity)를 사용하여, 과거 기상 패턴을 정확하게 식별할 수 있도록 최적화 과정이 이루어졌습니다.

- **Performance Highlights**: 연구 결과, 제안된 모델은 기존의 사전 학습(pretrained) 모델들과 비교할 때 과거 기상 데이터의 식별에서 우수한 성능을 보였으나, 유사성을 판별하는 데 한계도 존재했습니다. 이 모델은 기상학자들이 정보를 더욱 신속하고 정밀하게 분석할 수 있도록 도와줍니다.



### Machine learning for prediction of dose-volume histograms of organs-at-risk in prostate cancer from simple structure volume parameters (https://arxiv.org/abs/2411.05378)
- **What's New**: 이 연구는 기계 학습(machine learning)을 사용하여 직장(rectum)과 방광(bladder)의 용적(volume)으로부터 방사선 치료에서의 용적-선량(dose-volume) 예측을 하려는 새로운 접근법을 제안합니다.

- **Technical Details**: 94명의 전립선암 환자를 대상으로 하여 6000cGy의 방사선 치료 계획에 따른 용적-선량 정보를 텍스트 파일로 추출하였고, 이를 바탕으로 훈련 데이터셋을 생성하였습니다. 여러 통계 모델링, 기계 학습 방법 및 새로운 퍼지 규칙 기반 예측(fuzzy rule-based prediction, FRBP) 모델을 탐색하고 독립적인 39명의 환자 데이터셋에 대해 검증하였습니다.

- **Performance Highlights**: 직장과 방광의 경우, 4000-6420cGy 범위에서 각각 1.7-2.4% 및 2.0%-3.7%의 중앙 절대 오차(median absolute error)를 기록했습니다. FRBP 모델은 5300cGy, 5600cGy, 6000cGy에서 직장에 대해 1.2%, 1.3%, 0.9%의 오차를, 방광에 대해서는 1.6%, 1.2%, 0.1%의 오차를 보였습니다. 이러한 결과는 직장과 방광의 구조물 용적만으로도 임상적으로 중요한 용적-선량 매개변수를 정확하게 예측할 수 있음을 나타냅니다.



### RED: Residual Estimation Diffusion for Low-Dose PET Sinogram Reconstruction (https://arxiv.org/abs/2411.05354)
- **What's New**: 이번 연구에서는 positron emission tomography (PET)에서 저선량 (low-dose) 영상 재구성의 품질을 향상시키기 위해 residual estimation diffusion (RED)라는 새로운 diffusion 모델을 제안합니다.

- **Technical Details**: RED는 sinogram의 잔여(residual) 정보를 사용하여 Gaussian noise를 대체하며, 저선량과 전선량(full-dose) sinogram을 재구성의 시작점과 끝점으로 설정하는 차별화된 접근 방식을 채택합니다. 또한, 데이터 일관성 (data consistency) 측면에서 예측 오류를 줄이기 위해 drift correction 전략을 도입하였습니다.

- **Performance Highlights**: 실험 결과, RED가 저선량 sinogram의 품질 및 재구성 결과를 효과적으로 개선하는 것으로 나타났습니다.



### Controlling Grokking with Nonlinearity and Data Symmetry (https://arxiv.org/abs/2411.05353)
Comments:
          15 pages, 14 figures

- **What's New**: 이 논문은 신경망에서 모듈러 산술(modular arithmetic)의 grokking 행동을 활성화 함수(activation function)의 프로필 수정과 모델의 깊이(depth) 및 너비(width)를 변경하여 제어할 수 있음을 보여줍니다.

- **Technical Details**: 마지막 신경망(layer) 층의 가중치(weights)의 짝수 PCA 투영(even PCA projections)과 홀수 투영(odd projections)을 플로팅하여 비선형성(nonlinearity)을 증가시키면서 패턴이 더욱 고른 형태로 변하는 것을 발견했습니다. 이러한 패턴은 P가 비소수(nonprime)일 때 P를 인수분해(factor)하는 데 사용될 수 있습니다.

- **Performance Highlights**: 신경망의 일반화 능력(generalization ability)은 층 가중치의 엔트로피(entropy)로부터 유도되며, 비선형성의 정도는 최종 층의 뉴런 가중치(local entropy) 간의 상관관계(correlations)와 관련이 있습니다.



### Reinforcement Learning for Adaptive Resource Scheduling in Complex System Environments (https://arxiv.org/abs/2411.05346)
- **What's New**: 이번 연구에서는 Q-learning을 기반으로 한 새로운 컴퓨터 시스템 성능 최적화 및 적응형 작업 관리 스케줄링 알고리즘을 제시합니다. 현대의 컴퓨팅 환경에서 기존의 정적 스케줄링 방법들이 효과적인 자원 할당과 실시간 적응성이 부족하다는 문제를 해결하고자 하였습니다.

- **Technical Details**: Q-learning은 강화 학습(Deep Reinforcement Learning)의 일종으로, 시스템 상태 변화에 대한 지속적인 학습을 통해 동적 스케줄링과 자원 최적화를 가능하게 합니다. 연구는 기존의 Round-Robin 및 Priority Scheduling과 같은 전통적인 방법과 동적 자원 할당(Dynamic Resource Allocation, DRA) 알고리즘을 초월하는 성능을 보여줍니다.

- **Performance Highlights**: 실험을 통해 제안된 알고리즘이 작업 완료 시간 및 자원 활용도에서 뛰어난 성능을 보였으며, 이는 컴퓨팅 환경의 복잡성과 예측 불가능성이 증가하는 문제를 해결하는 데 기여할 수 있습니다. 또한 AI 기반의 적응형 스케줄링이 대규모 시스템에 통합될 가능성을 열어줍니다.



### Discovering Latent Structural Causal Models from Spatio-Temporal Data (https://arxiv.org/abs/2411.05331)
- **What's New**: 본 논문에서는 복잡한 상호작용을 가진 시공간 격자 데이터를 명시적으로 모델링하기 위해 SPACY(SPAtiotemporal Causal discoverY)라는 새로운 프레임워크를 제시합니다. 이 방법은 변분 추론(variational inference)에 기반하여 잠재적(time-series) 변수와 그것들의 인과 관계를 동시에 추론합니다.

- **Technical Details**: SPACY는 Radial Basis Functions (RBFs)를 사용하여 시공간 요인의 위치와 스케일 매개변수를 학습하며, 이러한 요인들은 추론된 잠재적 시계열과 관련된 격자 위치를 결정합니다. 또한, 이 프레임워크는 순간적인(edge) 효과와 중첩된 공간 요인을 처리할 수 있습니다.

- **Performance Highlights**: 실험적으로 본 방법은 합성 데이터에서 최첨단 기법보다 우수한 성능을 보여주었으며, 대규모 격자에서도 확장 가능성을 보여주었습니다. 또한, 실제 기후 데이터에서 주요 현상을 성공적으로 식별할 수 있었습니다.



### Inversion-based Latent Bayesian Optimization (https://arxiv.org/abs/2411.05330)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 본 논문에서는 Latent Bayesian Optimization (LBO) 방법의 문제점을 해결하기 위해 새로운 모듈인 Inversion-based Latent Bayesian Optimization (InvBO)를 제안합니다. 주로 'misalignment problem'과 trust region anchor selection의 한계를 다룹니다.

- **Technical Details**: InvBO는 두 가지 주요 구성 요소로 구성됩니다: inversion method와 potential-aware trust region anchor selection. Inversion method는 주어진 타겟 데이터를 완전히 재구성하는 latent code를 검색하며, potential-aware anchor selection은 optimization 과정의 향상을 고려하여 trust region의 중심을 선택합니다.

- **Performance Highlights**: 실험 결과, InvBO는 9개의 실제 벤치마크(예: molecule design, arithmetic expression fitting)의 성능에서 기존 방법에 비해 큰 성능 향상을 보이며, 최신 기술 수준(state-of-the-art)을 달성했습니다.



### SASWISE-UE: Segmentation and Synthesis with Interpretable Scalable Ensembles for Uncertainty Estimation (https://arxiv.org/abs/2411.05324)
Comments:
          16 pages, 12 figures, 5 tables

- **What's New**: 이 논문은 의료 딥러닝 모델의 해석 가능성을 향상시키는 효율적인 서브 모델 앙상블 프레임워크를 소개합니다. 이를 통해 모델 출력의 신뢰성을 평가할 수 있는 불확실성 맵을 생성합니다.

- **Technical Details**: SASWISE라는 효율적인 앙상블 방법을 제안하며, U-Net 및 UNETR 모델을 활용하여 CT 체적 분할(segmentation) 및 MR-CT 합성을 위한 데이터셋에서 테스트하였습니다. 모형 가족을 훈련하여 단일 체크포인트에서 다양한 모델을 생성할 수 있는 전략을 개발하였습니다.

- **Performance Highlights**: CT 체적 분할에서 평균 Dice 계수가 0.814, MR-CT 합성에서는 89.43 HU에서 88.17 HU로 개선되었습니다. 손상 및 언더샘플링 데이터에서도 불확실성과 오류 간의 상관관계를 유지하는 강인성을 보여줍니다.



### Fairness in Monotone $k$-submodular Maximization: Algorithms and Applications (https://arxiv.org/abs/2411.05318)
Comments:
          17 pages. To appear in IEEE BigData 2024

- **What's New**: 이 논문에서는 공정성을 고려한 k-submodular maximization 문제를 다루고 있습니다. 이 연구는 공정성을 포함하는 k-submodular 최적화의 최초 연구로, 기존 최적화 결과와 동등한 이론적 보장을 제공합니다.

- **Technical Details**: 제안된 greedy 알고리즘은 $\frac{1}{3}$ 근사 확률을 보장하며, 실행 시간은 $\mathcal{O}(knB)$입니다. 또한, 더 빠른 threshold 기반 알고리즘을 개발하여 $(\frac{1}{3} - \epsilon)$ 근사치를 achieve하면서 $\mathcal{O}(\frac{kn}{\epsilon} \log \frac{B}{\epsilon})$ 함수 평가를 수행합니다. 이 알고리즘들은 대략적인 oracle을 통해 접근할 수 있는 경우에서도 근사 보장을 제공합니다.

- **Performance Highlights**: 실험 결과에 따르면, 공정성 제약 조건이 해결책의 질에 미치는 영향은 크지 않음을 보여주었습니다. 특히, k개의 주제를 갖는 영향력 극대화 및 k개의 종류를 갖는 센서 배치 문제에 대한 사례 연구를 통해 '공정성의 대가'를 분석하였습니다.



### Exploring the Alignment Landscape: LLMs and Geometric Deep Models in Protein Representation (https://arxiv.org/abs/2411.05316)
Comments:
          24 pages, 9 figures

- **What's New**: 이번 연구는 단백질 분야에서 LLM(대형 언어 모델)과 GDM(기하학적 딥 모델) 간의 다중 모드 표현 정렬을 탐구하고, 이를 통해 단백질 관련 MLLM(다중 모드 대형 언어 모델)의 성능을 향상시키는 방법을 제안합니다.

- **Technical Details**: 연구에서는 Gemma2-2B, LLaMa3.1-8B, LLaMa3.1-70B 등 세 가지 최신 LLM과 GearNet, GVP, ScanNet, GAT 등 네 가지 단백질 전문 GDM을 평가했습니다. 또한, 두 가지 층 프로젝션 헤드를 사용하고 LLM을 단백질 특화 데이터로 세부 조정(fine-tuning)하면 정렬 품질이 크게 향상된다는 것을 발견했습니다.

- **Performance Highlights**: 정렬 품질을 향상시키는 여러 전략을 제안하며, GDM이 그래프 및 3D 구조 정보를 통합할수록 LLM과의 정렬 성능이 개선되고, 더 큰 LLM이 더 나은 정렬 능력을 보여주며, 단백질의 희귀성이 정렬 성능에 큰 영향을 미친다는 주요 결과를 도출했습니다.



### On Training of Kolmogorov-Arnold Networks (https://arxiv.org/abs/2411.05296)
Comments:
          7 pages, 6 figures

- **What's New**: 최근 Kolmogorov-Arnold Networks (KAN)은 다층 Perceptron 아키텍처에 대한 유연한 대안으로 소개되었습니다. 이 논문에서는 여러 KAN 아키텍처의 훈련 동역학을 분석하고 이를 해당하는 MLP 형태와 비교합니다.

- **Technical Details**: KAN 아키텍처는 B-spline을 사용하여 입력 데이터에 적합하면서 웨이브릿 개념을 기반으로 하며, HSIC Bottleneck과 같은 백 프로파게이션을 사용하지 않는 방법으로 훈련됩니다. 우리는 다양한 초기화 관리 기법과 최적화 기법, 학습률을 바탕으로 여러 가지 조합을 실험했습니다.

- **Performance Highlights**: 테스트 정확도에 따라 KAN은 고차원 데이터셋에서 MLP 아키텍처에 대한 효과적인 대안으로 간주되며, 파라미터 효율성에서 약간 더 우수하나 훈련 동역학에서는 더 불안정한 경향이 있음을 발견했습니다. 논문에서는 KAN 모델의 훈련 안정성을 향상시키기 위한 추천 사항도 제공합니다.



### GPT Semantic Cache: Reducing LLM Costs and Latency via Semantic Embedding Caching (https://arxiv.org/abs/2411.05276)
- **What's New**: 본 논문에서는 GPT Semantic Cache라는 새로운 방법을 소개합니다. 이 방법은 사용자 쿼리의 임베딩을 메모리 내 저장소인 Redis에 캐시하여, 유사한 질문을 효율적으로 검색하고, 미리 생성된 응답을 반환함으로써 API 호출을 줄입니다.

- **Technical Details**: GPT Semantic Cache 시스템은 세 가지 주요 구성 요소로 이루어집니다: 쿼리를 임베딩으로 변환하는 임베딩 생성, Redis를 활용한 메모리 내 캐싱, 유사성을 식별하는 근사 최근접 이웃(ANN) 검색. 쿼리가 도착하면 이를 임베딩으로 변환하고, 캐시에서 유사한 쿼리를 찾습니다. 일치하는 쿼리가 발견되면, 해당 응답을 바로 제공합니다.

- **Performance Highlights**: GPT Semantic Cache는 응답 속도를 향상시키고 운영 비용을 절감하는 데 기여합니다. 이 시스템은 높은 쿼리 볼륨을 효과적으로 처리하면서도 사용자에게 신속한 응답을 제공합니다.



### Distributed-Order Fractional Graph Operating Network (https://arxiv.org/abs/2411.05274)
- **What's New**: 이번 논문에서는 분산 차수의 분수 미적분을 포함하는 새로운 연속 그래프 신경망(GNN) 프레임워크인 DRAGON(Distributed-order fRActional Graph Operating Network)을 소개합니다.

- **Technical Details**: DRAGON은 기존의 정수 차수 또는 단일 분수 차수 미분 방정식을 사용하는 전통적인 연속 GNN과 달리, 실수 범위의 도함수 차수에 대해 학습 가능한 확률 분포를 사용합니다. 이를 통해 여러 도함수 차수를 유연하게 조합하여 복잡한 그래프 특징 업데이트 동역학을 포착할 수 있습니다. 이 프레임워크는 비정부적 그래프 랜덤 워크(non-Markovian graph random walk)와 이상 확산(anomalous diffusion) 과정에 의해 구동되는 노드 특징 업데이트 관점에서 그 능력을 해석합니다.

- **Performance Highlights**: DRAGON 프레임워크는 다양한 그래프 학습 작업에 대한 실험을 통해 전통적인 연속 GNN 모델에 비해 일관되게 우수한 성능을 보여줍니다.



### ZipNN: Lossless Compression for AI Models (https://arxiv.org/abs/2411.05239)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2404.15198

- **What's New**: 이 논문은 모델 크기 증가와 배포에 따른 인프라 부담을 해결하기 위해, 전통적인 lossless compression(무손실 압축) 방법을 사용하여 신경망을 효과적으로 압축하는 새로운 기술인 ZipNN을 소개합니다.

- **Technical Details**: ZipNN은 신경망에 특화된 무손실 압축 기법으로, 모델의 지수(exponent) 부분을 분리하여 압축하며, Entropy Encoding을 사용한 Huffman 코드로 성능과 압축 비율을 향상시킵니다. BF16 모델에서는 약 33%의 공간 절약을 달성하고, FP32 모델에서는 17%의 공간 절약을 보입니다. 또한, delta compression(델타 압축)을 활용하여 유사한 모델 간의 차이를 압축할 수 있는 가능성에 대해 연구했습니다.

- **Performance Highlights**: ZipNN은 기존의 압축 알고리즘인 Zstd에 비해 BF16 모델에서 17% 향상된 압축 비율과 62%의 속도 향상을 보여주며, clean 모델의 경우 34%의 공간 절약과 4.6배의 압축 시간 단축, 83%의 복원 속도 향상을 달성했습니다.



### Generating Highly Designable Proteins with Geometric Algebra Flow Matching (https://arxiv.org/abs/2411.05238)
Comments:
          To be published in proceedings of NeurIPS 2024

- **What's New**: 본 논문에서는 단백질 골격 설계를 위한 생성 모델을 소개하며, 기하학적 곱(Geometric Products)과 고차 메시지 전송(Higher Order Message Passing)을 수반합니다. 특히, 제안한 클리포드 프레임 주의(Clifford Frame Attention, CFA)는 AlphaFold2의 불변 점 주의(Invariant Point Attention, IPA) 아키텍처를 확장한 것입니다.

- **Technical Details**: CFA는 단백질의 잔기 프레임과 기하학적 특성을 원근 기하학 대수(Projective Geometric Algebra, PGA)에 기반하여 표현합니다. 이로 인해, 잔기 간의 기하학적으로 표현력 있는 메시지를 구성할 수 있으며, 고차 용어를 포함한 메시지를 생성할 수 있습니다. 제안된 모델은 단백질 골격 생성을 위한 FrameFlow라는 최첨단 흐름 일치 모델에 통합되어 구현되었습니다.

- **Performance Highlights**: 제안된 모델은 설계 가능성(Designability), 다양성(Diversity), 참신성(Novelty)에서 높은 성과를 달성하였으며, 자연에서 발생하는 단백질의 이차 구조 요소의 통계적 분포를 따르는 단백질 골격 샘플링을 성공적으로 수행합니다. 특히, 작은 단백질에 대해서도 α-헬릭스(α-helices)와 β-시트(β-sheets)의 균형 잡힌 분포를 캡처하며, 이는 다양한 기능을 가진 단백질 설계에 중요한 요소로 작용합니다.



### Pruning the Path to Optimal Care: Identifying Systematically Suboptimal Medical Decision-Making with Inverse Reinforcement Learning (https://arxiv.org/abs/2411.05237)
Comments:
          13 pages, 4 figures

- **What's New**: 의료 환경에서 수집된 관찰 데이터로부터 의료 의사결정에 대한 통찰력을 발견하기 위해, 동료의 행동에 기반한 비최적(가장 좋은 결과를 내지 못하는) 임상의 행동을 식별하는 새로운 형태의 Inverse Reinforcement Learning (IRL)을 제시합니다.

- **Technical Details**: 이 방법론은 IRL의 두 단계를 중심으로 구성되며, 전문가 상의 행동에서 크게 벗어난 경로를 제거하는 중간 단계를 포함하고 있습니다. 이로 인해 최적 및 비최적 임상 결정을 포함한 ICU 데이터에서 임상 우선 사항과 가치를 효과적으로 식별할 수 있습니다.

- **Performance Highlights**: 비최적 행동을 제거하는 이점은 질병에 따라 다르며, 특정 인구 집단에 차별적인 영향을 미치는 것으로 관찰되었습니다.



### Performative Reinforcement Learning with Linear Markov Decision Process (https://arxiv.org/abs/2411.05234)
- **What's New**: 본 연구는 	extit{performative reinforcement learning}의 설정을 연구하며, 배포된 정책이 보상(reward)과 기본 Markov 결정 프로세스의 전환(transition)에 미치는 영향을 분석합니다. 기존 연구는 탭(Tabular) 설정 하에 이 문제를 다루었고, 반복 재학습의 마지막 반복 수렴(last-iterate convergence)을 보여주었습니다. 이번 논문에서는 이 결과를 	extit{linear Markov decision processes}로 일반화하였습니다.

- **Technical Details**: 선형 MDP의 주요 도전 과제는 정규화된 목표(objective)가 더 이상 강한 볼록성(strong convexity)을 갖지 않으며, 우리는 꼭짓점 수가 무한할 수 있기 때문에 상태(state)가 아닌 특징(feature)의 차원(dimension)에 비례하는 경계를 얻고자 합니다. 반복적으로 정규화된 목표를 최적화하는 것이 	extit{performatively stable policy}로 수렴한다는 것을 처음으로 보였습니다. 강한 볼록성이 없는 상황에서, 우리의 분석은 최적 이중 해(solution)에 대한 특정 선형 조합을 이용한 새로운 재발 관계(recurrence relation)을 활용합니다.

- **Performance Highlights**: 유한 샘플 설정(finite sample setting)에서는 학습자가 현재 정책으로부터 얻은 궤적(trajectories) 집합에 접근할 수 있음을 고려하고, 샘플에서 최적화해야 할 경험적 Lagrangian(empirical Lagrangian)을 구성합니다. 	extit{bounded coverage} 조건 하에, 경험적 Lagrangian의 안장점(saddle point)을 반복적으로 해결하면 수행적인 안정 솔루션에 수렴함을 보여줍니다. 마지막으로, 다중 에이전트 시스템(multi-agent systems) 등 수행적 RL의 일반 프레임워크 여러 응용을 입증합니다.



### Solving Hidden Monotone Variational Inequalities with Surrogate Losses (https://arxiv.org/abs/2411.05228)
- **What's New**: 본 논문에서는 variational inequality (VI) 문제 해결을 위한 새로운 서로게이트 기반 접근 방식을 제안합니다. 우리는 우리 방법이 깊은 학습에 통합 가능하며, 실용적인 가정 하에 수렴을 보장하고 기존 방법에 대한 통합적 관점을 제공한다고 주장합니다.

- **Technical Details**: 이 연구는 VI 문제의 해결을 위해 서브가변 손실(surrogate loss)을 사용하는 새로운 알고리즘을 제안합니다. 이 접근법은 ADAM 최적화기와 호환되며, 모델 파라미터 맵핑 g의 구조를 활용합니다. 실험적으로, 이 방법은 min-max 최적화 및 프로젝션 벨만 오차(minimizing projected Bellman error) 문제에서 효과적임을 증명합니다.

- **Performance Highlights**: 우리는 현저하게 개선된 TD(0) 변형을 제안하여 계산 및 샘플 효율성을 높이고, 다양한 VI 문제에서 서브가변 손실 기반 최적화의 성능과 다재다능성을 입증했습니다.



### Private Algorithms for Stochastic Saddle Points and Variational Inequalities: Beyond Euclidean Geometry (https://arxiv.org/abs/2411.05198)
- **What's New**: 이번 연구에서는 확률적 새들 포인트 문제(SSP)와 확률적 변별 불평등(SVI)을 연구하였으며, $(	ext{ε}, 	ext{δ})$-차별적 프라이버시(DP) 하에서의 해법을 제시합니다. 이는 유클리드 및 비유클리드 환경 모두에서 적용됩니다. 특히 $p,q 	ext{in} [1,2]$인 경우에 대한 일반적인 알고리즘과 성과를 도출했습니다.

- **Technical Details**: 우리는 $	ilde{O}igg(rac{1}{	ext{sqrt}(n)} + rac{	ext{sqrt}(d)}{n	ext{ε}}igg)$의 강한 SP-gap 경계를 도출했습니다. 주목할 점은 이전 연구가 무조건적인 가정을 두지 않고 SSP에 대한 일반적인 알고리즘을 제공했다는 것입니다. 다차원 VIs의 경우, 경계 조건에서 비슷한 형태의 경계를 제공합니다.

- **Performance Highlights**: 제시된 알고리즘은 $	ilde{O}igg(	ext{min}iggig{rac{n^{2}	ext{ε}^{1.5}}{	ext{sqrt}(d)}, n^{3/2}igg)igg)$의 수렴 속도로 수렴합니다. 이 성과는 널리 알려진 하한과 비교했을 때 거의 최적의 비율을 달성했습니다.



### Hardware and Software Platform Inferenc (https://arxiv.org/abs/2411.05197)
- **What's New**: 이 논문에서는 큰 언어 모델(LLM)을 사용할 때의 투명성과 신뢰성을 높이기 위한 새로운 방법인 하드웨어 및 소프트웨어 플랫폼 추론(Hardware and Software Platform Inference, HSPI)을 제안합니다. HSPI는 모델의 입력-출력 행동만으로 GPU 아키텍처와 소프트웨어 스택을 식별할 수 있는 기법입니다.

- **Technical Details**: HSPI는 서로 다른 GPU 아키텍처와 컴파일러의 고유한 차이를 이용하여 특정 하드웨어 및 소프트웨어 환경을 구별합니다. 본 연구에서는 두 가지 방법인 HSPI with Border Inputs (HSPI-BI)와 HSPI with Logits Distributions (HSPI-LD)를 도입하며, 이 방법들이 다양한 모델을 대상으로 흰색 상자(white-box) 및 검은 상자(black-box) 설정에서 성능을 평가하였습니다.

- **Performance Highlights**: HSPI는 흰색 상자 설정에서는 83.9%에서 100%의 정확도로 다양한 GPU를 구별할 수 있었고, 검은 상자 설정에서도 무작위 추측 정확도의 최대 3배 높은 결과를 달성했습니다. 이 연구 결과는 모델의 투명성과 책임성을 높이는 데 기여할 수 있는 가능성을 보여줍니다.



### On Erroneous Agreements of CLIP Image Embeddings (https://arxiv.org/abs/2411.05195)
Comments:
          18 pages, 4 figures

- **What's New**: 본 연구에서는 Vision-Language Models (VLMs)의 시각적 추론에서 발생하는 오류의 주 원인이 항상 잘못된 합의(erroneous agreements) 때문이 아님을 보여줍니다. LLaVA-1.5-7B 모델은 CLIP 이미지 인코더를 사용하면서도 퀘리와 관련된 시각적 정보를 추출할 수 있음을 밝혔습니다.

- **Technical Details**: CLIP 이미지 인코더는 시각적으로 구별되는 이미지들이 높은 코사인 유사도(cosine similarity)로 애매하게 인코딩되는 문제를 가지고 있습니다. 그러나 LLaVA-1.5-7B는 이러한 CLIP 이미지 임베딩에서 추출 가능한 정보를 이용해 더 나은 성능을 보입니다. 이를 위해 M3ID(Multi-Modal Mutual-Information Decoding)와 같은 대체 디코딩 알고리즘을 사용하여 시각적 입력에 더 많은 주의를 기울일 수 있게 했습니다.

- **Performance Highlights**: LLaVA-1.5-7B는 What'sUp 벤치마크에서 100%에 가까운 정확도로 작업을 수행하였고, MMVP 벤치마크에서도 CLIP 기반 모델보다 우수한 성능을 나타냈습니다. 전체적으로, CLIP 이미지 인코더를 개선하는 것도 중요하지만, 고정된 이미지를 사용하는 모델에서도 정보를 더 잘 추출하고 활용하는 전략을 적용하는 여지가 남아 있음을 보여주고 있습니다.



### Interactive Dialogue Agents via Reinforcement Learning on Hindsight Regenerations (https://arxiv.org/abs/2411.05194)
Comments:
          23 pages, 5 figures

- **What's New**: 본 논문에서는 대화 에이전트가 대화를 효과적으로 이끌 수 있는 능력을 향상시키기 위한 새로운 방법을 제안합니다. 특히, 기존 데이터에 대한 후행(regeneration)을 통해 대화 에이전트를 훈련시키는 방법을 소개하며, 이는 특히 정신 건강 지원 및 자선 기부 요청과 같은 복잡한 대화 작업에 적용됩니다.

- **Technical Details**: 대화 에이전트는 오프라인 강화 학습(offline reinforcement learning, RL)을 사용하여 훈련됩니다. 기존 비효율적인 대화 데이터를 개선하고 새로운 대화 전략을 학습하기 위해, 사후에 생성된 합성 데이터(synthetic data)를 추가하여 적절한 행동을 나타내는 다양한 대화 전략을 포착합니다.

- **Performance Highlights**: 실제 사용자를 대상으로 한 연구 결과, 제안된 방법이 기존 최첨단 대화 에이전트에 비해 효과성, 자연스러움 및 유용성 측면에서 크게 우수함을 보여주었습니다.



### Q-SFT: Q-Learning for Language Models via Supervised Fine-Tuning (https://arxiv.org/abs/2411.05193)
Comments:
          16 pages, 4 figures

- **What's New**: 이번 연구에서는 기존의 가치 기반 강화 학습(value-based reinforcement learning, RL) 알고리즘의 한계를 극복하기 위한 새로운 오프라인 RL 알고리즘을 제안합니다. 이 알고리즘은 Q-learning을 수정된 감독 세부 조정(supervised fine-tuning, SFT) 문제로 간주하며, 이를 통해 언어 모델의 사전 학습(pretraining) 효과를 충분히 활용할 수 있습니다.

- **Technical Details**: 제안된 알고리즘은 최대 우도(maximum likelihood) 목표에 가중치를 추가하여, 잔여 정책(behavior policy) 대신 보수적으로 가치 함수를 추정하는 확률을 학습합니다. 이를 통해 비안정적인 가치 학습(regression) 목표를 피하고, 대규모 사전 학습에서 발생한 초기 우도(likelihood)를 직접 활용할 수 있습니다.

- **Performance Highlights**: 실험을 통해 LLM(대형 언어 모델)과 VLM(비전-언어 모델)의 다양한 작업에서 제안된 알고리즘의 효과를 입증했습니다. 자연어 대화, 로봇 조작 및 네비게이션 등 여러 과제에서 기존의 감독 세부 조정 방법을 초월하는 성능을 보였습니다.



### Adversarial Robustness of In-Context Learning in Transformers for Linear Regression (https://arxiv.org/abs/2411.05189)
- **What's New**: 이번 연구는 트랜스포머 모델의 인컨텍스트 학습에서 발생할 수 있는 hijacking 공격에 대한 취약성을 조사합니다. 특히, 선형 회귀 작업을 설정으로 사용하여, 단일 레이어 트랜스포머의 비강건성을 입증하고 복잡한 트랜스포머에 대한 공격 가능성을 제시합니다.

- **Technical Details**: 트랜스포머 모델의 hijacking 공격은 프롬프트 조작 공격으로, 공격자가 프롬프트를 수정하여 특정 출력을 강제하는 방식입니다. 온전한 공격이 성공하는 단일 레이어 선형 트랜스포머는 예제를 약간 조작하여 임의의 예측을 출력할 수 있습니다. 반면에, GPT-2 아키텍처를 가진 모델에서는 이러한 공격이 이전처럼 효과적이지 않지만, gradient 기반 공격을 통해 조작이 가능합니다. 또한 적대적 훈련(adversarial training)이 이러한 공격에 대한 강건성을 높인다는 사실도 발견했습니다.

- **Performance Highlights**: 실험 결과, 단일 레이어 선형 트랜스포머는 hijacking 공격에 비강건한 반면, 더 복잡한 GPT-2 아키텍처에서는 hijacking 공격의 이동성이 제한됩니다. 적대적 훈련이 적용된 경우, 트랜스포머 모델은 hijacking 공격으로부터 더욱 강건함을 보여주었습니다.



### Inverse Transition Learning: Learning Dynamics from Demonstrations (https://arxiv.org/abs/2411.05174)
- **What's New**: 본 논문에서는 오프라인 모델 기반 강화 학습(context of offline model-based reinforcement learning) 환경에서 near-optimal expert trajectories를 통해 상태 전이 동역학 T^*의 추정 문제를 다루고 있으며, Inverse Transition Learning(ITL)이라는 새로운 제약 기반(method)이 개발되었습니다.

- **Technical Details**: Inverse Transition Learning(ITL)은 전문가의 행동 궤적이 불완전한 데이터 샘플 시성을 띄고 있다는 사실을 특징으로 활용하며, Bayesian 접근법을 통합하여 T^*의 추정을 개선합니다. 이 방법은 MCE(Maximum Causal Entropy) 기반의 기존 접근 방식에 비해 더 빠르고 신뢰할 수 있습니다.

- **Performance Highlights**: 이 연구는 인공 환경(synthetic environments) 및 실제 의료 상황(real healthcare scenarios)에서의 효과를 입증하였습니다. ICU(중환자실) 환자 관리를 포함한 테스트는 의사 결정 개선의 상당한 향상을 보였으며, posterior는 전이가 성공할 시기를 예측하는 데 유용함을 나타냈습니다.



### DWFL: Enhancing Federated Learning through Dynamic Weighted Averaging (https://arxiv.org/abs/2411.05173)
Comments:
          Accepted at SIMBig 2024

- **What's New**: 이 연구에서는 단백질 서열 분류를 위한 향상된 연합 학습 방법, 즉 Dynamic Weight Federated Learning(DWFL)을 제안합니다. 이 방법은 모델 성능 지표를 기반으로 지역 모델 가중치를 동적으로 조정하여 초기 글로벌 모델의 강력한 구성 요소를 만드는 데 목적이 있습니다.

- **Technical Details**: DWFL은 트레이닝 과정에서 지역 데이터의 품질과 모델 성능을 고려하여 성능이 높은 모델에 더 높은 가중치를 부여합니다. 이는 전통적인 연합 학습 모델들이 간과한 개별 모델 성능을 동적으로 집계하는 데 효과적입니다. DWFL 모델은 단백질 서열 분류 작업에 최적화되어 있으며, 글로벌 모델은 G=1/N∑i=1Nβi⋅Li로 표현됩니다. 여기서 G는 생성된 글로벌 모델, N은 지역 모델의 총 수, Li는 i번째 지역 모델, βi는 동적 가중치입니다.

- **Performance Highlights**: 실험 결과, DWFL을 활용한 모델의 정확도가 기존 방법들보다 향상된 것으로 나타났습니다. 이로 인해 연합 학습이 더욱 견고하고 개인정보 보호를 보장하는 협업 기계 학습 작업에 선호되는 접근 방식이 되었음을 증명합니다.



### EPIC: Enhancing Privacy through Iterative Collaboration (https://arxiv.org/abs/2411.05167)
Comments:
          Accepted at SIMBig 2024

- **What's New**: 이 논문은 Federated Learning (FL) 기반의 EPIC 아키텍처를 제안하여 SARS-CoV-2 유전체 데이터 계통(classification) 문제를 해결하는 방법을 소개합니다. EPIC은 데이터를 중앙 서버로 전송하지 않고 로컬 모델에서 파라미터만 공유하여 데이터 프라이버시를 유지합니다.

- **Technical Details**: 연구에서 제안하는 EPIC 모델은 개인 데이터의 전송 없이, 로컬 모델의 출력 가중치만 글로벌 모델로 전달하여 데이터 프라이버시를 보장합니다. 또한, 지역적 특성을 반영하여 여러 나라의 데이터를 효과적으로 활용할 수 있도록 돕습니다. 이 연구는 GISAID에서 수집한 spike protein의 699327개 시퀀스를 사용하여 EPIC의 성능을 평가하였습니다.

- **Performance Highlights**: EPIC 모델은 기존의 중앙 집중된 딥러닝 모델과 다른 최첨단(State-of-the-art, SOTA) 방법들과 비교하여 데이터 프라이버시를 보호하면서도 높은 정확도의 유전체 계통 분류 결과를 보여주었습니다. 또한, 본 연구는 국가가 특정 상황에 맞춘 맞춤형 모델을 개발하는 데 기여할 수 있음을 시사합니다.



### Exploiting the Structure of Two Graphs with Graph Neural Networks (https://arxiv.org/abs/2411.05119)
- **What's New**: 본 논문에서는 두 개의 서로 다른 그래프에서 정의된 신호 쌍을 처리하기 위한 새로운 그래프 기반 딥러닝 아키텍처를 소개합니다. 이를 통해 다중 그래프를 기반으로 하는 실시간 데이터 처리의 한계를 극복하고 다양한 작업에 적용할 수 있는 유연한 접근 방식을 제공합니다.

- **Technical Details**: 이 아키텍처는 입력 그래프에서 신호를 처리하는 GNN을 시작으로, 잠재 공간(latent space)에서 신호의 변환을 수행한 후, 출력 그래프에서 작동하는 두 번째 GNN을 구현하는 세 가지 블록 구조로 구성되어 있습니다. 이 구조는 입력 신호로부터 출력 신호로 매핑하기 위해 두 개의 GNN과 변환 함수를 결합합니다.

- **Performance Highlights**: 실험 결과, 제안된 아키텍처는 기존의 딥러닝 아키텍처보다 더 나은 성능을 보여주었으며, 두 그래프의 정보를 이용함으로써 데이터 내의 복잡한 관계를 포착할 수 있음을 입증했습니다.



### On the cohesion and separability of average-link for hierarchical agglomerative clustering (https://arxiv.org/abs/2411.05097)
Comments:
          Accepted to Neurips 2024

- **What's New**: 이 논문은 metric space에서 average-link(평균 링크) 클러스터링 방법의 성능을 포괄적으로 연구하였으며, Dasgupta의 비용 함수보다 더 해석 가능하고 자연스러운 기준으로 분리 가능성(separability)과 응집성(cohesion)을 평가했습니다.

- **Technical Details**: average-link는 대표적인 병합적(agglomerative) 클러스터링 방법으로, n개의 입력 포인트에 대해 n개의 클러스터로 시작하여 반복적으로 두 클러스터를 병합하는 방식으로 작동합니다. 본 연구에서는 각 클러스터링에 대해 n개의 포인트가 포함된 엄격한 이진 트리가 생성되며, 이를 통해 데이터 포인트의 유사성을 기반으로 한 클러스터링을 형성합니다.

- **Performance Highlights**: 실험 결과, average-link는 응집성과 분리 가능성 모두에서 다른 관련 방법들과 비교할 때 더 나은 선택임을 보여주며, Dasgupta의 변형에 대해 평균 링크가 상대적으로 뛰어난 정량적 성능을 갖는다는 점을 증명했습니다.



### Watermarking Language Models through Language Models (https://arxiv.org/abs/2411.05091)
- **What's New**: 이 논문은 언어 모델을 통한 워터마킹(watermarking)을 위한 새로운 프레임워크를 제시합니다. 제안된 접근 방식은 다중 모델(multimodal) 설정을 활용하여 언어 모델이 생성한 프롬프트를 통해 워터마킹 지침을 생성하고, 생성된 컨텐츠에 워터마크를 삽입하며, 이러한 워터마크의 존재를 검증하는 모델들을 포함합니다.

- **Technical Details**: 논문에서는 ChatGPT와 Mistral을 프롬프트 생성 및 워터마킹 모델로 사용하고, 분류기(classifier) 모델을 통해 감지 정확도를 평가했습니다. 이 프레임워크는 다양한 구성에서 95%의 ChatGPT 감지 정확도와 88.79%의 Mistral 감지 정확도를 달성했습니다.

- **Performance Highlights**: 제안된 프레임워크는 다른 언어 모델 아키텍처에서 워터마킹 전략의 유효성과 적응성을 검증합니다. 이러한 결과는 콘텐츠 획득, 저작권 보호, 모델 인증 등의 응용 분야에서 유망한 가능성을 나타냅니다.



### The Fibonacci Network: A Simple Alternative for Positional Encoding (https://arxiv.org/abs/2411.05052)
- **What's New**: 이 논문에서는 기본적인 Multi-Layer Perceptrons (MLPs) 구조를 기반으로 한 Fibonacci Network를 제안합니다. 기존의 Positional Encoding (PE) 기술을 사용하지 않고도 고주파수 신호를 효과적으로 재구성할 수 있는 새로운 접근 방식을 보여줍니다.

- **Technical Details**: Fibonacci Network는 높은 주파수를 재구성하기 위해 이전 두 블록의 출력을 사용하는 블록 기반 구조입니다. 이 네트워크는 낮은 주파수의 입력을 통해 고주파수를 재구성하는 능력을 갖추고 있으며, 훈련 방법론은 주파수 범위를 점진적으로 증가시키는 독창적인 방식입니다.

- **Performance Highlights**: 이 연구에서 제안한 Fibonacci Network는 낮은 주파수와 고주파수의 조합을 통해 신호를 거의 완벽하게 재구성할 수 있으며, 일반화 측면에서도 우수한 성능을 보입니다. 훈련 데이터가 sparsely sampled 되었음에도 불구하고, 보지 못한 주파수 구간에서의 일반화 능력이 뛰어납니다.



### Curriculum Learning for Few-Shot Domain Adaptation in CT-based Airway Tree Segmentation (https://arxiv.org/abs/2411.05779)
Comments:
          Under review for 22nd IEEE International Symposium on Biomedical Imaging (ISBI), Houston, TX, USA

- **What's New**: 이 논문에서는 심층 학습(Deep Learning) 기술을 활용하여 흉부 CT 스캔에서 기도를 자동으로 분할하는 문제를 해결하기 위해 교육 과정 학습(Curriculum Learning)을 통합한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 훈련 데이터를 CT 스캔과 이에 해당하는 정답 트리 피처에서 얻은 복잡도 점수에 따라 배치로 분배하여 모델을 훈련시키며, 적은 샷 도메인 적응(few-shot domain adaptation)을 적용합니다. 이는 전체 미세 조정(Fine-Tuning) 데이터 세트를 수동으로 주석을 달기 어렵거나 비용이 높은 상황을 겨냥합니다.

- **Performance Highlights**: ATM22와 AIIB23 두 개의 대규모 공개 코호트에서 교육 과정 학습을 이용한 전훈련(Source domain) 및 적은 샷 미세 조정(Target domain)을 통해 높은 성능을 보였습니다. 그러나 전통적인 부트스트래핑 scoring function 사용 시 또는 올바른 스캔 순서를 사용하지 않을 시 발생할 수 있는 잠재적 부정적인 효과에 대한 통찰도 제공합니다.



### Sketched Equivariant Imaging Regularization and Deep Internal Learning for Inverse Problems (https://arxiv.org/abs/2411.05771)
- **What's New**: 이 논문에서는 기존의 Equivariant Imaging (EI) 기반 무감독 훈련 방식의 비효율성을 줄이기 위해 스케치된 EI 정규화 기법을 제안합니다. 이 기법은 무작위 스케치 기술을 활용하여 고차원 응용에서의 계산 효율성을 개선합니다.

- **Technical Details**: 제안된 Sketched Equivariant Deep Image Prior (DIP) 프레임워크는 단일 이미지 기반 및 작업 적응형 복원에 효율적으로 적용할 수 있으며, EI 정규화의 차원 축소를 통한 가속화를 통해 계산 성능을 대폭 향상시킵니다.

- **Performance Highlights**: X-ray CT 이미지 복원 작업에서의 연구 결과, 제안된 방법이 기존 EI 기반 방법에 비해 수량적 계산 가속화를 달성하며, 테스트 시 네트워크 적응이 가능함을 보여줍니다.



### FinDVer: Explainable Claim Verification over Long and Hybrid-Content Financial Documents (https://arxiv.org/abs/2411.05764)
Comments:
          EMNLP 2024

- **What's New**: 이번 논문에서는 FinDVer라는 새로운 벤치마크를 소개합니다. 이는 LLM(대형 언어 모델)의 설명 가능한 청구 검증 기능을 평가하기 위해 특별히 설계된 포괄적인 기준입니다. 유효한 정보 추출, 수치 추론, 지식 집약적 추론의 세 가지 하위 집합으로 나뉘어 있으며, 총 2,400개의 전문가 주석 샘플이 포함되어 있습니다.

- **Technical Details**: FinDVer는 금융 문서에서의 청구 검증을 위한 최초의 맥락 기반 벤치마크로, 두 가지 설정인 길이가 긴 문맥(long-context)과 검색 보강 생성(RAG) 설정에서 다양한 LLM을 평가합니다. 저자들은 9개 기관의 16개 모델을 테스트하였고, 각 예제는 세부적인 증거 지원과 단계별 추론 과정을 제공하는 주석으로 뒷받침됩니다.

- **Performance Highlights**: 실험 결과, 현재 가장 성능이 뛰어난 LLM인 GPT-4o조차도 인간 전문가보다 상당히 뒤처지는 것으로 나타났습니다(76.2% 대 93.3%). 이러한 결과는 LLM이 금융 문서의 복잡성을 이해하고 처리하는 데 여전히 많은 도전 과제가 있음을 보여줍니다.



### On Differentially Private String Distances (https://arxiv.org/abs/2411.05750)
- **What's New**: 이 논문에서는 Hamming 거리와 edit 거리 계산을 위한 새로운 차등 프라이버시(differentially private, DP) 데이터 구조를 제안합니다. 이 데이터 구조는 비밀성을 보장하면서도 높은 성능을 제공한다는 점에서 혁신적입니다.

- **Technical Details**: 논문에서는 데이터베이스에 포함된 비트 문자열의 거리 추정 문제를 다룹니다. Hamming 거리의 경우, 쿼리를 처리하는 데 걸리는 시간은 O(mk+n)이며, 각 추정값은 실제 거리와 최대 O(k/e^(ε/log k))의 차이를 가집니다. Edit 거리의 경우, 쿼리 처리 시간은 O(mk^2+n)이며, 추정값의 차이는 O(k/e^(ε/(log k log n)))로 제한됩니다.

- **Performance Highlights**: 제안된 데이터 구조는 ε-DP를 유지하며, moderate한 k값에 대해 서브라인 리어 쿼리 작업을 지원합니다. 이는 효율성을 고려한 차별화된 접근 방식을 보여줍니다.



### Learning Subsystem Dynamics in Nonlinear Systems via Port-Hamiltonian Neural Networks (https://arxiv.org/abs/2411.05730)
Comments:
          Preprint submitted to ECC 2025

- **What's New**: 이번 연구에서는 Port-Hamiltonian Neural Networks (pHNNs)를 사용하여 복잡한 비선형 상호연결 시스템의 개별 서브시스템을 식별하는 새로운 방법을 제안합니다.

- **Technical Details**: pHNNs는 물리 법칙과 딥러닝 기법을 통합하는 강력한 모델링 도구로, 입력-출력 데이터만으로 서브시스템의 동특성을 식별할 수 있는 알고리즘을 개발하였습니다. 이 방법은 실험적 접근 없이도 서브시스템의 동적 행동을 효과적으로 모델링할 수 있게 해줍니다. 또한, 측정 노이즈를 처리하기 위해 출력 오차(OE) 모델 구조를 선택하였습니다.

- **Performance Highlights**: 제안된 접근법은 여러 상호연결 시스템에 대한 테스트를 통해 그 효과를 입증했으며, 서브시스템 동역학을 식별하고 새로운 상호연결 모델에 통합할 수 있는 잠재력을 보여주었습니다.



### A Retrospective on the Robot Air Hockey Challenge: Benchmarking Robust, Reliable, and Safe Learning Techniques for Real-world Robotics (https://arxiv.org/abs/2411.05718)
Comments:
          Accept at NeurIPS 2024 Dataset and Benchmark Track

- **What's New**: 2023 NeurIPS에서 열린 Robot Air Hockey Challenge는 로봇학습을 위한 새로운 벤치마크로, 기존의 단순한 시뮬레이션 기반 테스트와는 달리 실제 로봇 환경에서의 적용 가능성을 중점적으로 다루고 있습니다.

- **Technical Details**: 이 대회는 시뮬레이션과 실제 환경 간의 간극(sim-to-real gap) 문제, 안전성 문제, 데이터 부족 등의 실제 로봇 문제를 해결하기 위한 방법론을 탐구합니다. Kuka LBR IIWA 14 로봇을 사용하여 공기 하키 디자인을 구현하였고, MuJoCo 시뮬레이터를 통해 제어 전략을 평가함으로써 다양한 환경 요인들을 반영했습니다.

- **Performance Highlights**: 학습 기반 접근 방식과 기존 지식을 결합한 솔루션이 데이터만 의존하는 방식보다 뛰어난 성능을 보였으며, 최상의 성과를 올린 에이전트들이 성공적인 실제 공기 하키 배치 사례를 수립했습니다.



### STARS: Sensor-agnostic Transformer Architecture for Remote Sensing (https://arxiv.org/abs/2411.05714)
- **What's New**: 본 논문에서는 센서에 독립적인 스펙트럼 변환기(Spectral Transformer)를 제안하며, 이를 통해 다양한 스펙트럼 기초 모델을 구축하는 기초로 삼고자 한다. 연구팀은 센서 메타데이터를 활용하여 어떤 스펙트럼 기기에서 얻은 스펙트럼을 공통 표현으로 인코딩하는 범용 스펙트라 표현(USR, Universal Spectral Representation)을 소개한다.

- **Technical Details**: 제안된 아키텍처는 세 가지 모듈로 구성된다: (i) 범용 스펙트라 표현(USR), (ii) 스펙트럼 변환기 인코더, (iii) 연산자 이론 기반 디코더. 이 방법들은 센서에 독립적인 모델 학습을 위해 강력한 데이터 증강 전략과 결합된다. 또한 자가 지도식 복원 작업을 통해 저해상도 스펙트럼을 높은 해상도 스펙트럼으로 복원하는 방법을 제안한다.

- **Performance Highlights**: 모델을 통해 훈련 과정에서 보지 못한 센서로부터 유도된 스펙트럼 서명(latent representation)을 비교함으로써, 제안한 방법이 다양한 스펙트럼 데이터에서 중요한 정보를 신뢰성 있게 추출할 수 있음을 보여준다. 또한 이러한 성능을 통해 스펙트럼 데이터의 다양성을 활용할 수 있는 기초 모델의 교육을 위한 새로운 방향을 제시한다.



### Visual-TCAV: Concept-based Attribution and Saliency Maps for Post-hoc Explainability in Image Classification (https://arxiv.org/abs/2411.05698)
Comments:
          Preprint currently under review

- **What's New**: 이 논문에서는 CNN (Convolutional Neural Networks)의 이미지 분류를 위한 새로운 설명 가능성 프레임워크인 Visual-TCAV를 소개합니다. 이 방법은 기존의 saliency 방법과 개념 기반 접근 방식을 통합하여, CNN의 예측에 대한 개념의 기여도를 숫자로 평가하는 동시에, 어떤 이미지에서 어떤 개념이 인식되었는지를 시각적으로 설명합니다.

- **Technical Details**: Visual-TCAV는 Concept Activation Vectors (CAVs)를 사용하여 네트워크가 개념을 인식하는 위치를 보여주는 saliency map을 생성합니다. 또한, Integrated Gradients의 일반화를 통해 이러한 개념이 특정 클래스 출력에 얼마나 중요한지를 추정할 수 있습니다. 이 프레임워크는 CNN 아키텍처에 적용 가능하며, 각 시각적 설명은 네트워크가 선택한 개념을 인식한 위치와 그 개념의 중요도를 포함합니다.

- **Performance Highlights**: Visual-TCAV는 기존의 TCAV와 비교하여 지역적(local) 및 글로벌(global) 설명 가능성을 모두 제공할 수 있으며, 실험을 통해 그 유효성이 확인되었습니다. 본 방법은 CNN 모델의 다양한 레이어에 적용할 수 있으며, 사용자 정의 개념을 통합하여 더 풍부한 설명을 제공합니다.



### IPMN Risk Assessment under Federated Learning Paradigm (https://arxiv.org/abs/2411.05697)
- **What's New**: 이번 연구에서 우리는 다기관 Intraductal Papillary Mucinous Neoplasms (IPMN) 분류를 위한 연합 학습(federated learning) 프레임워크를 개발했습니다. 이는 7개 의료 기관으로부터의 포괄적인 췌장 MRI 데이터 세트를 활용하여 이루어졌으며, 역사상 가장 크고 다양한 IPMN 분류 데이터 세트를 제공합니다.

- **Technical Details**: 이 연구에서는 하나의 통합된 췌장 MRI 데이터 세트를 사용하여 723개의 T1-weighted 및 739개의 T2-weighted MRI 이미지를 분석하였고, DenseNet-121을 3D convolutional model로 활용하였습니다. 각 기관은 로컬 데이터에 대해 독립적으로 3D DenseNet-121 모델을 훈련합니다.

- **Performance Highlights**: 연합 학습 방법이 중앙 집중적 학습과 비교했을 때 유사한 높은 분류 정확도를 달성하는 것을 보여주었으며, 여러 기관 간의 데이터 프라이버시를 보장했습니다. 이는 췌장 MRI에서 IPMN 분류의 협업에 있어 중요한 진전을 나타냅니다.



### Tell What You Hear From What You See -- Video to Audio Generation Through Tex (https://arxiv.org/abs/2411.05679)
Comments:
          NeurIPS 2024

- **What's New**: 본 연구에서는 비디오와 텍스트 프롬프트를 입력받아 오디오 및 텍스트 설명을 생성하는 다중 모달 생성 프레임워크인 VATT를 제안합니다. 이 모델은 비디오의 맥락을 보완하는 텍스트를 통해 오디오 생성 과정을 세밀하게 조정 가능하게 하며, 비디오에 대한 오디오 캡션을 생성하여 적절한 오디오를 제안할 수 있습니다.

- **Technical Details**: VATT에는 VATT Converter와 VATT Audio라는 두 가지 모듈이 포함되어 있습니다. VATT Converter는 미세 조정된 LLM(대형 언어 모델)으로, 비디오 특징을 LLM 벡터 공간에 매핑하는 프로젝션 레이어를 갖추고 있습니다. VATT Audio는 변환기로, 시각적 프레임 및 선택적 텍스트 프롬프트에서 오디오 토큰을 생성하는 병렬 디코딩을 통해 운영됩니다. 생성된 오디오 토큰은 사전 훈련된 뉴럴 코덱을 통해 웨이브폼으로 변환됩니다.

- **Performance Highlights**: 실험 결과에 따르면, VATT는 기존 방법 대비 경쟁력 있는 성능을 보이며, 오디오 캡션이 제공되지 않을 때에도 향상된 결과를 도출합니다. 오디오 캡션을 제공할 경우 KLD 점수가 1.41로 가장 낮은 성과를 기록했습니다. 주관적 연구에서는 VATT Audio에서 생성된 오디오가 기존 방법에 비해 높은 선호도를 나타냈습니다.



### Online-LoRA: Task-free Online Continual Learning via Low Rank Adaptation (https://arxiv.org/abs/2411.05663)
Comments:
          WACV 2025

- **What's New**: 이 논문에서는 작업 경계가 정의되지 않은 상황에서 온라인 지속 학습(Online Continual Learning, OCL)에서 발생하는 치명적인 망각(Catastrophic Forgetting) 문제를 해결하기 위해 'Online-LoRA'라는 새로운 프레임워크를 제안합니다.

- **Technical Details**: Online-LoRA는 사전 학습된 Vision Transformer (ViT) 모델을 실시간으로 미세 조정(finetuning)하여 리허설 버퍼의 한계와 사전 학습 모델의 성능 이점을 활용합니다. 이 접근법은 온라인 가중치 정규화 전략을 통해 중요한 모델 매개변수를 식별하고 통합하며, 손실 값의 훈련 동력을 활용하여 데이터 분포의 변화 인식을 자동화합니다.

- **Performance Highlights**: 다양한 작업 없는 OCL 시나리오와 기준 데이터 세트(CIFAR-100, ImageNet-R, ImageNet-S, CUB-200, CORe50)에서 진행된 광범위한 실험 결과, Online-LoRA는 다양한 ViT 아키텍처에 robust하게 적응할 수 있으며, 기존의 SOTA 방법들보다 더 나은 성능을 보임을 보여주었습니다.



### Multi-armed Bandits with Missing Outcom (https://arxiv.org/abs/2411.05661)
Comments:
          38 pages, 5 figures, multi-armed bandits, missing data

- **What's New**: 본 논문은 다중팔 밴딧(multi-armed bandit, MAB) 문제에서 누락된 결과를 다룰 수 있는 첫 번째 정식 연구로, 누락 메커니즘이 무작위가 아닌 경우(missing not at random, MNAR)와 무작위인 경우(missing at random, MAR)를 모두 분석합니다.

- **Technical Details**: 알고리즘은 누락의 메커니즘을 고려하여 보상 추정을 조정하며, 주어진 데이터와 누락 메커니즘에 기반하여 편향되지 않은 보상을 추정할 수 있도록 설계되었습니다. 이 연구는 MAB에서 누락된 결과의 영향에 대한 정량적인 분석을 제공하고, UCB(upper confidence bound) 알고리즘을 소개합니다.

- **Performance Highlights**: 누락을 고려했을 때 의사결정 과정에서 놀라운 개선이 이뤄짐을 분석적 연구 및 시뮬레이션을 통해 입증하였습니다. 본 논문에서 제안된 알고리즘은 실질적인 적용 가능성을 가지며, 의료, 교육 및 전자상거래 등의 여러 도메인에서 더 나은 의사결정으로 이어질 것입니다.



### Video RWKV:Video Action Recognition Based RWKV (https://arxiv.org/abs/2411.05636)
- **What's New**: 본 논문에서는 기존의 CNN 및 Transformer 기반 비디오 이해 방법의 높은 계산 비용과 장거리 의존성 문제를 해결하기 위해 LSTM CrossRWKV (LCR) 프레임워크를 제안합니다. LCR은 비디오 이해 작업을 위한 공간-시간(spatiotemporal) 표현 학습을 위해 설계되었으며, 새로운 Cross RWKV 게이트를 통해 현재 프레임의 엣지 정보와 과거 특징 간의 상호작용을 원활히 하여 주제를 보다 집중적으로 처리합니다.

- **Technical Details**: LCR은 선형 복잡성을 가지며, 현재 프레임의 엣지 특징과 과거 특징을 결합하여 동적 공간-시간 컨텍스트 모델링을 제공하는 Cross RWKV 게이트를 포함합니다. LCR은 향상된 LSTM 반복 실행 메커니즘을 통해 비디오 처리를 위한 장기 기억을 저장하며, 엣지 정보는 LSTM의 잊기 게이트 역할을 하여 장기 기억을 안내합니다.

- **Performance Highlights**: LSTM-CrossRWKV는 Kinetics-400, Sometingsometing-V2, Jester의 세 가지 데이터 세트에서 뛰어난 성능을 발휘하며, 비디오 이해를 위한 새로운 기준을 제공합니다. 이 모델의 코드는 공개되어 있어 누구나 활용할 수 있습니다.



### Physics-constrained coupled neural differential equations for one dimensional blood flow modeling (https://arxiv.org/abs/2411.05631)
- **What's New**: 본 연구는 기존의 1D 혈류 모델의 정확성을 향상시키는 새로운 물리 기반 머신 러닝 기법을 도입하였습니다. 이 방법은 3D 평균 데이터로 학습하여 전통적인 1D 접근 방식보다 더 높은 정확도를 달성합니다.

- **Technical Details**: 제안된 PCNDE(Physics-Constrained Neural Differential Equation) 프레임워크는 전통적인 시간 접근 방식을 벗어나 운동량 보존 방정식의 공간 형태를 갖추고 있는 혁신적인 모델입니다. 이 모델은 경계 조건 구현을 단순화하면서도 결합 안정성과 매끄러움을 극복합니다.

- **Performance Highlights**: 우리의 모델은 다양한 유입 경계 조건 파형 및 협착 비율에서 기존의 FEM 기반 1D 모델과 비교하여 우수한 성능을 보여주었습니다. 이는 흐름 속도, 면적 및 압력 변화를 정확하게 포착하는 능력을 갖추고 있습니다.



### Cross-validating causal discovery via Leave-One-Variable-Ou (https://arxiv.org/abs/2411.05625)
- **What's New**: 이 논문에서는 causal discovery 알고리즘을 ground truth 없이 검증하는 새로운 접근 방식인 "Leave-One-Variable-Out (LOVO)" 예측 방법을 제안합니다. 이는 causal 모델 학습 시 포함되지 않은 변수 쌍을 테스트하여 인과 관계를 평가합니다.

- **Technical Details**: LOVO 예측에서는 $Y$가 $X$로부터 추론되며, 이는 $X,Z_1,...,Z_k$ 데이터만 있고 $X$와 $Y$의 공동 관측이 없는 상황에서 이루어집니다. 여기에서 Acyclic Directed Mixed Graphs (ADMGs)의 형태로 두 개의 데이터 하위 집합에서 인과 모델이 적합합니다. 이 방법은 특정 선험적 가정에 의존하는 알고리즘을 위해 LOVO 예측기를 구성하는 방식도 포함됩니다.

- **Performance Highlights**: 시뮬레이션 결과, LOVO 예측 오차는 인과 출력의 정확성과 상관관계가 있는 것으로 나타났습니다. 논문은 인과 관계의 효율성을 확인하는 방법으로 LOVO 접근법의 유용성을 강조합니다.



### A Two-Step Concept-Based Approach for Enhanced Interpretability and Trust in Skin Lesion Diagnosis (https://arxiv.org/abs/2411.05609)
Comments:
          Preprint submitted for review

- **What's New**: 본 연구에서는 기존 Concept Bottleneck Models (CBMs)의 데이터 주석 부담 및 해석 가능성 부족 문제를 해결하기 위해 새로운 두 단계 접근 방식을 제안합니다. 이 방법은 pretrained Vision Language Model (VLM)과 Large Language Model (LLM)을 활용하여 임상 개념을 자동으로 예측하고 질병 진단을 생성합니다.

- **Technical Details**: 첫 번째 단계에서는 pretrained VLM을 사용하여 임상 개념의 존재를 예측합니다. 두 번째 단계에서는 예측된 개념을 맞춤형 프롬프트에 통합하여 LLM에 질병 진단을 요청합니다. 이 과정은 추가 훈련 없이 이루어지며, 새로운 개념의 추가 시 재훈련이 필요하지 않습니다.

- **Performance Highlights**: 이 방법은 세 가지 피부 병변 데이터셋에서 전통적인 CBMs 및 최첨단 설명 가능한 방법들보다 우수한 성능을 보였습니다. 추가 훈련 없이 몇 개의 주석된 예시만으로 최종 진단 클래스를 제공하여 해석 가능성과 성능을 동시에 개선하였습니다.



### Predicting Stroke through Retinal Graphs and Multimodal Self-supervised Learning (https://arxiv.org/abs/2411.05597)
Comments:
          Accepted as oral paper at ML-CDS workshop, MICCAI 2024

- **What's New**: 본 연구는 효율적인 망막 이미지 표현과 임상 정보를 결합하여 심혈관 건강에 대한 포괄적인 개요를 포착하는 새로운 접근 방식을 제안합니다. 이는 망막 이미지에서 유도된 혈관 그래프를 사용하여 대규모 다중 모드 데이터를 활용한 신뢰할 수 있는 뇌졸중 예측 모델을 개발하는 것을 포함합니다.

- **Technical Details**: 제안된 프레임워크는 여러 모드의 데이터를 통합하는 대조적 학습(contrastive learning) 방법을 기반으로 하며, 망막 사진(fundus photographs)의 세 가지 모듈을 통해 특징 임베딩을 추출합니다. 이미지 데이터에는 ResNet을 기반으로 한 모듈과 혈관 그래프 표현을 통한 그래프 신경망(Graph Neural Networks, GNNs)이 사용됩니다.

- **Performance Highlights**: 본 프레임워크는 감독 학습(supervised learning)에서 자가 감독 학습(self-supervised learning)으로 전환되는 과정에서 AUROC 점수가 3.78% 향상되었으며, 그래프 수준의 표현 방식이 이미지 인코더(image encoders)보다 우수한 성능을 보여줍니다. 이 결과는 망막 이미지를 활용한 심혈관 질환 예측의 비용 효율적인 개선을 입증합니다.



### Network EM Algorithm for Gaussian Mixture Model in Decentralized Federated Learning (https://arxiv.org/abs/2411.05591)
- **What's New**: 본 연구에서는 분산된 연합 학습(decentralized federated learning) 프레임워크 내에서 Gaussian 혼합 모델(Gaussian mixture model)을 위한 다양한 네트워크 Expectation-Maximization (EM) 알고리즘을 체계적으로 연구합니다. 기존 방법의 한계를 극복하기 위한 두 가지 새로운 해결책을 제안합니다.

- **Technical Details**: 이 연구에서는 동질적이지 않은 고객 간 데이터 특성을 다루기 위하여 모멘텀 네트워크 EM (MNEM) 알고리즘을 도입하고, 나눠져 있지 않은 Gaussian 구성 요소 문제를 해결하기 위해 반지도 학습 MNEM(semi-MNEM) 알고리즘을 개발합니다. MNEM은 현재 및 역사적 추정치를 결합하는 모멘텀 매개변수를 활용합니다.

- **Performance Highlights**: 엄격한 이론적 분석을 통해 MNEM 알고리즘이 특정 분리 조건을 만족하는 혼합 구성 요소에서 전체 샘플 추정량과 유사한 통계적 효율성을 달성할 수 있음을 보였습니다. 또한, semi-MNEM 추정량은 MNEM 알고리즘의 수렴 속도를 개선하여 분리되지 않은 시나리오에서의 수치적 수렴 문제를 효과적으로 해결합니다.



### Towards a Real-Time Simulation of Elastoplastic Deformation Using Multi-Task Neural Networks (https://arxiv.org/abs/2411.05575)
- **What's New**: 이 연구는 Proper Orthogonal Decomposition, Long Short-Term Memory Networks, 그리고 Multi-Task Learning을 융합하여 실시간으로 엘라스토플라스틱 변형을 정확하게 예측하는 Surrogate Modeling Framework를 제안합니다. 이 접근 방식은 단일 작업 신경망보다 뛰어난 성능을 보이며, 여러 상태 변수를 통해 평균 절대 오차가 0.40% 이하로 감소합니다.

- **Technical Details**: 제안된 모델은 Multi-Task Learning(다중 작업 학습)을 통해 여러 관련 작업을 동시에 훈련하여 예측 정확성을 높이며, 공유 레이어를 통해 과적합을 완화합니다. 이 프레임워크는 전통적인 유한 요소 방법(Finite Element Method)과 기계 학습(Machine Learning)을 통합하여 복잡한 재료 거동 모델링을 지원합니다.

- **Performance Highlights**: 제안된 모델은 전통적인 유한 요소 분석보다 약 100만 배 빠르며, 추가 변수를 훈련하는 데에 단 20개의 샘플만으로도 효과적으로 진행할 수 있습니다. 이는 일반적으로 약 100개의 샘플이 필요한 단일 작업 모델에 비해 매우 효율적입니다.



### Open-set object detection: towards unified problem formulation and benchmarking (https://arxiv.org/abs/2411.05564)
Comments:
          Accepted at ECCV 2024 Workshop: "The 3rd Workshop for Out-of-Distribution Generalization in Computer Vision Foundation Models"

- **What's New**: 이 연구에서는 OpenSet Object Detection(OSOD) 관련 여러 접근 방식의 평가를 통합하기 위한 새로운 벤치마크(OpenImagesRoad)를 소개하고, 기존의 편향된 데이터 세트와 평가 지표를 해결하는 데 초점을 맞추고 있습니다. 이를 통해 OSOD에 대한 명확한 문제 정의와 일관된 평가가 가능해집니다.

- **Technical Details**: 연구에서는 다양한 unknown object detection 접근 방식을 논의하며, VOC와 COCO 데이터 세트를 기반으로 하는 통합된 평가 방법론을 제안합니다. 새로운 OpenImagesRoad 벤치마크는 명확한 계층적 객체 정의와 새로운 평가 지표를 제공합니다. 또한 최근 자가 감독 방식(self-supervised)으로 학습된 Vision Transformers(DINOv2)를 활용하여 pseudo-labeling 기반의 OSOD를 개선하는 OW-DETR++ 모델을 제안합니다.

- **Performance Highlights**: 본 연구에서 제안한 벤치마크에서 state-of-the-art(OSOT) 방법들의 성능을 광범위하게 평가하였으며, OW-DETR++ 모델은 기존 pseudo-labeling 방법들 중에서 가장 우수한 성능을 기록했습니다. 이를 통해 OSOD 전략의 효과 및 경계에 대한 새로운 통찰을 제공합니다.



### Training objective drives the consistency of representational similarity across datasets (https://arxiv.org/abs/2411.05561)
Comments:
          26 pages

- **What's New**: 최근의 기초 모델들이 다운스트림 작업 성능에 따라 공유 표현 공간으로 수렴하고 있다는 새로운 가설인 Platonic Representation Hypothesis를 제시합니다. 이는 데이터 모달리티와 훈련 목표와는 무관하게 발표됩니다.

- **Technical Details**: 연구에서는 CKA와 RSA와 같은 유사도 측정 방법을 사용하여 모델 표현의 일관성을 측정하는 체계적인 방법을 제안했습니다. 실험 결과, 자기 지도 시각 모델이 이미지 분류 모델이나 이미지-텍스트 모델보다 다른 데이터셋에서의 쌍별 유사도를 더 잘 일반화하는 것으로 나타났습니다. 또한, 모델의 과제 행동과 표현 유사성 간의 관계는 데이터셋 의존적이라 밝혀졌습니다.

- **Performance Highlights**: 이 연구를 통해 쌍별 표현 유사도가 모델의 작업 성능 차이와 강한 상관관계를 보임을 발견하였으며, 단일 도메인 데이터셋에서 이러한 경향이 가장 뚜렷하게 나타났습니다.



### Towards Lifelong Few-Shot Customization of Text-to-Image Diffusion (https://arxiv.org/abs/2411.05544)
- **What's New**: 이번 연구는 Lifelong Few-Shot Diffusion (LFS-Diffusion) 방법을 제안하여 텍스트-이미지 확산 모델을 지속적으로 조정하면서 이전 지식을 보존할 수 있도록 합니다.

- **Technical Details**: 우리는 Relevant Concepts Forgetting (RCF)와 Previous Concepts Forgetting (PCF) 문제를 파악하였고, 이를 해결하기 위해 데이터 없는 knowledge distillation 전략과 In-Context Generation (ICGen) 패러다임을 개발하였습니다. ICGen은 입력 비전 컨텍스트에 기반하여 모델의 성능을 향상시키는 방법입니다.

- **Performance Highlights**: 제안된 방법은 CustomConcept101 및 DreamBooth 데이터셋에서 기존 방법보다 뛰어난 성능을 보여주며, 고품질의 이미지를 생성할 수 있음을 입증하였습니다.



### FGGP: Fixed-Rate Gradient-First Gradual Pruning (https://arxiv.org/abs/2411.05500)
- **What's New**: 최근 딥러닝 모델의 크기가 증가하고 계산 자원의 수요가 커짐에 따라, 정확도를 유지하면서 신경망을 가지치기(pruning)하는 방법에 대한 관심이 높아지고 있습니다. 본 연구에서는 그라디언트 우선의 크기 선택 전략을 소개하고, 이러한 접근법이 고정비율(subselection criterion) 기준을 통해 높은 성과를 낸다는 것을 보여줍니다.

- **Technical Details**: 이 연구에서는 CIFAR-10 데이터셋을 사용하여 VGG-19와 ResNet-50 네트워크를 기초로 하여 90%, 95%, 98%의 희소성(sparsity)에 대한 가지치기를 수행하였습니다. 제안한 고정비율 그라디언트 우선 가지치기(FGGP) 접근법은 대부분의 실험 설정에서 기존의 최첨단 기술보다 우수한 성능을 보였습니다.

- **Performance Highlights**: FGGP는 타 기술 대비 높은 순위를 기록하며, 경우에 따라 밀집 네트워크의 결과 상한선을 초과하는 성과를 달성하였습니다. 이는 신경망의 파라미터를 가지치기하는 방식에서 단계별 선택 프로세스의 질이 결과에 얼마나 중요한지를 입증합니다.



### Handling geometrical variability in nonlinear reduced order modeling through Continuous Geometry-Aware DL-ROMs (https://arxiv.org/abs/2411.05486)
Comments:
          30 pages, 15 figures

- **What's New**: 본 연구에서 우리는 기하학적 변동성 및 매개변수화된 영역을 다루기 위해 고안된 Continuous Geometry-Aware DL-ROMs (CGA-DL-ROMs)라는 새로운 구조를 제안합니다. 이 아키텍처는 기하학적으로 매개변수화된 문제에 적합한 공간 연속적 특성을 가지고 있습니다.

- **Technical Details**: CGA-DL-ROMs는 기하학적 매개변수화를 인식할 수 있는 강력한 귀납적 편향(inductive bias)을 가지고 있으며, 이는 압축 능력과 전반적인 성능을 향상시킵니다. 이 아키텍처는 다양한 해상도 데이터셋을 처리하는 데 적합한 무한 차원(space-continuous) 구조를 특징으로 합니다.

- **Performance Highlights**: 우리는 CGA-DL-ROMs의 성능을 이론적 분석과 함께 여러 물리적 및 기하학적으로 매개변수화된 PDE에 대한 수치 테스트를 통해 검증하였습니다. 시험 결과는 이 아키텍처가 기존 방법에 비해 놀라울 정도로 정확한 근사를 제공함을 보여주었습니다.



### Comparative Study of Probabilistic Atlas and Deep Learning Approaches for Automatic Brain Tissue Segmentation from MRI Using N4 Bias Field Correction and Anisotropic Diffusion Pre-processing Techniques (https://arxiv.org/abs/2411.05456)
- **What's New**: 이 논문은 MRI 이미지에서 뇌 조직을 자동으로 분할하는 데 대한 최신 연구를 제공합니다. 특히, 전통적인 통계적 방법과 현대의 딥러닝 접근법 간의 비교 분석을 수행하고 N4 Bias Field Correction 및 Anisotropic Diffusion과 같은 전처리 기법을 적용한 다양한 세분화 모델의 성능을 조사합니다.

- **Technical Details**: 연구에서는 Probabilistic ATLAS, U-Net, nnU-Net, LinkNet과 같은 다양한 세분화 모델을 사용하여 IBSR18 데이터셋에서 하얀질 (White Matter, WM), 회색질 (Gray Matter, GM), 및 뇌척수액 (Cerebrospinal Fluid, CSF)을 분할했습니다. nnU-Net 모델이 평균 Dice Coefficient (0.937 ± 0.012)에서 가장 뛰어난 성능을 보였으며, 2D nnU-Net 모델은 평균 Hausdorff Distance (5.005 ± 0.343 mm) 및 평균 Absolute Volumetric Difference (3.695 ± 2.931 mm)에서 가장 낮은 점수를 기록했습니다.

- **Performance Highlights**: 본 연구는 N4 Bias Field Correction 및 Anisotropic Diffusion 전처리 기법과 결합된 nnU-Net 모델의 우수성을 강조합니다. 이 모델은 MRI 데이터에서 뇌 조직의 정확한 분할에 있어 뛰어난 성과를 보여주며, GitHub를 통해 구현된 코드를 공개하였습니다.



### The sampling complexity of learning invertible residual neural networks (https://arxiv.org/abs/2411.05453)
- **What's New**: 이 논문에서는 피드포워드 ReLU 신경망이 점 샘플(point samples)으로부터 높은 균일 정확도를 확보하는 데 있어 차원의 저주(curse of dimensionality) 문제에 직면해 있음을 보여줍니다. 이를 기반으로 특정 신경망 아키텍처를 제한함으로써 샘플링 복잡성을 개선할 수 있는지를 연구하였습니다.

- **Technical Details**: 연구의 주된 내용은 가역 잔차 신경망(invertible residual neural networks) 아키텍처가 피드포워드 아키텍처 대비 샘플의 수요를 줄이지 못함을 증명하였다. 구체적으로, 가역 잔차 신경망과 가역 합성곱 잔차 신경망(invertible convolutional residual neural networks)의 uniform norm에 따른 근사화 시 필요한 계산 복잡성이 차원의 저주에 영향을 받음을 밝혔다. 

- **Performance Highlights**: 결과적으로 잔차 신경망 아키텍처가 복잡성 장벽을 극복하지 못하며, 샘플 수가 입력 차원에 대해 기하급수적으로 증가하는 특성을 보여줍니다. 이는 실질적인 훈련 방법에도 적용되며, 정확한 학습을 위한 추가적인 정규화 기법과 신경망 아키텍처 개발이 필요함을 시사합니다.



### ICE-T: A Multi-Faceted Concept for Teaching Machine Learning (https://arxiv.org/abs/2411.05424)
Comments:
          Accepted and presented at the 17th International Conference on Informatics in Schools (ISSEP 2024)

- **What's New**: 이번 논문은 인공지능(AI)과 머신 러닝(ML) 교육을 위해 다양한 플랫폼, 도구 및 게임을 활용하는 새로운 접근법을 소개합니다. 특히, ML을 교수할 때의 교육적 원칙과 이를 토대로 한 ICE-T라는 새로운 개념에 대해 설명합니다.

- **Technical Details**: ICE-T는 Intermodal transfer, Computational thinking, Explanatory thinking의 세 가지 요소로 구성된 다면적(멀티-팩터) 개념입니다. 이 개념은 교육자들이 ML 교육을 향상시키기 위해 사용할 수 있는 구조화된 접근 방식을 제공합니다. 또한 기존의 교수 도구들에서 이 요소들이 어떻게 구현되고 있는지를 평가합니다.

- **Performance Highlights**: 디지털 게임 기반 학습이 학생들의 동기 부여와 참여를 높이고, 인지 및 정서적 발달을 촉진하여 학습 효율성을 향상시킨다는 이전의 연구들을 언급하며, 학습 플랫폼의 부족한 부분을 보완하는 새로운 관점을 제안합니다.



### Ev2R: Evaluating Evidence Retrieval in Automated Fact-Checking (https://arxiv.org/abs/2411.05375)
Comments:
          10 pages

- **What's New**: 본 논문에서는 Automated Fact-Checking (AFC)을 위한 새로운 평가 프레임워크인 Ev2R을 소개합니다. Ev2R은 증거 평가를 위한 세 가지 접근 방식인 reference-based, proxy-reference, reference-less 평가 방식으로 구성되어 있습니다.

- **Technical Details**: Ev2R 프레임워크의 세 가지 평가자 그룹은 (1) reference-based 평가자: 참조 증거와 비교하여 검색된 증거를 평가, (2) proxy-reference 평가자: 시스템이 예측한 평결을 기준으로 증거를 평가, (3) reference-less 평가자: 입력 클레임만을 기반으로 증거를 평가하는 방식입니다. 구체적으로 LLMs를 활용하여 증거를 원자적 사실로 분해한 후 평가합니다.

- **Performance Highlights**: Ev2R의 reference-based 평가자는 전통적인 메트릭보다 인간 평가와 높은 상관관계를 보였습니다. Gemini 기반 평가자는 검색된 증거가 참조 증거를 얼마나 포함하고 있는지를 잘 평가한 반면, GPT-4 기반 평가자는 평결 일치에서 더 좋은 성능을 보였습니다. 이 결과는 Ev2R 프레임워크가 더욱 정확하고 강력한 증거 평가를 가능하게 함을 시사합니다.



### A Quality-Centric Framework for Generic Deepfake Detection (https://arxiv.org/abs/2411.05335)
- **What's New**: 본 논문은 딥페이크 탐지에서의 일반화 문제를 해결하기 위해 훈련 데이터의 위조 품질을 활용합니다. 다양한 위조 품질의 딥페이크가 혼합된 데이터로 탐지기를 교육하면 탐지기의 일반화 성능이 저하될 수 있다는 점을 지적합니다. 이를 해결하기 위해 새로운 품질 중심의 프레임워크를 제안하고, 이를 통해 저품질 데이터를 증강하는 방법과 학습 속도를 조절하는 전략을 구현합니다.

- **Technical Details**: 제안하는 프레임워크는 품질 평가기(Quality Evaluator), 저품질 데이터 증강 모듈, 그리고 학습 속도 조절 전략으로 구성됩니다. 위조 품질 점수(Forgery Quality Score, FQS)를 정적으로(예: ArcFace를 사용) 그리고 동적으로(모델의 피드백 사용) 평가하여 얻어진 FQS를 기반으로 훈련 샘플을 선택합니다. 저품질 샘플에 대해서는 주파수 데이터 증강(Frequency Data Augmentation, FreDA) 기법을 적용하여 위조 흔적을 줄이고 현실감을 개선하는 방법을 활용합니다.

- **Performance Highlights**: 실험 결과, 제안하는 방법은 다양한 딥페이크 탐지기의 범위 내 및 교차 데이터셋 성능을 크게 향상시킬 수 있음이 입증되었습니다. 여러 유명한 평가 데이터셋에서 기존 방법보다 약 10% 성능 개선을 달성하며, 손쉬운 샘플부터 어려운 샘플로 단계적으로 학습하도록 모델을 유도하는 커리큘럼 학습(curriculum learning)을 효과적으로 적용하였습니다.



### Differentiable Calibration of Inexact Stochastic Simulation Models via Kernel Score Minimization (https://arxiv.org/abs/2411.05315)
Comments:
          31 pages, 12 tables, 4 figures

- **What's New**: 본 연구에서는 stochastic simulation 모델의 입력 파라미터를 output-level 데이터만을 사용하여 학습하는 혁신적인 방법을 제안합니다. 이는 kernel score minimization과 stochastic gradient descent를 활용하여 이루어집니다.

- **Technical Details**: 제안된 방법은 새로운 한계 정리(asymptotic normality)를 기반으로 하고, 모델 부정확성(model inexactness)을 고려하여 도출됩니다. 구체적으로, multi-dimensional output-level 데이터 X1,…,Xm을 활용하여 simulation parameter θ를 추정한 후, Frequentist confidence set 절차를 통해 입력 파라미터의 불확실성을 정량화합니다.

- **Performance Highlights**: 이 방법은 G/G/1 queueing 모델에 대한 실험을 통해 평가되었으며, 기존의 방법들이 해결하지 못하는 input 파라미터의 불확실성을 효과적으로 학습하고 정량화할 수 있음을 보여줍니다.



### MicroScopiQ: Accelerating Foundational Models through Outlier-Aware Microscaling Quantization (https://arxiv.org/abs/2411.05282)
Comments:
          Under review

- **What's New**: 본 논문에서는 MicroScopiQ라는 새로운 공동 설계 기법을 제안합니다. 이는 pruning과 outlier-aware quantization을 결합하여 높은 정확도와 메모리 및 하드웨어 효율성을 달성하는 방안을 제시합니다.

- **Technical Details**: MicroScopiQ는 outlier를 더 높은 정밀도로 유지하면서 중요하지 않은 일부 가중치를 pruning합니다. 새로운 NoC(네트워크 온 칩) 아키텍처인 ReCoN을 통해 하드웨어의 복잡성을 효율적으로 추상화하고, 다양한 비트 정밀도를 지원하는 가속기 아키텍처를 설계합니다. MicroScaling 데이터 형식을 활용하여 아웃라이어 가중치를 정량화하는 방식도 포함되어 있습니다.

- **Performance Highlights**: MicroScopiQ는 다양한 정량화 설정에서 기존 기법 대비 3배 향상된 추론 성능을 달성하고, 에너지를 2배 절감하여 SoTA(SOTA: State of the Art) 정량화 성능을 달성합니다.



### Fox-1 Technical Repor (https://arxiv.org/abs/2411.05281)
Comments:
          Base model is available at this https URL and the instruction-tuned version is available at this https URL

- **What's New**: Fox-1은 새로운 훈련 커리큘럼 모델링을 도입한 소형 언어 모델(SLM) 시리즈로, 3조 개의 토큰과 5억 개의 지침 데이터로 사전 훈련 및 미세 조정되었습니다. 이 모델은 그룹화된 쿼리 어텐션(Grouped Query Attention, GQA)과 깊은 레이어 구조를 가지고 있어 성능과 효율성을 향상시킵니다.

- **Technical Details**: Fox-1-1.6B 모델은 훈련 데이터를 3단계 커리큘럼으로 구분하여 진행하며, 2K-8K 시퀀스 길이를 처리합니다. 데이터는 오픈 소스의 다양한 출처에서 수집된 고품질 데이터를 포함하고 있습니다. 이 모델은 텐서 오페라(TensorOpera) AI 플랫폼과 훌리 지픈(Facing)에서 공개되며, Apache 2.0 라이센스 하에 이용 가능합니다.

- **Performance Highlights**: Fox-1은 StableLM-2-1.6B, Gemma-2B와 같은 여러 벤치마크에서 경쟁력 있는 성능을 보여주며 빠른 추론 속도와 처리량을 자랑합니다.



### Revisiting the Robustness of Watermarking to Paraphrasing Attacks (https://arxiv.org/abs/2411.05277)
Comments:
          EMNLP 2024

- **What's New**: 본 연구는 언어 모델(LM)에서 생성된 텍스트의 감지를 위한 수조 방지 기법의 효과성을 강조합니다. 기존의 수조 방지 기법들은 높은 강인성을 주장하지만, 실제로는 역설계(reverse-engineering)가 용이하다는 점을 보여줍니다.

- **Technical Details**: 연구에서는 언어 모델의 출력에서 수조 신호를 포함시키는 방법을 논의하며, 이를 임의로 선택된 토큰 집합(그린 리스트)으로 부각시키는 기법을 설명합니다. 이 연구는 특정 수의 생성 결과만으로도 파라프레이징 공격(paraphrasing attacks)의 효과를 극대화할 수 있음을 나타냅니다.

- **Performance Highlights**: 200K개의 수조된 토큰을 사용할 경우, 그린 리스트의 정확도를 0.8 이상의 F1 점수를 기록할 수 있으며, 이를 통해 파라프레이징을 통해 수조 탐지율을 10% 이하로 떨어트릴 수 있습니다. 이러한 결과는 수조 방지 알고리즘의 강인성에 대한 우려를 제기합니다.



### Real-World Offline Reinforcement Learning from Vision Language Model Feedback (https://arxiv.org/abs/2411.05273)
Comments:
          7 pages. Accepted at the LangRob Workshop 2024 @ CoRL, 2024

- **What's New**: 새로운 시스템을 제안하여, Vision-Language Model (VLM)의 선호 피드백을 사용하여 오프라인 데이터셋의 보상 레이블을 자동으로 생성할 수 있도록 했습니다. 이 방법을 통해 라벨이 없는 서브 최적 오프라인 데이터셋에서도 효과적으로 정책(policy)을 학습할 수 있게 됩니다.

- **Technical Details**: 정확한 보상 레이블을 생성하기 위해, Vision-Language Model (VLM) 피드백을 활용한 기존 연구인 RL-VLM-F를 기반으로 하여, 주어진 오프라인 데이터셋으로부터 선호 데이터셋을 생성합니다. 이후 생성된 데이터셋을 통해 보상 함수를 학습하고, 이를 이용하여 주어진 데이터셋에 레이블을 붙입니다. 이렇게 레이블이 붙은 데이터셋은 기존의 오프라인 강화 학습 프레임워크를 통해 제어 정책을 학습하는 데 활용됩니다.

- **Performance Highlights**: 제안한 시스템은 복잡한 현실 세계의 로봇 보조 의상 착용 작업에 적용되었으며, 비최적 오프라인 데이터셋에서 효과적인 보상 함수와 정책을 학습하여 기존의 Behavior Cloning (BC) 및 Inverse Reinforcement Learning (IRL) 방법들보다 우수한 성능을 보였습니다.



### Cancer-Net SCa-Synth: An Open Access Synthetically Generated 2D Skin Lesion Dataset for Skin Cancer Classification (https://arxiv.org/abs/2411.05269)
- **What's New**: 미국에서 피부암은 가장 흔하게 진단되는 암으로, 조기에 발견하지 않으면 심각한 합병증의 위험이 있어 주요 공공 건강 문제로 인식되고 있습니다. 최신 연구는 피부암 분류를 위한 합성 데이터셋 Cancer-Net SCa-Synth를 공개하며, 이는 Stable Diffusion과 DreamBooth 기술을 활용하여 제작되었습니다.

- **Technical Details**: Cancer-Net SCa-Synth는 피부암 분류를 위한 2D 피부 병변 데이터셋으로, 총 10,000개의 이미지를 포함하고 있습니다. 이 데이터셋은 ISIC 2020 테스트 세트에 대한 학습을 통해 얻은 성능 향상을 증명합니다. Stable Diffusion 모델과 DreamBooth 트레이너를 이용해 두 가지 피부암 클래스에 대해 각각 학습이 이루어졌으며, 여기에서 합성 이미지는 300개의 무작위 샘플을 기반으로 생성되었습니다.

- **Performance Highlights**: Cancer-Net SCa-Synth와 ISIC 2020 훈련 세트를 결합하여 훈련했을 때, 공개 점수에서 0.09, 개인 점수에서 0.04 이상의 성능 향상이 발견되었습니다. 이는 단독으로만 사용된 데이터셋들로부터 훈련한 경우보다 성능이 향상된 결과입니다.



### Decoding Report Generators: A Cyclic Vision-Language Adapter for Counterfactual Explanations (https://arxiv.org/abs/2411.05261)
- **What's New**: 보고서 생성 모델에서 생성된 텍스트의 해석 가능성을 향상시키기 위한 혁신적인 접근 방식이 소개되었다. 이 방법은 사이클 텍스트 조작(cyclic text manipulation)과 시각적 비교(visual comparison)를 활용하여 원본 콘텐츠의 특징을 식별하고 설명한다.

- **Technical Details**: 새롭게 제안된 접근 방식은 Cyclic Vision-Language Adapters (CVLA)를 활용하여 설명의 생성 및 이미지 편집을 수행한다. counterfactual explanations를 통해 원본 이미지와 대조하여 수정된 이미지와의 비교를 통해 해석을 제공하며, 이는 모델에 구애받지 않는 방식으로 이루어진다.

- **Performance Highlights**: 이 연구의 방법론은 다양한 현재 보고서 생성 모델에서 적용 가능하며, 생성된 보고서의 신뢰성을 평가하는 데 기여할 것으로 기대된다.



### Evaluating GPT-4 at Grading Handwritten Solutions in Math Exams (https://arxiv.org/abs/2411.05231)
- **What's New**: 최근 generative artificial intelligence (AI)의 발전은 개방형 학생 응답을 정확하게 점수 매기는 데 가능성을 보여주었습니다. 이 연구에서는 특히 GPT-4o와 같은 최신 multi-modal AI 모델을 활용하여 대학 수준의 수학 시험에 대한 손글씨 응답을 자동으로 평가하는 방법을 제안합니다.

- **Technical Details**: 이 연구에서는 OpenAI의 GPT-4o 모델을 사용하여 실제 학생의 손글씨 시험 응답을 평가합니다. 평가 방법에는 3가지 prompting 방식이 포함되어 있으며(N - 문맥 없음, C - 정답 제공, CR - 정답과 루브릭 제공), 다양한 매트릭스를 통해 AI 모델의 점수 예측과 인간 채점자의 실제 점수를 비교합니다.

- **Performance Highlights**: GPT-4o 모델은 루브릭을 제공할 때 전반적인 점수 일치도가 개선되지만, 여전히 정확도는 너무 낮은 수준으로 실질적인 적용이 어렵습니다. 특히, 학생의 서술 방식 이해에 어려움이 있으며, 여러 요인으로 인해 점수 부여에서 많은 문제가 발생하고 있습니다.



### Generalizable Single-Source Cross-modality Medical Image Segmentation via Invariant Causal Mechanisms (https://arxiv.org/abs/2411.05223)
Comments:
          WACV 2025

- **What's New**: 이 논문에서는 단일 소스 도메인에서 학습한 모델이 보이지 않는 타겟 도메인에서 잘 일반화될 수 있도록 하는 단일 소스 도메인 일반화(SDG)를 제안하며, 의학 이미지 분할의 교차 모달리티 상황에 맞춘 방법론을 소개합니다.

- **Technical Details**: 본 연구에서는 직관적으로 다양한 이미징 스타일을 시뮬레이션하기 위해 통제된 diffusion models(DMs)를 활용하는 방법을 제안하며, 'intervention-augmentation equivariant' 원칙에 기초하여 다차원 스타일 변수를 포괄적으로 변화시킵니다. 이를 통해 의료 이미지 분할에서 교차 모달리티 문제를 해결하는 접근 방식을 제시합니다.

- **Performance Highlights**: 본 논문에서 제안하는 접근법은 세 가지 다른 해부학적 구조와 이미징 모달리티를 테스트한 결과, 기존의 SDG 방법들보다 일관되게 우수한 성능을 발휘하며, 이는 의료 영상 기술에 실질적인 기여를 할 것으로 기대됩니다.



### Don't Look Twice: Faster Video Transformers with Run-Length Tokenization (https://arxiv.org/abs/2411.05222)
Comments:
          16 pages, 6 figures. Accepted to NeurIPS 2024 (spotlight)

- **What's New**: 이번 논문에서는 비디오 트랜스포머의 처리 속도를 높이는 새로운 방법인 Run-Length Tokenization (RLT)을 제안합니다. RLT는 반복적으로 나타나는 패치를 효율적으로 찾아 제거하고, 이를 단일 패치와 위치 인코딩으로 대체하는 방식으로 동작합니다. 이는 기존의 방법들이 가진 조정 필요성이나 성능 손실을 피하면서, 토큰 수를 줄이는 혁신적인 접근입니다.

- **Technical Details**: RLT는 비디오의 스패치(Spatiotemporal) 패치를 표기하는 기존 방식 대신, 시간에 따라 반복되는 패치의 연속(run)을 찾아 이를 제거한 후, 남은 토큰의 변동 길이를 나타내는 정보를 추가합니다. 이는 데이터 압축을 위한 Run-Length Encoding에서 영감을 받았으며, 훈련 없이도 모델의 처리량을 35% 증가시키고, 정확도는 0.1% 떨어뜨리는 수준입니다.

- **Performance Highlights**: RLT를 사용하면 비디오 트랜스포머의 훈련 시간을 30% 단축시키면서도 성능은 기준 모델과 동일하게 유지됩니다. 또한, 훈련 시 30 FPS에서 100% 이상의 속도 향상을 이뤄내고, 긴 비디오 데이터셋에서는 토큰 수를 최대 80%까지 줄일 수 있습니다.



### Anticipatory Understanding of Resilient Agriculture to Clima (https://arxiv.org/abs/2411.05219)
- **What's New**: 이번 연구에서는 기후 변화와 지정학적 사건으로 인해 증가하는 식량 불안정 문제를 해결하기 위해 원격 감지(geospatial sensing), 딥 러닝(deep learning), 작물 수확량 모델링(crop yield modeling) 및 인과 모델링(causal modeling)을 결합한 식량 안전성 핫스팟 발굴 프레임워크를 제안합니다.

- **Technical Details**: 북인도의 밀 생산 중심지에서 원격 감지 및 딥 러닝을 이용하여 밀 농장 식별을 위한 정량적 분석을 제공합니다. 기후 변화에 따른 작물 수확량의 영향을 WOFOST 도구를 통해 모델링하고, 식량 배급 시스템의 주요 요인을 파악합니다. 또한, 시스템 다이내믹스 모델을 활용하여 식량 불안정성을 식별하는 방법을 제안합니다.

- **Performance Highlights**: 이 연구의 결과로, 북인도의 밀 농장을 효과적으로 식별하고, 기후 시나리오에 따른 작물 생산을 예측할 수 있는 시스템을 개발하였습니다. 이는 인도 및 다른 지역의 식량 시스템의 회복력을 강화하는 데 기여할 수 있는 기초 자료를 제공합니다.



### Toward Cultural Interpretability: A Linguistic Anthropological Framework for Describing and Evaluating Large Language Models (LLMs) (https://arxiv.org/abs/2411.05200)
Comments:
          Accepted for publication in Big Data & Society, November 2, 2024

- **What's New**: 이 논문은 언어 인류학(linguistic anthropology)과 머신 러닝(machine learning, ML)의 새로운 통합을 제안하며, 언어의 기반과 언어 기술의 사회적 책임에 대한 관심이 융합되는 지점을 탐구합니다.

- **Technical Details**: 연구에서는 LLM(대형 언어 모델) 기반 챗봇과 인간 사용자의 대화를 분석하여, 문화적 해석 가능성(cultural interpretability, CI)이라는 새로운 연구 분야를 제시합니다. CI는 인간-컴퓨터 상호작용의 발화적 인터페이스에서 인간 사용자가 AI 챗봇과 함께 의미를 생성하는 방식을 중심으로, 언어와 문화 간의 역동적인 관계를 강조합니다.

- **Performance Highlights**: CI는 LLM이 언어와 문화의 관계를 내부적으로 어떻게 '표현'하는지를 조사하며, (1) 언어 인류학의 오랜 질문에 대한 통찰을 제공하고, (2) 모델 개발자와 인터페이스 디자이너가 언어 모델과 다양한 스타일의 화자 및 문화적으로 다양한 언어 커뮤니티 사이의 가치 정렬을 개선할 수 있도록 돕습니다. 논문은 상대성(relativity), 변이(variation), 지시성(indexicality)이라는 세 가지 중요한 연구 축을 제안합니다.



### Explainable AI through a Democratic Lens: DhondtXAI for Proportional Feature Importance Using the D'Hondt Method (https://arxiv.org/abs/2411.05196)
- **What's New**: 이번 연구는 D'Hondt 투표 원칙을 활용하여 인공지능 모델의 해석 가능성을 증대시키는 DhondtXAI 방법론을 제시합니다. 이 방법론은 기계 학습에서의 feature importance 해석을 쉽게 하기 위해 자원 할당 개념을 적용합니다.

- **Technical Details**: DhondtXAI는 결정 트리 기반 알고리즘을 중심으로 feature importance를 평가하는 구조적 방법론입니다. 사용자가 설정하는 매개변수는 총 투표 수, 의석 수, 제외할 feature, feature 앨리언스, 임계값(Threshold) 등을 포함하며, 이는 유연하고 통찰력 있는 분석을 가능하게 합니다.

- **Performance Highlights**: SHAP와 DhondtXAI를 비교하여 CatBoost 및 XGBoost 모델에서 유방암 및 당뇨병 예측에 대한 feature attribution의 효과성을 평가했습니다. DhondtXAI는 해석 가능성을 증가시키기 위해 연합 형성과 임계값을 활용하며, 이는 AI 모델에서 feature importance를 이해하는 데에 기여할 수 있음을 보여줍니다.



### AGE2HIE: Transfer Learning from Brain Age to Predicting Neurocognitive Outcome for Infant Brain Injury (https://arxiv.org/abs/2411.05188)
Comments:
          Submitted to ISBI 2025

- **What's New**: 이번 연구에서는 Hypoxic-Ischemic Encephalopathy (HIE)와 관련된 신경인지 결과를 예측하기 위한 새로운 딥러닝 모델 AGE2HIE를 제안합니다. AGE2HIE는 건강한 뇌 MRI로부터 학습한 지식을 HIE 환자로 전이하여 신경인지 결과를 예측할 수 있도록 설계되었습니다.

- **Technical Details**: AGE2HIE는 다음의 여러 단계에서 작동합니다: (a) 건강한 뇌 MRI를 사용한 뇌 나이 추정에서 HIE 환자의 신경인지 결과 예측으로의 작업 간 전이, (b) 0-97세에서 영아(0-2주)로의 나이 간 전이, (c) 3D T1-weighted MRI로부터 3D 확산 MRI로의 모달리티 간 전이, (d) 건강한 대조군에서 HIE 환자까지의 건강 상태 간 전이를 포함합니다.

- **Performance Highlights**: AGE2HIE는 기존의 방법에 비해 예측 정확성을 3%에서 5%까지 개선할 수 있으며, 이는 다양한 사이트에서의 모델 일반화 또한 포함됩니다. 예를 들어, 크로스 사이트 검증에서 5% 성능 개선이 확인되었습니다.



### Interpretable Measurement of CNN Deep Feature Density using Copula and the Generalized Characteristic Function (https://arxiv.org/abs/2411.05183)
- **What's New**: 이 논문에서는 Convolutional Neural Networks (CNN)의 깊은 특성의 확률 밀도 함수(Probability Density Function, PDF)를 측정하기 위한 새로운 경험적 접근 방식을 제안합니다. 이 접근 방식은 CNN의 특징 이해와 이상 탐지(anomaly detection) 개선에 기여할 수 있습니다.

- **Technical Details**: 이 연구는 copula 분석(copula analysis)과 직교 모멘트 방법(Method of Orthogonal Moments, MOM)을 결합하여 다변량 깊은 특성 PDF의 일반화된 특징 함수(Generalized Characteristic Function, GCF)를 직접 측정합니다. 연구 결과, CNN의 비음수 깊은 특징들은 Gaussian 분포로 잘 근사되지 않으며, 네트워크가 깊어질수록 이러한 특징들은 지수 분포에 점점 가까워집니다. 또한, 깊은 특징들은 깊이가 증가할수록 독립적이 되어 가지만, 극단값 표현 사이에서는 강한 의존성(상관관계 혹은 반상관관계)을 보이는 것으로 나타났습니다.

- **Performance Highlights**: 이 연구는 CNN 특징의 확률 밀도를 측정하는 데 있어 기존의 가정(parametric assumptions)이나 선형성 가정을 사용하지 않으면서도 높은 정확도를 유지할 수 있는 방법론을 제시합니다. 또한, 극단값에 대해 높은 상관관계를 보이는 특징들이 컴퓨터 비전 탐지의 중요한 신호이며, 이는 앞으로의 특징 밀도 분석 방법론에 기초자료를 제공할 것으로 기대됩니다.



### Integrating Large Language Models for Genetic Variant Classification (https://arxiv.org/abs/2411.05055)
Comments:
          21 pages, 7 figures

- **What's New**: 본 연구에서는 변이 예측에 있어 최신 Large Language Models (LLMs), 특히 GPN-MSA, ESM1b 및 AlphaMissense를 통합하여 변이의 병원성을 예측하는 새로운 접근 방식을 제시합니다. 이 모델들은 DNA 및 단백질 서열 데이터와 구조적 통찰력을 활용하여 변이 분류를 위한 포괄적인 분석 프레임워크를 형성합니다.

- **Technical Details**: GPN-MSA는 MSA (Multiple Sequence Alignment) 데이터를 기반으로 한 DNA 언어 모델로, 100종의 종에서 수집된 진화 정보를 활용해 병원성 점수를 예측합니다. ESM1b는 단백질 언어 모델로, 20가지의 아미노산 변이에 대한 병원성을 예측하며, AlphaMissense는 단백질 서열로부터 구조를 예측한 후 병원성 예측을 수행합니다. 이들 모델을 통합하여 보다 정확하고 포괄적인 변이 병원성 예측 도구를 개발하였습니다.

- **Performance Highlights**: 연구 결과, 통합 모델들이 기존의 최첨단 도구에 비해 해석이 모호하고 임상적으로 불확실한 변이를 처리하는 데 있어 상당한 개선을 보여주었으며, ProteinGym 및 ClinVar 데이터셋에서 새로운 성능 기준을 설정했습니다. 이 결과는 진단 프로세스의 품질을 향상시키고 개인 맞춤형 의학의 경계를 확장할 수 있는 잠재력을 가지고 있습니다.



### Intellectual Property Protection for Deep Learning Model and Dataset Intelligenc (https://arxiv.org/abs/2411.05051)
- **What's New**: 본 논문은 딥러닝 모델과 데이터셋의 지적재산권(IP) 보호에 대한 포괄적인 리뷰를 제시합니다. 최근 대형 언어 모델(LLM)과 같은 최신 연구의 발전을 바탕으로, 고급 모델의 보호와 관련된 다양한 방법론과 이론적 배경을 다룹니다.

- **Technical Details**: 이 연구는 모델 지능과 데이터셋 지능의 IP 보호를 모두 포괄하며, IP 보호(IPP) 방법론을 반응적(reactive) 및 능동적(proactive) 관점에서 평가합니다. 또한, 분산 학습 환경에서의 IPP의 도전 과제를 다루며, 각기 다른 공격 유형에 대해 분석합니다.

- **Performance Highlights**: 논문에서는 고급 IP 보호 방법의 한계 및 각 방법론의 강점과 약점을 체계적으로 분석하며, 새로운 연구 방향과 실제 응용 가능성을 제시하고 있습니다. 또한, IPP에 관련된 측정 지표를 구분하여 종합적으로 제시합니다.



### Leveraging LLMs to Enable Natural Language Search on Go-to-market Platforms (https://arxiv.org/abs/2411.05048)
Comments:
          11 pages, 5 figures

- **What's New**: 이 논문은 Zoominfo 제품을 위한 자연어 쿼리(queries) 처리 솔루션을 제안하고, 대규모 언어 모델(LLMs)을 활용하여 검색 필드를 생성하고 이를 쿼리로 변환하는 방법을 평가합니다. 이 과정은 복잡한 메타데이터를 요구하는 기존의 고급 검색 방식의 단점을 극복하려는 시도입니다.

- **Technical Details**: 제안된 솔루션은 LLM을 사용하여 자연어 쿼리를 처리하며, 중간 검색 필드를 생성하고 이를 JSON 형식으로 변환하여 최종 검색 서비스를 위한 쿼리로 전환합니다. 이 과정에서는 여러 가지 고급 프롬프트 엔지니어링 기법(techniques)과 체계적 메시지(system message),few-shot prompting, 체인 오브 씽킹(chain-of-thought reasoning) 기법을 활용했습니다.

- **Performance Highlights**: 가장 정확한 모델은 Anthropic의 Claude 3.5 Sonnet으로, 쿼리당 평균 정확도(accuracy) 97%를 기록했습니다. 부가적으로, 작은 LLM 모델들 또한 유사한 결과를 보여 주었으며, 특히 Llama3-8B-Instruct 모델은 감독된 미세조정(supervised fine-tuning) 과정을 통해 성능을 향상시켰습니다.



### PhoneLM:an Efficient and Capable Small Language Model Family through Principled Pre-training (https://arxiv.org/abs/2411.05046)
- **What's New**: 이 연구는 작은 언어 모델(SLM)의 설계에서 하드웨어 특성을 고려하지 않던 기존의 방법에서 벗어나, 사전 학습(pre-training) 전에 최적의 런타임 효율성을 추구하는 새로운 원칙을 제시합니다. 이를 바탕으로 PhoneLM이라는 SLM 패밀리를 개발하였으며, 현재 0.5B 및 1.5B 변종이 포함되어 있습니다.

- **Technical Details**: PhoneLM은 스마트폰 하드웨어(예: Qualcomm Snapdragon SoC)를 위해 설계된 사전 학습 및 지시 모델로, 다양한 모델 변형을 포함합니다. SLM 아키텍처는 다층 구조와 피드 포워드 네트워크의 활성화 함수(activation function)와 같은 하이퍼파라미터를 포함하며, 특히 ReLU 활성화를 사용합니다. 연구에서는 100M 및 200M 파라미터 모델을 가지고 다양한 설정에서 런타임 속도(inference speed)와 손실(loss)을 비교했습니다.

- **Performance Highlights**: PhoneLM은 1.5B 모델이 Xiaomi 14에서 58 tokens/second의 속도로 실행되어 대안 모델보다 1.2배 빠르며, 654 tokens/second의 속도를 NPU에서 달성합니다. 또한, 7개의 대표 벤치마크에서 평균 67.3%의 정확도를 기록하며, 기존 비공식 데이터셋에서 훈련된 SLM보다 더 나은 언어 능력을 보입니다. 이 모든 자료는 완전 공개되어 있어 재현 가능성과 투명성을 제공합니다.



### Deep Heuristic Learning for Real-Time Urban Pathfinding (https://arxiv.org/abs/2411.05044)
- **What's New**: 이 논문은 전통적인 휴리스틱 기반 (heuristic-based) 알고리즘을 심층 학습 모델 (deep learning models)로 변환하여 실시간 컨텍스트 데이터(traffic 및 weather conditions)를 활용하는 새로운 도시 경로 탐색 방법을 소개합니다. 두 가지 방법을 제안합니다: 현재 환경 조건에 따라 경로를 동적으로 조정하는 향상된 A* 알고리즘과 과거 및 실시간 데이터를 사용해 다음 최적 경로를 예측하는 신경망 모델입니다.

- **Technical Details**: 연구에서는 MLP, GRU, LSTM, Autoencoders, Transformers와 같은 여러 심층 학습 모델을 체계적으로 비교하였으며, 베를린의 시뮬레이션된 도시 환경에서 평가되었습니다. 두 가지 접근법 중 신경망 모델이 전통적인 방법에 비해 최대 40%의 여행 시간을 단축시키며, 향상된 A* 알고리즘은 34%의 개선을 달성했습니다. 이러한 결과는 심층 학습이 실시간으로 도시 내비게이션을 최적화하는 잠재력을 보여줍니다.

- **Performance Highlights**: 제안된 모델을 통해 경로 탐색 효율성이 크게 향상되었으며, 특히 심층 학습 모델이 전통적인 방법 보다 효과적임을 입증하였습니다. 실시간 데이터에 대한 적응성을 통해 동적 도시 환경에서의 경로 최적화에 기여하고 있습니다.



### Towards Interpreting Language Models: A Case Study in Multi-Hop Reasoning (https://arxiv.org/abs/2411.05037)
Comments:
          University of Chicago, Computer Science, Master of Science Dissertation

- **What's New**: 이 논문에서는 다중 추론(multi-hop reasoning) 질문에 대한 모델의 성능을 개선하기 위해 특정 메모리 주입(memory injections) 메커니즘을 제안합니다. 이 방법은 언어 모델의 주의(attention) 헤드에 대한 목표 지향적인 수정을 제공하여 그들의 추론 오류를 식별하고 수정하는 것을 목표로 합니다.

- **Technical Details**: 제안된 방법은 먼저 GPT-2 모델의 단일 및 다중 추론 프롬프트에 대한 층(layer)별 활성화를 분석하여, 추론 과정에서 언어 모델에 관련된 프롬프트 específicos 정보(정보의 종류를 '기억'이라고 명명)를 주입할 수 있는 메커니즘을 제공합니다. 주의 헤드에서 적절히 위치에서 정보를 주입함으로써 다중 추론 작업에서 성공적인 다음 토큰의 확률을 최대 424%까지 높일 수 있음을 보였습니다. 이를 통해 'Attention Lens'라는 도구를 개발하여 주의 헤드의 출력을 사람에게 이해할 수 있는 형식으로 해석합니다.

- **Performance Highlights**: 이 방법을 통해 모델의 다중 추론 성능이 상당히 향상되었습니다. 메모리 주입을 통해 모델의 예측이 개선되었으며, 424%까지 성공적인 다음 토큰의 확률이 향상되었습니다. 이런 성과는 특정 주의 헤드의 작은 부분 집합이 모델의 예측에 상당한 영향을 미칠 수 있음을 나타냅니다.



### Diagonalization without Diagonalization: A Direct Optimization Approach for Solid-State Density Functional Theory (https://arxiv.org/abs/2411.05033)
- **What's New**: 본 논문에서는 밀도 함수 이론(DFT)에서의 변수 점유수(occupation number) 문제를 처리하기 위한 새로운 접근 방식을 제안합니다. 이 방법은 고유 함수(eigenfunction)와 점유 매트릭스(occupation matrix)를 매개변수화하여 자유 에너지(free energy)를 최소화합니다. 이를 통해 '자기 대각화(self-diagonalization)' 개념을 도입하며, 점유 매트릭스가 대각으로 가정될 때 Kohn-Sham 해밀토니안(Kohn-Sham Hamiltonian)도 자연스럽게 대각화됩니다.

- **Technical Details**: 이 접근법은 자유 에너지를 최적화하면서 고유 함수와 점유 매트릭스의 모든 물리적 제약 조건을 매개변수화에 통합하여 제한된 최적화 문제를 완전히 미분 가능한 비제한 문제로 변환합니다. QR 분해 방식을 사용하여 고유 함수의 직교성 제약을 처리하며, 이 방법은 사각 행렬에서도 적용 가능하여, 평면파를 사용하는 고체 상태 DFT에 더욱 적합합니다. 점유 매트릭스 매개변수화에서는 파울리 배타 원리(Pauli exclusion principle), 전하 보존(charge conservation) 및 에르미트성(Hermiticity)을 항상 만족하도록 하는 새로운 접근 방식을 제안합니다.

- **Performance Highlights**: 알루미늄과 실리콘에 대한 실험을 통해 자기 대각화가 효율적으로 이루어지고, 점유수의 페르미-디랙 분포(Fermi-Dirac distribution)가 생성되며, Quantum Espresso의 SCF 방법과 일치하는 밴드 구조를 산출함을 확인하였습니다. 이 모든 성과는 제안된 방법의 유효성을 입증합니다.



### Multimodal Quantum Natural Language Processing: A Novel Framework for using Quantum Methods to Analyse Real Data (https://arxiv.org/abs/2411.05023)
Comments:
          This thesis, awarded a distinction by the Department of Computer Science at University College London, was successfully defended by the author in September 2024 in partial fulfillment of the requirements for an MSc in Emerging Digital Technologies

- **What's New**: 이 논문은 양자 컴퓨팅을 언어의 구성(compositionality) 모델링에 적용하는 데 중심을 두고 있으며, 특히 이미지와 텍스트 데이터를 통합하여 다중 양자 자연어 처리(Multimodal Quantum Natural Language Processing, MQNLP) 영역을 발전시키려는 시도를 다룹니다.

- **Technical Details**: 이 연구는 Lambeq 툴킷을 활용하여 이미지-텍스트 양자 회로를 설계하고, 구문 기반 모델(DisCoCat 및 TreeReader)과 비구문 기반 모델의 성능을 평가하는 비교 분석을 포함합니다. 연구는 구문 구조를 이해하고 설명하는 데 초점을 두며, 군집 모델, bag-of-words 모델, 그리고 순차 모델의 성능을 비교합니다.

- **Performance Highlights**: 구문 기반 모델이 특히 우수한 성과를 보였으며, 이는 언어의 구성 구조를 고려했기 때문입니다. 본 연구의 결과는 양자 컴퓨팅 방법이 언어 모델링의 효율성을 높이고 향후 기술 발전을 이끄는 잠재력을 가지고 있음을 강조합니다.



### Reservoir computing for system identification and predictive control with limited data (https://arxiv.org/abs/2411.05016)
Comments:
          16 pages, 12 figures

- **What's New**: 이번 연구에서는 미리 학습된 RNN 변형들이 MPC(Model Predictive Control)의 대체 모델로서 제어 시스템의 동역학을 학습하는 능력을 평가합니다.

- **Technical Details**: 연구는 Echo State Networks (ESN)가 계산 복잡성 감소, 더 긴 유효 예측 시간, MPC의 비용 함수 감소 등 여러 이점이 있음을 보여줍니다. RNN(순환 신경망)의 다양한 변형, LSTM(장기 단기 기억 네트워크), GRU(게이티드 순환 유닛)과 ESN의 성능을 비교했습니다.

- **Performance Highlights**: ESN은 제어 성능 면에서 기존의 게이티드 RNN 아키텍처보다 우수한 결과를 보여주며, 잡음이 있는 환경에서도 효과적입니다. 또한 학습 시간도 획기적으로 단축될 수 있습니다.



### Fast and interpretable electricity consumption scenario generation for individual consumers (https://arxiv.org/abs/2411.05014)
- **What's New**: 본 논문에서는 재생 가능 에너지로의 전환을 위해 저전압 그리드(low-voltage grid)의 강화를 신속하게 진행할 수 있는 새로운 시나리오 생성 기법을 제안합니다.

- **Technical Details**: 제안하는 기법은 예측 클러스터링 트리(Predictive Clustering Trees, PCT)를 기반으로 하며, 기존의 복잡한 절차와 비교하여 보다 효율적이고 해석 가능한 시나리오 생성을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 세 가지 서로 다른 위치의 데이터셋에서 기존의 최첨단 기법과 동등 이상의 정확도를 가지면서 훈련 및 예측 속도가 적어도 7배 더 빠른 것으로 나타났습니다.



### Enhancing literature review with LLM and NLP methods. Algorithmic trading cas (https://arxiv.org/abs/2411.05013)
- **What's New**: 이 연구는 알고리즘 거래 분야의 지식을 분석하고 정리하기 위해 machine learning 알고리즘을 활용했습니다. 136 백만 개의 연구 논문 데이터셋을 필터링하여 1956년부터 2020년 1분기까지 발표된 14,342개의 관련 논문을 식별했습니다. 이들은 전통적인 키워드 기반 알고리즘과 최신 topic modeling 방법을 비교하여 알고리즘 거래의 다양한 접근 방식과 주제의 인기 및 진화를 평가했습니다.

- **Technical Details**: 연구는 자연어 처리(Natural Language Processing, NLP)를 통해 지식을 자동으로 추출할 수 있는 유용성을 입증하고, ChatGPT와 같은 최신 대형 언어 모델(Large Language Models, LLMs)의 새로운 가능성을 강조합니다. 연구는 알고리즘 거래에 대한 연구 기사가 전반적인 출판 수 증가 속도보다 빠르게 증가하고 있음을 발견했습니다. 머신 러닝 모델은 최근 몇 년간 가장 인기 있는 방법으로 자리 잡았습니다.

- **Performance Highlights**: 본 연구는 복잡한 질문을 해결하기 위해 작업을 더 작은 구성 요소로 분해하고 추론 단계를 포함함으로써 알고리즘 거래 방법론에 대한 깊은 이해를 도왔습니다. LLMs를 이용한 자동 문서 리뷰 및 데이터셋 개선의 효율성을 보여주며, 시간에 따른 자산 클래스, 시간 지평선 및 모델의 인기 변화를 상세히 분석했습니다.



### Scattered Forest Search: Smarter Code Space Exploration with LLMs (https://arxiv.org/abs/2411.05010)
- **What's New**: 본 연구는 코드 생성(cod generation)을 위한 LLM(대형 언어 모델) 추론(inference) 확장을 위한 새로운 접근 방식을 제안합니다. 코드 생성을 블랙박스 최적화 문제로 설정하고, 해결책의 다양성을 높이기 위해 Scattered Forest Search(SFS)라는 최적화 기법을 도입합니다.

- **Technical Details**: SFS는 입력 프롬프트를 동적으로 변경하여 다양한 출력(output)을 생성하는 Scattering 기법과, 초기화된 다양한 랜덤 시드를 사용하여 탐색 범위를 넓히는 Foresting 기법을 포함합니다. 또한 Ant Colony Optimization과 Particle Swarm Optimization에서 영감을 받아 Scouting 기법이 추가되어, 검색 단계에서 긍정적이거나 부정적인 결과를 공유함으로써 탐색(exploration)과 활용(exploitation)을 개선합니다.

- **Performance Highlights**: HumanEval+에서 67.1%, HumanEval에서는 87.2%의 pass@1 비율을 기록하며, 기존 최첨단 기술 대비 각각 8.6%와 4.3%의 성능 향상을 달성했습니다. 또한, 코드 컨테스트, Leetcode 등의 다양한 벤치마크에서 기존 방법에 비해 더 빠른 정확한 솔루션을 발견할 수 있음을 보여주었습니다.



### CSI-GPT: Integrating Generative Pre-Trained Transformer with Federated-Tuning to Acquire Downlink Massive MIMO Channels (https://arxiv.org/abs/2406.03438)
- **What's New**: 이번 연구는 generative pre-trained Transformer (GPT)를 federated-tuning 방식과 통합하여, 효과적인 downlink channel state information (CSI) 획득을 위한 CSI-GPT 접근 방식을 제안합니다.

- **Technical Details**: 우리는 Swin Transformer 기반의 채널 획득 네트워크(SWTCAN)를 제안하여, downlink CSI를 낮은 pilot/feedback 오버헤드로 획득합니다. VAE 기반의 채널 샘플 생성기(VAE-CSG)는 높은 품질의 CSI 샘플이 부족한 문제를 해결하며, 온라인 federated-tuning 방법으로 SWTCAN의 성능을 향상시킵니다.

- **Performance Highlights**: 시뮬레이션 결과에 따르면, 제안된 federated-tuning 방법은 전통적인 centralized learning (CL) 방법에 비해 업링크 통신 오버헤드를 최대 34.8%까지 줄일 수 있습니다.



### SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models (https://arxiv.org/abs/2411.05007)
Comments:
          Quantization Library: this https URL Inference Engine: this https URL Website: this https URL Demo: this https URL Blog: this https URL

- **What's New**: 본 논문에서는 Diffusion 모델의 메모리 사용량과 지연 시간을 줄이기 위해 weights와 activations를 4비트로 양자화하는 새로운 방법인 SVDQuant를 제안합니다. 이 방법은 기존의 post-training quantization 방법들이 가진 한계를 극복하는 데 중점을 두고 있습니다.

- **Technical Details**: SVDQuant는 outliers를 흡수하기 위해 low-rank branch를 도입합니다. 이 과정에서 activation에서 weight로 outliers를 이동시키고, Singular Value Decomposition (SVD)을 통해 low-rank branch를 운영하여 고정밀도로 quantization을 진행합니다. 또한, Nunchaku라는 inference engine을 통해 low-bit branch와 low-rank branch의 커널을 융합하여 메모리 접근을 최적화합니다.

- **Performance Highlights**: 실험 결과, 12B FLUX.1 모델에 대해 메모리 사용량을 3.5배 줄이고, 4비트 weights만 양자화한 기준에 비해 3.0배의 속도 향상을 달성했습니다. 이 성능은 16GB RTX 4090 GPU에서 수치를 측정했습니다.




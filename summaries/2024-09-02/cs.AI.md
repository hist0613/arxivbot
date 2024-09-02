New uploads on arXiv(cs.CL)

### SYNTHEVAL: Hybrid Behavioral Testing of NLP Models with Synthetic CheckLists (https://arxiv.org/abs/2408.17437)
- **What's New**: 이 논문에서는 SYNTHEVAL이라는 새로운 하이브리드 행동 테스트 프레임워크를 제안합니다. 이 프레임워크는 대형 언어 모델(Large Language Models, LLMs)을 활용하여 다양한 테스트 유형을 생성하여 NLP 모델을 종합적으로 평가할 수 있도록 합니다.

- **Technical Details**: SYNTHEVAL은 LLM을 사용하여 제어된 생성(controlled generation) 방식으로 문장을 생성한 후, 특정 작업에 대한 NLP 모델의 예측과 LLM의 예측을 비교하여 어려운 예제를 식별합니다. 마지막 단계에서는 전문 인력들이 이러한 예제를 조사하고, 템플릿을 수동으로 설계하며, 특정 작업 모델이 지속적으로 드러내는 실패 유형들을 구분합니다.

- **Performance Highlights**: SYNTHEVAL을 감정 분석(sentiment analysis) 및 유해 언어 탐지(toxic language detection) 두 분류 작업에 적용하여, 해당 작업들에서 강력한 모델들이 가진 약점을 효과적으로 식별하는 성과를 보였습니다.



### CLOCR-C: Context Leveraging OCR Correction with Pre-trained Language Models (https://arxiv.org/abs/2408.17428)
Comments:
          13 pages, 3 figures, currently under peer review

- **What's New**: 이번 연구는 Context Leveraging OCR Correction (CLOCR-C)을 도입하여, transformer 기반의 language model (LM)이 Optical Character Recognition (OCR) 품질을 향상시키는 데 어떻게 기여할 수 있는지를 다룹니다. 특히 신문 및 정기간행물에 사용된 LMs의 후처리 기능을 검증합니다.

- **Technical Details**: CLOCR-C는 transformer 기반 LM이 'infilling' 및 context-adaptive 능력을 활용하여 OCR 결과의 정확성을 높이는 방법입니다. 세 가지 질문을 통해 주요 연구를 진행하며, 다양한 데이터셋(Nineteenth Century Serials Edition, Overproof collection)을 사용하여 실험을 수행하였습니다.

- **Performance Highlights**: 실험 결과, 일부 LMs는 오류율을 60% 이상 감소시키는 성과를 보였으며, Named Entity Recognition과 같은 하위 NLP 작업의 성능에도 긍정적인 영향을 미쳤습니다. 사회문화적 맥락을 제공하는 것이 성능을 개선하는 데 중요한 역할을 한다고 밝혀졌습니다.



### NDP: Next Distribution Prediction as a More Broad Targ (https://arxiv.org/abs/2408.17377)
Comments:
          8 pages,5 figures

- **What's New**: 본 논문은 Next Distribution Prediction (NDP)라는 새로운 방법론을 제안하여, 기존의 next-token prediction (NTP) 패러다임의 한계를 극복하고, LLM의 학습 성능을 향상시키고자 합니다. NDP는 n-gram 분포를 사용하여 one-hot 타겟을 대체하며, 총 과제가 포함된 다양한 실험에서 눈에 띄는 성능 향상을 보여주었습니다.

- **Technical Details**: NDP는 LLM 훈련에서 n-gram 통계적 언어 모델을 활용한 새로운 접근법으로, 훈련 중 단일 one-hot 분포 대신 n-gram 분포를 타겟으로 설정합니다. 이 방법은 supervised 언어 모델링과 Causal Language Modeling (CLM) 분포를 결합하여 훈련합니다. 또한, NDP는 도메인 적응 및 언어 전이 상황에서 유리하게 작용합니다.

- **Performance Highlights**: NDP는 다음과 같은 두드러진 성과 향상을 기록했습니다: 번역 작업에서 최대 +2.97 COMET 개선, 일반 작업에서 평균 0.61점 개선, 의료 분야에서 평균 10.75점 개선. 이는 LLM이 narrow candidate 문제를 해결함으로써 성능을 크게 향상시킬 수 있음을 입증합니다.



### Assessing Generative Language Models in Classification Tasks: Performance and Self-Evaluation Capabilities in the Environmental and Climate Change Domain (https://arxiv.org/abs/2408.17362)
Comments:
          11 pages, to be published in NLDB 2024

- **What's New**: 이번 논문은 두 개의 대형 언어 모델(GPT-3.5, Llama2)과 한 개의 소형 언어 모델(Gemma)의 기후 변화 및 환경 분류 작업에서의 성능을 비교 분석하는 내용을 다루고 있습니다. 다양한 분류 작업( Eco-Relevance, Environmental Impact Analysis, Stance Detection)에 대해 BERT 기반 모델을 기준으로 이들 모델의 성능 차이를 평가하였으며, 특히 자기 평가 능력과 신뢰도 점수의 일치도를 분석했습니다.

- **Technical Details**: 이 연구는 텍스트 분류 작업에서 LLM과 SLM의 효과성을 탐구하고 있으며, BERT기반 모델을 기준으로 성능을 평가하였습니다. 세 가지 분류 작업에서 GPT-3.5-Turbo-0125는 폐쇄형 LLM, Llama-2-13b-chat-hf는 개방형 미드 LLM, Gemma는 개방형 소형 모델로 사용되었습니다. 또한, 각 모델의 신뢰도 점수를 통해 자기 평가능력을 분석하였습니다.

- **Performance Highlights**: 연구 결과, LLM들은 일반적으로 SLM보다 좋은 성능을 보였지만, BERT 기반 모델들이 여전히 이들보다 전반적으로 더 우수한 성능을 발휘하였습니다. 특히, GPT 모델은 높은 Recall을 기록하며 이목을 끌었고, 신뢰도 점수 평가에서는 GPT가 잘 일치된 결과를 보였고, Llama는 적절한 일치도를, Gemma는 다중 레이블 설정에서 일관성이 부족한 결과를 보였습니다.



### Impact of ChatGPT on the writing style of condensed matter physicists (https://arxiv.org/abs/2408.17325)
Comments:
          9 pages, 1 figure, 7 tables

- **What's New**: 이 연구는 ChatGPT 출시가 비영어 원어민 작가의 논문 초록 작성 스타일에 미치는 영향을 정량적으로 분석합니다. 결과적으로, 비영어 원어민의 영어 품질이 향상되었음을 발견했으며, 이는 ChatGPT의 사용이 광범위하게 이루어졌음을 시사합니다.

- **Technical Details**: 이 연구는 차별적 차이(difference-in-differences, DID) 방법을 사용하며, 이는 경제학에서 정책 효과를 평가하기 위해 개발된 기법입니다. 연구에서는 영어 원어민을 통제군으로, 비영어 원어민을 실험군으로 설정하여 ChatGPT 사용에 따른 작성 스타일 변화를 분석합니다. ‘Grammarly’ 소프트웨어를 활용하여 영어 작성 스타일을 정량적으로 측정하였고, Welch t-test, paired t-test 및 DID 방법을 사용하여 데이터를 분석하였습니다.

- **Performance Highlights**: 연구 결과, ChatGPT 출시 이후 비영어 원어민의 초록 작성 품질이 유의미하게 향상되었고, 특히 독일어 및 기타 인도유럽어 계통 작가들에서는 변화가 없었던 반면, 라틴계 및 우랄 알타이계 작가들에서는 중요하게 개선된 것으로 나타났습니다. 또한, 독창적인 단어 사용이 증가하고 희귀 단어 사용이 감소하는 경향을 보였습니다.



### Towards Tailored Recovery of Lexical Diversity in Literary Machine Translation (https://arxiv.org/abs/2408.17308)
Comments:
          Accepted to EAMT 2024

- **What's New**: 이 논문에서는 기계 번역(Machine Translation, MT)이 문학 작품 번역에서의 어휘 다양성(Lexical Diversity) 손실 문제를 해결하기 위해 새롭게 제안된 접근법을 소개합니다. 기계 번역은 일반적으로 인간 번역(Human Translation, HT)보다 어휘적으로 부족하다는 문제를 인식하고, 특정 소설에 대해 원본 및 번역된 텍스트를 구분하는 분류기로 번역 후보를 재정렬하는 방식으로 어휘 다양성을 복구하려고 합니다.

- **Technical Details**: 제안된 방법은 31개의 영어 소설을 네덜란드어로 번역하는 과정에서 평가되었습니다. 기존 접근법들이 어휘 다양성을 경직되게 증가시키는 반면, 이 연구에서는 각 소설의 어휘 다양성이 다르다는 점을 강조하며, 이를 토대로 어휘 다양성을 회복하는 유연한 방법을 제안합니다. 이는 번역 후보를 재정렬할 때 분류기를 이용하여 원본과 번역된 텍스트를 구별하는 방식으로 진행됩니다.

- **Performance Highlights**: 연구 결과, 본 방법은 특정 도서들에 대해 인간 번역과 유사한 어휘 다양성 점수를 조회하는 데 성공했습니다. 이로써 기계 번역이 갖는 제한된 어휘적 표현을 극복하는 데 있어 효과적임을 보여주고 있습니다.



### Improving Extraction of Clinical Event Contextual Properties from Electronic Health Records: A Comparative Study (https://arxiv.org/abs/2408.17181)
- **What's New**: 이번 연구는 의료 관련 자연어 처리(NLP) 모델의 성능을 비교 분석하여, 전자 건강 기록(EHR) 데이터를 기반으로 한 텍스트 분류 작업에서 BERT 모델의 효과를 입증합니다. 연구는 클래스 불균형(class imbalance) 문제를 해결하기 위한 다양한 접근 방식을 통합하여 더 나은 성능을 이끌어냅니다.

- **Technical Details**: 연구는 MedCAT 라이브러리를 활용하여 Named Entity Recognition과 Linking (NER+L) 작업을 수행한 후, BERT 모델을 기반으로 텍스트 분류를 진행합니다. 특히, 데이터 클래스 불균형 문제를 해결하기 위해 클래스 가중치(class weights) 기법과 LLM을 통한 합성 데이터 생성(synthetic data generation) 등의 방법을 적용하여 성능을 향상시키고자 했습니다. 실험 결과, BERT는 Bi-LSTM 모델보다 최대 28% 더 높은 재현율(recall)을 보였습니다.

- **Performance Highlights**: BERT 모델을 접목한 신경망 아키텍처는 두 개의 연결 계층을 추가하여 최고의 성능을 발휘하였으며, 합성 데이터 생성 방법 적용 시 성능이 더욱 향상됨을 확인하였습니다. 전체 데이터 세트에서 합성 데이터는 5% 미만으로 생성되었으며, 동일한 클래스 간 균형을 유지하기 위해 관리가 이루어졌습니다.



### MaFeRw: Query Rewriting with Multi-Aspect Feedbacks for Retrieval-Augmented Large Language Models (https://arxiv.org/abs/2408.17072)
- **What's New**: 이 논문에서는 사용자 정보 요구를 보다 잘 설명하기 위한 쿼리 재작성 방법인 MaFeRw를 제안합니다. 기존의 쿼리 재작성 방법이 한계가 있음을 지적하고, 다중 측면의 밀집 보상 피드백을 활용하여 RAG 시스템의 성능을 향상시킵니다.

- **Technical Details**: MaFeRw는 T5 모델을 초기화 단계로 사용하여 사용자 쿼리를 재작성하는 새로운 방법입니다. 세 가지 메트릭(재작성된 쿼리와 금본 문서 간의 유사도, 랭킹 메트릭, 생성된 응답과의 ROUGE)을 기반으로 강화 학습 피드백을 설계하였습니다. PPO(Proximal Policy Optimization) 알고리즘을 활용해 최적의 쿼리 재작성 전략을 탐색합니다.

- **Performance Highlights**: 두 개의 대화형 RAG 데이터셋에서 실험 결과, MaFeRw는 기존 방법에 비해 우수한 생성 메트릭과 더 안정적인 훈련을 달성함을 보여주었습니다. 경기 분석 결과 밀집 보상이 단일 보상에 비해 더 안정적인 훈련 과정과 생성 결과를 제공함을 입증했습니다.



### Novel-WD: Exploring acquisition of Novel World Knowledge in LLMs Using Prefix-Tuning (https://arxiv.org/abs/2408.17070)
- **What's New**: 이 논문은 새로운 세계 지식 사실을 사전 학습된 대형 언어 모델(PLM)에게 가르치는 문제를 다룹니다. Novel-WD라는 새로운 데이터셋을 생성하고, prefix-tuning을 활용하여 새로운 정보를 효과적으로 학습하는 방법을 제안합니다.

- **Technical Details**: 논문에서는 새로운 사실들을 포함하는 Novel-WD 데이터셋을 소개하며, causal language modeling 및 다중 선택 질문(MCQ) 형태의 두 가지 평가 작업을 설계하였습니다. 또한, prefix-tuning을 통해 모델의 기존 매개변수를 변경하지 않고도 정보 저장 용량을 증가시킬 수 있음을 나타냅니다.

- **Performance Highlights**: 실험 결과, prefix-tuning이 새로운 사실 학습에 있어 LoRA보다 우수한 성능을 발휘하는 것으로 나타났습니다.



### From Text to Emotion: Unveiling the Emotion Annotation Capabilities of LLMs (https://arxiv.org/abs/2408.17026)
Comments:
          to be published in Interspeech 2024

- **What's New**: 본 연구는 감정 인식 모델의 자동화 또는 지원을 위해 Large Language Models(LLMs), 특히 GPT-4의 잠재력을 탐색합니다. 기존의 사람이 주석을 단 데이터에 의존하는 방식보다, GPT-4를 사용한 감정 주석의 효과를 비교하고 분석합니다.

- **Technical Details**: 연구에서는 인간 주석과의 일치성, 인간 인식과의 정렬성, 모델 훈련에 대한 GPT-4의 영향 등 세 가지 측면에서 비교하였으며, GPT-4의 주석이 여러 데이터셋과 평가자에 걸쳐 인간 주석보다 일관되게 선호되었다는 결과를 도출했습니다.

- **Performance Highlights**: GPT-4는 기존의 감독받는 모델들과 비교했을 때 감정 인식 성능에서 유사한 수준을 보였고, 주석 필터링 과정으로서의 적용 가능성이 제시되었습니다. 또한, GPT-4의 주석을 통해 훈련 데이터셋의 질을 개선할 수 있는 방법이 제시되었습니다.



### InkubaLM: A small language model for low-resource African languages (https://arxiv.org/abs/2408.17024)
- **What's New**: 이 논문은 에이프리카(Africa) 언어를 위한 최초의 오픈 소스 소형 다국어 언어 모델인 InkubaLM을 소개합니다. 이 모델은 0.4억 개의 매개변수를 가지고 있으며, 기계 번역(machine translation), 질문-응답(question-answering), 그리고 AfriMMLU 및 AfriXnli 작업에서 더 큰 모델들과 동등한 성능을 발휘합니다.

- **Technical Details**: InkubaLM은 0.4억 개의 매개변수로 구성된 소형 모델로, 기존의 고자원 언어 모델보다 중요한 정보를 보존하면서도 성능이 향상된 모델입니다. 이 모델은 기계 번역, 감정 분석(sentiment analysis), 명명된 개체 인식(NER), 품사의 태깅(POS), 질문 응답 및 주제 분류와 같은 다양한 NLP(Natural Language Processing) 작업에 활용될 수 있습니다.

- **Performance Highlights**: InkubaLM은 많은 대형 모델들보다 감정 분석에서 더 좋은 성능을 보이며, 여러 언어에서 일관된 결과를 나타냅니다. 이 모델은 250배의 자기 무게를 옮길 수 있는 강력한 방법론을 채택하였으며, 자원 제약이 있는 환경에서도 효율적으로 작동할 수 있는 기반을 마련합니다.



### Dynamic Self-Consistency: Leveraging Reasoning Paths for Efficient LLM Sampling (https://arxiv.org/abs/2408.17017)
- **What's New**: 이번 연구에서는 Reasoning-Aware Self-Consistency (RASC)라는 혁신적인 조기 중지 프레임워크를 제안합니다. RASC는 Chain of Thought (CoT) prompting에서 출력 응답과 Reasoning Paths (RPs) 모두를 고려하여 샘플 생성 수를 동적으로 조정합니다.

- **Technical Details**: RASC는 각 샘플 RPs에 개별 신뢰도 점수를 매기고, 특정 기준이 충족될 때 중지를 수행합니다. 또한, 가중 다수결 투표(weighted majority voting)를 적용하여 샘플 사용을 최적화하고 답변 신뢰성을 향상시킵니다.

- **Performance Highlights**: RASC는 다양한 QA 데이터셋에서 여러 LLMs를 통해 테스트하여 기존 방법들보다 80% 이상의 샘플 사용을 줄이면서 정확도를 최대 5% 향상시켰습니다.



### Tool-Assisted Agent on SQL Inspection and Refinement in Real-World Scenarios (https://arxiv.org/abs/2408.16991)
Comments:
          work in progress

- **What's New**: 최근 Text-to-SQL 방법은 데이터베이스 관리 시스템(DBMS)에서 피드백을 통합하여 대형 언어 모델(LLM)의 이점을 활용합니다. 하지만 실행 오류는 해결할 수 있지만, 데이터베이스 불일치(database mismatches) 문제는 여전히 어려움을 겪고 있습니다. 본 연구에서는 이러한 문제를 해결하기 위해 SQL 검토 및 수정을 위한 도구 지원 에이전트 프레임워크인 Tool-SQL을 제안합니다.

- **Technical Details**: Tool-SQL 프레임워크는 두 가지 전문 도구를 활용하여 SQL 쿼리의 불일치를 진단하고 수정합니다: (1) Database Retriever - SQL 조건 절이 데이터베이스의 항목과 일치하지 않을 때 피드백을 제공하는 도구, (2) Error Detector - SQL 규칙이나 도메인 전문가가 정의한 엄격한 제약 조건의 불일치를 포함하여 실행 오류를 진단하는 도구입니다. 그리고 Spider-Mismatch라는 새로운 데이터셋을 소개하여 실제 시나리오에서 발생하는 조건 불일치 문제를 반영합니다.

- **Performance Highlights**: Tool-SQL은 few-shot 설정에서 Spider 및 Spider-Realistic 데이터셋의 평균 결과에서 가장 높은 성능을 기록하였으며, 실세계 시나리오에서 더 높은 애매성을 포함하는 Spider-Mismatch 데이터셋에서도 기존 방법들보다 월등히 우수한 성능을 보여줍니다.



### MemLong: Memory-Augmented Retrieval for Long Text Modeling (https://arxiv.org/abs/2408.16967)
- **What's New**: 이 연구는 MemLong: Memory-Augmented Retrieval for Long Text Generation을 소개합니다. MemLong은 외부 리트리버를 활용하여 과거 정보를 검색함으로써 긴 문맥 처리 능력을 향상시키기 위한 방법입니다.

- **Technical Details**: MemLong은 비차별적 'ret-mem' 모듈과 부분적으로 훈련 가능한 디코더 전용 언어 모델을 결합하고, 의미 수준 관련 청크를 활용하는 세밀하고 통제 가능한 검색 주의 메커니즘을 도입합니다. 이 방법은 메모리 뱅크에 저장된 과거 컨텍스트와 지식을 활용하여 키-값(K-V) 쌍을 검색합니다.

- **Performance Highlights**: MemLong은 3090 GPU에서 최대 80k의 문맥 길이로 확장 가능하며, 여러 긴 문맥 언어 모델 기준에서 다른 최신 LLM을 지속적으로 초월하는 성능을 보여주었습니다. MemLong은 OpenLLaMA보다 최대 10.2% 향상된 성능을 나타냅니다.



### A longitudinal sentiment analysis of Sinophobia during COVID-19 using large language models (https://arxiv.org/abs/2408.16942)
- **What's New**: COVID-19 팬데믹이 중국인 차별(Sinophobia)을 악화시킨 현상을 대기하는 감정 분석 프레임워크를 제안합니다. 대규모 언어 모델(LLM)을 활용하여 소셜 미디어에서 발언되는 Sinophobic 감정의 변화를 분석했습니다.

- **Technical Details**: 이 연구는 BERT 모델을 미세 조정하여 Sinophobia 관련 트윗의 감정을 분석하며, ’China virus’ 및 ’Wuhan virus’ 같은 키워드를 중심으로 감정을 분류하고 극성 점수를 계산합니다.

- **Performance Highlights**: COVID-19 확진자 수의 급증과 Sinophobic 트윗 및 감정 사이에 유의미한 상관관계를 발견했습니다. 연구 결과는 정치적 내러티브와 잘못된 정보가 대중의 감정 및 의견 형성에 미치는 영향을 강조합니다.



### Plausible-Parrots @ MSP2023: Enhancing Semantic Plausibility Modeling using Entity and Event Knowledg (https://arxiv.org/abs/2408.16937)
Comments:
          10 pages, 5 figures, 5 tables

- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)에 외부 지식을 주입하여 간단한 사건의 의미적 타당성을 확인하는 방법을 조사했습니다. 특정한 목표는 외부 지식 기반에서 추출한 세부적인 개체 유형과 사건 유형, 그리고 그 정의를 LLM에 통합하는 것입니다. 이러한 지식은 설계된 템플릿을 통해 시스템에 주입됩니다.

- **Technical Details**: 연구에서는 LLM의 성능을 향상시키기 위해 세밀한 개체 및 사건 유형을 외부 지식으로 주입하는 접근 방식을 제안합니다. 이 과정에서 자연어로 표현된 사건을 처리하기 위해 다양한 템플릿을 사용하여 개체(주어와 목적어) 및 사건(동사)의 유형과 정의를 포함한 자연어 프롬프트를 구성합니다. 데이터 분포의 불균형을 해결하기 위해 데이터 증가(data augmentation) 기법도 활용합니다.

- **Performance Highlights**: 실험 결과, 주입된 지식이 사건의 의미적 타당성을 모델링하는 데 효과적임을 입증하였습니다. 또한, 오류 분석을 통해 비사소한 개체 및 사건 유형을 식별하는 것이 중요함을 강조하였습니다.



### Event Extraction for Portuguese: A QA-driven Approach using ACE-2005 (https://arxiv.org/abs/2408.16932)
- **What's New**: 이 논문은 포르투갈어에서의 이벤트 추출(event extraction) 작업을 위한 새로운 프레임워크를 제안합니다. 기존의 연구가 영어에 비해 포르투갈어에서 부족했던 점을 해결하기 위해, ACE-2005 데이터셋의 포르투갈어 번역판을 사용하여 이벤트를 식별하고 분류하는 두 개의 BERT 기반 모델을 미세 조정했습니다.

- **Technical Details**: 이벤트 추출 작업은 주로 이벤트 트리거(trigger)와 해당 이벤트의 아규먼트(arguments)를 식별하는 두 가지 하위 작업으로 분해됩니다. 이벤트 트리거 식별은 토큰 분류(token classification) 모델을 사용하고, 아규먼트 추출에는 질문 응답(question answering) 모델을 활용하여 트리거에 대해 아규먼트 역할을 쿼리합니다. 이 과정에서 BERTimbau 모델을 사용하여 포르투갈어 텍스트 데이터로 사전 훈련되었습니다.

- **Performance Highlights**: 제안된 프레임워크는 트리거 분류에서 64.4의 F1 스코어, 아규먼트 분류에서는 46.7의 F1 스코어를 달성했습니다. 이는 포르투갈어 이벤트 추출 작업에서 새로운 최첨단(reference) 성능으로 기록됩니다.



### ACE-2005-PT: Corpus for Event Extraction in Portugues (https://arxiv.org/abs/2408.16928)
- **What's New**: 이 논문에서는 ACE-2005를 포르투갈어로 번역한 ACE-2005-PT 데이터셋을 소개합니다. 이 데이터셋은 유럽 및 브라질 변형을 포함하며, 이벤트 추출(task)에서 사용될 수 있도록 자동 번역 파이프라인을 개발하였습니다.

- **Technical Details**: ACE-2005-PT 데이터셋은 자동 번역을 통해 생성되었으며, lemmatization, fuzzy matching, synonym matching, 그리고 BERT 기반의 word aligner 등의 여러 정렬 기법을 포함하는 alignment pipeline을 적용했습니다. 번역의 정확성을 평가하기 위해 언어 전문가가 수동으로 일부 주석을 정렬했습니다.

- **Performance Highlights**: ACE-2005-PT 데이터셋의 정렬 파이프라인은 정확한 일치율 70.55% 및 완화된 일치율 87.55%를 달성하였으며, 이 데이터셋은 LDC에서 출판 승인을 받았습니다.



### Exploring Multiple Strategies to Improve Multilingual Coreference Resolution in CorefUD (https://arxiv.org/abs/2408.16893)
- **What's New**: 이 논문은 다국어 핵심 참조 해상도(coreference resolution) 시스템을 제안합니다. CorefUD 1.1 데이터셋을 활용하여 17개의 데이터셋을 12개 언어로 응용하고 성능을 개선하기 위한 여러 가지 확장을 포함합니다.

- **Technical Details**: 우리는 모노링구얼(monolingual) 및 크로스링구얼(cross-lingual) 변형을 포함한 강력한 기준 모델을 설정했습니다. 주요 확장으로는 크로스 링구얼 훈련, 구문(syntactic) 정보 통합, Span2Head 모델이 포함됩니다. 이들은 핵심어(headwords) 예측을 최적화하고 단일 체스(singleton modeling)를 개선하는 데 중점을 둡니다.

- **Performance Highlights**: 제안한 모델은 CorefUD 1.1 테스트 세트에서 평가를 수행하였고, CRAC 2023 공유 과제의 최우수 모델을 큰 폭으로 초과한 성과를 냈습니다.



### LLaVA-Chef: A Multi-modal Generative Model for Food Recipes (https://arxiv.org/abs/2408.16889)
- **What's New**: LLaVA-Chef는 다단계 접근 방식을 통해 다양한 요리 레시피 프롬프트로 훈련된 새로운 다중 모달 언어 모델입니다. 이 모델은 시각적 음식 이미지에서 언어 공간으로의 매핑을 정제하고, 관련 레시피 데이터로 LLaVA를 미세 조정하며, 다양한 프롬프트를 통해 모델의 레시피 이해도를 향상시킵니다.

- **Technical Details**: LLaVA-Chef는 Vicuna와 CLIP을 기반으로 하는 LLaVA 아키텍처를 확장하여, 시각적 및 텍스트 임베딩을 결합하고 이를 LLM에 입력하여 출력 생성을 수행합니다. 저자들은 100개 이상의 고유한 프롬프트를 도입하여 레시피의 제목, 재료 및 요리 지침과 같은 다양한 요소를 생성하는데, 이를 통해 모델의 성능을 향상시켰습니다. 또한, 사용자 정의 손실 함수를 사용하여 생성된 레시피의 언어 품질을 개선했습니다.

- **Performance Highlights**: LLaVA-Chef는 Recipe1M 데이터세트에서 평가되었으며, 대부분의 지표에서 사전 훈련된 LLM보다 일관되게 높은 점수를 기록했습니다. 특히, 다른 모델들이 0.1 CIDEr 점수를 넘지 못하는 반면, LLaVA-Chef는 21점 높은 점수를 달성하여 생성된 레시피의 질적 평가에서 모델의 장점을 확인했습니다.



### Modeling offensive content detection for TikTok (https://arxiv.org/abs/2408.16857)
Comments:
          Accepted as a conference paper at DPSH 2024, 8 pages

- **What's New**: 본 연구는 TikTok에서의 공격적인(content) 내용 탐지를 위한 기계 학습(machine learning) 및 심층 학습(deep learning) 모델의 구축과 관련된 데이터셋을 수집하고 분석하는 것을 목표로 합니다. 연구자는 120,423개의 TikTok 코멘트를 수집하였으며, 이 데이터셋은 공정한 이진 분류 접근법을 사용하고 있습니다.

- **Technical Details**: 데이터 수집은 웹 스크래핑(web scraping) 기술을 이용하여 2022년 4월부터 7월 사이의 데이터를 포함합니다. 연구에서는 자연어 처리(Natural Language Processing, NLP) 기법을 사용하여 텍스트 데이터를 전처리하고 분석하였습니다. 공격적인 언어와 관련된 언어 패턴을 정량적으로 조사하고 특정 단어 및 이모지의 출현 빈도를 평가하였습니다. BERT(Bidirectional Encoder Representations from Transformers), 로지스틱 회귀(logistic regression), 나이브 베이즈(naïve bayes) 알고리즘을 활용하여 공격적인 내용 탐지를 위한 여러 모델이 구축되었으며, BERT 모델은 F1 점수 0.863을 기록하며 성능이 가장 뛰어난 것으로 나타났습니다.

- **Performance Highlights**: 이 연구에서 수집된 데이터셋은 120,423개의 코멘트로 구성되어 있으며, 기계 학습 및 NLP 기법을 통해 공격적인 콘텐츠 탐지에 대한 F1 점수 0.863을 달성하였습니다. 이는 TikTok 사용자들이 직면하는 공격적인 언어의 비율을 이해하고 대처하는 데 기여할 것으로 기대됩니다.



### Inductive Learning of Logical Theories with LLMs: A Complexity-graded Analysis (https://arxiv.org/abs/2408.16779)
- **What's New**: 이 논문은 Formal Inference Engine으로부터 피드백을 받는 Large Language Models (LLMs)의 성능과 한계를 분석하기 위한 새로운 체계적 방법론을 제시합니다. 이 방법론은 규칙 의존성 구조에 따른 복잡성이 등급화되어 있어 LLM의 성능에서 특정 추론 과제를 정량화할 수 있도록 합니다.

- **Technical Details**: 이 연구는 Inductive Logic Programming (ILP) 시스템과의 관계를 통해 LLM의 유도 학습 특성을 평가하는 체계적인 방법론을 제안합니다. 제안한 방법은 LLM과 formal ILP inference engine 및 inductive reasoning dataset 생성을 위한 synthetic generator를 결합하여 LLM이 유도한 이론을 평가하게 됩니다. 정량적인 평가에서는 목표 규칙 집합의 의존성 복잡도에 따라 등급화됩니다.

- **Performance Highlights**: 실험 결과, 최대 LLM은 SOTA ILP 시스템 기준과 경쟁할 수 있는 성과를 보였으나, Predicate 관계 체인을 추적하는 것이 이론 복잡도보다 더 어려운 장애물임을 보여주었습니다.



### Bridging Episodes and Semantics: A Novel Framework for Long-Form Video Understanding (https://arxiv.org/abs/2408.17443)
Comments:
          Accepted to the EVAL-FoMo Workshop at ECCV'24. Project page: this https URL

- **What's New**: 기존의 연구가 긴 형식의 비디오를 단순히 짧은 비디오의 연장으로 취급하는 반면, 우리는 인간의 인지 방식을 보다 정확하게 반영하는 새로운 접근 방식을 제안합니다. 본 논문에서는 긴 형식 비디오 이해를 위해 BREASE(BRidging Episodes And SEmantics)라는 모델을 도입합니다. 이 모델은 에피소드 기억(Episodic memory) 축적을 시뮬레이션하여 행동 시퀀스를 캡처하고 비디오 전반에 분산된 의미적 지식으로 이를 강화합니다.

- **Technical Details**: BREASE는 두 가지 주요 요소로 구성됩니다. 첫 번째는 에피소드 압축기(Episodic COmpressor, ECO)로, 마이크로부터 반 매크로 수준까지 중요한 표현을 효율적으로 집계합니다. 두 번째는 의미 검색기(Semantics reTRiever, SeTR)로, 집계된 표현을 광범위한 맥락에 집중하여 의미적 정보를 보강하며, 이를 통해 특징 차원을 극적으로 줄이고 관련된 매크로 수준 정보를 유지합니다.

- **Performance Highlights**: BREASE는 여러 긴 형식 비디오 이해 벤치마크에서 최첨단 성능을 달성하며, 제로샷(zero-shot) 및 완전 감독(full-supervised) 설정 모두에서 7.3% 및 14.9%의 성능 향상을 보인 것으로 나타났습니다.



### Modularity in Transformers: Investigating Neuron Separability & Specialization (https://arxiv.org/abs/2408.17324)
Comments:
          11 pages, 6 figures

- **What's New**: 이번 연구는 Transformer 모델의 뉴런 모듈성과 태스크( task) 전문성에 대한 새로운 통찰을 제공합니다. Vision Transformer (ViT) 및 언어 모델 Mistral 7B에서 다양한 태스크에 대한 뉴런의 역할을 분석하여 태스크별 뉴런 클러스터를 발견했습니다.

- **Technical Details**: 연구에서는 선택적 가지치기(selective pruning) 및 Mixture of Experts (MoE) 군집화 기법을 사용하여 뉴런의 겹침(overlap) 및 전문성을 분석하였습니다. 연구 결과, 훈련된 모델과 임의 초기화된 모델 모두에서 뉴런 중요도 패턴이 일정 부분 지속됨을 발견했습니다.

- **Performance Highlights**: ViT 모델에서 특정 클래스에 맞는 뉴런을 비활성화함으로써 다른 클래스의 성능에 미치는 영향을 평가했으며, Mistral 모델에서는 각 태스크에서의 성능 저하가 상관 관계가 있음을 관찰했습니다.



### Investigating Neuron Ablation in Attention Heads: The Case for Peak Activation Centering (https://arxiv.org/abs/2408.17322)
Comments:
          9 pages, 2 figures, XAI World Conference 2024 Late-Breaking Work

- **What's New**: 본 연구에서는 언어 모델과 비전 트랜스포머에서 신경 세포의 활성화(ablation) 방법을 비교하며, 새로운 기법인 'peak ablation'을 제안합니다.

- **Technical Details**: 신경 활성화 양식을 각각 다르게 조정하는 네 가지 방법(Zero Ablation, Mean Ablation, Activation Resampling, Peak Ablation)을 사용하여 성능 저하 정도를 실험적으로 분석하였습니다. 특히, 'peak ablation' 기법에서는 모달 활성화(modal activation)를 기반으로 한 새로운 접근법을 사용합니다.

- **Performance Highlights**: 각기 다른 방법에서 모델 성능 저하가 가장 적은 방법을 식별하였으며, 일반적으로 resampling이 가장 큰 성능 저하를 발생시킨다는 결과를 도출했습니다.



### Bridging Domain Knowledge and Process Discovery Using Large Language Models (https://arxiv.org/abs/2408.17316)
Comments:
          This paper is accepted at the AI4BPM 2024 workshop and to be published in their proceedings

- **What's New**: 본 논문은 프로세스 발견(process discovery) 작업에서 도메인 지식(domain knowledge)을 통합하기 위해 Large Language Models (LLMs)을 활용하는 새로운 접근 방식을 제안합니다. 이는 기존의 자동화된 프로세스 발견 방법에서 간과되었던 전문가의 통찰 및 세부 프로세스 문서와 같은 정보를 효율적으로 활용할 수 있게 합니다.

- **Technical Details**: 이 연구는 LLMs로부터 파생된 규칙(rules)을 사용하여 모델 구성을 안내합니다. 이러한 방식은 도메인 지식과 실제 프로세스 실행 간의 일치를 보장하며, 자연어로 표현된 프로세스 지식과 견고한 프로세스 모델 발견을 결합하는 브리지 역할을 합니다.

- **Performance Highlights**: UWV 직원 보험 기관과의 사례 연구를 통해 프레임워크의 실용성과 효과성을 검증하며, 프로세스 분석 작업에서의 이점을 강조했습니다.



### Flexible and Effective Mixing of Large Language Models into a Mixture of Domain Experts (https://arxiv.org/abs/2408.17280)
- **What's New**: 본 연구는 훈련된 모델로부터 저비용 Mixture-of-Domain-Experts (MOE) 를 생성할 수 있는 툴킷(toolkit) 을 제시합니다. 이 툴킷은 모델 또는 어댑터(adapters) 로부터 혼합을 생성하는 데 사용될 수 있으며, 결과 MOE의 아키텍처 정의에 대한 지침 및 광범위한 테스트를 제공합니다.

- **Technical Details**: 본 툴킷은 다양한 방식을 통해 훈련된 모델을 Mixture of Domain Experts MOE에 활용할 수 있도록 유연성을 제공합니다. 특히, 전문가 모듈 또는 라우터(router)를 훈련하지 않고도 MOE를 생성할 수 있는 방법도 지원합니다. Gate-less MOE 구조는 라우터 기반 아키텍처에 비해 경쟁력이 있으며, 저비용으로 생산 가능합니다. Noisy MOE는 Gate-less MOE와 유사한 성능을 보여주며, 훈련이 필요하지 않으면서도 낮은 추론(inference) 비용을 제공합니다.

- **Performance Highlights**: Gate-less MOE 아키텍처는 소수의 고품질 전문가가 있을 경우 최적의 솔루션이 될 수 있으며, 다양한 실험을 통해 이런 구조가 효과적임을 입증했습니다. 또, Gate-less MOE는 라우터 기반 아키텍처보다 더 경쟁력이 있으며, 비용 측면에서 이점이 있습니다.



### Codec Does Matter: Exploring the Semantic Shortcoming of Codec for Audio Language Mod (https://arxiv.org/abs/2408.17175)
- **What's New**: 최근 오디오 생성 기술이 대규모 언어 모델(LLM)의 발전으로 크게 향상되었습니다. 본 연구에서는 기존 오디오 LLM 코드들이 생성된 오디오의 의미적 무결성을 유지하는 데에서 겪는 문제점을 해결하기 위해 새로운 방법인 X-Codec을 제안합니다.

- **Technical Details**: X-Codec은 사전 훈련된 의미 인코더로부터의 의미적 특징을 Residual Vector Quantization(RVQ) 단계 이전에 통합하고 RVQ 이후에 의미 재구성 손실을 도입합니다. 이를 통해 X-Codec은 음성 합성 작업에서 단어 오류율(Word Error Rate, WER)을 크게 줄이고, 음악 및 소리 생성과 같은 비음성 응용 프로그램에서도 이점을 제공합니다.

- **Performance Highlights**: 우리는 텍스트-음성 변환, 음악 연속 생성, 텍스트-소리 합성 작업에서 X-Codec의 효과를 종합적으로 평가했습니다. 결과는 제안된 방법의 효과를 일관되게 입증하며, VALL-E 기반 TTS와의 비교 평가에서 X-Codec이 기존의 분리 기법을 능가함을 확인했습니다.



### UserSumBench: A Benchmark Framework for Evaluating User Summarization Approaches (https://arxiv.org/abs/2408.16966)
- **What's New**: 이 논문은 LLM 기반의 사용자 요약 생성 기법 개발을 촉진하기 위해 설계된 UserSumBench라는 새로운 벤치마크 프레임워크를 도입합니다. 이 프레임워크는 레퍼런스가 없는 요약 품질 메트릭과 강력한 요약 방법을 포함하여 사용자 요약 접근법의 효과성을 평가할 수 있도록 돕습니다.

- **Technical Details**: UserSumBench는 두 가지 주요 구성 요소로 구성됩니다: (1) 참조가 없는 요약 품질 메트릭. 이 메트릭은 세 가지 다양한 데이터셋(영화 리뷰, Yelp, 아마존 리뷰)에서 인간 선호도와 잘 일치하는 효과성을 보여주었습니다. (2) 시간 계층적 요약 생성 및 자기 비판 검증기를 활용한 새로운 요약 방법으로 높은 품질의 요약을 생성하면서 오류를 최소화합니다.

- **Performance Highlights**: 제안된 품질 메트릭은 생성된 요약이 사용자의 향후 활동을 얼마나 잘 예측하는지 평가하여 요약 접근법의 효과를 정량적으로 검토합니다. UserSumBench는 성능 예측과 품질 평가에 있어 강력한 기준선 요약 방법을 제공하며, 이 방법은 향후 요약 기법 혁신의 기초로 작용할 것입니다.



### See or Guess: Counterfactually Regularized Image Captioning (https://arxiv.org/abs/2408.16809)
Comments:
          Accepted by ACM MM 2024

- **What's New**: 이번 논문에서는 인과 추론(causal inference)을 적용하여 기존의 이미지 캡셔닝(image captioning) 모델들이 개입적(interventional) 작업을 더 잘 수행할 수 있도록 하는 새로운 프레임워크를 제안합니다. 제안된 접근법은 전체 효과(total effect) 또는 자연적 직접 효과(natural direct effect)를 활용한 두 가지 변형이 포함되어 있습니다.

- **Technical Details**: 본 논문에서는 이미지 캡셔닝 작업을 순차적 인과 그래프(sequential causal graph)로 정형화합니다. 여기서 생성되는 각 단어는 이전 단어 및 이미지에 의해 영향을 받습니다. 이 인과 그래프를 기반으로 자가 규제(counterfactual regularization) 이미지 캡셔닝 프레임워크를 제안하며, 신뢰성을 높이는 동시에 hallucinatory 출력(hallucinations)을 줄이는 방법을 모색합니다.

- **Performance Highlights**: 다양한 데이터셋을 통한 광범위한 실험 결과, 제안된 방법은 hallucinatory 현상을 효과적으로 줄이고 이미지에 대한 모델의 충실도를 향상시킴을 입증했습니다. 또한, 소규모 및 대규모 이미지-텍스트 모델 모두에서 높은 이식성을 보여 주목할 만한 성과를 달성하였습니다.



### DualKanbaFormer: Kolmogorov-Arnold Networks and State Space Model Transformer for Multimodal Aspect-based Sentiment Analysis (https://arxiv.org/abs/2408.15379)
Comments:
          10 pages, 2 figures, and 3 tables

- **What's New**: 이번 연구에서는 텍스트와 이미지를 결합한 다중 모달 기반 감정 분석(Multi-modal aspect-based sentiment analysis, MABSA)에서의 새로운 접근법을 제안합니다. Kolmogorov-Arnold Networks (KANs)와 Selective State Space 모델(Mamba) 변환기(DualKanbaFormer)를 통해 장기 의존성(Long-range dependencies) 문제와 전역 맥락 의존성(Global-context dependencies) 문제를 해결하고자 합니다.

- **Technical Details**: 제안된 구조는 Mamba를 사용해 전역 맥락 의존성을 포착하고, Multi-head Attention (MHA)을 통해 지역 맥락 의존성을 포착합니다. 또한, KANs를 활용하여 텍스트 표현과 비주얼 표현 모두에 대한 비선형 모델링 패턴(Non-linear modelling patterns)을 캡처합니다. 텍스트 KanbaFormer와 비주얼 KanbaFormer는 게이티드 융합 층(Gated fusion layer)을 통해 상호 모달리티 역학(Inter-modality dynamics)을 포착합니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, 우리의 모델은 두 개의 공개 데이터셋에서 일부 최신 기술(SOTA) 연구들을 초월하는 성능을 보였습니다.



New uploads on arXiv(cs.IR)

### rerankers: A Lightweight Python Library to Unify Ranking Methods (https://arxiv.org/abs/2408.17344)
- **What's New**: 이 논문에서는 Python 라이브러리인 rerankers를 소개합니다. 이 라이브러리는 사용자가 가장 일반적으로 사용되는 re-ranking 접근 방식을 쉽게 사용할 수 있도록 하는 인터페이스를 제공합니다.

- **Technical Details**: rerankers는 재정렬을 통합한 단일 인터페이스를 제공하여, 사용자가 Python 코드 한 줄만 변경하여 다양한 방법을 탐색할 수 있도록 합니다. 이 라이브러리는 모든 최신 Python 버전과 호환되며, Hugging Face의 transformers 생태계에 통합되어 있습니다.

- **Performance Highlights**: 실험을 통해 rerankers의 구현은 기존 실행 결과와 성능에서 동등성을 달성했습니다. 이러한 성과는 MS Marco, Scifact 및 TREC-Covid 데이터셋에서 평가되었습니다.



### Not All Videos Become Outdated: Short-Video Recommendation by Learning to Deconfound Release Interval Bias (https://arxiv.org/abs/2408.17332)
- **What's New**: 본 논문에서는 단기 비디오 추천 시스템에서 발생하는 최근 발매 비디오에 대한 편향을 분석하고, 시간이 지남에 따라 사용자 관심을 유지하는 고전 비디오의 중요성을 강조합니다. 이를 극복하기 위해 학습과정의 편향을 제거하는 새로운 접근법인 LDRI (Learning to Deconfound the Release Interval Bias)를 제안합니다.

- **Technical Details**: LDRI는 모델에 구애받지 않는 인과적 아키텍처로, 사용자와 비디오의 상호작용에서 발생하는 데이터의 편향을 해소하기 위해 사용자 피드백과 비디오 출시 간의 관계를 사실 그래프를 통해 분석합니다. 이 과정에서 출시 간격이 혼란 변수로 작용하여 사용자 관심과 비디오 간의 비인과 관계를 형성한다는 것을 확인하고, 각 비디오의 출시 간격에 따른 민감도를 학습하는 방법론을 구현합니다.

- **Performance Highlights**: 두 개의 실제 데이터셋에서 실시한 광범위한 실험 결과, LDRI는 기존의 세 가지 표준 모델(DeepFM, NFM, AFM)보다 일관되게 우수한 성능을 보였으며, 최신 기술 모델 대비에서도 뛰어난 결과를 나타냈습니다. 이러한 분석을 통해 LDRI의 편향 제거 능력이 입증되었습니다.



### Metadata practices for simulation workflows (https://arxiv.org/abs/2408.17309)
Comments:
          19 pages, 5 figures

- **What's New**: 이 논문은 과학적 지식 생성에서 컴퓨터 시뮬레이션의 중요성과 메타데이터 관리의 필요성에 대해 논의합니다. 저자들은 메타데이터 수집과 관리의 일반적인 관행을 제시하고, 이를 통해 다양한 연구 분야에서 시뮬레이션 기반 연구의 재현성과 데이터 재사용을 촉진할 수 있는 방법을 제안합니다.

- **Technical Details**: 이 연구는 메타데이터 수집 및 관리 프로세스를 세 개의 하위 워크플로우로 나누어 설명합니다: 1) 시뮬레이션 실험, 2) 메타데이터 후처리, 3) 데이터 활용. 연구자는 메타데이터 처리를 지원하는 Python 도구인 'Archivist'를 개발하였으며, 이 도구는 사용자가 메타데이터를 선택하고 구조화하는 데 도움을 줍니다.

- **Performance Highlights**: 저자들은 연구와 관련된 두 가지 실제 사용 사례(신경과학 및 수문학)에서 제안된 관행을 적용하고, 이러한 접근 방식이 기존 워크플로에 쉽게 통합될 수 있음을 보여줍니다. 이로 인해 재현 가능한 수치적 워크플로우를 지원하고 과학자 간의 지식 이전을 촉진할 수 있습니다.



### Efficient Multi-task Prompt Tuning for Recommendation (https://arxiv.org/abs/2408.17214)
- **What's New**: 이번 논문에서는 새로운 작업을 처리할 때 멀티태스크 추천 시스템의 일반화 능력을 향상시키기 위한 방안으로, 두 단계로 구성된 프롬프트 튜닝 프레임워크인 MPT-Rec를 제안합니다. 이 프레임워크는 작업 간의 정보 공유 메커니즘을 개선하여 훈련 비용을 줄이고 기존 작업 성능에 미치는 부정적 영향을 최소화합니다.

- **Technical Details**: MPT-Rec는 두 개의 주요 단계로 구성됩니다: 1) 멀티태스크 프리트레이닝 단계와 2) 멀티태스크 프롬프트 튜닝 단계입니다. 여기서, 작업-인지 생성적 적대 네트워크를 통해 작업 특화 정보와 작업 공유 정보를 분리하여 고품질의 정보 전달을 보장합니다. 프롬프트 튜닝 단계에서는 기존 프리트레이닝 작업에서 전이된 유용한 지식을 활용하여 새로운 작업의 파라미터만 업데이트합니다.

- **Performance Highlights**: MPT-Rec는 SOTA(상태 최첨단) 멀티태스크 학습 방법인 CSRec에 비해 가장 우수한 성능을 보여주었습니다. 또한, 새로운 작업 학습 시 전체 훈련 방식과 비교하여 최대 10%의 파라미터만 사용함으로써 훈련 효율성을 상당히 향상시켰습니다.



### Understanding the User: An Intent-Based Ranking Datas (https://arxiv.org/abs/2408.17103)
- **What's New**: 정보 검색 시스템(Information Retrieval Systems)이 발전함에 따라 이 시스템들의 정확한 평가 및 벤치마킹이 중요한 과제가 되고 있습니다. 본 논문에서는 MS MARCO와 같은 웹 검색 데이터셋(Web Search Datasets)이 키워드 쿼리(Keyword Queries)만 제공하고, 의도(Intent)나 설명 없이 제공되며, 정보의 필요성을 이해하는 데 어려움이 있다는 점을 지적하고 있습니다.

- **Technical Details**: 이 연구에서는 두 가지 주요 벤치마크 데이터셋인 TREC-DL-21과 TREC-DL-22에 대해 LLMs(Large Language Models)를 활용하여 각 쿼리의 암묵적인 의도를 분석하고 이해하는 방법론을 제안합니다. 주요 의미적 요소(Semantic Elements)를 추출하여 쿼리에 대한 상세하고 맥락적인 설명을 구성합니다.

- **Performance Highlights**: 생성된 쿼리 설명을 검증하기 위해 크라우드소싱(Crowdsourcing) 방법을 사용하여 다양한 인적 관점을 통해 설명의 정확성과 정보성을 평가합니다. 이 정보는 순위 매기기(Ranking), 쿼리 재작성(Query Rewriting)과 같은 작업의 평가 세트로 활용될 수 있습니다.



### Evaluation of Table Representations to Answer Questions from Tables in Documents : A Case Study using 3GPP Specifications (https://arxiv.org/abs/2408.17008)
Comments:
          10 pages, 4 figures, 2 tables

- **What's New**: 이 연구는 기술 문서에서 발생하는 질문 응답(QA)에 대한 정보 추출 방식에 중점을 두고 있으며, 특히 표와 텍스트가 혼합된 상황에서의 데이터 표현 방식을 체계적으로 분석했습니다. 또한 3GPP 문서에서의 사례 연구와 전문적인 QA 데이터 세트를 기반으로 성능 향상 문제를 다루었습니다.

- **Technical Details**: 이 연구에서는 3GPP 문서에서 표와 텍스트가 혼합된 데이터를 파싱하고, 테이블 헤더 정보를 모든 셀에 포함시키는 방식의 행 단위 표현(row level representation)이 정보 검색 성능을 개선한다고 보고했습니다. 또한, 텍스트가 혼합된 정보의 검색 성능에 미치는 영향을 분석하기 위해 다양한 테이블 표현 방식을 실험적으로 평가했습니다.

- **Performance Highlights**: 연구 결과, 테이블에 대한 행 단위 임베딩이 전체 테이블의 단일 임베딩보다 성능이 뛰어나며, 테이블 헤더 정보를 포함하는 방식이 정보 검색의 정확성을 향상시키는 것으로 나타났습니다. 이러한 연구 결과는 공개된 사전 훈련된 모델들이 이와 같은 복합적인 정보 환경에서도 경쟁력 있는 성능을 발휘할 수 있음을 보여줍니다.



### Identifying and Clustering Counter Relationships of Team Compositions in PvP Games for Efficient Balance Analysis (https://arxiv.org/abs/2408.17180)
Comments:
          TMLR 09/2024 this https URL

- **What's New**: 이 논문에서는 게임 설정에서 균형을 정량화하는 새로운 방법론을 제시합니다. 특히, PvP 게임에서 팀 구성 간의 힘 관계를 분석할 수 있는 두 가지 고급 지표를 개발했습니다.

- **Technical Details**: 논문에서는 Bradley-Terry 모델과 Siamese neural networks를 결합하여 팀 구성의 힘을 예측하는 방법을 설명합니다. 이 모델은 게임 결과를 기반으로 힘 값을 도출하며, 텍스트의 주요 기법 중 하나로는 deterministic vector quantization이 포함되어 있습니다. 두 가지 새로운 균형 측정 지표인 Top-D Diversity와 Top-B Balance를 정의하여 모호한 승률을 보완합니다.

- **Performance Highlights**: 이 방법론은 Age of Empires II, Hearthstone, Brawl Stars, League of Legends와 같은 인기 온라인 게임에서 검증되었으며, 전통적인 쌍별 승리 예측과 유사한 정확성을 보이면서도 분석의 복잡성을 줄였습니다. 논문은 이 방법이 단순한 게임 외에도 스포츠, 영화 선호도, 동료 평가, 선거 등 다양한 경쟁 시나리오에 적용 가능하다고 강조합니다.



### A Prototype Model of Zero-Trust Architecture Blockchain with EigenTrust-Based Practical Byzantine Fault Tolerance Protocol to Manage Decentralized Clinical Trials (https://arxiv.org/abs/2408.16885)
Comments:
          NA

- **What's New**: COVID-19 팬데믹으로 인해 분산 임상 시험(Decentralized Clinical Trials, DCT)의 필요성이 대두되었으며, 이는 환자 유지, 시험 가속화, 데이터 접근성 향상, 가상 의료 지원, 통합 시스템을 통한 원활한 커뮤니케이션을 가능하게 했습니다.

- **Technical Details**: 이 논문에서는 블록체인 기술을 바탕으로 제로 트러스트 아키텍처(Zero-Trust Architecture)를 구축하여 DCT 운영 관리를 위한 환자 생성 임상 시험 데이터를 통합하는 프로토타입 모델인 Zero-Trust Architecture Blockchain (z-TAB)을 제안합니다. 이 시스템은 Hyperledger Fabric을 활용한 EigenTrust 기반의 Practical Byzantine Fault Tolerance (T-PBFT) 알고리즘을 합의 프로토콜로 통합하며, IoT(Internet of Things) 기술을 적용하여 블록체인 플랫폼 내 데이터 처리를 간소화합니다.

- **Performance Highlights**: 이 시스템은 DCT 자동화 및 운영 과정에서 임상 시험 데이터의 안전한 전송을 보장하고, 통합된 블록체인 시스템을 통해 다양한 이해관계자 간의 데이터 처리 효율성을 평가하기 위한 철저한 검증이 이루어졌습니다.



### Longitudinal Modularity, a Modularity for Link Streams (https://arxiv.org/abs/2408.16877)
- **What's New**: 본 논문은 link streams에 대한 모듈러리티(Modularity) 품질 함수의 첫 번째 적응을 소개합니다. 기존의 방법들과는 달리, 이 접근 방식은 분석의 시간 척도와 독립적입니다.

- **Technical Details**: link streams는 시간 간격(T), 노드 집합(V), 상호작용 집합(E)으로 정의됩니다. 이 연구에서는 Longitudinal Modularity (L-Modularity)를 제안하며, 이는 동적 커뮤니티 평가에서의 유의미성을 강조합니다.

- **Performance Highlights**: 실험을 통해 L-Modularity의 적용 가능성과 품질 함수의 유의미성을 입증했습니다. 이는 기존의 정적인 네트워크 분석 방법과 비교할 때, 동적 커뮤니티 발견 문제에 대한 한 걸음 더 나아간 발전을 의미합니다.



### SynDL: A Large-Scale Synthetic Test Collection for Passage Retrieva (https://arxiv.org/abs/2408.16312)
Comments:
          9 pages, resource paper

- **What's New**: 이 연구는 TREC Deep Learning 트랙에서 기존의 시험 컬렉션을 확장하여 대규모 문서 검색을 위한 새로운 시험 컬렉션 SynDL을 개발하였습니다. SynDL은 인공 신경망의 강력한 기능을 활용하여 인간의 정확도로 신뢰할 수 있는 관련성 판단을 생성할 수 있습니다.

- **Technical Details**: SynDL은 1,900개 이상의 테스트 쿼리로 구성되며, 인간의 평가자 대신 LLM(대형 언어 모델)의 합성 레이블을 활용하여 테스트 시스템을 평가할 수 있도록 설계되었습니다. 이는 효율적이고 비용 효과적인 방법으로, 더 깊이 있는 관련성 평가를 가능하게 합니다.

- **Performance Highlights**: 테스트 결과, SynDL의 시스템 평가 순위가 이전의 인간 레이블과 높은 상관관계를 보여 주었으며, 이러한 새로운 대규모 테스트 컬렉션이 정보 검색 모델의 발전에 기여할 것으로 기대됩니다.



New uploads on arXiv(cs.CV)

### Bridging Episodes and Semantics: A Novel Framework for Long-Form Video Understanding (https://arxiv.org/abs/2408.17443)
Comments:
          Accepted to the EVAL-FoMo Workshop at ECCV'24. Project page: this https URL

- **What's New**: 기존의 연구가 긴 형식의 비디오를 단순히 짧은 비디오의 연장으로 취급하는 반면, 우리는 인간의 인지 방식을 보다 정확하게 반영하는 새로운 접근 방식을 제안합니다. 본 논문에서는 긴 형식 비디오 이해를 위해 BREASE(BRidging Episodes And SEmantics)라는 모델을 도입합니다. 이 모델은 에피소드 기억(Episodic memory) 축적을 시뮬레이션하여 행동 시퀀스를 캡처하고 비디오 전반에 분산된 의미적 지식으로 이를 강화합니다.

- **Technical Details**: BREASE는 두 가지 주요 요소로 구성됩니다. 첫 번째는 에피소드 압축기(Episodic COmpressor, ECO)로, 마이크로부터 반 매크로 수준까지 중요한 표현을 효율적으로 집계합니다. 두 번째는 의미 검색기(Semantics reTRiever, SeTR)로, 집계된 표현을 광범위한 맥락에 집중하여 의미적 정보를 보강하며, 이를 통해 특징 차원을 극적으로 줄이고 관련된 매크로 수준 정보를 유지합니다.

- **Performance Highlights**: BREASE는 여러 긴 형식 비디오 이해 벤치마크에서 최첨단 성능을 달성하며, 제로샷(zero-shot) 및 완전 감독(full-supervised) 설정 모두에서 7.3% 및 14.9%의 성능 향상을 보인 것으로 나타났습니다.



### DARES: Depth Anything in Robotic Endoscopic Surgery with Self-supervised Vector-LoRA of the Foundation Mod (https://arxiv.org/abs/2408.17433)
Comments:
          11 pages

- **What's New**: 이 논문에서는 Depth Anything in Robotic Endoscopic Surgery (DARES)라는 새로운 접근 방식을 소개합니다. 이는 로봇 보조 수술(RAS) 환경에서 자가 지도(monocular self-supervised) 깊이 추정을 수행하기 위해 Depth Anything Model V2에 새로운 적응 기술인 Vector Low-Rank Adaptation (Vector-LoRA)을 적용합니다.

- **Technical Details**: Vector-LoRA는 초기 계층에서 더 많은 매개변수를 통합하고 후속 계층에서 덜 통합하여 매개변수를 점진적으로 감소시켜 학습 효율성을 높이는 방식으로 설계되었습니다. 또한, 다중 규모 SSIM 기반의 재투영 손실(reprojection loss)을 설계하여 수술 환경의 요구사항에 더 잘 맞추어 깊이 인식을 개선합니다. 이 방법은 SCARED 데이터셋에 대해 검증되었으며, 최근 최첨단(self-supervised) 단일 카메라 깊이 추정 기법에 비해 13.3%의 개선을 보여줍니다.

- **Performance Highlights**: DARES는 수술 문맥에서 깊이 추정의 잠재력과 효율성을 입증하며, 수술 환경에서 깊이 인식을 위한 최첨단 기법 대비 우수한 성능을 보여줍니다.



### CinePreGen: Camera Controllable Video Previsualization via Engine-powered Diffusion (https://arxiv.org/abs/2408.17424)
- **What's New**: CinePreGen 시스템은 게임 엔진과 확산 모델을 결합한 새로운 비주얼 프리비주얼라이제이션 시스템으로, 기존의 AI 영상 생성 문제점인 카메라 조정 및 스토리보드 작업의 불편함을 해결합니다.

- **Technical Details**: CinePreGen은 엔진 구동의 확산 모델을 활용하여 동적 카메라 조정 및 사용자 정의 카메라 움직임을 지원합니다. 특히, CineSpace라는 새로운 카메라 매개변수 공간 표현을 사용하여 두 샷 설계를 효율적으로 구현합니다.

- **Performance Highlights**: CinePreGen은 사용자 연구를 통해 카메라 움직임 조절의 용이성과 직관성을 입증했으며, 전문적인 영화 촬영 효과를 일관성 있게 생성하고 사용자 피드백을 긍정적으로 받았습니다.



### Open-vocabulary Temporal Action Localization using VLMs (https://arxiv.org/abs/2408.17422)
Comments:
          7 pages, 5 figures, 4 tables. Last updated on August 30th, 2024

- **What's New**: 본 논문에서는 비디오 액션 로컬리제이션(영상 내 특정 행동의 타이밍을 찾기)의 새로운 접근법을 제안합니다. 기존 학습 기반 방법들의 노동 집약적인 주석 작업 없이, Open-vocabulary(개방 어휘) 방식을 채택한 학습 없는(video action localization) 방법론을 도입하였습니다.

- **Technical Details**: 제안된 방법은 iterative visual prompting(반복적인 시각적 자극 기법)을 기반으로 합니다. 영상 프레임을 조합한 이미지를 생성하고, Vision-Language Models(비전-언어 모델)을 활용하여 행동의 시작 및 종료 지점을 나타내는 프레임을 추정합니다. 이 과정은 점진적으로 샘플링 윈도우를 좁혀가며 반복됩니다.

- **Performance Highlights**: 제안된 파이프라인은 평균 프레임 기준(mean-over-frame)에서 60% 이상의 정확도를 달성하며, 기존의 방법보다 나은 성능을 보였습니다. 또한, 이러한 방식은 로봇 교육을 포함해 다양한 연구 분야에서의 응용 가능성을 시사합니다.



### How Knowledge Distillation Mitigates the Synthetic Gap in Fair Face Recognition (https://arxiv.org/abs/2408.17399)
Comments:
          Accepted at ECCV 2024 Workshops

- **What's New**: 이 연구는 얼굴 인식 시스템의 훈련을 위해 합성 데이터 및 기존 데이터의 혼합을 활용한 새로운 Knowledge Distillation (KD) 전략을 제안합니다. 기존의 Teacher 모델에서 지식을 증류하여 더 작은 Student 모델을 훈련시키는 방법을 통해, 다양한 인종에 대한 성능을 향상시키고 편향을 줄일 수 있음을 보여줍니다.

- **Technical Details**: 이 연구에서는 33개의 다양한 모델을 다양한 데이터세트와 아키텍처, 손실 함수(Loss Functions)를 사용하여 훈련했습니다. 특히, 합성 데이터와 실제 데이터를 혼합한 데이터세트를 활용하여 KL-Divergence와 Cross-Entropy 손실 함수를 사용하여 지식을 증류하는 방법론을 채택하였습니다.

- **Performance Highlights**: KD 전략을 활용함으로써, 합성 데이터에서 훈련된 모델의 성능 향상과 편향 감소가 이루어졌습니다. 특히, 합성 데이터와 실제 데이터의 혼합을 통해 훈련된 모델이 실제 데이터로 훈련된 모델과의 성능 격차를 줄이는 데 기여하였습니다.



### Look, Learn and Leverage (L$^3$): Mitigating Visual-Domain Shift and Discovering Intrinsic Relations via Symbolic Alignmen (https://arxiv.org/abs/2408.17363)
Comments:
          17 pages, 9 figures, 6 tables

- **What's New**: 이 연구는 Look, Learn and Leverage (L3)라는 새로운 학습 프레임워크를 제안하여 시각적 도메인의 변화와 내재적 관계의 부재라는 과제를 해결하고자 합니다. 이 프레임워크는 세 가지 분리된 단계인 Look, Learn, Leverage로 학습 과정을 분해합니다.

- **Technical Details**: L3 프레임워크는 대칭적인 기호 공간으로서 클래스 비의존적 세그멘테이션 마스크(SegMasks)를 사용하고, 클래스 비의존적 세그멘테이션 마스크를 통해 시각적 도메인을 정렬합니다. 여기서는 Mask Self-Attention Fusion (MSF) 모듈과 Multi-Modal Cross-Attention Fusion (MMCF) 모듈을 통해 다양한 도메인에 걸쳐 위상을 접근합니다.

- **Performance Highlights**: L3는 분리된 표현 학습(Disentangled Representation Learning, DRL), 인과 표현 학습(Causal Representation Learning, CRL), 그리고 시각적 질문 응답(Visual Question Answering, VQA)의 세 가지 작업에서 우수한 성능을 보였습니다.



### LSMS: Language-guided Scale-aware MedSegmentor for Medical Image Referring Segmentation (https://arxiv.org/abs/2408.17347)
Comments:
          14 pages, 5 figures

- **What's New**: 이 논문은 의사들이 진단 및 치료를 위한 특정 병변을 식별하는 데 필요한 Medical Image Referring Segmentation (MIRS)이라는 새로운 작업을 소개합니다. 전통적인 의료 이미지 분할 방법으로는 이러한 요구를 충족하지 못하고, 이에 대한 해결책으로 Language-guided Scale-aware MedSegmentor (LSMS)라는 방법을 제안합니다.

- **Technical Details**: LSMS는 (1) 다채로운 합성곱 커널을 사용하는 Scale-aware Vision-Language Attention 모듈과 (2) 다양한 스케일 간 상호작용을 모델링하는 Full-Scale Decoder를 포함하여 정확한 병변 위치 파악과 경계 검출을 개선합니다. 이 논문에서 제안된 RefHL-Seg 데이터셋은 2,283개의 복부 CT 슬라이스와 이에 대한 언어 설명 및 분할 마스크를 포함합니다.

- **Performance Highlights**: LSMS는 다양한 데이터셋에서 기존의 방법들과 비교하여 우수한 성능을 보이며, 계산 비용 역시 낮습니다. 실험 결과, LSMS는 MIRS 및 전통적인 의료 이미지 분할 작업에서 높은 분할 정확성을 보여줍니다.



### Enhancing Underwater Imaging with 4-D Light Fields: Dataset and Method (https://arxiv.org/abs/2408.17339)
Comments:
          14 pages, 14 figures

- **What's New**: 이 논문에서는 4-D light fields (LFs)를 활용하여 수중 이미징 문제를 해결하는 새로운 접근 방식을 제안합니다. 전통적인 2-D RGB 이미징과는 달리, 4-D LF 이미징은 다중 관점에서 장면을 포착하여 기하학적 정보를 간접적으로 인코딩합니다. 또한, 75개의 수중 장면과 3675개의 고해상도 2K 이미지 쌍으로 구성된 최초의 4-D LF 기반 수중 이미지 데이터셋을 만들어 정량적 평가를 가능하게 합니다.

- **Technical Details**: 제안된 프레임워크는 명시적(depth-related) 및 암시적(depth-related dynamic) 깊이 단서를 사용하여 수중 4-D LF 이미지의 향상과 깊이 추정을 상호 보완적으로 수행합니다. 입력된 4-D LF 이미지에서 추정된 깊이 정보를 활용하여 출력 특성을 조절하고, 이를 통해 이 복잡한 작업을 여러 단계로 나누어 최적화합니다. 또한, 특성 정렬(feature alignment)을 통해 노이즈를 줄이는 동시에 기하학적 구조를 보존합니다.

- **Performance Highlights**: 제안된 방법은 색 편향을 효과적으로 보정하고, 기존 2-D RGB 기반 접근 방식에 비해 우수한 성능을 보여줍니다. 여러 실험을 통해 4-D LF 수중 이미징의 잠재성과 우수성을 입증하였으며, 향후 수중 비전 연구에 기여할 데이터셋과 코드를 제공할 예정입니다.



### Evaluating Reliability in Medical DNNs: A Critical Analysis of Feature and Confidence-Based OOD Detection (https://arxiv.org/abs/2408.17337)
Comments:
          Accepted for the Uncertainty for Safe Utilization of Machine Learning in Medical Imaging (UNSURE 2024) workshop at the MICCAI 2023

- **What's New**: 딥 뉴럴 네트워크(DNNs)의 의료 영상 분석에서 중요성이 증가하고 있으며, 훈련 데이터와 크게 다른 입력을 식별하기 위한 방법(아웃 오브 디스트리뷰션, OOD) 연구가 진행되었습니다. 기존의 신뢰 기반(confidence-based)과 특징 기반(feature-based) OOD 탐지 방법의 성능을 비교하고, 두 가지 방법의 결합을 통해 각각의 약점을 보완하는 방안을 제시합니다.

- **Technical Details**: 이 연구는 D7P(피부과) 및 BreastMNIST(초음파) 데이터셋을 활용하여 OOD 감지를 위한 두 개의 새로운 벤치마크를 생성했습니다. OOD 이미지를 검정하기 위해, 특성을 기반으로 한 방법(예: Mahalanobis score)의 성능을 탐지하고, 신뢰성을 기반으로 한 방법(예: MCP)의 한계를 분석했습니다. 또한, 각 OOD 이미지에 대해 인공물을 수동으로 제거하여 원래 특징과 모델 예측의 영향을 분석했습니다.

- **Performance Highlights**: 연구 결과, 특징 기반 방법은 일반적으로 신뢰 기반 방법보다 OOD 탐지 성능이 더 우수하지만, 올바른 예측과 잘못된 예측을 구별하는 데에는 부족함을 보였습니다. 따라서, OOD 탐지의 신뢰성을 높이기 위해 신뢰성과 특징 기법을 결합하는 접근 방식을 제안합니다.



### BOP-D: Revisiting 6D Pose Estimation Benchmark for Better Evaluation under Visual Ambiguities (https://arxiv.org/abs/2408.17297)
- **What's New**: 본 논문에서는 6D pose estimation 방법의 평가 방식에 새로운 접근법을 제안합니다. 특히, 기존 방법들이 대칭체에만 초점을 맞추던 것에서 벗어나, 관점(viewpoint)나 가리개(occlusion)의 영향을 고려하여 각각의 이미지에 특화된 6D pose 분포를 자동으로 재주석하는 방법을 개발했습니다.

- **Technical Details**: 제안한 방법은 주어진 이미지에서 객체 표면의 가시성을 고려하여 시각적 모호성을 올바르게 판단하는 자동화된 재주석 기법입니다. 이 방법은 기존 주석을 출발점으로 하여 가시성 마스크와 3D 모델을 활용하여 작동하며, T-LESS 데이터셋을 포함한 다른 데이터셋에도 적용할 수 있습니다.

- **Performance Highlights**: 평가 결과, 새로운 주석 방식으로 인해 기존의 6D pose 추정 방법들의 순위가 크게 변경되었으며, 이 방법으로 실물 이미지에서 멀티 모달 포즈 분포를 추정하는 최신 방법들을 최초로 벤치마킹할 수 있게 되었습니다. 전체 성능 측정 결과는 이전의 단일 pose 추정 방법들보다 더욱 정교한 성능 평가를 가능하게 하였습니다.



### DCUDF2: Improving Efficiency and Accuracy in Extracting Zero Level Sets from Unsigned Distance Fields (https://arxiv.org/abs/2408.17284)
- **What's New**: 새로운 연구에서는 기존의 DCUDF 방법의 한계를 극복하기 위해 DCUDF2라는 개선된 기법을 제안합니다. DCUDF2는 자기 적응 가중치를 이용한 정밀도 중심의 손실 함수와 토폴로지 수정 전략을 포함하여 기하학적 품질을 향상시키고, 처리 효율성도 높입니다.

- **Technical Details**: DCUDF2는 언싸인드 거리 필드(Unsigned Distance Field, UDF)로부터 제로 레벨 셋(zero level set)을 추출하기 위한 최첨단 방법입니다. 본 연구에서는 자기 적응 가중치(self-adaptive weights)를 포함한 정밀도 중심 손실 함수(accuracy-aware loss function)를 사용하여 고차원 기하학적 세부 정보를 상당히 개선하며, 하이퍼 파라미터에 대한 의존성을 줄이는 방법론이 포함되어 있습니다.

- **Performance Highlights**: DCUDF2는 다양한 데이터셋에서 표면 추출 실험을 통해 기하학적 충실도(geometric fidelity)와 토폴로지 정확도(topological accuracy) 모두에서 기존의 DCUDF 및 다른 방법들을 능가하는 성능을 보였습니다.



### UrBench: A Comprehensive Benchmark for Evaluating Large Multimodal Models in Multi-View Urban Scenarios (https://arxiv.org/abs/2408.17267)
Comments:
          9 pages, 6 figures

- **What's New**: 이번 논문에서는 복잡한 다중 시각 도시 환경을 평가하기 위해 설계된 UrBench라는 포괄적인 벤치마크를 제안합니다. 이 벤치마크는 도시 작업에 대한 다양한 평가를 포함하며, 기존의 단일 시점 도시 벤치마크의 한계를 극복합니다.

- **Technical Details**: UrBench는 지역 수준(Region-Level) 및 역할 수준(Role-Level)의 질문을 포함하며, Geo-Localization, Scene Reasoning, Scene Understanding, Object Understanding의 네 가지 작업 차원에 걸쳐 총 14가지 작업 유형을 제공합니다. 데이터 수집에는 11개 도시에서의 새롭게 수집된 주석과 기존 데이터셋의 데이터가 포함됩니다. 또한, 우리는 다중 뷰 관계를 이해하는 모델의 능력을 평가하기 위해 다양한 도시 관점을 통합합니다.

- **Performance Highlights**: 21개의 LMM(대규모 다중 모달 모델)에 대한 평가 결과, 현재 모델들은 도시 환경에서 여러 측면에서 인간보다 뒤처지는 것으로 나타났습니다. 예를 들어, GPT-4o 모델조차도 많은 작업에서 인간에 비해 평균 17.4%의 성능 격차를 보였으며, 특히 서로 다른 도시 뷰에 따라 일관성 없는 행동을 보이는 경향이 있었습니다.



### VisionTS: Visual Masked Autoencoders Are Free-Lunch Zero-Shot Time Series Forecasters (https://arxiv.org/abs/2408.17253)
Comments:
          26 pages, 11 figures

- **What's New**: 이 논문은 자연 이미지에서 구축된 TSF(시계열 예측) 기초 모델인 VisionTS를 소개합니다. 이 접근 방식은 기존의 텍스트 기반 모델 및 시계열 데이터 수집 방식과는 다른 세 번째 경로를 탐색합니다.

- **Technical Details**: VisionTS는 대마스크 오토인코더(visual masked autoencoder, MAE)를 기반으로 하여 1D 시계열 데이터를 패치 수준의 이미지 재구성 작업으로 변형합니다. 이 방법은 시계열 예측 작업을 이미지 렌더링과 맞춤형 패치 정렬을 통해 진행할 수 있게 합니다.

- **Performance Highlights**: VisionTS는 기존 TSF 기초 모델보다 우수한 제로샷(zero-shot) 예측 성능을 보여주며, 최소한의 미세 조정(fine-tuning)으로도 여러 장기 시계열 예측 기준에서 최신 성능을 달성합니다.



### Abstracted Gaussian Prototypes for One-Shot Concept Learning (https://arxiv.org/abs/2408.17251)
- **What's New**: 본 연구는 Omniglot 챌린지에서 영감을 받아 고급 시각 개념을 인코딩하기 위한 클러스터 기반 생성 이미지 분할 프레임워크를 소개합니다. 이 프레임워크는 Gaussian Mixture Model (GMM)의 구성 요소에서 각각의 매개변수를 추론하여 시각 개념의 독특한 위상 서브파트를 나타냅니다.

- **Technical Details**: Abstracted Gaussian Prototype (AGP)은 GMMs를 활용하여 손글씨 문자에 대한 시각 개념을 유연하게 모델링합니다. GMMs는 데이터 포인트를 유한한 수의 Gaussian 분포의 합으로 표현하는 비지도 클러스터링 알고리즘입니다. AGP는 최소한의 데이터로 새로운 개념을 학습할 수 있도록 돕습니다.

- **Performance Highlights**: 연구 결과, 생성 파이프라인이 인간이 만든 것과 구별할 수 없는 새롭고 다양한 시각 개념을 생성할 수 있음을 보여주었습니다. 본 시스템은 낮은 이론적 및 계산적 복잡성을 유지하면서도 기존 접근 방식에 비해 강력한 성능을 발휘하였습니다.



### CondSeg: Ellipse Estimation of Pupil and Iris via Conditioned Segmentation (https://arxiv.org/abs/2408.17231)
- **What's New**: 이 논문은 AR/VR 제품에서의 시선 추적과 관련하여 눈 구성 요소(동공, 홍채, 공막)의 분할을 위한 새로운 방법인 CondSeg를 제안합니다. 기존의 다중 클래스 분할 접근 방식 대신, 우리가 먼저 볼 수 있는 부분만으로 동공과 홍채를 추정할 수 있는 조건부 분할(task)을 정의하여, 전체 동공/홍채의 별도 주석 없이 세그멘테이션 레이블에서 직접 타원을 추정합니다.

- **Technical Details**: 이 연구에서는 두 가지 사전(프라이어)을 고려합니다. 첫째, 동공/홍채의 프로젝션이 타원(ellipse)으로 모델링 될 수 있다는 점과 둘째, 눈 영역의 개방 상태에 따라 동공/홍채의 가시성이 제어된다는 것입니다. 변환된 매개변수를 기반으로 조건부 분할 손실(Conditioned segmentation loss)을 사용하여 네트워크를 최적화합니다. CondSeg 네트워크는 눈 영역 마스크와 5D 매개변수 형식의 동공/홍채 타원을 생성하며, 손실을 계산할 수 있도록 합쳐줍니다.

- **Performance Highlights**: 이 연구는 OpenEDS-2019/-2020 공개 데이터셋에서 테스트되었으며, 세그멘테이션 메트릭 지표에서 경쟁력 있는 결과를 보여줍니다. 동공/홍채의 타원 매개변수를 정확하게 제공하여 시선 추적의 추가 응용에 유용성을 제공합니다.



### OG-Mapping: Octree-based Structured 3D Gaussians for Online Dense Mapping (https://arxiv.org/abs/2408.17223)
- **What's New**: 본 논문에서는 효율적이고 견고한 실시간 밀집 맵핑(online dense mapping)을 위한 OG-Mapping 프레임워크를 제안합니다. OG-Mapping은 희소 옥트리(sparse octree)와 구조적인 3D Gaussian 표현을 결합하여 맵의 질을 획기적으로 향상시킵니다.

- **Technical Details**: OG-Mapping은 세 가지 주요 구성 요소로 이루어져 있습니다: 1) Morton 코딩(Morton coding)을 사용하여 빠른 앵커(anchors) 할당 및 검색을 위한 희소 복셀 옥트를 구성합니다. 2) 앵커 기반의 점진적 맵 개선 전략을 통해 서사 구조를 다양한 세부 수준에서 회복합니다. 3) 동적 키프레임(window) 관리 방식을 사용하여 잘못된 지역 최소값(false local minima) 문제를 완화합니다.

- **Performance Highlights**: 실험 결과, OG-Mapping은 기존 Gaussian 기반 RGB-D 온라인 맵핑 방법에 비해 더 견고하고 우수한 현실감(realism) 있는 맵핑 결과를 제공하며, PSNR에서 약 5dB 향상을 달성하였습니다.



### How Could Generative AI Support Compliance with the EU AI Act? A Review for Safe Automated Driving Perception (https://arxiv.org/abs/2408.17222)
- **What's New**: 본 논문은 유럽연합(EU) 인공지능(AI) 법 및 자율주행(AD) 인식 시스템을 위한 새로운 규제 요구 사항을 충족하기 위해 생성적 AI 모델의 응용 가능성을 탐구합니다. 특히, 생성적 AI 모델이 안전성을 개선하는 데 어떻게 기여할 수 있는지를 살펴봅니다. 또한, 규제 요구 사항에 부합하기 위한 개발자 접근 방식을 제안합니다.

- **Technical Details**: 본 연구는 생성적 AI의 응용을 체계적으로 분류하는 프레임워크를 제시하며, 특히 자율주행의 인식 안전성과 관련하여 EU AI 법의 요구 사항에 대한 이해를 돕습니다. DNN(Deep Neural Networks) 기반의 인식 시스템은 많은 도전 과제가 있으며, 생성적 AI 모델들은 투명성과 견고성 측면에서 EU AI 법의 요구 사항을 해결하는 데 도움이 될 수 있습니다.

- **Performance Highlights**: 생성적 AI 모델들은 고위험 AI 시스템에서 요구되는 투명성과 신뢰성을 높이는 데 기여할 수 있는 잠재력을 지니며, 특히 다중 모달 LLMs(중대 언어 모델)와 통합될 경우 사용자와의 상호작용을 통해 보다 나은 설명 및 신뢰성 있는 의사 결정을 지원합니다.



### NanoMVG: USV-Centric Low-Power Multi-Task Visual Grounding based on Prompt-Guided Camera and 4D mmWave Radar (https://arxiv.org/abs/2408.17207)
Comments:
          8 pages, 6 figures

- **What's New**: 새로운 논문에서는 다중 센서(Multi-sensors) 설정을 통한 자동주행 시스템 및 무인 수상 자동차(USVs)에서 시각 기초(visual grounding) 기술을 효과적으로 적용하기 위한 저전력 다중 작업 모델인 NanoMVG를 설계하였다. 이 모델은 카메라와 4D 밀리미터 웨이브 레이더를 이용하여 자연어를 통해 특정 물체를 찾을 수 있도록 안내한다.

- **Technical Details**: NanoMVG는 RGB 이미지, 2D 레이더 맵, 그리고 텍스트 프롬프트를 입력으로 받고, 객체 마스크와 바운딩 박스를 출력으로 제공한다. 데이터를 효과적으로 통합하기 위해 Triplet-Modal Dynamic Fusion (TMDF)이라는 효율적인 융합 모듈을 사용하며, Mixture-of-Expert (MoE) 개념을 통해 경량화된 Edge-Neighbor MoE를 설계하여 서로 다른 과제의 성능을 향상시킨다.

- **Performance Highlights**: NanoMVG는 WaterVG 데이터셋에서 높은 경쟁력을 보여주며, 열악한 환경에서도 뛰어난 성능을 발휘한다. 무엇보다도 초저전력 소비를 자랑하여 USV의 지속적인 운용을 가능하게 한다.



### Covariance-corrected Whitening Alleviates Network Degeneration on Imbalanced Classification (https://arxiv.org/abs/2408.17197)
Comments:
          20 pages, 10 figures, 10 tables. arXiv admin note: text overlap with arXiv:2112.05958

- **What's New**: 이 논문에서는 이미지 분류에서 발생하는 클래스 불균형 문제를 해결하기 위해 Whitening-Net라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 클래스 불균형으로 인해 발생하는 네트워크 퇴화를 해결하기 위한 것으로, 분류기에 입력되는 특징들 간의 높은 선형 의존성을 낮추는 데 중점을 둡니다.

- **Technical Details**: Whitening-Net은 ZCA whitening을 통합하여 배치 샘플을 정규화(Normalization)하고 비상관화(Decorrelate)하여 분류기에 입력하기 전에 특징들을 분리합니다. 그러나 극단적인 클래스 불균형에서 배치 공분산 통계치가 크게 변동하는 문제가 발생하여 whitening 연산의 수렴을 방해합니다. 이를 해결하기 위해 두 가지 공분산 수정 모듈인 Group-based Relatively Balanced Batch Sampler (GRBS)와 Batch Embedded Training (BET)을 제안합니다.

- **Performance Highlights**: CIFAR-LT-10/100, ImageNet-LT, iNaturalist-LT와 같은 벤치마크 데이터셋에서 실시한 포괄적인 실험 평가를 통해 제안된 접근 방식의 효과성이 입증되었습니다. 이 방법은 추가적인 계산 비용 없이 전체적으로 교육할 수 있습니다.



### Hybrid Classification-Regression Adaptive Loss for Dense Object Detection (https://arxiv.org/abs/2408.17182)
- **What's New**: 이번 논문에서는 객체 탐지 모델의 성능 향상을 위해 Hybrid Classification-Regression Adaptive Loss(혼합 분류-회귀 적응 손실)인 HCRAL을 제안합니다. 이 방법은 분류(classification)와 IoU 간의 일관성을 통해 서로의 작업을 감독하며, 어려운 샘플에 집중할 수 있도록 하는 Conditioning Factor(CF)를 포함합니다.

- **Technical Details**: HCRAL은 Residual of Classification and IoU(분류 및 IoU의 잔차, RCI) 모듈과 함께 작동하여 작업 간 불일치를 해소합니다. 또한, Expand Adaptive Training Sample Selection(EATSS) 전략을 도입하여 분류 및 회귀 불일치를 보이는 추가 샘플을 제공합니다. GHM Loss 및 GIoU Loss를 기반으로 하여 RCI 모듈을 설계하였고, 샘플의 positive 및 negative에 대한 주의를 조정합니다.

- **Performance Highlights**: COCO test-dev에서 진행된 실험 결과, HCRAL을 적용한 모델이 기존의 최첨단 손실 함수에 비해 향상된 성능을 보였습니다. 또한 HCRAL을 인기 있는 일단계 모델과 결합하여 높은 정확도를 달성했습니다.



### EMHI: A Multimodal Egocentric Human Motion Dataset with HMD and Body-Worn IMUs (https://arxiv.org/abs/2408.17168)
- **What's New**: 새로운 EMHI 데이터셋은 체험형 VR/AR 응용 프로그램에서 중요한 역할을 하는 개인 중심의 사람 자세 추정(Egocentric Human Pose Estimation, HPE)을 위한 다중 모달(multi-modal) 데이터 셋입니다. 이 데이터셋은 헤드 마운트 디스플레이(Head-Mounted Display, HMD)와 몸에 착용하는 관성 측정 장치(Inertial Measurement Unit, IMU)를 사용하여 수집된 데이터로, 실제 VR 제품에 적합한 디자인으로 마련되었습니다.

- **Technical Details**: EMHI 데이터셋은 실시간 개인 중심 HPE를 수행하기 위해 다중 모달 퓨전 인코더(multimodal fusion encoder)와 시간적 특성 인코더(temporal feature encoder), MLP 기반 회귀 헤드(MLP-based regression heads)를 활용한 새로운 기준 방법인 MEPoser를 소개합니다. 데이터셋에는 58명의 피험자가 39가지 행동을 수행하는 동안 수집된 885개의 시퀀스가 포함되어 있으며, 각 시퀀스는 약 28.5시간의 기록으로 구성됩니다.

- **Performance Highlights**: MEPoser는 기존의 단일 모달 방법에 비해 더 높은 성능을 보이며, 다양한 테스트에서 우리의 데이터셋이 개인 중심 HPE 문제 해결에 기여할 수 있음을 증명합니다. 또한 EMHI 데이터셋의 출시는 VR/AR 제품에서 이 기술의 실제 구현을 촉진할 것으로 기대됩니다.



### Self-supervised Anomaly Detection Pretraining Enhances Long-tail ECG Diagnosis (https://arxiv.org/abs/2408.17154)
Comments:
          arXiv admin note: text overlap with arXiv:2404.04935

- **What's New**: 본 연구는 ECG(심전도) 진단 시스템의 성능을 개선하기 위해 자가 지도 학습 기반의 이상 탐지(pretraining) 접근 방식을 도입하였습니다. 이번 연구의 주된 목적은 희귀한 심장 이상을 효과적으로 감지하고 분류하는 것입니다.

- **Technical Details**: 제안된 두 단계 프레임워크는 먼저 이상 탐지 모델을 훈련하고, 이어서 분류(classification)를 수행하는 구조입니다. 이 과정에서 다중 스케일 크로스 어텐션(multi-scale cross-attention) 모듈이 사용되어 ECG 신호의 글로벌 및 로컬 특징을 통합적으로 분석합니다.

- **Performance Highlights**: 이 모델은 희귀한 ECG 유형에서 94.7% AUROC, 92.2% 민감도(sensitivity), 92.5% 특이도(specificity)를 달성하여 기존 방법들을 크게 초월했습니다. 실제 임상 환경에서 이 AI 기반 접근 방식은 진단의 효율성, 정확성, 완전성을 각각 32%, 6.7%, 11.8% 향상시켰습니다.



### Look, Compare, Decide: Alleviating Hallucination in Large Vision-Language Models via Multi-View Multi-Path Reasoning (https://arxiv.org/abs/2408.17150)
Comments:
          13 pages, 7 tables, 7 figures

- **What's New**: 최근 대형 비전-언어 모델(LVLM)들이 멀티모달(multi-modal) 맥락 이해에서 인상적인 능력을 보여주고 있으나, 여전히 이미지 내용과 일치하지 않는 출력을 생성하는 환각(hallucination) 문제를 겪고 있습니다. 본 논문에서는 훈련 없이 환각을 줄이기 위한 새로운 프레임워크인 MVP(Multi-View Multi-Path Reasoning)를 제안합니다.

- **Technical Details**: MVP는 이미지에 대한 다각적 정보 탐색 전략을 개발하고, 각 정보 뷰에 대해 다중 경로(reasoning) 추론을 도입하여 잠재적인 답변의 확신.certainty 점수를 정량화하여 집계합니다. 이를 통해 이미지 정보를 충분히 이해하고 잠재적 답변의 확신수를 고려하여 최종 답변을 결정합니다. 해당 방법은 CLIP, BLIP 등의 비전 인코더를 활용합니다.

- **Performance Highlights**: MVP는 네 가지 잘 알려진 LVLM에서 실험을 통해 환각 문제를 효과적으로 완화함을 입증했습니다. MVP는 기계 학습에 있어 추가 훈련 비용이나 외부 도구 없이 LVLM의 내재적 능력을 최대한 활용하는 데 중점을 두었습니다. 실험 결과는 저희 프레임워크가 최근의 훈련 없는 방법론들보다 우수한 성능을 보였음을 보여줍니다.



### GMM-IKRS: Gaussian Mixture Models for Interpretable Keypoint Refinement and Scoring (https://arxiv.org/abs/2408.17149)
Comments:
          Accepted at ECCV 2024

- **What's New**: 이 논문에서는 이미지에서 추출된 키포인트(keypoint)의 질을 더욱 명확하게 평가하고 개선할 수 있는 새로운 프레임워크를 제안합니다.

- **Technical Details**: 제안된 방법은 수정된 로버스트 가우시안 혼합 모델(modified robust Gaussian Mixture Model)을 활용하여 비모 Robust 키포인트를 제거하고 나머지를 정제합니다. 점수(score)는 두 가지 구성 요소로 나뉘며, 첫 번째는 다른 시점(viewpoint)에서 같은 키포인트를 추출할 확률과 관련이 있으며, 두 번째는 키포인트의 위치 정확도(localization accuracy)와 관련이 있습니다.

- **Performance Highlights**: 광범위한 실험 결과, 제안된 프레임워크가 인기 있는 키포인트 탐지기에서 키포인트의 반복성(repeatability) 및 호모그래피(homography)와 두 개/여러 뷰 포즈 회복(pose recovery) 작업에서 성능을 일관되게 향상시켰습니다.



### RenDetNet: Weakly-supervised Shadow Detection with Shadow Caster Verification (https://arxiv.org/abs/2408.17143)
Comments:
          AIM @ ECCV 2024 / code available at this https URL

- **What's New**: 새로운 논문에서는 기존 그림자 검출 모델들이 어두운 이미지 영역과 그림자를 구별하는 데 어려움을 겪는 문제를 해결하기 위해, 모든 검출된 그림자가 실제인지, 즉 그림자를 만드는 물체와 쌍을 이루는지 확인하는 방식을 제안합니다. 제안된 RenDetNet 모델은 학습 기반 그림자 검출 모델로서, 자기 지도 학습(self-supervised learning) 방식으로 감독 신호를 계산할 수 있는 첫 번째 모델입니다.

- **Technical Details**: RenDetNet 모델은 약하게 감독된 학습(weakly-supervised learning) 접근법을 사용하여 실제 그림자에 해당하는 물체가진 마스크(caster mask)와 그림자 마스크를 생성합니다. 이 과정에서는 합성(rendered) 장면에서 그림자 마스크와 캐스터 마스크를 추정하고, 이 마스크의 차이를 통해 그림자의 원인을 확인합니다. 모델은 Fully Convolutional Network 구조를 가지고 있으며, 두 개의 디코더 헤드(encoder heads)를 가진 방식입니다.

- **Performance Highlights**: 제안된 RenDetNet 모델은 기존의 최근 그림자 검출 방법들과 비교 시 더 우수한 성능을 보이며, 새로운 그림자-캐스터 데이터셋에 대해 효과적으로 작동합니다. 또한, Github에 코드를 공개하여 연구 커뮤니티와 공유하고 있습니다.



### Temporal and Interactive Modeling for Efficient Human-Human Motion Generation (https://arxiv.org/abs/2408.17135)
Comments:
          Homepage: this https URL

- **What's New**: 본 논문에서는 최신 모델 TIM(Temporal and Interactive Modeling)을 소개합니다. TIM은 RWKV(Receptance Weighted Key Value) 모델을 이용하여 인간 간의 모션 생성에서의 효율성을 극대화합니다.

- **Technical Details**: TIM은 Causal Interactive Injection, Role-Evolving Mixing, Localized Pattern Amplification의 세 가지 주요 구성 요소로 이루어져 있습니다. Causal Interactive Injection은 모션 시퀀스의 시간적 특성을 활용하여 RWKV 레이어에 주입하며, Role-Evolving Mixing은 상호작용 과정에서의 '능동'과 '수동' 역할을 동적으로 조정합니다. 마지막으로, Localized Pattern Amplification은 개별적인 단기 모션 패턴을 캡처하여 더 부드럽고 일관된 모션 생성이 가능하게 합니다.

- **Performance Highlights**: InterHuman 데이터셋에 대한 실험 결과, TIM은 단 32%의 파라미터만을 사용하여 기존 범주에서 최첨단 성과를 달성했습니다. 이는 인간 간의 모션 생성에서의 새로운 기준을 설정합니다.



### VQ4DiT: Efficient Post-Training Vector Quantization for Diffusion Transformers (https://arxiv.org/abs/2408.17131)
Comments:
          11 pages, 6 figures

- **What's New**: 이 논문에서는 DiT(Diffusion Transformers) 모델을 위한 새로운 포스트 트레이닝 벡터 양자화 방법인 VQ4DiT를 제안합니다. VQ4DiT는 모델 크기와 성능 간의 균형을 이루며, 매우 낮은 비트 폭인 2비트로 양자화하면서도 이미지 생성 품질을 유지하는 데 성공했습니다.

- **Technical Details**: VQ4DiT는 전통적인 VQ 방법의 한계를 극복하기 위해, 각 가중치 서브 벡터의 후보 할당 세트를 유클리드 거리 기반으로 계산하고, 가중 평균을 통해 서브 벡터를 재구성합니다. 이후 제로 데이터와 블록 단위 보정 방법을 사용하여 최적의 할당을 효율적으로 선택하며, 코드북을 동시에 보정합니다. 이 방법은 NVIDIA A100 GPU에서 DiT XL/2 모델을 20분에서 5시간 내에 양자화할 수 있습니다.

- **Performance Highlights**: 실험 결과, VQ4DiT는 ImageNet 기준에서 풀 정밀도 모델과 비교하여 경쟁력 있는 평가 결과를 보여줍니다. 특히, 이 방법은 디지털 이미지 생성 품질을 손상시키지 않으면서 가중치를 2비트 정확도로 양자화하는 데 성공했습니다.



### Multi-centric AI Model for Unruptured Intracranial Aneurysm Detection and Volumetric Segmentation in 3D TOF-MRI (https://arxiv.org/abs/2408.17115)
Comments:
          14 pages, 5 figures, 3 tables, 2 supplementary tables

- **What's New**: 이 논문에서는 개방형 오픈소스 nnU-Net 기반의 AI 모델을 개발하여 파열되지 않은 뇌동맥류(unruptured intracranial aneurysms, UICA)의 3D TOF-MRI에서의 탐지(detection) 및 분할(segmentation)을 결합하여 수행했습니다. 또한 뇌동맥류 유사 진단(diffential diagnoses) 데이터셋을 사용하여 훈련된 모델을 비교했습니다.

- **Technical Details**: 2020년부터 2023년까지의 회고적 연구로, 364명의 환자(평균 나이 59세, 60% 여성)로부터 수집된 385개의 익명 3D TOF-MRI 이미지를 포함했습니다. nnU-Net 프레임워크를 사용하여 모델 개발을 진행했으며, 민감도(sensitivity), False Positive (FP)/case 비율 및 DICE 점수(DICE score)를 통해 성능을 평가했습니다. 모델의 통계 분석은 카이제곱(chi-square), 맨 위트니 U(Mann-Whitney-U), 그리고 크루스칼-왈리스(Kruskal-Wallis) 테스트를 사용하여 유의미성을 검토했습니다.

- **Performance Highlights**: 모델은 전반적으로 82%에서 85% 사이의 높은 민감도를 보였으며, FP/case 비율은 0.20에서 0.31 사이로 안정적이었습니다. 특히, 주요 모델은 85% 민감도와 0.23 FP/case 비율을 기록하여 ADAM 챌린지 우승자(61%) 및 ADAM 데이터로 훈련된 nnU-Net(51%)에 비해 더욱 우수한 성능을 나타냈습니다. 평균 DICE 점수는 0.73, NSD는 0.84으로 정확한 UICA 탐지를 성공적으로 수행했습니다.



### UTrack: Multi-Object Tracking with Uncertain Detections (https://arxiv.org/abs/2408.17098)
Comments:
          Accepted for the ECCV 2024 Workshop on Uncertainty Quantification for Computer Vision

- **What's New**: 이 논문은 다중 객체 추적을 위한 새로운 접근 방식과 신뢰할 수 있는 불확실성 추정 방법을 제안합니다. 특히, YOLOv8과 같은 실시간 객체 탐지기의 예측 분포를 신속하게 계산하고 이를 다중 객체 추적에 통합하는 방법을 처음 소개합니다.

- **Technical Details**: 제안된 방법은 YOLOv8을 사용하여 객체 탐지의 예측 분포를 빠르게 추출합니다. 이 불확실성을 Kalman 필터와 같은 추적 연관 메커니즘을 통해 전파하며, 불확실한 IoU 계산을 위한 명확한 방법을 정의합니다. 기존의 방법들과 비교하여, 이 접근법은 더욱 정교한 불확실성 관리를 가능하게 합니다.

- **Performance Highlights**: MOT17, MOT20, DanceTrack 및 KITTI와 같은 여러 벤치마크에서 제안된 방법의 효과성을 입증하였습니다. 특히, 불확실성을 Kalman 필터에 통합하는 것으로 기존 방법보다 더 뛰어난 성능을 보여주었습니다.



### RISSOLE: Parameter-efficient Diffusion Models via Block-wise Generation and Retrieval-Guidanc (https://arxiv.org/abs/2408.17095)
- **What's New**: 이번 논문에서는 자원 제약이 있는 장치에서도 사용할 수 있는 컴팩트한 깊이 생성 모델에 대한 새로운 접근 방식을 제안합니다. 기존의 diffusion 모델의 큰 파라미터 문제를 해결하기 위해, block-wise generation 방식을 이용하여 각 블록을 생성해 나가는 방식을 탐구하였습니다.

- **Technical Details**: 논문에서 제안하는 접근법은 retrieval-augmented generation (RAG) 모듈을 활용하여 훈련과 생성 단계에서 블록의 일관성을 보장하는 것입니다. 이를 위해, blk-wise denoising diffusion model을 설계하여 각 이미지 블록이 외부 데이터베이스에서 검색된 참조 이미지의 해당 블록에 조건화되어 있습니다. 이 과정은 생성된 블록 간의 공간적 및 의미적 일관성을 유지하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 compact 모델 크기에서 우수한 생성 품질을 달성함을 보여주었습니다. 다양한 블록 간의 일관성을 유지하면서 이미지를 생성할 수 있는 새로운 가능성을 시사합니다.



### Focus-Consistent Multi-Level Aggregation for Compositional Zero-Shot Learning (https://arxiv.org/abs/2408.17083)
Comments:
          Compositional Zero-Shot Learning

- **What's New**: 이번 논문에서는 기존의 compositional zero-shot learning (CZSL)에서 가지는 branch 간 일관성(consistency)과 다양성(diversity) 문제를 해결하기 위해 Focus-Consistent Multi-Level Aggregation (FOMA)이라는 새로운 방법을 제안합니다.

- **Technical Details**: 제안된 FOMA는 Multi-Level Feature Aggregation (MFA) 모듈을 통해 이미지 콘텐츠를 기반으로 각 branch에 맞는 개인화된 특성을 생성하며, Focus-Consistent Constraint를 도입해 정보를 서로 공유함으로써 모든 branch 간의 공간 정보를 보존합니다.

- **Performance Highlights**: FOMA는 세 가지 벤치마크 데이터셋(UT-Zappos, C-GQA, Clothing16K)에서 기존의 최첨단 방법들을 초월하는 성능을 보이며, 이러한 성과는 다양한 특성과 유용한 영역에 대한 일관된 주의를 결합함으로써 이루어졌습니다.



### Stochastic Layer-Wise Shuffle: A Good Practice to Improve Vision Mamba Training (https://arxiv.org/abs/2408.17081)
- **What's New**: 최근의 Vision Mamba 모델들은 높은 해상도의 영상과 긴 비디오를 처리하기 위한 훨씬 낮은 복잡도를 가지고 있으며, Vision Transformers (ViTs)와의 경쟁력 있는 성능을 보입니다. 하지만 과적합(overfitting) 문제에 직면해 있으며, 기본 크기(약 80M)로 제한되어 있어 더 큰 사이즈로 효율적으로 확장하는 방법이 명확하지 않았습니다. 본 논문은 이러한 문제를 해결하기 위해 확률적 층별 셔플 규제(Stochastic Layer-Wise Shuffle Regularization)를 제안합니다.

- **Technical Details**: 제안된 ShuffleMamba 모델은 층별로 서로 다른 확률을 부여하여 더 깊은 층에서 Token 시퀀스를 셔플(shuffle)합니다. 이 과정은 모델 아키텍처에 변화를 주지 않으며, 훈련 시 오버피팅을 줄이고 성능을 향상시키는 데 도움을 줍니다. ShuffleMamba는 뛰어난 성능을 발휘하는 것을 입증하며, 흥미롭게도 인퍼런스(inference) 과정에서는 규제를 생략할 수 있습니다.

- **Performance Highlights**: ShuffleMamba 모델은 ImageNet1k에서 0.8% 및 1.0%의 분류 정확도로 유사 크기의 ViTs를 초과하는 성능을 보여주었으며, ADE20K 세분화 및 COCO 탐지 작업에서도 유의미한 개선을 보였습니다. 최종적으로 ShuffleMamba-L은 ImageNet 분류에서 83.6%의 정확도를 달성하고, ADE20K 세그멘테이션에서는 49.4 mIoU를 기록하여 기존 Vision Mamba 모델 및 유사 크기 ViTs에 비해 뛰어난 성능을 나타냈습니다.



### Generalizing Deepfake Video Detection with Plug-and-Play: Video-Level Blending and Spatiotemporal Adapter Tuning (https://arxiv.org/abs/2408.17065)
- **What's New**: 이 논문은 딥페이크 비디오 탐지의 세 가지 주요 문제를 해결하려고 합니다: (1) 복잡한 시간적 특성, (2) 공간적 및 시간적 모델의 불균형한 학습, (3) 자원 집약적인 비디오 처리 문제. 특히, Facial Feature Drift (FFD)라는 새로운 시간적 위조 특성을 발견하고, 이를 모델이 학습할 수 있도록 Video-level Blending (VB) 데이터를 도입했습니다. 또한, lightweight Spatiotemporal Adapter (StA)를 설계하여 이미지 모델을 비디오 모델로 변환하여 효율적인 학습이 가능하게 합니다.

- **Technical Details**: 제안된 Video-level Blending (VB) 접근법은 원본 이미지와 왜곡된 버전을 프레임별로 혼합하여 FFD를 재현합니다. StA는 두 개의 스트림 3D-Conv를 활용하여 공간적 및 시간적 특성을 별도로 처리할 수 있게 설계되었습니다. 이를 통해 기존의 프리트레인된 이미지 모델에서 스파이셜과 템포럴 데이터를 동시에 학습할 수 있는 능력을 부여합니다. 실험을 통해 이 방법의 효과성을 검증하였으며, 이전에 보지 못한 새로운 위조 비디오에서도 잘 일반화됨을 입증했습니다.

- **Performance Highlights**: 제안된 접근법은 기존과 비교하여 이전에 보지 못한 포괄적인 위조 비디오에서도 우수한 성능을 보였으며, 2024년에 발표된 최신 모델들과도 경쟁할 수 있는 성능을 달성했습니다. 코드를 공개하고 프리트레인된 가중치를 제공하여 연구 커뮤니티에서의 활용도를 높였습니다.



### Instant Adversarial Purification with Adversarial Consistency Distillation (https://arxiv.org/abs/2408.17064)
- **What's New**: 본 논문에서는 One Step Control Purification (OSCP)이라는 새로운 확산 기반의 정화 모델을 제안합니다. OSCP는 단 하나의 Neural Function Evaluation (NFE)만으로 적대적 이미지를 정화할 수 있어, 기존 방법들과 비교하여 더 빠르고 효율적입니다.

- **Technical Details**: OSCP는 Latent Consistency Model (LCM)과 ControlNet을 활용하여 구성됩니다. 여기에 Gaussian Adversarial Noise Distillation (GAND)라는 새로운 일관성 증류 프레임워크를 도입하여 자연적 및 적대적 다양체의 동적을 효과적으로 연결합니다.

- **Performance Highlights**: OSCP는 ImageNet에서 74.19%의 방어 성공률을 달성하며, 각각의 정화 작업이 0.1초만 소요됩니다. 이는 기존의 확산 기반 정화 방법들보다 훨씬 시간 효율적이며 전반적인 방어 능력을 유지하는 데 기여합니다.



### Vote&Mix: Plug-and-Play Token Reduction for Efficient Vision Transformer (https://arxiv.org/abs/2408.17062)
- **What's New**: 본 연구에서는 Vision Transformers (ViTs)의 계산 비용 문제를 해결하기 위해 Vote&Mix (VoMix)라는 새로운 파라미터 없는 토큰 감소 방법을 소개합니다. VoMix는 기존의 ViT 모델에 추가적인 학습 없이 적용할 수 있으며, 높은 동질성을 가진 토큰을 식별하여 처리 효율성을 높입니다.

- **Technical Details**: VoMix는 각 레이어에서 투표 메커니즘을 통해 동질성이 높은 토큰을 선택하고 이들을 유지된 토큰과 혼합합니다. 이 과정은 self-attention 메커니즘에만 영향을 미치며, O(N²D(1-r))의 시간 복잡도를 가지도록 최적화되어 있습니다. 이를 통해 VoMix는 토큰의 동질성을 줄이고, 처리 성능을 향상합니다.

- **Performance Highlights**: 실험 결과, VoMix는 ImageNet-1K에서 기존 ViT-H 모델의 처리량을 2배 증가시키고, Kinetics-400 비디오 데이터셋에서 ViT-L의 처리량을 2.4배 증가시켰습니다. 이 과정에서 top-1 정확도는 단 0.3% 감소했습니다.



### Efficient Image Restoration through Low-Rank Adaptation and Stable Diffusion XL (https://arxiv.org/abs/2408.17060)
Comments:
          10 pages

- **What's New**: 본 연구에서는 두 개의 저랭크 적응(LoRA) 모듈을 Stable Diffusion XL(SDXL) 프레임워크와 통합하여 이미지 복원 품질과 효율성을 크게 향상시키는 SUPIR 모델을 제안합니다.

- **Technical Details**: SUPIR는 2600개의 고화질 실제 이미지와 상세한 설명 텍스트를 사용하여 훈련됩니다. PSNR(peak signal-to-noise ratio), LPIPS(learned perceptual image patch similarity), SSIM(structural similarity index measurement) 지표에서 뛰어난 성능을 보였습니다. LoRA는 모델 파라미터를 미세 조정하고 이미지를 복원하는 성능을 개선하는 데 사용됩니다.

- **Performance Highlights**: SUPIR 모델은 고품질 고해상도 이미지를 생성하는 데 있어 시간 단축과 이미지 품질 개선을 달성했습니다. SDXL 기반의 보편적인 접근 방식을 사용하여 이미지 복원 작업에서 뛰어난 결과를 나타냈습니다.



### A Survey of the Self Supervised Learning Mechanisms for Vision Transformers (https://arxiv.org/abs/2408.17059)
Comments:
          34 Pages, 5 Figures, 7 Tables

- **What's New**: 최근 딥 러닝 알고리즘은 컴퓨터 비전과 자연어 처리 분야에서 인상적인 성과를 보였으나, 훈련에 필요한 대량의 레이블이 있는 데이터 수집은 비용이 많이 들고 시간이 오래 걸립니다. 자가 감독 학습(self-supervised learning, SSL)의 적용이 증가하며, 이 방식이 레이블이 없는 데이터에서 패턴을 학습할 수 있는 가능성을 보여주고 있습니다.

- **Technical Details**: 이 논문에서는 자가 감독 학습 방법들을 상세히 분류하고, ViTs(Vision Transformers) 훈련 시에 이들이 어떻게 활용될 수 있는지를 다룹니다. 주목할 만한 점은, SSL 기술이 대량의 레이블이 없는 데이터를 활용하여 비지도 학습을 가능하게 하고, 이를 통해 모델 성능을 향상시킨다는 점입니다. SSL 방법론은 대조, 생성, 예측 접근법으로 나뉘며 각 방법론은 고유한 패턴 학습을 통해 특징을 추출합니다.

- **Performance Highlights**: ViTs는 최근 이미지 분류 및 객체 탐지와 같은 다양한 컴퓨터 비전 작업에서 우수한 성과를 보여주었습니다. 대량의 데이터셋(JFT-300M)으로 훈련된 ViTs는 레이블이 부족한 상황에서도 강력하고 일반화 가능한 표현을 학습할 수 있는 경로를 제공합니다. 이 연구는 SSL의 발전과 함께 현존하는 다양한 알고리즘의 장단점을 비교하는 분석을 제공합니다.



### LAR-IQA: A Lightweight, Accurate, and Robust No-Reference Image Quality Assessment Mod (https://arxiv.org/abs/2408.17057)
- **What's New**: 이 논문에서는 고성능 No-Reference Image Quality Assessment (NR-IQA) 모델을 경량화하여 실제 환경에 적합하도록 개선한 방법을 제안합니다. 새로 제안된 모델은 ECCV AIM UHD-IQA 챌린지 검증 및 테스트 데이터셋에서 최신 성능을 기록하면서, 기존 모델보다 약 5.7배 빨라집니다.

- **Technical Details**: 제안된 모델은 이중 분기 아키텍처를 기반으로 하여, 한 분기는 합성 왜곡이 있는 이미지를, 다른 분기는 실제 왜곡이 있는 이미지를 각각 훈련합니다. 이를 통해 다양한 왜곡 유형에 대한 일반화 능력을 높였습니다. 훈련 과정에서 여러 색 공간을 통합하여 다양한 시각적 조건에서도 강인성을 향상시키며, Kolmogorov-Arnold Networks (KANs)를 사용하여 최종 품질 회귀를 수행합니다. 이와 함께, MLP(다층 퍼셉트론) 대신 KAN을 사용함으로써 더욱 높은 정확도를 달성했습니다.

- **Performance Highlights**: 제안된 경량 모델은 다양한 오픈 소스 데이터셋에서 높은 정확도와 강인성을 보였으며, 기존의 최첨단 모델(SOTA)과 비교하여 우수한 성능을 발휘하였습니다. 실용적이며 고정밀인 결과를 통해 모델의 새로운 가능성을 제시하였습니다.



### BTMuda: A Bi-level Multi-source unsupervised domain adaptation framework for breast cancer diagnosis (https://arxiv.org/abs/2408.17054)
- **What's New**: 이번 연구에서는 유방암 진단을 위한 Bi-level Multi-source Unsupervised Domain Adaptation 방법인 BTMuda를 제안합니다. 이 방법은 여러 도메인에서의 지식 전이를 통해 도메인 간 변동 문제를 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: BTMuda는 intra-domain과 inter-domain 두 가지 수준으로 도메인 이동 문제를 나누어 처리합니다. CNN과 Transformer를 이용한 두 경로의 도메인 혼합 특징 추출기를 공동 학습하여 강력한 표현을 얻는 방법을 채택하였습니다. 또한, 교차 주의(Cross-Attention) 및 증류(Distillation)를 적용한 세 가지 가지 구조로 Transformer를 재설계하여 여러 도메인에서 도메인 불변 표현을 학습합니다. 특징 정렬 및 분류기 정렬을 위한 두 가지 정렬 모듈도 도입되었습니다.

- **Performance Highlights**: 세 개의 공공 유방조영술(mammographic) 데이터셋을 이용한 광범위한 실험 결과, BTMuda는 최신 방법보다 뛰어난 성능을 보였습니다.



### Can We Leave Deepfake Data Behind in Training Deepfake Detector? (https://arxiv.org/abs/2408.17052)
- **What's New**: 이 논문에서는 deepfake 탐지기의 일반화 능력을 향상시키기 위한 새로운 접근 방식을 제시합니다. 즉, blendfake 데이터를 사용하여 효과적인 deepfake 탐지기를 학습할 수 있는 가능성을 탐구합니다. 또한, blendfake와 deepfake 간의 점진적인 전환을 위한 Oriented Progressive Regularizor (OPR)를 소개하여 탐지기의 성능을 개선함을 입증합니다.

- **Technical Details**: 이 논문은 deepfake 탐지기의 훈련에서 blendfake와 deepfake 데이터를 함께 사용하는 기존의 방법이 비효율적일 수 있음을 지적하고, blendfake를 중심으로 점진적인 학습 과정을 구성합니다. OPR을 통해 다양한 anchor의 분포를 조직하며, feature bridging을 사용해 인접한 anchor 간의 부드러운 전환을 가능하게 합니다. 이는 실제와 가짜 간의 전환 과정을 통해 탐지 성능을 향상시키는 데 기여합니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안된 방법은 blendfake와 deepfake 모두에서 효과적으로 위조 정보를 활용하여 탐지 성능을 크게 향상시킴을 보여주었습니다. 이 접근 방식은 기존의 단일 data type(예: blendfake)에 의존하는 방법들보다 더 나은 결과를 제공하며, deepfake 탐지의 새로운 가능성을 열어줍니다.



### Text-to-Image Generation Via Energy-Based CLIP (https://arxiv.org/abs/2408.17046)
- **What's New**: 본 논문에서는 Joint Energy Models (JEMs)을 확장하여 새로운 EB-CLIP 접근 방식을 제안합니다. EB-CLIP은 CLIP를 사용하여 멀티모달 비전-언어 도메인에서 작동하며, 생성적 및 판별적 목표를 통합합니다.

- **Technical Details**: EB-CLIP은 CLIP 공간에서 코사인 유사성 기반의 이미지-텍스트 결합 에너지 함수를 도입하고, 대조적 적대적 손실을 사용하여 멀티모달 도메인에서의 적대적 훈련 목표를 확장합니다. 이를 통해 CLIP의 시각적 인코더를 훈련하여 의미적으로 유의미한 그래디언트를 생성합니다.

- **Performance Highlights**: EB-CLIP은 텍스트에서 이미지로의 생성능력을 대폭 향상시키며 CLIPAG에 비해 20 FID 포인트 이상의 성능 개선을 달성했습니다. 또한 CLIP 기반의 생성 프레임워크의 성능을 크게 향상시키고, 텍스트 기반 이미지 편집을 위한 CLIP-Score 평가 지표로 활용할 수 있음을 보여줍니다.



### CP-VoteNet: Contrastive Prototypical VoteNet for Few-Shot Point Cloud Object Detection (https://arxiv.org/abs/2408.17036)
Comments:
          Accepted by PRCV 2024

- **What's New**: 본 논문에서는 Few-shot point cloud 3D object detection (FS3D) 문제에 접근하기 위해 새로운 모델인 Contrastive Prototypical VoteNet (CP-VoteNet)을 제안합니다. CP-VoteNet은 기존 여러 프로토타입 학습 방법의 한계를 극복하고 점진적인 기하학적 및 의미적 특성을 활용하여 보다 세밀하고 일반화 가능한 프로토타입 표현을 학습합니다.

- **Technical Details**: 소스 인스턴스에서의 의미적 대조 학습(semantic contrastive learning)과 원시 대조 학습(primitive contrastive learning)을 포함하는 CP-VoteNet은 각각 의미적 카테고리 내부의 유사성을 요구하고, 기하학적 구조에 따르는 포인트들의 유사성을 촉진합니다. 이러한 두 가지 대조 학습 전략을 균형 있게 적용함으로써 네트워크의 구별 및 일반화 능력을 synergistically 향상시킵니다.

- **Performance Highlights**: FS-ScanNet 및 FS-SUNRGBD의 두 가지 FS3D 벤치마크에서 실시한 실험 결과, CP-VoteNet은 기존의 최신 방법들보다 상당한 성능 향상을 보여주었습니다. 특히, 몇 가지 설정에서 현대 상태에 대한 성능이 크게 개선되었습니다.



### ConDense: Consistent 2D/3D Pre-training for Dense and Sparse Features from Multi-View Images (https://arxiv.org/abs/2408.17027)
Comments:
          ECCV 2024

- **What's New**: 이 논문은 기존의 2D 네트워크와 대규모 다중 뷰 데이터셋을 활용하여 3D 프리트레이닝을 위한 새로운 ConDense 프레임워크를 소개합니다. 2D-3D 공동 훈련 스킴을 통해 2D 및 3D 특징을 추출하고, NeRF와 유사한 볼륨 렌더링 과정을 통해 2D-3D 특성의 일관성을 유지합니다.

- **Technical Details**: ConDense는 밀집된 픽셀 단위 특징과 희소한 핵심 포인트 기반 대표성을 모두 활용하며, 2D 및 3D 데이터의 조화로운 쿼리를 가능하게 하는 통합 임베딩 공간을 형성합니다. 이 모델은 남은 훈련 없이도 2D 이미지를 3D 장면에 효과적으로 매칭하고 중복된 3D 장면을 탐지하는 등 여러 다운스트림 작업에 사용될 수 있습니다.

- **Performance Highlights**: 사전 훈련된 모델은 3D 분류와 세분화 작업 등 다양한 3D 작업을 위한 훌륭한 초기화를 제공하며, 기존 3D 프리트레이닝 방법들보다 상당한 성능 향상을 보여줍니다. 또한, 이 모델은 자연어를 통한 3D 장면 쿼리와 같은 흥미로운 다운스트림 응용 프로그램에 대한 가능성을 열어줍니다.



### Retrieval-Augmented Natural Language Reasoning for Explainable Visual Question Answering (https://arxiv.org/abs/2408.17006)
Comments:
          ICIP Workshop 2024

- **What's New**: 이번 논문에서는 복잡한 네트워크와 추가 데이터 세트에 의존하지 않고 메모리에서 검색 정보를 활용하여 시각적 질의 응답(VQA) 및 자연어 설명(NLE)의 정확한 답변과 설득력 있는 설명을 생성하는 새로운 모델 ReRe를 제안합니다.

- **Technical Details**: ReRe는 Encoder-Decoder 아키텍처 모델로, 사전 훈련된 CLIP 비전 인코더와 사전 훈련된 GPT-2 언어 모델을 디코더로 사용합니다. GPT-2의 크로스-어텐션(layer) 레이어가 추가되어 검색 기능을 처리합니다. ReRe는 메모리 데이터베이스에서 정보를 검색하여 K개의 샘플을 가져오고, 각 샘플의 답변과 설명을 인코딩하여 평균화한 후, 이를 통해 정확하고 논리적인 답변을 생성합니다.

- **Performance Highlights**: ReRe는 VQA 정확도 및 설명 점수에서 이전 방법들을 능가하며, NLE에서도 더 설득력 있고 신뢰성 있는 향상을 보여줍니다.



### AdaptVision: Dynamic Input Scaling in MLLMs for Versatile Scene Understanding (https://arxiv.org/abs/2408.16986)
- **What's New**: 본 논문에서는 AdaptVision이라는 새로운 멀티모달 대형 언어 모델(MLLM)을 소개합니다. 이 모델은 다양한 해상도의 입력 이미지를 동적으로 처리할 수 있도록 설계되었습니다.

- **Technical Details**: AdaptVision은 이미지의 크기와 종횡비에 따라 시각 토큰(visual tokens)의 수를 조정하는 동적 이미지 분할 모듈을 포함합니다. 이 모듈은 3x3 그리드를 생성하고, 입력 이미지를 최소한의 경계 사각형에 맞게 조정합니다. 따라서 시각적 정보의 왜곡을 줄이고, MLLMs의 이미지 이해 능력을 향상시킵니다.

- **Performance Highlights**: 다양한 데이터셋에서 extensive 실험을 수행하여, AdaptVision의 성능이 자연 및 텍스트 관련 장면에서 비전-언어 작업을 처리하는 데 있어 인상적인 결과를 달성했다는 것을 보여주었습니다.



### 2DGH: 2D Gaussian-Hermite Splatting for High-quality Rendering and Better Geometry Reconstruction (https://arxiv.org/abs/2408.16982)
- **What's New**: 이 논문에서는 3D 재구성을 위한 새로운 방법으로 Gaussian-Hermite kernel를 도입하며 기존의 Gaussian kernel보다 향상된 표현력을 보여준다. 이 새로운 커널은 양자 물리학에서 영감을 받아 기본적인 Gaussian 함수를 확장하여 더 나은 경계 표현이 가능하다.

- **Technical Details**: Gaussian-Hermite polynomials를 사용하여 Gaussian Splatting의 커널로 제안하며, 이는 높은 차수의 항을 포함하여 더 복잡한 형태와 경계를 표현할 수 있게 해준다. 새로운 활성화 함수 또한 도입하여 alpha blending 과정에서 유효한 불투명도 값을 유지할 수 있도록 한다.

- **Performance Highlights**: 새로운 Gaussian-Hermite kernel은 NVS와 geometry reconstruction 작업에서 기존 Gaussian kernel보다 뛰어난 성능을 보여주었다. 실험 결과는 이 새로운 커널이 고품질 3D 재구성과 렌더링에서의 가능성을 제시함을 보여준다.



### Cross Fusion RGB-T Tracking with Bi-directional Adapter (https://arxiv.org/abs/2408.16979)
- **What's New**: 본 연구에서는 RGB-T 추적에 있어 멀티 모달(multi-modal)과 시간 정보(temporal information)의 효과적인 균형을 이루기 위해 새로운 Cross Fusion RGB-T Tracking architecture (CFBT)를 제안합니다. 기존의 방법들은 시간 정보를 간과하거나 이를 충분히 활용하지 못했습니다. CFBT는 세 가지 새로운 크로스 스페이쇼-템포럴 정보 융합 모듈(Cross Spatio-Temporal Information Fusion Modules)을 통해 다중 모달의 참여와 동적 시간 정보 융합을 보장합니다.

- **Technical Details**: CFBT의 핵심 구성 요소로는 Cross Spatio-Temporal Augmentation Fusion (CSTAF), Cross Spatio-Temporal Complementarity Fusion (CSTCF), Dual-Stream Spatio-Temporal Adapter (DSTA)로 구성됩니다. CSTAF는 크로스-어텐션(cross-attention) 메커니즘을 통해 템플릿의 특징을 포괄적으로 강화하며, CSTCF는 다양한 브랜치 간의 보완 정보를 활용하여 목표 특징을 강화하고 배경 특징을 억제합니다. DSTA는 어댑터(adapter) 개념을 적용해 RGB 모달리티를 매개로 다양한 브랜치의 보완 정보를 적응적으로 융합합니다.

- **Performance Highlights**: 다양한 RGB-T 추적 벤치마크에서 CFBT가 기존 방법들에 비해 우수한 성능을 발휘했음을 보여주는 실험 결과가 있습니다. 0.259M의 적은 학습 가능 파라미터를 추가함으로써 다중 모달 추적을 효과적으로 처리하며, 새로운 최첨단 성능을 달성했습니다.



### Synthetic Lunar Terrain: A Multimodal Open Dataset for Training and Evaluating Neuromorphic Vision Algorithms (https://arxiv.org/abs/2408.16971)
Comments:
          7 pages, 5 figures, to be published at "International Symposium on Artificial Intelligence, Robotics and Automation in Space, i-SAIRAS, 2024

- **What's New**: Synthetic Lunar Terrain (SLT)는 고대비 조명 설정에서 수집된 합성 크레이터가 포함된 개방형 데이터셋입니다. 이 데이터셋은 이벤트 기반 카메라와 기존 RGB 카메라의 여러 관점에서 촬영된 자료를 포함하며, 깊이 추정을 위한 고해상도 3D 레이저 스캔으로 보완되었습니다. 특히 neuromorphic 비전 센서에서 기록된 event-stream은 높은 데이터 속도와 저전력 소모 그리고 높은 동적 범위(High Dynamic Range) 장면에 대한 강인성 같은 독특한 이점을 제공합니다.

- **Technical Details**: SLT 데이터셋은 Exterres Lunar Analogue Facility에서 생성되었으며, 3.6m x 4.8m 면적에 다양한 높이와 표면 특징을 가진 장면을 조각하여 제작되었습니다. 데이터는 고전적 RGB 카메라(Basler a2A1920-160ucPRO)와 이벤트 기반 카메라(Gen4 Prophesee)를 사용하여 녹화되었으며, FARO Focus S70 3D 스캐너로 3D 스캔이 이루어졌습니다. 데이터는 A1-A9(왼쪽) 및 B1-B9(오른쪽)과 같이 포지션에 따라 정리되었고, RGB 데이터는 .tif 포맷으로 제공됩니다.

- **Performance Highlights**: 새롭게 제공되는 SLT 데이터셋은 ARM 프로세서와 함께 neuromorphic 비전 센서를 이용하여 장기적으로 새로운 탐색 알고리즘이나 인공지능 모델의 연구를 지원하고, 달 탐사 미션에서 로버 내비게이션이나 크레이터 환경에서의 착륙 같은 응용 가능성을 높이는 데 기여할 것으로 예상됩니다.



### Contrastive Learning with Synthetic Positives (https://arxiv.org/abs/2408.16965)
Comments:
          8 pages, conference

- **What's New**: 이 논문에서는 Contrastive Learning with Synthetic Positives (CLSP)라는 새로운 접근 방식을 소개합니다. 이 방법은 unconditional diffusion model을 사용하여 생성된 합성 이미지(synthetic images)를 추가적인 긍정 샘플로 활용하여 모델이 다양한 양성 샘플로부터 학습할 수 있도록 돕습니다.

- **Technical Details**: CLSP는 diffusion 모델 샘플링 과정에서 특징 보간(feature interpolation)을 통해 생성된 이미지로, 동일한 의미적 내용을 갖추면서도 서로 다른 배경을 가진 이미지를 만듭니다. 이때 생성된 이미지는 기준 이미지(anchor image)에 대한 '어려운' 양성으로 간주되며, 이들을 contrastive loss에 추가함으로써 성능이 개선됩니다.

- **Performance Highlights**: CIFAR10과 같은 여러 기준 데이터셋에서 CLSP는 이전의 NNCLR 및 All4One 방법에 비해 선형 평가 성능이 각각 2% 및 1% 이상 향상되었습니다. 또한, CLSP는 8개의 이전 transfer learning 벤치마크 데이터셋 중 6개에서 기존 SSL 프레임워크보다 우수한 성능을 보였습니다.



### Causal Representation-Based Domain Generalization on Gaze Estimation (https://arxiv.org/abs/2408.16964)
- **What's New**: 논문은 눈 추정을 위한 새로운 도메인 일반화 방법인 CauGE(Causal Representation-Based Domain Generalization on Gaze Estimation) 프레임워크를 제안합니다. 이 방법은 인과적 메커니즘(causal mechanisms)의 원칙을 토대로 도메인 불변 특징(domain-invariant features)을 추출하여 모델 성능을 개선합니다.

- **Technical Details**: CauGE 프레임워크는 적대적 학습(adversarial training) 접근 방식을 사용하고 추가적인 페널티 항을 도입하여 도메인 불변 특징을 추출합니다. 이 프레임워크는 인과 관계를 학습하고 인과 메커니즘의 일반 원칙을 충족하는 특징을 학습함으로써, 주의(attention) 레이어를 통해 실제 눈의 방향을 추정하는 데 충분한 특징을 제공합니다.

- **Performance Highlights**: CauGE는 눈 추정 기준에서 도메인 일반화에 관한 최신 성능을 기록하며, 타겟 데이터셋(target dataset)에 접근하지 않고도 교차 데이터셋(cross-dataset)에서 잘 일반화됩니다. 또한, 여러 실험을 통해 이러한 일반화 능력을 평가하고 우수한 결과를 보여주었습니다.



### HiTSR: A Hierarchical Transformer for Reference-based Super-Resolution (https://arxiv.org/abs/2408.16959)
Comments:
          arXiv admin note: text overlap with arXiv:2307.08837

- **What's New**: 이번 논문에서는 이미지 초해상도(image super-resolution) 문제를 해결하기 위해 새로운 계층적 트랜스포머 모델인 HiTSR(Hierarchical Transformer for Super-Resolution)을 제안합니다. HiTSR는 저해상도(Low-Resolution, LR) 이미지를 고해상도(High-Resolution, HR) 참조 이미지로부터의 매칭 정보를 학습하여 향상시키는 능력을 가지고 있습니다.

- **Technical Details**: HiTSR은 기존의 다중 네트워크 및 다단계 접근 방식을 탈피하고 GAN 문헌에서 차용한 double attention block을 통합하여 아키텍처와 훈련 파이프라인을 간소화한 모델입니다. 모델은 독립적으로 두 가지 시각적 스트림을 처리하며, gating attention 전략을 통해 self-attention과 cross-attention 블록을 융합합니다. 또한 squeeze-and-excitation 모듈을 통해 입력 이미지에서 전역 컨텍스트를 캡처하고, 장거리 공간 상호작용을 촉진합니다. 모델은 SUN80, Urban100, Manga109의 세 가지 데이터셋에서 우수한 성능을 입증합니다.

- **Performance Highlights**: 특히 SUN80 데이터셋에서 HiTSR는 PSNR/SSIM 값 30.24/0.821을 달성하였으며, 이는 참조 기반 이미지 초해상도에서 attention 메커니즘의 효과성을 강조합니다. HiTSR은 목적에 맞춰 구축된 서브네트워크, 지식 증류, 다단계 학습이 필요 없이 최첨단 결과를 달성하였습니다.



### Transient Fault Tolerant Semantic Segmentation for Autonomous Driving (https://arxiv.org/abs/2408.16952)
Comments:
          Accepted ECCV 2024 UnCV Workshop - this https URL

- **What's New**: 이 연구는 자율주행차의 중요한 기능인 시맨틱 세그멘테이션(semantic segmentation) 분야에서 최초로 하드웨어 오류에 대한 내성을 분석하는 작업입니다. 새로운 활성화 함수 ReLUMax를 도입하여 이 기능의 신뢰성을 향상시키고자 했습니다.

- **Technical Details**: ReLUMax는 트랜지언트 fault(일시적인 오류)에 대한 회복력을 높이기 위해 설계된 단순하고 효과적인 활성화 함수로, 기존 아키텍처에 원활하게 통합될 수 있습니다. 기존의 하드닝 기법들과 비교하여, ReLUMax는 훈련 과정 동안 동적으로 최적의 클리핑 값을 계산하고, 추론 시 클리핑 및 오류 정정을 수행합니다.

- **Performance Highlights**: 실험 결과 ReLUMax는 트랜지언트 오류가 발생해도 높은 정확도를 유지하고, 특히 에러가 심각한 경우에도 예측 신뢰도를 개선하는 효과를 보였습니다. 이는 자율주행시스템의 신뢰성 향상에 기여하게 됩니다.



### VLM-KD: Knowledge Distillation from VLM for Long-Tail Visual Recognition (https://arxiv.org/abs/2408.16930)
- **What's New**: 이번 논문은 오프-the-shelf 비전-언어 모델(vision-language model, VLM)로부터 지식을 증류하는 새로운 메소드를 소개합니다. 이 방식은 기존의 비전 전용_teacher 모델에서의 감독 이외에 새로운 감독을 제공하며, 다양한 벤치마크 데이터셋에서 효과성을 입증합니다.

- **Technical Details**: VLM-KD라는 방법론을 통해 우리의 접근 방식은 비전-언어 모델에서 생성된 텍스트 감독(text supervision)을 이미지 분류기(image classifier)로 통합하는 데 중점을 둡니다. 이는 텍스트 설명을 포함해 다양한 데이터 유형의 정보를 활용하여 딥러닝 모델의 학습을 향상시킵니다.

- **Performance Highlights**: VLM-KD 접근 방식은 여러 현대 장기 꼬리(long-tail) 분류기와의 비교에서 여러 첨단 상태의 분류기(state-of-the-art classifiers)를 초월하는 성능을 발휘했습니다. 우리의 방법론은 추가적인 텍스트 설명이 더해질 때 더욱 큰 이점을 제공하며, 각 구성 요소를 평가하기 위한 철저한 블레이션 연구(ablation study)를 수행하여 성능을 검증했습니다.



### Enhancing Autism Spectrum Disorder Early Detection with the Parent-Child Dyads Block-Play Protocol and an Attention-enhanced GCN-xLSTM Hybrid Deep Learning Framework (https://arxiv.org/abs/2408.16924)
Comments:
          18 pages, 8 figures, and 4 tables

- **What's New**: 이번 연구에서는 자폐 스펙트럼 장애 (ASD)의 조기 발견을 위한 혁신적인 접근 방식을 제안합니다. 여기에는 부모-자녀 쌍의 블록 놀이(Parent-Child Dyads Block-Play, PCB) 프로토콜과 대규모 비디오 데이터셋, 그리고 하이브리드 심층 학습 프레임워크인 2sGCN-AxLSTM이 포함되어 있습니다.

- **Technical Details**: PCB 프로토콜은 ASD와 전형적으로 발달하는 유아를 구분 짓는 행동 양상을 인식하기 위해 동작 분석을 도와줍니다. 이 연구에서 다양한 참여자의 비디오 기록을 포함한 PCB4ASD-ED 데이터셋을 사용하여, 2sGCN-AxLSTM(투 스트림 그래프 컨볼루션 네트워크와 주의력 강화 xLSTM을 통합한 하이브리드 심층 학습 모델)을 통해 상위 신체 및 머리 움직임과 관련된 공간적 특징을 추출했습니다.

- **Performance Highlights**: 이 접근 방식은 자폐아 진단의 정확도를 89.6%로 향상시켜 동적인 인간 행동 패턴을 효과적으로 분석할 수 있는 잠재력을 보여줍니다.



### Ig3D: Integrating 3D Face Representations in Facial Expression Inferenc (https://arxiv.org/abs/2408.16907)
Comments:
          Accepted by ECCVW 2024

- **What's New**: 본 연구는 단일 이미지로부터 3D 얼굴 형상을 재구성하는 새로운 접근 방식을 제시하여 facial expression inference (FEI) 분야에 통합된 3D 표현을 활용하여 감정 분류 및 Valence-Arousal (VA) 추정 작업에서의 성능을 향상시키는 방법을 탐구합니다.

- **Technical Details**: 여기서는 3D Morphable Model (3DMM) 기반의 두 가지 얼굴 표현 모델, SMIRK와 EMOCA의 성능을 비교하고, 2D 추론 프레임워크와의 데이터 통합을 위한 중간융합(intermediate fusion) 및 후기융합(late fusion) 아키텍처를 분석합니다.

- **Performance Highlights**: 제안된 방법은 AffectNet VA 추정 및 RAF-DB 분류 작업에서 최신 기법을 초과한 성과를 달성하며, 다양한 감정 추론 작업에서 기존 방법의 성능을 보완할 수 있는 가능성을 보여줍니다.



### Tex-ViT: A Generalizable, Robust, Texture-based dual-branch cross-attention deepfake detector (https://arxiv.org/abs/2408.16892)
- **What's New**: 이 논문에서는 GAN을 활용하여 고도로 사실적인 얼굴 수정 콘텐츠인 Deepfake에 대해 연구하였으며, CNN의 취약성을 극복하기 위해 Tex-ViT(Texture-Vision Transformer)라는 새로운 모델을 제안했습니다.

- **Technical Details**: Tex-ViT 모델은 전통적인 ResNet 특성과 텍스처 모듈을 결합하여, ResNet의 다운 샘플링 작업 전에 텍스처 모듈이 병렬로 작동합니다. 이 텍스처 모듈은 교차 주의 기반 비전 트랜스포머의 이중 분기(input을 제공합니다). 이 모델은 조작된 샘플에서 전역 텍스처 모듈을 향상시켜 특징 맵 상관관계를 추출하는 데 집중합니다.

- **Performance Highlights**: Tex-ViT 모델은 다양한 FF++ 데이터셋(DF, f2f, FS, NT) 및 GAN 데이터셋에서 실험을 수행하였고, 교차 도메인 상황에서 98%의 정확도로 가장 진보된 모델을 초월하여 조작된 샘플의 구별 가능한 텍스처 특성을 학습할 수 있음을 증명했습니다.



### FineFACE: Fair Facial Attribute Classification Leveraging Fine-grained Features (https://arxiv.org/abs/2408.16881)
- **What's New**: 이 논문은 얼굴 속성 분류(Facial Attribute Classification)에서의 공정성을 향상시키기 위한 새로운 접근 방식을 제안합니다. 기존의 방법들이 인구 통계적 주석을 필요로 하는 것과 달리, 이 방법은 그러한 주석 없이 서로 다른 인구 집단 간의 공정성과 정확성을 모두 개선할 수 있도록 설계되었습니다.

- **Technical Details**: FineFACE 모델은 얕은 층부터 깊은 층까지의 CNN 레이어에서 로컬(local) 및 세부적인 특성과 높은 수준의 세멘틱(semantic) 특성을 통합하는 크로스 레이어 상호 주의 학습(Cross-layer Mutual Attention Learning) 기술을 사용하여 세분화된(fine-grained) 분석을 수행합니다. 이를 통해 얼굴의 측면 윤곽, 색상 및 구조를 포함한 다양한 고유한 특성을 학습하여, 다 인종 인구 집단의 얼굴 속성을 보다 정확히 분류할 수 있습니다.

- **Performance Highlights**: FineFACE 모델은 기존의 최첨단(SOTA) 편향 완화 기술에 비해 정확성을 1.32%에서 1.74% 증가시켰고, 공정성은 67%에서 83.6% 향상시켰습니다. 이 방법은 공정성과 정확성 간의 파레토 효율적인 균형을 달성하여 다양한 하위 분류 작업에 적용 가능하다는 점에서도 강점을 보입니다.



### MSLIQA: Enhancing Learning Representations for Image Quality Assessment through Multi-Scale Learning (https://arxiv.org/abs/2408.16879)
- **What's New**: 본 논문은 No-Reference Image Quality Assessment (NR-IQA) 모델의 성능을 향상시키기 위해 새로운 augmentation 전략을 도입하였으며, 이를 통해 기존 모델에서 28% 가까운 성능 향상을 달성하였습니다.

- **Technical Details**: NR-IQA 시스템의 경량화를 위해 MobileNetV3 (CNN 네트워크)를 기본으로 사용하였고, 다양한 크기와 줌 수준의 cropping 및 resizing 기법을 적용하여 다중 작업 학습을 수행하였습니다. 이 접근법은 Test-Time Augmentation (TTA)을 통해 성능을 더욱 향상시켰으며, 모델의 일반화 능력을 개선하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 경량화된 모델이 기존의 최첨단 NR-IQA 모델들과 비슷한 성능을 나타내었으며, 추가적인 손실 함수나 세부 조정 없이도 성능이 개선되었습니다. 여러 데이터셋에서 실험한 결과, 모델의 정확성이 증가하였고, 특히 TTA 기법을 적용하였을 때 성능이 현저히 향상되었습니다.



### GameIR: A Large-Scale Synthesized Ground-Truth Dataset for Image Restoration over Gaming Conten (https://arxiv.org/abs/2408.16866)
- **What's New**: 이 논문에서는 GameIR라는 대규모 고품질 컴퓨터 합성 정답 데이터셋을 개발하였습니다. 이 데이터셋은 게임 콘텐츠에 대한 이미지 복원 방법을 연구하는 데 도움을 주기 위해 만들어졌습니다. 특히, 이 데이터셋은 두 가지 응용 프로그램, 즉 후처리 렌더링을 이용한 초고해상도(super-resolution)와 새로운 뷰 합성(novel view synthesis)을 지원합니다.

- **Technical Details**: GameIR 데이터셋은 720p 및 1440p에서 렌더링된 640 개 비디오에서 derived된 19200개의 LR-HR 짝 데이터와 6 카메라 뷰를 가진 960개 비디오에서 57,600개의 HR 프레임을 포함하고 있습니다. 또한, 후처리 렌더링 단계에서의 GBuffers 데이터(세분화 맵 및 깊이 맵)도 제공하고 있으며, 이를 통해 복원 성능을 개선할 수 있습니다.

- **Performance Highlights**: 여러 최신 알고리즘(SOTA super-resolution algorithms 및 NeRF 기반 NVS 알고리즘)에 대해 평가지표를 테스트한 결과, GameIR 데이터가 게임 콘텐츠의 복원 성능 향상에 효과적임을 입증하였습니다. 추가적으로, GBuffers를 활용하여 초고해상도 및 NVS에서 성능을 더욱 개선할 수 있는 가능성을 평가하였습니다.



### Enabling Local Editing in Diffusion Models by Joint and Individual Component Analysis (https://arxiv.org/abs/2408.16845)
Comments:
          Code available here: this https URL

- **What's New**: 이번 논문에서는 Denoising Network의 잠재 의미를 분리하여 이미지의 지역적 조작(local image manipulation)을 가능하게 하는 새로운 방법을 제시합니다. 이전 연구들은 주로 전역적인 속성(global attributes) 발견에 초점을 맞추었으나, 본 연구는 이러한 한계를 극복하고자 한다.

- **Technical Details**: 이 논문에서 제안하는 방법은 Denoising Network의 Jacobian을 이용하여 특정 관심지역과 그에 해당하는 잠재 공간의 서브스페이스(subspace) 간의 관계를 설정하고, 해당 서브스페이스의 공동 및 개별 구성 요소를 분리(disentangle)합니다. 이를 통해 얻어진 잠재 방향(latent directions)은 지역적 이미지 조작에 적용될 수 있습니다. 실험 방법으로는 SVD와 JIVE 알고리즘을 활용하여 이 과정을 세밀히 설명합니다.

- **Performance Highlights**: 실험 결과, 해당 방법은 다양한 데이터셋에서 더 자세하고 정확한 세밀한 수정이 가능함을 보여 주며, 기존 최첨단 방법들보다 뛰어난 성능을 나타냅니다. 특히, 원하는 속성을 좀 더 국소화(localized)하여 극적인 시각적 수정 효과를 가져옵니다.



### Fluent and Accurate Image Captioning with a Self-Trained Reward Mod (https://arxiv.org/abs/2408.16827)
Comments:
          ICPR 2024

- **What's New**: 본 논문에서는 Self-Cap이라는 새로운 이미지 캡셔닝 접근 방식을 제안합니다. 이 방법은 자가 생성된 부정 표본을 바탕으로 한 학습 가능한 보상 모델을 기반으로 하며, 이미지와의 일관성에 따라 캡션을 구분할 수 있습니다.

- **Technical Details**: Self-Cap은 fine-tuned된 대조적 이미지-텍스트 모델을 사용하여 캡션의 정확성을 증진시키고, CLIP 기반 보상으로 인한 일반적인 왜곡을 피하도록 훈련됩니다. 부정 표본은 동결된 캡셔너로부터 직접 통합되어 생성됩니다.

- **Performance Highlights**: 실험 결과, Self-Cap은 COCO 데이터셋에서 탁월한 성능을 보여주었으며, 다양한 백본(backbone)에서의 강력한 성능을 입증했습니다. 또한, CC3M, nocaps, VizWiz와 같은 다른 데이터셋에서도 제로샷(zero-shot) 성능을 검증했습니다.



### See or Guess: Counterfactually Regularized Image Captioning (https://arxiv.org/abs/2408.16809)
Comments:
          Accepted by ACM MM 2024

- **What's New**: 이번 논문에서는 인과 추론(causal inference)을 적용하여 기존의 이미지 캡셔닝(image captioning) 모델들이 개입적(interventional) 작업을 더 잘 수행할 수 있도록 하는 새로운 프레임워크를 제안합니다. 제안된 접근법은 전체 효과(total effect) 또는 자연적 직접 효과(natural direct effect)를 활용한 두 가지 변형이 포함되어 있습니다.

- **Technical Details**: 본 논문에서는 이미지 캡셔닝 작업을 순차적 인과 그래프(sequential causal graph)로 정형화합니다. 여기서 생성되는 각 단어는 이전 단어 및 이미지에 의해 영향을 받습니다. 이 인과 그래프를 기반으로 자가 규제(counterfactual regularization) 이미지 캡셔닝 프레임워크를 제안하며, 신뢰성을 높이는 동시에 hallucinatory 출력(hallucinations)을 줄이는 방법을 모색합니다.

- **Performance Highlights**: 다양한 데이터셋을 통한 광범위한 실험 결과, 제안된 방법은 hallucinatory 현상을 효과적으로 줄이고 이미지에 대한 모델의 충실도를 향상시킴을 입증했습니다. 또한, 소규모 및 대규모 이미지-텍스트 모델 모두에서 높은 이식성을 보여 주목할 만한 성과를 달성하였습니다.



### STEREO: Towards Adversarially Robust Concept Erasing from Text-to-Image Generation Models (https://arxiv.org/abs/2408.16807)
Comments:
          Project Page: this https URL

- **What's New**: 이번 연구에서는 대규모 텍스트-이미지 생성(T2IG) 모델에서 개념 삭제의 안전성을 높이기 위해 STEREO라는 새로운 두 단계 접근법을 제안합니다. 기존의 개념 삭제 방식이 적대적 공격에 취약하다는 문제를 해결하기 위해, 강력한 적대적 프롬프트를 탐색하고 동시에 유틸리티 저하를 최소화하며 개념을 삭제하는 방법입니다.

- **Technical Details**: 제안하는 STEREO 접근 방식은 두 단계로 구성됩니다. 첫 번째 단계(STE: Search Thoroughly Enough)는 강력하고 다양한 적대적 프롬프트를 충분히 찾기 위한 것으로, 적대적 훈련(Adversarial Training, AT) 원칙을 활용하여 이루어집니다. 두 번째 단계(REO: Robustly Erase Once)는 앵커 개념 기반의 조합 목표를 도입하여 원래 모델에서 목표 개념을 단번에 강력하게 삭제하면서 모델 유틸리티 저하를 최소화하는 방법입니다. 이 방식은 기존의 개념 삭제 방법들과의 비교를 통해 더 나은 강인성-유틸리티 균형을 제공합니다.

- **Performance Highlights**: STEREO 접근 방식은 네 가지 최첨단 개념 삭제 방법에 대해 세 가지 적대적 공격 하에서 벤치마킹을 수행하였으며, 이를 통해 제공된 모델의 안정성이 향상됨을 보여주었습니다. 특히, STEREO는 삭제된 개념을 쉽게 재생성할 수 있는 적대적 프롬프트에 대해 높은 강인성을 유지하면서 유용성을 최대한 유지하는데 성공했습니다.



### Generative AI Enables Medical Image Segmentation in Ultra Low-Data Regimes (https://arxiv.org/abs/2408.17421)
- **What's New**: 이 논문에서는 ultra low-data regimes에서 사용될 수 있는 새로운 생성적 심층 학습 프레임워크인 GenSeg를 소개합니다. GenSeg는 고급 품질의 쌍으로 된 세분화 마스크와 의료 이미지를 생성하여 데이터가 부족한 환경에서 강인한 모델을 훈련하는 데 도움을 줍니다.

- **Technical Details**: GenSeg는 다단계 최적화(multi-level optimization) 접근 방식을 사용하여 데이터 생성 프로세스와 세분화 모델의 성능 간의 직접적인 연계를 형성합니다. 이 구조를 통해 생성된 데이터는 세분화 성능을 향상시키는 구체적인 목표를 가집니다. GenSeg는 기존 방법보다 8배에서 20배 적은 훈련 데이터를 사용하여 동등한 성과를 달성할 수 있습니다.

- **Performance Highlights**: GenSeg는 9개 세분화 작업과 16개 데이터셋에 걸쳐 우수한 일반화 성능을 보였습니다. UNet 및 DeepLab 모델이 포함된 여러 세분화 모델에서 성능 향상 폭이 10-20%에 달하며, 이는 데이터 부족 시나리오에서 큰 개선을 나타냅니다.



### Investigating Neuron Ablation in Attention Heads: The Case for Peak Activation Centering (https://arxiv.org/abs/2408.17322)
Comments:
          9 pages, 2 figures, XAI World Conference 2024 Late-Breaking Work

- **What's New**: 본 연구에서는 언어 모델과 비전 트랜스포머에서 신경 세포의 활성화(ablation) 방법을 비교하며, 새로운 기법인 'peak ablation'을 제안합니다.

- **Technical Details**: 신경 활성화 양식을 각각 다르게 조정하는 네 가지 방법(Zero Ablation, Mean Ablation, Activation Resampling, Peak Ablation)을 사용하여 성능 저하 정도를 실험적으로 분석하였습니다. 특히, 'peak ablation' 기법에서는 모달 활성화(modal activation)를 기반으로 한 새로운 접근법을 사용합니다.

- **Performance Highlights**: 각기 다른 방법에서 모델 성능 저하가 가장 적은 방법을 식별하였으며, 일반적으로 resampling이 가장 큰 성능 저하를 발생시킨다는 결과를 도출했습니다.



### Structuring a Training Strategy to Robustify Perception Models with Realistic Image Augmentations (https://arxiv.org/abs/2408.17311)
- **What's New**: 본 논문은 자율 시스템을 위한 기계 학습 기반의 인식 모델의 약점을 해결하기 위한 새로운 방법론을 소개하며, 특히 도전적인 작업 설계 분야(Operational Design Domains, ODD)에서의 모델 강건성과 성능을 향상시키기 위한 이미지 증강(aegmentation) 적용을 다룹니다.

- **Technical Details**: 이 연구에서는 사용자 정의 물리 기반 증강 함수인 Augmentation Function을 활용하여 다양한 ODD 시나리오를 시뮬레이션하는 현실적인 훈련 데이터를 생성하고, Hyperparameter Optimization과 Latent Space 최적화를 통해 증강 매개변수를 세밀하게 조정하였습니다. 이를 통해 ML 모델의 약점을 식별하고 적절한 증강 방법을 선택하여 효과적인 훈련 전략을 수립합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법론을 통해 일반적으로 사용되는 지표인 평균 평균 정밀도(mean Average Precision, mAP) 및 평균 교차 비율(mean Intersection over Union, mIoU)에서 성능 향상을 입증하였습니다. 본 연구는 자율주행 기능의 안전성과 신뢰성을 향상시키기 위한 커스터마이즈된 증강을 통합하는 것의 중요성을 강조합니다.



### A nonlinear elasticity model in computer vision (https://arxiv.org/abs/2408.17237)
- **What's New**: 본 논문은 저자들이 이전에 제안한 비선형 탄성 모델을 분석하여 두 이미지를 비교하는 새로운 접근 방식을 보여줍니다. 이 모델은 두 이미지를 영역 보존 변환으로 연결하며, 기능적 최소화 문제에 대한 존재성을 입증했습니다.

- **Technical Details**: 연구는 두 이미지 P1과 P2의 패턴을 비교하면서, 자연스러운 coercivity와 polyconvexity 조건 하에서, 집합의 경계를 보존하는 homeomorphisms의 최적 변환을 찾고 있습니다. 비선형 탄성 모델은 L∞ 환경을 통해 뉴멕시코와 M+n×n 관련성을 끌어올리며, 조건부 최소화 문제를 해결합니다.

- **Performance Highlights**: 연구는 이미지 비교 접근 방식에 대한 통찰을 제공하며, 비선형 탄성의 우수한 성능을 입증했습니다. 특정 예시는 선형 맵에 의해 연결될 때 최소화가 이루어지도록 보장하며, 기존 방법보다 큰 변형을 허용하고 회전 불변성을 존중하는 장점을 가지고 있습니다.



### Sparse Uncertainty-Informed Sampling from Federated Streaming Data (https://arxiv.org/abs/2408.17108)
Comments:
          Preprint, 6 pages, 3 figures, Accepted for ESANN 2024

- **What's New**: 이 연구는 제한된 자원과 희소한 라벨 데이터가 있는 연합 클라이언트 시스템에서 비동일분포(non-I.I.D.) 데이터 스트림 샘플링을 위한 수치적 안정성과 계산 효율성을 갖춘 새로운 접근 방식을 제시합니다.

- **Technical Details**: 제안된 방법은 고객 모델을 최적화하기 위해Streaming Active Learning을 기반으로 한 쿼리 전략을 활용하며, 자원 효율성 및 수치적 안정성을 우선시합니다. 이 방법은 라벨링 예산을 고려하여 비동일분포 데이터 스트림에서 가장 가치 있는 샘플을 선택합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존 전략에 비해 트레이닝 배치의 다양성이 향상되었고, 대규모 데이터 스트림에서 수치적 강건성이 개선되었습니다.



### FissionVAE: Federated Non-IID Image Generation with Latent Space and Decoder Decomposition (https://arxiv.org/abs/2408.17090)
- **What's New**: 본 논문에서는 비동일 분포의 데이터 환경에서의 연합 학습( federated learning )에서 Variational Autoencoder (VAE)를 활용하는 새로운 모델인 FissionVAE를 소개합니다. 이 모델은 각 클라이언트 그룹의 데이터 특성에 맞춰 라텐트 공간을 분해하고 개별 디코더 브랜치를 생성하여 맞춤형 학습을 가능하게 합니다.

- **Technical Details**: FissionVAE는 비동일 분포(non-IID) 데이터를 고려하여 라텐트 공간을 특성에 따라 구분하고, 각 클라이언트 그룹에 맞는 디코더 브랜치를 구축합니다. 이로 인해 여러 이미지 유형 간의 라텐트 공간 해석 문제를 낮출 수 있으며, 계층적 VAE 아키텍처와 다형성 디코더 아키텍처를 도입하여 모델의 유연성을 개선합니다.

- **Performance Highlights**: FissionVAE는 MNIST와 FashionMNIST의 조합 및 만화, 인간 얼굴, 동물, 해양 선박, 지구의 원격 감지 이미지로 구성된 복합 데이터셋에서 기존의 기반 연합 VAE 모델에 비해 생성 품질이 크게 향상됨을 실험적으로 입증하였습니다.



### Approximately Invertible Neural Network for Learned Image Compression (https://arxiv.org/abs/2408.17073)
- **What's New**: 최근 학습된 이미지 압축 기술이 발전하면서, 본 논문에서는 Approximately Invertible Neural Network (A-INN)이라는 새로운 프레임워크를 제안하고 있습니다. A-INN은 기존의 비가역적인 인코더와 디코더 대신, 역가변 신경망(Invertible Neural Network, INN)을 활용하여 이미지 압축 과정에서의 복원성을 향상시키는 것을 목표로 합니다.

- **Technical Details**: A-INN 프레임워크는 인코딩 및 디코딩 과정에서 손실 압축 손실과 잡음의 영향을 최소화하기 위해 프로그레시브 디노이징 모듈(Progressive Denoising Module, PDM)을 적용합니다. 또한 고차원 특징을 저차원 특징으로부터 학습하는 Cascaded Feature Recovery Module (CFRM)과 고주파 정보 손실을 완화하는 Frequency-enhanced Decomposition and Synthesis Module (FDSM)도 개발되었습니다.

- **Performance Highlights**: 실험 결과, A-INN은 Kodak, Tecnick, CLIC 및 CLIC Professional Test 데이터셋에서 기존의 인버터블 신경망 기반의 압축 방법과 학습된 이미지 압축 방법들을 초월하는 성능을 보여주었습니다. 각 구성 모듈의 효과가 검증된 ablation 연구도 수행되었습니다.



### Disease Classification and Impact of Pretrained Deep Convolution Neural Networks on Diverse Medical Imaging Datasets across Imaging Modalities (https://arxiv.org/abs/2408.17011)
Comments:
          15 pages, 3 figures, 4 tables

- **What's New**: 이 논문은 다양한 의료 이미징 데이터셋에서 사전 훈련된 딥 컨볼루션 신경망과 전이 학습을 활용한 이진 및 다중 클래스 분류의 아키텍처를 조사합니다.

- **Technical Details**: 연구는 VGG, ResNet, Inception, Xception 및 DenseNet과 같은 10개의 서로 다른 DCNN (Deep Convolutional Neural Networks) 아키텍처를 사용했으며, 사전 훈련 및 랜덤 초기화 모드에서 모델을 훈련 및 평가했습니다. 데이터셋에는 CXR, OCT 및 WSI가 포함됩니다.

- **Performance Highlights**: 사전 훈련 모델을 고정된 특징 추출기로 사용할 경우 성능이 좋지 않았으며, 반대로 병리학적 현미경 WSI에서 더 나은 성능을 보였습니다. 네트워크 아키텍처의 성능은 데이터셋에 따라 달라졌으며, 이는 특정 모달리티의 모델 성능이 동일한 분야 내 다른 모달리티에 대해 결론적이지 않음을 나타냅니다.



### Efficient Camera Exposure Control for Visual Odometry via Deep Reinforcement Learning (https://arxiv.org/abs/2408.17005)
Comments:
          8 pages, 7 figures

- **What's New**: 이 연구는 특정 조명 변화가 있는 환경에서도 비주얼 오도메트리 시스템(VO)의 이미징 성능을 향상시키기 위해 깊은 강화 학습(Deep Reinforcement Learning, DRL) 프레임워크를 활용하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: DRL 기반의 카메라 노출 제어 방법을 개발하여 조명 변화가 극심한 환경에서도 VO 시스템의 안정성을 높였습니다. 경량 이미지 시뮬레이터를 사용하여 훈련 과정을 지원하고, 다양한 보상 함수(reward functions)를 통해 DRL 에이전트를 훈련시킵니다. 이러한 접근을 통해 드론의 하드웨어 또는 실제 환경과의 직접 상호작용 없이 오프라인 훈련이 가능합니다.

- **Performance Highlights**: 실험 결과, 노출 제어 에이전트가 평균 1.58 ms/frame으로 CPU에서 뛰어난 효율성을 도출하였으며, 전통적인 피드백 제어 방식보다 더 빠르게 반응합니다. 적절한 보상 함수를 선택하여 에이전트가 움직임의 경향을 이해하고 미래 조명 변화를 예측할 수 있도록 하여 VO 시스템의 정확도를 크게 향상시켰습니다.



### LV-UNet: A Lightweight and Vanilla Model for Medical Image Segmentation (https://arxiv.org/abs/2408.16886)
- **What's New**: 본 논문에서는 의료 영상 분할을 위한 경량의 UNet 변형인 LV-UNet 모델을 제안합니다. 이 모델은 사전 훈련된 MobileNetv3-Large 모델을 활용하고, 매끄러운 모듈(fusible modules)을 도입하여 경량화 및 실시간 성능을 달성합니다.

- **Technical Details**: LV-UNet 모델은 인코더와 디코더로 구성되며, 인코더는 MobileNetv3-Large의 사전 훈련된 블록을 포함합니다. 이 모델은 훈련 단계에서 보다 복잡한 구조를 사용하여 학습을 최적화하고, 추론 단계에서 파라미터 수와 계산량을 줄입니다.

- **Performance Highlights**: ISIC 2016, BUSI, CVC-ClinicDB, CVC-ColonDB 및 Kvair-SEG 데이터셋에서 실험을 수행한 결과, 기존의 최신 기술 및 고전 모델들에 비해 경쟁력 있는 성능을 보였습니다.



### Revising Multimodal VAEs with Diffusion Decoders (https://arxiv.org/abs/2408.16883)
- **What's New**: 이번 연구는 전통적인 Multimodal VAE 프레임워크의 한계를 극복하기 위해 발전된 Diffusion Decoder를 도입하여 이미지 모달리티의 생성 품질을 향상시키고, 피드포워드 디코더에 의존하는 다른 모달리티의 성능에도 긍정적인 영향을 미친다는 점에서 혁신적입니다.

- **Technical Details**: 우리의 모델은 Diffusion Autoencoder와 표준 VAE의 하이브리드 구조로 설계되어 있으며, Joint Representation을 효과적으로 활용하여 고품질 샘플을 생성합니다. 또한, Product of Experts (PoE) 기법을 사용하여 서로 다른 모달리티 간의 관계를 포착합니다. 이 모델은 조건부 및 비조건부 생성 작업 모두에 유연하게 대응할 수 있습니다.

- **Performance Highlights**: 제안된 모델은 다양한 데이터셋에서 기존의 Multimodal VAE 모델들에 비해 더 높은 일관성과 우수한 생성 품질을 보여주며, 고차원 데이터에도 효과적으로 확장 가능한 성능을 자랑합니다.



### Comparative Analysis of Transfer Learning Models for Breast Cancer Classification (https://arxiv.org/abs/2408.16859)
- **What's New**: 이번 연구는 유방암의 조기 및 정확한 진단을 위한 병리 이미지 분류의 중요성을 강조하고, 다양한 심층 학습 모델을 사용해 침습성 유관암(Invasive Ductal Carcinoma, IDC)과 비IDC를 구분하는 효율성을 분석합니다.

- **Technical Details**: 모델로는 ResNet-50, DenseNet-121, ResNeXt-50, Vision Transformer (ViT), GoogLeNet (Inception v3), EfficientNet, MobileNet, SqueezeNet 등 8가지가 비교되었습니다. 이 연구에서 사용된 데이터셋은 277,524개의 이미지 패치로, 각 모델의 성능을 포괄적으로 평가했습니다. 특히 ViT 모델은 주의(attention) 기반 메커니즘의 뛰어난 효능을 보였고, 93%의 검증 정확도를 성취하여 전통적인 CNN들보다 우수한 성능을 보였습니다.

- **Performance Highlights**: 본 연구는 ViT 모델의 주목할 만한 성과와 함께, 최신 머신러닝 방법이 임상 환경에서 유방암 진단의 정확도와 효율성을 크게 향상시킬 수 있는 가능성을 보여주고 있습니다.



### A Permuted Autoregressive Approach to Word-Level Recognition for Urdu Digital Tex (https://arxiv.org/abs/2408.15119)
- **What's New**: 이 연구는 디지털 우르두 텍스트 인식을 위해 특별히 설계된 새로운 단어 수준의 Optical Character Recognition (OCR) 모델을 소개합니다. 이 모델은 다양한 텍스트 스타일과 글꼴, 변형 문제를 해결하기 위해 transformer 기반 아키텍처 및 attention 메커니즘을 활용합니다.

- **Technical Details**: 모델은 permuted autoregressive sequence (PARSeq) 아키텍처를 사용하여 맥락 인지 추론 및 여러 토큰 순열을 통한 반복적 개선을 가능하게 하여 성능을 향상시킵니다. 이를 통해 우르두 스크립트에서 일반적으로 발생하는 문자 재정렬 및 겹치는 문자를 효과적으로 관리합니다.

- **Performance Highlights**: 약 160,000개의 우르두 텍스트 이미지로 훈련된 이 모델은 CER(Character Error Rate) 0.178을 달성하며 우르두 스크립트의 복잡성을 잘 포착합니다. 비록 몇몇 텍스트 변형을 처리하는 데 지속적인 어려움이 있지만, 이 모델은 실제 응용에서 높은 정확성과 효율성을 보여줍니다.



New uploads on arXiv(cs.AI)

### Exploring the Effect of Explanation Content and Format on User Comprehension and Trus (https://arxiv.org/abs/2408.17401)
Comments:
          18 pages

- **What's New**: 이 논문은 블랙박스 AI 모델의 출력을 설명하는 방법에 대한 사용자 이해와 신뢰에 대한 연구를 다룹니다. 특히, SHAP와 occlusion-1이라는 두 가지 설명 방법의 내용을 비교하고, 설명의 형식과 내용이 이해 및 신뢰도에 미치는 영향을 분석합니다.

- **Technical Details**: 연구에서는 사용자 연구를 통해 일반 대중과 의료 훈련을 받은 참가자를 대상으로 XAI 방법의 내용 및 형식에 따라 설명의 이해도와 신뢰도를 측정했습니다. occlusion-1 방법이 SHAP보다 선호되는 경향을 보였으며, 텍스트 형식의 설명이 차트 형식보다 더 높은 주관적 이해 및 신뢰도를 보여주었습니다.

- **Performance Highlights**: 대부분의 경우 occlusion-1 설명이 SHAP 설명에 비해 사용자로부터 더 높은 주관적 이해도와 신뢰를 얻었으며, 특히 텍스트 형식에서 그 차이가 두드러졌습니다. 그러나 객관적인 이해도에는 두 설명 유형 간의 유의미한 차이가 없었습니다.



### Traffic expertise meets residual RL: Knowledge-informed model-based residual reinforcement learning for CAV trajectory contro (https://arxiv.org/abs/2408.17380)
- **What's New**: 이 논문에서는 전문가 지식을 활용하여 모델 기반 잔여 강화 학습(Residual Reinforcement Learning) 프레임워크를 제안하여 학습 효율성을 높이고, 기존의 학습 과정에서 처음부터 시작하는 문제를 피하고자 합니다.

- **Technical Details**: 제안된 프레임워크는 Intelligent Driver Model (IDM)과 신경망(neural networks)을 통합하여 가상 환경 모델을 구성하고, 잔여 다이나믹스를 학습합니다. 이 방법은 전통적 제어 방식과 잔여 RL을 결합하여 샘플 효율성을 개선하고, 복잡한 시나리오에 적응할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식은 기존의 기준 에이전트에 비해 경로 제어에 있어 샘플 효율성, 교통 흐름의 부드러움 및 교통 이동성을 향상시키는 뛰어난 성능을 보였습니다.



### Bridging Domain Knowledge and Process Discovery Using Large Language Models (https://arxiv.org/abs/2408.17316)
Comments:
          This paper is accepted at the AI4BPM 2024 workshop and to be published in their proceedings

- **What's New**: 본 논문은 프로세스 발견(process discovery) 작업에서 도메인 지식(domain knowledge)을 통합하기 위해 Large Language Models (LLMs)을 활용하는 새로운 접근 방식을 제안합니다. 이는 기존의 자동화된 프로세스 발견 방법에서 간과되었던 전문가의 통찰 및 세부 프로세스 문서와 같은 정보를 효율적으로 활용할 수 있게 합니다.

- **Technical Details**: 이 연구는 LLMs로부터 파생된 규칙(rules)을 사용하여 모델 구성을 안내합니다. 이러한 방식은 도메인 지식과 실제 프로세스 실행 간의 일치를 보장하며, 자연어로 표현된 프로세스 지식과 견고한 프로세스 모델 발견을 결합하는 브리지 역할을 합니다.

- **Performance Highlights**: UWV 직원 보험 기관과의 사례 연구를 통해 프레임워크의 실용성과 효과성을 검증하며, 프로세스 분석 작업에서의 이점을 강조했습니다.



### Flexible and Effective Mixing of Large Language Models into a Mixture of Domain Experts (https://arxiv.org/abs/2408.17280)
- **What's New**: 본 연구는 훈련된 모델로부터 저비용 Mixture-of-Domain-Experts (MOE) 를 생성할 수 있는 툴킷(toolkit) 을 제시합니다. 이 툴킷은 모델 또는 어댑터(adapters) 로부터 혼합을 생성하는 데 사용될 수 있으며, 결과 MOE의 아키텍처 정의에 대한 지침 및 광범위한 테스트를 제공합니다.

- **Technical Details**: 본 툴킷은 다양한 방식을 통해 훈련된 모델을 Mixture of Domain Experts MOE에 활용할 수 있도록 유연성을 제공합니다. 특히, 전문가 모듈 또는 라우터(router)를 훈련하지 않고도 MOE를 생성할 수 있는 방법도 지원합니다. Gate-less MOE 구조는 라우터 기반 아키텍처에 비해 경쟁력이 있으며, 저비용으로 생산 가능합니다. Noisy MOE는 Gate-less MOE와 유사한 성능을 보여주며, 훈련이 필요하지 않으면서도 낮은 추론(inference) 비용을 제공합니다.

- **Performance Highlights**: Gate-less MOE 아키텍처는 소수의 고품질 전문가가 있을 경우 최적의 솔루션이 될 수 있으며, 다양한 실험을 통해 이런 구조가 효과적임을 입증했습니다. 또, Gate-less MOE는 라우터 기반 아키텍처보다 더 경쟁력이 있으며, 비용 측면에서 이점이 있습니다.



### A methodological framework for Resilience as a Service (RaaS) in multimodal urban transportation networks (https://arxiv.org/abs/2408.17233)
- **What's New**: 이번 연구는 대중교통 시스템의 서비스 중단을 관리하기 위한 복원력 전략(Resilience as a Service, RaaS)을 탐구합니다. 새로운 최적화 모델을 개발하여 자원을 효과적으로 할당하고 운영자와 승객에게 비용을 최소화하는 방안을 제안합니다.

- **Technical Details**: 제안된 모델은 버스, 택시, 자동화 밴 등의 다양한 교통 수단을 포함하며, 서비스 중단 시 철도 대체 수단으로서의 위치, 용량, 속도, 이용 가능성을 평가합니다. 이를 통해 서비스 연속성을 유지하기 위해 가장 적합한 차량을 배치할 수 있습니다. 마이크로 시뮬레이션 기반의 사례 연구를 진행하여 기존의 버스 브리징 및 비상 차고 솔루션과 비교하였습니다.

- **Performance Highlights**: 연구 결과, 제안된 모델은 비용을 최소화하고 이해관계자의 만족도를 향상시키는 성과를 보였으며, 대중교통 중단 시 관리 최적화를 통해 효율성을 개선하였습니다.



### Towards Symbolic XAI -- Explanation Through Human Understandable Logical Relationships Between Features (https://arxiv.org/abs/2408.17198)
- **What's New**: 이 논문에서는 설명 가능한 인공지능(Explainable Artificial Intelligence, XAI)의 새로운 접근 방식을 제안합니다. 기존의 XAI 방법들이 단일 또는 다수의 입력 특성을 강조하는 heatmap과 같은 형식으로 한 수준의 추상화를 제공한 반면, 저자들은 모델의 추상적인 추론과 문제 해결 전략 또한 중요하다고 주장합니다. 이로 인해 'Symbolic XAI'라는 프레임워크를 제안하였습니다.

- **Technical Details**: Symbolic XAI 프레임워크는 입력 특성 간의 논리적 관계를 표현하는 기호 쿼리에 의미를 부여하여 모델 예측의 추상적 추론을 포착하는 방법론입니다. 이 방법론은 다중 차원 분해에 의존하여 모델 예측의 차이를 정의하며, GNN-LRP와 같은 높은 차수 전파 기반 방법론 및 일반적인 perturbation 기반 설명 방법론을 사용할 수 있습니다. 또한, 이 프레임워크는 논리 방정식(composed of a functionally complete set of logical connectives)을 구성하는 쿼리의 관련성을 수치화하여, 사용자 정의가 가능하고 인간이 이해할 수 있는 방식으로 모델 결정 과정을 설명합니다.

- **Performance Highlights**: Symbolic XAI 프레임워크는 자연어 처리(NLP), 비전 분야, 그리고 양자 화학(QC)에서의 실험을 통해 그 효과성을 입증하였습니다. 이 프레임워크를 사용함으로써 단순히 특징 기여도를 넘어서 다양한 특징 사이의 상호작용이 예측에 미치는 영향에 대한 통찰을 제공하며, 사용자가 가진 질문에 대해 맞춤형으로 응답할 수 있는 쿼리를 자동으로 생성하는 방법을 제공합니다.



### Reasoning with maximal consistent signatures (https://arxiv.org/abs/2408.17190)
- **What's New**: 이 논문에서는 Lang과 Marquis의 기억 기반 추론 일반 접근 방식을 구체적으로 분석합니다. 특히, 일관성이 없는 정보로 추론하기 위해 최대 일관성 부분서명을 사용하는 방법을 논의합니다.

- **Technical Details**: 최대 일관성 부분서명(maximal consistent subsignature)을 정의하고, 이는 나머지 명제가 기억됨으로써 일관성을 회복할 수 있는 최대 명제 집합입니다. 이러한 접근법은 Lang과 Marquis의 프레임워크의 특정 인스턴스로, 최대 일관성 부분서명과 최소 비일관성 부분서명의 관계를 깊이 분석합니다. 또한, 최대 일관성 부분서시에 기반한 추론 관계와 비단조(non-monotonic) 추론의 합리성 규칙을 고려합니다.

- **Performance Highlights**: 이 연구는 최대 일관성 부분서명의 특성을 분석하고, 이 구조가 클래식한 히팅 집합 대칭성을 따르며, 비일관성 측정을 위한 새로운 방식도 제시합니다. 특히, 단순한 기억 접근법을 사용하여 최대 일관성 부분서명으로부터 얻어지는 추론 방식이 Priest의 3값 논리와 동등함을 보였습니다.



### Identifying and Clustering Counter Relationships of Team Compositions in PvP Games for Efficient Balance Analysis (https://arxiv.org/abs/2408.17180)
Comments:
          TMLR 09/2024 this https URL

- **What's New**: 이 논문에서는 게임 설정에서 균형을 정량화하는 새로운 방법론을 제시합니다. 특히, PvP 게임에서 팀 구성 간의 힘 관계를 분석할 수 있는 두 가지 고급 지표를 개발했습니다.

- **Technical Details**: 논문에서는 Bradley-Terry 모델과 Siamese neural networks를 결합하여 팀 구성의 힘을 예측하는 방법을 설명합니다. 이 모델은 게임 결과를 기반으로 힘 값을 도출하며, 텍스트의 주요 기법 중 하나로는 deterministic vector quantization이 포함되어 있습니다. 두 가지 새로운 균형 측정 지표인 Top-D Diversity와 Top-B Balance를 정의하여 모호한 승률을 보완합니다.

- **Performance Highlights**: 이 방법론은 Age of Empires II, Hearthstone, Brawl Stars, League of Legends와 같은 인기 온라인 게임에서 검증되었으며, 전통적인 쌍별 승리 예측과 유사한 정확성을 보이면서도 분석의 복잡성을 줄였습니다. 논문은 이 방법이 단순한 게임 외에도 스포츠, 영화 선호도, 동료 평가, 선거 등 다양한 경쟁 시나리오에 적용 가능하다고 강조합니다.



### Strategic Arms with Side Communication Prevail Over Low-Regret MAB Algorithms (https://arxiv.org/abs/2408.17101)
- **What's New**: 이번 연구는 전략적 multi-armed bandit 환경에서 팔(arms)들이 플레이어의 행동에 대한 완전한 정보를 보유할 때, 그들이 거의 모든 가치를 유지하고 플레이어에게 상당한 (선형적) 후회를 남기는 평형 상태를 이룰 수 있음을 보여줍니다.

- **Technical Details**: 팔들이 완전한 정보를 모든 팔에게 공개하지 않더라도 서로 간에 정보를 공유하면 유사한 평형 상태를 달성할 수 있습니다. 이 연구의 주요 도전 과제는 팔들이 진실하게 소통하도록 유도하는 communication protocol을 설계하는 것입니다.

- **Performance Highlights**: 연구 결과, 통신 프로토콜이 잘 설계될 경우, 팔들이 자신들의 가치를 극대화하면서도 플레이어에게는 임의의 후회를 남길 수 있는 환경을 조성할 수 있음을 보여줍니다.



### Beyond Preferences in AI Alignmen (https://arxiv.org/abs/2408.16984)
Comments:
          26 pages (excl. references), 5 figures

- **What's New**: AI 정렬(Alignment) 분야에서 인간의 선호를 바탕으로 한 전통적인 방식인 preferentist 접근법을 비판하고, 이 접근법의 한계와 대안들을 제시하고 있습니다. 또한 AI 시스템의 역할에 맞는 규범적 기준과 관련 이해관계자 간의 합의 필요성을 강조합니다.

- **Technical Details**: AI 정렬의 전통적 접근은 인간의 선호를 기반으로 이러한 선호가 가치관을 잘 나타낸다고 가정합니다. 그러나 현실에서는 선호가 인간의 복잡한 가치관을 포착하지 못하고, 기대 효용 이론(expected utility theory, EUT)이 인간과 AI 모두에게 적합하지 않을 수 있다는 점을 지적합니다.

- **Performance Highlights**: 이 논문은 AI 시스템이 단일 인간의 선호가 아니라 사회적 역할에 적합한 규범적 기준에 맞춰 정렬되어야 하며, 다양한 선호를 조화롭게 수용할 수 있는 가능성을 제시합니다.



### Bridging Episodes and Semantics: A Novel Framework for Long-Form Video Understanding (https://arxiv.org/abs/2408.17443)
Comments:
          Accepted to the EVAL-FoMo Workshop at ECCV'24. Project page: this https URL

- **What's New**: 기존의 연구가 긴 형식의 비디오를 단순히 짧은 비디오의 연장으로 취급하는 반면, 우리는 인간의 인지 방식을 보다 정확하게 반영하는 새로운 접근 방식을 제안합니다. 본 논문에서는 긴 형식 비디오 이해를 위해 BREASE(BRidging Episodes And SEmantics)라는 모델을 도입합니다. 이 모델은 에피소드 기억(Episodic memory) 축적을 시뮬레이션하여 행동 시퀀스를 캡처하고 비디오 전반에 분산된 의미적 지식으로 이를 강화합니다.

- **Technical Details**: BREASE는 두 가지 주요 요소로 구성됩니다. 첫 번째는 에피소드 압축기(Episodic COmpressor, ECO)로, 마이크로부터 반 매크로 수준까지 중요한 표현을 효율적으로 집계합니다. 두 번째는 의미 검색기(Semantics reTRiever, SeTR)로, 집계된 표현을 광범위한 맥락에 집중하여 의미적 정보를 보강하며, 이를 통해 특징 차원을 극적으로 줄이고 관련된 매크로 수준 정보를 유지합니다.

- **Performance Highlights**: BREASE는 여러 긴 형식 비디오 이해 벤치마크에서 최첨단 성능을 달성하며, 제로샷(zero-shot) 및 완전 감독(full-supervised) 설정 모두에서 7.3% 및 14.9%의 성능 향상을 보인 것으로 나타났습니다.



### Advancing Multi-talker ASR Performance with Large Language Models (https://arxiv.org/abs/2408.17431)
Comments:
          8 pages, accepted by IEEE SLT 2024

- **What's New**: 이 논문에서는 다중 발화자(multi-talker)의 음성 인식을 위한 새로운 방법론인 LLM 기반 SOT(Serialized Output Training)를 제안합니다. 이는 자동 음성 인식(ASR) 분야의 과제를 해결하기 위해 프리 트레인(pre-trained)된 음성 인코더와 대형 언어 모델(large language model)을 활용합니다.

- **Technical Details**: 기존의 SOT 방법은 발화 시간에 따라 여러 발화자의 전사를 연결하는 방식입니다. 하지만 본 연구에서는 LLM을 사용하여 긴 맥락을 효과적으로 모델링하고, 다중 발화자 데이터셋에서 적절한 전략으로 파인튜닝(fine-tuning)하여 성능을 극대화합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 시뮬레이션 데이터셋 LibriMix 및 실제 데이터셋 AMI의 평가 세트에서 기존 AED 방법을 초월하는 성과를 보였습니다. 특히, 이전 연구에서 1000배 더 많은 감독 데이터를 사용한 AED 모델보다 우수한 성능을 기록했습니다.



### Open-vocabulary Temporal Action Localization using VLMs (https://arxiv.org/abs/2408.17422)
Comments:
          7 pages, 5 figures, 4 tables. Last updated on August 30th, 2024

- **What's New**: 본 논문에서는 비디오 액션 로컬리제이션(영상 내 특정 행동의 타이밍을 찾기)의 새로운 접근법을 제안합니다. 기존 학습 기반 방법들의 노동 집약적인 주석 작업 없이, Open-vocabulary(개방 어휘) 방식을 채택한 학습 없는(video action localization) 방법론을 도입하였습니다.

- **Technical Details**: 제안된 방법은 iterative visual prompting(반복적인 시각적 자극 기법)을 기반으로 합니다. 영상 프레임을 조합한 이미지를 생성하고, Vision-Language Models(비전-언어 모델)을 활용하여 행동의 시작 및 종료 지점을 나타내는 프레임을 추정합니다. 이 과정은 점진적으로 샘플링 윈도우를 좁혀가며 반복됩니다.

- **Performance Highlights**: 제안된 파이프라인은 평균 프레임 기준(mean-over-frame)에서 60% 이상의 정확도를 달성하며, 기존의 방법보다 나은 성능을 보였습니다. 또한, 이러한 방식은 로봇 교육을 포함해 다양한 연구 분야에서의 응용 가능성을 시사합니다.



### Getting Inspiration for Feature Elicitation: App Store- vs. LLM-based Approach (https://arxiv.org/abs/2408.17404)
Comments:
          To Appear In Proceedings of 39th IEEE/ACM International Conference on Automated Software Engineering (ASE 2024)

- **What's New**: 최근 Generative AI의 발전으로 인해 대규모 언어 모델 (LLM)을 활용한 요구사항 도출이 가능해졌습니다. 이 연구는 AppStore와 LLM 기반 접근 방식을 비교하여 특징을 향상시키는 과정을 다루고 있습니다.

- **Technical Details**: 연구에서는 1,200개의 서브 기능(sub-features)을 수동으로 분석하여 두 접근 방식의 이점과 도전 과제를 파악하였습니다. AppStore와 LLM 모두 명확한 설명을 가진 관련성 높은 서브 기능을 추천하지만, LLM은 특히 새로운 앱 범위와 관련하여 더 강력한 것으로 나타났습니다.

- **Performance Highlights**: 일부 추천된 기능은 비현실적이며 실행 가능성이 불분명한 경우가 있어, 요구 사항 도출 과정에서 인간 분석가의 역할이 중요함을 시사합니다.



### MoRe Fine-Tuning with 10x Fewer Parameters (https://arxiv.org/abs/2408.17383)
- **What's New**: 본 논문에서는 Monarch Rectangular Fine-tuning (MoRe)이라는 새로운 프레임워크를 제안하여 효율적인 파라미터 조정 구조를 갖춘 어댑터 아키텍처를 탐색합니다. MoRe는 Monarch 매트릭스 클래스를 기반으로 하여 기존의 LoRA보다 더 높은 표현 능력을 보입니다.

- **Technical Details**: MoRe는 파라미터 효율성과 성능을 동시에 개선하는 간단한 방법을 제공하며, Monarch 매트릭스가 다양한 구조화된 매트릭스를 표현 가능하게 하여 적합한 아키텍처를 학습합니다. MoRe의 최적화는 복잡한 아키텍처 탐색 과정을 회피하며 최소한의 조정 가능한 하이퍼파라미터를 유지합니다.

- **Performance Highlights**: MoRe는 LoRA 대비 10×-20× 더 파라미터 효율적이며, 다양한 작업과 모델에서 현존하는 PEFT 기술보다 더 우수한 성능을 보여줍니다. 실험 코드는 깃허브에 공개되어 재현 가능합니다.



### EMPOWER: Embodied Multi-role Open-vocabulary Planning with Online Grounding and Execution (https://arxiv.org/abs/2408.17379)
Comments:
          Accepted at IROS 2024

- **What's New**: EMPOWER 프레임워크를 소개하며, 로봇의 실제 환경에서 작업 계획을 위한 온라인 그라운딩(grounding) 및 계획(planning) 접근 방식을 개선합니다.

- **Technical Details**: EMPOWER는 다중 역할 프롬프트(multi-role prompting) 기법과 효율적인 사전 훈련된 모델(pre-trained models)을 활용하여 복잡한 장면에서 효과적인 작업 계획을 생성합니다. 이 프레임워크는 로봇의 제한된 컴퓨팅 자원을 고려하여 개발되었습니다.

- **Performance Highlights**: TIAGo 로봇을 사용한 6개의 실제 상황에서 평균 0.73의 성공률(Success Rate)을 달성하여 제안된 접근 방식의 효과성을 강조합니다.



### NDP: Next Distribution Prediction as a More Broad Targ (https://arxiv.org/abs/2408.17377)
Comments:
          8 pages,5 figures

- **What's New**: 본 논문은 Next Distribution Prediction (NDP)라는 새로운 방법론을 제안하여, 기존의 next-token prediction (NTP) 패러다임의 한계를 극복하고, LLM의 학습 성능을 향상시키고자 합니다. NDP는 n-gram 분포를 사용하여 one-hot 타겟을 대체하며, 총 과제가 포함된 다양한 실험에서 눈에 띄는 성능 향상을 보여주었습니다.

- **Technical Details**: NDP는 LLM 훈련에서 n-gram 통계적 언어 모델을 활용한 새로운 접근법으로, 훈련 중 단일 one-hot 분포 대신 n-gram 분포를 타겟으로 설정합니다. 이 방법은 supervised 언어 모델링과 Causal Language Modeling (CLM) 분포를 결합하여 훈련합니다. 또한, NDP는 도메인 적응 및 언어 전이 상황에서 유리하게 작용합니다.

- **Performance Highlights**: NDP는 다음과 같은 두드러진 성과 향상을 기록했습니다: 번역 작업에서 최대 +2.97 COMET 개선, 일반 작업에서 평균 0.61점 개선, 의료 분야에서 평균 10.75점 개선. 이는 LLM이 narrow candidate 문제를 해결함으로써 성능을 크게 향상시킬 수 있음을 입증합니다.



### Leveraging Graph Neural Networks to Forecast Electricity Consumption (https://arxiv.org/abs/2408.17366)
Comments:
          17 pages, ECML PKDD 2024 Workshop paper

- **What's New**: 본 논문은 전력 소비 예측을 위해 그래프 기반 모델인 Graph Convolutional Networks 및 Graph SAGE를 활용한 새로운 방법론을 제안합니다. 이는 기존의 Generalized Additive Model 프레임워크를 초월하는 접근 방식으로, 각 노드가 지역의 소비량을 나타내는 네트워크 구조 내의 관계를 효과적으로 포착합니다.

- **Technical Details**: 이 연구에서 제안한 방법론은 전력 소모 예측을 위한 그래프 구성 방법과 개발한 모델의 성능 및 설명 가능성을 평가하는 프레임워크를 포함합니다. 특히 소비 예측에 맞춰 조정된 그래프 추론 방법을 도입하며, 기본적인 GNN(Graph Neural Networks), 적응형 예측 알고리즘, Neural Additive Models를 통합해 전력 소비 데이터의 특성에 맞는 예측을 합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 프랑스 본토 지역의 전력 예측에 대해 기존의 통계적 방법보다 우수한 예측 성능을 보여줍니다. 그래프 기반 모델이 복잡한 시간 시계열 데이터 내의 관계를 효과적으로 캡처할 수 있음을 입증하였으며, 새로운 합성 데이터셋을 통해 다양한 GNN 모델의 성능을 비교할 수 있는 기초를 마련했습니다.



### C-RADAR: A Centralized Deep Learning System for Intrusion Detection in Software Defined Networks (https://arxiv.org/abs/2408.17356)
- **What's New**: 본 연구에서는 Software Defined Network (SDN)에서의 침입 탐지를 위한 심층 학습 기반 기술을 제안합니다. 기존의 침입 탐지 기술에 비해 심층 학습(Deep Learning) 방법이 더 효과적이라는 것을 실험을 통해 입증하였습니다.

- **Technical Details**: 연구에서는 Long Short Term Memory Network (LSTM) 및 Self-Attention 기반 아키텍처인 LSTM-Attn을 사용하였으며, 이 모델은 0.9721의 F1-score를 달성하였습니다. 실험 데이터는 CSE-CIC-IDS2018111 데이터셋을 사용하여 다양한 공격 유형을 탐지했습니다.

- **Performance Highlights**: 심층 학습 기반 접근 방식이 기존의 전통적인 방법들보다 탐지 정확도와 연산 효율성에서 더 우수한 성능을 보였습니다. 이 방법은 새로운 공격 패턴을 탐지할 수 있으며, SDN의 전반적인 보안을 향상시키는 데 기여할 수 있습니다.



### Bidirectional Decoding: Improving Action Chunking via Closed-Loop Resampling (https://arxiv.org/abs/2408.17355)
Comments:
          Project website: this https URL

- **What's New**: 이번 논문에서는 액션 청킹(action chunking)의 역할을 분석하여 로봇 학습에서의 정책의 성능에 미치는 영향을 조사합니다. 연구자들은 액션 청킹의 길이를 늘리면 정책이 과거 상태와 액션을 보다 잘 파악할 수 있지만, 이는 스토캐스틱 환경에서의 오류를 악화시킬 수 있음을 발견했습니다. 이 문제를 해결하기 위해 Bidirectional Decoding (BID)이라는 새로운 알고리즘을 제안하고, 이는 액션 청킹과 닫힌 루프(closed-loop) 작동을 연결합니다.

- **Technical Details**: BID는 각 시간 단계에서 여러 예측을 샘플링하고 최적의 예측을 선택하는 테스트 타임 추론 알고리즘입니다. 핵심 기준으로는 (i) backward coherence(역방향 일관성)와 (ii) forward contrast(전방향 대비)가 있습니다. BID는 액션 청킹의 내부 및 외부 모두에서 의사 결정을 연결하여 긴 시퀀스의 시간적 일관성을 향상시키면서 스토캐스틱 환경에서의 확장 가능한 재계획이 가능하게 합니다.

- **Performance Highlights**: 실험 결과 BID는 7개의 시뮬레이션 벤치마크 및 2개의 실제 과업에서 두 개의 최신 생성 정책의 전통적인 닫힌 루프 작업에 비해 상대 성능이 26% 이상 향상되었습니다. BID는 계산적으로 효율적이고 모델에 독립적이며 구현이 용이하여 테스트 시간에 생성적 행동 복제를 향상시키는 플러그 앤 플레이 컴포넌트로 작용합니다.



### Forget to Flourish: Leveraging Machine-Unlearning on Pretrained Language Models for Privacy Leakag (https://arxiv.org/abs/2408.17354)
- **What's New**: 이 논문은 사전 훈련된 대형 언어 모델(LLM)을 개인 데이터로 미세 조정하는 과정에서 발생하는 개인정보 노출 위험을 다룹니다. 새로운 모델 오염 공격 방법론을 제안하여 특정 데이터 지점을 기억하도록 훈련된 모델의 훈련 손실을 조정하여 프라이버시 노출을 극대화합니다.

- **Technical Details**: 이 연구는 머신 언러닝(machine unlearning) 개념을 기반으로 하여, 사전 훈련된 LLM에 대한 공격 도구로 활용합니다. 제안된 방법은 사용자 데이터로 미세 조정된 모델에 대한 두 가지 표준 프라이버시 공격인 멤버십 추론(membership inference) 및 데이터 추출(data extraction)을 측정하여, 다양한 LLM 및 데이터셋에서 성능을 평가합니다.

- **Performance Highlights**: 실험 결과, 제안된 공격 기법은 거의 모든 시나리오에서 기존의 성능 기준을 초과하며, 모델의 유용성을 유지하면서 프라이버시 노출을 증가시킵니다. 이 연구는 인증되지 않은 출처에서 사전 훈련된 모델을 다운로드하는 사용자를 위한 경고의 메시지를 담고 있습니다.



### AASIST3: KAN-Enhanced AASIST Speech Deepfake Detection using SSL Features and Additional Regularization for the ASVspoof 2024 Challeng (https://arxiv.org/abs/2408.17352)
Comments:
          8 pages, 2 figures, 2 tables. Accepted paper at the ASVspoof 2024 (the 25th Interspeech Conference)

- **What's New**: 이번 연구에서는 음성 스푸핑(voice spoofing) 탐지를 위한 새로운 구조인 AASIST3를 제안합니다. 기존 AASIST 프레임워크에 Kolmogorov-Arnold 네트워크를 추가하고, 여러 층과 인코더, 프리엠퍼시스(pre-emphasis) 기법을 통합하여 성능을 두 배 이상 향상시켰습니다.

- **Technical Details**: AASIST3는 GAT, GraphPool, HS-GAL의 주의(attention) 층을 수정하여 KAN(Kolmogorov-Arnold Networks)을 이용합니다. 이를 통해 중요한 특징을 추출하는 능력이 강화되었습니다. 데이터 전처리(data pre-processing)로는 다양한 데이터 증강과 프리엠퍼시스 기법을 사용하여 의미 있는 주파수 정보를 획득합니다.

- **Performance Highlights**: AASIST3는 폐쇄 조건(closed condition)에서는 0.5357, 개방 조건(open condition)에서는 0.1414의 minDCF(minimum Detection Cost Function) 결과를 나타내어, 합성 음성(synthetic voice)의 탐지 성능을 크게 향상시킴으로써 ASV 시스템의 보안을 개선했습니다.



### rerankers: A Lightweight Python Library to Unify Ranking Methods (https://arxiv.org/abs/2408.17344)
- **What's New**: 이 논문에서는 Python 라이브러리인 rerankers를 소개합니다. 이 라이브러리는 사용자가 가장 일반적으로 사용되는 re-ranking 접근 방식을 쉽게 사용할 수 있도록 하는 인터페이스를 제공합니다.

- **Technical Details**: rerankers는 재정렬을 통합한 단일 인터페이스를 제공하여, 사용자가 Python 코드 한 줄만 변경하여 다양한 방법을 탐색할 수 있도록 합니다. 이 라이브러리는 모든 최신 Python 버전과 호환되며, Hugging Face의 transformers 생태계에 통합되어 있습니다.

- **Performance Highlights**: 실험을 통해 rerankers의 구현은 기존 실행 결과와 성능에서 동등성을 달성했습니다. 이러한 성과는 MS Marco, Scifact 및 TREC-Covid 데이터셋에서 평가되었습니다.



### Modularity in Transformers: Investigating Neuron Separability & Specialization (https://arxiv.org/abs/2408.17324)
Comments:
          11 pages, 6 figures

- **What's New**: 이번 연구는 Transformer 모델의 뉴런 모듈성과 태스크( task) 전문성에 대한 새로운 통찰을 제공합니다. Vision Transformer (ViT) 및 언어 모델 Mistral 7B에서 다양한 태스크에 대한 뉴런의 역할을 분석하여 태스크별 뉴런 클러스터를 발견했습니다.

- **Technical Details**: 연구에서는 선택적 가지치기(selective pruning) 및 Mixture of Experts (MoE) 군집화 기법을 사용하여 뉴런의 겹침(overlap) 및 전문성을 분석하였습니다. 연구 결과, 훈련된 모델과 임의 초기화된 모델 모두에서 뉴런 중요도 패턴이 일정 부분 지속됨을 발견했습니다.

- **Performance Highlights**: ViT 모델에서 특정 클래스에 맞는 뉴런을 비활성화함으로써 다른 클래스의 성능에 미치는 영향을 평가했으며, Mistral 모델에서는 각 태스크에서의 성능 저하가 상관 관계가 있음을 관찰했습니다.



### Investigating Neuron Ablation in Attention Heads: The Case for Peak Activation Centering (https://arxiv.org/abs/2408.17322)
Comments:
          9 pages, 2 figures, XAI World Conference 2024 Late-Breaking Work

- **What's New**: 본 연구에서는 언어 모델과 비전 트랜스포머에서 신경 세포의 활성화(ablation) 방법을 비교하며, 새로운 기법인 'peak ablation'을 제안합니다.

- **Technical Details**: 신경 활성화 양식을 각각 다르게 조정하는 네 가지 방법(Zero Ablation, Mean Ablation, Activation Resampling, Peak Ablation)을 사용하여 성능 저하 정도를 실험적으로 분석하였습니다. 특히, 'peak ablation' 기법에서는 모달 활성화(modal activation)를 기반으로 한 새로운 접근법을 사용합니다.

- **Performance Highlights**: 각기 다른 방법에서 모델 성능 저하가 가장 적은 방법을 식별하였으며, 일반적으로 resampling이 가장 큰 성능 저하를 발생시킨다는 결과를 도출했습니다.



### Fair Best Arm Identification with Fixed Confidenc (https://arxiv.org/abs/2408.17313)
- **What's New**: 이번 연구에서는 공정성 제약을 고려한 최적 팔 식별(Best Arm Identification, BAI)을 위한 새로운 프레임워크인 F-BAI(fair BAI)를 제시합니다. F-BAI는 최적 팔을 최소 샘플 복잡도로 식별하는 전통적 BAI와 달리, 각 팔의 선택률에 대한 하한을 포함한 공정성 제약을 설정합니다.

- **Technical Details**: F-BAI에서 특정 인스턴스의 샘플 복잡도 하한을 수립하고, 공정성이 샘플 복잡도에 미치는 영향을 정량화하는 'price of fairness'을 분석합니다. 샘플 복잡도 하한을 기반으로 두 가지 공정성 제약(모델 비의존적 및 모델 의존적)을 만족하는 F-TaS 알고리즘을 제안하며, 이 알고리즘은 샘플 복잡도를 최소화하면서도 공정성 위반을 낮게 유지하는 성능을 보입니다.

- **Performance Highlights**: 수치 실험 결과, F-TaS는 인공 모델 및 실제 무선 스케줄링 응용 프로그램에서 샘플 효율성을 보여주며, 최소 공정성 위반을 지속적으로 달성하였습니다.



### Hybridizing Base-Line 2D-CNN Model with Cat Swarm Optimization for Enhanced Advanced Persistent Threat Detection (https://arxiv.org/abs/2408.17307)
Comments:
          6 pages, 5 figures

- **What's New**: 이 연구는 고급 지속 위협(APT) 탐지를 개선하기 위해 2D-CNN(Convolutional Neural Networks)과 Cat Swarm Optimization(CSO) 알고리즘을 융합한 새로운 접근법을 제시합니다. APT의 복잡한 특성을 감지하기 위한 혁신적인 방법론을 도입하였습니다.

- **Technical Details**: 연구에서는 CSO 알고리즘을 사용하여 2D-CNN 기반 모델의 성능을 향상시키고, 'DAPT 2020' 데이터셋을 이용하여 APT 공격의 다양한 단계에서의 정확도를 평가합니다. CSO는 두 가지 모드(Seeking Mode 및 Tracing Mode)로 작동하며, 모델의 솔루션을 최적화하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 본 연구의 모델은 놀라운 98.4%의 정확도를 기록하였으며, 이는 기존 탐지 방법들에 비해 월등히 향상된 성과를 나타냅니다. 이 연구는 다양한 공격 단계에서 APT 탐지의 효과성을 증명하며 사이버 보안 분야의 발전에 기여할 것입니다.



### Accelerating the discovery of steady-states of planetary interior dynamics with machine learning (https://arxiv.org/abs/2408.17298)
- **What's New**: 이번 연구에서는 기계 학습(Machine Learning)을 활용하여 맨틀 대류 시뮬레이션의 속도를 높이는 새로운 방법을 제시합니다. 128개의 이차원 시뮬레이션 데이터셋을 생성하고, 97개의 시뮬레이션에서 피드포워드 신경망(Feedforward Neural Network)을 훈련하여 안정 상태(steady-state) 온도 프로파일을 예측합니다.

- **Technical Details**: 연구팀은 서로 다른 시뮬레이션 파라미터에 대해 초기 조건으로 사용할 수 있는 온도 프로파일을 예측하였습니다. 이 신경망은 막대한 계산 단계 없이도 안정 상태를 달성할 수 있도록 해줍니다. 기존 초기화 방법에 비해 평균적으로 안정 상태에 도달하는 데 필요한 시간 단계의 수가 3.75배 감소합니다.

- **Performance Highlights**: 본 방법은 적은 수의 시뮬레이션만으로 훈련할 수 있어 예측 오류가 없고, 추론 시 컴퓨팅 오버헤드가 최소화됩니다. 이는 맨틀 대류 연구를 가속화하고, 행성 내부의 역학을 더 잘 이해하는 데 기여할 것으로 기대됩니다.



### Stationary Policies are Optimal in Risk-averse Total-reward MDPs with EVaR (https://arxiv.org/abs/2408.17286)
- **What's New**: 이 논문은 할인된 MDP(마르코프 결정 프로세스)에서의 위험 회피 목표 최적화의 도전 과제를 다룹니다. 특히, 엔트로픽 위험 측정(ERM) 및 엔트로픽 가치-at-위험(EVaR) 위험 측정 기준 아래에서 위험 회피 총 보상 기준(TRC)을 정적 정책으로 최적화할 수 있음을 보여줍니다.

- **Technical Details**: 이 논문의 주된 기여는 ERM 및 EVaR 위험 측정을 통해 위험 회피 TRC가 과도기 MDP(Transient MDP)에서 최적 정적 정책 및 최적 가치 함수를 수용할 수 있다는 점입니다. 또한, 값(iteration), 정책(iteration), 선형 프로그래밍을 경량 알고리즘 형태로 사용하여 이를 계산할 수 있음을 설명합니다.

- **Performance Highlights**: 연구 결과에 따르면, 이 방법은 위험 회피 강화 학습 영역의 넓은 범위에서 할인 기준보다 총 보상 기준이 선호될 수 있음을 시사합니다. 또한 EVaR는 해석과 계산에서 많은 장점을 제공하며, ERM보다 더 직관적인 위험 측정 기준으로 나타났습니다.



### UrBench: A Comprehensive Benchmark for Evaluating Large Multimodal Models in Multi-View Urban Scenarios (https://arxiv.org/abs/2408.17267)
Comments:
          9 pages, 6 figures

- **What's New**: 이번 논문에서는 복잡한 다중 시각 도시 환경을 평가하기 위해 설계된 UrBench라는 포괄적인 벤치마크를 제안합니다. 이 벤치마크는 도시 작업에 대한 다양한 평가를 포함하며, 기존의 단일 시점 도시 벤치마크의 한계를 극복합니다.

- **Technical Details**: UrBench는 지역 수준(Region-Level) 및 역할 수준(Role-Level)의 질문을 포함하며, Geo-Localization, Scene Reasoning, Scene Understanding, Object Understanding의 네 가지 작업 차원에 걸쳐 총 14가지 작업 유형을 제공합니다. 데이터 수집에는 11개 도시에서의 새롭게 수집된 주석과 기존 데이터셋의 데이터가 포함됩니다. 또한, 우리는 다중 뷰 관계를 이해하는 모델의 능력을 평가하기 위해 다양한 도시 관점을 통합합니다.

- **Performance Highlights**: 21개의 LMM(대규모 다중 모달 모델)에 대한 평가 결과, 현재 모델들은 도시 환경에서 여러 측면에서 인간보다 뒤처지는 것으로 나타났습니다. 예를 들어, GPT-4o 모델조차도 많은 작업에서 인간에 비해 평균 17.4%의 성능 격차를 보였으며, 특히 서로 다른 도시 뷰에 따라 일관성 없는 행동을 보이는 경향이 있었습니다.



### VisionTS: Visual Masked Autoencoders Are Free-Lunch Zero-Shot Time Series Forecasters (https://arxiv.org/abs/2408.17253)
Comments:
          26 pages, 11 figures

- **What's New**: 이 논문은 자연 이미지에서 구축된 TSF(시계열 예측) 기초 모델인 VisionTS를 소개합니다. 이 접근 방식은 기존의 텍스트 기반 모델 및 시계열 데이터 수집 방식과는 다른 세 번째 경로를 탐색합니다.

- **Technical Details**: VisionTS는 대마스크 오토인코더(visual masked autoencoder, MAE)를 기반으로 하여 1D 시계열 데이터를 패치 수준의 이미지 재구성 작업으로 변형합니다. 이 방법은 시계열 예측 작업을 이미지 렌더링과 맞춤형 패치 정렬을 통해 진행할 수 있게 합니다.

- **Performance Highlights**: VisionTS는 기존 TSF 기초 모델보다 우수한 제로샷(zero-shot) 예측 성능을 보여주며, 최소한의 미세 조정(fine-tuning)으로도 여러 장기 시계열 예측 기준에서 최신 성능을 달성합니다.



### Abstracted Gaussian Prototypes for One-Shot Concept Learning (https://arxiv.org/abs/2408.17251)
- **What's New**: 본 연구는 Omniglot 챌린지에서 영감을 받아 고급 시각 개념을 인코딩하기 위한 클러스터 기반 생성 이미지 분할 프레임워크를 소개합니다. 이 프레임워크는 Gaussian Mixture Model (GMM)의 구성 요소에서 각각의 매개변수를 추론하여 시각 개념의 독특한 위상 서브파트를 나타냅니다.

- **Technical Details**: Abstracted Gaussian Prototype (AGP)은 GMMs를 활용하여 손글씨 문자에 대한 시각 개념을 유연하게 모델링합니다. GMMs는 데이터 포인트를 유한한 수의 Gaussian 분포의 합으로 표현하는 비지도 클러스터링 알고리즘입니다. AGP는 최소한의 데이터로 새로운 개념을 학습할 수 있도록 돕습니다.

- **Performance Highlights**: 연구 결과, 생성 파이프라인이 인간이 만든 것과 구별할 수 없는 새롭고 다양한 시각 개념을 생성할 수 있음을 보여주었습니다. 본 시스템은 낮은 이론적 및 계산적 복잡성을 유지하면서도 기존 접근 방식에 비해 강력한 성능을 발휘하였습니다.



### AI-Driven Intrusion Detection Systems (IDS) on the ROAD dataset: A Comparative Analysis for automotive Controller Area Network (CAN) (https://arxiv.org/abs/2408.17235)
- **What's New**: 본 논문은 기존 문헌에서 부족했던 공개적이고 포괄적인 Intrusion Detection Systems (IDS) 테스트를 위한 데이터셋 문제를 다룹니다. 최신 ROAD 데이터셋을 사용하여 정밀한 침입 및 공격을 포함한 IDS 효과성을 평가합니다.

- **Technical Details**: 데이터셋 레이블링과 함께, 최신 심층 학습(Deep Learning) 모델과 전통적 머신 러닝(Machine Learning) 모델을 구현하여 기존 문헌에서 주로 사용되는 데이터셋과 ROAD 데이터셋 간의 성능 차이를 보여줍니다. 연구에는 Transformer 기반 Attention Network (TAN), Deep Convolutional Neural Network (DCNN), Long Short-Term Memory (LSTM) 기반 IDS, 그리고 LightGBM, Random Forest (RF) 등의 여러 머신 러닝 방법이 포함됩니다.

- **Performance Highlights**: ROAD 데이터셋의 효과를 입증하기 위해 HCRL Car Hacking 데이터셋 및 In-Vehicle Network Intrusion Detection Challenge 데이터셋과의 비교 분석을 통해 IDS 기술의 효용성을 평가합니다. 본 연구는 다양한 공격 카테고리를 포함하며, 심층 학습 기술의 성능을 향상시키기 위한 방법을 제안합니다.



### "Benefit Game: Alien Seaweed Swarms" -- Real-time Gamification of Digital Seaweed Ecology (https://arxiv.org/abs/2408.17186)
Comments:
          Paper accepted at ISEA 24, The 29th International Symposium on Electronic Art, Brisbane, Australia, 21-29 June 2024

- **What's New**: 이 논문에서는 "Benefit Game: Alien Seaweed Swarms"라는 프로젝트를 통해 인공 생명 예술과 인터랙티브 게임을 결합하여 인간 활동이 취약한 해조류 생태계에 미치는 영향을 탐구합니다. 이 게임은 디지털 해조류 생태계를 균형 있게 유지하고 생태적 의식을 고취시키는 것을 목표로 합니다.

- **Technical Details**: 이 프로젝트는 Procedural Content Generation via Machine Learning (PCGML) 기술을 사용하여 가상의 해조류와 공생 곰팡이의 변화를 생성합니다. 사용자와의 상호작용을 통해 해조류 군집의 성장에 영향을 미치고, 과도한 수확이 해조류 생태계의 멸종으로 이어질 수 있음을 알립니다.

- **Performance Highlights**: 이 게임은 최소한의 사용자 입력으로 제한된 알골리즘적 게임 콘텐츠 생성을 통해 경과를 실시간으로 반영하는 인공 해조류 생태계를 제공합니다. 사용자들은 게임 토큰을 통해 해조류를 수확하고, 해조류 생태계의 건강을 회복하는 데 필요한 전략을 고민하며 지속 가능한 균형을 찾도록 유도합니다.



### Causal Reasoning in Software Quality Assurance: A Systematic Review (https://arxiv.org/abs/2408.17183)
Comments:
          Preprint Journal Information and Software Technology

- **What's New**: 이번 연구는 소프트웨어 품질 보증(SQA)에서 인과적 추론(causal reasoning)의 활용이 어떻게 이루어지고 있는지를 폭넓고 상세하게 개관하고 있다.

- **Technical Details**: 연구 방법론으로는 SQA 분야에서의 인과적 추론을 다룬 체계적인 문헌 검토(systematic literature review)가 사용되었다. 주요 결과로는 인과적 추론이 주로 사용되는 SQA의 주요 영역과 선호되는 방법론, 제안된 솔루션의 성숙도 수준이 도출되었다.

- **Performance Highlights**: 인과적 추론은 결함 지역화(fault localization) 활동에서 가장 많이 활용되며, 특히 웹 서비스/web services 및 마이크로서비스/microservices 도메인에서 두드러진다. 또한, 인과적 추론에서는 Pearl의 그래픽 형식이 선호되고 있으며, 이는 직관성 덕분으로 여겨진다. 2021년 이후로 이러한 응용을 촉진하는 도구들이 빠르게 등장하고 있다.



### Codec Does Matter: Exploring the Semantic Shortcoming of Codec for Audio Language Mod (https://arxiv.org/abs/2408.17175)
- **What's New**: 최근 오디오 생성 기술이 대규모 언어 모델(LLM)의 발전으로 크게 향상되었습니다. 본 연구에서는 기존 오디오 LLM 코드들이 생성된 오디오의 의미적 무결성을 유지하는 데에서 겪는 문제점을 해결하기 위해 새로운 방법인 X-Codec을 제안합니다.

- **Technical Details**: X-Codec은 사전 훈련된 의미 인코더로부터의 의미적 특징을 Residual Vector Quantization(RVQ) 단계 이전에 통합하고 RVQ 이후에 의미 재구성 손실을 도입합니다. 이를 통해 X-Codec은 음성 합성 작업에서 단어 오류율(Word Error Rate, WER)을 크게 줄이고, 음악 및 소리 생성과 같은 비음성 응용 프로그램에서도 이점을 제공합니다.

- **Performance Highlights**: 우리는 텍스트-음성 변환, 음악 연속 생성, 텍스트-소리 합성 작업에서 X-Codec의 효과를 종합적으로 평가했습니다. 결과는 제안된 방법의 효과를 일관되게 입증하며, VALL-E 기반 TTS와의 비교 평가에서 X-Codec이 기존의 분리 기법을 능가함을 확인했습니다.



### Deep Feature Embedding for Tabular Data (https://arxiv.org/abs/2408.17162)
Comments:
          15 pages, 2figures, accepted to ICONIP 2024, Paper ID: 1399

- **What's New**: 이 논문에서는 경량의 딥 뉴럴 네트워크(lightweight deep neural network)를 활용하여 탭형 데이터(tabular data)의 수치적(feature) 및 범주적(categorical feature) 특성을 위한 효과적인 기능 임베딩(embedding)을 생성하는 새로운 딥 임베딩 프레임워크(deep embedding framework)를 제안합니다.

- **Technical Details**: 제안된 프레임워크는 수치적 특징을 위해 이단계 기능 확장(two-step feature expansion) 및 딥 변환(deep transformation) 기법을 사용합니다. 이 과정에서 각 수치적 특징은 학습 가능한 임베딩 센서티비티(embedding sensitivity)와 임베딩 바이어스(embedding bias) 벡터를 통해 확장된 후, 잔여 연결(residual connection) 및 exp-centered 활성화(activation)와 함께 DNN(d deep neural network)에 의해 처리됩니다. 범주적 특징의 경우, 고유한 식별 벡터(unique identification vector)를 사용하고, 이 특징은 파라미터화된 딥 임베딩 함수(parameterized deep embedding function)로 변환됩니다.

- **Performance Highlights**: 실제 데이터셋에서의 실험을 통해 제안된 딥 임베딩 프레임워크의 효과성과 효율성이 검증되었습니다. 이 프레임워크는 기존의 특징 임베딩 모듈을 대체할 수 있으며, 수치적 및 범주적 특징 모두에 대해 통합된 딥 임베딩을 제공합니다.



### Look, Compare, Decide: Alleviating Hallucination in Large Vision-Language Models via Multi-View Multi-Path Reasoning (https://arxiv.org/abs/2408.17150)
Comments:
          13 pages, 7 tables, 7 figures

- **What's New**: 최근 대형 비전-언어 모델(LVLM)들이 멀티모달(multi-modal) 맥락 이해에서 인상적인 능력을 보여주고 있으나, 여전히 이미지 내용과 일치하지 않는 출력을 생성하는 환각(hallucination) 문제를 겪고 있습니다. 본 논문에서는 훈련 없이 환각을 줄이기 위한 새로운 프레임워크인 MVP(Multi-View Multi-Path Reasoning)를 제안합니다.

- **Technical Details**: MVP는 이미지에 대한 다각적 정보 탐색 전략을 개발하고, 각 정보 뷰에 대해 다중 경로(reasoning) 추론을 도입하여 잠재적인 답변의 확신.certainty 점수를 정량화하여 집계합니다. 이를 통해 이미지 정보를 충분히 이해하고 잠재적 답변의 확신수를 고려하여 최종 답변을 결정합니다. 해당 방법은 CLIP, BLIP 등의 비전 인코더를 활용합니다.

- **Performance Highlights**: MVP는 네 가지 잘 알려진 LVLM에서 실험을 통해 환각 문제를 효과적으로 완화함을 입증했습니다. MVP는 기계 학습에 있어 추가 훈련 비용이나 외부 도구 없이 LVLM의 내재적 능력을 최대한 활용하는 데 중점을 두었습니다. 실험 결과는 저희 프레임워크가 최근의 훈련 없는 방법론들보다 우수한 성능을 보였음을 보여줍니다.



### Towards Hyper-parameter-free Federated Learning (https://arxiv.org/abs/2408.17145)
Comments:
          28 pages, 3 figures

- **What's New**: 이 논문에서는 데이터 중앙 집중화의 제약 속에서 분산 시스템에서 머신 러닝 모델을 트레이닝하는 Federated Learning (FL)에서의 글로벌 모델 업데이트를 위한 자동화된 스케일링 기법 두 가지를 소개합니다. 기존의 고정된 hyperparameter (하이퍼파라미터) 대신, 이를 자동으로 조정하는 알고리즘을 제안합니다.

- **Technical Details**: 제안된 두 가지 알고리즘은 다음과 같습니다. 1) Federated Line-search (FedLi-LS): Armijo line search (아르미조 선 탐색)를 클라이언트에서 사용하여, 서버에서 스케일링 글로벌 모델 업데이트를 자동으로 수행할 수 있도록 하였습니다. 2) Federated Linearized Updates (FedLi-LU): 클라이언트의 Sgd 업데이트를 바탕으로 하여, 글로벌 모델 업데이트를 위한 손실 보존 선형화를 최소화하여 최적의 스케일링 계수를 계산합니다.

- **Performance Highlights**: 제안된 FedLi 방법들은 이미지 분류 및 언어 태스크에서의 효과성을 광범위한 실험을 통해 입증하였으며, 강한 볼록 함수 및 일반 비볼록 문제 모두에서 선형 수렴을 보여주었습니다.



### Leveraging Digital Twin Technologies for Public Space Protection and Vulnerability Assessmen (https://arxiv.org/abs/2408.17136)
- **What's New**: 본 논문에서는 공공 장소 보호를 위한 혁신적인 Digital Twin-as-a-Security-Service (DTaaSS) 아키텍처를 소개합니다. 이 아키텍처는 이러한 공공 장소의 보안을 종합적이고 효과적으로 높이는 데 중점을 두고 있습니다.

- **Technical Details**: 제안된 DTaaSS 시스템은 Digital Twin (DT) 개념과 최첨단 기술인 Internet of Things (IoT), 클라우드 컴퓨팅, 빅 데이터 분석, 인공지능 (AI)을 결합하여 구성되었습니다. 이 시스템은 데이터 수집 및 분석, 지역 모니터링 및 제어, 사전 위협 탐지 등의 기능을 포함하여 실시간으로 작동합니다.

- **Performance Highlights**: 이 아키텍처는 복잡하고 혼합된 위협을 처리할 수 있는 높은 잠재력을 보여 주며, 대표적인 실제 응용 시나리오들, 예를 들어 지하철역, 레저 장소, 대성당 광장에서의 공격 사례에 대해 강력한 효과를 발휘할 수 있습니다.



### VQ4DiT: Efficient Post-Training Vector Quantization for Diffusion Transformers (https://arxiv.org/abs/2408.17131)
Comments:
          11 pages, 6 figures

- **What's New**: 이 논문에서는 DiT(Diffusion Transformers) 모델을 위한 새로운 포스트 트레이닝 벡터 양자화 방법인 VQ4DiT를 제안합니다. VQ4DiT는 모델 크기와 성능 간의 균형을 이루며, 매우 낮은 비트 폭인 2비트로 양자화하면서도 이미지 생성 품질을 유지하는 데 성공했습니다.

- **Technical Details**: VQ4DiT는 전통적인 VQ 방법의 한계를 극복하기 위해, 각 가중치 서브 벡터의 후보 할당 세트를 유클리드 거리 기반으로 계산하고, 가중 평균을 통해 서브 벡터를 재구성합니다. 이후 제로 데이터와 블록 단위 보정 방법을 사용하여 최적의 할당을 효율적으로 선택하며, 코드북을 동시에 보정합니다. 이 방법은 NVIDIA A100 GPU에서 DiT XL/2 모델을 20분에서 5시간 내에 양자화할 수 있습니다.

- **Performance Highlights**: 실험 결과, VQ4DiT는 ImageNet 기준에서 풀 정밀도 모델과 비교하여 경쟁력 있는 평가 결과를 보여줍니다. 특히, 이 방법은 디지털 이미지 생성 품질을 손상시키지 않으면서 가중치를 2비트 정확도로 양자화하는 데 성공했습니다.



### Controllable Edge-Type-Specific Interpretation in Multi-Relational Graph Neural Networks for Drug Response Prediction (https://arxiv.org/abs/2408.17129)
- **What's New**: 이번 연구에서는 암 약물 반응 예측을 위한 새로운 사후 해석 알고리즘인 CETExplainer를 제안하고 있습니다. 이 알고리즘은 제어 가능한 엣지 타입별 가중치 메커니즘을 도입하여 생물학적 의미 있는 해석을 제공합니다.

- **Technical Details**: CETExplainer는 다중 관계 그래프 신경망에서 상위 엣지 타입에 맞는 가중치를 사용하여 암 세포 주기 및 약물 간의 위상 관계를 탐색합니다. 또한, 서브그래프와 예측 간의 상호 정보를 최대화하고, 서브그래프 구조 점수를 포함하여 해석의 생물학적 중요성을 향상시킵니다.

- **Performance Highlights**: CETExplainer는 실제 데이터셋에서 우수한 안정성을 보여주고, 타 알고리즘에 비해 설명 품질을 개선하는 것으로 나타났습니다. 이는 암 약물 예측을 위한 강력하고 통찰력 있는 도구를 제공합니다.



### Exploring User Acceptance Of Portable Intelligent Personal Assistants: A Hybrid Approach Using PLS-SEM And fsQCA (https://arxiv.org/abs/2408.17119)
Comments:
          36,

- **What's New**: 이번 연구는 Rabbit R1이라는 새로운 휴대용 지능형 개인 비서(PIPA)의 사용자 수용에 영향을 미치는 요인들을 탐구합니다. 연구는 인공지능(AI) 특정 요인과 사용자 인터페이스 디자인 요인을 통합하여 기술 수용 모델(TAM)을 확장합니다.

- **Technical Details**: 연구는 의도적인 샘플링 방법을 사용하여 미국에서 824명의 사용자로부터 데이터를 수집하고, 부분 최소 제곱 구조 방정식 모델링(PLS-SEM) 및 퍼지 집합 질적 비교 분석(fsQCA)을 통해 샘플을 분석하였습니다.

- **Performance Highlights**: 모든 가설 관계가 지원되었으며, fsQCA는 PLS-SEM 결과를 뒷받침하고 높은 및 낮은 사용자 수용으로 이어지는 세 가지 구성(configuration)을 식별하였습니다.



### Understanding the User: An Intent-Based Ranking Datas (https://arxiv.org/abs/2408.17103)
- **What's New**: 정보 검색 시스템(Information Retrieval Systems)이 발전함에 따라 이 시스템들의 정확한 평가 및 벤치마킹이 중요한 과제가 되고 있습니다. 본 논문에서는 MS MARCO와 같은 웹 검색 데이터셋(Web Search Datasets)이 키워드 쿼리(Keyword Queries)만 제공하고, 의도(Intent)나 설명 없이 제공되며, 정보의 필요성을 이해하는 데 어려움이 있다는 점을 지적하고 있습니다.

- **Technical Details**: 이 연구에서는 두 가지 주요 벤치마크 데이터셋인 TREC-DL-21과 TREC-DL-22에 대해 LLMs(Large Language Models)를 활용하여 각 쿼리의 암묵적인 의도를 분석하고 이해하는 방법론을 제안합니다. 주요 의미적 요소(Semantic Elements)를 추출하여 쿼리에 대한 상세하고 맥락적인 설명을 구성합니다.

- **Performance Highlights**: 생성된 쿼리 설명을 검증하기 위해 크라우드소싱(Crowdsourcing) 방법을 사용하여 다양한 인적 관점을 통해 설명의 정확성과 정보성을 평가합니다. 이 정보는 순위 매기기(Ranking), 쿼리 재작성(Query Rewriting)과 같은 작업의 평가 세트로 활용될 수 있습니다.



### FissionVAE: Federated Non-IID Image Generation with Latent Space and Decoder Decomposition (https://arxiv.org/abs/2408.17090)
- **What's New**: 본 논문에서는 비동일 분포의 데이터 환경에서의 연합 학습( federated learning )에서 Variational Autoencoder (VAE)를 활용하는 새로운 모델인 FissionVAE를 소개합니다. 이 모델은 각 클라이언트 그룹의 데이터 특성에 맞춰 라텐트 공간을 분해하고 개별 디코더 브랜치를 생성하여 맞춤형 학습을 가능하게 합니다.

- **Technical Details**: FissionVAE는 비동일 분포(non-IID) 데이터를 고려하여 라텐트 공간을 특성에 따라 구분하고, 각 클라이언트 그룹에 맞는 디코더 브랜치를 구축합니다. 이로 인해 여러 이미지 유형 간의 라텐트 공간 해석 문제를 낮출 수 있으며, 계층적 VAE 아키텍처와 다형성 디코더 아키텍처를 도입하여 모델의 유연성을 개선합니다.

- **Performance Highlights**: FissionVAE는 MNIST와 FashionMNIST의 조합 및 만화, 인간 얼굴, 동물, 해양 선박, 지구의 원격 감지 이미지로 구성된 복합 데이터셋에서 기존의 기반 연합 VAE 모델에 비해 생성 품질이 크게 향상됨을 실험적으로 입증하였습니다.



### Instant Adversarial Purification with Adversarial Consistency Distillation (https://arxiv.org/abs/2408.17064)
- **What's New**: 본 논문에서는 One Step Control Purification (OSCP)이라는 새로운 확산 기반의 정화 모델을 제안합니다. OSCP는 단 하나의 Neural Function Evaluation (NFE)만으로 적대적 이미지를 정화할 수 있어, 기존 방법들과 비교하여 더 빠르고 효율적입니다.

- **Technical Details**: OSCP는 Latent Consistency Model (LCM)과 ControlNet을 활용하여 구성됩니다. 여기에 Gaussian Adversarial Noise Distillation (GAND)라는 새로운 일관성 증류 프레임워크를 도입하여 자연적 및 적대적 다양체의 동적을 효과적으로 연결합니다.

- **Performance Highlights**: OSCP는 ImageNet에서 74.19%의 방어 성공률을 달성하며, 각각의 정화 작업이 0.1초만 소요됩니다. 이는 기존의 확산 기반 정화 방법들보다 훨씬 시간 효율적이며 전반적인 방어 능력을 유지하는 데 기여합니다.



### A Survey of the Self Supervised Learning Mechanisms for Vision Transformers (https://arxiv.org/abs/2408.17059)
Comments:
          34 Pages, 5 Figures, 7 Tables

- **What's New**: 최근 딥 러닝 알고리즘은 컴퓨터 비전과 자연어 처리 분야에서 인상적인 성과를 보였으나, 훈련에 필요한 대량의 레이블이 있는 데이터 수집은 비용이 많이 들고 시간이 오래 걸립니다. 자가 감독 학습(self-supervised learning, SSL)의 적용이 증가하며, 이 방식이 레이블이 없는 데이터에서 패턴을 학습할 수 있는 가능성을 보여주고 있습니다.

- **Technical Details**: 이 논문에서는 자가 감독 학습 방법들을 상세히 분류하고, ViTs(Vision Transformers) 훈련 시에 이들이 어떻게 활용될 수 있는지를 다룹니다. 주목할 만한 점은, SSL 기술이 대량의 레이블이 없는 데이터를 활용하여 비지도 학습을 가능하게 하고, 이를 통해 모델 성능을 향상시킨다는 점입니다. SSL 방법론은 대조, 생성, 예측 접근법으로 나뉘며 각 방법론은 고유한 패턴 학습을 통해 특징을 추출합니다.

- **Performance Highlights**: ViTs는 최근 이미지 분류 및 객체 탐지와 같은 다양한 컴퓨터 비전 작업에서 우수한 성과를 보여주었습니다. 대량의 데이터셋(JFT-300M)으로 훈련된 ViTs는 레이블이 부족한 상황에서도 강력하고 일반화 가능한 표현을 학습할 수 있는 경로를 제공합니다. 이 연구는 SSL의 발전과 함께 현존하는 다양한 알고리즘의 장단점을 비교하는 분석을 제공합니다.



### Dynamic Self-Consistency: Leveraging Reasoning Paths for Efficient LLM Sampling (https://arxiv.org/abs/2408.17017)
- **What's New**: 이번 연구에서는 Reasoning-Aware Self-Consistency (RASC)라는 혁신적인 조기 중지 프레임워크를 제안합니다. RASC는 Chain of Thought (CoT) prompting에서 출력 응답과 Reasoning Paths (RPs) 모두를 고려하여 샘플 생성 수를 동적으로 조정합니다.

- **Technical Details**: RASC는 각 샘플 RPs에 개별 신뢰도 점수를 매기고, 특정 기준이 충족될 때 중지를 수행합니다. 또한, 가중 다수결 투표(weighted majority voting)를 적용하여 샘플 사용을 최적화하고 답변 신뢰성을 향상시킵니다.

- **Performance Highlights**: RASC는 다양한 QA 데이터셋에서 여러 LLMs를 통해 테스트하여 기존 방법들보다 80% 이상의 샘플 사용을 줄이면서 정확도를 최대 5% 향상시켰습니다.



### Disease Classification and Impact of Pretrained Deep Convolution Neural Networks on Diverse Medical Imaging Datasets across Imaging Modalities (https://arxiv.org/abs/2408.17011)
Comments:
          15 pages, 3 figures, 4 tables

- **What's New**: 이 논문은 다양한 의료 이미징 데이터셋에서 사전 훈련된 딥 컨볼루션 신경망과 전이 학습을 활용한 이진 및 다중 클래스 분류의 아키텍처를 조사합니다.

- **Technical Details**: 연구는 VGG, ResNet, Inception, Xception 및 DenseNet과 같은 10개의 서로 다른 DCNN (Deep Convolutional Neural Networks) 아키텍처를 사용했으며, 사전 훈련 및 랜덤 초기화 모드에서 모델을 훈련 및 평가했습니다. 데이터셋에는 CXR, OCT 및 WSI가 포함됩니다.

- **Performance Highlights**: 사전 훈련 모델을 고정된 특징 추출기로 사용할 경우 성능이 좋지 않았으며, 반대로 병리학적 현미경 WSI에서 더 나은 성능을 보였습니다. 네트워크 아키텍처의 성능은 데이터셋에 따라 달라졌으며, 이는 특정 모달리티의 모델 성능이 동일한 분야 내 다른 모달리티에 대해 결론적이지 않음을 나타냅니다.



### Improving Time Series Classification with Representation Soft Label Smoothing (https://arxiv.org/abs/2408.17010)
Comments:
          14 pages,6 figures

- **What's New**: 이 논문에서는 시간 시계열 분류 (Time Series Classification, TSC) 작업에서 과적합(overfitting) 문제를 줄이기 위한 새로운 방법인 representation soft label smoothing을 제안합니다. 기존의 label smoothing과 confidence penalty 기법을 확장하여, 더 신뢰할 수 있는 소프트 레이블을 생성하는 접근 방식을 소개합니다.

- **Technical Details**: 제안된 방법은 contrastive learning에 의해 사전 훈련된 시간 시계열 인코더 TS2Vec을 사용하여 샘플 표현을 구축하고, 유클리드 거리(Euclidean distance)를 기반으로 소프트 레이블을 생성합니다. 이는 온라인 레이블 스무딩(Online Label Smoothing)과 유사하지만, 모델의 출력 결과가 아닌 인코더의 잠재 공간(latent space)을 기준으로 소프트 레이블을 생성하는 점에서 차별화됩니다.

- **Performance Highlights**: 6개의 다양한 구조와 복잡성을 가진 모델을 훈련시키고, 평균 정확도(average accuracy)를 평가지표로 사용하여 성능을 비교한 결과, 제안된 방법은 기존의 하드 레이블 기반 훈련에 비해 경쟁력 있는 결과를 시현했습니다. 결과적으로, 다양한 구조와 복잡성을 가진 모델에서도 강력한 성능을 발휘하며, 다양한 관점에서 개선 효과가 입증되었습니다.



### Safety Layers of Aligned Large Language Models: The Key to LLM Security (https://arxiv.org/abs/2408.17003)
- **What's New**: 이 논문은 안전하게 조정된 대규모 언어 모델(Aligned LLMs)의 내부 매개변수가 보안을 유지하는 데 중요한 역할을 하고 있음을 밝힙니다. 저자들은 '안전층(safety layers)'이라고 불리는 몇몇 층을 식별하고, 이를 기반으로 한 새로운 미세 조정 기법인 안전 부분 매개변수 미세 조정(Safely Partial-Parameter Fine-Tuning, SPPFT)을 제안합니다.

- **Technical Details**: 연구팀은 다양한 조정된 LLM(대규모 언어 모델)에서 안전층의 존재를 확인하기 위해 알고리즘을 개발하였으며, 입력 벡터의 변화를 분석하여 각 모델의 내부 층에서 안전층을 정밀하게 찾아냈습니다. SPPFT 방법을 통해 안전층의 기울기를 고정하여 보안을 유지하며 미세 조정을 수행합니다.

- **Performance Highlights**: 실험 결과, SPPFT 방법이 전체 미세 조정에 비해 모델 보안을 상당히 보존하면서도 성능을 유지하고 계산 자원을 줄일 수 있다는 것을 보여주었습니다. 이 연구는 조정된 LLM의 보안 본질을 매개변수 수준에서 밝혀내어 더 안전한 AI 구축에 기여하는 중요한 기반을 제공합니다.



### Training Ultra Long Context Language Model with Fully Pipelined Distributed Transformer (https://arxiv.org/abs/2408.16978)
- **What's New**: 본 논문에서는 높은 하드웨어 효율성으로 긴 문맥을 가진 대형 언어 모델(LLMs)을 효율적으로 훈련하기 위한 Fully Pipelined Distributed Transformer (FPDT)를 제안합니다. 저자들은 FPDT를 사용하여 GPT 및 Llama 모델의 시퀀스 길이를 현재의 최첨단 솔루션에 비해 16배 증가시킬 수 있음을 보여주었습니다.

- **Technical Details**: FPDT는 현대 GPU 클러스터의 여러 메모리 계층을 활용하여 하드웨어 효율성과 비용 효율성을 극대화하며 높은 MFU(Mean Floating Utilization)을 달성합니다. 해당 방법은 2백만 시퀀스 길이의 8B LLM을 단 4개의 GPU로 훈련할 수 있게 해주며, 기존 기술과 보완적으로 작동할 수 있도록 설계되었습니다.

- **Performance Highlights**: FPDT는 32개의 GPU를 사용하여 4M 시퀀스 길이의 70B 모델을 훈련하는 동시에, 메모리 소비를 극적으로 줄이고 훈련 흐름에서 계산과의 중복을 최소화합니다. 이는 기존 솔루션보다 16배 더 긴 시퀀스 훈련을 가능하게 하며, MFU가 55% 이상에 도달하게 됩니다.



### Technical Report of HelixFold3 for Biomolecular Structure Prediction (https://arxiv.org/abs/2408.16975)
- **What's New**: 헬릭스폴드3(HelixFold3) 모델이 AlphaFold3의 기능을 재현하고 있으며, 이를 통해 생체분자 구조 예측이 가능해졌습니다. 헬릭스폴드 팀은 알파폴드 시리즈의 인사이트를 활용하며, 최초로 오픈 소스로 학계 연구에 공개하였습니다.

- **Technical Details**: 헬릭스폴드3는 단백질 데이터 뱅크(Protein Data Bank, PDB)에서 수집한 데이터를 활용하여 훈련되었으며, 소분자 리간드, 핵산(DNA 및 RNA), 단백질 구조를 예측하는 데 AlphaFold3와 유사한 정확도를 보여줍니다. 신뢰도 점수(confidence scores) 평가를 통해 예측 품질을 정량적으로 분석하였습니다.

- **Performance Highlights**: 헬릭스폴드3는 PoseBusters 벤치마크에서 높은 성공률을 기록하며, 실험적으로 입증된 AlphaFold3와 유사한 예측 정확성을 보였습니다. RNA, DNA 구조 예측에서도 RoseTTAFold 모델들을 능가하는 성능을 보여주고 있으며, 단백질-단백질 복합체 구조 예측에서는 AlphaFold-Multimer를 초과한 성공률을 기록했습니다.



### MemLong: Memory-Augmented Retrieval for Long Text Modeling (https://arxiv.org/abs/2408.16967)
- **What's New**: 이 연구는 MemLong: Memory-Augmented Retrieval for Long Text Generation을 소개합니다. MemLong은 외부 리트리버를 활용하여 과거 정보를 검색함으로써 긴 문맥 처리 능력을 향상시키기 위한 방법입니다.

- **Technical Details**: MemLong은 비차별적 'ret-mem' 모듈과 부분적으로 훈련 가능한 디코더 전용 언어 모델을 결합하고, 의미 수준 관련 청크를 활용하는 세밀하고 통제 가능한 검색 주의 메커니즘을 도입합니다. 이 방법은 메모리 뱅크에 저장된 과거 컨텍스트와 지식을 활용하여 키-값(K-V) 쌍을 검색합니다.

- **Performance Highlights**: MemLong은 3090 GPU에서 최대 80k의 문맥 길이로 확장 가능하며, 여러 긴 문맥 언어 모델 기준에서 다른 최신 LLM을 지속적으로 초월하는 성능을 보여주었습니다. MemLong은 OpenLLaMA보다 최대 10.2% 향상된 성능을 나타냅니다.



### UserSumBench: A Benchmark Framework for Evaluating User Summarization Approaches (https://arxiv.org/abs/2408.16966)
- **What's New**: 이 논문은 LLM 기반의 사용자 요약 생성 기법 개발을 촉진하기 위해 설계된 UserSumBench라는 새로운 벤치마크 프레임워크를 도입합니다. 이 프레임워크는 레퍼런스가 없는 요약 품질 메트릭과 강력한 요약 방법을 포함하여 사용자 요약 접근법의 효과성을 평가할 수 있도록 돕습니다.

- **Technical Details**: UserSumBench는 두 가지 주요 구성 요소로 구성됩니다: (1) 참조가 없는 요약 품질 메트릭. 이 메트릭은 세 가지 다양한 데이터셋(영화 리뷰, Yelp, 아마존 리뷰)에서 인간 선호도와 잘 일치하는 효과성을 보여주었습니다. (2) 시간 계층적 요약 생성 및 자기 비판 검증기를 활용한 새로운 요약 방법으로 높은 품질의 요약을 생성하면서 오류를 최소화합니다.

- **Performance Highlights**: 제안된 품질 메트릭은 생성된 요약이 사용자의 향후 활동을 얼마나 잘 예측하는지 평가하여 요약 접근법의 효과를 정량적으로 검토합니다. UserSumBench는 성능 예측과 품질 평가에 있어 강력한 기준선 요약 방법을 제공하며, 이 방법은 향후 요약 기법 혁신의 기초로 작용할 것입니다.



### The Future of Open Human Feedback (https://arxiv.org/abs/2408.16961)
- **What's New**: 이 논문은 언어 모델(LLMs)과의 대화에서 인간 피드백(human feedback)의 수집 방법과 그 중요성을 다룹니다. 논문에서는 개방형 생태계를 위한 피드백을 실현하는 과정에서의 기회와 도전 과제를 interdisciplinary 전문가들이 함께 평가합니다.

- **Technical Details**: 개방형 인간 피드백 생태계의 성공적인 사례를 peer production, open source, citizen science 커뮤니티에서 찾고, 주된 도전 과제를 특징화합니다. 또한, 데이터 수집 방식에서의 지속 가능성(sustainable data)과 참여적 데이터(participatory data)의 중요성을 강조합니다.

- **Performance Highlights**: 이 논문에서는 피드백 루프(feedback loops)와 다양한 이해관계자(stakeholders)를 통해 더 나은 데이터 수집 및 모델 개선을 위한 여러 제안들을 제시하며, 사용자와의 상호작용을 향상시킬 방안을 모색합니다.



### Discovery of False Data Injection Schemes on Frequency Controllers with Reinforcement Learning (https://arxiv.org/abs/2408.16958)
- **What's New**: 이 논문에서는 강화 학습 (reinforcement learning, RL)을 사용하여 전력 시스템의 스마트 인버터 (smart inverters)에서 발생할 수 있는 잠재적인 사이버 공격 (cyber attacks)과 시스템 취약성 (system vulnerabilities)을 탐지하는 방법을 제안하고 있습니다.

- **Technical Details**: 스마트 인버터의 주요 주제 중 하나는 잘못된 데이터 주입 (false data injection, FDI)을 통한 사이버 공격의 위험성입니다. 본 연구에서는 RL 모델을 기반으로 FDI 전략을 분석하고 있으며, 기본 드루프 제어 (default droop control)에 대한 최적의 FDI 전략을 학습함으로써 시스템의 빈도 제어를 손상시킬 수 있는 방법을 탐구하고 있습니다.

- **Performance Highlights**: RL 에이전트가 최적의 FDI 방법을 구별할 수 있다는 결과가 나타났으며, 이는 시스템의 빈도를 변동시킬 수 있는 잠재적인 재앙적 결과를 초래할 수 있습니다. 이 연구는 전력 시스템 운영자들이 사전 예방적 조치를 취할 수 있도록 도와줍니다.



### Transient Fault Tolerant Semantic Segmentation for Autonomous Driving (https://arxiv.org/abs/2408.16952)
Comments:
          Accepted ECCV 2024 UnCV Workshop - this https URL

- **What's New**: 이 연구는 자율주행차의 중요한 기능인 시맨틱 세그멘테이션(semantic segmentation) 분야에서 최초로 하드웨어 오류에 대한 내성을 분석하는 작업입니다. 새로운 활성화 함수 ReLUMax를 도입하여 이 기능의 신뢰성을 향상시키고자 했습니다.

- **Technical Details**: ReLUMax는 트랜지언트 fault(일시적인 오류)에 대한 회복력을 높이기 위해 설계된 단순하고 효과적인 활성화 함수로, 기존 아키텍처에 원활하게 통합될 수 있습니다. 기존의 하드닝 기법들과 비교하여, ReLUMax는 훈련 과정 동안 동적으로 최적의 클리핑 값을 계산하고, 추론 시 클리핑 및 오류 정정을 수행합니다.

- **Performance Highlights**: 실험 결과 ReLUMax는 트랜지언트 오류가 발생해도 높은 정확도를 유지하고, 특히 에러가 심각한 경우에도 예측 신뢰도를 개선하는 효과를 보였습니다. 이는 자율주행시스템의 신뢰성 향상에 기여하게 됩니다.



### Different Victims, Same Layout: Email Visual Similarity Detection for Enhanced Email Protection (https://arxiv.org/abs/2408.16945)
Comments:
          To be published in the proceedings of the ACM Conference on Computer and Communications Security (ACM CCS 2024)

- **What's New**: 이번 연구에서 제안하는 Pisco라는 새로운 이메일 시각적 유사성 탐지 접근법은 기존 스팸 이메일 탐지 시스템의 탐지 능력을 향상시키기 위해 개발되었습니다.

- **Technical Details**: Pisco는 받은 이메일의 HTML 소스 코드를 렌더링하고 전체적으로 렌더링된 이메일의 스크린샷을 캡처하여, 이를 딥러닝을 통해 숫자적 표현으로 변환하고 클러스터링합니다. 주요 기술 요소로는 이미지 임베딩(image embedding)과 OpenAI의 CLIP 모델을 사용하고, 이와 함께 이미지 유사성 탐지를 위하여 Milvus라는 오픈소스 벡터 데이터베이스를 활용합니다.

- **Performance Highlights**: 초기 결과로, 1개월 간 수집한 약 116,000개의 이메일 중 20개의 시각적으로 유사한 스팸 메시지를 탐지했습니다. 이 방법은 스팸 탐지의 어려움을 덜고, 이전의 탐지되지 않은 스팸 이메일과 비슷한 레이아웃을 가진 이메일을 효과적으로 식별할 수 있습니다.



### A longitudinal sentiment analysis of Sinophobia during COVID-19 using large language models (https://arxiv.org/abs/2408.16942)
- **What's New**: COVID-19 팬데믹이 중국인 차별(Sinophobia)을 악화시킨 현상을 대기하는 감정 분석 프레임워크를 제안합니다. 대규모 언어 모델(LLM)을 활용하여 소셜 미디어에서 발언되는 Sinophobic 감정의 변화를 분석했습니다.

- **Technical Details**: 이 연구는 BERT 모델을 미세 조정하여 Sinophobia 관련 트윗의 감정을 분석하며, ’China virus’ 및 ’Wuhan virus’ 같은 키워드를 중심으로 감정을 분류하고 극성 점수를 계산합니다.

- **Performance Highlights**: COVID-19 확진자 수의 급증과 Sinophobic 트윗 및 감정 사이에 유의미한 상관관계를 발견했습니다. 연구 결과는 정치적 내러티브와 잘못된 정보가 대중의 감정 및 의견 형성에 미치는 영향을 강조합니다.



### Event Extraction for Portuguese: A QA-driven Approach using ACE-2005 (https://arxiv.org/abs/2408.16932)
- **What's New**: 이 논문은 포르투갈어에서의 이벤트 추출(event extraction) 작업을 위한 새로운 프레임워크를 제안합니다. 기존의 연구가 영어에 비해 포르투갈어에서 부족했던 점을 해결하기 위해, ACE-2005 데이터셋의 포르투갈어 번역판을 사용하여 이벤트를 식별하고 분류하는 두 개의 BERT 기반 모델을 미세 조정했습니다.

- **Technical Details**: 이벤트 추출 작업은 주로 이벤트 트리거(trigger)와 해당 이벤트의 아규먼트(arguments)를 식별하는 두 가지 하위 작업으로 분해됩니다. 이벤트 트리거 식별은 토큰 분류(token classification) 모델을 사용하고, 아규먼트 추출에는 질문 응답(question answering) 모델을 활용하여 트리거에 대해 아규먼트 역할을 쿼리합니다. 이 과정에서 BERTimbau 모델을 사용하여 포르투갈어 텍스트 데이터로 사전 훈련되었습니다.

- **Performance Highlights**: 제안된 프레임워크는 트리거 분류에서 64.4의 F1 스코어, 아규먼트 분류에서는 46.7의 F1 스코어를 달성했습니다. 이는 포르투갈어 이벤트 추출 작업에서 새로운 최첨단(reference) 성능으로 기록됩니다.



### ACE-2005-PT: Corpus for Event Extraction in Portugues (https://arxiv.org/abs/2408.16928)
- **What's New**: 이 논문에서는 ACE-2005를 포르투갈어로 번역한 ACE-2005-PT 데이터셋을 소개합니다. 이 데이터셋은 유럽 및 브라질 변형을 포함하며, 이벤트 추출(task)에서 사용될 수 있도록 자동 번역 파이프라인을 개발하였습니다.

- **Technical Details**: ACE-2005-PT 데이터셋은 자동 번역을 통해 생성되었으며, lemmatization, fuzzy matching, synonym matching, 그리고 BERT 기반의 word aligner 등의 여러 정렬 기법을 포함하는 alignment pipeline을 적용했습니다. 번역의 정확성을 평가하기 위해 언어 전문가가 수동으로 일부 주석을 정렬했습니다.

- **Performance Highlights**: ACE-2005-PT 데이터셋의 정렬 파이프라인은 정확한 일치율 70.55% 및 완화된 일치율 87.55%를 달성하였으며, 이 데이터셋은 LDC에서 출판 승인을 받았습니다.



### Analyzing Inference Privacy Risks Through Gradients in Machine Learning (https://arxiv.org/abs/2408.16913)
- **What's New**: 본 논문에서는 분산 학습 환경에서 그래디언트 공유로 인한 개인 정보 유출을 체계적으로 분석하는 새로운 접근법을 소개합니다. 기존의 연구들은 그래디언트를 통한 개인 정보 유출의 다양한 위험성을 조사했으나, 본 논문은 이를 통합적으로 다루는 게임 기반 프레임워크를 제시합니다.

- **Technical Details**: 우리는 네 가지 유형의 추론 공격(속성 추론 공격, 속성 추론 공격, 분포 추론 공격, 사용자 추론 공격)을 포괄하는 통합 추론 게임을 정의하였습니다. 그리고 모델의 그래디언트에서 추출된 개인 정보 유출을 정보 이론적 관점에서 분석하며, 다섯 가지 데이터 세트(Adult, Health, CREMA-D, CelebA, UTKFace)와 다양한 적대적 설정 하에서 실험을 수행합니다. 또한, Gradient Pruning, Signed Stochastic Gradient Descent, Adversarial Perturbations, Variational Information Bottleneck, Differential Privacy와 같은 다섯 가지 방어 기법을 평가합니다.

- **Performance Highlights**: 우리는 단순히 데이터 집합에 의존하는 것이 분산 학습에서의 개인 정보 보호를 달성하기에 불충분하다는 것을 보여주었습니다. 예를 들어, Adult 데이터 세트에서만 100개의 그림자가 있는 데이터 샘플로도 0.92의 AUROC를 달성할 수 있었습니다. 대부분의 탐색적 방어 방법은 정적 적대자에 대해서는 성과를 보이지만 적응형 적대자에게는 쉽게 우회당할 수 있습니다.



### GSTAM: Efficient Graph Distillation with Structural Attention-Matching (https://arxiv.org/abs/2408.16871)
Comments:
          Accepted at ECCV-DD 2024

- **What's New**: 본 논문에서는 Graph Distillation with Structural Attention Matching (GSTAM)이라는 새로운 방법론을 제안하여 그래프 분류 데이터셋의 압축을 개선합니다. GSTAM은 GNN의 attention map을 활용하여 원본 데이터셋의 구조적 정보를 합성 그래프로 증류합니다.

- **Technical Details**: GSTAM은 GNN이 중요하게 여기는 입력 그래프의 영역을 효과적으로 추출하는 structural attention-matching 메커니즘을 포함합니다. 이 과정에서 bi-level optimization이 필요하지 않으며, 새로운 손실 함수가 도입되어 synthetic 그래프를 생성합니다. 이는 다양한 그래프 데이터셋에서 실험적으로 검증되었습니다.

- **Performance Highlights**: GSTAM은 극단적인 압축 비율에서 기존 방법들보다 0.45%에서 6.5% 더 나은 성능을 보여주는 등 그래프 분류 작업에서 우수한 결과를 나타냈습니다.



### Physics-Informed Neural Networks and Extensions (https://arxiv.org/abs/2408.16806)
Comments:
          Frontiers of Science Awards 2024

- **What's New**: 이 논문에서는 과학적 기계 학습(SciML)에서 주요 기둥이 된 Physics-Informed Neural Networks (PINNs)의 새로운 방법을 검토하고 최근 실용적인 확장 사항을 제시하며, 데이터 기반의 지배 미분 방정식 발견을 위한 구체적인 사례를 제공합니다.

- **Technical Details**: PINNs는 물리적 모델과 데이터를 통합하는 방식으로, 고전적 수치 계산 방법의 한계를 극복 할 수 있는 가능성을 보여줍니다. 이를 통해 고차원 불확실성을 포함한 복잡한 문제를 해결할 수 있으며, 최근의 연구에서는 Neural Tangent Kernel (NTK) 및 도메인 분해와 같은 새로운 접근법을 통해 학습 속도를 개선하고 있습니다.

- **Performance Highlights**: PINNs의 성능은 비선형 물리 및 역문제에 대해 정확한 해를 수렴함을 보여줍니다. 또한, 물질 속성과 같은 역문제를 다룰 때의 파라미터 추정 정밀도가 높아서 산업 복잡성 응용에서도 유용하게 사용될 수 있습니다. 다양한 확장 연구들은 120차원까지의 확률 미분 방정식을 성공적으로 해결하는 것을 입증했습니다.



### HLogformer: A Hierarchical Transformer for Representing Log Data (https://arxiv.org/abs/2408.16803)
- **What's New**: HLogformer는 로그 데이터에 특화된 최초의 동적 계층형(데이터의 계층 구조를 반영하는) 변환기 모델로, 기존의 단시간 처리 방식을 넘어 이 데이터의 독특한 구조를 활용하여 처리하는 혁신적인 기술이다.

- **Technical Details**: HLogformer는 로그 데이터의 계층적 구조를 활용하여 메모리 비용을 크게 줄이고, 효과적인 표현 학습을 가능하게 한다. 전통적인 모델들과는 달리, HLogformer는 로그 항목들을 고유의 계층적 조직을 유지하며 처리하여 세부 정보와 넓은 맥락 관계를 모두 포괄적으로 인코딩한다.

- **Performance Highlights**: HLogformer는 로그 데이터의 계층적 맥락 정보를 더 효과적으로 인코딩하며, 합성 이상 탐지(synthetic anomaly detection) 및 제품 추천(product recommendation)과 같은 다운스트림(하위 작업)에서 높은 성능을 보여준다.



### Generative AI in Ship Design (https://arxiv.org/abs/2408.16798)
- **What's New**: 이 논문은 전통적인 선박 설계 방법 대신 생성적 AI(Generative AI)를 활용하여 선박 선체 디자인을 최적화하는 새로운 접근 방식을 제시합니다.

- **Technical Details**: 프로세스에는 데이터셋 수집, 모델 아키텍처 선택, 훈련 및 검증 단계가 포함됩니다. 'SHIP-D' 데이터셋(30,000개의 선체 형상 포함)을 활용하여 Gaussian Mixture Model (GMM)을 생성 모델 아키텍처로 채택하였습니다. GMM은 데이터 분포를 분석하는 통계적 프레임워크를 제공하여 혁신적인 선박 디자인 생성을 지원합니다.

- **Performance Highlights**: 이 접근 방식은 넓은 디자인 공간을 탐색하고 다학제적 최적화 목표를 효과적으로 통합함으로써 선박 설계의 혁신을 가져올 가능성을 가지고 있습니다.



### Uncertainty-aware segmentation for rainfall prediction post processing (https://arxiv.org/abs/2408.16792)
Comments:
          Paper accepted at the 3rd Workshop on Uncertainty Reasoning and Quantification in Decision Making at ACM SIGKDD'24 (August 26, 2024, Barcelona)

- **What's New**: 이번 연구는 수치 기상 예측(NWP) 모델의 한계를 극복하기 위해 불확실성 인식 딥 러닝 모델을 탐색합니다. 특히, 매일 누적된 정량적 강수 예측(QPF)을 포스트 프로세싱하여 예측 불확실성을 개선하는 데 중점을 두었습니다. SDE-Net의 변형인 SDE U-Net을 제안하여 세분화(segmentation) 문제를 해결합니다.

- **Technical Details**: 연구에서는 NWP의 직접 모델 출력(DMO)의 불확실성 문제를 다루며, 불확실성을 정량화하는 데 있어 알레아토릭(aleatoric) 및 에피스테믹(epistemic) 불확실성 개념을 활용합니다. 딥 러닝 모델링을 위해 SDE-Net 아키텍처를 기반으로 하여 예측 결과의 분포를 생성할 수 있도록 설계된 SDE U-Net을 적용하였습니다. 

- **Performance Highlights**: SDE U-Net은 평균 기반 NWP 솔루션에 비해 전반적으로 높은 정확성과 신뢰성을 보여주었으며, 다양한 기상 이벤트에 대한 성능을 평가한 결과, 특히 강우 예측의 경우 다른 모델들보다 뛰어난 성능을 발휘했습니다. 운영 기상 예측 시스템에 통합되면 날씨 관련 사고에 대한 준비와 의사결정을 개선할 수 있습니다.



### $EvoAl^{2048}$ (https://arxiv.org/abs/2408.16780)
Comments:
          2 pages, GECCO'24 competition entry

- **What's New**: 이 논문은 2048 게임을 해결하기 위한 해석 가능하고 설명 가능한 정책을 찾기 위해 모델 기반 최적화(model-driven optimisation)를 적용한 연구 결과를 보고합니다.

- **Technical Details**: EvoAl이라는 오픈 소스 소프트웨어를 사용하여 정책 모델을 생성하고 진화 알고리즘(evolutionary algorithms)을 통해 가능한 솔루션을 생성하는 접근 방식을 제안하였습니다. 이 방법은 사용자 친화적인 질의 함수(query functions)를 사용하여 정책의 상태를 확인하고, 이를 통해 결정된 행동을 실행합니다.

- **Performance Highlights**: 최적화 과정의 결과로, 최고 타일 값이 2048에 도달하며 평균 최고 타일은 1.276로 나타났습니다. 이는 생성된 정책이 안정적임을 보여줍니다.



### Inductive Learning of Logical Theories with LLMs: A Complexity-graded Analysis (https://arxiv.org/abs/2408.16779)
- **What's New**: 이 논문은 Formal Inference Engine으로부터 피드백을 받는 Large Language Models (LLMs)의 성능과 한계를 분석하기 위한 새로운 체계적 방법론을 제시합니다. 이 방법론은 규칙 의존성 구조에 따른 복잡성이 등급화되어 있어 LLM의 성능에서 특정 추론 과제를 정량화할 수 있도록 합니다.

- **Technical Details**: 이 연구는 Inductive Logic Programming (ILP) 시스템과의 관계를 통해 LLM의 유도 학습 특성을 평가하는 체계적인 방법론을 제안합니다. 제안한 방법은 LLM과 formal ILP inference engine 및 inductive reasoning dataset 생성을 위한 synthetic generator를 결합하여 LLM이 유도한 이론을 평가하게 됩니다. 정량적인 평가에서는 목표 규칙 집합의 의존성 복잡도에 따라 등급화됩니다.

- **Performance Highlights**: 실험 결과, 최대 LLM은 SOTA ILP 시스템 기준과 경쟁할 수 있는 성과를 보였으나, Predicate 관계 체인을 추적하는 것이 이론 복잡도보다 더 어려운 장애물임을 보여주었습니다.



### Online Behavior Modification for Expressive User Control of RL-Trained Robots (https://arxiv.org/abs/2408.16776)
Comments:
          This work was published and presented at HRI 2024

- **What's New**: 이번 연구는 로봇이 자율적으로 작업을 수행하는 동안 사용자가 행동 스타일을 실시간으로 제어할 수 있는 온라인 행동 수정(online behavior modification)이라는 개념을 제안합니다.

- **Technical Details**: 이 연구에서는 사용자 중심의 다양성 기반 알고리즘, Adjustable Control Of RL Dynamics (ACORD)를 소개합니다. ACORD는 사용자가 사전 정의된 행동 특징에 대해 지속적인 제어를 하는 동시에 자율적인 작업 수행을 보장합니다. 연구에서는 23명의 비전문가 사용자와 함께 ACORD, 순수 RL 및 수정된 Shared Autonomy (SA)를 비교하여 ACORD의 효과를 검증했습니다.

- **Performance Highlights**: ACORD는 사용자가 동일한 수준의 제어를 제공받는다고 평가되었으며, 전반적인 작업 성능 또한 더 높은 것으로 나타났습니다. 예를 들어, 83%의 사용자가 ACORD를 선호하며, ACORD가 사용자의 제어를 더 잘 제공한다고 82%의 사람들이 동의했습니다.



### An Effective Information Theoretic Framework for Channel Pruning (https://arxiv.org/abs/2408.16772)
- **What's New**: 본 논문에서는 정보 이론(information theory)과 신경망 해석 가능성(interpretability)을 바탕으로 한 새로운 채널 프루닝(channel pruning) 방법을 제안합니다. 기존의 채널 프루닝 알고리즘은 레이어별 프루닝 비율을 적절히 지정하는 문제와 덜 중요한 채널을 폐기하는 기준이 미비했던 문제를 다룹니다.

- **Technical Details**: 정보 엔트로피(information entropy)를 사용하여 합성곱 층(convolutional layers)에 대한 기대 정보량을 고려하고, 채널 중요도를 평가하기 위해 샤플리 값(Shapley values)을 활용합니다. 이 연구에서는 정보 집중(information concentration)이라는 개념을 도입하여 레이어별 프루닝 비율을 설정하고, 이는 기존의 휴리스틱 방법이나 엔지니어링 조정을 대체합니다.

- **Performance Highlights**: CIFAR-10 데이터셋에서 ResNet-56 모델의 경우 45.5% FLOPs와 40.3% 매개변수를 제거하면서 정확도가 0.21% 향상되었습니다. ImageNet 데이터셋에서는 ResNet-50 모델에서 41.6% FLOPs와 35.0% 매개변수를 줄이면서 Top-1/Top-5 정확도의 손실이 각각 0.43%와 0.11%에 불과했습니다.



### A Permuted Autoregressive Approach to Word-Level Recognition for Urdu Digital Tex (https://arxiv.org/abs/2408.15119)
- **What's New**: 이 연구는 디지털 우르두 텍스트 인식을 위해 특별히 설계된 새로운 단어 수준의 Optical Character Recognition (OCR) 모델을 소개합니다. 이 모델은 다양한 텍스트 스타일과 글꼴, 변형 문제를 해결하기 위해 transformer 기반 아키텍처 및 attention 메커니즘을 활용합니다.

- **Technical Details**: 모델은 permuted autoregressive sequence (PARSeq) 아키텍처를 사용하여 맥락 인지 추론 및 여러 토큰 순열을 통한 반복적 개선을 가능하게 하여 성능을 향상시킵니다. 이를 통해 우르두 스크립트에서 일반적으로 발생하는 문자 재정렬 및 겹치는 문자를 효과적으로 관리합니다.

- **Performance Highlights**: 약 160,000개의 우르두 텍스트 이미지로 훈련된 이 모델은 CER(Character Error Rate) 0.178을 달성하며 우르두 스크립트의 복잡성을 잘 포착합니다. 비록 몇몇 텍스트 변형을 처리하는 데 지속적인 어려움이 있지만, 이 모델은 실제 응용에서 높은 정확성과 효율성을 보여줍니다.




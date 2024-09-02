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




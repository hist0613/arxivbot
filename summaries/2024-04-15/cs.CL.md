### Pre-training Small Base LMs with Fewer Tokens (https://arxiv.org/abs/2404.08634)
Comments: 15 pages, 6 figures, 10 tables

- **What's New**: 새로운 인접유전 (Inheritune) 방법을 통해 대규모 언어모델(LM)에서 일부 트랜스포머(transformer blocks)를 상속받아 작은 규모의 LM을 개발하는 접근법을 연구하였습니다. 소량의 데이터와 제한된 계산 자원을 사용하여도 뛰어난 성능을 보이는 LM을 생성 가능함을 보여줍니다.

- **Technical Details**: 주요 방법 (Inheritune)은 대형 언어모델의 초기 레이어 몇 개를 그대로 이용하고, 이후 매우 적은 양의 데이터(1B 토큰)로 추가 학습을 진행하는 것입니다. 이 기법은 이론적으로 매우 간단하지만, 다양한 작은 크기의 모델들과 벤치마크에서 우수한 성능을 입증하였습니다.

- **Performance Highlights**: 이 방법으로 개발된 1.5B 파라미터 규모의 LM은 다양한 평가 데이터세트와 MMLU 벤치마크에서 높은 성능을 보였습니다. 다른 1B-2B 크기의 모델들 대비 매우 적은 데이터를 사용했음에도 불구하고 비교적 우수한 성능을 달성했습니다.



### Is ChatGPT Transforming Academics' Writing Style? (https://arxiv.org/abs/2404.08627)
Comments: 15 pages, 19 figures

- **What's New**: 이 논문에서는 2018년 5월부터 2024년 1월까지 제출된 1백만 편의 arXiv 논문을 바탕으로, 통계적 분석을 통해 ChatGPT가 학술 논문 초록의 텍스트 밀도에 미치는 영향을 평가합니다. 컴퓨터 과학 분야에서는 특히 ChatGPT가 수정한 초록의 비율이 약 35%에 달하는 것으로 추정됩니다.

- **Technical Details**: 이 연구는 실제 초록과 ChatGPT로 수정된 초록(시뮬레이션 데이터)을 혼합하여 모델을 보정하고 검증했습니다. 세부적으로는, 단어의 빈도 변화를 분석하여 ChatGPT의 영향력을 측정했습니다. 연구 결과에 따르면, 특정 단어(예: 'significant', 'is', 'are')의 사용 빈도가 눈에 띄게 바뀌었으며, 이는 ChatGPT의 영향을 나타내는 통계적 신호로 해석됩니다.

- **Performance Highlights**: 분석 결과, ChatGPT가 학술 글쓰기 스타일, 특히 초록 작성에 미치는 긍정적 및 부정적 영향을 모두 조명합니다. 또한, ChatGPT의 어휘 선호도와 사용 빈도의 변화를 측정함으로써, 이 기술이 학계에 미치는 영향을 더 구체적으로 파악할 수 있었습니다.



### Synthetic Dataset Creation and Fine-Tuning of Transformer Models for  Question Answering in Serbian (https://arxiv.org/abs/2404.08617)
- **What's New**: 이 연구에서는 적응형 번역-정렬-검색(Translate-Align-Retrieve) 방법을 사용하여 합성 질문 응답(Question Answering, QA) 데이터셋을 생성했습니다. 이 방법을 통해 'SQuAD-sr'이라고 명명된 세르비아어로 된 가장 큰 QA 데이터셋을 만들었으며, 세르비아어의 문자 이중성을 고려하여 키릴 문자(Cyrillic) 버전과 라틴 문자(Latin) 버전 모두를 제작하였습니다.

- **Technical Details**: 세르비아어 QA 모델을 미세 조정(fine-tuning)하기 위해 여러 사전 훈련된 모델(pre-trained models)을 사용했으며, 그중 BERTić 모델이 라틴어 버전의 SQuAD-sr 데이터셋에서 가장 높은 성능을 보였습니다. 이 모델은 XQuAD 데이터셋의 세르비아어 번역본을 평가 기준으로 사용하여 73.91%의 정확도(Exact Match)와 82.97%의 F1 점수를 달성했습니다. 연구 결과 단일 언어 모델(monolingual model)이 다언어 모델(multilingual model)보다 우수하며, 라틴 문자를 사용할 때 성능이 향상되는 것으로 나타났습니다.

- **Performance Highlights**: 추가 분석을 통해 숫자 값이나 날짜에 관한 질문이 다른 유형의 질문보다 정답을 맞힐 가능성이 더 높다는 것을 확인했습니다. 이러한 분석을 바탕으로 SQuAD-sr 데이터셋이 수동으로 작성된 데이터셋이 없는 상황에서 세르비아어 QA 모델을 미세 조정하는 데 충분한 품질을 갖추고 있음을 결론지었습니다.



### Small Models Are (Still) Effective Cross-Domain Argument Extractors (https://arxiv.org/abs/2404.08579)
Comments: ACL Rolling Review Short Paper

- **What's New**: 이 연구에서는 최근 이벤트 인수 추출(EAE)을 위한 온톨로지 이전 효율성을 향상시키기 위해 제안된 두 가지 접근 방법인 질의 응답(Question Answering, QA)과 템플릿 채우기(Template Infilling, TI)가 특히 유망하다는 것을 발견하였습니다. 연구자들은 문장 및 문서 수준에서 여섯 가지 주요 EAE 데이터셋에서 이 두 기술을 사용하여 영역 간(Zero-shot) 전환이 가능한지 실험하였습니다. 이러한 전환은 새로운 온톨로지로의 적용 가능성을 열어줍니다.

- **Technical Details**: 본 논문은 Flan-T5 모델을 사용한 QA 및 TI 기법에 기반하여, 소스 온톨로지가 적절할 경우 GPT-3.5 및 GPT-4보다 우수한 성능을 나타낼 수 있는 것으로 나타났습니다. 연구팀은 모든 온톨로지에 대한 질문과 템플릿을 전문가가 작성하여 제공하며, 이는 학습 전이에 활용될 수 있습니다. 또한, 이 모델들은 다양한 소스 및 대상 데이터셋 조합에 대해 미세 조정 및 평가되었습니다.

- **Performance Highlights**: 실험 결과, Flan-T5 기반의 QA 및 TI 모델들은 GPT-3.5와 GPT-4를 사용한 Zero-shot 평가보다 우수한 성능을 보여주었습니다. 이는 적은 규모의 모델이 적합한 소스 온톨로지를 통해 큰 규모의 언어 모델보다 높은 효율성을 보일 수 있음을 시사합니다. 결과적으로, 이는 EAE 작업에 있어서 새로운 접근 방법의 가능성을 확인시켜주는 중요한 발견입니다.



### CATP: Cross-Attention Token Pruning for Accuracy Preserved Multimodal  Model Inferenc (https://arxiv.org/abs/2404.08567)
- **What's New**: 이 논문은 대형 다중모달(multimodal) 모델에 대한 관심이 증가함에 따라, CATP(Cross-Attention Token Pruning)라는 새로운 정밀도 기반 토큰 가지치기 방법을 소개합니다. CATP는 BLIP-2 모델의 크로스-어텐션(cross-attention) 계층을 활용하여 토큰의 중요성을 결정하는데 필요한 정보를 추출합니다.

- **Technical Details**: CATP는 모델의 헤드(heads)와 레이어(layers)에 걸쳐 정제된 투표 전략을 사용합니다. 각 입력 쿼리 토큰의 중요성은 Q-Former 내의 입력 이미지 토큰과의 크로스-어텐션 확률에 따라 결정됩니다. 이 확률은 각 이미지 토큰과 쿼리 토큰 간의 관련성을 나타내며, 모델 정확도를 유지하기 위해 밀접하게 관련된 쿼리 토큰을 가지치기하지 않는 것이 필요합니다. CATP는 다중 헤드 크로스-어텐션 계층에서 크로스-어텐션 확률 맵을 추출하고, 각 맵에서 이미지 토큰이 쿼리 토큰에 포인트를 투표하는 방식으로 중요도 점수를 계산합니다.

- **Performance Highlights**: 실험 결과, CATP는 기존의 토큰 가지치기 방법들에 비해 최대 12.1배 높은 정확도를 달성하였습니다. 이는 계산 효율성과 모델 정밀도 간의 기존의 트레이드오프를 해결하는 데 기여합니다. CATP의 접근 방식은 크로스-어텐션 메커니즘을 활용하여 쿼리 토큰 가지치기 작업에서 모델 성능을 향상시킬 수 있는 가능성을 보여줍니다.



### MoPE: Mixture of Prefix Experts for Zero-Shot Dialogue State Tracking (https://arxiv.org/abs/2404.08559)
Comments: Accepted to LREC-COLING 2024

- **What's New**: 이 논문에서는 새로운 접근 방식인 MoPE (Mixture of Prefix Experts)를 제안하여 보이지 않는 도메인에 대한 지식을 전달할 수 있는 제로-샷 대화 상태 추적(DST) 모델을 개발했습니다. MoPE는 비슷한 슬롯을 클러스터링하고 각 클러스터에 대한 전문가를 특별히 훈련시켜, 보이지 않는 도메인의 슬롯에 가장 관련성 높은 전문가를 동적으로 할당합니다.

- **Technical Details**: MoPE는 사전 훈련된 대형 언어 모델(LLM) 위에 구축되며, 각각의 전문가는 특수한 접두어 프롬프트로 구성됩니다. 비지도 클러스터링 알고리즘을 사용하여 비슷한 슬롯을 클러스터로 분류하고, 추론 시에는 클러스터 중심을 사용하여 보이지 않는 슬롯에 대한 가장 적합한 전문가를 찾아 대화 상태를 생성합니다. 또한, 모든 전문가를 훈련시키는 데 필요한 비용과 모델 크기를 고려하여, 각 전문가를 위한 파라미터 효율적 파인 튜닝(PEFT) 방법을 사용합니다.

- **Performance Highlights**: MoPE-DST는 MultiWOZ2.1과 SGD 데이터셋에서 각각 57.13%, 55.40%의 합동 목표 정확도(joint goal accuracy)를 달성했습니다. 이는 10B 미만 파라미터를 가진 모든 모델을 크게 능가하는 결과며, 대규모 언어 모델인 ChatGPT와 Codex 보다도 0.20% 높은 성능 향상을 보여주었습니다. 이러한 성과는 비슷한 슬롯을 통해 다른 도메인 간의 연결을 구축하고 도메인 간의 격차를 해소하는 데 중점을 둔 접근 방식이 크게 기여한 것으로 나타났습니다.



### VertAttack: Taking advantage of Text Classifiers' horizontal vision (https://arxiv.org/abs/2404.08538)
Comments: 14 pages, 4 figures, accepted to NAACL 2024

- **What's New**: 이 연구는 수직으로 쓰여진 텍스트를 처리하는 능력이 없는 자동 분류기(automatic classifiers)의 한계를 이용한 새로운 유형의 텍스트 적대적 공격(VertAttack)을 제안합니다. VertAttack은 중요한 단어들을 수직으로 재배열함으로써 분류기의 정확도를 크게 저하시키는 방법입니다.

- **Technical Details**: VertAttack은 정보가 풍부한 단어들을 선택하고, 이를 수직으로 변형하여 입력 텍스트를 조작합니다. 이 연구는 다양한 데이터셋과 트랜스포머 모델(transformer models)에서 VertAttack의 효과를 실험적으로 검증하며, 특히 RoBERTa 모델의 경우 SST2 데이터셋에서 정확도가 94%에서 13%로 대폭 감소했습니다.

- **Performance Highlights**: VertAttack는 검증된 바에 따르면, 평균적으로 분류기의 정확도를 36.6%로 낮추는 반면, 다른 텍스트 공격 방법인 BERT-ATTACK (47.5%) 및 Textbugger (63.2%)와 비교했을 때 더 낮은 성능을 보입니다. 그리고 인간 참여자들은 변형된 텍스트의 77%를 정확히 인식할 수 있었습니다, 변형되지 않은 원본 텍스트의 81%에 근접한 수치입니다.



### Mitigating Language-Level Performance Disparity in mPLMs via Teacher  Language Selection and Cross-lingual Self-Distillation (https://arxiv.org/abs/2404.08491)
Comments: NAACL 2024

- **What's New**: 새로운 연구에서는 다국어 사전 훈련된 언어 모델(mPLMs)이 여러 언어 간 성능 격차를 줄이기 위해 추가적인 다국어 레이블 데이터 없이도 'ALSACE'라는 새로운 접근 방식을 사용하여 효과적으로 개선할 수 있음을 제시합니다. 이 방법은 성능이 우수한 언어로부터 지식을 전달하여 성능이 떨어지는 언어의 성능을 향상시킵니다.

- **Technical Details**: 'ALSACE'는 '교사 언어 선택(Teacher Language Selection)'과 '교차 언어 자기증류(Cross-Lingual Self-Distillation)'의 두 단계로 구성됩니다. 첫 번째 단계에서는 다양한 언어의 출력을 바탕으로 신뢰할 수 있는 교사 언어를 선정하고, 두 번째 단계에서는 선택된 교사 언어의 지식을 다른 언어에 전달하여 모델 출력 분포의 일치를 꾀합니다. 이는 일관성 손실(consistency loss)를 사용하여 이루어집니다.

- **Performance Highlights**: ALSACE는 다양한 mPLMs에서 언어 간 성능 격차를 일관되게 완화시키는 것으로 나타났으며, 자원이 풍부한 언어뿐만 아니라 자원이 부족한 언어에서도 경쟁력 있는 성능을 보여줍니다. 또한, 실제 다국어 자연어 이해(NLU) 작업에서의 실험을 통해 ALSACE가 지식 전이를 효과적으로 수행하여 언어 특이적 지식을 포함한 일반 지식 전달이 가능함을 확인하였습니다.



### Thematic Analysis with Large Language Models: does it work with  languages other than English? A targeted test in Italian (https://arxiv.org/abs/2404.08488)
- **What's New**: 이 논문에서는 영어가 아닌 다른 언어로 된 데이터에서 Large Language Model(LLM)을 이용한 Thematic Analysis (TA)를 수행하는 테스트를 제안합니다. 영어 데이터에서 LLM을 이용한 TA에 대해 초기의 유망한 연구가 있었지만, 이 모델들이 다른 언어에서도 같은 분석을 합리적으로 잘 수행할 수 있는지에 대한 테스트는 부족했습니다. 본 논문에서는 이탈리아어로 된 반구조화된 인터뷰 데이터를 사용하는 테스트를 제안하고 있습니다.

- **Technical Details**: 테스트는 이탈리아어로 진행될 수 있도록 Italy어 프롬프트를 사용하여 사전 훈련된 모델이 데이터에서 TA를 수행할 수 있음을 보여줍니다. 또한, 모델이 인간 연구자가 독립적으로 생성한 주제와 상당히 유사한 주제를 생성할 수 있는 능력을 비교하는 테스트를 포함합니다.

- **Performance Highlights**: 이 테스트의 결과는 사전 훈련된 LLM이 모델이 지원하는 언어라면 다국어 상황에서 분석을 지원하는 데 적합할 수 있음을 시사합니다. 이는 LLM이 다양한 언어의 데이터에 대해 효과적으로 테마 분석을 수행할 수 있음을 보여줍니다.



### Learning representations of learning representations (https://arxiv.org/abs/2404.08403)
- **What's New**: ICLR 데이터세트는 2017년부터 2024년까지 제출된 모든 ICLR 논문의 초록, 메타데이터, 결정 점수 및 사용자 정의 키워드 기반 레이블(Labels)을 포함합니다. 이 데이터를 사용하여 기계 학습(Machine Learning) 분야의 변화를 연구하고, NLP(자연어 처리) 커뮤니티에 대한 도전 과제를 제시합니다.

- **Technical Details**: 데이터는 OpenReview에서 쿼리하여 제목, 초록, 저자 목록, 키워드, 심사 점수 및 컨퍼런스 결정들을 다운로드했습니다. 논문은 45개의 비중첩 클래스(Non-overlapping classes)에 할당됩니다. 초록의 텍스트는 백오브워즈(Bag-of-Words) 표현과 문장 트랜스포머(Sentence Transformers)를 사용하여 임베딩됩니다. TF-IDF가 k𝑘k(이탤릭) NN 분류 정확도에서 대부분의 문장 모델을 능가하고, 언어 모델 향상에 대한 도전 과제를 설정했습니다.

- **Performance Highlights**: TF-IDF는 k𝑘k(이탤릭) NN 정확도에서 59.2%를 달성하여 가장 높은 성능을 보였습니다. 비교를 위해, 특정 과학적 초록을 표현하기 위해 훈련된 세 가지 모델 모두 TF-IDF보다 낮은 성능을 보였고, SBERT와 상업용 모델들만이 TF-IDF를 소폭 상회했습니다. 이는 복잡한 문장 트랜스포머 모델들이 기본 단어 수준의 표현보다 크게 우수하지 않음을 시사합니다.



### Look at the Text: Instruction-Tuned Language Models are More Robust  Multiple Choice Selectors than You Think (https://arxiv.org/abs/2404.08382)
- **What's New**: 이 연구는 대규모 언어 모델(Large Language Models, LLMs)을 평가하는 데 흔히 사용되는 객관식 문제(Multiple Choice Questions, MCQs)에 대한 새로운 평가 방식을 제시합니다. 특히, 첫 번째 토큰의 확률(log probability)을 기반으로 한 평가 대신에 텍스트 답변을 기반으로 하는 평가의 강점을 조명합니다. 이는 기존의 첫 토큰 확률이 문제의 표현 변화에 취약하다는 점과 텍스트 답변과 일치하지 않는 경우가 많다는 이전 연구들의 한계를 극복하려는 시도입니다.

- **Technical Details**: 연구자들은 감정 조절된 대규모 언어 모델(instruction-tuned large language models)을 사용하여, MCQs에 대한 답변이 텍스트 답변에서 얼마나 견고한지(robust)를 조사했습니다. PriDe(PriDe)와 같은 최신 탈편향(debiasing) 방법을 사용하여 첫 번째 토큰 확률을 교정할 때와 비교해, 텍스트 답변이 더 높은 견고성을 보인다는 것을 발견했습니다. 이는 특히 첫 토큰 답변과 텍스트 답변 사이의 불일치가 클 때(50% 이상) 더욱 명백해집니다.

- **Performance Highlights**: 실험 결과, 텍스트 기반 접근 방식은 옵션 순서 변경과 같은 여러 문장 교란 요소에 대한 높은 견고성을 보였습니다. 또한, PriDe를 사용한 첫 토큰 확률 보정에 비해, 텍스트 답변 방식이 옵션 선택 편향(selection bias)이 더 적은 것으로 나타났습니다. 이는 MCQ 평가에서 텍스트 기반 접근 방식이 기존의 확률 기반 방식보다 우수할 수 있음을 시사합니다.



### ASR advancements for indigenous languages: Quechua, Guarani, Bribri,  Kotiria, and Wa'ikhana (https://arxiv.org/abs/2404.08368)
- **What's New**: 이 연구는 NeurIPS 2022의 Second AmericasNLP Competition Track 1에서 제안된 5개의 원주민 언어(Quechua, Guarani, Bribri, Kotiria, Wa'ikhana)에 대한 자동 음성 인식(ASR) 시스템 개발을 목표로 합니다. 특히, Wa'ikhana와 Kotiria 언어의 경우 이전에 보고된 ASR 모델이 없어 이번 연구가 첫 시도입니다. 이 연구는 Wav2vec2.0 XLS-R 모델(300M 및 1B parameters 바리언트)을 기반으로 하며, 데이터 증강(data augmentation)과 반복 학습(fine-tuning)을 사용하여 모델 성능을 최적화하였습니다.

- **Technical Details**: 이 연구팀은 다양한 음성 코퍼스를 크롤링하고 데이터 증강 방법을 적용하여 신뢰할 수 있는 ASR 모델을 구축했습니다. Bayesian search를 사용하여 다양한 하이퍼파라미터(hyperparameters)의 영향을 체계적으로 조사하였고, 특히, 동결된 파인 튜닝(freeze fine-tuning)과 드롭아웃(dropout) 비율이 전체 에포크 수나 학습률(learning rate, lr)보다 중요한 파라미터임을 발견했습니다. 또한, 글로벌 감도 분석(global sensitivity analysis)을 수행하여 최적 모델의 성능에 기여하는 다양한 하이퍼파라미터 설정을 평가했습니다.

- **Performance Highlights**: 개발된 ASR 시스템은 평균 문자 오류율(Character Error Rate, CER)이 26.85로, 경쟁에서 최고의 결과를 달성했습니다. 이는 Wa'ikhana와 Kotiria 언어 모델 개발에서 중요한 이정표입니다. 이 연구는 향후 연구자들이 소수 원주민 언어의 ASR을 개선할 수 있도록 연구 경로를 제공합니다.



### Improving Health Question Answering with Reliable and Time-Aware  Evidence Retrieva (https://arxiv.org/abs/2404.08359)
Comments: Accepted to NAACL 2024 (Findings)

- **What's New**: 이 연구는 오픈 도메인 질문 응답 (open-domain question answering) 시스템이 건강 관련 질문을 처리할 때, 다양한 증거 검색 전략의 성능을 평가합니다. 특히 인용된 문서의 수, 발행 연도, 인용 횟수와 같은 증기 검색 파라미터들의 영향을 관찰하였고, PubMed 데이터베이스를 사용하여 20백만개의 생물의학 연구 초록을 기반으로 실험을 수행하였습니다.

- **Technical Details**: 연구팀은 먼저 관련된 증거를 찾는 '검색자(retriever)'와 이 증거를 바탕으로 질문에 답을 하는 '읽기자(reader)'로 이루어진 기존의 검색-읽기 QA 파이프라인을 사용하였습니다. 검색된 문서의 수, 문장 선택 과정, 기사의 발행 연도 및 인용 횟수 등 다양한 검색 설정을 수정하여 QA 시스템의 성능을 관찰하였습니다.

- **Performance Highlights**: 연구 결과, 검색된 문서의 수를 줄이고 최근에 많이 인용된 문서를 선호하는 것이 최종 매크로 F1 점수를 최대 10%까지 향상시킬 수 있음을 보여주었습니다. 이는 시간을 고려한 증거 검색이 성능을 개선하는 데 효과적임을 시사합니다.



### Gaining More Insight into Neural Semantic Parsing with Challenging  Benchmarks (https://arxiv.org/abs/2404.08354)
- **What's New**: 이 논문에서는 기존의 PMB (Parallel Meaning Bank) 데이터 분할 방법의 문제점을 지적하며, 데이터 분할 및 테스트 셋의 난이도를 개선하기 위한 새로운 접근 방식을 제안합니다. PMB의 기존 성능 평가가 과대평가되었다고 주장하며, 새롭게 도입된 체계적인 데이터 분할 방식와 도전 셋 두 가지(장문 텍스트 및 구성적 일반화를 다룬 셋)를 통해 모델 성능을 실험적으로 평가합니다.

- **Technical Details**: 새로운 데이터 분할은 이전의 무작위 분할 대신 두 단계 정렬 접근법을 사용하여 더 신뢰성 있는 표준 개발 및 테스트 셋을 확립합니다. 또한, CCG (Combinatory Categorical Grammar) 파생 트리를 재결합하여 구성적 일반화 능력을 평가하는 새로운 테스트 셋을 생성하는 고급 기술을 사용합니다. 이러한 접근 방법은 모델이 장문 텍스트 및 복잡한 구문 구조에서도 의미 파싱을 얼마나 잘 수행하는지를 측정하고자 합니다.

- **Performance Highlights**: 결과적으로, 기존 표준 테스트 세트 대비 새롭게 제안된 도전 셋에서 모델의 성능이 눈에 띄게 하락하는 현상을 관찰할 수 있습니다. 이는 실제 세계의 복잡한 언어적 현상들에 대응하기 위한 모델의 한계를 보여주며, 구성적 일반화 및 장문 텍스트 처리에 특화된 추가적인 연구가 필요함을 시사합니다.



### FastSpell: the LangId Magic Sp (https://arxiv.org/abs/2404.08345)
- **What's New**: 이 논문은 언어 식별 도구인 FastSpell을 소개합니다. FastSpell은 fastText (사전훈련된 언어 식별 도구)와 Hunspell (철자 검사기)을 결합하여 텍스트에 할당될 언어를 결정하기 전에 정교한 2차 의견을 제공합니다. 특히, 유사 언어 간의 구분 및 기존 도구에서 무시되는 새로운 언어의 식별에 유용합니다.

- **Technical Details**: FastSpell은 원래 언어 식별 결과에 대해 2차 검사를 수행합니다. fastText를 사용하여 초기 예측을 요청한 후, 예측된 언어가 목표 언어와 유사한 언어 그룹에 속하는 경우, Hunspell 철자 검사기를 사용하여 더 정확히 식별합니다. 이는 fastText의 예측을 재확인하고 유사 언어를 더 잘 구분하는 데 도움을 줍니다. 이 도구는 GPL v3 (GNU General Public License v3) 라이선스 하에 배포되며, Bitextor 및 Monotextor 파이프라인에서 지원 및 유지 관리됩니다.

- **Performance Highlights**: FastSpell은 성능 측정을 통해 기타 언어 식별 도구보다 높은 정확도를 제공하는 것으로 나타났습니다. 특히 유사한 언어를 구별하는 데 큰 장점을 가지며, 기존 도구에서 처리하지 못하는 언어 유형을 식별할 수 있는 능력이 있습니다. 이는 언어 처리 파이프라인의 문맥에서 큰 이점을 제공합니다.



### Toward a Theory of Tokenization in LLMs (https://arxiv.org/abs/2404.08335)
Comments: 58 pages, 10 figures

- **What's New**: 이 논문에서는 토크나이제이션(tokenization) 없이 언어 모델링을 시도하는 기존 연구를 재검토하면서, 토크나이제이션의 이론적인 측면을 탐구합니다. 본 연구에서 밝혀진 새로운 발견은, 트랜스포머(transformers) 모델이 단순한 k차 마르코프 과정(k-th order Markov processes)에서 학습할 때 토크나이제이션 없이는 올바른 분포를 학습하지 못하고 유니그램(unigram) 모델을 사용하여 문자를 예측하는 현상을 경험적으로 관찰한 것입니다.

- **Technical Details**: 트랜스포머는 k차 마르코프 과정에서 데이터를 학습할 때 토크나이제이션을 추가하면, 소스에서 추출된 시퀀스의 확률을 거의 최적으로 모델링하며, 작은 교차 엔트로피 손실(cross-entropy loss)을 달성하는 것이 가능해집니다. 이 연구는 토크나이제이션을 사용함으로써 트랜스포머가 토큰(token)을 기반으로 학습한 가장 단순한 유니그램 모델조차도 마르코프 소스(Markov sources)로부터 추출된 시퀀스의 확률을 거의 최적으로 모델링할 수 있음을 보여줍니다.

- **Performance Highlights**: 토크나이제이션을 통해 트랜스포머는 마르코프 데이터(Markovian data)에 대한 이해를 바탕으로 효율적으로 모델링할 수 있으며, k차 마르코프 과정으로부터의 데이터에 대한 교차 엔트로피 손실을 최소화합니다.



### The Integration of Semantic and Structural Knowledge in Knowledge Graph  Entity Typing (https://arxiv.org/abs/2404.08313)
Comments: Accepted in NAACL2024 main

- **What's New**: 이 논문에서는 지식 그래프(Knowledge Graph, KG) 엔티티 분류(KGET, Knowledge Graph Entity Typing) 작업을 위한 새로운 프레임워크 SSET(Semantic and Structure-aware KG Entity Typing)을 제안합니다. 이는 기존의 구조적 지식(structural knowledge)만 활용하는 방식을 넘어, 엔티티, 관계, 유형의 텍스트 표현에서 중요한 의미적 지식(semantic knowledge)을 함께 활용하여 분류 정확도를 향상시킵니다.

- **Technical Details**: SSET은 세 가지 모듈로 구성됩니다: 1) 의미적 지식 인코딩(Semantic Knowledge Encoding) 모듈은 KG의 사실적 지식을 마스크된 엔티티 타이핑(Masked Entity Typing) 작업을 통해 인코드합니다. 2) 구조적 지식 집계(Structural Knowledge Aggregation) 모듈은 엔티티의 멀티홉 이웃(multi-hop neighborhood)에서 지식을 집계하여 누락된 유형을 추론합니다. 3) 비감독 유형 재순위(Unsupervised Type Re-ranking) 모듈은 상위 모듈에서 얻은 추론 결과를 사용하여 부정확한 샘플에 강건한 타입 예측을 생성합니다.

- **Performance Highlights**: SSET은 광범위한 실험을 통해 기존의 최첨단 방법들보다 높은 성능을 보여줍니다. SSET은 의미적 지식과 구조적 지식의 상호 작용을 통해 오류율(False-negative) 문제를 완화하는 데 기여하는 효과적인 접근 방식임을 확인하였습니다.



### Relational Prompt-based Pre-trained Language Models for Social Event  Detection (https://arxiv.org/abs/2404.08263)
Comments: ACM TOIS Under Review

- **What's New**: 이 연구는 사회적 이벤트 탐지(Social Event Detection, SED)를 위해 새로운 관점을 제시하여, 기존 그래프 신경망(Graph Neural Network, GNN) 방식의 한계를 극복하고자 합니다. 저자들은 관계형 프롬프트 기반 사전 훈련된 언어 모델(Relational prompt-based Pre-trained Language Models, RPLM_SED)을 소개하며, 이는 메시지 쌍을 다중 관계 시퀀스로 구성하여 메시지의 보다 포괄적인 표현을 학습할 수 있는 새로운 메커니즘을 제안합니다.

- **Technical Details**: RPLM_SED는 첫째, 메시지를 쌍으로 모델링하여 구조적 정보를 보존하는 새로운 쌍별 메시지 모델링 전략을 제안합니다. 둘째, 다중 관계 프롬프트 기반의 쌍별 메시지 학습 메커니즘을 통해 메시지 쌍의 의미적 및 구조적 정보에서 보다 상세하고 포괄적인 메시지 표현을 학습합니다. 셋째, 내부 클러스터의 밀도를 높이고 다른 클러스터와의 분산을 증가시키기 위한 새로운 클러스터링 제약을 설계하여 메시지 표현의 구별성을 향상시킵니다.

- **Performance Highlights**: 실제 세계 데이터셋 세 개에서 RPLM_SED 모델을 평가한 결과, 오프라인, 온라인, 저자원, 장기간 배포 시나리오에서의 사회 이벤트 탐지 작업에서 최고의 성능을 달성하였습니다. 이는 PLM의 내재된 능력을 활용하여 지식을 학습, 유지 및 확장함으로써 점진적인 사회 이벤트 탐지를 가능하게 합니다.



### Pretraining and Updating Language- and Domain-specific Large Language  Model: A Case Study in Japanese Business Domain (https://arxiv.org/abs/2404.08262)
Comments: 9 pages. preprint of COLM2024

- **What's New**: 이 연구는 일본어 비즈니스 특정 대규모 언어 모델(LLM)을 개발하여, 비영어권 및 특정 산업 영역에 초점을 맞춘 최초의 사례를 제시합니다. 13 billion parameters를 가진 모델이 신규 데이터셋을 기반으로 처음부터 훈련되어, 지속적으로 최신 비즈니스 문서로 사전 훈련(pretraining)을 계속 받습니다. 또한, 일본어 비즈니스 도메인에 특화된 새로운 벤치마크도 제안됩니다.

- **Technical Details**: 필터링 과정을 거쳐 수집된 데이터셋은 19.8%가 도메인 특화 데이터, 80.2%가 일반 도메인 데이터로 구성되어 있습니다. 벤치마크는 50개의 비즈니스 관련 질문을 포함하며, 세 가지 QA 작업 유형(question-only, question with automatically retrieved context, question with manually retrieved context)을 평가합니다. 모델은 AWS의 Trainium과 같은 고성능 머신러닝 하드웨어 가속기를 활용하여 훈련되었습니다.

- **Performance Highlights**: 이 모델은 일본어 비즈니스 도메인에서 우수한 성능을 보여, 기존 일본어 LLM들을 능가하는 정확도를 달성했습니다. 최신 지식을 반영하여 지속적으로 업데이트되는 사전훈련 모델은 최근 2개월간 발생한 사건들에 관한 질문에 더 정확한 답변을 제공합니다.



### Investigating Neural Machine Translation for Low-Resource Languages:  Using Bavarian as a Case Study (https://arxiv.org/abs/2404.08259)
Comments: Preprint accepted at the 3rd Annual Meeting of the Special Interest Group on Under-resourced Languages (SIGUL 2024)

- **What's New**: 이 연구에서는 독일어와 바이에른어 간 자동 번역 시스템을 개발하는 데 중점을 두면서 저자원 언어의 도전과제를 다루고 있습니다. 이는 저자원 언어에 대한 연구가 부족한 상황에서 큰 의미를 가집니다. 특히, 데이터 부족과 매개변수 민감도 같은 여건을 조사하고, 언어 유사성을 활용하는 창의적인 해결책을 제시합니다.

- **Technical Details**: 실험에서는 Transformer (트랜스포머) 모델을 기반으로 하여, Back-translation (백 번역)과 Transfer Learning (전이 학습)을 사용하여 추가 훈련 데이터를 자동으로 생성하고 높은 번역 성능을 달성하려고 시도했습니다. 또한, 다양한 메트릭스(BLEU, chrF, TER)를 조합하여 평가를 수행했으며, Bonferroni (보니페로니) 보정을 통한 통계적 유의성 결과도 제시되었습니다.

- **Performance Highlights**: Back-translation을 사용한 접근법이 기존 시스템 대비 유의미한 개선을 이루었음을 보여주었습니다. 제안된 모델은 데이터의 노이즈를 처리하고, 정확도 높은 번역을 제공하기 위한 텍스트 전처리 방법을 광범위하게 시행함으로써 향상된 성능을 선보였습니다.



### Measuring Cross-lingual Transfer in Bytes (https://arxiv.org/abs/2404.08191)
Comments: NAACL 2024

- **What's New**: 이 연구는 다양한 언어로 훈련된 모델이 타깃 언어에서 어떻게 지식을 전달하는지에 대한 새로운 이해를 제공합니다. 본 논문에서는 언어 구체적(language-specific) 요소와 언어 중립적(language-agnostic) 요소가 모두 중요하다는 것을 발견하였으며, 특히 언어 중립적 요소가 보편적 지식의 전달을 담당한다는 가설을 강화합니다.

- **Technical Details**: 연구자들은 '데이터 전송'(Data Transfer, DT) 지표를 사용하여 소스 언어에서 타깃 언어로의 지식 전달을 측정하였습니다. Byte-level 토크나이저(byte-level tokenizer)와 자기 지도 학습(self-supervised learning) 방식을 통해 훈련된 언어 모델(language model)을 사용했습니다. 이 모델들은 다양한 소스 언어에서 초기화하여 타깃 언어로 미세조정(finetuning) 과정을 거쳤고, 각 모델의 타깃 언어에 대한 성능을 비교 분석했습니다.

- **Performance Highlights**: 실험 결과, 이러한 모델들은 생각 외로 모든 타깃 언어(스페인어, 한국어, 핀란드어 등 포함)에 대해 유사한 성능을 보여주었고, 이는 언어 오염(language contamination)이나 언어 근접성(language proximity)과는 관련이 없음을 시사합니다. 이는 언어 모델이 언어에 구애받지 않는 지식을 학습하고 있음을 보여주며, 다양한 언어 간의 지식 전달 가능성을 새롭게 열어줍니다.



### Multimodal Contextual Dialogue Breakdown Detection for Conversational AI  Models (https://arxiv.org/abs/2404.08156)
Comments: Published in NAACL 2024 Industry Track

- **What's New**: 이 논문에서는 대화 분절(Dialogue Breakdown)을 실시간으로 감지하는 Multimodal Contextual Dialogue Breakdown (MultConDB) 모델을 소개합니다. 이 모델은 오디오 입력과 전사된 텍스트에 대한 NLP 모델의 추론을 실시간으로 처리하여 산업 환경에서의 대화 분절을 효과적으로 감지합니다.

- **Technical Details**: MultConDB 모델은 기존의 최고 모델들을 크게 앞서는 성능을 보여주며, F1 점수는 69.27에 이릅니다. 모델은 대화의 역사와 상태에 따라 유연하게 대화를 이끌 수 있는 능력이 필요한 헬스케어와 같은 산업 환경에서 특히 중요합니다. 이 모델은 음성 인식 시스템(ASR)과 텍스트 신호를 모두 사용하여 대화 분절을 감지합니다.

- **Performance Highlights**: MultConDB 모델은 전화 통화에서 평균 100턴의 대화에서 발생할 수 있는 복잡하고 다양한 흐름을 처리할 수 있으며, 이는 특히 전화를 통한 사용자 상호작용에서 중요합니다. 이 모델은 빠른 반응 시간을 유지하면서도 정확도가 높아 산업 환경에서의 요구사항을 만족시키는 데 효과적입니다.



### Graph Integrated Language Transformers for Next Action Prediction in  Complex Phone Calls (https://arxiv.org/abs/2404.08155)
Comments: Published in NAACL 2024 Industry Track

- **What's New**: 이 논문은 대화 관리자의 복잡성을 줄이고, 외부 자원의 의존성 없이 다음 행동을 예측하는 새로운 기법을 제안합니다. 그래프 통합 언어 변환기(Graph Integrated Language Transformers)는 인간의 발화와 행동 간의 관계를 파악하여 다음 행동을 예측합니다.

- **Technical Details**: 제안된 모델은 언어 변환기(language transformers)와 그래프 컴포넌트(graph component)를 결합하여, NLU 파이프라인(Natural Language Understanding pipelines)의 의존성을 제거합니다. 이러한 접근 방식은 대화의 현재 및 이전 상태를 추적하는 데 필요한 외부 정보 추출(예: 슬롯-필링(Slot-Filling) 및 의도 분류(Intent-Classification)) 없이 작동합니다.

- **Performance Highlights**: 실제 통화 데이터에 대한 실험 분석에서, 그래프 통합 언어 변환기 모델은 기존의 대화형 AI 시스템보다 높은 성능을 달성하였습니다. 이 모델은 실시간 제약 조건(예: 출력 생성의 지연(latency))과 인간 사용자 경험을 고려한 평가에서 강력한 성능을 보였습니다.



### Distilling Algorithmic Reasoning from LLMs via Explaining Solution  Programs (https://arxiv.org/abs/2404.08148)
Comments: pre-print

- **What's New**: 본 연구에서는 대규모 언어 모델(LLM)의 추론 능력을 향상시키기 위한 새로운 접근 방법을 제안합니다. 기존의 문제 해결(chain-of-thought, CoT)을 기반으로 하는 대신, LLM이 해결책을 설명하는 능력을 활용하여 '어떻게 문제를 해결할 것인가'에 대한 힌트를 생성하는 'Reasoner' 모델을 훈련시키는 방식입니다. 이 접근법은 특히 경쟁 수준의 프로그래밍 문제에서 기존 모델들보다 더 높은 성공률을 보여줍니다.

- **Technical Details**: 우리의 방식은 큰 LLM에서 문제 설명과 함께 휴먼 레벨의 프로그램 솔루션을 읽고, 그 솔루션을 설명하는 스타일의 추론 과정을 생성하는 'Explainer' 모델 사용을 포함합니다. 그 다음, 생성된 <문제, 설명> 쌍을 사용하여 작은 LLM을 신중하게 튜닝합니다. 이렇게 훈련된 'Reasoner'는 보이지 않는 문제에 대해 중간 추론 과정을 생성할 수 있으며, 이후 'Coder' 모델이 최종 코드를 구현합니다. 이 과정은 CoT 식 추론을 활용하는 기존 방법들과 비교하여 데이터 효율성과 문제 일반화에서 우월함을 보여주었습니다.

- **Performance Highlights**: 실험 결과, 이 방식은 경쟁 수준의 프로그래밍 문제들에 대한 해결률을 크게 향상시켰습니다. 특히, GPT-3.5와 Deepseek Coder 7b 모델에서는 입증된 다양한 데이터셋(8248 데이터 포인트, 8M 토큰)을 바탕으로 일관된 성능 향상을 관찰했습니다. 우리는 코드의 정확성뿐만 아니라 효율성 또한 고려해야 한다고 주장하며, 새로운 테스트 세트를 Codeforces 형식으로 제안하여 공정한 평가를 제안합니다.



### HLTCOE at TREC 2023 NeuCLIR Track (https://arxiv.org/abs/2404.08118)
Comments: 6 pages. Part of TREC 2023 Proceedings

- **What's New**: HLTCOE 팀은 TREC 2023 NeuCLIR 트랙에 참여하여 여러 차세대 기술로 클로스-랭귀지 인포메이션 리트리벌 (Cross-Language Information Retrieval, CLIR) 및 멀티-랭귀지 인포메이션 리트리벌 (Multilingual Information Retrieval, MLIR) 작업을 수행했습니다. 주요 사용 기술은 mT5 리랭커 (mT5 reranker), PLAID, 그리고 문서 번역 기법이었습니다. 이들은 다양한 언어로 이루어진 새로운 예측 모델 및 번역 기술을 집중적으로 실험하여 기존 방식을 향상시키기 위한 노력을 보여주었습니다.

- **Technical Details**: 팀은 ColBERT-X 모델 라이언을 사용하여 클로스-랭귀지 정보 검색 모델을 트레이닝하고, 번역 학습(Translate-Train, TT) 및 번역 증류(Translate-Distill, TD)과 같은 첨단 방법론을 적용했습니다. TT는 문서 언어로 자동 번역된 MS-MARCO v1 컬렉션의  쿼리와 패시지들을 트레이닝하는 기법이며, MTT (Multilingual Translate-Train)는 세 가지 문서 언어로 번역된 패시지를 혼합하여 단일 모델을 구축합니다. 또한, BM25 기반의 Patapsco 프레임워크를 사용하여 기본 성능을 평가하고 이를 mT5 리랭커로 재랭킹했습니다. mT5 리랭커는 각 쿼리-문서 쌍의 점수를 학습하는 깊은 학습기반이며, 특히 중국어 클로스-랭귀지 태스크에서 효과적인 순서 반전 기술을 사용했습니다.

- **Performance Highlights**: PLAID 및 Translate-Distill 모델은 NeuCLIR 트랙에서 매우 성공적이었습니다. 특히 Translate-Distill 방식이 Translate-Train 방식보다 뛰어난 성능을 보였으며, mT5 기반 모델은 문서 번역을 색인화하는 단일언어 ColBERT 모델보다 우수한 성능을 나타내었습니다. 이러한 성과는 다언어 및 다국어 정보 접근과 관련된 기술적 진보를 의미하며, 심층적인 번역 및 리랭킹 접근 방식이 향후 CLIR과 MLIR 문제 해결에 중요한 역할을 할 것임을 시사합니다.



### Data-Augmentation-Based Dialectal Adaptation for LLMs (https://arxiv.org/abs/2404.08092)
- **What's New**: 이 보고서는 대규모 언어 모델(LLMs)의 상식적 추론 능력을 남슬라브 소방언에서 평가하는 VarDial 2024의 Dialect-Copa 공동 작업에 GMUNLP가 참여한 내용을 소개합니다. 비표준 방언 변종을 처리할 수 있는 LLM의 능력을 평가하기 위해 데이터 증강 기술을 결합한 접근 방식을 제안하여 세 남슬라브 소방언인 차카비아어(Chakavian), 체르카노어(Cherkano), 토를라크어(Torlak)에서의 성능을 향상시켰습니다.

- **Technical Details**: BERTić(언어 가족 중심 인코더 기반 모델)과 AYA-101(도메인 불특정 다국어 모델)을 사용한 실험을 수행했습니다. 또한, 데이터 증강 기술을 사용하여 실제 학습 데이터와 결합하고, 세 가지 범주의 언어 모델(저자원 설정에 적합한 소형, 작업 특정 성능 및 언어 이해 기능 간의 균형을 유지하는 중간 크기, 고품질 합성 작업 데이터를 생성하는 폐쇄 소스)을 사용하여 방언 작업 성능을 극대화하는 데 초점을 맞추었습니다.

- **Performance Highlights**: GMUNLP의 접근 방식은 모든 세 개의 테스트 데이터 세트에서 최고 점수를 달성했으며, GPT-4 제로샷(zero-shot) 반복적 프롬프팅 접근법을 사용한 팀과 동등한 성능을 보였습니다. 데이터 증강 전략을 결합한 BERTić은 특히 저자원 설정에서 언어 모델의 성능을 향상시키는 데 효과적임을 보여주었습니다.



### SQBC: Active Learning using LLM-Generated Synthetic Data for Stance  Detection in Online Political Discussions (https://arxiv.org/abs/2404.08078)
- **What's New**: 이 연구에서는 온라인 정치 토론에서 스탠스(stance) 탐지 모델의 성능을 향상시키기 위해 LLM(대규모 언어 모델)에서 생성된 합성 데이터를 활용하는 두 가지 새로운 방법을 제안합니다. 첫 번째는 합성 데이터로 기존의 미세조정 데이터 세트를 보완하고, 두 번째는 'Query-by-Comittee' 접근 방식을 기반으로 한 새로운 활동 학습(Active Learning) 방법인 SQBC를 제안합니다.

- **Technical Details**: 이 논문에서는 BERT와 같은 사전 훈련된 언어 모델을 사용하여 주어진 스탠스 탐지 데이터 세트에 대해서 미세 조정(Fine-tuning)을 수행합니다. 합성 데이터를 통해 특정 질문에 대한 모델의 성능을 개선하는 첫 번째 방법과, 합성 데이터를 오라클로 사용하여 가장 유익한 레이블되지 않은 샘플을 식별하는 SQBC 방법을 도입합니다. 이를 통해 레이블링 작업을 줄이면서 동시에 모델 성능을 최대화할 수 있습니다.

- **Performance Highlights**: 실험 결과, 합성 데이터로 데이터 세트를 보완하는 것만으로도 스탠스 탐지 모델의 성능이 향상됨을 확인하였고, SQBC 방법을 통해 활동 학습을 수행할 경우 전체 데이터 세트를 사용했을 때 보다도 더 나은 성능을 달성할 수 있었습니다. 이는 합성 데이터와 활동 학습의 조합이 스탠스 탐지 작업에서 효과적일 수 있음을 시사합니다.



### MSciNLI: A Diverse Benchmark for Scientific Natural Language Inferenc (https://arxiv.org/abs/2404.08066)
Comments: Accepted to the NAACL 2024 Main Conference

- **What's New**: 최근 제안된 과학적 자연어 추론(Natural Language Inference, NLI) 작업은 연구 논문에서 추출한 두 문장 사이의 의미 관계를 예측하는 것입니다. 이 연구에서는 과학적 NLI 작업에 다양성을 도입하고자 MSciNLI 데이터셋을 제시했으며, 이는 컴퓨터 언어학(Computational Linguistics)을 포함한 다섯 개의 새로운 과학 분야에서 추출한 132,320 개의 문장 쌍을 포함하고 있습니다.

- **Technical Details**: 연구팀은 Pre-trained Language Models (PLMs)와 Large Language Models (LLMs)를 일련의 베이스라인(baselines)으로 세세하게 조정하여 MSciNLI 데이터셋은 튜닝하여 실험을 진행하였습니다. 이 과정을 통해 다양한 도메인에서의 도메인 이동(domain shift)을 연구할 수 있습니다. 또한, 과학적 NLI 모델의 성능 저하가 각 도메인의 다양한 특성을 보여주는 것으로 나타났습니다.

- **Performance Highlights**: PLM과 LLM 베이스라인의 최고 Macro F1 점수는 각각 77.21%와 51.77%로, MSciNLI가 두 모델 유형 모두에게 도전적인 데이터셋임을 보여주고 있습니다. 더불어, 연구진은 과학적 NLI 작업 과정에서 중간 과제 전이 학습(intermediate task transfer learning)을 사용하여 과학적 도메인에서 하류 작업(downstream tasks)의 성능을 개선할 수 있음을 보였습니다.



### RLHF Deciphered: A Critical Analysis of Reinforcement Learning from  Human Feedback for LLMs (https://arxiv.org/abs/2404.08555)
- **What's New**: 본 논문은 인간의 피드백에서 강화 학습(Reinforcement Learning from Human Feedback, RLHF)을 이용하여 대규모 언어 모델(Large Language Models, LLMs)을 인간의 목표와 일치시키기 위한 최신 연구를 분석합니다. 이 연구는 특히 보상 모델(reward model)의 핵심 요소에 중점을 두어, 보상 모델의 설계 선택과 그 한계를 탐구하고, 이를 통해 언어 모델의 성능에 미치는 영향을 심층적으로 검토합니다.

- **Technical Details**: RLHF는 최초로 인간의 선호도에 대한 피드백을 이용해 보상 기능 없이 문제를 해결하는 방법으로 주목받았습니다. 본 연구에서는 베이지안 관점(Bayesian perspective)을 채택하여 RLHF의 보상 함수의 중요성을 강조하고, 정확한 보상 모델을 학습하기 위해 필요한 피드백 데이터의 광범위한 요구 사항과, 한정된 피드백 데이터와 함수 근사(function approximation) 사용이 훈련 중 보이지 않은 입력에 대해 부정확한 보상 값을 할당하는 오류 일반화(misgeneralization)를 일으키는 문제를 분석하였습니다.

- **Performance Highlights**: 이론적으로 이상적인 보상(Oracular Reward)의 개념을 정립함으로써, RLHF의 현재 방법론의 한계를 명확히 하고, 보상 희소성(reward sparsity), 보상 모델 명세 오류(reward model misspecification) 등이 언어 모델의 성능에 미치는 영향을 구체적으로 설명합니다. 또한, 연구는 RLHF의 다양한 구성 요소를 개선하기 위한 최근 연구 동향에 대해 탐구하며, 이를 통해 향후 연구 방향성에 대한 이해를 돕습니다.



### Online Safety Analysis for LLMs: a Benchmark, an Assessment, and a Path  Forward (https://arxiv.org/abs/2404.08517)
- **What's New**: 이 논문은 Large Language Models (LLMs)의 온라인 안전 분석 분야에서 새로운 연구를 제시합니다. 기존의 안전성 분석 방법들이 대부분 생성된 결과물에 대한 사후 분석에 초점을 맞춘 반면, 이 연구는 생성 과정 초기부터 안전성을 평가할 수 있는 방법을 탐색합니다. 뿐만 아니라, LLMs의 온라인 안전성 분석을 위한 최초의 공개 벤치마크를 설정하여 다양한 방법, 모델, 작업, 데이터세트 및 평가 지표를 포함시킵니다.

- **Technical Details**: 연구 팀은 온라인 안전성 분석의 효과성을 평가하기 위해 다양한 최신 방법들을 실험하였습니다. 이를 위해 오픈 소스(open-source) 및 클로즈드 소스(closed-source) LLMs에 대한 광범위한 분석을 수행하였고, 개별 방법들의 장단점을 밝혀냈습니다. 또한, 여러 방법을 결합하는 하이브리드(hybridization) 방식을 통해 LLMs의 온라인 안전성 분석의 효율성을 제고할 수 있는 가능성을 탐구하였습니다.

- **Performance Highlights**: 이 연구는 다양한 온라인 안전성 분석 방법들의 성능을 체계적으로 비교하고, 특정 응용 프로그램 시나리오 및 작업 요구 사항에 따라 가장 적합한 방법을 선택하는 데 유용한 통찰력을 제공합니다. 하이브리드 방법을 사용함으로써, 여러 개별 방법의 장점을 조합하여 LLMs의 안전성을 더욱 향상시킬 수 있는 가능성이 확인되었습니다.



### Leveraging Multi-AI Agents for Cross-Domain Knowledge Discovery (https://arxiv.org/abs/2404.08511)
- **What's New**: 이 연구는 다양한 지식 분야에서 전문화된 다중 AI 에이전트를 활용하여 영역 간 지식 발견에 대한 새로운 접근 방식을 소개합니다. 이러한 AI 에이전트들은 도메인별 전문가로서 기능하며, 단일 영역 전문 지식의 한계를 초월하는 포괄적인 통찰력을 제공하기 위해 통합된 프레임워크 내에서 협력합니다.

- **Technical Details**: AI 에이전트는 각자의 도메인 전문 지식을 활용하여 복잡한 문제를 해결하기 위해 협력합니다. 각 에이전트는 관찰(observe), 사고(think), 행동(act)의 순서로 데이터를 처리하고 문제 해결 과정에 기여합니다. 이 멀티-에이전트 시스템은 MetaGPT를 사용하여 다양한 에이전트 간의 정보 전달을 관리하며, 통합된 지식 생성을 위한 복잡한 협업 문제를 효과적으로 해결합니다.

- **Performance Highlights**: 실험을 통해 멀티-AI 에이전트 시스템이 다양한 분야에서의 지식 통합 및 결정 정확도에서 우수한 성능을 보였습니다. 멀티-AI 에이전트 구성의 효율성, 정확성, 지식 통합의 폭을 평가한 결과, 도메인별 전문 지식을 활용한 협업이 강화된 지식 발견과 의사결정 과정을 향상시킬 수 있는 잠재력을 갖고 있음을 확인하였습니다.



### Efficient Interactive LLM Serving with Proxy Model-based Sequence Length  Prediction (https://arxiv.org/abs/2404.08509)
Comments: Accepted at AIOps'24

- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)의 효율적인 코어 서빙을 위한 새로운 스케줄링 방법을 제시한다. 기존의 FCFS(First-Come-First-Serve) 방식 대신, SSJF(Speculative Shortest-Job-First) 스케줄러를 사용하여 요청 처리 순서를 예측하고 최적화한다. SSJF는 가벼운 프록시 모델을 사용하여 LLM 출력 시퀀스 길이를 예측하고 이를 기반으로 작업 실행 시간을 추정한다.

- **Technical Details**: SSJF 스케줄러는 실행 시간의 불확실성을 해결하고 대기 시간(hold-of-line blocking issue)을 줄이면서, 메모리 관리나 배치 전략을 변경할 필요 없이 기존 LLM 서빙 시스템에 적용 가능하다. 이 스케줄러는 다양한 배치 설정(No batching, Dynamic batching, Continuous batching)에 걸쳐 효과적으로 작동하며, 라이트 프록시 모델로서 정교하게 튜닝된 BERT-base 모델을 사용한다.

- **Performance Highlights**: 실제 월드 데이터셋과 프로덕션 워크로드 추적을 통한 평가에서 SSJF는 평균 작업 완료 시간을 30.5-39.6% 단축시키고 처리량을 2.2-3.6배 증가시킨다는 결과를 보였다. 이러한 성능 개선은 LLM을 이용한 인터랙티브 애플리케이션의 사용자 경험을 크게 향상시킬 수 있다.



### Dataset Reset Policy Optimization for RLHF (https://arxiv.org/abs/2404.08495)
Comments: 28 pages, 6 tables, 3 Figures, 3 Algorithms

- **What's New**: 본 연구에서는 Reinforcement Learning from Human Preferences (RLHF)에서 '데이터셋 리셋 정책 최적화(Dataset Reset Policy Optimization, DR-PO)'라는 새로운 알고리즘을 제안합니다. 이 알고리즘은 온라인 정책 훈련 과정에서 기존의 오프라인 선호 데이터셋을 사용하여 정책 최적화기를 오프라인 데이터셋의 상태로 직접 리셋하는 방식을 통합합니다. 이는 기존에 초기 상태 분포에서 시작하는 것과 달리 데이터셋을 활용하여 보다 효과적인 학습이 가능하도록 설계되었습니다.

- **Technical Details**: DR-PO는 간단한 구현이 가능하며, 자연스러운 가정 하에 강력한 이론적 보장을 제공합니다. 특히, 오프라인 선호 데이터셋에서 학습된 보상 모델을 최적화할 때, DR-PO는 해당 데이터에 포함된 모든 정책보다 적어도 그만큼 좋거나 더 좋은 성능을 보이며 이는 일반 함수 근사와 유한 샘플 복잡성 하에 달성됩니다. 또한, DR-PO는 최대 우도 추정(Maximum Likelihood Estimation, MLE) 오라클과 최소 제곱 회귀(Least Squares Regression) 오라클만을 필요로 하므로 계산적으로 효율적입니다.

- **Performance Highlights**: DR-PO는 TL;DR 요약과 Anthropic Helpful Harmful (HH) 데이터셋에서 각각 Proximal Policy Optimization (PPO) 및 Direction Preference Optimization (DPO)보다 우수한 성능을 보이는 것으로 나타났습니다. 특히 GPT-4 승률 지표에 따르면, DR-PO를 통해 생성된 요약본이 기존 방법들과 비교했을 때 보다 높은 성과를 보였으며, CNN/DailyMail 뉴스 기사로의 제로샷 학습에서도 DR-PO의 전략이 다시 한번 높은 성능을 보여 과적합(overfitting) 없이 좋은 결과를 나타냈습니다.



### Decoding AI: The inside story of data analysis in ChatGP (https://arxiv.org/abs/2404.08480)
Comments: 15 pages with figures and appendix

- **What's New**: 최근 생성적 AI (Generative AI)의 발전으로 데이터 과학(Data Science) 분야는 다양한 변화에 직면하고 있습니다. 이 검토는 다양한 작업에 걸쳐 ChatGPT의 데이터 분석(Data Analysis, DA) 기능을 평가하여 성능을 평가합니다.

- **Technical Details**: ChatGPT는 고급 분석 기능을 제공하여 연구자와 실무자들에게 전례 없는 분석 능력을 제공하지만, 완벽하지는 않습니다. 따라서 이 기술의 한계를 인식하고 해결하는 것이 중요합니다.

- **Performance Highlights**: ChatGPT의 데이터 분석(Data Analysis) 성능이 폭넓은 작업에 걸쳐 평가되었으며, 그 결과 다양한 데이터 과학 작업에 대한 그 기능이 평가되고 잠재적인 한계점도 검토되었습니다.



### AdapterSwap: Continuous Training of LLMs with Data Removal and  Access-Control Guarantees (https://arxiv.org/abs/2404.08417)
- **What's New**: 언어 모델이 점점 더 많은 정보를 필요로 하는 작업을 완수할 수 있는 능력을 갖추고 있지만, 데이터 요구사항이 변화함에 따라 새로운 데이터 배치, 사용자 기반의 데이터 접근 제어, 문서의 동적 제거와 관련된 요구사항을 충족시키기 위한 새로운 방법이 필요합니다. 이러한 문제를 해결하기 위해 'AdapterSwap'이라는 새로운 학습 및 추론 스키마(training and inference scheme)를 소개합니다. AdapterSwap은 데이터 컬렉션에서의 지식을 낮은 순위의 어댑터(low-rank adapters) 세트로 구성하고 이를 추론 과정에서 동적으로 조합합니다.

- **Technical Details**: 'AdapterSwap'은 데이터가 진화하는 요구사항에 맞추어 모델이 구식 정보를 잊지 않으면서 새로운 데이터를 통합할 수 있도록 설계된 새로운 학습 및 추론 방법입니다. 이 기법은 어댑터를 사용하여 지식을 조직하고, 이를 추론 중에 동적으로 조합하여 지속적인 학습(continual learning)을 가능하게 합니다. 또한, 어댑터를 통해 데이터 접근 및 삭제를 미세하게 제어할 수 있습니다.

- **Performance Highlights**: AdapterSwap은 효율적인 지속적 학습을 지원하는 동시에 조직이 데이터 접근과 삭제를 세밀하게 제어할 수 있도록 함으로써, 기존의 대규모 언어 모델이 가지는 한계를 극복합니다. 실험을 통해 이러한 기능의 효과가 입증되었습니다.



### Subtoxic Questions: Dive Into Attitude Change of LLM's Response in  Jailbreak Attempts (https://arxiv.org/abs/2404.08309)
Comments: 4 pages, 2 figures. This paper was submitted to The 7th Deep Learning Security and Privacy Workshop (DLSP 2024) and was accepted as extended abstract, see this https URL

- **What's New**: 이 논문에서는 대형 언어 모델(Large Language Models, LLMs)의 프롬프트(Prompt) 탈옥(Jailbreaking)에 중점을 둔 새로운 접근 방식을 제안합니다. 특히, '부분 유해 질문(Subtoxic Questions)'이라는 새로운 범주의 질문을 도입하고 그 효용성을 설명하여, LLM의 보안을 강화하는 동시에 취약점을 평가하는 데 사용될 수 있는 새로운 프레임워크를 발전시킵니다.

- **Technical Details**: 이 연구에서 도입된 '부분 유해 질문'은 외관상 해가 없으나 LLM에 의해 잘못 분류될 가능성이 있는 질문들입니다. 이를 통해 연구자들은 점차적 태도 변경(Gradual Attitude Change, GAC) 모델을 개발하였습니다. 이 모델은 사용자의 프롬프트와 LLM의 반응 간의 상호작용을 이해하고, 보안과 기능성의 균형을 조사합니다. GAC 모델은 응답의 점진적 변화를 수학적으로 설명하며, 효과적인 프롬프트의 중첩 사용이 탈옥 성능을 개선할 수 있음을 보여줍니다.

- **Performance Highlights**: 실험을 통해 연구진은 부분 유해 질문이 진짜 유해 질문보다 탈옥 프롬프트에 더 효과적으로 반응한다는 것을 밝혔습니다. 이는 부분 유해 질문이 블랙박스 LLM 시스템의 탈옥 실험에 유용한 자원이 될 수 있음을 시사합니다. 또한, 연구 결과는 점차적 태도 변경 모델을 통해 얻은 통찰력이 LLM의 보안 강화와 진단적 평가 방법론 개선에 기여할 수 있음을 보여줍니다.



### Reducing hallucination in structured outputs via Retrieval-Augmented  Generation (https://arxiv.org/abs/2404.08189)
Comments: To be presented at NAACL 2024. 11 pages and 4 figures

- **What's New**: 이 논문은 자연어 요구사항을 작업 흐름(workflow)으로 변환하는 상업적 응용 프로그램을 개발하는 과정에서 발생하는 환각(hallucination) 문제를 줄이기 위해 검색 기능을 향상된 생성(Retrieval Augmented Generation, RAG)을 사용하는 새로운 시스템을 제안합니다. 이 시스템은 대규모 언어 모델(Large Language Models, LLM)의 출력에서 환각을 크게 감소시키고 학습된 소규모 검색 인코더를 사용함으로써 LLM의 크기를 줄이고 리소스 사용을 최소화합니다.

- **Technical Details**: 이 연구에서는 첫 단계로 검색자(retriever) 인코더를 훈련시켜 자연어와 JSON 객체 간의 정렬을 개선합니다. 특히, 고객마다 추가 가능한 수천 가지 단계와 데이터베이스 테이블 이름을 매핑하는 데 중점을 둡니다. 또한 소규모 시아미즈 변형자 인코더(Siamese Transformer Encoder)를 사용하여 사용자 쿼리와 단계나 테이블의 JSON 객체를 고정 길이 벡터로 인코딩하고, 이를 통해 LLM 프롬프트에 포함시켜 작업 흐름을 JSON 형식으로 생성합니다.

- **Performance Highlights**: 이 시스템은 기존의 LLM 기반 시스템이 주로 겪는 환각 문제를 절감함으로써 출력의 신뢰성을 높이고, 도메인 외 설정에서도 일반화 성능을 향상시킵니다. 검색 기능을 강화한 생성(RAG)을 적용함으로써, 더 작은 LLM을 사용하면서도 성능 손실 없이 시스템을 배치할 수 있습니다.



### Language Model Prompt Selection via Simulation Optimization (https://arxiv.org/abs/2404.08164)
- **What's New**: 이 연구에서는 생성 언어 모델(languge models)을 위한 프롬프트 선택 문제를 시뮬레이션 최적화(simulation optimization)의 문제로 재구성하고, 새로운 구조의 프레임워크를 제안하여 프롬프트 선택을 용이하게 합니다. 제안된 프레임워크는 실제 예시 프롬프트를 벡터로 변환하고, 이를 통해 충분하지만 유한한 수의 프롬프트로 구성된 실행 가능한 집합을 결정하는 검색 단계(search stage)와, 베이지안 파라메트릭 모델을 사용하여 평균 점수에 대한 대리 모델을 구축하는 평가 및 선택 단계(evaluation and selection stage)로 구성됩니다.

- **Technical Details**: 제안된 프레임워크는 텍스트 자동 인코더(text autoencoder)를 사용하여 초기 예제 프롬프트를 벡터로 변환하고, 주성분 분석(principal component analysis, PCA)을 적용하여 높은 차원의 벡터들을 중간 차원의 벡터로 축소시킵니다. 이렇게 생성된 벡터들을 '소프트 프롬프트(soft prompts)'라고 하며, 이들은 실행 가능한 집합을 구성합니다. 베이지안 파라메트릭 모델(Bayesian parametric model)을 사용하여 평균 점수에 대한 대리 모델을 구축하고, 이 모델을 기반으로 획득 함수(acquisition function)를 제안하여 소프트 프롬프트의 순차적 평가를 결정합니다.

- **Performance Highlights**: 수치 실험을 통해 제안된 프레임워크의 효율성을 입증하며, 프롬프트의 순차적 평가 절차의 일관성을 증명합니다. 따라서 이 연구는 생성 언어 모델을 사용하는 작은 기업이나 비영리 조직들이 비용 효율적으로 효과적인 프롬프트를 선택할 수 있는 방법을 제공하며, 이는 이러한 기관들의 한정된 자원을 효율적으로 사용할 수 있게 합니다.



### Extending Translate-Train for ColBERT-X to African Language CLIR (https://arxiv.org/abs/2404.08134)
Comments: 10 pages, 2 figures. System description paper for HLTCOE's participation in CIRAL@FIRE 2023

- **What's New**: 본 논문은 HLTCOE 팀이 2023년 FIRE에서 진행한 CIRAL CLIR 아프리카 언어 작업에 참여한 내용을 기술합니다. 특히, 기계 번역 모델을 사용하여 문서와 훈련 용구를 번역하고, ColBERT-X를 검색 모델로 사용한 점이 특징입니다. 아프리카 언어에서의 번역 모델의 어려움에도 불구하고 번역-훈련(Translate-Train) 기법을 적용하여 실험을 진행하였습니다.

- **Technical Details**: HLTCOE 팀은 Hausa, Somali, Swahili, Yoruba 문서를 검색하기 위해 영어 쿼리를 사용하는 CLIR 작업에 참여했습니다. 이들은 특히 ColBERT 모델의 변형인 ColBERT-X를 기반으로 번역-훈련 접근 방식을 사용하여 ML 학습을 수행했습니다. 또한, XLM-RoBERTa Large 모델에서 시작하여 언어 모델을 세밀하게 조정하고, JH POLO기법을 이용해 도메인 내 세밀 조정을 추가로 진행했습니다. 여기에는 대규모 언어 모델을 사용하여 새로운 영어 훈련 쿼리를 생성하는 과정이 포함됩니다.

- **Performance Highlights**: 번역-훈련 접근법을 사용한 ColBERT-X는 아프리카 언어 문서 검색에서 효과적이었습니다. 구체적으로, Yoruba어를 포함시키기 위해 Afriberta 코퍼스를 사용하여 추가적인 언어 모델 조정을 실시했습니다. 이는 MS MARCO 데이터를 이용하여 Hausa, Somali, Swahili, Yoruba 언어로 번역-훈련을 수행함으로써 대조적인 손실(contrastive loss)을 사용한 학습이 이루어졌음을 의미합니다. 또한, 공식 제출 이후에는 ColBERT v1 코드베이스를 기반으로 하는 ColBERT-X 구현을 사용하여 더 안정적이고 효과적인 훈련 과정을 보고하였습니다.



### S3Editor: A Sparse Semantic-Disentangled Self-Training Framework for  Face Video Editing (https://arxiv.org/abs/2404.08111)
- **What's New**: 새로운 얼굴 속성 편집 프레임워크인 S3Editor가 소개되었습니다. 이는 기존 방법의 한계를 극복하기 위해 고안된 Sparse Semantic-disentangled Self-training (희소 의미-분리 자가학습) 체계를 도입하여, 정체성 유지(identity preservation), 편집의 정확성(editing fidelity), 그리고 시간적 일관성(temporal consistency)을 개선하였습니다.

- **Technical Details**: S3Editor는 세 가지 주요 기술을 활용합니다. 첫째, 자가학습(self-training) 패러다임을 적용하여 반감독(semi-supervised) 학습을 통한 일반화 능력을 향상시켰습니다. 둘째, 동적 라우팅(dynamic routing) 메커니즘을 포함하는 의미론적 분리(semantic disentangled) 아키텍처를 제안하여 다양한 편집 요구 사항을 수용합니다. 셋째, 특정 주요 특성에만 초점을 맞춘 희소 최적화(sparse optimization) 전략을 구현하여 오버 에디팅(over-editing)을 방지합니다.

- **Performance Highlights**: S3Editor는 다양한 얼굴 비디오 편집 방법과 호환되며, GAN 기반의 Latent Transformer 및 DiffVAE와 같은 방법들을 포용합니다. 다양한 정성적 및 정량적 연구 결과를 통해, S3Editor는 개별 프레임의 정체성 유지 및 편집 충실도를 개선할 뿐만 아니라 시간적 일관성을 향상시키고 초과 편집을 방지하는 데에도 효과적임을 입증하였습니다.



### Variance-reduced Zeroth-Order Methods for Fine-Tuning Language Models (https://arxiv.org/abs/2404.08080)
Comments: 29 pages, 25 tables, 9 figures

- **What's New**: 이 연구에서는 언어 모델(LM)을 위한 메모리 효율적인 미세조정 방법 'Memory-Efficient Zeroth-Order Stochastic Variance-Reduced Gradient (MeZO-SVRG)'를 소개합니다. 기존의 Zeroth-Order 최적화 방식과는 달리, MeZO-SVRG는 분산 감소 기술을 통합해 안정성과 수렴성을 개선함으로써 특정 과제 프롬프트에 의존하지 않으면서도 높은 성능을 발휘할 수 있습니다.

- **Technical Details**: MeZO-SVRG는 기존의 MeZO 방식에서 한 단계 발전하여, 전체 배치 및 미니 배치 정보를 결합하여 저변동성, 무편향 경사 추정치를 생성합니다. 이 방법은 데이터 병렬 처리를 활용하여 처리 속도를 개선하고, 현장에서의 연산을 사용하여 메모리 사용량을 최소화합니다. 경사 추정치는 단일 교란 벡터를 사용해 계산되며, 이는 비동기적 요소의 병렬 처리를 가능하게 합니다.

- **Performance Highlights**: GLUE 및 SuperGLUE 벤치마크에서 평가된 MeZO-SVRG는 기존 MeZO에 비해 최대 20%의 테스트 정확도 향상을 보였으며, GPU 사용 시간도 절반으로 줄일 수 있었습니다. 또한, 첫 번째 차수의 SGD 방식에 비해 필요한 메모리 사용량을 최소 2배 이상 줄이면서, 배치 크기가 클수록 메모리 절감 효과가 더욱 개선되는 경향을 보였습니다.



### Augmenting Knowledge Graph Hierarchies Using Neural Transformers (https://arxiv.org/abs/2404.08020)
Comments: European Conference on Information Retrieval 2024

- **What's New**: 이 연구는 기존 지식 그래프(Knowledge Graph, KG)에 계층 구조를 생성하고 증강할 수 있는 새로운 접근 방식을 제시합니다. 이 방법은 Adobe의 사용자 데이터를 분석하고 창의적 목적을 이해하여 Adobe 자산을 추천하는 데 사용됩니다. 연구는 의도(intent)와 색상(color) 노드 유형에 대해 계층을 자동으로 생성하는 데 초점을 맞추었으며, 계층의 적용 확장을 구현하여 의도의 98%와 색상의 99%에 대한 커버리지 증가를 가능하게 했습니다.

- **Technical Details**: 이 연구는 대규모 변형기(neural transformers)를 활용하여 지식 그래프 내에서 복잡한 그래프 계층을 자동으로 생성합니다. 특히, 대규모 언어 모델(Large Language Models, LLM)을 이용하여 일부 주요 범주를 생성하고, 이를 통해 더 세부적인 계층으로 상세화하는 과정을 사용했습니다. 계층화는 두 단계로 진행되며, 분류 모듈(Classifier Module)과 생성 모듈(Generator Module)을 사용합니다. 'few-shot classification'과 'one-shot generation' 기법으로 적은 데이터로도 효과적인 노드 분류와 계층 생성이 가능함을 보여주었습니다.

- **Performance Highlights**: 이 방식은 의도와 색상 노드에 대한 계층 구조의 커버리지를 각각 98%와 99%로 크게 향상시켰습니다. 이는 이전에는 주로 평평했던 분류 체계를 확장하여 의미론적 중요성을 증강시키고 사용자의 검색과 추천의 정확성을 높입니다. 'few-shot learning'에서 12%의 정확도 향상을 보여, 대규모 그래프에서도 효율적인 계층 생성을 가능하게 하는 방법론을 제시합니다.



### Analyzing the Performance of Large Language Models on Code Summarization (https://arxiv.org/abs/2404.08018)
- **What's New**: 이 논문은 대규모 언어 모델(Large Language Models, LLMs)의 코드 요약(code summarization) 성능에 대한 분석을 통해 모델이 실제로 코드의 구조나 의미(semantics)를 얼마나 이해하고 있는지, 아니면 단순히 토큰의 유사성에 의존하고 있는지를 조사합니다. 특히 Llama 2와 같은 최신 모델의 성능을 다루며, 코드와 자연어 설명 사이의 토큰 겹침(subword token overlap)이 성능에 미치는 영향을 살펴봅니다.

- **Technical Details**: 연구팀은 코드와 해당 자연어 설명 사이의 토큰 겹침 정도에 따라 데이터셋의 예시를 여러 그룹으로 나누고, 그룹별 성능 차이를 관찰했습니다. 또한, 함수 이름(function names)을 변경하거나 코드 구조(code structure)를 제거하는 등의 변형을 적용하여 모델의 성능에 미치는 영향을 분석했습니다. 이러한 실험을 통해 모델이 코드의 구조적 및 논리적 측면보다는 토큰의 유사성에 더 많이 의존하고 있음을 발견했습니다. 사용된 평가 지표로는 BLEU와 BERTScore가 있으나, 이들 지표가 서로 높은 상관관계를 보여 추가적인 인사이트를 제공하지는 못했습니다.

- **Performance Highlights**: LLMs의 성능은 함수 이름이나 식별자 이름과 같이 코드의 설명과 높은 토큰 겹침을 가지는 부분에서 더 높게 나타났습니다. 이러한 토큰 겹침은 자연어로 된 요약이 입력 코드와 매우 유사한 문자열을 갖는 것에서 비롯된 것으로, 모델이 실제로 코드의 의미를 이해하기보다는 단순히 문자열의 유사성을 바탕으로 작동하는 경우가 많은 것으로 분석되었습니다.



### Sample-Efficient Human Evaluation of Large Language Models via Maximum  Discrepancy Competition (https://arxiv.org/abs/2404.08008)
Comments: 32 pages, 6 figures

- **What's New**: 이 연구에서는 대규모 언어 모델(LLMs)의 평가를 위해 새로운 효율적인 방법인 MAximum Discrepancy (MAD) 경쟁 방식을 제안합니다. 이 방법은 제한된 수의 샘플을 선택하여 LLMs 간의 성능을 비교하고 글로벌 랭킹을 생성하는 데에 Elo 등급 시스템을 사용합니다.

- **Technical Details**: MAD 경쟁은 다양하고 정보가 풍부한 지시문을 자동으로 선택합니다. 이 지시문은 두 LLM에 적용되며, 사용자는 세 가지 선택지 중 하나를 강제로 선택하여 응답을 평가합니다. 그 결과는 Elo 등급 시스템을 사용하여 글로벌 랭킹으로 집계됩니다. 연구자는 지식 이해, 수학적 추론, 글쓰기, 코딩의 네 가지 기술에 대해 8개의 대표적인 LLM을 선택하여 비교했습니다.

- **Performance Highlights**: 제안된 방법은 LLMs의 능력에 대해 신뢰할 수 있고 합리적인 랭킹을 제공합니다. 이는 각 모델의 강점과 약점을 식별하고 LLM의 진보를 위한 중요한 통찰력을 제공합니다.



### Xiwu: A Basis Flexible and Learnable LLM for High Energy Physics (https://arxiv.org/abs/2404.08001)
Comments: 15 pages, 8 figures

- **What's New**: 이 연구에서는 최신 대형 언어 모델 (Large Language Models, LLMs)을 고에너지 물리학 (High-Energy Physics, HEP) 분야에 적용하기 위한 'Xiwu'라는 새로운 시스템을 개발하였습니다. Xiwu 시스템은 최신의 기초 모델들 사이를 쉽게 전환할 수 있으며, 도메인 지식을 신속하게 교육할 수 있는 기능을 제공합니다.

- **Technical Details**: Xiwu 시스템은 'seed fission technology'를 도입하고, AI-Ready 데이터셋을 빠르게 수집 및 정제하기 위한 도구들을 개발했습니다. 또한, 'vector store technology'를 기반으로 하는 즉시 학습 시스템과, 지정된 기초 모델 하에서 빠르게 훈련할 수 있는 'on-the-fly fine-tuning system'을 개발하여 구현하였습니다.

- **Performance Highlights**: Xiwu 모델은 LLaMA, Vicuna, ChatGLM, Grok-1 등의 기초 모델들 간의 전환이 원활하게 이루어졌으며, HEP 지식 질의응답 및 코드 생성 분야에서 벤치마크 모델을 크게 능가하는 성능을 보였습니다. 이러한 전략은 모델 성능의 성장 가능성을 크게 향상시켜, GPT-4의 개발에 따라 추후 성능이 개선될 것으로 기대됩니다.



### A Multi-Level Framework for Accelerating Training Transformer Models (https://arxiv.org/abs/2404.07999)
Comments: ICLR 2024

- **What's New**: 이 논문은 대형 딥러닝 모델의 학습 시간과 비용을 획기적으로 줄이기 위한 새로운 다중 레벨 트레이닝 프레임워크를 제안합니다. BERT, GPT, ViT 같은 모델들의 훈련 과정에서 발견된 특징 맵(Feature Maps)과 어텐션(Attentions) 구조의 유사성을 이용해, 모델 크기를 점진적으로 축소 및 확대하며 효율적으로 학습하는 V-사이클 트레이닝 프로세스를 소개합니다.

- **Technical Details**: 본 프레임워크는 Coalescing, De-coalescing, Interpolation 세 가지 기본 연산자를 사용합니다. 이 연산자들을 조합하여 다중 레벨에서 모델의 크기를 점진적으로 조절하고, 인접 레벨의 모델 간에 파라미터를 효과적으로 전달할 수 있습니다. 특히, 작은 모델은 빠른 수렴을 보여주며, 이를 바탕으로 큰 모델로의 파라미터 전환 시 고품질의 중간 솔루션을 제공합니다. Interpolation 연산자는 De-coalescing에 의해 발생할 수 있는 뉴런의 대칭성을 깨뜨려 수렴 성능을 향상시킵니다.

- **Performance Highlights**: 이 프레임워크를 통해 BERT와 GPT 기반의 다양한 딥러닝 모델 학습에 있어 전통적인 학습 방법 대비 약 20%에서 최대 51.6%까지 계산 비용을 절감할 수 있으며, DeiT 모델에 대해서는 27.1%의 트레이닝 비용 절감 효과를 보였습니다. 이는 기존의 트레이닝 방식에 비해 상당한 속도 향상을 의미하며, 성능 역시 유지되었습니다.



### Rethinking How to Evaluate Language Model Jailbreak (https://arxiv.org/abs/2404.06407)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)의 안전성 확보를 위해 새로운 평가 척도를 제안합니다. 기존의 간단한 이진 평가 방식(성공 여부)을 넘어서, 새로운 3가지 메트릭(안전 침해, 정보성, 상대적 진실성)을 도입하고 이를 다면적 접근법으로 평가하는 방법을 소개하고 있습니다.

- **Technical Details**: 저자들은 언어 모델의 'jailbreak' 시도를 평가하기 위해 'safeguard violation(안전 침해)', 'informativeness(정보성)', 및 'relative truthfulness(상대적 진실성)'이라는 세 가지 새로운 평가 메트릭을 제안합니다. 이 메트릭들은 다양한 악의적 행위자들의 목표와 상관관계를 가지며, 자연어 생성 평가 방법을 확장하여 다면적 접근법을 통해 계산됩니다.

- **Performance Highlights**: 이 다면적 평가 방법은 벤치마크 데이터셋을 사용하여 기존 방법들과 비교되었으며, 평균적으로 기존 기준 대비 17% 향상된 F1 스코어를 보여줍니다. 이는 새롭게 제안된 평가 척도가 언어 모델의 안전성을 보다 정확하고 효과적으로 평가할 수 있음을 시사합니다.




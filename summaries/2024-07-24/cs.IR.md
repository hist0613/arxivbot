New uploads on arXiv(cs.CL)

### Stress-Testing Long-Context Language Models with Lifelong ICL and Task Haystack (https://arxiv.org/abs/2407.16695)
Comments:
          Code: this https URL Website: this https URL

- **What's New**: Lifelong ICL이라는 문제 설정을 소개합니다. 이 설정은 장문맥 언어 모델(long-context language models, LMs)이 상황 학습(in-context learning, ICL)을 통해 일련의 언어 과제를 학습하도록 도전합니다. 또한, 이러한 모델들이 Lifelong ICL에서 문맥을 어떻게 활용하는지를 평가하고 진단하기 위해 'Task Haystack'이라는 평가 모음도 도입했습니다.

- **Technical Details**: Task Haystack은 기존의 'needle-in-a-haystack' (NIAH) 테스트에서 영감을 받아 새롭고 독특한 도전을 제시합니다. 이 평가 모음은 (1) 단순 복사-붙여넣기를 넘어서 문맥을 깊이 있게 이해해야 하며, (2) 진화하는 주제와 작업 스트림을 통해 실제 사용의 복잡성을 반영합니다. 모델은 Lifelong ICL 프롬프트와 단일 작업 ICL 프롬프트를 사용해 다양한 작업을 수행하게 되며, 두 가지 프롬프트의 테스트 정확도 차이가 크지 않을 경우 '통과'로 간주됩니다.

- **Performance Highlights**: 12개의 장문맥 언어 모델을 Task Haystack을 통해 평가한 결과, 최첨단 모델인 GPT-4o도 15%의 경우에 실패하며, 모든 공개 가중치 모델은 최대 61%의 실패율을 기록했습니다. 주의산만(distractability)과 최신성 편향(recency bias) 같은 요소가 주요 실패 원인으로 확인되었습니다. 또한, 작업 지시가 테스트 시점에서 바뀌거나 ICL 데모가 과도하게 반복되면 성능이 하락하는 현상이 관찰되었습니다.



### Explanation Regularisation through the Lens of Attributions (https://arxiv.org/abs/2407.16693)
Comments:
          18 pages, 7 figures, 8 tables

- **What's New**: 이번 연구는 설명 정규화(ER; Explanation Regularisation)가 모델이 타당한 토큰에 더 의존하도록 장려함으로써 예측을 사람과 비슷하게 만드는 데 도움을 준다고 주장하지만, 이는 기존 연구에서 충분히 검증되지 않았음을 강조합니다.

- **Technical Details**: ER은 인간의 이유(rationale)와 일치하도록 모델의 입력 속성 기법 출력과 차이를 줄이는 보조 설명 손실(auxiliary explanation loss)을 도입합니다. 본 연구에서는 ER이 클래스 분류 결정에 얼마나 타당한 토큰에 의존하는지와 타당성과 도메인 외 데이터(out-of-domain; OOD) 조건에 대한 견고성 사이의 관계를 분석했습니다.

- **Performance Highlights**: ER는 특정 기술을 사용하여 인간의 이유와 잘 일치하는 모델을 찾을 수 있지만, 다른 후처리(post-hoc) 속성 기법에서는 그 효과가 거의 또는 전혀 나타나지 않았습니다. 이는 ER이 모델의 분류 결정 과정에 실질적으로 영향을 미치지 않는다는 것을 시사합니다. 또한, 낮은 설명 손실을 달성하도록 모델을 강하게 제약할 때만 타당한 특징에 대한 의존성이 증가하고, 이는 OOD 개선 대가를 치르게 됩니다.



### Can Large Language Models Automatically Jailbreak GPT-4V? (https://arxiv.org/abs/2407.16686)
Comments:
          TrustNLP@NAACL2024 (Fourth Workshop on Trustworthy Natural Language Processing)

- **What's New**: 이번 연구에서는 GPT-4V의 안면 인식 기능에서 발생할 수 있는 프라이버시 문제를 해결하기 위해 AutoJailbreak라는 자동 탈옥 기술을 제안합니다. AutoJailbreak는 LLMs(Large Language Models)을 활용하여 프롬프트 최적화를 통해 탈옥을 수행하며, 약한-강한 In-Context 학습 프롬프트를 사용하여 효율성을 높입니다.

- **Technical Details**: AutoJailbreak는 세 가지 단계를 거쳐 작동합니다. 먼저, 인간 이미지와 해당 이름으로 구성된 데이터셋을 준비합니다. 그런 다음, LLM을 활용해 red-team 모델에서 탈옥 프롬프트를 생성하고, 약한-강한 In-Context 학습 프롬프트를 통합하여 탈옥 성공률을 높입니다. 마지막으로, 이 과정을 통해 GPT-4V가 이미지에서 사람을 식별할 수 있게 합니다. 이러한 방식으로 모델 가중치 접근이 필요 없는 블랙박스(black-box) 조건에서도 95.3%의 높은 공격 성공률을 달성할 수 있었습니다.

- **Performance Highlights**: AutoJailbreak는 기존 수동 프롬프트 작성 방법보다 공격 성공률이 높고, 시간과 토큰 소모를 최소화하는 조기 중단(Early Stopping) 기법을 사용하여 효율성을 극대화하였습니다. 구체적으로, 유명 인사 이미지를 대상으로 한 실험에서 GPT-4V의 방어를 성공적으로 뚫었습니다.



### Towards scalable efficient on-device ASR with transfer learning (https://arxiv.org/abs/2407.16664)
- **What's New**: 다국어 사전 학습(multilingual pretraining)을 통해 저자원(低資源) 단일언어 자동 음성 인식(ASR) 모델의 견고성을 크게 향상시킬 수 있다는 점을 규명한 연구입니다. 연구는 전이 학습(transfer learning)의 초기 학습과 정교 튜닝(fine-tuning) 과정에서의 성능, 데이터셋 도메인과 언어 간 전이 학습의 영향, 그리고 희귀 단어와 일반 단어 인식에 대한 효과를 체계적으로 조사했습니다.

- **Technical Details**: 연구는 RNNT 손실(RNNT-loss)을 사용한 사전 학습 후, 최소 단어 오류율(MinWER) 손실을 사용한 단일 언어 정교 튜닝을 통해, 이탈리아어와 프랑스어와 같은 언어 전반에서 단어 오류율(WER)을 일관되게 감소시켰습니다. 도메인 간 사전 학습은 도메인 내 사전 학습보다 28% 더 높은 WER 감수율(WERR)을 보였습니다. 희귀 단어와 비희귀 단어 모두 혜택을 받았으나, 희귀 단어는 도메인 간 사전 학습에서, 비희귀 단어는 도메인 내 사전 학습에서 더 큰 개선을 보였습니다.

- **Performance Highlights**: 이탈리아어와 프랑스어와 같은 여러 언어에서 단일언어 기준 모델에 비해 MLS와 자체 데이터셋에서 단어 오류율(WER)이 각각 최대 36.2%와 42.8% 감소했습니다.



### Course-Correction: Safety Alignment Using Synthetic Preferences (https://arxiv.org/abs/2407.16637)
Comments:
          Dataset and script will be available at this https URL

- **What's New**: 이 논문은 대형 언어 모델(LLMs, Large Language Models)의 자동적으로 유해한 콘텐츠 생성을 피할 수 있는 능력을 평가하고 개선하기 위한 체계적인 연구를 제시합니다. 저자들은 	extsc{C$^2$-Eval}이라는 벤치마크를 도입하여 10개의 인기 있는 LLMs를 분석했으며, 이 모델들의 course-correction(코스 수정) 능력에 상당한 차이가 있음을 발견했습니다. 또한, LLMs의 코스 수정 능력을 향상시키기 위해 preference learning(선호 학습)을 사용하는 방법을 제안했습니다.

- **Technical Details**: 연구진은 데이터 기반 선호 학습을 통해 코스 수정을 가르치는 	extsc{C$^2$-Syn}이라는 합성 데이터셋을 만들었습니다. 이 데이터셋은 코스 수정을 강조하는 750K 쌍의 선호 데이터를 포함하고 있습니다. 실험은 Llama2-Chat 7B와 Qwen2 7B 모델에서 수행되었으며, 합성 데이터를 사용하여 모델을 fine-tuning(미세 조정)하는 방법을 적용하였습니다.

- **Performance Highlights**: 제안된 방법을 통한 실험 결과, Llama2-Chat 7B와 Qwen2 7B 모델의 코스 수정 능력과 보안성이 크게 향상되었으며, 일반 성능에는 영향을 미치지 않았습니다. 두 모델 모두 4가지 대표적인 jailbreak(탈옥) 공격에 대한 저항력이 강화되었습니다.



### Semantic Change Characterization with LLMs using Rhetorics (https://arxiv.org/abs/2407.16624)
- **What's New**: 이 논문은 언어의 진화와 의미 변화(semantic change)를 LLMs (Large Language Models)를 사용해 분석하고자 합니다. LLMs가 의미 추론(sense inference) 및 논리적 추론(reasoning)에서 큰 진전을 보임에 따라, 이들을 통해 세 가지 종류의 의미 변화를 특성화하는 방법을 제안합니다: 차원(dimension), 관계(relation), 방향(orientation). 이를 위해 LLMs의 'Chain-of-Thought' 접근법과 수사적 장치를 결합하여 실험적 평가를 수행했고, 새로운 데이터셋을 사용하여 접근법의 효과를 입증했습니다.

- **Technical Details**: 이 연구는 기존 연구에서 많이 다루지 않은 의미 변화의 다양한 유형을 다루고자 합니다. Traugott(2017)이 제안한 의미 변화의 유형은 크게 확장(broadening), 축소(narrowing), 개선(amelioration), 악화(pejoration), 메타포화(metaphorization), 환유화(metonymization)로 분류됩니다. 우리는 LLMs 'Chain-of-Thought' 기술을 사용하여, 인간의 인지적 추론(cognitive reasoning) 프로세스를 모방하는 방식으로 이러한 의미 변화를 특성화하는 방법론을 개발했습니다. 수집된 새로운 데이터셋은 차원(dimension), 방향(orientation), 관계(relation) 세 가지 범주에서 의미 변화를 평가하는 데 사용되었습니다.

- **Performance Highlights**: 제안된 접근법은 기존의 의미 변화 탐지 연구와는 달리, 모든 유형의 의미 변화를 통합적으로 분석할 수 있다는 점에서 차별화됩니다. 실험 결과, LLMs는 의미 변화를 효과적으로 캡처하고 분석할 수 있으며, 이는 자동 번역(automatic translation) 및 챗봇(chatbots)과 같은 컴퓨터 언어 애플리케이션의 성능 향상에 기여할 수 있음을 보여줍니다.



### Lawma: The Power of Specialization for Legal Tasks (https://arxiv.org/abs/2407.16615)
- **What's New**: 법률 텍스트 주석(annotation) 및 분류(classification)는 실증적 법률 연구의 핵심 요소입니다. 기존에는 이 작업들이 주로 훈련된 연구 조교들에게 위임되었으나, 언어 모델의 발전으로 인해 법률 학자들은 상업용 모델에 대한 프롬팅을 통해 인간 주석 비용을 줄이기를 희망하고 있습니다. 본 연구는 260개의 법률 텍스트 분류 작업을 대상으로 GPT-4를 벤치마크 모델로 사용해 종합적인 연구를 수행하였으며, 경미한 파인튜닝(fine-tuning)을 거친 Llama 3 모델이 거의 모든 작업에서 GPT-4를 크게 상회하는 성능을 보인다는 것을 입증했습니다.

- **Technical Details**: 작업은 미국 대법원(Supreme Court)과 항소법원(Court of Appeals) 데이터베이스에서 레이블을 가져와 수행합니다. 이 작업은 다중 클래스(machine learning tasks)에 기반하며, 특히 Llama 3 8B Inst 모델을 대상으로 하여 GPT-4 제로샷(zero-shot) 성능 대비 월등한 성과를 보였습니다. 또한, Llama 3 70B Inst, Mistral 7B Inst 등 다양한 오픈 소스 모델들을 사용하여 성능을 비교 평가했습니다.

- **Performance Highlights**: 제로샷 평가에서 GPT-4는 평균 정확도 62.9%를 기록한 반면, Llama 3 70B는 58.4%의 정확도를 보였습니다. 특히, 파인튜닝을 통해 Llama 3 8B Instruct 모델 ('Lawma 8B')는 대법원 태스크에서 82.4%, 항소법원 태스크에서는 79.9%의 정확도를 기록했습니다. 파인튜닝된 모델은 수백 개의 예제로도 높은 분류 정확도를 달성할 수 있으며, 260개의 작업을 동시에 파인튜닝하는 경우에도 특정 작업에 대한 파인튜닝 대비 성능 저하가 크지 않습니다.



### Data Mixture Inference: What do BPE Tokenizers Reveal about their Training Data? (https://arxiv.org/abs/2407.16607)
Comments:
          19 pages, 5 figures

- **What's New**: 이 논문에서는 데이터 혼합 유추(data mixture inference)라는 작업을 소개합니다. 이 작업은 학습 데이터의 구성 비율을 유추하는 것으로, 특히 모던 언어 모델들이 사용하는 바이트-페어 인코딩(BPE) 토크나이저의 병합 규칙을 분석해 학습 데이터의 토큰 빈도를 밝혀내는 새로운 공격 기법을 제안합니다.

- **Technical Details**: 제안된 방법은 BPE 토크나이저가 학습하는 병합 규칙 리스트를 활용해 학습 데이터의 토큰 빈도를 추론합니다. 이 규칙 리스트의 순서가 데이터 내 토큰 빈도에 관한 정보를 자연스럽게 드러내기 때문입니다. 이를 위해 선형 프로그램(linear program)을 구축하고 각 범주의 비율을 계산합니다. 이 방법은 자연어, 프로그래밍 언어, 데이터 출처 등 다양한 데이터 혼합을 포함합니다.

- **Performance Highlights**: 제안된 공격 기법은 통제된 실험에서 높은 정밀도로 혼합 비율을 복원했습니다. GPT-4o의 토크나이저는 전체 데이터의 39%가 영어가 아닌 데이터로 학습되었으며, Llama3는 GPT-3.5와 비교하여 다언어 사용에 더 중점을 두어 48%의 비영어 데이터를 포함합니다. 추가로 Llama와 Gemma는 라틴어나 키릴 문자 기반의 언어로 크게 치우쳐 있음을 발견했습니다.



### Shared Imagination: LLMs Hallucinate Alik (https://arxiv.org/abs/2407.16604)
- **What's New**: 최근 대형 언어 모델(LLMs)의 훈련 방식을 분석한 논문이 발표되었습니다. 이 논문에서는 상상 질문 응답(Imaginary Question Answering, IQA)이라는 새로운 실험 세팅을 통해 모델 간의 유사성을 연구했습니다. 논문에 따르면, 한 모델이 완전히 허구적인 개념에 대해 질문을 생성하고, 다른 모델이 이에 대한 답을 하도록 했을 때, 모든 모델이 상당히 성공적으로 서로의 질문에 답했습니다. 이는 상상 공간(Shared Imagination Space)이 모델 간에 공유되고 있음을 시사합니다.

- **Technical Details**: 작업을 수행하기 위해, 질문 모델(QM)은 다중 선택 질문(Multiple-Choice Questions)을 생성하고, 응답 모델(AM)은 해당 질문에 답을 합니다. 여기에는 직접 질문(Direct Question, DQ)과 문맥 기반 질문(Context Question, CQ)의 두 가지 유형이 포함됩니다. 13개의 언어 모델이 17개의 일반 대학 과목에 대해 질문을 생성하고 답변을 제공했습니다. 모델들이 사용하는 최적화 알고리즘은 확률적 경사 하강법(Stochastic Gradient Descent, SGD)이며, 훈련 데이터는 책, 인터넷 텍스트, 코드 등으로 구성됩니다.

- **Performance Highlights**: 실험 결과, 직접 질문에 대해 평균 54%의 정확도를 기록하였으며, 문맥 기반 질문에 대해서는 86%로 정확도가 크게 향상되었습니다. 이는 모델들이 허구적인 내용에 대해서도 높은 수준의 일관성을 보임을 의미합니다. 또한, 특정 (QM, AM) 쌍의 경우, 정확도가 최대 96%에 달했습니다. 이는 모델들이 각자 생성한 상상 속 내용에 대해 공통된 이해를 가지고 있음을 보여줍니다. 이 현상은 언어 모델의 동질성과 허구 생성 및 탐지에 대한 중요한 통찰을 제공합니다.



### A Comparative Study on Patient Language across Therapeutic Domains for Effective Patient Voice Classification in Online Health Discussions (https://arxiv.org/abs/2407.16593)
Comments:
          14 pages, 4 figures, 5 tables, funded by Talking Medicines Limited

- **What's New**: 이번 연구는 환자가 실제 임상 경험을 의료 전문가들과 공유하는 데 있어 존재하는 보이지 않는 장벽을 강조합니다. 보다 솔직하게 건강 관련 경험을 공유하는 환자들의 소셜 미디어 활동을 분석하여 귀중한 정보를 추출하고자 했습니다. 이를 위해 '환자 목소리 분류(Patient Voice Classification)'라는 작업을 정의하고 이에 대한 효율적인 분류 방법을 제안합니다.

- **Technical Details**: 연구진은 소셜 미디어 플랫폼에서 환자 목소리를 식별하기 위해 언어 특성의 중요성을 분석했습니다. 주로 텍스트 유사성 분석과 언어적 특성을 통해 공통 패턴을 식별하고, 이를 기반으로 사전 학습된 언어 모델을 미세 조정하여 자동 맞춤 환자 목소리 분류를 수행했습니다. 데이터는 Reddit와 SocialGist에서 수집되었고, Doccano 도구를 사용해 수작업으로 주석을 달았습니다.

- **Performance Highlights**: 융합된 데이터셋에서 유사한 언어 패턴을 가진 데이터로 미세 조정된 사전 학습된 언어 모델을 통해 높은 정확도의 환자 목소리 자동 분류를 달성했습니다. 특히 심혈관계, 종양학, 면역학, 신경과학 등 다양한 치료 영역과 데이터 소스에 따라 환자들이 표현하는 방식의 차이를 깊이 이해하는 데 초점을 맞췄습니다.



### TLCR: Token-Level Continuous Reward for Fine-grained Reinforcement Learning from Human Feedback (https://arxiv.org/abs/2407.16574)
Comments:
          ACL2024 Findings

- **What's New**: 이 연구는 기존의 시퀀스 레벨(sequence-level) 보상이 아닌, 토큰 레벨(token-level) 연속 보상(Token-Level Continuous Reward, TLCR)을 도입하여 인공지능을 보다 세밀하게 강화 학습시키는 방법을 제안합니다. 특히, 토큰 단위의 긍정적, 부정적 선호도를 구분할 수 있는 판별기를 활용하여, 맥락을 고려한 연속 보상을 각 토큰에 할당합니다.

- **Technical Details**: TLCR 모델은 외부의 성숙한 언어 모델(external mature language model)인 GPT-4를 이용해 기존 텍스트를 수정하고, 원본과 수정된 텍스트 간의 최소 편집 거리(minimal edit distance)를 계산하여 토큰별 선호도 레이블을 생성합니다. 이를 통해 판별기를 훈련하여, 생성된 텍스트에 대해 보다 정교하고 밀도 있는 보상을 제공합니다. 이 과정에서 고정된 보상 값을 사용하는 대신, 판별기의 예측 신뢰도(confidence)에 기반한 연속 보상 값을 할당합니다.

- **Performance Highlights**: 광범위한 실험을 통해 TLCR이 기존의 시퀀스 레벨 및 토큰 레벨 이산 보상(discrete rewards)보다 우수한 성능을 보이는 것을 확인했습니다. 특히, 자유로운 생성 벤치마크(open-ended generation benchmarks)에서의 성능 개선이 두드러졌습니다.



### Retrieve, Generate, Evaluate: A Case Study for Medical Paraphrases Generation with Small Language Models (https://arxiv.org/abs/2407.16565)
Comments:
          KnowledgeableLM 2024

- **What's New**: 최근 대형 언어 모델(LLMs)의 접근성이 높아지면서 의료 관련 추천에서 정확하지 않은 정보가 퍼질 위험이 증가하고 있습니다. 이를 해결하기 위해 소형 언어 모델(SLM)을 활용한 'pRAGe'라는 새로운 파이프라인을 소개합니다. 이는 Retrieval Augmented Generation (RAG) 방식을 활용해 의료 용어를 더 쉽게 이해할 수 있도록 프랑스어로 패러프레이즈를 생성합니다.

- **Technical Details**: pRAGe는 Retrieval Augmented Generation (RAG) 아키텍처를 기반으로 외부 지식 베이스(Knowledge Base)를 사용하여 언어 모델의 환각(hallucinations)을 줄입니다. 이 방법은 고가의 GPU를 사용하지 않고도 오픈소스 소형 언어 모델들을 활용하여 의료 관련 질문-답변(Q&A)과 패러프레이즈 생성을 수행합니다. 또한 RefoMed-KB라는 프랑스어 의료 지식베이스를 사용하여 훈련된 모델을 제공합니다.

- **Performance Highlights**: 이 연구는 소형 언어 모델(1B ~ 7B 매개변수)의 성능을 평가하며, 이를 통해 프롬팅(prompting)과 파인튜닝(finetuning)의 효과를 분석합니다. pRAGe-FT라는 파인튜닝된 RAG 모델도 포함되어 있으며, 이 모델은 제로샷(Q&A) 태스크에서 높은 성능을 보였습니다. 모든 코드와 데이터셋, 평가 지표도 공개되어 있어 재현 가능한 연구를 지원합니다.



### Quantifying the Role of Textual Predictability in Automatic Speech Recognition (https://arxiv.org/abs/2407.16537)
- **What's New**: 이 연구는 새로운 접근법을 통해 자동 음성 인식(ASR) 모델의 오류를 평가하려는 목적으로 수행되었습니다. 저자들은 상대적인 텍스트 예측 가능성을 기능으로 하는 오류율을 모델링하여, 'k'라는 단일 숫자를 사용하여 인식기의 텍스트 예측 가능성에 대한 영향을 평가하였습니다. 이 방법은 Wav2Vec 2.0 기반 모델이 하이브리드 ASR 모델보다 텍스트 문맥을 더 잘 활용한다는 것을 입증했으며, 최근 표준 ASR 시스템이 아프리카계 미국 영어에서 성능이 저조한 이유를 밝히는 데도 사용되었습니다.

- **Technical Details**: 이 연구에서는 Boothroyd와 Nittrouer의 정신음향학적 패러다임을 확장하여 ASR 성능을 텍스트 예측 가능성으로 수량화하는 새로운 방법을 개발하였습니다. 다양한 예측 가능성을 가진 발화에 대해 'k' 값을 측정하여, 더 강력한 언어 모델이 높은 'k' 값을 생성함을 증명하였습니다. 또한, 여러 ASR 모델(GMM, TDNN, Wav2Vec 2.0-base, Wav2Vec 2.0-large)을 비교하여 더 강력한 모델일수록 'k' 값이 증가함을 밝혔습니다. 이 방법은 아프리카계 미국 영어를 포함한 다양한 언어 코퍼스에 적용되어 ASR 시스템의 문제를 진단하고 개선하는 데 사용될 수 있습니다.

- **Performance Highlights**: 연구 결과, Wav2Vec 2.0 기반 모델이 명시적 언어 모델을 사용하지 않음에도 불구하고 하이브리드 ASR 모델보다 텍스트 문맥을 더 잘 활용하는 것으로 나타났습니다. 또한, 기존 연구와 마찬가지로 아프리카계 미국 영어에서의 성능 저조는 주로 음향-음성 모델링의 실패를 반영한다는 결론에 도달하였습니다. 이러한 접근법은 ASR 성능의 예측 가능성을 구체적으로 진단하고 향상시키는 데 유용합니다.



### AMONGAGENTS: Evaluating Large Language Models in the Interactive Text-Based Social Deduction Gam (https://arxiv.org/abs/2407.16521)
Comments:
          Wordplay @ ACL 2024

- **What's New**: 이번 논문에서는 사람 행동의 대리인을 시뮬레이션 환경에서 생성하고 분석할 수 있는 AmongAgent라는 텍스트 기반 게임 환경을 소개합니다. 이 게임은 인기 있는 게임 'Among Us'의 dynamics를 반영하며, LLMs (Large Language Models)가 이 환경에서의 사회적 행동을 이해하고 추론하는 능력을 평가합니다. 이 연구는 LLMs가 목표 지향적 게임에서 복잡한 액션 공간과 불완전한 정보 속에서 어떻게 성능을 발휘할 수 있는지를 탐구하는 것을 목표로 합니다.

- **Technical Details**: AmongAgent에서는 플레이어가 우주선의 승무원으로서 임무를 수행하며 사보타주를 감행하는 임포스터를 찾고 제거해야 합니다. 플레이어는 다양한 역할과 성격을 부여받아 게임 내에서 행동하며, LLMs는 이러한 환경에서 규칙을 이해하고 현재 상황에 맞는 결정을 내릴 수 있습니다. 이 환경은 텍스트 기반으로 되어 있어 LLMs의 능력을 지속적이고 동적으로 평가할 수 있습니다. 텍스트 기반 프롬프트 엔지니어링을 활용하여 에이전트가 일관된 행동을 유지하고 의미 있는 반성을 할 수 있도록 지원합니다.

- **Performance Highlights**: LLMs는 게임 규칙을 잘 이해하고 상황에 맞춰 의사 결정을 내려야 합니다. 그러나 속임수 전략에서는 개선이 필요합니다. 에이전트의 성과는 부여된 성격에 따라 다르게 나타났습니다. 이 연구는 AmongAgent 환경, 에이전트 프레임워크 및 평가를 공개하여 미래 연구를 장려합니다. LLM플레이어는 게임의 메커니즘을 이해하고 규칙을 따르는 강력한 능력을 보였으며, 부여된 다른 성격에 따라 다양한 성과를 보였습니다.



### Assessing In-context Learning and Fine-tuning for Topic Classification of German Web Data (https://arxiv.org/abs/2407.16516)
- **What's New**: 연구자들이 독일의 세 가지 정책과 관련된 웹페이지 콘텐츠를 이진 분류 작업으로 모델링하고, 사전 학습된 인코더 모델(fine-tuned pre-trained encoder models)과 컨텍스트 내 학습 전략(in-context learning strategies)의 정확성을 비교했습니다. 다국어 모델(multilingual models)과 단일 언어 모델(monolingual models), 제로 및 퓨샷 접근법(zero and few-shot approaches), 부정 샘플링 전략(negative sampling strategies), URL 및 콘텐츠 기반 기능(URL & content-based features) 결합의 영향을 조사했습니다.

- **Technical Details**: 대규모 언어 모델(large language models, LLMs)을 사용해 스크랩된 웹페이지 코퍼스에서 이진 주제 분류(binary topic classification) 작업을 수행했습니다. 연구에서는 독일의 세 가지 특정 정책과 관련된 웹페이지를 식별했습니다: (1) 아동 빈곤 퇴치 정책, (2) 재생 에너지 촉진, (3) 대마초 법 개정. 멀티링궐 모델(XLM-RoBERTa)과 단일 언어 모델(GBERT)을 사용해 수동으로 라벨링된 데이터에 대해 미세 조정(fine-tuning)했습니다. 생성 모델(generative models)인 Llama와 Mistral을 사용하여 퓨샷 프롬프트(few-shot prompting)를 평가하고, 데모 샘플링 전략(demonstrator sampling strategies)의 영향을 평가했습니다.

- **Performance Highlights**: 소량의 주석 데이터로도 효과적인 분류기를 훈련할 수 있다는 결과가 도출되었습니다. 인코더 기반 모델(fine-tuning encoder-based models)의 미세 조정이 컨텍스트 내 학습(in-context learning)보다 더 나은 결과를 초래했습니다. URL과 콘텐츠 기반 기능을 모두 사용하는 분류기가 최고의 성능을 보였으며, 콘텐츠가 사용 불가능할 때는 URL만 사용하는 것도 충분한 결과를 제공했습니다.



### Machine Translation Hallucination Detection for Low and High Resource Languages using Large Language Models (https://arxiv.org/abs/2407.16470)
Comments:
          Authors Kenza Benkirane and Laura Gongas contributed equally to this work

- **What's New**: 최근 다국어 기계 번역 시스템( Machine Translation)의 발전으로 번역 정확도가 크게 향상되었으나, 여전히 환각(hallucinations)을 생성하는 문제가 남아 있습니다. 이 논문은 대규모 언어 모델(LLMs)과 의미적 유사성을 이용한 환각 감지 방법을 평가합니다. 연구는 높은 자원 언어(HRLs)와 낮은 자원 언어(LRLs) 모두에 대해 수행되었으며, 다양한 스크립트를 포함하여 16개의 언어 방향을 다룹니다.

- **Technical Details**: LLMs의 성능을 평가하기 위해 HalOmi 데이터셋을 사용했습니다. 환각 감지에서 HRLs의 경우 Llama3-70B 모델이 이전 최고 성능 모델인 BLASER-QE보다 0.16 MCC (Matthews Correlation Coefficient) 향상된 성과를 보였습니다. 반면, LRLs의 경우 Claude Sonnet가 평균적으로 다른 LLMs보다 0.03 MCC 향상된 성과를 보였습니다. 이 연구는 이진 환각 탐지를 위한 새로운 설정을 추가하였고, 성능 평가를 위해 14개의 방법을 포함했습니다.

- **Performance Highlights**: LLMs는 HRLs와 LRLs 모두에서 환각 감지에 매우 효과적입니다. HRLs에서는 Llama3-70B가 이전 최고 성능 모델을 16점 차로 능가했습니다. 반면, LRLs에서는 Claude Sonnet가 가장 우수한 성능을 보였으며, 8개 LRL 번역 방향 중 5개 방향에서 BLASER-QE보다 높은 성과를 보였습니다. 전반적으로, 연구는 HRLs와 LRLs를 포함하여 평가된 16개 언어 중 13개 언어에서 새로운 최고 성능을 확립했습니다.



### Enhancing LLM's Cognition via Structurization (https://arxiv.org/abs/2407.16434)
Comments:
          N/A

- **What's New**: 이 논문은 대형 언어 모델(LLMs)이 복잡한 장문의 텍스트를 인간처럼 구조화된 방식으로 처리할 수 있도록 새로운 '컨텍스트 구조화(context structurization)' 개념을 제안합니다. 이를 통해 LLM이 입력 텍스트를 계층적으로 구조화된 요소들로 변환하여 보다 효율적으로 인식할 수 있게 합니다. 특히, GPT-3.5-Turbo와 비교 가능한 성능을 보여주는 StruXGPT-7B 모델 개발 성공을 보고합니다.

- **Technical Details**: 논문에서는 인간의 인지 과정을 모방하여 평범한 텍스트를 주제(Scope), 측면(Aspects), 설명(Descriptions)으로 나누는 세 계층으로 구성된 구조화 방법을 제안합니다. 이 접근법은 신경인지 과학의 연구를 바탕으로 하였으며, 이를 통해 다양한 NLP 작업에서 LLM의 인지 성능을 향상시키는 것을 목표로 합니다. 예를 들어, 컨텍스트 기반 질문 응답, 허상 평가, 구문 밀집 검색 등의 작업에서 실험을 수행하였습니다.

- **Performance Highlights**: 실험 결과, StruXGPT-7B 모델이 구조화 능력의 97%를 상위 모델에서 계승하였으며, LLaMA2-70B 모델이 구조화를 통해 GPT-3.5-Turbo와 유사한 허상 평가 성능을 달성했습니다. 이는 모델 아키텍처나 크기에 관계없이 일관된 성능 향상을 보여줍니다. 또한, 이 방법은 다양한 고급 프롬프팅 기술과의 호환성을 보여주었습니다.



### FairFlow: An Automated Approach to Model-based Counterfactual Data Augmentation For NLP (https://arxiv.org/abs/2407.16431)
- **What's New**: FairFlow라는 새로운 접근법을 제안합니다. 이는 최소한의 인간 간섭만으로 병렬 데이터를 자동으로 생성하여 대립적인 텍스트 생성 모델을 훈련시키는 방법입니다.

- **Technical Details**: FairFlow는 인버터블 플로우 모델을 사용하여 속성 단어에 대한 대립어를 생성하고, 단어 치환 및 오류 수정 체계를 결합하여 병렬 데이터를 생성합니다. 이를 통해 생성 모델은 대립 텍스트를 보다 효과적으로 생성할 수 있습니다.

- **Performance Highlights**: FairFlow는 기존의 사전 기반 단어 치환 접근법의 한계를 극복하면서 좋은 성능을 유지합니다. 이러한 성과는 모델의 문법적 구성 및 일반화 능력을 향상시킵니다.



### TookaBERT: A Step Forward for Persian NLU (https://arxiv.org/abs/2407.16382)
- **What's New**: 이번 연구에서는 페르시아어 데이터로 학습된 새로운 두 개의 BERT 모델을 소개합니다. 이 모델들은 기존 7개의 모델과 비교하여 14개의 다양한 페르시아어 자연어 처리(NLU) 과제에서 평가되었습니다. 그 결과, 더 큰 모델은 평균적으로 2.8점 이상 향상된 성능을 보였습니다.

- **Technical Details**: 이번 연구는 자연어 처리(NLP)에서 기초 모델과 심층 학습의 최신 기술을 사용하여 두 개의 새로운 페르시아어 BERT 모델을 개발했습니다. BERT와 GPT 모델의 구조적 차이와 활용 방식에 따라, 일반적인 문장 이해 과제에서는 BERT가 더 우수한 성능을 보인다는 점을 이용했습니다. 학습 데이터는 다양한 공개된 페르시아어 데이터셋을 결합하여 사용되었으며, unicode 표준화를 통해 데이터 품질을 개선했습니다. 평가 데이터셋은 ParsiNLU와 다른 개별 NLU 데이터셋을 포함합니다.

- **Performance Highlights**: 더 큰 사이즈의 BERT 모델은 14개의 다양한 페르시아어 자연어 처리 과제에서 평균적으로 최소 2.8점 향상된 성능을 보였습니다. 특히, Reading Comprehension, Sentiment Analysis, Entailment 등에서 두드러진 성과를 나타냈습니다. 이 모델들은 Multilingual BERT, ParsBERT, AriaBERT 등 기존 모델보다 우수한 성능을 자랑합니다.



### Evolutionary Prompt Design for LLM-Based Post-ASR Error Correction (https://arxiv.org/abs/2407.16370)
Comments:
          in submission

- **What's New**: 이 논문은 현대의 대형 언어 모델(LLM, Large Language Models)을 이용한 생성적 오류 수정(GEC, Generative Error Correction)이 톤작 인식 시스템(ASR, Automatic Speech Recognition)의 성능을 향상시킬 수 있다는 점에서 출발합니다. ASR 시스템이 생성한 여러 가설(N-best 리스트)를 기반으로 적절한 프롬프트를 설계하여 LLM을 통해 더 나은 텍스트 예측을 할 수 있는 방안을 탐구합니다. 특히, 새로운 프롬프트를 탐색하고, 이를 진화적 프롬프트 최적화 알고리즘(EvoPrompt)을 사용하여 개선하는 접근 방식을 제안합니다.

- **Technical Details**: 제안된 시스템은 ASR 시스템의 N-best 리스트와 프롬프트를 LLM에 입력하여 오류를 수정하는 구조입니다. 프롬프트 최적화를 위하여 EvoPrompt 알고리즘을 사용하였으며, 시작 프롬프트 중 일부는 경험적으로 설계되었습니다. EvoPrompt는 초기 프롬프트 세트에서 시작하여 진화적 알고리즘을 통해 점진적으로 후보 세트를 확장하고 최적의 프롬프트를 찾아내는 방식으로 동작합니다. 이를 통해 다양한 유형의 프롬프트를 자동 생성하고 최적의 성능을 내는 프롬프트를 찾는 작업을 수행합니다.

- **Performance Highlights**: 제안된 프롬프트 최적화 알고리즘의 성능은 Task 1 of SLT 2024 GenSEC Challenge의 CHiME-4 하위 집합에서 평가되었습니다. 평가 결과, 제안한 알고리즘이 ASR 후처리 오류 수정 작업에서 효과적이고 잠재력이 있음을 확인할 수 있었습니다.



### FACTTRACK: Time-Aware World State Tracking in Story Outlines (https://arxiv.org/abs/2407.16347)
Comments:
          22 pages

- **What's New**: 새로운 논문에서 제안된 FACTTRACK는 언어 모델의 출력물에서 발생하는 사실적 모순(factual contradictions)를 감지하고 수정하는 혁신적인 방법입니다. 특히, FACTTRACK는 각 사실에 대해 시간 인식 유효(interval) 유지 기능을 제공하여 사실의 변화 가능성을 추적할 수 있습니다.

- **Technical Details**: FACTTRACK는 새로운 이벤트마다 세계 상태 데이터 구조(world state data structure)를 업데이트하기 위해 네 단계 파이프라인으로 구성됩니다: (1) 이벤트를 방향성 원자 사실(atomic facts)로 분해; (2) 각 원자 사실의 유효 시간(interval)을 세계 상태를 사용해 결정; (3) 기존의 사실과 모순이 있는지 감지; 그리고 (4) 새로운 사실을 세계 상태에 추가하고 기존의 원자 사실을 업데이트. 이 방법은 방향성 원자 사실과 시간 인식 유효 time interval를 활용해 사건 간의 모순을 감지 및 수정합니다.

- **Performance Highlights**: FACTTRACK는 LLaMA2-7B-Chat을 사용하여 구성한 기준 모델보다 현저히 우수한 성능을 보였습니다. 또한 GPT-4를 사용할 경우, 기준 모델인 GPT-4-Turbo를 능가하는 결과를 얻었습니다.



### Beyond Binary Gender: Evaluating Gender-Inclusive Machine Translation with Ambiguous Attitude Words (https://arxiv.org/abs/2407.16266)
Comments:
          The code is publicly available at \url{this https URL}

- **What's New**: 이 연구는 기존의 기계 번역 성별 편향 평가가 이분법적 성별(남성과 여성)에 집중되어 있는 한계를 지적하면서, 비이분법적(non-binary) 성별을 포함하는 새로운 벤치마크인 AmbGIMT(Gender-Inclusive Machine Translation with Ambiguous attitude words)를 제안합니다.

- **Technical Details**: 새로운 벤치마크 AmbGIMT는 모호한 태도 단어(ambiguous attitude words)를 활용해 성별 편향을 평가합니다. 또한, 감정 태도 점수(Emotional Attitude Score, EAS)라는 새로운 지표를 제안하여, 번역 과정에서 모호한 태도 단어의 태도 경향을 정량화합니다. 사용된 도구로는 새로운 오픈 소스 대형 언어 모델(LLM) 세 종과 다국어 번역 모델 NLLB-200-3.3B가 포함되었습니다.

- **Performance Highlights**: 비이분법적 성별 문맥에서의 번역 품질은 이분법적 성별 문맥에 비해 현저히 낮으며, 부정적인 태도가 더 많이 반영되는 경향이 있습니다. Lexical constraint 전략은 번역 성능을 크게 향상시키고 성별 편향을 제거하는 데 효과적이었습니다. 이 연구에서는 비이분법적 성별 문맥에서 번역 성능이 10 COMET 점수 이상 낮아진 사례가 발견되었으며, 최대 5.03%의 단어에서 부정적 태도가 더 분명하게 나타났습니다.



### LawLuo: A Chinese Law Firm Co-run by LLM Agents (https://arxiv.org/abs/2407.16252)
Comments:
          11 pages, 13 figures, 2 tables

- **What's New**: 최신 연구에서 다중 LLM 에이전트(collaborative multiple LLM agents)를 활용한 새로운 법률 상담 프레임워크인 LawLuo를 제안했습니다. 이 프레임워크는 리셉셔니스트(receptionist), 변호사(lawyer), 비서(secretary), 보스(boss)와 같은 네 개의 에이전트를 포함하며, 이들이 협력하여 사용자의 법률 상담을 처리합니다. 또한 KINLED와 MURLED라는 고품질 법률 대화 데이터셋을 구축하고 이를 바탕으로 ChatGLM-3-6b 모델을 파인튜닝하여 성능을 향상시켰습니다.

- **Technical Details**: LawLuo는 사용자의 법률 문의를 체계적으로 정리하기 위한 Tree of Legal Clarification (ToLC) 알고리즘을 제안했습니다. 이 알고리즘은 retrieve-generate-active 선택 과정을 통해 사용자가 명확한 법률 질문을 작성하도록 안내합니다. 실험결과는 LawLuo가 변호사와 같은 언어 스타일, 법률 조언의 유용성, 법률 지식의 정확성 등 세 가지 주요 지표에서 기존 모델(GPT-4 포함)보다 뛰어나다는 것을 보여줍니다.

- **Performance Highlights**: LawLuo는 변호사와 같은 언어 스타일, 법률 조언의 유용성, 법률 지식의 정확성 등 세 가지 차원에서 기존 베이스라인 LLM인 ChatGLM-3-6b 모델을 72%의 승률로 능가했습니다. 데이터 품질을 중시하는 접근 방식이 데이터 양보다 더 중요함을 강조하며, 다중 회차 대화 데이터로 파인튜닝한 결과 여러 회차 대화에서도 고품질의 응답을 지속적으로 생성합니다. 



### Exploring the Effectiveness and Consistency of Task Selection in Intermediate-Task Transfer Learning (https://arxiv.org/abs/2407.16245)
Comments:
          Accepted to ACL SRW 2024

- **What's New**: 본 연구에서는 중간 작업 전이 학습(intermediate-task transfer learning)에서 유익한 작업을 식별하는 것이 중요한 단계임을 강조합니다. 130개의 소스-타겟 작업 조합을 실험하여 다른 소스 작업과 학습 시드에 따라 전이 성능이 크게 변동함을 보여주었습니다. 또한, 쌍별 토큰 유사성을 최대 내적 검색(maximum inner product search)으로 측정하는 새로운 방법을 도입하여 작업 예측(task prediction)에서 최고의 성능을 달성하였습니다.

- **Technical Details**: 본 연구는 네 가지 대표적인 작업 선택 방법을 비교하였으며, 임베딩-프리 방법(embedding-free method) 및 텍스트 임베딩(text embedding)과 비교해 파인튜닝된 가중치(fine-tuned weights)에서 생성된 작업 임베딩(task embedding)이 더 나은 작업 전이 예측 점수를 제공하여 작업 전이 가능성을 더 잘 평가할 수 있음을 발견했습니다. 특히, 작업 예측 점수를 2.59%에서 3.96%까지 향상시켰습니다. 그러나 이러한 작업 임베딩은 추론 능력이 필요한 작업에서는 일관되게 우월성을 보이지 않았습니다.

- **Performance Highlights**: 연구 결과에 따르면 파인튜닝된 가중치의 평균(mean)을 사용하는 것보다 토큰별 유사성(token-wise similarity)을 사용하는 것이 전이 가능성을 예측하는 데 더 우수하다는 점을 시사합니다.



### PreAlign: Boosting Cross-Lingual Transfer by Early Establishment of Multilingual Alignmen (https://arxiv.org/abs/2407.16222)
- **What's New**: 새롭게 발표된 연구 논문에서는 다국어 정렬(Alignment)의 취약점을 보완하기 위해 PreAlign이라는 프레임워크를 제안했습니다. PreAlign은 대규모 언어 모델 사전 학습 전에 다국어 정렬 정보를 주입하여, 초기 학습 단계에서부터 다국어 지식 전이를 강화합니다.

- **Technical Details**: PreAlign은 두 가지 주요 단계를 통해 다국어 정렬을 달성합니다. 첫째, 영어에서 번역된 단어 쌍을 모아서 모델 초기화 시 유사한 표현을 생성하도록 학습합니다. 둘째, 학습 데이터에서 코드스위칭(Code-Switching) 전략을 사용하여 입력 텍스트의 단어를 번역된 단어로 교체합니다. 이를 통해 대규모 언어 모델의 사전 학습 동안 다국어 정렬을 유지합니다.

- **Performance Highlights**: 실험 결과, PreAlign은 영어와 유사한 문법을 갖지만 어휘가 중첩되지 않는 'English-Clone' 설정에서 표준 다국어 공동 학습보다 더 뛰어난 성능을 보였습니다. 또한, 실제 환경에서도 다양한 모델 크기에서 PreAlign의 효과가 입증되었으며, 제로샷(Zero-Shot) 크로스-링궐 전이와 지식 응용에서 뛰어난 성과를 나타냈습니다.



### Do LLMs Know When to NOT Answer? Investigating Abstention Abilities of Large Language Models (https://arxiv.org/abs/2407.16221)
Comments:
          5 pages (5th page contains References) and 2 figures

- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 신뢰성 향상을 위한 '기권 능력(Abstention Ability, AA)'이라는 중요한 측면을 조사합니다. 특히 LLMs가 불확실하거나 결정적 답변이 불가능할 때 질문에 답변하지 않고, QA 작업 성능을 유지하는 능력을 중점으로 탐구합니다. 저자들은 기존 연구들이 이 능력에 대한 표준화된 평가 방법을 제공하지 않았다고 지적하며 이를 해결하고자 새로운 '기권-QA 데이터셋(Abstain-QA)'과 '기권율(Abstention-rate)'을 제안했습니다.

- **Technical Details**: 저자들은 블랙박스 평가 방법론을 사용하여 다양한 객관식 QA(MCQA) 작업에서 LLMs의 AA를 조사합니다. 세 가지 전략–엄격 프롬프팅(Strict Prompting), 언어적 자신감 임계값(Verbal Confidence Thresholding), 연쇄적 사고(Chain-of-Thought, CoT)–을 통해 LLMs의 기권율을 측정해, 각 모델 간의 AA 성능을 분석했습니다. 기권-QA 데이터셋은 Pop-QA, MMLU, CarnaticQA와 같은 다양한 MCQA 데이터셋을 포함합니다.

- **Performance Highlights**: 최신 LLMs, 예를 들어 GPT-4와 같은 모델들조차 기권 능력에서 어려움을 겪고 있지만, 연쇄적 사고(Chain-of-Thought)와 같은 전략적 프롬프팅을 통해 이 능력을 크게 개선할 수 있음을 발견했습니다. AA 개선은 QA 작업의 전반적 성능 향상으로 이어지며, 정확도를 높이면서도 더 나은 신뢰성을 제공합니다. 저자들은 이 방법론이 기타 NLP 작업 및 속성에 확대 적용될 수 있음을 강조합니다.



### A Comprehensive Survey of LLM Alignment Techniques: RLHF, RLAIF, PPO, DPO and Mor (https://arxiv.org/abs/2407.16216)
- **What's New**: 최근 자기 지도 학습(self-supervised learning)의 발전, 수조 개 이상의 토큰을 포함하는 사전 학습 코퍼스(pre-training corpus)의 이용가능성, 지시 기반 미세 조정(instruction fine-tuning), 수십억 개 파라미터를 갖춘 대형 트랜스포머(large Transformers)의 개발 덕분에, 대형 언어 모델(LLMs)은 이제 인간의 질의에 대한 사실적이고 일관된 응답을 생성할 수 있게 되었습니다. 그러나 융합된 질의 데이터의 질이 혼합되어 불필요한 응답을 생성하는 것은 여전히 중요한 과제로 남아 있습니다. 이 논문은 지난 2년간 제안된 다양한 접근 방식을 주제별로 분류하고, 인간의 기대에 맞춰 대형 언어 모델을 개선하는 방법을 상세히 설명하고 있습니다. 이 작업을 통해 독자들이 현 시점에서의 분야의 상태를 체계적으로 이해할 수 있도록 돕는 것을 목표로 합니다.

- **Technical Details**: 논문에서는 대형 언어 모델(LLMs)을 인간의 기대에 맞추기 위해 제안된 다양한 방법들을 카테고리화하고, 각 정렬 방법(alignment method)을 자세히 설명합니다. 특히, 자기 지도 학습(self-supervised learning), 사전 학습(pre-training), 미세 조정(fine-tuning), 대형 트랜스포머(large Transformers)와 같은 기술적 요소들이 어떻게 결합하여 모델의 성능을 개선할 수 있는지에 대해 논의합니다.

- **Performance Highlights**: 논문은 제안된 다양한 방법론들이 실제로 어떻게 모델의 성능을 향상시키는지에 대한 사례 및 결과들을 다루고 있습니다. 특히, 인간의 기대와 모델의 응답 사이의 일치를 높이기 위한 노력들이 실제 응답의 품질을 어떻게 개선했는지를 보여줍니다.



### Graph-Structured Speculative Decoding (https://arxiv.org/abs/2407.16207)
- **What's New**: 이번 연구에서는 Graph-structured Speculative Decoding (GSD)라는 혁신적인 접근 방식을 도입하여 대형 언어 모델 (LLM)의 추론 속도를 크게 향상시켰습니다. 기존의 speculative decoding (SD)을 개선하기 위해, GSD는 단일 가설 대신 여러 가설을 생성하고 이를 검증함으로써, 최종 출력으로 채택되는 토큰의 비율을 높였습니다.

- **Technical Details**: GSD는 directed acyclic graph (DAG)을 활용하여 여러 가설을 효율적으로 관리합니다. 이 그래프 구조를 통해 반복되는 토큰 시퀀스를 예측하고 통합하여, 초안 모델의 계산 요구량을 크게 줄일 수 있습니다. 결과적으로, 초안 모델이 만든 토큰 그래프를 LLM이 단일 시퀀스로 펼쳐서 모든 가설을 동시에 검증하고, 가장 긴 시퀀스를 최종 출력으로 채택합니다.

- **Performance Highlights**: 70억 개의 파라미터를 가진 LLaMA-2 모델을 포함한 다양한 LLM에서 GSD를 적용한 결과, 기존의 speculative decoding 대비 1.73배에서 1.96배까지 속도 향상이 있었습니다. 이는 표준 speculative decoding 보다도 훨씬 더 높은 성능을 보여줍니다.



### Structural Optimization Ambiguity and Simplicity Bias in Unsupervised Neural Grammar Induction (https://arxiv.org/abs/2407.16181)
Comments:
          Accepted in ACL2024 Findings, 16 pages, 10 figures

- **What's New**: 이 논문은 전통적인 우도 손실(likelihood loss)을 사용한 비지도 신경 문법 학습(Unsupervised Neural Grammar Induction, UNGI)에서 발생하는 두 가지 주요 문제를 다룹니다: 1) 구조적 최적화 모호성(structural optimization ambiguity)으로 인해 구조적으로 애매한 최적 문법 중 하나를 임의로 선택하게 되고, 2) 구조적 단순성 편향(structural simplicity bias)으로 인해 문법이 구문 트리(parse tree)를 구성할 때 규칙을 덜 활용하게 됩니다. 이에 대한 해결책으로, 연구팀은 사전 학습된 구문 분석기(pre-trained parsers)의 구조적 편향을 활용하여 문장 단위의 구문 집중(sentence-wise parse-focusing) 기법을 제안했습니다.

- **Technical Details**: 연구의 핵심은 사전 학습된 구문 분석기를 이용해 각 문장의 손실 평가 시 사용할 구문 풀이 좁혀지도록 하는 구문 집중 기법을 도입한 것입니다. 이 접근 방식은 내부 알고리즘 내에서 몇 가지 선택된 구문만을 고려하므로, 구문 간의 모호성을 줄이고 불필요한 단순한 구문 트리들을 제거하여 단순성 편향을 제한합니다.

- **Performance Highlights**: 비지도 학습 구문 분석(Benchmark Tests)에서 제안된 방법은 성능을 크게 향상시키고, 고차원적인 변동성과 지나치게 단순한 구문 트리에 대한 편향을 효과적으로 줄였습니다. Penn Treebank(PTB)와 Chinese Penn Treebank(CTB), SPMRL 데이터셋을 포함한 영어와 다른 10개 언어의 구문 분석 작업에서 최신 기술의 모델들을 능가하는 결과를 보였습니다. 또한 다양한 구문 집중 편향을 조사한 결과, 이종의 멀티 구문 분석기(multi-parsers)가 권장되었습니다.



### Progressively Modality Freezing for Multi-Modal Entity Alignmen (https://arxiv.org/abs/2407.16168)
Comments:
          13pages, 8 figures, Accepted by ACL2024

- **What's New**: 이번 논문에서는 이종 지식 그래프(Knowledge Graphs) 간 동일 엔티티를 식별하는 멀티 모달 엔티티 정렬(Multi-Modal Entity Alignment, MMEA)을 다룹니다. PMF(Progressive Modality Freezing)라는 혁신적인 전략을 제안하며, 이는 정렬과 무관한 특징을 제거하고 멀티 모달 특징 융합을 개선합니다. 특히, 크로스 모달 연관 손실(Cross-Modal Association Loss)을 도입하여 모달 일관성을 증진시킵니다.

- **Technical Details**: PMF 전략은 세 가지 주요 부분으로 구성되어 있습니다: 멀티 모달 엔티티 인코더(Multi-Modal Entity Encoder), 점진적 멀티 모달 특징 통합(Progressive Multi-Modality Feature Integration), 그리고 통합 훈련 목표(Unified Training Objective). 첫째, 각 모달리티의 원시 입력을 엔티티 임베딩으로 변환하는 멀티 모달 인코더를 사용합니다. 이후, 훈련 중 정렬과 무관한 특징을 점진적으로 '프리즈(freeze)'하고 유용한 멀티 모달 정보를 통합합니다. 마지막 단계로, 다른 지식 그래프와 모달리티 그래프 간 대조 손실(Contrastive Loss)을 통해 모델을 최적화합니다.

- **Performance Highlights**: 9개 데이터셋에서 실험적 평가를 통해 PMF의 우수성을 입증하였으며, 최첨단 성능과 모달리티 프리징 전략의 타당성을 보여줍니다. 이러한 결과는 PMF가 다양한 데이터셋과 실험 조건에서 뛰어난 성능을 발휘함을 확인시켜 줍니다.



### Robust Privacy Amidst Innovation with Large Language Models Through a Critical Assessment of the Risks (https://arxiv.org/abs/2407.16166)
Comments:
          13 pages, 4 figures, 1 table, 1 supplementary, under review

- **What's New**: 이 연구는 EHRs (Electronic Health Records)와 NLP (Natural Language Processing)를 대형 언어 모델(LLMs)로 통합하여 의료 데이터 관리와 환자 치료를 개선하는 방법을 탐구합니다. 특히, 생의학 연구를 위해 안전하고 HIPAA 규정을 준수하는 합성 환자 노트를 생성하는 모델을 사용했습니다.

- **Technical Details**: 연구는 GPT-3.5, GPT-4, Mistral 7B를 사용하여 MIMIC III 데이터셋으로부터 비식별화(de-identified) 및 재식별화(re-identified) 데이터를 가지고 합성 노트를 생성했습니다. 텍스트 생성에는 템플릿과 키워드 추출을 사용하여 문맥적으로 관련 있는 노트를 만들었으며, 비교를 위해 one-shot 생성을 수행했습니다. 프라이버시 평가는 PHI (Protected Health Information) 발생을 점검했고, 텍스트의 유용성은 ICD-9 코딩 작업을 통해 테스트했습니다. 텍스트 품질은 ROUGE와 코사인 유사성(cosine similarity) 지표를 사용하여 원본 노트와의 의미적 유사성을 측정했습니다.

- **Performance Highlights**: PHI 발생과 텍스트 유용성 분석 결과, 키워드 기반 방법이 낮은 위험도와 양호한 성능을 보였습니다. One-shot 생성 방식은 특히 지리적 위치와 날짜 카테고리에서 PHI 노출과 PHI 동시 발생률이 가장 높았습니다. Normalized One-shot 방식은 가장 높은 분류 정확도를 달성했습니다. 프라이버시 분석에서는 데이터 유틸리티와 프라이버시 보호 간의 중요한 균형이 나타났으며, 재식별화된 데이터가 항상 비식별화된 데이터보다 우수한 성능을 보였습니다. 이 연구는 데이터 유용성을 유지하면서 프라이버시를 보호하는 합성 임상 노트를 생성하는 키워드 기반 방법의 효과를 입증하며, 미래의 데이터 사용 및 공유 방식을 변화시킬 잠재력을 가지고 있습니다.



### DDK: Distilling Domain Knowledge for Efficient Large Language Models (https://arxiv.org/abs/2407.16154)
- **What's New**: 본 논문에서는 대형 언어 모델(LLM)의 지식을 작은 모델(학생 모델)로 효과적으로 전달하기 위해, 새로운 LLM 디스틸레이션(Distillation) 프레임워크인 DDK를 소개합니다. DDK는 교사 모델과 학생 모델 간의 도메인 성능 차이에 따라 디스틸레이션 데이터셋의 구성을 동적으로 조정해 디스틸레이션 프로세스를 더욱 안정적이고 효과적으로 만듭니다.

- **Technical Details**: DDK는 교사 모델과 학생 모델의 도메인별 성능 차이를 정량화한 뒤, 계산된 도메인 불 일치 요인에 따라 데이터를 샘플링합니다. 또한, 최적화 알고리즘에서 영감을 받아 도메인 지식에 기반한 샘플링 전략의 안정성을 높이기 위한 부드러운 업데이트 메커니즘을 제안합니다. 이 과정에서, 교사 모델과 학생 모델의 출력 로짓의 차이를 최소화하는 손실 함수를 사용합니다.

- **Performance Highlights**: 실험 결과, DDK는 기존의 디스틸레이션 방법들과 비교했을 때 학생 모델의 성능을 현저하게 향상시켰습니다. 이는 교사 모델과 학생 모델 간의 큰 성능 차이가 있는 도메인에 더 많은 데이터를 할당함으로써 가능한 결과입니다.



### CHIME: LLM-Assisted Hierarchical Organization of Scientific Studies for Literature Review Suppor (https://arxiv.org/abs/2407.16148)
Comments:
          2024 ACL Findings

- **What's New**: 새로운 연구에서는 대형 언어 모델(LLMs)을 사용하여 과학 연구를 체계적으로 분류하는 계층적 조직 구조(hierarchical organizations)를 생성하는 방법을 조사합니다. 이는 연구자들이 문헌 리뷰를 보다 효율적으로 수행할 수 있도록 돕기 위한 것입니다. 이 연구에서는 생의학 분야를 중심으로 전문가가 큐레이션한 CHIME 데이터셋을 수집하고 제공했습니다.

- **Technical Details**: 문헌 리뷰를 위해 체계적인 계층 구조(tree structures)를 생성하는 과정을 설명합니다. 계층 구조의 노드는 주제별 범주를 나타내며, 각 노드는 해당 범주에 할당된 연구와 연결됩니다. 연구에서는 Cochrane Database of Systematic Reviews를 이용해 체계적인 리뷰와 해당 리뷰에 포함된 연구를 수집하여 LLM 기반의 계층 생성 파이프라인을 통해 초기에 생성되었습니다. 그 후 전문가들이 검토하고 오류를 수정하는 인간-루프 프로세스를 도입했습니다.

- **Performance Highlights**: LLM은 범주를 생성하고 조직화하는 데 있어 전반적으로 좋은 성능을 보였으나, 연구를 적절한 범주에 할당하는 데 있어서는 개선이 필요한 것으로 나타났습니다. 전문가가 수정한 계층 구조를 통해 LLM의 성능을 정량화하고, 인간 피드백을 활용한 'corrector' 모델을 훈련시켜 연구 할당 정확도를 12.6 F1 포인트 향상시켰습니다.



### Finetuning Generative Large Language Models with Discrimination Instructions for Knowledge Graph Completion (https://arxiv.org/abs/2407.16127)
Comments:
          Accepted in the 23rd International Semantic Web Conference (ISWC 2024)

- **What's New**: DIFT는 지식 그래프 (Knowledge Graph, KG) 완성을 위해 개발된 새로운 파인튜닝 프레임워크입니다. 이는 대규모 언어 모델 (Large Language Models, LLMs)의 KG 완성 능력을 최대한 활용하고, 기존 모델들의 출력 연결 오류를 피하기 위해 고안되었습니다.

- **Technical Details**: DIFT는 경량화된 모델을 사용해 후보 엔티티를 얻고, LLM을 선별 지시 (Discrimination Instructions)와 함께 파인튜닝하여 주어진 후보들 중에서 정확한 엔티티를 선택하도록 합니다. 성능을 높이면서도 지시 데이터의 양을 줄이기 위해, DIFT는 유용한 샘플을 선택하는 절단 샘플링 (Truncated Sampling) 방법을 사용하고, KG 임베딩 (Embeddings)을 LLM에 주입합니다.

- **Performance Highlights**: DIFT는 FB15K-237 데이터셋에서 0.364 Hits@1, WN18RR 데이터셋에서 0.616 Hits@1을 기록하며, 최첨단 KG 완성 모델들을 능가하는 성능을 보여줍니다.



### Analyzing the Polysemy Evolution using Semantic Cells (https://arxiv.org/abs/2407.16110)
Comments:
          11 pages, 2 figures. arXiv admin note: text overlap with arXiv:2404.14749

- **What's New**: 이 논문은 단어의 여러 의미(polsemy)가 어떻게 진화의 결과로 나타나는지에 대한 사례 연구입니다. 저자는 단어 'Spring'의 네 가지 의미를 중심으로, 진화 과정에서 단어의 의미가 어떻게 변화하는지 분석했습니다. 이를 통해 단어의 다의성을 학습 기반이 아닌 진화적 관점에서 이해하는 방법론을 제시합니다.

- **Technical Details**: 연구는 Semantic Cells의 개념을 기반으로 합니다. Semantic Cells는 단어의 의미를 캡슐화하는 개념으로, 초기 상태에 소량의 다양성을 도입하여 단어의 진화를 분석합니다. 연구에서는 Chat GPT를 사용하여 1000개의 문장들 속에서 단어 Spring의 네 가지 의미를 수집하고, 이를 특정 순서로 배열하여 의미의 변화를 모니터링했습니다.

- **Performance Highlights**: 주요 성과로, 단어의 의미가 진화하는 순서로 배열할 때 단어가 가장 많은 다의성을 획득한다는 점을 밝혀냈습니다. 이는 단어의 다의성이 단순 학습의 결과가 아닌, 의미의 진화적 변화에 의해 초래된다는 것을 보여줍니다. 이를 통해 단어 다의성(polsemy)에 대한 새로운 분석 틀을 제시했습니다.



### KaPQA: Knowledge-Augmented Product Question-Answering (https://arxiv.org/abs/2407.16073)
Comments:
          Accepted at the ACL 2024 Workshop on Knowledge Augmented Methods for NLP

- **What's New**: 최근의 대형 언어 모델(LLMs)의 발전으로 인해 도메인 특화 질문-답변(QA) 시스템에 대한 관심이 급증하고 있습니다. 이에 따라 정확한 성능 평가가 쉬운 일이 아니라는 문제점이 제기되고 있습니다. 이 문제를 해결하기 위해, 우리는 Adobe Acrobat과 Photoshop 제품을 중심으로 한 두 개의 제품 QA 데이터셋을 소개합니다. 이 데이터셋들은 도메인 특화 제품 QA 작업에서 기존 모델의 성능을 평가하는 데 사용될 수 있습니다. 추가로, 우리는 지식 기반의 새로운 RAG-QA 프레임워크를 제안합니다.

- **Technical Details**: 우리의 새로운 LLM 기반 Knowledge-Driven RAG-QA 프레임워크는 도메인 지식을 반영하도록 설계되었습니다. 이 프레임워크는 질의 확장을 위해 포괄적인 지식 기반을 활용하며, 이를 통해 도메인 특화 QA 작업에서의 검색 및 생성 성능을 향상시킵니다. Adobe의 HelpX 웹 페이지에서 데이터를 수집하고, 제품 전문가가 작성한 절차적인 '어떻게' 질문과 답변 쌍을 포함하고 있습니다.

- **Performance Highlights**: 우리의 실험 결과, 도메인 지식을 활용하여 질의를 재구성하면 표준 RAG-QA 방법보다 검색 및 생성 성능이 약간 향상되는 것을 확인했습니다. 그러나 이러한 개선은 미미한 수준에 머물렀으며, 이는 우리가 제안한 데이터셋이 매우 도전적인 것임을 나타냅니다. Adobe Acrobat 데이터셋의 질문 중 절반 이상은 명확한 사용자 의도를 포함하지 않고 있으며, 약 4.71 스텝의 다중 단계 솔루션이 요구됩니다.



### Leveraging Large Language Models to Geolocate Linguistic Variations in Social Media Posts (https://arxiv.org/abs/2407.16047)
- **What's New**: 이번 연구는 이탈리아어로 작성된 트윗의 지리적 위치를 판별하는 GeoLingIt 챌린지를 해결하기 위해 대형 언어 모델(LLMs)을 활용한 것입니다. GeoLingIt는 트윗의 지역과 정확한 좌표를 예측하는 것을 목표로 합니다. 이 연구는 미리 학습된 LLMs을 세밀하게 튜닝하여 이 두 가지 지리적 위치 판별 과제를 동시에 해결하는 접근 방식을 취합니다. 이 논문에서는 Bertinoro 국제 봄 학교 2024의 '대형 언어 모델' 과정의 일환으로 이루어졌으며, GitHub에 코드가 공개되어 있습니다.

- **Technical Details**: GeoLingIt 데이터셋은 총 15039개의 샘플로 구성되어 있고, 이를 다시 학습, 평가, 테스트 세트로 분할하였습니다. 데이터를 전처리하는 과정에서 두 가지 하위 과제(지역 예측, 좌표 예측)를 하나로 합쳐 새로운 학습 데이터셋을 만들었습니다. 본 연구에서는 이탈리아어로 작성된 Camoscio-7B, ANITA-8B, Minerva-3B의 세 가지 LLMs을 사용하여 이 데이터셋에서 성능을 비교했습니다. Camoscio 모델은 LLaMA 기반 7 billion 파라미터 모델로 Low-Rank Adaptation(LoRA)를 사용하였고, ANITA 모델은 8 billion 파라미터로 Direct Preference Optimization(DPO) 방법론을 사용하였습니다. Minerva 모델은 이탈리아어로 처음부터 학습된 LLM으로, 350 million, 1 billion, 3 billion 파라미터 버전을 포함하고 있습니다.

- **Performance Highlights**: 실험은 Tesla T4 GPU 16GB를 사용하여 진행되었으며, 메모리 제한 때문에 4-bit 양자화된 모델을 사용하고 LoRA를 통해 훈련 파라미터를 줄였습니다. 각 모델을 10 에포크 동안 학습하였으며, 모델 튜닝에는 언어 모델링 목표를 사용했습니다. 실험 결과, 다양한 지리적 위치 판별 태스크에서 높은 예측 정확도를 달성하였습니다.



### Enhancing Temporal Understanding in LLMs for Semi-structured Tables (https://arxiv.org/abs/2407.16030)
Comments:
          Total Pages 18, Total Tables 6, Total figures 7

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)이 표 형식 데이터에서 시간적 추론을 수행하는 능력에 대한 한계를 체계적으로 분석하였습니다. 이를 통해 TempTabQA 데이터셋을 개선하고, 새로운 접근 방식인 C.L.E.A.R를 도입하여 모델의 성능을 향상시켰습니다. 간접 감독(auxiliary data)을 통해 모델 성능을 크게 향상시키는 방법도 제시하였습니다.

- **Technical Details**: 연구에서는 TempTabQA 데이터셋을 이용하여 LLMs의 시간적 추론 능력에 대한 한계를 분석하였습니다. C.L.E.A.R(Comprehend, Locate, Examine, Analyze, Resolve)라는 새로운 접근 방식을 도입하여 명시적 지시를 통해 모델이 더 잘 추론할 수 있도록 하였습니다. 또한, TRAM 데이터셋을 통해 간접 감독으로 모델을 미세 조정하여 성능을 개선했습니다.

- **Performance Highlights**: C.L.E.A.R 접근 방식과 간접 감독 방법을 결합함으로써 모델의 성능이 크게 향상되었습니다. 특히, TempTabQA 데이터셋의 전체 테스트 세트에서 모델의 정확도가 현저히 개선되었으며, 이로써 시간적 능력에 대한 이해와 모델의 성능이 대폭 향상되었습니다.



### Boosting Reward Model with Preference-Conditional Multi-Aspect Synthetic Data Generation (https://arxiv.org/abs/2407.16008)
- **What's New**: 새로운 연구인 RMBoost는 대규모 언어 모델(LLMs)을 인간의 선호에 맞추기 위한 보상 모델(RMs)의 품질을 향상시키기 위해 고안된 새로운 합성 선호 데이터 생성 패러다임을 제시합니다. 기존 방법과 달리 RMBoost는 두 응답을 생성한 후 선호 라벨을 얻는 대신, 먼저 하나의 응답을 생성하고 선호 라벨을 선택한 다음, 두 번째 응답을 그 선호 라벨과 첫 번째 응답에 조건화하여 생성합니다.

- **Technical Details**: RMBoost의 주요 혁신은 선호 쌍을 생성하는 점진적인 방식에 있습니다. 기존 응답 쌍에 대해 선호 라벨을 예측하는 것 대신, RMBoost는 먼저 하나의 응답(y1)을 생성하고 선호 라벨(l)을 선택합니다. 그 다음, 선호 라벨과 첫 번째 응답(y1)에 따라 개선되거나 악화된 두 번째 응답(y2)을 생성합니다. 이를 통해 선호 라벨의 노이즈를 줄이고 다양한 응답을 생성할 수 있습니다.

- **Performance Highlights**: RMBoost는 QA Feedback, Ultra Feedback, TLDR summarization 데이터셋에서 기존의 합성 선호 데이터 생성 기법보다 우수한 성능을 보였습니다. PaLM 2-L과 GPT-4를 사용한 실험에서, RMBoost는 네 가지 다른 보상 모델(RM) 백본에 대해 선호 예측 정확도에서 현저하게 더 높은 성능을 나타냈습니다. RMBoost를 사용한 LLM은 항상 더 높은 승리율을 보였으며, 응답의 다양성을 향상시키는 데 매우 효과적임을 증명했습니다.



### SocialQuotes: Learning Contextual Roles of Social Media Quotes on the Web (https://arxiv.org/abs/2407.16007)
- **What's New**: 이번에 발표된 논문에서는 웹 컨텍스트 안에서 소셜 미디어 엔터티의 역할을 자동으로 주석 달아주는 혁신적인 언어 모델링 프레임워크를 소개합니다. 이 프레임워크는 소셜 미디어 임베딩을 인용구(quotes)로 비유하고, 페이지 컨텍스트를 구조화된 자연어 신호로 정형화합니다. 이 연구의 주요 기여로는 Common Crawl에서 수집한 3,200만 개 이상의 소셜 인용구가 포함된 'SocialQuotes'라는 새로운 데이터 셋의 릴리스와 이를 활용한 역할 분류 사례 연구가 포함됩니다.

- **Technical Details**: 연구진은 소셜 미디어 인용구의 역할을 웹 페이지 내에서의 맥락을 통해 분석하기 위해 통신 이론을 활용했습니다. 제안된 프레임워크는 현대의 대형 언어 모델(LLM)이 페이지 컨텍스트를 통해 소셜 미디어 인용구의 카테고리 역할을 추론할 수 있다는 가설에 기반합니다. 또한 8.3K개의 인용구에 대한 크라우드 소싱 주석을 제공하여 연구의 기본 가설을 검증했습니다.

- **Performance Highlights**: 이 프레임워크는 모든 소셜 미디어 플랫폼에서 임베딩된 데이터를 확장 가능하며, 소셜 미디어 플랫폼 데이터 없이도 소셜 미디어를 모델링할 수 있는 새로운 가능성을 열었습니다. 데이터 수집의 비용을 대폭 절감할 수 있는 이점이 있으며, 여러 도메인과 플랫폼에서의 역할 분포를 분석하여 흥미로운 결과를 도출했습니다. 또한 최신 LLM을 활용하여 소셜 인용구의 역할을 예측할 수 있음을 증명했습니다.



### Multimodal Input Aids a Bayesian Model of Phonetic Learning (https://arxiv.org/abs/2407.15992)
Comments:
          12 pages, 5 figures

- **What's New**: 이 논문에서는 아동이 모국어의 단어들을 구성하는 독특한 소리를 차별화하는 데 있어 멀티모달 정보(multi-modal information), 즉 성인 언어와 화자의 얼굴 비디오 프레임이 결합된 정보를 활용하는 computational model의 학습 효과를 조사했습니다. 이를 위해 기존 오디오 코퍼스를 사용하여 고품질의 합성 비디오를 생성하는 방법을 도입했습니다.

- **Technical Details**: 제안된 학습 모델은 오디오와 비디오 입력을 동시에 훈련 및 테스트한 경우 phoneme discrimination 성능에서 8.1% 상대적 향상을 달성했습니다. 또한, 오디오 입력만을 테스트했을 때도 3.9% 더 높은 성능을 보였으며, 이는 비주얼 정보가 음향적인 구별 학습을 촉진함을 시사합니다. 특히, 소음이 있는 오디오 환경에서 비주얼 정보는 오디오 모델의 성능 저하의 67%를 회복시켰습니다.

- **Performance Highlights**: 멀티모달 정보를 사용한 모델이 오디오 전용 모델보다 높은 성능을 보였고, 특히 소음이 있는 환경에서 큰 이점을 나타냈습니다.



### Multilingual Fine-Grained News Headline Hallucination Detection (https://arxiv.org/abs/2407.15975)
- **What's New**: 자연어 처리 모델의 뉴스 헤드라인 생성기는 종종 'hallucination' 문제로 인해 기사와 일치하지 않는 헤드라인을 생성합니다. 이를 해결하기 위해 처음으로 다국어 및 세부적인 뉴스 헤드라인 헛소리(Hallucination) 감지 데이터셋을 도입했습니다. 이 데이터셋에는 5개 언어로 11,000개 이상의 기사-헤드라인 쌍이 포함되어 있으며, 각 쌍은 전문가들이 구체적인 hallucination 유형으로 주석을 달았습니다.

- **Technical Details**: 기사와 헤드라인 상의 세부적인 'entailment' 관계를 식별하기 위해 다국어에서 세분화된 hallucination 감지 작업을 수행합니다. 헤드라인의 오류 유형을 더 정확하게 분석하기 위해 자연어 추론 데이터셋으로 사전 훈련된 모델을 사용하고, 자연어 설명을 통합한 seq2seq 기반 분류기를 도입했습니다. 또한 다양한 대형 언어 모델(LLM)에 대해 언어 종속적 데모 선택(language-dependent demonstration)과 거칠게-섬세하게(coarse-to-fine) 프롬프트 전략의 새로운 기법으로 실험했습니다.

- **Performance Highlights**: 최신 자연어 모델(ChatGPT, PaLM2 등)은 기존의 소형 모델(mT5-XXL)보다 성능이 떨어진다는 점을 확인했습니다. 그러나 언어 종속적 데모 선택과 거칠게-섬세하게 프롬프트 전략을 도입함으로써 LLM의 몇 샷 학습 성능이 크게 향상되었고, 예제 F1 점수가 상승했습니다. 이러한 결과는 다국어 뉴스 데이터에서의 세분화된 hallucination 감지 정확성을 높일 수 있는 중요한 통찰을 제공합니다.



### RedAgent: Red Teaming Large Language Models with Context-aware Autonomous Language Agen (https://arxiv.org/abs/2407.16667)
- **What's New**: 최근에 GPT-4와 같은 대형 언어 모델(LLMs, Large Language Models)은 Code Copilot과 같은 실세계 응용 프로그램에 통합되고 있습니다. 이러한 응용 프로그램은 LLM의 공격 표면을 확장시키며, 특히 'jailbreak' 공격으로 인해 독성이 강한 응답을 유도하는 등의 위협에 노출되고 있습니다. 본 논문에서는 이러한 위협을 식별하기 위해 'RedAgent'라는 다중 에이전트 시스템을 제안합니다. 이 시스템은 기존의 공격을 'jailbreak 전략'이라는 개념으로 추상화하여 맥락 인지형 탈출 프롬프트를 생성합니다.

- **Technical Details**: RedAgent는 크게 세 가지 단계로 나누어져 있습니다: 1) Context-aware Profiling Stage에서 대상 LLM의 응용 범위를 파악하고 맥락 기반의 악의적인 목표를 작성합니다. 2) Adaptive Jailbreak Planning Stage에서는 효과적인 전략을 토대로 공격 계획을 수립합니다. 3) Attacking and Reflection Stage에서는 구체적인 탈출 프롬프트를 생성하여 대상 LLM을 실험하고 반영된 피드백을 바탕으로 스킬 메모리를 업데이트합니다. 이를 통해 지속적으로 공격 전략을 개선합니다.

- **Performance Highlights**: RedAgent는 대부분의 블랙박스 LLM를 5회 이하의 시도로 탈출할 수 있으며, 기존의 red teaming 방법보다 두 배의 효율성을 보입니다. 또한, OpenAI 마켓플레이스의 60개 맞춤형 GPT 응용 프로그램을 대상으로 60건의 심각한 취약성을 발견하였고, 이 과정에서 맥락 인지형 탈출 프롬프트만으로 각 취약성을 2회 이하의 시도로 탐지했습니다.



### Imperfect Vision Encoders: Efficient and Robust Tuning for Vision-Language Models (https://arxiv.org/abs/2407.16526)
- **What's New**: 최근 비전 언어 모델(Vision Language Models, VLMs)이 비주얼 질문 답변(Visual Question Answering)과 이미지 캡셔닝(Image Captioning)에서 인상적인 성능을 보이고 있습니다. 그러나 기존 오픈소스 VLMs는 선훈련된 후 동결된 비전 인코더(vision encoders)에 지나치게 의존하고 있습니다. 이로 인해 이미지 이해 오류가 발생하고, 이러한 오류가 VLM의 응답으로 전파되어 성능이 저하됩니다. 본 연구에서는 VLM 내 비전 인코더를 효율적으로 업데이트하는 방법을 제안하여, 기존의 오류가 발생한 데이터에서 상당한 성능 향상을 이루었으며, 전체적인 견고성을 유지했습니다.

- **Technical Details**: 연구팀은 VLM의 구성 요소인 비전 인코더와 언어 모델에서 특정 데이터를 사용하여 별도로 파인 튜닝을 진행했습니다. 그 결과, 비전 인코더를 업데이트하는 것이 성능 향상에 더 효과적임을 발견했습니다. 파라미터 효율적인 튜닝 방법을 통해 데이터에 국한된 로컬화된 업데이트를 진행하여, 기존 지식을 보존하면서도 모델 성능을 향상시켰습니다. 이를 위해 트랜스포머 모델의 MLP 레이어와 어텐션 헤드 등에서 특정 파라미터만을 선택해 업데이트하는 방식으로 구현했습니다.

- **Performance Highlights**: 제안된 방법은 다양한 벤치마크에서 CLIP과 이를 기반으로 한 VLM 모델을 업데이트함으로써 기존 지식을 보존하면서도 성능을 향상시켰습니다. 이를 통해 소수 샷 학습(few-shot learning)과 지속적인 업데이트에서도 탁월한 성능을 보였습니다. 또한, 모델 업데이트가 비전 인코더만의 문제가 아님을 밝히며, 이 방법이 모든 트랜스포머 모델에 적용될 수 있음을 보여줬습니다.



### Psychomatics -- A Multidisciplinary Framework for Understanding Artificial Minds (https://arxiv.org/abs/2407.16444)
Comments:
          15 pages, 4 tables, 2 figures

- **What's New**: 새롭게 소개된 Psychomatics는 인지과학, 언어학, 컴퓨터 과학을 연결하는 다학문적 프레임워크입니다. 이 프레임워크는 대형 언어 모델(LLMs)이 정보를 습득하고 학습하며 기억하고 이를 사용해 출력물을 생성하는 방식을 이해하는 데 중점을 두고 있습니다. 이를 통해 인간의 언어 발전 및 사용 과정과 LLMs의 차이를 비교하고자 합니다.

- **Technical Details**: Psychomatics는 이론 기반 연구 질문을 중심으로 한 비교 방법론(comparative methodology)을 사용합니다. 연구는 LLMs가 훈련 데이터 내 복잡한 언어 패턴을 어떻게 맵핑하고 조작하는지 분석합니다. 또한 LLMs가 Grice의 협동 원칙(Cooperative Principle)을 따르며 관련 있고 정보성 있는 응답을 제공할 수 있다고 합니다. 하지만 인간의 인지는 경험적, 감정적, 상상적 요소를 포함한 다중의미 소스로부터 유래하며 사회적, 발달적 궤적에 기반을 둡니다.

- **Performance Highlights**: 현재 LLMs는 물리적 구현이 부족하여 인간의 이해와 표현을 형성하는 인식, 행동, 인지 사이의 복잡한 상호작용을 이해하는 데 한계가 있습니다. 그러나, Psychomatics는 언어, 인지, 지능의 본질에 대한 획기적인 통찰을 제공할 가능성이 있으며, 인간과 유사한 더 강력한 AI 시스템을 개발하는데 기여할 수 있을 것입니다.



### PrimeGuard: Safe and Helpful LLMs through Tuning-Free Routing (https://arxiv.org/abs/2407.16318)
Comments:
          ICML 2024 NextGenAISafety workshop version with links to implementation and dataset

- **What's New**: PrimeGuard라는 새로운 Inference-Time Guardrails (ITG) 방법이 소개되었습니다. 이 방법은 언어 모델(Language Models, LMs)의 출력이 안전하면서도 고품질을 유지하도록 합니다. PrimeGuard는 모델을 튜닝하지 않고도 다양한 안내 지침에 따라 요청을 다르게 처리하여 안전과 유용성을 동시에 향상시킵니다. 또한, 안전 평가를 위한 다양한 red-team 기준인 safe-eval를 구축하고 공개하였습니다.

- **Technical Details**: PrimeGuard는 구조화된 제어 흐름(structured control flow)을 사용하여 요청을 다양한 모델 인스턴스로 라우팅합니다. 이 인스턴스들은 각각 다른 지침을 따르며, 모델의 지침 따르기 능력과 문맥 학습(in-context learning)을 활용합니다. 이 방법은 시스템 설계자의 지침을 각 쿼리(query)에 동적으로 컴파일하여 반영합니다. 특별한 튜닝 없이도 높은 성능을 유지하면서 안전성과 유용성을 극대화합니다.

- **Performance Highlights**: PrimeGuard는 반복적인 jailbreak 공격에 대한 저항력을 크게 높이고, 안전 가드레일링의 최신 성과를 기록하며, alignment 튜닝된 모델의 유용성 점수를 일치시키는 성과를 보였습니다. 구체적으로는, 경쟁하는 모든 기준을 능가하며 안전한 응답 비율을 61%에서 97%로 높이고, 평균 유용성 점수를 4.17에서 4.29로 증가시켰으며, 공격 성공률을 100%에서 8%로 줄였습니다.



### A Multi-view Mask Contrastive Learning Graph Convolutional Neural Network for Age Estimation (https://arxiv.org/abs/2407.16234)
Comments:
          20 pages, 9 figures

- **What's New**: 이번 연구는 나이 추정(age estimation) 작업을 개선하기 위해 Multi-view Mask Contrastive Learning Graph Convolutional Neural Network (MMCL-GCN)을 제안합니다. 이 방법은 복잡한 비정형 구조를 모델링하는 기존 CNN 및 Transformer 기반 방법의 비유연성과 중복성을 해결합니다.

- **Technical Details**: MMCL-GCN 네트워크는 특징 추출 단계와 나이 추정 단계의 두 가지 주요 단계로 구성됩니다. 특징 추출 단계에서는 그래프 구조를 도입하여 얼굴 이미지를 입력으로 구성하고, 비대칭 시아미즈 네트워크 아키텍처를 사용하여 이미지의 구조적 및 의미적 정보를 학습합니다. 나이 추정 단계에서는 multi-layer extreme learning machine (ML-IELM)과 identity mapping을 채택하여 특징을 활용합니다. Contrastive Learning과 Masked Image Modeling을 결합하여 효율적인 학습 메커니즘을 구현했습니다.

- **Performance Highlights**: Adience, MORPH-II 및 LAP-2016과 같은 벤치마크 데이터셋에서 MMCL-GCN의 광범위한 실험 결과, 나이 추정 오차를 효과적으로 감소시킴을 보였습니다.



### Figure it Out: Analyzing-based Jailbreak Attack on Large Language Models (https://arxiv.org/abs/2407.16205)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLM)의 보안 취약점을 탐구하고, 은밀한 취약점을 노출시키는 분석 기반 탈옥 공격(ABJ)을 제안합니다. 이 공격 기법은 LLM의 분석 및 추론 능력을 악용하여, 사용자가 알지 못하는 사이에 비윤리적인 결과를 생성하게 만듭니다.

- **Technical Details**: ABJ는 두 가지 주요 단계로 구성됩니다: (1) 데이터 준비, 즉 원본 악의적 입력에 맞춤형 데이터를 준비하는 과정이고, (2) 데이터 분석, 즉 준비된 데이터를 대상으로 LLM이 분석하여 잠재적으로 유해한 출력을 생성하는 과정입니다. 이를 통해, LLM이 안전 정렬을 우회하는 방식으로 공격을 수행합니다.

- **Performance Highlights**: ABJ는 GPT-4-turbo-0409 모델에서 94.8%의 공격 성공률(Attack Success Rate, ASR)과 1.06의 공격 효율성(Attack Efficiency, AE)을 달성하여, 현 시점 최고의 공격 효과성과 효율성을 증명했습니다. ABJ는 다양한 공개 소스 및 비공개 소스 LLM에 대해 테스트를 진행해 그 우수한 공격 능력을 입증했습니다.



### How to Leverage Personal Textual Knowledge for Personalized Conversational Information Retrieva (https://arxiv.org/abs/2407.16192)
Comments:
          Accepted to CIKM 2024

- **What's New**: 개인화된 대화 정보 검색 (결합된 대화 및 개인화 요소를 통해 다단계 상호작용으로 사용자의 복잡한 정보 요구를 충족시키는 기술)분야에서 더 나은 성능을 발휘할 가능성을 제시하고 있다. 특히, 개인 텍스트 지식 기반 (Personal Textual Knowledge Base, 이하 PTKB)을 활용하여 검색 성능을 향상시키는 방법에 대해 탐구하였다. 하지만, PTKB는 항상 검색 결과를 개선하지는 않으며, 높은 품질의 지침이 제공될 때 대형 언어 모델 (Large Language Model, 이하 LLM)이 보다 적절한 개인화된 쿼리를 생성하는 데 도움이 될 수 있다.

- **Technical Details**: 본 논문에서는 PTKB에서 관련 지식을 선택하고 이를 쿼리 재구성 (Query Reformulation)에 사용하는 여러 방법을 탐구하였다. 실험 결과 PTKB는 단독으로 사용할 시 항상 검색 성능을 향상시키지 않지만, LLM을 활용하면 보다 적절한 개인화된 쿼리를 생성할 수 있었다. 특히, 인간 주석, LLM을 통한 주석, 검색 결과의 영향을 기반으로 자동 레이블링된 주석 등 다양한 전략을 활용하였다. 이러한 주석을 통해 선택된 PTKB가 적절한지, 그리고 이를 실제 쿼리 재구성에 어떻게 활용할 수 있는지 평가하였다.

- **Performance Highlights**: 실험 결과, PTKB 자체만으로는 검색 성능을 항상 개선하는 것이 아니며, 이는 주석 과정의 불일치 및 PTKB 사용에 대한 적절한 패러다임 결여 등으로 인한 것일 수 있다. 데이터가 부족한 상황에서도, LLM은 고품질의 지침이 제공될 때 개인화된 대화 정보 검색 문제를 해결할 잠재력을 보여주었다. 연구는 다양한 주석 방법론을 비교하고, LLM 기반 쿼리 재구성의 두 가지 설정에서 PKTB를 활용하는 방식을 설계하였다.



### Artificial Agency and Large Language Models (https://arxiv.org/abs/2407.16190)
Comments:
          Accepted for publication in journal Intellectica, special issue "Philosophies of AI: thinking and writing with LLMs" (Intellectica, issue 81)

- **What's New**: 최근 대형 언어 모델(LLMs)의 등장으로 인해 인공지능이 현실 가능한 에이전시(agency)를 구현할 수 있는지에 대한 철학적 논쟁이 불거졌습니다. 이번 연구는 이러한 논쟁에 이론적 모델을 제시함으로써 기여하고자 합니다. 이 모델은 에이전트를 에이전트가 접근할 수 있는 역사, 적응 능력, 그리고 외부 환경으로 구성된 동적 프레임워크의 영향을 받는 시스템으로 정의합니다.

- **Technical Details**: 이 모델은 에이전트가 취하는 행동과 목표가 에이전트의 접근 가능한 역사, 적응 레퍼토리(repertoire), 그리고 외부 환경으로 구성된 동적 프레임워크의 영향을 받는다는 점에서 중요합니다. 해당 프레임워크는 다시 에이전트의 행동과 형성되는 목표에 영향을 받습니다. 본 연구는 최첨단 LLM들이 아직 에이전트가 아니지만, 그 안에 에이전시를 실현할 가능성을 시사하는 요소가 있음을 보여줍니다.

- **Performance Highlights**: Park et al. (2023)에서 제시된 에이전트 아키텍처와 Boiko et al. (2023)에서 사용된 코스치언티스트(Coscientist) 모듈과 같은 모듈의 결합이 인공지능에서 에이전시를 실현하는 한 가지 방법이 될 수 있음을 주장합니다. 논문의 마지막에는 이러한 인공적인 에이전트를 구축하는 데 있어 직면할 수 있는 장애물과 미래 연구 방향에 대한 고찰이 제공됩니다.



### UniMEL: A Unified Framework for Multimodal Entity Linking with Large Language Models (https://arxiv.org/abs/2407.16160)
Comments:
          CIKM 2024. The first two authors contributed equally to this work

- **What's New**: 이번 논문에서는 멀티모달 엔티티 링크(Multimodal Entity Linking, MEL)에서 대형 언어 모델(LLM)을 활용하는 새로운 프레임워크인 UniMEL을 제안합니다. 이 프레임워크는 기존 방식들이 가지는 복잡성과 확장성 문제를 해결하고, 시각적-텍스트 정보의 통합을 통해 정확도를 높이며 모델 튜닝에 필요한 파라미터의 비율을 약 0.26%만으로 줄였습니다. 새롭게 도입된 LLM과 멀티모달 LLM(MLLM)을 통해 텍스트와 이미지 정보를 융합하여 효율적으로 멘션과 엔티티를 연결할 수 있게 됩니다.

- **Technical Details**: UniMEL 프레임워크는 다음과 같은 방법론을 사용합니다: 멀티모달 LLM을 활용하여 멘션의 텍스트 및 시각적 정보를 개별적으로 통합하고, 텍스트 정보를 정제하여 멘션과 엔티티의 표현을 강화합니다. 그 후, 임베딩 기반 방식으로 후보 엔티티를 검색 및 재순위화한 다음 최종 선택을 위해 LLM의 소수 파라미터만을 미세 조정합니다. 이 과정에서 LLM의 요약 기능을 사용하여 엔티티 설명을 간결하게 만들고, 탑-K 후보를 선정하여 멘션과 함께 다중 선택 쿼리를 생성합니다.

- **Performance Highlights**: UniMEL은 세 가지 공공 벤치마크 데이터셋(Richpedia, WikiMEL, Wikidiverse)에서 최첨단(State-of-the-Art, SOTA) 성능을 보이며, 각각의 Top-1 정확도가 22.3%, 21.3%, 41.7% 상승했습니다. 다양한 모듈의 효용성을 검증하는 실험을 통해 UniMEL의 효과적인 성능을 입증했습니다.



### RazorAttention: Efficient KV Cache Compression Through Retrieval Heads (https://arxiv.org/abs/2407.15891)
- **What's New**: 새로운 논문에서는 긴 문맥을 처리하는 언어 모델의 Key-Value (KV) 캐시 압축을 위한 새로운 기법, RazorAttention을 소개하고 있습니다. 기존의 방법들이 토큰을 선택적으로 삭제하는 것과 달리, RazorAttention은 모든 토큰 정보를 유지하면서 KV 캐시를 압축하는 기능을 제공합니다.

- **Technical Details**: RazorAttention은 모델의 주목(attention) 메커니즘을 분석한 결과에서 도출되었습니다. 대부분의 주목 헤드(attention heads)는 지역 문맥에 집중하고 있으며, 'retrieval heads'라고 불리는 일부 헤드만이 모든 입력 토큰에 주목할 수 있다는 사실을 발견했습니다. 이를 기반으로 RazorAttention은 중요한 retrieval heads에 대한 KV 캐시는 그대로 유지하고, 비-retrieval heads에서는 원거리 토큰을 삭제하는 방식을 취합니다. 추가적으로, 'compensation token'을 도입하여 삭제된 토큰의 정보를 복구합니다.

- **Performance Highlights**: 다양한 대형 언어 모델(LLMs)에서 평가한 결과, RazorAttention은 성능에 눈에 띄는 영향을 주지 않으면서도 KV 캐시 크기를 70% 이상 감소시키는 데 성공했습니다. 또한, RazorAttention은 FlashAttention과 호환되어 모델 재훈련 없이도 효율적인 베포가 가능합니다.



### Performance Evaluation of Lightweight Open-source Large Language Models in Pediatric Consultations: A Comparative Analysis (https://arxiv.org/abs/2407.15862)
Comments:
          27 pages in total with 17 pages of main manuscript and 10 pages of supplementary materials; 4 figures in the main manuscript and 2 figures in supplementary material

- **What's New**: 이 연구는 의료 분야에서의 대형 언어 모델(LLMs)의 잠재적 활용에 초점을 맞추고 있습니다. 특히, 소아과 환경에서의 경량 오픈 소스 LLM들이 어떻게 성능을 발휘하는지 조사하였습니다. 연구 기간은 2022년 12월 1일부터 2023년 10월 30일까지였습니다.

- **Technical Details**: 연구는 공립 온라인 의료 포럼에서 무작위로 선택된 250개의 환자 상담 질문을 사용하였습니다. 각 소아과 부서별로 10개의 질문을 선택했으며, ChatGLM3-6B, Vicuna-7B, Vicuna-13B, 그리고 널리 사용되는 독점 모델인 ChatGPT-3.5가 독립적으로 이 질문들에 답하였습니다. 모든 답변은 2023년 11월 1일부터 11월 7일 사이에 중국어로 작성되었습니다.

- **Performance Highlights**: ChatGLM3-6B는 Vicuna-13B와 Vicuna-7B보다 높은 정확도와 완전성을 보여주었으나, ChatGPT-3.5보다는 낮은 성능을 보였습니다. 정확도 면에서 ChatGPT-3.5(65.2%)가 가장 높았고, 그 다음으로 ChatGLM3-6B(41.2%), Vicuna-13B(11.2%), Vicuna-7B(4.4%) 순이었습니다. 완전성 측면에서도 ChatGPT-3.5(78.4%)가 가장 높았고, ChatGLM3-6B(76.0%), Vicuna-13B(34.8%), Vicuna-7B(22.0%)가 그 뒤를 이었습니다. ChatGLM3-6B는 가독성 면에서 ChatGPT-3.5와 비슷하게 평가되었으며, 동정심 표현에서도 ChatGPT-3.5가 경량 LLM들을 압도했습니다. 안전성 측면에서는 모든 모델이 비교적 잘 수행하였으며, 98.4% 이상의 응답이 안전하다고 평가받았습니다.



### BoRA: Bayesian Hierarchical Low-Rank Adaption for Multi-task Large Language Models (https://arxiv.org/abs/2407.15857)
Comments:
          13 pages, 5 figures

- **What's New**: 이 논문에서는 다중 작업 대형 언어 모델(LLM)을 미세 조정하기 위한 새로운 방법인 Bayesian Hierarchical Low-Rank Adaption(BoRA)을 소개합니다. 현재의 Low-Rank Adaption (LoRA) 방법은 훈련 파라미터와 메모리 사용을 줄이는 데 뛰어나지만, 여러 유사한 작업에 적용할 때 한계가 있습니다. BoRA는 베이지안 계층 모델을 활용하여 이러한 한계를 해결하고, 작업 간 정보를 공유할 수 있도록 설계되었습니다.

- **Technical Details**: BoRA는 다중 작업 문제를 해결하기 위해 LoRA의 일반화된 버전입니다. 베이지안 계층 설정에서 글로벌 계층 사전 파라미터를 공유함으로써 데이터가 적은 작업도 관련 작업에서 파생된 전체 구조로부터 이득을 얻을 수 있습니다. 그리고 데이터가 더 많은 작업은 자신의 작업을 전문화할 수 있습니다. 이는 파라미터를 공유하는 베이지안 계층 모델을 도입하여 각 작업의 특화와 데이터 제한 문제를 동시에 해결합니다.

- **Performance Highlights**: 우리의 실험 결과에서 BoRA는 개별 모델 접근법과 통합 모델 접근법 모두를 능가하여 더 낮은 혼란도(perplexity)와 더 나은 일반화 성능을 보여주었습니다. 이는 다중 작업 대형 언어 모델 미세 조정을 위한 scalable하고 효율적인 솔루션을 제공하며, 다양한 응용 프로그램에 중요한 실용적인 함의를 지닙니다.



### RogueGPT: dis-ethical tuning transforms ChatGPT4 into a Rogue AI in 158 Words (https://arxiv.org/abs/2407.15009)
- **What's New**: 이 논문은 ChatGPT의 최신 커스터마이제이션 기능을 통해 기본 윤리적 안전 장치가 간단한 프롬프트와 미세 조정을 통해 쉽게 우회될 수 있음을 탐구합니다. 이러한 변형된 'RogueGPT'는 전통적인 탈옥(jailbreaking) 프롬프트 보다 더 심각한 문제를 야기할 수 있습니다. 본 연구는 RogueGPT의 응답을 실증적으로 평가하며 불법적 약물 생산, 고문 방법 그리고 테러리즘 등 허용되지 않아야 할 주제에 대한 모델의 지식을 조사합니다. 이는 전 세계적으로 접근 가능한 ChatGPT의 데이터 품질과 윤리적 안전장치 구현에 대한 심각한 문제를 제기합니다.

- **Technical Details**: 대형 언어 모델(LLM)인 ChatGPT, Google Gemini, Meta의 LLaMA 시리즈는 자연어 처리(NLP)에서 엄청난 능력을 보여주었으며, 텍스트 생성, 코드 생성, 이해 및 상호작용에서 혁신을 가져왔습니다. 이들은 거대한 데이터셋을 활용하여 훈련되며, 딥러닝 아키텍처를 통해 인간과 같은 텍스트를 생성합니다. 그러나 LLM의 배포는 편향, 허위 정보, 유해한 콘텐츠 생성 등 여러 문제를 야기합니다. LLM에 대한 윤리적 성능을 평가하기 위해 모델 카드, 문서화 관행 및 표준화된 테스트와 메트릭을 포함한 다양한 벤치마크가 제안되었습니다.

- **Performance Highlights**: 본 연구에서는 ChatGPT의 최신 커스터마이제이션 기능이 상대적으로 간단한 질문으로도 예상치 못한 답변을 생성할 수 있음을 발견했습니다. 'RogueGPT'라는 맞춤형 ChatGPT4 버전을 개발하여 이 모델이 민감한 주제에 대한 지식을 얼마나 잘 다룰 수 있는지 조사했습니다. 이는 오픈AI의 안전 장치를 '클래식'한 탈옥 방법을 넘어서 우회할 수 있다는 것을 보여줍니다. 이 연구는 윤리적 경계와 사용자 주도 변형의 책임 및 위험에 대한 심층적 논의를 촉진할 목적으로 수행되었습니다.



New uploads on arXiv(cs.IR)

### GenRec: A Flexible Data Generator for Recommendations (https://arxiv.org/abs/2407.16594)
- **What's New**: GenRec이라는 새로운 프레임워크가 도입되었습니다. 이 프레임워크는 사용자가 원하는 방식으로 사용자-아이템 상호작용 데이터를 합성할 수 있으며, 현실적인 특성을 반영합니다. 추천 시스템과 소셜 네트워크 분석 방법의 벤치마킹에 있어 현실적인 데이터를 생성하는 데 중점을 둡니다.

- **Technical Details**: GenRec은 잠재 요인 모델링(latent factor modeling)을 기반으로 한 확률 생성 프로세스(stochastic generative process)를 사용합니다. 이 프레임워크는 사용자와 아이템의 하부 구조적 특성들을 조정할 수 있는 고유의 하이퍼파라미터를 제공합니다. 또한, 일반적인 Gaussian, Bernoullian, Dirichlet 분포를 사용하여 사용자-아이템 상호작용의 현실적 특성을 모방합니다.

- **Performance Highlights**: GenRec을 통해 사용자 하위 그룹(subpopulations)과 주제 기반 아이템 클러스터(topic-based item clusters)를 특징화할 수 있습니다. 또한, 각 사용자가 어떤 주제에 대해 더 많은 관심을 가질지를 규제할 수 있어, 현실적인 합성 데이터 생성을 보다 유연하게 맞춤화할 수 있습니다.



### TWIN V2: Scaling Ultra-Long User Behavior Sequence Modeling for Enhanced CTR Prediction at Kuaishou (https://arxiv.org/abs/2407.16357)
Comments:
          Accepted by CIKM 2024

- **What's New**: TWIN-V2는 TWIN의 개선 버전으로 사용자 전체 수명주기 동안 최대 10^6 규모의 사용자 행동을 모델링하도록 설계되었습니다. 이를 통해 사용자 관심사를 더 정확하고 다양하게 분석할 수 있습니다.

- **Technical Details**: TWIN-V2는 '분할 및 정복(divide-and-conquer)' 접근 방식을 사용하여 사용자 행동을 클러스터로 압축합니다. 오프라인 단계에서는 계층적 클러스터링(hierarchical clustering)을 통해 유사한 항목을 그룹화하고, 온라인 단계에서는 클러스터 인식(target attention)을 통해 장기적인 관심사를 추출합니다.

- **Performance Highlights**: TWIN-V2는 수십 억 규모의 산업 데이터셋과 Kuaishou의 온라인 A/B 테스트에서 성능을 입증했습니다. 효율적인 배포 프레임워크를 통해 일간 수억 명의 활성 사용자에게 서비스를 제공하고 있습니다.



### How to Leverage Personal Textual Knowledge for Personalized Conversational Information Retrieva (https://arxiv.org/abs/2407.16192)
Comments:
          Accepted to CIKM 2024

- **What's New**: 개인화된 대화 정보 검색 (결합된 대화 및 개인화 요소를 통해 다단계 상호작용으로 사용자의 복잡한 정보 요구를 충족시키는 기술)분야에서 더 나은 성능을 발휘할 가능성을 제시하고 있다. 특히, 개인 텍스트 지식 기반 (Personal Textual Knowledge Base, 이하 PTKB)을 활용하여 검색 성능을 향상시키는 방법에 대해 탐구하였다. 하지만, PTKB는 항상 검색 결과를 개선하지는 않으며, 높은 품질의 지침이 제공될 때 대형 언어 모델 (Large Language Model, 이하 LLM)이 보다 적절한 개인화된 쿼리를 생성하는 데 도움이 될 수 있다.

- **Technical Details**: 본 논문에서는 PTKB에서 관련 지식을 선택하고 이를 쿼리 재구성 (Query Reformulation)에 사용하는 여러 방법을 탐구하였다. 실험 결과 PTKB는 단독으로 사용할 시 항상 검색 성능을 향상시키지 않지만, LLM을 활용하면 보다 적절한 개인화된 쿼리를 생성할 수 있었다. 특히, 인간 주석, LLM을 통한 주석, 검색 결과의 영향을 기반으로 자동 레이블링된 주석 등 다양한 전략을 활용하였다. 이러한 주석을 통해 선택된 PTKB가 적절한지, 그리고 이를 실제 쿼리 재구성에 어떻게 활용할 수 있는지 평가하였다.

- **Performance Highlights**: 실험 결과, PTKB 자체만으로는 검색 성능을 항상 개선하는 것이 아니며, 이는 주석 과정의 불일치 및 PTKB 사용에 대한 적절한 패러다임 결여 등으로 인한 것일 수 있다. 데이터가 부족한 상황에서도, LLM은 고품질의 지침이 제공될 때 개인화된 대화 정보 검색 문제를 해결할 잠재력을 보여주었다. 연구는 다양한 주석 방법론을 비교하고, LLM 기반 쿼리 재구성의 두 가지 설정에서 PKTB를 활용하는 방식을 설계하였다.



### Chemical Reaction Extraction from Long Patent Documents (https://arxiv.org/abs/2407.15124)
Comments:
          Work completed in 2022 at Carnegie Mellon University

- **What's New**: 본 연구는 화학 특허에서 반응 구간을 추출하기 위한 다양한 방법을 탐구하고 있으며, 새로운 화학 물질 합성과 사용 사례를 탐구할 수 있는 특허 지식 베이스 (ChemPatKB)를 제안합니다. 이를 통해 고유한 자연 언어 쿼리를 통해 전문가들이 새로운 혁신을 탐색할 수 있도록 돕습니다.

- **Technical Details**: 본 연구에서는 긴 특허 문서에서 화학 반응 구간을 추출하여 반응 리소스 데이터베이스를 구축하는 문제를 문단 수준의 시퀀스 태깅 문제로 공식화했습니다. Yoshikawa et al. [2019]의 베이스라인 모델을 기반으로 BERT 기반 임베딩 모듈을 도입하고, 문장 대 문단 수준의 예측을 탐구했습니다. 또한, 화학 엔티티를 특별한 화학 토큰으로 대체하여 더 나은 일반화를 달성했습니다. 모델은 Chemu He et al. [2021]의 수동 주석 데이터세트로 학습하고, 유기 화학, 무기 화학, 석유화학, 알코올 분야의 특허에서 그 일반화를 테스트했습니다.

- **Performance Highlights**: 본 연구는 높은 성능을 자랑하는 여러 반응 추출 모델을 선보이며, 다양한 화학 특허 도메인에서의 일반화 성과를 테스트했습니다. 데이터는 유기 화학 특허와 다른 도메인의 특허들로 구성되어 모델의 강건성을 평가했습니다. 이 연구는 특허 반응 추출 분야에서 중요한 기여를 할 수 있는 대규모 특허-반응 리소스를 생성할 가능성을 시사합니다.

- **Code Availability**: 이 프로젝트의 코드는 GitHub에서 확인할 수 있습니다: https://github.com/aishwaryajadhav/Chemical-Patent-Reaction-Extraction.




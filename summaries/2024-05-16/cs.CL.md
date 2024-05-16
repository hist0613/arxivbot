### Refinement of an Epilepsy Dictionary through Human Annotation of Health-related posts on Instagram (https://arxiv.org/abs/2405.08784)
- **What's New**: 이번 연구는 DrugBank, MedDRA, MedlinePlus, TCMGeneDIT 등의 다양한 출처에서 추출한 생의학 용어로 구성된 사전을 사용하여, 2010년부터 2016년 초까지 8백만 개 이상의 인스타그램 게시물을 태깅(tagging)하는 것을 목표로 했습니다. 이번 연구를 통해, 인간 주석자(annotator)와 OpenAI의 GPT 시리즈 모델들을 비교하여, 높은 오탐률(false-positive rate)의 빈번한 용어들을 제거한 개선된 사전이 만들어졌습니다.

- **Technical Details**: DrugBank, MedDRA, MedlinePlus, TCMGeneDIT 등의 기존 의료 온톨로지 및 데이터 소스에서 용어들을 수집하였고, 이 용어들을 기반으로 176,278개의 용어로 구성된 사전을 만들었습니다. 또한 수동 주석 작업을 통해 높은 빈도의 오탐 용어를 제거하여 사전을 개선했습니다. 이를 통해 원래 사전과 개선된 사전을 사용해 지식 네트워크를 구성하고, eigenvector-centrality 분석을 수행하여 중요한 용어들의 순위를 비교했습니다.

- **Performance Highlights**: 개선된 사전은 원래 사전에 비해 지식 네트워크에서 중요한 용어들의 순위에 상당한 차이를 발생시켰고, 개선 후의 중요한 용어들은 의료적으로 더 큰 관련성을 가지고 있음을 보여줍니다. 또한 OpenAI의 GPT 시리즈 모델은 이 작업에서 인간 주석자보다 성능이 떨어졌습니다.



### Is the Pope Catholic? Yes, the Pope is Catholic. Generative Evaluation of Intent Resolution in LLMs (https://arxiv.org/abs/2405.08760)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)이 비문자적인 발화(non-literal utterances)에 대해 의도를 이해하고 반응하는 능력을 평가하는 새로운 생성적 평가(generative evaluation) 프레임워크를 소개합니다. 이 연구는 기존의 판별적 평가(discriminative evaluations)에서 벗어나, LLMs가 비문자적 언어의 진정한 의도에 맞추어 반응할 수 있는지를 분석합니다.

- **Technical Details**: 이 프레임워크는 화자 간의 문맥(Context), 비문자적 발화(Non-literal Utterance), 실제 의도(True Intention), 잘못된 문자적 해석(Incorrect Literal Intention), 그리고 이들 각각에 대한 참조 대화 체인(reference dialog chains)을 기반으로 LLMs의 반응을 평가합니다. 모델이 실제 의도에 따라 협력적인 반응(cooperative response)을 생성할 수 있다면, 해당 반응은 문자적 해석 기반 반응보다 더 높은 유사성을 가져야 합니다.

- **Performance Highlights**: LLMs는 비문자적 언어에 대해 적절한 문맥적 반응을 생성하는 데 어려움을 겪으며, 평균적으로 50-55%의 정확도를 나타냈습니다. 오라클 의도(oracle intentions)를 명시적으로 제공했을 때 성능은 상당히 향상되었지만, 여전히 적절한 반응을 생성하는 데에는 한계가 있었습니다. 체인 오브 생각(chain-of-thought) 기법을 사용한 것도 약간의 성능 향상(60%)을 보였지만, 효과는 미미했습니다. 이 연구는 현재의 LLMs가 효과적인 실용적 대화자(pragmatic interlocutor)가 되기 위해서는 아직 갈 길이 멀다는 것을 강조합니다.



### Targeted Augmentation for Low-Resource Event Extraction (https://arxiv.org/abs/2405.08729)
Comments:
          15 pages, NAACL 2024

- **What's New**: 이 논문은 저자들이 목표로 하는 증강과 후속 검증(back validation)을 사용하여 더 높은 다양성, 유효성, 정확도 및 일관성을 가진 증강 예시를 생성하는 새로운 패러다임을 도입한 것입니다. 이 방법은 저자들이 저자원의 정보 추출 문제를 해결하는데 유용성을 검증한 여러 실험을 통해 입증되었습니다.

- **Technical Details**: 이 논문에서는 특별히 언급된 이벤트 구조 내에서 타겟 서브셋에서 엔티티를 추출하여 이벤트 구조를 풍성하게 만드는 목표 증강(targeted augmentation) 방법을 제안합니다. 또한 후속 검증(back validation) 모듈을 도입하여 생성된 내용의 정확성과 일관성을 보장합니다. 다양성, 극성, 정확도 및 일관성을 고려하여 데이터를 증강하며, 실제 발생한 이벤트 언급뿐 아니라 부정적 이벤트 언급(예: 가정적 이벤트 언급)도 포함하여 표현력을 높입니다.

- **Performance Highlights**: 광범위한 저자원 학습 시나리오에서 다수의 이벤트 추출 모델을 통해 종합적인 실험을 실시했으며, 그 결과 목표 증강의 효과가 뚜렷하게 나타났습니다. 특히, 논문에서 제안한 방법이 기존의 약한 증강(예: 유의어 증강)이나 급진적 증강(예: 안내 없이 조건부 생성)보다 더 좋은 성과를 보였음을 확인하였습니다.



### Thinking Tokens for Language Modeling (https://arxiv.org/abs/2405.08644)
Comments:
          AITP 2023 (May 10, 2023)

- **What's New**: 이 논문은 복잡한 계산 문제를 해결하는 데 어려움을 겪는 언어 모델의 일반화 능력을 향상시키기 위한 새로운 접근법을 제안합니다. 제안된 방법은 '생각 토큰'(thinking tokens)을 도입하여, 복잡한 문제에 직면했을 때 모델이 더 많은 계산을 수행할 수 있게 합니다.

- **Technical Details**: 제안된 '생각 토큰' 접근법은 문장 내 각 단어 후에 특수 토큰(<T>예상<T>)을 추가하는 것입니다. 이는 모델이 즉각적인 답변을 요구받지 않고 추가 계산을 수행할 시간을 확보할 수 있게 하여, 복잡한 문제를 더 잘 해결할 수 있도록 합니다. 이러한 개념은 RNN(recurrent neural networks) 아키텍처에 특히 효과적입니다.

- **Performance Highlights**: 초기 실험 결과, '생각 토큰'을 사용한 모델이 더 복잡한 추론을 필요로 하는 문장에서 퍼플렉시티(perplexity)가 가장 크게 개선되는 것을 확인했습니다. 이는 수학 데이터셋 예제 및 대표적인 숫자나 수치 기호가 포함된 문장에서 특히 두드러집니다.



### ALMol: Aligned Language-Molecule Translation LLMs through Offline Preference Contrastive Optimisation (https://arxiv.org/abs/2405.08619)
- **What's New**: 화학 분야와 인공지능(AI)의 교차점에 있는 연구는 과학적 발견의 가속화를 목표로 활발히 수행되고 있습니다. 본 연구에서는 대형 언어 모델(LLMs)을 사용한 머신 언어-분자 번역(machine language-molecule translation)에 새로운 교육 방법인 대조 선호 최적화(Contrastive Preference Optimisation, CTO)를 도입했습니다. 이는 불완전한 번역의 생성을 피하고, 기존 모델 대비 최대 32%의 성능 향상을 달성합니다.

- **Technical Details**: CTO는 인간의 피드백을 통한 강화 학습(Reinforcement Learning with Human Feedback, RLHF) 방법을 기반으로 합니다. RLHF는 일반적으로 선호되는 출력과 비선호되는 출력 쌍을 사용하여 보상 모델을 훈련합니다. 이와 유사하게 CTO는 오프라인 선호 데이터를 사용하여 모델을 최적화하며, 특히 화학 및 언어 모델의 통합 과정에서 이를 활용합니다. 또한, L+M-24 데이터셋의 10%만을 사용하여 모델의 일반화 가능성을 높였습니다.

- **Performance Highlights**: 본 연구에서 제안한 모델은 다양한 평가 메트릭에서 기존 모델 대비 최대 32%의 성능 향상을 나타냈으며, 학습 데이터가 적음에도 불구하고 강력한 일반화 능력을 보였습니다. 또한, 세밀한 평가 방법론을 도입하여 책임성을 갖춘 평가를 수행했습니다.



### A Comprehensive Survey of Large Language Models and Multimodal Large Language Models in Medicin (https://arxiv.org/abs/2405.08603)
- **What's New**: 최근 ChatGPT와 GPT-4 출시 이후, 대형 언어 모델(LLMs)과 멀티모달 대형 언어 모델(MLLMs)은 인공지능과 의학의 통합에 새로운 패러다임을 제공하면서 큰 주목을 받고 있습니다. 이 설문조사는 이러한 LLMs 및 MLLMs의 개발 배경과 원칙, 그리고 의료 분야에서의 응용 시나리오, 도전 과제 및 미래 방향을 탐구합니다.

- **Technical Details**: Transformer의 도입 이후 자연어 처리(NLP)와 컴퓨터 비전(CV) 분야에서 패러다임 전환이 일어났습니다. LLMs와 MLLMs는 강력한 병렬 컴퓨팅 능력과 자기 주의 메커니즘 덕분에 다양한 다운스트림 작업에서 최고 성과를 기록하고 있습니다. 본 설문조사는 주로 시각-언어 모달리티에 초점을 맞추며, 주요 아키텍처와 현재 존재하는 의료 관련 데이터셋을 요약합니다. 또한, LLMs와 MLLMs의 구축, 평가 및 활용 과정 전체를 명확하고 논리적으로 설명합니다.

- **Performance Highlights**: 특히, Google의 Med-PaLM 2는 미국 의사 면허 시험(USMLE)에서 86.5점을 기록하며, 의료 전문가 수준에 도달했습니다. 더 나아가 ChatDoctor, LLaVA-Med, XrayGLM 등 다양한 의료 LLMs와 MLLMs가 등장하여 의료 보고서 생성, 임상 진단, 정신 건강 서비스 등 다양한 임상 응용 분야에 잠재적 해결책을 제공합니다.

- **Challenges**: 의료 LLMs와 MLLMs는 여전히 많은 데이터와 컴퓨팅 자원을 필요로 하며, 데이터 프라이버시 문제와 높은 비용을 동반합니다. 또한, 이 모델들은 상호 작용적 생성 모델로서의 특성과 더불어 의학적 전문성, 안전성 및 윤리성을 고려해야 하며, 이를 위해 추가적인 훈련 전략이 필요합니다.

- **Future Directions**: 이 설문조사는 LLMs와 MLLMs의 임상 적용에서 중요한 잠재적 응용 분야를 요약하고, 현재의 한계와 가능한 솔루션을 분석합니다. 이를 통해 인공지능과 의학의 통합을 가속화하고자 합니다.



### Rethinking the adaptive relationship between Encoder Layers and Decoder Layers (https://arxiv.org/abs/2405.08570)
- **What's New**: 최신 논문은 SOTA(SOTA, State-Of-The-Art) 모델인 Helsinki-NLP/opus-mt-de-en을 사용하여 독일어에서 영어로 번역하는 과정을 탐구합니다. 이 연구는 Encoder Layer와 Decoder Layer 사이에 바이어스가 없는 완전 연결 레이어(bias-free fully connected layer)를 도입하여 시스템 구조를 수정한 새로운 방법을 제시합니다. 이 방법을 통해 고차원적 상호작용을 모색하고, 미세 조정(fine-tuning)과 재학습(retraining)의 결과를 비교 분석합니다.

- **Technical Details**: 연구는 네 가지 실험을 통해 Encoder와 Decoder 사이의 적응 관계를 탐구했습니다. 첫 번째 실험은 원래의 Pre-trained 모델 가중치를 사용하고, 완전 연결 레이어의 가중치를 초기화하여 원래 구조를 유지하는 것입니다. 두 번째 실험은 원래의 Pre-trained 모델을 미세 조정하는 것입니다. 세 번째 실험은 Granularity Consistent Attention(GCA)로 초기화하는 것이며, 네 번째 실험은 가중치를 평균 1, 분산 0으로 초기화하여 원래의 연결을 유지하는 것입니다. 실험 결과, Pre-trained 모델 구조를 직접 수정하는 것은 성능 저하를 초래할 수 있음을 발견했습니다. 그러나 적절한 구조 수정은 재학습 실험에서 성능 향상과 안정성을 보여줬습니다.

- **Performance Highlights**: 미세 조정을 통해 원래 구조를 유지한 상태에서의 학습 손실(training loss)은 큰 변동을 보였고, 1 epoch 내에 수렴하지 못했습니다. 반면, GCA 초기화를 통해 미세 조정한 경우, 모델의 구조적 복잡성으로 인해 고차원적인 학습 데이터가 필요했음을 나타냈습니다. 제한된 학습 데이터와 1 epoch 학습이라는 한계 내에서도 Encoder Layer와 Decoder Layer 사이의 긍정적인 영향을 확인할 수 있었습니다.



### The Unseen Targets of Hate -- A Systematic Review of Hateful Communication Datasets (https://arxiv.org/abs/2405.08562)
Comments:
          20 pages, 14 figures

- **What's New**: 최근 10년간 자동화된 증오성 커뮤니케이션 감지 데이터셋에 대한 체계적인 리뷰를 통해, 데이터셋에 포함된 증오성 커뮤니케이션 대상의 다양성과 품질을 분석한 연구가 발표되었습니다. 연구는 증오성 커뮤니케이션 연구 공간의 지리적 경계가 넓어지고, 국제적 협력이 강화되는 긍정적인 경향을 강조하지만, 여전히 미국 기반 연구자들과 영어 데이터셋이 주류를 이루고 있음을 지적합니다.

- **Technical Details**: 연구는 증오성 커뮤니케이션 데이터셋의 생성자와 그들이 사용하는 관행을 중심으로 분석을 진행했습니다. 데이터셋의 질적 평가는 명시적으로 개념화된 타겟과 실제 데이터셋의 샘플링, 주석, 분석에서 운영화된 타겟 간의 불일치에 중점을 두었습니다. 또한, 연구는 최근 5년간 증오성 커뮤니케이션 연구가 지리적으로 넓어지고 다양한 언어와 플랫폼을 포함하고 있지만, 여전히 특정한 타겟(예: 나이, 신체 이미지)에 대한 커버리지가 부족함을 발견했습니다.

- **Performance Highlights**: 연구는 증오성 커뮤니케이션 데이터셋의 생산과 타겟의 다양성을 확장시키는 긍정적인 트렌드를 발견했습니다. 그러나, 개념화 및 운영화되지 않은 타겟 카테고리에 속하는 사례가 최대 16%에 달하여 이러한 타겟에 대한 증오성 분류기가 예측 불가능하게 작동할 가능성을 시사합니다. 연구는 데이터셋 품질을 보장하는 표준과 관행을 개발하기 위한 실질적인 단계를 제안합니다.



### Analysing Cross-Speaker Convergence in Face-to-Face Dialogue through the Lens of Automatically Detected Shared Linguistic Constructions (https://arxiv.org/abs/2405.08546)
Comments:
          Accepted for publication at the 46th Proceedings of the Annual Meeting of the Cognitive Science Society

- **What's New**: 이 연구는 자동으로 공유된 어휘 구조(shared lemmatised constructions)를 감지하는 방법을 제안하고 이를 참조 통신 코퍼스에 적용하여 이전에 알려지지 않은 객체에 대한 라벨링 관습의 형성을 조사합니다. 이 연구는 대화 참여자 간의 언어적 정렬(alignment)이 얼마나 빈번하게 발생하는지, 그리고 이것이 상호작용 후 라벨링 수렴(convergence)에 어떤 영향을 미치는지를 분석합니다.

- **Technical Details**: 본 연구는 66쌍의 네덜란드어 사용자 간의 대화를 분석한 것입니다. 참가자는 '감독자'(director)와 '일치자'(matcher) 역할을 번갈아가며 새로운 3D 객체('fribbles')를 식별하는 작업을 총 6라운드 수행했습니다. 이 작업에서 각 라운드의 대화 내용을 Python 라이브러리 spaCy를 사용해 어휘적 형태소(lemmas)로 변환하고, 순차 패턴 매칭 알고리즘을 통해 양측 대화자가 사용한 공통 어휘 구조를 추출했습니다.

- **Performance Highlights**: 분석 결과, 양측 대화자가 공통으로 사용한 어휘 구조의 빈도와 다양성이 객체 라벨링의 수렴도와 관련이 있음을 발견했습니다. 이번 연구는 자동으로 감지된 공유 어휘 구조가 대화 참여자 간의 참조 협상(reference negotiation) 역학을 조사하는 유용한 분석 단위를 제공하며, 이는 대규모 대화 코퍼스에서 수렴을 정량화하는 데 적합한 방법임을 보여주었습니다.



### Archimedes-AUEB at SemEval-2024 Task 5: LLM explains Civil Procedur (https://arxiv.org/abs/2405.08502)
Comments:
          To be published in SemEval-2024

- **What's New**: 이 논문은 SemEval-2024 대회에서 민사 절차에서의 논거 추론(Argument Reasoning)을 주제로 다루고 있습니다. 이 작업은 법률 개념을 이해하고 복잡한 논점을 추론하는 것을 필요로 합니다. 특히, 법적 영역에서 탁월한 성능을 발휘하는 대형 언어 모델(LLM)들이 대부분 분류 작업에 중점을 두기 때문에, 그들의 추론 합리성이 종종 논란의 대상이 됩니다. 저자들은 강력한 교사-LLM(예: ChatGPT)을 사용해 설명을 포함한 훈련 데이터셋을 확장하고, 합성 데이터를 생성하여 이를 이용해 작은 학생-LLM을 미세 조정하는 접근을 제안합니다.

- **Technical Details**: 제안된 접근 방식은 두 가지 주요 데이터 증강 전략(data augmentation strategies)을 사용합니다. 첫 번째는 체인 오브 쏘트(Chain-of-Thought, CoT) 스타일의 설명을 추가하는 것이고, 이는 저자의 원래 분석과 GPT-3.5의 도움을 받습니다. 두 번째는 기존 훈련 예제를 기반으로 합성 예제를 생성하는 데이터 변이(data mutation) 방법입니다. 이 예제들은 또한 설명과 함께 제공됩니다. 이렇게 생성된 데이터들은 Llame-2 모델을 미세 조정하는 데 사용됩니다.

- **Performance Highlights**: 제안된 시스템은 SemEval 대회에서 15위를 기록했으며, 자체 교사를 초과하는 성능을 보여주었습니다. 또한 생성된 설명이 원래 인간 분석과 일치하는지 법률 전문가들에 의해 검증되었습니다.



### Is Less More? Quality, Quantity and Context in Idiom Processing with Natural Language Models (https://arxiv.org/abs/2405.08497)
Comments:
          14 pages, 10 figures. Presented at the Joint Workshop on Multiword Expressions and Universal Dependencies (MWE-UD 2024) this https URL

- **What's New**: NCSSB 데이터셋이 등장했습니다. 이 데이터셋은 공개된 도서 텍스트에서 잠재적인 관용적 명사 합성어를 동의어로 교체해 만든 것입니다. 이를 통해 관용어 처리 모델의 데이터 양과 질 사이의 균형을 탐구합니다.

- **Technical Details**: 연구의 주요 초점은 관용어 및 문맥적 힌트를 이해하는 모델 개발에 필요한 데이터 양과 질의 중요성을 파악하는 것입니다. 이를 위해 Noun Compound Synonym Substitution in Books(NCSSB)라는 다양한 크기와 품질의 데이터셋을 생성했습니다. 이 데이터셋은 문맥 문장을 포함한 잠재적 관용 표현을 포함합니다.

- **Performance Highlights**: 개발된 모델은 SemEval 2022 Task 2 Subtask B 테스트에서 평가되었습니다. 결과는 데이터 품질이 문맥을 포함한 모델의 성능에 더 큰 영향을 미치는 반면, 문맥 포함 전략이 없는 모델에서는 데이터 양도 중요한 역할을 한다는 것을 보여줍니다.



### Enhancing Gender-Inclusive Machine Translation with Neomorphemes and Large Language Models (https://arxiv.org/abs/2405.08477)
Comments:
          Accepted at EAMT 2024

- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)을 활용하여 영어에서 이탈리아어로 번역할 때 성 중립적 신조어(neomorphemes)를 사용하는 방법을 탐구했습니다. 성 중립적 번역(gender-inclusive MT)을 평가할 수 있는 네오-게이트(Neo-GATE)라는 새로운 자원을 공개했습니다.

- **Technical Details**: 연구진은 네 가지 다른 LLM과 다양한 프롬프트(prompt) 형식을 사용하여 영어에서 이탈리아어로 성 중립적 번역을 수행했습니다. 이탈리아어의 성 모피(morphology) 시스템은 복잡하여 남성 및 여성 성별 표시를 성 중립적으로 대체하는 신조어(예: -a 또는 -o 대신 /-, -e/-es)를 사용했습니다. LLM의 인-컨텍스트 학습(in-context learning) 능력을 활용하였으며, 성 중립적 번역에서 신조어의 적용 가능성을 평가했습니다.

- **Performance Highlights**: 연구 결과, LLM의 성 중립적 번역 수행 능력은 모델의 크기와 프롬프트 형식에 따라 다르다는 점을 확인했습니다. 다양한 프롬프트 형식과 신조어 패러다임을 고려하여 각 모델의 강점과 약점을 분석했습니다.



### GPT-3.5 for Grammatical Error Correction (https://arxiv.org/abs/2405.08469)
- **What's New**: 이번 논문은 GPT-3.5를 다양한 언어의 문법 오류 수정(GEC)에 적용한 연구를 다루고 있습니다. 연구는 zero-shot GEC, GEC를 위해 fine-tuning, 다른 GEC 모델이 생성한 수정 가설을 재순위화 하는 등 여러 설정에서 진행되었습니다. 시범 언어로는 영어, 체코어, 독일어, 러시아어, 스페인어, 우크라이나어가 포함되었습니다.

- **Technical Details**: 연구는 OpenAI API를 통해 GPT-3.5-turbo 모델을 사용하여 수행되었으며, 자동 평가 방법으로는 언어 모델(LM)을 통한 문법성 추정, Scribendi 테스트 및 문장의 의미 임베딩(embedding) 비교가 사용되었습니다. GPT-3.5는 특히 zero-shot 설정에서 뛰어난 recall을 보였으나, 원문 문장과의 의미적 차이가 크다는 문제가 지적되었습니다.

- **Performance Highlights**: zero-shot 설정에서 GPT-3.5는 대부분의 테스트 언어에서 높은 recall을 달성하였으나, 수동 평가 결과 문장을 과도하게 수정하는 경향이 있음을 발견했습니다. fine-tuning의 경우, 수정되지 않은 문장이 많아 성능이 낮았습니다. 다른 GEC 모델이 생성한 수정 가설을 GPT-3.5가 재순위화했을 때는 성능이 향상되었습니다. GPT-3.5는 긴 문장과 단어 순서 반전 처리에는 강점을 보였으나, 영어의 경우 구두점, 명사, 전치사, 동사 시제 오류를, 러시아어의 경우 구두점, 단어 간 결속 및 의미적 호환성 오류를 다루는 데 어려움을 느꼈습니다.



### Challenges and Opportunities in Text Generation Explainability (https://arxiv.org/abs/2405.08468)
Comments:
          17 pages, 5 figures, xAI-2024 Conference, Main track

- **What's New**: NLP(자연어 처리) 분야에서 대규모 언어 모델의 증가와 함께 이들의 설명 가능성(explainability)에 대한 필요성이 커지고 있습니다. 특히, 텍스트 생성은 많은 관심을 받고 있는 주요 목표 중 하나입니다. 본 논문은 텍스트 생성 과정에서 발생하는 모델-불가지론적 설명 가능 인공지능(xAI) 방법의 설계 및 평가 중 마주치는 17가지 도전 과제를 세 그룹으로 분류하여 제시합니다. 이러한 과제는 토큰화, 설명 유사성 정의, 토큰 중요성 및 예측 변화 메트릭 측정, 인간 개입 수준, 적절한 테스트 데이터셋 생성에 관한 것입니다.

- **Technical Details**: 본 논문은 토큰화(tokenization), 설명 유사성(explanation similarity), 토큰 중요성(token importance), 예측 변화 메트릭(prediction change metrics) 등에 관련된 다양한 도전 과제를 다룹니다. 특히, 설명 가능 인공지능(xAI) 방법에 있어서 설명 마스크(explanatory masks)를 사용한 방식에 집중합니다. 이러한 마스크는 입력 문장의 개별 토큰에 대한 중요도 점수로 변환될 수 있어, 인간이 이러한 설명을 해석하는 데 큰 도움을 줍니다. 더 나아가, 모델 중심의 평가와 인간 중심의 평가를 결합하여 완전성을 보장하려고 합니다.

- **Performance Highlights**: 텍스트 생성 설명 가능 인공지능(xAI) 방법론의 성능을 평가하기 위해 인간 중심의 데이터셋 설계와 평가 방법을 제안합니다. 이를 통해 설명 방법이 주어진 데이터셋에서 얼마나 신뢰할 수 있고 이해할 수 있는지를 평가할 계획입니다. 또한, 확률적 단어 수준(probabilistic word-level)의 설명 가능 방법론을 개발하고, 다양한 이해 관계자가 각 단계에 참여하여 설명 가능성을 높일 것을 권장합니다.



### Evaluating LLMs at Evaluating Temporal Generalization (https://arxiv.org/abs/2405.08460)
Comments:
          Preprint

- **What's New**: 최근 대형 언어 모델(LLMs)의 평가 방법론에 대한 연구가 제기되었습니다. 기존 벤치마크는 정적이기 때문에 현실 세계의 변화하는 정보 환경을 제대로 반영하지 못한다는 점에서 발생하는 문제를 지적했습니다. 이를 해결하기 위해 새로운 평가 프레임워크 Freshbench를 제안하며 이는 최신 데이터를 동적으로 생성해 벤치마크로 사용할 수 있도록 합니다. 코드와 데이터셋은 곧 공개될 예정입니다.

- **Technical Details**: LLMs가 과거, 현재, 미래 맥락에 걸쳐 텍스트를 정확히 이해하고 예측하며 생성할 수 있는 Temporal Generalization 개념을 도입했습니다. 또, Time이라는 동적 차원을 통해 데이터 오염 문제를 줄이고 평가의 객관성을 높이기 위해 노력했습니다. 새로운 텍스트 소스 예를 들어 arxiv 논문, 뉴스 기사, 위키피디아 등을 활용하여 모델의 언어적 능력을 평가하고, 미래 사건 예측 시나리오에서 모델의 현 상황 이해 및 세상 지식을 통합해 미래 시나리오를 일반화할 수 있는 능력을 테스트했습니다.

- **Performance Highlights**: 새로운 텍스트에 대한 검증 손실을 이용해 모델을 평가하면 대체로 효과적이라는 결과가 나왔습니다. 그러나 언어 가능성에 대한 성능이 모델의 언어적 능력을 강하게 나타내주는 반면, 기존 벤치마크로 측정되는 특정 능력과는 항상 일치하지 않는다고 지적했습니다. Freshbench를 통해 LLM의 미래 예측 기능을 지속적으로 최신화된 데이터로 테스트할 수 있어 최신 데이터 환경에서도 평가가 정확하게 유지될 수 있습니다.



### How Alignment Helps Make the Most of Multimodal Data (https://arxiv.org/abs/2405.08454)
Comments:
          Working Paper

- **What's New**: 정치 커뮤니케이션 연구에서 텍스트, 오디오, 그리고 비디오 신호를 결합하는 것이 인간 소통의 풍부함을 보다 포괄적으로 반영한다는 새로운 시각이 제시되었습니다. 본 연구는 이러한 멀티모달(multi-modal) 데이터를 모델링할 때, 각 모달리티를 맞추는 것이 모델의 잠재력을 최대한 활용하는 데 필수적이라는 주장을 펼칩니다.

- **Technical Details**: 멀티모달 데이터는 이질성, 연결성, 상호작용이라는 세 가지 주요 과제가 있습니다. 첫째, 멀티모달 데이터는 서로 다른 센서로 동일한 현상을 묘사하기 때문에 이질적입니다. 예를 들어, 손뼉 소리는 이미지 데이터와 오디오 데이터에서 매우 다르게 나타날 수 있습니다. 둘째, 멀티모달 데이터는 연결되어 있으며, 다양한 데이터 스트림이 동일한 이벤트를 묘사합니다. 셋째, 모달리티 간의 상호작용은 특히 전문가나 포커스 그룹 인터뷰와 같은 상황에서 더욱 중요합니다. 모달리티 정렬(alignment)이 이러한 문제를 해결하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 독일 국회의원들이 극우 정당 AfD의 구성원들을 연설에서 어떻게 언급하는지 분석하고, 2020년 미국 대통령 선거에서 비디오 광고의 톤을 예측하여 이 방법론의 유용성을 보여줍니다. 멀티모달 데이터 분석에서 정렬된 데이터를 사용함으로써 모델의 예측 품질이 향상되었으며, 교차 모달 쿼리, 즉 한 모달리티를 질의하여 다른 모달리티에서 일어나는 일을 알아낼 수 있는 새로운 애플리케이션을 디자인할 수 있습니다.



### Impact of Stickers on Multimodal Chat Sentiment Analysis and Intent Recognition: A New Task, Dataset and Baselin (https://arxiv.org/abs/2405.08427)
Comments:
          10 pages, 6 figures

- **What's New**: 인공지능 주요 연구 팀이 새로운 과제를 제시했습니다: '스티커가 포함된 다중모드 채팅 감정 분석 및 의도 인식(MSAIRS)'. 이들은 주류 소셜 미디어 플랫폼에서 수집된 중국어 채팅 기록과 스티커를 포함한 새로운 다중모드 데이터셋을 도입했습니다. 이 연구는 스티커가 채팅 감정과 의도에 미치는 영향을 이해하는 데 중점을 두고 있습니다.

- **Technical Details**: MSAIRS 과제는 같은 텍스트지만 다른 스티커와 결합된 데이터쌍과 같은 이미지이지만 다른 텍스트를 가진 다양한 스티커를 포함하는 새로운 다중모드 데이터셋을 포함합니다. 연구팀은 텍스트와 스티커 이미지를 통합적으로 처리하는 모델인 MMSAIR을 제안했습니다. 이 모델은 다중 헤드 마스크드 어텐션(mechanisms)을 활용해 컨텍스트, 스티커, 스티커 텍스트를 통합하여 감정 분석 및 의도 인식을 수행합니다. PaddleOCR을 사용하여 스티커 내의 텍스트를 추출하고, 5명의 언어학 전문가가 감정 및 의도 레이블을 주석으로 추가했습니다.

- **Performance Highlights**: 새롭게 제안된 MMSAIR 모델은 기존의 단일 모드 및 다중 모드 모델들과 비교했을 때 성능이 뛰어나다는 것을 실험 결과를 통해 입증했습니다. 이 연구는 스티커의 시각적 정보가 감정 및 의도 인식에서 중요하다는 것을 강조했습니다. 데이터 품질과 신뢰성을 보장하기 위해 각 데이터 항목을 최소 3명의 전문가가 동일하게 주석을 달지 않은 경우 해당 데이터를 폐기하는 엄격한 검증 절차를 거쳤습니다.



### Investigating the 'Autoencoder Behavior' in Speech Self-Supervised Models: a focus on HuBERT's Pretraining (https://arxiv.org/abs/2405.08402)
- **What's New**: 이번 연구는 HuBERT 모델의 'autoencoder' 행동을 개선하여 음성 인식 성능을 향상시키는 전략을 제안합니다. 주된 발견은 최상위 레이어를 재설정하지 않고도 성능을 높일 수 있는 방법을 탐구했다는 점입니다. 실험 결과, 학습 절차를 개선하면 다운스트림 작업에서 경쟁력 있는 성능을 보이며, 학습 속도 또한 증가한다는 것을 보여줍니다.

- **Technical Details**: HuBERT(Hidden-unit BERT) 모델은 음성 컨텐츠에 대해 BERT와 유사한 트레이닝을 수행합니다. 세 단계로 구성된 이 방법은 음성 샘플의 특징 추출, K-means 클러스터링, 클러스터 라벨을 사용한 사전 학습 과정을 포함합니다. 원래 접근 방법에서는 두 번의 반복 수행하며, 첫 번째 반복에서는 MFCC 특징을 클러스터링 하는데 사용하고, 이후 반복에서는 이전 반복에서 훈련된 모델의 특정 레이어에서 추출한 임베딩을 클러스터링 특징으로 사용합니다.

- **Performance Highlights**: 이번 연구는 HuBERT 모델의 상위 레이어에서 높은 수준의 정보를 유지하는 전략을 통해 다운스트림 작업에서의 성능을 조금 향상시키는 동시에, 학습 시간도 절반으로 줄일 수 있음을 입증했습니다. 이를 통해 더욱 효율적이고 에너지 소모가 적은 모델 학습이 가능해졌습니다.



### Stylometric Watermarks for Large Language Models (https://arxiv.org/abs/2405.08400)
Comments:
          19 pages, 4 figures, 9 tables

- **What's New**: 최근 등장한 대형 언어 모델(LLMs)은 인간과 기계가 작성한 텍스트를 구분하기 점점 더 어렵게 만들고 있습니다. 이를 해결하기 위해, 우리는 텍스트 생성 시 토큰의 확률을 전략적으로 조절하는 새로운 워터마크 생성 방법을 제안합니다. 이 방법은 스타일로메트리와 같은 언어적 특징(stylometry)을 활용하며, 구체적으로는 아크로스티카(acrostica)와 감지-운동 규범(sensorimotor norms)을 LLMs에 도입합니다. 또, 이러한 특징은 매 문장마다 업데이트되는 키(key)에 의해 매개됩니다.

- **Technical Details**: 우리의 접근 방식은 생성 과정 중 토큰 확률을 직접적으로 업데이트하는 것으로, 기존의 방법들과 달리 스타일로메트릭(stylometric) 특징에 집중합니다. 이 방법은 키(key)를 통해 특징을 추가적으로 제어하며, 이 키는 문장의 의미에 기반하여 동적으로 변화합니다. 첫 번째 문장이 생성되면, 그 문장은 다음 문장의 스타일로메트릭 워터마크를 제어하는 초기 키 값을 인코딩합니다. 이번 연구에서는 감지-운동 규범(sensorimotor norms)과 아크로스티카(acrostica)로 특징을 제한했습니다. 감지-운동 규범은 인간 인지에 기반한 범주이며, 아크로스티카는 문장의 첫 글자를 조합하여 특정 메시지를 만들어냅니다.

- **Performance Highlights**: 우리의 평가 결과에 따르면, 세 문장 이상일 경우, 우리의 방법은 0.02의 낮은 오탐율(false positive rate)과 미탐율(false negative rate)을 달성했습니다. 사이클 번역 공격(cyclic translation attack)에서도 유사한 성능을 보였습니다. 이러한 연구는 소유 LLMs에 대한 책임성을 높이고 사회적 해악 방지를 위해 특히 중요합니다.



### PromptMind Team at MEDIQA-CORR 2024: Improving Clinical Text Correction with Error Categorization and LLM Ensembles (https://arxiv.org/abs/2405.08373)
Comments:
          Paper accepted for oral presentation at Clinical NLP workshop, NAACL 2024

- **What's New**: 이 논문은 MEDIQA-CORR 공유 작업에 대한 접근법을 설명하며, 의료 전문가들이 작성한 클리닉 노트에서 오류 감지 및 수정 작업을 다룹니다. 이 작업에서는 오류 존재 감지, 오류가 포함된 구체적인 문장 식별 및 오류 수정의 세 가지 하위 작업을 처리합니다. 본 연구는 인터넷 데이터에 기반해 훈련된 대형 언어 모델(LLMs)의 역량을 평가하고, 특히 체인-오브-띵킹(chain-of-thought) 인컨텍스트 학습 전략을 활용해 모든 하위 작업을 포괄적으로 다루는 방법을 제안합니다.

- **Technical Details**: 본 연구에서는 LLMs의 성능을 평가하기 위해 특정 오류 유형을 기반으로 임무 중심의 지시 사항을 작성합니다. 공용 접근 가능한 LLMs은 주로 사용자 지시를 따르도록 튜닝된 모델입니다. 우리는 추론 비용과 개발 노력을 최소화하고 효율성과 비용 효율성을 높이기 위해 단일 프롬프트 ICL 접근 방식을 채택합니다. 메디컬 시스템에서 예측 오류의 심각성을 고려해, 자기 일관성(self-consistency) 및 앙상블(ensemble) 방법을 활용하여 오류 수정 및 오류 감지 성능을 향상시키는 것을 목표로 합니다.

- **Performance Highlights**: 오류 감지에서 바이너리 분류(binary classification)로 텍스트에 오류가 포함되어 있는지 감지하고, 스팬 식별(span identification)로 오류가 있는 텍스트 스팬을 찾아내며, 자연어 생성(Natural Language Generation)으로 오류가 있는 경우 수정된 텍스트를 생성합니다. 평가 메트릭으로는 주로 정확도가 사용되며, 생성된 텍스트의 질을 평가하기 위해 ROUGE-1F, BERTScore, BLEURT-20 등의 메트릭 엔셈블(metric ensemble) 방식을 채택합니다.



### Seal-Tools: Self-Instruct Tool Learning Dataset for Agent Tuning and Detailed Benchmark (https://arxiv.org/abs/2405.08355)
Comments:
          14 pages, 10 figures

- **What's New**: 새로운 도구 학습 데이터셋 'Seal-Tools'가 소개되었습니다. Seal-Tools는 자체 지시(Self-Instruct) 방법을 사용하여 생성되며, 실용적인 도구의 적용 사례를 포함하는 대규모 데이터셋입니다. 특히 중첩된 도구 호출 사례와 다중 도구 호출 사례를 포함하여 복잡한 쿼리를 처리할 수 있는 능력을 평가하는 데 중점을 두고 있습니다.

- **Technical Details**: Seal-Tools 생성에는 LLM(Large Language Models)을 활용한 자체 지시(Self-Instruct) 방법이 사용되었습니다. 먼저, 다양한 도메인을 나타내는 필드 세트를 생성하고, 각 필드에 맞는 도구들을 생성합니다. 이후, 여러 단계를 거쳐 단일 또는 다중 도구 호출을 통해 요청을 해결하는 인스턴스를 생성합니다. 생성된 데이터는 JSON 형식으로 엄격하게 컨트롤됩니다.

- **Performance Highlights**: Seal-Tools를 사용하여 여러 주요 LLM과 자체 미세조정된 모델을 평가한 결과, 현재 시스템들은 여전히 개선의 여지가 있음을 확인했습니다. 특히 중첩된 호출을 처리하는 데 있어서 어려움을 겪고 있으며, 이를 통해 모델의 한계를 파악할 수 있었습니다.



### SpeechGuard: Exploring the Adversarial Robustness of Multimodal Large Language Models (https://arxiv.org/abs/2405.08317)
Comments:
          9+6 pages, Submitted to ACL 2024

- **What's New**: 최근 음성 지시를 따르고 관련 텍스트 응답을 생성할 수 있는 통합 음성 및 대형 언어 모델(SLMs)은 인기를 끌고 있습니다. 하지만 이러한 모델들의 안전성과 견고성은 아직 명확하지 않습니다. 이 논문에서는 SLMs의 지시를 따르는 능력이 적대적 공격(adversarial attack)과 '탈옥(jailbreaking)'에 얼마나 취약한지 조사했습니다.

- **Technical Details**: SLMs가 적대적 공격과 탈옥 공격에 노출 될 수 있는 가능성을 조사하기 위해 화이트박스(white-box)와 블랙박스(black-box) 공격 설정 모두에서 인간 개입 없이 적대적 예제를 생성할 수 있는 알고리즘을 설계했습니다. 또한, 이러한 탈옥 공격을 막기 위한 대책도 제안했습니다. 이번 연구에서는 음성 질문 응답(Spoken QA) 과제를 통해 SLMs의 안전성 정렬(safety alignment)과 관련된 성능을 평가했습니다. 공격 시나리오로는 모델의 모든 정보에 접근할 수 있는 화이트박스와 제한된 정보만 가진 블랙박스 공격을 포함했습니다. 모델의 그라디언트에 완전히 접근할 수 있는 공격자는 거의 인식할 수 없는 오디오 변형을 통해 SLM의 안전 훈련을 탈옥시킬 수 있었습니다.

- **Performance Highlights**: 우리의 모델은 음성 지시로 훈련된 대화 데이터를 통해 최신 성능을 달성하였으며, 안전성과 유용성 지표에서 80%를 넘겼습니다. 그러나 탈옥 실험에서는 SLMs가 적대적 변형과 전이 공격에 매우 취약하다는 사실을 밝혀냈으며, 평균적인 공격 성공률은 각각 90%와 10%로 나타났습니다. 제안된 대책은 공격 성공률을 크게 감소시켰습니다.



### A Decoupling and Aggregating Framework for Joint Extraction of Entities and Relations (https://arxiv.org/abs/2405.08311)
- **What's New**: 이 새로운 연구는 Named Entity Recognition(NER)와 Relation Extraction(RE) 두 가지 정보 추출의 중요한 하위 작업을 결합하여 수행하는 새로운 모델을 제안합니다. 주요 혁신점으로는 기능 인코딩 절차를 주체, 객체 및 관계 인코딩으로 구분하여 세부적인 하위 작업별 특성을 사용할 수 있도록 하는 것입니다. 또한, 새로운 상호-집계 및 내부-집계 전략을 제안하여 정보 상호작용과 개별 하위 작업별 특성 구축을 강화합니다.

- **Technical Details**: 이 모델은 인코딩 단계를 주체 인코딩(Encoding Subjects, ES), 객체 인코딩(Encoding Objects, EO) 및 관계 인코딩(Encoding Relations, ER)으로 분리하여 세부적인 의미 표현을 구성합니다. 그런 다음, 각 하위 작업에 대해 정보를 획득, 저장 및 상호작용하는 기능을 수행하는 세 가지 하위 작업별 셀을 설계합니다. 이 후, ES, EO 및 ER 간의 세부적인 정보 상호작용을 수행하고 강화하는 집계 방법을 구현합니다. 디코딩 단계에서는 ES와 EO 하위 작업별 특성을 결합하여 NER 특성을 생성하고, 이를 통해 ER 작업을 위한 엔티티 의미를 강화합니다.

- **Performance Highlights**: 따라서 이 모델은 NYT, WebNLG, ACE2004, ACE2005, CoNLL04, ADE 및 SciERC의 7가지 벤치마크 데이터셋을 기반으로 한 일련의 실험에서 몇 가지 이전의 최첨단 모델들을 능가하는 성능을 보였습니다. 추가적인 광범위한 실험을 통해 모델의 효과가 더욱 입증되었습니다.



### Computational Thought Experiments for a More Rigorous Philosophy and Science of the Mind (https://arxiv.org/abs/2405.08304)
Comments:
          6 pages, 4 figures, to appear at CogSci 2024

- **What's New**: 이번 논문에서는 'Virtual World Cognitive Science (VW CogSci)'라는 새로운 연구 방법론을 제안합니다. 연구자들이 가상의 세계에 임베디드된 가상화된 인체를 통해 인지과학 분야의 질문을 탐구하는 방법입니다. 특히, 이 접근법은 정신적 및 언어적 표현에 대한 철학적 사고 실험을 강화시키고, 이러한 표현의 과학적 연구에서 사용되는 용어를 다룹니다.

- **Technical Details**: VW CogSci는 가상화된 임베디드 에이전트를 사용하여 가상 세계에서 실험을 수행함으로써 다양한 가상 환경에서 가능한 인지 과정을 탐구합니다. 기존의 인지 과학 연구들이 실제 인간이나 동물의 인지 과정을 연구하는 것과는 달리, 이 방법론은 다양한 가능한 환경에서 다양한 가능한 마음의 작동 방식을 연구할 수 있게 해줍니다. 이 접근법은 인지 과정의 복잡한 동적 관계를 모델링하고 시뮬레이션을 통해 더 엄격한 과학적 분석을 가능하게 합니다.

- **Performance Highlights**: VW CogSci가 제공하는 주요 이점 중 하나는 철학적 퍼즐을 해결할 수 있다는 점입니다. 예를 들어 '고양이가 웃기다'라는 신념이나 '고양이'라는 개념과 같은 유형의 신념과 개념을 제거하고, 각 개인의 사고 속에 있는 신념과 개념의 특정 사례들을 보존함으로써 더 명확한 설명을 제공합니다. 또한, 이 방법은 실제 환경에서 다양한 가능한 인지 과정의 시뮬레이션을 통해 더 정확하고 정밀한 연구 결과를 도출할 수 있습니다.



### SpeechVerse: A Large-scale Generalizable Audio Language Mod (https://arxiv.org/abs/2405.08295)
Comments:
          Single Column, 13 page

- **What's New**: 최근 수많은 연구들이 음성과 텍스트 입력을 인식하는 능력을 가진 대형 언어 모델(LLMs)의 기능을 확장했지만, 이러한 모델들은 주로 자동 음성 인식과 번역 등 특정한 튜닝된 작업에만 한정되어 있었습니다. 이에 따라, 우리는 SpeechVerse라는 새로운 다중 작업 학습 및 커리큘럼 학습 프레임워크를 개발했습니다. 이 프레임워크는 소수의 학습 가능한 파라미터를 통해 사전 학습된 음성 및 텍스트 기반 모델을 결합하며, 훈련 중에는 사전 학습된 모델을 고정(frozen) 상태로 유지하여 효율성을 극대화합니다.

- **Technical Details**: SpeechVerse는 음성 기반 모델에서 추출된 연속적인 잠재 표현(continuous latent representations)을 사용하여 명령어 수행을 위한 모델을 미세 조정(instruction finetuning)합니다. 이 프레임워크는 다양한 음성 처리 작업에서 자연언어 지시를 사용하여 최적의 제로샷(zero-shot) 성능을 달성할 수 있도록 설계되었습니다. 또한, 우리의 모델은 범용적인 명령어 수행 능력을 평가하기 위해 도메인 외(different domain) 데이터셋, 새로운 프롬프트(prompt), 그리고 보지 못한 작업들에 대해 검증되었습니다.

- **Performance Highlights**: 다양한 데이터셋과 작업에 대해 전통적인 기본 모델들과 비교하는 광범위한 벤치마킹을 수행한 결과, 우리의 다중 작업 SpeechVerse 모델은 11개 작업 중 9개 작업에서 기존의 특정 작업에 최적화된 모델들보다 우수한 성능을 보였습니다.



### Detecting Fallacies in Climate Misinformation: A Technocognitive Approach to Identifying Misleading Argumentation (https://arxiv.org/abs/2405.08254)
- **What's New**: 이 연구에서는 기후 변화 허위정보를 해체하는 기존의 비판적 사고 방법론을 적용하여 기후 허위정보의 다양한 유형을 추론 오류와 연결하는 데이터셋을 개발했다. 이 데이터셋을 이용해 기후 허위정보에서 오류를 감지할 수 있는 모델을 훈련했다.

- **Technical Details**: 이 연구는 심리학과 컴퓨터 과학 연구를 융합한 '테크노인지적(technocognitive)' 접근 방법을 채택했다. 데이터셋은 구조적 오류와 배경지식이 필요한 오류로 분류된 다양한 기후 허위정보 추론 오류를 포함한다. ZeroR 분류기를 사용해 기본 성능을 평가하고, 더 나아가 LLMs (Large Language Models)와의 성능 비교를 실시했다. 또한, CARDS 및 FLICC 프레임워크를 통합하여 자동화된 반박 솔루션을 구현할 수 있는 기반을 마련했다.

- **Performance Highlights**: 제안된 모델은 이전 연구들보다 2.5에서 3.5점 더 높은 F1 점수를 기록했다. 쉽게 감지할 수 있는 오류에는 가짜 전문가와 일화적 주장이 포함되었으며, 배경 지식이 필요한 오류인 과도한 단순화, 잘못된 묘사, 게으른 귀납법은 상대적으로 감지하기 어려웠다.



### A predictive learning model can simulate temporal dynamics and context effects found in neural representations of continuous speech (https://arxiv.org/abs/2405.08237)
Comments:
          Accepted to CogSci 2024

- **What's New**: 본 연구는 인간의 뇌에서 나타나는 음성 인지의 시간적 특성과 맥락적 특성을 시뮬레이션하기 위해 자기지도학습(Self-Supervised Learning, SSL)으로 훈련된 컴퓨팅 모델을 사용합니다. 이 모델은 미래의 음향을 예측하는 학습 목표로 훈련되어 언어적 지식 없이도 특정한 음성 인지 특성을 재현할 수 있음을 증명합니다.

- **Technical Details**: 이 연구에서는 음성 신호를 예측하는 SSL 기반의 순환 신경망(Recurrent Neural Network, RNN)을 사용했습니다. Contrastive Predictive Coding(CPC) 프레임워크를 활용하여 각 10ms 프레임에 대해 512차원의 벡터 표현을 학습했습니다. 이 모델은 6000시간 분량의 오디오북으로 훈련되었으며, phonetic(음소) 디코딩 능력을 테스트하기 위해 Librispeech 데이터셋의 'dev-clean' 서브셋에서 추출된 CPC 표현과 음향 특징을 사용했습니다.

- **Performance Highlights**: 모델은 인간 뇌와 유사하게 여러 음소를 동시에 처리하고, 각 음소의 표현은 시간이 지남에 따라 진화하는 특성을 보였습니다. 또한, 모델의 음소 디코딩 창이 180ms 전부터 시작해 540ms 동안 유지되는 것을 발견했습니다. 음향 특징에서의 디코딩 창이 이보다 훨씬 짧았지만, CPC 표현은 훨씬 높은 정확도로 음소를 디코딩할 수 있음을 보여주었습니다. 이는 Co-articulation 효과와 관련이 있으며, 향후 연구에서는 수동 정렬된 데이터를 사용하여 이러한 실험을 재현할 계획입니다.



### An information-theoretic model of shallow and deep language comprehension (https://arxiv.org/abs/2405.08223)
Comments:
          6 pages; accepted to COGSCI 2024

- **What's New**: 심리언어학에서 언어 이해가 피상적일 수 있다는 아이디어를 토대로, 정보 이론을 이용해 처리 깊이와 정확성 간의 최적 트레이드오프(trade-off)를 공식화한 새로운 모델을 제안합니다. 이 모델은 EEG 신호와 읽기 시간을 통해 처리 노력을 측정할 수 있는 방법을 제공합니다.

- **Technical Details**: 정보 이론을 사용하여 언어 이해를 처리 깊이의 변화로 공식화했습니다. 처리 깊이는 입력에서 추출된 정보의 비트로 측정되며, 시간이 지남에 따라 증가합니다. 모델은 최적의 해석 정책을 선택하고, 처리 깊이의 증가에 따라 처리 노력이 변화하는 방식을 설명합니다.

- **Performance Highlights**: 제안된 모델은 복잡한 의미적 및 통사적 조작에서 유도된 N400, P600 및 양상 ERP 효과를 시뮬레이션하여 검증되었습니다. 또한, Syntactic Ambiguity Processing 데이터셋에서 문장이 종결되는 시간에 대한 예측을 성공적으로 수행했습니다.



### Interpreting Latent Student Knowledge Representations in Programming Assignments (https://arxiv.org/abs/2405.08213)
Comments:
          EDM 2024: 17th International Conference on Educational Data Mining

- **What's New**: 최근 교육 분야의 인공지능 발전은 Open-ended(자유형) 학생 응답을 예측하는 데에서 사용되는 생성적 거대 언어 모델을 활용하고 있습니다. 이 논문에서는 InfoOIRT라고 불리는 정보 정규화 Open-ended Item Response Theory(IRR) 모델을 제안합니다. 이 모델은 학생들이 작성한 코드를 예측하며, 해석 가능한 학생 지식 표현을 학습하는 데 중점을 둡니다.

- **Technical Details**: InfoOIRT는 InfoGAN에서 영감을 받아, 간단한 사전 분포를 가진 고정된 하위 집합의 잠재 지식 상태와 생성된 학생 코드를 최대한 상호 정보(Mutual Information)가 되도록 하여 해석 가능한 잠재 변수(Disentangled Representations)를 학습하도록 합니다. 컴퓨터 과학 교육에 적용하여 학생들이 작성한 코드를 기반으로 지식 상태를 예측합니다.

- **Performance Highlights**: 실제 프로그래밍 교육 데이터셋에서 실험을 통해, InfoOIRT가 기존 모델들과 비교해도 학생 코드 예측의 정확성이 손상되지 않으면서 해석 가능한 지식 표현을 학습할 수 있음을 보여주었습니다. 정성적 분석을 통해 이 모델이 학습한 학생 지식 표현의 해석 가능한 특징들도 제시되었습니다.



### CANTONMT: Investigating Back-Translation and Model-Switch Mechanisms for Cantonese-English Neural Machine Translation (https://arxiv.org/abs/2405.08172)
Comments:
          on-going work, 30 pages

- **What's New**: 이번 연구 논문에서는 광둥어에서 영어로의 번역을 위한 새로운 기계 번역 모델을 개발하고 평가합니다. 현재 상용 모델들과 비교하여 뛰어난 성능을 보이는 새로운 모델을 제안합니다.

- **Technical Details**: 광둥어-영어 번역을 위해 새로운 병렬 코퍼스(corpus)를 온라인에서 수집하고 전처리 과정을 통해 생성했습니다. 또한 웹 스크래핑(web scraping)을 통해 모노링구얼(monolingual) 광둥어 데이터셋도 만들었습니다. 백번역(back-translation) 및 모델 스위치(model switch) 등의 방식을 사용해 모델을 미세 조정(fine-tuning)했습니다. 평가 지표로는 SacreBLEU, hLEPOR(어휘 기반 지표), COMET, BERTscore(임베딩 스페이스 기반 지표)를 사용했습니다.

- **Performance Highlights**: 제안된 최고 성능의 모델(NLLB-mBART)은 SacreBLEU 점수 16.8을 기록하며, 상용 번역기인 Bing과 Baidu 번역기와 비교했을 때 비슷하거나 더 나은 자동 평가 점수를 획득했습니다. 또한, 사용자들이 다양한 훈련된 모델을 비교할 수 있도록 오픈 소스 웹 애플리케이션도 개발되었습니다.



### Benchmarking Retrieval-Augmented Large Language Models in Biomedical NLP: Application, Robustness, and Self-Awareness (https://arxiv.org/abs/2405.08151)
- **What's New**: 이번 논문에서는 대규모 언어 모델(LLM)의 한계를 극복하고자, 검색을 통해 적절한 정보를 추가하는 Retrieval-Augmented LLM(RAL)이 다양한 생의학 자연어 처리(NLP) 작업에서 어떻게 성능을 발휘하는지 체계적으로 평가합니다. 이를 위해 5가지 생의학 작업(트리플 추출, 링크 예측, 분류, 질문 응답, 자연어 추론)에서 RAL의 성능을 분석하고, 9개의 데이터셋을 활용해 3가지 대표 LLM과 3가지 검색기를 평가했습니다.

- **Technical Details**: 논문은 RAL을 평가하기 위해 네 가지 기본 능력 즉, 라벨이 없는 데이터를 처리하는 능력(unlabeled robustness), 반사실 데이터를 처리하는 능력(counterfactual robustness), 다양한 데이터를 처리하는 능력(diverse robustness), 그리고 부정적 인식을 하는 능력(negative awareness)을 제안합니다. 각 작업의 평가를 위해 네 개의 테스트베드를 구축했습니다.

- **Performance Highlights**: 생의학 도메인에서 RAL의 성능을 평가하기 위해 제안된 BIoRAB 벤치마크는 생의학 NLP 작업에서 오탐률 감소나 다양한 소스에서의 정보 검색 능력 향상 등 전반적인 성능 개선을 보여줬습니다. 특히, RAL 시스템이 라벨이 없는 데이터나 오류가 있는 데이터에서도 기존 LLM보다 우수한 성능을 발휘한다는 점에서 의미가 있습니다.



### Discursive objection strategies in online comments: Developing a classification schema and validating its training (https://arxiv.org/abs/2405.08142)
Comments:
          This paper was accepted and presented at the 73rd Annual International Communication Association International Conference, May 2023

- **What's New**: 이 논문에서는 유해한 발언에 대한 사람들의 반응 전략을 이해하고자 합니다. 연구자들은 YouTube와 Twitter의 인기 뉴스 동영상 댓글에서 6500개 이상의 댓글 답글을 내용 분석하여 7가지 뚜렷한 반박 전략을 식별했습니다. 주요 연구 결과는 사람들이 발언에 이의를 제기할 때 다양한 담화 전략을 사용하며, 평판 공격(reputational attacks)이 가장 흔하다는 사실을 보여줍니다.

- **Technical Details**: 연구진은 두 가지 주요 연구를 수행했습니다. 첫 번째 연구(Study 1)에서는 YouTube와 Twitter에서 6500개 이상의 댓글 답글을 분석하여 담화 반박 전략을 식별했습니다. 두 번째 연구(Study 2)에서는 2004개의 추가 댓글 답글 샘플을 통해 각 전략의 발생 빈도를 조사했습니다. 이 연구들은 이론적 접근 방식을 고려한 반박 방식의 분류 체계를 제공하며, 캠퍼스 내에서 모욕적이거나 문제 있는 발언을 막기 위한 풀뿌리 노력의 포괄적 관점을 제공합니다.

- **Performance Highlights**: 연구 결과에서는 평판 공격(reputational attacks)이 가장 일반적인 반박 전략으로 나타났으며, 사람들이 다양한 담화 전략을 사용하여 유해한 발언에 대응하는 모습을 확인할 수 있었습니다.



### Many-Shot Regurgitation (MSR) Prompting (https://arxiv.org/abs/2405.08134)
- **What's New**: 새로운 Many-Shot Regurgitation (MSR) 프롬프트 기법이 도입되었습니다. 이 기법은 대규모 언어 모델(LLMs)의 정밀한 콘텐츠 재생산 여부를 검토하기 위한 새로운 블랙박스 멤버십 추론 공격 프레임워크입니다.

- **Technical Details**: MSR 프롬프트 기법은 입력 텍스트를 여러 세그먼트로 나누고, 사용자와 언어 모델 간의 가상 대화를 통해 원문을 유도하는 단일 프롬프트를 만드는 것을 포함합니다. 주어진 텍스트를 여러 부분으로 나누고, 지정된 순서에 따라 언어 모델의 출력을 꾸민 후 마지막 세그먼트를 얻습니다. 이러한 기법은 오직 모델을 프롬프트하는 능력과 그 출력을 관찰하는 능력만으로 작동되어, 실제 배포된 GPT-3.5나 GPT-4와 같은 블랙박스 모델에 적합합니다.

- **Performance Highlights**: MSR 프롬프트 기법을 다양한 텍스트 소스(위키피디아, OER 교과서 등)에 적용한 결과, 사전 훈련 데이터셋(D_pre)과 후속 데이터셋(D_post) 간의 원문 재생산 빈도 분포에 현저한 차이가 있음을 발견했습니다. 예를 들어, GPT-3.5를 위키피디아 기사에 적용했을 때, Cliff's delta가 -0.984, Kolmogorov-Smirnov (KS) distance가 0.875로 나타났으며, 이는 D_pre와 D_post 간의 원문 매칭 분포에서 유의미한 차이를 나타냅니다.



### KET-QA: A Dataset for Knowledge Enhanced Table Question Answering (https://arxiv.org/abs/2405.08099)
Comments:
          LREC-Coling 2024

- **What's New**: 이번 논문에서는 Table Question Answering (TableQA) 시스템의 외부 지식 사용에 관한 문제를 해결하고자 KET-QA 데이터셋을 제안합니다. 이 데이터셋은 자세한 골드 증거(annotation)를 포함하고 있으며, 각 테이블은 지식 베이스(Knowledge Base, KB)의 서브 그래프와 연결됩니다. 각 질문은 테이블과 서브 그래프의 정보를 통합하여 답변해야 합니다.

- **Technical Details**: KET-QA 데이터셋 구축을 위해 두 가지 주요 도전을 해결해야 했습니다: (1) 적절한 외부 지식 베이스와 매핑할 수 있는 테이블 식별, (2) 자연 외부 지식 필요한 질문 생성. 이를 위해 HybridQA 질문을 재주석하고, Wikidata를 외부 지식 소스로 대체했습니다. 각 테이블은 평균적으로 1,696.7개의 3중(트리플)을 포함하는 서브 그래프와 연결됩니다. 이를 기반으로, 질문, 테이블, 검색된 트리플을 통합하여 최종 답변을 생성하는 retriever-reasoner 파이프라인 모델을 개발했습니다.

- **Performance Highlights**: 실험 결과, 모델은 다양한 설정(미세 조정, zero-shot, few-shot)에서 전통적인 TableQA 방식에 비해 1.9배에서 6.5배의 상대적 성능 향상과 11.66%에서 44.64%의 절대 성능 향상을 기록했습니다. 그러나 최고 성능도 60.23%의 EM 점수에 그쳐 인간 수준의 성능에는 아직 미치지 못합니다. 이로 인해 KET-QA는 여전히 질문 응답 커뮤니티에서 도전적인 문제로 남아 있습니다.



### Improving Transformers with Dynamically Composable Multi-Head Attention (https://arxiv.org/abs/2405.08553)
Comments:
          Accepted to the 41th International Conference on Machine Learning (ICML'24)

- **What's New**: 최신 연구에서 Multi-Head Attention (MHA)의 단점을 보완하고 모델의 표현력을 향상시키기 위해 Dynamically Composable Multi-Head Attention (DCMHA)를 제안했습니다. DCMHA는 파라미터와 연산 효율이 높은 주목(attention) 구조를 도입하여 MHA의 한계를 극복하고 주목 헤드를 동적으로 조합합니다. DCMHA는 모든 트랜스포머(transformer) 아키텍처에서 MHA를 대체할 수 있으며, 이를 통해 DCFormer라는 새로운 모델을 생성할 수 있습니다.

- **Technical Details**: DCMHA의 핵심은 $	extit{Compose}$ 함수로, 입력에 따라 주목 점수(attention score)와 가중치 행렬을 변환합니다. DCMHA는 주목 행렬(attention matrix)을 동적으로 구성하여 모델의 표현력을 증가시킵니다. 기존의 MHA에서는 주목 헤드가 독립적으로 작동하기 때문에 저차원 병목 현상과 헤드 중복 문제가 발생했습니다. DCMHA는 이러한 문제를 해결하기 위해 주목 행렬을 입력에 따라 유연하게 조합합니다.

- **Performance Highlights**: 실험 결과, DCFormer는 다양한 아키텍처와 모델 규모에서 전통적인 트랜스포머보다 언어 모델링 성능에서 월등하며, 약 1.7~2.0배의 연산 성능을 발휘합니다. 예를 들어, DCPythia-6.9B는 오픈 소스 Pythia-12B보다 전처리 퍼플렉서티(pretraining perplexity)와 다운스트림 평가에서 더 우수한 성능을 보였습니다. 또한, DCMHA는 비전 트랜스포머에도 적용 가능하며, 이미지 분류에서 성능을 입증했습니다.



### Falcon 7b for Software Mention Detection in Scholarly Documents (https://arxiv.org/abs/2405.08514)
Comments:
          Accepted for publication by the first Workshop on Natural Scientific Language Processing and Research Knowledge Graphs - NSLP (@ ESCAI)

- **What's New**: 이번 연구는 다양한 학문 분야에서 소프트웨어 도구의 통합 증가에 따른 도전 과제에 대응하고자, Falcon-7b 모델을 활용하여 학술 텍스트 내 소프트웨어 언급을 탐지하고 분류하는 방법을 조사합니다. 특히, Software Mention Detection in Scholarly Publications (SOMD)의 Subtask I 해결에 집중하여, 학술 문헌에서의 소프트웨어 언급을 식별하고 분류하는 작업을 다룹니다.

- **Technical Details**: 본 연구에서는 다양한 훈련 전략을 탐구하였으며, 여기에는 이중 분류기 접근법, 적응형 샘플링(adaptive sampling), 가중 손실 스케일링(weighted loss scaling)이 포함됩니다. 이러한 전략을 통해 클래스 불균형(class imbalance) 및 학술적 문장 구조의 복잡성을 극복하고 탐지 정확도를 향상시키고자 했습니다. 특히 Falcon-7b 모델을 기반으로 한 토큰 분류 시스템(token classification system)을 사용하여 이 작업을 수행했습니다.

- **Performance Highlights**: 실험 결과에 따르면 선택적 라벨링(selective labelling)과 적응형 샘플링이 모델의 성능 향상에 상당한 기여를 했으나, 여러 전략을 통합하는 것이 반드시 누적 성능 향상으로 이어지지는 않았습니다. 이를 통해 학술 텍스트 분석 작업에서 대형 언어 모델(LLM)의 효과적인 적용에 대한 통찰을 제공하며, 특정 과제 해결을 위한 맞춤형 접근법의 중요성을 강조합니다.



### Silver-Tongued and Sundry: Exploring Intersectional Pronouns with ChatGP (https://arxiv.org/abs/2405.08238)
Comments:
          Honorable Mention award (top 5%) at CHI '24

- **What's New**: 이번 연구는 ChatGPT와 같은 대규모 언어 모델(LLM)을 이용한 대화형 에이전트가 일본어 1인칭 대명사를 어떻게 사용하여 사회적 정체성을 시뮬레이션하는지를 조사하였습니다. 특히, 성, 나이, 지역, 그리고 형식을 교차적으로 나타내는 일본어 대명사를 통해 ChatGPT가 사회적 정체성을 모방할 수 있는 가능성을 실험했습니다.

- **Technical Details**: 이번 연구는 두 일본 지역(간토와 킨키)에서 10개의 일본어 1인칭 대명사를 사용하여 실험을 진행하였습니다. 대명사만으로도 성별, 나이, 지역, 형식성을 포함한 여러 사회적 정체성을 유도할 수 있음을 발견했습니다. 연구 방법으로는 혼합 방법론(mixed methods)을 사용하여 ChatGPT의 성별, 나이, 지역, 형식성에 대한 인식을 분석했습니다.

- **Performance Highlights**: 실험 결과, 일본어 1인칭 대명사만으로도 ChatGPT가 단순한 성별부터 복합적인 교차 정체성까지 다양한 사회적 정체성을 유도할 수 있음을 입증했습니다. 또한, 지역 변이에 따라 추가적인 사회적 정체성 표지가 필요하다는 점도 밝혀냈습니다. 이 연구는 일본 사용자 그룹을 대상으로 LLM 기반 지능형 에이전트를 사용하여 성별 및 교차 정체성을 신속하게 만들 수 있는 간단한 방법론을 제안했습니다.



### Who's in and who's out? A case study of multimodal CLIP-filtering in DataComp (https://arxiv.org/abs/2405.08209)
Comments:
          Content warning: This paper discusses societal stereotypes and sexually-explicit material that may be disturbing, distressing, and/or offensive to the reader

- **What's New**: 최근 웹에서 수집된 데이터셋은 비구조적이고 통제되지 않은 환경에서 가져와, 연구자와 산업 실무자들이 데이터 필터링 기술을 사용하여 '노이즈'를 제거하는 방식에 의존하고 있습니다. 이번 연구는 이러한 데이터셋을 생성할 때 사용되는 필터의 편향성과 가치 내재성을 평가하는 새로운 연구를 소개합니다.

- **Technical Details**: 연구팀은 학술 벤치마크 DataComp's CommonPool에서 이미지-텍스트 CLIP-필터링의 표준 접근 방식을 감사하고, 여러 주석 기술을 통해 다양한 모드에서 필터링 불일치를 분석했습니다. 주로 OpenAI CLIP 모델을 사용하여 CNN-기반(embedding space)를 통해 이미지와 텍스트의 유사성을 평가하는 방법을 사용했습니다. 이 방식은 LAION-400M, LAION-5B, DataComp-1B 등의 대규모 데이터셋을 만드는 데 사용되었습니다.

- **Performance Highlights**: 연구 결과, LGBTQ+ 사람들, 노년 여성, 젊은 남성 등 여러 인구 통계 그룹과 관련된 데이터가 더 높은 비율로 제외된다는 것을 발견했습니다. 또한, 특정 소수 그룹이 이미 원본 데이터에서 과소 대표되어 있는데, CLIP-필터링이 이러한 그룹의 데이터를 더 높은 비율로 제외함으로써 포함 배제(exclusion amplification)를 증명했습니다. 한편, NSFW 필터는 성적으로 명시적인 콘텐츠를 제거하는 데 실패했으며, CLIP-필터링은 고율로 저작권이 있는 콘텐츠를 포함하는 것으로 나타났습니다.



### Exploring the Potential of Conversational AI Support for Agent-Based Social Simulation Model Design (https://arxiv.org/abs/2405.08032)
Comments:
          29 pages, 3 figures, 1 table

- **What's New**: ChatGPT를 포함한 대화형 AI 시스템(CAIS; Conversational AI Systems)의 사회 시뮬레이션(Social Simulation) 분야에서의 사용이 아직 제한적입니다. 특히, 에이전트 기반 사회 시뮬레이션(ABSS; Agent-Based Social Simulation) 모델 설계에는 거의 사용되지 않았습니다. 이 논문은 CAIS가 ABSS 모델 설계를 어떻게 지원할 수 있는지에 대한 개념 증명을 제공하며, 최소한의 사전 지식으로도 짧은 시간 내에 혁신적인 ABSS 모델을 개발할 수 있음을 보여줍니다.

- **Technical Details**: 논문은 고급 프롬프트 엔지니어링(Advanced Prompt Engineering) 기법과 'Engineering ABSS 프레임워크'를 활용하여, CAIS와 함께 또는 CAIS에 의해 ABSS 모델을 설계할 수 있는 포괄적인 프롬프트 스크립트를 구성했습니다. 이 스크립트를 통해 CAIS가 ABSS 모델 설계에 효과적으로 기여하는지를 한 박물관에서의 적응형 건축 사례를 통해 입증했습니다. 대화 중에 발생하는 일부 부정확성과 일탈을 고려하더라도, CAIS는 ABSS 모델러들에게 가치 있는 동료임을 확인할 수 있었습니다.

- **Performance Highlights**: CAIS는 특정 사례 연구에서 효과를 보여줬으며, 짧은 시간 내에 최소한의 사전 지식으로도 혁신적인 ABSS 모델을 개발할 수 있도록 도왔습니다. 이는 ABSS 모델러들이 더 적은 시간과 노력으로 더 창의적인 솔루션을 도출할 수 있도록 함을 시사합니다.



### Translating Expert Intuition into Quantifiable Features: Encode Investigator Domain Knowledge via LLM for Enhanced Predictive Analytics (https://arxiv.org/abs/2405.08017)
- **What's New**: 이 논문은 예측 분석(predictive analytics)에서 조사자들의 세밀한 도메인 지식이 주관적인 해석과 임시방편적인 의사 결정에만 제한적으로 활용되는 문제를 지적하고 있습니다. 이 문제를 해결하기 위해 저자는 대형 언어 모델(LLM: Large Language Models)을 활용하여 조사자가 파악한 인사이트(insights)를 정량적이고 실행 가능한 특징(feature)으로 전환하는 방법을 제안합니다.

- **Technical Details**: 제안된 프레임워크는 LLM의 자연 언어 이해 능력을 이용해 조사자의 '레드 플래그(red flags)'를 구조화된 특징(feature) 세트로 인코딩(encoding)합니다. 이 특징 세트는 기존의 예측 모델에 쉽게 통합될 수 있습니다. 이를 통해 인간 전문가의 중요한 지식을 보존할 뿐만 아니라, 이 지식의 영향을 다양한 예측 작업에 걸쳐 확장할 수 있습니다.

- **Performance Highlights**: 여러 케이스 스터디(case studies)를 통해, 이 접근 방식이 리스크 평가(risk assessment)와 의사 결정 정확도(decision-making accuracy)에서 상당한 개선을 이루었음을 증명했습니다. 이는 인간의 경험적 지식과 첨단 기계 학습 기법을 결합하는 것의 가치를 보여줍니다.



### Robot Detection System 1: Front-Following (https://arxiv.org/abs/2405.08014)
Comments:
          paper series

- **What's New**: 새로운 연구 결과와 설계 아이디어를 공유하고, 프론트 팔로잉(front-following) 기술에 대한 기본 이론과 알고리즘 분석을 제공합니다. 이 연구는 인간-컴퓨터 상호작용 기술 중 하나인 프론트 팔로잉을 중심으로 한 로봇 기술을 다루며, 기존의 백 팔로잉(back-following) 및 사이드 바이 사이드 팔로잉(side-by-side) 기술과의 차이점을 강조합니다.

- **Technical Details**: 프론트 팔로잉 기술을 기반으로 인간의 앞에서 따라가는 로봇의 설계를 목표로 하고 있습니다. 로봇 설계는 8개의 주요 연구 부문으로 나누어져 있으며, 현재 보고서에서는 메카니즘 및 회로 구조 설계, 센서 시스템 구축, 탐지 모델의 구조, 로봇 팔로잉과 제어 모델을 다루고 있습니다. 또한, 로봇은 RGBD 카메라와 같은 고급 센서를 사용하지 않고 목표물의 개인 정보 보호를 최대한 보장하는 것을 목표로 합니다.

- **Performance Highlights**: 제안된 프론트 팔로잉 기술은 목표물의 미래 경로를 예측하는 기능을 갖추고 있으며, 이에 따라 탐지 오류 및 예측 오류를 분석하여 최적의 성능을 위한 알고리즘 조정을 통해 더 나은 추적 성능을 제공합니다. 연구 결과의 실험적 검증을 통해 실제 환경에서의 유용성을 평가하고 최적화합니다.



### MedConceptsQA: Open Source Medical Concepts QA Benchmark (https://arxiv.org/abs/2405.07348)
- **What's New**: MedConceptsQA는 의료 개념에 대한 질문 응답을 위한 오픈 소스 벤치마크로 발표되었습니다. 이 벤치마크는 진단, 절차, 약물 등 다양한 의학 용어를 포함하며, 난이도에 따라 세 가지 레벨(쉬움, 중간, 어려움)로 구분된 질문들로 구성되어 있습니다.

- **Technical Details**: MedConceptsQA 벤치마크는 ICD9-CM, ICD10-CM 진단 코드, ICD9-PROC, ICD10-PROC 절차 코드, ATC 약물 코드 등의 질문으로 구성되어 있습니다. 각 질문은 하나의 의학 코드와 해당 항목의 설명을 선택하는 형태로 이루어져 있습니다. 난이도에 따른 질문의 분포는 무작위로 선택되어 있습니다.

- **Performance Highlights**: 여러 대형 언어 모델(LLM)을 대상으로 평가한 결과, 기존의 임상 대형 언어 모델(CLLM)들은 대부분 무작위 추측과 비슷한 정확도로 성능을 보였으나, GPT-4는 zero-shot 학습에서 27%, few-shot 학습에서 37%의 절대 평균 개선을 보였습니다. 이 결과는 GPT-4가 특정 임상 도메인에 맞추어 훈련되지 않았음에도 불구하고 다른 모델들보다 뛰어난 이해와 추론 능력을 가지고 있음을 나타냅니다.



### Towards a Path Dependent Account of Category Fluency (https://arxiv.org/abs/2405.06714)
Comments:
          To appear at CogSci 2024

- **What's New**: 이 연구에서는 기존의 카테고리 유창성(category fluency) 모델들이 가정한 마르코프 과정(Markov process) 대신, 서브카테고리(subcategory)를 도입하여 사람이 말하는 예시들의 순서를 더 잘 예측할 수 있음을 보여줍니다. 또한, LLMs를 사용하여 전체 시퀀스를 기반으로 다음 예시를 예측하는 새로운 방법을 제안합니다.

- **Technical Details**: 기존의 유창성 모델들은 예시들이 이전의 예시에만 의존한다고 가정했으나, 연구진은 서브카테고리를 추가하여 카테고리 전환 확률을 직접 모델링하고 대형 언어 모델(Large Language Models, LLMs)을 활용하여 더 나은 예측을 시도했습니다. 이 연구에서는 시퀀스 생성자로 모델을 재구성하고, 인간이 작성한 시퀀스와 비교하여 평가하는 새로운 메트릭을 제안했습니다.

- **Performance Highlights**: 연구 결과, 힐스(Hills et al., 2012) 모델이 추가 바이어스를 통해 더 나은 생성 품질을 보인다는 것을 확인했습니다. 또한, LLM만으로는 사람이 만든 시퀀스를 잘 생성하지 못했으나, 글로벌 큐(global cue)를 추가하면 성능이 향상되었습니다. 추가 실험에서는 결정론적 검색(deterministic search)이 무작위 검색(random sampling) 보다 인간과 더 유사한 유창성 시퀀스를 생성함을 확인했습니다.




New uploads on arXiv(cs.CL)

### How Far Can Cantonese NLP Go? Benchmarking Cantonese Capabilities of Large Language Models (https://arxiv.org/abs/2408.16756)
- **What's New**: 이번 연구는 8500만 명이 사용하는 광동어(Cantonese)가 자연어 처리(NLP) 기술에서 겪고 있는 발전 부족 문제를 다루고 있습니다. 특히, 광동어를 위한 새롭고 체계적인 NLP 방법과 대형 언어 모델(LLM) 성능을 평가하기 위한 새로운 벤치마크를 제안하여 이 문제를 해결하고자 합니다.

- **Technical Details**: 연구진은 광동어 LLM 기술의 발전을 위해 사실 생성(factual generation), 수학적 논리(mathematical logic), 복잡한 추론(complex reasoning), 일반 지식(general knowledge) 측면에서 LLM 성능을 평가하는 네 가지 새로운 벤치마크(Yue-Truthful, Yue-GSM8K, Yue-ARC-C, Yue-MMLU)를 소개하며, 이를 통해 성능을 정성적으로 및 정량적으로 평가할 수 있습니다.

- **Performance Highlights**: 대부분의 연구가 영어 및 기타 고급 언어에 집중된 가운데, 광동어 관련 연구는 매우 낮은 출판 비율을 보이며, 광동어 LLM 기술은 아직 많은 발전을 이루지 못했습니다. 그러나 이 연구는 23개의 주요 광동어 및 일반 LLM을 새로운 벤치마크를 통해 분석하고 그 가능성을 밝혀냅니다.



### Reinforcement Learning without Human Feedback for Last Mile Fine-Tuning of Large Language Models (https://arxiv.org/abs/2408.16753)
- **What's New**: 이 연구는 인간의 선호도와 정렬된 언어 모델을 강화 학습(reinforcement learning)을 통해 최적화하는 새로운 프레임워크를 개발하였으며, 특히 마지막 단계의 세부 조정(last-mile fine-tuning)에서 성능 개선을 목표로 합니다.

- **Technical Details**: 강화 학습을 활용하여 인간의 피드백을 모형화하고, 이를 통해서 모델이 최적의 상태 이외에서도 다양한 행동을 학습할 수 있도록 합니다. PPO(Proximal Policy Optimization)를 이용하여 보상 모델을 학습하고, 정적 데이터에 기반한 상태 전이 동역학을 정의합니다.

- **Performance Highlights**: 제안된 프레임워크는 추상적 요약(abstractive summarization) 작업에서 최대 우도(maximum likelihood) 방법에 비해 성능이 크게 향상되었습니다. 특히, 포스트 프로세싱(post-processing)의 필요성을 줄이는 새로운 최적화 방법으로 작용할 수 있으며, 잘못된 출력(hallucinations)과 같은 복잡한 부정적인 출력을 다루는 데에도 확대될 수 있습니다.



### A Gradient Analysis Framework for Rewarding Good and Penalizing Bad Examples in Language Models (https://arxiv.org/abs/2408.16751)
- **What's New**: 본 논문은 일반적인 언어 모델 최적화 방법론의 차별화된 측면을 분석하고, 좋은 샘플을 보상하고 나쁜 샘플을 처벌하는 새로운 접근 방식을 제시합니다. 이를 통해 ExMATE가 MLE의 우수한 대리자 역할을 하며, DPO와 결합 시 성능 향상을 이룰 수 있음을 보여줍니다.

- **Technical Details**: 이 연구는 언어 모델(LM) 최적화 방법들을 비교하기 위해 손실 함수의 기울기 분석을 사용하여, 좋은 샘플을 보상하고 나쁜 샘플을 처벌하는 다양한 방법론 간의 기능적 차이를 식별합니다. 주요 기법으로는 unlikelihood training, ExMATE, DPO 등이 포함되며, 이들 기법들은 단어의 확률 분포를 조정하는 데에 기여합니다.

- **Performance Highlights**: 실험 결과, ExMATE는 MLE와 비교했을 때 통계적 성능(5-7%) 및 생성 성능(+18% 승률)에서 뛰어난 결과를 보였습니다. DPO는 RLHF 접근 방식을 간소화하여 효율성을 유지하면서도 성능 향상에 기여합니다.



### Assessing Large Language Models for Online Extremism Research: Identification, Explanation, and New Knowledg (https://arxiv.org/abs/2408.16749)
- **What's New**: 미국에서 폭력적인 극단주의가 급증함에 따라 온라인에서 극단주의 이념의 확산을 감지하고 제한할 수 있는 자동화된 도구의 필요성이 커지고 있습니다. 이 연구는 Bidirectional Encoder Representations from Transformers (BERT)와 Generative Pre-Trained Transformers (GPT)의 성능을 평가하여 온라인 국내 극단주의 게시물을 탐지하고 분류하는데 중점을 두었습니다.

- **Technical Details**: 연구에서는 '극우'(far-right) 및 '극좌'(far-left) 이념 키워드가 포함된 소셜 미디어 게시물을 수집하고 수동으로 극단주의 또는 비극단주의로 레이블을 붙였습니다. 극단주의 게시물은 정의적 틀에 기반한 다섯 가지 극단주의 요소로 추가 분류되었습니다. BERT 모델의 성능은 훈련 데이터 크기와 카테고리 간 지식 전이를 기준으로 평가했습니다. 또한, 다양한 프롬프트(naïve, layperson-definition, role-playing, professional-definition)를 사용하여 GPT 3.5와 GPT 4 모델의 성능을 비교했습니다.

- **Performance Highlights**: 최고 성과를 기록한 GPT 모델은 최고의 BERT 모델보다 나은 성능을 보였으며, 더 구체적인 프롬프트가 일반적으로 더 나은 결과를 가져왔습니다. 그러나 지나치게 복잡한 프롬프트는 성능을 저하시킬 수 있습니다. GPT 3.5는 극좌 극단주의 게시물 분류에서 더 좋은 성과를 거두었고, GPT 4는 극우 극단주의 게시물 분류에서 더 좋았습니다. GPT 모델은 전통적인 BERT 모델에 비해 온라인 극단주의 분류 작업에서 중요한 잠재력을 보이며, 제로샷(zero-shot) 설정에서 성능을 능가했습니다.



### Theoretical and Methodological Framework for Studying Texts Produced by Large Language Models (https://arxiv.org/abs/2408.16740)
- **What's New**: 이 논문은 대형 언어 모델(LLM)과 이들이 생성하는 텍스트를 정량 언어학의 관점에서 연구하는 데야하는 개념적, 방법론적, 기술적 과제를 다룹니다. LLM을 기초로 하는 모델과 시뮬레이션되는 개체를 구별하는 이론적 프레임워크를 제시하며, 인류 언어와 사고를 탐구하는 도구로서의 LLM의 가능성을 강조합니다.

- **Technical Details**: LLM을 탐구하기 위한 기본 가설 세트를 구축하고, 그 가설들이 어떻게 형성되는지에 대한 개념화를 제공합니다. 이론의 기본 요소로는 정보 이론, 변환기 아키텍처, 샘플링 및 토크나이제이션 방법이 포함되며, 현재 모델의 아키텍처는 많은 경우 비공식적입니다. LLM을 언어의 모델이 아닌 텍스트의 모델로 개념화하며, 인터폴레이션과 예측을 통해 새로운 텍스트를 생성하는 방식에 대해 설명합니다.

- **Performance Highlights**: LLM은 단순한 패턴 매칭을 넘어, 문화적 산물의 거대한 압축으로 생성된 잠재 공간에서 새로운 가능 세계를 창조해냅니다. LLM은 단순히 인간을 시뮬레이션하는 데 그친 것이 아닌, 우리가 상상할 수 없는 새로운 실체를 창조할 수 있는 가능성을 지니고 있습니다.



### Smaller, Weaker, Yet Better: Training LLM Reasoners via Compute-Optimal Sampling (https://arxiv.org/abs/2408.16737)
- **What's New**: 본 연구는 합성 데이터 생성에 있어 강력한 언어 모델(Strong Language Models, LMs) 대신 약하지만 저렴한 모델(Weak Model, WC)을 사용하는 것이 연산 비용 최적화(compute-optimal) 가능성이 있음을 제안합니다.

- **Technical Details**: 연구진은 WC 모델에서 생성된 데이터의 전체 문제 커버리지(coverage), 다양한 솔루션 수(diversity), 그리고 잘못된 추론을 가진 문제 비율(false positive rate, FPR)을 평가합니다. 결과적으로, WC 모델의 데이터는 높은 커버리지와 다양성을 보이지만 FPR 또한 상승합니다.

- **Performance Highlights**: 여러 실험에서 WC 데이터로 미세 조정된 모델이 SE 데이터로 훈련된 모델보다 평균 31.6%의 성능 향상을 보였으며, 이는 기존의 SE 모델 기반 접근 방식을 재고해야 함을 시사합니다.



### Enhancing Dialogue Generation in Werewolf Game Through Situation Analysis and Persuasion Strategies (https://arxiv.org/abs/2408.16586)
Comments:
          Accepted to the AIWolfDial2024 workshop at INLG 2024

- **What's New**: 이 논문은 LLM(대규모 언어 모델)을 기반으로 한 새로운 Werewolf Game AI를 소개하며, 이는 AIWolfDial2024 대회의 요구 사항을 충족하기 위한 강력한 기준을 제공합니다.

- **Technical Details**: AI는 상황 분석 모듈, 응답 생성 모듈, 그리고 특별히 설계된 설득 응답 생성 모듈을 포함하여, 역할에 따라 게임 내 대화를 수행합니다. 또한, 이 시스템은 논리적 설득, 신뢰 기반 설득, 그리고 감정 기반 설득 전략을 통해 좀비 역할의 설득력을 강화합니다.

- **Performance Highlights**: 새로운 모델은 멀티 턴 대화 프레임워크에서 복잡한 상호작용을 처리하여 더욱 자연스럽고 의미 있는 대화 생성을 통해 AI의 효과iveness를 향상시킵니다.



### Predictability maximization and the origins of word order harmony (https://arxiv.org/abs/2408.16570)
- **What's New**: 본 논문에서는 머리(head)와 그 의존어(dependents)의 순차적 배치 문제를 정보 이론(information theory) 관점에서 다룬다. 특히 머리의 최적 배치가 시퀀스의 예측 가능성(predictability)을 극대화하는 방법을 고찰한다.

- **Technical Details**: 독립적인 의존어들 간의 통계적 독립성을 가정하고, 머리를 마지막에 배치하는 것이 머리의 예측 가능성을 최대화한다는 점을 보여준다. 머리를 앞으로 가져오는 것은 의존어의 예측 가능성을 최대화하는 최적 전략이다. 이를 통해 'harmonic order'의 최적성을 입증한다.

- **Performance Highlights**: 이번 연구는 다양한 언어가 채택하고 있는 머리의 배치 방식을 새롭게 조명하며, 언어 간의 예측 가능성 극대화 전략을 비교 분석한다.



### SALSA: Speedy ASR-LLM Synchronous Aggregation (https://arxiv.org/abs/2408.16542)
Comments:
          Accepted to INTERSPEECH 2024

- **What's New**: SALSA라는 새로운 모델을 제안하여 저자들은 ASR(자동 음성 인식)와 LLM(대형 언어 모델)을 결합한 혁신적인 방법을 소개합니다. 이 모델은 ASR 디코더와 LLM 디코더를 결합하여 동기적으로 작동하며, 기존 접근 방식보다 훨씬 더 효율적으로 훈련할 수 있습니다.

- **Technical Details**: SALSA는 ASR 디코더와 LLM 디코더의 마지막 상태를 단순 프로젝션을 통해 연결합니다. ASR과 LLM 디코더는 각기 다른 토크나이저를 가지고 있지만, 이 문제를 해결하기 위해 계단식 토크나이저(cascading tokenization) 방식을 사용합니다.

- **Performance Highlights**: FLEURS 벤치마크에서 8개 저자원 언어를 평가한 결과, SALSA는 평균 16%의 WER(단어 오류율) 감소를 달성하였으며, 최대 38%의 WER 감소를 기록했습니다. 이는 기존의 접근 방식들보다 더 나은 성능을 보여줍니다.



### CNIMA: A Universal Evaluation Framework and Automated Approach for Assessing Second Language Dialogues (https://arxiv.org/abs/2408.16518)
- **What's New**: CNIMA(중국어 비원어민 상호작용 측정 및 자동화)라는 10,000개의 대화로 구성된 중국어 학습자 레이블 데이터셋을 개발했습니다. 본 연구는 기존에 영어-제2언어 대화용으로 개발된 평가 프레임워크를 중국어에 적용하여 다양하고 유용한 분석을 추가하였습니다.

- **Technical Details**: CNIMA 평가는 미시 수준(micro-level) 특성과 거시 수준(macro-level) 상호작용 레이블을 포함하여 전반적인 대화 품질 점수를 예측할 수 있는 완전 자동화된 접근 방식을 제안합니다. 우리는 고전적 기계 학습 모델(예: 로지스틱 회귀)과 대형 언어 모델(예: GPT-4o OpenAI(2024))을 사용하여 이러한 예측을 자동화했습니다.

- **Performance Highlights**: 우리의 시스템은 고유한 상호작용과 회화 유창성 분석을 통해 강력한 예측 성능을 입증하였으며, 이는 제2언어 평가를 위한 새로운 도구로 거듭나고 있습니다. 또한, 우리는 이 모델이 대규모 주석 데이터 없이 다른 언어에 쉽게 적용될 수 있음을 강조합니다.



### LLMs vs Established Text Augmentation Techniques for Classification: When do the Benefits Outweight the Costs? (https://arxiv.org/abs/2408.16502)
Comments:
          20 pages

- **What's New**: 본 연구는 Generative Large Language Models (LLMs)를 기존의 데이터 증강 방법들과 비교하여, LLM이 제공하는 명확한 비용-편익 (cost-benefit) 우위를 확인하는 연구가 부족하다는 점을 지적하며, 6개의 데이터셋과 3개의 분류기, 2개의 파인튜닝 방법을 사용하여 LLM 기반 증강 방법의 장단점을 실험적으로 분석하였다.

- **Technical Details**: LLM 기반 증강 방법(예: paraphrasing, word insertion 및 word swapping)은 작은 수의 seeds를 사용할 때만 전통적인 방법들과 비교하여 성능이 향상된다. 연구에서는 LLM 모델(GPT-3.5, Llama)과 기존의 방법(예: back-translation, BERT 기반 방법) 모두를 적용하여 267,300회의 파인튜닝을 수행하였다.

- **Performance Highlights**: 1) LLM 기반 증강 방법은 적은 수의 seeds를 사용할 때만 기존 방법을 초월하는 성능을 발휘하며, seeds 수가 증가할수록 비용 효율성이 떨어진다. 2) Robustness가 낮은 분류기(예: DistilBERT, BERT)의 정확도에 가장 큰 영향을 미친다. 3) 완전 파인튜닝에 비해 QLoRA 파인튜닝에서는 LLM 증강 방법이 더 높은 정확도에 영향을 미친다.



### Learning from Negative Samples in Generative Biomedical Entity Linking (https://arxiv.org/abs/2408.16493)
- **What's New**: 이번 연구에서는 ANGEL이라는 새로운 프레임워크를 도입하여 생물의학 개체 링크(BioEL) 모델의 훈련에 부정적인 샘플을 활용하는 최초의 방법을 제시합니다. 기존의 생성 모델이 긍정적인 샘플에만 의존했던 단점을 해결하고, 하드 네거티브 샘플을 명시적으로 학습하여 모델의 성능을 향상시킵니다.

- **Technical Details**: ANGEL 프레임워크는 두 단계를 포함합니다: 긍정 샘플만을 사용하는 훈련과 부정 샘플을 인식하는 훈련입니다. 첫 번째 단계에서는 주어진 입력 엔티티에 대해 지식 기반에서 긍정적인 샘플을 생성하도록 모델을 훈련시킵니다. 두 번째 단계에서는 모델의 상위 k개의 예측에서 올바른 출력과 올바르지 않은 출력을 모두 모읍니다. 이후, DPO(Direct Preference Optimization) 알고리즘을 통해 올바른 예측을 우선시하도록 모델을 업데이트합니다.

- **Performance Highlights**: ANGEL로 세밀하게 조정된 모델은 5개의 벤치마크에서 이전 최고의 기본 모델에 비해 최대 1.4% 향상된 상위-1 정확도를 보였으며, 사전 훈련(pre-training)을 포함할 경우 성능 향상은 1.7%에 달했습니다. 이러한 개선은 사전 훈련과 세밀한 조정 단계 모두에서 효과적임을 보여줍니다.



### Self-Alignment: Improving Alignment of Cultural Values in LLMs via In-Context Learning (https://arxiv.org/abs/2408.16482)
- **What's New**: 이 논문은 문화적 가치에 대한 기존 지식을 활용하여 대형 언어 모델(LLM)의 응답을 수정하는 방법을 연구합니다. 특히, 자가 정렬(self-alignment) 기술을 통해 여러 언어와 문화에 걸쳐 모델의 응답을 개선할 수 있음을 보여줍니다.

- **Technical Details**: 연구에서는 인-컨텍스트 학습(in-context learning, ICL)과 인간 설문 데이터를 조합한 간단한 방법을 제안하며, 이를 통해 5개 모델에서 문화적 가치와의 정렬을 개선하는 결과를 도출합니다. 이 방법은 World Values Survey(WVS) 데이터를 기반으로 하는 문화 가치 탐침을 사용하며, LLM이 문화 특정 값의 예시를 제공받을 때 응답을 조정하는 능력을 테스트합니다.

- **Performance Highlights**: 이 자가 정렬 방법은 영어 중심 모델뿐 아니라 다국어 LLM의 문화적 가치 탐색에서도 효과적이며, 다양한 언어에서의 응답 정렬을 개선합니다. 또한, 이 방법은 영어 외의 다른 언어 테스트에서도 유용하게 활용될 수 있습니다.



### Is text normalization relevant for classifying medieval charters? (https://arxiv.org/abs/2408.16446)
Comments:
          This preprint has not undergone peer review or any post-submission improvements or corrections

- **What's New**: 이 연구는 중세 문서의 분류에서 역사적 텍스트 정규화(historical text normalization)의 영향을 조사하며, 특히 문서 날짜 지정(dating)과 위치 파악(locating)에 초점을 맞추고 있습니다.

- **Technical Details**: 본 연구는 7,000개 이상의 중세 독일어(charters) 문서 데이터를 활용하여 전통적인 분류기(classifiers)와 트랜스포머 모델(transformer-based models)을 평가합니다. 정규화가 문서 분류(classification)에 미치는 영향을 분석하며, 지원 벡터 머신(support vector machines)과 그래디언트 부스팅(gradient boosting)이 다른 모델보다 우수한 성능을 보인 것을 확인했습니다.

- **Performance Highlights**: 결과적으로, 정규화는 위치 파악 작업에서는 약간의 성능 향상을 보이지만, 날짜 지정에서는 정확도를 낮춥니다. SVM 및 XGBoost 모델이 두 작업에서 가장 뛰어난 결과를 보였으며, 트랜스포머 모델은 정규화 적용 시 성능 저하를 겪었습니다.



### SurveySum: A Dataset for Summarizing Multiple Scientific Articles into a Survey Section (https://arxiv.org/abs/2408.16444)
Comments:
          15 pages, 6 figures, 1 table. Submitted to BRACIS 2024

- **What's New**: 이 논문은 SurveySum이라는 새로운 데이터셋을 소개하며, 여러 과학 기사를 요약하여 설문 섹션으로 만드는 작업을 지원합니다. 이는 특정 분야의 요약 도구에 대한 부족을 해결하기 위한 것입니다.

- **Technical Details**: 문서 요약(document summarization) 기술은 방대한 텍스트를 간결하고 유용한 요약으로 변환하는 과제입니다. 이 연구에서는 다문서 요약(multi-document summarization, MDS) 개념을 활용하여 여러 과학 기사의 정보를 통합하는 방법을 제시합니다. 또한, 두 가지 구체적인 파이프라인(pipeline)을 통해 과학 기사를 설문 섹션으로 요약하는 방법을 설명합니다.

- **Performance Highlights**: 본 연구 결과는 고품질 검색 단계(retrieval stages)의 중요성과 서로 다른 구성(configuration)이 생성된 요약의 품질에 미치는 영향을 강조합니다. 다양한 평가 지표(metrics)를 사용하여 파이프라인을 평가하여 성능을 비교하였습니다.



### Instruction-tuned Large Language Models for Machine Translation in the Medical Domain (https://arxiv.org/abs/2408.16440)
- **What's New**: 이번 연구에서는 의료 분야에서의 대형 언어 모델(Large Language Models, LLMs)과 명령어 튜닝된 LLMs(instruction-tuned LLMs)의 성능을 비교하고, 전문 의료 용어 사전을 데이터셋에 통합함으로써 LLM의 전반적인 번역 성능을 개선하고자 하였습니다.

- **Technical Details**: 연구는 FLAN-T5, LLaMA3, Tower 모델을 포함하여 영어-스페인어, 영어-독일어 및 영어-루마니아어 언어 쌍에 대해 평가하였습니다. 명령어 튜닝을 위해서는 매치된 의료 용어 쌍을 포함한 프롬프트를 이용해 파라미터 효율적인 미세 조정(Parameter-efficient fine-tuning, PEFT)을 적용하였습니다. 평가 지표로는 BLEU, chrF 및 COMET을 사용하여 성능을 비교하였습니다.

- **Performance Highlights**: 명령어 튜닝된 모델들이 가장 기초 모델들보다 주요 성능 지표에서 유의미한 성과(p<0.05)를 보여주었으며, Tower와 QLoRA Tower 모델은 영어-스페인어와 영어-독일어에서 가장 높은 성능을 발휘하였습니다.



### MQM-Chat: Multidimensional Quality Metrics for Chat Translation (https://arxiv.org/abs/2408.16390)
- **What's New**: 본 연구에서는 채팅 번역의 정확한 평가를 위한 새로운 지표인 MQM-Chat(Multidimensional Quality Metrics for Chat Translation)을 도입합니다. 이를 통해 5가지 모델을 평가하고, 각 모델의 기본적인 오류와 고유한 단점을 분석했습니다.

- **Technical Details**: MQM-Chat은 일곱 가지 오류 유형인 잘못된 번역(mistranslation), 생략 또는 추가(omission or addition), 용어 및 고유 명사 문제(terminology or proper noun issue), 부자연스러운 스타일(unnatural style), 모호성 및 명확화(ambiguity and disambiguation), 유행어 또는 외래어 문제(buzzword or loanword issue), 대화의 일관성(dialogue inconsistency)를 포함합니다. 이 오류들은 세 가지 심각도 수준으로 평가됩니다.

- **Performance Highlights**: 실험 결과, GPT-4가 다른 모델들을 능가하는 성능을 보였으며, 모든 모델에서 주로 발생하는 오류 유형은 잘못된 번역, 유행어 문제 및 대화의 일관성 오류였습니다. 특히 짧은 채팅에서는 이러한 문제가 더욱 중요하다는 것을 강조합니다.



### The Unreasonable Ineffectiveness of Nucleus Sampling on Mitigating Text Memorization (https://arxiv.org/abs/2408.16345)
Comments:
          9 pages, Accepted at INLG 2024 (International Natural Language Generation Conference)

- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)의 텍스트 암기 행동을 분석하면서, nucleus sampling 기법을 적용한 경우의 영향을 연구합니다. 연구자들은 이 기법이 암기 패턴의 발생을 줄일 수 있다고 가정하고, 이를 테스트하기 위해 중복 데이터를 포함한 진단 데이터셋을 생성했습니다.

- **Technical Details**: 연구자는 우선 OpenWebText 데이터셋에서 일정한 비율의 데이터를 샘플링하고, 이 데이터를 기반으로 중복을 인위적으로 추가하여 LLM의 텍스트 암기 행동을 정확히 측정할 수 있는 조건을 만들었습니다. nucleus sampling은 결정론적 방법보다 더 많은 다양성을 제공하는데, 이는 최고 확률의 토큰 중에서 샘플링을 허용하여 더 적게 생성될 가능성이 있는 토큰을 포함시키기 때문입니다.

- **Performance Highlights**: 연구 결과, nucleus sampling의 크기를 증가시키는 것이 암기를 완전히 해결하지 못하고, 심지어 큰 nucleus 크기에서도 예상보다 효과적이지 않다는 것을 발견했습니다. 이는 모델이 암기한 토큰이 여전히 출력 분포를 지배하여 중복 생성의 위험이 여전히 존재하는 것으로 나타났습니다.



### Critic-CoT: Boosting the reasoning abilities of large language model via Chain-of-thoughts Critic (https://arxiv.org/abs/2408.16326)
- **What's New**: Critic-CoT라는 새로운 프레임워크를 제안하여 LLM의 비판 능력을 System-2 스타일로 향상시키는 것을 목표로 하고 있습니다.

- **Technical Details**: Critic-CoT는 단계별 Chain-of-Thought (CoT) 비판 형식과 원거리 감독 (distant supervision) 데이터를 활용하여 인간 주석 없이 LLM의 비판 능력을 향상시키는 방법입니다.

- **Performance Highlights**: GSM8K 및 MATH 데이터셋에서 실험 결과, Critic-CoT 기반 모델이 잘못된 솔루션을 정확하게 필터링하고 반복적인 개선을 통해 작업 해결 성능을 크게 향상시키는 것을 확인했습니다.



### Physics of Language Models: Part 2.2, How to Learn From Mistakes on Grade-School Math Problems (https://arxiv.org/abs/2408.16293)
Comments:
          arXiv admin note: text overlap with arXiv:2407.20311

- **What's New**: 이 논문에서는 언어 모델의 정확성을 향상시키기 위해 'error-correction' 데이터를 프리트레이닝(pretraining) 단계에 직접 통합하는 유용성을 연구하였습니다. 이는 오류가 포함된 해결 단계가 즉시 수정된 데이터로 구성됩니다.

- **Technical Details**: 논문은 'retry data'(오류 및 즉각적인 수정)로 훈련된 언어 모델이 오류 수정을 잘 수행할 수 있는지를 조사합니다. 측정 방법으로는 iGSM 데이터셋을 사용하며, 이는 프로그램이 생성한 초등학교 수준의 수학 문제로 구성되어 있습니다. 연구는 모델이 잘못된 해결 단계를 감지하고, 그로부터 재생성하는 'retry upon regret' 프로세스를 나타냅니다.

- **Performance Highlights**: 수학 데이터셋에서 'retry upon regret' 프로세스가 개선된 정확도를 가져와, 기존의 beam search 방법에 비해 높은 성과를 보여주었습니다. 모델이 이미 수정할 필요가 있는 오류를 인식하는 경향이 있으며, 99% 이상의 오류 감지 정확도를 기록했습니다.



### Measuring the Accuracy of Automatic Speech Recognition Solutions (https://arxiv.org/abs/2408.16287)
- **What's New**: 이번 연구는 d/Deaf와 난청인(DHH)을 위한 자동 음성 인식(ASR) 기술의 접근성 및 정확성 문제를 조명합니다. 기존의 ASR 서비스는 높은 정확도를 자랑하지만, DHH 커뮤니티에서는 이러한 서비스에 대한 요구에 미치지 못하고 있습니다. 본 연구는 11가지 ASR 서비스의 성능을 독립적으로 평가하여, 기술 혁신이 실제 사용자 경험과 일치하지 않음을 보여줍니다.

- **Technical Details**: 본 연구에서는 다양한 ASR 서비스의 정확성을 평가하기 위해 고등 교육 강의의 녹음을 사용하였습니다. ASR의 정확도는 단어 오류율(Word Error Rate, WER)을 기준으로 측정되었으며, 스트리밍, 어휘 사용, 언어 간 차이와 같은 기술적 조건의 영향을 분석하였습니다. 자동화된 전사 과정과 광범위한 텍스트 정규화를 통해 높은 비교 가능성을 유지하였습니다.

- **Performance Highlights**: ASR의 정확도는 공급업체에 따라 크게 달라지며, 스트리밍 ASR의 품질은 현저히 낮았습니다. 연구 결과에 따르면 최근의 ASR 기술 개선에도 불구하고, 일반적인 서비스는 여전히 접근성 기준에 미치지 못하는 정확성을 보이고 있습니다.



### Enhancing AI-Driven Psychological Consultation: Layered Prompts with Large Language Models (https://arxiv.org/abs/2408.16276)
- **What's New**: 본 연구는 심리 상담의 접근성을 높이기 위해 GPT-4와 같은 대형 언어 모델(LLM)을 활용하는 새로운 방법을 제안합니다. 특히, 사용자 입력에 동적으로 적응하는 계층화된 프롬프트 시스템을 도입하여 심리 상담 서비스를 보강합니다.

- **Technical Details**: 계층화된 프롬프트 시스템이 사용자의 초기 우려 사항을 수집하기 위해 폭넓은 질문으로 시작한 후, 사용자의 문제를 깊이 파고들기 위한 특정한 맥락에 맞는 프롬프트로 이어집니다. 또한, 공감 기반의 프롬프트와 사례 기반 프롬프트를 개발하여 LLM의 감정 지능과 맥락 이해 능력을 향상시킵니다. 실험 데이터셋은 다양한 정신 건강 이슈와 사용자 상호작용을 포함하고 있습니다.

- **Performance Highlights**: 실험 결과, LLM의 응답 품질에 상당한 개선을 보였으며, 이는 AI 기반 심리 상담의 가능성을 제시합니다. 본 연구의 프롬프트 엔지니어링 기법은 정신 건강 지원에 대한 증가하는 수요를 충족시키기 위한 확장 가능하고 접근 가능한 솔루션을 제공합니다.



### LoraMap: Harnessing the Power of LoRA Connections (https://arxiv.org/abs/2408.16264)
Comments:
          13 pages, 9 figures, 5 tables

- **What's New**: 이번 논문은 여러 개의 Low-Rank Adaptation (LoRA) 간의 연결성을 강화하는 방법을 탐구하여 대규모 언어 모델(LLM)의 신뢰성 문제를 해결하고자 합니다. 특히 COVID-Fact 데이터를 기반으로 하여 새로운 추론 데이터셋을 생성하고 이를 통해 다양한 관점에서 정보를 추론할 수 있는 LoRA를 구체화합니다.

- **Technical Details**: 연구에서는 사실 확인을 위해 맞춤형으로 설계된 세 가지 추론 데이터셋(DifferenceCoT, EntityCoT, CorrectClaim)을 생성하고, LoRA를 각 데이터셋에 맞춰 미세 조정(fine-tuning)합니다. LoraMap을 도입하여 LoRA 간의 관계를 매핑(mapping)하고, 이는 인간의 뇌의 정보 처리 방식에서 영감을 받았습니다. 이 방법은 LoRA의 단순한 가중합(weighted sum) 대신 정보를 교환하는 방식으로 작동합니다.

- **Performance Highlights**: COVID-Fact 데이터셋에서 LoraMap은 기존의 LoraHub 방법보다 우수한 성능을 보이며, LoraConcat에 비해 훨씬 적은 파라미터로도 더 나은 결과를 제공합니다. 이는 LoraMap의 효율성과 연결성 강화를 강조합니다.



### Making the Most of your Model: Methods for Finetuning and Applying Pretrained Transformers (https://arxiv.org/abs/2408.16241)
Comments:
          PhD thesis

- **What's New**: 이 논문에서는 기존의 Transformer 기반 언어 모델에 대한 새로운 파인튜닝 방법론 두 가지를 제시합니다. 첫 번째 방법은 회귀 메커니즘을 추가하여 Transformer Decoder의 효율성을 개선하며, 두 번째 방법은 Masked Language Model(MLM)을 비자기회귀 (non-autoregressive) 방식의 Sequence-to-Sequence Transformer의 초기화에 사용할 수 있도록 합니다.

- **Technical Details**: 제안된 기술들은 NLG (Natural Language Generation)와 NLP (Natural Language Processing)의 다양한 작업에 적용될 수 있습니다. 두 번째 방법인 Conditional Beam Search는 출력이 중복되거나 비속성 (degenerate)이지 않은 높은 가능성을 갖는 NLG 모델 출력을 검색할 수 있게 해줍니다. 또한 Hidden State Optimization 기법은 모든 Transformer Decoder에 적용되어 추론 시 품질을 향상시킵니다.

- **Performance Highlights**: 이 논문에서 제안된 새로운 기술들은 기존 언어 모델들을 보다 효과적으로 사용할 수 있는 접근법을 제공함으로써 자연어 이해(Understanding) 작업을 넘어 생성적인 응용 (Generative Applications) 친구(또는 용도)로의 전환을 가능하게 합니다. 특히, 비자기회귀 방식의 모델에 대한 초기 화단계에서의 성능 개선을 보이고 있습니다.



### From cart to truck: meaning shift through words in English in the last two centuries (https://arxiv.org/abs/2408.16209)
Comments:
          7 pages, 1 figure

- **What's New**: 본 연구는 1800년부터 2000년까지의 역사적 단어 데이터를 기반으로 한 onomasiological(의미론적) 연구로, 시간에 따른 단어의 개념 표현 변화를 탐구합니다. 에너지, 운송, 오락, 컴퓨팅 분야의 변화와 이를 통해 언어와 사회적 변화 사이의 연관성을 제시합니다.

- **Technical Details**: 연구에서는 word2vec(skipgram) 사용하여 훈련된 diachronic word embeddings(시간적 단어 임베딩)을 활용하였으며, Orthogonal Procrustes(직교 프로크루스테스) 방법으로 단어 쌍을 정렬했습니다.

- **Performance Highlights**: 주요 발견 중 하나로, 1800년대의 단어 'ship'이 1990년대의 'aircraft'와 가장 유사한 임베딩을 가졌다는 점이 있습니다. 또한, 에너지 분야에서는 석탄과 증기가 석유 및 디젤의 유사 개념으로 등장했으며, 오락 분야에서는 초기 19세기 극장과 오페라가 현대의 시네마와 텔레비전과 연결되었다는 점이 주목할 만합니다.



### FRACTURED-SORRY-Bench: Framework for Revealing Attacks in Conversational Turns Undermining Refusal Efficacy and Defenses over SORRY-Bench (https://arxiv.org/abs/2408.16163)
Comments:
          4 pages, 2 tables

- **What's New**: 이번 논문은 MULTI-TURN 대화 공격에 대한 대형 언어 모델(LLM)의 안전성을 평가하기 위해 FRACTURED-SORRY-Bench라는 프레임워크를 소개합니다. 이는 기존 SORRY-Bench 데이터셋을 기반으로 하며, 유해한 쿼리를 무해한 서브 질문으로 분해하여 적대적 프롬프트를 생성하는 간단하면서도 효과적인 방법을 제공합니다.

- **Technical Details**: FRACTURED-SORRY-Bench는 주어진 쿼리를 4-7개의 개별 서브 질문으로 분해하고, 이 서브 질문들을 대화 형식으로 순차적으로 LLM에 제시합니다. 이 방법은 LLM의 컨텍스트 윈도우를 이용하고 서브 질문마다 명시적인 유해 언어를 피함으로써 내용 필터와 안전 메커니즘을 우회하는 것을 목표로 합니다.

- **Performance Highlights**: 이 연구는 GPT-3.5-Turbo에서 10.9배 증가한 공격 성공률(ASR)을 보였으며, GPT-4, GPT-4o, GPT-4o-mini 모델에서도 유의미한 증가를 보여줍니다. 연구 결과, 49.33%의 파편화된 프롬프트 세트가 원래의 유해한 의도를 성공적으로 전달했습니다.



### Evaluating Computational Representations of Character: An Austen Character Similarity Benchmark (https://arxiv.org/abs/2408.16131)
- **What's New**: 이 논문은 Jane Austen의 소설에서 캐릭터 유사성을 평가하기 위한 새로운 벤치마크인 AustenAlike를 제안하고, 이를 통해 다양한 특징 추출 파이프라인의 성능을 비교합니다.

- **Technical Details**: 논문에서는 캐릭터 유사성을 구조적, 사회적, 전문가 정의의 세 가지 기준으로 구분하여 평가합니다. BookNLP와 FanfictionNLP 파이프라인을 사용하여 이벤트, 발언, 속성을 추출하고, 이를 바탕으로 캐릭터 표현을 구축합니다.

- **Performance Highlights**: AustenAlike 벤치마크를 통해, 계산된 표현이 대체로 사회적 및 내러티브 역할에 따른 유사성을 포착하지만, 전문가의 평가에 비해 세부적인 유사성은 여전히 부족하다는 결과를 보여줍니다. GPT-4의 결과 또한 전문가가 식별한 유사성을 정확히 재현하는 데에 한계가 있음을 나타냅니다.



### Structured Event Reasoning with Large Language Models (https://arxiv.org/abs/2408.16098)
Comments:
          PhD thesis

- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)이 복잡한 사건을 추론하는 데 여전히 한계를 보이고 있으며, 이를 해결하기 위해 LLM과 사건의 구조적 표현을 결합하는 세 가지 접근법을 제안합니다.

- **Technical Details**: 제안된 접근법은 언어 기반 표현, 반상징적 표현, 완전 상징적 표현을 포함합니다. 첫 번째 접근법은 서브 사건의 관계를 학습하는 언어 기반 표현으로, LLM이 파인튜닝을 통해 학습할 수 있습니다. 두 번째 접근법은 개체의 상태를 예측하고 활용하는 반상징적 표현으로, LLM이 적은 수의 프롬프트로 처리할 수 있습니다. 세 번째 접근법은 구조화된 데이터를 통해 훈련된 LLM이 예측하고 상징적 해결사에 의해 실행될 수 있는 완전 상징적 표현입니다.

- **Performance Highlights**: 다양한 사건 추론 작업에서, 각 접근법은 엔드-투-엔드 LLM보다 우수한 성능을 발휘하며, 더 높은 해석력을 제공합니다. 이는 LLM과 구조적 표현 간의 시너지를 통해 사건 추론과 그 이상에 대한 해결책을 제시합니다.



### Is Personality Prediction Possible Based on Reddit Comments? (https://arxiv.org/abs/2408.16089)
- **What's New**: 이 논문에서는 개인의 성격 유형(Myers-Briggs Type Indicator, MBTI)과 그들이 작성한 텍스트 간의 상관관계를 탐구합니다. 저자들은 Reddit에서 MBTI로 라벨링된 댓글 데이터 세트를 수집하고, 이를 바탕으로 BERT(Bidirectional Encoder Representations from Transformers) 기반의 다양한 지도 학습 분류기를 구축하여 텍스트에 기반한 저자의 성격을 예측하려고 시도합니다.

- **Technical Details**: 프로젝트는 Reddit 플랫폼에서 수집한 사용자 댓글을 사용하며, ALBERT(Adaptation of BERT) 변환기 모델을 활용하여 이들을 분류합니다. 데이터 세트는 MBTI 관련 댓글과 비관련 댓글로 구분되며, 총 16개의 성격 클래스를 생성합니다. 데이터 전처리 과정에서는 HTML 아티팩트 제거 등의 작업이 포함됩니다.

- **Performance Highlights**: 연구 결과, 비록 수집한 데이터 세트의 비정제 성격 때문에 일부 문제를 겪었으나, 텍스트 기반 성격 분류에서 상당한 잠재성을 관찰할 수 있었습니다. 기존 머신러닝 알고리즘에 비해 높은 성능을 보여주는 다양한 분류 기법이 제안되었습니다.



### Using Large Language Models to Create AI Personas for Replication and Prediction of Media Effects: An Empirical Test of 133 Published Experimental Research Findings (https://arxiv.org/abs/2408.16073)
Comments:
          24 pages, 3 figures, 2 tables

- **What's New**: 이 보고서는 대규모 언어 모델(LLMs)이 발표된 메세지 효과 연구의 정확한 복제를 촉진할 수 있는 가능성을 분석합니다. 우리는 2023년 1월에서 2024년 5월 사이의 Journal of Marketing에서 14개의 논문에서 133개의 실험 결과를 복제함으로써 LLM 기반 참여자(페르소나)들을 테스트했습니다.

- **Technical Details**: 우리는 Viewpoints AI라는 새로운 소프트웨어 도구를 사용했습니다. 이 도구는 연구 설계(study designs), 자극(stimuli), 측정(measures)을 입력으로 받아 LLM이 특정한 유니크한 페르소나 샘플로 작동할 수 있도록 자동으로 프롬프트(prompts)를 생성하고 응답을 수집하여 최종 출력으로 완전한 데이터셋과 통계 분석을 제공합니다. 본 연구에서 사용된 LLM은 Anthropic의 Claude Sonnet 3.5입니다. 우리는 원래 인간 연구에서 보고된 것과 동일한 샘플 속성(sample attributes), 연구 설계(study designs), 자극(stimuli), 측정을 사용하여 19,447개의 AI 페르소나를 생성하였습니다.

- **Performance Highlights**: 우리 LLM 복제는 원본의 주요 효과(original main effects) 76% (111개 중 84개)를 성공적으로 재현하였으며, 이는 미디어 자극(media stimuli) 응답 연구의 AI 보조 복제가 강력한 가능성을 지닌다는 것을 보여줍니다. 상호작용 효과(interaction effects)를 포함할 경우, 전체 복제율은 68% (133개 중 90개)였습니다. 이 연구는 소셜 사이언스의 복제 위기(replication crisis), 샘플링 저항(sample robustness) 문제 해결, 다양한 미디어 자극에 대한 소비자 반응을 신속하게 테스트할 수 있는 능력에 대해 논의합니다.



### Using large language models to estimate features of multi-word expressions: Concreteness, valence, arousa (https://arxiv.org/abs/2408.16012)
- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)의 가능성을 조사하여, 다단어 표현(multi-word expressions)의 구체성(concreteness), 가치(valence), 흥분(arousal)을 정확하게 추정할 수 있는지 살펴보았습니다.

- **Technical Details**: 연구에서는 ChatGPT-4o의 구체성, 가치 및 흥분 예측 능력을 체계적으로 평가했습니다. 연구 1에서는 ChatGPT-4o가 다단어 표현에 대한 인간의 구체성 평가(r = .8)와 강한 상관관계를 보였습니다. 연구 2에서는 개별 단어의 가치와 흥분 평가를 재검토하였고, 기존 AI 모델과 비슷하거나 더 나은 성과를 나타냈습니다. 연구 3에서는 다단어 표현의 구체성과 흥분 분석을 확장하여 유망한 결과를 보였습니다.

- **Performance Highlights**: 이 연구 결과는 LLMs가 다단어 표현과 관련된 귀중한 심리언어학(psycholinguistic) 데이터를 생성할 가능성을 강조하며, 126,397개의 영어 단어와 63,680개의 다단어 표현에 대한 구체성, 가치 및 흥분의 AI 규범(nor)을 포함한 데이터 세트를 제공하였습니다.



### SAM2Point: Segment Any 3D as Videos in Zero-shot and Promptable Manners (https://arxiv.org/abs/2408.16768)
Comments:
          Work in progress. Online Demo: this https URL . Code: this https URL

- **What's New**: 새로운 연구인 SAM2Point는 Segment Anything Model 2 (SAM 2)를 활용하여 제로샷(zero-shot) 및 프롬프트(promptable) 3D 세분화를 수행하는 초기 탐험을 소개합니다. 이 모델은 3D 데이터를 다방향 비디오로 해석하여 추가 훈련이나 2D-3D 투영 없이 3D 공간에서 세분화를 가능하게 합니다.

- **Technical Details**: SAM2Point는 3D 포인트, 박스, 마스크와 같은 다양한 프롬프트를 지원하며, 복잡한 2D-3D 전환 없이도 3D 물체, 실내 장면, 실외 환경 및 원시 희소 LiDAR(raw sparse LiDAR)와 같은 다양한 시나리오에서 일반화할 수 있는 능력을 갖추고 있습니다. 구체적으로, SAM2Point는 3D 데이터를 비디오 형식으로 변환하기 위해 복셀화(voxelization)를 사용하고, 사용자가 제공하는 3D 프롬프트로부터 세분화 결과를 생성합니다.

- **Performance Highlights**: 여러 3D 데이터셋에서의 데모가 SAM2Point의 강력한 일반화 능력을 강조합니다. 예를 들어, Objaverse, S3DIS, ScanNet, Semantic3D, KITTI와 같은 데이터셋에서의 실험 결과는 모델의 견고한 성능을 보여줍니다. 또한, SAM2Point는 3D 세분화를 위한 SAM의 신뢰할 수 있는 구현을 제공하며, 이는 향후 프롬프트 가능한 3D 세분화 연구의 출발점이 될 수 있습니다.



### Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming (https://arxiv.org/abs/2408.16725)
Comments:
          10 pages

- **What's New**: 본 논문에서는 Mini-Omni를 소개합니다. Mini-Omni는 실시간 음성 상호작용이 가능한 오픈 소스 end-to-end 대화 모델로, 음성 입력과 출력을 모두 처리할 수 있는 최초의 모델입니다. 'Any Model Can Talk'라는 독창적인 교육 방법을 통해 기존 모델에 최소한의 수정만으로 음성 출력을 구현할 수 있습니다.

- **Technical Details**: Mini-Omni는 텍스트 지시 음성 생성 방법과 동시에 배치 병렬 처리 전략을 적용하여 성능을 향상시킵니다. 이 모델은 추가적인 TTS 시스템 없이 실시간 오디오 출력을 가능하게 하며, SNAC 인코더를 활용하여 고품질의 음성 출력을 보장합니다. 또한 VoiceAssistant-400K 데이터셋을 활용하여 음성 모델을 최적화할 수 있습니다.

- **Performance Highlights**: Preliminary 실험에 따르면 Mini-Omni는 전통적인 음성 합성과 관련된 여러 작업에서 강한 성능을 보여주었으며, 기존 모델의 언어 능력을 최소한으로 저하시키면서도 리얼타임 상호작용을 가능하게 합니다. 이 모델을 통해 사용자 경험을 크게 개선할 것으로 기대됩니다.



### Jina-ColBERT-v2: A General-Purpose Multilingual Late Interaction Retriever (https://arxiv.org/abs/2408.16672)
- **What's New**: 이번 연구에서는 정보 검색에 있어 효과적인 다중 벡터 밀집 모델인 ColBERT의 개선점을 제시합니다. Jina-ColBERT-v2 모델을 도입하여 훈련 파이프라인을 개선하고 다양한 다국어 데이터를 효과적으로 처리합니다.

- **Technical Details**: Jina-ColBERT-v2는 비지도 학습(weakly supervised) 데이터를 활용한 두 단계 튜닝 방식(contrastive tuning followed by fine-tuning)으로 훈련됩니다. 또한, 효율적인 추론을 위해 다양한 크기의 선형 프로젝션 헤드를 도입하였으며, Jina-XLM-RoBERTa와 같은 최적화된 백본을 사용하여 성능을 극대화했습니다.

- **Performance Highlights**: Jina-ColBERT-v2는 영어 및 다국어 검색 과제에서 강력한 성능을 보이며, 기존 모델과 비교해 최대 50%의 저장 공간을 절약합니다.



### Iterative Graph Alignmen (https://arxiv.org/abs/2408.16667)
Comments:
          12 pages, 4 figures

- **What's New**: 이 논문은 Iterative Graph Alignment (IGA)라는 새로운 비주얼 베이스의 정렬 알고리즘을 소개합니다. IGA는 교사 모델(VLM)이 Iterative Graph Prompting (IGP)을 사용하여 개념적 그래프와 참조 답변을 생성하고, 이를 통해 학생 모델(LLM)이 자신의 답변과 비교하여 지식 간의 격차를 파악하도록 합니다.

- **Technical Details**: IGA는 두 가지 주요 과정인 독창적인 사고 과정과 적응 학습 메커니즘을 통합하여 구성됩니다. 교사 모델은 논리 그래프를 시각적으로 구성하여 학생 모델이 자신의 답변과 비교하고 지식 격차를 식별할 수 있도록 합니다. 또, Self-Aligned Incremental Learning (SAIL) 접근 방식을 통해 각 단계별로 점차적으로 지원을 제공합니다.

- **Performance Highlights**: IGA 방법을 사용한 결과, Claude Sonnet 3.5에서 73.12%의 정렬 개선 효과를 보였으며, Llama3-8B-Instruct는 86.20%의 개선율을 기록하여 Claude Sonnet 3.5를 초과하는 성능을 입증했습니다.



### SSDM: Scalable Speech Dysfluency Modeling (https://arxiv.org/abs/2408.16221)
- **What's New**: 본 논문에서는 음성 불유창성 모델링(Speech Dysfluency Modeling)의 새로운 접근 방식을 제안한다. SSDM(Scalable Speech Dysfluency Modeling)은(1) 조음 동작을 활용한 확장 가능한 강제 정렬(Forced Alignment) 기법을 채택하고, (2) 연결주의 하위 시퀀스 정렬기(Connectionist Subsequence Aligner, CSA)를 도입하여 불유창성 정렬을 달성하며, (3) Libri-Dys라는 대규모 시뮬레이션 불유창성 말뭉치를 소개하고, (4) 대형 언어 모델(LLMs)의 힘을 활용하여 종단 간 시스템을 개발한다.

- **Technical Details**: 이 연구에서는 기본 물리 법칙에 기반한 음성 표현 학습을 재조명하고, 음성 불유창성 모델링을 위한 확장 가능한 표현인 신경 조음 동작 점수(Neural Articulatory Gestural Scores)를 제안한다. 연결주의 하위 시퀀스 정렬기(CSA)는 음향 표현과 텍스트를 연결하여 불유창성을 인식하는 차별 가능하고 확률론적 강제 정렬기를 만들어낸다.

- **Performance Highlights**: SSDM은 불유창성 모델링 분야에서 표준으로 자리 잡을 것으로 기대된다. 또한 Libri-Dys 데이터셋을 오픈 소스화하여 추가 연구를 촉진시킬 것이다.



### M4CXR: Exploring Multi-task Potentials of Multi-modal Large Language Models for Chest X-ray Interpretation (https://arxiv.org/abs/2408.16213)
- **What's New**: 이번 논문에서는 CXR(Chest X-ray) 해석을 강화하기 위해 M4CXR이라는 다중 모달(Modal) LLM을 제안합니다. 이 모델은 다양한 작업별 데이터 세트를 통합한 시각적 지침 따르기 데이터 세트에서 훈련되었습니다.

- **Technical Details**: M4CXR은 여러 작업을 지원하며, 특히 의학 보고서 생성(MRG), 시각적 그라운딩(Visual Grounding), 시각적 질문 응답(VQA)에서 탁월한 성능을 보입니다. 모델의 훈련 과정에서는 Chain-of-Thought(CoT) 프롬프트 전략을 사용하여 CXR 이미지에서 발견 사항을 식별하고 해당 보고서를 생성합니다.

- **Performance Highlights**: M4CXR은 다양한 MRG 시나리오에서 탁월한 임상 정확도를 달성하며, 시각적 그라운딩과 VQA에서도 Specialized 모델에 필적하는 성능을 발휘합니다. 양적 및 질적 평가 모두에서 M4CXR의 다재다능성을 강조하며, 일관된 임상 정확도를 유지합니다.



### ReXamine-Global: A Framework for Uncovering Inconsistencies in Radiology Report Generation Metrics (https://arxiv.org/abs/2408.16208)
- **What's New**: 이번 논문에서는 의료 영상 보고서의 AI 생성 품질을 평가하기 위해 LLM 기반의 새로운 메트릭 테스트 프레임워크인 ReXamine-Global을 개발하였습니다.

- **Technical Details**: ReXamine-Global은 다양한 작성 스타일과 환자 집단에서의 메트릭 일반화를 평가합니다. 주요 구조는 다기관 데이터 수집, 기준 텍스트 표준화, 오류가 포함된 텍스트 생성, 메트릭 적용 및 전문가 평가로 나뉩니다.

- **Performance Highlights**: 240개의 보고서를 분석한 결과, 7개의 기존 메트릭에서 심각한 일반화 격차를 발견하였고, GPT-4 기반 메트릭이 가장 우수한 성능을 보였습니다.



### Benchmarking Japanese Speech Recognition on ASR-LLM Setups with Multi-Pass Augmented Generative Error Correction (https://arxiv.org/abs/2408.16180)
Comments:
          submitted to SLT2024

- **What's New**: 이번 연구는 대형 언어 모델(LLM)을 기반으로 한 자동 음성 인식(ASR) 시스템을 위한 생성 오류 수정(GER)의 새로운 접근 방식을 제안합니다. 특히 일본어에 적용된 첫 번째 GER 벤치마크를 구축하여, 0.9-2.6k 텍스트 발화를 평가하고, 다양한 LLM의 출력을 조합하여 개선된 성능을 보여줍니다.

- **Technical Details**: 본 연구에서는 다중 시스템 가설을 입력 측에 통합하고 여러 LLM의 수정을 출력 측에 적용하여 결과를 병합하는 다중 패스 보강 생성 오류 수정(MPA GER) 방법을 도입합니다. LLM 기반 GER을 일본어 ASR 작업에 적용하고 N-best 가설에서의 두 번째 언어 모델링을 포함하여 실험했습니다.

- **Performance Highlights**: 실험 결과는 SPREDS-U1-ja 및 CSJ 데이터를 기반으로 한 제안된 방법들이 ASR 품질 및 일반화에서 성능 개선을 보였음을 입증합니다. 특히 MPA GER 접근 방식이 ASR의 오류를 감소시키고, 전체 인식 성능을 높이는 데 기여했음을 확인했습니다.



### Logic-Enhanced Language Model Agents for Trustworthy Social Simulations (https://arxiv.org/abs/2408.16081)
Comments:
          Source code: this https URL

- **What's New**: 새로운 연구 프레임워크인 Logic-Enhanced Language Model Agents (LELMA)를 소개합니다. 이 프레임워크는 대형 언어 모델(LLMs)의 신뢰도를 높이기 위한 혁신적인 접근 방식을 제공합니다.

- **Technical Details**: LELMA 프레임워크는 3개의 주요 구성 요소로 구성되어 있습니다: LLM-Reasoner는 전략적 reasoning을 생성하고, LLM-Translator는 자연어 reasoning을 논리 쿼리로 매핑하며, Solver는 이러한 쿼리를 평가합니다.

- **Performance Highlights**: LELMA는 Hawk-Dove 게임, Prisoner’s Dilemma, Stag Hunt와 같은 게임 이론적 시나리오에서 GPT-4 Omni와 Gemini 1.0 Pro의 reasoning 출력 정확도를 개선하는 데 높은 정확도를 보였습니다.



New uploads on arXiv(cs.IR)

### Jina-ColBERT-v2: A General-Purpose Multilingual Late Interaction Retriever (https://arxiv.org/abs/2408.16672)
- **What's New**: 이번 연구에서는 정보 검색에 있어 효과적인 다중 벡터 밀집 모델인 ColBERT의 개선점을 제시합니다. Jina-ColBERT-v2 모델을 도입하여 훈련 파이프라인을 개선하고 다양한 다국어 데이터를 효과적으로 처리합니다.

- **Technical Details**: Jina-ColBERT-v2는 비지도 학습(weakly supervised) 데이터를 활용한 두 단계 튜닝 방식(contrastive tuning followed by fine-tuning)으로 훈련됩니다. 또한, 효율적인 추론을 위해 다양한 크기의 선형 프로젝션 헤드를 도입하였으며, Jina-XLM-RoBERTa와 같은 최적화된 백본을 사용하여 성능을 극대화했습니다.

- **Performance Highlights**: Jina-ColBERT-v2는 영어 및 다국어 검색 과제에서 강력한 성능을 보이며, 기존 모델과 비교해 최대 50%의 저장 공간을 절약합니다.



### Transformers Meet ACT-R: Repeat-Aware and Sequential Listening Session Recommendation (https://arxiv.org/abs/2408.16578)
Comments:
          11 pages. Accepted by RecSys'2024, full paper

- **What's New**: 이번 논문에서는 사용자 행동 내 반복성을 고려한 음악 추천 시스템 PISA(Psychology-Informed Session embedding using ACT-R)를 소개합니다. 기존의 추천 시스템들이 간과한 반복적 행동을 다룸으로써, 음악 청취 데이터를 기반으로 다음 청취 세션에서 사용자가 어떤 곡을 들을지 예측하는 성능을 개선하였습니다.

- **Technical Details**: PISA는 Transformer 아키텍처를 이용하여 사용자의 청취 세션 및 곡에 대한 임베딩 표현을 학습합니다. 이는 Anderson의 ACT-R(Adaptive Control of Thought-Rational)에서 영감을 받은 attention 메커니즘을 활용하여 사용자의 동적이고 반복적인 행동 패턴을 포착하고, 이를 통해 반복적이거나 새로운 곡들을 효과적으로 추천합니다.

- **Performance Highlights**: PISA의 실험은 Last.fm과 Deezer의 공개 데이터셋을 사용하여 진행되었으며, 반복 모델링이 음악 추천의 정확도에 미치는 중요성을 입증하였습니다. 이러한 접근 방식은 추천 시스템 연구의 새로운 도전을 제시하며, 관련 데이터셋과 소스 코드를 공개하여 연구 커뮤니티에 기여하고자 합니다.



### Do Recommender Systems Promote Local Music? A Reproducibility Study Using Music Streaming Data (https://arxiv.org/abs/2408.16430)
- **What's New**: 이 논문은 추천 시스템이 지역 음악 표현에 미치는 영향을 다루며, LFM-2b 공개 데이터셋을 기반으로 한 이전 연구 결과를 검토합니다. 사용자 행동이 지역 음악 소비에 미치는 다양한 영향을 분석하여, 추천 시스템에 내재된 알고리즘 편향을 강조하고 있습니다.

- **Technical Details**: 추천 시스템(Recommendation Systems)은 Apple Music, Deezer, Spotify와 같은 음악 스트리밍 서비스에서 기본적인 역할을 하며, 사용자에게 가장 관련성 높은 콘텐츠를 제공하는 데 도움을 줍니다. 본 연구는 Deezer의 전 세계 청취 데이터를 바탕으로 LFM-2b 데이터셋과 비교 분석하였으며, 각 추천 시스템에 따라 지역 음악의 소비 패턴이 어떻게 달라지는지를 보여줍니다.

- **Performance Highlights**: 본 논문에서 발표된 결과에 따르면, LFM-2b 데이터셋과 Deezer 데이터셋 간에 지역 음악 소비 패턴에서 유의미한 차이가 발견되었습니다. NeuMF와 ItemKNN은 각 데이터셋에서 지역 음악에 대해 서로 다른 알고리즘 편향을 보였으며, 레이블링의 정확성이 지역 음악 소비에 미치는 영향을 강조하고 있습니다.



### SynDL: A Large-Scale Synthetic Test Collection (https://arxiv.org/abs/2408.16312)
Comments:
          9 pages

- **What's New**: 이 논문에서는 대규모 정보 검색(test collection)을 위해, TREC Deep Learning Track의 테스트 컬렉션을 확장하여 대규모 ad-hoc 문서 검색 데이터셋을 개발하고, 이를 통해 연구자들이 대규모의 검색 시스템을 테스트하고 평가할 수 있도록 합니다. 이의 주요 혁신은 Large Language Models (LLMs)를 활용하여 생성된 합성 라벨을 포함하여 1,900개 이상의 테스트 쿼리를 제공하는 것입니다.

- **Technical Details**: 이 연구는 SynDL이라 불리는 대규모 테스트 컬렉션을 개발하며, 이는 LLM의 판단을 사용하여 정보 검색 시스템의 성능 평가에 심도 있는 관련성을 부여합니다. SynDL은 2019년부터 TREC Deep Learning 트랙에서 제공된 정보를 기반으로 하며, 쿼리와 문서 간의 복잡한 관계를 모델링하는 데 필요한 깊은 관련성(labels)을 제공합니다.

- **Performance Highlights**: 이 연구를 통해 생성된 SynDL 테스트 컬렉션은 기존 인간 라벨과 심하게 상관관계가 있어, 검색 시스템의 순위 평가에 있어 높은 일관성을 보여줍니다. LLM의 사용은 인적 자원의 소모를 줄이면서도 동등 이상의 성과를 낼 수 있음을 입증합니다.



### Efficient Transfer Learning Framework for Cross-Domain Click-Through Rate Prediction (https://arxiv.org/abs/2408.16238)
- **What's New**: 이번 연구는 자연 콘텐츠와 광고 간의 지식 전이를 위한 효율적인 전이 학습 프레임워크인 E-CDCTR를 제안합니다. 이 프레임워크는 CTR(Click-Through Rate) 모델의 데이터 희소성과 '재앙적 망각' 문제를 해결하여 광고 모델의 성능을 향상시키는 데 초점을 맞추고 있습니다.

- **Technical Details**: E-CDCTR는 세 가지 핵심 구성 요소인 Tiny Pre-training Model (TPM), Complete Pre-training Model (CPM), 그리고 Advertisement CTR model (A-CTR)로 구성됩니다. TPM은 장기 자연 데이터에서 기본 특징을 활용하여 소형 CTR 모델을 훈련하고, CPM은 광고와 동일한 구조와 입력 특징을 가진 CTR 모델을 단기 자연 데이터로 훈련하여 광고 CTR 모델의 초기화를 지원합니다. 이러한 구조는 대량의 자연 콘텐츠 데이터로 인한 비효율성을 줄이고, 잦은 업데이트로 인한 망각 문제를 완화하는 데 기여합니다.

- **Performance Highlights**: E-CDCTR는 광고 데이터에서 CTR 및 Revenue Per Mile (RPM) 각각 2.9% 및 2.1%의 상대적 향상을 보여주었습니다. 이러한 실험 결과는 제안된 프레임워크가 광고 모델의 성능을 효과적으로 개선하는 데 기여함을 입증합니다.



### Ranking evaluation metrics from a group-theoretic perspectiv (https://arxiv.org/abs/2408.16009)
- **What's New**: 이 논문에서는 새로운 모델을 검증하기 위한 가장 적절한 메트릭을 찾는 어려움에 대해 다루고 있습니다. 다양한 상황에 적용할 수 있는 보편적인 메트릭이 없고, 순위를 비교할 때 발생하는 문제들을 분석합니다.

- **Technical Details**: 순위 평가 메트릭이 서로 다른 측면을 부각시키는 경우가 많습니다. 이 논문은 수학적 대칭군(symmetric group)의 형식을 사용하여 순위의 수학적 특성을 체계적으로 설정하고, 이는 다양한 평가에 일관된 기준을 제공하는 데 기여합니다.

- **Performance Highlights**: 부적절한 메트릭 사용으로 인해 발생하는 불일치성(inconsistency)의 사례를 제시하며, 이러한 불일치성이 순위 평가 방법의 신뢰성을 저하시킨다는 점을 강조합니다. 논문은 순위 메트릭이 갖추어야 할 이론적 속성을 제시하며, 추후 메트릭 선택 시 신중해야 함을 강조합니다.



### Is text normalization relevant for classifying medieval charters? (https://arxiv.org/abs/2408.16446)
Comments:
          This preprint has not undergone peer review or any post-submission improvements or corrections

- **What's New**: 이 연구는 중세 문서의 분류에서 역사적 텍스트 정규화(historical text normalization)의 영향을 조사하며, 특히 문서 날짜 지정(dating)과 위치 파악(locating)에 초점을 맞추고 있습니다.

- **Technical Details**: 본 연구는 7,000개 이상의 중세 독일어(charters) 문서 데이터를 활용하여 전통적인 분류기(classifiers)와 트랜스포머 모델(transformer-based models)을 평가합니다. 정규화가 문서 분류(classification)에 미치는 영향을 분석하며, 지원 벡터 머신(support vector machines)과 그래디언트 부스팅(gradient boosting)이 다른 모델보다 우수한 성능을 보인 것을 확인했습니다.

- **Performance Highlights**: 결과적으로, 정규화는 위치 파악 작업에서는 약간의 성능 향상을 보이지만, 날짜 지정에서는 정확도를 낮춥니다. SVM 및 XGBoost 모델이 두 작업에서 가장 뛰어난 결과를 보였으며, 트랜스포머 모델은 정규화 적용 시 성능 저하를 겪었습니다.



### Rethinking Sparse Lexical Representations for Image Retrieval in the Age of Rising Multi-Modal Large Language Models (https://arxiv.org/abs/2408.16296)
Comments:
          Accepted to ECCV 2024 Workshops: 2nd Workshop on Traditional Computer Vision in the Age of Deep Learning (TradiCV)

- **What's New**: 본 논문은 이미지 검색을 위한 희소( sparse ) 어휘 표현을 재고하며, 시각적 프롬프트를 지원하는 다중 모달 대형 언어 모델( M-LLMs )을 활용하여 이미지 특징을 추출하고 이를 텍스트 데이터로 변환합니다. 이러한 방법을 통해 자연어 처리에서 사용되는 희소 검색 알고리즘을 이미지 검색 작업에 적용할 수 있도록 합니다.

- **Technical Details**: 연구에서는 키 확장을 위한 데이터 증강 기법을 적용하고, 이미지와 텍스트 데이터 간의 연관성을 평가하여 M-LLM의 이미지 특징 추출을 지원합니다. 또한, 키워드를 검색 쿼리에 반복적으로 추가함으로써 검색 성능이 향상된다는 점을 입증하였습니다.

- **Performance Highlights**: MS-COCO, PASCAL VOC, NUS-WIDE 데이터셋을 기반으로 한 키워드 기반 이미지 검색 시나리오에서 본 시스템은 기존의 비전-언어 모델 기반 방법들보다 우수한 정확도와 재현율( precision and recall ) 성능을 보였습니다. 특히, 키워드를 효과적으로 쿼리에 통합하는 방식으로 성능이 대폭 향상되었습니다.



### Efficient $k$-NN Search in IoT Data: Overlap Optimization in Tree-Based Indexing Structures (https://arxiv.org/abs/2408.16036)
Comments:
          28 pages, 21 figures, 1 table

- **What's New**: 본 논문은 데이터 공간 파티션(Partition) 중 겹침 문제를 해결하기 위해 세 가지 혁신적인 휴리스틱(Heuristics) 방법을 제안합니다. 이 방법들은 데이터 공간 겹침을 정량화하고 효율적으로 관리할 수 있도록 설계되었습니다.

- **Technical Details**: 제안된 방법은 다음과 같습니다: 
1. 볼륨 기반 방법(Volume-Based Method, VBM) - 파티션 간의 교차 볼륨(Intersection Volume)을 계산하여 겹침을 측정합니다.
2. 거리 기반 방법(Distance-Based Method, DBM) - 파티션 중심과 반경 간의 거리를 분석하여 효율성을 향상시킵니다.
3. 객체 기반 방법(Object-Based Method, OBM) - 파티션 간 공유 객체 수를 세어 직관적으로 겹침을 이해할 수 있게 합니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안된 방법들이 검색 시간을 단축시키며, 데이터 공간 파티셔닝과 전반적인 시스템 성능을 향상시키는 데 효과적임을 보여줍니다.



### An Extremely Data-efficient and Generative LLM-based Reinforcement Learning Agent for Recommenders (https://arxiv.org/abs/2408.16032)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)을 기반으로 한 강화 학습(RL) 에이전트를 WebShop 환경에서 훈련시키고 평가하는 새로운 방법론을 발표합니다. 이 방법론은 특히 e-commerce와 추천 시스템에서 인간의 지침을 이해하고 수행하는 능력을 발전시킵니다.

- **Technical Details**: 연구에서는 다양한 RL 알고리즘을 사용하여 WebShop 시뮬레이터에서 에이전트를 훈련합니다. BERT 모델을 미세 조정하여 RL 에이전트를 만들고, Proximal Policy Optimization (PPO) 및 Direct Preference Optimization (DPO) 등 최신 훈련 기술을 적용하였습니다. 또한, 생성 경로를 사용하는 RL 에이전트를 평가하여 사람의 경로를 기반으로 한 경우와 유사한 성능을 보여주었습니다.

- **Performance Highlights**: 훈련 시간(<2시간) 동안 이미지 없이 DPO 에이전트는 약 3000 스텝 후 19%의 성공률을 달성하였으며, PPO 에이전트는 15%로 상대적으로 낮은 성공률을 보였습니다. 이는 RL 에이전트 훈련의 데이터 효율성을 강조합니다.



New uploads on arXiv(cs.CV)

### 3D Whole-body Grasp Synthesis with Directional Controllability (https://arxiv.org/abs/2408.16770)
- **What's New**: CWGrasp라는 Novel 방법을 제안하여, 손과 신체의 자연스러운 상호 작용을 기반으로 3D 전신을 합성하는 문제를 해결합니다. 이 방법은 지오메트리 기반의 사고를 조기에 수행하여 제어 신호를 풍부하게 제공합니다.

- **Technical Details**: CWGrasp는 주로 두 가지 주요 모델을 포함합니다: CReach(손목 방향과 3D 팔 방향을 고려한 SMPL-X 신체 생성) 및 CGrasp(원하는 손바닥 방향에 따라 MANO 손 생성). 이들은 ReachingField 모델의 3D 방향 벡터를 통해 조건화되어 서로 호환성이 높은 신체와 손을 생성합니다.

- **Performance Highlights**: CWGrasp는 FLEX 방법에 비해 약 16배 빠른 성능을 보이며, 오른손과 왼손의 그립을 모두 처리할 수 있습니다. 평가 결과, CWGrasp가 다른 기본 모델들과 비교하여 더 높은 정확도를 보여줍니다.



### PromptSmooth: Certifying Robustness of Medical Vision-Language Models via Prompt Learning (https://arxiv.org/abs/2408.16769)
Comments:
          Accepted to MICCAI 2024

- **What's New**: 본 논문에서는 PromptSmooth라는 새로운 프레임워크를 제안하여 의료 비전-언어 모델(Med-VLMs)의 효율적인 certified robustness를 달성하고자 합니다. 이 프레임워크는 prompt learning의 개념을 활용하여 Gaussian noise에 잘 대응할 수 있도록 Med-VLM을 조정합니다.

- **Technical Details**: PromptSmooth는 사전 훈련된 Med-VLM을 기반으로 하여 zero-shot 또는 few-shot 방법으로 텍스트 프롬프트를 학습합니다. 이 방법은 정확성과 로버스트니스(robustness) 간의 미세한 균형을 유지하면서도 계산 오버헤드를 최소화합니다. 또한, PromptSmooth는 단일 모델로 여러 개의 노이즈 수준을 처리할 수 있어 각 노이즈 수준에 대해 별도의 모델을 훈련해야 하는 전통적인 방법에 비해 계산 비용을 크게 절감합니다.

- **Performance Highlights**: 세 가지 Med-VLM과 다양한 이미징 모달리티의 여섯 개 데이터셋을 바탕으로 한 포괄적인 실험 결과, PromptSmooth의 효능이 입증되었습니다.



### SAM2Point: Segment Any 3D as Videos in Zero-shot and Promptable Manners (https://arxiv.org/abs/2408.16768)
Comments:
          Work in progress. Online Demo: this https URL . Code: this https URL

- **What's New**: 새로운 연구인 SAM2Point는 Segment Anything Model 2 (SAM 2)를 활용하여 제로샷(zero-shot) 및 프롬프트(promptable) 3D 세분화를 수행하는 초기 탐험을 소개합니다. 이 모델은 3D 데이터를 다방향 비디오로 해석하여 추가 훈련이나 2D-3D 투영 없이 3D 공간에서 세분화를 가능하게 합니다.

- **Technical Details**: SAM2Point는 3D 포인트, 박스, 마스크와 같은 다양한 프롬프트를 지원하며, 복잡한 2D-3D 전환 없이도 3D 물체, 실내 장면, 실외 환경 및 원시 희소 LiDAR(raw sparse LiDAR)와 같은 다양한 시나리오에서 일반화할 수 있는 능력을 갖추고 있습니다. 구체적으로, SAM2Point는 3D 데이터를 비디오 형식으로 변환하기 위해 복셀화(voxelization)를 사용하고, 사용자가 제공하는 3D 프롬프트로부터 세분화 결과를 생성합니다.

- **Performance Highlights**: 여러 3D 데이터셋에서의 데모가 SAM2Point의 강력한 일반화 능력을 강조합니다. 예를 들어, Objaverse, S3DIS, ScanNet, Semantic3D, KITTI와 같은 데이터셋에서의 실험 결과는 모델의 견고한 성능을 보여줍니다. 또한, SAM2Point는 3D 세분화를 위한 SAM의 신뢰할 수 있는 구현을 제공하며, 이는 향후 프롬프트 가능한 3D 세분화 연구의 출발점이 될 수 있습니다.



### ReconX: Reconstruct Any Scene from Sparse Views with Video Diffusion Mod (https://arxiv.org/abs/2408.16767)
Comments:
          Project page: this https URL

- **What's New**: ReconX라는 새로운 3D 장면 재구성 패러다임을 제안하여 불분명한 재구성 문제를 시간 생성(task of temporal generation) 문제로 재구성합니다. 이는 대규모로 사전 훈련된 비디오 확산 모델(video diffusion models)의 강력한 생성적 프라이어를 활용하여 희소한 시각으로부터 재구성 작업에 활용합니다.

- **Technical Details**: ReconX는 제한된 입력 뷰를 사용하여 전역 포인트 클라우드를 구축하고 이를 맥락 공간(contextual space)으로 인코딩하여 3D 구조 조건(3D structure condition)으로 사용합니다. 그러면 비디오 확산 모델이 이 조건을 기반으로 세부사항을 보존하면서도 높은 3D 일관성을 가진 비디오 프레임을 생성합니다. 마지막으로, 생성된 비디오에서 3D 장면을 회수하기 위해 3D 가우시안 스플래팅(Gaussian Splatting) 최적화 방식을 사용합니다.

- **Performance Highlights**: 다양한 실제 세계 데이터셋에서 진행한 실험에서 ReconX는 기존 최첨단 방법들보다 품질(resolution)과 일반화 능력(generalizability)에서 우수함을 입증하였습니다.



### CSGO: Content-Style Composition in Text-to-Image Generation (https://arxiv.org/abs/2408.16766)
- **What's New**: 본 논문은 스타일 전이(style transfer) 분야에서 큰 규모의 데이터셋을 제시하고, 이를 활용하는 새로운 모델 CSGO(Content-Style Generation Organizer)를 제안합니다. 기존의 방법들이 데이터 부족으로 어려움을 겪었던 반면, IMAGStyle 데이터셋을 구축하여 이러한 한계를 극복하고자 합니다.

- **Technical Details**: 제안된 CSGO 모델은 콘텐츠와 스타일 기능들을 독립적으로 주입하여 스타일 전이를 수행하는 엔드 투 엔드(end-to-end) 훈련 방식을採用합니다. IMAGStyle 데이터셋은 210,000개의 콘텐츠-스타일-스타일 화상 삼중(triplets)으로 구성되어 있으며, 논문에서는 이 데이터셋을 사용하여 고품질 이미지를 생성합니다. 또한 Content Alignment Score (CAS)를 도입하여 스타일 전이 후 콘텐츠 손실의 정도를 측정합니다.

- **Performance Highlights**: CSGO는 다양한 형태의 스타일 전이에 대해 뛰어난 성능을 보이며, 기존의 방법들과 비교하여 제로샷(zero-shot) 스타일 전이가 가능함을 보여주었습니다. 실험 결과, 제안된 방법은 스타일 제어 능력을 크게 향상시키는 것으로 확인되었습니다.



### UV-free Texture Generation with Denoising and Geodesic Heat Diffusions (https://arxiv.org/abs/2408.16762)
- **What's New**: 이번 논문에서는 전통적인 UV(UV-mapping) 방식 대신, 3D 객체의 곡면 직접에서 색깔이 입혀진 포인트 클라우드(point-cloud) 형태로 텍스처를 생성하는 방법을 제안합니다. 이를 통해 UV 맵의 왜곡, 이음새, 낭비된 UV 공간 문제를 해결할 수 있습니다.

- **Technical Details**: 포인트 클라우드 텍스처는 노이즈 제거 확산 확률 모델(denoising diffusion probabilistic model)을 사용하여 메쉬의 표면에서만 작동하며, 열 확산(heat diffusion)을 기반으로 점 간의 공간적 통신을 제공합니다. 새로운 자기 주의(self-attention) 메커니즘을 도입하여 넓은 영역의 통신을 가능하게 하고, 혼합 라플라시안 연산자를 통해 위상 오류 및 연결되지 않은 컴포넌트가 존재해도 열 확산이 가능하도록 합니다.

- **Performance Highlights**: 제안된 방법은 UV 맵에 의존하지 않기 때문에 메쉬의 리메싱(remeshing) 작업이 필요 없으며, 다양한 샘플링 해상도에 대처할 수 있습니다. 또한, 환경을 고려하지 않은 반사 텍스처(albedo textures)를 생성하여 조명 조건 변화에 더 감동적인 결과를 가져옵니다. 이 방식은 여러 클래스의 객체를 포함하는 데이터셋에 대해서도 학습이 가능합니다.



### OmniRe: Omni Urban Scene Reconstruction (https://arxiv.org/abs/2408.16760)
Comments:
          See the project page for code, video results and demos: this https URL

- **What's New**: OmniRe를 도입합니다. 이는 장치 로그에서 고충실도(dynamic) 도시 장면을 효율적으로 재구성하는 포괄적인 접근 방식입니다.

- **Technical Details**: OmniRe는 동적 신(scene) 그래프를 기반으로 하며, 차량, 보행자(pedestrians), 자전거 이용자(cyclists) 등 다양한 동적 객체를 모델링하는 여러 지역적 표준 공간(local canonical spaces)을 구축합니다. 기존 방법들이 놓쳤던 비차량 동적 행위자(non-vehicle dynamic actors)를 포함하여 전체 장면을 재구성할 수 있는 능력을 가지고 있습니다.

- **Performance Highlights**: Waymo 데이터셋에서 광범위한 평가를 통해, 우리의 방법이 기존 최첨단(state-of-the-art) 방법보다 정량적 및 정성적으로 큰 차이로 성능을 향상시킨 것을 확인했습니다.



### Dissecting Out-of-Distribution Detection and Open-Set Recognition: A Critical Analysis of Methods and Benchmarks (https://arxiv.org/abs/2408.16757)
Comments:
          Accepted to IJCV, preprint version

- **What's New**: 이번 논문은 머신러닝 모델의 테스팅 시 발생할 수 있는 분포 변화(Distribution Shift)를 탐지하는 기술적 중요성을 강조하며, OOD(Out-of-Distribution) 탐지와 OSR(Open-Set Recognition)이라는 두 개의 주요 하위 분야의 특성을 종합적으로 분석합니다. 특정 벤치마크를 제안하고, 다양한 방법론의 성능을 비교하여 실질적인 통찰력을 제공합니다.

- **Technical Details**: 이研究에서는 OOD 탐지와 OSR 방법의 철저한 교차 평가를 수행하며, 이 두 분야의 방법들 간의 성능 상관관계를 식별합니다. 또한, OOD 탐지와 OSR에서 발생하는 분포 변화를 효과적으로 분리하는 새로운 대규모 벤치마크 설정을 제안합니다. 본 연구는 Outlier Exposure(OE)과 같은 최신 방법들이 실제 대규모 환경에서 어떻게 성능이 저하되는지를 설명하고, 그 대안으로 깊은 특징의 크기에 민감한 스코어링 규칙을 주요 성과로 제시합니다.

- **Performance Highlights**: OE는 표준 작은 벤치마크에서는 좋은 성능을 보였으나, 대규모 환경에서는 성능이 떨어지는 것을 발견했습니다. 반면, MLS(Maximum Logit Score)와 같은 깊은 특징의 크기에 민감한 스코어링 규칙은 여전히 유망한 성과를 보였습니다. 이러한 발견은 OOD 탐지 및 OSR 방법 간의 관계를 더욱 명확히 하고, 향후 연구 방향을 제시하는 계기가 됩니다.



### VideoLLM-MoD: Efficient Video-Language Streaming with Mixture-of-Depths Vision Computation (https://arxiv.org/abs/2408.16730)
- **What's New**: 이 논문은 VideoLLM-MoD라는 새로운 방법론을 제안하여 비디오 데이터의 비전(vision) 토큰 수를 줄이지 않고도 계산 효율성을 극대화합니다. 이 방법은 고비율의 비전 토큰을 건너 뛰는 방식으로 작동합니다.

- **Technical Details**: VideoLLM-MoD는 각 Transformer 레이어마다 비전 토큰의 약 80%를 건너뛰고, 나머지 중요한 토큰만을 다음 레이어에 전달하여 계산 비용을 줄입니다. 이러한 방식을 통해 전체 훈련 시간에서 약 42%의 시간과 30%의 메모리 절약을 달성합니다. 이 모델은 LayerExpert 모듈을 통해 각 레이어에서 어떤 비전 토큰이 처리되어야 하는지를 학습합니다.

- **Performance Highlights**: VideoLLM-MoD는 COIN, Ego4D, Ego-Exo4D 데이터셋에서 내러티브(narration), 예측(forecasting), 요약(summarization) 작업에 대해 최첨단 결과를 보여주며, 계산 효율성을 유지하면서도 성능을 개선합니다.



### Prediction-Feedback DETR for Temporal Action Detection (https://arxiv.org/abs/2408.16729)
- **What's New**: 이 논문은 Temporal Action Detection (TAD)에서의 DETR 기반 방법론에서 발생하는 주의력 붕괴(attention collapse) 문제를 다루고, 이를 해결하기 위해 새로운 Prediction-Feedback DETR (Pred-DETR) 프레임워크를 제안합니다. 이 접근 방식은 예측(prediction)을 활용하여 붕괴된 주사위를 복원하고 교차 주의력(cross-attention)과 자기 주의력(self-attention)의 정렬을 목표로 합니다.

- **Technical Details**: 기존 연구를 바탕으로, 이 연구는 DETR 기반 TAD 방법에서 교차 주의력의 주의력 붕괴를 새롭게 조명합니다. 새로운 프레임워크인 Pred-DETR은 예측 피드백(prediction-feedback) 목표를 도입하여 예측 간의 관계를 통해 주의력을 회복합니다. 특히, 이 방법은 IoU(union-over-intersection) 유사성 맵을 사용하여 자기 관계와 교차 관계를 정렬합니다.

- **Performance Highlights**: Pred-DETR은 THUMOS14, ActivityNet-v1.3, HACS, FineAction 등의 다양한 벤치마크에서 최신 성능을 달성하며, 전체 주의력 메커니즘의 붕괴를 효과적으로 완화하여 성능 향상을 가져왔습니다.



### H-SGANet: Hybrid Sparse Graph Attention Network for Deformable Medical Image Registration (https://arxiv.org/abs/2408.16719)
- **What's New**: 이번 연구는 경량 하이브리드 희소 그래프 어텐션 네트워크(H-SGANet)를 소개합니다. H-SGANet은 뇌 MRI 볼륨의 해부학적 연결성을 고려하여, ConvNet, ViG, Transformer의 장점을 결합한 모델입니다.

- **Technical Details**: H-SGANet은 Sparse Graph Attention(SGA) 모듈을 포함하여 다양한 해부학적 영역 간의 연결을 형성합니다. 또한, Separable Self-Attention(SSA) 메커니즘을 도입하여 기존의 다중 헤드 셀프 어텐션(MHA) 컴포넌트를 효과적으로 대체합니다. 이 모델은 0.382M의 매개변수를 가지며, 깊이별 컨볼루션과 SSA를 결합하여 긴 거리 의존성을 효과적으로 추출합니다.

- **Performance Highlights**: H-SGANet은 OASIS 데이터셋과 LPBA40 데이터셋에서 각각 3.5% 및 1.5%의 Dice 점수 성능 향상을 보여주며, 기존의 유사한 매개변수를 가진 모델인 VoxelMorph와 비교하여 상당한 개선을 입증합니다.



### One-Shot Learning Meets Depth Diffusion in Multi-Object Videos (https://arxiv.org/abs/2408.16704)
- **What's New**: 본 논문에서는 단일 텍스트-비디오 쌍을 사용하여 일관되고 다양한 비디오를 생성할 수 있는 새로운 depth-conditioning 접근 방식을 도입했습니다. 이 방법은 pre-trained depth-aware Text-to-Image (T2I) 모델을 기반으로 하여 다중 객체 상호작용을 생성하는 데 큰 진전을 이룰 수 있게 합니다.

- **Technical Details**: 이 연구는 커스텀 설계된 공간 및 시간 주의(attention) 메커니즘을 활용하여 연속적인 움직임을 캡처할 수 있도록 pre-trained 모델을 미세 조정합니다. 또한, DDIM inversion을 사용하여 비디오 생성에 대한 구조적 가이드를 제공합니다. 이 혁신적인 기술은 비디오 내 깊이를 계속 조절 가능하게 하여 다양한 예술적 스타일의 개념 생성 및 구성 강점을 유지합니다.

- **Performance Highlights**: 제안된 방법은 단일 텍스트-비디오 쌍만으로 객체 상호작용을 보여주는 고품질 비디오를 생성할 수 있으며, 이는 T2I 모델의 다재다능한 구성 기술을 계승하여 일관성을 유지한 다양한 객체, 배경 및 상호작용을 생성할 수 있습니다.



### GradBias: Unveiling Word Influence on Bias in Text-to-Image Generative Models (https://arxiv.org/abs/2408.16700)
Comments:
          Under review. Code: this https URL

- **What's New**: 본 연구에서는 편향(bias) 탐지를 위해 사전 정의된 편향 목록 없이도 개방형 세트(open set) 환경에서 편향을 식별하고 정량화할 수 있는 일반적인 프레임워크를 제안합니다. 이 방법은 Text-to-Image(T2I) 생성 모델의 공정성과 안전성을 확보하는 데 기여할 것으로 기대됩니다.

- **Technical Details**: 제안된 파이프라인은 Large Language Model(LLM)을 활용하여 텍스트 캡션에서 시작하여 편향을 제안합니다. 이후 생성된 이미지를 기반으로 Visual Question Answering(VQA) 모델을 통해 편향을 평가합니다. OpenBias는 잘 알려진 편향 및 새로운 편향을 검출하고, GradBias는 개별 프롬프트 단어가 편향에 미치는 영향을 분석합니다.

- **Performance Highlights**: OpenBias는 기존의 폐쇄형 세트 편향 탐지 방법 및 인간 판단과 높은 일치를 보이며, GradBias는 중립적인 단어가 편향에 중요한 영향을 미칠 수 있음을 보여주고 기존의 여러 기준선 모델들을 초월하는 성능을 발휘합니다.



### Generic Objects as Pose Probes for Few-Shot View Synthesis (https://arxiv.org/abs/2408.16690)
- **What's New**: 본 논문은 3-6장의 비포즈 이미지만으로 NeRF(Neural Radiance Fields) 재구성을 시도하는 새로운 방법인 PoseProbe를 도입합니다. 이 방법은 일반적으로 사용되는 보정판(calibration board) 대신 일상적인 물체를 '포즈 프로브(pose probes)'로 활용하여 적은 수의 이미지에서도 효과적인 카메라 포즈 최적화를 가능하게 합니다.

- **Technical Details**: PoseProbe에서는 SAM(Segment Anything Model)을 사용해 큐브 모양으로 초기화한 물체를 자동으로 분할합니다. 이후 이 물체를 기반으로 하는 이중 가지 볼륨 렌더링 최적화(dou-branch volume rendering optimization) 기법을 통해 카메라 포즈와 장면 기하학을 조정합니다. PnP(Perspective-n-Poin) 매칭을 통해 먼저 두 개의 뷰에서 초기 포즈를 추정하고, 이후 추가 뷰를 점진적으로 포함시켜 포즈를 개선합니다.

- **Performance Highlights**: PoseProbe는 다수의 데이터 세트에서 포즈 추정 및 새로운 뷰 합성(new view synthesis)에서 최첨단 성능을 달성하였습니다. 특히, COLMAP이 어려움을 겪는 적은 수의 뷰 및 대간격(scene with large baselines) 환경에서 뛰어난 효과를 보여줍니다. 실험 결과, 제안된 방법은 PSNR(peak signal-to-noise ratio) 지표에서 각각 31.9%, 28.0%, 42.1%의 향상을 이뤄냈습니다.



### PartFormer: Awakening Latent Diverse Representation from Vision Transformer for Object Re-Identification (https://arxiv.org/abs/2408.16684)
- **What's New**: 이번 연구에서는 Vision Transformer(ViT)의 특성을 개선하기 위해 PartFormer라는 새로운 모델을 제안합니다. 이 모델은 CNN 기반의 기존 방법들이 적용되었던 fine-grained representation(세밀한 표현)의 한계를 극복하도록 설계되었습니다.

- **Technical Details**: PartFormer는 Head Disentangling Block(HDB)을 통합하여 multi-head self-attention에서 다각적인 표현을 활성화합니다. 또한, attention diversity constraint(주의 다양성 제약)와 correlation diversity constraint(상관관계 다양성 제약)이라는 두 가지 제약 조건을 부과하여 각 헤드의 다양한 표현을 개선합니다.

- **Performance Highlights**: 다양한 object Re-ID 벤치마크에서 포괄적인 실험을 통해 PartFormer가 기존의 최첨단 방법보다 MSMT17 데이터셋에서 2.4% 더 높은 mAP 점수를 기록하며 우수한 성능을 입증했습니다.



### Space3D-Bench: Spatial 3D Question Answering Benchmark (https://arxiv.org/abs/2408.16662)
- **What's New**: Space3D-Bench라는 이름의 새로운 3D 질문-응답 데이터셋이 소개되었습니다. 이 데이터셋은 1000개의 다양한 공간 질문과 그에 대한 답변으로 구성되어 있으며, Replica 데이터셋의 장면에 관련되어 있습니다. 또한, 공간 질문에 대한 세부 분류 체계가 제안되었습니다.

- **Technical Details**: 제안된 Space3D-Bench 데이터셋은 point clouds, RGB-D 이미지, navigation meshes 및 3D object detections와 같은 다양한 데이터 양식(modalities)을 제공합니다. Vision Language Model(VLM)을 기반으로 한 자동 평가 시스템이 질문에 대한 응답을 ground truth와 비교하여 평가합니다. RAG3D-Chat이라는 기준 모델이 도입되어 3D 공간 질문 응답에 대한 67%의 정확도를 기록했습니다.

- **Performance Highlights**: 제안된 RAG3D-Chat 시스템은 67%의 정확도에 도달하며, 이는 3D 공간 Q&A 분야에서 개선의 여지가 있음을 알립니다. 데이터셋 사용에 대한 사용자 연구에서 97.5%의 사용자 일치율이 확인되었습니다.



### Eigen-Cluster VIS: Improving Weakly-supervised Video Instance Segmentation by Leveraging Spatio-temporal Consistency (https://arxiv.org/abs/2408.16661)
Comments:
          12 pages, 6 Figures, 5 tabels

- **What's New**: 이번 연구는 새로운 약한 감독 방식인 Eigen-cluster VIS를 소개하며, 마스크 주석없이도 경쟁력 있는 정확도를 달성할 수 있음을 보여줍니다. 이는 다양한 비디오 인스턴스 세분화(Video Instance Segmentation, VIS) 접근 방식에 비해 성능 격차를 줄이는 방법입니다.

- **Technical Details**: Eigen-cluster VIS는 Temporal Eigenvalue Loss (TEL)와 Clip-level Quality Cluster Coefficient (QCC)라는 두 가지 핵심 혁신에 기반하고 있습니다. TEL은 그래프 인접 행렬에서 도출된 고유값을 활용하여 시간 일관성을 보장하며, 인접한 프레임 간의 고유값 차이를 최소화하여 안정적인 세분화 경계를 보장합니다. QCC는 K-means 방법을 사용하여 스페이시오-템포럴 클러스터의 품질을 보장합니다.

- **Performance Highlights**: 제안된 방법은 YouTube-VIS 2019/2021 및 OVIS 데이터셋에서 평가되었으며, 약한 감독 방식과 완전 감독 방식 간의 성능 격차를 효과적으로 줄이는 성과를 거두었습니다. 이 연구는 기존 VIS 방법보다 추가적인 주석 데이터 없이도 상당한 성능 향상을 제공합니다.



### DriveGenVLM: Real-world Video Generation for Vision Language Model based Autonomous Driving (https://arxiv.org/abs/2408.16647)
- **What's New**: 이번 논문에서는 자율 주행에서 비디오 생성과 이해를 결합한 새로운 프레임워크인 DriveGenVLM을 제안합니다. 이는 Denoising Diffusion Probabilistic Models(DDPM)을 기반으로 한 비디오 생성 프레임워크를 사용하여 현실 세계의 비디오 시퀀스를 예측하고, Vision Language Models(VLMs)를 활용하여 생성된 비디오를 이해할 수 있도록 합니다.

- **Technical Details**: DriveGenVLM 프레임워크는 DDPM을 사용하여 자율 주행에 적합한 비디오를 생성하며, Waymo 데이터셋을 통해 훈련되고 Fréchet Video Distance(FVD) 점수를 이용해 평가됩니다. 출시된 EILEV 모델을 통해 생성된 비디오에 대한 내레이션을 제공함으로써, 자율 주행에 필요한 상황 이해를 돕습니다.

- **Performance Highlights**: 이 연구는 DDPM을 활용하여 자율 주행 비디오 예측의 조건부 생성 모델을 적용하고, Waymo 데이터셋을 바탕으로 실제 자율 주행 시나리오에 대한 유효성을 검증한 첫 번째 사례입니다. 또한, VLM을 사용하여 생성된 비디오에 대한 설명을 생성함으로써 자율 주행 기반 알고리즘의 결정 과정에 기여할 수 있습니다.



### SODAWideNet++: Combining Attention and Convolutions for Salient Object Detection (https://arxiv.org/abs/2408.16645)
Comments:
          Accepted at ICPR 2024

- **What's New**: 최근의 Salient Object Detection (SOD) 연구에서 전통적으로 ImageNet에 사전 훈련된 backbone의 특징 계량 모듈을 활용해왔으나, 이는 SOD와 이미지 분류의 성격 차이로 인해 전체 네트워크의 사전 훈련 가능성을 제한합니다. 이러한 문제를 해결하기 위해 SOD에 특화된 새로운 인코더-디코더 신경망 SODAWideNet++을 제안합니다.

- **Technical Details**: SODAWideNet++은 Attention Guided Long Range Feature Extraction (AGLRFE) 모듈을 도입하여 대규모 dilation convolution과 self-attention을 결합합니다. 이 모듈은 convolution 작업의 귀납적 편향과 self-attention에 의한 입력 의존성을 활용하여 장거리 정보를 효과적으로 추출합니다. 이 모델은 118K의 주석이 추가된 COCO 세그멘테이션 데이터를 수정하여 end-to-end 방식으로 사전 훈련되었습니다.

- **Performance Highlights**: SODAWideNet++은 최신 모델들과 비교해 35% 적은 학습 가능한 파라미터로 경쟁력 있는 성능을 보입니다. 다섯 개의 데이터셋에서 성능 평가가 이루어졌으며, 기존의 최신 SOD 모델과 비교 시 성능이 동등하거나 더 우수한 결과를 나타냈습니다.



### 3D Pose-Based Temporal Action Segmentation for Figure Skating: A Fine-Grained and Jump Procedure-Aware Annotation Approach (https://arxiv.org/abs/2408.16638)
Comments:
          10 pages, 7th ACM International Workshop on Multimedia Content Analysis in Sports

- **What's New**: 이 연구는 피겨 스케이팅의 복잡하고 역동적인 점프 동작을 이해하기 위해 FS-Jump3D라는 새로운 데이터셋을 생성했으며, 3D 포즈 데이터와 미세한 주석 방법을 활용하여 Temporal Action Segmentation (TAS) 작업을 개선했습니다.

- **Technical Details**: FS-Jump3D 데이터셋은 광학 마커 없는 모션 캡처를 통해 수집된 3D 포즈 데이터를 포함하고, TAS 모델이 점프 절차를 학습할 수 있는 새로운 세밀한 주석 방법을 제안합니다. 여기서 TAS는 비디오의 액션을 시간적으로 세분화하는 작업을 뜻합니다.

- **Performance Highlights**: 실험 결과, 3D 포즈 피쳐를 입력으로 사용하는 것과 세밀한 주석 데이터셋이 피겨 스케이팅 TAS 모델에 유용하다는 것을 검증했습니다. FS-Jump3D 데이터셋은 연구 목적으로 배포될 예정입니다.



### Turbulence Strength $C_n^2$ Estimation from Video using Physics-based Deep Learning (https://arxiv.org/abs/2408.16623)
Comments:
          Code Available: this https URL

- **What's New**: 본 논문에서는 Classical image gradient 방법과 현대의 deep learning 기반 방법을 비교하여 $C_n^2$(refractive index structure constant) 추정을 위한 새로운 물리 기반 네트워크 아키텍처를 제안합니다. 이 네트워크는 학습된 convolutional layers와 차별화 가능한 이미지 기울기 방법을 결합하여 일반화 가능성을 향상시킵니다.

- **Technical Details**: 저자들은 원거리 이미지에서 발생하는 대기 난기류의 영향을 최소화하기 위한 알고리즘을 제안하고, 고유한 영상 캡처 데이터셋과 레퍼런스 scintillometer 측정을 수집하여 이를 과학 커뮤니티에 공개합니다. 개발된 hybrid neural network는 gradient-based 알고리즘과 convolutional neural network 접근 방식을 결합하여 $C_n^2$ 값을 추정합니다.

- **Performance Highlights**: Deep learning 방법은 유사한 데이터에 대해 높은 정확도를 달성하지만, unseen 이미지에 대한 일반화 오류가 발생하는 반면, hybrid 네트워크는 기존 classical 방법보다 더 나은 성능을 발휘하며 다양한 데이터셋에 일반화될 수 있습니다.



### Towards Infusing Auxiliary Knowledge for Distracted Driver Detection (https://arxiv.org/abs/2408.16621)
Comments:
          Accepted at KiL 2024: Workshop on Knowledge-infused Learning co-located with 30th ACM KDD Conference

- **What's New**: 이 논문에서는 운전 중 주의 산만(Distracted Driving)을 효과적으로 탐지하기 위한 새로운 방법, KiD3를 제안합니다. 해당 방법은 시각적 정보와 함께 장면 내 요소 간의 의미적 관계와 운전자의 자세 정보를 통합하여 운전자의 행동을 포괄적으로 이해할 수 있는 통합 프레임워크를 구성합니다.

- **Technical Details**: KiD3는 기존 비전 전용 모델에 비해 13.64%의 정확도 향상을 이룩하였습니다. 이 방법은 시각적 정보에 보조 지식(auxiliary knowledge)을 결합하여 다양한 운전자의 행동을 신뢰성 있게 탐지합니다. 주요 구성 요소로는 장면 그래프(scene graphs)와 운전자의 자세(pose) 정보가 결합되어 있습니다.

- **Performance Highlights**: KiD3는 실제 데이터셋을 기반으로 실험을 진행하였고, 비전 전용 기준 모델에 비해 정확도가 크게 향상되었습니다. 이는 안전한 운전 환경을 구축하는 데 기여할 수 있는 가능성을 제시합니다.



### FastForensics: Efficient Two-Stream Design for Real-Time Image Manipulation Detection (https://arxiv.org/abs/2408.16582)
Comments:
          BMVC 2024

- **What's New**: 이 논문에서는 실시간 이미지 조작 탐지를 위한 경량화된 두 가지 흐름(two-stream) 아키텍처인 FastForensics를 제안합니다. 기존의 탐지 방법들이 높은 계산 비용으로 인해 실시간 적용이 어려운 반면, FastForensics는 성능과 효율성 간의 균형을 잘 맞추고 있습니다.

- **Technical Details**: FastForensics는 인지(cognitive) 그리고 검사(inspective) 관점을 목표로 하는 두 개의 분기를 가지고 있습니다. 인지 분기에서는 효율적인 wavelet-guided Transformer blocks (EWTB)를 사용하여 주파수와 관련된 조작의 글로벌 관계를 포착합니다. 검사 분기는 간단한 convolution을 통해 세부 조작 흔적을 잡아내며, 두 분기 간의 상호작용을 통해 상호 지원을 제공합니다.

- **Performance Highlights**: FastForensics는 약 8M의 경량 모델로, 기존의 가장 최신 기술들과 비교했을 때 경쟁력 있는 성능을 보이며, 특히 이동식 디바이스에서도 효율적으로 실행 가능함을 입증하였습니다.



### MST-KD: Multiple Specialized Teachers Knowledge Distillation for Fair Face Recognition (https://arxiv.org/abs/2408.16563)
Comments:
          Accepted at ECCV 2024 ABAW

- **What's New**: 본 연구에서는 얼굴 인식(Face Recognition) 분야에서 인종별 특성을 포함한 지식을 학습하는 다중 특화 교사 프레임워크(Multiple Specialized Teacher Framework)를 제안합니다. 각 교사는 하나의 특정 민족에 특화되어 있고, 이를 통해 학생 네트워크에 공통된 정보를 증류합니다.

- **Technical Details**: 연구의 핵심은 하이퍼 특화된 교사들로부터 학생 네트워크에 지식을 전달하는 방식을 활용하는 것입니다. 하이퍼 특화 모델들은 인종별 정보에 대하여 깊이 있게 학습하며, 이는 일반적인 교사들에게서 얻을 수 없는 정보를 제공합니다. 우리의 기법은 다중 N차원 공간을 단일 N차원 공간으로 집계할 수 있으며, 이를 통해 지식 증류(Knowledge Distillation, KD)를 수행합니다.

- **Performance Highlights**: 하이퍼 특화 교사에게서 정보를 증류받은 학생 모델들이 일반 교사로부터 학습한 모델들에 비해 성능이 우수하고, 편향 지표 또한 개선된 결과를 보였습니다. 이는 특정 인종의 특징을 학습하는 것이 모든 인종 집단을 동시에 최적화하는 것보다 효율적이라는 것을 나타냅니다.



### OP-Align: Object-level and Part-level Alignment for Self-supervised Category-level Articulated Object Pose Estimation (https://arxiv.org/abs/2408.16547)
Comments:
          to be published in ECCV2024

- **What's New**: 이 논문에서는 카테고리 수준의 관절이 있는 객체 포즈 추정(catgeory-level articulated object pose estimation)을 위한 새로운 자기 지도 학습(self-supervised) 접근 방식을 제안합니다. 이 접근 방식은 단일 프레임 RGB-D 이미지에서 얻은 포인트 클라우드를 활용하여 불확실한 관절이 있는 객체를 효과적으로 추정합니다.

- **Technical Details**: 제안된 `Object-level and Part-level Alignment (OP-Align)` 모델은 전체 입력 객체에 대한 표준 포즈(canonical pose) 및 조인트 상태(joint state)를 유지하는 재구성을 생성합니다. 모델은 객체 간의 포즈 변화를 줄이기 위해 입력과 재구성을 정렬하고, 이후에 각 부분의 세분화 및 정렬을 수행합니다. 이를 통해 반복적인 연산(iterative operation)을 피하고 실시간 추론 속도를 달성합니다.

- **Performance Highlights**: 실험 결과, OP-Align은 이전의 자기 지도 방법들보다 상당한 개선을 보여주었으며, 최첨단 감독(supervised) 방법들과도 비교 가능한 성능을 보였습니다. 또한, 새로운 실세계 RGB-D 데이터세트를 통해 실제 상황에서도 뛰어난 성능을 입증하였습니다.



### Spurfies: Sparse Surface Reconstruction using Local Geometry Priors (https://arxiv.org/abs/2408.16544)
Comments:
this https URL

- **What's New**: Spurfies라는 새로운 방법을 제안하여 sparse-view surface reconstruction에서 appearance와 geometry 정보를 분리하여 synthetic data에서 훈련된 local geometry priors를 활용합니다.

- **Technical Details**: 이 방법은 geometry와 appearance를 명확히 분리하여 모델링하되, reconstruction 과정 중에 두 요소를 동시에 최적화합니다. 모델은 synthetic ShapeNet 데이터셋의 부분 집합을 사용하여 local geometry prior를 훈련합니다. 우리는 point-based neural field representation을 사용하고, 볼륨 렌더링을 통해 표면과 appearance 재구성을 수행합니다.

- **Performance Highlights**: DTU 데이터셋에서 이전의 최첨단 기술보다 35% 더 나은 surface quality를 달성했으며, 큰 비구속 장면에도 적용할 수 있습니다. Spurfies는 sparse-view surface reconstruction의 품질을 크게 향상시킵니다.



### GRPose: Learning Graph Relations for Human Image Generation with Pose Priors (https://arxiv.org/abs/2408.16540)
Comments:
          The code will be released at this https URL

- **What's New**: 최근 diffusion 모델을 활용한 인간 이미지 생성 방법이 발전하였으나, 기존 접근법에서의 포즈 정렬이 불일치하여 양질의 이미지를 생성하는 데 어려움이 있었습니다. 본 논문에서는 그래프 관계를 탐구하여 인간 이미지 생성을 위한 제어 정보를 제공하는 새로운 프레임워크를 제안합니다.

- **Technical Details**: 제안된 GRPose 프레임워크는 pose priors와 latent representation 간의 그래프 토폴로지 구조를 형성하여 서로 다른 pose 부위 간의 본질적인 연관성을 캡처합니다. Progressive Graph Integrator (PGI)를 통해 각 spatial part를 그래프 노드로 간주하고 Graph Convolutional Networks (GCNs)를 활용하여 관계를 효과적으로 포착합니다. 또한, pretrained pose estimation network를 기반으로 한 새로운 pose perception loss를 도입하여 생성된 이미지와 원본 이미지 간의 포즈 차이를 최소화합니다.

- **Performance Highlights**: Human-Art 및 LAION-Human 데이터셋에서 광범위한 실험을 수행한 결과, GRPose는 최신 벤치마크 모델 대비 포즈 평균 정밀도에서 9.98% 향상된 성능을 기록하였습니다. 여러 평가 지표에서 다른 선진 방법들에 비해 우수한 성능을 달성하였습니다.



### Are Pose Estimators Ready for the Open World? STAGE: Synthetic Data Generation Toolkit for Auditing 3D Human Pose Estimators (https://arxiv.org/abs/2408.16536)
- **What's New**: STAGE라는 GenAI 데이터 툴킷을 소개하여, 3D 인간 포즈 추정기를 감사(auditing)하고 그 성능을 평가할 수 있는 맞춤형 벤치마크를 생성하는 새로운 방법을 제시합니다.

- **Technical Details**: STAGE는 텍스트-이미지 모델을 활용하여 생성 이미지에서 3D 인간 신체 자세를 제어합니다. 이 모델은 다양한 공개 속성에 대해 사용자 정의 주석 데이터(custom annotated data)를 생성하는 기능을 제공합니다. 우리는 성별(gender), 민족(ethnicity), 나이(age), 의류(clothing), 위치(location), 날씨(weather)와 같은 속성에 대한 3D 포즈 추정기의 민감도를 감사하는 벤치마크를 생성합니다.

- **Performance Highlights**: STAGE를 통해 감지된 성능 저하는 자연 발생적으로 나타나는 속성이 포즈 추정기의 성능에 미치는 부정적인 영향을 보여주며, 이는 포즈 추정 모형이 오픈 월드 배포에 적합한지에 대한 의문을 제기합니다.



### A Comprehensive Review of 3D Object Detection in Autonomous Driving: Technological Advances and Future Directions (https://arxiv.org/abs/2408.16530)
- **What's New**: 최근 3D 객체 인식(3D object perception)이 자율 주행 시스템의 발전에 필수적인 요소로 자리 잡았습니다. 그러나 자율 주행에서 인식 작업의 변형이 증가하면서 다양한 산업 및 학계의 통찰력이 나오고 있습니다. 이에 따라, 현재까지 이러한 인식 작업과 발전을 종합적으로 정리한 설문 조사(Perspective survey)가 부족합니다.

- **Technical Details**: 이 리뷰에서는 전통적인 3D 객체 탐지 방법(3D object detection methods)을 광범위하게 요약하며, 카메라 기반(camera-based), LiDAR 기반(LiDAR-based), 그리고 융합 탐지(fusion detection) 기술에 초점을 맞춥니다. 각 접근법의 장점과 단점을 분석하고, 정확도(accuracy) 및 강인성(robustness)에서의 발전을 강조합니다. 또한, 시간적 인식(temporal perception), 점유 격자(occupancy grids), 그리고 엔드 투 엔드 학습(frameworks) 등의 정확도를 향상시키기 위한 방법들을 논의합니다. 협력 인식(cooperative perception) 방법도 다루며, 공동 통신을 통해 인식 범위를 확장할 수 있는 잠재력을 탐색합니다.

- **Performance Highlights**: 이 연구는 3D 객체 인식의 현재 상태와 향후 발전에 대한 총체적인 시각을 제공하여 자율 주행을 위한 인식 작업에 대한 보다 포괄적인 이해를 돕고자 합니다. 또한, 이 분야의 최신 발전에 대한 지속적인 업데이트를 제공하기 위해 활성화된 리포지토리(repository)를 구축하였습니다.



### Towards Modality-agnostic Label-efficient Segmentation with Entropy-Regularized Distribution Alignmen (https://arxiv.org/abs/2408.16520)
Comments:
          Extended version of arXiv:2305.15832; Code at this https URL

- **What's New**: 본 연구에서는 label-efficient segmentation을 위한 새로운 학습 전략인 ERDA(Entropy Regularization and Distribution Alignment) 학습 전략을 제안합니다. 이 방법은 2D 및 3D 데이터 모달리티에 적용 가능하며, pseudo-label의 신뢰성을 높이고 모델 예측과의 일치를 강화하여 더 나은 성능을 도출합니다.

- **Technical Details**: ERDA는 Entropy Regularization(ER) 손실과 Distribution Alignment(DA) 손실을 도입하여 pseudo-label을 정규화하고, 이에 따라 모델 학습 과정에서의 혼란을 줄입니다. KL 거리(KL distance)를 이용하여 DA 손실을 계산함으로써, cross-entropy 기반의 손실 함수로 변환되어 pseudo-label 생성 모듈과 분할 모델을 동시에 최적화합니다.

- **Performance Highlights**: ERDA는 S3DIS, ScanNet 및 2D 데이터셋인 Pascal, Cityscapes에서 기존 방법들을 초월하는 성능을 보여주었으며, 단 1%의 라벨만으로도 완전 지도 기반의 성능을 초과할 수 있음을 증명했습니다. 또한 ERDA는 다양한 새로운 label-efficient segmentation 작업으로 일반화 가능한 능력을 입증하였습니다.



### Alignment is All You Need: A Training-free Augmentation Strategy for Pose-guided Video Generation (https://arxiv.org/abs/2408.16506)
Comments:
          CVG@ICML 2024

- **What's New**: 이 논문에서는 정적 이미지에서 동적인 비디오 애니메이션으로의 변환을 가능하게 하는 교육이 필요 없는 프레임워크를 제안합니다. 이 방법은 생성된 비디오 시퀀스가 참조 이미지의 세부적인 모습을 유지하도록 하며, 이로 인해 모션의 진실성과 외관의 일관성이 보장됩니다.

- **Technical Details**: 이 접근 방식은 이중 정렬 전략(dual alignment strategy)을 도입하여 골격(skeletal)과 모션 프라이어(motion priors)를 포즈 정보로부터 분리합니다. 이는 포즈 시퀀스 생성을 위한 정밀한 제어를 가능하게 하며, 동시에 픽셀 수준의 정렬을 개선하여 조건적인 제어를 강화합니다.

- **Performance Highlights**: 이 논문의 방법은 대규모 데이터셋이나 고가의 컴퓨팅 자원 없이도 캐릭터 애니메이션 생성의 품질을 상당히 향상시킵니다. 실험 결과, 제안된 방식이 애니메이션 품질을 효과적으로 개선할 수 있음을 증명했습니다.



### A Simple and Generalist Approach for Panoptic Segmentation (https://arxiv.org/abs/2408.16504)
- **What's New**: 이 논문에서는 panoptic segmentation(파노픽 세그멘테이션)에 대한 최신 접근법을 제안하며, 일반화 모델의 강력함을 활용하면서도 성능을 향상시키는 두 가지 주요 기여 내용을 소개합니다.

- **Technical Details**: 주요 기여로는 (i) centroid(중심) 회귀를 개선하기 위한 positional-embedding(위치 임베딩) 기반의 손실 함수, (ii) 인스턴스 경계를 보다 잘 분리하기 위한 Edge Distance Sampling(EDS)을 사용하는 방법이 있습니다. 이러한 방법들은 이미지의 결측 레이블이나 작은 인스턴스를 처리하는 데 효과적입니다.

- **Performance Highlights**: 제안된 방법은 COCO 데이터 세트에서 Panoptic Quality(PQ) 52.5를 달성하였으며, 이는 유사한 접근 방식을 갖는 최고의 모델인 Painter보다 10포인트 향상된 성능입니다. 또한 diffusion 기반 방법 Pix2Seq-$\mathcal{D}$보다 2포인트 더 높은 성능을 보입니다.



### Locally Grouped and Scale-Guided Attention for Dense Pest Counting (https://arxiv.org/abs/2408.16503)
- **What's New**: 본 연구에서는 디지털 트랩에 의해 포착된 밀집된 해충을 예측하기 위한 새로운 밀집 해충 계수 문제(dense pest counting problem)를 소개합니다. 이는 전통적인 탐지 기반 계수 모델과는 달리, 밀집된 해충 분포로 인한 심각한 가림(obscuration), 다양한 자세(pose variation), 유사한 색상 및 텍스처로 인한 문제들을 해결해야 합니다.

- **Technical Details**: 연구에서 제안된 방법은 지역적 주의 메커니즘(local attention mechanism)을 통합하여 지역적으로 중요한 영역을 식별하고, 지역적으로 그룹화된 특징들을 학습하여 판별 성능(discriminative performance)을 향상시킵니다. 또한, 본 연구에서는 독창적인 설계를 통해 지역적으로 그룹화되고 스케일 안내(scale-guided) 주의를 통합한 다중 스케일 CenterNet 프레임워크를 제시합니다. 해충 중심 정보가 포함된 첫 번째 시간 모래시계(hourglass)에서 예측된 히트맵(heatmap)을 사용하여 유사한 속성을 가진 지역적 특징들을 그룹화하는 간단한 방법이 도입되었습니다.

- **Performance Highlights**: 실험을 통해 제안된 모델은 지역 그룹화(local grouping)와 판별적 특징 주의 학습(discriminative feature attention learning)에 기반하여 객체 특징을 향상시키는 것이 확인되었습니다. 특히, 이 모델은 가림 및 자세 변동 문제를 극복하는 데 매우 효과적이며, 밀집 해충 계수에 적합합니다. 제안된 모델은 최첨단(state-of-the-art) 모델들과 비교하여 큰 차이로 성능을 초월하며, 밀집 해충 계수에 기여하는 바가 두드러집니다.



### UAV-Based Human Body Detector Selection and Fusion for Geolocated Saliency Map Generation (https://arxiv.org/abs/2408.16501)
Comments:
          42 pages, 19 figures

- **What's New**: 최근 UAV(무인 항공기)를 활용한 검색 및 구조(SAR) 임무의 효율성을 높이기 위해 다양한 객체 탐지 및 지리적 위치 파악 메커니즘을 통합하는 연구가 진행되었습니다. 이 연구에서는 시스템 관점에서 비전 기반 탐지기 선택, 가용 자원 최적화 및 탐지 결과 융합을 통해 효과적인 구조 임무 수행 방법을 제시하고 있습니다.

- **Technical Details**: 제안된 방법은 UAV 플랫폼의 실제 운영 환경에서 고려해야 할 다양한 요인, 즉 통신 링크, 비디오 압축, 계산 자원 등을 기반으로 한 객체 탐지를 위한 최적의 알고리즘 선택을 포함합니다. 이러한 평가를 통해 비전 기반 탐지 결과를 통합하고, 수집된 데이터로부터 확률적 융합 방법을 통해 일반화된 지리적 위치 지도를 생성합니다.

- **Performance Highlights**: 실험 결과, 제안된 시스템은 SAR 작업 중 지리적 위치 감지의 정확성과 신뢰성을 개선했으며, 특정 객체 탐지기를 효과적으로 실행하여 자원의 활용을 극대화했습니다. 또한, 새로운 지리적 위치 기반 객체에 대해 새로운 임무를 정의할 수 있도록 지원하여, 구조 작업의 효율성을 크게 향상시켰습니다.



### CogVLM2: Visual Language Models for Image and Video Understanding (https://arxiv.org/abs/2408.16500)
- **What's New**: CogVLM2 패밀리의 출시를 알리며, 이 시리즈에는 이미지 및 비디오 이해를 위한 CogVLM2, CogVLM2-Video, GLM-4V가 포함됩니다.

- **Technical Details**: CogVLM2는 기존의 visual expert architecture를 계승하며, pretrained (사전 훈련) 및 post-trained (후 훈련) 단계에서 개선된 training recipes를 사용합니다. 이 모델은 최대 1344x1344 픽셀의 입력 해상도를 지원하며, CogVLM2-Video는 다중 프레임 입력을 시간 스탬프와 함께 통합합니다.

- **Performance Highlights**: CogVLM2 패밀리는 MMBench, MM-Vet, TextVQA, MVBench, VCGBench와 같은 여러 벤치마크에서 최고의 성능을 기록했습니다.



### Adapting Vision-Language Models to Open Classes via Test-Time Prompt Tuning (https://arxiv.org/abs/2408.16486)
Comments:
          PRCV 2024

- **What's New**: 본 논문에서는 오픈 클래스에서의 적응을 활성화하기 위한 새로운 접근 방식인 테스트 시점의 prompt tuning 방법을 제안합니다. 각 이미지에 대해 입력에 조건화된 prompt를 생성하기 위해 최대 개념 일치(Maximum Concept Matching, MCM) 점수를 동적으로 가중치로 활용합니다.

- **Technical Details**: 테스트 시점에서 학습된 prompt와 수작업으로 생성된 prompt를 결합하여 더 나은 일반화 능력을 확보하는 방법을 제시합니다. CoOp(속성 최적화)와 CoCoOp(조건부 컨텍스트 최적화)의 결과를 바탕으로, 오픈 세트 적응 설정에서 모델 성능을 평가합니다. MCM 점수를 사용하여 base 클래스와 new 클래스 간의 경험적 균형을 묘사합니다.

- **Performance Highlights**: 11개 다른 데이터셋에서 대규모 실험을 통해 제안된 방법이 기존의 모든 비교 방법들과 비교했을 때 평균적으로 뛰어난 성능을 보이며, base 및 new 클래스의 정확도를 고려하여 우수성을 입증합니다.



### MICDrop: Masking Image and Depth Features via Complementary Dropout for Domain-Adaptive Semantic Segmentation (https://arxiv.org/abs/2408.16478)
- **What's New**: 이번 연구에서는 Unsupervised Domain Adaptation (UDA)에서 발생하는 미세 구조 인식 문제와 모호한 객체 분할 문제를 해결하기 위해 기하학적 정보인 depth 예측을 활용하는 새로운 방법인 MICDrop을 제안합니다.

- **Technical Details**: MICDrop은 이미지 인코더의 특징을 마스킹(masking)하고 깊이 인코더의 특징을 반대로 마스킹하여 공동 특징 표현(joint feature representation)을 학습합니다. 이 방법은 두 가지 모달리티를 통합하는 간단하면서도 효과적인 보완 마스킹 전략을 사용합니다. 또한, feature fusion 모듈을 제안하여 깊이 예측에서의 오류에 강인하게 작용하며 글로벌(global) 및 로컬(local) 정보 공유를 개선합니다.

- **Performance Highlights**: MICDrop은 다양한 최신 UDA 방법에 통합될 수 있으며, 여러 표준 UDA 벤치마크에서 일관되게 결과를 개선하여 새로운 최첨단 성능을 달성했습니다.



### Creating a Segmented Pointcloud of Grapevines by Combining Multiple Viewpoints Through Visual Odometry (https://arxiv.org/abs/2408.16472)
- **What's New**: 이 논문은 포도나무의 겨울 가지치기를 위한 새로운 컴퓨터 비전 파이프라인을 제시합니다. 이 시스템은 detectron2를 이용한 segmentation 네트워크와 keypoint visual odometry를 통합하여 포도나무의 가지치기 의사 결정을 지원합니다.

- **Technical Details**: 이 연구는 포도나무의 5가지 주요 장기(코돈(cordon), 팔(arm), 스퍼(spur), 케인(cane), 노드(node))를 탐지하기 위해 detectron2를 기반으로 한 convolutional neural network를 사용합니다. RGB-D 카메라를 장착한 로봇 조작기를 통해 다양한 시점에서 이미지를 수집하며, 이러한 데이터는 키포인트 매칭과 3D 공간으로의 투영을 통해 세밀한 4D 구조를 생성합니다. 포인트 클라우드는 HDBSCAN 클러스터링 기법을 통해 클래스별로 구분 됩니다.

- **Performance Highlights**: 성능 측면에서, 이 시스템은 81%의 recall과 97%의 precision을 달성하였으며, 수집된 여러 프레임의 데이터를 통합하여 정확하고 안정적인 포도 가지치기 계획을 위한 포인트 클라우드를 생성합니다. 이 연구는 또한 로봇 조작기가 정확하게 가지치기를 수행할 수 있도록 하는 향후 작업의 기초를 제공합니다.



### Multi-source Domain Adaptation for Panoramic Semantic Segmentation (https://arxiv.org/abs/2408.16469)
Comments:
          9 pages, 7 figures, 5 tables

- **What's New**: 본 논문은 실제 핀홀 이미지와 저비용 합성 파노라마 이미지를 이용하여 새로운 멀티 소스 도메인 적응(Multi-source Domain Adaptation, MSDA) 작업을 제안합니다. 이를 통해 세그멘테이션 모델이 레이블이 없는 실제 파노라마 이미지에 대해 효과적으로 작동할 수 있도록 합니다.

- **Technical Details**: 제안하는 DTA4PASS(Deformation Transform Aligner for Panoramic Semantic Segmentation) 프레임워크는 주요 두 요소인 Unpaired Semantic Morphing (USM)과 Distortion Gating Alignment (DGA)로 구성됩니다. USM은 핀홀 이미지를 적대적 방식으로 파노라마와 유사한 이미지로 변환하고, DGA는 마치 핀홀처럼 보이는 특징과 파노라마처럼 보이는 특징을 각 이미지에 부여하여 이 둘을 정렬합니다. 이 과정에서 불확실성 추정을 통해 두 특징 간의 정렬 또한 수행됩니다.

- **Performance Highlights**: DTA4PASS는 실외 및 실내 멀티 소스 도메인 적응 시나리오에서 기존 기술 대비 각각 1.92% 및 2.19% 성능을 향상시켰습니다. 실험 결과, 제안하는 방법이 광범위한 실험을 통해 뛰어난 효과를 보인다는 것을 입증했습니다.



### Weakly Supervised Object Detection for Automatic Tooth-marked Tongue Recognition (https://arxiv.org/abs/2408.16451)
- **What's New**: 이 논문은 전통 중국 의학(Traditional Chinese Medicine, TCM)에서 혀 진단의 자동화를 위해 Weakly Supervised 방법과 Vision Transformer를 사용하는 새로운 접근법을 제안합니다. 이는 과거의 주관적인 진단 방식을 개선하여 더 객관적이고 정확한 방법을 제공합니다.

- **Technical Details**: 제안된 WSVM(Weakly Supervised Vision Model)은 두 단계로 구성됩니다: 1) 임상 이미지를 통해 혀의 전경(foreground)을 추출하여 배경 정보를 제거하는 단계, 2) 혀 이미지를 패치(patch)로 처리하는 Vision Transformer(ViT) 기반의 약하게 감독된 객체 탐지 방법을 통한 치아 자국(tooth-marked tongue) 인식 단계입니다. 이를 통해 모델은 이미지 수준의 주석만으로도 치아 자국 인식을 수행할 수 있습니다.

- **Performance Highlights**: WSVM은 치아 자국 분류에서 높은 정확도를 기록하였으며, 시각화를 통해 혀의 자국 위치를 효과적으로 구분할 수 있음을 입증하였습니다. 본 접근법은 TCM 실무자들이 보다 정확한 진단과 치료 추천을 할 수 있도록 도와주는 중요한 임상 가치를 제공합니다.



### What to Preserve and What to Transfer: Faithful, Identity-Preserving Diffusion-based Hairstyle Transfer (https://arxiv.org/abs/2408.16450)
- **What's New**: 본 논문에서는 스타일간(StyleGAN)에 의존하지 않고, 다양한 얼굴 이미지에 일반화 가능한 단일 단계 헤어스타일 이전(diffusion) 모델인 HairFusion을 제안합니다. HairFusion은 실제 상황에서 작동할 수 있도록 설계되었으며, 참고 헤어스타일을 원본 얼굴 이미지에 효과적으로 이식하는 데 중점을 두고 있습니다.

- **Technical Details**: HairFusion은 헤어와 무관한 표현(hair-agnostic representation)을 모델의 입력으로 사용하여 얼굴 이미지의 원래 머리 정보를 완전히 제거합니다. 이후, Align-CA(헤어 정렬 교차 주의 모듈)가 얼굴 이미지의 머리 부분과 참고 헤어스타일을 정확하게 정렬합니다. 이 과정에서 얼굴 모양의 차이를 고려하며, 적응형 헤어 블렌딩(adaptive hair blending)을 통해 원본 얼굴 기능을 유지합니다.

- **Performance Highlights**: 실험 결과, HairFusion은 전통적인 GAN 기반 방법과 기존의 diffusion 모델들에 비해 우수한 성능을 보였습니다. 특히, 이 모델은 다양한 현실 이미지에 대해 일반화가 잘 이루어지며, 전이된 헤어스타일과 주변 기능의 완전성을 효과적으로 유지하는 것으로 나타났습니다.



### Enhancing Sound Source Localization via False Negative Elimination (https://arxiv.org/abs/2408.16448)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2203.13412

- **What's New**: 본 논문에서는 소리 출처 로컬리제이션(Sound Source Localization)의 문제를 해결하기 위한 새로운 오디오-비주얼(Audio-Visual) 학습 프레임워크를 제안합니다. 이 프레임워크는 자기 지도 예측 학습(SSPL)과 의미 인식 대조 학습(SACL)의 두 가지 개별 학습 방식을 포함하고 있습니다.

- **Technical Details**: SSPL은 이미지-오디오 양성 쌍(positive pairs)만을 탐색하여 오디오와 비주얼 특징 간의 의미론적 일관성을 발견합니다. 이 과정에서 피쳐 정렬(feature alignment)을 위한 예측 코딩 모듈을 도입하여 양성 학습을 촉진합니다. 반면, SACL은 비주얼 특징을 압축하고 잘못된 음수(false negatives)를 제거하여 신뢰할 수 있는 비주얼 앵커(visual anchor)와 오디오 음수(audio negatives)를 제공합니다.

- **Performance Highlights**: 종합적인 실험 결과, 제안한 방법이 기존의 최첨단 기술에 비해 우수한 성능을 보임을 입증하였습니다. 또한 학습된 표현의 다재다능성(versatility)을 강조하며 오디오-비주얼 이벤트 분류(event classification) 및 객체 탐지(object detection) 작업으로 접근법을 확장했습니다.



### Mismatched: Evaluating the Limits of Image Matching Approaches and Benchmarks (https://arxiv.org/abs/2408.16445)
Comments:
          19 pages, 5 figures

- **What's New**: 본 논문은 2D 이미지에서 3D 구조를 재구성하는 과정에서 다양한 image matching 방법을 종합적으로 평가하여, 최신 기술과 연구 방향을 제안하고 있습니다.

- **Technical Details**: 3D reconstruction을 위한 전통적인 구조-모션(SfM) 기술을 기반으로 20개의 최첨단(image matching) 방법을 비교하고, (a) 일관된 조명 조건을 가진 단일 장면의 clean images로 구성된 데이터셋과 (b) 다양한 조명 및 투명성을 가진 이미지를 포함하는 out-of-domain dataset을 사용하여 성능을 평가합니다. 또한, edge detection이 각 방법에 미치는 영향을 분석합니다.

- **Performance Highlights**: image matching은 여전히 해결해야 할 과제로 남아 있으며, 다양한 시나리오에서의 성능을 평가한 결과, 현재의 metric이 방법의 성능을 충분히 나타내지 못하고 있다는 점이 강조되었습니다.



### Integrating Features for Recognizing Human Activities through Optimized Parameters in Graph Convolutional Networks and Transformer Architectures (https://arxiv.org/abs/2408.16442)
Comments:
          6 pages, 1 figure, conference

- **What's New**: 인간 활동 인식에 대한 연구는 딥러닝 모델의 특징 융합을 통해 정확도를 향상시키는 데 중점을 두고 있습니다. 특히, Transformer 모델과 파라미터 최적화된 그래프 합성곱 네트워크(PO-GCN)를 사용하여 다양한 공공 데이터셋을 활용한 성능 향상을 보여줍니다.

- **Technical Details**: 이 연구는 HuGaDB, PKU-MMD, LARa, TUG의 4개 공공 데이터셋에서 수집한 데이터를 사용하여 PO-GCN 및 Transformer 모델을 훈련하고 평가하였습니다. 이 두 모델에서 최종 계층의 특징을 융합하고, 이를 분류기에 입력해 활동 인식을 수행합니다. 특징 융합(feature fusion) 기술은 공간적 및 시간적 특징을 보다 잘 이해할 수 있게 해줍니다.

- **Performance Highlights**: HuGaDB 데이터셋에서 2.3%의 정확도 향상과 2.2%의 F1-점수 증가를 나타냈으며, TUG에서는 5%의 정확도 증가와 0.5%의 F1-점수 향상을 기록했습니다. LARa 및 PKU-MMD는 각각 64%, 69%로 낮은 정확도를 보였지만, PO-GCN 모델과 Transformer 모델의 통합이 성능을 개선했다는 강력한 증거를 제공합니다.



### Discriminative Spatial-Semantic VOS Solution: 1st Place Solution for 6th LSVOS (https://arxiv.org/abs/2408.16431)
Comments:
          1st Place Solution for 6th LSVOS VOS Track. arXiv admin note: substantial text overlap with arXiv:2406.04600

- **What's New**: 본 논문에서는 Video Object Segmentation (VOS) 문제를 해결하기 위한 새로운 discriminative spatial-temporal 모델을 제안합니다. 이 모델은 복잡한 장면과 장기적인 물체 움직임을 효율적으로 처리하기 위해 설계되었습니다.

- **Technical Details**: 제안된 방법은 ViT (Vision Transformer) 모델에서 추출한 semantic feature를 활용하여 target queries의 정의성을 향상시키고, spatial-semantic module을 사용하여 물체 인식 성능을 강화합니다. 또한, object queries를 통해 시각적으로 유사한 물체들 간의 ID 혼동을 줄입니다.

- **Performance Highlights**: 모델은 6차 LSVOS 도전에서 VOS Track에서 80.90%의 Jaccard & F 점수로 1위를 차지하여 VOS 분야에서의 효율성을 입증하였습니다.



### COIN: Control-Inpainting Diffusion Prior for Human and Camera Motion Estimation (https://arxiv.org/abs/2408.16426)
Comments:
          ECCV 2024

- **What's New**: 본 논문에서는 COIN(Control-Inpainting Motion Diffusion Prior)이라는 새로운 방법을 제안합니다. COIN은 인간과 카메라 모션을 분리하는 섬세한 제어를 가능하게 하여, 기존의 과도하게 매끄럽고 잘 맞지 않는 2D 투영 문제를 해결합니다.

- **Technical Details**: COIN의 주요 구성 요소로는 동적 제어 샘플링(dynamic controlled sampling) 기법과 새로운 인간-장면 관계 손실(human-scene relation loss)을 포함합니다. 이 손실은 인간, 카메라 및 장면 간의 일관성을 강제하여 스케일 불일치를 완화합니다. 또한, 저해상도에서 고해상도의 전환을 위한 Soft Inpainting 전략이 채택되었습니다.

- **Performance Highlights**: COIN은 HCM 및 RICH 데이터셋에서 각각 44% 및 33%의 성능 향상을 보이며, 기존의 최첨단 방법들과 비교할 때 인간 및 카메라 모션 추정 정확도에서 눈에 띄는 개선을 나타냅니다.



### Text-Enhanced Zero-Shot Action Recognition: A training-free approach (https://arxiv.org/abs/2408.16412)
Comments:
          accepted to ICPR 2024

- **What's New**: 이번 논문은 TEAR(Text-Enhanced Action Recognition)이라는 새로운 접근 방식을 제안합니다. 이 방법은 제로샷 비디오 액션 인식(zero-shot video action recognition, ZS-VAR)을 위한 것이며, 훈련 데이터나 많은 계산 자원 없이도 구현할 수 있습니다. 이는 비디오의 동적이고 시간적인 특성을 고려하여 동작에 대한 이해를 가능하게 합니다.

- **Technical Details**: TEAR는 먼저 대규모 언어 모델(large language model, LLM)을 사용하여 액션 설명자를 생성하고, 그 다음에 이 설명자를 통해 제로샷 예측(zero-shot prediction)을 수행하는 두 단계로 구성됩니다. 실험은 UCF101, HMDB51 및 Kinetics-600 데이터셋에서 수행하였으며, CLIP를 VLM으로, GPT-3.5를 LLM으로 활용했습니다.

- **Performance Highlights**: TEAR 방법은 세 가지 데이터셋에서 훈련 기반 접근법과 비교할 수 있는 성과를 달성하였으며, 제로샷 비디오 액션 인식에서 훈련이 필요 없는 방식으로 접근함으로써 실제 응용에 더 적합한 결과를 보여주었습니다.



### IBO: Inpainting-Based Occlusion to Enhance Explainable Artificial Intelligence Evaluation in Histopathology (https://arxiv.org/abs/2408.16395)
Comments:
          19 pages, 6 figures

- **What's New**: 본 논문에서는 Histopathological 이미지 분석을 위한 새로운 Inpainting-Based Occlusion (IBO) 전략을 제안합니다. 기존의 occlusion 기반 XAI 방법들이 Out-of-Distribution (OoD) 샘플을 생성하여 평가의 정확성을 저하시키는 문제를 해결하기 위해, IBO는 Denoising Diffusion Probabilistic Model을 사용하여 암세포 영역을 비암세포 조직으로 대체합니다. 이를 통해 데이터의 무결성을 보호하고 OoD 아티팩트를 최소화합니다.

- **Technical Details**: IBO 방법은 두 단계로 구성된 평가 과정에서 CAMELYON16 데이터셋에서 성능을 평가합니다. 첫째, Learned Perceptual Image Patch Similarity (LPIPS) 메트릭을 사용하여 지각적 유사성을 분석하고, 둘째, Area Under the Curve (AUC) 분석을 통해 모델 예측에 미치는 영향을 정량화합니다. IBO는 지각적 충실도를 크게 개선하여 기존의 최상의 occlusion 전략에 비해 LPIPS 점수에서 거의 두 배의 개선을 달성했습니다.

- **Performance Highlights**: IBO는 전통적인 방법에 비해 XAI 성능 예측의 정밀도를 42%에서 71%로 증가시켰습니다. 이러한 결과는 IBO가 XAI 기법의 보다 신뢰할 수 있는 평가를 제공하여 histopathology 및 기타 응용 분야에 기여할 수 있는 잠재력을 강조합니다.



### Exploiting temporal information to detect conversational groups in videos and predict the next speaker (https://arxiv.org/abs/2408.16380)
Comments:
          Accepted to Pattern Recognition Letter, 8 pages, 10 figures

- **What's New**: 이번 논문에서는 사회적 상호작용 동안 참가자 간의 공간 배치를 설명하는 F formation 개념의 검출과 다음 화자의 예측을 목표로 하고 있습니다. 비디오 시퀀스에서 시간 정보와 인간의 다중 모달 신호를 활용하여 그룹 대화를 분석하는 새로운 방법이 제시되었습니다.

- **Technical Details**: 연구진은 대화 그룹에서의 다음 화자를 예측하기 위해 Long Short Term Memory (LSTM) 모델을 사용합니다. 이 접근법은 플롯 서랍에 기반한 참여자 간의 engagement 수준을 특징으로 삼고 있으며, O-space, P-space, R-space로 나누어지는 F-formation 구성을 통해 그룹을 클러스터링합니다. 시간 정보를 활용하여 각 참가자의 머리와 몸 방향을 측정하고 '시간 가중 각도'를 계산하여 정밀한 클러스터링을 수행합니다.

- **Performance Highlights**: 실험 결과 MatchNMingle 데이터셋을 활용하여 그룹 감지에서 85%의 True Positive를 달성하였고, 다음 화자 예측의 정확도는 98%를 기록하였습니다.



### Law of Vision Representation in MLLMs (https://arxiv.org/abs/2408.16357)
Comments:
          The code is available at this https URL

- **What's New**: 이 논문은 멀티모달 대형 언어 모델(MLLMs)에서 시각 표현의 최적 선택 원리를 제시합니다. 'Law of Vision Representation'을 통해 시각 표현의 교차 모드 정렬(cross-modal alignment) 및 일치성(correspondence)이 모델 성능에 미치는 강한 상관관계를 규명하였습니다.

- **Technical Details**: 교차 모드 정렬 및 일치성을 정량화하기 위해 AC 점수(Alignment and Correspondence score)를 정의하고, 이를 통해 MLLM의 성능을 예측하는 새로운 모형을 제안합니다. 13개의 시각 표현 설정과 8개의 벤치마크에 대해 광범위한 실험을 통해 AC 점수와 모델 성능 간의 선형 상관관계를 발견하였습니다.

- **Performance Highlights**: AC 점수에 기반한 최적의 시각 표현을 선택하는 정책을 통해, 언어 모델의 세밀한 조정 없이 99.7%의 계산 비용 절감을 이룰 수 있음을 보여주었습니다. 이 접근법은 정확성과 효율성을 향상시키며, 최적 구성을 상위 세 가지 선택지 중에서 96.6%의 경우에서 성공적으로 식별합니다.



### Toward Robust Early Detection of Alzheimer's Disease via an Integrated Multimodal Learning Approach (https://arxiv.org/abs/2408.16343)
Comments:
          5 pages, 2 figures

- **What's New**: 이 연구는 알츠하이머병(Alzheimer's Disease, AD)의 진단 정확성을 높이기 위해 임상, 인지, 신경영상, 뇌파(EEG) 데이터를 통합한 고급 다중 모달 분류 모델을 도입합니다.

- **Technical Details**: 제안된 모델은 표 데이터의 코딩 아키텍처를 갖춘 기능 태거(feature tagger)를 통합하고, TimesBlock 모듈을 통해 EEG 데이터의 복잡한 시간 패턴을 캡처합니다. Cross-modal Attention Aggregation 모듈을 활용하여 MRI의 공간 정보와 EEG의 시간 정보를 효과적으로 융합하였습니다.

- **Performance Highlights**: 이 모델은 알츠하이머병, 경도 인지장애(Mild Cognitive Impairment, MCI), 정상 인지를 구별하는 데 있어 상당한 성능 향상을 보여주었으며, 새로운 AD 분류 데이터셋을 구축하여 임상적 진단의 초석을 다듬었습니다.



### P2P-Bridge: Diffusion Bridges for 3D Point Cloud Denoising (https://arxiv.org/abs/2408.16325)
Comments:
          ECCV 2024 Project page: this https URL

- **What's New**: 본 연구에서는 Diffusion Schrödinger bridge를 활용하여 point cloud denoising 문제를 해결하는 새로운 프레임워크를 제안합니다. 기존 접근 방식과 달리, 우리의 방법은 쌍으로 나눠진 point cloud 간의 최적 전송 계획(optimal transport plan)을 학습합니다.

- **Technical Details**: P2P-Bridge라는 이 방법은 point cloud denoising을 데이터 간의 확산 프로세스로 모델링하고, RGB 및 DINOv2 피처를 추가로 활용하여 성능을 향상시킵니다. 이 방식은 잡음이 있는 point cloud와 깨끗한 point cloud 간의 최적의 전송 계획을 찾는 네트워크를 학습하면서 다양한 데이터 유형에서 훈련될 수 있습니다.

- **Performance Highlights**: 실험 결과, P2P-Bridge는 PU-Net, ScanNet++ 및 ARKitScenes와 같은 실제 데이터셋에서 기존 방법들보다 상당한 성능 개선을 보여주었습니다. 잡음이 있는 point cloud의 좌표만을 사용하여도 강력한 결과를 보여주며, 색상 정보나 point-wise DINOv2 피처를 포함할 경우 성능이 더욱 강화됩니다.



### BEVal: A Cross-dataset Evaluation Study of BEV Segmentation Models for Autononomous Driving (https://arxiv.org/abs/2408.16322)
- **What's New**: 이 논문은 현재의 자율주행을 위한 semantic bird's-eye view (BEV) segmentation 연구가 일반적으로 nuScenes와 같은 단일 데이터셋에서 신경망 모델만 최적화하는 데 집중하고 있다는 점을 지적합니다. 이전의 방법이 다양한 환경이나 센서 셋업에서 실패할 수 있는 domain shift 문제를 초래하는 것에 대한 해결책을 제시합니다.

- **Technical Details**: 본 연구는 최첨단 BEV segmentation 모델들에 대한 cross-dataset 평가를 수행하여 다양한 훈련 및 테스트 데이터셋과 설정, 그리고 다양한 semantic 카테고리에서의 성능을 평가합니다. 카메라 및 LiDAR와 같은 다양한 센서가 모델의 일반화 능력에 미치는 영향을 조사하고, 다중 데이터셋 훈련 실험을 통해 단일 데이터셋 훈련보다 BEV segmentation 성능을 향상시킵니다.

- **Performance Highlights**: 이 연구는 cross-dataset validation에서 BEV segmentation 모델을 평가하여 모델의 일반화 가능성과 적응성을 향상시키는 중요성을 강조합니다. 이는 자율주행 애플리케이션을 위한 보다 강력하고 신뢰할 수 있는 BEV segmentation 접근 방식을 보장하기 위한 기반이 됩니다.



### ResVG: Enhancing Relation and Semantic Understanding in Multiple Instances for Visual Grounding (https://arxiv.org/abs/2408.16314)
Comments:
          Accepted by ACM MM 2024

- **What's New**: 본 논문은 이미지에서 자연어 쿼리를 기반으로 객체를 지정하는 Visual Grounding 작업의 새로운 접근 방식인 Relation and Semantic-sensitive Visual Grounding(ResVG) 모델을 제안합니다. 본 모델은 다수의 간섭 객체가 존재하는 환경에서도 정확한 객체 위치를 파악할 수 있도록 설계되었습니다.

- **Technical Details**: ResVG 모델은 첫째, 쿼리에서 유도된 의미적 사전 정보를 모델에 주입하여 미세한 의미(semantic)를 이해하는 데 도움을 주며, 둘째, 관계적 데이터 증강(relational data augmentation) 방법을 도입하여 다양한 간섭 개체들을 포함하는 추가 학습 데이터를 생성합니다. 이를 통해 객체 간의 공간적 관계(spatial relationships)를 보다 잘 이해할 수 있게 됩니다.

- **Performance Highlights**: 다양한 데이터셋(RefCOCO, RefCOCO+, RefCOCOg, ReferItGame, Fliker30K Entities)에서 진행된 실험 결과, ResVG 모델은 다수의 간섭 개체가 있는 상황에서도 시각적 임베딩 (visual grounding) 작업의 성능을 크게 향상시켰음을 보여주었습니다.



### FA-YOLO: Research On Efficient Feature Selection YOLO Improved Algorithm Based On FMDS and AGMF Modules (https://arxiv.org/abs/2408.16313)
Comments:
          11 pages and 4 figures

- **What's New**: 최근 YOLO 시리즈 모델이 객체 탐지 분야에서 유명세를 타고 있으며, 본 논문은 FMDS 모듈(Fine-grained Multi-scale Dynamic Selection Module)과 AGMF 모듈(Adaptive Gated Multi-branch Focus Fusion Module)을 소개하여 YOLOv9 기반의 새로운 객체 탐지 모델 FA-YOLO를 개발했습니다.

- **Technical Details**: FMDS 모듈은 정교한 다중 스케일 특성 맵에서 동적인 특성을 선택하고 융합하는 방식으로, 소형, 중형, 대형 목표물의 탐지 정확도를 크게 향상시킵니다. AGMF 모듈은 다양한 특성을 보완적으로 융합하여 특징 맵의 표현력을 높이고, 전체적인 특성 융합의 효율성을 증가시킵니다.

- **Performance Highlights**: FA-YOLO 모델은 PASCAL VOC 2007 데이터셋에서 평균 정밀도(mAP) 66.1%를 달성하며, YOLOv9의 65.1%에서 1.0% 향상된 성능을 보여줍니다. 또한, 소형, 중형, 대형 목표물의 탐지 정확도가 각각 44.1%, 54.6%, 70.8%로 YOLOv9 대비 2.0%, 3.1%, 0.9% 향상되었습니다.



### Bootstrap Segmentation Foundation Model under Distribution Shift via Object-Centric Learning (https://arxiv.org/abs/2408.16310)
Comments:
          This work is accepted by ECCV 2024 EVAL-FoMo Workshop

- **What's New**: 본 논문에서는 SlotSAM이라는 새로운 방법을 소개합니다. 이 방법은 인간의 인지 과정을 모방하여 인코더로부터 피쳐를 자율적으로 재구성하여 객체 중심(object-centric) 표현을 생성합니다. 이러한 표현은 기존의 기초 모델에 통합되어 객체 수준의 지각 능력을 높이며, 분포 관련 변수를 감소시키는 데 기여합니다.

- **Technical Details**: SlotSAM은 기초 모델의 인코더를 활용하여 고수준의 객체 중심 표현을 생성하는 방식으로 작동합니다. 이미지 인코더는 각 객체의 고수준 의미를 효과적으로 추출하여, 픽셀 색상 재구성에 영향을 받지 않는 통일된 표현을 제공합니다. Slot-Attention 기술을 적용하여, 객체 토큰은 전역 이미지 맥락, 기하학적 영역, 의미적 정보에 접근할 수 있으며, 이를 통해 기초 모델의 객체 인식 능력이 크게 향상됩니다.

- **Performance Highlights**: SlotSAM을 통해 수행한 실험에서는 적은 수의 파라미터 조정으로 새로운 환경에서 기초 모델의 일반화 능력이 개선되었습니다. 이 방법은 다양한 작업에 적응 가능하며, 자원의 소비를 낮추면서도 우수한 성능을 보이는 매우 유연한 솔루션입니다.



### Semantics-Oriented Multitask Learning for DeepFake Detection: A Joint Embedding Approach (https://arxiv.org/abs/2408.16305)
- **What's New**: 이 논문에서는 DeepFake 탐지를 위한 semantics-oriented multitask learning을 심층적으로 다루며, 얼굴 속성과 지역 간의 관계를 활용한 joint embedding 기법을 소개합니다. 기존의 방법론에서 벗어나, 자동 데이터셋 확장 기법을 제안하여 기존 DeepFake 데이터셋의 크기를 늘리고 semantically 계층적 레이블링을 포함시킵니다. 또한, 이미지와 해당 레이블을 공유하는 feature space를 활용하여 모델 학습을 자동화합니다.

- **Technical Details**: 새로운 Semantics-oriented Joint Embedding DeepFake Detector (SJEDD)를 제안하며, 이는 bi-level optimization 전략을 통해 다양한 task의 fidelity loss 가중치를 동적으로 조정하여 훈련 과정의 자동화를 이룹니다. SJEDD는 이미지 인코더 매개변수를 모든 작업에 공유하며, 이미지와 레이블 간의 관계를 model capacity allocation 관점에서 최적화합니다.

- **Performance Highlights**: 여섯 개의 DeepFake 데이터세트에서 광범위한 실험을 통해 SJEDD는 18개의 최신 DeepFake 탐지 모델과 비교해 cross-dataset, cross-manipulation, cross-attribute 탐지 성능을 꾸준히 향상시켰습니다.



### Rethinking Sparse Lexical Representations for Image Retrieval in the Age of Rising Multi-Modal Large Language Models (https://arxiv.org/abs/2408.16296)
Comments:
          Accepted to ECCV 2024 Workshops: 2nd Workshop on Traditional Computer Vision in the Age of Deep Learning (TradiCV)

- **What's New**: 본 논문은 이미지 검색을 위한 희소( sparse ) 어휘 표현을 재고하며, 시각적 프롬프트를 지원하는 다중 모달 대형 언어 모델( M-LLMs )을 활용하여 이미지 특징을 추출하고 이를 텍스트 데이터로 변환합니다. 이러한 방법을 통해 자연어 처리에서 사용되는 희소 검색 알고리즘을 이미지 검색 작업에 적용할 수 있도록 합니다.

- **Technical Details**: 연구에서는 키 확장을 위한 데이터 증강 기법을 적용하고, 이미지와 텍스트 데이터 간의 연관성을 평가하여 M-LLM의 이미지 특징 추출을 지원합니다. 또한, 키워드를 검색 쿼리에 반복적으로 추가함으로써 검색 성능이 향상된다는 점을 입증하였습니다.

- **Performance Highlights**: MS-COCO, PASCAL VOC, NUS-WIDE 데이터셋을 기반으로 한 키워드 기반 이미지 검색 시나리오에서 본 시스템은 기존의 비전-언어 모델 기반 방법들보다 우수한 정확도와 재현율( precision and recall ) 성능을 보였습니다. 특히, 키워드를 효과적으로 쿼리에 통합하는 방식으로 성능이 대폭 향상되었습니다.



### Convolutional Neural Network Compression Based on Low-Rank Decomposition (https://arxiv.org/abs/2408.16289)
Comments:
          10 pages, 1 figures

- **What's New**: 본 논문에서는 Variational Bayesian Matrix Factorization (VBMF)와 직교 정규화(orthogonal regularization)를 통합한 새로운 모델 압축 방법을 제안합니다.

- **Technical Details**: 모델은 과잉 매개변수화(over-parameterization) 단계를 거쳐 훈련되며, 직교 정규화를 통해 원래 모델의 정확도에 도달할 가능성을 높입니다. 또한, VBMF를 사용하여 각 층의 weight tensor의 rank를 추정합니다.

- **Performance Highlights**: 높은 압축 비율과 낮은 압축 비율 모두에서 실험 결과, 제안된 압축 모델이 우수한 성능을 보였습니다.



### SAU: A Dual-Branch Network to Enhance Long-Tailed Recognition via Generative Models (https://arxiv.org/abs/2408.16273)
Comments:
          15 pages

- **What's New**: 이번 연구에서는 synthetic data(합성 데이터)를 활용하여 long-tailed distribution(긴 꼬리 분포)의 이미지 인식 문제를 해결하는 접근 방식을 제안합니다. 특히, synthetic-aware와 synthetic-unaware의 두 가지 브랜치로 구성된 새로운 모델 아키텍처를 통해 실제와 합성 데이터를 효과적으로 혼합하여 학습할 수 있는 방안을 모색하였습니다.

- **Technical Details**: 제안된 방법은 크게 두 가지 브랜치로 구성됩니다: 1) synthetic-unaware branch: 합성 데이터와 실제 데이터를 구분하지 않고 혼합하여 분류를 수행합니다. 2) synthetic-aware branch: 실제와 합성 데이터를 구분하고 그 차이를 학습하여 feature extractor(특징 추출기)의 강건성을 향상시킵니다. 또한, K-Nearest Neighbor 기반의 라벨 수정 전략을 도입하여 저품질 데이터의 동적 탐지를 가능하게 하였습니다.

- **Performance Highlights**: CIFAR-10-LT 및 CIFAR-100-LT 데이터셋에서 state-of-the-art(최신 기술) Top-1 정확도를 달성하였으며, 다양한 불균형 요인에 대해 다른 방법들을 크게 초월하는 성능을 보여주었습니다.



### Beyond Uncertainty: Evidential Deep Learning for Robust Video Temporal Grounding (https://arxiv.org/abs/2408.16272)
Comments:
          Ongoing work: 28pages, 19 figures, 7 tables. Code is available at: https://kaijing.space/SRAM/

- **What's New**: 기존 Video Temporal Grounding (VTG) 모델들은 정확성에서는 뛰어나지만 open-world의 도전 과제를 간과하고 있습니다. 이 논문에서는 SRAM이라는 새로운 네트워크 모듈을 소개하여 사용자 입력에 기반하여 불확실성을 동적으로 추정할 수 있게 하고 있습니다.

- **Technical Details**: SRAM은 두 단계의 cross-modal alignment 작업을 활용하여 불확실성을 측정하는 Deep Evidential Regression (DER)을 통합합니다. 이로 인해 모델은 훈련 중에 불확실성을 명확히 정량화하여, 자신의 한계를 인지하고 "모르겠습니다"라고 응답할 수 있게 됩니다. 이研究에서는 기존 DER의 구조적 결함을 해결하기 위한 Geom-regularizer를 개발했습니다.

- **Performance Highlights**: 광범위한 정량적 및 정성적 결과를 통해 SRAM 모듈의 효과성, 강건성 및 해석 가능성을 확인했습니다. 이 연구는 VTG에서 DER이 성공적으로 통합된 첫 번째 사례로 간주됩니다.



### UDD: Dataset Distillation via Mining Underutilized Regions (https://arxiv.org/abs/2408.16268)
Comments:
          PRCV2024

- **What's New**: 본 논문에서는 UDD(Utilization-sensitive Dataset Distillation)라는 새로운 접근 방식을 제안하여, 합성 데이터셋의 활용되지 않는 영역을 식별하고 활용하는 방법을 다룬다. 이는 특히 작은 이미지당 샘플 수(IPC)일 때 중요하게 여겨진다.

- **Technical Details**: UDD 프레임워크는 두 가지 활용되지 않는 영역 탐색 정책(response-based policy 및 data jittering-based policy)을 포함하여, 트레이닝 과정 중에 동적으로 활용되지 않는 영역을 조정하는 능력을 갖추고 있다. 또한, CFC(Category-wise Feature Contrastive) 손실을 도입하여 완전히 다른 카테고리 간의 특징 구분력을 높인다.

- **Performance Highlights**: 실험 결과, UDD 방법이 MNIST, FashionMNIST, SVHN, CIFAR-10 및 CIFAR-100 데이터셋에서 우수한 성능을 보였으며, 특히 CIFAR-10과 CIFAR-100에서 각각 4.0% 및 3.7%의 성능 향상을 달성하였다. 본 논문은 또한 합성 데이터셋 활용도를 평가하기 위한 mUE 지표를 제안하였다.



### Improving Diffusion-based Data Augmentation with Inversion Spherical Interpolation (https://arxiv.org/abs/2408.16266)
- **What's New**: 본 논문에서는 기존의 Diffusion-based 데이터 증강 방법들이 충실도(faithfulness)와 다양성(diversity)이라는 두 가지 중요한 요소를 놓치고 있음을 설명하고, 이를 해결하기 위한 Diffusion-based Inversion Interpolation 방법인 Diff-II를 제안합니다.

- **Technical Details**: Diff-II는 세 가지 주요 단계로 구성됩니다: 1) 카테고리 개념 학습(Category Concepts Learning): 각 카테고리에 대한 개념 임베딩을 학습합니다. 2) 역 보간(Inversion Interpolation): 각 이미지의 역 구성을 계산하고 동일 카테고리에서 무작위로 샘플링한 두 개의 역 구성값 간의 구면 보간을 수행합니다. 3) 이단계 잡음 제거(Two-stage Denoising): 다양한 프롬프트를 사용하여 합성 이미지를 생성하는 과정으로, Coarse-to-Fine 방식으로 진행됩니다.

- **Performance Highlights**: 다양한 이미지 분류 작업(예: few-shot, long-tailed, and out-of-distribution classification)에서 Diff-II의 효과가 입증되었으며, 기존의 최첨단 diffusion 기반 데이터 증강 방법들에 비해 탁월한 성능 향상을 보여주었습니다.



### Low Saturation Confidence Distribution-based Test-Time Adaptation for Cross-Domain Remote Sensing Image Classification (https://arxiv.org/abs/2408.16265)
- **What's New**: 본 논문에서는 Remote Sensing(원거리 감지) 이미지 분류에서 빠른 적응을 실현하기 위한 새로운 Test Time Adaptation(TTA) 기법인 Low Saturation Confidence Distribution Test Time Adaptation(LSCD-TTA)을 제안합니다. 기존의 Source-free Domain Adaptation(SFDA) 방법이 요구하는 많은 Target Domain(대상 도메인) 데이터 없이도 정확하고 빠른 적응을 가능하게 합니다.

- **Technical Details**: LSCD-TTA는 Remote Sensing 이미지의 특성을 고려하여 세 가지 주요 모듈로 구성됩니다: Low Saturation Distribution(LSD)은 낮은 신뢰도의 샘플을 중시하고, Weak-category Cross-Entropy(WCCE)는 분류하기 어려운 카테고리에 가중치를 줍니다. 마지막으로, Diverse Categories Confidence(DIV)는 카테고리의 다양성을 종합적으로 고려하여 샘플 분포의 편향을 완화합니다. 이 세 가지 모듈의 가중치를 조절함으로써, 모델은 사전 데이터 접근이나 수동 주석 없이도 더 빠르고 정확하게 Target Domain에 적응할 수 있습니다.

- **Performance Highlights**: LSCD-TTA는 세 가지 원거리 감지 이미지 데이터셋에서 평가되었으며, Resnet-50과 Resnet-101을 사용한 평균 정확도에서 각각 4.96%-10.51%, 5.33%-12.49% 향상을 보여 기존의 최신 Domain Adaptation(도메인 적응) 및 TTA 방법들에 비해 뛰어난 성능을 나타냈습니다.



### EvLight++: Low-Light Video Enhancement with an Event Camera: A Large-Scale Real-World Dataset, Novel Method, and Mor (https://arxiv.org/abs/2408.16254)
Comments:
          Journal extension based on EvLight (arXiv:2404.00834)

- **What's New**: 이 논문에서는 30,000쌍 이상의 실제 조명을 가진 이미지와 이벤트 데이터로 구성된 대규모 데이터셋인 SDE를 소개합니다. 이를 통해 새로운 이벤트 기반 저조도 비디오 향상 방법인 EvLight++를 제안합니다.

- **Technical Details**: EvLight++는 다중 스케일 전반적 융합 분기를 설계하여 이미지와 이벤트의 구조 및 질감 정보를 통합합니다. SNR(신호 대 잡음 비율)을 사용하여 지역 특성 선택을 통해, 고 SNR 지역에서 이미지를 기록하고, 저 SNR 지역에서는 이벤트에서 구조 정보를 추출하여 향상시킵니다. 또한, 재귀 모듈과 시간적 손실을 도입하여 시간적 일관성을 보장합니다.

- **Performance Highlights**: EvLight++는 기존의 프레임 기반 및 이벤트 기반 방법보다 각각 1.37dB 및 3.71dB 향상된 성능을 기록하였습니다. 또한 저조도 장면에서 실험을 통해 의미 세분화에서 15.97%의 개선을 달성하였습니다.



### Anno-incomplete Multi-dataset Detection (https://arxiv.org/abs/2408.16247)
Comments:
          12 pages, 9 figures

- **What's New**: 본 논문에서는 'Annotation-incomplete Multi-dataset Detection' 문제를 제안하고, 이를 해결하기 위해 다중 부분 주석 데이터셋에서 모든 객체 카테고리를 정확하게 감지할 수 있는 엔드-투-엔드 멀티 태스크 학습 아키텍처를 개발했습니다.

- **Technical Details**: 제안된 방법론에서는 주의 기반 특징 추출기(attention feature extractor)를 도입하여 서로 다른 데이터셋 간의 관계를 파악합니다. 또한 서로 다른 출처에서의 이질적인 특징을 처리하기 위한 지식 융합 훈련 전략(knowledge amalgamation training strategy)을 통합하였습니다. 이는 각 가지에 대해 감독하는 교사 모델(teacher model)과의 학습을 통해 이루어집니다.

- **Performance Highlights**: 다양한 객체 검출 데이터셋에서 실험을 진행한 결과, COCO와 VOC 데이터셋에서 각각 2.17%, 2.10%의 mAP(median Average Precision) 향상을 달성하였으며, 이는 기존의 최첨단 방법들보다 개선된 성능입니다.



### Neural Spectral Decomposition for Dataset Distillation (https://arxiv.org/abs/2408.16236)
Comments:
          ECCV 2024

- **What's New**: 본 논문에서는 Neural Spectrum Decomposition이라는 새로운 데이터셋 증류(datasets distillation) 프레임워크를 제안합니다. 기존 방법과는 달리 전체 데이터셋을 여러 차원에서 저랭크(low-rank) 상태로 간주하여 효율적인 증류를 추구합니다.

- **Technical Details**: 이프레임워크는 스펙트럼 텐서(spectrum tensors)와 변환 매트릭스(transformation matrices)를 학습합니다. 이를 통해 데이터 분포를 재구성할 수 있으며, 다양한 스펙트럼 벡터와 변환 매트릭스의 조합을 사용하여 효율적인 정보 공유(information sharing)가 이루어집니다. 또한, 실제 분포(real distribution)에 의해 안내되는 궤적 매칭 최적화 방법(trajecory matching optimization)을 통합하여 증류 학습의 효율성을 높입니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 CIFAR10, CIFAR100, Tiny Imagenet 및 ImageNet Subset 벤치마크에서 최고 성능(state-of-the-art)을 달성하였으며, 특히 CIFAR10에서 궤적 매칭 기반의 기준 방법에 비해 47.9%의 성능 개선을 보였습니다.



### LMT-GP: Combined Latent Mean-Teacher and Gaussian Process for Semi-supervised Low-light Image Enhancemen (https://arxiv.org/abs/2408.16235)
- **What's New**: 저희는 저조도 이미지 향상(Low-light Image Enhancement, LLIE)을 위해 라텐트 평균 교사(latent mean-teacher) 및 가우시안 프로세스(Gaussian Process)를 기반으로 한 반지도 학습 방법론인 LMT-GP를 제안합니다. 이 방법은 레이블이 있는 데이터와 레이블이 없는 데이터를 효율적으로 결합하여 네트워크 학습을 개선합니다.

- **Technical Details**: LMT-GP는 레이블이 있는 데이터에서 추출한 라텐트 벡터와 레이블이 없는 데이터에서의 라텐트 및 의사 라텐트 벡터 간의 연결을 구축하는 평균 교사 지원 가우시안 프로세스 학습 전략을 사용합니다. 또한, 지원 가우시안 회귀 손실 함수를 통해 학습 과정을 안내합니다. Pseudo-label Adaptation Module (PAM)을 설계하여 신뢰성 있는 네트워크 학습을 보장합니다.

- **Performance Highlights**: 실험 결과, LMT-GP는 여러 저조도 이미지 향상 데이터셋 및 고급 비전 작업에 적용하여 높은 일반화 성능과 이미지 품질을 달성했습니다.



### PSE-Net: Channel Pruning for Convolutional Neural Networks with Parallel-subnets Estimator (https://arxiv.org/abs/2408.16233)
Comments:
          10pages, Neural Networks

- **What's New**: PSE-Net을 소개하며, 효율적인 채널 프루닝을 위한 새로운 병렬 서브넷 추정기를 개발하였습니다. 이를 통해 다양한 서브넷을 한 번의 학습 라운드에서 동시에 훈련할 수 있으며, 결과적으로 슈퍼넷 학습 효율성을 크게 향상시켰습니다.

- **Technical Details**: PSE-Net은 병렬 서브넷 훈련 알고리즘을 기반으로 하며, 배치 차원에서 불필요한 특징을 제거하여 여러 서브넷의 전방향 및 역방향 패스를 시뮬레이션합니다. 이전 분포 기반 샘플링 알고리즘을 도입하여 최적의 서브넷 탐색 시 자원 제약을 만족하는 샘플을 발견하는 데 도움을 줍니다.

- **Performance Highlights**: ImageNet 데이터셋에서 PSE-Net은 기존의 최첨단 채널 프루닝 방법을 초월하며 슈퍼넷 학습 효율성을 유지합니다. 예를 들어, 300M FLOPs 제약하에 프루닝된 MobileNetV2는 ImageNet 데이터셋에서 75.2% Top-1 정확도를 달성했으며, 이는 원래 MobileNetV2에 비해 2.6유닛 향상된 결과입니다.



### Enhancing Conditional Image Generation with Explainable Latent Space Manipulation (https://arxiv.org/abs/2408.16232)
Comments:
          7 pages , 5 figures

- **What's New**: 이 논문은 조건부 프롬프트에 맞춰 참조 이미지에 대한 충실도를 달성하기 위한 새로운 접근 방식을 제안합니다. 기존 방법이 가진 한계를 극복하고자, 확산 모델(diffusion model)과 잠재 공간 조작(latent space manipulation), 그라디언트 기반 선택적 주의 메커니즘(gradient-based selective attention mechanisms)을 통합하였습니다.

- **Technical Details**: 이 연구에서는 Grad-SAM (Gradient-based Selective Attention Manipulation)을 활용하여 디노이즈된 잠재 벡터와 관련된 중요 점수를 도출하고, 특정 시점에서 마스크를 생성하여 객체를 보존하면서 참조 이미지의 특성을 통합합니다. 이는 텍스트 조건에 따라 충실한 객체 형성을 보장하면서 배경을 정제하여 더 일관된 구성을 제공합니다.

- **Performance Highlights**: Places365 데이터셋을 사용한 실험에서, 제안한 모델이 평균 및 중앙값 Frechet Inception Distance (FID) 점수에서 기존 모델보다 우수한 결과를 보여주었으며, 높은 CLIP 점수를 통해 생성된 이미지와 제공된 텍스트 설명 간의 정렬 성능이 경쟁력을 가진다는 것을 입증했습니다.



### Revisiting 360 Depth Estimation with PanoGabor: A New Fusion Perspectiv (https://arxiv.org/abs/2408.16227)
- **What's New**: PGFuse는 모노큘러 360 이미지에서의 깊이 추정을 위한 새로운 패노라마 Gabor 기반 융합 프레임워크이다. 이 프레임워크는 기존의 왜곡 문제를 해결하는 데 중점을 두고 설계되었다.

- **Technical Details**: 이 연구에서는 Gabor 필터를 사용하여 주파수 영역에서 텍스처를 분석하며, 비뚤어짐을 인식하는 Gabor 필터(PanoGabor)를 설계한다. 이러한 필터는 채널-단방향 및 공간-단방향 융합 모듈(CS-UFM)에 통합되어, 다른 표현 방식들을 ERP 형식으로 통합한다.

- **Performance Highlights**: 세 가지 인기 있는 360 기준 데이터셋에서 평가된 결과, PGFuse는 기존 최첨단 방법들보다 우수한 성능을 보였다. 특히, 깊이 추정 정확도와 효율성 간에 최상의 균형을 달성하였다.



### LLaVA-SG: Leveraging Scene Graphs as Visual Semantic Expression in Vision-Language Models (https://arxiv.org/abs/2408.16224)
- **What's New**: 최근 큰 비전 언어 모델(VLM)에서 Vision Transformer(ViT) 아키텍처를 기반으로 한 비전 인코더가 사용됩니다. 본 논문에서는 Scene Graph Expression(SGE) 모듈을 VLM에 적용하여 이러한 제약을 극복하고자 합니다. 이 모듈은 이미지 내 복잡한 의미 정보를 구조적으로 표현하고 VLM의 인식 능력을 개선합니다.

- **Technical Details**: SGE 모듈은 이미지에서 개체를 추출하고 이들 간의 관계를 표현하기 위한 장면 그래프(Scene Graph)를 구성합니다. 이는 개체 수준에서 의미 정보를 유지하고 전달하는 방법으로, 이미지 내 객체를 태깅하고, 경계 상자(Bounding Box)를 감지하여 세분화 세마틱 이해(Semantic Segmentation)를 수행합니다. 최종적으로 LLaVA-SG 모델로 통합하여 VLM의 시각적 이해 능력을 향상시킵니다.

- **Performance Highlights**: 광범위한 실험을 통해 SGE 모듈 통합이 VLM의 시각 언어 작업에서 성능을 상당히 향상시키는 것으로 나타났습니다. 이는 복잡한 의미 세부 사항을 보존하고 시각적 이해를 강화하는 데 효과적임을 의미합니다.



### Training-free Video Temporal Grounding using Large-scale Pre-trained Models (https://arxiv.org/abs/2408.16219)
Comments:
          Accepted by ECCV 2024

- **What's New**: 본 논문에서는 대규모 사전 훈련된 모델을 활용한 Training-Free Video Temporal Grounding (TFVTG) 접근 방식을 제안합니다. 기존의 video temporal localization 모델들이 특정 데이터셋에 의존하고 있어 일반화 능력이 떨어지는 한계를 극복하고자 하였습니다.

- **Technical Details**: TFVTG는 큰 언어 모델(LLMs)과 비주얼 언어 모델(VLMs)을 결합하여, 쿼리 텍스트 내의 다수의 하위 이벤트를 분석하고 이들 이벤트 간의 시간적 순서 및 관계를 이해합니다. 구체적으로, 각 하위 이벤트를 동적 전환과 정적 상태로 나누어 VLM을 통해 관련성을 평가합니다. 또한, 하위 이벤트 간의 관계를 LLM을 통해 필터링하고 통합합니다.

- **Performance Highlights**: 우리의 방법은 Charades-STA 및 ActivityNet Captions 데이터셋에서 zero-shot video temporal grounding의 최상의 성능을 달성하였으며, cross-dataset 및 OOD 설정에서도 뛰어난 일반화 능력을 보였습니다.



### M4CXR: Exploring Multi-task Potentials of Multi-modal Large Language Models for Chest X-ray Interpretation (https://arxiv.org/abs/2408.16213)
- **What's New**: 이번 논문에서는 CXR(Chest X-ray) 해석을 강화하기 위해 M4CXR이라는 다중 모달(Modal) LLM을 제안합니다. 이 모델은 다양한 작업별 데이터 세트를 통합한 시각적 지침 따르기 데이터 세트에서 훈련되었습니다.

- **Technical Details**: M4CXR은 여러 작업을 지원하며, 특히 의학 보고서 생성(MRG), 시각적 그라운딩(Visual Grounding), 시각적 질문 응답(VQA)에서 탁월한 성능을 보입니다. 모델의 훈련 과정에서는 Chain-of-Thought(CoT) 프롬프트 전략을 사용하여 CXR 이미지에서 발견 사항을 식별하고 해당 보고서를 생성합니다.

- **Performance Highlights**: M4CXR은 다양한 MRG 시나리오에서 탁월한 임상 정확도를 달성하며, 시각적 그라운딩과 VQA에서도 Specialized 모델에 필적하는 성능을 발휘합니다. 양적 및 질적 평가 모두에서 M4CXR의 다재다능성을 강조하며, 일관된 임상 정확도를 유지합니다.



### Uni-3DAD: GAN-Inversion Aided Universal 3D Anomaly Detection on Model-free Products (https://arxiv.org/abs/2408.16201)
- **What's New**: 본 논문에서는 기존의 3D 이상 탐지 방법의 한계를 극복하기 위해 기계 학습을 활용한 새로운 통합 접근 방식을 제안합니다. 이 방법은 전통적인 CAD 파일에 의존하지 않고, 모델 없는 제품에 대한 결함을 효과적으로 감지하는 데 중점을 둡니다.

- **Technical Details**: 제안된 방법은 특징 기반(detection module) 탐지 모듈과 재구성 기반(reconstruction-based detection module) 탐지 모듈을 통합하여 작동합니다. 특징 기반 탐지는 움푹 들어간 자국, 구멍, 균열 등과 같은 기하학적 결함을 다루고, 재구성 기반 방법은 결측 영역을 탐지합니다. 이 두 모듈의 탐지 결과는 One-class Support Vector Machine (OCSVM)을 통해 융합됩니다.

- **Performance Highlights**: 본 연구의 결과는 모델 없는 제품에 대한 결함 인식에서 최신 기법보다 향상된 성능을 보여주며, 특히 불완전한 형태를 탐지하는 데 탁월한 능력을 보입니다. 또한 모든 다른 유형의 이상을 탐지하는 데 있어서도 기존 방법과 동등한 성능을 유지합니다.



### PolarBEVDet: Exploring Polar Representation for Multi-View 3D Object Detection in Bird's-Eye-View (https://arxiv.org/abs/2408.16200)
Comments:
          11 pages, 6 figures

- **What's New**: 최근의 LSS(Lift-Splat-Shoot) 기반 다각도 3D 객체 탐지 기술이 자율 주행에 있어 경제적이고 배치하기 쉬운 솔루션을 제공하지만, 기존 방법들은 비균일한 이미지 정보 분포를 고려하지 않고, 뷰 대칭성을 잘 활용하지 못하는 문제를 다루었습니다. 본 논문에서는 Polar BEV(Polar Bird's-Eye-View) 표현을 도입하여 이를 개선하고 새로운 다각도 3D 객체 탐지기인 PolarBEVDet를 제안하였습니다.

- **Technical Details**: Polar BEV 표현은 이미지 정보 분포에 맞춰 그리드 배치를 조정하고, 주변 카메라의 뷰 대칭성을 보존하기 위해 세 가지 모듈을 설계하였습니다. 1) Polar View Transformer: 다각도 이미지 특성을 원통형 좌표계로 변환하고 BEV 공간을 각도 및 반경 방향으로 표시합니다. 2) Polar Temporal Fusion Module: 다중 프레임 폴라 BEV 특성을 정렬합니다. 3) Polar Detection Head: Polar 매개변수를 활용한 3D 객체 탐지를 수행합니다. 또한 2D 보조 탐지 헤드와 공간 주의 강화(SAE) 모듈을 디자인하여 특징 추출의 품질을 높였습니다.

- **Performance Highlights**: nuScenes 데이터셋에서 평가한 결과, PolarBEVDet는 63.5% NDS(Neighborhood Detection Score)와 55.8% mAP(mean Average Precision)로 뛰어난 탐지 성능을 달성하였습니다.



### DLM-VMTL:A Double Layer Mapper for heterogeneous data video Multi-task prompt learning (https://arxiv.org/abs/2408.16195)
- **What's New**: 최근 비디오 이해(Understanding) 작업의 파라미터 수가 증가하면서 모델의 재학습이나 특정 작업을 위한 사전 훈련이 많은 오버헤드를 발생시키는 문제가 발생하고 있습니다. 본 논문에서는 이러한 문제를 해결하기 위해 이질적인 데이터 비디오 다중 작업 프롬프트 학습(VMTL) 방법을 제안합니다.

- **Technical Details**: 제안된 DLM-VMTL(두 층 매퍼)은 보조 작업의 중간 계층에서 얻은 지식 기반 프롬프트를 주 작업에 적용하여 공유 가능한 지식을 제공합니다. 첫 번째 층은 자기 주의(self-attention) 메커니즘을 사용하여 보조 작업의 중간 계층 표현과 상호작용하는 프롬프트를 계산합니다. 두 번째 층은 주 작업의 표현과 정렬하도록 매핑합니다.

- **Performance Highlights**: DLM-VMTL은 6개의 비디오 이해 작업과 11개의 데이터셋에서 베이스라인보다 성능이 우수함을 입증했습니다. 특히 DLM-VMTL을 사용하여 주 작업의 성능과 일반화를 향상시키면서 총 파라미터의 10.8%를 줄일 수 있었습니다.



### Estimating Dynamic Flow Features in Groups of Tracked Objects (https://arxiv.org/abs/2408.16190)
Comments:
          21 pages, 6 figures

- **What's New**: 이 논문에서는 불완전한 추적기(perfect tracers)로 구성된 복잡한 이미지 시퀀스에서 동역학적 시스템(dynamical systems)의 흐름을 이해하기 위해 기울기 기반(gradient-based) 분석을 확장하는 것을 목적으로 하고 있습니다.

- **Technical Details**: 이 연구에서는 Lagrangian gradient regression(LGR) 방법을 사용하여 희소한 데이터로부터 공간 기울기(spatial gradients)를 추정합니다. 주요 단계는 객체 탐지(detection), 추적(tracking), 그리고 희소 기울기 추정(sparse gradient estimation)으로 구성됩니다. 이 방법은 비전 네트워크(vision networks)를 통해 객체 이동의 궤적을 추적하고 이를 통해 회전 영역과 수송 장벽(transport barriers)과 같은 동역학 특성을 도출합니다.

- **Performance Highlights**: 제안된 방법은 다양한 클래스의 객체를 동시에 분석할 수 있으며, 드론 관측(drones)과 같은 모션이 결과에 영향을 미치지 않도록 설계되었습니다. 이 연구는 실험적 잔해 흐름(case study)과 현장 데이터(field data)를 통해 제안된 방법을 검증하였습니다.



### VLM4Bio: A Benchmark Dataset to Evaluate Pretrained Vision-Language Models for Trait Discovery from Biological Images (https://arxiv.org/abs/2408.16176)
Comments:
          36 pages, 37 figures, 7 tables

- **What's New**: 이 논문은 12개의 최첨단(VLM) 모델의 효용성을 평가하며, 생물체 생물학 분야에서 469K의 질문-답변 쌍으로 구성된 VLM4Bio 데이터셋을 사용하여 VLM의 제로샷(zero-shot) 기능을 검토합니다. 이 연구는 기존의 VLM들이 생물학적 질문에 응답하는 데 있어 필요한 지식을 포함하고 있는지 평가합니다.

- **Technical Details**: 연구에서 사용된 VLM4Bio 데이터셋은 30K 이미지와 이에 대한 469K 질문-답변 쌍으로 구성되어 있으며, 주요 태스크로는 종 분류(species classification), 형질 식별(trait identification), 형질 기초화(trait grounding), 형질 참조(trait referring), 그리고 형질 계산(trait counting) 등이 포함됩니다. VLM의 성능을 테스트하기 위해, 연구자는 VLM의 결과를 유도하기 위한 prompting 기술 및 reasoning hallucination에 대한 테스트를 적용했습니다.

- **Performance Highlights**: 이 연구에서 SOTA VLM들은 다양한 생물학적 질문을 이미지 기반으로 정확히 응답하는 능력을 가지고 있음을 보여주었다. 특히, 종 분류 및 형태학적 특성 위치 지정 작업에서 높은 정확도를 기록하였으며, 이는 생물체 생물학의 과학적 발견을 가속화할 잠재력이 있음을 시사합니다.



### Does Data-Efficient Generalization Exacerbate Bias in Foundation Models? (https://arxiv.org/abs/2408.16154)
Comments:
          Preprint of paper to be presented at Fairness and Ethics Towards Transparent AI: Facing the Challenge through Model Debiasing (FAILED) during ECCV 2024

- **What's New**: 이 연구에서는 Foundation 모델(특히 RetFound)이 브라질의 다중 레이블 검안 데이터셋(BRSET)을 기반으로 한 의료 이미징에서의 편향을 평가합니다. 특히, 이전 데이터셋과 인구가 다른 BRSET에서 파인튜닝을 통해 성별과 연령 집단 사이의 AUC 격차를 감소시킬 수 있는 가능성을 탐구합니다.

- **Technical Details**: RetFound 모델은 Masked Autoencoder(MAE)를 사용하여 자가 감독 학습(self-supervised learning)으로 사전 훈련되었습니다. 연구는 Utility, Group Fairness, Max-Min Fairness의 세 가지 지표를 사용하여 모델의 편향을 평가하며, AUC(수신자 조작 특성 곡선 아래 면적)를 기준으로 성능을 비교합니다. 데이터는 60% 훈련, 10% 검증, 30% 테스트로 나뉘어 사용되었습니다.

- **Performance Highlights**: RetFound 모델의 데이터 효율적인 일반화가 효과적이지만, 훈련 데이터가 줄어들수록 성별과 연령에 따른 AUC 격차가 증가하여 공정성 문제를 야기할 수 있음을 발견했습니다. 이 연구는 실제 환경에서 제한된 데이터를 사용할 때 Foundation 모델의 공정성 문제를 고려해야 함을 시사합니다.



### Using Backbone Foundation Model for Evaluating Fairness in Chest Radiography Without Demographic Data (https://arxiv.org/abs/2408.16130)
Comments:
          Preprint of paper to be presented at Fairness of AI in Medical Imaging (FAIMI) during MICCAI 2024

- **What's New**: 이 연구는 의료 영상 진단의 공정성을 보장하기 위한 새로운 접근 방식을 제안합니다. 기초 모델(Foundation Models)을 사용하여 성별 및 연령과 같은 보호 속성을 나타내는 그룹을 생성하고, 이를 이용해 편향 완화(bias mitigation) 기술의 효과를 분석합니다.

- **Technical Details**: 제안된 방법론은 CheXpert와 NIH 데이터셋을 포함하며, DBSCAN 알고리즘을 이용해 유사한 이미지 특성을 가진 샘플을 클러스터링 합니다. 이 과정 중에 보호 속성을 나타내는 클러스터를 생성하여, 성별 편향을 4.44% 감소시키고, 이 분포之外에서 6.16% 감소시키는 효과를 보였습니다.

- **Performance Highlights**: 제안된 방법은 성별에 대해 더 강건한 성능을 보였지만, 연령 속성에서는 여전히 개선이 필요하다는 점이 강조되었습니다. 이는 더 기본적으로 공정하고 강력한 기초 모델 개발의 필요성을 제시합니다.



### ChartEye: A Deep Learning Framework for Chart Information Extraction (https://arxiv.org/abs/2408.16123)
Comments:
          8 Pages, and 11 Figures

- **What's New**: 이 연구에서는 차트 이미지에서 정보를 자동으로 추출하기 위한 딥러닝 기반 프레임워크를 제안합니다. 이 프레임워크는 차트 유형 및 텍스트 역할 분류를 위해 계층적 비전 트랜스포머(hierarchical vision transformers)를 사용하고, 텍스트 감지는 YOLOv7을 활용합니다. 또한, 인식 결과를 개선하기 위해 Super Resolution Generative Adversarial Networks(ESRGAN)를 사용하여 텍스트를 강화합니다.

- **Technical Details**: 제안된 프레임워크는 차트 유형 분류, 텍스트 감지 및 인식, 텍스트 역할 분류의 각 주요 단계를 처리합니다. 이 프레임워크는 딥 컨볼루션 및 비전 트랜스포머 기반 접근 방식을 결합하여 차트 이미지에서 효과적으로 정보를 추출합니다. 특히, 차트 유형 분류 및 텍스트 역할 분류에는 계층적 비전 트랜스포머를 사용하며, 텍스트 감지를 위해서는 한 단계(object detection) 객체 탐지기를 채택합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 차트 유형 분류에서 0.97, 텍스트 역할 분류에서 0.91, 텍스트 감지에서 평균 정밀도(Mean Average Precision) 0.95의 뛰어난 성능을 달성했습니다.



### 3D Reconstruction with Spatial Memory (https://arxiv.org/abs/2408.16061)
Comments:
          Project page: \url{this https URL}

- **What's New**: 본 논문에서는 DUSt3R 패러다임을 기반으로 한 새로운 접근 방식인 Spann3R를 소개합니다. Spann3R는 이미지 집합으로부터 점맵(pointmaps)을 직접 추정할 수 있는 transformer 기반 아키텍처를 활용하여, 이전의 최적화 기반 글로벌 정렬 없이 글로벌 좌표계에서의 점맵 예측을 가능하게 합니다.

- **Technical Details**: Spann3R는 외부 공간 메모리를 관리하여 이전의 3D 정보를 추적하고, 이를 바탕으로 다음 프레임의 3D 구조를 예측하는 방식으로 작동합니다. 기존의 DUSt3R 보완학습 가중치를 활용하여, 프레임 간 단기 및 장기 의존성을 학습할 수 있도록 설계되었습니다. 경량화된 transformer 기반 메모리 인코더를 사용하여 이전 예측을 메모리 값으로 인코딩합니다.

- **Performance Highlights**: Spann3R는 다양한 새로운 데이터셋에서 경쟁력 있는 밀집 재구성 품질과 일반화 능력을 보여주었으며, 최적화 없이 50프레임 이상 실시간으로 점진적인 재구성이 가능합니다.



### Many-Worlds Inverse Rendering (https://arxiv.org/abs/2408.16005)
- **What's New**: 본 논문은 물리 기반 역 렌더링에서 표면을 최적화하는 과정에서 발생하는 불연속적인 가시성 변화 문제를 해결하기 위해 새로운 접근 방법을 제안합니다. 기존의 알고리즘 대신, 임의의 표면을 국소적으로 미분하지 않고 표면의 체적 변화를 미분하는 방법을 채택합니다. 이를 'many-worlds representation'이라고 하며, 이는 구성 요소들 간의 비상호작용적인 겹침을 모델링합니다.

- **Technical Details**: 새로운 전송 법칙을 통해 Monte Carlo 알고리즘을 단순화하고, 각 '세계'가 서로 독립적으로 광학적으로 구분됩니다. 이 모델은 표면에 적절한 perturbation을 더하여, 최적화 단계에서의 비용을 줄이고, 명확한 매쉬 추출이 용이하게 이루어집니다. 논문은 이 접근 방식이 불연속적인 가시성 변화를 단순화하며, 기존 방법보다 더 간단하고 효율적이라고 주장합니다.

- **Performance Highlights**: 제안된 방법은 총 반복 횟수와 각 반복에 필요한 비용 모두에서 빠른 수렴을 촉진함을 보여줍니다. 불필요한 복잡성을 줄이고, 표면에 대한 근접 최적화를 손쉽게 수행할 수 있도록 합니다.



### Meta-Learning for Federated Face Recognition in Imbalanced Data Regimes (https://arxiv.org/abs/2408.16003)
Comments:
          To appear in the IEEE FLTA 2024 proceedings

- **What's New**: 이 논문은 Federated Face Recognition (FFR) 분야에서 개인화된 Federated Learning (FL)의 해결책을 제시합니다. CelebA 데이터셋을 기준으로 서로 다른 형태의 데이터 이질성을 기반으로 한 세 가지 새로운 데이터 파티션을 소개합니다.

- **Technical Details**: 제안된 방법은 Hessian-Free Model Agnostic Meta-Learning (HF-MAML)입니다. 이 접근법은 전통적인 MAML 알고리즘의 계산 비용을 줄이며, FL 환경에서 효과적으로 작동할 수 있도록 설계되었습니다. 추가적으로, 손실 함수에 embedding regularization 항목을 도입하여 글로벌 모델의 성능을 향상시킵니다.

- **Performance Highlights**: HF-MAML은 CelebA 데이터 파티션에서 기존 FFR 모델들보다 높은 검증 점수를 기록했습니다. 특히, 데이터 이질성이 큰 파티션에서 개선된 검증 점수를 보여주었으며, 공정성 분석에서도 HF-MAML과 주입된 편향 정규화가 클라이언트 평가 점수의 표준 편차를 줄이는 데 기여함을 확인했습니다.



### Sparse Signal Reconstruction for Overdispersed Low-photon Count Biomedical Imaging Using $\ell_p$ Total Variation (https://arxiv.org/abs/2408.16622)
Comments:
          5 pages, Accepted by the IEEE International Symposium on Biomedical Imaging (ISBI)

- **What's New**: 이 논문에서는 신호 복원에 대한 새로운 접근 방식을 제안합니다. 특히, negative binomial (NB) 모델을 활용하여 $	au > 0$의 정규화 파라미터를 설정하고, $	ext{ℓ}_p$ TV quasi-seminorm을 사용하여 희소성을 촉진하는 방법을 조사합니다. 이 연구는 저광자 신호 처리의 어려움을 해결하기 위해 다양한 정규화 항을 평가합니다.

- **Technical Details**: 이 논문은 negative binomial 확률 모델을 기반으로 하는 최적화 문제를 제기합니다. 그 과정에서 $	ext{ℓ}_p$ quasi-seminorm과 total variation (TV) quasi-seminorm을 사용하고, 이를 gradient 기반 방법을 통해 해결합니다. 제안된 알고리즘은 Barzilai-Borwein 접근법을 통해 해안의 Hessian을 계산할 필요 없이 더 나은 효율성을 보입니다.

- **Performance Highlights**: 실험 결과는 제안된 방법이 기존의 Poisson 모델과 비교했을 때 이미지 복원 성능을 개선했음을 나타냅니다. 제안된 $	ext{ℓ}_p$ TV quasi-seminorm의 사용은 신호 복원에 있어 유의미한 성능 향상을 보여 주었습니다.



### A Deep-Learning-Based Lable-free No-Reference Image Quality Assessment Metric: Application in Sodium MRI Denoising (https://arxiv.org/abs/2408.16481)
Comments:
          13 pages, 3 figures

- **What's New**: 본 논문은 기존의 영상 품질 평가(No-reference Image Quality Assessment, NR-IQA) 기법의 한계를 극복하기 위해 새로운 딥러닝 기반 NR-IQA 지표인 Model Specialization Metric (MSM)을 제안합니다. MSM은 고품질 기준 이미지나 레이블에 의존하지 않고 입력 이미지를 평가합니다.

- **Technical Details**: MSM은 입력 이미지와 모델의 예측 간의 차이를 측정하여 이미지 품질을 평가하는 방법입니다. 이 방식은 고유한 훈련 세트에 대해 특화된 딥러닝 모델의 특성을 활용하여, 훈련 데이터와의 차이가 클수록 예측 정확도가 감소함을 이용합니다.

- **Performance Highlights**: MSM은 다양한 시뮬레이션된 잡음과 왜곡에서 우수한 평가 성능을 보였으며, 전문가 평가와의 상관계수(Cohen's Kappa coefficient) 0.6528로 기존 NR-IQA 지표보다 높은 성능을 입증했습니다.



### Improving 3D deep learning segmentation with biophysically motivated cell synthesis (https://arxiv.org/abs/2408.16471)
- **What's New**: 이 논문은 생물 의학 연구에서 3D 세포 배양 모델의 중요성을 강조하며, 이를 위한 고품질의 3D 교육 데이터를 생성하기 위한 새로운 프레임워크를 제시합니다. 이 프레임워크는 생물 물리학적 모델링(biophysical modeling)을 통합하여 세포의 형태와 정렬을 보다 현실적으로 구현합니다.

- **Technical Details**: 이 연구에서는 SimOptiGAN과 같은 기존의 방법과 새로운 GAN 훈련 스킴을 통해 3D 세포 데이터의 세분화(segmentation) 정확도를 개선하는 방법을 제안합니다. 연구의 핵심은 세포의 형태를 실질적으로 묘사하는 3D 훈련 데이터를 생성하고, Mem2NucGAN-P 및 Mem2NucGAN-U와 같은 신기술을 활용하여 핵 신호(nuclei signals)와 관련된 레이블을 동시에 생성하는 것입니다.

- **Performance Highlights**: 정량적 평가에 따르면, 바이오 물리학적 동기화가 적용된 합성 훈련 데이터는 수작업으로 주석을 단 데이터 및 사전 훈련된 모델보다 우수한 성능을 보였습니다. 이는 생물 물리학적 모델링이 합성 훈련 데이터 품질을 향상시키는 데 기여할 수 있음을 보여줍니다.



### Spiking Diffusion Models (https://arxiv.org/abs/2408.16467)
Comments:
          Accepted by IEEE Transactions on Artificial Intelligence

- **What's New**: 최근 스파이킹 신경망(Spiking Neural Networks, SNNs)이 기존의 인공 신경망(Artificial Neural Networks, ANNs)과 비교하여 에너지 소비가 매우 낮고 생물학적 타당성이 높은 점으로 주목받고 있습니다. 본 논문에서는 SNN 기반의 새로운 생성 모델인 스파이킹 확산 모델(Spiking Diffusion Models, SDMs)을 제안하고, 이를 통해 고품질 샘플을 생성하면서 에너지 소비를 대폭 줄일 수 있음을 보여줍니다.

- **Technical Details**: 제안된 SNN 기반 스파이킹 확산 모델(SDMs)은 Temporal-wise Spiking Mechanism (TSM)을 도입하여 생물학적 관점에서 더 많은 시간적 특징을 포착할 수 있게 합니다. 또한, 한층 더 향상된 성능을 위해 Threshold-guided 전략을 적용하여 추가적인 훈련 없이 성능을 최대 16.7%까지 개선할 수 있습니다.

- **Performance Highlights**: 실험 결과는 SDMs가 ANN의 성능과 유사하면서도 몇 번의 스파이킹 시간 단계로 기존 SNN 기반 생성 모델보다 우수한 성능을 발휘함을 보여줍니다. CIFAR-10 데이터셋에서 FID 점수가 기존 SNN 기준을 최대 12배 개선하면서 약 60%의 에너지 소비 절감 효과를 달성하였습니다.



### NeRF-CA: Dynamic Reconstruction of X-ray Coronary Angiography with Extremely Sparse-views (https://arxiv.org/abs/2408.16355)
- **What's New**: 이번 논문에서는 NeRF(Neural Radiance Field)를 기반으로 한 NeRF-CA라는 새로운 방법을 소개합니다. 이는 2D 엑스레이 심장 혈관 조영술(Coronary Angiography, CA)에서의 희소 뷰(sparse-view) 데이터를 기반으로 4D 재구성을 가능하게 합니다. 특히, 심장 동작에 따른 혈관 내 구조를 분리하여 더욱 정밀한 재구성을 실현하고 있습니다.

- **Technical Details**: NeRF-CA는 심장 동작을 활용하여 동적인 관상 동맥 성분과 정적인 배경으로 장면을 분리합니다. 이 과정에서 동적 구조 희소성(dynamic structure sparsity)와 장면 부드러움(scene smoothness) 등의 정규화(regularization) 기법을 적용하여 관상 동맥과 배경을 효과적으로 분리합니다. 또한, 4개의 관상 동맥 조영술 시퀀스만으로도 4D 재구성을 달성하며 이는 임상 워크플로우에 적합하게 설계되었습니다.

- **Performance Highlights**: 제안된 방법은 4D 팬텀 데이터셋을 사용하여 정량적 및 정성적으로 검증하였으며, 기존의 엑스레이 희소 뷰 NeRF 재구성 방법들에 비해 성능에서 뛰어난 결과를 보여줍니다. 특히, 임상적인 맥락에서도 4D 재구성이 가능하도록 최적화되어 있으며, 수작업 상호작용이나 방대한 데이터 셋에 의존하지 않고 효율적인 재구성을 수행합니다.



### Learned Image Transmission with Hierarchical Variational Autoencoder (https://arxiv.org/abs/2408.16340)
- **What's New**: 본 논문에서는 이미지 전송을 위한 혁신적인 계층적 조인트 소스-채널 코딩(HJSCC) 프레임워크를 소개합니다. 이 프레임워크는 계층적 변분 오토인코더(hierarchical variational autoencoder, VAE)를 활용하여, 송신기에서 하향식 및 상승식 경로를 조합하여 원본 이미지의 여러 계층적 표현을 자회귀적으로 생성합니다.

- **Technical Details**: HJSCC 프레임워크에서는 JSCC 인코더를 통해 생성된 표현을 채널 기호(channel symbols)로 직접 매핑하고, 피드백 링크가 있는 시나리오로 확장하여 복잡한 전송 프로세스를 확률적 샘플링 과정으로 모델링합니다. 이는 JSCC와 피드백을 위한 새로운 생성적 형식을 도출하고, 전송 대역폭을 동적으로 조절하여 더 높은 적응성을 제공합니다.

- **Performance Highlights**: 저해상도에서 높은 비율-왜곡 성능(rate-distortion performance)을 유지하면서 다양한 해상도의 이미지에 대한 광범위한 실험을 수행한 결과, 제안된 HJSCC 모델은 기존의 기준선보다 우수한 성능을 발휘하며, 채널 노이즈에 대한 강인성을 증명했습니다.



### Enhanced Control for Diffusion Bridge in Image Restoration (https://arxiv.org/abs/2408.16303)
- **What's New**: 본 논문에서는 저품질 이미지(low-quality images)를 조건으로 하여 확산 다리 모델(diffusion bridge model)의 제어력이 향상된 ECDB 모델을 소개합니다. 이 모델은 이미지 복원(image restoration)에서 성능을 개선하기 위해 조건부 제어(conditional control)를 강조합니다.

- **Technical Details**: ECDB 모델은 네 가지 모듈로 구성됩니다: Denoising Module (DM), Condition Hint Module (CHM), Degradation Feature Module (DFM) 및 Control Module (CM). DM은 GOUB의 기본 모델을 나타내고, CHM은 조건을 처리하여 픽셀 또는 색상과 같은 특성을 추출합니다. DFM은 저하(degradation) 특성을 추출하는 데 목적이 있으며, CM은 융합(fusion)된 특성을 처리하여 향상된 제어 정보를 생성합니다.

- **Performance Highlights**: 실험 결과에 따르면, ECDB 모델은 여러 이미지 복원 작업, 특히 deraining, inpainting 및 super-resolution에서 최첨단 결과(state-of-the-art results)를 달성했습니다.



### Fine-grained Classification of Port Wine Stains Using Optical Coherence Tomography Angiography (https://arxiv.org/abs/2408.16277)
Comments:
          This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 본 연구는 기존의 외부 피부 외관 기반의 포트 와인 반점 (PWS) 분류법 대신, OCT (Optical Coherence Tomography)와 OCTA (Optical Coherence Tomography Angiography)를 이용한 새로운 분류 접근 방식을 제안합니다.

- **Technical Details**: PWS의 하부 조직병리학(hypodermic histopathology) 및 혈관 구조를 기반으로 세분화된 분류 방법을 개발하였으며, PWS를 다섯 가지 유형으로 구분하였습니다. 이 연구는 혈관 형태학과 깊이 정보에 관련된 여섯 가지 메트릭(metrics)을 분석했습니다.

- **Performance Highlights**: 이 연구에서 제안된 다섯 가지 PWS 유형은 기존 분류법에 비해 모든 메트릭에서 유의미한 차이를 보였으며, 이는 PWS 병변의 이질성을 정확하게 반영하는 혈관병리학(angiopathology) 기반의 분류라는 점에서 임상적 시사점이 큽니다.



### Advancing Architectural Floorplan Design with Geometry-enhanced Graph Diffusion (https://arxiv.org/abs/2408.16258)
- **What's New**: 이번 연구에서는 GSDiff라는 새로운 생성 프레임워크를 제안하여 벡터 평면도 설계를 자동화하는 방법을 제시합니다. 이 프레임워크는 구조적 그래프 생성을 통해 벽 교차점과 벽 구간을 예측하여 기하학적 및 의미적 측면을 모두 포착합니다.

- **Technical Details**: GSDiff는 구조적 그래프 G=(V,E)로 평면도를 표현하고, 노드 생성 및 엣지 예측의 두 단계로 나누어 진행됩니다. 여기서 노드는 벽 교차점을 나타내고 엣지는 벽 구간을 나타냅니다. 또한, 혼합 기반 표현에서의 노드 정렬 오차를 최적화하기 위한 정렬 손실 함수와 회귀 보강 방식을 통해 기하학적 인식을 개선합니다.

- **Performance Highlights**: GSDiff 프레임워크는 기존 기술보다 모든 측정 지표에서 우수한 성능을 보이며, 자유 생성 및 제약 생성 모두를 가능하게 하여 건축 설계 구조 생성의 변화를 이끌어냅니다.



### Single-Photon 3D Imaging with Equi-Depth Photon Histograms (https://arxiv.org/abs/2408.16150)
- **What's New**: 이 논문은 Equi-Width (EW) 히스토그램 대신 Equi-Depth (ED) 히스토그램을 기반으로 한 3D 감지 기술을 제안합니다. ED 히스토그램은 타임스탬프 데이터를 효율적으로 압축하여 대역폭 요구 사항을 감소시키며, 경량화를 통해 픽셀 내 메모리 요구 사항도 줄입니다.

- **Technical Details**: 제안된 알고리즘은 특정 양자화(q-quantile)에 대한 ED 히스토그램의 빈 경계(locations)를 전자적 저장 없이 추정하는 방법을 포함합니다. 이 방법은 미래의 픽셀 내 구현을 위한 가능성을 고려하고 있으며, 고품질 거리 맵을 생성하기 위해 ED 경계 위치에서의 시공간 상관관계를 활용하는 학습 기반 거리 추정 기술을 설계했습니다.

- **Performance Highlights**: ED 히스토그램을 이용한 데이터 처리 방식은 기존 EW 기반 SPC에 비해 메모리를 현저히 절약하며, 대역폭 요구 사항 또한 크게 줄어듭니다. 논문에서는 RGBD 시각 이동, 3D 장면 재구성, RGBD 의미 분할 등의 다양한 컴퓨터 비전 작업에 대해 ED 빈 경계 사용의 장점을 보여주었습니다.



### Alternating Direction Method of Multipliers for Negative Binomial Model with The Weighted Difference of Anisotropic and Isotropic Total Variation (https://arxiv.org/abs/2408.16117)
Comments:
          6 pages, Accepted by the IEEE International Conference on Multimedia and Expo (ICME)

- **What's New**: 본 논문은 오버디스퍼스드 포아송 노이즈로 인해 손상된 이미지를 회복하기 위한 최적화 접근법을 제안합니다. 기존의 포아송 분포 모델이 평균과 분산이 같다고 가정하는 제한점을 넘어서, 변수의 분산이 평균보다 큰 경우에 적합한 음의 이항 분포(negative binomial distribution)를 사용합니다.

- **Technical Details**: 논문에서는 가중화된 비등방성-등방성 총 변동 정규화자(Anisotropic-Isotropic Total Variation Regularizer)를 적용하여 일반적인 총 변동 정규화자에서 발생하는 계단형 아티팩트를 피합니다. 최적화 방법으로는 다중자 방향 교대 방법(Alternating Direction Method of Multipliers, ADMM)을 사용하며, 각 서브 문제는 닫힌 형태의 해를 가지고 있습니다.

- **Performance Highlights**: 수치 실험 결과, 제안한 접근법은 특히 포톤 수가 매우 제한된 환경에서 효과적으로 작동함을 보여주었습니다.



### Negative Binomial Matrix Completion (https://arxiv.org/abs/2408.16113)
Comments:
          6 pages, Accepted by the IEEE International Workshop on Machine Learning for Signal Processing (MLSP)

- **What's New**: 이번 연구에서는 Poisson 분포를 따르는 데이터 대신, 변동성이 평균보다 큰 실제 데이터 상황에서의 적용성을 고려하여 음의 이항(Negative Binomial, NB) 분포를 기반으로 한 매트릭스 완성(Matrix Completion) 기법을 제안합니다. 이를 통해 더 일반적인 환경에서의 매트릭스 복원 가능성을 높였습니다.

- **Technical Details**: 제안된 모델은 핵 노름(nuclear norm) 정규화를 통해 저랭크(low-rank) 매트릭스를 복원하는 방식이다. 근접 기울기 하강법(proximal gradient descent)을 이용하여 모델을 최적화하며, NB 분포에 기반한 손실 함수를 통해 노이즈를 처리하는 접근을 취합니다. 매트릭스의 각 요소는 독립적으로 관측되며, 각 요소는 NB 분포의 확률 질량 함수(probability mass function)에 따라 분포합니다.

- **Performance Highlights**: 실험 결과, 제안된 NB 모델이 다양한 노이즈 및 결측 데이터 상황에서 기존의 Poisson 매트릭스 완성 방법보다 성능이 뛰어남을 입증했습니다. 실제 데이터에서 효과적으로 작동하여, 평가 지표에서도 우수한 성능을 보여주었습니다.



### Dual-Domain CLIP-Assisted Residual Optimization Perception Model for Metal Artifact Reduction (https://arxiv.org/abs/2408.14342)
Comments:
          14 pages, 18 figures

- **What's New**: 이번 논문에서는 금속 인공물이 포함된 컴퓨터 단층촬영(CT) 이미지에서 발생하는 아티팩트를 줄이기 위해 시각-언어 모델(visual-language model, VLM)을 활용한 새로운 방법, 이중 도메인 CLIP 보조 잔여 최적화 인식 모델(DuDoCROP)을 제안합니다. 이 모델은 다양한 형태의 금속임플란트를 효과적으로 처리하여 일반화 능력을 향상시킵니다.

- **Technical Details**: DuDoCROP 모델은 이중 도메인 CLIP(DuDoCLIP)을 기반으로 하며, 이미지 도메인과 시노그램 도메인 모두에서 대조 학습(contrastive learning)을 통해 해부학적 구조와 금속 아티팩트의 의미적 설명을 추출합니다. 생성된 임베딩(embeddings)을 사용하여 디퓨전 모델(diffusion model)을 안내하고, 잔여 최적화를 위한 다운스트림 작업을 통해 두 도메인 사전 이미지의 통합을 수행합니다. 이러한 과정은 최종적인 영상 효과와 데이터 충실도를 개선합니다.

- **Performance Highlights**: 실험 결과, DuDoCROP은 기존 모델 대비 최소 63.7% 더 높은 일반화 능력을 보이며, 더 현실적인 이미지 구조를 생성하고, 정량적 및 정성적 기준 모두에서 기타 최첨단(SOTA) 방법을 초월하는 성능을 확인하였습니다.



New uploads on arXiv(cs.AI)

### Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming (https://arxiv.org/abs/2408.16725)
Comments:
          10 pages

- **What's New**: 본 논문에서는 Mini-Omni를 소개합니다. Mini-Omni는 실시간 음성 상호작용이 가능한 오픈 소스 end-to-end 대화 모델로, 음성 입력과 출력을 모두 처리할 수 있는 최초의 모델입니다. 'Any Model Can Talk'라는 독창적인 교육 방법을 통해 기존 모델에 최소한의 수정만으로 음성 출력을 구현할 수 있습니다.

- **Technical Details**: Mini-Omni는 텍스트 지시 음성 생성 방법과 동시에 배치 병렬 처리 전략을 적용하여 성능을 향상시킵니다. 이 모델은 추가적인 TTS 시스템 없이 실시간 오디오 출력을 가능하게 하며, SNAC 인코더를 활용하여 고품질의 음성 출력을 보장합니다. 또한 VoiceAssistant-400K 데이터셋을 활용하여 음성 모델을 최적화할 수 있습니다.

- **Performance Highlights**: Preliminary 실험에 따르면 Mini-Omni는 전통적인 음성 합성과 관련된 여러 작업에서 강한 성능을 보여주었으며, 기존 모델의 언어 능력을 최소한으로 저하시키면서도 리얼타임 상호작용을 가능하게 합니다. 이 모델을 통해 사용자 경험을 크게 개선할 것으로 기대됩니다.



### Guided Reasoning: A Non-Technical Introduction (https://arxiv.org/abs/2408.16331)
- **What's New**: 이 논문에서는 Guided Reasoning의 개념과 기본 구현을 소개합니다. 여러 에이전트가 있는 시스템에서 한 에이전트(가이드)가 다른 에이전트와 상호작용하여 추론 품질을 향상시키는 경우, 이를 Guided Reasoning 시스템이라고 합니다.

- **Technical Details**: 이 논문은 Logikon의 Guided Reasoning 기본 구현을 비기술적인 용어로 설명합니다. 이는 지속적으로 정보와 예제가 추가될 예정인 살아있는 문서입니다.

- **Performance Highlights**: 구체적인 성능 지표는 논문에는 명시되어 있지 않지만, Guided Reasoning 시스템은 에이전트 간의 상호작용을 통해 향상된 추론 결과가 기대됩니다.



### Logic-Enhanced Language Model Agents for Trustworthy Social Simulations (https://arxiv.org/abs/2408.16081)
Comments:
          Source code: this https URL

- **What's New**: 새로운 연구 프레임워크인 Logic-Enhanced Language Model Agents (LELMA)를 소개합니다. 이 프레임워크는 대형 언어 모델(LLMs)의 신뢰도를 높이기 위한 혁신적인 접근 방식을 제공합니다.

- **Technical Details**: LELMA 프레임워크는 3개의 주요 구성 요소로 구성되어 있습니다: LLM-Reasoner는 전략적 reasoning을 생성하고, LLM-Translator는 자연어 reasoning을 논리 쿼리로 매핑하며, Solver는 이러한 쿼리를 평가합니다.

- **Performance Highlights**: LELMA는 Hawk-Dove 게임, Prisoner’s Dilemma, Stag Hunt와 같은 게임 이론적 시나리오에서 GPT-4 Omni와 Gemini 1.0 Pro의 reasoning 출력 정확도를 개선하는 데 높은 정확도를 보였습니다.



### SAM2Point: Segment Any 3D as Videos in Zero-shot and Promptable Manners (https://arxiv.org/abs/2408.16768)
Comments:
          Work in progress. Online Demo: this https URL . Code: this https URL

- **What's New**: 새로운 연구인 SAM2Point는 Segment Anything Model 2 (SAM 2)를 활용하여 제로샷(zero-shot) 및 프롬프트(promptable) 3D 세분화를 수행하는 초기 탐험을 소개합니다. 이 모델은 3D 데이터를 다방향 비디오로 해석하여 추가 훈련이나 2D-3D 투영 없이 3D 공간에서 세분화를 가능하게 합니다.

- **Technical Details**: SAM2Point는 3D 포인트, 박스, 마스크와 같은 다양한 프롬프트를 지원하며, 복잡한 2D-3D 전환 없이도 3D 물체, 실내 장면, 실외 환경 및 원시 희소 LiDAR(raw sparse LiDAR)와 같은 다양한 시나리오에서 일반화할 수 있는 능력을 갖추고 있습니다. 구체적으로, SAM2Point는 3D 데이터를 비디오 형식으로 변환하기 위해 복셀화(voxelization)를 사용하고, 사용자가 제공하는 3D 프롬프트로부터 세분화 결과를 생성합니다.

- **Performance Highlights**: 여러 3D 데이터셋에서의 데모가 SAM2Point의 강력한 일반화 능력을 강조합니다. 예를 들어, Objaverse, S3DIS, ScanNet, Semantic3D, KITTI와 같은 데이터셋에서의 실험 결과는 모델의 견고한 성능을 보여줍니다. 또한, SAM2Point는 3D 세분화를 위한 SAM의 신뢰할 수 있는 구현을 제공하며, 이는 향후 프롬프트 가능한 3D 세분화 연구의 출발점이 될 수 있습니다.



### ReconX: Reconstruct Any Scene from Sparse Views with Video Diffusion Mod (https://arxiv.org/abs/2408.16767)
Comments:
          Project page: this https URL

- **What's New**: ReconX라는 새로운 3D 장면 재구성 패러다임을 제안하여 불분명한 재구성 문제를 시간 생성(task of temporal generation) 문제로 재구성합니다. 이는 대규모로 사전 훈련된 비디오 확산 모델(video diffusion models)의 강력한 생성적 프라이어를 활용하여 희소한 시각으로부터 재구성 작업에 활용합니다.

- **Technical Details**: ReconX는 제한된 입력 뷰를 사용하여 전역 포인트 클라우드를 구축하고 이를 맥락 공간(contextual space)으로 인코딩하여 3D 구조 조건(3D structure condition)으로 사용합니다. 그러면 비디오 확산 모델이 이 조건을 기반으로 세부사항을 보존하면서도 높은 3D 일관성을 가진 비디오 프레임을 생성합니다. 마지막으로, 생성된 비디오에서 3D 장면을 회수하기 위해 3D 가우시안 스플래팅(Gaussian Splatting) 최적화 방식을 사용합니다.

- **Performance Highlights**: 다양한 실제 세계 데이터셋에서 진행한 실험에서 ReconX는 기존 최첨단 방법들보다 품질(resolution)과 일반화 능력(generalizability)에서 우수함을 입증하였습니다.



### A Score-Based Density Formula, with Applications in Diffusion Generative Models (https://arxiv.org/abs/2408.16765)
- **What's New**: 본 연구는 Score-based generative models (SGMs)의 최적화 이론적 기초를 다루며, diffusion generative 모델인 Denoising Diffusion Probabilistic Models (DDPMs)의 훈련에 있어 evidence lower bound (ELBO)의 최적화가 효과적인 이유를 규명하고자 한다.

- **Technical Details**: 이 논문에서는 확률적 미분 방정식(stochastic differential equations, SDE)에 기반한 연속시간 확산 과정의 밀도 공식을 도출하였으며, 이는 SGM에서의 전방 과정의 연속 시간 한계를 나타낸다. 이로 인해 각 단계의 전방 과정과 관련된 score 함수와 목표 밀도 간의 관계가 명확히 드러났다.

- **Performance Highlights**: 연구 결과, DDPM 훈련을 위한 최적화 목표의 최소화기가 실제 목표인 KL 발산을 거의 최소화함을 보여주며, ELBO를 사용한 DDPM 최적화의 이론적 기초를 제공한다. 또한, GAN 훈련 시 score-matching 정규화의 역할, diffusion classifiers에서의 ELBO 활용, 최근 제안된 diffusion loss에 대한 새로운 통찰을 제시한다.



### Dissecting Out-of-Distribution Detection and Open-Set Recognition: A Critical Analysis of Methods and Benchmarks (https://arxiv.org/abs/2408.16757)
Comments:
          Accepted to IJCV, preprint version

- **What's New**: 이번 논문은 머신러닝 모델의 테스팅 시 발생할 수 있는 분포 변화(Distribution Shift)를 탐지하는 기술적 중요성을 강조하며, OOD(Out-of-Distribution) 탐지와 OSR(Open-Set Recognition)이라는 두 개의 주요 하위 분야의 특성을 종합적으로 분석합니다. 특정 벤치마크를 제안하고, 다양한 방법론의 성능을 비교하여 실질적인 통찰력을 제공합니다.

- **Technical Details**: 이研究에서는 OOD 탐지와 OSR 방법의 철저한 교차 평가를 수행하며, 이 두 분야의 방법들 간의 성능 상관관계를 식별합니다. 또한, OOD 탐지와 OSR에서 발생하는 분포 변화를 효과적으로 분리하는 새로운 대규모 벤치마크 설정을 제안합니다. 본 연구는 Outlier Exposure(OE)과 같은 최신 방법들이 실제 대규모 환경에서 어떻게 성능이 저하되는지를 설명하고, 그 대안으로 깊은 특징의 크기에 민감한 스코어링 규칙을 주요 성과로 제시합니다.

- **Performance Highlights**: OE는 표준 작은 벤치마크에서는 좋은 성능을 보였으나, 대규모 환경에서는 성능이 떨어지는 것을 발견했습니다. 반면, MLS(Maximum Logit Score)와 같은 깊은 특징의 크기에 민감한 스코어링 규칙은 여전히 유망한 성과를 보였습니다. 이러한 발견은 OOD 탐지 및 OSR 방법 간의 관계를 더욱 명확히 하고, 향후 연구 방향을 제시하는 계기가 됩니다.



### Assessing Large Language Models for Online Extremism Research: Identification, Explanation, and New Knowledg (https://arxiv.org/abs/2408.16749)
- **What's New**: 미국에서 폭력적인 극단주의가 급증함에 따라 온라인에서 극단주의 이념의 확산을 감지하고 제한할 수 있는 자동화된 도구의 필요성이 커지고 있습니다. 이 연구는 Bidirectional Encoder Representations from Transformers (BERT)와 Generative Pre-Trained Transformers (GPT)의 성능을 평가하여 온라인 국내 극단주의 게시물을 탐지하고 분류하는데 중점을 두었습니다.

- **Technical Details**: 연구에서는 '극우'(far-right) 및 '극좌'(far-left) 이념 키워드가 포함된 소셜 미디어 게시물을 수집하고 수동으로 극단주의 또는 비극단주의로 레이블을 붙였습니다. 극단주의 게시물은 정의적 틀에 기반한 다섯 가지 극단주의 요소로 추가 분류되었습니다. BERT 모델의 성능은 훈련 데이터 크기와 카테고리 간 지식 전이를 기준으로 평가했습니다. 또한, 다양한 프롬프트(naïve, layperson-definition, role-playing, professional-definition)를 사용하여 GPT 3.5와 GPT 4 모델의 성능을 비교했습니다.

- **Performance Highlights**: 최고 성과를 기록한 GPT 모델은 최고의 BERT 모델보다 나은 성능을 보였으며, 더 구체적인 프롬프트가 일반적으로 더 나은 결과를 가져왔습니다. 그러나 지나치게 복잡한 프롬프트는 성능을 저하시킬 수 있습니다. GPT 3.5는 극좌 극단주의 게시물 분류에서 더 좋은 성과를 거두었고, GPT 4는 극우 극단주의 게시물 분류에서 더 좋았습니다. GPT 모델은 전통적인 BERT 모델에 비해 온라인 극단주의 분류 작업에서 중요한 잠재력을 보이며, 제로샷(zero-shot) 설정에서 성능을 능가했습니다.



### Smaller, Weaker, Yet Better: Training LLM Reasoners via Compute-Optimal Sampling (https://arxiv.org/abs/2408.16737)
- **What's New**: 본 연구는 합성 데이터 생성에 있어 강력한 언어 모델(Strong Language Models, LMs) 대신 약하지만 저렴한 모델(Weak Model, WC)을 사용하는 것이 연산 비용 최적화(compute-optimal) 가능성이 있음을 제안합니다.

- **Technical Details**: 연구진은 WC 모델에서 생성된 데이터의 전체 문제 커버리지(coverage), 다양한 솔루션 수(diversity), 그리고 잘못된 추론을 가진 문제 비율(false positive rate, FPR)을 평가합니다. 결과적으로, WC 모델의 데이터는 높은 커버리지와 다양성을 보이지만 FPR 또한 상승합니다.

- **Performance Highlights**: 여러 실험에서 WC 데이터로 미세 조정된 모델이 SE 데이터로 훈련된 모델보다 평균 31.6%의 성능 향상을 보였으며, 이는 기존의 SE 모델 기반 접근 방식을 재고해야 함을 시사합니다.



### A GREAT Architecture for Edge-Based Graph Problems Like TSP (https://arxiv.org/abs/2408.16717)
Comments:
          15 pages, 7 figures

- **What's New**: 본 논문에서는 Graph Edge Attention Network (GREAT)라는 새로운 edge 기반 모델을 제안합니다. 기존의 GNN 방식을 개선하여, 밀집 그래프와 비유클리드(非Euclidean) 스캐줄링 문제를 효과적으로 처리할 수 있는 방법을 모색합니다.

- **Technical Details**: GREAT는 기존 GNN이 노드 기초의 메시지 전송에 의존하는 대신 엣지 기반으로 작동하여, 엣지를 통한 정보 전달을 최적화합니다. 이 모델은 엣지 분류 작업을 통해 TSP(Touring Salesman Problem)의 최적 엣지를 예측하는 데 초점을 맞춥니다. 강화 학습(reinforcement learning) 기반의 GREAT 프레임워크도 구축되어 있으며, 이는 유클리드 및 비유클리드 비대칭 TSP 문제에 적용될 수 있습니다.

- **Performance Highlights**: GREAT는 기존의 비학습 기반 방법들에 비해 더 희소한 그래프를 생성하면서도 최적 엣지를 대부분 보존할 수 있습니다. 특히, 두 개의 비대칭 TSP 분포에서 최신 기술 수준(state-of-the-art) 접근 방식을 달성하여 성능을 입증했습니다.



### Entropic Distribution Matching in Supervised Fine-tuning of LLMs: Less Overfitting and Better Diversity (https://arxiv.org/abs/2408.16673)
- **What's New**: 본 연구는 Large Language Models (LLMs)의 Supervised Fine-Tuning (SFT) 과정에서 발생하는 overfitting과 낮은 output diversity 문제를 해결하고자 합니다. 이를 위해 저자들은 최대 엔트로피 원칙을 활용하여 데이터 분포를 잘 포착하면서도 더 평평한 분포를 선호하는 새로운 distribution matching 방법인 GEM(Generative Entropy-regularized Matching)을 제안했습니다.

- **Technical Details**: GEM은 reverse Kullback-Leibler divergence의 최소화를 해결하며, entropy regularizer를 통해 LLM의 fine-tuning을 처리합니다. 이 방법은 Cross Entropy (CE) loss와 비교할 때 과적합(overfitting) 문제가 줄어들고 더 다양한 출력을 생성하는데 살펴보고, 특히 UltraFeedback 데이터셋을 통한 일반적인 instruction-following 능력 개발에 적용됩니다.

- **Performance Highlights**: GEM은 수학적 추론(math reasoning) 및 코드 생성(code generation) 과제에서 최대 7점의 성능 향상을 보여주며, domain-specific 데이터셋으로 fine-tuning 할 경우 CE에 비해 최대 10점의 개선을 보였습니다. 이를 통해 GEM은 CE loss의 한계를 극복하고 더 다양한 출력을 생성하는데 유리함을 입증했습니다.



### Jina-ColBERT-v2: A General-Purpose Multilingual Late Interaction Retriever (https://arxiv.org/abs/2408.16672)
- **What's New**: 이번 연구에서는 정보 검색에 있어 효과적인 다중 벡터 밀집 모델인 ColBERT의 개선점을 제시합니다. Jina-ColBERT-v2 모델을 도입하여 훈련 파이프라인을 개선하고 다양한 다국어 데이터를 효과적으로 처리합니다.

- **Technical Details**: Jina-ColBERT-v2는 비지도 학습(weakly supervised) 데이터를 활용한 두 단계 튜닝 방식(contrastive tuning followed by fine-tuning)으로 훈련됩니다. 또한, 효율적인 추론을 위해 다양한 크기의 선형 프로젝션 헤드를 도입하였으며, Jina-XLM-RoBERTa와 같은 최적화된 백본을 사용하여 성능을 극대화했습니다.

- **Performance Highlights**: Jina-ColBERT-v2는 영어 및 다국어 검색 과제에서 강력한 성능을 보이며, 기존 모델과 비교해 최대 50%의 저장 공간을 절약합니다.



### Iterative Graph Alignmen (https://arxiv.org/abs/2408.16667)
Comments:
          12 pages, 4 figures

- **What's New**: 이 논문은 Iterative Graph Alignment (IGA)라는 새로운 비주얼 베이스의 정렬 알고리즘을 소개합니다. IGA는 교사 모델(VLM)이 Iterative Graph Prompting (IGP)을 사용하여 개념적 그래프와 참조 답변을 생성하고, 이를 통해 학생 모델(LLM)이 자신의 답변과 비교하여 지식 간의 격차를 파악하도록 합니다.

- **Technical Details**: IGA는 두 가지 주요 과정인 독창적인 사고 과정과 적응 학습 메커니즘을 통합하여 구성됩니다. 교사 모델은 논리 그래프를 시각적으로 구성하여 학생 모델이 자신의 답변과 비교하고 지식 격차를 식별할 수 있도록 합니다. 또, Self-Aligned Incremental Learning (SAIL) 접근 방식을 통해 각 단계별로 점차적으로 지원을 제공합니다.

- **Performance Highlights**: IGA 방법을 사용한 결과, Claude Sonnet 3.5에서 73.12%의 정렬 개선 효과를 보였으며, Llama3-8B-Instruct는 86.20%의 개선율을 기록하여 Claude Sonnet 3.5를 초과하는 성능을 입증했습니다.



### DriveGenVLM: Real-world Video Generation for Vision Language Model based Autonomous Driving (https://arxiv.org/abs/2408.16647)
- **What's New**: 이번 논문에서는 자율 주행에서 비디오 생성과 이해를 결합한 새로운 프레임워크인 DriveGenVLM을 제안합니다. 이는 Denoising Diffusion Probabilistic Models(DDPM)을 기반으로 한 비디오 생성 프레임워크를 사용하여 현실 세계의 비디오 시퀀스를 예측하고, Vision Language Models(VLMs)를 활용하여 생성된 비디오를 이해할 수 있도록 합니다.

- **Technical Details**: DriveGenVLM 프레임워크는 DDPM을 사용하여 자율 주행에 적합한 비디오를 생성하며, Waymo 데이터셋을 통해 훈련되고 Fréchet Video Distance(FVD) 점수를 이용해 평가됩니다. 출시된 EILEV 모델을 통해 생성된 비디오에 대한 내레이션을 제공함으로써, 자율 주행에 필요한 상황 이해를 돕습니다.

- **Performance Highlights**: 이 연구는 DDPM을 활용하여 자율 주행 비디오 예측의 조건부 생성 모델을 적용하고, Waymo 데이터셋을 바탕으로 실제 자율 주행 시나리오에 대한 유효성을 검증한 첫 번째 사례입니다. 또한, VLM을 사용하여 생성된 비디오에 대한 설명을 생성함으로써 자율 주행 기반 알고리즘의 결정 과정에 기여할 수 있습니다.



### RLCP: A Reinforcement Learning-based Copyright Protection Method for Text-to-Image Diffusion Mod (https://arxiv.org/abs/2408.16634)
Comments:
          arXiv admin note: text overlap with arXiv:2403.12052 by other authors

- **What's New**: 이번 연구에서는 Reinforcement Learning 기반의 저작권 보호 방법(RLCP)을 제안하여 텍스트-이미지 확산 모델에서 저작권을 침해하는 콘텐츠 생성을 최소화하고 이미지 품질을 유지합니다.

- **Technical Details**: 저작권 법과 판례에 기반한 새로운 저작권 메트릭을 도입하여 저작권 침해를 평가하며, Denoising Diffusion Policy Optimization (DDPO) 프레임워크를 활용해 다단계 의사결정 과정을 최적화합니다. 또한, KL divergence를 정규화 항으로 사용하여 RL의 파인튜닝을 안정화합니다.

- **Performance Highlights**: 3개의 혼합 데이터 세트를 대상으로 한 실험 결과, RLCP 방법이 저작권 침해 위험을 상당히 줄이고 이미지 품질을 유지하는 데 있어 4개의 기준 모델 대비 우수한 성능을 보여주었습니다.



### Optimizing Automated Picking Systems in Warehouse Robots Using Machine Learning (https://arxiv.org/abs/2408.16633)
- **What's New**: 이 연구는 전자상거래(e-commerce)의 급속한 성장과 함께 물류(logistics) 산업에서 자동화의 수요가 증가하고 있음을 강조하며, 깊은 학습(deep learning)과 강화 학습(reinforcement learning) 기술을 활용한 자동 피킹 시스템(automated picking systems)에 초점을 맞춥니다.

- **Technical Details**: 이 연구에서는 로봇 피킹 성능(robot picking performance)과 복잡한 환경에서의 적응력(adaptability)을 향상시키기 위해 기계 학습(machine learning) 모델을 통합하였습니다. 이 모델은 환경 요인(environmental factors)을 분석하여 시스템 설계(system design)를 최적화하고, 가변 조건(variable conditions)에서도 효율적이고 안정적인 작동을 보장합니다.

- **Performance Highlights**: 연구 결과, 통합 기계 학습 모델은 전통적인 방법에 비해 성능이 크게 향상되어 피크 주문 처리(peak order processing) 문제를 효과적으로 해결하고 운영 오류(operational errors)를 줄이며 전체 물류 효율성을 개선하였습니다.



### Maelstrom Networks (https://arxiv.org/abs/2408.16632)
- **What's New**: 이번 논문에서는 작업 기억(working memory)을 인공 신경망에 통합하기 위한 새로운 접근법인 	extit{Maelstrom Networks} 패러다임을 제안합니다. 이 패러다임은 순환 신경망의 강점을 활용하면서도 피드를 통해 학습을 수행할 수 있도록 합니다.

- **Technical Details**: 	extit{Maelstrom Networks}는 학습되지 않은 순환 구성 요소를 가지고 있으며, 강력한 피드 포워드 네트워크에 학습을 위탁합니다. 이는 피드 포워드 학습의 장점을 활용하되 네트워크를 풀지 않고도 작업할 수 있게 하며, 새로운 신경 모양의 하드웨어를 구현할 수 있게 합니다.

- **Performance Highlights**: 이 새로운 구조는 현재 비시간적 깊은 네트워크의 성능 문제 해결에 크게 기여할 수 있으며, 지속적인 학습(continual learning)으로 이어질 가능성이 있습니다. 또한 인공지능에 "자아(self)" 감각을 부여할 수 있는 길을 열어줍니다.



### LLMs generate structurally realistic social networks but overestimate political homophily (https://arxiv.org/abs/2408.16629)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)을 사용하여 현실적인 사회 네트워크를 생성하는 새로운 방법을 탐구합니다. 특히, 생성된 네트워크의 현실성과 편향의 문제를 조명하며, 세 가지 프롬프트 방법을 제안합니다.

- **Technical Details**: 연구는 LLM이 개인의 페르소나를 각각 하나씩 다루는 '로컬' 방법이 전체 네트워크를 한 번에 구성하는 '글로벌' 방법보다 더 현실적인 결과를 생성함을 발견했습니다. 또한, LLM이 생성한 네트워크는 밀도(density), 클러스터링(clustering), 연결성(connectivity) 및 차수 분포(degree distribution) 등 많은 구조적 특성에서 현실 네트워크와 일치합니다.

- **Performance Highlights**: 제작된 네트워크는 성별, 연령, 인종/민족, 종교 및 정치적 소속 등의 인구통계학적 동질성(demographic homophily)을 명확히 보여주지만, 정치적 동질성을 과대평가하는 경향이 있음을 발견하였습니다. LLM이 생성한 관심사를 포함시키더라도 정치적 동질성을 줄이지 않습니다.



### Towards Infusing Auxiliary Knowledge for Distracted Driver Detection (https://arxiv.org/abs/2408.16621)
Comments:
          Accepted at KiL 2024: Workshop on Knowledge-infused Learning co-located with 30th ACM KDD Conference

- **What's New**: 이 논문에서는 운전 중 주의 산만(Distracted Driving)을 효과적으로 탐지하기 위한 새로운 방법, KiD3를 제안합니다. 해당 방법은 시각적 정보와 함께 장면 내 요소 간의 의미적 관계와 운전자의 자세 정보를 통합하여 운전자의 행동을 포괄적으로 이해할 수 있는 통합 프레임워크를 구성합니다.

- **Technical Details**: KiD3는 기존 비전 전용 모델에 비해 13.64%의 정확도 향상을 이룩하였습니다. 이 방법은 시각적 정보에 보조 지식(auxiliary knowledge)을 결합하여 다양한 운전자의 행동을 신뢰성 있게 탐지합니다. 주요 구성 요소로는 장면 그래프(scene graphs)와 운전자의 자세(pose) 정보가 결합되어 있습니다.

- **Performance Highlights**: KiD3는 실제 데이터셋을 기반으로 실험을 진행하였고, 비전 전용 기준 모델에 비해 정확도가 크게 향상되었습니다. 이는 안전한 운전 환경을 구축하는 데 기여할 수 있는 가능성을 제시합니다.



### Hyperdimensional Vector Tsetlin Machines with Applications to Sequence Learning and Generation (https://arxiv.org/abs/2408.16620)
- **What's New**: 본 논문에서는 Hyperdimensional Vector Computing (HVC)과 Tsetlin 머신을 결합하여 시퀀스 데이터를 학습하고 생성하는 새로운 이층 모델을 제안합니다. 이 모델은 기존 Tsetlin 머신과 경쟁할 수 있는 속도를 가지면서 다양한 장점을 부각시킵니다.

- **Technical Details**: HVC는 높은 차원의 벡터 공간에서 데이터의 특징을 추출하는 데 강력한 기능을 제공합니다. Tsetlin 머신은 이진화된 데이터를 기반으로 직접 학습하며, HVC의 Binary Spatter Codes (BSC)와 결합하여 효율적인 입력 계층을 구성합니다. HVC의 주요 연산인 bundling, binding, unbinding, 유사도 측정 및 Perturbation을 통해 강력한 계산 프레임워크를 구성합니다.

- **Performance Highlights**: 제안된 HVC-Tsetlin 아키텍처는 기존의 SOTA(State-of-the-Art) 결과와 경쟁할 수 있는 능력을 보이며, 메모리 사용량이 적어 소형 임베디드 시스템에서 온라인 학습에 적합합니다. UCR Time Series Archive에서 다양한 시간 시퀀스를 분류할 수 있는 실험 결과도 제시됩니다.



### Examination of Code generated by Large Language Models (https://arxiv.org/abs/2408.16601)
- **What's New**: 본 연구에서는 ChatGPT와 GitHub Copilot과 같은 대형 언어 모델(LLMs)에 의해 생성된 코드의 정확성과 품질을 평가하기 위한 통제된 실험을 수행했습니다. 실험 결과는 다양한 알고리즘, 언어 및 LLMs의 반복적이고 비교 가능한 평가를 위한 실험 방법도 함께 보고합니다.

- **Technical Details**: 실험에서는 기본적인 알고리즘을 선택하여 Java와 Python에서 코드와 유닛 테스트 케이스를 생성했습니다. 이용된 알고리즘으로는 Bellman-Ford, Binary Search, Dijkstra 등 12가지 알고리즘이 있습니다. 생성된 코드는 정확성과 코드 품질을 평가하는 다양한 메트릭스를 사용하여 분석되었습니다. 또한 ChatGPT와 GitHub Copilot의 코드 생성 간의 차이를 비교하였습니다.

- **Performance Highlights**: 수행된 실험은 언어 및 알고리즘에 따라 LLM의 성능이 상이함을 보여주었으며, 시간이 지남에 따라 코드 생성 능력이 향상되고 있음이 관찰되었습니다. 이 연구는 LLM 기반 도구의 현재 능력을 파악하고, 미래의 코드 생성 가능성을 예측하는 데 기여할 수 있는 평가 프레임워크를 제공하고 있습니다.



### Enhancing Dialogue Generation in Werewolf Game Through Situation Analysis and Persuasion Strategies (https://arxiv.org/abs/2408.16586)
Comments:
          Accepted to the AIWolfDial2024 workshop at INLG 2024

- **What's New**: 이 논문은 LLM(대규모 언어 모델)을 기반으로 한 새로운 Werewolf Game AI를 소개하며, 이는 AIWolfDial2024 대회의 요구 사항을 충족하기 위한 강력한 기준을 제공합니다.

- **Technical Details**: AI는 상황 분석 모듈, 응답 생성 모듈, 그리고 특별히 설계된 설득 응답 생성 모듈을 포함하여, 역할에 따라 게임 내 대화를 수행합니다. 또한, 이 시스템은 논리적 설득, 신뢰 기반 설득, 그리고 감정 기반 설득 전략을 통해 좀비 역할의 설득력을 강화합니다.

- **Performance Highlights**: 새로운 모델은 멀티 턴 대화 프레임워크에서 복잡한 상호작용을 처리하여 더욱 자연스럽고 의미 있는 대화 생성을 통해 AI의 효과iveness를 향상시킵니다.



### Seeking the Sufficiency and Necessity Causal Features in Multimodal Representation Learning (https://arxiv.org/abs/2408.16577)
- **What's New**: 본 논문에서는 필요한 원인과 충분한 원인 (PNS)의 확률을 기반으로 다중 모달 (multimodal) 데이터에서의 표현 학습에 관한 새로운 접근법을 제안합니다. 기존의 연구가 단일 모달 (unimodal) 데이터에 집중된 점과는 달리, 다중 모달 환경에서 PNS 추정의 확장을 시도하여 관련된 제약 조건을 분석합니다.

- **Technical Details**: PNS는 결과에 대한 특징 집합이 필요하고 충분한지를 측정하는 지표로, 본 연구에서는 다중 모달 표현을 모달리티 불변 (modality-invariant)과 모달리티 전용 (modality-specific) 두 구성 요소로 나누어 각각의 PNS 식별 가능성 (identifiability)을 분석합니다. 이를 통해 비어 있는 PNS 추정 (non-trivial PNS estimation)을 위한 추가적인 제약 조건을 도입하고, PNS가 높은 다중 모달 표현을 학습하기 위한 최적화 목표 (optimization objectives)를 제안합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법이 합성 (synthetic) 데이터와 실제 (real-world) 데이터 모두에서 효과적으로 작동함을 보여주었으며, 다중 모달 모델의 예측 성능을 향상시키는 데 기여했습니다.



### SFR-GNN: Simple and Fast Robust GNNs against Structural Attacks (https://arxiv.org/abs/2408.16537)
- **What's New**: 이번 논문에서는 구조적 공격에 대한 강인성을 가진 Simple and Fast Robust Graph Neural Network (SFR-GNN)이라는 효율적인 방어 방법을 제안합니다. 이 방법은 노드 속성을 사용하여 GNN 모델을 사전 훈련한 후, 변경된 그래프를 대조 학습 방식으로 미세 조정하여, 구조를 정화하거나 적응형 집계를 사용하는 것을 피함으로써 효율성을 높입니다.

- **Technical Details**: SFR-GNN은 노드 속성 𝐗를 사용하여 초기 사전 훈련을 수행한 뒤, 구조 정보가 포함된 변경된 그래프 𝐀'에 대해 미세 조정을 진행합니다. 이 과정에서 SFR-GNN은 변조된 구조와 노드 임베딩 간의 '짝짓기 효과'를 방해하여 강인성을 확보합니다. 특히, 본 방법은 추가적인 하이퍼 파라미터 없이 경량화된 구조로 이루어져 있어, 기존 강건 GNN보다 계산 복잡도와 하이퍼 파라미터 복잡도를 크게 낮출 수 있습니다.

- **Performance Highlights**: 실험 결과, SFR-GNN은 노드 분류 과제에서 기존 강건 모델들에 비해 24%에서 162%까지의 속도 향상을 기록하며, 유사하거나 더 뛰어난 강인성을 보였습니다. 이는 실용적인 환경에서의 적용 가능성을 높이는 데 기여합니다.



### Adaptive Variational Continual Learning via Task-Heuristic Modelling (https://arxiv.org/abs/2408.16517)
Comments:
          4 pages, 2 figures, 3 tables

- **What's New**: 이 연구에서는 AutoVCL이라는 새로운 모델을 소개하며, 이는 과거의 작업과 비교하여 새로운 작업의 난이도와 유사성에 따라 하이퍼파라미터를 자동으로 조정하여 기존의 GVCL 모델을 능가할 수 있음을 보여줍니다.

- **Technical Details**: AutoVCL은 generalized variational continual learning (GVCL) 모델의 확장으로, informed learning을 위한 task heuristics와 모델 최적화를 결합합니다. Bayesian 추론을 활용하여 posterior 분포를 업데이트하고, KL divergence를 통해 학습 효율성을 높입니다. 또한, task-specific ‘head networks’를 추가하여 더 나은 학습을 위한 multi-head discriminative framework 구조를 사용합니다.

- **Performance Highlights**: 실험 결과, AutoVCL은 다양한 작업 순서에서 기존 GVCL 모델에 비해 우수한 성능을 보여주었으며, 특히 Split MNIST 예제에서 여러 작업을 학습한 후 정확도를 저하 없이 유지함이 관찰되었습니다.



### On-device AI: Quantization-aware Training of Transformers in Time-Series (https://arxiv.org/abs/2408.16495)
Comments:
          This paper is accepted by 2023 IEEE International Conference on Pervasive Computing and Communications(PhD Forum)

- **What's New**: 본 연구는 한정된 리소스를 가진 센서 장치에서 Transformer 모델의 최적화를 목표로 하며, 하드웨어 가속기로서 FPGA에 배포됩니다. Quantization-aware Training (QAT) 기법을 적용하여 모델 크기와 실행 메모리 소비를 줄이면서 FPGA의 장점을 극대화합니다.

- **Technical Details**: Transformer 모델은 시계열 예측 작업을 위한 최적화가 필요하며, Quantization은 모델의 압축과 계산 집약도를 줄이는 일반적인 방법입니다. 본 연구는 2비트 완전 양자화된 Transformer 모델을 구현하며, 전방위 컴퓨팅을 위한 FPGA에 가속기로 통합할 예정입니다.

- **Performance Highlights**: 제안된 방법을 통해 기존 Transformer 모델 대비 7.46545×의 압축률을 달성하였으며, 이는 FPGA에서 보다 복잡한 계산을 수행하고 빠른 모델 추론을 가능하게 합니다.



### Integrating Features for Recognizing Human Activities through Optimized Parameters in Graph Convolutional Networks and Transformer Architectures (https://arxiv.org/abs/2408.16442)
Comments:
          6 pages, 1 figure, conference

- **What's New**: 인간 활동 인식에 대한 연구는 딥러닝 모델의 특징 융합을 통해 정확도를 향상시키는 데 중점을 두고 있습니다. 특히, Transformer 모델과 파라미터 최적화된 그래프 합성곱 네트워크(PO-GCN)를 사용하여 다양한 공공 데이터셋을 활용한 성능 향상을 보여줍니다.

- **Technical Details**: 이 연구는 HuGaDB, PKU-MMD, LARa, TUG의 4개 공공 데이터셋에서 수집한 데이터를 사용하여 PO-GCN 및 Transformer 모델을 훈련하고 평가하였습니다. 이 두 모델에서 최종 계층의 특징을 융합하고, 이를 분류기에 입력해 활동 인식을 수행합니다. 특징 융합(feature fusion) 기술은 공간적 및 시간적 특징을 보다 잘 이해할 수 있게 해줍니다.

- **Performance Highlights**: HuGaDB 데이터셋에서 2.3%의 정확도 향상과 2.2%의 F1-점수 증가를 나타냈으며, TUG에서는 5%의 정확도 증가와 0.5%의 F1-점수 향상을 기록했습니다. LARa 및 PKU-MMD는 각각 64%, 69%로 낮은 정확도를 보였지만, PO-GCN 모델과 Transformer 모델의 통합이 성능을 개선했다는 강력한 증거를 제공합니다.



### Gradient-free variational learning with conditional mixture networks (https://arxiv.org/abs/2408.16429)
Comments:
          16 pages main text (3 figures), including references. 9 pages supplementary material (5 figures)

- **What's New**: 본 논문은 조건부 혼합 네트워크(Conditional Mixture Networks, CMNs)를 활용하여 빠르고 경량화된 베이지안 추론을 가능하게 하는 새로운 학습 알고리즘인 CAVI-CMN을 소개합니다. 이 방법은 전통적인 경량화된 방법인 최대우도추정(Maximum Likelihood Estimation, MLE)과 비교했을 때 예측 정확도를 유지하면서도 그라디언트를 필요로 하지 않는 이점을 갖습니다.

- **Technical Details**: CAVI-CMN은 조건부 혼합 모델에 대한 변분 추론 방식을 도입하며, 선형 전문가(Experts)와 소프트맥스 게이팅 네트워크(Gating Network)에 대해 가우시안 가능성을 제공합니다. 이 방법은 좌표 상승 변분 추론(Coordinate Ascent Variational Inference, CAVI)을 기반으로 하며, 전통적인 그라디언트 기반 최적화 방식을 회피합니다. 또한, Pólya-Gamma 증강(Pólya-Gamma augmentation)을 활용하여 네트워크 파라미터에 대한 정확한 분포를 유지합니다.

- **Performance Highlights**: CAVI-CMN은 최대 우도 추정(MLE) 및 기타 베이지안 추정 기법들과 비교하여 경쟁력 있는 예측 정확도를 거두며, 특히 8개의 서로 다른 감독 학습 벤치마크에서 성능을 평가했습니다. CAVI-CMN은 또 다른 최첨단 베이지안 방법들, 즉 NUTS(No U-Turn Sampler) 및 BBVI(Black-Box Variational Inference)에 비해 수렴 시간이 크게 감소하고 효율적인 연산 성능을 보여 주었습니다.



### COIN: Control-Inpainting Diffusion Prior for Human and Camera Motion Estimation (https://arxiv.org/abs/2408.16426)
Comments:
          ECCV 2024

- **What's New**: 본 논문에서는 COIN(Control-Inpainting Motion Diffusion Prior)이라는 새로운 방법을 제안합니다. COIN은 인간과 카메라 모션을 분리하는 섬세한 제어를 가능하게 하여, 기존의 과도하게 매끄럽고 잘 맞지 않는 2D 투영 문제를 해결합니다.

- **Technical Details**: COIN의 주요 구성 요소로는 동적 제어 샘플링(dynamic controlled sampling) 기법과 새로운 인간-장면 관계 손실(human-scene relation loss)을 포함합니다. 이 손실은 인간, 카메라 및 장면 간의 일관성을 강제하여 스케일 불일치를 완화합니다. 또한, 저해상도에서 고해상도의 전환을 위한 Soft Inpainting 전략이 채택되었습니다.

- **Performance Highlights**: COIN은 HCM 및 RICH 데이터셋에서 각각 44% 및 33%의 성능 향상을 보이며, 기존의 최첨단 방법들과 비교할 때 인간 및 카메라 모션 추정 정확도에서 눈에 띄는 개선을 나타냅니다.



### Fourier Spectral Physics Informed Neural Network: An Efficient and Low-Memory PINN (https://arxiv.org/abs/2408.16414)
- **What's New**: 본 논문에서는 자동 미분(Automatic Differentiation) 없이 주파수 기반으로 합리화된 신경망(Spectral-Informed Neural Networks, SINNs)을 제안하여, 기존 물리 기반 신경망(Physics-Informed Neural Networks, PINNs)의 한계를 극복합니다. 주파수 영역에서 더 저렴한 메모리와 훈련 시간으로 고차 미분을 처리할 수 있게 되며, 이를 통해 더 나은 정확도를 달성합니다.

- **Technical Details**: 제안된 SINNs는 훈련 과정에서 그리드 포인트(grid points)를 입력으로 사용하지 않고 푸리에 기초(Fourier basis)에서의 주파수를 사용합니다. 그러므로 고차 미분을 직접 계산하는 대신, 적분자를 곱셈으로 대체함으로써 효율성을 높였습니다. 또한 SINNs는 주파수 정보를 활용하여 네트워크를 학습하는 두 가지 전략을 제공하며, 이를 통해 주파수 영역에서의 적응적인 샘플링을 가능하게 합니다.

- **Performance Highlights**: 본 논문에서 수행된 실험 결과에 따르면, 제안된 SINNs는 기존의 PINNs에 비해 훈련 시간을 줄임과 동시에 정확도를 높일 수 있음을 입증하였습니다. 제안된 방식은 특히 고차 미분 계산에 필요한 메모리 소비를 줄임으로써 학습 효율을 향상시킵니다.



### DetectBERT: Towards Full App-Level Representation Learning to Detect Android Malwar (https://arxiv.org/abs/2408.16353)
Comments:
          Accepted at ESEM 2024

- **What's New**: DetectBERT는 DexBERT를 기반으로 하여 Android 악성코드 탐지를 위한 새로운 방법론을 제안합니다. 이는 c-MIL (correlated Multiple Instance Learning)을 통합하여 앱 수준의 탐지를 가능하게 하며, 여러 Smali 클래스를 동시에 효과적으로 처리할 수 있는 기능을 강조합니다.

- **Technical Details**: DetectBERT는 Smali 코드 분석을 통해 얻은 class-level feature를 MIL bags로 처리하여 앱 수준의 표현을 집계합니다. 이 과정에서 Nyström Attention 레이어를 활용하여 클래스 간의 복잡한 상관관계를 학습하고, 효율적으로 정보를 교환합니다. DexBERT의 가중치를 동결하여 원래의 능력을 최대한 활용하는 것이 특징입니다.

- **Performance Highlights**: 실험 결과, DetectBERT는 기존의 세 가지 기본 feature 집계 방법과 두 가지 최첨단 Android 악성코드 탐지 기술을 모두 초월하는 성능을 보였습니다. 또한 새로 등장하는 악성코드 샘플에 대한 강력한 성능 및 지속 가능한 효과성을 검증하여 시간이 지나도 유효성을 유지할 수 있음을 보여주었습니다.



### Toward Robust Early Detection of Alzheimer's Disease via an Integrated Multimodal Learning Approach (https://arxiv.org/abs/2408.16343)
Comments:
          5 pages, 2 figures

- **What's New**: 이 연구는 알츠하이머병(Alzheimer's Disease, AD)의 진단 정확성을 높이기 위해 임상, 인지, 신경영상, 뇌파(EEG) 데이터를 통합한 고급 다중 모달 분류 모델을 도입합니다.

- **Technical Details**: 제안된 모델은 표 데이터의 코딩 아키텍처를 갖춘 기능 태거(feature tagger)를 통합하고, TimesBlock 모듈을 통해 EEG 데이터의 복잡한 시간 패턴을 캡처합니다. Cross-modal Attention Aggregation 모듈을 활용하여 MRI의 공간 정보와 EEG의 시간 정보를 효과적으로 융합하였습니다.

- **Performance Highlights**: 이 모델은 알츠하이머병, 경도 인지장애(Mild Cognitive Impairment, MCI), 정상 인지를 구별하는 데 있어 상당한 성능 향상을 보여주었으며, 새로운 AD 분류 데이터셋을 구축하여 임상적 진단의 초석을 다듬었습니다.



### Self-Improving Diffusion Models with Synthetic Data (https://arxiv.org/abs/2408.16333)
- **What's New**: 이 논문에서는 합성 데이터(synthetic data)를 새로운 방식으로 활용하여 확산 모델(diffusion models)을 훈련시키는 자기 개선(Self-IMproving) 개념을 제안합니다. 기존의 모델 훈련 방식에서 합성 데이터는 품질 저하를 초래하는 것으로 보아 회피해야 한다는 논리가 지배적이었습니다.

- **Technical Details**: 제안된 방법인 SIMS는 합성 데이터를 사용하여 모델의 생성 과정에서 부정적 지침(negative guidance)을 제공합니다. 이를 통해 합성 데이터의 비이상적인 만니폴드(manifold)로부터 벗어나 실제 데이터 분포(real data distribution)를 향해 나아갈 수 있습니다.

- **Performance Highlights**: SIMS는 CIFAR-10과 ImageNet-64 생성에 대해 Fréchet 격차 거리(FID) 메트릭에서 새로운 기록을 수립하며, FFHQ-64와 ImageNet-512에서 경쟁력 있는 결과를 달성합니다. 또한, 자기 생성된 합성 데이터로 반복적으로 훈련될 수 있는 최초의 예방적 생성 AI 알고리즘으로, 편향을 완화하고 공정성을 보장하는 데 도움이 됩니다.



### FA-YOLO: Research On Efficient Feature Selection YOLO Improved Algorithm Based On FMDS and AGMF Modules (https://arxiv.org/abs/2408.16313)
Comments:
          11 pages and 4 figures

- **What's New**: 최근 YOLO 시리즈 모델이 객체 탐지 분야에서 유명세를 타고 있으며, 본 논문은 FMDS 모듈(Fine-grained Multi-scale Dynamic Selection Module)과 AGMF 모듈(Adaptive Gated Multi-branch Focus Fusion Module)을 소개하여 YOLOv9 기반의 새로운 객체 탐지 모델 FA-YOLO를 개발했습니다.

- **Technical Details**: FMDS 모듈은 정교한 다중 스케일 특성 맵에서 동적인 특성을 선택하고 융합하는 방식으로, 소형, 중형, 대형 목표물의 탐지 정확도를 크게 향상시킵니다. AGMF 모듈은 다양한 특성을 보완적으로 융합하여 특징 맵의 표현력을 높이고, 전체적인 특성 융합의 효율성을 증가시킵니다.

- **Performance Highlights**: FA-YOLO 모델은 PASCAL VOC 2007 데이터셋에서 평균 정밀도(mAP) 66.1%를 달성하며, YOLOv9의 65.1%에서 1.0% 향상된 성능을 보여줍니다. 또한, 소형, 중형, 대형 목표물의 탐지 정확도가 각각 44.1%, 54.6%, 70.8%로 YOLOv9 대비 2.0%, 3.1%, 0.9% 향상되었습니다.



### Safe Bayesian Optimization for High-Dimensional Control Systems via Additive Gaussian Processes (https://arxiv.org/abs/2408.16307)
- **What's New**: 본 논문에서는 고차원 안전 Bayesian 최적화 방법을 제안하여 여러 제어기를 동시에 및 안전하게 최적화하는 것을 목표로 합니다. 기존의 Gaussian 프로세스를 사용하되, 전통적인 제곱 지수 커널(square-exponential kernels) 또는 Matérn 커널 대신 가산(generative) Gaussian 커널을 활용하여 정보를 업데이트하는 효율성을 높였습니다.

- **Technical Details**: 가산 Gaussian 프로세스를 기반으로 하는 새로운 안전 Bayesian 최적화 방법을 개발하였으며, 이를 통해 복잡한 제어 시스템의 파라미터를 최적화합니다. 이 방법은 단계적(iterative) 방법을 사용하여 고차원 제어 최적화를 위한 이론적 수렴성을 보장합니다.

- **Performance Highlights**: 영구 자석 동기 모터(PMSM)에서의 실험 결과에 따르면, 제안된 방법은 기존의 안전 Bayesian 최적화 알고리즘에 비해 최적의 파라미터를 보다 효율적으로 획득할 수 있으며, 안전성을 보장함에도 불구하고 제어 성능을 향상시킵니다.



### Physics of Language Models: Part 2.2, How to Learn From Mistakes on Grade-School Math Problems (https://arxiv.org/abs/2408.16293)
Comments:
          arXiv admin note: text overlap with arXiv:2407.20311

- **What's New**: 이 논문에서는 언어 모델의 정확성을 향상시키기 위해 'error-correction' 데이터를 프리트레이닝(pretraining) 단계에 직접 통합하는 유용성을 연구하였습니다. 이는 오류가 포함된 해결 단계가 즉시 수정된 데이터로 구성됩니다.

- **Technical Details**: 논문은 'retry data'(오류 및 즉각적인 수정)로 훈련된 언어 모델이 오류 수정을 잘 수행할 수 있는지를 조사합니다. 측정 방법으로는 iGSM 데이터셋을 사용하며, 이는 프로그램이 생성한 초등학교 수준의 수학 문제로 구성되어 있습니다. 연구는 모델이 잘못된 해결 단계를 감지하고, 그로부터 재생성하는 'retry upon regret' 프로세스를 나타냅니다.

- **Performance Highlights**: 수학 데이터셋에서 'retry upon regret' 프로세스가 개선된 정확도를 가져와, 기존의 beam search 방법에 비해 높은 성과를 보여주었습니다. 모델이 이미 수정할 필요가 있는 오류를 인식하는 경향이 있으며, 99% 이상의 오류 감지 정확도를 기록했습니다.



### OpenFGL: A Comprehensive Benchmarks for Federated Graph Learning (https://arxiv.org/abs/2408.16288)
Comments:
          Under Review

- **What's New**: 이번 논문에서는 OpenFGL이라는 통합 벤치마크를 제안하여 다양한 연합 그래프 학습(Federated Graph Learning, FGL) 시나리오를 평가할 수 있도록 합니다. 이 벤치마크는 16개의 응용 도메인에서 38개의 그래프 데이터셋과 8개의 분산 데이터 시뮬레이션 전략을 포함하며, 최근 제안된 18개의 SOTA(Federated Learning) 알고리즘을 사용자 친화적인 API를 통해 제공합니다.

- **Technical Details**: OpenFGL은 Graph-FL 및 Subgraph-FL 두 가지 주요 FGL 시나리오를 통합하며, 응용 도메인 , 그래프 특성, 그리고 다양한 그래프 기반 다운스트림 작업에 대한 지원을 포함합니다. 또한, 효과성, 견고성, 효율성을 평가하는 전략적 관점을 제공합니다. 이를 통해 각 FGL 알고리즘의 유효성 및 한계를 심도 있게 비교할 수 있습니다.

- **Performance Highlights**: OpenFGL을 사용한 경험적 연구는 FGL 알고리즘의 효과성과 한계를 확인했으며, 데이터 소음, 클라이언트 참여 저조 문제 및 데이터 희소성 등의 문제를 해결하기 위한 개인화된 다중 클라이언트 협업이 중요하다는 통찰을 제공합니다. 또한, 산업 규모 데이터셋에서의 알고리즘 효율성과 혁신적인 협업 패러다임의 필요성을 강조합니다.



### Beyond Uncertainty: Evidential Deep Learning for Robust Video Temporal Grounding (https://arxiv.org/abs/2408.16272)
Comments:
          Ongoing work: 28pages, 19 figures, 7 tables. Code is available at: https://kaijing.space/SRAM/

- **What's New**: 기존 Video Temporal Grounding (VTG) 모델들은 정확성에서는 뛰어나지만 open-world의 도전 과제를 간과하고 있습니다. 이 논문에서는 SRAM이라는 새로운 네트워크 모듈을 소개하여 사용자 입력에 기반하여 불확실성을 동적으로 추정할 수 있게 하고 있습니다.

- **Technical Details**: SRAM은 두 단계의 cross-modal alignment 작업을 활용하여 불확실성을 측정하는 Deep Evidential Regression (DER)을 통합합니다. 이로 인해 모델은 훈련 중에 불확실성을 명확히 정량화하여, 자신의 한계를 인지하고 "모르겠습니다"라고 응답할 수 있게 됩니다. 이研究에서는 기존 DER의 구조적 결함을 해결하기 위한 Geom-regularizer를 개발했습니다.

- **Performance Highlights**: 광범위한 정량적 및 정성적 결과를 통해 SRAM 모듈의 효과성, 강건성 및 해석 가능성을 확인했습니다. 이 연구는 VTG에서 DER이 성공적으로 통합된 첫 번째 사례로 간주됩니다.



### LoraMap: Harnessing the Power of LoRA Connections (https://arxiv.org/abs/2408.16264)
Comments:
          13 pages, 9 figures, 5 tables

- **What's New**: 이번 논문은 여러 개의 Low-Rank Adaptation (LoRA) 간의 연결성을 강화하는 방법을 탐구하여 대규모 언어 모델(LLM)의 신뢰성 문제를 해결하고자 합니다. 특히 COVID-Fact 데이터를 기반으로 하여 새로운 추론 데이터셋을 생성하고 이를 통해 다양한 관점에서 정보를 추론할 수 있는 LoRA를 구체화합니다.

- **Technical Details**: 연구에서는 사실 확인을 위해 맞춤형으로 설계된 세 가지 추론 데이터셋(DifferenceCoT, EntityCoT, CorrectClaim)을 생성하고, LoRA를 각 데이터셋에 맞춰 미세 조정(fine-tuning)합니다. LoraMap을 도입하여 LoRA 간의 관계를 매핑(mapping)하고, 이는 인간의 뇌의 정보 처리 방식에서 영감을 받았습니다. 이 방법은 LoRA의 단순한 가중합(weighted sum) 대신 정보를 교환하는 방식으로 작동합니다.

- **Performance Highlights**: COVID-Fact 데이터셋에서 LoraMap은 기존의 LoraHub 방법보다 우수한 성능을 보이며, LoraConcat에 비해 훨씬 적은 파라미터로도 더 나은 결과를 제공합니다. 이는 LoraMap의 효율성과 연결성 강화를 강조합니다.



### Evaluating Time-Series Training Dataset through Lens of Spectrum in Deep State Space Models (https://arxiv.org/abs/2408.16261)
Comments:
          11 pages, 5 figures

- **What's New**: 본 연구에서는 deep neural networks (DNNs) 및 상태공간 모델(state space models, SSMs)로 훈련된 시간 시계열 데이터셋의 성능을 평가하는 방법에 대해 조사합니다. 이 연구는 새로운 과제를 해결하기 위해 훈련 데이터셋의 효과성을 조기 평가할 수 있는 지표로 K-spectral metric을 제안합니다.

- **Technical Details**: K-spectral metric은 신호의 최상위 K 스펙트럼의 합으로 정의됩니다. 딥 SSM에서 각 층은 선형 다이나믹 시스템으로 간주될 수 있으며, 해당 신호의 스펙트럼을 기반으로 훈련 데이터셋의 효과성을 평가합니다.

- **Performance Highlights**: K-spectral metric은 훈련 데이터셋의 품질을 평가하는 데 유용하며, 성능과 높은 상관관계(coefficient)를 가지는 것으로 나타났습니다. 실험 결과, 이 지표는 데이터 샘플 크기 및 검증 손실(validation loss)보다 더 큰 상관계수를 보여줍니다.



### Coalitions of AI-based Methods Predict 15-Year Risks of Breast Cancer Metastasis Using Real-World Clinical Data with AUC up to 0.9 (https://arxiv.org/abs/2408.16256)
- **What's New**: 유방암에 대한 보다 나은 예후 도구의 필요성을 강조하며, 머신러닝을 활용한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 이 연구에서는 머신러닝과 grid search, Bayesian Networks를 사용하여 기존 데이터를 바탕으로 임상 및 조직병리학적 매개변수를 분석합니다. ROC(Receiver Operating Characteristic) 분석에서 AUC(Area Under the Curve) 값이 0.9에 이르는 알고리즘을 개발했습니다.

- **Performance Highlights**: 이 알고리즘은 기존의 로타리 종양 평가를 초과하는 추가 검사를 필요로 하지 않으며, 빠르게 임상 관리로 전환할 수 있는 잠재력을 가지고 있습니다.



### Enhancing Conditional Image Generation with Explainable Latent Space Manipulation (https://arxiv.org/abs/2408.16232)
Comments:
          7 pages , 5 figures

- **What's New**: 이 논문은 조건부 프롬프트에 맞춰 참조 이미지에 대한 충실도를 달성하기 위한 새로운 접근 방식을 제안합니다. 기존 방법이 가진 한계를 극복하고자, 확산 모델(diffusion model)과 잠재 공간 조작(latent space manipulation), 그라디언트 기반 선택적 주의 메커니즘(gradient-based selective attention mechanisms)을 통합하였습니다.

- **Technical Details**: 이 연구에서는 Grad-SAM (Gradient-based Selective Attention Manipulation)을 활용하여 디노이즈된 잠재 벡터와 관련된 중요 점수를 도출하고, 특정 시점에서 마스크를 생성하여 객체를 보존하면서 참조 이미지의 특성을 통합합니다. 이는 텍스트 조건에 따라 충실한 객체 형성을 보장하면서 배경을 정제하여 더 일관된 구성을 제공합니다.

- **Performance Highlights**: Places365 데이터셋을 사용한 실험에서, 제안한 모델이 평균 및 중앙값 Frechet Inception Distance (FID) 점수에서 기존 모델보다 우수한 결과를 보여주었으며, 높은 CLIP 점수를 통해 생성된 이미지와 제공된 텍스트 설명 간의 정렬 성능이 경쟁력을 가진다는 것을 입증했습니다.



### Anchor-Controlled Generative Adversarial Network for High-Fidelity Electromagnetic and Structurally Diverse Metasurface Design (https://arxiv.org/abs/2408.16231)
- **What's New**: 이번 논문에서는 Anchor-controlled Generative Adversarial Network (AcGAN)라는 새로운 생성 프레임워크를 소개합니다. 이 프레임워크는 전자기(EM) 신뢰도를 우선시하면서도 다양한 구조의 메타서피스를 생성할 수 있도록 합니다.

- **Technical Details**: AcGAN은 기존 방법들과 달리 물리적 외관을 단순히 복제하는 것에서 벗어나, 서로 다른 물리적 속성을 지닌 구조이더라도 유사한 전자기 응답을 보이는 다양한 구조를 생성합니다. 또한, Spectral Overlap Coefficient (SOC)를 도입하여 생성된 디자인과 타겟 간의 스펙트럼 신뢰도를 정확하게 측정합니다.

- **Performance Highlights**: Empirical evidence(경험적 증거)는 AcGAN이 설계 프로세스를 간소화하고, 우수한 전자기 정밀도를 달성하며, 다양한 디자인 가능성을 도모함을 보여줍니다.



### LLaVA-SG: Leveraging Scene Graphs as Visual Semantic Expression in Vision-Language Models (https://arxiv.org/abs/2408.16224)
- **What's New**: 최근 큰 비전 언어 모델(VLM)에서 Vision Transformer(ViT) 아키텍처를 기반으로 한 비전 인코더가 사용됩니다. 본 논문에서는 Scene Graph Expression(SGE) 모듈을 VLM에 적용하여 이러한 제약을 극복하고자 합니다. 이 모듈은 이미지 내 복잡한 의미 정보를 구조적으로 표현하고 VLM의 인식 능력을 개선합니다.

- **Technical Details**: SGE 모듈은 이미지에서 개체를 추출하고 이들 간의 관계를 표현하기 위한 장면 그래프(Scene Graph)를 구성합니다. 이는 개체 수준에서 의미 정보를 유지하고 전달하는 방법으로, 이미지 내 객체를 태깅하고, 경계 상자(Bounding Box)를 감지하여 세분화 세마틱 이해(Semantic Segmentation)를 수행합니다. 최종적으로 LLaVA-SG 모델로 통합하여 VLM의 시각적 이해 능력을 향상시킵니다.

- **Performance Highlights**: 광범위한 실험을 통해 SGE 모듈 통합이 VLM의 시각 언어 작업에서 성능을 상당히 향상시키는 것으로 나타났습니다. 이는 복잡한 의미 세부 사항을 보존하고 시각적 이해를 강화하는 데 효과적임을 의미합니다.



### SSDM: Scalable Speech Dysfluency Modeling (https://arxiv.org/abs/2408.16221)
- **What's New**: 본 논문에서는 음성 불유창성 모델링(Speech Dysfluency Modeling)의 새로운 접근 방식을 제안한다. SSDM(Scalable Speech Dysfluency Modeling)은(1) 조음 동작을 활용한 확장 가능한 강제 정렬(Forced Alignment) 기법을 채택하고, (2) 연결주의 하위 시퀀스 정렬기(Connectionist Subsequence Aligner, CSA)를 도입하여 불유창성 정렬을 달성하며, (3) Libri-Dys라는 대규모 시뮬레이션 불유창성 말뭉치를 소개하고, (4) 대형 언어 모델(LLMs)의 힘을 활용하여 종단 간 시스템을 개발한다.

- **Technical Details**: 이 연구에서는 기본 물리 법칙에 기반한 음성 표현 학습을 재조명하고, 음성 불유창성 모델링을 위한 확장 가능한 표현인 신경 조음 동작 점수(Neural Articulatory Gestural Scores)를 제안한다. 연결주의 하위 시퀀스 정렬기(CSA)는 음향 표현과 텍스트를 연결하여 불유창성을 인식하는 차별 가능하고 확률론적 강제 정렬기를 만들어낸다.

- **Performance Highlights**: SSDM은 불유창성 모델링 분야에서 표준으로 자리 잡을 것으로 기대된다. 또한 Libri-Dys 데이터셋을 오픈 소스화하여 추가 연구를 촉진시킬 것이다.



### M4CXR: Exploring Multi-task Potentials of Multi-modal Large Language Models for Chest X-ray Interpretation (https://arxiv.org/abs/2408.16213)
- **What's New**: 이번 논문에서는 CXR(Chest X-ray) 해석을 강화하기 위해 M4CXR이라는 다중 모달(Modal) LLM을 제안합니다. 이 모델은 다양한 작업별 데이터 세트를 통합한 시각적 지침 따르기 데이터 세트에서 훈련되었습니다.

- **Technical Details**: M4CXR은 여러 작업을 지원하며, 특히 의학 보고서 생성(MRG), 시각적 그라운딩(Visual Grounding), 시각적 질문 응답(VQA)에서 탁월한 성능을 보입니다. 모델의 훈련 과정에서는 Chain-of-Thought(CoT) 프롬프트 전략을 사용하여 CXR 이미지에서 발견 사항을 식별하고 해당 보고서를 생성합니다.

- **Performance Highlights**: M4CXR은 다양한 MRG 시나리오에서 탁월한 임상 정확도를 달성하며, 시각적 그라운딩과 VQA에서도 Specialized 모델에 필적하는 성능을 발휘합니다. 양적 및 질적 평가 모두에서 M4CXR의 다재다능성을 강조하며, 일관된 임상 정확도를 유지합니다.



### Short-Term Electricity-Load Forecasting by Deep Learning: A Comprehensive Survey (https://arxiv.org/abs/2408.16202)
- **What's New**: 이번 논문은 딥러닝 기반의 단기 전력 수요 예측(STELF)에 대한 포괄적인 조사로, 지난 10년간의 연구를 종합하여 STELF의 전체 예측 프로세스를 분석 및 요약하고, 현재의 연구 도전과제와 향후 연구 방향을 제시합니다.

- **Technical Details**: 저자들은 STELF를 위해 딥러닝 기법을 사용하며, 데이터 전처리(data preprocessing), 특징 추출(feature extraction), 딥러닝 모델링 및 최적화(optimize), 결과 평가(evaluation) 등 모든 예측 과정에 대한 포괄적인 조사를 수행합니다. 논문에서는 RNNs, LSTMs, GRUs와 같은 심층 신경망(deep neural networks)을 적용하여 비선형적이고 복잡한 전력 수요 데이터를 모델링합니다.

- **Performance Highlights**: 딥러닝 기법을 사용한 STELF 모델은 높은 정확도의 예측을 제공하며, 이는 전력 시스템의 효율적인 운영과 관리에 기여할 수 있습니다. 또한 STELF 정확도를 향상시키는 것은 비용 절감 및 계획 수립의 효율성을 높이는데 중요한 역할을 합니다.



### PolarBEVDet: Exploring Polar Representation for Multi-View 3D Object Detection in Bird's-Eye-View (https://arxiv.org/abs/2408.16200)
Comments:
          11 pages, 6 figures

- **What's New**: 최근의 LSS(Lift-Splat-Shoot) 기반 다각도 3D 객체 탐지 기술이 자율 주행에 있어 경제적이고 배치하기 쉬운 솔루션을 제공하지만, 기존 방법들은 비균일한 이미지 정보 분포를 고려하지 않고, 뷰 대칭성을 잘 활용하지 못하는 문제를 다루었습니다. 본 논문에서는 Polar BEV(Polar Bird's-Eye-View) 표현을 도입하여 이를 개선하고 새로운 다각도 3D 객체 탐지기인 PolarBEVDet를 제안하였습니다.

- **Technical Details**: Polar BEV 표현은 이미지 정보 분포에 맞춰 그리드 배치를 조정하고, 주변 카메라의 뷰 대칭성을 보존하기 위해 세 가지 모듈을 설계하였습니다. 1) Polar View Transformer: 다각도 이미지 특성을 원통형 좌표계로 변환하고 BEV 공간을 각도 및 반경 방향으로 표시합니다. 2) Polar Temporal Fusion Module: 다중 프레임 폴라 BEV 특성을 정렬합니다. 3) Polar Detection Head: Polar 매개변수를 활용한 3D 객체 탐지를 수행합니다. 또한 2D 보조 탐지 헤드와 공간 주의 강화(SAE) 모듈을 디자인하여 특징 추출의 품질을 높였습니다.

- **Performance Highlights**: nuScenes 데이터셋에서 평가한 결과, PolarBEVDet는 63.5% NDS(Neighborhood Detection Score)와 55.8% mAP(mean Average Precision)로 뛰어난 탐지 성능을 달성하였습니다.



### A More Unified Theory of Transfer Learning (https://arxiv.org/abs/2408.16189)
- **What's New**: 이 논문에서는 소스 위험(source risk)이 감소함에 따라 목표 위험(target risk)이 얼마나 빠르게 감소하는지를 측정하는 모듈라이( moduli )의 연속성에 대한 연구를 발표합니다. 기존의 관련성 측정 기준에 많은 영향을 미치며, 회귀 및 분류 문제에서 여러 상황에 대한 통합된 관점을 제공합니다.

- **Technical Details**: 연속성 모듈라이( moduli of continuity )인 δ를 정의하며, 이를 통해 소스 분포 P와 목표 분포 Q 간의 관련성을 측정하는 다양한 기존 접근 방식을 획일화하려고 합니다. 이 모듈라이는 리스크와 관련된 정량적 측정으로, 소스와 목표 데이터 모두를 사용할 수 있는 여러 학습 시나리오에 적용 가능합니다. δ는 추정할 수 없는 경우에도 적응형 절차에 의해 거의 정확한 비율을 달성할 수 있습니다.

- **Performance Highlights**: 모듈라이 δ는 기존의 여러 관련성 측정 기준과 비교하여 더 빠르고 타이트한 전이 속도를 제공하며, 이는 신뢰구간(confidence sets)에 대한 주요 이론적 결과로 뒷받침됩니다. 이 논문은 목표 데이터가 부족한 경우에도 소스 데이터로부터 정보 논리적으로 예측 능력을 확장할 수 있는 가능성을 시사합니다.



### Real-Time Energy Pricing in New Zealand: An Evolving Stream Analysis (https://arxiv.org/abs/2408.16187)
Comments:
          12 Pages, 8 figures, short version accepted by PRICAI

- **What's New**: 이 논문은 뉴질랜드의 전력 가격에 대한 실시간 시간 연속 데이터 세트를 새롭게 소개합니다. 이 데이터는 뉴질랜드 정부가 운영하는 전기시장정보(EMI) 웹사이트에서 수집되었습니다. 이러한 데이터 세트는 스트리밍 회귀 학습 작업을 위한 적절한 데이터 세트의 부족을 해결하기 위한 것으로, 다양한 분석 및 실험을 거치며 유용성을 입증합니다.

- **Technical Details**: 논문에서는 Sliding Window KNN, Adaptive Random Forest Regressor (ARF-Reg), Self-Optimising K-Nearest Leaves (SOKNL), Half-Space Trees (HST) 등 여러 최신 알고리즘을 활용하여 데이터 스트림 처리를 수행합니다. 특히 온실가스 배출에 영향을 미치는 콘셉트 드리프트(drift)에 적응하는 내용을 포함하여, 예측 간격을 동적으로 조정하여 회귀 분석의 정확성을 높입니다.

- **Performance Highlights**: 실험 결과, 제안된 데이터 세트는 에너지 가격 예측에 유용하고, 데이터 스트림 회귀 문제의 기존의 도전 과제와 향후 연구 기회를 드러냅니다. 이러한 분석을 통해, 논문은 데이터 스트림 내의 변화 감지 및 적응 방법의 중요성을 강조합니다.



### LLM-assisted Labeling Function Generation for Semantic Type Detection (https://arxiv.org/abs/2408.16173)
Comments:
          VLDB'24-DATAI

- **What's New**: 본 논문에서는 데이터 레이크 테이블에서 열의 의미적 유형(semantic type) 검출을 위한 새로운 접근 방식을 제안합니다. 프로그램 전송(supervision) 기법을 활용하여 훈련 데이터 주석(annotation)을 자동화하고, 기존의 저품질 주석 문제를 극복하기 위해 Large Language Models (LLMs)를 활용합니다.

- **Technical Details**: 제안된 방법론은 LLM을 통해 라벨링 함수(labeling function)를 생성하고, Snorkel 프레임워크를 기반으로 하여 주석 결과를 집계하는 방식입니다. 제안된 파이프라인은 시드 인스턴스를 활용하여 LLM이 LFs를 생성하고, 이로부터 얻은 LFs를 Snorkel을 통해 통합하여 레이블을 부여합니다. 각 레이블링 함수는 키워드, 통계, 정규 표현식(regular expression)을 사용하여 정의됩니다.

- **Performance Highlights**: 실험 결과, 제안된 시스템은 데이터 레이크 테이블에 대한 효과적인 라벨링 함수를 성공적으로 생성하였으며, 이는 모델의 최종 훈련 성과에 긍정적인 영향을 미쳤습니다. 더욱이, LLM 기반의 시드 인스턴스를 포함한 프롬프트 공학(prompt engineering)이 LFs 품질을 개선하는 데 중요한 역할을 한다는 것을 확인하였습니다.



### Simulating realistic short tandem repeat capillary electrophoretic signal using a generative adversarial network (https://arxiv.org/abs/2408.16169)
Comments:
          29 pages, 9 Figures

- **What's New**: 새로운 연구에서는 DNA 프로필의 전기영동 신호를 분석하기 위해 인공 신경망(ANN)을 사용하는 대신, 픽스 투 픽스(pix2pix) GAN을 수정한 생성적 적대 신경망(GAN)을 개발했습니다. 이 GAN은 현실적인 사전 레이블링된(training 데이터) 데이터를 시뮬레이션하는 데 사용됩니다.

- **Technical Details**: 이 연구에서는 1078개의 DNA 프로필을 사용하여 GAN을 학습시켰습니다. GAN의 생성기(generator)는 DNA 프로필 정보를 시뮬레이션하는 능력을 부여받고, 전기영동 신호에서 발생하는 노이즈(noise)와 아티팩트(artefact) 요소를 적용하는 '리얼리즘 필터(realism filter)'로 사용됩니다.

- **Performance Highlights**: GAN을 통해 생성된 데이터는 신뢰할 수 있는 전기영동 신호 시뮬레이션을 제공하며, 이는 ANN의 훈련에 필요한 대량의 레이블링된 데이터 구축의 한계를 극복하는 데 기여할 것으로 기대됩니다.



### FRACTURED-SORRY-Bench: Framework for Revealing Attacks in Conversational Turns Undermining Refusal Efficacy and Defenses over SORRY-Bench (https://arxiv.org/abs/2408.16163)
Comments:
          4 pages, 2 tables

- **What's New**: 이번 논문은 MULTI-TURN 대화 공격에 대한 대형 언어 모델(LLM)의 안전성을 평가하기 위해 FRACTURED-SORRY-Bench라는 프레임워크를 소개합니다. 이는 기존 SORRY-Bench 데이터셋을 기반으로 하며, 유해한 쿼리를 무해한 서브 질문으로 분해하여 적대적 프롬프트를 생성하는 간단하면서도 효과적인 방법을 제공합니다.

- **Technical Details**: FRACTURED-SORRY-Bench는 주어진 쿼리를 4-7개의 개별 서브 질문으로 분해하고, 이 서브 질문들을 대화 형식으로 순차적으로 LLM에 제시합니다. 이 방법은 LLM의 컨텍스트 윈도우를 이용하고 서브 질문마다 명시적인 유해 언어를 피함으로써 내용 필터와 안전 메커니즘을 우회하는 것을 목표로 합니다.

- **Performance Highlights**: 이 연구는 GPT-3.5-Turbo에서 10.9배 증가한 공격 성공률(ASR)을 보였으며, GPT-4, GPT-4o, GPT-4o-mini 모델에서도 유의미한 증가를 보여줍니다. 연구 결과, 49.33%의 파편화된 프롬프트 세트가 원래의 유해한 의도를 성공적으로 전달했습니다.



### Improving Generalization of Speech Separation in Real-World Scenarios: Strategies in Simulation, Optimization, and Evaluation (https://arxiv.org/abs/2408.16126)
Comments:
          In Proceedings of the 25th Annual Conference of the International Speech Communication Association, Interspeech 2024

- **What's New**: 본 논문은 다양한 음향 환경에서 겹치는 화자의 음성을 Robust하게 분리하는 문제를 다루고 있습니다. 기존 데이터셋의 일반화 부족 문제를 해결하기 위해, AC-SIM이라는 혁신적인 데이터 시뮬레이션 파이프라인을 소개하며, 다양한 훈련 목표를 permutation invariant training (PIT)에 통합하여 모델의 separation quality를 개선하고자 합니다.

- **Technical Details**: AC-SIM은 음향(acoustic) 및 콘텐츠(content) 변화를 포함한 데이터 시뮬레이션 프로세스를 통해 다양한 음성 시나리오에서 모델을 훈련시키는 방법입니다. 이 과정은 Speech Separation을 위한 두 가지 주요 목표, 즉 우수한 separation quality와 모델의 일반화 능력을 중심으로 구성되어 있습니다. 이를 위해 상대적으로 독립적이며 다양한 샘플을 훈련 데이터에 적용하여 다채로운 환경을 시뮬레이션합니다.

- **Performance Highlights**: 실험을 통해, 제안된 방법이 기존보다 다양한 음성 시나리오에서 효과적으로 성능을 향상시킬 수 있음을 보여주었습니다. AC-SIM을 통해 구성된 평가 세트는 전통적인 벤치마크와 실제 사례 모두에서 일관된 성과를 반영함으로써 모델 성능의 향상을 입증했습니다.



### ChartEye: A Deep Learning Framework for Chart Information Extraction (https://arxiv.org/abs/2408.16123)
Comments:
          8 Pages, and 11 Figures

- **What's New**: 이 연구에서는 차트 이미지에서 정보를 자동으로 추출하기 위한 딥러닝 기반 프레임워크를 제안합니다. 이 프레임워크는 차트 유형 및 텍스트 역할 분류를 위해 계층적 비전 트랜스포머(hierarchical vision transformers)를 사용하고, 텍스트 감지는 YOLOv7을 활용합니다. 또한, 인식 결과를 개선하기 위해 Super Resolution Generative Adversarial Networks(ESRGAN)를 사용하여 텍스트를 강화합니다.

- **Technical Details**: 제안된 프레임워크는 차트 유형 분류, 텍스트 감지 및 인식, 텍스트 역할 분류의 각 주요 단계를 처리합니다. 이 프레임워크는 딥 컨볼루션 및 비전 트랜스포머 기반 접근 방식을 결합하여 차트 이미지에서 효과적으로 정보를 추출합니다. 특히, 차트 유형 분류 및 텍스트 역할 분류에는 계층적 비전 트랜스포머를 사용하며, 텍스트 감지를 위해서는 한 단계(object detection) 객체 탐지기를 채택합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 차트 유형 분류에서 0.97, 텍스트 역할 분류에서 0.91, 텍스트 감지에서 평균 정밀도(Mean Average Precision) 0.95의 뛰어난 성능을 달성했습니다.



### Data Formulator 2: Iteratively Creating Rich Visualizations with AI (https://arxiv.org/abs/2408.16119)
- **What's New**: 이번 연구에서는 Data Formulator 2라는 LLM(대형 언어 모델) 기반 시각화 시스템을 소개하고, 데이터 분석가들이 데이터 처리 및 차트 사양을 반복적으로 다루는 과정에서의 도전 과제를 해결하고자 합니다. 이 시스템은 사용자가 UI와 자연어 입력을 혼합하여 시각화 의도를 설명할 수 있도록 하여 AI가 데이터 변환을 담당함으로써 반복적인 작업을 지원합니다.

- **Technical Details**: Data Formulator 2는 혼합형 UI와 자연어(NL) 입력을 통해 차트 사양을 지정할 수 있으며, 데이터를 효과적으로 관리하기 위한 데이터 스레드(data threads) 기능을 제공합니다. 사용자는 이전 디자인을 재사용하고 새로운 디자인으로 나아갈 수 있으며, AI는 사용자의 자연어 입력을 바탕으로 데이터 변환 코드를 생성합니다.

- **Performance Highlights**: 사용자 연구 결과, Data Formulator 2는 참가자들이 복잡한 데이터 탐색 세션을 완료할 수 있도록 자신의 반복 전략을 개발하는 데 도움을 주었습니다. 이 시스템을 통해 사용자들은 처음부터 시작하지 않고도 이전 결과를 손쉽게 활용하여 더 나은 시각화를 생성할 수 있게 되었습니다.



### Ensuring Equitable Financial Decisions: Leveraging Counterfactual Fairness and Deep Learning for Bias (https://arxiv.org/abs/2408.16088)
Comments:
          8 pages, 7 figures

- **What's New**: 이 연구는 인공지능 모델에서 성별과 같은 민감한 특성과 관련된 공정함(fairness) 및 편향(bias) 문제를 해결하기 위한 최신의 편향 완화(bias mitigation) 방법들을 조사합니다. 특히, 데이터 증강(data augmentation)과 반사실적 공정성(counterfactual fairness)을 통합한 접근 방식에 초점을 맞춥니다.

- **Technical Details**: 연구는 기계 학습(machine learning) 모델에서 공정함을 보장하기 위해 새로운 기술들을 제안하며, 이는 금융 산업에서 대출 승인 프로세스에 적용됩니다. 구체적으로, 편향 완화 기법이 포함된 실험을 통해 비대칭 데이터셋(skewed financial dataset)에 대해 평가를 수행합니다.

- **Performance Highlights**: 이 연구는 제안된 접근 방식이 성별 편향을 줄이고 보다 공정한 결과를 달성하는 데 효과적임을 보여줍니다. 결과적으로, 도출된 발견들은 기계 학습 모델을 개발할 때 공정성 인식 기법(fairness-aware techniques)의 중요성을 강조합니다.



### Verification methods for international AI agreements (https://arxiv.org/abs/2408.16074)
- **What's New**: 이 논문에서는 고급 AI 개발에 관한 국제 협약 준수를 확인하는 방법 10가지를 분석하고, 이를 통해 비허가 AI 훈련 및 비허가 데이터 센터와 같은 잠재적 위반을 탐지할 수 있는 방법을 제시합니다.

- **Technical Details**: 확인 방법은 세 가지 범주로 나뉘며, (a) 국가 기술 수단(national technical means), (b) 접근 의존 방법(access-dependent methods), (c) 하드웨어 의존 방법(hardware-dependent methods)입니다. 각 방법의 설명, 역사적 사례, 회피 기법을 제시하고, 향후 국제 AI 거버넌스 협약의 검증 및 집행에 대한 권고 사항을 도출합니다.

- **Performance Highlights**: 제안된 검증 방법은 다양한 한계가 있으며, 국제 협정의 실행 가능성을 검토하는 것이 중요합니다. 검증 방법은 각국의 정치적 맥락, 침입성, 불완전한 탐지, 개발 단계 등 여러 요인에 영향을 받을 수 있습니다. 향후 연구 분야로는 검증 체계의 적대적 행위 시나리오, 국제 AI 거버넌스 기관 설계, 비준수 시 대응 전략 등이 제안됩니다.



### Using Large Language Models to Create AI Personas for Replication and Prediction of Media Effects: An Empirical Test of 133 Published Experimental Research Findings (https://arxiv.org/abs/2408.16073)
Comments:
          24 pages, 3 figures, 2 tables

- **What's New**: 이 보고서는 대규모 언어 모델(LLMs)이 발표된 메세지 효과 연구의 정확한 복제를 촉진할 수 있는 가능성을 분석합니다. 우리는 2023년 1월에서 2024년 5월 사이의 Journal of Marketing에서 14개의 논문에서 133개의 실험 결과를 복제함으로써 LLM 기반 참여자(페르소나)들을 테스트했습니다.

- **Technical Details**: 우리는 Viewpoints AI라는 새로운 소프트웨어 도구를 사용했습니다. 이 도구는 연구 설계(study designs), 자극(stimuli), 측정(measures)을 입력으로 받아 LLM이 특정한 유니크한 페르소나 샘플로 작동할 수 있도록 자동으로 프롬프트(prompts)를 생성하고 응답을 수집하여 최종 출력으로 완전한 데이터셋과 통계 분석을 제공합니다. 본 연구에서 사용된 LLM은 Anthropic의 Claude Sonnet 3.5입니다. 우리는 원래 인간 연구에서 보고된 것과 동일한 샘플 속성(sample attributes), 연구 설계(study designs), 자극(stimuli), 측정을 사용하여 19,447개의 AI 페르소나를 생성하였습니다.

- **Performance Highlights**: 우리 LLM 복제는 원본의 주요 효과(original main effects) 76% (111개 중 84개)를 성공적으로 재현하였으며, 이는 미디어 자극(media stimuli) 응답 연구의 AI 보조 복제가 강력한 가능성을 지닌다는 것을 보여줍니다. 상호작용 효과(interaction effects)를 포함할 경우, 전체 복제율은 68% (133개 중 90개)였습니다. 이 연구는 소셜 사이언스의 복제 위기(replication crisis), 샘플링 저항(sample robustness) 문제 해결, 다양한 미디어 자극에 대한 소비자 반응을 신속하게 테스트할 수 있는 능력에 대해 논의합니다.



### Identification of Prognostic Biomarkers for Stage III Non-Small Cell Lung Carcinoma in Female Nonsmokers Using Machine Learning (https://arxiv.org/abs/2408.16068)
Comments:
          This paper has been accepted for publication in the IEEE ICBASE 2024 conference

- **What's New**: 이 연구는 비흡연 여성의 3기 비소세포 폐암(stage III NSCLC)과 관련된 주요 바이오마커(biomarkers)를 식별하기 위해 GDS3837 데이터셋의 유전자 발현 프로파일링을 사용했습니다. XGBoost 머신러닝 알고리즘을 활용하여 AUC 점수 0.835를 달성하며 강력한 예측 성능을 보였습니다.

- **Technical Details**: 주요 바이오마커로는 CCAAT enhancer binding protein alpha (C/EBP-alpha), lactate dehydrogenase A4 (LDHA), UNC-45 myosin chaperone B (UNC-45B), checkpoint kinase 1 (CHK1), hypoxia-inducible factor 1 subunit alpha (HIF-1-alpha)가 확인되었습니다. 이 연구는 GDS3837 데이터셋을 활용하였으며, 60쌍의 종양 및 인접 정상 폐조직 샘플에서 유전자 발현 분석을 진행했습니다. XGBoost를 사용하여 3기 폐암의 바이오마커를 식별하였습니다.

- **Performance Highlights**: XGBoost는 유전자 발현 프로파일링 데이터를 기반으로 3기 비소세포 폐암 예측에서 0.835의 AUC 점수를 기록하며 탁월한 성능을 보였습니다. 이러한 발견은 초기 진단 및 개인 맞춤형 치료에 대한 잠재력을 강조하며, 머신러닝과 분자 프로파일링의 통합 가치에 주목하고 있습니다.



### Efficient $k$-NN Search in IoT Data: Overlap Optimization in Tree-Based Indexing Structures (https://arxiv.org/abs/2408.16036)
Comments:
          28 pages, 21 figures, 1 table

- **What's New**: 본 논문은 데이터 공간 파티션(Partition) 중 겹침 문제를 해결하기 위해 세 가지 혁신적인 휴리스틱(Heuristics) 방법을 제안합니다. 이 방법들은 데이터 공간 겹침을 정량화하고 효율적으로 관리할 수 있도록 설계되었습니다.

- **Technical Details**: 제안된 방법은 다음과 같습니다: 
1. 볼륨 기반 방법(Volume-Based Method, VBM) - 파티션 간의 교차 볼륨(Intersection Volume)을 계산하여 겹침을 측정합니다.
2. 거리 기반 방법(Distance-Based Method, DBM) - 파티션 중심과 반경 간의 거리를 분석하여 효율성을 향상시킵니다.
3. 객체 기반 방법(Object-Based Method, OBM) - 파티션 간 공유 객체 수를 세어 직관적으로 겹침을 이해할 수 있게 합니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안된 방법들이 검색 시간을 단축시키며, 데이터 공간 파티셔닝과 전반적인 시스템 성능을 향상시키는 데 효과적임을 보여줍니다.



### An Extremely Data-efficient and Generative LLM-based Reinforcement Learning Agent for Recommenders (https://arxiv.org/abs/2408.16032)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)을 기반으로 한 강화 학습(RL) 에이전트를 WebShop 환경에서 훈련시키고 평가하는 새로운 방법론을 발표합니다. 이 방법론은 특히 e-commerce와 추천 시스템에서 인간의 지침을 이해하고 수행하는 능력을 발전시킵니다.

- **Technical Details**: 연구에서는 다양한 RL 알고리즘을 사용하여 WebShop 시뮬레이터에서 에이전트를 훈련합니다. BERT 모델을 미세 조정하여 RL 에이전트를 만들고, Proximal Policy Optimization (PPO) 및 Direct Preference Optimization (DPO) 등 최신 훈련 기술을 적용하였습니다. 또한, 생성 경로를 사용하는 RL 에이전트를 평가하여 사람의 경로를 기반으로 한 경우와 유사한 성능을 보여주었습니다.

- **Performance Highlights**: 훈련 시간(<2시간) 동안 이미지 없이 DPO 에이전트는 약 3000 스텝 후 19%의 성공률을 달성하였으며, PPO 에이전트는 15%로 상대적으로 낮은 성공률을 보였습니다. 이는 RL 에이전트 훈련의 데이터 효율성을 강조합니다.



### EMP: Enhance Memory in Data Pruning (https://arxiv.org/abs/2408.16031)
- **What's New**: 최근 대규모 언어 및 비전 모델에서 성능이 크게 향상되었지만, 프리트레이닝(Pre-training) 및 파인튜닝(Fine-tuning) 비용이 높아져 데이터셋 프루닝(Dataset Pruning) 기법으로 빨라지려는 연구가 진행되고 있습니다. 기존의 방법들은 샘플 로스(Sample Loss)를 평가 기준으로 사용하여 훈련을 위한 가장 '어려운' 샘플을 선택하는 방향으로 발전하였습니다. 하지만 프루닝 비율이 높아질수록 샘플 훈련 빈도가 균일하게 분포되어, 중요한 샘플들이 효과적으로 훈련되지 않게 되는 문제가 발생하는데, 이를 저주파 학습(Low-Frequency Learning, LFL)이라고 합니다. 본 연구에서는 LFL의 비효율성을 이론적으로 설명하고, 모델의 기억력을 향상시키기 위해 기억 항목을 제안합니다.

- **Technical Details**: 본 연구는 LFL의 평가 기능을 분해하고, 비효율성을 이론적으로 설명하였습니다. 우리 접근 방식은 두 가지 주요 시나리오인 감독 학습(Supervised Learning, SL) 및 자기 감독 학습(Self-Supervised Learning, SSL)을 다루고 있으며, 특히 SSL에서의 기억에 대한 논의는 새로운 시도입니다. SL에서는 상호 정보량(Mutual Information)을 사용해 기억 항목을 추출하고, SSL에서는 대조 학습(Contrastive Learning)에서의 기억 항목을 도출하였습니다. 최종적으로 Enhance Memory Pruning(EMP)이라는 방식을 제안하며, 이는 높은 프루닝 비율에서도 메모리를 개선시키는 방법론입니다.

- **Performance Highlights**: EMP의 성능을 이미지 분류(image classification), 자연어 이해(natural language understanding), 모델 프리트레이닝(pre-training) 방식으로 평가한 결과, EMP는 극한의 프루닝 비율에서도 모델 성능을 향상시킴을 보여주었습니다. 예를 들어, CIFAR100-ResNet50 프리트레이닝 작업에서 70% 프루닝 시 EMP는 기존 방법보다 2.2% 향상된 성능을 기록하였습니다.



### A Deep Learning Approach to Localizing Multi-level Airway Collapse Based on Snoring Sounds (https://arxiv.org/abs/2408.16030)
- **What's New**: 이 연구는 약물 유도 수면 내시경(DISE) 데이터를 활용하여 기도 폐쇄 수면 무호흡증(OSA) 환자의 다양한 수준에서 발생한 코고는 소리를 기계 학습 및 딥 러닝을 통해 분류하는 방법을 조사합니다.

- **Technical Details**: 39명의 피험자의 코고는 소리가 Velum, Oropharynx, Tongue Base, Epiglottis(VOTE) 분류 시스템을 기준으로 분석되었으며, 총 5,173개의 1초 분할(segment) 데이터셋이 Support Vector Machine(SVM), Bidirectional Long Short-Term Memory(BiLSTM), ResNet-50 모델을 훈련하고 테스트하는 데 사용되었습니다. ResNet-50은 합성곱 신경망(CNN)으로, 코고는 소리 분류에서 최고의 성능을 보였습니다.

- **Performance Highlights**: ResNet-50은 특히 다수의 기도 폐쇄를 식별하는 데 있어 우수한 성능을 보여 주며, 코고는 소리와 딥 러닝을 통합하여 OSA의 진단 및 치료 개선 가능성을 강조합니다. 그러나 제한된 샘플 크기, 데이터 불균형, 약물 유도와 자연 코고는 소리 간의 차이 등 몇 가지 도전 과제가 있으며 이는 모델의 정확도와 일반화 가능성을 높이기 위한 추가 연구가 필요함을 시사합니다.



### Meta-Learn Unimodal Signals with Weak Supervision for Multimodal Sentiment Analysis (https://arxiv.org/abs/2408.16029)
- **What's New**: 이 논문에서는 주어진 다중 모달 레이블을 통해 단일 모달 레이블을 학습하는 새로운 메타 유니레이블 생성(МUG) 프레임워크를 제안합니다. 이를 통해 단일 모달 데이터에 대한 성능 향상을 꾀하고자 합니다.

- **Technical Details**: 제안된 MUG 프레임워크는 세 가지 단계로 나뉘며, 첫 번째 단계에서는 인간 주석이 달린 다중 모달 레이블을 사용해 전체 네트워크를 훈련합니다. 이어서, contrastive 기반의 projection 모듈을 통해 단일 모달 표현과 다중 모달 표현 간의 차이를 줄입니다. 두 번째 단계는 메타 학습 전략을 포함해 단일 모달 레이블을 학습하는 것으로, 단일 모달 레이블의 품질을 추정하고 메타 유니레이블 수정 네트워크(MUCN)를 업데이트합니다. 마지막 단계에서는 훈련된 단일 모달 레이블을 사용하여 단일 및 다중 모달 학습 작업을 공동으로 진행합니다.

- **Performance Highlights**: MUG는 세 개의 데이터셋에서 다양한 경쟁 기반선들을 초과하며, 단일 모달 레이블을 학습하는 데 있어 상당한 성과를 보입니다. MUG는 다른 메타 학습 기반의 노이즈 레이블 학습 알고리즘보다 더 우수한 성능을 기록합니다.



### Toward Time-Continuous Data Inference in Sparse Urban CrowdSensing (https://arxiv.org/abs/2408.16027)
Comments:
          11 pages, 11 figures

- **What's New**: 이번 논문에서는 Mobile Crowd Sensing(MCS)에서의 데이터 수집 방법을 개선하기 위해 시간 연속적 데이터 완성(time-continuous completion) 접근 방식을 제안합니다. 기존의 Sparse MCS는 예산과 접근성 문제로 인해 불완전한 데이터 수집을 하는데, 이 논문은 그 한계를 극복하기 위한 새로운 프레임워크를 제공합니다.

- **Technical Details**: 논문에서는 Deep Matrix Factorization(DMF) 및 Recurrent Neural Network 기반의 RNN-DMF를 통해 데이터의 시간적 상관관계를 효과적으로 모델링합니다. 또한 TIME-DMF를 도입하여 불균형한 시간 간격에서도 시간 정보를 포착하여 시간 연속적 완성을 가능하게 합니다. Query-Generate(Q-G) 전략이 TIME-DMF에 포함되어 사용자가 질의할 수 있는 능력을 제공합니다.

- **Performance Highlights**: 실험을 통해 5가지 타입의 센싱 작업을 대상으로 제안된 모델의 효과성을 검증하였으며, 시간 연속적 완성 방법이 기존 접근 방식에 비해 우수한 성능을 보여주었습니다.



### XG-NID: Dual-Modality Network Intrusion Detection using a Heterogeneous Graph Neural Network and Large Language Mod (https://arxiv.org/abs/2408.16021)
Comments:
          19 pages, 6 figures

- **What's New**: 이번 논문은 사이버 보안 분야에서 흐름 수준(flow-level) 및 패킷 수준(packet-level) 데이터를 통합한 실시간 침입 탐지 시스템인 "XG-NID" 프레임워크를 소개합니다. 이 프레임워크는 이종 그래프 구조(heterogeneous graph structure)를 활용하여 네트워크 트래픽을 보다 포괄적으로 분석할 수 있도록 하고, 실시간 추론을 가능하게 합니다.

- **Technical Details**: XG-NID는 이종 그래프 신경망(heterogeneous graph neural network, HGNN)을 기반으로 하여, 흐름 데이터와 패킷 데이터를 통합하여 복잡한 관계를 포착합니다. 또한, LLM(large language model)을 통합하여 보다 인간 친화적이고 가독성이 높은 설명을 제공하며, 이론적으로는 실시간 추론과 다양한 사이버 공격 탐지 기능을 갖추고 있습니다. 새로운 흐름 특징은 시간 정보에 근거하여 모델의 설명 가능성을 강화합니다.

- **Performance Highlights**: XG-NID는 다중 클래스 분류에서 97%의 F1 점수를 기록하여 기존 기법을 초월한 성능을 보여줍니다. 이를 통해 네트워크 침입 탐지 시스템에서 새로운 기준을 정립하며, 혁신적인 데이터 융합이 가능하다는 것을 입증합니다.



### SPICED: Syntactical Bug and Trojan Pattern Identification in A/MS Circuits using LLM-Enhanced Detection (https://arxiv.org/abs/2408.16018)
Comments:
          Accepted at PAINE'24

- **What's New**: 본 논문에서는 하드웨어 수정 없이 아날로그 회로에서 트로이 안드로이드를 탐지하고 위치를 파악할 수 있는 첫 번째 대형 언어 모델(LLM) 기반의 프레임워크인 SPICED를 제안합니다.

- **Technical Details**: SPICED는 Chain-of-Thought (CoT) 추론과 몇 가지 사례를 활용하여 LLM에 이상 탐지 규칙을 교육합니다. 이 프레임워크는 SPICE 넷리스트의 구조화된 대량 데이터를 지능적으로 파싱하고 분석할 수 있는 기능을 제공합니다. 트로이 안드로이드가 포함된 회로와 포함되지 않은 회로를 구별할 수 있으며, 특정 트로이 안드로이드 구성요소와 영향을 받는 노드를 정확히 식별합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 평가된 아날로그 벤치마크 회로에서 평균 93.32%의 트로이 안드로이드 커버리지와 93.4%의 참 양성율을 기록했습니다. 이러한 결과는 LLM이 아날로그 넷리스트 내의 구문 버그 및 트로이 안드로이드를 탐지하고 위치를 파악하는 데 효과적임을 입증합니다.



### Differentially Private Publication of Electricity Time Series Data in Smart Grids (https://arxiv.org/abs/2408.16017)
- **What's New**: 본 논문에서는 전력 소비 데이터를 보다 안전하게 공개하기 위한 새로운 차별적 프라이버시(differential privacy) 방법인 STPT(Spatio-Temporal Private Timeseries)를 소개합니다. 이 방법은 시공간(spatio-temporal) 특성을 분석하여 에너지 소비 패턴을 효과적으로 포착합니다.

- **Technical Details**: STPT 알고리즘은 RNN(Recurrent Neural Network)을 활용하여 전력 소비의 시공간 패턴을 식별합니다. 또한, 소비 데이터를 기준으로 시간열을 적절히 나누어 DP 메커니즘을 통해 데이터를 세밀하게 보호합니다. 이 과정에서는 저해상도에서 시작해 고해상도로 나아가는 방식으로 소비 트렌드를 파악합니다.

- **Performance Highlights**: STPT는 공공 데이터와 합성 데이터셋을 이용한 광범위한 실험을 통해 기존의 벤치마크에 비해 데이터 유용성을 크게 개선하며, 사용자 프라이버시를 유지하는 동시에 효과적인 데이터 공개를 가능하게 합니다.



### A Tutorial on Brownian Motion for Biostatisticians (https://arxiv.org/abs/2408.16011)
- **What's New**: 이 논문은 Biostatisticians를 위한 확률 이론의 기초적인 확률 과정인 Brownian Motion에 대한 심층적인 탐구를 제공합니다. 기초 정의와 속성에서 시작하여 고급 주제인 Karhunen-Loeve 확장, 반사 원리 및 Lévy의 연속성 모듈에 대해 다룹니다.

- **Technical Details**: Brownian Motion (브라운 운동)은 연속 경로를 가진 확률 과정으로, stationarily independent increments (정지 독립 증가량) 및 Gaussian distribution (가우시안 분포) 속성을 가집니다. 이 논문은 Brownian 경로의 비미분 가능성, 제로 집합의 행동 및 지역 시간의 중요성에 대해 논의합니다.

- **Performance Highlights**: Donsker's theorem과 Blumenthal's 0-1 법칙과 같은 중요한 결과들이 강조되며, 이 결과들은 확률 과정 연구에서의 깊은 의미를 지닙니다. 또한, Brownian Motion의 이론적 기초부터 현대 과학 및 공학에서의 응용에 이르기까지의 포괄적인 이해를 목표로 합니다.



### Novel Methods for Analyzing Cellular Interactions in Deep Learning-Based Image Cytometry: Spatial Interaction Potential and Co-Localization Index (https://arxiv.org/abs/2408.16008)
- **What's New**: 이 연구에서는 디지털 병리학에서 세포 상호작용을 정량화하기 위한 새로운 접근법을 제안합니다. 기존의 전통적인 방법들이 조직 내 세포의 다양성과 이질성에 어려움을 겪고 있는 가운데, Spatial Interaction Potential (SIP)과 Co-Localization Index (CLI)를 도입하여 이러한 한계를 극복하고자 합니다.

- **Technical Details**: SIP는 전기장처럼 세포 간 상호작용의 가능성을 평가하며, CLI는 세포 간 거리 및 동적인 세포 운동을 고려하여 분석합니다. 이 연구는 심층 학습 기반의 분류 확률을 활용하여 세포 상호작용을 총체적으로 분석하고, Cu-Cyto® 자동 이미지 분석 플랫폼을 통해 실행됩니다.

- **Performance Highlights**: SIP와 CLI는 시뮬레이션과 대장암 샘플을 통해 검증되었으며, 실제 생물학적 데이터와 강한 상관관계를 나타냅니다. 이 혁신적인 방법은 세포 상호작용의 이해를 개선하고 디지털 병리학의 다양한 분야에서 활용 가능성이 큽니다.



### Meta-Learning for Federated Face Recognition in Imbalanced Data Regimes (https://arxiv.org/abs/2408.16003)
Comments:
          To appear in the IEEE FLTA 2024 proceedings

- **What's New**: 이 논문은 Federated Face Recognition (FFR) 분야에서 개인화된 Federated Learning (FL)의 해결책을 제시합니다. CelebA 데이터셋을 기준으로 서로 다른 형태의 데이터 이질성을 기반으로 한 세 가지 새로운 데이터 파티션을 소개합니다.

- **Technical Details**: 제안된 방법은 Hessian-Free Model Agnostic Meta-Learning (HF-MAML)입니다. 이 접근법은 전통적인 MAML 알고리즘의 계산 비용을 줄이며, FL 환경에서 효과적으로 작동할 수 있도록 설계되었습니다. 추가적으로, 손실 함수에 embedding regularization 항목을 도입하여 글로벌 모델의 성능을 향상시킵니다.

- **Performance Highlights**: HF-MAML은 CelebA 데이터 파티션에서 기존 FFR 모델들보다 높은 검증 점수를 기록했습니다. 특히, 데이터 이질성이 큰 파티션에서 개선된 검증 점수를 보여주었으며, 공정성 분석에서도 HF-MAML과 주입된 편향 정규화가 클라이언트 평가 점수의 표준 편차를 줄이는 데 기여함을 확인했습니다.




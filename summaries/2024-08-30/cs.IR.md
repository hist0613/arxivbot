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



